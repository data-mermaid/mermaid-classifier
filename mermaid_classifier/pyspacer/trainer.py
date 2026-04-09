"""
MermaidTrainer: a pyspacer ClassifierTrainer with batched calibration
and per-epoch MLflow callbacks.
"""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV, _fit_calibrator
from torch.utils.data import DataLoader

from spacer import config
from spacer.data_classes import ValResults
from spacer.messages import TrainClassifierReturnMsg
from spacer.storage import storage_factory
from spacer.train_classifier import ClassifierTrainer
from spacer.train_utils import calc_acc, evaluate_classifier

from mermaid_classifier.pyspacer.materialized_data import (
    MemmapFeatureDataset,
    cleanup_materialized,
    load_materialized_batches,
    materialize_split,
)
from mermaid_classifier.pyspacer.torch_classifier import (
    PrefetchDataLoader,
    StreamingFeatureDataset,
    TorchMLPClassifier,
)

logger = getLogger(__name__)


def _prefetch_to_cache(labels, cache_dir, max_workers=50):
    """Download remote feature vectors into filesystem cache in parallel."""
    all_locs = set()
    for split in [labels.train, labels.ref, labels.val]:
        for loc in split.keys():
            if loc.is_remote:
                all_locs.add(loc)

    to_fetch = [
        loc for loc in all_locs
        if not Path(cache_dir, loc.key).exists()
    ]
    if not to_fetch:
        return

    # Pre-create cache directories to avoid a race condition in
    # pyspacer's FileSystemStorage.store (os.makedirs without exist_ok).
    dirs = {Path(cache_dir, loc.key).parent for loc in to_fetch}
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Prefetching {len(to_fetch)} feature vectors"
        f" ({len(all_locs) - len(to_fetch)} already cached)"
        f" with {max_workers} workers...")

    def _fetch_one(loc):
        storage = storage_factory(loc.storage_type, loc.bucket_name)
        storage.load(loc.key, filesystem_cache=cache_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_fetch_one, to_fetch))


class MermaidTrainer(ClassifierTrainer):
    """
    ClassifierTrainer subclass with batched calibration, batched accuracy,
    and per-epoch MLflow callbacks.

    Reference data and training data are never held in memory
    simultaneously: ref accuracy is computed by streaming features
    from disk in batches after each epoch's training batches are
    released, and calibration streams ref data in batches too
    (accumulating only scalar prediction scores, not feature vectors).
    """

    def __init__(
        self,
        on_epoch_end: Callable[[dict], None] | None = None,
        class_balancing: bool = False,
        device: str = 'cpu',
        optimizer: str = 'adam',
        learning_rate: float | None = None,
        weight_decay: float = 1e-4,
        hidden_layer_sizes: tuple[int, ...] | None = None,
        minibatch_size: int = 512,
        io_batch_size: int = 10_000,
        io_workers: int = 4,
        feature_cache_dir: str | None = None,
        materialize_data: bool = True,
    ):
        self.minibatch_size = minibatch_size
        self.io_batch_size = io_batch_size
        self.io_workers = io_workers
        self.on_epoch_end = on_epoch_end
        self.class_balancing = class_balancing
        self.device = device
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_layer_sizes = hidden_layer_sizes
        self.feature_cache_dir = feature_cache_dir
        self.materialize_data = materialize_data

    def __call__(self, labels, nbr_epochs, pc_models, clf_type):
        # Set filesystem cache and prefetch feature vectors from S3.
        # This must happen after preprocess_labels (called by
        # spacer_train_classifier before invoking this trainer), because
        # preprocess_labels may create new ImageLabels instances that
        # lose any previously set filesystem_cache.
        if self.feature_cache_dir:
            labels.set_filesystem_cache(self.feature_cache_dir)
            _prefetch_to_cache(labels, self.feature_cache_dir)

        logger.debug(
            f"Unique classes:"
            f" Train + Ref = {len(labels.ref.classes_set)},"
            f" Val = {len(labels.val.classes_set)}")
        logger.debug(
            f"Label count:"
            f" Train = {labels.train.label_count},"
            f" Ref = {labels.ref.label_count},"
            f" Val = {labels.val.label_count},"
            f" Total = {labels.label_count}")
        logger.debug(
            f"Data sets:"
            f" Train = {len(labels.train)} images,"
            f" {labels.train.label_count} labels;"
            f" Ref = {len(labels.ref)} images,"
            f" {labels.ref.label_count} labels")
        logger.debug(
            f"Minibatch size: {self.minibatch_size} labels")
        logger.debug(
            f"IO batch size: {self.io_batch_size} labels")
        logger.debug(
            f"IO workers: {self.io_workers}")

        assert clf_type in config.CLASSIFIER_TYPES

        classes_list = list(labels.ref.classes_set)

        # Compute global class weights from full training set
        if self.class_balancing:
            total = labels.train.label_count
            n_classes = len(classes_list)
            class_weight = {}
            for cls in classes_list:
                count = labels.train.label_count_per_class.get(cls, 0)
                if count > 0:
                    class_weight[cls] = total / (n_classes * count)
                else:
                    class_weight[cls] = 1.0
            logger.debug(
                f"Class balancing enabled."
                f" Weight range:"
                f" {min(class_weight.values()):.4f}"
                f" - {max(class_weight.values()):.4f}")
        else:
            class_weight = None

        label_to_idx = {
            label: idx for idx, label in enumerate(classes_list)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        # Materialization: extract matched features into memmap files
        use_materialized = (
            self.materialize_data and self.feature_cache_dir is not None)
        materialized = {}  # split_name -> (feat_path, label_path, n, dim)
        self._materialization_seconds = None
        materialized_dir = None

        if use_materialized:
            t_mat = time.perf_counter()
            materialized_dir = Path(self.feature_cache_dir, '_materialized')

            # Detect feature_dim from first .npz in train split
            first_key = next(iter(labels.train.keys()))
            first_npz = np.load(
                str(Path(self.feature_cache_dir, first_key.key)))
            feature_dim = first_npz['feat'].shape[1]

            for split_name, split_labels in [
                ('train', labels.train),
                ('ref', labels.ref),
                ('val', labels.val),
            ]:
                feat_path, label_path, n = materialize_split(
                    split_labels, label_to_idx,
                    self.feature_cache_dir, materialized_dir,
                    split_name, feature_dim=feature_dim,
                    max_workers=self.io_workers)
                materialized[split_name] = (
                    feat_path, label_path, n, feature_dim)

            self._materialization_seconds = time.perf_counter() - t_mat
            logger.info(
                f"Materialization completed in"
                f" {self._materialization_seconds:.1f}s")

        try:
            return self._train_and_evaluate(
                labels, nbr_epochs, pc_models, clf_type,
                classes_list, class_weight, label_to_idx, idx_to_label,
                use_materialized, materialized)
        finally:
            if materialized_dir is not None:
                cleanup_materialized(materialized_dir)

    def _train_and_evaluate(
        self, labels, nbr_epochs, pc_models, clf_type,
        classes_list, class_weight, label_to_idx, idx_to_label,
        use_materialized, materialized,
    ):
        # Initialize classifier and train
        with config.log_entry_and_exit("training using MLP"):
            # Larger networks + lower LR for datasets >= 50K labels
            large_dataset = labels.train.label_count >= 50000
            hls = self.hidden_layer_sizes or (
                (200, 100) if large_dataset else (100,))
            lr = self.learning_rate or (
                1e-4 if large_dataset else 1e-3)

            clf = TorchMLPClassifier(
                hidden_layer_sizes=hls,
                learning_rate_init=lr,
                class_weight=class_weight,
                optimizer=self.optimizer,
                weight_decay=self.weight_decay,
                device=self.device)

            use_cuda = clf.device.type != 'cpu'
            ref_accs = []
            t0 = time.time()

            effective_io = min(self.io_batch_size, self.minibatch_size)
            accum_steps = max(1, self.minibatch_size // effective_io)

            for epoch in range(nbr_epochs):
                t_epoch = time.perf_counter()
                epoch_loss_start = len(clf.loss_curve_)

                if use_materialized:
                    feat_path, label_path, n, dim = materialized['train']
                    dataset = MemmapFeatureDataset(
                        feat_path, label_path, n, dim,
                        batch_size=effective_io,
                        random_seed=epoch)
                else:
                    dataset = StreamingFeatureDataset(
                        labels.train, label_to_idx,
                        batch_size=effective_io,
                        random_seed=epoch,
                        num_workers=self.io_workers)

                dataloader = DataLoader(
                    dataset, batch_size=None,
                    num_workers=0, pin_memory=use_cuda)
                prefetched = PrefetchDataLoader(dataloader)

                accumulated = 0
                accumulated_loss = 0.0

                t_data_total = 0.0
                t_gpu_total = 0.0
                t_batch_start = time.perf_counter()

                for x_batch, y_batch in prefetched:
                    t_data_total += time.perf_counter() - t_batch_start

                    if clf._model is None:
                        clf.init_model(x_batch.shape[1], classes_list)
                        clf._optimizer.zero_grad()

                    t_gpu_start = time.perf_counter()
                    loss_val = clf.forward_backward(
                        x_batch, y_batch,
                        loss_scale=1.0 / accum_steps)
                    accumulated_loss += loss_val
                    accumulated += 1

                    if accumulated == accum_steps:
                        clf.optimizer_step()
                        clf.loss_curve_.append(
                            accumulated_loss / accumulated)
                        accumulated = 0
                        accumulated_loss = 0.0

                    t_gpu_total += time.perf_counter() - t_gpu_start
                    t_batch_start = time.perf_counter()

                if accumulated > 0:
                    clf.optimizer_step()
                    clf.loss_curve_.append(
                        accumulated_loss / accumulated)

                # Ref accuracy
                t_ref_start = time.perf_counter()
                ref_batches = self._ref_batch_iter(
                    use_materialized, materialized, labels, idx_to_label)
                ref_accs.append(self._calc_acc(clf, ref_batches))
                t_ref_seconds = time.perf_counter() - t_ref_start

                logger.debug(f"Epoch {epoch}, acc: {ref_accs[-1]}")

                if self.on_epoch_end is not None:
                    epoch_losses = clf.loss_curve_[epoch_loss_start:]
                    epoch_seconds = time.perf_counter() - t_epoch
                    self.on_epoch_end({
                        "epoch": epoch,
                        "ref_accuracy": ref_accs[-1],
                        "training_loss": (
                            float(np.mean(epoch_losses))
                            if epoch_losses else None),
                        "cumulative_seconds": time.time() - t0,
                        "epoch_seconds": epoch_seconds,
                        "data_load_seconds": t_data_total,
                        "gpu_compute_seconds": t_gpu_total,
                        "ref_accuracy_seconds": t_ref_seconds,
                    })

        # Calibration: stream ref data in batches — avoids loading full feature
        # vectors into memory. Only scalar prediction scores accumulate
        # (O(N * K) instead of O(N * 4096)).
        with config.log_entry_and_exit("calibration"):
            ref_batches = self._ref_batch_iter(
                use_materialized, materialized, labels, idx_to_label)
            clf_calibrated = self._calibrate(clf, ref_batches)

        classes = clf_calibrated.classes_.tolist()

        # Evaluate new classifier on validation set
        if use_materialized:
            feat_path, label_path, n, dim = materialized['val']
            val_batches = load_materialized_batches(
                feat_path, label_path, n, dim,
                self.minibatch_size, idx_to_label)
        else:
            val_batches = labels.val.load_data_in_batches(
                batch_size=self.minibatch_size)
        val_gts, val_ests, val_scores = self._evaluate(
            clf_calibrated, val_batches)

        # Evaluate previous classifiers on validation set
        pc_accs = []
        for pc_model in pc_models:
            pc_gts, pc_ests, _ = evaluate_classifier(pc_model, labels.val)
            pc_accs.append(calc_acc(pc_gts, pc_ests))

        val_results = ValResults(
            scores=val_scores,
            gt=[classes.index(member) for member in val_gts],
            est=[classes.index(member) for member in val_ests],
            classes=classes,
        )

        return_message = TrainClassifierReturnMsg(
            acc=calc_acc(val_gts, val_ests),
            pc_accs=pc_accs,
            ref_accs=ref_accs,
            runtime=time.time() - t0,
        )

        return clf_calibrated, val_results, return_message

    def _ref_batch_iter(self, use_materialized, materialized,
                        labels, idx_to_label):
        """Return a batch iterator for the ref split."""
        if use_materialized:
            feat_path, label_path, n, dim = materialized['ref']
            return load_materialized_batches(
                feat_path, label_path, n, dim,
                self.minibatch_size, idx_to_label)
        return labels.ref.load_data_in_batches(
            batch_size=self.minibatch_size)

    def _calc_acc(self, clf, batch_iter) -> float:
        """Compute accuracy from a (features, labels) batch iterator.

        Only predictions and ground-truth labels (tiny lists of strings)
        accumulate — not feature vectors. Memory use is O(batch_size).
        """
        gt, pred = [], []
        for x, y in batch_iter:
            pred.extend(clf.predict(x))
            gt.extend(y)
        return calc_acc(gt, pred)

    def _calibrate(self, clf, batch_iter) -> CalibratedClassifierCV:
        """
        Platt calibration from a (features, labels) batch iterator.

        Collects only scalar prediction scores (N x K) rather than
        feature vectors (N x 4096). Fits sigmoid calibrators identically
        to CalibratedClassifierCV(cv='prefit').fit().

        Returns a valid CalibratedClassifierCV instance compatible with
        spacer.storage.store_classifier() and ClassifierUnpickler.load().
        """
        all_preds, all_y = [], []

        for x_batch, y_batch in batch_iter:
            x_arr = np.asarray(x_batch)
            y_arr = np.asarray(y_batch)
            all_preds.append(clf.predict_proba(x_arr))
            all_y.append(y_arr)

        predictions = np.vstack(all_preds)
        y = np.concatenate(all_y)

        # _fit_calibrator is a private sklearn API used here for memory
        # efficiency — it calibrates from pre-computed predictions instead of
        # re-running predict_proba on the full dataset at once. If an sklearn
        # upgrade breaks this, it will fail at training time (not inference).
        # The serialized model format is identical to the public .fit() API.
        calibrated_inner = _fit_calibrator(
            clf, predictions, y, clf.classes_, method="sigmoid"
        )

        wrapper = CalibratedClassifierCV(clf, cv="prefit")
        wrapper.calibrated_classifiers_ = [calibrated_inner]
        wrapper.classes_ = clf.classes_

        return wrapper

    @staticmethod
    def _evaluate(clf_calibrated, batch_iter):
        """Evaluate a calibrated classifier on a (features, labels) batch iterator."""
        scores, gts, ests = [], [], []

        for x_batch, y_batch in batch_iter:
            scores.extend(
                clf_calibrated.predict_proba(x_batch).max(axis=1).tolist())
            ests.extend(clf_calibrated.predict(x_batch))
            gts.extend(y_batch)

        if not gts:
            raise ValueError(
                "Evaluation produced no ground-truth labels;"
                " validation split may be empty.")
        return gts, ests, scores

    def serialize(self) -> dict:
        data = super().serialize()
        data['minibatch_size'] = self.minibatch_size
        data['io_batch_size'] = self.io_batch_size
        data['io_workers'] = self.io_workers
        if self.class_balancing:
            data['class_balancing'] = self.class_balancing
        data['device'] = self.device
        data['optimizer'] = self.optimizer
        if self.learning_rate is not None:
            data['learning_rate'] = self.learning_rate
        data['weight_decay'] = self.weight_decay
        if self.hidden_layer_sizes is not None:
            data['hidden_layer_sizes'] = self.hidden_layer_sizes
        data['materialize_data'] = self.materialize_data
        # on_epoch_end and feature_cache_dir are runtime concerns; excluded
        return data
