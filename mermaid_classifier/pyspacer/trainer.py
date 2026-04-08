"""
MermaidTrainer: a pyspacer ClassifierTrainer with batched calibration
and per-epoch MLflow callbacks.
"""

import time
from collections.abc import Callable
from logging import getLogger

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV, _fit_calibrator
from torch.utils.data import DataLoader

from spacer import config
from spacer.data_classes import ImageLabels, ValResults
from spacer.messages import TrainClassifierReturnMsg
from spacer.train_classifier import ClassifierTrainer
from spacer.train_utils import calc_acc, evaluate_classifier

from mermaid_classifier.pyspacer.torch_classifier import (
    FeatureDataset,
    TorchMLPClassifier,
)

logger = getLogger(__name__)


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
        io_batch_size: int,
        on_epoch_end: Callable[[dict], None] | None = None,
        class_balancing: bool = False,
        device: str = 'cpu',
        num_workers: int = 0,
        optimizer: str = 'adam',
        learning_rate: float | None = None,
        weight_decay: float = 1e-4,
        hidden_layer_sizes: tuple[int, ...] | None = None,
        minibatch_size: int = 512,
    ):
        self.io_batch_size = io_batch_size
        self.minibatch_size = minibatch_size
        self.on_epoch_end = on_epoch_end
        self.class_balancing = class_balancing
        self.device = device
        self.num_workers = num_workers
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_layer_sizes = hidden_layer_sizes

    def __call__(self, labels, nbr_epochs, pc_models, clf_type):
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
            f"IO batch size: {self.io_batch_size} labels")
        logger.debug(
            f"Minibatch size: {self.minibatch_size} labels")

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

            # Materialize training data into a Dataset for DataLoader
            label_to_idx = {
                label: idx for idx, label in enumerate(classes_list)}
            dataset = FeatureDataset(
                labels.train, label_to_idx, self.io_batch_size)
            clf.init_model(dataset.X.shape[1], classes_list)

            use_cuda = clf.device.type != 'cpu'
            dataloader = DataLoader(
                dataset, batch_size=self.minibatch_size, shuffle=True,
                num_workers=self.num_workers,
                pin_memory=use_cuda,
                generator=torch.Generator().manual_seed(0))

            ref_accs = []
            t0 = time.time()

            for epoch in range(nbr_epochs):
                epoch_loss_start = len(clf.loss_curve_)
                for x_batch, y_batch in dataloader:
                    clf.train_step(x_batch, y_batch)

                # Ref accuracy: stream ref features from disk in batches.
                # Only predictions (tiny) accumulate, not feature arrays.
                ref_accs.append(
                    self._calc_acc_batched(clf, labels.ref))
                logger.debug(f"Epoch {epoch}, acc: {ref_accs[-1]}")

                if self.on_epoch_end is not None:
                    epoch_losses = clf.loss_curve_[epoch_loss_start:]
                    self.on_epoch_end({
                        "epoch": epoch,
                        "ref_accuracy": ref_accs[-1],
                        "training_loss": (
                            float(np.mean(epoch_losses))
                            if epoch_losses else None),
                        "cumulative_seconds": time.time() - t0,
                    })

        # Calibration: stream ref data in batches — avoids loading full feature
        # vectors into memory. Only scalar prediction scores accumulate
        # (O(N * K) instead of O(N * 4096)).
        with config.log_entry_and_exit("calibration"):
            clf_calibrated = self._calibrate_in_batches(clf, labels.ref)

        classes = clf_calibrated.classes_.tolist()

        # Evaluate new classifier on validation set
        val_gts, val_ests, val_scores = evaluate_classifier(
            clf_calibrated, labels.val)

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

    def _calc_acc_batched(self, clf, labels: ImageLabels) -> float:
        """Compute accuracy by streaming features from disk in batches,
        avoiding loading the full dataset into memory.

        Only predictions and ground-truth labels (tiny lists of strings)
        accumulate — not feature vectors. Memory use is O(io_batch_size)
        instead of O(dataset_size).
        """
        gt, pred = [], []
        for x, y in labels.load_data_in_batches(batch_size=self.io_batch_size):
            pred.extend(clf.predict(x))
            gt.extend(y)
        return calc_acc(gt, pred)

    def _calibrate_in_batches(
        self,
        clf,
        ref_labels: ImageLabels,
    ) -> CalibratedClassifierCV:
        """
        Platt calibration without loading full feature vectors into memory.

        Streams ref data in batches, collecting only scalar prediction scores
        (N x K) rather than feature vectors (N x 4096). Fits sigmoid
        calibrators identically to CalibratedClassifierCV(cv='prefit').fit().

        Returns a valid CalibratedClassifierCV instance compatible with
        spacer.storage.store_classifier() and ClassifierUnpickler.load().
        """
        all_preds, all_y = [], []

        for x_batch, y_batch in ref_labels.load_data_in_batches(
                batch_size=self.io_batch_size):
            x_arr = np.array(x_batch)
            y_arr = np.array(y_batch)

            preds = clf.predict_proba(x_arr)

            all_preds.append(preds)
            all_y.append(y_arr)

        predictions = np.vstack(all_preds)  # (N, K)
        y = np.concatenate(all_y)           # (N,)

        # _fit_calibrator is a private sklearn API used here for memory
        # efficiency — it calibrates from pre-computed predictions instead of
        # re-running predict_proba on the full dataset at once. If an sklearn
        # upgrade breaks this, it will fail at training time (not inference).
        # The serialized model format is identical to the public .fit() API.
        calibrated_inner = _fit_calibrator(
            clf, predictions, y, clf.classes_, method="sigmoid"
        )

        # Construct CalibratedClassifierCV without calling .fit().
        # Sets all attributes that spacer/storage.py validates:
        # - isinstance check: real CalibratedClassifierCV instance
        # - clf.cv == 'prefit'
        # - hasattr(clf, 'calibrated_classifiers_')
        # - clf.estimator, clf.ensemble, clf.n_jobs (set by __init__)
        wrapper = CalibratedClassifierCV(clf, cv="prefit")
        wrapper.calibrated_classifiers_ = [calibrated_inner]
        wrapper.classes_ = clf.classes_

        return wrapper

    def serialize(self) -> dict:
        data = super().serialize()
        data['io_batch_size'] = self.io_batch_size
        data['minibatch_size'] = self.minibatch_size
        if self.class_balancing:
            data['class_balancing'] = self.class_balancing
        data['device'] = self.device
        data['num_workers'] = self.num_workers
        data['optimizer'] = self.optimizer
        if self.learning_rate is not None:
            data['learning_rate'] = self.learning_rate
        data['weight_decay'] = self.weight_decay
        if self.hidden_layer_sizes is not None:
            data['hidden_layer_sizes'] = self.hidden_layer_sizes
        # on_epoch_end is not JSON-serializable; excluded
        return data
