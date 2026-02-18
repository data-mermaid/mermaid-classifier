"""
Local training loop with epoch-level callbacks.

Reimplements the small training loop from PySpacer's train_utils.train()
and the orchestration from tasks.train_classifier() / MiniBatchTrainer,
adding an on_epoch_end callback for real-time metric logging.
"""

import contextlib
import tempfile
import time
from collections.abc import Callable
from logging import getLogger

from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

from spacer import config
from spacer.data_classes import ImageLabels, ValResults
from spacer.messages import TrainClassifierMsg, TrainClassifierReturnMsg
from spacer.storage import load_classifier, store_classifier
from spacer.train_utils import calc_acc, evaluate_classifier

logger = getLogger(__name__)


def train_with_callbacks(
    train_labels: ImageLabels,
    ref_labels: ImageLabels,
    nbr_epochs: int,
    clf_type: str,
    on_epoch_end: Callable[[dict], None] | None = None,
) -> tuple[CalibratedClassifierCV, list[float]]:
    """
    Train a classifier with per-epoch callbacks.

    Mirrors spacer.train_utils.train() with the addition of an
    on_epoch_end callback that receives a dict of metrics after
    each epoch completes.
    """
    logger.debug(
        f"Data sets:"
        f" Train = {len(train_labels)} images,"
        f" {train_labels.label_count} labels;"
        f" Ref = {len(ref_labels)} images,"
        f" {ref_labels.label_count} labels")
    logger.debug(
        f"Mini-batch size: {config.TRAINING_BATCH_LABEL_COUNT} labels")

    classes_list = list(ref_labels.classes_set)

    # Load reference data (must hold in memory for calibration)
    with config.log_entry_and_exit("loading of reference data"):
        refx, refy = ref_labels.load_all_data()

    # Initialize classifier
    with config.log_entry_and_exit("training using " + clf_type):
        if clf_type == 'MLP':
            if train_labels.label_count >= 50000:
                hls, lr = (200, 100), 1e-4
            else:
                hls, lr = (100,), 1e-3
            clf = MLPClassifier(hidden_layer_sizes=hls, learning_rate_init=lr)
        else:
            clf = SGDClassifier(loss='log_loss', average=True, random_state=0)

        ref_acc = []
        t0 = time.time()

        for epoch in range(nbr_epochs):
            for x, y in train_labels.load_data_in_batches(
                batch_size=config.TRAINING_BATCH_LABEL_COUNT,
                random_seed=epoch,
            ):
                clf.partial_fit(x, y, classes=classes_list)

            ref_acc.append(calc_acc(refy, clf.predict(refx)))
            logger.debug(f"Epoch {epoch}, acc: {ref_acc[-1]}")

            if on_epoch_end is not None:
                loss_curve = getattr(clf, 'loss_curve_', [None])
                on_epoch_end({
                    "epoch": epoch,
                    "ref_accuracy": ref_acc[-1],
                    "training_loss": loss_curve[-1] if loss_curve else None,
                    "cumulative_seconds": time.time() - t0,
                })

    with config.log_entry_and_exit("calibration"):
        clf_calibrated = CalibratedClassifierCV(clf, cv="prefit")
        clf_calibrated.fit(refx, refy)

    return clf_calibrated, ref_acc


def train_classifier_with_callbacks(
    msg: TrainClassifierMsg,
    on_epoch_end: Callable[[dict], None] | None = None,
) -> TrainClassifierReturnMsg:
    """
    Train a classifier with epoch-level callbacks.

    Mirrors spacer.tasks.train_classifier() and
    spacer.train_classifier.MiniBatchTrainer.__call__(), wiring through
    the on_epoch_end callback to the inner training loop.
    """
    labels = msg.labels

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

    # Feature caching wrapper (mirrors spacer.tasks.train_classifier)
    @contextlib.contextmanager
    def wrapper():
        if (
            labels.has_remote_data
            and msg.feature_cache_dir != msg.FeatureCache.DISABLED
        ):
            if msg.feature_cache_dir == msg.FeatureCache.AUTO:
                feature_cache_dir = None
            else:
                feature_cache_dir = msg.feature_cache_dir

            with tempfile.TemporaryDirectory(
                    dir=feature_cache_dir) as local_feature_dir:
                labels.set_filesystem_cache(local_feature_dir)
                yield
        else:
            yield

    with wrapper():
        # Train (mirrors MiniBatchTrainer.__call__)
        assert msg.clf_type in config.CLASSIFIER_TYPES

        t0 = time.time()
        clf, ref_accs = train_with_callbacks(
            labels.train, labels.ref, msg.nbr_epochs, msg.clf_type,
            on_epoch_end=on_epoch_end,
        )
        classes = clf.classes_.tolist()

        # Evaluate new classifier on validation set
        val_gts, val_ests, val_scores = evaluate_classifier(
            clf, labels.val)

        # Evaluate previous classifiers on validation set
        pc_accs = []
        for pc_model in [load_classifier(loc)
                         for loc in msg.previous_model_locs]:
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

    with config.log_entry_and_exit('storing classifier and val res'):
        store_classifier(msg.model_loc, clf)
        val_results.store(msg.valresult_loc)

    return return_message
