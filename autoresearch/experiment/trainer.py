"""
Autoresearch experiment: trainer.

Seeded from mermaid_classifier/pyspacer/trainer.py.
The agent may freely modify this file: change the optimizer, add
learning rate schedules, gradient clipping, warmup, modify the
epoch logic, etc.

The trainer must be a ClassifierTrainer subclass with __call__
returning (clf_calibrated, val_results, return_message).
"""

import copy
import time
from collections.abc import Callable
from logging import getLogger

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, _fit_calibrator
from sklearn.metrics import log_loss as sklearn_log_loss

from spacer import config
from spacer.data_classes import ImageLabels, ValResults
from spacer.messages import TrainClassifierReturnMsg
from spacer.train_classifier import ClassifierTrainer
from spacer.train_utils import calc_acc, evaluate_classifier

from classifier import ExperimentMLPClassifier

logger = getLogger(__name__)


class ExperimentTrainer(ClassifierTrainer):
    """
    ClassifierTrainer for autoresearch experiments.

    Seeded from MermaidTrainer. The agent can modify the training loop,
    optimizer, learning rate schedule, classifier construction, etc.
    """

    def __init__(
        self,
        batch_size: int,
        on_epoch_end: Callable[[dict], None] | None = None,
        class_weight: dict | None = None,
        hidden_layer_sizes: tuple[int, ...] | None = None,
        learning_rate_init: float | None = None,
        early_stopping_patience: int | None = None,
        random_state: int = 0,
        dropout: float = 0.0,
    ):
        if (hidden_layer_sizes is None) != (learning_rate_init is None):
            raise ValueError(
                "hidden_layer_sizes and learning_rate_init must be"
                " supplied together (or both omitted to use the"
                " label-count heuristic)."
            )
        if (early_stopping_patience is not None
                and early_stopping_patience < 1):
            raise ValueError(
                f"early_stopping_patience must be >= 1 or None,"
                f" got {early_stopping_patience!r}"
            )
        self.batch_size = batch_size
        self.on_epoch_end = on_epoch_end
        self.class_weight = class_weight
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.dropout = float(dropout)
        self._early_stop_info: dict | None = None

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
            f"Batch size: {self.batch_size} labels")

        assert clf_type in config.CLASSIFIER_TYPES

        classes_list = list(labels.ref.classes_set)

        with config.log_entry_and_exit("training using " + clf_type):
            if clf_type == 'MLP':
                if self.hidden_layer_sizes is not None:
                    hls = self.hidden_layer_sizes
                    lr = self.learning_rate_init
                elif labels.train.label_count >= 50000:
                    hls, lr = (200, 100), 1e-4
                else:
                    hls, lr = (100,), 1e-3
                clf = ExperimentMLPClassifier(
                    hidden_layer_sizes=hls,
                    learning_rate_init=lr,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    dropout=self.dropout,
                )
            else:
                from sklearn.linear_model import SGDClassifier
                clf = SGDClassifier(
                    loss='log_loss', average=True,
                    random_state=self.random_state)

            ref_accs = []
            t0 = time.time()

            best_val_loss: float = float('inf')
            best_clf_snapshot = None
            best_epoch_idx: int | None = None
            epochs_since_best: int = 0
            stop_reason: str = 'budget_exhausted'

            for epoch in range(nbr_epochs):
                for x, y in labels.train.load_data_in_batches(
                    batch_size=self.batch_size,
                    random_seed=epoch,
                ):
                    clf.partial_fit(x, y, classes=classes_list)

                ref_accs.append(
                    self._calc_acc_batched(clf, labels.ref))
                logger.debug(f"Epoch {epoch}, acc: {ref_accs[-1]}")

                val_acc, val_loss = self._calc_acc_and_log_loss_batched(
                    clf, labels.val, classes_list)
                logger.debug(
                    f"Epoch {epoch}, val_acc: {val_acc}, val_loss: {val_loss}")

                if self.early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch_idx = epoch
                        best_clf_snapshot = copy.deepcopy(clf)
                        epochs_since_best = 0
                    else:
                        epochs_since_best += 1

                will_stop_after_this = (
                    epoch == nbr_epochs - 1
                    or (self.early_stopping_patience is not None
                        and epochs_since_best
                        >= self.early_stopping_patience)
                )

                if self.on_epoch_end is not None:
                    loss_curve = getattr(clf, 'loss_curve_', [None])
                    cb_metrics: dict = {
                        "epoch": epoch,
                        "ref_accuracy": ref_accs[-1],
                        "val_accuracy": val_acc,
                        "val_loss": val_loss,
                        "training_loss":
                            loss_curve[-1] if loss_curve else None,
                        "cumulative_seconds": time.time() - t0,
                    }
                    if will_stop_after_this:
                        early_stopped = (
                            self.early_stopping_patience is not None
                            and epochs_since_best
                            >= self.early_stopping_patience
                        )
                        cb_metrics["final_epoch"] = epoch + 1
                        cb_metrics["early_stopped"] = early_stopped
                        if best_epoch_idx is not None:
                            cb_metrics["best_val_epoch"] = (
                                best_epoch_idx + 1)
                            cb_metrics["best_val_loss"] = best_val_loss
                    self.on_epoch_end(cb_metrics)

                if (self.early_stopping_patience is not None
                        and epochs_since_best
                        >= self.early_stopping_patience):
                    stop_reason = 'early_stopping'
                    logger.info(
                        f"Early stopping at epoch {epoch + 1}:"
                        f" val_loss has not improved for"
                        f" {self.early_stopping_patience} consecutive"
                        f" epochs. Best was epoch"
                        f" {(best_epoch_idx or 0) + 1}"
                        f" (val_loss={best_val_loss:.4f})."
                    )
                    break

            if (self.early_stopping_patience is not None
                    and best_clf_snapshot is not None
                    and best_epoch_idx != epoch):
                logger.info(
                    f"Restoring classifier from epoch"
                    f" {(best_epoch_idx or 0) + 1}"
                    f" (val_loss={best_val_loss:.4f}); latest epoch"
                    f" was {epochs_since_best} epochs past best."
                )
                clf = best_clf_snapshot
            self._early_stop_info = {
                'enabled': self.early_stopping_patience is not None,
                'patience': self.early_stopping_patience,
                'stop_reason': stop_reason,
                'final_epoch': epoch + 1,
                'best_val_epoch': (
                    best_epoch_idx + 1
                    if best_epoch_idx is not None else None),
                'best_val_loss': (
                    best_val_loss
                    if best_val_loss != float('inf') else None),
            }

        with config.log_entry_and_exit("calibration"):
            clf_calibrated = self._calibrate_in_batches(clf, labels.ref)

        classes = clf_calibrated.classes_.tolist()

        val_gts, val_ests, val_scores = evaluate_classifier(
            clf_calibrated, labels.val)

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
        gt, pred = [], []
        for x, y in labels.load_data_in_batches(batch_size=self.batch_size):
            pred.extend(clf.predict(x))
            gt.extend(y)
        return calc_acc(gt, pred)

    def _calc_acc_and_log_loss_batched(
        self,
        clf,
        labels: ImageLabels,
        classes_list: list,
    ) -> tuple[float, float]:
        gt: list = []
        all_proba: list = []
        for x, y in labels.load_data_in_batches(batch_size=self.batch_size):
            all_proba.append(clf.predict_proba(x))
            gt.extend(y)
        proba = np.vstack(all_proba)
        clf_classes = list(clf.classes_)
        pred = [clf_classes[i] for i in proba.argmax(axis=1)]
        acc = calc_acc(gt, pred)
        loss = float(sklearn_log_loss(gt, proba, labels=clf_classes))
        return acc, loss

    def _calibrate_in_batches(
        self,
        clf,
        ref_labels: ImageLabels,
    ) -> CalibratedClassifierCV:
        all_preds, all_y = [], []

        for x_batch, y_batch in ref_labels.load_data_in_batches(
                batch_size=self.batch_size):
            x_arr = np.array(x_batch)
            y_arr = np.array(y_batch)

            if hasattr(clf, 'decision_function'):
                preds = clf.decision_function(x_arr)
            else:
                preds = clf.predict_proba(x_arr)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)

            all_preds.append(preds)
            all_y.append(y_arr)

        predictions = np.vstack(all_preds)
        y = np.concatenate(all_y)

        calibrated_inner = _fit_calibrator(
            clf, predictions, y, clf.classes_, method="sigmoid"
        )

        wrapper = CalibratedClassifierCV(clf, cv="prefit")
        wrapper.calibrated_classifiers_ = [calibrated_inner]
        wrapper.classes_ = clf.classes_

        return wrapper

    def serialize(self) -> dict:
        data = super().serialize()
        data['batch_size'] = self.batch_size
        return data
