"""
MermaidTrainer: a pyspacer ClassifierTrainer with batched calibration
and per-epoch MLflow callbacks.
"""

import copy
import time
from collections.abc import Callable
from logging import getLogger

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, _fit_calibrator
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss as sklearn_log_loss

from spacer import config
from spacer.data_classes import ImageLabels, ValResults
from spacer.messages import TrainClassifierReturnMsg
from spacer.train_classifier import ClassifierTrainer
from spacer.train_utils import calc_acc, evaluate_classifier

from mermaid_classifier.pyspacer.torch_classifier import TorchMLPClassifier

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
        batch_size: int,
        on_epoch_end: Callable[[dict], None] | None = None,
        class_weight: dict | None = None,
        early_stopping_patience: int | None = None,
    ):
        if (early_stopping_patience is not None
                and early_stopping_patience < 1):
            raise ValueError(
                f"early_stopping_patience must be >= 1 or None,"
                f" got {early_stopping_patience!r}"
            )
        self.batch_size = batch_size
        self.on_epoch_end = on_epoch_end
        # Optional per-class loss weighting passed through to the
        # underlying TorchMLPClassifier. Only used for clf_type='MLP'.
        # The SGDClassifier branch ignores this (sklearn SGD has its own
        # class_weight kwarg, but plumbing it would expand scope).
        self.class_weight = class_weight
        # Early stopping. None disables. When set: each epoch the val
        # loss is computed (already logged to MLflow); a deepcopy of
        # clf is taken on each new minimum; if val_loss fails to
        # improve for `patience` consecutive epochs, training breaks
        # and clf is restored to its best-val_loss state. The restored
        # clf is what gets calibrated and returned.
        self.early_stopping_patience = early_stopping_patience
        # Populated by __call__; readable by the runner for MLflow
        # logging. Pre-initialized so the runner never hits an
        # AttributeError when patience is None.
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
            f"Data sets:"
            f" Train = {len(labels.train)} images,"
            f" {labels.train.label_count} labels;"
            f" Ref = {len(labels.ref)} images,"
            f" {labels.ref.label_count} labels")
        logger.debug(
            f"Batch size: {self.batch_size} labels")

        assert clf_type in config.CLASSIFIER_TYPES

        classes_list = list(labels.ref.classes_set)

        # Initialize classifier and train
        with config.log_entry_and_exit("training using " + clf_type):
            if clf_type == 'MLP':
                # Production MLP architecture from the hidden-layer
                # experiments (see docs/hidden-layer-experiments.md).
                # random_state=0 keeps weight init deterministic across
                # runs.
                clf = TorchMLPClassifier(
                    hidden_layer_sizes=(500, 300, 100),
                    learning_rate_init=1e-4,
                    class_weight=self.class_weight,
                    random_state=0,
                )
            else:
                clf = SGDClassifier(
                    loss='log_loss', average=True,
                    random_state=0)

            ref_accs = []
            t0 = time.time()

            # Early-stopping bookkeeping. All three remain None when
            # early_stopping_patience is None, so the no-ES path adds
            # zero overhead.
            best_val_loss: float = float('inf')
            best_clf_snapshot = None
            best_epoch_idx: int | None = None
            epochs_since_best: int = 0
            stop_reason: str = 'budget_exhausted'

            for epoch in range(nbr_epochs):
                # Training: load batches from disk, partial_fit, then
                # x and y go out of scope at end of loop body.
                for x, y in labels.train.load_data_in_batches(
                    batch_size=self.batch_size,
                    random_seed=epoch,
                ):
                    clf.partial_fit(x, y, classes=classes_list)

                # Ref accuracy: stream ref features from disk in batches.
                # Only predictions (tiny) accumulate, not feature arrays.
                # Training batch data is no longer in memory at this point.
                ref_accs.append(
                    self._calc_acc_batched(clf, labels.ref))
                logger.debug(f"Epoch {epoch}, acc: {ref_accs[-1]}")

                # Validation accuracy + log_loss: streamed in batches,
                # same pattern as ref_accuracy. These give the canonical
                # overfitting signal -- if training_loss continues to
                # drop while val_loss rises, the model is starting to
                # memorize. val and ref are independent held-out splits,
                # so both are needed for a robust signal.
                #
                # Note: clf here is the uncalibrated MLP head (Platt
                # calibration runs after all epochs). predict_proba on
                # the uncalibrated head is still meaningful for log_loss
                # trend analysis -- we are watching the *change* across
                # epochs, not the absolute value.
                val_acc, val_loss = self._calc_acc_and_log_loss_batched(
                    clf, labels.val, classes_list)
                logger.debug(
                    f"Epoch {epoch}, val_acc: {val_acc}, val_loss: {val_loss}")

                # Early-stopping bookkeeping (no-op if patience is None).
                if self.early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch_idx = epoch
                        # Snapshot the current clf state so we can
                        # restore it if a later epoch overfits. deepcopy
                        # is the safe path for sklearn-compatible
                        # estimators with arbitrary internal state
                        # (TorchMLPClassifier, SGDClassifier, etc.).
                        best_clf_snapshot = copy.deepcopy(clf)
                        epochs_since_best = 0
                    else:
                        epochs_since_best += 1

                # Determine if this is the final epoch we will run.
                # Used to decorate the callback with stop-summary fields.
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
                        # One-shot summary fields fire only on the
                        # final epoch. The runner uses these to write
                        # MLflow tags / scalar metrics; absence on
                        # non-final epochs is intentional.
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

            # Restore the best-val_loss classifier when early stopping
            # is active. Done unconditionally (not just on early-stop
            # break) so a run that completed its full epoch budget
            # still uses the best-val_loss snapshot rather than the
            # last-epoch one. When patience is None we never took a
            # snapshot, and clf is whatever the last epoch produced.
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
            # Stash a small summary on self for runner-side logging
            # (read by MLflowTrainingRunner._log_early_stop_info).
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
        accumulate — not feature vectors. Memory use is O(batch_size)
        instead of O(dataset_size).
        """
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
        """Compute accuracy AND log_loss in a single streaming pass.

        Used for per-epoch validation metrics. The probability matrix is
        large (N x K floats), so we accumulate it as a list of small
        per-batch matrices and stack at the end -- still O(N x K)
        memory but avoids the per-row Python overhead of accumulating
        a giant Python list.

        Returns (accuracy, log_loss). The log_loss is computed against
        the full class set so that classes absent from this particular
        eval set still register as zero-probability columns; matches
        the convention used by sklearn.metrics.log_loss(labels=...).
        """
        gt: list = []
        all_proba: list = []
        for x, y in labels.load_data_in_batches(batch_size=self.batch_size):
            all_proba.append(clf.predict_proba(x))
            gt.extend(y)
        proba = np.vstack(all_proba)
        # Argmax over class axis gives the predicted class index in
        # clf.classes_ order; convert back to class label for calc_acc.
        clf_classes = list(clf.classes_)
        pred = [clf_classes[i] for i in proba.argmax(axis=1)]
        acc = calc_acc(gt, pred)
        # log_loss with explicit labels= ensures column ordering matches
        # the proba matrix even if some classes are absent from gt.
        loss = float(sklearn_log_loss(gt, proba, labels=clf_classes))
        return acc, loss

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
                batch_size=self.batch_size):
            x_arr = np.array(x_batch)
            y_arr = np.array(y_batch)

            # Use the same response method sklearn uses internally:
            # decision_function (preferred) or predict_proba (fallback).
            # See CalibratedClassifierCV.fit() which calls
            # _get_response_values with ["decision_function", "predict_proba"].
            if hasattr(clf, 'decision_function'):
                preds = clf.decision_function(x_arr)
            else:
                preds = clf.predict_proba(x_arr)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)

            all_preds.append(preds)
            all_y.append(y_arr)

        predictions = np.vstack(all_preds)  # (N, K) or (N, 1)
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
        data['batch_size'] = self.batch_size
        # on_epoch_end is not JSON-serializable; excluded
        return data
