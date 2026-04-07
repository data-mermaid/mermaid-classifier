"""
MermaidTrainer: a pyspacer ClassifierTrainer with batched calibration
and per-epoch MLflow callbacks.
"""

import time
from collections.abc import Callable
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV, _fit_calibrator
from sklearn.linear_model import SGDClassifier

from spacer import config
from spacer.data_classes import ImageLabels, ValResults
from spacer.messages import TrainClassifierReturnMsg
from spacer.train_classifier import ClassifierTrainer
from spacer.train_utils import calc_acc, evaluate_classifier

logger = getLogger(__name__)


class _MLPModule(nn.Module):
    """PyTorch MLP matching sklearn MLPClassifier's architecture:
    ReLU activations, configurable hidden layer sizes."""

    def __init__(self, input_dim, hidden_layer_sizes, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier:
    """Drop-in replacement for sklearn MLPClassifier that adds
    class_weight support via PyTorch's CrossEntropyLoss(weight=...).

    Implements the sklearn estimator interface used by MermaidTrainer:
    partial_fit, predict, predict_proba, classes_, loss_curve_.
    Compatible with CalibratedClassifierCV(cv='prefit') and pickle.

    Intentionally omits decision_function to match sklearn
    MLPClassifier behavior (calibration uses predict_proba).
    """

    # Tell sklearn this is a classifier (required by CalibratedClassifierCV
    # and _get_response_values).
    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        learning_rate_init=1e-3,
        class_weight=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.class_weight = class_weight

        self.classes_ = None
        self.loss_curve_ = []

        self._model = None
        self._optimizer = None
        self._loss_fn = None
        self._label_to_idx = None

    def fit(self, X, y):
        """No-op fit to satisfy sklearn's estimator interface check.
        Training is done via partial_fit. This method exists so that
        CalibratedClassifierCV recognizes this as a valid estimator."""
        return self

    def get_params(self, deep=True):
        """Required by sklearn's estimator interface."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'learning_rate_init': self.learning_rate_init,
            'class_weight': self.class_weight,
        }

    def set_params(self, **params):
        """Required by sklearn's estimator interface."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def partial_fit(self, X, y, classes=None):
        x_arr = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(x_arr)

        if self._model is None:
            if classes is None:
                raise ValueError(
                    "classes must be provided on the first call"
                    " to partial_fit.")
            self._init_model(x_arr.shape[1], classes)

        y_indices = np.array(
            [self._label_to_idx[label] for label in y], dtype=np.int64)
        y_tensor = torch.from_numpy(y_indices)

        self._model.train()
        self._optimizer.zero_grad()
        logits = self._model(x_tensor)
        loss = self._loss_fn(logits, y_tensor)
        loss.backward()
        self._optimizer.step()

        self.loss_curve_.append(loss.item())

    def _init_model(self, input_dim, classes):
        torch.manual_seed(0)

        self.classes_ = np.array(classes)
        self._label_to_idx = {
            label: idx for idx, label in enumerate(classes)}
        n_classes = len(classes)

        self._model = _MLPModule(
            input_dim, self.hidden_layer_sizes, n_classes)

        # Match sklearn MLPClassifier's L2 regularization (alpha=0.0001)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate_init,
            weight_decay=0.0001,
        )

        self._loss_fn = self._build_loss_fn(n_classes)

    def _build_loss_fn(self, n_classes):
        if self.class_weight is not None:
            weight_tensor = torch.zeros(n_classes, dtype=torch.float32)
            for label, idx in self._label_to_idx.items():
                weight_tensor[idx] = self.class_weight.get(label, 1.0)
            return nn.CrossEntropyLoss(weight=weight_tensor)
        return nn.CrossEntropyLoss()

    def predict(self, X):
        x_arr = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(x_arr)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(x_tensor)
            indices = logits.argmax(dim=1).numpy()
        return self.classes_[indices]

    def predict_proba(self, X):
        x_arr = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(x_arr)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(x_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs.astype(np.float64)

    def __getstate__(self):
        state = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'learning_rate_init': self.learning_rate_init,
            'class_weight': self.class_weight,
            'classes_': self.classes_,
            'loss_curve_': self.loss_curve_,
            '_label_to_idx': self._label_to_idx,
        }
        if self._model is not None:
            # Convert torch tensors to numpy for pickle protocol=2 compat
            model_sd = self._model.state_dict()
            state['model_state'] = {
                k: v.numpy() for k, v in model_sd.items()}

            opt_sd = self._optimizer.state_dict()
            state['optimizer_state'] = _optimizer_state_to_numpy(opt_sd)
        else:
            state['model_state'] = None
            state['optimizer_state'] = None
        return state

    def __setstate__(self, state):
        self.hidden_layer_sizes = state['hidden_layer_sizes']
        self.learning_rate_init = state['learning_rate_init']
        self.class_weight = state['class_weight']
        self.classes_ = state['classes_']
        self.loss_curve_ = state['loss_curve_']
        self._label_to_idx = state['_label_to_idx']

        if state['model_state'] is not None:
            # Infer input_dim from the first layer's weight shape
            first_weight_key = next(
                k for k in state['model_state'] if k.endswith('.weight'))
            input_dim = state['model_state'][first_weight_key].shape[1]

            # Rebuild model, optimizer, and loss fn via _init_model
            self._init_model(input_dim, self.classes_)

            # Restore learned weights and optimizer state
            torch_sd = {
                k: torch.from_numpy(v)
                for k, v in state['model_state'].items()}
            self._model.load_state_dict(torch_sd)

            opt_sd = _optimizer_state_from_numpy(state['optimizer_state'])
            self._optimizer.load_state_dict(opt_sd)
        else:
            self._model = None
            self._optimizer = None
            self._loss_fn = None


def _optimizer_state_to_numpy(opt_state_dict):
    """Recursively convert torch tensors in optimizer state_dict to numpy."""
    result = {}
    for key, value in opt_state_dict.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.numpy()
        elif isinstance(value, dict):
            result[key] = _optimizer_state_to_numpy(value)
        elif isinstance(value, list):
            result[key] = [
                _optimizer_state_to_numpy(v) if isinstance(v, dict)
                else v.numpy() if isinstance(v, torch.Tensor)
                else v
                for v in value
            ]
        else:
            result[key] = value
    return result


def _optimizer_state_from_numpy(opt_state_dict):
    """Recursively convert numpy arrays in optimizer state_dict to tensors."""
    result = {}
    for key, value in opt_state_dict.items():
        if isinstance(value, np.ndarray):
            result[key] = torch.from_numpy(value)
        elif isinstance(value, dict):
            result[key] = _optimizer_state_from_numpy(value)
        elif isinstance(value, list):
            result[key] = [
                _optimizer_state_from_numpy(v) if isinstance(v, dict)
                else torch.from_numpy(v) if isinstance(v, np.ndarray)
                else v
                for v in value
            ]
        else:
            result[key] = value
    return result


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
        class_balancing: bool = False,
    ):
        self.batch_size = batch_size
        self.on_epoch_end = on_epoch_end
        self.class_balancing = class_balancing

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
        with config.log_entry_and_exit("training using " + clf_type):
            if clf_type == 'MLP':
                if labels.train.label_count >= 50000:
                    hls, lr = (200, 100), 1e-4
                else:
                    hls, lr = (100,), 1e-3
                clf = TorchMLPClassifier(
                    hidden_layer_sizes=hls,
                    learning_rate_init=lr,
                    class_weight=class_weight)
            else:
                # class_weight is applied via sample_weight in partial_fit,
                # not at init — passing both would double-apply weights.
                clf = SGDClassifier(
                    loss='log_loss', average=True, random_state=0)

            ref_accs = []
            t0 = time.time()

            for epoch in range(nbr_epochs):
                for x, y in labels.train.load_data_in_batches(
                    batch_size=self.batch_size,
                    random_seed=epoch,
                ):
                    if class_weight is not None and clf_type != 'MLP':
                        sw = np.array(
                            [class_weight[label] for label in y])
                        clf.partial_fit(
                            x, y, classes=classes_list,
                            sample_weight=sw)
                    else:
                        clf.partial_fit(x, y, classes=classes_list)

                # Ref accuracy: stream ref features from disk in batches.
                # Only predictions (tiny) accumulate, not feature arrays.
                # Training batch data is no longer in memory at this point.
                ref_accs.append(
                    self._calc_acc_batched(clf, labels.ref))
                logger.debug(f"Epoch {epoch}, acc: {ref_accs[-1]}")

                if self.on_epoch_end is not None:
                    loss_curve = getattr(clf, 'loss_curve_', [None])
                    self.on_epoch_end({
                        "epoch": epoch,
                        "ref_accuracy": ref_accs[-1],
                        "training_loss": loss_curve[-1] if loss_curve else None,
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
        accumulate — not feature vectors. Memory use is O(batch_size)
        instead of O(dataset_size).
        """
        gt, pred = [], []
        for x, y in labels.load_data_in_batches(batch_size=self.batch_size):
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
        if self.class_balancing:
            data['class_balancing'] = self.class_balancing
        # on_epoch_end is not JSON-serializable; excluded
        return data
