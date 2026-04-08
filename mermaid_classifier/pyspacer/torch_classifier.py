"""
PyTorch MLP classifier with sklearn-compatible interface for PySpacer storage.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from spacer.data_classes import ImageLabels


class FeatureDataset(Dataset):
    """PyTorch Dataset that materializes all feature/label pairs from
    ImageLabels.load_data_in_batches() into tensors for random-access
    iteration via DataLoader.

    Feature vectors are float32 x D (~16 KB each at D=4096). Even 500K
    samples is ~8 GB, fitting the same memory budget already allocated
    by training_batch_size(). Materializing once avoids re-reading from
    disk every epoch.
    """

    def __init__(self, labels: ImageLabels, label_to_idx: dict,
                 io_batch_size: int):
        xs, ys = [], []
        for x_batch, y_batch in labels.load_data_in_batches(
                batch_size=io_batch_size):
            xs.append(torch.tensor(np.array(x_batch), dtype=torch.float32))
            ys.append(torch.tensor(
                [label_to_idx[l] for l in y_batch], dtype=torch.long))
        self.X = torch.cat(xs)
        self.y = torch.cat(ys)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
}


def resolve_device(device: str) -> torch.device:
    """Resolve device string: 'auto' picks CUDA if available."""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


class TorchMLPClassifier:
    """PyTorch MLP classifier with class_weight support via
    CrossEntropyLoss(weight=...).

    Training API: init_model() then train_step() per batch.
    Implements the sklearn estimator interface needed by
    CalibratedClassifierCV: predict, predict_proba, classes_, fit.
    Compatible with pickle protocol 2 for pyspacer storage.

    Intentionally omits decision_function so calibration
    uses predict_proba (matching sklearn MLPClassifier).
    """

    # Tell sklearn this is a classifier (required by CalibratedClassifierCV
    # and _get_response_values).
    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        learning_rate_init=1e-3,
        class_weight=None,
        optimizer='adam',
        weight_decay=1e-4,
        device='cpu',
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.class_weight = class_weight
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.device = resolve_device(device)

        self.classes_ = None
        self.loss_curve_ = []

        self._model = None
        self._optimizer = None
        self._loss_fn = None
        self._label_to_idx = None

    def fit(self, X, y):
        """No-op fit to satisfy sklearn's estimator interface check.
        Training is done via train_step. This method exists so that
        CalibratedClassifierCV recognizes this as a valid estimator."""
        return self

    def get_params(self, deep=True):
        """Required by sklearn's estimator interface."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'learning_rate_init': self.learning_rate_init,
            'class_weight': self.class_weight,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'device': str(self.device),
        }

    def set_params(self, **params):
        """Required by sklearn's estimator interface."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Run one forward/backward/step on pre-indexed tensors.
        Returns the loss scalar."""
        x = x.to(self.device)
        y = y.to(self.device)
        self._model.train()
        self._optimizer.zero_grad()
        logits = self._model(x)
        loss = self._loss_fn(logits, y)
        loss.backward()
        self._optimizer.step()

        loss_val = loss.item()
        self.loss_curve_.append(loss_val)
        return loss_val

    def init_model(self, input_dim, classes):
        torch.manual_seed(0)

        self.classes_ = np.array(classes)
        self._label_to_idx = {
            label: idx for idx, label in enumerate(classes)}
        n_classes = len(classes)

        self._model = _MLPModule(
            input_dim, self.hidden_layer_sizes, n_classes)
        self._model.to(self.device)

        opt_cls = OPTIMIZERS[self.optimizer]
        self._optimizer = opt_cls(
            self._model.parameters(),
            lr=self.learning_rate_init,
            weight_decay=self.weight_decay,
        )

        self._loss_fn = self._build_loss_fn(n_classes)

    def _build_loss_fn(self, n_classes):
        if self.class_weight is not None:
            weight_tensor = torch.zeros(n_classes, dtype=torch.float32)
            for label, idx in self._label_to_idx.items():
                weight_tensor[idx] = self.class_weight.get(label, 1.0)
            return nn.CrossEntropyLoss(
                weight=weight_tensor.to(self.device))
        return nn.CrossEntropyLoss()

    def predict(self, X):
        x_arr = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(x_arr).to(self.device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(x_tensor)
            indices = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[indices]

    def predict_proba(self, X):
        x_arr = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(x_arr).to(self.device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs.astype(np.float64)

    def __getstate__(self):
        state = {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'learning_rate_init': self.learning_rate_init,
            'class_weight': self.class_weight,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'classes_': self.classes_,
            'loss_curve_': self.loss_curve_,
            '_label_to_idx': self._label_to_idx,
        }
        if self._model is not None:
            # Move to CPU before converting to numpy for pickle protocol=2
            model_sd = self._model.state_dict()
            state['model_state'] = {
                k: v.cpu().numpy() for k, v in model_sd.items()}

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
        # Backward compat: older pickles may lack these fields
        self.optimizer = state.get('optimizer', 'adam')
        self.weight_decay = state.get('weight_decay', 1e-4)
        # Always restore on CPU — device is a runtime concern
        self.device = torch.device('cpu')
        self.classes_ = state['classes_']
        self.loss_curve_ = state['loss_curve_']
        self._label_to_idx = state['_label_to_idx']

        if state['model_state'] is not None:
            # Infer input_dim from the first layer's weight shape
            first_weight_key = next(
                k for k in state['model_state'] if k.endswith('.weight'))
            input_dim = state['model_state'][first_weight_key].shape[1]

            # Rebuild model, optimizer, and loss fn via init_model
            self.init_model(input_dim, self.classes_)

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
            result[key] = value.cpu().numpy()
        elif isinstance(value, dict):
            result[key] = _optimizer_state_to_numpy(value)
        elif isinstance(value, list):
            result[key] = [
                _optimizer_state_to_numpy(v) if isinstance(v, dict)
                else v.cpu().numpy() if isinstance(v, torch.Tensor)
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
