"""
Autoresearch experiment: MLP classifier.

Seeded from mermaid_classifier/pyspacer/torch_classifier.py.
The agent may freely modify this file: change the architecture, add
dropout, batch norm, skip connections, different activations, weight
init, or replace the MLP entirely.

The classifier must expose the same API surface used by MermaidTrainer:
  - partial_fit(X, y, classes=None)
  - predict(X) -> np.ndarray
  - predict_proba(X) -> np.ndarray
  - classes_ attribute (sorted numpy array)
  - loss_curve_ attribute (list[float])
  - decision_function(X) or predict_proba(X) for calibration
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLPModule(nn.Module):
    """Linear -> ReLU -> Dropout -> ... -> Linear stack. Output returns raw logits."""

    def __init__(
        self,
        n_features_in: int,
        hidden_layer_sizes: Sequence[int],
        n_outputs: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layer_sizes = [n_features_in, *hidden_layer_sizes, n_outputs]
        layers: list[nn.Linear] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
        self.linears = nn.ModuleList(layers)
        self.dropout = float(dropout)

        for linear in self.linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:
                x = F.relu(x)
                if self.dropout > 0.0:
                    x = F.dropout(
                        x, p=self.dropout, training=self.training)
        return x


class ExperimentMLPClassifier:
    """MLP classifier for autoresearch experiments.

    Drop-in replacement for TorchMLPClassifier with GPU support.
    The agent can modify the architecture, optimizer, loss function, etc.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (100,),
        activation: str = "relu",
        solver: str = "adamw",
        alpha: float = 0.0001,
        batch_size: int | str = "auto",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: int | None = None,
        tol: float = 1e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        class_weight: dict | None = None,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
    ):
        if activation != "relu":
            raise ValueError(
                f"Only activation='relu' supported, got {activation!r}.")
        if solver not in ("adam", "adamw"):
            raise ValueError(
                f"Only solver in {{'adam','adamw'}} supported, got {solver!r}.")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0), got {dropout!r}.")
        if float(weight_decay) < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {weight_decay!r}.")

        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.class_weight = class_weight
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)

    def _resolve_batch_size(self, n_samples: int) -> int:
        if self.batch_size == "auto":
            return min(200, n_samples)
        return min(int(self.batch_size), n_samples)

    def _seed_rng(self) -> np.random.Generator:
        base_seed = self.random_state
        if base_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(base_seed))

    def _labels_to_indices(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        idx = np.searchsorted(self.classes_, y)
        missing = idx >= len(self.classes_)
        if missing.any() or not np.array_equal(self.classes_[idx], y):
            bad = set(np.asarray(y).tolist()) - set(self.classes_.tolist())
            raise ValueError(
                f"Labels {sorted(bad)} are not in classes_"
                f" {self.classes_.tolist()}.")
        return idx

    def _init_module(self) -> None:
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        self._module = _MLPModule(
            n_features_in=self.n_features_in_,
            hidden_layer_sizes=self.hidden_layer_sizes,
            n_outputs=len(self.classes_),
            dropout=self.dropout,
        )

    def _init_optimizer(self) -> None:
        # AdamW decouples weight decay from the gradient update, giving
        # principled L2-style regularization that the prior Adam +
        # alpha=1e-4 L2 penalty (effectively ~2.5e-7 per step) did not
        # provide. weight_decay=0 reproduces plain Adam behavior.
        if self.solver == "adamw":
            self._optimizer = torch.optim.AdamW(
                self._module.parameters(),
                lr=self.learning_rate_init,
                betas=(self.beta_1, self.beta_2),
                eps=self.epsilon,
                weight_decay=self.weight_decay,
            )
        else:
            self._optimizer = torch.optim.Adam(
                self._module.parameters(),
                lr=self.learning_rate_init,
                betas=(self.beta_1, self.beta_2),
                eps=self.epsilon,
            )

    def _build_class_weight_tensor(self) -> torch.Tensor | None:
        if self.class_weight is None:
            return None
        weights: list[float] = []
        for cls in self.classes_:
            if cls not in self.class_weight:
                bad = sorted(set(self.classes_.tolist())
                             - set(self.class_weight))
                raise ValueError(
                    f"class_weight is missing weights for {bad!r}.")
            w = float(self.class_weight[cls])
            if w < 0:
                raise ValueError(
                    f"class_weight for {cls!r} is negative ({w!r}).")
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)

    def _l2_penalty(self) -> torch.Tensor:
        penalty = torch.zeros(1, dtype=torch.float32)
        for linear in self._module.linears:
            penalty = penalty + (linear.weight ** 2).sum()
        return penalty.squeeze()

    def partial_fit(
        self,
        X: np.ndarray | list,
        y: np.ndarray | list,
        classes: Sequence | None = None,
    ) -> "ExperimentMLPClassifier":
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")

        first_call = not hasattr(self, "_module")
        if first_call:
            if classes is None:
                self.classes_ = np.unique(np.asarray(y))
            else:
                self.classes_ = np.unique(np.asarray(classes))
            self.n_features_in_ = int(X_arr.shape[1])
            self.n_iter_ = 0
            self.loss_curve_ = []
            self._init_module()
            self._init_optimizer()
            self._class_weight_tensor = self._build_class_weight_tensor()
        else:
            if X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X_arr.shape[1]} features, expected"
                    f" {self.n_features_in_}")

        y_indices = self._labels_to_indices(np.asarray(y))
        n_samples = X_arr.shape[0]
        batch_size = self._resolve_batch_size(n_samples)

        rng = self._seed_rng()
        order = np.arange(n_samples)
        if self.shuffle:
            rng.shuffle(order)

        X_tensor = torch.from_numpy(X_arr[order])
        y_tensor = torch.from_numpy(y_indices[order].astype(np.int64))

        self._module.train()
        total_weighted_loss = 0.0
        total_seen = 0

        # If AdamW is supplying weight decay, skip the inline L2 penalty
        # to avoid double-regularizing. Keep it active for plain Adam to
        # preserve prior baseline semantics.
        use_inline_l2 = self.solver != "adamw"

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xb = X_tensor[start:end]
            yb = y_tensor[start:end]
            mb_size = end - start

            self._optimizer.zero_grad()
            logits = self._module(xb)
            data_loss = F.cross_entropy(
                logits, yb, weight=self._class_weight_tensor)
            if use_inline_l2:
                reg_loss = (0.5 * self.alpha / mb_size) * self._l2_penalty()
                loss = data_loss + reg_loss
            else:
                loss = data_loss
            loss.backward()
            self._optimizer.step()

            total_weighted_loss += loss.item() * mb_size
            total_seen += mb_size

        avg_loss = total_weighted_loss / max(total_seen, 1)
        self.loss_curve_.append(float(avg_loss))
        self.n_iter_ += 1
        return self

    def fit(
        self,
        X: np.ndarray | list,
        y: np.ndarray | list,
    ) -> "ExperimentMLPClassifier":
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        for attr in ("_module", "_optimizer", "classes_", "n_features_in_",
                     "n_iter_", "loss_curve_"):
            if hasattr(self, attr):
                delattr(self, attr)
        prev_loss = np.inf
        for _ in range(self.max_iter):
            self.partial_fit(X, y_arr, classes=classes)
            cur = self.loss_curve_[-1]
            if abs(prev_loss - cur) < self.tol:
                break
            prev_loss = cur
        return self

    def _forward_probs(self, X: np.ndarray | list) -> np.ndarray:
        if not hasattr(self, "_module"):
            raise RuntimeError(
                "Classifier is not fitted. Call partial_fit or fit first.")
        X_arr = np.asarray(X, dtype=np.float32)
        self._module.eval()
        with torch.no_grad():
            probs = F.softmax(self._module(torch.from_numpy(X_arr)), dim=1)
        return probs.numpy().astype(np.float64)

    def predict_proba(self, X: np.ndarray | list) -> np.ndarray:
        return self._forward_probs(X)

    def predict(self, X: np.ndarray | list) -> np.ndarray:
        return self.classes_[np.argmax(self._forward_probs(X), axis=1)]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "tol": self.tol,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "class_weight": getattr(self, "class_weight", None),
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
        }

    def set_params(self, **params: Any) -> "ExperimentMLPClassifier":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"Invalid parameter {key!r} for ExperimentMLPClassifier")
            setattr(self, key, value)
        return self

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        module = state.pop("_module", None)
        optimizer = state.pop("_optimizer", None)
        if module is not None:
            state["_module_state"] = {
                k: v.detach().cpu() for k, v in module.state_dict().items()
            }
        if optimizer is not None:
            state["_optimizer_state"] = optimizer.state_dict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        module_state = state.pop("_module_state", None)
        optimizer_state = state.pop("_optimizer_state", None)
        self.__dict__.update(state)
        self.__dict__.setdefault("class_weight", None)
        self.__dict__.setdefault("dropout", 0.0)
        self.__dict__.setdefault("weight_decay", 0.0)
        self.__dict__.setdefault("solver", "adam")
        if module_state is not None:
            self._module = _MLPModule(
                n_features_in=self.n_features_in_,
                hidden_layer_sizes=self.hidden_layer_sizes,
                n_outputs=len(self.classes_),
                dropout=self.dropout,
            )
            self._module.load_state_dict(module_state)
        if optimizer_state is not None:
            self._init_optimizer()
            self._optimizer.load_state_dict(optimizer_state)
