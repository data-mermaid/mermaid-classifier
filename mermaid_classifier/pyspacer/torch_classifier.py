"""
PyTorch reimplementation of sklearn.neural_network.MLPClassifier,
scoped to the subset of behavior that mermaid-classifier uses.

Drop-in replacement for `sklearn.neural_network.MLPClassifier` where the
MLP is trained via `partial_fit` and wrapped in
`sklearn.calibration.CalibratedClassifierCV(cv="prefit")`.

Supported API (matches sklearn.MLPClassifier names and semantics):
  - partial_fit(X, y, classes=None)
  - fit(X, y)
  - predict(X)
  - predict_proba(X)
  - classes_ (sorted numpy array)
  - loss_curve_ (list[float], one entry per partial_fit call = one pass)
  - n_iter_ (int, number of partial_fit calls)

Differences from sklearn:
  - Weight init uses `xavier_uniform` (matching sklearn for non-logistic
    activations). Biases initialised to zero.
  - L2 penalty applied in-loss (not via optimizer weight_decay) so the
    biases are excluded, matching sklearn's convention of regularising
    coefs only.
  - `solver` is fixed to Adam. Passing any other value raises.
  - `activation` is fixed to ReLU.
  - `early_stopping` not implemented.

Pickling: `_module` and `_optimizer` are serialised via `state_dict()`
and rebuilt on unpickle so loaded classifiers remain callable without
requiring the exact same torch internal object graph.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Upper bound on the row-sum drift we expect from a softmax computed in
# float32 then cast to float64. float32 eps ≈ 1.2e-7; even naive summation
# over thousands of classes keeps |row_sum - 1| well under 1e-4. Anything
# larger indicates a real numerical issue (extreme logits, NaN/Inf, or a
# bypassed softmax) rather than rounding, and we surface it as a warning.
_EXPECTED_FP_DRIFT_TOL = 1e-4


class _MLPModule(nn.Module):
    """Linear -> ReLU -> ... -> Linear stack. Output returns raw logits."""

    def __init__(
        self,
        n_features_in: int,
        hidden_layer_sizes: Sequence[int],
        n_outputs: int,
    ):
        super().__init__()
        layer_sizes = [n_features_in, *hidden_layer_sizes, n_outputs]
        layers: list[nn.Linear] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
        self.linears = nn.ModuleList(layers)

        for linear in self.linears:
            # Glorot uniform — matches sklearn MLP's init for non-logistic
            # activations (factor=6 in sklearn's _init_coef).
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:
                x = F.relu(x)
        return x


class TorchMLPClassifier:
    """PyTorch reimplementation of sklearn.neural_network.MLPClassifier.

    See module docstring for the supported API subset.
    """

    # Marker so isinstance checks inside serialisation paths (e.g. the
    # patch_estimator SGDClassifier branch in spacer.storage) short-circuit
    # cleanly. Also exposed as `_estimator_type = "classifier"` for any
    # duck-typed caller that inspects it.
    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (100,),
        activation: str = "relu",
        solver: str = "adam",
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
    ):
        if activation != "relu":
            raise ValueError(
                f"TorchMLPClassifier only supports activation='relu', got"
                f" {activation!r}."
            )
        if solver != "adam":
            raise ValueError(
                f"TorchMLPClassifier only supports solver='adam', got"
                f" {solver!r}."
            )

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
        # Per-class loss weighting. dict maps class label (must match a
        # value in classes_) -> non-negative float. Built into a tensor
        # on first partial_fit, once classes_ ordering is known. None
        # disables weighting (loss reverts to unweighted CE).
        self.class_weight = class_weight

    def _resolve_batch_size(self, n_samples: int) -> int:
        if self.batch_size == "auto":
            return min(200, n_samples)
        return min(int(self.batch_size), n_samples)

    def _seed_rng(self) -> np.random.Generator:
        # int random_state: re-create a seeded RNG from `self.random_state`
        # every call, so identical input + same `random_state` reproduces
        # the same shuffle across partial_fit calls.
        base_seed = self.random_state
        if base_seed is not None:
            return np.random.default_rng(int(base_seed))
        # random_state=None: seed a per-instance RNG *once* from NumPy's
        # global RNG and reuse it across calls. This matches sklearn, whose
        # `check_random_state(None)` returns the global singleton — so
        # `np.random.seed(...)` makes shuffling reproducible — while still
        # advancing between calls (no fixed shuffle).
        if not hasattr(self, "_none_rng"):
            self._none_rng = np.random.default_rng(
                np.random.randint(0, np.iinfo(np.int32).max)
            )
        return self._none_rng

    def _labels_to_indices(self, y: np.ndarray) -> np.ndarray:
        # searchsorted on sorted classes_ is fast and stable.
        y = np.asarray(y)
        idx = np.searchsorted(self.classes_, y)
        # Guard against labels missing from classes_ (searchsorted returns
        # an out-of-range index or the wrong index silently otherwise).
        missing = idx >= len(self.classes_)
        if missing.any() or not np.array_equal(self.classes_[idx], y):
            bad = set(np.asarray(y).tolist()) - set(self.classes_.tolist())
            raise ValueError(
                f"Labels {sorted(bad)} are not in classes_"
                f" {self.classes_.tolist()}. Pass all classes to the first"
                f" partial_fit call."
            )
        return idx

    def _init_module(self) -> None:
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        self._module = _MLPModule(
            n_features_in=self.n_features_in_,
            hidden_layer_sizes=self.hidden_layer_sizes,
            n_outputs=len(self.classes_),
        )

    def _init_optimizer(self) -> None:
        self._optimizer = torch.optim.Adam(
            self._module.parameters(),
            lr=self.learning_rate_init,
            betas=(self.beta_1, self.beta_2),
            eps=self.epsilon,
        )

    def _build_class_weight_tensor(self) -> torch.Tensor | None:
        """Materialize ``self.class_weight`` (a dict label->float) into a
        tensor in classes_ order. Returns None if no weighting was
        requested. Raises ValueError if any class in ``classes_`` is
        missing from the weight dict (defensive — class_weight should
        always cover every class the classifier will see)."""
        if self.class_weight is None:
            return None
        weights: list[float] = []
        for cls in self.classes_:
            if cls not in self.class_weight:
                bad = sorted(set(self.classes_.tolist())
                             - set(self.class_weight))
                raise ValueError(
                    f"class_weight is missing weights for {bad!r}."
                    f" Pass weights for every class in classes_."
                )
            w = float(self.class_weight[cls])
            if w < 0:
                raise ValueError(
                    f"class_weight for {cls!r} is negative ({w!r});"
                    f" weights must be >= 0."
                )
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)

    def _l2_penalty(self) -> torch.Tensor:
        # sklearn's MLP applies 0.5 * alpha / n_samples * sum(W^2) to loss,
        # over weights (not biases). We compute sum(W^2) here; the caller
        # scales it.
        penalty = torch.zeros(1, dtype=torch.float32)
        for linear in self._module.linears:
            penalty = penalty + (linear.weight ** 2).sum()
        return penalty.squeeze()

    def partial_fit(
        self,
        X: np.ndarray | list,
        y: np.ndarray | list,
        classes: Sequence | None = None,
    ) -> "TorchMLPClassifier":
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
            # Build the class_weight tensor once classes_ is finalized.
            # Stored as None when no weighting requested (faster CE path).
            self._class_weight_tensor = self._build_class_weight_tensor()
        else:
            if X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X_arr.shape[1]} features, expected"
                    f" {self.n_features_in_}"
                )

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

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            xb = X_tensor[start:end]
            yb = y_tensor[start:end]
            mb_size = end - start

            self._optimizer.zero_grad()
            logits = self._module(xb)
            # When class_weight is provided, F.cross_entropy uses
            # weight[y_i] / sum_j weight[y_j] as the per-sample weight
            # in the mean reduction. Classes assigned weight=0 contribute
            # zero direct gradient via their own CE term, but their
            # logits still participate in the softmax denominator for
            # other classes — for full exclusion, filter the dataset.
            data_loss = F.cross_entropy(
                logits, yb, weight=self._class_weight_tensor)
            # Match sklearn's MLP: L2 penalty per mini-batch is
            # (0.5 * alpha / mb_size) * sum(W^2) — scaled by the size of
            # this specific mini-batch, not the full partial_fit input.
            # See sklearn/neural_network/_multilayer_perceptron.py::_backprop
            # where n_samples = X.shape[0] of the mini-batch.
            reg_loss = (0.5 * self.alpha / mb_size) * self._l2_penalty()
            loss = data_loss + reg_loss
            loss.backward()
            self._optimizer.step()

            # Match sklearn: loss_curve_ records the regularised loss
            # (data + L2) averaged across the full partial_fit input.
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
    ) -> "TorchMLPClassifier":
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        # Reset so fit() starts fresh even on a previously-trained instance.
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
                "TorchMLPClassifier is not fitted. Call partial_fit or fit"
                " before predict/predict_proba."
            )
        X_arr = np.asarray(X, dtype=np.float32)
        # Match sklearn's MLPClassifier contract: X must be 2D with the
        # fitted feature count. Validating here turns an opaque downstream
        # softmax crash into an actionable error.
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected"
                f" {self.n_features_in_}"
            )
        self._module.eval()
        with torch.no_grad():
            probs = F.softmax(self._module(torch.from_numpy(X_arr)), dim=1)
        probs_np = probs.numpy().astype(np.float64)
        # Renormalize so each row sums to exactly 1.0 in float64.
        # Softmax in float32 leaves rows summing to 1.0 ± ~1e-6 once
        # cast to float64, which trips sklearn's tight float64 tolerance
        # inside log_loss. Renormalizing is cheap (one division per row)
        # and only changes the argmax in pathological near-tie cases.
        row_sums = probs_np.sum(axis=1)
        max_drift = float(np.max(np.abs(row_sums - 1.0)))
        if max_drift > _EXPECTED_FP_DRIFT_TOL:
            warnings.warn(
                f"predict_proba row sums deviate from 1.0 by up to "
                f"{max_drift:.2e}, exceeding the expected float32 "
                f"softmax drift bound "
                f"({_EXPECTED_FP_DRIFT_TOL:.0e}). Renormalizing anyway, "
                f"but this likely indicates a numerical issue (extreme "
                f"logits, NaN/Inf, or a bypassed softmax) rather than "
                f"rounding.",
                RuntimeWarning,
                stacklevel=2,
            )
        probs_np /= row_sums[:, np.newaxis]
        return probs_np

    def predict_proba(self, X: np.ndarray | list) -> np.ndarray:
        return self._forward_probs(X)

    def predict(self, X: np.ndarray | list) -> np.ndarray:
        return self.classes_[np.argmax(self._forward_probs(X), axis=1)]

    # --- sklearn parameter protocol (lightweight) -------------------------

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
            # ``getattr`` defends against pickles produced before the
            # class_weight attribute existed; old artifacts unpickle
            # cleanly into a no-weighting classifier.
            "class_weight": getattr(self, "class_weight", None),
        }

    def set_params(self, **params: Any) -> "TorchMLPClassifier":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"Invalid parameter {key!r} for TorchMLPClassifier"
                )
            setattr(self, key, value)
        return self

    # --- pickle support ---------------------------------------------------

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
        # Backfill attributes added after the original release so old
        # pickles unpickle cleanly. Add new defaults here as the class
        # grows, rather than mutating __getstate__ to inject them.
        self.__dict__.setdefault("class_weight", None)
        # ``_class_weight_tensor`` is normally built on the first
        # ``partial_fit`` call, but a pre-class_weight pickle is restored
        # with ``_module`` already present, so that first-call branch is
        # skipped. Backfill None here so resumed training reads a valid
        # (no-weighting) tensor instead of hitting AttributeError.
        self.__dict__.setdefault("_class_weight_tensor", None)
        if module_state is not None:
            # Rebuild module using stored metadata; weights replaced below.
            self._module = _MLPModule(
                n_features_in=self.n_features_in_,
                hidden_layer_sizes=self.hidden_layer_sizes,
                n_outputs=len(self.classes_),
            )
            self._module.load_state_dict(module_state)
        if optimizer_state is not None:
            self._init_optimizer()
            self._optimizer.load_state_dict(optimizer_state)
