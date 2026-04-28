"""Strategy registry and convenience entry point.

The registry lives in its own file so concrete Strategy classes can be
imported by name without circular dependencies. Each strategy module
imports ``Strategy`` from ``base`` and registers itself via
``register_strategy``. The trainer looks up strategies by string name.
"""
from __future__ import annotations

from typing import Type

from mermaid_classifier.training.sample_weighting.base import Strategy
from mermaid_classifier.training.sample_weighting.options import (
    SampleWeightingOptions,
)


STRATEGY_REGISTRY: dict[str, Type[Strategy]] = {}


def register_strategy(name: str):
    """Decorator that registers a Strategy subclass under a string name."""

    def _decorator(cls: Type[Strategy]) -> Type[Strategy]:
        if name in STRATEGY_REGISTRY:
            raise ValueError(
                f"Strategy name {name!r} is already registered to"
                f" {STRATEGY_REGISTRY[name].__name__}."
            )
        STRATEGY_REGISTRY[name] = cls
        return cls

    return _decorator


# Import strategy modules so their @register_strategy decorators run.
# Imports are at the bottom of the file to avoid circular dependency
# (each strategy module imports ``register_strategy`` from this module).
from mermaid_classifier.training.sample_weighting import (  # noqa: E402,F401
    decomposed,
    effective_number,
    leaf_inverse,
    tree_balanced,
)


def compute_class_weights(
    class_counts: dict[str, int],
    ba_library,
    gf_library,
    options: SampleWeightingOptions,
) -> dict[str, float]:
    """Look up the configured strategy and compute per-class weights.

    Returns a dict mapping each BA+GF combo string in ``class_counts``
    to its weight. Returns ``{}`` if ``options.enabled`` is False or
    ``class_counts`` is empty.
    """
    if not options.enabled or not class_counts:
        return {}

    if options.strategy not in STRATEGY_REGISTRY:
        available = sorted(STRATEGY_REGISTRY)
        raise ValueError(
            f"Unknown weighting strategy {options.strategy!r}."
            f" Available: {available!r}"
        )

    cls = STRATEGY_REGISTRY[options.strategy]
    strategy = cls(
        alpha=options.alpha,
        weight_ratio_cap=options.weight_ratio_cap,
    )
    return strategy.compute(class_counts, ba_library, gf_library)
