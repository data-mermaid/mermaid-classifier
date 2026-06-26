"""Shared input context for all metric groups."""

import dataclasses
import typing

import numpy as np
from spacer.data_classes import ValResults

if typing.TYPE_CHECKING:
    from mermaid_classifier.common.benthic_attributes import (
        BenthicAttributeLibrary,
        GrowthFormLibrary,
    )
    from mermaid_classifier.pyspacer.train import TrainingDataset


class MetricsContextError(Exception):
    """Raised when MetricsContext validation fails."""


@dataclasses.dataclass
class MetricsContext:
    """Bundles all inputs needed by any metric group.

    All metric functions receive this context, picking out what they need.
    """

    val_results: ValResults
    ba_library: "BenthicAttributeLibrary"
    gf_library: "GrowthFormLibrary"
    format_func: typing.Callable[[float], typing.Any]
    dataset: "TrainingDataset | None" = None
    clf: typing.Any = None
    val_proba: "np.ndarray | None" = None
    val_gt_labels: "list | None" = None
    ba_to_top: "dict[str, str] | None" = None
    ba_paths: "dict[str, list[str]] | None" = None

    def validate(self):
        """Check that inputs are valid for metric computation.

        Raises MetricsContextError if validation fails.
        """
        if not self.val_results.gt or not self.val_results.est:
            raise MetricsContextError("val_results has no predictions (gt or est is empty)")

        # Check all class IDs used in gt/est exist in val_results.classes
        num_classes = len(self.val_results.classes)
        for idx in set(self.val_results.gt) | set(self.val_results.est):
            if idx < 0 or idx >= num_classes:
                raise MetricsContextError(
                    f"Class index {idx} out of range for {num_classes} classes"
                )

        # Check all class IDs in val_results.classes can be resolved
        # by ba_library
        for class_id in self.val_results.classes:
            try:
                self.ba_library.bagf_id_to_name(class_id, self.gf_library)
            except Exception as e:
                raise MetricsContextError(
                    f"Class ID {class_id!r} not found in ba_library: {e}"
                ) from e

        if self.clf is not None and (
            not hasattr(self.clf, "classes_") or len(self.clf.classes_) == 0
        ):
            raise MetricsContextError("clf has no classes_ attribute or it is empty")
