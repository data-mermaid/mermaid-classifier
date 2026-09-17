"""Region-mismatch metrics over the frozen region probe.

The probe is a fixed slice of MERMAID annotation points, frozen once with the
benthic-attribute region map and display names it is read against. Scoring it
answers what a run's own validation split cannot: most committed configs set
`include_mermaid: false`, and a fixed image set makes one model version
comparable with the next rather than with whatever data its config happened
to draw.

Scoring goes through `ctx.clf`, the `Predictor` the run built from the
exported `model.pt` and `model.json`, so what is measured is the artifact
that ships rather than the in-memory estimator it was exported from.

`settings.region_probe_dir` locates the probe. Unset, the group does nothing:
a run with no probe to score against is the common case, not a failure. A
directory that is configured but not readable raises instead, because
silently scoring no points would read as a model with no incidents. The
coordinator catches that raise, leaving `region_probe/scored` at the 0 it
logged before the group ran.

The rates carry the `region_probe/` prefix throughout. The validation-split
group measures the same behaviour over a different population with different
denominators, and the two prefixes never merge.
"""

from pathlib import Path

from mermaid_classifier.pyspacer.metrics._context import MetricsContext
from mermaid_classifier.pyspacer.metrics._results import MetricGroupResult
from mermaid_classifier.pyspacer.metrics.region import emit_region_metrics
from mermaid_classifier.pyspacer.settings import settings
from mermaid_classifier.region_eval.metrics import (
    compute_region_metrics,
    prepare_scored_points,
)
from mermaid_classifier.region_eval.score import load_probe, predict_with_probabilities

PROBE_PREFIX = "region_probe"


def compute_region_probe(ctx: MetricsContext) -> MetricGroupResult:
    """Score the frozen probe through the run's exported predictor."""
    probe_dir = settings.region_probe_dir
    if not probe_dir or ctx.clf is None:
        return MetricGroupResult()

    probe = load_probe(Path(probe_dir))
    classes = tuple(str(label) for label in ctx.clf.classes_)
    predictions, _probabilities = predict_with_probabilities(ctx.clf, probe.features.features)

    points = prepare_scored_points(
        image_ids=probe.features.image_ids,
        image_region_ids=probe.features.region_ids,
        gt_labels=probe.features.gt_labels,
        pred_labels=predictions,
        region_ids_by_attribute=probe.region_ids_by_attribute,
        model_classes=classes,
        held_out=probe.features.held_out.tolist(),
    )
    metrics = compute_region_metrics(
        points,
        label_names=probe.label_display_names([*classes, *probe.features.gt_labels, *predictions]),
        region_names=probe.names.regions,
    )
    return emit_region_metrics(PROBE_PREFIX, metrics)
