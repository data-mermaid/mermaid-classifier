"""Portable classifier artifact: TorchScript head export + serve-time loader.

Lives in ``mermaid-classifier[inference]``. Modules in this subpackage import
only torch / numpy / json / stdlib (sklearn is used at export time, pulled
transitively by pyspacer). They must NOT import the training-only settings
layer, so the [inference] dependency split holds.
"""

SCHEMA_VERSION = 1
TASK_NAME = "pyspacer_mlp_classifier"


class ParityError(Exception):
    """Raised when the frozen graph diverges from the source model beyond the
    parity tolerance — fails the export/build."""


class ManifestError(Exception):
    """Raised at load time when model.json is incompatible with the graph
    (schema version, class count, or input_dim mismatch)."""


from mermaid_classifier.pyspacer.inference.export import export_artifact  # noqa: E402
from mermaid_classifier.pyspacer.inference.loader import (  # noqa: E402
    Predictor, load_predictor,
)

__all__ = [
    "SCHEMA_VERSION", "TASK_NAME", "ParityError", "ManifestError",
    "export_artifact", "Predictor", "load_predictor",
]
