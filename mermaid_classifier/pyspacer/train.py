"""
Train a classifier using feature vectors and annotations
provided on S3.

.. deprecated::
    This module is a transitional re-export shim. Import directly from the
    focused modules instead:
    - mermaid_classifier.pyspacer.dataset  (TrainingDataset)
    - mermaid_classifier.pyspacer.runner   (TrainingRunner, MLflowTrainingRunner)
    - mermaid_classifier.pyspacer.options  (DatasetOptions, TrainingOptions, MLflowOptions, Sites, Artifacts)
    - mermaid_classifier.pyspacer.label_specs (LabelFilter, LabelRollupSpec, CNSourceFilter)
    - mermaid_classifier.pyspacer._pipeline_utils (section_profiling, download_features_parallel)
"""

from mermaid_classifier.pyspacer._pipeline_utils import (
    download_features_parallel,
    section_profiling,
)
from mermaid_classifier.pyspacer.dataset import TrainingDataset
from mermaid_classifier.pyspacer.label_specs import (
    CNSourceFilter,
    LabelFilter,
    LabelRollupSpec,
)
from mermaid_classifier.pyspacer.options import (
    Artifacts,
    DatasetOptions,
    MLflowOptions,
    Sites,
    TrainingOptions,
)
from mermaid_classifier.pyspacer.runner import (
    MLflowTrainingRunner,
    TrainingRunner,
)

__all__ = [
    "Artifacts",
    "CNSourceFilter",
    "DatasetOptions",
    "LabelFilter",
    "LabelRollupSpec",
    "MLflowOptions",
    "MLflowTrainingRunner",
    "Sites",
    "TrainingDataset",
    "TrainingOptions",
    "TrainingRunner",
    "download_features_parallel",
    "section_profiling",
]
