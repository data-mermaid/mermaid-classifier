"""
Autoresearch experiment: custom strategies.

The agent can define new sample weighting or subsampling strategies
here. Import existing strategies from the registries or define novel
ones inline.

To use a new weighting strategy, modify train_experiment.py to pass
it to the SampleWeightingOptions or wire it directly into the
ExperimentRunner.

To use a new subsampling strategy, modify train_experiment.py to
configure the SubsampleOptions accordingly.
"""

# Existing strategy registries — available for import and use.
from mermaid_classifier.training.sample_weighting.registry import (
    compute_class_weights,
)
from mermaid_classifier.training.subsample.registry import (
    compute_per_class_targets,
)

# Example: define a custom weighting strategy
#
# from mermaid_classifier.training.sample_weighting.strategy import Strategy
#
# class MyCustomWeighting(Strategy):
#     def compute_raw_weights(self, class_counts, ba_library, gf_library):
#         # Return dict mapping class_label -> weight
#         total = sum(class_counts.values())
#         return {
#             label: total / (len(class_counts) * count)
#             for label, count in class_counts.items()
#         }
