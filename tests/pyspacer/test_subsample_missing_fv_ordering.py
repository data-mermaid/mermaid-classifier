"""TrainingDataset.__init__ must drop missing-featurevector annotations
before computing per-class subsample targets, not after.

Subsample targets (including the ``min_per_class`` floor) are derived from
whatever the ``annotations`` table holds at the moment ``_apply_subsample``
runs. If that happens before ``handle_missing_feature_vectors``, the
deterministic row selection can land entirely on rows whose feature vector
is later found absent from the feature index -- collapsing a class to far below its floor,
and leaving the ``subsample/per_class_counts`` audit's ``realized_n``
pointing at a row count that no longer exists once training actually
begins. This drives a real ``TrainingDataset.__init__()`` end-to-end (against an S3 fake
holding a published dataset version) to pin the ordering that prevents that.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from mermaid_classifier.pyspacer.options import DatasetOptions, PinnedDataset
from mermaid_classifier.training.subsample import SubsampleOptions
from pyspacer.segdata_fixtures import (
    FakeS3,
    Image,
    label_row,
    load_training_dataset,
    publish_dataset,
    publish_feature_set,
)

# CoralNet label 1 is the rare class, label 2 the common one.
_MAPPING = [
    {
        "provider_id": str(label_id),
        "provider_label": ba_id,
        "benthic_attribute_id": ba_id,
        "benthic_attribute_name": ba_id,
        "growth_form_id": None,
        "growth_form_name": None,
    }
    for label_id, ba_id in ((1, "ba_rare"), (2, "ba_common"))
]


def _points() -> list[tuple[str, int, int, int]]:
    points: list[tuple[str, int, int, int]] = []

    def add_image(image_id: str, label_id: int, n_points: int) -> None:
        points.extend((image_id, i, i, label_id) for i in range(n_points))

    # Rare class: 16 rows pre-drop, spread over 3 images. img_a/img_b (8
    # rows total) are absent from the feature index; only img_c (8 rows)
    # has features.
    add_image("img_a", 1, 4)
    add_image("img_b", 1, 4)
    add_image("img_c", 1, 8)
    # Common class: present in full, just there to give the 'balanced'
    # allocator a second class to split its budget across.
    add_image("img_common", 2, 20)
    return points


class SubsampleAfterMissingFeatureVectorDropTest(unittest.TestCase):
    """Drives a real TrainingDataset.__init__() and pins the ordering."""

    def test_min_per_class_and_audit_survive_uneven_feature_coverage(self):
        fake = FakeS3()
        cache = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, cache, ignore_errors=True)
        images = [Image(i, "s1") for i in ("img_a", "img_b", "img_c", "img_common")]
        pin = publish_dataset(fake, "coralnet-raw", images, _points(), [label_row(1), label_row(2)])
        features = publish_feature_set(fake, pin, ["img_c", "img_common"], n_missing=2)
        options = DatasetOptions(
            datasets=[PinnedDataset(pin=pin, features=features, labels="coralnet")],
            feature_extractor_weights="s3://test-bucket/efficientnet.pt",
            subsample=SubsampleOptions(strategy="balanced", total_annotations=24, min_per_class=6),
            ref_val_ratios=(0.1, 0.1),
        )

        dataset = load_training_dataset(
            fake,
            options,
            _MAPPING,
            feature_cache_dir=str(cache),
            download_max_workers=1,
            # True upstream gap is 2 of 4 distinct images (50%); 70 also
            # clears the pre-fix code's inflated post-subsample ratio, so the
            # abort itself doesn't mask the assertions below.
            training_inputs_percent_missing_allowed=70,
        )
        self.addCleanup(dataset.cleanup)

        # 8 of the rare class's 16 rows truly have a feature vector; the
        # subsample must size its target off that true count, not shrink
        # further once an already-applied subsample meets a later drop.
        realized_rare = dataset.duck_conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE benthic_attribute_id = 'ba_rare'"
        ).fetchone()[0]
        self.assertEqual(realized_rare, 8)

        assert dataset._subsample_audit_df is not None
        audit_row = dataset._subsample_audit_df.loc[
            dataset._subsample_audit_df["benthic_attribute_id"] == "ba_rare"
        ].iloc[0]

        # pre_count is read at the top of _apply_subsample: 8 (post-drop),
        # not 16 (pre-drop), proves the drop ran first.
        self.assertEqual(audit_row["pre_count"], 8)
        # The audit's realized_n must describe what actually reached the
        # final annotations table, not a count snapshotted before a later
        # drop silently invalidated it.
        self.assertEqual(audit_row["realized_n"], realized_rare)


if __name__ == "__main__":
    unittest.main()
