"""TrainingDataset.__init__ must drop missing-featurevector annotations
before computing per-class subsample targets, not after.

Subsample targets (including the ``min_per_class`` floor) are derived from
whatever the ``annotations`` table holds at the moment ``_apply_subsample``
runs. If that happens before ``handle_missing_feature_vectors``, the
deterministic row selection can land entirely on rows whose feature vector
is later found missing on S3 -- collapsing a class to far below its floor,
and leaving the ``subsample/per_class_counts`` audit's ``realized_n``
pointing at a row count that no longer exists once training actually
begins. This drives a real ``TrainingDataset.__init__()`` end-to-end (all
S3/API touchpoints stubbed) to pin the ordering that prevents that.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock

import duckdb
import pandas as pd
from support.extractor import make_extractor_spec
from support.settings import override_settings

from mermaid_classifier.pyspacer.dataset import TrainingDataset
from mermaid_classifier.pyspacer.extraction import SIDECAR_FILENAME
from mermaid_classifier.pyspacer.options import DatasetOptions
from mermaid_classifier.training.subsample import SubsampleOptions

TEST_BUCKET = "test-mermaid-bucket"
MERMAID_PREFIX = f"s3://{TEST_BUCKET}/mermaid/"

# Two of the rare class's three images are missing their feature vector;
# img_c is the only one actually present on S3.
_PRESENT_FEATURE_PATHS = {
    f"{TEST_BUCKET}/mermaid/img_c_featurevector",
    f"{TEST_BUCKET}/mermaid/img_common_featurevector",
}


class _FakeS3:
    """Stands in for the s3fs handle: __init__ calls only find(), exists(),
    and cat_file() on it, all fully determined ahead of time here."""

    def __init__(self, present: set[str], sidecars: dict[str, bytes]) -> None:
        self._present = present
        self._sidecars = sidecars

    def find(self, path: str) -> list[str]:  # noqa: ARG002 -- single prefix in play
        return sorted(self._present)

    def exists(self, key: str) -> bool:
        return key in self._sidecars

    def cat_file(self, key: str) -> bytes:
        return self._sidecars[key]


class _FakeBALibrary:
    def id_to_name(self, ba_id: str | None) -> str | None:
        return None if ba_id is None else f"BA[{ba_id}]"


class _FakeGFLibrary:
    def id_to_name(self, gf_id: str | None) -> str | None:
        if gf_id is None:
            return None
        return "" if gf_id == "" else f"GF[{gf_id}]"


def _annotations_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_image(image_id: str, ba_id: str, n_points: int) -> None:
        for i in range(n_points):
            rows.append(
                {
                    "image_id": image_id,
                    "row": i,
                    "col": i,
                    "benthic_attribute_id": ba_id,
                    "growth_form_id": "",
                }
            )

    # Rare class: 16 rows pre-drop, spread over 3 images. img_a/img_b (8
    # rows total) are missing their feature vector; only img_c (8 rows)
    # is truly available with features.
    add_image("img_a", "ba_rare", 4)
    add_image("img_b", "ba_rare", 4)
    add_image("img_c", "ba_rare", 8)
    # Common class: present in full, just there to give the 'balanced'
    # allocator a second class to split its budget across.
    add_image("img_common", "ba_common", 20)
    return rows


class SubsampleAfterMissingFeatureVectorDropTest(unittest.TestCase):
    """Drives a real TrainingDataset.__init__() and pins the ordering."""

    def _run_dataset(self, parquet_path: str) -> TrainingDataset:
        sidecar_spec = make_extractor_spec(8)
        sidecars = {MERMAID_PREFIX + SIDECAR_FILENAME: json.dumps(sidecar_spec.to_dict()).encode()}
        fake_s3 = _FakeS3(_PRESENT_FEATURE_PATHS, sidecars)

        options = DatasetOptions(
            include_mermaid=True,
            coralnet_manifest_uri=None,
            subsample=SubsampleOptions(strategy="balanced", total_annotations=24, min_per_class=6),
            ref_val_ratios=(0.1, 0.1),
        )

        with (
            override_settings(
                aws_anonymous="True",
                mermaid_train_data_bucket=TEST_BUCKET,
                mermaid_annotations_parquet_pattern=parquet_path,
                download_max_workers=1,
                # True upstream gap is 2 of 4 distinct feature paths (50%);
                # 70 also clears the pre-fix code's inflated post-subsample
                # ratio, so the abort itself doesn't mask the assertions below.
                training_inputs_percent_missing_allowed=70,
            ),
            mock.patch("mermaid_classifier.pyspacer.dataset.S3FileSystem", return_value=fake_s3),
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.download_features_parallel",
                return_value=set(),
            ),
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.get_benthic_attribute_library",
                return_value=_FakeBALibrary(),
            ),
            mock.patch(
                "mermaid_classifier.pyspacer.dataset.get_growth_form_library",
                return_value=_FakeGFLibrary(),
            ),
        ):
            return TrainingDataset(options)

    def test_min_per_class_and_audit_survive_uneven_feature_coverage(self):
        annotations_df = pd.DataFrame(_annotations_rows())  # noqa: F841 -- referenced by name below

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete_on_close=False) as parquet_f:
            # DuckDB reopens the file to write it, so close our handle first.
            parquet_f.close()
            writer_conn = duckdb.connect()
            writer_conn.register("annotations_df", annotations_df)
            writer_conn.execute(
                f"COPY (SELECT * FROM annotations_df) TO '{parquet_f.name}' (FORMAT parquet)"
            )
            writer_conn.close()

            dataset = self._run_dataset(parquet_f.name)
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
