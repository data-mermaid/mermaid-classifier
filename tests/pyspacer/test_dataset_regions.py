"""
Characterization tests for how region_id/region_name flow through the
TrainingDataset DuckDB pipeline (step 2 of the region-mismatch-metrics plan).

Step 1 (mermaid_classifier/common/) added region lookups and the
is_out_of_region rule. This module's job is only to carry each image's
MERMAID region through read_mermaid_data, read_coralnet_manifest, and
prep_annotations_for_pyspacer's feature_loc_to_region map -- no metric reads
this data yet.

DuckDB is exercised for real (a live connection over small parquet/dataframe
fixtures), never mocked, since a mocked cursor cannot catch a wrong join key
or a dropped column -- which is the entire risk in this change.
"""

import shutil
import tempfile
import unittest
from unittest import mock

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mermaid_classifier.common.benthic_attributes import CoralNetMermaidMapping
from mermaid_classifier.pyspacer.options import DatasetOptions, Sites

# Reuse scaffolding from the existing test module.
from pyspacer.test_train import BaseTrainTest, NoInitDataset, override_settings, same_char_uuid

_MAPPING = [
    {
        "benthic_attribute_id": same_char_uuid("0"),
        "growth_form_id": same_char_uuid("1"),
        "benthic_attribute_name": "BA1",
        "growth_form_name": "GF1",
        "provider_id": "123",
        "provider_label": "Label123",
    }
]


def _write_coralnet_manifest(source_id: int, image_id: str, coralnet_id: int) -> str:
    table = pa.table(
        {
            "source_id": pa.array([source_id], pa.int32()),
            "image_id": pa.array([image_id], pa.string()),
            "row": pa.array([100], pa.int32()),
            "col": pa.array([200], pa.int32()),
            "coralnet_id": pa.array([coralnet_id], pa.int32()),
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as manifest_f:
        manifest_f.close()
        pq.write_table(table, manifest_f.name)
        return manifest_f.name


def _write_mermaid_parquet(image_id: str, region_id: str, region_name: str) -> str:
    df = pd.DataFrame(  # noqa: F841 — referenced by name in DuckDB SQL
        {
            "image_id": [image_id],
            "row": [10],
            "col": [20],
            "benthic_attribute_id": [same_char_uuid("5")],
            "growth_form_id": [same_char_uuid("7")],
            "region_id": [region_id],
            "region_name": [region_name],
        }
    )
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t AS SELECT * FROM df")
    with tempfile.NamedTemporaryFile(delete=False) as parquet_f:
        parquet_f.close()
        conn.execute(f"COPY (SELECT * FROM t) TO '{parquet_f.name}' (FORMAT parquet)")
        return parquet_f.name


def _write_mermaid_parquet_with_null_region(image_id: str) -> str:
    """A MERMAID parquet row with NULL region_id/region_name.

    A site with no matching Region polygon, or an image with no site at all,
    writes NULL rather than ''. Built with pyarrow directly (rather than
    pandas) so the column keeps an explicit string type despite carrying no
    non-null value, matching how the manifest fixture above is built.
    """
    table = pa.table(
        {
            "image_id": pa.array([image_id], pa.string()),
            "row": pa.array([10], pa.int32()),
            "col": pa.array([20], pa.int32()),
            "benthic_attribute_id": pa.array([same_char_uuid("5")], pa.string()),
            "growth_form_id": pa.array([same_char_uuid("7")], pa.string()),
            "region_id": pa.array([None], pa.string()),
            "region_name": pa.array([None], pa.string()),
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as parquet_f:
        parquet_f.close()
        pq.write_table(table, parquet_f.name)
        return parquet_f.name


class RegionColumnsThroughDuckDBTest(BaseTrainTest):
    """read_mermaid_data / read_coralnet_manifest carry region_id/region_name
    into the annotations table."""

    def test_mermaid_rows_carry_real_region(self):
        """A MERMAID row's region_id/region_name reach the annotations table
        unchanged -- omitting them from the SELECT would drop them instead."""
        dataset = NoInitDataset()
        parquet_path = _write_mermaid_parquet(same_char_uuid("3"), "region-uuid-1", "Region One")

        with override_settings(mermaid_annotations_parquet_pattern=parquet_path):
            dataset.read_mermaid_data()

        region_id, region_name = dataset.duck_conn.execute(
            "SELECT region_id, region_name FROM annotations"
        ).fetchone()
        self.assertEqual(region_id, "region-uuid-1")
        self.assertEqual(region_name, "Region One")

    def test_coralnet_rows_get_empty_region(self):
        """CoralNet has no MERMAID region; its rows must carry '' rather than
        NULL (NULL breaks DuckDB JOINs in this repo's convention) or an
        omitted column."""
        dataset = NoInitDataset()
        manifest_path = _write_coralnet_manifest(23, "12345", 123)
        dataset.options = DatasetOptions(coralnet_manifest_uri=manifest_path)

        with mock.patch.object(CoralNetMermaidMapping, "_download_mapping", return_value=_MAPPING):
            dataset.read_coralnet_manifest()

        region_id, region_name = dataset.duck_conn.execute(
            "SELECT region_id, region_name FROM annotations"
        ).fetchone()
        self.assertEqual(region_id, "")
        self.assertEqual(region_name, "")

    def test_null_region_id_in_mermaid_parquet_becomes_empty_string(self):
        """A NULL region_id/region_name from the MERMAID parquet must land as
        '', not NULL. NULL survives into `str(row["region_id"])` downstream as
        the literal 'None', a non-empty string that is_out_of_region cannot
        distinguish from a real region id -- every prediction on such an image
        would score as a mismatch."""
        dataset = NoInitDataset()
        parquet_path = _write_mermaid_parquet_with_null_region(same_char_uuid("4"))

        with override_settings(mermaid_annotations_parquet_pattern=parquet_path):
            dataset.read_mermaid_data()

        region_id, region_name = dataset.duck_conn.execute(
            "SELECT region_id, region_name FROM annotations"
        ).fetchone()
        self.assertEqual(region_id, "")
        self.assertEqual(region_name, "")

    def test_coralnet_then_mermaid_load_order_does_not_raise(self):
        """CoralNet is read first and creates the annotations table; MERMAID
        then does INSERT INTO annotations BY NAME. Without matching
        region_id/region_name columns on the CoralNet-created table, that
        insert fails outright -- this is what makes the CoralNet SELECT's
        '' AS region_id, '' AS region_name mandatory, not cosmetic."""
        dataset = NoInitDataset()
        manifest_path = _write_coralnet_manifest(23, "12345", 123)
        dataset.options = DatasetOptions(coralnet_manifest_uri=manifest_path)

        with mock.patch.object(CoralNetMermaidMapping, "_download_mapping", return_value=_MAPPING):
            dataset.read_coralnet_manifest()

        mermaid_image_id = same_char_uuid("3")
        parquet_path = _write_mermaid_parquet(mermaid_image_id, "region-uuid-2", "Region Two")

        with override_settings(mermaid_annotations_parquet_pattern=parquet_path):
            dataset.read_mermaid_data()

        rows = dataset.duck_conn.execute(
            "SELECT site, region_id, region_name FROM annotations ORDER BY site"
        ).fetchall()
        by_site = {site: (region_id, region_name) for site, region_id, region_name in rows}

        self.assertEqual(
            set(by_site),
            {Sites.CORALNET.value, Sites.MERMAID.value},
            msg="Both CoralNet and MERMAID rows should have loaded into the combined table.",
        )
        self.assertEqual(by_site[Sites.CORALNET.value], ("", ""))
        self.assertEqual(by_site[Sites.MERMAID.value], ("region-uuid-2", "Region Two"))


def _make_prep_dataset(test_case: unittest.TestCase) -> NoInitDataset:
    """Return a NoInitDataset with the attributes prep_annotations_for_pyspacer needs."""
    dataset = NoInitDataset()
    dataset._feature_dir = tempfile.mkdtemp()
    test_case.addCleanup(shutil.rmtree, dataset._feature_dir, ignore_errors=True)
    dataset.profiled_sections = []
    dataset._feature_path_to_s3_location = {}
    dataset.feature_loc_to_source = {}
    dataset.feature_loc_to_region = {}
    dataset.options = DatasetOptions(ref_val_ratios=(0.1, 0.1))
    return dataset


_N_POINTS_PER_IMAGE = 10


def _seed_mixed_annotations(dataset: NoInitDataset) -> None:
    """Seed one MERMAID image (agreeing region) and one CoralNet image (no
    region). Each gets enough points to clear preprocess_labels' stratified
    split threshold, which drops classes with fewer than 3 annotations."""
    rows = []
    for i in range(_N_POINTS_PER_IMAGE):
        rows.append(
            {
                "image_id": "m-img-1",
                "row": i,
                "col": i,
                "benthic_attribute_id": "ba_a",
                "growth_form_id": "",
                "site": Sites.MERMAID.value,
                "bucket": "mermaid-bucket",
                "project_id": "all",
                "feature_vector": "mermaid/m-img-1_featurevector",
                "region_id": "region-x",
                "region_name": "Region X",
            }
        )
    for i in range(_N_POINTS_PER_IMAGE):
        rows.append(
            {
                "image_id": "12345",
                "row": i,
                "col": i,
                "benthic_attribute_id": "ba_c",
                "growth_form_id": "",
                "site": Sites.CORALNET.value,
                "bucket": "cn-bucket",
                "project_id": "s1",
                "feature_vector": "s1/features/i12345.featurevector",
                "region_id": "",
                "region_name": "",
            }
        )
    df = pd.DataFrame(rows)  # noqa: F841 — referenced by name in DuckDB SQL
    dataset.duck_conn.execute("CREATE OR REPLACE TABLE annotations AS SELECT * FROM df")


class FeatureLocToRegionTest(unittest.TestCase):
    """prep_annotations_for_pyspacer populates feature_loc_to_region."""

    def setUp(self):
        self.override = override_settings(aws_anonymous="True", download_max_workers=1)
        self.override.__enter__()
        self.dataset = _make_prep_dataset(self)

    def tearDown(self):
        self.override.__exit__(None, None, None)

    def _call_prep(self):
        with mock.patch(
            "mermaid_classifier.pyspacer.dataset.download_features_parallel",
            return_value=set(),
        ):
            return self.dataset.prep_annotations_for_pyspacer()

    def _region_by_s3_key(self) -> dict[tuple[str, str], tuple[str, str]]:
        return {
            self.dataset._feature_path_to_s3_location[loc.key]: region
            for loc, region in self.dataset.feature_loc_to_region.items()
        }

    def test_mermaid_image_holds_real_region(self):
        """The MERMAID image's feature_loc_to_region entry is its real
        (region_id, region_name), not omitted or defaulted."""
        _seed_mixed_annotations(self.dataset)
        self._call_prep()

        region_by_s3_key = self._region_by_s3_key()
        self.assertEqual(
            region_by_s3_key[("mermaid-bucket", "mermaid/m-img-1_featurevector")],
            ("region-x", "Region X"),
        )

    def test_coralnet_image_holds_empty_region_not_omitted(self):
        """The CoralNet image's feature_loc_to_region entry is ('', ''),
        distinguishing "no region" from the image being absent from the map."""
        _seed_mixed_annotations(self.dataset)
        self._call_prep()

        region_by_s3_key = self._region_by_s3_key()
        cn_key = ("cn-bucket", "s1/features/i12345.featurevector")
        self.assertIn(cn_key, region_by_s3_key)
        self.assertEqual(region_by_s3_key[cn_key], ("", ""))

    def test_disagreeing_regions_on_one_image_raises(self):
        """Two rows sharing one image but disagreeing on region_id signal
        corrupted input; silently taking the first row's value would
        mislabel the image and quietly corrupt every downstream rate."""
        rows = [
            {
                "image_id": "m-img-2",
                "row": 0,
                "col": 0,
                "benthic_attribute_id": "ba_a",
                "growth_form_id": "",
                "site": Sites.MERMAID.value,
                "bucket": "mermaid-bucket",
                "project_id": "all",
                "feature_vector": "mermaid/m-img-2_featurevector",
                "region_id": "region-x",
                "region_name": "Region X",
            },
            {
                "image_id": "m-img-2",
                "row": 1,
                "col": 1,
                "benthic_attribute_id": "ba_b",
                "growth_form_id": "",
                "site": Sites.MERMAID.value,
                "bucket": "mermaid-bucket",
                "project_id": "all",
                "feature_vector": "mermaid/m-img-2_featurevector",
                "region_id": "region-y",
                "region_name": "Region Y",
            },
        ]
        df = pd.DataFrame(rows)  # noqa: F841 — referenced by name in DuckDB SQL
        self.dataset.duck_conn.execute("CREATE OR REPLACE TABLE annotations AS SELECT * FROM df")

        with self.assertRaises(ValueError) as ctx:
            self._call_prep()

        self.assertIn("m-img-2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
