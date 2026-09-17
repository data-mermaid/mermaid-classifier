"""Tests for scripts/generate_report.py.

Tests the pure transformation functions without requiring an MLflow server.
"""

import base64
import struct

# Adjust sys.path so we can import from scripts/.
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from generate_report import (
    _artifact_key,
    build_template_context,
    encode_png_as_base64,
    fetch_scalar_metrics,
    load_artifact_data,
    load_csv_as_html_table,
    load_yaml_file,
    render_report,
)

REGION_RUN_METRICS = {
    "accuracy": 0.85,
    "region_val/oor_rate": 0.0384,
    "region_val/oor_rate_lo95": 0.0371,
    "region_val/oor_rate_hi95": 0.0398,
    "region_val/gt_oor_rate": 0.0052,
    "region_val/gt_oor_rate_lo95": 0.0044,
    "region_val/gt_oor_rate_hi95": 0.0061,
    "region_probe/oor_rate": 0.1402,
    "region_probe/oor_rate_lo95": 0.1310,
    "region_probe/oor_rate_hi95": 0.1495,
}


def _make_minimal_png(path: Path):
    """Write a minimal valid 1x1 white PNG to the given path."""
    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # IHDR: 1x1, 8-bit grayscale
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)
    # IDAT: single white pixel, filter byte 0
    raw = b"\x00\xff"
    idat = _chunk(b"IDAT", zlib.compress(raw))
    # IEND
    iend = _chunk(b"IEND", b"")

    path.write_bytes(sig + ihdr + idat + iend)


class TestArtifactKey(unittest.TestCase):
    def test_simple_filename(self):
        self.assertEqual(_artifact_key("metrics_per_label.csv"), "metrics_per_label_csv")

    def test_subdirectory_path(self):
        self.assertEqual(_artifact_key("confusion_matrix/frequencies.png"), "frequencies_png")

    def test_yaml_extension(self):
        self.assertEqual(_artifact_key("system_specs.yaml"), "system_specs_yaml")


class TestEncodePngAsBase64(unittest.TestCase):
    def test_returns_data_uri(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "test.png"
            _make_minimal_png(png_path)
            result = encode_png_as_base64(png_path)
            self.assertTrue(result.startswith("data:image/png;base64,"))

    def test_base64_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "test.png"
            _make_minimal_png(png_path)
            result = encode_png_as_base64(png_path)
            b64_part = result.split(",", 1)[1]
            decoded = base64.b64decode(b64_part)
            # Should start with PNG signature.
            self.assertTrue(decoded.startswith(b"\x89PNG"))


class TestLoadCsvAsHtmlTable(unittest.TestCase):
    def test_basic_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("name,value\nalpha,0.95\nbeta,0.87\n")
            result = load_csv_as_html_table(csv_path)
            self.assertIn("<table", result)
            self.assertIn("alpha", result)
            self.assertIn("0.9500", result)
            self.assertIn("dataframe", result)

    def test_no_index_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("a,b\n1,2\n")
            result = load_csv_as_html_table(csv_path)
            # Should not contain a default pandas index column.
            self.assertNotIn("<th></th>", result)


class TestLoadYamlFile(unittest.TestCase):
    def test_basic_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text("total_ram_gb: 16.0\nfree_storage_gb: 100.5\n")
            result = load_yaml_file(yaml_path)
            self.assertEqual(result["total_ram_gb"], 16.0)
            self.assertEqual(result["free_storage_gb"], 100.5)

    def test_nested_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text("parent:\n  child: value\n")
            result = load_yaml_file(yaml_path)
            self.assertEqual(result["parent"]["child"], "value")


class TestFetchScalarMetrics(unittest.TestCase):
    def _make_mock_run(self, metrics_dict):
        run = MagicMock()
        run.data.metrics = metrics_dict
        return run

    def test_full_metrics(self):
        metrics = {
            "accuracy": 0.85,
            "balanced_accuracy": 0.82,
            "f1_macro": 0.80,
            "precision_macro": 0.81,
            "recall_macro": 0.79,
            "mcc": 0.75,
            "ece": 0.03,
            "log_loss": 1.2,
            "top_1_accuracy": 0.85,
            "top_3_accuracy": 0.95,
            "top_5_accuracy": 0.97,
            "top_10_accuracy": 0.99,
            "mrr": 0.90,
            "cover_mean_abs_bias_pct": 2.1,
            "cover_mean_rmse_pct": 3.5,
            "cross_branch_error_rate": 0.15,
        }
        run = self._make_mock_run(metrics)
        result = fetch_scalar_metrics(run)

        self.assertIsNotNone(result["executive"])
        self.assertEqual(len(result["executive"]), 8)
        self.assertIsNotNone(result["topk"])
        self.assertEqual(len(result["topk"]), 5)
        self.assertIsNotNone(result["cover"])
        self.assertEqual(len(result["cover"]), 2)  # Only 2 of 4 cover metrics present.
        self.assertIsNotNone(result["taxonomic"])

    def test_minimal_metrics(self):
        """Only executive metrics present, optional groups absent."""
        metrics = {
            "accuracy": 0.85,
            "f1_macro": 0.80,
        }
        run = self._make_mock_run(metrics)
        result = fetch_scalar_metrics(run)

        self.assertIsNotNone(result["executive"])
        self.assertEqual(len(result["executive"]), 2)
        self.assertIsNone(result["topk"])
        self.assertIsNone(result["cover"])
        self.assertIsNone(result["taxonomic"])

    def test_empty_metrics(self):
        run = self._make_mock_run({})
        result = fetch_scalar_metrics(run)
        self.assertIsNone(result["executive"])
        self.assertIsNone(result["topk"])


class TestRegionMetricGrouping(unittest.TestCase):
    """A region rate is unreadable without the ground-truth floor beside it,
    and a rate is unreadable without its interval."""

    def _make_mock_run(self, metrics_dict):
        run = MagicMock()
        run.data.metrics = metrics_dict
        return run

    def test_executive_summary_pairs_the_rate_with_its_floor(self):
        result = fetch_scalar_metrics(self._make_mock_run(REGION_RUN_METRICS))
        labels = [label for label, _value in result["executive"]]
        self.assertIn("Out-of-Region Rate", labels)
        self.assertIn("Ground-Truth Floor", labels)
        self.assertEqual(
            labels.index("Ground-Truth Floor") - labels.index("Out-of-Region Rate"),
            1,
            msg=f"the floor must sit next to the rate it explains; got {labels}",
        )

    def test_region_group_carries_both_populations(self):
        result = fetch_scalar_metrics(self._make_mock_run(REGION_RUN_METRICS))
        values = dict(result["region"])
        self.assertIn("Validation: Out-of-Region", values)
        self.assertIn("Probe: Out-of-Region", values)
        self.assertAlmostEqual(values["Validation: Out-of-Region"], 0.0384)
        self.assertAlmostEqual(values["Probe: Out-of-Region"], 0.1402)

    def test_a_run_with_no_region_metrics_has_no_region_group(self):
        result = fetch_scalar_metrics(self._make_mock_run({"accuracy": 0.85}))
        self.assertIsNone(result["region"])

    def test_interval_bounds_are_paired_with_the_rate_they_bound(self):
        result = fetch_scalar_metrics(self._make_mock_run(REGION_RUN_METRICS))
        self.assertEqual(result["intervals"]["Out-of-Region Rate"], (0.0371, 0.0398))
        self.assertEqual(result["intervals"]["Probe: Out-of-Region"], (0.1310, 0.1495))

    def test_a_rate_whose_bounds_were_not_logged_is_still_flagged_as_a_rate(self):
        """The rate must never render as a bare number; without bounds the
        report has to say the interval is missing rather than omit it."""
        result = fetch_scalar_metrics(self._make_mock_run({"region_val/oor_rate": 0.0384}))
        self.assertIn("Out-of-Region Rate", result["intervals"])
        self.assertIsNone(result["intervals"]["Out-of-Region Rate"])

    def test_a_metric_that_is_not_a_rate_has_no_interval_entry(self):
        result = fetch_scalar_metrics(self._make_mock_run(REGION_RUN_METRICS))
        self.assertNotIn("Accuracy", result["intervals"])


class TestLoadArtifactData(unittest.TestCase):
    def test_with_all_required_sections(self):
        """Create a minimal artifact tree with required sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)

            # Create confusion_matrix artifacts.
            cm_dir = artifact_dir / "confusion_matrix"
            cm_dir.mkdir()
            _make_minimal_png(cm_dir / "frequencies.png")
            _make_minimal_png(cm_dir / "percents.png")

            # Create calibration artifacts.
            cal_dir = artifact_dir / "calibration"
            cal_dir.mkdir()
            _make_minimal_png(cal_dir / "reliability_diagram.png")

            # Create taxonomic artifacts.
            tax_dir = artifact_dir / "taxonomic"
            tax_dir.mkdir()
            _make_minimal_png(tax_dir / "error_attribution.png")
            _make_minimal_png(tax_dir / "top_level_confusion.png")
            _make_minimal_png(tax_dir / "gf_confusion.png")

            result = load_artifact_data(artifact_dir)

            self.assertIn("confusion_matrix", result["sections"])
            self.assertIn("calibration", result["sections"])
            self.assertIn("taxonomic", result["sections"])
            # Optional sections should be absent.
            self.assertNotIn("cover", result["sections"])
            self.assertNotIn("probability", result["sections"])
            self.assertNotIn("ranking", result["sections"])

    def test_optional_section_present(self):
        """Cover section appears when its artifacts exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            cover_dir = artifact_dir / "cover"
            cover_dir.mkdir()
            _make_minimal_png(cover_dir / "per_class_bias.png")

            result = load_artifact_data(artifact_dir)
            self.assertIn("cover", result["sections"])
            self.assertIsNotNone(result["sections"]["cover"]["per_class_bias_png"])

    def test_region_sections_appear_when_their_tables_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            for section in ("region_val", "region_probe"):
                section_dir = artifact_dir / section
                section_dir.mkdir()
                (section_dir / "per_region.csv").write_text("region_id,oor_rate\nta,0.04\n")

            result = load_artifact_data(artifact_dir)
            self.assertIn("region_val", result["sections"])
            self.assertIn("region_probe", result["sections"])
            self.assertIsNotNone(result["sections"]["region_val"]["per_region_csv"])

    def test_region_sections_are_absent_when_the_run_has_no_region_tables(self):
        """A CoralNet-only run logs neither, and the report must render
        without them rather than with two empty shells."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "system_specs.yaml").write_text("total_ram_gb: 16\n")

            result = load_artifact_data(artifact_dir)
            self.assertNotIn("region_val", result["sections"])
            self.assertNotIn("region_probe", result["sections"])

    def test_training_artifacts(self):
        """Training artifacts are loaded when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "system_specs.yaml").write_text("total_ram_gb: 16\n")
            (artifact_dir / "train_summary.yaml").write_text("total_images: 5000\n")

            result = load_artifact_data(artifact_dir)
            self.assertTrue(result["has_training"])
            self.assertIsNotNone(result["training"]["system_specs_yaml"])
            self.assertEqual(result["training"]["system_specs_yaml"]["total_ram_gb"], 16)


class TestRegionSectionOrdering(unittest.TestCase):
    def test_region_sections_are_ordered_after_the_per_source_breakdown(self):
        context = build_template_context(
            {"run_id": "x", "run_name": "y", "experiment_name": "z"},
            {"executive": None, "topk": None, "cover": None, "taxonomic": None, "region": None},
            {"sections": {}, "root_eval": {}, "training": {}, "has_training": False},
        )
        order = context["section_order"]
        self.assertIn("region_val", order)
        self.assertIn("region_probe", order)
        self.assertGreater(order.index("region_val"), order.index("per_source"))
        self.assertGreater(order.index("region_probe"), order.index("region_val"))


class TestBuildTemplateContext(unittest.TestCase):
    def test_basic_context(self):
        metadata = {
            "run_id": "abc123",
            "run_name": "test",
            "experiment_name": "exp1",
        }
        metrics = {
            "executive": [("Accuracy", 0.85)],
            "topk": None,
            "cover": None,
            "taxonomic": None,
        }
        artifacts = {
            "sections": {},
            "root_eval": {"metrics_per_label_csv": None, "metrics_overall_yaml": None},
            "training": {},
            "has_training": False,
        }

        context = build_template_context(metadata, metrics, artifacts)
        self.assertEqual(context["title"], "Classifier Report - exp1 - test")
        self.assertIn("generated_at", context)
        self.assertEqual(context["metadata"], metadata)

    def test_custom_title(self):
        metadata = {"run_id": "x", "run_name": "y", "experiment_name": "z"}
        metrics = {"executive": None, "topk": None, "cover": None, "taxonomic": None}
        artifacts = {"sections": {}, "root_eval": {}, "training": {}, "has_training": False}

        context = build_template_context(metadata, metrics, artifacts, title="Custom Title")
        self.assertEqual(context["title"], "Custom Title")


class TestRenderReport(unittest.TestCase):
    def test_renders_valid_html(self):
        """Render with a minimal context and verify output is HTML."""
        context = {
            "title": "Test Report",
            "generated_at": "2025-01-01 00:00 UTC",
            "metadata": {
                "run_id": "abc123def456",
                "run_name": "test-run",
                "experiment_name": "test-experiment",
                "status": "FINISHED",
                "start_time": "2025-01-01 00:00 UTC",
                "end_time": "2025-01-01 01:00 UTC",
                "duration": "1h 0m 0s",
                "params": {"epochs": "10"},
                "tags": {},
            },
            "metrics": {
                "executive": [("Accuracy", 0.85), ("F1 (Macro)", 0.80)],
                "topk": None,
                "cover": None,
                "taxonomic": None,
            },
            "sections": {},
            "root_eval": {
                "metrics_per_label_csv": None,
                "metrics_overall_yaml": None,
            },
            "training": {},
            "has_training": False,
            "section_order": [
                "confusion_matrix",
                "calibration",
                "cover",
                "probability",
                "ranking",
                "taxonomic",
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            render_report(context, output_path)

            self.assertTrue(output_path.exists())
            html = output_path.read_text()
            self.assertIn("<!DOCTYPE html>", html)
            self.assertIn("Test Report", html)
            self.assertIn("80.0%", html)  # F1 (Macro) formatted as percentage
            self.assertIn("abc123def456", html)

    def test_conditional_sections_absent(self):
        """Sections with no data should not appear in output."""
        context = {
            "title": "Test",
            "generated_at": "2025-01-01",
            "metadata": {
                "run_id": "x",
                "run_name": "y",
                "experiment_name": "z",
                "status": "FINISHED",
                "start_time": "N/A",
                "end_time": "N/A",
                "duration": "N/A",
                "params": {},
                "tags": {},
            },
            "metrics": {"executive": None, "topk": None, "cover": None, "taxonomic": None},
            "sections": {},
            "root_eval": {"metrics_per_label_csv": None, "metrics_overall_yaml": None},
            "training": {},
            "has_training": False,
            "section_order": [
                "confusion_matrix",
                "calibration",
                "cover",
                "probability",
                "ranking",
                "taxonomic",
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            render_report(context, output_path)
            html = output_path.read_text()

            # No evaluation sections should be rendered.
            self.assertNotIn('id="confusion_matrix"', html)
            self.assertNotIn('id="cover"', html)
            self.assertNotIn('id="ranking"', html)
            # No training section.
            self.assertNotIn('id="run-configuration"', html)
            # No per-label detail.
            self.assertNotIn('id="per-label-detail"', html)


class TestRenderRegionReport(unittest.TestCase):
    """What a reader sees: never a rate without the floor it is judged
    against, and never a rate without its interval."""

    def _context(self, **overrides):
        context = {
            "title": "Region Report",
            "generated_at": "2025-01-01 00:00 UTC",
            "metadata": {
                "run_id": "abc123def456",
                "run_name": "test-run",
                "experiment_name": "test-experiment",
                "status": "FINISHED",
                "start_time": "N/A",
                "end_time": "N/A",
                "duration": "N/A",
                "params": {},
                "tags": {},
            },
            "metrics": {
                "executive": [
                    ("Accuracy", 0.85),
                    ("Out-of-Region Rate", 0.0384),
                    ("Ground-Truth Floor", 0.0052),
                ],
                "topk": None,
                "cover": None,
                "taxonomic": None,
                "region": [
                    ("Validation: Out-of-Region", 0.0384),
                    ("Probe: Out-of-Region", 0.1402),
                ],
                "intervals": {
                    "Out-of-Region Rate": (0.0371, 0.0398),
                    "Ground-Truth Floor": (0.0044, 0.0061),
                    "Validation: Out-of-Region": (0.0371, 0.0398),
                    "Probe: Out-of-Region": None,
                },
            },
            "sections": {
                "region_val": {
                    "title": "Region Mismatch \u2014 Validation Split",
                    "per_region_csv": "<table>val rows</table>",
                },
                "region_probe": {
                    "title": "Region Mismatch \u2014 Frozen Probe",
                    "per_label_csv": "<table>probe rows</table>",
                },
            },
            "root_eval": {"metrics_per_label_csv": None, "metrics_overall_yaml": None},
            "training": {},
            "has_training": False,
            "section_order": ["region_val", "region_probe"],
        }
        context.update(overrides)
        return context

    def _render(self, context) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            render_report(context, output_path)
            return output_path.read_text()

    def test_the_floor_is_rendered_beside_the_rate(self):
        html = self._render(self._context())
        self.assertIn("Out-of-Region Rate", html)
        self.assertIn("Ground-Truth Floor", html)
        self.assertIn("3.84%", html)
        self.assertIn("0.52%", html)

    def test_a_rate_is_never_rendered_without_its_interval(self):
        html = self._render(self._context())
        self.assertIn("3.71%", html)
        self.assertIn("3.98%", html)

    def test_a_rate_whose_bounds_are_missing_says_so_rather_than_reading_exact(self):
        html = self._render(self._context())
        self.assertIn("not logged", html)

    def test_both_region_sections_render_their_tables(self):
        html = self._render(self._context())
        self.assertIn('id="region_val"', html)
        self.assertIn('id="region_probe"', html)
        self.assertIn("val rows", html)
        self.assertIn("probe rows", html)

    def test_a_run_without_region_metrics_renders_without_the_region_blocks(self):
        html = self._render(
            self._context(
                metrics={
                    "executive": [("Accuracy", 0.85)],
                    "topk": None,
                    "cover": None,
                    "taxonomic": None,
                    "region": None,
                    "intervals": {},
                },
                sections={},
            )
        )
        self.assertNotIn('id="region_val"', html)
        self.assertNotIn('id="region_probe"', html)
        self.assertNotIn("Out-of-Region", html)
        self.assertIn("85.0%", html)


if __name__ == "__main__":
    unittest.main()
