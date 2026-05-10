"""Tests for the autoresearch harness utilities.

Tests the hash verification, results tracking, git operations,
prompt construction, and file change application. Does NOT test
actual training or Claude API calls.
"""

import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add autoresearch to path for imports.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "autoresearch"))

import autoresearch as ar


class TestSHA256(unittest.TestCase):
    """Test the SHA256 hash helper."""

    def test_sha256_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("hello world")
            f.flush()
            path = Path(f.name)
        try:
            h1 = ar._sha256(path)
            h2 = ar._sha256(path)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)  # SHA256 hex digest length
        finally:
            path.unlink()

    def test_sha256_changes_with_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("content A")
            f.flush()
            path = Path(f.name)
        try:
            h1 = ar._sha256(path)
            path.write_text("content B")
            h2 = ar._sha256(path)
            self.assertNotEqual(h1, h2)
        finally:
            path.unlink()


class TestHashVerification(unittest.TestCase):
    """Test frozen file hash computation and verification."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_root = ar.ROOT_DIR
        self.orig_frozen_dirs = ar.FROZEN_DIRS
        self.orig_frozen_files = ar.FROZEN_FILES
        self.orig_hashes_file = ar.HASHES_FILE

        # Set up a mock frozen directory.
        ar.ROOT_DIR = Path(self.tmpdir)
        frozen_dir = Path(self.tmpdir) / "frozen_pkg"
        frozen_dir.mkdir()
        (frozen_dir / "module.py").write_text("x = 1")
        (frozen_dir / "utils.py").write_text("y = 2")
        ar.FROZEN_DIRS = [frozen_dir]
        ar.FROZEN_FILES = []
        ar.HASHES_FILE = Path(self.tmpdir) / "hashes.json"

    def tearDown(self):
        ar.ROOT_DIR = self.orig_root
        ar.FROZEN_DIRS = self.orig_frozen_dirs
        ar.FROZEN_FILES = self.orig_frozen_files
        ar.HASHES_FILE = self.orig_hashes_file

    def test_compute_and_verify_unchanged(self):
        hashes = ar.compute_frozen_hashes()
        ar.save_hashes(hashes)
        ok, changed = ar.verify_hashes()
        self.assertTrue(ok)
        self.assertEqual(changed, [])

    def test_verify_detects_modification(self):
        hashes = ar.compute_frozen_hashes()
        ar.save_hashes(hashes)

        # Modify a frozen file.
        frozen_file = ar.FROZEN_DIRS[0] / "module.py"
        frozen_file.write_text("x = 999  # modified!")

        ok, changed = ar.verify_hashes()
        self.assertFalse(ok)
        self.assertEqual(len(changed), 1)
        self.assertIn("module.py", changed[0])

    def test_verify_detects_new_file(self):
        hashes = ar.compute_frozen_hashes()
        ar.save_hashes(hashes)

        # Add a new file.
        (ar.FROZEN_DIRS[0] / "new_file.py").write_text("z = 3")

        ok, changed = ar.verify_hashes()
        self.assertFalse(ok)
        self.assertTrue(any("new_file.py" in c for c in changed))


class TestResultsTracking(unittest.TestCase):
    """Test results.tsv creation and appending."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_results = ar.RESULTS_TSV
        ar.RESULTS_TSV = Path(self.tmpdir) / "results.tsv"

    def tearDown(self):
        ar.RESULTS_TSV = self.orig_results

    def _row(self, **overrides) -> dict:
        row = {f: "" for f in ar.RESULTS_FIELDS}
        row.update(overrides)
        return row

    def test_init_creates_header(self):
        ar.init_results_tsv()
        content = ar.RESULTS_TSV.read_text()
        self.assertIn("id\t", content)
        self.assertIn("hypothesis", content)
        self.assertIn("balanced_accuracy", content)
        # New headline columns must appear on a fresh init.
        for key in ar.HEADLINE_METRIC_KEYS:
            self.assertIn(key, content)
        self.assertIn("analysis_excerpt", content)

    def test_append_and_read(self):
        ar.init_results_tsv()
        ar.append_result(self._row(
            id=1,
            timestamp="2026-05-09T22:00:00",
            hypothesis="baseline",
            balanced_accuracy="0.4523",
            best_so_far="0.4523",
            status="KEPT",
            duration_s="1200",
            commit_sha="abc1234",
        ))

        content = ar.read_results_tsv()
        self.assertIn("baseline", content)
        self.assertIn("0.4523", content)
        self.assertIn("KEPT", content)

    def test_next_experiment_id(self):
        ar.init_results_tsv()
        self.assertEqual(ar.next_experiment_id(), 1)

        ar.append_result(self._row(id=1, hypothesis="test"))
        self.assertEqual(ar.next_experiment_id(), 2)

    def test_init_is_idempotent(self):
        ar.init_results_tsv()
        ar.append_result(self._row(id=1, hypothesis="test"))
        # Calling init again should NOT overwrite.
        ar.init_results_tsv()
        self.assertEqual(ar.next_experiment_id(), 2)


class TestResultsTsvMigration(unittest.TestCase):
    """Test that an old (pre-headline-columns) results.tsv migrates cleanly."""

    OLD_FIELDS = [
        "id", "timestamp", "hypothesis", "balanced_accuracy",
        "best_so_far", "status", "duration_s", "commit_sha", "error",
    ]

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_results = ar.RESULTS_TSV
        ar.RESULTS_TSV = Path(self.tmpdir) / "results.tsv"
        # Write an old-style results.tsv with two baseline rows.
        with open(ar.RESULTS_TSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.OLD_FIELDS, delimiter="\t")
            w.writeheader()
            w.writerow({
                "id": "1", "timestamp": "2026-05-10T00:09:11+00:00",
                "hypothesis": "baseline", "balanced_accuracy": "0.779000",
                "best_so_far": "0.779000", "status": "KEPT",
                "duration_s": "4037", "commit_sha": "396a401", "error": "",
            })
            w.writerow({
                "id": "2", "timestamp": "2026-05-10T02:03:39+00:00",
                "hypothesis": "baseline", "balanced_accuracy": "0.780000",
                "best_so_far": "0.780000", "status": "KEPT",
                "duration_s": "3874", "commit_sha": "83bc6d1", "error": "",
            })

    def tearDown(self):
        ar.RESULTS_TSV = self.orig_results

    def test_migration_expands_header(self):
        ar.init_results_tsv()
        with open(ar.RESULTS_TSV) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            rows = list(reader)
        self.assertEqual(header, ar.RESULTS_FIELDS)
        self.assertEqual(len(rows), 2)

    def test_migration_preserves_existing_values(self):
        ar.init_results_tsv()
        with open(ar.RESULTS_TSV) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        self.assertEqual(rows[0]["balanced_accuracy"], "0.779000")
        self.assertEqual(rows[0]["commit_sha"], "396a401")
        # New columns are padded with empty string.
        self.assertEqual(rows[0]["mcc"], "")
        self.assertEqual(rows[0]["analysis_excerpt"], "")

    def test_migration_is_idempotent(self):
        ar.init_results_tsv()
        first = ar.RESULTS_TSV.read_text()
        ar.init_results_tsv()
        second = ar.RESULTS_TSV.read_text()
        self.assertEqual(first, second)


class TestPromptConstruction(unittest.TestCase):
    """Test Claude prompt building."""

    def test_build_user_prompt_basic(self):
        prompt = ar.build_user_prompt(
            experiment_files={"train_experiment.py": "x = 1", "classifier.py": "y = 2"},
            results_tsv="id\thypo\n1\tbaseline",
            recent_diffs="diff --git a/...",
        )
        self.assertIn("train_experiment.py", prompt)
        self.assertIn("classifier.py", prompt)
        self.assertIn("baseline", prompt)
        self.assertIn("Recent Diffs", prompt)
        self.assertIn("Headline Metrics History", prompt)

    def test_build_user_prompt_with_error(self):
        prompt = ar.build_user_prompt(
            experiment_files={},
            results_tsv="",
            recent_diffs="",
            last_error="RuntimeError: OOM",
        )
        self.assertIn("Last Experiment Error", prompt)
        self.assertIn("OOM", prompt)

    def test_build_user_prompt_without_error(self):
        prompt = ar.build_user_prompt(
            experiment_files={},
            results_tsv="",
            recent_diffs="",
            last_error=None,
        )
        self.assertNotIn("Last Experiment Error", prompt)

    def test_build_user_prompt_includes_telemetry_and_analysis(self):
        prompt = ar.build_user_prompt(
            experiment_files={},
            results_tsv="",
            recent_diffs="",
            prior_telemetry=[
                (3, "## Headline metrics\nbalanced_accuracy=0.81"),
                (2, "## Headline metrics\nbalanced_accuracy=0.78"),
            ],
            prior_analyses=[
                (3, "Recall on Pectinia is 0.4"),
                (2, "Calibration gap at bin 5 is 0.18"),
            ],
        )
        self.assertIn("Last 2 Experiments — Full Telemetry", prompt)
        self.assertIn("### Experiment 3", prompt)
        self.assertIn("balanced_accuracy=0.81", prompt)
        self.assertIn("Last 2 Experiments — Analysis", prompt)
        self.assertIn("Pectinia", prompt)
        self.assertIn("Calibration gap", prompt)

    def test_build_user_prompt_omits_telemetry_when_empty(self):
        prompt = ar.build_user_prompt(
            experiment_files={},
            results_tsv="",
            recent_diffs="",
            prior_telemetry=[],
            prior_analyses=[],
        )
        self.assertNotIn("Last 2 Experiments — Full Telemetry", prompt)
        self.assertNotIn("Last 2 Experiments — Analysis", prompt)


class TestTelemetryPersistence(unittest.TestCase):
    """Test telemetry / analysis file persistence and recent-artifact loading."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_telemetry = ar.TELEMETRY_DIR
        self.orig_analyses = ar.ANALYSES_DIR
        ar.TELEMETRY_DIR = Path(self.tmpdir) / "telemetry"
        ar.ANALYSES_DIR = Path(self.tmpdir) / "analyses"

    def tearDown(self):
        ar.TELEMETRY_DIR = self.orig_telemetry
        ar.ANALYSES_DIR = self.orig_analyses

    def test_write_telemetry_file_creates_dir(self):
        path = ar.write_telemetry_file(7, "# telemetry body")
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "7.md")
        self.assertEqual(path.read_text(), "# telemetry body")

    def test_write_analysis_file_creates_dir(self):
        path = ar.write_analysis_file(4, "Recall on X is low")
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), "Recall on X is low")

    def test_load_recent_artifacts_orders_descending(self):
        ar.write_telemetry_file(2, "two")
        ar.write_telemetry_file(5, "five")
        ar.write_telemetry_file(3, "three")
        recent = ar._load_recent_artifacts(ar.TELEMETRY_DIR, current_id=99, n=2)
        self.assertEqual([eid for eid, _ in recent], [5, 3])

    def test_load_recent_artifacts_skips_current_id(self):
        ar.write_telemetry_file(2, "two")
        ar.write_telemetry_file(5, "five")
        recent = ar._load_recent_artifacts(ar.TELEMETRY_DIR, current_id=5, n=2)
        self.assertEqual([eid for eid, _ in recent], [2])

    def test_load_recent_artifacts_when_dir_missing(self):
        # Directory does not yet exist — should return [].
        recent = ar._load_recent_artifacts(ar.TELEMETRY_DIR, current_id=1)
        self.assertEqual(recent, [])

    def test_analysis_excerpt_collapses_whitespace(self):
        excerpt = ar._analysis_excerpt(
            "Line one\n\n  with  spaces\nand  newlines.", limit=30)
        self.assertEqual(excerpt, "Line one with spaces and newli")
        self.assertNotIn("\n", excerpt)
        self.assertNotIn("\t", excerpt)


class TestFileChanges(unittest.TestCase):
    """Test applying file changes from Claude's response."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_experiment_dir = ar.EXPERIMENT_DIR
        self.orig_modifiable = ar.MODIFIABLE_FILES
        ar.EXPERIMENT_DIR = Path(self.tmpdir)
        ar.MODIFIABLE_FILES = [
            Path(self.tmpdir) / "train_experiment.py",
            Path(self.tmpdir) / "classifier.py",
            Path(self.tmpdir) / "trainer.py",
            Path(self.tmpdir) / "strategies.py",
        ]
        # Create initial files.
        for f in ar.MODIFIABLE_FILES:
            f.write_text("# original")

    def tearDown(self):
        ar.EXPERIMENT_DIR = self.orig_experiment_dir
        ar.MODIFIABLE_FILES = self.orig_modifiable

    def test_apply_valid_change(self):
        changes = [{"filename": "classifier.py", "content": "# modified"}]
        changed = ar.apply_file_changes(changes)
        self.assertEqual(changed, ["classifier.py"])
        self.assertEqual(
            (Path(self.tmpdir) / "classifier.py").read_text(), "# modified")

    def test_rejects_non_modifiable(self):
        changes = [{"filename": "settings.py", "content": "# evil"}]
        changed = ar.apply_file_changes(changes)
        self.assertEqual(changed, [])

    def test_multiple_changes(self):
        changes = [
            {"filename": "classifier.py", "content": "# new classifier"},
            {"filename": "trainer.py", "content": "# new trainer"},
        ]
        changed = ar.apply_file_changes(changes)
        self.assertEqual(set(changed), {"classifier.py", "trainer.py"})


class TestClaudeResponseParsing(unittest.TestCase):
    """Test parsing Claude CLI responses."""

    def test_parse_structured_output(self):
        """Test that structured_output from --json-schema is extracted."""
        cli_response = json.dumps({
            "result": "",
            "structured_output": {
                "analysis": "Recall on Pectinia is 0.4 with only 1 sample.",
                "hypothesis": "test dropout",
                "file_changes": [
                    {"filename": "classifier.py", "content": "# with dropout"}
                ],
            },
        })
        mock_result = MagicMock()
        mock_result.stdout = cli_response
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = ar.call_claude("model", "system", "user")

        self.assertIn("Pectinia", result["analysis"])
        self.assertEqual(result["hypothesis"], "test dropout")
        self.assertEqual(len(result["file_changes"]), 1)

    def test_response_schema_requires_analysis(self):
        """The response schema must include analysis as a required field."""
        self.assertIn("analysis", ar.RESPONSE_SCHEMA["required"])
        self.assertIn("analysis", ar.RESPONSE_SCHEMA["properties"])

    def test_cli_error_is_surfaced(self):
        """When the CLI reports is_error, the result message is raised."""
        cli_response = json.dumps({
            "is_error": True,
            "result": "Not logged in · Please run /login",
        })
        mock_result = MagicMock()
        mock_result.stdout = cli_response
        mock_result.stderr = ""
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            with self.assertRaises(RuntimeError) as ctx:
                ar.call_claude("model", "system", "user")

        self.assertIn("Not logged in", str(ctx.exception))


class TestTelemetryRendering(unittest.TestCase):
    """Test the artifact rendering helpers in autoresearch/telemetry.py.

    Avoids MLflow by exercising the private renderers directly with
    in-memory DataFrames. Skipped if pandas isn't importable (not the
    case in the pyspacer env, but keeps this file usable in other
    Python envs).
    """

    @classmethod
    def setUpClass(cls):
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("pandas not installed")
        import telemetry as t
        cls.t = t
        cls.pd = pd

    def test_render_metrics_per_label_picks_extremes(self):
        df = self.pd.DataFrame({
            "bagf_id": [f"id_{i}" for i in range(20)],
            "precision": [i / 20 for i in range(20)],
            "recall": [i / 20 for i in range(20)],
            "f1": [i / 20 for i in range(20)],
            "n_samples": [10 * i for i in range(20)],
        })
        out = self.t._render_metrics_per_label(df)
        self.assertIn("10 worst classes by recall", out)
        self.assertIn("10 best classes by recall", out)
        self.assertIn("Summary stats", out)

    def test_render_confusion_top_pairs(self):
        df = self.pd.DataFrame({
            "-": ["A", "B", "C"],
            "A": [90, 5, 0],
            "B": [3, 88, 12],
            "C": [7, 7, 88],
        })
        out = self.t._render_confusion_top_pairs(df)
        self.assertIn("Top-15 confused pairs", out)
        # Row 2 ("C") column "B" = 12: 12% of true class C was predicted as B.
        # The largest off-diagonal entry — sort the table puts it at top.
        first_data_line = [
            line for line in out.splitlines()
            if line.startswith("| ") and "%" in line
        ][0]
        self.assertEqual(first_data_line, "| C | B | 12.0% |")

    def test_render_per_source_top_bottom(self):
        df = self.pd.DataFrame({
            "source": [f"src{i}" for i in range(10)],
            "balanced_accuracy": [0.1 * i for i in range(10)],
        })
        out = self.t._render_per_source(df)
        self.assertIn("5 worst sources", out)
        self.assertIn("5 best sources", out)
        self.assertIn("src0", out)
        self.assertIn("src9", out)

    def test_render_error_attribution(self):
        df = self.pd.DataFrame({
            "lca_node": ["x", "y", "z"],
            "lca_name": ["X", "Y", "Z"],
            "pct_of_errors": [50.0, 30.0, 20.0],
        })
        out = self.t._render_error_attribution(df)
        self.assertIn("Top-10 LCA error nodes", out)
        self.assertIn("X", out)

    def test_extract_run_telemetry_with_mocked_mlflow(self):
        """Drive ``extract_run_telemetry`` against a mock MLflow surface."""
        import telemetry as t
        # Mock pandas-based search_runs row.
        run_row = self.pd.Series({
            "run_id": "abc123",
            "experiment_id": "13",
            "status": "FINISHED",
            "metrics.balanced_accuracy": 0.81,
            "metrics.mcc": 0.84,
            "metrics.ece": 0.09,
            "metrics.system/cpu_utilization_percentage": 42.0,
            "metrics.epoch/val_loss": 0.6,
            "params.learning_rate_init": "0.0001",
        })
        runs_df = self.pd.DataFrame([run_row])
        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = MagicMock(
            experiment_id="13")
        mock_mlflow.search_runs.return_value = runs_df

        # Mock client.get_metric_history to return one short curve.
        history_point = MagicMock(step=0, value=0.5)
        mock_client = MagicMock()
        mock_client.get_metric_history.return_value = [history_point]
        mock_mlflow_client = MagicMock(return_value=mock_client)

        # No artifacts available -> all artifact sections become "(unavailable)".
        mock_mlflow.artifacts.download_artifacts.side_effect = RuntimeError(
            "no artifact")

        with patch.dict(
            "sys.modules",
            {
                "mlflow": mock_mlflow,
                "mlflow.tracking": MagicMock(MlflowClient=mock_mlflow_client),
                "mlflow.artifacts": mock_mlflow.artifacts,
            },
        ):
            tel = t.extract_run_telemetry(experiment_name="autoresearch")

        self.assertIsNotNone(tel)
        self.assertEqual(tel.run_id, "abc123")
        self.assertEqual(tel.headline.get("balanced_accuracy"), 0.81)
        self.assertEqual(tel.headline.get("mcc"), 0.84)
        # System metrics are filtered out.
        self.assertNotIn(
            "system/cpu_utilization_percentage", tel.all_scalars)
        self.assertIn("learning_rate_init", tel.params)
        # Artifact sections rendered the "(unavailable)" notice.
        self.assertTrue(
            any("unavailable" in v for v in tel.artifact_sections.values()))
        self.assertIn("Headline metrics", tel.full_markdown)
        self.assertIn("balanced_accuracy", tel.full_markdown)


class TestCreateTrainerExtraction(unittest.TestCase):
    """Test that the _create_trainer extraction in TrainingRunner works."""

    def test_create_trainer_method_exists(self):
        from mermaid_classifier.pyspacer.train import TrainingRunner
        self.assertTrue(hasattr(TrainingRunner, '_create_trainer'))

    def test_create_trainer_returns_mermaid_trainer(self):
        from mermaid_classifier.pyspacer.train import TrainingRunner, TrainingOptions
        from mermaid_classifier.pyspacer.trainer import MermaidTrainer

        runner = TrainingRunner(
            training_options=TrainingOptions(
                hidden_layer_sizes=(100,),
                learning_rate_init=1e-3,
                epochs=1,
            )
        )
        trainer = runner._create_trainer(batch_size=100, class_weight=None)
        self.assertIsInstance(trainer, MermaidTrainer)
        self.assertEqual(trainer.batch_size, 100)
        self.assertEqual(trainer.hidden_layer_sizes, (100,))
        self.assertEqual(trainer.learning_rate_init, 1e-3)

    def test_create_trainer_can_be_overridden(self):
        from mermaid_classifier.pyspacer.train import TrainingRunner, TrainingOptions
        from mermaid_classifier.pyspacer.trainer import MermaidTrainer

        class CustomTrainer(MermaidTrainer):
            custom_flag = True

        class CustomRunner(TrainingRunner):
            def _create_trainer(self, batch_size, class_weight):
                return CustomTrainer(
                    batch_size=batch_size, class_weight=class_weight,
                    hidden_layer_sizes=(50,), learning_rate_init=1e-3)

        runner = CustomRunner(
            training_options=TrainingOptions(epochs=1))
        trainer = runner._create_trainer(batch_size=50, class_weight=None)
        self.assertIsInstance(trainer, CustomTrainer)
        self.assertTrue(trainer.custom_flag)


if __name__ == "__main__":
    unittest.main()
