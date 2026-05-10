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

    def test_init_creates_header(self):
        ar.init_results_tsv()
        content = ar.RESULTS_TSV.read_text()
        self.assertIn("id\t", content)
        self.assertIn("hypothesis", content)
        self.assertIn("balanced_accuracy", content)

    def test_append_and_read(self):
        ar.init_results_tsv()
        ar.append_result({
            "id": 1,
            "timestamp": "2026-05-09T22:00:00",
            "hypothesis": "baseline",
            "balanced_accuracy": "0.4523",
            "best_so_far": "0.4523",
            "status": "KEPT",
            "duration_s": "1200",
            "commit_sha": "abc1234",
            "error": "",
        })

        content = ar.read_results_tsv()
        self.assertIn("baseline", content)
        self.assertIn("0.4523", content)
        self.assertIn("KEPT", content)

    def test_next_experiment_id(self):
        ar.init_results_tsv()
        self.assertEqual(ar.next_experiment_id(), 1)

        ar.append_result({
            "id": 1, "timestamp": "", "hypothesis": "test",
            "balanced_accuracy": "", "best_so_far": "",
            "status": "", "duration_s": "", "commit_sha": "", "error": "",
        })
        self.assertEqual(ar.next_experiment_id(), 2)

    def test_init_is_idempotent(self):
        ar.init_results_tsv()
        ar.append_result({
            "id": 1, "timestamp": "", "hypothesis": "test",
            "balanced_accuracy": "", "best_so_far": "",
            "status": "", "duration_s": "", "commit_sha": "", "error": "",
        })
        # Calling init again should NOT overwrite.
        ar.init_results_tsv()
        self.assertEqual(ar.next_experiment_id(), 2)


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

        self.assertEqual(result["hypothesis"], "test dropout")
        self.assertEqual(len(result["file_changes"]), 1)

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
