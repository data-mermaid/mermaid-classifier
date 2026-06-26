"""Guards that the train/eval/store path stays free of the classifier pickle.

If train.py or trainer.py re-imports pyspacer's classifier store/load/train
glue, the pickle round-trip has crept back onto the critical path. Scoped to
these two files: scripts/evaluate_model.py is intentionally still pickle-based
(out of scope, tracked as #61) and must not trip this guard.
"""
import ast
import unittest
from pathlib import Path

# mermaid_classifier/pyspacer/ — two levels up from tests/pyspacer/.
PYSPACER_DIR = (
    Path(__file__).resolve().parents[2]
    / "mermaid_classifier" / "pyspacer"
)

# Pickle-glue symbols that must never reach the train/eval/store path,
# whether imported by name (``from spacer.storage import load_classifier``)
# or reached as an attribute (``import spacer.storage as s; s.load_classifier``).
FORBIDDEN = {"load_classifier", "train_classifier", "store_classifier",
             "TrainClassifierMsg"}

# Whole modules whose only purpose on this path is the pickle glue. Importing
# them at all in train.py/trainer.py is a re-entry signal, even before any use.
FORBIDDEN_MODULES = {"spacer.storage", "spacer.tasks"}


def _glue_references(source: str) -> set[str]:
    """Return every pickle-glue symbol/module this source reaches.

    Catches three re-entry shapes:
      1. ``from spacer.storage import load_classifier`` (imported name)
      2. ``import spacer.storage`` / ``import spacer.tasks`` (glue module)
      3. ``import spacer.storage as s; s.load_classifier(...)`` (attribute
         access on a module alias) — the gap the name-only guard missed.
    """
    offenders: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN:
                    offenders.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN or alias.name in FORBIDDEN_MODULES:
                    offenders.add(alias.name)
        elif isinstance(node, ast.Attribute):
            # e.g. ``storage.load_classifier`` or
            # ``spacer.storage.store_classifier``.
            if node.attr in FORBIDDEN:
                offenders.add(node.attr)
    return offenders


class PickleFreeTrainingTest(unittest.TestCase):
    def test_train_py_does_not_import_pickle_glue(self):
        source = (PYSPACER_DIR / "train.py").read_text()
        offenders = _glue_references(source)
        self.assertEqual(
            offenders, set(),
            msg=f"train.py must not reference {offenders} (pickle path)")

    def test_trainer_py_does_not_import_pickle_glue(self):
        source = (PYSPACER_DIR / "trainer.py").read_text()
        offenders = _glue_references(source)
        self.assertEqual(
            offenders, set(),
            msg=f"trainer.py must not reference {offenders} (pickle path)")


if __name__ == "__main__":
    unittest.main()
