"""Guards that mermaid-classifier stays free of the classifier pickle.

If any module under ``mermaid_classifier/`` or ``scripts/`` re-imports
pyspacer's classifier store/load/train glue, the pickle round-trip has crept
back in. This covers the whole codebase: the train/eval/store path and the
CLI scripts. (The last pickle consumer, ``scripts/evaluate_model.py``, was
removed in #61.)
"""
import ast
import unittest
from pathlib import Path

# Repo root — two levels up from tests/pyspacer/.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_DIRS = (REPO_ROOT / "mermaid_classifier", REPO_ROOT / "scripts")

# Pickle-glue symbols that must never reach the train/eval/store path,
# whether imported by name (``from spacer.storage import load_classifier``)
# or reached as an attribute (``import spacer.storage as s; s.load_classifier``).
FORBIDDEN = {"load_classifier", "train_classifier", "store_classifier",
             "TrainClassifierMsg"}

# Whole modules whose only purpose on this path is the pickle glue. Importing
# them at all is a re-entry signal, even before any use.
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
    def test_no_classifier_pickle_glue_referenced_anywhere(self):
        for base in SCANNED_DIRS:
            for path in base.rglob("*.py"):
                offenders = _glue_references(path.read_text())
                self.assertEqual(
                    offenders, set(),
                    msg=f"{path.relative_to(REPO_ROOT)} must not reference "
                        f"{offenders} (pickle path)")


if __name__ == "__main__":
    unittest.main()
