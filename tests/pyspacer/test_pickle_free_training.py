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

FORBIDDEN = {"load_classifier", "train_classifier", "store_classifier",
             "TrainClassifierMsg"}


def _imported_names(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


class PickleFreeTrainingTest(unittest.TestCase):
    def test_no_classifier_pickle_glue_imported_anywhere(self):
        for base in SCANNED_DIRS:
            for path in base.rglob("*.py"):
                offenders = FORBIDDEN & _imported_names(path.read_text())
                self.assertEqual(
                    offenders, set(),
                    msg=f"{path.relative_to(REPO_ROOT)} must not import "
                        f"{offenders} (pickle path)")


if __name__ == "__main__":
    unittest.main()
