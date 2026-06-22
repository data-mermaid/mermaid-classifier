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
    def test_train_py_does_not_import_pickle_glue(self):
        source = (PYSPACER_DIR / "train.py").read_text()
        offenders = FORBIDDEN & _imported_names(source)
        self.assertEqual(
            offenders, set(),
            msg=f"train.py must not import {offenders} (pickle path)")

    def test_trainer_py_does_not_import_pickle_glue(self):
        source = (PYSPACER_DIR / "trainer.py").read_text()
        offenders = FORBIDDEN & _imported_names(source)
        self.assertEqual(
            offenders, set(),
            msg=f"trainer.py must not import {offenders} (pickle path)")


if __name__ == "__main__":
    unittest.main()
