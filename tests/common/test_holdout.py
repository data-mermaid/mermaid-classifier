import unittest

from mermaid_classifier.common.holdout import is_cv_fold


class IsCvFoldTest(unittest.TestCase):
    def test_is_cv_fold(self):
        cases = [
            ("0", True),
            ("12", True),
            ("test", False),
            ("", False),
            ("-1", False),
            ("1.0", False),
            (" 1", False),
            ("١", False),  # Arabic-Indic digit one
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertIs(is_cv_fold(value), expected)


if __name__ == "__main__":
    unittest.main()
