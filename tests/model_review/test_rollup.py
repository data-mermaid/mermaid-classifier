import unittest

from mermaid_classifier.model_review import rollup


class RollupTest(unittest.TestCase):
    def setUp(self):
        self.roll = rollup.make_rollup_fn(strict=False)
        self.toplevel = rollup.load_toplevel()

    def test_toplevel_has_twelve_categories(self):
        self.assertEqual(len(self.toplevel), 12)

    def test_toplevel_id_rolls_to_itself(self):
        some_top_id = next(iter(self.toplevel))
        self.assertEqual(self.roll(f"{some_top_id}::"), some_top_id)

    def test_unmapped_returns_none_when_not_strict(self):
        self.assertIsNone(self.roll("not-a-real-ba::not-a-gf"))

    def test_strict_raises_on_unmapped(self):
        strict = rollup.make_rollup_fn(strict=True)
        with self.assertRaises(KeyError):
            strict("not-a-real-ba::not-a-gf")
