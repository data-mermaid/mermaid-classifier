import unittest

from mermaid_classifier.model_review.comparison import COMPARISON_MODEL_VERSION, comparison_text

_PATHS = {
    "hc-branch::": "Hard coral :: Acroporidae :: Acropora :: Branching",
    "hc-branch-dup::": "Hard coral :: Acroporidae :: Acropora :: Branching",
    "hc::": "Hard coral :: Branching",
    "sand::": "Sand",
}


def _label_name(bagf):
    return _PATHS[bagf]


class ComparisonTextTest(unittest.TestCase):
    def _text(self, entries):
        return comparison_text(entries, _label_name)

    def test_one_line_per_set_in_the_given_order(self):
        text = self._text([("ground-truth", "hc-branch::"), ("v1", "hc::"), ("Beta", "sand::")])
        names = [line.split(":", 1)[0] for line in text.splitlines()]
        self.assertEqual(names, ["ground-truth", "v1", "Beta"])

    def test_reference_line_carries_no_marker(self):
        text = self._text([("ground-truth", "hc-branch::"), ("v1", "hc::")])
        self.assertNotIn("—", text.splitlines()[0])

    def test_matching_and_differing_sets_are_marked(self):
        text = self._text(
            [("ground-truth", "hc-branch::"), ("v1", "hc-branch::"), ("Beta", "sand::")]
        )
        lines = text.splitlines()
        self.assertTrue(lines[1].endswith("— match"))
        self.assertTrue(lines[2].endswith("— differs"))

    def test_marker_follows_the_rendered_label_not_the_bagf(self):
        # Two distinct BA::GF ids that render to the same path read as a match, so the
        # marker never contradicts the two lines on screen.
        text = self._text([("ground-truth", "hc-branch::"), ("v1", "hc-branch-dup::")])
        self.assertTrue(text.splitlines()[1].endswith("— match"))

    def test_fine_label_is_the_full_path(self):
        text = self._text([("ground-truth", "hc-branch::")])
        self.assertIn("Hard coral :: Acroporidae :: Acropora :: Branching", text)

    def test_extra_sets_need_no_format_change(self):
        # The reviewer's own label lands here once annotations exist.
        text = self._text(
            [("ground-truth", "hc-branch::"), ("v1", "hc::"), ("Beta", "hc::"), ("you", "sand::")]
        )
        self.assertEqual(len(text.splitlines()), 4)
        self.assertTrue(text.splitlines()[3].startswith("you: Sand"))

    def test_absent_reference_leaves_every_line_unmarked(self):
        text = self._text([("v1", "hc::"), ("Beta", "sand::")])
        self.assertNotIn("—", text)

    def test_model_version_is_the_tab_name(self):
        self.assertEqual(COMPARISON_MODEL_VERSION, "Comparison")
