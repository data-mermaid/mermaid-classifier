import tempfile
import unittest

from mermaid_classifier.model_review import sample

_HEADER = "row,col,image_id,label_id,site,bucket,project_id,feature_vector,benthic_attribute_id,growth_form_id,training_set"
_ROWS = [
    # img A, 2 points
    "10,20,A,444,coralnet,2605-coralnet-public-sources,109,s109/features/iA.featurevector,ba1,gf1,val",
    "30,40,A,444,coralnet,2605-coralnet-public-sources,109,s109/features/iA.featurevector,ba1,,val",
    # img B, 1 point
    "50,60,B,7,coralnet,2605-coralnet-public-sources,7,s7/features/iB.featurevector,ba2,,val",
    # img C, 1 point
    "70,80,C,9,coralnet,2605-coralnet-public-sources,9,s9/features/iC.featurevector,ba3,gf2,val",
]


def _write_csv(extra_rows: list[str] | None = None) -> str:
    rows = _ROWS + (extra_rows or [])
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(_HEADER + "\n" + "\n".join(rows) + "\n")
        return tmp.name


# A MERMAID row whose feature key would NOT parse as a CoralNet source id.
_MERMAID_ROW = "5,6,M,0,mermaid,coral-reef-training,,mermaid/abc_featurevector,ba9,,val"


class SampleTest(unittest.TestCase):
    def test_source_id_parsing(self):
        self.assertEqual(sample.source_id_from_feature_key("s109/features/iA.featurevector"), "109")

    def test_selects_requested_image_count_deterministically(self):
        path = _write_csv()
        a = sample.select_images(path, n_images=2, seed=1)
        b = sample.select_images(path, n_images=2, seed=1)
        self.assertEqual([p.image_id for p in a], [p.image_id for p in b])
        self.assertEqual(len({p.image_id for p in a}), 2)

    def test_returns_all_points_for_selected_images_sorted(self):
        path = _write_csv()
        # force selection of image A by choosing a seed that includes it, or select all 3
        pts = sample.select_images(path, n_images=3, seed=1)
        a_points = [(p.row, p.col, p.gt_bagf) for p in pts if p.image_id == "A"]
        self.assertEqual(a_points, [(10, 20, "ba1::gf1"), (30, 40, "ba1::")])

    def test_gt_bagf_uses_double_colon_and_empty_gf(self):
        path = _write_csv()
        pts = sample.select_images(path, n_images=3, seed=1)
        by = {p.image_id: p for p in pts}
        self.assertEqual(by["B"].gt_bagf, "ba2::")

    def test_non_coralnet_rows_excluded_and_do_not_crash(self):
        # The raw val CSV mixes MERMAID rows whose feature key would not parse;
        # the default coralnet site filter must skip them without raising.
        path = _write_csv(extra_rows=[_MERMAID_ROW])
        pts = sample.select_images(path, n_images=10, seed=1)  # site defaults to coralnet
        self.assertEqual({p.image_id for p in pts}, {"A", "B", "C"})
