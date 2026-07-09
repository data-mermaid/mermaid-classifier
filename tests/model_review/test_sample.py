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

    def test_duplicate_rowcol_rows_are_deduped(self):
        # image A already has points (10,20) and (30,40); add a duplicate of (10,20)
        dup = "10,20,A,444,coralnet,2605-coralnet-public-sources,109,s109/features/iA.featurevector,ba1,gf1,val"
        path = _write_csv(extra_rows=[dup])
        pts = [p for p in sample.select_images(path, n_images=3, seed=1) if p.image_id == "A"]
        self.assertEqual(sorted((p.row, p.col) for p in pts), [(10, 20), (30, 40)])  # not 3

    def test_min_points_per_image_filters_sparse_images(self):
        # A has 2 points, B and C have 1 each; require >=2 -> only A eligible.
        path = _write_csv()
        pts = sample.select_images(path, n_images=10, seed=1, min_points_per_image=2)
        self.assertEqual({p.image_id for p in pts}, {"A"})

    def test_non_coralnet_rows_excluded_and_do_not_crash(self):
        # The raw val CSV mixes MERMAID rows whose feature key would not parse;
        # the default coralnet site filter must skip them without raising.
        path = _write_csv(extra_rows=[_MERMAID_ROW])
        pts = sample.select_images(path, n_images=10, seed=1)  # site defaults to coralnet
        self.assertEqual({p.image_id for p in pts}, {"A", "B", "C"})


def _fake_mapper(coralnet_id):
    # cn "999" is unmappable; others map to "ba<id>::"
    return None if coralnet_id == "999" else f"ba{coralnet_id}::"


def _mrow(source_id, image_id, row, col, coralnet_id):
    return {
        "source_id": source_id,
        "image_id": image_id,
        "row": row,
        "col": col,
        "coralnet_id": coralnet_id,
    }


class FullGtTest(unittest.TestCase):
    def test_eligible_ids_are_distinct_coralnet_images(self):
        path = _write_csv(extra_rows=[_MERMAID_ROW])
        self.assertEqual(sample.eligible_image_ids_from_val(path), {"A", "B", "C"})

    def test_manifest_rows_mapped_deduped_and_unmapped_dropped(self):
        rows = [
            _mrow(7, "X", 1, 1, "444"),
            _mrow(7, "X", 1, 1, "444"),  # duplicate (row,col) -> deduped
            _mrow(7, "X", 2, 2, "999"),  # unmappable -> dropped
            _mrow(7, "X", 3, 3, "12"),
        ]
        by_image = sample.review_points_from_manifest_rows(rows, _fake_mapper, "feat-bucket")
        pts = by_image["X"]
        self.assertEqual(
            sorted((p.row, p.col, p.gt_bagf) for p in pts), [(1, 1, "ba444::"), (3, 3, "ba12::")]
        )
        self.assertEqual(pts[0].feature_key, "s7/features/iX.featurevector")
        self.assertEqual(pts[0].bucket, "feat-bucket")

    def test_sample_full_gt_filters_and_returns_all_points(self):
        by_image = {
            "big": [sample.ReviewPoint("big", "1", r, r, "ba::", "k", "b") for r in range(20)],
            "small": [sample.ReviewPoint("small", "1", 0, 0, "ba::", "k", "b")],
        }
        pts = sample.sample_full_gt_images(by_image, n_images=5, seed=1, min_points_per_image=15)
        self.assertEqual({p.image_id for p in pts}, {"big"})  # 'small' excluded (<15)
        self.assertEqual(len(pts), 20)  # all points of the selected image
