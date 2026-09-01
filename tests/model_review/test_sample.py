import random
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


# A MERMAID row: its feature key would NOT parse as a CoralNet source id.
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


def _mmrow(image_id, row, col, ba_id, gf_id=None):
    return {
        "image_id": image_id,
        "row": row,
        "col": col,
        "benthic_attribute_id": ba_id,
        "growth_form_id": gf_id,
    }


def _point(image_id, row=0, site=sample.CORALNET):
    return sample.ReviewPoint(site, image_id, "1", row, row, "ba::", "k", "b")


class EligibilityTest(unittest.TestCase):
    def test_eligible_ids_are_grouped_by_site(self):
        path = _write_csv(extra_rows=[_MERMAID_ROW])
        self.assertEqual(
            sample.eligible_image_ids_by_site(path),
            {"coralnet": {"A", "B", "C"}, "mermaid": {"M"}},
        )


class RowMappingTest(unittest.TestCase):
    def test_coralnet_rows_mapped_deduped_and_unmapped_dropped(self):
        rows = [
            _mrow(7, "X", 1, 1, "444"),
            _mrow(7, "X", 1, 1, "444"),  # duplicate (row,col) -> deduped
            _mrow(7, "X", 2, 2, "999"),  # unmappable -> dropped
            _mrow(7, "X", 3, 3, "12"),
        ]
        pts = sample.coralnet_points_from_manifest_rows(rows, _fake_mapper, "feat-bucket")["X"]
        self.assertEqual(
            sorted((p.row, p.col, p.gt_bagf) for p in pts), [(1, 1, "ba444::"), (3, 3, "ba12::")]
        )
        self.assertEqual(pts[0].site, "coralnet")
        self.assertEqual(pts[0].feature_key, "s7/features/iX.featurevector")
        self.assertEqual(pts[0].bucket, "feat-bucket")

    def test_unmappable_point_does_not_block_a_later_mappable_duplicate(self):
        # (2,2) first appears unmappable; the mappable duplicate must still count.
        rows = [_mrow(7, "X", 2, 2, "999"), _mrow(7, "X", 2, 2, "12")]
        pts = sample.coralnet_points_from_manifest_rows(rows, _fake_mapper, "b")["X"]
        self.assertEqual([(p.row, p.col, p.gt_bagf) for p in pts], [(2, 2, "ba12::")])

    def test_mermaid_rows_carry_site_and_mermaid_feature_key(self):
        rows = [_mmrow("uuid-1", 5, 6, "ba9", "gf3")]
        pts = sample.mermaid_points_from_manifest_rows(
            rows, lambda bagf: bagf, "coral-reef-training"
        )
        point = pts["uuid-1"][0]
        self.assertEqual(point.site, "mermaid")
        self.assertEqual(point.source_id, "")  # MERMAID has no source
        self.assertEqual(point.gt_bagf, "ba9::gf3")
        self.assertEqual(point.feature_key, "mermaid/uuid-1_featurevector")
        self.assertEqual(point.bucket, "coral-reef-training")

    def test_mermaid_absent_growth_form_normalizes_to_empty(self):
        # The parquet stores SQL NULL (-> None, or NaN via pandas); dataset.py sees 'None'.
        for absent in (None, float("nan"), "None", ""):
            with self.subTest(absent=absent):
                rows = [_mmrow("i", 1, 1, "ba9", absent)]
                pts = sample.mermaid_points_from_manifest_rows(rows, lambda b: b, "b")["i"]
                self.assertEqual(pts[0].gt_bagf, "ba9::")

    def test_mermaid_unmappable_label_is_dropped(self):
        rows = [_mmrow("i", 1, 1, "ba9"), _mmrow("i", 2, 2, "nope")]
        keep = lambda bagf: None if bagf.startswith("nope") else bagf  # noqa: E731
        pts = sample.mermaid_points_from_manifest_rows(rows, keep, "b")["i"]
        self.assertEqual([(p.row, p.col) for p in pts], [(1, 1)])


class QuotaTest(unittest.TestCase):
    WEIGHTS = {"coralnet": 2.0, "mermaid": 1.0}

    def test_two_to_one_over_twenty_images(self):
        self.assertEqual(sample.allocate_quota(20, self.WEIGHTS), {"coralnet": 13, "mermaid": 7})

    def test_exact_thirds_when_divisible(self):
        self.assertEqual(sample.allocate_quota(21, self.WEIGHTS), {"coralnet": 14, "mermaid": 7})

    def test_always_sums_to_n_images(self):
        for n in range(0, 61):
            self.assertEqual(sum(sample.allocate_quota(n, self.WEIGHTS).values()), n, n)

    def test_single_seat_goes_to_the_larger_weight(self):
        self.assertEqual(sample.allocate_quota(1, self.WEIGHTS), {"coralnet": 1, "mermaid": 0})

    def test_even_split_ties_break_by_name(self):
        even = {"coralnet": 1.0, "mermaid": 1.0}
        self.assertEqual(sample.allocate_quota(3, even), {"coralnet": 2, "mermaid": 1})

    def test_non_positive_weights_rejected(self):
        with self.assertRaises(ValueError):
            sample.allocate_quota(10, {"coralnet": 0.0})


class StratumSeedTest(unittest.TestCase):
    def test_is_stable_across_processes(self):
        # Pinned literal: this is the cross-process reproducibility contract. It fails
        # loudly if someone swaps blake2b for PYTHONHASHSEED-salted hash().
        self.assertEqual(sample.stratum_seed(1, "coralnet"), 17947860225498774239)

    def test_sites_get_distinct_streams(self):
        self.assertNotEqual(sample.stratum_seed(1, "coralnet"), sample.stratum_seed(1, "mermaid"))

    def test_seed_changes_the_stream(self):
        self.assertNotEqual(sample.stratum_seed(1, "coralnet"), sample.stratum_seed(2, "coralnet"))


class StratumDrawTest(unittest.TestCase):
    """`select_stratum` takes an injected fetcher, so none of this touches S3."""

    N_CANDIDATES = 200

    def setUp(self):
        # Every 4th image is too sparse to qualify.
        self.ids = [f"img{i:03d}" for i in range(self.N_CANDIDATES)]
        self.qualified = {i for n, i in enumerate(self.ids) if n % 4}
        self.fetched_batches = []

    def fetch(self, image_ids):
        self.fetched_batches.append(list(image_ids))
        return {
            i: [_point(i, row=r) for r in range(20 if i in self.qualified else 3)]
            for i in image_ids
        }

    def draw(self, quota=10, seed=7, **kw):
        pts = sample.select_stratum(
            self.ids, self.fetch, quota=quota, seed=seed, min_points_per_image=15, **kw
        )
        return sorted({p.image_id for p in pts})

    def _expected(self, quota, seed):
        order = sorted(self.ids)
        random.Random(seed).shuffle(order)
        return sorted([i for i in order if i in self.qualified][:quota])

    def test_takes_the_first_qualifiers_in_permutation_order(self):
        self.assertEqual(self.draw(quota=10, seed=7), self._expected(10, 7))

    def test_result_is_invariant_to_batch_size(self):
        # If the keep-loop ever iterates the fetched mapping instead of the batch's
        # permutation order, batch size becomes load-bearing and this fails.
        results = [self.draw(quota=10, batch_size=b) for b in (1, 3, 7, 64, 10_000)]
        self.assertEqual(len(set(map(tuple, results))), 1, results)

    def test_result_is_invariant_to_candidate_input_order(self):
        # Candidates arrive in DuckDB GROUP BY order, which is not stable.
        baseline = self.draw()
        self.ids = list(reversed(self.ids))
        self.assertEqual(self.draw(), baseline)
        random.Random(99).shuffle(self.ids)
        self.assertEqual(self.draw(), baseline)

    def test_selection_is_nested_in_quota(self):
        self.assertTrue(set(self.draw(quota=5)).issubset(self.draw(quota=10)))

    def test_returns_every_point_of_each_chosen_image_sorted(self):
        pts = sample.select_stratum(self.ids, self.fetch, quota=2, seed=7, min_points_per_image=15)
        self.assertEqual(len(pts), 40)  # 2 images x 20 points
        by_image = {}
        for p in pts:
            by_image.setdefault(p.image_id, []).append((p.row, p.col))
        for rowcols in by_image.values():
            self.assertEqual(rowcols, sorted(rowcols))

    def test_keep_image_predicate_narrows_the_qualified_pool(self):
        # Stands in for "this image has no display object in S3".
        blocked = self._expected(1, 7)[0]
        drawn = sample.select_stratum(
            self.ids,
            self.fetch,
            quota=10,
            seed=7,
            min_points_per_image=15,
            keep_image=lambda p: p.image_id != blocked,
        )
        self.assertNotIn(blocked, {p.image_id for p in drawn})
        self.assertEqual(len({p.image_id for p in drawn}), 10)

    def test_exhaustion_raises_instead_of_returning_short(self):
        with self.assertRaises(ValueError) as ctx:
            sample.select_stratum(
                self.ids, self.fetch, quota=999, seed=7, min_points_per_image=15, site="mermaid"
            )
        self.assertIn("mermaid", str(ctx.exception))
        self.assertIn("999", str(ctx.exception))

    def test_zero_quota_draws_nothing_and_reads_nothing(self):
        self.assertEqual(sample.select_stratum(self.ids, self.fetch, 0, 7, 15), [])
        self.assertEqual(self.fetched_batches, [])

    def test_changing_one_sites_weight_does_not_move_anothers_draw(self):
        # A site's seed is a function of (seed, site) only, so its permutation is fixed
        # when another site's weight changes; only its quota moves, and the draw nests.
        thirteen_seven = sample.allocate_quota(20, {"coralnet": 2.0, "mermaid": 1.0})
        sixteen_four = sample.allocate_quota(20, {"coralnet": 4.0, "mermaid": 1.0})
        self.assertNotEqual(thirteen_seven["coralnet"], sixteen_four["coralnet"])

        cn_seed = sample.stratum_seed(1, "coralnet")
        smaller = set(self.draw(quota=thirteen_seven["coralnet"], seed=cn_seed))
        larger = set(self.draw(quota=sixteen_four["coralnet"], seed=cn_seed))
        self.assertTrue(smaller.issubset(larger))
