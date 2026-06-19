"""Tests for mermaid_classifier.pyspacer.metrics._taxonomy_helpers."""

import unittest

from pyspacer.metrics_test_helpers import MockBALibrary

from mermaid_classifier.pyspacer.metrics._taxonomy_helpers import (
    build_ba_paths,
    build_ba_to_top,
    find_lca,
    group_by_top_level,
    taxonomic_similarity,
    top_level_ancestor,
)


class TopLevelAncestorTest(unittest.TestCase):
    """Tests for top_level_ancestor()."""

    def setUp(self):
        self.lib = MockBALibrary()

    def test_leaf_node_returns_root(self):
        self.assertEqual(top_level_ancestor('A1', self.lib), 'A')

    def test_other_leaf_node_returns_root(self):
        self.assertEqual(top_level_ancestor('B2', self.lib), 'B')

    def test_root_node_returns_itself(self):
        self.assertEqual(top_level_ancestor('A', self.lib), 'A')

    def test_other_root_node_returns_itself(self):
        self.assertEqual(top_level_ancestor('B', self.lib), 'B')


class BuildBaToTopTest(unittest.TestCase):
    """Tests for build_ba_to_top()."""

    def setUp(self):
        self.lib = MockBALibrary()

    def test_maps_leaf_classes_to_roots(self):
        classes = ['A1::', 'A2::', 'B1::']
        result = build_ba_to_top(classes, self.lib)
        self.assertEqual(result['A1'], 'A')
        self.assertEqual(result['A2'], 'A')
        self.assertEqual(result['B1'], 'B')

    def test_maps_root_classes_to_themselves(self):
        classes = ['A::', 'B::']
        result = build_ba_to_top(classes, self.lib)
        self.assertEqual(result['A'], 'A')
        self.assertEqual(result['B'], 'B')

    def test_deduplicates_ba_ids(self):
        classes = ['A1::', 'A1::gf1', 'A1::gf2']
        result = build_ba_to_top(classes, self.lib)
        self.assertEqual(list(result.keys()), ['A1'])

    def test_classes_with_growth_forms(self):
        classes = ['B1::gf1', 'B2::gf2']
        result = build_ba_to_top(classes, self.lib)
        self.assertEqual(result['B1'], 'B')
        self.assertEqual(result['B2'], 'B')

    def test_all_classes_present(self):
        classes = ['A1::', 'A2::', 'B1::', 'B2::']
        result = build_ba_to_top(classes, self.lib)
        self.assertEqual(set(result.keys()), {'A1', 'A2', 'B1', 'B2'})


class BuildBaPathsTest(unittest.TestCase):
    """Tests for build_ba_paths()."""

    def setUp(self):
        self.lib = MockBALibrary()

    def test_leaf_path_is_root_to_leaf(self):
        classes = ['A1::']
        result = build_ba_paths(classes, self.lib)
        self.assertEqual(result['A1'], ['A', 'A1'])

    def test_root_path_is_just_root(self):
        classes = ['A::']
        result = build_ba_paths(classes, self.lib)
        self.assertEqual(result['A'], ['A'])

    def test_multiple_classes_different_trees(self):
        classes = ['A1::', 'B2::']
        result = build_ba_paths(classes, self.lib)
        self.assertEqual(result['A1'], ['A', 'A1'])
        self.assertEqual(result['B2'], ['B', 'B2'])

    def test_deduplicates_ba_ids(self):
        classes = ['A1::', 'A1::gf1']
        result = build_ba_paths(classes, self.lib)
        self.assertEqual(list(result.keys()), ['A1'])

    def test_all_leaves_present(self):
        classes = ['A1::', 'A2::', 'B1::', 'B2::']
        result = build_ba_paths(classes, self.lib)
        self.assertEqual(set(result.keys()), {'A1', 'A2', 'B1', 'B2'})


class FindLcaTest(unittest.TestCase):
    """Tests for find_lca()."""

    def setUp(self):
        self.lib = MockBALibrary()
        self.ba_paths = {
            'A': ['A'],
            'A1': ['A', 'A1'],
            'A2': ['A', 'A2'],
            'B': ['B'],
            'B1': ['B', 'B1'],
            'B2': ['B', 'B2'],
        }

    def test_siblings_return_parent(self):
        lca = find_lca('A1', 'A2', self.ba_paths)
        self.assertEqual(lca, 'A')

    def test_same_node_returns_itself(self):
        lca = find_lca('A1', 'A1', self.ba_paths)
        self.assertEqual(lca, 'A1')

    def test_root_and_leaf_same_branch_returns_root(self):
        lca = find_lca('A', 'A1', self.ba_paths)
        self.assertEqual(lca, 'A')

    def test_different_branches_returns_none(self):
        lca = find_lca('A1', 'B1', self.ba_paths)
        self.assertIsNone(lca)

    def test_different_roots_returns_none(self):
        lca = find_lca('A', 'B', self.ba_paths)
        self.assertIsNone(lca)


class TaxonomicSimilarityTest(unittest.TestCase):
    """Tests for taxonomic_similarity()."""

    def setUp(self):
        self.lib = MockBALibrary()
        self.ba_paths = {
            'A': ['A'],
            'A1': ['A', 'A1'],
            'A2': ['A', 'A2'],
            'B': ['B'],
            'B1': ['B', 'B1'],
            'B2': ['B', 'B2'],
        }

    def test_same_node_returns_one(self):
        sim = taxonomic_similarity('A1', 'A1', self.ba_paths, self.lib)
        self.assertEqual(sim, 1.0)

    def test_siblings_return_expected_fraction(self):
        # shared_depth = len(ancestors of 'A') + 1 = 0 + 1 = 1
        # max_depth = max(len(['A', 'A1']), len(['A', 'A2'])) = 2
        # expected = 1 / 2 = 0.5
        sim = taxonomic_similarity('A1', 'A2', self.ba_paths, self.lib)
        self.assertAlmostEqual(sim, 0.5)

    def test_different_branches_return_zero(self):
        sim = taxonomic_similarity('A1', 'B1', self.ba_paths, self.lib)
        self.assertEqual(sim, 0.0)

    def test_different_roots_return_zero(self):
        sim = taxonomic_similarity('A', 'B', self.ba_paths, self.lib)
        self.assertEqual(sim, 0.0)

    def test_symmetry(self):
        sim_ab = taxonomic_similarity('A1', 'A2', self.ba_paths, self.lib)
        sim_ba = taxonomic_similarity('A2', 'A1', self.ba_paths, self.lib)
        self.assertAlmostEqual(sim_ab, sim_ba)

    def test_root_siblings_return_expected_fraction(self):
        # shared_depth = len(ancestors of 'B') + 1 = 0 + 1 = 1
        # max_depth = max(len(['B', 'B1']), len(['B', 'B2'])) = 2
        # expected = 1 / 2 = 0.5
        sim = taxonomic_similarity('B1', 'B2', self.ba_paths, self.lib)
        self.assertAlmostEqual(sim, 0.5)


class GroupByTopLevelTest(unittest.TestCase):
    """Tests for group_by_top_level()."""

    def setUp(self):
        self.lib = MockBALibrary()
        # classes: 0=A1::, 1=A2::, 2=B1::, 3=B2::
        self.classes = ['A1::', 'A2::', 'B1::', 'B2::']
        self.ba_to_top = {'A1': 'A', 'A2': 'A', 'B1': 'B', 'B2': 'B'}

    def test_groups_by_top_level_ba(self):
        # gt: samples 0,1 -> A1 (top=A); samples 2,3 -> B1 (top=B)
        gt_indices = [0, 0, 2, 2]
        sample_indices = list(range(4))
        groups = group_by_top_level(
            sample_indices, gt_indices, self.classes,
            self.ba_to_top, self.lib, min_samples=1,
        )
        top_ba_ids = {g['top_ba_id'] for g in groups}
        self.assertEqual(top_ba_ids, {'A', 'B'})

    def test_group_has_correct_fields(self):
        gt_indices = [0, 0, 2, 2]
        sample_indices = list(range(4))
        groups = group_by_top_level(
            sample_indices, gt_indices, self.classes,
            self.ba_to_top, self.lib, min_samples=1,
        )
        for group in groups:
            self.assertIn('top_ba_id', group)
            self.assertIn('name', group)
            self.assertIn('indices', group)
            self.assertIn('n_samples', group)

    def test_group_name_from_library(self):
        gt_indices = [0, 0]
        sample_indices = [0, 1]
        groups = group_by_top_level(
            sample_indices, gt_indices, self.classes,
            self.ba_to_top, self.lib, min_samples=1,
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['name'], 'TopA')

    def test_min_samples_filters_small_groups(self):
        # Only 2 samples for A, 1 sample for B
        gt_indices = [0, 0, 2]
        sample_indices = [0, 1, 2]
        groups = group_by_top_level(
            sample_indices, gt_indices, self.classes,
            self.ba_to_top, self.lib, min_samples=2,
        )
        top_ba_ids = {g['top_ba_id'] for g in groups}
        self.assertIn('A', top_ba_ids)
        self.assertNotIn('B', top_ba_ids)

    def test_n_samples_matches_indices_length(self):
        gt_indices = [0, 0, 0, 2, 2]
        sample_indices = list(range(5))
        groups = group_by_top_level(
            sample_indices, gt_indices, self.classes,
            self.ba_to_top, self.lib, min_samples=1,
        )
        groups_by_id = {g['top_ba_id']: g for g in groups}
        self.assertEqual(groups_by_id['A']['n_samples'], 3)
        self.assertEqual(groups_by_id['A']['n_samples'],
                         len(groups_by_id['A']['indices']))
        self.assertEqual(groups_by_id['B']['n_samples'], 2)

    def test_empty_sample_indices_returns_no_groups(self):
        groups = group_by_top_level(
            [], [], self.classes,
            self.ba_to_top, self.lib, min_samples=1,
        )
        self.assertEqual(groups, [])

    def test_all_below_min_samples_returns_no_groups(self):
        gt_indices = [0, 2]
        sample_indices = [0, 1]
        groups = group_by_top_level(
            sample_indices, gt_indices, self.classes,
            self.ba_to_top, self.lib, min_samples=30,
        )
        self.assertEqual(groups, [])


if __name__ == '__main__':
    unittest.main()
