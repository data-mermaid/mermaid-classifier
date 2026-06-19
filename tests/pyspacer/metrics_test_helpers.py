"""Shared test mocks and helpers for metrics test modules."""

from collections import defaultdict

import numpy as np
from spacer.data_classes import ValResults


class MockBALibrary:
    """Mock with a simple 2-level taxonomy tree.

    Tree:
    A (root) -> A1, A2
    B (root) -> B1, B2
    """

    def __init__(self):
        self.by_id = {
            'A': {'id': 'A', 'name': 'TopA', 'parent': None},
            'A1': {'id': 'A1', 'name': 'ChildA1', 'parent': 'A'},
            'A2': {'id': 'A2', 'name': 'ChildA2', 'parent': 'A'},
            'B': {'id': 'B', 'name': 'TopB', 'parent': None},
            'B1': {'id': 'B1', 'name': 'ChildB1', 'parent': 'B'},
            'B2': {'id': 'B2', 'name': 'ChildB2', 'parent': 'B'},
        }
        self.by_parent = defaultdict(list)
        for ba in self.by_id.values():
            self.by_parent[ba['parent']].append(ba)

    def get_ancestor_ids(self, ba_id):
        parent = self.by_id[ba_id]['parent']
        if parent is None:
            return []
        return self.get_ancestor_ids(parent) + [parent]

    def id_to_name(self, ba_id):
        if ba_id == '':
            return ''
        return self.by_id[ba_id]['name']

    def bagf_id_to_name(self, bagf_id, gf_library):
        return f'name_{bagf_id}'

    def get_descendants(self, ba_id):
        if ba_id not in self.by_parent:
            return []
        children = self.by_parent[ba_id]
        result = []
        for child in sorted(children, key=lambda c: c['name']):
            result.extend(self.get_descendants(child['id']))
        return children + result


class MockGFLibrary:
    """Mock of GrowthFormLibrary for testing."""

    def __init__(self):
        self.by_id = {'gf1': 'Branching', 'gf2': 'Massive', 'gf3': 'Encrusting'}

    def id_to_name(self, gf_id):
        if gf_id == '':
            return ''
        return self.by_id[gf_id]


class MockClf:
    """Mock classifier with classes_ attribute."""
    def __init__(self, classes):
        self.classes_ = np.array(classes)


def make_val_results(gt_indices, est_indices, classes, scores=None):
    """Helper to build a ValResults with optional scores."""
    if scores is None:
        scores = [1.0] * len(gt_indices)
    return ValResults(scores=scores, gt=gt_indices, est=est_indices, classes=classes)


def format_metric(value):
    return round(float(value), 3)
