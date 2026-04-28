"""LabelTransformsOptions and TransformSpec validation tests."""
from __future__ import annotations

import unittest

from mermaid_classifier.training.label_transforms import (
    LabelTransformsOptions,
    TransformSpec,
)


class TransformSpecTest(unittest.TestCase):
    def test_basic_construction(self):
        spec = TransformSpec(name="drop_rare", params={"min_count": 50})
        self.assertEqual(spec.name, "drop_rare")
        self.assertEqual(spec.params, {"min_count": 50})

    def test_empty_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "name"):
            TransformSpec(name="", params={})

    def test_non_string_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "name"):
            TransformSpec(name=42, params={})  # type: ignore[arg-type]

    def test_non_dict_params_rejected(self):
        with self.assertRaisesRegex(ValueError, "params"):
            TransformSpec(name="drop_rare", params=[("min_count", 50)])  # type: ignore[arg-type]


class LabelTransformsOptionsTest(unittest.TestCase):
    def test_disabled_default(self):
        opts = LabelTransformsOptions()
        self.assertFalse(opts.enabled)
        self.assertEqual(opts.pipeline, [])

    def test_enabled_requires_pipeline(self):
        with self.assertRaisesRegex(ValueError, "pipeline"):
            LabelTransformsOptions(enabled=True, pipeline=[])

    def test_pipeline_accepts_dicts(self):
        opts = LabelTransformsOptions(
            enabled=True,
            pipeline=[{"name": "drop_rare", "params": {"min_count": 50}}],
        )
        self.assertEqual(len(opts.pipeline), 1)
        self.assertIsInstance(opts.pipeline[0], TransformSpec)
        self.assertEqual(opts.pipeline[0].name, "drop_rare")

    def test_pipeline_accepts_tuples(self):
        opts = LabelTransformsOptions(
            enabled=True,
            pipeline=[("drop_rare", {"min_count": 50})],
        )
        self.assertEqual(len(opts.pipeline), 1)
        self.assertEqual(opts.pipeline[0].params["min_count"], 50)

    def test_invalid_pipeline_entry_rejected(self):
        with self.assertRaisesRegex(ValueError, "pipeline"):
            LabelTransformsOptions(
                enabled=True,
                pipeline=["drop_rare"],  # bare string is not a spec
            )

    def test_to_log_dict_disabled(self):
        opts = LabelTransformsOptions()
        d = opts.to_log_dict()
        self.assertEqual(d["label_transforms/enabled"], False)
        self.assertEqual(d["label_transforms/n_stages"], 0)

    def test_to_log_dict_flat_per_stage(self):
        opts = LabelTransformsOptions(
            enabled=True,
            pipeline=[
                TransformSpec(name="drop_rare", params={"min_count": 50}),
                TransformSpec(name="merge_rare", params={"min_count": 20}),
            ],
        )
        d = opts.to_log_dict()
        self.assertTrue(d["label_transforms/enabled"])
        self.assertEqual(d["label_transforms/n_stages"], 2)
        self.assertEqual(d["label_transforms/0/name"], "drop_rare")
        self.assertEqual(d["label_transforms/0/min_count"], 50)
        self.assertEqual(d["label_transforms/1/name"], "merge_rare")
        self.assertEqual(d["label_transforms/1/min_count"], 20)


if __name__ == "__main__":
    unittest.main()
