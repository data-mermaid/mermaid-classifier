"""Tests for holdout evaluation and the metrics prefix it relies on."""

import types
import unittest
from unittest import mock

import duckdb
from spacer.data_classes import DataLocation, ImageLabels

from mermaid_classifier.pyspacer import holdout_eval
from mermaid_classifier.pyspacer.metrics import MetricsCoordinator
from pyspacer.metrics_test_helpers import (
    MockBALibrary,
    MockClf,
    MockGFLibrary,
    format_metric,
    make_ctx,
)

COORD = "mermaid_classifier.pyspacer.metrics.coordinator"

# (feature key, site, project, [(row, col, bagf)])
IMAGES = [
    ("cn/img1", "coralnet", "10", [(1, 1, "A1::"), (2, 2, "A2::"), (3, 3, "A1::")]),
    ("cn/img2", "coralnet", "10", [(1, 1, "A2::"), (2, 2, "B2::")]),  # B2 unknown
    ("mm/img3", "mermaid", "p1", [(1, 1, "A1::"), (2, 2, "A1::")]),
]


def _fake_evaluate_classifier(clf, labels):
    """Predict A1 for every point, walking images and points in order."""
    gts = [label for loc in labels.keys() for _r, _c, label in labels[loc]]  # noqa: SIM118 -- ImageLabels is not a dict
    return gts, ["A1::"] * len(gts), [0.9] * len(gts)


def _make_dataset():
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE annotations (bucket VARCHAR, feature_vector VARCHAR, row INTEGER,"
        " col INTEGER, site VARCHAR, project_id VARCHAR, benthic_attribute_id VARCHAR,"
        " growth_form_id VARCHAR, training_set VARCHAR, is_holdout BOOLEAN,"
        " is_excluded BOOLEAN)"
    )
    labels = ImageLabels()
    loc_to_source = {}
    path_to_s3 = {}
    for key, site, project, annotations in IMAGES:
        loc = DataLocation("filesystem", f"/tmp/{key}")
        labels.add_image(loc, annotations)
        loc_to_source[loc] = (site, project)
        path_to_s3[loc.key] = ("bkt", key)
        for row, col, bagf in annotations:
            ba, gf = bagf.split("::")
            conn.execute(
                "INSERT INTO annotations VALUES ('bkt', ?, ?, ?, ?, ?, ?, ?, 'holdout', TRUE, FALSE)",
                [key, row, col, site, project, ba, gf],
            )
    # A train row that must not appear in the holdout CSV.
    conn.execute(
        "INSERT INTO annotations VALUES ('bkt', 'tr/img', 1, 1, 'coralnet', '9', 'A1', '',"
        " 'train', FALSE, FALSE)"
    )
    return types.SimpleNamespace(
        holdout_labels=labels,
        duck_conn=conn,
        feature_loc_to_source=loc_to_source,
        _feature_path_to_s3_location=path_to_s3,
    )


class PrefixTest(unittest.TestCase):
    def _run(self, **kwargs):
        with (
            mock.patch(f"{COORD}.mlflow") as m_mlflow,
            mock.patch(f"{COORD}.log_dataframe") as m_df,
        ):
            ctx = make_ctx([0, 1, 2, 0, 1, 2, 0], [0, 1, 2, 1, 0, 2, 0], ["A1::", "A2::", "B1::"])
            MetricsCoordinator(ctx, duckdb.connect(), **kwargs).compute_and_log_all()
        metrics = [c.args[0] for c in m_mlflow.log_metric.call_args_list]
        dicts = [c.args[1] for c in m_mlflow.log_dict.call_args_list]
        frames = [c.args[2] for c in m_df.call_args_list]
        figures = [c.args[1] for c in m_mlflow.log_figure.call_args_list]
        return metrics, dicts + frames + figures

    def test_unprefixed_names_unchanged(self):
        metrics, paths = self._run()
        self.assertIn("precision_macro", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("metrics_overall.yaml", paths)
        self.assertIn("metrics_per_label", paths)

    def test_prefix_applies_to_every_name_and_path(self):
        metrics, paths = self._run(prefix="holdout/")
        self.assertIn("holdout/precision_macro", metrics)
        self.assertIn("holdout/metrics_overall.yaml", paths)
        self.assertTrue(all(n.startswith("holdout/") for n in metrics + paths))


class HoldoutEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_dataset()
        self.clf = MockClf(["A1::", "A2::"])
        patcher = mock.patch.object(holdout_eval, "evaluate_classifier", _fake_evaluate_classifier)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_unknown_class_points_flagged_and_excluded(self):
        ev = holdout_eval.evaluate_holdout(self.clf, self.dataset)
        self.assertEqual(len(ev.points), 7)
        self.assertEqual(ev.n_unknown_class, 1)
        assert ev.val_results is not None
        self.assertEqual(len(ev.val_results.gt), 6)
        self.assertEqual(sum(len(ev.known_labels[k]) for k in ev.known_labels.keys()), 6)  # noqa: SIM118

    def test_annotations_one_row_per_point_with_predictions(self):
        ev = holdout_eval.evaluate_holdout(self.clf, self.dataset)
        df = holdout_eval.holdout_annotations_df(self.dataset.duck_conn, ev.points)
        self.assertEqual(len(df), 7)
        self.assertNotIn("is_holdout", df.columns)
        self.assertNotIn("is_excluded", df.columns)
        self.assertTrue((df["training_set"] == "holdout").all())
        self.assertTrue((df["pred_bagf"] == "A1::").all())
        unknown = df[df["benthic_attribute_id"] == "B2"]
        self.assertEqual(len(unknown), 1)
        self.assertFalse(bool(unknown["in_model_classes"].iloc[0]))
        self.assertEqual(int(df["in_model_classes"].sum()), 6)

    def test_per_site_metrics_split_by_site(self):
        ev = holdout_eval.evaluate_holdout(self.clf, self.dataset)
        m = holdout_eval.per_site_metrics(ev.points, format_metric)
        # coralnet known points: A1, A2, A1, A2 -> predicted A1 -> 2/4 correct.
        self.assertEqual(m["coralnet/accuracy"], 0.5)
        self.assertEqual(m["coralnet/macro_recall"], 0.5)
        self.assertEqual(m["mermaid/accuracy"], 1.0)
        self.assertEqual(m["mermaid/macro_recall"], 1.0)

    def test_site_without_rows_is_skipped(self):
        ev = holdout_eval.evaluate_holdout(self.clf, self.dataset)
        only_cn = ev.points[ev.points["site"] == "coralnet"]
        m = holdout_eval.per_site_metrics(only_cn, format_metric)
        self.assertNotIn("mermaid/accuracy", m)
        self.assertIn("coralnet/accuracy", m)

    def test_log_holdout_evaluation_logs_namespaced_results(self):
        with (
            mock.patch.object(holdout_eval, "mlflow") as m_mlflow,
            mock.patch.object(holdout_eval, "log_dataframe") as m_df,
            mock.patch(f"{COORD}.mlflow") as c_mlflow,
            mock.patch(f"{COORD}.log_dataframe"),
        ):
            holdout_eval.log_holdout_evaluation(
                dataset=self.dataset,
                clf_calibrated=self.clf,
                predictor=None,
                ba_library=MockBALibrary(),
                gf_library=MockGFLibrary(),
                format_func=format_metric,
            )
        logged = {c.args[0]: c.args[1] for c in m_mlflow.log_metric.call_args_list}
        self.assertEqual(logged["holdout/points_unknown_class"], 1.0)
        self.assertEqual(logged["holdout/coralnet/accuracy"], 0.5)
        self.assertEqual(m_df.call_args.args[2], "annotations_holdout")
        coord_metrics = [c.args[0] for c in c_mlflow.log_metric.call_args_list]
        self.assertIn("holdout/precision_macro", coord_metrics)
        self.assertTrue(all(n.startswith("holdout/") for n in coord_metrics))

    def test_no_holdout_rows_logs_nothing(self):
        self.dataset.holdout_labels = ImageLabels()
        with (
            mock.patch.object(holdout_eval, "mlflow") as m_mlflow,
            mock.patch.object(holdout_eval, "log_dataframe") as m_df,
            mock.patch.object(holdout_eval, "evaluate_classifier") as m_eval,
        ):
            holdout_eval.log_holdout_evaluation(
                dataset=self.dataset,
                clf_calibrated=self.clf,
                predictor=None,
                ba_library=MockBALibrary(),
                gf_library=MockGFLibrary(),
                format_func=format_metric,
            )
        m_mlflow.log_metric.assert_not_called()
        m_df.assert_not_called()
        m_eval.assert_not_called()


if __name__ == "__main__":
    unittest.main()
