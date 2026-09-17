"""Unit tests for region_eval/probe_set.py.

The fixture corpus is three regions of four-point images, sized so every
allocation number below can be derived by hand:

    region                 images  eligible  policy
    Tropical Atlantic          6         3   census (all 6, held_out False)
    Central Indo-Pacific      45        40   sample (held_out True)
    Western Indo-Pacific       8         8   sample (held_out True)

An image is eligible when at least one of its four ground-truth annotations
carries a region-discriminating benthic attribute. The Atlantic census takes
the three ineligible images too, which is the composition decision the probe
set exists to hold fixed.

With target_images=20 and region_floor=5: the census takes 6, leaving 14 to
split over eligible pools of 40 and 8. Largest-remainder over 48 gives
14*40//48 = 11 (remainder .67) and 14*8//48 = 2 (remainder .33), so the one
leftover image goes to Central Indo-Pacific -> 12 and 2. The floor then lifts
Western Indo-Pacific to 5. Realized: 6 + 12 + 5 = 23 images, 92 points.

Nothing here touches S3 or the network; the only I/O is a local parquet
written into a temp dir.
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mermaid_classifier.region_eval.metrics import required_n_for_detection
from mermaid_classifier.region_eval.probe_set import (
    PROBE_COLUMNS,
    NameSnapshot,
    ProbeSelectionOptions,
    ancestry_snapshot,
    ancestry_snapshot_hash,
    ancestry_snapshot_json,
    build_manifest,
    build_probe_set,
    ground_truth_counts_hash,
    ground_truth_counts_json,
    minimum_detectable_effect_table,
    name_snapshot_hash,
    name_snapshot_json,
    probe_content_hash,
    read_annotations,
    region_snapshot,
    region_snapshot_hash,
)

TROPICAL_ATLANTIC = "3c3c3c3c-0000-4000-8000-000000000001"
CENTRAL_INDO_PACIFIC = "3c3c3c3c-0000-4000-8000-000000000002"
WESTERN_INDO_PACIFIC = "3c3c3c3c-0000-4000-8000-000000000003"

REGION_NAMES = {
    TROPICAL_ATLANTIC: "Tropical Atlantic",
    CENTRAL_INDO_PACIFIC: "Central Indo-Pacific",
    WESTERN_INDO_PACIFIC: "Western Indo-Pacific",
}
ALL_REGIONS = frozenset(REGION_NAMES)

BA_GLOBAL = "4d4d4d4d-0000-4000-8000-000000000001"
BA_PACIFIC = "4d4d4d4d-0000-4000-8000-000000000002"
BA_UNRECORDED = "4d4d4d4d-0000-4000-8000-000000000003"

REGION_IDS_BY_ATTRIBUTE = {
    BA_GLOBAL: ALL_REGIONS,
    BA_PACIFIC: frozenset({CENTRAL_INDO_PACIFIC, WESTERN_INDO_PACIFIC}),
    BA_UNRECORDED: frozenset(),
}

POINTS_PER_IMAGE = 4

BA_ROOT = "4d4d4d4d-0000-4000-8000-000000000000"
GF_BRANCHING = "5e5e5e5e-0000-4000-8000-000000000001"

NAMES = NameSnapshot(
    benthic_attributes={
        BA_ROOT: "Hard coral",
        BA_GLOBAL: "Porites",
        BA_PACIFIC: "Acropora",
        BA_UNRECORDED: "Sand",
    },
    growth_forms={GF_BRANCHING: "Branching"},
    regions=REGION_NAMES,
)

# Root-to-leaf paths, root first and ending in the attribute itself, which is
# the shape `within_branch_share` reads.
ANCESTRY = {
    BA_ROOT: [BA_ROOT],
    BA_GLOBAL: [BA_ROOT, BA_GLOBAL],
    BA_PACIFIC: [BA_ROOT, BA_PACIFIC],
    BA_UNRECORDED: [BA_UNRECORDED],
}


def _image_rows(image_id: str, region_id: str, *, eligible: bool) -> list[dict[str, object]]:
    """Four annotation rows for one image.

    An eligible image carries exactly one region-discriminating attribute, so
    a selection that kept only the discriminating points would return one row
    per image instead of four.
    """
    attributes = [BA_PACIFIC if eligible else BA_GLOBAL, BA_GLOBAL, BA_UNRECORDED, BA_GLOBAL]
    return [
        {
            "id": f"{image_id}-a{position}",
            "image_id": image_id,
            "point_id": f"{image_id}-p{position}",
            "row": 10 * (position + 1),
            "col": 20 * (position + 1),
            "benthic_attribute_id": attribute,
            "benthic_attribute_name": f"BA-{attribute[-1]}",
            "growth_form_id": "",
            "growth_form_name": "",
            "updated_on": "2026-01-01T00:00:00Z",
            "region_id": region_id,
            "region_name": REGION_NAMES[region_id],
            "site_id": "",
        }
        for position, attribute in enumerate(attributes)
    ]


def _annotations(
    *,
    atlantic_eligible: int = 3,
    atlantic_plain: int = 3,
    central_eligible: int = 40,
    central_plain: int = 5,
    western_eligible: int = 8,
) -> pd.DataFrame:
    """A normalized annotation frame, in the shape read_annotations emits."""
    rows: list[dict[str, object]] = []
    for index in range(atlantic_eligible):
        rows += _image_rows(f"ta-e{index:03d}", TROPICAL_ATLANTIC, eligible=True)
    for index in range(atlantic_plain):
        rows += _image_rows(f"ta-p{index:03d}", TROPICAL_ATLANTIC, eligible=False)
    for index in range(central_eligible):
        rows += _image_rows(f"cip-e{index:03d}", CENTRAL_INDO_PACIFIC, eligible=True)
    for index in range(central_plain):
        rows += _image_rows(f"cip-p{index:03d}", CENTRAL_INDO_PACIFIC, eligible=False)
    for index in range(western_eligible):
        rows += _image_rows(f"wip-e{index:03d}", WESTERN_INDO_PACIFIC, eligible=True)
    return pd.DataFrame(rows)


def _options(**overrides: object) -> ProbeSelectionOptions:
    defaults: dict[str, object] = {"seed": 0, "target_images": 20, "region_floor": 5}
    defaults.update(overrides)
    return ProbeSelectionOptions(**defaults)  # pyright: ignore[reportArgumentType]


def _build(annotations: pd.DataFrame | None = None, **overrides: object):
    return build_probe_set(
        _annotations() if annotations is None else annotations,
        region_ids_by_attribute=REGION_IDS_BY_ATTRIBUTE,
        options=_options(**overrides),
    )


def _stratum(probe, region_id: str):
    return next(entry for entry in probe.strata if entry.region_id == region_id)


def _images(probe, prefix: str = "") -> set[str]:
    selected = set(probe.rows["image_id"])
    return {image_id for image_id in selected if image_id.startswith(prefix)}


class DeterminismTest(unittest.TestCase):
    def test_same_seed_gives_an_identical_probe_row_set(self):
        first = _build(seed=7)
        second = _build(seed=7)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            list(first.rows.itertuples(index=False)),
            list(second.rows.itertuples(index=False)),
        )

    def test_a_different_seed_gives_a_different_probe_row_set(self):
        self.assertNotEqual(_build(seed=0).content_hash, _build(seed=1).content_hash)
        self.assertNotEqual(_images(_build(seed=0), "cip-"), _images(_build(seed=1), "cip-"))

    def test_the_atlantic_census_does_not_move_with_the_seed(self):
        self.assertEqual(_images(_build(seed=0), "ta-"), _images(_build(seed=1), "ta-"))


class CompositionTest(unittest.TestCase):
    def test_every_atlantic_image_is_included_even_when_ineligible(self):
        probe = _build()
        expected = {f"ta-e{index:03d}" for index in range(3)} | {
            f"ta-p{index:03d}" for index in range(3)
        }
        self.assertEqual(_images(probe, "ta-"), expected)

    def test_atlantic_points_are_not_held_out(self):
        rows = _build().rows
        atlantic = rows[rows["region_id"] == TROPICAL_ATLANTIC]
        self.assertEqual(len(atlantic), 6 * POINTS_PER_IMAGE)
        self.assertFalse(bool(atlantic["held_out"].any()))

    def test_indo_pacific_points_are_held_out(self):
        rows = _build().rows
        pacific = rows[rows["region_id"] != TROPICAL_ATLANTIC]
        self.assertTrue(bool(pacific["held_out"].all()))

    def test_proportional_allocation_uses_only_the_eligible_pool(self):
        probe = _build()
        central = _stratum(probe, CENTRAL_INDO_PACIFIC)
        self.assertEqual(central.n_region_images, 45)
        self.assertEqual(central.n_eligible_images, 40)
        self.assertEqual(central.n_images, 12)
        self.assertEqual(_images(probe, "cip-p"), set())

    def test_the_region_floor_lifts_a_small_proportional_share(self):
        # Western Indo-Pacific's proportional share is 2; the floor of 5 wins.
        probe = _build()
        western = _stratum(probe, WESTERN_INDO_PACIFIC)
        self.assertEqual(western.n_eligible_images, 8)
        self.assertEqual(western.n_images, 5)

    def test_the_floor_cannot_exceed_a_smaller_eligible_pool(self):
        probe = _build(_annotations(western_eligible=3))
        western = _stratum(probe, WESTERN_INDO_PACIFIC)
        self.assertEqual(western.n_eligible_images, 3)
        self.assertEqual(western.n_images, 3)

    def test_all_points_on_a_selected_image_are_taken(self):
        probe = _build()
        per_image = probe.rows.groupby("image_id").size()
        self.assertEqual(set(per_image.unique()), {POINTS_PER_IMAGE})
        self.assertEqual(len(probe.rows), 23 * POINTS_PER_IMAGE)

    def test_realized_stratum_points_match_the_selected_rows(self):
        probe = _build()
        for stratum in probe.strata:
            rows = probe.rows[probe.rows["region_id"] == stratum.region_id]
            self.assertEqual(stratum.n_points, len(rows), stratum.region_name)
            self.assertEqual(stratum.n_images, rows["image_id"].nunique(), stratum.region_name)

    def test_an_unmatched_census_region_name_raises(self):
        with self.assertRaises(ValueError):
            _build(census_region_names=frozenset({"Temperate Northern Atlantic"}))

    def test_rows_with_an_unrecorded_region_are_dropped_and_counted(self):
        annotations = _annotations()
        annotations.loc[annotations["image_id"] == "cip-e000", "region_id"] = ""
        probe = _build(annotations)
        self.assertEqual(probe.n_rows_unrecorded_region_dropped, POINTS_PER_IMAGE)
        self.assertNotIn("cip-e000", set(probe.rows["image_id"]))
        self.assertEqual(_stratum(probe, CENTRAL_INDO_PACIFIC).n_eligible_images, 39)


class ProbeRowSchemaTest(unittest.TestCase):
    def test_rows_carry_the_declared_columns_in_order(self):
        self.assertEqual(tuple(_build().rows.columns), PROBE_COLUMNS)

    def test_ground_truth_label_joins_attribute_and_growth_form(self):
        rows = _build().rows
        first = rows.iloc[0]
        self.assertEqual(first["gt_label"], f"{first['benthic_attribute_id']}::")


class ContentHashTest(unittest.TestCase):
    def test_hash_is_stable_under_row_reordering(self):
        rows = _build().rows
        shuffled = rows.iloc[::-1].reset_index(drop=True)
        self.assertEqual(probe_content_hash(rows), probe_content_hash(shuffled))

    def test_hash_changes_when_a_probe_row_changes(self):
        rows = _build().rows
        edited = rows.copy()
        edited.loc[edited.index[0], "col"] = int(edited.iloc[0]["col"]) + 1
        self.assertNotEqual(probe_content_hash(rows), probe_content_hash(edited))

    def test_hash_changes_when_a_held_out_flag_flips(self):
        rows = _build().rows
        edited = rows.copy()
        edited.loc[edited.index[0], "held_out"] = not bool(edited.iloc[0]["held_out"])
        self.assertNotEqual(probe_content_hash(rows), probe_content_hash(edited))


class RegionSnapshotTest(unittest.TestCase):
    def test_snapshot_sorts_region_ids_for_a_stable_hash(self):
        snapshot = region_snapshot(REGION_IDS_BY_ATTRIBUTE)
        self.assertEqual(snapshot[BA_PACIFIC], sorted([CENTRAL_INDO_PACIFIC, WESTERN_INDO_PACIFIC]))
        self.assertEqual(snapshot[BA_UNRECORDED], [])

    def test_hash_ignores_key_insertion_order(self):
        reversed_map = dict(reversed(list(REGION_IDS_BY_ATTRIBUTE.items())))
        self.assertEqual(
            region_snapshot_hash(REGION_IDS_BY_ATTRIBUTE),
            region_snapshot_hash(reversed_map),
        )

    def test_hash_changes_when_a_region_is_added_to_an_attribute(self):
        widened = dict(REGION_IDS_BY_ATTRIBUTE)
        widened[BA_PACIFIC] = ALL_REGIONS
        self.assertNotEqual(
            region_snapshot_hash(REGION_IDS_BY_ATTRIBUTE),
            region_snapshot_hash(widened),
        )


class GroundTruthCountsTest(unittest.TestCase):
    """The counts triage reads to tell an incomplete region list from a mistake.

    Each eligible image carries exactly one BA_PACIFIC annotation, so the
    corpus holds 40 of them in the Central Indo-Pacific while the probe
    selects 12 of those images. A count read off the probe is that lower
    number, and a threshold meant for the corpus one buckets past it.
    """

    def test_counts_span_the_whole_corpus_rather_than_the_selected_probe(self):
        probe = _build()
        selected = probe.rows[
            (probe.rows["benthic_attribute_id"] == BA_PACIFIC)
            & (probe.rows["region_id"] == CENTRAL_INDO_PACIFIC)
        ]
        self.assertEqual(len(selected), 12)
        self.assertEqual(probe.ground_truth_counts[(BA_PACIFIC, CENTRAL_INDO_PACIFIC)], 40)
        self.assertEqual(probe.ground_truth_counts[(BA_PACIFIC, WESTERN_INDO_PACIFIC)], 8)
        self.assertEqual(probe.ground_truth_counts[(BA_PACIFIC, TROPICAL_ATLANTIC)], 3)

    def test_every_annotated_attribute_is_counted_not_just_discriminating_ones(self):
        """45 Central Indo-Pacific images: 40 eligible carrying two globals
        each, 5 plain carrying three.
        """
        self.assertEqual(_build().ground_truth_counts[(BA_GLOBAL, CENTRAL_INDO_PACIFIC)], 95)

    def test_annotations_without_a_recorded_region_are_not_counted(self):
        """A count keyed on an unrecorded region answers no triage question,
        and would inflate nothing but the pair list.
        """
        annotations = _annotations()
        annotations.loc[annotations["image_id"] == "cip-e000", "region_id"] = ""
        counts = _build(annotations).ground_truth_counts
        self.assertEqual(counts[(BA_PACIFIC, CENTRAL_INDO_PACIFIC)], 39)
        self.assertNotIn((BA_PACIFIC, ""), counts)

    def test_hash_ignores_pair_insertion_order(self):
        counts = {(BA_PACIFIC, CENTRAL_INDO_PACIFIC): 40, (BA_GLOBAL, TROPICAL_ATLANTIC): 15}
        self.assertEqual(
            ground_truth_counts_hash(counts),
            ground_truth_counts_hash(dict(reversed(list(counts.items())))),
        )

    def test_hash_changes_when_a_pair_count_changes(self):
        """The counts are frozen for the same reason the region map is: a
        score must not move because the corpus grew afterwards.
        """
        counts = {(BA_PACIFIC, CENTRAL_INDO_PACIFIC): 40}
        self.assertNotEqual(
            ground_truth_counts_hash(counts),
            ground_truth_counts_hash({(BA_PACIFIC, CENTRAL_INDO_PACIFIC): 41}),
        )

    def test_the_json_snapshot_round_trips_to_the_same_counts(self):
        probe = _build()
        payload = json.loads(ground_truth_counts_json(probe.ground_truth_counts))
        restored = {
            (attribute_id, region_id): count
            for attribute_id, by_region in payload.items()
            for region_id, count in by_region.items()
        }
        self.assertEqual(restored, probe.ground_truth_counts)


class MinimumDetectableEffectTest(unittest.TestCase):
    def test_required_n_matches_the_shared_formula(self):
        table = minimum_detectable_effect_table(
            _build().strata,
            alpha=0.05,
            baseline_rates=(0.05,),
            effects=(0.05,),
            design_effects=(1.0,),
        )
        entry = next(row for row in table if row["region_id"] == TROPICAL_ATLANTIC)
        self.assertEqual(
            entry["required_n"],
            required_n_for_detection(0.05, 0.05, alpha=0.05, design_effect=1.0),
        )
        self.assertEqual(entry["required_n"], 299)

    def test_resolvable_compares_the_realized_points_against_the_requirement(self):
        table = minimum_detectable_effect_table(
            _build().strata,
            alpha=0.05,
            baseline_rates=(0.05,),
            effects=(0.10, 0.20),
            design_effects=(1.0,),
        )
        atlantic = {row["effect"]: row for row in table if row["region_id"] == TROPICAL_ATLANTIC}
        # 24 Atlantic points: short of the 75 a 10-point effect needs, past the
        # 19 a 20-point one does.
        self.assertEqual(atlantic[0.10]["n_points"], 24)
        self.assertEqual(atlantic[0.10]["required_n"], 75)
        self.assertFalse(atlantic[0.10]["resolvable"])
        self.assertEqual(atlantic[0.20]["required_n"], 19)
        self.assertTrue(atlantic[0.20]["resolvable"])


class NameSnapshotTest(unittest.TestCase):
    """The display names frozen beside the region map.

    A name resolved at scoring time makes the artifact depend on a later state
    of the taxonomy, so the names travel with the probe and the manifest pins
    which ones they were.
    """

    def test_hash_ignores_key_insertion_order(self):
        reordered = NameSnapshot(
            benthic_attributes=dict(reversed(list(NAMES.benthic_attributes.items()))),
            growth_forms=dict(NAMES.growth_forms),
            regions=dict(reversed(list(NAMES.regions.items()))),
        )
        self.assertEqual(name_snapshot_hash(NAMES), name_snapshot_hash(reordered))

    def test_hash_changes_when_a_name_changes(self):
        renamed = NameSnapshot(
            benthic_attributes={**NAMES.benthic_attributes, BA_PACIFIC: "Acropora sp."},
            growth_forms=NAMES.growth_forms,
            regions=NAMES.regions,
        )
        self.assertNotEqual(name_snapshot_hash(NAMES), name_snapshot_hash(renamed))

    def test_hash_distinguishes_the_sections(self):
        """A flat id-to-name map would hash a region renamed to an attribute's
        name as no change at all."""
        swapped = NameSnapshot(
            benthic_attributes=NAMES.regions,
            growth_forms=NAMES.growth_forms,
            regions=NAMES.benthic_attributes,
        )
        self.assertNotEqual(name_snapshot_hash(NAMES), name_snapshot_hash(swapped))

    def test_the_json_snapshot_round_trips_every_section(self):
        payload = json.loads(name_snapshot_json(NAMES))
        self.assertEqual(payload["benthic_attributes"][BA_PACIFIC], "Acropora")
        self.assertEqual(payload["growth_forms"][GF_BRANCHING], "Branching")
        self.assertEqual(payload["regions"][TROPICAL_ATLANTIC], "Tropical Atlantic")


class AncestrySnapshotTest(unittest.TestCase):
    def test_the_json_snapshot_keeps_each_path_root_first(self):
        """Reversing a path would make two attributes share an ancestor only
        when they share a leaf, which inverts the statistic that reads it."""
        payload = json.loads(ancestry_snapshot_json(ANCESTRY))
        self.assertEqual(payload[BA_PACIFIC], [BA_ROOT, BA_PACIFIC])
        self.assertEqual(payload[BA_UNRECORDED], [BA_UNRECORDED])

    def test_hash_ignores_key_insertion_order(self):
        self.assertEqual(
            ancestry_snapshot_hash(ANCESTRY),
            ancestry_snapshot_hash(dict(reversed(list(ANCESTRY.items())))),
        )

    def test_hash_changes_when_an_attribute_is_regrafted(self):
        regrafted = {**ANCESTRY, BA_PACIFIC: [BA_UNRECORDED, BA_PACIFIC]}
        self.assertNotEqual(ancestry_snapshot_hash(ANCESTRY), ancestry_snapshot_hash(regrafted))

    def test_snapshot_copies_the_paths_it_was_given(self):
        snapshot = ancestry_snapshot(ANCESTRY)
        self.assertEqual(snapshot[BA_GLOBAL], [BA_ROOT, BA_GLOBAL])


class ManifestTest(unittest.TestCase):
    def _manifest(self, probe=None, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "source_uri": "s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet",
            "source_etag": '"abc123"',
            "source_row_count": 459_025,
            "builder_git_sha": "deadbeef",
        }
        arguments.update(overrides)
        return build_manifest(
            probe if probe is not None else _build(),
            **arguments,  # pyright: ignore[reportArgumentType]
        )

    def test_manifest_is_json_serializable(self):
        json.dumps(self._manifest())

    def test_manifest_records_the_source_and_the_builder(self):
        manifest = self._manifest()
        self.assertEqual(manifest["source_etag"], '"abc123"')
        self.assertEqual(manifest["source_row_count"], 459_025)
        self.assertEqual(manifest["builder_git_sha"], "deadbeef")
        self.assertEqual(manifest["seed"], 0)

    def test_manifest_content_hash_tracks_the_probe_rows(self):
        probe = _build()
        self.assertEqual(manifest_hash := self._manifest(probe)["content_hash"], probe.content_hash)
        self.assertNotEqual(manifest_hash, self._manifest(_build(seed=1))["content_hash"])

    def test_manifest_records_the_frozen_ground_truth_counts(self):
        """Two scores are comparable only if they read the same corpus counts,
        so the hash travels beside the region-snapshot one."""
        probe = _build()
        manifest = self._manifest(probe)
        self.assertEqual(manifest["ground_truth_counts_hash"], probe.ground_truth_counts_hash)
        self.assertEqual(manifest["n_ground_truth_pairs"], len(probe.ground_truth_counts))

    def test_manifest_pins_the_frozen_names_and_ancestry(self):
        """Two scores are comparable only if they rendered the same names off
        the same taxonomy, so both snapshots are hashed beside the region map's.
        """
        manifest = self._manifest(names=NAMES, ancestry=ANCESTRY)
        self.assertEqual(manifest["names_hash"], name_snapshot_hash(NAMES))
        self.assertEqual(manifest["ancestry_hash"], ancestry_snapshot_hash(ANCESTRY))
        self.assertEqual(manifest["n_ancestry_attributes"], len(ANCESTRY))
        self.assertEqual(
            manifest["n_names"],
            {"benthic_attributes": 4, "growth_forms": 1, "regions": 3},
        )

    def test_manifest_hashes_move_with_a_renamed_attribute(self):
        renamed = NameSnapshot(
            benthic_attributes={**NAMES.benthic_attributes, BA_PACIFIC: "Acropora sp."},
            growth_forms=NAMES.growth_forms,
            regions=NAMES.regions,
        )
        self.assertNotEqual(
            self._manifest(names=NAMES, ancestry=ANCESTRY)["names_hash"],
            self._manifest(names=renamed, ancestry=ANCESTRY)["names_hash"],
        )

    def test_a_probe_frozen_without_names_records_no_hash_for_them(self):
        """An older probe carries neither snapshot; the manifest says so rather
        than pinning a hash of nothing."""
        manifest = self._manifest()
        self.assertIsNone(manifest["names_hash"])
        self.assertIsNone(manifest["ancestry_hash"])

    def test_freezing_names_leaves_the_probe_content_hash_alone(self):
        """The feature cache is keyed on the selected points. Folding names
        into the content hash would invalidate every cached shard whenever the
        taxonomy was renamed.
        """
        probe = _build()
        self.assertEqual(
            self._manifest(probe, names=NAMES, ancestry=ANCESTRY)["content_hash"],
            probe.content_hash,
        )

    def test_manifest_counts_match_the_realized_strata(self):
        manifest = self._manifest()
        self.assertEqual(manifest["n_images"], 23)
        self.assertEqual(manifest["n_points"], 23 * POINTS_PER_IMAGE)
        strata = {entry["region_id"]: entry for entry in manifest["strata"]}
        self.assertEqual(strata[TROPICAL_ATLANTIC]["policy"], "census")
        self.assertFalse(strata[TROPICAL_ATLANTIC]["held_out"])
        self.assertEqual(strata[CENTRAL_INDO_PACIFIC]["policy"], "sample")
        self.assertTrue(strata[CENTRAL_INDO_PACIFIC]["held_out"])


class ReadAnnotationsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, columns: dict[str, pa.Array]) -> Path:
        path = Path(self.tmp.name) / "annotations.parquet"
        pq.write_table(pa.table(columns), path)
        return path

    def _base_columns(self) -> dict[str, pa.Array]:
        return {
            "id": pa.array(["a1", "a2"]),
            "image_id": pa.array(["i1", "i1"]),
            "point_id": pa.array(["p1", "p2"]),
            "row": pa.array([10, 20], pa.int32()),
            "col": pa.array([30, 40], pa.int32()),
            "benthic_attribute_id": pa.array([BA_GLOBAL, BA_PACIFIC]),
            "benthic_attribute_name": pa.array(["Global", "Pacific"]),
            "growth_form_id": pa.array([None, "gf1"], pa.string()),
            "growth_form_name": pa.array([None, "Branching"], pa.string()),
            "updated_on": pa.array(["2026-01-01", "2026-01-02"]),
            "region_id": pa.array([CENTRAL_INDO_PACIFIC, CENTRAL_INDO_PACIFIC]),
            "region_name": pa.array(["Central Indo-Pacific", "Central Indo-Pacific"]),
        }

    def test_null_growth_form_becomes_an_empty_string(self):
        frame = read_annotations(self._write(self._base_columns()))
        self.assertEqual(list(frame["growth_form_id"]), ["", "gf1"])
        self.assertEqual(list(frame["growth_form_name"]), ["", "Branching"])

    def test_a_parquet_without_site_id_yields_an_empty_site_id_column(self):
        frame = read_annotations(self._write(self._base_columns()))
        self.assertEqual(list(frame["site_id"]), ["", ""])

    def test_a_parquet_with_site_id_carries_it_through(self):
        columns = self._base_columns()
        columns["site_id"] = pa.array(["s1", None], pa.string())
        frame = read_annotations(self._write(columns))
        self.assertEqual(list(frame["site_id"]), ["s1", ""])

    def test_a_null_region_becomes_an_empty_string(self):
        columns = self._base_columns()
        columns["region_id"] = pa.array([CENTRAL_INDO_PACIFIC, None], pa.string())
        columns["region_name"] = pa.array(["Central Indo-Pacific", None], pa.string())
        frame = read_annotations(self._write(columns))
        self.assertEqual(list(frame["region_id"]), [CENTRAL_INDO_PACIFIC, ""])
        self.assertEqual(list(frame["region_name"]), ["Central Indo-Pacific", ""])

    def test_the_selection_runs_over_a_parquet_read_from_disk(self):
        annotations = _annotations()
        path = Path(self.tmp.name) / "corpus.parquet"
        annotations.to_parquet(path, index=False)
        probe = build_probe_set(
            read_annotations(path),
            region_ids_by_attribute=REGION_IDS_BY_ATTRIBUTE,
            options=_options(),
        )
        self.assertEqual(probe.content_hash, _build(annotations).content_hash)


if __name__ == "__main__":
    unittest.main()
