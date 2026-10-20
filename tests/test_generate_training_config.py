"""Tests for scripts/generate_training_config.py.

The script's only network-touching seams are `_load_ba_library`,
`_load_gf_library`, and `_load_cn_mapping`. Each test patches those
with hand-built fakes that exercise specific corners of the hierarchy
walk, the Porites GF-bucket logic, and the EXCLUDED_NAMES filter.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from support.paths import add_scripts_to_path

add_scripts_to_path()

import generate_training_config as gtc  # noqa: E402

from mermaid_classifier.common.benthic_attributes import (  # noqa: E402
    BenthicAttributeLibrary,
    CoralNetMermaidMapping,
    GrowthFormLibrary,
    LabelMappingEntry,
)


def _canned_urlopen(*args, **kwargs):
    """Tiny but valid responses so module-level imports of
    `mermaid_classifier.pyspacer.dataset` and `mermaid_classifier.pyspacer.runner`
    don't reach the network."""
    url = args[0] if args else kwargs.get("url", "")
    if isinstance(url, urllib.request.Request):
        url = url.full_url
    if "benthicattributes" in url:
        payload: object = {"results": [], "next": None}
    elif "labelmappings" in url:
        payload = {"results": [], "next": None}
    elif "choices" in url:
        payload = [{"name": "growthforms", "data": []}]
    else:
        raise ValueError(f"unexpected url in test: {url!r}")
    return io.BytesIO(json.dumps(payload).encode())


def setUpModule():
    with mock.patch("urllib.request.urlopen", side_effect=_canned_urlopen):
        import mermaid_classifier.pyspacer.dataset  # noqa: F401
        import mermaid_classifier.pyspacer.runner  # noqa: F401


# ---------- Fixture UUIDs ----------

HARD_CORAL = "ba-hardcoral-0000-0000-000000000001"
ACROPORIDAE = "ba-acroporidae-000-0000-000000000002"
ACROPORA = "ba-acropora-0000-0000-000000000003"
ACROPORA_HUMILIS = "ba-acrhum-0000-0000-000000000004"
TURF_ALGAE = "ba-turf-0000-0000-000000000005"
ORPHAN_BA = "ba-orphan-0000-0000-000000000006"
ORPHAN_FAMILY = "ba-orphfam-0000-0000-000000000007"
UNKNOWN_UUID = "ba-unknown-0000-0000-000000000099"

# Iteration 2: Porites + species + exclusions.
PORITIDAE = "ba-poritidae-0000-0000-00000000000a"  # NOT top-108
PORITES = "ba-porites-0000-0000-00000000000b"  # top-108, genus
PORITES_LOBATA = "ba-porlob-0000-0000-00000000000c"  # inherent: massive
PORITES_COMPRESSA = "ba-porcom-0000-0000-00000000000d"  # inherent: branching
PORITES_RUS = "ba-porrus-0000-0000-00000000000e"  # inherent: submassive (->'')
PORITES_ANNAE = "ba-porann-0000-0000-00000000000f"  # inherent: blank (->'')
PORITES_ASTREOIDES = "ba-poras-0000-0000-000000000010"  # top-108 species

BARE_SUBSTRATE = "ba-bare-0000-0000-000000000011"  # top-108
DEAD_CORAL = "ba-dead-0000-0000-000000000012"  # excluded; parent Bare substrate
BLEACHED_CORAL = "ba-bleached-000-0000-000000000013"  # excluded; parent Hard coral
OTHER_INVERTEBRATES = "ba-otherinv-000-0000-000000000014"  # excluded; no parent

# Coarsening fixtures: a single-region genus with a species, a species
# under a suffixed genus name, and a species whose only ancestor is excluded.
AGARICIIDAE = "ba-agariciidae-00-0000-000000000015"
AGARICIA = "ba-agaricia-0000-0000-000000000016"  # single region
AGARICIA_AGARICITES = "ba-agaag-0000-0000-000000000017"  # single region
TURBINARIA_CORAL = "ba-turbcoral-000-0000-000000000018"
TURBINARIA_RENIFORMIS = "ba-turbren-0000-0000-000000000019"
ACANTHASTER_PLANCI = "ba-cots-0000-0000-00000000001a"  # parent excluded
HELIOSERIS_CUCULLATA = "ba-helio-0000-0000-00000000001b"  # not top-108; under Agariciidae

REGION_ATLANTIC = "region-atlantic"
REGION_INDO_PACIFIC = "region-indo-pacific"

# Growth form UUIDs.
GF_BRANCHING = "gf-branching-000-0000-000000000001"
GF_MASSIVE = "gf-massive-000-0000-000000000002"
GF_ENCRUSTING = "gf-encrusting-00-0000-000000000003"
GF_FOLIOSE = "gf-foliose-0000-0000-000000000004"

# A top-108 name absent from the fake BA lib (KeyError path).
GHOST_NAME = "Ghost Coral That Was Renamed Yesterday"


def _ba_record(uuid: str, name: str, parent: str | None) -> dict:
    return {"id": uuid, "name": name, "parent": parent}


# Hierarchy:
#
#   HARD_CORAL (top-108)
#     ACROPORIDAE
#       ACROPORA (top-108)
#         ACROPORA_HUMILIS
#     PORITIDAE
#       PORITES (top-108)
#         PORITES_LOBATA, PORITES_COMPRESSA, PORITES_RUS, PORITES_ANNAE
#         PORITES_ASTREOIDES (top-108)
#     BLEACHED_CORAL (excluded)
#   TURF_ALGAE (top-108)
#   BARE_SUBSTRATE (top-108)
#     DEAD_CORAL (excluded)
#   ORPHAN_FAMILY
#     ORPHAN_BA
#   OTHER_INVERTEBRATES (excluded, no parent)
BA_RECORDS = [
    _ba_record(HARD_CORAL, "Hard coral", None),
    _ba_record(ACROPORIDAE, "Acroporidae", HARD_CORAL),
    _ba_record(ACROPORA, "Acropora", ACROPORIDAE),
    _ba_record(ACROPORA_HUMILIS, "Acropora humilis", ACROPORA),
    _ba_record(PORITIDAE, "Poritidae", HARD_CORAL),
    _ba_record(PORITES, "Porites", PORITIDAE),
    _ba_record(PORITES_LOBATA, "Porites lobata", PORITES),
    _ba_record(PORITES_COMPRESSA, "Porites compressa", PORITES),
    _ba_record(PORITES_RUS, "Porites rus", PORITES),
    _ba_record(PORITES_ANNAE, "Porites annae", PORITES),
    _ba_record(PORITES_ASTREOIDES, "Porites astreoides", PORITES),
    _ba_record(BLEACHED_CORAL, "Bleached coral", HARD_CORAL),
    _ba_record(TURF_ALGAE, "Turf algae", None),
    _ba_record(BARE_SUBSTRATE, "Bare substrate", None),
    _ba_record(DEAD_CORAL, "Dead coral", BARE_SUBSTRATE),
    _ba_record(OTHER_INVERTEBRATES, "Other invertebrates", None),
    _ba_record(ORPHAN_FAMILY, "Orphan family", None),
    _ba_record(ORPHAN_BA, "Orphan species", ORPHAN_FAMILY),
    _ba_record(AGARICIIDAE, "Agariciidae", HARD_CORAL),
    _ba_record(AGARICIA, "Agaricia", AGARICIIDAE),
    _ba_record(AGARICIA_AGARICITES, "Agaricia agaricites", AGARICIA),
    _ba_record(TURBINARIA_CORAL, "Turbinaria-coral", HARD_CORAL),
    _ba_record(TURBINARIA_RENIFORMIS, "Turbinaria reniformis", TURBINARIA_CORAL),
    _ba_record(ACANTHASTER_PLANCI, "Acanthaster planci", OTHER_INVERTEBRATES),
    _ba_record(HELIOSERIS_CUCULLATA, "Helioseris cucullata", AGARICIIDAE),
]

SINGLE_REGION_BAS = {AGARICIA, AGARICIA_AGARICITES}
# The labels CSV's `id` column holds the MERMAID BA UUID.
_NAME_TO_UUID = {r["name"]: r["id"] for r in BA_RECORDS}

# Top-108 names. Includes the three EXCLUDED_NAMES with top100=1 to
# verify the defensive exclusion path.
TOP108_NAMES = [
    "Hard coral",
    "Acropora",
    "Turf algae",
    "Porites",
    "Porites astreoides",
    "Bare substrate",
    # Excluded — should be filtered out even with top100=1:
    "Dead coral",
    "Bleached coral",
    "Other invertebrates",
    GHOST_NAME,
]

# Species inherent GF (column 'growth forms' in the labels CSV).
LABELS_GROWTH_FORMS = {
    "Porites": "branching, massive, other/none",  # multi-value: skipped by _load_species_gf_lookup
    "Porites lobata": "massive",
    "Porites compressa": "branching",
    "Porites rus": "submassive",
    "Porites annae": "",
    "Porites astreoides": "massive",
}

# UUIDs that should appear in included_labels (top-108 minus excluded
# minus the unresolvable GHOST_NAME).
EXPECTED_INCLUDED_BA_UUIDS = {
    HARD_CORAL,
    ACROPORA,
    TURF_ALGAE,
    PORITES,
    PORITES_ASTREOIDES,
    BARE_SUBSTRATE,
}


def _make_fake_ba_lib() -> BenthicAttributeLibrary:
    fake = BenthicAttributeLibrary.__new__(BenthicAttributeLibrary)
    fake.raw_results = list(BA_RECORDS)
    fake.by_id = {r["id"]: r for r in BA_RECORDS}
    fake.by_name = {r["name"]: r for r in BA_RECORDS}
    fake.by_parent = {}
    for r in BA_RECORDS:
        fake.by_parent.setdefault(r["parent"], []).append(r)
    fake.region_ids_by_id = {
        r["id"]: frozenset({REGION_ATLANTIC})
        if r["id"] in SINGLE_REGION_BAS
        else frozenset({REGION_ATLANTIC, REGION_INDO_PACIFIC})
        for r in BA_RECORDS
    }
    return fake


def _make_fake_gf_library() -> GrowthFormLibrary:
    fake = GrowthFormLibrary.__new__(GrowthFormLibrary)
    fake.by_id = {
        GF_BRANCHING: "Branching",
        GF_MASSIVE: "Massive",
        GF_ENCRUSTING: "Encrusting",
        GF_FOLIOSE: "Foliose",
    }
    return fake


def _make_fake_cn_mapping(entries: list[LabelMappingEntry]) -> CoralNetMermaidMapping:
    fake = CoralNetMermaidMapping.__new__(CoralNetMermaidMapping)
    fake._endpoint = "http://test.invalid/"
    fake._mapping = {e.provider_id: e for e in entries}
    return fake


def _entry(
    provider_id: str, ba_id: str, gf_id: str = "", ba_name: str = "", gf_name: str = ""
) -> LabelMappingEntry:
    return LabelMappingEntry(
        provider_label=f"CN-{provider_id}",
        benthic_attribute_name=ba_name,
        growth_form_name=gf_name,
        provider_id=provider_id,
        benthic_attribute_id=ba_id,
        growth_form_id=gf_id,
    )


# Default CN mapping for the broad smoke tests. Specific tests build
# narrow CN lists when they need to isolate a behavior.
DEFAULT_CN_ENTRIES = [
    # General BA cases (carried over from Iteration 1):
    _entry("1", ACROPORA, "", "Acropora"),
    _entry("2", ACROPORA, GF_BRANCHING, "Acropora", "Branching"),
    _entry("3", ACROPORA_HUMILIS, "", "Acropora humilis"),
    _entry("4", ACROPORA_HUMILIS, GF_MASSIVE, "Acropora humilis", "Massive"),
    _entry("5", ACROPORIDAE, "", "Acroporidae"),
    _entry("6", TURF_ALGAE, "", "Turf algae"),
    _entry("7", ORPHAN_BA, "", "Orphan species"),
    _entry("8", UNKNOWN_UUID, "", "Stale label"),
    _entry("9", "", "", ""),
    # Iteration 2: Porites genus + buckets.
    _entry("p1", PORITES, "", "Porites"),
    _entry("p2", PORITES, GF_BRANCHING, "Porites", "Branching"),
    _entry("p3", PORITES, GF_MASSIVE, "Porites", "Massive"),
    _entry("p4", PORITES, GF_ENCRUSTING, "Porites", "Encrusting"),
    _entry("p5", PORITES, GF_FOLIOSE, "Porites", "Foliose"),
    # Porites species — CN never supplies GF (per real-data audit).
    _entry("s1", PORITES_LOBATA, "", "Porites lobata"),  # inherent: massive
    _entry("s2", PORITES_COMPRESSA, "", "Porites compressa"),  # inherent: branching
    _entry("s3", PORITES_RUS, "", "Porites rus"),  # inherent: submassive -> ''
    _entry("s4", PORITES_ANNAE, "", "Porites annae"),  # inherent: blank -> ''
    _entry("s5", PORITES_ASTREOIDES, "", "Porites astreoides"),  # top-108 species
    # Excluded labels (defensive — these would otherwise walk to a parent).
    _entry("e1", DEAD_CORAL, "", "Dead coral"),
    _entry("e2", BLEACHED_CORAL, "", "Bleached coral"),
    _entry("e3", OTHER_INVERTEBRATES, "", "Other invertebrates"),
]


# ---------- Synthetic input writers ----------


def _write_labels_csv(path: Path, names: list[str]) -> None:
    """Write a `mapped_to_mermaid_attributes`-shaped CSV.

    Includes the `growth forms` column for the species GF lookup.
    All rows have top100=1 (including the excluded names — defensive).
    """
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "top100", "growth forms"])
        w.writeheader()
        for i, name in enumerate(names):
            w.writerow(
                {
                    "id": _NAME_TO_UUID.get(name, f"csv-id-{i}"),
                    "name": name,
                    "top100": 1,
                    "growth forms": LABELS_GROWTH_FORMS.get(name, ""),
                }
            )
        # Add a few non-top-108 rows for the species GF lookup. Even
        # though top100=blank, the script reads the full labels_df for
        # _load_species_gf_lookup. These match the BA_RECORDS species.
        for species_name in ("Porites lobata", "Porites compressa", "Porites rus", "Porites annae"):
            w.writerow(
                {
                    "id": _name_to_uuid(species_name),
                    "name": species_name,
                    "top100": "",
                    "growth forms": LABELS_GROWTH_FORMS.get(species_name, ""),
                }
            )


def _name_to_uuid(name: str) -> str:
    """Map a fixture name to its UUID (mirrors BA_RECORDS)."""
    for r in BA_RECORDS:
        if r["name"] == name:
            return r["id"]
    raise KeyError(name)


def _write_sources_csv(path: Path, ids: list[int], column: str = "id") -> None:
    with path.open("w", newline="") as f:
        f.write(column + "\n")
        for i in ids:
            f.write(f"{i}\n")


# ---------- Network blocker ----------


class _NetworkAccess(AssertionError):
    pass


def _block_network(*args, **kwargs):
    raise _NetworkAccess(f"test attempted live network call: args={args!r} kwargs={kwargs!r}")


# ---------- Test helpers ----------


class _GenerateConfigTestCase(unittest.TestCase):
    """Base case: build inputs, run main(), and read outputs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        self.labels_csv = self.tmp / "labels.csv"
        self.sources_csv = self.tmp / "sources_in.csv"
        self.output_dir = self.tmp / "config_out"

        _write_labels_csv(self.labels_csv, TOP108_NAMES)
        _write_sources_csv(self.sources_csv, [10, 20, 30])

        self.ba_lib = _make_fake_ba_lib()
        self.gf_library = _make_fake_gf_library()
        self.cn_mapping = _make_fake_cn_mapping(DEFAULT_CN_ENTRIES)

    def run_main(
        self,
        *,
        extra_args: list[str] | None = None,
        cn_entries: list[LabelMappingEntry] | None = None,
        sources_csv: Path | None = None,
    ) -> int:
        cn = _make_fake_cn_mapping(cn_entries) if cn_entries is not None else self.cn_mapping
        argv = [
            "--output-dir",
            str(self.output_dir),
            "--labels-csv",
            str(self.labels_csv),
            "--sources-csv",
            str(sources_csv or self.sources_csv),
        ]
        if extra_args:
            argv.extend(extra_args)
        with (
            mock.patch.object(gtc, "_load_ba_library", return_value=self.ba_lib),
            mock.patch.object(gtc, "_load_gf_library", return_value=self.gf_library),
            mock.patch.object(gtc, "_load_cn_mapping", return_value=cn),
            mock.patch("urllib.request.urlopen", _block_network),
        ):
            return gtc.main(argv)

    def read_csv_rows(self, name: str) -> list[dict]:
        with (self.output_dir / name).open() as f:
            return list(csv.DictReader(f))

    def get_rollup_lookup(self) -> dict[tuple[str, str], tuple[str, str]]:
        return {
            (r["from_ba_id"], r["from_gf_id"]): (r["to_ba_id"], r["to_gf_id"])
            for r in self.read_csv_rows("rollups.csv")
        }

    def get_included_set(self) -> set[tuple[str, str]]:
        return {(r["ba_id"], r["gf_id"]) for r in self.read_csv_rows("included_labels.csv")}


# ---------- Tests ----------


class HierarchyWalkTests(_GenerateConfigTestCase):
    def test_top108_self_emits_no_rollup(self):
        self.run_main()
        self.assertNotIn((ACROPORA, ""), set(self.get_rollup_lookup()))

    def test_nested_species_walks_to_nearest_top108(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(ACROPORA_HUMILIS, "")], (ACROPORA, ""))

    def test_orphan_dropped(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        included_ba = {ba for (ba, _) in self.get_included_set()}
        self.assertNotIn((ORPHAN_BA, ""), set(rollups))
        self.assertNotIn(ORPHAN_BA, included_ba)

    def test_unknown_uuid_dropped(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertNotIn((UNKNOWN_UUID, ""), set(rollups))

    def test_null_ba_id_skipped(self):
        self.run_main()
        for from_ba, _gf in self.get_rollup_lookup():
            self.assertNotEqual(from_ba, "")

    def test_root_top108_label(self):
        self.run_main()
        rollups_for_turf = [
            r for r in self.read_csv_rows("rollups.csv") if r["from_ba_id"] == TURF_ALGAE
        ]
        included_ba = {ba for (ba, _) in self.get_included_set()}
        self.assertEqual(rollups_for_turf, [])
        self.assertIn(TURF_ALGAE, included_ba)


class GrowthFormCollapseTests(_GenerateConfigTestCase):
    def test_top108_with_gf_collapses_to_empty(self):
        """Non-Porites top-108 BA with a GF -> rollup to (BA, '')."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(ACROPORA, GF_BRANCHING)], (ACROPORA, ""))

    def test_cross_ba_rollup_drops_gf(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(ACROPORA_HUMILIS, GF_MASSIVE)], (ACROPORA, ""))

    def test_non_porites_included_labels_have_blank_gf(self):
        """Every included row has blank GF EXCEPT Porites buckets."""
        self.run_main()
        for r in self.read_csv_rows("included_labels.csv"):
            if r["ba_id"] == PORITES:
                # Porites contributes both '' and the bucket UUIDs.
                continue
            self.assertEqual(r["gf_id"], "", f"non-Porites row {r} has unexpected GF")


class PoritesBucketTests(_GenerateConfigTestCase):
    def test_porites_three_buckets_in_included_labels(self):
        self.run_main()
        included = self.get_included_set()
        porites_rows = {row for row in included if row[0] == PORITES}
        self.assertEqual(
            porites_rows,
            {
                (PORITES, ""),
                (PORITES, GF_BRANCHING),
                (PORITES, GF_MASSIVE),
            },
        )

    def test_porites_genus_other_gf_rollup_to_empty(self):
        """(Porites, Encrusting) -> (Porites, '')."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(PORITES, GF_ENCRUSTING)], (PORITES, ""))
        self.assertEqual(rollups[(PORITES, GF_FOLIOSE)], (PORITES, ""))


class PoritesSpeciesTests(_GenerateConfigTestCase):
    def test_porites_species_inherent_massive(self):
        """Porites lobata (inherent: massive) -> (Porites, Massive_uuid)."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(PORITES_LOBATA, "")], (PORITES, GF_MASSIVE))

    def test_porites_species_submassive_collapses_to_empty(self):
        """Porites rus (inherent: submassive) -> (Porites, '')."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(PORITES_RUS, "")], (PORITES, ""))

    def test_porites_species_blank_inherent_collapses(self):
        """Porites annae (inherent: blank) -> (Porites, '')."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(PORITES_ANNAE, "")], (PORITES, ""))

    def test_porites_species_cn_gf_overrides_inherent(self):
        """If CN supplies a non-empty GF for a Porites species, CN wins."""
        # Porites lobata's inherent is 'massive', but CN says Branching.
        cn = list(DEFAULT_CN_ENTRIES) + [
            _entry("s_override", PORITES_LOBATA, GF_BRANCHING, "Porites lobata", "Branching"),
        ]
        self.run_main(cn_entries=cn)
        rollups = self.get_rollup_lookup()
        # The CN-supplied GF entry rolls to Branching, overriding the
        # inherent massive bucket.
        self.assertEqual(rollups[(PORITES_LOBATA, GF_BRANCHING)], (PORITES, GF_BRANCHING))
        # The inherent-GF entry (no CN GF) still rolls to Massive.
        self.assertEqual(rollups[(PORITES_LOBATA, "")], (PORITES, GF_MASSIVE))

    def test_porites_astreoides_stays_own_class(self):
        """Top-108 Porites species: classified as itself, not in Porites buckets."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        included = self.get_included_set()
        self.assertNotIn((PORITES_ASTREOIDES, ""), rollups)
        self.assertIn((PORITES_ASTREOIDES, ""), included)


class ExclusionTests(_GenerateConfigTestCase):
    def test_dead_coral_dropped_not_walked(self):
        """Dead coral annotations should NOT roll up to Bare substrate."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        included = self.get_included_set()
        # No rollup row from Dead coral.
        self.assertNotIn((DEAD_CORAL, ""), rollups)
        # Not in included_labels.
        self.assertNotIn((DEAD_CORAL, ""), included)
        # Bare substrate IS in included_labels (parent of Dead coral but in top-108).
        self.assertIn((BARE_SUBSTRATE, ""), included)


class IncludedLabelCountTests(_GenerateConfigTestCase):
    def test_included_label_ba_uuids_match_expected(self):
        """included_labels has exactly the resolvable, non-excluded BAs."""
        self.run_main()
        included_ba = {ba for (ba, _) in self.get_included_set()}
        self.assertEqual(included_ba, EXPECTED_INCLUDED_BA_UUIDS)


class SourcesPassthroughTests(_GenerateConfigTestCase):
    def test_sources_passthrough_id_column(self):
        self.run_main()
        ids = [r["id"] for r in self.read_csv_rows("sources.csv")]
        self.assertEqual(ids, ["10", "20", "30"])

    def test_sources_passthrough_source_id_column(self):
        alt = self.tmp / "sources_alt.csv"
        _write_sources_csv(alt, [42, 99], column="Source ID")
        self.run_main(sources_csv=alt)
        ids = [r["id"] for r in self.read_csv_rows("sources.csv")]
        self.assertEqual(ids, ["42", "99"])


class ValidationTests(_GenerateConfigTestCase):
    def test_every_to_ba_in_included_labels(self):
        self.run_main(extra_args=["--skip-validation"])
        rollups_path = self.output_dir / "rollups.csv"
        with rollups_path.open() as f:
            content = f.read()
        broken = content.rstrip() + "\nbogus-from-ba,,bogus-to-ba,\n"
        rollups_path.write_text(broken)
        with self.assertRaises(ValueError):
            gtc.validate_outputs(self.output_dir)

    def test_sources_csv_missing_id_column_rejected(self):
        self.run_main(extra_args=["--skip-validation"])
        sources_path = self.output_dir / "sources.csv"
        sources_path.write_text("not_id\n1\n2\n")
        with self.assertRaises(ValueError):
            gtc.validate_outputs(self.output_dir)


class UnresolvedTop108Tests(_GenerateConfigTestCase):
    def test_skipped_top108_is_named_in_the_output_not_silently_dropped(self):
        rc = self.run_main()
        self.assertEqual(rc, 0)
        readme = (self.output_dir / "README.md").read_text()
        self.assertIn(GHOST_NAME, readme)


class IsSpeciesTests(unittest.TestCase):
    def setUp(self):
        self.ba_lib = _make_fake_ba_lib()

    def test_binomial_under_its_genus_is_species(self):
        self.assertTrue(gtc.is_species(ACROPORA_HUMILIS, self.ba_lib))
        self.assertTrue(gtc.is_species(PORITES_ASTREOIDES, self.ba_lib))

    def test_binomial_under_suffixed_genus_is_species(self):
        self.assertTrue(gtc.is_species(TURBINARIA_RENIFORMIS, self.ba_lib))

    def test_orphan_species_name_is_species(self):
        self.assertTrue(gtc.is_species(ACANTHASTER_PLANCI, self.ba_lib))

    def test_genus_and_common_names_are_not_species(self):
        for ba_id in (ACROPORA, HARD_CORAL, TURF_ALGAE, BARE_SUBSTRATE, ORPHAN_BA):
            with self.subTest(ba_id=ba_id):
                self.assertFalse(gtc.is_species(ba_id, self.ba_lib))


class IsRegionSpecificTests(unittest.TestCase):
    def setUp(self):
        self.ba_lib = _make_fake_ba_lib()

    def test_single_region_is_region_specific(self):
        self.assertTrue(gtc.is_region_specific(AGARICIA, self.ba_lib))

    def test_multi_region_is_not_region_specific(self):
        self.assertFalse(gtc.is_region_specific(ACROPORA, self.ba_lib))

    def test_no_regions_is_not_region_specific(self):
        self.ba_lib.region_ids_by_id[ACROPORA] = frozenset()
        self.assertFalse(gtc.is_region_specific(ACROPORA, self.ba_lib))


COARSEN_TOP108_NAMES = TOP108_NAMES + ["Agaricia", "Agaricia agaricites", "Acanthaster planci"]
COARSEN_CN_ENTRIES = DEFAULT_CN_ENTRIES + [
    _entry("a1", AGARICIA, "", "Agaricia"),
    _entry("a2", AGARICIA_AGARICITES, "", "Agaricia agaricites"),
    _entry("a3", AGARICIA_AGARICITES, GF_FOLIOSE, "Agaricia agaricites", "Foliose"),
    _entry("c1", ACANTHASTER_PLANCI, "", "Acanthaster planci"),
    _entry("h1", HELIOSERIS_CUCULLATA, "", "Helioseris cucullata"),
]


class CoarseningTests(_GenerateConfigTestCase):
    def setUp(self):
        super().setUp()
        _write_labels_csv(self.labels_csv, COARSEN_TOP108_NAMES)

    def run_coarsened(self, *flags: str) -> None:
        self.run_main(extra_args=list(flags), cn_entries=COARSEN_CN_ENTRIES)

    def test_species_roll_up_to_genus(self):
        self.run_coarsened("--roll-up-species")
        included_ba = {ba for (ba, _) in self.get_included_set()}
        rollups = self.get_rollup_lookup()
        self.assertNotIn(AGARICIA_AGARICITES, included_ba)
        self.assertIn(AGARICIA, included_ba)
        self.assertEqual(rollups[(AGARICIA_AGARICITES, "")], (AGARICIA, ""))
        self.assertEqual(rollups[(AGARICIA_AGARICITES, GF_FOLIOSE)], (AGARICIA, ""))

    def test_porites_species_rolls_into_its_growth_form_bucket(self):
        self.run_coarsened("--roll-up-species")
        included = self.get_included_set()
        rollups = self.get_rollup_lookup()
        self.assertNotIn((PORITES_ASTREOIDES, ""), included)
        self.assertEqual(rollups[(PORITES_ASTREOIDES, "")], (PORITES, GF_MASSIVE))

    def test_region_specific_genus_cascades_to_family(self):
        self.run_coarsened("--roll-up-species", "--roll-up-region-specific")
        included_ba = {ba for (ba, _) in self.get_included_set()}
        rollups = self.get_rollup_lookup()
        self.assertNotIn(AGARICIA, included_ba)
        self.assertIn(AGARICIIDAE, included_ba)
        self.assertEqual(rollups[(AGARICIA, "")], (AGARICIIDAE, ""))
        self.assertEqual(rollups[(AGARICIA_AGARICITES, "")], (AGARICIIDAE, ""))

    def test_species_under_excluded_parent_drops(self):
        self.run_coarsened("--roll-up-species")
        included_ba = {ba for (ba, _) in self.get_included_set()}
        self.assertNotIn(ACANTHASTER_PLANCI, included_ba)
        self.assertNotIn(OTHER_INVERTEBRATES, included_ba)
        self.assertNotIn((ACANTHASTER_PLANCI, ""), self.get_rollup_lookup())
        readme = (self.output_dir / "README.md").read_text()
        self.assertIn("| Acanthaster planci | Acanthaster planci | (dropped) |", readme)

    def test_label_outside_top108_reroutes_to_added_class(self):
        self.run_coarsened("--roll-up-species", "--roll-up-region-specific")
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(HELIOSERIS_CUCULLATA, "")], (AGARICIIDAE, ""))
        readme = (self.output_dir / "README.md").read_text()
        self.assertIn("| Helioseris cucullata | Hard coral | Agariciidae |", readme)
        self.assertIn("Classes added as roll-up targets (1): Agariciidae.", readme)

    def test_readme_lists_each_class_change(self):
        self.run_coarsened("--roll-up-species", "--roll-up-region-specific")
        readme = (self.output_dir / "README.md").read_text()
        self.assertIn("| Agaricia agaricites | Agaricia agaricites | Agariciidae |", readme)
        self.assertIn("| Porites astreoides | Porites astreoides | Porites |", readme)

    def test_label_with_no_coarser_ancestor_is_kept(self):
        self.ba_lib.region_ids_by_id[TURF_ALGAE] = frozenset({REGION_ATLANTIC})
        self.run_coarsened("--roll-up-region-specific")
        included_ba = {ba for (ba, _) in self.get_included_set()}
        self.assertIn(TURF_ALGAE, included_ba)


def _key(ba_id: str, gf_id: str = "") -> str:
    return f"{ba_id}::{gf_id}"


def _support(n_each: int, *group_ids: int) -> dict[str, int]:
    """{group: points}: n_each points in each of the numbered groups."""
    return {f"site-{g}": n_each for g in group_ids}


# 1000 points over 20 groups: above the default 500-point, 10-group floor.
WELL_SUPPORTED = _support(50, *range(20))


def _write_counts_csv(path: Path, counts: dict[str, dict[str, int]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_key", "site", "project_id", "n"])
        for class_key, groups in counts.items():
            for site, n in groups.items():
                w.writerow([class_key, site, "proj", n])


class MinSupportTests(_GenerateConfigTestCase):
    """Every class of the (uncoarsened) top-108 set is well supported unless
    a test overrides its counts."""

    def run_min_support(self, names: list[str], overrides: dict[str, dict[str, int]]) -> None:
        _write_labels_csv(self.labels_csv, names)
        counts: dict[str, dict[str, int]] = {}
        for name in names:
            if name in _NAME_TO_UUID and name not in gtc.EXCLUDED_NAMES:
                counts[_key(_NAME_TO_UUID[name])] = WELL_SUPPORTED
        if "Porites" in names:
            counts[_key(PORITES, GF_BRANCHING)] = WELL_SUPPORTED
            counts[_key(PORITES, GF_MASSIVE)] = WELL_SUPPORTED
        counts.update(overrides)
        self.run_with_counts(counts)

    def run_with_counts(self, counts: dict[str, dict[str, int]], *flags: str) -> None:
        counts_csv = self.tmp / "support_counts.csv"
        _write_counts_csv(counts_csv, counts)
        self.run_main(
            extra_args=["--min-support-counts", str(counts_csv), *flags],
            cn_entries=COARSEN_CN_ENTRIES,
        )

    def add_ba(self, ba_id: str, name: str, parent: str) -> None:
        """Add a BA to this test's library only, leaving BA_RECORDS (and so
        the golden README digests) untouched."""
        record = _ba_record(ba_id, name, parent)
        self.ba_lib.by_id[ba_id] = record
        self.ba_lib.by_name[name] = record
        self.ba_lib.by_parent.setdefault(parent, []).append(record)
        self.ba_lib.region_ids_by_id[ba_id] = frozenset({REGION_ATLANTIC, REGION_INDO_PACIFIC})

    def included_bas(self) -> set[str]:
        return {ba for (ba, _) in self.get_included_set()}

    def test_merge_sums_points_and_unions_groups(self):
        # Acropora (200 pts, groups 4-8) merges into Acroporidae (400 pts,
        # groups 0-7): 600 points but 9 distinct groups, so Acroporidae is
        # still below the 10-group floor and rolls on to Hard coral.
        self.run_min_support(
            TOP108_NAMES + ["Acroporidae"],
            {
                _key(ACROPORA): _support(40, *range(4, 9)),
                _key(ACROPORIDAE): _support(50, *range(8)),
            },
        )
        self.assertNotIn(ACROPORA, self.included_bas())
        self.assertNotIn(ACROPORIDAE, self.included_bas())
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(ACROPORA, "")], (HARD_CORAL, ""))
        self.assertEqual(rollups[(ACROPORA_HUMILIS, "")], (HARD_CORAL, ""))
        readme = (self.output_dir / "README.md").read_text()
        self.assertIn("| Acroporidae | 600 | 9 | Hard coral |", readme)

    def test_chain_stops_at_first_supported_ancestor(self):
        # Acropora humilis -> Acropora (new, still thin) -> Acroporidae.
        self.run_min_support(
            ["Hard coral", "Acroporidae", "Acropora humilis", "Turf algae"],
            {_key(ACROPORA_HUMILIS): _support(30, 1, 2, 3)},
        )
        self.assertEqual(self.included_bas(), {HARD_CORAL, ACROPORIDAE, TURF_ALGAE})
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(ACROPORA_HUMILIS, "")], (ACROPORIDAE, ""))
        self.assertEqual(rollups[(ACROPORA, "")], (ACROPORIDAE, ""))

    def test_thin_siblings_create_their_parent_as_a_new_class(self):
        # Equal points: abrotanoides sorts first and creates Acropora, then
        # humilis (sorting before Acropora) merges in, passing both floors.
        acropora_abrotanoides = "ba-acrabr-0000-0000-00000000001c"
        self.add_ba(acropora_abrotanoides, "Acropora abrotanoides", ACROPORA)
        self.run_min_support(
            ["Hard coral", "Acropora abrotanoides", "Acropora humilis", "Turf algae"],
            {
                _key(acropora_abrotanoides): _support(50, *range(6)),
                _key(ACROPORA_HUMILIS): _support(50, *range(4, 10)),
            },
        )
        self.assertEqual(self.included_bas(), {HARD_CORAL, ACROPORA, TURF_ALGAE})
        self.assertEqual(self.get_rollup_lookup()[(ACROPORA_HUMILIS, "")], (ACROPORA, ""))
        readme = (self.output_dir / "README.md").read_text()
        self.assertIn("| Acropora abrotanoides | 300 | 6 | Acropora |", readme)

    def test_excluded_or_missing_ancestor_drops_the_class(self):
        # Dead coral (excluded) sits under Bare substrate, which would
        # otherwise absorb its child; Turf algae is a root.
        coral_rubble = "ba-rubble-0000-0000-00000000001d"
        self.add_ba(coral_rubble, "Coral rubble", DEAD_CORAL)
        self.run_min_support(
            ["Hard coral", "Bare substrate", "Coral rubble", "Turf algae"],
            {
                _key(coral_rubble): _support(10, 1),
                _key(TURF_ALGAE): _support(10, 1),
            },
        )
        self.assertEqual(self.included_bas(), {HARD_CORAL, BARE_SUBSTRATE})
        self.assertNotIn((TURF_ALGAE, ""), self.get_rollup_lookup())
        readme = (self.output_dir / "README.md").read_text()
        self.assertIn("| Coral rubble | 10 | 1 | (dropped) |", readme)
        self.assertIn("| Turf algae | 10 | 1 | (dropped) |", readme)

    def test_thin_porites_carries_its_buckets_to_the_parent(self):
        self.run_min_support(TOP108_NAMES, {_key(PORITES): _support(10, 1)})
        included = self.get_included_set()
        self.assertEqual({row for row in included if row[0] == PORITES}, set())
        self.assertIn((PORITIDAE, ""), included)
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(PORITES, GF_BRANCHING)], (PORITIDAE, ""))
        self.assertEqual(rollups[(PORITES_LOBATA, "")], (PORITIDAE, ""))

    def test_class_absent_from_counts_has_no_support(self):
        _write_labels_csv(self.labels_csv, ["Hard coral", "Acropora"])
        self.run_with_counts({_key(HARD_CORAL): WELL_SUPPORTED})
        self.assertEqual(self.included_bas(), {HARD_CORAL})
        self.assertEqual(self.get_rollup_lookup()[(ACROPORA, "")], (HARD_CORAL, ""))

    def test_porites_bucket_collapses_into_porites_first(self):
        self.run_min_support(
            TOP108_NAMES,
            {_key(PORITES, GF_BRANCHING): _support(10, 1)},
        )
        porites_rows = {row for row in self.get_included_set() if row[0] == PORITES}
        self.assertEqual(porites_rows, {(PORITES, ""), (PORITES, GF_MASSIVE)})
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(PORITES, GF_BRANCHING)], (PORITES, ""))
        self.assertEqual(rollups[(PORITES_COMPRESSA, "")], (PORITES, ""))
        self.assertEqual(rollups[(PORITES_LOBATA, "")], (PORITES, GF_MASSIVE))
        readme = (self.output_dir / "README.md").read_text()
        self.assertIn("| Porites::Branching | 10 | 1 | Porites |", readme)

    def test_equal_points_resolve_by_class_key(self):
        # Acropora's key sorts first, so it merges into Acroporidae, which then
        # passes both floors. The other order sends both to Hard coral.
        self.run_min_support(
            TOP108_NAMES + ["Acroporidae"],
            {
                _key(ACROPORA): _support(60, *range(5)),
                _key(ACROPORIDAE): _support(60, *range(5, 10)),
            },
        )
        self.assertIn(ACROPORIDAE, self.included_bas())
        self.assertEqual(self.get_rollup_lookup()[(ACROPORA, "")], (ACROPORIDAE, ""))

    def test_thresholds_come_from_the_flags(self):
        # Exactly at both floors is enough to stay.
        _write_labels_csv(self.labels_csv, ["Hard coral", "Acropora"])
        self.run_with_counts(
            {_key(HARD_CORAL): WELL_SUPPORTED, _key(ACROPORA): _support(10, 1, 2)},
            "--min-points",
            "20",
            "--min-groups",
            "2",
        )
        self.assertEqual(self.included_bas(), {HARD_CORAL, ACROPORA})

    def test_readme_states_rule_and_lists_rolled_classes(self):
        self.run_min_support(
            ["Hard coral", "Acroporidae", "Acropora humilis", "Turf algae"],
            {_key(ACROPORA_HUMILIS): _support(30, 1, 2, 3)},
        )
        readme = (self.output_dir / "README.md").read_text()
        self.assertIn(
            "min support (>= 500 points and >= 10 groups, counts from support_counts.csv)",
            readme,
        )
        self.assertIn("| Acropora humilis | 90 | 3 | Acroporidae |", readme)
        self.assertIn("| Acropora | 90 | 3 | Acroporidae |", readme)


def _normalized_readme(text: str, tmp: Path) -> str:
    """README with the run-dependent parts (time, temp paths, mtimes) masked."""
    text = text.replace(str(tmp), "<tmp>")
    text = re.sub(r"Generated \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", "Generated <time>", text)
    return re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", "<mtime>", text)


class OutputStabilityTests(_GenerateConfigTestCase):
    """Without --min-support-counts the outputs match the generator's
    output before min support existed. The digests come from that version;
    regenerate them only for an intended change to the outputs."""

    GOLDEN_SHA256 = {
        (): {
            "rollups.csv": "c1a106106943799b953b391f047717750dd57098b6b238c561e9dc1d9d9c6f88",
            "included_labels.csv": "f38978b65922fac6397f9595485260ce27f3b429b302cf0efcf2f73d559e9bd7",
            "README.md": "62d8597474bca87bb4fd8f6885e5a27c54ad3387add60b1f4a1093b722048c08",
        },
        ("--roll-up-species", "--roll-up-region-specific"): {
            "rollups.csv": "674fc1b5a8b6ae0456fadfb92dc8b33d2a55e72b11558556dede4a6fa75a23cf",
            "included_labels.csv": "52fad936d0e2df235c096668b84703f20308afc806bd51912f4ad80db1de5b35",
            "README.md": "ef33361467e7b47d70562b50167f9d5b7af86f5e23433ea0dc3cbff2b6f2df49",
        },
    }

    def test_outputs_are_byte_identical_without_min_support(self):
        _write_labels_csv(self.labels_csv, COARSEN_TOP108_NAMES)
        for flags, expected in self.GOLDEN_SHA256.items():
            with self.subTest(flags=flags):
                with mock.patch.object(sys, "argv", ["generate_training_config.py", *flags]):
                    self.run_main(extra_args=list(flags), cn_entries=COARSEN_CN_ENTRIES)
                actual = {
                    name: (self.output_dir / name).read_text()
                    for name in ("rollups.csv", "included_labels.csv")
                }
                actual["README.md"] = _normalized_readme(
                    (self.output_dir / "README.md").read_text(), self.tmp
                )
                digests = {
                    name: hashlib.sha256(text.encode()).hexdigest() for name, text in actual.items()
                }
                self.assertEqual(digests, expected)


if __name__ == "__main__":
    unittest.main()
