"""Tests for scripts/generate_training_config.py.

The script's only network-touching seams are `_load_ba_library`,
`_load_gf_library`, and `_load_cn_mapping`. Each test patches those
with hand-built fakes that exercise specific corners of the hierarchy
walk, the Porites GF-bucket logic, and the EXCLUDED_NAMES filter.
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

# Allow importing scripts/generate_training_config.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import generate_training_config as gtc  # noqa: E402

from mermaid_classifier.common.benthic_attributes import (  # noqa: E402
    BenthicAttributeLibrary,
    CoralNetMermaidMapping,
    GrowthFormLibrary,
    LabelMappingEntry,
)


def _canned_urlopen(*args, **kwargs):
    """Tiny but valid responses so module-level imports of
    `mermaid_classifier.pyspacer.train` don't reach the network."""
    url = args[0] if args else kwargs.get('url', '')
    if isinstance(url, urllib.request.Request):
        url = url.full_url
    if 'benthicattributes' in url:
        payload: object = {'results': [], 'next': None}
    elif 'labelmappings' in url:
        payload = {'results': [], 'next': None}
    elif 'choices' in url:
        payload = [{'name': 'growthforms', 'data': []}]
    else:
        raise ValueError(f"unexpected url in test: {url!r}")
    return io.BytesIO(json.dumps(payload).encode())


def setUpModule():
    with mock.patch('urllib.request.urlopen', side_effect=_canned_urlopen):
        import mermaid_classifier.pyspacer.train  # noqa: F401


# ---------- Fixture UUIDs ----------

HARD_CORAL = 'ba-hardcoral-0000-0000-000000000001'
ACROPORIDAE = 'ba-acroporidae-000-0000-000000000002'
ACROPORA = 'ba-acropora-0000-0000-000000000003'
ACROPORA_HUMILIS = 'ba-acrhum-0000-0000-000000000004'
TURF_ALGAE = 'ba-turf-0000-0000-000000000005'
ORPHAN_BA = 'ba-orphan-0000-0000-000000000006'
ORPHAN_FAMILY = 'ba-orphfam-0000-0000-000000000007'
UNKNOWN_UUID = 'ba-unknown-0000-0000-000000000099'

# Iteration 2: Porites + species + exclusions.
PORITIDAE = 'ba-poritidae-0000-0000-00000000000a'           # NOT top-108
PORITES = 'ba-porites-0000-0000-00000000000b'                # top-108, genus
PORITES_LOBATA = 'ba-porlob-0000-0000-00000000000c'          # inherent: massive
PORITES_COMPRESSA = 'ba-porcom-0000-0000-00000000000d'       # inherent: branching
PORITES_RUS = 'ba-porrus-0000-0000-00000000000e'             # inherent: submassive (->'')
PORITES_ANNAE = 'ba-porann-0000-0000-00000000000f'           # inherent: blank (->'')
PORITES_ASTREOIDES = 'ba-poras-0000-0000-000000000010'       # top-108 species

BARE_SUBSTRATE = 'ba-bare-0000-0000-000000000011'             # top-108
DEAD_CORAL = 'ba-dead-0000-0000-000000000012'                 # excluded; parent Bare substrate
BLEACHED_CORAL = 'ba-bleached-000-0000-000000000013'         # excluded; parent Hard coral
OTHER_INVERTEBRATES = 'ba-otherinv-000-0000-000000000014'    # excluded; no parent

# Growth form UUIDs.
GF_BRANCHING = 'gf-branching-000-0000-000000000001'
GF_MASSIVE = 'gf-massive-000-0000-000000000002'
GF_ENCRUSTING = 'gf-encrusting-00-0000-000000000003'
GF_FOLIOSE = 'gf-foliose-0000-0000-000000000004'

# A top-108 name absent from the fake BA lib (KeyError path).
GHOST_NAME = 'Ghost Coral That Was Renamed Yesterday'


def _ba_record(uuid: str, name: str, parent: str | None) -> dict:
    return {'id': uuid, 'name': name, 'parent': parent}


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
    _ba_record(HARD_CORAL, 'Hard coral', None),
    _ba_record(ACROPORIDAE, 'Acroporidae', HARD_CORAL),
    _ba_record(ACROPORA, 'Acropora', ACROPORIDAE),
    _ba_record(ACROPORA_HUMILIS, 'Acropora humilis', ACROPORA),
    _ba_record(PORITIDAE, 'Poritidae', HARD_CORAL),
    _ba_record(PORITES, 'Porites', PORITIDAE),
    _ba_record(PORITES_LOBATA, 'Porites lobata', PORITES),
    _ba_record(PORITES_COMPRESSA, 'Porites compressa', PORITES),
    _ba_record(PORITES_RUS, 'Porites rus', PORITES),
    _ba_record(PORITES_ANNAE, 'Porites annae', PORITES),
    _ba_record(PORITES_ASTREOIDES, 'Porites astreoides', PORITES),
    _ba_record(BLEACHED_CORAL, 'Bleached coral', HARD_CORAL),
    _ba_record(TURF_ALGAE, 'Turf algae', None),
    _ba_record(BARE_SUBSTRATE, 'Bare substrate', None),
    _ba_record(DEAD_CORAL, 'Dead coral', BARE_SUBSTRATE),
    _ba_record(OTHER_INVERTEBRATES, 'Other invertebrates', None),
    _ba_record(ORPHAN_FAMILY, 'Orphan family', None),
    _ba_record(ORPHAN_BA, 'Orphan species', ORPHAN_FAMILY),
]

# Top-108 names. Includes the three EXCLUDED_NAMES with top100=1 to
# verify the defensive exclusion path.
TOP108_NAMES = [
    'Hard coral', 'Acropora', 'Turf algae',
    'Porites', 'Porites astreoides', 'Bare substrate',
    # Excluded — should be filtered out even with top100=1:
    'Dead coral', 'Bleached coral', 'Other invertebrates',
    GHOST_NAME,
]

# Species inherent GF (column 'growth forms' in the labels CSV).
LABELS_GROWTH_FORMS = {
    'Porites': 'branching, massive, other/none',  # multi-value: skipped by _load_species_gf_lookup
    'Porites lobata': 'massive',
    'Porites compressa': 'branching',
    'Porites rus': 'submassive',
    'Porites annae': '',
    'Porites astreoides': 'massive',
}

# UUIDs that should appear in included_labels (top-108 minus excluded
# minus the unresolvable GHOST_NAME).
EXPECTED_INCLUDED_BA_UUIDS = {
    HARD_CORAL, ACROPORA, TURF_ALGAE, PORITES, PORITES_ASTREOIDES,
    BARE_SUBSTRATE,
}


def _make_fake_ba_lib() -> BenthicAttributeLibrary:
    fake = BenthicAttributeLibrary.__new__(BenthicAttributeLibrary)
    fake.raw_results = list(BA_RECORDS)
    fake.by_id = {r['id']: r for r in BA_RECORDS}
    fake.by_name = {r['name']: r for r in BA_RECORDS}
    fake.by_parent = {}
    for r in BA_RECORDS:
        fake.by_parent.setdefault(r['parent'], []).append(r)
    return fake


def _make_fake_gf_library() -> GrowthFormLibrary:
    fake = GrowthFormLibrary.__new__(GrowthFormLibrary)
    fake.by_id = {
        GF_BRANCHING: 'Branching',
        GF_MASSIVE: 'Massive',
        GF_ENCRUSTING: 'Encrusting',
        GF_FOLIOSE: 'Foliose',
    }
    return fake


def _make_fake_cn_mapping(entries: list[LabelMappingEntry]) -> CoralNetMermaidMapping:
    fake = CoralNetMermaidMapping.__new__(CoralNetMermaidMapping)
    fake._endpoint = 'http://test.invalid/'
    fake._mapping = {e.provider_id: e for e in entries}
    return fake


def _entry(provider_id: str, ba_id: str, gf_id: str = '',
           ba_name: str = '', gf_name: str = '') -> LabelMappingEntry:
    return LabelMappingEntry(
        provider_label=f'CN-{provider_id}',
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
    _entry('1', ACROPORA, '', 'Acropora'),
    _entry('2', ACROPORA, GF_BRANCHING, 'Acropora', 'Branching'),
    _entry('3', ACROPORA_HUMILIS, '', 'Acropora humilis'),
    _entry('4', ACROPORA_HUMILIS, GF_MASSIVE, 'Acropora humilis', 'Massive'),
    _entry('5', ACROPORIDAE, '', 'Acroporidae'),
    _entry('6', TURF_ALGAE, '', 'Turf algae'),
    _entry('7', ORPHAN_BA, '', 'Orphan species'),
    _entry('8', UNKNOWN_UUID, '', 'Stale label'),
    _entry('9', '', '', ''),

    # Iteration 2: Porites genus + buckets.
    _entry('p1', PORITES, '', 'Porites'),
    _entry('p2', PORITES, GF_BRANCHING, 'Porites', 'Branching'),
    _entry('p3', PORITES, GF_MASSIVE, 'Porites', 'Massive'),
    _entry('p4', PORITES, GF_ENCRUSTING, 'Porites', 'Encrusting'),
    _entry('p5', PORITES, GF_FOLIOSE, 'Porites', 'Foliose'),

    # Porites species — CN never supplies GF (per real-data audit).
    _entry('s1', PORITES_LOBATA, '', 'Porites lobata'),       # inherent: massive
    _entry('s2', PORITES_COMPRESSA, '', 'Porites compressa'),  # inherent: branching
    _entry('s3', PORITES_RUS, '', 'Porites rus'),              # inherent: submassive -> ''
    _entry('s4', PORITES_ANNAE, '', 'Porites annae'),          # inherent: blank -> ''
    _entry('s5', PORITES_ASTREOIDES, '', 'Porites astreoides'),  # top-108 species

    # Excluded labels (defensive — these would otherwise walk to a parent).
    _entry('e1', DEAD_CORAL, '', 'Dead coral'),
    _entry('e2', BLEACHED_CORAL, '', 'Bleached coral'),
    _entry('e3', OTHER_INVERTEBRATES, '', 'Other invertebrates'),
]


# ---------- Synthetic input writers ----------

def _write_labels_csv(path: Path, names: list[str]) -> None:
    """Write a `mapped_to_mermaid_attributes`-shaped CSV.

    Includes the `growth forms` column for the species GF lookup.
    All rows have top100=1 (including the excluded names — defensive).
    """
    with path.open('w', newline='') as f:
        w = csv.DictWriter(
            f, fieldnames=['id', 'name', 'top100', 'growth forms'])
        w.writeheader()
        for i, name in enumerate(names):
            w.writerow({
                'id': f'csv-id-{i}',
                'name': name,
                'top100': 1,
                'growth forms': LABELS_GROWTH_FORMS.get(name, ''),
            })
        # Add a few non-top-108 rows for the species GF lookup. Even
        # though top100=blank, the script reads the full labels_df for
        # _load_species_gf_lookup. These match the BA_RECORDS species.
        for species_name in (
                'Porites lobata', 'Porites compressa',
                'Porites rus', 'Porites annae'):
            w.writerow({
                'id': _name_to_uuid(species_name),
                'name': species_name,
                'top100': '',
                'growth forms': LABELS_GROWTH_FORMS.get(species_name, ''),
            })


def _name_to_uuid(name: str) -> str:
    """Map a fixture name to its UUID (mirrors BA_RECORDS)."""
    for r in BA_RECORDS:
        if r['name'] == name:
            return r['id']
    raise KeyError(name)


def _write_sources_csv(path: Path, ids: list[int],
                        column: str = 'id') -> None:
    with path.open('w', newline='') as f:
        f.write(column + '\n')
        for i in ids:
            f.write(f'{i}\n')


# ---------- Network blocker ----------

class _NetworkAccess(AssertionError):
    pass


def _block_network(*args, **kwargs):
    raise _NetworkAccess(
        f"test attempted live network call: args={args!r} kwargs={kwargs!r}")


# ---------- Test helpers ----------

class _GenerateConfigTestCase(unittest.TestCase):
    """Base case: build inputs, run main(), and read outputs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        self.labels_csv = self.tmp / 'labels.csv'
        self.sources_csv = self.tmp / 'sources_in.csv'
        self.output_dir = self.tmp / 'config_out'

        _write_labels_csv(self.labels_csv, TOP108_NAMES)
        _write_sources_csv(self.sources_csv, [10, 20, 30])

        self.ba_lib = _make_fake_ba_lib()
        self.gf_library = _make_fake_gf_library()
        self.cn_mapping = _make_fake_cn_mapping(DEFAULT_CN_ENTRIES)

    def run_main(self, *, extra_args: list[str] | None = None,
                 cn_entries: list[LabelMappingEntry] | None = None,
                 sources_csv: Path | None = None) -> int:
        cn = (_make_fake_cn_mapping(cn_entries)
              if cn_entries is not None else self.cn_mapping)
        argv = [
            '--output-dir', str(self.output_dir),
            '--labels-csv', str(self.labels_csv),
            '--sources-csv', str(sources_csv or self.sources_csv),
        ]
        if extra_args:
            argv.extend(extra_args)
        with mock.patch.object(gtc, '_load_ba_library',
                               return_value=self.ba_lib), \
             mock.patch.object(gtc, '_load_gf_library',
                               return_value=self.gf_library), \
             mock.patch.object(gtc, '_load_cn_mapping',
                               return_value=cn), \
             mock.patch('urllib.request.urlopen', _block_network):
            return gtc.main(argv)

    def read_csv_rows(self, name: str) -> list[dict]:
        with (self.output_dir / name).open() as f:
            return list(csv.DictReader(f))

    def get_rollup_lookup(self) -> dict[tuple[str, str], tuple[str, str]]:
        return {(r['from_ba_id'], r['from_gf_id']): (
                    r['to_ba_id'], r['to_gf_id'])
                for r in self.read_csv_rows('rollups.csv')}

    def get_included_set(self) -> set[tuple[str, str]]:
        return {(r['ba_id'], r['gf_id'])
                for r in self.read_csv_rows('included_labels.csv')}


# ---------- Tests ----------

class HierarchyWalkTests(_GenerateConfigTestCase):

    def test_top108_self_emits_no_rollup(self):
        self.run_main()
        self.assertNotIn((ACROPORA, ''), set(self.get_rollup_lookup()))

    def test_top108_appears_in_included(self):
        self.run_main()
        included = self.get_included_set()
        self.assertIn((ACROPORA, ''), included)
        self.assertIn((HARD_CORAL, ''), included)
        self.assertIn((TURF_ALGAE, ''), included)

    def test_nested_species_walks_to_nearest_top108(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(ACROPORA_HUMILIS, '')], (ACROPORA, ''))

    def test_intermediate_taxon_walks_one_level_up(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(ACROPORIDAE, '')], (HARD_CORAL, ''))

    def test_orphan_dropped(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        included_ba = {ba for (ba, _) in self.get_included_set()}
        self.assertNotIn((ORPHAN_BA, ''), set(rollups))
        self.assertNotIn(ORPHAN_BA, included_ba)

    def test_unknown_uuid_dropped(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertNotIn((UNKNOWN_UUID, ''), set(rollups))

    def test_null_ba_id_skipped(self):
        self.run_main()
        for from_ba, _gf in self.get_rollup_lookup():
            self.assertNotEqual(from_ba, '')

    def test_root_top108_label(self):
        self.run_main()
        rollups_for_turf = [r for r in self.read_csv_rows('rollups.csv')
                             if r['from_ba_id'] == TURF_ALGAE]
        included_ba = {ba for (ba, _) in self.get_included_set()}
        self.assertEqual(rollups_for_turf, [])
        self.assertIn(TURF_ALGAE, included_ba)


class GrowthFormCollapseTests(_GenerateConfigTestCase):

    def test_top108_with_gf_collapses_to_empty(self):
        """Non-Porites top-108 BA with a GF -> rollup to (BA, '')."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(ACROPORA, GF_BRANCHING)], (ACROPORA, ''))

    def test_cross_ba_rollup_drops_gf(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(
            rollups[(ACROPORA_HUMILIS, GF_MASSIVE)], (ACROPORA, ''))

    def test_non_porites_included_labels_have_blank_gf(self):
        """Every included row has blank GF EXCEPT Porites buckets."""
        self.run_main()
        for r in self.read_csv_rows('included_labels.csv'):
            if r['ba_id'] == PORITES:
                # Porites contributes both '' and the bucket UUIDs.
                continue
            self.assertEqual(
                r['gf_id'], '',
                f"non-Porites row {r} has unexpected GF")


class PoritesBucketTests(_GenerateConfigTestCase):

    def test_porites_three_buckets_in_included_labels(self):
        self.run_main()
        included = self.get_included_set()
        porites_rows = {row for row in included if row[0] == PORITES}
        self.assertEqual(porites_rows, {
            (PORITES, ''),
            (PORITES, GF_BRANCHING),
            (PORITES, GF_MASSIVE),
        })

    def test_porites_genus_branching_no_rollup(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        # (Porites, Branching) is already in included_labels — no rollup needed.
        self.assertNotIn((PORITES, GF_BRANCHING), rollups)

    def test_porites_genus_massive_no_rollup(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertNotIn((PORITES, GF_MASSIVE), rollups)

    def test_porites_genus_no_gf_no_rollup(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertNotIn((PORITES, ''), rollups)

    def test_porites_genus_other_gf_rollup_to_empty(self):
        """(Porites, Encrusting) -> (Porites, '')."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(PORITES, GF_ENCRUSTING)], (PORITES, ''))
        self.assertEqual(rollups[(PORITES, GF_FOLIOSE)], (PORITES, ''))


class PoritesSpeciesTests(_GenerateConfigTestCase):

    def test_porites_species_inherent_massive(self):
        """Porites lobata (inherent: massive) -> (Porites, Massive_uuid)."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(
            rollups[(PORITES_LOBATA, '')], (PORITES, GF_MASSIVE))

    def test_porites_species_inherent_branching(self):
        """Porites compressa (inherent: branching) -> (Porites, Branching_uuid)."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(
            rollups[(PORITES_COMPRESSA, '')], (PORITES, GF_BRANCHING))

    def test_porites_species_submassive_collapses_to_empty(self):
        """Porites rus (inherent: submassive) -> (Porites, '')."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(PORITES_RUS, '')], (PORITES, ''))

    def test_porites_species_blank_inherent_collapses(self):
        """Porites annae (inherent: blank) -> (Porites, '')."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        self.assertEqual(rollups[(PORITES_ANNAE, '')], (PORITES, ''))

    def test_porites_species_cn_gf_overrides_inherent(self):
        """If CN supplies a non-empty GF for a Porites species, CN wins."""
        # Porites lobata's inherent is 'massive', but CN says Branching.
        cn = list(DEFAULT_CN_ENTRIES) + [
            _entry('s_override', PORITES_LOBATA, GF_BRANCHING,
                   'Porites lobata', 'Branching'),
        ]
        self.run_main(cn_entries=cn)
        rollups = self.get_rollup_lookup()
        # The CN-supplied GF entry rolls to Branching, overriding the
        # inherent massive bucket.
        self.assertEqual(
            rollups[(PORITES_LOBATA, GF_BRANCHING)], (PORITES, GF_BRANCHING))
        # The inherent-GF entry (no CN GF) still rolls to Massive.
        self.assertEqual(
            rollups[(PORITES_LOBATA, '')], (PORITES, GF_MASSIVE))

    def test_porites_astreoides_stays_own_class(self):
        """Top-108 Porites species: classified as itself, not in Porites buckets."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        included = self.get_included_set()
        self.assertNotIn((PORITES_ASTREOIDES, ''), rollups)
        self.assertIn((PORITES_ASTREOIDES, ''), included)


class ExclusionTests(_GenerateConfigTestCase):

    def test_dead_coral_dropped_not_walked(self):
        """Dead coral annotations should NOT roll up to Bare substrate."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        included = self.get_included_set()
        # No rollup row from Dead coral.
        self.assertNotIn((DEAD_CORAL, ''), rollups)
        # Not in included_labels.
        self.assertNotIn((DEAD_CORAL, ''), included)
        # Bare substrate IS in included_labels (parent of Dead coral but in top-108).
        self.assertIn((BARE_SUBSTRATE, ''), included)

    def test_bleached_coral_dropped(self):
        """Bleached coral annotations should NOT roll up to Hard coral."""
        self.run_main()
        rollups = self.get_rollup_lookup()
        included = self.get_included_set()
        self.assertNotIn((BLEACHED_CORAL, ''), rollups)
        self.assertNotIn((BLEACHED_CORAL, ''), included)

    def test_other_invertebrates_dropped(self):
        self.run_main()
        rollups = self.get_rollup_lookup()
        included = self.get_included_set()
        self.assertNotIn((OTHER_INVERTEBRATES, ''), rollups)
        self.assertNotIn((OTHER_INVERTEBRATES, ''), included)

    def test_excluded_top108_membership_defensive(self):
        """Even with top100=1 in the labels CSV, EXCLUDED_NAMES are dropped."""
        # The fixture's TOP108_NAMES already includes the three excluded
        # names with top100=1; just confirm none make it to included.
        self.run_main()
        included_ba = {ba for (ba, _) in self.get_included_set()}
        self.assertNotIn(DEAD_CORAL, included_ba)
        self.assertNotIn(BLEACHED_CORAL, included_ba)
        self.assertNotIn(OTHER_INVERTEBRATES, included_ba)


class IncludedLabelCountTests(_GenerateConfigTestCase):

    def test_included_label_ba_uuids_match_expected(self):
        """included_labels has exactly the resolvable, non-excluded BAs."""
        self.run_main()
        included_ba = {ba for (ba, _) in self.get_included_set()}
        self.assertEqual(included_ba, EXPECTED_INCLUDED_BA_UUIDS)


class SourcesPassthroughTests(_GenerateConfigTestCase):

    def test_sources_passthrough_id_column(self):
        self.run_main()
        ids = [r['id'] for r in self.read_csv_rows('sources.csv')]
        self.assertEqual(ids, ['10', '20', '30'])

    def test_sources_passthrough_source_id_column(self):
        alt = self.tmp / 'sources_alt.csv'
        _write_sources_csv(alt, [42, 99], column='Source ID')
        self.run_main(sources_csv=alt)
        ids = [r['id'] for r in self.read_csv_rows('sources.csv')]
        self.assertEqual(ids, ['42', '99'])


class ValidationTests(_GenerateConfigTestCase):

    def test_outputs_round_trip_through_pipeline(self):
        rc = self.run_main()
        self.assertEqual(rc, 0)

    def test_every_to_ba_in_included_labels(self):
        self.run_main(extra_args=['--skip-validation'])
        rollups_path = self.output_dir / 'rollups.csv'
        with rollups_path.open() as f:
            content = f.read()
        broken = content.rstrip() + '\nbogus-from-ba,,bogus-to-ba,\n'
        rollups_path.write_text(broken)
        with self.assertRaises(ValueError):
            gtc.validate_outputs(self.output_dir)


class UnresolvedTop108Tests(_GenerateConfigTestCase):

    def test_skipped_top108_logged_not_crashed(self):
        rc = self.run_main()
        self.assertEqual(rc, 0)
        readme = (self.output_dir / 'README.md').read_text()
        self.assertIn('Unresolved top-108 names', readme)
        self.assertIn(GHOST_NAME, readme)
        # Resolved BAs match expected (excluded names + GHOST_NAME filtered out).
        included_ba = {ba for (ba, _) in self.get_included_set()}
        self.assertEqual(included_ba, EXPECTED_INCLUDED_BA_UUIDS)


class ReadmeTests(_GenerateConfigTestCase):

    def test_readme_documents_gf_deviation(self):
        self.run_main()
        readme = (self.output_dir / 'README.md').read_text()
        self.assertIn('Deviation from notebook', readme)
        self.assertIn('drop_growthforms', readme)

    def test_readme_lists_excluded_names(self):
        self.run_main()
        readme = (self.output_dir / 'README.md').read_text()
        self.assertIn('Excluded labels', readme)
        for name in ('Dead coral', 'Bleached coral', 'Other invertebrates'):
            self.assertIn(name, readme)

    def test_readme_lists_porites_buckets(self):
        self.run_main()
        readme = (self.output_dir / 'README.md').read_text()
        self.assertIn('Porites buckets', readme)
        self.assertIn('Branching', readme)
        self.assertIn('Massive', readme)


class NoNetworkTests(_GenerateConfigTestCase):

    def test_no_network_in_tests(self):
        rc = self.run_main()
        self.assertEqual(rc, 0)


if __name__ == '__main__':
    unittest.main()
