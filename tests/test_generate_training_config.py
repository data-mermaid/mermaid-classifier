"""Smoke tests for scripts/generate_training_config.py.

Build tiny synthetic input CSVs, run the script as a function, then
assert each output CSV (a) contains the expected rows and (b)
round-trips through the pipeline's CsvSpec subclasses without error.
"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

# Allow importing scripts/generate_training_config.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import generate_training_config as gtc  # noqa: E402


# --- Synthetic fixture builders -------------------------------------------

def _write_sources_csv(p: Path, rows: list[dict]) -> None:
    p.write_text('')  # touch
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'Source ID', 'Source name', 'ToKeep', 'ImageQuality',
            'CoralDiversity', 'InS3',
        ])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_labels_csv(p: Path, rows: list[dict]) -> None:
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'id', 'name', 'top100', 'CoralNetAnnotations',
            'priority', 'priority_notes', 'parent',
        ])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_label_mapping_csv(p: Path, rows: list[dict]) -> None:
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'benthic attribute', 'growth form', 'CoralNetAnnotations',
            'provider', 'provider id', 'provider label',
        ])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_growthforms_csv(p: Path, rows: list[dict]) -> None:
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'name'])
        w.writeheader()
        for row in rows:
            w.writerow(row)


# Stable UUIDs for fixtures.
TURF = 'ba-turf-0000-0000-000000000001'
PORITES = 'ba-porites-000-0000-000000000002'
PORITES_LOBATA = 'ba-porlob-000-0000-000000000003'
ACROPORA = 'ba-acro-0000-0000-000000000004'
SYMPHYLLIA = 'ba-symph-0000-0000-000000000005'
MUSSIDAE = 'ba-mussid-000-0000-000000000006'
BLEACHED = 'ba-bleach-000-0000-000000000007'
SUBLABEL = 'ba-sub-0000-0000-000000000008'

GF_BRANCHING = 'gf-branch-000-0000-000000000001'
GF_MASSIVE = 'gf-massive-00-0000-000000000002'
GF_ENCRUSTING = 'gf-encrust-00-0000-000000000003'


def _make_inputs(tmp: Path) -> dict:
    sources = tmp / 'sources_in.csv'
    labels = tmp / 'labels.csv'
    label_mapping = tmp / 'label_mapping.csv'
    growthforms = tmp / 'growthforms.csv'

    _write_sources_csv(sources, [
        {'Source ID': 100, 'Source name': 'a', 'ToKeep': 'Yes',
         'ImageQuality': 5, 'CoralDiversity': 3, 'InS3': True},
        {'Source ID': 200, 'Source name': 'b', 'ToKeep': 'No',
         'ImageQuality': 5, 'CoralDiversity': 3, 'InS3': True},
        {'Source ID': 300, 'Source name': 'c', 'ToKeep': 'Yes',
         'ImageQuality': 2, 'CoralDiversity': 0, 'InS3': True},
    ])

    _write_labels_csv(labels, [
        # Top100, ann=1500 (kept).
        {'id': TURF, 'name': 'Turf algae', 'top100': 1.0,
         'CoralNetAnnotations': 1500, 'priority': 1.0,
         'priority_notes': 'top level category', 'parent': ''},
        # Top100, ann=900 (dropped by min_annotations).
        {'id': BLEACHED, 'name': 'Bleached coral', 'top100': 1.0,
         'CoralNetAnnotations': 900, 'priority': '', 'priority_notes': '',
         'parent': 'Hard coral'},
        # Top100, Porites — should get GF buckets.
        {'id': PORITES, 'name': 'Porites', 'top100': 1.0,
         'CoralNetAnnotations': 5000, 'priority': 2.0,
         'priority_notes': 'should be separated by growth form',
         'parent': 'Poritidae'},
        # Non-top100 species under Porites — gets rolled up.
        {'id': PORITES_LOBATA, 'name': 'Porites lobata', 'top100': '',
         'CoralNetAnnotations': 2000, 'priority': '',
         'priority_notes': '', 'parent': 'Porites'},
        # Top100 Acropora — should appear as plain (acropora_id, "").
        {'id': ACROPORA, 'name': 'Acropora', 'top100': 1.0,
         'CoralNetAnnotations': 3000, 'priority': 2.0,
         'priority_notes': '', 'parent': 'Acroporidae'},
        # Non-top100 with priority_notes "rolled up to Mussidae".
        {'id': SYMPHYLLIA, 'name': 'Symphyllia', 'top100': '',
         'CoralNetAnnotations': 50, 'priority': 3.0,
         'priority_notes': 'coral genus; rolled up to Mussidae in current model.',
         'parent': 'Mussidae'},
        # Non-top100 Mussidae — referenced as legacy rollup target.
        {'id': MUSSIDAE, 'name': 'Mussidae', 'top100': '',
         'CoralNetAnnotations': 100, 'priority': 2.0,
         'priority_notes': '', 'parent': 'Hard coral'},
        # A non-top100 sub-label whose parent isn't in the included set —
        # should not appear in any rollup row.
        {'id': SUBLABEL, 'name': 'Some sub-label', 'top100': '',
         'CoralNetAnnotations': 10, 'priority': 3.0,
         'priority_notes': '', 'parent': 'Mussidae'},
    ])

    _write_label_mapping_csv(label_mapping, [
        # Acropora has a Branching mapping -> rolled to (Acropora, '').
        {'benthic attribute': 'Acropora', 'growth form': 'Branching',
         'CoralNetAnnotations': 100, 'provider': 'CoralNet',
         'provider id': '999', 'provider label': 'Acropora branching'},
        # Porites Massive — Porites kept bucket, no rollup.
        {'benthic attribute': 'Porites', 'growth form': 'Massive',
         'CoralNetAnnotations': 200, 'provider': 'CoralNet',
         'provider id': '998', 'provider label': 'Porites massive'},
        # Porites Encrusting — Porites collapsed bucket, rollup needed.
        {'benthic attribute': 'Porites', 'growth form': 'Encrusting',
         'CoralNetAnnotations': 50, 'provider': 'CoralNet',
         'provider id': '997', 'provider label': 'Porites encrusting'},
    ])

    _write_growthforms_csv(growthforms, [
        {'id': GF_BRANCHING, 'name': 'Branching'},
        {'id': GF_MASSIVE, 'name': 'Massive'},
        {'id': GF_ENCRUSTING, 'name': 'Encrusting'},
    ])

    return dict(sources=sources, labels=labels,
                label_mapping=label_mapping, growthforms=growthforms)


# --- Tests ------------------------------------------------------------------

class GenerateTrainingConfigTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.inputs = _make_inputs(self.tmp)
        self.out = self.tmp / 'out'

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, extra_args: list[str] | None = None) -> int:
        argv = [
            '--output-dir', str(self.out),
            '--sources-csv', str(self.inputs['sources']),
            '--labels-csv', str(self.inputs['labels']),
            '--label-mapping-csv', str(self.inputs['label_mapping']),
            '--growthforms-csv', str(self.inputs['growthforms']),
            '--min-annotations', '1000',
            # Skip the S3 reality check; tests don't have AWS access.
            '--no-filter-by-s3-status',
            # Skip pipeline-schema validation in unit tests; we exercise it
            # in a dedicated test below to keep this fast.
            '--skip-validation',
        ]
        if extra_args:
            argv.extend(extra_args)
        return gtc.main(argv)

    def _read_csv(self, name: str) -> list[dict]:
        with (self.out / name).open() as f:
            return list(csv.DictReader(f))

    # -- sources --

    def test_sources_filtered_by_tokeep(self):
        self._run()
        rows = self._read_csv('sources.csv')
        ids = {r['id'] for r in rows}
        # Only ToKeep=Yes survive (ID 100, 300). With default min_image_quality=0,
        # both pass.
        self.assertEqual(ids, {'100', '300'})

    def test_sources_min_image_quality(self):
        self._run(['--min-image-quality', '4'])
        rows = self._read_csv('sources.csv')
        # Only ID 100 has ImageQuality=5; ID 300 has 2.
        self.assertEqual([r['id'] for r in rows], ['100'])

    def test_sources_filtered_by_s3_status_csv(self):
        # Build a tiny S3-status CSV that drops ID 100 but keeps 300.
        status = self.tmp / 's3_status.csv'
        status.write_text('id,has_annotations_csv\n100,False\n300,True\n')

        argv = [
            '--output-dir', str(self.out),
            '--sources-csv', str(self.inputs['sources']),
            '--labels-csv', str(self.inputs['labels']),
            '--label-mapping-csv', str(self.inputs['label_mapping']),
            '--growthforms-csv', str(self.inputs['growthforms']),
            '--s3-status-csv', str(status),
            '--min-annotations', '1000',
            '--skip-validation',
        ]
        gtc.main(argv)
        rows = self._read_csv('sources.csv')
        self.assertEqual([r['id'] for r in rows], ['300'])

    # -- included labels --

    def test_included_labels_top100_and_threshold(self):
        self._run()
        rows = self._read_csv('included_labels.csv')

        # Bleached coral (top100 but <1000 ann + defensive exclude) dropped.
        ba_ids = [r['ba_id'] for r in rows]
        self.assertNotIn(BLEACHED, ba_ids)

        # Turf algae and Acropora present as (ba, "").
        self.assertIn({'ba_id': TURF, 'gf_id': ''}, rows)
        self.assertIn({'ba_id': ACROPORA, 'gf_id': ''}, rows)

        # Porites contributes 3 rows: Branching, Massive, "" (in some order).
        porites_rows = [r for r in rows if r['ba_id'] == PORITES]
        gf_ids = sorted(r['gf_id'] for r in porites_rows)
        self.assertEqual(gf_ids, sorted([GF_BRANCHING, GF_MASSIVE, '']))

    # -- rollups --

    def test_rollups_species_to_genus(self):
        self._run()
        rows = self._read_csv('rollups.csv')
        # Porites lobata (no GF) -> Porites (no GF).
        self.assertIn(
            {'from_ba_id': PORITES_LOBATA, 'from_gf_id': '',
             'to_ba_id': PORITES, 'to_gf_id': ''},
            rows,
        )

    def test_rollups_nonporites_genus_gf_collapsed(self):
        self._run()
        rows = self._read_csv('rollups.csv')
        # (Acropora, Branching) -> (Acropora, '').
        self.assertIn(
            {'from_ba_id': ACROPORA, 'from_gf_id': GF_BRANCHING,
             'to_ba_id': ACROPORA, 'to_gf_id': ''},
            rows,
        )

    def test_rollups_porites_encrusting_collapsed(self):
        self._run()
        rows = self._read_csv('rollups.csv')
        self.assertIn(
            {'from_ba_id': PORITES, 'from_gf_id': GF_ENCRUSTING,
             'to_ba_id': PORITES, 'to_gf_id': ''},
            rows,
        )

    def test_rollups_porites_kept_buckets_have_no_rollup(self):
        self._run()
        rows = self._read_csv('rollups.csv')
        # Massive is a kept bucket -> no rollup row from (Porites, Massive).
        rollups_from_porites_massive = [
            r for r in rows
            if r['from_ba_id'] == PORITES and r['from_gf_id'] == GF_MASSIVE
        ]
        self.assertEqual(rollups_from_porites_massive, [])

    def test_rollups_legacy_priority_notes(self):
        self._run()
        rows = self._read_csv('rollups.csv')
        # Symphyllia has 'rolled up to Mussidae' in priority_notes ->
        # (Symphyllia, '') -> (Mussidae, '').
        self.assertIn(
            {'from_ba_id': SYMPHYLLIA, 'from_gf_id': '',
             'to_ba_id': MUSSIDAE, 'to_gf_id': ''},
            rows,
        )

    def test_rollups_orphan_subtree_not_included(self):
        self._run()
        rows = self._read_csv('rollups.csv')
        # Some sub-label's parent (Mussidae) isn't in the included set, so
        # we should NOT emit a species->genus rollup for it.
        self.assertNotIn(
            {'from_ba_id': SUBLABEL, 'from_gf_id': '',
             'to_ba_id': MUSSIDAE, 'to_gf_id': ''},
            rows,
        )

    # -- README --

    def test_readme_written(self):
        self._run()
        readme = (self.out / 'README.md').read_text()
        self.assertIn('# Training Config:', readme)
        self.assertIn('--min-annotations', readme)
        self.assertIn('Source-of-truth inputs', readme)

    # -- pipeline schema round-trip --

    def test_outputs_validate_against_pipeline_schemas(self):
        # No --skip-validation; this exercises CNSourceFilter / LabelFilter /
        # LabelRollupSpec construction over the produced CSVs.
        argv = [
            '--output-dir', str(self.out),
            '--sources-csv', str(self.inputs['sources']),
            '--labels-csv', str(self.inputs['labels']),
            '--label-mapping-csv', str(self.inputs['label_mapping']),
            '--growthforms-csv', str(self.inputs['growthforms']),
            '--min-annotations', '1000',
        ]
        self.assertEqual(gtc.main(argv), 0)


if __name__ == '__main__':
    unittest.main()
