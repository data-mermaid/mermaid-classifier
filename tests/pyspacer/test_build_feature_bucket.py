"""Unit tests for scripts/build_feature_bucket.py.

All S3 and pyspacer interactions are mocked. We don't run the real
EfficientNet extractor here -- that's covered upstream in pyspacer's
own test suite.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

# Make scripts/ importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_feature_bucket as bfb  # noqa: E402


# ---- helpers --------------------------------------------------------


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        target_bucket='tgt-bucket',
        source_bucket='src-bucket',
        source_prefix='coralnet-public-images/',
        weights=None,
        max_io_workers=4,
        aws_profile='wcs',
        skip_existing=True,
        dry_run=False,
        error_log=None,
        progress_log=None,
        log_level='INFO',
        sources_csv=None,
        source_ids=None,
        source_id_column=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class FakeProgressWriter:
    def __init__(self):
        self.records = []

    def write(self, s):
        self.records.append(json.loads(s.strip()))

    def flush(self):
        pass


class FakeErrorWriter:
    def __init__(self):
        self.rows = []

    def writerow(self, row):
        self.rows.append(row)


# ---- 1) Source-ID CSV parsing --------------------------------------


class SourceIdCsvParsingTest(unittest.TestCase):

    def _write_csv(self, header: str, rows: list[str]) -> Path:
        f = tempfile.NamedTemporaryFile('w', suffix='.csv',
                                        delete=False, newline='')
        f.write(header + '\n')
        for r in rows:
            f.write(r + '\n')
        f.close()
        return Path(f.name)

    def test_id_column(self):
        path = self._write_csv('id,', ['23,', '57,', '99,'])
        ids = bfb.load_source_ids_from_csv(path, override=None)
        self.assertEqual(ids, ['23', '57', '99'])

    def test_source_id_column_with_space(self):
        # Matches the "Source ID" header used in coralnet_best_sources CSV.
        path = self._write_csv('Source ID,Source name',
                               ['1968,Biofouling', '2855,Kauai', '3581,NFWF'])
        ids = bfb.load_source_ids_from_csv(path, override=None)
        self.assertEqual(ids, ['1968', '2855', '3581'])

    def test_underscore_source_id_column(self):
        path = self._write_csv('source_id,note', ['12,a', '34,b'])
        ids = bfb.load_source_ids_from_csv(path, override=None)
        self.assertEqual(ids, ['12', '34'])

    def test_missing_column_raises(self):
        path = self._write_csv('foo,bar', ['1,2', '3,4'])
        with self.assertRaises(ValueError) as cm:
            bfb.load_source_ids_from_csv(path, override=None)
        self.assertIn('source-ID column', str(cm.exception))

    def test_explicit_override(self):
        path = self._write_csv('SID,note', ['7,x', '8,y'])
        ids = bfb.load_source_ids_from_csv(path, override='SID')
        self.assertEqual(ids, ['7', '8'])

    def test_duplicates_dropped_preserve_order(self):
        path = self._write_csv('id,', ['5,', '7,', '5,', '9,'])
        ids = bfb.load_source_ids_from_csv(path, override=None)
        self.assertEqual(ids, ['5', '7', '9'])


# ---- 2) URI construction -------------------------------------------


class BuildExtractMsgTest(unittest.TestCase):

    def test_uri_construction(self):
        # We stub the pyspacer imports inside build_extract_msg by
        # patching the symbols it looks up at call time.
        with mock.patch.dict(sys.modules) as mods:
            fake_dc = mock.MagicMock()
            fake_msg = mock.MagicMock()
            mods['spacer.data_classes'] = fake_dc
            mods['spacer.messages'] = fake_msg

            DataLocation = fake_dc.DataLocation
            ExtractFeaturesMsg = fake_msg.ExtractFeaturesMsg

            DataLocation.side_effect = lambda **kw: kw
            ExtractFeaturesMsg.side_effect = lambda **kw: kw

            bfb.build_extract_msg(
                source_id='7', image_id='42',
                rowcols=[(10, 20), (30, 40)],
                extractor='STUB',
                source_bucket='src-bucket',
                source_prefix='coralnet-public-images/',
                target_bucket='tgt-bucket',
            )

        # Two DataLocation calls: image_loc then feature_loc.
        image_call = DataLocation.call_args_list[0].kwargs
        feature_call = DataLocation.call_args_list[1].kwargs
        msg_call = ExtractFeaturesMsg.call_args.kwargs

        self.assertEqual(image_call['storage_type'], 's3')
        self.assertEqual(image_call['bucket_name'], 'src-bucket')
        self.assertEqual(image_call['key'],
                         'coralnet-public-images/s7/images/42.jpg')

        self.assertEqual(feature_call['storage_type'], 's3')
        self.assertEqual(feature_call['bucket_name'], 'tgt-bucket')
        self.assertEqual(feature_call['key'],
                         's7/features/i42.featurevector')

        self.assertEqual(msg_call['rowcols'], [(10, 20), (30, 40)])
        self.assertEqual(msg_call['job_token'], 's7_i42')


# ---- shared fakes for process_source tests --------------------------


class _FakeBody:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeObject:
    def __init__(self, payload: bytes):
        self._payload = payload

    def get(self):
        return {'Body': _FakeBody(self._payload)}


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        return iter(self._pages)


class _FakeClient:
    def __init__(self, list_pages, head_exists=False):
        self.list_pages = list_pages
        self.head_exists = head_exists
        self.head_calls: list[tuple[str, str]] = []
        self.put_calls: list[dict] = []

    def get_paginator(self, _name):
        return _FakePaginator(self.list_pages)

    def head_object(self, Bucket, Key):
        self.head_calls.append((Bucket, Key))
        if self.head_exists:
            return {}
        from botocore.exceptions import ClientError
        raise ClientError(
            {'Error': {'Code': '404', 'Message': 'Not Found'}}, 'HeadObject')

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        return {}


class _FakeS3:
    """Returns a _FakeObject keyed off the (in-bucket) key.

    Pass `key_to_payload` mapping keys (without bucket) to bytes. A simple
    bytes value is shorthand for "any key returns this payload".
    """

    def __init__(self, client, key_to_payload):
        self.meta = types.SimpleNamespace(client=client)
        if isinstance(key_to_payload, (bytes, bytearray)):
            self._default = bytes(key_to_payload)
            self._mapping: dict[str, bytes] = {}
        else:
            self._default = None
            self._mapping = dict(key_to_payload)

    def Object(self, bucket, key):
        if key in self._mapping:
            return _FakeObject(self._mapping[key])
        if self._default is not None:
            return _FakeObject(self._default)
        raise KeyError(f"FakeS3 has no payload for key {key!r}")


def _make_csv_payload(image_rows: dict[str, list[tuple[int, int]]]) -> bytes:
    """Build an annotations.csv payload with an `Image ID` column."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['Image ID', 'Row', 'Column', 'Label ID'])
    for image_id, rcs in image_rows.items():
        for r, c in rcs:
            w.writerow([image_id, r, c, 'L'])
    return buf.getvalue().encode('utf-8')


def _make_name_csv_payload(name_rows: dict[str, list[tuple[int, int]]]) -> bytes:
    """Build an annotations.csv payload with the new bucket's `Name` column."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['Name', 'Row', 'Column', 'Label ID'])
    for name, rcs in name_rows.items():
        for r, c in rcs:
            w.writerow([name, r, c, 'L'])
    return buf.getvalue().encode('utf-8')


def _make_image_list_payload(name_to_id: dict[str, str]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['Name', 'Image Page', 'Image URL'])
    for name, iid in name_to_id.items():
        # Mirror the real format: "<filename> - Confirmed" / "/image/<id>/view/".
        w.writerow([f'{name} - Confirmed', f'/image/{iid}/view/', 'https://example/'])
    return buf.getvalue().encode('utf-8')


# ---- 3) Skip-existing -----------------------------------------------


class SkipExistingTest(unittest.TestCase):

    def test_skips_existing_image_only(self):
        csv_payload = _make_csv_payload({
            '42': [(10, 20)],
            '43': [(30, 40)],
        })
        # Pre-existing skip set says image 42 is already done.
        list_pages = [{
            'Contents': [
                {'Key': 's7/features/i42.featurevector'},
            ],
        }]
        client = _FakeClient(list_pages=list_pages, head_exists=True)
        s3 = _FakeS3(client, csv_payload)

        progress = FakeProgressWriter()
        errors = FakeErrorWriter()
        counters = bfb.RunCounters()
        args = make_args()

        fake_extract = mock.MagicMock()
        with mock.patch.dict(sys.modules, {
            'spacer.tasks': mock.MagicMock(extract_features=fake_extract),
            'spacer.data_classes': mock.MagicMock(
                DataLocation=lambda **kw: dict(kw)),
            'spacer.messages': mock.MagicMock(
                ExtractFeaturesMsg=lambda **kw: dict(kw)),
        }):
            bfb.process_source(
                source_id='7', extractor='STUB', s3=s3, args=args,
                counters=counters,
                progress_writer=progress, error_writer=errors)

        self.assertEqual(fake_extract.call_count, 1)
        sent_msg = fake_extract.call_args.args[0]
        self.assertEqual(sent_msg['feature_loc']['key'],
                         's7/features/i43.featurevector')
        # Image 42 should appear as a 'skipped/exists' progress record.
        outcomes = [(r['image_id'], r['outcome']) for r in progress.records]
        self.assertIn(('42', 'skipped'), outcomes)
        self.assertIn(('43', 'ok'), outcomes)


# ---- 4) Resume after crash ------------------------------------------


class ResumeAfterCrashTest(unittest.TestCase):

    def test_resume_finishes_remaining_images(self):
        csv_payload = _make_csv_payload({
            '42': [(1, 1)],
            '43': [(2, 2)],
            '44': [(3, 3)],
        })
        # Skip-set already contains image 42 from a prior run.
        list_pages = [{
            'Contents': [
                {'Key': 's7/features/i42.featurevector'},
            ],
        }]
        # head_exists=True simulates s7/annotations.csv already present in
        # the target bucket -- the copy step must be skipped.
        client = _FakeClient(list_pages=list_pages, head_exists=True)
        s3 = _FakeS3(client, csv_payload)

        progress = FakeProgressWriter()
        errors = FakeErrorWriter()
        counters = bfb.RunCounters()
        args = make_args()

        fake_extract = mock.MagicMock()
        with mock.patch.dict(sys.modules, {
            'spacer.tasks': mock.MagicMock(extract_features=fake_extract),
            'spacer.data_classes': mock.MagicMock(
                DataLocation=lambda **kw: dict(kw)),
            'spacer.messages': mock.MagicMock(
                ExtractFeaturesMsg=lambda **kw: dict(kw)),
        }):
            bfb.process_source(
                source_id='7', extractor='STUB', s3=s3, args=args,
                counters=counters,
                progress_writer=progress, error_writer=errors)

        # Exactly two new extractions: i43 and i44.
        self.assertEqual(fake_extract.call_count, 2)
        keys = [
            call.args[0]['feature_loc']['key']
            for call in fake_extract.call_args_list
        ]
        self.assertEqual(sorted(keys), [
            's7/features/i43.featurevector',
            's7/features/i44.featurevector',
        ])

        # No upload of annotations.csv (target version already present).
        self.assertEqual(client.put_calls, [])

        # Progress log: i42 skipped, i43 + i44 ok.
        outcomes = {(r['image_id'], r['outcome']) for r in progress.records}
        self.assertIn(('42', 'skipped'), outcomes)
        self.assertIn(('43', 'ok'), outcomes)
        self.assertIn(('44', 'ok'), outcomes)

        # Counters reflect the resume.
        self.assertEqual(counters.images_ok, 2)
        self.assertEqual(counters.images_skipped, 1)
        self.assertEqual(counters.images_failed, 0)
        self.assertEqual(counters.annotations_copied, 0)
        self.assertEqual(counters.annotations_skipped, 1)


# ---- 5) Name -> Image ID mapping (new segmentation bucket format) ---


class NameToImageIdMappingTest(unittest.TestCase):

    def test_prepare_source_maps_name_to_numeric_id(self):
        # annotations.csv has Name (no Image ID); image_list.csv has the
        # filename + " - Confirmed" status suffix and an /image/<id>/view/
        # page URL.
        annotations = _make_name_csv_payload({
            '001-CA2M-1.JPG': [(10, 20), (30, 40)],
            '002-CA2M-2.JPG': [(5, 5)],
            'unknown.JPG':    [(0, 0)],   # unmappable -> should be dropped
        })
        image_list = _make_image_list_payload({
            '001-CA2M-1.JPG': '1719202',
            '002-CA2M-2.JPG': '1750080',
        })
        s3 = _FakeS3(
            client=_FakeClient(list_pages=[]),
            key_to_payload={
                'coralnet-public-images/s1968/annotations.csv': annotations,
                'coralnet-public-images/s1968/image_list.csv': image_list,
            },
        )

        prepared = bfb.prepare_source(
            s3=s3, source_bucket='src-bucket',
            source_prefix='coralnet-public-images/', source_id='1968')

        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.n_unmapped_rows, 1)
        # Image IDs come back as strings keyed by the numeric value.
        self.assertEqual(
            sorted(prepared.images.keys()), ['1719202', '1750080'])
        self.assertEqual(prepared.images['1719202'], [(10, 20), (30, 40)])
        self.assertEqual(prepared.images['1750080'], [(5, 5)])

        # The transformed CSV body must contain an `Image ID` column whose
        # values are the looked-up numeric IDs -- this is what the
        # classifier's read_coralnet_data expects.
        out_df = pd.read_csv(io.BytesIO(prepared.transformed_csv))
        self.assertIn('Image ID', out_df.columns)
        self.assertEqual(set(out_df['Image ID'].astype(str)),
                         {'1719202', '1750080'})

    def test_status_suffix_variants_are_stripped(self):
        # image_list.csv may report Unconfirmed / Unclassified statuses too.
        annotations = _make_name_csv_payload({
            'a.JPG': [(1, 1)],
            'b.JPG': [(2, 2)],
            'c.JPG': [(3, 3)],
        })
        # Hand-craft image_list to use varied statuses.
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['Name', 'Image Page', 'Image URL'])
        w.writerow(['a.JPG - Confirmed',    '/image/100/view/', ''])
        w.writerow(['b.JPG - Unconfirmed',  '/image/200/view/', ''])
        w.writerow(['c.JPG - Unclassified', '/image/300/view/', ''])
        image_list = buf.getvalue().encode()

        s3 = _FakeS3(
            client=_FakeClient(list_pages=[]),
            key_to_payload={
                'coralnet-public-images/s7/annotations.csv': annotations,
                'coralnet-public-images/s7/image_list.csv': image_list,
            },
        )
        prepared = bfb.prepare_source(
            s3=s3, source_bucket='src-bucket',
            source_prefix='coralnet-public-images/', source_id='7')

        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.n_unmapped_rows, 0)
        self.assertEqual(sorted(prepared.images.keys()), ['100', '200', '300'])


# ---- 6) resolve_device ---------------------------------------------


class ResolveDeviceTest(unittest.TestCase):

    def _mock_torch(self, mps: bool, cuda: bool):
        return mock.patch.dict(sys.modules, {
            'torch': mock.MagicMock(
                cuda=mock.MagicMock(is_available=mock.MagicMock(return_value=cuda)),
                backends=mock.MagicMock(
                    mps=mock.MagicMock(is_available=mock.MagicMock(return_value=mps)),
                ),
            ),
        })

    def test_auto_prefers_mps(self):
        with self._mock_torch(mps=True, cuda=True):
            self.assertEqual(bfb.resolve_device('auto'), 'mps')

    def test_auto_falls_back_to_cuda(self):
        with self._mock_torch(mps=False, cuda=True):
            self.assertEqual(bfb.resolve_device('auto'), 'cuda')

    def test_auto_falls_back_to_cpu(self):
        with self._mock_torch(mps=False, cuda=False):
            self.assertEqual(bfb.resolve_device('auto'), 'cpu')

    def test_explicit_mps_when_unavailable_raises(self):
        with self._mock_torch(mps=False, cuda=False):
            with self.assertRaises(RuntimeError):
                bfb.resolve_device('mps')

    def test_explicit_cpu_always_works(self):
        with self._mock_torch(mps=False, cuda=False):
            self.assertEqual(bfb.resolve_device('cpu'), 'cpu')


# ---- bonus sanity: parse_weights_location ---------------------------


class ParseWeightsLocationTest(unittest.TestCase):

    def test_s3_uri(self):
        with mock.patch.dict(sys.modules, {
            'spacer.data_classes': mock.MagicMock(
                DataLocation=lambda **kw: dict(kw)),
        }):
            loc = bfb.parse_weights_location('s3://my-bucket/models/eff.pt')
        self.assertEqual(loc, {
            'storage_type': 's3',
            'key': 'models/eff.pt',
            'bucket_name': 'my-bucket',
        })

    def test_filesystem_path(self):
        with mock.patch.dict(sys.modules, {
            'spacer.data_classes': mock.MagicMock(
                DataLocation=lambda **kw: dict(kw)),
        }):
            loc = bfb.parse_weights_location('/tmp/eff.pt')
        self.assertEqual(loc, {
            'storage_type': 'filesystem',
            'key': '/tmp/eff.pt',
        })


if __name__ == '__main__':
    unittest.main()
