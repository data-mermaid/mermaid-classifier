"""Unit tests for scripts/launch_feature_extraction_sagemaker.py.

The launcher fans out the existing build_feature_bucket.py script across
N SageMaker Processing Jobs by sharding source IDs. These tests cover
the local logic (chunking, request shaping, polling) with a fully
mocked SageMaker client -- no AWS calls.
"""
from __future__ import annotations

import argparse
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Make scripts/ importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_feature_extraction_sagemaker as lfe  # noqa: E402


# ---- chunk_sources --------------------------------------------------


class ChunkSourcesTest(unittest.TestCase):

    def test_even_split(self):
        sources = [str(i) for i in range(1, 17)]  # 16 sources
        chunks = lfe.chunk_sources(sources, n_workers=4)
        self.assertEqual(len(chunks), 4)
        for c in chunks:
            self.assertEqual(len(c), 4)
        # All sources covered, no duplicates.
        flat = [s for c in chunks for s in c]
        self.assertEqual(sorted(flat, key=int), sorted(sources, key=int))

    def test_uneven_split(self):
        # 10 sources / 3 workers -> 4, 3, 3.
        sources = [str(i) for i in range(1, 11)]
        chunks = lfe.chunk_sources(sources, n_workers=3)
        self.assertEqual([len(c) for c in chunks], [4, 3, 3])
        flat = sorted([s for c in chunks for s in c], key=int)
        self.assertEqual(flat, sorted(sources, key=int))

    def test_more_workers_than_sources_drops_empty_chunks(self):
        sources = ['7', '8', '9']
        chunks = lfe.chunk_sources(sources, n_workers=10)
        # Only 3 non-empty chunks should be returned.
        self.assertEqual(len(chunks), 3)
        for c in chunks:
            self.assertEqual(len(c), 1)

    def test_single_worker_returns_single_chunk(self):
        sources = ['5', '12', '7']
        chunks = lfe.chunk_sources(sources, n_workers=1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(sorted(chunks[0], key=int), ['5', '7', '12'])

    def test_zero_workers_raises(self):
        with self.assertRaises(ValueError):
            lfe.chunk_sources(['1', '2'], n_workers=0)

    def test_negative_workers_raises(self):
        with self.assertRaises(ValueError):
            lfe.chunk_sources(['1', '2'], n_workers=-1)

    def test_192_sources_into_16_workers(self):
        # The expected production case.
        sources = [str(i) for i in range(1, 193)]
        chunks = lfe.chunk_sources(sources, n_workers=16)
        self.assertEqual(len(chunks), 16)
        for c in chunks:
            self.assertEqual(len(c), 12)


# ---- build_processing_job_request -----------------------------------


def _base_request_kwargs(**overrides):
    defaults = dict(
        job_name='mermaid-features-20260514T000000Z-0',
        worker_idx=0,
        source_ids=['7', '12'],
        target_bucket='tgt-bucket',
        weights_uri='s3://w/efficientnet.pt',
        ecr_image='1234.dkr.ecr.us-east-1.amazonaws.com/mermaid-features:latest',
        role_arn='arn:aws:iam::1234:role/MermaidFeatureExtractionRole',
        instance_type='ml.g5.xlarge',
        volume_gb=100,
        max_runtime_s=43200,
        run_id='20260514T000000Z',
    )
    defaults.update(overrides)
    return defaults


class BuildProcessingJobRequestTest(unittest.TestCase):

    def test_top_level_fields(self):
        req = lfe.build_processing_job_request(**_base_request_kwargs())
        self.assertEqual(req['ProcessingJobName'],
                         'mermaid-features-20260514T000000Z-0')
        self.assertEqual(req['RoleArn'],
                         'arn:aws:iam::1234:role/MermaidFeatureExtractionRole')

    def test_image_uri(self):
        req = lfe.build_processing_job_request(**_base_request_kwargs())
        self.assertEqual(
            req['AppSpecification']['ImageUri'],
            '1234.dkr.ecr.us-east-1.amazonaws.com/mermaid-features:latest',
        )

    def test_container_arguments_include_source_ids_comma_joined(self):
        req = lfe.build_processing_job_request(**_base_request_kwargs(
            source_ids=['7', '12', '99'],
        ))
        args = req['AppSpecification']['ContainerArguments']
        # The args list must contain --source-ids followed by a comma-joined value.
        idx = args.index('--source-ids')
        self.assertEqual(args[idx + 1], '7,12,99')

    def test_container_arguments_include_required_flags(self):
        req = lfe.build_processing_job_request(**_base_request_kwargs())
        args = req['AppSpecification']['ContainerArguments']
        # Must pass: target bucket, weights, device=cuda, skip-existing.
        self.assertIn('--target-bucket', args)
        self.assertEqual(args[args.index('--target-bucket') + 1], 'tgt-bucket')
        self.assertIn('--weights', args)
        self.assertEqual(args[args.index('--weights') + 1],
                         's3://w/efficientnet.pt')
        self.assertIn('--device', args)
        self.assertEqual(args[args.index('--device') + 1], 'cuda')
        self.assertIn('--skip-existing', args)

    def test_container_arguments_disable_aws_profile_bootstrap(self):
        # In-container, the task role provides credentials directly; we
        # pass --no-aws-bootstrap so build_feature_bucket.py skips its
        # SSO bootstrap path. (SageMaker rejects empty strings in
        # ContainerArguments, ruling out the obvious --aws-profile "".)
        req = lfe.build_processing_job_request(**_base_request_kwargs())
        args = req['AppSpecification']['ContainerArguments']
        self.assertIn('--no-aws-bootstrap', args)
        self.assertNotIn('--aws-profile', args)

    def test_container_arguments_have_no_empty_strings(self):
        # SageMaker's CreateProcessingJob validation rejects null/empty
        # strings in ContainerArguments. Guard against regressions.
        req = lfe.build_processing_job_request(**_base_request_kwargs())
        args = req['AppSpecification']['ContainerArguments']
        for i, a in enumerate(args):
            self.assertIsInstance(a, str, f'arg {i} is not a string: {a!r}')
            self.assertNotEqual(a, '', f'arg {i} is an empty string')

    def test_cluster_config(self):
        req = lfe.build_processing_job_request(**_base_request_kwargs(
            instance_type='ml.g5.xlarge', volume_gb=100,
        ))
        cluster = req['ProcessingResources']['ClusterConfig']
        self.assertEqual(cluster['InstanceCount'], 1)
        self.assertEqual(cluster['InstanceType'], 'ml.g5.xlarge')
        self.assertEqual(cluster['VolumeSizeInGB'], 100)

    def test_stopping_condition(self):
        req = lfe.build_processing_job_request(**_base_request_kwargs(
            max_runtime_s=3600,
        ))
        self.assertEqual(
            req['StoppingCondition']['MaxRuntimeInSeconds'], 3600)

    def test_tags_include_run_id_and_worker_idx(self):
        req = lfe.build_processing_job_request(**_base_request_kwargs(
            worker_idx=5, run_id='RUN-42',
        ))
        tags = {t['Key']: t['Value'] for t in req['Tags']}
        self.assertEqual(tags.get('Project'), 'mermaid-features')
        self.assertEqual(tags.get('RunId'), 'RUN-42')
        self.assertEqual(tags.get('WorkerIdx'), '5')


# ---- submit_jobs ----------------------------------------------------


class SubmitJobsTest(unittest.TestCase):

    def test_calls_client_once_per_request(self):
        client = mock.MagicMock()
        client.create_processing_job.return_value = {
            'ProcessingJobArn': 'arn:.../job'}
        reqs = [
            _base_request_kwargs(job_name=f'job-{i}', worker_idx=i)
            for i in range(4)
        ]
        # build_processing_job_request returns dicts; submit_jobs takes
        # those dicts and posts each to client.create_processing_job.
        request_dicts = [
            lfe.build_processing_job_request(**r) for r in reqs
        ]
        names = lfe.submit_jobs(client, request_dicts)
        self.assertEqual(client.create_processing_job.call_count, 4)
        self.assertEqual(sorted(names), [f'job-{i}' for i in range(4)])

    def test_propagates_errors(self):
        client = mock.MagicMock()
        client.create_processing_job.side_effect = RuntimeError('boom')
        reqs = [lfe.build_processing_job_request(**_base_request_kwargs())]
        with self.assertRaises(RuntimeError):
            lfe.submit_jobs(client, reqs)


# ---- wait_for_completion --------------------------------------------


class WaitForCompletionTest(unittest.TestCase):

    def test_returns_immediately_when_all_jobs_already_terminal(self):
        client = mock.MagicMock()
        client.describe_processing_job.return_value = {
            'ProcessingJobStatus': 'Completed'}
        names = ['j1', 'j2', 'j3']
        with mock.patch.object(lfe.time, 'sleep') as sleep_mock:
            status = lfe.wait_for_completion(
                client, names, poll_interval_s=999)
        self.assertEqual(set(status.values()), {'Completed'})
        # No sleep on the first pass when everything is already done.
        sleep_mock.assert_not_called()

    def test_polls_until_all_terminal(self):
        client = mock.MagicMock()
        # First pass: InProgress for both. Second pass: one Completed,
        # one Failed. Loop exits.
        responses = [
            {'ProcessingJobStatus': 'InProgress'},
            {'ProcessingJobStatus': 'InProgress'},
            {'ProcessingJobStatus': 'Completed'},
            {'ProcessingJobStatus': 'Failed'},
        ]
        client.describe_processing_job.side_effect = responses
        with mock.patch.object(lfe.time, 'sleep'):
            status = lfe.wait_for_completion(
                client, ['j1', 'j2'], poll_interval_s=1)
        self.assertEqual(status, {'j1': 'Completed', 'j2': 'Failed'})

    def test_stopped_is_terminal(self):
        client = mock.MagicMock()
        client.describe_processing_job.return_value = {
            'ProcessingJobStatus': 'Stopped'}
        with mock.patch.object(lfe.time, 'sleep'):
            status = lfe.wait_for_completion(
                client, ['j1'], poll_interval_s=1)
        self.assertEqual(status, {'j1': 'Stopped'})

    def test_terminal_jobs_are_not_repolled(self):
        # Once a job reaches a terminal state, we should not call
        # describe_processing_job on it again. This matters at scale --
        # don't burn API calls on jobs that are already done.
        client = mock.MagicMock()
        # j1 finishes immediately; j2 is in-progress, then completes.
        def describe_side_effect(ProcessingJobName):
            calls_for_job = j_calls.setdefault(ProcessingJobName, 0)
            j_calls[ProcessingJobName] += 1
            if ProcessingJobName == 'j1':
                return {'ProcessingJobStatus': 'Completed'}
            # j2: first call InProgress, second call Completed.
            if calls_for_job == 0:
                return {'ProcessingJobStatus': 'InProgress'}
            return {'ProcessingJobStatus': 'Completed'}
        j_calls = {}
        client.describe_processing_job.side_effect = describe_side_effect
        with mock.patch.object(lfe.time, 'sleep'):
            status = lfe.wait_for_completion(
                client, ['j1', 'j2'], poll_interval_s=1)
        # j1 only polled once; j2 polled twice.
        self.assertEqual(j_calls['j1'], 1)
        self.assertEqual(j_calls['j2'], 2)
        self.assertEqual(status, {'j1': 'Completed', 'j2': 'Completed'})


# ---- estimate_cost --------------------------------------------------


class EstimateCostTest(unittest.TestCase):

    def test_g5_xlarge_math(self):
        # 16 workers * $1.408/hr * 3h = $67.58
        cost = lfe.estimate_cost(
            n_workers=16, instance_type='ml.g5.xlarge', est_hours=3.0)
        self.assertAlmostEqual(cost, 16 * 1.408 * 3.0, places=2)

    def test_unknown_instance_returns_zero(self):
        cost = lfe.estimate_cost(
            n_workers=4, instance_type='ml.totally-fake', est_hours=2.0)
        self.assertEqual(cost, 0.0)


# ---- job-name generation --------------------------------------------


class JobNamingTest(unittest.TestCase):

    def test_job_name_includes_run_id_and_worker_idx(self):
        name = lfe.make_job_name(run_id='20260514T000000Z', worker_idx=3)
        self.assertIn('20260514T000000Z', name)
        self.assertTrue(name.endswith('-3'))

    def test_make_run_id_is_utc_iso_compact(self):
        # Should be a 16-char-ish ISO-without-separators UTC stamp,
        # e.g. 20260514T000000Z. Used as a SageMaker tag value and
        # embedded in job names (which have a strict charset).
        run_id = lfe.make_run_id()
        self.assertRegex(run_id, r'^\d{8}T\d{6}Z$')

    def test_job_name_charset_safe_for_sagemaker(self):
        # SageMaker requires [a-zA-Z0-9-]; no underscores or dots.
        name = lfe.make_job_name(run_id='20260514T000000Z', worker_idx=12)
        self.assertRegex(name, r'^[a-zA-Z0-9-]+$')


# ---- load_source_ids ------------------------------------------------


class LoadSourceIdsTest(unittest.TestCase):

    def test_comma_separated_string(self):
        args = argparse.Namespace(
            sources_csv=None, source_ids='7,12,99', source_id_column=None)
        ids = lfe.load_source_ids(args)
        self.assertEqual(ids, ['7', '12', '99'])

    def test_comma_separated_strips_whitespace_and_blanks(self):
        args = argparse.Namespace(
            sources_csv=None, source_ids=' 7 , ,12, 99 ',
            source_id_column=None)
        ids = lfe.load_source_ids(args)
        self.assertEqual(ids, ['7', '12', '99'])

    def test_csv_path(self):
        f = tempfile.NamedTemporaryFile(
            'w', suffix='.csv', delete=False, newline='')
        f.write('Source ID,Source name\n')
        f.write('1968,Biofouling\n')
        f.write('2855,Kauai\n')
        f.close()
        args = argparse.Namespace(
            sources_csv=Path(f.name), source_ids=None,
            source_id_column=None)
        ids = lfe.load_source_ids(args)
        self.assertEqual(sorted(ids, key=int), ['1968', '2855'])

    def test_missing_both_raises(self):
        args = argparse.Namespace(
            sources_csv=None, source_ids=None, source_id_column=None)
        with self.assertRaises(ValueError):
            lfe.load_source_ids(args)


# ---- main() dry-run --------------------------------------------------


class MainDryRunTest(unittest.TestCase):

    def test_dry_run_does_not_submit(self):
        # Patch boto3.client so any accidental submission would be
        # detected; --dry-run should never touch it.
        with mock.patch.object(lfe.boto3, 'client') as client_factory:
            fake_client = mock.MagicMock()
            client_factory.return_value = fake_client
            rc = lfe.main([
                '--source-ids', '1,2,3,4',
                '--target-bucket', 'tgt',
                '--weights-s3-uri', 's3://w/e.pt',
                '--ecr-image', 'img:latest',
                '--role-arn', 'arn:aws:iam::1:role/r',
                '--workers', '2',
                '--dry-run',
            ])
        self.assertEqual(rc, 0)
        fake_client.create_processing_job.assert_not_called()

    def test_dry_run_prints_plan_to_stdout(self):
        captured = io.StringIO()
        with mock.patch.object(lfe.sys, 'stdout', captured):
            with mock.patch.object(lfe.boto3, 'client'):
                lfe.main([
                    '--source-ids', '1,2,3,4',
                    '--target-bucket', 'tgt',
                    '--weights-s3-uri', 's3://w/e.pt',
                    '--ecr-image', 'img:latest',
                    '--role-arn', 'arn:aws:iam::1:role/r',
                    '--workers', '2',
                    '--dry-run',
                ])
        out = captured.getvalue()
        # Plan should report worker count and estimated cost.
        self.assertIn('2 worker', out)
        self.assertIn('Estimated cost', out)


# ---- main() submit path ---------------------------------------------


class MainSubmitTest(unittest.TestCase):

    def test_submits_one_job_per_worker(self):
        with mock.patch.object(lfe.boto3, 'client') as client_factory:
            fake_client = mock.MagicMock()
            fake_client.create_processing_job.return_value = {
                'ProcessingJobArn': 'arn:.../job'}
            fake_client.describe_processing_job.return_value = {
                'ProcessingJobStatus': 'Completed'}
            client_factory.return_value = fake_client
            with mock.patch.object(lfe.time, 'sleep'):
                rc = lfe.main([
                    '--source-ids', '1,2,3,4,5,6,7,8',
                    '--target-bucket', 'tgt',
                    '--weights-s3-uri', 's3://w/e.pt',
                    '--ecr-image', 'img:latest',
                    '--role-arn', 'arn:aws:iam::1:role/r',
                    '--workers', '4',
                ])
        # Should have submitted exactly 4 jobs (one per chunk).
        self.assertEqual(rc, 0)
        self.assertEqual(fake_client.create_processing_job.call_count, 4)
        # And polled each to completion.
        self.assertGreaterEqual(
            fake_client.describe_processing_job.call_count, 4)


if __name__ == '__main__':
    unittest.main()
