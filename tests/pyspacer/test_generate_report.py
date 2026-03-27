"""Tests for scripts/generate_report.py.

Tests the pure transformation functions without requiring an MLflow server.
"""
import base64
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import MagicMock

# Adjust sys.path so we can import from scripts/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from generate_report import (
    _artifact_key,
    build_template_context,
    encode_png_as_base64,
    fetch_scalar_metrics,
    load_artifact_data,
    load_csv_as_html_table,
    load_yaml_file,
    render_report,
    EVALUATION_SECTIONS,
)


def _make_minimal_png(path: Path):
    """Write a minimal valid 1x1 white PNG to the given path."""
    # PNG signature
    sig = b'\x89PNG\r\n\x1a\n'

    def _chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)

    # IHDR: 1x1, 8-bit grayscale
    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 0, 0, 0, 0)
    ihdr = _chunk(b'IHDR', ihdr_data)
    # IDAT: single white pixel, filter byte 0
    raw = b'\x00\xff'
    idat = _chunk(b'IDAT', zlib.compress(raw))
    # IEND
    iend = _chunk(b'IEND', b'')

    path.write_bytes(sig + ihdr + idat + iend)


class TestArtifactKey(unittest.TestCase):
    def test_simple_filename(self):
        self.assertEqual(_artifact_key('metrics_per_label.csv'), 'metrics_per_label_csv')

    def test_subdirectory_path(self):
        self.assertEqual(_artifact_key('confusion_matrix/frequencies.png'), 'frequencies_png')

    def test_yaml_extension(self):
        self.assertEqual(_artifact_key('system_specs.yaml'), 'system_specs_yaml')


class TestEncodePngAsBase64(unittest.TestCase):
    def test_returns_data_uri(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / 'test.png'
            _make_minimal_png(png_path)
            result = encode_png_as_base64(png_path)
            self.assertTrue(result.startswith('data:image/png;base64,'))

    def test_base64_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / 'test.png'
            _make_minimal_png(png_path)
            result = encode_png_as_base64(png_path)
            b64_part = result.split(',', 1)[1]
            decoded = base64.b64decode(b64_part)
            # Should start with PNG signature.
            self.assertTrue(decoded.startswith(b'\x89PNG'))


class TestLoadCsvAsHtmlTable(unittest.TestCase):
    def test_basic_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'test.csv'
            csv_path.write_text('name,value\nalpha,0.95\nbeta,0.87\n')
            result = load_csv_as_html_table(csv_path)
            self.assertIn('<table', result)
            self.assertIn('alpha', result)
            self.assertIn('0.9500', result)
            self.assertIn('dataframe', result)

    def test_no_index_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'test.csv'
            csv_path.write_text('a,b\n1,2\n')
            result = load_csv_as_html_table(csv_path)
            # Should not contain a default pandas index column.
            self.assertNotIn('<th></th>', result)


class TestLoadYamlFile(unittest.TestCase):
    def test_basic_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / 'test.yaml'
            yaml_path.write_text('total_ram_gb: 16.0\nfree_storage_gb: 100.5\n')
            result = load_yaml_file(yaml_path)
            self.assertEqual(result['total_ram_gb'], 16.0)
            self.assertEqual(result['free_storage_gb'], 100.5)

    def test_nested_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / 'test.yaml'
            yaml_path.write_text('parent:\n  child: value\n')
            result = load_yaml_file(yaml_path)
            self.assertEqual(result['parent']['child'], 'value')


class TestFetchScalarMetrics(unittest.TestCase):
    def _make_mock_run(self, metrics_dict):
        run = MagicMock()
        run.data.metrics = metrics_dict
        return run

    def test_full_metrics(self):
        metrics = {
            'accuracy': 0.85,
            'balanced_accuracy': 0.82,
            'f1_macro': 0.80,
            'precision_macro': 0.81,
            'recall_macro': 0.79,
            'mcc': 0.75,
            'ece': 0.03,
            'log_loss': 1.2,
            'top_1_accuracy': 0.85,
            'top_3_accuracy': 0.95,
            'top_5_accuracy': 0.97,
            'top_10_accuracy': 0.99,
            'mrr': 0.90,
            'cover_mean_abs_bias_pct': 2.1,
            'cover_mean_rmse_pct': 3.5,
            'cross_branch_error_rate': 0.15,
        }
        run = self._make_mock_run(metrics)
        result = fetch_scalar_metrics(run)

        self.assertIsNotNone(result['executive'])
        self.assertEqual(len(result['executive']), 8)
        self.assertIsNotNone(result['topk'])
        self.assertEqual(len(result['topk']), 5)
        self.assertIsNotNone(result['cover'])
        self.assertEqual(len(result['cover']), 2)  # Only 2 of 4 cover metrics present.
        self.assertIsNotNone(result['taxonomic'])

    def test_minimal_metrics(self):
        """Only executive metrics present, optional groups absent."""
        metrics = {
            'accuracy': 0.85,
            'f1_macro': 0.80,
        }
        run = self._make_mock_run(metrics)
        result = fetch_scalar_metrics(run)

        self.assertIsNotNone(result['executive'])
        self.assertEqual(len(result['executive']), 2)
        self.assertIsNone(result['topk'])
        self.assertIsNone(result['cover'])
        self.assertIsNone(result['taxonomic'])

    def test_empty_metrics(self):
        run = self._make_mock_run({})
        result = fetch_scalar_metrics(run)
        self.assertIsNone(result['executive'])
        self.assertIsNone(result['topk'])


class TestLoadArtifactData(unittest.TestCase):
    def test_with_all_required_sections(self):
        """Create a minimal artifact tree with required sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)

            # Create confusion_matrix artifacts.
            cm_dir = artifact_dir / 'confusion_matrix'
            cm_dir.mkdir()
            _make_minimal_png(cm_dir / 'frequencies.png')
            _make_minimal_png(cm_dir / 'percents.png')

            # Create calibration artifacts.
            cal_dir = artifact_dir / 'calibration'
            cal_dir.mkdir()
            _make_minimal_png(cal_dir / 'reliability_diagram.png')

            # Create taxonomic artifacts.
            tax_dir = artifact_dir / 'taxonomic'
            tax_dir.mkdir()
            _make_minimal_png(tax_dir / 'error_attribution.png')
            _make_minimal_png(tax_dir / 'top_level_confusion.png')
            _make_minimal_png(tax_dir / 'gf_confusion.png')

            result = load_artifact_data(artifact_dir)

            self.assertIn('confusion_matrix', result['sections'])
            self.assertIn('calibration', result['sections'])
            self.assertIn('taxonomic', result['sections'])
            # Optional sections should be absent.
            self.assertNotIn('cover', result['sections'])
            self.assertNotIn('probability', result['sections'])
            self.assertNotIn('ranking', result['sections'])

    def test_optional_section_present(self):
        """Cover section appears when its artifacts exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            cover_dir = artifact_dir / 'cover'
            cover_dir.mkdir()
            _make_minimal_png(cover_dir / 'per_class_bias.png')

            result = load_artifact_data(artifact_dir)
            self.assertIn('cover', result['sections'])
            self.assertIsNotNone(
                result['sections']['cover']['per_class_bias_png'])

    def test_training_artifacts(self):
        """Training artifacts are loaded when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / 'system_specs.yaml').write_text(
                'total_ram_gb: 16\n')
            (artifact_dir / 'train_summary.yaml').write_text(
                'total_images: 5000\n')

            result = load_artifact_data(artifact_dir)
            self.assertTrue(result['has_training'])
            self.assertIsNotNone(result['training']['system_specs_yaml'])
            self.assertEqual(
                result['training']['system_specs_yaml']['total_ram_gb'], 16)


class TestBuildTemplateContext(unittest.TestCase):
    def test_basic_context(self):
        metadata = {
            'run_id': 'abc123',
            'run_name': 'test',
            'experiment_name': 'exp1',
        }
        metrics = {'executive': [('Accuracy', 0.85)], 'topk': None,
                   'cover': None, 'taxonomic': None}
        artifacts = {
            'sections': {},
            'root_eval': {'metrics_per_label_csv': None,
                          'metrics_overall_yaml': None},
            'training': {},
            'has_training': False,
        }

        context = build_template_context(metadata, metrics, artifacts)
        self.assertEqual(context['title'],
                         'Classifier Report - exp1 - test')
        self.assertIn('generated_at', context)
        self.assertEqual(context['metadata'], metadata)

    def test_custom_title(self):
        metadata = {'run_id': 'x', 'run_name': 'y',
                    'experiment_name': 'z'}
        metrics = {'executive': None, 'topk': None,
                   'cover': None, 'taxonomic': None}
        artifacts = {'sections': {}, 'root_eval': {}, 'training': {},
                     'has_training': False}

        context = build_template_context(
            metadata, metrics, artifacts, title='Custom Title')
        self.assertEqual(context['title'], 'Custom Title')


class TestRenderReport(unittest.TestCase):
    def test_renders_valid_html(self):
        """Render with a minimal context and verify output is HTML."""
        context = {
            'title': 'Test Report',
            'generated_at': '2025-01-01 00:00 UTC',
            'metadata': {
                'run_id': 'abc123def456',
                'run_name': 'test-run',
                'experiment_name': 'test-experiment',
                'status': 'FINISHED',
                'start_time': '2025-01-01 00:00 UTC',
                'end_time': '2025-01-01 01:00 UTC',
                'duration': '1h 0m 0s',
                'params': {'epochs': '10'},
                'tags': {},
            },
            'metrics': {
                'executive': [('Accuracy', 0.85), ('F1 (Macro)', 0.80)],
                'topk': None,
                'cover': None,
                'taxonomic': None,
            },
            'sections': {},
            'root_eval': {
                'metrics_per_label_csv': None,
                'metrics_overall_yaml': None,
            },
            'training': {},
            'has_training': False,
            'section_order': ['confusion_matrix', 'calibration', 'cover',
                              'probability', 'ranking', 'taxonomic'],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'report.html'
            render_report(context, output_path)

            self.assertTrue(output_path.exists())
            html = output_path.read_text()
            self.assertIn('<!DOCTYPE html>', html)
            self.assertIn('Test Report', html)
            self.assertIn('0.8500', html)  # F1 macro formatted
            self.assertIn('abc123def456', html)

    def test_conditional_sections_absent(self):
        """Sections with no data should not appear in output."""
        context = {
            'title': 'Test',
            'generated_at': '2025-01-01',
            'metadata': {
                'run_id': 'x', 'run_name': 'y',
                'experiment_name': 'z', 'status': 'FINISHED',
                'start_time': 'N/A', 'end_time': 'N/A',
                'duration': 'N/A', 'params': {}, 'tags': {},
            },
            'metrics': {'executive': None, 'topk': None,
                        'cover': None, 'taxonomic': None},
            'sections': {},
            'root_eval': {'metrics_per_label_csv': None,
                          'metrics_overall_yaml': None},
            'training': {},
            'has_training': False,
            'section_order': ['confusion_matrix', 'calibration', 'cover',
                              'probability', 'ranking', 'taxonomic'],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'report.html'
            render_report(context, output_path)
            html = output_path.read_text()

            # No evaluation sections should be rendered.
            self.assertNotIn('id="confusion_matrix"', html)
            self.assertNotIn('id="cover"', html)
            self.assertNotIn('id="ranking"', html)
            # No training section.
            self.assertNotIn('id="run-configuration"', html)
            # No per-label detail.
            self.assertNotIn('id="per-label-detail"', html)


if __name__ == '__main__':
    unittest.main()
