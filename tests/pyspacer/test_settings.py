"""
Tests for mermaid_classifier.pyspacer.settings.training_batch_size.

training_batch_size() sizes partial_fit batches from currently available
RAM, per-point feature/activation memory, and a fixed usable-memory
headroom fraction. Each case patches psutil.virtual_memory().available to
a fixed byte count and asserts a batch size derived by hand from the
documented formula, so a wrong constant or a wrong headroom factor fails
the test.
"""

import unittest
from unittest import mock

from mermaid_classifier.pyspacer.settings import training_batch_size


class TrainingBatchSizeTest(unittest.TestCase):
    def _mock_available(self, available_bytes):
        mock_vm = mock.Mock(available=available_bytes)
        return mock.patch(
            "mermaid_classifier.pyspacer.settings.psutil.virtual_memory",
            return_value=mock_vm,
        )

    def test_batch_size_at_default_num_classes(self):
        # bytes_per_point = 2*(1280*8) + 16*(900+300) = 39680;
        # batch = int(16_000_000_000*0.80 / 39680) = 322580.
        with self._mock_available(16_000_000_000):
            batch_size, available_gb = training_batch_size()
        self.assertEqual(batch_size, 322580)
        self.assertEqual(available_gb, 16.0)

    def test_batch_size_at_v1_num_classes(self):
        # bytes_per_point = 2*(1280*8) + 16*(900+95) = 36400;
        # batch = int(8_000_000_000*0.80 / 36400) = 175824.
        with self._mock_available(8_000_000_000):
            batch_size, available_gb = training_batch_size(num_classes=95)
        self.assertEqual(batch_size, 175824)
        self.assertEqual(available_gb, 8.0)

    def test_batch_size_floors_at_min_batch_size_under_tiny_memory(self):
        # bytes_per_point = 39680 (num_classes=300); int(1_000_000*0.80 /
        # 39680) = 20, below _MIN_BATCH_SIZE, so the floor applies.
        with self._mock_available(1_000_000):
            batch_size, _ = training_batch_size()
        self.assertEqual(batch_size, 5000)


if __name__ == "__main__":
    unittest.main()
