import dataclasses
import unittest

from mermaid_classifier.model_review.image_set import IMAGE_SET, ImageSet


class ImageSetTest(unittest.TestCase):
    def test_default_image_set_matches_current_setup_except_20_images(self):
        self.assertIsInstance(IMAGE_SET, ImageSet)
        # First test run: 20 images (was 50), everything else = current defaults.
        self.assertEqual(IMAGE_SET.n_images, 20)
        self.assertEqual(IMAGE_SET.min_points, 15)
        self.assertEqual(IMAGE_SET.seed, 1)
        self.assertEqual(IMAGE_SET.classifier, "s3://mermaid-config/classifier/v2/")
        self.assertEqual(IMAGE_SET.name, "model-review")
        self.assertEqual(IMAGE_SET.image_bucket, "dev-datamermaid-sm-sources")
        self.assertEqual(IMAGE_SET.feature_bucket, "2605-coralnet-public-sources")
        self.assertEqual(
            IMAGE_SET.image_key_template,
            "coralnet-public-images/s{source_id}/images/{image_id}.jpg",
        )

    def test_image_set_is_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            IMAGE_SET.n_images = 99  # type: ignore[misc]
