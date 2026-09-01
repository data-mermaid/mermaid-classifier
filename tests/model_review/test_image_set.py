import dataclasses
import unittest

from mermaid_classifier.model_review.image_set import IMAGE_SET, ImageSet, SiteSpec
from mermaid_classifier.model_review.sample import allocate_quota


class ImageSetTest(unittest.TestCase):
    def _spec(self, site):
        return next(s for s in IMAGE_SET.sites if s.site == site)

    def test_sampling_knobs(self):
        self.assertIsInstance(IMAGE_SET, ImageSet)
        self.assertEqual(IMAGE_SET.n_images, 20)
        self.assertEqual(IMAGE_SET.min_points, 15)
        self.assertEqual(IMAGE_SET.seed, 1)
        self.assertEqual(IMAGE_SET.classifier, "s3://mermaid-config/classifier/v2/")
        self.assertEqual(IMAGE_SET.beta_classifier, "../models/beta_model/classifier.pkl")

    def test_two_sites_weighted_two_to_one(self):
        self.assertEqual([s.site for s in IMAGE_SET.sites], ["coralnet", "mermaid"])
        self.assertEqual(self._spec("coralnet").weight, 2.0)
        self.assertEqual(self._spec("mermaid").weight, 1.0)

    def test_weights_yield_thirteen_coralnet_and_seven_mermaid(self):
        quota = allocate_quota(IMAGE_SET.n_images, {s.site: s.weight for s in IMAGE_SET.sites})
        self.assertEqual(quota, {"coralnet": 13, "mermaid": 7})

    def test_coralnet_site_layout(self):
        spec = self._spec("coralnet")
        self.assertEqual(spec.image_bucket, "dev-datamermaid-sm-sources")
        self.assertEqual(spec.feature_bucket, "2605-coralnet-public-sources")
        self.assertEqual(spec.image_prefix, "coralnet-public-images/")
        self.assertEqual(
            spec.image_key_template, "coralnet-public-images/s{source_id}/images/{image_id}.jpg"
        )
        self.assertEqual(spec.image_key_extensions, ())  # exact key, no probe

    def test_mermaid_site_layout(self):
        spec = self._spec("mermaid")
        # Display images and feature vectors share the one prefix in the training bucket.
        self.assertEqual(spec.image_bucket, "coral-reef-training")
        self.assertEqual(spec.feature_bucket, "coral-reef-training")
        self.assertEqual(spec.image_prefix, "mermaid/")
        self.assertEqual(spec.image_bucket_region, "us-east-1")
        self.assertEqual(spec.image_key_template, "mermaid/{image_id}.{ext}")
        self.assertEqual(spec.image_key_extensions, ("png", "jpg", "jpeg"))
        self.assertEqual(
            spec.manifest_uri,
            "s3://coral-reef-training/mermaid/mermaid_confirmed_annotations.parquet",
        )

    def test_no_site_key_template_needs_a_source_id_it_lacks(self):
        for spec in IMAGE_SET.sites:
            if "{source_id}" in spec.image_key_template:
                self.assertEqual(spec.site, "coralnet", "only CoralNet images have a source")

    def test_image_set_and_site_spec_are_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            IMAGE_SET.n_images = 99  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            IMAGE_SET.sites[0].weight = 9.0  # type: ignore[misc]

    def test_site_spec_defaults_to_an_exact_key(self):
        spec = SiteSpec("s", 1.0, "u", "fb", "ib", "p/", "us-east-1", "p/{image_id}.jpg")
        self.assertEqual(spec.image_key_extensions, ())
