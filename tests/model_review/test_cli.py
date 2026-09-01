import io
import unittest

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary,
    GrowthFormLibrary,
)
from mermaid_classifier.model_review import cli
from mermaid_classifier.model_review.sample import ReviewPoint

_BA = [
    {"id": "hc", "name": "Hard coral", "parent": None},
    {"id": "acr", "name": "Acropora", "parent": "hc"},
]


def _fake_ba():
    lib = BenthicAttributeLibrary.__new__(BenthicAttributeLibrary)
    lib.raw_results = list(_BA)
    lib.by_id = {r["id"]: r for r in _BA}
    lib.by_name = {r["name"]: r for r in _BA}
    lib.by_parent = {}
    for r in _BA:
        lib.by_parent.setdefault(r["parent"], []).append(r)
    return lib


def _fake_gf():
    lib = GrowthFormLibrary.__new__(GrowthFormLibrary)
    lib.by_id = {"br": "Branching"}
    return lib


class CliTest(unittest.TestCase):
    def test_label_path_builds_full_path_list(self):
        path = cli.make_label_path(_fake_ba(), _fake_gf())
        self.assertEqual(path("acr::br"), ["Hard coral", "Acropora", "Branching"])

    def test_label_path_ba_only(self):
        path = cli.make_label_path(_fake_ba(), _fake_gf())
        self.assertEqual(path("hc::"), ["Hard coral"])

    def test_path_to_bagf_roundtrips_with_growth_form(self):
        to_bagf = cli.make_path_to_bagf(_fake_ba(), _fake_gf())
        self.assertEqual(to_bagf(["Hard coral", "Acropora", "Branching"]), "acr::br")

    def test_path_to_bagf_ba_only(self):
        to_bagf = cli.make_path_to_bagf(_fake_ba(), _fake_gf())
        self.assertEqual(to_bagf(["Hard coral"]), "hc::")

    def test_label_path_and_inverse_roundtrip(self):
        path = cli.make_label_path(_fake_ba(), _fake_gf())
        to_bagf = cli.make_path_to_bagf(_fake_ba(), _fake_gf())
        for bagf in ("acr::br", "hc::", "acr::"):
            self.assertEqual(to_bagf(path(bagf)), bagf)

    def test_v1_label_mapper_rolls_and_filters_to_classes(self):
        raw = {"1": "ba1::", "2": "baX::gf", "3": None}  # coralnet "3" is unmappable
        rollup = {"ba1::": "ROLLED::", "baX::gf": "OUT::"}  # baX::gf rolls outside V1
        classes = {"ROLLED::"}
        m = cli.make_v1_label_mapper(lambda c: raw.get(c), lambda b: rollup.get(b, b), classes)
        self.assertEqual(m("1"), "ROLLED::")  # mapped, rolled, in V1 classes
        self.assertIsNone(m("2"))  # rolls to a non-class -> dropped (like training)
        self.assertIsNone(m("3"))  # unmappable coralnet id -> dropped

    def test_build_tasks_defaults_come_from_image_set(self):
        from mermaid_classifier.model_review.image_set import IMAGE_SET

        parser = cli.build_parser()
        args = parser.parse_args(["build-tasks"])  # no flags -> all defaults
        self.assertEqual(args.n_images, IMAGE_SET.n_images)
        self.assertEqual(args.seed, IMAGE_SET.seed)
        self.assertEqual(args.min_points, IMAGE_SET.min_points)
        self.assertEqual(args.classifier, IMAGE_SET.classifier)  # S3, not required
        self.assertEqual(args.beta_classifier, IMAGE_SET.beta_classifier)
        self.assertEqual(args.name, IMAGE_SET.name)

    def test_image_set_from_args_roundtrips(self):
        from mermaid_classifier.model_review.image_set import IMAGE_SET

        parser = cli.build_parser()
        args = parser.parse_args(["build-tasks", "--n-images", "7", "--name", "custom"])
        image_set = cli.image_set_from_args(args)
        self.assertEqual(image_set.n_images, 7)
        self.assertEqual(image_set.name, "custom")
        self.assertEqual(image_set.classifier, IMAGE_SET.classifier)  # untouched default
        self.assertEqual(image_set.beta_classifier, IMAGE_SET.beta_classifier)
        self.assertEqual(image_set.min_points, IMAGE_SET.min_points)
        self.assertEqual(image_set.sites, IMAGE_SET.sites)  # per-site layout is not a flag

    def _image_set(self, name="model-review"):
        from mermaid_classifier.model_review.image_set import ImageSet, SiteSpec

        return ImageSet(
            name=name,
            classifier="s3://x/v2/",
            beta_classifier="beta/classifier.pkl",
            heldout_csv="h.csv",
            v1_rollup_csv="r.csv",
            n_images=3,
            min_points=1,
            seed=1,
            sites=(
                SiteSpec(
                    site="coralnet",
                    weight=2.0,
                    manifest_uri="s3://m",
                    feature_bucket="fb",
                    image_bucket="ib",
                    image_prefix="pfx/",
                    image_bucket_region="us-east-1",
                    image_key_template="pfx/s{source_id}/{image_id}.jpg",
                ),
                SiteSpec(
                    site="mermaid",
                    weight=1.0,
                    manifest_uri="s3://mm",
                    feature_bucket="mb",
                    image_bucket="mib",
                    image_prefix="mermaid/",
                    image_bucket_region="us-east-1",
                    image_key_template="mermaid/{image_id}.{ext}",
                    image_key_extensions=("png", "jpg"),
                ),
            ),
        )

    def test_orchestrate_creates_new_project_in_order(self):
        from mermaid_classifier.model_review.ls_tasks import BLANK_MODEL_VERSION

        calls = []

        class FakeClient:
            def project_titles(self):
                calls.append(("titles",))
                return set()

            def create_project(self, title, config):
                calls.append(("create", title, config))
                return 7

            def add_s3_presign_storage(self, pid, bucket, prefix, region, title="images"):
                calls.append(("storage", pid, bucket, prefix, region, title))

            def import_tasks(self, pid, tasks):
                calls.append(("import", pid, len(tasks)))

            def set_model_version(self, pid, mv):
                calls.append(("modelversion", pid, mv))

        pid = cli.orchestrate_create_project(
            FakeClient(), self._image_set(), [{"data": {}}], "<View/>"
        )
        self.assertEqual(pid, 7)
        self.assertEqual(
            calls,
            [
                ("titles",),
                ("create", "model-review", "<View/>"),
                ("storage", 7, "ib", "pfx/", "us-east-1", "coralnet-images"),
                ("storage", 7, "mib", "mermaid/", "us-east-1", "mermaid-images"),
                ("import", 7, 1),
                ("modelversion", 7, BLANK_MODEL_VERSION),
            ],
        )

    def test_orchestrate_refuses_duplicate_title_without_creating(self):
        calls = []

        class FakeClient:
            def project_titles(self):
                return {"model-review"}

            def create_project(self, *a):
                calls.append("create")
                return 1

            def add_s3_presign_storage(self, *a):
                calls.append("storage")

            def import_tasks(self, *a):
                calls.append("import")

            def set_model_version(self, *a):
                calls.append("modelversion")

        with self.assertRaises(SystemExit):
            cli.orchestrate_create_project(FakeClient(), self._image_set(), [{"data": {}}], "<x/>")
        self.assertEqual(calls, [])  # nothing was created or imported

    def test_create_project_subparser_reads_token_and_url(self):
        parser = cli.build_parser()
        args = parser.parse_args(["create-project", "--token", "TT", "--ls-url", "http://h:9"])
        self.assertEqual(args.token, "TT")
        self.assertEqual(args.ls_url, "http://h:9")
        self.assertEqual(args.func, cli.create_project_command)


class _FakeS3:
    """Records HEAD/GET calls; `present` is the set of keys that exist."""

    def __init__(self, present, image_bytes=b""):
        self.present = set(present)
        self.image_bytes = image_bytes
        self.heads = []
        self.gets = []

    def head_object(self, Bucket, Key):  # noqa: N803 — boto3's parameter names
        from botocore.exceptions import ClientError

        self.heads.append((Bucket, Key))
        if Key not in self.present:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3's parameter names
        self.gets.append((Bucket, Key))
        return {"Body": io.BytesIO(self.image_bytes)}


def _png(width, height):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height)).save(buf, format="PNG")
    return buf.getvalue()


class ImageResolverTest(unittest.TestCase):
    def setUp(self):
        from mermaid_classifier.model_review.image_set import SiteSpec

        self.specs = {
            "coralnet": SiteSpec(
                site="coralnet",
                weight=2.0,
                manifest_uri="s3://m",
                feature_bucket="fb",
                image_bucket="cn-bucket",
                image_prefix="imgs/",
                image_bucket_region="us-east-1",
                image_key_template="imgs/s{source_id}/images/{image_id}.jpg",
            ),
            "mermaid": SiteSpec(
                site="mermaid",
                weight=1.0,
                manifest_uri="s3://mm",
                feature_bucket="mb",
                image_bucket="mm-bucket",
                image_prefix="mermaid/",
                image_bucket_region="us-east-1",
                image_key_template="mermaid/{image_id}.{ext}",
                image_key_extensions=("png", "jpg", "jpeg"),
            ),
        }

    def _point(self, site, image_id, source_id=""):
        return ReviewPoint(site, image_id, source_id, 1, 2, "ba::", "k", "b")

    def test_coralnet_key_is_exact_and_never_probed(self):
        s3 = _FakeS3(present=[])
        resolver = cli.ImageResolver(s3, self.specs)
        uri = resolver.uri(self._point("coralnet", "4242", source_id="109"))
        self.assertEqual(uri, "s3://cn-bucket/imgs/s109/images/4242.jpg")
        self.assertEqual(s3.heads, [])  # no extension probe for an exact template

    def test_mermaid_probes_extensions_in_order_and_stops_at_the_first_hit(self):
        s3 = _FakeS3(present=["mermaid/uuid-1.jpg"])
        resolver = cli.ImageResolver(s3, self.specs)
        self.assertEqual(
            resolver.uri(self._point("mermaid", "uuid-1")), "s3://mm-bucket/mermaid/uuid-1.jpg"
        )
        self.assertEqual(
            [key for _, key in s3.heads], ["mermaid/uuid-1.png", "mermaid/uuid-1.jpg"]
        )  # jpeg never tried

    def test_each_image_is_probed_once_and_the_result_reused(self):
        s3 = _FakeS3(present=["mermaid/uuid-1.png"], image_bytes=_png(64, 32))
        resolver = cli.ImageResolver(s3, self.specs)
        point = self._point("mermaid", "uuid-1")
        self.assertTrue(resolver.can_display(point))
        self.assertEqual(resolver.uri(point), "s3://mm-bucket/mermaid/uuid-1.png")
        self.assertEqual(resolver.size(point), (64, 32))
        self.assertEqual(len(s3.heads), 1)  # can_display/uri/size share one probe
        self.assertEqual(s3.gets, [("mm-bucket", "mermaid/uuid-1.png")])

    def test_an_image_with_no_display_object_is_not_displayable(self):
        s3 = _FakeS3(present=[])
        resolver = cli.ImageResolver(s3, self.specs)
        point = self._point("mermaid", "gone")
        self.assertFalse(resolver.can_display(point))
        with self.assertRaises(FileNotFoundError) as ctx:
            resolver.uri(point)
        self.assertIn("gone", str(ctx.exception))


class SiteMixTest(unittest.TestCase):
    def test_counts_images_and_points_per_site(self):
        tasks = [
            {"data": {"source": "mermaid", "original_points": [{}] * 25}},
            {"data": {"source": "coralnet", "original_points": [{}] * 20}},
            {"data": {"source": "coralnet", "original_points": [{}] * 18}},
        ]
        self.assertEqual(cli.site_mix(tasks), {"coralnet": (2, 38), "mermaid": (1, 25)})


class ProjectTitleCheckTest(unittest.TestCase):
    """The title is validated before the build, which downloads features and scores
    every model — a title problem found afterwards wastes all of it."""

    class _Client:
        def __init__(self, titles=()):
            self.titles = set(titles)

        def project_titles(self):
            return self.titles

    def test_title_over_the_label_studio_limit_is_rejected(self):
        from mermaid_classifier.model_review.ls_client import PROJECT_TITLE_MAX_LENGTH

        too_long = "M" * (PROJECT_TITLE_MAX_LENGTH + 1)
        with self.assertRaises(SystemExit) as ctx:
            cli.check_project_title(self._Client(), too_long)
        self.assertIn(str(PROJECT_TITLE_MAX_LENGTH), str(ctx.exception))

    def test_existing_title_is_rejected(self):
        with self.assertRaises(SystemExit):
            cli.check_project_title(self._Client({"taken"}), "taken")

    def test_shipped_image_set_name_fits_the_limit(self):
        from mermaid_classifier.model_review.image_set import IMAGE_SET
        from mermaid_classifier.model_review.ls_client import PROJECT_TITLE_MAX_LENGTH

        self.assertLessEqual(len(IMAGE_SET.name), PROJECT_TITLE_MAX_LENGTH)

    def test_acceptable_title_passes(self):
        cli.check_project_title(self._Client({"other"}), "fine")
