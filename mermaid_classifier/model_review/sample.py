"""Select a small review sample from the held-out CoralNet val set."""

import csv
import random
import re
from dataclasses import dataclass

from mermaid_classifier.common.benthic_attributes import combine_ba_gf

_SOURCE_RE = re.compile(r"^s(\d+)/")


@dataclass(frozen=True)
class ReviewPoint:
    image_id: str
    source_id: str
    row: int
    col: int
    gt_bagf: str
    feature_key: str
    bucket: str


def source_id_from_feature_key(feature_key: str) -> str:
    m = _SOURCE_RE.match(feature_key)
    if not m:
        raise ValueError(f"cannot parse source id from {feature_key!r}")
    return m.group(1)


def select_images(
    csv_path: str, n_images: int, seed: int, site: str = "coralnet"
) -> list[ReviewPoint]:
    by_image: dict[str, list[ReviewPoint]] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["site"] != site:
                continue
            image_id = r["image_id"]
            point = ReviewPoint(
                image_id=image_id,
                source_id=source_id_from_feature_key(r["feature_vector"]),
                row=int(r["row"]),
                col=int(r["col"]),
                gt_bagf=combine_ba_gf(r["benthic_attribute_id"], r["growth_form_id"]),
                feature_key=r["feature_vector"],
                bucket=r["bucket"],
            )
            by_image.setdefault(image_id, []).append(point)

    image_ids = sorted(by_image)  # stable universe
    rng = random.Random(seed)
    chosen = sorted(rng.sample(image_ids, min(n_images, len(image_ids))))

    points: list[ReviewPoint] = []
    for image_id in chosen:
        points.extend(sorted(by_image[image_id], key=lambda p: (p.row, p.col)))
    return points
