"""Select a review sample of CoralNet images.

Two selection modes:
- `select_images`: the val-split points only (each image shows its held-out points).
- `select_images_full_gt`: every ground-truth point of each image (from the CoralNet
  manifest, mapped to BA::GF), for images that had >=1 annotation in the val split
  and have >=N total ground-truth points. This is the "full grid" mode.
"""

import csv
import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

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
    csv_path: str,
    n_images: int,
    seed: int,
    site: str = "coralnet",
    min_points_per_image: int = 1,
) -> list[ReviewPoint]:
    by_image: dict[str, list[ReviewPoint]] = {}
    seen_rc: dict[str, set[tuple[int, int]]] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["site"] != site:
                continue
            image_id = r["image_id"]
            row, col = int(r["row"]), int(r["col"])
            # The val CSV contains duplicate rows per point (an export fan-out);
            # keep one ReviewPoint per distinct (row, col) so points aren't stacked.
            if (row, col) in seen_rc.setdefault(image_id, set()):
                continue
            seen_rc[image_id].add((row, col))
            point = ReviewPoint(
                image_id=image_id,
                source_id=source_id_from_feature_key(r["feature_vector"]),
                row=row,
                col=col,
                gt_bagf=combine_ba_gf(r["benthic_attribute_id"], r["growth_form_id"]),
                feature_key=r["feature_vector"],
                bucket=r["bucket"],
            )
            by_image.setdefault(image_id, []).append(point)

    # Only consider images with enough points to be worth reviewing (the val
    # split leaves most images with 1-2 points; a useful review needs a grid).
    image_ids = sorted(img for img, pts in by_image.items() if len(pts) >= min_points_per_image)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(image_ids, min(n_images, len(image_ids))))

    points: list[ReviewPoint] = []
    for image_id in chosen:
        points.extend(sorted(by_image[image_id], key=lambda p: (p.row, p.col)))
    return points


# --- Full ground-truth mode (all points per image, from the CoralNet manifest) ---


def eligible_image_ids_from_val(csv_path: str, site: str = "coralnet") -> set[str]:
    """Image ids that had >=1 annotation in the val split (test-set eligibility)."""
    eligible: set[str] = set()
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["site"] == site:
                eligible.add(str(r["image_id"]))
    return eligible


def review_points_from_manifest_rows(
    rows: Iterable[dict[str, Any]],
    map_coralnet: Callable[[str], str | None],
    coralnet_bucket: str,
) -> dict[str, list[ReviewPoint]]:
    """Group manifest rows into per-image ReviewPoints, mapping coralnet_id -> BA::GF.

    `map_coralnet(coralnet_id) -> "ba::gf" or None` (None = unmappable label, dropped).
    Duplicate (row, col) per image are deduped.
    """
    by_image: dict[str, list[ReviewPoint]] = {}
    seen: dict[str, set[tuple[int, int]]] = {}
    for r in rows:
        image_id = str(r["image_id"])
        row, col = int(r["row"]), int(r["col"])
        if (row, col) in seen.setdefault(image_id, set()):
            continue
        bagf = map_coralnet(str(r["coralnet_id"]))
        if bagf is None:  # label has no BA::GF mapping -> drop this point
            continue
        seen[image_id].add((row, col))
        source_id = str(r["source_id"])
        by_image.setdefault(image_id, []).append(
            ReviewPoint(
                image_id=image_id,
                source_id=source_id,
                row=row,
                col=col,
                gt_bagf=bagf,
                feature_key=f"s{source_id}/features/i{image_id}.featurevector",
                bucket=coralnet_bucket,
            )
        )
    return by_image


def sample_full_gt_images(
    by_image: dict[str, list[ReviewPoint]],
    n_images: int,
    seed: int,
    min_points_per_image: int,
) -> list[ReviewPoint]:
    """Random sample of images with >= min mapped points; returns all their points."""
    qualified = sorted(img for img, pts in by_image.items() if len(pts) >= min_points_per_image)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(qualified, min(n_images, len(qualified))))
    points: list[ReviewPoint] = []
    for image_id in chosen:
        points.extend(sorted(by_image[image_id], key=lambda p: (p.row, p.col)))
    return points


def _default_coralnet_mapper() -> Callable[[str], str | None]:
    from mermaid_classifier.common.benthic_attributes import CoralNetMermaidMapping

    mapping = CoralNetMermaidMapping()

    def map_coralnet(coralnet_id: str) -> str | None:
        if coralnet_id not in mapping:
            return None
        entry = mapping[coralnet_id]
        return combine_ba_gf(entry.benthic_attribute_id, entry.growth_form_id)

    return map_coralnet


def select_images_full_gt(
    val_csv: str,
    manifest_uri: str,
    coralnet_bucket: str,
    n_images: int,
    seed: int,
    min_points_per_image: int = 15,
    map_coralnet: Callable[[str], str | None] | None = None,
    pool_factor: int = 5,
) -> list[ReviewPoint]:
    """Random sample of test-set images, each with ALL its ground-truth points.

    Eligibility: image had >=1 annotation in `val_csv`. Full points come from the
    CoralNet manifest parquet (`manifest_uri`), mapped to BA::GF; images with
    >=`min_points_per_image` mapped points are eligible for sampling.
    """
    import duckdb
    import pandas as pd

    if map_coralnet is None:
        map_coralnet = _default_coralnet_mapper()

    eligible = eligible_image_ids_from_val(val_csv)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("CREATE SECRET (TYPE s3, PROVIDER credential_chain);")
    con.register("elig", pd.DataFrame({"image_id": sorted(eligible)}))
    # candidate images: eligible + >= min TOTAL manifest points (cheap pre-filter)
    candidates = con.execute(
        "SELECT DISTINCT m.image_id AS image_id "
        f"FROM read_parquet('{manifest_uri}') m "
        "JOIN elig ON CAST(m.image_id AS VARCHAR) = elig.image_id "
        "GROUP BY m.image_id HAVING count(*) >= ?",
        [min_points_per_image],
    ).fetch_df()
    cand_ids = [str(i) for i in candidates["image_id"].tolist()]
    rng = random.Random(seed)
    rng.shuffle(cand_ids)
    # over-sample a pool so images that drop below min after mapping still leave >= n
    pool = cand_ids[: max(n_images * pool_factor, n_images)]
    con.register("pool", pd.DataFrame({"image_id": pool}))
    rows = (
        con.execute(
            "SELECT m.source_id, m.image_id, m.row, m.col, m.coralnet_id "
            f"FROM read_parquet('{manifest_uri}') m "
            "JOIN pool ON CAST(m.image_id AS VARCHAR) = pool.image_id"
        )
        .fetch_df()
        .to_dict("records")
    )
    con.close()

    by_image = review_points_from_manifest_rows(rows, map_coralnet, coralnet_bucket)
    return sample_full_gt_images(by_image, n_images, seed, min_points_per_image)
