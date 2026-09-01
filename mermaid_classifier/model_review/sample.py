"""Select a stratified review sample of held-out images.

The review pool spans two sites — CoralNet and MERMAID — whose ground-truth grids,
feature vectors and display images live in different places. Each site is a stratum
drawn independently, and each stratum is an exact simple random sample without
replacement of its own *qualified* population (see `select_stratum`).

Selection modes:
- `select_stratum` / `allocate_quota`: the stratified full-grid draw the CLI uses.
- `select_images`: the val-split points only (each image shows its held-out points).
"""

import csv
import hashlib
import math
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mermaid_classifier.common.benthic_attributes import combine_ba_gf

# Site names: the values of the val CSV's `site` column, and of pyspacer's `Sites` enum.
# Duplicated as plain strings so this module stays free of the training-lane imports
# (`pyspacer.options` pulls in pandas and the settings layer).
CORALNET = "coralnet"
MERMAID = "mermaid"

_SOURCE_RE = re.compile(r"^s(\d+)/")


@dataclass(frozen=True)
class ReviewPoint:
    site: str  # CORALNET | MERMAID — selects the image/feature layout for this point
    image_id: str
    source_id: str  # CoralNet source id; "" for MERMAID, which has no source
    row: int
    col: int
    gt_bagf: str
    feature_key: str
    bucket: str  # bucket holding `feature_key`


def source_id_from_feature_key(feature_key: str) -> str:
    m = _SOURCE_RE.match(feature_key)
    if not m:
        raise ValueError(f"cannot parse source id from {feature_key!r}")
    return m.group(1)


def select_images(
    csv_path: str,
    n_images: int,
    seed: int,
    site: str = CORALNET,
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
                site=site,
                image_id=image_id,
                source_id=(
                    source_id_from_feature_key(r["feature_vector"]) if site == CORALNET else ""
                ),
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


# --- Eligibility -------------------------------------------------------------


def eligible_image_ids_by_site(csv_path: str) -> dict[str, set[str]]:
    """Image ids that had >=1 annotation in the val split, grouped by site.

    Test-set eligibility. One pass over the CSV covers every site, which matters:
    the materialized val split is ~250k rows.
    """
    eligible: dict[str, set[str]] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            eligible.setdefault(r["site"], set()).add(str(r["image_id"]))
    return eligible


# --- Quota allocation and per-stratum seeding --------------------------------


def allocate_quota(n_images: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Largest-remainder (Hare) allocation of `n_images` across weighted sites.

    The result always sums to exactly `n_images`. Ties in the fractional remainder
    break by site name ascending, so the allocation is deterministic.

    Weights are relative, not fractions: ``{"coralnet": 2.0, "mermaid": 1.0}`` at
    ``n_images=20`` gives ``{"coralnet": 13, "mermaid": 7}``.
    """
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"site weights must sum to a positive number, got {dict(weights)}")
    exact = {site: n_images * w / total for site, w in weights.items()}
    quota = {site: math.floor(e) for site, e in exact.items()}
    seats_left = n_images - sum(quota.values())
    by_remainder = sorted(exact, key=lambda s: (-(exact[s] - quota[s]), s))
    for site in by_remainder[:seats_left]:
        quota[site] += 1
    return quota


def stratum_seed(seed: int, site: str) -> int:
    """RNG seed for one stratum: a stable function of ``(seed, site)`` and nothing else.

    Independence from quota / n_images / min_points is what keeps one site's draw from
    moving when another site's share changes. blake2b rather than ``hash()`` because
    ``hash()`` is PYTHONHASHSEED-salted and so differs between processes.
    """
    digest = hashlib.blake2b(f"model-review/{seed}/{site}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


# --- Grouping manifest rows into per-image points ----------------------------


def _normalize_growth_form(value: object) -> str:
    """An absent growth form is ``''`` — never NULL, and never the string ``'None'``.

    The MERMAID parquet stores it as SQL NULL, which reaches Python as ``None`` or, via
    pandas, as ``NaN``; `pyspacer.dataset.read_mermaid_data` also sees the literal
    ``'None'``. All three collapse to ``''``, matching the CoralNet mapping's convention.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    return "" if text in ("", "None", "nan") else text


def _group_points(
    rows: Iterable[Mapping[str, Any]],
    make_point: Callable[[Mapping[str, Any]], ReviewPoint | None],
) -> dict[str, list[ReviewPoint]]:
    """Group manifest rows per image, deduping ``(row, col)``.

    ``make_point`` returns None for a label with no usable BA::GF; such a row is dropped
    *without* marking its ``(row, col)`` as seen, so a later duplicate of the same point
    carrying a mappable label still counts.
    """
    by_image: dict[str, list[ReviewPoint]] = {}
    seen: dict[str, set[tuple[int, int]]] = {}
    for r in rows:
        image_id = str(r["image_id"])
        rc = (int(r["row"]), int(r["col"]))
        if rc in seen.setdefault(image_id, set()):
            continue
        point = make_point(r)
        if point is None:
            continue
        seen[image_id].add(rc)
        by_image.setdefault(image_id, []).append(point)
    return by_image


def coralnet_points_from_manifest_rows(
    rows: Iterable[Mapping[str, Any]],
    map_label: Callable[[str], str | None],
    feature_bucket: str,
) -> dict[str, list[ReviewPoint]]:
    """CoralNet manifest rows -> per-image ReviewPoints.

    ``map_label(coralnet_id) -> "ba::gf" or None`` (None = unusable label, point dropped).
    """

    def make_point(r: Mapping[str, Any]) -> ReviewPoint | None:
        bagf = map_label(str(r["coralnet_id"]))
        if bagf is None:
            return None
        source_id = str(r["source_id"])
        image_id = str(r["image_id"])
        return ReviewPoint(
            site=CORALNET,
            image_id=image_id,
            source_id=source_id,
            row=int(r["row"]),
            col=int(r["col"]),
            gt_bagf=bagf,
            feature_key=f"s{source_id}/features/i{image_id}.featurevector",
            bucket=feature_bucket,
        )

    return _group_points(rows, make_point)


def mermaid_points_from_manifest_rows(
    rows: Iterable[Mapping[str, Any]],
    map_label: Callable[[str], str | None],
    feature_bucket: str,
) -> dict[str, list[ReviewPoint]]:
    """MERMAID annotation rows -> per-image ReviewPoints.

    MERMAID ground truth is already BA::GF, so ``map_label`` receives the combined
    ``"ba::gf"`` rather than a provider label id.
    """

    def make_point(r: Mapping[str, Any]) -> ReviewPoint | None:
        raw = combine_ba_gf(
            str(r["benthic_attribute_id"]), _normalize_growth_form(r["growth_form_id"])
        )
        bagf = map_label(raw)
        if bagf is None:
            return None
        image_id = str(r["image_id"])
        return ReviewPoint(
            site=MERMAID,
            image_id=image_id,
            source_id="",
            row=int(r["row"]),
            col=int(r["col"]),
            gt_bagf=bagf,
            feature_key=f"mermaid/{image_id}_featurevector",
            bucket=feature_bucket,
        )

    return _group_points(rows, make_point)


# --- The stratum draw --------------------------------------------------------


def select_stratum(
    candidate_ids: Sequence[str],
    fetch_points: Callable[[Sequence[str]], dict[str, list[ReviewPoint]]],
    quota: int,
    seed: int,
    min_points_per_image: int,
    site: str = "",
    keep_image: Callable[[ReviewPoint], bool] | None = None,
    batch_size: int = 0,
) -> list[ReviewPoint]:
    """Draw `quota` images from `candidate_ids`, returning all points of each.

    An exact simple random sample without replacement of the *qualified* population Q
    (candidates with >= `min_points_per_image` mapped points that also pass `keep_image`):
    the candidates are shuffled into one uniform random permutation, then walked in order
    keeping qualifiers until the quota is met. Restricting a uniform permutation of the
    candidates to its members of Q yields a uniform permutation of Q, so the first `quota`
    qualifiers are a uniform SRS of Q.

    Selection is therefore *nested* in `quota`: raising it extends the previous draw
    rather than redrawing it.

    Points are fetched in batches purely to bound S3 round-trips. Batching cannot change
    the result — the keep-loop walks each batch in permutation order, never in the order
    `fetch_points` happens to return. `batch_size` 0 grows the window geometrically;
    passing an explicit size holds it fixed (used by the tests that pin this invariance).

    Raises ValueError if the candidates are exhausted before the quota is met, rather
    than silently returning a short sample.
    """
    if quota <= 0:
        return []
    # sorted() first: `candidate_ids` typically arrives in DuckDB GROUP BY order, which is
    # not stable across versions or thread counts, and shuffle() is a function of the
    # input order as well as the seed.
    order = sorted(candidate_ids)
    random.Random(seed).shuffle(order)

    window = batch_size or max(4 * quota, 32)
    chosen: list[str] = []
    points_by_image: dict[str, list[ReviewPoint]] = {}
    cursor = 0
    while cursor < len(order) and len(chosen) < quota:
        batch = order[cursor : cursor + window]
        cursor += len(batch)
        fetched = fetch_points(batch)
        for image_id in batch:
            points = fetched.get(image_id, [])
            if len(points) < min_points_per_image:
                continue
            if keep_image is not None and not keep_image(points[0]):
                continue
            chosen.append(image_id)
            points_by_image[image_id] = points
            if len(chosen) == quota:
                break
        if not batch_size:
            window = min(window * 2, 4096)

    if len(chosen) < quota:
        raise ValueError(
            f"{site or 'stratum'}: only {len(chosen)} of {quota} images qualified "
            f"(>= {min_points_per_image} mapped points) out of {len(order)} candidates. "
            f"Lower min_points for this site, lower its weight, or widen eligibility."
        )

    selected: list[ReviewPoint] = []
    for image_id in sorted(chosen):
        selected.extend(sorted(points_by_image[image_id], key=lambda p: (p.row, p.col)))
    return selected


def default_coralnet_mapper() -> Callable[[str], str | None]:
    from mermaid_classifier.common.benthic_attributes import CoralNetMermaidMapping

    mapping = CoralNetMermaidMapping()

    def map_coralnet(coralnet_id: str) -> str | None:
        if coralnet_id not in mapping:
            return None
        entry = mapping[coralnet_id]
        return combine_ba_gf(entry.benthic_attribute_id, entry.growth_form_id)

    return map_coralnet


# --- Reading the ground-truth grids (the only S3 I/O in this module) ---------


@dataclass(frozen=True)
class SiteSource:
    """One sampling stratum: where its full ground-truth grid lives and how to read it."""

    site: str
    weight: float  # relative share of n_images (largest-remainder allocation)
    manifest_uri: str  # parquet holding every ground-truth point of every image
    columns: tuple[str, ...]  # manifest columns to project
    to_points: Callable[[list[dict[str, Any]]], dict[str, list[ReviewPoint]]]
    keep_image: Callable[[ReviewPoint], bool] | None = None


def coralnet_source(
    manifest_uri: str,
    feature_bucket: str,
    weight: float,
    map_label: Callable[[str], str | None],
    keep_image: Callable[[ReviewPoint], bool] | None = None,
) -> SiteSource:
    return SiteSource(
        site=CORALNET,
        weight=weight,
        manifest_uri=manifest_uri,
        columns=("source_id", "image_id", "row", "col", "coralnet_id"),
        to_points=lambda rows: coralnet_points_from_manifest_rows(rows, map_label, feature_bucket),
        keep_image=keep_image,
    )


def mermaid_source(
    manifest_uri: str,
    feature_bucket: str,
    weight: float,
    map_label: Callable[[str], str | None],
    keep_image: Callable[[ReviewPoint], bool] | None = None,
) -> SiteSource:
    return SiteSource(
        site=MERMAID,
        weight=weight,
        manifest_uri=manifest_uri,
        columns=("image_id", "row", "col", "benthic_attribute_id", "growth_form_id"),
        to_points=lambda rows: mermaid_points_from_manifest_rows(rows, map_label, feature_bucket),
        keep_image=keep_image,
    )


def _projection(columns: Sequence[str]) -> str:
    # "row" and "col" collide with DuckDB builtins, so every column is quoted.
    return ", ".join(f'm."{c}"' for c in columns)


def _candidate_image_ids(
    con: Any, manifest_uri: str, eligible: set[str], min_points: int
) -> list[str]:
    """Eligible images carrying at least `min_points` raw manifest rows.

    A cheap upper-bound pre-filter on raw row count; the binding test is the mapped-point
    count applied per image in `select_stratum`.
    """
    import pandas as pd

    con.register("elig", pd.DataFrame({"image_id": sorted(eligible)}))
    df = con.execute(
        f"SELECT m.image_id AS image_id FROM read_parquet('{manifest_uri}') m "
        "JOIN elig ON CAST(m.image_id AS VARCHAR) = elig.image_id "
        "GROUP BY m.image_id HAVING count(*) >= ?",
        [min_points],
    ).fetch_df()
    return [str(i) for i in df["image_id"].tolist()]


def _grid_fetcher(
    con: Any, source: SiteSource
) -> Callable[[Sequence[str]], dict[str, list[ReviewPoint]]]:
    import pandas as pd

    def fetch(image_ids: Sequence[str]) -> dict[str, list[ReviewPoint]]:
        con.register("batch", pd.DataFrame({"image_id": list(image_ids)}))
        rows = (
            con.execute(
                f"SELECT {_projection(source.columns)} "
                f"FROM read_parquet('{source.manifest_uri}') m "
                "JOIN batch ON CAST(m.image_id AS VARCHAR) = batch.image_id"
            )
            .fetch_df()
            .to_dict("records")
        )
        return source.to_points(rows)

    return fetch


def select_images_stratified(
    val_csv: str,
    sources: Sequence[SiteSource],
    n_images: int,
    seed: int,
    min_points_per_image: int,
) -> list[ReviewPoint]:
    """Stratified full-grid sample: `n_images` split across `sources` by their weights.

    Each stratum is drawn independently by `select_stratum` from its own seed, so one
    site's share can change without moving another site's draw.
    """
    import duckdb

    quota = allocate_quota(n_images, {s.site: s.weight for s in sources})
    eligible = eligible_image_ids_by_site(val_csv)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("CREATE SECRET (TYPE s3, PROVIDER credential_chain);")
    try:
        points: list[ReviewPoint] = []
        for source in sources:
            if quota[source.site] == 0:
                continue
            ids = eligible.get(source.site, set())
            if not ids:
                raise ValueError(
                    f"{source.site}: no eligible images in {val_csv} "
                    f"(its `site` column has {sorted(eligible)})"
                )
            candidates = _candidate_image_ids(con, source.manifest_uri, ids, min_points_per_image)
            points.extend(
                select_stratum(
                    candidates,
                    _grid_fetcher(con, source),
                    quota=quota[source.site],
                    seed=stratum_seed(seed, source.site),
                    min_points_per_image=min_points_per_image,
                    site=source.site,
                    keep_image=source.keep_image,
                )
            )
    finally:
        con.close()
    return points
