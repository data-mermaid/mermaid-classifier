"""The frozen region probe set: a fixed slice of MERMAID annotation points
that any model can be scored on, plus the manifest that pins what it is.

Most committed training configs set `include_mermaid: false`, so a rate
computed on a run's own validation split would have nothing to compute on. A
probe set fixed once and scored afterwards works on any run, scores the
already-published model, and makes v1-vs-v2 a paired comparison on identical
points.

Composition is a measured decision, not a sample of convenience.

*The Atlantic in full.* Every Tropical Atlantic image enters, including those
whose ground truth is not itself region-discriminating: the Atlantic
direction's denominator is predictions on Atlantic images, so the truth label
need not discriminate for the point to count.

*The Indo-Pacific proportionally, with a floor.* The two Indo-Pacific regions
are sampled proportional to their eligible pools, and a per-region floor stops
the smaller of the two being swamped by the larger.

*Whole images.* Every point on a selected image is taken, so per-image
aggregates and the image-clustered intervals in `metrics` stay valid.

*No stratification on label content.* Stratifying on the thing being measured
invites a selection artifact, so selection sees only region and eligibility.

`held_out` is per image and the Atlantic images are deliberately not held out.
CoralNet supplies 99.5% of the training signal for the Atlantic-only classes,
so keeping those MERMAID images in training costs almost nothing -- and they
are the only MERMAID-domain Atlantic data in existence, the very
counterexamples that stop a model learning "looks like a MERMAID photo implies
Indo-Pacific". Withholding them would reinforce the failure being measured.

The benthic-attribute region map travels with the probe as a hashed snapshot.
MERMAID curates those region lists; without the freeze, adding a region to one
coral next month would move a model's score for reasons unrelated to the
model.

Selection is pure: nothing here reaches S3 or the network, and
`read_annotations` takes a path DuckDB can open.
"""

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.common.region_rules import is_region_discriminating
from mermaid_classifier.region_eval.metrics import required_n_for_detection

PROBE_VERSION = "1"

PROBE_COLUMNS = (
    "image_id",
    "point_id",
    "row",
    "col",
    "benthic_attribute_id",
    "benthic_attribute_name",
    "growth_form_id",
    "growth_form_name",
    "gt_label",
    "region_id",
    "region_name",
    "site_id",
    "held_out",
)

# What selection needs from the source export, beyond the optional site_id. A
# source missing any of these is a schema change, not something to warn about.
SOURCE_COLUMNS = (
    "image_id",
    "point_id",
    "row",
    "col",
    "benthic_attribute_id",
    "benthic_attribute_name",
    "growth_form_id",
    "growth_form_name",
    "region_id",
    "region_name",
)

# Clusters are images, not sites: the export carries no site_id yet. The column
# exists so the schema accepts one without a rewrite, and until it arrives every
# reported interval is a lower bound on the true width.
OPTIONAL_SOURCE_COLUMNS = ("site_id",)

DEFAULT_TARGET_IMAGES = 2500
# 400 images is ~10,000 points, which resolves a two-point difference in a
# five-percent rate even at the design effect image clustering implies.
DEFAULT_REGION_FLOOR = 400
DEFAULT_CENSUS_REGION_NAMES = frozenset({"Tropical Atlantic"})

CENSUS = "census"
SAMPLE = "sample"

MDE_BASELINE_RATES = (0.02, 0.05, 0.10)
MDE_EFFECTS = (0.01, 0.02, 0.05)
# 1.0 is the unclustered floor. 5.0 is what ~25 correlated points per image
# imply at an intra-image correlation near 0.2; the realized value arrives only
# with a scored run, so the table brackets rather than picks.
MDE_DESIGN_EFFECTS = (1.0, 5.0)

_OVERALL_REGION_NAME = "all"
_FIELD_SEPARATOR = "\x1f"
_RECORD_SEPARATOR = b"\x1e"


@dataclasses.dataclass(frozen=True)
class ProbeSelectionOptions:
    """Knobs that, with the source data, determine the probe set exactly.

    `target_images` counts the whole probe including the census regions, so
    the sampled budget is what remains after they are taken. `region_floor`
    applies per sampled region and is capped by that region's eligible pool.
    """

    seed: int = 0
    target_images: int = DEFAULT_TARGET_IMAGES
    region_floor: int = DEFAULT_REGION_FLOOR
    census_region_names: frozenset[str] = DEFAULT_CENSUS_REGION_NAMES
    alpha: float = 0.05


@dataclasses.dataclass(frozen=True)
class StratumCounts:
    """What one region contributed, and under which policy."""

    region_id: str
    region_name: str
    policy: str
    held_out: bool
    n_region_images: int
    n_eligible_images: int
    n_images: int
    n_points: int


@dataclasses.dataclass(frozen=True)
class ProbeSet:
    rows: pd.DataFrame
    strata: tuple[StratumCounts, ...]
    options: ProbeSelectionOptions
    n_rows_unrecorded_region_dropped: int
    content_hash: str
    region_snapshot_hash: str


@dataclasses.dataclass(frozen=True, slots=True)
class _Annotation:
    """One source annotation, in the fields selection and the probe schema use."""

    image_id: str
    point_id: str
    row: int
    col: int
    benthic_attribute_id: str
    benthic_attribute_name: str
    growth_form_id: str
    growth_form_name: str
    region_id: str
    region_name: str
    site_id: str


@dataclasses.dataclass(slots=True)
class _Image:
    """One image's region, its eligibility, and every point on it."""

    region_id: str
    region_name: str
    eligible: bool
    points: list[_Annotation]


def read_annotations(
    parquet_path: str | Path,
    *,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """Load the confirmed-annotation parquet into the shape selection expects.

    Empty growth forms, regions, names and site ids arrive as `''` rather than
    NULL, which is the convention the rest of the pipeline joins on. `site_id`
    is synthesized empty when the export does not carry it yet.
    """
    connection = duckdb.connect() if conn is None else conn
    try:
        available = {
            str(entry[0])
            for entry in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(parquet_path)]
            ).fetchall()
        }
        missing = [column for column in SOURCE_COLUMNS if column not in available]
        if missing:
            raise ValueError(f"{parquet_path} is missing column(s): {', '.join(missing)}")

        projections = [
            "image_id",
            "CAST(point_id AS VARCHAR) AS point_id",
            "CAST(row AS BIGINT) AS row",
            "CAST(col AS BIGINT) AS col",
            "benthic_attribute_id",
            "COALESCE(benthic_attribute_name, '') AS benthic_attribute_name",
            "COALESCE(growth_form_id, '') AS growth_form_id",
            "COALESCE(growth_form_name, '') AS growth_form_name",
            "COALESCE(region_id, '') AS region_id",
            "COALESCE(region_name, '') AS region_name",
        ]
        projections += [
            f"COALESCE({column}, '') AS {column}" if column in available else f"'' AS {column}"
            for column in OPTIONAL_SOURCE_COLUMNS
        ]
        return connection.execute(
            f"SELECT {', '.join(projections)} FROM read_parquet(?)", [str(parquet_path)]
        ).df()
    finally:
        if conn is None:
            connection.close()


def build_probe_set(
    annotations: pd.DataFrame,
    *,
    region_ids_by_attribute: Mapping[str, frozenset[str]],
    options: ProbeSelectionOptions | None = None,
) -> ProbeSet:
    """Select the probe points and the counts that describe them.

    `region_ids_by_attribute` is the frozen snapshot: an attribute it omits
    never makes an image eligible, exactly as an attribute with no recorded
    regions does not.
    """
    selection = options or ProbeSelectionOptions()
    if selection.target_images < 0:
        raise ValueError(f"target_images must not be negative, got {selection.target_images}")
    if selection.region_floor < 0:
        raise ValueError(f"region_floor must not be negative, got {selection.region_floor}")

    records = _annotation_records(annotations)
    recorded = [record for record in records if record.region_id]
    images = _image_index(recorded, region_ids_by_attribute)
    census_region_ids = _census_region_ids(images, selection.census_region_names)

    by_region: dict[str, list[str]] = {}
    for image_id, image in images.items():
        by_region.setdefault(image.region_id, []).append(image_id)

    selected: set[str] = set()
    pools: dict[str, list[str]] = {}
    for region_id in sorted(by_region):
        if region_id in census_region_ids:
            selected.update(by_region[region_id])
        else:
            pools[region_id] = _hash_order(
                (image_id for image_id in by_region[region_id] if images[image_id].eligible),
                selection.seed,
            )

    allocation = _allocate(
        {region_id: len(pool) for region_id, pool in pools.items()},
        budget=selection.target_images - len(selected),
        floor=selection.region_floor,
    )
    for region_id, pool in pools.items():
        selected.update(pool[: allocation[region_id]])

    rows = _probe_rows(images, selected, census_region_ids)
    strata = tuple(
        _stratum_counts(region_id, by_region[region_id], images, selected, census_region_ids)
        for region_id in sorted(by_region)
    )

    return ProbeSet(
        rows=rows,
        strata=strata,
        options=selection,
        n_rows_unrecorded_region_dropped=len(records) - len(recorded),
        content_hash=probe_content_hash(rows),
        region_snapshot_hash=region_snapshot_hash(region_ids_by_attribute),
    )


def probe_content_hash(rows: pd.DataFrame) -> str:
    """A hash of the probe rows themselves, in a canonical order.

    Sorting first makes the hash an identity for the *set* of points, so a
    reordered parquet is the same probe while a changed cell is not.
    """
    ordered = rows.sort_values(["image_id", "point_id"], kind="stable")
    columns = [ordered[column].tolist() for column in PROBE_COLUMNS]
    digest = hashlib.sha256()
    for record in zip(*columns, strict=True):
        digest.update(_FIELD_SEPARATOR.join(_canonical(value) for value in record).encode())
        digest.update(_RECORD_SEPARATOR)
    return digest.hexdigest()


def region_snapshot(
    region_ids_by_attribute: Mapping[str, frozenset[str]],
) -> dict[str, list[str]]:
    """The region map as sorted, JSON-ready lists."""
    return {
        attribute_id: sorted(region_ids)
        for attribute_id, region_ids in region_ids_by_attribute.items()
    }


def region_snapshot_json(region_ids_by_attribute: Mapping[str, frozenset[str]]) -> str:
    """The canonical serialization the snapshot hash is taken over."""
    return json.dumps(
        region_snapshot(region_ids_by_attribute), sort_keys=True, separators=(",", ":")
    )


def region_snapshot_hash(region_ids_by_attribute: Mapping[str, frozenset[str]]) -> str:
    return hashlib.sha256(region_snapshot_json(region_ids_by_attribute).encode()).hexdigest()


def minimum_detectable_effect_table(
    strata: Sequence[StratumCounts],
    *,
    alpha: float = 0.05,
    baseline_rates: Sequence[float] = MDE_BASELINE_RATES,
    effects: Sequence[float] = MDE_EFFECTS,
    design_effects: Sequence[float] = MDE_DESIGN_EFFECTS,
) -> list[dict[str, object]]:
    """Which v1-vs-v2 differences the realized probe can resolve.

    One row per (population, baseline rate, effect, design effect), so a delta
    cannot be read off the probe without the sample size that would be needed
    to believe it. The leading rows cover the probe as a whole.

    The requirement is the two-sample detection one, not a margin of error:
    two rates each measured to a half-width of `effect` still have overlapping
    intervals at a true difference of `effect`, and a v1-vs-v2 delta is read
    precisely to decide whether the difference is real.
    """
    populations: list[tuple[str, str, int]] = [
        ("", _OVERALL_REGION_NAME, sum(stratum.n_points for stratum in strata))
    ]
    populations += [
        (stratum.region_id, stratum.region_name, stratum.n_points) for stratum in strata
    ]

    table: list[dict[str, object]] = []
    for region_id, region_name, n_points in populations:
        for rate in baseline_rates:
            for effect in effects:
                for inflation in design_effects:
                    required = required_n_for_detection(
                        rate, effect, alpha=alpha, design_effect=inflation
                    )
                    table.append(
                        {
                            "region_id": region_id,
                            "region_name": region_name,
                            "n_points": n_points,
                            "baseline_rate": rate,
                            "effect": effect,
                            "design_effect": inflation,
                            "required_n": required,
                            "resolvable": n_points >= required,
                        }
                    )
    return table


def build_manifest(
    probe: ProbeSet,
    *,
    source_uri: str,
    source_etag: str | None,
    source_row_count: int,
    builder_git_sha: str,
    baseline_rates: Sequence[float] = MDE_BASELINE_RATES,
    effects: Sequence[float] = MDE_EFFECTS,
    design_effects: Sequence[float] = MDE_DESIGN_EFFECTS,
) -> dict[str, object]:
    """Everything needed to say whether two scores were taken on the same probe.

    The source ETag and row count pin the export, the content hash pins the
    points, the region-snapshot hash pins the taxonomy, and the
    minimum-detectable-effect table pins what a difference between two scores
    is allowed to mean.
    """
    return {
        "probe_version": PROBE_VERSION,
        "probe_columns": list(PROBE_COLUMNS),
        "source_uri": source_uri,
        "source_etag": source_etag,
        "source_row_count": int(source_row_count),
        "source_rows_unrecorded_region_dropped": int(probe.n_rows_unrecorded_region_dropped),
        "content_hash": probe.content_hash,
        "region_snapshot_hash": probe.region_snapshot_hash,
        "seed": probe.options.seed,
        "target_images": probe.options.target_images,
        "region_floor": probe.options.region_floor,
        "census_region_names": sorted(probe.options.census_region_names),
        "n_images": sum(stratum.n_images for stratum in probe.strata),
        "n_points": sum(stratum.n_points for stratum in probe.strata),
        "strata": [dataclasses.asdict(stratum) for stratum in probe.strata],
        "builder_git_sha": builder_git_sha,
        "minimum_detectable_effect": minimum_detectable_effect_table(
            probe.strata,
            alpha=probe.options.alpha,
            baseline_rates=baseline_rates,
            effects=effects,
            design_effects=design_effects,
        ),
    }


def _annotation_records(annotations: pd.DataFrame) -> list[_Annotation]:
    """The source frame as records, independent of the dtypes it arrived with."""
    missing = [column for column in SOURCE_COLUMNS if column not in annotations.columns]
    if missing:
        raise ValueError(f"annotations are missing column(s): {', '.join(missing)}")

    values = {column: annotations[column].tolist() for column in SOURCE_COLUMNS}
    site_ids = (
        annotations["site_id"].tolist()
        if "site_id" in annotations.columns
        else [""] * len(annotations)
    )
    return [
        _Annotation(
            image_id=str(values["image_id"][position]),
            point_id=str(values["point_id"][position]),
            row=int(values["row"][position]),
            col=int(values["col"][position]),
            benthic_attribute_id=str(values["benthic_attribute_id"][position]),
            benthic_attribute_name=str(values["benthic_attribute_name"][position]),
            growth_form_id=str(values["growth_form_id"][position]),
            growth_form_name=str(values["growth_form_name"][position]),
            region_id=str(values["region_id"][position]),
            region_name=str(values["region_name"][position]),
            site_id=str(site_ids[position]),
        )
        for position in range(len(annotations))
    ]


def _image_index(
    records: Sequence[_Annotation],
    region_ids_by_attribute: Mapping[str, frozenset[str]],
) -> dict[str, _Image]:
    """Group annotations by image, deciding each image's eligibility as it goes.

    An image is eligible when at least one of its ground-truth annotations
    carries a region-discriminating attribute. Discrimination is judged against
    the regions the corpus actually contains, not a fixed MERMAID region list.
    """
    observed = frozenset(record.region_id for record in records)
    discriminating = {
        attribute_id: is_region_discriminating(
            region_ids_by_attribute.get(attribute_id, frozenset()), observed
        )
        for attribute_id in {record.benthic_attribute_id for record in records}
    }

    images: dict[str, _Image] = {}
    for record in records:
        image = images.get(record.image_id)
        if image is None:
            images[record.image_id] = _Image(
                region_id=record.region_id,
                region_name=record.region_name,
                eligible=discriminating[record.benthic_attribute_id],
                points=[record],
            )
            continue
        if image.region_id != record.region_id:
            raise ValueError(
                f"image {record.image_id!r} carries annotations with disagreeing"
                f" regions: {sorted({image.region_id, record.region_id})}"
            )
        image.eligible = image.eligible or discriminating[record.benthic_attribute_id]
        image.points.append(record)
    return images


def _census_region_ids(
    images: Mapping[str, _Image], census_region_names: frozenset[str]
) -> frozenset[str]:
    """Region ids for the named census regions, raising on a name that matches none.

    A rename upstream must stop the build rather than silently produce a probe
    with no Atlantic census in it, which would be an Indo-Pacific-only probe
    wearing the right filename.
    """
    by_name = {image.region_name: image.region_id for image in images.values()}
    unmatched = sorted(census_region_names - set(by_name))
    if unmatched:
        raise ValueError(
            f"census region(s) {unmatched} match no region in the data;"
            f" available: {sorted(by_name)}"
        )
    return frozenset(by_name[name] for name in census_region_names)


def _hash_order(image_ids: Iterable[str], seed: int) -> list[str]:
    """Image ids in a seeded but stable order.

    Hashing the id with the seed gives the same probe for the same inputs on
    any machine and any library version, while a new seed reshuffles.
    """
    return sorted(
        image_ids,
        key=lambda image_id: hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest(),
    )


def _allocate(pools: Mapping[str, int], *, budget: int, floor: int) -> dict[str, int]:
    """Split `budget` images across pools proportional to size, then floor and cap.

    Largest remainder over integers, so the split never depends on floating
    point. The floor lifts a region whose proportional share would swamp it;
    the pool size caps whatever the floor asks for.
    """
    total = sum(pools.values())
    if total == 0:
        return dict.fromkeys(pools, 0)

    usable = max(budget, 0)
    allocated = {region_id: usable * size // total for region_id, size in pools.items()}
    by_remainder = sorted(
        pools, key=lambda region_id: (-((usable * pools[region_id]) % total), region_id)
    )
    for region_id in by_remainder[: usable - sum(allocated.values())]:
        allocated[region_id] += 1

    return {
        region_id: min(size, max(floor, allocated[region_id])) for region_id, size in pools.items()
    }


def _probe_rows(
    images: Mapping[str, _Image],
    selected: Iterable[str],
    census_region_ids: frozenset[str],
) -> pd.DataFrame:
    """Every annotation on a selected image, in the frozen probe schema.

    Emitted already in the canonical (image, point) order the content hash
    reads, so the written parquet and its hash agree by construction.
    """
    rows = [
        {
            "image_id": point.image_id,
            "point_id": point.point_id,
            "row": point.row,
            "col": point.col,
            "benthic_attribute_id": point.benthic_attribute_id,
            "benthic_attribute_name": point.benthic_attribute_name,
            "growth_form_id": point.growth_form_id,
            "growth_form_name": point.growth_form_name,
            "gt_label": combine_ba_gf(point.benthic_attribute_id, point.growth_form_id),
            "region_id": point.region_id,
            "region_name": point.region_name,
            "site_id": point.site_id,
            "held_out": images[image_id].region_id not in census_region_ids,
        }
        for image_id in sorted(selected)
        for point in sorted(images[image_id].points, key=lambda point: point.point_id)
    ]
    return pd.DataFrame(rows, columns=pd.Index(PROBE_COLUMNS))


def _stratum_counts(
    region_id: str,
    region_image_ids: Sequence[str],
    images: Mapping[str, _Image],
    selected: frozenset[str] | set[str],
    census_region_ids: frozenset[str],
) -> StratumCounts:
    taken = [image_id for image_id in region_image_ids if image_id in selected]
    return StratumCounts(
        region_id=region_id,
        region_name=images[region_image_ids[0]].region_name,
        policy=CENSUS if region_id in census_region_ids else SAMPLE,
        held_out=region_id not in census_region_ids,
        n_region_images=len(region_image_ids),
        n_eligible_images=sum(1 for image_id in region_image_ids if images[image_id].eligible),
        n_images=len(taken),
        n_points=sum(len(images[image_id].points) for image_id in taken),
    )


def _canonical(value: object) -> str:
    """One probe cell as text, independent of the dtype it arrived in."""
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    if isinstance(value, int | np.integer):
        return str(int(value))
    return str(value)
