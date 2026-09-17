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

So do the corpus-wide (attribute, region) annotation counts, on the same
grounds and read over the whole export rather than the selected points. Triage
reads them to tell an incomplete region list from a model mistake, and the
probe holds a few of what the corpus holds hundreds of: counted on the probe
alone, a broadly-annotated pair would not clear the threshold.

So do the display names and the taxonomic ancestry. A report that resolves a
name at scoring time depends on a later state of the taxonomy than the score
it annotates, and `within_branch_share` reads ancestry the same way the region
predicates read regions. Both are pinned by their own manifest hash and sit
outside `content_hash`, which covers the selected points alone: a rename must
not invalidate a cached feature shard.

Selection is pure: nothing here reaches S3 or the network, and
`read_annotations` takes a path DuckDB can open.
"""

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

import duckdb
import numpy as np
import pandas as pd

from mermaid_classifier.common.benthic_attributes import combine_ba_gf
from mermaid_classifier.region_eval.region_rules import is_region_discriminating

PROBE_VERSION = "1"

# The probe directory's layout, named where the probe is defined so the builder
# that writes a file and the scorer that reads it cannot drift apart.
PROBE_POINTS_FILE = "probe_points.parquet"
PROBE_REGIONS_FILE = "ba_regions.json"
PROBE_COUNTS_FILE = "ba_region_counts.json"
PROBE_NAMES_FILE = "names.json"
PROBE_ANCESTRY_FILE = "ba_ancestry.json"
PROBE_MANIFEST_FILE = "manifest.json"
PROBE_FEATURES_FILE = "probe_features.npz"

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
class NameSnapshot:
    """Display names for everything a region report renders, in three sections.

    Sectioned rather than flattened so a region and a benthic attribute that
    share a name stay distinguishable, and so a reader of the JSON can tell
    which lookup failed when a name is missing.
    """

    benthic_attributes: Mapping[str, str]
    growth_forms: Mapping[str, str]
    regions: Mapping[str, str]


@dataclasses.dataclass(frozen=True)
class ProbeSet:
    """The selected points, what they were drawn from, and the hashes that pin both.

    `ground_truth_counts` is corpus-wide: every confirmed annotation in the
    source export, not only the ones selected here.
    """

    rows: pd.DataFrame
    strata: tuple[StratumCounts, ...]
    options: ProbeSelectionOptions
    n_rows_unrecorded_region_dropped: int
    content_hash: str
    region_snapshot_hash: str
    ground_truth_counts: dict[tuple[str, str], int]
    ground_truth_counts_hash: str


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

    frame = _normalized_frame(annotations)
    recorded = frame.loc[frame["region_id"] != ""]
    image_regions = _image_table(recorded, region_ids_by_attribute)
    census_region_ids = _census_region_ids(image_regions, selection.census_region_names)

    selected: set[str] = set()
    pools: dict[str, list[str]] = {}
    for region_id_key, group in image_regions.groupby("region_id"):
        region_id = str(region_id_key)
        if region_id in census_region_ids:
            selected.update(group.index)
        else:
            pools[region_id] = _hash_order(group.index[group["eligible"]], selection.seed)

    allocation = _allocate(
        {region_id: len(pool) for region_id, pool in pools.items()},
        budget=selection.target_images - len(selected),
        floor=selection.region_floor,
    )
    for region_id, pool in pools.items():
        selected.update(pool[: allocation[region_id]])

    rows = _probe_rows(recorded, selected, census_region_ids)
    image_points = recorded.groupby("image_id").size()
    strata = tuple(
        _stratum_counts(region_id, image_regions, image_points, selected, census_region_ids)
        for region_id in sorted(image_regions["region_id"].unique())
    )

    counts = ground_truth_counts(frame)
    return ProbeSet(
        rows=rows,
        strata=strata,
        options=selection,
        n_rows_unrecorded_region_dropped=len(frame) - len(recorded),
        content_hash=probe_content_hash(rows),
        region_snapshot_hash=region_snapshot_hash(region_ids_by_attribute),
        ground_truth_counts=counts,
        ground_truth_counts_hash=ground_truth_counts_hash(counts),
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


def _canonical_json(payload: object) -> str:
    """The canonical serialization every probe-snapshot hash is taken over."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _snapshot_hash(payload: object) -> str:
    """The sha256 hex digest of a payload's canonical serialization."""
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def region_snapshot_json(region_ids_by_attribute: Mapping[str, frozenset[str]]) -> str:
    return _canonical_json(region_snapshot(region_ids_by_attribute))


def region_snapshot_hash(region_ids_by_attribute: Mapping[str, frozenset[str]]) -> str:
    return _snapshot_hash(region_snapshot(region_ids_by_attribute))


def ground_truth_counts(annotations: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Confirmed annotations of each (benthic attribute, region) pair.

    Read over every row given, which is the whole export rather than the
    probe subset: this is the corpus evidence triage weighs an out-of-region
    prediction against, and the probe holds too few of any one pair to carry a
    threshold meant for the corpus.

    A row whose region is unrecorded answers no triage question -- every
    lookup arrives with an image's recorded region -- and is not counted.
    """
    recorded = annotations.loc[annotations["region_id"] != ""]
    grouped = recorded.groupby(["benthic_attribute_id", "region_id"]).size()
    return {cast(tuple[str, str], key): int(count) for key, count in grouped.items()}


def ground_truth_counts_snapshot(
    counts: Mapping[tuple[str, str], int],
) -> dict[str, dict[str, int]]:
    """The counts nested by attribute then region, JSON-ready."""
    nested: dict[str, dict[str, int]] = {}
    for (attribute_id, region_id), count in counts.items():
        nested.setdefault(attribute_id, {})[region_id] = int(count)
    return nested


def ground_truth_counts_json(counts: Mapping[tuple[str, str], int]) -> str:
    return _canonical_json(ground_truth_counts_snapshot(counts))


def ground_truth_counts_hash(counts: Mapping[tuple[str, str], int]) -> str:
    return _snapshot_hash(ground_truth_counts_snapshot(counts))


def name_snapshot(names: NameSnapshot) -> dict[str, dict[str, str]]:
    """The display names as JSON-ready sections."""
    return {
        "benthic_attributes": dict(names.benthic_attributes),
        "growth_forms": dict(names.growth_forms),
        "regions": dict(names.regions),
    }


def name_snapshot_json(names: NameSnapshot) -> str:
    return _canonical_json(name_snapshot(names))


def name_snapshot_hash(names: NameSnapshot) -> str:
    return _snapshot_hash(name_snapshot(names))


def ancestry_snapshot(
    ancestry_by_attribute: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """The ancestry paths as JSON-ready lists, each kept root-first."""
    return {attribute_id: list(path) for attribute_id, path in ancestry_by_attribute.items()}


def ancestry_snapshot_json(ancestry_by_attribute: Mapping[str, Sequence[str]]) -> str:
    return _canonical_json(ancestry_snapshot(ancestry_by_attribute))


def ancestry_snapshot_hash(ancestry_by_attribute: Mapping[str, Sequence[str]]) -> str:
    return _snapshot_hash(ancestry_snapshot(ancestry_by_attribute))


def build_manifest(
    probe: ProbeSet,
    *,
    source_uri: str,
    source_etag: str | None,
    source_row_count: int,
    builder_git_sha: str,
    names: NameSnapshot | None = None,
    ancestry: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Everything needed to say whether two scores were taken on the same probe.

    The source ETag and row count pin the export, the content hash pins the
    points, the region-snapshot hash pins the taxonomy, and the name and
    ancestry hashes pin what the reports render off it.

    A probe frozen without the names or the ancestry records a null hash for
    them, which is the state a score degrades against rather than a hash of
    nothing.
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
        "ground_truth_counts_hash": probe.ground_truth_counts_hash,
        "n_ground_truth_pairs": len(probe.ground_truth_counts),
        "names_hash": None if names is None else name_snapshot_hash(names),
        "n_names": {
            section: len(entries)
            for section, entries in ({} if names is None else name_snapshot(names)).items()
        },
        "ancestry_hash": None if ancestry is None else ancestry_snapshot_hash(ancestry),
        "n_ancestry_attributes": 0 if ancestry is None else len(ancestry),
        "seed": probe.options.seed,
        "target_images": probe.options.target_images,
        "region_floor": probe.options.region_floor,
        "census_region_names": sorted(probe.options.census_region_names),
        "n_images": sum(stratum.n_images for stratum in probe.strata),
        "n_points": sum(stratum.n_points for stratum in probe.strata),
        "strata": [dataclasses.asdict(stratum) for stratum in probe.strata],
        "builder_git_sha": builder_git_sha,
    }


def _normalized_frame(annotations: pd.DataFrame) -> pd.DataFrame:
    """The source frame coerced to the dtypes selection and hashing depend on.

    A caller's own frame may carry numeric ids or omit `site_id` entirely; the
    probe schema and its content hash must be stable regardless.
    """
    missing = [column for column in SOURCE_COLUMNS if column not in annotations.columns]
    if missing:
        raise ValueError(f"annotations are missing column(s): {', '.join(missing)}")

    text_columns = [column for column in SOURCE_COLUMNS if column not in ("row", "col")]
    site_id = (
        annotations["site_id"]
        if "site_id" in annotations.columns
        else pd.Series([""] * len(annotations), index=annotations.index)
    )
    normalized = annotations.assign(
        **{column: annotations[column].astype(str) for column in text_columns},
        row=annotations["row"].astype(int),
        col=annotations["col"].astype(int),
        site_id=site_id.astype(str),
    )
    columns: list[str] = [*SOURCE_COLUMNS, "site_id"]
    return normalized.loc[:, columns]


def _image_table(
    recorded: pd.DataFrame,
    region_ids_by_attribute: Mapping[str, frozenset[str]],
) -> pd.DataFrame:
    """One row per image: its region and whether any of its annotations is
    region-discriminating.

    An image is eligible when at least one of its ground-truth annotations
    carries a region-discriminating attribute. Discrimination is judged against
    the regions the corpus actually contains, not a fixed MERMAID region list.
    Raises when one image's annotations carry more than one region -- a
    data-integrity break upstream, not something selection should resolve by
    picking one.
    """
    region_counts = recorded.groupby("image_id")["region_id"].nunique()
    conflicted = region_counts.loc[region_counts > 1]
    if len(conflicted) > 0:
        image_id = str(conflicted.index[0])
        regions = sorted(recorded.loc[recorded["image_id"] == image_id, "region_id"].unique())
        raise ValueError(
            f"image {image_id!r} carries annotations with disagreeing regions: {regions}"
        )

    observed = frozenset(recorded["region_id"])
    discriminating_by_attribute = {
        attribute_id: is_region_discriminating(
            region_ids_by_attribute.get(attribute_id, frozenset()), observed
        )
        for attribute_id in recorded["benthic_attribute_id"].unique()
    }
    tagged = recorded.assign(
        discriminating=recorded["benthic_attribute_id"].map(
            lambda attribute_id: discriminating_by_attribute[attribute_id]
        )
    )
    return cast(
        pd.DataFrame,
        tagged.groupby("image_id").agg(
            region_id=("region_id", "first"),
            region_name=("region_name", "first"),
            eligible=("discriminating", "any"),
        ),
    )


def _census_region_ids(
    image_regions: pd.DataFrame, census_region_names: frozenset[str]
) -> frozenset[str]:
    """Region ids for the named census regions, raising on a name that matches none.

    A rename upstream must stop the build rather than silently produce a probe
    with no Atlantic census in it, which would be an Indo-Pacific-only probe
    wearing the right filename.
    """
    by_name = image_regions.drop_duplicates("region_name").set_index("region_name")["region_id"]
    unmatched = sorted(census_region_names - set(by_name.index))
    if unmatched:
        raise ValueError(
            f"census region(s) {unmatched} match no region in the data;"
            f" available: {sorted(by_name.index)}"
        )
    return frozenset(str(by_name[name]) for name in census_region_names)


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
    recorded: pd.DataFrame,
    selected: Iterable[str],
    census_region_ids: frozenset[str],
) -> pd.DataFrame:
    """Every annotation on a selected image, in the frozen probe schema.

    Emitted already in the canonical (image, point) order the content hash
    reads, so the written parquet and its hash agree by construction.
    """
    points = recorded.loc[recorded["image_id"].isin(list(selected))].sort_values(
        ["image_id", "point_id"], kind="stable"
    )
    points = points.assign(
        gt_label=[
            combine_ba_gf(attribute_id, growth_form_id)
            for attribute_id, growth_form_id in zip(
                points["benthic_attribute_id"], points["growth_form_id"], strict=True
            )
        ],
        held_out=~points["region_id"].isin(list(census_region_ids)),
    )
    return cast(pd.DataFrame, points.loc[:, list(PROBE_COLUMNS)]).reset_index(drop=True)


def _stratum_counts(
    region_id: str,
    image_regions: pd.DataFrame,
    image_points: pd.Series,
    selected: frozenset[str] | set[str],
    census_region_ids: frozenset[str],
) -> StratumCounts:
    region_images = image_regions.loc[image_regions["region_id"] == region_id]
    taken = [image_id for image_id in region_images.index if image_id in selected]
    return StratumCounts(
        region_id=region_id,
        region_name=str(region_images["region_name"].iloc[0]),
        policy=CENSUS if region_id in census_region_ids else SAMPLE,
        held_out=region_id not in census_region_ids,
        n_region_images=len(region_images),
        n_eligible_images=int(region_images["eligible"].sum()),
        n_images=len(taken),
        n_points=int(image_points.loc[taken].sum()),
    )


def _canonical(value: object) -> str:
    """One probe cell as text, independent of the dtype it arrived in."""
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    if isinstance(value, int | np.integer):
        return str(int(value))
    return str(value)
