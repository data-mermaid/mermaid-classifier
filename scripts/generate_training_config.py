"""Generate a self-contained training-config directory.

Replicates the CoralNet -> MERMAID -> top-108 mapping logic from the
mermaid-segmentation notebook coralnet_eda_validated.ipynb, against the
live MERMAID API. Produces sources.csv, included_labels.csv,
rollups.csv, and an audit README.md.

For each CoralNet-mapped benthic attribute:

1. If the BA is in EXCLUDED_NAMES, drop entirely (no rollup, no class).
2. If already in the top-108 set, keep as-is.
3. Otherwise walk parents (nearest first) to find a top-108 ancestor.
4. Otherwise drop the label.

With --roll-up-species and/or --roll-up-region-specific, each top-108 BA
that is a species, or that MERMAID lists in exactly one region, is
replaced in the top-108 set by its nearest ancestor that is neither, so
its annotations roll up to that ancestor.

With --min-support-counts, each class then below --min-points points or
--min-groups (site, project) groups rolls up to its parent, thinnest
first, until every class meets both floors.

Porites is the one BA that retains growth-form distinctions: three
buckets (Branching, Massive, other/none) appear as separate classes in
included_labels.csv. Porites species and Porites genus annotations are
routed to the appropriate bucket via rollups (per Iain/Emily, 2026-05-08).

Run from the mermaid-classifier directory:

    uv run python scripts/generate_training_config.py

The script is the documented way to regenerate the training config on
demand. See `--help`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import pandas as pd

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary,
    CoralNetMermaidMapping,
    GrowthFormLibrary,
)
from mermaid_classifier.pyspacer.settings import set_env_vars_for_packages

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent

# Output goes in-repo so the generated config dir can be committed under
# sagemaker/configs/ and consumed repo-root-relative by classifier_train.py and
# the SageMaker launcher. The raw INPUTS, by contrast, live in the surrounding
# workspace and are not committed to this repo (the curated CoralNet source list
# and the Google-Drive-exported label mapping). Override any of these with the
# --output-dir / --sources-csv / --labels-csv flags.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sagemaker" / "configs" / "coralnet_top108"
DEFAULT_SOURCES_CSV = WORKSPACE_ROOT / "sagemaker" / "sources" / "CoralNetSourcesKept.csv"
DEFAULT_LABELS_CSV = (
    WORKSPACE_ROOT
    / "drive_label_mappings"
    / ("coralnet_labels_mermaid_mapping_annotations - mapped_to_mermaid_attributes.csv")
)

# Porites is the only genus that retains GF distinctions (Iain confirmed
# with Emily 2026-05-08). The bucket names are MERMAID growth-form names
# resolved to UUIDs at runtime via GrowthFormLibrary.
PORITES_NAME = "Porites"
PORITES_KEPT_GF_NAMES = ("Branching", "Massive")

# Excluded by name; defensive against the labels CSV being updated to
# set top100=1 for any of these. Annotations of these labels are dropped
# entirely (not rolled up to a parent — see priority_notes in the labels
# CSV for rationale; e.g. Dead coral would otherwise pollute Bare substrate).
EXCLUDED_NAMES = ("Dead coral", "Bleached coral", "Other invertebrates")

# Genus-species names, optionally with a parenthesized subgenus.
SPECIES_NAME_PATTERN = re.compile(r"^[A-Z][a-z]+( \([A-Z][a-z]+\))? [a-z]+$")
# Genus BAs whose name is disambiguated with a suffix, e.g. Turbinaria-coral.
GENUS_NAME_SUFFIXES = ("-coral", "-algae")
# Species whose genus is not one of their ancestors in the MERMAID tree,
# so the name-vs-ancestor check in is_species cannot recognize them.
ORPHAN_SPECIES_NAMES = (
    "Acanthaster planci",
    "Balanophyllia (Balanophyllia) elegans",
    "Culcita novaeguineae",
    "Culicia tenella",
    "Didemnum molle",
    "Diplosoma virens",
    "Enhalus acoroides",
    "Linckia laevigata",
    "Neopetrosia chaliniformis",
    "Pinctada margaritifera",
    "Pinctada maxima",
    "Radianthus magnifica",
    "Radianthus malu",
    "Spirobranchus giganteus",
    "Stephanocoenia intersepta",
    "Stephanometra indica",
    "Stichaster australis",
    "Svenzea zeai",
    "Trididemnum solidum",
    "Verongula rigida",
)

logger = logging.getLogger(__name__)


# ---------- API client constructors (injection points for tests) ----------


def _load_ba_library() -> BenthicAttributeLibrary:
    return BenthicAttributeLibrary()


def _load_gf_library() -> GrowthFormLibrary:
    return GrowthFormLibrary()


def _load_cn_mapping() -> CoralNetMermaidMapping:
    return CoralNetMermaidMapping()


# ---------- Porites GF helpers ----------


def _porites_bucket(gf_name: str) -> str:
    """Map any GF name to one of PORITES_KEPT_GF_NAMES, or '' for the
    'other/none' bucket. Case-insensitive.
    """
    canonical = (gf_name or "").strip().title()
    return canonical if canonical in PORITES_KEPT_GF_NAMES else ""


def _load_species_gf_lookup(labels_df: pd.DataFrame) -> dict[str, str]:
    """ba_id -> canonical Porites bucket name (or '').

    Reads the labels CSV's `growth forms` column. Single-value entries
    (species' inherent GF, e.g. Porites lobata = 'massive') are bucketed
    via _porites_bucket(). Multi-value entries (e.g. Porites genus's
    'branching, massive, other/none') describe bucket *choices* for a
    parent rather than an inherent GF, so they're skipped.
    """
    if "growth forms" not in labels_df.columns:
        return {}
    lookup: dict[str, str] = {}
    for _, row in labels_df.iterrows():
        raw = row.get("growth forms")
        if not isinstance(raw, str) or not raw.strip():
            continue
        if "," in raw:
            continue  # multi-value: bucket choices, not inherent GF
        bucket = _porites_bucket(raw)
        lookup[str(row["id"])] = bucket
    return lookup


def _build_bucket_gf_uuid_lookup(gf_library: GrowthFormLibrary) -> dict[str, str]:
    """{'': '', 'Branching': <uuid>, 'Massive': <uuid>}."""
    name_to_uuid = {name: uuid for uuid, name in gf_library.by_id.items()}
    out = {"": ""}
    for bucket in PORITES_KEPT_GF_NAMES:
        if bucket not in name_to_uuid:
            raise KeyError(
                f"GrowthFormLibrary has no '{bucket}' growth form; cannot build Porites buckets."
            )
        out[bucket] = name_to_uuid[bucket]
    return out


class _FileAudit(TypedDict):
    path: str
    mtime: str
    sha1_prefix: str


class _Audits(TypedDict):
    labels: _FileAudit
    sources: _FileAudit
    ba_api: str
    cn_api: str


# ---------- File audit ----------


def _file_audit(path: Path) -> _FileAudit:
    if not path.exists():
        return {"path": str(path), "mtime": "MISSING", "sha1_prefix": "MISSING"}
    stat = path.stat()
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return {
        "path": str(path.relative_to(WORKSPACE_ROOT))
        if path.is_relative_to(WORKSPACE_ROOT)
        else str(path),
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sha1_prefix": h.hexdigest()[:10],
    }


def _iter_audit(items: Any, key: Any = None) -> str:
    """sha1 prefix of a JSON-stringified iterable.

    `key` extracts a hashable representation of each item; defaults to
    `__dict__` for class instances and the item itself for dicts.
    """
    h = hashlib.sha1()
    for item in items:
        if key is not None:
            payload = key(item)
        elif hasattr(item, "__dict__"):
            payload = item.__dict__
        else:
            payload = item
        h.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:10]


# ---------- Step A: top-108 set ----------


def resolve_top108_uuids(
    top108_df: pd.DataFrame,
    ba_lib: BenthicAttributeLibrary,
) -> tuple[set[str], list[str], list[str]]:
    """Resolve top-108 names -> UUIDs via the live BA library.

    Names in EXCLUDED_NAMES are dropped (defensive against future CSV
    updates that flip top100=1 for them). Names absent from the API are
    logged and skipped, not raised, so a slow-moving CSV doesn't crash
    the run.

    Returns (uuid_set, unresolved_names, excluded_names_seen).
    """
    uuids: set[str] = set()
    unresolved: list[str] = []
    excluded_seen: list[str] = []
    for name in top108_df["name"]:
        if name in EXCLUDED_NAMES:
            excluded_seen.append(name)
            continue
        try:
            uuids.add(ba_lib.name_to_id(name))
        except KeyError:
            unresolved.append(name)
    if unresolved:
        logger.warning(
            "%d top-108 name(s) not in MERMAID API: %s",
            len(unresolved),
            sorted(unresolved),
        )
    if excluded_seen:
        logger.warning(
            "%d top-108 name(s) excluded by EXCLUDED_NAMES: %s",
            len(excluded_seen),
            sorted(excluded_seen),
        )
    return uuids, unresolved, excluded_seen


def resolve_excluded_uuids(
    ba_lib: BenthicAttributeLibrary,
) -> set[str]:
    """Resolve EXCLUDED_NAMES -> UUIDs. Missing names are logged + skipped."""
    out: set[str] = set()
    for name in EXCLUDED_NAMES:
        try:
            out.add(ba_lib.name_to_id(name))
        except KeyError:
            logger.warning(
                "EXCLUDED_NAMES contains %r which is not in MERMAID API;"
                " skipping (will not block any annotations).",
                name,
            )
    return out


# ---------- Step A2: coarsen the top-108 set ----------

BaPredicate = Callable[[str, BenthicAttributeLibrary], bool]


def _genus_name(ba_name: str) -> str:
    for suffix in GENUS_NAME_SUFFIXES:
        ba_name = ba_name.removesuffix(suffix)
    return ba_name


def is_species(ba_id: str, ba_lib: BenthicAttributeLibrary) -> bool:
    """A BA is a species when its name is a binomial whose genus is one of
    its ancestors, or when it is listed in ORPHAN_SPECIES_NAMES.

    The MERMAID tree carries no taxonomic rank, so rank is inferred from
    the name; the ancestor check rejects common names like "Hard coral".
    """
    name = ba_lib.id_to_name(ba_id)
    if name in ORPHAN_SPECIES_NAMES:
        return True
    if not SPECIES_NAME_PATTERN.match(name):
        return False
    genus = name.split(" ", 1)[0]
    return any(_genus_name(ba_lib.id_to_name(a)) == genus for a in ba_lib.get_ancestor_ids(ba_id))


def is_region_specific(ba_id: str, ba_lib: BenthicAttributeLibrary) -> bool:
    """True when MERMAID lists the BA in exactly one region."""
    return len(ba_lib.region_ids_by_id.get(ba_id, frozenset())) == 1


def coarsen_top108(
    top108_uuids: set[str],
    ba_lib: BenthicAttributeLibrary,
    excluded_uuids: set[str],
    predicates: list[BaPredicate],
) -> set[str]:
    """Replace each top-108 BA that matches any predicate with its nearest
    ancestor that matches none.

    A replacement that is an excluded BA is not added. A BA with no
    non-matching ancestor stays in the set, since nothing coarser exists.
    """

    def matches(ba_id: str) -> bool:
        return any(p(ba_id, ba_lib) for p in predicates)

    coarsened: set[str] = set()
    for ba_id in sorted(top108_uuids):
        if not matches(ba_id):
            coarsened.add(ba_id)
            continue
        ancestors_leaf_first = list(reversed(ba_lib.get_ancestor_ids(ba_id)))
        replacement = next((a for a in ancestors_leaf_first if not matches(a)), None)
        if replacement is None:
            logger.warning(
                "%s matches a roll-up rule but has no coarser ancestor; keeping it.",
                ba_lib.id_to_name(ba_id),
            )
            coarsened.add(ba_id)
        elif replacement not in excluded_uuids:
            coarsened.add(replacement)
    return coarsened


ClassKey = tuple[str, str]
"""A class of included_labels.csv: (ba_id, gf_id), gf_id '' outside the Porites buckets."""


@dataclass
class ClassSupport:
    points: int = 0
    groups: set[tuple[str, str]] = field(default_factory=lambda: set[tuple[str, str]]())


class MinSupportRollup(NamedTuple):
    class_key: ClassKey
    points: int
    n_groups: int
    target: ClassKey | None


def read_support_counts(path: Path) -> dict[ClassKey, ClassSupport]:
    """Per-class points and (site, project_id) group set from a
    `class_key, site, project_id, n` counts CSV."""
    support: dict[ClassKey, ClassSupport] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = {"class_key", "site", "project_id", "n"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"support counts CSV {path} missing columns: {sorted(missing)}")
        for row in reader:
            ba_id, _, gf_id = row["class_key"].partition("::")
            entry = support.setdefault((ba_id, gf_id), ClassSupport())
            entry.points += int(row["n"])
            entry.groups.add((row["site"], row["project_id"]))
    return support


def apply_min_support(
    classes: set[ClassKey],
    counts: dict[ClassKey, ClassSupport],
    ba_lib: BenthicAttributeLibrary,
    excluded_uuids: set[str],
    porites_uuid: str | None,
    min_points: int,
    min_groups: int,
) -> tuple[set[ClassKey], list[MinSupportRollup]]:
    """Roll the thinnest class below either floor into its parent until
    every class meets both, returning the final classes and each roll-up
    with its final target (None when dropped).

    A Porites GF bucket merges into `(Porites, '')`, and `(Porites, '')`
    carries any surviving buckets with it. A class whose parent is missing
    or excluded drops. Each step moves a class strictly up the tree, so the
    loop terminates.
    """
    support = {
        key: ClassSupport(counts[key].points, set(counts[key].groups))
        if key in counts
        else ClassSupport()
        for key in classes
    }
    unknown = sorted(set(counts) - classes)
    if unknown:
        logger.warning(
            "%d support-count class key(s) are not classes of this config; ignoring: %s",
            len(unknown),
            unknown,
        )
    # A rolled-up class forwards to its target, so a later roll-up into it
    # follows the chain and never re-creates it.
    moved: dict[ClassKey, ClassKey | None] = {}

    def resolve(key: ClassKey | None) -> ClassKey | None:
        while key is not None and key in moved:
            key = moved[key]
        return key

    rolled: list[MinSupportRollup] = []
    while True:
        thin = [
            (entry.points, key)
            for key, entry in support.items()
            if entry.points < min_points or len(entry.groups) < min_groups
        ]
        if not thin:
            break
        _, key = min(thin)
        ba_id, gf_id = key
        leaving = [key]
        if gf_id:
            target: ClassKey | None = (ba_id, "")
        else:
            ancestors = ba_lib.get_ancestor_ids(ba_id)
            parent = ancestors[-1] if ancestors else None
            target = None if parent is None or parent in excluded_uuids else (parent, "")
            if ba_id == porites_uuid:
                leaving += sorted(k for k in support if k[0] == ba_id and k[1])
        target = resolve(target)
        merged = ClassSupport()
        for k in leaving:
            entry = support.pop(k)
            rolled.append(MinSupportRollup(k, entry.points, len(entry.groups), target))
            moved[k] = target
            merged.points += entry.points
            merged.groups |= entry.groups
        if target is not None:
            dest = support.setdefault(target, ClassSupport())
            dest.points += merged.points
            dest.groups |= merged.groups
    return set(support), [r._replace(target=resolve(r.target)) for r in rolled]


def _nearest_included(
    ba_id: str, ba_lib: BenthicAttributeLibrary, included: set[str]
) -> str | None:
    """The BA itself if included, else its nearest included ancestor."""
    if ba_id in included:
        return ba_id
    ancestors_leaf_first = reversed(ba_lib.get_ancestor_ids(ba_id))
    return next((a for a in ancestors_leaf_first if a in included), None)


def class_changes(
    ba_ids: set[str],
    ba_lib: BenthicAttributeLibrary,
    excluded_uuids: set[str],
    before: set[str],
    after: set[str],
) -> list[tuple[str, str | None, str | None]]:
    """(BA, class before, class after) for each BA whose class differs
    between two included sets; None means its annotations drop."""
    changes: list[tuple[str, str | None, str | None]] = []
    for ba_id in sorted(ba_ids):
        if ba_id in excluded_uuids or ba_id not in ba_lib.by_id:
            continue
        old = _nearest_included(ba_id, ba_lib, before)
        new = _nearest_included(ba_id, ba_lib, after)
        if old != new:
            changes.append((ba_id, old, new))
    return changes


# ---------- Step B: hierarchy walk -> rollup rows ----------


def build_rollup_rows(
    cn_mapping: CoralNetMermaidMapping,
    ba_lib: BenthicAttributeLibrary,
    gf_library: GrowthFormLibrary,
    top108_uuids: set[str],
    porites_uuid: str | None,
    excluded_uuids: set[str],
    bucket_gf_uuid_lookup: dict[str, str],
    species_gf_lookup: dict[str, str],
) -> tuple[list[tuple[str, str, str, str]], dict[str, int]]:
    """Walk the BA hierarchy from each CoralNet-mapped BA up to a
    top-108 ancestor, emitting rollup rows.

    Porites is special-cased: target_gf is one of Branching/Massive/'',
    chosen from CN's GF if non-empty, else the species' inherent GF
    from the labels CSV. All other top-108 BAs collapse GF to ''.

    Annotations whose source BA is in excluded_uuids are dropped (no
    rollup row, no walking past).
    """
    rollups: dict[tuple[str, str], tuple[str, str]] = {}
    dropped = {
        "no_top_level": 0,
        "unknown_uuid": 0,
        "null_ba": 0,
        "excluded": 0,
    }
    for entry in cn_mapping.mapping.values():
        src_ba = entry.benthic_attribute_id
        src_gf = entry.growth_form_id or ""
        if not src_ba:
            dropped["null_ba"] += 1
            continue
        if src_ba in excluded_uuids:
            dropped["excluded"] += 1
            continue
        if src_ba not in ba_lib.by_id:
            dropped["unknown_uuid"] += 1
            logger.warning(
                "CoralNet provider_id %s maps to BA UUID %s not in the live MERMAID API; dropping.",
                entry.provider_id,
                src_ba,
            )
            continue

        target_ba = _nearest_included(src_ba, ba_lib, top108_uuids)
        if target_ba is None:
            dropped["no_top_level"] += 1
            continue

        # Determine target_gf. Porites is special; everyone else collapses.
        if porites_uuid is not None and target_ba == porites_uuid:
            if src_gf:
                bucket_name = _porites_bucket(gf_library.id_to_name(src_gf))
            elif src_ba == porites_uuid:
                # Porites genus, no CN GF -> the empty bucket.
                bucket_name = ""
            else:
                # Porites species, no CN GF -> use species' inherent.
                bucket_name = species_gf_lookup.get(src_ba, "")
            target_gf = bucket_gf_uuid_lookup[bucket_name]
        else:
            target_gf = ""

        # Skip self-mappings (no rollup needed).
        if (src_ba, src_gf) == (target_ba, target_gf):
            continue
        rollups[(src_ba, src_gf)] = (target_ba, target_gf)

    rows = sorted(
        (src_ba, src_gf, to_ba, to_gf) for (src_ba, src_gf), (to_ba, to_gf) in rollups.items()
    )
    # Categorize the deduped rollup rows for README reporting.
    breakdown = {
        "gf_collapse": sum(1 for (sb, _), (tb, _) in rollups.items() if sb == tb),
        "cross_ba": sum(1 for (sb, _), (tb, _) in rollups.items() if sb != tb),
    }
    porites_breakdown = _categorize_porites_rollups(
        rollups,
        porites_uuid,
        bucket_gf_uuid_lookup,
    )
    return rows, {**dropped, **breakdown, **porites_breakdown}


def _categorize_porites_rollups(
    rollups: dict[tuple[str, str], tuple[str, str]],
    porites_uuid: str | None,
    bucket_gf_uuid_lookup: dict[str, str],
) -> dict[str, int]:
    """Count Porites-related rollup rows for README reporting.

    Operates on the deduped rollups dict so counters match the rollup
    row count exactly (avoids double-counting CN entries that share a
    (from_ba, from_gf) key).
    """
    out = {
        "porites_genus_gf_collapse": 0,
        "porites_species_to_branching": 0,
        "porites_species_to_massive": 0,
        "porites_species_to_other": 0,
    }
    if porites_uuid is None:
        return out
    branching_uuid = bucket_gf_uuid_lookup.get("Branching", "")
    massive_uuid = bucket_gf_uuid_lookup.get("Massive", "")
    for (src_ba, _src_gf), (to_ba, to_gf) in rollups.items():
        if to_ba != porites_uuid:
            continue
        # A bucket merged into (Porites, '') maps to '' and counts as other.
        if src_ba == porites_uuid:
            out["porites_genus_gf_collapse"] += 1
        elif branching_uuid and to_gf == branching_uuid:
            out["porites_species_to_branching"] += 1
        elif massive_uuid and to_gf == massive_uuid:
            out["porites_species_to_massive"] += 1
        else:
            out["porites_species_to_other"] += 1
    return out


# ---------- Step C: included labels ----------


def build_included_label_rows(
    top108_uuids: set[str],
    porites_uuid: str | None,
    bucket_gf_uuid_lookup: dict[str, str],
) -> list[tuple[str, str]]:
    """Standard `(uuid, '')` rows + three Porites bucket rows."""
    rows: set[tuple[str, str]] = set()
    for uuid in top108_uuids:
        if porites_uuid is not None and uuid == porites_uuid:
            rows.add((porites_uuid, ""))
            for bucket in PORITES_KEPT_GF_NAMES:
                rows.add((porites_uuid, bucket_gf_uuid_lookup[bucket]))
        else:
            rows.add((uuid, ""))
    return sorted(rows)


# ---------- Step D: writers ----------


def write_rollups_csv(rows: list[tuple[str, str, str, str]], out_path: Path) -> int:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from_ba_id", "from_gf_id", "to_ba_id", "to_gf_id"])
        for r in rows:
            w.writerow(r)
    return len(rows)


def write_included_labels_csv(rows: list[tuple[str, str]], out_path: Path) -> int:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ba_id", "gf_id"])
        for r in rows:
            w.writerow(r)
    return len(rows)


def copy_sources_csv(in_path: Path, out_path: Path) -> int:
    """Pass-through: read input, normalize id-or-Source-ID column, write."""
    df = pd.read_csv(in_path)
    if "id" in df.columns:
        col = "id"
    elif "Source ID" in df.columns:
        col = "Source ID"
    else:
        raise ValueError(f"{in_path}: must have an 'id' or 'Source ID' column")
    out_path.write_text("id\n" + "\n".join(str(int(s)) for s in df[col]) + "\n")
    return len(df)


# ---------- Step E: README ----------


def write_readme(
    out_path: Path,
    audits: _Audits,
    counts: dict[str, Any],
    unresolved_top108: list[str],
    excluded_top108_seen: list[str],
    coarsening_rules: list[str] | None = None,
    added_classes: list[str] | None = None,
    class_change_rows: list[tuple[str, str, str]] | None = None,
    min_support_rows: list[tuple[str, int, int, str]] | None = None,
) -> None:
    cli = " ".join(sys.argv)
    coarsening_block = ""
    if coarsening_rules:
        added = added_classes or []
        changes = class_change_rows or []
        min_support_block = ""
        if min_support_rows is not None:
            min_support_block = (
                f"\nClasses rolled up by min support ({len(min_support_rows)}),"
                " in roll-up order:\n\n"
                "| Class | Points | Groups | Target |\n| --- | --- | --- | --- |\n"
                + "".join(
                    f"| {cls} | {points} | {groups} | {target or '(dropped)'} |\n"
                    for cls, points, groups, target in min_support_rows
                )
            )
        coarsening_block = (
            "\n## Coarsened labels\n\n"
            f"Rules: {', '.join(coarsening_rules)}.\n\n"
            f"Classes added as roll-up targets ({len(added)}):"
            f" {', '.join(added) or 'none'}.\n\n"
            f"Labels whose class changes ({len(changes)}), CoralNet-mapped or"
            " top-108:\n\n"
            "| Label | Class without rules | Class with rules |\n| --- | --- | --- |\n"
            + "".join(
                f"| {ba} | {old or '(dropped)'} | {new or '(dropped)'} |\n"
                for ba, old, new in changes
            )
            + min_support_block
        )
    unresolved_block = ""
    if unresolved_top108:
        unresolved_block = (
            "\n## Unresolved top-108 names\n\n"
            f"The following {len(unresolved_top108)} top-108 name(s) were"
            " not found in the live MERMAID API and were skipped (not in"
            " `included_labels.csv`):\n\n"
            + "".join(f"- `{n}`\n" for n in sorted(unresolved_top108))
        )
    excluded_top108_block = ""
    if excluded_top108_seen:
        excluded_top108_block = (
            "\n*Note*: the labels CSV currently sets `top100=1` for"
            f" {sorted(excluded_top108_seen)}, but EXCLUDED_NAMES dropped"
            " them anyway.\n"
        )
    body = f"""# Training Config: {out_path.parent.name}

Generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")} by
`scripts/generate_training_config.py`. Regenerate with:

```
{cli}
```

## Logic

Replicates the validated EDA notebook
(`data-mermaid/mermaid-segmentation@coralnet_eda/nbs/EDA/coralnet_eda_validated.ipynb`)
in CoralNet -> MERMAID -> top-108 hierarchy-walk space, against the live
MERMAID API. For each CoralNet-mapped benthic attribute:

1. If the BA is in `EXCLUDED_NAMES` ({", ".join(EXCLUDED_NAMES)}), drop entirely.
2. If already in the top-108 set, keep as-is.
3. Otherwise walk parents (nearest first) to find a top-108 ancestor.
4. Otherwise drop the label (`LabelFilter` removes it at training time).

## Source-of-truth inputs

| File | Modified | sha1[:10] |
| --- | --- | --- |
| `{audits["labels"]["path"]}` | {audits["labels"]["mtime"]} | `{audits["labels"]["sha1_prefix"]}` |
| `{audits["sources"]["path"]}` | {audits["sources"]["mtime"]} | `{audits["sources"]["sha1_prefix"]}` |

Live MERMAID API (snapshotted at run time):
- `https://api.datamermaid.org/v1/benthicattributes/?limit=5000` -> sha1 `{audits["ba_api"]}`
- `https://api.datamermaid.org/v1/classification/labelmappings/?provider=CoralNet` -> sha1 `{audits["cn_api"]}`

## Outputs

- `sources.csv` - **{counts["n_sources"]}** CoralNet source IDs (pass-through; no filtering).
- `included_labels.csv` - **{counts["n_included"]}** `(ba_id, gf_id)` rows
  (Porites contributes 3 rows for the Branching / Massive / other-none buckets;
  every other top-108 BA contributes 1 row).
- `rollups.csv` - **{counts["n_rollups"]}** unique `(from_ba, from_gf)` rows:
    - GF-collapse rows (top-108 BA with non-bucket GF -> same BA, empty GF): {counts["n_gf_collapse"]}
    - cross-BA rollups (species or genus -> top-108 ancestor): {counts["n_cross_ba"]}

### Porites buckets

Per Iain/Emily (2026-05-08), Porites is the only genus that retains
growth-form distinctions. `included_labels.csv` has three Porites rows:
`(Porites, Branching)`, `(Porites, Massive)`, `(Porites, '')`.

Rollups feeding the buckets:
- Porites genus with non-Branching/non-Massive GF -> `(Porites, '')`: {counts["n_porites_genus_gf_collapse"]}
- Porites species -> `(Porites, Branching)`: {counts["n_porites_species_to_branching"]}
- Porites species -> `(Porites, Massive)`: {counts["n_porites_species_to_massive"]}
- Porites species -> `(Porites, '')` (other/no inherent GF): {counts["n_porites_species_to_other"]}

Species' inherent GF comes from the labels CSV's `growth forms` column
(single-value rows only). CN-supplied GF wins over the inherent GF when
both are present.

### Excluded labels

These labels are excluded from training entirely (no rollup, no class):
{chr(10).join(f"- `{n}`" for n in EXCLUDED_NAMES)}

CoralNet mappings dropped because their BA is in EXCLUDED_NAMES: {counts["n_excluded"]}.
{excluded_top108_block}
### Other dropped CoralNet mappings

- Source BA UUID not in MERMAID API: {counts["n_unknown_uuid"]}
- No top-108 ancestor in the parent chain: {counts["n_no_top_level"]}
- CoralNet mapping with null benthic_attribute_id: {counts["n_null_ba"]}

## Deviation from notebook (justified)

The notebook works on benthic-attribute *names* only and ignores growth
forms. The training pipeline's `LabelRollupSpec` does literal
`(from_ba_id, from_gf_id)` tuple lookup, so to keep this config
self-contained (i.e. usable without setting
`DatasetOptions.drop_growthforms=True`) this script enumerates every
`(BA, GF)` pair present in the live CoralNet->MERMAID mapping and emits
one rollup row per pair. Most pairs collapse to `(top-108 BA, '')`;
Porites is the only exception (see above).
{coarsening_block}{unresolved_block}"""
    out_path.write_text(body)


# ---------- Validation ----------


def validate_outputs(out_dir: Path) -> None:
    """Round-trip included_labels.csv and rollups.csv through the pipeline's
    CsvSpec subclasses; check sources.csv has an 'id' column."""
    from mermaid_classifier.pyspacer.label_specs import (
        LabelFilter,
        LabelRollupSpec,
    )

    sources_path = out_dir / "sources.csv"
    with sources_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(
                f"{sources_path} must contain an 'id' column"
                f" (got columns: {list(reader.fieldnames or [])})"
            )
    with (out_dir / "included_labels.csv").open() as f:
        included = LabelFilter(f, inclusion=True)
    with (out_dir / "rollups.csv").open() as f:
        rollups = LabelRollupSpec(f)

    # Soft consistency: every rollup target must be in included_labels.
    included_ba_ids = {ba for (ba, _gf) in included.bagf_set}
    for (from_ba, from_gf), (to_ba, _to_gf) in rollups.lookup.items():
        if to_ba not in included_ba_ids:
            raise ValueError(
                f"rollups.csv has to_ba_id={to_ba}"
                f" (from {from_ba}, gf={from_gf!r}) not present in"
                f" included_labels.csv - generation bug."
            )


# ---------- Main ----------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--sources-csv",
        type=Path,
        default=DEFAULT_SOURCES_CSV,
        help="Pass-through input. Must have an `id` or `Source ID` column.",
    )
    p.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="Top-108 source-of-truth. Filter is `top100 == 1`.",
    )
    p.add_argument(
        "--roll-up-species",
        action="store_true",
        help="Replace species in the top-108 set with their nearest coarser ancestor.",
    )
    p.add_argument(
        "--roll-up-region-specific",
        action="store_true",
        help=(
            "Replace BAs that MERMAID lists in exactly one region with their"
            " nearest ancestor not listed in exactly one region (and, with"
            " --roll-up-species, not a species)."
        ),
    )
    p.add_argument(
        "--min-support-counts",
        type=Path,
        default=None,
        help=(
            "Counts CSV (class_key, site, project_id, n) for the coarsened classes."
            " Rolls each class below --min-points or --min-groups up to its parent."
        ),
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=500,
        help="Points a class needs to stay (only with --min-support-counts).",
    )
    p.add_argument(
        "--min-groups",
        type=int,
        default=10,
        help="Distinct (site, project) groups a class needs to stay (only with --min-support-counts).",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip round-tripping outputs through pipeline schemas (useful for tests).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    set_env_vars_for_packages()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    audits: _Audits = {
        "labels": _file_audit(args.labels_csv),
        "sources": _file_audit(args.sources_csv),
        "ba_api": "n/a",
        "cn_api": "n/a",
    }
    if audits["labels"]["mtime"] == "MISSING":
        raise FileNotFoundError(f"labels CSV not found: {args.labels_csv}")
    if audits["sources"]["mtime"] == "MISSING":
        raise FileNotFoundError(f"sources CSV not found: {args.sources_csv}")

    labels_df = pd.read_csv(args.labels_csv)
    required = {"id", "name", "top100"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels CSV {args.labels_csv} missing columns: {sorted(missing)}")
    top108_df: pd.DataFrame = labels_df[labels_df["top100"].fillna(0).astype(int) == 1].reset_index(
        drop=True
    )  # pyright: ignore[reportAssignmentType]  # pandas boolean-index always returns DataFrame here
    logger.info(
        "Loaded %d top-108 rows from %s",
        len(top108_df),
        args.labels_csv.name,
    )

    ba_lib = _load_ba_library()
    audits["ba_api"] = _iter_audit(ba_lib.raw_results)
    gf_library = _load_gf_library()
    cn_mapping = _load_cn_mapping()
    # Touch .mapping to trigger lazy load before fingerprinting.
    audits["cn_api"] = _iter_audit(cn_mapping.mapping.values())

    top108_uuids, unresolved, excluded_top108_seen = resolve_top108_uuids(top108_df, ba_lib)
    excluded_uuids = resolve_excluded_uuids(ba_lib)
    predicates: list[BaPredicate] = []
    coarsening_rules: list[str] = []
    if args.roll_up_species:
        predicates.append(is_species)
        coarsening_rules.append("species")
    if args.roll_up_region_specific:
        predicates.append(is_region_specific)
        coarsening_rules.append("region-specific (exactly one MERMAID region)")
    uncoarsened_uuids = top108_uuids
    top108_uuids = coarsen_top108(uncoarsened_uuids, ba_lib, excluded_uuids, predicates)
    try:
        porites_uuid: str | None = ba_lib.name_to_id(PORITES_NAME)
    except KeyError:
        logger.warning("Porites not found in MERMAID API; Porites bucket logic disabled.")
        porites_uuid = None
    bucket_gf_uuid_lookup = (
        _build_bucket_gf_uuid_lookup(gf_library) if porites_uuid is not None else {"": ""}
    )
    min_support_rows: list[tuple[str, int, int, str]] | None = None
    if args.min_support_counts is not None:
        final_classes, rolled = apply_min_support(
            set(build_included_label_rows(top108_uuids, porites_uuid, bucket_gf_uuid_lookup)),
            read_support_counts(args.min_support_counts),
            ba_lib,
            excluded_uuids,
            porites_uuid,
            args.min_points,
            args.min_groups,
        )
        top108_uuids = {ba for ba, _ in final_classes}
        # A merged bucket maps to '', so its annotations land in (Porites, '').
        bucket_gf_uuid_lookup = {
            name: gf if (porites_uuid, gf) in final_classes else ""
            for name, gf in bucket_gf_uuid_lookup.items()
        }
        coarsening_rules.append(
            f"min support (>= {args.min_points} points and >= {args.min_groups} groups,"
            f" counts from {args.min_support_counts.name})"
        )

        def class_name(key: ClassKey | None) -> str:
            if key is None:
                return ""
            ba_id, gf_id = key
            name = ba_lib.id_to_name(ba_id)
            return f"{name}::{gf_library.id_to_name(gf_id)}" if gf_id else name

        min_support_rows = [
            (class_name(r.class_key), r.points, r.n_groups, class_name(r.target)) for r in rolled
        ]
    changes = class_changes(
        {e.benthic_attribute_id for e in cn_mapping.mapping.values() if e.benthic_attribute_id}
        | uncoarsened_uuids,
        ba_lib,
        excluded_uuids,
        uncoarsened_uuids,
        top108_uuids,
    )
    species_gf_lookup = _load_species_gf_lookup(labels_df)

    rollup_rows, stats = build_rollup_rows(
        cn_mapping,
        ba_lib,
        gf_library,
        top108_uuids,
        porites_uuid,
        excluded_uuids,
        bucket_gf_uuid_lookup,
        species_gf_lookup,
    )
    included_rows = build_included_label_rows(top108_uuids, porites_uuid, bucket_gf_uuid_lookup)

    n_sources = copy_sources_csv(args.sources_csv, args.output_dir / "sources.csv")
    n_rollups = write_rollups_csv(rollup_rows, args.output_dir / "rollups.csv")
    n_included = write_included_labels_csv(included_rows, args.output_dir / "included_labels.csv")

    counts = {
        "n_sources": n_sources,
        "n_included": n_included,
        "n_rollups": n_rollups,
        "n_gf_collapse": stats["gf_collapse"],
        "n_cross_ba": stats["cross_ba"],
        "n_no_top_level": stats["no_top_level"],
        "n_unknown_uuid": stats["unknown_uuid"],
        "n_null_ba": stats["null_ba"],
        "n_excluded": stats["excluded"],
        "n_porites_genus_gf_collapse": stats["porites_genus_gf_collapse"],
        "n_porites_species_to_branching": stats["porites_species_to_branching"],
        "n_porites_species_to_massive": stats["porites_species_to_massive"],
        "n_porites_species_to_other": stats["porites_species_to_other"],
    }
    write_readme(
        out_path=args.output_dir / "README.md",
        audits=audits,
        counts=counts,
        unresolved_top108=unresolved,
        excluded_top108_seen=excluded_top108_seen,
        coarsening_rules=coarsening_rules,
        added_classes=sorted(ba_lib.id_to_name(b) for b in top108_uuids - uncoarsened_uuids),
        class_change_rows=sorted(
            (ba_lib.id_to_name(ba), ba_lib.id_to_name(old or ""), ba_lib.id_to_name(new or ""))
            for ba, old, new in changes
        ),
        min_support_rows=min_support_rows,
    )

    if not args.skip_validation:
        validate_outputs(args.output_dir)

    logger.info(
        "Wrote %d sources, %d included labels, %d rollups"
        " (%d cross-BA, %d GF collapses, %d Porites species buckets,"
        " %d excluded CN entries) to %s",
        n_sources,
        n_included,
        n_rollups,
        stats["cross_ba"],
        stats["gf_collapse"],
        stats["porites_species_to_branching"]
        + stats["porites_species_to_massive"]
        + stats["porites_species_to_other"],
        stats["excluded"],
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
