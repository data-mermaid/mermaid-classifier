"""Generate a self-contained training-config directory.

Produces sources.csv, included_labels.csv, rollups.csv, and an audit
README.md from the source-of-truth files in `drive_label_mappings/` and
`coralnet_best_sources - coralnet_best_sources.csv`.

Default invocation reproduces the `tiela77_top100_min1k` config:
- 77 sources where Tiela's `ToKeep == "Yes"`
- 109 top100 labels filtered to >= 1000 CoralNetAnnotations
- Porites kept with three GF buckets (Branching, Massive, blank)
- Species under top100 genera rolled up to the genus
- Non-Porites genus growth forms collapsed to no-growth-form
- Legacy "rolled up to <X>" rollups extracted from priority_notes

Run from the mermaid-classifier directory:

    uv run python scripts/generate_training_config.py

The same script is the documented way to regenerate with different
thresholds — see `--help`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent

# Defaults (relative to workspace root, resolved at runtime).
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / 'sagemaker' / 'configs' / 'tiela77_top100_min1k'
DEFAULT_SOURCES_CSV = WORKSPACE_ROOT / 'coralnet_best_sources - coralnet_best_sources.csv'
DEFAULT_LABELS_CSV = WORKSPACE_ROOT / 'drive_label_mappings' / (
    'coralnet_labels_mermaid_mapping_annotations - mapped_to_mermaid_attributes.csv'
)
DEFAULT_LABEL_MAPPING_CSV = WORKSPACE_ROOT / 'drive_label_mappings' / (
    'coralnet_labels_mermaid_mapping_annotations - label_mapping.csv'
)
DEFAULT_GROWTHFORMS_CSV = WORKSPACE_ROOT / 'sagemaker' / 'labels' / 'all_benthic_attributes' / 'growthforms.csv'
DEFAULT_S3_STATUS_CSV = WORKSPACE_ROOT / 'sagemaker' / 'sources' / 'coralnet_s3_actual_status.csv'

# Labels Emily explicitly told us not to use; defensive — top100 already excludes them.
DEFENSIVE_EXCLUDES = {
    'Dead coral', 'Bleached coral', 'Other invertebrates', 'Other', 'Unknown',
}

PORITES_NAME = 'Porites'
DEFAULT_PORITES_GF_BUCKETS = ('Branching', 'Massive')

logger = logging.getLogger(__name__)


# ---------- Helpers ----------

def _file_audit(path: Path) -> dict:
    """Return path, mtime, sha1 prefix for a file we read."""
    if not path.exists():
        return {'path': str(path), 'mtime': 'MISSING', 'sha1_prefix': 'MISSING'}
    stat = path.stat()
    h = hashlib.sha1()
    with path.open('rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return {
        'path': str(path.relative_to(WORKSPACE_ROOT))
                if path.is_relative_to(WORKSPACE_ROOT) else str(path),
        'mtime': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                  .strftime('%Y-%m-%d %H:%M:%S UTC'),
        'sha1_prefix': h.hexdigest()[:10],
    }


def _build_gf_lookup(growthforms_csv: Path) -> dict[str, str]:
    """name -> UUID for every growth form."""
    df = pd.read_csv(growthforms_csv)
    return dict(zip(df['name'], df['id']))


def _build_ba_lookup(labels_df: pd.DataFrame) -> dict[str, str]:
    """name -> UUID using the user-supplied mapped_to_mermaid_attributes file."""
    return dict(zip(labels_df['name'], labels_df['id']))


# ---------- Step A: sources ----------

def select_sources(
    sources_df: pd.DataFrame,
    tokeep_value: str,
    min_image_quality: int,
    min_coral_diversity: int,
    s3_status_csv: Path | None = None,
) -> pd.DataFrame:
    """Filter Tiela's scored sources to the kept set.

    The InS3 column in `sources_df` is unreliable (the CSV claims 1,312
    sources are present but only 555 actually have `annotations.csv` on S3
    as of 2026-04-28). When `s3_status_csv` is provided, sources are
    filtered against the freshly-probed `has_annotations_csv` column
    instead. Pass None to skip S3 filtering entirely (e.g. for tests).
    """
    required = {
        'Source ID', 'ToKeep', 'ImageQuality', 'CoralDiversity',
    }
    missing = required - set(sources_df.columns)
    if missing:
        raise ValueError(f"sources CSV missing columns: {sorted(missing)}")

    df = sources_df.copy()
    df['Source ID'] = df['Source ID'].astype(int)
    df = df[df['ToKeep'].astype(str) == tokeep_value]
    df = df[df['ImageQuality'].fillna(-1).astype(int) >= min_image_quality]
    df = df[df['CoralDiversity'].fillna(-1).astype(int) >= min_coral_diversity]
    pre_s3 = len(df)

    if s3_status_csv is not None:
        if not s3_status_csv.exists():
            raise FileNotFoundError(
                f"S3 status CSV {s3_status_csv} not found. Generate it first"
                f" with `scripts/probe_coralnet_s3_status.py`, or pass"
                f" --no-filter-by-s3-status to skip the S3 reality check."
            )
        status = pd.read_csv(s3_status_csv)
        status['id'] = status['id'].astype(int)
        present_ids = set(status[status['has_annotations_csv'] == True]['id'])  # noqa: E712
        df = df[df['Source ID'].isin(present_ids)]
        dropped = pre_s3 - len(df)
        if dropped:
            logger.info(
                "Dropped %d source(s) lacking annotations.csv on S3 per %s",
                dropped, s3_status_csv.name,
            )

    return df.sort_values('Source ID').reset_index(drop=True)


def write_sources_csv(sources_df: pd.DataFrame, out_path: Path) -> int:
    """Write the 'id'-only CSV the pipeline expects."""
    out_path.write_text(
        'id\n' + '\n'.join(str(int(s)) for s in sources_df['Source ID']) + '\n'
    )
    return len(sources_df)


# ---------- Step B: included labels ----------

def select_included_labels(
    labels_df: pd.DataFrame,
    top100_only: bool,
    min_annotations: int,
) -> pd.DataFrame:
    """Select labels per top100 + min annotation threshold + defensive excludes."""
    required = {'id', 'name', 'top100', 'CoralNetAnnotations', 'parent'}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels CSV missing columns: {sorted(missing)}")

    df = labels_df.copy()
    if top100_only:
        df = df[df['top100'] == 1.0]
    df = df[df['CoralNetAnnotations'].fillna(0) >= min_annotations]
    pre_defensive = len(df)
    df = df[~df['name'].isin(DEFENSIVE_EXCLUDES)]
    if len(df) < pre_defensive:
        logger.info(
            "Defensive filter removed %d row(s) named in DEFENSIVE_EXCLUDES",
            pre_defensive - len(df),
        )
    return df.sort_values('name').reset_index(drop=True)


def build_included_label_rows(
    included_df: pd.DataFrame,
    porites_gf_buckets: tuple[str, ...],
    gf_lookup: dict[str, str],
) -> list[tuple[str, str]]:
    """Emit (ba_id, gf_id) rows. Porites contributes one row per bucket + blank."""
    rows: list[tuple[str, str]] = []
    porites_id = None

    for _, row in included_df.iterrows():
        if row['name'] == PORITES_NAME:
            porites_id = row['id']
            continue
        rows.append((row['id'], ''))

    if porites_id is not None:
        # Branching, Massive (named buckets) plus the empty bucket.
        for gf_name in porites_gf_buckets:
            if gf_name not in gf_lookup:
                raise ValueError(
                    f"Porites GF bucket '{gf_name}' not in growthforms.csv"
                )
            rows.append((porites_id, gf_lookup[gf_name]))
        rows.append((porites_id, ''))

    rows.sort()
    return rows


def write_included_labels_csv(rows: list[tuple[str, str]], out_path: Path) -> int:
    with out_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ba_id', 'gf_id'])
        for ba_id, gf_id in rows:
            w.writerow([ba_id, gf_id])
    return len(rows)


# ---------- Step C: rollups ----------

ROLLED_UP_RE = re.compile(r'rolled up to (\w+)', re.IGNORECASE)


def build_rollup_rows(
    labels_df: pd.DataFrame,
    label_mapping_df: pd.DataFrame,
    included_label_names: set[str],
    gf_lookup: dict[str, str],
    ba_lookup: dict[str, str],
    porites_gf_buckets: tuple[str, ...],
) -> dict[str, list[tuple[str, str, str, str]]]:
    """Build the four categories of rollup rows.

    Returns a dict from category name -> list of
    (from_ba_id, from_gf_id, to_ba_id, to_gf_id).
    Categories: 'species_to_genus', 'nonporites_gf', 'porites_gf', 'legacy'.
    """
    species_rows: list[tuple[str, str, str, str]] = []
    nonporites_gf_rows: list[tuple[str, str, str, str]] = []
    porites_gf_rows: list[tuple[str, str, str, str]] = []
    legacy_rows: list[tuple[str, str, str, str]] = []

    # 1) Species/sub-taxa -> direct top100 parent.
    for _, row in labels_df.iterrows():
        if row['name'] in included_label_names:
            continue  # this label IS in the included set; no rollup needed
        parent_name = row['parent']
        if (
            isinstance(parent_name, str)
            and parent_name in included_label_names
            and parent_name in ba_lookup
        ):
            species_rows.append(
                (row['id'], '', ba_lookup[parent_name], '')
            )

    # 2) Non-Porites genus GF -> empty GF.
    porites_kept_gf_ids = {
        gf_lookup[name] for name in porites_gf_buckets if name in gf_lookup
    }
    seen_gf_pairs: set[tuple[str, str]] = set()
    for _, row in label_mapping_df.iterrows():
        ba_name = row['benthic attribute']
        gf_name = row.get('growth form')
        if not isinstance(gf_name, str) or not gf_name.strip():
            continue
        if ba_name not in included_label_names:
            continue  # rollup target wouldn't be in scope anyway
        if ba_name not in ba_lookup or gf_name not in gf_lookup:
            continue
        ba_id = ba_lookup[ba_name]
        gf_id = gf_lookup[gf_name]
        pair = (ba_id, gf_id)
        if pair in seen_gf_pairs:
            continue
        seen_gf_pairs.add(pair)

        if ba_name == PORITES_NAME:
            if gf_id in porites_kept_gf_ids:
                continue  # keep Branching/Massive as-is
            porites_gf_rows.append((ba_id, gf_id, ba_id, ''))
        else:
            nonporites_gf_rows.append((ba_id, gf_id, ba_id, ''))

    # 3) Legacy "rolled up to X" rollups from priority_notes.
    for _, row in labels_df.iterrows():
        notes = row.get('priority_notes')
        if not isinstance(notes, str):
            continue
        m = ROLLED_UP_RE.search(notes)
        if not m:
            continue
        target_name = m.group(1)
        if target_name not in ba_lookup:
            logger.warning(
                "priority_notes for %r references unknown rollup target %r;"
                " skipping legacy rollup",
                row['name'], target_name,
            )
            continue
        legacy_rows.append((row['id'], '', ba_lookup[target_name], ''))

    return {
        'species_to_genus': sorted(species_rows),
        'nonporites_gf': sorted(nonporites_gf_rows),
        'porites_gf': sorted(porites_gf_rows),
        'legacy': sorted(legacy_rows),
    }


def write_rollups_csv(
    categorized_rows: dict[str, list[tuple[str, str, str, str]]],
    out_path: Path,
) -> int:
    flat = []
    for rows in categorized_rows.values():
        flat.extend(rows)
    flat.sort()
    with out_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['from_ba_id', 'from_gf_id', 'to_ba_id', 'to_gf_id'])
        for r in flat:
            w.writerow(r)
    return len(flat)


# ---------- Step D: README ----------

def write_readme(
    out_path: Path,
    args: argparse.Namespace,
    audits: dict[str, dict],
    counts: dict,
    unresolved_legacy: list[str],
) -> None:
    cli = ' '.join(sys.argv)
    legacy_caveat = (
        f"\n  - {len(unresolved_legacy)} legacy 'rolled up to X' rollups had"
        f" targets not in the labels CSV; logged as warnings and dropped."
        if unresolved_legacy else ''
    )
    body = f"""# Training Config: {out_path.parent.name}

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by
`scripts/generate_training_config.py`. Regenerate with:

```
{cli}
```

## Parameters

- `--tiela-tokeep`           = `{args.tiela_tokeep}`
- `--min-image-quality`      = `{args.min_image_quality}`
- `--min-coral-diversity`    = `{args.min_coral_diversity}`
- `--top100-only`            = `{args.top100_only}`
- `--min-annotations`        = `{args.min_annotations}`
- `--porites-gf-buckets`     = `{', '.join(args.porites_gf_buckets)}`

## Source-of-truth inputs

| File | Modified | sha1[:10] |
| --- | --- | --- |
| `{audits['sources']['path']}` | {audits['sources']['mtime']} | `{audits['sources']['sha1_prefix']}` |
| `{audits['labels']['path']}` | {audits['labels']['mtime']} | `{audits['labels']['sha1_prefix']}` |
| `{audits['label_mapping']['path']}` | {audits['label_mapping']['mtime']} | `{audits['label_mapping']['sha1_prefix']}` |
| `{audits['growthforms']['path']}` | {audits['growthforms']['mtime']} | `{audits['growthforms']['sha1_prefix']}` |

## Outputs

- `sources.csv` — **{counts['n_sources']}** CoralNet source IDs.
- `included_labels.csv` — **{counts['n_included_rows']}** `(ba_id, gf_id)` rows
  ({counts['n_included_labels']} distinct labels, of which Porites contributes
  {counts['n_porites_rows']} rows for the kept growth forms + empty bucket).
- `rollups.csv` — **{counts['n_rollups']}** rows total:
    - species → top100 genus: {counts['n_rollup_species']}
    - non-Porites genus GF → empty GF: {counts['n_rollup_nonporites_gf']}
    - Porites GF (other than kept buckets) → empty GF: {counts['n_rollup_porites_gf']}
    - legacy `priority_notes` rollups (e.g. → Mussidae): {counts['n_rollup_legacy']}

## Limitations and caveats

- The user's slack message says 19 of the 109 top100 labels have <1000
  annotations; the source CSV says **28**. Per Q3 in planning, the file is
  authoritative; the discrepancy is likely a stale slack count.
- Legacy `rolled up to <X>` rollups are preserved verbatim per Q5. If the
  rollup *target* X is not in `included_labels.csv`, the rolled-up annotations
  will still be filtered out at training time — the rollup row exists for
  documentation only.{legacy_caveat}
- Annotation totals shown for selecting/excluding labels come from
  `mapped_to_mermaid_attributes.csv` and are *across all CoralNet sources*;
  the per-source-subset totals will be smaller. Per Q3 we did not recompute.
- After training, inspect the MLflow `unmapped_labels.csv` artifact for the
  authoritative list of CoralNet labels in the chosen sources that did not
  map into the trained label space.
"""
    out_path.write_text(body)


# ---------- Validation ----------

def validate_outputs(out_dir: Path) -> None:
    """Round-trip the produced CSVs through the pipeline's CsvSpec subclasses."""
    # Imported here so that script tests don't pay the import cost when not
    # validating, and so we can run without a full pyspacer install in
    # smoke tests.
    from mermaid_classifier.pyspacer.train import (
        CNSourceFilter, LabelFilter, LabelRollupSpec,
    )

    with (out_dir / 'sources.csv').open() as f:
        CNSourceFilter(f)
    with (out_dir / 'included_labels.csv').open() as f:
        LabelFilter(f, inclusion=True)
    with (out_dir / 'rollups.csv').open() as f:
        LabelRollupSpec(f)


# ---------- Main ----------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split('\n', 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--sources-csv', type=Path, default=DEFAULT_SOURCES_CSV)
    p.add_argument('--labels-csv', type=Path, default=DEFAULT_LABELS_CSV)
    p.add_argument('--label-mapping-csv', type=Path, default=DEFAULT_LABEL_MAPPING_CSV)
    p.add_argument('--growthforms-csv', type=Path, default=DEFAULT_GROWTHFORMS_CSV)
    p.add_argument('--s3-status-csv', type=Path, default=DEFAULT_S3_STATUS_CSV,
                   help='CSV with id,has_annotations_csv from a fresh S3 probe '
                        '(see scripts/probe_coralnet_s3_status.py). '
                        'Sources without an actual annotations.csv on S3 are '
                        'dropped from the output, regardless of the InS3 flag '
                        'in the source-of-truth CSV.')
    p.add_argument('--no-filter-by-s3-status', action='store_true',
                   help='Skip the S3 reality check and trust the InS3 column. '
                        'Only use for tests or when the S3 status CSV is '
                        'genuinely unavailable.')
    p.add_argument('--tiela-tokeep', default='Yes',
                   help='String value of the ToKeep column to keep')
    p.add_argument('--min-image-quality', type=int, default=0)
    p.add_argument('--min-coral-diversity', type=int, default=0)
    p.add_argument('--top100-only', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--min-annotations', type=int, default=1000)
    p.add_argument('--porites-gf-buckets', nargs='+',
                   default=list(DEFAULT_PORITES_GF_BUCKETS))
    p.add_argument('--skip-validation', action='store_true',
                   help='Skip round-tripping outputs through pipeline schemas '
                        '(useful for tests)')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Audit inputs up front.
    audits = {
        'sources': _file_audit(args.sources_csv),
        'labels': _file_audit(args.labels_csv),
        'label_mapping': _file_audit(args.label_mapping_csv),
        'growthforms': _file_audit(args.growthforms_csv),
    }
    for k, a in audits.items():
        if a['mtime'] == 'MISSING':
            raise FileNotFoundError(f"input {k}: {a['path']} not found")

    # Load.
    sources_df = pd.read_csv(args.sources_csv)
    labels_df = pd.read_csv(args.labels_csv)
    label_mapping_df = pd.read_csv(args.label_mapping_csv)
    gf_lookup = _build_gf_lookup(args.growthforms_csv)
    ba_lookup = _build_ba_lookup(labels_df)

    # Step A: sources.
    sources_kept = select_sources(
        sources_df,
        tokeep_value=args.tiela_tokeep,
        min_image_quality=args.min_image_quality,
        min_coral_diversity=args.min_coral_diversity,
        s3_status_csv=None if args.no_filter_by_s3_status else args.s3_status_csv,
    )
    n_sources = write_sources_csv(sources_kept, args.output_dir / 'sources.csv')

    # Step B: included labels.
    included_df = select_included_labels(
        labels_df,
        top100_only=args.top100_only,
        min_annotations=args.min_annotations,
    )
    porites_gf_buckets = tuple(args.porites_gf_buckets)
    included_rows = build_included_label_rows(
        included_df,
        porites_gf_buckets=porites_gf_buckets,
        gf_lookup=gf_lookup,
    )
    n_included_rows = write_included_labels_csv(
        included_rows, args.output_dir / 'included_labels.csv')

    # Step C: rollups.
    included_label_names = set(included_df['name'])
    categorized = build_rollup_rows(
        labels_df=labels_df,
        label_mapping_df=label_mapping_df,
        included_label_names=included_label_names,
        gf_lookup=gf_lookup,
        ba_lookup=ba_lookup,
        porites_gf_buckets=porites_gf_buckets,
    )
    n_rollups = write_rollups_csv(categorized, args.output_dir / 'rollups.csv')

    # Step D: README.
    n_porites_rows = sum(1 for ba_id, _ in included_rows
                          if ba_id == ba_lookup.get(PORITES_NAME))
    counts = dict(
        n_sources=n_sources,
        n_included_rows=n_included_rows,
        n_included_labels=len(included_df),
        n_porites_rows=n_porites_rows,
        n_rollups=n_rollups,
        n_rollup_species=len(categorized['species_to_genus']),
        n_rollup_nonporites_gf=len(categorized['nonporites_gf']),
        n_rollup_porites_gf=len(categorized['porites_gf']),
        n_rollup_legacy=len(categorized['legacy']),
    )
    write_readme(
        out_path=args.output_dir / 'README.md',
        args=args,
        audits=audits,
        counts=counts,
        unresolved_legacy=[],
    )

    # Step E: validate.
    if not args.skip_validation:
        validate_outputs(args.output_dir)

    logger.info(
        "Wrote %d sources, %d included-label rows (%d distinct labels), "
        "%d rollups to %s",
        n_sources, n_included_rows, len(included_df), n_rollups,
        args.output_dir,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
