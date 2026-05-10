"""
Autoresearch harness for mermaid-classifier.

Autonomous overnight experiment loop inspired by Karpathy's autoresearch.
Calls the Claude API in a loop, proposes modifications to experiment/
files, runs training, and keeps or reverts based on balanced_accuracy.

Usage:
    uv run python autoresearch/autoresearch.py [options]

Options:
    --max-hours N        Total time budget in hours (default: 12)
    --max-experiments N  Maximum number of experiments (default: 100)
    --timeout N          Training timeout in seconds (default: 1800)
    --model MODEL        Claude model to use (default: claude-sonnet-4-6)
    --max-consecutive-failures N  Stop after N consecutive failures (default: 10)
    --dry-run            Show what would be done without running training
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = AUTORESEARCH_DIR / "experiment"
PROGRAM_MD = AUTORESEARCH_DIR / "program.md"
RESULTS_TSV = AUTORESEARCH_DIR / "results.tsv"
HASHES_FILE = AUTORESEARCH_DIR / "baseline_hashes.json"
TELEMETRY_DIR = AUTORESEARCH_DIR / "telemetry"
ANALYSES_DIR = AUTORESEARCH_DIR / "analyses"

# Files the agent is allowed to modify.
MODIFIABLE_FILES = [
    EXPERIMENT_DIR / "train_experiment.py",
    EXPERIMENT_DIR / "classifier.py",
    EXPERIMENT_DIR / "trainer.py",
    EXPERIMENT_DIR / "strategies.py",
]

# Directories/files to hash-protect.
FROZEN_DIRS = [
    ROOT_DIR / "mermaid_classifier",
]
FROZEN_FILES = [
    AUTORESEARCH_DIR / "autoresearch.py",
    # Config CSVs are referenced by relative path from the experiment
    # file; resolve them relative to ROOT_DIR.
]

# ── Hash verification ──────────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_frozen_hashes() -> dict[str, str]:
    """Compute SHA256 hashes of all frozen files."""
    hashes = {}
    for d in FROZEN_DIRS:
        for p in sorted(d.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and "__pycache__" not in str(p):
                hashes[str(p.relative_to(ROOT_DIR))] = _sha256(p)
    for f in FROZEN_FILES:
        if f.exists():
            hashes[str(f.relative_to(ROOT_DIR))] = _sha256(f)
    return hashes


def save_hashes(hashes: dict[str, str]) -> None:
    with open(HASHES_FILE, "w") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)


def load_hashes() -> dict[str, str]:
    with open(HASHES_FILE) as f:
        return json.load(f)


def verify_hashes() -> tuple[bool, list[str]]:
    """Verify frozen files haven't changed. Returns (ok, changed_files)."""
    baseline = load_hashes()
    current = compute_frozen_hashes()
    changed = []
    for path, expected_hash in baseline.items():
        actual = current.get(path)
        if actual != expected_hash:
            changed.append(path)
    # Check for new files not in baseline.
    for path in current:
        if path not in baseline:
            changed.append(f"{path} (new file)")
    return len(changed) == 0, changed


# ── Results tracking ───────────────────────────────────────────────

HEADLINE_METRIC_KEYS = (
    "balanced_accuracy",
    "mcc",
    "ece",
    "top_5_accuracy",
    "cross_branch_error_rate",
    "within_branch_error_rate",
    "precision_macro",
    "recall_macro",
)

RESULTS_FIELDS = [
    "id", "timestamp", "hypothesis",
    *HEADLINE_METRIC_KEYS,
    "best_so_far", "status", "duration_s", "commit_sha",
    "analysis_excerpt", "error",
]


def init_results_tsv() -> None:
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, delimiter="\t")
            writer.writeheader()
        return
    _migrate_results_tsv()


def _migrate_results_tsv() -> None:
    """Bring an existing results.tsv up to the current RESULTS_FIELDS.

    Idempotent: if the header already matches, no-op. Otherwise rewrite
    with the new header, padding existing rows with empty strings for
    new columns.
    """
    with open(RESULTS_TSV, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        rows = list(reader)
    if header == RESULTS_FIELDS:
        return
    name_to_idx = {name: i for i, name in enumerate(header or [])}
    with open(RESULTS_TSV, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(RESULTS_FIELDS)
        for row in rows:
            new_row = []
            for field in RESULTS_FIELDS:
                idx = name_to_idx.get(field)
                if idx is not None and idx < len(row):
                    new_row.append(row[idx])
                else:
                    new_row.append("")
            writer.writerow(new_row)


def append_result(row: dict) -> None:
    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, delimiter="\t")
        writer.writerow(row)


def read_results_tsv() -> str:
    if not RESULTS_TSV.exists():
        return ""
    return RESULTS_TSV.read_text()


def next_experiment_id() -> int:
    if not RESULTS_TSV.exists():
        return 1
    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        ids = [int(row["id"]) for row in reader if row.get("id")]
    return max(ids, default=0) + 1


# ── Git operations ─────────────────────────────────────────────────


def git_commit(message: str) -> str:
    """Commit all changes in experiment/ and return the commit SHA."""
    subprocess.run(
        ["git", "add"] + [str(f) for f in MODIFIABLE_FILES],
        cwd=ROOT_DIR, check=True, capture_output=True,
    )
    # Stage results.tsv and the per-experiment telemetry / analysis
    # files alongside the experiment code so they roll back together
    # on REVERTED runs.
    extra_paths = [RESULTS_TSV, TELEMETRY_DIR, ANALYSES_DIR]
    for p in extra_paths:
        if p.exists():
            subprocess.run(
                ["git", "add", str(p)],
                cwd=ROOT_DIR, check=True, capture_output=True,
            )
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=ROOT_DIR, check=True, capture_output=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()[:7]


def git_reset_hard() -> None:
    """Reset to the previous commit, discarding the failed experiment."""
    subprocess.run(
        ["git", "reset", "--hard", "HEAD~1"],
        cwd=ROOT_DIR, check=True, capture_output=True,
    )


def git_recent_diffs(n: int = 5) -> str:
    """Get diffs of the last N commits."""
    result = subprocess.run(
        ["git", "log", f"-{n}", "--patch", "--", "autoresearch/experiment/"],
        cwd=ROOT_DIR, capture_output=True, text=True,
    )
    return result.stdout[:8000]  # Truncate to keep prompt reasonable.


# ── Training subprocess ────────────────────────────────────────────


def run_training(timeout_seconds: int) -> tuple[int, str, str, float]:
    """Run training as a subprocess. Returns (returncode, stdout, stderr, duration)."""
    t0 = time.time()
    try:
        result = subprocess.run(
            ["uv", "run", "python", str(EXPERIMENT_DIR / "train_experiment.py")],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        duration = time.time() - t0
        return result.returncode, result.stdout, result.stderr, duration
    except subprocess.TimeoutExpired:
        duration = time.time() - t0
        return -1, "", f"TIMEOUT: training exceeded {timeout_seconds}s limit", duration


# ── MLflow telemetry extraction ───────────────────────────────────

# Imported lazily inside helpers so the module loads even when mlflow
# is unavailable (e.g. running unit tests without the pyspacer extras).


def fetch_run_telemetry(experiment_name: str = "autoresearch"):
    """Pull a full :class:`telemetry.RunTelemetry` for the latest run.

    Returns ``None`` if extraction fails. Errors are logged but not
    raised, so the loop can fall through to a CRASH status row.
    """
    try:
        from telemetry import extract_run_telemetry
        return extract_run_telemetry(experiment_name=experiment_name)
    except Exception as e:
        logger.error(f"Failed to extract MLflow telemetry: {e}")
        return None


def write_telemetry_file(experiment_id: int, markdown: str) -> Path:
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    path = TELEMETRY_DIR / f"{experiment_id}.md"
    path.write_text(markdown)
    return path


def write_analysis_file(experiment_id: int, analysis: str) -> Path:
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSES_DIR / f"{experiment_id}.md"
    path.write_text(analysis)
    return path


def _load_recent_artifacts(directory: Path, current_id: int, n: int = 2) -> list[tuple[int, str]]:
    """Load the most recent ``n`` markdown files under ``directory``.

    Returns ``(experiment_id, body)`` tuples, ordered most-recent first.
    Files for ``current_id`` are excluded so the prompt-builder doesn't
    echo the in-flight experiment back.
    """
    if not directory.exists():
        return []
    ids: list[int] = []
    for p in directory.glob("*.md"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            continue
    ids = [i for i in sorted(ids, reverse=True) if i != current_id][:n]
    return [(i, (directory / f"{i}.md").read_text()) for i in ids]


# ── Claude CLI ─────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are an ML research agent running autonomous experiments on a coral reef classifier.

{program_md}

## Response Format

Your response must be a JSON object with these fields, in order:

- "analysis": A multi-paragraph walk-through of the most recent experiment's telemetry. Cite specific numbers from the **Last 2 Experiments — Full Telemetry** block: per-class precision/recall, top confusion pairs, calibration bin gaps, training-loss vs. val-loss trajectory, top error-attribution LCAs, per-source min/max accuracy. Reference at least 3 specific telemetry observations. Do not propose changes here — only diagnose what the last run reveals.
- "hypothesis": A one-sentence description of what you're testing next and why, derived from the analysis above.
- "file_changes": A list of objects, each with:
  - "filename": One of "train_experiment.py", "classifier.py", "trainer.py", "strategies.py"
  - "content": The complete new content of the file.

Only include files you are actually changing. Unchanged files should be omitted."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "hypothesis": {"type": "string"},
        "file_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    "required": ["analysis", "hypothesis", "file_changes"],
}


def build_user_prompt(
    experiment_files: dict[str, str],
    results_tsv: str,
    recent_diffs: str,
    last_error: str | None = None,
    prior_telemetry: list[tuple[int, str]] | None = None,
    prior_analyses: list[tuple[int, str]] | None = None,
) -> str:
    parts = ["## Current Experiment Files\n"]
    for name, content in sorted(experiment_files.items()):
        parts.append(f"### {name}\n```python\n{content}\n```\n")
    parts.append(f"## Headline Metrics History\n```\n{results_tsv}\n```\n")

    if prior_telemetry:
        parts.append("## Last 2 Experiments — Full Telemetry\n")
        for exp_id, body in prior_telemetry:
            parts.append(f"### Experiment {exp_id}\n{body}\n")

    if prior_analyses:
        parts.append("## Last 2 Experiments — Analysis\n")
        for exp_id, body in prior_analyses:
            parts.append(f"### Experiment {exp_id}\n{body}\n")

    if recent_diffs:
        parts.append(f"## Recent Diffs (last 5 kept experiments)\n```\n{recent_diffs}\n```\n")
    if last_error:
        parts.append(f"## Last Experiment Error\n```\n{last_error[:3000]}\n```\n")
    return "\n".join(parts)


def read_experiment_files() -> dict[str, str]:
    files = {}
    for f in MODIFIABLE_FILES:
        if f.exists():
            files[f.name] = f.read_text()
    return files


def call_claude(
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    """Call Claude via the claude CLI and return structured JSON response."""
    result = subprocess.run(
        [
            "claude", "-p", user_prompt,
            "--output-format", "json",
            "--json-schema", json.dumps(RESPONSE_SCHEMA),
            "--system-prompt", system_prompt,
            "--model", model,
        ],
        capture_output=True,
        text=True,
    )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"claude CLI returned exit {result.returncode} with non-JSON stdout. "
            f"stdout={result.stdout[:500]!r} stderr={result.stderr[:500]!r}"
        ) from e
    if response.get("is_error") or "structured_output" not in response:
        raise RuntimeError(
            f"claude CLI error (exit {result.returncode}): "
            f"{response.get('result', '<no result field>')}"
        )
    return response["structured_output"]


def apply_file_changes(changes: list[dict]) -> list[str]:
    """Apply file changes from Claude's response. Returns list of changed filenames."""
    changed = []
    valid_filenames = {f.name for f in MODIFIABLE_FILES}
    for change in changes:
        filename = change["filename"]
        if filename not in valid_filenames:
            logger.warning(f"Ignoring change to non-modifiable file: {filename}")
            continue
        filepath = EXPERIMENT_DIR / filename
        filepath.write_text(change["content"])
        changed.append(filename)
    return changed


# ── Main loop ──────────────────────────────────────────────────────


def _empty_headline() -> dict[str, str]:
    return {k: "" for k in HEADLINE_METRIC_KEYS}


def _format_headline(headline: dict[str, float]) -> dict[str, str]:
    return {k: f"{headline[k]:.6f}" if k in headline else "" for k in HEADLINE_METRIC_KEYS}


def _analysis_excerpt(analysis: str, limit: int = 120) -> str:
    """First ``limit`` chars of analysis with whitespace collapsed for TSV."""
    cleaned = " ".join(analysis.split())
    return cleaned[:limit]


def run_autoresearch(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info(f"Using model: {args.model}")

    # Initialize results tracking.
    init_results_tsv()

    # Compute and save frozen file hashes.
    logger.info("Computing frozen file hashes...")
    hashes = compute_frozen_hashes()
    save_hashes(hashes)
    logger.info(f"Hashed {len(hashes)} frozen files")

    # Build system prompt.
    program_md = PROGRAM_MD.read_text()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(program_md=program_md)

    # Run baseline.
    logger.info("=" * 60)
    logger.info("Running baseline experiment...")
    logger.info("=" * 60)

    returncode, stdout, stderr, duration = run_training(args.timeout)
    if returncode != 0:
        logger.error(f"Baseline training failed:\n{stderr[-2000:]}")
        sys.exit(1)

    baseline_telemetry = fetch_run_telemetry()
    if baseline_telemetry is None or "balanced_accuracy" not in baseline_telemetry.headline:
        logger.error("Failed to extract baseline telemetry from MLflow")
        sys.exit(1)

    baseline_metric = baseline_telemetry.headline["balanced_accuracy"]
    write_telemetry_file(1, baseline_telemetry.full_markdown)

    best_so_far = baseline_metric
    sha = git_commit(f"autoresearch baseline: balanced_accuracy={baseline_metric:.4f}")

    baseline_row = {
        "id": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": "baseline",
        **_format_headline(baseline_telemetry.headline),
        "best_so_far": f"{baseline_metric:.6f}",
        "status": "KEPT",
        "duration_s": f"{duration:.0f}",
        "commit_sha": sha,
        "analysis_excerpt": "",
        "error": "",
    }
    append_result(baseline_row)

    # Update program.md with baseline metric.
    program_md = program_md.replace("{BASELINE_METRIC}", f"{baseline_metric:.4f}")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(program_md=program_md)

    logger.info(f"Baseline balanced_accuracy: {baseline_metric:.4f}")
    logger.info(f"Starting experiment loop (max {args.max_hours}h, {args.max_experiments} experiments)")

    # Main experiment loop.
    start_time = time.time()
    consecutive_failures = 0
    last_error = None
    experiment_id = 2

    while True:
        # Check stopping conditions.
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours >= args.max_hours:
            logger.info(f"Time budget exhausted ({args.max_hours}h)")
            break
        if experiment_id > args.max_experiments:
            logger.info(f"Experiment limit reached ({args.max_experiments})")
            break
        if consecutive_failures >= args.max_consecutive_failures:
            logger.info(
                f"Stopped: {consecutive_failures} consecutive failures")
            break

        logger.info("=" * 60)
        logger.info(f"Experiment {experiment_id}")
        logger.info("=" * 60)

        # 1. Build prompt.
        experiment_files = read_experiment_files()
        results_tsv = read_results_tsv()
        recent_diffs = git_recent_diffs(5)
        prior_telemetry = _load_recent_artifacts(TELEMETRY_DIR, experiment_id)
        prior_analyses = _load_recent_artifacts(ANALYSES_DIR, experiment_id)

        user_prompt = build_user_prompt(
            experiment_files,
            results_tsv,
            recent_diffs,
            last_error,
            prior_telemetry=prior_telemetry,
            prior_analyses=prior_analyses,
        )

        # 2. Call Claude.
        logger.info("Calling Claude API...")
        try:
            response = call_claude(args.model, system_prompt, user_prompt)
            analysis = response.get("analysis", "")
            hypothesis = response.get("hypothesis", "no hypothesis provided")
            file_changes = response.get("file_changes", [])
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            consecutive_failures += 1
            last_error = str(e)
            experiment_id += 1
            continue

        logger.info(f"Hypothesis: {hypothesis}")

        if not file_changes:
            logger.warning("No file changes proposed, skipping")
            consecutive_failures += 1
            last_error = "No file changes proposed"
            experiment_id += 1
            continue

        # Persist analysis text immediately so the next iteration can
        # cite it even if training crashes.
        if analysis:
            write_analysis_file(experiment_id, analysis)

        # 3. Apply changes and commit.
        changed = apply_file_changes(file_changes)
        logger.info(f"Changed files: {', '.join(changed)}")

        if args.dry_run:
            logger.info("[DRY RUN] Would run training now")
            experiment_id += 1
            continue

        commit_sha = git_commit(
            f"autoresearch experiment {experiment_id}: {hypothesis}")

        # 4. Verify frozen files.
        ok, changed_frozen = verify_hashes()
        if not ok:
            logger.error(
                f"ABORT: frozen files modified: {changed_frozen}")
            git_reset_hard()
            append_result({
                "id": experiment_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "hypothesis": hypothesis,
                **_empty_headline(),
                "best_so_far": f"{best_so_far:.6f}",
                "status": "ABORT",
                "duration_s": "0",
                "commit_sha": commit_sha,
                "analysis_excerpt": _analysis_excerpt(analysis),
                "error": f"frozen files modified: {changed_frozen}",
            })
            consecutive_failures += 1
            last_error = f"Frozen files were modified: {changed_frozen}"
            experiment_id += 1
            continue

        # 5. Run training.
        logger.info("Running training...")
        returncode, stdout, stderr, duration = run_training(args.timeout)

        if returncode != 0:
            error_msg = stderr[-2000:] if stderr else "unknown error"
            if returncode == -1:
                status = "TIMEOUT"
            else:
                status = "CRASH"
            logger.warning(f"Training {status}: {error_msg[:200]}")
            git_reset_hard()
            append_result({
                "id": experiment_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "hypothesis": hypothesis,
                **_empty_headline(),
                "best_so_far": f"{best_so_far:.6f}",
                "status": status,
                "duration_s": f"{duration:.0f}",
                "commit_sha": commit_sha,
                "analysis_excerpt": _analysis_excerpt(analysis),
                "error": error_msg[:500],
            })
            consecutive_failures += 1
            last_error = error_msg
            experiment_id += 1
            continue

        # 6. Extract telemetry.
        telemetry = fetch_run_telemetry()
        if telemetry is None or "balanced_accuracy" not in telemetry.headline:
            logger.warning("Failed to extract telemetry, treating as crash")
            git_reset_hard()
            append_result({
                "id": experiment_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "hypothesis": hypothesis,
                **_empty_headline(),
                "best_so_far": f"{best_so_far:.6f}",
                "status": "CRASH",
                "duration_s": f"{duration:.0f}",
                "commit_sha": commit_sha,
                "analysis_excerpt": _analysis_excerpt(analysis),
                "error": "Failed to extract telemetry from MLflow",
            })
            consecutive_failures += 1
            last_error = "Failed to extract telemetry from MLflow"
            experiment_id += 1
            continue

        write_telemetry_file(experiment_id, telemetry.full_markdown)
        metric = telemetry.headline["balanced_accuracy"]

        # 7. Keep or revert.
        prior_best = best_so_far
        if metric > best_so_far:
            best_so_far = metric
            status = "KEPT"
            consecutive_failures = 0
            last_error = None
            logger.info(
                f"KEPT: balanced_accuracy={metric:.4f}"
                f" (improvement from {prior_best:.4f})")
        else:
            status = "REVERTED"
            consecutive_failures += 1
            last_error = None
            logger.info(
                f"REVERTED: balanced_accuracy={metric:.4f}"
                f" (no improvement over {best_so_far:.4f})")
            git_reset_hard()

        append_result({
            "id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hypothesis": hypothesis,
            **_format_headline(telemetry.headline),
            "best_so_far": f"{best_so_far:.6f}",
            "status": status,
            "duration_s": f"{duration:.0f}",
            "commit_sha": commit_sha,
            "analysis_excerpt": _analysis_excerpt(analysis),
            "error": "",
        })

        experiment_id += 1

    # Summary.
    logger.info("=" * 60)
    logger.info("AUTORESEARCH COMPLETE")
    logger.info(f"Experiments run: {experiment_id - 1}")
    logger.info(f"Best balanced_accuracy: {best_so_far:.4f}")
    logger.info(f"Results: {RESULTS_TSV}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch harness for mermaid-classifier")
    parser.add_argument(
        "--max-hours", type=float, default=12,
        help="Total time budget in hours (default: 12)")
    parser.add_argument(
        "--max-experiments", type=int, default=100,
        help="Maximum number of experiments (default: 100)")
    parser.add_argument(
        "--timeout", type=int, default=4500,
        help="Training timeout in seconds (default: 4500)")
    parser.add_argument(
        "--model", type=str, default="claude-opus-4-7",
        help="Claude model to use (default: claude-opus-4-7)")
    parser.add_argument(
        "--max-consecutive-failures", type=int, default=5,
        help="Stop after N consecutive failures (default: 5)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without running training")
    args = parser.parse_args()

    # Handle Ctrl+C gracefully.
    def signal_handler(sig, frame):
        logger.info("\nInterrupted. Shutting down gracefully...")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    run_autoresearch(args)


if __name__ == "__main__":
    main()
