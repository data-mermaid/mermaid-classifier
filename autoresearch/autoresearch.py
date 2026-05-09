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

import anthropic

logger = logging.getLogger(__name__)

CLAUDE_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"


def get_anthropic_client() -> anthropic.Anthropic:
    """Create an Anthropic client, using OAuth credentials from Claude Code
    if ANTHROPIC_API_KEY is not set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        logger.info("Using ANTHROPIC_API_KEY from environment")
        return anthropic.Anthropic(api_key=api_key)

    if CLAUDE_CREDENTIALS_PATH.exists():
        with open(CLAUDE_CREDENTIALS_PATH) as f:
            creds = json.load(f)
        oauth = creds.get("claudeAiOauth", {})
        token = oauth.get("accessToken")
        if token:
            logger.info("Using OAuth token from Claude Code credentials")
            return anthropic.Anthropic(api_key=token)

    raise RuntimeError(
        "No Anthropic credentials found. Either set ANTHROPIC_API_KEY "
        "or authenticate with Claude Code (which stores OAuth tokens "
        f"in {CLAUDE_CREDENTIALS_PATH})."
    )


# ── Paths ──────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = AUTORESEARCH_DIR / "experiment"
PROGRAM_MD = AUTORESEARCH_DIR / "program.md"
RESULTS_TSV = AUTORESEARCH_DIR / "results.tsv"
HASHES_FILE = AUTORESEARCH_DIR / "baseline_hashes.json"

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

RESULTS_FIELDS = [
    "id", "timestamp", "hypothesis", "balanced_accuracy",
    "best_so_far", "status", "duration_s", "commit_sha", "error",
]


def init_results_tsv() -> None:
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, delimiter="\t")
            writer.writeheader()


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
    # Also stage results.tsv if it exists.
    if RESULTS_TSV.exists():
        subprocess.run(
            ["git", "add", str(RESULTS_TSV)],
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


# ── MLflow metric extraction ──────────────────────────────────────


def query_mlflow_balanced_accuracy(experiment_name: str = "autoresearch") -> float | None:
    """Query the most recent MLflow run for balanced_accuracy."""
    try:
        import mlflow

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"MLflow experiment '{experiment_name}' not found")
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            logger.warning("No MLflow runs found")
            return None

        metric_col = "metrics.balanced_accuracy"
        if metric_col not in runs.columns:
            logger.warning(f"'{metric_col}' not in MLflow run columns")
            return None

        return float(runs.iloc[0][metric_col])
    except Exception as e:
        logger.error(f"Failed to query MLflow: {e}")
        return None


# ── Claude API ─────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are an ML research agent running autonomous experiments on a coral reef classifier.

{program_md}

## Response Format

Respond with a JSON object containing:
- "hypothesis": A one-sentence description of what you're testing and why.
- "file_changes": A list of objects, each with:
  - "filename": One of "train_experiment.py", "classifier.py", "trainer.py", "strategies.py"
  - "content": The complete new content of the file.

Only include files you are actually changing. Unchanged files should be omitted.

Example:
```json
{{
  "hypothesis": "Add dropout of 0.2 between hidden layers to reduce overfitting",
  "file_changes": [
    {{
      "filename": "classifier.py",
      "content": "... full file content ..."
    }}
  ]
}}
```

IMPORTANT: Return ONLY the JSON object. No markdown code fences, no explanation outside the JSON."""


def build_user_prompt(
    experiment_files: dict[str, str],
    results_tsv: str,
    recent_diffs: str,
    last_error: str | None = None,
) -> str:
    parts = ["## Current Experiment Files\n"]
    for name, content in sorted(experiment_files.items()):
        parts.append(f"### {name}\n```python\n{content}\n```\n")
    parts.append(f"## Experiment History\n```\n{results_tsv}\n```\n")
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
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    """Call Claude API and parse response as JSON."""
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        temperature=1.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = response.content[0].text.strip()

    # Strip markdown code fences if present.
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return json.loads(text)


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


def run_autoresearch(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Initialize Claude client.
    client = get_anthropic_client()
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

    baseline_metric = query_mlflow_balanced_accuracy()
    if baseline_metric is None:
        logger.error("Failed to extract baseline balanced_accuracy from MLflow")
        sys.exit(1)

    best_so_far = baseline_metric
    sha = git_commit(f"autoresearch baseline: balanced_accuracy={baseline_metric:.4f}")

    append_result({
        "id": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypothesis": "baseline",
        "balanced_accuracy": f"{baseline_metric:.6f}",
        "best_so_far": f"{baseline_metric:.6f}",
        "status": "KEPT",
        "duration_s": f"{duration:.0f}",
        "commit_sha": sha,
        "error": "",
    })

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

        user_prompt = build_user_prompt(
            experiment_files, results_tsv, recent_diffs, last_error)

        # 2. Call Claude.
        logger.info("Calling Claude API...")
        try:
            response = call_claude(client, args.model, system_prompt, user_prompt)
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
                "balanced_accuracy": "",
                "best_so_far": f"{best_so_far:.6f}",
                "status": "ABORT",
                "duration_s": "0",
                "commit_sha": commit_sha,
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
                "balanced_accuracy": "",
                "best_so_far": f"{best_so_far:.6f}",
                "status": status,
                "duration_s": f"{duration:.0f}",
                "commit_sha": commit_sha,
                "error": error_msg[:500],
            })
            consecutive_failures += 1
            last_error = error_msg
            experiment_id += 1
            continue

        # 6. Extract metric.
        metric = query_mlflow_balanced_accuracy()
        if metric is None:
            logger.warning("Failed to extract balanced_accuracy, treating as crash")
            git_reset_hard()
            append_result({
                "id": experiment_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "hypothesis": hypothesis,
                "balanced_accuracy": "",
                "best_so_far": f"{best_so_far:.6f}",
                "status": "CRASH",
                "duration_s": f"{duration:.0f}",
                "commit_sha": commit_sha,
                "error": "Failed to extract balanced_accuracy from MLflow",
            })
            consecutive_failures += 1
            last_error = "Failed to extract balanced_accuracy from MLflow"
            experiment_id += 1
            continue

        # 7. Keep or revert.
        if metric > best_so_far:
            best_so_far = metric
            status = "KEPT"
            consecutive_failures = 0
            last_error = None
            logger.info(
                f"KEPT: balanced_accuracy={metric:.4f}"
                f" (improvement from {best_so_far:.4f})")
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
            "balanced_accuracy": f"{metric:.6f}",
            "best_so_far": f"{best_so_far:.6f}",
            "status": status,
            "duration_s": f"{duration:.0f}",
            "commit_sha": commit_sha,
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
