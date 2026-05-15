"""SageMaker container entrypoint: YAML -> MLflowTrainingRunner.

Lifecycle inside the container:

  1. Parse --config-dir (default: /opt/ml/input/data/config).
  2. Load training_config.yaml.
  3. Log first-line dump: python/pkg versions, env vars (redacted),
     /opt/ml/input/data/ listing.
  4. Apply YAML's env block to os.environ.
  5. Import MLflowTrainingRunner (heavy import; reads Settings()).
  6. Build options.
  7. Run.

If anything raises, the traceback is logged and the process exits 1
so SageMaker marks the job Failed.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path


DEFAULT_CONFIG_DIR = "/opt/ml/input/data/config"
CONFIG_FILENAME = "training_config.yaml"

# Env vars whose values are sensitive enough to redact in the
# startup dump. Substring match on the key (uppercased).
_SECRET_KEY_FRAGMENTS = ("SECRET", "TOKEN", "PASSWORD", "KEY")


log = logging.getLogger("sagemaker_train_entrypoint")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


@contextmanager
def _stage(name: str):
    log.info("[stage:%s] ENTER", name)
    try:
        yield
    except Exception:
        log.info("[stage:%s] FAIL", name)
        raise
    else:
        log.info("[stage:%s] EXIT", name)


def _redact_env(env: dict) -> dict:
    out = {}
    for key, value in env.items():
        upper = key.upper()
        if any(frag in upper for frag in _SECRET_KEY_FRAGMENTS):
            out[key] = "***"
        else:
            out[key] = value
    return out


def _first_line_dump(config_dir: Path) -> None:
    """Log everything a future debugger will want to see first."""
    log.info("python: %s", sys.version.replace("\n", " "))
    try:
        from importlib.metadata import version
        for pkg in ("pyspacer", "mlflow", "duckdb", "torch",
                    "pydantic", "pyyaml"):
            try:
                log.info("pkg %s: %s", pkg, version(pkg))
            except Exception:
                log.info("pkg %s: <not installed>", pkg)
    except Exception:
        log.warning("could not read package versions")

    log.info("config_dir: %s", config_dir)
    try:
        if config_dir.is_dir():
            for entry in sorted(config_dir.iterdir()):
                log.info(
                    "config_dir entry: %s (%d bytes)",
                    entry.name,
                    entry.stat().st_size if entry.is_file() else -1,
                )
        else:
            log.warning("config_dir does not exist: %s", config_dir)
    except Exception as e:
        log.warning("config_dir listing failed: %s", e)

    redacted = _redact_env(dict(os.environ))
    for k in sorted(redacted):
        log.info("env %s=%s", k, redacted[k])


def _resolve_runner_factory():
    """Return the MLflowTrainingRunner class (heavy import).

    Factored into a function so tests can patch it without triggering
    the pyspacer import side effects.
    """
    from mermaid_classifier.pyspacer.train import MLflowTrainingRunner
    return MLflowTrainingRunner


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        default=os.environ.get(
            "SAGEMAKER_CONFIG_DIR", DEFAULT_CONFIG_DIR),
        help=(
            "Directory containing training_config.yaml and sibling "
            "CSVs. Default: /opt/ml/input/data/config (where SageMaker "
            "mounts the 'config' input channel)."
        ),
    )
    args = parser.parse_args(argv)
    config_dir = Path(args.config_dir).resolve()

    try:
        _first_line_dump(config_dir)

        with _stage("load_config"):
            from mermaid_classifier.sagemaker.config import (
                TrainingRunConfig,
            )
            config = TrainingRunConfig.from_yaml_path(
                config_dir / CONFIG_FILENAME)

        with _stage("apply_env"):
            config.apply_env()

        with _stage("build_options"):
            dataset_options, training_options, mlflow_options = (
                config.build_options(config_dir=config_dir))
            # build_options imports mermaid_classifier.pyspacer.train,
            # whose module-level logging.config.dictConfig call sets
            # disable_existing_loggers=True (the default), which marks
            # any logger created before that dictConfig call as disabled.
            # Re-enable this logger so stage markers after this point are
            # still emitted.
            logging.getLogger("sagemaker_train_entrypoint").disabled = False
            log.info("dataset_options: %s", dataset_options)
            log.info("training_options: %s", training_options)
            log.info("mlflow_options: %s", mlflow_options)

        with _stage("runner_run"):
            RunnerClass = _resolve_runner_factory()
            runner = RunnerClass(
                dataset_options=dataset_options,
                training_options=training_options,
                mlflow_options=mlflow_options,
            )
            runner.run()

    except Exception:
        log.error(
            "Training run failed; full traceback follows.\n%s",
            traceback.format_exc(),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
