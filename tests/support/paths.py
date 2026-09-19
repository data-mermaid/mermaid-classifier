"""Where this repo's directories are, derived once.

Every consumer used to re-derive the repo root from its own location, so the
expression encoded the test module's depth under tests/ and moving a file
changed what landed on sys.path.

scripts/ holds CLI drivers rather than an installed package, so a test that
imports one has to put it on sys.path first.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def add_scripts_to_path() -> None:
    """Put scripts/ at the front of sys.path, once."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
