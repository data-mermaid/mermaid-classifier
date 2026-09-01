"""Model-evaluation annotation review tool (Label Studio based)."""

from pathlib import Path

_DATA = Path(__file__).resolve().parent / "data"
ROLLUP_CSV = _DATA / "ba_rollup_top_level.csv"
TOPLEVEL_CSV = _DATA / "top_level_all.csv"
