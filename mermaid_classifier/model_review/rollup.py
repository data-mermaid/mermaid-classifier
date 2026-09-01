"""Top-level BA::GF rollup, self-contained in the package data dir.

Duplicated (intentionally) from the throwaway reports/model_benchmark harness so
the logic lives in the reusable mermaid_classifier package.
"""

import csv
from collections.abc import Callable

from mermaid_classifier.model_review import ROLLUP_CSV, TOPLEVEL_CSV


def split_bagf(bagf: str) -> tuple[str, str]:
    ba, _, gf = bagf.partition("::")
    return ba, gf


def load_toplevel() -> dict[str, str]:
    out: dict[str, str] = {}
    with open(TOPLEVEL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = row["name"]
    return out


def load_rollup() -> dict[tuple[str, str], tuple[str, str]]:
    lookup: dict[tuple[str, str], tuple[str, str]] = {}
    with open(ROLLUP_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[(row["from_ba_id"], row["from_gf_id"])] = (
                row["to_ba_id"],
                row["to_gf_id"],
            )
    return lookup


def make_rollup_fn(strict: bool = True) -> Callable[[str], str | None]:
    toplevel = load_toplevel()
    lookup = load_rollup()

    def roll(bagf: str) -> str | None:
        ba, gf = split_bagf(bagf)
        if (ba, gf) in lookup:
            return lookup[(ba, gf)][0]
        if ba in toplevel:  # already top-level
            return ba
        if strict:
            raise KeyError(f"no top-level rollup for {bagf!r}")
        return None

    return roll
