"""Group-level test and cross-validation fold selection.

A group is a (site, project_id) pair: a CoralNet source or a MERMAID
project. Selection is a deterministic greedy over per-(class, group) point
counts; it reads no data itself.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import pandas as pd

Group = tuple[str, str]

TEST_FOLD = "test"
# Rows whose MERMAID image has no project; they always train.
UNKNOWN_GROUP = "unknown"
CORALNET_SITE = "coralnet"
_COUNT_COLUMNS = ("class_key", "site", "project_id", "n")


@dataclasses.dataclass(frozen=True)
class SelectionParams:
    test_frac: float
    folds: int
    min_test_points: int
    max_class_group_loss: float
    seed: int
    # Per site, test groups hold at most this fraction of the site's points; None or 0 disables.
    max_test_point_frac: float | None = None


@dataclasses.dataclass(frozen=True)
class SiteShare:
    test_groups: int
    groups: int
    test_points: int
    total_points: int


@dataclasses.dataclass
class HoldoutSelection:
    # Group -> "test" or "0".."K-1". 'unknown' groups are absent.
    assignments: dict[Group, str]
    warnings: list[str]


def _tiebreak(seed: int, group: Group) -> float:
    """A seed-dependent, input-order-independent value in [0, 1)."""
    digest = hashlib.sha256(f"{seed}|{group[0]}|{group[1]}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _target_test_groups(n_groups: int, test_frac: float) -> int:
    if test_frac <= 0 or n_groups < 2:
        return 0
    return max(1, math.floor(test_frac * n_groups + 0.5))


def select_holdout(
    counts: pd.DataFrame, params: SelectionParams, beta_sources: set[str]
) -> HoldoutSelection:
    """
    Choose test groups per site, then spread the remaining groups over folds.

    `counts` has columns class_key, site, project_id, n (points). CoralNet
    test candidates exclude `beta_sources`; 'unknown' groups never become
    test and are not assigned a fold.
    """
    missing = set(_COUNT_COLUMNS) - set(counts.columns)
    if missing:
        raise ValueError(f"counts is missing columns: {sorted(missing)}")

    class_points: dict[Group, dict[str, int]] = {}
    for class_key, site, project_id, n in zip(
        counts["class_key"], counts["site"], counts["project_id"], counts["n"], strict=True
    ):
        group = (str(site), str(project_id))
        by_class = class_points.setdefault(group, {})
        by_class[str(class_key)] = by_class.get(str(class_key), 0) + int(n)

    groups_per_class: dict[str, int] = {}
    for by_class in class_points.values():
        for class_key in by_class:
            groups_per_class[class_key] = groups_per_class.get(class_key, 0) + 1
    max_test_groups = {
        c: math.floor(params.max_class_group_loss * total + 1e-9)
        for c, total in groups_per_class.items()
    }

    warnings: list[str] = []
    test_groups: set[Group] = set()
    # Per (site, class): CoralNet test points must not make MERMAID look covered.
    test_points: dict[tuple[str, str], int] = {}
    test_group_count: dict[str, int] = {}

    sites = sorted({site for site, _ in class_points})
    for site in sites:
        site_groups = sorted(g for g in class_points if g[0] == site and g[1] != UNKNOWN_GROUP)
        target = _target_test_groups(len(site_groups), params.test_frac)
        candidates = [
            g for g in site_groups if not (site == CORALNET_SITE and g[1] in beta_sources)
        ]
        # The site total includes 'unknown' and Beta groups, as `total_points_{site}` does.
        site_total = sum(sum(class_points[g].values()) for g in class_points if g[0] == site)
        point_cap = params.max_test_point_frac * site_total if params.max_test_point_frac else None
        site_test_points = 0
        cap_blocked = False
        chosen = 0
        while chosen < target:
            best: tuple[float, float, Group] | None = None
            for group in candidates:
                if group in test_groups:
                    continue
                by_class = class_points[group]
                if any(test_group_count.get(c, 0) + 1 > max_test_groups[c] for c in by_class):
                    continue
                if (
                    point_cap is not None
                    and site_test_points + sum(by_class.values()) > point_cap + 1e-9
                ):
                    cap_blocked = True
                    continue
                gain = sum(
                    min(n, params.min_test_points - test_points.get((site, c), 0))
                    for c, n in by_class.items()
                    if test_points.get((site, c), 0) < params.min_test_points
                )
                key = (-gain / max(params.min_test_points, 1), _tiebreak(params.seed, group), group)
                if best is None or key < best:
                    best = key
            if best is None:
                if chosen == 0 and cap_blocked and len(candidates) >= 2:
                    warnings.append(
                        f"site {site!r}: max_test_point_frac={params.max_test_point_frac} leaves"
                        f" 0 of {target} test groups; every eligible group exceeds the cap."
                    )
                else:
                    warnings.append(
                        f"site {site!r}: selected {chosen} of {target} test groups; no remaining"
                        f" eligible group satisfies max_class_group_loss"
                        f"{' and max_test_point_frac' if cap_blocked else ''}."
                    )
                break
            group = best[2]
            test_groups.add(group)
            chosen += 1
            site_test_points += sum(class_points[group].values())
            for c, n in class_points[group].items():
                test_points[(site, c)] = test_points.get((site, c), 0) + n
                test_group_count[c] = test_group_count.get(c, 0) + 1

    assignments: dict[Group, str] = dict.fromkeys(test_groups, TEST_FOLD)
    remaining = sorted(g for g in class_points if g[1] != UNKNOWN_GROUP and g not in test_groups)
    assignments.update(_assign_folds(class_points, remaining, params))
    return HoldoutSelection(assignments=assignments, warnings=warnings)


def _assign_folds(
    class_points: dict[Group, dict[str, int]], groups: list[Group], params: SelectionParams
) -> dict[Group, str]:
    """Greedy: rarest-class groups first, each to the fold that least unbalances classes."""
    k = params.folds
    totals: dict[str, int] = {}
    for group in groups:
        for c, n in class_points[group].items():
            totals[c] = totals.get(c, 0) + n
    target = {c: total / k for c, total in totals.items()}

    def order_key(group: Group) -> tuple[float, float, Group]:
        rarest = min((totals[c] for c in class_points[group]), default=0)
        return (rarest, _tiebreak(params.seed, group), group)

    fold_points: list[dict[str, int]] = [{} for _ in range(k)]
    fold_sizes = [0] * k
    result: dict[Group, str] = {}
    for group in sorted(groups, key=order_key):
        by_class = class_points[group]

        def cost(fold: int, by_class: dict[str, int] = by_class) -> float:
            increase = 0.0
            for c, n in by_class.items():
                have = fold_points[fold].get(c, 0)
                t = target[c]
                increase += ((have + n - t) ** 2 - (have - t) ** 2) / t**2
            return increase + 1e-6 * fold_sizes[fold]

        fold = min(range(k), key=lambda f: (cost(f), f))
        result[group] = str(fold)
        fold_sizes[fold] += 1
        for c, n in by_class.items():
            fold_points[fold][c] = fold_points[fold].get(c, 0) + n
    return result


def site_test_shares(counts: pd.DataFrame, assignments: dict[Group, str]) -> dict[str, SiteShare]:
    """
    Per site, the test share in groups and in points.

    Points are the site's total (as `total_points_{site}`); groups exclude 'unknown'.
    """
    points: dict[Group, int] = {}
    for site, project_id, n in zip(counts["site"], counts["project_id"], counts["n"], strict=True):
        group = (str(site), str(project_id))
        points[group] = points.get(group, 0) + int(n)
    shares: dict[str, SiteShare] = {}
    for site in sorted({g[0] for g in points}):
        here = [g for g in points if g[0] == site]
        known = [g for g in here if g[1] != UNKNOWN_GROUP]
        tests = [g for g in here if assignments.get(g) == TEST_FOLD]
        shares[site] = SiteShare(
            test_groups=len(tests),
            groups=len(known),
            test_points=sum(points[g] for g in tests),
            total_points=sum(points[g] for g in here),
        )
    return shares


def build_coverage_report(
    counts: pd.DataFrame, assignments: dict[Group, str], folds: int
) -> pd.DataFrame:
    """
    One row per class. 'unknown' groups count toward total_groups and
    train_groups (they train) but have no fold. `total_points_{site}` and
    `test_points_{site}` split the point counts by site.
    """
    sites = sorted({str(s) for s in counts["site"]})
    rows: list[dict[str, object]] = []
    for class_key, sub in counts.groupby("class_key"):
        entry: dict[str, object] = {"class_key": class_key}
        groups = [(str(s), str(p)) for s, p in zip(sub["site"], sub["project_id"], strict=True)]
        points = [int(n) for n in sub["n"]]
        folds_of = [assignments.get(g) for g in groups]
        entry["total_groups"] = len(set(groups))
        entry["total_points"] = sum(points)
        entry["test_groups"] = len(
            {g for g, f in zip(groups, folds_of, strict=True) if f == TEST_FOLD}
        )
        entry["test_points"] = sum(
            n for n, f in zip(points, folds_of, strict=True) if f == TEST_FOLD
        )
        for site in sites:
            at_site = [str(s) == site for s in sub["site"]]
            entry[f"total_points_{site}"] = sum(
                n for n, here in zip(points, at_site, strict=True) if here
            )
            entry[f"test_points_{site}"] = sum(
                n
                for n, f, here in zip(points, folds_of, at_site, strict=True)
                if here and f == TEST_FOLD
            )
        entry["train_groups"] = entry["total_groups"] - entry["test_groups"]  # pyright: ignore[reportOperatorIssue]
        for fold in range(folds):
            entry[f"points_fold_{fold}"] = sum(
                n for n, f in zip(points, folds_of, strict=True) if f == str(fold)
            )
        rows.append(entry)
    return pd.DataFrame(rows)


def coverage_warnings(report: pd.DataFrame, min_test_points: int, folds: int) -> list[str]:
    """Classes thin in a site's test set (only sites with points for the class) or absent from a fold."""
    out: list[str] = []
    sites = [c.removeprefix("test_points_") for c in report.columns if c.startswith("test_points_")]
    for row in report.to_dict("records"):
        name = row["class_key"]
        for site in sites:
            if row[f"total_points_{site}"] > 0 and row[f"test_points_{site}"] < min_test_points:
                out.append(
                    f"class {name!r} has {row[f'test_points_{site}']} {site} test points"
                    f" (< {min_test_points})."
                )
        for fold in range(folds):
            if row[f"points_fold_{fold}"] == 0:
                out.append(f"class {name!r} has no points in fold {fold}.")
    return out
