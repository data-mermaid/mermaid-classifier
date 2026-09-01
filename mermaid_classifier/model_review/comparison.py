"""Render one point's label from every annotation set as a single readable block.

The reference layers live on separate Label Studio tabs, and switching tabs discards
the selected region — each annotation is its own store in the editor, and Community
edition has no side-by-side compare. So the comparison is assembled into one
per-region text block on a single tab instead, where selecting a point shows every
set's fine label at once.

``comparison_text`` takes the sets in display order, so adding a reviewer's own label
later is a longer sequence, not a different format.
"""

from collections.abc import Callable, Sequence

# The read-only tab carrying the per-point comparison.
COMPARISON_MODEL_VERSION = "Comparison"

GROUND_TRUTH = "ground-truth"


def comparison_text(
    entries: Sequence[tuple[str, str]],
    label_name: Callable[[str], str],
    reference: str = GROUND_TRUTH,
) -> str:
    """One line per set: ``<set>: <fine label path>``, marked against ``reference``.

    ``entries`` is ``(set name, BA::GF)`` in the order the lines should read. The
    reference set carries no marker; every other line is marked match or differs, so
    a disagreement is visible without comparing the paths character by character.

    The marker compares the *rendered* labels, not the BA::GF ids behind them, so it
    always agrees with the two lines a reviewer is reading.
    """
    rendered = [(name, label_name(bagf)) for name, bagf in entries]
    ref_label = next((label for name, label in rendered if name == reference), None)
    lines: list[str] = []
    for name, label in rendered:
        line = f"{name}: {label}"
        if name != reference and ref_label is not None:
            line += "  — match" if label == ref_label else "  — differs"
        lines.append(line)
    return "\n".join(lines)
