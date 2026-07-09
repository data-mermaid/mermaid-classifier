"""Generate Label Studio labeling-config XML for the review task.

Points are regions of a colored top-level ``KeyPointLabels`` control (``toplevel``)
so each point renders in its highest-level-category colour, plus a perRegion
``Taxonomy`` control (``label``) for the fine BA::GF label the experts assign.
"""

import xml.etree.ElementTree as ET

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary,
    GrowthFormLibrary,
)
from mermaid_classifier.model_review.rollup import load_toplevel

# Placeholder top-level for seeded, not-yet-labelled expert points.
UNLABELED_TOPLEVEL = "Unlabeled"
UNLABELED_COLOR = "#c8c8c8"

# Distinct, reasonably colour-blind-aware palette for the ~12 top-level categories.
TOPLEVEL_PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fab1c0",
    "#008080",
    "#9a6324",
    "#800000",
    "#808000",
    "#000075",
    "#e6beff",
]


def toplevel_label_colors() -> list[tuple[str, str]]:
    """(name, hex colour) for each top-level category, plus the Unlabeled placeholder."""
    names = sorted(load_toplevel().values())
    pairs = [(name, TOPLEVEL_PALETTE[i % len(TOPLEVEL_PALETTE)]) for i, name in enumerate(names)]
    pairs.append((UNLABELED_TOPLEVEL, UNLABELED_COLOR))
    return pairs


def _add_ba_subtree(
    parent_el: ET.Element,
    ba_lib: BenthicAttributeLibrary,
    gf_lib: GrowthFormLibrary,
    ba_id: str | None,
) -> None:
    children = ba_lib.by_parent.get(ba_id, [])
    children = sorted(children, key=lambda r: r["name"])
    for child in children:
        choice = ET.SubElement(parent_el, "Choice", {"value": child["name"]})
        _add_ba_subtree(choice, ba_lib, gf_lib, child["id"])
        # leaf BA: offer growth forms as selectable leaves
        if not ba_lib.by_parent.get(child["id"]):
            for gf_name in sorted(gf_lib.by_id.values()):
                ET.SubElement(choice, "Choice", {"value": gf_name})


def build_taxonomy_config(
    ba_lib: BenthicAttributeLibrary,
    gf_lib: GrowthFormLibrary,
    image_value: str = "$image_url",
) -> str:
    view = ET.Element("View")
    ET.SubElement(view, "Image", {"name": "image", "value": image_value, "zoom": "true"})
    # Colored top-level control: creates the keypoint regions and colours each
    # point by its highest-level category (GT/V1 references set it; expert points
    # are seeded to Unlabeled/grey).
    kpl = ET.SubElement(view, "KeyPointLabels", {"name": "toplevel", "toName": "image"})
    for name, color in toplevel_label_colors():
        ET.SubElement(kpl, "Label", {"value": name, "background": color})
    tax = ET.SubElement(
        view,
        "Taxonomy",
        {
            "name": "label",
            "toName": "image",
            "perRegion": "true",
            "showFullPath": "true",
            "pathSeparator": "::",
        },
    )
    _add_ba_subtree(tax, ba_lib, gf_lib, None)
    ET.SubElement(
        view,
        "TextArea",
        {
            "name": "notes",
            "toName": "image",
            "editable": "true",
            "rows": "4",
            "placeholder": "Notes on the ground-truth and V1 reference labels…",
        },
    )
    return ET.tostring(view, encoding="unicode")


def build_flat_config(labels: list[str], image_value: str = "$image_url") -> str:
    view = ET.Element("View")
    ET.SubElement(view, "Image", {"name": "image", "value": image_value, "zoom": "true"})
    kpl = ET.SubElement(view, "KeyPointLabels", {"name": "label", "toName": "image"})
    for label in labels:
        ET.SubElement(kpl, "Label", {"value": label})
    ET.SubElement(
        view,
        "TextArea",
        {
            "name": "notes",
            "toName": "image",
            "editable": "true",
            "rows": "4",
            "placeholder": "Notes on the ground-truth and V1 reference labels…",
        },
    )
    return ET.tostring(view, encoding="unicode")
