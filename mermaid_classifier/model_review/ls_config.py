"""Generate Label Studio labeling-config XML for the review task."""

import xml.etree.ElementTree as ET

from mermaid_classifier.common.benthic_attributes import (
    BenthicAttributeLibrary,
    GrowthFormLibrary,
)


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
    ET.SubElement(view, "KeyPoint", {"name": "kp", "toName": "image"})
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
