"""Unit tests for region lookups on benthic attributes and RegionLibrary.

Regions are a benthic-attribute level property in MERMAID, carried in the
`regions` field of each /v1/benthicattributes/ result. RegionLibrary is a
separate, /v1/choices/-backed lookup from region id to name: the
benthic-attribute response carries region ids and no names, so a report that
renders a region by name resolves it through RegionLibrary instead.

urllib.request.urlopen is mocked.
"""

import io
import json
import unittest
from typing import Any
from unittest import mock

from mermaid_classifier.common.benthic_attributes import BenthicAttributeLibrary, RegionLibrary

# Benthic attribute UUIDs.
NULL_REGIONS_BA = "aa000000-0000-4000-8000-000000000004"
EMPTY_REGIONS_BA = "aa000000-0000-4000-8000-000000000005"

# MERMAID region UUIDs.
TROPICAL_ATLANTIC = "f0f0f0f0-0000-4000-8000-000000000001"
CENTRAL_INDO_PACIFIC = "f0f0f0f0-0000-4000-8000-000000000002"

# Growth form UUID.
BRANCHING = "cc000000-0000-4000-8000-000000000001"


def _ba_result(ba_id: str, name: str, regions: list[str] | None) -> dict[str, Any]:
    """One /v1/benthicattributes/ result, carrying the fields __init__ reads."""
    return {"id": ba_id, "name": name, "parent": None, "regions": regions}


# The live API returns null for all 11 of 810 unrecorded attributes; [] is the
# other shape the field is declared to take.
BA_PAYLOAD = {
    "count": 2,
    "next": None,
    "previous": None,
    "results": [
        _ba_result(NULL_REGIONS_BA, "Algal assemblage", None),
        _ba_result(EMPTY_REGIONS_BA, "Juvenile coral", []),
    ],
}


def _canned_ba_urlopen(*_args: object, **_kwargs: object) -> io.BytesIO:
    return io.BytesIO(json.dumps(BA_PAYLOAD).encode())


# One /v1/choices/ response. Each region carries the polygon the API returns
# beside its name, and the growth-form set sits alongside it: a payload holding
# only the regions would pass while production code picking the wrong set by
# position breaks.
CANNED_CHOICES = [
    {
        "name": "growthforms",
        "data": [{"id": BRANCHING, "name": "branching", "updated_on": "2020-01-01T00:00:00Z"}],
    },
    {
        "name": "regions",
        "data": [
            {
                "id": TROPICAL_ATLANTIC,
                "name": "Tropical Atlantic",
                "geom": {"type": "MultiPolygon", "coordinates": [[[[0.0, 0.0]]]]},
            },
            {
                "id": CENTRAL_INDO_PACIFIC,
                "name": "Central Indo-Pacific",
                "geom": {"type": "MultiPolygon", "coordinates": [[[[1.0, 1.0]]]]},
            },
        ],
    },
]


def _canned_choices_urlopen(payload: object):
    def urlopen(*_args: object, **_kwargs: object) -> io.BytesIO:
        return io.BytesIO(json.dumps(payload).encode())

    return urlopen


class BenthicAttributeRegionIdsTest(unittest.TestCase):
    """region_ids_by_id, built eagerly from the /v1/benthicattributes/ response."""

    def test_null_and_empty_regions_both_read_back_as_an_empty_set(self):
        """Unrecorded regions stay empty, not absent, in either shape.

        Of 810 real attributes 11 are unrecorded, and every one of them comes
        back as JSON null rather than []. Reading the field without a null
        guard raises TypeError against the live API, for every caller of
        BenthicAttributeLibrary rather than only the region ones.
        """
        patcher = mock.patch("urllib.request.urlopen", side_effect=_canned_ba_urlopen)
        patcher.start()
        self.addCleanup(patcher.stop)
        library = BenthicAttributeLibrary()

        self.assertEqual(library.region_ids_by_id[NULL_REGIONS_BA], frozenset())
        self.assertEqual(library.region_ids_by_id[EMPTY_REGIONS_BA], frozenset())


class RegionLibraryTest(unittest.TestCase):
    """Region names, which the probe freezes so a report can render them.

    The benthic-attribute response carries region ids and no names, so the
    names come from the /v1/choices/ region set.
    """

    def _library(self, payload: object = CANNED_CHOICES) -> RegionLibrary:
        patcher = mock.patch("urllib.request.urlopen", side_effect=_canned_choices_urlopen(payload))
        patcher.start()
        self.addCleanup(patcher.stop)
        return RegionLibrary()

    def test_names_are_keyed_by_region_id(self):
        """Keyed by name the lookup would be inverted, and every report would
        render a UUID where it meant to render a name."""
        library = self._library()
        self.assertEqual(library.by_id[TROPICAL_ATLANTIC], "Tropical Atlantic")
        self.assertEqual(library.id_to_name(CENTRAL_INDO_PACIFIC), "Central Indo-Pacific")

    def test_the_region_set_is_read_rather_than_whichever_set_comes_first(self):
        library = self._library()
        self.assertNotIn(BRANCHING, library.by_id)
        self.assertEqual(len(library.by_id), 2)

    def test_a_response_without_regions_raises_rather_than_naming_nothing(self):
        """Silently returning an empty map would freeze a name snapshot that
        names no region at all, and every table would read as unresolved."""
        with self.assertRaises(ValueError):
            self._library([{"name": "growthforms", "data": []}])


if __name__ == "__main__":
    unittest.main()
