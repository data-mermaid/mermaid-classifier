"""Unit tests for region lookup on BenthicAttributeLibrary.

Regions are a benthic-attribute level property in MERMAID, carried in the
`regions` field of each /v1/benthicattributes/ result. These tests pin that the
lookup is built from the response already in memory, that an unrecorded region
field stays distinguishable from an unknown attribute, and that a growth form
never narrows the set.

urllib.request.urlopen is mocked. The canned payload mirrors the full shape of
a real result -- a partial payload would pass here while production code
reading an omitted field breaks -- and carries both shapes an unrecorded
`regions` field takes, JSON null and [].
"""

import io
import json
import unittest
from typing import Any
from unittest import mock

from mermaid_classifier.common.benthic_attributes import BenthicAttributeLibrary, RegionLibrary

# Benthic attribute UUIDs.
HARD_CORAL = "aa000000-0000-4000-8000-000000000001"
ACROPORA = "aa000000-0000-4000-8000-000000000002"
DENDROGYRA = "aa000000-0000-4000-8000-000000000003"
NULL_REGIONS_BA = "aa000000-0000-4000-8000-000000000004"
EMPTY_REGIONS_BA = "aa000000-0000-4000-8000-000000000005"
ABSENT_BA = "aa000000-0000-4000-8000-0000000000ff"

# MERMAID region UUIDs.
TROPICAL_ATLANTIC = "f0f0f0f0-0000-4000-8000-000000000001"
CENTRAL_INDO_PACIFIC = "f0f0f0f0-0000-4000-8000-000000000002"
WESTERN_INDO_PACIFIC = "f0f0f0f0-0000-4000-8000-000000000003"

# Growth form UUID.
BRANCHING = "cc000000-0000-4000-8000-000000000001"

CREATED_BY = "bb000000-0000-4000-8000-000000000001"
TOP_LEVEL = "dd000000-0000-4000-8000-000000000001"


def _result(ba_id: str, name: str, parent: str | None, regions: list[str] | None) -> dict[str, Any]:
    """One /v1/benthicattributes/ result, with every field the API returns."""
    return {
        "id": ba_id,
        "updated_by": None,
        "status": 90,
        "top_level_category": TOP_LEVEL,
        "created_on": "2020-01-01T00:00:00.000000Z",
        "updated_on": "2024-01-01T00:00:00.000000Z",
        "name": name,
        "notes": "",
        "created_by": CREATED_BY,
        "parent": parent,
        "regions": regions,
        "life_histories": None,
        "growth_form_life_histories": None,
    }


CANNED_RESULTS = [
    _result(
        HARD_CORAL,
        "Hard coral",
        None,
        [TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC, WESTERN_INDO_PACIFIC],
    ),
    _result(ACROPORA, "Acropora", HARD_CORAL, [CENTRAL_INDO_PACIFIC, WESTERN_INDO_PACIFIC]),
    _result(DENDROGYRA, "Dendrogyra", HARD_CORAL, [TROPICAL_ATLANTIC]),
    # The live API returns null for all 11 unrecorded attributes; [] is the
    # other shape the field is declared to take.
    _result(NULL_REGIONS_BA, "Algal assemblage", HARD_CORAL, None),
    _result(EMPTY_REGIONS_BA, "Juvenile coral", HARD_CORAL, []),
]

CANNED_PAYLOAD = {
    "count": len(CANNED_RESULTS),
    "next": None,
    "previous": None,
    "results": CANNED_RESULTS,
}


def _canned_urlopen(*_args: object, **_kwargs: object) -> io.BytesIO:
    return io.BytesIO(json.dumps(CANNED_PAYLOAD).encode())


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


class BenthicAttributeRegionTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("urllib.request.urlopen", side_effect=_canned_urlopen)
        self.urlopen = patcher.start()
        self.addCleanup(patcher.stop)
        self.library = BenthicAttributeLibrary()

    def test_region_ids_come_back_as_the_attributes_own_regions(self):
        self.assertEqual(
            self.library.get_region_ids(ACROPORA),
            frozenset({CENTRAL_INDO_PACIFIC, WESTERN_INDO_PACIFIC}),
        )

    def test_single_region_attribute(self):
        self.assertEqual(self.library.get_region_ids(DENDROGYRA), frozenset({TROPICAL_ATLANTIC}))

    def test_empty_regions_array_gives_an_empty_set(self):
        """Unrecorded regions stay empty, not absent, in either shape.

        Of 810 real attributes 11 are unrecorded, and every one of them comes
        back as JSON null rather than []. Reading the field without a null
        guard raises TypeError against the live API, for every caller of
        BenthicAttributeLibrary rather than only the region ones.
        """
        self.assertEqual(self.library.get_region_ids(NULL_REGIONS_BA), frozenset())
        self.assertEqual(self.library.get_region_ids(EMPTY_REGIONS_BA), frozenset())

    def test_unknown_attribute_raises_rather_than_looking_unrecorded(self):
        """An unknown id must not be silently indistinguishable from regions: [].

        An empty set means 'never counted', so returning one for a missing id
        would drop those predictions from every rate without a trace.
        """
        with self.assertRaises(KeyError):
            self.library.get_region_ids(ABSENT_BA)

    def test_lookup_makes_no_further_request(self):
        """The regions arrive with the one response the constructor already read."""
        calls_after_construction = self.urlopen.call_count
        self.library.get_region_ids(ACROPORA)
        self.assertEqual(self.urlopen.call_count, calls_after_construction)

    def test_region_ids_are_hashable_and_immutable(self):
        """Callers put these sets in dicts and compare them; a list would not do."""
        self.assertIsInstance(self.library.get_region_ids(ACROPORA), frozenset)


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
