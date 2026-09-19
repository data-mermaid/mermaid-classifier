"""The benthic attributes the pure statistics modules are exercised against.

metrics, decisions and triage each build their point tables on the same four
attributes -- one recorded in both regions, one Atlantic-only, one
Pacific-only, and one with no region recorded at all -- and then add whichever
fifth case that module is about. Sharing the four means the line a module
adds is the only thing that differs on screen, rather than being buried in
four identical blocks.

Each module composes its own map from BASE_REGION_IDS_BY_ATTRIBUTE, because
the fifth attribute is not the same concept in all three: `...005` is an
off-label-space attribute in metrics and decisions, and a rare Pacific one in
triage, and the two off-list copies deliberately sit in different regions.
"""

TROPICAL_ATLANTIC = "1a1a1a1a-0000-4000-8000-000000000001"
CENTRAL_INDO_PACIFIC = "1a1a1a1a-0000-4000-8000-000000000002"
EASTERN_PACIFIC = "1a1a1a1a-0000-4000-8000-000000000003"

BA_GLOBAL = "2b2b2b2b-0000-4000-8000-000000000001"
BA_ATLANTIC = "2b2b2b2b-0000-4000-8000-000000000002"
BA_PACIFIC = "2b2b2b2b-0000-4000-8000-000000000003"
BA_NO_REGIONS = "2b2b2b2b-0000-4000-8000-000000000004"

GLOBAL_LABEL = f"{BA_GLOBAL}::"
ATLANTIC_LABEL = f"{BA_ATLANTIC}::"
PACIFIC_LABEL = f"{BA_PACIFIC}::"
NO_REGIONS_LABEL = f"{BA_NO_REGIONS}::"

BASE_REGION_IDS_BY_ATTRIBUTE = {
    BA_GLOBAL: frozenset({TROPICAL_ATLANTIC, CENTRAL_INDO_PACIFIC}),
    BA_ATLANTIC: frozenset({TROPICAL_ATLANTIC}),
    BA_PACIFIC: frozenset({CENTRAL_INDO_PACIFIC}),
    BA_NO_REGIONS: frozenset(),
}
