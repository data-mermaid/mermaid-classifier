"""Test fakes for the BA / GF library Protocols.

The real ``BenthicAttributeLibrary`` hits the MERMAID API on construction
and is therefore unsuitable for unit tests. These fakes implement the
minimal surface the trainer wiring reads when the taxonomy-library
accessors are monkeypatched:

  - ``by_id``: dict[ba_id] -> {'id', 'name', 'parent'}
  - ``get_ancestor_ids(ba_id)``: list[str], root-first
  - ``GFLibrary.id_to_name`` / ``BALibrary.id_to_name``: id -> name.
"""

from __future__ import annotations


class FakeBALibrary:
    """Minimal BA library backed by an in-memory tree.

    Construct with a list of (ba_id, parent_id) edges plus an optional
    name dict. ``parent_id == None`` means the BA is a root.
    """

    def __init__(
        self,
        edges: list[tuple[str, str | None]],
        names: dict[str, str] | None = None,
    ):
        names = names or {}
        self.by_id: dict[str, dict] = {}
        for ba_id, parent in edges:
            self.by_id[ba_id] = {
                "id": ba_id,
                "parent": parent,
                "name": names.get(ba_id, ba_id),
            }

    def get_ancestor_ids(self, ba_id: str) -> list[str]:
        chain: list[str] = []
        cur = self.by_id[ba_id]["parent"]
        while cur is not None:
            chain.append(cur)
            cur = self.by_id[cur]["parent"]
        return list(reversed(chain))

    def id_to_name(self, ba_id: str) -> str:
        if ba_id == "":
            return ""
        return self.by_id[ba_id]["name"]


class FakeGFLibrary:
    def __init__(self, names: dict[str, str] | None = None):
        names = names or {}
        self.by_id: dict[str, str] = dict(names)

    def id_to_name(self, gf_id: str) -> str:
        if gf_id == "":
            return ""
        return self.by_id.get(gf_id, gf_id)


def small_tree() -> FakeBALibrary:
    """A small canonical tree used by several tests:

    root (None)
     |- A
     |   |- A1
     |   |- A2
     |- B
     |   |- B1
     |- C   (single-child chain to test depth collapse)
     |   |- C1
     |       |- C1a
    """
    edges: list[tuple[str, str | None]] = [
        ("A", None),
        ("A1", "A"),
        ("A2", "A"),
        ("B", None),
        ("B1", "B"),
        ("C", None),
        ("C1", "C"),
        ("C1a", "C1"),
    ]
    return FakeBALibrary(edges)
