"""Compact structural interning for direct DAG construction."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import JsonValue

from geml.graph.schema import ChildRef, Graph, GraphNode, GraphRoot
from geml.graph.signatures import signature_from_parts
from geml.graph.validate import validate_graph


@dataclass(frozen=True, slots=True)
class InternedNode:
    """An owner-checked reference to one structurally interned node."""

    node_id: str
    signature: str
    _owner: object = field(repr=False, compare=False)


class HashConsTable:
    """Intern nodes by compact child signatures without expanding shared trees."""

    def __init__(self, family: str) -> None:
        if not isinstance(family, str) or not family.strip():
            raise ValueError("family must be a nonblank string")
        self._family = family
        self._owner = object()
        self._nodes: dict[str, GraphNode] = {}
        self._refs: dict[str, InternedNode] = {}
        self._intern_requests = 0
        self._cache_hits = 0
        self._peak_size = 0

    @property
    def family(self) -> str:
        return self._family

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def intern_requests(self) -> int:
        return self._intern_requests

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def peak_size(self) -> int:
        """Return the maximum number of unique entries held simultaneously."""

        return self._peak_size

    def _require_owned(self, ref: InternedNode) -> InternedNode:
        if not isinstance(ref, InternedNode) or ref._owner is not self._owner:
            raise ValueError("child reference belongs to a different interning table")
        if ref.node_id not in self._nodes:
            raise ValueError("child reference does not exist in this interning table")
        return ref

    def intern(
        self,
        *,
        kind: str,
        label: str | None,
        value: JsonValue = None,
        children: tuple[InternedNode, ...] = (),
    ) -> InternedNode:
        """Return the unique node matching one exact ordered structure."""

        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("node kind must be a nonblank string")
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise ValueError("node label must be None or a nonblank string")
        owned_children = tuple(self._require_owned(child) for child in children)
        signature = signature_from_parts(
            family=self._family,
            kind=kind,
            label=label,
            value=value,
            children=((slot, child.signature) for slot, child in enumerate(owned_children)),
        )
        self._intern_requests += 1
        if existing := self._refs.get(signature):
            self._cache_hits += 1
            return existing

        node_id = f"{self._family}-{signature}"
        node = GraphNode(
            node_id=node_id,
            family=self._family,
            kind=kind,
            label=label,
            value=value,
            children=tuple(
                ChildRef(slot=slot, target_id=child.node_id)
                for slot, child in enumerate(owned_children)
            ),
        )
        ref = InternedNode(node_id=node_id, signature=signature, _owner=self._owner)
        self._nodes[node_id] = node
        self._refs[signature] = ref
        self._peak_size = max(self._peak_size, len(self._nodes))
        return ref

    def _reachable_ids(self, root: InternedNode) -> set[str]:
        reachable: set[str] = set()
        stack = [root.node_id]
        while stack:
            node_id = stack.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            stack.extend(child.target_id for child in self._nodes[node_id].children)
        return reachable

    def to_graph(
        self,
        root: InternedNode,
        *,
        root_id: str,
        representation_mode: str,
    ) -> Graph:
        """Freeze the complete table as one validated rooted graph."""

        root = self._require_owned(root)
        unreachable = set(self._nodes) - self._reachable_ids(root)
        if unreachable:
            sample = sorted(unreachable)[:3]
            raise ValueError(
                "interning table contains unreachable nodes: "
                + ", ".join(repr(node_id) for node_id in sample)
            )

        graph = Graph(
            nodes=self._nodes,
            roots=(
                GraphRoot(
                    root_id=root_id,
                    target_id=root.node_id,
                    representation_mode=representation_mode,
                ),
            ),
        )
        validation = validate_graph(graph)
        if not validation.valid:
            raise ValueError(
                "interning table produced an invalid graph: " + "; ".join(validation.errors)
            )
        return graph
