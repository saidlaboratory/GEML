"""Representation-neutral records for validated directed acyclic graphs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import JsonValue

type RepresentationFamily = str

AST_FAMILY = "ast"
EML_FAMILY = "eml"
MACRO_FAMILY = "macro"
MOTIF_FAMILY = "motif"
REPRESENTATION_FAMILIES = frozenset({AST_FAMILY, EML_FAMILY, MACRO_FAMILY, MOTIF_FAMILY})

EML_OPERATOR_KIND = "eml"
EML_VARIABLE_KIND = "variable"
EML_ONE_KIND = "one"


class _FrozenJsonList(list[JsonValue]):
    """A JSON array snapshot that retains normal list read semantics."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("JSON value snapshots are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __copy__(self) -> _FrozenJsonList:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenJsonList:
        memo[id(self)] = self
        return self


class _FrozenJsonDict(dict[str, JsonValue]):
    """A JSON object snapshot that retains normal dict read semantics."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("JSON value snapshots are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __copy__(self) -> _FrozenJsonDict:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenJsonDict:
        memo[id(self)] = self
        return self


def _snapshot_json_value(
    value: object,
    *,
    active_container_ids: set[int],
) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, tuple):
        raise TypeError("tuples are not valid JSON arrays; use a list")
    if not isinstance(value, (list, dict)):
        raise TypeError(
            "JSON values must be null, bool, int, finite float, str, list, or string-keyed dict"
        )

    container_id = id(value)
    if container_id in active_container_ids:
        raise ValueError("JSON values cannot contain reference cycles")
    active_container_ids.add(container_id)
    try:
        if isinstance(value, list):
            return _FrozenJsonList(
                _snapshot_json_value(item, active_container_ids=active_container_ids)
                for item in value
            )

        snapshot: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            snapshot[str(key)] = _snapshot_json_value(
                item,
                active_container_ids=active_container_ids,
            )
        return _FrozenJsonDict(snapshot)
    finally:
        active_container_ids.remove(container_id)


def strict_json_snapshot(value: object) -> JsonValue:
    """Return a recursively immutable copy of one strict JSON value.

    Lists remain list-compatible JSON arrays and dictionaries remain
    dict-compatible JSON objects. Tuples are rejected rather than being
    silently normalized to arrays, preserving typed structural identity.
    """

    return _snapshot_json_value(value, active_container_ids=set())


@dataclass(frozen=True, slots=True)
class ChildRef:
    """One explicit reference in a parent's ordered child slots."""

    slot: int
    target_id: str


@dataclass(frozen=True, slots=True)
class GraphRoot:
    """One ordered root reference and its representation-mode label."""

    root_id: str
    target_id: str
    representation_mode: str


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One node in a representation-neutral graph."""

    node_id: str
    family: RepresentationFamily
    kind: str
    label: str | None = None
    value: JsonValue = None
    children: tuple[ChildRef, ...] = ()

    def __post_init__(self) -> None:
        """Snapshot all nested records so structural identity cannot drift."""

        object.__setattr__(self, "value", strict_json_snapshot(self.value))
        object.__setattr__(self, "children", tuple(self.children))


@dataclass(frozen=True, slots=True)
class Graph:
    """An immutable graph snapshot with one or more ordered root references."""

    nodes: Mapping[str, GraphNode]
    roots: tuple[GraphRoot, ...]

    def __post_init__(self) -> None:
        """Copy mutable inputs into a read-only graph snapshot."""

        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))
        object.__setattr__(self, "roots", tuple(self.roots))


@dataclass(frozen=True, slots=True)
class GraphStatistics:
    """Exact statistics for a validated graph."""

    node_count: int
    edge_count: int
    leaf_count: int
    root_count: int
    max_depth: int

    @property
    def child_reference_count(self) -> int:
        """Return the number of explicit child references."""

        return self.edge_count


def _maximum_depth(graph: Graph) -> int:
    """Compute leaf-zero depth iteratively over a validated DAG."""

    depths: dict[str, int] = {}
    for root in graph.roots:
        stack: list[tuple[str, bool]] = [(root.target_id, False)]
        while stack:
            node_id, leaving = stack.pop()
            if node_id in depths:
                continue
            node = graph.nodes[node_id]
            if not leaving:
                stack.append((node_id, True))
                for child in reversed(node.children):
                    if child.target_id not in depths:
                        stack.append((child.target_id, False))
                continue
            depths[node_id] = (
                0
                if not node.children
                else 1 + max(depths[child.target_id] for child in node.children)
            )
    return max((depths[root.target_id] for root in graph.roots), default=0)


def compute_statistics(graph: Graph) -> GraphStatistics:
    """Return exact counts and depth for a valid graph.

    Invalid structures are rejected instead of being partially counted. This keeps
    missing nodes, cycles, and unreachable components visible to callers.
    """

    from geml.graph.validate import validate_graph

    validation = validate_graph(graph)
    if not validation.valid:
        raise ValueError(
            "cannot compute statistics for an invalid graph: " + "; ".join(validation.errors)
        )

    return GraphStatistics(
        node_count=len(graph.nodes),
        edge_count=sum(len(node.children) for node in graph.nodes.values()),
        leaf_count=sum(not node.children for node in graph.nodes.values()),
        root_count=len(graph.roots),
        max_depth=_maximum_depth(graph),
    )
