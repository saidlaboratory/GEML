"""Representation-neutral records for validated directed acyclic graphs."""

from __future__ import annotations

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
        """Snapshot child references so graph records cannot drift after creation."""

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
