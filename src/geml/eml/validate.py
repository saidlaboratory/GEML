"""Strict structural validation and exact tree statistics for pure EML."""

from __future__ import annotations

from dataclasses import dataclass

from geml.eml.ir import EML, EMLTerm, One, Variable


class PureEMLValidationError(ValueError):
    """A value is cyclic, malformed, or outside the pure EML vocabulary."""


@dataclass(frozen=True, slots=True)
class PureEMLStatistics:
    """Exact occurrence counts using the convention that a leaf has depth zero."""

    node_count: int
    edge_count: int
    leaf_count: int
    operator_count: int
    depth: int


def validate_pure_eml(root: EMLTerm, *, maximum_nodes: int | None = None) -> PureEMLStatistics:
    """Validate ``root`` and return exact expanded-tree statistics.

    Reusing one immutable Python object in multiple child slots still represents
    multiple tree occurrences.  Only an identity encountered again on its active
    ancestor path is a cycle.  This preserves repeated source occurrences without
    introducing Goal 3 DAG semantics.
    """

    if maximum_nodes is not None and (
        isinstance(maximum_nodes, bool) or not isinstance(maximum_nodes, int) or maximum_nodes < 1
    ):
        raise ValueError("maximum_nodes must be a positive integer or None")

    node_count = 0
    leaf_count = 0
    operator_count = 0
    maximum_depth = 0
    active_ids: set[int] = set()
    events: list[tuple[EMLTerm, int, bool]] = [(root, 0, False)]

    while events:
        node, depth, leaving = events.pop()
        node_id = id(node)
        if leaving:
            active_ids.remove(node_id)
            continue
        if node_id in active_ids:
            raise PureEMLValidationError("pure EML structure contains a cycle")
        if type(node) not in {One, Variable, EML}:
            raise PureEMLValidationError(f"forbidden pure EML node type: {type(node).__name__!r}")

        node_count += 1
        if maximum_nodes is not None and node_count > maximum_nodes:
            raise PureEMLValidationError("pure EML tree exceeds the configured node limit")
        maximum_depth = max(maximum_depth, depth)

        if isinstance(node, (One, Variable)):
            leaf_count += 1
            continue

        operator_count += 1
        active_ids.add(node_id)
        events.append((node, depth, True))
        events.append((node.right, depth + 1, False))
        events.append((node.left, depth + 1, False))

    if node_count == 0:  # pragma: no cover - the typed root always contributes one node
        raise PureEMLValidationError("pure EML tree must have exactly one root")
    return PureEMLStatistics(
        node_count=node_count,
        edge_count=node_count - 1,
        leaf_count=leaf_count,
        operator_count=operator_count,
        depth=maximum_depth,
    )
