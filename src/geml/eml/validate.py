"""Strict structural validation and exact tree statistics for pure EML."""

from __future__ import annotations

from dataclasses import dataclass

from geml.eml.ir import EMLTerm, One, Variable, is_eml_term, is_valid_source_variable_name


class PureEMLValidationError(ValueError):
    """A value is cyclic, malformed, or outside the pure EML vocabulary."""


@dataclass(frozen=True, slots=True)
class PureEMLStatistics:
    """Exact occurrence counts using the convention that a leaf has depth zero.

    ``reused_object_count`` reports syntactic occurrences backed by an immutable
    Python object already seen elsewhere in the traversal.  It makes physical
    reuse visible without assigning DAG identity to recursive value objects.
    """

    node_count: int
    edge_count: int
    leaf_count: int
    operator_count: int
    depth: int
    reused_object_count: int


def validate_pure_eml(root: EMLTerm, *, maximum_nodes: int | None = None) -> PureEMLStatistics:
    """Validate ``root`` and return exact expanded-tree statistics.

    A child slot is a syntactic occurrence, not an object-identity reference.
    Reusing one immutable value in multiple slots therefore counts as multiple
    tree occurrences.  Only identity reuse on the active ancestor path is a
    cycle.  Explicit node identity and sharing belong to Goal 3's DAG model.
    """

    if maximum_nodes is not None and (
        isinstance(maximum_nodes, bool) or not isinstance(maximum_nodes, int) or maximum_nodes < 1
    ):
        raise ValueError("maximum_nodes must be a positive integer or None")

    if root is None:
        raise PureEMLValidationError("pure EML root is missing")
    if isinstance(root, (list, tuple)):
        if not root:
            raise PureEMLValidationError("pure EML structure has no root")
        if len(root) > 1 and all(is_eml_term(candidate) for candidate in root):
            raise PureEMLValidationError("pure EML structure has multiple roots")

    node_count = 0
    leaf_count = 0
    operator_count = 0
    maximum_depth = 0
    active_ids: set[int] = set()
    seen_ids: set[int] = set()
    reused_object_count = 0
    events: list[tuple[EMLTerm, int, bool]] = [(root, 0, False)]

    while events:
        node, depth, leaving = events.pop()
        node_id = id(node)
        if leaving:
            active_ids.remove(node_id)
            continue
        if node_id in active_ids:
            raise PureEMLValidationError("pure EML structure contains a cycle")
        if node_id in seen_ids:
            reused_object_count += 1
        else:
            seen_ids.add(node_id)
        if not is_eml_term(node):
            raise PureEMLValidationError(f"forbidden pure EML node type: {type(node).__name__!r}")

        node_count += 1
        if maximum_nodes is not None and node_count > maximum_nodes:
            raise PureEMLValidationError("pure EML tree exceeds the configured node limit")
        maximum_depth = max(maximum_depth, depth)

        if isinstance(node, One):
            leaf_count += 1
            continue

        if isinstance(node, Variable):
            try:
                name = node.name
            except AttributeError as error:
                raise PureEMLValidationError("variable leaf is missing its name") from error
            if not is_valid_source_variable_name(name):
                raise PureEMLValidationError(
                    "variable leaf contains an invalid or compound source name"
                )
            leaf_count += 1
            continue

        operator_count += 1
        try:
            left = node.left
        except AttributeError as error:
            raise PureEMLValidationError("eml node is missing its left child") from error
        try:
            right = node.right
        except AttributeError as error:
            raise PureEMLValidationError("eml node is missing its right child") from error
        active_ids.add(node_id)
        events.append((node, depth, True))
        events.append((right, depth + 1, False))
        events.append((left, depth + 1, False))

    return PureEMLStatistics(
        node_count=node_count,
        edge_count=node_count - 1,
        leaf_count=leaf_count,
        operator_count=operator_count,
        depth=maximum_depth,
        reused_object_count=reused_object_count,
    )
