"""Deterministic construction of frozen binary AST contracts from stored srepr."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import JsonValue

from geml.ast.statistics import calculate_statistics
from geml.contracts.ast import ASTEdge, ASTNode, ASTTree
from geml.contracts.expression import ExpressionRecord
from geml.parsing.srepr import (
    ParsedSreprNode,
    ParserLimits,
    parse_expression_record,
)

_OPERATOR_LABELS = {
    "Add": "add",
    "Mul": "multiply",
    "Pow": "power",
    "exp": "exp",
    "log": "log",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
}
_DEFAULT_LIMITS = ParserLimits()


@dataclass(frozen=True)
class _BinaryNode:
    node_kind: str
    label: str
    value: JsonValue = None
    children: tuple[_BinaryNode, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)


def _fold_left(constructor: str, children: tuple[_BinaryNode, ...]) -> _BinaryNode:
    source_arity = len(children)
    accumulated = children[0]
    for fold_step, child in enumerate(children[1:], start=1):
        accumulated = _BinaryNode(
            node_kind="operator",
            label=_OPERATOR_LABELS[constructor],
            children=(accumulated, child),
            metadata={
                "sympy_constructor": constructor,
                "binary_fold": "left",
                "source_arity": source_arity,
                "fold_step": fold_step,
            },
        )
    return accumulated


def _to_binary(node: ParsedSreprNode) -> _BinaryNode:
    if node.constructor == "Symbol":
        return _BinaryNode(
            node_kind="leaf",
            label="symbol",
            value={"name": node.value, "assumptions": dict(node.assumptions)},
            metadata={"sympy_constructor": "Symbol"},
        )
    if node.constructor == "Integer":
        return _BinaryNode(
            node_kind="leaf",
            label="one" if node.value == 1 else "integer",
            value=node.value,
            metadata={"sympy_constructor": "Integer"},
        )
    if node.constructor == "Rational":
        numerator, denominator = node.value  # type: ignore[misc]
        return _BinaryNode(
            node_kind="leaf",
            label="rational",
            value={"numerator": numerator, "denominator": denominator},
            metadata={"sympy_constructor": "Rational"},
        )

    children = tuple(_to_binary(child) for child in node.children)
    if node.constructor in {"Add", "Mul"}:
        return _fold_left(node.constructor, children)
    return _BinaryNode(
        node_kind="operator",
        label=_OPERATOR_LABELS[node.constructor],
        children=children,
        metadata={"sympy_constructor": node.constructor},
    )


def build_ast_from_parsed(parsed: ParsedSreprNode, *, expression_id: str) -> ASTTree:
    """Build stable pre-order nodes and ordered edges from validated syntax."""

    binary_root = _to_binary(parsed)
    nodes: list[ASTNode] = []
    edges: list[ASTEdge] = []
    node_ids: dict[int, str] = {}
    events: list[tuple[str, _BinaryNode, int, _BinaryNode | None]] = [
        ("visit", binary_root, -1, None)
    ]
    while events:
        action, node, child_slot, child = events.pop()
        if action == "edge":
            if child is None:  # pragma: no cover - internal event invariant
                raise RuntimeError("edge event requires a child")
            edges.append(
                ASTEdge(
                    source_id=node_ids[id(node)],
                    target_id=node_ids[id(child)],
                    child_slot=child_slot,
                )
            )
            continue

        node_id = f"n{len(nodes):06d}"
        node_ids[id(node)] = node_id
        nodes.append(
            ASTNode(
                node_id=node_id,
                node_kind=node.node_kind,
                label=node.label,
                arity=len(node.children),
                value=node.value,
                metadata=node.metadata,
            )
        )
        for slot in reversed(range(len(node.children))):
            child = node.children[slot]
            # LIFO events reproduce the former recursive ordering: the complete child subtree is
            # emitted before its parent edge, with left slots preceding right slots.
            events.append(("edge", node, slot, child))
            events.append(("visit", child, -1, None))

    root_id = node_ids[id(binary_root)]
    statistics = calculate_statistics(nodes, edges, root_id)
    return ASTTree(
        expression_id=expression_id,
        root_id=root_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        statistics=statistics,
    )


def build_ast(
    record: ExpressionRecord,
    *,
    limits: ParserLimits = _DEFAULT_LIMITS,
) -> ASTTree:
    """Parse and build one authoritative stored expression."""

    return build_ast_from_parsed(
        parse_expression_record(record, limits=limits),
        expression_id=record.expression_id,
    )
