"""Deterministic, non-authoritative plain-text views of frozen binary ASTs.

The renderer follows explicit child slots and never uses its output as source structure.
Subtraction, division, and negation are presentation interpretations of the lowering patterns
frozen by the operator registry; the underlying AST remains authoritative.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import IntEnum
from math import gcd
from typing import cast

from geml.ast.statistics import ASTStructureError, calculate_statistics
from geml.contracts.ast import ASTNode, ASTTree
from geml.spec.operators import OPERATOR_REGISTRY, EMLConstructionStatus

DISPLAY_SOURCE_OPERATORS = frozenset(
    {
        "symbol",
        "one",
        "integer",
        "rational",
        "add",
        "subtract",
        "multiply",
        "divide",
        "negate",
        "power",
        "exp",
        "log",
    }
)

_SIMPLE_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SYMBOL_ASSUMPTIONS = frozenset({"real", "positive", "nonzero"})
_LEAF_LABELS = frozenset({"symbol", "one", "integer", "rational"})
_OPERATOR_ARITIES = {"add": 2, "multiply": 2, "power": 2, "exp": 1, "log": 1}


class DisplayRenderError(ValueError):
    """Base error for AST display rendering failures."""


class MalformedDisplayTreeError(DisplayRenderError):
    """The supplied records do not form the validated frozen AST vocabulary."""


class UnsupportedDisplayNodeError(DisplayRenderError):
    """A structurally valid AST node has no approved display interpretation."""

    def __init__(self, node: ASTNode) -> None:
        self.node_id = node.node_id
        self.label = node.label
        super().__init__(f"unsupported display node {node.node_id!r} with label {node.label!r}")


class _Precedence(IntEnum):
    ADD = 10
    MULTIPLY = 20
    NEGATE = 30
    POWER = 40
    FUNCTION = 50
    ATOM = 60


@dataclass(frozen=True)
class _Rendered:
    text: str
    precedence: _Precedence


class _NodeShapeError(ValueError):
    """A known AST label has fields inconsistent with the frozen builder vocabulary."""


def _validate_node_shape(node: ASTNode) -> None:
    try:
        node_id = node.node_id
        node_kind = node.node_kind
        label = node.label
        arity = node.arity
        value = node.value
        metadata = node.metadata
    except AttributeError as error:
        raise _NodeShapeError("AST node is missing a required field") from error
    if not isinstance(node_id, str) or not node_id.strip():
        raise _NodeShapeError("AST node_id must be a nonblank string")
    if not isinstance(node_kind, str) or not node_kind.strip():
        raise _NodeShapeError("AST node_kind must be a nonblank string")
    if not isinstance(label, str) or not label.strip():
        raise _NodeShapeError("AST label must be a nonblank string")
    if isinstance(arity, bool) or not isinstance(arity, int) or not 0 <= arity <= 2:
        raise _NodeShapeError("AST arity must be an integer from zero through two")
    if not isinstance(metadata, dict):
        raise _NodeShapeError("AST metadata must be an object")

    if label in _LEAF_LABELS:
        if node_kind != "leaf" or arity != 0:
            raise _NodeShapeError(f"known leaf {label!r} must have leaf kind and arity zero")
        if label == "symbol":
            _symbol_name(node)
        elif label == "one" and _integer_value(node) != 1:
            raise _NodeShapeError("one leaf must contain exact integer 1")
        elif label == "integer" and _integer_value(node) is None:
            raise _NodeShapeError("integer leaf must contain a non-one exact integer")
        elif label == "rational":
            _rational_parts(node)
        return
    if label in _OPERATOR_ARITIES:
        expected_arity = _OPERATOR_ARITIES[label]
        if node_kind != "operator" or arity != expected_arity or value is not None:
            raise _NodeShapeError(
                f"known operator {label!r} must have operator kind, arity "
                f"{expected_arity}, and a null value"
            )


@dataclass(frozen=True)
class _TreeView:
    node_by_id: dict[str, ASTNode]
    children_by_id: dict[str, tuple[str, ...]]
    root_id: str

    @classmethod
    def from_ast(cls, tree: ASTTree) -> _TreeView:
        if not isinstance(tree, ASTTree):
            raise MalformedDisplayTreeError("display rendering requires an ASTTree")
        try:
            statistics = calculate_statistics(tree.nodes, tree.edges, tree.root_id)
            if statistics != tree.statistics:
                raise MalformedDisplayTreeError("stored AST statistics do not match its structure")

            node_by_id = {node.node_id: node for node in tree.nodes}
            for node in tree.nodes:
                _validate_node_shape(node)
            child_slots: dict[str, list[str | None]] = {
                node.node_id: [None] * node.arity for node in tree.nodes
            }
            for edge in tree.edges:
                child_slots[edge.source_id][edge.child_slot] = edge.target_id
            if any(child is None for children in child_slots.values() for child in children):
                raise MalformedDisplayTreeError("AST child slots are incomplete")
            return cls(
                node_by_id=node_by_id,
                children_by_id={
                    node_id: tuple(cast(str, child) for child in children)
                    for node_id, children in child_slots.items()
                },
                root_id=tree.root_id,
            )
        except MalformedDisplayTreeError:
            raise
        except (
            ASTStructureError,
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise MalformedDisplayTreeError(str(error)) from error

    def children(self, node: ASTNode) -> tuple[ASTNode, ...]:
        return tuple(self.node_by_id[node_id] for node_id in self.children_by_id[node.node_id])


def _approved_enabled_operators() -> frozenset[str]:
    return frozenset(
        name
        for name, operator in OPERATOR_REGISTRY.items()
        if operator.enabled_for_generation
        and operator.eml_construction_status is EMLConstructionStatus.APPROVED
    )


def _validate_operator_coverage() -> None:
    missing = _approved_enabled_operators() - DISPLAY_SOURCE_OPERATORS
    if missing:
        raise RuntimeError(
            "display renderer is missing approved operators: " + ", ".join(sorted(missing))
        )


def _integer_value(node: ASTNode) -> int | None:
    if node.node_kind != "leaf" or node.arity != 0:
        return None
    if not isinstance(node.value, int) or isinstance(node.value, bool):
        return None
    if node.label == "one" and node.value == 1:
        return 1
    if node.label == "integer" and node.value != 1:
        return node.value
    return None


def _symbol_name(node: ASTNode) -> str:
    if not isinstance(node.value, dict) or set(node.value) != {"name", "assumptions"}:
        raise _NodeShapeError("symbol value must contain a name and assumptions")
    name = node.value["name"]
    assumptions = node.value["assumptions"]
    if not isinstance(name, str) or not name.strip() or not isinstance(assumptions, dict):
        raise _NodeShapeError("symbol value must contain a name and assumptions")
    if not assumptions or any(
        key not in _SYMBOL_ASSUMPTIONS or value is not True for key, value in assumptions.items()
    ):
        raise _NodeShapeError("symbol assumptions are outside the enabled AST policy")
    if "nonzero" in assumptions and not (assumptions.get("real") or assumptions.get("positive")):
        raise _NodeShapeError("nonzero symbol assumptions must establish a real domain")
    return name


def _rational_parts(node: ASTNode) -> tuple[int, int]:
    if not isinstance(node.value, dict) or set(node.value) != {"numerator", "denominator"}:
        raise _NodeShapeError("rational value must contain numerator and denominator")
    numerator = node.value["numerator"]
    denominator = node.value["denominator"]
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or denominator < 2
        or gcd(abs(numerator), denominator) != 1
    ):
        raise _NodeShapeError("rational payload must contain exact canonical integers")
    return numerator, denominator


def _negated_operand(view: _TreeView, node: ASTNode) -> ASTNode | None:
    if node.node_kind != "operator" or node.label != "multiply" or node.arity != 2:
        return None
    left, right = view.children(node)
    return right if _integer_value(left) == -1 else None


def _reciprocal_base(view: _TreeView, node: ASTNode) -> ASTNode | None:
    if node.node_kind != "operator" or node.label != "power" or node.arity != 2:
        return None
    base, exponent = view.children(node)
    return base if _integer_value(exponent) == -1 else None


def _parenthesize(rendered: _Rendered, needed: bool) -> str:
    return f"({rendered.text})" if needed else rendered.text


class _DisplayRenderer:
    def __init__(self, view: _TreeView) -> None:
        self.view = view

    def render(self, node: ASTNode) -> _Rendered:
        if node.node_kind == "leaf":
            return self._render_leaf(node)
        if node.node_kind != "operator":
            raise UnsupportedDisplayNodeError(node)
        if node.label == "add":
            return self._render_add(node)
        if node.label == "multiply":
            return self._render_multiply(node)
        if node.label == "power":
            return self._render_power(node)
        if node.label in {"exp", "log"}:
            argument = self.render(self.view.children(node)[0])
            return _Rendered(f"{node.label}({argument.text})", _Precedence.FUNCTION)
        raise UnsupportedDisplayNodeError(node)

    def _render_leaf(self, node: ASTNode) -> _Rendered:
        if node.label == "symbol":
            name = _symbol_name(node)
            text = (
                name
                if _SIMPLE_SYMBOL.fullmatch(name)
                else f"Symbol({json.dumps(name, ensure_ascii=False)})"
            )
            return _Rendered(text, _Precedence.ATOM)
        if node.label == "one":
            return _Rendered("1", _Precedence.ATOM)
        if node.label == "integer":
            value = cast(int, node.value)
            precedence = _Precedence.NEGATE if value < 0 else _Precedence.ATOM
            return _Rendered(str(value), precedence)
        if node.label == "rational":
            numerator, denominator = _rational_parts(node)
            return _Rendered(f"{numerator}/{denominator}", _Precedence.MULTIPLY)
        raise UnsupportedDisplayNodeError(node)

    def _render_add(self, node: ASTNode) -> _Rendered:
        left_node, right_node = self.view.children(node)
        negated = _negated_operand(self.view, right_node)
        left = self.render(left_node)
        right = self.render(negated if negated is not None else right_node)
        operator = "-" if negated is not None else "+"
        left_text = _parenthesize(left, left.precedence < _Precedence.ADD)
        right_text = _parenthesize(right, right.precedence <= _Precedence.ADD)
        return _Rendered(f"{left_text} {operator} {right_text}", _Precedence.ADD)

    def _render_multiply(self, node: ASTNode) -> _Rendered:
        left_node, right_node = self.view.children(node)
        denominator = _reciprocal_base(self.view, right_node)
        if denominator is not None:
            numerator = self.render(left_node)
            rendered_denominator = self.render(denominator)
            left_text = _parenthesize(numerator, numerator.precedence < _Precedence.MULTIPLY)
            right_text = _parenthesize(
                rendered_denominator,
                rendered_denominator.precedence <= _Precedence.MULTIPLY,
            )
            return _Rendered(f"{left_text} / {right_text}", _Precedence.MULTIPLY)

        negated = _negated_operand(self.view, node)
        if negated is not None:
            operand = self.render(negated)
            text = _parenthesize(operand, operand.precedence <= _Precedence.NEGATE)
            return _Rendered(f"-{text}", _Precedence.NEGATE)

        left = self.render(left_node)
        right = self.render(right_node)
        left_text = _parenthesize(left, left.precedence < _Precedence.MULTIPLY)
        right_text = _parenthesize(right, right.precedence <= _Precedence.MULTIPLY)
        return _Rendered(f"{left_text} * {right_text}", _Precedence.MULTIPLY)

    def _render_power(self, node: ASTNode) -> _Rendered:
        base_node, exponent_node = self.view.children(node)
        base = self.render(base_node)
        exponent = self.render(exponent_node)
        base_text = _parenthesize(base, base.precedence <= _Precedence.POWER)
        exponent_text = _parenthesize(exponent, exponent.precedence < _Precedence.POWER)
        return _Rendered(f"{base_text}**{exponent_text}", _Precedence.POWER)


def render_display(tree: ASTTree) -> str:
    """Render a stable readable view without changing or superseding the AST."""

    view = _TreeView.from_ast(tree)
    try:
        return _DisplayRenderer(view).render(view.node_by_id[view.root_id]).text
    except RecursionError as error:
        raise DisplayRenderError("AST exceeds the display renderer recursion limit") from error


_validate_operator_coverage()
