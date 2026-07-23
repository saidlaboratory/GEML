"""Deterministic, non-authoritative LaTeX views of frozen binary ASTs."""

from __future__ import annotations

import re
from typing import cast

from geml.contracts.ast import ASTNode, ASTTree
from geml.parsing.display import (
    DISPLAY_SOURCE_OPERATORS,
    MalformedDisplayTreeError,
    _negated_operand,
    _Precedence,
    _rational_parts,
    _reciprocal_base,
    _Rendered,
    _symbol_name,
    _TreeView,
)

LATEX_SOURCE_OPERATORS = DISPLAY_SOURCE_OPERATORS

_SINGLE_LETTER_SYMBOL = re.compile(r"[A-Za-z]\Z")
_LATEX_FUNCTION_COMMANDS = {
    "exp": "exp",
    "log": "log",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
}
_LATEX_ESCAPES = {
    "\\": r"\backslash{}",
    "{": r"\{",
    "}": r"\}",
    "_": r"\_",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "&": r"\&",
}


class LatexRenderError(ValueError):
    """Base error for AST-to-LaTeX rendering failures."""


class MalformedLatexTreeError(LatexRenderError):
    """The supplied records do not form the validated frozen AST vocabulary."""


class UnsupportedLatexNodeError(LatexRenderError):
    """A structurally valid AST node has no approved LaTeX interpretation."""

    def __init__(self, node: ASTNode) -> None:
        self.node_id = node.node_id
        self.label = node.label
        super().__init__(f"unsupported LaTeX node {node.node_id!r} with label {node.label!r}")


def _latex_group(rendered: _Rendered, needed: bool) -> str:
    return rf"\left({rendered.text}\right)" if needed else rendered.text


def _escape_symbol(name: str) -> str:
    def escape(character: str) -> str:
        if character.isspace() or not character.isprintable() or character in {"^", "~"}:
            return rf"\langle\mathtt{{U+{ord(character):04X}}}\rangle"
        return _LATEX_ESCAPES.get(character, character)

    escaped = "".join(escape(character) for character in name)
    return escaped if _SINGLE_LETTER_SYMBOL.fullmatch(name) else rf"\mathrm{{{escaped}}}"


class _LatexRenderer:
    def __init__(self, view: _TreeView) -> None:
        self.view = view

    def render(self, node: ASTNode) -> _Rendered:
        if node.node_kind == "leaf":
            return self._render_leaf(node)
        if node.node_kind != "operator":
            raise UnsupportedLatexNodeError(node)
        if node.label == "add":
            return self._render_add(node)
        if node.label == "multiply":
            return self._render_multiply(node)
        if node.label == "power":
            return self._render_power(node)
        if node.label in _LATEX_FUNCTION_COMMANDS:
            argument = self.render(self.view.children(node)[0])
            command = _LATEX_FUNCTION_COMMANDS[node.label]
            return _Rendered(
                rf"\{command}\left({argument.text}\right)",
                _Precedence.FUNCTION,
            )
        raise UnsupportedLatexNodeError(node)

    def _render_leaf(self, node: ASTNode) -> _Rendered:
        if node.label == "symbol":
            name = _symbol_name(node)
            return _Rendered(_escape_symbol(name), _Precedence.ATOM)
        if node.label == "one":
            return _Rendered("1", _Precedence.ATOM)
        if node.label == "integer":
            value = cast(int, node.value)
            precedence = _Precedence.NEGATE if value < 0 else _Precedence.ATOM
            return _Rendered(str(value), precedence)
        if node.label == "rational":
            numerator, denominator = _rational_parts(node)
            if numerator < 0:
                return _Rendered(
                    rf"-\frac{{{-numerator}}}{{{denominator}}}",
                    _Precedence.NEGATE,
                )
            return _Rendered(rf"\frac{{{numerator}}}{{{denominator}}}", _Precedence.ATOM)
        raise UnsupportedLatexNodeError(node)

    def _render_add(self, node: ASTNode) -> _Rendered:
        left_node, right_node = self.view.children(node)
        negated = _negated_operand(self.view, right_node)
        left = self.render(left_node)
        right = self.render(negated if negated is not None else right_node)
        operator = "-" if negated is not None else "+"
        left_text = _latex_group(left, left.precedence < _Precedence.ADD)
        right_text = _latex_group(right, right.precedence <= _Precedence.ADD)
        return _Rendered(f"{left_text} {operator} {right_text}", _Precedence.ADD)

    def _render_multiply(self, node: ASTNode) -> _Rendered:
        left_node, right_node = self.view.children(node)
        denominator = _reciprocal_base(self.view, right_node)
        if denominator is not None:
            numerator = self.render(left_node)
            rendered_denominator = self.render(denominator)
            return _Rendered(
                rf"\frac{{{numerator.text}}}{{{rendered_denominator.text}}}",
                _Precedence.MULTIPLY,
            )

        negated = _negated_operand(self.view, node)
        if negated is not None:
            operand = self.render(negated)
            text = _latex_group(operand, operand.precedence <= _Precedence.NEGATE)
            return _Rendered(f"-{text}", _Precedence.NEGATE)

        left = self.render(left_node)
        right = self.render(right_node)
        left_text = _latex_group(left, left.precedence < _Precedence.MULTIPLY)
        right_text = _latex_group(right, right.precedence <= _Precedence.MULTIPLY)
        return _Rendered(rf"{left_text} \cdot {right_text}", _Precedence.MULTIPLY)

    def _render_power(self, node: ASTNode) -> _Rendered:
        base_node, exponent_node = self.view.children(node)
        base = self.render(base_node)
        exponent = self.render(exponent_node)
        base_text = _latex_group(base, base.precedence <= _Precedence.POWER)
        return _Rendered(rf"{base_text}^{{{exponent.text}}}", _Precedence.POWER)


def render_latex(tree: ASTTree) -> str:
    """Emit stable LaTeX from ordered AST child slots without simplifying the tree."""

    try:
        view = _TreeView.from_ast(tree)
    except MalformedDisplayTreeError as error:
        raise MalformedLatexTreeError(str(error)) from error
    try:
        return _LatexRenderer(view).render(view.node_by_id[view.root_id]).text
    except RecursionError as error:
        raise LatexRenderError("AST exceeds the LaTeX renderer recursion limit") from error
