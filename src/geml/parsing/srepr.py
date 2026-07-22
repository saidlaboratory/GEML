"""Safe parser for the authoritative enabled ``sympy_srepr`` subset."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from math import gcd

from geml.contracts.expression import ExpressionRecord
from geml.spec.operators import OPERATOR_REGISTRY

_SUPPORTED_CONSTRUCTORS = frozenset(
    {"Symbol", "Integer", "Rational", "Add", "Mul", "Pow", "exp", "log"}
)
_SUPPORTED_REGISTRY_OPERATORS = frozenset(
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
_SYMBOL_ASSUMPTIONS = frozenset({"real", "positive", "nonzero"})
_CANONICAL_INTEGER = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")


class SreprParseError(ValueError):
    """Authoritative structural text is malformed or outside parser limits."""


class UnsupportedNodeError(SreprParseError):
    """Structural text names a constructor outside the enabled registry subset."""

    def __init__(self, constructor: str) -> None:
        self.constructor = constructor
        super().__init__(f"unsupported srepr constructor: {constructor!r}")


@dataclass(frozen=True)
class ParserLimits:
    """Resource limits applied before and during syntax traversal."""

    maximum_source_characters: int = 1_000_000
    maximum_nodes: int = 4_096
    maximum_depth: int = 256
    maximum_integer_digits: int = 4_096

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum_source_characters", self.maximum_source_characters),
            ("maximum_nodes", self.maximum_nodes),
            ("maximum_depth", self.maximum_depth),
            ("maximum_integer_digits", self.maximum_integer_digits),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ParsedSreprNode:
    """Validated structural constructor with ordered expression children."""

    constructor: str
    children: tuple[ParsedSreprNode, ...] = ()
    value: str | int | tuple[int, int] | None = None
    assumptions: tuple[tuple[str, bool], ...] = ()


_DEFAULT_LIMITS = ParserLimits()


def _validate_registry_coverage() -> None:
    enabled = {
        name for name, operator in OPERATOR_REGISTRY.items() if operator.enabled_for_generation
    }
    unsupported = enabled - _SUPPORTED_REGISTRY_OPERATORS
    if unsupported:
        raise RuntimeError(
            "enabled operator registry exceeds the srepr parser: " + ", ".join(sorted(unsupported))
        )


def _integer_literal(node: ast.expr, *, source: str, maximum_digits: int) -> int:
    segment = ast.get_source_segment(source, node)
    if segment is None or _CANONICAL_INTEGER.fullmatch(segment) is None:
        raise SreprParseError("integer payloads must be canonical literal base-10 integers")
    digits = segment[1:] if segment.startswith("-") else segment
    if len(digits) > maximum_digits:
        raise SreprParseError("integer payload exceeds the configured digit limit")
    try:
        return int(segment, 10)
    except ValueError as error:
        raise SreprParseError("integer payload cannot be represented safely") from error


class _SafeParser:
    def __init__(self, source: str, limits: ParserLimits) -> None:
        self.source = source
        self.limits = limits
        self.node_count = 0

    def parse_node(self, node: ast.expr, *, depth: int) -> ParsedSreprNode:
        self.node_count += 1
        if self.node_count > self.limits.maximum_nodes:
            raise SreprParseError("srepr exceeds the configured node limit")
        if depth > self.limits.maximum_depth:
            raise SreprParseError("srepr exceeds the configured depth limit")
        if isinstance(node, ast.Name):
            raise UnsupportedNodeError(node.id)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise SreprParseError("every expression node must be a direct constructor call")

        constructor = node.func.id
        if constructor not in _SUPPORTED_CONSTRUCTORS:
            raise UnsupportedNodeError(constructor)
        if any(keyword.arg is None for keyword in node.keywords):
            raise SreprParseError("starred keyword arguments are not allowed")

        if constructor == "Symbol":
            return self._parse_symbol(node)
        if constructor == "Integer":
            return self._parse_integer(node)
        if constructor == "Rational":
            return self._parse_rational(node)
        return self._parse_operator(node, constructor=constructor, depth=depth)

    @staticmethod
    def _reject_keywords(node: ast.Call, constructor: str) -> None:
        if node.keywords:
            raise SreprParseError(f"{constructor} does not accept keyword arguments")

    def _parse_symbol(self, node: ast.Call) -> ParsedSreprNode:
        if len(node.args) != 1:
            raise SreprParseError("Symbol requires exactly one name argument")
        name_node = node.args[0]
        if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
            raise SreprParseError("Symbol name must be a literal string")
        if not name_node.value.strip():
            raise SreprParseError("Symbol name must be nonblank")

        assumptions: dict[str, bool] = {}
        for keyword in node.keywords:
            assumption = keyword.arg
            if assumption not in _SYMBOL_ASSUMPTIONS:
                raise SreprParseError(f"unsupported Symbol assumption: {assumption!r}")
            if assumption in assumptions:
                raise SreprParseError(f"duplicate Symbol assumption: {assumption!r}")
            if not isinstance(keyword.value, ast.Constant) or keyword.value.value is not True:
                raise SreprParseError("enabled Symbol assumptions must be literal True")
            assumptions[assumption] = True
        if not assumptions:
            raise SreprParseError("Symbol must declare real, positive, or nonzero assumptions")
        if "nonzero" in assumptions and not (
            assumptions.get("real") or assumptions.get("positive")
        ):
            raise SreprParseError("nonzero Symbol assumptions must also establish a real domain")
        return ParsedSreprNode(
            constructor="Symbol",
            value=name_node.value,
            assumptions=tuple(sorted(assumptions.items())),
        )

    def _parse_integer(self, node: ast.Call) -> ParsedSreprNode:
        self._reject_keywords(node, "Integer")
        if len(node.args) != 1:
            raise SreprParseError("Integer requires exactly one payload")
        return ParsedSreprNode(
            constructor="Integer",
            value=_integer_literal(
                node.args[0],
                source=self.source,
                maximum_digits=self.limits.maximum_integer_digits,
            ),
        )

    def _parse_rational(self, node: ast.Call) -> ParsedSreprNode:
        self._reject_keywords(node, "Rational")
        if len(node.args) != 2:
            raise SreprParseError("Rational requires numerator and denominator")
        numerator, denominator = (
            _integer_literal(
                argument,
                source=self.source,
                maximum_digits=self.limits.maximum_integer_digits,
            )
            for argument in node.args
        )
        if denominator < 2:
            raise SreprParseError(
                "canonical Rational denominator must be at least two; use Integer otherwise"
            )
        if gcd(abs(numerator), denominator) != 1:
            raise SreprParseError("Rational payload must already be in canonical lowest terms")
        return ParsedSreprNode(
            constructor="Rational",
            value=(numerator, denominator),
        )

    def _parse_operator(
        self,
        node: ast.Call,
        *,
        constructor: str,
        depth: int,
    ) -> ParsedSreprNode:
        self._reject_keywords(node, constructor)
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            raise SreprParseError("starred positional arguments are not allowed")
        arity = len(node.args)
        if constructor in {"Add", "Mul"} and arity < 2:
            raise SreprParseError(f"{constructor} requires at least two operands")
        if constructor == "Pow" and arity != 2:
            raise SreprParseError("Pow requires base and exponent")
        if constructor in {"exp", "log"} and arity != 1:
            raise SreprParseError(f"{constructor} requires exactly one argument")
        return ParsedSreprNode(
            constructor=constructor,
            children=tuple(self.parse_node(argument, depth=depth + 1) for argument in node.args),
        )


def parse_srepr(
    sympy_srepr: str,
    *,
    limits: ParserLimits = _DEFAULT_LIMITS,
) -> ParsedSreprNode:
    """Parse authoritative text without evaluating or importing anything from it."""

    _validate_registry_coverage()
    if not isinstance(sympy_srepr, str) or not sympy_srepr.strip():
        raise SreprParseError("sympy_srepr must be a nonblank string")
    if len(sympy_srepr) > limits.maximum_source_characters:
        raise SreprParseError("srepr exceeds the configured source-length limit")
    try:
        syntax = ast.parse(sympy_srepr, mode="eval")
    except (MemoryError, RecursionError, SyntaxError, ValueError) as error:
        raise SreprParseError("invalid Python-call syntax in sympy_srepr") from error
    try:
        return _SafeParser(sympy_srepr, limits).parse_node(syntax.body, depth=1)
    except RecursionError as error:
        raise SreprParseError("srepr traversal exceeds the runtime recursion limit") from error


def parse_expression_record(
    record: ExpressionRecord,
    *,
    limits: ParserLimits = _DEFAULT_LIMITS,
) -> ParsedSreprNode:
    """Parse the authoritative structural field of one frozen expression record."""

    return parse_srepr(record.sympy_srepr, limits=limits)
