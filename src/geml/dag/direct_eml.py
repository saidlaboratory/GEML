"""Direct official pure-EML DAG construction with structural interning."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import gcd

from geml.contracts.ast import ASTNode, ASTTree
from geml.dag.hashcons import HashConsTable, InternedNode
from geml.eml.compiler_core import CompilerMode, require_compiler_mode
from geml.eml.ir import is_valid_source_variable_name
from geml.graph.schema import (
    EML_FAMILY,
    EML_ONE_KIND,
    EML_OPERATOR_KIND,
    EML_VARIABLE_KIND,
    Graph,
)
from geml.spec.operators import OPERATORS, EMLConstructionStatus

DIRECT_SOURCE_OPERATORS = frozenset(
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
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
    }
)


class UnsupportedASTOperatorError(ValueError):
    """A validated AST uses an operator outside the direct compiler dispatch."""


@dataclass(frozen=True, slots=True)
class ConstructionStats:
    """Direct-construction telemetry and representation identity."""

    elapsed_seconds: float
    peak_interning_table_size: int
    final_node_count: int
    intern_requests: int
    cache_hits: int
    compiler_mode: CompilerMode
    representation_mode: str
    construction_path: str = "direct_hashcons"


class DirectEMLCompiler:
    """Mode-bound memoized counterparts of the authoritative Goal 2 formulas."""

    def __init__(
        self,
        *,
        mode: CompilerMode = CompilerMode.OFFICIAL_V4,
        table: HashConsTable | None = None,
    ) -> None:
        self.mode = require_compiler_mode(mode)
        self.table = table if table is not None else HashConsTable(EML_FAMILY)
        if self.table.family != EML_FAMILY:
            raise ValueError("DirectEMLCompiler requires an eml-family interning table")

    def emit_variable(self, name: str) -> InternedNode:
        if not is_valid_source_variable_name(name):
            raise ValueError("EML variable names must be nonblank ASCII identifiers")
        return self.table.intern(
            kind=EML_VARIABLE_KIND,
            label=name,
            value=name,
        )

    def emit_one(self) -> InternedNode:
        return self.table.intern(
            kind=EML_ONE_KIND,
            label="1",
            value=1,
        )

    def emit_primitive(
        self,
        left: InternedNode,
        right: InternedNode,
    ) -> InternedNode:
        return self.table.intern(
            kind=EML_OPERATOR_KIND,
            label="eml",
            children=(left, right),
        )

    def emit_exp(self, value: InternedNode) -> InternedNode:
        return self.emit_primitive(value, self.emit_one())

    def emit_log(self, value: InternedNode) -> InternedNode:
        return self.emit_primitive(
            self.emit_one(),
            self.emit_exp(self.emit_primitive(self.emit_one(), value)),
        )

    def emit_zero(self) -> InternedNode:
        return self.emit_log(self.emit_one())

    def emit_subtract(
        self,
        left: InternedNode,
        right: InternedNode,
    ) -> InternedNode:
        return self.emit_primitive(self.emit_log(left), self.emit_exp(right))

    def emit_negate(self, value: InternedNode) -> InternedNode:
        if self.mode is CompilerMode.OFFICIAL_V4:
            return self.emit_subtract(self.emit_zero(), value)

        e_for_offset = self.emit_exp(self.emit_one())
        e_minus_one = self.emit_subtract(e_for_offset, self.emit_one())
        e_for_sum = self.emit_exp(self.emit_one())
        one_plus_value = self.emit_subtract(
            e_for_sum,
            self.emit_subtract(e_minus_one, value),
        )
        return self.emit_subtract(self.emit_one(), one_plus_value)

    def emit_add(
        self,
        left: InternedNode,
        right: InternedNode,
    ) -> InternedNode:
        return self.emit_subtract(left, self.emit_negate(right))

    def emit_inverse(self, value: InternedNode) -> InternedNode:
        return self.emit_exp(self.emit_negate(self.emit_log(value)))

    def emit_multiply(
        self,
        left: InternedNode,
        right: InternedNode,
    ) -> InternedNode:
        return self.emit_exp(self.emit_add(self.emit_log(left), self.emit_log(right)))

    def emit_divide(
        self,
        numerator: InternedNode,
        denominator: InternedNode,
    ) -> InternedNode:
        return self.emit_multiply(numerator, self.emit_inverse(denominator))

    def emit_power(
        self,
        base: InternedNode,
        exponent: InternedNode,
    ) -> InternedNode:
        return self.emit_exp(self.emit_multiply(exponent, self.emit_log(base)))

    def emit_integer(self, value: int) -> InternedNode:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("integer value must be an int, not bool")
        if value == 1:
            return self.emit_one()
        if value == 0:
            return self.emit_zero()
        if value < 0:
            return self.emit_negate(self.emit_integer(-value))

        accumulator: InternedNode | None = None
        term = self.emit_one()
        remaining = value
        while remaining:
            if remaining & 1:
                accumulator = term if accumulator is None else self.emit_add(accumulator, term)
            remaining >>= 1
            if remaining:
                term = self.emit_add(term, term)
        if accumulator is None:  # pragma: no cover - positive input always sets it
            raise RuntimeError("integer compiler failed to initialize its accumulator")
        return accumulator

    def emit_rational(self, numerator: int, denominator: int) -> InternedNode:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (numerator, denominator)
        ):
            raise TypeError("rational numerator and denominator must be ints, not bools")
        if denominator < 1:
            raise ValueError("rational denominator must be positive")
        if numerator == 0 and denominator != 1:
            raise ValueError("zero must use the canonical denominator one")
        if gcd(abs(numerator), denominator) != 1:
            raise ValueError("rational input must already be in canonical lowest terms")
        if denominator == 1:
            return self.emit_integer(numerator)

        result = self.emit_multiply(
            self.emit_integer(abs(numerator)),
            self.emit_inverse(self.emit_integer(denominator)),
        )
        return result if numerator >= 0 else self.emit_negate(result)

    def emit_decimal(self, value: str | Decimal | float) -> InternedNode:
        if isinstance(value, bool) or not isinstance(value, (str, Decimal, float)):
            raise TypeError("decimal value must be str, Decimal, or float")
        try:
            decimal = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise ValueError("decimal value must be a finite base-10 number") from error
        if not decimal.is_finite():
            raise ValueError("decimal value must be finite")
        numerator, denominator = decimal.as_integer_ratio()
        return self.emit_rational(numerator, denominator)

    def _emit_internal_i_branch(self) -> InternedNode:
        minus_one = self.emit_negate(self.emit_one())
        half_log = self.emit_divide(
            self.emit_log(minus_one),
            self.emit_integer(2),
        )
        return self.emit_negate(self.emit_exp(half_log))

    def _oscillatory_terms(
        self,
        value: InternedNode,
    ) -> tuple[InternedNode, InternedNode, InternedNode, InternedNode]:
        internal_i = self._emit_internal_i_branch()
        minus_one = self.emit_integer(-1)
        positive_exponent = self.emit_multiply(internal_i, value)
        negative_exponent = self.emit_multiply(
            self.emit_multiply(minus_one, internal_i),
            value,
        )
        return (
            internal_i,
            minus_one,
            self.emit_exp(positive_exponent),
            self.emit_exp(negative_exponent),
        )

    def emit_sin(self, value: InternedNode) -> InternedNode:
        internal_i, minus_one, exp_positive, exp_negative = self._oscillatory_terms(value)
        difference = self.emit_add(
            self.emit_multiply(minus_one, exp_negative),
            exp_positive,
        )
        coefficient = self.emit_multiply(
            self.emit_rational(-1, 2),
            internal_i,
        )
        return self.emit_multiply(coefficient, difference)

    def emit_cos(self, value: InternedNode) -> InternedNode:
        _, _, exp_positive, exp_negative = self._oscillatory_terms(value)
        half = self.emit_rational(1, 2)
        negative_term = self.emit_multiply(half, exp_negative)
        positive_term = self.emit_multiply(half, exp_positive)
        return self.emit_add(negative_term, positive_term)

    def emit_tan(self, value: InternedNode) -> InternedNode:
        internal_i, minus_one, exp_positive, exp_negative = self._oscillatory_terms(value)
        denominator = self.emit_add(exp_negative, exp_positive)
        reciprocal = self.emit_power(denominator, minus_one)
        numerator = self.emit_add(
            exp_negative,
            self.emit_multiply(minus_one, exp_positive),
        )
        return self.emit_multiply(
            self.emit_multiply(internal_i, reciprocal),
            numerator,
        )

    def _emit_double(self, value: InternedNode) -> InternedNode:
        return self.emit_add(value, value)

    def emit_sinh(self, value: InternedNode) -> InternedNode:
        exp_two_value = self.emit_exp(self._emit_double(value))
        exp_value = self.emit_exp(value)
        denominator = self.emit_multiply(self.emit_integer(2), exp_value)
        return self.emit_divide(
            self.emit_subtract(exp_two_value, self.emit_one()),
            denominator,
        )

    def emit_cosh(self, value: InternedNode) -> InternedNode:
        exp_two_value = self.emit_exp(self._emit_double(value))
        exp_value = self.emit_exp(value)
        denominator = self.emit_multiply(self.emit_integer(2), exp_value)
        return self.emit_divide(
            self.emit_add(exp_two_value, self.emit_one()),
            denominator,
        )

    def emit_tanh(self, value: InternedNode) -> InternedNode:
        exp_two_value = self.emit_exp(self._emit_double(value))
        return self.emit_divide(
            self.emit_subtract(exp_two_value, self.emit_one()),
            self.emit_add(exp_two_value, self.emit_one()),
        )


type DirectBuilder = Callable[[DirectEMLCompiler], InternedNode]


def compile_with_stats(
    build: DirectBuilder,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
    root_id: str = "root",
) -> tuple[Graph, str, ConstructionStats]:
    """Build one direct DAG, validate it, and return construction telemetry."""

    if not callable(build):
        raise TypeError("build must be callable")
    compiler = DirectEMLCompiler(mode=mode)
    start = time.perf_counter()
    root = build(compiler)
    elapsed_seconds = time.perf_counter() - start
    representation_mode = f"pure_eml:{compiler.mode.value}"
    graph = compiler.table.to_graph(
        root,
        root_id=root_id,
        representation_mode=representation_mode,
    )
    return (
        graph,
        root.node_id,
        ConstructionStats(
            elapsed_seconds=elapsed_seconds,
            peak_interning_table_size=compiler.table.peak_size,
            final_node_count=len(graph.nodes),
            intern_requests=compiler.table.intern_requests,
            cache_hits=compiler.table.cache_hits,
            compiler_mode=compiler.mode,
            representation_mode=representation_mode,
        ),
    )


def _ordered_children(tree: ASTTree) -> dict[str, tuple[str, ...]]:
    children: dict[str, dict[int, str]] = {node.node_id: {} for node in tree.nodes}
    for edge in tree.edges:
        children[edge.source_id][edge.child_slot] = edge.target_id
    return {
        node.node_id: tuple(children[node.node_id][slot] for slot in range(node.arity))
        for node in tree.nodes
    }


def _integer_value(node: ASTNode) -> int:
    if isinstance(node.value, bool) or not isinstance(node.value, int):
        raise ValueError(f"AST {node.label!r} leaf must contain an integer payload")
    return node.value


def _compile_ast(
    tree: ASTTree,
    compiler: DirectEMLCompiler,
) -> InternedNode:
    node_by_id = {node.node_id: node for node in tree.nodes}
    children = _ordered_children(tree)
    values: dict[str, InternedNode] = {}
    unary = {
        "negate": compiler.emit_negate,
        "exp": compiler.emit_exp,
        "log": compiler.emit_log,
        "sin": compiler.emit_sin,
        "cos": compiler.emit_cos,
        "tan": compiler.emit_tan,
        "sinh": compiler.emit_sinh,
        "cosh": compiler.emit_cosh,
        "tanh": compiler.emit_tanh,
    }
    binary = {
        "add": compiler.emit_add,
        "subtract": compiler.emit_subtract,
        "multiply": compiler.emit_multiply,
        "divide": compiler.emit_divide,
        "power": compiler.emit_power,
    }

    events: list[tuple[str, bool]] = [(tree.root_id, False)]
    while events:
        node_id, leaving = events.pop()
        node = node_by_id[node_id]
        if not leaving:
            events.append((node_id, True))
            events.extend((child_id, False) for child_id in reversed(children[node_id]))
            continue

        child_values = tuple(values[child_id] for child_id in children[node_id])
        if node.arity == 0:
            if node.label == "symbol":
                if not isinstance(node.value, dict) or not isinstance(node.value.get("name"), str):
                    raise ValueError("symbol AST leaf has no valid source name")
                result = compiler.emit_variable(node.value["name"])
            elif node.label == "one":
                if _integer_value(node) != 1:
                    raise ValueError("one AST leaf must contain the exact integer one")
                result = compiler.emit_one()
            elif node.label == "integer":
                result = compiler.emit_integer(_integer_value(node))
            elif node.label == "rational":
                if not isinstance(node.value, dict):
                    raise ValueError("rational AST leaf must contain numerator/denominator payload")
                numerator = node.value.get("numerator")
                denominator = node.value.get("denominator")
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (numerator, denominator)
                ):
                    raise ValueError("rational AST payload must contain exact integers")
                result = compiler.emit_rational(numerator, denominator)
            else:
                raise UnsupportedASTOperatorError(node.label)
        elif node.arity == 1:
            try:
                result = unary[node.label](child_values[0])
            except KeyError as error:
                raise UnsupportedASTOperatorError(node.label) from error
        elif node.arity == 2:
            try:
                result = binary[node.label](child_values[0], child_values[1])
            except KeyError as error:
                raise UnsupportedASTOperatorError(node.label) from error
        else:  # pragma: no cover - AST contracts restrict arity
            raise UnsupportedASTOperatorError(node.label)
        values[node_id] = result

    return values[tree.root_id]


def _validate_registry_coverage() -> None:
    enabled_approved = {
        operator.name
        for operator in OPERATORS
        if operator.enabled_for_generation
        and operator.eml_construction_status is EMLConstructionStatus.APPROVED
    }
    if enabled_approved != DIRECT_SOURCE_OPERATORS:
        missing = sorted(enabled_approved - DIRECT_SOURCE_OPERATORS)
        extra = sorted(DIRECT_SOURCE_OPERATORS - enabled_approved)
        raise RuntimeError(
            f"direct compiler registry coverage mismatch; missing={missing}, extra={extra}"
        )


def compile_ast_to_eml_dag(
    tree: ASTTree,
    *,
    mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> tuple[Graph, str, ConstructionStats]:
    """Compile a validated AST directly into the exact pure EML DAG."""

    if not isinstance(tree, ASTTree):
        raise TypeError("tree must be a validated ASTTree")
    _validate_registry_coverage()
    return compile_with_stats(
        lambda compiler: _compile_ast(tree, compiler),
        mode=mode,
        root_id=tree.expression_id,
    )
