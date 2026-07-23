"""Validation pipeline for extracted candidates.

This module compiles each extracted :class:`~geml.egraph.candidates.Candidate` through the
official Pure EML compiler, builds its source AST, and verifies it against a reference
expression using only frozen Goal 2 primitives.  No candidate is ever silently discarded:
a compilation, semantic, or domain failure produces a retained row carrying an explicit
status and reason.

The converters map the closed e-graph operator vocabulary onto the official constructors
and the frozen AST contract.  They introduce no derived, macro, or synthetic nodes: every
EML node is produced by the authoritative ``geml.eml.compiler_*`` formulas, and every AST
node uses the labels the frozen direct compiler already dispatches on.

Semantic and domain verification are numeric.  A deterministic, domain-aware set of exact
sample points is chosen per variable, both expressions are evaluated with the frozen
:func:`geml.verification.eml.numeric.evaluate_pure_eml`, and the candidate is compared to
the reference: a candidate that disagrees where both are finite is a semantic mismatch, and
a candidate whose defined domain differs from the reference is a domain failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction

from geml.ast.statistics import calculate_statistics
from geml.contracts.ast import ASTEdge, ASTNode, ASTTree
from geml.egraph.candidates import Candidate
from geml.egraph.ir import EClassId, Expr, Operator
from geml.egraph.policy import RewriteMode
from geml.egraph.rewrite_engine import Assumption, AssumptionEnvironment
from geml.eml.compiler_arithmetic import (
    eml_divide,
    eml_integer,
    eml_multiply,
    eml_power,
    eml_rational,
)
from geml.eml.compiler_core import (
    CompilerMode,
    eml_add,
    eml_exp,
    eml_log,
    eml_negate,
    eml_subtract,
)
from geml.eml.ir import EMLTerm, variable
from geml.verification.eml.numeric import NumericBackend, evaluate_pure_eml

_AST_LEAF_KIND = "leaf"
_AST_OPERATOR_KIND = "operator"

_BINARY_LABELS: dict[Operator, str] = {
    Operator.ADD: "add",
    Operator.MUL: "multiply",
    Operator.SUB: "subtract",
    Operator.DIV: "divide",
    Operator.POW: "power",
}
_UNARY_LABELS: dict[Operator, str] = {
    Operator.NEG: "negate",
    Operator.EXP: "exp",
    Operator.LOG: "log",
}

_PROBE_PRECISION_DIGITS = 50
_ABSOLUTE_TOLERANCE = 1e-9
_RELATIVE_TOLERANCE = 1e-9
_MAX_SAMPLE_ASSIGNMENTS = 5


class CandidateCompilationError(ValueError):
    """Raised when an expression cannot be compiled to official Pure EML."""


class ValidationStatus(StrEnum):
    """Terminal validation classification for one candidate.

    Every value except :attr:`VALID` marks a retained failure carrying a reason.
    """

    VALID = "valid"
    WRONG_ROOT = "wrong_root"
    COMPILE_FAILED = "compile_failed"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    DOMAIN_MISMATCH = "domain_mismatch"


@dataclass(frozen=True, slots=True)
class VerificationContext:
    """Declared conditions under which candidates are validated.

    ``mode`` and ``assumptions`` select the sample domain for each variable; a
    positive-real formal run samples declared-positive variables only at positive points.
    ``reference`` is the expression every candidate is checked for equivalence against; when
    it is ``None`` the caller-designated first compiling candidate is used.
    """

    mode: RewriteMode = RewriteMode.SAFE_REAL
    assumptions: AssumptionEnvironment = field(default_factory=AssumptionEnvironment)
    reference: Expr | None = None
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4


@dataclass(frozen=True, slots=True)
class ValidatedCandidate:
    """One candidate together with its compiled artifacts and validation verdict."""

    candidate: Candidate
    eml_term: EMLTerm | None
    ast_tree: ASTTree | None
    status: ValidationStatus
    reason: str
    sample_points_checked: int

    @property
    def valid(self) -> bool:
        """Return whether the candidate passed every validation stage."""
        return self.status is ValidationStatus.VALID


def compile_expr_to_eml(
    expr: Expr,
    *,
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile an e-graph expression into an official Pure EML term.

    Every node is produced by the authoritative ``geml.eml`` constructors, so the result
    contains only primitive EML, ``1``, and variable leaves.
    """
    if not isinstance(expr, Expr):
        raise TypeError("compile_expr_to_eml requires an Expr")
    if expr.op is Operator.VARIABLE:
        return variable(str(expr.payload))
    if expr.op is Operator.CONSTANT:
        return _compile_constant(expr.payload)

    children = [compile_expr_to_eml(child, compiler_mode=compiler_mode) for child in expr.children]
    if expr.op is Operator.ADD:
        return eml_add(children[0], children[1], mode=compiler_mode)
    if expr.op is Operator.MUL:
        return eml_multiply(children[0], children[1], mode=compiler_mode)
    if expr.op is Operator.SUB:
        return eml_subtract(children[0], children[1])
    if expr.op is Operator.DIV:
        return eml_divide(children[0], children[1], mode=compiler_mode)
    if expr.op is Operator.POW:
        return eml_power(children[0], children[1], mode=compiler_mode)
    if expr.op is Operator.NEG:
        return eml_negate(children[0], mode=compiler_mode)
    if expr.op is Operator.EXP:
        return eml_exp(children[0])
    if expr.op is Operator.LOG:
        return eml_log(children[0])
    raise CandidateCompilationError(f"unsupported operator for EML compilation: {expr.op}")


def _compile_constant(payload: object) -> EMLTerm:
    """Compile an exact rational constant into official Pure EML."""
    if not isinstance(payload, Fraction):
        raise CandidateCompilationError("a constant must carry an exact Fraction payload")
    if payload.denominator == 1:
        return eml_integer(payload.numerator)
    return eml_rational(payload.numerator, payload.denominator)


def expr_to_ast_tree(expr: Expr, *, expression_id: str) -> ASTTree:
    """Build a validated source :class:`ASTTree` from an e-graph expression.

    Each occurrence becomes a distinct AST node so the result is a genuine tree.  Node
    labels match the frozen direct compiler's dispatch, and statistics are computed by the
    frozen :func:`geml.ast.statistics.calculate_statistics`, never counted by hand.
    """
    if not isinstance(expr, Expr):
        raise TypeError("expr_to_ast_tree requires an Expr")
    nodes: list[ASTNode] = []
    edges: list[ASTEdge] = []
    counter = _Counter()
    root_id = _build_ast_nodes(expr, nodes, edges, counter)
    statistics = calculate_statistics(nodes, edges, root_id)
    return ASTTree(
        expression_id=expression_id,
        root_id=root_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        statistics=statistics,
    )


@dataclass(slots=True)
class _Counter:
    """A monotone identifier source for deterministic AST node ids."""

    value: int = 0

    def next_id(self) -> str:
        """Return the next unique node identifier."""
        node_id = f"ast-{self.value}"
        self.value += 1
        return node_id


def _build_ast_nodes(
    expr: Expr,
    nodes: list[ASTNode],
    edges: list[ASTEdge],
    counter: _Counter,
) -> str:
    """Append the AST nodes and edges for ``expr`` and return its node id."""
    node_id = counter.next_id()
    kind, label, value, arity = _ast_fields(expr)
    nodes.append(ASTNode(node_id=node_id, node_kind=kind, label=label, arity=arity, value=value))
    for slot, child in enumerate(expr.children):
        child_id = _build_ast_nodes(child, nodes, edges, counter)
        edges.append(ASTEdge(source_id=node_id, target_id=child_id, child_slot=slot))
    return node_id


def _ast_fields(expr: Expr) -> tuple[str, str, object, int]:
    """Return the AST ``(node_kind, label, value, arity)`` for an expression node."""
    if expr.op is Operator.VARIABLE:
        return _AST_LEAF_KIND, "symbol", {"name": str(expr.payload)}, 0
    if expr.op is Operator.CONSTANT:
        return _AST_LEAF_KIND, *_constant_ast_fields(expr.payload), 0
    if expr.op in _UNARY_LABELS:
        return _AST_OPERATOR_KIND, _UNARY_LABELS[expr.op], None, 1
    if expr.op in _BINARY_LABELS:
        return _AST_OPERATOR_KIND, _BINARY_LABELS[expr.op], None, 2
    raise CandidateCompilationError(f"unsupported operator for AST construction: {expr.op}")


def _constant_ast_fields(payload: object) -> tuple[str, object]:
    """Return the AST ``(label, value)`` for an exact rational constant."""
    if not isinstance(payload, Fraction):
        raise CandidateCompilationError("a constant must carry an exact Fraction payload")
    if payload.denominator == 1:
        return "integer", payload.numerator
    return "rational", {"numerator": payload.numerator, "denominator": payload.denominator}


def collect_variables(expr: Expr) -> tuple[str, ...]:
    """Return the source-variable names occurring in ``expr`` in sorted order."""
    names: set[str] = set()
    stack: list[Expr] = [expr]
    while stack:
        current = stack.pop()
        if current.op is Operator.VARIABLE:
            names.add(str(current.payload))
        else:
            stack.extend(current.children)
    return tuple(sorted(names))


def validate_candidate(
    candidate: Candidate,
    root: EClassId,
    context: VerificationContext,
    reference_eml: EMLTerm | None,
) -> ValidatedCandidate:
    """Validate one candidate through compilation, semantic, and domain checks.

    ``reference_eml`` is the compiled reference every candidate is compared against.  When
    it is ``None`` (no reference is available yet) only structural and compilation checks
    run, and the candidate is accepted structurally so it can itself become the reference.
    """
    if candidate.eclass != root:
        return ValidatedCandidate(
            candidate=candidate,
            eml_term=None,
            ast_tree=None,
            status=ValidationStatus.WRONG_ROOT,
            reason=f"candidate originates from e-class {candidate.eclass}, not root {root}",
            sample_points_checked=0,
        )

    try:
        eml_term = compile_expr_to_eml(candidate.expression, compiler_mode=context.compiler_mode)
        ast_tree = expr_to_ast_tree(
            candidate.expression,
            expression_id=f"candidate-{candidate.metadata.enumeration_index}",
        )
    except Exception as error:
        return ValidatedCandidate(
            candidate=candidate,
            eml_term=None,
            ast_tree=None,
            status=ValidationStatus.COMPILE_FAILED,
            reason=f"{type(error).__name__}: {error}",
            sample_points_checked=0,
        )

    if reference_eml is None:
        return ValidatedCandidate(
            candidate=candidate,
            eml_term=eml_term,
            ast_tree=ast_tree,
            status=ValidationStatus.VALID,
            reason="accepted as the verification reference",
            sample_points_checked=0,
        )

    status, reason, checked = _verify_numeric(
        candidate.expression, eml_term, reference_eml, context
    )
    return ValidatedCandidate(
        candidate=candidate,
        eml_term=eml_term,
        ast_tree=ast_tree,
        status=status,
        reason=reason,
        sample_points_checked=checked,
    )


def _verify_numeric(
    expr: Expr,
    candidate_eml: EMLTerm,
    reference_eml: EMLTerm,
    context: VerificationContext,
) -> tuple[ValidationStatus, str, int]:
    """Compare a candidate to the reference at deterministic sample points."""
    reference_expr = context.reference if context.reference is not None else expr
    variables = tuple(sorted(set(collect_variables(expr)) | set(collect_variables(reference_expr))))
    assignments = _sample_assignments(variables, context)

    agreements = 0
    for assignment in assignments:
        reference_value, reference_finite = _safe_eval(reference_eml, assignment)
        candidate_value, candidate_finite = _safe_eval(candidate_eml, assignment)
        if not reference_finite and not candidate_finite:
            continue
        if reference_finite != candidate_finite:
            return (
                ValidationStatus.DOMAIN_MISMATCH,
                f"candidate and reference disagree on definedness at {_format(assignment)}",
                agreements,
            )
        if not _close(candidate_value, reference_value):
            return (
                ValidationStatus.SEMANTIC_MISMATCH,
                f"candidate differs from reference at {_format(assignment)}",
                agreements,
            )
        agreements += 1

    if agreements == 0:
        return (
            ValidationStatus.VALID,
            "no sample point evaluated finitely for both candidate and reference",
            0,
        )
    return (
        ValidationStatus.VALID,
        f"verified numerically at {agreements} sample point(s)",
        agreements,
    )


def _sample_assignments(
    variables: tuple[str, ...],
    context: VerificationContext,
) -> tuple[dict[str, float], ...]:
    """Return a bounded, deterministic, domain-aware set of variable assignments."""
    if not variables:
        return ({},)
    pools = {name: _sample_pool(name, context) for name in variables}
    assignments: list[dict[str, float]] = []
    for index in range(_MAX_SAMPLE_ASSIGNMENTS):
        assignment = {
            name: float(pools[name][(index + offset) % len(pools[name])])
            for offset, name in enumerate(variables)
        }
        assignments.append(assignment)
    return tuple(assignments)


def _sample_pool(name: str, context: VerificationContext) -> tuple[Fraction, ...]:
    """Return the ordered sample values allowed for one variable under the context."""
    declared = context.assumptions.assumptions_for(name)
    if Assumption.POSITIVE in declared:
        return (Fraction(1, 2), Fraction(2), Fraction(3))
    if Assumption.NONNEGATIVE in declared:
        return (Fraction(1), Fraction(2), Fraction(3))
    if Assumption.NONZERO in declared:
        return (Fraction(1), Fraction(-2), Fraction(3))
    return (Fraction(2), Fraction(-1), Fraction(3, 2))


def _safe_eval(term: EMLTerm, assignment: dict[str, float]) -> tuple[complex | None, bool]:
    """Evaluate a term at an assignment, returning ``(value, finite)``."""
    try:
        value, _descendant_nonfinite = evaluate_pure_eml(
            term,
            variables=dict(assignment),
            backend=NumericBackend.MPMATH,
            precision_digits=_PROBE_PRECISION_DIGITS,
        )
        converted = complex(value)
    except Exception:
        return None, False
    if math.isfinite(converted.real) and math.isfinite(converted.imag):
        return converted, True
    return None, False


def _close(candidate: complex | None, reference: complex | None) -> bool:
    """Return whether two finite complex values agree within tolerance."""
    if candidate is None or reference is None:
        return False
    tolerance = _ABSOLUTE_TOLERANCE + _RELATIVE_TOLERANCE * abs(reference)
    return abs(candidate - reference) <= tolerance


def _format(assignment: dict[str, float]) -> str:
    """Return a stable textual form of a variable assignment."""
    if not assignment:
        return "the constant point"
    return ", ".join(f"{name}={assignment[name]}" for name in sorted(assignment))
