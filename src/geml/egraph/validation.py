"""Independent validation of expressions extracted from a Goal 4 e-graph.

An extracted expression is accepted only when all of the following hold:

* its concrete structure can be found in the saturated root e-class without mutating the
  e-graph;
* the authoritative Goal 3 direct compiler can compile its source AST;
* the exact Goal 2 count-only compiler can account for its expanded EML tree; and
* an independent source-expression evaluator finds no semantic or defined-domain mismatch
  against the caller-supplied source expression and observes at least one finite agreement.

Same-e-class membership is the formal equivalence evidence.  Numeric probing is an
independent bug-detection audit, not a proof and never a substitute for membership.  A
missing reference or a probe set with no finite evidence is therefore retained as a
failure rather than silently accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction

from geml.ast.statistics import calculate_statistics
from geml.contracts.ast import ASTEdge, ASTNode, ASTTree
from geml.egraph.candidates import Candidate
from geml.egraph.core import EGraph
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
    require_compiler_mode,
)
from geml.eml.counting import (
    CountedEML,
    count_eml_add,
    count_eml_divide,
    count_eml_exp,
    count_eml_integer,
    count_eml_log,
    count_eml_multiply,
    count_eml_negate,
    count_eml_power,
    count_eml_rational,
    count_eml_subtract,
    count_variable,
)
from geml.eml.ir import EMLTerm, variable
from geml.interfaces.eml_dag_cost import (
    EMLDagCostResult,
    EMLDagCostStatus,
    compute_eml_dag_cost,
)

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

_ABSOLUTE_TOLERANCE = 1e-10
_RELATIVE_TOLERANCE = 1e-10
_MAX_SAMPLE_ASSIGNMENTS = 8


class CandidateCompilationError(ValueError):
    """Raised when an expression cannot be compiled by a frozen EML interface."""


class ValidationStatus(StrEnum):
    """Terminal validation classification for one retained candidate."""

    VALID = "valid"
    WRONG_ROOT = "wrong_root"
    GRAPH_NOT_CLOSED = "graph_not_closed"
    REFERENCE_MISSING = "reference_missing"
    REFERENCE_OUTSIDE_ROOT = "reference_outside_root"
    COMPILE_FAILED = "compile_failed"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    DOMAIN_MISMATCH = "domain_mismatch"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class VerificationContext:
    """Declared conditions under which candidates are validated."""

    mode: RewriteMode = RewriteMode.SAFE_REAL
    assumptions: AssumptionEnvironment = field(default_factory=AssumptionEnvironment)
    reference: Expr | None = None
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RewriteMode):
            raise TypeError("mode must be a RewriteMode")
        if not isinstance(self.assumptions, AssumptionEnvironment):
            raise TypeError("assumptions must be an AssumptionEnvironment")
        if self.reference is not None and not isinstance(self.reference, Expr):
            raise TypeError("reference must be an Expr or None")
        require_compiler_mode(self.compiler_mode)


@dataclass(frozen=True, slots=True)
class ValidatedCandidate:
    """One candidate with compilation evidence and a validation verdict.

    ``eml_term`` is retained for API compatibility but deliberately remains ``None`` in the
    scalable path.  Goal 4 validates and costs through the direct source-AST compiler so it
    never has to materialize the potentially enormous expanded EML tree.
    """

    candidate: Candidate
    eml_term: EMLTerm | None
    ast_tree: ASTTree | None
    eml_tree_count: CountedEML | None
    dag_cost_result: EMLDagCostResult | None
    compiler_mode: CompilerMode
    status: ValidationStatus
    reason: str
    sample_points_checked: int

    @property
    def valid(self) -> bool:
        return self.status is ValidationStatus.VALID


def compile_expr_to_eml(
    expr: Expr,
    *,
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Materialize a small expression through the authoritative Pure EML constructors.

    Production validation and cost evaluation do not call this function; they use the
    direct and count-only compilers below.  It remains useful for focused formula audits.
    """
    if not isinstance(expr, Expr):
        raise TypeError("compile_expr_to_eml requires an Expr")
    require_compiler_mode(compiler_mode)
    if expr.op is Operator.VARIABLE:
        return variable(str(expr.payload))
    if expr.op is Operator.CONSTANT:
        return _compile_constant(expr.payload, compiler_mode)

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


def _compile_constant(payload: object, compiler_mode: CompilerMode) -> EMLTerm:
    if not isinstance(payload, Fraction):
        raise CandidateCompilationError("a constant must carry an exact Fraction payload")
    if payload.denominator == 1:
        return eml_integer(payload.numerator, mode=compiler_mode)
    return eml_rational(payload.numerator, payload.denominator, mode=compiler_mode)


def count_expr_eml_tree(
    expr: Expr,
    *,
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> CountedEML:
    """Return the exact expanded EML tree count without materializing that tree."""
    if not isinstance(expr, Expr):
        raise TypeError("count_expr_eml_tree requires an Expr")
    mode = require_compiler_mode(compiler_mode)
    memo: dict[int, CountedEML] = {}
    stack: list[tuple[Expr, bool]] = [(expr, False)]
    while stack:
        current, expanded = stack.pop()
        key = id(current)
        if key in memo:
            continue
        if not expanded:
            stack.append((current, True))
            for child in reversed(current.children):
                stack.append((child, False))
            continue
        children = [memo[id(child)] for child in current.children]
        if current.op is Operator.VARIABLE:
            result = count_variable(str(current.payload), mode=mode)
        elif current.op is Operator.CONSTANT:
            payload = current.payload
            if not isinstance(payload, Fraction):
                raise CandidateCompilationError("a constant must carry an exact Fraction payload")
            result = (
                count_eml_integer(payload.numerator, mode=mode)
                if payload.denominator == 1
                else count_eml_rational(payload.numerator, payload.denominator, mode=mode)
            )
        elif current.op is Operator.ADD:
            result = count_eml_add(children[0], children[1], mode=mode)
        elif current.op is Operator.MUL:
            result = count_eml_multiply(children[0], children[1], mode=mode)
        elif current.op is Operator.SUB:
            result = count_eml_subtract(children[0], children[1], mode=mode)
        elif current.op is Operator.DIV:
            result = count_eml_divide(children[0], children[1], mode=mode)
        elif current.op is Operator.POW:
            result = count_eml_power(children[0], children[1], mode=mode)
        elif current.op is Operator.NEG:
            result = count_eml_negate(children[0], mode=mode)
        elif current.op is Operator.EXP:
            result = count_eml_exp(children[0], mode=mode)
        elif current.op is Operator.LOG:
            result = count_eml_log(children[0], mode=mode)
        else:  # pragma: no cover - Operator is a closed enum
            raise CandidateCompilationError(f"unsupported operator for EML counting: {current.op}")
        memo[key] = result
    return memo[id(expr)]


def expr_to_ast_tree(expr: Expr, *, expression_id: str) -> ASTTree:
    """Build a validated source AST with ordered child slots."""
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
    value: int = 0

    def next_id(self) -> str:
        node_id = f"ast-{self.value}"
        self.value += 1
        return node_id


def _build_ast_nodes(
    expr: Expr,
    nodes: list[ASTNode],
    edges: list[ASTEdge],
    counter: _Counter,
) -> str:
    node_id = counter.next_id()
    kind, label, value, arity = _ast_fields(expr)
    nodes.append(ASTNode(node_id=node_id, node_kind=kind, label=label, arity=arity, value=value))
    for slot, child in enumerate(expr.children):
        child_id = _build_ast_nodes(child, nodes, edges, counter)
        edges.append(ASTEdge(source_id=node_id, target_id=child_id, child_slot=slot))
    return node_id


def _ast_fields(expr: Expr) -> tuple[str, str, object, int]:
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
    if not isinstance(payload, Fraction):
        raise CandidateCompilationError("a constant must carry an exact Fraction payload")
    if payload.denominator == 1:
        return "integer", payload.numerator
    return "rational", {
        "numerator": payload.numerator,
        "denominator": payload.denominator,
    }


def collect_variables(expr: Expr) -> tuple[str, ...]:
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
    egraph: EGraph,
) -> ValidatedCandidate:
    """Validate one candidate against an explicit source reference and e-graph."""
    if not isinstance(egraph, EGraph):
        raise TypeError("validate_candidate requires the source EGraph")
    if egraph.pending_repairs:
        return _failure(
            candidate,
            context,
            ValidationStatus.GRAPH_NOT_CLOSED,
            "candidate validation requires a congruence-closed e-graph",
        )
    canonical_root = egraph.find(root)
    try:
        metadata_root = egraph.find(candidate.eclass)
    except Exception as error:
        return _failure(
            candidate,
            context,
            ValidationStatus.WRONG_ROOT,
            f"invalid candidate e-class: {type(error).__name__}: {error}",
        )
    if metadata_root != canonical_root:
        return _failure(
            candidate,
            context,
            ValidationStatus.WRONG_ROOT,
            f"candidate metadata identifies e-class {metadata_root}, not root {canonical_root}",
        )
    structural_root = egraph.lookup_expr(candidate.expression)
    if structural_root is None or egraph.find(structural_root) != canonical_root:
        return _failure(
            candidate,
            context,
            ValidationStatus.WRONG_ROOT,
            "candidate structure is not present in the claimed root e-class",
        )
    if context.reference is None:
        return _failure(
            candidate,
            context,
            ValidationStatus.REFERENCE_MISSING,
            "validation requires an explicit caller-supplied source reference",
        )
    reference_root = egraph.lookup_expr(context.reference)
    if reference_root is None or egraph.find(reference_root) != canonical_root:
        return _failure(
            candidate,
            context,
            ValidationStatus.REFERENCE_OUTSIDE_ROOT,
            "the caller-supplied source reference is not present in the root e-class",
        )

    try:
        ast_tree = expr_to_ast_tree(
            candidate.expression,
            expression_id=f"candidate-{candidate.metadata.enumeration_index}",
        )
        tree_count = count_expr_eml_tree(candidate.expression, compiler_mode=context.compiler_mode)
        dag_result = compute_eml_dag_cost(ast_tree, compiler_mode=context.compiler_mode)
    except Exception as error:
        return _failure(
            candidate,
            context,
            ValidationStatus.COMPILE_FAILED,
            f"{type(error).__name__}: {error}",
        )
    if dag_result.status is not EMLDagCostStatus.SUCCESS:
        return ValidatedCandidate(
            candidate=candidate,
            eml_term=None,
            ast_tree=ast_tree,
            eml_tree_count=tree_count,
            dag_cost_result=dag_result,
            compiler_mode=context.compiler_mode,
            status=ValidationStatus.COMPILE_FAILED,
            reason=(
                "official direct EML compilation failed: "
                f"{dag_result.error_type}: {dag_result.error_message}"
            ),
            sample_points_checked=0,
        )

    status, reason, checked = _verify_numeric(candidate.expression, context.reference, context)
    if status is ValidationStatus.VALID:
        reason = (
            "root e-class membership verified independently; "
            f"source-semantics audit agreed at {checked} point(s)"
        )
    return ValidatedCandidate(
        candidate=candidate,
        eml_term=None,
        ast_tree=ast_tree,
        eml_tree_count=tree_count,
        dag_cost_result=dag_result,
        compiler_mode=context.compiler_mode,
        status=status,
        reason=reason,
        sample_points_checked=checked,
    )


def _failure(
    candidate: Candidate,
    context: VerificationContext,
    status: ValidationStatus,
    reason: str,
) -> ValidatedCandidate:
    return ValidatedCandidate(
        candidate=candidate,
        eml_term=None,
        ast_tree=None,
        eml_tree_count=None,
        dag_cost_result=None,
        compiler_mode=context.compiler_mode,
        status=status,
        reason=reason,
        sample_points_checked=0,
    )


def _verify_numeric(
    candidate: Expr,
    reference: Expr,
    context: VerificationContext,
) -> tuple[ValidationStatus, str, int]:
    """Run an independent, deterministic source-semantics audit."""
    variables = tuple(sorted(set(collect_variables(candidate)) | set(collect_variables(reference))))
    agreements = 0
    for assignment in _sample_assignments(variables, context):
        reference_value, reference_finite = _safe_eval_expr(reference, assignment)
        candidate_value, candidate_finite = _safe_eval_expr(candidate, assignment)
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
            ValidationStatus.INCONCLUSIVE,
            "no deterministic sample point produced finite evidence for both expressions",
            0,
        )
    return ValidationStatus.VALID, "numeric audit passed", agreements


def _sample_assignments(
    variables: tuple[str, ...],
    context: VerificationContext,
) -> tuple[dict[str, float], ...]:
    if not variables:
        return ({},)
    pools = {name: _sample_pool(name, context) for name in variables}
    assignments: list[dict[str, float]] = []
    for index in range(_MAX_SAMPLE_ASSIGNMENTS):
        assignments.append(
            {
                name: float(pools[name][(index + offset) % len(pools[name])])
                for offset, name in enumerate(variables)
            }
        )
    return tuple(assignments)


def _sample_pool(name: str, context: VerificationContext) -> tuple[Fraction, ...]:
    declared = context.assumptions.assumptions_for(name)
    if Assumption.POSITIVE in declared:
        return (
            Fraction(1, 4),
            Fraction(1, 2),
            Fraction(1),
            Fraction(3, 2),
            Fraction(2),
            Fraction(3),
        )
    if Assumption.NONNEGATIVE in declared:
        return (
            Fraction(0),
            Fraction(1, 4),
            Fraction(1),
            Fraction(2),
            Fraction(3),
        )
    if Assumption.NONZERO in declared:
        return (
            Fraction(1, 2),
            Fraction(1),
            Fraction(2),
            Fraction(-1),
            Fraction(-2),
            Fraction(3),
        )
    return (
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
        Fraction(2),
        Fraction(-1),
        Fraction(-2),
        Fraction(3, 2),
        Fraction(3),
    )


def _safe_eval_expr(expr: Expr, assignment: dict[str, float]) -> tuple[float | None, bool]:
    try:
        value = _eval_expr(expr, assignment)
    except (ArithmeticError, OverflowError, ValueError, KeyError):
        return None, False
    return (value, True) if math.isfinite(value) else (None, False)


def _eval_expr(expr: Expr, assignment: dict[str, float]) -> float:
    memo: dict[int, float] = {}
    stack: list[tuple[Expr, bool]] = [(expr, False)]
    while stack:
        current, expanded = stack.pop()
        key = id(current)
        if key in memo:
            continue
        if not expanded:
            stack.append((current, True))
            for child in reversed(current.children):
                stack.append((child, False))
            continue
        children = [memo[id(child)] for child in current.children]
        if current.op is Operator.VARIABLE:
            result = assignment[str(current.payload)]
        elif current.op is Operator.CONSTANT:
            payload = current.payload
            if not isinstance(payload, Fraction):
                raise ValueError("malformed constant")
            result = payload.numerator / payload.denominator
        elif current.op is Operator.ADD:
            result = children[0] + children[1]
        elif current.op is Operator.MUL:
            result = children[0] * children[1]
        elif current.op is Operator.NEG:
            result = -children[0]
        elif current.op is Operator.SUB:
            result = children[0] - children[1]
        elif current.op is Operator.DIV:
            if children[1] == 0:
                raise ZeroDivisionError
            result = children[0] / children[1]
        elif current.op is Operator.POW:
            result = math.pow(children[0], children[1])
        elif current.op is Operator.EXP:
            result = math.exp(children[0])
        elif current.op is Operator.LOG:
            if children[0] <= 0:
                raise ValueError("real logarithm outside its domain")
            result = math.log(children[0])
        else:  # pragma: no cover - Operator is closed
            raise ValueError(f"unsupported operator {current.op}")
        if not math.isfinite(result):
            raise OverflowError("non-finite source evaluation")
        memo[key] = result
    return memo[id(expr)]


def _close(candidate: float | None, reference: float | None) -> bool:
    if candidate is None or reference is None:
        return False
    tolerance = _ABSOLUTE_TOLERANCE + _RELATIVE_TOLERANCE * abs(reference)
    return abs(candidate - reference) <= tolerance


def _format(assignment: dict[str, float]) -> str:
    if not assignment:
        return "the constant point"
    return ", ".join(f"{name}={assignment[name]}" for name in sorted(assignment))


__all__ = [
    "CandidateCompilationError",
    "ValidatedCandidate",
    "ValidationStatus",
    "VerificationContext",
    "collect_variables",
    "compile_expr_to_eml",
    "count_expr_eml_tree",
    "expr_to_ast_tree",
    "validate_candidate",
]
