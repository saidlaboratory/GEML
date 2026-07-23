"""Failure-aware equivalence audit for direct and post-hoc pure-EML DAGs.

The official comparison is intentionally small but stratified.  Each case is a
validated, hand-built AST fixture compiled in two independent ways:

1. materialize the official-v4 pure-EML tree, then share exact subtrees; and
2. compile the AST directly into a hash-consed pure-EML DAG.

Every requested case produces one terminal result.  Construction failures,
unsupported cases, and comparison mismatches are retained rather than skipped.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import mpmath as mp

from geml.ast.statistics import calculate_statistics
from geml.contracts.ast import ASTEdge, ASTNode, ASTTree
from geml.contracts.corpus import CorpusSplit
from geml.dag.direct_eml import ConstructionStats, compile_ast_to_eml_dag
from geml.dag.eml import (
    EMLDagStatistics,
    convert_with_stats,
    dag_to_eml,
    validate_eml_dag,
)
from geml.eml.compiler_core import CompilerMode
from geml.experiments.goal2.run import materialize_ast_official
from geml.graph.schema import Graph, compute_statistics
from geml.graph.signatures import compute_signature, compute_signatures
from geml.spec.corpus_families import (
    CORPUS_FAMILIES,
    CORPUS_FAMILY_REGISTRY,
    CorpusPolicyKind,
)
from geml.spec.domains import DOMAIN_POLICIES
from geml.spec.operators import OPERATORS, EMLConstructionStatus
from geml.verification.eml.numeric import NumericBackend, evaluate_pure_eml

AUDIT_SCHEMA_VERSION: Final = "geml-goal3-equivalence-audit-v1"
AUDIT_SIZE_BUCKETS: Final = (
    (1, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, 128),
)
AUDIT_PRECISION_DIGITS: Final = 80
_EVALUATION_TOLERANCE_TEXT: Final = "1e-60"

_ENABLED_APPROVED_OPERATORS: Final = tuple(
    operator
    for operator in OPERATORS
    if operator.enabled_for_generation
    and operator.eml_construction_status is EMLConstructionStatus.APPROVED
)
REQUIRED_OPERATOR_NAMES: Final = tuple(operator.name for operator in _ENABLED_APPROVED_OPERATORS)
REQUIRED_OPERATOR_FAMILIES: Final = tuple(
    dict.fromkeys(operator.operator_family for operator in _ENABLED_APPROVED_OPERATORS)
)
REQUIRED_CORPUS_FAMILIES: Final = tuple(family.family_id for family in CORPUS_FAMILIES)
REQUIRED_DOMAIN_MODES: Final = tuple(
    policy.name for policy in DOMAIN_POLICIES if policy.enabled_for_generation
)
REQUIRED_SPLITS: Final = tuple(CorpusSplit)

type SizeBucket = tuple[int, int]
type NumericBinding = int | float
type DirectCompiler = Callable[[ASTTree], tuple[Graph, str, ConstructionStats]]
type PostHocCompiler = Callable[[ASTTree], tuple[Graph, str, EMLDagStatistics]]


class AuditStatus(StrEnum):
    """Terminal classification for one requested audit case."""

    MATCH = "match"
    MISMATCH = "mismatch"
    BLOCKED = "blocked"
    FAILURE = "failure"


class ComparisonAxis(StrEnum):
    """Independently reported structural and executable comparisons."""

    CANONICAL_SIGNATURE = "canonical_signature"
    CANONICAL_TOPOLOGY = "canonical_topology"
    NODE_COUNT = "node_count"
    CHILD_REFERENCES = "child_references"
    DEPTH = "depth"
    EVALUATION = "evaluation"
    PURITY = "purity"


@dataclass(frozen=True, slots=True)
class AuditCase:
    """One stratified AST fixture or one explicitly blocked request."""

    case_id: str
    operator_names: tuple[str, ...]
    corpus_family: str
    size_bucket: SizeBucket
    split: CorpusSplit
    domain_mode: str
    ast: ASTTree | None
    evaluation_bindings: tuple[tuple[str, NumericBinding], ...] = ()
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be nonblank")
        if not self.operator_names or len(set(self.operator_names)) != len(self.operator_names):
            raise ValueError("operator_names must be nonempty and unique")
        unknown_operators = set(self.operator_names) - set(REQUIRED_OPERATOR_NAMES)
        if unknown_operators:
            raise ValueError(f"audit case references non-approved operators: {unknown_operators}")
        if self.corpus_family not in CORPUS_FAMILY_REGISTRY:
            raise ValueError(f"unknown corpus family {self.corpus_family!r}")
        corpus_spec = CORPUS_FAMILY_REGISTRY[self.corpus_family]
        if self.size_bucket not in AUDIT_SIZE_BUCKETS:
            raise ValueError(f"unknown audit size bucket {self.size_bucket!r}")
        if not isinstance(self.split, CorpusSplit):
            raise TypeError("split must be a CorpusSplit")
        if (corpus_spec.policy_kind is CorpusPolicyKind.OOD_STRESS) != (
            self.split is CorpusSplit.TEST_OOD
        ):
            raise ValueError("OOD-stress cases and test_ood split cases must coincide")
        if self.domain_mode not in REQUIRED_DOMAIN_MODES:
            raise ValueError(f"domain mode {self.domain_mode!r} is not generation-enabled")
        if self.domain_mode not in corpus_spec.allowed_domain_modes:
            raise ValueError("case domain is not allowed by its corpus family")

        binding_names = [name for name, _ in self.evaluation_bindings]
        if any(not name.strip() for name in binding_names) or len(binding_names) != len(
            set(binding_names)
        ):
            raise ValueError("evaluation binding names must be nonblank and unique")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for _, value in self.evaluation_bindings
        ):
            raise TypeError("evaluation bindings must contain finite int or float values")

        if self.blocked_reason is not None:
            if not self.blocked_reason.strip():
                raise ValueError("blocked_reason must be nonblank when supplied")
            return
        if self.ast is None:
            raise ValueError("a runnable audit case requires an AST")
        minimum, maximum = self.size_bucket
        if not minimum <= self.ast.statistics.node_count <= maximum:
            raise ValueError(
                f"AST size {self.ast.statistics.node_count} is outside bucket {minimum}-{maximum}"
            )
        actual_operators = {node.label for node in self.ast.nodes}
        if set(self.operator_names) != actual_operators:
            raise ValueError(
                "operator_names must exactly describe AST labels; "
                f"declared={sorted(self.operator_names)}, actual={sorted(actual_operators)}"
            )
        actual_operator_families = {
            operator.operator_family
            for operator in _ENABLED_APPROVED_OPERATORS
            if operator.name in actual_operators
        }
        if corpus_spec.eligible_operators:
            disallowed = actual_operators - set(corpus_spec.eligible_operators)
        else:
            disallowed = actual_operator_families - set(corpus_spec.operator_family_constraints)
        if disallowed:
            raise ValueError(
                f"case contains operators outside corpus-family policy: {sorted(disallowed)}"
            )

        symbol_names = {
            node.value["name"]
            for node in self.ast.nodes
            if node.label == "symbol"
            and isinstance(node.value, dict)
            and isinstance(node.value.get("name"), str)
        }
        if set(binding_names) != symbol_names:
            raise ValueError(
                "evaluation bindings must exactly cover source symbols; "
                f"bindings={sorted(binding_names)}, symbols={sorted(symbol_names)}"
            )
        binding_values = dict(self.evaluation_bindings).values()
        if self.domain_mode == "positive_real" and any(value <= 0 for value in binding_values):
            raise ValueError("positive_real evaluation bindings must be strictly positive")
        if self.domain_mode == "nonzero_real" and any(value == 0 for value in binding_values):
            raise ValueError("nonzero_real evaluation bindings must be nonzero")


@dataclass(frozen=True, slots=True)
class AxisComparison:
    """One retained direct-versus-post-hoc comparison."""

    axis: ComparisonAxis
    matched: bool
    direct_value: str
    posthoc_value: str
    detail: str
    error: bool = False


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Complete terminal result for one requested audit case."""

    case_id: str
    status: AuditStatus
    operator_names: tuple[str, ...]
    operator_families: tuple[str, ...]
    corpus_family: str
    size_bucket: SizeBucket
    split: CorpusSplit
    domain_mode: str
    comparisons: tuple[AxisComparison, ...] = ()
    blocker_reason: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None

    @property
    def all_match(self) -> bool:
        """Return whether every required axis completed and matched."""

        return self.status is AuditStatus.MATCH

    @property
    def mismatch_details(self) -> tuple[str, ...]:
        """Return stable human-readable diagnostics for unsuccessful axes."""

        details = tuple(
            f"{comparison.axis.value}: {comparison.detail}"
            for comparison in self.comparisons
            if not comparison.matched
        )
        if self.blocker_reason is not None:
            return (*details, self.blocker_reason)
        if self.failure_message is not None:
            return (*details, self.failure_message)
        return details


@dataclass(frozen=True, slots=True)
class AuditSummary:
    """Coverage, readiness, and deterministic identity for one complete audit."""

    schema_version: str
    results: tuple[AuditResult, ...]
    missing_operators: tuple[str, ...]
    missing_operator_families: tuple[str, ...]
    missing_corpus_families: tuple[str, ...]
    missing_size_buckets: tuple[SizeBucket, ...]
    missing_splits: tuple[CorpusSplit, ...]
    missing_domain_modes: tuple[str, ...]
    fingerprint: str

    @property
    def ready(self) -> bool:
        """Return whether coverage is complete and every case matched."""

        return (
            bool(self.results)
            and all(result.status is AuditStatus.MATCH for result in self.results)
            and not self.missing_operators
            and not self.missing_operator_families
            and not self.missing_corpus_families
            and not self.missing_size_buckets
            and not self.missing_splits
            and not self.missing_domain_modes
        )

    @property
    def blockers(self) -> tuple[AuditResult, ...]:
        """Return every blocked, failed, or mismatched case in input order."""

        return tuple(result for result in self.results if result.status is not AuditStatus.MATCH)


@dataclass(frozen=True, slots=True)
class _FixtureNode:
    label: str
    value: Any = None
    children: tuple[_FixtureNode, ...] = ()


def _leaf(label: str, value: Any) -> _FixtureNode:
    return _FixtureNode(label=label, value=value)


def _unary(label: str, child: _FixtureNode) -> _FixtureNode:
    return _FixtureNode(label=label, children=(child,))


def _binary(label: str, left: _FixtureNode, right: _FixtureNode) -> _FixtureNode:
    return _FixtureNode(label=label, children=(left, right))


def _fixture_ast(case_id: str, root: _FixtureNode) -> ASTTree:
    """Build a tiny authoritative AST contract without parsing source text."""

    nodes: list[ASTNode] = []
    edges: list[ASTEdge] = []
    pending: list[tuple[_FixtureNode, str | None, int | None]] = [(root, None, None)]
    while pending:
        fixture, parent_id, child_slot = pending.pop()
        node_id = f"n{len(nodes):06d}"
        nodes.append(
            ASTNode(
                node_id=node_id,
                node_kind="leaf" if not fixture.children else "operator",
                label=fixture.label,
                arity=len(fixture.children),
                value=fixture.value,
                metadata={"fixture": "goal3_equivalence_audit"},
            )
        )
        if parent_id is not None and child_slot is not None:
            edges.append(ASTEdge(source_id=parent_id, target_id=node_id, child_slot=child_slot))
        for slot in reversed(range(len(fixture.children))):
            pending.append((fixture.children[slot], node_id, slot))

    root_id = nodes[0].node_id
    return ASTTree(
        expression_id=case_id,
        root_id=root_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        statistics=calculate_statistics(nodes, edges, root_id),
    )


def _operator_names(ast: ASTTree) -> tuple[str, ...]:
    return tuple(
        operator.name
        for operator in _ENABLED_APPROVED_OPERATORS
        if any(node.label == operator.name for node in ast.nodes)
    )


def _case(
    *,
    case_id: str,
    root: _FixtureNode,
    corpus_family: str,
    size_bucket: SizeBucket,
    split: CorpusSplit,
    domain_mode: str,
    bindings: tuple[tuple[str, NumericBinding], ...] = (),
) -> AuditCase:
    ast = _fixture_ast(case_id, root)
    return AuditCase(
        case_id=case_id,
        operator_names=_operator_names(ast),
        corpus_family=corpus_family,
        size_bucket=size_bucket,
        split=split,
        domain_mode=domain_mode,
        ast=ast,
        evaluation_bindings=bindings,
    )


def _log_exp_chain(pair_count: int) -> _FixtureNode:
    value = _leaf("symbol", {"name": "x", "assumptions": {"real": True}})
    for _ in range(pair_count):
        value = _unary("log", _unary("exp", value))
    return value


_X = _leaf("symbol", {"name": "x", "assumptions": {"real": True}})
_Y = _leaf("symbol", {"name": "y", "assumptions": {"real": True}})
_ONE = _leaf("one", 1)
_TWO = _leaf("integer", 2)
_HALF = _leaf("rational", {"numerator": 1, "denominator": 2})
_XY_BINDINGS: Final = (("x", 0.25), ("y", 2.0))

# The five size buckets match configs/goal2_final.yaml.  The 9/17/33/65
# fixtures use stable log(exp(...)) pairs so the audit reaches each bucket
# without making production artifacts or generating pathological towers.
STRATIFIED_AUDIT_SET: Final = (
    _case(
        case_id="symbol_train_safe",
        root=_X,
        corpus_family="algebraic_core",
        size_bucket=(1, 8),
        split=CorpusSplit.TRAIN,
        domain_mode="safe_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="one_validation_positive",
        root=_ONE,
        corpus_family="algebraic_core",
        size_bucket=(1, 8),
        split=CorpusSplit.VALIDATION,
        domain_mode="positive_real",
    ),
    _case(
        case_id="integer_test_iid_nonzero",
        root=_TWO,
        corpus_family="algebraic_core",
        size_bucket=(1, 8),
        split=CorpusSplit.TEST_IID,
        domain_mode="nonzero_real",
    ),
    _case(
        case_id="rational_train_safe",
        root=_HALF,
        corpus_family="powers_division_rationals",
        size_bucket=(1, 8),
        split=CorpusSplit.TRAIN,
        domain_mode="safe_real",
    ),
    _case(
        case_id="negate_validation_positive",
        root=_unary("negate", _X),
        corpus_family="algebraic_core",
        size_bucket=(1, 8),
        split=CorpusSplit.VALIDATION,
        domain_mode="positive_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="exp_train_nonzero",
        root=_unary("exp", _X),
        corpus_family="exp_log",
        size_bucket=(1, 8),
        split=CorpusSplit.TRAIN,
        domain_mode="nonzero_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="log_validation_positive",
        root=_unary("log", _X),
        corpus_family="exp_log",
        size_bucket=(1, 8),
        split=CorpusSplit.VALIDATION,
        domain_mode="positive_real",
        bindings=(("x", 2.0),),
    ),
    _case(
        case_id="sin_test_iid_safe",
        root=_unary("sin", _X),
        corpus_family="trig_hyperbolic",
        size_bucket=(1, 8),
        split=CorpusSplit.TEST_IID,
        domain_mode="safe_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="cos_train_positive",
        root=_unary("cos", _X),
        corpus_family="trig_hyperbolic",
        size_bucket=(1, 8),
        split=CorpusSplit.TRAIN,
        domain_mode="positive_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="tan_validation_nonzero",
        root=_unary("tan", _X),
        corpus_family="trig_hyperbolic",
        size_bucket=(1, 8),
        split=CorpusSplit.VALIDATION,
        domain_mode="nonzero_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="sinh_test_iid_safe",
        root=_unary("sinh", _X),
        corpus_family="trig_hyperbolic",
        size_bucket=(1, 8),
        split=CorpusSplit.TEST_IID,
        domain_mode="safe_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="cosh_train_positive",
        root=_unary("cosh", _X),
        corpus_family="trig_hyperbolic",
        size_bucket=(1, 8),
        split=CorpusSplit.TRAIN,
        domain_mode="positive_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="tanh_validation_nonzero",
        root=_unary("tanh", _X),
        corpus_family="trig_hyperbolic",
        size_bucket=(1, 8),
        split=CorpusSplit.VALIDATION,
        domain_mode="nonzero_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="add_test_iid_safe",
        root=_binary("add", _X, _Y),
        corpus_family="algebraic_core",
        size_bucket=(1, 8),
        split=CorpusSplit.TEST_IID,
        domain_mode="safe_real",
        bindings=_XY_BINDINGS,
    ),
    _case(
        case_id="subtract_train_positive",
        root=_binary("subtract", _X, _Y),
        corpus_family="algebraic_core",
        size_bucket=(1, 8),
        split=CorpusSplit.TRAIN,
        domain_mode="positive_real",
        bindings=_XY_BINDINGS,
    ),
    _case(
        case_id="multiply_validation_nonzero",
        root=_binary("multiply", _X, _Y),
        corpus_family="algebraic_core",
        size_bucket=(1, 8),
        split=CorpusSplit.VALIDATION,
        domain_mode="nonzero_real",
        bindings=_XY_BINDINGS,
    ),
    _case(
        case_id="divide_test_iid_safe",
        root=_binary("divide", _X, _Y),
        corpus_family="powers_division_rationals",
        size_bucket=(1, 8),
        split=CorpusSplit.TEST_IID,
        domain_mode="safe_real",
        bindings=_XY_BINDINGS,
    ),
    _case(
        case_id="power_train_positive",
        root=_binary("power", _X, _TWO),
        corpus_family="powers_division_rationals",
        size_bucket=(1, 8),
        split=CorpusSplit.TRAIN,
        domain_mode="positive_real",
        bindings=(("x", 2.0),),
    ),
    _case(
        case_id="mixed_validation_safe",
        root=_unary("exp", _unary("sin", _X)),
        corpus_family="mixed_elementary",
        size_bucket=(1, 8),
        split=CorpusSplit.VALIDATION,
        domain_mode="safe_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="ood_size_9_safe",
        root=_log_exp_chain(4),
        corpus_family="ood_stress",
        size_bucket=(9, 16),
        split=CorpusSplit.TEST_OOD,
        domain_mode="safe_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="ood_size_17_positive",
        root=_log_exp_chain(8),
        corpus_family="ood_stress",
        size_bucket=(17, 32),
        split=CorpusSplit.TEST_OOD,
        domain_mode="positive_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="ood_size_33_nonzero",
        root=_log_exp_chain(16),
        corpus_family="ood_stress",
        size_bucket=(33, 64),
        split=CorpusSplit.TEST_OOD,
        domain_mode="nonzero_real",
        bindings=(("x", 0.25),),
    ),
    _case(
        case_id="ood_size_65_safe",
        root=_log_exp_chain(32),
        corpus_family="ood_stress",
        size_bucket=(65, 128),
        split=CorpusSplit.TEST_OOD,
        domain_mode="safe_real",
        bindings=(("x", 0.25),),
    ),
)


def _compile_direct(ast: ASTTree) -> tuple[Graph, str, ConstructionStats]:
    return compile_ast_to_eml_dag(ast, mode=CompilerMode.OFFICIAL_V4)


def _compile_posthoc(ast: ASTTree) -> tuple[Graph, str, EMLDagStatistics]:
    tree = materialize_ast_official(ast)
    graph, statistics = convert_with_stats(
        tree,
        root_id=ast.expression_id,
        representation_mode=f"pure_eml:{CompilerMode.OFFICIAL_V4.value}",
    )
    return graph, graph.roots[0].target_id, statistics


def _canonical_topology(graph: Graph) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    signatures = compute_signatures(graph, graph.nodes)
    nodes = tuple(
        sorted(
            (
                signatures[node.node_id],
                node.family,
                node.kind,
                node.label,
                json.dumps(
                    node.value,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                tuple(
                    (child.slot, signatures[child.target_id])
                    for child in sorted(node.children, key=lambda reference: reference.slot)
                ),
            )
            for node in graph.nodes.values()
        )
    )
    roots = tuple(
        (
            root.root_id,
            signatures[root.target_id],
            root.representation_mode,
        )
        for root in graph.roots
    )
    return nodes, roots


def _topology_digest(topology: tuple[tuple[Any, ...], tuple[Any, ...]]) -> str:
    payload = json.dumps(
        topology,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _short_repr(value: Any, limit: int = 180) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else f"{rendered[: limit - 3]}..."


def _first_difference(left: Any, right: Any, path: str = "root") -> str:
    if isinstance(left, tuple) and isinstance(right, tuple):
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            if left_item != right_item:
                return _first_difference(left_item, right_item, f"{path}[{index}]")
        if len(left) != len(right):
            return f"{path} has different lengths: direct={len(left)}, posthoc={len(right)}"
    return f"first difference at {path}: direct={_short_repr(left)}, posthoc={_short_repr(right)}"


def _numeric_text(value: Any) -> str:
    with mp.workdps(AUDIT_PRECISION_DIGITS):
        converted = mp.mpc(value)
        return f"({mp.nstr(converted.real, 70)},{mp.nstr(converted.imag, 70)})"


def _evaluate_graph(
    graph: Graph,
    root_id: str,
    bindings: tuple[tuple[str, NumericBinding], ...],
) -> tuple[Any, bool]:
    return evaluate_pure_eml(
        dag_to_eml(graph, root_id),
        variables=dict(bindings),
        backend=NumericBackend.MPMATH,
        precision_digits=AUDIT_PRECISION_DIGITS,
    )


def _axis(
    axis: ComparisonAxis,
    direct: Callable[[], Any],
    posthoc: Callable[[], Any],
    *,
    matches: Callable[[Any, Any], bool] | None = None,
    render: Callable[[Any], str] = repr,
    mismatch_detail: Callable[[Any, Any], str] = _first_difference,
) -> AxisComparison:
    direct_error: Exception | None = None
    posthoc_error: Exception | None = None
    try:
        direct_value = direct()
    except Exception as error:  # both sides are attempted even when the first fails
        direct_value = None
        direct_error = error
    try:
        posthoc_value = posthoc()
    except Exception as error:
        posthoc_value = None
        posthoc_error = error

    if direct_error is not None or posthoc_error is not None:
        direct_text = (
            f"<{type(direct_error).__name__}: {direct_error}>"
            if direct_error is not None
            else render(direct_value)
        )
        posthoc_text = (
            f"<{type(posthoc_error).__name__}: {posthoc_error}>"
            if posthoc_error is not None
            else render(posthoc_value)
        )
        failures = []
        if direct_error is not None:
            failures.append(f"direct {type(direct_error).__name__}: {direct_error}")
        if posthoc_error is not None:
            failures.append(f"posthoc {type(posthoc_error).__name__}: {posthoc_error}")
        return AxisComparison(
            axis=axis,
            matched=False,
            direct_value=direct_text,
            posthoc_value=posthoc_text,
            detail="; ".join(failures),
            error=True,
        )

    matched = matches(direct_value, posthoc_value) if matches else direct_value == posthoc_value
    detail = "matched" if matched else mismatch_detail(direct_value, posthoc_value)
    return AxisComparison(
        axis=axis,
        matched=matched,
        direct_value=render(direct_value),
        posthoc_value=render(posthoc_value),
        detail=detail,
    )


def _evaluation_matches(
    direct: tuple[Any, bool],
    posthoc: tuple[Any, bool],
) -> bool:
    direct_value, direct_extended = direct
    posthoc_value, posthoc_extended = posthoc
    if direct_extended != posthoc_extended:
        return False
    with mp.workdps(AUDIT_PRECISION_DIGITS):
        direct_complex = mp.mpc(direct_value)
        posthoc_complex = mp.mpc(posthoc_value)
        if not all(
            mp.isfinite(component)
            for component in (
                direct_complex.real,
                direct_complex.imag,
                posthoc_complex.real,
                posthoc_complex.imag,
            )
        ):
            return False
        difference = abs(direct_complex - posthoc_complex)
        scale = max(mp.mpf(1), abs(direct_complex), abs(posthoc_complex))
        return bool(difference <= mp.mpf(_EVALUATION_TOLERANCE_TEXT) * scale)


def _evaluation_text(value: tuple[Any, bool]) -> str:
    numeric_value, extended = value
    return f"value={_numeric_text(numeric_value)}; extended_intermediate={extended}"


def _purity_text(errors: tuple[str, ...]) -> str:
    return "valid" if not errors else "; ".join(errors)


def audit_one(
    case: AuditCase,
    *,
    direct_compiler: DirectCompiler = _compile_direct,
    posthoc_compiler: PostHocCompiler = _compile_posthoc,
) -> AuditResult:
    """Audit one case and retain a terminal classification and all axis results."""

    operator_families = tuple(
        dict.fromkeys(
            operator.operator_family
            for operator in _ENABLED_APPROVED_OPERATORS
            if operator.name in case.operator_names
        )
    )
    common = {
        "case_id": case.case_id,
        "operator_names": case.operator_names,
        "operator_families": operator_families,
        "corpus_family": case.corpus_family,
        "size_bucket": case.size_bucket,
        "split": case.split,
        "domain_mode": case.domain_mode,
    }
    if case.blocked_reason is not None:
        return AuditResult(
            **common,
            status=AuditStatus.BLOCKED,
            blocker_reason=case.blocked_reason,
        )
    if case.ast is None:  # protected by AuditCase validation
        return AuditResult(
            **common,
            status=AuditStatus.FAILURE,
            failure_type="MissingAST",
            failure_message="runnable case has no AST",
        )

    direct_result: tuple[Graph, str, ConstructionStats] | None = None
    posthoc_result: tuple[Graph, str, EMLDagStatistics] | None = None
    direct_error: Exception | None = None
    posthoc_error: Exception | None = None
    try:
        direct_result = direct_compiler(case.ast)
    except Exception as error:
        direct_error = error
    try:
        posthoc_result = posthoc_compiler(case.ast)
    except Exception as error:
        posthoc_error = error

    if direct_error is not None or posthoc_error is not None:
        failure_type = (
            "MultipleConstructionFailures"
            if direct_error is not None and posthoc_error is not None
            else type(direct_error or posthoc_error).__name__
        )
        messages = []
        if direct_error is not None:
            messages.append(f"direct {type(direct_error).__name__}: {direct_error}")
        if posthoc_error is not None:
            messages.append(f"posthoc {type(posthoc_error).__name__}: {posthoc_error}")
        return AuditResult(
            **common,
            status=AuditStatus.FAILURE,
            failure_type=failure_type,
            failure_message="; ".join(messages),
        )
    if direct_result is None or posthoc_result is None:  # pragma: no cover - guarded above
        return AuditResult(
            **common,
            status=AuditStatus.FAILURE,
            failure_type="MissingConstructionResult",
            failure_message="compiler returned no result without raising an exception",
        )
    direct_graph, direct_root, _ = direct_result
    posthoc_graph, posthoc_root, _ = posthoc_result

    direct_topology: tuple[tuple[Any, ...], tuple[Any, ...]] | None = None
    posthoc_topology: tuple[tuple[Any, ...], tuple[Any, ...]] | None = None

    def direct_canonical_topology() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        nonlocal direct_topology
        if direct_topology is None:
            direct_topology = _canonical_topology(direct_graph)
        return direct_topology

    def posthoc_canonical_topology() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        nonlocal posthoc_topology
        if posthoc_topology is None:
            posthoc_topology = _canonical_topology(posthoc_graph)
        return posthoc_topology

    comparisons = (
        _axis(
            ComparisonAxis.CANONICAL_SIGNATURE,
            lambda: compute_signature(direct_graph, direct_root),
            lambda: compute_signature(posthoc_graph, posthoc_root),
            render=str,
        ),
        _axis(
            ComparisonAxis.CANONICAL_TOPOLOGY,
            direct_canonical_topology,
            posthoc_canonical_topology,
            render=_topology_digest,
        ),
        _axis(
            ComparisonAxis.NODE_COUNT,
            lambda: compute_statistics(direct_graph).node_count,
            lambda: compute_statistics(posthoc_graph).node_count,
            render=str,
        ),
        _axis(
            ComparisonAxis.CHILD_REFERENCES,
            lambda: compute_statistics(direct_graph).child_reference_count,
            lambda: compute_statistics(posthoc_graph).child_reference_count,
            render=str,
        ),
        _axis(
            ComparisonAxis.DEPTH,
            lambda: compute_statistics(direct_graph).max_depth,
            lambda: compute_statistics(posthoc_graph).max_depth,
            render=str,
        ),
        _axis(
            ComparisonAxis.EVALUATION,
            lambda: _evaluate_graph(direct_graph, direct_root, case.evaluation_bindings),
            lambda: _evaluate_graph(posthoc_graph, posthoc_root, case.evaluation_bindings),
            matches=_evaluation_matches,
            render=_evaluation_text,
        ),
        _axis(
            ComparisonAxis.PURITY,
            lambda: validate_eml_dag(direct_graph).errors,
            lambda: validate_eml_dag(posthoc_graph).errors,
            matches=lambda direct, posthoc: not direct and not posthoc,
            render=_purity_text,
        ),
    )
    if any(comparison.error for comparison in comparisons):
        status = AuditStatus.FAILURE
    elif all(comparison.matched for comparison in comparisons):
        status = AuditStatus.MATCH
    else:
        status = AuditStatus.MISMATCH
    return AuditResult(**common, status=status, comparisons=comparisons)


def _matched_coverage(
    results: Iterable[AuditResult],
    attribute: str,
) -> set[Any]:
    coverage: set[Any] = set()
    for result in results:
        if result.status is not AuditStatus.MATCH:
            continue
        value = getattr(result, attribute)
        if isinstance(value, tuple) and attribute in {"operator_names", "operator_families"}:
            coverage.update(value)
        else:
            coverage.add(value)
    return coverage


def _result_payload(result: AuditResult) -> dict[str, Any]:
    return {
        "blocker_reason": result.blocker_reason,
        "case_id": result.case_id,
        "comparisons": [
            {
                "axis": comparison.axis.value,
                "detail": comparison.detail,
                "direct_value": comparison.direct_value,
                "error": comparison.error,
                "matched": comparison.matched,
                "posthoc_value": comparison.posthoc_value,
            }
            for comparison in result.comparisons
        ],
        "corpus_family": result.corpus_family,
        "domain_mode": result.domain_mode,
        "failure_message": result.failure_message,
        "failure_type": result.failure_type,
        "operator_families": list(result.operator_families),
        "operator_names": list(result.operator_names),
        "size_bucket": list(result.size_bucket),
        "split": result.split.value,
        "status": result.status.value,
    }


def _audit_fingerprint(
    results: tuple[AuditResult, ...],
    missing: dict[str, tuple[Any, ...]],
) -> str:
    payload = {
        "missing": {
            name: [
                list(item) if isinstance(item, tuple) else getattr(item, "value", item)
                for item in values
            ]
            for name, values in missing.items()
        },
        "results": [_result_payload(result) for result in results],
        "schema_version": AUDIT_SCHEMA_VERSION,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_audit(
    cases: Iterable[AuditCase] = STRATIFIED_AUDIT_SET,
    *,
    direct_compiler: DirectCompiler = _compile_direct,
    posthoc_compiler: PostHocCompiler = _compile_posthoc,
) -> AuditSummary:
    """Run every requested case and return complete deterministic coverage."""

    requested_cases = tuple(cases)
    case_ids = [case.case_id for case in requested_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("audit case IDs must be unique")
    results = tuple(
        audit_one(
            case,
            direct_compiler=direct_compiler,
            posthoc_compiler=posthoc_compiler,
        )
        for case in requested_cases
    )

    operator_coverage = _matched_coverage(results, "operator_names")
    operator_family_coverage = _matched_coverage(results, "operator_families")
    corpus_family_coverage = _matched_coverage(results, "corpus_family")
    size_coverage = _matched_coverage(results, "size_bucket")
    split_coverage = _matched_coverage(results, "split")
    domain_coverage = _matched_coverage(results, "domain_mode")
    missing: dict[str, tuple[Any, ...]] = {
        "operators": tuple(
            name for name in REQUIRED_OPERATOR_NAMES if name not in operator_coverage
        ),
        "operator_families": tuple(
            family
            for family in REQUIRED_OPERATOR_FAMILIES
            if family not in operator_family_coverage
        ),
        "corpus_families": tuple(
            family for family in REQUIRED_CORPUS_FAMILIES if family not in corpus_family_coverage
        ),
        "size_buckets": tuple(
            bucket for bucket in AUDIT_SIZE_BUCKETS if bucket not in size_coverage
        ),
        "splits": tuple(split for split in REQUIRED_SPLITS if split not in split_coverage),
        "domain_modes": tuple(
            mode for mode in REQUIRED_DOMAIN_MODES if mode not in domain_coverage
        ),
    }
    return AuditSummary(
        schema_version=AUDIT_SCHEMA_VERSION,
        results=results,
        missing_operators=missing["operators"],
        missing_operator_families=missing["operator_families"],
        missing_corpus_families=missing["corpus_families"],
        missing_size_buckets=missing["size_buckets"],
        missing_splits=missing["splits"],
        missing_domain_modes=missing["domain_modes"],
        fingerprint=_audit_fingerprint(results, missing),
    )
