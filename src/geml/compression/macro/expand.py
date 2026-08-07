"""Exact expansion and independent validation of official macro DAGs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from geml.contracts.ast import ASTTree
from geml.dag.direct_eml import (
    DirectEMLCompiler,
    UnsupportedASTOperatorError,
    compile_ast_to_eml_dag,
)
from geml.dag.eml import validate_eml_dag
from geml.dag.hashcons import InternedNode
from geml.eml.compiler_core import CompilerMode
from geml.graph.schema import Graph, compute_statistics
from geml.graph.signatures import compute_signature

from .schema import (
    MACRO_PAYLOAD_FIELD,
    MACRO_RULE_FIELD,
    MacroGraphRecord,
    MacroRule,
    pure_eml_representation_mode,
)
from .validate import validate_macro_graph, validate_macro_source_binding


class MacroExpansionStatus(StrEnum):
    """Terminal status for one macro-expansion validation."""

    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED = "unsupported"
    MISMATCH = "mismatch"
    FAILURE = "failure"


class MacroExpansionFailureStage(StrEnum):
    """Stage at which a non-successful validation terminated."""

    MACRO_VALIDATION = "macro_validation"
    SOURCE_VALIDATION = "source_validation"
    EXPANSION = "expansion"
    PURE_EML_VALIDATION = "pure_eml_validation"
    REFERENCE_COMPILATION = "reference_compilation"
    COMPARISON = "comparison"


class MacroExpansionError(ValueError):
    """A stored macro graph cannot be expanded to a valid pure-EML DAG."""


@dataclass(frozen=True, slots=True)
class PureEMLIdentity:
    """Canonical structural identity and exact DAG costs for one pure EML root."""

    eml_dag_node_count: int
    eml_dag_child_reference_count: int
    eml_dag_depth: int
    root_signature: str
    representation_mode: str


@dataclass(frozen=True, slots=True)
class MacroExpansionResult:
    """A successful comparison or one retained terminal failure."""

    status: MacroExpansionStatus
    expression_id: str | None
    compiler_mode: CompilerMode | None
    expanded_identity: PureEMLIdentity | None = None
    reference_identity: PureEMLIdentity | None = None
    expanded_graph: Graph | None = None
    failure_stage: MacroExpansionFailureStage | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, MacroExpansionStatus):
            raise TypeError("status must be a MacroExpansionStatus")
        if self.status is MacroExpansionStatus.SUCCESS:
            if not isinstance(self.expanded_identity, PureEMLIdentity) or not isinstance(
                self.reference_identity, PureEMLIdentity
            ):
                raise ValueError("successful validation requires expanded and reference identities")
            if self.expanded_identity != self.reference_identity:
                raise ValueError("successful validation identities must be exact matches")
            if (
                self.failure_stage is not None
                or self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError("successful validation cannot contain failure details")
            return
        if self.expanded_graph is not None:
            raise ValueError("failed validation cannot expose a partial expanded graph")
        if not isinstance(self.failure_stage, MacroExpansionFailureStage):
            raise ValueError("failed validation requires a failure stage")
        if not isinstance(self.error_type, str) or not self.error_type.strip():
            raise ValueError("failed validation requires an error type")
        if not isinstance(self.error_message, str) or not self.error_message.strip():
            raise ValueError("failed validation requires an error message")


def _failure(
    status: MacroExpansionStatus,
    stage: MacroExpansionFailureStage,
    *,
    expression_id: str | None,
    compiler_mode: CompilerMode | None,
    error: Exception | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    expanded_identity: PureEMLIdentity | None = None,
    reference_identity: PureEMLIdentity | None = None,
) -> MacroExpansionResult:
    if status is MacroExpansionStatus.SUCCESS:
        raise ValueError("_failure cannot construct a successful result")
    if error is not None:
        error_type = type(error).__name__
        error_message = str(error).strip() or f"{error_type} reported no message"
    if error_type is None or error_message is None:
        raise ValueError("failure construction requires an exception or explicit details")
    return MacroExpansionResult(
        status=status,
        expression_id=expression_id,
        compiler_mode=compiler_mode,
        expanded_identity=expanded_identity,
        reference_identity=reference_identity,
        failure_stage=stage,
        error_type=error_type,
        error_message=error_message,
    )


def _postorder(graph: Graph) -> tuple[str, ...]:
    order: list[str] = []
    completed: set[str] = set()
    stack: list[tuple[str, bool]] = [(graph.roots[0].target_id, False)]
    while stack:
        node_id, leaving = stack.pop()
        if node_id in completed:
            continue
        if leaving:
            completed.add(node_id)
            order.append(node_id)
            continue
        stack.append((node_id, True))
        node = graph.nodes[node_id]
        for child in reversed(sorted(node.children, key=lambda ref: ref.slot)):
            if child.target_id not in completed:
                stack.append((child.target_id, False))
    return tuple(order)


def _expand_node(
    compiler: DirectEMLCompiler,
    rule: MacroRule,
    payload: object,
    children: tuple[InternedNode, ...],
) -> InternedNode:
    if rule is MacroRule.VARIABLE:
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            raise MacroExpansionError("variable expansion requires a named payload")
        return compiler.emit_variable(payload["name"])
    if rule is MacroRule.ONE:
        return compiler.emit_one()
    if rule is MacroRule.INTEGER:
        if type(payload) is not int:
            raise MacroExpansionError("integer expansion requires an exact integer payload")
        return compiler.emit_integer(payload)
    if rule is MacroRule.RATIONAL:
        if not isinstance(payload, dict):
            raise MacroExpansionError("rational expansion requires an object payload")
        numerator = payload.get("numerator")
        denominator = payload.get("denominator")
        if type(numerator) is not int or type(denominator) is not int:
            raise MacroExpansionError(
                "rational expansion requires exact integer numerator and denominator"
            )
        return compiler.emit_rational(
            numerator,
            denominator,
        )

    if rule is MacroRule.NEGATE:
        return compiler.emit_negate(children[0])
    if rule is MacroRule.EXP:
        return compiler.emit_exp(children[0])
    if rule is MacroRule.LOG:
        return compiler.emit_log(children[0])
    if rule is MacroRule.SIN:
        return compiler.emit_sin(children[0])
    if rule is MacroRule.COS:
        return compiler.emit_cos(children[0])
    if rule is MacroRule.TAN:
        return compiler.emit_tan(children[0])
    if rule is MacroRule.SINH:
        return compiler.emit_sinh(children[0])
    if rule is MacroRule.COSH:
        return compiler.emit_cosh(children[0])
    if rule is MacroRule.TANH:
        return compiler.emit_tanh(children[0])
    if rule is MacroRule.ADD:
        return compiler.emit_add(children[0], children[1])
    if rule is MacroRule.SUBTRACT:
        return compiler.emit_subtract(children[0], children[1])
    if rule is MacroRule.MULTIPLY:
        return compiler.emit_multiply(children[0], children[1])
    if rule is MacroRule.DIVIDE:
        return compiler.emit_divide(children[0], children[1])
    if rule is MacroRule.POWER:
        return compiler.emit_power(children[0], children[1])
    raise MacroExpansionError(f"no expansion dispatch exists for {rule.value!r}")


def expand_macro_graph(record: MacroGraphRecord) -> Graph:
    """Recursively expand stored rules and ordered bindings into pure EML."""

    validation = validate_macro_graph(record)
    if not validation.valid:
        raise MacroExpansionError(
            "cannot expand an invalid macro graph: " + "; ".join(validation.errors)
        )

    compiler = DirectEMLCompiler(mode=record.compiler_mode)
    expanded: dict[str, InternedNode] = {}
    for node_id in _postorder(record.graph):
        node = record.graph.nodes[node_id]
        value = node.value
        if not isinstance(value, dict):  # pragma: no cover - validation guarantees this
            raise MacroExpansionError(f"macro node {node_id!r} has no rule object")
        rule = MacroRule(value[MACRO_RULE_FIELD])
        payload = value[MACRO_PAYLOAD_FIELD]
        children = tuple(
            expanded[child.target_id] for child in sorted(node.children, key=lambda ref: ref.slot)
        )
        expanded[node_id] = _expand_node(compiler, rule, payload, children)

    macro_root = record.graph.roots[0]
    result = compiler.table.to_graph(
        expanded[macro_root.target_id],
        root_id=record.source_expression_id,
        representation_mode=pure_eml_representation_mode(record.compiler_mode),
    )
    validation = validate_eml_dag(result)
    if not validation.valid:  # pragma: no cover - protects the public boundary
        raise MacroExpansionError(
            "macro expansion produced invalid pure EML: " + "; ".join(validation.errors)
        )
    return result


def _identity(graph: Graph, *, expected_mode: str) -> PureEMLIdentity:
    validation = validate_eml_dag(graph)
    if not validation.valid:
        raise MacroExpansionError("pure-EML DAG validation failed: " + "; ".join(validation.errors))
    if len(graph.roots) != 1:
        raise MacroExpansionError("a pure-EML comparison requires exactly one root")
    root = graph.roots[0]
    if root.representation_mode != expected_mode:
        raise MacroExpansionError(
            f"expected representation mode {expected_mode!r}, observed {root.representation_mode!r}"
        )
    statistics = compute_statistics(graph)
    return PureEMLIdentity(
        eml_dag_node_count=statistics.node_count,
        eml_dag_child_reference_count=statistics.child_reference_count,
        eml_dag_depth=statistics.max_depth,
        root_signature=compute_signature(graph, root.target_id),
        representation_mode=root.representation_mode,
    )


def _stored_cost_mismatches(
    record: MacroGraphRecord,
    identity: PureEMLIdentity,
) -> tuple[str, ...]:
    cost = record.expansion_cost
    mismatches: list[str] = []
    comparisons = (
        ("node count", cost.eml_dag_node_count, identity.eml_dag_node_count),
        (
            "child-reference count",
            cost.eml_dag_child_reference_count,
            identity.eml_dag_child_reference_count,
        ),
        ("depth", cost.eml_dag_depth, identity.eml_dag_depth),
        ("root signature", cost.root_signature, identity.root_signature),
        (
            "representation mode",
            cost.representation_mode,
            identity.representation_mode,
        ),
    )
    for label, stored, observed in comparisons:
        if stored != observed:
            mismatches.append(f"stored expansion {label} {stored!r} != observed {observed!r}")
    return tuple(mismatches)


def validate_macro_expansion(
    record: object,
    source_ast: object,
    *,
    retain_expanded_graph: bool = False,
) -> MacroExpansionResult:
    """Expand and compare against an independent official source-AST compile.

    The default result retains only identities, making a streaming 250k-row
    audit bounded by the largest current row. Set ``retain_expanded_graph`` to
    true only when a caller explicitly needs the successful expanded graph.
    """

    if not isinstance(retain_expanded_graph, bool):
        return _failure(
            MacroExpansionStatus.INVALID_INPUT,
            MacroExpansionFailureStage.MACRO_VALIDATION,
            expression_id=None,
            compiler_mode=None,
            error=TypeError("retain_expanded_graph must be a bool"),
        )
    expression_id = record.source_expression_id if isinstance(record, MacroGraphRecord) else None
    mode = record.compiler_mode if isinstance(record, MacroGraphRecord) else None

    macro_validation = validate_macro_graph(record)
    if not macro_validation.valid:
        return _failure(
            MacroExpansionStatus.INVALID_INPUT,
            MacroExpansionFailureStage.MACRO_VALIDATION,
            expression_id=expression_id,
            compiler_mode=mode,
            error_type="MacroGraphValidationError",
            error_message="; ".join(macro_validation.errors),
        )
    if not isinstance(record, MacroGraphRecord):  # pragma: no cover - validated above
        raise AssertionError("validated macro record has wrong type")

    source_validation = validate_macro_source_binding(record, source_ast)
    if not source_validation.valid:
        return _failure(
            MacroExpansionStatus.INVALID_INPUT,
            MacroExpansionFailureStage.SOURCE_VALIDATION,
            expression_id=expression_id,
            compiler_mode=mode,
            error_type="MacroSourceBindingError",
            error_message="; ".join(source_validation.errors),
        )
    if not isinstance(source_ast, ASTTree):  # pragma: no cover - validated above
        raise AssertionError("validated source AST has wrong type")

    expected_mode = pure_eml_representation_mode(record.compiler_mode)
    try:
        expanded_graph = expand_macro_graph(record)
    except Exception as error:
        return _failure(
            MacroExpansionStatus.FAILURE,
            MacroExpansionFailureStage.EXPANSION,
            expression_id=expression_id,
            compiler_mode=mode,
            error=error,
        )
    try:
        expanded_identity = _identity(expanded_graph, expected_mode=expected_mode)
    except Exception as error:
        return _failure(
            MacroExpansionStatus.FAILURE,
            MacroExpansionFailureStage.PURE_EML_VALIDATION,
            expression_id=expression_id,
            compiler_mode=mode,
            error=error,
        )

    try:
        reference_graph, reference_root_id, construction = compile_ast_to_eml_dag(
            source_ast,
            mode=record.compiler_mode,
        )
    except UnsupportedASTOperatorError as error:
        return _failure(
            MacroExpansionStatus.UNSUPPORTED,
            MacroExpansionFailureStage.REFERENCE_COMPILATION,
            expression_id=expression_id,
            compiler_mode=mode,
            error=error,
            expanded_identity=expanded_identity,
        )
    except (TypeError, ValueError) as error:
        return _failure(
            MacroExpansionStatus.INVALID_INPUT,
            MacroExpansionFailureStage.REFERENCE_COMPILATION,
            expression_id=expression_id,
            compiler_mode=mode,
            error=error,
            expanded_identity=expanded_identity,
        )
    except Exception as error:
        return _failure(
            MacroExpansionStatus.FAILURE,
            MacroExpansionFailureStage.REFERENCE_COMPILATION,
            expression_id=expression_id,
            compiler_mode=mode,
            error=error,
            expanded_identity=expanded_identity,
        )

    try:
        inconsistent_provenance = (
            construction.compiler_mode is not record.compiler_mode
            or construction.representation_mode != expected_mode
            or len(reference_graph.roots) != 1
            or reference_graph.roots[0].target_id != reference_root_id
        )
    except Exception as error:
        return _failure(
            MacroExpansionStatus.FAILURE,
            MacroExpansionFailureStage.REFERENCE_COMPILATION,
            expression_id=expression_id,
            compiler_mode=mode,
            error=error,
            expanded_identity=expanded_identity,
        )
    if inconsistent_provenance:
        return _failure(
            MacroExpansionStatus.FAILURE,
            MacroExpansionFailureStage.REFERENCE_COMPILATION,
            expression_id=expression_id,
            compiler_mode=mode,
            error_type="ReferenceCompilerProvenanceError",
            error_message="reference compiler returned inconsistent provenance",
            expanded_identity=expanded_identity,
        )
    try:
        reference_identity = _identity(reference_graph, expected_mode=expected_mode)
    except Exception as error:
        return _failure(
            MacroExpansionStatus.FAILURE,
            MacroExpansionFailureStage.PURE_EML_VALIDATION,
            expression_id=expression_id,
            compiler_mode=mode,
            error=error,
            expanded_identity=expanded_identity,
        )

    mismatches = list(_stored_cost_mismatches(record, expanded_identity))
    if expanded_identity != reference_identity:
        mismatches.append("macro expansion identity does not match independent source-AST compile")
    if mismatches:
        return _failure(
            MacroExpansionStatus.MISMATCH,
            MacroExpansionFailureStage.COMPARISON,
            expression_id=expression_id,
            compiler_mode=mode,
            error_type="MacroExpansionMismatch",
            error_message="; ".join(mismatches),
            expanded_identity=expanded_identity,
            reference_identity=reference_identity,
        )

    return MacroExpansionResult(
        status=MacroExpansionStatus.SUCCESS,
        expression_id=expression_id,
        compiler_mode=mode,
        expanded_identity=expanded_identity,
        reference_identity=reference_identity,
        expanded_graph=expanded_graph if retain_expanded_graph else None,
    )


def iter_validate_macro_expansions(
    rows: Iterable[tuple[object, object]],
    *,
    retain_expanded_graph: bool = False,
) -> Iterator[MacroExpansionResult]:
    """Validate rows lazily, retaining every success, mismatch, and failure."""

    if not isinstance(retain_expanded_graph, bool):
        raise TypeError("retain_expanded_graph must be a bool")
    for row in rows:
        try:
            record, source_ast = row
        except (TypeError, ValueError) as error:
            yield _failure(
                MacroExpansionStatus.INVALID_INPUT,
                MacroExpansionFailureStage.MACRO_VALIDATION,
                expression_id=None,
                compiler_mode=None,
                error=error,
            )
            continue
        yield validate_macro_expansion(
            record,
            source_ast,
            retain_expanded_graph=retain_expanded_graph,
        )
