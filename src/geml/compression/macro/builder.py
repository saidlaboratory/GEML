"""Deterministic construction of transparent macro DAGs from validated ASTs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from geml.ast.statistics import structural_signature
from geml.contracts.ast import ASTNode, ASTTree
from geml.dag.hashcons import HashConsTable, InternedNode
from geml.eml.compiler_core import CompilerMode, require_compiler_mode
from geml.graph.schema import MACRO_FAMILY, Graph
from geml.interfaces.eml_dag_cost import (
    EMLDagCostResult,
    EMLDagCostStatus,
    compute_eml_dag_cost,
)
from geml.spec.operators import OPERATOR_REGISTRY, EMLConstructionStatus

from .schema import (
    MACRO_NODE_KIND,
    MACRO_RULE_BY_OPERATOR,
    MacroExpansionCost,
    MacroGraphRecord,
    macro_node_value,
    macro_payload_errors,
    macro_representation_mode,
)


class MacroBuildStatus(StrEnum):
    """Terminal status for one macro-graph build request."""

    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED = "unsupported"
    FAILURE = "failure"


class MacroBuildFailureStage(StrEnum):
    """Stage at which a non-successful macro build terminated."""

    INPUT_VALIDATION = "input_validation"
    GRAPH_CONSTRUCTION = "graph_construction"
    EXPANSION_COST = "expansion_cost"


@dataclass(frozen=True, slots=True)
class MacroBuildResult:
    """A complete macro graph or one retained terminal failure."""

    status: MacroBuildStatus
    expression_id: str | None
    compiler_mode: CompilerMode | None
    macro_graph: MacroGraphRecord | None = None
    failure_stage: MacroBuildFailureStage | None = None
    error_type: str | None = None
    error_message: str | None = None
    expansion_cost_result: EMLDagCostResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, MacroBuildStatus):
            raise TypeError("status must be a MacroBuildStatus")
        if self.status is MacroBuildStatus.SUCCESS:
            if not isinstance(self.macro_graph, MacroGraphRecord):
                raise ValueError("a successful build must contain a MacroGraphRecord")
            if self.failure_stage is not None or self.error_type is not None or self.error_message:
                raise ValueError("a successful build cannot contain failure details")
            if self.expansion_cost_result is not None:
                raise ValueError("successful results expose cost through the macro record")
            return
        if self.macro_graph is not None:
            raise ValueError("a failed build cannot expose a partial macro record")
        if not isinstance(self.failure_stage, MacroBuildFailureStage):
            raise ValueError("a failed build requires a failure stage")
        if not isinstance(self.error_type, str) or not self.error_type.strip():
            raise ValueError("a failed build requires an error type")
        if not isinstance(self.error_message, str) or not self.error_message.strip():
            raise ValueError("a failed build requires an error message")


def _failure(
    status: MacroBuildStatus,
    stage: MacroBuildFailureStage,
    *,
    expression_id: str | None,
    compiler_mode: CompilerMode | None,
    error: Exception | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    expansion_cost_result: EMLDagCostResult | None = None,
) -> MacroBuildResult:
    if status is MacroBuildStatus.SUCCESS:
        raise ValueError("_failure cannot construct a successful result")
    if error is not None:
        error_type = type(error).__name__
        error_message = str(error).strip() or f"{error_type} reported no message"
    if error_type is None or error_message is None:
        raise ValueError("failure construction requires an exception or explicit details")
    return MacroBuildResult(
        status=status,
        expression_id=expression_id,
        compiler_mode=compiler_mode,
        failure_stage=stage,
        error_type=error_type,
        error_message=error_message,
        expansion_cost_result=expansion_cost_result,
    )


def _resolve_mode(mode: CompilerMode | None) -> CompilerMode:
    return CompilerMode.OFFICIAL_V4 if mode is None else require_compiler_mode(mode)


def _ordered_children(tree: ASTTree) -> dict[str, tuple[str, ...]]:
    child_slots: dict[str, dict[int, str]] = {node.node_id: {} for node in tree.nodes}
    for edge in tree.edges:
        child_slots[edge.source_id][edge.child_slot] = edge.target_id
    return {
        node.node_id: tuple(child_slots[node.node_id][slot] for slot in range(node.arity))
        for node in tree.nodes
    }


def _postorder(tree: ASTTree, children: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    order: list[str] = []
    stack: list[tuple[str, bool]] = [(tree.root_id, False)]
    while stack:
        node_id, leaving = stack.pop()
        if leaving:
            order.append(node_id)
            continue
        stack.append((node_id, True))
        stack.extend((child_id, False) for child_id in reversed(children[node_id]))
    return tuple(order)


def _node_errors(node: ASTNode) -> tuple[str, ...]:
    operator = OPERATOR_REGISTRY.get(node.label)
    if operator is None:
        return (f"AST node {node.node_id!r} uses unknown operator {node.label!r}",)
    if (
        not operator.enabled_for_generation
        or operator.eml_construction_status is not EMLConstructionStatus.APPROVED
        or node.label not in MACRO_RULE_BY_OPERATOR
    ):
        return (f"AST node {node.node_id!r} uses unsupported operator {node.label!r}",)
    spec = MACRO_RULE_BY_OPERATOR[node.label]
    errors: list[str] = []
    expected_kind = "leaf" if spec.arity == 0 else "operator"
    if node.node_kind != expected_kind:
        errors.append(
            f"AST node {node.node_id!r} operator {node.label!r} requires "
            f"node_kind {expected_kind!r}"
        )
    if node.arity != spec.arity:
        errors.append(
            f"AST node {node.node_id!r} arity {node.arity} does not match "
            f"{node.label!r} arity {spec.arity}"
        )
    errors.extend(
        f"AST node {node.node_id!r}: {message}"
        for message in macro_payload_errors(node.label, node.value)
    )
    return tuple(errors)


def _construct_graph(
    tree: ASTTree,
    *,
    mode: CompilerMode,
) -> tuple[Graph, dict[str, str], dict[str, tuple[str, ...]]]:
    children = _ordered_children(tree)
    nodes = {node.node_id: node for node in tree.nodes}
    table = HashConsTable(MACRO_FAMILY)
    source_refs: dict[str, InternedNode] = {}

    for source_id in _postorder(tree, children):
        source = nodes[source_id]
        spec = MACRO_RULE_BY_OPERATOR[source.label]
        child_refs = tuple(source_refs[child_id] for child_id in children[source_id])
        source_refs[source_id] = table.intern(
            kind=MACRO_NODE_KIND,
            label=source.label,
            value=macro_node_value(spec.rule, source.value),
            children=child_refs,
        )

    graph = table.to_graph(
        source_refs[tree.root_id],
        root_id=tree.expression_id,
        representation_mode=macro_representation_mode(mode),
    )
    source_to_macro = {node.node_id: source_refs[node.node_id].node_id for node in tree.nodes}
    inverse_lists: dict[str, list[str]] = {node_id: [] for node_id in graph.nodes}
    for source_id in sorted(source_to_macro):
        inverse_lists[source_to_macro[source_id]].append(source_id)
    macro_to_source = {
        macro_id: tuple(source_ids) for macro_id, source_ids in sorted(inverse_lists.items())
    }
    return graph, source_to_macro, macro_to_source


def build_macro_graph(
    tree: ASTTree,
    *,
    compiler_mode: CompilerMode | None = None,
) -> MacroBuildResult:
    """Build one exact transparent macro DAG or retain a terminal failure.

    ``compiler_mode=None`` selects :attr:`CompilerMode.OFFICIAL_V4`.
    Structurally identical source subtrees are hash-consed, while both
    occurrence directions remain available in immutable provenance sidecars.
    """

    expression_id = tree.expression_id if isinstance(tree, ASTTree) else None
    try:
        mode = _resolve_mode(compiler_mode)
    except Exception as error:
        return _failure(
            MacroBuildStatus.INVALID_INPUT,
            MacroBuildFailureStage.INPUT_VALIDATION,
            expression_id=expression_id,
            compiler_mode=None,
            error=error,
        )
    if not isinstance(tree, ASTTree):
        return _failure(
            MacroBuildStatus.INVALID_INPUT,
            MacroBuildFailureStage.INPUT_VALIDATION,
            expression_id=None,
            compiler_mode=mode,
            error=TypeError("tree must be a validated ASTTree"),
        )

    errors = tuple(message for node in tree.nodes for message in _node_errors(node))
    if errors:
        unsupported = any(
            "unknown operator" in error or "unsupported operator" in error for error in errors
        )
        return _failure(
            MacroBuildStatus.UNSUPPORTED if unsupported else MacroBuildStatus.INVALID_INPUT,
            MacroBuildFailureStage.INPUT_VALIDATION,
            expression_id=tree.expression_id,
            compiler_mode=mode,
            error_type="MacroSourceValidationError",
            error_message="; ".join(errors),
        )

    try:
        graph, source_to_macro, macro_to_source = _construct_graph(tree, mode=mode)
    except Exception as error:
        return _failure(
            MacroBuildStatus.FAILURE,
            MacroBuildFailureStage.GRAPH_CONSTRUCTION,
            expression_id=tree.expression_id,
            compiler_mode=mode,
            error=error,
        )

    cost_result = compute_eml_dag_cost(tree, compiler_mode=mode)
    if cost_result.status is not EMLDagCostStatus.SUCCESS:
        status = (
            MacroBuildStatus.UNSUPPORTED
            if cost_result.status is EMLDagCostStatus.UNSUPPORTED
            else MacroBuildStatus.INVALID_INPUT
            if cost_result.status is EMLDagCostStatus.INVALID_INPUT
            else MacroBuildStatus.FAILURE
        )
        return _failure(
            status,
            MacroBuildFailureStage.EXPANSION_COST,
            expression_id=tree.expression_id,
            compiler_mode=mode,
            error_type=cost_result.error_type or "EMLDagCostError",
            error_message=cost_result.error_message or "EML DAG cost failed without details",
            expansion_cost_result=cost_result,
        )

    try:
        record = MacroGraphRecord(
            graph=graph,
            compiler_mode=mode,
            source_expression_id=tree.expression_id,
            source_root_id=tree.root_id,
            source_ast_signature=structural_signature(tree),
            source_to_macro_node=source_to_macro,
            macro_to_source_nodes=macro_to_source,
            expansion_cost=MacroExpansionCost.from_cost_result(cost_result),
        )
    except Exception as error:
        return _failure(
            MacroBuildStatus.FAILURE,
            MacroBuildFailureStage.GRAPH_CONSTRUCTION,
            expression_id=tree.expression_id,
            compiler_mode=mode,
            error=error,
        )
    return MacroBuildResult(
        status=MacroBuildStatus.SUCCESS,
        expression_id=tree.expression_id,
        compiler_mode=mode,
        macro_graph=record,
    )
