"""Frozen exact pure-EML DAG cost boundary for later extraction stages.

The interface accepts only the project's validated source AST contract or an
already-materialized pure-EML tree.  Source ASTs use the direct hash-consing
compiler, so computing their DAG cost never requires materializing the expanded
EML tree.  Provided EML trees use exact post-hoc structural sharing.

This module deliberately has no dependency on Goal 3 experiment or analysis
code and performs no file I/O.  A successful result is structural evidence
only; it does not assert semantic equivalence or domain validity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from geml.contracts.ast import ASTTree
from geml.dag.direct_eml import (
    UnsupportedASTOperatorError,
    compile_ast_to_eml_dag,
)
from geml.dag.eml import eml_to_dag, validate_eml_dag
from geml.eml.compiler_core import CompilerMode, require_compiler_mode
from geml.eml.ir import EMLTerm, is_eml_term
from geml.eml.validate import PureEMLValidationError
from geml.graph.schema import Graph, compute_statistics
from geml.graph.signatures import compute_signature

_PROVIDED_TREE_REPRESENTATION_MODE = "pure_eml:provided_tree"
_POSTHOC_CONSTRUCTION_PATH = "posthoc_hashcons"


class EMLDagCostStatus(StrEnum):
    """Terminal status for one exact cost request."""

    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED = "unsupported"
    FAILURE = "failure"


class EMLDagCostInputKind(StrEnum):
    """Validated input representation accepted by the cost boundary."""

    SOURCE_AST = "source_ast"
    PURE_EML_TREE = "pure_eml_tree"


class EMLDagCostFailureStage(StrEnum):
    """Stage at which a non-successful request terminated."""

    INPUT_VALIDATION = "input_validation"
    COMPILATION = "compilation"
    DAG_VALIDATION = "dag_validation"
    COST_COMPUTATION = "cost_computation"


def _is_root_signature(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class EMLDagCostResult:
    """Exact immutable cost or a complete retained terminal failure.

    ``root_signature`` identifies exact ordered structure, not semantic
    equivalence.  ``compiler_mode`` is intentionally absent for a provided EML
    tree because a recursive value does not carry trustworthy compiler
    provenance.
    """

    status: EMLDagCostStatus
    input_kind: EMLDagCostInputKind | None
    eml_dag_node_count: int | None = None
    eml_dag_child_reference_count: int | None = None
    eml_dag_depth: int | None = None
    root_signature: str | None = None
    compiler_mode: CompilerMode | None = None
    representation_mode: str | None = None
    construction_path: str | None = None
    failure_stage: EMLDagCostFailureStage | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Enforce success/failure field invariants at the public boundary."""

        if not isinstance(self.status, EMLDagCostStatus):
            raise TypeError("status must be an EMLDagCostStatus")
        if self.input_kind is not None and not isinstance(
            self.input_kind,
            EMLDagCostInputKind,
        ):
            raise TypeError("input_kind must be an EMLDagCostInputKind or None")

        cost_fields = (
            self.eml_dag_node_count,
            self.eml_dag_child_reference_count,
            self.eml_dag_depth,
            self.root_signature,
        )
        if self.status is EMLDagCostStatus.SUCCESS:
            if self.input_kind is None or any(value is None for value in cost_fields):
                raise ValueError("a successful cost result must contain input identity and costs")
            if type(self.eml_dag_node_count) is not int or self.eml_dag_node_count < 1:
                raise ValueError("successful DAG node count must be a positive exact integer")
            if (
                type(self.eml_dag_child_reference_count) is not int
                or self.eml_dag_child_reference_count < 0
            ):
                raise ValueError(
                    "successful child-reference count must be a nonnegative exact integer"
                )
            if type(self.eml_dag_depth) is not int or self.eml_dag_depth < 0:
                raise ValueError("successful cost counts must be nonnegative exact integers")
            if not _is_root_signature(self.root_signature):
                raise ValueError("root_signature must be a lowercase SHA-256 hex digest")
            if (
                not isinstance(self.representation_mode, str)
                or not self.representation_mode.strip()
            ):
                raise ValueError("a successful cost result requires a representation mode")
            if not isinstance(self.construction_path, str) or not self.construction_path.strip():
                raise ValueError("a successful cost result requires a construction path")
            if self.input_kind is EMLDagCostInputKind.SOURCE_AST:
                if not isinstance(self.compiler_mode, CompilerMode):
                    raise ValueError("a source-AST cost requires an explicit compiler mode")
            elif self.compiler_mode is not None:
                raise ValueError("a provided EML tree cannot claim compiler provenance")
            if any(
                value is not None
                for value in (self.failure_stage, self.error_type, self.error_message)
            ):
                raise ValueError("a successful cost result cannot contain failure details")
            return

        if any(value is not None for value in cost_fields):
            raise ValueError("a non-successful cost result cannot expose partial costs")
        if not isinstance(self.failure_stage, EMLDagCostFailureStage):
            raise ValueError("a non-successful cost result requires a failure stage")
        if not isinstance(self.error_type, str) or not self.error_type.strip():
            raise ValueError("a non-successful cost result requires an error type")
        if not isinstance(self.error_message, str) or not self.error_message.strip():
            raise ValueError("a non-successful cost result requires an error message")


def _failure(
    status: EMLDagCostStatus,
    stage: EMLDagCostFailureStage,
    *,
    input_kind: EMLDagCostInputKind | None,
    compiler_mode: CompilerMode | None = None,
    representation_mode: str | None = None,
    construction_path: str | None = None,
    error: Exception | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> EMLDagCostResult:
    if status is EMLDagCostStatus.SUCCESS:
        raise ValueError("_failure cannot construct a successful result")
    if error is not None:
        error_type = type(error).__name__
        error_message = str(error).strip() or f"{error_type} reported no message"
    if error_type is None or error_message is None:
        raise ValueError("failure construction requires an exception or explicit details")
    return EMLDagCostResult(
        status=status,
        input_kind=input_kind,
        compiler_mode=compiler_mode,
        representation_mode=representation_mode,
        construction_path=construction_path,
        failure_stage=stage,
        error_type=error_type,
        error_message=error_message,
    )


def _cost_from_graph(
    graph: Graph,
    root_node_id: str,
    *,
    input_kind: EMLDagCostInputKind,
    compiler_mode: CompilerMode | None,
    expected_representation_mode: str,
    construction_path: str,
    expected_node_count: int | None = None,
) -> EMLDagCostResult:
    try:
        validation = validate_eml_dag(graph)
    except Exception as error:
        return _failure(
            EMLDagCostStatus.FAILURE,
            EMLDagCostFailureStage.DAG_VALIDATION,
            input_kind=input_kind,
            compiler_mode=compiler_mode,
            representation_mode=expected_representation_mode,
            construction_path=construction_path,
            error=error,
        )
    if not validation.valid:
        return _failure(
            EMLDagCostStatus.FAILURE,
            EMLDagCostFailureStage.DAG_VALIDATION,
            input_kind=input_kind,
            compiler_mode=compiler_mode,
            representation_mode=expected_representation_mode,
            construction_path=construction_path,
            error_type="EMLDagValidationError",
            error_message=(
                "; ".join(validation.errors)
                or "EML-DAG validation failed without diagnostic details"
            ),
        )
    if len(graph.roots) != 1:
        return _failure(
            EMLDagCostStatus.FAILURE,
            EMLDagCostFailureStage.DAG_VALIDATION,
            input_kind=input_kind,
            compiler_mode=compiler_mode,
            representation_mode=expected_representation_mode,
            construction_path=construction_path,
            error_type="EMLDagRootError",
            error_message="an exact EML-DAG cost requires exactly one graph root",
        )

    root = graph.roots[0]
    if root.target_id != root_node_id:
        return _failure(
            EMLDagCostStatus.FAILURE,
            EMLDagCostFailureStage.DAG_VALIDATION,
            input_kind=input_kind,
            compiler_mode=compiler_mode,
            representation_mode=expected_representation_mode,
            construction_path=construction_path,
            error_type="EMLDagRootError",
            error_message="the returned root node does not match the graph root reference",
        )
    if root.representation_mode != expected_representation_mode:
        return _failure(
            EMLDagCostStatus.FAILURE,
            EMLDagCostFailureStage.DAG_VALIDATION,
            input_kind=input_kind,
            compiler_mode=compiler_mode,
            representation_mode=expected_representation_mode,
            construction_path=construction_path,
            error_type="EMLDagRepresentationModeError",
            error_message=(
                f"expected representation mode {expected_representation_mode!r}, "
                f"observed {root.representation_mode!r}"
            ),
        )

    try:
        statistics = compute_statistics(graph)
        signature = compute_signature(graph, root_node_id)
    except Exception as error:
        return _failure(
            EMLDagCostStatus.FAILURE,
            EMLDagCostFailureStage.COST_COMPUTATION,
            input_kind=input_kind,
            compiler_mode=compiler_mode,
            representation_mode=expected_representation_mode,
            construction_path=construction_path,
            error=error,
        )
    if expected_node_count is not None and statistics.node_count != expected_node_count:
        return _failure(
            EMLDagCostStatus.FAILURE,
            EMLDagCostFailureStage.COST_COMPUTATION,
            input_kind=input_kind,
            compiler_mode=compiler_mode,
            representation_mode=expected_representation_mode,
            construction_path=construction_path,
            error_type="EMLDagStatisticsMismatch",
            error_message=(
                f"construction reported {expected_node_count} nodes, "
                f"but the validated graph contains {statistics.node_count}"
            ),
        )

    return EMLDagCostResult(
        status=EMLDagCostStatus.SUCCESS,
        input_kind=input_kind,
        eml_dag_node_count=statistics.node_count,
        eml_dag_child_reference_count=statistics.child_reference_count,
        eml_dag_depth=statistics.max_depth,
        root_signature=signature,
        compiler_mode=compiler_mode,
        representation_mode=root.representation_mode,
        construction_path=construction_path,
    )


def _cost_source_ast(
    tree: ASTTree,
    compiler_mode: CompilerMode | None,
) -> EMLDagCostResult:
    try:
        resolved_mode = (
            CompilerMode.OFFICIAL_V4
            if compiler_mode is None
            else require_compiler_mode(compiler_mode)
        )
    except Exception as error:
        return _failure(
            EMLDagCostStatus.INVALID_INPUT,
            EMLDagCostFailureStage.INPUT_VALIDATION,
            input_kind=EMLDagCostInputKind.SOURCE_AST,
            error=error,
        )

    representation_mode = f"pure_eml:{resolved_mode.value}"
    construction_path = "direct_hashcons"
    try:
        graph, root_node_id, construction = compile_ast_to_eml_dag(
            tree,
            mode=resolved_mode,
        )
    except UnsupportedASTOperatorError as error:
        return _failure(
            EMLDagCostStatus.UNSUPPORTED,
            EMLDagCostFailureStage.COMPILATION,
            input_kind=EMLDagCostInputKind.SOURCE_AST,
            compiler_mode=resolved_mode,
            representation_mode=representation_mode,
            construction_path=construction_path,
            error=error,
        )
    except (TypeError, ValueError) as error:
        return _failure(
            EMLDagCostStatus.INVALID_INPUT,
            EMLDagCostFailureStage.COMPILATION,
            input_kind=EMLDagCostInputKind.SOURCE_AST,
            compiler_mode=resolved_mode,
            representation_mode=representation_mode,
            construction_path=construction_path,
            error=error,
        )
    except Exception as error:
        return _failure(
            EMLDagCostStatus.FAILURE,
            EMLDagCostFailureStage.COMPILATION,
            input_kind=EMLDagCostInputKind.SOURCE_AST,
            compiler_mode=resolved_mode,
            representation_mode=representation_mode,
            construction_path=construction_path,
            error=error,
        )

    if (
        construction.compiler_mode is not resolved_mode
        or construction.representation_mode != representation_mode
    ):
        return _failure(
            EMLDagCostStatus.FAILURE,
            EMLDagCostFailureStage.DAG_VALIDATION,
            input_kind=EMLDagCostInputKind.SOURCE_AST,
            compiler_mode=resolved_mode,
            representation_mode=representation_mode,
            construction_path=construction.construction_path,
            error_type="EMLDagConstructionProvenanceError",
            error_message="direct construction returned inconsistent compiler provenance",
        )
    return _cost_from_graph(
        graph,
        root_node_id,
        input_kind=EMLDagCostInputKind.SOURCE_AST,
        compiler_mode=resolved_mode,
        expected_representation_mode=representation_mode,
        construction_path=construction.construction_path,
        expected_node_count=construction.final_node_count,
    )


def _cost_provided_eml(tree: EMLTerm) -> EMLDagCostResult:
    try:
        graph = eml_to_dag(
            tree,
            root_id="eml-dag-cost-input",
            representation_mode=_PROVIDED_TREE_REPRESENTATION_MODE,
        )
    except PureEMLValidationError as error:
        return _failure(
            EMLDagCostStatus.INVALID_INPUT,
            EMLDagCostFailureStage.INPUT_VALIDATION,
            input_kind=EMLDagCostInputKind.PURE_EML_TREE,
            representation_mode=_PROVIDED_TREE_REPRESENTATION_MODE,
            construction_path=_POSTHOC_CONSTRUCTION_PATH,
            error=error,
        )
    except Exception as error:
        return _failure(
            EMLDagCostStatus.FAILURE,
            EMLDagCostFailureStage.COMPILATION,
            input_kind=EMLDagCostInputKind.PURE_EML_TREE,
            representation_mode=_PROVIDED_TREE_REPRESENTATION_MODE,
            construction_path=_POSTHOC_CONSTRUCTION_PATH,
            error=error,
        )

    try:
        root_node_id = graph.roots[0].target_id
    except (AttributeError, IndexError, TypeError) as error:
        return _failure(
            EMLDagCostStatus.FAILURE,
            EMLDagCostFailureStage.DAG_VALIDATION,
            input_kind=EMLDagCostInputKind.PURE_EML_TREE,
            representation_mode=_PROVIDED_TREE_REPRESENTATION_MODE,
            construction_path=_POSTHOC_CONSTRUCTION_PATH,
            error=error,
        )
    return _cost_from_graph(
        graph,
        root_node_id,
        input_kind=EMLDagCostInputKind.PURE_EML_TREE,
        compiler_mode=None,
        expected_representation_mode=_PROVIDED_TREE_REPRESENTATION_MODE,
        construction_path=_POSTHOC_CONSTRUCTION_PATH,
    )


def compute_eml_dag_cost(
    expression: ASTTree | EMLTerm,
    *,
    compiler_mode: CompilerMode | None = None,
) -> EMLDagCostResult:
    """Return the exact pure-EML DAG cost or a retained terminal failure.

    ``compiler_mode=None`` selects :attr:`CompilerMode.OFFICIAL_V4` for
    source ASTs.  ``CLEAN_NEGATION`` is available only through an explicit
    argument and remains separately labeled.  A provided EML tree rejects a
    compiler-mode argument because its construction provenance is not encoded
    in the recursive value.

    The result never contains a partial or estimated cost.  Callers own outer
    candidate limits and timeouts; this boundary owns only exact structural
    compilation, validation, counting, and signing.
    """

    if isinstance(expression, ASTTree):
        return _cost_source_ast(expression, compiler_mode)
    if is_eml_term(expression):
        if compiler_mode is not None:
            return _failure(
                EMLDagCostStatus.INVALID_INPUT,
                EMLDagCostFailureStage.INPUT_VALIDATION,
                input_kind=EMLDagCostInputKind.PURE_EML_TREE,
                error=ValueError("a provided EML tree cannot accept or claim a compiler mode"),
            )
        return _cost_provided_eml(expression)
    return _failure(
        EMLDagCostStatus.INVALID_INPUT,
        EMLDagCostFailureStage.INPUT_VALIDATION,
        input_kind=None,
        error=TypeError("expression must be a validated ASTTree or a pure EML term"),
    )
