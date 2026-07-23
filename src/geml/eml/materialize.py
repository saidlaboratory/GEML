"""Bounded materialization guarded by exact count-only preflight."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from geml.eml.compiler_core import CompilerMode, require_compiler_mode
from geml.eml.counting import CountedEML, validate_materialized_eml
from geml.eml.ir import EMLTerm
from geml.eml.validate import PureEMLStatistics, PureEMLValidationError


class MaterializationStatus(StrEnum):
    """Terminal status for one count-first materialization attempt."""

    MATERIALIZED = "materialized"
    NODE_LIMIT_EXCEEDED = "node_limit_exceeded"
    DEPTH_LIMIT_EXCEEDED = "depth_limit_exceeded"
    RECURSION_OR_STEP_LIMIT_EXCEEDED = "recursion_or_step_limit_exceeded"
    COUNT_FAILED = "count_failed"
    BUILDER_FAILED = "builder_failed"
    VALIDATION_FAILED = "validation_failed"
    COUNT_MISMATCH = "count_mismatch"
    UNSUPPORTED = "unsupported"


def _require_optional_limit(value: int | None, *, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class MaterializationLimits:
    """Optional positive preflight limits for an audit materialization."""

    maximum_nodes: int | None = None
    maximum_depth: int | None = None
    maximum_construction_steps: int | None = None

    def __post_init__(self) -> None:
        _require_optional_limit(self.maximum_nodes, name="maximum_nodes")
        _require_optional_limit(self.maximum_depth, name="maximum_depth")
        _require_optional_limit(
            self.maximum_construction_steps,
            name="maximum_construction_steps",
        )


@dataclass(frozen=True, slots=True)
class MaterializationRequest:
    """One immutable count-first materialization request."""

    label: str
    compiler_mode: CompilerMode
    counter: Callable[[], CountedEML]
    builder: Callable[[], EMLTerm] | None
    limits: MaterializationLimits = MaterializationLimits()

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("materialization label must be nonblank")
        require_compiler_mode(self.compiler_mode)
        if not callable(self.counter):
            raise TypeError("materialization counter must be callable")
        if self.builder is not None and not callable(self.builder):
            raise TypeError("materialization builder must be callable or None")
        if not isinstance(self.limits, MaterializationLimits):
            raise TypeError("limits must be MaterializationLimits")


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Complete terminal result; failed attempts never expose a partial tree."""

    label: str
    compiler_mode: CompilerMode
    status: MaterializationStatus
    exact_count: CountedEML | None
    tree: EMLTerm | None
    validated_statistics: PureEMLStatistics | None
    error_type: str | None
    error_message: str | None


def _failure(
    request: MaterializationRequest,
    status: MaterializationStatus,
    *,
    exact_count: CountedEML | None,
    error: Exception | None = None,
    message: str | None = None,
    statistics: PureEMLStatistics | None = None,
) -> MaterializationResult:
    return MaterializationResult(
        label=request.label,
        compiler_mode=request.compiler_mode,
        status=status,
        exact_count=exact_count,
        tree=None,
        validated_statistics=statistics,
        error_type=type(error).__name__ if error is not None else None,
        error_message=str(error) if error is not None else message,
    )


def materialize_bounded(request: MaterializationRequest) -> MaterializationResult:
    """Count first, enforce bounds, then build and require exact statistic equality."""

    if not isinstance(request, MaterializationRequest):
        raise TypeError("request must be a MaterializationRequest")
    try:
        exact_count = request.counter()
        if not isinstance(exact_count, CountedEML):
            raise TypeError("counter must return CountedEML")
        if exact_count.compiler_mode is not request.compiler_mode:
            raise ValueError("counter result mode does not match materialization request")
    except RecursionError as error:
        return _failure(
            request,
            MaterializationStatus.RECURSION_OR_STEP_LIMIT_EXCEEDED,
            exact_count=None,
            error=error,
        )
    except Exception as error:
        return _failure(
            request,
            MaterializationStatus.COUNT_FAILED,
            exact_count=None,
            error=error,
        )

    limits = request.limits
    if limits.maximum_nodes is not None and exact_count.node_count > limits.maximum_nodes:
        return _failure(
            request,
            MaterializationStatus.NODE_LIMIT_EXCEEDED,
            exact_count=exact_count,
            message=(
                f"exact node count {exact_count.node_count} exceeds limit {limits.maximum_nodes}"
            ),
        )
    if limits.maximum_depth is not None and exact_count.depth > limits.maximum_depth:
        return _failure(
            request,
            MaterializationStatus.DEPTH_LIMIT_EXCEEDED,
            exact_count=exact_count,
            message=f"exact depth {exact_count.depth} exceeds limit {limits.maximum_depth}",
        )
    construction_steps = sum(exact_count.operation_counts_dict().values())
    if (
        limits.maximum_construction_steps is not None
        and construction_steps > limits.maximum_construction_steps
    ):
        return _failure(
            request,
            MaterializationStatus.RECURSION_OR_STEP_LIMIT_EXCEEDED,
            exact_count=exact_count,
            message=(
                f"construction trace count {construction_steps} exceeds limit "
                f"{limits.maximum_construction_steps}"
            ),
        )
    if request.builder is None:
        return _failure(
            request,
            MaterializationStatus.UNSUPPORTED,
            exact_count=exact_count,
            message="no materializing builder was supplied",
        )

    try:
        tree = request.builder()
    except RecursionError as error:
        return _failure(
            request,
            MaterializationStatus.RECURSION_OR_STEP_LIMIT_EXCEEDED,
            exact_count=exact_count,
            error=error,
        )
    except Exception as error:
        return _failure(
            request,
            MaterializationStatus.BUILDER_FAILED,
            exact_count=exact_count,
            error=error,
        )

    try:
        statistics = validate_materialized_eml(
            tree,
            maximum_nodes=limits.maximum_nodes,
            maximum_depth=limits.maximum_depth,
        )
        if statistics.reused_object_count:
            raise ValueError("materialized result contains shared node identities")
    except PureEMLValidationError as error:
        if limits.maximum_nodes is not None and "node limit" in str(error):
            return _failure(
                request,
                MaterializationStatus.NODE_LIMIT_EXCEEDED,
                exact_count=exact_count,
                error=error,
            )
        if limits.maximum_depth is not None and "depth limit" in str(error):
            return _failure(
                request,
                MaterializationStatus.DEPTH_LIMIT_EXCEEDED,
                exact_count=exact_count,
                error=error,
            )
        return _failure(
            request,
            MaterializationStatus.VALIDATION_FAILED,
            exact_count=exact_count,
            error=error,
        )
    except Exception as error:
        return _failure(
            request,
            MaterializationStatus.VALIDATION_FAILED,
            exact_count=exact_count,
            error=error,
        )

    expected = (
        exact_count.node_count,
        exact_count.edge_count,
        exact_count.leaf_count,
        exact_count.operator_count,
        exact_count.depth,
    )
    observed = (
        statistics.node_count,
        statistics.edge_count,
        statistics.leaf_count,
        statistics.operator_count,
        statistics.depth,
    )
    if observed != expected:
        return _failure(
            request,
            MaterializationStatus.COUNT_MISMATCH,
            exact_count=exact_count,
            message=f"counted statistics {expected} do not match materialized {observed}",
            statistics=statistics,
        )

    return MaterializationResult(
        label=request.label,
        compiler_mode=request.compiler_mode,
        status=MaterializationStatus.MATERIALIZED,
        exact_count=exact_count,
        tree=tree,
        validated_statistics=statistics,
        error_type=None,
        error_message=None,
    )


def materialize_full(
    *,
    label: str,
    compiler_mode: CompilerMode,
    counter: Callable[[], CountedEML],
    builder: Callable[[], EMLTerm],
) -> MaterializationResult:
    """Materialize an unbounded audit fixture while still enforcing count equality."""

    return materialize_bounded(
        MaterializationRequest(
            label=label,
            compiler_mode=compiler_mode,
            counter=counter,
            builder=builder,
        )
    )
