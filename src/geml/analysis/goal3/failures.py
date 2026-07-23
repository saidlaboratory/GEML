"""Deterministic Goal 3 outcome mining with separate structural claims."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

from geml.analysis.goal3.metrics import AnalysisRow


def _exact(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True, slots=True)
class RankedResult:
    """A valid expression shown on both compression and remaining-alpha axes."""

    expression_id: str
    family: str
    split: str
    domain: str
    operator_signature: str
    eml_compression: Fraction
    remaining_alpha: Fraction

    def as_dict(self) -> dict[str, object]:
        return {
            "expression_id": self.expression_id,
            "family": self.family,
            "split": self.split,
            "domain": self.domain,
            "operator_signature": self.operator_signature,
            "eml_compression": {
                "exact": _exact(self.eml_compression),
                "value": float(self.eml_compression),
            },
            "remaining_alpha": {
                "metric": "dag_alpha_vs_ast_tree",
                "exact": _exact(self.remaining_alpha),
                "value": float(self.remaining_alpha),
            },
        }


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """One retained processing failure with complete saved diagnostics."""

    expression_id: str
    family: str
    split: str
    domain: str
    operator_signature: str | None
    actual_ast_size: int | None
    actual_ast_depth: int | None
    input_shard_id: str
    input_shard_path: str
    input_row_index: int
    error_stage: str
    error_type: str
    error_message: str

    @classmethod
    def from_row(cls, row: AnalysisRow) -> FailureRecord:
        if row.valid:
            raise ValueError("failure records require unsuccessful rows")
        if row.error_stage is None or row.error_type is None or row.error_message is None:
            raise ValueError("failure row lacks retained diagnostics")
        return cls(
            expression_id=row.expression_id,
            family=row.family,
            split=row.split,
            domain=row.domain,
            operator_signature=row.operator_signature,
            actual_ast_size=row.actual_ast_size,
            actual_ast_depth=row.actual_ast_depth,
            input_shard_id=row.input_shard_id,
            input_shard_path=row.input_shard_path,
            input_row_index=row.input_row_index,
            error_stage=row.error_stage,
            error_type=row.error_type,
            error_message=row.error_message,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "expression_id": self.expression_id,
            "family": self.family,
            "split": self.split,
            "domain": self.domain,
            "operator_signature": self.operator_signature,
            "actual_ast_size": self.actual_ast_size,
            "actual_ast_depth": self.actual_ast_depth,
            "input_shard_id": self.input_shard_id,
            "input_shard_path": self.input_shard_path,
            "input_row_index": self.input_row_index,
            "error_stage": self.error_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class OutcomeReport:
    """Separate rankings plus every retained construction failure."""

    ranking_limit: int
    valid_count: int
    failure_count: int
    structurally_competitive_count: int
    failure_stage_counts: tuple[tuple[str, int], ...]
    failure_type_counts: tuple[tuple[str, int], ...]
    highest_compression: tuple[RankedResult, ...]
    lowest_compression: tuple[RankedResult, ...]
    lowest_remaining_alpha: tuple[RankedResult, ...]
    highest_remaining_alpha: tuple[RankedResult, ...]
    processing_failures: tuple[FailureRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ranking_limit": self.ranking_limit,
            "valid_count": self.valid_count,
            "failure_count": self.failure_count,
            "structurally_competitive_count": self.structurally_competitive_count,
            "failure_stage_counts": dict(self.failure_stage_counts),
            "failure_type_counts": dict(self.failure_type_counts),
            "claim_separation": {
                "highest_compression": ("largest eml_tree_node_count / eml_dag_node_count"),
                "lowest_compression": ("smallest eml_tree_node_count / eml_dag_node_count"),
                "lowest_remaining_alpha": ("smallest eml_dag_node_count / ast_tree_node_count"),
                "interpretation": (
                    "compressing a representation well and ending structurally small "
                    "are reported as independent observations"
                ),
            },
            "highest_compression": [result.as_dict() for result in self.highest_compression],
            "lowest_compression": [result.as_dict() for result in self.lowest_compression],
            "lowest_remaining_alpha": [result.as_dict() for result in self.lowest_remaining_alpha],
            "highest_remaining_alpha": [
                result.as_dict() for result in self.highest_remaining_alpha
            ],
            "processing_failures": [failure.as_dict() for failure in self.processing_failures],
        }


def _ranking_row(row: AnalysisRow) -> RankedResult:
    if row.metrics is None or row.operator_signature is None:
        raise ValueError("rankings require a successful row")
    return RankedResult(
        expression_id=row.expression_id,
        family=row.family,
        split=row.split,
        domain=row.domain,
        operator_signature=row.operator_signature,
        eml_compression=row.metrics.ratio("eml_compression"),
        remaining_alpha=row.metrics.ratio("dag_alpha_vs_ast_tree"),
    )


def _trim(
    values: list[RankedResult],
    *,
    key: str,
    reverse: bool,
    limit: int,
) -> None:
    if key == "compression":
        values.sort(
            key=lambda value: (
                -value.eml_compression if reverse else value.eml_compression,
                value.expression_id,
            )
        )
    elif key == "alpha":
        values.sort(
            key=lambda value: (
                -value.remaining_alpha if reverse else value.remaining_alpha,
                value.expression_id,
            )
        )
    else:  # pragma: no cover - private invariant
        raise ValueError(f"unknown ranking key {key!r}")
    del values[limit:]


class OutcomeMiner:
    """Streaming bounded rankings plus unabridged failure retention."""

    def __init__(self, *, limit: int = 20) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("ranking limit must be a positive integer")
        self._limit = limit
        self._valid_count = 0
        self._structurally_competitive_count = 0
        self._highest_compression: list[RankedResult] = []
        self._lowest_compression: list[RankedResult] = []
        self._lowest_alpha: list[RankedResult] = []
        self._highest_alpha: list[RankedResult] = []
        self._failures: list[FailureRecord] = []

    def add(self, row: AnalysisRow) -> None:
        if not row.valid:
            self._failures.append(FailureRecord.from_row(row))
            return
        self._valid_count += 1
        ranked = _ranking_row(row)
        if ranked.remaining_alpha <= 1:
            self._structurally_competitive_count += 1
        self._highest_compression.append(ranked)
        _trim(
            self._highest_compression,
            key="compression",
            reverse=True,
            limit=self._limit,
        )
        self._lowest_compression.append(ranked)
        _trim(
            self._lowest_compression,
            key="compression",
            reverse=False,
            limit=self._limit,
        )
        self._lowest_alpha.append(ranked)
        _trim(
            self._lowest_alpha,
            key="alpha",
            reverse=False,
            limit=self._limit,
        )
        self._highest_alpha.append(ranked)
        _trim(
            self._highest_alpha,
            key="alpha",
            reverse=True,
            limit=self._limit,
        )

    def finish(self) -> OutcomeReport:
        stage_counts = Counter(failure.error_stage for failure in self._failures)
        type_counts = Counter(failure.error_type for failure in self._failures)
        return OutcomeReport(
            ranking_limit=self._limit,
            valid_count=self._valid_count,
            failure_count=len(self._failures),
            structurally_competitive_count=self._structurally_competitive_count,
            failure_stage_counts=tuple(sorted(stage_counts.items())),
            failure_type_counts=tuple(sorted(type_counts.items())),
            highest_compression=tuple(self._highest_compression),
            lowest_compression=tuple(self._lowest_compression),
            lowest_remaining_alpha=tuple(self._lowest_alpha),
            highest_remaining_alpha=tuple(self._highest_alpha),
            processing_failures=tuple(self._failures),
        )
