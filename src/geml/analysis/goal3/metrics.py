"""Reproducible Goal 3 analysis over validated saved DAG-run artifacts.

The analysis is deliberately structural.  It reports exact-sharing ratios,
reuse accounting, runtime, and resource measurements without making claims
about transformations that the Goal 3 experiment did not perform.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
import weakref
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Final

from geml.experiments.goal3.run import (
    iter_metric_tables,
    iter_shard_telemetry,
)
from geml.experiments.goal3.runtime import (
    Goal3ArtifactError,
    atomic_write_json,
    canonical_json,
    load_json_mapping,
    publish_temporary_file,
    sha256_file,
)

if TYPE_CHECKING:
    from geml.analysis.goal3.failures import OutcomeReport
    from geml.plots.goal3 import StabilityPoint

ANALYSIS_SCHEMA_VERSION: Final = "geml-goal3-analysis-v3"
ANALYSIS_ARTIFACT_SCHEMA_VERSION: Final = "geml-goal3-analysis-artifacts-v3"
OPERATOR_SIGNATURE_SIDECAR_SCHEMA_VERSION: Final = "geml-goal3-operator-signature-strata-v1"
METRICS_SUMMARY_NAME: Final = "metrics.summary.json"
OPERATOR_SIGNATURE_SIDECAR_NAME: Final = "operator-signature.strata.jsonl.gz"
OPERATOR_SIGNATURE_COMPRESSION_LEVEL: Final = 9
AGGREGATE_MEAN_METHOD: Final = "ordered_compensated_decimal_mean"
AGGREGATE_WORKING_PRECISION_DIGITS: Final = 80
AGGREGATE_REPORTED_PRECISION_DIGITS: Final = 50
AGGREGATE_ROUNDING: Final = "ROUND_HALF_EVEN"
AGGREGATE_CONTEXT_EMIN: Final = -999_999_999
AGGREGATE_CONTEXT_EMAX: Final = 999_999_999
AGGREGATE_CONTEXT_CAPITALS: Final = 1
AGGREGATE_CONTEXT_CLAMP: Final = 0
AGGREGATE_CONTEXT_TRAPS: Final = (
    "InvalidOperation",
    "DivisionByZero",
    "Overflow",
)
UNAVAILABLE_STRATUM: Final = "<unavailable>"
RATIO_NAMES: Final = (
    "raw_tree_alpha",
    "dag_alpha_vs_ast_tree",
    "dag_alpha_vs_ast_dag",
    "ast_compression",
    "eml_compression",
)


class AnalysisArtifactError(RuntimeError):
    """A validated Goal 3 run cannot be analyzed or safely exported."""


class StratificationAxis(StrEnum):
    """Independent grouping dimensions required by the Goal 3 analysis."""

    FAMILY = "family"
    OPERATOR_SIGNATURE = "operator_signature"
    ACTUAL_AST_SIZE = "actual_ast_size"
    ACTUAL_AST_DEPTH = "actual_ast_depth"
    SPLIT = "split"
    DOMAIN = "domain"
    REUSE_PATTERN = "reuse_pattern"


class ReusePattern(StrEnum):
    """Which exact-sharing paths contain at least one reused node."""

    NONE = "none"
    AST_ONLY = "ast_only"
    EML_ONLY = "eml_only"
    BOTH = "both"


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnalysisArtifactError(f"metric row has invalid {name}")
    return value


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name=name)


def _required_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisArtifactError(f"metric row has invalid {name}")
    return value


def _optional_integer(value: object, *, name: str, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _required_integer(value, name=name, minimum=minimum)


def _fraction_from_columns(row: Mapping[str, object], prefix: str) -> Fraction:
    try:
        value = Fraction(
            int(_required_text(row.get(f"{prefix}_numerator"), name=f"{prefix} numerator")),
            int(_required_text(row.get(f"{prefix}_denominator"), name=f"{prefix} denominator")),
        )
    except (ValueError, ZeroDivisionError) as error:
        raise AnalysisArtifactError(f"metric row has invalid exact {prefix}") from error
    expected = f"{value.numerator}/{value.denominator}"
    if row.get(f"{prefix}_exact") != expected:
        raise AnalysisArtifactError(f"metric row has noncanonical exact {prefix}")
    approximate = row.get(f"{prefix}_value")
    if not isinstance(approximate, float) or approximate != float(value):
        raise AnalysisArtifactError(f"metric row has inconsistent float {prefix}")
    return value


@dataclass(frozen=True, slots=True)
class ExactMean:
    """An exact arithmetic mean with its explicit valid-only denominator."""

    numerator: int
    denominator: int
    sample_count: int

    def __post_init__(self) -> None:
        if self.denominator <= 0 or self.sample_count <= 0:
            raise ValueError("exact means require positive denominators and sample counts")
        canonical = Fraction(self.numerator, self.denominator)
        if (canonical.numerator, canonical.denominator) != (
            self.numerator,
            self.denominator,
        ):
            raise ValueError("exact means must be stored in canonical form")

    @classmethod
    def from_total(cls, total: Fraction | int, sample_count: int) -> ExactMean:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        mean = Fraction(total, sample_count)
        return cls(
            numerator=mean.numerator,
            denominator=mean.denominator,
            sample_count=sample_count,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExactMean:
        expected_keys = {
            "numerator",
            "denominator",
            "exact",
            "value",
            "sample_count",
        }
        if set(value) != expected_keys:
            raise ValueError("exact-mean fields are incomplete")
        numerator_text = value["numerator"]
        denominator_text = value["denominator"]
        sample_count = value["sample_count"]
        if (
            not isinstance(numerator_text, str)
            or not isinstance(denominator_text, str)
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
        ):
            raise ValueError("exact-mean fields have invalid types")
        try:
            numerator = int(numerator_text)
            denominator = int(denominator_text)
        except ValueError as error:
            raise ValueError("exact-mean integers are invalid") from error
        if str(numerator) != numerator_text or str(denominator) != denominator_text:
            raise ValueError("exact-mean integers are not canonical")
        result = cls(
            numerator=numerator,
            denominator=denominator,
            sample_count=sample_count,
        )
        approximate = value["value"]
        if (
            value["exact"] != result.exact
            or not isinstance(approximate, float)
            or approximate != result.value
        ):
            raise ValueError("exact-mean redundant representations disagree")
        return result

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @property
    def exact(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @property
    def value(self) -> float:
        return float(self.fraction)

    def as_dict(self) -> dict[str, object]:
        return {
            "numerator": str(self.numerator),
            "denominator": str(self.denominator),
            "exact": self.exact,
            "value": self.value,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class DecimalMean:
    """A bounded-size, explicitly approximate mean of exact input ratios."""

    decimal: str
    sample_count: int
    method: str = AGGREGATE_MEAN_METHOD
    working_precision_digits: int = AGGREGATE_WORKING_PRECISION_DIGITS
    reported_precision_digits: int = AGGREGATE_REPORTED_PRECISION_DIGITS
    rounding: str = AGGREGATE_ROUNDING

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("decimal means require a positive sample count")
        if not isinstance(self.decimal, str) or not self.decimal:
            raise ValueError("decimal mean text must be nonblank")
        try:
            value = Decimal(self.decimal)
        except (InvalidOperation, ValueError) as error:
            raise ValueError("decimal mean text must be a valid Decimal") from error
        if not value.is_finite():
            raise ValueError("decimal means must be finite")
        if self.method != AGGREGATE_MEAN_METHOD:
            raise ValueError("decimal mean method is not the pinned analysis method")
        if (
            self.working_precision_digits != AGGREGATE_WORKING_PRECISION_DIGITS
            or self.reported_precision_digits != AGGREGATE_REPORTED_PRECISION_DIGITS
            or self.rounding != AGGREGATE_ROUNDING
        ):
            raise ValueError("decimal mean precision policy is not pinned")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> DecimalMean:
        expected_keys = {
            "decimal",
            "value",
            "approximate",
            "sample_count",
            "method",
            "working_precision_digits",
            "reported_precision_digits",
            "rounding",
        }
        if set(value) != expected_keys:
            raise ValueError("decimal-mean fields are incomplete")
        decimal_text = value["decimal"]
        sample_count = value["sample_count"]
        method = value["method"]
        working_precision = value["working_precision_digits"]
        reported_precision = value["reported_precision_digits"]
        rounding = value["rounding"]
        if (
            not isinstance(decimal_text, str)
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or not isinstance(method, str)
            or isinstance(working_precision, bool)
            or not isinstance(working_precision, int)
            or isinstance(reported_precision, bool)
            or not isinstance(reported_precision, int)
            or not isinstance(rounding, str)
        ):
            raise ValueError("decimal-mean fields have invalid types")
        result = cls(
            decimal=decimal_text,
            sample_count=sample_count,
            method=method,
            working_precision_digits=working_precision,
            reported_precision_digits=reported_precision,
            rounding=rounding,
        )
        approximate = value["value"]
        if (
            value["approximate"] is not True
            or not isinstance(approximate, float)
            or approximate != result.value
        ):
            raise ValueError("decimal-mean redundant representations disagree")
        return result

    @property
    def decimal_value(self) -> Decimal:
        return Decimal(self.decimal)

    @property
    def value(self) -> float:
        return float(self.decimal_value)

    def as_dict(self) -> dict[str, object]:
        return {
            "decimal": self.decimal,
            "value": self.value,
            "approximate": True,
            "sample_count": self.sample_count,
            "method": self.method,
            "working_precision_digits": self.working_precision_digits,
            "reported_precision_digits": self.reported_precision_digits,
            "rounding": self.rounding,
        }


def aggregate_decimal_context(*, precision: int) -> Context:
    """Return a fresh Decimal context independent of ambient process settings."""

    if precision <= 0:
        raise ValueError("decimal precision must be positive")
    return Context(
        prec=precision,
        rounding=ROUND_HALF_EVEN,
        Emin=AGGREGATE_CONTEXT_EMIN,
        Emax=AGGREGATE_CONTEXT_EMAX,
        capitals=AGGREGATE_CONTEXT_CAPITALS,
        clamp=AGGREGATE_CONTEXT_CLAMP,
        flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


def aggregate_mean_policy() -> dict[str, object]:
    """Describe the complete deterministic aggregate Decimal policy."""

    return {
        "method": AGGREGATE_MEAN_METHOD,
        "working_precision_digits": AGGREGATE_WORKING_PRECISION_DIGITS,
        "reported_precision_digits": AGGREGATE_REPORTED_PRECISION_DIGITS,
        "rounding": AGGREGATE_ROUNDING,
        "context": {
            "Emin": AGGREGATE_CONTEXT_EMIN,
            "Emax": AGGREGATE_CONTEXT_EMAX,
            "capitals": AGGREGATE_CONTEXT_CAPITALS,
            "clamp": AGGREGATE_CONTEXT_CLAMP,
            "traps": list(AGGREGATE_CONTEXT_TRAPS),
        },
        "order": "authoritative corpus row order",
        "input": "exact per-row rational numerator and denominator",
    }


@dataclass(slots=True)
class _DecimalMeanAccumulator:
    """Fixed-memory compensated summation of exact rational observations."""

    total: Decimal = field(default_factory=lambda: Decimal(0))
    compensation: Decimal = field(default_factory=lambda: Decimal(0))
    sample_count: int = 0

    def add(self, value: Fraction) -> None:
        context = aggregate_decimal_context(precision=AGGREGATE_WORKING_PRECISION_DIGITS)
        with localcontext(context):
            decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
            corrected = decimal_value - self.compensation
            updated = self.total + corrected
            self.compensation = (updated - self.total) - corrected
            self.total = updated
        self.sample_count += 1

    def mean(self) -> DecimalMean:
        if self.sample_count <= 0:
            raise ValueError("cannot compute an empty decimal mean")
        working_context = aggregate_decimal_context(precision=AGGREGATE_WORKING_PRECISION_DIGITS)
        with localcontext(working_context):
            mean = self.total / Decimal(self.sample_count)
        reported_context = aggregate_decimal_context(precision=AGGREGATE_REPORTED_PRECISION_DIGITS)
        with localcontext(reported_context):
            reported = +mean
            decimal_text = str(reported)
        return DecimalMean(
            decimal=decimal_text,
            sample_count=self.sample_count,
        )


@dataclass(frozen=True, slots=True)
class RowReuse:
    """Validated per-expression reuse measurements for one representation."""

    reused_node_count: int
    reused_reference_count: int
    excess_reference_count: int
    child_reference_overhead: int
    max_reuse_indegree: int
    reuse_depth_sum: int
    reuse_depth_count: int
    sharing_concentration: Fraction

    @classmethod
    def from_mapping(cls, row: Mapping[str, object], prefix: str) -> RowReuse:
        reused_nodes = _required_integer(
            row.get(f"{prefix}_reused_node_count"),
            name=f"{prefix} reused-node count",
        )
        reused_references = _required_integer(
            row.get(f"{prefix}_reused_reference_count"),
            name=f"{prefix} reused-reference count",
        )
        child_overhead = _required_integer(
            row.get(f"{prefix}_child_reference_overhead"),
            name=f"{prefix} child-reference overhead",
        )
        max_reuse = _required_integer(
            row.get(f"{prefix}_max_reuse_count"),
            name=f"{prefix} max-reuse indegree",
        )
        depth_sum = _required_integer(
            row.get(f"{prefix}_reuse_depth_sum"),
            name=f"{prefix} reuse-depth sum",
        )
        depth_count = _required_integer(
            row.get(f"{prefix}_reuse_depth_count"),
            name=f"{prefix} reuse-depth count",
        )
        excess = reused_references - reused_nodes
        if excess < 0 or excess != child_overhead:
            raise AnalysisArtifactError(
                f"{prefix} reused-reference and child-overhead accounting disagree"
            )
        if depth_count != reused_nodes:
            raise AnalysisArtifactError(f"{prefix} reuse-depth denominator is inconsistent")
        if reused_nodes == 0:
            if any((reused_references, child_overhead, max_reuse, depth_sum, depth_count)):
                raise AnalysisArtifactError(f"{prefix} no-reuse row has nonzero reuse values")
        elif max_reuse < 2:
            raise AnalysisArtifactError(f"{prefix} reused nodes require indegree at least two")
        return cls(
            reused_node_count=reused_nodes,
            reused_reference_count=reused_references,
            excess_reference_count=excess,
            child_reference_overhead=child_overhead,
            max_reuse_indegree=max_reuse,
            reuse_depth_sum=depth_sum,
            reuse_depth_count=depth_count,
            sharing_concentration=_fraction_from_columns(
                row,
                f"{prefix}_sharing_concentration",
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidMetrics:
    """All exact ratios and reuse measurements present on a successful row."""

    ratios: tuple[tuple[str, Fraction], ...]
    ast_reuse: RowReuse
    eml_reuse: RowReuse

    def ratio(self, name: str) -> Fraction:
        for candidate, value in self.ratios:
            if candidate == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class AnalysisRow:
    """Typed analysis view of one validated Goal 3 metric row."""

    expression_id: str
    status: str
    family: str
    split: str
    domain: str
    operator_signature: str | None
    actual_ast_size: int | None
    actual_ast_depth: int | None
    input_shard_id: str
    input_shard_path: str
    input_row_index: int
    metrics: ValidMetrics | None
    error_stage: str | None
    error_type: str | None
    error_message: str | None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> AnalysisRow:
        status = _required_text(row.get("status"), name="status")
        if status not in {"success", "failure"}:
            raise AnalysisArtifactError(f"unknown metric-row status {status!r}")
        metrics: ValidMetrics | None = None
        if status == "success":
            metrics = ValidMetrics(
                ratios=tuple((name, _fraction_from_columns(row, name)) for name in RATIO_NAMES),
                ast_reuse=RowReuse.from_mapping(row, "ast_dag"),
                eml_reuse=RowReuse.from_mapping(row, "eml_dag"),
            )
        result = cls(
            expression_id=_required_text(row.get("expression_id"), name="expression ID"),
            status=status,
            family=_required_text(row.get("operator_family"), name="family"),
            split=_required_text(row.get("split"), name="split"),
            domain=_required_text(row.get("domain_mode"), name="domain"),
            operator_signature=_optional_text(
                row.get("operator_signature"),
                name="operator signature",
            ),
            actual_ast_size=_optional_integer(
                row.get("ast_tree_node_count"),
                name="actual AST size",
                minimum=1,
            ),
            actual_ast_depth=_optional_integer(
                row.get("ast_tree_depth"),
                name="actual AST depth",
            ),
            input_shard_id=_required_text(
                row.get("input_shard_id"),
                name="input shard ID",
            ),
            input_shard_path=_required_text(
                row.get("input_shard_path"),
                name="input shard path",
            ),
            input_row_index=_required_integer(
                row.get("input_row_index"),
                name="input row index",
            ),
            metrics=metrics,
            error_stage=_optional_text(row.get("error_stage"), name="error stage"),
            error_type=_optional_text(row.get("error_type"), name="error type"),
            error_message=_optional_text(row.get("error_message"), name="error message"),
        )
        if status == "success":
            if result.operator_signature is None:
                raise AnalysisArtifactError("successful row has no operator signature")
            if result.actual_ast_size is None or result.actual_ast_depth is None:
                raise AnalysisArtifactError("successful row has no actual AST size/depth")
            if any(
                value is not None
                for value in (
                    result.error_stage,
                    result.error_type,
                    result.error_message,
                )
            ):
                raise AnalysisArtifactError("successful row contains failure diagnostics")
        elif not all(
            value is not None
            for value in (
                result.error_stage,
                result.error_type,
                result.error_message,
            )
        ):
            raise AnalysisArtifactError("failure row lacks detailed diagnostics")
        return result

    @property
    def valid(self) -> bool:
        return self.status == "success"

    @property
    def reuse_pattern(self) -> ReusePattern | None:
        if self.metrics is None:
            return None
        ast = self.metrics.ast_reuse.reused_node_count > 0
        eml = self.metrics.eml_reuse.reused_node_count > 0
        if ast and eml:
            return ReusePattern.BOTH
        if ast:
            return ReusePattern.AST_ONLY
        if eml:
            return ReusePattern.EML_ONLY
        return ReusePattern.NONE

    def stratum(self, axis: StratificationAxis) -> str:
        values: dict[StratificationAxis, object | None] = {
            StratificationAxis.FAMILY: self.family,
            StratificationAxis.OPERATOR_SIGNATURE: self.operator_signature,
            StratificationAxis.ACTUAL_AST_SIZE: self.actual_ast_size,
            StratificationAxis.ACTUAL_AST_DEPTH: self.actual_ast_depth,
            StratificationAxis.SPLIT: self.split,
            StratificationAxis.DOMAIN: self.domain,
            StratificationAxis.REUSE_PATTERN: (
                self.reuse_pattern.value if self.reuse_pattern is not None else None
            ),
        }
        value = values[axis]
        return UNAVAILABLE_STRATUM if value is None else str(value)


@dataclass(frozen=True, slots=True)
class ReuseAggregate:
    """Valid-only aggregate of one representation's exact-sharing behavior."""

    total_reused_node_count: int
    total_reused_reference_count: int
    total_excess_reference_count: int
    total_child_reference_overhead: int
    max_reuse_indegree: int
    mean_reused_node_count: ExactMean
    mean_reused_reference_count: ExactMean
    mean_excess_reference_count: ExactMean
    mean_child_reference_overhead: ExactMean
    mean_reuse_depth: ExactMean | None
    mean_sharing_concentration: DecimalMean

    def __post_init__(self) -> None:
        totals = (
            self.total_reused_node_count,
            self.total_reused_reference_count,
            self.total_excess_reference_count,
            self.total_child_reference_overhead,
            self.max_reuse_indegree,
        )
        if any(isinstance(value, bool) or value < 0 for value in totals):
            raise ValueError("reuse aggregate counts must be nonnegative integers")
        if (
            self.total_reused_reference_count - self.total_reused_node_count
            != self.total_excess_reference_count
            or self.total_excess_reference_count != self.total_child_reference_overhead
        ):
            raise ValueError("reuse aggregate accounting is inconsistent")
        valid_count = self.mean_reused_node_count.sample_count
        exact_total_means = (
            (self.mean_reused_node_count, self.total_reused_node_count),
            (self.mean_reused_reference_count, self.total_reused_reference_count),
            (self.mean_excess_reference_count, self.total_excess_reference_count),
            (self.mean_child_reference_overhead, self.total_child_reference_overhead),
        )
        for mean, total in exact_total_means:
            if mean.sample_count != valid_count or mean.fraction != Fraction(total, valid_count):
                raise ValueError("reuse aggregate exact mean disagrees with its total")
        if self.mean_sharing_concentration.sample_count != valid_count:
            raise ValueError("reuse sharing-concentration denominator is inconsistent")
        if self.total_reused_node_count == 0:
            if (
                self.total_reused_reference_count != 0
                or self.total_excess_reference_count != 0
                or self.total_child_reference_overhead != 0
                or self.max_reuse_indegree != 0
                or self.mean_reuse_depth is not None
            ):
                raise ValueError("no-reuse aggregate has nonzero reuse state")
        elif (
            self.total_reused_reference_count < 2 * self.total_reused_node_count
            or self.total_excess_reference_count < self.total_reused_node_count
            or self.max_reuse_indegree < 2
            or self.max_reuse_indegree > self.total_reused_reference_count
            or self.mean_reuse_depth is None
            or self.mean_reuse_depth.sample_count != self.total_reused_node_count
        ):
            raise ValueError("reuse aggregate depth or maximum indegree is inconsistent")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        valid_count: int,
    ) -> ReuseAggregate:
        expected_keys = {
            "total_reused_node_count",
            "total_reused_reference_count",
            "total_excess_reference_count",
            "total_child_reference_overhead",
            "max_reuse_indegree",
            "mean_reused_node_count",
            "mean_reused_reference_count",
            "mean_excess_reference_count",
            "mean_child_reference_overhead",
            "mean_reuse_depth",
            "mean_sharing_concentration",
        }
        if set(value) != expected_keys:
            raise ValueError("reuse-aggregate fields are incomplete")

        def count(name: str) -> int:
            candidate = value[name]
            if isinstance(candidate, bool) or not isinstance(candidate, int):
                raise ValueError(f"reuse-aggregate {name} is invalid")
            return candidate

        def exact_mean(name: str) -> ExactMean:
            candidate = value[name]
            if not isinstance(candidate, dict):
                raise ValueError(f"reuse-aggregate {name} is invalid")
            return ExactMean.from_mapping(candidate)

        sharing = value["mean_sharing_concentration"]
        if not isinstance(sharing, dict):
            raise ValueError("reuse sharing-concentration mean is invalid")
        depth_payload = value["mean_reuse_depth"]
        if depth_payload is not None and not isinstance(depth_payload, dict):
            raise ValueError("reuse-depth mean is invalid")
        result = cls(
            total_reused_node_count=count("total_reused_node_count"),
            total_reused_reference_count=count("total_reused_reference_count"),
            total_excess_reference_count=count("total_excess_reference_count"),
            total_child_reference_overhead=count("total_child_reference_overhead"),
            max_reuse_indegree=count("max_reuse_indegree"),
            mean_reused_node_count=exact_mean("mean_reused_node_count"),
            mean_reused_reference_count=exact_mean("mean_reused_reference_count"),
            mean_excess_reference_count=exact_mean("mean_excess_reference_count"),
            mean_child_reference_overhead=exact_mean("mean_child_reference_overhead"),
            mean_reuse_depth=(
                ExactMean.from_mapping(depth_payload) if depth_payload is not None else None
            ),
            mean_sharing_concentration=DecimalMean.from_mapping(sharing),
        )
        if result.mean_reused_node_count.sample_count != valid_count:
            raise ValueError("reuse aggregate valid-only denominator is inconsistent")
        return result

    def as_dict(self) -> dict[str, object]:
        return {
            "total_reused_node_count": self.total_reused_node_count,
            "total_reused_reference_count": self.total_reused_reference_count,
            "total_excess_reference_count": self.total_excess_reference_count,
            "total_child_reference_overhead": self.total_child_reference_overhead,
            "max_reuse_indegree": self.max_reuse_indegree,
            "mean_reused_node_count": self.mean_reused_node_count.as_dict(),
            "mean_reused_reference_count": self.mean_reused_reference_count.as_dict(),
            "mean_excess_reference_count": self.mean_excess_reference_count.as_dict(),
            "mean_child_reference_overhead": self.mean_child_reference_overhead.as_dict(),
            "mean_reuse_depth": (
                self.mean_reuse_depth.as_dict() if self.mean_reuse_depth is not None else None
            ),
            "mean_sharing_concentration": self.mean_sharing_concentration.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class GroupStatistics:
    """One stratum with honest all-processed and valid-only denominators."""

    key: str
    all_processed_count: int
    valid_count: int
    failure_count: int
    ratio_means: tuple[tuple[str, DecimalMean], ...]
    ast_reuse: ReuseAggregate | None
    eml_reuse: ReuseAggregate | None

    def __post_init__(self) -> None:
        if (
            not self.key
            or any(
                isinstance(value, bool) or value < 0
                for value in (
                    self.all_processed_count,
                    self.valid_count,
                    self.failure_count,
                )
            )
            or self.all_processed_count != self.valid_count + self.failure_count
        ):
            raise ValueError("group denominators do not account for every processed row")
        if self.valid_count == 0:
            if self.ratio_means or self.ast_reuse is not None or self.eml_reuse is not None:
                raise ValueError("zero-valid group has valid-only aggregates")
            return
        if (
            tuple(name for name, _ in self.ratio_means) != RATIO_NAMES
            or any(mean.sample_count != self.valid_count for _, mean in self.ratio_means)
            or self.ast_reuse is None
            or self.eml_reuse is None
            or self.ast_reuse.mean_reused_node_count.sample_count != self.valid_count
            or self.eml_reuse.mean_reused_node_count.sample_count != self.valid_count
        ):
            raise ValueError("group valid-only aggregates are inconsistent")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GroupStatistics:
        expected_keys = {
            "key",
            "all_processed_count",
            "valid_count",
            "failure_count",
            "ratio_means",
            "ast_reuse",
            "eml_reuse",
        }
        if set(value) != expected_keys:
            raise ValueError("group-statistics fields are incomplete")
        key = value["key"]
        counts = (
            value["all_processed_count"],
            value["valid_count"],
            value["failure_count"],
        )
        if (
            not isinstance(key, str)
            or not key
            or any(isinstance(count, bool) or not isinstance(count, int) for count in counts)
        ):
            raise ValueError("group-statistics fields have invalid types")
        all_processed, valid, failure = counts
        ratio_payload = value["ratio_means"]
        if not isinstance(ratio_payload, dict):
            raise ValueError("group-statistics ratio means are invalid")
        if ratio_payload and set(ratio_payload) != set(RATIO_NAMES):
            raise ValueError("group-statistics ratio names are invalid")
        if any(not isinstance(mean, dict) for mean in ratio_payload.values()):
            raise ValueError("group-statistics ratio mean is invalid")
        ratio_means = tuple(
            (name, DecimalMean.from_mapping(ratio_payload[name]))
            for name in RATIO_NAMES
            if name in ratio_payload
        )
        ast_payload = value["ast_reuse"]
        eml_payload = value["eml_reuse"]
        if ast_payload is not None and not isinstance(ast_payload, dict):
            raise ValueError("group-statistics AST reuse aggregate is invalid")
        if eml_payload is not None and not isinstance(eml_payload, dict):
            raise ValueError("group-statistics EML reuse aggregate is invalid")
        return cls(
            key=key,
            all_processed_count=all_processed,
            valid_count=valid,
            failure_count=failure,
            ratio_means=ratio_means,
            ast_reuse=(
                ReuseAggregate.from_mapping(ast_payload, valid_count=valid)
                if ast_payload is not None
                else None
            ),
            eml_reuse=(
                ReuseAggregate.from_mapping(eml_payload, valid_count=valid)
                if eml_payload is not None
                else None
            ),
        )

    def ratio_mean(self, name: str) -> DecimalMean | None:
        for candidate, value in self.ratio_means:
            if candidate == name:
                return value
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "all_processed_count": self.all_processed_count,
            "valid_count": self.valid_count,
            "failure_count": self.failure_count,
            "ratio_means": {name: value.as_dict() for name, value in self.ratio_means},
            "ast_reuse": self.ast_reuse.as_dict() if self.ast_reuse else None,
            "eml_reuse": self.eml_reuse.as_dict() if self.eml_reuse else None,
        }


@dataclass(frozen=True, slots=True)
class StratifiedTable:
    """Deterministically ordered groups for one independent axis."""

    axis: StratificationAxis
    groups: tuple[GroupStatistics, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis.value,
            "groups": [group.as_dict() for group in self.groups],
        }


@dataclass(frozen=True, slots=True)
class ExternalStratumDescriptor:
    """Logical identity and deterministic storage policy for one streamed axis."""

    schema_version: str
    axis: StratificationAxis
    path: str
    group_count: int
    all_processed_count: int
    valid_count: int
    failure_count: int
    content_sha256: str
    uncompressed_byte_count: int
    encoding: str = "canonical-jsonl-utf8"
    compression: str = "gzip"
    compression_level: int = OPERATOR_SIGNATURE_COMPRESSION_LEVEL
    gzip_mtime: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExternalStratumDescriptor:
        def text(name: str) -> str:
            candidate = value.get(name)
            if not isinstance(candidate, str) or not candidate:
                raise AnalysisArtifactError(
                    f"external operator-signature descriptor has invalid {name}"
                )
            return candidate

        def integer(name: str) -> int:
            candidate = value.get(name)
            if isinstance(candidate, bool) or not isinstance(candidate, int):
                raise AnalysisArtifactError(
                    f"external operator-signature descriptor has invalid {name}"
                )
            return candidate

        try:
            axis = StratificationAxis(text("axis"))
            return cls(
                schema_version=text("schema_version"),
                axis=axis,
                path=text("path"),
                group_count=integer("group_count"),
                all_processed_count=integer("all_processed_count"),
                valid_count=integer("valid_count"),
                failure_count=integer("failure_count"),
                content_sha256=text("content_sha256"),
                uncompressed_byte_count=integer("uncompressed_byte_count"),
                encoding=text("encoding"),
                compression=text("compression"),
                compression_level=integer("compression_level"),
                gzip_mtime=integer("gzip_mtime"),
            )
        except (ValueError, KeyError) as error:
            raise AnalysisArtifactError(
                "external operator-signature descriptor is invalid"
            ) from error

    def __post_init__(self) -> None:
        if self.schema_version != OPERATOR_SIGNATURE_SIDECAR_SCHEMA_VERSION:
            raise ValueError("external-stratum schema version is not pinned")
        if self.axis is not StratificationAxis.OPERATOR_SIGNATURE:
            raise ValueError("only operator-signature strata use external storage")
        if self.path != OPERATOR_SIGNATURE_SIDECAR_NAME:
            raise ValueError("external-stratum path is not pinned")
        if (
            self.group_count < 0
            or self.all_processed_count < 0
            or self.valid_count < 0
            or self.failure_count < 0
            or self.uncompressed_byte_count < 0
        ):
            raise ValueError("external-stratum counts must be nonnegative")
        if self.all_processed_count != self.valid_count + self.failure_count:
            raise ValueError("external-stratum denominators are inconsistent")
        if (
            len(self.content_sha256) != 64
            or self.content_sha256 != self.content_sha256.lower()
            or any(character not in "0123456789abcdef" for character in self.content_sha256)
        ):
            raise ValueError("external-stratum content hash must be lowercase SHA-256")
        if (
            self.encoding != "canonical-jsonl-utf8"
            or self.compression != "gzip"
            or self.compression_level != OPERATOR_SIGNATURE_COMPRESSION_LEVEL
            or self.gzip_mtime != 0
        ):
            raise ValueError("external-stratum storage policy is not pinned")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "axis": self.axis.value,
            "path": self.path,
            "group_count": self.group_count,
            "all_processed_count": self.all_processed_count,
            "valid_count": self.valid_count,
            "failure_count": self.failure_count,
            "content_sha256": self.content_sha256,
            "uncompressed_byte_count": self.uncompressed_byte_count,
            "encoding": self.encoding,
            "compression": self.compression,
            "compression_level": self.compression_level,
            "gzip_mtime": self.gzip_mtime,
        }


def _signature_group_line(group: GroupStatistics) -> bytes:
    return (canonical_json(group.as_dict()) + "\n").encode("utf-8")


def _canonical_sha256(value: object) -> str:
    """Hash canonical JSON incrementally without constructing one large string."""

    encoder = json.JSONEncoder(
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256()
    for fragment in encoder.iterencode(value):
        digest.update(fragment.encode("utf-8"))
    return digest.hexdigest()


def _remove_managed_spool(path: Path) -> None:
    with suppress(OSError):
        path.unlink(missing_ok=True)


@dataclass(eq=False, slots=True, weakref_slot=True)
class _ManagedSidecarSpool:
    """A closed temporary sidecar retained for repeatable save operations."""

    path: Path
    sha256: str
    byte_count: int
    _finalizer: weakref.finalize = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._finalizer = weakref.finalize(self, _remove_managed_spool, self.path)

    def cleanup(self) -> None:
        self._finalizer()


@dataclass(frozen=True, slots=True)
class CheckpointMetrics:
    """High-precision cumulative metric state at an exact corpus prefix."""

    processed_count: int
    valid_count: int
    failure_count: int
    ratio_means: tuple[tuple[str, DecimalMean], ...]

    def ratio_mean(self, name: str) -> DecimalMean | None:
        for candidate, value in self.ratio_means:
            if candidate == name:
                return value
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "processed_count": self.processed_count,
            "valid_count": self.valid_count,
            "failure_count": self.failure_count,
            "ratio_means": {name: value.as_dict() for name, value in self.ratio_means},
        }


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Complete structural, failure, and scale analysis of one saved run."""

    schema_version: str
    source_manifest_path: str
    source_manifest_sha256: str
    source_science_fingerprint: str
    source_stage: str
    overall: GroupStatistics
    strata: tuple[StratifiedTable, ...]
    operator_signature_stratum: ExternalStratumDescriptor
    operator_signature_spool: _ManagedSidecarSpool = field(repr=False, compare=False)
    checkpoints: tuple[CheckpointMetrics, ...]
    stability_curve: tuple[StabilityPoint, ...]
    outcomes: OutcomeReport
    missing_standard_checkpoints: tuple[int, ...]
    fingerprint: str

    def stratum(self, axis: StratificationAxis) -> StratifiedTable:
        for table in self.strata:
            if table.axis is axis:
                return table
        raise KeyError(axis)

    def metrics_payload(self) -> dict[str, object]:
        summary_strata = [
            table.as_dict()
            for table in self.strata
            if table.axis is not StratificationAxis.OPERATOR_SIGNATURE
        ]
        return {
            "schema_version": self.schema_version,
            "source": {
                "manifest_path": self.source_manifest_path,
                "manifest_sha256": self.source_manifest_sha256,
                "science_fingerprint": self.source_science_fingerprint,
                "stage": self.source_stage,
            },
            "metric_definitions": {
                "ratio_means": (
                    "approximate arithmetic mean of exact per-expression ratios "
                    "using the pinned bounded-size Decimal policy"
                ),
                "aggregate_mean_policy": aggregate_mean_policy(),
                "reuse_depth": (
                    "reused-node-weighted mean of minimum root-to-node child-reference distance"
                ),
                "reused_reference_count": (
                    "sum of indegrees over nodes whose indegree is greater than one"
                ),
                "excess_reference_count": "sum of indegree minus one over reused nodes",
                "sharing_concentration": (
                    "per-expression max excess indegree divided by total excess indegree, "
                    "then averaged over valid expressions"
                ),
                "child_reference_overhead": (
                    "child-reference count beyond a rooted tree; equal to excess references"
                ),
            },
            "overall": self.overall.as_dict(),
            "strata": summary_strata,
            "external_strata": [self.operator_signature_stratum.as_dict()],
            "checkpoints": [checkpoint.as_dict() for checkpoint in self.checkpoints],
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self.metrics_payload(),
            "stability_curve": [point.as_dict() for point in self.stability_curve],
            "outcomes": self.outcomes.as_dict(),
            "missing_standard_checkpoints": list(self.missing_standard_checkpoints),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class SavedAnalysis:
    """Paths for deterministic analysis products written outside run artifacts."""

    output_directory: Path
    manifest_path: Path
    metrics_path: Path
    operator_signature_strata_path: Path
    outcomes_path: Path
    plot_data_path: Path
    analysis_fingerprint: str


def _manifest_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AnalysisArtifactError(f"analysis manifest has invalid {name}")
    return value


def _manifest_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalysisArtifactError(f"analysis manifest has invalid {name}")
    return value


def _signature_manifest_entry(
    manifest_path: Path,
) -> tuple[Path, ExternalStratumDescriptor, str, int]:
    manifest = load_json_mapping(manifest_path, label="Goal 3 analysis manifest")
    if manifest.get("schema_version") != ANALYSIS_ARTIFACT_SCHEMA_VERSION:
        raise AnalysisArtifactError("analysis manifest schema version is not supported")
    entries = manifest.get("external_strata")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise AnalysisArtifactError(
            "analysis manifest must have one external operator-signature stratum"
        )
    entry = entries[0]
    products = manifest.get("products")
    if not isinstance(products, list):
        raise AnalysisArtifactError("analysis manifest has no product list")
    matching_products = [
        product
        for product in products
        if isinstance(product, dict) and product.get("path") == entry.get("path")
    ]
    if len(matching_products) != 1 or matching_products[0] != entry:
        raise AnalysisArtifactError(
            "analysis manifest operator-signature product descriptor is inconsistent"
        )
    descriptor = ExternalStratumDescriptor.from_mapping(entry)
    root = manifest_path.parent.resolve()
    source = (root / descriptor.path).resolve()
    if source.parent != root:
        raise AnalysisArtifactError(
            "operator-signature sidecar resolves outside the analysis directory"
        )
    expected_sha256 = _manifest_sha256(
        entry.get("sha256"),
        name="operator-signature physical SHA-256",
    )
    expected_byte_count = _manifest_count(
        entry.get("byte_count"),
        name="operator-signature physical byte count",
    )
    if not source.is_file():
        raise AnalysisArtifactError(f"missing operator-signature sidecar: {source}")
    if source.stat().st_size != expected_byte_count or sha256_file(source) != expected_sha256:
        raise AnalysisArtifactError("operator-signature sidecar physical identity differs")

    summary_products = [
        product
        for product in products
        if isinstance(product, dict) and product.get("path") == METRICS_SUMMARY_NAME
    ]
    if len(summary_products) != 1:
        raise AnalysisArtifactError("analysis manifest has no unique metrics summary")
    summary_product = summary_products[0]
    summary_path = (root / METRICS_SUMMARY_NAME).resolve()
    if summary_path.parent != root or not summary_path.is_file():
        raise AnalysisArtifactError("metrics summary is missing or outside its analysis directory")
    summary_sha256 = _manifest_sha256(
        summary_product.get("sha256"),
        name="metrics-summary SHA-256",
    )
    summary_byte_count = _manifest_count(
        summary_product.get("byte_count"),
        name="metrics-summary byte count",
    )
    if (
        summary_path.stat().st_size != summary_byte_count
        or sha256_file(summary_path) != summary_sha256
    ):
        raise AnalysisArtifactError("metrics summary physical identity differs")
    summary = load_json_mapping(summary_path, label="Goal 3 metrics summary")
    if summary.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise AnalysisArtifactError("metrics summary schema version is not supported")
    if summary.get("external_strata") != [descriptor.as_dict()]:
        raise AnalysisArtifactError(
            "metrics summary and manifest disagree about operator-signature strata"
        )
    overall_payload = summary.get("overall")
    if not isinstance(overall_payload, dict):
        raise AnalysisArtifactError("metrics summary has no overall aggregate")
    try:
        overall = GroupStatistics.from_mapping(overall_payload)
    except (TypeError, ValueError) as error:
        raise AnalysisArtifactError("metrics summary overall aggregate is invalid") from error
    if (
        descriptor.all_processed_count,
        descriptor.valid_count,
        descriptor.failure_count,
    ) != (
        overall.all_processed_count,
        overall.valid_count,
        overall.failure_count,
    ):
        raise AnalysisArtifactError(
            "operator-signature descriptor denominators differ from the metrics summary"
        )
    return source, descriptor, expected_sha256, expected_byte_count


def _validated_group_denominators(
    payload: Mapping[str, object],
    *,
    line_number: int,
) -> tuple[int, int, int]:
    try:
        group = GroupStatistics.from_mapping(payload)
    except (TypeError, ValueError) as error:
        raise AnalysisArtifactError(
            f"operator-signature record {line_number} has invalid aggregates: {error}"
        ) from error
    return group.all_processed_count, group.valid_count, group.failure_count


def iter_operator_signature_group_payloads(
    manifest_path: str | Path,
) -> Iterator[dict[str, object]]:
    """Stream a saved signature axis and validate its complete logical identity.

    The caller must exhaust the iterator for final count and content-hash checks.
    """

    analysis_manifest = Path(manifest_path).resolve()
    source, descriptor, _, _ = _signature_manifest_entry(analysis_manifest)
    content_digest = hashlib.sha256()
    content_byte_count = 0
    group_count = 0
    total_processed = 0
    total_valid = 0
    total_failure = 0
    try:
        with gzip.open(source, "rb") as stream:
            previous_key: str | None = None
            for line_number, line in enumerate(stream, start=1):
                if not line.endswith(b"\n"):
                    raise AnalysisArtifactError(
                        f"operator-signature record {line_number} lacks a newline"
                    )
                try:
                    payload = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise AnalysisArtifactError(
                        f"invalid operator-signature JSONL record {line_number}: {source}"
                    ) from error
                if not isinstance(payload, dict):
                    raise AnalysisArtifactError(
                        f"operator-signature record {line_number} is not an object: {source}"
                    )
                key = payload.get("key")
                if not isinstance(key, str) or not key:
                    raise AnalysisArtifactError(
                        f"operator-signature record {line_number} has no key: {source}"
                    )
                if previous_key is not None and key <= previous_key:
                    raise AnalysisArtifactError(
                        "operator-signature records are not strictly key-ordered"
                    )
                if line != (canonical_json(payload) + "\n").encode("utf-8"):
                    raise AnalysisArtifactError(
                        f"operator-signature record {line_number} is not canonical"
                    )
                processed, valid, failure = _validated_group_denominators(
                    payload,
                    line_number=line_number,
                )
                content_digest.update(line)
                content_byte_count += len(line)
                group_count += 1
                total_processed += processed
                total_valid += valid
                total_failure += failure
                previous_key = key
                yield payload
    except (EOFError, gzip.BadGzipFile, OSError) as error:
        raise AnalysisArtifactError(f"cannot read operator-signature sidecar: {source}") from error
    observed = (
        group_count,
        content_digest.hexdigest(),
        content_byte_count,
        total_processed,
        total_valid,
        total_failure,
    )
    expected = (
        descriptor.group_count,
        descriptor.content_sha256,
        descriptor.uncompressed_byte_count,
        descriptor.all_processed_count,
        descriptor.valid_count,
        descriptor.failure_count,
    )
    if observed != expected:
        raise AnalysisArtifactError(
            "operator-signature sidecar logical identity differs from its descriptor"
        )


@dataclass(slots=True)
class _ReuseAccumulator:
    valid_count: int = 0
    reused_nodes: int = 0
    reused_references: int = 0
    excess_references: int = 0
    child_overhead: int = 0
    max_reuse_indegree: int = 0
    reuse_depth_sum: int = 0
    reuse_depth_count: int = 0
    concentration_mean: _DecimalMeanAccumulator = field(default_factory=_DecimalMeanAccumulator)

    def add(self, reuse: RowReuse) -> None:
        self.valid_count += 1
        self.reused_nodes += reuse.reused_node_count
        self.reused_references += reuse.reused_reference_count
        self.excess_references += reuse.excess_reference_count
        self.child_overhead += reuse.child_reference_overhead
        self.max_reuse_indegree = max(
            self.max_reuse_indegree,
            reuse.max_reuse_indegree,
        )
        self.reuse_depth_sum += reuse.reuse_depth_sum
        self.reuse_depth_count += reuse.reuse_depth_count
        self.concentration_mean.add(reuse.sharing_concentration)

    def finish(self) -> ReuseAggregate:
        if self.valid_count <= 0:
            raise ValueError("cannot finish an empty reuse aggregate")
        return ReuseAggregate(
            total_reused_node_count=self.reused_nodes,
            total_reused_reference_count=self.reused_references,
            total_excess_reference_count=self.excess_references,
            total_child_reference_overhead=self.child_overhead,
            max_reuse_indegree=self.max_reuse_indegree,
            mean_reused_node_count=ExactMean.from_total(
                self.reused_nodes,
                self.valid_count,
            ),
            mean_reused_reference_count=ExactMean.from_total(
                self.reused_references,
                self.valid_count,
            ),
            mean_excess_reference_count=ExactMean.from_total(
                self.excess_references,
                self.valid_count,
            ),
            mean_child_reference_overhead=ExactMean.from_total(
                self.child_overhead,
                self.valid_count,
            ),
            mean_reuse_depth=(
                ExactMean.from_total(self.reuse_depth_sum, self.reuse_depth_count)
                if self.reuse_depth_count
                else None
            ),
            mean_sharing_concentration=self.concentration_mean.mean(),
        )


@dataclass(slots=True)
class _GroupAccumulator:
    all_processed_count: int = 0
    valid_count: int = 0
    ratio_means: dict[str, _DecimalMeanAccumulator] = field(
        default_factory=lambda: {name: _DecimalMeanAccumulator() for name in RATIO_NAMES}
    )
    ast_reuse: _ReuseAccumulator = field(default_factory=_ReuseAccumulator)
    eml_reuse: _ReuseAccumulator = field(default_factory=_ReuseAccumulator)

    def add(self, row: AnalysisRow) -> None:
        self.add_metrics(row.metrics)

    def add_metrics(self, metrics: ValidMetrics | None) -> None:
        self.all_processed_count += 1
        if metrics is None:
            return
        self.valid_count += 1
        for name in RATIO_NAMES:
            self.ratio_means[name].add(metrics.ratio(name))
        self.ast_reuse.add(metrics.ast_reuse)
        self.eml_reuse.add(metrics.eml_reuse)

    def finish(self, key: str) -> GroupStatistics:
        means = (
            tuple(
                (
                    name,
                    self.ratio_means[name].mean(),
                )
                for name in RATIO_NAMES
            )
            if self.valid_count
            else ()
        )
        return GroupStatistics(
            key=key,
            all_processed_count=self.all_processed_count,
            valid_count=self.valid_count,
            failure_count=self.all_processed_count - self.valid_count,
            ratio_means=means,
            ast_reuse=self.ast_reuse.finish() if self.valid_count else None,
            eml_reuse=self.eml_reuse.finish() if self.valid_count else None,
        )


def _reuse_observation(reuse: RowReuse) -> list[int]:
    return [
        reuse.reused_node_count,
        reuse.reused_reference_count,
        reuse.child_reference_overhead,
        reuse.max_reuse_indegree,
        reuse.reuse_depth_sum,
        reuse.reuse_depth_count,
        reuse.sharing_concentration.numerator,
        reuse.sharing_concentration.denominator,
    ]


def _signature_observation(metrics: ValidMetrics | None) -> str | None:
    if metrics is None:
        return None
    ratios = [
        [metrics.ratio(name).numerator, metrics.ratio(name).denominator] for name in RATIO_NAMES
    ]
    return canonical_json(
        [
            ratios,
            _reuse_observation(metrics.ast_reuse),
            _reuse_observation(metrics.eml_reuse),
        ]
    )


def _reuse_from_observation(value: object) -> RowReuse:
    if (
        not isinstance(value, list)
        or len(value) != 8
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise AnalysisArtifactError("temporary signature observation has invalid reuse data")
    (
        reused_nodes,
        reused_references,
        child_overhead,
        max_reuse,
        depth_sum,
        depth_count,
        concentration_numerator,
        concentration_denominator,
    ) = value
    try:
        concentration = Fraction(concentration_numerator, concentration_denominator)
    except ZeroDivisionError as error:
        raise AnalysisArtifactError(
            "temporary signature observation has invalid concentration"
        ) from error
    excess = reused_references - reused_nodes
    if (
        min(
            reused_nodes,
            reused_references,
            child_overhead,
            max_reuse,
            depth_sum,
            depth_count,
        )
        < 0
        or excess < 0
        or excess != child_overhead
        or depth_count != reused_nodes
        or (reused_nodes == 0 and any(value[:6]))
        or (reused_nodes > 0 and max_reuse < 2)
    ):
        raise AnalysisArtifactError("temporary signature observation has inconsistent reuse data")
    return RowReuse(
        reused_node_count=reused_nodes,
        reused_reference_count=reused_references,
        excess_reference_count=excess,
        child_reference_overhead=child_overhead,
        max_reuse_indegree=max_reuse,
        reuse_depth_sum=depth_sum,
        reuse_depth_count=depth_count,
        sharing_concentration=concentration,
    )


def _metrics_from_signature_observation(payload: str | None) -> ValidMetrics | None:
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise AnalysisArtifactError("temporary signature observation is invalid") from error
    if not isinstance(value, list) or len(value) != 3 or not isinstance(value[0], list):
        raise AnalysisArtifactError("temporary signature observation has invalid shape")
    ratio_payload = value[0]
    if len(ratio_payload) != len(RATIO_NAMES):
        raise AnalysisArtifactError("temporary signature observation has incomplete ratios")
    ratios: list[tuple[str, Fraction]] = []
    for name, parts in zip(RATIO_NAMES, ratio_payload, strict=True):
        if (
            not isinstance(parts, list)
            or len(parts) != 2
            or any(isinstance(part, bool) or not isinstance(part, int) for part in parts)
        ):
            raise AnalysisArtifactError("temporary signature observation has invalid ratio")
        try:
            ratio = Fraction(parts[0], parts[1])
        except ZeroDivisionError as error:
            raise AnalysisArtifactError(
                "temporary signature observation has invalid ratio"
            ) from error
        ratios.append((name, ratio))
    return ValidMetrics(
        ratios=tuple(ratios),
        ast_reuse=_reuse_from_observation(value[1]),
        eml_reuse=_reuse_from_observation(value[2]),
    )


class _SignatureObservationStore:
    """Disk-backed signature observations sorted with bounded Python memory."""

    def __init__(self) -> None:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".geml-goal3-signature-observations-",
            suffix=".sqlite3",
        )
        os.close(file_descriptor)
        self.path = Path(temporary_name)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-8192")
        self.connection.execute(
            """
            CREATE TABLE observations (
                signature_key BLOB NOT NULL,
                signature TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                payload TEXT
            )
            """
        )
        self.sequence = 0
        self.closed = False

    def add(self, row: AnalysisRow) -> None:
        signature = row.stratum(StratificationAxis.OPERATOR_SIGNATURE)
        self.connection.execute(
            "INSERT INTO observations VALUES (?, ?, ?, ?)",
            (
                signature.encode("utf-8"),
                signature,
                self.sequence,
                _signature_observation(row.metrics),
            ),
        )
        self.sequence += 1

    def iter_ordered(self) -> Iterator[tuple[str, str | None]]:
        self.connection.commit()
        self.connection.execute(
            "CREATE INDEX observations_order ON observations(signature_key, sequence)"
        )
        cursor = self.connection.execute(
            """
            SELECT signature, payload
            FROM observations
            ORDER BY signature_key, sequence
            """
        )
        yield from cursor

    def close(self) -> None:
        if self.closed:
            return
        self.connection.close()
        self.closed = True
        self.path.unlink(missing_ok=True)


def _build_operator_signature_spool(
    observations: Iterator[tuple[str, str | None]],
) -> tuple[ExternalStratumDescriptor, _ManagedSidecarSpool]:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-goal3-signatures-",
        suffix=".jsonl.gz",
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    content_digest = hashlib.sha256()
    content_byte_count = 0
    group_count = 0
    total_processed = 0
    total_valid = 0
    total_failure = 0
    current_key: str | None = None
    accumulator: _GroupAccumulator | None = None

    def write_group(stream: gzip.GzipFile) -> None:
        nonlocal content_byte_count
        nonlocal group_count
        nonlocal total_processed
        nonlocal total_valid
        nonlocal total_failure
        if current_key is None or accumulator is None:
            return
        group = accumulator.finish(current_key)
        line = _signature_group_line(group)
        stream.write(line)
        content_digest.update(line)
        content_byte_count += len(line)
        group_count += 1
        total_processed += group.all_processed_count
        total_valid += group.valid_count
        total_failure += group.failure_count

    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=OPERATOR_SIGNATURE_COMPRESSION_LEVEL,
                fileobj=raw_stream,
                mtime=0,
            ) as compressed_stream:
                for signature, payload in observations:
                    if signature != current_key:
                        write_group(compressed_stream)
                        current_key = signature
                        accumulator = _GroupAccumulator()
                    if accumulator is None:  # pragma: no cover - guarded above
                        raise AssertionError("signature accumulator was not initialized")
                    accumulator.add_metrics(_metrics_from_signature_observation(payload))
                write_group(compressed_stream)
            raw_stream.flush()
            os.fsync(raw_stream.fileno())

        descriptor = ExternalStratumDescriptor(
            schema_version=OPERATOR_SIGNATURE_SIDECAR_SCHEMA_VERSION,
            axis=StratificationAxis.OPERATOR_SIGNATURE,
            path=OPERATOR_SIGNATURE_SIDECAR_NAME,
            group_count=group_count,
            all_processed_count=total_processed,
            valid_count=total_valid,
            failure_count=total_failure,
            content_sha256=content_digest.hexdigest(),
            uncompressed_byte_count=content_byte_count,
        )
        spool = _ManagedSidecarSpool(
            path=temporary,
            sha256=sha256_file(temporary),
            byte_count=temporary.stat().st_size,
        )
        return descriptor, spool
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _checkpoint_snapshot(
    processed_count: int,
    accumulator: _GroupAccumulator,
) -> CheckpointMetrics:
    group = accumulator.finish("all")
    return CheckpointMetrics(
        processed_count=processed_count,
        valid_count=group.valid_count,
        failure_count=group.failure_count,
        ratio_means=group.ratio_means,
    )


def analyze_goal3_artifacts(
    manifest_path: str | Path,
    *,
    scale_checkpoints: Sequence[int] | None = None,
    ranking_limit: int = 20,
) -> AnalysisReport:
    """Analyze a validated saved Goal 3 run without modifying its artifacts."""

    from geml.analysis.goal3.failures import OutcomeMiner
    from geml.plots.goal3 import (
        STANDARD_SCALE_CHECKPOINTS,
        build_runtime_curve,
        build_stability_curve,
    )

    source = Path(manifest_path).resolve()
    source_manifest_sha256 = sha256_file(source)
    manifest = load_json_mapping(source, label="Goal 3 completion manifest")
    processed_expected = _required_integer(
        manifest.get("processed_count"),
        name="manifest processed count",
        minimum=1,
    )
    success_expected = _required_integer(
        manifest.get("success_count"),
        name="manifest success count",
    )
    failure_expected = _required_integer(
        manifest.get("failure_count"),
        name="manifest failure count",
    )
    if processed_expected != success_expected + failure_expected:
        raise AnalysisArtifactError("Goal 3 summary denominators are inconsistent")

    requested = (
        tuple(scale_checkpoints) if scale_checkpoints is not None else STANDARD_SCALE_CHECKPOINTS
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in requested
    ) or requested != tuple(sorted(set(requested))):
        raise ValueError("scale checkpoints must be positive, sorted, and unique")
    active_checkpoints = tuple(value for value in requested if value <= processed_expected)

    overall = _GroupAccumulator()
    by_axis: dict[StratificationAxis, dict[str, _GroupAccumulator]] = {
        axis: defaultdict(_GroupAccumulator)
        for axis in StratificationAxis
        if axis is not StratificationAxis.OPERATOR_SIGNATURE
    }
    signature_observations = _SignatureObservationStore()
    checkpoints: list[CheckpointMetrics] = []
    checkpoint_set = set(active_checkpoints)
    outcomes = OutcomeMiner(limit=ranking_limit)
    processed = 0
    try:
        for table in iter_metric_tables(source):
            for batch in table.to_batches(max_chunksize=2_048):
                for mapping in batch.to_pylist():
                    row = AnalysisRow.from_mapping(mapping)
                    processed += 1
                    overall.add(row)
                    for axis, groups in by_axis.items():
                        groups[row.stratum(axis)].add(row)
                    signature_observations.add(row)
                    outcomes.add(row)
                    if processed in checkpoint_set:
                        checkpoints.append(_checkpoint_snapshot(processed, overall))

        overall_finished = overall.finish("all")
        if (
            processed != processed_expected
            or overall_finished.valid_count != success_expected
            or overall_finished.failure_count != failure_expected
        ):
            raise AnalysisArtifactError("analysis denominators differ from the validated summary")
        missing_metric_checkpoints = sorted(
            checkpoint_set - {checkpoint.processed_count for checkpoint in checkpoints}
        )
        if missing_metric_checkpoints:
            raise AnalysisArtifactError(
                f"metric rows do not reach checkpoints {missing_metric_checkpoints}"
            )

        strata = tuple(
            StratifiedTable(
                axis=axis,
                groups=tuple(groups[key].finish(key) for key in sorted(groups)),
            )
            for axis, groups in by_axis.items()
        )
        operator_signature_stratum, operator_signature_spool = _build_operator_signature_spool(
            signature_observations.iter_ordered()
        )
    finally:
        signature_observations.close()
    if (
        operator_signature_stratum.all_processed_count != overall_finished.all_processed_count
        or operator_signature_stratum.valid_count != overall_finished.valid_count
        or operator_signature_stratum.failure_count != overall_finished.failure_count
    ):
        raise AnalysisArtifactError(
            "operator-signature strata do not cover the overall analysis denominators"
        )
    runtime_curve = build_runtime_curve(iter_shard_telemetry(source))
    if sha256_file(source) != source_manifest_sha256:
        raise AnalysisArtifactError("Goal 3 manifest changed while it was being analyzed")
    stability_curve = build_stability_curve(
        checkpoints,
        runtime_curve,
        required_checkpoints=active_checkpoints,
    )
    standard_present = {point.processed_count for point in stability_curve}
    missing_standard = tuple(
        checkpoint
        for checkpoint in STANDARD_SCALE_CHECKPOINTS
        if checkpoint <= processed_expected and checkpoint not in standard_present
    )
    summary_descriptor = manifest.get("summary")
    if not isinstance(summary_descriptor, dict):
        raise AnalysisArtifactError("validated manifest has no summary descriptor")
    summary_payload = load_json_mapping(
        source.parent / str(summary_descriptor.get("path", "")),
        label="validated Goal 3 summary",
    )
    source_science_fingerprint = _required_text(
        summary_payload.get("science_fingerprint"),
        name="summary science fingerprint",
    )
    provisional = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "source_manifest_path": source.as_posix(),
        "source_manifest_sha256": source_manifest_sha256,
        "source_science_fingerprint": source_science_fingerprint,
        "source_stage": _required_text(manifest.get("stage"), name="source stage"),
        "overall": overall_finished,
        "strata": strata,
        "operator_signature_stratum": operator_signature_stratum,
        "operator_signature_spool": operator_signature_spool,
        "checkpoints": tuple(checkpoints),
        "stability_curve": stability_curve,
        "outcomes": outcomes.finish(),
        "missing_standard_checkpoints": missing_standard,
    }
    report_without_fingerprint = AnalysisReport(
        **provisional,
        fingerprint="",
    )
    fingerprint_payload = report_without_fingerprint.as_dict()
    fingerprint_payload.pop("fingerprint")
    fingerprint_payload["source"].pop("manifest_path")
    fingerprint = _canonical_sha256(fingerprint_payload)
    return AnalysisReport(**provisional, fingerprint=fingerprint)


def _assert_external_output(source_manifest: Path, output_directory: Path) -> None:
    source_root = source_manifest.resolve().parent
    destination = output_directory.resolve()
    if destination == source_root or source_root in destination.parents:
        raise AnalysisArtifactError(
            "analysis output must be outside the read-only Goal 3 run artifact tree"
        )


def save_analysis(
    report: AnalysisReport,
    output_directory: str | Path,
) -> SavedAnalysis:
    """Write deterministic tables and plot-ready data outside source artifacts."""

    from geml.plots.goal3 import plot_data_payload

    source = Path(report.source_manifest_path)
    destination = Path(output_directory).resolve()
    _assert_external_output(source, destination)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / METRICS_SUMMARY_NAME
    operator_signature_strata_path = destination / OPERATOR_SIGNATURE_SIDECAR_NAME
    outcomes_path = destination / "outcomes.table.json"
    plot_data_path = destination / "stability.plot-data.json"
    manifest_path = destination / "analysis.manifest.json"

    spool = report.operator_signature_spool
    if (
        not spool.path.is_file()
        or spool.path.stat().st_size != spool.byte_count
        or sha256_file(spool.path) != spool.sha256
    ):
        raise AnalysisArtifactError("managed operator-signature spool is missing or corrupt")
    signature_sha256, signature_byte_count = publish_temporary_file(
        spool.path,
        operator_signature_strata_path,
    )
    atomic_write_json(metrics_path, report.metrics_payload(), resume_identical=True)
    atomic_write_json(
        outcomes_path,
        report.outcomes.as_dict(),
        resume_identical=True,
    )
    atomic_write_json(
        plot_data_path,
        plot_data_payload(report.stability_curve),
        resume_identical=True,
    )
    regular_products = tuple(
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        }
        for path in (metrics_path, outcomes_path, plot_data_path)
    )
    signature_descriptor = report.operator_signature_stratum.as_dict()
    signature_manifest_entry = {
        **signature_descriptor,
        "sha256": signature_sha256,
        "byte_count": signature_byte_count,
    }
    products = (
        regular_products[0],
        signature_manifest_entry,
        *regular_products[1:],
    )
    atomic_write_json(
        manifest_path,
        {
            "schema_version": ANALYSIS_ARTIFACT_SCHEMA_VERSION,
            "analysis_fingerprint": report.fingerprint,
            "source_manifest_path": report.source_manifest_path,
            "source_manifest_sha256": report.source_manifest_sha256,
            "source_science_fingerprint": report.source_science_fingerprint,
            "products": list(products),
            "external_strata": [signature_manifest_entry],
            "reproduction_command": (
                "python -m geml.analysis.goal3.metrics "
                f"--manifest {report.source_manifest_path} "
                f"--output-dir {destination.as_posix()}"
            ),
        },
        resume_identical=True,
    )
    return SavedAnalysis(
        output_directory=destination,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        operator_signature_strata_path=operator_signature_strata_path,
        outcomes_path=outcomes_path,
        plot_data_path=plot_data_path,
        analysis_fingerprint=report.fingerprint,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ranking-limit", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run reproducible analysis and print its deterministic product manifest."""

    arguments = _parser().parse_args(argv)
    report: AnalysisReport | None = None
    try:
        report = analyze_goal3_artifacts(
            arguments.manifest,
            ranking_limit=arguments.ranking_limit,
        )
        saved = save_analysis(report, arguments.output_dir)
    except (AnalysisArtifactError, Goal3ArtifactError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                indent=2,
                sort_keys=True,
            ),
            file=os.sys.stderr,
        )
        return 2
    finally:
        if report is not None:
            report.operator_signature_spool.cleanup()
    print(
        json.dumps(
            {
                "passed": True,
                "analysis_fingerprint": saved.analysis_fingerprint,
                "manifest_path": saved.manifest_path.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
