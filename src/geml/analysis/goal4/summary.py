"""Deterministic, claim-disciplined summary of Goal 4 experiment rows.

This module reads the JSONL rows emitted by :mod:`geml.experiments.goal4.run` and produces
mode-separated statistics with explicit denominators.  Two denominators are always reported
side by side: a success-only view (over rows that produced an official cost improvement
measurement) and an all-processed view (over every retained row, including failures).  The
two rewrite modes are never merged, averaged, or mixed.

Nothing here asserts optimality.  An "improvement" is only the difference between the
official Goal 3 EML DAG cost of the input and of the best enumerated, validated candidate
under the configured resource limits.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

_OPTIMIZED = "optimized"
_UNCHANGED = "unchanged"
_COSTED_STATUSES = frozenset({_OPTIMIZED, _UNCHANGED})


class SummaryError(ValueError):
    """A Goal 4 result row is missing a required field or is malformed."""


@dataclass(frozen=True, slots=True)
class ResultRow:
    """One validated Goal 4 result row projected to the fields the analysis needs.

    ``cost_before`` and ``cost_after`` are present only for rows that reached official cost
    evaluation; every other field is always present so failed rows remain fully accountable.
    """

    expression_id: str
    rewrite_mode: str
    domain_mode: str
    operator_family: str
    split: str
    size_bucket: str
    rule_library: str
    stage_status: str
    saturation_status: str | None
    extraction_status: str | None
    validation_status: str | None
    timeout: bool
    failure_reason: str | None
    rewrites_applied: int | None
    candidate_count: int | None
    branch_sensitive_applications: int
    applied_rules: Mapping[str, int]
    cost_before: int | None
    cost_after: int | None

    @property
    def costed(self) -> bool:
        """Return whether the row carries a before/after official cost measurement."""
        return self.cost_before is not None and self.cost_after is not None

    @property
    def improved(self) -> bool:
        """Return whether the best candidate strictly reduced the official EML DAG cost."""
        return self.costed and self.cost_after < self.cost_before  # type: ignore[operator]

    @property
    def absolute_improvement(self) -> int | None:
        """Return the exact cost reduction, or ``None`` when uncosted."""
        if not self.costed:
            return None
        return self.cost_before - self.cost_after  # type: ignore[operator]


def _require(row: Mapping[str, object], key: str) -> object:
    if key not in row:
        raise SummaryError(f"result row is missing required field {key!r}")
    return row[key]


def _as_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SummaryError(f"expected an integer or null, got {value!r}")
    return value


def parse_row(row: Mapping[str, object]) -> ResultRow:
    """Project one raw JSON row into a typed :class:`ResultRow`."""
    provenance = row.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        raise SummaryError("provenance must be a mapping when present")
    applied = provenance.get("applied_rules") or {}
    if not isinstance(applied, Mapping):
        raise SummaryError("applied_rules must be a mapping when present")
    branch_sensitive = provenance.get("branch_sensitive_applications") or 0
    if isinstance(branch_sensitive, bool) or not isinstance(branch_sensitive, int):
        raise SummaryError("branch_sensitive_applications must be an integer")
    return ResultRow(
        expression_id=str(_require(row, "expression_id")),
        rewrite_mode=str(_require(row, "rewrite_mode")),
        domain_mode=str(row.get("domain_mode", "")),
        operator_family=str(row.get("operator_family", "")),
        split=str(row.get("split", "")),
        size_bucket=str(row.get("size_bucket", "")),
        rule_library=str(row.get("rule_library", "")),
        stage_status=str(_require(row, "stage_status")),
        saturation_status=_optional_str(row.get("saturation_status")),
        extraction_status=_optional_str(row.get("extraction_status")),
        validation_status=_optional_str(row.get("validation_status")),
        timeout=bool(row.get("timeout", False)),
        failure_reason=_optional_str(row.get("failure_reason")),
        rewrites_applied=_as_int_or_none(row.get("rewrites_applied")),
        candidate_count=_as_int_or_none(row.get("candidate_count")),
        branch_sensitive_applications=branch_sensitive,
        applied_rules={str(name): int(count) for name, count in applied.items()},
        cost_before=_as_int_or_none(row.get("eml_dag_cost_before")),
        cost_after=_as_int_or_none(row.get("eml_dag_cost_after")),
    )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def parse_rows(rows: Iterable[Mapping[str, object]]) -> tuple[ResultRow, ...]:
    """Project every raw row, preserving order and dropping nothing."""
    return tuple(parse_row(row) for row in rows)


@dataclass(frozen=True, slots=True)
class ExactRatio:
    """A ratio stored as both an exact reduced fraction and a float rendering."""

    numerator: int
    denominator: int

    @classmethod
    def of(cls, numerator: int, denominator: int) -> ExactRatio:
        """Build a reduced ratio; a zero denominator yields ``0/0`` sentinel."""
        if denominator == 0:
            return cls(0, 0)
        fraction = Fraction(numerator, denominator)
        return cls(fraction.numerator, fraction.denominator)

    @property
    def value(self) -> float | None:
        """Return the float value, or ``None`` when the denominator is zero."""
        if self.denominator == 0:
            return None
        return self.numerator / self.denominator

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy with both exact and float renderings."""
        return {
            "exact": f"{self.numerator}/{self.denominator}",
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class GroupSummary:
    """Success-only and all-processed statistics for one group of rows.

    Every rate carries its explicit denominator.  ``success_rate`` is over costed rows;
    ``processing_success_rate`` is over every processed row in the group, so failures stay
    visible in the denominator.
    """

    label: str
    rewrite_mode: str
    processed_count: int
    costed_count: int
    improved_count: int
    failure_count: int
    timeout_count: int
    total_absolute_improvement: int
    success_rate: ExactRatio
    processing_success_rate: ExactRatio
    mean_absolute_improvement_over_costed: ExactRatio
    mean_relative_improvement_permille_over_improved: ExactRatio

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "label": self.label,
            "rewrite_mode": self.rewrite_mode,
            "denominators": {
                "processed": self.processed_count,
                "costed": self.costed_count,
                "improved": self.improved_count,
                "failures": self.failure_count,
                "timeouts": self.timeout_count,
            },
            "total_absolute_improvement": self.total_absolute_improvement,
            "improved_over_costed": self.success_rate.as_dict(),
            "costed_over_processed": self.processing_success_rate.as_dict(),
            "mean_absolute_improvement_over_costed": (
                self.mean_absolute_improvement_over_costed.as_dict()
            ),
            "mean_relative_improvement_permille_over_improved": (
                self.mean_relative_improvement_permille_over_improved.as_dict()
            ),
        }


def summarize_group(label: str, rewrite_mode: str, rows: Sequence[ResultRow]) -> GroupSummary:
    """Summarize one already-filtered, single-mode group of rows.

    The caller is responsible for restricting ``rows`` to a single rewrite mode; this
    function never mixes modes.
    """
    processed = len(rows)
    costed = [row for row in rows if row.costed]
    improved = [row for row in costed if row.improved]
    failures = [row for row in rows if row.stage_status not in _COSTED_STATUSES]
    timeouts = [row for row in rows if row.timeout]
    total_improvement = sum(row.absolute_improvement or 0 for row in improved)
    relative_permille_sum = sum(_relative_permille(row) for row in improved if row.cost_before)
    return GroupSummary(
        label=label,
        rewrite_mode=rewrite_mode,
        processed_count=processed,
        costed_count=len(costed),
        improved_count=len(improved),
        failure_count=len(failures),
        timeout_count=len(timeouts),
        total_absolute_improvement=total_improvement,
        success_rate=ExactRatio.of(len(improved), len(costed)),
        processing_success_rate=ExactRatio.of(len(costed), processed),
        mean_absolute_improvement_over_costed=ExactRatio.of(total_improvement, len(costed)),
        mean_relative_improvement_permille_over_improved=ExactRatio.of(
            relative_permille_sum, len(improved)
        ),
    )


def _relative_permille(row: ResultRow) -> int:
    """Return the per-mille relative improvement as an exact rounded integer."""
    if not row.cost_before:
        return 0
    improvement = row.absolute_improvement or 0
    return (improvement * 1000) // row.cost_before


_STRATIFICATION_AXES: tuple[str, ...] = (
    "operator_family",
    "size_bucket",
    "split",
    "domain_mode",
    "stage_status",
    "saturation_status",
)


@dataclass(frozen=True, slots=True)
class ModeReport:
    """The full stratified summary for one rewrite mode."""

    rewrite_mode: str
    overall: GroupSummary
    nontrivial: GroupSummary
    identity_heavy: GroupSummary
    strata: Mapping[str, Mapping[str, GroupSummary]]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "rewrite_mode": self.rewrite_mode,
            "overall": self.overall.as_dict(),
            "nontrivial": self.nontrivial.as_dict(),
            "identity_heavy": self.identity_heavy.as_dict(),
            "strata": {
                axis: {key: group.as_dict() for key, group in groups.items()}
                for axis, groups in self.strata.items()
            },
        }


@dataclass(frozen=True, slots=True)
class Goal4Summary:
    """The complete Goal 4 summary, always separated by rewrite mode."""

    total_rows: int
    modes: Mapping[str, ModeReport]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "total_rows": self.total_rows,
            "modes": {mode: report.as_dict() for mode, report in self.modes.items()},
        }


def _is_nontrivial(row: ResultRow) -> bool:
    """Return whether a row's saturation applied at least one rewrite."""
    return bool(row.rewrites_applied)


def _is_identity_heavy(row: ResultRow) -> bool:
    """Return whether a row's improvement came mainly from identity/fold rules.

    A row is identity-heavy when it improved and every applied rule name contains one of
    the identity, inverse, fold, or zero markers; this is a descriptive label, not a claim
    about optimality.
    """
    if not row.improved or not row.applied_rules:
        return False
    markers = ("ZERO", "ONE", "INVERSE", "NEG", "FOLD")
    return all(any(marker in name.upper() for marker in markers) for name in row.applied_rules)


def summarize(rows: Iterable[Mapping[str, object]]) -> Goal4Summary:
    """Build the complete mode-separated Goal 4 summary from raw JSON rows.

    Rows are grouped by rewrite mode and never mixed.  For each mode the summary reports an
    overall group, a nontrivial subgroup, an identity-heavy subgroup, and one group per
    stratum value along every stratification axis.
    """
    parsed = parse_rows(rows)
    by_mode: dict[str, list[ResultRow]] = {}
    for row in parsed:
        by_mode.setdefault(row.rewrite_mode, []).append(row)

    reports: dict[str, ModeReport] = {}
    for mode in sorted(by_mode):
        mode_rows = by_mode[mode]
        strata: dict[str, dict[str, GroupSummary]] = {}
        for axis in _STRATIFICATION_AXES:
            groups: dict[str, list[ResultRow]] = {}
            for row in mode_rows:
                groups.setdefault(str(getattr(row, axis)), []).append(row)
            strata[axis] = {
                key: summarize_group(f"{axis}={key}", mode, sorted_rows)
                for key, sorted_rows in sorted(groups.items())
            }
        reports[mode] = ModeReport(
            rewrite_mode=mode,
            overall=summarize_group("overall", mode, mode_rows),
            nontrivial=summarize_group(
                "nontrivial", mode, [row for row in mode_rows if _is_nontrivial(row)]
            ),
            identity_heavy=summarize_group(
                "identity_heavy", mode, [row for row in mode_rows if _is_identity_heavy(row)]
            ),
            strata=strata,
        )
    return Goal4Summary(total_rows=len(parsed), modes=reports)
