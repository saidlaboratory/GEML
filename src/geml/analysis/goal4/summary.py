"""Strict, denominator-explicit analysis of Goal 4 result rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from geml.egraph.policy import RewriteMode
from geml.experiments.goal4.run import ROW_SCHEMA_VERSION, StageStatus

_SUCCESS_STATUSES = frozenset({StageStatus.OPTIMIZED.value, StageStatus.UNCHANGED.value})
_EXPECTED_MODES = frozenset(mode.value for mode in RewriteMode)


class SummaryError(ValueError):
    """A result row or result set violates the Goal 4 audit contract."""


@dataclass(frozen=True, slots=True)
class ResultRow:
    """Strict analysis projection of one retained experiment row."""

    run_id: str
    expression_id: str
    rewrite_mode: str
    domain_mode: str
    operator_family: str
    split: str
    size_bucket: str
    difficulty_profile: str
    rule_library: str
    stage_status: str
    saturation_status: str | None
    extraction_status: str | None
    validation_status: str | None
    timeout: bool
    failure_stage: str | None
    failure_reason: str | None
    rewrites_applied: int | None
    candidate_count: int | None
    branch_sensitive_applications: int
    applied_rules: Mapping[str, int]
    validation_failure_count: int
    cost_before: int | None
    cost_after: int | None
    wall_seconds: float | None
    rss_bytes_after: int | None

    @property
    def costed(self) -> bool:
        return self.cost_before is not None and self.cost_after is not None

    @property
    def improved(self) -> bool:
        return self.costed and self.cost_after < self.cost_before  # type: ignore[operator]

    @property
    def degraded(self) -> bool:
        return self.costed and self.cost_after > self.cost_before  # type: ignore[operator]

    @property
    def absolute_improvement(self) -> int | None:
        if not self.costed:
            return None
        return self.cost_before - self.cost_after  # type: ignore[operator]


def _require(row: Mapping[str, object], key: str) -> object:
    if key not in row:
        raise SummaryError(f"result row is missing required field {key!r}")
    return row[key]


def _required_str(row: Mapping[str, object], key: str) -> str:
    value = _require(row, key)
    if not isinstance(value, str) or not value:
        raise SummaryError(f"{key} must be a non-blank string")
    return value


def _optional_str(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SummaryError(f"{field} must be a string or null")
    return value


def _int_or_none(
    value: object,
    *,
    field: str,
    minimum: int = 0,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SummaryError(f"{field} must be null or an integer greater than or equal to {minimum}")
    return value


def _number_or_none(
    value: object,
    *,
    field: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SummaryError(f"{field} must be a finite nonnegative number or null")
    number = float(value)
    if number < 0 or number != number or number == float("inf"):
        raise SummaryError(f"{field} must be a finite nonnegative number or null")
    return number


def _positive_mapping(value: object, *, field: str) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SummaryError(f"{field} must be a mapping")
    result: dict[str, int] = {}
    for key, count in value.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise SummaryError(f"{field} must map non-blank strings to positive integers")
        result[key] = count
    return result


def parse_row(row: Mapping[str, object]) -> ResultRow:
    """Validate and project one row without coercing malformed values."""
    if not isinstance(row, Mapping):
        raise SummaryError("each result row must be a mapping")
    if _require(row, "schema_version") != ROW_SCHEMA_VERSION:
        raise SummaryError("result row has an incompatible schema_version")

    rewrite_mode = _required_str(row, "rewrite_mode")
    if rewrite_mode not in _EXPECTED_MODES:
        raise SummaryError(f"unknown rewrite_mode {rewrite_mode!r}")
    stage_status = _required_str(row, "stage_status")
    if stage_status not in {status.value for status in StageStatus}:
        raise SummaryError(f"unknown stage_status {stage_status!r}")
    timeout = _require(row, "timeout")
    if type(timeout) is not bool:
        raise SummaryError("timeout must be a boolean")

    provenance = row.get("provenance")
    if provenance is None:
        applied_rules: dict[str, int] = {}
        branch_sensitive = 0
    else:
        if not isinstance(provenance, Mapping):
            raise SummaryError("provenance must be a mapping or null")
        if (
            provenance.get("application_log_complete") is not True
            or provenance.get("attempt_aggregates_complete") is not True
        ):
            raise SummaryError(
                "present provenance must declare complete application and attempt accounting"
            )
        applied_rules = _positive_mapping(
            provenance.get("applied_rules"),
            field="provenance.applied_rules",
        )
        branch_sensitive = _int_or_none(
            provenance.get("branch_sensitive_applications", 0),
            field="branch_sensitive_applications",
        )
        assert branch_sensitive is not None

    validation_failures = row.get("validation_failures")
    validation_failure_counts = _positive_mapping(
        validation_failures,
        field="validation_failures",
    )
    before = _int_or_none(
        row.get("eml_dag_cost_before"),
        field="eml_dag_cost_before",
        minimum=1,
    )
    after = _int_or_none(
        row.get("eml_dag_cost_after"),
        field="eml_dag_cost_after",
        minimum=1,
    )
    if before is None and after is not None:
        raise SummaryError("a cost_after cannot exist without cost_before")
    costed = before is not None and after is not None
    recorded_improvement = row.get("absolute_improvement")
    if costed:
        if (
            isinstance(recorded_improvement, bool)
            or not isinstance(recorded_improvement, int)
            or recorded_improvement != before - after
        ):
            raise SummaryError("absolute_improvement does not match before - after")
    elif recorded_improvement is not None:
        raise SummaryError("uncosted rows cannot report absolute_improvement")

    failure_reason = _optional_str(
        row.get("failure_reason"),
        field="failure_reason",
    )
    if stage_status in _SUCCESS_STATUSES:
        if not costed:
            raise SummaryError("successful rows require before and after costs")
        if failure_reason is not None:
            raise SummaryError("successful rows cannot carry a failure_reason")
        if stage_status == StageStatus.OPTIMIZED.value and not after < before:
            raise SummaryError("optimized rows require a strict cost reduction")
        if stage_status == StageStatus.UNCHANGED.value and after != before:
            raise SummaryError("unchanged rows require equal before and after costs")
    if stage_status == StageStatus.DEGRADED_REJECTED.value and (not costed or not after > before):
        raise SummaryError("degraded_rejected rows require after > before")

    resources = _require(row, "resources")
    if not isinstance(resources, Mapping):
        raise SummaryError("resources must be a mapping")
    wall_seconds = _number_or_none(
        resources.get("wall_seconds"),
        field="resources.wall_seconds",
    )
    rss_after = _int_or_none(
        resources.get("rss_bytes_after"),
        field="resources.rss_bytes_after",
    )
    return ResultRow(
        run_id=_required_str(row, "run_id"),
        expression_id=_required_str(row, "expression_id"),
        rewrite_mode=rewrite_mode,
        domain_mode=_required_str(row, "domain_mode"),
        operator_family=_required_str(row, "operator_family"),
        split=_required_str(row, "split"),
        size_bucket=_required_str(row, "size_bucket"),
        difficulty_profile=_required_str(row, "difficulty_profile"),
        rule_library=_required_str(row, "rule_library"),
        stage_status=stage_status,
        saturation_status=_optional_str(
            row.get("saturation_status"),
            field="saturation_status",
        ),
        extraction_status=_optional_str(
            row.get("extraction_status"),
            field="extraction_status",
        ),
        validation_status=_optional_str(
            row.get("validation_status"),
            field="validation_status",
        ),
        timeout=timeout,
        failure_stage=_optional_str(
            row.get("failure_stage"),
            field="failure_stage",
        ),
        failure_reason=failure_reason,
        rewrites_applied=_int_or_none(
            row.get("rewrites_applied"),
            field="rewrites_applied",
        ),
        candidate_count=_int_or_none(
            row.get("candidate_count"),
            field="candidate_count",
        ),
        branch_sensitive_applications=branch_sensitive,
        applied_rules=applied_rules,
        validation_failure_count=sum(validation_failure_counts.values()),
        cost_before=before,
        cost_after=after,
        wall_seconds=wall_seconds,
        rss_bytes_after=rss_after,
    )


def parse_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[ResultRow, ...]:
    """Validate a complete paired-mode result set and reject duplicates."""
    parsed = tuple(parse_row(row) for row in rows)
    if not parsed:
        return ()
    run_ids = {row.run_id for row in parsed}
    if len(run_ids) != 1:
        raise SummaryError("result rows mix multiple run_id values")
    seen: set[tuple[str, str]] = set()
    by_expression: dict[str, list[ResultRow]] = {}
    for row in parsed:
        key = (row.expression_id, row.rewrite_mode)
        if key in seen:
            raise SummaryError(f"duplicate result work unit {key!r}")
        seen.add(key)
        by_expression.setdefault(row.expression_id, []).append(row)
    for expression_id, expression_rows in by_expression.items():
        modes = {row.rewrite_mode for row in expression_rows}
        if modes != _EXPECTED_MODES:
            raise SummaryError(
                f"expression {expression_id!r} does not have exactly both rewrite modes"
            )
        metadata = {
            (
                row.domain_mode,
                row.operator_family,
                row.split,
                row.size_bucket,
                row.difficulty_profile,
            )
            for row in expression_rows
        }
        if len(metadata) != 1:
            raise SummaryError(f"expression {expression_id!r} has inconsistent cross-mode metadata")
    return parsed


@dataclass(frozen=True, slots=True)
class ExactRatio:
    """An exact reduced ratio with a convenience float rendering."""

    numerator: int
    denominator: int

    @classmethod
    def of(cls, numerator: int, denominator: int) -> ExactRatio:
        if denominator == 0:
            return cls(0, 0)
        value = Fraction(numerator, denominator)
        return cls(value.numerator, value.denominator)

    @classmethod
    def from_fraction(cls, value: Fraction) -> ExactRatio:
        return cls(value.numerator, value.denominator)

    @property
    def value(self) -> float | None:
        return None if self.denominator == 0 else self.numerator / self.denominator

    def as_dict(self) -> dict[str, object]:
        return {
            "exact": f"{self.numerator}/{self.denominator}",
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class GroupSummary:
    """Outcome, coverage, and after-rate metrics for one single-mode group."""

    label: str
    rewrite_mode: str
    processed_count: int
    costed_count: int
    improved_count: int
    unchanged_count: int
    degraded_count: int
    failure_count: int
    timeout_count: int
    signed_total_improvement: int
    positive_total_improvement: int
    improved_over_costed: ExactRatio
    improved_over_processed: ExactRatio
    costed_over_processed: ExactRatio
    mean_signed_improvement_over_costed: ExactRatio
    mean_relative_improvement_over_costed: ExactRatio

    @property
    def success_rate(self) -> ExactRatio:
        """Compatibility name for the explicitly costed denominator."""
        return self.improved_over_costed

    @property
    def processing_success_rate(self) -> ExactRatio:
        """Compatibility name for cost coverage, not an all-processed after-rate."""
        return self.costed_over_processed

    @property
    def total_absolute_improvement(self) -> int:
        """Compatibility name; signed so degradation cannot be hidden."""
        return self.signed_total_improvement

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "rewrite_mode": self.rewrite_mode,
            "denominators": {
                "processed": self.processed_count,
                "costed": self.costed_count,
                "improved": self.improved_count,
                "unchanged": self.unchanged_count,
                "degraded": self.degraded_count,
                "failures": self.failure_count,
                "timeouts": self.timeout_count,
            },
            "signed_total_improvement": self.signed_total_improvement,
            "positive_total_improvement": self.positive_total_improvement,
            "improved_over_costed": self.improved_over_costed.as_dict(),
            "improved_over_processed": self.improved_over_processed.as_dict(),
            "costed_over_processed": self.costed_over_processed.as_dict(),
            "mean_signed_improvement_over_costed": (
                self.mean_signed_improvement_over_costed.as_dict()
            ),
            "mean_relative_improvement_over_costed": (
                self.mean_relative_improvement_over_costed.as_dict()
            ),
        }


def summarize_group(
    label: str,
    rewrite_mode: str,
    rows: Sequence[ResultRow],
) -> GroupSummary:
    processed = len(rows)
    costed = [row for row in rows if row.costed]
    improved = [row for row in costed if row.improved]
    unchanged = [row for row in costed if row.cost_before == row.cost_after]
    degraded = [row for row in costed if row.degraded]
    failures = [row for row in rows if row.stage_status not in _SUCCESS_STATUSES]
    signed_total = sum(row.absolute_improvement or 0 for row in costed)
    positive_total = sum(row.absolute_improvement or 0 for row in improved)
    relative_sum = sum(
        (
            Fraction(row.absolute_improvement or 0, row.cost_before)
            for row in costed
            if row.cost_before is not None
        ),
        start=Fraction(),
    )
    mean_relative = Fraction(0, 1) if not costed else relative_sum / len(costed)
    return GroupSummary(
        label=label,
        rewrite_mode=rewrite_mode,
        processed_count=processed,
        costed_count=len(costed),
        improved_count=len(improved),
        unchanged_count=len(unchanged),
        degraded_count=len(degraded),
        failure_count=len(failures),
        timeout_count=sum(1 for row in rows if row.timeout),
        signed_total_improvement=signed_total,
        positive_total_improvement=positive_total,
        improved_over_costed=ExactRatio.of(len(improved), len(costed)),
        improved_over_processed=ExactRatio.of(len(improved), processed),
        costed_over_processed=ExactRatio.of(len(costed), processed),
        mean_signed_improvement_over_costed=ExactRatio.of(
            signed_total,
            len(costed),
        ),
        mean_relative_improvement_over_costed=ExactRatio.from_fraction(mean_relative),
    )


_STRATIFICATION_AXES: tuple[str, ...] = (
    "operator_family",
    "size_bucket",
    "difficulty_profile",
    "split",
    "domain_mode",
    "stage_status",
    "saturation_status",
    "extraction_status",
    "validation_status",
)


@dataclass(frozen=True, slots=True)
class OutcomeExample:
    expression_id: str
    stage_status: str
    absolute_improvement: int | None
    failure_reason: str | None
    applied_rules: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "expression_id": self.expression_id,
            "stage_status": self.stage_status,
            "absolute_improvement": self.absolute_improvement,
            "failure_reason": self.failure_reason,
            "applied_rules": list(self.applied_rules),
        }


@dataclass(frozen=True, slots=True)
class ModeReport:
    rewrite_mode: str
    overall: GroupSummary
    nontrivial: GroupSummary
    identity_heavy: GroupSummary
    strata: Mapping[str, Mapping[str, GroupSummary]]
    top_improvements: tuple[OutcomeExample, ...]
    top_failures: tuple[OutcomeExample, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "rewrite_mode": self.rewrite_mode,
            "overall": self.overall.as_dict(),
            "nontrivial": self.nontrivial.as_dict(),
            "identity_heavy": self.identity_heavy.as_dict(),
            "strata": {
                axis: {key: group.as_dict() for key, group in groups.items()}
                for axis, groups in self.strata.items()
            },
            "top_improvements": [example.as_dict() for example in self.top_improvements],
            "top_failures": [example.as_dict() for example in self.top_failures],
        }


@dataclass(frozen=True, slots=True)
class Goal4Summary:
    run_id: str | None
    total_rows: int
    total_expressions: int
    modes: Mapping[str, ModeReport]

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "total_rows": self.total_rows,
            "total_expressions": self.total_expressions,
            "modes": {mode: report.as_dict() for mode, report in self.modes.items()},
        }


def _is_nontrivial(row: ResultRow) -> bool:
    return bool(row.rewrites_applied)


def _is_identity_heavy(row: ResultRow) -> bool:
    if not row.improved or not row.applied_rules:
        return False
    markers = ("ZERO", "ONE", "INVERSE", "NEG", "FOLD")
    applied = [name for name, count in row.applied_rules.items() if count > 0]
    return bool(applied) and all(
        any(marker in name.upper() for marker in markers) for name in applied
    )


def _example(row: ResultRow) -> OutcomeExample:
    return OutcomeExample(
        expression_id=row.expression_id,
        stage_status=row.stage_status,
        absolute_improvement=row.absolute_improvement,
        failure_reason=row.failure_reason,
        applied_rules=tuple(sorted(row.applied_rules)),
    )


def summarize(
    rows: Iterable[Mapping[str, object]],
) -> Goal4Summary:
    """Build a complete paired-mode summary with both required denominators."""
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
                key: summarize_group(f"{axis}={key}", mode, grouped)
                for key, grouped in sorted(groups.items())
            }
        improvements = sorted(
            (row for row in mode_rows if row.improved),
            key=lambda row: (
                -(row.absolute_improvement or 0),
                row.expression_id,
            ),
        )
        failures = sorted(
            (row for row in mode_rows if row.stage_status not in _SUCCESS_STATUSES),
            key=lambda row: (row.stage_status, row.expression_id),
        )
        reports[mode] = ModeReport(
            rewrite_mode=mode,
            overall=summarize_group("overall", mode, mode_rows),
            nontrivial=summarize_group(
                "nontrivial",
                mode,
                [row for row in mode_rows if _is_nontrivial(row)],
            ),
            identity_heavy=summarize_group(
                "identity_heavy",
                mode,
                [row for row in mode_rows if _is_identity_heavy(row)],
            ),
            strata=strata,
            top_improvements=tuple(_example(row) for row in improvements[:10]),
            top_failures=tuple(_example(row) for row in failures[:10]),
        )
    return Goal4Summary(
        run_id=None if not parsed else parsed[0].run_id,
        total_rows=len(parsed),
        total_expressions=len({row.expression_id for row in parsed}),
        modes=reports,
    )


def write_analysis_artifacts(
    rows_path: str,
    output_dir: str,
) -> tuple[str, ...]:
    """Write the complete summary, failure audit, plot data, and six plots.

    Production rows are read-only inputs.  JSON publication is create-only and permits
    only a byte-identical resumed result.
    """
    from pathlib import Path

    from geml.analysis.goal4.failures import analyze_failures
    from geml.experiments.goal4.runtime import atomic_write_json, iter_jsonl
    from geml.plots.goal4 import build_plot_data, render_plots

    summary = summarize(iter_jsonl(rows_path))
    failures = analyze_failures(iter_jsonl(rows_path))
    plot_data = build_plot_data(iter_jsonl(rows_path))
    directory = Path(output_dir)
    paths = (
        atomic_write_json(directory / "summary.json", summary.as_dict()),
        atomic_write_json(directory / "failures.json", failures.as_dict()),
        atomic_write_json(directory / "plot_data.json", plot_data.as_dict()),
        *render_plots(plot_data, directory / "plots"),
    )
    return tuple(str(path) for path in paths)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a complete Goal 4 rows artifact.")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args(argv)
    for path in write_analysis_artifacts(
        arguments.rows,
        arguments.output_dir,
    ):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
