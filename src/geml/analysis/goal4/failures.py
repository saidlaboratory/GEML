"""Failure, timeout, validation, and provenance-bias analysis for Goal 4."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from geml.analysis.goal4.summary import ResultRow, parse_rows

_TERMINAL_SUCCESS = frozenset({"optimized", "unchanged"})


@dataclass(frozen=True, slots=True)
class FailureCategory:
    category: str
    count: int
    share_of_failures_permille: int

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "count": self.count,
            "share_of_failures_permille": self.share_of_failures_permille,
        }


@dataclass(frozen=True, slots=True)
class BiasEntry:
    stratum: str
    failure_count: int
    processed_count: int
    failure_rate_permille: int

    def as_dict(self) -> dict[str, object]:
        return {
            "stratum": self.stratum,
            "failure_count": self.failure_count,
            "processed_count": self.processed_count,
            "failure_rate_permille": self.failure_rate_permille,
        }


@dataclass(frozen=True, slots=True)
class FailureAuditExample:
    expression_id: str
    stage_status: str
    failure_stage: str | None
    failure_reason: str | None
    timeout: bool
    validation_failure_count: int
    applied_rules: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "expression_id": self.expression_id,
            "stage_status": self.stage_status,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "timeout": self.timeout,
            "validation_failure_count": self.validation_failure_count,
            "applied_rules": list(self.applied_rules),
        }


@dataclass(frozen=True, slots=True)
class ModeFailureReport:
    rewrite_mode: str
    processed_count: int
    failure_count: int
    timeout_count: int
    validation_failure_count: int
    categories: tuple[FailureCategory, ...]
    family_bias: tuple[BiasEntry, ...]
    size_bias: tuple[BiasEntry, ...]
    difficulty_bias: tuple[BiasEntry, ...]
    timeout_family_bias: tuple[BiasEntry, ...]
    validation_family_bias: tuple[BiasEntry, ...]
    example_reasons: tuple[str, ...]
    provenance_audit_examples: tuple[FailureAuditExample, ...]

    @property
    def overall_failure_rate_permille(self) -> int:
        return (
            0 if self.processed_count == 0 else (self.failure_count * 1000) // self.processed_count
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "rewrite_mode": self.rewrite_mode,
            "processed_count": self.processed_count,
            "failure_count": self.failure_count,
            "timeout_count": self.timeout_count,
            "validation_failure_count": self.validation_failure_count,
            "overall_failure_rate_permille": self.overall_failure_rate_permille,
            "categories": [item.as_dict() for item in self.categories],
            "family_bias": [item.as_dict() for item in self.family_bias],
            "size_bias": [item.as_dict() for item in self.size_bias],
            "difficulty_bias": [item.as_dict() for item in self.difficulty_bias],
            "timeout_family_bias": [item.as_dict() for item in self.timeout_family_bias],
            "validation_family_bias": [item.as_dict() for item in self.validation_family_bias],
            "example_reasons": list(self.example_reasons),
            "provenance_audit_examples": [
                item.as_dict() for item in self.provenance_audit_examples
            ],
        }


@dataclass(frozen=True, slots=True)
class FailureReport:
    run_id: str | None
    total_rows: int
    modes: Mapping[str, ModeFailureReport]

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "total_rows": self.total_rows,
            "modes": {mode: report.as_dict() for mode, report in self.modes.items()},
        }


def _is_failure(row: ResultRow) -> bool:
    return row.stage_status not in _TERMINAL_SUCCESS


def _bias(
    affected: Sequence[ResultRow],
    processed: Sequence[ResultRow],
    attribute: str,
) -> tuple[BiasEntry, ...]:
    processed_counts = Counter(str(getattr(row, attribute)) for row in processed)
    affected_counts = Counter(str(getattr(row, attribute)) for row in affected)
    return tuple(
        BiasEntry(
            stratum=stratum,
            failure_count=affected_counts.get(stratum, 0),
            processed_count=processed_counts[stratum],
            failure_rate_permille=(affected_counts.get(stratum, 0) * 1000)
            // processed_counts[stratum],
        )
        for stratum in sorted(processed_counts)
    )


def _categories(
    failures: Sequence[ResultRow],
) -> tuple[FailureCategory, ...]:
    counts = Counter(row.stage_status for row in failures)
    total = sum(counts.values())
    return tuple(
        FailureCategory(
            category=category,
            count=count,
            share_of_failures_permille=(0 if total == 0 else (count * 1000) // total),
        )
        for category, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def _example_reasons(
    failures: Sequence[ResultRow],
    *,
    limit: int = 10,
) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for row in sorted(failures, key=lambda item: item.expression_id):
        if row.failure_reason:
            seen.setdefault(row.failure_reason, None)
        if len(seen) >= limit:
            break
    return tuple(seen)


def _audit_examples(
    failures: Sequence[ResultRow],
    *,
    limit: int = 10,
) -> tuple[FailureAuditExample, ...]:
    return tuple(
        FailureAuditExample(
            expression_id=row.expression_id,
            stage_status=row.stage_status,
            failure_stage=row.failure_stage,
            failure_reason=row.failure_reason,
            timeout=row.timeout,
            validation_failure_count=row.validation_failure_count,
            applied_rules=tuple(sorted(row.applied_rules)),
        )
        for row in sorted(
            failures,
            key=lambda item: (item.stage_status, item.expression_id),
        )[:limit]
    )


def analyze_failures(
    rows: Iterable[Mapping[str, object]],
) -> FailureReport:
    """Report failures and whether timeout/validation failures are stratum-biased."""
    return analyze_parsed_failures(parse_rows(rows))


def analyze_parsed_failures(
    parsed: Sequence[ResultRow],
) -> FailureReport:
    """Analyze an already validated, paired collection of compact result rows."""
    by_mode: dict[str, list[ResultRow]] = {}
    for row in parsed:
        by_mode.setdefault(row.rewrite_mode, []).append(row)

    reports: dict[str, ModeFailureReport] = {}
    for mode in sorted(by_mode):
        mode_rows = by_mode[mode]
        failures = [row for row in mode_rows if _is_failure(row)]
        timeouts = [row for row in mode_rows if row.timeout]
        validation_failures = [
            row
            for row in mode_rows
            if row.validation_failure_count > 0 or row.stage_status == "validation_failed"
        ]
        reports[mode] = ModeFailureReport(
            rewrite_mode=mode,
            processed_count=len(mode_rows),
            failure_count=len(failures),
            timeout_count=len(timeouts),
            validation_failure_count=len(validation_failures),
            categories=_categories(failures),
            family_bias=_bias(failures, mode_rows, "operator_family"),
            size_bias=_bias(failures, mode_rows, "size_bucket"),
            difficulty_bias=_bias(
                failures,
                mode_rows,
                "difficulty_profile",
            ),
            timeout_family_bias=_bias(
                timeouts,
                mode_rows,
                "operator_family",
            ),
            validation_family_bias=_bias(
                validation_failures,
                mode_rows,
                "operator_family",
            ),
            example_reasons=_example_reasons(failures),
            provenance_audit_examples=_audit_examples(failures),
        )
    return FailureReport(
        run_id=None if not parsed else parsed[0].run_id,
        total_rows=len(parsed),
        modes=reports,
    )
