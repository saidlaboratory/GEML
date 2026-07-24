"""Failure accounting for Goal 4 results.

Every retained failure row is kept and classified; no failed row is removed from the
analysis.  For each rewrite mode this module reports the frequency of each failure category,
the operator-family and size bias of those failures, and their impact on the costed
denominator.  The two rewrite modes are reported separately and never merged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from geml.analysis.goal4.summary import ResultRow, parse_rows

_TERMINAL_SUCCESS = frozenset({"optimized", "unchanged"})


@dataclass(frozen=True, slots=True)
class FailureCategory:
    """Frequency and share of one failure category within a rewrite mode."""

    category: str
    count: int
    share_of_failures_permille: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "category": self.category,
            "count": self.count,
            "share_of_failures_permille": self.share_of_failures_permille,
        }


@dataclass(frozen=True, slots=True)
class BiasEntry:
    """One stratum's contribution to a mode's failures, with its within-stratum rate."""

    stratum: str
    failure_count: int
    processed_count: int
    failure_rate_permille: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "stratum": self.stratum,
            "failure_count": self.failure_count,
            "processed_count": self.processed_count,
            "failure_rate_permille": self.failure_rate_permille,
        }


@dataclass(frozen=True, slots=True)
class ModeFailureReport:
    """The complete failure account for one rewrite mode."""

    rewrite_mode: str
    processed_count: int
    failure_count: int
    timeout_count: int
    categories: tuple[FailureCategory, ...]
    family_bias: tuple[BiasEntry, ...]
    size_bias: tuple[BiasEntry, ...]
    example_reasons: tuple[str, ...]

    @property
    def overall_failure_rate_permille(self) -> int:
        """Return the per-mille share of processed rows that failed."""
        if self.processed_count == 0:
            return 0
        return (self.failure_count * 1000) // self.processed_count

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "rewrite_mode": self.rewrite_mode,
            "processed_count": self.processed_count,
            "failure_count": self.failure_count,
            "timeout_count": self.timeout_count,
            "overall_failure_rate_permille": self.overall_failure_rate_permille,
            "categories": [category.as_dict() for category in self.categories],
            "family_bias": [entry.as_dict() for entry in self.family_bias],
            "size_bias": [entry.as_dict() for entry in self.size_bias],
            "example_reasons": list(self.example_reasons),
        }


@dataclass(frozen=True, slots=True)
class FailureReport:
    """Mode-separated failure accounting over an entire result set."""

    total_rows: int
    modes: Mapping[str, ModeFailureReport]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "total_rows": self.total_rows,
            "modes": {mode: report.as_dict() for mode, report in self.modes.items()},
        }


def _is_failure(row: ResultRow) -> bool:
    """Return whether a row is a retained failure rather than a costed outcome."""
    return row.stage_status not in _TERMINAL_SUCCESS


def _bias(
    failures: Sequence[ResultRow],
    processed: Sequence[ResultRow],
    attribute: str,
) -> tuple[BiasEntry, ...]:
    """Return per-stratum failure counts and within-stratum failure rates."""
    processed_counts = Counter(str(getattr(row, attribute)) for row in processed)
    failure_counts = Counter(str(getattr(row, attribute)) for row in failures)
    entries = []
    for stratum in sorted(processed_counts):
        processed_total = processed_counts[stratum]
        failed = failure_counts.get(stratum, 0)
        rate = (failed * 1000) // processed_total if processed_total else 0
        entries.append(
            BiasEntry(
                stratum=stratum,
                failure_count=failed,
                processed_count=processed_total,
                failure_rate_permille=rate,
            )
        )
    return tuple(entries)


def _categories(failures: Sequence[ResultRow]) -> tuple[FailureCategory, ...]:
    """Return failure categories ordered by descending count then name."""
    counts = Counter(row.stage_status for row in failures)
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        FailureCategory(
            category=category,
            count=count,
            share_of_failures_permille=(count * 1000) // total if total else 0,
        )
        for category, count in ordered
    )


def _example_reasons(failures: Sequence[ResultRow], *, limit: int = 5) -> tuple[str, ...]:
    """Return a stable, deduplicated sample of failure reasons."""
    seen: dict[str, None] = {}
    for row in sorted(failures, key=lambda item: item.expression_id):
        if row.failure_reason:
            seen.setdefault(row.failure_reason, None)
        if len(seen) >= limit:
            break
    return tuple(seen)


def analyze_failures(rows: Iterable[Mapping[str, object]]) -> FailureReport:
    """Build the mode-separated failure report from raw JSON rows.

    Failures are counted per category and attributed to operator-family and size strata so
    any bias is visible.  No row is dropped; costed rows remain in the processed denominator
    that every failure rate is measured against.
    """
    parsed = parse_rows(rows)
    by_mode: dict[str, list[ResultRow]] = {}
    for row in parsed:
        by_mode.setdefault(row.rewrite_mode, []).append(row)

    reports: dict[str, ModeFailureReport] = {}
    for mode in sorted(by_mode):
        mode_rows = by_mode[mode]
        failures = [row for row in mode_rows if _is_failure(row)]
        reports[mode] = ModeFailureReport(
            rewrite_mode=mode,
            processed_count=len(mode_rows),
            failure_count=len(failures),
            timeout_count=sum(1 for row in mode_rows if row.timeout),
            categories=_categories(failures),
            family_bias=_bias(failures, mode_rows, "operator_family"),
            size_bias=_bias(failures, mode_rows, "size_bucket"),
            example_reasons=_example_reasons(failures),
        )
    return FailureReport(total_rows=len(parsed), modes=reports)
