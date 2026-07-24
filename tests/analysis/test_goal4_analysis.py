"""Tiny-fixture tests for the Goal 4 result analysis, failure accounting, and plots."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from geml.analysis.goal4.failures import analyze_failures
from geml.analysis.goal4.summary import ExactRatio, SummaryError, parse_row, summarize
from geml.plots.goal4 import build_plot_data, render_plots

_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _row(
    expression_id: str,
    mode: str,
    status: str,
    *,
    before: int | None = None,
    after: int | None = None,
    family: str = "algebraic_core",
    size: str = "<= 8",
    applied: dict[str, int] | None = None,
    saturation: str | None = "success",
    timeout: bool = False,
    reason: str | None = None,
    wall: float = 0.02,
    memory: int | None = None,
) -> dict:
    return {
        "expression_id": expression_id,
        "rewrite_mode": mode,
        "domain_mode": "safe_real",
        "operator_family": family,
        "split": "train",
        "size_bucket": size,
        "rule_library": "safe_real" if mode == "safe_real" else "safe_plus_domain",
        "stage_status": status,
        "saturation_status": saturation,
        "extraction_status": "success",
        "validation_status": "valid",
        "timeout": timeout,
        "failure_reason": reason,
        "rewrites_applied": 1 if applied else 0,
        "candidate_count": 2,
        "provenance": {
            "applied_rules": applied or {},
            "branch_sensitive_applications": 1 if applied and "DOMAIN" in "".join(applied) else 0,
            "guard_outcomes": {},
            "assumptions_used": [],
        },
        "eml_dag_cost_before": before,
        "eml_dag_cost_after": after,
        "resources": {"wall_seconds": wall, "cpu_seconds": wall, "peak_memory_bytes": memory},
    }


def _fixture_rows() -> list[dict]:
    """Return a tiny, fixed, mode-balanced result set covering success and failure."""
    return [
        _row("a", "safe_real", "optimized", before=30, after=20, applied={"SAFE-ADD-ZERO": 1}),
        _row("b", "safe_real", "unchanged", before=22, after=22),
        _row(
            "c",
            "safe_real",
            "unsupported_operator",
            family="trigonometric",
            reason="sin",
            saturation=None,
        ),
        _row(
            "d", "safe_real", "no_candidate", before=14, reason="no valid candidate", size="<= 16"
        ),
        _row(
            "a",
            "positive_real_formal",
            "optimized",
            before=30,
            after=16,
            applied={"DOMAIN-LOG-EXP": 1},
        ),
        _row("b", "positive_real_formal", "unchanged", before=22, after=22),
        _row(
            "c",
            "positive_real_formal",
            "unsupported_operator",
            family="trigonometric",
            reason="sin",
            saturation=None,
        ),
        _row(
            "d",
            "positive_real_formal",
            "no_candidate",
            before=14,
            reason="no valid candidate",
            size="<= 16",
        ),
    ]


class TestRowParsing:
    def test_missing_required_field_is_explicit(self):
        with pytest.raises(SummaryError, match="required field"):
            parse_row({"rewrite_mode": "safe_real"})

    def test_uncosted_row_parses(self):
        parsed = parse_row(_fixture_rows()[2])
        assert parsed.costed is False
        assert parsed.absolute_improvement is None

    def test_costed_row_reports_improvement(self):
        parsed = parse_row(_fixture_rows()[0])
        assert parsed.costed
        assert parsed.improved
        assert parsed.absolute_improvement == 10


class TestExactRatio:
    def test_reduces_fraction(self):
        assert ExactRatio.of(2, 4).as_dict()["exact"] == "1/2"

    def test_zero_denominator_is_sentinel(self):
        ratio = ExactRatio.of(3, 0)
        assert ratio.value is None
        assert ratio.as_dict()["exact"] == "0/0"


class TestSummaryDenominators:
    def test_modes_are_separated(self):
        summary = summarize(_fixture_rows())
        assert set(summary.modes) == {"safe_real", "positive_real_formal"}

    def test_success_only_denominator_is_costed(self):
        summary = summarize(_fixture_rows())
        safe = summary.modes["safe_real"].overall
        assert safe.costed_count == 2
        assert safe.improved_count == 1
        assert safe.success_rate.as_dict()["exact"] == "1/2"

    def test_all_processed_denominator_includes_failures(self):
        summary = summarize(_fixture_rows())
        safe = summary.modes["safe_real"].overall
        assert safe.processed_count == 4
        assert safe.failure_count == 2
        assert safe.processing_success_rate.as_dict()["exact"] == "1/2"

    def test_total_absolute_improvement_is_exact(self):
        summary = summarize(_fixture_rows())
        assert summary.modes["positive_real_formal"].overall.total_absolute_improvement == 14

    def test_modes_are_not_averaged_together(self):
        summary = summarize(_fixture_rows())
        safe = summary.modes["safe_real"].overall.total_absolute_improvement
        formal = summary.modes["positive_real_formal"].overall.total_absolute_improvement
        assert safe == 10
        assert formal == 14

    def test_no_row_is_dropped_from_processed(self):
        rows = _fixture_rows()
        summary = summarize(rows)
        processed = sum(report.overall.processed_count for report in summary.modes.values())
        assert processed == len(rows)


class TestStratification:
    def test_stratifies_by_operator_family(self):
        summary = summarize(_fixture_rows())
        strata = summary.modes["safe_real"].strata["operator_family"]
        assert "algebraic_core" in strata
        assert "trigonometric" in strata

    def test_stratum_failures_are_visible(self):
        summary = summarize(_fixture_rows())
        trig = summary.modes["safe_real"].strata["operator_family"]["trigonometric"]
        assert trig.failure_count == 1
        assert trig.costed_count == 0

    def test_nontrivial_subgroup_excludes_no_rewrites(self):
        summary = summarize(_fixture_rows())
        nontrivial = summary.modes["safe_real"].nontrivial
        assert nontrivial.processed_count == 1

    def test_identity_heavy_subgroup_is_reported(self):
        summary = summarize(_fixture_rows())
        assert summary.modes["safe_real"].identity_heavy.improved_count == 1


class TestFailureAccounting:
    def test_every_failure_is_categorized(self):
        report = analyze_failures(_fixture_rows())
        safe = report.modes["safe_real"]
        categories = {category.category for category in safe.categories}
        assert categories == {"unsupported_operator", "no_candidate"}

    def test_failure_counts_match_processed(self):
        report = analyze_failures(_fixture_rows())
        safe = report.modes["safe_real"]
        assert safe.processed_count == 4
        assert safe.failure_count == 2

    def test_family_bias_exposes_trig_failures(self):
        report = analyze_failures(_fixture_rows())
        bias = {entry.stratum: entry for entry in report.modes["safe_real"].family_bias}
        assert bias["trigonometric"].failure_rate_permille == 1000
        assert bias["trigonometric"].failure_count == 1
        assert bias["algebraic_core"].failure_count == 1
        assert bias["algebraic_core"].processed_count == 3

    def test_example_reasons_are_retained(self):
        report = analyze_failures(_fixture_rows())
        reasons = report.modes["safe_real"].example_reasons
        assert any("sin" in reason for reason in reasons)

    def test_modes_are_separated_in_failures(self):
        report = analyze_failures(_fixture_rows())
        assert set(report.modes) == {"safe_real", "positive_real_formal"}


class TestPlots:
    def test_plot_data_keeps_modes_separate(self):
        data = build_plot_data(_fixture_rows())
        assert data.modes == ("safe_real", "positive_real_formal")
        assert set(data.success_rate.series) == {"safe_real", "positive_real_formal"}

    def test_success_series_uses_processed_costed_improved(self):
        data = build_plot_data(_fixture_rows())
        assert data.success_rate.edges == ("processed", "costed", "improved")
        assert data.success_rate.series["safe_real"] == (4, 2, 1)

    def test_failure_breakdown_matches_failure_report(self):
        data = build_plot_data(_fixture_rows())
        assert "no_candidate" in data.failure_breakdown.categories
        assert "unsupported_operator" in data.failure_breakdown.categories

    def test_plot_data_is_deterministic(self):
        rows = _fixture_rows()
        assert build_plot_data(rows).as_dict() == build_plot_data(rows).as_dict()

    @pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib is not installed")
    def test_render_writes_png_files(self, tmp_path: Path):
        data = build_plot_data(_fixture_rows())
        paths = render_plots(data, tmp_path / "plots")
        assert paths
        assert all(path.is_file() and path.suffix == ".png" for path in paths)


class TestDeterminism:
    def test_summary_is_deterministic(self):
        rows = _fixture_rows()
        assert summarize(rows).as_dict() == summarize(rows).as_dict()

    def test_failure_report_is_deterministic(self):
        rows = _fixture_rows()
        assert analyze_failures(rows).as_dict() == analyze_failures(rows).as_dict()

    def test_summary_is_order_independent(self):
        rows = _fixture_rows()
        forward = summarize(rows).as_dict()
        backward = summarize(list(reversed(rows))).as_dict()
        assert forward == backward
