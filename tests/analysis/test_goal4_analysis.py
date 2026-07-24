"""Tiny strict fixtures for Goal 4 summaries, failures, and plots."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from geml.analysis.goal4.failures import analyze_failures
from geml.analysis.goal4.summary import (
    ExactRatio,
    SummaryError,
    parse_row,
    summarize,
)
from geml.experiments.goal4.run import ROW_SCHEMA_VERSION
from geml.plots.goal4 import build_plot_data, render_plots

_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
_RUN_ID = "f" * 64


def _row(
    expression_id: str,
    mode: str,
    status: str,
    *,
    before: int | None = None,
    after: int | None = None,
    family: str = "algebraic_core",
    size: str = "<= 8",
    difficulty: str = "ordinary",
    applied: dict[str, int] | None = None,
    saturation: str | None = "success",
    timeout: bool = False,
    reason: str | None = None,
    wall: float | None = 0.02,
    memory: int | None = None,
    validation_failures: dict[str, int] | None = None,
) -> dict:
    successful = status in {"optimized", "unchanged"}
    improvement = None if before is None or after is None else before - after
    provenance = (
        None
        if saturation is None
        else {
            "application_log_complete": True,
            "attempt_aggregates_complete": True,
            "individual_nonapplication_attempts_retained": False,
            "attempt_count": sum((applied or {}).values()),
            "attempt_digest_sha256": "a" * 64,
            "per_rule": [],
            "applications": [],
            "applied_rules": applied or {},
            "branch_sensitive_applications": (1 if applied and "DOMAIN" in "".join(applied) else 0),
            "assumptions_used": [],
        }
    )
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "run_id": _RUN_ID,
        "expression_id": expression_id,
        "rewrite_mode": mode,
        "domain_mode": "safe_real",
        "operator_family": family,
        "split": "train",
        "size_bucket": size,
        "difficulty_profile": difficulty,
        "rule_library": ("safe_real" if mode == "safe_real" else "safe_plus_domain"),
        "stage_status": status,
        "saturation_status": saturation,
        "extraction_status": (None if saturation is None else "success"),
        "validation_status": "valid" if successful else None,
        "timeout": timeout,
        "failure_stage": None if successful else "fixture_stage",
        "failure_reason": reason,
        "rewrites_applied": (None if saturation is None else sum((applied or {}).values())),
        "candidate_count": None if saturation is None else 2,
        "provenance": provenance,
        "validation_failures": validation_failures,
        "eml_dag_cost_before": before,
        "eml_dag_cost_after": after,
        "absolute_improvement": improvement,
        "resources": {
            "wall_seconds": wall,
            "cpu_seconds": wall,
            "rss_bytes_before": memory,
            "rss_bytes_after": memory,
        },
    }


def _fixture_rows() -> list[dict]:
    """Mode-balanced rows covering improvement, unchanged, and failures."""
    return [
        _row(
            "a",
            "safe_real",
            "optimized",
            before=30,
            after=20,
            applied={"SAFE-ADD-ZERO": 1},
            memory=100,
        ),
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
            "d",
            "safe_real",
            "no_candidate",
            before=14,
            reason="no valid candidate",
            size="<= 16",
            difficulty="stress",
            timeout=True,
            validation_failures={"inconclusive": 2},
        ),
        _row(
            "a",
            "positive_real_formal",
            "optimized",
            before=30,
            after=16,
            applied={"DOMAIN-LOG-EXP": 1},
            memory=100,
        ),
        _row(
            "b",
            "positive_real_formal",
            "unchanged",
            before=22,
            after=22,
        ),
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
            difficulty="stress",
            timeout=True,
            validation_failures={"inconclusive": 2},
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

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("timeout", "false", "boolean"),
            ("eml_dag_cost_before", True, "integer"),
            ("expression_id", None, "non-blank"),
        ],
    )
    def test_lax_coercions_are_rejected(self, field, value, message):
        row = dict(_fixture_rows()[0])
        row[field] = value
        with pytest.raises(SummaryError, match=message):
            parse_row(row)

    def test_inconsistent_improvement_is_rejected(self):
        row = dict(_fixture_rows()[0])
        row["absolute_improvement"] = 999
        with pytest.raises(SummaryError, match="does not match"):
            parse_row(row)

    def test_duplicate_units_are_rejected(self):
        rows = _fixture_rows()
        with pytest.raises(SummaryError, match="duplicate"):
            summarize([*rows, rows[0]])

    def test_unpaired_modes_are_rejected(self):
        with pytest.raises(SummaryError, match="exactly both"):
            summarize(_fixture_rows()[:-1])


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
        assert set(summary.modes) == {
            "safe_real",
            "positive_real_formal",
        }
        assert summary.total_expressions == 4

    def test_success_only_denominator_is_costed(self):
        safe = summarize(_fixture_rows()).modes["safe_real"].overall
        assert safe.costed_count == 2
        assert safe.improved_count == 1
        assert safe.improved_over_costed.as_dict()["exact"] == "1/2"

    def test_all_processed_after_rate_includes_failures(self):
        safe = summarize(_fixture_rows()).modes["safe_real"].overall
        assert safe.processed_count == 4
        assert safe.failure_count == 2
        assert safe.improved_over_processed.as_dict()["exact"] == "1/4"
        assert safe.costed_over_processed.as_dict()["exact"] == "1/2"

    def test_signed_and_positive_totals_are_explicit(self):
        formal = summarize(_fixture_rows()).modes["positive_real_formal"].overall
        assert formal.signed_total_improvement == 14
        assert formal.positive_total_improvement == 14

    def test_modes_are_not_averaged_together(self):
        summary = summarize(_fixture_rows())
        assert summary.modes["safe_real"].overall.signed_total_improvement == 10
        assert summary.modes["positive_real_formal"].overall.signed_total_improvement == 14

    def test_no_row_is_dropped(self):
        rows = _fixture_rows()
        summary = summarize(rows)
        assert sum(report.overall.processed_count for report in summary.modes.values()) == len(rows)


class TestStratificationAndExamples:
    def test_stratifies_by_family_and_difficulty(self):
        report = summarize(_fixture_rows()).modes["safe_real"]
        assert "trigonometric" in report.strata["operator_family"]
        assert "stress" in report.strata["difficulty_profile"]

    def test_nontrivial_and_identity_heavy_are_reported(self):
        report = summarize(_fixture_rows()).modes["safe_real"]
        assert report.nontrivial.processed_count == 1
        assert report.identity_heavy.improved_count == 1

    def test_top_success_and_failure_examples_are_retained(self):
        report = summarize(_fixture_rows()).modes["safe_real"]
        assert report.top_improvements[0].expression_id == "a"
        assert {item.expression_id for item in report.top_failures} == {
            "c",
            "d",
        }


class TestFailureAccounting:
    def test_every_failure_is_categorized(self):
        safe = analyze_failures(_fixture_rows()).modes["safe_real"]
        assert {category.category for category in safe.categories} == {
            "unsupported_operator",
            "no_candidate",
        }

    def test_failure_timeout_and_validation_counts_are_separate(self):
        safe = analyze_failures(_fixture_rows()).modes["safe_real"]
        assert safe.failure_count == 2
        assert safe.timeout_count == 1
        assert safe.validation_failure_count == 1

    def test_family_and_difficulty_bias_are_visible(self):
        safe = analyze_failures(_fixture_rows()).modes["safe_real"]
        family = {entry.stratum: entry for entry in safe.family_bias}
        assert family["trigonometric"].failure_rate_permille == 1000
        difficulty = {entry.stratum: entry for entry in safe.difficulty_bias}
        assert difficulty["stress"].failure_rate_permille == 1000

    def test_provenance_audit_examples_keep_reasons(self):
        safe = analyze_failures(_fixture_rows()).modes["safe_real"]
        assert any(example.failure_reason == "sin" for example in safe.provenance_audit_examples)


class TestPlots:
    def test_plot_data_keeps_modes_separate(self):
        data = build_plot_data(_fixture_rows())
        assert data.modes == ("safe_real", "positive_real_formal")
        assert set(data.success_rate.series) == set(data.modes)

    def test_success_series_uses_processed_costed_improved(self):
        data = build_plot_data(_fixture_rows())
        assert data.success_rate.edges == (
            "processed",
            "costed",
            "improved",
        )
        assert data.success_rate.series["safe_real"] == (4, 2, 1)

    def test_missing_runtime_is_not_binned_as_zero(self):
        rows = _fixture_rows()
        rows[0]["resources"]["wall_seconds"] = None
        data = build_plot_data(rows)
        assert sum(data.runtime_distribution.series["safe_real"]) == 3

    def test_memory_availability_uses_honest_rss_snapshot(self):
        data = build_plot_data(_fixture_rows())
        assert data.memory_availability["safe_real"] == {
            "present": 1,
            "absent": 3,
        }

    def test_failure_breakdown_matches_failure_report(self):
        categories = build_plot_data(_fixture_rows()).failure_breakdown.categories
        assert "no_candidate" in categories
        assert "unsupported_operator" in categories

    def test_plot_data_is_deterministic(self):
        rows = _fixture_rows()
        assert build_plot_data(rows).as_dict() == build_plot_data(rows).as_dict()

    @pytest.mark.skipif(
        not _HAS_MATPLOTLIB,
        reason="matplotlib is not installed",
    )
    def test_render_writes_all_six_png_files(self, tmp_path: Path):
        paths = render_plots(
            build_plot_data(_fixture_rows()),
            tmp_path / "plots",
        )
        assert len(paths) == 6
        assert all(path.is_file() and path.suffix == ".png" for path in paths)


class TestDeterminism:
    def test_summary_and_failure_report_are_order_independent(self):
        rows = _fixture_rows()
        assert summarize(rows).as_dict() == summarize(list(reversed(rows))).as_dict()
        assert analyze_failures(rows).as_dict() == analyze_failures(list(reversed(rows))).as_dict()
