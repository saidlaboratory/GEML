"""Issue 6-6 tests: aggregation, refusals, and the Gate G6 state machine on tiny fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from geml.analysis.goal6.summary import (
    FIXED_SCALE_CAVEATS,
    GOAL6_ANALYSIS_SCHEMA_VERSION,
    AnalysisError,
    GateState,
    MissingReason,
    MissingValue,
    aggregate,
    assert_not_pooled_across_representations,
    build_summary,
    cluster_bootstrap_interval,
    evaluate_gate_g6,
    paired_contrast,
    structural_metric_table,
    validate_ood_membership,
    validate_rows,
)
from geml.learning.harness.seeds import PRODUCTION_SEEDS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPOSITORY_ROOT / "docs" / "goals" / "goal6"

VIEW = "test_iid"
ARMS = ("graph::pure_eml_dag", "trivial_floor")


def make_row(
    arm_id: str,
    seed: int,
    accuracy: float | None = 0.8,
    status: str = "complete",
    commit: str = "998a139",
    structural: tuple[dict[str, object], ...] = (),
    denominators: dict[str, int] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "geml-goal6-grid-row-v1",
        "arm_id": arm_id,
        "seed": seed,
        "status": status,
        "commit": commit,
        "config_hash": "a" * 64,
        "structural_metrics": list(structural),
        "denominators_by_view": {
            VIEW: denominators
            or {"attempted": 25, "valid": 25, "failed": 0, "unsupported": 0, "timed_out": 0}
        },
    }
    if accuracy is not None:
        row["metrics_by_view"] = {VIEW: {"accuracy": accuracy}}
    return row


def complete_rows() -> list[dict[str, object]]:
    rows = []
    for index, seed in enumerate(PRODUCTION_SEEDS):
        rows.append(make_row(ARMS[0], seed, 0.80 + index * 0.02))
        rows.append(make_row(ARMS[1], seed, 0.60 + index * 0.01))
    return rows


def test_validate_rows_accepts_a_consistent_manifest() -> None:
    validate_rows(complete_rows(), PRODUCTION_SEEDS)


def test_duplicate_cell_identities_are_refused() -> None:
    rows = complete_rows()
    rows.append(make_row(ARMS[0], PRODUCTION_SEEDS[0], 0.99))
    with pytest.raises(AnalysisError, match="duplicate cell identities"):
        validate_rows(rows, PRODUCTION_SEEDS)


def test_mixed_commits_require_an_explicit_grouping_decision() -> None:
    rows = complete_rows()
    rows[0] = make_row(ARMS[0], PRODUCTION_SEEDS[0], 0.8, commit="deadbee")
    with pytest.raises(AnalysisError, match="multiple commits"):
        validate_rows(rows, PRODUCTION_SEEDS)
    validate_rows(rows, PRODUCTION_SEEDS, require_single_commit=False)


def test_unreconstructable_denominators_are_refused() -> None:
    rows = complete_rows()
    rows[0]["denominators_by_view"] = {VIEW: {"attempted": 25, "valid": 25}}
    with pytest.raises(AnalysisError, match="cannot reconstruct its denominator"):
        validate_rows(rows, PRODUCTION_SEEDS)


def test_denominators_that_do_not_sum_are_refused() -> None:
    rows = complete_rows()
    rows[0]["denominators_by_view"] = {
        VIEW: {"attempted": 25, "valid": 20, "failed": 0, "unsupported": 0, "timed_out": 0}
    }
    with pytest.raises(AnalysisError, match="do not sum to attempted"):
        validate_rows(rows, PRODUCTION_SEEDS)


def test_unexpected_seed_is_refused() -> None:
    rows = complete_rows()
    rows.append(make_row(ARMS[0], 19990101, 0.9))
    with pytest.raises(AnalysisError, match="unexpected seed"):
        validate_rows(rows, PRODUCTION_SEEDS)


def test_mismatched_ood_membership_is_refused() -> None:
    rows = [
        {**make_row(ARMS[0], PRODUCTION_SEEDS[0]), "ood_membership": {"test_depth_ood": ["a"]}},
        {
            **make_row(ARMS[1], PRODUCTION_SEEDS[0]),
            "ood_membership": {"test_depth_ood": ["a", "b"]},
        },
    ]
    with pytest.raises(AnalysisError, match="disagree about the membership"):
        validate_ood_membership(rows, "test_depth_ood")


def test_aggregate_keeps_raw_seeds_and_computes_spread() -> None:
    result = aggregate(complete_rows(), ARMS[0], VIEW, "accuracy", PRODUCTION_SEEDS)
    assert result.complete is True
    assert len(result.raw_by_seed) == 3
    assert result.mean == pytest.approx(0.82)
    assert result.spread == pytest.approx(0.04)
    payload = result.as_dict()
    assert set(payload["raw_by_seed"]) == {str(seed) for seed in PRODUCTION_SEEDS}


def test_missing_seed_is_never_treated_as_zero() -> None:
    rows = [row for row in complete_rows() if row["seed"] != PRODUCTION_SEEDS[2]]
    result = aggregate(rows, ARMS[0], VIEW, "accuracy", PRODUCTION_SEEDS)
    assert result.complete is False
    assert isinstance(result.mean, MissingValue)
    assert result.mean.reason is MissingReason.CELL_MISSING
    assert result.as_dict()["mean"]["missing"] is True


def test_failed_cell_contributes_no_value() -> None:
    rows = complete_rows()
    rows[0] = make_row(ARMS[0], PRODUCTION_SEEDS[0], accuracy=None, status="failed")
    result = aggregate(rows, ARMS[0], VIEW, "accuracy", PRODUCTION_SEEDS)
    assert PRODUCTION_SEEDS[0] not in result.raw_by_seed
    assert result.complete is False


def test_paired_contrast_pairs_within_seed() -> None:
    contrast = paired_contrast(
        complete_rows(), ARMS[0], ARMS[1], VIEW, "accuracy", PRODUCTION_SEEDS
    )
    assert set(contrast.per_seed_difference) == set(PRODUCTION_SEEDS)
    assert contrast.mean_difference == pytest.approx(0.21)
    assert isinstance(contrast.effect_size, float)
    assert "not an asymptotic significance test" in contrast.as_dict()["interpretation_note"]


def test_paired_contrast_reports_missing_when_no_shared_seeds() -> None:
    rows = [make_row(ARMS[0], PRODUCTION_SEEDS[0]), make_row(ARMS[1], PRODUCTION_SEEDS[1])]
    contrast = paired_contrast(rows, ARMS[0], ARMS[1], VIEW, "accuracy", PRODUCTION_SEEDS)
    assert isinstance(contrast.mean_difference, MissingValue)


def test_cluster_bootstrap_resamples_groups_not_rows() -> None:
    interval = cluster_bootstrap_interval(
        {"group-a": [0.1, 0.2], "group-b": [0.8, 0.9], "group-c": [0.5]},
        seed=PRODUCTION_SEEDS[0],
        iterations=200,
    )
    assert isinstance(interval, tuple)
    low, high = interval
    assert low <= high


def test_cluster_bootstrap_refuses_a_single_group() -> None:
    result = cluster_bootstrap_interval({"only": [0.5]}, seed=1, iterations=10)
    assert isinstance(result, MissingValue)


def test_structural_metrics_are_never_pooled_across_representations() -> None:
    rows = [
        {
            **make_row(ARMS[0], PRODUCTION_SEEDS[0]),
            "representation_mode": "eml:eml:official_v4:is_pure_eml=true",
            "structural_metrics": [
                {"name": "alpha", "value": 1.5, "comparable_across_channels": False}
            ],
        },
        {
            **make_row(ARMS[1], PRODUCTION_SEEDS[0]),
            "representation_mode": "macro:macro:official_v4:is_pure_eml=false",
            "structural_metrics": [
                {"name": "alpha", "value": 2.5, "comparable_across_channels": False}
            ],
        },
    ]
    table = structural_metric_table(rows)
    with pytest.raises(AnalysisError, match="flagged incomparable"):
        assert_not_pooled_across_representations(table)


def test_comparable_metrics_may_share_a_name() -> None:
    table = [
        {
            "name": "node_count",
            "value": 3,
            "comparable_across_channels": True,
            "representation_mode": "a",
        },
        {
            "name": "node_count",
            "value": 4,
            "comparable_across_channels": True,
            "representation_mode": "b",
        },
    ]
    assert_not_pooled_across_representations(table)


def test_gate_refuses_a_verdict_from_fixture_rows() -> None:
    verdict = evaluate_gate_g6(
        complete_rows(),
        PRODUCTION_SEEDS,
        ARMS,
        [(ARMS[0], ARMS[1], VIEW, "accuracy")],
        is_fixture=True,
    )
    assert verdict.state is GateState.INSUFFICIENT_EVIDENCE
    assert verdict.is_fixture is True
    assert "fixture" in verdict.rationale


def test_gate_reports_insufficient_evidence_for_missing_seeds() -> None:
    rows = [row for row in complete_rows() if row["seed"] != PRODUCTION_SEEDS[2]]
    verdict = evaluate_gate_g6(rows, PRODUCTION_SEEDS, ARMS, [(ARMS[0], ARMS[1], VIEW, "accuracy")])
    assert verdict.state is GateState.INSUFFICIENT_EVIDENCE
    assert any("missing seeds" in item for item in verdict.unmet_requirements)


def test_gate_reports_insufficient_evidence_for_a_failed_cell() -> None:
    rows = complete_rows()
    rows[0] = make_row(ARMS[0], PRODUCTION_SEEDS[0], accuracy=None, status="failed")
    verdict = evaluate_gate_g6(rows, PRODUCTION_SEEDS, ARMS, [(ARMS[0], ARMS[1], VIEW, "accuracy")])
    assert verdict.state is GateState.INSUFFICIENT_EVIDENCE
    assert any("failed cells" in item for item in verdict.unmet_requirements)


def test_gate_passes_on_a_decisive_predeclared_contrast() -> None:
    verdict = evaluate_gate_g6(
        complete_rows(), PRODUCTION_SEEDS, ARMS, [(ARMS[0], ARMS[1], VIEW, "accuracy")]
    )
    assert verdict.state is GateState.PASS
    assert verdict.supporting_contrasts
    assert verdict.caveats == FIXED_SCALE_CAVEATS


def test_gate_fails_plainly_on_a_null_result() -> None:
    """A null result is a first-class finding, not something to rerun away."""

    rows = []
    for index, seed in enumerate(PRODUCTION_SEEDS):
        rows.append(make_row(ARMS[0], seed, 0.70 + index * 0.01))
        rows.append(make_row(ARMS[1], seed, 0.70 + index * 0.01))
    verdict = evaluate_gate_g6(rows, PRODUCTION_SEEDS, ARMS, [(ARMS[0], ARMS[1], VIEW, "accuracy")])
    assert verdict.state is GateState.FAIL
    assert "null result" in verdict.rationale


def test_build_summary_lists_missing_and_failed_cells_explicitly() -> None:
    rows = complete_rows()
    rows[0] = make_row(ARMS[0], PRODUCTION_SEEDS[0], accuracy=None, status="failed")
    rows = [row for row in rows if row["seed"] != PRODUCTION_SEEDS[2]]

    summary = build_summary(rows, PRODUCTION_SEEDS, ARMS, [VIEW], ["accuracy"])
    assert summary["schema_version"] == GOAL6_ANALYSIS_SCHEMA_VERSION
    assert summary["expected_cell_count"] == 6
    assert len(summary["missing_cells"]) == 2
    assert len(summary["failed_cells"]) == 1
    assert summary["caveats"] == list(FIXED_SCALE_CAVEATS)


def test_empty_rows_are_refused() -> None:
    with pytest.raises(AnalysisError, match="no result rows"):
        validate_rows([], PRODUCTION_SEEDS)


def test_published_docs_declare_the_pending_state_without_inventing_numbers() -> None:
    gate = (DOCS / "GATE_G6.md").read_text(encoding="utf-8")
    summary = (DOCS / "GOAL6_SUMMARY.md").read_text(encoding="utf-8")
    assert "insufficient_evidence" in gate
    assert "verdict not yet issued" in gate
    assert "phase_a_implemented" in summary
    assert "no production run has happened" in summary
    # The measured fixture parameter counts are the only real numbers Phase A may publish.
    assert "185,732" in summary


def test_plots_refuse_to_pool_incomparable_structural_metrics(tmp_path: Path) -> None:
    plots = pytest.importorskip(
        "geml.plots.goal6", reason="matplotlib is required for the plotting module"
    )
    table = [
        {
            "name": "alpha",
            "value": 1.0,
            "comparable_across_channels": False,
            "representation_mode": "eml",
            "arm_id": "a",
        },
        {
            "name": "alpha",
            "value": 2.0,
            "comparable_across_channels": False,
            "representation_mode": "macro",
            "arm_id": "b",
        },
    ]
    with pytest.raises(AnalysisError, match="flagged incomparable"):
        plots.plot_structural_metrics(table, tmp_path / "structural.png")


def test_plot_renders_raw_seed_points(tmp_path: Path) -> None:
    plots = pytest.importorskip(
        "geml.plots.goal6", reason="matplotlib is required for the plotting module"
    )
    aggregates = [
        aggregate(complete_rows(), arm, VIEW, "accuracy", PRODUCTION_SEEDS).as_dict()
        for arm in ARMS
    ]
    output = plots.plot_arm_metric_by_seed(aggregates, VIEW, "accuracy", tmp_path / "seeds.png")
    assert output.exists()
    assert output.stat().st_size > 0
