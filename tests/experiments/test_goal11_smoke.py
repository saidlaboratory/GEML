"""Tiny fixture tests for Goal 11 synthesis and Gate G11."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from geml.analysis.goal11.final_eval import (
    ControlledTrack,
    ExternalReference,
    GateG11Criteria,
    GateG11Status,
    SeedValue,
    TraceableMetric,
    TrackEvidence,
    TrackOutcome,
    build_goal11_synthesis,
    evaluate_gate_g11,
    render_gate_g11_markdown,
    render_goal11_summary_markdown,
)
from geml.analysis.goal11.scaling import (
    FixedScaleResult,
    MetricDirection,
)
from geml.experiments.goal11.corpus_v3 import (
    ArtifactFormat,
    ArtifactReference,
    CompletenessState,
    WorkshopManifest,
    audit_workshop_manifest,
    canonical_json_bytes,
    manifest_sha256,
)
from geml.experiments.goal11.run_final_eval import (
    Goal11FinalRunError,
    build_authenticated_synthesis,
    load_final_config,
    run_final_eval,
)
from geml.plots.goal11_final import build_plot_data, render_plots

_SEEDS = (20260726, 20260727, 20260728)


def _metric(metric_id: str = "accuracy") -> TraceableMetric:
    return TraceableMetric(
        metric_id=metric_id,
        evaluation_view="frozen_test",
        ood_axis=None,
        unit="fraction",
        direction=MetricDirection.HIGHER_IS_BETTER,
        estimate=0.8,
        ci_low=0.7,
        ci_high=0.9,
        attempted_count=10,
        valid_count=8,
        failed_count=1,
        invalid_count=0,
        unsupported_count=0,
        timeout_count=1,
        seed_values=tuple(SeedValue(seed=seed, value=0.8) for seed in _SEEDS),
        requires_three_seeds=True,
        source_artifact_id="results",
        source_sha256="a" * 64,
        source_locator=f"/rows/{metric_id}",
    )


def _tracks(
    outcomes: tuple[TrackOutcome, TrackOutcome, TrackOutcome],
) -> tuple[TrackEvidence, ...]:
    return tuple(
        TrackEvidence(
            track=track,
            outcome=outcome,
            metrics=() if outcome is TrackOutcome.INSUFFICIENT else (_metric(track.value),),
            rationale=f"fixture {outcome.value}",
            material_contradiction=outcome is TrackOutcome.CONTRADICTORY,
            decision_rule_digest="d" * 64,
            source_artifact_id=(None if outcome is TrackOutcome.INSUFFICIENT else "results"),
            source_sha256=(None if outcome is TrackOutcome.INSUFFICIENT else "a" * 64),
            source_locator=(
                None if outcome is TrackOutcome.INSUFFICIENT else f"/outcomes/{track.value}"
            ),
        )
        for track, outcome in zip(ControlledTrack, outcomes, strict=True)
    )


def _criteria(*, frozen: bool, minimum: int = 3) -> GateG11Criteria:
    return GateG11Criteria(
        expected_seeds=_SEEDS,
        required_tracks=tuple(ControlledTrack),
        minimum_supporting_tracks=minimum,
        minimum_fixed_scale_panels=1,
        allow_material_contradiction=False,
        production_threshold_frozen=frozen,
        decision_rule_digest="d" * 64 if frozen else None,
        decision_rule_artifact_id="rules" if frozen else None,
        decision_rule_source_sha256="e" * 64 if frozen else None,
        decision_rule_source_locator="/rules/g11" if frozen else None,
    )


def _empty_fixed_scale() -> FixedScaleResult:
    return FixedScaleResult(
        analysis_config_sha256="1" * 64,
        observations_sha256="2" * 64,
        bootstrap_seed=7,
        bootstrap_replicates=100,
        panels=(),
        unassigned_row_ids=(),
    )


def test_phase_a_threshold_forces_insufficient_even_with_positive_fixtures():
    gate = evaluate_gate_g11(
        _tracks((TrackOutcome.POSITIVE,) * 3),
        _criteria(frozen=False),
        complete_fixed_scale_panel_count=1,
    )

    assert gate.status is GateG11Status.INSUFFICIENT_EVIDENCE
    assert "not frozen" in gate.reasons[0]


def test_frozen_boolean_without_decision_rules_cannot_pass():
    criteria = _criteria(frozen=False).model_copy(update={"production_threshold_frozen": True})
    gate = evaluate_gate_g11(
        _tracks((TrackOutcome.POSITIVE,) * 3),
        criteria,
        complete_fixed_scale_panel_count=1,
    )

    assert gate.status is GateG11Status.INSUFFICIENT_EVIDENCE
    assert any("decision rules" in reason for reason in gate.reasons)


def test_frozen_but_unauthenticated_decision_rules_cannot_pass():
    gate = evaluate_gate_g11(
        _tracks((TrackOutcome.POSITIVE,) * 3),
        _criteria(frozen=True),
        complete_fixed_scale_panel_count=1,
    )

    assert gate.status is GateG11Status.INSUFFICIENT_EVIDENCE
    assert any("source-authenticated" in reason for reason in gate.reasons)


def test_complete_frozen_fixture_exercises_pass_and_fail():
    passed = evaluate_gate_g11(
        _tracks((TrackOutcome.POSITIVE,) * 3),
        _criteria(frozen=True),
        complete_fixed_scale_panel_count=1,
        decision_rules_authenticated=True,
        producer_gates_authenticated=True,
    )
    failed = evaluate_gate_g11(
        _tracks((TrackOutcome.POSITIVE, TrackOutcome.NULL, TrackOutcome.NEGATIVE)),
        _criteria(frozen=True),
        complete_fixed_scale_panel_count=1,
        decision_rules_authenticated=True,
        producer_gates_authenticated=True,
    )

    assert passed.status is GateG11Status.PASS
    assert failed.status is GateG11Status.FAIL


def test_missing_track_or_bad_seed_set_is_insufficient():
    tracks = _tracks((TrackOutcome.POSITIVE,) * 3)
    missing = evaluate_gate_g11(
        tracks[:-1],
        _criteria(frozen=True),
        complete_fixed_scale_panel_count=1,
    )
    bad_metric = (
        tracks[0]
        .metrics[0]
        .model_copy(
            update={
                "seed_values": (
                    SeedValue(seed=_SEEDS[0], value=0.8),
                    SeedValue(seed=_SEEDS[1], value=0.8),
                    SeedValue(seed=99, value=0.8),
                )
            }
        )
    )
    bad_tracks = (
        tracks[0].model_copy(update={"metrics": (bad_metric,)}),
        *tracks[1:],
    )
    bad_seed = evaluate_gate_g11(
        bad_tracks,
        _criteria(frozen=True),
        complete_fixed_scale_panel_count=1,
    )

    assert missing.status is GateG11Status.INSUFFICIENT_EVIDENCE
    assert bad_seed.status is GateG11Status.INSUFFICIENT_EVIDENCE


def test_complete_contradiction_fails_under_frozen_policy():
    gate = evaluate_gate_g11(
        _tracks(
            (
                TrackOutcome.POSITIVE,
                TrackOutcome.CONTRADICTORY,
                TrackOutcome.POSITIVE,
            )
        ),
        _criteria(frozen=True, minimum=2),
        complete_fixed_scale_panel_count=1,
        decision_rules_authenticated=True,
        producer_gates_authenticated=True,
    )

    assert gate.status is GateG11Status.FAIL
    assert gate.contradictory_tracks == (ControlledTrack.REWRITE_PROOF_SIMPLIFICATION,)


def test_missing_fixed_scale_analysis_is_insufficient():
    gate = evaluate_gate_g11(
        _tracks((TrackOutcome.POSITIVE,) * 3),
        _criteria(frozen=True),
        complete_fixed_scale_panel_count=0,
    )

    assert gate.status is GateG11Status.INSUFFICIENT_EVIDENCE
    assert any("fixed-scale efficiency" in reason for reason in gate.reasons)


def test_external_llm_rows_are_optional_and_do_not_change_gate():
    tracks = _tracks((TrackOutcome.POSITIVE,) * 3)
    fixed = _empty_fixed_scale()
    external = ExternalReference(
        model_id="external-model",
        task_id="proof",
        attempted_count=2,
        valid_count=1,
        failed_count=1,
        invalid_count=0,
        unsupported_count=0,
        timeout_count=0,
        source_artifact_id="llm",
        source_sha256="b" * 64,
        source_locator="rows/0",
    )

    without = build_goal11_synthesis(tracks, fixed, _criteria(frozen=True))
    with_external = build_goal11_synthesis(
        tracks,
        fixed,
        _criteria(frozen=True),
        external_references=(external,),
    )

    assert without.gate == with_external.gate
    assert with_external.external_references[0].controlled is False
    assert all(not point.external for point in build_plot_data(with_external))


def test_metric_requires_complete_denominators_and_source_trace():
    with pytest.raises(ValidationError, match="account for every attempt"):
        _metric().model_copy(
            update={"attempted_count": 11},
            deep=True,
        ).model_validate(_metric().model_dump() | {"attempted_count": 11})

    with pytest.raises(ValidationError, match="source_locator"):
        TraceableMetric(
            **(_metric().model_dump() | {"source_locator": ""}),
        )


def test_cross_track_synthesis_has_no_scalar_leaderboard():
    synthesis = build_goal11_synthesis(
        _tracks((TrackOutcome.POSITIVE,) * 3),
        _empty_fixed_scale(),
        _criteria(frozen=False),
    )

    assert not hasattr(synthesis, "aggregate_score")
    assert "No 10-100x" in synthesis.boundaries[-1]


def test_fixture_summary_gate_and_plots_rebuild(tmp_path: Path):
    synthesis = build_goal11_synthesis(
        _tracks((TrackOutcome.POSITIVE,) * 3),
        _empty_fixed_scale(),
        _criteria(frozen=False),
    )
    summary = render_goal11_summary_markdown(synthesis)
    gate = render_gate_g11_markdown(synthesis.gate)
    plots = render_plots(synthesis, tmp_path / "plots")

    assert "External non-controlled reference panel" in summary
    assert "interval=[" in summary
    assert str(_SEEDS[0]) in summary
    assert "insufficient_evidence" in gate
    assert plots
    assert all(path.is_file() for path in plots)


def test_checked_in_config_is_explicitly_production_pending():
    root = Path(__file__).resolve().parents[2]
    payload = load_final_config(root / "configs" / "goal11_final.yaml")

    assert payload["gate_criteria"]["production_threshold_frozen"] is False

    with pytest.raises(Goal11FinalRunError, match="production inputs are not frozen"):
        run_final_eval(root / "configs" / "goal11_final.yaml")


def test_final_eval_runner_rebuilds_authenticated_fixture_outputs(tmp_path: Path):
    import hashlib
    import json

    tracks = _tracks((TrackOutcome.POSITIVE,) * 3)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "fixture-v1",
                "rows": {
                    metric.metric_id: metric.evidence_projection()
                    for track in tracks
                    for metric in track.metrics
                },
                "outcomes": {track.track.value: track.evidence_projection() for track in tracks},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    tracks = tuple(
        track.model_copy(
            update={
                "source_sha256": source_sha,
                "metrics": tuple(
                    metric.model_copy(update={"source_sha256": source_sha})
                    for metric in track.metrics
                ),
            }
        )
        for track in tracks
    )
    manifest = WorkshopManifest(
        config_sha256="c" * 64,
        expected_seeds=_SEEDS,
        artifacts=(
            ArtifactReference(
                artifact_id="results",
                producer_issue="fixture",
                category="result_table",
                roles=("gate_g11",),
                required=True,
                relative_path="source.json",
                artifact_format=ArtifactFormat.JSON,
                state=CompletenessState.COMPLETE,
                observed_sha256=source_sha,
                size_bytes=len(source.read_bytes()),
                observed_schema_version="fixture-v1",
            ),
        ),
        deferred_experiments=(),
    )
    audit = audit_workshop_manifest(manifest, tmp_path)
    manifest_path = tmp_path / "manifest.json"
    audit_path = tmp_path / "audit.json"
    fixed_path = tmp_path / "fixed.json"
    fixed_run_config_path = tmp_path / "fixed-run-config.yaml"
    fixed_observations_path = tmp_path / "fixed-observations.jsonl"
    tracks_path = tmp_path / "tracks.jsonl"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    audit_path.write_bytes(canonical_json_bytes(audit))
    audit_sha = hashlib.sha256(canonical_json_bytes(audit)).hexdigest()
    fixed_run_config_path.write_text("schema_version: fixture\n", encoding="utf-8")
    fixed_observations_path.write_text('{"fixture":true}\n', encoding="utf-8")
    fixed_path.write_text(
        _empty_fixed_scale()
        .model_copy(
            update={
                "manifest_sha256": manifest_sha256(manifest),
                "manifest_audit_sha256": audit_sha,
                "observations_file_sha256": hashlib.sha256(
                    fixed_observations_path.read_bytes()
                ).hexdigest(),
                "run_config_sha256": hashlib.sha256(fixed_run_config_path.read_bytes()).hexdigest(),
            }
        )
        .model_dump_json(),
        encoding="utf-8",
    )
    tracks_path.write_text(
        "".join(track.model_dump_json() + "\n" for track in tracks),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "geml-goal11-final-run-config-v1",
                "artifact_root": str(tmp_path),
                "manifest_path": str(manifest_path),
                "manifest_audit_path": str(audit_path),
                "fixed_scale_result_path": str(fixed_path),
                "fixed_scale_run_config_path": str(fixed_run_config_path),
                "fixed_scale_observations_path": str(fixed_observations_path),
                "track_evidence_path": str(tracks_path),
                "external_llm_path": None,
                "result_path": str(tmp_path / "out" / "result.json"),
                "summary_path": str(tmp_path / "out" / "summary.md"),
                "gate_path": str(tmp_path / "out" / "gate.md"),
                "plot_dir": str(tmp_path / "out" / "plots"),
                "gate_criteria": _criteria(frozen=False).model_dump(mode="json"),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    synthesis = run_final_eval(config_path)

    assert synthesis.gate.status is GateG11Status.INSUFFICIENT_EVIDENCE
    assert (tmp_path / "out" / "result.json").is_file()
    assert (tmp_path / "out" / "summary.md").is_file()
    assert (tmp_path / "out" / "gate.md").is_file()
    assert tuple((tmp_path / "out" / "plots").glob("*.png"))
    first_track = tracks[0]
    first_metric = first_track.metrics[0]
    tampered_metric = first_metric.model_copy(update={"estimate": 0.123})
    tampered_tracks = (
        first_track.model_copy(update={"metrics": (tampered_metric,)}),
        *tracks[1:],
    )
    fixed = FixedScaleResult.model_validate_json(fixed_path.read_text(encoding="utf-8"))
    with pytest.raises(Goal11FinalRunError, match="source/row"):
        build_authenticated_synthesis(
            manifest,
            audit,
            fixed,
            tampered_tracks,
            _criteria(frozen=False),
            tmp_path,
        )
