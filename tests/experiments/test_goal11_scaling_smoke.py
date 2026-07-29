"""Tiny fixture tests for fixed-scale Goal 11 efficiency analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from geml.analysis.goal11.scaling import (
    ComparisonSpec,
    EfficiencyObservation,
    FixedScaleAnalysisConfig,
    FixedScaleAnalysisError,
    MeasurementMethod,
    MetricAvailability,
    MetricDirection,
    ObservationStatus,
    OutcomeCounts,
    QualityObservation,
    ResourceObservation,
    Track,
    render_fixed_scale_markdown,
    summarize_fixed_scale,
)
from geml.experiments.goal11.corpus_v3 import (
    ArtifactFormat,
    ArtifactReference,
    CompletenessState,
    WorkshopManifest,
    audit_workshop_manifest,
    canonical_json_bytes,
)
from geml.experiments.goal11.run_scaling import (
    FixedScaleRunError,
    analyze_authenticated_rows,
    load_fixed_scale_config,
    run_fixed_scale,
)
from geml.plots.goal11_scaling import build_plot_data, render_plots

_SEEDS = (20260726, 20260727, 20260728)


def _quality(value: float = 0.8, *, attempted: int = 1) -> QualityObservation:
    return QualityObservation(
        metric_id="accuracy",
        unit="fraction",
        direction=MetricDirection.HIGHER_IS_BETTER,
        availability=MetricAvailability.AVAILABLE,
        value=value,
        attempted_count=attempted,
        valid_count=attempted,
        failed_count=0,
        invalid_count=0,
        unsupported_count=0,
        timeout_count=0,
    )


def _resource(
    metric_id: str,
    value: float | None,
    *,
    method: MeasurementMethod = MeasurementMethod.MEASURED,
) -> ResourceObservation:
    return ResourceObservation(
        metric_id=metric_id,
        unit="count" if metric_id == "flops" else "seconds",
        availability=(
            MetricAvailability.AVAILABLE if value is not None else MetricAvailability.UNAVAILABLE
        ),
        value=value,
        method=method if value is not None else None,
        reason=None if value is not None else "telemetry was not published",
    )


def _row(
    method: str,
    seed: int,
    group: str,
    *,
    quality: float = 0.8,
    flops: float | None = 10.0,
    wall: float | None = 2.0,
    protocol: str = "a",
    hardware: str = "b",
    status: ObservationStatus = ObservationStatus.COMPLETE,
    config_digest: str = "3",
    resource_method: MeasurementMethod = MeasurementMethod.MEASURED,
) -> EfficiencyObservation:
    suffix = hashlib_sha(f"{method}:{seed}:{group}:{protocol}")
    return EfficiencyObservation(
        row_id=suffix,
        source_artifact_id="results",
        source_sha256="1" * 64,
        source_locator=f"/rows/{suffix}",
        track=Track.EQUIVALENCE,
        task_view="test",
        method_id=method,
        representation_id=f"{method}-representation",
        seed=seed,
        group_id=group,
        cohort_digest="2" * 64,
        comparison_protocol_digest=protocol * 64,
        config_digest=config_digest * 64,
        hardware_digest=hardware * 64,
        precision="bf16",
        status=status,
        outcomes=OutcomeCounts(
            attempted_count=0 if status is ObservationStatus.MISSING else 1,
            valid_count=1 if status is ObservationStatus.COMPLETE else 0,
            failed_count=1 if status is ObservationStatus.FAILED else 0,
            invalid_count=1 if status is ObservationStatus.INVALID else 0,
            unsupported_count=1 if status is ObservationStatus.UNSUPPORTED else 0,
            timeout_count=1 if status is ObservationStatus.TIMEOUT else 0,
        ),
        quality=_quality(quality) if status is ObservationStatus.COMPLETE else None,
        resources=(
            _resource("flops", flops, method=resource_method),
            _resource("wall_time_seconds", wall, method=resource_method),
        ),
        failure_reason=(
            None if status is ObservationStatus.COMPLETE else "retained fixture failure"
        ),
    )


def hashlib_sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _config(*, require_hardware: bool = False) -> FixedScaleAnalysisConfig:
    return FixedScaleAnalysisConfig(
        expected_seeds=_SEEDS,
        bootstrap_seed=7,
        bootstrap_replicates=200,
        comparisons=(
            ComparisonSpec(
                panel_id="fixture",
                track=Track.EQUIVALENCE,
                task_view="test",
                quality_metric_id="accuracy",
                resource_metric_ids=("flops", "wall_time_seconds"),
                require_same_hardware=require_hardware,
                require_same_precision=True,
            ),
        ),
    )


def test_comparable_rows_make_raw_seed_points_group_intervals_and_pareto():
    rows = tuple(
        _row(method, seed, group, quality=0.9 if method == "geml" else 0.8)
        for method in ("geml", "control")
        for seed in _SEEDS
        for group in ("g1", "g2", "g3")
    )
    result = summarize_fixed_scale(rows, _config())
    panel = result.panels[0]

    assert len(panel.seed_points) == 6
    assert panel.contrasts[0].cluster_count == 3
    assert panel.contrasts[0].availability is MetricAvailability.AVAILABLE
    assert all(point.eligible for point in panel.pareto_points)
    assert panel.quality_metric_id == "accuracy"
    assert panel.missing_seeds == ()
    assert panel.source_bindings[0].source_artifact_id == "results"
    assert panel.source_bindings[0].source_rows[0].source_locator.startswith("/rows/")
    assert all(point.source_row_ids for point in panel.seed_points)


def test_protocol_and_hardware_incompatibility_create_separate_panels():
    protocol_rows = (
        _row("left", _SEEDS[0], "g1", protocol="a"),
        _row("right", _SEEDS[0], "g1", protocol="c"),
    )
    assert len(summarize_fixed_scale(protocol_rows, _config()).panels) == 2

    hardware_rows = (
        _row("left", _SEEDS[0], "g1", hardware="b"),
        _row("right", _SEEDS[0], "g1", hardware="d"),
    )
    assert len(summarize_fixed_scale(hardware_rows, _config(require_hardware=True)).panels) == 2


def test_config_digest_incompatibility_creates_separate_panels():
    rows = (
        _row("left", _SEEDS[0], "g1", config_digest="3"),
        _row("right", _SEEDS[0], "g1", config_digest="4"),
    )
    assert len(summarize_fixed_scale(rows, _config()).panels) == 2


def test_measurement_methods_are_not_silently_pooled():
    rows = (
        _row(
            "measured",
            _SEEDS[0],
            "g1",
            resource_method=MeasurementMethod.MEASURED,
        ),
        _row(
            "estimated",
            _SEEDS[0],
            "g1",
            resource_method=MeasurementMethod.ESTIMATED,
        ),
    )
    result = summarize_fixed_scale(rows, _config())

    assert len(result.panels) == 2
    methods = {
        metric.method
        for panel in result.panels
        for point in panel.pareto_points
        for metric in point.resources
        if metric.availability is MetricAvailability.AVAILABLE
    }
    assert methods == {MeasurementMethod.MEASURED, MeasurementMethod.ESTIMATED}


def test_per_method_seed_coverage_cannot_hide_disjoint_seed_sets():
    rows = tuple(
        _row(method, seed, "g1")
        for method, seed in zip(("left", "right", "third"), _SEEDS, strict=True)
    )
    panel = summarize_fixed_scale(rows, _config()).panels[0]

    assert panel.missing_seeds == ()
    assert all(len(item.missing_seeds) == 2 for item in panel.method_seed_coverage)
    assert all(not point.eligible for point in panel.pareto_points)


def test_missing_telemetry_is_not_zero_and_is_pareto_ineligible():
    rows = (
        _row("complete", _SEEDS[0], "g1"),
        _row("missing", _SEEDS[0], "g1", wall=None),
    )
    result = summarize_fixed_scale(rows, _config())
    point = next(
        item
        for panel in result.panels
        for item in panel.pareto_points
        if item.method_id == "missing"
    )
    plot_point = next(
        item
        for item in build_plot_data(result)
        if item.method_id == "missing" and item.resource_metric_id == "wall_time_seconds"
    )

    assert not point.eligible
    assert plot_point.resource is None
    assert plot_point.resource != 0.0


def test_fixture_tables_and_plots_rebuild(tmp_path: Path):
    rows = tuple(
        _row(method, seed, group, quality=0.9 if method == "geml" else 0.8)
        for method in ("geml", "control")
        for seed in _SEEDS
        for group in ("g1", "g2")
    )
    result = summarize_fixed_scale(rows, _config())
    markdown = render_fixed_scale_markdown(result)
    plots = render_plots(result, tmp_path / "plots")

    assert "fixed-scale efficiency results" in markdown
    assert "Raw seed points" in markdown
    assert plots
    assert all(path.is_file() for path in plots)


def test_incompatible_comparison_keys_render_separate_plot_files(tmp_path: Path):
    rows = (
        _row("left", _SEEDS[0], "g1", protocol="a"),
        _row("right", _SEEDS[0], "g1", protocol="c"),
    )
    result = summarize_fixed_scale(rows, _config())
    plots = render_plots(result, tmp_path / "plots")

    assert len(result.panels) == 2
    assert len(plots) == 4
    assert len({path.name for path in plots}) == 4


def test_failed_cells_are_retained_without_becoming_quality_zeros():
    rows = (
        _row("ok", _SEEDS[0], "g1"),
        _row("failed", _SEEDS[0], "g2", status=ObservationStatus.FAILED),
    )
    result = summarize_fixed_scale(rows, _config())
    panel = result.panels[0]

    assert panel.retained_noncomplete_row_ids == (rows[1].row_id,)
    assert result.retained_noncomplete_row_ids == (rows[1].row_id,)
    assert panel.retained_noncomplete[0].status is ObservationStatus.FAILED
    assert panel.retained_noncomplete[0].outcomes.failed_count == 1
    assert panel.retained_noncomplete[0].failure_reason == "retained fixture failure"
    assert {point.method_id for point in panel.pareto_points} == {"ok"}


def test_failed_only_input_remains_visible_without_a_panel():
    row = _row("failed", _SEEDS[0], "g1", status=ObservationStatus.FAILED)
    result = summarize_fixed_scale((row,), _config())

    assert result.panels == ()
    assert result.retained_noncomplete_row_ids == (row.row_id,)


def test_invalid_cells_remain_distinct_from_execution_failures():
    row = _row("invalid", _SEEDS[0], "g1", status=ObservationStatus.INVALID)
    result = summarize_fixed_scale((row,), _config())

    assert result.retained_noncomplete[0].status is ObservationStatus.INVALID
    assert result.retained_noncomplete[0].outcomes.invalid_count == 1
    assert result.retained_noncomplete[0].outcomes.failed_count == 0


def test_paired_contrast_refuses_mismatched_group_cohorts():
    rows = (
        _row("left", _SEEDS[0], "g1"),
        _row("left", _SEEDS[0], "g2"),
        _row("right", _SEEDS[0], "g1"),
    )
    contrast = summarize_fixed_scale(rows, _config()).panels[0].contrasts[0]

    assert contrast.availability is MetricAvailability.UNAVAILABLE
    assert "cohorts differ" in (contrast.reason or "")


def test_parameter_count_alone_is_rejected():
    with pytest.raises(ValidationError, match="parameter count alone"):
        ComparisonSpec(
            panel_id="bad",
            track=Track.EQUIVALENCE,
            task_view="test",
            quality_metric_id="accuracy",
            resource_metric_ids=("parameter_count",),
        )
    with pytest.raises(ValidationError):
        ComparisonSpec(
            panel_id="../escape",
            track=Track.EQUIVALENCE,
            task_view="test",
            quality_metric_id="accuracy",
            resource_metric_ids=("flops", "wall_time_seconds"),
        )


def test_duplicate_scientific_cells_are_rejected():
    row = _row("same", _SEEDS[0], "g1")
    duplicate = row.model_copy(update={"row_id": "f" * 64})
    with pytest.raises(FixedScaleAnalysisError, match="duplicate method/seed/group"):
        summarize_fixed_scale((row, duplicate), _config())


def test_only_one_paired_group_keeps_interval_unavailable():
    rows = tuple(_row(method, _SEEDS[0], "g1") for method in ("left", "right"))
    contrast = summarize_fixed_scale(rows, _config()).panels[0].contrasts[0]

    assert contrast.availability is MetricAvailability.UNAVAILABLE
    assert contrast.cluster_count == 1


def test_checked_in_config_is_fixed_scale_and_production_pending():
    root = Path(__file__).resolve().parents[2]
    config, payload = load_fixed_scale_config(root / "configs" / "goal11_scaling.yaml")

    assert config.bootstrap_replicates == 2000
    assert payload["manifest_path"] is None
    with pytest.raises(FixedScaleRunError, match="production inputs are not frozen"):
        run_fixed_scale(root / "configs" / "goal11_scaling.yaml")


def test_fixed_scale_runner_rebuilds_authenticated_fixture_outputs(tmp_path: Path):
    import hashlib

    source = tmp_path / "source.json"
    rows = tuple(
        _row(method, seed, group)
        for method in ("left", "right")
        for seed in _SEEDS
        for group in ("g1", "g2")
    )
    source.write_text(
        json.dumps(
            {
                "schema_version": "fixture-v1",
                "rows": {row.row_id: row.evidence_projection() for row in rows},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = tuple(row.model_copy(update={"source_sha256": source_sha}) for row in rows)
    manifest = WorkshopManifest(
        config_sha256="c" * 64,
        expected_seeds=_SEEDS,
        artifacts=(
            ArtifactReference(
                artifact_id="results",
                producer_issue="fixture",
                category="result_table",
                roles=("fixed_scale",),
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
    observations_path = tmp_path / "rows.jsonl"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    audit_path.write_bytes(canonical_json_bytes(audit))
    observations_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "geml-goal11-fixed-scale-run-config-v1",
                "artifact_root": str(tmp_path),
                "manifest_path": str(manifest_path),
                "manifest_audit_path": str(audit_path),
                "observations_path": str(observations_path),
                "result_path": str(tmp_path / "out" / "result.json"),
                "markdown_path": str(tmp_path / "out" / "results.md"),
                "plot_dir": str(tmp_path / "out" / "plots"),
                "analysis": _config().model_dump(mode="json"),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = run_fixed_scale(config_path)

    assert result.panels
    assert (tmp_path / "out" / "result.json").is_file()
    assert (tmp_path / "out" / "results.md").is_file()
    assert tuple((tmp_path / "out" / "plots").glob("*.png"))
    tampered_quality = rows[0].quality.model_copy(update={"value": 0.123})
    tampered = rows[0].model_copy(update={"quality": tampered_quality})
    with pytest.raises(FixedScaleRunError, match="exact source record"):
        analyze_authenticated_rows(
            manifest,
            audit,
            (tampered, *rows[1:]),
            _config(),
            tmp_path,
        )
