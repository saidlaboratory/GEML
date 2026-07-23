"""Tiny-fixture tests for reproducible Goal 3 artifact analysis."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal, Inexact, localcontext
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

import geml.analysis.goal3.metrics as goal3_metrics
from geml.analysis.goal3.failures import OutcomeMiner
from geml.analysis.goal3.metrics import (
    RATIO_NAMES,
    AnalysisArtifactError,
    AnalysisRow,
    ReusePattern,
    StratificationAxis,
    analyze_goal3_artifacts,
    save_analysis,
)
from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.data.storage.manifests import (
    build_corpus_manifest,
    build_split_manifest,
    write_manifest_bundle,
)
from geml.data.storage.shards import write_shards
from geml.experiments.goal3.run import (
    Goal3Stage,
    process_expression_record,
    run_goal3_stage,
)
from geml.experiments.goal3.runtime import Goal3ArtifactError, sha256_file
from geml.plots.goal3 import (
    STANDARD_SCALE_CHECKPOINTS,
    PlotDataError,
    build_runtime_curve,
    missing_checkpoints,
    plot_data_payload,
    stability_deltas,
)

_RATIO_ORIENTATION = (
    "raw=eml_tree/ast_tree;dag_tree=eml_dag/ast_tree;"
    "dag_dag=eml_dag/ast_dag;ast_compression=ast_tree/ast_dag;"
    "eml_compression=eml_tree/eml_dag"
)


def _expression_id(srepr: str, domain: str = "safe_real") -> str:
    payload = f"geml-expression-v1\0{domain}\0{srepr}".encode()
    return hashlib.sha256(payload).hexdigest()


def _record(
    index: int,
    *,
    split: CorpusSplit = CorpusSplit.TRAIN,
    family: str = "algebraic_core",
    domain: str = "safe_real",
    srepr: str | None = None,
    operator_counts: dict[str, int] | None = None,
    target_ast_size: int = 3,
    target_depth: int = 1,
) -> ExpressionRecord:
    source = srepr or f"Add(Symbol('x', real=True), Integer({index + 2}))"
    return ExpressionRecord(
        expression_id=_expression_id(source, domain),
        sympy_srepr=source,
        display_text=f"fixture-{index}",
        latex_text=None,
        split=split,
        operator_family=family,
        domain_mode=domain,
        variables=("x",),
        target_ast_size=target_ast_size,
        target_depth=target_depth,
        generator_seed=100 + index,
        generator_metadata={
            "operator_counts": operator_counts or {"add": 1, "integer": 1, "symbol": 1},
            "fixture": "goal3-analysis",
        },
    )


def _processed_mapping(
    index: int,
    **record_arguments: object,
) -> dict[str, object]:
    record = _record(index, **record_arguments)
    return process_expression_record(
        record,
        shard_id=f"fixture-shard-{index}",
        shard_path=f"data/fixture-{index}.parquet",
        input_row_index=index,
    )


def _set_ratio(
    row: dict[str, object],
    name: str,
    numerator: int,
    denominator: int,
) -> None:
    value = Fraction(numerator, denominator)
    row[f"{name}_numerator"] = str(value.numerator)
    row[f"{name}_denominator"] = str(value.denominator)
    row[f"{name}_exact"] = f"{value.numerator}/{value.denominator}"
    row[f"{name}_value"] = float(value)


def _set_reuse(
    row: dict[str, object],
    prefix: str,
    *,
    nodes: int,
    references: int,
    overhead: int,
    maximum: int,
    depth_sum: int,
    concentration_numerator: int,
    concentration_denominator: int,
) -> None:
    row[f"{prefix}_reused_node_count"] = nodes
    row[f"{prefix}_reused_reference_count"] = references
    row[f"{prefix}_child_reference_overhead"] = overhead
    row[f"{prefix}_max_reuse_count"] = maximum
    row[f"{prefix}_reuse_depth_sum"] = depth_sum
    row[f"{prefix}_reuse_depth_count"] = nodes
    row[f"{prefix}_mean_reuse_depth"] = float(Fraction(depth_sum, nodes)) if nodes else None
    _set_ratio(
        row,
        f"{prefix}_sharing_concentration",
        concentration_numerator,
        concentration_denominator,
    )


def _analysis_rows() -> list[dict[str, object]]:
    first = _processed_mapping(0)
    second = _processed_mapping(
        1,
        split=CorpusSplit.VALIDATION,
        family="exp_log",
        domain="positive_real",
        srepr="exp(Symbol('x', positive=True))",
        operator_counts={"exp": 1, "symbol": 1},
        target_ast_size=99,
        target_depth=99,
    )
    third = _processed_mapping(
        2,
        split=CorpusSplit.TEST_IID,
        family="trig_hyperbolic",
        srepr="sin(Symbol('x', real=True))",
        operator_counts={"sin": 1, "symbol": 1},
    )
    failure = _processed_mapping(
        3,
        split=CorpusSplit.TEST_OOD,
        family="ood_stress",
        domain="nonzero_real",
        srepr="DefinitelyNotValid(",
        operator_counts={"symbol": 1},
    )
    assert all(row["status"] == "success" for row in (first, second, third))
    assert failure["status"] == "failure"

    # Exact values make the compression and remaining-alpha winners diverge.
    _set_ratio(first, "eml_compression", 10, 1)
    _set_ratio(first, "dag_alpha_vs_ast_tree", 4, 1)
    _set_ratio(second, "eml_compression", 6, 5)
    _set_ratio(second, "dag_alpha_vs_ast_tree", 1, 1)
    _set_ratio(third, "eml_compression", 3, 1)
    _set_ratio(third, "dag_alpha_vs_ast_tree", 3, 2)

    _set_reuse(
        first,
        "ast_dag",
        nodes=0,
        references=0,
        overhead=0,
        maximum=0,
        depth_sum=0,
        concentration_numerator=0,
        concentration_denominator=1,
    )
    _set_reuse(
        first,
        "eml_dag",
        nodes=2,
        references=5,
        overhead=3,
        maximum=3,
        depth_sum=7,
        concentration_numerator=2,
        concentration_denominator=3,
    )
    for row in (second, third):
        for prefix in ("ast_dag", "eml_dag"):
            _set_reuse(
                row,
                prefix,
                nodes=0,
                references=0,
                overhead=0,
                maximum=0,
                depth_sum=0,
                concentration_numerator=0,
                concentration_denominator=1,
            )
    return [first, second, third, failure]


def _telemetry() -> tuple[dict[str, object], ...]:
    return (
        {
            "telemetry_scope": (
                "input read excluded; direct row processing, Parquet construction, "
                "and process-tree RSS sampling included"
            ),
            "processing_wall_seconds": 2.0,
            "peak_resident_memory_bytes": 120,
            "progress_samples": [
                {
                    "global_processed_count": 1,
                    "shard_processed_count": 1,
                    "processing_wall_seconds": 0.5,
                    "peak_resident_memory_bytes": 90,
                },
                {
                    "global_processed_count": 2,
                    "shard_processed_count": 2,
                    "processing_wall_seconds": 2.0,
                    "peak_resident_memory_bytes": 120,
                },
            ],
        },
        {
            "telemetry_scope": (
                "input read excluded; direct row processing, Parquet construction, "
                "and process-tree RSS sampling included"
            ),
            "processing_wall_seconds": 1.0,
            "peak_resident_memory_bytes": 100,
            "progress_samples": [
                {
                    "global_processed_count": 3,
                    "shard_processed_count": 1,
                    "processing_wall_seconds": 0.4,
                    "peak_resident_memory_bytes": 100,
                },
                {
                    "global_processed_count": 4,
                    "shard_processed_count": 2,
                    "processing_wall_seconds": 1.0,
                    "peak_resident_memory_bytes": 100,
                },
            ],
        },
    )


def _patch_public_readers(
    monkeypatch: pytest.MonkeyPatch,
    manifest_path: Path,
    rows: list[dict[str, object]],
    *,
    chunked: bool,
    telemetry_payloads: tuple[dict[str, object], ...] | None = None,
) -> None:
    success_count = sum(row["status"] == "success" for row in rows)
    failure_count = len(rows) - success_count
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "final",
                "processed_count": len(rows),
                "success_count": success_count,
                "failure_count": failure_count,
                "summary": {"path": "summary.json"},
            }
        ),
        encoding="utf-8",
    )
    (manifest_path.parent / "summary.json").write_text(
        json.dumps({"science_fingerprint": "a" * 64}),
        encoding="utf-8",
    )

    def tables(_path):
        if chunked:
            yield pa.Table.from_pylist(rows[:1])
            yield pa.Table.from_pylist(rows[1:3])
            yield pa.Table.from_pylist(rows[3:])
        else:
            yield pa.Table.from_pylist(rows)

    monkeypatch.setattr(goal3_metrics, "iter_metric_tables", tables)
    monkeypatch.setattr(
        goal3_metrics,
        "iter_shard_telemetry",
        lambda _path: iter(telemetry_payloads or _telemetry()),
    )


def test_row_adapter_uses_pinned_reuse_definitions() -> None:
    mapping = _analysis_rows()[0]
    row = AnalysisRow.from_mapping(mapping)

    assert row.reuse_pattern is ReusePattern.EML_ONLY
    assert row.metrics is not None
    assert row.metrics.ast_reuse.reused_reference_count == 0
    assert row.metrics.ast_reuse.max_reuse_indegree == 0
    assert row.metrics.ast_reuse.sharing_concentration == 0
    assert row.metrics.eml_reuse.reused_node_count == 2
    assert row.metrics.eml_reuse.reused_reference_count == 5
    assert row.metrics.eml_reuse.excess_reference_count == 3
    assert row.metrics.eml_reuse.child_reference_overhead == 3
    assert row.metrics.eml_reuse.max_reuse_indegree == 3
    assert row.metrics.eml_reuse.reuse_depth_sum == 7
    assert row.metrics.eml_reuse.sharing_concentration == Fraction(2, 3)


def test_row_adapter_rejects_inconsistent_reuse_accounting() -> None:
    mapping = _analysis_rows()[0]
    mapping["eml_dag_child_reference_overhead"] = 4

    with pytest.raises(AnalysisArtifactError, match="accounting disagree"):
        AnalysisRow.from_mapping(mapping)


def test_analysis_stratifies_exact_metrics_and_retains_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _analysis_rows()
    manifest = tmp_path / "manifest.json"
    _patch_public_readers(monkeypatch, manifest, rows, chunked=True)

    report = analyze_goal3_artifacts(
        manifest,
        scale_checkpoints=(2, 4),
        ranking_limit=2,
    )

    assert report.overall.all_processed_count == 4
    assert report.overall.valid_count == 3
    assert report.overall.failure_count == 1
    compression_mean = report.overall.ratio_mean("eml_compression")
    assert compression_mean is not None
    assert compression_mean.decimal_value == Decimal(
        "4.7333333333333333333333333333333333333333333333333"
    )
    assert compression_mean.sample_count == 3
    assert compression_mean.as_dict()["approximate"] is True
    assert "exact" not in compression_mean.as_dict()
    assert {table.axis for table in report.strata} == set(StratificationAxis)
    strata = {table.axis: table for table in report.strata}
    assert {group.key for group in strata[StratificationAxis.FAMILY].groups} == {
        "algebraic_core",
        "exp_log",
        "trig_hyperbolic",
        "ood_stress",
    }
    assert "2" in {group.key for group in strata[StratificationAxis.ACTUAL_AST_SIZE].groups}
    assert "99" not in {group.key for group in strata[StratificationAxis.ACTUAL_AST_SIZE].groups}
    assert "<unavailable>" in {
        group.key for group in strata[StratificationAxis.ACTUAL_AST_SIZE].groups
    }
    eml_only = next(
        group
        for group in strata[StratificationAxis.REUSE_PATTERN].groups
        if group.key == "eml_only"
    )
    assert eml_only.eml_reuse is not None
    assert eml_only.eml_reuse.total_reused_reference_count == 5
    assert eml_only.eml_reuse.total_excess_reference_count == 3
    assert eml_only.eml_reuse.mean_reuse_depth.fraction == Fraction(7, 2)
    assert report.outcomes.failure_count == 1
    assert report.outcomes.failure_stage_counts == (("ast_build", 1),)
    assert report.outcomes.processing_failures[0].error_message
    assert (
        report.outcomes.highest_compression[0].expression_id
        != report.outcomes.lowest_remaining_alpha[0].expression_id
    )
    assert report.outcomes.structurally_competitive_count == 1


def test_outcome_rankings_compare_exact_fractions_when_floats_collide() -> None:
    lower_mapping = _processed_mapping(0)
    higher_mapping = _processed_mapping(1)
    denominator = 2**54
    _set_ratio(lower_mapping, "eml_compression", denominator + 1, denominator)
    _set_ratio(higher_mapping, "eml_compression", denominator + 2, denominator)
    lower = AnalysisRow.from_mapping(lower_mapping)
    higher = AnalysisRow.from_mapping(higher_mapping)
    assert lower.metrics is not None
    assert higher.metrics is not None
    assert float(lower.metrics.ratio("eml_compression")) == float(
        higher.metrics.ratio("eml_compression")
    )

    miner = OutcomeMiner(limit=1)
    miner.add(lower)
    miner.add(higher)
    outcomes = miner.finish()

    assert outcomes.lowest_compression[0].expression_id == lower.expression_id
    assert outcomes.highest_compression[0].expression_id == higher.expression_id


def test_analysis_is_independent_of_table_chunking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _analysis_rows()
    manifest = tmp_path / "manifest.json"
    _patch_public_readers(monkeypatch, manifest, rows, chunked=True)
    chunked = analyze_goal3_artifacts(manifest, scale_checkpoints=(2, 4))
    _patch_public_readers(monkeypatch, manifest, rows, chunked=False)
    combined = analyze_goal3_artifacts(manifest, scale_checkpoints=(2, 4))

    assert chunked.fingerprint == combined.fingerprint
    assert chunked.as_dict() == combined.as_dict()


def test_decimal_mean_compensation_recovers_a_low_order_term() -> None:
    accumulator = goal3_metrics._DecimalMeanAccumulator()
    for value in (Fraction(10**80), Fraction(1), Fraction(-(10**80))):
        accumulator.add(value)

    assert accumulator.mean().decimal == ("0.33333333333333333333333333333333333333333333333333")


def test_runtime_and_stability_use_exact_cumulative_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _analysis_rows()
    manifest = tmp_path / "manifest.json"
    _patch_public_readers(monkeypatch, manifest, rows, chunked=True)
    report = analyze_goal3_artifacts(manifest, scale_checkpoints=(2, 4))

    assert [point.processed_count for point in report.stability_curve] == [2, 4]
    first, second = report.stability_curve
    assert first.processing_wall_seconds == 2.0
    assert first.throughput_rows_per_second == 1.0
    assert first.peak_resident_memory_bytes == 120
    assert second.processing_wall_seconds == 3.0
    assert second.throughput_rows_per_second == pytest.approx(4 / 3)
    assert second.peak_resident_memory_bytes == 120
    assert second.processing_time_scope.startswith("input read excluded")
    assert "worker descendants" in second.peak_memory_scope
    assert first.valid_count == 2
    assert second.valid_count == 3
    assert second.failure_count == 1
    checkpoint_mean = second.ratio_mean("eml_compression")
    assert checkpoint_mean is not None
    assert checkpoint_mean.decimal_value == Decimal(
        "4.7333333333333333333333333333333333333333333333333"
    )
    deltas = stability_deltas(report.stability_curve, "eml_compression")
    assert len(deltas) == 1
    assert deltas[0].approximate is True
    assert plot_data_payload(report.stability_curve)["x_axis"]["values"] == [2, 4]
    assert STANDARD_SCALE_CHECKPOINTS == (10_000, 50_000, 100_000, 250_000)


def test_missing_exact_runtime_checkpoint_is_not_substituted() -> None:
    payload = _telemetry()[0]
    runtime = build_runtime_curve((payload,))

    assert missing_checkpoints(runtime, (1, 2, 3)) == (3,)
    assert {point.processed_count for point in runtime} == {1, 2}


def test_missing_checkpoint_causes_a_retained_analysis_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _analysis_rows()
    manifest = tmp_path / "manifest.json"
    _patch_public_readers(monkeypatch, manifest, rows, chunked=False)
    monkeypatch.setattr(
        goal3_metrics,
        "iter_shard_telemetry",
        lambda _path: iter(_telemetry()[:1]),
    )

    with pytest.raises(PlotDataError, match="runtime=\\[4\\]"):
        analyze_goal3_artifacts(manifest, scale_checkpoints=(2, 4))


def test_all_failure_run_preserves_denominators_and_plot_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _processed_mapping(
            index,
            srepr=f"DefinitelyNotValid{index}(",
            operator_counts={"symbol": 1},
        )
        for index in range(4)
    ]
    assert all(row["status"] == "failure" for row in rows)
    source_root = tmp_path / "source"
    source_root.mkdir()
    manifest = source_root / "manifest.json"
    _patch_public_readers(monkeypatch, manifest, rows, chunked=True)

    report = analyze_goal3_artifacts(manifest, scale_checkpoints=(2, 4))
    payload = plot_data_payload(report.stability_curve)

    assert report.overall.all_processed_count == 4
    assert report.overall.valid_count == 0
    assert report.overall.failure_count == 4
    assert report.overall.ratio_means == ()
    assert payload["metric_stability"]["eml_compression"]["decimal"] == [None, None]
    assert payload["metric_stability"]["eml_compression"]["valid_denominator"] == [0, 0]
    with pytest.raises(PlotDataError, match="no valid-only denominator"):
        stability_deltas(report.stability_curve, "eml_compression")
    assert save_analysis(report, tmp_path / "analysis").manifest_path.is_file()


def test_rankings_use_exact_values_and_deterministic_ties() -> None:
    first, second, third, _ = _analysis_rows()
    _set_ratio(first, "eml_compression", 3, 1)
    _set_ratio(second, "eml_compression", 3, 1)
    _set_ratio(third, "eml_compression", 1, 1)
    first_row = AnalysisRow.from_mapping(first)
    second_row = AnalysisRow.from_mapping(second)
    third_row = AnalysisRow.from_mapping(third)
    miner = OutcomeMiner(limit=3)
    for row in (second_row, third_row, first_row):
        miner.add(row)
    report = miner.finish()

    tied = [
        result.expression_id for result in report.highest_compression if result.eml_compression == 3
    ]
    assert tied == sorted(tied)
    assert report.lowest_compression[0].expression_id == third_row.expression_id
    assert report.lowest_compression[0].eml_compression == 1
    assert report.lowest_remaining_alpha[0].remaining_alpha == 1


def _artifact_checksums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_saving_is_deterministic_and_never_edits_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _analysis_rows()
    source_root = tmp_path / "source"
    source_root.mkdir()
    manifest = source_root / "manifest.json"
    _patch_public_readers(monkeypatch, manifest, rows, chunked=True)
    report = analyze_goal3_artifacts(manifest, scale_checkpoints=(2, 4))
    before = _artifact_checksums(source_root)

    output = tmp_path / "analysis"
    first = save_analysis(report, output)
    first_checksums = _artifact_checksums(output)
    second = save_analysis(report, output)

    assert first == second
    assert _artifact_checksums(output) == first_checksums
    assert _artifact_checksums(source_root) == before
    saved_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["analysis_fingerprint"] == report.fingerprint
    assert {product["path"] for product in saved_manifest["products"]} == {
        "metrics.table.json",
        "outcomes.table.json",
        "stability.plot-data.json",
    }

    with pytest.raises(AnalysisArtifactError, match="outside"):
        save_analysis(report, source_root / "analysis")


def _first_primes(count: int) -> tuple[int, ...]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _coprime_ratio_rows(count: int) -> list[dict[str, object]]:
    template = _processed_mapping(0)
    rows: list[dict[str, object]] = []
    for index, denominator in enumerate(_first_primes(count)):
        row = dict(template)
        row["expression_id"] = hashlib.sha256(f"coprime-{index}".encode()).hexdigest()
        row["input_row_index"] = index
        for name in RATIO_NAMES:
            _set_ratio(row, name, denominator + 1, denominator)
        rows.append(row)
    return rows


def _single_checkpoint_telemetry(count: int) -> tuple[dict[str, object], ...]:
    return (
        {
            "telemetry_scope": (
                "input read excluded; direct row processing, Parquet construction, "
                "and process-tree RSS sampling included"
            ),
            "processing_wall_seconds": 1.0,
            "peak_resident_memory_bytes": 100,
            "progress_samples": [
                {
                    "global_processed_count": count,
                    "shard_processed_count": count,
                    "processing_wall_seconds": 1.0,
                    "peak_resident_memory_bytes": 100,
                }
            ],
        },
    )


def test_decimal_aggregation_ignores_ambient_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    denominator = 3 * 10**100
    rows = [_processed_mapping(0), _processed_mapping(1)]
    for numerator, row in enumerate(rows, start=1):
        for name in RATIO_NAMES:
            _set_ratio(row, name, numerator, denominator)

    manifest = tmp_path / "manifest.json"
    _patch_public_readers(
        monkeypatch,
        manifest,
        rows,
        chunked=True,
        telemetry_payloads=_single_checkpoint_telemetry(2),
    )
    baseline = analyze_goal3_artifacts(manifest, scale_checkpoints=(2,))

    with localcontext() as ambient:
        ambient.prec = 2
        ambient.rounding = ROUND_DOWN
        ambient.Emin = -9
        ambient.Emax = 9
        ambient.capitals = 0
        ambient.clamp = 1
        ambient.traps[Inexact] = True
        contaminated = analyze_goal3_artifacts(manifest, scale_checkpoints=(2,))

    mean = contaminated.overall.ratio_mean("raw_tree_alpha")
    assert mean is not None
    assert "E" in mean.decimal
    assert baseline.fingerprint == contaminated.fingerprint
    assert baseline.as_dict() == contaminated.as_dict()


def test_coprime_denominators_have_bounded_aggregate_artifact_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_sizes: list[int] = []
    for count in (25, 1_000):
        source_root = tmp_path / f"source-{count}"
        source_root.mkdir()
        manifest = source_root / "manifest.json"
        _patch_public_readers(
            monkeypatch,
            manifest,
            _coprime_ratio_rows(count),
            chunked=True,
            telemetry_payloads=_single_checkpoint_telemetry(count),
        )
        report = analyze_goal3_artifacts(
            manifest,
            scale_checkpoints=(count,),
            ranking_limit=5,
        )
        saved = save_analysis(report, tmp_path / f"analysis-{count}")
        observed_sizes.append(saved.metrics_path.stat().st_size)
        mean = report.overall.ratio_mean("raw_tree_alpha")
        assert mean is not None
        assert len(mean.decimal) <= 60
        assert mean.sample_count == count
        assert mean.as_dict()["approximate"] is True

    small_size, large_size = observed_sizes
    assert large_size < 200_000
    assert large_size <= small_size + 10_000


def test_cli_reports_upstream_artifact_errors_as_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_analysis(*_args: object, **_kwargs: object) -> None:
        raise Goal3ArtifactError("invalid saved run")

    monkeypatch.setattr(goal3_metrics, "analyze_goal3_artifacts", fail_analysis)

    exit_code = goal3_metrics.main(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "analysis"),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert exit_code == 2
    assert payload == {
        "passed": False,
        "error_type": "Goal3ArtifactError",
        "message": "invalid saved run",
    }


def _write_fixture_corpus(root: Path) -> Path:
    records = (
        _record(
            0,
            split=CorpusSplit.TRAIN,
            srepr="Symbol('x', real=True)",
            operator_counts={"symbol": 1},
            target_ast_size=1,
            target_depth=0,
        ),
        _record(
            1,
            split=CorpusSplit.VALIDATION,
            srepr="Add(Symbol('x', real=True), Symbol('x', real=True))",
            operator_counts={"add": 1, "symbol": 2},
        ),
        _record(
            2,
            split=CorpusSplit.TEST_IID,
            family="exp_log",
            srepr="exp(Symbol('x', real=True))",
            operator_counts={"exp": 1, "symbol": 1},
            target_ast_size=2,
        ),
        _record(
            3,
            split=CorpusSplit.TEST_OOD,
            family="ood_stress",
            domain="nonzero_real",
            srepr="DefinitelyNotValid(",
            operator_counts={"symbol": 1},
        ),
    )
    run_root = root / "corpus"
    source_config = root / "generator.yaml"
    source_config.write_text("fixture: goal3-analysis\n", encoding="utf-8")
    split_manifests = []
    for split in CorpusSplit:
        split_records = [record for record in records if record.split is split]
        shards = write_shards(
            split_records,
            run_root / "data" / split.value,
            corpus_id="goal3-analysis-fixture",
            split=split,
            schema_version="geml-corpus-v1",
            minimum_rows=1,
            maximum_rows=4,
            allow_small_fixture=True,
            manifest_root=run_root,
        )
        split_manifests.append(build_split_manifest(shards))
    manifest = build_corpus_manifest(
        split_manifests,
        corpus_id="goal3-analysis-fixture",
        schema_version="geml-corpus-v1",
        config_path=source_config,
        generator_seed=20260723,
        git_commit="fixture",
        created_at=datetime(2026, 7, 23, tzinfo=UTC),
        package_names=("geml",),
    )
    return write_manifest_bundle(
        manifest,
        run_root / "manifests",
        artifact_root=run_root,
        config_path=source_config,
    ).corpus_manifest


def _write_runner_config(root: Path, corpus_manifest: Path) -> Path:
    config = {
        "schema_version": "geml-goal3-config-v1",
        "output_root": (root / "goal3-run").as_posix(),
        "compiler_mode": "official_v4",
        "construction_path": "direct_hashcons",
        "stages": {
            "smoke": {
                "manifest": corpus_manifest.as_posix(),
                "source_label": "tiny_analysis_fixture",
                "expected_count": 4,
                "row_limit": None,
            }
        },
        "input_validation": {
            "require_manifest_sidecars": True,
            "require_qa_pass": False,
            "require_unique_expression_ids": True,
        },
        "processing": {
            "worker_processes": 1,
            "worker_batch_size": 1,
            "worker_chunksize": 1,
            "parquet_row_group_size": 2,
            "resume": True,
            "atomic_finalization": True,
        },
        "audit": {"require_ready": True},
        "telemetry": {
            "package_versions": ["geml", "mpmath", "psutil", "pyarrow", "sympy"],
            "scale_checkpoints": [2, 4],
        },
        "scientific_metrics": {
            "ratio_orientation": _RATIO_ORIENTATION,
            "reuse_depth": "minimum_root_distance",
            "sharing_concentration": "max_excess_reference_share",
            "reused_reference_count": "sum_indegrees_of_reused_nodes",
            "max_reuse_count": "maximum_reused_node_indegree",
            "child_reference_overhead": "sum_excess_references",
        },
    }
    path = root / "goal3-analysis.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_analysis_reproduces_from_real_validated_tiny_artifacts(
    tmp_path: Path,
) -> None:
    corpus_manifest = _write_fixture_corpus(tmp_path)
    config = _write_runner_config(tmp_path, corpus_manifest)
    run = run_goal3_stage(config, Goal3Stage.SMOKE)
    before = _artifact_checksums(run.output_root)

    first = analyze_goal3_artifacts(run.manifest_path, scale_checkpoints=(2, 4))
    second = analyze_goal3_artifacts(run.manifest_path, scale_checkpoints=(2, 4))
    saved = save_analysis(first, tmp_path / "analysis-output")

    assert first.fingerprint == second.fingerprint
    assert first.overall.all_processed_count == 4
    assert first.overall.valid_count == 3
    assert first.overall.failure_count == 1
    assert [point.processed_count for point in first.stability_curve] == [2, 4]
    assert len(first.outcomes.processing_failures) == 1
    assert saved.manifest_path.is_file()
    assert _artifact_checksums(run.output_root) == before


def test_ratio_names_are_complete_and_claims_remain_separate() -> None:
    assert RATIO_NAMES == (
        "raw_tree_alpha",
        "dag_alpha_vs_ast_tree",
        "dag_alpha_vs_ast_dag",
        "ast_compression",
        "eml_compression",
    )
