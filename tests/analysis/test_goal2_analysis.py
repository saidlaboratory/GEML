"""Tiny synthetic-table tests for Goal 2 analysis and plotting."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from geml.analysis.goal2.failures import (
    build_failure_tables,
    semantic_backend_status_counts,
)
from geml.analysis.goal2.stratified import (
    ast_size_bucket,
    build_stratified_table,
    build_threshold_table,
    run_analysis,
    summarize_population,
    validate_analysis_manifest,
)
from geml.data.storage.shards import sha256_file
from geml.eml.compiler_core import CompilerMode
from geml.experiments.goal2.run import (
    _METRIC_SCHEMA,
    _METRIC_SCHEMA_SHA256,
    Goal2ArtifactError,
    load_goal2_config,
)
from geml.plots.goal2 import generate_plots, validate_plot_manifest
from geml.verification.eml.numeric import ProbeStatus

_SCENARIO = {
    "name": "fixture_threshold",
    "scope_families": ["algebraic_core"],
    "definition_status": "defined",
    "K": 1,
    "L": 1,
    "formula": "1 + ln(K) / ln(4L)",
    "threshold_value": 1.0,
    "derivation": "fixture K=1 and L=1",
    "references": ["temporary fixture"],
}


def _outcome(status: str, *, applicable: bool = True) -> str:
    reported_status = status if applicable else "not_applicable"
    return json.dumps(
        [
            {
                "scenario_name": "fixture_threshold",
                "status": reported_status,
                "passed": (status == "passed" if reported_status in {"passed", "failed"} else None),
                "threshold_value": 1.0,
                "K": 1,
                "L": 1,
                "formula": "1 + ln(K) / ln(4L)",
                "message": None,
            }
        ],
        separators=(",", ":"),
        sort_keys=True,
    )


def _probe_results(semantic_status: str, *, family: str) -> str:
    if semantic_status == "semantic_not_selected":
        return "[]"
    probe_status = "mismatch" if semantic_status == "semantic_mismatch" else "pass"
    return json.dumps(
        [
            {
                "backend": backend,
                "compiler_mode": CompilerMode.OFFICIAL_V4.value,
                "operator": "source_expression",
                "operator_family": family,
                "domain_mode": "safe_real",
                "sample_label": "probe-000",
                "status": probe_status,
                "variable_assignments": [["x", "2"]],
            }
            for backend in ("mpmath", "numpy_complex128")
        ],
        separators=(",", ":"),
        sort_keys=True,
    )


def _row(
    index: int,
    *,
    alpha: float | None,
    threshold_status: str,
    family: str = "algebraic_core",
    ast_nodes: int = 8,
    ast_depth: int = 2,
    count_status: str = "success",
    semantic_status: str = "passed",
    error_stage: str | None = None,
) -> dict[str, object]:
    eml_nodes = None if alpha is None else int(alpha * ast_nodes)
    successful = count_status == "success"
    applicable = family == "algebraic_core"
    return {
        "schema_version": "geml-goal2-metrics-v1",
        "expression_id": f"{index:064x}",
        "input_shard_id": "fixture-shard",
        "input_shard_path": "data/fixture.parquet",
        "split": "train" if index % 2 else "test_ood",
        "iid_ood": "iid" if index % 2 else "ood",
        "operator_family": family,
        "operator_signature": "symbol:1",
        "source_operator_counts_json": '{"symbol":1}',
        "domain_mode": "safe_real",
        "variables_json": '["x"]',
        "variable_count": 1,
        "source_constant_counts_json": '{"integer":0,"one":0,"rational":0}',
        "source_constant_count": 0,
        "sympy_srepr": "Symbol('x', real=True)",
        "target_ast_size": ast_nodes,
        "target_depth": ast_depth,
        "ast_node_count": ast_nodes if successful else None,
        "ast_edge_count": ast_nodes - 1 if successful else None,
        "ast_leaf_count": 1 if successful else None,
        "ast_operator_count": ast_nodes - 1 if successful else None,
        "ast_depth": ast_depth if successful else None,
        "compiler_mode": "official_v4",
        "eml_node_count": str(eml_nodes) if eml_nodes is not None else None,
        "eml_edge_count": str(eml_nodes - 1) if eml_nodes is not None else None,
        "eml_leaf_count": str((eml_nodes + 1) // 2) if eml_nodes is not None else None,
        "eml_operator_count": (
            str(eml_nodes - ((eml_nodes + 1) // 2)) if eml_nodes is not None else None
        ),
        "eml_depth": ast_depth + 2 if eml_nodes is not None else None,
        "compiler_operation_counts_json": '{"primitive_eml":1}' if successful else None,
        "compiler_operation_total": "1" if successful else None,
        "tree_alpha_numerator": str(eml_nodes) if eml_nodes is not None else None,
        "tree_alpha_denominator": ast_nodes if eml_nodes is not None else None,
        "tree_alpha_exact_ratio": f"{eml_nodes}/{ast_nodes}" if eml_nodes is not None else None,
        "tree_alpha_value": alpha,
        "tree_alpha_status": "success" if alpha is not None else "missing_denominator",
        "threshold_outcomes_json": _outcome(threshold_status, applicable=applicable),
        "processing_status": count_status,
        "count_status": count_status,
        "semantic_selected": semantic_status != "semantic_not_selected",
        "materialization_status": "materialized"
        if semantic_status != "semantic_not_selected"
        else "not_selected",
        "semantic_status": semantic_status,
        "semantic_unique_assignment_count": (
            1 if semantic_status != "semantic_not_selected" else 0
        ),
        "semantic_requested_count": 2 if semantic_status != "semantic_not_selected" else 0,
        "semantic_pass_count": 2 if semantic_status == "passed" else 0,
        "semantic_failure_count": 2
        if semantic_status not in {"passed", "semantic_not_selected"}
        else 0,
        "semantic_maximum_absolute_error": "0" if semantic_status == "passed" else None,
        "semantic_maximum_relative_error": "0" if semantic_status == "passed" else None,
        "semantic_status_counts_json": (
            '{"pass":2}'
            if semantic_status == "passed"
            else ('{"mismatch":2}' if semantic_status == "semantic_mismatch" else "{}")
        ),
        "semantic_probe_results_json": _probe_results(semantic_status, family=family),
        "semantic_assumptions_json": "[]",
        "semantic_methods_json": "[]",
        "processing_elapsed_seconds": index / 1000,
        "error_stage": error_stage,
        "error_type": "FixtureError" if error_stage else None,
        "error_message": "retained fixture failure" if error_stage else None,
    }


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    final = pd.DataFrame.from_records(
        [
            _row(1, alpha=0.5, threshold_status="passed", ast_nodes=8),
            _row(2, alpha=1.0, threshold_status="failed", ast_nodes=9),
            _row(3, alpha=2.0, threshold_status="failed", ast_nodes=16),
            _row(
                4,
                alpha=None,
                threshold_status="invalid_alpha",
                ast_nodes=17,
                count_status="count_error",
                semantic_status="semantic_not_selected",
                error_stage="count",
            ),
            _row(
                5,
                alpha=3.0,
                threshold_status="not_applicable",
                family="exp_log",
                ast_nodes=32,
                semantic_status="semantic_mismatch",
                error_stage="semantic",
            ),
        ]
    )
    pilot = final.iloc[[0, 1, 4]].copy().reset_index(drop=True)
    return pilot, final


def _enrich(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    value["ast_size_bucket"] = value["ast_node_count"].map(
        lambda item: ast_size_bucket(item, ((1, 8), (9, 16), (17, 32)))
    )
    value["canonical_ast_constant_category"] = "none"
    value["canonical_ast_constant_count"] = value["source_constant_count"]
    value["count_semantic_status"] = value["count_status"] + "|" + value["semantic_status"]
    value["eml_node_number"] = pd.to_numeric(value["eml_node_count"], errors="coerce")
    value["compiler_operation_number"] = pd.to_numeric(
        value["compiler_operation_total"], errors="coerce"
    )
    value["failure"] = value["error_stage"].notna() | value["count_status"].ne("success")
    return value


def _write_metrics(
    root: Path,
    rows: pd.DataFrame,
    *,
    stage: str,
    config_hash: str,
    source_label: str,
    runner_fingerprint: str = "a" * 64,
) -> Path:
    root.mkdir(parents=True)
    shard = root / "metrics.parquet"
    records = [
        {name: None if pd.isna(value) else value for name, value in row.items()}
        for row in rows.to_dict(orient="records")
    ]
    scenarios = [
        scenario.as_dict() for scenario in load_goal2_config("configs/goal2_final.yaml").thresholds
    ]
    for record in records:
        outcomes = []
        for scenario in scenarios:
            applicable = record["operator_family"] in scenario["scope_families"]
            alpha = record["tree_alpha_value"]
            status = "not_applicable"
            if applicable:
                status = (
                    "invalid_alpha"
                    if alpha is None
                    else ("passed" if alpha < scenario["threshold_value"] else "failed")
                )
            outcomes.append(
                {
                    "scenario_name": scenario["name"],
                    "status": status,
                    "passed": status == "passed" if status in {"passed", "failed"} else None,
                    "threshold_value": scenario["threshold_value"],
                    "K": scenario["K"],
                    "L": scenario["L"],
                    "formula": scenario["formula"],
                    "message": None,
                }
            )
        record["threshold_outcomes_json"] = json.dumps(
            outcomes, separators=(",", ":"), sort_keys=True
        )
    table = pa.Table.from_pylist(records, schema=_METRIC_SCHEMA).replace_schema_metadata(
        {
            b"geml_config_hash": config_hash.encode(),
            b"geml_runner_fingerprint": runner_fingerprint.encode(),
            b"geml_processing_wall_seconds": b"1.0",
            b"geml_peak_resident_memory_bytes": b"1024",
        }
    )
    pq.write_table(table, shard, compression="zstd")
    checksum = sha256_file(shard)
    manifest = {
        "schema_version": "geml-goal2-manifest-v1",
        "metric_schema_version": "geml-goal2-metrics-v1",
        "metric_schema_sha256": _METRIC_SCHEMA_SHA256,
        "stage": stage,
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "config_hash": config_hash,
        "runner_fingerprint": runner_fingerprint,
        "input_manifest_sha256": "f" * 64,
        "processed_count": len(rows),
        "count_success_count": int(rows["count_status"].eq("success").sum()),
        "failure_count": int(rows["count_status"].ne("success").sum()),
        "semantic_audited_count": int(rows["semantic_selected"].sum()),
        "semantic_valid_count": int(rows["semantic_status"].eq("passed").sum()),
        "elapsed_seconds": 1.0,
        "resumed_from_partial": False,
        "shards": [
            {
                "path": shard.name,
                "row_count": len(rows),
                "byte_count": shard.stat().st_size,
                "checksum": {"algorithm": "sha256", "digest": checksum},
                "schema_sha256": _METRIC_SCHEMA_SHA256,
                "config_hash": config_hash,
                "runner_fingerprint": runner_fingerprint,
                "processing_wall_seconds": 1.0,
                "peak_resident_memory_bytes": 1024,
            }
        ],
        "run_metadata": {
            "stage": stage,
            "compiler_mode": CompilerMode.OFFICIAL_V4.value,
            "config_hash": config_hash,
            "runner_fingerprint": runner_fingerprint,
            "input_manifest_sha256": "f" * 64,
            "source_label": source_label,
            "threshold_scenarios": scenarios,
            "semantic_policy": {
                "backends": ["mpmath", "numpy_complex128"],
                "probe_count": 1,
            },
            "elapsed_seconds": 1.0,
            "elapsed_scope": "fixture metric aggregation",
            "metric_aggregation_elapsed_seconds": 1.0,
            "metric_aggregation_newly_processed_count": len(rows),
            "metric_aggregation_throughput_rows_per_second": float(len(rows)),
            "cumulative_shard_processing_wall_seconds": 1.0,
            "cumulative_shard_processing_scope": "fixture shard processing",
            "processing_throughput_rows_per_second": float(len(rows)),
            "resumed_from_partial": False,
            "checkpoint_reuse_count": 0,
            "orphan_recovery_count": 0,
            "throughput_rows_per_second": float(len(rows)),
            "peak_resident_memory_bytes": 1024,
            "peak_resident_memory_scope": "fixture process tree RSS",
            "provisional": False,
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_config(
    path: Path,
    *,
    pilot_count: int,
    final_count: int,
    pilot_label: str,
    final_label: str,
) -> Path:
    raw = yaml.safe_load(Path("configs/goal2_final.yaml").read_text(encoding="utf-8"))
    raw["stages"]["pilot"].update(
        expected_count=pilot_count,
        source_label=pilot_label,
    )
    raw["stages"]["final"].update(
        expected_count=final_count,
        source_label=final_label,
    )
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_denominators_quantiles_thresholds_and_bucket_boundaries() -> None:
    _, raw = _frames()
    frame = _enrich(raw)
    summary = summarize_population(frame, quantile_method="linear", population="all_processed")
    assert summary["all_processed_count"] == 5
    assert summary["valid_alpha_count"] == 4
    assert summary["invalid_alpha_count"] == 1
    assert summary["alpha_median"] == 1.5
    assert summary["alpha_mean"] == 1.625
    empty = summarize_population(
        frame.assign(tree_alpha_value=None),
        quantile_method="linear",
        population="valid_alpha",
    )
    assert empty["population_count"] == 0
    assert empty["alpha_mean"] is None
    assert ast_size_bucket(8, ((1, 8), (9, 16))) == "1-8"
    assert ast_size_bucket(9, ((1, 8), (9, 16))) == "9-16"

    thresholds = build_threshold_table(frame, [_SCENARIO]).iloc[0]
    assert thresholds["all_processed_count"] == 4
    assert thresholds["valid_applicable_count"] == 3
    assert thresholds["pass_count"] == 1
    assert thresholds["valid_only_pass_rate"] == 1 / 3
    assert thresholds["all_processed_pass_rate"] == 1 / 4
    assert thresholds["inapplicable_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda values: values.clear(), "incomplete or unexpected"),
        (lambda values: values.append(values[0].copy()), "duplicate threshold outcome"),
        (
            lambda values: values[0].update(status="not_applicable", passed=None),
            "inconsistent with scope and alpha",
        ),
        (
            lambda values: values[0].update(threshold_value=2.0),
            "threshold metadata is corrupt",
        ),
    ],
)
def test_threshold_outcomes_are_exhaustively_reconciled(
    mutation: Callable[[list[dict[str, object]]], None], message: str
) -> None:
    _, raw = _frames()
    frame = _enrich(raw.iloc[[0]].copy())
    outcomes = json.loads(frame.iloc[0]["threshold_outcomes_json"])
    mutation(outcomes)
    frame.loc[frame.index[0], "threshold_outcomes_json"] = json.dumps(outcomes)
    with pytest.raises(Goal2ArtifactError, match=message):
        build_threshold_table(frame, [_SCENARIO])


def test_strata_failures_top_cases_and_survivorship_are_complete() -> None:
    _, raw = _frames()
    frame = _enrich(raw)
    strata = build_stratified_table(frame, quantile_method="linear", minimum_group_count=10)
    size_rows = strata.loc[strata["stratum"].eq("ast_size_bucket")]
    assert size_rows["all_processed_count"].sum() == 5
    assert size_rows["underpowered"].all()
    family_size_rows = strata.loc[strata["stratum"].eq("operator_family x ast_size_bucket")]
    assert family_size_rows["all_processed_count"].sum() == 5
    signature = strata.loc[strata["stratum"].eq("operator_signature")]
    assert len(signature) == 1
    assert signature.iloc[0]["all_processed_count"] == 5

    tables = build_failure_tables(frame, top_count=3)
    assert set(tables["failure_details"]["expression_id"]) == {
        f"{4:064x}",
        f"{5:064x}",
    }
    assert tables["failure_survivorship"]["terminal_issue_count"].sum() >= 2
    top_alpha = tables["top_explosions"].loc[
        tables["top_explosions"]["ranking"].eq("highest_alpha")
    ]
    assert top_alpha.iloc[0]["expression_id"] == f"{5:064x}"
    assert top_alpha["tree_alpha_value"].notna().all()
    tied = frame.copy()
    tied.loc[tied["expression_id"].eq(f"{3:064x}"), "tree_alpha_value"] = 3.0
    tied_top = build_failure_tables(tied, top_count=3)["top_explosions"]
    tied_top = tied_top.loc[tied_top["ranking"].eq("highest_alpha")]
    assert tied_top.iloc[:2]["expression_id"].tolist() == [f"{3:064x}", f"{5:064x}"]

    taxonomy = semantic_backend_status_counts(
        frame,
        expected_backends=("mpmath", "numpy_complex128"),
        expected_compiler_mode=CompilerMode.OFFICIAL_V4.value,
    )
    assert len(taxonomy) == 2 * len(ProbeStatus)
    mismatch = taxonomy.loc[taxonomy["probe_status"].eq("mismatch")]
    assert mismatch["probe_result_count"].tolist() == [1, 1]
    assert mismatch["backend_audited_expression_count"].tolist() == [4, 4]
    assert mismatch["selected_expression_incidence_rate"].tolist() == [1 / 4, 1 / 4]
    assert mismatch["materialized_expression_incidence_rate"].tolist() == [1 / 4, 1 / 4]
    assert mismatch["all_processed_expression_incidence_rate"].tolist() == [1 / 5, 1 / 5]
    assert mismatch["all_processed_rate_caveat"].str.contains("not a corpus-wide").all()


def test_analysis_and_plots_use_only_tiny_saved_tables(tmp_path: Path) -> None:
    pilot, final = _frames()
    pilot_label = "separate_fixture_pilot_not_final_subset"
    final_label = "fixture_final"
    config_path = _write_config(
        tmp_path / "goal2.yaml",
        pilot_count=len(pilot),
        final_count=len(final),
        pilot_label=pilot_label,
        final_label=final_label,
    )
    config_hash = load_goal2_config(config_path).config_hash
    pilot_path = _write_metrics(
        tmp_path / "pilot",
        pilot,
        stage="pilot",
        config_hash=config_hash,
        source_label=pilot_label,
    )
    final_path = _write_metrics(
        tmp_path / "final",
        final,
        stage="final",
        config_hash=config_hash,
        source_label=final_label,
    )
    analysis_path = run_analysis(
        metrics_manifest=final_path,
        pilot_manifest=pilot_path,
        config_path=config_path,
        output_dir=tmp_path / "analysis",
    )
    analysis = validate_analysis_manifest(analysis_path)
    assert set(analysis["tables"]) == {
        "alpha_histogram",
        "alpha_log_histogram",
        "failure_details",
        "failure_status_counts",
        "failure_summary",
        "failure_survivorship",
        "overall",
        "scaling",
        "scatter_sample",
        "semantic_backend_status_counts",
        "stability_distribution",
        "stability_family",
        "stability_overall",
        "stratified",
        "thresholds",
        "top_explosions",
    }
    overall = pd.read_parquet(analysis_path.parent / analysis["tables"]["overall"]["path"])
    final_all = overall.loc[
        overall["dataset"].eq("final") & overall["population"].eq("all_processed")
    ].iloc[0]
    assert final_all["semantic_unique_assignment_total"] == 4
    assert final_all["semantic_selected_unique_assignment_coverage_rate"] == 1.0
    strata = pd.read_parquet(analysis_path.parent / analysis["tables"]["stratified"]["path"])
    assert "canonical_ast_constant_category" in set(strata["stratum"])
    assert "canonical_ast_constant_count" in set(strata["stratum"])
    assert "source_constant_count" not in set(strata["stratum"])
    stability = pd.read_parquet(
        analysis_path.parent / analysis["tables"]["stability_overall"]["path"]
    )
    median_delta = stability.loc[
        stability["metric"].eq("alpha_median"), "delta_final_minus_pilot"
    ].item()
    assert median_delta == 0.5
    assert stability["shared_expression_id_count"].unique().tolist() == [len(pilot)]
    assert stability["pilot_expression_id_overlap_rate"].unique().tolist() == [1.0]
    assert stability["final_expression_id_overlap_rate"].unique().tolist() == [3 / 5]
    selected_issue = stability.loc[stability["metric"].eq("selected_terminal_issue_rate")].iloc[0]
    assert selected_issue["pilot_value"] == 1 / 3
    assert selected_issue["final_value"] == 1 / 4
    all_processed_issue = stability.loc[
        stability["metric"].eq("all_processed_terminal_issue_incidence")
    ].iloc[0]
    assert "not a semantic failure-rate comparison" in all_processed_issue["interpretation"]
    family_stability = pd.read_parquet(
        analysis_path.parent / analysis["tables"]["stability_family"]["path"]
    )
    assert {"pilot_median_rank", "final_median_rank", "median_rank_correlation"} <= set(
        family_stability
    )
    assert family_stability["shared_expression_id_count"].sum() == len(pilot)
    stability_distribution = pd.read_parquet(
        analysis_path.parent / analysis["tables"]["stability_distribution"]["path"]
    )
    for dataset in ("pilot_rate", "final_rate"):
        depth = stability_distribution.loc[stability_distribution["dimension"].eq("ast_depth")]
        assert depth[dataset].sum() == pytest.approx(1.0)
    assert "<null>" in set(
        stability_distribution.loc[stability_distribution["dimension"].eq("ast_depth"), "value"]
    )
    scaling = pd.read_parquet(analysis_path.parent / analysis["tables"]["scaling"]["path"])
    runtime = scaling.loc[scaling["relationship"].eq("runtime_vs_ast_size")]
    assert runtime.sort_values("x_order")["x_value"].tolist() == [
        "1-8",
        "9-16",
        "17-32",
        "missing",
    ]
    assert (
        run_analysis(
            metrics_manifest=final_path,
            pilot_manifest=pilot_path,
            config_path=config_path,
            output_dir=tmp_path / "analysis",
        )
        == analysis_path
    )

    plots_path = generate_plots(analysis_path)
    plots = validate_plot_manifest(plots_path)
    assert len(plots["artifacts"]) == 20
    assert {item["format"] for item in plots["artifacts"]} == {"png", "svg"}
    assert {item["plot_name"] for item in plots["artifacts"]} == {
        f"{index:02d}_{name}"
        for index, name in enumerate(
            (
                "raw_alpha_distribution",
                "log_alpha_distribution",
                "ast_vs_eml_nodes",
                "alpha_vs_ast_nodes",
                "alpha_vs_ast_depth",
                "family_alpha_comparison",
                "threshold_pass_rates",
                "failure_status_distribution",
                "pilot_final_stability",
                "runtime_scaling",
            ),
            start=1,
        )
    }
    stability_svg = plots_path.parent / "09_pilot_final_stability.svg"
    assert pilot_label in stability_svg.read_text(encoding="utf-8")
    assert generate_plots(analysis_path) == plots_path

    analysis_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    bad_analysis_path = analysis_path.parent / "bad-analysis-manifest.json"
    missing_table = json.loads(json.dumps(analysis_payload))
    missing_table["tables"].pop("overall")
    bad_analysis_path.write_text(json.dumps(missing_table), encoding="utf-8")
    with pytest.raises(Goal2ArtifactError, match="table set is incomplete or unexpected"):
        validate_analysis_manifest(bad_analysis_path)
    stale_analysis = json.loads(json.dumps(analysis_payload))
    stale_analysis["analysis_source_fingerprint"] = "0" * 64
    bad_analysis_path.write_text(json.dumps(stale_analysis), encoding="utf-8")
    with pytest.raises(Goal2ArtifactError, match="source fingerprint differs"):
        validate_analysis_manifest(bad_analysis_path)

    plot_payload = json.loads(plots_path.read_text(encoding="utf-8"))
    bad_plot_path = plots_path.parent / "bad-plot-manifest.json"
    duplicated_plot = json.loads(json.dumps(plot_payload))
    duplicated_plot["artifacts"] = [duplicated_plot["artifacts"][0]] * 20
    bad_plot_path.write_text(json.dumps(duplicated_plot), encoding="utf-8")
    with pytest.raises(Goal2ArtifactError, match="duplicate or unexpected"):
        validate_plot_manifest(bad_plot_path)
    stale_plot = json.loads(json.dumps(plot_payload))
    stale_plot["plot_source_fingerprint"] = "0" * 64
    bad_plot_path.write_text(json.dumps(stale_plot), encoding="utf-8")
    with pytest.raises(Goal2ArtifactError, match="source fingerprint differs"):
        validate_plot_manifest(bad_plot_path)


def test_incompatible_metric_configs_and_missing_plot_source_are_rejected(
    tmp_path: Path,
) -> None:
    pilot, final = _frames()
    config_path = _write_config(
        tmp_path / "goal2.yaml",
        pilot_count=len(pilot),
        final_count=len(final),
        pilot_label="pilot",
        final_label="final",
    )
    config_hash = load_goal2_config(config_path).config_hash
    pilot_path = _write_metrics(
        tmp_path / "pilot", pilot, stage="pilot", config_hash="0" * 64, source_label="pilot"
    )
    final_path = _write_metrics(
        tmp_path / "final", final, stage="final", config_hash=config_hash, source_label="final"
    )
    with pytest.raises(Goal2ArtifactError, match="config hashes are incompatible"):
        run_analysis(
            metrics_manifest=final_path,
            pilot_manifest=pilot_path,
            config_path=config_path,
            output_dir=tmp_path / "rejected",
        )

    compatible_pilot = _write_metrics(
        tmp_path / "pilot-compatible",
        pilot,
        stage="pilot",
        config_hash=config_hash,
        source_label="pilot",
    )
    analysis_path = run_analysis(
        metrics_manifest=final_path,
        pilot_manifest=compatible_pilot,
        config_path=config_path,
        output_dir=tmp_path / "analysis",
    )
    analysis = validate_analysis_manifest(analysis_path)
    source = analysis_path.parent / analysis["tables"]["scaling"]["path"]
    source.unlink()
    with pytest.raises(Goal2ArtifactError, match="missing analysis table"):
        generate_plots(analysis_path)


def test_analysis_rejects_swapped_roles_wrong_labels_and_wrong_counts(tmp_path: Path) -> None:
    pilot, final = _frames()
    config_path = _write_config(
        tmp_path / "goal2.yaml",
        pilot_count=len(pilot),
        final_count=len(final),
        pilot_label="configured_pilot",
        final_label="configured_final",
    )
    config_hash = load_goal2_config(config_path).config_hash
    pilot_path = _write_metrics(
        tmp_path / "pilot",
        pilot,
        stage="pilot",
        config_hash=config_hash,
        source_label="configured_pilot",
    )
    final_path = _write_metrics(
        tmp_path / "final",
        final,
        stage="final",
        config_hash=config_hash,
        source_label="configured_final",
    )
    with pytest.raises(Goal2ArtifactError, match="final analysis input has stage"):
        run_analysis(
            metrics_manifest=pilot_path,
            pilot_manifest=final_path,
            config_path=config_path,
            output_dir=tmp_path / "swapped",
        )

    wrong_label = _write_metrics(
        tmp_path / "wrong-label",
        final,
        stage="final",
        config_hash=config_hash,
        source_label="wrong_final",
    )
    with pytest.raises(Goal2ArtifactError, match="source_label differs"):
        run_analysis(
            metrics_manifest=wrong_label,
            pilot_manifest=pilot_path,
            config_path=config_path,
            output_dir=tmp_path / "wrong-label-output",
        )

    corrupt_constants = final.copy()
    corrupt_constants.loc[corrupt_constants.index[0], "source_constant_count"] = 1
    corrupt_constants_path = _write_metrics(
        tmp_path / "corrupt-constants",
        corrupt_constants,
        stage="final",
        config_hash=config_hash,
        source_label="configured_final",
    )
    with pytest.raises(Goal2ArtifactError, match="constant count does not reconcile"):
        run_analysis(
            metrics_manifest=corrupt_constants_path,
            pilot_manifest=pilot_path,
            config_path=config_path,
            output_dir=tmp_path / "corrupt-constants-output",
        )

    wrong_count_config = _write_config(
        tmp_path / "wrong-count.yaml",
        pilot_count=len(pilot),
        final_count=len(final) + 1,
        pilot_label="configured_pilot",
        final_label="configured_final",
    )
    wrong_count_hash = load_goal2_config(wrong_count_config).config_hash
    wrong_count_pilot = _write_metrics(
        tmp_path / "wrong-count-pilot",
        pilot,
        stage="pilot",
        config_hash=wrong_count_hash,
        source_label="configured_pilot",
    )
    wrong_count_final = _write_metrics(
        tmp_path / "wrong-count-final",
        final,
        stage="final",
        config_hash=wrong_count_hash,
        source_label="configured_final",
    )
    with pytest.raises(Goal2ArtifactError, match="count differs from configured"):
        run_analysis(
            metrics_manifest=wrong_count_final,
            pilot_manifest=wrong_count_pilot,
            config_path=wrong_count_config,
            output_dir=tmp_path / "wrong-count-output",
        )
