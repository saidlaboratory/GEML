"""Tiny-fixture integration tests for the production Goal 2 runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import geml.experiments.goal2.run as goal2_run
from geml.analysis.goal2.alpha import (
    ThresholdScenario,
    ThresholdStatus,
    calculate_threshold,
    calculate_tree_alpha,
    evaluate_threshold,
    strict_threshold_pass,
)
from geml.contracts.ast import ASTEdge, ASTNode, ASTStatistics, ASTTree
from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.data.storage.manifests import (
    build_corpus_manifest,
    build_split_manifest,
    write_manifest_bundle,
)
from geml.data.storage.shards import sha256_file, write_shards
from geml.eml.compiler_core import CompilerMode
from geml.eml.counting import (
    count_eml_divide,
    count_eml_negate,
    count_eml_subtract,
    count_variable,
)
from geml.experiments.goal2.run import (
    Goal2ArtifactError,
    Goal2Stage,
    UnsupportedASTOperatorError,
    _probe_samples,
    count_ast_official,
    load_goal2_config,
    run_goal2_stage,
    validate_metrics_manifest,
)

_SOURCES = (
    ("Symbol('x', real=True)", "algebraic_core", "safe_real", ("x",)),
    ("Integer(1)", "algebraic_core", "safe_real", ()),
    ("Integer(-2)", "algebraic_core", "safe_real", ()),
    ("Rational(1, 2)", "powers_division_rationals", "safe_real", ()),
    (
        "Add(Symbol('x', real=True), Integer(1))",
        "algebraic_core",
        "safe_real",
        ("x",),
    ),
    (
        "Mul(Symbol('x', real=True), Integer(2))",
        "algebraic_core",
        "safe_real",
        ("x",),
    ),
    (
        "Pow(Symbol('x', real=True), Integer(2))",
        "powers_division_rationals",
        "safe_real",
        ("x",),
    ),
    (
        "Pow(Symbol('x', nonzero=True, real=True), Integer(-1))",
        "powers_division_rationals",
        "nonzero_real",
        ("x",),
    ),
    ("exp(Symbol('x', real=True))", "exp_log", "safe_real", ("x",)),
    ("log(Symbol('x', positive=True))", "exp_log", "positive_real", ("x",)),
    ("sin(Symbol('x', real=True))", "trig_hyperbolic", "safe_real", ("x",)),
    ("cos(Symbol('x', real=True))", "trig_hyperbolic", "safe_real", ("x",)),
    ("tan(Symbol('x', real=True))", "trig_hyperbolic", "safe_real", ("x",)),
    ("sinh(Symbol('x', real=True))", "trig_hyperbolic", "safe_real", ("x",)),
    ("cosh(Symbol('x', real=True))", "trig_hyperbolic", "safe_real", ("x",)),
    ("tanh(Symbol('x', real=True))", "trig_hyperbolic", "safe_real", ("x",)),
    (
        "Add(exp(Symbol('x', real=True)), sin(Symbol('x', real=True)))",
        "mixed_elementary",
        "safe_real",
        ("x",),
    ),
    (
        "Mul(log(Symbol('x', positive=True)), cos(Symbol('x', positive=True)))",
        "mixed_elementary",
        "positive_real",
        ("x",),
    ),
    (
        "Add(Pow(Symbol('x', real=True), Integer(2)), Integer(1))",
        "ood_stress",
        "safe_real",
        ("x",),
    ),
    (
        "Mul(exp(Symbol('x', real=True)), tanh(Symbol('x', real=True)))",
        "mixed_elementary",
        "safe_real",
        ("x",),
    ),
    ("Rational(-7, 5)", "powers_division_rationals", "safe_real", ()),
    (
        "Add(Symbol('x', real=True), Symbol('y', real=True))",
        "algebraic_core",
        "safe_real",
        ("x", "y"),
    ),
    (
        "Pow(Add(Symbol('x', real=True), Integer(1)), Integer(3))",
        "ood_stress",
        "safe_real",
        ("x",),
    ),
    (
        "exp(log(Symbol('x', positive=True)))",
        "exp_log",
        "positive_real",
        ("x",),
    ),
    ("Symbol('z', real=True)", "algebraic_core", "safe_real", ("z",)),
)


def _record(index: int, split: CorpusSplit) -> ExpressionRecord:
    source, family, domain, variables = _SOURCES[index]
    metadata: dict[str, object] = {"operator_counts": {"symbol": 1}}
    if index == len(_SOURCES) - 1:
        metadata = {"fixture_failure": True}
    return ExpressionRecord(
        expression_id=f"{index + 1:064x}",
        sympy_srepr=source,
        display_text="fixture display is never parsed",
        latex_text=None,
        split=split,
        operator_family=family,
        domain_mode=domain,
        variables=variables,
        target_ast_size=1,
        target_depth=0,
        generator_seed=index,
        generator_metadata=metadata,
    )


def _input_manifest(tmp_path: Path) -> Path:
    run_root = tmp_path / "goal1-fixture"
    corpus_id = "goal1-fixture-25"
    split_sizes = {
        CorpusSplit.TRAIN: 7,
        CorpusSplit.VALIDATION: 6,
        CorpusSplit.TEST_IID: 6,
        CorpusSplit.TEST_OOD: 6,
    }
    split_manifests = []
    cursor = 0
    for split, count in split_sizes.items():
        records = [_record(index, split) for index in range(cursor, cursor + count)]
        cursor += count
        shards = write_shards(
            records,
            run_root / "data" / split.value,
            corpus_id=corpus_id,
            split=split,
            schema_version="geml-expression-record-v1",
            minimum_rows=1,
            maximum_rows=25,
            allow_small_fixture=True,
            manifest_root=run_root,
        )
        split_manifests.append(build_split_manifest(shards))
    source_config = tmp_path / "goal1-fixture.yaml"
    source_config.write_text("schema_version: fixture-v1\n", encoding="utf-8")
    manifest = build_corpus_manifest(
        split_manifests,
        corpus_id=corpus_id,
        schema_version="geml-expression-record-v1",
        config_path=source_config,
        generator_seed=17,
        git_commit="fixture",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        package_names=("geml",),
    )
    return write_manifest_bundle(
        manifest,
        run_root / "manifests",
        artifact_root=run_root,
    ).corpus_manifest


def _runner_config(tmp_path: Path, manifest_path: Path) -> Path:
    stage = {
        "manifest": manifest_path.as_posix(),
        "source_label": "temporary_goal1_fixture_25",
        "expected_count": 25,
        "row_limit": None,
        "semantic_selection_modulus": 1,
    }
    config = {
        "schema_version": "geml-goal2-config-v1",
        "output_root": (tmp_path / "goal2-output").as_posix(),
        "primary_compiler_mode": "official_v4",
        "run_seed": 19,
        "stages": {name: stage for name in ("smoke", "pilot", "final")},
        "input_validation": {
            "require_manifest_sidecars": True,
            "require_qa_pass": False,
            "require_unique_expression_ids": True,
        },
        "metrics": {
            "output_shard_size": 25,
            "worker_processes": 1,
            "resume": True,
            "atomic_finalization": True,
        },
        "count_only": {"count_every_row": True, "materialize_for_counts": False},
        "materialization": {
            "maximum_nodes": 2000,
            "maximum_depth": 128,
            "maximum_construction_steps": 10000,
        },
        "semantic_audit": {
            "backends": ["mpmath", "numpy_complex128"],
            "probe_count": 1,
            "precision_digits": 50,
            "mpmath_absolute_tolerance": "1e-30",
            "mpmath_relative_tolerance": "1e-25",
            "numpy_absolute_tolerance": "1e-10",
            "numpy_relative_tolerance": "1e-9",
            "selection_hash": "sha256",
        },
        "threshold_scenarios": [
            {
                "name": "fixture_threshold",
                "scope_families": sorted({row[1] for row in _SOURCES}),
                "definition_status": "defined",
                "K": 1,
                "L": 1,
                "derivation": "fixture K=1 and L=1",
                "references": ["temporary test fixture"],
            }
        ],
        "telemetry": {"peak_resident_memory": True, "package_versions": []},
        "analysis": {
            "ast_size_buckets": [[1, 8], [9, 16]],
            "minimum_group_count": 1,
            "top_case_count": 5,
            "scatter_sample_size": 25,
            "sample_seed": 19,
            "quantile_method": "linear",
        },
    }
    path = tmp_path / "goal2-fixture.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _prepared_runner(tmp_path: Path) -> Path:
    return _runner_config(tmp_path, _input_manifest(tmp_path))


def _write_config_variant(
    source: Path,
    target: Path,
    update: Callable[[dict[str, object]], None],
) -> Path:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    update(config)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def _logical_ast(label: str, arity: int) -> ASTTree:
    nodes = [ASTNode(node_id="root", node_kind="operator", label=label, arity=arity)]
    edges = []
    for slot, name in enumerate(("x", "y")[:arity]):
        node_id = f"leaf-{slot}"
        nodes.append(
            ASTNode(
                node_id=node_id,
                node_kind="leaf",
                label="symbol",
                arity=0,
                value={"name": name, "assumptions": {"real": True}},
            )
        )
        edges.append(ASTEdge(source_id="root", target_id=node_id, child_slot=slot))
    return ASTTree(
        expression_id="logical-fixture",
        root_id="root",
        nodes=tuple(nodes),
        edges=tuple(edges),
        statistics=ASTStatistics(
            node_count=arity + 1,
            edge_count=arity,
            leaf_count=arity,
            operator_count=1,
            depth=1,
        ),
    )


def test_alpha_exact_ratio_and_strict_threshold_boundary() -> None:
    alpha = calculate_tree_alpha(3, 7)
    assert alpha.exact_ratio == "7/3"
    assert alpha.numerator == 7
    assert alpha.denominator == 3
    assert alpha.value == 7 / 3
    assert calculate_tree_alpha(None, 7).value is None
    assert calculate_tree_alpha(0, 7).value is None
    assert calculate_tree_alpha(True, 7).denominator is None
    assert calculate_tree_alpha(1, True).numerator is None

    threshold = calculate_threshold(1, 1)
    assert threshold == 1.0
    assert not strict_threshold_pass(threshold, threshold)
    scenario = ThresholdScenario(
        name="boundary",
        scope_families=("fixture",),
        derivation="K=1, L=1",
        references=("fixture",),
        operator_label_count=1,
        leaf_label_count=1,
    )
    outcome = evaluate_threshold(
        calculate_tree_alpha(1, 1),
        scenario,
        operator_family="fixture",
    )
    assert outcome.status is ThresholdStatus.FAILED


def test_probe_assignments_are_deterministic_and_distinct_when_possible() -> None:
    variable_record = _record(0, CorpusSplit.TRAIN)
    first = _probe_samples(variable_record, seed=19, count=5)
    second = _probe_samples(variable_record, seed=19, count=5)
    assert first == second
    assert len({sample.variables for sample in first}) == 5

    constant_record = _record(1, CorpusSplit.TRAIN)
    constant_samples = _probe_samples(constant_record, seed=19, count=2)
    assert len({sample.variables for sample in constant_samples}) == 1


def test_invalid_scientific_configuration_fails_before_processing(tmp_path: Path) -> None:
    source = _prepared_runner(tmp_path)

    def invalid_schema(config: dict[str, object]) -> None:
        config["schema_version"] = "typo"

    def empty_thresholds(config: dict[str, object]) -> None:
        config["threshold_scenarios"] = []

    def nonfinite_tolerance(config: dict[str, object]) -> None:
        semantic = config["semantic_audit"]
        assert isinstance(semantic, dict)
        semantic["mpmath_absolute_tolerance"] = "nan"

    def string_scope(config: dict[str, object]) -> None:
        scenarios = config["threshold_scenarios"]
        assert isinstance(scenarios, list)
        assert isinstance(scenarios[0], dict)
        scenarios[0]["scope_families"] = "algebraic_core"

    for index, update in enumerate(
        (invalid_schema, empty_thresholds, nonfinite_tolerance, string_scope)
    ):
        path = _write_config_variant(source, tmp_path / f"invalid-{index}.yaml", update)
        with pytest.raises(goal2_run.Goal2ConfigurationError):
            load_goal2_config(path)


def test_logical_lowering_labels_dispatch_to_frozen_exact_counters() -> None:
    mode = CompilerMode.OFFICIAL_V4
    x = count_variable("x", mode=mode)
    y = count_variable("y", mode=mode)
    expected = {
        "negate": count_eml_negate(x, mode=mode),
        "subtract": count_eml_subtract(x, y, mode=mode),
        "divide": count_eml_divide(x, y, mode=mode),
    }
    for label, counted in expected.items():
        tree = _logical_ast(label, 1 if label == "negate" else 2)
        assert count_ast_official(tree) == counted
    with pytest.raises(UnsupportedASTOperatorError, match="reserved_pending_operator"):
        count_ast_official(_logical_ast("reserved_pending_operator", 1))


def test_smoke_path_writes_one_official_row_per_input_and_resumes(tmp_path: Path) -> None:
    config_path = _prepared_runner(tmp_path)
    first = run_goal2_stage(config_path, Goal2Stage.SMOKE)
    assert first.processed_count == 25
    assert first.count_success_count == 24
    assert first.failure_count == 1
    assert not first.resumed

    manifest = validate_metrics_manifest(first.manifest_path)
    assert manifest["processed_count"] == 25
    assert manifest["compiler_mode"] == CompilerMode.OFFICIAL_V4.value
    tables = [pq.read_table(first.output_root / shard["path"]) for shard in manifest["shards"]]
    rows = [row for table in tables for row in table.to_pylist()]
    assert len(rows) == len({row["expression_id"] for row in rows}) == 25
    symbol = next(row for row in rows if row["expression_id"] == f"{1:064x}")
    assert symbol["tree_alpha_numerator"] == "1"
    assert symbol["tree_alpha_denominator"] == 1
    assert symbol["tree_alpha_exact_ratio"] == "1/1"
    probe_results = json.loads(symbol["semantic_probe_results_json"])
    assert len(probe_results) == 2
    assert {result["backend"] for result in probe_results} == {
        "mpmath",
        "numpy_complex128",
    }
    assert all(result["sample_label"] == "probe-000" for result in probe_results)
    assert symbol["semantic_unique_assignment_count"] == 1
    failure = next(row for row in rows if row["expression_id"] == f"{25:064x}")
    assert failure["processing_status"] == "record_contract_error"
    assert failure["tree_alpha_value"] is None
    failure_thresholds = json.loads(failure["threshold_outcomes_json"])
    assert len(failure_thresholds) == 1
    assert failure_thresholds[0]["status"] == "invalid_alpha"
    assert {row["operator_family"] for row in rows} == {row[1] for row in _SOURCES}

    resumed = run_goal2_stage(config_path, "smoke")
    assert resumed.resumed
    assert resumed.processed_count == 25


def test_backend_exception_retains_every_requested_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = goal2_run.audit_semantic_case

    def fail_numpy(*args: object, **kwargs: object) -> object:
        if kwargs.get("backend") is goal2_run.NumericBackend.NUMPY_COMPLEX128:
            raise RuntimeError("injected backend failure")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(goal2_run, "audit_semantic_case", fail_numpy)
    result = run_goal2_stage(_prepared_runner(tmp_path), "smoke")
    manifest = validate_metrics_manifest(result.manifest_path)
    rows = [
        row
        for shard in manifest["shards"]
        for row in pq.read_table(result.output_root / shard["path"]).to_pylist()
    ]
    symbol = next(row for row in rows if row["expression_id"] == f"{1:064x}")
    probes = json.loads(symbol["semantic_probe_results_json"])
    assert symbol["semantic_requested_count"] == 2
    assert symbol["semantic_pass_count"] == 1
    assert symbol["semantic_failure_count"] == 1
    assert len(probes) == 2
    assert {probe["backend"] for probe in probes} == {
        "mpmath",
        "numpy_complex128",
    }
    numpy_probe = next(probe for probe in probes if probe["backend"] == "numpy_complex128")
    assert numpy_probe["status"] == "eml_evaluation_error"
    assert "injected backend failure" in numpy_probe["message"]


def test_corrupt_completed_artifact_is_rejected(tmp_path: Path) -> None:
    config_path = _prepared_runner(tmp_path)
    result = run_goal2_stage(config_path, "smoke")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    shard_path = result.output_root / manifest["shards"][0]["path"]
    with shard_path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(Goal2ArtifactError, match="checksum mismatch"):
        run_goal2_stage(config_path, "smoke")


def test_manifest_shape_and_row_denominators_are_reconciled(tmp_path: Path) -> None:
    first = run_goal2_stage(_prepared_runner(tmp_path), "smoke")
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    shard = manifest["shards"][0]
    original_path = shard.pop("path")
    first.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Goal2ArtifactError, match="path metadata is invalid"):
        validate_metrics_manifest(first.manifest_path)

    shard["path"] = original_path
    shard_path = first.output_root / original_path
    table = pq.read_table(shard_path)
    rows = table.to_pylist()
    audited = next(
        row
        for row in rows
        if row["semantic_selected"] and row["materialization_status"] == "materialized"
    )
    audited["semantic_requested_count"] += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), shard_path)
    shard["byte_count"] = shard_path.stat().st_size
    shard["checksum"]["digest"] = sha256_file(shard_path)
    first.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Goal2ArtifactError, match="semantic denominator"):
        validate_metrics_manifest(first.manifest_path)


def test_manifest_reconstructs_threshold_formula_metadata(tmp_path: Path) -> None:
    result = run_goal2_stage(_prepared_runner(tmp_path), "smoke")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["run_metadata"]["threshold_scenarios"][0]["threshold_value"] += 0.125
    result.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Goal2ArtifactError, match="threshold metadata is not canonical"):
        validate_metrics_manifest(result.manifest_path)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("depth", "ast_depth"),
        ("processing_time", "processing time"),
        ("assignment", "assignment violates its domain"),
        ("probe_counts", "pass/failure counts"),
        ("stray_error", "failure record"),
    ],
)
def test_metric_row_scientific_fields_are_reconciled(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    result = run_goal2_stage(_prepared_runner(tmp_path), "smoke")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    shard = manifest["shards"][0]
    shard_path = result.output_root / shard["path"]
    table = pq.read_table(shard_path)
    rows = table.to_pylist()
    row = next(
        value
        for value in rows
        if value["count_status"] == "success"
        and value["materialization_status"] == "materialized"
        and value["semantic_status"] == "passed"
        and value["variable_count"]
    )
    if corruption == "depth":
        row["ast_depth"] = None
    elif corruption == "processing_time":
        row["processing_elapsed_seconds"] = -1.0
    elif corruption == "assignment":
        probes = json.loads(row["semantic_probe_results_json"])
        probes[0]["variable_assignments"][0][1] = "999"
        row["semantic_probe_results_json"] = json.dumps(
            probes, separators=(",", ":"), sort_keys=True
        )
    elif corruption == "probe_counts":
        row["semantic_pass_count"] -= 1
        row["semantic_failure_count"] += 1
    elif corruption == "stray_error":
        row["error_stage"] = "fixture"
        row["error_type"] = "FixtureError"
        row["error_message"] = "stray error"
    else:  # pragma: no cover - protected by the parametrization
        raise AssertionError(corruption)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), shard_path)
    shard["byte_count"] = shard_path.stat().st_size
    shard["checksum"]["digest"] = sha256_file(shard_path)
    result.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Goal2ArtifactError, match=message):
        validate_metrics_manifest(result.manifest_path)


def test_corrupt_checkpoint_is_rejected_before_resume(tmp_path: Path) -> None:
    config_path = _prepared_runner(tmp_path)
    result = run_goal2_stage(config_path, "smoke")
    result.manifest_path.unlink()
    result.summary_path.unlink()
    checkpoint = next((result.output_root / "checkpoints").glob("*.json"))
    checkpoint.write_text("{not-json", encoding="utf-8")
    with pytest.raises(Goal2ArtifactError, match="invalid checkpoint"):
        run_goal2_stage(config_path, "smoke")


def test_orphan_shard_and_each_partial_finalization_sidecar_resume(tmp_path: Path) -> None:
    config_path = _prepared_runner(tmp_path)
    first = run_goal2_stage(config_path, "smoke")
    checkpoint = next((first.output_root / "checkpoints").glob("*.json"))
    checkpoint.unlink()
    first.manifest_path.unlink()
    first.summary_path.unlink()
    (first.output_root / "run.metadata.json").unlink()

    recovered = run_goal2_stage(config_path, "smoke")
    assert recovered.resumed
    assert recovered.processed_count == 25

    recovered.manifest_path.unlink()
    recovered.summary_path.unlink()
    metadata_only = run_goal2_stage(config_path, "smoke")
    assert metadata_only.resumed

    metadata_only.manifest_path.unlink()
    (metadata_only.output_root / "run.metadata.json").unlink()
    summary_only = run_goal2_stage(config_path, "smoke")
    assert summary_only.resumed
    validate_metrics_manifest(summary_only.manifest_path)


def test_orphan_shard_with_mismatched_provenance_is_rejected(tmp_path: Path) -> None:
    config_path = _prepared_runner(tmp_path)
    result = run_goal2_stage(config_path, "smoke")
    checkpoint_path = next((result.output_root / "checkpoints").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    shard_path = result.output_root / checkpoint["output_shard"]["path"]
    table = pq.read_table(shard_path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"geml_runner_fingerprint"] = b"0" * 64
    pq.write_table(table.replace_schema_metadata(metadata), shard_path)
    checkpoint_path.unlink()
    result.manifest_path.unlink()
    result.summary_path.unlink()
    (result.output_root / "run.metadata.json").unlink()

    with pytest.raises(Goal2ArtifactError, match="orphan metric shard provenance mismatch"):
        run_goal2_stage(config_path, "smoke")


def test_dependency_change_during_processing_aborts_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _prepared_runner(tmp_path)
    dependency = tmp_path / "synthetic-dependency.py"
    dependency.write_text("VERSION = 1\n", encoding="utf-8")
    original_paths = goal2_run._runner_dependency_paths
    original_process = goal2_run._process_records

    def dependency_paths(
        loaded: goal2_run.LoadedGoal2Config,
    ) -> tuple[tuple[str, Path], ...]:
        return (*original_paths(loaded), ("synthetic-dependency.py", dependency))

    def process_then_edit(*args: object, **kwargs: object) -> object:
        result = original_process(*args, **kwargs)  # type: ignore[arg-type]
        dependency.write_text("VERSION = 2\n", encoding="utf-8")
        return result

    monkeypatch.setattr(goal2_run, "_runner_dependency_paths", dependency_paths)
    monkeypatch.setattr(goal2_run, "_process_records", process_then_edit)
    with pytest.raises(Goal2ArtifactError, match="fingerprint changed"):
        run_goal2_stage(config_path, "smoke")
    assert not list((tmp_path / "goal2-output").rglob("*.parquet"))


def test_spawn_workers_verify_the_same_executable_fingerprint(tmp_path: Path) -> None:
    source = _prepared_runner(tmp_path)

    def use_two_workers(config: dict[str, object]) -> None:
        metrics = config["metrics"]
        assert isinstance(metrics, dict)
        metrics["worker_processes"] = 2

    config_path = _write_config_variant(source, tmp_path / "workers.yaml", use_two_workers)
    result = run_goal2_stage(config_path, "smoke")
    assert result.processed_count == 25
    assert validate_metrics_manifest(result.manifest_path)["processed_count"] == 25
