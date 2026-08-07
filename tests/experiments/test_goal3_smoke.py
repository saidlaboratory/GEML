"""Fresh-clone tests for the audited, resumable Goal 3 metric pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

import geml.experiments.goal2.run as goal2_run
import geml.experiments.goal3.run as goal3_run
from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.data.storage.manifests import (
    build_corpus_manifest,
    build_split_manifest,
    write_manifest_bundle,
)
from geml.data.storage.shards import write_shards
from geml.experiments.goal3.run import (
    CONSTRUCTION_PATH,
    Goal3ArtifactError,
    Goal3Stage,
    iter_metric_tables,
    load_goal3_config,
    load_goal3_summary,
    process_expression_record,
    run_goal3_stage,
    validate_goal3_manifest,
)

_RATIO_ORIENTATION = (
    "raw=eml_tree/ast_tree;dag_tree=eml_dag/ast_tree;"
    "dag_dag=eml_dag/ast_dag;ast_compression=ast_tree/ast_dag;"
    "eml_compression=eml_tree/eml_dag"
)
_SPLIT_COUNTS = {
    CorpusSplit.TRAIN: 7,
    CorpusSplit.VALIDATION: 6,
    CorpusSplit.TEST_IID: 6,
    CorpusSplit.TEST_OOD: 6,
}


def _expression_id(srepr: str, domain_mode: str = "safe_real") -> str:
    payload = f"geml-expression-v1\0{domain_mode}\0{srepr}".encode()
    return hashlib.sha256(payload).hexdigest()


def _record(
    index: int,
    split: CorpusSplit,
    *,
    srepr: str | None = None,
) -> ExpressionRecord:
    source = (
        srepr
        if srepr is not None
        else (
            "Symbol('x', real=True)"
            if index == 0
            else f"Add(Symbol('x', real=True), Integer({index}))"
        )
    )
    operator_counts = (
        {"symbol": 1} if index == 0 and srepr is None else {"add": 1, "integer": 1, "symbol": 1}
    )
    return ExpressionRecord(
        expression_id=_expression_id(source),
        sympy_srepr=source,
        display_text=f"x + {index}",
        latex_text=None,
        split=split,
        operator_family="algebraic_core",
        domain_mode="safe_real",
        variables=("x",),
        target_ast_size=3,
        target_depth=1,
        generator_seed=index + 100,
        generator_metadata={
            "operator_counts": operator_counts,
            "fixture": "goal3-count-25",
        },
    )


def _fixture_records() -> list[ExpressionRecord]:
    records: list[ExpressionRecord] = []
    index = 0
    for split, count in _SPLIT_COUNTS.items():
        records.extend(_record(index + offset, split) for offset in range(count))
        index += count
    assert len(records) == 25
    assert len({record.expression_id for record in records}) == 25
    return records


def _write_fixture_corpus(root: Path) -> Path:
    run_root = root / "goal1-fixture"
    source_config = root / "fixture-generator.yaml"
    source_config.write_text("fixture: goal3-count-25\n", encoding="utf-8")
    records = _fixture_records()
    split_manifests = []
    for split in CorpusSplit:
        split_records = [record for record in records if record.split is split]
        shard_manifests = write_shards(
            split_records,
            run_root / "data" / split.value,
            corpus_id="goal3-test-corpus",
            split=split,
            schema_version="geml-corpus-v1",
            minimum_rows=1,
            maximum_rows=25,
            allow_small_fixture=True,
            manifest_root=run_root,
        )
        split_manifests.append(build_split_manifest(shard_manifests))
    manifest = build_corpus_manifest(
        split_manifests,
        corpus_id="goal3-test-corpus",
        schema_version="geml-corpus-v1",
        config_path=source_config,
        generator_seed=20260723,
        git_commit="fixture-commit",
        created_at=datetime(2026, 7, 23, tzinfo=UTC),
        package_names=("geml",),
    )
    bundle = write_manifest_bundle(
        manifest,
        run_root / "manifests",
        artifact_root=run_root,
        config_path=source_config,
    )
    return bundle.corpus_manifest


def _write_runner_config(
    root: Path,
    *,
    manifest_path: Path,
    output_root: Path,
    worker_processes: int = 1,
) -> Path:
    payload = {
        "schema_version": "geml-goal3-config-v1",
        "output_root": output_root.as_posix(),
        "compiler_mode": "official_v4",
        "construction_path": "direct_hashcons",
        "stages": {
            "smoke": {
                "manifest": manifest_path.as_posix(),
                "source_label": "temporary_count_25",
                "expected_count": 25,
                "row_limit": None,
            }
        },
        "input_validation": {
            "require_manifest_sidecars": True,
            "require_qa_pass": False,
            "require_unique_expression_ids": True,
        },
        "processing": {
            "worker_processes": worker_processes,
            "worker_batch_size": 4,
            "worker_chunksize": 1,
            "parquet_row_group_size": 4,
            "resume": True,
            "atomic_finalization": True,
        },
        "audit": {"require_ready": True},
        "telemetry": {
            "package_versions": [
                "geml",
                "mpmath",
                "psutil",
                "pyarrow",
                "pydantic",
                "sympy",
            ],
            "scale_checkpoints": [10, 25],
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
    config_path = root / f"{output_root.name}.goal3.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.fixture
def fixture_manifest(tmp_path: Path) -> Path:
    return _write_fixture_corpus(tmp_path)


def test_production_config_loads_and_pins_safe_construction() -> None:
    loaded = load_goal3_config("configs/goal3_final.yaml")
    assert loaded.config.compiler_mode == "official_v4"
    assert loaded.config.construction_path == CONSTRUCTION_PATH
    assert loaded.config.stages["final"].expected_count == 250_000
    assert loaded.config.audit.require_ready is True
    assert "mpmath" in loaded.config.telemetry.package_versions


def test_process_record_reports_exact_ratios_without_materializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(0, CorpusSplit.TRAIN)

    def forbidden_materialization(*args: object, **kwargs: object) -> None:
        raise AssertionError("raw EML materialization is forbidden in Goal 3")

    monkeypatch.setattr(goal2_run, "materialize_ast_official", forbidden_materialization)
    row = process_expression_record(
        record,
        shard_id="fixture:train:0",
        shard_path="data/train.parquet",
        input_row_index=0,
    )
    assert row["status"] == "success"
    assert row["raw_tree_alpha_exact"] == "1/1"
    assert row["dag_alpha_vs_ast_tree_exact"] == "1/1"
    assert row["ast_compression_exact"] == "1/1"
    assert row["eml_compression_exact"] == "1/1"
    assert row["construction_path"] == CONSTRUCTION_PATH
    assert row["compiler_mode"] == "official_v4"


def test_repeated_source_subtree_has_exact_ast_reuse_metrics() -> None:
    source = "Add(Symbol('x', real=True), Symbol('x', real=True))"
    record = _record(30, CorpusSplit.TRAIN, srepr=source)
    row = process_expression_record(
        record,
        shard_id="fixture:train:0",
        shard_path="data/train.parquet",
        input_row_index=0,
    )
    assert row["status"] == "success"
    assert row["ast_tree_node_count"] == 3
    assert row["ast_dag_node_count"] == 2
    assert row["ast_compression_exact"] == "3/2"
    assert row["ast_dag_reused_node_count"] == 1
    assert row["ast_dag_child_reference_overhead"] == 1
    assert row["ast_dag_sharing_concentration_exact"] == "1/1"


def test_processing_failure_is_retained_as_an_explicit_row() -> None:
    record = _record(31, CorpusSplit.TRAIN, srepr="DefinitelyNotSrepr")
    row = process_expression_record(
        record,
        shard_id="fixture:train:0",
        shard_path="data/train.parquet",
        input_row_index=0,
    )
    assert row["status"] == "failure"
    assert row["error_stage"] == "ast_build"
    assert row["error_type"]
    assert row["error_message"]
    assert row["raw_tree_alpha_exact"] is None


def test_count_25_fresh_clone_pipeline_publishes_every_row(
    tmp_path: Path,
    fixture_manifest: Path,
) -> None:
    config = _write_runner_config(
        tmp_path,
        manifest_path=fixture_manifest,
        output_root=tmp_path / "goal3-output",
        worker_processes=2,
    )
    result = run_goal3_stage(config, Goal3Stage.SMOKE)
    assert result.processed_count == 25
    assert result.success_count == 25
    assert result.failure_count == 0
    manifest = validate_goal3_manifest(result.manifest_path)
    assert manifest["processed_count"] == 25

    tables = list(iter_metric_tables(result.manifest_path))
    combined = pa.concat_tables(tables)
    assert combined.num_rows == 25
    rows = combined.to_pylist()
    assert len({row["expression_id"] for row in rows}) == 25
    assert all(row["status"] == "success" for row in rows)
    assert all(row["construction_path"] == CONSTRUCTION_PATH for row in rows)
    assert all(row["ast_dag_reused_node_count"] is not None for row in rows)
    assert all(row["eml_dag_reused_node_count"] is not None for row in rows)
    assert all(row["ast_dag_sharing_concentration_exact"] is not None for row in rows)
    assert all(row["eml_dag_sharing_concentration_exact"] is not None for row in rows)

    summary = load_goal3_summary(result.manifest_path)
    assert summary["processed_count"] == 25
    assert summary["status_counts"] == {"success": 25}


def test_resume_has_identical_deterministic_summary(
    tmp_path: Path,
    fixture_manifest: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumed_config = _write_runner_config(
        tmp_path,
        manifest_path=fixture_manifest,
        output_root=tmp_path / "resumed-output",
    )
    original_write_checkpoint = goal3_run._write_checkpoint
    calls = 0

    class InjectedInterruptionError(RuntimeError):
        pass

    def interrupt_after_checkpoint(*args: object, **kwargs: object) -> None:
        nonlocal calls
        original_write_checkpoint(*args, **kwargs)
        calls += 1
        if calls == 1:
            raise InjectedInterruptionError("simulated process interruption")

    monkeypatch.setattr(goal3_run, "_write_checkpoint", interrupt_after_checkpoint)
    with pytest.raises(InjectedInterruptionError):
        run_goal3_stage(resumed_config, Goal3Stage.SMOKE)
    monkeypatch.setattr(goal3_run, "_write_checkpoint", original_write_checkpoint)

    resumed = run_goal3_stage(resumed_config, Goal3Stage.SMOKE)
    assert resumed.resumed is True
    assert resumed.processed_count == 25

    uninterrupted_config = _write_runner_config(
        tmp_path,
        manifest_path=fixture_manifest,
        output_root=tmp_path / "uninterrupted-output",
    )
    uninterrupted = run_goal3_stage(uninterrupted_config, Goal3Stage.SMOKE)
    assert load_goal3_summary(resumed.manifest_path) == load_goal3_summary(
        uninterrupted.manifest_path
    )


def test_audit_gate_blocks_before_any_stage_artifact(
    tmp_path: Path,
    fixture_manifest: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "blocked-output"
    config = _write_runner_config(
        tmp_path,
        manifest_path=fixture_manifest,
        output_root=output_root,
    )

    def blocked_audit() -> tuple[dict[str, object], str]:
        raise Goal3ArtifactError("injected audit blocker")

    monkeypatch.setattr(goal3_run, "_audit_gate_payload", blocked_audit)
    with pytest.raises(Goal3ArtifactError, match="audit blocker"):
        run_goal3_stage(config, Goal3Stage.SMOKE)
    assert not (output_root / "smoke").exists()


def test_checkpoint_rejects_changed_output_binding(
    tmp_path: Path,
    fixture_manifest: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "checkpoint-mismatch-output"
    config = _write_runner_config(
        tmp_path,
        manifest_path=fixture_manifest,
        output_root=output_root,
    )
    original_write_checkpoint = goal3_run._write_checkpoint

    class InjectedInterruptionError(RuntimeError):
        pass

    def interrupt_after_checkpoint(*args: object, **kwargs: object) -> None:
        original_write_checkpoint(*args, **kwargs)
        raise InjectedInterruptionError("stop after first checkpoint")

    monkeypatch.setattr(goal3_run, "_write_checkpoint", interrupt_after_checkpoint)
    with pytest.raises(InjectedInterruptionError):
        run_goal3_stage(config, Goal3Stage.SMOKE)
    monkeypatch.setattr(goal3_run, "_write_checkpoint", original_write_checkpoint)

    checkpoint = next((output_root / "smoke" / "checkpoints").glob("*.json"))
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["output_shard"]["audit_fingerprint"] = "0" * 64
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Goal3ArtifactError, match="output-shard bindings mismatch"):
        run_goal3_stage(config, Goal3Stage.SMOKE)


def test_orphan_metric_and_telemetry_are_recovered(
    tmp_path: Path,
    fixture_manifest: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "orphan-output"
    config = _write_runner_config(
        tmp_path,
        manifest_path=fixture_manifest,
        output_root=output_root,
    )
    original_write_checkpoint = goal3_run._write_checkpoint
    calls = 0

    class InjectedInterruptionError(RuntimeError):
        pass

    def interrupt_before_checkpoint(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InjectedInterruptionError("metric and telemetry published, checkpoint absent")
        original_write_checkpoint(*args, **kwargs)

    monkeypatch.setattr(goal3_run, "_write_checkpoint", interrupt_before_checkpoint)
    with pytest.raises(InjectedInterruptionError):
        run_goal3_stage(config, Goal3Stage.SMOKE)
    assert list((output_root / "smoke" / "data").rglob("*.parquet"))
    assert list((output_root / "smoke" / "telemetry").rglob("*.json"))
    assert not list((output_root / "smoke" / "checkpoints").glob("*.json"))

    monkeypatch.setattr(goal3_run, "_write_checkpoint", original_write_checkpoint)
    recovered = run_goal3_stage(config, Goal3Stage.SMOKE)
    assert recovered.resumed is True
    assert recovered.processed_count == 25
    metadata = json.loads((recovered.output_root / "run.metadata.json").read_text(encoding="utf-8"))
    assert metadata["orphan_recovery_count"] == 1


def test_corrupt_metric_shard_is_rejected(
    tmp_path: Path,
    fixture_manifest: Path,
) -> None:
    config = _write_runner_config(
        tmp_path,
        manifest_path=fixture_manifest,
        output_root=tmp_path / "corrupt-output",
    )
    result = run_goal3_stage(config, Goal3Stage.SMOKE)
    manifest = validate_goal3_manifest(result.manifest_path)
    shard_path = result.manifest_path.parent / manifest["shards"][0]["path"]
    with shard_path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(Goal3ArtifactError, match="checksum mismatch"):
        validate_goal3_manifest(result.manifest_path)
