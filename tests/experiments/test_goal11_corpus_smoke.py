"""Tiny fresh-clone tests for the Goal 11 workshop manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from geml.experiments.goal11.corpus_v3 import (
    ArtifactFormat,
    CompletenessState,
    DeferredExperiment,
    ExpectedArtifact,
    Goal11ManifestConfig,
    HardwareMetadata,
    PackageVersion,
    RunMetadata,
    WorkshopManifestValidationError,
    assemble_workshop_manifest,
    audit_workshop_manifest,
    canonical_json_bytes,
    load_manifest_config,
    manifest_sha256,
    require_valid_workshop_manifest,
    run_manifest_audit,
    sha256_file,
)

_SEEDS = (20260726, 20260727, 20260728)


def _artifact(
    artifact_id: str,
    path: str | None,
    *,
    required: bool = True,
    expected_sha256: str | None = None,
    expected_schema: str | None = "fixture-v1",
    expected_count: int | None = 2,
    configured_state: CompletenessState | None = None,
    configured_reason: str | None = None,
) -> ExpectedArtifact:
    return ExpectedArtifact(
        artifact_id=artifact_id,
        producer_issue="fixture",
        category="result_table",
        roles=("fixed_scale", "manifest"),
        required=required,
        relative_path=path,
        artifact_format=ArtifactFormat.JSON,
        expected_sha256=expected_sha256,
        expected_schema_version=expected_schema,
        expected_count=expected_count,
        count_field="row_count",
        configured_state=configured_state,
        configured_reason=configured_reason,
    )


def _config(*artifacts: ExpectedArtifact) -> Goal11ManifestConfig:
    return Goal11ManifestConfig(
        expected_seeds=_SEEDS,
        artifacts=artifacts,
        deferred_experiments=(
            DeferredExperiment(
                experiment_id="scaling_10x_to_100x",
                reason="outside the fixed workshop scale",
            ),
        ),
    )


def _write_fixture(path: Path, *, rows: int = 2, schema: str = "fixture-v1") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": schema, "row_count": rows}, sort_keys=True),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_manifest_is_authenticated_and_deterministic(tmp_path: Path):
    expected_hash = _write_fixture(tmp_path / "results.json")
    config = _config(_artifact("results", "results.json", expected_sha256=expected_hash))

    first = assemble_workshop_manifest(config, tmp_path, config_sha256="a" * 64)
    second = assemble_workshop_manifest(config, tmp_path, config_sha256="a" * 64)
    audit = audit_workshop_manifest(first, tmp_path)

    assert first == second
    assert manifest_sha256(first) == manifest_sha256(second)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first.artifacts[0].state is CompletenessState.COMPLETE
    assert audit.valid
    require_valid_workshop_manifest(audit)


@pytest.mark.parametrize(
    ("mutator", "expected_state"),
    [
        (lambda _: None, CompletenessState.MISSING),
        (
            lambda path: _write_fixture(path, schema="wrong-v1"),
            CompletenessState.SCHEMA_MISMATCH,
        ),
        (
            lambda path: _write_fixture(path, rows=3),
            CompletenessState.FAILED,
        ),
    ],
)
def test_missing_schema_and_count_failures_are_distinct(
    tmp_path: Path,
    mutator,
    expected_state: CompletenessState,
):
    path = tmp_path / "results.json"
    mutator(path)
    config = _config(_artifact("results", "results.json"))
    manifest = assemble_workshop_manifest(config, tmp_path, config_sha256="a" * 64)
    audit = audit_workshop_manifest(manifest, tmp_path)

    assert manifest.artifacts[0].state is expected_state
    assert not audit.valid
    with pytest.raises(WorkshopManifestValidationError) as raised:
        require_valid_workshop_manifest(audit)
    assert raised.value.audit == audit


def test_checksum_mutation_after_assembly_is_detected(tmp_path: Path):
    path = tmp_path / "results.json"
    expected_hash = _write_fixture(path)
    manifest = assemble_workshop_manifest(
        _config(_artifact("results", "results.json", expected_sha256=expected_hash)),
        tmp_path,
        config_sha256="a" * 64,
    )
    path.write_text('{"schema_version":"fixture-v1","row_count":2,"changed":true}')

    audit = audit_workshop_manifest(manifest, tmp_path)

    assert audit.items[0].effective_state is CompletenessState.CHECKSUM_MISMATCH
    assert not audit.valid


def test_failed_unsupported_and_optional_missing_are_retained(tmp_path: Path):
    config = _config(
        _artifact(
            "failed",
            None,
            configured_state=CompletenessState.FAILED,
            configured_reason="producer attempt failed",
        ),
        _artifact(
            "unsupported",
            None,
            configured_state=CompletenessState.UNSUPPORTED,
            configured_reason="contract decision pending",
        ),
        _artifact("optional", None, required=False),
    )
    manifest = assemble_workshop_manifest(config, tmp_path, config_sha256="b" * 64)
    audit = audit_workshop_manifest(manifest, tmp_path)

    assert [item.state for item in manifest.artifacts] == [
        CompletenessState.FAILED,
        CompletenessState.UNSUPPORTED,
        CompletenessState.MISSING,
    ]
    assert audit.required_error_count == 2
    assert all(item.reason for item in manifest.artifacts)


def test_complete_run_artifact_retains_resource_and_failure_metadata(tmp_path: Path):
    expected_hash = _write_fixture(tmp_path / "run.json")
    metadata = RunMetadata(
        config_sha256="d" * 64,
        git_commit="fixture-commit",
        seeds=_SEEDS,
        package_versions=(PackageVersion(name="geml", version="0.1.0"),),
        hardware=HardwareMetadata(
            platform="fixture",
            cpu="fixture-cpu",
            accelerator="fixture-gpu",
            accelerator_count=2,
        ),
        precision="bf16",
        parameter_count=100,
        flops=200.0,
        wall_time_seconds=3.0,
        peak_host_memory_bytes=400,
        peak_gpu_memory_bytes=500,
        gpu_hours=0.001,
        attempted_count=10,
        success_count=7,
        failure_count=1,
        invalid_count=0,
        unsupported_count=1,
        timeout_count=1,
        reproduction_command="python -m fixture",
    )
    artifact = _artifact(
        "run",
        "run.json",
        expected_sha256=expected_hash,
    ).model_copy(update={"run_metadata_required": True, "run_metadata": metadata})

    manifest = assemble_workshop_manifest(
        _config(artifact),
        tmp_path,
        config_sha256="e" * 64,
    )

    assert manifest.artifacts[0].state is CompletenessState.COMPLETE
    assert manifest.artifacts[0].run_metadata == metadata
    assert metadata.failure_count == 1

    wrong_seed_artifact = artifact.model_copy(
        update={
            "requires_expected_seeds": True,
            "run_metadata": metadata.model_copy(update={"seeds": (1, 2, 3)}),
        }
    )
    wrong_seed_manifest = assemble_workshop_manifest(
        _config(wrong_seed_artifact),
        tmp_path,
        config_sha256="e" * 64,
    )
    assert wrong_seed_manifest.artifacts[0].state is CompletenessState.FAILED
    assert "seed set differs" in (wrong_seed_manifest.artifacts[0].reason or "")


def test_contract_rejects_duplicates_bad_seeds_and_unsafe_paths():
    artifact = _artifact("same", "same.json")
    with pytest.raises(ValidationError, match="artifact IDs must be unique"):
        _config(artifact, artifact)
    with pytest.raises(ValidationError, match="three distinct seeds"):
        Goal11ManifestConfig(
            expected_seeds=(1, 1, 2),
            artifacts=(),
            deferred_experiments=(),
        )
    with pytest.raises(ValidationError, match="canonical relative POSIX"):
        _artifact("unsafe", "../escape.json")
    with pytest.raises(ValidationError, match="path-backed expected_count"):
        ExpectedArtifact(
            artifact_id="unobservable-count",
            producer_issue="fixture",
            category="dataset",
            roles=("manifest",),
            required=True,
            relative_path="opaque.bin",
            artifact_format=ArtifactFormat.OPAQUE,
            expected_count=2,
        )


def test_config_loads_and_deferred_is_not_an_artifact_state(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_config().model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    loaded = load_manifest_config(config_path)

    assert loaded.expected_seeds == _SEEDS
    assert loaded.deferred_experiments[0].state is CompletenessState.DEFERRED
    with pytest.raises(ValidationError, match="missing, failed, or unsupported"):
        ExpectedArtifact(
            artifact_id="bad",
            producer_issue="fixture",
            category="result_table",
            roles=("manifest",),
            required=True,
            relative_path=None,
            artifact_format=ArtifactFormat.JSON,
            configured_state=CompletenessState.DEFERRED,
            configured_reason="not allowed here",
        )


def test_checked_in_manifest_contract_is_honest_and_production_pending():
    root = Path(__file__).resolve().parents[2]
    config = load_manifest_config(root / "configs" / "goal11_corpus_v3.yaml")
    manifest = assemble_workshop_manifest(
        config,
        root,
        config_sha256="f" * 64,
    )
    audit = audit_workshop_manifest(manifest, root)

    assert config.expected_seeds == _SEEDS
    assert all(item.relative_path is None for item in config.artifacts)
    assert any(item.state is CompletenessState.UNSUPPORTED for item in manifest.artifacts)
    assert all(
        item.state in {CompletenessState.MISSING, CompletenessState.UNSUPPORTED}
        for item in manifest.artifacts
    )
    assert not audit.valid
    assert {item.experiment_id for item in manifest.deferred_experiments} >= {
        "production_corpus_v2",
        "production_corpus_v3",
        "scaling_10x_to_100x",
        "scaling_law_fit_or_extrapolation",
    }


def test_strict_runner_writes_audit_before_raising(tmp_path: Path):
    config = _config(_artifact("missing", "absent.json"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "out" / "manifest.json"
    audit_path = tmp_path / "out" / "audit.json"

    with pytest.raises(WorkshopManifestValidationError):
        run_manifest_audit(
            config_path,
            tmp_path,
            manifest_path,
            audit_path,
        )

    assert manifest_path.is_file()
    assert audit_path.is_file()
    assert json.loads(audit_path.read_text())["valid"] is False
    assert sha256_file(config_path)[0] == hashlib.sha256(config_path.read_bytes()).hexdigest()
