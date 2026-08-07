"""Tests for corpus, run-metadata, and retained-error contracts."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from geml.contracts.corpus import (
    FINAL_CORPUS_SPLIT_COUNTS,
    FINAL_CORPUS_TOTAL_COUNT,
    ChecksumRecord,
    CorpusManifest,
    CorpusShardManifest,
    CorpusSplit,
    ErrorRow,
    RunMetadata,
    SplitManifest,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _shard(
    *,
    shard_id: str = "train-00000",
    shard_index: int = 0,
    split: str = "train",
    row_count: int = 2,
) -> CorpusShardManifest:
    return CorpusShardManifest(
        schema_version="1.0.0",
        corpus_id="tiny-fixture",
        shard_id=shard_id,
        path=f"{split}/{shard_id}.parquet",
        split=split,
        shard_index=shard_index,
        row_count=row_count,
        byte_count=512,
        checksum=ChecksumRecord(algorithm="sha256", digest="a" * 64),
        error_row_count=0,
        metadata={"fixture": True},
    )


def _split_manifest(*shards: CorpusShardManifest) -> SplitManifest:
    selected_shards = shards or (_shard(),)
    return SplitManifest(
        schema_version="1.0.0",
        corpus_id="tiny-fixture",
        split="train",
        shards=selected_shards,
        total_row_count=sum(shard.row_count for shard in selected_shards),
        total_error_row_count=sum(shard.error_row_count for shard in selected_shards),
    )


def _corpus_manifest() -> CorpusManifest:
    split = _split_manifest()
    return CorpusManifest(
        schema_version="1.0.0",
        corpus_id="tiny-fixture",
        splits=(split,),
        total_row_count=split.total_row_count,
        total_error_row_count=split.total_error_row_count,
        config_hash="config-sha256-example",
        generator_seed=1729,
        created_at=NOW,
        git_commit="0123456789abcdef",
        python_version="3.12.0",
        platform="test-platform",
        package_versions={"geml": "0.1.0"},
        metadata={"fixture": True},
    )


def _run_metadata() -> RunMetadata:
    return RunMetadata(
        run_id="run-0001",
        stage="corpus_generation",
        config_hash="config-sha256-example",
        random_seed=1729,
        git_commit="0123456789abcdef",
        python_version="3.12.0",
        platform="test-platform",
        package_versions={"geml": "0.1.0"},
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=1),
        elapsed_seconds=1.0,
        input_manifests=("inputs/manifest.json",),
        processed_count=2,
        success_count=1,
        failure_count=1,
        reproduction_command="python -m geml.example --config test.yml",
        metadata={"fixture": True},
    )


def test_valid_tiny_fixture_manifest_json_round_trip() -> None:
    manifest = _corpus_manifest()
    payload = manifest.model_dump(mode="json")

    restored = CorpusManifest.model_validate(payload)

    assert restored == manifest
    assert restored.total_row_count == 2
    assert json.loads(json.dumps(payload)) == payload


def test_final_split_count_specification_totals_250000() -> None:
    assert dict(FINAL_CORPUS_SPLIT_COUNTS) == {
        CorpusSplit.TRAIN: 175_000,
        CorpusSplit.VALIDATION: 25_000,
        CorpusSplit.TEST_IID: 25_000,
        CorpusSplit.TEST_OOD: 25_000,
    }
    assert FINAL_CORPUS_TOTAL_COUNT == 250_000
    assert sum(FINAL_CORPUS_SPLIT_COUNTS.values()) == 250_000


def test_shard_rejects_invalid_split_name() -> None:
    with pytest.raises(ValidationError):
        _shard(split="holdout")


def test_split_manifest_rejects_duplicate_shard_ids() -> None:
    first = _shard(shard_id="duplicate", shard_index=0)
    second = _shard(shard_id="duplicate", shard_index=1)

    with pytest.raises(ValidationError):
        _split_manifest(first, second)


@pytest.mark.parametrize("indexes", [(1,), (0, 2), (1, 0)])
def test_split_manifest_rejects_noncontiguous_or_reordered_indexes(
    indexes: tuple[int, ...],
) -> None:
    shards = tuple(
        _shard(shard_id=f"train-{position}", shard_index=shard_index)
        for position, shard_index in enumerate(indexes)
    )

    with pytest.raises(ValidationError, match="contiguous and match shard order"):
        _split_manifest(*shards)


def test_split_manifest_rejects_inconsistent_total() -> None:
    shard = _shard(row_count=2)

    with pytest.raises(ValidationError):
        SplitManifest(
            schema_version="1.0.0",
            corpus_id="tiny-fixture",
            split="train",
            shards=(shard,),
            total_row_count=3,
        )


@pytest.mark.parametrize("digest", ["", "a", "not-hex"])
def test_checksum_rejects_invalid_digest_shape(digest: str) -> None:
    with pytest.raises(ValidationError):
        ChecksumRecord(algorithm="sha256", digest=digest)


def test_run_metadata_json_round_trip() -> None:
    run = _run_metadata()
    payload = run.model_dump(mode="json")

    restored = RunMetadata.model_validate(payload)

    assert restored == run
    assert json.loads(json.dumps(payload)) == payload


def test_error_row_json_round_trip() -> None:
    error = ErrorRow(
        expression_id="expr-000001",
        shard_id="train-00000",
        stage="parsing",
        error_type="UnsupportedOperator",
        message="operator is outside the approved registry",
        recoverable=False,
        status="unsupported",
        metadata={"operator": "example"},
    )

    payload = error.model_dump(mode="json")
    restored = ErrorRow.model_validate(payload)

    assert restored == error
    assert json.loads(json.dumps(payload)) == payload


def test_run_metadata_rejects_missing_required_metadata() -> None:
    data = _run_metadata().model_dump(mode="json")
    del data["package_versions"]

    with pytest.raises(ValidationError):
        RunMetadata.model_validate(data)


def test_run_metadata_rejects_incomplete_accounting() -> None:
    data = _run_metadata().model_dump(mode="json")
    data["failure_count"] = 0

    with pytest.raises(ValidationError):
        RunMetadata.model_validate(data)


@pytest.mark.parametrize("field_name", ["shard_index", "row_count", "error_row_count"])
def test_shard_rejects_boolean_integer_fields(field_name: str) -> None:
    data = _shard().model_dump(mode="json")
    data[field_name] = True

    with pytest.raises(ValidationError):
        CorpusShardManifest.model_validate(data)


def test_manifest_rejects_naive_timestamp() -> None:
    data = _corpus_manifest().model_dump(mode="json")
    data["created_at"] = "2026-01-01T00:00:00"

    with pytest.raises(ValidationError):
        CorpusManifest.model_validate(data)


@pytest.mark.parametrize("timestamp", [0, 1.5, True, "0"])
def test_manifest_rejects_numeric_timestamp_coercion(timestamp: object) -> None:
    data = _corpus_manifest().model_dump(mode="json")
    data["created_at"] = timestamp

    with pytest.raises(ValidationError):
        CorpusManifest.model_validate(data)


def test_error_row_rejects_coerced_recoverability() -> None:
    with pytest.raises(ValidationError):
        ErrorRow(
            stage="parsing",
            error_type="ExampleError",
            message="example",
            recoverable=1,
            status="error",
        )


def test_run_metadata_rejects_boolean_elapsed_time() -> None:
    data = _run_metadata().model_dump(mode="json")
    data["elapsed_seconds"] = True

    with pytest.raises(ValidationError):
        RunMetadata.model_validate(data)
