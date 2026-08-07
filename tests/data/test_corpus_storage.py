"""Temporary-fixture tests for deterministic corpus storage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from geml.contracts.corpus import (
    FINAL_CORPUS_SPLIT_COUNTS,
    FINAL_CORPUS_TOTAL_COUNT,
    CorpusManifest,
    CorpusSplit,
)
from geml.contracts.expression import ExpressionRecord
from geml.data.storage.dedup import (
    DeduplicationError,
    DeduplicationSession,
    DeduplicationStats,
)
from geml.data.storage.manifests import (
    ManifestIntegrityError,
    build_corpus_manifest,
    build_split_manifest,
    load_corpus_manifest,
    validate_manifest,
    write_corpus_manifest,
    write_manifest_bundle,
)
from geml.data.storage.shards import (
    ShardFormat,
    ShardIntegrityError,
    ShardSizeError,
    ShardStorageError,
    plan_shard_sizes,
    read_shard,
    sha256_file,
    write_shards,
)
from geml.data.storage.splits import (
    SplitAssignmentError,
    SplitSizeMismatchError,
    assign_splits,
)


def _record(
    index: int,
    *,
    split: CorpusSplit = CorpusSplit.TRAIN,
    family: str = "algebraic_core",
    domain: str = "safe_real",
    srepr: str | None = None,
    expression_id: str | None = None,
) -> ExpressionRecord:
    return ExpressionRecord(
        expression_id=expression_id or f"{index + 1:064x}",
        sympy_srepr=srepr or f"Add(Symbol('x{index}'), Integer(1))",
        display_text=f"x{index} + 1",
        latex_text=None,
        split=split,
        operator_family=family,
        domain_mode=domain,
        variables=(f"x{index}",),
        target_ast_size=3,
        target_depth=1,
        generator_seed=2**200 + index,
        generator_metadata={"fixture_index": index, "nested": {"enabled": True}},
    )


def test_disk_deduplication_is_exact_streaming_and_audited(tmp_path: Path) -> None:
    first = _record(0)
    duplicate = _record(1, srepr=first.sympy_srepr)
    different_domain = _record(2, srepr=first.sympy_srepr, domain="positive_real")
    audit_path = tmp_path / "duplicates.jsonl"

    with DeduplicationSession(
        tmp_path / "dedup.sqlite3",
        duplicate_audit_path=audit_path,
    ) as session:
        unique = tuple(session.iter_unique((first, duplicate, different_domain)))
        stats = session.stats

    assert unique == (first, different_domain)
    assert stats == DeduplicationStats(3, 2, 1, 0)
    audit = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert audit == [
        {
            "domain_mode": first.domain_mode,
            "duplicate_expression_id": duplicate.expression_id,
            "kept_expression_id": first.expression_id,
            "sympy_srepr": first.sympy_srepr,
        }
    ]


def test_deduplication_database_resumes_and_rejects_identity_conflicts(
    tmp_path: Path,
) -> None:
    database = tmp_path / "dedup.sqlite3"
    first = _record(0)
    with DeduplicationSession(database) as session:
        assert session.register(first)

    with DeduplicationSession(database) as resumed:
        assert not resumed.register(first)
        conflicting = _record(
            1,
            expression_id=first.expression_id,
            srepr="Mul(Symbol('x'), Integer(2))",
        )
        with pytest.raises(DeduplicationError, match="maps to both"):
            resumed.register(conflicting)

        other = _record(2)
        assert resumed.register(other)
        crossed = _record(
            3,
            expression_id=first.expression_id,
            domain=other.domain_mode,
            srepr=other.sympy_srepr,
        )
        with pytest.raises(DeduplicationError, match="maps to both"):
            resumed.register(crossed)
        assert resumed.stats.identity_conflict_count == 2


def test_deduplication_checkpoint_follows_durable_downstream_output(tmp_path: Path) -> None:
    database = tmp_path / "dedup.sqlite3"
    record = _record(0)

    interrupted = DeduplicationSession(database)
    pending = interrupted.iter_unique((record,))
    assert next(pending) == record
    interrupted.close(commit=False)

    with DeduplicationSession(database) as replayed:
        assert replayed.register(record)
        (tmp_path / "durable-shard.marker").write_text(record.expression_id, encoding="utf-8")
        replayed.checkpoint()

    with DeduplicationSession(database) as resumed:
        assert not resumed.register(record)


def test_duplicate_audit_rolls_back_with_identity_index(tmp_path: Path) -> None:
    database = tmp_path / "dedup.sqlite3"
    audit = tmp_path / "duplicates.jsonl"
    first = _record(0)
    first_duplicate = _record(1, srepr=first.sympy_srepr)
    with DeduplicationSession(database, duplicate_audit_path=audit) as session:
        tuple(session.iter_unique((first, first_duplicate)))
    committed_snapshot = audit.read_bytes()

    second = _record(2)
    second_duplicate = _record(3, srepr=second.sympy_srepr)
    interrupted = DeduplicationSession(database, duplicate_audit_path=audit)
    tuple(interrupted.iter_unique((second, second_duplicate)))
    interrupted.close(commit=False)

    assert audit.read_bytes() == committed_snapshot
    with DeduplicationSession(database, duplicate_audit_path=audit) as replayed:
        assert replayed.register(second)
        assert not replayed.register(second_duplicate)
    assert len(audit.read_text(encoding="utf-8").splitlines()) == 2


def test_duplicate_audit_cannot_overwrite_its_sqlite_database(tmp_path: Path) -> None:
    path = tmp_path / "dedup.sqlite3"
    with pytest.raises(ValueError, match="must differ"):
        DeduplicationSession(path, duplicate_audit_path=path)


def test_split_assignment_is_exact_semantic_and_input_order_independent() -> None:
    iid = [_record(index) for index in range(8)]
    ood = [
        _record(
            100 + index,
            family="ood_stress",
            split=CorpusSplit.TRAIN,
        )
        for index in range(2)
    ]
    sizes = {
        CorpusSplit.TRAIN: 5,
        CorpusSplit.VALIDATION: 2,
        CorpusSplit.TEST_IID: 1,
        CorpusSplit.TEST_OOD: 2,
    }
    forward = assign_splits((*iid, *ood), sizes, seed=17)
    reverse = assign_splits(tuple(reversed((*iid, *ood))), sizes, seed=17)

    assert forward.records_by_split == reverse.records_by_split
    assert {split: len(records) for split, records in forward.records_by_split.items()} == sizes
    assert all(
        record.operator_family == "ood_stress" and record.split is CorpusSplit.TEST_OOD
        for record in forward.records_by_split[CorpusSplit.TEST_OOD]
    )
    assert all(
        record.operator_family != "ood_stress"
        for split in (
            CorpusSplit.TRAIN,
            CorpusSplit.VALIDATION,
            CorpusSplit.TEST_IID,
        )
        for record in forward.records_by_split[split]
    )


def test_split_assignment_validates_names_counts_and_uniqueness() -> None:
    records = [_record(index) for index in range(4)]
    valid_sizes = {
        "train": 2,
        "validation": 1,
        "test_iid": 1,
        "test_ood": 0,
    }
    with pytest.raises(SplitAssignmentError, match="unknown corpus split"):
        assign_splits(records, valid_sizes | {"iid_test": 0})
    with pytest.raises(SplitAssignmentError, match="nonnegative integer"):
        assign_splits(records, valid_sizes | {"train": -1})
    with pytest.raises(SplitSizeMismatchError, match="exact split counts"):
        assign_splits(records, valid_sizes | {"train": 3})
    with pytest.raises(SplitAssignmentError, match="duplicate expression_id"):
        assign_splits([records[0], records[0]], {**valid_sizes, "train": 0, "test_iid": 1})


def test_frozen_final_split_constants_are_exact() -> None:
    assert dict(FINAL_CORPUS_SPLIT_COUNTS) == {
        CorpusSplit.TRAIN: 175_000,
        CorpusSplit.VALIDATION: 25_000,
        CorpusSplit.TEST_IID: 25_000,
        CorpusSplit.TEST_OOD: 25_000,
    }
    assert FINAL_CORPUS_TOTAL_COUNT == 250_000


def test_final_250k_layout_is_representable_with_production_shard_bounds() -> None:
    planned = {split: plan_shard_sizes(count) for split, count in FINAL_CORPUS_SPLIT_COUNTS.items()}
    assert planned[CorpusSplit.TRAIN] == (25_000,) * 7
    assert planned[CorpusSplit.VALIDATION] == (25_000,)
    assert planned[CorpusSplit.TEST_IID] == (25_000,)
    assert planned[CorpusSplit.TEST_OOD] == (25_000,)
    assert sum(sum(sizes) for sizes in planned.values()) == 250_000


@pytest.mark.parametrize("shard_format", [ShardFormat.PARQUET, ShardFormat.JSONL_GZ])
def test_full_expression_records_round_trip_in_both_formats(
    tmp_path: Path,
    shard_format: ShardFormat,
) -> None:
    records = tuple(_record(index) for index in range(5))
    manifests = write_shards(
        records,
        tmp_path / "data" / "train",
        corpus_id="fixture",
        split=CorpusSplit.TRAIN,
        schema_version="fixture-v1",
        shard_format=shard_format,
        minimum_rows=1,
        maximum_rows=3,
        allow_small_fixture=True,
        manifest_root=tmp_path,
    )
    assert [manifest.row_count for manifest in manifests] == [3, 2]
    assert (
        tuple(record for manifest in manifests for record in read_shard(manifest, tmp_path))
        == records
    )
    assert all(manifest.checksum.algorithm == "sha256" for manifest in manifests)


def test_resuming_requires_byte_identical_shards(tmp_path: Path) -> None:
    records = tuple(_record(index) for index in range(2))
    arguments = {
        "records": records,
        "output_dir": tmp_path / "data" / "train",
        "corpus_id": "fixture",
        "split": CorpusSplit.TRAIN,
        "schema_version": "fixture-v1",
        "minimum_rows": 1,
        "maximum_rows": 2,
        "allow_small_fixture": True,
        "manifest_root": tmp_path,
    }
    first = write_shards(**arguments)
    assert write_shards(**arguments) == first

    shard_path = tmp_path / first[0].path
    shard_path.write_bytes(b"corrupted")
    with pytest.raises(ShardIntegrityError, match="does not match"):
        write_shards(**arguments)


def test_shard_policy_rejects_invalid_sizes_formats_and_splits(tmp_path: Path) -> None:
    record = _record(0)
    common = {
        "records": (record,),
        "output_dir": tmp_path / "data",
        "corpus_id": "fixture",
        "split": CorpusSplit.TRAIN,
        "schema_version": "fixture-v1",
        "manifest_root": tmp_path,
    }
    with pytest.raises(ShardSizeError, match="minimum_rows cannot exceed"):
        write_shards(
            **common,
            minimum_rows=2,
            maximum_rows=1,
            allow_small_fixture=True,
        )
    with pytest.raises(ShardSizeError, match="production shards"):
        write_shards(**common, minimum_rows=1, maximum_rows=2)
    with pytest.raises(ValueError, match="not a valid ShardFormat"):
        write_shards(
            **common,
            shard_format="csv",
            minimum_rows=1,
            maximum_rows=1,
            allow_small_fixture=True,
        )
    with pytest.raises(ShardStorageError, match="declared split"):
        write_shards(
            **(common | {"split": CorpusSplit.VALIDATION}),
            minimum_rows=1,
            maximum_rows=1,
            allow_small_fixture=True,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"corpus_id": ""}, "corpus_id"),
        ({"schema_version": "   "}, "schema_version"),
        ({"resume": 1}, "resume must be a boolean"),
        ({"allow_small_fixture": 1}, "allow_small_fixture must be a boolean"),
    ],
)
def test_invalid_shard_requests_publish_no_artifacts(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    output_dir = tmp_path / "data"
    arguments: dict[str, object] = {
        "records": (_record(0),),
        "output_dir": output_dir,
        "corpus_id": "fixture",
        "split": CorpusSplit.TRAIN,
        "schema_version": "fixture-v1",
        "minimum_rows": 1,
        "maximum_rows": 1,
        "allow_small_fixture": True,
        "manifest_root": tmp_path,
    }
    with pytest.raises(ShardStorageError, match=message):
        write_shards(**(arguments | overrides))  # type: ignore[arg-type]
    assert not list(output_dir.glob("train-*.parquet"))


def _fixture_manifest(tmp_path: Path) -> tuple[CorpusManifest, Path]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    split_manifests = []
    for index, split in enumerate(CorpusSplit):
        family = "ood_stress" if split is CorpusSplit.TEST_OOD else "algebraic_core"
        record = _record(index, split=split, family=family)
        shards = write_shards(
            (record,),
            tmp_path / "data" / split.value,
            corpus_id="fixture",
            split=split,
            schema_version="fixture-v1",
            minimum_rows=1,
            maximum_rows=1,
            allow_small_fixture=True,
            manifest_root=tmp_path,
            error_row_counts=(index,),
        )
        split_manifests.append(build_split_manifest(shards))

    manifest = build_corpus_manifest(
        split_manifests,
        corpus_id="fixture",
        schema_version="fixture-v1",
        config_path=config_path,
        generator_seed=1729,
        git_commit="abc123",
        created_at=datetime(2026, 7, 21, tzinfo=UTC),
        package_names=("pyarrow", "pydantic"),
        deduplication_stats=DeduplicationStats(5, 4, 1, 0),
        rejection_counts={"invalid_domain": 2},
    )
    return manifest, config_path


def test_manifest_uses_frozen_contract_and_complete_metadata(tmp_path: Path) -> None:
    manifest, config_path = _fixture_manifest(tmp_path)
    assert manifest.total_row_count == 4
    assert manifest.total_error_row_count == sum(range(4))
    assert manifest.config_hash
    assert manifest.generator_seed == 1729
    assert manifest.git_commit == "abc123"
    assert manifest.python_version
    assert manifest.platform
    assert set(manifest.package_versions) == {"pyarrow", "pydantic"}
    assert manifest.metadata["deduplication"] == {
        "processed_count": 5,
        "unique_count": 4,
        "duplicate_count": 1,
        "identity_conflict_count": 0,
    }
    assert manifest.metadata["rejection_counts"] == {"invalid_domain": 2}
    assert validate_manifest(manifest, tmp_path, config_path=config_path).valid


def test_manifest_bundle_is_atomic_resumable_and_strict(tmp_path: Path) -> None:
    manifest, _ = _fixture_manifest(tmp_path)
    bundle = write_manifest_bundle(manifest, tmp_path / "manifests")
    resumed = write_manifest_bundle(manifest, tmp_path / "manifests")
    assert bundle == resumed
    assert load_corpus_manifest(bundle.corpus_manifest) == manifest

    changed = manifest.model_copy(update={"metadata": {"changed": True}})
    with pytest.raises(ManifestIntegrityError, match="differs from resumed output"):
        write_corpus_manifest(changed, bundle.corpus_manifest)

    assert validate_manifest(manifest, tmp_path, manifest_dir=tmp_path / "manifests").valid
    bundle.shard_manifests[0].unlink()
    sidecar_result = validate_manifest(
        manifest,
        tmp_path,
        manifest_dir=tmp_path / "manifests",
    )
    assert not sidecar_result.valid
    assert any("missing manifest sidecar" in error for error in sidecar_result.errors)


def test_immutable_manifest_publish_never_overwrites_a_concurrent_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _fixture_manifest(tmp_path)
    destination = tmp_path / "race" / "corpus.manifest.json"
    competing_payload = b"concurrent-writer\n"

    def publish_competing_file(_source: object, target: object) -> None:
        Path(target).write_bytes(competing_payload)
        raise FileExistsError

    monkeypatch.setattr("geml.data.storage.manifests.os.link", publish_competing_file)
    with pytest.raises(ManifestIntegrityError, match="differs from resumed output"):
        write_corpus_manifest(manifest, destination)
    assert destination.read_bytes() == competing_payload


def test_manifest_bundle_refuses_completion_marker_for_invalid_shards(tmp_path: Path) -> None:
    manifest, config_path = _fixture_manifest(tmp_path)
    first_shard = manifest.splits[0].shards[0]
    (tmp_path / first_shard.path).unlink()

    with pytest.raises(ManifestIntegrityError, match="cannot finalize an invalid corpus"):
        write_manifest_bundle(
            manifest,
            tmp_path / "manifests",
            artifact_root=tmp_path,
            config_path=config_path,
        )
    assert not (tmp_path / "manifests" / "corpus.manifest.json").exists()


@pytest.mark.parametrize("failure_mode", ["missing", "corrupt", "config"])
def test_manifest_validation_fails_for_missing_or_corrupt_artifacts(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    manifest, config_path = _fixture_manifest(tmp_path)
    first_shard = manifest.splits[0].shards[0]
    if failure_mode == "missing":
        (tmp_path / first_shard.path).unlink()
    elif failure_mode == "corrupt":
        (tmp_path / first_shard.path).write_bytes(b"not parquet")
    else:
        config_path.write_text("fixture: false\n", encoding="utf-8")

    result = validate_manifest(manifest, tmp_path, config_path=config_path)
    assert not result.valid
    assert result.errors


def test_manifest_validation_retains_checksum_valid_parquet_schema_errors(
    tmp_path: Path,
) -> None:
    manifest, config_path = _fixture_manifest(tmp_path)
    original = manifest.splits[0].shards[0]
    shard_path = tmp_path / original.path
    pq.write_table(pa.table({"wrong_column": [1]}), shard_path)
    malformed = original.model_copy(
        update={
            "byte_count": shard_path.stat().st_size,
            "checksum": original.checksum.model_copy(update={"digest": sha256_file(shard_path)}),
        }
    )
    malformed_split = manifest.splits[0].model_copy(update={"shards": (malformed,)})
    malformed_manifest = manifest.model_copy(
        update={"splits": (malformed_split, *manifest.splits[1:])}
    )

    result = validate_manifest(malformed_manifest, tmp_path, config_path=config_path)
    assert not result.valid
    assert any("Parquet schema mismatch" in error for error in result.errors)


def test_checked_in_final_config_matches_frozen_counts() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "goal1_corpus_final.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["corpus"]["total_expressions"] == FINAL_CORPUS_TOTAL_COUNT
    assert config["splits"] == {
        split.value: count for split, count in FINAL_CORPUS_SPLIT_COUNTS.items()
    }
    assert config["shards"]["minimum_rows"] >= 10_000
    assert config["shards"]["maximum_rows"] <= 25_000
    assert config["deduplication"]["identity_fields"] == [
        "domain_mode",
        "sympy_srepr",
    ]
    assert config["deduplication"]["checkpoint_policy"] == "after_atomic_shard_publication"
