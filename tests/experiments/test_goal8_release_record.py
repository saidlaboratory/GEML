"""Fixture tests for the frozen Goal 8 release record emitter and loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from geml.analysis.goal8.summary import (
    AnalysisError,
    ManifestKind,
    authenticate_producer_run,
)
from geml.experiments.goal8.release_record import (
    RELEASE_RECORD_SCHEMA,
    ReleaseRecordError,
    emit_release_record,
    load_release_record,
)

_ATP_CELL_SCHEMA = "fixture-atp-row-v1"
_SIMPLIFY_CELL_SCHEMA = "fixture-simplification-row-v1"
_CONFIG_DIGEST = "b" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_run(root: Path, kind: ManifestKind, config_digest: str = _CONFIG_DIGEST) -> Path:
    attestation = {
        "rule_set_sha256": "f" * 64,
        "verifier_sha256": "1" * 64,
        "implementation_sha256": "2" * 64,
    }
    if kind is ManifestKind.PROOF:
        binding = {
            "benchmark_manifest_sha256": "c" * 64,
            "benchmark_projection_digest": "d" * 64,
            "budget_digest": "e" * 64,
        }
        run_identity = {
            "config_digest": config_digest,
            "benchmark_manifest_sha256": binding["benchmark_manifest_sha256"],
            "benchmark_projection_digest": binding["benchmark_projection_digest"],
            **attestation,
        }
        run_domain = b"geml-goal8-atp-run-v1\0"
        cell_schema = _ATP_CELL_SCHEMA
        shard_schema = "geml-goal8-atp-shard-v1"
    else:
        binding = {
            "sample_manifest_digest": "5" * 64,
            "source_manifest_sha256": "6" * 64,
            "learned_exclusion_manifest_sha256": "7" * 64,
        }
        run_identity = {
            "config_digest": config_digest,
            "sample_manifest_digest": binding["sample_manifest_digest"],
            "source_manifest_sha256": binding["source_manifest_sha256"],
            **attestation,
        }
        run_domain = b"geml-goal8-simplify-run-v1\0"
        cell_schema = _SIMPLIFY_CELL_SCHEMA
        shard_schema = "geml-goal8-simplify-shard-v1"
    run_id = hashlib.sha256(run_domain + _canonical(run_identity)).hexdigest()

    rows = []
    for index in range(2):
        row = {
            "schema_version": cell_schema,
            "cell_id": hashlib.sha256(f"{kind.value}-cell-{index}".encode()).hexdigest(),
            "run_id": run_id,
            "config_digest": config_digest,
            "status": "complete",
            **binding,
            **attestation,
        }
        row["content_digest"] = _digest(row)
        rows.append(row)

    run_dir = root / kind.value / run_id
    (run_dir / "shards" / "shard-00000").mkdir(parents=True)
    for row in rows:
        cell_id = str(row["cell_id"])
        cell_path = run_dir / "cells" / cell_id[:2] / f"{cell_id}.json"
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        cell_path.write_text(json.dumps(row), encoding="utf-8")
    expected_ids = sorted(str(row["cell_id"]) for row in rows)
    completion = {
        "schema_version": shard_schema,
        "run_id": run_id,
        "shard_index": 0,
        "shard_count": 1,
        "config_digest": config_digest,
        "expected_cell_ids": expected_ids,
        "expected_count": len(expected_ids),
        "attempted_count": len(expected_ids),
        "status_counts": {"complete": len(expected_ids)},
        "cell_content_digests": {str(row["cell_id"]): _digest(row) for row in rows},
        **binding,
    }
    (run_dir / "shards" / "shard-00000" / "shard.complete.json").write_text(
        json.dumps(completion),
        encoding="utf-8",
    )
    return run_dir


def _emit(tmp_path: Path) -> tuple[Path, Path, Path]:
    proof_dir = _write_run(tmp_path, ManifestKind.PROOF)
    simplify_dir = _write_run(tmp_path, ManifestKind.SIMPLIFICATION)
    record_path = tmp_path / "release" / "goal8_release_record.json"
    emit_release_record(
        record_path,
        proof_run_directory=proof_dir,
        proof_config_digest=_CONFIG_DIGEST,
        simplification_run_directory=simplify_dir,
        simplification_config_digest=_CONFIG_DIGEST,
        proof_cell_schema_version=_ATP_CELL_SCHEMA,
        simplification_cell_schema_version=_SIMPLIFY_CELL_SCHEMA,
    )
    return record_path, proof_dir, simplify_dir


def _authenticate(run_dir: Path, kind: ManifestKind, anchor: dict):
    cell_schema = _ATP_CELL_SCHEMA if kind is ManifestKind.PROOF else _SIMPLIFY_CELL_SCHEMA
    return authenticate_producer_run(
        run_dir,
        kind=kind,
        cell_schema_version=cell_schema,
        expected_aggregate_sha256=anchor["aggregate_sha256"],
        expected_config_digest=anchor["config_digest"],
    )


class TestEmitterAndAnalysisAuthentication:
    def test_analysis_side_authentication_accepts_the_emitted_record(self, tmp_path: Path):
        record_path, proof_dir, simplify_dir = _emit(tmp_path)
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == RELEASE_RECORD_SCHEMA
        anchors = load_release_record(record_path)
        for kind, run_dir in (
            (ManifestKind.PROOF, proof_dir),
            (ManifestKind.SIMPLIFICATION, simplify_dir),
        ):
            anchor = anchors[kind.value]
            bundle = _authenticate(run_dir, kind, anchor)
            assert bundle.trust_anchor_verified
            assert bundle.run_id == anchor["run_id"]
            assert len(bundle.rows) == anchor["cell_count"]
            assert bundle.byte_count == anchor["byte_count"]

    def test_tampered_run_bytes_fail_against_the_record_anchor(self, tmp_path: Path):
        record_path, proof_dir, _ = _emit(tmp_path)
        anchor = load_release_record(record_path)["proof_benchmark"]
        cell = next((proof_dir / "cells").rglob("*.json"))
        # Same parsed JSON, different exact bytes: only the aggregate can catch it.
        cell.write_text(cell.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(AnalysisError, match="aggregate trust anchor"):
            _authenticate(proof_dir, ManifestKind.PROOF, anchor)

    def test_tampered_record_content_is_rejected_by_the_loader(self, tmp_path: Path):
        record_path, _, _ = _emit(tmp_path)
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        payload["bundles"]["proof_benchmark"]["aggregate_sha256"] = "0" * 64
        record_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ReleaseRecordError, match="content_digest"):
            load_release_record(record_path)

    def test_reforged_record_checksum_cannot_authenticate_the_run(self, tmp_path: Path):
        record_path, proof_dir, _ = _emit(tmp_path)
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        payload["bundles"]["proof_benchmark"]["aggregate_sha256"] = "0" * 64
        content = {key: value for key, value in payload.items() if key != "content_digest"}
        payload["content_digest"] = hashlib.sha256(
            b"geml-goal8-release-record-v1\0" + _canonical(content)
        ).hexdigest()
        record_path.write_text(json.dumps(payload), encoding="utf-8")
        anchors = load_release_record(record_path)
        with pytest.raises(AnalysisError, match="aggregate trust anchor"):
            _authenticate(proof_dir, ManifestKind.PROOF, anchors["proof_benchmark"])


class TestFailClosedEmission:
    def test_emission_requires_the_pre_run_config_digest(self, tmp_path: Path):
        proof_dir = _write_run(tmp_path, ManifestKind.PROOF)
        simplify_dir = _write_run(tmp_path, ManifestKind.SIMPLIFICATION)
        with pytest.raises(ReleaseRecordError, match="config digest"):
            emit_release_record(
                tmp_path / "record.json",
                proof_run_directory=proof_dir,
                proof_config_digest="9" * 64,
                simplification_run_directory=simplify_dir,
                simplification_config_digest=_CONFIG_DIGEST,
                proof_cell_schema_version=_ATP_CELL_SCHEMA,
                simplification_cell_schema_version=_SIMPLIFY_CELL_SCHEMA,
            )
        assert not (tmp_path / "record.json").exists()

    def test_corrupt_cell_digest_blocks_emission(self, tmp_path: Path):
        proof_dir = _write_run(tmp_path, ManifestKind.PROOF)
        simplify_dir = _write_run(tmp_path, ManifestKind.SIMPLIFICATION)
        cell = next((proof_dir / "cells").rglob("*.json"))
        row = json.loads(cell.read_text(encoding="utf-8"))
        row["status"] = "failed"
        cell.write_text(json.dumps(row), encoding="utf-8")
        with pytest.raises(ReleaseRecordError, match="proof_benchmark run cannot be anchored"):
            emit_release_record(
                tmp_path / "record.json",
                proof_run_directory=proof_dir,
                proof_config_digest=_CONFIG_DIGEST,
                simplification_run_directory=simplify_dir,
                simplification_config_digest=_CONFIG_DIGEST,
                proof_cell_schema_version=_ATP_CELL_SCHEMA,
                simplification_cell_schema_version=_SIMPLIFY_CELL_SCHEMA,
            )
        assert not (tmp_path / "record.json").exists()

    def test_record_must_live_outside_both_run_directories(self, tmp_path: Path):
        proof_dir = _write_run(tmp_path, ManifestKind.PROOF)
        simplify_dir = _write_run(tmp_path, ManifestKind.SIMPLIFICATION)
        with pytest.raises(ReleaseRecordError, match="outside"):
            emit_release_record(
                proof_dir / "goal8_release_record.json",
                proof_run_directory=proof_dir,
                proof_config_digest=_CONFIG_DIGEST,
                simplification_run_directory=simplify_dir,
                simplification_config_digest=_CONFIG_DIGEST,
                proof_cell_schema_version=_ATP_CELL_SCHEMA,
                simplification_cell_schema_version=_SIMPLIFY_CELL_SCHEMA,
            )

    def test_existing_record_is_immutable_but_identical_reemission_resumes(self, tmp_path: Path):
        record_path, proof_dir, simplify_dir = _emit(tmp_path)
        original = record_path.read_bytes()
        emit_release_record(
            record_path,
            proof_run_directory=proof_dir,
            proof_config_digest=_CONFIG_DIGEST,
            simplification_run_directory=simplify_dir,
            simplification_config_digest=_CONFIG_DIGEST,
            proof_cell_schema_version=_ATP_CELL_SCHEMA,
            simplification_cell_schema_version=_SIMPLIFY_CELL_SCHEMA,
        )
        assert record_path.read_bytes() == original
        record_path.write_text("{}", encoding="utf-8")
        with pytest.raises(ReleaseRecordError, match="already exists"):
            emit_release_record(
                record_path,
                proof_run_directory=proof_dir,
                proof_config_digest=_CONFIG_DIGEST,
                simplification_run_directory=simplify_dir,
                simplification_config_digest=_CONFIG_DIGEST,
                proof_cell_schema_version=_ATP_CELL_SCHEMA,
                simplification_cell_schema_version=_SIMPLIFY_CELL_SCHEMA,
            )
