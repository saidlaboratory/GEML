"""Fixture tests for the Goal 6 shard build loop over a tiny delivered corpus."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from geml.data.pairs.__main__ import load_production_config
from geml.data.pairs.build import PairBuildError, load_corpus_manifest, run_production_build
from geml.data.pairs.generate import PairContractError, PairRecordV1
from geml.data.pairs.providers import bind_production_providers

_CONFIG = Path("configs/goal6_pairs.yaml")
_X = "Symbol('x', real=True)"

_ROWS = {
    "train": [
        ("t1", f"Add({_X}, Integer(0))", "algebraic_core", "safe_real", 3, 1),
        ("t2", f"Add({_X}, Integer(1))", "algebraic_core", "safe_real", 3, 1),
        ("t3", f"Add({_X}, Integer(2))", "algebraic_core", "safe_real", 30, 1),
        ("t4", f"sin({_X})", "trig_hyperbolic", "safe_real", 2, 1),
        ("t7", f"Add({_X}, Integer(4))", "algebraic_core", "safe_real", 60, 1),
        ("t5", f"exp({_X})", "exp_log", "safe_real", 2, 1),
        ("t6", f"exp(Add({_X}, Integer(1)))", "exp_log", "safe_real", 4, 2),
    ],
    "validation": [
        ("v1", f"Mul({_X}, Integer(1))", "algebraic_core", "safe_real", 3, 1),
        ("v2", f"Mul({_X}, Integer(2))", "algebraic_core", "safe_real", 3, 1),
    ],
    "test_iid": [
        ("i1", f"Add(Mul({_X}, Integer(2)), Integer(0))", "algebraic_core", "safe_real", 5, 2),
        ("i2", "exp(log(Symbol('y', positive=True)))", "exp_log", "positive_real", 3, 2),
    ],
    "test_ood": [
        ("o1", f"Add({_X}, Add({_X}, Integer(0)))", "ood_stress", "safe_real", 5, 2),
        ("o2", f"Add({_X}, Integer(3))", "ood_stress", "safe_real", 3, 1),
    ],
}


def _write_corpus(root: Path, rows_by_split=None) -> Path:
    rows_by_split = _ROWS if rows_by_split is None else rows_by_split
    splits = []
    for split, rows in rows_by_split.items():
        shard_dir = root / "data" / split
        shard_dir.mkdir(parents=True)
        path = shard_dir / f"{split}-00000.parquet"
        pq.write_table(
            pa.table(
                {
                    "expression_id": [row[0] for row in rows],
                    "sympy_srepr": [row[1] for row in rows],
                    "split": [split] * len(rows),
                    "operator_family": [row[2] for row in rows],
                    "domain_mode": [row[3] for row in rows],
                    "target_ast_size": [row[4] for row in rows],
                    "target_depth": [row[5] for row in rows],
                }
            ),
            path,
        )
        splits.append(
            {
                "schema_version": "geml-corpus-v1",
                "corpus_id": "fixture",
                "split": split,
                "shards": [
                    {
                        "schema_version": "geml-corpus-v1",
                        "corpus_id": "fixture",
                        "shard_id": f"fixture:{split}:00000",
                        "path": f"data/{split}/{split}-00000.parquet",
                        "split": split,
                        "shard_index": 0,
                        "row_count": len(rows),
                        "checksum": {
                            "algorithm": "sha256",
                            "digest": hashlib.sha256(path.read_bytes()).hexdigest(),
                        },
                    }
                ],
                "total_row_count": len(rows),
            }
        )
    manifest = {
        "schema_version": "geml-corpus-v1",
        "corpus_id": "fixture",
        "splits": splits,
        "total_row_count": sum(len(rows) for rows in rows_by_split.values()),
        "config_hash": "ab" * 32,
        "generator_seed": 1,
        "created_at": "2026-07-29T00:00:00Z",
        "git_commit": "fixture",
        "python_version": "3.12",
        "platform": "test",
        "package_versions": {"geml": "test"},
    }
    path = root / "corpus_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _run(tmp_path: Path, *, output_name: str = "out", base_counts: dict | None = None):
    manifest_path = tmp_path / "corpus_manifest.json"
    if not manifest_path.is_file():
        _write_corpus(tmp_path)
    config, digest = load_production_config(_CONFIG)
    config["output_root"] = str(tmp_path / output_name)
    if base_counts is not None:
        config["base_counts"] = base_counts
    return run_production_build(
        config=config,
        config_digest=digest,
        corpus_manifest_path=manifest_path,
        providers=bind_production_providers(),
    )


def _shard_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_reports_exact_shortfalls_and_writes_shards(tmp_path) -> None:
    summary = _run(tmp_path)

    assert summary["status"] == "complete_short"
    for split, entry in summary["splits"].items():
        assert entry["shortfall"] == entry["target"] - entry["accepted"]
        assert entry["accepted"] == entry["accepted_positive"] + entry["accepted_negative"]
        shard = tmp_path / "out" / entry["shard_path"]
        assert shard.is_file()
        manifest = json.loads((shard.parent / (shard.name + ".manifest.json")).read_text())
        assert manifest["accepted_count"] == entry["accepted"]
        assert manifest["content_digest"] == entry["content_digest"]
        assert manifest["split_counts"] == {split: manifest["record_count"]}
        for line in shard.read_text(encoding="utf-8").splitlines():
            PairRecordV1.model_validate_json(line)
    assert json.loads((tmp_path / "out" / "pairs.run.manifest.json").read_text()) == summary


def test_missing_split_yields_full_shortfall_not_invented_records(tmp_path) -> None:
    _write_corpus(tmp_path, {split: rows for split, rows in _ROWS.items() if split != "validation"})
    summary = _run(tmp_path)

    entry = summary["splits"]["validation"]
    assert entry == {
        "accepted": 0,
        "accepted_negative": 0,
        "accepted_positive": 0,
        "content_digest": entry["content_digest"],
        "failed": 0,
        "outcome_counts": {},
        "rejected": 0,
        "shard_path": "pairs-validation-00000.jsonl",
        "shortfall": 5000,
        "source_rows": 0,
        "target": 5000,
    }
    assert (tmp_path / "out" / "pairs-validation-00000.jsonl").read_bytes() == b""


def test_rejections_and_failures_are_retained_with_typed_outcomes(tmp_path) -> None:
    summary = _run(tmp_path)
    records = _shard_records(tmp_path / "out" / "pairs-train-00000.jsonl")
    outcomes = {record["outcome_type"] for record in records}

    assert "trajectory_unsupported" in outcomes  # sin(x) source is retained, not dropped
    assert "negative_rejected_size_mismatch" in outcomes  # sizes 3 vs 30 candidate
    assert "negative_unresolved" in outcomes  # exp pair has no exact rational witness
    unresolved = next(r for r in records if r["outcome_type"] == "negative_unresolved")
    assert unresolved["status"] == "rejected"
    assert unresolved["label"] is None
    accepted_negative = next(
        r for r in records if r["label"] is False and r["status"] == "accepted"
    )
    witness = accepted_negative["non_equivalence_evidence"]
    assert witness["tier"] == "formal_counterexample"
    assert witness["rigorous"] is True
    assert summary["splits"]["train"]["failed"] >= 1
    assert summary["splits"]["train"]["rejected"] >= 2


def test_same_seed_reproduces_identical_shard_bytes(tmp_path) -> None:
    first = _run(tmp_path, output_name="first")
    second = _run(tmp_path, output_name="second")

    assert first["splits"] == second["splits"]
    for entry in first["splits"].values():
        assert (tmp_path / "first" / entry["shard_path"]).read_bytes() == (
            tmp_path / "second" / entry["shard_path"]
        ).read_bytes()
    assert (tmp_path / "first" / "pairs.run.manifest.json").read_bytes() == (
        tmp_path / "second" / "pairs.run.manifest.json"
    ).read_bytes()


def test_rerun_resumes_exact_bytes_and_refuses_corruption(tmp_path) -> None:
    first = _run(tmp_path)
    assert _run(tmp_path) == first  # byte-identical resume against existing outputs

    shard = tmp_path / "out" / "pairs-train-00000.jsonl"
    shard.write_bytes(shard.read_bytes() + b"tampered\n")
    with pytest.raises(PairContractError, match="differs"):
        _run(tmp_path)


def test_reachable_base_counts_complete_without_shortfall(tmp_path) -> None:
    summary = _run(
        tmp_path, base_counts={"train": 1, "validation": 1, "test_iid": 1, "test_ood": 1}
    )

    assert summary["status"] == "complete"
    assert all(entry["shortfall"] == 0 for entry in summary["splits"].values())
    assert all(entry["accepted"] == 1 for entry in summary["splits"].values())


def test_invalid_corpus_manifest_is_a_typed_refusal(tmp_path) -> None:
    path = tmp_path / "corpus_manifest.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(PairBuildError, match="invalid"):
        load_corpus_manifest(path)


def test_shard_checksum_mismatch_refuses(tmp_path) -> None:
    _write_corpus(tmp_path)
    shard = tmp_path / "data" / "train" / "train-00000.parquet"
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(PairBuildError, match="checksum mismatch"):
        _run(tmp_path)


def test_cli_builds_real_pair_shards_end_to_end(tmp_path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "corpus").mkdir()
    _write_corpus(root / "corpus")
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    config["output_root"] = str(tmp_path / "out")
    config_path = tmp_path / "pairs.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "geml.data.pairs",
            "--config",
            str(config_path),
            "--corpus-manifest",
            "${GEML_ARTIFACTS_ROOT}/corpus/corpus_manifest.json",
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src", "GEML_ARTIFACTS_ROOT": str(root)},
    )

    assert completed.returncode == 0, completed.stderr
    assert "pair build complete_short" in completed.stdout
    assert "shortfall" in completed.stdout
    summary = json.loads((tmp_path / "out" / "pairs.run.manifest.json").read_text())
    assert summary["status"] == "complete_short"
    for entry in summary["splits"].values():
        records = _shard_records(tmp_path / "out" / entry["shard_path"])
        assert len(records) == entry["accepted"] + entry["rejected"] + entry["failed"]
