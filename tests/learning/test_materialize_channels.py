"""Fixture tests for the channel materialization CLI driver (torch-free)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from geml.contracts.corpus import CorpusSplit
from geml.learning.datasets.channels import RepresentationChannel
from geml.learning.datasets.materialize import validate_channel_alignment
from geml.learning.datasets.materialize_channels import RUN_MANIFEST_NAME, main

from .goal6_chain_fixtures import PAIR_PLAN, build_pair_shards, write_macro_vocabulary

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "goal6_channels.yaml"


def _cli(pairs_root: Path, output_root: Path, *extra: str) -> int:
    return main(
        [
            "--config",
            str(CONFIG_PATH),
            "--pairs-root",
            str(pairs_root),
            "--output-root",
            str(output_root),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One full materialization of all four channels from fixture pair shards."""

    root = tmp_path_factory.mktemp("chain")
    pairs = root / "pairs"
    out = root / "channels"
    pair_ids = build_pair_shards(pairs)
    vocabulary = root / "macro.vocabulary.json"
    write_macro_vocabulary(vocabulary)
    code = _cli(pairs, out, "--macro-vocabulary", str(vocabulary))
    assert code == 0
    return {"pairs": pairs, "out": out, "vocabulary": vocabulary, "pair_ids": pair_ids}


def _load_rows(shard: Path):
    from geml.learning.datasets.materialize import ChannelFailureV1, GraphTensorV1

    tensors, failures = [], []
    for line in shard.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        kind = row.pop("record_type")
        model = {"tensor": GraphTensorV1, "failure": ChannelFailureV1}[kind]
        parsed = model.model_validate_json(json.dumps(row))
        (tensors if kind == "tensor" else failures).append(parsed)
    return tensors, failures


def test_cli_materializes_all_four_channels_with_aligned_ledgers(materialized) -> None:
    out = materialized["out"]
    manifest = json.loads((out / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert set(manifest["channels"]) == {member.value for member in RepresentationChannel}
    assert all(entry["status"] == "materialized" for entry in manifest["channels"].values())
    assert manifest["macro_vocabulary"] is not None
    assert manifest["motif_ast_control_vocabulary"] is not None

    splits = sorted({split.value for split, _, _ in PAIR_PLAN})
    for split in splits:
        tensors, failures = [], []
        for member in RepresentationChannel:
            shard = out / member.value / f"pairs-{split}-00000.jsonl"
            assert shard.is_file()
            assert shard.with_suffix(shard.suffix + ".manifest.json").is_file()
            shard_tensors, shard_failures = _load_rows(shard)
            tensors.extend(shard_tensors)
            failures.extend(shard_failures)
        expected = materialized["pair_ids"][split]
        validate_channel_alignment(tensors, failures, expected_pair_ids=expected)


def test_cli_reruns_byte_identically(materialized, tmp_path: Path) -> None:
    out = materialized["out"]
    code = _cli(materialized["pairs"], out, "--macro-vocabulary", str(materialized["vocabulary"]))
    assert code == 0

    fresh = tmp_path / "fresh"
    code = _cli(materialized["pairs"], fresh, "--macro-vocabulary", str(materialized["vocabulary"]))
    assert code == 0
    for shard in sorted(out.rglob("pairs-*.jsonl")):
        relative = shard.relative_to(out)
        assert (fresh / relative).read_bytes() == shard.read_bytes()


def test_cli_without_vocabulary_reports_explicit_unavailable(materialized, tmp_path: Path) -> None:
    out = tmp_path / "channels"
    code = _cli(materialized["pairs"], out)
    assert code == 3

    manifest = json.loads((out / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    for name in ("frequent_macro_motif_dag", "motif_ast_fair_control"):
        entry = manifest["channels"][name]
        assert entry["status"] == "unavailable"
        assert "Goal 5" in entry["detail"]
        assert not (out / name).exists()
    for name in ("ast_dag", "pure_eml_dag"):
        assert manifest["channels"][name]["status"] == "materialized"
        assert (out / name / "pairs-train-00000.jsonl").is_file()


def test_cli_missing_pair_shards_fails_closed(tmp_path: Path, capsys) -> None:
    assert _cli(tmp_path / "absent", tmp_path / "out") == 1
    assert "no pair shards" in capsys.readouterr().err


def test_cli_corrupt_pair_shard_fails_closed(materialized, tmp_path: Path, capsys) -> None:
    pairs = tmp_path / "pairs"
    shutil.copytree(materialized["pairs"], pairs)
    shard = pairs / "pairs-train-00000.jsonl"
    shard.write_bytes(shard.read_bytes() + b"{}\n")
    assert _cli(pairs, tmp_path / "out") == 1
    assert "checksum" in capsys.readouterr().err


def test_cli_unknown_channel_fails_closed(materialized, tmp_path: Path, capsys) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["channels"].append({"name": "bogus_dag", "state": "available", "source": "nowhere"})
    config = tmp_path / "channels.yaml"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    code = main(
        [
            "--config",
            str(config),
            "--pairs-root",
            str(materialized["pairs"]),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "unknown channel" in capsys.readouterr().err


def test_cli_retains_endpoint_failures_as_ledger_rows(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs"
    pair_ids = build_pair_shards(pairs, include_broken_pair=True)
    out = tmp_path / "channels"
    assert _cli(pairs, out) == 3  # no vocabulary; ast_dag + pure_eml_dag still materialize

    manifest = json.loads((out / RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    train_pairs = len(pair_ids[CorpusSplit.TRAIN.value])
    for name in ("ast_dag", "pure_eml_dag"):
        shards = {shard["path"]: shard for shard in manifest["channels"][name]["shards"]}
        train = shards[f"{name}/pairs-train-00000.jsonl"]
        assert train["failure_count"] == 1
        assert train["tensor_count"] + train["failure_count"] == 2 * train_pairs
        _, failures = _load_rows(out / train["path"])
        assert failures[0].status.value == "invalid_source_graph"
        assert "AST" in failures[0].detail


def test_documented_module_command_dispatches_to_the_driver() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "geml.learning.datasets.materialize", "--help"],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")},
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0
    assert "--pairs-root" in completed.stdout
