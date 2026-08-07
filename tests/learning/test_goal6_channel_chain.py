"""End-to-end fixture chain: pair shards -> materialize CLI -> registry -> run_cell -> GridRow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch", reason="the optional [ml] extra is not installed")

from geml.experiments.goal6.registry import (  # noqa: E402
    ChannelRegistryError,
    ProductionChannelRegistry,
)
from geml.experiments.goal6.run_cell import main as run_cell_main  # noqa: E402
from geml.experiments.goal6.run_grid import (  # noqa: E402
    ChannelRegistry,
    GridContractError,
    build_manifest,
)
from geml.learning.datasets.materialize_channels import main as materialize_main  # noqa: E402
from geml.learning.datasets.shard_provider import ShardIntegrityError  # noqa: E402
from geml.learning.harness.seeds import PRODUCTION_SEEDS  # noqa: E402

from .goal6_chain_fixtures import build_pair_shards, write_macro_vocabulary  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "goal6_channels.yaml"


def _materialize(root: Path, *, include_broken_pair: bool = False) -> dict[str, object]:
    pairs = root / "pairs"
    out = root / "channels"
    pair_ids = build_pair_shards(pairs, include_broken_pair=include_broken_pair)
    vocabulary = root / "macro.vocabulary.json"
    write_macro_vocabulary(vocabulary)
    code = materialize_main(
        [
            "--config",
            str(CONFIG_PATH),
            "--pairs-root",
            str(pairs),
            "--output-root",
            str(out),
            "--macro-vocabulary",
            str(vocabulary),
        ]
    )
    assert code == 0
    return {"out": out, "pair_ids": pair_ids}


@pytest.fixture(scope="module")
def chain_root(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    return _materialize(tmp_path_factory.mktemp("chain"))


def _cell_configs(tmp_path: Path) -> tuple[Path, Path]:
    """A frozen-width grid config and a tiny harness config for one CPU fixture cell."""

    grid_config = tmp_path / "grid.yaml"
    payload = yaml.safe_load((REPOSITORY_ROOT / "configs" / "goal6_grid.yaml").read_text())
    payload["arms"]["graph"]["hidden_width"] = 64
    payload["arms"]["graph"]["use_virtual_node"] = False
    grid_config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    harness_config = tmp_path / "harness.yaml"
    harness_config.write_text(
        yaml.safe_dump(
            {
                "run_name": "goal6-fixture-cell",
                "output_root": str(tmp_path / "harness"),
                "seed": PRODUCTION_SEEDS[0],
                "max_epochs": 1,
            }
        ),
        encoding="utf-8",
    )
    return grid_config, harness_config


def test_production_registry_approves_the_materialized_channels(chain_root) -> None:
    registry = ProductionChannelRegistry(chain_root["out"])
    assert isinstance(registry, ChannelRegistry)
    channels = registry.approved_channels()
    assert [spec.channel_name for spec in channels] == [
        "ast_dag",
        "pure_eml_dag",
        "frequent_macro_motif_dag",
        "motif_ast_fair_control",
    ]
    assert all(spec.approved for spec in channels)
    assert all(spec.label_vocabulary_size > 0 for spec in channels)
    pure = next(spec for spec in channels if spec.channel_name == "pure_eml_dag")
    assert pure.is_pure_eml
    assert pure.representation_mode == "pure_eml:official_v4"

    expected = sorted(pid for ids in chain_root["pair_ids"].values() for pid in ids)
    for spec in channels:
        assert list(registry.aligned_pair_ids(spec.channel_name)) == expected
    assert registry.view_status("train") == "available"
    assert registry.view_status("test_ood_stress") == "available"
    assert registry.view_status("test_depth_ood") == "unsupported_not_disjoint"
    assert registry.view_status("test_family_ood") == "unsupported_not_excluded"


def test_registry_fails_closed_on_missing_or_corrupt_evidence(chain_root, tmp_path: Path) -> None:
    with pytest.raises(ChannelRegistryError, match="run manifest is missing"):
        ProductionChannelRegistry(tmp_path / "absent")
    with pytest.raises(ChannelRegistryError, match="no materialized evidence"):
        ProductionChannelRegistry(chain_root["out"]).aligned_pair_ids("not_a_channel")

    import shutil

    corrupt = tmp_path / "corrupt"
    shutil.copytree(chain_root["out"], corrupt)
    shard = corrupt / "ast_dag" / "pairs-train-00000.jsonl"
    shard.write_bytes(shard.read_bytes() + b"{}\n")
    with pytest.raises(ShardIntegrityError, match="checksum"):
        ProductionChannelRegistry(corrupt)

    tampered = tmp_path / "tampered"
    shutil.copytree(chain_root["out"], tampered)
    manifest_path = tampered / "materialize.run.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["channels"]["ast_dag"]["shards"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ChannelRegistryError, match="shard list"):
        ProductionChannelRegistry(tampered)


def test_retained_failure_rows_block_channel_approval(tmp_path: Path) -> None:
    materialized = _materialize(tmp_path, include_broken_pair=True)
    registry = ProductionChannelRegistry(materialized["out"])
    assert all(not spec.approved for spec in registry.approved_channels())
    with pytest.raises(GridContractError, match="requires exactly 4 approved"):
        build_manifest(registry, commit="a" * 40)


def test_full_chain_runs_one_cell_to_a_complete_grid_row(chain_root, tmp_path: Path) -> None:
    grid_root = tmp_path / "grid"
    channels_root = str(chain_root["out"])
    assert (
        run_cell_main(
            [
                "--write-manifest",
                "--grid-root",
                str(grid_root),
                "--channels-root",
                channels_root,
            ]
        )
        == 0
    )
    manifest = json.loads((grid_root / "grid.manifest.json").read_text(encoding="utf-8"))
    assert manifest["cell_count"] == 18
    assert manifest["aligned_pair_count"] == sum(
        len(ids) for ids in chain_root["pair_ids"].values()
    )

    grid_config, harness_config = _cell_configs(tmp_path)
    cell_id = f"graph__ast_dag__seed-{PRODUCTION_SEEDS[0]}"
    arguments = [
        "--grid-root",
        str(grid_root),
        "--channels-root",
        channels_root,
        "--cell-id",
        cell_id,
        "--grid-config",
        str(grid_config),
        "--harness-config",
        str(harness_config),
    ]
    assert run_cell_main(arguments) == 0

    row_path = grid_root / "cells" / f"{cell_id}.json"  # GridRunner.existing_row identity
    row = json.loads(row_path.read_text(encoding="utf-8"))
    assert row["schema_version"] == "geml-goal6-grid-row-v1"
    assert row["status"] == "complete"
    assert row["arm_id"] == "graph::ast_dag"
    assert row["seed"] == PRODUCTION_SEEDS[0]
    assert set(row["metrics_by_view"]) == {"train", "validation", "test_iid", "test_ood_stress"}
    assert row["denominators_by_view"]["test_iid"]["attempted"] == 1
    assert row["parameters"] > 0
    assert row["commit"] == manifest["commit"]

    # A second invocation must skip the complete cell rather than recompute it.
    before = row_path.read_bytes()
    assert run_cell_main(arguments) == 0
    assert row_path.read_bytes() == before


def test_run_cell_refusals_leave_no_row(chain_root, tmp_path: Path, capsys) -> None:
    grid_root = tmp_path / "grid"
    channels_root = str(chain_root["out"])
    common = ["--grid-root", str(grid_root), "--channels-root", channels_root]

    # No frozen manifest yet.
    assert run_cell_main([*common, "--cell-id", "graph__ast_dag__seed-20260726"]) == 2
    assert "--write-manifest" in capsys.readouterr().err
    assert run_cell_main([*common, "--write-manifest"]) == 0

    # Unknown cell id.
    assert run_cell_main([*common, "--cell-id", "graph__ast_dag__seed-1"]) == 2
    assert "not one of" in capsys.readouterr().err

    # The shipped grid config still has the width freeze pending: refuse, write nothing.
    assert (
        run_cell_main(
            [
                *common,
                "--cell-id",
                "graph__ast_dag__seed-20260726",
                "--grid-config",
                str(REPOSITORY_ROOT / "configs" / "goal6_grid.yaml"),
                "--harness-config",
                str(REPOSITORY_ROOT / "configs" / "goal6_harness_defaults.yaml"),
            ]
        )
        == 2
    )
    assert "width" in capsys.readouterr().err
    assert not (grid_root / "cells").exists()

    # A tampered manifest commit must refuse before any evidence exists.
    manifest_path = grid_root / "grid.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["commit"] = "0" * 40
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    grid_config, harness_config = _cell_configs(tmp_path)
    assert (
        run_cell_main(
            [
                *common,
                "--cell-id",
                "graph__ast_dag__seed-20260726",
                "--grid-config",
                str(grid_config),
                "--harness-config",
                str(harness_config),
            ]
        )
        == 2
    )
    assert "commit" in capsys.readouterr().err
    assert not (grid_root / "cells").exists()

    # A structurally tampered manifest (arm entry stripped) refuses with a typed error.
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["arms"][0]["family"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert run_cell_main([*common, "--cell-id", "graph__ast_dag__seed-20260726"]) == 2
    assert "malformed" in capsys.readouterr().err
    assert not (grid_root / "cells").exists()


def test_run_cell_without_provider_writes_an_explicit_failed_row(chain_root, tmp_path) -> None:
    grid_root = tmp_path / "grid"
    channels_root = str(chain_root["out"])
    assert (
        run_cell_main(
            ["--write-manifest", "--grid-root", str(grid_root), "--channels-root", channels_root]
        )
        == 0
    )
    grid_config, harness_config = _cell_configs(tmp_path)
    cell_id = f"trivial_floor__seed-{PRODUCTION_SEEDS[0]}"
    code = run_cell_main(
        [
            "--grid-root",
            str(grid_root),
            "--channels-root",
            channels_root,
            "--cell-id",
            cell_id,
            "--grid-config",
            str(grid_config),
            "--harness-config",
            str(harness_config),
        ]
    )
    assert code == 1
    row = json.loads((grid_root / "cells" / f"{cell_id}.json").read_text(encoding="utf-8"))
    assert row["status"] == "failed"
    assert "no dataset provider" in row["failure_reason"]


def test_grid_manifest_feeds_the_h100_scheduler(chain_root, tmp_path: Path) -> None:
    from scripts.h100.schedule_cells import load_cells

    grid_root = tmp_path / "grid"
    assert (
        run_cell_main(
            [
                "--write-manifest",
                "--grid-root",
                str(grid_root),
                "--channels-root",
                str(chain_root["out"]),
            ]
        )
        == 0
    )
    cells = load_cells(grid_root / "grid.manifest.json", grid_root)
    assert len(cells) == 18
    sample = next(cell for cell in cells if cell.cell_id == "graph__ast_dag__seed-20260726")
    assert sample.row_path == grid_root / "cells" / "graph__ast_dag__seed-20260726.json"
