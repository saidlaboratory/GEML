"""Run exactly one Goal 6 (arm, seed) production cell, or freeze the grid manifest.

This is the ``--cmd`` target for ``scripts/h100/schedule_cells.py``:

.. code-block:: bash

    # once, after channel materialization:
    python -m geml.experiments.goal6.run_cell --write-manifest \
        --grid-root outputs/final/goal6/grid --channels-root outputs/final/goal6/channels

    # per cell (what the scheduler renders from its template):
    python -m geml.experiments.goal6.run_cell --cell-id {cell_id} \
        --grid-root outputs/final/goal6/grid --channels-root outputs/final/goal6/channels

The cell run resolves its arm and seed from the frozen ``grid.manifest.json``, binds the
production shard-backed providers through :class:`ProductionChannelRegistry`, runs
:class:`ProductionCellRunner`, and writes the row exactly where ``GridRunner.existing_row``
reads it (``cells/<cell_id>.json``).  Exit codes: 0 for a COMPLETE row (or an already-complete
cell), 1 for a retained FAILED row (still written), 2 for a refusal before any evidence exists
(missing manifest, unknown cell, pending width freeze, commit mismatch) -- the scheduler then
records ``no_row`` instead of trusting a fabricated outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from geml.experiments.goal6.cell_runner import (
    CellRunnerError,
    ProductionCellRunner,
    resolve_commit,
)
from geml.experiments.goal6.registry import ChannelRegistryError, ProductionChannelRegistry
from geml.experiments.goal6.run_grid import (
    GRID_SCHEMA_VERSION,
    PRODUCTION_GRID_ROOT,
    ArmFamily,
    ArmSpec,
    ChannelSpec,
    GridContractError,
    build_manifest,
    grid_cell_id,
)
from geml.learning.backbones.gin import EncoderConfig, EncoderConfigurationError
from geml.learning.datasets.shard_provider import ShardProviderError
from geml.learning.harness.config import HarnessConfig, HarnessConfigError, load_harness_config
from geml.learning.harness.train import CellStatus, atomic_write_json

DEFAULT_CHANNELS_ROOT = "outputs/final/goal6/channels"


class RunCellError(RuntimeError):
    """The cell cannot be attempted; refused before any row evidence exists."""


def _load_yaml(path: Path, label: str) -> dict:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RunCellError(f"{label} {path} is unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise RunCellError(f"{label} {path} is not a mapping")
    return payload


def load_encoder_config(grid_config_path: Path) -> EncoderConfig:
    """Read the frozen width decision; refuse while it is still pending the preflight."""

    payload = _load_yaml(grid_config_path, "grid config")
    try:
        graph_arm = payload["arms"]["graph"]
    except (KeyError, TypeError) as error:
        raise RunCellError(f"grid config {grid_config_path} has no arms.graph section") from error
    width = graph_arm.get("hidden_width")
    virtual = graph_arm.get("use_virtual_node")
    if type(width) is not int or not isinstance(virtual, bool):
        raise RunCellError(
            f"grid config {grid_config_path} has hidden_width={width!r} and "
            f"use_virtual_node={virtual!r}; the width freeze is still pending -- run the width "
            "preflight and freeze both values first (runbook prerequisite 5)"
        )
    permitted = graph_arm.get("permitted_widths")
    if isinstance(permitted, list) and width not in permitted:
        raise RunCellError(
            f"grid config {grid_config_path} froze hidden_width={width}, outside the permitted "
            f"widths {permitted}"
        )
    try:
        return EncoderConfig(hidden_width=width, use_virtual_node=virtual)
    except EncoderConfigurationError as error:
        raise RunCellError(str(error)) from error


def load_grid_manifest(grid_root: Path) -> dict:
    path = grid_root / "grid.manifest.json"
    if not path.is_file():
        raise RunCellError(
            f"grid manifest is missing: {path}; freeze it first with "
            "python -m geml.experiments.goal6.run_cell --write-manifest"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunCellError(f"grid manifest {path} is unreadable: {error}") from error
    if payload.get("schema_version") != GRID_SCHEMA_VERSION:
        raise RunCellError(f"grid manifest {path} does not carry schema {GRID_SCHEMA_VERSION!r}")
    return payload


def resolve_cell(manifest: dict, cell_id: str) -> tuple[ArmSpec, int]:
    """Resolve one manifest cell id back to its frozen (arm, seed)."""

    try:
        arms = []
        for entry in manifest["arms"]:
            channel = entry.get("channel")
            arms.append(
                ArmSpec(
                    arm_id=str(entry["arm_id"]),
                    family=ArmFamily(str(entry["family"])),
                    channel=ChannelSpec(**channel) if channel else None,
                )
            )
        seeds = [int(seed) for seed in manifest["seeds"]]
    except (KeyError, TypeError, ValueError) as error:
        raise RunCellError(
            f"grid manifest is malformed ({type(error).__name__}: {error}); refreeze it with "
            "--write-manifest"
        ) from error
    for arm in arms:
        for seed in seeds:
            if grid_cell_id(arm, seed) == cell_id:
                return arm, seed
    raise RunCellError(
        f"cell id {cell_id!r} is not one of the {len(arms) * len(seeds)} cells frozen in the "
        "grid manifest"
    )


def load_cell_harness_config(path: Path) -> HarnessConfig:
    try:
        return load_harness_config(_load_yaml(path, "harness config"))
    except HarnessConfigError as error:
        raise RunCellError(f"harness config {path} is invalid: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m geml.experiments.goal6.run_cell",
        description="Run one Goal 6 (arm, seed) production cell.",
    )
    parser.add_argument("--grid-root", type=Path, default=Path(PRODUCTION_GRID_ROOT))
    parser.add_argument("--channels-root", type=Path, default=Path(DEFAULT_CHANNELS_ROOT))
    parser.add_argument("--cell-id", help="one grid_cell_id from the frozen manifest")
    parser.add_argument(
        "--harness-config", type=Path, default=Path("configs/goal6_harness_defaults.yaml")
    )
    parser.add_argument("--grid-config", type=Path, default=Path("configs/goal6_grid.yaml"))
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="validate the production registry and freeze grid.manifest.json, then exit",
    )
    args = parser.parse_args(argv)

    try:
        if args.write_manifest:
            registry = ProductionChannelRegistry(args.channels_root)
            manifest = build_manifest(registry, commit=resolve_commit())
            atomic_write_json(args.grid_root / "grid.manifest.json", manifest.as_dict())
            print(
                json.dumps(
                    {
                        "cell_count": manifest.as_dict()["cell_count"],
                        "manifest_digest": manifest.manifest_digest,
                        "written": str(args.grid_root / "grid.manifest.json"),
                    },
                    sort_keys=True,
                )
            )
            return 0

        if not args.cell_id:
            parser.error("--cell-id is required unless --write-manifest is given")
        manifest = load_grid_manifest(args.grid_root)
        arm, seed = resolve_cell(manifest, args.cell_id)
        commit = resolve_commit()
        if commit != manifest["commit"]:
            raise RunCellError(
                f"checkout commit {commit} differs from the manifest commit "
                f"{manifest['commit']}; a cell must run the exact frozen code -- rebuild the "
                "manifest or check out its commit"
            )
        config = load_cell_harness_config(args.harness_config)
        # Non-graph arms never touch the encoder; the default keeps the runner constructible
        # without inventing a width decision for the graph arms.
        encoder_config = (
            load_encoder_config(args.grid_config)
            if arm.family is ArmFamily.GRAPH
            else EncoderConfig()
        )
        registry = ProductionChannelRegistry(args.channels_root)
        runner = ProductionCellRunner(
            registry.graph_providers(), encoder_config=encoder_config, commit=commit
        )

        row_path = args.grid_root / "cells" / f"{args.cell_id}.json"
        if row_path.is_file():
            existing = json.loads(row_path.read_text(encoding="utf-8"))
            if existing.get("status") == str(CellStatus.COMPLETE):
                print(
                    json.dumps(
                        {
                            "cell_id": args.cell_id,
                            "status": "complete",
                            "skipped": "existing complete row",
                        },
                        sort_keys=True,
                    )
                )
                return 0

        row = runner.run_cell(arm, seed, config)
        atomic_write_json(row_path, row.as_dict())
        print(
            json.dumps(
                {
                    "cell_id": args.cell_id,
                    "status": str(row.status),
                    "failure_reason": row.failure_reason,
                    "row": str(row_path),
                    "wall_seconds": row.wall_seconds,
                },
                sort_keys=True,
            )
        )
        return 0 if row.status is CellStatus.COMPLETE else 1
    except (
        RunCellError,
        ChannelRegistryError,
        GridContractError,
        ShardProviderError,
        CellRunnerError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
