"""Frozen, denominator-complete orchestration for the Goal 6 equivalence grid.

This module intentionally does not manufacture pair data, materialized graphs, model
weights, or numerical results.  It creates the fixed 18-cell plan and records either
completed executor results or explicit terminal rows, including the blocked motif-AST
fair control described in the Goal 6 channel contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from geml.learning.harness.seeds import PRODUCTION_SEEDS

GRID_SCHEMA_VERSION = "geml-goal6-grid-v1"
GRID_MANIFEST_FILENAME = "goal6.grid.json"
FIXTURE_PAIR_COUNT = 25


class Goal6GridError(ValueError):
    """The frozen grid plan or a retained cell result is invalid."""


class ArmFamily(StrEnum):
    GINE = "gine"
    PREFIX_TRANSFORMER = "prefix_transformer"
    TRIVIAL = "trivial"


class CellStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"


class EvaluationView(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST_IID = "test_iid"
    TEST_DEPTH_OOD = "test_depth_ood"
    TEST_FAMILY_OOD = "test_family_ood"


@dataclass(frozen=True, slots=True)
class GridArm:
    """One preregistered model/input arm; graph arms share the same GINE policy."""

    arm_id: str
    family: ArmFamily
    channel: str | None
    availability: CellStatus
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.family is ArmFamily.GINE and self.channel is None:
            raise Goal6GridError("a GINE arm must declare exactly one input channel")
        if self.family is not ArmFamily.GINE and self.channel is not None:
            raise Goal6GridError("non-graph controls cannot declare a graph channel")
        if self.availability is CellStatus.UNSUPPORTED and not self.unavailable_reason:
            raise Goal6GridError("an unsupported arm must retain its reason")
        if self.availability is not CellStatus.UNSUPPORTED and self.unavailable_reason is not None:
            raise Goal6GridError("available arms cannot carry an unavailable reason")


FIXED_ARMS: tuple[GridArm, ...] = (
    GridArm("ast_gine", ArmFamily.GINE, "ast_dag", CellStatus.PENDING),
    GridArm("pure_eml_gine", ArmFamily.GINE, "pure_eml_dag", CellStatus.PENDING),
    GridArm(
        "frequent_macro_motif_gine",
        ArmFamily.GINE,
        "frequent_macro_motif_dag",
        CellStatus.PENDING,
    ),
    GridArm(
        "motif_ast_control_gine",
        ArmFamily.GINE,
        "motif_ast_control_dag",
        CellStatus.UNSUPPORTED,
        "blocked: no authoritative motif-AST artifacts exist; frequent macro motifs are not a fair "
        "motif-AST substitute",
    ),
    GridArm("prefix_transformer", ArmFamily.PREFIX_TRANSFORMER, None, CellStatus.PENDING),
    GridArm("trivial_operator_count", ArmFamily.TRIVIAL, None, CellStatus.PENDING),
)


@dataclass(frozen=True, slots=True)
class GridConfigV1:
    """The fixed-scale plan, frozen before any Goal 6 production execution."""

    dataset_id: str
    input_manifest: str
    input_manifest_sha256: str
    output_directory: str
    gnn_hidden_width: int
    gnn_use_virtual_node: bool
    transformer_hidden_width: int
    epochs: int
    train_pair_count: int
    validation_pair_count: int
    test_pair_count: int

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.input_manifest.strip():
            raise Goal6GridError("dataset_id and input_manifest must be nonblank")
        if not self.input_manifest_sha256.startswith("sha256:"):
            raise Goal6GridError("input_manifest_sha256 must be a qualified SHA-256 digest")
        if self.gnn_hidden_width not in {64, 96} or self.transformer_hidden_width not in {64, 96}:
            raise Goal6GridError("GNN and transformer widths must be exactly 64 or 96")
        if self.epochs < 1 or self.epochs > 30:
            raise Goal6GridError("Goal 6 epochs must remain within the 1--30 cap")
        if min(self.train_pair_count, self.validation_pair_count, self.test_pair_count) < 1:
            raise Goal6GridError("every declared pair split must be nonempty")

    @property
    def config_hash(self) -> str:
        payload = asdict(self)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """One denominator-explicit evaluation view from one completed cell."""

    view: EvaluationView
    attempted: int
    valid: int
    correct: int
    macro_f1: float | None
    calibration_error: float | None

    def __post_init__(self) -> None:
        if self.attempted < 0 or self.valid < 0 or self.correct < 0:
            raise Goal6GridError("evaluation counts must be nonnegative")
        if self.valid > self.attempted or self.correct > self.valid:
            raise Goal6GridError(
                "valid and correct counts must remain within attempted denominator"
            )
        for value in (self.macro_f1, self.calibration_error):
            if value is not None and not 0.0 <= value <= 1.0:
                raise Goal6GridError("macro-F1 and calibration error must be in [0, 1]")

    def as_dict(self) -> dict[str, object]:
        return {
            "accuracy": None if self.valid == 0 else self.correct / self.valid,
            "attempted": self.attempted,
            "calibration_error": self.calibration_error,
            "correct": self.correct,
            "macro_f1": self.macro_f1,
            "valid": self.valid,
            "view": self.view.value,
        }


@dataclass(frozen=True, slots=True)
class GridCell:
    """One independent seed/arm cell; this identity never includes a channel-specific GNN."""

    arm: GridArm
    seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.arm.arm_id}:seed-{self.seed}"


@dataclass(frozen=True, slots=True)
class GridCellResult:
    """One retained terminal row for a fixed Goal 6 cell."""

    cell: GridCell
    status: CellStatus
    evaluations: tuple[EvaluationMetrics, ...]
    parameter_count: int | None
    flop_estimate: int | None
    wall_seconds: float | None
    peak_host_memory_bytes: int | None
    peak_gpu_memory_bytes: int | None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status is CellStatus.COMPLETE:
            expected_views = frozenset(EvaluationView)
            if {item.view for item in self.evaluations} != expected_views:
                raise Goal6GridError(
                    "complete rows must retain every train/validation/IID/OOD view"
                )
            required_evidence = (self.parameter_count, self.flop_estimate, self.wall_seconds)
            if any(value is None for value in required_evidence):
                raise Goal6GridError(
                    "complete rows require parameter, FLOP, and wall-time evidence"
                )
        elif self.evaluations:
            raise Goal6GridError("non-complete rows cannot invent partial evaluation summaries")
        if self.status is CellStatus.UNSUPPORTED and not self.error:
            raise Goal6GridError("unsupported rows require the blocking reason")
        if self.status is CellStatus.FAILED and not self.error:
            raise Goal6GridError("failed rows require the retained failure message")
        for value in (
            self.parameter_count,
            self.flop_estimate,
            self.peak_host_memory_bytes,
            self.peak_gpu_memory_bytes,
        ):
            if value is not None and value < 0:
                raise Goal6GridError("resource counts must be nonnegative")
        if self.wall_seconds is not None and self.wall_seconds < 0:
            raise Goal6GridError("wall_seconds must be nonnegative")

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.cell.arm.arm_id,
            "cell_id": self.cell.cell_id,
            "channel": self.cell.arm.channel,
            "error": self.error,
            "evaluations": [item.as_dict() for item in self.evaluations],
            "family": self.cell.arm.family.value,
            "flop_estimate": self.flop_estimate,
            "parameter_count": self.parameter_count,
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
            "peak_host_memory_bytes": self.peak_host_memory_bytes,
            "seed": self.cell.seed,
            "status": self.status.value,
            "wall_seconds": self.wall_seconds,
        }


CellExecutor = Callable[[GridCell, GridConfigV1], GridCellResult]


def fixed_grid_cells() -> tuple[GridCell, ...]:
    """Return the preregistered six-arm by three-seed, 18-cell execution plan."""

    return tuple(GridCell(arm=arm, seed=seed) for arm in FIXED_ARMS for seed in PRODUCTION_SEEDS)


def _unsupported_result(cell: GridCell) -> GridCellResult:
    return GridCellResult(
        cell=cell,
        status=CellStatus.UNSUPPORTED,
        evaluations=(),
        parameter_count=None,
        flop_estimate=None,
        wall_seconds=None,
        peak_host_memory_bytes=None,
        peak_gpu_memory_bytes=None,
        error=cell.arm.unavailable_reason,
    )


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_grid_manifest(
    config: GridConfigV1,
    *,
    executor: CellExecutor | None = None,
) -> dict[str, object]:
    """Build a plan or execute available cells, retaining unsupported cells explicitly.

    A real executor is deliberately injected by the production launcher after it
    authenticates the pair and tensor manifests.  With no executor this function
    produces only pending/unsupported phase-A planning rows.
    """

    rows: list[GridCellResult] = []
    for cell in fixed_grid_cells():
        if cell.arm.availability is CellStatus.UNSUPPORTED:
            rows.append(_unsupported_result(cell))
        elif executor is None:
            rows.append(
                GridCellResult(
                    cell=cell,
                    status=CellStatus.PENDING,
                    evaluations=(),
                    parameter_count=None,
                    flop_estimate=None,
                    wall_seconds=None,
                    peak_host_memory_bytes=None,
                    peak_gpu_memory_bytes=None,
                )
            )
        else:
            result = executor(cell, config)
            if result.cell != cell:
                raise Goal6GridError("the executor returned a row for a different fixed grid cell")
            rows.append(result)
    return {
        "cells": [row.as_dict() for row in rows],
        "config": asdict(config),
        "config_hash": config.config_hash,
        "phase": (
            "phase_a_planning" if executor is None else "executed_fixture_or_authenticated_run"
        ),
        "schema_version": GRID_SCHEMA_VERSION,
    }


def write_grid_manifest(config: GridConfigV1, *, output_path: Path) -> Path:
    """Atomically publish a phase-A grid manifest without claiming a production run."""

    _atomic_json(output_path, build_grid_manifest(config))
    return output_path


def load_grid_config(path: Path) -> GridConfigV1:
    """Load the strict frozen subset of the checked-in Goal 6 YAML configuration."""

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise Goal6GridError("Goal 6 grid config must be a mapping")
    required = {
        "dataset_id",
        "input_manifest",
        "input_manifest_sha256",
        "output_directory",
        "gnn",
        "transformer",
        "epochs",
        "pair_counts",
    }
    missing = required - set(loaded)
    if missing:
        raise Goal6GridError(f"Goal 6 grid config is missing keys: {sorted(missing)}")
    gnn = loaded["gnn"]
    transformer = loaded["transformer"]
    pair_counts = loaded["pair_counts"]
    if not all(isinstance(value, dict) for value in (gnn, transformer, pair_counts)):
        raise Goal6GridError("gnn, transformer, and pair_counts must be mappings")
    return GridConfigV1(
        dataset_id=str(loaded["dataset_id"]),
        input_manifest=str(loaded["input_manifest"]),
        input_manifest_sha256=str(loaded["input_manifest_sha256"]),
        output_directory=str(loaded["output_directory"]),
        gnn_hidden_width=int(gnn["hidden_width"]),
        gnn_use_virtual_node=bool(gnn["use_virtual_node"]),
        transformer_hidden_width=int(transformer["hidden_width"]),
        epochs=int(loaded["epochs"]),
        train_pair_count=int(pair_counts["train"]),
        validation_pair_count=int(pair_counts["validation"]),
        test_pair_count=int(pair_counts["test_iid"]),
    )


def main() -> None:
    """Write a plan only; production execution requires an authenticated external launcher."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_grid_manifest(load_grid_config(args.config), output_path=args.output)


if __name__ == "__main__":  # pragma: no cover - CLI wiring.
    main()
