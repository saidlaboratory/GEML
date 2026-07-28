"""Phase-A fixed-cell planner for the Goal 7 verifier-valid rewrite-policy grid."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from geml.learning.harness.seeds import PRODUCTION_SEEDS

GRID_SCHEMA_VERSION = "geml-goal7-grid-v1"
FIXTURE_STEP_COUNT = 25


class Goal7GridError(ValueError):
    """A fixed policy cell or retained result row violates the preregistered grid."""


class PolicyFamily(StrEnum):
    GINE = "gine"
    PREFIX = "prefix_transformer"
    UNIFORM = "uniform_legal_action"


class CellStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class PolicyArm:
    arm_id: str
    family: PolicyFamily
    channel: str | None
    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.family is PolicyFamily.GINE and self.channel is None:
            raise Goal7GridError("GINE arms require one declared graph channel")
        if self.family is not PolicyFamily.GINE and self.channel is not None:
            raise Goal7GridError("non-GINE controls cannot declare graph channels")
        if self.available == (self.reason is not None):
            raise Goal7GridError("arm availability and reason must agree")


FIXED_ARMS = (
    PolicyArm("ast_gine", PolicyFamily.GINE, "ast_dag", True),
    PolicyArm("pure_eml_gine", PolicyFamily.GINE, "pure_eml_dag", True),
    PolicyArm("frequent_macro_motif_gine", PolicyFamily.GINE, "frequent_macro_motif_dag", True),
    PolicyArm(
        "motif_ast_control_gine",
        PolicyFamily.GINE,
        "motif_ast_fair_control",
        False,
        "blocked: no authoritative motif-AST fair-control artifact exists",
    ),
    PolicyArm("prefix_transformer", PolicyFamily.PREFIX, None, True),
    PolicyArm("uniform_legal_action", PolicyFamily.UNIFORM, None, True),
)


@dataclass(frozen=True, slots=True)
class Goal7ConfigV1:
    step_manifest: str
    step_manifest_sha256: str
    output_directory: str
    epochs: int
    top_k: int

    def __post_init__(self) -> None:
        if not self.step_manifest.strip() or not self.output_directory.strip():
            raise Goal7GridError("manifest and output paths must be nonblank")
        if not self.step_manifest_sha256.startswith("sha256:"):
            raise Goal7GridError("step manifest checksum must be SHA-256 qualified")
        if not 1 <= self.epochs <= 30 or self.top_k < 1:
            raise Goal7GridError("epochs must be 1--30 and top_k must be positive")

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class PolicyCell:
    arm: PolicyArm
    seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.arm.arm_id}:seed-{self.seed}"


@dataclass(frozen=True, slots=True)
class PolicyCellResult:
    cell: PolicyCell
    status: CellStatus
    attempted: int
    valid: int
    verifier_valid_top_k: int
    exact_top_k: int
    invalid: int
    no_action: int
    parameter_count: int | None
    flop_estimate: int | None
    error: str | None = None

    def __post_init__(self) -> None:
        counts = (
            self.attempted,
            self.valid,
            self.verifier_valid_top_k,
            self.exact_top_k,
            self.invalid,
            self.no_action,
        )
        if min(counts) < 0:
            raise Goal7GridError("policy counts must be nonnegative")
        if self.valid > self.attempted or self.verifier_valid_top_k > self.valid:
            raise Goal7GridError("policy counts violate their attempted/valid denominator")
        if self.status is CellStatus.COMPLETE and self.parameter_count is None:
            raise Goal7GridError("complete cells require computed parameter evidence")
        if self.status in {CellStatus.FAILED, CellStatus.UNSUPPORTED} and not self.error:
            raise Goal7GridError("failed and unsupported cells require retained detail")

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.cell.arm.arm_id,
            "attempted": self.attempted,
            "cell_id": self.cell.cell_id,
            "channel": self.cell.arm.channel,
            "error": self.error,
            "exact_top_k": self.exact_top_k,
            "family": self.cell.arm.family.value,
            "flop_estimate": self.flop_estimate,
            "invalid": self.invalid,
            "no_action": self.no_action,
            "parameter_count": self.parameter_count,
            "seed": self.cell.seed,
            "status": self.status.value,
            "valid": self.valid,
            "verifier_valid_top_k": self.verifier_valid_top_k,
        }


CellExecutor = Callable[[PolicyCell, Goal7ConfigV1], PolicyCellResult]


def fixed_cells() -> tuple[PolicyCell, ...]:
    return tuple(PolicyCell(arm=arm, seed=seed) for arm in FIXED_ARMS for seed in PRODUCTION_SEEDS)


def build_grid_manifest(
    config: Goal7ConfigV1,
    *,
    executor: CellExecutor | None = None,
) -> dict[str, object]:
    """Return all 18 planning rows, or injected authenticated/fixture execution rows."""

    rows: list[PolicyCellResult] = []
    for cell in fixed_cells():
        if not cell.arm.available:
            rows.append(
                PolicyCellResult(
                    cell,
                    CellStatus.UNSUPPORTED,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    cell.arm.reason,
                )
            )
        elif executor is None:
            rows.append(PolicyCellResult(cell, CellStatus.PENDING, 0, 0, 0, 0, 0, 0, None, None))
        else:
            result = executor(cell, config)
            if result.cell != cell:
                raise Goal7GridError("executor returned a result for a different fixed cell")
            rows.append(result)
    return {
        "cells": [row.as_dict() for row in rows],
        "config": asdict(config),
        "config_hash": config.config_hash,
        "phase": ("phase_a_planning" if executor is None else "fixture_or_authenticated_execution"),
        "schema_version": GRID_SCHEMA_VERSION,
    }


def write_grid_manifest(config: Goal7ConfigV1, output_path: Path) -> Path:
    payload = json.dumps(build_grid_manifest(config), indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


def load_grid_config(path: Path) -> Goal7ConfigV1:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Goal7GridError("Goal 7 config must be a mapping")
    return Goal7ConfigV1(
        step_manifest=str(payload["step_manifest"]),
        step_manifest_sha256=str(payload["step_manifest_sha256"]),
        output_directory=str(payload["output_directory"]),
        epochs=int(payload["epochs"]),
        top_k=int(payload["top_k"]),
    )
