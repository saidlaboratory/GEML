"""Issue 6-5: the fixed six-arm Goal 6 equivalence grid.

The grid is four graph channels through **one** shared encoder, one compute-matched prefix
transformer, and one transparent trivial floor, each on exactly three preregistered seeds.

Two contracts are enforced here rather than assumed:

**Channel names are never hard-coded.**
    Issue 6-2/#56 carries an unresolved contradiction: the issue names a "frequent-motif EML-DAG"
    and a motif-AST fair control, but the immutable Goal 5 export contains a macro-derived
    ``frequent_motif_dag`` and no motif-AST channel at all.  This module therefore consumes channel
    identities from the merged registry and refuses to run production without four aligned,
    approved channels.  It will not invent a fourth channel, and it will not relabel a macro-derived
    channel as pure EML.

**There is no single cross-channel alpha.**
    Pure-EML alpha, ordinary node/edge size, macro size, and dictionary-inclusive motif MDL are
    different quantities.  Reporting them under one column would manufacture a comparison that does
    not exist.  Structural metrics are therefore named per channel and carry an explicit
    :attr:`StructuralMetric.comparable_across_channels` flag.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from geml.learning.harness.config import HarnessConfig, canonical_json_bytes
from geml.learning.harness.seeds import PRODUCTION_SEEDS
from geml.learning.harness.train import CellStatus, atomic_write_json

GRID_SCHEMA_VERSION = "geml-goal6-grid-v1"
GRID_ROW_SCHEMA_VERSION = "geml-goal6-grid-row-v1"

#: The frozen production output root for issue 6-5.
PRODUCTION_GRID_ROOT = "outputs/final/goal6/grid"

#: The grid is exactly four graph channels plus two non-graph controls.
REQUIRED_GRAPH_CHANNEL_COUNT = 4


class GridContractError(RuntimeError):
    """The grid cannot run without violating a frozen scientific contract."""


class ArmFamily(StrEnum):
    """The three backbone families; graph arms differ only by input channel."""

    GRAPH = "graph"
    PREFIX_TRANSFORMER = "prefix_transformer"
    TRIVIAL_FLOOR = "trivial_floor"


class EvaluationView(StrEnum):
    """Evaluation views.  The two strict OOD views are conditional, not automatic."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST_IID = "test_iid"
    TEST_OOD_STRESS = "test_ood_stress"
    TEST_DEPTH_OOD = "test_depth_ood"
    TEST_FAMILY_OOD = "test_family_ood"


class ViewStatus(StrEnum):
    """Whether a view may be reported as the thing its name claims."""

    AVAILABLE = "available"
    UNSUPPORTED_NOT_DISJOINT = "unsupported_not_disjoint"
    UNSUPPORTED_NOT_EXCLUDED = "unsupported_not_excluded"
    MISSING = "missing"


@dataclass(frozen=True)
class StructuralMetric:
    """One structural quantity, named honestly and flagged for comparability."""

    name: str
    value: float
    comparable_across_channels: bool
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "comparable_across_channels": self.comparable_across_channels,
            "note": self.note,
        }


@dataclass(frozen=True)
class ChannelSpec:
    """One approved graph channel as declared by the issue 6-2/#56 registry."""

    channel_name: str
    representation_family: str
    representation_mode: str
    is_pure_eml: bool
    approved: bool = False
    label_vocabulary_size: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "channel_name": self.channel_name,
            "representation_family": self.representation_family,
            "representation_mode": self.representation_mode,
            "is_pure_eml": self.is_pure_eml,
            "approved": self.approved,
            "label_vocabulary_size": self.label_vocabulary_size,
        }


@runtime_checkable
class ChannelRegistry(Protocol):
    """The merged issue 6-2/#56 registry this runner consumes.

    During Phase A this is satisfied by a fixture; after that branch merges, the canonical registry
    should satisfy it structurally and the fixture becomes test-only code.
    """

    def approved_channels(self) -> Sequence[ChannelSpec]:
        """Return the approved, aligned graph channels."""

    def aligned_pair_ids(self, channel_name: str) -> Sequence[str]:
        """Return the accepted pair identities materialized for ``channel_name``."""

    def view_status(self, view: str) -> ViewStatus:
        """Return whether ``view`` may be reported under its own name."""


def validate_channels(registry: ChannelRegistry) -> tuple[ChannelSpec, ...]:
    """Return the four approved channels, or refuse to proceed.

    Refusal is the correct outcome while issue 6-2/#56 is unresolved.  Substituting a macro-DAG for
    the missing motif-AST fair control, or running with three channels and calling it four, would
    both produce a result table that misrepresents what was compared.
    """

    channels = tuple(registry.approved_channels())
    approved = tuple(channel for channel in channels if channel.approved)
    if len(approved) != REQUIRED_GRAPH_CHANNEL_COUNT:
        raise GridContractError(
            f"the Goal 6 grid requires exactly {REQUIRED_GRAPH_CHANNEL_COUNT} approved aligned "
            f"graph channels, found {len(approved)}; issue 6-2/#56 must be resolved before "
            "production. Do not substitute a macro-derived channel for the motif-AST fair control"
        )
    names = [channel.channel_name for channel in approved]
    if len(set(names)) != len(names):
        raise GridContractError(f"approved channel names must be unique, got {names}")
    for channel in approved:
        if channel.is_pure_eml and "macro" in channel.representation_mode:
            raise GridContractError(
                f"channel {channel.channel_name!r} claims strict pure EML but carries the macro "
                f"representation mode {channel.representation_mode!r}; the Goal 5 motif channels "
                "end with ':macro:macro:official_v4:is_pure_eml=false' and may not be relabelled"
            )
    return approved


def validate_alignment(
    registry: ChannelRegistry, channels: Sequence[ChannelSpec]
) -> tuple[str, ...]:
    """Return the shared pair identities, refusing to compare different subsets."""

    if not channels:
        raise GridContractError("cannot validate alignment with zero channels")
    reference = tuple(registry.aligned_pair_ids(channels[0].channel_name))
    for channel in channels[1:]:
        observed = tuple(registry.aligned_pair_ids(channel.channel_name))
        if observed != reference:
            missing = sorted(set(reference) - set(observed))
            extra = sorted(set(observed) - set(reference))
            raise GridContractError(
                f"channel {channel.channel_name!r} is not aligned with "
                f"{channels[0].channel_name!r}: {len(missing)} missing and {len(extra)} extra pair "
                "identities. Misaligned pairs belong in the failure ledger; the arms must never be "
                "scored on different subsets"
            )
    return reference


@dataclass(frozen=True)
class ArmSpec:
    """One of the six arms."""

    arm_id: str
    family: ArmFamily
    channel: ChannelSpec | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "family": str(self.family),
            "channel": self.channel.as_dict() if self.channel else None,
        }


def build_arms(channels: Sequence[ChannelSpec]) -> tuple[ArmSpec, ...]:
    """Build exactly six arms: four graph channels and the two non-graph controls."""

    if len(channels) != REQUIRED_GRAPH_CHANNEL_COUNT:
        raise GridContractError(
            f"expected {REQUIRED_GRAPH_CHANNEL_COUNT} graph channels, got {len(channels)}"
        )
    arms = [
        ArmSpec(arm_id=f"graph::{channel.channel_name}", family=ArmFamily.GRAPH, channel=channel)
        for channel in channels
    ]
    arms.append(ArmSpec(arm_id="prefix_transformer", family=ArmFamily.PREFIX_TRANSFORMER))
    arms.append(ArmSpec(arm_id="trivial_floor", family=ArmFamily.TRIVIAL_FLOOR))
    return tuple(arms)


@dataclass
class GridRow:
    """One (arm, seed) result row with complete denominators and provenance."""

    arm_id: str
    family: ArmFamily
    channel_name: str | None
    representation_mode: str | None
    seed: int
    status: CellStatus
    config_hash: str
    commit: str
    metrics_by_view: dict[str, dict[str, float]] = field(default_factory=dict)
    view_status: dict[str, str] = field(default_factory=dict)
    denominators_by_view: dict[str, dict[str, int]] = field(default_factory=dict)
    structural_metrics: tuple[StructuralMetric, ...] = ()
    parameters: int | None = None
    forward_flops: int | None = None
    wall_seconds: float | None = None
    peak_host_memory_bytes: int | None = None
    peak_device_memory_bytes: int | None = None
    input_graph_statistics: dict[str, float] = field(default_factory=dict)
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": GRID_ROW_SCHEMA_VERSION,
            "arm_id": self.arm_id,
            "family": str(self.family),
            "channel_name": self.channel_name,
            "representation_mode": self.representation_mode,
            "seed": self.seed,
            "status": str(self.status),
            "config_hash": self.config_hash,
            "commit": self.commit,
            "metrics_by_view": {k: dict(v) for k, v in sorted(self.metrics_by_view.items())},
            "view_status": dict(sorted(self.view_status.items())),
            "denominators_by_view": {
                k: dict(v) for k, v in sorted(self.denominators_by_view.items())
            },
            "structural_metrics": [metric.as_dict() for metric in self.structural_metrics],
            "parameters": self.parameters,
            "forward_flops": self.forward_flops,
            "wall_seconds": self.wall_seconds,
            "peak_host_memory_bytes": self.peak_host_memory_bytes,
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
            "input_graph_statistics": dict(sorted(self.input_graph_statistics.items())),
            "failure_reason": self.failure_reason,
        }


def grid_cell_id(arm: ArmSpec, seed: int) -> str:
    """Return a filesystem-safe, collision-free identity for one grid cell."""

    return f"{arm.arm_id.replace('::', '__')}__seed-{seed}"


@dataclass
class GridManifest:
    """The frozen description of what the grid intends to run."""

    arms: tuple[ArmSpec, ...]
    seeds: tuple[int, ...]
    commit: str
    aligned_pair_count: int
    view_status: dict[str, str]
    is_fixture: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": GRID_SCHEMA_VERSION,
            "arms": [arm.as_dict() for arm in self.arms],
            "seeds": list(self.seeds),
            "commit": self.commit,
            "aligned_pair_count": self.aligned_pair_count,
            "view_status": dict(sorted(self.view_status.items())),
            "is_fixture": self.is_fixture,
            "cell_count": len(self.arms) * len(self.seeds),
        }

    @property
    def manifest_digest(self) -> str:
        import hashlib

        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


def build_manifest(
    registry: ChannelRegistry,
    commit: str,
    seeds: Sequence[int] = PRODUCTION_SEEDS,
    *,
    is_fixture: bool = False,
) -> GridManifest:
    """Validate every contract, then freeze the grid manifest."""

    if not is_fixture and tuple(seeds) != PRODUCTION_SEEDS:
        raise GridContractError(
            f"production runs use exactly the preregistered seeds {PRODUCTION_SEEDS}; a seed is "
            "never swapped because its result is unfavorable"
        )
    channels = validate_channels(registry)
    aligned = validate_alignment(registry, channels)
    return GridManifest(
        arms=build_arms(channels),
        seeds=tuple(seeds),
        commit=commit,
        aligned_pair_count=len(aligned),
        view_status={view.value: str(registry.view_status(view.value)) for view in EvaluationView},
        is_fixture=is_fixture,
    )


@runtime_checkable
class CellRunner(Protocol):
    """Executes one (arm, seed) cell.  Injected so the grid never owns model or data code."""

    def run_cell(self, arm: ArmSpec, seed: int, config: HarnessConfig) -> GridRow:
        """Run one cell and return its row, including explicit failure rows."""


class GridRunner:
    """Drive the six-arm grid as independent, resumable, content-addressed cells."""

    def __init__(self, manifest: GridManifest, root: Path, runner: CellRunner) -> None:
        self.manifest = manifest
        self.root = Path(root)
        self.runner = runner

    def row_path(self, arm: ArmSpec, seed: int) -> Path:
        return self.root / "cells" / f"{grid_cell_id(arm, seed)}.json"

    def existing_row(self, arm: ArmSpec, seed: int) -> dict[str, object] | None:
        path = self.row_path(arm, seed)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def run(self, config_for: dict[str, HarnessConfig]) -> list[GridRow]:
        """Run every outstanding cell, preserving existing complete and failed evidence."""

        rows: list[GridRow] = []
        for arm in self.manifest.arms:
            for seed in self.manifest.seeds:
                existing = self.existing_row(arm, seed)
                if existing is not None and existing["status"] == str(CellStatus.COMPLETE):
                    rows.append(_row_from_dict(existing))
                    continue
                row = self.runner.run_cell(arm, seed, config_for[arm.arm_id])
                atomic_write_json(self.row_path(arm, seed), row.as_dict())
                rows.append(row)
        self.write_manifest()
        return rows

    def write_manifest(self) -> None:
        atomic_write_json(self.root / "grid.manifest.json", self.manifest.as_dict())

    def completeness_report(self) -> dict[str, object]:
        """Report which of the expected cells exist, without inventing the missing ones."""

        expected = [(arm, seed) for arm in self.manifest.arms for seed in self.manifest.seeds]
        present = {
            grid_cell_id(arm, seed) for arm, seed in expected if self.row_path(arm, seed).exists()
        }
        missing = [
            grid_cell_id(arm, seed)
            for arm, seed in expected
            if grid_cell_id(arm, seed) not in present
        ]
        return {
            "expected_cells": len(expected),
            "present_cells": len(present),
            "missing_cells": sorted(missing),
            "complete": not missing,
        }


def _row_from_dict(payload: dict[str, object]) -> GridRow:
    """Rebuild a row from disk without losing any recorded evidence.

    Every field written by :meth:`GridRow.as_dict` is restored.  Dropping structural metrics or
    memory telemetry here would silently shrink the evidence of an already-complete cell the next
    time the grid resumed, which is exactly the kind of quiet loss the reporting policy forbids.
    """

    return GridRow(
        arm_id=str(payload["arm_id"]),
        family=ArmFamily(str(payload["family"])),
        channel_name=payload.get("channel_name"),  # type: ignore[arg-type]
        representation_mode=payload.get("representation_mode"),  # type: ignore[arg-type]
        seed=int(payload["seed"]),  # type: ignore[arg-type]
        status=CellStatus(str(payload["status"])),
        config_hash=str(payload["config_hash"]),
        commit=str(payload["commit"]),
        metrics_by_view={
            str(view): dict(metrics)
            for view, metrics in dict(payload.get("metrics_by_view", {})).items()  # type: ignore[arg-type]
        },
        view_status=dict(payload.get("view_status", {})),  # type: ignore[arg-type]
        denominators_by_view={
            str(view): dict(counts)
            for view, counts in dict(payload.get("denominators_by_view", {})).items()  # type: ignore[arg-type]
        },
        structural_metrics=tuple(
            StructuralMetric(
                name=str(entry["name"]),
                value=float(entry["value"]),
                comparable_across_channels=bool(entry["comparable_across_channels"]),
                note=str(entry.get("note", "")),
            )
            for entry in list(payload.get("structural_metrics", ()))  # type: ignore[arg-type]
        ),
        parameters=payload.get("parameters"),  # type: ignore[arg-type]
        forward_flops=payload.get("forward_flops"),  # type: ignore[arg-type]
        wall_seconds=payload.get("wall_seconds"),  # type: ignore[arg-type]
        peak_host_memory_bytes=payload.get("peak_host_memory_bytes"),  # type: ignore[arg-type]
        peak_device_memory_bytes=payload.get("peak_device_memory_bytes"),  # type: ignore[arg-type]
        input_graph_statistics=dict(payload.get("input_graph_statistics", {})),  # type: ignore[arg-type]
        failure_reason=payload.get("failure_reason"),  # type: ignore[arg-type]
    )
