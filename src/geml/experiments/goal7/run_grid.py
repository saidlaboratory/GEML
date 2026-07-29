"""Deterministic, resumable Phase-A runner contract for the Goal 7 grid.

Production training is deliberately injected through :class:`GridCellExecutor`.
Workstream 2 owns the shared training harness and model run envelope; this
module owns only Goal 7 cell identity, equal-grid enforcement, immutable
results, and retained per-step metric evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import yaml

CONFIG_SCHEMA_VERSION = "geml-goal7-grid-config-v1"
CELL_SCHEMA_VERSION = "geml-goal7-grid-cell-v1"
PLAN_SCHEMA_VERSION = "geml-goal7-grid-plan-v1"
RUN_SCHEMA_VERSION = "geml-goal7-grid-run-v1"
STEP_MANIFEST_AUTH_SCHEMA_VERSION = "geml-goal7-step-manifest-authentication-v1"
UNIFORM_DRAW_AUDIT_SCHEMA_VERSION = "geml-goal7-uniform-draw-audit-v1"
PRODUCTION_SEEDS = (20260726, 20260727, 20260728)

_CONFIG_DOMAIN = b"geml-goal7-grid-config-v1\0"
_CELL_ID_DOMAIN = b"geml-goal7-grid-cell-id-v1\0"
_CELL_CONTENT_DOMAIN = b"geml-goal7-grid-cell-v1\0"
_PLAN_CONTENT_DOMAIN = b"geml-goal7-grid-plan-v1\0"
_RUN_ID_DOMAIN = b"geml-goal7-grid-run-id-v1\0"
_RUN_CONTENT_DOMAIN = b"geml-goal7-grid-run-v1\0"
_STEP_POPULATION_DOMAIN = b"geml-goal7-step-population-v1\0"
_UNIFORM_ORDER_DOMAIN = b"geml-goal7-uniform-valid-order-v1\0"
_HEX = frozenset("0123456789abcdef")
_STEP_POPULATION_FIELDS = (
    "record_id",
    "trace_id",
    "source_group",
    "lineage_group_ids",
    "authoritative_split",
    "current_signature",
    "goal_signature",
    "target_successor_signature",
    "current_family",
    "goal_family",
    "evaluation_views",
    "remaining_witness_steps",
    "trace_length",
    "demonstration_action",
)

type JsonValue = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None


class Goal7ProtocolError(ValueError):
    """A Goal 7 config, cell, or persisted run violates its frozen contract."""


class GridStage(StrEnum):
    FIXTURE = "fixture"
    PRODUCTION = "production"


class ProposerFamily(StrEnum):
    GNN = "gnn"
    TRANSFORMER = "transformer"
    UNIFORM_VALID = "uniform_valid"


class GridCellStatus(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    INTERRUPTED = "interrupted"


class BudgetStopReason(StrEnum):
    COMPLETED = "completed"
    EARLY_STOPPING = "early_stopping"
    EPOCH_LIMIT = "epoch_limit"
    OPTIMIZER_STEP_LIMIT = "optimizer_step_limit"
    WALL_TIME_LIMIT = "wall_time_limit"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class GraphChannelSpec:
    """One honestly labeled graph channel in the frozen comparison."""

    channel_id: str
    representation_mode: str
    enabled: bool
    blocker: str | None = None

    def __post_init__(self) -> None:
        _nonblank(self.channel_id, field="channel_id")
        _nonblank(self.representation_mode, field="representation_mode")
        if type(self.enabled) is not bool:
            raise TypeError("channel enabled must be a strict boolean")
        if self.blocker is not None:
            _nonblank(self.blocker, field="channel blocker")
        if self.enabled and self.blocker is not None:
            raise Goal7ProtocolError("an enabled channel cannot retain a blocker")
        if not self.enabled and self.blocker is None:
            raise Goal7ProtocolError("a disabled channel requires an explicit blocker")


@dataclass(frozen=True, slots=True)
class GridBudgetV1:
    """Budget fields that every Goal 7 arm receives unchanged."""

    maximum_epochs: int
    early_stopping_patience: int
    maximum_optimizer_steps: int
    node_edge_batch_budget: int
    wall_time_seconds: float
    top_k: tuple[int, ...]
    parameter_match_tolerance_fraction: float
    flop_match_tolerance_fraction: float
    comparison_unit: str

    def __post_init__(self) -> None:
        for name in (
            "maximum_epochs",
            "early_stopping_patience",
            "maximum_optimizer_steps",
            "node_edge_batch_budget",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise TypeError(f"{name} must be a positive exact integer")
        if (
            type(self.wall_time_seconds) is not float
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds <= 0
        ):
            raise TypeError("wall_time_seconds must be a positive finite float")
        if (
            type(self.top_k) is not tuple
            or not self.top_k
            or any(type(value) is not int or value <= 0 for value in self.top_k)
            or tuple(sorted(set(self.top_k))) != self.top_k
        ):
            raise TypeError("top_k must be a nonempty sorted tuple of unique positive integers")
        for name in (
            "parameter_match_tolerance_fraction",
            "flop_match_tolerance_fraction",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1:
                raise TypeError(f"{name} must be a finite float in [0, 1]")
        _nonblank(self.comparison_unit, field="comparison_unit")

    @property
    def digest(self) -> str:
        return _sha256_json(self.as_dict())

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "maximum_epochs": self.maximum_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "maximum_optimizer_steps": self.maximum_optimizer_steps,
            "node_edge_batch_budget": self.node_edge_batch_budget,
            "wall_time_seconds": self.wall_time_seconds,
            "top_k": list(self.top_k),
            "parameter_match_tolerance_fraction": (self.parameter_match_tolerance_fraction),
            "flop_match_tolerance_fraction": self.flop_match_tolerance_fraction,
            "comparison_unit": self.comparison_unit,
        }


@dataclass(frozen=True, slots=True)
class BudgetConsumptionV1:
    """Observed training exposure retained for one completed cell."""

    epochs_completed: int
    optimizer_steps_completed: int
    maximum_observed_node_edge_batch: int
    early_stopping_bad_epochs: int
    stop_reason: BudgetStopReason

    def __post_init__(self) -> None:
        for field in (
            "epochs_completed",
            "optimizer_steps_completed",
            "maximum_observed_node_edge_batch",
            "early_stopping_bad_epochs",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise TypeError(f"{field} must be a nonnegative exact integer")
        if not isinstance(self.stop_reason, BudgetStopReason):
            raise TypeError("stop_reason must be a BudgetStopReason")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "early_stopping_bad_epochs": self.early_stopping_bad_epochs,
            "epochs_completed": self.epochs_completed,
            "maximum_observed_node_edge_batch": (self.maximum_observed_node_edge_batch),
            "optimizer_steps_completed": self.optimizer_steps_completed,
            "stop_reason": self.stop_reason.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> BudgetConsumptionV1:
        expected = {
            "early_stopping_bad_epochs",
            "epochs_completed",
            "maximum_observed_node_edge_batch",
            "optimizer_steps_completed",
            "stop_reason",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise Goal7ProtocolError("budget-consumption fields are incompatible")
        try:
            stop_reason = BudgetStopReason(value["stop_reason"])
        except (TypeError, ValueError) as error:
            raise Goal7ProtocolError("budget consumption has an invalid stop reason") from error
        return cls(
            epochs_completed=value["epochs_completed"],  # type: ignore[arg-type]
            optimizer_steps_completed=value["optimizer_steps_completed"],  # type: ignore[arg-type]
            maximum_observed_node_edge_batch=value[  # type: ignore[arg-type]
                "maximum_observed_node_edge_batch"
            ],
            early_stopping_bad_epochs=value["early_stopping_bad_epochs"],  # type: ignore[arg-type]
            stop_reason=stop_reason,
        )


@dataclass(frozen=True, slots=True)
class Goal7GridConfig:
    """Frozen scientific identity for one Goal 7 grid."""

    schema_version: str
    stage: GridStage
    output_root: str
    seeds: tuple[int, ...]
    expected_step_count: int | None
    step_manifest: str | None
    step_manifest_sha256: str | None
    rule_registry_sha256: str | None
    verifier_sha256: str | None
    shared_harness_sha256: str | None
    shared_gnn_architecture_sha256: str | None
    transformer_architecture_sha256: str | None
    compute_reference_sha256: str | None
    implementation_sha256: str | None
    channel_contract_resolved: bool
    channels: tuple[GraphChannelSpec, ...]
    budget: GridBudgetV1
    reproduction_command: str | None
    selection_split: str = "validation"
    evaluation_splits: tuple[str, ...] = ("test_iid", "test_ood")
    training_config_sha256: str | None = None
    training_family_inventory_sha256: str | None = None
    step_population_sha256: str | None = None
    analysis_reproduction_command: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise Goal7ProtocolError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
        if type(self.stage) is not GridStage:
            raise TypeError("stage must be GridStage")
        _nonblank(self.output_root, field="output_root")
        if (
            type(self.seeds) is not tuple
            or any(type(seed) is not int or seed < 0 for seed in self.seeds)
            or tuple(sorted(set(self.seeds))) != self.seeds
        ):
            raise TypeError("seeds must be a sorted tuple of unique nonnegative integers")
        if self.seeds != PRODUCTION_SEEDS:
            raise Goal7ProtocolError(f"Goal 7 seeds must be exactly {PRODUCTION_SEEDS}")
        if self.expected_step_count is not None and (
            type(self.expected_step_count) is not int or self.expected_step_count <= 0
        ):
            raise TypeError("expected_step_count must be a positive exact integer or None")
        if type(self.channel_contract_resolved) is not bool:
            raise TypeError("channel_contract_resolved must be a strict boolean")
        if (
            type(self.channels) is not tuple
            or not self.channels
            or any(not isinstance(channel, GraphChannelSpec) for channel in self.channels)
        ):
            raise TypeError("channels must be a nonempty tuple of GraphChannelSpec values")
        channel_ids = tuple(channel.channel_id for channel in self.channels)
        if channel_ids != tuple(sorted(set(channel_ids))):
            raise Goal7ProtocolError("channels must be uniquely sorted by channel_id")
        enabled = tuple(channel for channel in self.channels if channel.enabled)
        if self.channel_contract_resolved and len(enabled) != 4:
            raise Goal7ProtocolError(
                "a resolved Goal 7 channel contract requires exactly four arms"
            )
        if not isinstance(self.budget, GridBudgetV1):
            raise TypeError("budget must be GridBudgetV1")
        if self.reproduction_command is not None:
            _nonblank(self.reproduction_command, field="reproduction_command")
            if "{cell_id}" not in self.reproduction_command:
                raise Goal7ProtocolError("reproduction_command must contain {cell_id}")
        if self.analysis_reproduction_command is not None:
            _nonblank(
                self.analysis_reproduction_command,
                field="analysis_reproduction_command",
            )
            if "{run_id}" not in self.analysis_reproduction_command:
                raise Goal7ProtocolError("analysis_reproduction_command must contain {run_id}")
            try:
                rendered_analysis_command = self.analysis_reproduction_command.format(
                    run_id="0" * 64
                )
            except (KeyError, ValueError) as error:
                raise Goal7ProtocolError(
                    "analysis_reproduction_command has invalid template fields"
                ) from error
            if "{" in rendered_analysis_command or "}" in rendered_analysis_command:
                raise Goal7ProtocolError(
                    "analysis_reproduction_command has unresolved template fields"
                )
        if self.selection_split != "validation":
            raise Goal7ProtocolError("Goal 7 model selection is validation-only")
        if self.evaluation_splits != ("test_iid", "test_ood"):
            raise Goal7ProtocolError(
                "Goal 7 evaluation splits must be exactly test_iid and test_ood"
            )
        for field in (
            "step_manifest_sha256",
            "rule_registry_sha256",
            "verifier_sha256",
            "shared_harness_sha256",
            "shared_gnn_architecture_sha256",
            "transformer_architecture_sha256",
            "compute_reference_sha256",
            "implementation_sha256",
            "training_config_sha256",
            "training_family_inventory_sha256",
            "step_population_sha256",
        ):
            value = getattr(self, field)
            if value is not None:
                _require_sha256(value, field=field)
        if self.step_manifest is not None:
            _nonblank(self.step_manifest, field="step_manifest")
        if (self.step_manifest is None) != (self.step_manifest_sha256 is None):
            raise Goal7ProtocolError("step manifest path and digest must be resolved together")

    @property
    def enabled_channels(self) -> tuple[GraphChannelSpec, ...]:
        return tuple(channel for channel in self.channels if channel.enabled)

    @property
    def digest(self) -> str:
        return hashlib.sha256(_CONFIG_DOMAIN + _canonical_json(self.identity_payload())).hexdigest()

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "output_root": self.output_root,
            "seeds": list(self.seeds),
            "expected_step_count": self.expected_step_count,
            "step_manifest": self.step_manifest,
            "step_manifest_sha256": self.step_manifest_sha256,
            "rule_registry_sha256": self.rule_registry_sha256,
            "verifier_sha256": self.verifier_sha256,
            "shared_harness_sha256": self.shared_harness_sha256,
            "shared_gnn_architecture_sha256": self.shared_gnn_architecture_sha256,
            "transformer_architecture_sha256": self.transformer_architecture_sha256,
            "compute_reference_sha256": self.compute_reference_sha256,
            "implementation_sha256": self.implementation_sha256,
            "channel_contract_resolved": self.channel_contract_resolved,
            "channels": [
                {
                    "channel_id": channel.channel_id,
                    "representation_mode": channel.representation_mode,
                    "enabled": channel.enabled,
                    "blocker": channel.blocker,
                }
                for channel in self.channels
            ],
            "budget": self.budget.as_dict(),
            "reproduction_command": self.reproduction_command,
            "selection_split": self.selection_split,
            "evaluation_splits": list(self.evaluation_splits),
            "training_config_sha256": self.training_config_sha256,
            "training_family_inventory_sha256": (self.training_family_inventory_sha256),
            "step_population_sha256": self.step_population_sha256,
            "analysis_reproduction_command": self.analysis_reproduction_command,
        }

    def production_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.expected_step_count is None:
            blockers.append("expected_step_count is unresolved")
        if self.reproduction_command is None:
            blockers.append("reproduction_command is unresolved")
        if self.analysis_reproduction_command is None:
            blockers.append("analysis_reproduction_command is unresolved")
        if not self.channel_contract_resolved:
            blockers.append("issue #56 four-channel contract is unresolved")
        if len(self.enabled_channels) != 4:
            blockers.append("exactly four approved enabled graph channels are required")
        for field in (
            "step_manifest",
            "step_manifest_sha256",
            "rule_registry_sha256",
            "verifier_sha256",
            "shared_harness_sha256",
            "shared_gnn_architecture_sha256",
            "transformer_architecture_sha256",
            "compute_reference_sha256",
            "implementation_sha256",
            "training_config_sha256",
            "training_family_inventory_sha256",
            "step_population_sha256",
        ):
            value = getattr(self, field)
            if value is None:
                blockers.append(f"{field} is unresolved")
        return tuple(blockers)

    def require_runnable(self) -> None:
        if self.stage is GridStage.PRODUCTION:
            blockers = self.production_blockers()
            if blockers:
                raise Goal7ProtocolError(
                    "production Goal 7 config is not runnable: " + "; ".join(blockers)
                )


@dataclass(frozen=True, slots=True)
class GridArmV1:
    """One controlled proposer arm derived from the frozen channel contract."""

    arm_id: str
    proposer_family: ProposerFamily
    channel_id: str | None
    representation_mode: str | None

    def __post_init__(self) -> None:
        _nonblank(self.arm_id, field="arm_id")
        if type(self.proposer_family) is not ProposerFamily:
            raise TypeError("proposer_family must be ProposerFamily")
        if self.proposer_family is ProposerFamily.GNN:
            if self.channel_id is None or self.representation_mode is None:
                raise Goal7ProtocolError("a GNN arm requires channel and representation identities")
            _nonblank(self.channel_id, field="channel_id")
            _nonblank(self.representation_mode, field="representation_mode")
        elif self.channel_id is not None or self.representation_mode is not None:
            raise Goal7ProtocolError("non-GNN arms cannot claim a graph channel")


@dataclass(frozen=True, slots=True)
class GridCellRequest:
    """One immutable arm/seed cell presented to the integrated training harness."""

    run_id: str
    cell_id: str
    config_digest: str
    arm: GridArmV1
    seed: int
    budget: GridBudgetV1
    step_manifest_sha256: str
    rule_registry_sha256: str
    verifier_sha256: str
    reproduction_command: str

    def __post_init__(self) -> None:
        for field in (
            "run_id",
            "cell_id",
            "config_digest",
            "step_manifest_sha256",
            "rule_registry_sha256",
            "verifier_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if not isinstance(self.arm, GridArmV1):
            raise TypeError("arm must be GridArmV1")
        if self.seed not in PRODUCTION_SEEDS:
            raise Goal7ProtocolError("cell seed is outside the frozen Goal 7 seed set")
        if not isinstance(self.budget, GridBudgetV1):
            raise TypeError("budget must be GridBudgetV1")
        _nonblank(self.reproduction_command, field="reproduction_command")


class RunEnvelopeAdapter(Protocol):
    """Read-only bridge to Workstream 2's authoritative run envelope."""

    def __call__(
        self,
        envelope: object,
        *,
        stage: GridStage,
    ) -> Mapping[str, object]: ...


class StepManifestAuthenticator(Protocol):
    """Injected Workstream-1 parser that authenticates the actual step manifest.

    The provider must hash the referenced bytes, parse the canonical manifest
    schema, and derive the accepted-row count from that authenticated content.
    Returning the caller's expected values without reading the manifest violates
    this boundary.
    """

    def __call__(
        self,
        manifest_reference: str,
        *,
        expected_sha256: str,
        expected_step_count: int,
        expected_training_family_inventory_sha256: str,
        expected_step_population_sha256: str,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class UniformDrawAuditV1:
    """Complete shared-inventory evidence for one uniform-valid proposal."""

    record_id: str
    inventory_status: str
    inventory_action_digests: tuple[str, ...]
    legal_mask: tuple[bool, ...]
    ranked_action_digests: tuple[str, ...]
    schema_version: str = UNIFORM_DRAW_AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != UNIFORM_DRAW_AUDIT_SCHEMA_VERSION:
            raise Goal7ProtocolError("unsupported uniform-draw audit schema")
        _nonblank(self.record_id, field="uniform audit record_id")
        if self.inventory_status not in {
            "ready",
            "no_legal_action",
            "unsupported",
            "invalid",
        }:
            raise Goal7ProtocolError("uniform audit has an unknown inventory status")
        if not isinstance(self.inventory_action_digests, tuple):
            raise TypeError("inventory_action_digests must be a tuple")
        for digest in self.inventory_action_digests:
            _require_sha256(digest, field="uniform audit inventory action digest")
        if len(set(self.inventory_action_digests)) != len(self.inventory_action_digests):
            raise Goal7ProtocolError("uniform audit inventory action digests must be unique")
        if (
            not isinstance(self.legal_mask, tuple)
            or len(self.legal_mask) != len(self.inventory_action_digests)
            or any(not isinstance(value, bool) for value in self.legal_mask)
        ):
            raise TypeError(
                "uniform audit legal_mask must be a bool tuple aligned with the inventory"
            )
        if self.inventory_status != "ready" and any(self.legal_mask):
            raise Goal7ProtocolError(
                "a non-ready uniform audit cannot mark an inventory action legal"
            )
        if not isinstance(self.ranked_action_digests, tuple):
            raise TypeError("ranked_action_digests must be a tuple")
        for digest in self.ranked_action_digests:
            _require_sha256(digest, field="uniform audit ranked action digest")
        legal_digests = tuple(
            digest
            for digest, legal in zip(
                self.inventory_action_digests,
                self.legal_mask,
                strict=True,
            )
            if legal
        )
        if len(self.ranked_action_digests) != len(set(self.ranked_action_digests)) or set(
            self.ranked_action_digests
        ) != set(legal_digests):
            raise Goal7ProtocolError(
                "uniform audit ranking must be a permutation of all and only legal actions"
            )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "inventory_action_digests": list(self.inventory_action_digests),
            "inventory_status": self.inventory_status,
            "legal_mask": list(self.legal_mask),
            "ranked_action_digests": list(self.ranked_action_digests),
            "record_id": self.record_id,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> UniformDrawAuditV1:
        expected = {
            "inventory_action_digests",
            "inventory_status",
            "legal_mask",
            "ranked_action_digests",
            "record_id",
            "schema_version",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise Goal7ProtocolError("uniform-draw audit fields are incompatible")
        action_digests = value["inventory_action_digests"]
        legal_mask = value["legal_mask"]
        ranked_digests = value["ranked_action_digests"]
        if (
            not isinstance(action_digests, list)
            or not isinstance(legal_mask, list)
            or not isinstance(ranked_digests, list)
        ):
            raise Goal7ProtocolError("uniform-draw audit arrays are invalid")
        return cls(
            record_id=value["record_id"],  # type: ignore[arg-type]
            inventory_status=value["inventory_status"],  # type: ignore[arg-type]
            inventory_action_digests=tuple(action_digests),
            legal_mask=tuple(legal_mask),
            ranked_action_digests=tuple(ranked_digests),
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class GridCellExecution:
    """Injected harness output before the runner authenticates and persists it."""

    status: GridCellStatus
    metric_rows: tuple[Mapping[str, object], ...]
    checkpoint_sha256: str | None
    parameter_count: int | None
    estimated_flops: float | None
    wall_time_seconds: float
    peak_host_memory_bytes: int | None
    peak_device_memory_bytes: int | None
    error_type: str | None = None
    error_message: str | None = None
    rejected_metric_rows: tuple[Mapping[str, object], ...] = ()
    uniform_draw_audits: tuple[UniformDrawAuditV1, ...] = ()
    rejected_uniform_draw_audits: tuple[UniformDrawAuditV1, ...] = ()
    run_envelope: Mapping[str, object] | None = None
    budget_consumption: BudgetConsumptionV1 | None = None
    runner_observed_wall_time_seconds: float | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not GridCellStatus:
            raise TypeError("status must be GridCellStatus")
        if type(self.metric_rows) is not tuple or any(
            not isinstance(row, Mapping) for row in self.metric_rows
        ):
            raise TypeError("metric_rows must be a tuple of mappings")
        _validate_metric_rows(self.metric_rows)
        if type(self.rejected_metric_rows) is not tuple or any(
            not isinstance(row, Mapping) for row in self.rejected_metric_rows
        ):
            raise TypeError("rejected_metric_rows must be a tuple of mappings")
        for index, row in enumerate(self.rejected_metric_rows):
            _validate_json(row, label=f"rejected_metric_rows[{index}]")
        for field in ("uniform_draw_audits", "rejected_uniform_draw_audits"):
            audits = getattr(self, field)
            if not isinstance(audits, tuple) or any(
                not isinstance(audit, UniformDrawAuditV1) for audit in audits
            ):
                raise TypeError(f"{field} must be a tuple of UniformDrawAuditV1 values")
            record_ids = tuple(audit.record_id for audit in audits)
            if len(record_ids) != len(set(record_ids)):
                raise Goal7ProtocolError(f"{field} contains duplicate record IDs")
        if self.run_envelope is not None:
            _snapshot_run_envelope(self.run_envelope)
        if self.budget_consumption is not None and not isinstance(
            self.budget_consumption,
            BudgetConsumptionV1,
        ):
            raise TypeError("budget_consumption must be BudgetConsumptionV1 or None")
        if self.checkpoint_sha256 is not None:
            _require_sha256(self.checkpoint_sha256, field="checkpoint_sha256")
        for field in ("parameter_count", "peak_host_memory_bytes", "peak_device_memory_bytes"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value < 0):
                raise TypeError(f"{field} must be a nonnegative exact integer or None")
        if self.estimated_flops is not None and (
            type(self.estimated_flops) is not float
            or not math.isfinite(self.estimated_flops)
            or self.estimated_flops < 0
        ):
            raise TypeError("estimated_flops must be a finite nonnegative float or None")
        if (
            type(self.wall_time_seconds) is not float
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds < 0
        ):
            raise TypeError("wall_time_seconds must be a finite nonnegative float")
        if self.runner_observed_wall_time_seconds is not None and (
            type(self.runner_observed_wall_time_seconds) is not float
            or not math.isfinite(self.runner_observed_wall_time_seconds)
            or self.runner_observed_wall_time_seconds < 0
        ):
            raise TypeError(
                "runner_observed_wall_time_seconds must be a finite nonnegative float or None"
            )
        if self.status is GridCellStatus.COMPLETE:
            if self.error_type is not None or self.error_message is not None:
                raise Goal7ProtocolError("complete cells cannot carry an error")
            if self.rejected_metric_rows:
                raise Goal7ProtocolError("complete cells cannot carry rejected metric rows")
            if self.rejected_uniform_draw_audits:
                raise Goal7ProtocolError("complete cells cannot carry rejected uniform audits")
        else:
            if self.error_type is None or self.error_message is None:
                raise Goal7ProtocolError("non-complete cells require typed error evidence")
        for field in ("error_type", "error_message"):
            value = getattr(self, field)
            if value is not None:
                _nonblank(value, field=field)


class GridCellExecutor(Protocol):
    """Workstream 2 integration boundary; Phase-A tests inject tiny executors."""

    def __call__(self, request: GridCellRequest) -> GridCellExecution: ...


@dataclass(frozen=True, slots=True)
class GridRunReceipt:
    run_id: str
    run_directory: Path
    expected_cell_count: int
    retained_cell_count: int
    resumed_cell_count: int
    status_counts: tuple[tuple[str, int], ...]
    complete: bool
    completion_path: Path | None


@dataclass(frozen=True, slots=True)
class Goal7RunEvidence:
    """Authenticated completion manifest and immutable cell rows."""

    run_directory: Path
    manifest: Mapping[str, JsonValue]
    cells: tuple[Mapping[str, JsonValue], ...]
    missing_cell_ids: tuple[str, ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class _FrozenEvidenceContract:
    """Self-contained scientific contract persisted in a run plan/completion."""

    stage: GridStage
    expected_step_count: int
    selection_split: str
    evaluation_splits: tuple[str, ...]
    budget: GridBudgetV1
    reproduction_command: str
    analysis_reproduction_command: str
    cell_contracts: Mapping[str, Mapping[str, JsonValue]]


def grid_arms(config: Goal7GridConfig) -> tuple[GridArmV1, ...]:
    """Return the exact controlled arm order for a resolved channel contract."""

    if not config.channel_contract_resolved or len(config.enabled_channels) != 4:
        raise Goal7ProtocolError("cannot enumerate the grid before four channels are approved")
    gnn = tuple(
        GridArmV1(
            arm_id=f"gnn:{channel.channel_id}",
            proposer_family=ProposerFamily.GNN,
            channel_id=channel.channel_id,
            representation_mode=channel.representation_mode,
        )
        for channel in config.enabled_channels
    )
    return (
        *gnn,
        GridArmV1(
            arm_id="transformer",
            proposer_family=ProposerFamily.TRANSFORMER,
            channel_id=None,
            representation_mode=None,
        ),
        GridArmV1(
            arm_id="uniform_valid",
            proposer_family=ProposerFamily.UNIFORM_VALID,
            channel_id=None,
            representation_mode=None,
        ),
    )


def enumerate_grid_cells(config: Goal7GridConfig) -> tuple[GridCellRequest, ...]:
    """Enumerate six arms by three seeds with canonical IDs and commands."""

    config.require_runnable()
    if (
        config.step_manifest_sha256 is None
        or config.rule_registry_sha256 is None
        or config.verifier_sha256 is None
        or config.reproduction_command is None
    ):
        raise Goal7ProtocolError(
            "grid enumeration requires step, registry, verifier, and command identities"
        )
    arms = grid_arms(config)
    run_id = _run_id(config, arms)
    requests: list[GridCellRequest] = []
    for arm in arms:
        for seed in config.seeds:
            identity = {
                "run_id": run_id,
                "config_digest": config.digest,
                "arm_id": arm.arm_id,
                "seed": seed,
            }
            cell_id = hashlib.sha256(_CELL_ID_DOMAIN + _canonical_json(identity)).hexdigest()
            command = config.reproduction_command.format(cell_id=cell_id)
            if "{" in command or "}" in command:
                raise Goal7ProtocolError("reproduction command contains unresolved template fields")
            requests.append(
                GridCellRequest(
                    run_id=run_id,
                    cell_id=cell_id,
                    config_digest=config.digest,
                    arm=arm,
                    seed=seed,
                    budget=config.budget,
                    step_manifest_sha256=config.step_manifest_sha256,
                    rule_registry_sha256=config.rule_registry_sha256,
                    verifier_sha256=config.verifier_sha256,
                    reproduction_command=command,
                )
            )
    return tuple(requests)


def uniform_valid_order(
    legal_action_indices: tuple[int, ...],
    *,
    seed: int,
    record_id: str,
) -> tuple[int, ...]:
    """Return a deterministic uniform permutation of the shared legal mask.

    Rejection sampling avoids modulo bias. The helper receives legal indices
    from the common registry-derived inventory; it never creates a second
    legality policy.
    """

    if (
        not isinstance(legal_action_indices, tuple)
        or any(type(index) is not int or index < 0 for index in legal_action_indices)
        or legal_action_indices != tuple(sorted(set(legal_action_indices)))
    ):
        raise TypeError("legal_action_indices must be a sorted unique tuple")
    if type(seed) is not int or seed < 0:
        raise TypeError("seed must be a nonnegative exact integer")
    _nonblank(record_id, field="record_id")
    result = list(legal_action_indices)
    counter = 0
    for upper in range(len(result) - 1, 0, -1):
        selected, counter = _uniform_index(
            upper + 1,
            seed=seed,
            record_id=record_id,
            counter=counter,
        )
        result[upper], result[selected] = result[selected], result[upper]
    return tuple(result)


def compute_step_population_digest(
    records: Sequence[Mapping[str, object]],
) -> str:
    """Hash the exact accepted step identities used by every grid cell.

    A Workstream-1 manifest authenticator derives the same payload from
    accepted ``StepRecordV1`` rows, adapting ``next_signature`` to
    ``target_successor_signature``. Metric-only fields and proposal outcomes
    are intentionally excluded.
    """

    identities: list[dict[str, JsonValue]] = []
    record_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"step population record {index} must be a mapping")
        missing = set(_STEP_POPULATION_FIELDS) - set(record)
        if missing:
            raise Goal7ProtocolError(
                "step population record lacks scientific identity fields: "
                + ", ".join(sorted(missing))
            )
        identity = {field: _json_clone(record[field]) for field in _STEP_POPULATION_FIELDS}
        record_id = _nonblank(identity["record_id"], field="step population record_id")
        record_ids.append(record_id)
        identities.append(identity)
    if len(record_ids) != len(set(record_ids)):
        raise Goal7ProtocolError("step population contains duplicate record IDs")
    identities.sort(key=lambda item: str(item["record_id"]))
    return hashlib.sha256(_STEP_POPULATION_DOMAIN + _canonical_json(identities)).hexdigest()


def _authenticate_step_manifest(
    config: Goal7GridConfig,
    *,
    authenticator: StepManifestAuthenticator | None,
) -> dict[str, JsonValue]:
    if (
        config.step_manifest is None
        or config.step_manifest_sha256 is None
        or config.expected_step_count is None
        or config.training_family_inventory_sha256 is None
        or config.step_population_sha256 is None
    ):
        raise Goal7ProtocolError("step-manifest authentication requires a frozen manifest contract")
    if authenticator is None:
        if config.stage is GridStage.PRODUCTION:
            raise Goal7ProtocolError(
                "production execution requires an injected step-manifest authenticator"
            )
        return {
            "accepted_step_count": config.expected_step_count,
            "manifest_reference": config.step_manifest,
            "manifest_sha256": config.step_manifest_sha256,
            "training_family_inventory_sha256": (config.training_family_inventory_sha256),
            "step_population_sha256": config.step_population_sha256,
            "schema_version": STEP_MANIFEST_AUTH_SCHEMA_VERSION,
            "status": "fixture_unverified",
        }
    snapshot = _json_clone(
        authenticator(
            config.step_manifest,
            expected_sha256=config.step_manifest_sha256,
            expected_step_count=config.expected_step_count,
            expected_training_family_inventory_sha256=(config.training_family_inventory_sha256),
            expected_step_population_sha256=config.step_population_sha256,
        )
    )
    if not isinstance(snapshot, dict):
        raise Goal7ProtocolError("step-manifest authenticator must return a JSON object")
    expected = {
        "accepted_step_count": config.expected_step_count,
        "manifest_reference": config.step_manifest,
        "manifest_sha256": config.step_manifest_sha256,
        "training_family_inventory_sha256": (config.training_family_inventory_sha256),
        "step_population_sha256": config.step_population_sha256,
        "schema_version": STEP_MANIFEST_AUTH_SCHEMA_VERSION,
        "status": "authenticated",
    }
    if snapshot != expected:
        raise Goal7ProtocolError(
            "step-manifest authentication evidence disagrees with the frozen contract"
        )
    return snapshot


def run_goal7_grid(
    config: Goal7GridConfig,
    *,
    executor: GridCellExecutor,
    run_envelope: object,
    envelope_adapter: RunEnvelopeAdapter,
    step_manifest_authenticator: StepManifestAuthenticator | None = None,
    interrupt_after_new_cells: int | None = None,
    requested_cell_ids: tuple[str, ...] | None = None,
) -> GridRunReceipt:
    """Execute or resume every immutable Goal 7 cell.

    A retained failure is a completed scientific attempt and is never replaced
    by a later call with the same run identity.
    """

    if not isinstance(config, Goal7GridConfig):
        raise TypeError("config must be Goal7GridConfig")
    step_manifest_authentication = _authenticate_step_manifest(
        config,
        authenticator=step_manifest_authenticator,
    )
    runtime = _snapshot_run_envelope(
        envelope_adapter(run_envelope, stage=config.stage),
    )
    if interrupt_after_new_cells is not None and (
        type(interrupt_after_new_cells) is not int or interrupt_after_new_cells <= 0
    ):
        raise TypeError("interrupt_after_new_cells must be a positive integer or None")

    requests = enumerate_grid_cells(config)
    run_id = requests[0].run_id
    root = Path(config.output_root) / run_id
    cells_directory = root / "cells"
    cells_directory.mkdir(parents=True, exist_ok=True)
    plan_path = root / "run.plan.json"
    completion_path = root / "run.complete.json"
    plan = _run_plan(
        config=config,
        run_id=run_id,
        requests=requests,
        step_manifest_authentication=step_manifest_authentication,
    )
    if plan_path.is_file():
        if _read_json_object(plan_path) != plan:
            raise Goal7ProtocolError("existing Goal 7 run plan is immutable")
    else:
        _atomic_create_json(plan_path, plan)
    if requested_cell_ids is not None:
        if (
            not isinstance(requested_cell_ids, tuple)
            or not requested_cell_ids
            or requested_cell_ids != tuple(sorted(set(requested_cell_ids)))
        ):
            raise TypeError("requested_cell_ids must be a sorted unique nonempty tuple")
        available_ids = {request.cell_id for request in requests}
        unknown = set(requested_cell_ids) - available_ids
        if unknown:
            raise Goal7ProtocolError("requested_cell_ids contains an unknown Goal 7 cell")
        selected_requests = tuple(
            request for request in requests if request.cell_id in requested_cell_ids
        )
    else:
        selected_requests = requests

    retained: dict[str, dict[str, JsonValue]] = {}
    resumed = 0
    created = 0
    interrupted = False
    population_reference: tuple[tuple[object, ...], ...] | None = None
    for request in selected_requests:
        path = _cell_path(cells_directory, request.cell_id)
        if path.is_file():
            row = _load_cell(path, request=request, config=config)
            population_reference = _check_population_reference(
                row,
                reference=population_reference,
            )
            retained[request.cell_id] = row
            resumed += 1
            continue
        if interrupt_after_new_cells is not None and created >= interrupt_after_new_cells:
            interrupted = True
            break
        execution = _execute_retaining_failure(executor, request)
        cell_run_envelope: dict[str, JsonValue] | None = None
        envelope_source: str | None = None
        try:
            cell_run_envelope, envelope_source = _resolve_cell_run_envelope(
                execution,
                fallback=runtime,
                adapter=envelope_adapter,
                config=config,
                request=request,
            )
            _validate_execution(execution, config=config, request=request)
            next_population = _check_population_reference(
                {
                    "status": execution.status.value,
                    "metric_rows": list(execution.metric_rows),
                },
                reference=population_reference,
            )
        except (Goal7ProtocolError, TypeError, ValueError) as error:
            execution = _invalid_execution(execution, error=error)
            if cell_run_envelope is None or envelope_source is None:
                cell_run_envelope, envelope_source = _rejected_cell_run_envelope(
                    execution,
                    fallback=runtime,
                    stage=config.stage,
                )
        else:
            population_reference = next_population
        assert cell_run_envelope is not None
        assert envelope_source is not None
        row = _cell_payload(
            config=config,
            request=request,
            execution=execution,
            cell_run_envelope=cell_run_envelope,
            run_envelope_source=envelope_source,
        )
        _atomic_create_json(path, row)
        retained[request.cell_id] = row
        created += 1

    for request in requests:
        if request.cell_id in retained:
            continue
        path = _cell_path(cells_directory, request.cell_id)
        if not path.is_file():
            continue
        row = _load_cell(path, request=request, config=config)
        population_reference = _check_population_reference(
            row,
            reference=population_reference,
        )
        retained[request.cell_id] = row
        resumed += 1

    expected_ids = tuple(request.cell_id for request in requests)
    complete = not interrupted and set(retained) == set(expected_ids)
    if complete:
        _validate_cross_cell_population(retained.values(), config=config)
        if completion_path.is_file():
            load_goal7_run_evidence(
                root,
                expected_config_digest=config.digest,
                expected_step_manifest_sha256=config.step_manifest_sha256,
                expected_rule_registry_sha256=config.rule_registry_sha256,
            )
        else:
            completion = _run_completion(
                config=config,
                run_id=run_id,
                requests=requests,
                rows=retained,
                step_manifest_authentication=step_manifest_authentication,
            )
            _atomic_create_json(completion_path, completion)
    elif completion_path.exists():
        raise Goal7ProtocolError("a partial run cannot retain a completion manifest")

    counts = Counter(str(row["status"]) for row in retained.values())
    return GridRunReceipt(
        run_id=run_id,
        run_directory=root,
        expected_cell_count=len(requests),
        retained_cell_count=len(retained),
        resumed_cell_count=resumed,
        status_counts=tuple(sorted(counts.items())),
        complete=complete,
        completion_path=completion_path if complete else None,
    )


def load_goal7_grid_config(path: str | Path) -> Goal7GridConfig:
    """Load the checked-in YAML without converting unresolved nulls into evidence."""

    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as error:
        raise Goal7ProtocolError(f"cannot load Goal 7 config {source}: {error}") from error
    if not isinstance(payload, Mapping):
        raise Goal7ProtocolError("Goal 7 config root must be an object")
    channel_contract = _mapping(payload, "channel_contract")
    raw_channels = channel_contract.get("channels")
    if not isinstance(raw_channels, list):
        raise Goal7ProtocolError("channel_contract.channels must be a list")
    channels = tuple(
        sorted(
            (
                GraphChannelSpec(
                    channel_id=_string(item, "channel_id"),
                    representation_mode=_string(item, "representation_mode"),
                    enabled=_boolean(item, "enabled"),
                    blocker=_optional_string(item.get("blocker"), field="blocker"),
                )
                for item in raw_channels
                if isinstance(item, Mapping)
            ),
            key=lambda item: item.channel_id,
        )
    )
    if len(channels) != len(raw_channels):
        raise Goal7ProtocolError("every channel entry must be an object")
    budget = _mapping(payload, "budget")
    return Goal7GridConfig(
        schema_version=_string(payload, "schema_version"),
        stage=GridStage(_string(payload, "stage")),
        output_root=_string(payload, "output_root"),
        seeds=tuple(_integer_sequence(payload, "seeds")),
        expected_step_count=_optional_integer(
            payload.get("expected_step_count"),
            field="expected_step_count",
            minimum=1,
        ),
        step_manifest=_optional_string(payload.get("step_manifest"), field="step_manifest"),
        step_manifest_sha256=_optional_string(
            payload.get("step_manifest_sha256"),
            field="step_manifest_sha256",
        ),
        rule_registry_sha256=_optional_string(
            payload.get("rule_registry_sha256"),
            field="rule_registry_sha256",
        ),
        verifier_sha256=_optional_string(
            payload.get("verifier_sha256"),
            field="verifier_sha256",
        ),
        shared_harness_sha256=_optional_string(
            payload.get("shared_harness_sha256"),
            field="shared_harness_sha256",
        ),
        shared_gnn_architecture_sha256=_optional_string(
            payload.get("shared_gnn_architecture_sha256"),
            field="shared_gnn_architecture_sha256",
        ),
        transformer_architecture_sha256=_optional_string(
            payload.get("transformer_architecture_sha256"),
            field="transformer_architecture_sha256",
        ),
        compute_reference_sha256=_optional_string(
            payload.get("compute_reference_sha256"),
            field="compute_reference_sha256",
        ),
        implementation_sha256=_optional_string(
            payload.get("implementation_sha256"),
            field="implementation_sha256",
        ),
        training_config_sha256=_optional_string(
            payload.get("training_config_sha256"),
            field="training_config_sha256",
        ),
        training_family_inventory_sha256=_optional_string(
            payload.get("training_family_inventory_sha256"),
            field="training_family_inventory_sha256",
        ),
        step_population_sha256=_optional_string(
            payload.get("step_population_sha256"),
            field="step_population_sha256",
        ),
        analysis_reproduction_command=_optional_string(
            payload.get("analysis_reproduction_command"),
            field="analysis_reproduction_command",
        ),
        channel_contract_resolved=_boolean(channel_contract, "resolved"),
        channels=channels,
        budget=_parse_budget(budget),
        reproduction_command=_optional_string(
            payload.get("reproduction_command"),
            field="reproduction_command",
        ),
        selection_split=_string(payload, "selection_split"),
        evaluation_splits=tuple(_string_sequence(payload, "evaluation_splits")),
    )


def _parse_budget(value: Mapping[str, object]) -> GridBudgetV1:
    return GridBudgetV1(
        maximum_epochs=_integer(value, "maximum_epochs", minimum=1),
        early_stopping_patience=_integer(
            value,
            "early_stopping_patience",
            minimum=1,
        ),
        maximum_optimizer_steps=_integer(
            value,
            "maximum_optimizer_steps",
            minimum=1,
        ),
        node_edge_batch_budget=_integer(
            value,
            "node_edge_batch_budget",
            minimum=1,
        ),
        wall_time_seconds=_number(value, "wall_time_seconds", positive=True),
        top_k=tuple(_integer_sequence(value, "top_k", minimum=1)),
        parameter_match_tolerance_fraction=_number(
            value,
            "parameter_match_tolerance_fraction",
            positive=False,
        ),
        flop_match_tolerance_fraction=_number(
            value,
            "flop_match_tolerance_fraction",
            positive=False,
        ),
        comparison_unit=_string(value, "comparison_unit"),
    )


def _frozen_evidence_contract(
    manifest: Mapping[str, object],
) -> _FrozenEvidenceContract:
    """Validate the self-contained contract shared by a plan and completion."""

    try:
        stage = GridStage(_string(manifest, "stage"))
    except ValueError as error:
        raise Goal7ProtocolError("run manifest has an invalid stage") from error
    if manifest.get("channel_contract_resolved") is not True:
        raise Goal7ProtocolError("persisted Goal 7 evidence requires a resolved channel contract")
    _require_sha256(
        manifest.get("training_config_sha256"),
        field="training_config_sha256",
    )
    training_family_inventory_sha256 = _require_sha256(
        manifest.get("training_family_inventory_sha256"),
        field="training_family_inventory_sha256",
    )
    step_population_sha256 = _require_sha256(
        manifest.get("step_population_sha256"),
        field="step_population_sha256",
    )
    expected_step_count = _integer(manifest, "expected_step_count", minimum=1)
    selection_split = _string(manifest, "selection_split")
    if selection_split != "validation":
        raise Goal7ProtocolError("Goal 7 evidence must use validation-only model selection")
    evaluation_splits = tuple(_string_sequence(manifest, "evaluation_splits"))
    if evaluation_splits != ("test_iid", "test_ood"):
        raise Goal7ProtocolError("Goal 7 evidence has invalid evaluation splits")
    budget = _parse_budget(_mapping(manifest, "budget"))
    if manifest.get("budget_digest") != budget.digest:
        raise Goal7ProtocolError("run manifest budget digest mismatch")
    reproduction_command = _string(manifest, "reproduction_command")
    if "{cell_id}" not in reproduction_command:
        raise Goal7ProtocolError("run manifest command must contain {cell_id}")
    analysis_reproduction_command = _string(
        manifest,
        "analysis_reproduction_command",
    )
    if "{run_id}" not in analysis_reproduction_command:
        raise Goal7ProtocolError("run manifest analysis command must contain {run_id}")
    try:
        rendered_analysis_command = analysis_reproduction_command.format(run_id="0" * 64)
    except (KeyError, ValueError) as error:
        raise Goal7ProtocolError("run manifest analysis command is invalid") from error
    if "{" in rendered_analysis_command or "}" in rendered_analysis_command:
        raise Goal7ProtocolError("run manifest analysis command has unresolved fields")
    manifest_reference = _string(manifest, "step_manifest")
    manifest_sha256 = _require_sha256(
        manifest.get("step_manifest_sha256"),
        field="step_manifest_sha256",
    )
    authentication = manifest.get("step_manifest_authentication")
    allowed_statuses = (
        {"authenticated"}
        if stage is GridStage.PRODUCTION
        else {"authenticated", "fixture_unverified"}
    )
    observed_status = authentication.get("status") if isinstance(authentication, Mapping) else None
    expected_authentication = {
        "accepted_step_count": expected_step_count,
        "manifest_reference": manifest_reference,
        "manifest_sha256": manifest_sha256,
        "training_family_inventory_sha256": training_family_inventory_sha256,
        "step_population_sha256": step_population_sha256,
        "schema_version": STEP_MANIFEST_AUTH_SCHEMA_VERSION,
        "status": observed_status,
    }
    if observed_status not in allowed_statuses or authentication != expected_authentication:
        raise Goal7ProtocolError("run manifest has invalid step-manifest authentication evidence")

    raw_ids = manifest.get("expected_cell_ids")
    raw_arms = manifest.get("arm_ids")
    raw_seeds = manifest.get("seeds")
    raw_contracts = manifest.get("cell_contracts")
    if (
        not isinstance(raw_ids, list)
        or any(type(cell_id) is not str for cell_id in raw_ids)
        or len(raw_ids) != len(set(raw_ids))
        or not isinstance(raw_arms, list)
        or any(type(arm_id) is not str or not arm_id for arm_id in raw_arms)
        or len(raw_arms) != len(set(raw_arms))
        or not isinstance(raw_seeds, list)
        or any(type(seed) is not int for seed in raw_seeds)
        or tuple(raw_seeds) != PRODUCTION_SEEDS
        or not isinstance(raw_contracts, list)
        or any(not isinstance(item, Mapping) for item in raw_contracts)
    ):
        raise Goal7ProtocolError("run manifest has an invalid frozen cell grid")
    if len(raw_arms) != 6 or len(raw_ids) != len(raw_arms) * len(raw_seeds):
        raise Goal7ProtocolError("Goal 7 evidence must contain six arms by three seeds")

    run_id = _require_sha256(manifest.get("run_id"), field="run_id")
    config_digest = _require_sha256(manifest.get("config_digest"), field="config_digest")
    expected_ids = set(raw_ids)
    contracts: dict[str, Mapping[str, JsonValue]] = {}
    arm_scientific_identity: dict[str, tuple[str, object, object]] = {}
    observed_arm_seeds: set[tuple[str, int]] = set()
    exact_fields = {
        "arm_id",
        "cell_id",
        "channel_id",
        "proposer_family",
        "representation_mode",
        "reproduction_command",
        "seed",
    }
    for raw_contract in raw_contracts:
        if set(raw_contract) != exact_fields:
            raise Goal7ProtocolError("cell contract fields are incompatible")
        contract = _json_clone(raw_contract)
        assert isinstance(contract, dict)
        cell_id = _require_sha256(contract.get("cell_id"), field="cell contract cell_id")
        arm_id = _nonblank(contract.get("arm_id"), field="cell contract arm_id")
        seed = contract.get("seed")
        if type(seed) is not int or seed not in raw_seeds:
            raise Goal7ProtocolError("cell contract seed is outside the frozen grid")
        try:
            family = ProposerFamily(
                _nonblank(contract.get("proposer_family"), field="cell proposer_family")
            )
        except ValueError as error:
            raise Goal7ProtocolError("cell contract has an invalid proposer family") from error
        channel_id = contract.get("channel_id")
        representation_mode = contract.get("representation_mode")
        if family is ProposerFamily.GNN:
            _nonblank(channel_id, field="cell channel_id")
            _nonblank(representation_mode, field="cell representation_mode")
        elif channel_id is not None or representation_mode is not None:
            raise Goal7ProtocolError("non-GNN cell contract cannot claim a graph channel")

        expected_id = hashlib.sha256(
            _CELL_ID_DOMAIN
            + _canonical_json(
                {
                    "run_id": run_id,
                    "config_digest": config_digest,
                    "arm_id": arm_id,
                    "seed": seed,
                }
            )
        ).hexdigest()
        if cell_id != expected_id:
            raise Goal7ProtocolError("cell contract ID disagrees with its arm/seed identity")
        try:
            expected_command = reproduction_command.format(cell_id=cell_id)
        except (KeyError, ValueError) as error:
            raise Goal7ProtocolError("run manifest command template is invalid") from error
        if "{" in expected_command or "}" in expected_command:
            raise Goal7ProtocolError("run manifest command contains unresolved template fields")
        if contract.get("reproduction_command") != expected_command:
            raise Goal7ProtocolError("cell contract reproduction command mismatch")
        identity = (family.value, channel_id, representation_mode)
        previous_identity = arm_scientific_identity.setdefault(arm_id, identity)
        if previous_identity != identity:
            raise Goal7ProtocolError("one arm has inconsistent scientific identity across seeds")
        if cell_id in contracts:
            raise Goal7ProtocolError("run manifest repeats a cell contract")
        contracts[cell_id] = contract
        observed_arm_seeds.add((arm_id, seed))

    expected_arm_seeds = {(arm_id, seed) for arm_id in raw_arms for seed in raw_seeds}
    if set(contracts) != expected_ids or observed_arm_seeds != expected_arm_seeds:
        raise Goal7ProtocolError("cell contracts do not form the frozen arm/seed grid")
    family_counts = Counter(identity[0] for identity in arm_scientific_identity.values())
    if family_counts != Counter({"gnn": 4, "transformer": 1, "uniform_valid": 1}):
        raise Goal7ProtocolError("Goal 7 requires four GNN, one transformer, and one uniform arm")
    gnn_channels: set[str] = set()
    for arm_id, (family, channel_id, _) in arm_scientific_identity.items():
        if family == ProposerFamily.GNN.value:
            assert isinstance(channel_id, str)
            if arm_id != f"gnn:{channel_id}":
                raise Goal7ProtocolError("GNN arm ID must be derived from its channel ID")
            gnn_channels.add(channel_id)
        elif arm_id != family:
            raise Goal7ProtocolError("non-GNN arm ID must equal its proposer family")
    if len(gnn_channels) != 4:
        raise Goal7ProtocolError("Goal 7 requires four distinct graph channels")
    return _FrozenEvidenceContract(
        stage=stage,
        expected_step_count=expected_step_count,
        selection_split=selection_split,
        evaluation_splits=evaluation_splits,
        budget=budget,
        reproduction_command=reproduction_command,
        analysis_reproduction_command=analysis_reproduction_command,
        cell_contracts=contracts,
    )


def load_goal7_run_evidence(
    run_directory: str | Path,
    *,
    expected_config_digest: str | None = None,
    expected_step_manifest_sha256: str | None = None,
    expected_rule_registry_sha256: str | None = None,
    allow_incomplete: bool = False,
) -> Goal7RunEvidence:
    """Authenticate a completed run without reinterpreting metric rows."""

    root = Path(run_directory)
    completion_path = root / "run.complete.json"
    complete = completion_path.is_file()
    if complete:
        manifest = _read_json_object(completion_path)
        _validate_content_digest(
            manifest,
            domain=_RUN_CONTENT_DOMAIN,
            label="Goal 7 completion manifest",
        )
        if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
            raise Goal7ProtocolError("unexpected Goal 7 completion schema")
    elif allow_incomplete:
        manifest = _read_json_object(root / "run.plan.json")
        _validate_content_digest(
            manifest,
            domain=_PLAN_CONTENT_DOMAIN,
            label="Goal 7 run plan",
        )
        if manifest.get("schema_version") != PLAN_SCHEMA_VERSION:
            raise Goal7ProtocolError("unexpected Goal 7 run-plan schema")
    else:
        raise Goal7ProtocolError("Goal 7 run has no completion manifest")
    expectations = {
        "config_digest": expected_config_digest,
        "step_manifest_sha256": expected_step_manifest_sha256,
        "rule_registry_sha256": expected_rule_registry_sha256,
    }
    for field, expected in expectations.items():
        if expected is not None and manifest.get(field) != expected:
            raise Goal7ProtocolError(f"Goal 7 {field} does not match the requested evidence")
    frozen_contract = _frozen_evidence_contract(manifest)

    expected_ids = manifest.get("expected_cell_ids")
    content_digests = manifest.get("cell_content_digests") if complete else None
    envelope_digests = manifest.get("cell_run_envelope_digests") if complete else None
    if (
        not isinstance(expected_ids, list)
        or any(not isinstance(cell_id, str) for cell_id in expected_ids)
        or len(expected_ids) != len(set(expected_ids))
        or (complete and not isinstance(content_digests, Mapping))
        or (complete and not isinstance(envelope_digests, Mapping))
    ):
        raise Goal7ProtocolError("completion manifest has invalid cell identities")
    if _integer(manifest, "expected_cell_count", minimum=1) != len(expected_ids):
        raise Goal7ProtocolError("completion manifest cell denominator mismatch")
    if complete and _integer(manifest, "retained_cell_count", minimum=1) != len(expected_ids):
        raise Goal7ProtocolError("completion manifest is incomplete")

    cells: list[Mapping[str, JsonValue]] = []
    missing_cell_ids: list[str] = []
    observed_statuses: Counter[str] = Counter()
    observed_arm_seeds: set[tuple[str, int]] = set()
    for cell_id in expected_ids:
        _require_sha256(cell_id, field="cell_id")
        cell_path = _cell_path(root / "cells", cell_id)
        if not cell_path.is_file():
            missing_cell_ids.append(cell_id)
            continue
        cell = _read_json_object(cell_path)
        _validate_content_digest(
            cell,
            domain=_CELL_CONTENT_DOMAIN,
            label=f"Goal 7 cell {cell_id}",
        )
        if cell.get("cell_id") != cell_id or cell.get("run_id") != manifest.get("run_id"):
            raise Goal7ProtocolError(f"Goal 7 cell identity mismatch: {cell_id}")
        if cell.get("config_digest") != manifest.get("config_digest"):
            raise Goal7ProtocolError(f"Goal 7 cell config mismatch: {cell_id}")
        if cell.get("schema_version") != CELL_SCHEMA_VERSION:
            raise Goal7ProtocolError(f"Goal 7 cell schema mismatch: {cell_id}")
        expected_cell = frozen_contract.cell_contracts[cell_id]
        for field, expected in expected_cell.items():
            if cell.get(field) != expected:
                raise Goal7ProtocolError(f"Goal 7 cell {field} mismatch: {cell_id}")
        if cell.get("selection_split") != frozen_contract.selection_split:
            raise Goal7ProtocolError(f"Goal 7 cell selection split mismatch: {cell_id}")
        if cell.get("evaluation_splits") != list(frozen_contract.evaluation_splits):
            raise Goal7ProtocolError(f"Goal 7 cell evaluation splits mismatch: {cell_id}")
        if cell.get("budget") != frozen_contract.budget.as_dict():
            raise Goal7ProtocolError(f"Goal 7 cell budget mismatch: {cell_id}")
        for field in (
            "step_manifest_sha256",
            "rule_registry_sha256",
            "verifier_sha256",
            "shared_harness_sha256",
            "shared_gnn_architecture_sha256",
            "transformer_architecture_sha256",
            "compute_reference_sha256",
            "implementation_sha256",
            "training_config_sha256",
            "training_family_inventory_sha256",
            "step_population_sha256",
            "budget_digest",
        ):
            if cell.get(field) != manifest.get(field):
                raise Goal7ProtocolError(f"Goal 7 cell {field} mismatch: {cell_id}")
        if complete and content_digests.get(cell_id) != cell.get("content_digest"):
            raise Goal7ProtocolError(f"Goal 7 cell digest ledger mismatch: {cell_id}")
        run_envelope = cell.get("run_envelope")
        _snapshot_run_envelope(run_envelope)
        if complete and envelope_digests.get(cell_id) != _sha256_json(run_envelope):
            raise Goal7ProtocolError(f"Goal 7 cell run-envelope mismatch: {cell_id}")
        _validate_persisted_outcome_contract(
            cell,
            stage=frozen_contract.stage,
            expected_step_count=frozen_contract.expected_step_count,
            rule_registry_sha256=_require_sha256(
                manifest.get("rule_registry_sha256"),
                field="rule_registry_sha256",
            ),
            step_manifest_sha256=_require_sha256(
                manifest.get("step_manifest_sha256"),
                field="step_manifest_sha256",
            ),
            evaluation_splits=frozen_contract.evaluation_splits,
            budget=frozen_contract.budget,
            training_family_inventory_sha256=_require_sha256(
                manifest.get("training_family_inventory_sha256"),
                field="training_family_inventory_sha256",
            ),
            step_population_sha256=_require_sha256(
                manifest.get("step_population_sha256"),
                field="step_population_sha256",
            ),
        )
        status = _string(cell, "status")
        try:
            GridCellStatus(status)
        except ValueError as error:
            raise Goal7ProtocolError(f"Goal 7 cell status is invalid: {cell_id}") from error
        observed_statuses[status] += 1
        observed_arm_seeds.add(
            (
                _string(cell, "arm_id"),
                _integer(cell, "seed", minimum=0),
            )
        )
        cells.append(cell)
    if complete and set(content_digests) != set(expected_ids):
        raise Goal7ProtocolError("completion manifest contains unexpected cell digests")
    if complete and set(envelope_digests) != set(expected_ids):
        raise Goal7ProtocolError("completion manifest contains unexpected envelope digests")
    raw_arms = manifest.get("arm_ids")
    raw_seeds = manifest.get("seeds")
    if (
        not isinstance(raw_arms, list)
        or not isinstance(raw_seeds, list)
        or any(not isinstance(arm, str) or not arm for arm in raw_arms)
        or any(type(seed) is not int for seed in raw_seeds)
        or len(raw_arms) != len(set(raw_arms))
        or tuple(raw_seeds) != PRODUCTION_SEEDS
    ):
        raise Goal7ProtocolError("completion manifest has an invalid arm/seed grid")
    expected_arm_seeds = {(arm, seed) for arm in raw_arms for seed in raw_seeds}
    if not observed_arm_seeds <= expected_arm_seeds:
        raise Goal7ProtocolError("run contains an unexpected arm/seed cell")
    if complete and (
        observed_arm_seeds != expected_arm_seeds or len(cells) != len(expected_arm_seeds)
    ):
        raise Goal7ProtocolError("completion cells do not form the exact arm/seed grid")
    if complete and manifest.get("status_counts") != dict(sorted(observed_statuses.items())):
        raise Goal7ProtocolError("completion status ledger disagrees with cell evidence")
    _validate_complete_populations(cells)
    return Goal7RunEvidence(
        run_directory=root,
        manifest=manifest,
        cells=tuple(cells),
        missing_cell_ids=tuple(missing_cell_ids),
        complete=complete,
    )


def current_fixture_run_envelope(*, exact_command: str) -> dict[str, JsonValue]:
    """Return non-scientific runtime evidence for a tiny local fixture."""

    return {
        "schema_version": "geml-fixture-run-envelope-v1",
        "git_commit": "fixture",
        "python_version": platform.python_version(),
        "package_versions": [],
        "hardware": f"fixture:{platform.machine() or 'unknown'}",
        "precision": "fixture:float64",
        "exact_command": _nonblank(exact_command, field="exact_command"),
        "determinism": "fixture_only",
    }


def fixture_run_envelope_adapter(
    envelope: object,
    *,
    stage: GridStage,
) -> Mapping[str, object]:
    """Fixture-only compatibility adapter pending Workstream 2 integration."""

    if stage is not GridStage.FIXTURE:
        raise Goal7ProtocolError(
            "the fixture run-envelope adapter cannot authenticate production evidence"
        )
    if not isinstance(envelope, Mapping):
        raise TypeError("fixture run envelope must be a mapping")
    return envelope


def _execute_retaining_failure(
    executor: GridCellExecutor,
    request: GridCellRequest,
) -> GridCellExecution:
    started = time.perf_counter()
    try:
        execution = executor(request)
        if not isinstance(execution, GridCellExecution):
            raise TypeError("cell executor must return GridCellExecution")
        return replace(
            execution,
            runner_observed_wall_time_seconds=time.perf_counter() - started,
        )
    except TimeoutError as error:
        elapsed = time.perf_counter() - started
        return GridCellExecution(
            status=GridCellStatus.TIMEOUT,
            metric_rows=(),
            checkpoint_sha256=None,
            parameter_count=None,
            estimated_flops=None,
            wall_time_seconds=elapsed,
            peak_host_memory_bytes=None,
            peak_device_memory_bytes=None,
            error_type=type(error).__name__,
            error_message=_bounded_message(error),
            runner_observed_wall_time_seconds=elapsed,
        )
    except Exception as error:
        elapsed = time.perf_counter() - started
        return GridCellExecution(
            status=GridCellStatus.FAILED,
            metric_rows=(),
            checkpoint_sha256=None,
            parameter_count=None,
            estimated_flops=None,
            wall_time_seconds=elapsed,
            peak_host_memory_bytes=None,
            peak_device_memory_bytes=None,
            error_type=type(error).__name__,
            error_message=_bounded_message(error),
            runner_observed_wall_time_seconds=elapsed,
        )


def _validate_execution(
    execution: GridCellExecution,
    *,
    config: Goal7GridConfig,
    request: GridCellRequest,
) -> None:
    if execution.status is GridCellStatus.COMPLETE:
        if execution.budget_consumption is None:
            raise Goal7ProtocolError("complete cell lacks observed training-budget consumption")
        _validate_budget_consumption(
            execution.budget_consumption,
            budget=request.budget,
            proposer_family=request.arm.proposer_family,
        )
        if execution.runner_observed_wall_time_seconds is None:
            raise Goal7ProtocolError("complete cell lacks runner-observed wall-time evidence")
        if execution.runner_observed_wall_time_seconds > request.budget.wall_time_seconds:
            raise Goal7ProtocolError(
                "complete cell exceeded the frozen runner-observed wall-time budget"
            )
        if execution.wall_time_seconds > request.budget.wall_time_seconds:
            raise Goal7ProtocolError(
                "complete cell exceeded the frozen wall-time budget: "
                f"{execution.wall_time_seconds} > {request.budget.wall_time_seconds}"
            )
        if config.expected_step_count is None:
            raise Goal7ProtocolError("complete cells require a frozen step denominator")
        if len(execution.metric_rows) != config.expected_step_count:
            raise Goal7ProtocolError("complete cell does not cover the frozen step denominator")
        if (
            config.step_population_sha256 is None
            or compute_step_population_digest(execution.metric_rows)
            != config.step_population_sha256
        ):
            raise Goal7ProtocolError(
                "complete cell does not match the authenticated step population"
            )
        if request.arm.proposer_family is ProposerFamily.UNIFORM_VALID:
            if (
                execution.checkpoint_sha256 is not None
                or execution.parameter_count != 0
                or execution.estimated_flops != 0.0
            ):
                raise Goal7ProtocolError(
                    "uniform-valid cells require no checkpoint and zero model parameters/FLOPs"
                )
        elif (
            execution.checkpoint_sha256 is None
            or execution.parameter_count is None
            or execution.parameter_count <= 0
            or execution.estimated_flops is None
            or execution.estimated_flops <= 0
        ):
            raise Goal7ProtocolError(
                "complete learned cells require checkpoint and positive parameter/FLOP telemetry"
            )
        if config.stage is GridStage.PRODUCTION and execution.run_envelope is None:
            raise Goal7ProtocolError(
                "production cells require their post-execution authoritative run envelope"
            )
    for row in execution.metric_rows:
        _validate_metric_contract(
            row,
            config=config,
            require_legal_mask=execution.status is GridCellStatus.COMPLETE,
        )
    if request.arm.proposer_family is ProposerFamily.UNIFORM_VALID:
        if execution.status is GridCellStatus.COMPLETE:
            _validate_uniform_draw_audits(
                execution.uniform_draw_audits,
                metric_rows=execution.metric_rows,
                seed=request.seed,
            )
        elif execution.uniform_draw_audits:
            raise Goal7ProtocolError(
                "non-complete uniform cells cannot carry accepted uniform audits"
            )
    elif execution.uniform_draw_audits:
        raise Goal7ProtocolError("learned cells cannot carry uniform-draw audits")


def _validate_budget_consumption(
    consumption: BudgetConsumptionV1,
    *,
    budget: GridBudgetV1,
    proposer_family: ProposerFamily,
) -> None:
    if consumption.epochs_completed > budget.maximum_epochs:
        raise Goal7ProtocolError("cell exceeded the frozen epoch budget")
    if consumption.optimizer_steps_completed > budget.maximum_optimizer_steps:
        raise Goal7ProtocolError("cell exceeded the frozen optimizer-step budget")
    if consumption.maximum_observed_node_edge_batch > budget.node_edge_batch_budget:
        raise Goal7ProtocolError("cell exceeded the frozen node/edge batch budget")
    if proposer_family is ProposerFamily.UNIFORM_VALID:
        if consumption != BudgetConsumptionV1(
            epochs_completed=0,
            optimizer_steps_completed=0,
            maximum_observed_node_edge_batch=0,
            early_stopping_bad_epochs=0,
            stop_reason=BudgetStopReason.NOT_APPLICABLE,
        ):
            raise Goal7ProtocolError("uniform-valid cells cannot claim training-budget consumption")
        return
    if (
        consumption.epochs_completed == 0
        or consumption.optimizer_steps_completed == 0
        or consumption.maximum_observed_node_edge_batch == 0
        or consumption.stop_reason is BudgetStopReason.NOT_APPLICABLE
    ):
        raise Goal7ProtocolError(
            "complete learned cells require positive observed training exposure"
        )
    if (
        consumption.stop_reason is BudgetStopReason.EARLY_STOPPING
        and consumption.early_stopping_bad_epochs < budget.early_stopping_patience
    ):
        raise Goal7ProtocolError("early-stopped cell did not reach the frozen patience")
    if (
        consumption.stop_reason is BudgetStopReason.EPOCH_LIMIT
        and consumption.epochs_completed != budget.maximum_epochs
    ):
        raise Goal7ProtocolError("epoch-limit stop reason disagrees with observed epochs")
    if (
        consumption.stop_reason is BudgetStopReason.OPTIMIZER_STEP_LIMIT
        and consumption.optimizer_steps_completed != budget.maximum_optimizer_steps
    ):
        raise Goal7ProtocolError("optimizer-step-limit reason disagrees with observed steps")


def _validate_uniform_draw_audits(
    audits: Sequence[UniformDrawAuditV1],
    *,
    metric_rows: Sequence[Mapping[str, object]],
    seed: int,
) -> None:
    """Authenticate uniform rankings against the exact shared inventory."""

    from geml.learning.eval.step_metrics import StepMetricOutcomeV1
    from geml.learning.policy.head import (
        ActionInventoryStatus,
        compute_legal_mask_digest,
    )

    if len(audits) != len(metric_rows):
        raise Goal7ProtocolError(
            "complete uniform cells require exactly one audit per metric record"
        )
    audits_by_record = {audit.record_id: audit for audit in audits}
    if len(audits_by_record) != len(audits):
        raise Goal7ProtocolError("uniform-draw audits contain duplicate record IDs")
    for raw_row in metric_rows:
        try:
            outcome = StepMetricOutcomeV1.from_dict(dict(raw_row))
        except (KeyError, TypeError, ValueError) as error:
            raise Goal7ProtocolError(f"invalid typed step-metric row: {error}") from error
        try:
            audit = audits_by_record.pop(outcome.record_id)
        except KeyError as error:
            raise Goal7ProtocolError(
                f"uniform audit is missing metric record {outcome.record_id}"
            ) from error
        try:
            inventory_status = ActionInventoryStatus(audit.inventory_status)
        except ValueError as error:  # pragma: no cover - the audit constructor rejects this.
            raise Goal7ProtocolError("uniform audit has an unknown inventory status") from error
        computed_mask_digest = compute_legal_mask_digest(
            action_digests=audit.inventory_action_digests,
            legal_mask=audit.legal_mask,
            current_signature=outcome.current_signature,
            goal_signature=outcome.goal_signature,
            registry_digest=outcome.rule_registry_digest,
            status=inventory_status,
        )
        if computed_mask_digest != outcome.legal_mask_digest:
            raise Goal7ProtocolError(
                f"uniform audit legal-mask digest mismatch for {outcome.record_id}"
            )
        legal_indices = tuple(index for index, legal in enumerate(audit.legal_mask) if legal)
        if len(legal_indices) != outcome.legal_action_count:
            raise Goal7ProtocolError(
                f"uniform audit legal-action count mismatch for {outcome.record_id}"
            )
        order = uniform_valid_order(
            legal_indices,
            seed=seed,
            record_id=outcome.record_id,
        )
        expected_ranking = tuple(audit.inventory_action_digests[index] for index in order)
        if audit.ranked_action_digests != expected_ranking:
            raise Goal7ProtocolError(
                f"uniform audit ranking is not the deterministic draw for {outcome.record_id}"
            )
        candidate_digests: list[str] = []
        for candidate in outcome.candidates:
            if candidate.action is None:
                raise Goal7ProtocolError(
                    f"uniform metric candidate lacks action identity for {outcome.record_id}"
                )
            candidate_digests.append(candidate.action.action_digest)
        if tuple(candidate_digests) != expected_ranking[: len(candidate_digests)]:
            raise Goal7ProtocolError(
                f"uniform metric ranking disagrees with its draw audit for {outcome.record_id}"
            )
    if audits_by_record:  # pragma: no cover - equal counts plus all row hits imply empty.
        raise Goal7ProtocolError("uniform audits contain records absent from metric evidence")


def _resolve_cell_run_envelope(
    execution: GridCellExecution,
    *,
    fallback: Mapping[str, JsonValue],
    adapter: RunEnvelopeAdapter,
    config: Goal7GridConfig,
    request: GridCellRequest,
) -> tuple[dict[str, JsonValue], str]:
    if execution.run_envelope is None:
        source = (
            "fixture_fallback"
            if config.stage is GridStage.FIXTURE
            else "pre_run_fallback_unverified"
        )
        return dict(fallback), source
    snapshot = _snapshot_run_envelope(
        adapter(execution.run_envelope, stage=config.stage),
    )
    if config.stage is GridStage.PRODUCTION:
        expected: dict[str, object] = {
            "exact_command": request.reproduction_command,
            "config_digest": request.config_digest,
            "step_manifest_sha256": request.step_manifest_sha256,
            "rule_registry_sha256": request.rule_registry_sha256,
            "verifier_sha256": request.verifier_sha256,
            "training_config_sha256": config.training_config_sha256,
            "training_family_inventory_sha256": (config.training_family_inventory_sha256),
            "step_population_sha256": config.step_population_sha256,
            "budget_consumption": (
                None
                if execution.budget_consumption is None
                else execution.budget_consumption.as_dict()
            ),
            "wall_time_seconds": execution.wall_time_seconds,
            "seed": request.seed,
            "cell_id": request.cell_id,
        }
        mismatched = [field for field, value in expected.items() if snapshot.get(field) != value]
        if mismatched:
            raise Goal7ProtocolError(
                "production cell run envelope has mismatched bindings: " + ", ".join(mismatched)
            )
    return snapshot, "authoritative"


def _rejected_cell_run_envelope(
    execution: GridCellExecution,
    *,
    fallback: Mapping[str, JsonValue],
    stage: GridStage,
) -> tuple[dict[str, JsonValue], str]:
    if execution.run_envelope is not None:
        return _snapshot_run_envelope(execution.run_envelope), "rejected_unverified"
    return (
        dict(fallback),
        "fixture_fallback" if stage is GridStage.FIXTURE else "pre_run_fallback_unverified",
    )


def _invalid_execution(
    execution: GridCellExecution,
    *,
    error: BaseException,
) -> GridCellExecution:
    rejected = (*execution.rejected_metric_rows, *execution.metric_rows)
    rejected_audits = (
        *execution.rejected_uniform_draw_audits,
        *execution.uniform_draw_audits,
    )
    message = _bounded_message(error)
    if execution.error_type is not None and execution.error_message is not None:
        message = _bounded_message(
            RuntimeError(f"{message}; prior {execution.error_type}: {execution.error_message}")
        )
    return GridCellExecution(
        status=GridCellStatus.INVALID,
        metric_rows=(),
        checkpoint_sha256=execution.checkpoint_sha256,
        parameter_count=execution.parameter_count,
        estimated_flops=execution.estimated_flops,
        wall_time_seconds=execution.wall_time_seconds,
        peak_host_memory_bytes=execution.peak_host_memory_bytes,
        peak_device_memory_bytes=execution.peak_device_memory_bytes,
        error_type=type(error).__name__,
        error_message=message,
        rejected_metric_rows=rejected,
        rejected_uniform_draw_audits=rejected_audits,
        run_envelope=execution.run_envelope,
        budget_consumption=execution.budget_consumption,
        runner_observed_wall_time_seconds=(execution.runner_observed_wall_time_seconds),
    )


def _cell_payload(
    *,
    config: Goal7GridConfig,
    request: GridCellRequest,
    execution: GridCellExecution,
    cell_run_envelope: Mapping[str, JsonValue],
    run_envelope_source: str,
) -> dict[str, JsonValue]:
    metric_rows = sorted(
        (_json_clone(row) for row in execution.metric_rows),
        key=lambda row: str(row["record_id"]),
    )
    rejected_metric_rows = sorted(
        (_json_clone(row) for row in execution.rejected_metric_rows),
        key=lambda row: str(row.get("record_id", "")),
    )
    uniform_draw_audits = sorted(
        (audit.as_dict() for audit in execution.uniform_draw_audits),
        key=lambda audit: str(audit["record_id"]),
    )
    rejected_uniform_draw_audits = sorted(
        (audit.as_dict() for audit in execution.rejected_uniform_draw_audits),
        key=lambda audit: str(audit["record_id"]),
    )
    run_envelope = _snapshot_run_envelope(cell_run_envelope)
    payload: dict[str, JsonValue] = {
        "schema_version": CELL_SCHEMA_VERSION,
        "run_id": request.run_id,
        "cell_id": request.cell_id,
        "config_digest": request.config_digest,
        "arm_id": request.arm.arm_id,
        "proposer_family": request.arm.proposer_family.value,
        "channel_id": request.arm.channel_id,
        "representation_mode": request.arm.representation_mode,
        "seed": request.seed,
        "selection_split": config.selection_split,
        "evaluation_splits": list(config.evaluation_splits),
        "step_manifest_sha256": request.step_manifest_sha256,
        "rule_registry_sha256": request.rule_registry_sha256,
        "verifier_sha256": request.verifier_sha256,
        "shared_harness_sha256": config.shared_harness_sha256,
        "shared_gnn_architecture_sha256": config.shared_gnn_architecture_sha256,
        "transformer_architecture_sha256": config.transformer_architecture_sha256,
        "compute_reference_sha256": config.compute_reference_sha256,
        "implementation_sha256": config.implementation_sha256,
        "training_config_sha256": config.training_config_sha256,
        "training_family_inventory_sha256": (config.training_family_inventory_sha256),
        "step_population_sha256": config.step_population_sha256,
        "budget": request.budget.as_dict(),
        "budget_digest": request.budget.digest,
        "budget_consumption": (
            None if execution.budget_consumption is None else execution.budget_consumption.as_dict()
        ),
        "status": execution.status.value,
        "metric_rows": metric_rows,
        "metric_row_count": len(metric_rows),
        "metric_rows_digest": _sha256_json(metric_rows),
        "rejected_metric_rows": rejected_metric_rows,
        "rejected_metric_row_count": len(rejected_metric_rows),
        "rejected_metric_rows_digest": _sha256_json(rejected_metric_rows),
        "uniform_draw_audits": uniform_draw_audits,
        "uniform_draw_audit_count": len(uniform_draw_audits),
        "uniform_draw_audits_digest": _sha256_json(uniform_draw_audits),
        "rejected_uniform_draw_audits": rejected_uniform_draw_audits,
        "rejected_uniform_draw_audit_count": len(rejected_uniform_draw_audits),
        "rejected_uniform_draw_audits_digest": _sha256_json(rejected_uniform_draw_audits),
        "checkpoint_sha256": execution.checkpoint_sha256,
        "parameter_count": execution.parameter_count,
        "estimated_flops": execution.estimated_flops,
        "wall_time_seconds": execution.wall_time_seconds,
        "runner_observed_wall_time_seconds": (execution.runner_observed_wall_time_seconds),
        "peak_host_memory_bytes": execution.peak_host_memory_bytes,
        "peak_device_memory_bytes": execution.peak_device_memory_bytes,
        "error_type": execution.error_type,
        "error_message": execution.error_message,
        "run_envelope": run_envelope,
        "run_envelope_source": run_envelope_source,
        "reproduction_command": request.reproduction_command,
    }
    payload["content_digest"] = hashlib.sha256(
        _CELL_CONTENT_DOMAIN + _canonical_json(payload)
    ).hexdigest()
    return payload


def _load_cell(
    path: Path,
    *,
    request: GridCellRequest,
    config: Goal7GridConfig,
) -> dict[str, JsonValue]:
    row = _read_json_object(path)
    digest = row.get("content_digest")
    content = {key: value for key, value in row.items() if key != "content_digest"}
    if digest != hashlib.sha256(_CELL_CONTENT_DOMAIN + _canonical_json(content)).hexdigest():
        raise Goal7ProtocolError(f"Goal 7 cell content digest mismatch: {path}")
    expected = {
        "schema_version": CELL_SCHEMA_VERSION,
        "run_id": request.run_id,
        "cell_id": request.cell_id,
        "config_digest": request.config_digest,
        "arm_id": request.arm.arm_id,
        "proposer_family": request.arm.proposer_family.value,
        "channel_id": request.arm.channel_id,
        "representation_mode": request.arm.representation_mode,
        "seed": request.seed,
        "selection_split": config.selection_split,
        "evaluation_splits": list(config.evaluation_splits),
        "step_manifest_sha256": request.step_manifest_sha256,
        "rule_registry_sha256": request.rule_registry_sha256,
        "verifier_sha256": request.verifier_sha256,
        "shared_harness_sha256": config.shared_harness_sha256,
        "shared_gnn_architecture_sha256": config.shared_gnn_architecture_sha256,
        "transformer_architecture_sha256": config.transformer_architecture_sha256,
        "compute_reference_sha256": config.compute_reference_sha256,
        "implementation_sha256": config.implementation_sha256,
        "training_config_sha256": config.training_config_sha256,
        "training_family_inventory_sha256": (config.training_family_inventory_sha256),
        "step_population_sha256": config.step_population_sha256,
        "budget": request.budget.as_dict(),
        "budget_digest": request.budget.digest,
        "reproduction_command": request.reproduction_command,
    }
    if any(row.get(field) != value for field, value in expected.items()):
        raise Goal7ProtocolError(f"Goal 7 cell identity mismatch: {path}")
    _validate_persisted_cell_outcome(row, config=config)
    return row


def _validate_persisted_cell_outcome(
    row: Mapping[str, object],
    *,
    config: Goal7GridConfig,
) -> None:
    if (
        config.expected_step_count is None
        or config.rule_registry_sha256 is None
        or config.step_manifest_sha256 is None
        or config.training_family_inventory_sha256 is None
        or config.step_population_sha256 is None
    ):
        raise Goal7ProtocolError("persisted cell validation requires a frozen production contract")
    _validate_persisted_outcome_contract(
        row,
        stage=config.stage,
        expected_step_count=config.expected_step_count,
        rule_registry_sha256=config.rule_registry_sha256,
        step_manifest_sha256=config.step_manifest_sha256,
        evaluation_splits=config.evaluation_splits,
        budget=config.budget,
        training_family_inventory_sha256=(config.training_family_inventory_sha256),
        step_population_sha256=config.step_population_sha256,
    )


def _validate_persisted_outcome_contract(
    row: Mapping[str, object],
    *,
    stage: GridStage,
    expected_step_count: int,
    rule_registry_sha256: str,
    step_manifest_sha256: str,
    evaluation_splits: tuple[str, ...],
    budget: GridBudgetV1,
    training_family_inventory_sha256: str,
    step_population_sha256: str,
) -> None:
    try:
        status = GridCellStatus(_string(row, "status"))
    except ValueError as error:
        raise Goal7ProtocolError("unknown Goal 7 cell status") from error
    raw_rows = row.get("metric_rows")
    if not isinstance(raw_rows, list):
        raise Goal7ProtocolError("metric_rows must be a list")
    if any(not isinstance(item, Mapping) for item in raw_rows):
        raise Goal7ProtocolError("metric_rows must contain objects")
    _validate_metric_rows(tuple(raw_rows))
    if _integer(row, "metric_row_count", minimum=0) != len(raw_rows):
        raise Goal7ProtocolError("metric_row_count disagrees with metric_rows")
    if _sha256_json(raw_rows) != _string(row, "metric_rows_digest"):
        raise Goal7ProtocolError("metric_rows_digest mismatch")
    rejected_rows = row.get("rejected_metric_rows")
    if not isinstance(rejected_rows, list) or any(
        not isinstance(item, Mapping) for item in rejected_rows
    ):
        raise Goal7ProtocolError("rejected_metric_rows must contain objects")
    for index, item in enumerate(rejected_rows):
        _validate_json(item, label=f"rejected_metric_rows[{index}]")
    if _integer(row, "rejected_metric_row_count", minimum=0) != len(rejected_rows):
        raise Goal7ProtocolError("rejected_metric_row_count disagrees with retained evidence")
    if _sha256_json(rejected_rows) != _string(row, "rejected_metric_rows_digest"):
        raise Goal7ProtocolError("rejected_metric_rows_digest mismatch")
    uniform_audits = _persisted_uniform_audits(row, rejected=False)
    rejected_uniform_audits = _persisted_uniform_audits(row, rejected=True)
    if status is GridCellStatus.COMPLETE:
        if len(raw_rows) != expected_step_count:
            raise Goal7ProtocolError("complete cell does not cover the frozen step denominator")
        if compute_step_population_digest(raw_rows) != step_population_sha256:
            raise Goal7ProtocolError(
                "complete cell does not match the authenticated step population"
            )
        if rejected_rows:
            raise Goal7ProtocolError("complete cell cannot retain rejected metric rows")
        if rejected_uniform_audits:
            raise Goal7ProtocolError("complete cell cannot retain rejected uniform audits")
        if row.get("error_type") is not None or row.get("error_message") is not None:
            raise Goal7ProtocolError("complete cell cannot retain an error")
    elif row.get("error_type") is None or row.get("error_message") is None:
        raise Goal7ProtocolError("non-complete cell requires error evidence")
    checkpoint = _optional_digest(row.get("checkpoint_sha256"), field="checkpoint_sha256")
    parameter_count = _optional_integer(
        row.get("parameter_count"),
        field="parameter_count",
        minimum=0,
    )
    estimated_flops = _optional_number(
        row.get("estimated_flops"),
        field="estimated_flops",
        minimum=0.0,
    )
    for field in ("peak_host_memory_bytes", "peak_device_memory_bytes"):
        _optional_integer(row.get(field), field=field, minimum=0)
    wall_time = _number(row, "wall_time_seconds", positive=False)
    runner_observed_wall_time = _optional_number(
        row.get("runner_observed_wall_time_seconds"),
        field="runner_observed_wall_time_seconds",
        minimum=0.0,
    )
    try:
        proposer_family = ProposerFamily(_string(row, "proposer_family"))
    except ValueError as error:
        raise Goal7ProtocolError("persisted cell has an invalid proposer family") from error
    raw_consumption = row.get("budget_consumption")
    consumption = (
        None if raw_consumption is None else BudgetConsumptionV1.from_dict(raw_consumption)
    )
    if status is GridCellStatus.COMPLETE:
        if runner_observed_wall_time is None:
            raise Goal7ProtocolError("complete cell lacks runner-observed wall-time evidence")
        if runner_observed_wall_time > budget.wall_time_seconds:
            raise Goal7ProtocolError(
                "complete cell exceeded the frozen runner-observed wall-time budget"
            )
        if consumption is None:
            raise Goal7ProtocolError("complete cell lacks observed training-budget consumption")
        _validate_budget_consumption(
            consumption,
            budget=budget,
            proposer_family=proposer_family,
        )
        if wall_time > budget.wall_time_seconds:
            raise Goal7ProtocolError("complete cell exceeded the frozen wall-time budget")
        if proposer_family is ProposerFamily.UNIFORM_VALID:
            if checkpoint is not None or parameter_count != 0 or estimated_flops != 0.0:
                raise Goal7ProtocolError(
                    "uniform-valid cells require no checkpoint and zero model parameters/FLOPs"
                )
        elif (
            checkpoint is None
            or parameter_count is None
            or parameter_count <= 0
            or estimated_flops is None
            or estimated_flops <= 0
        ):
            raise Goal7ProtocolError(
                "complete learned cells require checkpoint and positive parameter/FLOP telemetry"
            )
    if proposer_family is ProposerFamily.UNIFORM_VALID:
        if status is GridCellStatus.COMPLETE:
            _validate_uniform_draw_audits(
                uniform_audits,
                metric_rows=tuple(raw_rows),
                seed=_integer(row, "seed", minimum=0),
            )
        elif uniform_audits:
            raise Goal7ProtocolError(
                "non-complete uniform cells cannot carry accepted uniform audits"
            )
    elif uniform_audits:
        raise Goal7ProtocolError("learned cells cannot carry uniform-draw audits")
    envelope = _snapshot_run_envelope(row.get("run_envelope"))
    envelope_source = _string(row, "run_envelope_source")
    if envelope_source not in {
        "authoritative",
        "fixture_fallback",
        "pre_run_fallback_unverified",
        "rejected_unverified",
    }:
        raise Goal7ProtocolError("unknown cell run-envelope source")
    if stage is GridStage.PRODUCTION and status is GridCellStatus.COMPLETE:
        if envelope_source != "authoritative":
            raise Goal7ProtocolError("complete production cell lacks an authoritative envelope")
        expected_bindings = {
            "exact_command": row.get("reproduction_command"),
            "config_digest": row.get("config_digest"),
            "step_manifest_sha256": row.get("step_manifest_sha256"),
            "rule_registry_sha256": row.get("rule_registry_sha256"),
            "verifier_sha256": row.get("verifier_sha256"),
            "training_config_sha256": row.get("training_config_sha256"),
            "training_family_inventory_sha256": row.get("training_family_inventory_sha256"),
            "step_population_sha256": row.get("step_population_sha256"),
            "budget_consumption": row.get("budget_consumption"),
            "wall_time_seconds": row.get("wall_time_seconds"),
            "seed": row.get("seed"),
            "cell_id": row.get("cell_id"),
        }
        if any(envelope.get(field) != value for field, value in expected_bindings.items()):
            raise Goal7ProtocolError("production cell run-envelope binding mismatch")
    for metric_row in raw_rows:
        assert isinstance(metric_row, Mapping)
        _validate_metric_contract_values(
            metric_row,
            rule_registry_sha256=rule_registry_sha256,
            step_manifest_sha256=step_manifest_sha256,
            top_k=budget.top_k,
            evaluation_splits=evaluation_splits,
            require_family_evidence=stage is GridStage.PRODUCTION,
            require_legal_mask=status is GridCellStatus.COMPLETE,
            training_family_inventory_sha256=(training_family_inventory_sha256),
        )


def _persisted_uniform_audits(
    row: Mapping[str, object],
    *,
    rejected: bool,
) -> tuple[UniformDrawAuditV1, ...]:
    prefix = "rejected_uniform_draw" if rejected else "uniform_draw"
    raw = row.get(f"{prefix}_audits")
    if not isinstance(raw, list):
        raise Goal7ProtocolError(f"{prefix}_audits must be a list")
    try:
        audits = tuple(UniformDrawAuditV1.from_dict(item) for item in raw)
    except (TypeError, ValueError) as error:
        raise Goal7ProtocolError(f"invalid {prefix} audit: {error}") from error
    record_ids = tuple(audit.record_id for audit in audits)
    if len(record_ids) != len(set(record_ids)):
        raise Goal7ProtocolError(f"{prefix}_audits contain duplicate record IDs")
    if _integer(row, f"{prefix}_audit_count", minimum=0) != len(audits):
        raise Goal7ProtocolError(f"{prefix}_audit_count disagrees with retained audits")
    if _sha256_json(raw) != _string(row, f"{prefix}_audits_digest"):
        raise Goal7ProtocolError(f"{prefix}_audits_digest mismatch")
    return audits


def _validate_cross_cell_population(
    rows: Sequence[Mapping[str, object]],
    *,
    config: Goal7GridConfig,
) -> None:
    for row in rows:
        _validate_persisted_cell_outcome(row, config=config)
    _validate_complete_populations(rows)


def _validate_complete_populations(rows: Sequence[Mapping[str, object]]) -> None:
    complete_populations: set[tuple[tuple[object, ...], ...]] = set()
    for row in rows:
        if row["status"] == GridCellStatus.COMPLETE.value:
            metric_rows = row["metric_rows"]
            assert isinstance(metric_rows, list)
            complete_populations.add(_population_fingerprint(metric_rows))
    if len(complete_populations) > 1:
        raise Goal7ProtocolError(
            "complete cells evaluated different steps, goals, or legal-action masks"
        )
    _validate_shared_legal_inventories(rows)


def _validate_shared_legal_inventories(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Cross-bind every claimed legal candidate to uniform-arm inventory evidence."""

    from geml.learning.eval.step_metrics import (
        LegalityStatus,
        StepMetricOutcomeV1,
    )

    reference: dict[str, tuple[tuple[str, ...], tuple[bool, ...]]] | None = None
    for row in rows:
        if (
            row.get("status") != GridCellStatus.COMPLETE.value
            or row.get("proposer_family") != ProposerFamily.UNIFORM_VALID.value
        ):
            continue
        audits = _persisted_uniform_audits(row, rejected=False)
        observed = {
            audit.record_id: (
                audit.inventory_action_digests,
                audit.legal_mask,
            )
            for audit in audits
        }
        if reference is None:
            reference = observed
        elif observed != reference:
            raise Goal7ProtocolError(
                "uniform cells disagree on the shared legal-action inventories"
            )
    if reference is None:
        return

    legal_by_record = {
        record_id: {
            action_digest for action_digest, legal in zip(inventory, mask, strict=True) if legal
        }
        for record_id, (inventory, mask) in reference.items()
    }
    for cell in rows:
        if cell.get("status") != GridCellStatus.COMPLETE.value:
            continue
        metric_rows = cell.get("metric_rows")
        if not isinstance(metric_rows, list):
            raise Goal7ProtocolError("complete cell has invalid metric rows")
        for raw_metric in metric_rows:
            try:
                outcome = StepMetricOutcomeV1.from_dict(raw_metric)
                legal_digests = legal_by_record[outcome.record_id]
            except (KeyError, TypeError, ValueError) as error:
                raise Goal7ProtocolError(
                    "complete cell cannot be aligned to uniform legal-action evidence"
                ) from error
            for candidate in outcome.candidates:
                if candidate.legality_status is not LegalityStatus.LEGAL:
                    continue
                if candidate.action is None or candidate.action.action_digest not in legal_digests:
                    raise Goal7ProtocolError(
                        "a candidate marked legal is absent from the shared legal inventory"
                    )


def _check_population_reference(
    row: Mapping[str, object],
    *,
    reference: tuple[tuple[object, ...], ...] | None,
) -> tuple[tuple[object, ...], ...] | None:
    if row.get("status") != GridCellStatus.COMPLETE.value:
        return reference
    metric_rows = row.get("metric_rows")
    if not isinstance(metric_rows, list | tuple) or any(
        not isinstance(item, Mapping) for item in metric_rows
    ):
        raise Goal7ProtocolError("complete cell has invalid metric rows")
    observed = _population_fingerprint(metric_rows)
    if reference is not None and observed != reference:
        raise Goal7ProtocolError(
            "complete cells evaluated different steps, goals, or legal-action masks"
        )
    return observed


def _population_fingerprint(
    metric_rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[object, ...], ...]:
    invariant_fields = (
        "record_id",
        "trace_id",
        "source_group",
        "lineage_group_ids",
        "authoritative_split",
        "current_signature",
        "goal_signature",
        "target_successor_signature",
        "current_family",
        "goal_family",
        "evaluation_views",
        "family_generalization",
        "family_evidence_manifest_digest",
        "training_family_inventory_digest",
        "unseen_family_roles",
        "remaining_witness_steps",
        "trace_length",
        "demonstration_action",
        "registered_rule_ids",
        "registered_rule_directions",
        "rule_registry_digest",
        "requested_top_ks",
        "legal_action_count",
        "legal_mask_digest",
    )
    return tuple(
        sorted(
            (
                item["record_id"],
                _sha256_json({field: item.get(field) for field in invariant_fields}),
            )
            for item in metric_rows
        )
    )


def _run_completion(
    *,
    config: Goal7GridConfig,
    run_id: str,
    requests: Sequence[GridCellRequest],
    rows: Mapping[str, Mapping[str, JsonValue]],
    step_manifest_authentication: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    expected_ids = tuple(request.cell_id for request in requests)
    if set(rows) != set(expected_ids):
        raise Goal7ProtocolError("cannot complete a run with missing or unexpected cells")
    status_counts = Counter(str(rows[cell_id]["status"]) for cell_id in expected_ids)
    content: dict[str, JsonValue] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "config_digest": config.digest,
        "stage": config.stage.value,
        "channel_contract_resolved": config.channel_contract_resolved,
        "arm_ids": [arm.arm_id for arm in grid_arms(config)],
        "seeds": list(config.seeds),
        "expected_cell_ids": list(expected_ids),
        "expected_cell_count": len(expected_ids),
        "expected_step_count": config.expected_step_count,
        "retained_cell_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "selection_split": config.selection_split,
        "evaluation_splits": list(config.evaluation_splits),
        "budget": config.budget.as_dict(),
        "reproduction_command": config.reproduction_command,
        "cell_contracts": [_cell_contract(request) for request in requests],
        "step_manifest": config.step_manifest,
        "step_manifest_authentication": dict(step_manifest_authentication),
        "cell_content_digests": {
            cell_id: str(rows[cell_id]["content_digest"]) for cell_id in expected_ids
        },
        "cell_run_envelope_digests": {
            cell_id: _sha256_json(rows[cell_id]["run_envelope"]) for cell_id in expected_ids
        },
        "step_manifest_sha256": config.step_manifest_sha256,
        "rule_registry_sha256": config.rule_registry_sha256,
        "verifier_sha256": config.verifier_sha256,
        "shared_harness_sha256": config.shared_harness_sha256,
        "shared_gnn_architecture_sha256": config.shared_gnn_architecture_sha256,
        "transformer_architecture_sha256": config.transformer_architecture_sha256,
        "compute_reference_sha256": config.compute_reference_sha256,
        "implementation_sha256": config.implementation_sha256,
        "training_config_sha256": config.training_config_sha256,
        "training_family_inventory_sha256": (config.training_family_inventory_sha256),
        "step_population_sha256": config.step_population_sha256,
        "analysis_reproduction_command": config.analysis_reproduction_command,
        "budget_digest": config.budget.digest,
    }
    content["content_digest"] = hashlib.sha256(
        _RUN_CONTENT_DOMAIN + _canonical_json(content)
    ).hexdigest()
    return content


def _run_plan(
    *,
    config: Goal7GridConfig,
    run_id: str,
    requests: Sequence[GridCellRequest],
    step_manifest_authentication: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    content: dict[str, JsonValue] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": run_id,
        "config_digest": config.digest,
        "stage": config.stage.value,
        "channel_contract_resolved": config.channel_contract_resolved,
        "arm_ids": [arm.arm_id for arm in grid_arms(config)],
        "seeds": list(config.seeds),
        "expected_cell_ids": [request.cell_id for request in requests],
        "expected_cell_count": len(requests),
        "expected_step_count": config.expected_step_count,
        "selection_split": config.selection_split,
        "evaluation_splits": list(config.evaluation_splits),
        "budget": config.budget.as_dict(),
        "reproduction_command": config.reproduction_command,
        "cell_contracts": [_cell_contract(request) for request in requests],
        "step_manifest": config.step_manifest,
        "step_manifest_authentication": dict(step_manifest_authentication),
        "step_manifest_sha256": config.step_manifest_sha256,
        "rule_registry_sha256": config.rule_registry_sha256,
        "verifier_sha256": config.verifier_sha256,
        "shared_harness_sha256": config.shared_harness_sha256,
        "shared_gnn_architecture_sha256": config.shared_gnn_architecture_sha256,
        "transformer_architecture_sha256": config.transformer_architecture_sha256,
        "compute_reference_sha256": config.compute_reference_sha256,
        "implementation_sha256": config.implementation_sha256,
        "training_config_sha256": config.training_config_sha256,
        "training_family_inventory_sha256": (config.training_family_inventory_sha256),
        "step_population_sha256": config.step_population_sha256,
        "analysis_reproduction_command": config.analysis_reproduction_command,
        "budget_digest": config.budget.digest,
    }
    content["content_digest"] = hashlib.sha256(
        _PLAN_CONTENT_DOMAIN + _canonical_json(content)
    ).hexdigest()
    return content


def _cell_contract(request: GridCellRequest) -> dict[str, JsonValue]:
    """Persist enough immutable identity to validate cells without a live config file."""

    return {
        "arm_id": request.arm.arm_id,
        "cell_id": request.cell_id,
        "channel_id": request.arm.channel_id,
        "proposer_family": request.arm.proposer_family.value,
        "representation_mode": request.arm.representation_mode,
        "reproduction_command": request.reproduction_command,
        "seed": request.seed,
    }


def _run_id(config: Goal7GridConfig, arms: Sequence[GridArmV1]) -> str:
    identity = {
        "config_digest": config.digest,
        "step_manifest_sha256": config.step_manifest_sha256,
        "rule_registry_sha256": config.rule_registry_sha256,
        "verifier_sha256": config.verifier_sha256,
        "arm_ids": [arm.arm_id for arm in arms],
        "seeds": list(config.seeds),
    }
    return hashlib.sha256(_RUN_ID_DOMAIN + _canonical_json(identity)).hexdigest()


def _validate_metric_rows(rows: Sequence[Mapping[str, object]]) -> None:
    identities: list[str] = []
    for index, row in enumerate(rows):
        _validate_json(row, label=f"metric_rows[{index}]")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise Goal7ProtocolError("every metric row requires a non-blank record_id")
        source_group = row.get("source_group")
        if not isinstance(source_group, str) or not source_group.strip():
            raise Goal7ProtocolError("every metric row requires a non-blank source_group")
        identities.append(record_id)
    if len(identities) != len(set(identities)):
        raise Goal7ProtocolError("metric record IDs must be unique within a cell")


def _validate_metric_contract(
    row: Mapping[str, object],
    *,
    config: Goal7GridConfig,
    require_legal_mask: bool,
) -> None:
    if config.rule_registry_sha256 is None:
        raise Goal7ProtocolError("metric validation requires a frozen rule registry")
    if config.step_manifest_sha256 is None:
        raise Goal7ProtocolError("metric validation requires a frozen step manifest")
    _validate_metric_contract_values(
        row,
        rule_registry_sha256=config.rule_registry_sha256,
        step_manifest_sha256=config.step_manifest_sha256,
        top_k=config.budget.top_k,
        evaluation_splits=config.evaluation_splits,
        require_family_evidence=config.stage is GridStage.PRODUCTION,
        require_legal_mask=require_legal_mask,
        training_family_inventory_sha256=(config.training_family_inventory_sha256),
    )


def _validate_metric_contract_values(
    row: Mapping[str, object],
    *,
    rule_registry_sha256: str,
    step_manifest_sha256: str,
    top_k: tuple[int, ...],
    evaluation_splits: tuple[str, ...],
    require_family_evidence: bool,
    require_legal_mask: bool,
    training_family_inventory_sha256: str | None,
) -> None:
    from geml.learning.eval.step_metrics import (
        FamilyGeneralization,
        StepMetricOutcomeV1,
    )

    try:
        outcome = StepMetricOutcomeV1.from_dict(dict(row))
    except (KeyError, TypeError, ValueError) as error:
        raise Goal7ProtocolError(f"invalid typed step-metric row: {error}") from error
    if row.get("rule_registry_digest") != rule_registry_sha256:
        raise Goal7ProtocolError("metric row rule-registry digest mismatch")
    if row.get("requested_top_ks") != list(top_k):
        raise Goal7ProtocolError("metric row top-k contract mismatch")
    if outcome.authoritative_split not in evaluation_splits:
        raise Goal7ProtocolError("metric row is outside the frozen evaluation splits")
    if (
        outcome.family_evidence_manifest_digest is not None
        and outcome.family_evidence_manifest_digest != step_manifest_sha256
    ):
        raise Goal7ProtocolError("metric family evidence is bound to a different step manifest")
    if require_family_evidence and outcome.family_generalization is FamilyGeneralization.UNKNOWN:
        raise Goal7ProtocolError("production metric row lacks training-family evidence")
    if (
        outcome.training_family_inventory_digest is not None
        and outcome.training_family_inventory_digest != training_family_inventory_sha256
    ):
        raise Goal7ProtocolError(
            "metric family evidence is bound to a different training-family inventory"
        )
    if require_family_evidence and outcome.training_family_inventory_digest is None:
        raise Goal7ProtocolError("production metric row lacks training-family inventory evidence")
    mask_digest = row.get("legal_mask_digest")
    if require_legal_mask or mask_digest is not None:
        _require_sha256(mask_digest, field="metric legal_mask_digest")


def _cell_path(cells_directory: Path, cell_id: str) -> Path:
    return cells_directory / cell_id[:2] / f"{cell_id}.json"


def _atomic_create_json(path: Path, payload: Mapping[str, object]) -> None:
    raw = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            observed = path.read_bytes()
            if observed != raw:
                raise Goal7ProtocolError(
                    f"refusing to replace immutable evidence: {path}"
                ) from None
        except OSError as error:
            raise Goal7ProtocolError(f"cannot atomically publish {path}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path) -> dict[str, JsonValue]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Goal7ProtocolError(f"cannot read Goal 7 JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise Goal7ProtocolError(f"Goal 7 JSON root must be an object: {path}")
    _validate_json(payload, label=str(path))
    return payload


def _validate_content_digest(
    payload: Mapping[str, object],
    *,
    domain: bytes,
    label: str,
) -> None:
    claimed = payload.get("content_digest")
    content = {key: value for key, value in payload.items() if key != "content_digest"}
    observed = hashlib.sha256(domain + _canonical_json(content)).hexdigest()
    if claimed != observed:
        raise Goal7ProtocolError(f"{label} content digest mismatch")


def _validate_json(value: object, *, label: str) -> None:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        restored = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise Goal7ProtocolError(f"{label} must be strict finite JSON") from error
    if restored != value:
        raise Goal7ProtocolError(f"{label} changes under canonical JSON round trip")


def _json_clone(value: object) -> JsonValue:
    _validate_json(value, label="JSON payload")
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _canonical_json(value: object) -> bytes:
    _validate_json(value, label="canonical payload")
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _uniform_index(
    upper_exclusive: int,
    *,
    seed: int,
    record_id: str,
    counter: int,
) -> tuple[int, int]:
    limit = (1 << 256) - ((1 << 256) % upper_exclusive)
    while True:
        payload = {
            "seed": seed,
            "record_id": record_id,
            "counter": counter,
        }
        draw = int.from_bytes(
            hashlib.sha256(_UNIFORM_ORDER_DOMAIN + _canonical_json(payload)).digest(),
            "big",
        )
        counter += 1
        if draw < limit:
            return draw % upper_exclusive, counter


def _snapshot_run_envelope(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise Goal7ProtocolError("run-envelope adapter must return a mapping")
    snapshot = _json_clone(value)
    if not isinstance(snapshot, dict):
        raise Goal7ProtocolError("run-envelope snapshot must be a JSON object")
    _nonblank(snapshot.get("schema_version"), field="run-envelope schema_version")
    return snapshot


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise Goal7ProtocolError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _optional_digest(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, field=field)


def _nonblank(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{field} must be a non-blank string")
    return value


def _mapping(value: Mapping[str, object], field: str) -> Mapping[str, object]:
    result = value.get(field)
    if not isinstance(result, Mapping):
        raise Goal7ProtocolError(f"{field} must be an object")
    return result


def _string(value: Mapping[str, object], field: str) -> str:
    return _nonblank(value.get(field), field=field)


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _nonblank(value, field=field)


def _boolean(value: Mapping[str, object], field: str) -> bool:
    result = value.get(field)
    if type(result) is not bool:
        raise Goal7ProtocolError(f"{field} must be a strict boolean")
    return result


def _integer(
    value: Mapping[str, object],
    field: str,
    *,
    minimum: int,
) -> int:
    result = value.get(field)
    if type(result) is not int or result < minimum:
        raise Goal7ProtocolError(f"{field} must be an integer >= {minimum}")
    return result


def _optional_integer(
    value: object,
    *,
    field: str,
    minimum: int,
) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < minimum:
        raise Goal7ProtocolError(f"{field} must be an integer >= {minimum} or null")
    return value


def _number(
    value: Mapping[str, object],
    field: str,
    *,
    positive: bool,
) -> float:
    result = value.get(field)
    if (
        isinstance(result, bool)
        or not isinstance(result, int | float)
        or not math.isfinite(float(result))
        or (float(result) <= 0 if positive else float(result) < 0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise Goal7ProtocolError(f"{field} must be a finite {qualifier} number")
    return float(result)


def _optional_number(
    value: object,
    *,
    field: str,
    minimum: float,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise Goal7ProtocolError(f"{field} must be a finite number >= {minimum} or null")
    return float(value)


def _integer_sequence(
    value: Mapping[str, object],
    field: str,
    *,
    minimum: int = 0,
) -> list[int]:
    result = value.get(field)
    if not isinstance(result, list) or any(
        type(item) is not int or item < minimum for item in result
    ):
        raise Goal7ProtocolError(f"{field} must be a list of integers >= {minimum}")
    return result


def _string_sequence(value: Mapping[str, object], field: str) -> list[str]:
    result = value.get(field)
    if not isinstance(result, list) or any(
        not isinstance(item, str) or not item.strip() for item in result
    ):
        raise Goal7ProtocolError(f"{field} must be a list of nonblank strings")
    return result


def _bounded_message(error: BaseException, maximum: int = 1_000) -> str:
    value = str(error).strip() or type(error).__name__
    return value[:maximum]


def _validate_only(path: Path) -> int:
    config = load_goal7_grid_config(path)
    blockers = config.production_blockers()
    print(
        json.dumps(
            {
                "schema_version": config.schema_version,
                "stage": config.stage.value,
                "config_digest": config.digest,
                "status": "production_pending" if blockers else "phase_a_implemented",
                "production_blockers": list(blockers),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell-id")
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.validate_only:
        return _validate_only(arguments.config)
    raise Goal7ProtocolError(
        "production execution requires the integrated Workstream 1/2 providers; "
        "Phase A supports --validate-only and injected fixture executors"
    )


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "CELL_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "PRODUCTION_SEEDS",
    "RUN_SCHEMA_VERSION",
    "STEP_MANIFEST_AUTH_SCHEMA_VERSION",
    "UNIFORM_DRAW_AUDIT_SCHEMA_VERSION",
    "Goal7GridConfig",
    "Goal7ProtocolError",
    "Goal7RunEvidence",
    "GraphChannelSpec",
    "GridArmV1",
    "GridBudgetV1",
    "GridCellExecution",
    "GridCellExecutor",
    "GridCellRequest",
    "GridCellStatus",
    "GridRunReceipt",
    "GridStage",
    "ProposerFamily",
    "RunEnvelopeAdapter",
    "StepManifestAuthenticator",
    "UniformDrawAuditV1",
    "current_fixture_run_envelope",
    "enumerate_grid_cells",
    "fixture_run_envelope_adapter",
    "grid_arms",
    "load_goal7_grid_config",
    "load_goal7_run_evidence",
    "main",
    "run_goal7_grid",
    "uniform_valid_order",
]
