"""Frozen, content-addressed training configuration contracts for Goals 6--9."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    model_validator,
)

HARNESS_CONFIG_SCHEMA_VERSION = "geml-training-config-v1"
RUN_ENVELOPE_SCHEMA_VERSION = "geml-run-envelope-v1"

_NonBlankStr = Annotated[str, Field(min_length=1, pattern=r"\S")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class PrecisionMode(StrEnum):
    """Recorded effective precision rather than an implicit accelerator default."""

    FLOAT32 = "float32"
    BF16 = "bf16"


class TrainingStatus(StrEnum):
    """Retained outcome statuses for any attempted training cell."""

    COMPLETE = "complete"
    EARLY_STOPPED = "early_stopped"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"


class _HarnessContract(BaseModel):
    """Strict base for reproducibility artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize configuration and result identities without process-local hashes."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_digest(value: object) -> str:
    """Return a qualified SHA-256 digest of one canonical JSON-compatible object."""

    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


class DynamicBatchConfig(_HarnessContract):
    """One deterministic node/edge-bounded optimizer-update policy."""

    node_budget: StrictInt = Field(gt=0)
    edge_budget: StrictInt = Field(gt=0)
    gradient_accumulation_steps: StrictInt = Field(default=1, gt=0)


class EarlyStoppingConfig(_HarnessContract):
    """Validation-only early stopping policy frozen before test evaluation."""

    patience: StrictInt = Field(default=5, ge=0)
    minimize_metric: StrictBool = True
    minimum_delta: StrictFloat = Field(default=0.0, ge=0)


class TrainingConfigV1(_HarnessContract):
    """Complete shared task-neutral training-cell configuration."""

    schema_version: str = HARNESS_CONFIG_SCHEMA_VERSION
    run_id: _NonBlankStr
    seed: StrictInt
    epochs: StrictInt = Field(default=30, ge=1, le=30)
    optimizer: _NonBlankStr = "adamw"
    learning_rate: StrictFloat = Field(gt=0)
    weight_decay: StrictFloat = Field(default=0.0, ge=0)
    precision: PrecisionMode = PrecisionMode.FLOAT32
    deterministic_algorithms_requested: StrictBool = True
    dynamic_batch: DynamicBatchConfig
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpoint_path: _NonBlankStr
    result_path: _NonBlankStr
    timeout_seconds: StrictFloat | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_schema(self) -> Self:
        if self.schema_version != HARNESS_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unexpected training config schema {self.schema_version!r}")
        return self

    @property
    def config_hash(self) -> str:
        """Content hash used to reject unsafe checkpoint/config reuse."""

        return sha256_digest(self.model_dump(mode="json"))


class OutcomeCountsV1(_HarnessContract):
    """Complete attempted/success/failure denominator accounting."""

    attempted: _NonNegativeInt
    successful: _NonNegativeInt
    failed: _NonNegativeInt
    unsupported: _NonNegativeInt
    invalid: _NonNegativeInt
    timeout: _NonNegativeInt

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        terminal_count = (
            self.successful + self.failed + self.unsupported + self.invalid + self.timeout
        )
        if self.attempted != terminal_count:
            raise ValueError("attempted must equal the complete terminal-outcome denominator")
        return self


class RunEnvelopeV1(_HarnessContract):
    """Common append-only run evidence required by every learning/search cell."""

    schema_version: str = RUN_ENVELOPE_SCHEMA_VERSION
    configuration: dict[str, object]
    configuration_hash: _NonBlankStr
    git_commit: _NonBlankStr
    package_versions: dict[_NonBlankStr, _NonBlankStr]
    seeds: tuple[StrictInt, ...]
    hardware: dict[str, object]
    precision: PrecisionMode
    deterministic_settings: dict[str, object]
    input_checksums: dict[_NonBlankStr, _NonBlankStr]
    output_checksums: dict[_NonBlankStr, _NonBlankStr]
    started_at: _NonBlankStr
    ended_at: _NonBlankStr | None = None
    wall_seconds: StrictFloat | None = Field(default=None, ge=0)
    resource_telemetry: dict[str, object]
    outcomes: OutcomeCountsV1
    resume_lineage: dict[str, object]
    reproduction_command: _NonBlankStr

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        if self.schema_version != RUN_ENVELOPE_SCHEMA_VERSION:
            raise ValueError(f"unexpected run envelope schema {self.schema_version!r}")
        if self.configuration_hash != sha256_digest(self.configuration):
            raise ValueError("configuration_hash must bind canonical configuration content")
        if not self.seeds:
            raise ValueError("run envelope must retain every seed")
        return self
