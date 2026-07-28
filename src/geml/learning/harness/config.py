"""Issue 6-4: typed, hashable, task-neutral harness configuration.

The harness is one implementation for Goals 6-9.  Nothing here forks on task or on representation:
a channel name is carried as opaque metadata and never changes behaviour.

Every configuration is canonically serialized and bound to a version-tagged SHA-256 digest, so a
result row can always be traced back to the exact configuration that produced it.  Unknown keys are
rejected rather than ignored, because a silently dropped key is a silently changed experiment.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    model_validator,
)

from geml.learning.harness.seeds import PRODUCTION_SEEDS

HARNESS_CONFIG_SCHEMA_VERSION = "geml-goal6-harness-config-v1"
RUN_ENVELOPE_SCHEMA_VERSION = "geml-goal6-run-envelope-v1"

#: Issue 6-4 caps training at approximately 30 epochs absent a documented approved exception.
DEFAULT_MAX_EPOCHS = 30

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class HarnessConfigError(ValueError):
    """A harness configuration violates the frozen issue 6-4 contract."""


class Precision(StrEnum):
    """Declared numerical precision; the harness records what it actually used."""

    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


class ComparisonUnit(StrEnum):
    """The frozen unit that compute matching is expressed in."""

    PER_PAIR = "per_pair"
    PER_NODE = "per_node"
    PER_OPTIMIZER_STEP = "per_optimizer_step"


def _enum_from_name(enum_type: type[StrEnum]):
    """Accept the exact spelling of an enum member from YAML without relaxing strict mode.

    Strict validation is kept everywhere else: a misspelled member still raises, and no numeric or
    boolean coercion is permitted anywhere in the configuration.
    """

    def coerce(value: object) -> object:
        if isinstance(value, str) and not isinstance(value, enum_type):
            try:
                return enum_type(value)
            except ValueError as error:
                permitted = ", ".join(member.value for member in enum_type)
                raise ValueError(
                    f"{value!r} is not a valid {enum_type.__name__}; permitted values: {permitted}"
                ) from error
        return value

    return coerce


PrecisionField = Annotated[Precision, BeforeValidator(_enum_from_name(Precision))]
ComparisonUnitField = Annotated[ComparisonUnit, BeforeValidator(_enum_from_name(ComparisonUnit))]


class _HarnessContract(BaseModel):
    """Strict base: unknown keys are an error, and values are never coerced."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BatchBudget(_HarnessContract):
    """Dynamic batching bounded by node and edge counts rather than by example count.

    Pure-EML graphs are typically far larger than AST or macro graphs for the same expression, so a
    fixed example-count batch would silently give the channels different compute per step.
    """

    max_nodes_per_batch: Annotated[StrictInt, Field(ge=1)] = 4096
    max_edges_per_batch: Annotated[StrictInt, Field(ge=1)] = 8192
    max_examples_per_batch: Annotated[StrictInt, Field(ge=1)] = 64
    gradient_accumulation_steps: Annotated[StrictInt, Field(ge=1)] = 1


class EarlyStopping(_HarnessContract):
    """One preregistered validation criterion, frozen before training."""

    criterion: _NonBlankStr = "validation_loss"
    mode: Literal["min", "max"] = "min"
    patience: Annotated[StrictInt, Field(ge=1)] = 5
    min_delta: Annotated[StrictFloat, Field(ge=0.0)] = 0.0


class ComputeMatchPolicy(_HarnessContract):
    """The frozen compute-matching contract shared by every arm."""

    reference_workload: _NonBlankStr = "goal6-equivalence-reference-batch-v1"
    parameter_tolerance: Annotated[StrictFloat, Field(gt=0.0, le=1.0)] = 0.15
    flop_tolerance: Annotated[StrictFloat, Field(gt=0.0, le=1.0)] = 0.25
    comparison_unit: ComparisonUnitField = ComparisonUnit.PER_OPTIMIZER_STEP
    total_optimizer_steps: Annotated[StrictInt, Field(ge=1)] = 20_000
    vocabulary_embeddings_count_toward_match: StrictBool = True


class OptimizerConfig(_HarnessContract):
    """Identical optimizer policy across arms; not a per-channel tuning surface."""

    name: Literal["adamw"] = "adamw"
    learning_rate: Annotated[StrictFloat, Field(gt=0.0)] = 1e-3
    weight_decay: Annotated[StrictFloat, Field(ge=0.0)] = 0.01
    gradient_clip_norm: Annotated[StrictFloat, Field(gt=0.0)] | None = 1.0


class HarnessConfig(_HarnessContract):
    """The complete, hashable harness configuration."""

    schema_version: Literal["geml-goal6-harness-config-v1"] = HARNESS_CONFIG_SCHEMA_VERSION
    run_name: _NonBlankStr
    output_root: _NonBlankStr
    seed: StrictInt
    max_epochs: Annotated[StrictInt, Field(ge=1)] = DEFAULT_MAX_EPOCHS
    max_epochs_exception_reason: _NonBlankStr | None = None
    precision: PrecisionField = Precision.FLOAT32
    device: _NonBlankStr = "cpu"
    batch_budget: BatchBudget = BatchBudget()
    early_stopping: EarlyStopping = EarlyStopping()
    optimizer: OptimizerConfig = OptimizerConfig()
    compute_match: ComputeMatchPolicy = ComputeMatchPolicy()
    deterministic: StrictBool = True
    require_production_seed: StrictBool = True
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_frozen_policy(self) -> Self:
        if self.require_production_seed and self.seed not in PRODUCTION_SEEDS:
            raise ValueError(
                f"seed {self.seed} is not one of the preregistered production seeds "
                f"{PRODUCTION_SEEDS}; set require_production_seed=false only for fixtures"
            )
        if self.max_epochs > DEFAULT_MAX_EPOCHS and not self.max_epochs_exception_reason:
            raise ValueError(
                f"max_epochs above {DEFAULT_MAX_EPOCHS} requires an explicit documented "
                "max_epochs_exception_reason"
            )
        return self

    @property
    def config_hash(self) -> str:
        """The version-tagged SHA-256 digest of the canonical configuration payload."""

        return config_digest(self)


def canonical_json_bytes(value: object) -> bytes:
    """Serialize deterministically: sorted keys, no insignificant whitespace, UTF-8."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def config_digest(config: HarnessConfig) -> str:
    """Return the version-tagged content digest of a harness configuration."""

    payload = canonical_json_bytes(config.model_dump(mode="json"))
    tagged = HARNESS_CONFIG_SCHEMA_VERSION.encode("utf-8") + b"\0" + payload
    return hashlib.sha256(tagged).hexdigest()


def load_harness_config(payload: dict[str, object]) -> HarnessConfig:
    """Validate an untrusted mapping into a :class:`HarnessConfig`.

    Pydantic's ``extra="forbid"`` turns an unrecognised key into an error, so a typo in a YAML file
    fails loudly instead of silently reverting to a default.
    """

    try:
        return HarnessConfig.model_validate(payload)
    except ValueError as error:
        raise HarnessConfigError(str(error)) from error


class RunEnvelope(_HarnessContract):
    """The reproducibility envelope bound to every dataset, cell, shard, and analysis."""

    schema_version: Literal["geml-goal6-run-envelope-v1"] = RUN_ENVELOPE_SCHEMA_VERSION
    config_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    git_commit: _NonBlankStr
    python_version: _NonBlankStr
    package_versions: dict[str, str]
    seed: StrictInt
    device: _NonBlankStr
    precision: PrecisionField
    deterministic_algorithms: StrictBool
    nondeterministic_notes: tuple[str, ...] = ()
    input_checksums: dict[str, str] = Field(default_factory=dict)
    output_checksums: dict[str, str] = Field(default_factory=dict)
    started_at: _NonBlankStr
    ended_at: _NonBlankStr | None = None
    wall_seconds: Annotated[StrictFloat, Field(ge=0.0)] | None = None
    parameters: Annotated[StrictInt, Field(ge=0)] | None = None
    forward_flops: Annotated[StrictInt, Field(ge=0)] | None = None
    peak_host_memory_bytes: Annotated[StrictInt, Field(ge=0)] | None = None
    peak_device_memory_bytes: Annotated[StrictInt, Field(ge=0)] | None = None
    attempted: Annotated[StrictInt, Field(ge=0)] = 0
    succeeded: Annotated[StrictInt, Field(ge=0)] = 0
    failed: Annotated[StrictInt, Field(ge=0)] = 0
    unsupported: Annotated[StrictInt, Field(ge=0)] = 0
    timed_out: Annotated[StrictInt, Field(ge=0)] = 0
    resume_lineage: tuple[str, ...] = ()
    reproduction_command: _NonBlankStr

    @model_validator(mode="after")
    def validate_denominators(self) -> Self:
        accounted = self.succeeded + self.failed + self.unsupported + self.timed_out
        if accounted > self.attempted:
            raise ValueError(
                f"outcome counts ({accounted}) exceed attempted ({self.attempted}); every "
                "aggregate must reconstruct its own denominator"
            )
        return self
