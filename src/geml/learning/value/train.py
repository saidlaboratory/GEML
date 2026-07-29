"""Leakage-safe orchestration and metrics for the Goal 8 value head.

This module intentionally delegates optimization, early stopping, and checkpoint
resume to the common Workstream 2 training harness. It does not introduce a second
training loop.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from geml.data.proofs.benchmark import (
    PRODUCTION_PROBLEM_COUNT,
    ManifestKind,
    load_benchmark_manifest,
    verify_frozen_manifest,
)
from geml.learning.value.head import ValueHeadConfig, ValuePredictionV1

PRODUCTION_SEEDS = (20260726, 20260727, 20260728)
BENCHMARK_SIZE = PRODUCTION_PROBLEM_COUNT
_REQUIRED_PRODUCTION_PACKAGES = frozenset({"geml", "numpy", "torch"})
_PROVENANCE_PLACEHOLDERS = frozenset(
    {
        "fixture",
        "n/a",
        "none",
        "not-installed",
        "not-recorded",
        "null",
        "unavailable",
        "unknown",
    }
)


class ValueTrainingConfigurationError(ValueError):
    """Training inputs violate a frozen scientific or leakage contract."""


class ValueRowStatus(StrEnum):
    ACCEPTED = "accepted"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"
    REPLAY_FAILED = "replay_failed"
    SCORING_ERROR = "scoring_error"


class ValueTrainingStatus(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"


class ValueExampleAdapter[RowT](Protocol):
    """Read only the shared StepRecordV1 fields needed by this workstream."""

    def record_id(self, row: RowT) -> str: ...

    def group_id(self, row: RowT) -> str: ...

    def related_group_ids(self, row: RowT) -> Sequence[str]: ...

    def record_digest(self, row: RowT) -> str: ...

    def split(self, row: RowT) -> str: ...

    def current_state(self, row: RowT) -> Any: ...

    def goal_state(self, row: RowT) -> Any: ...

    def remaining_witness_steps(self, row: RowT) -> int | None: ...

    def replay_verified(self, row: RowT) -> bool: ...

    def status(self, row: RowT) -> str: ...

    def failure_detail(self, row: RowT) -> str | None: ...


@dataclass(frozen=True, slots=True)
class FrozenBenchmarkReferenceV1:
    """Exact on-disk identity of the authoritative #67 benchmark manifest."""

    path: str
    expected_file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("benchmark manifest path must be non-blank")
        _require_sha256(self.expected_file_sha256, label="expected_file_sha256")


@dataclass(frozen=True, slots=True)
class FixtureBenchmarkExclusionsV1:
    """Explicit non-production exclusions for tiny self-contained tests only."""

    group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.group_ids, tuple)
            or not self.group_ids
            or any(not isinstance(value, str) or not value.strip() for value in self.group_ids)
            or tuple(sorted(set(self.group_ids))) != self.group_ids
        ):
            raise ValueError("fixture group_ids must be non-empty, unique, and sorted")

    @classmethod
    def create(cls, group_ids: Sequence[str]) -> FixtureBenchmarkExclusionsV1:
        return cls(group_ids=tuple(sorted(set(group_ids))))

    @property
    def content_digest(self) -> str:
        return _sha256_json(
            {
                "schema_version": "geml-value-fixture-exclusions-v1",
                "group_ids": list(self.group_ids),
            }
        )


@dataclass(frozen=True, slots=True)
class _ResolvedBenchmarkExclusions:
    group_ids: frozenset[str]
    manifest_content_sha256: str
    manifest_file_sha256: str | None


@dataclass(frozen=True, slots=True)
class ValueTrainingConfigV1:
    """Frozen three-seed cell configuration consumed by the common harness."""

    head: ValueHeadConfig
    stage: str = "fixture"
    benchmark_manifest_file_sha256: str | None = None
    benchmark_manifest_content_sha256: str | None = None
    seeds: tuple[int, ...] = PRODUCTION_SEEDS
    maximum_epochs: int = 100
    early_stopping_patience: int = 10
    batch_size: int = 64
    regression_loss: str = "right_censored_huber"
    huber_delta: float = 1.0
    regression_weight: float = 1.0
    ordinal_weight: float = 0.25
    selection_metric: str = "validation_mae"

    def __post_init__(self) -> None:
        if not isinstance(self.head, ValueHeadConfig):
            raise ValueError("head must be a ValueHeadConfig")
        if self.stage not in {"fixture", "production"}:
            raise ValueError("stage must be 'fixture' or 'production'")
        for label, digest in (
            ("benchmark_manifest_file_sha256", self.benchmark_manifest_file_sha256),
            (
                "benchmark_manifest_content_sha256",
                self.benchmark_manifest_content_sha256,
            ),
        ):
            if digest is not None:
                _require_sha256(digest, label=label)
        if (
            not isinstance(self.seeds, tuple)
            or any(type(seed) is not int for seed in self.seeds)
            or self.seeds != PRODUCTION_SEEDS
        ):
            raise ValueError(f"seeds must be exactly {PRODUCTION_SEEDS}")
        limits = (self.maximum_epochs, self.early_stopping_patience, self.batch_size)
        if any(type(value) is not int or value <= 0 for value in limits):
            raise ValueError("training limits must be positive")
        if self.early_stopping_patience > self.maximum_epochs:
            raise ValueError("early_stopping_patience cannot exceed maximum_epochs")
        if (
            self.regression_loss != "right_censored_huber"
            or self.selection_metric != "validation_mae"
        ):
            raise ValueError(
                "only frozen right-censored Huber loss and validation-MAE selection are supported"
            )
        numeric = (self.huber_delta, self.regression_weight, self.ordinal_weight)
        if any(
            type(value) is not float or not math.isfinite(value) or value <= 0 for value in numeric
        ):
            raise ValueError("loss delta and weights must be positive and finite")


@dataclass(frozen=True, slots=True)
class ValueTrainingEnvironmentV1:
    """Exact software, hardware, and command identity for delegated cells."""

    git_commit: str
    python_version: str
    package_versions: tuple[tuple[str, str], ...]
    hardware: str
    exact_command: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.git_commit,
                self.python_version,
                self.hardware,
                self.exact_command,
            )
        ):
            raise ValueError("value training environment fields must be non-blank")
        if (
            not isinstance(self.package_versions, tuple)
            or not self.package_versions
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or any(not isinstance(value, str) or not value.strip() for value in item)
                for item in self.package_versions
            )
        ):
            raise ValueError("package_versions must not be empty")
        package_names = tuple(name for name, _version in self.package_versions)
        if (
            len(set(package_names)) != len(package_names)
            or tuple(sorted(self.package_versions)) != self.package_versions
        ):
            raise ValueError("package_versions must be unique and canonically sorted")

    def require_production_ready(self) -> None:
        if not _is_git_commit(self.git_commit):
            raise ValueTrainingConfigurationError(
                "production git_commit must be a concrete full hexadecimal SHA"
            )
        for label, value in (
            ("python_version", self.python_version),
            ("hardware", self.hardware),
            ("exact_command", self.exact_command),
        ):
            if not _is_concrete_provenance(value):
                raise ValueTrainingConfigurationError(
                    f"production {label} must be concrete and non-placeholder"
                )
        versions = dict(self.package_versions)
        missing = sorted(_REQUIRED_PRODUCTION_PACKAGES - set(versions))
        if missing:
            raise ValueTrainingConfigurationError(
                "production package_versions is missing " + ", ".join(missing)
            )
        if any(
            not _is_concrete_provenance(name) or not _is_concrete_provenance(version)
            for name, version in self.package_versions
        ):
            raise ValueTrainingConfigurationError(
                "production package_versions contains blank or placeholder values"
            )

    @property
    def digest(self) -> str:
        return _sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class ValueTrainingInputIdentityV1:
    """Frozen upstream data, split, encoder, and common-harness identities."""

    step_dataset_sha256: str
    split_manifest_sha256: str
    shared_encoder_checkpoint_sha256: str
    shared_training_harness_sha256: str

    def __post_init__(self) -> None:
        for label, digest in asdict(self).items():
            _require_sha256(digest, label=label)

    @property
    def digest(self) -> str:
        return _sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class ValueTrainingRequestV1:
    """One seed delegated to the shared Workstream 2 training loop."""

    seed: int
    training_rows: tuple[Any, ...]
    validation_rows: tuple[Any, ...]
    config: ValueTrainingConfigV1
    config_digest: str
    exclusion_manifest_digest: str
    exclusion_manifest_file_sha256: str | None
    environment: ValueTrainingEnvironmentV1
    environment_digest: str
    input_identity: ValueTrainingInputIdentityV1
    input_identity_digest: str
    training_rows_digest: str
    validation_rows_digest: str
    resume: bool

    def __post_init__(self) -> None:
        if not isinstance(self.config, ValueTrainingConfigV1):
            raise ValueError("request config must be ValueTrainingConfigV1")
        if type(self.seed) is not int or self.seed not in self.config.seeds:
            raise ValueError("request seed must be one exact frozen seed")
        if type(self.resume) is not bool:
            raise ValueError("request resume must be a strict Boolean")
        if not isinstance(self.environment, ValueTrainingEnvironmentV1):
            raise ValueError("request environment must be ValueTrainingEnvironmentV1")
        if not isinstance(self.input_identity, ValueTrainingInputIdentityV1):
            raise ValueError("request input_identity must be ValueTrainingInputIdentityV1")
        if not isinstance(self.training_rows, tuple) or not isinstance(
            self.validation_rows,
            tuple,
        ):
            raise ValueError("request row collections must be tuples")
        _require_sha256(self.config_digest, label="config_digest")
        if self.config_digest != _sha256_json(asdict(self.config)):
            raise ValueError("value training config digest mismatch")
        _require_sha256(
            self.exclusion_manifest_digest,
            label="exclusion_manifest_digest",
        )
        if self.exclusion_manifest_file_sha256 is not None:
            _require_sha256(
                self.exclusion_manifest_file_sha256,
                label="exclusion_manifest_file_sha256",
            )
        _require_sha256(self.environment_digest, label="environment_digest")
        if self.environment_digest != self.environment.digest:
            raise ValueError("value training environment digest mismatch")
        _require_sha256(self.input_identity_digest, label="input_identity_digest")
        if self.input_identity_digest != self.input_identity.digest:
            raise ValueError("value training input identity digest mismatch")
        _require_sha256(self.training_rows_digest, label="training_rows_digest")
        _require_sha256(self.validation_rows_digest, label="validation_rows_digest")

    @property
    def digest(self) -> str:
        """Bind a delegated cell to every scientific input except runtime row objects."""

        return _sha256_json(
            {
                "schema_version": "geml-value-training-request-v1",
                "seed": self.seed,
                "config_digest": self.config_digest,
                "exclusion_manifest_digest": self.exclusion_manifest_digest,
                "exclusion_manifest_file_sha256": self.exclusion_manifest_file_sha256,
                "environment_digest": self.environment_digest,
                "input_identity_digest": self.input_identity_digest,
                "training_rows_digest": self.training_rows_digest,
                "validation_rows_digest": self.validation_rows_digest,
                "resume": self.resume,
            }
        )


@dataclass(frozen=True, slots=True)
class ValueSeedResultV1:
    """One retained seed outcome, including explicit training failures."""

    seed: int
    status: ValueTrainingStatus
    validation_mae: float | None
    checkpoint_digest: str | None
    epochs_completed: int
    resumed: bool | None
    request_digest: str
    resumed_from_checkpoint_digest: str | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in PRODUCTION_SEEDS:
            raise ValueError("unexpected value-head seed")
        if not isinstance(self.status, ValueTrainingStatus):
            raise ValueError("status must be a ValueTrainingStatus")
        if self.resumed is not None and type(self.resumed) is not bool:
            raise ValueError("resumed must be Boolean or None")
        if type(self.epochs_completed) is not int or self.epochs_completed < 0:
            raise ValueError("epochs_completed must be non-negative")
        _require_sha256(self.request_digest, label="request_digest")
        if self.resumed_from_checkpoint_digest is not None:
            _require_sha256(
                self.resumed_from_checkpoint_digest,
                label="resumed_from_checkpoint_digest",
            )
        if self.resumed is True and self.resumed_from_checkpoint_digest is None:
            raise ValueError("resumed seed requires its source checkpoint digest")
        if self.resumed is not True and self.resumed_from_checkpoint_digest is not None:
            raise ValueError("non-resumed seed cannot name a source checkpoint")
        if self.status is ValueTrainingStatus.COMPLETE:
            if self.resumed is None:
                raise ValueError("completed seed must attest whether it resumed")
            if (
                type(self.validation_mae) is not float
                or not math.isfinite(self.validation_mae)
                or self.validation_mae < 0
            ):
                raise ValueError("completed seed requires a finite non-negative validation MAE")
            if self.checkpoint_digest is None:
                raise ValueError("completed seed requires a checkpoint digest")
            _require_sha256(self.checkpoint_digest, label="checkpoint_digest")
            if self.failure_reason is not None:
                raise ValueError("completed seed cannot carry a failure reason")
        else:
            if not isinstance(self.failure_reason, str) or not self.failure_reason.strip():
                raise ValueError("failed seed requires a non-blank failure reason")
            if self.validation_mae is not None:
                raise ValueError("failed seed cannot expose a validation MAE")
            if self.checkpoint_digest is not None:
                _require_sha256(self.checkpoint_digest, label="checkpoint_digest")


@dataclass(frozen=True, slots=True)
class ValueTrainingInputRowV1:
    """One retained training/selection-source disposition."""

    record_id: str
    record_digest: str
    split: str
    related_group_ids: tuple[str, ...]
    status: ValueRowStatus
    target: int | None
    included: bool
    failure_reason: str | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.record_id, str)
            or not self.record_id.strip()
            or not isinstance(self.split, str)
            or not self.split.strip()
        ):
            raise ValueError("input-audit record_id and split must be non-blank")
        _require_sha256(self.record_digest, label="input-audit record_digest")
        if (
            not self.related_group_ids
            or tuple(sorted(set(self.related_group_ids))) != self.related_group_ids
        ):
            raise ValueError("input-audit related groups must be non-empty and sorted")
        if not isinstance(self.status, ValueRowStatus):
            raise ValueError("input-audit status must be a ValueRowStatus")
        if type(self.included) is not bool:
            raise ValueError("input-audit included must be a strict Boolean")
        if self.target is not None and not _is_non_negative_int(self.target):
            raise ValueError("input-audit target must be a non-negative exact integer or None")
        if self.included != (self.status is ValueRowStatus.ACCEPTED):
            raise ValueError("only accepted input-audit rows may be included")
        if self.included and (self.target is None or self.failure_reason is not None):
            raise ValueError("included input row requires a target and no failure")
        if not self.included and self.failure_reason is None:
            raise ValueError("excluded input row requires a failure reason")


@dataclass(frozen=True, slots=True)
class ValueTrainingInputAuditV1:
    rows: tuple[ValueTrainingInputRowV1, ...]
    attempted_count: int
    included_count: int
    unsupported_count: int
    failure_count: int

    def __post_init__(self) -> None:
        for label in (
            "attempted_count",
            "included_count",
            "unsupported_count",
            "failure_count",
        ):
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise ValueError(f"input audit {label} must be a non-negative exact integer")
        if self.attempted_count != len(self.rows):
            raise ValueError("input audit attempted_count must equal retained rows")
        if self.included_count != sum(row.included for row in self.rows):
            raise ValueError("input audit included_count mismatch")
        if self.unsupported_count != sum(
            row.status is ValueRowStatus.UNSUPPORTED for row in self.rows
        ):
            raise ValueError("input audit unsupported_count mismatch")
        if (
            self.included_count + self.unsupported_count + self.failure_count
            != self.attempted_count
        ):
            raise ValueError("input audit denominators do not sum to attempted_count")


@dataclass(frozen=True, slots=True)
class ValueTrainingSummaryV1:
    seed_results: tuple[ValueSeedResultV1, ...]
    selected_seed: int | None
    selection_metric: str
    benchmark_exclusion_manifest_digest: str
    benchmark_exclusion_file_sha256: str | None
    environment_digest: str
    input_identity_digest: str
    training_rows_digest: str
    validation_rows_digest: str
    input_audit: ValueTrainingInputAuditV1
    production_run: bool

    def __post_init__(self) -> None:
        if type(self.production_run) is not bool:
            raise ValueError("production_run must be a strict Boolean")
        _require_sha256(
            self.benchmark_exclusion_manifest_digest,
            label="benchmark_exclusion_manifest_digest",
        )
        _require_sha256(self.environment_digest, label="environment_digest")
        _require_sha256(self.input_identity_digest, label="input_identity_digest")
        _require_sha256(self.training_rows_digest, label="training_rows_digest")
        _require_sha256(self.validation_rows_digest, label="validation_rows_digest")
        if self.benchmark_exclusion_file_sha256 is not None:
            _require_sha256(
                self.benchmark_exclusion_file_sha256,
                label="benchmark_exclusion_file_sha256",
            )
        if self.production_run and self.benchmark_exclusion_file_sha256 is None:
            raise ValueError("production summary requires an authenticated benchmark file")
        if tuple(result.seed for result in self.seed_results) != PRODUCTION_SEEDS:
            raise ValueError("summary must retain exactly one ordered row per frozen seed")
        completed_seeds = {
            result.seed
            for result in self.seed_results
            if result.status is ValueTrainingStatus.COMPLETE
        }
        if self.selected_seed is not None and self.selected_seed not in completed_seeds:
            raise ValueError("selected_seed must identify a completed seed")


class SharedValueTrainingHarness(Protocol):
    """The only optimization-loop boundary used by this workstream."""

    def run_value_training_cell(
        self,
        request: ValueTrainingRequestV1,
    ) -> ValueSeedResultV1: ...


@dataclass(frozen=True, slots=True)
class HybridLossV1:
    total: float
    regression: float
    ordinal: float
    censored_count: int
    attempted_count: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
            for value in (self.total, self.regression, self.ordinal)
        ):
            raise ValueError("hybrid loss components must be finite and non-negative")
        if (
            type(self.censored_count) is not int
            or type(self.attempted_count) is not int
            or not 0 <= self.censored_count <= self.attempted_count
        ):
            raise ValueError("hybrid loss counts are inconsistent")


@dataclass(frozen=True, slots=True)
class ValueEvaluationRowV1:
    record_id: str
    group_id: str
    target: int | None
    status: ValueRowStatus
    prediction: float | None
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class CalibrationTierV1:
    name: str
    lower_inclusive: int
    upper_inclusive: int | None
    attempted_count: int
    valid_count: int
    unsupported_count: int
    failure_count: int
    mean_target: float | None
    mean_prediction: float | None
    mean_absolute_error: float | None

    @property
    def count(self) -> int:
        """Backward-readable alias for the valid calibration denominator."""

        return self.valid_count


@dataclass(frozen=True, slots=True)
class ValueMetricsV1:
    attempted_count: int
    valid_count: int
    unsupported_count: int
    failure_count: int
    unstratified_count: int
    mean_absolute_error: float | None
    spearman_rank_correlation: float | None
    calibration_by_witness_tier: tuple[CalibrationTierV1, ...]


@dataclass(frozen=True, slots=True)
class ValueEvaluationReportV1:
    rows: tuple[ValueEvaluationRowV1, ...]
    metrics: ValueMetricsV1


def run_three_seed_training[RowT](
    *,
    training_rows: Sequence[RowT],
    validation_rows: Sequence[RowT],
    adapter: ValueExampleAdapter[RowT],
    harness: SharedValueTrainingHarness,
    config: ValueTrainingConfigV1,
    environment: ValueTrainingEnvironmentV1,
    input_identity: ValueTrainingInputIdentityV1,
    benchmark_exclusions: FrozenBenchmarkReferenceV1 | FixtureBenchmarkExclusionsV1 | None,
    resume: bool = True,
) -> ValueTrainingSummaryV1:
    """Validate leakage boundaries, then delegate exactly three training cells."""

    if type(resume) is not bool:
        raise ValueTrainingConfigurationError("resume must be a strict Boolean")
    if not isinstance(config, ValueTrainingConfigV1):
        raise ValueTrainingConfigurationError("config must be ValueTrainingConfigV1")
    if not isinstance(environment, ValueTrainingEnvironmentV1):
        raise ValueTrainingConfigurationError("environment must be ValueTrainingEnvironmentV1")
    if config.stage == "production":
        environment.require_production_ready()
    if not isinstance(input_identity, ValueTrainingInputIdentityV1):
        raise ValueTrainingConfigurationError("input_identity must be ValueTrainingInputIdentityV1")
    if input_identity.shared_encoder_checkpoint_sha256 != config.head.shared_encoder_digest:
        raise ValueTrainingConfigurationError(
            "training input encoder does not match the value-head config"
        )
    resolved_exclusions = _resolve_benchmark_exclusions(
        benchmark_exclusions,
        config=config,
    )
    (
        eligible_training,
        eligible_validation,
        input_audit,
        training_rows_digest,
        validation_rows_digest,
    ) = _audit_training_rows(
        training_rows=training_rows,
        validation_rows=validation_rows,
        adapter=adapter,
        excluded_groups=resolved_exclusions.group_ids,
    )

    results: list[ValueSeedResultV1] = []
    for seed in config.seeds:
        request = ValueTrainingRequestV1(
            seed=seed,
            training_rows=eligible_training,
            validation_rows=eligible_validation,
            config=config,
            config_digest=_sha256_json(asdict(config)),
            exclusion_manifest_digest=resolved_exclusions.manifest_content_sha256,
            exclusion_manifest_file_sha256=resolved_exclusions.manifest_file_sha256,
            environment=environment,
            environment_digest=environment.digest,
            input_identity=input_identity,
            input_identity_digest=input_identity.digest,
            training_rows_digest=training_rows_digest,
            validation_rows_digest=validation_rows_digest,
            resume=resume,
        )
        try:
            result = harness.run_value_training_cell(request)
            if result.seed != seed:
                raise ValueError(f"harness returned seed {result.seed} for requested seed {seed}")
            if result.request_digest != request.digest:
                raise ValueError("harness result is not bound to the delegated request")
            if result.epochs_completed > config.maximum_epochs:
                raise ValueError("harness exceeded the configured maximum_epochs")
            if result.resumed is True and not resume:
                raise ValueError("harness reported a resumed cell when resume was disabled")
        except Exception as error:
            result = ValueSeedResultV1(
                seed=seed,
                status=ValueTrainingStatus.FAILED,
                validation_mae=None,
                checkpoint_digest=None,
                epochs_completed=0,
                resumed=None,
                request_digest=request.digest,
                failure_reason=_bounded_message(error),
            )
        results.append(result)

    completed = [
        result
        for result in results
        if result.status is ValueTrainingStatus.COMPLETE and result.validation_mae is not None
    ]
    selected_seed = (
        min(completed, key=lambda result: (result.validation_mae, result.seed)).seed
        if completed
        else None
    )
    return ValueTrainingSummaryV1(
        seed_results=tuple(results),
        selected_seed=selected_seed,
        selection_metric=config.selection_metric,
        benchmark_exclusion_manifest_digest=(resolved_exclusions.manifest_content_sha256),
        benchmark_exclusion_file_sha256=resolved_exclusions.manifest_file_sha256,
        environment_digest=environment.digest,
        input_identity_digest=input_identity.digest,
        training_rows_digest=training_rows_digest,
        validation_rows_digest=validation_rows_digest,
        input_audit=input_audit,
        production_run=config.stage == "production",
    )


def hybrid_value_loss(
    predictions: Sequence[ValuePredictionV1],
    targets: Sequence[int],
    config: ValueTrainingConfigV1,
) -> HybridLossV1:
    """Compute the frozen Huber plus ordinal-BCE objective.

    Targets above ``maximum_witness_steps`` are right-censored at that maximum.
    Their regression term is therefore a lower-bound surrogate, while every
    observable ordinal threshold remains positive.
    """

    if not isinstance(config, ValueTrainingConfigV1):
        raise ValueError("config must be ValueTrainingConfigV1")
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must be non-empty and aligned")
    if any(not isinstance(prediction, ValuePredictionV1) for prediction in predictions):
        raise ValueError("predictions must contain ValuePredictionV1 values")
    if any(
        isinstance(target, bool) or not isinstance(target, int) or target < 0 for target in targets
    ):
        raise ValueError("remaining_witness_steps targets must be non-negative integers")

    maximum = config.head.maximum_witness_steps
    clipped = np.asarray([min(target, maximum) for target in targets], dtype=np.float64)
    predicted = np.asarray(
        [prediction.remaining_witness_steps for prediction in predictions],
        dtype=np.float64,
    )
    residual = predicted - clipped
    censored = np.asarray([target > maximum for target in targets], dtype=np.bool_)
    # For a right-censored target we know only y > K. Predictions at or above K
    # satisfy that lower bound and must not be penalized for overshooting an
    # unobserved exact value; predictions below K retain a one-sided shortfall.
    residual = np.where(censored, np.minimum(residual, 0.0), residual)
    absolute = np.abs(residual)
    regression_terms = np.where(
        absolute <= config.huber_delta,
        0.5 * residual**2,
        config.huber_delta * (absolute - 0.5 * config.huber_delta),
    )
    regression = float(np.mean(regression_terms))

    probabilities = np.asarray(
        [prediction.ordinal_probabilities for prediction in predictions],
        dtype=np.float64,
    )
    expected_shape = (len(targets), config.head.ordinal_thresholds)
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"ordinal probability matrix must have shape {expected_shape}, "
            f"got {probabilities.shape}"
        )
    thresholds = np.arange(config.head.ordinal_thresholds, dtype=np.float64)
    ordinal_targets = clipped[:, None] > thresholds[None, :]
    epsilon = np.finfo(np.float64).eps
    bounded = np.clip(probabilities, epsilon, 1.0 - epsilon)
    ordinal_terms = -(
        ordinal_targets * np.log(bounded) + (1.0 - ordinal_targets) * np.log(1.0 - bounded)
    )
    ordinal = float(np.mean(ordinal_terms))
    total = config.regression_weight * regression + config.ordinal_weight * ordinal
    return HybridLossV1(
        total=total,
        regression=regression,
        ordinal=ordinal,
        censored_count=sum(target > maximum for target in targets),
        attempted_count=len(targets),
    )


def evaluate_value_head[RowT](
    *,
    rows: Sequence[RowT],
    adapter: ValueExampleAdapter[RowT],
    score_batch: Callable[[Sequence[Any], Sequence[Any]], Sequence[float]],
) -> ValueEvaluationReportV1:
    """Retain every attempted row and compute metrics only on valid predictions."""

    record_ids = [adapter.record_id(row) for row in rows]
    if any(not isinstance(value, str) or not value.strip() for value in record_ids):
        raise ValueError("evaluation record IDs must be non-blank strings")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("evaluation record IDs must be unique")

    targets_by_index = [adapter.remaining_witness_steps(row) for row in rows]
    replay_by_index = [adapter.replay_verified(row) for row in rows]
    if any(type(value) is not bool for value in replay_by_index):
        raise ValueError("evaluation replay_verified values must be strict Booleans")
    statuses_by_index = [adapter.status(row) for row in rows]
    accepted_indices = [
        index
        for index, row in enumerate(rows)
        if statuses_by_index[index] == ValueRowStatus.ACCEPTED
        and replay_by_index[index]
        and _is_non_negative_int(targets_by_index[index])
    ]
    scores: tuple[float, ...] | None = None
    scoring_error: str | None = None
    if accepted_indices:
        currents = [adapter.current_state(rows[index]) for index in accepted_indices]
        goals = [adapter.goal_state(rows[index]) for index in accepted_indices]
        try:
            raw_scores = tuple(score_batch(currents, goals))
            if len(raw_scores) != len(accepted_indices):
                raise ValueError("value scorer returned the wrong number of rows")
            if any(
                isinstance(score, bool)
                or not isinstance(score, int | float)
                or not math.isfinite(score)
                or score < 0
                for score in raw_scores
            ):
                raise ValueError("value scorer returned an invalid score")
            scores = tuple(float(score) for score in raw_scores)
        except Exception as error:
            scoring_error = _bounded_message(error)

    score_by_index = dict(zip(accepted_indices, scores, strict=True)) if scores is not None else {}
    retained: list[ValueEvaluationRowV1] = []
    for index, row in enumerate(rows):
        raw_target = targets_by_index[index]
        target = raw_target if _is_non_negative_int(raw_target) else None
        declared_status = statuses_by_index[index]
        source_detail = adapter.failure_detail(row)
        if declared_status == ValueRowStatus.UNSUPPORTED:
            status = ValueRowStatus.UNSUPPORTED
            prediction = None
            failure = source_detail or "unsupported input"
        elif (
            declared_status != ValueRowStatus.ACCEPTED
            or not replay_by_index[index]
            or target is None
        ):
            status = (
                ValueRowStatus.REPLAY_FAILED
                if not replay_by_index[index]
                else ValueRowStatus.INVALID
            )
            prediction = None
            failure = source_detail or (
                "remaining_witness_steps is unavailable"
                if target is None
                else f"retained source row status: {declared_status}"
            )
        elif scoring_error is not None:
            status = ValueRowStatus.SCORING_ERROR
            prediction = None
            failure = scoring_error
        else:
            status = ValueRowStatus.ACCEPTED
            prediction = score_by_index[index]
            failure = None
        retained.append(
            ValueEvaluationRowV1(
                record_id=adapter.record_id(row),
                group_id=adapter.group_id(row),
                target=target,
                status=status,
                prediction=prediction,
                failure_reason=failure,
            )
        )

    valid_rows = [row for row in retained if row.status is ValueRowStatus.ACCEPTED]
    targets = [float(row.target) for row in valid_rows if row.target is not None]
    predictions = [float(row.prediction) for row in valid_rows if row.prediction is not None]
    mae = (
        float(np.mean(np.abs(np.asarray(predictions) - np.asarray(targets))))
        if valid_rows
        else None
    )
    rank = _spearman(targets, predictions) if len(valid_rows) >= 2 else None
    unsupported_count = sum(row.status is ValueRowStatus.UNSUPPORTED for row in retained)
    return ValueEvaluationReportV1(
        rows=tuple(retained),
        metrics=ValueMetricsV1(
            attempted_count=len(retained),
            valid_count=len(valid_rows),
            unsupported_count=unsupported_count,
            failure_count=len(retained) - len(valid_rows) - unsupported_count,
            unstratified_count=sum(row.target is None for row in retained),
            mean_absolute_error=mae,
            spearman_rank_correlation=rank,
            calibration_by_witness_tier=_calibration_tiers(retained),
        ),
    )


def _resolve_benchmark_exclusions(
    source: FrozenBenchmarkReferenceV1 | FixtureBenchmarkExclusionsV1 | None,
    *,
    config: ValueTrainingConfigV1,
) -> _ResolvedBenchmarkExclusions:
    if config.stage == "production" and (
        config.benchmark_manifest_file_sha256 is None
        or config.benchmark_manifest_content_sha256 is None
    ):
        raise ValueTrainingConfigurationError(
            "production config requires preregistered #67 file and content digests"
        )
    if source is None:
        raise ValueTrainingConfigurationError(
            "proof-benchmark exclusions are required before value training"
        )
    if isinstance(source, FixtureBenchmarkExclusionsV1):
        if config.stage == "production":
            raise ValueTrainingConfigurationError(
                "production training requires an authenticated frozen #67 benchmark manifest"
            )
        return _ResolvedBenchmarkExclusions(
            group_ids=frozenset(source.group_ids),
            manifest_content_sha256=source.content_digest,
            manifest_file_sha256=None,
        )
    if (
        config.benchmark_manifest_file_sha256 is not None
        and source.expected_file_sha256 != config.benchmark_manifest_file_sha256
    ):
        raise ValueTrainingConfigurationError(
            "benchmark reference does not match the preregistered config file digest"
        )

    try:
        receipt = verify_frozen_manifest(
            Path(source.path),
            expected_file_sha256=source.expected_file_sha256,
        )
        manifest = load_benchmark_manifest(source.path)
        verified_after_load = verify_frozen_manifest(
            Path(source.path),
            expected_file_sha256=source.expected_file_sha256,
        )
    except Exception as error:
        raise ValueTrainingConfigurationError(
            f"proof-benchmark manifest authentication failed: {_bounded_message(error)}"
        ) from error
    if verified_after_load != receipt:
        raise ValueTrainingConfigurationError(
            "proof-benchmark manifest changed while it was being authenticated"
        )
    if config.stage == "production" and manifest.manifest_kind is not ManifestKind.PRODUCTION:
        raise ValueTrainingConfigurationError(
            "production training requires the production-kind 256-problem #67 manifest"
        )
    if config.stage == "production" and (
        manifest.target_count != BENCHMARK_SIZE or len(manifest.accepted) != BENCHMARK_SIZE
    ):
        raise ValueTrainingConfigurationError(
            "production training requires exactly 256 accepted benchmark problems"
        )
    if receipt.content_sha256 != manifest.content_sha256:
        raise ValueTrainingConfigurationError(
            "proof-benchmark manifest changed while it was being authenticated"
        )
    if (
        config.benchmark_manifest_content_sha256 is not None
        and manifest.content_sha256 != config.benchmark_manifest_content_sha256
    ):
        raise ValueTrainingConfigurationError(
            "benchmark manifest content does not match the preregistered config digest"
        )

    group_ids = {
        related_id
        for problem in manifest.accepted
        for related_id in (
            problem.candidate.group_id,
            *problem.candidate.lineage_group_ids,
            *problem.candidate.eclass_relative_ids,
        )
    }
    if not group_ids:
        raise ValueTrainingConfigurationError(
            "proof-benchmark manifest contains no accepted exclusion groups"
        )
    return _ResolvedBenchmarkExclusions(
        group_ids=frozenset(group_ids),
        manifest_content_sha256=manifest.content_sha256,
        manifest_file_sha256=receipt.file_sha256,
    )


def _audit_training_rows[RowT](
    *,
    training_rows: Sequence[RowT],
    validation_rows: Sequence[RowT],
    adapter: ValueExampleAdapter[RowT],
    excluded_groups: frozenset[str],
) -> tuple[
    tuple[RowT, ...],
    tuple[RowT, ...],
    ValueTrainingInputAuditV1,
    str,
    str,
]:
    if not training_rows or not validation_rows:
        raise ValueTrainingConfigurationError("training and validation rows must both be non-empty")
    all_rows = tuple(training_rows) + tuple(validation_rows)
    record_ids = [adapter.record_id(row) for row in all_rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueTrainingConfigurationError("value training record IDs must be unique")

    training_audit = tuple(
        _audit_training_row(row, adapter, expected_split="train") for row in training_rows
    )
    validation_audit = tuple(
        _audit_training_row(row, adapter, expected_split="validation") for row in validation_rows
    )
    train_groups = {group_id for row in training_audit for group_id in row.related_group_ids}
    validation_groups = {group_id for row in validation_audit for group_id in row.related_group_ids}
    leaked = (train_groups | validation_groups) & excluded_groups
    if leaked:
        raise ValueTrainingConfigurationError(
            "proof-benchmark group or relative leaked into value training/selection: "
            f"{sorted(leaked)}"
        )
    cross_split = train_groups & validation_groups
    if cross_split:
        raise ValueTrainingConfigurationError(
            f"groups or relatives cross train/validation split: {sorted(cross_split)}"
        )
    eligible_training = tuple(
        row for row, audit in zip(training_rows, training_audit, strict=True) if audit.included
    )
    eligible_validation = tuple(
        row for row, audit in zip(validation_rows, validation_audit, strict=True) if audit.included
    )
    if not eligible_training or not eligible_validation:
        raise ValueTrainingConfigurationError(
            "training and validation must each retain at least one replay-verified row"
        )

    audit_rows = training_audit + validation_audit
    included_count = sum(row.included for row in audit_rows)
    unsupported_count = sum(row.status is ValueRowStatus.UNSUPPORTED for row in audit_rows)
    audit = ValueTrainingInputAuditV1(
        rows=audit_rows,
        attempted_count=len(audit_rows),
        included_count=included_count,
        unsupported_count=unsupported_count,
        failure_count=len(audit_rows) - included_count - unsupported_count,
    )
    return (
        eligible_training,
        eligible_validation,
        audit,
        _row_projection_digest(training_audit, split="train"),
        _row_projection_digest(validation_audit, split="validation"),
    )


def _audit_training_row[RowT](
    row: RowT,
    adapter: ValueExampleAdapter[RowT],
    *,
    expected_split: str,
) -> ValueTrainingInputRowV1:
    record_id = adapter.record_id(row)
    split = adapter.split(row)
    if split != expected_split:
        raise ValueTrainingConfigurationError(
            f"row {record_id} is not in expected {expected_split} split"
        )
    record_digest = adapter.record_digest(row)
    _require_sha256(record_digest, label=f"row {record_id} record_digest")
    related_groups = tuple(adapter.related_group_ids(row))
    primary_group = adapter.group_id(row)
    if (
        not primary_group
        or primary_group not in related_groups
        or tuple(sorted(set(related_groups))) != related_groups
        or any(not isinstance(group_id, str) or not group_id for group_id in related_groups)
    ):
        raise ValueTrainingConfigurationError(
            f"row {record_id} has invalid or incomplete related_group_ids"
        )
    try:
        declared_status = ValueRowStatus(adapter.status(row))
    except ValueError as error:
        raise ValueTrainingConfigurationError(
            f"row {record_id} has an unknown source status"
        ) from error
    replay_verified = adapter.replay_verified(row)
    if not isinstance(replay_verified, bool):
        raise ValueTrainingConfigurationError(f"row {record_id} replay_verified must be Boolean")
    target = adapter.remaining_witness_steps(row)
    target_valid = not isinstance(target, bool) and isinstance(target, int) and target >= 0
    included = declared_status is ValueRowStatus.ACCEPTED and replay_verified and target_valid
    if declared_status is ValueRowStatus.ACCEPTED and replay_verified and not target_valid:
        raise ValueTrainingConfigurationError(
            f"row {record_id} has invalid remaining_witness_steps"
        )
    if included:
        status = ValueRowStatus.ACCEPTED
        failure_reason = None
    elif declared_status is ValueRowStatus.UNSUPPORTED:
        status = ValueRowStatus.UNSUPPORTED
        failure_reason = adapter.failure_detail(row) or "unsupported input"
    elif not replay_verified:
        status = ValueRowStatus.REPLAY_FAILED
        failure_reason = adapter.failure_detail(row) or "stored witness replay failed"
    else:
        status = ValueRowStatus.INVALID
        failure_reason = adapter.failure_detail(row) or f"source status: {declared_status.value}"
    return ValueTrainingInputRowV1(
        record_id=record_id,
        record_digest=record_digest,
        split=split,
        related_group_ids=related_groups,
        status=status,
        target=target if target_valid else None,
        included=included,
        failure_reason=(_bounded_message(failure_reason) if failure_reason is not None else None),
    )


def _row_projection_digest(
    rows: Sequence[ValueTrainingInputRowV1],
    *,
    split: str,
) -> str:
    return _sha256_json(
        {
            "schema_version": "geml-value-training-row-projection-v1",
            "split": split,
            "rows": [asdict(row) for row in rows],
        }
    )


def _calibration_tiers(
    rows: Sequence[ValueEvaluationRowV1],
) -> tuple[CalibrationTierV1, ...]:
    definitions = (
        ("zero", 0, 0),
        ("one", 1, 1),
        ("two_to_three", 2, 3),
        ("four_to_seven", 4, 7),
        ("eight_plus", 8, None),
    )
    tiers: list[CalibrationTierV1] = []
    for name, lower, upper in definitions:
        attempted = [
            row
            for row in rows
            if row.target is not None
            and row.target >= lower
            and (upper is None or row.target <= upper)
        ]
        valid = [row for row in attempted if row.status is ValueRowStatus.ACCEPTED]
        if valid:
            targets = np.asarray([row.target for row in valid], dtype=np.float64)
            predictions = np.asarray(
                [row.prediction for row in valid],
                dtype=np.float64,
            )
            mean_target = float(np.mean(targets))
            mean_prediction = float(np.mean(predictions))
            mae = float(np.mean(np.abs(predictions - targets)))
        else:
            mean_target = None
            mean_prediction = None
            mae = None
        tiers.append(
            CalibrationTierV1(
                name=name,
                lower_inclusive=lower,
                upper_inclusive=upper,
                attempted_count=len(attempted),
                valid_count=len(valid),
                unsupported_count=sum(
                    row.status is ValueRowStatus.UNSUPPORTED for row in attempted
                ),
                failure_count=sum(
                    row.status not in {ValueRowStatus.ACCEPTED, ValueRowStatus.UNSUPPORTED}
                    for row in attempted
                ),
                mean_target=mean_target,
                mean_prediction=mean_prediction,
                mean_absolute_error=mae,
            )
        )
    return tuple(tiers)


def _is_non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_git_commit(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and value == value.lower()
        and len(value) in {40, 64}
        and len(set(value)) >= 2
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_concrete_provenance(value: object) -> bool:
    if not isinstance(value, str) or value != value.strip() or not value:
        return False
    normalized = "-".join(value.casefold().replace("_", "-").split())
    return normalized not in _PROVENANCE_PLACEHOLDERS


def _spearman(targets: Sequence[float], predictions: Sequence[float]) -> float | None:
    target_ranks = _average_ranks(targets)
    prediction_ranks = _average_ranks(predictions)
    target_centered = target_ranks - np.mean(target_ranks)
    prediction_centered = prediction_ranks - np.mean(prediction_ranks)
    denominator = math.sqrt(float(np.sum(target_centered**2) * np.sum(prediction_centered**2)))
    if denominator == 0:
        return None
    return float(np.sum(target_centered * prediction_centered) / denominator)


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average
        start = end
    return ranks


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _bounded_message(error: BaseException | str, maximum: int = 1_000) -> str:
    message = str(error).strip() or type(error).__name__
    return message[:maximum]


__all__ = [
    "BENCHMARK_SIZE",
    "PRODUCTION_SEEDS",
    "CalibrationTierV1",
    "FixtureBenchmarkExclusionsV1",
    "FrozenBenchmarkReferenceV1",
    "HybridLossV1",
    "SharedValueTrainingHarness",
    "ValueEvaluationReportV1",
    "ValueEvaluationRowV1",
    "ValueExampleAdapter",
    "ValueMetricsV1",
    "ValueRowStatus",
    "ValueSeedResultV1",
    "ValueTrainingConfigV1",
    "ValueTrainingConfigurationError",
    "ValueTrainingEnvironmentV1",
    "ValueTrainingInputAuditV1",
    "ValueTrainingInputIdentityV1",
    "ValueTrainingInputRowV1",
    "ValueTrainingRequestV1",
    "ValueTrainingStatus",
    "ValueTrainingSummaryV1",
    "evaluate_value_head",
    "hybrid_value_loss",
    "run_three_seed_training",
]
