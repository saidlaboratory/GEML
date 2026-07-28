"""Compact goal-conditioned value head over a shared graph encoder.

The target is ``remaining_witness_steps`` for a particular ordered
``(current, goal)`` pair. It is a witnessed upper bound unless the trace producer
has separately proved shortest-path optimality; the score is therefore
non-admissible search guidance, not a shortest-proof certificate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt

StateT = TypeVar("StateT")
FloatArray = npt.NDArray[np.float64]


class SharedGraphEncoder(Protocol[StateT]):
    """Narrow evaluation boundary implemented by the Workstream 2 encoder adapter."""

    @property
    def embedding_dimension(self) -> int: ...

    @property
    def encoder_digest(self) -> str: ...

    @property
    def trainable_parameter_count(self) -> int: ...

    def encode_batch(
        self,
        states: Sequence[StateT],
        *,
        training: bool,
    ) -> Sequence[Sequence[float]] | npt.ArrayLike: ...


@dataclass(frozen=True, slots=True)
class ValueHeadConfig:
    """Architecture and scientifically explicit target clipping policy."""

    embedding_dimension: int
    shared_encoder_digest: str
    hidden_dimension: int = 32
    ordinal_thresholds: int = 16
    maximum_witness_steps: int = 16
    maximum_trainable_parameters: int = 20_000
    maximum_total_trainable_parameters: int = 1_000_000
    target_policy: str = "right_censor_at_maximum"

    def __post_init__(self) -> None:
        _require_sha256(self.shared_encoder_digest, label="shared_encoder_digest")
        dimensions = (
            self.embedding_dimension,
            self.hidden_dimension,
            self.ordinal_thresholds,
            self.maximum_witness_steps,
            self.maximum_trainable_parameters,
            self.maximum_total_trainable_parameters,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("value-head dimensions and limits must be positive")
        if self.ordinal_thresholds != self.maximum_witness_steps:
            raise ValueError(
                "ordinal_thresholds must equal maximum_witness_steps for thresholds 0..K-1"
            )
        if self.target_policy != "right_censor_at_maximum":
            raise ValueError("unsupported target clipping/censoring policy")
        if self.maximum_total_trainable_parameters < self.maximum_trainable_parameters:
            raise ValueError(
                "maximum_total_trainable_parameters cannot be smaller than the head budget"
            )


@dataclass(frozen=True, slots=True)
class ValueHeadParameters:
    """Framework-neutral parameters that an injected training harness may optimize."""

    hidden_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    regression_weights: tuple[float, ...]
    regression_bias: float
    ordinal_weights: tuple[tuple[float, ...], ...]
    ordinal_bias: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ValuePredictionV1:
    """One deterministic search-facing value prediction."""

    remaining_witness_steps: float
    ordinal_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not _is_finite_real(self.remaining_witness_steps):
            raise ValueError("remaining_witness_steps prediction must be a finite real number")
        if float(self.remaining_witness_steps) < 0:
            raise ValueError("remaining_witness_steps prediction must be non-negative")
        if not isinstance(self.ordinal_probabilities, tuple) or not self.ordinal_probabilities:
            raise ValueError("ordinal probabilities must be a non-empty tuple")
        if any(
            not _is_finite_real(probability) or not 0.0 <= float(probability) <= 1.0
            for probability in self.ordinal_probabilities
        ):
            raise ValueError("ordinal probabilities must lie in [0, 1]")
        if any(
            left < right
            for left, right in zip(
                self.ordinal_probabilities,
                self.ordinal_probabilities[1:],
                strict=False,
            )
        ):
            raise ValueError("cumulative ordinal probabilities must be non-increasing")


class GoalConditionedValueHead:
    """Order-sensitive regression/ordinal head that reuses, but never copies, an encoder."""

    def __init__(
        self,
        encoder: SharedGraphEncoder[StateT],
        config: ValueHeadConfig,
        *,
        seed: int,
        parameters: ValueHeadParameters | None = None,
    ) -> None:
        if not isinstance(config, ValueHeadConfig):
            raise ValueError("config must be a ValueHeadConfig")
        if type(seed) is not int:
            raise ValueError("value-head seed must be an exact integer")
        if (
            type(encoder.embedding_dimension) is not int
            or encoder.embedding_dimension != config.embedding_dimension
        ):
            raise ValueError(
                "shared encoder embedding dimension does not match the value-head config"
            )
        _require_sha256(encoder.encoder_digest, label="shared encoder digest")
        if encoder.encoder_digest != config.shared_encoder_digest:
            raise ValueError("shared encoder digest does not match the value-head config")
        if (
            isinstance(encoder.trainable_parameter_count, bool)
            or not isinstance(encoder.trainable_parameter_count, int)
            or encoder.trainable_parameter_count < 0
        ):
            raise ValueError("shared encoder parameter count must be a non-negative integer")
        self._encoder = encoder
        self._config = config
        if parameters is not None and not isinstance(parameters, ValueHeadParameters):
            raise ValueError("parameters must be ValueHeadParameters or None")
        self._parameters = parameters or initialize_parameters(config, seed=seed)
        self._validate_parameter_shapes()
        if self.trainable_parameter_count > config.maximum_trainable_parameters:
            raise ValueError(
                "value head exceeds maximum_trainable_parameters: "
                f"{self.trainable_parameter_count} > {config.maximum_trainable_parameters}"
            )
        if self.total_trainable_parameter_count > config.maximum_total_trainable_parameters:
            raise ValueError(
                "shared encoder plus value head exceeds maximum_total_trainable_parameters: "
                f"{self.total_trainable_parameter_count} > "
                f"{config.maximum_total_trainable_parameters}"
            )

    @property
    def config(self) -> ValueHeadConfig:
        return self._config

    @property
    def parameters(self) -> ValueHeadParameters:
        return self._parameters

    @property
    def shared_encoder_digest(self) -> str:
        return self._encoder.encoder_digest

    @property
    def trainable_parameter_count(self) -> int:
        input_dimension = 4 * self._config.embedding_dimension
        return (
            input_dimension * self._config.hidden_dimension
            + self._config.hidden_dimension
            + self._config.hidden_dimension
            + 1
            + self._config.hidden_dimension * self._config.ordinal_thresholds
            + self._config.ordinal_thresholds
        )

    @property
    def total_trainable_parameter_count(self) -> int:
        return self._encoder.trainable_parameter_count + self.trainable_parameter_count

    def predict_batch(
        self,
        current_states: Sequence[StateT],
        goal_states: Sequence[StateT],
    ) -> tuple[ValuePredictionV1, ...]:
        """Return deterministic evaluation-mode predictions for ordered pairs."""

        if len(current_states) != len(goal_states):
            raise ValueError("current_states and goal_states must have equal length")
        if not current_states:
            return ()
        current = _embedding_matrix(
            self._encoder.encode_batch(current_states, training=False),
            rows=len(current_states),
            columns=self._config.embedding_dimension,
            label="current",
        )
        goal = _embedding_matrix(
            self._encoder.encode_batch(goal_states, training=False),
            rows=len(goal_states),
            columns=self._config.embedding_dimension,
            label="goal",
        )
        # This is intentionally directional: swapping current and goal swaps the
        # first two blocks and negates the third block.
        features = np.concatenate((current, goal, current - goal, current * goal), axis=1)
        hidden_weights = np.asarray(self._parameters.hidden_weights, dtype=np.float64)
        hidden_bias = np.asarray(self._parameters.hidden_bias, dtype=np.float64)
        hidden = np.tanh(features @ hidden_weights + hidden_bias)

        regression_weights = np.asarray(
            self._parameters.regression_weights,
            dtype=np.float64,
        )
        raw_regression = hidden @ regression_weights + self._parameters.regression_bias
        regression = np.logaddexp(0.0, raw_regression)

        ordinal_weights = np.asarray(self._parameters.ordinal_weights, dtype=np.float64)
        ordinal_bias = np.asarray(self._parameters.ordinal_bias, dtype=np.float64)
        ordinal_logits = hidden @ ordinal_weights + ordinal_bias
        # Project independent threshold logits onto the cumulative ordinal cone:
        # P(Y > k) must be non-increasing as k grows.
        ordinal_probabilities = np.minimum.accumulate(_sigmoid(ordinal_logits), axis=1)

        return tuple(
            ValuePredictionV1(
                remaining_witness_steps=float(regression[row]),
                ordinal_probabilities=tuple(float(value) for value in ordinal_probabilities[row]),
            )
            for row in range(len(current_states))
        )

    def score_batch(
        self,
        current_states: Sequence[StateT],
        goal_states: Sequence[StateT],
    ) -> tuple[float, ...]:
        """Expose lower-is-better deterministic scores to the common search frontier."""

        return tuple(
            prediction.remaining_witness_steps
            for prediction in self.predict_batch(current_states, goal_states)
        )

    def score_search_batch(
        self,
        current_states: Sequence[StateT],
        goal_state: StateT,
    ) -> tuple[float, ...]:
        """Match the Goal 8 frontier's ``(states, one goal)`` value-scorer boundary."""

        return self.score_batch(current_states, [goal_state] * len(current_states))

    def _validate_parameter_shapes(self) -> None:
        hidden_input = 4 * self._config.embedding_dimension
        hidden = self._config.hidden_dimension
        ordinal = self._config.ordinal_thresholds
        expected = {
            "hidden_weights": (hidden_input, hidden),
            "hidden_bias": (hidden,),
            "regression_weights": (hidden,),
            "ordinal_weights": (hidden, ordinal),
            "ordinal_bias": (ordinal,),
        }
        actual = {
            "hidden_weights": np.asarray(self._parameters.hidden_weights).shape,
            "hidden_bias": np.asarray(self._parameters.hidden_bias).shape,
            "regression_weights": np.asarray(self._parameters.regression_weights).shape,
            "ordinal_weights": np.asarray(self._parameters.ordinal_weights).shape,
            "ordinal_bias": np.asarray(self._parameters.ordinal_bias).shape,
        }
        mismatches = [
            f"{name}: expected {expected[name]}, got {shape}"
            for name, shape in actual.items()
            if shape != expected[name]
        ]
        if mismatches:
            raise ValueError("invalid value-head parameter shapes: " + "; ".join(mismatches))
        arrays = (
            self._parameters.hidden_weights,
            self._parameters.hidden_bias,
            self._parameters.regression_weights,
            (self._parameters.regression_bias,),
            self._parameters.ordinal_weights,
            self._parameters.ordinal_bias,
        )
        for values in arrays:
            raw = np.asarray(values, dtype=object)
            if any(not _is_finite_real(value) for value in raw.flat):
                raise ValueError("value-head parameters must all be finite real numbers")


def initialize_parameters(config: ValueHeadConfig, *, seed: int) -> ValueHeadParameters:
    """Initialize compact-head parameters from stable cryptographic bytes.

    This avoids Python ``hash()`` and makes fixture initialization independent of
    process hash randomization.
    """

    if not isinstance(config, ValueHeadConfig):
        raise ValueError("config must be a ValueHeadConfig")
    if type(seed) is not int:
        raise ValueError("value-head seed must be an exact integer")
    hidden_input = 4 * config.embedding_dimension
    hidden_scale = math.sqrt(6.0 / (hidden_input + config.hidden_dimension))
    output_scale = math.sqrt(6.0 / (config.hidden_dimension + 1))
    ordinal_scale = math.sqrt(6.0 / (config.hidden_dimension + config.ordinal_thresholds))
    hidden_weights = _stable_matrix(
        rows=hidden_input,
        columns=config.hidden_dimension,
        seed=seed,
        name="hidden_weights",
        scale=hidden_scale,
    )
    regression_weights = _stable_vector(
        length=config.hidden_dimension,
        seed=seed,
        name="regression_weights",
        scale=output_scale,
    )
    ordinal_weights = _stable_matrix(
        rows=config.hidden_dimension,
        columns=config.ordinal_thresholds,
        seed=seed,
        name="ordinal_weights",
        scale=ordinal_scale,
    )
    return ValueHeadParameters(
        hidden_weights=hidden_weights,
        hidden_bias=(0.0,) * config.hidden_dimension,
        regression_weights=regression_weights,
        regression_bias=0.0,
        ordinal_weights=ordinal_weights,
        ordinal_bias=(0.0,) * config.ordinal_thresholds,
    )


def _embedding_matrix(
    values: Sequence[Sequence[float]] | npt.ArrayLike,
    *,
    rows: int,
    columns: int,
    label: str,
) -> FloatArray:
    raw = np.asarray(values, dtype=object)
    if raw.shape != (rows, columns):
        raise ValueError(
            f"{label} encoder output must have shape {(rows, columns)}, got {raw.shape}"
        )
    if any(not _is_finite_real(value) for value in raw.flat):
        raise ValueError(f"{label} encoder output contains a non-finite or non-real value")
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} encoder output contains non-finite values")
    return matrix


def _sigmoid(values: FloatArray) -> FloatArray:
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _stable_matrix(
    *,
    rows: int,
    columns: int,
    seed: int,
    name: str,
    scale: float,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            _stable_uniform(seed=seed, name=name, index=row * columns + column) * scale
            for column in range(columns)
        )
        for row in range(rows)
    )


def _stable_vector(
    *,
    length: int,
    seed: int,
    name: str,
    scale: float,
) -> tuple[float, ...]:
    return tuple(
        _stable_uniform(seed=seed, name=name, index=index) * scale for index in range(length)
    )


def _stable_uniform(*, seed: int, name: str, index: int) -> float:
    payload = f"geml-value-head-v1\0{seed}\0{name}\0{index}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    unit_interval = integer / float(2**64)
    return 2.0 * unit_interval - 1.0


def _require_sha256(value: str, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _is_finite_real(value: object) -> bool:
    return bool(
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


__all__ = [
    "GoalConditionedValueHead",
    "SharedGraphEncoder",
    "ValueHeadConfig",
    "ValueHeadParameters",
    "ValuePredictionV1",
    "initialize_parameters",
]
