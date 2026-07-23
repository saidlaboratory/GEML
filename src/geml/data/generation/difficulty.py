"""Deterministic difficulty targets for source-expression generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import exp, log
from random import Random
from typing import Annotated, Never

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

PositiveInt = Annotated[StrictInt, Field(ge=1)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveWeight = Annotated[StrictFloat, Field(gt=0)]
Probability = Annotated[StrictFloat, Field(ge=0, le=1)]


class _FrozenDict[Key, Value](dict[Key, Value]):
    """A JSON-serializable dictionary that rejects in-place mutation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        if getattr(self, "_initialized", False):
            self._reject_mutation()
        dict.__init__(self, *args, **kwargs)
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, _name: str, _value: object) -> Never:
        self._reject_mutation()

    def __delattr__(self, _name: str) -> Never:
        self._reject_mutation()

    def _reject_mutation(self, *_args: object, **_kwargs: object) -> Never:
        raise TypeError("generation policy mappings are immutable")

    def __copy__(self) -> _FrozenDict[Key, Value]:
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> _FrozenDict[Key, Value]:
        return self

    def __hash__(self) -> int:
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[type[_FrozenDict], tuple[dict[Key, Value]]]:
        return (_FrozenDict, (dict(self),))

    def copy(self) -> _FrozenDict[Key, Value]:
        return self

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __ior__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation


def freeze_mapping[Key, Value](mapping: Mapping[Key, Value]) -> dict[Key, Value]:
    """Return an immutable dict subclass that Pydantic can serialize normally."""

    return _FrozenDict(mapping)


class SizeBucket(BaseModel):
    """Inclusive target-size range with a sampling weight."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    minimum: PositiveInt
    maximum: PositiveInt
    weight: PositiveWeight

    @model_validator(mode="after")
    def validate_range(self) -> SizeBucket:
        if self.maximum < self.minimum:
            raise ValueError("size bucket maximum cannot be less than minimum")
        return self


class DifficultyProfile(BaseModel):
    """Immutable target distributions for one named generator profile."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    size_buckets: tuple[SizeBucket, ...] = Field(min_length=1)
    depth_weights: Mapping[NonNegativeInt, PositiveWeight] = Field(min_length=1)
    variable_count_weights: Mapping[PositiveInt, PositiveWeight] = Field(min_length=1)
    intermediate_leaf_probability: Probability

    @model_validator(mode="after")
    def validate_profile(self) -> DifficultyProfile:
        if any(count > 6 for count in self.variable_count_weights):
            raise ValueError("variable-count policies may cover only 1 through 6 variables")
        object.__setattr__(self, "depth_weights", freeze_mapping(self.depth_weights))
        object.__setattr__(
            self,
            "variable_count_weights",
            freeze_mapping(self.variable_count_weights),
        )
        return self


class DifficultyTarget(BaseModel):
    """One explicitly sampled logical source-AST target."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    target_size: PositiveInt
    target_depth: NonNegativeInt
    variable_count: PositiveInt
    intermediate_leaf_probability: Probability

    @model_validator(mode="after")
    def validate_target(self) -> DifficultyTarget:
        if self.target_depth >= self.target_size:
            raise ValueError("target depth must be less than target size")
        if self.variable_count > 6:
            raise ValueError("variable count cannot exceed six")
        return self


def _weighted_choice[T](rng: Random, values: tuple[T, ...], weights: tuple[float, ...]) -> T:
    """Choose from a nonempty immutable sequence with local RNG state only."""

    if not values or len(values) != len(weights):
        raise ValueError("weighted choices require equally sized nonempty values and weights")
    return rng.choices(values, weights=weights, k=1)[0]


def _depth_is_shape_feasible(size: int, depth: int) -> bool:
    """Check a unary/binary tree's closed-form size/depth bounds."""

    return size >= 1 and size.bit_length() - 1 <= depth < size


def _maximum_leaf_count(size: int, depth: int) -> int:
    """Return the unary/binary leaf-capacity bound for an exact target."""

    return min((size + 1) // 2, size - depth)


def _draw_target(
    profile: DifficultyProfile,
    rng: Random,
    eligible_buckets: tuple[SizeBucket, ...],
    *,
    minimum_size: int,
) -> DifficultyTarget | None:
    """Draw once from the configured hierarchy, returning ``None`` if infeasible."""

    bucket = _weighted_choice(
        rng,
        eligible_buckets,
        tuple(bucket.weight for bucket in eligible_buckets),
    )
    lower = max(bucket.minimum, minimum_size)
    target_size = rng.randint(lower, bucket.maximum)
    feasible_depths = tuple(
        (depth, weight)
        for depth, weight in sorted(profile.depth_weights.items())
        if _depth_is_shape_feasible(target_size, depth)
    )
    if not feasible_depths:
        return None
    target_depth = _weighted_choice(
        rng,
        tuple(depth for depth, _ in feasible_depths),
        tuple(weight for _, weight in feasible_depths),
    )

    maximum_leaf_count = _maximum_leaf_count(target_size, target_depth)
    feasible_variable_counts = tuple(
        (count, weight)
        for count, weight in sorted(profile.variable_count_weights.items())
        if count <= maximum_leaf_count
    )
    if not feasible_variable_counts:
        return None
    variable_count = _weighted_choice(
        rng,
        tuple(count for count, _ in feasible_variable_counts),
        tuple(weight for _, weight in feasible_variable_counts),
    )
    return DifficultyTarget(
        target_size=target_size,
        target_depth=target_depth,
        variable_count=variable_count,
        intermediate_leaf_probability=profile.intermediate_leaf_probability,
    )


def _feasible_target_support(
    profile: DifficultyProfile,
    eligible_buckets: tuple[SizeBucket, ...],
    *,
    minimum_size: int,
    target_predicate: Callable[[DifficultyTarget], bool] | None,
) -> tuple[tuple[DifficultyTarget, float], ...]:
    """Enumerate log-weighted support as a fallback for sparse policies."""

    support: list[tuple[DifficultyTarget, float]] = []
    for bucket in eligible_buckets:
        lower = max(bucket.minimum, minimum_size)
        log_size_weight = log(bucket.weight) - log(bucket.maximum - lower + 1)
        for target_size in range(lower, bucket.maximum + 1):
            feasible_depths = tuple(
                (depth, weight)
                for depth, weight in sorted(profile.depth_weights.items())
                if _depth_is_shape_feasible(target_size, depth)
            )
            depth_weight_total = sum(weight for _, weight in feasible_depths)
            if depth_weight_total <= 0:
                continue
            for target_depth, depth_weight in feasible_depths:
                maximum_leaf_count = _maximum_leaf_count(target_size, target_depth)
                feasible_variables = tuple(
                    (count, weight)
                    for count, weight in sorted(profile.variable_count_weights.items())
                    if count <= maximum_leaf_count
                )
                variable_weight_total = sum(weight for _, weight in feasible_variables)
                if variable_weight_total <= 0:
                    continue
                for variable_count, variable_weight in feasible_variables:
                    target = DifficultyTarget(
                        target_size=target_size,
                        target_depth=target_depth,
                        variable_count=variable_count,
                        intermediate_leaf_probability=profile.intermediate_leaf_probability,
                    )
                    if target_predicate is not None and not target_predicate(target):
                        continue
                    log_weight = (
                        log_size_weight
                        + log(depth_weight)
                        - log(depth_weight_total)
                        + log(variable_weight)
                        - log(variable_weight_total)
                    )
                    support.append((target, log_weight))
    return tuple(support)


def sample_difficulty_target(
    profile: DifficultyProfile,
    rng: Random,
    *,
    minimum_size: int = 1,
    target_predicate: Callable[[DifficultyTarget], bool] | None = None,
) -> DifficultyTarget:
    """Sample size, exact depth, and feasible variable count deterministically.

    Feasibility here assumes the approved grammar has both unary and binary
    productions. ``grammar.py`` performs the final operator-specific check.
    """

    if minimum_size < 1:
        raise ValueError("minimum target size must be positive")
    eligible_buckets = tuple(
        bucket for bucket in profile.size_buckets if bucket.maximum >= minimum_size
    )
    if not eligible_buckets:
        raise ValueError(f"no size bucket can satisfy minimum target size {minimum_size}")

    for _ in range(128):
        target = _draw_target(
            profile,
            rng,
            eligible_buckets,
            minimum_size=minimum_size,
        )
        if target is not None and (target_predicate is None or target_predicate(target)):
            return target

    support = _feasible_target_support(
        profile,
        eligible_buckets,
        minimum_size=minimum_size,
        target_predicate=target_predicate,
    )
    if not support:
        raise ValueError("difficulty profile has no feasible target")
    maximum_log_weight = max(weight for _, weight in support)
    return _weighted_choice(
        rng,
        tuple(target for target, _ in support),
        tuple(exp(weight - maximum_log_weight) for _, weight in support),
    )
