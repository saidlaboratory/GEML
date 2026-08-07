"""Predeclared, result-independent tiers for the Goal 8 proof benchmark."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

_PositiveInt = Annotated[StrictInt, Field(ge=1)]


class WitnessLengthTier(StrEnum):
    """Known replayable-witness length buckets."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    LENGTH_OOD = "length_ood"


class RuleDiversityTier(StrEnum):
    """Buckets for the number of distinct rules in a known witness."""

    SINGLE = "single"
    MODERATE = "moderate"
    HIGH = "high"


class DifficultyTier(StrEnum):
    """Static difficulty buckets computed before any benchmark search."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class OODTier(StrEnum):
    """Length/family OOD status, separate from Goal 1's combined OOD split."""

    LENGTH_FAMILY_IN_DISTRIBUTION = "length_family_in_distribution"
    LENGTH_OOD = "length_ood"
    FAMILY_OOD = "family_ood"
    LENGTH_AND_FAMILY_OOD = "length_and_family_ood"


class TierPolicyV1(BaseModel):
    """Frozen boundaries used to assign every benchmark candidate.

    ``in_distribution_witness_max`` must equal the largest witnessed length made
    available to model development. Consequently, the length-OOD bucket begins
    at exactly one greater than that observed training/validation support.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    short_witness_max: _PositiveInt
    medium_witness_max: _PositiveInt
    in_distribution_witness_max: _PositiveInt
    moderate_rule_diversity_max: _PositiveInt
    easy_difficulty_max: _PositiveInt
    medium_difficulty_max: _PositiveInt
    state_size_divisor: _PositiveInt

    @model_validator(mode="after")
    def validate_boundaries(self) -> Self:
        """Require nonoverlapping ordered buckets."""
        if not (
            self.short_witness_max < self.medium_witness_max < self.in_distribution_witness_max
        ):
            raise ValueError("witness boundaries must satisfy short < medium < in_distribution")
        if self.moderate_rule_diversity_max < 2:
            raise ValueError("moderate_rule_diversity_max must be at least two")
        if self.easy_difficulty_max >= self.medium_difficulty_max:
            raise ValueError("difficulty boundaries must satisfy easy < medium")
        return self


class TierAssignmentV1(BaseModel):
    """Complete deterministic assignment for one candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    witness_length: _PositiveInt
    rule_diversity: _PositiveInt
    maximum_state_size: _PositiveInt
    difficulty_score: _PositiveInt
    witness_length_tier: WitnessLengthTier
    rule_diversity_tier: RuleDiversityTier
    difficulty_tier: DifficultyTier
    ood_tier: OODTier

    @model_validator(mode="after")
    def validate_trace_invariants(self) -> Self:
        if self.rule_diversity > self.witness_length:
            raise ValueError("rule_diversity cannot exceed witness_length")
        return self


def tier_combination_is_feasible(
    *,
    witness_length_tier: WitnessLengthTier,
    rule_diversity_tier: RuleDiversityTier,
    difficulty_tier: DifficultyTier,
    policy: TierPolicyV1,
) -> bool:
    """Return whether at least one concrete trace can inhabit three static tiers.

    A witness with ``n`` actions can contain at most ``n`` distinct rules.
    Maximum state size is otherwise unbounded, so it can raise the difficulty
    score to any value at or above the combination's minimum score.
    """

    witness_bounds = {
        WitnessLengthTier.SHORT: (1, policy.short_witness_max),
        WitnessLengthTier.MEDIUM: (
            policy.short_witness_max + 1,
            policy.medium_witness_max,
        ),
        WitnessLengthTier.LONG: (
            policy.medium_witness_max + 1,
            policy.in_distribution_witness_max,
        ),
        WitnessLengthTier.LENGTH_OOD: (
            policy.in_distribution_witness_max + 1,
            None,
        ),
    }
    minimum_rule_diversity = {
        RuleDiversityTier.SINGLE: 1,
        RuleDiversityTier.MODERATE: 2,
        RuleDiversityTier.HIGH: policy.moderate_rule_diversity_max + 1,
    }[rule_diversity_tier]
    minimum_witness, maximum_witness = witness_bounds[witness_length_tier]
    minimum_witness = max(minimum_witness, minimum_rule_diversity)
    if maximum_witness is not None and minimum_witness > maximum_witness:
        return False

    minimum_score = minimum_witness + 2 * (minimum_rule_diversity - 1) + 1
    if difficulty_tier is DifficultyTier.EASY:
        return minimum_score <= policy.easy_difficulty_max
    if difficulty_tier is DifficultyTier.MEDIUM:
        return minimum_score <= policy.medium_difficulty_max
    return True


def assign_tiers(
    *,
    witness_length: int,
    rule_diversity: int,
    maximum_state_size: int,
    family_is_held_out: bool,
    policy: TierPolicyV1,
) -> TierAssignmentV1:
    """Assign static tiers without using search outcomes.

    Difficulty is a preregistered structural score:

    ``witness_length + 2 * (rule_diversity - 1) + ceil(max_state_size / divisor)``.

    It uses only the known replayable witness and source-state structure. Search
    success, expansions, runtime, and model scores cannot affect the tier.
    """

    for name, value in (
        ("witness_length", witness_length),
        ("rule_diversity", rule_diversity),
        ("maximum_state_size", maximum_state_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if rule_diversity > witness_length:
        raise ValueError("rule_diversity cannot exceed witness_length")
    if not isinstance(family_is_held_out, bool):
        raise ValueError("family_is_held_out must be a boolean")

    if witness_length <= policy.short_witness_max:
        witness_tier = WitnessLengthTier.SHORT
    elif witness_length <= policy.medium_witness_max:
        witness_tier = WitnessLengthTier.MEDIUM
    elif witness_length <= policy.in_distribution_witness_max:
        witness_tier = WitnessLengthTier.LONG
    else:
        witness_tier = WitnessLengthTier.LENGTH_OOD

    if rule_diversity == 1:
        diversity_tier = RuleDiversityTier.SINGLE
    elif rule_diversity <= policy.moderate_rule_diversity_max:
        diversity_tier = RuleDiversityTier.MODERATE
    else:
        diversity_tier = RuleDiversityTier.HIGH

    difficulty_score = (
        witness_length
        + 2 * (rule_diversity - 1)
        + (maximum_state_size + policy.state_size_divisor - 1) // policy.state_size_divisor
    )
    if difficulty_score <= policy.easy_difficulty_max:
        difficulty_tier = DifficultyTier.EASY
    elif difficulty_score <= policy.medium_difficulty_max:
        difficulty_tier = DifficultyTier.MEDIUM
    else:
        difficulty_tier = DifficultyTier.HARD

    is_length_ood = witness_tier is WitnessLengthTier.LENGTH_OOD
    if is_length_ood and family_is_held_out:
        ood_tier = OODTier.LENGTH_AND_FAMILY_OOD
    elif is_length_ood:
        ood_tier = OODTier.LENGTH_OOD
    elif family_is_held_out:
        ood_tier = OODTier.FAMILY_OOD
    else:
        ood_tier = OODTier.LENGTH_FAMILY_IN_DISTRIBUTION

    return TierAssignmentV1(
        witness_length=witness_length,
        rule_diversity=rule_diversity,
        maximum_state_size=maximum_state_size,
        difficulty_score=difficulty_score,
        witness_length_tier=witness_tier,
        rule_diversity_tier=diversity_tier,
        difficulty_tier=difficulty_tier,
        ood_tier=ood_tier,
    )
