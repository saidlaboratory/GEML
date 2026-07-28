"""Predeclared deterministic tiering for the held-out Goal 8 proof benchmark."""

from __future__ import annotations

from enum import StrEnum


class ProofTier(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG_OOD = "long_ood"


def tier_for_remaining_distance(distance: int, *, short_max: int, medium_max: int) -> ProofTier:
    if distance < 0 or short_max < 0 or medium_max < short_max:
        raise ValueError("proof tier thresholds must be nonnegative and ordered")
    if distance <= short_max:
        return ProofTier.SHORT
    if distance <= medium_max:
        return ProofTier.MEDIUM
    return ProofTier.LONG_OOD
