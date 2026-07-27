"""Hard-negative evidence rules for the Goal 6 equivalence dataset."""

from __future__ import annotations

from dataclasses import dataclass

from geml.data.pairs.generate import (
    NonEquivalenceEvidenceV1,
    VerificationTier,
    canonical_json_bytes,
    sha256_digest,
)


@dataclass(frozen=True, slots=True)
class NegativeCandidate:
    """A size-matched near-miss before it is admitted as a labeled negative."""

    left_signature: str
    right_signature: str
    left_size: int
    right_size: int
    operator_family: str

    def __post_init__(self) -> None:
        if not self.left_signature or not self.right_signature:
            raise ValueError("negative-candidate signatures must be nonblank")
        if self.left_size < 0 or self.right_size < 0:
            raise ValueError("negative-candidate sizes must be nonnegative")
        if not self.operator_family.strip():
            raise ValueError("negative-candidate operator_family must be nonblank")


def within_size_tolerance(candidate: NegativeCandidate, *, absolute_tolerance: int) -> bool:
    """Return whether a candidate meets the predeclared absolute size tolerance."""

    if type(absolute_tolerance) is not int or absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be a nonnegative integer")
    return abs(candidate.left_size - candidate.right_size) <= absolute_tolerance


def reject_structural_match(candidate: NegativeCandidate) -> bool:
    """Reject a candidate whose exact structural signatures already match."""

    return candidate.left_signature == candidate.right_signature


def formal_counterexample_evidence(
    *,
    method: str,
    detail: str,
    witness: object,
) -> NonEquivalenceEvidenceV1:
    """Create accepted-negative evidence only when the supplied witness is formal/rigorous."""

    payload = {
        "detail": detail,
        "method": method,
        "tier": VerificationTier.FORMAL_COUNTEREXAMPLE.value,
        "witness": witness,
    }
    return NonEquivalenceEvidenceV1(
        tier=VerificationTier.FORMAL_COUNTEREXAMPLE,
        evidence_digest=sha256_digest(canonical_json_bytes(payload)),
        method=method,
        detail=detail,
        rigorous=True,
    )


def numerical_disagreement_evidence(
    *,
    method: str,
    detail: str,
    samples: object,
) -> NonEquivalenceEvidenceV1:
    """Record a high-precision disagreement without incorrectly promoting it to a negative label."""

    payload = {
        "detail": detail,
        "method": method,
        "samples": samples,
        "tier": VerificationTier.NUMERIC_COUNTEREXAMPLE.value,
    }
    return NonEquivalenceEvidenceV1(
        tier=VerificationTier.NUMERIC_COUNTEREXAMPLE,
        evidence_digest=sha256_digest(canonical_json_bytes(payload)),
        method=method,
        detail=detail,
        rigorous=False,
    )
