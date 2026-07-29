"""Hard-negative evidence rules for the Goal 6 equivalence dataset."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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


class NegativeDisposition(StrEnum):
    """The retained decision for one hard-negative candidate."""

    ACCEPTED = "accepted"
    UNRESOLVED = "unresolved"
    REJECTED_STRUCTURAL_MATCH = "rejected_structural_match"
    REJECTED_SIZE_MISMATCH = "rejected_size_mismatch"


@dataclass(frozen=True, slots=True)
class NegativeAssessment:
    """A transparent hard-negative decision that never upgrades weak evidence."""

    candidate: NegativeCandidate
    disposition: NegativeDisposition
    evidence: NonEquivalenceEvidenceV1 | None
    detail: str


def within_size_tolerance(candidate: NegativeCandidate, *, absolute_tolerance: int) -> bool:
    """Return whether a candidate meets the predeclared absolute size tolerance."""

    if type(absolute_tolerance) is not int or absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be a nonnegative integer")
    return abs(candidate.left_size - candidate.right_size) <= absolute_tolerance


def reject_structural_match(candidate: NegativeCandidate) -> bool:
    """Reject a candidate whose exact structural signatures already match."""

    return candidate.left_signature == candidate.right_signature


def assess_hard_negative(
    candidate: NegativeCandidate,
    *,
    absolute_size_tolerance: int,
    evidence: NonEquivalenceEvidenceV1 | None,
) -> NegativeAssessment:
    """Apply the predeclared near-miss/same-family admission policy transparently.

    The candidate's family is retained by `NegativeCandidate`; callers may use
    it for quota accounting but cannot bypass the structural, size, and formal
    evidence gates. Numeric disagreement remains an unresolved ledger row.
    """

    if reject_structural_match(candidate):
        return NegativeAssessment(
            candidate=candidate,
            disposition=NegativeDisposition.REJECTED_STRUCTURAL_MATCH,
            evidence=evidence,
            detail="candidate endpoints have the same structural signature",
        )
    if not within_size_tolerance(candidate, absolute_tolerance=absolute_size_tolerance):
        return NegativeAssessment(
            candidate=candidate,
            disposition=NegativeDisposition.REJECTED_SIZE_MISMATCH,
            evidence=evidence,
            detail="candidate violates the predeclared absolute size tolerance",
        )
    if evidence is None:
        return NegativeAssessment(
            candidate=candidate,
            disposition=NegativeDisposition.UNRESOLVED,
            evidence=None,
            detail="no non-equivalence evidence was established",
        )
    if evidence.tier is VerificationTier.FORMAL_COUNTEREXAMPLE and evidence.rigorous:
        return NegativeAssessment(
            candidate=candidate,
            disposition=NegativeDisposition.ACCEPTED,
            evidence=evidence,
            detail="accepted with rigorous formal counterexample evidence",
        )
    return NegativeAssessment(
        candidate=candidate,
        disposition=NegativeDisposition.UNRESOLVED,
        evidence=evidence,
        detail="evidence is not a rigorous formal non-equivalence certificate",
    )


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
