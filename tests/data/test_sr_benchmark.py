"""Core SR benchmark invariants that prevent numeric fits becoming exact-recovery claims."""

from __future__ import annotations

from geml.data.sr.benchmark import ExactRecoveryStatus, exact_recovery_claim


def test_numeric_fit_is_not_exact_recovery_without_a_verifier() -> None:
    assert not exact_recovery_claim(ExactRecoveryStatus.NUMERIC_ONLY, numeric_error=0.0)
    assert not exact_recovery_claim(ExactRecoveryStatus.VERIFIER_UNAVAILABLE, numeric_error=0.0)
    assert exact_recovery_claim(ExactRecoveryStatus.VERIFIED, numeric_error=0.0)
