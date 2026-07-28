"""Goal 11 fixed-scale status is not a data-scaling claim."""

from __future__ import annotations

import pytest

from geml.analysis.goal11.scaling import (
    FixedScaleEfficiencyV1,
    FixedScalePointV1,
)
from geml.experiments.goal11.run_scaling import fixed_scale_status, validate_fixed_scale_results


def test_fixed_scale_status_requires_all_recorded_efficiency_fields() -> None:
    incomplete = FixedScaleEfficiencyV1(1, 2, None, 4)
    complete = FixedScaleEfficiencyV1(1, 2, 0.5, 4)

    assert incomplete.complete is False
    assert fixed_scale_status(incomplete) == "insufficient_evidence"
    assert complete.complete is True
    assert fixed_scale_status(complete) == "complete"


def test_fixed_scale_join_rejects_mixed_controlled_budgets() -> None:
    first = FixedScalePointV1(
        "equivalence",
        "ast",
        1,
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
        "complete",
        0.5,
        1.0,
        FixedScaleEfficiencyV1(1, 2, 0.5, 4),
    )
    incompatible = FixedScalePointV1(
        "equivalence",
        "eml",
        1,
        "sha256:" + "a" * 64,
        "sha256:" + "c" * 64,
        "complete",
        0.5,
        1.0,
        FixedScaleEfficiencyV1(1, 2, 0.5, 4),
    )

    with pytest.raises(ValueError, match="incompatible"):
        validate_fixed_scale_results((first, incompatible))


def test_fixed_scale_points_reject_noncanonical_status_or_checksum() -> None:
    with pytest.raises(ValueError, match="status"):
        FixedScalePointV1(
            "proof",
            "policy",
            None,
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
            "finished",
            None,
            None,
            FixedScaleEfficiencyV1(None, None, None, None),
        )
    with pytest.raises(ValueError, match="digests"):
        FixedScalePointV1(
            "proof",
            "policy",
            None,
            "sha256:short",
            "sha256:" + "b" * 64,
            "pending",
            None,
            None,
            FixedScaleEfficiencyV1(None, None, None, None),
        )
