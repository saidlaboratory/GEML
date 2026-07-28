"""SR baseline identities never silently relabel a GP fallback as PySR."""

from __future__ import annotations

import pytest

from geml.learning.sr.baselines import BaselineIdentityV1, BaselineKind, compatible_budget
from geml.learning.sr.guided_search import SRBudgetV1


def test_baseline_identity_and_budget_matching_are_explicit() -> None:
    budget = SRBudgetV1(1.0, 2, 3, 4)
    assert compatible_budget(budget, budget)
    fallback = BaselineIdentityV1(BaselineKind.GP_FALLBACK, None, "PySR unavailable")
    assert fallback.kind is BaselineKind.GP_FALLBACK
    with pytest.raises(ValueError, match="unavailable PySR"):
        BaselineIdentityV1(BaselineKind.PYSR, None)
