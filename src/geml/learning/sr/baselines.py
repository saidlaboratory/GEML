"""Explicit optional PySR/GP identity and compact transformer-SR compatibility contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from geml.learning.sr.guided_search import SRBudgetV1


class BaselineKind(StrEnum):
    PYSR = "pysr"
    GP_FALLBACK = "gp_fallback"
    PREFIX_TRANSFORMER = "prefix_transformer"


@dataclass(frozen=True, slots=True)
class BaselineIdentityV1:
    kind: BaselineKind
    version: str | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        pysr_unavailable = self.kind is BaselineKind.PYSR and self.version is None
        if pysr_unavailable and self.unavailable_reason is None:
            raise ValueError("unavailable PySR must retain an explicit reason")
        if self.kind is BaselineKind.GP_FALLBACK and not self.unavailable_reason:
            raise ValueError("GP fallback must be labeled with its PySR unavailability reason")


def compatible_budget(baseline_budget: SRBudgetV1, guided_budget: SRBudgetV1) -> bool:
    """Require all declared controlled dimensions to match exactly."""

    return baseline_budget == guided_budget
