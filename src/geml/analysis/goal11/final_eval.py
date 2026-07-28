"""Cross-track Gate G11 aggregation that keeps incomplete gates visible."""

from __future__ import annotations

from dataclasses import dataclass

_CONTROLLED_TRACKS = frozenset({"equivalence", "proof", "sr"})


@dataclass(frozen=True, slots=True)
class GateSynthesisV1:
    gate_statuses: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.gate_statuses)
        if len(set(names)) != len(names):
            raise ValueError("each controlled track may have only one gate status")
        unknown = sorted(set(names) - _CONTROLLED_TRACKS)
        if unknown:
            raise ValueError(f"unexpected controlled tracks: {unknown}")

    @property
    def verdict(self) -> str:
        observed = {name for name, _ in self.gate_statuses}
        incomplete = observed != _CONTROLLED_TRACKS or any(
            status != "pass" for _, status in self.gate_statuses
        )
        return "insufficient_evidence" if incomplete else "pass"
