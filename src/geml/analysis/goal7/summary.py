"""Conservative Gate G7 evaluation for fixed policy-grid manifests."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from geml.experiments.goal7.run_grid import GRID_SCHEMA_VERSION


class GateVerdict(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class Goal7Summary:
    status_counts: tuple[tuple[str, int], ...]
    verdict: GateVerdict
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status_counts": dict(self.status_counts),
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
        }


def summarize_manifest(manifest: dict[str, object]) -> Goal7Summary:
    if manifest.get("schema_version") != GRID_SCHEMA_VERSION:
        raise ValueError("unexpected Goal 7 grid schema")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 18:
        raise ValueError("Goal 7 requires its complete 18-cell fixed plan")
    counts = Counter(str(row["status"]) for row in cells if isinstance(row, dict))
    reasons = []
    if counts["pending"]:
        reasons.append("planned cells have not produced authenticated replay metrics")
    if counts["unsupported"]:
        reasons.append("motif-AST fair-control cells are explicitly unsupported")
    if counts["failed"] or counts["timeout"]:
        reasons.append("failed or timeout cells require denominator-complete reporting")
    reasons.append("no fixed-scale comparison can be claimed from phase-A planning rows")
    return Goal7Summary(
        tuple(sorted(counts.items())),
        GateVerdict.INSUFFICIENT_EVIDENCE,
        tuple(reasons),
    )


def render_summary_markdown(summary: Goal7Summary) -> str:
    return (
        "# Goal 7 rewrite-policy summary\n\n**INSUFFICIENT EVIDENCE.** "
        + "; ".join(summary.reasons)
        + ".\n"
    )
