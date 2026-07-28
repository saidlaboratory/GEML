"""Final-report renderer that states absent production evidence plainly."""

from __future__ import annotations

from collections.abc import Mapping


def render_final_report(gate_statuses: Mapping[str, str], *, archive_staged: bool) -> str:
    lines = ["# GEML Goals 1-12 final report", "", "## Evidence status", ""]
    for goal, status in sorted(gate_statuses.items()):
        lines.append(f"- {goal}: `{status}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Goals 6-12 production claims require frozen, checksum-authenticated result manifests.",
            f"Authenticated external archive staged locally: `{archive_staged}`.",
            "External LLM rows are reference-only and excluded from controlled gates.",
        ]
    )
    return "\n".join(lines) + "\n"
