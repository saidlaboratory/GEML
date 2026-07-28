"""Goal 9 summaries preserve numeric-only fits separately from exact recovery."""

from __future__ import annotations

from geml.analysis.goal9.summary import GateVerdict, summarize_rows


def test_goal9_summary_keeps_numeric_only_separate() -> None:
    summary = summarize_rows(
        ({"exact_recovery_status": "numeric_only", "timeout": False, "error": None},)
    )
    assert summary.verifier_exact == 0
    assert summary.numeric_only == 1
    assert summary.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
