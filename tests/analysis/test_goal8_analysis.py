"""Goal 8 gate keeps planning rows and external references out of performance claims."""

from __future__ import annotations

from geml.analysis.goal8.summary import GateVerdict, summarize_rows
from geml.plots.goal8 import build_plot_data


def test_goal8_summary_stays_insufficient_with_pending_rows() -> None:
    summary = summarize_rows(({"status": "pending", "solved": False, "verifier_replayed": False},))
    assert summary.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
    assert build_plot_data(summary)["attempted"] == 1
