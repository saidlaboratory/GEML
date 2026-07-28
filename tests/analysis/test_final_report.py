"""Final report output explicitly retains unavailable evidence."""

from __future__ import annotations

from geml.analysis.final.report import render_final_report


def test_final_report_labels_unstaged_archive_and_gate_status() -> None:
    report = render_final_report({"G6": "insufficient_evidence"}, archive_staged=False)
    assert "archive staged locally: `False`" in report
    assert "G6: `insufficient_evidence`" in report
