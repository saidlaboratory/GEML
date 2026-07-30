"""CPU fixture coverage for the H100 multi-GPU cell scheduler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.h100.schedule_cells import load_cells, main

# Writes a goal6-shaped row file; "flaky" mode fails every seed-2 cell so the
# retained-failure path is exercised without a real training harness.
_FAKE_RUNNER = """\
import json, os, pathlib, sys
row = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
status = "failed" if mode == "flaky" and "seed-2" in row.name else "complete"
row.parent.mkdir(parents=True, exist_ok=True)
row.write_text(json.dumps({"status": status, "device": os.environ["CUDA_VISIBLE_DEVICES"]}))
"""


def _grid(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "grid"
    root.mkdir()
    manifest = root / "grid.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "geml-goal6-grid-v1",
                "arms": [{"arm_id": "graph::ast_dag"}, {"arm_id": "trivial_floor"}],
                "seeds": [1, 2],
            }
        )
    )
    return root, manifest


def _cmd(tmp_path: Path, root: Path, mode: str) -> str:
    runner = tmp_path / "fake_cell.py"
    runner.write_text(_FAKE_RUNNER)
    return f"{sys.executable} {runner} {root}/cells/{{cell_id}}.json {mode}"


def test_schedules_all_cells_across_two_fake_devices(tmp_path, capsys):
    root, manifest = _grid(tmp_path)
    code = main(
        ["--manifest", str(manifest), "--gpus", "0,1", "--cmd", _cmd(tmp_path, root, "complete")]
    )
    assert code == 0

    cells = load_cells(manifest, root)
    assert len(cells) == 4
    rows = [json.loads(cell.row_path.read_text()) for cell in cells]
    assert all(row["status"] == "complete" for row in rows)
    assert {row["device"] for row in rows} == {"0", "1"}

    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["newly_complete"] == 4
    assert summary["grid_complete"] is True

    log_lines = (root / "scheduler.log.jsonl").read_text().splitlines()
    events = [json.loads(line)["event"] for line in log_lines]
    assert events.count("cell_started") == 4
    assert events.count("cell_finished") == 4

    # Resume: a second run finds every row complete and launches nothing.
    code = main(
        ["--manifest", str(manifest), "--gpus", "0,1", "--cmd", _cmd(tmp_path, root, "complete")]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["already_complete"] == 4
    assert summary["newly_complete"] == 0


def test_retains_failures_and_requeues_only_with_retry_flag(tmp_path, capsys):
    root, manifest = _grid(tmp_path)
    code = main(
        ["--manifest", str(manifest), "--gpus", "0,1", "--cmd", _cmd(tmp_path, root, "flaky")]
    )
    assert code == 1
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["newly_complete"] == 2
    assert set(summary["failed"].values()) == {"retained_failure"}

    # Without --retry-failed the retained rows stay untouched evidence.
    code = main(
        ["--manifest", str(manifest), "--gpus", "0,1", "--cmd", _cmd(tmp_path, root, "complete")]
    )
    assert code == 1
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["retained_failures_skipped"] == 2
    assert summary["newly_complete"] == 0

    code = main(
        [
            "--manifest",
            str(manifest),
            "--gpus",
            "0,1",
            "--cmd",
            _cmd(tmp_path, root, "complete"),
            "--retry-failed",
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["grid_complete"] is True


def test_crash_without_row_is_reported_not_invented(tmp_path, capsys):
    root, manifest = _grid(tmp_path)
    code = main(["--manifest", str(manifest), "--gpus", "0,1", "--cmd", "exit 3"])
    assert code == 1
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert set(summary["failed"].values()) == {"no_row"}
    assert not (root / "cells").exists()


def test_timeout_kills_the_cell_and_shell_tokens_pass_through(tmp_path, capsys):
    root, manifest = _grid(tmp_path)
    # ${UNSET:-tick} would crash str.format; it must reach the shell intact.
    cmd = 'echo "${SCHED_UNSET:-tick} {cell_id}" && sleep 60'
    code = main(["--manifest", str(manifest), "--gpus", "0,1", "--cmd", cmd, "--cell-timeout", "1"])
    assert code == 1
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert len(summary["failed"]) == 4
    assert set(summary["failed"].values()) == {"timeout"}

    worker_logs = sorted((root / "scheduler").glob("*.log"))
    assert len(worker_logs) == 4
    for log_path in worker_logs:
        assert log_path.read_text() == f"tick {log_path.stem}\n"
