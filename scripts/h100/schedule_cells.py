"""Multi-GPU cell scheduler for the Goal 6 and Goal 7 grids.

Enumerates (arm, seed) cells from a grid manifest, runs one cell per GPU via
``CUDA_VISIBLE_DEVICES``, queues the rest, and cooperates with the runners'
persisted row files:

* Goal 6 (``geml-goal6-grid-v1``): rows live at ``<root>/cells/<cell_id>.json``
  with ``cell_id = arm_id.replace('::', '__') + f'__seed-{seed}'`` -- the same
  identity ``GridRunner.existing_row`` reads. A row with status ``complete``
  is never re-run; any other retained row (``failed``, ``timed_out``, ...) is
  kept as evidence and only requeued under ``--retry-failed``, matching
  ``GridRunner.run`` which overwrites non-complete rows atomically.
* Goal 7 (``geml-goal7-grid-plan-v1``): cells come from the plan's
  ``cell_contracts`` and rows live at ``<root>/cells/<id[:2]>/<id>.json``.
  Goal 7 rows are immutable (``_atomic_create_json`` refuses to overwrite), so
  a retained failure there needs a new run identity; ``--retry-failed`` will
  requeue it but the runner itself is expected to refuse.

The scheduler never deletes or rewrites a row file itself; success/failure is
decided by re-reading the row the worker produced, not by exit code alone.
Every event is appended to a JSONL scheduler log, and each worker's output is
retained under ``<root>/scheduler/<cell_id>.log``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

POLL_SECONDS = 0.5

GOAL6_SCHEMA = "geml-goal6-grid-v1"
GOAL7_PLAN_SCHEMA = "geml-goal7-grid-plan-v1"


@dataclass(frozen=True)
class Cell:
    cell_id: str
    arm_id: str
    seed: int
    row_path: Path


def load_cells(manifest_path: Path, root: Path) -> list[Cell]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    cells_dir = root / "cells"
    if schema == GOAL6_SCHEMA:
        cells = []
        for arm in payload["arms"]:
            arm_id = str(arm["arm_id"])
            for seed in payload["seeds"]:
                cell_id = f"{arm_id.replace('::', '__')}__seed-{seed}"
                cells.append(Cell(cell_id, arm_id, int(seed), cells_dir / f"{cell_id}.json"))
        return cells
    if schema == GOAL7_PLAN_SCHEMA:
        return [
            Cell(
                str(contract["cell_id"]),
                str(contract["arm_id"]),
                int(contract["seed"]),
                cells_dir / str(contract["cell_id"])[:2] / f"{contract['cell_id']}.json",
            )
            for contract in payload["cell_contracts"]
        ]
    raise SystemExit(f"unsupported manifest schema {schema!r} in {manifest_path}")


def render_command(template: str, cell: Cell, gpu: str) -> str:
    """Substitute the four placeholders; leave every other brace untouched.

    Plain ``str.format`` would choke on shell text like ``${VAR:-x}`` or awk
    ``{print}``, and goal7 reproduction commands frozen from the configs use
    ``${GEML_ARTIFACTS_ROOT}``-style placeholders.
    """

    for key, value in (
        ("{cell_id}", cell.cell_id),
        ("{arm_id}", cell.arm_id),
        ("{seed}", str(cell.seed)),
        ("{gpu}", gpu),
    ):
        template = template.replace(key, value)
    return template


def row_status(cell: Cell) -> str | None:
    """Return the persisted row status, ``None`` when no row exists yet."""

    if not cell.row_path.is_file():
        return None
    try:
        return str(json.loads(cell.row_path.read_text(encoding="utf-8"))["status"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return "unreadable"


class SchedulerLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: object) -> None:
        row = {"ts": datetime.datetime.now(datetime.UTC).isoformat(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"[schedule] {event} {fields}", flush=True)


def run_schedule(
    cells: list[Cell],
    gpus: list[str],
    command: str,
    log: SchedulerLog,
    worker_log_dir: Path,
    cell_timeout: float | None = None,
) -> dict[str, str]:
    """Run every cell across the GPUs; return ``cell_id -> outcome``."""

    worker_log_dir.mkdir(parents=True, exist_ok=True)
    queue = list(cells)
    running: dict[str, tuple[Cell, subprocess.Popen, object, float]] = {}
    outcomes: dict[str, str] = {}

    def finish(gpu: str, returncode: int | None, timed_out: bool = False) -> None:
        cell, _, handle, started = running.pop(gpu)
        handle.close()
        status = row_status(cell)
        if timed_out:
            outcome = "timeout"
        elif status == "complete":
            outcome = "complete"
        elif status is not None:
            outcome = "retained_failure"
        else:
            outcome = "no_row"
        outcomes[cell.cell_id] = outcome
        log.write(
            "cell_finished",
            cell_id=cell.cell_id,
            arm_id=cell.arm_id,
            seed=cell.seed,
            gpu=gpu,
            returncode=returncode,
            row_status=status,
            outcome=outcome,
            wall_seconds=round(time.monotonic() - started, 3),
        )

    while queue or running:
        for gpu in gpus:
            if gpu in running or not queue:
                continue
            cell = queue.pop(0)
            rendered = render_command(command, cell, gpu)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            handle = (worker_log_dir / f"{cell.cell_id}.log").open("ab")
            # start_new_session: the worker gets its own process group so a
            # timeout kill reaches the real training process, not just the
            # wrapping shell (killing only the sh would leak a GPU-holding
            # orphan into the next cell's device).
            proc = subprocess.Popen(
                rendered,
                shell=True,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[gpu] = (cell, proc, handle, time.monotonic())
            log.write(
                "cell_started",
                cell_id=cell.cell_id,
                arm_id=cell.arm_id,
                seed=cell.seed,
                gpu=gpu,
                pid=proc.pid,
                command=rendered,
            )
        if not running:
            continue
        time.sleep(POLL_SECONDS)
        for gpu in list(running):
            cell, proc, _, started = running[gpu]
            code = proc.poll()
            if code is not None:
                finish(gpu, code)
            elif cell_timeout is not None and time.monotonic() - started > cell_timeout:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except AttributeError:
                    # ``killpg`` is POSIX-only.  On Windows, terminate the
                    # shell and every command it launched so a timed-out
                    # worker cannot keep running after its cell is released.
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait()
                log.write("cell_killed_timeout", cell_id=cell.cell_id, gpu=gpu)
                finish(gpu, None, timed_out=True)
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="grid.manifest.json (goal6) or run.plan.json (goal7)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="grid output root holding cells/ (default: the manifest's directory)",
    )
    parser.add_argument(
        "--gpus", default="0,1,2,3", help="comma-separated GPU ids, one worker per id"
    )
    parser.add_argument(
        "--cmd",
        required=True,
        help="per-cell shell command; placeholders {cell_id} {arm_id} {seed} {gpu}",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="requeue retained non-complete rows (goal6 semantics; goal7 rows are immutable)",
    )
    parser.add_argument(
        "--cell-timeout",
        type=float,
        help="kill a cell after this many seconds; the harness wall budget is the real limit",
    )
    parser.add_argument(
        "--log", type=Path, help="scheduler JSONL log (default: <root>/scheduler.log.jsonl)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list outstanding cells without running"
    )
    args = parser.parse_args(argv)

    root = args.root if args.root is not None else args.manifest.parent
    cells = load_cells(args.manifest, root)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus named no devices")

    pending: list[Cell] = []
    complete = 0
    retained: list[tuple[str, str]] = []
    for cell in cells:
        status = row_status(cell)
        if status is None:
            pending.append(cell)
        elif status == "complete":
            complete += 1
        elif args.retry_failed:
            pending.append(cell)
        else:
            retained.append((cell.cell_id, status))

    if args.dry_run:
        for cell in pending:
            print(f"pending {cell.cell_id} (arm={cell.arm_id} seed={cell.seed})")
        print(f"{complete} complete, {len(retained)} retained failures, {len(pending)} pending")
        return 0

    log = SchedulerLog(args.log if args.log is not None else root / "scheduler.log.jsonl")
    log.write(
        "schedule_started",
        manifest=str(args.manifest),
        cells_total=len(cells),
        already_complete=complete,
        retained_failures=[f"{cell_id}:{status}" for cell_id, status in retained],
        pending=len(pending),
        gpus=gpus,
    )
    outcomes = run_schedule(
        pending, gpus, args.cmd, log, root / "scheduler", cell_timeout=args.cell_timeout
    )

    ran_complete = sum(1 for outcome in outcomes.values() if outcome == "complete")
    failures = {cid: out for cid, out in outcomes.items() if out != "complete"}
    summary = {
        "cells_total": len(cells),
        "already_complete": complete,
        "newly_complete": ran_complete,
        "failed": failures,
        "retained_failures_skipped": len(retained),
        "grid_complete": complete + ran_complete == len(cells),
    }
    log.write("schedule_finished", **summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["grid_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
