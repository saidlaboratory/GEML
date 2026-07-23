"""
runtime.py - checkpoint/resume, atomic shard writing, environment recording

owned by 3-6

the core idea: results get appended to a jsonl output file as they're
computed. on resume, we read back whatever's already in that file and
skip re-processing those expression_ids. the final summary is always
computed from the COMPLETE output file content (not just what happened
in this particular call) - that's what makes an uninterrupted run and
a resumed run produce identical final summaries, even though the
individual run's timing/throughput numbers will naturally differ
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Any
import json
import os
import platform
import subprocess
import sys
import tempfile
import time


def _peak_memory_kb() -> float:
    try:
        import resource
    except ImportError:
        return -1.0  # not available on this platform (e.g. windows)
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on linux, bytes on macOS - normalize to KB best-effort
    return raw / 1024 if sys.platform == "darwin" else float(raw)


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "unknown"


@dataclass
class EnvironmentInfo:
    python_version: str
    platform: str
    git_commit: str
    package_versions: dict[str, str]


def capture_environment(extra_packages: dict[str, str] | None = None) -> EnvironmentInfo:
    return EnvironmentInfo(
        python_version=sys.version,
        platform=platform.platform(),
        git_commit=_git_commit_hash(),
        package_versions=dict(extra_packages) if extra_packages else {},
    )


@dataclass
class RowResult:
    expression_id: str
    status: str  # "success" | "failure"
    metrics: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None


def _atomic_write_jsonl(path: Path, all_rows: list[dict]) -> None:
    # rewrites the whole file via temp+rename - simple and safe at the
    # scale one shard needs, and it's what makes a crash mid-write never
    # leave a half-written output file at the real path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_all_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_completed_ids(output_path: Path) -> set[str]:
    return {row["expression_id"] for row in _read_all_rows(output_path)}


@dataclass
class RunSummary:
    run_id: str
    started_at: float
    ended_at: float
    elapsed_seconds: float
    processed_count: int
    success_count: int
    failure_count: int
    peak_memory_kb: float
    throughput_per_second: float
    environment: EnvironmentInfo
    reproduction_command: str


def _get_expression_id(record: Any) -> str:
    if hasattr(record, "expression_id"):
        return record.expression_id
    return record["expression_id"]


def process_corpus(
    records: Iterable[Any],
    process_fn: Callable[[Any], RowResult],
    output_path: Path,
    run_id: str,
    reproduction_command: str,
    resume: bool = True,
) -> RunSummary:
    """
    walks through records, skips anything already in output_path if
    resuming, calls process_fn on everything else, writes results
    atomically, and returns a summary computed from the FULL output
    file - so resuming and running straight through both end up with
    the same final counts, even if the two runs took different amounts
    of wall time to get there
    """
    started_at = time.time()

    if not resume and output_path.exists():
        output_path.unlink()

    completed_ids = load_completed_ids(output_path) if resume else set()
    all_rows = _read_all_rows(output_path) if resume else []

    new_row_count = 0
    for record in records:
        expr_id = _get_expression_id(record)
        if expr_id in completed_ids:
            continue  # already done - this is the actual resume behavior

        result = process_fn(record)
        all_rows.append({
            "expression_id": result.expression_id,
            "status": result.status,
            "metrics": result.metrics,
            "error_type": result.error_type,
            "error_message": result.error_message,
        })
        new_row_count += 1

    if new_row_count:
        _atomic_write_jsonl(output_path, all_rows)

    ended_at = time.time()
    elapsed = ended_at - started_at

    success_count = sum(1 for r in all_rows if r["status"] == "success")
    failure_count = sum(1 for r in all_rows if r["status"] == "failure")
    throughput = new_row_count / elapsed if elapsed > 0 else 0.0

    return RunSummary(
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=elapsed,
        processed_count=len(all_rows),
        success_count=success_count,
        failure_count=failure_count,
        peak_memory_kb=_peak_memory_kb(),
        throughput_per_second=throughput,
        environment=capture_environment(),
        reproduction_command=reproduction_command,
    )
