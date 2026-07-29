#!/usr/bin/env python3
"""Run bounded fixture-only tests for one implemented workshop goal."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE_TARGETS = {
    "6": (
        "tests/data/test_pairs.py",
        "tests/learning/test_harness.py",
        "tests/experiments/test_goal6_smoke.py",
        "tests/analysis/test_goal6_analysis.py",
    ),
    "7": (
        "tests/data/test_steps.py",
        "tests/learning/test_policy.py",
        "tests/learning/test_step_metrics.py",
        "tests/experiments/test_goal7_smoke.py",
    ),
    "8": (
        "tests/data/test_proof_benchmark.py",
        "tests/learning/test_search_harness.py",
        "tests/learning/test_value_head.py",
        "tests/experiments/test_goal8_smoke.py",
        "tests/experiments/test_goal8_simplify_smoke.py",
        "tests/analysis/test_goal8_analysis.py",
    ),
    "9": (
        "tests/data/test_sr_benchmark.py",
        "tests/learning/test_sr_baselines.py",
        "tests/experiments/test_goal9_smoke.py",
        "tests/analysis/test_goal9_analysis.py",
    ),
    "11": (
        "tests/experiments/test_goal11_corpus_smoke.py",
        "tests/experiments/test_goal11_scaling_smoke.py",
        "tests/experiments/test_goal11_smoke.py",
        "tests/experiments/test_goal11_llm_smoke.py",
    ),
    "12": ("tests/analysis/test_final_report.py",),
}


def smoke_command(goal: str) -> list[str]:
    """Build a no-network, no-GPU fixture command for an implemented goal."""

    try:
        targets = SMOKE_TARGETS[goal]
    except KeyError as error:
        raise ValueError(
            f"Goal {goal} has no reproducible smoke entry point; see the blocker ledger"
        ) from error
    return [sys.executable, "-m", "pytest", "-q", *targets]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", choices=(*tuple(sorted(SMOKE_TARGETS)), "all"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    goals = tuple(sorted(SMOKE_TARGETS)) if arguments.goal == "all" else (arguments.goal,)
    for goal in goals:
        command = smoke_command(goal)
        print(" ".join(command))
        if not arguments.dry_run:
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
