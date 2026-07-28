"""Reproducibility entry points remain local, bounded, and secret-free."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_goal8_smoke_runner_has_no_network_or_production_output_dependency() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/repro/run_smoke.py", "--goal", "8", "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "tests/data/test_proof_benchmark.py" in completed.stdout
    assert "outputs/" not in completed.stdout


def test_llm_preflight_is_offline_and_does_not_echo_credentials() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/repro/preflight_llm.py", "--provider", "openai"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "network_calls=0" in completed.stdout
    assert "OPENAI_API_KEY" in completed.stdout
