"""Read-only compatibility reader for external #82 LLM reference rows."""

from __future__ import annotations

import json
from pathlib import Path


def read_external_reference_rows(path: Path) -> tuple[dict[str, object], ...]:
    """Load external rows without provider calls or inclusion in controlled Gate G8 metrics."""

    rows = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("external LLM reference rows must be JSON objects")
    return rows
