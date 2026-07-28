"""Core-only tests for Goal 8 value-target leakage and diagnostics contracts."""

from __future__ import annotations

import pytest

from geml.learning.value.head import ValueContractError, ValueExampleV1, value_diagnostics
from geml.learning.value.train import exclude_benchmark_groups, require_three_seed_rows


def test_benchmark_related_training_groups_are_excluded() -> None:
    first = ValueExampleV1("a", "graph-a", ("group-a",), 1)
    second = ValueExampleV1("b", "graph-b", ("group-b", "relative-b"), 3)
    retained = exclude_benchmark_groups((first, second), benchmark_group_ids=("relative-b",))

    assert retained == (first,)
    assert require_three_seed_rows((20260726, 20260727, 20260728)) == (
        20260726,
        20260727,
        20260728,
    )
    with pytest.raises(ValueContractError, match="three retained"):
        require_three_seed_rows((1, 1))


def test_value_diagnostics_report_distance_tiers() -> None:
    diagnostics = value_diagnostics(
        (0.0, 2.0, 4.0),
        (0, 3, 3),
        tier_names=("short", "medium", "medium"),
    )
    assert diagnostics.mae == pytest.approx(2 / 3)
    assert dict(diagnostics.tier_mae)["medium"] == pytest.approx(1.0)
