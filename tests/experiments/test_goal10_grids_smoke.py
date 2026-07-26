"""Offline compatibility checks for issue 10-3."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from geml.eml.compiler_core import CompilerMode
from geml.experiments.goal10.rerun_grids import (
    CompatibilityConfig,
    CompatibilityError,
    GrammarSelection,
    group_rows_explicitly,
    make_tiny_compatibility_row,
    require_homogeneous_rows,
    resolve_selection,
    run_compatibility_checks,
)
from geml.spec.domains import GrammarVersion

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _config() -> CompatibilityConfig:
    payload = yaml.safe_load((REPOSITORY_ROOT / "configs/goal10_grids.yaml").read_text())
    return CompatibilityConfig.model_validate(payload)


def test_omitted_selection_is_v1_official_and_v2_is_explicit():
    assert resolve_selection() == GrammarSelection(
        grammar_version=GrammarVersion.V1,
        compiler_mode=CompilerMode.OFFICIAL_V4,
    )
    assert resolve_selection({"grammar_version": "v2"}).grammar_version is GrammarVersion.V2
    with pytest.raises(ValueError, match="grammar_version"):
        resolve_selection({"grammar_version": "v3"})
    with pytest.raises(ValueError, match="Extra inputs"):
        resolve_selection({"implicit_v2": True})


def test_version_mode_labels_are_distinct_and_mixed_rows_are_rejected():
    rows = tuple(make_tiny_compatibility_row(selection) for selection in _config().matrix)
    assert len({row.representation_label for row in rows}) == 4
    assert all(row.grammar_version.value in row.representation_label for row in rows)

    with pytest.raises(CompatibilityError, match="mixed grammar/compiler"):
        require_homogeneous_rows(rows)
    grouped = group_rows_explicitly(rows)
    assert len(grouped) == 4
    assert all(require_homogeneous_rows(cohort) == cohort for cohort in grouped.values())


def test_compatibility_matrix_preserves_immutable_goals_1_to_5_references():
    rows, failures = run_compatibility_checks(_config(), repository_root=REPOSITORY_ROOT)

    assert len(rows) == 4
    assert failures == ()
    assert {row.grammar_version for row in rows} == {
        GrammarVersion.V1,
        GrammarVersion.V2,
    }
    assert {row.compiler_mode for row in rows} == set(CompilerMode)


def test_historical_grid_module_has_no_training_or_production_surface():
    import geml.experiments.goal10.rerun_grids as module

    forbidden = ("train", "fit", "optimizer", "corpus", "motif", "dag")
    public_names = {name.lower() for name in vars(module) if not name.startswith("_")}
    assert not public_names.intersection(forbidden)
    text = (REPOSITORY_ROOT / "docs/goals/goal10/GRID_RERUN.md").read_text()
    assert "no Goal 6 or Goal 7" in text
