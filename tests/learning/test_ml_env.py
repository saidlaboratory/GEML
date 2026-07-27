"""Tests for the optional, pinned Goals 6--12 ML environment policy."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _policy() -> dict[str, object]:
    with (ROOT / "configs" / "ml_env.yaml").open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def test_ml_policy_freezes_registered_seeds_and_bounded_profiles() -> None:
    policy = _policy()

    assert policy["schema_version"] == "geml-ml-environment-v1"
    assert policy["production_seeds"] == [20260726, 20260727, 20260728]
    assert policy["development_selection"] == {
        "selection_split": "validation",
        "selected_once_before_test": True,
        "broad_hyperparameter_sweeps": "prohibited",
    }
    profiles = policy["compute_profiles"]
    assert isinstance(profiles, dict)
    assert profiles["default"]["gpus"] == 2
    assert profiles["conditional_four_gpu"]["gpus"] == 4
    assert profiles["conditional_four_gpu"]["linear_scaling_claims"] == "prohibited"


def test_ml_environment_versions_when_optional_extra_is_installed() -> None:
    policy = _policy()
    packages = policy["packages"]
    assert isinstance(packages, dict)

    try:
        torch_version = metadata.version("torch")
        pyg_version = metadata.version("torch-geometric")
    except metadata.PackageNotFoundError:
        pytest.skip("optional [ml] extra is not installed")

    assert torch_version == packages["torch"]
    assert pyg_version == packages["torch_geometric"]
