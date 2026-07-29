"""Fixture tests for the ``python -m geml.data.pairs`` production driver (issue 6-1)."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from geml.data.pairs.__main__ import (
    MissingProductionDependencyError,
    PairProductionError,
    load_production_config,
    require_production_providers,
    resolve_corpus_manifest,
)

_CONFIG = Path("configs/goal6_pairs.yaml")


def _config_copy(tmp_path, **overrides):
    loaded = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    loaded.update(overrides)
    path = tmp_path / "pairs.yaml"
    path.write_text(yaml.safe_dump(loaded), encoding="utf-8")
    return path


def test_shipped_config_loads_and_digests():
    config, digest = load_production_config(_CONFIG)
    assert config["base_counts"] == {
        "train": 50000,
        "validation": 5000,
        "test_iid": 5000,
        "test_ood": 5000,
    }
    assert digest.startswith("sha256:")
    assert len(digest) == 71


def test_missing_config_is_a_typed_error(tmp_path):
    with pytest.raises(PairProductionError, match="does not exist"):
        load_production_config(tmp_path / "nope.yaml")


def test_wrong_schema_version_is_rejected(tmp_path):
    path = _config_copy(tmp_path, schema_version="geml-goal6-pairs-config-v0")
    with pytest.raises(PairProductionError, match="schema_version"):
        load_production_config(path)


def test_missing_split_in_base_counts_is_rejected(tmp_path):
    path = _config_copy(
        tmp_path, base_counts={"train": 50000, "validation": 5000, "test_iid": 5000}
    )
    with pytest.raises(PairProductionError, match="exactly the splits"):
        load_production_config(path)


def test_unset_artifacts_root_fails_closed(monkeypatch):
    monkeypatch.delenv("GEML_ARTIFACTS_ROOT", raising=False)
    with pytest.raises(MissingProductionDependencyError, match="GEML_ARTIFACTS_ROOT is not set"):
        resolve_corpus_manifest("${GEML_ARTIFACTS_ROOT}/corpus/corpus_manifest.json")


def test_relative_artifacts_root_is_rejected(monkeypatch):
    monkeypatch.setenv("GEML_ARTIFACTS_ROOT", "relative/path")
    with pytest.raises(PairProductionError, match="absolute path"):
        resolve_corpus_manifest("${GEML_ARTIFACTS_ROOT}/corpus/corpus_manifest.json")


def test_path_escaping_the_root_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("GEML_ARTIFACTS_ROOT", str(tmp_path))
    with pytest.raises(PairProductionError, match="remain within"):
        resolve_corpus_manifest("${GEML_ARTIFACTS_ROOT}/../outside/manifest.json")


def test_path_without_placeholder_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("GEML_ARTIFACTS_ROOT", str(tmp_path))
    with pytest.raises(PairProductionError, match="placeholder"):
        resolve_corpus_manifest(str(tmp_path / "corpus_manifest.json"))


def test_missing_manifest_names_the_full_path(monkeypatch, tmp_path):
    monkeypatch.setenv("GEML_ARTIFACTS_ROOT", str(tmp_path))
    with pytest.raises(MissingProductionDependencyError, match=str(tmp_path)):
        resolve_corpus_manifest("${GEML_ARTIFACTS_ROOT}/corpus/corpus_manifest.json")


def test_present_manifest_resolves_but_providers_still_refuse(monkeypatch, tmp_path):
    """Even with the corpus delivered, unbound engine/verifier providers stop the build -
    the driver can never quietly produce records from anything but the real wiring."""
    manifest = tmp_path / "corpus" / "corpus_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GEML_ARTIFACTS_ROOT", str(tmp_path))
    resolved = resolve_corpus_manifest("${GEML_ARTIFACTS_ROOT}/corpus/corpus_manifest.json")
    assert resolved == manifest
    with pytest.raises(MissingProductionDependencyError, match="never substituted"):
        require_production_providers()


def test_cli_refuses_closed_end_to_end():
    """The documented command exits nonzero with the typed refusal, not a traceback."""
    completed = subprocess.run(
        [sys.executable, "-m", "geml.data.pairs", "--config", str(_CONFIG)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src"},
    )
    assert completed.returncode == 2
    assert "refusing to build" in completed.stderr
    assert "GEML_ARTIFACTS_ROOT" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_help_documents_the_command():
    completed = subprocess.run(
        [sys.executable, "-m", "geml.data.pairs", "--help"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src"},
    )
    assert completed.returncode == 0
    assert "--corpus-manifest" in completed.stdout
