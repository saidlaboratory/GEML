"""Tiny deterministic tests for the bounded Goal 10 conformance set."""

from __future__ import annotations

from pathlib import Path

import pytest

from geml.experiments.goal10 import corpus_v2
from geml.experiments.goal10.corpus_v2 import (
    CONFORMANCE_MAXIMUM_RECORDS,
    ConformanceConfig,
    ExpectedCaseStatus,
    ProbeStatus,
    build_conformance_run,
    load_conformance_config,
    write_conformance_run,
)
from geml.spec.domains import GRAMMAR_V2_DOMAIN_REGISTRY

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _config() -> ConformanceConfig:
    return load_conformance_config(REPOSITORY_ROOT / "configs/goal10_corpus_v2.yaml")


def test_predeclared_set_is_bounded_and_has_complete_required_coverage():
    config = _config()
    run = build_conformance_run(config)

    assert 0 < run.manifest.record_count <= CONFORMANCE_MAXIMUM_RECORDS
    assert run.manifest.record_count == len(config.cases) * len(config.compiler_modes)
    assert run.manifest.operator_counts == config.operator_quotas
    assert run.manifest.region_counts == config.region_quotas
    assert set(run.manifest.operator_counts) == {
        "e",
        "pi",
        "atan",
        "asin",
        "acos",
        "atan_asin",
        "acos_atan",
    }
    assert run.manifest.learning_split is None
    assert run.manifest.benchmark_kind == "bounded_compiler_conformance_not_a_corpus"
    assert all(record.domain_mode in GRAMMAR_V2_DOMAIN_REGISTRY for record in run.records)


def test_two_identical_runs_have_identical_manifest_and_content_hashes(tmp_path: Path):
    first = build_conformance_run(_config())
    second = build_conformance_run(_config())

    assert first == second
    assert first.manifest.config_sha256 == second.manifest.config_sha256
    assert first.manifest.content_sha256 == second.manifest.content_sha256
    assert first.manifest.manifest_sha256 == second.manifest.manifest_sha256

    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    write_conformance_run(first, first_path)
    write_conformance_run(second, second_path)
    assert (first_path / "manifest.json").read_bytes() == (
        second_path / "manifest.json"
    ).read_bytes()
    assert (first_path / "records.jsonl").read_bytes() == (
        second_path / "records.jsonl"
    ).read_bytes()


def test_writer_preflights_all_conflicts_before_creating_any_file(tmp_path: Path):
    run = build_conformance_run(_config())
    output = tmp_path / "conflict"
    output.mkdir()
    (output / "records.jsonl").write_text("different\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_conformance_run(run, output)

    assert not (output / "manifest.json").exists()
    assert (output / "records.jsonl").read_text(encoding="utf-8") == "different\n"


def test_invalid_unsupported_and_signed_zero_rows_are_retained():
    run = build_conformance_run(_config())
    invalid = [
        row for row in run.records if row.expected_status is ExpectedCaseStatus.INVALID_DOMAIN
    ]
    nonfinite = [row for row in run.records if row.expected_status is ExpectedCaseStatus.NONFINITE]
    unsupported = [row for row in run.records if row.observed_status is ProbeStatus.UNSUPPORTED]
    signed_zero = [row for row in run.records if row.region.value == "signed_zero"]

    assert invalid
    assert all(row.observed_status is ProbeStatus.INVALID_DOMAIN for row in invalid)
    assert all(row.pure_eml_fingerprint is not None for row in invalid)
    assert nonfinite
    assert all(row.observed_status is ProbeStatus.NONFINITE_INPUT for row in nonfinite)
    assert all(row.pure_eml_fingerprint is not None for row in nonfinite)
    assert unsupported
    assert signed_zero
    assert all(row.numeric is not None for row in signed_zero)
    assert all(row.failure_type is not None for row in (*invalid, *nonfinite, *unsupported))


def test_pure_eml_rows_have_fingerprints_statistics_and_typed_outcomes():
    run = build_conformance_run(_config())

    compiled = [row for row in run.records if row.pure_eml_fingerprint is not None]
    assert compiled
    assert all(row.node_count is not None and row.depth is not None for row in compiled)
    assert all(len(row.pure_eml_fingerprint or "") == 64 for row in compiled)
    assert all(
        row.failure_type is not None
        for row in run.records
        if row.observed_status
        not in {ProbeStatus.PASS, ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE}
    )


def test_numeric_timeout_is_retained_as_its_own_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
):
    def timeout(*args: object, **kwargs: object) -> None:
        raise TimeoutError("fixture deadline")

    config = _config()
    monkeypatch.setattr(corpus_v2, "_evaluate_iterative", timeout)
    record = corpus_v2._run_case(
        config.cases[0],
        config.compiler_modes[0],
        config,
        {},
    )

    assert record.observed_status is ProbeStatus.TIMEOUT
    assert record.failure_type == "TimeoutError"


def test_qa_explicitly_denies_corpus_and_learning_artifacts():
    text = (REPOSITORY_ROOT / "docs/goals/goal10/CORPUS_V2_QA.md").read_text()
    assert "No corpus v2 was generated" in text
    assert "No compression, learning, or corpus claim" in text
