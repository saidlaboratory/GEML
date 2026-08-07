"""Fixture audit and Gate G10 semantics for issue 10-2."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest
import yaml

import geml.experiments.goal10.rerun_studies as rerun_studies
from geml.analysis.goal10.summary import render_goal10_summary
from geml.experiments.goal10.corpus_v2 import (
    ConformanceManifest,
    ConformanceRun,
    ProbeStatus,
    VerifierStatus,
    build_conformance_run,
    canonical_json_bytes,
    load_conformance_config,
    write_conformance_run,
)
from geml.experiments.goal10.rerun_studies import (
    EvidenceTier,
    GateState,
    Goal10AuditConfig,
    audit_conformance,
    compute_v1_snapshot,
    load_audit_config,
    load_conformance_run,
    write_audit_report,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _configs():
    conformance = load_conformance_config(REPOSITORY_ROOT / "configs/goal10_corpus_v2.yaml")
    audit_payload = yaml.safe_load((REPOSITORY_ROOT / "configs/goal10_rerun.yaml").read_text())
    return conformance, Goal10AuditConfig.model_validate(audit_payload)


def _rehash_run(
    original: ConformanceRun,
    *,
    records: tuple | None = None,
    manifest_updates: dict | None = None,
) -> ConformanceRun:
    updated_records = original.records if records is None else records
    content = b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in updated_records
    )
    manifest_payload = original.manifest.model_dump(
        mode="json",
        exclude={"manifest_sha256"},
    )
    manifest_payload.update(manifest_updates or {})
    manifest_payload["record_count"] = len(updated_records)
    manifest_payload["content_sha256"] = hashlib.sha256(content).hexdigest()
    manifest_payload["status_counts"] = dict(
        sorted(Counter(row.observed_status.value for row in updated_records).items())
    )
    manifest = ConformanceManifest.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest_payload)).hexdigest(),
        }
    )
    return ConformanceRun(manifest=manifest, records=updated_records)


def test_v1_registry_domain_rules_and_formula_bytes_match_frozen_snapshot():
    _, config = _configs()
    assert compute_v1_snapshot() == config.expected_v1


def test_fixture_audit_recomputes_structure_and_retains_endpoint_failures():
    conformance_config, audit_config = _configs()
    run = build_conformance_run(conformance_config)
    report = audit_conformance(
        run,
        conformance_config=conformance_config,
        audit_config=audit_config,
        evidence_tier=EvidenceTier.FIXTURE,
    )

    assert report.gate is GateState.FAIL
    assert report.integrity_failures == ()
    assert report.v1_failures == ()
    assert report.coverage_failures == ()
    assert report.structural_rows
    assert all(row.passed for row in report.structural_rows)
    assert sum(row.attempted for row in report.numeric_denominators) == len(run.records)
    assert len(report.conformance_failures) == 8
    assert all("nonfinite_result" in failure for failure in report.conformance_failures)


def test_rendered_summary_retains_denominators_blockers_and_nonclaims():
    conformance_config, audit_config = _configs()
    report = audit_conformance(
        build_conformance_run(conformance_config),
        conformance_config=conformance_config,
        audit_config=audit_config,
        evidence_tier=EvidenceTier.FIXTURE,
    )
    markdown = render_goal10_summary(report)

    assert "Gate G10: `fail`" in markdown
    assert f"Conformance configuration SHA-256: `{report.conformance_config_sha256}`" in markdown
    assert f"Audit criteria SHA-256: `{report.audit_config_sha256}`" in markdown
    assert "Max precision units" in markdown
    assert "official_v4:sinh" in markdown
    assert "40 decimal guard digits" in markdown
    assert "Exact denominators" in markdown
    assert "Signed-zero" in markdown
    assert "No corpus-v2, alpha, DAG, motif, compression" in markdown


def test_audit_rejects_self_consistent_tampered_numeric_outcomes():
    conformance_config, audit_config = _configs()
    original = build_conformance_run(conformance_config)
    records = list(original.records)
    index = next(
        index
        for index, row in enumerate(records)
        if row.region.value == "endpoint" and row.observed_status is ProbeStatus.NONFINITE_RESULT
    )
    records[index] = records[index].model_copy(
        update={
            "observed_status": ProbeStatus.PASS,
            "verifier_status": VerifierStatus.PASSED,
            "failure_type": None,
            "failure_message": None,
        }
    )
    tampered = _rehash_run(original, records=tuple(records))

    report = audit_conformance(
        tampered,
        conformance_config=conformance_config,
        audit_config=audit_config,
        evidence_tier=EvidenceTier.FINAL,
    )

    assert report.gate is GateState.FAIL
    assert any("deterministic numeric recomputation" in item for item in report.integrity_failures)


def test_structural_compiler_failure_is_cached_and_fails_the_gate(monkeypatch):
    conformance_config, audit_config = _configs()
    run = build_conformance_run(conformance_config)
    compile_calls: list[tuple[str, object]] = []

    def fail_compilation(case, mode):
        compile_calls.append((case.constructor, mode))
        raise RuntimeError("simulated compiler drift")

    monkeypatch.setattr(rerun_studies, "_compile_case", fail_compilation)
    report = audit_conformance(
        run,
        conformance_config=conformance_config,
        audit_config=audit_config,
        evidence_tier=EvidenceTier.FIXTURE,
    )

    expected_keys = {
        (record.constructor, record.compiler_mode)
        for record in run.records
        if record.pure_eml_fingerprint is not None
    }
    assert set(compile_calls) == expected_keys
    assert len(compile_calls) == len(expected_keys)
    assert report.gate is GateState.FAIL
    assert report.structural_rows
    assert all(row.failure_type == "RuntimeError" for row in report.structural_rows)
    assert all(row.failure_message == "simulated compiler drift" for row in report.structural_rows)


def test_audit_rejects_a_run_with_weaker_than_preregistered_numeric_protocol():
    conformance_config, audit_config = _configs()
    weakened = conformance_config.model_copy(
        update={
            "precision_digits": 80,
            "absolute_tolerance": "1e-20",
            "relative_tolerance": "1e-20",
            "imaginary_tolerance": "1e-20",
        }
    )
    report = audit_conformance(
        build_conformance_run(weakened),
        conformance_config=weakened,
        audit_config=audit_config,
        evidence_tier=EvidenceTier.FIXTURE,
    )

    assert report.gate is GateState.FAIL
    assert {
        item.split()[1] for item in report.integrity_failures if item.startswith("conformance ")
    } == {
        "precision_digits",
        "absolute_tolerance",
        "relative_tolerance",
        "imaginary_tolerance",
    }


def test_final_evidence_requires_a_clean_known_implementation_revision():
    conformance_config, audit_config = _configs()
    run = _rehash_run(
        build_conformance_run(conformance_config),
        manifest_updates={
            "implementation_commit": "unknown",
            "worktree_dirty": True,
        },
    )

    report = audit_conformance(
        run,
        conformance_config=conformance_config,
        audit_config=audit_config,
        evidence_tier=EvidenceTier.FINAL,
    )

    assert report.gate is GateState.FAIL
    assert "final evidence has no known implementation commit" in report.integrity_failures
    assert "final evidence was produced from a dirty worktree" in report.integrity_failures


def test_audit_rejects_self_consistent_forged_runtime_metadata():
    conformance_config, audit_config = _configs()
    original = build_conformance_run(conformance_config)
    run = _rehash_run(
        original,
        manifest_updates={
            "runtime_metadata": {key: "forged" for key in original.manifest.runtime_metadata}
        },
    )

    report = audit_conformance(
        run,
        conformance_config=conformance_config,
        audit_config=audit_config,
        evidence_tier=EvidenceTier.FIXTURE,
    )

    assert report.gate is GateState.FAIL
    assert (
        "runtime metadata disagrees with the fresh audit environment" in report.integrity_failures
    )


def test_audit_runner_loads_and_publishes_immutable_evidence(tmp_path: Path):
    conformance_config, audit_config = _configs()
    run_directory = tmp_path / "run"
    write_conformance_run(build_conformance_run(conformance_config), run_directory)

    loaded = load_conformance_run(run_directory)
    assert loaded.manifest.record_count == len(loaded.records)
    assert load_audit_config(REPOSITORY_ROOT / "configs/goal10_rerun.yaml") == audit_config

    report = audit_conformance(
        loaded,
        conformance_config=conformance_config,
        audit_config=audit_config,
        evidence_tier=EvidenceTier.FIXTURE,
    )
    audit_path, summary_path = write_audit_report(report, tmp_path / "audit")

    assert audit_path.is_file()
    assert summary_path.is_file()
    assert '"gate":"fail"' in audit_path.read_text(encoding="utf-8")
    assert f'"audit_config_sha256":"{report.audit_config_sha256}"' in audit_path.read_text(
        encoding="utf-8"
    )
    assert "Gate G10: `fail`" in summary_path.read_text(encoding="utf-8")
    assert write_audit_report(report, tmp_path / "audit") == (
        audit_path,
        summary_path,
    )


def test_audit_writer_preflights_all_conflicts(tmp_path: Path):
    conformance_config, audit_config = _configs()
    report = audit_conformance(
        build_conformance_run(conformance_config),
        conformance_config=conformance_config,
        audit_config=audit_config,
        evidence_tier=EvidenceTier.FIXTURE,
    )
    output = tmp_path / "audit"
    output.mkdir()
    (output / "GOAL10_SUMMARY.md").write_text("different\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_audit_report(report, output)

    assert not (output / "audit.json").exists()
