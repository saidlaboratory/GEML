"""Tiny fixtures for the audit-linked Goals 1-12 final report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from geml.analysis.final.report import (
    ClaimTopic,
    EvidenceClass,
    FinalReportValidationError,
    ReportSection,
    TraceableClaim,
    audit_final_report,
    build_final_report,
    render_final_report_markdown,
    report_sha256,
    require_certified_report,
)
from geml.experiments.goal11.corpus_v3 import (
    ArtifactFormat,
    ArtifactReference,
    CompletenessState,
    WorkshopManifest,
)


def _source_file(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "fixture-v1",
                "rows": {
                    claim.claim_id: claim.evidence_projection() for claim in _all_claims("a" * 64)
                },
            },
            sort_keys=True,
        )
    )
    import hashlib

    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    checksum: str | None,
    *,
    state: CompletenessState = CompletenessState.COMPLETE,
    required: bool = True,
) -> WorkshopManifest:
    return WorkshopManifest(
        config_sha256="c" * 64,
        expected_seeds=(1, 2, 3),
        artifacts=(
            ArtifactReference(
                artifact_id="results",
                producer_issue="fixture",
                category="report_evidence",
                roles=("final_report",),
                required=required,
                relative_path="results.json" if checksum is not None else None,
                artifact_format=ArtifactFormat.JSON,
                state=state,
                reason=None if state is CompletenessState.COMPLETE else "fixture missing",
                observed_sha256=checksum,
                size_bytes=1 if checksum is not None else None,
                observed_schema_version="fixture-v1" if checksum is not None else None,
            ),
        ),
        deferred_experiments=(),
    )


def _claim(checksum: str) -> TraceableClaim:
    return TraceableClaim(
        claim_id="goal6_accuracy",
        goal_id="6",
        evidence_class=EvidenceClass.CONTROLLED_LEARNED,
        topic=ClaimTopic.PREDICTIVE_UTILITY,
        statement="Fixture balanced accuracy",
        state=CompletenessState.COMPLETE,
        numeric_value=0.8,
        unit="fraction",
        attempted_count=10,
        valid_count=8,
        failed_count=1,
        invalid_count=0,
        unsupported_count=0,
        timeout_count=1,
        ci_low=0.7,
        ci_high=0.9,
        source_artifact_id="results",
        source_sha256=checksum,
        source_locator="/rows/goal6_accuracy",
    )


def _all_claims(checksum: str) -> tuple[TraceableClaim, ...]:
    base = _claim(checksum)
    variants = (
        (
            "goals1_5_structural",
            "5",
            EvidenceClass.IMMUTABLE_STRUCTURAL,
            ClaimTopic.STRUCTURAL_COMPRESSION,
        ),
        (
            "goal6_accuracy",
            "6",
            EvidenceClass.CONTROLLED_LEARNED,
            ClaimTopic.PREDICTIVE_UTILITY,
        ),
        (
            "goal7_policy",
            "7",
            EvidenceClass.CONTROLLED_LEARNED,
            ClaimTopic.PREDICTIVE_UTILITY,
        ),
        (
            "goal8_proof",
            "8",
            EvidenceClass.CONTROLLED_LEARNED,
            ClaimTopic.PROOF_SEARCH,
        ),
        (
            "goal9_sr",
            "9",
            EvidenceClass.CONTROLLED_LEARNED,
            ClaimTopic.SYMBOLIC_REGRESSION,
        ),
        (
            "goal10_conformance",
            "10",
            EvidenceClass.COMPILER_CONFORMANCE,
            ClaimTopic.COMPILER_CONFORMANCE,
        ),
        (
            "goal11_efficiency",
            "11",
            EvidenceClass.FIXED_SCALE_SYNTHESIS,
            ClaimTopic.FIXED_SCALE_EFFICIENCY,
        ),
        (
            "goal12_reproducibility",
            "12",
            EvidenceClass.LIMITATION,
            ClaimTopic.REPRODUCIBILITY,
        ),
    )
    return tuple(
        TraceableClaim.model_validate(
            base.model_dump()
            | {
                "claim_id": claim_id,
                "goal_id": goal_id,
                "evidence_class": evidence_class,
                "topic": topic,
                "source_locator": f"/rows/{claim_id}",
            }
        )
        for claim_id, goal_id, evidence_class, topic in variants
    )


def _all_sections(checksum: str) -> tuple[ReportSection, ...]:
    claims = {claim.claim_id: claim for claim in _all_claims(checksum)}
    return (
        ReportSection(
            section_id="goals1_5",
            title="Goals 1-5",
            required=True,
            state=CompletenessState.COMPLETE,
            claims=(claims["goals1_5_structural"],),
        ),
        ReportSection(
            section_id="goal6",
            title="Goal 6",
            required=True,
            state=CompletenessState.COMPLETE,
            claims=(claims["goal6_accuracy"],),
        ),
        ReportSection(
            section_id="goal7",
            title="Goal 7",
            required=True,
            state=CompletenessState.COMPLETE,
            claims=(claims["goal7_policy"],),
        ),
        ReportSection(
            section_id="goal8",
            title="Goal 8",
            required=True,
            state=CompletenessState.COMPLETE,
            claims=(claims["goal8_proof"],),
        ),
        ReportSection(
            section_id="goal9",
            title="Goal 9",
            required=True,
            state=CompletenessState.COMPLETE,
            claims=(claims["goal9_sr"],),
        ),
        ReportSection(
            section_id="goal10",
            title="Goal 10",
            required=True,
            state=CompletenessState.COMPLETE,
            claims=(claims["goal10_conformance"],),
        ),
        ReportSection(
            section_id="goal11",
            title="Goal 11",
            required=True,
            state=CompletenessState.COMPLETE,
            claims=(claims["goal11_efficiency"],),
        ),
        ReportSection(
            section_id="external_llm",
            title="External LLM",
            required=False,
            state=CompletenessState.MISSING,
            reason="optional fixture evidence is absent",
        ),
        ReportSection(
            section_id="reproducibility",
            title="Reproducibility",
            required=True,
            state=CompletenessState.COMPLETE,
            claims=(claims["goal12_reproducibility"],),
        ),
    )


def _missing_sections() -> tuple[ReportSection, ...]:
    return tuple(
        ReportSection(
            section_id=section_id,
            title=section_id,
            required=required,
            state=CompletenessState.MISSING,
            reason="production results are absent",
        )
        for section_id, required in (
            ("goals1_5", True),
            ("goal6", True),
            ("goal7", True),
            ("goal8", True),
            ("goal9", True),
            ("goal10", True),
            ("goal11", True),
            ("external_llm", False),
            ("reproducibility", True),
        )
    )


def test_complete_claim_renders_with_checksum_locator_and_denominators(tmp_path: Path):
    _, checksum = _source_file(tmp_path)
    report, audit = build_final_report(
        _manifest(checksum),
        _all_sections(checksum),
        tmp_path,
        threats_to_validity=("fixture only",),
    )
    markdown = render_final_report_markdown(report, audit)

    assert report.certified
    assert audit.valid
    assert "`results` / `/rows/goal6_accuracy`" in markdown
    assert "attempted=10; valid=8" in markdown
    assert any(item[0] == "goal6_accuracy" for item in audit.claim_to_artifact)
    require_certified_report(audit)


def test_missing_required_source_and_section_remain_explicit(tmp_path: Path):
    manifest = _manifest(None, state=CompletenessState.MISSING)
    report, audit = build_final_report(
        manifest,
        _missing_sections(),
        tmp_path,
        threats_to_validity=("fixture only",),
    )
    markdown = render_final_report_markdown(report, audit)

    assert not report.certified
    assert "State: `missing`" in markdown
    with pytest.raises(FinalReportValidationError) as raised:
        require_certified_report(audit)
    assert raised.value.audit == audit


def test_mutated_source_invalidates_previously_complete_report(tmp_path: Path):
    path, checksum = _source_file(tmp_path)
    report, _ = build_final_report(
        _manifest(checksum),
        _all_sections(checksum),
        tmp_path,
        threats_to_validity=("fixture only",),
    )
    path.write_text('{"changed":true}')
    audit = audit_final_report(report, tmp_path)

    assert not audit.valid
    assert any("checksum mismatch" in error for error in audit.errors)


def test_claim_value_and_locator_must_match_the_frozen_source_row(tmp_path: Path):
    _, checksum = _source_file(tmp_path)
    sections = list(_all_sections(checksum))
    goal6 = sections[1]
    tampered = goal6.claims[0].model_copy(update={"numeric_value": 0.123})
    sections[1] = goal6.model_copy(update={"claims": (tampered,)})

    report, audit = build_final_report(
        _manifest(checksum),
        tuple(sections),
        tmp_path,
        threats_to_validity=("fixture only",),
    )

    assert not report.certified
    assert any("field disagrees" in error for error in audit.errors)


def test_numeric_claim_requires_source_unit_and_complete_denominators():
    payload = _claim("a" * 64).model_dump()
    payload["source_locator"] = None
    with pytest.raises(ValidationError, match="artifact, checksum, and row locator"):
        TraceableClaim.model_validate(payload)

    payload = _claim("a" * 64).model_dump()
    payload["attempted_count"] = 11
    with pytest.raises(ValidationError, match="account for every attempt"):
        TraceableClaim.model_validate(payload)


def test_goal10_and_external_claim_taxonomies_are_enforced():
    payload = _claim("a" * 64).model_dump()
    payload.update({"goal_id": "10", "topic": ClaimTopic.PREDICTIVE_UTILITY})
    with pytest.raises(ValidationError, match="must be compiler-conformance"):
        TraceableClaim.model_validate(payload)

    payload = _claim("a" * 64).model_dump()
    payload.update(
        {
            "evidence_class": EvidenceClass.EXTERNAL_NONCONTROLLED,
            "topic": ClaimTopic.PREDICTIVE_UTILITY,
        }
    )
    with pytest.raises(ValidationError, match="external-context"):
        TraceableClaim.model_validate(payload)


def test_prohibited_scaling_or_v2_learning_topics_are_not_in_public_enum():
    assert "scaling_law" not in {item.value for item in ClaimTopic}
    assert "v2_learning_effect" not in {item.value for item in ClaimTopic}
    with pytest.raises(ValidationError):
        TraceableClaim.model_validate(_claim("a" * 64).model_dump() | {"topic": "scaling_law"})


def test_goals1_to_5_cannot_be_recast_as_controlled_learned_evidence():
    payload = _claim("a" * 64).model_dump()
    payload.update({"goal_id": "5", "evidence_class": EvidenceClass.CONTROLLED_LEARNED})
    with pytest.raises(ValidationError, match="immutable structural"):
        TraceableClaim.model_validate(payload)

    payload = _claim("a" * 64).model_dump()
    payload.update(
        {
            "goal_id": "5",
            "evidence_class": EvidenceClass.IMMUTABLE_STRUCTURAL,
            "topic": ClaimTopic.PREDICTIVE_UTILITY,
        }
    )
    with pytest.raises(ValidationError, match="topics must remain structural"):
        TraceableClaim.model_validate(payload)


def test_goal10_cannot_be_labeled_controlled_learned():
    payload = _claim("a" * 64).model_dump()
    payload.update(
        {
            "goal_id": "10",
            "evidence_class": EvidenceClass.CONTROLLED_LEARNED,
            "topic": ClaimTopic.COMPILER_CONFORMANCE,
        }
    )
    with pytest.raises(ValidationError, match="must be compiler-conformance"):
        TraceableClaim.model_validate(payload)


def test_empty_report_inventory_is_rejected(tmp_path: Path):
    _, checksum = _source_file(tmp_path)
    with pytest.raises(ValidationError, match="frozen Goals 1-12 inventory"):
        build_final_report(
            _manifest(checksum),
            (),
            tmp_path,
            threats_to_validity=("fixture only",),
        )


def test_render_and_report_hash_are_deterministic(tmp_path: Path):
    _, checksum = _source_file(tmp_path)
    first, first_audit = build_final_report(
        _manifest(checksum),
        _all_sections(checksum),
        tmp_path,
        threats_to_validity=("fixture only",),
    )
    second, second_audit = build_final_report(
        _manifest(checksum),
        _all_sections(checksum),
        tmp_path,
        threats_to_validity=("fixture only",),
    )

    assert report_sha256(first) == report_sha256(second)
    assert render_final_report_markdown(first, first_audit) == render_final_report_markdown(
        second, second_audit
    )


def test_checked_in_report_is_an_incomplete_scaffold():
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs" / "goals" / "FINAL_REPORT.md").read_text(encoding="utf-8")

    assert "incomplete — production evidence pending" in text
    assert "No production claim is available" in text
    assert "scaling-law conclusion" in text
