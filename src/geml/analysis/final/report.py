"""Audit-linked final report generation for Goals 1-12.

Scientific results enter the renderer only as typed, source-located claims.
Missing inputs remain explicit report rows; this module never runs experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from geml.experiments.goal11.corpus_v3 import (
    CompletenessState,
    DeferredExperiment,
    EvidenceAuthenticationCache,
    EvidenceAuthenticationError,
    WorkshopManifest,
    authenticate_source_record,
    canonical_json_bytes,
    sha256_file,
)

FINAL_REPORT_SCHEMA_VERSION = "geml-final-goals-1-12-report-v1"
FINAL_REPORT_AUDIT_SCHEMA_VERSION = "geml-final-goals-1-12-report-audit-v1"

_NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]

_REQUIRED_SECTION_FLAGS = {
    "goals1_5": True,
    "goal6": True,
    "goal7": True,
    "goal8": True,
    "goal9": True,
    "goal10": True,
    "goal11": True,
    "external_llm": False,
    "reproducibility": True,
}


class FinalReportError(ValueError):
    """Final-report inputs or claims violate the scientific reporting contract."""


class FinalReportValidationError(FinalReportError):
    """A report cannot be certified; the complete audit remains available."""

    def __init__(self, audit: FinalReportAudit) -> None:
        self.audit = audit
        super().__init__(f"final report is not certifiable ({audit.error_count} errors)")


class EvidenceClass(StrEnum):
    IMMUTABLE_STRUCTURAL = "immutable_structural"
    CONTROLLED_LEARNED = "controlled_learned"
    COMPILER_CONFORMANCE = "compiler_conformance"
    FIXED_SCALE_SYNTHESIS = "fixed_scale_synthesis"
    EXTERNAL_NONCONTROLLED = "external_noncontrolled"
    LIMITATION = "limitation"
    DEFERRED = "deferred"


class ClaimTopic(StrEnum):
    STRUCTURAL_COMPRESSION = "structural_compression"
    PREDICTIVE_UTILITY = "predictive_utility"
    PROOF_SEARCH = "proof_search"
    SYMBOLIC_REGRESSION = "symbolic_regression"
    COMPILER_CONFORMANCE = "compiler_conformance"
    FIXED_SCALE_EFFICIENCY = "fixed_scale_efficiency"
    EXTERNAL_CONTEXT = "external_context"
    REPRODUCIBILITY = "reproducibility"
    LIMITATION = "limitation"
    DEFERRED_WORK = "deferred_work"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


def _canonical_relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("report source paths must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("report source paths must be canonical relative POSIX paths")
    return value


class ReportSource(_FrozenModel):
    artifact_id: _NonBlank
    producer_issue: _NonBlank
    category: _NonBlank
    roles: tuple[_NonBlank, ...]
    required: StrictBool
    state: CompletenessState
    relative_path: _NonBlank | None = None
    sha256: _Sha256 | None = None
    schema_version: _NonBlank | None = None
    reason: _NonBlank | None = None

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_relative_path(value)

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.state is CompletenessState.COMPLETE:
            if self.relative_path is None or self.sha256 is None:
                raise ValueError("complete report sources require path and checksum")
            if self.reason is not None:
                raise ValueError("complete report sources cannot carry a reason")
        elif self.reason is None:
            raise ValueError("non-complete report sources require a reason")
        return self


class TraceableClaim(_FrozenModel):
    """One report claim with exact artifact-row provenance."""

    claim_id: _NonBlank
    goal_id: _NonBlank
    evidence_class: EvidenceClass
    topic: ClaimTopic
    statement: _NonBlank
    state: CompletenessState
    numeric_value: StrictInt | StrictFloat | None = None
    unit: _NonBlank | None = None
    attempted_count: _NonNegativeInt | None = None
    valid_count: _NonNegativeInt | None = None
    failed_count: _NonNegativeInt | None = None
    invalid_count: _NonNegativeInt | None = None
    unsupported_count: _NonNegativeInt | None = None
    timeout_count: _NonNegativeInt | None = None
    ci_low: StrictFloat | None = None
    ci_high: StrictFloat | None = None
    source_artifact_id: _NonBlank | None = None
    source_sha256: _Sha256 | None = None
    source_locator: _NonBlank | None = None
    reason: _NonBlank | None = None

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        source_fields = (
            self.source_artifact_id,
            self.source_sha256,
            self.source_locator,
        )
        if self.state is CompletenessState.COMPLETE:
            if any(value is None for value in source_fields):
                raise ValueError("complete claims require artifact, checksum, and row locator")
            if self.reason is not None:
                raise ValueError("complete claims cannot carry an incomplete reason")
        else:
            if self.numeric_value is not None:
                raise ValueError("non-complete claims cannot publish a numeric result")
            if self.reason is None:
                raise ValueError("non-complete claims require a reason")
        if self.numeric_value is not None:
            denominator_fields = (
                self.attempted_count,
                self.valid_count,
                self.failed_count,
                self.invalid_count,
                self.unsupported_count,
                self.timeout_count,
            )
            if self.unit is None or any(value is None for value in denominator_fields):
                raise ValueError("numeric claims require unit and complete denominators")
            attempted, valid, failed, invalid, unsupported, timeout = denominator_fields
            if (
                valid is None
                or failed is None
                or invalid is None
                or unsupported is None
                or timeout is None
                or attempted != valid + failed + invalid + unsupported + timeout
            ):
                raise ValueError("numeric claim denominators must account for every attempt")
            if valid == 0:
                raise ValueError("numeric claims require at least one valid row")
        elif self.unit is not None:
            raise ValueError("unit is only valid for numeric claims")
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("claim uncertainty requires both interval bounds")
        if self.numeric_value is None and self.ci_low is not None:
            raise ValueError("uncertainty is only valid for numeric claims")
        if (
            self.numeric_value is not None
            and self.ci_low is not None
            and self.ci_high is not None
            and not self.ci_low <= float(self.numeric_value) <= self.ci_high
        ):
            raise ValueError("numeric result must lie within its uncertainty interval")
        if self.goal_id == "10" and (
            self.topic is not ClaimTopic.COMPILER_CONFORMANCE
            or self.evidence_class is not EvidenceClass.COMPILER_CONFORMANCE
        ):
            raise ValueError("Goal 10 claims must be compiler-conformance evidence")
        if self.evidence_class is EvidenceClass.COMPILER_CONFORMANCE and self.goal_id != "10":
            raise ValueError("compiler-conformance evidence belongs to Goal 10")
        if self.evidence_class is EvidenceClass.EXTERNAL_NONCONTROLLED:
            if self.topic is not ClaimTopic.EXTERNAL_CONTEXT:
                raise ValueError("external evidence must use the external-context topic")
        elif self.topic is ClaimTopic.EXTERNAL_CONTEXT:
            raise ValueError("external context must be labeled external_noncontrolled")
        if self.goal_id in {"1", "2", "3", "4", "5"} and self.evidence_class not in {
            EvidenceClass.IMMUTABLE_STRUCTURAL,
            EvidenceClass.LIMITATION,
        }:
            raise ValueError("Goals 1-5 are immutable structural evidence in this report")
        if self.goal_id in {"1", "2", "3", "4", "5"}:
            expected_topic = (
                ClaimTopic.STRUCTURAL_COMPRESSION
                if self.evidence_class is EvidenceClass.IMMUTABLE_STRUCTURAL
                else ClaimTopic.LIMITATION
            )
            if self.topic is not expected_topic:
                raise ValueError("Goals 1-5 topics must remain structural or limitations")
        if self.evidence_class is EvidenceClass.FIXED_SCALE_SYNTHESIS and (
            self.goal_id != "11" or self.topic is not ClaimTopic.FIXED_SCALE_EFFICIENCY
        ):
            raise ValueError("fixed-scale synthesis evidence belongs to Goal 11")
        return self

    def evidence_projection(self) -> dict[str, object]:
        return {
            "goal_id": self.goal_id,
            "evidence_class": self.evidence_class.value,
            "topic": self.topic.value,
            "numeric_value": self.numeric_value,
            "unit": self.unit,
            "attempted_count": self.attempted_count,
            "valid_count": self.valid_count,
            "failed_count": self.failed_count,
            "invalid_count": self.invalid_count,
            "unsupported_count": self.unsupported_count,
            "timeout_count": self.timeout_count,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


class ReportSection(_FrozenModel):
    section_id: _NonBlank
    title: _NonBlank
    required: StrictBool
    state: CompletenessState
    claims: tuple[TraceableClaim, ...] = ()
    reason: _NonBlank | None = None

    @model_validator(mode="after")
    def validate_section(self) -> Self:
        if self.state is CompletenessState.COMPLETE:
            if self.reason is not None:
                raise ValueError("complete report sections cannot carry a reason")
            if self.required and not self.claims:
                raise ValueError("complete required sections require traceable claims")
        elif self.reason is None:
            raise ValueError("non-complete report sections require a reason")
        claim_ids = tuple(item.claim_id for item in self.claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("claim IDs must be unique within a section")
        return self


class FinalReportModel(_FrozenModel):
    schema_version: Literal["geml-final-goals-1-12-report-v1"] = FINAL_REPORT_SCHEMA_VERSION
    certified: StrictBool
    workshop_manifest: WorkshopManifest
    sources: tuple[ReportSource, ...]
    sections: tuple[ReportSection, ...]
    deferred_experiments: tuple[DeferredExperiment, ...]
    threats_to_validity: tuple[_NonBlank, ...] = Field(min_length=1)
    boundaries: tuple[_NonBlank, ...] = (
        "Goals 1-5 are immutable structural/compression evidence, not downstream utility.",
        "Goals 6-9 are controlled learned/search benchmarks with explicit denominators.",
        "Goal 10 is opt-in compiler conformance; no corpus or learned rerun was performed.",
        "Goal 11 is fixed-scale only; no 10-100x scaling-law conclusion is available.",
        "Proprietary LLM rows are external, non-controlled context outside all gates.",
    )

    @model_validator(mode="after")
    def validate_uniqueness(self) -> Self:
        source_ids = tuple(item.artifact_id for item in self.sources)
        section_ids = tuple(item.section_id for item in self.sections)
        claim_ids = tuple(claim.claim_id for section in self.sections for claim in section.claims)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("report source IDs must be unique")
        if len(set(section_ids)) != len(section_ids):
            raise ValueError("report section IDs must be unique")
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("claim IDs must be globally unique")
        observed_flags = {item.section_id: item.required for item in self.sections}
        if observed_flags != _REQUIRED_SECTION_FLAGS:
            raise ValueError("report sections must match the frozen Goals 1-12 inventory")
        return self


class FinalReportAudit(_FrozenModel):
    schema_version: Literal["geml-final-goals-1-12-report-audit-v1"] = (
        FINAL_REPORT_AUDIT_SCHEMA_VERSION
    )
    valid: StrictBool
    error_count: _NonNegativeInt
    errors: tuple[_NonBlank, ...]
    claim_to_artifact: tuple[tuple[_NonBlank, _NonBlank, _Sha256, _NonBlank], ...]

    @model_validator(mode="after")
    def validate_audit(self) -> Self:
        if self.error_count != len(self.errors) or self.valid != (not self.errors):
            raise ValueError("final-report audit counters are inconsistent")
        return self


def sources_from_manifest(manifest: WorkshopManifest) -> tuple[ReportSource, ...]:
    """Translate the canonical #79 manifest without repairing any state."""

    return tuple(
        ReportSource(
            artifact_id=item.artifact_id,
            producer_issue=item.producer_issue,
            category=item.category,
            roles=item.roles,
            required=item.required,
            state=item.state,
            relative_path=item.relative_path,
            sha256=item.observed_sha256,
            schema_version=item.observed_schema_version,
            reason=item.reason,
        )
        for item in manifest.artifacts
    )


def _resolve_inside(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise FinalReportError(f"report source escapes artifact root: {relative_path}") from error
    return candidate


def audit_final_report(
    report: FinalReportModel,
    artifact_root: str | Path,
) -> FinalReportAudit:
    """Re-authenticate sources and require exact provenance for every complete claim."""

    root = Path(artifact_root)
    errors: list[str] = []
    expected_sources = sources_from_manifest(report.workshop_manifest)
    if report.sources != expected_sources:
        errors.append("report source inventory disagrees with the embedded workshop manifest")
    source_by_id = {item.artifact_id: item for item in report.sources}
    evidence_cache = EvidenceAuthenticationCache()
    for source in report.sources:
        if source.state is not CompletenessState.COMPLETE:
            if source.required:
                errors.append(f"required source {source.artifact_id} is {source.state.value}")
            continue
        if source.relative_path is None or source.sha256 is None:
            errors.append(f"complete source {source.artifact_id} lacks path/checksum")
            continue
        path = _resolve_inside(root, source.relative_path)
        if not path.is_file():
            errors.append(f"source file is missing: {source.artifact_id}")
            continue
        observed, _ = sha256_file(path)
        if observed != source.sha256:
            errors.append(f"source checksum mismatch: {source.artifact_id}")
    claim_index = []
    for section in report.sections:
        if section.required and section.state is not CompletenessState.COMPLETE:
            errors.append(f"required section {section.section_id} is {section.state.value}")
        for claim in section.claims:
            if claim.state is not CompletenessState.COMPLETE:
                if section.required:
                    errors.append(f"required section claim {claim.claim_id} is {claim.state.value}")
                continue
            if (
                claim.source_artifact_id is None
                or claim.source_sha256 is None
                or claim.source_locator is None
            ):
                errors.append(f"claim {claim.claim_id} lacks complete provenance")
                continue
            source = source_by_id.get(claim.source_artifact_id)
            if source is None:
                errors.append(f"claim {claim.claim_id} names an unknown source")
                continue
            if (
                source.state is not CompletenessState.COMPLETE
                or source.sha256 != claim.source_sha256
            ):
                errors.append(f"claim {claim.claim_id} is not bound to a complete source")
                continue
            allowed_categories = {
                EvidenceClass.IMMUTABLE_STRUCTURAL: (
                    "immutable_input",
                    "dataset",
                    "report",
                    "report_evidence",
                ),
                EvidenceClass.CONTROLLED_LEARNED: (
                    "result_table",
                    "report_evidence",
                ),
                EvidenceClass.COMPILER_CONFORMANCE: (
                    "conformance",
                    "report_evidence",
                ),
                EvidenceClass.FIXED_SCALE_SYNTHESIS: (
                    "result_table",
                    "report_evidence",
                ),
                EvidenceClass.EXTERNAL_NONCONTROLLED: (
                    "external_reference",
                    "report_evidence",
                ),
                EvidenceClass.LIMITATION: (),
                EvidenceClass.DEFERRED: (),
            }[claim.evidence_class]
            try:
                authenticate_source_record(
                    report.workshop_manifest,
                    root,
                    artifact_id=claim.source_artifact_id,
                    source_sha256=claim.source_sha256,
                    source_locator=claim.source_locator,
                    expected_fields=claim.evidence_projection(),
                    required_roles=("final_report",),
                    allowed_categories=allowed_categories,
                    cache=evidence_cache,
                )
            except EvidenceAuthenticationError as error:
                errors.append(f"claim {claim.claim_id} source row is not authenticated: {error}")
                continue
            claim_index.append(
                (
                    claim.claim_id,
                    claim.source_artifact_id,
                    claim.source_sha256,
                    claim.source_locator,
                )
            )
    return FinalReportAudit(
        valid=not errors,
        error_count=len(errors),
        errors=tuple(errors),
        claim_to_artifact=tuple(sorted(claim_index)),
    )


def build_final_report(
    manifest: WorkshopManifest,
    sections: tuple[ReportSection, ...],
    artifact_root: str | Path,
    *,
    threats_to_validity: tuple[str, ...],
) -> tuple[FinalReportModel, FinalReportAudit]:
    """Build and audit a report; certification is derived, never caller asserted."""

    provisional = FinalReportModel(
        certified=False,
        workshop_manifest=manifest,
        sources=sources_from_manifest(manifest),
        sections=sections,
        deferred_experiments=manifest.deferred_experiments,
        threats_to_validity=threats_to_validity,
    )
    provisional_audit = audit_final_report(provisional, artifact_root)
    report = provisional.model_copy(update={"certified": provisional_audit.valid})
    return report, audit_final_report(report, artifact_root)


def require_certified_report(audit: FinalReportAudit) -> None:
    if not audit.valid:
        raise FinalReportValidationError(audit)


def _value_text(claim: TraceableClaim) -> str:
    if claim.numeric_value is None:
        return "—"
    return f"{claim.numeric_value} {claim.unit}"


def _denominator_text(claim: TraceableClaim) -> str:
    if claim.attempted_count is None:
        return "—"
    return (
        f"attempted={claim.attempted_count}; valid={claim.valid_count}; "
        f"failed={claim.failed_count}; invalid={claim.invalid_count}; "
        f"unsupported={claim.unsupported_count}; "
        f"timeout={claim.timeout_count}"
    )


def render_final_report_markdown(
    report: FinalReportModel,
    audit: FinalReportAudit,
) -> str:
    """Render deterministic Markdown solely from typed claims and audit rows."""

    lines = [
        "# GEML Goals 1-12 final findings report",
        "",
        f"Certification: **{'certified' if audit.valid else 'incomplete'}**",
        "",
        "## Evidence boundaries",
        "",
        *(f"- {item}" for item in report.boundaries),
        "",
    ]
    for section in report.sections:
        lines.extend(
            [
                f"## {section.title}",
                "",
                f"State: `{section.state.value}`",
                "",
            ]
        )
        if section.reason is not None:
            lines.extend([f"Reason: {section.reason}", ""])
        if section.claims:
            lines.extend(
                [
                    "| Claim | State | Result | Denominators | Source |",
                    "|---|---|---:|---|---|",
                ]
            )
            for claim in section.claims:
                source = (
                    "—"
                    if claim.source_artifact_id is None
                    else (
                        f"`{claim.source_artifact_id}` / "
                        f"`{claim.source_locator}` / `{claim.source_sha256}`"
                    )
                )
                lines.append(
                    f"| {claim.statement} | `{claim.state.value}` | "
                    f"{_value_text(claim)} | {_denominator_text(claim)} | {source} |"
                )
            lines.append("")
    lines.extend(["## Explicitly deferred experiments", ""])
    if report.deferred_experiments:
        lines.extend(
            f"- `{item.experiment_id}`: {item.reason}" for item in report.deferred_experiments
        )
    else:
        lines.append("- _none recorded_")
    lines.extend(["", "## Threats to validity", ""])
    lines.extend(f"- {item}" for item in report.threats_to_validity)
    lines.extend(["", "## Claim-to-artifact index", ""])
    lines.extend(
        [
            "| Claim ID | Artifact ID | SHA-256 | Row locator |",
            "|---|---|---|---|",
        ]
    )
    if audit.claim_to_artifact:
        lines.extend(
            f"| `{claim_id}` | `{artifact_id}` | `{checksum}` | `{locator}` |"
            for claim_id, artifact_id, checksum, locator in audit.claim_to_artifact
        )
    else:
        lines.append("| _none: production evidence is pending_ | — | — | — |")
    lines.extend(["", "## Audit", ""])
    if audit.errors:
        lines.extend(f"- `{error}`" for error in audit.errors)
    else:
        lines.append("- All required sources and claim references authenticated.")
    return "\n".join(lines) + "\n"


def report_sha256(report: FinalReportModel) -> str:
    return hashlib.sha256(canonical_json_bytes(report)).hexdigest()


def load_workshop_manifest(path: str | Path) -> WorkshopManifest:
    try:
        return WorkshopManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise FinalReportError(f"could not load workshop manifest: {path}") from error


def _load_sections(path: str | Path) -> tuple[ReportSection, ...]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalReportError(f"could not load report sections: {path}") from error
    if not isinstance(payload, list):
        raise FinalReportError("report sections input must be a JSON list")
    try:
        return tuple(ReportSection.model_validate(item, strict=False) for item in payload)
    except ValidationError as error:
        raise FinalReportError("invalid report section input") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sections", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--markdown-out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parser().parse_args(argv)
    manifest = load_workshop_manifest(args.manifest)
    report, audit = build_final_report(
        manifest,
        _load_sections(args.sections),
        args.artifact_root,
        threats_to_validity=(
            "The workshop evaluates one fixed corpus scale.",
            "Only three seeds are available for learned comparisons.",
        ),
    )
    destination = Path(args.markdown_out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_final_report_markdown(report, audit),
        encoding="utf-8",
        newline="\n",
    )
    require_certified_report(audit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
