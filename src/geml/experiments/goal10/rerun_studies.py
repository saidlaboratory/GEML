"""Exact structure/domain/numeric audit for grammar-v2 conformance.

The historical filename is retained, but this module performs no alpha, DAG,
motif, compression, corpus, or learning rerun.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import mpmath as mp
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from geml.egraph.rules_domain import DOMAIN_RULE_IDS
from geml.eml.compiler_core import CompilerMode
from geml.eml.compiler_transcendental import eml_cosh, eml_sinh, eml_tanh
from geml.eml.compiler_trig import eml_cos, eml_sin, eml_tan
from geml.eml.emitter import emit_eml
from geml.eml.ir import Variable
from geml.eml.validate import validate_pure_eml
from geml.experiments.goal10.corpus_v2 import (
    CONFORMANCE_SCHEMA_VERSION,
    ConformanceCase,
    ConformanceConfig,
    ConformanceManifest,
    ConformanceRecord,
    ConformanceRun,
    ExpectedCaseStatus,
    ProbeStatus,
    VerifierStatus,
    _compile_case,
    _write_immutable_payloads,
    build_conformance_run,
    canonical_json_bytes,
    load_conformance_config,
)
from geml.spec.domains import DOMAIN_POLICIES
from geml.spec.operators import OPERATORS

AUDIT_SCHEMA_VERSION = "geml-goal10-audit-v1"


class GateState(StrEnum):
    """Preregistered Goal 10 gate outcome."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvidenceTier(StrEnum):
    """Whether evidence is a tiny Phase-A fixture or final bounded audit."""

    FIXTURE = "fixture"
    FINAL = "final"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExpectedStructure(_FrozenModel):
    """Independent exact structural expectation for one v2 constructor/mode."""

    constructor: str = Field(pattern=r"^(e|pi|atan|asin|acos)$")
    compiler_mode: CompilerMode
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    node_count: int = Field(ge=1)
    depth: int = Field(ge=0)


class V1Snapshot(_FrozenModel):
    """Canonical byte-level v1 compatibility snapshot."""

    operators_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    domains_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain_rule_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcendental_fingerprints: dict[str, str]


class Goal10AuditConfig(_FrozenModel):
    """Criteria frozen before the bounded audit."""

    schema_version: str
    conformance_schema_version: str
    conformance_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    precision_digits: int = Field(ge=50, le=500)
    absolute_tolerance: str
    relative_tolerance: str
    imaginary_tolerance: str
    required_constructors: tuple[str, ...]
    required_modes: tuple[CompilerMode, ...]
    expected_structures: tuple[ExpectedStructure, ...]
    expected_v1: V1Snapshot
    known_ownership_blockers: tuple[str, ...]

    @model_validator(mode="after")
    def validate_criteria(self) -> Goal10AuditConfig:
        if self.schema_version != AUDIT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {AUDIT_SCHEMA_VERSION!r}")
        if self.conformance_schema_version != CONFORMANCE_SCHEMA_VERSION:
            raise ValueError("conformance schema version does not match the owned producer")
        if self.required_constructors != ("e", "pi", "atan", "asin", "acos"):
            raise ValueError("required_constructors must contain the five v2 additions")
        if self.required_modes != tuple(CompilerMode):
            raise ValueError("required_modes must contain both compiler modes")
        expected_cells = {
            (constructor, mode)
            for constructor in self.required_constructors
            for mode in self.required_modes
        }
        observed_cells = {(row.constructor, row.compiler_mode) for row in self.expected_structures}
        if observed_cells != expected_cells or len(self.expected_structures) != len(expected_cells):
            raise ValueError("expected_structures must contain every constructor/mode cell once")
        with mp.workdps(self.precision_digits):
            for field_name in (
                "absolute_tolerance",
                "relative_tolerance",
                "imaginary_tolerance",
            ):
                value = mp.mpf(getattr(self, field_name))
                if not mp.isfinite(value) or value < 0:
                    raise ValueError(f"{field_name} must be finite and nonnegative")
        return self


class StructuralAuditRow(_FrozenModel):
    """Recomputed structure and purity result for one compiled record."""

    record_id: str
    constructor: str
    compiler_mode: CompilerMode
    fingerprint_matches_record: bool
    fingerprint_matches_preregistered: bool
    node_count_matches: bool
    depth_matches: bool
    strict_purity_passed: bool
    failure_type: str | None = None
    failure_message: str | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> StructuralAuditRow:
        if (self.failure_type is None) != (self.failure_message is None):
            raise ValueError("structural failure type and message must be set together")
        return self

    @property
    def passed(self) -> bool:
        return self.failure_type is None and all(
            (
                self.fingerprint_matches_record,
                self.fingerprint_matches_preregistered,
                self.node_count_matches,
                self.depth_matches,
                self.strict_purity_passed,
            )
        )


@dataclass(frozen=True)
class _StructuralEvidence:
    fingerprint: str | None
    node_count: int | None
    depth: int | None
    strict_purity_passed: bool
    failure_type: str | None = None
    failure_message: str | None = None


class NumericDenominator(_FrozenModel):
    """One exact operator/mode/region denominator with retained failures."""

    constructor: str
    compiler_mode: CompilerMode
    region: str
    attempted: int
    passed: int
    invalid_domain: int
    nonfinite_input: int
    nonfinite_result: int
    compile_failure: int
    numeric_failure: int
    singular: int
    overflow: int
    underflow: int
    timeout: int
    unsupported: int
    failed: int
    maximum_absolute_error: str | None
    maximum_relative_error: str | None
    maximum_precision_unit_error: str | None

    @model_validator(mode="after")
    def validate_denominator(self) -> NumericDenominator:
        if self.attempted != (
            self.passed
            + self.invalid_domain
            + self.nonfinite_input
            + self.nonfinite_result
            + self.compile_failure
            + self.numeric_failure
            + self.singular
            + self.overflow
            + self.underflow
            + self.timeout
            + self.unsupported
            + self.failed
        ):
            raise ValueError("numeric denominator components do not sum to attempted")
        return self


class Goal10AuditReport(_FrozenModel):
    """Complete bounded audit and preregistered Gate G10 result."""

    schema_version: str = AUDIT_SCHEMA_VERSION
    evidence_tier: EvidenceTier
    gate: GateState
    record_count: int
    precision_digits: int
    case_timeout_seconds: float
    absolute_tolerance: str
    relative_tolerance: str
    imaginary_tolerance: str
    conformance_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conformance_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conformance_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_commit: str
    worktree_dirty: bool
    integrity_failures: tuple[str, ...]
    structural_rows: tuple[StructuralAuditRow, ...]
    numeric_denominators: tuple[NumericDenominator, ...]
    v1_observed: V1Snapshot
    v1_failures: tuple[str, ...]
    coverage_failures: tuple[str, ...]
    conformance_failures: tuple[str, ...]
    ownership_blockers: tuple[str, ...]
    deferred_claims: tuple[str, ...] = (
        "corpus-v2 effects",
        "pure-EML alpha effects",
        "AST/EML DAG effects",
        "macro/motif compression effects",
        "Goal 6/7 learned effects",
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compute_v1_snapshot() -> V1Snapshot:
    """Compute independent canonical registry/rule/formula v1 identities."""

    fingerprints: dict[str, str] = {}
    compilers = {
        "sin": eml_sin,
        "cos": eml_cos,
        "tan": eml_tan,
        "sinh": eml_sinh,
        "cosh": eml_cosh,
        "tanh": eml_tanh,
    }
    for mode in CompilerMode:
        for name, compiler in compilers.items():
            tree = compiler(Variable("x"), mode=mode)
            fingerprints[f"{mode.value}:{name}"] = _sha256(emit_eml(tree).encode("utf-8"))
    return V1Snapshot(
        operators_sha256=_sha256(
            canonical_json_bytes([row.model_dump(mode="json") for row in OPERATORS])
        ),
        domains_sha256=_sha256(
            canonical_json_bytes([row.model_dump(mode="json") for row in DOMAIN_POLICIES])
        ),
        domain_rule_ids_sha256=_sha256(canonical_json_bytes(list(DOMAIN_RULE_IDS))),
        transcendental_fingerprints=fingerprints,
    )


def _verify_run_integrity(
    run: ConformanceRun,
    conformance_config: ConformanceConfig,
    audit_config: Goal10AuditConfig,
    evidence_tier: EvidenceTier,
) -> tuple[str, ...]:
    """Authenticate stored bytes and compare them with a fresh exact rerun."""

    failures: list[str] = []
    if run.manifest.record_count != len(run.records):
        failures.append("manifest record_count disagrees with records")
    content = b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in run.records
    )
    if _sha256(content) != run.manifest.content_sha256:
        failures.append("content SHA-256 mismatch")
    config_hash = _sha256(canonical_json_bytes(conformance_config.model_dump(mode="json")))
    if config_hash != run.manifest.config_sha256:
        failures.append("configuration SHA-256 mismatch")
    if config_hash != audit_config.conformance_config_sha256:
        failures.append("configuration SHA-256 disagrees with preregistered audit criteria")
    manifest_payload = run.manifest.model_dump(
        mode="json",
        exclude={"manifest_sha256"},
    )
    if _sha256(canonical_json_bytes(manifest_payload)) != run.manifest.manifest_sha256:
        failures.append("manifest SHA-256 mismatch")
    if run.manifest.base_commit != conformance_config.base_commit:
        failures.append("base commit disagrees with conformance configuration")
    protocol_fields = (
        "precision_digits",
        "absolute_tolerance",
        "relative_tolerance",
        "imaginary_tolerance",
    )
    for field_name in protocol_fields:
        if getattr(conformance_config, field_name) != getattr(audit_config, field_name):
            failures.append(f"conformance {field_name} disagrees with preregistered audit criteria")

    try:
        expected_run = build_conformance_run(conformance_config)
    except Exception as error:
        expected_run = None
        message = str(error).strip() or "no error message"
        failures.append(f"deterministic recomputation failed ({type(error).__name__}: {message})")
    if expected_run is not None:
        observed_roster = tuple(
            (
                record.record_id,
                record.case_name,
                record.constructor,
                record.input_text,
                record.region,
                record.compiler_mode,
            )
            for record in run.records
        )
        expected_roster = tuple(
            (
                record.record_id,
                record.case_name,
                record.constructor,
                record.input_text,
                record.region,
                record.compiler_mode,
            )
            for record in expected_run.records
        )
        if observed_roster != expected_roster:
            failures.append("record roster/order disagrees with the configured case-by-mode matrix")
        if run.records != expected_run.records:
            failures.append("record outcomes disagree with deterministic numeric recomputation")

        comparable_manifest = run.manifest.model_dump(
            mode="json",
            exclude={"manifest_sha256", "runtime_metadata"},
        )
        expected_manifest = expected_run.manifest.model_dump(
            mode="json",
            exclude={"manifest_sha256", "runtime_metadata"},
        )
        if comparable_manifest != expected_manifest:
            failures.append("manifest counters or labels disagree with deterministic recomputation")
        if run.manifest.runtime_metadata != expected_run.manifest.runtime_metadata:
            failures.append("runtime metadata disagrees with the fresh audit environment")
    required_runtime_fields = {
        "python",
        "implementation",
        "geml",
        "mpmath",
        "pydantic",
        "pyyaml",
        "platform",
        "machine",
        "processor",
    }
    if set(run.manifest.runtime_metadata) != required_runtime_fields or any(
        not value for value in run.manifest.runtime_metadata.values()
    ):
        failures.append("runtime metadata is incomplete")
    if evidence_tier is EvidenceTier.FINAL:
        if run.manifest.implementation_commit == "unknown":
            failures.append("final evidence has no known implementation commit")
        if run.manifest.worktree_dirty:
            failures.append("final evidence was produced from a dirty worktree")
    return tuple(failures)


def _case_from_record(record: ConformanceRecord) -> ConformanceCase:
    return ConformanceCase(
        case_name=record.case_name,
        constructor=record.constructor,
        input_text=record.input_text,
        region=record.region,
        expected_status=record.expected_status,
    )


def _expected_structure_map(
    config: Goal10AuditConfig,
) -> dict[tuple[str, CompilerMode], ExpectedStructure]:
    return {(row.constructor, row.compiler_mode): row for row in config.expected_structures}


def _recompute_structure(
    case: ConformanceCase,
    mode: CompilerMode,
) -> _StructuralEvidence:
    try:
        tree = _compile_case(case, mode)
        statistics = validate_pure_eml(tree)
        return _StructuralEvidence(
            fingerprint=_sha256(emit_eml(tree).encode("utf-8")),
            node_count=statistics.node_count,
            depth=statistics.depth,
            strict_purity_passed=True,
        )
    except Exception as error:
        return _StructuralEvidence(
            fingerprint=None,
            node_count=None,
            depth=None,
            strict_purity_passed=False,
            failure_type=type(error).__name__,
            failure_message=str(error).strip() or "no error message",
        )


def _audit_structure(
    records: tuple[ConformanceRecord, ...],
    config: Goal10AuditConfig,
) -> tuple[StructuralAuditRow, ...]:
    expected = _expected_structure_map(config)
    evidence_by_constructor: dict[tuple[str, CompilerMode], _StructuralEvidence] = {}
    rows: list[StructuralAuditRow] = []
    for record in records:
        if record.pure_eml_fingerprint is None:
            continue
        key = (record.constructor, record.compiler_mode)
        if key not in evidence_by_constructor:
            evidence_by_constructor[key] = _recompute_structure(
                _case_from_record(record),
                record.compiler_mode,
            )
        evidence = evidence_by_constructor[key]
        preregistered = expected.get(key)
        # Compositions are checked against their retained record identity but
        # are not one of the five constructor goldens.
        preregistered_match = evidence.fingerprint is not None and (
            preregistered is None or evidence.fingerprint == preregistered.fingerprint
        )
        expected_nodes = record.node_count if preregistered is None else preregistered.node_count
        expected_depth = record.depth if preregistered is None else preregistered.depth
        rows.append(
            StructuralAuditRow(
                record_id=record.record_id,
                constructor=record.constructor,
                compiler_mode=record.compiler_mode,
                fingerprint_matches_record=(evidence.fingerprint == record.pure_eml_fingerprint),
                fingerprint_matches_preregistered=preregistered_match,
                node_count_matches=(evidence.node_count == record.node_count == expected_nodes),
                depth_matches=evidence.depth == record.depth == expected_depth,
                strict_purity_passed=evidence.strict_purity_passed,
                failure_type=evidence.failure_type,
                failure_message=evidence.failure_message,
            )
        )
    return tuple(rows)


def _numeric_denominators(
    records: tuple[ConformanceRecord, ...],
) -> tuple[NumericDenominator, ...]:
    groups: dict[tuple[str, CompilerMode, str], list[ConformanceRecord]] = defaultdict(list)
    for record in records:
        groups[(record.constructor, record.compiler_mode, record.region.value)].append(record)
    summaries: list[NumericDenominator] = []
    for (constructor, mode, region), rows in sorted(groups.items()):
        passed = sum(row.verifier_status is VerifierStatus.PASSED for row in rows)
        invalid = sum(row.observed_status is ProbeStatus.INVALID_DOMAIN for row in rows)
        nonfinite = sum(row.observed_status is ProbeStatus.NONFINITE_INPUT for row in rows)
        nonfinite_result = sum(row.observed_status is ProbeStatus.NONFINITE_RESULT for row in rows)
        compile_failure = sum(row.observed_status is ProbeStatus.COMPILE_FAILURE for row in rows)
        numeric_failure = sum(row.observed_status is ProbeStatus.NUMERIC_FAILURE for row in rows)
        singular = sum(row.observed_status is ProbeStatus.SINGULAR for row in rows)
        overflow = sum(row.observed_status is ProbeStatus.OVERFLOW for row in rows)
        underflow = sum(row.observed_status is ProbeStatus.UNDERFLOW for row in rows)
        timeout = sum(row.observed_status is ProbeStatus.TIMEOUT for row in rows)
        unsupported = sum(row.observed_status is ProbeStatus.UNSUPPORTED for row in rows)
        failed = (
            len(rows)
            - passed
            - invalid
            - nonfinite
            - nonfinite_result
            - compile_failure
            - numeric_failure
            - singular
            - overflow
            - underflow
            - timeout
            - unsupported
        )
        numeric = [row.numeric for row in rows if row.numeric is not None]
        with mp.workdps(max((item.precision_digits for item in numeric), default=50)):
            absolute = [mp.mpf(item.absolute_error) for item in numeric]
            relative = [mp.mpf(item.relative_error) for item in numeric]
            precision_units = [mp.mpf(item.precision_unit_error) for item in numeric]
            summaries.append(
                NumericDenominator(
                    constructor=constructor,
                    compiler_mode=mode,
                    region=region,
                    attempted=len(rows),
                    passed=passed,
                    invalid_domain=invalid,
                    nonfinite_input=nonfinite,
                    nonfinite_result=nonfinite_result,
                    compile_failure=compile_failure,
                    numeric_failure=numeric_failure,
                    singular=singular,
                    overflow=overflow,
                    underflow=underflow,
                    timeout=timeout,
                    unsupported=unsupported,
                    failed=failed,
                    maximum_absolute_error=(mp.nstr(max(absolute), 20) if absolute else None),
                    maximum_relative_error=(mp.nstr(max(relative), 20) if relative else None),
                    maximum_precision_unit_error=(
                        mp.nstr(max(precision_units), 20) if precision_units else None
                    ),
                )
            )
    return tuple(summaries)


def _compare_v1(observed: V1Snapshot, expected: V1Snapshot) -> tuple[str, ...]:
    failures: list[str] = []
    for field_name in (
        "operators_sha256",
        "domains_sha256",
        "domain_rule_ids_sha256",
    ):
        if getattr(observed, field_name) != getattr(expected, field_name):
            failures.append(f"v1 {field_name} drift")
    if observed.transcendental_fingerprints != expected.transcendental_fingerprints:
        differing = sorted(
            key
            for key in set(observed.transcendental_fingerprints)
            | set(expected.transcendental_fingerprints)
            if observed.transcendental_fingerprints.get(key)
            != expected.transcendental_fingerprints.get(key)
        )
        failures.extend(f"v1 transcendental fingerprint drift: {key}" for key in differing)
    return tuple(failures)


def _coverage_failures(
    records: tuple[ConformanceRecord, ...],
    config: Goal10AuditConfig,
) -> tuple[str, ...]:
    observed = {(row.constructor, row.compiler_mode) for row in records}
    expected = {
        (constructor, mode)
        for constructor in config.required_constructors
        for mode in config.required_modes
    }
    missing = sorted(expected - observed)
    failures = [f"missing constructor/mode cell: {name}/{mode.value}" for name, mode in missing]
    for status in (ExpectedCaseStatus.INVALID_DOMAIN, ExpectedCaseStatus.NONFINITE):
        if not any(row.expected_status is status for row in records):
            failures.append(f"no retained {status.value} case")
    if not any(row.observed_status is ProbeStatus.UNSUPPORTED for row in records):
        failures.append("no retained unsupported measurement case")
    return tuple(failures)


def audit_conformance(
    run: ConformanceRun,
    *,
    conformance_config: ConformanceConfig,
    audit_config: Goal10AuditConfig,
    evidence_tier: EvidenceTier,
) -> Goal10AuditReport:
    """Audit a bounded set and determine Gate G10 without scientific overclaim."""

    integrity_failures = _verify_run_integrity(
        run,
        conformance_config,
        audit_config,
        evidence_tier,
    )
    structural_rows = _audit_structure(run.records, audit_config)
    structural_failures = tuple(
        (
            f"structural audit failed: {row.record_id} ({row.failure_type}: {row.failure_message})"
            if row.failure_type is not None
            else f"structural mismatch: {row.record_id}"
        )
        for row in structural_rows
        if not row.passed
    )
    observed_v1 = compute_v1_snapshot()
    v1_failures = _compare_v1(observed_v1, audit_config.expected_v1)
    coverage_failures = _coverage_failures(run.records, audit_config)
    conformance_failures = tuple(
        f"{row.record_id}: {row.observed_status.value}"
        for row in run.records
        if row.expected_status is ExpectedCaseStatus.VALID
        and row.observed_status
        not in {
            ProbeStatus.PASS,
            ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE,
            # A declared measurement limitation is retained, not converted to
            # a numeric pass.  It keeps the gate insufficient below.
            ProbeStatus.UNSUPPORTED,
        }
    )
    hard_failures = (
        *integrity_failures,
        *structural_failures,
        *v1_failures,
        *coverage_failures,
        *conformance_failures,
    )
    if hard_failures:
        gate = GateState.FAIL
    elif (
        evidence_tier is EvidenceTier.FIXTURE
        or audit_config.known_ownership_blockers
        or any(
            row.expected_status is ExpectedCaseStatus.VALID
            and row.observed_status is ProbeStatus.UNSUPPORTED
            for row in run.records
        )
    ):
        gate = GateState.INSUFFICIENT_EVIDENCE
    else:
        gate = GateState.PASS
    return Goal10AuditReport(
        evidence_tier=evidence_tier,
        gate=gate,
        record_count=len(run.records),
        precision_digits=audit_config.precision_digits,
        case_timeout_seconds=conformance_config.case_timeout_seconds,
        absolute_tolerance=audit_config.absolute_tolerance,
        relative_tolerance=audit_config.relative_tolerance,
        imaginary_tolerance=audit_config.imaginary_tolerance,
        conformance_config_sha256=run.manifest.config_sha256,
        audit_config_sha256=_sha256(canonical_json_bytes(audit_config.model_dump(mode="json"))),
        conformance_content_sha256=run.manifest.content_sha256,
        conformance_manifest_sha256=run.manifest.manifest_sha256,
        implementation_commit=run.manifest.implementation_commit,
        worktree_dirty=run.manifest.worktree_dirty,
        integrity_failures=integrity_failures,
        structural_rows=structural_rows,
        numeric_denominators=_numeric_denominators(run.records),
        v1_observed=observed_v1,
        v1_failures=v1_failures,
        coverage_failures=coverage_failures,
        conformance_failures=conformance_failures,
        ownership_blockers=audit_config.known_ownership_blockers,
    )


def load_audit_config(path: str | Path) -> Goal10AuditConfig:
    """Load and strictly validate the preregistered Gate G10 criteria."""

    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return Goal10AuditConfig.model_validate(payload)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"could not load Goal 10 audit config: {path}") from error


def load_conformance_run(path: str | Path) -> ConformanceRun:
    """Load the immutable two-file conformance output without repairing rows."""

    root = Path(path)
    try:
        manifest = ConformanceManifest.model_validate_json(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        records = tuple(
            ConformanceRecord.model_validate_json(line)
            for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return ConformanceRun(manifest=manifest, records=records)
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise ValueError(f"could not load Goal 10 conformance run: {root}") from error


def write_audit_report(
    report: Goal10AuditReport,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Publish immutable machine-readable and Markdown Gate G10 evidence."""

    from geml.analysis.goal10.summary import render_goal10_summary

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root / "audit.json"
    summary_path = root / "GOAL10_SUMMARY.md"
    outputs = {
        audit_path: canonical_json_bytes(report.model_dump(mode="json")) + b"\n",
        summary_path: (render_goal10_summary(report).rstrip() + "\n").encode("utf-8"),
    }
    _write_immutable_payloads(
        outputs,
        error_type=FileExistsError,
        description="Goal 10 evidence",
    )
    return audit_path, summary_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded authenticated audit; never start a corpus/learning job."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conformance-config", required=True, type=Path)
    parser.add_argument("--audit-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--evidence-tier",
        choices=tuple(item.value for item in EvidenceTier),
        required=True,
    )
    arguments = parser.parse_args(argv)
    report = audit_conformance(
        load_conformance_run(arguments.run_dir),
        conformance_config=load_conformance_config(arguments.conformance_config),
        audit_config=load_audit_config(arguments.audit_config),
        evidence_tier=EvidenceTier(arguments.evidence_tier),
    )
    write_audit_report(report, arguments.output_dir)
    print(report.model_dump_json(indent=2))
    return {
        GateState.PASS: 0,
        GateState.FAIL: 1,
        GateState.INSUFFICIENT_EVIDENCE: 2,
    }[report.gate]


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
