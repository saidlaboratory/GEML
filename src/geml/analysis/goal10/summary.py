"""Deterministic Markdown rendering for the bounded Goal 10 audit."""

from __future__ import annotations

from geml.experiments.goal10.rerun_studies import Goal10AuditReport


def render_goal10_summary(report: Goal10AuditReport) -> str:
    """Render denominators, failures, blockers, and Gate G10 without invention."""

    lines = [
        "# Goal 10 compiler-conformance summary",
        "",
        f"- Evidence tier: `{report.evidence_tier.value}`",
        f"- Gate G10: `{report.gate.value}`",
        f"- Retained rows: {report.record_count}",
        "- Scope: opt-in grammar-v2 compiler conformance only",
        "",
        "## Authenticated evidence",
        "",
        f"- Conformance configuration SHA-256: `{report.conformance_config_sha256}`",
        f"- Audit criteria SHA-256: `{report.audit_config_sha256}`",
        f"- Content SHA-256: `{report.conformance_content_sha256}`",
        f"- Manifest SHA-256: `{report.conformance_manifest_sha256}`",
        f"- Implementation commit: `{report.implementation_commit}`",
        f"- Dirty worktree: `{str(report.worktree_dirty).lower()}`",
        "",
        "## Numeric protocol",
        "",
        f"- Decimal precision: {report.precision_digits} digits",
        f"- Per-case timeout: {report.case_timeout_seconds:g} seconds",
        f"- Absolute tolerance: `{report.absolute_tolerance}`",
        f"- Relative tolerance: `{report.relative_tolerance}`",
        f"- Imaginary-component tolerance: `{report.imaginary_tolerance}`",
        (
            "- The `1e-60` acceptance scale leaves 40 decimal guard digits at "
            "100-digit working precision; precision-unit error is error divided "
            "by `10^-precision_digits`, not an IEEE binary ULP."
        ),
        "",
        "## Exact denominators",
        "",
        "| Constructor | Mode | Region | Attempted | Passed | Invalid | Nonfinite input | "
        "Nonfinite result | Compile fail | Numeric fail | Singular | Over | Under | "
        "Timeout | Unsupported | Other fail | Max abs | Max rel | Max precision units |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|",
    ]
    for row in report.numeric_denominators:
        max_absolute = row.maximum_absolute_error or "n/a"
        max_relative = row.maximum_relative_error or "n/a"
        max_precision_units = row.maximum_precision_unit_error or "n/a"
        lines.append(
            f"| {row.constructor} | {row.compiler_mode.value} | {row.region} | "
            f"{row.attempted} | {row.passed} | {row.invalid_domain} | "
            f"{row.nonfinite_input} | {row.nonfinite_result} | "
            f"{row.compile_failure} | {row.numeric_failure} | "
            f"{row.singular} | {row.overflow} | {row.underflow} | {row.timeout} | "
            f"{row.unsupported} | {row.failed} | {max_absolute} | {max_relative} | "
            f"{max_precision_units} |"
        )
    lines.extend(("", "## Frozen v1 compatibility hashes", ""))
    lines.extend(
        (
            f"- Operator registry: `{report.v1_observed.operators_sha256}`",
            f"- Domain registry: `{report.v1_observed.domains_sha256}`",
            f"- Domain rule IDs: `{report.v1_observed.domain_rule_ids_sha256}`",
        )
    )
    for label, fingerprint in sorted(report.v1_observed.transcendental_fingerprints.items()):
        lines.append(f"- `{label}`: `{fingerprint}`")
    lines.extend(("", "## Failures and blockers", ""))
    failures = (
        *report.integrity_failures,
        *(
            (
                f"structural audit failed: {row.record_id} "
                f"({row.failure_type}: {row.failure_message})"
                if row.failure_type is not None
                else f"structural mismatch: {row.record_id}"
            )
            for row in report.structural_rows
            if not row.passed
        ),
        *report.v1_failures,
        *report.coverage_failures,
        *report.conformance_failures,
        *report.ownership_blockers,
    )
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- None.")
    lines.extend(
        (
            "",
            "## Explicitly deferred",
            "",
            *(f"- {claim}" for claim in report.deferred_claims),
            "",
            "No corpus-v2, alpha, DAG, motif, compression, or learned-effect conclusion "
            "is authorized by this audit.",
            "",
        )
    )
    return "\n".join(lines)
