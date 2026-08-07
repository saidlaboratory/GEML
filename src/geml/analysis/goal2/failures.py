"""Deterministic Goal 2 failure mining, survivorship, and top-case selection."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence

import pandas as pd

from geml.verification.eml.numeric import ProbeStatus

_PASS_PROBE_STATUSES = frozenset(
    {ProbeStatus.PASS.value, ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE.value}
)


def message_category(message: object) -> str:
    """Normalize one bounded diagnostic without discarding its retained raw text."""

    if message is None or pd.isna(message):
        return "none"
    lowered = str(message).casefold()
    categories = (
        ("node_limit", ("node count", "node limit")),
        ("depth_limit", ("depth", "depth limit")),
        ("step_limit", ("step limit", "recursion")),
        ("count_mismatch", ("count mismatch", "counted")),
        ("overflow", ("overflow",)),
        ("nonfinite", ("nonfinite", "extended real")),
        ("domain", ("domain", "pole", "singular")),
        ("semantic_mismatch", ("semantic_mismatch", "mismatch")),
        ("unsupported", ("unsupported",)),
        ("parse", ("parse", "srepr")),
        ("validation", ("validation", "contract")),
    )
    for category, needles in categories:
        if any(needle in lowered for needle in needles):
            return category
    return "other"


def _failure_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["error_stage"].notna() | frame["count_status"].ne("success")


def build_failure_tables(frame: pd.DataFrame, *, top_count: int) -> dict[str, pd.DataFrame]:
    """Return detailed, aggregate, survivorship, and deterministic top-case tables."""

    if isinstance(top_count, bool) or not isinstance(top_count, int) or top_count < 1:
        raise ValueError("top_count must be a positive integer")
    required = {
        "expression_id",
        "operator_family",
        "operator_signature",
        "domain_mode",
        "split",
        "iid_ood",
        "ast_node_count",
        "ast_depth",
        "compiler_mode",
        "eml_node_count",
        "eml_depth",
        "tree_alpha_value",
        "compiler_operation_total",
        "processing_elapsed_seconds",
        "count_status",
        "semantic_status",
        "semantic_selected",
        "semantic_unique_assignment_count",
        "materialization_status",
        "semantic_maximum_absolute_error",
        "semantic_probe_results_json",
        "semantic_methods_json",
        "error_stage",
        "error_type",
        "error_message",
        "sympy_srepr",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"failure mining input is missing columns: {missing}")

    enriched = frame.copy()
    enriched["failure"] = _failure_mask(enriched)
    enriched["message_category"] = enriched["error_message"].map(message_category)
    details_columns = [
        "expression_id",
        "error_stage",
        "error_type",
        "message_category",
        "error_message",
        "count_status",
        "semantic_status",
        "semantic_unique_assignment_count",
        "semantic_methods_json",
        "semantic_probe_results_json",
        "compiler_mode",
        "materialization_status",
        "operator_family",
        "operator_signature",
        "ast_node_count",
        "ast_depth",
        "domain_mode",
        "split",
        "iid_ood",
        "semantic_selected",
        "sympy_srepr",
    ]
    details = enriched.loc[enriched["failure"], details_columns].sort_values(
        ["error_stage", "error_type", "message_category", "expression_id"],
        na_position="last",
        kind="mergesort",
    )

    dimensions = [
        "error_stage",
        "error_type",
        "message_category",
        "operator_family",
        "operator_signature",
        "ast_size_bucket",
        "ast_depth",
        "domain_mode",
        "split",
        "semantic_status",
        "semantic_methods_json",
        "compiler_mode",
    ]
    summaries: list[dict[str, object]] = []
    failed = enriched.loc[enriched["failure"]]
    for dimension in dimensions:
        for value, group in failed.groupby(dimension, dropna=False, sort=True):
            summaries.append(
                {
                    "dimension": dimension,
                    "value": "<null>" if pd.isna(value) else str(value),
                    "failure_count": len(group),
                    "all_failure_count": len(failed),
                    "failure_share": len(group) / len(failed) if len(failed) else None,
                }
            )
    summary = pd.DataFrame.from_records(summaries)

    survivorship_rows: list[dict[str, object]] = []
    for dimension in ("operator_family", "ast_size_bucket", "ast_depth", "domain_mode", "split"):
        for value, group in enriched.groupby(dimension, dropna=False, sort=True):
            valid_alpha_count = int(group["tree_alpha_value"].notna().sum())
            excluded = len(group) - valid_alpha_count
            terminal_issue_count = int(group["failure"].sum())
            survivorship_rows.append(
                {
                    "dimension": dimension,
                    "value": "<null>" if pd.isna(value) else str(value),
                    "all_processed_count": len(group),
                    "valid_only_count": valid_alpha_count,
                    "excluded_count": excluded,
                    "excluded_rate": excluded / len(group),
                    "terminal_issue_count": terminal_issue_count,
                    "issue_free_count": len(group) - terminal_issue_count,
                    "terminal_issue_rate": terminal_issue_count / len(group),
                }
            )
    survivorship = pd.DataFrame.from_records(survivorship_rows)

    successful = enriched.loc[enriched["count_status"].eq("success")].copy()
    successful["eml_node_number"] = pd.to_numeric(successful["eml_node_count"], errors="coerce")
    successful["compiler_operation_number"] = pd.to_numeric(
        successful["compiler_operation_total"], errors="coerce"
    )
    top_specs = (
        ("highest_alpha", "tree_alpha_value", False),
        ("largest_eml_node_count", "eml_node_number", False),
        ("largest_eml_depth", "eml_depth", False),
        ("largest_compiler_operation_count", "compiler_operation_number", False),
        ("slowest_successful", "processing_elapsed_seconds", False),
    )
    top_rows: list[pd.DataFrame] = []
    retained_columns = [
        "expression_id",
        "operator_family",
        "operator_signature",
        "domain_mode",
        "split",
        "ast_node_count",
        "ast_depth",
        "eml_node_count",
        "eml_depth",
        "tree_alpha_value",
        "compiler_operation_total",
        "processing_elapsed_seconds",
        "materialization_status",
        "semantic_status",
        "sympy_srepr",
    ]
    for ranking, metric, ascending in top_specs:
        selected = successful.loc[successful[metric].notna()].sort_values(
            [metric, "expression_id"],
            ascending=[ascending, True],
            kind="mergesort",
        )
        selected = selected.head(top_count)[retained_columns].copy()
        selected.insert(0, "rank", range(1, len(selected) + 1))
        selected.insert(0, "ranking", ranking)
        top_rows.append(selected)
    semantic_success = successful.loc[
        successful["semantic_status"].eq("passed")
        & successful["materialization_status"].eq("materialized")
    ].sort_values(["eml_node_number", "expression_id"], ascending=[False, True])
    semantic_success = semantic_success.head(top_count)[retained_columns].copy()
    semantic_success.insert(0, "rank", range(1, len(semantic_success) + 1))
    semantic_success.insert(0, "ranking", "largest_successful_semantic_audit_tree")
    top_rows.append(semantic_success)
    top_explosions = pd.concat(top_rows, ignore_index=True)

    if len(details) != int(enriched["failure"].sum()):
        raise RuntimeError("at least one retained failure row was lost during categorization")
    return {
        "failure_details": details.reset_index(drop=True),
        "failure_summary": summary.reset_index(drop=True),
        "failure_survivorship": survivorship.reset_index(drop=True),
        "top_explosions": top_explosions.reset_index(drop=True),
    }


def failure_status_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Return plot-ready count and semantic terminal-status counts."""

    rows: list[dict[str, object]] = []
    for status_kind, column in (
        ("count", "count_status"),
        ("semantic", "semantic_status"),
        ("materialization", "materialization_status"),
    ):
        counts = frame[column].value_counts(dropna=False, sort=False)
        for status, count in sorted(counts.items(), key=lambda item: str(item[0])):
            rows.append(
                {
                    "status_kind": status_kind,
                    "status": "<null>" if pd.isna(status) else str(status),
                    "count": int(count),
                    "all_processed_count": len(frame),
                    "rate": int(count) / len(frame) if len(frame) else None,
                }
            )
    return pd.DataFrame.from_records(rows)


def semantic_backend_status_counts(
    frame: pd.DataFrame,
    *,
    expected_backends: Sequence[str],
    expected_compiler_mode: str,
) -> pd.DataFrame:
    """Return exhaustive backend/probe-status counts with explicit denominators.

    Probe rates are comparable within a backend. Expression-incidence rates use
    audited, selected, and materialized populations separately. The
    all-processed incidence is retained only as a selection-diluted descriptive
    quantity and must not be interpreted as a corpus-wide semantic failure rate.
    """

    required = {
        "expression_id",
        "semantic_selected",
        "semantic_unique_assignment_count",
        "materialization_status",
        "semantic_probe_results_json",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"semantic taxonomy input is missing columns: {missing}")
    backends = tuple(expected_backends)
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("expected_backends must be nonempty and unique")
    if not expected_compiler_mode.strip():
        raise ValueError("expected_compiler_mode must be nonblank")
    if frame["expression_id"].duplicated().any():
        raise ValueError("semantic taxonomy input contains duplicate expression IDs")

    result_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    status_expression_ids: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    backend_expression_ids: defaultdict[str, set[str]] = defaultdict(set)
    allowed_statuses = {status.value for status in ProbeStatus}

    for row in frame.itertuples(index=False):
        expression_id = str(row.expression_id)
        try:
            results = json.loads(str(row.semantic_probe_results_json))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid semantic probe JSON for expression {expression_id}"
            ) from error
        if not isinstance(results, list):
            raise ValueError(
                f"semantic probe results must be a list for expression {expression_id}"
            )
        if results and row.semantic_selected is not True:
            raise ValueError(
                f"unselected expression {expression_id} contains semantic probe results"
            )
        if results and row.materialization_status != "materialized":
            raise ValueError(
                f"non-materialized expression {expression_id} contains semantic probe results"
            )
        for result in results:
            if not isinstance(result, dict):
                raise ValueError(
                    f"semantic probe result must be an object for expression {expression_id}"
                )
            backend = result.get("backend")
            status = result.get("status")
            compiler_mode = result.get("compiler_mode")
            if backend not in backends:
                raise ValueError(
                    f"unexpected semantic backend {backend!r} for expression {expression_id}"
                )
            if status not in allowed_statuses:
                raise ValueError(
                    f"unexpected semantic probe status {status!r} for expression {expression_id}"
                )
            if compiler_mode != expected_compiler_mode:
                raise ValueError(
                    f"semantic probe compiler mode differs for expression {expression_id}"
                )
            key = (str(backend), str(status))
            result_counts[key] += 1
            status_expression_ids[key].add(expression_id)
            backend_expression_ids[str(backend)].add(expression_id)

    all_processed_count = len(frame)
    selected_count = int(frame["semantic_selected"].eq(True).sum())
    materialized_count = int(frame["materialization_status"].eq("materialized").sum())
    unique_assignments = pd.to_numeric(frame["semantic_unique_assignment_count"], errors="raise")
    selected_indices = frame.index[frame["semantic_selected"].eq(True)]
    materialized_indices = frame.index[frame["materialization_status"].eq("materialized")]
    selected_with_unique_count = int(unique_assignments.loc[selected_indices].gt(0).sum())
    caveat = "selection-diluted descriptive incidence only; not a corpus-wide semantic failure rate"
    rows: list[dict[str, object]] = []
    for backend in backends:
        backend_probe_count = sum(
            count for (name, _), count in result_counts.items() if name == backend
        )
        audited_expression_count = len(backend_expression_ids[backend])
        for status in ProbeStatus:
            key = (backend, status.value)
            probe_count = result_counts[key]
            expression_count = len(status_expression_ids[key])
            rows.append(
                {
                    "backend": backend,
                    "compiler_mode": expected_compiler_mode,
                    "probe_status": status.value,
                    "status_class": ("pass" if status.value in _PASS_PROBE_STATUSES else "failure"),
                    "probe_result_count": probe_count,
                    "backend_probe_result_count": backend_probe_count,
                    "probe_status_rate": (
                        probe_count / backend_probe_count if backend_probe_count else None
                    ),
                    "status_expression_count": expression_count,
                    "backend_audited_expression_count": audited_expression_count,
                    "backend_audited_expression_incidence_rate": (
                        expression_count / audited_expression_count
                        if audited_expression_count
                        else None
                    ),
                    "semantic_selected_count": selected_count,
                    "semantic_unique_assignment_total": int(unique_assignments.sum()),
                    "semantic_selected_with_unique_assignment_count": selected_with_unique_count,
                    "semantic_selected_unique_assignment_coverage_rate": (
                        selected_with_unique_count / selected_count if selected_count else None
                    ),
                    "semantic_unique_assignments_per_selected_mean": (
                        float(unique_assignments.loc[selected_indices].mean())
                        if selected_count
                        else None
                    ),
                    "selected_expression_incidence_rate": (
                        expression_count / selected_count if selected_count else None
                    ),
                    "materialized_expression_count": materialized_count,
                    "semantic_unique_assignments_per_materialized_mean": (
                        float(unique_assignments.loc[materialized_indices].mean())
                        if materialized_count
                        else None
                    ),
                    "materialized_expression_incidence_rate": (
                        expression_count / materialized_count if materialized_count else None
                    ),
                    "all_processed_count": all_processed_count,
                    "all_processed_expression_incidence_rate": (
                        expression_count / all_processed_count if all_processed_count else None
                    ),
                    "all_processed_rate_caveat": caveat,
                }
            )
    return pd.DataFrame.from_records(rows)


def assert_failure_coverage(
    detailed_expression_ids: Iterable[str], expected_expression_ids: Iterable[str]
) -> None:
    """Raise if any expected failure is absent from the saved detail population."""

    detailed = list(detailed_expression_ids)
    expected = set(expected_expression_ids)
    if len(detailed) != len(set(detailed)):
        raise ValueError("failure detail table contains duplicate expression IDs")
    if set(detailed) != expected:
        missing = sorted(expected - set(detailed))
        extra = sorted(set(detailed) - expected)
        raise ValueError(
            "failure detail coverage mismatch: "
            + json.dumps({"missing": missing, "extra": extra}, sort_keys=True)
        )
