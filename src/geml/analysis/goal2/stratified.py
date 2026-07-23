"""Validated, deterministic stratified analysis for Goal 2 expansion metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from geml.analysis.goal2.alpha import strict_threshold_pass
from geml.analysis.goal2.failures import (
    assert_failure_coverage,
    build_failure_tables,
    failure_status_counts,
    semantic_backend_status_counts,
)
from geml.data.storage.shards import sha256_file
from geml.eml.compiler_core import CompilerMode
from geml.experiments.goal2.run import (
    Goal2ArtifactError,
    LoadedGoal2Config,
    load_goal2_config,
    validate_metrics_manifest,
)

ANALYSIS_SCHEMA_VERSION = "geml-goal2-analysis-v1"
QUANTILES = (0.10, 0.25, 0.75, 0.90, 0.95, 0.99)
_ANALYSIS_TABLE_NAMES = frozenset(
    {
        "alpha_histogram",
        "alpha_log_histogram",
        "failure_details",
        "failure_status_counts",
        "failure_summary",
        "failure_survivorship",
        "overall",
        "scaling",
        "scatter_sample",
        "semantic_backend_status_counts",
        "stability_distribution",
        "stability_family",
        "stability_overall",
        "stratified",
        "thresholds",
        "top_explosions",
    }
)
_INPUT_COLUMNS = (
    "expression_id",
    "split",
    "iid_ood",
    "operator_family",
    "operator_signature",
    "source_operator_counts_json",
    "domain_mode",
    "variables_json",
    "variable_count",
    "source_constant_counts_json",
    "source_constant_count",
    "sympy_srepr",
    "target_ast_size",
    "target_depth",
    "ast_node_count",
    "ast_depth",
    "compiler_mode",
    "eml_node_count",
    "eml_depth",
    "compiler_operation_total",
    "tree_alpha_exact_ratio",
    "tree_alpha_value",
    "threshold_outcomes_json",
    "processing_status",
    "count_status",
    "semantic_selected",
    "semantic_unique_assignment_count",
    "materialization_status",
    "semantic_status",
    "semantic_maximum_absolute_error",
    "semantic_probe_results_json",
    "semantic_methods_json",
    "processing_elapsed_seconds",
    "error_stage",
    "error_type",
    "error_message",
)


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _analysis_source_fingerprint() -> str:
    """Hash every local executable dependency that defines or admits saved tables."""

    digest = hashlib.sha256()
    goal2_directory = Path(__file__).resolve().parent
    geml_directory = goal2_directory.parents[1]
    for label, path in (
        ("stratified.py", goal2_directory / "stratified.py"),
        ("failures.py", goal2_directory / "failures.py"),
        ("alpha.py", goal2_directory / "alpha.py"),
        ("run.py", geml_directory / "experiments/goal2/run.py"),
    ):
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-goal2-analysis-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise Goal2ArtifactError(f"analysis artifact already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    _atomic_bytes(path, payload)


def _atomic_table(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-goal2-table-", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, temporary, compression="zstd", data_page_version="2.0")
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        checksum = sha256_file(temporary)
        byte_count = temporary.stat().st_size
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or sha256_file(path) != checksum:
                raise Goal2ArtifactError(
                    f"analysis table already exists with different content: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": path.as_posix(),
        "row_count": len(frame),
        "byte_count": byte_count,
        "checksum": {"algorithm": "sha256", "digest": checksum},
    }


def validate_analysis_manifest(path: str | Path) -> dict[str, Any]:
    """Validate one completed analysis manifest and every saved source table."""

    manifest_path = Path(path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise Goal2ArtifactError(f"invalid analysis manifest: {manifest_path}") from error
    if manifest.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise Goal2ArtifactError("analysis manifest schema version is unsupported")
    if manifest.get("compiler_mode") != CompilerMode.OFFICIAL_V4.value:
        raise Goal2ArtifactError("analysis manifest compiler mode is not official_v4")
    if not _is_sha256(manifest.get("config_hash")):
        raise Goal2ArtifactError("analysis manifest config hash is invalid")
    if not _is_sha256(manifest.get("metrics_runner_fingerprint")):
        raise Goal2ArtifactError("analysis metrics runner fingerprint is invalid")
    if manifest.get("analysis_source_fingerprint") != _analysis_source_fingerprint():
        raise Goal2ArtifactError("analysis source fingerprint differs from the current code")
    for label in ("final", "pilot"):
        source = manifest.get(f"{label}_metrics_manifest")
        if not isinstance(source, str) or not source.strip() or not Path(source).is_absolute():
            raise Goal2ArtifactError(f"analysis {label} input path is invalid")
        expected_source_hash = manifest.get(f"{label}_metrics_manifest_sha256")
        if not _is_sha256(expected_source_hash):
            raise Goal2ArtifactError(f"analysis {label} input hash is invalid")
        source_path = Path(source)
        if not source_path.is_file() or sha256_file(source_path) != expected_source_hash:
            raise Goal2ArtifactError(f"analysis {label} input manifest has changed or is missing")
    if not isinstance(manifest.get("pilot_label"), str) or not manifest["pilot_label"].strip():
        raise Goal2ArtifactError("analysis pilot label is invalid")
    if manifest.get("quantile_method") != "linear" or manifest.get("standard_deviation_ddof") != 0:
        raise Goal2ArtifactError("analysis statistic metadata is invalid")
    tables = manifest.get("tables")
    if not isinstance(tables, dict) or set(tables) != _ANALYSIS_TABLE_NAMES:
        raise Goal2ArtifactError("analysis manifest table set is incomplete or unexpected")
    resolved_paths: set[Path] = set()
    for name, artifact in tables.items():
        if not isinstance(name, str) or not isinstance(artifact, dict):
            raise Goal2ArtifactError("analysis table metadata is malformed")
        relative = Path(str(artifact.get("path")))
        if relative.is_absolute() or relative.as_posix() != f"tables/{name}.parquet":
            raise Goal2ArtifactError(f"analysis table path is invalid: {name}")
        table_path = (manifest_path.parent / relative).resolve()
        try:
            table_path.relative_to(manifest_path.parent.resolve())
        except ValueError as error:
            raise Goal2ArtifactError("analysis table path escapes its root") from error
        if table_path in resolved_paths:
            raise Goal2ArtifactError("analysis table paths are not unique")
        resolved_paths.add(table_path)
        if not table_path.is_file():
            raise Goal2ArtifactError(f"missing analysis table: {name}")
        checksum = artifact.get("checksum")
        if (
            not isinstance(checksum, dict)
            or checksum.get("algorithm") != "sha256"
            or not _is_sha256(checksum.get("digest"))
        ):
            raise Goal2ArtifactError(f"invalid checksum metadata for analysis table: {name}")
        if sha256_file(table_path) != checksum.get("digest"):
            raise Goal2ArtifactError(f"analysis table checksum mismatch: {name}")
        byte_count = artifact.get("byte_count")
        row_count = artifact.get("row_count")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 1
            or table_path.stat().st_size != byte_count
        ):
            raise Goal2ArtifactError(f"analysis table byte count mismatch: {name}")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 0
            or pq.read_metadata(table_path).num_rows != row_count
        ):
            raise Goal2ArtifactError(f"analysis table row count mismatch: {name}")
    return manifest


def _load_metrics(path: Path, *, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = validate_metrics_manifest(path)
    if manifest.get("compiler_mode") != CompilerMode.OFFICIAL_V4.value:
        raise Goal2ArtifactError(f"{label} primary metrics are not official_v4")
    frames = [
        pq.read_table(path.parent / shard["path"], columns=list(_INPUT_COLUMNS)).to_pandas()
        for shard in manifest["shards"]
    ]
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != manifest["processed_count"]:
        raise Goal2ArtifactError(f"{label} metrics row count differs from its manifest")
    if frame["expression_id"].duplicated().any():
        raise Goal2ArtifactError(f"{label} metrics contain duplicate expression IDs")
    frame.insert(0, "dataset", label)
    return frame, manifest


def _validate_metrics_role(
    manifest: Mapping[str, object],
    loaded: LoadedGoal2Config,
    *,
    expected_stage: str,
) -> None:
    """Require an input manifest to match its exact configured analysis role."""

    policy = loaded.config.stages[expected_stage]
    metadata = manifest.get("run_metadata")
    if manifest.get("stage") != expected_stage:
        raise Goal2ArtifactError(
            f"{expected_stage} analysis input has stage {manifest.get('stage')!r}"
        )
    if manifest.get("processed_count") != policy.expected_count:
        raise Goal2ArtifactError(
            f"{expected_stage} analysis input count differs from configured expected_count"
        )
    if not isinstance(metadata, dict) or metadata.get("source_label") != policy.source_label:
        raise Goal2ArtifactError(
            f"{expected_stage} analysis input source_label differs from configuration"
        )


def ast_size_bucket(value: object, buckets: Sequence[Sequence[int]]) -> str:
    """Return the frozen inclusive AST-size bucket label for one exact count."""

    if value is None or pd.isna(value):
        return "missing"
    count = int(value)
    for minimum, maximum in buckets:
        if minimum <= count <= maximum:
            return f"{minimum}-{maximum}"
    return "out_of_policy"


def _constant_profile(value: object) -> tuple[str, int | None]:
    if value is None or pd.isna(value):
        return "missing", None
    try:
        counts = json.loads(str(value))
    except (TypeError, ValueError) as error:
        raise Goal2ArtifactError("canonical AST constant counts are invalid JSON") from error
    if not isinstance(counts, dict) or any(
        not isinstance(name, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in counts.items()
    ):
        raise Goal2ArtifactError("canonical AST constant counts are malformed")
    present = sorted(name for name, count in counts.items() if count > 0)
    return ("+".join(present) if present else "none"), sum(counts.values())


def _prepare(frame: pd.DataFrame, loaded: LoadedGoal2Config) -> pd.DataFrame:
    prepared = frame.copy()
    buckets = loaded.config.analysis.ast_size_buckets
    prepared["ast_size_bucket"] = prepared["ast_node_count"].map(
        lambda value: ast_size_bucket(value, buckets)
    )
    constant_profiles = [
        _constant_profile(value) for value in prepared["source_constant_counts_json"]
    ]
    reported_counts = pd.to_numeric(prepared["source_constant_count"], errors="coerce")
    for (_, derived_count), reported_count in zip(constant_profiles, reported_counts, strict=True):
        if derived_count is None:
            if pd.notna(reported_count):
                raise Goal2ArtifactError("missing canonical AST constant JSON has a count")
        elif (
            pd.isna(reported_count)
            or not float(reported_count).is_integer()
            or int(reported_count) != derived_count
        ):
            raise Goal2ArtifactError("canonical AST constant count does not reconcile")
    prepared["canonical_ast_constant_category"] = [category for category, _ in constant_profiles]
    prepared["canonical_ast_constant_count"] = [count for _, count in constant_profiles]
    prepared["count_semantic_status"] = (
        prepared["count_status"].astype(str) + "|" + prepared["semantic_status"].astype(str)
    )
    prepared["eml_node_number"] = pd.to_numeric(prepared["eml_node_count"], errors="coerce")
    prepared["compiler_operation_number"] = pd.to_numeric(
        prepared["compiler_operation_total"], errors="coerce"
    )
    prepared["eml_log10_bucket"] = prepared["eml_node_number"].map(
        lambda value: (
            f"10^{math.floor(math.log10(float(value)))}"
            if pd.notna(value) and value > 0
            else "missing"
        )
    )
    prepared["failure"] = prepared["error_stage"].notna() | prepared["count_status"].ne("success")
    return prepared


def _numeric_summary(values: Iterable[object], *, quantile_method: str) -> dict[str, object]:
    array = np.asarray([float(value) for value in values if value is not None and pd.notna(value)])
    if array.size == 0:
        return {
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    quantiles = np.quantile(array, QUANTILES, method=quantile_method)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array, ddof=0)),
        "minimum": float(np.min(array)),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "maximum": float(np.max(array)),
    }


def summarize_population(
    frame: pd.DataFrame, *, quantile_method: str, population: str
) -> dict[str, object]:
    """Summarize explicit all-processed or valid-alpha denominators."""

    valid = frame.loc[frame["tree_alpha_value"].notna()]
    selected = frame if population == "all_processed" else valid
    semantic_selected = frame.loc[frame["semantic_selected"].eq(True)]
    materialized = frame.loc[frame["materialization_status"].eq("materialized")]
    unique_assignments = pd.to_numeric(frame["semantic_unique_assignment_count"], errors="raise")
    alpha = _numeric_summary(valid["tree_alpha_value"], quantile_method=quantile_method)
    return {
        "population": population,
        "population_count": len(selected),
        "all_processed_count": len(frame),
        "count_success_count": int(frame["count_status"].eq("success").sum()),
        "count_failure_count": int(frame["count_status"].ne("success").sum()),
        "semantic_status_counts_json": _json_text(
            dict(sorted(Counter(frame["semantic_status"]).items()))
        ),
        "semantic_unique_assignment_total": int(unique_assignments.sum()),
        "semantic_selected_with_unique_assignment_count": int(
            unique_assignments.loc[semantic_selected.index].gt(0).sum()
        ),
        "semantic_selected_unique_assignment_coverage_rate": (
            float(unique_assignments.loc[semantic_selected.index].gt(0).mean())
            if len(semantic_selected)
            else None
        ),
        "semantic_unique_assignments_per_selected_mean": (
            float(unique_assignments.loc[semantic_selected.index].mean())
            if len(semantic_selected)
            else None
        ),
        "semantic_unique_assignments_per_materialized_mean": (
            float(unique_assignments.loc[materialized.index].mean()) if len(materialized) else None
        ),
        "valid_alpha_count": len(valid),
        "invalid_alpha_count": len(frame) - len(valid),
        "alpha_statistic_denominator": len(valid),
        **{f"alpha_{name}": value for name, value in alpha.items()},
        "ast_nodes_mean": float(valid["ast_node_count"].mean()) if len(valid) else None,
        "ast_nodes_median": float(valid["ast_node_count"].median()) if len(valid) else None,
        "eml_nodes_mean": float(valid["eml_node_number"].mean()) if len(valid) else None,
        "eml_nodes_median": float(valid["eml_node_number"].median()) if len(valid) else None,
        "eml_depth_mean": float(valid["eml_depth"].mean()) if len(valid) else None,
        "eml_depth_median": float(valid["eml_depth"].median()) if len(valid) else None,
    }


def build_stratified_table(
    frame: pd.DataFrame, *, quantile_method: str, minimum_group_count: int
) -> pd.DataFrame:
    """Build every required single and interpretable cross-stratum summary."""

    dimensions: tuple[tuple[str, ...], ...] = (
        ("ast_size_bucket",),
        ("ast_depth",),
        ("operator_family",),
        ("operator_signature",),
        ("variable_count",),
        ("canonical_ast_constant_category",),
        ("canonical_ast_constant_count",),
        ("domain_mode",),
        ("split",),
        ("iid_ood",),
        ("count_status",),
        ("semantic_status",),
        ("count_semantic_status",),
        ("operator_family", "ast_size_bucket"),
        ("operator_family", "ast_depth"),
        ("operator_family", "domain_mode"),
        ("split", "operator_family"),
    )
    rows: list[dict[str, object]] = []
    for columns in dimensions:
        grouping: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
        for key, group in frame.groupby(grouping, dropna=False, sort=True):
            keys = key if isinstance(key, tuple) else (key,)
            valid = group.loc[group["tree_alpha_value"].notna()]
            alpha = _numeric_summary(valid["tree_alpha_value"], quantile_method=quantile_method)
            rows.append(
                {
                    "stratum": " x ".join(columns),
                    "key_json": _json_text(
                        {
                            column: "<null>" if pd.isna(value) else value
                            for column, value in zip(columns, keys, strict=True)
                        }
                    ),
                    "all_processed_count": len(group),
                    "count_success_count": int(group["count_status"].eq("success").sum()),
                    "count_failure_count": int(group["count_status"].ne("success").sum()),
                    "valid_alpha_count": len(valid),
                    "excluded_alpha_count": len(group) - len(valid),
                    "semantic_status_counts_json": _json_text(
                        dict(sorted(Counter(group["semantic_status"]).items()))
                    ),
                    "underpowered": len(group) < minimum_group_count,
                    **{f"alpha_{name}": value for name, value in alpha.items()},
                    "ast_nodes_mean": float(valid["ast_node_count"].mean()) if len(valid) else None,
                    "ast_nodes_median": float(valid["ast_node_count"].median())
                    if len(valid)
                    else None,
                    "eml_nodes_mean": float(valid["eml_node_number"].mean())
                    if len(valid)
                    else None,
                    "eml_nodes_median": float(valid["eml_node_number"].median())
                    if len(valid)
                    else None,
                    "eml_depth_mean": float(valid["eml_depth"].mean()) if len(valid) else None,
                    "eml_depth_median": float(valid["eml_depth"].median()) if len(valid) else None,
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(["stratum", "key_json"], kind="mergesort")


def build_threshold_table(
    frame: pd.DataFrame, scenarios: Sequence[Mapping[str, object]]
) -> pd.DataFrame:
    """Validate and reconcile every row/scenario outcome under both denominators."""

    scenario_by_name = {str(scenario["name"]): scenario for scenario in scenarios}
    if len(scenario_by_name) != len(scenarios):
        raise Goal2ArtifactError("threshold scenario names are not unique")
    expected_names = set(scenario_by_name)
    parsed: list[dict[str, Mapping[str, object]]] = []
    for expression_id, raw_outcomes in zip(
        frame["expression_id"], frame["threshold_outcomes_json"], strict=True
    ):
        try:
            outcomes = json.loads(str(raw_outcomes))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise Goal2ArtifactError(
                f"invalid threshold outcome JSON for expression {expression_id}"
            ) from error
        if not isinstance(outcomes, list) or not all(
            isinstance(outcome, dict) for outcome in outcomes
        ):
            raise Goal2ArtifactError(
                f"threshold outcomes must be a list of objects for expression {expression_id}"
            )
        names = [outcome.get("scenario_name") for outcome in outcomes]
        if any(not isinstance(name, str) or not name for name in names):
            raise Goal2ArtifactError(
                f"threshold outcome has an invalid scenario name for expression {expression_id}"
            )
        if len(names) != len(set(names)):
            raise Goal2ArtifactError(f"duplicate threshold outcome for expression {expression_id}")
        if set(names) != expected_names:
            raise Goal2ArtifactError(
                f"threshold outcome set is incomplete or unexpected for expression {expression_id}"
            )
        parsed.append({str(outcome["scenario_name"]): outcome for outcome in outcomes})

    reset = frame.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for name, scenario in scenario_by_name.items():
        scope = set(scenario["scope_families"])  # type: ignore[arg-type]
        definition_status = str(scenario["definition_status"])
        statuses: Counter[str] = Counter()
        for index, source_row in reset.iterrows():
            outcome = parsed[index][name]
            status = str(outcome.get("status"))
            applicable = source_row["operator_family"] in scope
            alpha = source_row["tree_alpha_value"]
            expected_status = "not_applicable"
            if applicable:
                if definition_status == "not_defined":
                    expected_status = "not_defined"
                elif pd.isna(alpha):
                    expected_status = "invalid_alpha"
                else:
                    expected_status = (
                        "passed"
                        if strict_threshold_pass(float(alpha), float(scenario["threshold_value"]))
                        else "failed"
                    )
            if status != expected_status:
                raise Goal2ArtifactError(
                    f"threshold status for {name!r} is inconsistent with scope and alpha"
                )
            expected_passed = status == "passed" if status in {"passed", "failed"} else None
            expected_threshold = (
                scenario.get("threshold_value") if definition_status == "defined" else None
            )
            expected_k = scenario.get("K") if definition_status == "defined" else None
            expected_l = scenario.get("L") if definition_status == "defined" else None
            if (
                outcome.get("passed") != expected_passed
                or outcome.get("threshold_value") != expected_threshold
                or outcome.get("K") != expected_k
                or outcome.get("L") != expected_l
                or outcome.get("formula") != scenario["formula"]
            ):
                raise Goal2ArtifactError(f"threshold metadata is corrupt for {name!r}")
            statuses[status] += 1
        valid_count = statuses["passed"] + statuses["failed"]
        processed_count = int(reset["operator_family"].isin(scope).sum())
        if sum(statuses.values()) != len(reset):
            raise Goal2ArtifactError(f"threshold outcomes do not reconcile for {name!r}")
        inapplicable_count = statuses["not_applicable"]
        if inapplicable_count != len(reset) - processed_count:
            raise Goal2ArtifactError(f"threshold scope does not reconcile for {name!r}")
        rows.append(
            {
                "scenario_name": name,
                "scope_families_json": _json_text(sorted(scope)),
                "definition_status": scenario["definition_status"],
                "K": scenario.get("K"),
                "L": scenario.get("L"),
                "formula": scenario["formula"],
                "threshold_value": scenario.get("threshold_value"),
                "derivation": scenario["derivation"],
                "references_json": _json_text(scenario["references"]),
                "all_processed_count": processed_count,
                "valid_applicable_count": valid_count,
                "pass_count": statuses["passed"],
                "fail_count": statuses["failed"],
                "invalid_alpha_count": statuses["invalid_alpha"],
                "not_defined_count": statuses["not_defined"],
                "missing_outcome_count": 0,
                "valid_only_pass_rate": statuses["passed"] / valid_count if valid_count else None,
                "all_processed_pass_rate": statuses["passed"] / processed_count
                if processed_count
                else None,
                "inapplicable_count": inapplicable_count,
            }
        )
    return pd.DataFrame.from_records(rows)


def _category_text(value: object) -> str:
    return "<null>" if value is None or pd.isna(value) else str(value)


def _natural_category_key(value: object) -> tuple[int, float, float, str]:
    """Sort numeric values and frozen range/log buckets by numeric meaning."""

    text = _category_text(value)
    if text in {"<null>", "missing", "out_of_policy"}:
        return (3, 0.0, 0.0, text)
    if text.startswith("10^"):
        try:
            exponent = float(text[3:])
        except ValueError:
            pass
        else:
            return (1, exponent, exponent, text)
    pieces = text.split("-", maxsplit=1)
    if len(pieces) == 2:
        try:
            minimum, maximum = (float(piece) for piece in pieces)
        except ValueError:
            pass
        else:
            return (1, minimum, maximum, text)
    try:
        number = float(text)
    except ValueError:
        return (2, 0.0, 0.0, text)
    return (0, number, number, text)


def _scaling_table(frame: pd.DataFrame, *, quantile_method: str) -> pd.DataFrame:
    specs = (
        ("eml_nodes_vs_ast_nodes", "ast_node_count", "eml_node_number"),
        ("alpha_vs_ast_nodes", "ast_node_count", "tree_alpha_value"),
        ("alpha_vs_ast_depth", "ast_depth", "tree_alpha_value"),
        ("eml_depth_vs_ast_depth", "ast_depth", "eml_depth"),
        ("compiler_operations_vs_ast_size", "ast_size_bucket", "compiler_operation_number"),
        ("runtime_vs_ast_size", "ast_size_bucket", "processing_elapsed_seconds"),
        ("runtime_vs_eml_size", "eml_log10_bucket", "processing_elapsed_seconds"),
    )
    rows: list[dict[str, object]] = []
    for relationship, x_name, y_name in specs:
        groups = list(frame.groupby(x_name, dropna=False, sort=False))
        groups.sort(key=lambda item: _natural_category_key(item[0]))
        for x_order, (x_value, group) in enumerate(groups):
            summary = _numeric_summary(group[y_name], quantile_method=quantile_method)
            rows.append(
                {
                    "relationship": relationship,
                    "x_name": x_name,
                    "x_value": _category_text(x_value),
                    "x_order": x_order,
                    "y_name": y_name,
                    "all_processed_count": len(group),
                    "valid_y_count": int(group[y_name].notna().sum()),
                    **summary,
                }
            )
    return pd.DataFrame.from_records(rows)


def _histogram(values: pd.Series, *, logarithmic: bool, bins: int = 60) -> pd.DataFrame:
    array = values.dropna().to_numpy(dtype=float)
    if logarithmic:
        array = np.log10(array[array > 0])
    if array.size == 0:
        return pd.DataFrame(columns=("bin_left", "bin_right", "count", "transformation"))
    minimum, maximum = float(np.min(array)), float(np.max(array))
    if minimum == maximum:
        edges = np.array([minimum - 0.5, maximum + 0.5])
    else:
        edges = np.linspace(minimum, maximum, bins + 1)
    counts, edges = np.histogram(array, bins=edges)
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts,
            "transformation": "log10" if logarithmic else "none",
        }
    )


def _scatter_sample(frame: pd.DataFrame, *, seed: int, sample_size: int) -> pd.DataFrame:
    valid = frame.loc[frame["tree_alpha_value"].notna()].copy()
    valid["sample_key"] = valid["expression_id"].map(
        lambda value: hashlib.sha256(f"geml-goal2-plot-v1\0{seed}\0{value}".encode()).hexdigest()
    )
    selected = valid.sort_values("sample_key", kind="mergesort").head(min(sample_size, len(valid)))
    selected["log10_eml_node_count"] = selected["eml_node_number"].map(
        lambda value: math.log10(float(value)) if pd.notna(value) and value > 0 else None
    )
    selected = selected.sort_values("sample_key", kind="mergesort")
    return selected[
        [
            "expression_id",
            "operator_family",
            "split",
            "ast_node_count",
            "ast_depth",
            "eml_node_number",
            "log10_eml_node_count",
            "eml_depth",
            "tree_alpha_value",
            "processing_elapsed_seconds",
        ]
    ].reset_index(drop=True)


def build_stability_tables(
    pilot: pd.DataFrame,
    final: pd.DataFrame,
    *,
    quantile_method: str,
    pilot_label: str,
    pilot_thresholds: pd.DataFrame,
    final_thresholds: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute exact pilot/final deltas, family ranks, and structure shifts."""

    pilot_ids = set(pilot["expression_id"])
    final_ids = set(final["expression_id"])
    shared_ids = pilot_ids & final_ids
    overlap_metadata = {
        "pilot_expression_count": len(pilot),
        "final_expression_count": len(final),
        "shared_expression_id_count": len(shared_ids),
        "pilot_expression_id_overlap_rate": len(shared_ids) / len(pilot) if len(pilot) else None,
        "final_expression_id_overlap_rate": len(shared_ids) / len(final) if len(final) else None,
        "overlap_caveat": (
            "descriptive comparison with shared expression IDs; not an independent-sample estimate"
        ),
    }
    shared_identity_columns = (
        "operator_family",
        "split",
        "sympy_srepr",
        "domain_mode",
        "ast_node_count",
        "ast_depth",
        "eml_node_count",
        "eml_depth",
        "tree_alpha_exact_ratio",
    )
    shared = pilot.loc[
        pilot["expression_id"].isin(shared_ids),
        ["expression_id", *shared_identity_columns],
    ].merge(
        final.loc[
            final["expression_id"].isin(shared_ids),
            ["expression_id", *shared_identity_columns],
        ],
        on="expression_id",
        suffixes=("_pilot", "_final"),
        validate="one_to_one",
    )
    for column in shared_identity_columns:
        pilot_values = shared[f"{column}_pilot"].fillna("<null>")
        final_values = shared[f"{column}_final"].fillna("<null>")
        if pilot_values.ne(final_values).any():
            raise Goal2ArtifactError(
                f"shared pilot/final expressions changed identity field {column!r}"
            )

    overall_rows: list[dict[str, object]] = []
    pilot_alpha = _numeric_summary(pilot["tree_alpha_value"], quantile_method=quantile_method)
    final_alpha = _numeric_summary(final["tree_alpha_value"], quantile_method=quantile_method)
    for metric in ("mean", "median", "p90", "p95"):
        overall_rows.append(
            {
                "pilot_label": pilot_label,
                "metric": f"alpha_{metric}",
                "denominator": "valid_alpha_rows",
                "interpretation": "structural raw pure-EML tree-alpha statistic",
                "pilot_value": pilot_alpha[metric],
                "final_value": final_alpha[metric],
                "delta_final_minus_pilot": final_alpha[metric] - pilot_alpha[metric],
            }
        )
    rate_specs = (
        (
            "selected_terminal_issue_rate",
            pilot["semantic_selected"].eq(True),
            final["semantic_selected"].eq(True),
            lambda group: group["failure"],
            "deterministically selected semantic rows",
            "comparable within the selected semantic-audit populations",
        ),
        (
            "materialized_semantic_nonpass_rate",
            pilot["materialization_status"].eq("materialized"),
            final["materialization_status"].eq("materialized"),
            lambda group: group["semantic_status"].ne("passed"),
            "materialized semantic-audit rows",
            "comparable within the materialized semantic-audit populations",
        ),
        (
            "all_processed_terminal_issue_incidence",
            pd.Series(True, index=pilot.index),
            pd.Series(True, index=final.index),
            lambda group: group["failure"],
            "all processed rows",
            (
                "selection-diluted descriptive incidence only; stage sampling moduli differ "
                "and this is not a semantic failure-rate comparison"
            ),
        ),
    )
    for metric, pilot_mask, final_mask, numerator, denominator, interpretation in rate_specs:
        pilot_group = pilot.loc[pilot_mask]
        final_group = final.loc[final_mask]
        pilot_rate = float(numerator(pilot_group).mean()) if len(pilot_group) else None
        final_rate = float(numerator(final_group).mean()) if len(final_group) else None
        overall_rows.append(
            {
                "pilot_label": pilot_label,
                "metric": metric,
                "denominator": denominator,
                "interpretation": interpretation,
                "pilot_value": pilot_rate,
                "final_value": final_rate,
                "delta_final_minus_pilot": (
                    final_rate - pilot_rate
                    if pilot_rate is not None and final_rate is not None
                    else None
                ),
            }
        )
    merged_thresholds = pilot_thresholds.merge(
        final_thresholds,
        on="scenario_name",
        suffixes=("_pilot", "_final"),
        validate="one_to_one",
    )
    for row in merged_thresholds.itertuples():
        for denominator in ("valid_only_pass_rate", "all_processed_pass_rate"):
            pilot_value = getattr(row, f"{denominator}_pilot")
            final_value = getattr(row, f"{denominator}_final")
            overall_rows.append(
                {
                    "pilot_label": pilot_label,
                    "metric": f"threshold:{row.scenario_name}:{denominator}",
                    "denominator": denominator,
                    "interpretation": "strict named-threshold pass rate",
                    "pilot_value": pilot_value,
                    "final_value": final_value,
                    "delta_final_minus_pilot": final_value - pilot_value
                    if pd.notna(pilot_value) and pd.notna(final_value)
                    else None,
                }
            )

    family_rows: list[dict[str, object]] = []
    for family in sorted(set(pilot["operator_family"]) | set(final["operator_family"])):
        pilot_group = pilot.loc[pilot["operator_family"].eq(family)]
        final_group = final.loc[final["operator_family"].eq(family)]
        pilot_summary = _numeric_summary(
            pilot_group["tree_alpha_value"], quantile_method=quantile_method
        )
        final_summary = _numeric_summary(
            final_group["tree_alpha_value"], quantile_method=quantile_method
        )
        shared_count = int(shared["operator_family_pilot"].eq(family).sum())
        family_rows.append(
            {
                "pilot_label": pilot_label,
                "operator_family": family,
                "pilot_count": len(pilot_group),
                "final_count": len(final_group),
                "shared_expression_id_count": shared_count,
                "pilot_expression_id_overlap_rate": (
                    shared_count / len(pilot_group) if len(pilot_group) else None
                ),
                "final_expression_id_overlap_rate": (
                    shared_count / len(final_group) if len(final_group) else None
                ),
                "overlap_caveat": overlap_metadata["overlap_caveat"],
                "pilot_alpha_median": pilot_summary["median"],
                "final_alpha_median": final_summary["median"],
                "median_delta": (
                    final_summary["median"] - pilot_summary["median"]
                    if final_summary["median"] is not None and pilot_summary["median"] is not None
                    else None
                ),
                "pilot_alpha_p90": pilot_summary["p90"],
                "final_alpha_p90": final_summary["p90"],
                "p90_delta": (
                    final_summary["p90"] - pilot_summary["p90"]
                    if final_summary["p90"] is not None and pilot_summary["p90"] is not None
                    else None
                ),
            }
        )
    family = pd.DataFrame.from_records(family_rows)
    family["pilot_median_rank"] = family["pilot_alpha_median"].rank(method="min", ascending=False)
    family["final_median_rank"] = family["final_alpha_median"].rank(method="min", ascending=False)
    family["rank_delta"] = family["final_median_rank"] - family["pilot_median_rank"]
    rank_correlation = family[["pilot_median_rank", "final_median_rank"]].corr().iloc[0, 1]
    family["median_rank_correlation"] = float(rank_correlation)

    distribution_rows: list[dict[str, object]] = []
    for dimension in ("ast_size_bucket", "ast_depth"):
        pilot_categories = pilot[dimension].map(_category_text)
        final_categories = final[dimension].map(_category_text)
        values = sorted(set(pilot_categories) | set(final_categories), key=_natural_category_key)
        for value in values:
            pilot_rate = float(pilot_categories.eq(value).mean())
            final_rate = float(final_categories.eq(value).mean())
            distribution_rows.append(
                {
                    "pilot_label": pilot_label,
                    "dimension": dimension,
                    "value": value,
                    "pilot_rate": pilot_rate,
                    "final_rate": final_rate,
                    "delta_final_minus_pilot": final_rate - pilot_rate,
                }
            )
    stability_overall = pd.DataFrame.from_records(overall_rows).assign(**overlap_metadata)
    return {
        "stability_overall": stability_overall,
        "stability_family": family,
        "stability_distribution": pd.DataFrame.from_records(distribution_rows),
    }


def _table_metadata(root: Path, value: dict[str, object]) -> dict[str, object]:
    path = Path(str(value["path"]))
    return {**value, "path": path.resolve().relative_to(root.resolve()).as_posix()}


def run_analysis(
    *,
    metrics_manifest: str | Path,
    pilot_manifest: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Validate production inputs and atomically publish all Goal 2 analysis tables."""

    final_path = Path(metrics_manifest).resolve()
    pilot_path = Path(pilot_manifest).resolve()
    output = Path(output_dir).resolve()
    loaded = load_goal2_config(config_path)
    analysis_source_fingerprint = _analysis_source_fingerprint()
    final_hash = sha256_file(final_path)
    pilot_hash = sha256_file(pilot_path)
    completion = output / "manifest.json"
    if completion.exists():
        manifest = validate_analysis_manifest(completion)
        expected = {
            "final_metrics_manifest_sha256": final_hash,
            "pilot_metrics_manifest_sha256": pilot_hash,
            "config_hash": loaded.config_hash,
        }
        if any(manifest.get(name) != value for name, value in expected.items()):
            raise Goal2ArtifactError("completed analysis inputs differ from the request")
        return completion

    final, final_manifest = _load_metrics(final_path, label="final")
    pilot, pilot_manifest_value = _load_metrics(pilot_path, label="pilot")
    _validate_metrics_role(final_manifest, loaded, expected_stage="final")
    _validate_metrics_role(pilot_manifest_value, loaded, expected_stage="pilot")
    if final_manifest.get("config_hash") != pilot_manifest_value.get("config_hash"):
        raise Goal2ArtifactError("pilot and final metric config hashes are incompatible")
    if final_manifest.get("runner_fingerprint") != pilot_manifest_value.get("runner_fingerprint"):
        raise Goal2ArtifactError("pilot and final metric runner fingerprints are incompatible")
    if final_manifest.get("config_hash") != loaded.config_hash:
        raise Goal2ArtifactError("analysis config differs from the metric production config")
    final = _prepare(final, loaded)
    pilot = _prepare(pilot, loaded)
    policy = loaded.config.analysis
    scenarios = final_manifest["run_metadata"]["threshold_scenarios"]
    if pilot_manifest_value["run_metadata"]["threshold_scenarios"] != scenarios:
        raise Goal2ArtifactError("pilot and final threshold scenarios differ")
    configured_scenarios = [scenario.as_dict() for scenario in loaded.thresholds]
    if scenarios != configured_scenarios:
        raise Goal2ArtifactError("metric threshold scenarios differ from the analysis config")

    tables: dict[str, pd.DataFrame] = {}
    overall_rows = []
    stratified_rows = []
    threshold_frames: dict[str, pd.DataFrame] = {}
    for label, frame in (("pilot", pilot), ("final", final)):
        for population in ("all_processed", "valid_alpha"):
            overall_rows.append(
                {
                    "dataset": label,
                    "quantile_method": policy.quantile_method,
                    "standard_deviation_ddof": 0,
                    **summarize_population(
                        frame,
                        quantile_method=policy.quantile_method,
                        population=population,
                    ),
                }
            )
        strata = build_stratified_table(
            frame,
            quantile_method=policy.quantile_method,
            minimum_group_count=policy.minimum_group_count,
        )
        strata.insert(0, "dataset", label)
        stratified_rows.append(strata)
        thresholds = build_threshold_table(frame, scenarios)
        thresholds.insert(0, "dataset", label)
        threshold_frames[label] = thresholds
    tables["overall"] = pd.DataFrame.from_records(overall_rows)
    tables["stratified"] = pd.concat(stratified_rows, ignore_index=True)
    tables["thresholds"] = pd.concat(threshold_frames.values(), ignore_index=True)
    tables["scaling"] = _scaling_table(final, quantile_method=policy.quantile_method)
    tables["alpha_histogram"] = _histogram(final["tree_alpha_value"], logarithmic=False)
    tables["alpha_log_histogram"] = _histogram(final["tree_alpha_value"], logarithmic=True)
    tables["scatter_sample"] = _scatter_sample(
        final,
        seed=policy.sample_seed,
        sample_size=policy.scatter_sample_size,
    )
    tables["failure_status_counts"] = failure_status_counts(final)
    tables["semantic_backend_status_counts"] = semantic_backend_status_counts(
        final,
        expected_backends=loaded.config.semantic_audit.backends,
        expected_compiler_mode=CompilerMode.OFFICIAL_V4.value,
    )
    failure_tables = build_failure_tables(final, top_count=policy.top_case_count)
    expected_failures = final.loc[final["failure"], "expression_id"]
    assert_failure_coverage(failure_tables["failure_details"]["expression_id"], expected_failures)
    tables.update(failure_tables)
    tables.update(
        build_stability_tables(
            pilot,
            final,
            quantile_method=policy.quantile_method,
            pilot_label=str(pilot_manifest_value["run_metadata"]["source_label"]),
            pilot_thresholds=threshold_frames["pilot"],
            final_thresholds=threshold_frames["final"],
        )
    )

    table_metadata: dict[str, dict[str, object]] = {}
    if _analysis_source_fingerprint() != analysis_source_fingerprint:
        raise Goal2ArtifactError("analysis source changed before table publication")
    for name, frame in sorted(tables.items()):
        metadata = _atomic_table(output / "tables" / f"{name}.parquet", frame)
        table_metadata[name] = _table_metadata(output, metadata)
    if _analysis_source_fingerprint() != analysis_source_fingerprint:
        raise Goal2ArtifactError("analysis source changed during table publication")
    if (
        sha256_file(final_path) != final_hash
        or sha256_file(pilot_path) != pilot_hash
        or sha256_file(loaded.path) != loaded.config_hash
    ):
        raise Goal2ArtifactError("analysis inputs changed during table publication")
    manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_source_fingerprint": analysis_source_fingerprint,
        "compiler_mode": CompilerMode.OFFICIAL_V4.value,
        "config_hash": loaded.config_hash,
        "metrics_runner_fingerprint": final_manifest["runner_fingerprint"],
        "final_metrics_manifest": final_path.as_posix(),
        "final_metrics_manifest_sha256": final_hash,
        "pilot_metrics_manifest": pilot_path.as_posix(),
        "pilot_metrics_manifest_sha256": pilot_hash,
        "pilot_label": pilot_manifest_value["run_metadata"]["source_label"],
        "quantile_method": policy.quantile_method,
        "standard_deviation_ddof": 0,
        "minimum_group_count": policy.minimum_group_count,
        "deterministic_plot_sample": {
            "algorithm": "lowest sha256(geml-goal2-plot-v1\\0seed\\0expression_id)",
            "seed": policy.sample_seed,
            "maximum_count": policy.scatter_sample_size,
        },
        "tables": table_metadata,
    }
    _atomic_json(completion, manifest)
    validate_analysis_manifest(completion)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-manifest", required=True, type=Path)
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        path = run_analysis(
            metrics_manifest=arguments.metrics_manifest,
            pilot_manifest=arguments.pilot_manifest,
            config_path=arguments.config,
            output_dir=arguments.output_dir,
        )
    except (Goal2ArtifactError, OSError, ValueError) as error:
        print(_json_text({"status": "failed", "message": str(error)}))
        return 1
    print(_json_text({"status": "complete", "manifest": path.as_posix()}))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
