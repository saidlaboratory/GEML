"""Run fixed-scale Goal 11 analysis from authenticated normalized rows."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from geml.analysis.goal11.scaling import (
    EfficiencyObservation,
    FixedScaleAnalysisConfig,
    FixedScaleAnalysisError,
    FixedScaleResult,
    render_fixed_scale_markdown,
    summarize_fixed_scale,
)
from geml.experiments.goal11.corpus_v3 import (
    EvidenceAuthenticationCache,
    EvidenceAuthenticationError,
    WorkshopManifest,
    WorkshopManifestAudit,
    authenticate_source_record,
    canonical_json_bytes,
    manifest_sha256,
    require_valid_workshop_manifest,
    sha256_file,
)
from geml.plots.goal11_scaling import render_plots


class FixedScaleRunError(ValueError):
    """The fixed-scale run inputs are missing, unauthenticated, or invalid."""


def load_fixed_scale_config(path: str | Path) -> tuple[FixedScaleAnalysisConfig, dict[str, object]]:
    """Load analysis settings while retaining runner paths separately."""

    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise FixedScaleRunError(f"could not read fixed-scale config: {path}") from error
    if not isinstance(payload, dict):
        raise FixedScaleRunError("fixed-scale config must be a YAML mapping")
    if payload.get("schema_version") != "geml-goal11-fixed-scale-run-config-v1":
        raise FixedScaleRunError("fixed-scale run config has the wrong schema version")
    analysis_payload = payload.get("analysis")
    if not isinstance(analysis_payload, dict):
        raise FixedScaleRunError("fixed-scale config requires an analysis mapping")
    try:
        analysis_payload = dict(analysis_payload)
        for key in ("expected_seeds", "comparisons"):
            if isinstance(analysis_payload.get(key), list):
                analysis_payload[key] = tuple(analysis_payload[key])
        for item in analysis_payload.get("comparisons", ()):
            if isinstance(item, dict) and isinstance(item.get("resource_metric_ids"), list):
                item["resource_metric_ids"] = tuple(item["resource_metric_ids"])
        config = FixedScaleAnalysisConfig.model_validate(analysis_payload)
    except ValidationError as error:
        raise FixedScaleRunError("invalid fixed-scale analysis configuration") from error
    return config, payload


def load_observations(path: str | Path) -> tuple[EfficiencyObservation, ...]:
    rows = []
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(EfficiencyObservation.model_validate_json(line))
                except ValidationError as error:
                    raise FixedScaleRunError(
                        f"invalid fixed-scale row at line {line_number}"
                    ) from error
    except OSError as error:
        raise FixedScaleRunError(f"could not read observations: {path}") from error
    return tuple(rows)


def analyze_authenticated_rows(
    manifest: WorkshopManifest,
    audit: WorkshopManifestAudit,
    observations: tuple[EfficiencyObservation, ...],
    config: FixedScaleAnalysisConfig,
    artifact_root: str | Path,
) -> FixedScaleResult:
    """Require manifest integrity and bind every row to a complete source artifact."""

    if audit.manifest_sha256 != manifest_sha256(manifest):
        raise FixedScaleRunError("manifest audit does not authenticate the supplied manifest")
    require_valid_workshop_manifest(audit)
    evidence_cache = EvidenceAuthenticationCache()
    for row in observations:
        try:
            authenticate_source_record(
                manifest,
                artifact_root,
                artifact_id=row.source_artifact_id,
                source_sha256=row.source_sha256,
                source_locator=row.source_locator,
                expected_fields=row.evidence_projection(),
                required_roles=("fixed_scale",),
                allowed_categories=("result_table",),
                cache=evidence_cache,
            )
        except EvidenceAuthenticationError as error:
            raise FixedScaleRunError(
                f"row {row.row_id} is not authenticated against its exact source record"
            ) from error
    result = summarize_fixed_scale(observations, config)
    audit_sha256 = hashlib.sha256(canonical_json_bytes(audit)).hexdigest()
    return result.model_copy(
        update={
            "manifest_sha256": manifest_sha256(manifest),
            "manifest_audit_sha256": audit_sha256,
        }
    )


def _load_model(path: str | Path, model_type: type[BaseModel]) -> BaseModel:
    try:
        return model_type.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise FixedScaleRunError(f"could not load {model_type.__name__}: {path}") from error


def _write_result(result: FixedScaleResult, path: str | Path) -> Path:
    destination = Path(path)
    payload = canonical_json_bytes(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != payload:
        raise FileExistsError(f"fixed-scale output already exists with different bytes: {path}")
    destination.write_bytes(payload)
    return destination


def _write_text(value: str, path: str | Path) -> Path:
    destination = Path(path)
    payload = (value.rstrip() + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != payload:
        raise FileExistsError(f"fixed-scale output already exists with different bytes: {path}")
    destination.write_bytes(payload)
    return destination


def run_fixed_scale(config_path: str | Path) -> FixedScaleResult:
    config, payload = load_fixed_scale_config(config_path)
    required_paths = {
        name: payload.get(name)
        for name in (
            "manifest_path",
            "manifest_audit_path",
            "observations_path",
            "artifact_root",
            "result_path",
            "markdown_path",
            "plot_dir",
        )
    }
    missing = sorted(name for name, value in required_paths.items() if not isinstance(value, str))
    if missing:
        raise FixedScaleRunError("production inputs are not frozen: " + ", ".join(missing))
    manifest = _load_model(required_paths["manifest_path"], WorkshopManifest)
    audit = _load_model(required_paths["manifest_audit_path"], WorkshopManifestAudit)
    if not isinstance(manifest, WorkshopManifest) or not isinstance(audit, WorkshopManifestAudit):
        raise FixedScaleRunError("unexpected manifest model")
    observations = load_observations(required_paths["observations_path"])
    result = analyze_authenticated_rows(
        manifest,
        audit,
        observations,
        config,
        required_paths["artifact_root"],
    )
    observations_file_sha256, _ = sha256_file(required_paths["observations_path"])
    run_config_sha256, _ = sha256_file(config_path)
    result = result.model_copy(
        update={
            "observations_file_sha256": observations_file_sha256,
            "run_config_sha256": run_config_sha256,
        }
    )
    _write_result(result, required_paths["result_path"])
    _write_text(render_fixed_scale_markdown(result), required_paths["markdown_path"])
    render_plots(result, required_paths["plot_dir"])
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parser().parse_args(argv)
    try:
        run_fixed_scale(args.config)
    except FixedScaleAnalysisError as error:
        raise FixedScaleRunError(str(error)) from error
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
