"""Validate frozen inputs and build the no-retraining Goal 11 synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from geml.analysis.goal11.final_eval import (
    ExternalReference,
    GateG11Criteria,
    Goal11Synthesis,
    Goal11SynthesisError,
    TrackEvidence,
    build_goal11_synthesis,
    render_gate_g11_markdown,
    render_goal11_summary_markdown,
)
from geml.analysis.goal11.scaling import FixedScaleResult
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
from geml.plots.goal11_final import render_plots


class Goal11FinalRunError(ValueError):
    """Goal 11 synthesis inputs are missing, corrupt, or unauthenticated."""


def _load_model(path: str | Path, model_type: type[BaseModel]) -> BaseModel:
    try:
        return model_type.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise Goal11FinalRunError(f"could not load {model_type.__name__}: {path}") from error


def _load_jsonl(path: str | Path, model_type: type[BaseModel]) -> tuple[BaseModel, ...]:
    rows = []
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(model_type.model_validate_json(line))
                except ValidationError as error:
                    raise Goal11FinalRunError(
                        f"invalid {model_type.__name__} at line {line_number}"
                    ) from error
    except OSError as error:
        raise Goal11FinalRunError(f"could not read {path}") from error
    return tuple(rows)


def _normalize_criteria(payload: dict[str, object]) -> GateG11Criteria:
    normalized = dict(payload)
    for key in ("expected_seeds", "required_tracks"):
        if isinstance(normalized.get(key), list):
            normalized[key] = tuple(normalized[key])
    try:
        return GateG11Criteria.model_validate(normalized, strict=False)
    except ValidationError as error:
        raise Goal11FinalRunError("invalid Gate G11 criteria") from error


def load_final_config(path: str | Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise Goal11FinalRunError(f"could not read Goal 11 final config: {path}") from error
    if not isinstance(payload, dict):
        raise Goal11FinalRunError("Goal 11 final config must be a YAML mapping")
    if payload.get("schema_version") != "geml-goal11-final-run-config-v1":
        raise Goal11FinalRunError("Goal 11 final config has the wrong schema version")
    return payload


def validate_evidence_sources(
    manifest: WorkshopManifest,
    fixed_scale: FixedScaleResult,
    track_evidence: tuple[TrackEvidence, ...],
    external_references: tuple[ExternalReference, ...],
    criteria: GateG11Criteria,
    artifact_root: str | Path,
) -> None:
    evidence_cache = EvidenceAuthenticationCache()
    try:
        for track in track_evidence:
            if track.outcome.value != "insufficient":
                if (
                    track.source_artifact_id is None
                    or track.source_sha256 is None
                    or track.source_locator is None
                ):
                    raise EvidenceAuthenticationError(
                        f"controlled track lacks outcome provenance: {track.track.value}"
                    )
                authenticate_source_record(
                    manifest,
                    artifact_root,
                    artifact_id=track.source_artifact_id,
                    source_sha256=track.source_sha256,
                    source_locator=track.source_locator,
                    expected_fields=track.evidence_projection(),
                    required_roles=("gate_g11",),
                    allowed_categories=("result_table",),
                    cache=evidence_cache,
                )
            for metric in track.metrics:
                authenticate_source_record(
                    manifest,
                    artifact_root,
                    artifact_id=metric.source_artifact_id,
                    source_sha256=metric.source_sha256,
                    source_locator=metric.source_locator,
                    expected_fields=metric.evidence_projection(),
                    required_roles=("gate_g11", "final_report"),
                    allowed_categories=("result_table",),
                    cache=evidence_cache,
                )
        for binding in (
            binding for panel in fixed_scale.panels for binding in panel.source_bindings
        ):
            reference = next(
                (
                    item
                    for item in manifest.artifacts
                    if item.artifact_id == binding.source_artifact_id
                ),
                None,
            )
            if (
                reference is None
                or reference.observed_sha256 != binding.source_sha256
                or reference.state.value != "complete"
                or "fixed_scale" not in reference.roles
                or reference.category != "result_table"
            ):
                raise EvidenceAuthenticationError(
                    f"fixed-scale source is not approved: {binding.source_artifact_id}"
                )
        for item in external_references:
            authenticate_source_record(
                manifest,
                artifact_root,
                artifact_id=item.source_artifact_id,
                source_sha256=item.source_sha256,
                source_locator=item.source_locator,
                expected_fields=item.evidence_projection(),
                required_roles=("external_noncontrolled",),
                allowed_categories=("external_reference",),
                cache=evidence_cache,
            )
        if criteria.decision_rule_digest is not None:
            if (
                criteria.decision_rule_artifact_id is None
                or criteria.decision_rule_source_sha256 is None
                or criteria.decision_rule_source_locator is None
            ):
                raise EvidenceAuthenticationError("frozen decision rules lack complete provenance")
            authenticate_source_record(
                manifest,
                artifact_root,
                artifact_id=criteria.decision_rule_artifact_id,
                source_sha256=criteria.decision_rule_source_sha256,
                source_locator=criteria.decision_rule_source_locator,
                expected_fields={
                    "decision_rule_digest": criteria.decision_rule_digest,
                    "minimum_supporting_tracks": criteria.minimum_supporting_tracks,
                    "minimum_fixed_scale_panels": criteria.minimum_fixed_scale_panels,
                    "allow_material_contradiction": criteria.allow_material_contradiction,
                },
                required_roles=("gate_g11",),
                allowed_categories=("configuration",),
                cache=evidence_cache,
            )
    except EvidenceAuthenticationError as error:
        raise Goal11FinalRunError(
            f"evidence source/row is not complete and authenticated: {error}"
        ) from error


def build_authenticated_synthesis(
    manifest: WorkshopManifest,
    audit: WorkshopManifestAudit,
    fixed_scale: FixedScaleResult,
    tracks: tuple[TrackEvidence, ...],
    criteria: GateG11Criteria,
    artifact_root: str | Path,
    *,
    external_references: tuple[ExternalReference, ...] = (),
) -> Goal11Synthesis:
    if audit.manifest_sha256 != manifest_sha256(manifest):
        raise Goal11FinalRunError("manifest audit does not authenticate the supplied manifest")
    require_valid_workshop_manifest(audit)
    if fixed_scale.manifest_sha256 != manifest_sha256(manifest):
        raise Goal11FinalRunError("fixed-scale result names a different workshop manifest")
    audit_sha256 = hashlib.sha256(canonical_json_bytes(audit)).hexdigest()
    if fixed_scale.manifest_audit_sha256 != audit_sha256:
        raise Goal11FinalRunError("fixed-scale result names a different manifest audit")
    validate_evidence_sources(
        manifest,
        fixed_scale,
        tracks,
        external_references,
        criteria,
        artifact_root,
    )
    synthesis = build_goal11_synthesis(
        tracks,
        fixed_scale,
        criteria,
        external_references=external_references,
        decision_rules_authenticated=criteria.decision_rule_digest is not None,
        producer_gates_authenticated=True,
    )
    return synthesis.model_copy(
        update={
            "manifest_sha256": manifest_sha256(manifest),
            "manifest_audit_sha256": audit_sha256,
            "fixed_scale_sha256": _json_sha256(fixed_scale.model_dump(mode="json")),
            "track_evidence_sha256": _json_sha256(
                [item.model_dump(mode="json") for item in tracks]
            ),
            "external_evidence_sha256": _json_sha256(
                [item.model_dump(mode="json") for item in external_references]
            ),
            "criteria_sha256": _json_sha256(criteria.model_dump(mode="json")),
        }
    )


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_final_eval(config_path: str | Path) -> Goal11Synthesis:
    payload = load_final_config(config_path)
    required_names = (
        "manifest_path",
        "manifest_audit_path",
        "fixed_scale_result_path",
        "fixed_scale_run_config_path",
        "fixed_scale_observations_path",
        "track_evidence_path",
        "result_path",
        "summary_path",
        "gate_path",
        "plot_dir",
        "artifact_root",
    )
    missing = [name for name in required_names if not isinstance(payload.get(name), str)]
    if missing:
        raise Goal11FinalRunError("production inputs are not frozen: " + ", ".join(sorted(missing)))
    criteria_payload = payload.get("gate_criteria")
    if not isinstance(criteria_payload, dict):
        raise Goal11FinalRunError("gate_criteria must be a mapping")
    criteria = _normalize_criteria(criteria_payload)
    manifest = _load_model(payload["manifest_path"], WorkshopManifest)
    audit = _load_model(payload["manifest_audit_path"], WorkshopManifestAudit)
    fixed_scale = _load_model(payload["fixed_scale_result_path"], FixedScaleResult)
    tracks = _load_jsonl(payload["track_evidence_path"], TrackEvidence)
    external_path = payload.get("external_llm_path")
    external = () if external_path is None else _load_jsonl(external_path, ExternalReference)
    if (
        not isinstance(manifest, WorkshopManifest)
        or not isinstance(audit, WorkshopManifestAudit)
        or not isinstance(fixed_scale, FixedScaleResult)
    ):
        raise Goal11FinalRunError("unexpected input model")
    if fixed_scale.run_config_sha256 != sha256_file(payload["fixed_scale_run_config_path"])[0]:
        raise Goal11FinalRunError("fixed-scale run-config digest is not authenticated")
    if (
        fixed_scale.observations_file_sha256
        != sha256_file(payload["fixed_scale_observations_path"])[0]
    ):
        raise Goal11FinalRunError("fixed-scale observation-file digest is not authenticated")
    synthesis = build_authenticated_synthesis(
        manifest,
        audit,
        fixed_scale,
        tuple(item for item in tracks if isinstance(item, TrackEvidence)),
        criteria,
        payload["artifact_root"],
        external_references=tuple(item for item in external if isinstance(item, ExternalReference)),
    )
    external_file_sha256 = None if external_path is None else sha256_file(external_path)[0]
    synthesis = synthesis.model_copy(
        update={
            "run_config_sha256": sha256_file(config_path)[0],
            "fixed_scale_file_sha256": sha256_file(payload["fixed_scale_result_path"])[0],
            "track_evidence_file_sha256": sha256_file(payload["track_evidence_path"])[0],
            "external_evidence_file_sha256": external_file_sha256,
        }
    )
    destination = Path(payload["result_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(synthesis)
    if destination.exists() and destination.read_bytes() != data:
        raise FileExistsError(f"Goal 11 result already exists with different bytes: {destination}")
    destination.write_bytes(data)
    for output_name, markdown in (
        ("summary_path", render_goal11_summary_markdown(synthesis)),
        ("gate_path", render_gate_g11_markdown(synthesis.gate)),
    ):
        output = Path(payload[output_name])
        text = markdown.rstrip().encode("utf-8") + b"\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.read_bytes() != text:
            raise FileExistsError(f"Goal 11 output already exists with different bytes: {output}")
        output.write_bytes(text)
    render_plots(synthesis, payload["plot_dir"])
    return synthesis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parser().parse_args(argv)
    try:
        run_final_eval(args.config)
    except Goal11SynthesisError as error:
        raise Goal11FinalRunError(str(error)) from error
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
