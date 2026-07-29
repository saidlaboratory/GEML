"""Frozen Goal 8 release record: external aggregate/config trust anchors.

Gate G8 refuses any producer bundle whose expected aggregate and pre-run
config digests were not frozen outside the analyzed run directory, and the
#68/#69 run layouts publish shard completions only.  This module is the
orchestration-owned step (see ``docs/specs/GOAL8_RELEASE_RECORD.md``) that
revalidates both finalized producer runs through the analysis-side
authenticator and freezes their aggregate/config anchors in one immutable
record retained outside both run directories.  The #70 analyzer consumes the
record; it never issues one for itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from geml.analysis.goal8.summary import (
    AnalysisError,
    ManifestKind,
    authenticate_producer_run,
)
from geml.experiments.goal4.runtime import (
    Goal4RuntimeError,
    atomic_write_json,
    canonical_json,
    sha256_hex,
)

RELEASE_RECORD_SCHEMA = "geml-goal8-release-record-v1"

_RECORD_DOMAIN = b"geml-goal8-release-record-v1\0"
_HEX = frozenset("0123456789abcdef")
_BUNDLE_FIELDS = frozenset(
    {
        "kind",
        "run_id",
        "config_digest",
        "aggregate_sha256",
        "shard_count",
        "cell_count",
        "byte_count",
        "cell_schema_version",
        "shard_schema_version",
    }
)


class ReleaseRecordError(ValueError):
    """The release record or a producer run violates the frozen contract."""


def emit_release_record(
    record_path: str | Path,
    *,
    proof_run_directory: str | Path,
    proof_config_digest: str,
    simplification_run_directory: str | Path,
    simplification_config_digest: str,
    proof_cell_schema_version: str | None = None,
    proof_shard_schema_version: str | None = None,
    simplification_cell_schema_version: str | None = None,
    simplification_shard_schema_version: str | None = None,
) -> Path:
    """Freeze both runs' aggregate/config anchors into one immutable record.

    The config digests must come from each pre-run resolved configuration (the
    runner CLIs print them under ``--validate-only`` before execution); the
    emitter never derives them from the run outputs it is anchoring.  Schema
    overrides exist only for tiny fixture runs, exactly as in
    ``authenticate_producer_run``.
    """

    destination = Path(record_path)
    resolved_destination = destination.resolve()
    bundles: dict[str, dict[str, object]] = {}
    for kind, run_directory, config_digest, cell_schema, shard_schema in (
        (
            ManifestKind.PROOF,
            proof_run_directory,
            proof_config_digest,
            proof_cell_schema_version,
            proof_shard_schema_version,
        ),
        (
            ManifestKind.SIMPLIFICATION,
            simplification_run_directory,
            simplification_config_digest,
            simplification_cell_schema_version,
            simplification_shard_schema_version,
        ),
    ):
        resolved = _resolved_run_directory(run_directory)
        if resolved_destination.is_relative_to(resolved):
            raise ReleaseRecordError(
                f"release record must be retained outside the {kind.value} run directory"
            )
        aggregate = _run_directory_aggregate(resolved)
        try:
            bundle = authenticate_producer_run(
                resolved,
                kind=kind,
                cell_schema_version=cell_schema,
                shard_schema_version=shard_schema,
                expected_aggregate_sha256=aggregate,
                expected_config_digest=config_digest,
            )
        except AnalysisError as error:
            raise ReleaseRecordError(f"{kind.value} run cannot be anchored: {error}") from error
        bundles[kind.value] = {
            "kind": kind.value,
            "run_id": bundle.run_id,
            "config_digest": bundle.config_digest,
            "aggregate_sha256": bundle.aggregate_sha256,
            "shard_count": bundle.shard_count,
            "cell_count": len(bundle.rows),
            "byte_count": bundle.byte_count,
            "cell_schema_version": bundle.cell_schema_version,
            "shard_schema_version": bundle.shard_schema_version,
        }
    payload: dict[str, object] = {
        "schema_version": RELEASE_RECORD_SCHEMA,
        "bundles": bundles,
    }
    payload["content_digest"] = _content_digest(payload)
    try:
        return atomic_write_json(destination, payload)
    except Goal4RuntimeError as error:
        raise ReleaseRecordError(str(error)) from error


def load_release_record(path: str | Path) -> dict[str, dict[str, object]]:
    """Verify one frozen release record and return its per-kind trust anchors."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ReleaseRecordError(f"cannot read release record {source}: {error}") from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseRecordError(f"release record is not valid UTF-8 JSON: {source}") from error
    if not isinstance(payload, dict):
        raise ReleaseRecordError("release record root must be a JSON object")
    if payload.get("schema_version") != RELEASE_RECORD_SCHEMA:
        raise ReleaseRecordError("unknown release record schema")
    if set(payload) != {"schema_version", "bundles", "content_digest"}:
        raise ReleaseRecordError("release record has unexpected top-level fields")
    if payload["content_digest"] != _content_digest(payload):
        raise ReleaseRecordError("release record content_digest mismatch")
    bundles = payload["bundles"]
    if not isinstance(bundles, dict) or set(bundles) != {kind.value for kind in ManifestKind}:
        raise ReleaseRecordError("release record must anchor exactly the two producer bundles")
    for key, entry in bundles.items():
        if not isinstance(entry, dict) or set(entry) != _BUNDLE_FIELDS:
            raise ReleaseRecordError(f"release record bundle {key} has the wrong fields")
        if entry["kind"] != key:
            raise ReleaseRecordError(f"release record bundle {key} has a mismatched kind")
        for field in ("run_id", "config_digest", "aggregate_sha256"):
            if not _is_sha256(entry[field]):
                raise ReleaseRecordError(f"release record bundle {key} has a malformed {field}")
        for field in ("shard_count", "cell_count", "byte_count"):
            value = entry[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ReleaseRecordError(f"release record bundle {key} has an invalid {field}")
        for field in ("cell_schema_version", "shard_schema_version"):
            value = entry[field]
            if not isinstance(value, str) or not value.strip():
                raise ReleaseRecordError(f"release record bundle {key} has a blank {field}")
    return bundles


def _resolved_run_directory(run_directory: str | Path) -> Path:
    source = Path(run_directory)
    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise ReleaseRecordError(
            f"cannot resolve producer run directory {source}: {error}"
        ) from error
    if not resolved.is_dir():
        raise ReleaseRecordError("producer run source must be a directory")
    return resolved


def _run_directory_aggregate(resolved: Path) -> str:
    """Hash the exact evidence bytes with the analyzer's aggregate framing.

    The file set and framing mirror ``authenticate_producer_run``: every shard
    completion plus every file under ``cells/``, sorted by path, each framed as
    length-prefixed relative path and raw bytes.  Any structural disagreement
    (extra, missing, or tampered files) is caught when the authenticator
    revalidates the run against this value before the record is written.
    """

    paths = sorted(
        {
            *(path.resolve() for path in resolved.glob("shards/*/shard.complete.json")),
            *(path.resolve() for path in (resolved / "cells").rglob("*") if path.is_file()),
        },
        key=lambda path: path.as_posix(),
    )
    if not paths:
        raise ReleaseRecordError(f"producer run {resolved} contains no completion or cell files")
    aggregate = hashlib.sha256()
    for path in paths:
        try:
            relative = path.relative_to(resolved).as_posix().encode("utf-8")
        except ValueError as error:
            raise ReleaseRecordError(f"producer file escapes its run directory: {path}") from error
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ReleaseRecordError(f"cannot read producer file {path}: {error}") from error
        aggregate.update(len(relative).to_bytes(8, "big"))
        aggregate.update(relative)
        aggregate.update(len(raw).to_bytes(8, "big"))
        aggregate.update(raw)
    return aggregate.hexdigest()


def _content_digest(payload: Mapping[str, object]) -> str:
    content = {key: value for key, value in payload.items() if key != "content_digest"}
    try:
        encoded = canonical_json(content).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReleaseRecordError("release record is not finite canonical JSON") from error
    return sha256_hex(_RECORD_DOMAIN + encoded)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in _HEX for character in value)
    )
