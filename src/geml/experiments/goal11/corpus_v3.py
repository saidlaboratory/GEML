"""Resource-bounded workshop manifest assembly and integrity auditing.

The historical module name is retained for compatibility.  This module never
generates a corpus: it records and authenticates immutable workshop inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

WORKSHOP_MANIFEST_CONFIG_SCHEMA_VERSION = "geml-goal11-workshop-manifest-config-v1"
WORKSHOP_MANIFEST_SCHEMA_VERSION = "geml-goal11-workshop-manifest-v1"
WORKSHOP_MANIFEST_AUDIT_SCHEMA_VERSION = "geml-goal11-workshop-manifest-audit-v1"

_NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
_NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class WorkshopManifestError(ValueError):
    """The workshop manifest configuration or one referenced artifact is invalid."""


class WorkshopManifestValidationError(WorkshopManifestError):
    """Strict validation failed after a complete audit was retained."""

    def __init__(self, audit: WorkshopManifestAudit) -> None:
        self.audit = audit
        super().__init__(
            f"workshop manifest is not complete ({audit.required_error_count} required errors)"
        )


class EvidenceAuthenticationError(WorkshopManifestError):
    """A scientific row is not present in, or disagrees with, its frozen artifact."""


class EvidenceAuthenticationCache:
    """Per-audit cache for immutable file hashes and parsed structured artifacts."""

    def __init__(self) -> None:
        self.file_hashes: dict[Path, str] = {}
        self.payloads: dict[tuple[Path, ArtifactFormat], object] = {}


class CompletenessState(StrEnum):
    """Typed state of one expected workshop artifact."""

    COMPLETE = "complete"
    MISSING = "missing"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    DEFERRED = "deferred"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"


class AuditCode(StrEnum):
    """Machine-readable integrity finding independent of claimed state."""

    OK = "ok"
    MISSING_PATH = "missing_path"
    MISSING_ARTIFACT = "missing_artifact"
    DUPLICATE_ARTIFACT_ID = "duplicate_artifact_id"
    DUPLICATE_PATH = "duplicate_path"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    STALE_REFERENCE = "stale_reference"
    COUNT_MISMATCH = "count_mismatch"
    SEED_MISMATCH = "seed_mismatch"
    UNSAFE_PATH = "unsafe_path"
    INVALID_STATE = "invalid_state"


class ArtifactFormat(StrEnum):
    """Supported bounded metadata formats."""

    JSON = "json"
    JSONL = "jsonl"
    YAML = "yaml"
    OPAQUE = "opaque"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class PackageVersion(_FrozenModel):
    name: _NonBlank
    version: _NonBlank


class HardwareMetadata(_FrozenModel):
    platform: _NonBlank
    cpu: _NonBlank
    accelerator: _NonBlank | None = None
    accelerator_count: _NonNegativeInt = 0
    host_memory_bytes: _NonNegativeInt | None = None


class RunMetadata(_FrozenModel):
    """Reproducibility and resource fields copied from a frozen run envelope."""

    config_sha256: _Sha256
    git_commit: _NonBlank
    seeds: tuple[StrictInt, ...]
    package_versions: tuple[PackageVersion, ...]
    hardware: HardwareMetadata
    precision: _NonBlank | None = None
    parameter_count: _NonNegativeInt | None = None
    flops: _NonNegativeFloat | None = None
    wall_time_seconds: _NonNegativeFloat
    peak_host_memory_bytes: _NonNegativeInt | None = None
    peak_gpu_memory_bytes: _NonNegativeInt | None = None
    gpu_hours: _NonNegativeFloat | None = None
    attempted_count: _NonNegativeInt
    success_count: _NonNegativeInt
    failure_count: _NonNegativeInt
    invalid_count: _NonNegativeInt
    unsupported_count: _NonNegativeInt
    timeout_count: _NonNegativeInt
    reproduction_command: _NonBlank
    unavailable_fields: tuple[_NonBlank, ...] = ()

    @field_validator("seeds", "package_versions", "unavailable_fields", mode="before")
    @classmethod
    def normalize_yaml_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_run_metadata(self) -> Self:
        if len(set(self.seeds)) != len(self.seeds) or self.seeds != tuple(sorted(self.seeds)):
            raise ValueError("run metadata seeds must be unique and sorted")
        packages = tuple(item.name for item in self.package_versions)
        if len(set(packages)) != len(packages) or packages != tuple(sorted(packages)):
            raise ValueError("package versions must have unique sorted names")
        accounted = (
            self.success_count
            + self.failure_count
            + self.invalid_count
            + self.unsupported_count
            + self.timeout_count
        )
        if accounted != self.attempted_count:
            raise ValueError("run metadata outcomes must account for every attempt")
        optional_fields = {
            "precision",
            "parameter_count",
            "flops",
            "peak_host_memory_bytes",
            "peak_gpu_memory_bytes",
            "gpu_hours",
        }
        unavailable = set(self.unavailable_fields)
        if unavailable - optional_fields:
            raise ValueError("unavailable_fields contains an unknown resource field")
        if len(unavailable) != len(self.unavailable_fields) or self.unavailable_fields != tuple(
            sorted(unavailable)
        ):
            raise ValueError("unavailable_fields must be unique and sorted")
        observed_missing = {name for name in optional_fields if getattr(self, name) is None}
        if unavailable != observed_missing:
            raise ValueError("every unavailable run field must be named explicitly")
        return self


def _canonical_relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("artifact paths must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("artifact paths must be canonical relative POSIX paths")
    return value


class ExpectedArtifact(_FrozenModel):
    """One canonical artifact role expected by the workshop plan."""

    artifact_id: _NonBlank
    producer_issue: _NonBlank
    category: _NonBlank
    roles: tuple[_NonBlank, ...] = Field(min_length=1)
    required: StrictBool = True
    relative_path: _NonBlank | None = None
    artifact_format: ArtifactFormat = ArtifactFormat.OPAQUE
    expected_sha256: _Sha256 | None = None
    expected_schema_version: _NonBlank | None = None
    schema_field: _NonBlank = "schema_version"
    expected_count: _NonNegativeInt | None = None
    count_field: _NonBlank | None = None
    seed: StrictInt | None = None
    configured_state: CompletenessState | None = None
    configured_reason: _NonBlank | None = None
    run_metadata_required: StrictBool = False
    requires_expected_seeds: StrictBool = False
    run_metadata: RunMetadata | None = None

    @field_validator("roles", mode="before")
    @classmethod
    def normalize_yaml_roles(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("artifact_format", mode="before")
    @classmethod
    def normalize_yaml_format(cls, value: object) -> object:
        return ArtifactFormat(value) if isinstance(value, str) else value

    @field_validator("configured_state", mode="before")
    @classmethod
    def normalize_yaml_state(cls, value: object) -> object:
        return CompletenessState(value) if isinstance(value, str) else value

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_relative_path(value)

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or value != tuple(sorted(value)):
            raise ValueError("roles must be unique and sorted")
        return value

    @model_validator(mode="after")
    def validate_configured_state(self) -> Self:
        if self.configured_state in {
            CompletenessState.COMPLETE,
            CompletenessState.CHECKSUM_MISMATCH,
            CompletenessState.SCHEMA_MISMATCH,
            CompletenessState.DEFERRED,
        }:
            raise ValueError("configured_state may only declare missing, failed, or unsupported")
        if (self.configured_state is None) != (self.configured_reason is None):
            raise ValueError("configured_state and configured_reason must be supplied together")
        if self.count_field is not None and self.artifact_format not in {
            ArtifactFormat.JSON,
            ArtifactFormat.YAML,
        }:
            raise ValueError("count_field is only valid for JSON or YAML artifacts")
        if self.relative_path is not None and self.expected_count is not None:
            count_is_observable = self.artifact_format is ArtifactFormat.JSONL or (
                self.artifact_format in {ArtifactFormat.JSON, ArtifactFormat.YAML}
                and self.count_field is not None
            )
            if not count_is_observable:
                raise ValueError(
                    "a path-backed expected_count requires JSONL or a JSON/YAML count_field"
                )
        return self


class DeferredExperiment(_FrozenModel):
    """One explicitly planned experiment that is outside workshop scope."""

    experiment_id: _NonBlank
    reason: _NonBlank
    state: Literal[CompletenessState.DEFERRED] = CompletenessState.DEFERRED


class Goal11ManifestConfig(_FrozenModel):
    """Checked-in declaration of workshop inputs and deliberate deferrals."""

    schema_version: Literal["geml-goal11-workshop-manifest-config-v1"] = (
        WORKSHOP_MANIFEST_CONFIG_SCHEMA_VERSION
    )
    artifact_root: _NonBlank = "."
    output_root: _NonBlank = "outputs/final/goal11/manifest"
    expected_seeds: tuple[StrictInt, StrictInt, StrictInt]
    artifacts: tuple[ExpectedArtifact, ...]
    deferred_experiments: tuple[DeferredExperiment, ...]

    @field_validator(
        "expected_seeds",
        "artifacts",
        "deferred_experiments",
        mode="before",
    )
    @classmethod
    def normalize_yaml_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("artifact_root", "output_root")
    @classmethod
    def validate_root(cls, value: str) -> str:
        return _canonical_relative_path(value) if value != "." else value

    @model_validator(mode="after")
    def validate_unique_contract(self) -> Self:
        if len(set(self.expected_seeds)) != 3:
            raise ValueError("expected_seeds must contain exactly three distinct seeds")
        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifact IDs must be unique")
        configured_paths = tuple(
            item.relative_path for item in self.artifacts if item.relative_path is not None
        )
        if len(set(configured_paths)) != len(configured_paths):
            raise ValueError("configured artifact paths must be unique")
        deferred_ids = tuple(item.experiment_id for item in self.deferred_experiments)
        if len(set(deferred_ids)) != len(deferred_ids):
            raise ValueError("deferred experiment IDs must be unique")
        if set(artifact_ids) & set(deferred_ids):
            raise ValueError("artifact and deferred experiment IDs must be disjoint")
        return self


class ArtifactReference(_FrozenModel):
    """One observed artifact entry in the frozen workshop manifest."""

    artifact_id: _NonBlank
    producer_issue: _NonBlank
    category: _NonBlank
    roles: tuple[_NonBlank, ...]
    required: StrictBool
    relative_path: _NonBlank | None = None
    artifact_format: ArtifactFormat
    state: CompletenessState
    reason: _NonBlank | None = None
    expected_sha256: _Sha256 | None = None
    observed_sha256: _Sha256 | None = None
    size_bytes: _NonNegativeInt | None = None
    expected_schema_version: _NonBlank | None = None
    observed_schema_version: _NonBlank | None = None
    expected_count: _NonNegativeInt | None = None
    observed_count: _NonNegativeInt | None = None
    seed: StrictInt | None = None
    run_metadata: RunMetadata | None = None
    requires_expected_seeds: StrictBool = False

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        return None if value is None else _canonical_relative_path(value)

    @model_validator(mode="after")
    def validate_state_evidence(self) -> Self:
        if self.state is CompletenessState.DEFERRED:
            raise ValueError("deferred work belongs in deferred_experiments")
        if self.state is CompletenessState.COMPLETE:
            if (
                self.relative_path is None
                or self.observed_sha256 is None
                or self.size_bytes is None
            ):
                raise ValueError("complete artifacts require path, checksum, and size")
            if self.reason is not None:
                raise ValueError("complete artifacts cannot carry a failure reason")
            if self.expected_sha256 is not None and self.expected_sha256 != self.observed_sha256:
                raise ValueError("complete artifact checksum disagrees with expectation")
            if (
                self.expected_schema_version is not None
                and self.expected_schema_version != self.observed_schema_version
            ):
                raise ValueError("complete artifact schema disagrees with expectation")
            if self.expected_count is not None and self.expected_count != self.observed_count:
                raise ValueError("complete artifact count disagrees with expectation")
        elif self.reason is None:
            raise ValueError("non-complete artifacts require a reason")
        return self


class WorkshopManifest(_FrozenModel):
    """Canonical references for the resource-bounded workshop program."""

    schema_version: Literal["geml-goal11-workshop-manifest-v1"] = WORKSHOP_MANIFEST_SCHEMA_VERSION
    config_sha256: _Sha256
    expected_seeds: tuple[StrictInt, StrictInt, StrictInt]
    artifacts: tuple[ArtifactReference, ...]
    deferred_experiments: tuple[DeferredExperiment, ...]

    @model_validator(mode="after")
    def validate_unique_entries(self) -> Self:
        ids = tuple(item.artifact_id for item in self.artifacts)
        if len(set(ids)) != len(ids):
            raise ValueError("manifest artifact IDs must be unique")
        paths = tuple(
            item.relative_path for item in self.artifacts if item.relative_path is not None
        )
        if len(set(paths)) != len(paths):
            raise ValueError("manifest artifact paths must be unique")
        return self


class ManifestAuditItem(_FrozenModel):
    """Audited effective state of one expected artifact."""

    artifact_id: _NonBlank
    required: StrictBool
    declared_state: CompletenessState
    effective_state: CompletenessState
    codes: tuple[AuditCode, ...] = Field(min_length=1)
    messages: tuple[_NonBlank, ...] = Field(min_length=1)


class WorkshopManifestAudit(_FrozenModel):
    """Complete integrity audit retained even when strict validation fails."""

    schema_version: Literal["geml-goal11-workshop-manifest-audit-v1"] = (
        WORKSHOP_MANIFEST_AUDIT_SCHEMA_VERSION
    )
    manifest_sha256: _Sha256
    valid: StrictBool
    required_error_count: _NonNegativeInt
    items: tuple[ManifestAuditItem, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        observed = sum(
            item.required and item.effective_state is not CompletenessState.COMPLETE
            for item in self.items
        )
        if observed != self.required_error_count:
            raise ValueError("required_error_count disagrees with audit items")
        if self.valid != (observed == 0):
            raise ValueError("valid disagrees with required_error_count")
        return self


def canonical_json_bytes(value: BaseModel | Mapping[str, object]) -> bytes:
    """Return deterministic UTF-8 JSON bytes."""

    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def canonical_json_value_bytes(value: object) -> bytes:
    """Return canonical JSON bytes for one already-validated evidence value."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: str | Path) -> tuple[str, int]:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def manifest_sha256(manifest: WorkshopManifest) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise WorkshopManifestError(f"could not read manifest config: {path}") from error
    if not isinstance(payload, dict):
        raise WorkshopManifestError("workshop manifest config must be a YAML mapping")
    return payload


def load_manifest_config(path: str | Path) -> Goal11ManifestConfig:
    """Load and strictly validate the workshop manifest configuration."""

    config_path = Path(path)
    try:
        return Goal11ManifestConfig.model_validate(_load_yaml_mapping(config_path))
    except ValidationError as error:
        raise WorkshopManifestError("invalid workshop manifest configuration") from error


def _resolve_inside(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise WorkshopManifestError(
            f"artifact path escapes artifact root: {relative_path}"
        ) from error
    return candidate


def _resolve_json_pointer(payload: object, pointer: str) -> object:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise EvidenceAuthenticationError("JSON evidence locators must be JSON pointers")
    current = payload
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise EvidenceAuthenticationError(
                    f"evidence locator does not exist at token {token!r}"
                )
            current = current[token]
        elif isinstance(current, list):
            if not token.isdecimal():
                raise EvidenceAuthenticationError(
                    f"list evidence locator token is not an index: {token!r}"
                )
            index = int(token)
            if index >= len(current):
                raise EvidenceAuthenticationError(
                    f"evidence locator index is out of range: {index}"
                )
            current = current[index]
        else:
            raise EvidenceAuthenticationError(
                f"evidence locator traverses a scalar at token {token!r}"
            )
    return current


def _load_evidence_record(
    path: Path,
    artifact_format: ArtifactFormat,
    locator: str,
    cache: EvidenceAuthenticationCache | None,
) -> Mapping[str, object]:
    try:
        if artifact_format in {ArtifactFormat.JSON, ArtifactFormat.YAML}:
            cache_key = (path, artifact_format)
            if cache is not None and cache_key in cache.payloads:
                payload = cache.payloads[cache_key]
            else:
                payload = _structured_payload(path, artifact_format)
                if cache is not None:
                    cache.payloads[cache_key] = payload
            pointer = (
                locator.removeprefix("json-pointer:")
                if locator.startswith("json-pointer:")
                else locator
            )
            record = _resolve_json_pointer(payload, pointer)
        elif artifact_format is ArtifactFormat.JSONL:
            prefix = "jsonl-line:"
            if not locator.startswith(prefix) or not locator[len(prefix) :].isdecimal():
                raise EvidenceAuthenticationError(
                    "JSONL evidence locators must use jsonl-line:<one-based-record>"
                )
            target = int(locator[len(prefix) :])
            if target < 1:
                raise EvidenceAuthenticationError("JSONL evidence record numbers start at one")
            cache_key = (path, artifact_format)
            if cache is not None and cache_key in cache.payloads:
                records = cache.payloads[cache_key]
            else:
                with path.open("r", encoding="utf-8") as stream:
                    records = tuple(json.loads(line) for line in stream if line.strip())
                if cache is not None:
                    cache.payloads[cache_key] = records
            if not isinstance(records, tuple):
                raise EvidenceAuthenticationError("cached JSONL evidence is invalid")
            if target > len(records):
                raise EvidenceAuthenticationError(
                    f"JSONL evidence locator is out of range: {target}"
                )
            record = records[target - 1]
        else:
            raise EvidenceAuthenticationError("opaque artifacts cannot authenticate result rows")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise EvidenceAuthenticationError(
            f"could not load cited evidence row: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(record, dict):
        raise EvidenceAuthenticationError("the cited evidence locator must resolve to a mapping")
    return record


def _assert_projection(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    prefix: str = "",
) -> None:
    for key, expected_value in expected.items():
        field = f"{prefix}.{key}" if prefix else key
        if key not in observed:
            raise EvidenceAuthenticationError(f"cited evidence row is missing {field}")
        observed_value = observed[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(observed_value, dict):
                raise EvidenceAuthenticationError(f"cited evidence field {field} is not a mapping")
            _assert_projection(observed_value, expected_value, prefix=field)
        elif canonical_json_value_bytes(observed_value) != canonical_json_value_bytes(
            expected_value
        ):
            raise EvidenceAuthenticationError(f"cited evidence field disagrees: {field}")


def authenticate_source_record(
    manifest: WorkshopManifest,
    artifact_root: str | Path,
    *,
    artifact_id: str,
    source_sha256: str,
    source_locator: str,
    expected_fields: Mapping[str, object],
    required_roles: tuple[str, ...],
    allowed_categories: tuple[str, ...] = (),
    cache: EvidenceAuthenticationCache | None = None,
) -> Mapping[str, object]:
    """Authenticate one exact source row and its scientific field projection."""

    reference = next(
        (item for item in manifest.artifacts if item.artifact_id == artifact_id),
        None,
    )
    if reference is None:
        raise EvidenceAuthenticationError(f"unknown evidence artifact: {artifact_id}")
    if (
        reference.state is not CompletenessState.COMPLETE
        or reference.observed_sha256 != source_sha256
    ):
        raise EvidenceAuthenticationError(
            f"evidence artifact is not complete/authenticated: {artifact_id}"
        )
    if required_roles and not set(required_roles).intersection(reference.roles):
        raise EvidenceAuthenticationError(f"evidence artifact {artifact_id} lacks an approved role")
    if allowed_categories and reference.category not in allowed_categories:
        raise EvidenceAuthenticationError(
            f"evidence artifact {artifact_id} has category {reference.category!r}"
        )
    if reference.relative_path is None:
        raise EvidenceAuthenticationError(f"evidence artifact {artifact_id} has no path")
    try:
        path = _resolve_inside(Path(artifact_root), reference.relative_path)
    except WorkshopManifestError as error:
        raise EvidenceAuthenticationError(str(error)) from error
    if not path.is_file():
        raise EvidenceAuthenticationError(f"evidence artifact is missing: {artifact_id}")
    if cache is not None and path in cache.file_hashes:
        observed_sha256 = cache.file_hashes[path]
    else:
        try:
            observed_sha256, _ = sha256_file(path)
        except OSError as error:
            raise EvidenceAuthenticationError(
                f"evidence artifact could not be read: {type(error).__name__}: {error}"
            ) from error
        if cache is not None:
            cache.file_hashes[path] = observed_sha256
    if observed_sha256 != source_sha256:
        raise EvidenceAuthenticationError(
            f"evidence artifact changed after manifest assembly: {artifact_id}"
        )
    record = _load_evidence_record(
        path,
        reference.artifact_format,
        source_locator,
        cache,
    )
    try:
        _assert_projection(record, expected_fields)
    except (TypeError, ValueError) as error:
        raise EvidenceAuthenticationError(
            f"cited evidence contains a non-canonical value: {error}"
        ) from error
    return record


def _structured_payload(path: Path, artifact_format: ArtifactFormat) -> object:
    if artifact_format is ArtifactFormat.JSON:
        return json.loads(path.read_text(encoding="utf-8"))
    if artifact_format is ArtifactFormat.YAML:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if artifact_format is ArtifactFormat.JSONL:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    return json.loads(line)
        return None
    return None


def _observed_metadata(
    path: Path,
    expected: ExpectedArtifact,
) -> tuple[str | None, int | None]:
    if expected.artifact_format is ArtifactFormat.OPAQUE:
        return None, None
    payload = _structured_payload(path, expected.artifact_format)
    if expected.artifact_format is ArtifactFormat.JSONL:
        count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    elif expected.count_field is not None and isinstance(payload, dict):
        raw_count = payload.get(expected.count_field)
        count = (
            raw_count if isinstance(raw_count, int) and not isinstance(raw_count, bool) else None
        )
    else:
        count = None
    schema = (
        payload.get(expected.schema_field)
        if isinstance(payload, dict) and isinstance(payload.get(expected.schema_field), str)
        else None
    )
    return schema, count


def _noncomplete_reference(
    expected: ExpectedArtifact,
    state: CompletenessState,
    reason: str,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=expected.artifact_id,
        producer_issue=expected.producer_issue,
        category=expected.category,
        roles=expected.roles,
        required=expected.required,
        relative_path=expected.relative_path,
        artifact_format=expected.artifact_format,
        state=state,
        reason=reason,
        expected_sha256=expected.expected_sha256,
        expected_schema_version=expected.expected_schema_version,
        expected_count=expected.expected_count,
        seed=expected.seed,
        run_metadata=expected.run_metadata,
        requires_expected_seeds=expected.requires_expected_seeds,
    )


def _observe_artifact(
    expected: ExpectedArtifact,
    root: Path,
    expected_seeds: tuple[int, int, int],
) -> ArtifactReference:
    if expected.configured_state is not None:
        return _noncomplete_reference(
            expected,
            expected.configured_state,
            expected.configured_reason or "configured non-complete state",
        )
    if expected.relative_path is None:
        return _noncomplete_reference(
            expected,
            CompletenessState.MISSING,
            "producer manifest path is not frozen",
        )
    path = _resolve_inside(root, expected.relative_path)
    if not path.is_file():
        return _noncomplete_reference(
            expected,
            CompletenessState.MISSING,
            f"artifact does not exist: {expected.relative_path}",
        )
    try:
        observed_sha256, size_bytes = sha256_file(path)
        observed_schema, observed_count = _observed_metadata(path, expected)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        return _noncomplete_reference(
            expected,
            CompletenessState.FAILED,
            f"artifact could not be inspected: {type(error).__name__}: {error}",
        )
    common = {
        "artifact_id": expected.artifact_id,
        "producer_issue": expected.producer_issue,
        "category": expected.category,
        "roles": expected.roles,
        "required": expected.required,
        "relative_path": expected.relative_path,
        "artifact_format": expected.artifact_format,
        "expected_sha256": expected.expected_sha256,
        "observed_sha256": observed_sha256,
        "size_bytes": size_bytes,
        "expected_schema_version": expected.expected_schema_version,
        "observed_schema_version": observed_schema,
        "expected_count": expected.expected_count,
        "observed_count": observed_count,
        "seed": expected.seed,
        "run_metadata": expected.run_metadata,
        "requires_expected_seeds": expected.requires_expected_seeds,
    }
    if expected.expected_sha256 is not None and observed_sha256 != expected.expected_sha256:
        return ArtifactReference(
            **common,
            state=CompletenessState.CHECKSUM_MISMATCH,
            reason="observed checksum differs from the frozen checksum",
        )
    if (
        expected.expected_schema_version is not None
        and observed_schema != expected.expected_schema_version
    ):
        return ArtifactReference(
            **common,
            state=CompletenessState.SCHEMA_MISMATCH,
            reason="observed schema version differs from the frozen schema",
        )
    if expected.expected_count is not None and observed_count != expected.expected_count:
        return ArtifactReference(
            **common,
            state=CompletenessState.FAILED,
            reason="observed count differs from the frozen count",
        )
    if expected.run_metadata_required and expected.run_metadata is None:
        return ArtifactReference(
            **common,
            state=CompletenessState.FAILED,
            reason="completed run artifact lacks required run-envelope metadata",
        )
    if (
        expected.requires_expected_seeds
        and expected.run_metadata is not None
        and expected.run_metadata.seeds != tuple(sorted(expected_seeds))
    ):
        return ArtifactReference(
            **common,
            state=CompletenessState.FAILED,
            reason="run metadata seed set differs from the frozen workshop seeds",
        )
    return ArtifactReference(**common, state=CompletenessState.COMPLETE)


def assemble_workshop_manifest(
    config: Goal11ManifestConfig,
    artifact_root: str | Path,
    *,
    config_sha256: str,
) -> WorkshopManifest:
    """Inspect expected inputs without generating or modifying any producer artifact."""

    root = Path(artifact_root)
    return WorkshopManifest(
        config_sha256=config_sha256,
        expected_seeds=config.expected_seeds,
        artifacts=tuple(
            _observe_artifact(item, root, config.expected_seeds) for item in config.artifacts
        ),
        deferred_experiments=config.deferred_experiments,
    )


def _audit_reference(reference: ArtifactReference, root: Path) -> ManifestAuditItem:
    codes: list[AuditCode] = []
    messages: list[str] = []
    effective = reference.state
    if reference.state is CompletenessState.COMPLETE:
        if reference.relative_path is None:
            effective = CompletenessState.MISSING
            codes.append(AuditCode.MISSING_PATH)
            messages.append("complete artifact has no path")
        else:
            try:
                path = _resolve_inside(root, reference.relative_path)
            except WorkshopManifestError as error:
                effective = CompletenessState.FAILED
                codes.append(AuditCode.UNSAFE_PATH)
                messages.append(str(error))
            else:
                if not path.is_file():
                    effective = CompletenessState.MISSING
                    codes.append(AuditCode.MISSING_ARTIFACT)
                    messages.append(f"artifact no longer exists: {reference.relative_path}")
                else:
                    try:
                        observed_sha256, _ = sha256_file(path)
                    except OSError as error:
                        effective = CompletenessState.FAILED
                        codes.append(AuditCode.INVALID_STATE)
                        messages.append(
                            "artifact could not be re-authenticated: "
                            f"{type(error).__name__}: {error}"
                        )
                    if (
                        effective is CompletenessState.COMPLETE
                        and observed_sha256 != reference.observed_sha256
                    ):
                        effective = CompletenessState.CHECKSUM_MISMATCH
                        codes.append(AuditCode.CHECKSUM_MISMATCH)
                        messages.append("artifact bytes changed after manifest assembly")
    if not codes:
        state_code = {
            CompletenessState.COMPLETE: AuditCode.OK,
            CompletenessState.MISSING: AuditCode.MISSING_ARTIFACT,
            CompletenessState.CHECKSUM_MISMATCH: AuditCode.CHECKSUM_MISMATCH,
            CompletenessState.SCHEMA_MISMATCH: AuditCode.SCHEMA_MISMATCH,
            CompletenessState.FAILED: AuditCode.INVALID_STATE,
            CompletenessState.UNSUPPORTED: AuditCode.INVALID_STATE,
        }[effective]
        codes.append(state_code)
        messages.append(
            "artifact is authenticated"
            if effective is CompletenessState.COMPLETE
            else reference.reason or f"artifact state is {effective.value}"
        )
    return ManifestAuditItem(
        artifact_id=reference.artifact_id,
        required=reference.required,
        declared_state=reference.state,
        effective_state=effective,
        codes=tuple(codes),
        messages=tuple(messages),
    )


def audit_workshop_manifest(
    manifest: WorkshopManifest,
    artifact_root: str | Path,
) -> WorkshopManifestAudit:
    """Re-authenticate every complete artifact and retain every non-complete state."""

    root = Path(artifact_root)
    items = tuple(_audit_reference(item, root) for item in manifest.artifacts)
    errors = sum(
        item.required and item.effective_state is not CompletenessState.COMPLETE for item in items
    )
    return WorkshopManifestAudit(
        manifest_sha256=manifest_sha256(manifest),
        valid=errors == 0,
        required_error_count=errors,
        items=items,
    )


def require_valid_workshop_manifest(audit: WorkshopManifestAudit) -> None:
    """Fail loudly while preserving the caller-visible audit object."""

    if not audit.valid:
        raise WorkshopManifestValidationError(audit)


def _write_immutable_json(value: BaseModel, path: Path) -> Path:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return path
        raise FileExistsError(f"immutable output already exists with different bytes: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-goal11-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def run_manifest_audit(
    config_path: str | Path,
    artifact_root: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
    *,
    require_complete: bool = True,
) -> tuple[WorkshopManifest, WorkshopManifestAudit]:
    """Assemble and publish a manifest plus audit; strict failure happens last."""

    config_file = Path(config_path)
    config_sha256, _ = sha256_file(config_file)
    config = load_manifest_config(config_file)
    manifest = assemble_workshop_manifest(
        config,
        artifact_root,
        config_sha256=config_sha256,
    )
    audit = audit_workshop_manifest(manifest, artifact_root)
    _write_immutable_json(manifest, Path(manifest_path))
    _write_immutable_json(audit, Path(audit_path))
    if require_complete:
        require_valid_workshop_manifest(audit)
    return manifest, audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--audit-out", required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="publish the audit without requiring every workshop input",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    args = _parser().parse_args(argv)
    run_manifest_audit(
        args.config,
        args.artifact_root,
        args.manifest_out,
        args.audit_out,
        require_complete=not args.allow_incomplete,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
