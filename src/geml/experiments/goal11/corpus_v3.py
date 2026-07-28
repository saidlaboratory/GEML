"""Machine-readable resource-bounded workshop run manifest; no corpus-v3 generation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

WORKSHOP_MANIFEST_SCHEMA_VERSION = "geml-workshop-run-manifest-v1"
_QUALIFIED_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ArtifactStatus(StrEnum):
    COMPLETE = "complete"
    DEFERRED = "deferred"


class ArtifactAuditStatus(StrEnum):
    VERIFIED = "verified"
    DEFERRED = "deferred"
    MISSING = "missing"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    INVALID_PATH = "invalid_path"


@dataclass(frozen=True, slots=True)
class WorkshopArtifactV1:
    name: str
    path: str
    checksum: str
    status: str

    def __post_init__(self) -> None:
        if not self.name or not self.path or not _QUALIFIED_SHA256.fullmatch(self.checksum):
            raise ValueError("workshop artifact identity must be explicit and checksummed")
        if self.status not in {status.value for status in ArtifactStatus}:
            raise ValueError("workshop artifact status must be complete or deferred")
        artifact_path = Path(self.path)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise ValueError("workshop artifact path must be a safe relative path")


@dataclass(frozen=True, slots=True)
class WorkshopRunManifestV1:
    artifacts: tuple[WorkshopArtifactV1, ...]
    deferred_experiments: tuple[str, ...]
    content_digest: str
    schema_version: str = WORKSHOP_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKSHOP_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unexpected workshop manifest schema")
        if len({item.name for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("workshop manifest artifact names must be unique")
        if self.content_digest != manifest_digest(self.artifacts, self.deferred_experiments):
            raise ValueError("workshop manifest digest does not bind its contents")


def manifest_digest(
    artifacts: tuple[WorkshopArtifactV1, ...],
    deferred_experiments: tuple[str, ...],
) -> str:
    payload = {
        "artifacts": [asdict(item) for item in artifacts],
        "deferred": list(deferred_experiments),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactAuditV1:
    name: str
    status: ArtifactAuditStatus
    detail: str


@dataclass(frozen=True, slots=True)
class WorkshopManifestAuditV1:
    manifest_digest: str
    artifacts: tuple[ArtifactAuditV1, ...]

    @property
    def verified(self) -> bool:
        return all(row.status is ArtifactAuditStatus.VERIFIED for row in self.artifacts)


def require_artifact_names(
    manifest: WorkshopRunManifestV1,
    required_names: tuple[str, ...],
) -> None:
    """Fail loudly if a planned workshop assembly omits a canonical artifact name."""

    present = {artifact.name for artifact in manifest.artifacts}
    missing = sorted(set(required_names) - present)
    if missing:
        raise ValueError(f"workshop manifest is missing required artifacts: {missing}")


def audit_manifest(
    manifest: WorkshopRunManifestV1,
    *,
    artifact_root: Path,
) -> WorkshopManifestAuditV1:
    """Verify staged artifact bytes and retain missing/deferred rows rather than hiding them."""

    root = artifact_root.resolve()
    rows = []
    for artifact in manifest.artifacts:
        candidate = (root / artifact.path).resolve()
        if not candidate.is_relative_to(root):
            rows.append(
                ArtifactAuditV1(artifact.name, ArtifactAuditStatus.INVALID_PATH, artifact.path)
            )
        elif artifact.status == ArtifactStatus.DEFERRED.value:
            rows.append(
                ArtifactAuditV1(artifact.name, ArtifactAuditStatus.DEFERRED, "explicitly deferred")
            )
        elif not candidate.is_file():
            rows.append(ArtifactAuditV1(artifact.name, ArtifactAuditStatus.MISSING, artifact.path))
        elif _file_digest(candidate) != artifact.checksum:
            rows.append(
                ArtifactAuditV1(
                    artifact.name,
                    ArtifactAuditStatus.CHECKSUM_MISMATCH,
                    artifact.path,
                )
            )
        else:
            rows.append(ArtifactAuditV1(artifact.name, ArtifactAuditStatus.VERIFIED, artifact.path))
    return WorkshopManifestAuditV1(manifest.content_digest, tuple(rows))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
