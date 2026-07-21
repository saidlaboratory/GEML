"""
manifests.py - corpus/shard manifests with checksums for validation

owned by 1-5

a manifest is just a JSON file recording what SHOULD be on disk (which
shards, how many rows, what their checksums are, plus metadata for
reproducibility). validate_manifest() checks reality against that
record and reports exactly what's wrong instead of silently trusting
whatever's there.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import time

from geml.data.storage.shards import ShardInfo


def _file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_hash() -> str | None:
    # "when available" - just returns None if git's not around or this
    # isn't actually a git repo, doesn't blow up either way
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def _config_hash(config_path: Path | None) -> str | None:
    if config_path is None:
        return None
    config_path = Path(config_path)
    if not config_path.exists():
        return None
    return _file_checksum(config_path)


@dataclass
class ShardManifestEntry:
    path: str  # stored as string, not Path, so this round-trips cleanly through JSON
    row_count: int
    checksum: str


@dataclass
class CorpusManifest:
    shards: list[ShardManifestEntry]
    total_rows: int
    config_hash: str | None
    python_version: str
    package_versions: dict[str, str]
    timestamp: float
    git_commit_hash: str | None = None


def build_manifest(
    shards: list[ShardInfo],
    config_path: Path | None = None,
    extra_package_versions: dict[str, str] | None = None,
) -> CorpusManifest:
    import pyarrow

    package_versions = {"pyarrow": pyarrow.__version__}
    if extra_package_versions:
        package_versions.update(extra_package_versions)

    entries = [
        ShardManifestEntry(
            path=str(s.path),
            row_count=s.row_count,
            checksum=_file_checksum(s.path),
        )
        for s in shards
    ]

    return CorpusManifest(
        shards=entries,
        total_rows=sum(s.row_count for s in shards),
        config_hash=_config_hash(config_path),
        python_version=sys.version,
        package_versions=package_versions,
        timestamp=time.time(),
        git_commit_hash=_git_commit_hash(),
    )


def write_manifest(manifest: CorpusManifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)


def load_manifest(path: Path) -> CorpusManifest:
    with open(path) as f:
        data = json.load(f)
    data["shards"] = [ShardManifestEntry(**s) for s in data["shards"]]
    return CorpusManifest(**data)


@dataclass
class ManifestValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


def validate_manifest(manifest: CorpusManifest) -> ManifestValidationResult:
    """
    checks every shard the manifest claims should exist actually does,
    and that its checksum still matches. a missing file OR a mismatched
    checksum both get reported explicitly - never just "looks fine"
    when it isn't.
    """
    errors = []
    for entry in manifest.shards:
        p = Path(entry.path)
        if not p.exists():
            errors.append(f"missing shard: {entry.path}")
            continue
        actual_checksum = _file_checksum(p)
        if actual_checksum != entry.checksum:
            errors.append(
                f"checksum mismatch for {entry.path}: "
                f"expected {entry.checksum}, got {actual_checksum}"
            )
    return ManifestValidationResult(valid=(len(errors) == 0), errors=errors)
