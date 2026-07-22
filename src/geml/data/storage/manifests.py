"""Frozen manifest construction, atomic finalization, and integrity validation."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

from pydantic import BaseModel

from geml.contracts.corpus import (
    CorpusManifest,
    CorpusShardManifest,
    CorpusSplit,
    SplitManifest,
)
from geml.data.storage.dedup import DeduplicationStats
from geml.data.storage.shards import ShardIntegrityError, read_shard, sha256_file

_SPLIT_ORDER = (
    CorpusSplit.TRAIN,
    CorpusSplit.VALIDATION,
    CorpusSplit.TEST_IID,
    CorpusSplit.TEST_OOD,
)
_DEFAULT_PACKAGES = ("geml", "pyarrow", "pydantic", "sympy")


class ManifestIntegrityError(ValueError):
    """A manifest or referenced artifact is missing, inconsistent, or corrupted."""


@dataclass(frozen=True)
class ManifestValidationResult:
    """Complete retained validation outcome."""

    valid: bool
    errors: tuple[str, ...]
    validated_shard_count: int
    validated_row_count: int


@dataclass(frozen=True)
class ManifestBundlePaths:
    """Paths written during corpus finalization."""

    corpus_manifest: Path
    split_manifests: tuple[Path, ...]
    shard_manifests: tuple[Path, ...]


def build_split_manifest(
    shards: Sequence[CorpusShardManifest],
    *,
    metadata: Mapping[str, object] | None = None,
) -> SplitManifest:
    """Build one validated split manifest in contiguous shard-index order."""

    if not shards:
        raise ManifestIntegrityError("a split manifest requires at least one shard")
    ordered = tuple(sorted(shards, key=lambda shard: shard.shard_index))
    first = ordered[0]
    return SplitManifest(
        schema_version=first.schema_version,
        corpus_id=first.corpus_id,
        split=first.split,
        shards=ordered,
        total_row_count=sum(shard.row_count for shard in ordered),
        total_error_row_count=sum(shard.error_row_count for shard in ordered),
        metadata=dict(metadata or {}),
    )


def _package_versions(package_names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in package_names:
        if not package_name.strip():
            raise ManifestIntegrityError("package names must be nonblank")
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[package_name] = "unavailable"
    return versions


def _discover_git_commit(repository_path: str | Path | None) -> str:
    working_directory = None if repository_path is None else Path(repository_path)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_directory,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return "unavailable"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unavailable"


def build_corpus_manifest(
    split_manifests: Iterable[SplitManifest],
    *,
    corpus_id: str,
    schema_version: str,
    config_path: str | Path,
    generator_seed: int,
    repository_path: str | Path | None = None,
    git_commit: str | None = None,
    created_at: datetime | None = None,
    package_names: Iterable[str] = _DEFAULT_PACKAGES,
    deduplication_stats: DeduplicationStats | None = None,
    rejection_counts: Mapping[str, int] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CorpusManifest:
    """Build the frozen top-level manifest with complete reproducibility fields."""

    config = Path(config_path)
    if not config.is_file():
        raise ManifestIntegrityError(f"configuration file does not exist: {config}")
    if isinstance(generator_seed, bool) or not isinstance(generator_seed, int):
        raise ManifestIntegrityError("generator_seed must be an integer")

    by_split: dict[CorpusSplit, SplitManifest] = {}
    for split_manifest in split_manifests:
        if split_manifest.split in by_split:
            raise ManifestIntegrityError(
                f"duplicate split manifest for {split_manifest.split.value!r}"
            )
        by_split[split_manifest.split] = split_manifest
    if set(by_split) != set(CorpusSplit):
        missing = sorted(split.value for split in set(CorpusSplit) - set(by_split))
        raise ManifestIntegrityError(f"corpus manifest requires all four splits; missing={missing}")
    ordered = tuple(by_split[split] for split in _SPLIT_ORDER)

    retained_metadata = dict(metadata or {})
    if deduplication_stats is not None:
        retained_metadata["deduplication"] = asdict(deduplication_stats)
    if rejection_counts is not None:
        validated_rejections: dict[str, int] = {}
        for reason, count in rejection_counts.items():
            if not reason.strip():
                raise ManifestIntegrityError("rejection reasons must be nonblank")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ManifestIntegrityError("rejection counts must be nonnegative integers")
            validated_rejections[reason] = count
        retained_metadata["rejection_counts"] = validated_rejections

    resolved_commit = _discover_git_commit(repository_path) if git_commit is None else git_commit
    if not resolved_commit.strip():
        raise ManifestIntegrityError("git_commit must be nonblank when supplied")
    return CorpusManifest(
        schema_version=schema_version,
        corpus_id=corpus_id,
        splits=ordered,
        total_row_count=sum(split.total_row_count for split in ordered),
        total_error_row_count=sum(split.total_error_row_count for split in ordered),
        config_hash=sha256_file(config),
        generator_seed=generator_seed,
        created_at=created_at or datetime.now(UTC),
        git_commit=resolved_commit,
        python_version=sys.version,
        platform=platform.platform(),
        package_versions=_package_versions(package_names),
        metadata=retained_metadata,
    )


def _serialized_manifest(manifest: BaseModel) -> bytes:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (payload + "\n").encode("utf-8")


def _write_immutable_json(
    manifest: BaseModel,
    path: str | Path,
    *,
    resume: bool,
) -> Path:
    if not isinstance(resume, bool):
        raise ManifestIntegrityError("resume must be a boolean")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialized_manifest(manifest)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-manifest-",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Publish without replacement so a concurrent writer cannot overwrite an immutable
            # manifest between an existence check and the final filesystem operation.
            os.link(temporary_path, destination)
        except FileExistsError:
            if not resume:
                raise FileExistsError(f"immutable manifest already exists: {destination}") from None
            if destination.is_file() and destination.read_bytes() == payload:
                return destination
            raise ManifestIntegrityError(
                f"existing manifest differs from resumed output: {destination}"
            ) from None
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def write_shard_manifest(
    manifest: CorpusShardManifest,
    path: str | Path,
    *,
    resume: bool = True,
) -> Path:
    return _write_immutable_json(manifest, path, resume=resume)


def write_split_manifest(
    manifest: SplitManifest,
    path: str | Path,
    *,
    resume: bool = True,
) -> Path:
    return _write_immutable_json(manifest, path, resume=resume)


def write_corpus_manifest(
    manifest: CorpusManifest,
    path: str | Path,
    *,
    resume: bool = True,
) -> Path:
    return _write_immutable_json(manifest, path, resume=resume)


def write_manifest_bundle(
    manifest: CorpusManifest,
    output_dir: str | Path,
    *,
    resume: bool = True,
    artifact_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> ManifestBundlePaths:
    """Validate artifacts, write sidecars, then publish the corpus completion marker."""

    root = Path(output_dir)
    shard_root = root.parent if artifact_root is None else Path(artifact_root)
    validation = validate_manifest(manifest, shard_root, config_path=config_path)
    if not validation.valid:
        raise ManifestIntegrityError(
            "cannot finalize an invalid corpus: " + "; ".join(validation.errors)
        )
    shard_paths: list[Path] = []
    split_paths: list[Path] = []
    for split_manifest in manifest.splits:
        for shard in split_manifest.shards:
            shard_paths.append(
                write_shard_manifest(
                    shard,
                    _shard_sidecar_path(root, shard),
                    resume=resume,
                )
            )
        split_paths.append(
            write_split_manifest(
                split_manifest,
                _split_sidecar_path(root, split_manifest),
                resume=resume,
            )
        )
    sidecar_errors = _manifest_sidecar_errors(manifest, root, include_corpus=False)
    if sidecar_errors:
        raise ManifestIntegrityError("invalid manifest sidecars: " + "; ".join(sidecar_errors))
    corpus_path = write_corpus_manifest(
        manifest,
        root / "corpus.manifest.json",
        resume=resume,
    )
    return ManifestBundlePaths(
        corpus_manifest=corpus_path,
        split_manifests=tuple(split_paths),
        shard_manifests=tuple(shard_paths),
    )


def load_corpus_manifest(path: str | Path) -> CorpusManifest:
    """Load and strictly validate one frozen corpus manifest."""

    try:
        return CorpusManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise ManifestIntegrityError(f"invalid corpus manifest: {path}") from error


def _shard_sidecar_path(root: Path, shard: CorpusShardManifest) -> Path:
    return root / "shards" / f"{shard.split.value}-{shard.shard_index:05d}.manifest.json"


def _split_sidecar_path(root: Path, split_manifest: SplitManifest) -> Path:
    return root / "splits" / f"{split_manifest.split.value}.manifest.json"


def _manifest_sidecar_errors(
    manifest: CorpusManifest,
    manifest_dir: str | Path,
    *,
    include_corpus: bool,
) -> list[str]:
    root = Path(manifest_dir)
    expected: list[tuple[Path, BaseModel]] = []
    for split_manifest in manifest.splits:
        expected.extend(
            (_shard_sidecar_path(root, shard), shard) for shard in split_manifest.shards
        )
        expected.append((_split_sidecar_path(root, split_manifest), split_manifest))
    if include_corpus:
        expected.append((root / "corpus.manifest.json", manifest))

    errors: list[str] = []
    for path, expected_manifest in expected:
        if not path.is_file():
            errors.append(f"missing manifest sidecar: {path}")
            continue
        try:
            loaded = type(expected_manifest).model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            errors.append(f"invalid manifest sidecar: {path}")
            continue
        if loaded != expected_manifest:
            errors.append(f"manifest sidecar differs from corpus manifest: {path}")
    return errors


def validate_manifest(
    manifest: CorpusManifest,
    root_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    manifest_dir: str | Path | None = None,
) -> ManifestValidationResult:
    """Validate every referenced shard and retain all detected failures."""

    errors: list[str] = []
    validated_shards = 0
    validated_rows = 0
    if config_path is not None:
        config = Path(config_path)
        if not config.is_file():
            errors.append(f"missing configuration: {config}")
        elif sha256_file(config) != manifest.config_hash:
            errors.append(f"config checksum mismatch: {config}")

    for split_manifest in manifest.splits:
        for shard in split_manifest.shards:
            try:
                records = read_shard(shard, root_dir)
            except (OSError, ShardIntegrityError, ValueError) as error:
                errors.append(str(error))
                continue
            validated_shards += 1
            validated_rows += len(records)

    if manifest_dir is not None:
        errors.extend(_manifest_sidecar_errors(manifest, manifest_dir, include_corpus=True))

    if not errors and validated_rows != manifest.total_row_count:
        errors.append(
            f"validated row total {validated_rows} differs from manifest {manifest.total_row_count}"
        )
    return ManifestValidationResult(
        valid=not errors,
        errors=tuple(errors),
        validated_shard_count=validated_shards,
        validated_row_count=validated_rows,
    )
