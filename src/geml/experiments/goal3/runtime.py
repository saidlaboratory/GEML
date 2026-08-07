"""Atomic artifacts, reproducibility fingerprints, and resource telemetry for Goal 3."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import psutil


class Goal3ArtifactError(RuntimeError):
    """A Goal 3 artifact is missing, corrupt, or incompatible with the run."""


def canonical_json(value: object) -> str:
    """Serialize a JSON value deterministically without accepting non-finite floats."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``payload``."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file without loading it at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(
    path: str | Path,
    payload: bytes,
    *,
    resume_identical: bool = False,
) -> Path:
    """Publish bytes create-only after an fsync.

    A resumed caller may accept an existing byte-identical artifact. Different
    content is never overwritten.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-goal3-",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if resume_identical and destination.is_file() and destination.read_bytes() == payload:
                return destination
            raise Goal3ArtifactError(
                f"immutable artifact already exists with different content: {destination}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_write_json(
    path: str | Path,
    payload: object,
    *,
    resume_identical: bool = False,
) -> Path:
    """Publish a stable, human-readable JSON artifact create-only."""

    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return atomic_write_bytes(path, encoded, resume_identical=resume_identical)


def publish_temporary_file(
    temporary_path: str | Path,
    destination_path: str | Path,
) -> tuple[str, int]:
    """Publish a completed temporary file without replacing an existing artifact.

    If an identical file already exists, it is treated as an interrupted
    publication that can be resumed. A different file is a hard error.
    """

    temporary = Path(temporary_path)
    destination = Path(destination_path)
    if not temporary.is_file():
        raise Goal3ArtifactError(f"temporary artifact does not exist: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksum = sha256_file(temporary)
    byte_count = temporary.stat().st_size
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if not destination.is_file():
            raise Goal3ArtifactError(
                f"artifact destination is not a regular file: {destination}"
            ) from None
        if destination.stat().st_size != byte_count or sha256_file(destination) != checksum:
            raise Goal3ArtifactError(
                f"existing immutable artifact differs from resumed output: {destination}"
            ) from None
    return checksum, byte_count


def load_json_mapping(path: str | Path, *, label: str) -> dict[str, Any]:
    """Load one JSON object with a useful artifact error."""

    source = Path(path)
    if not source.is_file():
        raise Goal3ArtifactError(f"missing {label}: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise Goal3ArtifactError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise Goal3ArtifactError(f"{label} must contain a JSON object: {source}")
    return value


def package_versions(names: Iterable[str]) -> dict[str, str]:
    """Return installed versions for an ordered set of package names."""

    versions: dict[str, str] = {}
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("package names must be nonblank strings")
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def run_git(repository_root: str | Path, *arguments: str) -> str:
    """Run a bounded read-only Git query, returning ``unavailable`` on failure."""

    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=Path(repository_root),
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip()


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Stable environment and Git provenance captured for a run."""

    python_version: str
    python_implementation: str
    platform: str
    machine: str
    package_versions: Mapping[str, str]
    git_commit: str
    working_tree_dirty: bool
    working_tree_status: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""

        return {
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "machine": self.machine,
            "package_versions": dict(self.package_versions),
            "git_commit": self.git_commit,
            "working_tree_dirty": self.working_tree_dirty,
            "working_tree_status": list(self.working_tree_status),
        }


def capture_environment(
    repository_root: str | Path,
    *,
    packages: Iterable[str],
) -> EnvironmentInfo:
    """Capture runtime versions and the exact local Git state."""

    status_text = run_git(repository_root, "status", "--short")
    status_lines = (
        ()
        if status_text in {"", "unavailable"}
        else tuple(line for line in status_text.splitlines() if line)
    )
    return EnvironmentInfo(
        python_version=sys.version,
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        package_versions=package_versions(packages),
        git_commit=run_git(repository_root, "rev-parse", "HEAD"),
        working_tree_dirty=bool(status_lines),
        working_tree_status=status_lines,
    )


def executable_fingerprint(
    dependency_paths: Iterable[tuple[str, Path]],
    *,
    environment: EnvironmentInfo,
    additional_values: Mapping[str, object],
) -> str:
    """Hash every executable input that can affect a Goal 3 metric row."""

    digest = hashlib.sha256()
    seen_labels: set[str] = set()
    for label, path in sorted(dependency_paths, key=lambda item: item[0]):
        if label in seen_labels:
            continue
        seen_labels.add(label)
        if not path.is_file():
            raise Goal3ArtifactError(f"runner dependency is missing: {path}")
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    stable_environment = {
        "machine": environment.machine,
        "package_versions": dict(environment.package_versions),
        "platform": environment.platform,
        "python_implementation": environment.python_implementation,
        "python_version": environment.python_version,
    }
    digest.update(canonical_json(stable_environment).encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical_json(dict(additional_values)).encode("utf-8"))
    return digest.hexdigest()


def bounded_message(error: BaseException | str | None, *, maximum: int = 1_000) -> str | None:
    """Return one safe single-line error message with bounded storage."""

    if error is None:
        return None
    text = str(error).replace("\x00", "\\0").replace("\r", " ").replace("\n", " ")
    if not text:
        text = type(error).__name__ if isinstance(error, BaseException) else "unspecified error"
    return text if len(text) <= maximum else text[: maximum - 3] + "..."


def process_tree_resident_memory(process: psutil.Process) -> int:
    """Sample simultaneous RSS for a process and every live descendant."""

    try:
        total = int(process.memory_info().rss)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return 0
    for child in process.children(recursive=True):
        try:
            total += int(child.memory_info().rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return total


class PeakRSSMonitor:
    """Sample process-tree resident memory in a lightweight background thread."""

    def __init__(self, *, interval_seconds: float = 0.05) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._interval_seconds = interval_seconds
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0
        self._lock = threading.Lock()

    @property
    def peak_bytes(self) -> int:
        """Return the maximum sampled simultaneous RSS."""

        with self._lock:
            return self._peak

    def _sample(self) -> None:
        value = process_tree_resident_memory(self._process)
        with self._lock:
            self._peak = max(self._peak, value)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def __enter__(self) -> PeakRSSMonitor:
        self._sample()
        self._thread = threading.Thread(
            target=self._run,
            name="geml-goal3-rss-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 4))
        self._sample()
