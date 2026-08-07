"""Core, dependency-free helpers for reproducible GEML runs.

This module deliberately uses only the Python standard library.  In particular,
artifact authentication, smoke-test planning, and provider cost guards remain
available before optional ML or provider dependencies are installed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import tarfile
import time
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_MANIFEST_PATH = Path(__file__).with_name("ARTIFACT_SOURCES.json")
SMOKE_TIMEOUT_SECONDS = 29 * 60
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "openai": "OPENAI_API_KEY",
}
_FORBIDDEN_MODEL_ALIASES = {"auto", "default", "latest"}
_SPEND_APPROVAL = "I-CONFIRM-PAID-API-CALLS"
_TREE_HASH_SCHEMES = {"geml-artifact-tree-v1", "path-sha256-lines-v1"}
_GOAL_AVAILABILITY_STATES = {"complete", "deferred", "missing", "private", "unavailable"}
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r"(?:api[_-]?key|auth|connection[_-]?string|cookie|credential|database[_-]?url|"
    r"password|private[_-]?key|secret|session|token)",
    re.IGNORECASE,
)
_SENSITIVE_ENVIRONMENT_PREFIXES = (
    "ANTHROPIC_",
    "AWS_",
    "AZURE_",
    "GCP_",
    "GEMINI_",
    "GITHUB_",
    "GOOGLE_",
    "HF_",
    "MOONSHOT_",
    "OPENAI_",
)

# Each command exercises only tiny fixtures.  Missing paths are reported as
# merge-pending rather than silently replaced with weaker tests.
_SMOKE_TESTS: dict[int, tuple[str, ...]] = {
    1: ("tests/experiments/test_goal1_smoke.py",),
    2: ("tests/experiments/test_goal2_smoke.py",),
    3: ("tests/experiments/test_goal3_smoke.py",),
    4: ("tests/experiments/test_goal4_smoke.py",),
    5: ("tests/experiments/test_goal5_integration.py",),
    6: ("tests/experiments/test_goal6_smoke.py",),
    7: ("tests/experiments/test_goal7_smoke.py",),
    8: (
        "tests/experiments/test_goal8_smoke.py",
        "tests/experiments/test_goal8_simplify_smoke.py",
    ),
    9: ("tests/experiments/test_goal9_smoke.py",),
    10: (
        "tests/eml/test_compiler_trig_v2.py",
        "tests/egraph/test_rules_domain_v2.py",
        "tests/experiments/test_goal10_corpus_smoke.py",
        "tests/experiments/test_goal10_rerun_smoke.py",
        "tests/experiments/test_goal10_grids_smoke.py",
    ),
    11: (
        "tests/experiments/test_goal11_corpus_smoke.py",
        "tests/experiments/test_goal11_scaling_smoke.py",
        "tests/experiments/test_goal11_smoke.py",
        "tests/experiments/test_goal11_llm_smoke.py",
    ),
    12: (
        "tests/analysis/test_final_report.py",
        "tests/test_repro_scripts.py",
    ),
}


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(_canonical_json(payload))
    os.replace(temporary, path)


def _portable_path(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    """Return a stable path without publishing a host username or absolute directory."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _is_portable_basename(name: str) -> bool:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or ":" in name
        or name.endswith((" ", "."))
        or any(ord(character) < 32 for character in name)
    ):
        return False
    return name.split(".", maxsplit=1)[0].casefold() not in _WINDOWS_RESERVED_NAMES


def _validate_file_descriptor(
    item: Any,
    *,
    field: str,
    digest_key: str = "sha256",
) -> tuple[str, int, str]:
    if not isinstance(item, dict):
        raise ValueError(f"{field} entries must be objects")
    name = item.get("name")
    digest = item.get(digest_key)
    size = item.get("bytes")
    if not isinstance(name, str) or not _is_portable_basename(name):
        raise ValueError(f"invalid {field} filename: {name!r}")
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"invalid SHA-256 for {name}")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"invalid byte count for {name}")
    return name, size, digest


def load_artifact_manifest(path: Path = ARTIFACT_MANIFEST_PATH) -> dict[str, Any]:
    """Load and validate the release artifact-source registry."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "geml-repro-artifacts-v1":
        raise ValueError("unsupported artifact registry schema")
    public_url = payload.get("public_folder_url")
    if public_url is not None:
        parsed_url = urlparse(public_url) if isinstance(public_url, str) else None
        if parsed_url is None or parsed_url.scheme != "https" or parsed_url.hostname != "osf.io":
            raise ValueError("artifact registry URL must be an HTTPS OSF URL or null")

    files = payload.get("delivery_files")
    if not isinstance(files, list) or not files:
        raise ValueError("artifact registry must contain delivery_files")
    names: set[str] = set()
    for item in files:
        name, _, _ = _validate_file_descriptor(item, field="delivery-file")
        if name in names:
            raise ValueError(f"invalid or duplicate delivery filename: {name!r}")
        if "provider_id" in item:
            raise ValueError(f"anonymous artifact entries cannot contain provider IDs: {name}")
        names.add(name)

    archive_name = payload.get("archive_name")
    if (
        not isinstance(archive_name, str)
        or archive_name not in names
        or not archive_name.endswith(".tar")
    ):
        raise ValueError("archive_name must identify one delivery TAR")

    expected = payload.get("extracted_tree")
    if not isinstance(expected, dict):
        raise ValueError("artifact registry must contain extracted_tree")
    root_name = expected.get("root_name")
    if not isinstance(root_name, str) or not _is_portable_basename(root_name):
        raise ValueError("invalid extracted-tree root name")
    if (
        not isinstance(expected.get("files"), int)
        or isinstance(expected["files"], bool)
        or expected["files"] < 1
    ):
        raise ValueError("invalid extracted-tree file count")
    if (
        not isinstance(expected.get("bytes"), int)
        or isinstance(expected["bytes"], bool)
        or expected["bytes"] < 1
    ):
        raise ValueError("invalid extracted-tree byte count")
    for field in ("minimum_free_bytes_before_extraction", "minimum_free_inodes"):
        value = expected.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"invalid extracted-tree {field}")

    directories = payload.get("canonical_directories")
    if not isinstance(directories, list) or not directories:
        raise ValueError("artifact registry must contain canonical_directories")
    directory_names: set[str] = set()
    for item in directories:
        name, _, _ = _validate_file_descriptor(
            item,
            field="canonical-directory",
            digest_key="tree_sha256",
        )
        if name in directory_names:
            raise ValueError(f"duplicate canonical directory: {name}")
        count = item.get("files")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"invalid file count for canonical directory {name}")
        scheme = item.get("tree_hash_scheme")
        if scheme not in _TREE_HASH_SCHEMES:
            raise ValueError(f"invalid tree hash scheme for canonical directory {name}")
        directory_names.add(name)

    root_files = payload.get("root_files")
    if not isinstance(root_files, list) or not root_files:
        raise ValueError("artifact registry must contain root_files")
    root_names: set[str] = set()
    for item in root_files:
        name, size, digest = _validate_file_descriptor(item, field="root-file")
        if name in root_names or name in directory_names:
            raise ValueError(f"duplicate extracted-root entry: {name}")
        matching_delivery = next(
            (delivery for delivery in files if delivery["name"] == name),
            None,
        )
        if matching_delivery is not None and (
            matching_delivery["bytes"] != size or matching_delivery["sha256"] != digest
        ):
            raise ValueError(f"root-file and delivery checksums disagree for {name}")
        root_names.add(name)

    indexed_files = sum(item["files"] for item in directories) + len(root_files)
    indexed_bytes = sum(item["bytes"] for item in directories) + sum(
        item["bytes"] for item in root_files
    )
    if indexed_files != expected["files"] or indexed_bytes != expected["bytes"]:
        raise ValueError("extracted-tree totals do not match the indexed root entries")

    availability = payload.get("goal_availability")
    if not isinstance(availability, list) or not availability:
        raise ValueError("artifact registry must contain goal_availability")
    goal_rows: set[str] = set()
    for item in availability:
        if not isinstance(item, dict):
            raise ValueError("goal-availability entries must be objects")
        goals = item.get("goals")
        state = item.get("state")
        location = item.get("location")
        if not isinstance(goals, str) or not goals or goals in goal_rows:
            raise ValueError("invalid or duplicate goal-availability range")
        if state not in _GOAL_AVAILABILITY_STATES:
            raise ValueError(f"invalid availability state for Goals {goals}")
        if location is not None and (not isinstance(location, str) or not location):
            raise ValueError(f"invalid artifact location for Goals {goals}")
        if state == "complete" and location is None:
            raise ValueError(f"complete Goals {goals} artifacts require a location")
        goal_rows.add(goals)
    return payload


def smoke_plan(
    goal: int | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    python_executable: str = sys.executable,
) -> list[dict[str, Any]]:
    """Return smoke commands and explicit merge-readiness for Goals 1--12."""

    if goal is not None and goal not in _SMOKE_TESTS:
        raise ValueError("goal must be an integer from 1 through 12")
    goals = (goal,) if goal is not None else tuple(_SMOKE_TESTS)
    rows: list[dict[str, Any]] = []
    for goal_id in goals:
        tests = _SMOKE_TESTS[goal_id]
        missing = [item for item in tests if not (repo_root / item).is_file()]
        portable_executable = _portable_path(Path(python_executable), repo_root=repo_root)
        command = [
            portable_executable,
            "-m",
            "pytest",
            "-q",
            "--maxfail=1",
            *tests,
        ]
        rows.append(
            {
                "goal": goal_id,
                "state": "ready" if not missing else "merge_pending",
                "missing_targets": missing,
                "command": command,
                "timeout_seconds": SMOKE_TIMEOUT_SECONDS,
                "constraints": {
                    "network": False,
                    "gpu": False,
                    "api_keys": False,
                    "production_outputs": False,
                },
                "enforcement": {
                    "network": "cooperative_offline_policy_not_an_os_network_sandbox",
                    "gpu": "CUDA_VISIBLE_DEVICES is empty",
                    "api_keys": "credential-like environment variables are removed",
                    "production_outputs": "test-contract requirement",
                },
            }
        )
    return rows


def _smoke_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not _SENSITIVE_ENVIRONMENT_NAME.search(key)
        and not key.upper().startswith(_SENSITIVE_ENVIRONMENT_PREFIXES)
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "GEML_LLM_MODE": "mock",
            "GEML_REPRO_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def run_smoke(
    goal: int,
    *,
    repo_root: Path = REPO_ROOT,
    python_executable: str = sys.executable,
    timeout_seconds: int = SMOKE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one bounded smoke target and return a typed, JSON-compatible result."""

    if timeout_seconds < 1 or timeout_seconds > SMOKE_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be between 1 and {SMOKE_TIMEOUT_SECONDS}")
    plan = smoke_plan(goal, repo_root=repo_root, python_executable=python_executable)[0]
    if plan["state"] != "ready":
        return {
            **plan,
            "state": "merge_pending",
            "returncode": None,
            "elapsed_seconds": 0.0,
        }

    tests = _SMOKE_TESTS[goal]
    execution_command = [
        python_executable,
        "-m",
        "pytest",
        "-q",
        "--maxfail=1",
        *tests,
    ]

    start = time.monotonic()
    try:
        completed = subprocess.run(
            execution_command,
            cwd=repo_root,
            env=_smoke_environment(),
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            **plan,
            "state": "timeout",
            "returncode": 124,
            "elapsed_seconds": round(time.monotonic() - start, 6),
        }

    return {
        **plan,
        "state": "complete" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - start, 6),
    }


def verify_delivery(
    delivery_directory: Path,
    *,
    manifest_path: Path = ARTIFACT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Authenticate all downloaded handoff files by size and SHA-256."""

    manifest = load_artifact_manifest(manifest_path)
    entries: list[dict[str, Any]] = []
    for expected in manifest["delivery_files"]:
        path = delivery_directory / expected["name"]
        if not path.is_file() or _is_link_or_junction(path):
            entries.append(
                {
                    "name": expected["name"],
                    "state": "missing",
                    "expected_bytes": expected["bytes"],
                    "expected_sha256": expected["sha256"],
                }
            )
            continue
        actual_size = path.stat().st_size
        if actual_size != expected["bytes"]:
            entries.append(
                {
                    "name": expected["name"],
                    "state": "size_mismatch",
                    "expected_bytes": expected["bytes"],
                    "actual_bytes": actual_size,
                    "expected_sha256": expected["sha256"],
                }
            )
            continue
        actual_size, actual_digest_bytes = _stable_file_digest(path)
        actual_digest = actual_digest_bytes.hex()
        entries.append(
            {
                "name": expected["name"],
                "state": (
                    "complete" if actual_digest == expected["sha256"] else "checksum_mismatch"
                ),
                "expected_bytes": expected["bytes"],
                "actual_bytes": actual_size,
                "expected_sha256": expected["sha256"],
                "actual_sha256": actual_digest,
            }
        )
    return {
        "schema_version": "geml-delivery-verification-v1",
        "state": "complete" if all(row["state"] == "complete" for row in entries) else "failed",
        "delivery_directory": _portable_path(delivery_directory),
        "files": entries,
    }


def _portable_tar_member_name(name: str, *, expected_root: str) -> str:
    if not name or "\\" in name or "\0" in name:
        raise ValueError(f"unsafe TAR member name: {name!r}")
    path = PurePosixPath(name)
    parts = path.parts
    normalized = path.as_posix()
    if (
        path.is_absolute()
        or normalized != name
        or not parts
        or parts[0] != expected_root
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"unsafe TAR member path: {name!r}")
    for part in parts:
        if (
            any(ord(character) < 32 for character in part)
            or ":" in part
            or part.endswith((" ", "."))
            or part.split(".", maxsplit=1)[0].casefold() in _WINDOWS_RESERVED_NAMES
        ):
            raise ValueError(f"non-portable TAR member path: {name!r}")
    return normalized


def preflight_tar_archive(
    archive_path: Path,
    *,
    manifest_path: Path = ARTIFACT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Reject unsafe or contract-incompatible TAR members before extraction.

    This validates the member table and declared sizes. ``verify_delivery`` must
    authenticate the archive bytes immediately before this preflight.
    """

    manifest = load_artifact_manifest(manifest_path)
    expected = manifest["extracted_tree"]
    expected_archive_name = manifest["archive_name"]
    if archive_path.name != expected_archive_name:
        raise ValueError(f"archive must be named {expected_archive_name}")
    if _is_link_or_junction(archive_path) or not archive_path.is_file():
        raise ValueError("archive must be a regular, non-linked file")
    descriptor = next(
        item for item in manifest["delivery_files"] if item["name"] == expected_archive_name
    )
    if archive_path.stat().st_size != descriptor["bytes"]:
        return {
            "schema_version": "geml-tar-member-preflight-v1",
            "state": "size_mismatch",
            "archive": archive_path.name,
            "expected_bytes": descriptor["bytes"],
            "actual_bytes": archive_path.stat().st_size,
        }

    seen_names: set[str] = set()
    portable_names: set[str] = set()
    regular_files = 0
    directories = 0
    declared_bytes = 0
    root_directory_seen = False
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                normalized = _portable_tar_member_name(
                    member.name,
                    expected_root=expected["root_name"],
                )
                portable_key = unicodedata.normalize("NFC", normalized).casefold()
                if normalized in seen_names or portable_key in portable_names:
                    raise ValueError(f"duplicate or platform-colliding TAR member: {member.name!r}")
                seen_names.add(normalized)
                portable_names.add(portable_key)
                if member.isdir():
                    if member.mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
                        raise ValueError(f"unsafe TAR permission bits: {member.name!r}")
                    directories += 1
                    root_directory_seen |= normalized == expected["root_name"]
                    continue
                if (
                    not member.isreg()
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                    or member.issparse()
                ):
                    raise ValueError(
                        f"unsupported TAR member type for safe extraction: {member.name!r}"
                    )
                if member.mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
                    raise ValueError(f"unsafe TAR permission bits: {member.name!r}")
                regular_files += 1
                declared_bytes += member.size
                if regular_files > expected["files"] or declared_bytes > expected["bytes"]:
                    raise ValueError("TAR member totals exceed the authenticated tree contract")
    except tarfile.TarError as error:
        raise ValueError(f"invalid TAR archive: {error}") from error

    matches = (
        root_directory_seen
        and regular_files == expected["files"]
        and declared_bytes == expected["bytes"]
    )
    return {
        "schema_version": "geml-tar-member-preflight-v1",
        "state": "complete" if matches else "failed",
        "archive": archive_path.name,
        "content_authentication": "requires successful verify-delivery immediately beforehand",
        "expected_root": expected["root_name"],
        "root_directory_seen": root_directory_seen,
        "expected_files": expected["files"],
        "actual_files": regular_files,
        "expected_bytes": expected["bytes"],
        "actual_bytes": declared_bytes,
        "directories": directories,
        "members": regular_files + directories,
    }


def _iter_regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if _is_link_or_junction(path):
            raise ValueError(f"links are not permitted in authenticated trees: {path.name}")
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            yield from _iter_regular_files(path)
        elif stat.S_ISREG(mode):
            yield path
        else:
            raise ValueError(f"non-regular filesystem entry in authenticated tree: {path.name}")


def _stable_file_digest(path: Path) -> tuple[int, bytes]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or _is_link_or_junction(path):
        raise ValueError(f"expected a regular, non-linked file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise ValueError(f"file changed while it was authenticated: {path.name}")
    return after.st_size, digest.digest()


def _tree_digest(root: Path, *, scheme: str) -> tuple[int, int, str]:
    hasher = hashlib.sha256()
    file_count = 0
    byte_count = 0
    files: Iterable[Path] = _iter_regular_files(root)
    if scheme == "path-sha256-lines-v1":
        files = sorted(files, key=lambda path: path.relative_to(root).as_posix())
        first = True
        for path in files:
            size, digest = _stable_file_digest(path)
            if not first:
                hasher.update(b"\n")
            first = False
            hasher.update(path.relative_to(root).as_posix().encode())
            hasher.update(b"\0")
            hasher.update(digest.hex().encode())
            file_count += 1
            byte_count += size
    elif scheme == "geml-artifact-tree-v1":
        hasher.update(b"geml-artifact-tree-v1\n")
        for path in files:
            size, digest = _stable_file_digest(path)
            hasher.update(path.relative_to(root).as_posix().encode())
            hasher.update(b"\0")
            hasher.update(str(size).encode())
            hasher.update(b"\0")
            hasher.update(digest)
            hasher.update(b"\n")
            file_count += 1
            byte_count += size
    else:  # guarded by manifest validation
        raise ValueError(f"unsupported tree hash scheme: {scheme}")
    return file_count, byte_count, hasher.hexdigest()


def verify_extracted_tree(
    artifact_root: Path,
    *,
    manifest_path: Path = ARTIFACT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Cryptographically authenticate every indexed extracted-root file."""

    manifest = load_artifact_manifest(manifest_path)
    expected = manifest["extracted_tree"]
    if (
        not artifact_root.is_dir()
        or _is_link_or_junction(artifact_root)
        or artifact_root.name != expected["root_name"]
    ):
        return {
            "schema_version": "geml-extracted-tree-verification-v1",
            "state": "missing",
            "artifact_root": _portable_path(artifact_root),
            "expected_files": expected["files"],
            "expected_bytes": expected["bytes"],
        }

    expected_entries = {
        *(item["name"] for item in manifest["canonical_directories"]),
        *(item["name"] for item in manifest["root_files"]),
    }
    actual_entries = {path.name for path in artifact_root.iterdir()}
    unexpected_entries = sorted(actual_entries - expected_entries)

    directory_rows: list[dict[str, Any]] = []
    for descriptor in manifest["canonical_directories"]:
        path = artifact_root / descriptor["name"]
        if not path.is_dir() or _is_link_or_junction(path):
            directory_rows.append(
                {
                    **descriptor,
                    "state": "missing",
                }
            )
            continue
        try:
            file_count, byte_count, digest = _tree_digest(
                path,
                scheme=descriptor["tree_hash_scheme"],
            )
        except (OSError, ValueError) as error:
            directory_rows.append(
                {
                    **descriptor,
                    "state": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        matches = (
            file_count == descriptor["files"]
            and byte_count == descriptor["bytes"]
            and digest == descriptor["tree_sha256"]
        )
        directory_rows.append(
            {
                **descriptor,
                "state": "complete" if matches else "checksum_mismatch",
                "actual_files": file_count,
                "actual_bytes": byte_count,
                "actual_tree_sha256": digest,
            }
        )

    root_file_rows: list[dict[str, Any]] = []
    for descriptor in manifest["root_files"]:
        path = artifact_root / descriptor["name"]
        if not path.is_file() or _is_link_or_junction(path):
            root_file_rows.append({**descriptor, "state": "missing"})
            continue
        try:
            size, digest = _stable_file_digest(path)
        except (OSError, ValueError) as error:
            root_file_rows.append(
                {
                    **descriptor,
                    "state": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        digest_hex = digest.hex()
        root_file_rows.append(
            {
                **descriptor,
                "state": (
                    "complete"
                    if size == descriptor["bytes"] and digest_hex == descriptor["sha256"]
                    else "checksum_mismatch"
                ),
                "actual_bytes": size,
                "actual_sha256": digest_hex,
            }
        )

    file_count = sum(row.get("actual_files", 0) for row in directory_rows) + sum(
        1 for row in root_file_rows if "actual_bytes" in row
    )
    byte_count = sum(row.get("actual_bytes", 0) for row in directory_rows) + sum(
        row.get("actual_bytes", 0) for row in root_file_rows
    )
    complete = (
        not unexpected_entries
        and all(row["state"] == "complete" for row in directory_rows)
        and all(row["state"] == "complete" for row in root_file_rows)
        and file_count == expected["files"]
        and byte_count == expected["bytes"]
    )
    return {
        "schema_version": "geml-extracted-tree-verification-v1",
        "state": "complete" if complete else "failed",
        "artifact_root": _portable_path(artifact_root),
        "expected_files": expected["files"],
        "actual_files": file_count,
        "expected_bytes": expected["bytes"],
        "actual_bytes": byte_count,
        "unexpected_root_entries": unexpected_entries,
        "canonical_directories": directory_rows,
        "root_files": root_file_rows,
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[name.lower()] = distribution.version
    return dict(sorted(versions.items()))


def _git_commit(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and _GIT_COMMIT_PATTERN.fullmatch(value) else None


def collect_output_manifest(
    root: Path,
    output_path: Path,
    *,
    artifact_id: str,
    config_paths: Sequence[Path] = (),
    seeds: Sequence[int] = (),
    profile: str = "cpu",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Create a checksum-indexed artifact manifest without mutating source files."""

    artifact_id = artifact_id.strip()
    if not artifact_id:
        raise ValueError("artifact_id must be non-empty")
    if profile not in {"cpu", "2xh100", "4xh100-conditional"}:
        raise ValueError("unsupported compute profile")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds) or len(
        set(seeds)
    ) != len(seeds):
        raise ValueError("seeds must be unique integers")
    if _is_link_or_junction(root):
        raise ValueError("root must not be a link or junction")
    root = root.resolve()
    output_path = output_path.resolve()
    if not root.is_dir():
        raise ValueError("root must be a regular directory")
    if output_path == root or root in output_path.parents:
        raise ValueError("output manifest must be outside the indexed root")

    files: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in _iter_regular_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            size, digest = _stable_file_digest(path)
            files.append(
                {
                    "path": relative,
                    "bytes": size,
                    "sha256": digest.hex(),
                    "state": "complete",
                }
            )
        except (OSError, ValueError) as error:
            failures.append(
                {
                    "path": relative,
                    "state": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if not files and not failures:
        failures.append(
            {
                "path": ".",
                "state": "failed",
                "error": "indexed root contains no regular files",
            }
        )

    configs: list[dict[str, Any]] = []
    for path in config_paths:
        if _is_link_or_junction(path):
            configs.append(
                {
                    "path": _portable_path(path, repo_root=repo_root),
                    "state": "failed",
                    "error": "config path is a link or junction",
                }
            )
            continue
        resolved = path.resolve()
        if not resolved.is_file():
            configs.append(
                {
                    "path": _portable_path(path, repo_root=repo_root),
                    "state": "missing",
                }
            )
        else:
            try:
                size, digest = _stable_file_digest(resolved)
            except (OSError, ValueError) as error:
                configs.append(
                    {
                        "path": _portable_path(path, repo_root=repo_root),
                        "state": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            else:
                configs.append(
                    {
                        "path": _portable_path(path, repo_root=repo_root),
                        "state": "complete",
                        "bytes": size,
                        "sha256": digest.hex(),
                    }
                )

    payload: dict[str, Any] = {
        "schema_version": "geml-collected-artifact-v1",
        "artifact_id": artifact_id,
        "state": "complete"
        if not failures and all(c["state"] == "complete" for c in configs)
        else "failed",
        "root": _portable_path(root, repo_root=repo_root),
        "files": files,
        "failures": failures,
        "attempted_file_count": len(files) + sum(item["path"] != "." for item in failures),
        "file_count": len(files),
        "failure_count": len(failures),
        "byte_count": sum(item["bytes"] for item in files),
        "files_digest": hashlib.sha256(_canonical_json(files)).hexdigest(),
        "run_environment": {
            "git_commit": _git_commit(repo_root),
            "python": sys.version,
            "platform": platform.platform(),
            "compute_profile": profile,
            "seeds": list(seeds),
            "packages": _package_versions(),
        },
        "configs": configs,
    }
    _atomic_json(output_path, payload)
    return payload


def provider_preflight(
    provider: str,
    *,
    mode: str,
    model_id: str | None = None,
    attempts: int = 200,
    estimated_cost_usd: float | None = None,
    max_cost_usd: float | None = None,
    spend_approval: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate provider execution mode and spending guards without network calls."""

    if provider not in _PROVIDER_KEYS:
        raise ValueError(f"provider must be one of {', '.join(sorted(_PROVIDER_KEYS))}")
    if mode not in {"mock", "live"}:
        raise ValueError("mode must be mock or live")
    if attempts != 200:
        raise ValueError("the frozen panel requires exactly 200 attempts per configured model")

    base = {
        "schema_version": "geml-provider-preflight-v1",
        "provider": provider,
        "mode": mode,
        "attempts": attempts,
        "network_called": False,
    }
    if mode == "mock":
        return {
            **base,
            "state": "ready",
            "credentials_required": False,
            "cost_guard": "no_paid_calls",
            "model_availability": "mocked",
        }

    if model_id is None or not model_id.strip():
        raise ValueError("live mode requires an exact operator-selected model_id")
    normalized_model = model_id.strip().lower()
    if normalized_model in _FORBIDDEN_MODEL_ALIASES or normalized_model.endswith("-latest"):
        raise ValueError("live mode refuses mutable or silent model aliases")
    if (
        estimated_cost_usd is None
        or not math.isfinite(estimated_cost_usd)
        or estimated_cost_usd <= 0
    ):
        raise ValueError("live mode requires a positive estimated_cost_usd")
    if max_cost_usd is None or not math.isfinite(max_cost_usd) or max_cost_usd <= 0:
        raise ValueError("live mode requires a positive max_cost_usd")
    if estimated_cost_usd > max_cost_usd:
        raise ValueError("estimated cost exceeds the configured cost guard")
    if spend_approval != _SPEND_APPROVAL:
        raise ValueError("live mode requires explicit spend approval")
    env = os.environ if environment is None else environment
    key_name = _PROVIDER_KEYS[provider]
    if not env.get(key_name, "").strip():
        raise ValueError(f"live mode requires {key_name}")
    return {
        **base,
        "state": "local_checks_complete",
        "credentials_required": True,
        "credential_environment_variable": key_name,
        "credential_present": True,
        "model_id": model_id.strip(),
        "estimated_cost_usd": estimated_cost_usd,
        "max_cost_usd": max_cost_usd,
        "spend_approved": True,
        "model_availability": "must_be_verified_by_goal11_adapter_against_provider_api",
    }
