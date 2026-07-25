"""Authenticated Goal 5 integration runner and Goal 6 handoff freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from geml.analysis.goal5.summary import (
    ArtifactSchemaState,
    ClaimId,
    ClaimOutcome,
    EvidenceStatus,
    FrozenGoal6Export,
    Goal5IntegrationEvidence,
    GraphTrackName,
    MdlScope,
    MetricAvailability,
    RankerMethod,
    SourceArtifact,
    SplitName,
    render_compression_study_markdown,
    render_goal5_summary_markdown,
    render_goals_1_to_5_status_markdown,
    summarize,
)
from geml.export.schema import ProductionExportManifest
from geml.plots.goal5 import build_plot_data, render_plots

INTEGRATION_RUN_COMPLETE_SCHEMA_VERSION = "geml-goal5-integration-run-complete-v1"
_EXPECTED_OUTPUT_PATHS = frozenset(
    {
        "integration.evidence.json",
        "goal5.summary.json",
        "goal5.plot-data.json",
        "GOAL5_SUMMARY.md",
        "GOAL5_COMPRESSION_STUDY.md",
        "FINAL_GOALS_1_TO_5_STATUS.md",
        "plots/graph_size.png",
        "plots/mdl_cost.png",
        "plots/validity.png",
        "plots/runtime.png",
        "plots/memory.png",
        "plots/ranker_exact_best.png",
    }
)


class Goal5IntegrationError(ValueError):
    """Integration evidence or a generated artifact failed authentication."""


@dataclass(frozen=True, slots=True)
class IntegrationRunResult:
    """Locations and identity for one immutable integration run."""

    run_dir: Path
    completion_path: Path
    evidence_sha256: str
    implementation_sha256: str
    run_digest: str


class ProducerArtifactKind(StrEnum):
    GOAL1_CORPUS = "goal1_corpus"
    GOAL2_FINAL = "goal2_final"
    GOAL3_FINAL = "goal3_final"
    GOAL4_RUN = "goal4_run"
    GOAL4_ROWS = "goal4_rows"
    FREQUENT_MOTIFS = "issue_5_5_frequent_motifs"
    LEARNED_MOTIFS = "issue_5_6_learned_motifs"
    NEURAL_RANKER = "issue_5_7_neural_ranker"
    PRODUCTION_EXPORT = "issue_5_8_production_export"


@dataclass(frozen=True, slots=True)
class ProducerArtifactSpec:
    """One versioned producer artifact at its repository-standard path."""

    kind: ProducerArtifactKind
    path_pattern: str
    schema_version: str
    media_type: str
    atomic_completion: bool


@dataclass(frozen=True, slots=True)
class LoadedProducerArtifact:
    """Authenticated bytes and identity for one producer artifact."""

    spec: ProducerArtifactSpec
    path: Path
    sha256: str
    size_bytes: int
    record_count: int

    def as_source_artifact(self, repository_root: str | Path) -> SourceArtifact:
        """Convert authenticated bytes into the normalized evidence descriptor."""

        root = Path(repository_root).resolve()
        if not self.path.is_relative_to(root):
            raise Goal5IntegrationError("producer artifact is outside repository root")
        return SourceArtifact(
            name=self.spec.kind.value,
            path=self.path.relative_to(root).as_posix(),
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            media_type=self.spec.media_type,
            schema_state=ArtifactSchemaState.VERSIONED,
            schema_version=self.spec.schema_version,
            unversioned_reason=None,
        )


STANDARD_PRODUCER_ARTIFACTS = (
    ProducerArtifactSpec(
        ProducerArtifactKind.GOAL1_CORPUS,
        "outputs/final/goal1/final/run/manifests/corpus.manifest.json",
        "geml-corpus-v1",
        "application/json",
        True,
    ),
    ProducerArtifactSpec(
        ProducerArtifactKind.GOAL2_FINAL,
        "outputs/final/goal2/final/manifest.json",
        "geml-goal2-manifest-v1",
        "application/json",
        True,
    ),
    ProducerArtifactSpec(
        ProducerArtifactKind.GOAL3_FINAL,
        "outputs/final/goal3/final/manifest.json",
        "geml-goal3-manifest-v1",
        "application/json",
        True,
    ),
    ProducerArtifactSpec(
        ProducerArtifactKind.GOAL4_RUN,
        "outputs/final/goal4/final/final.run.json",
        "geml-goal4-run-v1",
        "application/json",
        True,
    ),
    ProducerArtifactSpec(
        ProducerArtifactKind.GOAL4_ROWS,
        "outputs/final/goal4/final/final.rows.jsonl",
        "geml-goal4-row-v2",
        "application/x-ndjson",
        False,
    ),
    ProducerArtifactSpec(
        ProducerArtifactKind.FREQUENT_MOTIFS,
        "outputs/final/goal5/motif_sweeps/final/*/run.complete.json",
        "geml-goal5-frequent-run-complete-v1",
        "application/json",
        True,
    ),
    ProducerArtifactSpec(
        ProducerArtifactKind.LEARNED_MOTIFS,
        "outputs/final/goal5/learned_motifs/*/run.complete.json",
        "geml-goal5-learned-run-complete-v1",
        "application/json",
        True,
    ),
    ProducerArtifactSpec(
        ProducerArtifactKind.NEURAL_RANKER,
        "outputs/final/goal5/neural_ranker/*/run.complete.json",
        "geml-goal5-neural-ranker-complete-v1",
        "application/json",
        True,
    ),
    ProducerArtifactSpec(
        ProducerArtifactKind.PRODUCTION_EXPORT,
        "outputs/final/goal5/export/run-*/run.complete.json",
        "geml-goal5-production-export-v1",
        "application/json",
        True,
    ),
)

_PRODUCER_SPEC_BY_KIND = {item.kind: item for item in STANDARD_PRODUCER_ARTIFACTS}
_FINAL_ATOMIC_GOAL5_PREREQUISITES = (
    ProducerArtifactKind.NEURAL_RANKER,
    ProducerArtifactKind.PRODUCTION_EXPORT,
)


@dataclass(frozen=True, slots=True)
class Goal4NontrivialCohorts:
    """Expression-ID cohorts derived only from versioned Goal 4 rewrite rows."""

    safe_expression_ids: tuple[str, ...]
    domain_expression_ids: tuple[str, ...]
    safe_sha256: str
    domain_sha256: str
    row_sha256: str
    split_by_expression: tuple[tuple[str, str], ...]
    safe_by_split: tuple[tuple[str, tuple[str, ...]], ...]
    domain_by_split: tuple[tuple[str, tuple[str, ...]], ...]
    safe_semantics: str = "branch_insensitive_finite_real"
    domain_semantics: str = "conditional_positive_real_formal_under_recorded_assumptions"


@dataclass(frozen=True, slots=True)
class CrossTrackCohortJoin:
    """An exact same-expression cohort proven present in every named track."""

    cohort_name: str
    expression_ids: tuple[str, ...]
    track_names: tuple[str, ...]
    expression_ids_sha256: str


def canonical_json_bytes(value: object) -> bytes:
    """Serialize deterministic UTF-8 JSON with one trailing line feed."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _validate_normalized_evidence_payload(
    payload: Mapping[str, object],
) -> Goal5IntegrationEvidence:
    """Validate the normalized object at its strict JSON serialization boundary."""

    return Goal5IntegrationEvidence.model_validate_json(canonical_json_bytes(payload))


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Goal5IntegrationError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _decode_json_object(data: bytes, *, label: str) -> dict[str, object]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Goal5IntegrationError(f"{label} is not UTF-8") from error
    if text.startswith("\ufeff"):
        raise Goal5IntegrationError(f"{label} must not contain a UTF-8 BOM")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=lambda token: _raise_nonfinite_json(token, label=label),
        )
    except (Goal5IntegrationError, json.JSONDecodeError) as error:
        if isinstance(error, Goal5IntegrationError):
            raise
        raise Goal5IntegrationError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise Goal5IntegrationError(f"{label} must contain one JSON object")
    return value


def _decode_json_lines(data: bytes, *, label: str) -> tuple[dict[str, object], ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Goal5IntegrationError(f"{label} is not UTF-8") from error
    if text.startswith("\ufeff"):
        raise Goal5IntegrationError(f"{label} must not contain a UTF-8 BOM")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise Goal5IntegrationError(f"{label} contains blank JSONL line {line_number}")
        records.append(
            _decode_json_object(
                line.encode("utf-8"),
                label=f"{label} line {line_number}",
            )
        )
    if not records:
        raise Goal5IntegrationError(f"{label} must contain at least one JSONL record")
    return tuple(records)


def _raise_nonfinite_json(token: str, *, label: str) -> None:
    raise Goal5IntegrationError(f"{label} contains non-finite JSON token {token!r}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(slots=True)
class _JsonlStream:
    """One-pass JSONL reader that authenticates the exact bytes it parses."""

    path: Path
    label: str
    expected_sha256: str | None = None
    expected_size: int | None = None
    expected_count: int | None = None
    schema_state: ArtifactSchemaState | None = None
    schema_version: str | None = None
    sha256: str = field(init=False, default="")
    size_bytes: int = field(init=False, default=0)
    record_count: int = field(init=False, default=0)
    _stream: BinaryIO | None = field(init=False, default=None, repr=False)
    _digest: object = field(init=False, repr=False)
    _finished: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._digest = hashlib.sha256()

    def __enter__(self) -> _JsonlStream:
        try:
            self._stream = self.path.open("rb")
        except OSError as error:
            raise Goal5IntegrationError(f"cannot read {self.label}: {self.path}") from error
        return self

    def __exit__(self, *_: object) -> None:
        if self._stream is not None:
            self._stream.close()

    def __iter__(self) -> Iterator[dict[str, object]]:
        if self._stream is None:
            raise RuntimeError("_JsonlStream must be used as a context manager")
        if self._finished:
            raise RuntimeError("_JsonlStream can be consumed only once")
        for line_number, raw_line in enumerate(self._stream, start=1):
            self._digest.update(raw_line)  # type: ignore[attr-defined]
            self.size_bytes += len(raw_line)
            if line_number == 1 and raw_line.startswith(b"\xef\xbb\xbf"):
                raise Goal5IntegrationError(f"{self.label} must not contain a UTF-8 BOM")
            line = raw_line.removesuffix(b"\n").removesuffix(b"\r")
            if not line:
                raise Goal5IntegrationError(f"{self.label} contains blank JSONL line {line_number}")
            payload = _decode_json_object(
                line,
                label=f"{self.label} line {line_number}",
            )
            if self.schema_state is not None:
                _validate_declared_schema(
                    (payload,),
                    schema_state=self.schema_state,
                    schema_version=self.schema_version,
                    label=f"{self.label} line {line_number}",
                )
            self.record_count = line_number
            yield payload
        if self.record_count == 0:
            raise Goal5IntegrationError(f"{self.label} must contain at least one JSONL record")
        self.sha256 = self._digest.hexdigest()  # type: ignore[attr-defined]
        self._finished = True
        self._validate_expected()

    def _validate_expected(self) -> None:
        if self.expected_size is not None and self.size_bytes != self.expected_size:
            raise Goal5IntegrationError(
                f"{self.label} has size {self.size_bytes}, expected {self.expected_size}"
            )
        if self.expected_sha256 is not None and self.sha256 != self.expected_sha256:
            raise Goal5IntegrationError(
                f"{self.label} SHA-256 is {self.sha256}, expected {self.expected_sha256}"
            )
        if self.expected_count is not None and self.record_count != self.expected_count:
            raise Goal5IntegrationError(
                f"{self.label} has {self.record_count} records, expected {self.expected_count}"
            )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise Goal5IntegrationError(f"cannot read artifact: {path}") from error
    return digest.hexdigest(), size


def _path_within(root: Path, relative_path: str) -> Path:
    candidate = (root / Path(*relative_path.split("/"))).resolve()
    if not candidate.is_relative_to(root):
        raise Goal5IntegrationError(f"artifact path escapes the repository root: {relative_path!r}")
    return candidate


def _is_json_lines_media_type(media_type: str) -> bool:
    return media_type in {"application/jsonl", "application/x-ndjson"} or media_type.endswith(
        "+jsonl"
    )


def _validate_declared_schema(
    records: Iterable[dict[str, object]],
    *,
    schema_state: ArtifactSchemaState,
    schema_version: str | None,
    label: str,
) -> int:
    record_count = 0
    for payload in records:
        record_count += 1
        observed = payload.get("schema_version")
        if schema_state is ArtifactSchemaState.VERSIONED:
            if observed != schema_version:
                raise Goal5IntegrationError(
                    f"{label} schema_version does not match its evidence descriptor"
                )
        elif "schema_version" in payload:
            raise Goal5IntegrationError(
                f"{label} is declared explicitly unversioned but contains schema_version"
            )
    return record_count


def _verify_source_artifacts(
    evidence: Goal5IntegrationEvidence,
    *,
    repository_root: Path,
    skip_paths: frozenset[str] = frozenset(),
) -> None:
    for artifact in evidence.source_artifacts:
        if artifact.path in skip_paths:
            continue
        path = _path_within(repository_root, artifact.path)
        if not path.is_file():
            raise Goal5IntegrationError(
                f"source artifact {artifact.name!r} does not exist: {artifact.path}"
            )
        label = f"source artifact {artifact.name!r}"
        if _is_json_lines_media_type(artifact.media_type):
            with _JsonlStream(
                path,
                label,
                expected_sha256=artifact.sha256,
                expected_size=artifact.size_bytes,
                schema_state=artifact.schema_state,
                schema_version=artifact.schema_version,
            ) as records:
                for _ in records:
                    pass
            continue
        try:
            data = path.read_bytes()
        except OSError as error:
            raise Goal5IntegrationError(f"cannot read {label}") from error
        if len(data) != artifact.size_bytes:
            raise Goal5IntegrationError(
                f"source artifact {artifact.name!r} has size {len(data)}, "
                f"expected {artifact.size_bytes}"
            )
        digest = _sha256_bytes(data)
        if digest != artifact.sha256:
            raise Goal5IntegrationError(
                f"source artifact {artifact.name!r} SHA-256 is {digest}, expected {artifact.sha256}"
            )
        if artifact.media_type == "application/json" or artifact.media_type.endswith("+json"):
            records = (_decode_json_object(data, label=label),)
        else:
            records = ()
        if artifact.schema_state is ArtifactSchemaState.VERSIONED and not records:
            raise Goal5IntegrationError(
                f"{label} uses a versioned schema with an unsupported media type"
            )
        _validate_declared_schema(
            records,
            schema_state=artifact.schema_state,
            schema_version=artifact.schema_version,
            label=label,
        )


def _producer_path(
    repository_root: Path,
    spec: ProducerArtifactSpec,
    *,
    override: str | Path | None,
) -> Path:
    if override is None:
        matches_list: list[Path] = []
        for path in repository_root.glob(spec.path_pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(repository_root):
                raise Goal5IntegrationError(
                    f"{spec.kind.value} standard artifact resolves outside repository root"
                )
            matches_list.append(resolved)
        matches = tuple(sorted(matches_list))
        if len(matches) != 1:
            raise Goal5IntegrationError(
                f"{spec.kind.value} requires exactly one standard artifact matching "
                f"{spec.path_pattern!r}; found {len(matches)}"
            )
        return matches[0]
    candidate = Path(override)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(repository_root):
        raise Goal5IntegrationError(f"{spec.kind.value} override escapes repository root")
    relative = candidate.relative_to(repository_root).as_posix()
    if not PurePosixPath(relative).match(spec.path_pattern):
        raise Goal5IntegrationError(
            f"{spec.kind.value} override is outside its standard path pattern"
        )
    if not candidate.is_file():
        raise Goal5IntegrationError(f"{spec.kind.value} artifact does not exist")
    return candidate


def read_standard_producer_artifact(
    repository_root: str | Path,
    kind: ProducerArtifactKind,
    *,
    path_override: str | Path | None = None,
) -> LoadedProducerArtifact:
    """Authenticate one versioned producer artifact at its frozen standard path."""

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise Goal5IntegrationError(f"repository root does not exist: {root}")
    spec = _PRODUCER_SPEC_BY_KIND[kind]
    path = _producer_path(root, spec, override=path_override)
    label = f"{kind.value} producer artifact"
    if _is_json_lines_media_type(spec.media_type):
        with _JsonlStream(
            path,
            label,
            schema_state=ArtifactSchemaState.VERSIONED,
            schema_version=spec.schema_version,
        ) as records:
            for _ in records:
                pass
        digest = records.sha256
        size = records.size_bytes
        count = records.record_count
    else:
        try:
            data = path.read_bytes()
        except OSError as error:
            raise Goal5IntegrationError(f"cannot read {kind.value} producer artifact") from error
        records = (_decode_json_object(data, label=label),)
        count = _validate_declared_schema(
            records,
            schema_state=ArtifactSchemaState.VERSIONED,
            schema_version=spec.schema_version,
            label=label,
        )
        digest = _sha256_bytes(data)
        size = len(data)
    return LoadedProducerArtifact(
        spec=spec,
        path=path,
        sha256=digest,
        size_bytes=size,
        record_count=count,
    )


def require_final_goal5_completions(
    repository_root: str | Path,
    *,
    path_overrides: Mapping[ProducerArtifactKind, str | Path] | None = None,
) -> tuple[LoadedProducerArtifact, ...]:
    """Require atomic issue 5-7 and 5-8 completions before final reporting."""

    overrides = path_overrides or {}
    return tuple(
        read_standard_producer_artifact(
            repository_root,
            kind,
            path_override=overrides.get(kind),
        )
        for kind in _FINAL_ATOMIC_GOAL5_PREREQUISITES
    )


def load_standard_production_evidence(
    repository_root: str | Path,
    *,
    path_overrides: Mapping[ProducerArtifactKind, str | Path] | None = None,
) -> tuple[LoadedProducerArtifact, ...]:
    """Read all versioned Goal 1-4 and issue 5-5 through 5-8 producer interfaces."""

    overrides = path_overrides or {}
    return tuple(
        read_standard_producer_artifact(
            repository_root,
            spec.kind,
            path_override=overrides.get(spec.kind),
        )
        for spec in STANDARD_PRODUCER_ARTIFACTS
    )


def _matching_producer_sources(
    evidence: Goal5IntegrationEvidence,
    kind: ProducerArtifactKind,
) -> list[SourceArtifact]:
    spec = _PRODUCER_SPEC_BY_KIND[kind]
    return [
        artifact
        for artifact in evidence.source_artifacts
        if PurePosixPath(artifact.path).match(spec.path_pattern)
        and artifact.schema_state is ArtifactSchemaState.VERSIONED
        and artifact.schema_version == spec.schema_version
        and artifact.media_type == spec.media_type
    ]


def _require_atomic_prerequisite_sources(evidence: Goal5IntegrationEvidence) -> None:
    for kind in _FINAL_ATOMIC_GOAL5_PREREQUISITES:
        matches = _matching_producer_sources(evidence, kind)
        if len(matches) != 1:
            raise Goal5IntegrationError(
                f"final integration requires exactly one authenticated atomic "
                f"{kind.value} completion"
            )


def _require_standard_producer_sources(evidence: Goal5IntegrationEvidence) -> None:
    for kind in ProducerArtifactKind:
        spec = _PRODUCER_SPEC_BY_KIND[kind]
        matches = _matching_producer_sources(evidence, kind)
        if len(matches) != 1:
            raise Goal5IntegrationError(
                "final integration requires exactly one authenticated versioned "
                f"{kind.value} source at {spec.path_pattern!r}"
            )


def _expression_id(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Goal5IntegrationError(f"{label} must be a lowercase SHA-256 expression ID")
    return value


def _cohort_digest(expression_ids: Iterable[str]) -> str:
    digest = hashlib.sha256(b"geml-goal5-expression-cohort-v1\0")
    for expression_id in expression_ids:
        digest.update(expression_id.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _Goal4RowObservation:
    expression_id: str
    rewrite_mode: str
    split: str
    rewrites_applied: int | None
    stage_status: str | None
    eml_dag_cost_after: int | None
    validation_status: str | None
    failure_stage: str | None
    wall_seconds: float | None


@dataclass(frozen=True, slots=True)
class _Goal4RowScan:
    cohorts: Goal4NontrivialCohorts
    rows: tuple[_Goal4RowObservation, ...]
    record_count: int


def _optional_nonnegative_int(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Goal5IntegrationError(f"{label} must be a nonnegative integer or null")
    return value


def _optional_nonnegative_float(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise Goal5IntegrationError(f"{label} must be a nonnegative finite float or null")
    return value


def _scan_goal4_rows(
    rows_path: str | Path,
    *,
    expected_source: SourceArtifact | None = None,
    run_manifest: Mapping[str, object] | None = None,
) -> _Goal4RowScan:
    path = Path(rows_path).resolve()
    expected_modes = {"safe_real", "positive_real_formal"}
    expected_splits = {split.value for split in SplitName}
    by_key: dict[tuple[str, str], int | None] = {}
    modes_by_expression: dict[str, set[str]] = {}
    split_by_expression: dict[str, str] = {}
    observations: list[_Goal4RowObservation] = []

    expected_run_id: str | None = None
    expected_config_sha256: str | None = None
    expected_selected_count: int | None = None
    if run_manifest is not None:
        if run_manifest.get("row_schema_version") != "geml-goal4-row-v2":
            raise Goal5IntegrationError("Goal 4 run manifest has incompatible row schema")
        expected_run_id = _required_sha256(
            run_manifest.get("run_id"),
            field="Goal 4 run_id",
        )
        expected_config_sha256 = _required_sha256(
            run_manifest.get("config_sha256"),
            field="Goal 4 config_sha256",
        )
        selected_count = run_manifest.get("selected_expression_count")
        if (
            isinstance(selected_count, bool)
            or not isinstance(selected_count, int)
            or selected_count < 0
        ):
            raise Goal5IntegrationError(
                "Goal 4 selected_expression_count must be a nonnegative integer"
            )
        expected_selected_count = selected_count
        if run_manifest.get("modes") != ["safe_real", "positive_real_formal"]:
            raise Goal5IntegrationError("Goal 4 run manifest must contain both frozen modes")

    with _JsonlStream(
        path,
        "Goal 4 rows",
        expected_sha256=None if expected_source is None else expected_source.sha256,
        expected_size=None if expected_source is None else expected_source.size_bytes,
        schema_state=ArtifactSchemaState.VERSIONED,
        schema_version="geml-goal4-row-v2",
    ) as stream:
        for row_number, row in enumerate(stream, start=1):
            expression_id = _expression_id(
                row.get("expression_id"),
                label=f"Goal 4 row {row_number} expression_id",
            )
            mode = row.get("rewrite_mode")
            if mode not in expected_modes:
                raise Goal5IntegrationError(
                    f"Goal 4 row {row_number} has unsupported rewrite_mode {mode!r}"
                )
            split = row.get("split")
            if split not in expected_splits:
                raise Goal5IntegrationError(
                    f"Goal 4 row {row_number} has unsupported split {split!r}"
                )
            if expected_run_id is not None and (
                row.get("run_id") != expected_run_id
                or row.get("config_sha256") != expected_config_sha256
            ):
                raise Goal5IntegrationError(
                    f"Goal 4 row {row_number} identity does not match the run manifest"
                )
            prior_split = split_by_expression.setdefault(expression_id, split)
            if prior_split != split:
                raise Goal5IntegrationError(
                    f"Goal 4 expression {expression_id} appears in multiple splits"
                )
            rewrites = _optional_nonnegative_int(
                row.get("rewrites_applied"),
                label=f"Goal 4 row {row_number} rewrites_applied",
            )
            key = (expression_id, mode)
            if key in by_key:
                raise Goal5IntegrationError(f"Goal 4 rows contain duplicate key {key!r}")
            by_key[key] = rewrites
            modes_by_expression.setdefault(expression_id, set()).add(mode)

            resources = row.get("resources")
            wall_seconds = None
            if resources is not None:
                if not isinstance(resources, dict):
                    raise Goal5IntegrationError(
                        f"Goal 4 row {row_number} resources must be an object or null"
                    )
                wall_seconds = _optional_nonnegative_float(
                    resources.get("wall_seconds"),
                    label=f"Goal 4 row {row_number} resources.wall_seconds",
                )
            stage_status = row.get("stage_status")
            validation_status = row.get("validation_status")
            failure_stage = row.get("failure_stage")
            for name, value in (
                ("stage_status", stage_status),
                ("validation_status", validation_status),
                ("failure_stage", failure_stage),
            ):
                if value is not None and (not isinstance(value, str) or not value):
                    raise Goal5IntegrationError(
                        f"Goal 4 row {row_number} {name} must be nonblank or null"
                    )
            observations.append(
                _Goal4RowObservation(
                    expression_id=expression_id,
                    rewrite_mode=mode,
                    split=split,
                    rewrites_applied=rewrites,
                    stage_status=stage_status,
                    eml_dag_cost_after=_optional_nonnegative_int(
                        row.get("eml_dag_cost_after"),
                        label=f"Goal 4 row {row_number} eml_dag_cost_after",
                    ),
                    validation_status=validation_status,
                    failure_stage=failure_stage,
                    wall_seconds=wall_seconds,
                )
            )

    incomplete = sorted(
        expression_id
        for expression_id, modes in modes_by_expression.items()
        if modes != expected_modes
    )
    if incomplete:
        raise Goal5IntegrationError(
            "Goal 4 cohort derivation requires both rewrite modes for every expression; "
            f"missing pair for {incomplete[0]}"
        )
    if expected_selected_count is not None:
        if stream.record_count != expected_selected_count * len(expected_modes):
            raise Goal5IntegrationError("Goal 4 row count does not match the run manifest")
        if len(modes_by_expression) != expected_selected_count:
            raise Goal5IntegrationError(
                "Goal 4 unique expression count does not match the run manifest"
            )

    safe_ids = tuple(
        sorted(
            expression_id
            for (expression_id, mode), count in by_key.items()
            if mode == "safe_real" and count is not None and count > 0
        )
    )
    domain_ids = tuple(
        sorted(
            expression_id
            for (expression_id, mode), count in by_key.items()
            if mode == "positive_real_formal" and count is not None and count > 0
        )
    )
    safe_set = set(safe_ids)
    domain_set = set(domain_ids)
    safe_by_split = tuple(
        (
            split.value,
            tuple(
                sorted(
                    expression_id
                    for expression_id, observed_split in split_by_expression.items()
                    if observed_split == split.value and expression_id in safe_set
                )
            ),
        )
        for split in SplitName
    )
    domain_by_split = tuple(
        (
            split.value,
            tuple(
                sorted(
                    expression_id
                    for expression_id, observed_split in split_by_expression.items()
                    if observed_split == split.value and expression_id in domain_set
                )
            ),
        )
        for split in SplitName
    )
    cohorts = Goal4NontrivialCohorts(
        safe_expression_ids=safe_ids,
        domain_expression_ids=domain_ids,
        safe_sha256=_cohort_digest(safe_ids),
        domain_sha256=_cohort_digest(domain_ids),
        row_sha256=stream.sha256,
        split_by_expression=tuple(sorted(split_by_expression.items())),
        safe_by_split=safe_by_split,
        domain_by_split=domain_by_split,
    )
    return _Goal4RowScan(
        cohorts=cohorts,
        rows=tuple(observations),
        record_count=stream.record_count,
    )


def derive_goal4_nontrivial_cohorts(
    rows_path: str | Path,
) -> Goal4NontrivialCohorts:
    """Derive safe/domain nontrivial IDs from exact versioned Goal 4 rows."""

    return _scan_goal4_rows(rows_path).cohorts


def join_cohort_across_tracks(
    cohort_name: str,
    expression_ids: Iterable[str],
    track_expression_ids: Mapping[str, Iterable[str]],
) -> CrossTrackCohortJoin:
    """Prove an exact same-expression cohort is present once in every track."""

    if not isinstance(cohort_name, str) or not cohort_name.strip():
        raise Goal5IntegrationError("cohort_name must be nonblank")
    cohort_values = tuple(expression_ids)
    cohort = tuple(sorted(cohort_values))
    if len(set(cohort)) != len(cohort):
        raise Goal5IntegrationError("cohort expression IDs must be unique")
    for index, expression_id in enumerate(cohort):
        _expression_id(expression_id, label=f"cohort expression_id {index}")
    if not track_expression_ids:
        raise Goal5IntegrationError("cross-track joins require at least one track")
    if any(not isinstance(name, str) for name in track_expression_ids):
        raise Goal5IntegrationError("track names must be strings")
    track_names = tuple(sorted(track_expression_ids))
    for track_name in track_names:
        if not track_name.strip():
            raise Goal5IntegrationError("track names must be nonblank")
        values = tuple(track_expression_ids[track_name])
        if len(set(values)) != len(values):
            raise Goal5IntegrationError(f"track {track_name!r} contains duplicate expression IDs")
        for index, expression_id in enumerate(values):
            _expression_id(
                expression_id,
                label=f"track {track_name!r} expression_id {index}",
            )
        available = set(values)
        missing = tuple(item for item in cohort if item not in available)
        if missing:
            raise Goal5IntegrationError(
                f"track {track_name!r} is missing {len(missing)} exact cohort IDs; "
                f"first missing ID is {missing[0]}"
            )
    return CrossTrackCohortJoin(
        cohort_name=cohort_name,
        expression_ids=cohort,
        track_names=track_names,
        expression_ids_sha256=_cohort_digest(cohort),
    )


def _verify_goal4_cohort_bindings(
    evidence: Goal5IntegrationEvidence,
    *,
    repository_root: Path,
) -> None:
    run_source = _matching_producer_sources(evidence, ProducerArtifactKind.GOAL4_RUN)[0]
    source = _matching_producer_sources(evidence, ProducerArtifactKind.GOAL4_ROWS)[0]
    run_path = _path_within(repository_root, run_source.path)
    rows_path = _path_within(repository_root, source.path)
    run = _decode_json_object(run_path.read_bytes(), label="Goal 4 run manifest")
    cohorts = _scan_goal4_rows(
        rows_path,
        expected_source=source,
        run_manifest=run,
    ).cohorts
    subsets = {item.name: item for item in evidence.subset_definitions}
    expected = {
        "safe_nontrivial": cohorts.safe_sha256,
        "domain_nontrivial": cohorts.domain_sha256,
    }
    for name, digest in expected.items():
        subset = subsets[name]
        if subset.expression_ids_sha256 != digest:
            raise Goal5IntegrationError(
                f"{name} expression-ID digest does not match versioned Goal 4 rows"
            )
        if source.name not in subset.source_artifacts:
            raise Goal5IntegrationError(f"{name} must cite the authenticated Goal 4 row artifact")
    split_cohorts = {
        ("safe_nontrivial", split): expression_ids
        for split, expression_ids in cohorts.safe_by_split
    }
    split_cohorts.update(
        {
            ("domain_nontrivial", split): expression_ids
            for split, expression_ids in cohorts.domain_by_split
        }
    )
    for join in evidence.cohort_joins:
        expression_ids = split_cohorts[(join.subset, join.split.value)]
        if join.expression_count != len(
            expression_ids
        ) or join.expression_ids_sha256 != _cohort_digest(expression_ids):
            raise Goal5IntegrationError(
                f"{join.split.value}/{join.subset} join does not match Goal 4 IDs"
            )
        if source.name not in join.source_artifacts:
            raise Goal5IntegrationError(
                f"{join.split.value}/{join.subset} join must cite Goal 4 rows"
            )


def _repository_source(
    *,
    name: str,
    path: Path,
    repository_root: Path,
    sha256: str,
    size_bytes: int,
    media_type: str,
    schema_version: str | None,
    unversioned_reason: str | None = None,
) -> SourceArtifact:
    resolved = path.resolve()
    if not resolved.is_relative_to(repository_root):
        raise Goal5IntegrationError(f"source artifact {name!r} is outside repository root")
    versioned = schema_version is not None
    return SourceArtifact(
        name=name,
        path=resolved.relative_to(repository_root).as_posix(),
        sha256=sha256,
        size_bytes=size_bytes,
        media_type=media_type,
        schema_state=(
            ArtifactSchemaState.VERSIONED
            if versioned
            else ArtifactSchemaState.EXPLICITLY_UNVERSIONED
        ),
        schema_version=schema_version,
        unversioned_reason=None if versioned else unversioned_reason,
    )


def _artifact_reference_path(
    run_dir: Path,
    reference: object,
    *,
    label: str,
) -> tuple[Path, str, int | None]:
    if not isinstance(reference, dict):
        raise Goal5IntegrationError(f"{label} reference must be an object")
    relative = reference.get("path")
    sha256 = reference.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise Goal5IntegrationError(f"{label} reference path must be nonblank")
    expected_sha256 = _required_sha256(sha256, field=f"{label}.sha256")
    path = _path_within(run_dir, relative)
    if not path.is_file():
        raise Goal5IntegrationError(f"{label} artifact does not exist: {relative}")
    expected_size = reference.get("byte_count")
    if expected_size is not None and (
        isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0
    ):
        raise Goal5IntegrationError(f"{label}.byte_count must be nonnegative")
    return path, expected_sha256, expected_size


def _load_referenced_json(
    *,
    repository_root: Path,
    run_dir: Path,
    reference: object,
    source_name: str,
    schema_version: str,
    label: str,
) -> tuple[dict[str, object], SourceArtifact]:
    path, expected_sha256, expected_size = _artifact_reference_path(
        run_dir,
        reference,
        label=label,
    )
    try:
        data = path.read_bytes()
    except OSError as error:
        raise Goal5IntegrationError(f"cannot read {label}") from error
    if expected_size is not None and len(data) != expected_size:
        raise Goal5IntegrationError(f"{label} has size {len(data)}, expected {expected_size}")
    digest = _sha256_bytes(data)
    if digest != expected_sha256:
        raise Goal5IntegrationError(f"{label} SHA-256 is {digest}, expected {expected_sha256}")
    payload = _decode_json_object(data, label=label)
    if payload.get("schema_version") != schema_version:
        raise Goal5IntegrationError(f"{label} has incompatible schema_version")
    return (
        payload,
        _repository_source(
            name=source_name,
            path=path,
            repository_root=repository_root,
            sha256=digest,
            size_bytes=len(data),
            media_type="application/json",
            schema_version=schema_version,
        ),
    )


@dataclass(slots=True)
class _GraphCounts:
    denominator_count: int = 0
    success_count: int = 0
    failure_counts: Counter[str] = field(default_factory=Counter)
    node_observation_count: int = 0
    node_total: int = 0
    edge_observation_count: int = 0
    edge_total: int = 0
    structural_attempted_count: int = 0
    structural_passed_count: int = 0
    reconstruction_attempted_count: int = 0
    reconstruction_passed_count: int = 0
    expansion_attempted_count: int = 0
    expansion_passed_count: int = 0
    runtime_observation_count: int = 0
    runtime_total_seconds: float = 0.0

    @property
    def failure_count(self) -> int:
        return sum(self.failure_counts.values())


def _available_integer(
    *,
    denominator_count: int,
    observation_count: int,
    total: int,
    unit: str,
    sources: Iterable[str],
) -> dict[str, object]:
    return {
        "availability": MetricAvailability.AVAILABLE.value,
        "denominator_count": denominator_count,
        "observation_count": observation_count,
        "missing_count": denominator_count - observation_count,
        "total": total,
        "unit": unit,
        "unavailable_reason": None,
        "source_artifacts": sorted(set(sources)),
    }


def _unavailable_integer(
    *,
    denominator_count: int,
    unit: str,
    reason: str,
    sources: Iterable[str],
) -> dict[str, object]:
    return {
        "availability": MetricAvailability.UNAVAILABLE.value,
        "denominator_count": denominator_count,
        "observation_count": 0,
        "missing_count": denominator_count,
        "total": None,
        "unit": unit,
        "unavailable_reason": reason,
        "source_artifacts": sorted(set(sources)),
    }


def _available_float(
    *,
    denominator_count: int,
    observation_count: int,
    total: float,
    sources: Iterable[str],
) -> dict[str, object]:
    return {
        "availability": MetricAvailability.AVAILABLE.value,
        "denominator_count": denominator_count,
        "observation_count": observation_count,
        "missing_count": denominator_count - observation_count,
        "total": total,
        "unit": "seconds",
        "unavailable_reason": None,
        "source_artifacts": sorted(set(sources)),
    }


def _unavailable_float(
    *,
    denominator_count: int,
    reason: str,
    sources: Iterable[str],
) -> dict[str, object]:
    return {
        "availability": MetricAvailability.UNAVAILABLE.value,
        "denominator_count": denominator_count,
        "observation_count": 0,
        "missing_count": denominator_count,
        "total": None,
        "unit": "seconds",
        "unavailable_reason": reason,
        "source_artifacts": sorted(set(sources)),
    }


def _unavailable_memory(
    *,
    denominator_count: int,
    reason: str,
    sources: Iterable[str],
) -> dict[str, object]:
    return {
        "availability": MetricAvailability.UNAVAILABLE.value,
        "denominator_count": denominator_count,
        "observation_count": 0,
        "missing_count": denominator_count,
        "peak": None,
        "unit": "bytes",
        "unavailable_reason": reason,
        "source_artifacts": sorted(set(sources)),
    }


def _audit_payload(
    *,
    denominator_count: int,
    attempted_count: int,
    passed_count: int,
    availability: MetricAvailability,
    reason: str | None,
    sources: Iterable[str],
) -> dict[str, object]:
    failed_count = attempted_count - passed_count
    return {
        "availability": availability.value,
        "denominator_count": denominator_count,
        "attempted_count": attempted_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "unobserved_count": denominator_count - attempted_count,
        "unavailable_reason": reason,
        "source_artifacts": sorted(set(sources)),
    }


def _not_applicable_audit(
    *,
    denominator_count: int,
    reason: str,
    sources: Iterable[str],
) -> dict[str, object]:
    return _audit_payload(
        denominator_count=denominator_count,
        attempted_count=0,
        passed_count=0,
        availability=MetricAvailability.NOT_APPLICABLE,
        reason=reason,
        sources=sources,
    )


def _unavailable_mdl(
    *,
    denominator_count: int,
    scope: MdlScope,
    reason: str,
    sources: Iterable[str],
) -> dict[str, object]:
    return {
        "availability": MetricAvailability.UNAVAILABLE.value,
        "denominator_count": denominator_count,
        "observation_count": 0,
        "missing_count": denominator_count,
        "total_bits": None,
        "codec": None,
        "scope": scope.value,
        "unavailable_reason": reason,
        "source_artifacts": sorted(set(sources)),
    }


def _available_mdl(
    *,
    denominator_count: int,
    total_bits: int,
    scope: MdlScope,
    sources: Iterable[str],
) -> dict[str, object]:
    return {
        "availability": MetricAvailability.AVAILABLE.value,
        "denominator_count": denominator_count,
        "observation_count": denominator_count,
        "missing_count": 0,
        "total_bits": total_bits,
        "codec": "geml-motif-mdl-v1",
        "scope": scope.value,
        "unavailable_reason": None,
        "source_artifacts": sorted(set(sources)),
    }


def _subset_memberships(
    expression_id: str,
    *,
    safe_ids: frozenset[str],
    domain_ids: frozenset[str],
) -> tuple[str, ...]:
    subsets = ["all"]
    if expression_id in safe_ids:
        subsets.append("safe_nontrivial")
    if expression_id in domain_ids:
        subsets.append("domain_nontrivial")
    return tuple(subsets)


def _aggregate_goal4_graph_rows(
    scan: _Goal4RowScan,
) -> dict[tuple[str, str, str], _GraphCounts]:
    safe_ids = frozenset(scan.cohorts.safe_expression_ids)
    domain_ids = frozenset(scan.cohorts.domain_expression_ids)
    counts: dict[tuple[str, str, str], _GraphCounts] = defaultdict(_GraphCounts)
    success_statuses = {"optimized", "unchanged"}
    for row in scan.rows:
        for subset in _subset_memberships(
            row.expression_id,
            safe_ids=safe_ids,
            domain_ids=domain_ids,
        ):
            aggregate = counts[(row.rewrite_mode, row.split, subset)]
            aggregate.denominator_count += 1
            if row.wall_seconds is not None:
                aggregate.runtime_observation_count += 1
                aggregate.runtime_total_seconds += row.wall_seconds
            successful = row.stage_status in success_statuses
            if successful:
                if row.eml_dag_cost_after is None or row.validation_status != "valid":
                    raise Goal5IntegrationError(
                        "successful Goal 4 row lacks validated official EML-DAG cost"
                    )
                aggregate.success_count += 1
                aggregate.node_observation_count += 1
                aggregate.node_total += row.eml_dag_cost_after
                aggregate.structural_attempted_count += 1
                aggregate.structural_passed_count += 1
                continue
            category = row.stage_status or row.failure_stage or "unclassified_failure"
            aggregate.failure_counts[category] += 1
    for aggregate in counts.values():
        if aggregate.success_count + aggregate.failure_count != aggregate.denominator_count:
            raise Goal5IntegrationError("Goal 4 graph accounting lost an attempted row")
    return counts


def _graph_slice_payload(
    *,
    split: str,
    subset: str,
    counts: _GraphCounts,
    source_names: Iterable[str],
    edge_available: bool,
    mdl: dict[str, object],
    reconstruction_required: bool,
    expansion_required: bool,
    runtime_available: bool,
    unavailable_runtime_reason: str,
    unavailable_memory_reason: str,
) -> dict[str, object]:
    sources = tuple(sorted(set(source_names)))
    if edge_available:
        edge_count = _available_integer(
            denominator_count=counts.success_count,
            observation_count=counts.edge_observation_count,
            total=counts.edge_total,
            unit="edges",
            sources=sources,
        )
    else:
        edge_count = _unavailable_integer(
            denominator_count=counts.success_count,
            unit="edges",
            reason=(
                "Goal 4 publishes only post-rewrite official EML-DAG node cost; "
                "selected_signature is non-reversible and cannot prove edge count"
            ),
            sources=sources,
        )
    structural = _audit_payload(
        denominator_count=counts.success_count,
        attempted_count=counts.structural_attempted_count,
        passed_count=counts.structural_passed_count,
        availability=MetricAvailability.AVAILABLE,
        reason=None,
        sources=sources,
    )
    reconstruction = (
        _audit_payload(
            denominator_count=counts.success_count,
            attempted_count=counts.reconstruction_attempted_count,
            passed_count=counts.reconstruction_passed_count,
            availability=MetricAvailability.AVAILABLE,
            reason=None,
            sources=sources,
        )
        if reconstruction_required
        else _not_applicable_audit(
            denominator_count=counts.success_count,
            reason="this representation has no motif reconstruction operation",
            sources=sources,
        )
    )
    expansion = (
        _audit_payload(
            denominator_count=counts.success_count,
            attempted_count=counts.expansion_attempted_count,
            passed_count=counts.expansion_passed_count,
            availability=MetricAvailability.AVAILABLE,
            reason=None,
            sources=sources,
        )
        if expansion_required
        else _not_applicable_audit(
            denominator_count=counts.success_count,
            reason="this representation has no macro expansion operation",
            sources=sources,
        )
    )
    runtime = (
        _available_float(
            denominator_count=counts.denominator_count,
            observation_count=counts.runtime_observation_count,
            total=counts.runtime_total_seconds,
            sources=sources,
        )
        if runtime_available
        else _unavailable_float(
            denominator_count=counts.denominator_count,
            reason=unavailable_runtime_reason,
            sources=sources,
        )
    )
    return {
        "split": split,
        "subset": subset,
        "denominator_count": counts.denominator_count,
        "success_count": counts.success_count,
        "failure_count": counts.failure_count,
        "failure_counts": dict(sorted(counts.failure_counts.items())),
        "node_count": _available_integer(
            denominator_count=counts.success_count,
            observation_count=counts.node_observation_count,
            total=counts.node_total,
            unit="nodes",
            sources=sources,
        ),
        "edge_count": edge_count,
        "mdl_cost": mdl,
        "structural_validation": structural,
        "reconstruction": reconstruction,
        "expansion": expansion,
        "runtime": runtime,
        "memory": _unavailable_memory(
            denominator_count=counts.denominator_count,
            reason=unavailable_memory_reason,
            sources=sources,
        ),
        "source_artifacts": list(sources),
    }


@dataclass(frozen=True, slots=True)
class _MdlEvidence:
    processed_count: int
    success_count: int
    reconstruction_failure_count: int
    baseline_data_bits: int
    total_mdl_bits: int
    source_name: str


def _required_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Goal5IntegrationError(f"{label} must be an object")
    return value


def _required_list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise Goal5IntegrationError(f"{label} must be a list")
    return value


def _required_nonnegative_int(value: object, *, label: str) -> int:
    result = _optional_nonnegative_int(value, label=label)
    if result is None:
        raise Goal5IntegrationError(f"{label} must not be null")
    return result


def _mdl_evidence(
    evaluation: object,
    *,
    source_name: str,
    label: str,
) -> _MdlEvidence:
    payload = _required_mapping(evaluation, label=label)
    summary = _required_mapping(payload.get("summary"), label=f"{label}.summary")
    processed = _required_nonnegative_int(
        summary.get("processed_count"),
        label=f"{label}.summary.processed_count",
    )
    success = _required_nonnegative_int(
        summary.get("success_count"),
        label=f"{label}.summary.success_count",
    )
    reconstruction_failures = _required_nonnegative_int(
        summary.get("reconstruction_failure_count"),
        label=f"{label}.summary.reconstruction_failure_count",
    )
    if success + reconstruction_failures != processed:
        raise Goal5IntegrationError(f"{label} MDL accounting does not partition processed rows")
    return _MdlEvidence(
        processed_count=processed,
        success_count=success,
        reconstruction_failure_count=reconstruction_failures,
        baseline_data_bits=_required_nonnegative_int(
            summary.get("baseline_data_bits"),
            label=f"{label}.summary.baseline_data_bits",
        ),
        total_mdl_bits=_required_nonnegative_int(
            summary.get("total_mdl_bits"),
            label=f"{label}.summary.total_mdl_bits",
        ),
        source_name=source_name,
    )


def _collect_mdl_evidence(
    *,
    frequent_completion: Mapping[str, object],
    frequent_run_dir: Path,
    learned_completion: Mapping[str, object],
    learned_run_dir: Path,
    repository_root: Path,
) -> tuple[
    dict[str, _MdlEvidence],
    dict[str, _MdlEvidence],
    dict[str, _MdlEvidence],
    dict[str, SourceArtifact],
    dict[str, object],
]:
    frequent_artifacts = _required_mapping(
        frequent_completion.get("artifacts"),
        label="issue 5-5 completion artifacts",
    )
    sweep_table, sweep_source = _load_referenced_json(
        repository_root=repository_root,
        run_dir=frequent_run_dir,
        reference=frequent_artifacts.get("sweep_table"),
        source_name="issue_5_5_sweep_table",
        schema_version="geml-goal5-frequent-sweep-table-v1",
        label="issue 5-5 sweep table",
    )
    heldout_frequent, frequent_heldout_source = _load_referenced_json(
        repository_root=repository_root,
        run_dir=frequent_run_dir,
        reference=frequent_artifacts.get("heldout_results"),
        source_name="issue_5_5_heldout_results",
        schema_version="geml-goal5-frequent-heldout-results-v1",
        label="issue 5-5 held-out results",
    )
    configurations = _required_list(
        sweep_table.get("configurations"),
        label="issue 5-5 sweep configurations",
    )
    selected_digest = frequent_completion.get("selected_configuration_digest")
    selected = [
        _required_mapping(item, label="issue 5-5 sweep configuration")
        for item in configurations
        if isinstance(item, dict) and item.get("configuration_digest") == selected_digest
    ]
    if len(selected) != 1:
        raise Goal5IntegrationError(
            "issue 5-5 selected configuration is not unique in the sweep table"
        )
    selected_configuration = selected[0]
    frequent_by_split = {
        "train": _mdl_evidence(
            selected_configuration.get("train"),
            source_name=sweep_source.name,
            label="issue 5-5 TRAIN",
        ),
        "validation": _mdl_evidence(
            selected_configuration.get("validation"),
            source_name=sweep_source.name,
            label="issue 5-5 VALIDATION",
        ),
    }
    heldout_splits = _required_mapping(
        heldout_frequent.get("splits"),
        label="issue 5-5 held-out splits",
    )
    for split in ("test_iid", "test_ood"):
        frequent_by_split[split] = _mdl_evidence(
            heldout_splits.get(split),
            source_name=frequent_heldout_source.name,
            label=f"issue 5-5 {split}",
        )

    learned_artifacts = _required_mapping(
        learned_completion.get("artifacts"),
        label="issue 5-6 completion artifacts",
    )
    validation, validation_source = _load_referenced_json(
        repository_root=repository_root,
        run_dir=learned_run_dir,
        reference=learned_artifacts.get("validation_results"),
        source_name="issue_5_6_validation_results",
        schema_version="geml-goal5-learned-validation-results-v1",
        label="issue 5-6 validation results",
    )
    heldout_learned, learned_heldout_source = _load_referenced_json(
        repository_root=repository_root,
        run_dir=learned_run_dir,
        reference=learned_artifacts.get("heldout_results"),
        source_name="issue_5_6_heldout_results",
        schema_version="geml-goal5-learned-heldout-results-v1",
        label="issue 5-6 held-out results",
    )
    experiment_result, experiment_source = _load_referenced_json(
        repository_root=repository_root,
        run_dir=learned_run_dir,
        reference=learned_artifacts.get("experiment_result"),
        source_name="issue_5_6_experiment_result",
        schema_version="geml-goal5-learned-results-v1",
        label="issue 5-6 experiment result",
    )
    candidates = _required_list(
        validation.get("candidates"),
        label="issue 5-6 validation candidates",
    )
    selected_index = _required_nonnegative_int(
        validation.get("selected_candidate_index"),
        label="issue 5-6 selected_candidate_index",
    )
    if selected_index >= len(candidates):
        raise Goal5IntegrationError("issue 5-6 selected candidate index is out of bounds")
    selected_candidate = _required_mapping(
        candidates[selected_index],
        label="issue 5-6 selected validation candidate",
    )
    learned_by_split = {
        "validation": _mdl_evidence(
            selected_candidate.get("evaluation"),
            source_name=validation_source.name,
            label="issue 5-6 learned VALIDATION",
        )
    }
    macro_by_split = {
        split: _MdlEvidence(
            processed_count=evidence.processed_count,
            success_count=evidence.success_count,
            reconstruction_failure_count=evidence.reconstruction_failure_count,
            baseline_data_bits=evidence.baseline_data_bits,
            total_mdl_bits=evidence.baseline_data_bits,
            source_name=evidence.source_name,
        )
        for split, evidence in frequent_by_split.items()
    }
    for split in ("test_iid", "test_ood"):
        split_payload = _required_mapping(
            heldout_learned.get(split),
            label=f"issue 5-6 {split} result",
        )
        learned_by_split[split] = _mdl_evidence(
            split_payload.get("learned"),
            source_name=learned_heldout_source.name,
            label=f"issue 5-6 learned {split}",
        )
    sources = {
        item.name: item
        for item in (
            sweep_source,
            frequent_heldout_source,
            validation_source,
            learned_heldout_source,
            experiment_source,
        )
    }
    return (
        macro_by_split,
        frequent_by_split,
        learned_by_split,
        sources,
        {
            "validation": validation,
            "heldout": heldout_learned,
            "experiment": experiment_result,
        },
    )


@dataclass(slots=True)
class _RankerCounts:
    attempted_group_count: int = 0
    validated_selection_count: int = 0
    failed_selected_count: int = 0
    exact_best_match_count: int = 0
    regret_group_count: int = 0
    total_regret: int = 0
    max_regret: int | None = None
    official_cost_scoring_calls: int = 0
    official_cost_scoring_seconds: float = 0.0


def _require_exact_ranker_method_cohorts(
    method_group_ids: Mapping[str, set[str]],
    *,
    expected_methods: Sequence[str],
    evaluable_count: int,
    split: str,
) -> None:
    """Require every ranker method to evaluate the same exact group-ID cohort."""

    if set(method_group_ids) != set(expected_methods):
        raise Goal5IntegrationError(f"issue 5-7 {split} method cohorts are incomplete")
    reference = method_group_ids[expected_methods[0]]
    if len(reference) != evaluable_count:
        raise Goal5IntegrationError(
            f"issue 5-7 {split} method cohort differs from evaluable_group_count"
        )
    for method in expected_methods[1:]:
        if method_group_ids[method] != reference:
            raise Goal5IntegrationError(
                f"issue 5-7 {split} methods cover different exact group-ID cohorts"
            )


def _require_goal4_candidate_split(
    expression_id: str,
    split: str,
    *,
    split_by_expression: Mapping[str, str],
) -> None:
    """Bind every ranker candidate group to its authoritative Goal 4 split."""

    expected_split = split_by_expression.get(expression_id)
    if expected_split is None:
        raise Goal5IntegrationError(
            "issue 5-7 candidate group references an expression outside Goal 4"
        )
    if split != expected_split:
        raise Goal5IntegrationError("issue 5-7 candidate group split disagrees with Goal 4")


def _scan_neural_ranker(
    *,
    repository_root: Path,
    completion: Mapping[str, object],
    run_dir: Path,
    cohorts: Goal4NontrivialCohorts,
) -> tuple[dict[str, object], dict[str, SourceArtifact], str]:
    artifacts = _required_mapping(
        completion.get("artifacts"),
        label="issue 5-7 completion artifacts",
    )
    run_manifest, run_manifest_source = _load_referenced_json(
        repository_root=repository_root,
        run_dir=run_dir,
        reference=artifacts.get("run.manifest.json"),
        source_name="issue_5_7_run_manifest",
        schema_version="geml-goal5-neural-ranker-run-v1",
        label="issue 5-7 run manifest",
    )
    reproduction_command = _required_string(
        run_manifest.get("reproduction_command"),
        field="issue 5-7 reproduction_command",
    )
    report, report_source = _load_referenced_json(
        repository_root=repository_root,
        run_dir=run_dir,
        reference=artifacts.get("report.json"),
        source_name="issue_5_7_report",
        schema_version="geml-goal5-neural-ranker-report-v1",
        label="issue 5-7 scientific report",
    )
    dataset = _required_mapping(report.get("dataset"), label="issue 5-7 report dataset")
    completion_dataset = _required_mapping(
        completion.get("dataset"),
        label="issue 5-7 completion dataset",
    )
    if dataset != completion_dataset:
        raise Goal5IntegrationError("issue 5-7 report and completion dataset summaries differ")

    candidate_path, candidate_sha256, candidate_size = _artifact_reference_path(
        run_dir,
        artifacts.get("candidate_groups.jsonl"),
        label="issue 5-7 candidate groups",
    )
    group_count = _required_nonnegative_int(
        dataset.get("group_count"),
        label="issue 5-7 group_count",
    )
    group_by_id: dict[str, tuple[str, str, str]] = {}
    group_keys: set[tuple[str, str]] = set()
    expression_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    candidate_count = 0
    valid_candidate_count = 0
    official_label_count = 0
    replay_mismatch_count = 0
    empty_group_count = 0
    goal4_split_by_expression = dict(cohorts.split_by_expression)
    with _JsonlStream(
        candidate_path,
        "issue 5-7 candidate groups",
        expected_sha256=candidate_sha256,
        expected_size=candidate_size,
        expected_count=group_count,
        schema_state=ArtifactSchemaState.VERSIONED,
        schema_version="geml-egraph-candidate-group-v1",
    ) as candidate_stream:
        for row_number, group in enumerate(candidate_stream, start=1):
            group_id = _required_sha256(
                group.get("group_id"),
                field=f"issue 5-7 candidate group {row_number} group_id",
            )
            expression_id = _expression_id(
                group.get("expression_id"),
                label=f"issue 5-7 candidate group {row_number} expression_id",
            )
            mode = group.get("rewrite_mode")
            split = group.get("split")
            status = group.get("source_stage_status")
            if mode not in {"safe_real", "positive_real_formal"}:
                raise Goal5IntegrationError(
                    f"issue 5-7 candidate group {row_number} has invalid rewrite_mode"
                )
            if split not in {item.value for item in SplitName}:
                raise Goal5IntegrationError(
                    f"issue 5-7 candidate group {row_number} has invalid split"
                )
            if not isinstance(status, str) or not status:
                raise Goal5IntegrationError(
                    f"issue 5-7 candidate group {row_number} has invalid source status"
                )
            key = (expression_id, mode)
            if group_id in group_by_id or key in group_keys:
                raise Goal5IntegrationError("issue 5-7 candidate groups contain duplicate keys")
            _require_goal4_candidate_split(
                expression_id,
                split,
                split_by_expression=goal4_split_by_expression,
            )
            group_by_id[group_id] = (expression_id, mode, split)
            group_keys.add(key)
            expression_ids.add(expression_id)
            split_counts[split] += 1
            status_counts[status] += 1
            candidates = _required_list(
                group.get("candidates"),
                label=f"issue 5-7 candidate group {row_number} candidates",
            )
            candidate_count += len(candidates)
            empty_group_count += not candidates
            replay_mismatch_count += group.get("replay_status") != "matched"
            for candidate in candidates:
                candidate_payload = _required_mapping(
                    candidate,
                    label=f"issue 5-7 candidate group {row_number} candidate",
                )
                rankable = (
                    candidate_payload.get("validation_status") == "valid"
                    and candidate_payload.get("official_cost_status") == "success"
                    and candidate_payload.get("official_eml_dag_cost") is not None
                )
                valid_candidate_count += rankable
                official_label_count += candidate_payload.get("official_eml_dag_cost") is not None
    expected_dataset_scalars = {
        "group_count": candidate_stream.record_count,
        "expression_count": len(expression_ids),
        "candidate_count": candidate_count,
        "valid_candidate_count": valid_candidate_count,
        "failed_candidate_count": candidate_count - valid_candidate_count,
        "official_cost_label_count": official_label_count,
        "replay_mismatch_count": replay_mismatch_count,
        "empty_group_count": empty_group_count,
        "groups_by_split": dict(sorted(split_counts.items())),
        "groups_by_source_status": dict(sorted(status_counts.items())),
    }
    for name, observed in expected_dataset_scalars.items():
        if dataset.get(name) != observed:
            raise Goal5IntegrationError(
                f"issue 5-7 candidate groups disagree with dataset field {name}"
            )
    for mode, ids in (
        ("safe_real", cohorts.safe_expression_ids),
        ("positive_real_formal", cohorts.domain_expression_ids),
    ):
        missing = next(
            (expression_id for expression_id in ids if (expression_id, mode) not in group_keys),
            None,
        )
        if missing is not None:
            raise Goal5IntegrationError(
                f"issue 5-7 candidate groups are missing exact cohort ID {missing}"
            )
    candidate_source = _repository_source(
        name="issue_5_7_candidate_groups",
        path=candidate_path,
        repository_root=repository_root,
        sha256=candidate_stream.sha256,
        size_bytes=candidate_stream.size_bytes,
        media_type="application/x-ndjson",
        schema_version="geml-egraph-candidate-group-v1",
    )

    safe_ids = frozenset(cohorts.safe_expression_ids)
    domain_ids = frozenset(cohorts.domain_expression_ids)
    counts: dict[tuple[str, str, str], _RankerCounts] = defaultdict(_RankerCounts)
    outcome_sources: dict[str, SourceArtifact] = {}
    expected_methods = tuple(method.value for method in RankerMethod)
    for split in ("validation", "test_iid", "test_ood"):
        split_report = _required_mapping(
            report.get(split),
            label=f"issue 5-7 report {split}",
        )
        evaluable_count = _required_nonnegative_int(
            split_report.get("evaluable_group_count"),
            label=f"issue 5-7 {split} evaluable_group_count",
        )
        path, expected_sha256, expected_size = _artifact_reference_path(
            run_dir,
            artifacts.get(f"{split}.outcomes.jsonl"),
            label=f"issue 5-7 {split} outcomes",
        )
        source_name = f"issue_5_7_{split}_outcomes"
        with _JsonlStream(
            path,
            f"issue 5-7 {split} outcomes",
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            expected_count=evaluable_count * len(expected_methods),
            schema_state=ArtifactSchemaState.EXPLICITLY_UNVERSIONED,
            schema_version=None,
        ) as outcome_stream:
            seen: set[tuple[str, str]] = set()
            method_group_ids = {method: set() for method in expected_methods}
            for row_number, outcome in enumerate(outcome_stream, start=1):
                group_id = _required_sha256(
                    outcome.get("group_id"),
                    field=f"issue 5-7 {split} outcome {row_number} group_id",
                )
                method = outcome.get("method")
                if method not in expected_methods:
                    raise Goal5IntegrationError(
                        f"issue 5-7 {split} outcome {row_number} has invalid method"
                    )
                key = (group_id, method)
                if key in seen:
                    raise Goal5IntegrationError(
                        f"issue 5-7 {split} outcomes contain duplicate method/group"
                    )
                seen.add(key)
                method_group_ids[method].add(group_id)
                group_info = group_by_id.get(group_id)
                if group_info is None:
                    raise Goal5IntegrationError(
                        f"issue 5-7 {split} outcome references unknown candidate group"
                    )
                expression_id, rewrite_mode, group_split = group_info
                if (
                    group_split != split
                    or outcome.get("split") != split
                    or outcome.get("expression_id") != expression_id
                    or outcome.get("rewrite_mode") != rewrite_mode
                ):
                    raise Goal5IntegrationError(
                        f"issue 5-7 {split} outcome identity does not match its group"
                    )
                subsets = ["all"]
                if rewrite_mode == "safe_real" and expression_id in safe_ids:
                    subsets.append("safe_nontrivial")
                if rewrite_mode == "positive_real_formal" and expression_id in domain_ids:
                    subsets.append("domain_nontrivial")
                selected_valid = outcome.get("selected_valid")
                exact_best_match = outcome.get("exact_best_match")
                if type(selected_valid) is not bool or type(exact_best_match) is not bool:
                    raise Goal5IntegrationError(f"issue 5-7 {split} outcome booleans are invalid")
                regret = outcome.get("regret")
                if regret is not None and (
                    isinstance(regret, bool) or not isinstance(regret, int) or regret < 0
                ):
                    raise Goal5IntegrationError(f"issue 5-7 {split} outcome regret is invalid")
                calls = _required_nonnegative_int(
                    outcome.get("official_cost_scoring_calls"),
                    label=f"issue 5-7 {split} outcome official calls",
                )
                seconds = _optional_nonnegative_float(
                    outcome.get("official_cost_scoring_seconds"),
                    label=f"issue 5-7 {split} outcome official seconds",
                )
                if seconds is None:
                    raise Goal5IntegrationError(
                        f"issue 5-7 {split} outcome official seconds must be observed"
                    )
                for subset in subsets:
                    aggregate = counts[(method, split, subset)]
                    aggregate.attempted_group_count += 1
                    aggregate.validated_selection_count += selected_valid
                    aggregate.failed_selected_count += not selected_valid
                    aggregate.exact_best_match_count += exact_best_match
                    if regret is not None:
                        aggregate.regret_group_count += 1
                        aggregate.total_regret += regret
                        aggregate.max_regret = (
                            regret
                            if aggregate.max_regret is None
                            else max(aggregate.max_regret, regret)
                        )
                    aggregate.official_cost_scoring_calls += calls
                    aggregate.official_cost_scoring_seconds += seconds
        _require_exact_ranker_method_cohorts(
            method_group_ids,
            expected_methods=expected_methods,
            evaluable_count=evaluable_count,
            split=split,
        )
        outcome_sources[source_name] = _repository_source(
            name=source_name,
            path=path,
            repository_root=repository_root,
            sha256=outcome_stream.sha256,
            size_bytes=outcome_stream.size_bytes,
            media_type="application/x-ndjson",
            schema_version=None,
            unversioned_reason=(
                "issue 5-7 outcome rows predate a row-level schema_version; "
                "their exact bytes are authenticated by the atomic completion"
            ),
        )

        report_metrics = _required_list(
            split_report.get("metrics"),
            label=f"issue 5-7 {split} metrics",
        )
        metrics_by_method = {
            item.get("method"): item
            for item in (
                _required_mapping(value, label=f"issue 5-7 {split} metric")
                for value in report_metrics
            )
        }
        if tuple(metrics_by_method) != expected_methods:
            raise Goal5IntegrationError(
                f"issue 5-7 {split} method ordering differs from the frozen contract"
            )
        for method in expected_methods:
            observed = counts[(method, split, "all")]
            expected = metrics_by_method[method]
            exact_fields = {
                "attempted_group_count": observed.attempted_group_count,
                "validated_selection_count": observed.validated_selection_count,
                "failed_selected_count": observed.failed_selected_count,
                "exact_best_match_count": observed.exact_best_match_count,
                "regret_group_count": observed.regret_group_count,
                "total_regret": observed.total_regret,
                "max_regret": observed.max_regret,
                "official_cost_scoring_calls": observed.official_cost_scoring_calls,
            }
            if any(expected.get(name) != value for name, value in exact_fields.items()):
                raise Goal5IntegrationError(
                    f"issue 5-7 {split}/{method} outcomes disagree with report metrics"
                )
            expected_seconds = expected.get("official_cost_scoring_seconds")
            if type(expected_seconds) is not float or not math.isclose(
                observed.official_cost_scoring_seconds,
                expected_seconds,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise Goal5IntegrationError(
                    f"issue 5-7 {split}/{method} scoring time disagrees with report"
                )

    cohort_by_split = {("safe_nontrivial", split): ids for split, ids in cohorts.safe_by_split}
    cohort_by_split.update(
        {("domain_nontrivial", split): ids for split, ids in cohorts.domain_by_split}
    )
    slices: list[dict[str, object]] = []
    for method in RankerMethod:
        for split in ("validation", "test_iid", "test_ood"):
            split_report = _required_mapping(
                report.get(split),
                label=f"issue 5-7 report {split}",
            )
            for subset in ("all", "safe_nontrivial", "domain_nontrivial"):
                denominator = (
                    _required_nonnegative_int(
                        split_report.get("total_group_count"),
                        label=f"issue 5-7 {split} total_group_count",
                    )
                    if subset == "all"
                    else len(cohort_by_split[(subset, split)])
                )
                aggregate = counts[(method.value, split, subset)]
                if aggregate.attempted_group_count > denominator:
                    raise Goal5IntegrationError(
                        f"issue 5-7 {split}/{subset} outcomes exceed exact denominator"
                    )
                sources = sorted(
                    {
                        report_source.name,
                        candidate_source.name,
                        outcome_sources[f"issue_5_7_{split}_outcomes"].name,
                        *(() if subset == "all" else ("goal4_rows",)),
                    }
                )
                slices.append(
                    {
                        "method": method.value,
                        "split": split,
                        "subset": subset,
                        "denominator_count": denominator,
                        "evaluable_group_count": aggregate.attempted_group_count,
                        "unevaluable_group_count": (denominator - aggregate.attempted_group_count),
                        "attempted_group_count": aggregate.attempted_group_count,
                        "validated_selection_count": (aggregate.validated_selection_count),
                        "failed_selected_count": aggregate.failed_selected_count,
                        "exact_best_match_count": aggregate.exact_best_match_count,
                        "regret_group_count": aggregate.regret_group_count,
                        "total_regret_eml_dag_nodes": aggregate.total_regret,
                        "max_regret_eml_dag_nodes": aggregate.max_regret,
                        "official_cost_scoring_calls": (aggregate.official_cost_scoring_calls),
                        "official_cost_scoring_seconds": (aggregate.official_cost_scoring_seconds),
                        "source_artifacts": sources,
                    }
                )
    runtime = _required_mapping(report.get("runtime"), label="issue 5-7 runtime")
    fit = _required_mapping(report.get("fit"), label="issue 5-7 fit")
    neural_payload = {
        "ground_truth_cost": "official_pure_eml_dag_nodes",
        "dataset": {
            "group_count": dataset.get("group_count"),
            "expression_count": dataset.get("expression_count"),
            "candidate_count": dataset.get("candidate_count"),
            "valid_candidate_count": dataset.get("valid_candidate_count"),
            "failed_candidate_count": dataset.get("failed_candidate_count"),
            "official_cost_label_count": dataset.get("official_cost_label_count"),
            "replay_mismatch_count": dataset.get("replay_mismatch_count"),
            "empty_group_count": dataset.get("empty_group_count"),
            "groups_by_split": dataset.get("groups_by_split"),
            "groups_by_source_status": dataset.get("groups_by_source_status"),
            "source_artifacts": sorted(
                {
                    "issue_5_7_neural_ranker",
                    candidate_source.name,
                    report_source.name,
                    run_manifest_source.name,
                }
            ),
        },
        "fit": {
            "training_group_count": fit.get("training_group_count"),
            "training_candidate_count": fit.get("training_candidate_count"),
            "validation_group_count": fit.get("validation_group_count"),
            "selected_ridge": fit.get("selected_ridge"),
            "source_artifacts": [report_source.name],
        },
        "runtime": {
            **runtime,
            "source_artifacts": [report_source.name],
        },
        "slices": slices,
    }
    sources = {
        report_source.name: report_source,
        run_manifest_source.name: run_manifest_source,
        candidate_source.name: candidate_source,
        **outcome_sources,
    }
    return neural_payload, sources, reproduction_command


def _oci_content_descriptor(
    value: object,
    *,
    label: str,
) -> tuple[str, int]:
    payload = _required_mapping(value, label=label)
    digest = payload.get("digest")
    size = payload.get("size")
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != len("sha256:") + 64
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise Goal5IntegrationError(f"{label}.digest must be an OCI SHA-256 digest")
    return (
        digest[7:],
        _required_nonnegative_int(size, label=f"{label}.size"),
    )


def _export_shard_stream(
    *,
    batch_dir: Path,
    descriptor: Mapping[str, object],
    label: str,
) -> _JsonlStream:
    relative = descriptor.get("path")
    if not isinstance(relative, str) or not relative:
        raise Goal5IntegrationError(f"{label}.path must be nonblank")
    path = _path_within(batch_dir, relative)
    expected_sha256, expected_size = _oci_content_descriptor(
        descriptor.get("content"),
        label=f"{label}.content",
    )
    return _JsonlStream(
        path,
        label,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        expected_count=_required_nonnegative_int(
            descriptor.get("row_count"),
            label=f"{label}.row_count",
        ),
    )


def _validate_production_batch_manifest_header(
    manifest: Mapping[str, object],
    *,
    production_dataset_id: str,
    batch_id: str,
) -> None:
    """Validate the 5-8 contract's batch-scoped inner dataset identity."""

    expected_dataset_id = f"{production_dataset_id}:{batch_id}"
    if (
        manifest.get("schema_version") != "geml-goal5-graph-export-v1"
        or manifest.get("dataset_id") != expected_dataset_id
        or manifest.get("subset_label_policy") != "explicit-only-default-empty"
    ):
        raise Goal5IntegrationError("issue 5-8 batch manifest contract mismatch")


def _model_plane_size_index(
    rows: Iterable[Mapping[str, object]],
    *,
    representation_mode: str,
    representation_family: str,
    label: str,
) -> dict[str, tuple[int, int]]:
    """Index a bounded model shard by digest; model rows are not expression ordered."""

    sizes: dict[str, tuple[int, int]] = {}
    for row_number, model in enumerate(rows, start=1):
        digest = _required_string(
            model.get("model_payload_digest"),
            field=f"{label} row {row_number} model_payload_digest",
        )
        if (
            not digest.startswith("sha256:")
            or len(digest) != len("sha256:") + 64
            or any(character not in "0123456789abcdef" for character in digest[7:])
        ):
            raise Goal5IntegrationError(f"{label} row {row_number} model_payload_digest is invalid")
        payload = _required_mapping(
            model.get("payload"),
            label=f"{label} row {row_number} payload",
        )
        if (
            payload.get("schema_version") != "geml-goal5-model-features-v1"
            or payload.get("representation_mode") != representation_mode
            or payload.get("representation_family") != representation_family
        ):
            raise Goal5IntegrationError(f"{label} row {row_number} model binding mismatch")
        nodes = _required_list(
            payload.get("nodes"),
            label=f"{label} row {row_number} nodes",
        )
        edges = _required_list(
            payload.get("edges"),
            label=f"{label} row {row_number} edges",
        )
        if not nodes:
            raise Goal5IntegrationError(f"{label} row {row_number} has no nodes")
        if digest in sizes:
            raise Goal5IntegrationError(f"{label} repeats model payload digest {digest}")
        sizes[digest] = (len(nodes), len(edges))
    return sizes


def _bind_production_graph_plane(
    *,
    metadata_rows: Iterable[Mapping[str, object]],
    audit_rows: Iterable[Mapping[str, object]],
    model_sizes_by_digest: Mapping[str, tuple[int, int]],
    expected_expression_ids: set[str],
    split: str,
    representation_mode: str,
    representation_family: str,
    reconstruction_status: str,
    label: str,
) -> dict[str, tuple[int, int]]:
    """Join expression-bound rows to content-deduplicated model payloads."""

    audits_by_expression: dict[str, Mapping[str, object]] = {}
    for row_number, audit in enumerate(audit_rows, start=1):
        expression_id = _expression_id(
            audit.get("expression_id"),
            label=f"{label} audit row {row_number} expression ID",
        )
        if expression_id not in expected_expression_ids:
            raise Goal5IntegrationError(f"{label} audit references an unknown expression")
        if expression_id in audits_by_expression:
            raise Goal5IntegrationError(f"{label} repeats an audit expression ID")
        if (
            audit.get("split") != split
            or audit.get("subset_labels") != []
            or audit.get("representation_mode") != representation_mode
            or audit.get("representation_family") != representation_family
        ):
            raise Goal5IntegrationError(f"{label} audit identity mismatch")
        audits_by_expression[expression_id] = audit

    observations: dict[str, tuple[int, int]] = {}
    referenced_model_digests: set[str] = set()
    for row_number, metadata in enumerate(metadata_rows, start=1):
        expression_id = _expression_id(
            metadata.get("expression_id"),
            label=f"{label} metadata row {row_number} expression ID",
        )
        if expression_id not in expected_expression_ids:
            raise Goal5IntegrationError(f"{label} metadata references an unknown expression")
        if expression_id in observations:
            raise Goal5IntegrationError(f"{label} repeats a metadata expression ID")
        if (
            metadata.get("split") != split
            or metadata.get("subset_labels") != []
            or metadata.get("representation_mode") != representation_mode
            or metadata.get("representation_family") != representation_family
        ):
            raise Goal5IntegrationError(f"{label} metadata identity mismatch")
        audit = audits_by_expression.pop(expression_id, None)
        if audit is None:
            raise Goal5IntegrationError(f"{label} metadata has no expression-bound audit")
        graph_digest = metadata.get("graph_digest")
        if audit.get("graph_digest") != graph_digest:
            raise Goal5IntegrationError(f"{label} metadata/audit graph digest mismatch")
        model_digest = _required_string(
            metadata.get("model_payload_digest"),
            field=f"{label} metadata row {row_number} model_payload_digest",
        )
        size = model_sizes_by_digest.get(model_digest)
        if size is None:
            raise Goal5IntegrationError(f"{label} references a missing model payload digest")
        if audit.get("validation_status") != "passed":
            raise Goal5IntegrationError(
                "completed issue 5-8 export contains failed graph validation"
            )
        if audit.get("reconstruction_status") != reconstruction_status:
            raise Goal5IntegrationError(f"{label} reconstruction status is invalid")
        observations[expression_id] = size
        referenced_model_digests.add(model_digest)

    if set(observations) != expected_expression_ids:
        raise Goal5IntegrationError(f"{label} metadata expression cohort is incomplete")
    if audits_by_expression:
        raise Goal5IntegrationError(f"{label} audit expression cohort is incomplete")
    orphan_model_digests = set(model_sizes_by_digest) - referenced_model_digests
    if orphan_model_digests:
        raise Goal5IntegrationError(f"{label} contains an orphan model payload digest")
    return observations


def _validated_production_export_completion(
    completion_path: Path,
    expected_payload: Mapping[str, object],
) -> ProductionExportManifest:
    """Load the authoritative 5-8 completion model from its exact JSON bytes."""

    try:
        data = completion_path.read_bytes()
    except OSError as error:
        raise Goal5IntegrationError("cannot read issue 5-8 production completion") from error
    decoded = _decode_json_object(data, label="issue 5-8 production completion")
    if decoded != dict(expected_payload):
        raise Goal5IntegrationError("issue 5-8 completion bytes changed after authentication")
    try:
        completion = ProductionExportManifest.model_validate_json(data)
    except Exception as error:
        raise Goal5IntegrationError(
            "issue 5-8 completion violates the authoritative production schema"
        ) from error
    if completion.model_dump(mode="json", by_alias=True) != decoded:
        raise Goal5IntegrationError("issue 5-8 completion is not losslessly normalized")
    return completion


def _validate_production_hierarchy_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_expression_ids: Sequence[str],
    split: str,
    label: str,
) -> None:
    """Require one correctly ordered hierarchy row for every batch expression."""

    observed_count = 0
    for row_number, hierarchy in enumerate(rows):
        if (
            row_number >= len(expected_expression_ids)
            or hierarchy.get("schema_version") != "geml-goal5-hierarchy-v1"
            or hierarchy.get("expression_id") != expected_expression_ids[row_number]
            or hierarchy.get("split") != split
            or hierarchy.get("subset_labels") != []
        ):
            raise Goal5IntegrationError(f"{label} hierarchy alignment failed")
        observed_count += 1
    if observed_count != len(expected_expression_ids):
        raise Goal5IntegrationError(f"{label} hierarchy row count is incomplete")


def _scan_production_export(
    *,
    repository_root: Path,
    completion: Mapping[str, object],
    completion_path: Path,
    cohorts: Goal4NontrivialCohorts,
) -> tuple[
    dict[tuple[GraphTrackName, str, str], _GraphCounts],
    str,
    dict[str, object],
]:
    validated_completion = _validated_production_export_completion(
        completion_path,
        completion,
    )
    completion = validated_completion.model_dump(mode="json", by_alias=True)
    run_dir = completion_path.parent
    dataset_id = _required_string(completion.get("dataset_id"), field="issue 5-8 dataset_id")
    representations = _required_list(
        completion.get("representations"),
        label="issue 5-8 representations",
    )
    expected_track_names = (
        GraphTrackName.AST_DAG,
        GraphTrackName.PURE_EML_DAG,
        GraphTrackName.MACRO_DAG,
        GraphTrackName.FREQUENT_MOTIF_DAG,
        GraphTrackName.LEARNED_MOTIF_DAG,
    )
    representation_by_mode: dict[str, tuple[GraphTrackName, str]] = {}
    observed_names: list[GraphTrackName] = []
    for index, raw_representation in enumerate(representations):
        representation = _required_mapping(
            raw_representation,
            label=f"issue 5-8 representation {index}",
        )
        try:
            track = GraphTrackName(representation.get("name"))
        except ValueError as error:
            raise Goal5IntegrationError(
                f"issue 5-8 representation {index} has an invalid name"
            ) from error
        mode = _required_string(
            representation.get("representation_mode"),
            field=f"issue 5-8 representation {index} mode",
        )
        family = _required_string(
            representation.get("representation_family"),
            field=f"issue 5-8 representation {index} family",
        )
        if mode in representation_by_mode:
            raise Goal5IntegrationError("issue 5-8 representation modes must be unique")
        observed_names.append(track)
        representation_by_mode[mode] = (track, family)
    if tuple(observed_names) != expected_track_names:
        raise Goal5IntegrationError("issue 5-8 representations differ from frozen order")

    safe_ids = frozenset(cohorts.safe_expression_ids)
    domain_ids = frozenset(cohorts.domain_expression_ids)
    counts: dict[tuple[GraphTrackName, str, str], _GraphCounts] = defaultdict(_GraphCounts)
    split_expression_counts: Counter[str] = Counter()
    batches = _required_list(completion.get("batches"), label="issue 5-8 batches")
    with tempfile.TemporaryDirectory(prefix="geml-goal5-export-ids-") as temporary:
        database_path = Path(temporary) / "expression-ids.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute(
                "CREATE TABLE expression_ids (expression_id TEXT PRIMARY KEY, split TEXT NOT NULL)"
            )
            for batch_index, raw_batch in enumerate(batches):
                batch = _required_mapping(
                    raw_batch,
                    label=f"issue 5-8 batch {batch_index}",
                )
                relative_batch_path = batch.get("path")
                if not isinstance(relative_batch_path, str) or not relative_batch_path:
                    raise Goal5IntegrationError(
                        f"issue 5-8 batch {batch_index} path must be nonblank"
                    )
                batch_dir = _path_within(run_dir, relative_batch_path)
                manifest_path = _path_within(batch_dir, "manifest.json")
                expected_manifest_sha, expected_manifest_size = _oci_content_descriptor(
                    batch.get("manifest"),
                    label=f"issue 5-8 batch {batch_index} manifest",
                )
                try:
                    manifest_data = manifest_path.read_bytes()
                except OSError as error:
                    raise Goal5IntegrationError(
                        f"cannot read issue 5-8 batch {batch_index} manifest"
                    ) from error
                if len(manifest_data) != expected_manifest_size:
                    raise Goal5IntegrationError(
                        f"issue 5-8 batch {batch_index} manifest size mismatch"
                    )
                if _sha256_bytes(manifest_data) != expected_manifest_sha:
                    raise Goal5IntegrationError(
                        f"issue 5-8 batch {batch_index} manifest SHA-256 mismatch"
                    )
                manifest = _decode_json_object(
                    manifest_data,
                    label=f"issue 5-8 batch {batch_index} manifest",
                )
                batch_id = _required_string(
                    batch.get("batch_id"),
                    field=f"issue 5-8 batch {batch_index} batch_id",
                )
                try:
                    _validate_production_batch_manifest_header(
                        manifest,
                        production_dataset_id=dataset_id,
                        batch_id=batch_id,
                    )
                except Goal5IntegrationError as error:
                    raise Goal5IntegrationError(
                        f"issue 5-8 batch {batch_index} manifest contract mismatch"
                    ) from error
                batch_expression_count = _required_nonnegative_int(
                    batch.get("expression_count"),
                    label=f"issue 5-8 batch {batch_index} expression_count",
                )
                batch_hierarchy_count = _required_nonnegative_int(
                    batch.get("hierarchy_count"),
                    label=f"issue 5-8 batch {batch_index} hierarchy_count",
                )
                if (
                    manifest.get("expression_count") != batch_expression_count
                    or manifest.get("graph_count")
                    != _required_nonnegative_int(
                        batch.get("graph_count"),
                        label=f"issue 5-8 batch {batch_index} graph_count",
                    )
                    or manifest.get("validation_failure_count") != 0
                    or manifest.get("reconstruction_failure_count") != 0
                ):
                    raise Goal5IntegrationError(
                        f"issue 5-8 batch {batch_index} aggregate counts mismatch"
                    )
                if batch_hierarchy_count != batch_expression_count:
                    raise Goal5IntegrationError(
                        f"issue 5-8 batch {batch_index} hierarchy count is incomplete"
                    )
                batch_split = batch.get("split")
                if batch_split not in {item.value for item in SplitName}:
                    raise Goal5IntegrationError(f"issue 5-8 batch {batch_index} split is invalid")
                raw_shards = _required_list(
                    manifest.get("shards"),
                    label=f"issue 5-8 batch {batch_index} shards",
                )
                shard_by_key: dict[tuple[str, str | None], dict[str, object]] = {}
                for raw_shard in raw_shards:
                    shard = _required_mapping(
                        raw_shard,
                        label=f"issue 5-8 batch {batch_index} shard",
                    )
                    key = (shard.get("record_type"), shard.get("representation_mode"))
                    if key in shard_by_key:
                        raise Goal5IntegrationError(
                            f"issue 5-8 batch {batch_index} has duplicate shard group"
                        )
                    shard_by_key[key] = shard
                expected_keys = {
                    ("expression_metadata", None),
                    ("hierarchy_metadata", None),
                    *(
                        (record_type, mode)
                        for mode in representation_by_mode
                        for record_type in (
                            "graph_metadata",
                            "model_graph",
                            "graph_audit",
                        )
                    ),
                }
                if set(shard_by_key) != expected_keys:
                    raise Goal5IntegrationError(
                        f"issue 5-8 batch {batch_index} shard set is incomplete"
                    )

                expression_ids: list[str] = []
                expression_descriptor = shard_by_key[("expression_metadata", None)]
                with _export_shard_stream(
                    batch_dir=batch_dir,
                    descriptor=expression_descriptor,
                    label=f"issue 5-8 batch {batch_index} expression metadata",
                ) as expression_stream:
                    for row_number, expression in enumerate(expression_stream, start=1):
                        expression_id = _expression_id(
                            expression.get("expression_id"),
                            label=(f"issue 5-8 batch {batch_index} expression {row_number} ID"),
                        )
                        if (
                            expression.get("split") != batch_split
                            or expression.get("subset_labels") != []
                        ):
                            raise Goal5IntegrationError(
                                f"issue 5-8 batch {batch_index} expression metadata "
                                "violates split/subset contract"
                            )
                        expression_ids.append(expression_id)
                if len(expression_ids) != batch_expression_count:
                    raise Goal5IntegrationError(
                        f"issue 5-8 batch {batch_index} expression row count mismatch"
                    )
                try:
                    connection.executemany(
                        "INSERT INTO expression_ids(expression_id, split) VALUES (?, ?)",
                        ((expression_id, batch_split) for expression_id in expression_ids),
                    )
                except sqlite3.IntegrityError as error:
                    raise Goal5IntegrationError(
                        "issue 5-8 export contains duplicate expression IDs"
                    ) from error
                split_expression_counts[batch_split] += len(expression_ids)

                hierarchy_descriptor = shard_by_key[("hierarchy_metadata", None)]
                with _export_shard_stream(
                    batch_dir=batch_dir,
                    descriptor=hierarchy_descriptor,
                    label=f"issue 5-8 batch {batch_index} hierarchy metadata",
                ) as hierarchy_stream:
                    _validate_production_hierarchy_rows(
                        hierarchy_stream,
                        expected_expression_ids=expression_ids,
                        split=batch_split,
                        label=f"issue 5-8 batch {batch_index}",
                    )
                if hierarchy_stream.record_count != batch_hierarchy_count:
                    raise Goal5IntegrationError(
                        f"issue 5-8 batch {batch_index} hierarchy descriptor count mismatch"
                    )

                expected_expression_ids = set(expression_ids)
                for mode, (track, family) in representation_by_mode.items():
                    metadata_descriptor = shard_by_key[("graph_metadata", mode)]
                    model_descriptor = shard_by_key[("model_graph", mode)]
                    audit_descriptor = shard_by_key[("graph_audit", mode)]
                    label = f"issue 5-8 batch {batch_index} {track.value}"
                    with _export_shard_stream(
                        batch_dir=batch_dir,
                        descriptor=model_descriptor,
                        label=f"{label} model graph",
                    ) as model_stream:
                        model_sizes = _model_plane_size_index(
                            model_stream,
                            representation_mode=mode,
                            representation_family=family,
                            label=f"{label} model graph",
                        )
                    with (
                        _export_shard_stream(
                            batch_dir=batch_dir,
                            descriptor=metadata_descriptor,
                            label=f"{label} graph metadata",
                        ) as metadata_stream,
                        _export_shard_stream(
                            batch_dir=batch_dir,
                            descriptor=audit_descriptor,
                            label=f"{label} graph audit",
                        ) as audit_stream,
                    ):
                        observations = _bind_production_graph_plane(
                            metadata_rows=metadata_stream,
                            audit_rows=audit_stream,
                            model_sizes_by_digest=model_sizes,
                            expected_expression_ids=expected_expression_ids,
                            split=batch_split,
                            representation_mode=mode,
                            representation_family=family,
                            reconstruction_status=(
                                "passed"
                                if track
                                in {
                                    GraphTrackName.MACRO_DAG,
                                    GraphTrackName.FREQUENT_MOTIF_DAG,
                                    GraphTrackName.LEARNED_MOTIF_DAG,
                                }
                                else "not_requested"
                            ),
                            label=label,
                        )
                    for expression_id, (node_count, edge_count) in observations.items():
                        for subset in _subset_memberships(
                            expression_id,
                            safe_ids=safe_ids,
                            domain_ids=domain_ids,
                        ):
                            aggregate = counts[(track, batch_split, subset)]
                            aggregate.denominator_count += 1
                            aggregate.success_count += 1
                            aggregate.node_observation_count += 1
                            aggregate.node_total += node_count
                            aggregate.edge_observation_count += 1
                            aggregate.edge_total += edge_count
                            aggregate.structural_attempted_count += 1
                            aggregate.structural_passed_count += 1
                            if track is GraphTrackName.MACRO_DAG:
                                aggregate.expansion_attempted_count += 1
                                aggregate.expansion_passed_count += 1
                            elif track in {
                                GraphTrackName.FREQUENT_MOTIF_DAG,
                                GraphTrackName.LEARNED_MOTIF_DAG,
                            }:
                                aggregate.reconstruction_attempted_count += 1
                                aggregate.reconstruction_passed_count += 1
                if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(batches):
                    print(
                        f"[goal5-integration] authenticated {batch_index + 1}/"
                        f"{len(batches)} issue 5-8 batches",
                        flush=True,
                    )
            connection.commit()
            observed_expression_count = connection.execute(
                "SELECT COUNT(*) FROM expression_ids"
            ).fetchone()[0]
            expected_expression_count = _required_nonnegative_int(
                completion.get("expression_count"),
                label="issue 5-8 expression_count",
            )
            if observed_expression_count != expected_expression_count:
                raise Goal5IntegrationError(
                    "issue 5-8 unique expression count disagrees with completion"
                )
            digest = hashlib.sha256(b"geml-goal5-expression-cohort-v1\0")
            for (expression_id,) in connection.execute(
                "SELECT expression_id FROM expression_ids ORDER BY expression_id"
            ):
                digest.update(expression_id.encode("ascii"))
                digest.update(b"\0")
            all_ids_sha256 = digest.hexdigest()
        finally:
            connection.close()

    if sum(split_expression_counts.values()) != completion.get("expression_count"):
        raise Goal5IntegrationError("issue 5-8 split expression counts are incomplete")
    cohort_by_split = {("safe_nontrivial", split): ids for split, ids in cohorts.safe_by_split}
    cohort_by_split.update(
        {("domain_nontrivial", split): ids for split, ids in cohorts.domain_by_split}
    )
    for track in expected_track_names:
        for split in SplitName:
            all_count = counts[(track, split.value, "all")].denominator_count
            if all_count != split_expression_counts[split.value]:
                raise Goal5IntegrationError(
                    f"issue 5-8 {track.value}/{split.value} all denominator mismatch"
                )
            for subset in ("safe_nontrivial", "domain_nontrivial"):
                observed = counts[(track, split.value, subset)].denominator_count
                expected = len(cohort_by_split[(subset, split.value)])
                if observed != expected:
                    raise Goal5IntegrationError(
                        f"issue 5-8 {track.value}/{split.value}/{subset} exact join mismatch"
                    )
    production_export = {
        "batch_count": len(batches),
        "expression_count": completion.get("expression_count"),
        "graph_count": completion.get("graph_count"),
        "hierarchy_count": completion.get("hierarchy_count"),
        "validation_failure_count": completion.get("validation_failure_count"),
        "reconstruction_failure_count": completion.get("reconstruction_failure_count"),
        "representation_names": [item.value for item in expected_track_names],
        "subset_labels_available": False,
        "subset_label_reason": (
            "issue 5-8 freezes explicit-only-default-empty subset labels; Goal 4 "
            "nontrivial membership is joined externally by exact expression ID"
        ),
        "runtime_available": False,
        "runtime_reason": (
            "the issue 5-8 production completion and batch schemas do not publish "
            "per-graph or run runtime observations"
        ),
        "memory_available": False,
        "memory_reason": (
            "the issue 5-8 production completion and batch schemas do not publish "
            "peak-memory observations"
        ),
        "source_artifacts": ["issue_5_8_production_export"],
    }
    return counts, all_ids_sha256, production_export


def _completion_payload(path: Path, *, label: str) -> dict[str, object]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise Goal5IntegrationError(f"cannot read {label}") from error
    return _decode_json_object(data, label=label)


def _mdl_payload_for_slice(
    *,
    track: GraphTrackName,
    split: str,
    subset: str,
    success_count: int,
    macro_mdl: Mapping[str, _MdlEvidence],
    frequent_mdl: Mapping[str, _MdlEvidence],
    learned_mdl: Mapping[str, _MdlEvidence],
    default_source: str,
) -> dict[str, object]:
    if track in {
        GraphTrackName.AST_DAG,
        GraphTrackName.PURE_EML_DAG,
    }:
        return _unavailable_mdl(
            denominator_count=success_count,
            scope=MdlScope.STANDALONE_GRAPH,
            reason=(
                "issue 5-8 freezes lossless model graphs but does not publish an "
                "authenticated standalone MDL code length for this representation"
            ),
            sources=(default_source,),
        )
    if track in {
        GraphTrackName.SAFE_EGRAPH_EML_DAG,
        GraphTrackName.DOMAIN_EGRAPH_EML_DAG,
    }:
        return _unavailable_mdl(
            denominator_count=success_count,
            scope=MdlScope.STANDALONE_GRAPH,
            reason=(
                "Goal 4 selected_signature is non-reversible and cannot support an "
                "authenticated standalone MDL code length"
            ),
            sources=("goal4_rows",),
        )
    scope = (
        MdlScope.DICTIONARY_INCLUSIVE_MOTIF
        if track
        in {
            GraphTrackName.FREQUENT_MOTIF_DAG,
            GraphTrackName.LEARNED_MOTIF_DAG,
        }
        else MdlScope.STANDALONE_GRAPH
    )
    if subset != "all":
        return _unavailable_mdl(
            denominator_count=success_count,
            scope=scope,
            reason=(
                "the locked 5-5/5-6 MDL artifacts publish exact split aggregates but "
                "not per-expression code lengths for post-hoc Goal 4 cohorts"
            ),
            sources=(default_source,),
        )
    evidence_by_track = {
        GraphTrackName.MACRO_DAG: macro_mdl,
        GraphTrackName.FREQUENT_MOTIF_DAG: frequent_mdl,
        GraphTrackName.LEARNED_MOTIF_DAG: learned_mdl,
    }
    evidence = evidence_by_track[track].get(split)
    if evidence is None:
        return _unavailable_mdl(
            denominator_count=success_count,
            scope=scope,
            reason=(
                "the locked learned-vocabulary protocol does not publish a final "
                "TRAIN evaluation for the validation-selected learned vocabulary"
            ),
            sources=(default_source,),
        )
    if (
        evidence.processed_count != success_count
        or evidence.success_count != success_count
        or evidence.reconstruction_failure_count != 0
    ):
        raise Goal5IntegrationError(
            f"{track.value}/{split} MDL denominator does not match production export"
        )
    total_bits = (
        evidence.baseline_data_bits
        if track is GraphTrackName.MACRO_DAG
        else evidence.total_mdl_bits
    )
    return _available_mdl(
        denominator_count=success_count,
        total_bits=total_bits,
        scope=scope,
        sources=(evidence.source_name,),
    )


def _graph_tracks_payload(
    *,
    goal4_counts: Mapping[tuple[str, str, str], _GraphCounts],
    export_counts: Mapping[tuple[GraphTrackName, str, str], _GraphCounts],
    macro_mdl: Mapping[str, _MdlEvidence],
    frequent_mdl: Mapping[str, _MdlEvidence],
    learned_mdl: Mapping[str, _MdlEvidence],
) -> list[dict[str, object]]:
    definitions = {
        GraphTrackName.AST_DAG: (
            "Source AST-DAG",
            "ast",
            "ast",
            False,
        ),
        GraphTrackName.PURE_EML_DAG: (
            "Direct official-v4 Pure EML-DAG",
            "eml",
            "pure_eml:official_v4",
            True,
        ),
        GraphTrackName.SAFE_EGRAPH_EML_DAG: (
            "Safe-rule e-graph-selected Pure EML-DAG",
            "eml",
            "egraph-selected:safe_real:pure_eml:official_v4",
            True,
        ),
        GraphTrackName.DOMAIN_EGRAPH_EML_DAG: (
            "Domain-rule e-graph-selected Pure EML-DAG",
            "eml",
            "egraph-selected:positive_real_formal:pure_eml:official_v4",
            True,
        ),
        GraphTrackName.MACRO_DAG: (
            "Macro DAG",
            "macro",
            "macro:official_v4:is_pure_eml=false",
            False,
        ),
        GraphTrackName.FREQUENT_MOTIF_DAG: (
            "Frequent-motif DAG",
            "motif",
            "motif:frequent:locked-official-v4-macro",
            False,
        ),
        GraphTrackName.LEARNED_MOTIF_DAG: (
            "Learned-motif DAG",
            "motif",
            "motif:learned:locked-official-v4-macro",
            False,
        ),
    }
    tracks: list[dict[str, object]] = []
    for track in GraphTrackName:
        display_name, family, mode, is_pure_eml = definitions[track]
        goal4_mode = {
            GraphTrackName.SAFE_EGRAPH_EML_DAG: "safe_real",
            GraphTrackName.DOMAIN_EGRAPH_EML_DAG: "positive_real_formal",
        }.get(track)
        slices: list[dict[str, object]] = []
        for split in SplitName:
            for subset in ("all", "safe_nontrivial", "domain_nontrivial"):
                if goal4_mode is None:
                    counts = export_counts[(track, split.value, subset)]
                    base_sources = ["issue_5_8_production_export"]
                    mdl = _mdl_payload_for_slice(
                        track=track,
                        split=split.value,
                        subset=subset,
                        success_count=counts.success_count,
                        macro_mdl=macro_mdl,
                        frequent_mdl=frequent_mdl,
                        learned_mdl=learned_mdl,
                        default_source="issue_5_8_production_export",
                    )
                    slices.append(
                        _graph_slice_payload(
                            split=split.value,
                            subset=subset,
                            counts=counts,
                            source_names=(*base_sources, *mdl["source_artifacts"]),
                            edge_available=True,
                            mdl=mdl,
                            reconstruction_required=track
                            in {
                                GraphTrackName.FREQUENT_MOTIF_DAG,
                                GraphTrackName.LEARNED_MOTIF_DAG,
                            },
                            expansion_required=track is GraphTrackName.MACRO_DAG,
                            runtime_available=False,
                            unavailable_runtime_reason=(
                                "issue 5-8 does not publish per-graph or run runtime observations"
                            ),
                            unavailable_memory_reason=(
                                "issue 5-8 does not publish peak-memory observations"
                            ),
                        )
                    )
                    continue
                counts = goal4_counts[(goal4_mode, split.value, subset)]
                mdl = _mdl_payload_for_slice(
                    track=track,
                    split=split.value,
                    subset=subset,
                    success_count=counts.success_count,
                    macro_mdl=macro_mdl,
                    frequent_mdl=frequent_mdl,
                    learned_mdl=learned_mdl,
                    default_source="goal4_rows",
                )
                runtime_available = (
                    counts.runtime_observation_count > 0 or counts.denominator_count == 0
                )
                slices.append(
                    _graph_slice_payload(
                        split=split.value,
                        subset=subset,
                        counts=counts,
                        source_names=("goal4_rows", "goal4_run"),
                        edge_available=False,
                        mdl=mdl,
                        reconstruction_required=False,
                        expansion_required=False,
                        runtime_available=runtime_available,
                        unavailable_runtime_reason=(
                            "Goal 4 rows contain no wall-time observation for this cohort"
                        ),
                        unavailable_memory_reason=(
                            "Goal 4 records before/after RSS snapshots, not an "
                            "authenticated peak-memory measurement"
                        ),
                    )
                )
        tracks.append(
            {
                "name": track.value,
                "display_name": display_name,
                "representation_family": family,
                "representation_mode": mode,
                "is_pure_eml": is_pure_eml,
                "slices": slices,
            }
        )
    return tracks


def _learned_claims(
    learned_results: Mapping[str, object],
    neural_report: Mapping[str, object],
) -> list[dict[str, object]]:
    heldout = _required_mapping(
        learned_results.get("heldout"),
        label="issue 5-6 held-out results",
    )
    iid = _required_mapping(heldout.get("test_iid"), label="issue 5-6 TEST_IID")

    def total_bits(arm: str) -> int:
        evaluation = _required_mapping(iid.get(arm), label=f"issue 5-6 TEST_IID {arm}")
        summary = _required_mapping(
            evaluation.get("summary"),
            label=f"issue 5-6 TEST_IID {arm} summary",
        )
        return _required_nonnegative_int(
            summary.get("total_mdl_bits"),
            label=f"issue 5-6 TEST_IID {arm} total_mdl_bits",
        )

    learned_bits = total_bits("learned")
    frequent_bits = total_bits("frequent")
    macro_bits = total_bits("macro")
    random_median = iid.get("random_median_total_mdl_bits")
    if type(random_median) is not float or not math.isfinite(random_median):
        raise Goal5IntegrationError("issue 5-6 random median MDL must be a finite float")
    learned_count = _required_mapping(
        _required_mapping(iid.get("learned"), label="issue 5-6 learned IID").get("summary"),
        label="issue 5-6 learned IID summary",
    ).get("processed_count")
    denominator = _required_nonnegative_int(
        learned_count,
        label="issue 5-6 learned IID processed_count",
    )
    experiment = _required_mapping(
        learned_results.get("experiment"),
        label="issue 5-6 experiment result",
    )
    if experiment.get("claim_status") != "null_result":
        raise Goal5IntegrationError(
            "issue 5-6 production claim status changed from the locked null result"
        )

    iid_ranker = _required_mapping(
        neural_report.get("test_iid"),
        label="issue 5-7 TEST_IID report",
    )
    metrics = {
        item.get("method"): item
        for item in (
            _required_mapping(value, label="issue 5-7 TEST_IID metric")
            for value in _required_list(
                iid_ranker.get("metrics"),
                label="issue 5-7 TEST_IID metrics",
            )
        )
    }
    neural = metrics[RankerMethod.NEURAL.value]
    comparison = _required_mapping(
        _required_mapping(
            neural_report.get("heuristic_comparison"),
            label="issue 5-7 heuristic comparison",
        ).get("test_iid"),
        label="issue 5-7 TEST_IID heuristic comparison",
    )
    outperformers = _required_list(
        comparison.get("heuristics_outperform_neural"),
        label="issue 5-7 outperforming heuristics",
    )
    baseline_parts = []
    for method in outperformers:
        if not isinstance(method, str) or method not in metrics:
            raise Goal5IntegrationError("issue 5-7 comparison names an unknown heuristic")
        metric = metrics[method]
        baseline_parts.append(
            f"{method}: {metric.get('exact_best_match_count')}/"
            f"{iid_ranker.get('total_group_count')} exact-best, "
            f"{metric.get('total_regret')} total regret"
        )
    ranker_outcome = (
        ClaimOutcome.NULL_RESULT
        if comparison.get("claim_status") == "null_result"
        else ClaimOutcome.POSITIVE
    )
    return [
        {
            "claim_id": ClaimId.LEARNED_VS_FREQUENT.value,
            "outcome": (
                ClaimOutcome.POSITIVE.value
                if learned_bits < frequent_bits
                else ClaimOutcome.NULL_RESULT.value
            ),
            "statement": (
                "The locked learned vocabulary did not beat the equal-budget "
                "frequent vocabulary on TEST_IID."
            ),
            "metric": "dictionary-inclusive total_mdl_bits (lower is better)",
            "split": "test_iid",
            "subset": "all",
            "exact_denominator_count": denominator,
            "subject_value": f"{learned_bits} bits",
            "baseline_value": f"{frequent_bits} bits",
            "source_artifacts": [
                "issue_5_6_experiment_result",
                "issue_5_6_heldout_results",
            ],
        },
        {
            "claim_id": ClaimId.LEARNED_VS_RANDOM.value,
            "outcome": (
                ClaimOutcome.POSITIVE.value
                if learned_bits < random_median
                else ClaimOutcome.NULL_RESULT.value
            ),
            "statement": (
                "The locked learned vocabulary beat the median of 30 fixed "
                "equal-budget random vocabularies on TEST_IID."
            ),
            "metric": "dictionary-inclusive total_mdl_bits (lower is better)",
            "split": "test_iid",
            "subset": "all",
            "exact_denominator_count": denominator,
            "subject_value": f"{learned_bits} bits",
            "baseline_value": f"{random_median:.1f} median bits",
            "source_artifacts": ["issue_5_6_heldout_results"],
        },
        {
            "claim_id": ClaimId.LEARNED_VS_MACRO.value,
            "outcome": (
                ClaimOutcome.POSITIVE.value
                if learned_bits < macro_bits
                else ClaimOutcome.NULL_RESULT.value
            ),
            "statement": (
                "The locked learned vocabulary beat the uncompressed macro baseline on TEST_IID."
            ),
            "metric": "geml-motif-mdl-v1 code length (lower is better)",
            "split": "test_iid",
            "subset": "all",
            "exact_denominator_count": denominator,
            "subject_value": f"{learned_bits} bits",
            "baseline_value": f"{macro_bits} bits",
            "source_artifacts": ["issue_5_6_heldout_results"],
        },
        {
            "claim_id": ClaimId.NEURAL_VS_HEURISTICS.value,
            "outcome": ranker_outcome.value,
            "statement": (
                "The neural ranker did not beat every structural heuristic on "
                "TEST_IID under the frozen validation-rate, mean-regret, then "
                "exact-best comparison order."
            ),
            "metric": "official-v4 candidate ranking over all groups",
            "split": "test_iid",
            "subset": "all",
            "exact_denominator_count": _required_nonnegative_int(
                iid_ranker.get("total_group_count"),
                label="issue 5-7 TEST_IID total_group_count",
            ),
            "subject_value": (
                f"{neural.get('exact_best_match_count')}/"
                f"{iid_ranker.get('total_group_count')} exact-best, "
                f"{neural.get('total_regret')} total regret"
            ),
            "baseline_value": "; ".join(baseline_parts),
            "source_artifacts": ["issue_5_7_report"],
        },
    ]


def _production_reproduction_commands(
    payloads: Mapping[ProducerArtifactKind, Mapping[str, object]],
    *,
    neural_ranker_command: str,
) -> tuple[str, ...]:
    """Return the integration command followed by every Goal 5 producer command."""

    producer_commands = (
        (
            ProducerArtifactKind.FREQUENT_MOTIFS,
            payloads[ProducerArtifactKind.FREQUENT_MOTIFS].get("reproduction_command"),
        ),
        (
            ProducerArtifactKind.LEARNED_MOTIFS,
            payloads[ProducerArtifactKind.LEARNED_MOTIFS].get("reproduction_command"),
        ),
        (ProducerArtifactKind.NEURAL_RANKER, neural_ranker_command),
        (
            ProducerArtifactKind.PRODUCTION_EXPORT,
            payloads[ProducerArtifactKind.PRODUCTION_EXPORT].get("reproduction_command"),
        ),
    )
    return (
        (
            "python -m geml.experiments.goal5.run "
            "--build-production-evidence "
            "outputs/final/goal5/integration/production.evidence.json "
            "--output-root outputs/final/goal5/integration "
            "--repository-root ."
        ),
        *(
            _required_string(
                command,
                field=f"{kind.value} reproduction_command",
            )
            for kind, command in producer_commands
        ),
    )


def build_production_evidence(
    *,
    repository_root: str | Path,
    evidence_path: str | Path | None = None,
) -> tuple[Goal5IntegrationEvidence, bytes]:
    """Authenticate production Goals 1-5 artifacts and normalize issue 5-9 evidence."""

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise Goal5IntegrationError(f"repository root does not exist: {root}")
    loaded: dict[ProducerArtifactKind, LoadedProducerArtifact] = {}
    print("[goal5-integration] authenticating standard producer completions", flush=True)
    for spec in STANDARD_PRODUCER_ARTIFACTS:
        if spec.kind is ProducerArtifactKind.GOAL4_ROWS:
            continue
        loaded[spec.kind] = read_standard_producer_artifact(root, spec.kind)
    payloads = {
        kind: _completion_payload(
            artifact.path,
            label=f"{kind.value} producer artifact",
        )
        for kind, artifact in loaded.items()
    }

    goal4_run = payloads[ProducerArtifactKind.GOAL4_RUN]
    goal4_rows_spec = _PRODUCER_SPEC_BY_KIND[ProducerArtifactKind.GOAL4_ROWS]
    goal4_rows_path = _producer_path(
        root,
        goal4_rows_spec,
        override=None,
    )
    print("[goal5-integration] streaming Goal 4 rows once", flush=True)
    goal4_scan = _scan_goal4_rows(
        goal4_rows_path,
        run_manifest=goal4_run,
    )
    goal4_rows_source = _repository_source(
        name=ProducerArtifactKind.GOAL4_ROWS.value,
        path=goal4_rows_path,
        repository_root=root,
        sha256=goal4_scan.cohorts.row_sha256,
        size_bytes=goal4_rows_path.stat().st_size,
        media_type=goal4_rows_spec.media_type,
        schema_version=goal4_rows_spec.schema_version,
    )
    goal4_counts = _aggregate_goal4_graph_rows(goal4_scan)

    frequent = loaded[ProducerArtifactKind.FREQUENT_MOTIFS]
    learned = loaded[ProducerArtifactKind.LEARNED_MOTIFS]
    (
        macro_mdl,
        frequent_mdl,
        learned_mdl,
        mdl_sources,
        learned_results,
    ) = _collect_mdl_evidence(
        frequent_completion=payloads[ProducerArtifactKind.FREQUENT_MOTIFS],
        frequent_run_dir=frequent.path.parent,
        learned_completion=payloads[ProducerArtifactKind.LEARNED_MOTIFS],
        learned_run_dir=learned.path.parent,
        repository_root=root,
    )

    print("[goal5-integration] streaming issue 5-7 groups and outcomes once", flush=True)
    neural_payload, neural_sources, neural_reproduction_command = _scan_neural_ranker(
        repository_root=root,
        completion=payloads[ProducerArtifactKind.NEURAL_RANKER],
        run_dir=loaded[ProducerArtifactKind.NEURAL_RANKER].path.parent,
        cohorts=goal4_scan.cohorts,
    )
    print("[goal5-integration] streaming issue 5-8 export shards once", flush=True)
    export_counts, all_ids_sha256, production_export = _scan_production_export(
        repository_root=root,
        completion=payloads[ProducerArtifactKind.PRODUCTION_EXPORT],
        completion_path=loaded[ProducerArtifactKind.PRODUCTION_EXPORT].path,
        cohorts=goal4_scan.cohorts,
    )
    graph_tracks = _graph_tracks_payload(
        goal4_counts=goal4_counts,
        export_counts=export_counts,
        macro_mdl=macro_mdl,
        frequent_mdl=frequent_mdl,
        learned_mdl=learned_mdl,
    )
    export_representations = {
        representation.get("name"): representation
        for representation in (
            _required_mapping(item, label="issue 5-8 representation")
            for item in _required_list(
                payloads[ProducerArtifactKind.PRODUCTION_EXPORT].get("representations"),
                label="issue 5-8 representations",
            )
        )
    }
    for track in graph_tracks:
        representation = export_representations.get(track["name"])
        if representation is not None:
            track["representation_family"] = representation.get("representation_family")
            track["representation_mode"] = representation.get("representation_mode")

    sources = {
        artifact.spec.kind.value: artifact.as_source_artifact(root) for artifact in loaded.values()
    }
    sources[goal4_rows_source.name] = goal4_rows_source
    sources.update(mdl_sources)
    sources.update(neural_sources)
    cohorts = goal4_scan.cohorts
    safe_by_split = dict(cohorts.safe_by_split)
    domain_by_split = dict(cohorts.domain_by_split)
    cohort_joins = []
    for split in SplitName:
        for subset, expression_ids in (
            ("safe_nontrivial", safe_by_split[split.value]),
            ("domain_nontrivial", domain_by_split[split.value]),
        ):
            cohort_joins.append(
                {
                    "split": split.value,
                    "subset": subset,
                    "expression_count": len(expression_ids),
                    "expression_ids_sha256": _cohort_digest(expression_ids),
                    "track_names": [track.value for track in GraphTrackName],
                    "source_artifacts": [
                        "goal4_rows",
                        "issue_5_8_production_export",
                    ],
                }
            )

    neural_report = _required_mapping(
        _completion_payload(
            loaded[ProducerArtifactKind.NEURAL_RANKER].path.parent / "report.json",
            label="issue 5-7 report",
        ),
        label="issue 5-7 report",
    )
    claims = _learned_claims(learned_results, neural_report)
    goal1 = payloads[ProducerArtifactKind.GOAL1_CORPUS]
    goal2 = payloads[ProducerArtifactKind.GOAL2_FINAL]
    goal3 = payloads[ProducerArtifactKind.GOAL3_FINAL]
    goal5_sources = [
        ProducerArtifactKind.FREQUENT_MOTIFS.value,
        ProducerArtifactKind.LEARNED_MOTIFS.value,
        ProducerArtifactKind.NEURAL_RANKER.value,
        ProducerArtifactKind.PRODUCTION_EXPORT.value,
    ]
    evidence_payload = {
        "schema_version": "geml-goal5-integration-evidence-v1",
        "status": EvidenceStatus.COMPLETE.value,
        "dataset_id": payloads[ProducerArtifactKind.PRODUCTION_EXPORT].get("dataset_id"),
        "goal6_export": {},
        "source_artifacts": [sources[name].model_dump(mode="json") for name in sorted(sources)],
        "subset_definitions": [
            {
                "name": "all",
                "definition": (
                    "Every expression in the authenticated issue 5-8 full-corpus export; "
                    "track-specific all-processed denominators remain explicit."
                ),
                "is_nontrivial": False,
                "expression_ids_sha256": all_ids_sha256,
                "rewrite_mode": "all",
                "semantics": "all authenticated production corpus expressions",
                "source_artifacts": ["issue_5_8_production_export"],
            },
            {
                "name": "safe_nontrivial",
                "definition": (
                    "Exact Goal 4 expression IDs whose safe_real row records rewrites_applied > 0."
                ),
                "is_nontrivial": True,
                "expression_ids_sha256": cohorts.safe_sha256,
                "rewrite_mode": "safe_real",
                "semantics": cohorts.safe_semantics,
                "source_artifacts": ["goal4_rows"],
            },
            {
                "name": "domain_nontrivial",
                "definition": (
                    "Exact Goal 4 expression IDs whose positive_real_formal row "
                    "records rewrites_applied > 0."
                ),
                "is_nontrivial": True,
                "expression_ids_sha256": cohorts.domain_sha256,
                "rewrite_mode": "positive_real_formal",
                "semantics": cohorts.domain_semantics,
                "source_artifacts": ["goal4_rows"],
            },
        ],
        "cohort_joins": cohort_joins,
        "production_export": production_export,
        "graph_tracks": graph_tracks,
        "neural_ranker": neural_payload,
        "claims": claims,
        "goal_statuses": [
            {
                "goal_number": 1,
                "status": "complete",
                "summary": (
                    f"Deterministic corpus complete: {goal1.get('total_row_count')} "
                    f"rows and {goal1.get('total_error_row_count')} error rows."
                ),
                "source_artifacts": ["goal1_corpus"],
            },
            {
                "goal_number": 2,
                "status": "complete",
                "summary": (
                    f"Official-v4 EML compilation complete for "
                    f"{goal2.get('processed_count')} inputs with "
                    f"{goal2.get('failure_count')} retained failures."
                ),
                "source_artifacts": ["goal2_final"],
            },
            {
                "goal_number": 3,
                "status": "complete",
                "summary": (
                    f"Graph construction/audit complete for "
                    f"{goal3.get('processed_count')} inputs with "
                    f"{goal3.get('failure_count')} retained failures."
                ),
                "source_artifacts": ["goal3_final"],
            },
            {
                "goal_number": 4,
                "status": "complete",
                "summary": (
                    f"Both frozen rewrite modes complete for "
                    f"{goal4_run.get('selected_expression_count')} selected "
                    f"expressions ({goal4_scan.record_count} retained rows)."
                ),
                "source_artifacts": ["goal4_rows", "goal4_run"],
            },
            {
                "goal_number": 5,
                "status": "complete",
                "summary": (
                    "Frequent motifs, learned motifs, neural ranker, and the "
                    "250,000-expression five-representation export are complete; "
                    "issue 5-9 integrates their authenticated results."
                ),
                "source_artifacts": goal5_sources,
            },
        ],
        "reproduction_commands": _production_reproduction_commands(
            payloads,
            neural_ranker_command=neural_reproduction_command,
        ),
        "missing_requirements": [],
    }
    try:
        evidence = _validate_normalized_evidence_payload(evidence_payload)
    except Exception as error:
        raise Goal5IntegrationError(
            "normalized production evidence violates the issue 5-9 schema"
        ) from error
    data = canonical_json_bytes(evidence.model_dump(mode="json"))
    if evidence_path is not None:
        target = Path(evidence_path)
        if not target.is_absolute():
            target = root / target
        target = target.resolve()
        if not target.is_relative_to(root):
            raise Goal5IntegrationError("production evidence path escapes repository root")
        _write_atomic(data, target)
    return evidence, data


def load_integration_evidence(
    path: str | Path,
    *,
    repository_root: str | Path,
    require_complete: bool = True,
) -> tuple[Goal5IntegrationEvidence, bytes]:
    """Load strict evidence and authenticate every artifact it cites."""

    source = Path(path).resolve()
    try:
        data = source.read_bytes()
    except OSError as error:
        raise Goal5IntegrationError(f"cannot read integration evidence: {source}") from error
    _decode_json_object(data, label="integration evidence")
    try:
        evidence = Goal5IntegrationEvidence.model_validate_json(data)
    except Exception as error:
        raise Goal5IntegrationError("integration evidence violates its frozen schema") from error
    if require_complete and evidence.status is not EvidenceStatus.COMPLETE:
        raise Goal5IntegrationError(
            "final integration requires complete evidence; missing: "
            + "; ".join(evidence.missing_requirements)
        )
    if require_complete:
        _require_atomic_prerequisite_sources(evidence)
        _require_standard_producer_sources(evidence)
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise Goal5IntegrationError(f"repository root does not exist: {repository}")
    goal4_rows_paths = (
        frozenset(
            artifact.path
            for artifact in _matching_producer_sources(
                evidence,
                ProducerArtifactKind.GOAL4_ROWS,
            )
        )
        if require_complete
        else frozenset()
    )
    _verify_source_artifacts(
        evidence,
        repository_root=repository,
        skip_paths=goal4_rows_paths,
    )
    if require_complete:
        _verify_goal4_cohort_bindings(evidence, repository_root=repository)
    return evidence, data


def _implementation_digest() -> str:
    """Bind integration outputs to all three issue-owned Python modules."""

    module_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "analysis" / "goal5" / "summary.py",
        Path(__file__).resolve().parents[2] / "plots" / "goal5.py",
    )
    digest = hashlib.sha256(b"geml-goal5-integration-implementation-v1\0")
    for path in module_paths:
        data = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def _run_digest(evidence_sha256: str, implementation_sha256: str) -> str:
    payload = {
        "evidence_sha256": evidence_sha256,
        "implementation_sha256": implementation_sha256,
    }
    digest = hashlib.sha256(b"geml-goal5-integration-run-v1\0")
    digest.update(canonical_json_bytes(payload))
    return digest.hexdigest()


def _write_atomic(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            _write_and_sync(stream, data)
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _write_and_sync(stream: BinaryIO, data: bytes) -> None:
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())


def _write_immutable(data: bytes, path: Path) -> None:
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise Goal5IntegrationError(f"cannot authenticate existing output: {path}") from error
        if existing != data:
            raise Goal5IntegrationError(f"immutable integration output differs: {path}")
        return
    _write_atomic(data, path)


def _output_descriptor(path: Path, *, run_dir: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": _sha256_bytes(data),
        "size_bytes": len(data),
    }


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise Goal5IntegrationError(f"{field} must be a nonblank string")
    return value


def _required_sha256(value: object, *, field: str) -> str:
    text = _required_string(value, field=field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise Goal5IntegrationError(f"{field} must be a lowercase SHA-256")
    return text


def load_completed_integration_run(
    completion_path: str | Path,
) -> IntegrationRunResult:
    """Authenticate a completed issue 5-9 bundle without producer outputs."""

    path = Path(completion_path).resolve()
    try:
        data = path.read_bytes()
    except OSError as error:
        raise Goal5IntegrationError(f"cannot read integration completion: {path}") from error
    completion = _decode_json_object(data, label="integration completion")
    expected_keys = {
        "schema_version",
        "status",
        "dataset_id",
        "evidence_sha256",
        "implementation_sha256",
        "run_digest",
        "goal6_export",
        "outputs",
        "reproduction_command",
    }
    if set(completion) != expected_keys:
        raise Goal5IntegrationError("integration completion fields do not match its schema")
    if completion["schema_version"] != INTEGRATION_RUN_COMPLETE_SCHEMA_VERSION:
        raise Goal5IntegrationError("integration completion has an incompatible schema")
    if completion["status"] not in {status.value for status in EvidenceStatus}:
        raise Goal5IntegrationError("integration completion has an invalid status")
    _required_string(completion["dataset_id"], field="dataset_id")
    evidence_sha256 = _required_sha256(
        completion["evidence_sha256"],
        field="evidence_sha256",
    )
    implementation_sha256 = _required_sha256(
        completion["implementation_sha256"],
        field="implementation_sha256",
    )
    run_digest = _required_sha256(completion["run_digest"], field="run_digest")
    if run_digest != _run_digest(evidence_sha256, implementation_sha256):
        raise Goal5IntegrationError("integration completion run digest is inconsistent")
    run_dir = path.parent
    if run_dir.name != f"run-{run_digest}":
        raise Goal5IntegrationError("integration run directory is not content addressed")
    frozen_export = FrozenGoal6Export().model_dump(mode="json")
    if completion["goal6_export"] != frozen_export:
        raise Goal5IntegrationError("integration completion changed the Goal 6 freeze")
    _required_string(completion["reproduction_command"], field="reproduction_command")

    raw_outputs = completion["outputs"]
    if not isinstance(raw_outputs, dict) or set(raw_outputs) != _EXPECTED_OUTPUT_PATHS:
        raise Goal5IntegrationError("integration completion output set is incomplete")
    retained_evidence_data: bytes | None = None
    for relative_path, raw_descriptor in raw_outputs.items():
        if not isinstance(relative_path, str) or not isinstance(raw_descriptor, dict):
            raise Goal5IntegrationError("integration output descriptors must be JSON objects")
        if set(raw_descriptor) != {"path", "sha256", "size_bytes"}:
            raise Goal5IntegrationError("integration output descriptor fields are invalid")
        if raw_descriptor["path"] != relative_path:
            raise Goal5IntegrationError("integration output descriptor path is inconsistent")
        output_path = _path_within(run_dir, relative_path)
        try:
            output_data = output_path.read_bytes()
        except OSError as error:
            raise Goal5IntegrationError(
                f"cannot read completed integration output: {relative_path}"
            ) from error
        size = raw_descriptor["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise Goal5IntegrationError("integration output size must be nonnegative")
        if len(output_data) != size:
            raise Goal5IntegrationError(f"integration output size mismatch: {relative_path}")
        digest = _required_sha256(
            raw_descriptor["sha256"],
            field=f"outputs[{relative_path!r}].sha256",
        )
        if _sha256_bytes(output_data) != digest:
            raise Goal5IntegrationError(f"integration output SHA-256 mismatch: {relative_path}")
        if relative_path == "integration.evidence.json":
            retained_evidence_data = output_data
    evidence_descriptor = raw_outputs["integration.evidence.json"]
    if evidence_descriptor["sha256"] != evidence_sha256:
        raise Goal5IntegrationError("completion and retained evidence SHA-256 disagree")
    if retained_evidence_data is None:  # pragma: no cover - output-set check guarantees it
        raise Goal5IntegrationError("completed integration evidence is missing")
    _decode_json_object(retained_evidence_data, label="retained integration evidence")
    try:
        retained_evidence = Goal5IntegrationEvidence.model_validate_json(retained_evidence_data)
    except Exception as error:
        raise Goal5IntegrationError(
            "retained integration evidence violates its strict schema"
        ) from error
    semantic_bindings = {
        "status": (retained_evidence.status.value, completion["status"]),
        "dataset_id": (retained_evidence.dataset_id, completion["dataset_id"]),
        "goal6_export": (
            retained_evidence.goal6_export.model_dump(mode="json"),
            completion["goal6_export"],
        ),
    }
    for field_name, (evidence_value, completion_value) in semantic_bindings.items():
        if evidence_value != completion_value:
            raise Goal5IntegrationError(
                f"integration completion {field_name} disagrees with retained evidence"
            )
    return IntegrationRunResult(
        run_dir=run_dir,
        completion_path=path,
        evidence_sha256=evidence_sha256,
        implementation_sha256=implementation_sha256,
        run_digest=run_digest,
    )


def _render_plot_files(plot_data, *, run_dir: Path) -> tuple[Path, ...]:
    with tempfile.TemporaryDirectory(prefix="geml-goal5-plots-") as temporary:
        rendered = render_plots(plot_data, temporary)
        final_paths: list[Path] = []
        for source in rendered:
            target = run_dir / "plots" / source.name
            _write_immutable(source.read_bytes(), target)
            final_paths.append(target)
    return tuple(final_paths)


def run_integration(
    evidence_path: str | Path,
    *,
    output_root: str | Path,
    repository_root: str | Path,
    require_complete: bool = True,
) -> IntegrationRunResult:
    """Authenticate inputs and emit one content-addressed report bundle."""

    evidence, evidence_data = load_integration_evidence(
        evidence_path,
        repository_root=repository_root,
        require_complete=require_complete,
    )
    reproduction_command = (
        "python -m geml.experiments.goal5.run "
        f"--evidence {Path(evidence_path).as_posix()} "
        f"--output-root {Path(output_root).as_posix()} "
        f"--repository-root {Path(repository_root).as_posix()}"
    )
    if not require_complete:
        reproduction_command += " --allow-incomplete"
    return _emit_integration(
        evidence,
        evidence_data,
        output_root=output_root,
        reproduction_command=reproduction_command,
    )


def _emit_integration(
    evidence: Goal5IntegrationEvidence,
    evidence_data: bytes,
    *,
    output_root: str | Path,
    reproduction_command: str,
) -> IntegrationRunResult:
    """Emit reports from evidence already authenticated in the current process."""

    evidence_sha256 = _sha256_bytes(evidence_data)
    implementation_sha256 = _implementation_digest()
    run_digest = _run_digest(evidence_sha256, implementation_sha256)
    run_dir = Path(output_root).resolve() / f"run-{run_digest}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(evidence)
    plot_data = build_plot_data(summary)
    output_payloads = {
        "integration.evidence.json": evidence_data,
        "goal5.summary.json": canonical_json_bytes(summary.as_dict()),
        "goal5.plot-data.json": canonical_json_bytes(plot_data.as_dict()),
        "GOAL5_SUMMARY.md": render_goal5_summary_markdown(summary).encode("utf-8"),
        "GOAL5_COMPRESSION_STUDY.md": render_compression_study_markdown(summary).encode("utf-8"),
        "FINAL_GOALS_1_TO_5_STATUS.md": render_goals_1_to_5_status_markdown(summary).encode(
            "utf-8"
        ),
    }
    output_paths: list[Path] = []
    for relative_path, data in output_payloads.items():
        target = run_dir / relative_path
        _write_immutable(data, target)
        output_paths.append(target)
    output_paths.extend(_render_plot_files(plot_data, run_dir=run_dir))

    output_descriptors = {
        descriptor["path"]: descriptor
        for descriptor in (_output_descriptor(path, run_dir=run_dir) for path in output_paths)
    }
    completion = {
        "schema_version": INTEGRATION_RUN_COMPLETE_SCHEMA_VERSION,
        "status": evidence.status.value,
        "dataset_id": evidence.dataset_id,
        "evidence_sha256": evidence_sha256,
        "implementation_sha256": implementation_sha256,
        "run_digest": run_digest,
        "goal6_export": evidence.goal6_export.model_dump(mode="json"),
        "outputs": dict(sorted(output_descriptors.items())),
        "reproduction_command": reproduction_command,
    }
    completion_path = run_dir / "run.complete.json"
    _write_immutable(canonical_json_bytes(completion), completion_path)
    return load_completed_integration_run(completion_path)


def run_production_integration(
    *,
    evidence_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
) -> IntegrationRunResult:
    """Build normalized production evidence once and emit the final report bundle."""

    repository = Path(repository_root).resolve()
    evidence_target = Path(evidence_path)
    if not evidence_target.is_absolute():
        evidence_target = repository / evidence_target
    output_target = Path(output_root)
    if not output_target.is_absolute():
        output_target = repository / output_target
    evidence, evidence_data = build_production_evidence(
        repository_root=repository,
        evidence_path=evidence_target,
    )
    reproduction_command = (
        "python -m geml.experiments.goal5.run "
        f"--build-production-evidence {Path(evidence_path).as_posix()} "
        f"--output-root {Path(output_root).as_posix()} "
        f"--repository-root {Path(repository_root).as_posix()}"
    )
    result = _emit_integration(
        evidence,
        evidence_data,
        output_root=output_target,
        reproduction_command=reproduction_command,
    )
    return load_completed_integration_run(result.completion_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate Goal 5 evidence and generate final comparison reports."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence")
    source.add_argument(
        "--build-production-evidence",
        metavar="PATH",
        help=(
            "One-pass authenticate the standard production artifacts, write normalized "
            "evidence to PATH, and emit the final report."
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Generate an explicitly incomplete scaffold; never publish it as a final result.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    arguments = _parser().parse_args(argv)
    if arguments.build_production_evidence is not None:
        if arguments.allow_incomplete:
            raise Goal5IntegrationError(
                "--allow-incomplete cannot be combined with production evidence building"
            )
        result = run_production_integration(
            evidence_path=arguments.build_production_evidence,
            output_root=arguments.output_root,
            repository_root=arguments.repository_root,
        )
    else:
        result = run_integration(
            arguments.evidence,
            output_root=arguments.output_root,
            repository_root=arguments.repository_root,
            require_complete=not arguments.allow_incomplete,
        )
    print(result.completion_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
