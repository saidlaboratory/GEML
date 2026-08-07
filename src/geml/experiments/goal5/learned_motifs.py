"""Leakage-safe learned motif selection with locked equal-budget baselines.

The learned scorer is fitted only from exact singleton-vocabulary MDL gains on
a deterministic subset of TRAIN.  Ridge hyperparameters are selected only by
VALIDATION total MDL.  The selected scorer, vocabulary, frequent comparator,
and thirty random comparator vocabularies are locked and persisted before a
held-out split loader is called.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import math
import os
import shlex
import statistics
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import chain
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Literal

import numpy as np
import pyarrow.parquet as pq
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    field_validator,
)

from geml.compression.motif.boundary import (
    MotifOccurrence,
    find_vocabulary_occurrences,
    graph_structure_fingerprint,
)
from geml.compression.motif.mdl import (
    MotifGraphMDLResult,
    PreparedGraphMDL,
    SplitMDLSummary,
    fallback_mdl_result,
    motif_graph_mdl_result,
    prepare_graph_mdl,
    vocabulary_mdl_bits,
)
from geml.compression.motif.vocabulary import (
    MotifTemplate,
    MotifVocabulary,
    build_motif_vocabulary,
    motif_rank_key,
)
from geml.contracts.corpus import CorpusSplit
from geml.experiments.goal5 import motif_sweeps as frequent_sweeps
from geml.experiments.goal5.motif_sweeps import (
    SWEEP_ARTIFACT_VERSION,
    GraphCacheDescriptor,
    iter_cached_graphs,
    load_completed_graph_cache,
    sweep_implementation_digest,
    vocabulary_from_payload,
    vocabulary_payload,
    vocabulary_payload_digest,
)
from geml.graph.schema import MACRO_FAMILY, Graph
from geml.graph.validate import validate_graph
from geml.learning.motif_selector import (
    SELECTOR_VERSION,
    RidgeMotifSelector,
    SelectorTrainingExample,
    candidate_pool_digest,
    fit_ridge_selector,
    select_frequent_templates,
    select_learned_templates,
    select_random_templates,
    selector_from_payload,
    selector_payload,
    selector_payload_digest,
    training_target_digest,
)

LEARNED_CONFIG_VERSION = "geml-goal5-learned-motifs-config-v1"
LEARNED_ARTIFACT_VERSION = "geml-goal5-learned-motifs-v1"
LEARNED_LOCK_VERSION = "geml-goal5-learned-selection-lock-v1"
LEARNED_RESULT_VERSION = "geml-goal5-learned-results-v1"
LEARNED_COMPLETE_VERSION = "geml-goal5-learned-run-complete-v1"
LEARNED_HELDOUT_SPLIT_VERSION = "geml-goal5-learned-heldout-split-v1"
FREQUENT_LOCK_VERSION = "geml-goal5-frequent-selection-lock-v1"
FREQUENT_COMPLETE_VERSION = "geml-goal5-frequent-run-complete-v1"
PARTITION_VERSION = "geml-train-partition-sha256-v1"
SPLIT_IDENTITY_VERSION = "geml-split-identity-sha256-v1"
TARGET_DEFINITION = "exact-singleton-two-part-mdl-gain-v1"
RANDOM_BASELINE_VERSION = "geml-motif-random-baseline-v1"

_LOGGER = logging.getLogger(__name__)

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
_PositiveInt = Annotated[StrictInt, Field(gt=0)]
_Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


class LearnedMotifConfigurationError(ValueError):
    """The experiment config or its frozen 5-5 inputs are invalid."""


class LearnedMotifProtocolError(ValueError):
    """A scientific boundary would be crossed or required evidence is absent."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )


class SelectorExperimentConfig(_FrozenModel):
    """Train-only target and validation-only ridge-selection policy."""

    method: Literal["geml-deterministic-ridge-mdl-v1"] = SELECTOR_VERSION
    target_definition: Literal["exact-singleton-two-part-mdl-gain-v1"] = TARGET_DEFINITION
    train_fit_fraction: Annotated[StrictFloat, Field(gt=0.0, le=1.0)]
    partition_method: Literal["geml-train-partition-sha256-v1"] = PARTITION_VERSION
    maximum_target_graphs: _PositiveInt
    candidate_prefilter: Literal["train-frequency-rank-v1"]
    maximum_candidate_motifs: Annotated[StrictInt, Field(ge=1024)]
    ridge_lambdas: tuple[Annotated[StrictFloat, Field(ge=0.0)], ...] = Field(min_length=1)
    singular_value_rcond: Annotated[StrictFloat, Field(gt=0.0, lt=1.0)]

    @field_validator("ridge_lambdas", mode="before")
    @classmethod
    def normalize_yaml_ridge_lambdas(cls, values: object) -> object:
        """Normalize YAML sequences while keeping scalar validation strict."""

        return tuple(values) if isinstance(values, list) else values

    @field_validator("ridge_lambdas")
    @classmethod
    def validate_ridge_lambdas(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        """Require a stable, nonduplicated search order."""

        if tuple(sorted(values)) != values or len(set(values)) != len(values):
            raise ValueError("ridge_lambdas must be unique and increasing")
        return values


class BaselineExperimentConfig(_FrozenModel):
    """Equal-budget frequent, random, and uncompressed-macro controls."""

    frequent: Literal[True] = True
    random: Literal[True] = True
    macro_without_motifs: Literal[True] = True
    equal_motif_budget: Literal[True] = True
    random_seed: _NonNegativeInt
    random_repetitions: Literal[30] = 30
    random_method: Literal["geml-motif-random-baseline-v1"] = RANDOM_BASELINE_VERSION


class EvaluationExperimentConfig(_FrozenModel):
    """Frozen lossless evaluation and post-hoc claim policy."""

    codec: Literal["geml-motif-mdl-v1"]
    occurrence_policy: Literal["deterministic_safe_greedy_v1"]
    require_exact_reconstruction: Literal[True]
    failure_fallback: Literal["encode_original_graph"]
    validation_selects_hyperparameters: Literal[True]
    evaluate_tests_once_after_lock: Literal[True]
    posthoc_claim_rule: Literal[
        "report_only_if_learned_beats_frequent_random_median_and_macro_on_test_iid"
    ]
    report_ood_independently: Literal[True]


class RuntimeExperimentConfig(_FrozenModel):
    """Deterministic persistence controls."""

    seed: _NonNegativeInt
    resume: StrictBool
    atomic_finalization: Literal[True]


class Goal5LearnedMotifConfig(_FrozenModel):
    """Complete issue 5-6 production protocol."""

    schema_version: Literal["geml-goal5-learned-motifs-config-v1"] = LEARNED_CONFIG_VERSION
    frequent_sweep_run_dir: _NonBlankStr | None = None
    output_root: _NonBlankStr
    graph_family: Literal["macro"] = MACRO_FAMILY
    representation_mode: Literal["macro:official_v4:is_pure_eml=false"]
    selector: SelectorExperimentConfig
    baselines: BaselineExperimentConfig
    evaluation: EvaluationExperimentConfig
    runtime: RuntimeExperimentConfig


@dataclass(frozen=True, slots=True)
class LoadedLearnedMotifConfig:
    """Validated config plus containment-checked paths."""

    config: Goal5LearnedMotifConfig
    repository_root: Path
    config_path: Path
    frequent_sweep_run_dir: Path | None
    output_root: Path
    config_digest: str


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LearnedMotifProtocolError(f"value is not strict JSON: {error}") from error


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def learned_config_digest(config: Goal5LearnedMotifConfig) -> str:
    """Return the canonical digest of one fully validated 5-6 configuration."""

    if not isinstance(config, Goal5LearnedMotifConfig):
        raise TypeError("config must be a Goal5LearnedMotifConfig")
    return _sha256_bytes(_canonical_json_bytes(config.model_dump(mode="json")))


def learned_implementation_digest(repository_root: str | Path) -> str:
    """Fingerprint the 5-5 closure plus the exact issue 5-6 implementation."""

    root = Path(repository_root).resolve()
    paths = (
        root / "src" / "geml" / "learning" / "motif_selector.py",
        Path(__file__).resolve(),
    )
    if any(not path.is_file() for path in paths):
        raise LearnedMotifConfigurationError(
            "learned-motif implementation source files are missing"
        )
    digest = hashlib.sha256(b"geml-goal5-learned-implementation-v1\0")
    digest.update(sweep_implementation_digest(root).encode("ascii"))
    numpy_config = np.show_config(mode="dicts")
    build_dependencies = numpy_config.get("Build Dependencies", {})
    simd_extensions = numpy_config.get("SIMD Extensions", {})
    numeric_runtime = _canonical_json_bytes(
        {
            "blas": build_dependencies.get("blas", {}),
            "lapack": build_dependencies.get("lapack", {}),
            "numpy": np.__version__,
            "simd": simd_extensions,
        }
    )
    digest.update(len(numeric_runtime).to_bytes(8, "big"))
    digest.update(numeric_runtime)
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "geml").is_dir():
            return candidate.resolve()
    raise LearnedMotifConfigurationError("could not locate the GEML repository root")


def _resolve_inside(root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise LearnedMotifConfigurationError(
            f"{label} must remain inside the repository"
        ) from error
    return path


def load_learned_motif_config(
    path: str | Path,
    *,
    frequent_sweep_run_dir: str | Path | None = None,
    require_inputs: bool = True,
) -> LoadedLearnedMotifConfig:
    """Load strict YAML without requiring production artifacts in fixture tests."""

    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise LearnedMotifConfigurationError(
            f"could not read learned-motif config: {config_path}"
        ) from error
    if not isinstance(raw, dict):
        raise LearnedMotifConfigurationError("learned-motif config must be a YAML mapping")
    try:
        config = Goal5LearnedMotifConfig.model_validate(raw)
    except Exception as error:
        raise LearnedMotifConfigurationError("invalid learned-motif configuration") from error

    root = _repository_root(config_path.parent)
    configured_run = (
        frequent_sweep_run_dir
        if frequent_sweep_run_dir is not None
        else config.frequent_sweep_run_dir
    )
    resolved_run = (
        None
        if configured_run is None
        else _resolve_inside(root, str(configured_run), label="frequent_sweep_run_dir")
    )
    if require_inputs and resolved_run is None:
        raise LearnedMotifConfigurationError(
            "a locked frequent_sweep_run_dir is required for execution"
        )
    if require_inputs and resolved_run is not None and not resolved_run.is_dir():
        raise LearnedMotifConfigurationError(
            f"frequent sweep run directory does not exist: {resolved_run}"
        )
    output_root = _resolve_inside(root, config.output_root, label="output_root")
    return LoadedLearnedMotifConfig(
        config=config,
        repository_root=root,
        config_path=config_path,
        frequent_sweep_run_dir=resolved_run,
        output_root=output_root,
        config_digest=learned_config_digest(config),
    )


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """One immutable path and SHA-256 pair relative to an artifact root."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("artifact path must be nonblank")
        normalized = PurePosixPath(self.path)
        if (
            "\\" in self.path
            or normalized.is_absolute()
            or not normalized.parts
            or normalized.as_posix() != self.path
            or any(part in {"", ".", ".."} for part in normalized.parts)
            or (normalized.parts and ":" in normalized.parts[0])
        ):
            raise ValueError("artifact path must be a canonical relative POSIX path")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("artifact sha256 must be lowercase hexadecimal")


def _artifact_ref(payload: object, *, label: str) -> ArtifactReference:
    if not isinstance(payload, dict) or set(payload) != {"path", "sha256"}:
        raise LearnedMotifConfigurationError(f"{label} must contain exactly path and sha256")
    try:
        return ArtifactReference(
            path=payload["path"],
            sha256=payload["sha256"],
        )
    except (TypeError, ValueError) as error:
        raise LearnedMotifConfigurationError(f"invalid {label}") from error


def _artifact_path(root: Path, reference: ArtifactReference) -> Path:
    path = root / Path(reference.path)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise LearnedMotifConfigurationError(
            f"artifact path escapes run directory: {reference.path!r}"
        ) from error
    return path


def _read_artifact(root: Path, reference: ArtifactReference) -> bytes:
    path = _artifact_path(root, reference)
    if not path.is_file():
        raise LearnedMotifConfigurationError(f"missing artifact: {reference.path}")
    data = path.read_bytes()
    observed = _sha256_bytes(data)
    if observed != reference.sha256:
        raise LearnedMotifConfigurationError(
            f"artifact checksum mismatch for {reference.path}: "
            f"expected {reference.sha256}, observed {observed}"
        )
    return data


def _json_object(data: bytes, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise LearnedMotifConfigurationError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise LearnedMotifConfigurationError(f"{label} must contain a JSON object")
    if data != _canonical_json_bytes(payload) + b"\n":
        raise LearnedMotifConfigurationError(f"{label} is not canonical JSON")
    return payload


@dataclass(frozen=True, slots=True)
class FrequentSweepProvenance:
    """Immutable 5-5 artifact identity inherited by the learned run."""

    run_directory: str
    selection_lock_sha256: str
    run_complete_sha256: str
    config_digest: str
    input_manifest_sha256: str
    implementation_digest: str
    selected_configuration_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_directory, str)
            or not self.run_directory.strip()
            or "\\" in self.run_directory
            or Path(self.run_directory).is_absolute()
            or ".." in Path(self.run_directory).parts
        ):
            raise ValueError("run_directory must be a repository-relative POSIX path")
        for name, value in (
            ("selection_lock_sha256", self.selection_lock_sha256),
            ("run_complete_sha256", self.run_complete_sha256),
            ("config_digest", self.config_digest),
            ("input_manifest_sha256", self.input_manifest_sha256),
            ("implementation_digest", self.implementation_digest),
            (
                "selected_configuration_digest",
                self.selected_configuration_digest,
            ),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")


def _frequent_provenance_payload(
    provenance: FrequentSweepProvenance,
) -> dict[str, str]:
    return {
        "config_digest": provenance.config_digest,
        "implementation_digest": provenance.implementation_digest,
        "input_manifest_sha256": provenance.input_manifest_sha256,
        "run_directory": provenance.run_directory,
        "run_complete_sha256": provenance.run_complete_sha256,
        "selected_configuration_digest": (provenance.selected_configuration_digest),
        "selection_lock_sha256": provenance.selection_lock_sha256,
    }


@dataclass(frozen=True, slots=True)
class FrequentSweepInputs:
    """Authenticated, completed issue 5-5 inputs for learned selection."""

    run_dir: Path
    candidate_pool: MotifVocabulary
    selected_frequent: MotifVocabulary
    selection_lock_sha256: str
    run_complete_sha256: str
    config_digest: str
    input_manifest_sha256: str
    implementation_digest: str
    selected_configuration_digest: str
    graph_cache_references: Mapping[CorpusSplit, ArtifactReference]
    graph_cache_descriptors: Mapping[CorpusSplit, GraphCacheDescriptor]

    def __post_init__(self) -> None:
        for name, value in (
            ("selection_lock_sha256", self.selection_lock_sha256),
            ("run_complete_sha256", self.run_complete_sha256),
            ("config_digest", self.config_digest),
            ("input_manifest_sha256", self.input_manifest_sha256),
            ("implementation_digest", self.implementation_digest),
            (
                "selected_configuration_digest",
                self.selected_configuration_digest,
            ),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        if set(self.graph_cache_references) != {split for split in CorpusSplit}:
            raise ValueError("graph_cache_references must cover every split")
        if set(self.graph_cache_descriptors) != {split for split in CorpusSplit}:
            raise ValueError("graph_cache_descriptors must cover every split")
        if any(
            descriptor.split is not split
            for split, descriptor in self.graph_cache_descriptors.items()
        ):
            raise ValueError("graph_cache_descriptors contain a split mismatch")
        object.__setattr__(
            self,
            "graph_cache_references",
            MappingProxyType(dict(self.graph_cache_references)),
        )
        object.__setattr__(
            self,
            "graph_cache_descriptors",
            MappingProxyType(dict(self.graph_cache_descriptors)),
        )

    @property
    def provenance(self) -> FrequentSweepProvenance:
        repository_root = _repository_root(self.run_dir)
        return FrequentSweepProvenance(
            run_directory=self.run_dir.resolve().relative_to(repository_root).as_posix(),
            selection_lock_sha256=self.selection_lock_sha256,
            run_complete_sha256=self.run_complete_sha256,
            config_digest=self.config_digest,
            input_manifest_sha256=self.input_manifest_sha256,
            implementation_digest=self.implementation_digest,
            selected_configuration_digest=self.selected_configuration_digest,
        )


def _frequent_config_path_from_completion(
    run_dir: Path,
    completion: Mapping[str, object],
) -> Path:
    """Recover the checked-in 5-5 config named by its reproduction command."""

    command = completion.get("reproduction_command")
    if not isinstance(command, str) or not command.strip():
        raise LearnedMotifConfigurationError(
            "frequent completion reproduction_command must be nonblank"
        )
    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise LearnedMotifConfigurationError(
            "frequent completion reproduction_command is invalid"
        ) from error
    config_indexes = [index for index, token in enumerate(tokens) if token == "--config"]
    stage_indexes = [index for index, token in enumerate(tokens) if token == "--stage"]
    if (
        len(config_indexes) != 1
        or config_indexes[0] + 1 >= len(tokens)
        or len(stage_indexes) != 1
        or stage_indexes[0] + 1 >= len(tokens)
        or tokens[stage_indexes[0] + 1] != "final"
    ):
        raise LearnedMotifConfigurationError(
            "frequent completion must name exactly one final-stage config"
        )
    repository_root = _repository_root(run_dir)
    config_path = _resolve_inside(
        repository_root,
        tokens[config_indexes[0] + 1],
        label="frequent reproduction config",
    )
    if not config_path.is_file():
        raise LearnedMotifConfigurationError(
            f"frequent reproduction config does not exist: {config_path}"
        )
    return config_path


def load_frequent_sweep_inputs(run_dir: str | Path) -> FrequentSweepInputs:
    """Authenticate the frozen 5-5 completion and selection-lock chain."""

    root = Path(run_dir).resolve()
    complete_path = root / "run.complete.json"
    lock_path = root / "selection.lock.json"
    if not complete_path.is_file() or not lock_path.is_file():
        raise LearnedMotifConfigurationError(
            "frequent run requires run.complete.json and selection.lock.json"
        )
    complete_data = complete_path.read_bytes()
    lock_data = lock_path.read_bytes()
    complete = _json_object(complete_data, label="frequent run completion")
    lock = _json_object(lock_data, label="frequent selection lock")
    if complete.get("schema_version") != FREQUENT_COMPLETE_VERSION:
        raise LearnedMotifConfigurationError("unsupported frequent completion schema")
    if lock.get("schema_version") != FREQUENT_LOCK_VERSION:
        raise LearnedMotifConfigurationError("unsupported frequent selection lock schema")
    if complete.get("artifact_version") != SWEEP_ARTIFACT_VERSION:
        raise LearnedMotifConfigurationError("frequent completion artifact version mismatch")
    if lock.get("artifact_version") != SWEEP_ARTIFACT_VERSION:
        raise LearnedMotifConfigurationError("frequent lock artifact version mismatch")
    if lock.get("heldout_artifacts_absent_at_lock") is not True:
        raise LearnedMotifConfigurationError(
            "frequent lock does not prove held-out artifacts were absent"
        )
    if lock.get("stage") != "final" or complete.get("stage") != "final":
        raise LearnedMotifConfigurationError(
            "learned motif selection requires the completed final 5-5 run"
        )
    implementation_digest = lock.get("implementation_digest")
    if (
        not isinstance(implementation_digest, str)
        or len(implementation_digest) != 64
        or any(character not in "0123456789abcdef" for character in implementation_digest)
    ):
        raise LearnedMotifConfigurationError("frequent lock implementation_digest is invalid")
    if complete.get("implementation_digest") != implementation_digest:
        raise LearnedMotifConfigurationError(
            "frequent completion and lock disagree on implementation_digest"
        )
    repository_root = _repository_root(Path(__file__))
    observed_implementation = sweep_implementation_digest(repository_root)
    if observed_implementation != implementation_digest:
        raise LearnedMotifConfigurationError(
            "frequent artifacts were produced by a stale implementation"
        )
    try:
        loaded_sweep = frequent_sweeps.load_sweep_config(
            _frequent_config_path_from_completion(root, complete)
        )
        verified_run = frequent_sweeps._completed_run(
            complete_path,
            loaded=loaded_sweep,
            stage_name="final",
            run_dir=root,
            input_manifest_sha256=str(complete.get("input_manifest_sha256")),
            implementation_digest=implementation_digest,
        )
    except (OSError, TypeError, ValueError) as error:
        raise LearnedMotifConfigurationError(
            "frequent run failed recursive completion authentication"
        ) from error
    if verified_run.run_dir.resolve() != root:
        raise LearnedMotifConfigurationError(
            "frequent completion verifier resolved the wrong run directory"
        )

    raw_artifacts = complete.get("artifacts")
    raw_graph_caches = complete.get("graph_caches")
    if not isinstance(raw_artifacts, dict) or not isinstance(raw_graph_caches, dict):
        raise LearnedMotifConfigurationError(
            "frequent completion requires artifacts and graph_caches objects"
        )
    required_artifacts = {
        "candidate_pool",
        "mining",
        "sweep_table",
        "selected_vocabulary",
        "selection_lock",
        "heldout_results",
    }
    if set(raw_artifacts) != required_artifacts:
        raise LearnedMotifConfigurationError(
            "frequent completion artifact keys do not match the frozen contract"
        )
    artifacts = {
        name: _artifact_ref(value, label=f"frequent artifact {name}")
        for name, value in raw_artifacts.items()
    }
    for reference in artifacts.values():
        _read_artifact(root, reference)
    if artifacts["selection_lock"].sha256 != _sha256_bytes(lock_data):
        raise LearnedMotifConfigurationError(
            "frequent completion selection-lock checksum is inconsistent"
        )

    expected_splits = {split.value for split in CorpusSplit}
    if set(raw_graph_caches) != expected_splits:
        raise LearnedMotifConfigurationError(
            "frequent completion must reference all four split graph caches"
        )
    graph_caches: dict[CorpusSplit, ArtifactReference] = {}
    for split in CorpusSplit:
        reference = _artifact_ref(
            raw_graph_caches[split.value],
            label=f"{split.value} graph cache",
        )
        _read_artifact(root, reference)
        graph_caches[split] = reference

    lock_candidate = _artifact_ref(
        lock.get("candidate_pool"),
        label="frequent lock candidate_pool",
    )
    lock_sweep = _artifact_ref(
        lock.get("sweep_table"),
        label="frequent lock sweep_table",
    )
    lock_selected = _artifact_ref(
        lock.get("selected_vocabulary"),
        label="frequent lock selected_vocabulary",
    )
    for name, lock_ref in (
        ("candidate_pool", lock_candidate),
        ("sweep_table", lock_sweep),
        ("selected_vocabulary", lock_selected),
    ):
        if lock_ref != artifacts[name]:
            raise LearnedMotifConfigurationError(f"frequent lock and completion disagree on {name}")

    candidate_payload = _json_object(
        _read_artifact(root, lock_candidate),
        label="candidate pool vocabulary",
    )
    selected_payload = _json_object(
        _read_artifact(root, lock_selected),
        label="selected frequent vocabulary",
    )
    try:
        candidate = vocabulary_from_payload(candidate_payload)
        selected = vocabulary_from_payload(selected_payload)
    except (KeyError, TypeError, ValueError) as error:
        raise LearnedMotifConfigurationError(
            "frequent vocabulary artifact failed structural validation"
        ) from error

    selected_configuration = lock.get("selected_configuration")
    if not isinstance(selected_configuration, dict):
        raise LearnedMotifConfigurationError(
            "frequent selection lock requires selected_configuration"
        )
    required_selected_fields = {
        "configuration_digest",
        "minimum_motif_size",
        "maximum_motif_size",
        "requested_vocabulary_size",
        "actual_vocabulary_size",
        "vocabulary_id",
        "vocabulary_payload_sha256",
        "train_total_mdl_bits",
        "validation_total_mdl_bits",
    }
    if set(selected_configuration) != required_selected_fields:
        raise LearnedMotifConfigurationError(
            "frequent selected_configuration fields do not match the frozen contract"
        )
    if selected_configuration["vocabulary_id"] != selected.vocabulary_id:
        raise LearnedMotifConfigurationError("frequent lock selected vocabulary ID is inconsistent")
    if selected_configuration["actual_vocabulary_size"] != len(selected.templates):
        raise LearnedMotifConfigurationError(
            "frequent lock selected vocabulary size is inconsistent"
        )
    if selected_configuration["vocabulary_payload_sha256"] != vocabulary_payload_digest(selected):
        raise LearnedMotifConfigurationError(
            "frequent lock selected vocabulary payload digest is inconsistent"
        )
    configuration_digest_value = selected_configuration["configuration_digest"]
    if (
        not isinstance(configuration_digest_value, str)
        or len(configuration_digest_value) != 64
        or any(character not in "0123456789abcdef" for character in configuration_digest_value)
    ):
        raise LearnedMotifConfigurationError("frequent selected configuration digest is invalid")
    configuration_digest = configuration_digest_value
    if complete.get("selected_configuration_digest") != configuration_digest:
        raise LearnedMotifConfigurationError(
            "frequent completion selected configuration digest is inconsistent"
        )
    for name in ("stage", "config_digest", "input_manifest_sha256"):
        if complete.get(name) != lock.get(name):
            raise LearnedMotifConfigurationError(f"frequent completion and lock disagree on {name}")
    for name in ("config_digest", "input_manifest_sha256"):
        value = lock.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise LearnedMotifConfigurationError(f"frequent lock {name} is invalid")
    requested_size = selected_configuration["requested_vocabulary_size"]
    if (
        selected_configuration["minimum_motif_size"] != selected.min_size
        or selected_configuration["maximum_motif_size"] != selected.max_size
        or isinstance(requested_size, bool)
        or not isinstance(requested_size, int)
        or requested_size < len(selected.templates)
    ):
        raise LearnedMotifConfigurationError(
            "frequent selected vocabulary bounds/budget are inconsistent"
        )
    graph_cache_descriptors: dict[CorpusSplit, GraphCacheDescriptor] = {}
    for split, reference in graph_caches.items():
        try:
            descriptor = load_completed_graph_cache(
                _artifact_path(root, reference),
                run_dir=root,
                config_digest=lock["config_digest"],
                input_manifest_sha256=lock["input_manifest_sha256"],
                implementation_digest=implementation_digest,
            )
        except (OSError, TypeError, ValueError) as error:
            raise LearnedMotifConfigurationError(
                f"frequent graph cache for {split.value} is invalid"
            ) from error
        if descriptor.split is not split:
            raise LearnedMotifConfigurationError(
                f"frequent graph cache for {split.value} has the wrong split"
            )
        graph_cache_descriptors[split] = descriptor

    return FrequentSweepInputs(
        run_dir=root,
        candidate_pool=candidate,
        selected_frequent=selected,
        selection_lock_sha256=_sha256_bytes(lock_data),
        run_complete_sha256=_sha256_bytes(complete_data),
        config_digest=lock["config_digest"],
        input_manifest_sha256=lock["input_manifest_sha256"],
        implementation_digest=implementation_digest,
        selected_configuration_digest=configuration_digest,
        graph_cache_references=graph_caches,
        graph_cache_descriptors=graph_cache_descriptors,
    )


def frequent_graph_split_loader(
    inputs: FrequentSweepInputs,
) -> SplitGraphLoader:
    """Build a lazy authenticated split loader from completed 5-5 caches."""

    if not isinstance(inputs, FrequentSweepInputs):
        raise TypeError("inputs must be FrequentSweepInputs")

    def load(split: CorpusSplit) -> SplitGraphBatch:
        descriptor = _load_frequent_graph_cache(inputs, split)
        _LOGGER.info(
            "loading %s graph cache (%d successes, %d retained failures)",
            split.value,
            descriptor.success_count,
            descriptor.failure_count,
        )
        records = tuple(
            GraphExample(
                expression_id=expression_id,
                split=split,
                graph=graph,
            )
            for expression_id, graph in iter_cached_graphs(descriptor)
        )
        failures = _graph_cache_failures(descriptor.failures, split=split)
        _LOGGER.info("loaded %s graph cache", split.value)
        return SplitGraphBatch(
            split=split,
            records=records,
            failures=failures,
        )

    return load


def _load_frequent_graph_cache(
    inputs: FrequentSweepInputs,
    split: CorpusSplit,
):
    reference = inputs.graph_cache_references[split]
    manifest_path = _artifact_path(inputs.run_dir, reference)
    descriptor = load_completed_graph_cache(
        manifest_path,
        run_dir=inputs.run_dir,
        config_digest=inputs.config_digest,
        input_manifest_sha256=inputs.input_manifest_sha256,
        implementation_digest=inputs.implementation_digest,
    )
    if descriptor.split is not split:
        raise LearnedMotifConfigurationError(
            f"graph cache reference for {split.value} resolves to {descriptor.split.value}"
        )
    return descriptor


def _graph_cache_failures(
    failures: Iterable[Mapping[str, str]],
    *,
    split: CorpusSplit,
) -> tuple[GraphInputFailure, ...]:
    return tuple(
        GraphInputFailure(
            expression_id=failure.get("expression_id", "<unknown>"),
            split=split,
            stage=failure.get("stage", "graph_cache"),
            error_type=failure.get("error_type", "GraphCacheError"),
            error_message=failure.get(
                "error_message",
                "graph cache failure omitted a message",
            ),
        )
        for failure in failures
    )


def _graph_cache_expression_ids(
    descriptor: GraphCacheDescriptor,
) -> tuple[str, ...]:
    """Read only identity evidence from an already authenticated graph cache."""

    expression_ids = [failure.get("expression_id", "<unknown>") for failure in descriptor.failures]
    success_count = 0
    parquet = pq.ParquetFile(descriptor.data_path)
    for batch in parquet.iter_batches(
        batch_size=50_000,
        columns=("expression_id", "split"),
    ):
        batch_expression_ids = batch.column(0).to_pylist()
        split_values = batch.column(1).to_pylist()
        if any(value != descriptor.split.value for value in split_values):
            raise LearnedMotifProtocolError(
                f"{descriptor.split.value} graph cache contains a split mismatch"
            )
        success_count += len(batch_expression_ids)
        expression_ids.extend(batch_expression_ids)
    if success_count != descriptor.success_count:
        raise LearnedMotifProtocolError(
            f"{descriptor.split.value} identity count disagrees with its graph cache"
        )
    ordered = tuple(sorted(expression_ids))
    if (
        len(ordered) != descriptor.processed_count
        or len(set(ordered)) != len(ordered)
        or any(
            not isinstance(expression_id, str) or not expression_id.strip()
            for expression_id in ordered
        )
    ):
        raise LearnedMotifProtocolError(
            f"{descriptor.split.value} graph-cache identities are invalid"
        )
    return ordered


@dataclass(frozen=True, slots=True)
class GraphExample:
    """One graph identity used for partitioning and audit, never as a feature."""

    expression_id: str
    split: CorpusSplit
    graph: Graph

    def __post_init__(self) -> None:
        if not isinstance(self.expression_id, str) or not self.expression_id.strip():
            raise ValueError("expression_id must be nonblank")
        if not isinstance(self.split, CorpusSplit):
            raise TypeError("split must be a CorpusSplit")
        validation = validate_graph(self.graph)
        if not validation.valid:
            raise ValueError("graph is invalid: " + "; ".join(validation.errors))


@dataclass(frozen=True, slots=True)
class GraphInputFailure:
    """One upstream row that could not produce an evaluable graph."""

    expression_id: str
    split: CorpusSplit
    stage: str
    error_type: str
    error_message: str

    def __post_init__(self) -> None:
        for name, value in (
            ("expression_id", self.expression_id),
            ("stage", self.stage),
            ("error_type", self.error_type),
            ("error_message", self.error_message),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonblank")
        if not isinstance(self.split, CorpusSplit):
            raise TypeError("split must be a CorpusSplit")


@dataclass(frozen=True, slots=True)
class SplitGraphBatch:
    """Every success and retained upstream failure for one corpus split."""

    split: CorpusSplit
    records: tuple[GraphExample, ...]
    failures: tuple[GraphInputFailure, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "records",
            tuple(sorted(self.records, key=lambda record: record.expression_id)),
        )
        object.__setattr__(
            self,
            "failures",
            tuple(sorted(self.failures, key=lambda failure: failure.expression_id)),
        )
        if any(record.split is not self.split for record in self.records):
            raise ValueError("every graph record must match its batch split")
        if any(failure.split is not self.split for failure in self.failures):
            raise ValueError("every graph failure must match its batch split")
        identities = [
            *(record.expression_id for record in self.records),
            *(failure.expression_id for failure in self.failures),
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("split batch contains duplicate expression IDs")

    @property
    def source_row_count(self) -> int:
        return len(self.records) + len(self.failures)


@dataclass(frozen=True, slots=True)
class TrainPartition:
    """Deterministic TRAIN-only target-fitting partition."""

    batch: SplitGraphBatch
    fraction: float
    maximum_graphs: int | None
    seed: int
    digest: str
    full_train_graph_count: int
    fractional_graph_count: int
    source_expression_ids: tuple[str, ...]
    source_identity_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_expression_ids", tuple(self.source_expression_ids))
        if self.batch.split is not CorpusSplit.TRAIN:
            raise ValueError("train partition batch must be TRAIN")
        if (
            type(self.fraction) is not float
            or not math.isfinite(self.fraction)
            or not 0.0 < self.fraction <= 1.0
        ):
            raise ValueError("fraction must be an exact finite float in (0, 1]")
        if self.maximum_graphs is not None and (
            isinstance(self.maximum_graphs, bool)
            or not isinstance(self.maximum_graphs, int)
            or self.maximum_graphs < 1
        ):
            raise ValueError("maximum_graphs must be null or a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            not isinstance(self.digest, str)
            or len(self.digest) != 64
            or any(character not in "0123456789abcdef" for character in self.digest)
        ):
            raise ValueError("digest must be lowercase SHA-256 hex")
        for name, value in (
            ("full_train_graph_count", self.full_train_graph_count),
            ("fractional_graph_count", self.fractional_graph_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        expected_fractional_count = max(
            1,
            math.ceil(self.full_train_graph_count * self.fraction),
        )
        if self.fractional_graph_count != expected_fractional_count:
            raise ValueError("fractional graph count is inconsistent")
        selected_count = (
            self.fractional_graph_count
            if self.maximum_graphs is None
            else min(self.fractional_graph_count, self.maximum_graphs)
        )
        if len(self.batch.records) != selected_count:
            raise ValueError("selected TRAIN graph count is inconsistent")
        if (
            self.source_expression_ids != tuple(sorted(self.source_expression_ids))
            or len(set(self.source_expression_ids)) != len(self.source_expression_ids)
            or any(
                not isinstance(expression_id, str) or not expression_id.strip()
                for expression_id in self.source_expression_ids
            )
            or len(self.source_expression_ids)
            != self.full_train_graph_count + len(self.batch.failures)
        ):
            raise ValueError("TRAIN source identity evidence is inconsistent")
        if self.source_identity_digest != _split_identity_digest(self.source_expression_ids):
            raise ValueError("TRAIN source identity digest is inconsistent")


def _split_identity_digest(expression_ids: Sequence[str]) -> str:
    digest = hashlib.sha256(SPLIT_IDENTITY_VERSION.encode("ascii"))
    for expression_id in expression_ids:
        encoded = expression_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_partition_parameters(
    *,
    fraction: float,
    seed: int,
    maximum_graphs: int | None,
) -> None:
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0.0 < fraction <= 1.0
    ):
        raise ValueError("fraction must be finite and lie in (0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if maximum_graphs is not None and (
        isinstance(maximum_graphs, bool)
        or not isinstance(maximum_graphs, int)
        or maximum_graphs < 1
    ):
        raise ValueError("maximum_graphs must be null or a positive integer")


def _train_partition_key(
    record: GraphExample,
    *,
    seed: int,
) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    digest.update(PARTITION_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(record.expression_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(graph_structure_fingerprint(record.graph).encode("ascii"))
    return digest.digest(), record.expression_id


def _build_train_partition(
    selected: Iterable[GraphExample],
    *,
    failures: tuple[GraphInputFailure, ...],
    source_expression_ids: Iterable[str],
    full_train_graph_count: int,
    fraction: float,
    seed: int,
    maximum_graphs: int | None,
) -> TrainPartition:
    fractional_count = max(
        1,
        math.ceil(full_train_graph_count * float(fraction)),
    )
    selected = tuple(sorted(selected, key=lambda record: record.expression_id))
    expression_ids = tuple(sorted(source_expression_ids))
    source_identity_digest = _split_identity_digest(expression_ids)
    digest_payload = {
        "failures": [
            {
                "error_type": failure.error_type,
                "expression_id": failure.expression_id,
                "stage": failure.stage,
            }
            for failure in failures
        ],
        "fraction": float(fraction),
        "fractional_graph_count": fractional_count,
        "maximum_graphs": maximum_graphs,
        "records": [
            {
                "expression_id": record.expression_id,
                "graph_fingerprint": graph_structure_fingerprint(record.graph),
            }
            for record in selected
        ],
        "seed": seed,
        "source_identity_digest": source_identity_digest,
        "version": PARTITION_VERSION,
    }
    digest = _sha256_bytes(_canonical_json_bytes(digest_payload))
    return TrainPartition(
        batch=SplitGraphBatch(
            split=CorpusSplit.TRAIN,
            records=selected,
            failures=failures,
        ),
        fraction=float(fraction),
        maximum_graphs=maximum_graphs,
        seed=seed,
        digest=digest,
        full_train_graph_count=full_train_graph_count,
        fractional_graph_count=fractional_count,
        source_expression_ids=expression_ids,
        source_identity_digest=source_identity_digest,
    )


def deterministic_train_partition(
    train_batch: SplitGraphBatch,
    *,
    fraction: float,
    seed: int,
    maximum_graphs: int | None = None,
) -> TrainPartition:
    """Choose a stable TRAIN subset without exposing IDs to motif features."""

    if train_batch.split is not CorpusSplit.TRAIN:
        raise LearnedMotifProtocolError("target fitting requires the TRAIN split")
    _validate_partition_parameters(
        fraction=fraction,
        seed=seed,
        maximum_graphs=maximum_graphs,
    )
    if not train_batch.records:
        raise LearnedMotifProtocolError("TRAIN contains no evaluable graphs")
    fractional_count = max(
        1,
        math.ceil(len(train_batch.records) * float(fraction)),
    )
    selected_count = (
        fractional_count if maximum_graphs is None else min(fractional_count, maximum_graphs)
    )
    selected = heapq.nsmallest(
        selected_count,
        train_batch.records,
        key=lambda record: _train_partition_key(record, seed=seed),
    )
    return _build_train_partition(
        selected,
        failures=train_batch.failures,
        source_expression_ids=_batch_expression_ids(train_batch),
        full_train_graph_count=len(train_batch.records),
        fraction=fraction,
        seed=seed,
        maximum_graphs=maximum_graphs,
    )


def load_frequent_train_partition(
    inputs: FrequentSweepInputs,
    *,
    fraction: float,
    seed: int,
    maximum_graphs: int | None,
) -> TrainPartition:
    """Stream an authenticated TRAIN cache while retaining only deterministic top-k graphs."""

    if not isinstance(inputs, FrequentSweepInputs):
        raise TypeError("inputs must be FrequentSweepInputs")
    _validate_partition_parameters(
        fraction=fraction,
        seed=seed,
        maximum_graphs=maximum_graphs,
    )
    descriptor = _load_frequent_graph_cache(inputs, CorpusSplit.TRAIN)
    if descriptor.success_count < 1:
        raise LearnedMotifProtocolError("TRAIN contains no evaluable graphs")
    fractional_count = max(
        1,
        math.ceil(descriptor.success_count * float(fraction)),
    )
    selected_count = (
        fractional_count if maximum_graphs is None else min(fractional_count, maximum_graphs)
    )
    source_expression_ids = [
        failure.get("expression_id", "<unknown>") for failure in descriptor.failures
    ]
    streamed_count = 0

    def records() -> Iterable[GraphExample]:
        nonlocal streamed_count
        for expression_id, graph in iter_cached_graphs(descriptor):
            streamed_count += 1
            source_expression_ids.append(expression_id)
            if streamed_count % 25_000 == 0:
                _LOGGER.info(
                    "streamed %d/%d TRAIN graphs for deterministic partitioning",
                    streamed_count,
                    descriptor.success_count,
                )
            yield GraphExample(
                expression_id=expression_id,
                split=CorpusSplit.TRAIN,
                graph=graph,
            )

    _LOGGER.info(
        "streaming TRAIN cache to retain %d of %d evaluable graphs",
        selected_count,
        descriptor.success_count,
    )
    selected = heapq.nsmallest(
        selected_count,
        records(),
        key=lambda record: _train_partition_key(record, seed=seed),
    )
    if streamed_count != descriptor.success_count:
        raise LearnedMotifProtocolError(
            "streamed TRAIN graph count disagrees with the authenticated descriptor"
        )
    partition = _build_train_partition(
        selected,
        failures=_graph_cache_failures(
            descriptor.failures,
            split=CorpusSplit.TRAIN,
        ),
        source_expression_ids=source_expression_ids,
        full_train_graph_count=descriptor.success_count,
        fraction=fraction,
        seed=seed,
        maximum_graphs=maximum_graphs,
    )
    _LOGGER.info(
        "locked deterministic TRAIN partition (%d retained graphs)",
        len(partition.batch.records),
    )
    return partition


class ExperimentMethod(StrEnum):
    """Compared representations in the locked experiment."""

    LEARNED = "learned"
    FREQUENT = "frequent"
    RANDOM = "random"
    MACRO = "macro_without_motifs"


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """One retained motif compression or reconstruction failure."""

    expression_id: str
    error_type: str
    error_message: str
    attempted_selected_occurrence_count: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("expression_id", self.expression_id),
            ("error_type", self.error_type),
            ("error_message", self.error_message),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonblank")
        if (
            isinstance(self.attempted_selected_occurrence_count, bool)
            or not isinstance(self.attempted_selected_occurrence_count, int)
            or self.attempted_selected_occurrence_count < 0
        ):
            raise ValueError("attempted_selected_occurrence_count must be nonnegative")


@dataclass(frozen=True, slots=True)
class VocabularyArm:
    """One frozen vocabulary evaluated under a named experimental method."""

    arm_id: str
    method: ExperimentMethod
    vocabulary: MotifVocabulary
    budget: int
    ridge_lambda: float | None = None
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.arm_id, str) or not self.arm_id.strip():
            raise ValueError("arm_id must be nonblank")
        if not isinstance(self.method, ExperimentMethod):
            raise TypeError("method must be an ExperimentMethod")
        if not isinstance(self.vocabulary, MotifVocabulary):
            raise TypeError("vocabulary must be a MotifVocabulary")
        if (
            isinstance(self.budget, bool)
            or not isinstance(self.budget, int)
            or self.budget != len(self.vocabulary.templates)
            or self.budget < 1
        ):
            raise ValueError("arm budget must equal its positive vocabulary size")
        if self.method is ExperimentMethod.LEARNED:
            if (
                type(self.ridge_lambda) is not float
                or not math.isfinite(self.ridge_lambda)
                or self.ridge_lambda < 0.0
                or self.random_seed is not None
            ):
                raise ValueError("learned arms require only ridge_lambda")
        elif self.method is ExperimentMethod.RANDOM:
            if (
                isinstance(self.random_seed, bool)
                or not isinstance(self.random_seed, int)
                or self.random_seed < 0
                or self.ridge_lambda is not None
            ):
                raise ValueError("random arms require only random_seed")
        elif self.ridge_lambda is not None or self.random_seed is not None:
            raise ValueError("frequent arms cannot carry learned/random hyperparameters")


@dataclass(frozen=True, slots=True)
class VocabularyEvaluation:
    """Exact split MDL plus every upstream and reconstruction failure."""

    split: CorpusSplit
    arm_id: str
    method: ExperimentMethod
    budget: int
    vocabulary_id: str | None
    vocabulary_payload_sha256: str | None
    summary: SplitMDLSummary
    upstream_failures: tuple[GraphInputFailure, ...] = ()
    reconstruction_failures: tuple[EvaluationFailure, ...] = ()
    ridge_lambda: float | None = None
    random_seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "upstream_failures", tuple(self.upstream_failures))
        object.__setattr__(
            self,
            "reconstruction_failures",
            tuple(self.reconstruction_failures),
        )
        if not isinstance(self.split, CorpusSplit):
            raise TypeError("split must be a CorpusSplit")
        if not isinstance(self.arm_id, str) or not self.arm_id.strip():
            raise ValueError("arm_id must be nonblank")
        if not isinstance(self.method, ExperimentMethod):
            raise TypeError("method must be an ExperimentMethod")
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        if not isinstance(self.summary, SplitMDLSummary):
            raise TypeError("summary must be a SplitMDLSummary")
        if any(
            not isinstance(failure, GraphInputFailure) or failure.split is not self.split
            for failure in self.upstream_failures
        ):
            raise ValueError("upstream failures must match the evaluation split")
        if any(
            not isinstance(failure, EvaluationFailure) for failure in self.reconstruction_failures
        ):
            raise TypeError("reconstruction failures must be EvaluationFailure records")
        if len(self.reconstruction_failures) != self.summary.reconstruction_failure_count:
            raise ValueError("reconstruction failure rows do not match the MDL summary")
        if self.method is ExperimentMethod.MACRO:
            if (
                self.budget != 0
                or self.vocabulary_id is not None
                or self.vocabulary_payload_sha256 is not None
            ):
                raise ValueError("macro baseline cannot carry a motif vocabulary")
        elif (
            self.budget < 1
            or not isinstance(self.vocabulary_id, str)
            or not self.vocabulary_id.strip()
            or not isinstance(self.vocabulary_payload_sha256, str)
            or len(self.vocabulary_payload_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.vocabulary_payload_sha256
            )
        ):
            raise ValueError("motif evaluations require a positive typed vocabulary")
        if self.method is ExperimentMethod.LEARNED:
            if (
                type(self.ridge_lambda) is not float
                or not math.isfinite(self.ridge_lambda)
                or self.ridge_lambda < 0.0
                or self.random_seed is not None
            ):
                raise ValueError("learned evaluations require only ridge_lambda")
        elif self.method is ExperimentMethod.RANDOM:
            if (
                isinstance(self.random_seed, bool)
                or not isinstance(self.random_seed, int)
                or self.random_seed < 0
                or self.ridge_lambda is not None
            ):
                raise ValueError("random evaluations require only random_seed")
        elif self.ridge_lambda is not None or self.random_seed is not None:
            raise ValueError("frequent/macro evaluations cannot carry model parameters")

    @property
    def total_failure_count(self) -> int:
        return len(self.upstream_failures) + len(self.reconstruction_failures)

    @property
    def source_row_count(self) -> int:
        return self.summary.processed_count + len(self.upstream_failures)

    @property
    def eligible(self) -> bool:
        return self.total_failure_count == 0


@dataclass(slots=True)
class _MDLAccumulator:
    vocabulary: MotifVocabulary
    baseline_data_bits: int = 0
    conditional_data_bits: int = 0
    framing_bits: int = 0
    residual_bits: int = 0
    occurrence_bits: int = 0
    candidate_occurrence_count: int = 0
    selected_occurrence_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    motif_counts: Counter[str] = field(default_factory=Counter)
    failures: list[EvaluationFailure] = field(default_factory=list)

    def add(self, expression_id: str, result: MotifGraphMDLResult) -> None:
        self.baseline_data_bits += result.baseline_bits
        self.conditional_data_bits += result.conditional_data_bits
        self.framing_bits += result.framing_bits
        self.residual_bits += result.residual_bits
        self.occurrence_bits += result.occurrence_bits
        self.candidate_occurrence_count += result.candidate_occurrence_count
        self.selected_occurrence_count += result.selected_occurrence_count
        self.success_count += result.success
        self.failure_count += result.reconstruction_failure_count
        self.motif_counts.update(dict(result.selected_motif_counts))
        if not result.success:
            self.failures.append(
                EvaluationFailure(
                    expression_id=expression_id,
                    error_type=result.error_type or "MotifCompressionError",
                    error_message=result.error_message or "motif compression failed",
                    attempted_selected_occurrence_count=(
                        result.attempted_selected_occurrence_count
                    ),
                )
            )

    def finish(
        self,
        *,
        split: CorpusSplit,
        arm: VocabularyArm,
        upstream_failures: tuple[GraphInputFailure, ...],
    ) -> VocabularyEvaluation:
        baseline_dictionary = vocabulary_mdl_bits(())
        dictionary = vocabulary_mdl_bits(self.vocabulary.templates)
        processed = self.success_count + self.failure_count
        summary = SplitMDLSummary(
            processed_count=processed,
            success_count=self.success_count,
            reconstruction_failure_count=self.failure_count,
            baseline_dictionary_bits=baseline_dictionary,
            baseline_data_bits=self.baseline_data_bits,
            baseline_total_bits=baseline_dictionary + self.baseline_data_bits,
            dictionary_bits=dictionary,
            conditional_data_bits=self.conditional_data_bits,
            total_mdl_bits=dictionary + self.conditional_data_bits,
            framing_bits=self.framing_bits,
            residual_bits=self.residual_bits,
            occurrence_bits=self.occurrence_bits,
            candidate_occurrence_count=self.candidate_occurrence_count,
            selected_occurrence_count=self.selected_occurrence_count,
            selected_motif_counts=tuple(sorted(self.motif_counts.items())),
        )
        return VocabularyEvaluation(
            split=split,
            arm_id=arm.arm_id,
            method=arm.method,
            budget=arm.budget,
            vocabulary_id=arm.vocabulary.vocabulary_id,
            vocabulary_payload_sha256=vocabulary_payload_digest(arm.vocabulary),
            summary=summary,
            upstream_failures=upstream_failures,
            reconstruction_failures=tuple(self.failures),
            ridge_lambda=arm.ridge_lambda,
            random_seed=arm.random_seed,
        )


def _macro_evaluation(
    batch: SplitGraphBatch,
    reference: VocabularyEvaluation,
) -> VocabularyEvaluation:
    summary = SplitMDLSummary(
        processed_count=reference.summary.processed_count,
        success_count=reference.summary.processed_count,
        reconstruction_failure_count=0,
        baseline_dictionary_bits=reference.summary.baseline_dictionary_bits,
        baseline_data_bits=reference.summary.baseline_data_bits,
        baseline_total_bits=reference.summary.baseline_total_bits,
        dictionary_bits=reference.summary.baseline_dictionary_bits,
        conditional_data_bits=reference.summary.baseline_data_bits,
        total_mdl_bits=reference.summary.baseline_total_bits,
        framing_bits=0,
        residual_bits=reference.summary.baseline_data_bits,
        occurrence_bits=0,
        candidate_occurrence_count=0,
        selected_occurrence_count=0,
        selected_motif_counts=(),
    )
    return VocabularyEvaluation(
        split=batch.split,
        arm_id="macro",
        method=ExperimentMethod.MACRO,
        budget=0,
        vocabulary_id=None,
        vocabulary_payload_sha256=None,
        summary=summary,
        upstream_failures=batch.failures,
    )


def evaluate_vocabulary_arms(
    batch: SplitGraphBatch,
    arms: Sequence[VocabularyArm],
    *,
    candidate_pool: MotifVocabulary,
) -> tuple[VocabularyEvaluation, ...]:
    """Evaluate all fixed vocabularies with one union match pass per graph."""

    materialized = tuple(arms)
    if not materialized:
        raise ValueError("at least one vocabulary arm is required")
    if not isinstance(candidate_pool, MotifVocabulary) or not candidate_pool.templates:
        raise ValueError("candidate_pool must be a nonempty MotifVocabulary")
    arm_ids = [arm.arm_id for arm in materialized]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("vocabulary arm IDs must be unique")
    candidate_templates = candidate_pool.by_id()
    for arm in materialized:
        if (
            arm.vocabulary.pool is not candidate_pool.pool
            or arm.vocabulary.training_fingerprint != candidate_pool.training_fingerprint
        ):
            raise ValueError("all arms must derive from the supplied train-only pool")
        for template in arm.vocabulary.templates:
            if candidate_templates.get(template.motif_id) != template:
                raise ValueError("arm contains a motif outside the candidate pool")

    vocabulary_keys = {
        arm.arm_id: (
            arm.vocabulary.vocabulary_id,
            vocabulary_payload_digest(arm.vocabulary),
        )
        for arm in materialized
    }
    representative_arms: dict[tuple[str, str], VocabularyArm] = {}
    for arm in materialized:
        representative_arms.setdefault(vocabulary_keys[arm.arm_id], arm)
    accumulators = {
        key: _MDLAccumulator(arm.vocabulary) for key, arm in representative_arms.items()
    }
    motif_ids_by_key = {
        key: tuple(template.motif_id for template in arm.vocabulary.templates)
        for key, arm in representative_arms.items()
    }
    for record_index, record in enumerate(batch.records, start=1):
        prepared_graph = prepare_graph_mdl(record.graph)
        try:
            occurrences = find_vocabulary_occurrences(record.graph, candidate_pool)
        except Exception as error:
            matching_failure = fallback_mdl_result(
                record.graph,
                error_type=type(error).__name__,
                error_message=(
                    "candidate-pool union matching failed: " + (str(error) or type(error).__name__)
                ),
                prepared_graph=prepared_graph,
            )
            for key in representative_arms:
                accumulators[key].add(
                    record.expression_id,
                    matching_failure,
                )
            continue
        else:
            by_motif: dict[str, list[MotifOccurrence]] = {}
            for occurrence in occurrences:
                by_motif.setdefault(occurrence.motif_id, []).append(occurrence)
            occurrence_batches = {
                key: tuple(
                    chain.from_iterable(by_motif.get(motif_id, ()) for motif_id in motif_ids)
                )
                for key, motif_ids in motif_ids_by_key.items()
            }
        for key, arm in representative_arms.items():
            result = motif_graph_mdl_result(
                record.graph,
                arm.vocabulary,
                occurrences=occurrence_batches[key],
                prepared_graph=prepared_graph,
            )
            accumulators[key].add(record.expression_id, result)
        if record_index % 1_000 == 0 or record_index == len(batch.records):
            _LOGGER.info(
                "evaluated %d/%d %s graphs across %d unique motif vocabularies",
                record_index,
                len(batch.records),
                batch.split.value,
                len(representative_arms),
            )

    return tuple(
        accumulators[vocabulary_keys[arm.arm_id]].finish(
            split=batch.split,
            arm=arm,
            upstream_failures=batch.failures,
        )
        for arm in materialized
    )


def _selected_vocabulary(
    candidate_pool: MotifVocabulary,
    templates: Iterable[MotifTemplate],
    *,
    budget: int,
) -> MotifVocabulary:
    selected = tuple(templates)
    if len(selected) != budget:
        raise LearnedMotifProtocolError("selected vocabulary does not meet the locked budget")
    return build_motif_vocabulary(
        pool=candidate_pool.pool,
        min_size=candidate_pool.min_size,
        max_size=candidate_pool.max_size,
        min_support_count=candidate_pool.min_support_count,
        vocabulary_limit=budget,
        training_transaction_count=candidate_pool.training_transaction_count,
        processed_count=candidate_pool.processed_count,
        failure_count=candidate_pool.failure_count,
        training_fingerprint=candidate_pool.training_fingerprint,
        templates=selected,
    )


def align_frequent_candidate_pool(
    candidate_pool: MotifVocabulary,
    selected_frequent: MotifVocabulary,
) -> tuple[MotifVocabulary, int]:
    """Restrict the train-only pool to the locked 5-5 size range and budget."""

    if candidate_pool.pool is not selected_frequent.pool:
        raise LearnedMotifProtocolError("candidate and frequent pools differ")
    if candidate_pool.training_fingerprint != selected_frequent.training_fingerprint:
        raise LearnedMotifProtocolError("candidate and frequent training fingerprints differ")
    candidate_by_id = candidate_pool.by_id()
    for template in selected_frequent.templates:
        if candidate_by_id.get(template.motif_id) != template:
            raise LearnedMotifProtocolError(
                "selected frequent vocabulary is not a subset of the candidate pool"
            )
    eligible_templates = tuple(
        template
        for template in candidate_pool.templates
        if selected_frequent.min_size <= template.internal_node_count <= selected_frequent.max_size
    )
    eligible = build_motif_vocabulary(
        pool=candidate_pool.pool,
        min_size=selected_frequent.min_size,
        max_size=selected_frequent.max_size,
        min_support_count=candidate_pool.min_support_count,
        vocabulary_limit=None,
        training_transaction_count=candidate_pool.training_transaction_count,
        processed_count=candidate_pool.processed_count,
        failure_count=candidate_pool.failure_count,
        training_fingerprint=candidate_pool.training_fingerprint,
        templates=eligible_templates,
    )
    budget = len(selected_frequent.templates)
    if budget < 1 or budget > len(eligible.templates):
        raise LearnedMotifProtocolError("locked frequent budget is not usable")
    expected_frequent = select_frequent_templates(eligible, budget=budget)
    if {template.motif_id for template in expected_frequent} != {
        template.motif_id for template in selected_frequent.templates
    }:
        raise LearnedMotifProtocolError(
            "selected 5-5 vocabulary is not the canonical equal-budget frequent baseline"
        )
    return eligible, budget


@dataclass(frozen=True, slots=True)
class CandidatePrefilterAudit:
    """TRAIN-only deterministic cap applied before any target or validation work."""

    method: str
    maximum_candidate_motifs: int
    source_candidate_count: int
    selected_candidate_count: int
    locked_frequent_budget: int
    source_candidate_pool_digest: str
    selected_candidate_pool_digest: str

    def __post_init__(self) -> None:
        if self.method != "train-frequency-rank-v1":
            raise ValueError("unsupported candidate prefilter method")
        for name, value in (
            ("maximum_candidate_motifs", self.maximum_candidate_motifs),
            ("source_candidate_count", self.source_candidate_count),
            ("selected_candidate_count", self.selected_candidate_count),
            ("locked_frequent_budget", self.locked_frequent_budget),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.selected_candidate_count > self.maximum_candidate_motifs:
            raise ValueError("selected candidate count exceeds the prefilter cap")
        if self.locked_frequent_budget > self.selected_candidate_count:
            raise ValueError("prefilter cannot be smaller than the frequent budget")
        for name, value in (
            ("source_candidate_pool_digest", self.source_candidate_pool_digest),
            ("selected_candidate_pool_digest", self.selected_candidate_pool_digest),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")


def prefilter_train_candidates(
    candidate_pool: MotifVocabulary,
    selected_frequent: MotifVocabulary,
    *,
    maximum_candidate_motifs: int,
) -> tuple[MotifVocabulary, CandidatePrefilterAudit]:
    """Cap candidates by the frozen train-frequency order shared by every arm."""

    if (
        isinstance(maximum_candidate_motifs, bool)
        or not isinstance(maximum_candidate_motifs, int)
        or maximum_candidate_motifs < 1
    ):
        raise ValueError("maximum_candidate_motifs must be a positive integer")
    budget = len(selected_frequent.templates)
    if maximum_candidate_motifs < budget:
        raise LearnedMotifProtocolError(
            "candidate prefilter cap is below the locked frequent budget"
        )
    if tuple(sorted(candidate_pool.templates, key=motif_rank_key)) != (candidate_pool.templates):
        raise LearnedMotifProtocolError(
            "candidate pool does not use the canonical train-frequency order"
        )
    selected_templates = candidate_pool.templates[:maximum_candidate_motifs]
    selected_ids = {template.motif_id for template in selected_templates}
    if any(template.motif_id not in selected_ids for template in selected_frequent.templates):
        raise LearnedMotifProtocolError(
            "train-only candidate prefilter removed a locked frequent motif"
        )
    selected_pool = build_motif_vocabulary(
        pool=candidate_pool.pool,
        min_size=candidate_pool.min_size,
        max_size=candidate_pool.max_size,
        min_support_count=candidate_pool.min_support_count,
        vocabulary_limit=maximum_candidate_motifs,
        training_transaction_count=candidate_pool.training_transaction_count,
        processed_count=candidate_pool.processed_count,
        failure_count=candidate_pool.failure_count,
        training_fingerprint=candidate_pool.training_fingerprint,
        templates=selected_templates,
    )
    return selected_pool, CandidatePrefilterAudit(
        method="train-frequency-rank-v1",
        maximum_candidate_motifs=maximum_candidate_motifs,
        source_candidate_count=len(candidate_pool.templates),
        selected_candidate_count=len(selected_pool.templates),
        locked_frequent_budget=budget,
        source_candidate_pool_digest=candidate_pool_digest(candidate_pool),
        selected_candidate_pool_digest=candidate_pool_digest(selected_pool),
    )


def _candidate_prefilter_payload(
    audit: CandidatePrefilterAudit,
) -> dict[str, object]:
    return {
        "locked_frequent_budget": audit.locked_frequent_budget,
        "maximum_candidate_motifs": audit.maximum_candidate_motifs,
        "method": audit.method,
        "selected_candidate_count": audit.selected_candidate_count,
        "selected_candidate_pool_digest": audit.selected_candidate_pool_digest,
        "source_candidate_count": audit.source_candidate_count,
        "source_candidate_pool_digest": audit.source_candidate_pool_digest,
    }


@dataclass(frozen=True, slots=True)
class ExactTrainingTarget:
    """One exact singleton-vocabulary TRAIN target and its complete audit."""

    example: SelectorTrainingExample
    summary: SplitMDLSummary
    upstream_failures: tuple[GraphInputFailure, ...]
    reconstruction_failures: tuple[EvaluationFailure, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "upstream_failures", tuple(self.upstream_failures))
        object.__setattr__(
            self,
            "reconstruction_failures",
            tuple(self.reconstruction_failures),
        )
        if not isinstance(self.example, SelectorTrainingExample):
            raise TypeError("example must be a SelectorTrainingExample")
        if not isinstance(self.summary, SplitMDLSummary):
            raise TypeError("summary must be a SplitMDLSummary")
        if self.example.mdl_gain_bits != self.summary.savings_bits:
            raise ValueError("training target gain must equal exact summary savings")
        if any(
            not isinstance(failure, GraphInputFailure) or failure.split is not CorpusSplit.TRAIN
            for failure in self.upstream_failures
        ):
            raise ValueError("target upstream failures must be TRAIN failures")
        if any(
            not isinstance(failure, EvaluationFailure) for failure in self.reconstruction_failures
        ):
            raise TypeError("target reconstruction failures must be EvaluationFailure records")
        if len(self.reconstruction_failures) != self.summary.reconstruction_failure_count:
            raise ValueError("target reconstruction failures do not match the exact summary")

    @property
    def total_failure_count(self) -> int:
        return len(self.upstream_failures) + len(self.reconstruction_failures)


@dataclass(frozen=True, slots=True)
class TrainingTargetAudit:
    """All exact targets used to fit every ridge candidate."""

    partition: TrainPartition
    targets: tuple[ExactTrainingTarget, ...]
    candidate_pool_digest: str
    candidate_prefilter: CandidatePrefilterAudit | None = None
    union_match_pass_count: int = 0
    base_singleton_evaluation_count: int = 0
    sparse_singleton_evaluation_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        if not self.targets or any(
            not isinstance(target, ExactTrainingTarget) for target in self.targets
        ):
            raise ValueError("target audit requires exact training targets")
        motif_ids = [target.example.motif_id for target in self.targets]
        if len(set(motif_ids)) != len(motif_ids):
            raise ValueError("target audit motif IDs must be unique")
        if (
            not isinstance(self.candidate_pool_digest, str)
            or len(self.candidate_pool_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.candidate_pool_digest)
        ):
            raise ValueError("candidate_pool_digest must be lowercase SHA-256 hex")
        for name, value in (
            ("union_match_pass_count", self.union_match_pass_count),
            (
                "base_singleton_evaluation_count",
                self.base_singleton_evaluation_count,
            ),
            (
                "sparse_singleton_evaluation_count",
                self.sparse_singleton_evaluation_count,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.union_match_pass_count != len(self.partition.batch.records):
            raise ValueError("target audit must retain one union-match pass per graph")
        if self.base_singleton_evaluation_count != len(self.partition.batch.records):
            raise ValueError("target audit must retain one base evaluation per graph")
        maximum_sparse_evaluations = len(self.partition.batch.records) * len(self.targets)
        if self.sparse_singleton_evaluation_count > maximum_sparse_evaluations:
            raise ValueError("sparse singleton evaluation count is impossible")
        if self.candidate_prefilter is not None and (
            self.candidate_prefilter.selected_candidate_pool_digest != self.candidate_pool_digest
        ):
            raise ValueError("candidate prefilter audit digest is inconsistent")
        for target in self.targets:
            if (
                target.summary.processed_count != len(self.partition.batch.records)
                or target.upstream_failures != self.partition.batch.failures
            ):
                raise ValueError("target audit rows must share the exact TRAIN partition evidence")

    @property
    def total_failure_count(self) -> int:
        return sum(target.total_failure_count for target in self.targets)


@dataclass(slots=True)
class _SparseTargetDelta:
    """Difference from the no-occurrence singleton result for observed motifs."""

    conditional_data_bits: int = 0
    framing_bits: int = 0
    residual_bits: int = 0
    occurrence_bits: int = 0
    candidate_occurrence_count: int = 0
    selected_occurrence_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    motif_counts: Counter[str] = field(default_factory=Counter)
    failure_overrides: dict[str, EvaluationFailure | None] = field(default_factory=dict)

    def add(
        self,
        expression_id: str,
        *,
        base: MotifGraphMDLResult,
        observed: MotifGraphMDLResult,
    ) -> None:
        if observed.baseline_bits != base.baseline_bits:
            raise LearnedMotifProtocolError("singleton and no-occurrence baseline costs disagree")
        self.conditional_data_bits += observed.conditional_data_bits - base.conditional_data_bits
        self.framing_bits += observed.framing_bits - base.framing_bits
        self.residual_bits += observed.residual_bits - base.residual_bits
        self.occurrence_bits += observed.occurrence_bits - base.occurrence_bits
        self.candidate_occurrence_count += (
            observed.candidate_occurrence_count - base.candidate_occurrence_count
        )
        self.selected_occurrence_count += (
            observed.selected_occurrence_count - base.selected_occurrence_count
        )
        self.success_count += int(observed.success) - int(base.success)
        self.failure_count += (
            observed.reconstruction_failure_count - base.reconstruction_failure_count
        )
        self.motif_counts.subtract(dict(base.selected_motif_counts))
        self.motif_counts.update(dict(observed.selected_motif_counts))
        if not base.success or not observed.success:
            self.failure_overrides[expression_id] = (
                None
                if observed.success
                else EvaluationFailure(
                    expression_id=expression_id,
                    error_type=observed.error_type or "MotifCompressionError",
                    error_message=(observed.error_message or "motif compression failed"),
                    attempted_selected_occurrence_count=(
                        observed.attempted_selected_occurrence_count
                    ),
                )
            )


def _union_matching_failure_result(
    graph: Graph,
    error: Exception,
    *,
    prepared_graph: PreparedGraphMDL,
) -> MotifGraphMDLResult:
    """Encode the original graph and retain a union-matching failure."""

    return fallback_mdl_result(
        graph,
        error_type=type(error).__name__,
        error_message=(
            "candidate-pool union matching failed: " + (str(error) or type(error).__name__)
        ),
        prepared_graph=prepared_graph,
    )


def _sparse_target_summary(
    *,
    vocabulary: MotifVocabulary,
    base: _MDLAccumulator,
    delta: _SparseTargetDelta | None,
) -> tuple[SplitMDLSummary, tuple[EvaluationFailure, ...]]:
    delta = delta or _SparseTargetDelta()
    baseline_dictionary = vocabulary_mdl_bits(())
    dictionary = vocabulary_mdl_bits(vocabulary.templates)
    success_count = base.success_count + delta.success_count
    failure_count = base.failure_count + delta.failure_count
    selected_counts = base.motif_counts.copy()
    selected_counts.update(delta.motif_counts)
    selected_motif_counts = tuple(
        sorted((motif_id, count) for motif_id, count in selected_counts.items() if count)
    )
    conditional_data_bits = base.conditional_data_bits + delta.conditional_data_bits
    summary = SplitMDLSummary(
        processed_count=success_count + failure_count,
        success_count=success_count,
        reconstruction_failure_count=failure_count,
        baseline_dictionary_bits=baseline_dictionary,
        baseline_data_bits=base.baseline_data_bits,
        baseline_total_bits=baseline_dictionary + base.baseline_data_bits,
        dictionary_bits=dictionary,
        conditional_data_bits=conditional_data_bits,
        total_mdl_bits=dictionary + conditional_data_bits,
        framing_bits=base.framing_bits + delta.framing_bits,
        residual_bits=base.residual_bits + delta.residual_bits,
        occurrence_bits=base.occurrence_bits + delta.occurrence_bits,
        candidate_occurrence_count=(
            base.candidate_occurrence_count + delta.candidate_occurrence_count
        ),
        selected_occurrence_count=(
            base.selected_occurrence_count + delta.selected_occurrence_count
        ),
        selected_motif_counts=selected_motif_counts,
    )
    failures = {failure.expression_id: failure for failure in base.failures}
    for expression_id, replacement in delta.failure_overrides.items():
        if replacement is None:
            failures.pop(expression_id, None)
        else:
            failures[expression_id] = replacement
    return summary, tuple(failures[expression_id] for expression_id in sorted(failures))


def build_exact_training_targets(
    candidate_pool: MotifVocabulary,
    partition: TrainPartition,
    *,
    candidate_prefilter: CandidatePrefilterAudit | None = None,
) -> TrainingTargetAudit:
    """Measure exact singleton MDL targets with sparse per-graph evaluation.

    A singleton vocabulary with no matching occurrence has the same conditional
    data code for every template: the codec never consults an unused dictionary
    entry.  We therefore cost that no-occurrence case once per graph, union-match
    the candidate pool once, and run exact compression/reconstruction only for
    motif IDs that actually occur.  Dictionary cost remains candidate-specific
    and is added exactly once when each complete two-part summary is finalized.
    """

    if not isinstance(candidate_pool, MotifVocabulary):
        raise TypeError("candidate_pool must be a MotifVocabulary")
    if not candidate_pool.templates:
        raise LearnedMotifProtocolError("exact target construction requires at least one candidate")
    if partition.batch.split is not CorpusSplit.TRAIN:
        raise LearnedMotifProtocolError("exact target construction requires a TRAIN partition")
    singletons = {
        template.motif_id: _selected_vocabulary(
            candidate_pool,
            (template,),
            budget=1,
        )
        for template in candidate_pool.templates
    }
    representative = singletons[candidate_pool.templates[0].motif_id]
    base = _MDLAccumulator(representative)
    deltas: dict[str, _SparseTargetDelta] = {}
    sparse_evaluation_count = 0

    for record_index, record in enumerate(partition.batch.records, start=1):
        if record_index == 1 or record_index % 256 == 0:
            _LOGGER.info(
                "constructing exact TRAIN targets from graph %d/%d",
                record_index,
                len(partition.batch.records),
            )
        prepared_graph = prepare_graph_mdl(record.graph)
        try:
            occurrences = find_vocabulary_occurrences(
                record.graph,
                candidate_pool,
            )
        except Exception as error:
            base.add(
                record.expression_id,
                _union_matching_failure_result(
                    record.graph,
                    error,
                    prepared_graph=prepared_graph,
                ),
            )
            continue

        base_result = motif_graph_mdl_result(
            record.graph,
            representative,
            occurrences=(),
            prepared_graph=prepared_graph,
        )
        base.add(record.expression_id, base_result)
        occurrences_by_motif: dict[str, list[MotifOccurrence]] = {}
        for occurrence in occurrences:
            occurrences_by_motif.setdefault(occurrence.motif_id, []).append(occurrence)
        for motif_id, motif_occurrences in occurrences_by_motif.items():
            observed = motif_graph_mdl_result(
                record.graph,
                singletons[motif_id],
                occurrences=tuple(motif_occurrences),
                prepared_graph=prepared_graph,
            )
            deltas.setdefault(motif_id, _SparseTargetDelta()).add(
                record.expression_id,
                base=base_result,
                observed=observed,
            )
            sparse_evaluation_count += 1

    _LOGGER.info(
        "completed exact TRAIN target graph pass (%d graphs)",
        len(partition.batch.records),
    )
    targets: list[ExactTrainingTarget] = []
    for template in candidate_pool.templates:
        summary, reconstruction_failures = _sparse_target_summary(
            vocabulary=singletons[template.motif_id],
            base=base,
            delta=deltas.get(template.motif_id),
        )
        targets.append(
            ExactTrainingTarget(
                example=SelectorTrainingExample(
                    motif_id=template.motif_id,
                    split=CorpusSplit.TRAIN,
                    mdl_gain_bits=summary.savings_bits,
                ),
                summary=summary,
                upstream_failures=partition.batch.failures,
                reconstruction_failures=reconstruction_failures,
            )
        )
    return TrainingTargetAudit(
        partition=partition,
        targets=tuple(targets),
        candidate_pool_digest=candidate_pool_digest(candidate_pool),
        candidate_prefilter=candidate_prefilter,
        union_match_pass_count=len(partition.batch.records),
        base_singleton_evaluation_count=len(partition.batch.records),
        sparse_singleton_evaluation_count=sparse_evaluation_count,
    )


@dataclass(frozen=True, slots=True)
class ValidationCandidate:
    """One ridge candidate measured only on VALIDATION."""

    selector: RidgeMotifSelector
    selector_payload_sha256: str
    arm: VocabularyArm
    evaluation: VocabularyEvaluation

    @property
    def eligible(self) -> bool:
        return self.evaluation.eligible


@dataclass(frozen=True, slots=True)
class LockedExperiment:
    """All fixed choices needed for held-out evaluation; contains no test data."""

    lock_digest: str
    config_digest: str
    implementation_digest: str
    candidate_pool: MotifVocabulary
    selected_frequent: MotifVocabulary
    selector: RidgeMotifSelector
    learned_arm: VocabularyArm
    frequent_arm: VocabularyArm
    random_arms: tuple[VocabularyArm, ...]
    validation_total_mdl_bits: int
    training_target_audit: TrainingTargetAudit
    frequent_provenance: FrequentSweepProvenance | None = None
    source_graph_cache_references: Mapping[CorpusSplit, ArtifactReference] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "random_arms", tuple(self.random_arms))
        if self.source_graph_cache_references is not None:
            if set(self.source_graph_cache_references) != {split for split in CorpusSplit}:
                raise ValueError("source graph-cache references must cover all splits")
            object.__setattr__(
                self,
                "source_graph_cache_references",
                MappingProxyType(dict(self.source_graph_cache_references)),
            )
        if (self.frequent_provenance is None) != (self.source_graph_cache_references is None):
            raise ValueError(
                "frequent provenance and source graph-cache references must be supplied together"
            )
        for name, value in (
            ("lock_digest", self.lock_digest),
            ("config_digest", self.config_digest),
            ("implementation_digest", self.implementation_digest),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        if len(self.random_arms) != 30:
            raise ValueError("locked experiment requires exactly thirty random arms")
        budgets = {
            self.learned_arm.budget,
            self.frequent_arm.budget,
            *(arm.budget for arm in self.random_arms),
        }
        if len(budgets) != 1:
            raise ValueError("all motif arms must use the same locked budget")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Train-target audit and validation-only model/baseline measurements."""

    candidates: tuple[ValidationCandidate, ...]
    selected_candidate_index: int
    frequent: VocabularyEvaluation
    random: tuple[VocabularyEvaluation, ...]
    macro: VocabularyEvaluation

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "random", tuple(self.random))
        if not self.candidates or any(
            not isinstance(candidate, ValidationCandidate) for candidate in self.candidates
        ):
            raise ValueError("validation report requires learned candidates")
        if not 0 <= self.selected_candidate_index < len(self.candidates):
            raise ValueError("selected validation candidate index is invalid")
        if len(self.random) != 30:
            raise ValueError("validation report requires thirty random baselines")
        lambdas = [candidate.selector.ridge_lambda for candidate in self.candidates]
        if lambdas != sorted(lambdas) or len(set(lambdas)) != len(lambdas):
            raise ValueError("validation ridge candidates must be unique and increasing")
        for candidate in self.candidates:
            if (
                candidate.selector_payload_sha256 != selector_payload_digest(candidate.selector)
                or candidate.arm.method is not ExperimentMethod.LEARNED
                or candidate.arm.ridge_lambda != candidate.selector.ridge_lambda
                or candidate.evaluation.split is not CorpusSplit.VALIDATION
                or candidate.evaluation.method is not ExperimentMethod.LEARNED
                or candidate.evaluation.arm_id != candidate.arm.arm_id
                or candidate.evaluation.budget != candidate.arm.budget
                or candidate.evaluation.vocabulary_id != candidate.arm.vocabulary.vocabulary_id
                or candidate.evaluation.vocabulary_payload_sha256
                != vocabulary_payload_digest(candidate.arm.vocabulary)
                or candidate.evaluation.ridge_lambda != candidate.selector.ridge_lambda
            ):
                raise ValueError("validation learned candidate binding is inconsistent")
        eligible_indexes = [
            index for index, candidate in enumerate(self.candidates) if candidate.eligible
        ]
        if not eligible_indexes:
            raise ValueError("validation report has no eligible learned candidate")
        expected_selected_index = min(
            eligible_indexes,
            key=lambda index: (
                self.candidates[index].evaluation.summary.total_mdl_bits,
                self.candidates[index].selector.ridge_lambda,
                self.candidates[index].selector_payload_sha256,
            ),
        )
        if self.selected_candidate_index != expected_selected_index:
            raise ValueError("validation selected candidate is not the deterministic optimum")
        reference = self.selected.evaluation
        motif_evaluations = (
            *(candidate.evaluation for candidate in self.candidates),
            self.frequent,
            *self.random,
        )
        if (
            self.frequent.split is not CorpusSplit.VALIDATION
            or self.frequent.method is not ExperimentMethod.FREQUENT
            or self.macro.split is not CorpusSplit.VALIDATION
            or self.macro.method is not ExperimentMethod.MACRO
            or any(
                evaluation.split is not CorpusSplit.VALIDATION
                or evaluation.method is not ExperimentMethod.RANDOM
                for evaluation in self.random
            )
            or any(
                evaluation.source_row_count != reference.source_row_count
                or evaluation.upstream_failures != reference.upstream_failures
                or evaluation.summary.processed_count != reference.summary.processed_count
                or evaluation.summary.baseline_dictionary_bits
                != reference.summary.baseline_dictionary_bits
                or evaluation.summary.baseline_data_bits != reference.summary.baseline_data_bits
                or evaluation.summary.baseline_total_bits != reference.summary.baseline_total_bits
                for evaluation in motif_evaluations
            )
        ):
            raise ValueError("validation arms do not share one source denominator")

    @property
    def selected(self) -> ValidationCandidate:
        return self.candidates[self.selected_candidate_index]


def _validate_graph_batches(
    batches: Iterable[SplitGraphBatch],
    candidate_pool: MotifVocabulary,
) -> None:
    observed_ids: set[str] = set()
    template = candidate_pool.templates[0]
    for batch in batches:
        for expression_id in (
            *(record.expression_id for record in batch.records),
            *(failure.expression_id for failure in batch.failures),
        ):
            if expression_id in observed_ids:
                raise LearnedMotifProtocolError(
                    f"expression {expression_id!r} appears in multiple splits"
                )
            observed_ids.add(expression_id)
        for record in batch.records:
            families = {node.family for node in record.graph.nodes.values()}
            modes = {root.representation_mode for root in record.graph.roots}
            if families != {template.source_family} or modes != {template.representation_mode}:
                raise LearnedMotifProtocolError(
                    f"graph {record.expression_id!r} does not match candidate family/mode"
                )


def _batch_expression_ids(batch: SplitGraphBatch) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                *(record.expression_id for record in batch.records),
                *(failure.expression_id for failure in batch.failures),
            )
        )
    )


def _lambda_arm_id(ridge_lambda: float) -> str:
    return f"learned:lambda={ridge_lambda.hex()}"


def _random_arms(
    candidate_pool: MotifVocabulary,
    *,
    budget: int,
    base_seed: int,
    repetitions: int,
) -> tuple[VocabularyArm, ...]:
    return tuple(
        VocabularyArm(
            arm_id=f"random:{repetition:02d}:seed={base_seed + repetition}",
            method=ExperimentMethod.RANDOM,
            vocabulary=_selected_vocabulary(
                candidate_pool,
                select_random_templates(
                    candidate_pool,
                    budget=budget,
                    seed=base_seed + repetition,
                ),
                budget=budget,
            ),
            budget=budget,
            random_seed=base_seed + repetition,
        )
        for repetition in range(repetitions)
    )


def fit_and_lock_selection(
    train_batch: SplitGraphBatch,
    validation_batch: SplitGraphBatch,
    *,
    candidate_pool: MotifVocabulary,
    selected_frequent: MotifVocabulary,
    config: Goal5LearnedMotifConfig,
    config_digest: str,
    implementation_digest: str | None = None,
    frequent_provenance: FrequentSweepProvenance | None = None,
    train_partition: TrainPartition | None = None,
    source_graph_cache_references: Mapping[CorpusSplit, ArtifactReference] | None = None,
) -> tuple[LockedExperiment, ValidationReport]:
    """Fit on TRAIN, choose λ on VALIDATION, and freeze every comparator."""

    if train_batch.split is not CorpusSplit.TRAIN:
        raise LearnedMotifProtocolError("train_batch must be TRAIN")
    if validation_batch.split is not CorpusSplit.VALIDATION:
        raise LearnedMotifProtocolError("validation_batch must be VALIDATION")
    if (
        not isinstance(config_digest, str)
        or len(config_digest) != 64
        or any(character not in "0123456789abcdef" for character in config_digest)
    ):
        raise ValueError("config_digest must be lowercase SHA-256 hex")
    if learned_config_digest(config) != config_digest:
        raise LearnedMotifProtocolError(
            "config_digest does not authenticate the supplied learned-motif config"
        )
    if frequent_provenance is not None and not isinstance(
        frequent_provenance,
        FrequentSweepProvenance,
    ):
        raise TypeError("frequent_provenance must be None or FrequentSweepProvenance")
    if source_graph_cache_references is not None and set(source_graph_cache_references) != {
        split for split in CorpusSplit
    }:
        raise ValueError("source_graph_cache_references must cover all four splits")
    if (frequent_provenance is None) != (source_graph_cache_references is None):
        raise ValueError(
            "frequent provenance and source graph-cache references must be supplied together"
        )
    resolved_implementation_digest = (
        learned_implementation_digest(_repository_root(Path(__file__)))
        if implementation_digest is None
        else implementation_digest
    )
    if (
        not isinstance(resolved_implementation_digest, str)
        or len(resolved_implementation_digest) != 64
        or any(character not in "0123456789abcdef" for character in resolved_implementation_digest)
    ):
        raise ValueError("implementation_digest must be lowercase SHA-256 hex")
    size_eligible_pool, budget = align_frequent_candidate_pool(
        candidate_pool,
        selected_frequent,
    )
    eligible_pool, candidate_prefilter = prefilter_train_candidates(
        size_eligible_pool,
        selected_frequent,
        maximum_candidate_motifs=(config.selector.maximum_candidate_motifs),
    )
    if not eligible_pool.templates:
        raise LearnedMotifProtocolError("eligible learned candidate pool is empty")
    _validate_graph_batches((train_batch, validation_batch), eligible_pool)
    if train_partition is None:
        partition = deterministic_train_partition(
            train_batch,
            fraction=config.selector.train_fit_fraction,
            seed=config.runtime.seed,
            maximum_graphs=config.selector.maximum_target_graphs,
        )
    else:
        if (
            not isinstance(train_partition, TrainPartition)
            or train_partition.batch != train_batch
            or train_partition.fraction != config.selector.train_fit_fraction
            or train_partition.seed != config.runtime.seed
            or train_partition.maximum_graphs != config.selector.maximum_target_graphs
        ):
            raise LearnedMotifProtocolError(
                "precomputed TRAIN partition does not match the locked selector protocol"
            )
        partition = train_partition
    target_audit = build_exact_training_targets(
        eligible_pool,
        partition,
        candidate_prefilter=candidate_prefilter,
    )
    if target_audit.total_failure_count:
        raise LearnedMotifProtocolError(
            "exact TRAIN target construction retained failures; no selector may be locked"
        )
    examples = tuple(target.example for target in target_audit.targets)

    selectors: list[RidgeMotifSelector] = []
    learned_arms: list[VocabularyArm] = []
    for ridge_lambda in config.selector.ridge_lambdas:
        selector = fit_ridge_selector(
            eligible_pool,
            examples,
            ridge_lambda=ridge_lambda,
            train_partition_digest=partition.digest,
            singular_value_rcond=config.selector.singular_value_rcond,
            training_seed=config.runtime.seed,
        )
        vocabulary = _selected_vocabulary(
            eligible_pool,
            select_learned_templates(eligible_pool, selector, budget=budget),
            budget=budget,
        )
        selectors.append(selector)
        learned_arms.append(
            VocabularyArm(
                arm_id=_lambda_arm_id(ridge_lambda),
                method=ExperimentMethod.LEARNED,
                vocabulary=vocabulary,
                budget=budget,
                ridge_lambda=ridge_lambda,
            )
        )

    frequent_arm = VocabularyArm(
        arm_id="frequent",
        method=ExperimentMethod.FREQUENT,
        vocabulary=selected_frequent,
        budget=budget,
    )
    random_arms = _random_arms(
        eligible_pool,
        budget=budget,
        base_seed=config.baselines.random_seed,
        repetitions=config.baselines.random_repetitions,
    )
    all_arms = (*learned_arms, frequent_arm, *random_arms)
    evaluations = evaluate_vocabulary_arms(
        validation_batch,
        all_arms,
        candidate_pool=eligible_pool,
    )
    evaluation_by_id = {evaluation.arm_id: evaluation for evaluation in evaluations}
    candidates = tuple(
        ValidationCandidate(
            selector=selector,
            selector_payload_sha256=selector_payload_digest(selector),
            arm=arm,
            evaluation=evaluation_by_id[arm.arm_id],
        )
        for selector, arm in zip(selectors, learned_arms, strict=True)
    )
    eligible_indexes = [index for index, candidate in enumerate(candidates) if candidate.eligible]
    if not eligible_indexes:
        raise LearnedMotifProtocolError(
            "no zero-failure learned validation candidate can be locked"
        )
    selected_index = min(
        eligible_indexes,
        key=lambda index: (
            candidates[index].evaluation.summary.total_mdl_bits,
            candidates[index].selector.ridge_lambda,
            candidates[index].selector_payload_sha256,
        ),
    )
    selected = candidates[selected_index]
    frequent_evaluation = evaluation_by_id[frequent_arm.arm_id]
    random_evaluations = tuple(evaluation_by_id[arm.arm_id] for arm in random_arms)
    macro = _macro_evaluation(validation_batch, selected.evaluation)

    lock_payload = {
        "budget": budget,
        "candidate_pool_digest": candidate_pool_digest(eligible_pool),
        "candidate_prefilter": _candidate_prefilter_payload(candidate_prefilter),
        "config_digest": config_digest,
        "frequent_vocabulary_sha256": vocabulary_payload_digest(selected_frequent),
        "frequent_sweep_provenance": (
            None
            if frequent_provenance is None
            else _frequent_provenance_payload(frequent_provenance)
        ),
        "implementation_digest": resolved_implementation_digest,
        "learned_vocabulary_sha256": vocabulary_payload_digest(selected.arm.vocabulary),
        "partition_digest": partition.digest,
        "random_arms": [
            {
                "seed": arm.random_seed,
                "vocabulary_sha256": vocabulary_payload_digest(arm.vocabulary),
            }
            for arm in random_arms
        ],
        "ridge_lambda": selected.selector.ridge_lambda,
        "selector_sha256": selected.selector_payload_sha256,
        "source_graph_caches": (
            None
            if source_graph_cache_references is None
            else {
                split.value: _ref_payload(source_graph_cache_references[split])
                for split in CorpusSplit
            }
        ),
        "target_computation": {
            "base_singleton_evaluation_count": (target_audit.base_singleton_evaluation_count),
            "sparse_singleton_evaluation_count": (target_audit.sparse_singleton_evaluation_count),
            "union_match_pass_count": target_audit.union_match_pass_count,
        },
        "validation_total_mdl_bits": selected.evaluation.summary.total_mdl_bits,
        "version": LEARNED_LOCK_VERSION,
    }
    locked = LockedExperiment(
        lock_digest=_sha256_bytes(_canonical_json_bytes(lock_payload)),
        config_digest=config_digest,
        implementation_digest=resolved_implementation_digest,
        candidate_pool=eligible_pool,
        selected_frequent=selected_frequent,
        selector=selected.selector,
        learned_arm=selected.arm,
        frequent_arm=frequent_arm,
        random_arms=random_arms,
        validation_total_mdl_bits=selected.evaluation.summary.total_mdl_bits,
        training_target_audit=target_audit,
        frequent_provenance=frequent_provenance,
        source_graph_cache_references=source_graph_cache_references,
    )
    return locked, ValidationReport(
        candidates=candidates,
        selected_candidate_index=selected_index,
        frequent=frequent_evaluation,
        random=random_evaluations,
        macro=macro,
    )


class HeldoutClaimStatus(StrEnum):
    """Post-hoc report status; never changes the locked vocabulary."""

    SUPPORTED = "supported"
    NULL_RESULT = "null_result"


@dataclass(frozen=True, slots=True)
class HeldoutSplitReport:
    """One held-out split evaluated exactly once with fixed arms."""

    split: CorpusSplit
    learned: VocabularyEvaluation
    frequent: VocabularyEvaluation
    random: tuple[VocabularyEvaluation, ...]
    macro: VocabularyEvaluation
    random_median_total_mdl_bits: float
    learned_beats_all_baselines: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "random", tuple(self.random))
        if self.split not in {CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD}:
            raise ValueError("held-out report requires TEST_IID or TEST_OOD")
        if len(self.random) != 30:
            raise ValueError("held-out report requires thirty random baselines")
        if not math.isfinite(self.random_median_total_mdl_bits):
            raise ValueError("random median must be finite")
        if not isinstance(self.learned_beats_all_baselines, bool):
            raise TypeError("learned_beats_all_baselines must be a boolean")
        if (
            self.learned.split is not self.split
            or self.frequent.split is not self.split
            or self.macro.split is not self.split
            or any(evaluation.split is not self.split for evaluation in self.random)
        ):
            raise ValueError("held-out evaluations must match the report split")
        if (
            self.learned.method is not ExperimentMethod.LEARNED
            or self.frequent.method is not ExperimentMethod.FREQUENT
            or self.macro.method is not ExperimentMethod.MACRO
            or any(evaluation.method is not ExperimentMethod.RANDOM for evaluation in self.random)
        ):
            raise ValueError("held-out evaluation methods are inconsistent")
        median = float(
            statistics.median(evaluation.summary.total_mdl_bits for evaluation in self.random)
        )
        if self.random_median_total_mdl_bits != median:
            raise ValueError("random median is inconsistent")
        all_exact = (
            self.learned.eligible
            and self.frequent.eligible
            and self.macro.eligible
            and all(evaluation.eligible for evaluation in self.random)
        )
        beats = (
            all_exact
            and self.learned.summary.total_mdl_bits < self.frequent.summary.total_mdl_bits
            and self.learned.summary.total_mdl_bits < median
            and self.learned.summary.total_mdl_bits < self.macro.summary.total_mdl_bits
        )
        if self.learned_beats_all_baselines is not beats:
            raise ValueError("held-out baseline comparison is inconsistent")


def evaluate_locked_heldout(
    locked: LockedExperiment,
    batch: SplitGraphBatch,
) -> HeldoutSplitReport:
    """Evaluate fixed learned/frequent/random arms without modifying the lock."""

    if batch.split not in {CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD}:
        raise LearnedMotifProtocolError("held-out evaluation requires a test split")
    _validate_graph_batches((batch,), locked.candidate_pool)
    arms = (locked.learned_arm, locked.frequent_arm, *locked.random_arms)
    evaluations = evaluate_vocabulary_arms(
        batch,
        arms,
        candidate_pool=locked.candidate_pool,
    )
    by_id = {evaluation.arm_id: evaluation for evaluation in evaluations}
    learned = by_id[locked.learned_arm.arm_id]
    frequent = by_id[locked.frequent_arm.arm_id]
    random = tuple(by_id[arm.arm_id] for arm in locked.random_arms)
    macro = _macro_evaluation(batch, learned)
    random_median = float(
        statistics.median(evaluation.summary.total_mdl_bits for evaluation in random)
    )
    all_exact = (
        learned.eligible
        and frequent.eligible
        and macro.eligible
        and all(evaluation.eligible for evaluation in random)
    )
    beats = (
        all_exact
        and learned.summary.total_mdl_bits < frequent.summary.total_mdl_bits
        and learned.summary.total_mdl_bits < random_median
        and learned.summary.total_mdl_bits < macro.summary.total_mdl_bits
    )
    report = HeldoutSplitReport(
        split=batch.split,
        learned=learned,
        frequent=frequent,
        random=random,
        macro=macro,
        random_median_total_mdl_bits=random_median,
        learned_beats_all_baselines=beats,
    )
    _validate_heldout_against_locked(report, locked)
    return report


def _validate_evaluation_arm_binding(
    evaluation: VocabularyEvaluation,
    arm: VocabularyArm,
) -> None:
    if (
        evaluation.arm_id != arm.arm_id
        or evaluation.method is not arm.method
        or evaluation.budget != arm.budget
        or evaluation.vocabulary_id != arm.vocabulary.vocabulary_id
        or evaluation.vocabulary_payload_sha256 != vocabulary_payload_digest(arm.vocabulary)
        or evaluation.ridge_lambda != arm.ridge_lambda
        or evaluation.random_seed != arm.random_seed
        or evaluation.summary.dictionary_bits != vocabulary_mdl_bits(arm.vocabulary.templates)
        or not {motif_id for motif_id, _count in evaluation.summary.selected_motif_counts}.issubset(
            {template.motif_id for template in arm.vocabulary.templates}
        )
    ):
        raise LearnedMotifProtocolError(
            f"held-out evaluation is not bound to locked arm {arm.arm_id!r}"
        )


def _validate_heldout_against_locked(
    report: HeldoutSplitReport,
    locked: LockedExperiment,
) -> None:
    _validate_evaluation_arm_binding(report.learned, locked.learned_arm)
    _validate_evaluation_arm_binding(report.frequent, locked.frequent_arm)
    for evaluation, arm in zip(
        report.random,
        locked.random_arms,
        strict=True,
    ):
        _validate_evaluation_arm_binding(evaluation, arm)
    _validate_shared_evaluation_evidence(
        split=report.split,
        reference=report.learned,
        motif_evaluations=(report.learned, report.frequent, *report.random),
        macro=report.macro,
    )


def _validate_shared_evaluation_evidence(
    *,
    split: CorpusSplit,
    reference: VocabularyEvaluation,
    motif_evaluations: Sequence[VocabularyEvaluation],
    macro: VocabularyEvaluation,
) -> None:
    motif_evaluations = (*motif_evaluations,)
    if any(
        evaluation.source_row_count != reference.source_row_count
        or evaluation.upstream_failures != reference.upstream_failures
        or evaluation.summary.processed_count != reference.summary.processed_count
        or evaluation.summary.baseline_dictionary_bits != reference.summary.baseline_dictionary_bits
        or evaluation.summary.baseline_data_bits != reference.summary.baseline_data_bits
        or evaluation.summary.baseline_total_bits != reference.summary.baseline_total_bits
        for evaluation in motif_evaluations
    ):
        raise LearnedMotifProtocolError(
            f"{split.value} arms do not share one exact source denominator"
        )
    expected_macro = _macro_evaluation(
        SplitGraphBatch(
            split=split,
            records=(),
            failures=reference.upstream_failures,
        ),
        reference,
    )
    if macro != expected_macro:
        raise LearnedMotifProtocolError(
            f"{split.value} macro baseline is inconsistent with the source graphs"
        )


def _validate_evaluation_against_lock_record(
    evaluation: VocabularyEvaluation,
    *,
    arm_id: object,
    method: ExperimentMethod,
    budget: int,
    vocabulary_id: object,
    vocabulary_payload_sha256: object,
    ridge_lambda: float | None,
    random_seed: int | None,
    motif_ids: Sequence[str] | None = None,
    dictionary_bits: int | None = None,
) -> None:
    if (
        evaluation.arm_id != arm_id
        or evaluation.method is not method
        or evaluation.budget != budget
        or evaluation.vocabulary_id != vocabulary_id
        or evaluation.vocabulary_payload_sha256 != vocabulary_payload_sha256
        or evaluation.ridge_lambda != ridge_lambda
        or evaluation.random_seed != random_seed
        or (
            motif_ids is not None
            and not {
                motif_id for motif_id, _count in evaluation.summary.selected_motif_counts
            }.issubset(set(motif_ids))
        )
        or (dictionary_bits is not None and evaluation.summary.dictionary_bits != dictionary_bits)
    ):
        raise LearnedMotifProtocolError(
            f"{evaluation.split.value} evaluation is inconsistent with locked {method.value} arm"
        )


def _validate_report_against_lock_payload(
    report: HeldoutSplitReport,
    *,
    lock: Mapping[str, object],
    budget: int,
    learned_vocabulary: MotifVocabulary,
    frequent_vocabulary: MotifVocabulary,
    random_dictionary_bits: Mapping[int, int] | None = None,
) -> None:
    _validate_evaluation_against_lock_record(
        report.learned,
        arm_id=lock.get("learned_arm_id"),
        method=ExperimentMethod.LEARNED,
        budget=budget,
        vocabulary_id=lock.get("learned_vocabulary_id"),
        vocabulary_payload_sha256=lock.get("learned_vocabulary_payload_sha256"),
        ridge_lambda=lock.get("ridge_lambda"),
        random_seed=None,
        motif_ids=tuple(template.motif_id for template in learned_vocabulary.templates),
        dictionary_bits=vocabulary_mdl_bits(learned_vocabulary.templates),
    )
    _validate_evaluation_against_lock_record(
        report.frequent,
        arm_id=lock.get("frequent_arm_id"),
        method=ExperimentMethod.FREQUENT,
        budget=budget,
        vocabulary_id=lock.get("frequent_vocabulary_id"),
        vocabulary_payload_sha256=lock.get("frequent_vocabulary_payload_sha256"),
        ridge_lambda=None,
        random_seed=None,
        motif_ids=tuple(template.motif_id for template in frequent_vocabulary.templates),
        dictionary_bits=vocabulary_mdl_bits(frequent_vocabulary.templates),
    )
    raw_random_arms = lock.get("random_arms")
    if not isinstance(raw_random_arms, list) or len(raw_random_arms) != len(report.random):
        raise LearnedMotifProtocolError("locked random arms are inconsistent")
    for evaluation, arm in zip(report.random, raw_random_arms, strict=True):
        if not isinstance(arm, dict):
            raise LearnedMotifProtocolError("locked random arm must be an object")
        _validate_evaluation_against_lock_record(
            evaluation,
            arm_id=arm.get("arm_id"),
            method=ExperimentMethod.RANDOM,
            budget=budget,
            vocabulary_id=arm.get("vocabulary_id"),
            vocabulary_payload_sha256=arm.get("vocabulary_payload_sha256"),
            ridge_lambda=None,
            random_seed=arm.get("seed"),
            motif_ids=arm.get("motif_ids"),
            dictionary_bits=(
                None
                if random_dictionary_bits is None
                else random_dictionary_bits.get(arm.get("seed"))
            ),
        )
    _validate_shared_evaluation_evidence(
        split=report.split,
        reference=report.learned,
        motif_evaluations=(report.learned, report.frequent, *report.random),
        macro=report.macro,
    )


@dataclass(frozen=True, slots=True)
class LearnedMotifExperimentResult:
    """Complete result with a lock that is independent of held-out outcomes."""

    locked: LockedExperiment
    validation: ValidationReport
    test_iid: HeldoutSplitReport
    test_ood: HeldoutSplitReport
    claim_status: HeldoutClaimStatus
    claim_message: str

    def __post_init__(self) -> None:
        expected = (
            HeldoutClaimStatus.SUPPORTED
            if self.test_iid.learned_beats_all_baselines
            else HeldoutClaimStatus.NULL_RESULT
        )
        if self.claim_status is not expected:
            raise ValueError("claim_status must report the locked TEST_IID comparison")


def _claim_from_iid(
    report: HeldoutSplitReport,
) -> tuple[HeldoutClaimStatus, str]:
    if not isinstance(report, HeldoutSplitReport) or (report.split is not CorpusSplit.TEST_IID):
        raise ValueError("claim reporting requires a TEST_IID report")
    status = (
        HeldoutClaimStatus.SUPPORTED
        if report.learned_beats_all_baselines
        else HeldoutClaimStatus.NULL_RESULT
    )
    message = (
        "Locked learned motifs beat the equal-budget frequent vocabulary, "
        "the median of 30 fixed random vocabularies, and the macro baseline "
        "on TEST_IID with zero retained failures."
        if status is HeldoutClaimStatus.SUPPORTED
        else "Null result: the locked learned vocabulary did not beat every "
        "required TEST_IID baseline with complete zero-failure evidence."
    )
    return status, message


def _summary_payload(summary: SplitMDLSummary) -> dict[str, object]:
    return {
        "baseline_data_bits": summary.baseline_data_bits,
        "baseline_dictionary_bits": summary.baseline_dictionary_bits,
        "baseline_total_bits": summary.baseline_total_bits,
        "candidate_occurrence_count": summary.candidate_occurrence_count,
        "conditional_data_bits": summary.conditional_data_bits,
        "dictionary_bits": summary.dictionary_bits,
        "framing_bits": summary.framing_bits,
        "occurrence_bits": summary.occurrence_bits,
        "processed_count": summary.processed_count,
        "reconstruction_failure_count": summary.reconstruction_failure_count,
        "residual_bits": summary.residual_bits,
        "savings_bits": summary.savings_bits,
        "selected_motif_counts": [
            {"count": count, "motif_id": motif_id}
            for motif_id, count in summary.selected_motif_counts
        ],
        "selected_occurrence_count": summary.selected_occurrence_count,
        "success_count": summary.success_count,
        "total_mdl_bits": summary.total_mdl_bits,
    }


def _evaluation_payload(evaluation: VocabularyEvaluation) -> dict[str, object]:
    return {
        "arm_id": evaluation.arm_id,
        "budget": evaluation.budget,
        "method": evaluation.method.value,
        "random_seed": evaluation.random_seed,
        "reconstruction_failures": [
            {
                "attempted_selected_occurrence_count": (
                    failure.attempted_selected_occurrence_count
                ),
                "error_message": failure.error_message,
                "error_type": failure.error_type,
                "expression_id": failure.expression_id,
            }
            for failure in evaluation.reconstruction_failures
        ],
        "ridge_lambda": evaluation.ridge_lambda,
        "source_row_count": evaluation.source_row_count,
        "split": evaluation.split.value,
        "summary": _summary_payload(evaluation.summary),
        "total_failure_count": evaluation.total_failure_count,
        "upstream_failures": [
            {
                "error_message": failure.error_message,
                "error_type": failure.error_type,
                "expression_id": failure.expression_id,
                "stage": failure.stage,
            }
            for failure in evaluation.upstream_failures
        ],
        "vocabulary_id": evaluation.vocabulary_id,
        "vocabulary_payload_sha256": evaluation.vocabulary_payload_sha256,
    }


def _target_payload(audit: TrainingTargetAudit) -> dict[str, object]:
    prefilter = audit.candidate_prefilter
    return {
        "candidate_pool_digest": audit.candidate_pool_digest,
        "candidate_prefilter": (
            None if prefilter is None else _candidate_prefilter_payload(prefilter)
        ),
        "computation": {
            "base_singleton_evaluation_count": (audit.base_singleton_evaluation_count),
            "brute_force_evaluation_count": (
                len(audit.partition.batch.records) * len(audit.targets)
            ),
            "sparse_singleton_evaluation_count": (audit.sparse_singleton_evaluation_count),
            "union_match_pass_count": audit.union_match_pass_count,
        },
        "partition": {
            "digest": audit.partition.digest,
            "fraction": audit.partition.fraction,
            "fractional_graph_count": audit.partition.fractional_graph_count,
            "full_train_graph_count": audit.partition.full_train_graph_count,
            "maximum_graphs": audit.partition.maximum_graphs,
            "selected_graph_count": len(audit.partition.batch.records),
            "seed": audit.partition.seed,
            "source_expression_count": len(audit.partition.source_expression_ids),
            "source_identity_digest": audit.partition.source_identity_digest,
            "upstream_failure_count": len(audit.partition.batch.failures),
            "upstream_failures": [
                {
                    "error_message": failure.error_message,
                    "error_type": failure.error_type,
                    "expression_id": failure.expression_id,
                    "stage": failure.stage,
                }
                for failure in audit.partition.batch.failures
            ],
        },
        "schema_version": "geml-goal5-exact-training-targets-v1",
        "targets": [
            {
                "mdl_gain_bits": target.example.mdl_gain_bits,
                "motif_id": target.example.motif_id,
                "reconstruction_failure_count": len(target.reconstruction_failures),
                "reconstruction_failures": [
                    {
                        "attempted_selected_occurrence_count": (
                            failure.attempted_selected_occurrence_count
                        ),
                        "error_message": failure.error_message,
                        "error_type": failure.error_type,
                        "expression_id": failure.expression_id,
                    }
                    for failure in target.reconstruction_failures
                ],
                "source_row_count": (
                    target.summary.processed_count + len(target.upstream_failures)
                ),
                "split": target.example.split.value,
                "summary": _summary_payload(target.summary),
                "upstream_failure_count": len(target.upstream_failures),
            }
            for target in audit.targets
        ],
        "total_failure_count": audit.total_failure_count,
    }


def _validation_payload(report: ValidationReport) -> dict[str, object]:
    return {
        "candidates": [
            {
                "evaluation": _evaluation_payload(candidate.evaluation),
                "selector_payload_sha256": candidate.selector_payload_sha256,
            }
            for candidate in report.candidates
        ],
        "frequent": _evaluation_payload(report.frequent),
        "macro": _evaluation_payload(report.macro),
        "random": [_evaluation_payload(evaluation) for evaluation in report.random],
        "schema_version": "geml-goal5-learned-validation-results-v1",
        "selected_candidate_index": report.selected_candidate_index,
    }


def _heldout_payload(report: HeldoutSplitReport) -> dict[str, object]:
    return {
        "frequent": _evaluation_payload(report.frequent),
        "learned": _evaluation_payload(report.learned),
        "learned_beats_all_baselines": report.learned_beats_all_baselines,
        "macro": _evaluation_payload(report.macro),
        "random": [_evaluation_payload(evaluation) for evaluation in report.random],
        "random_median_total_mdl_bits": report.random_median_total_mdl_bits,
        "split": report.split.value,
    }


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LearnedMotifProtocolError(f"{label} must be an exact integer >= {minimum}")
    return value


def _exact_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise LearnedMotifProtocolError(f"{label} must be an exact finite float")
    return value


def _exact_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LearnedMotifProtocolError(f"{label} must be a nonblank string")
    return value


def _require_canonical_payload(
    observed: object,
    expected: object,
    *,
    label: str,
) -> None:
    if _canonical_json_bytes(observed) != _canonical_json_bytes(expected):
        raise LearnedMotifProtocolError(f"{label} is not canonical")


def _require_exact_artifact(
    data: bytes,
    expected: object,
    *,
    label: str,
) -> None:
    if data != _canonical_json_bytes(expected) + b"\n":
        raise LearnedMotifProtocolError(f"{label} differs from deterministic replay")


def _summary_from_payload(payload: object) -> SplitMDLSummary:
    expected = {
        "baseline_data_bits",
        "baseline_dictionary_bits",
        "baseline_total_bits",
        "candidate_occurrence_count",
        "conditional_data_bits",
        "dictionary_bits",
        "framing_bits",
        "occurrence_bits",
        "processed_count",
        "reconstruction_failure_count",
        "residual_bits",
        "savings_bits",
        "selected_motif_counts",
        "selected_occurrence_count",
        "success_count",
        "total_mdl_bits",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LearnedMotifProtocolError("held-out summary fields do not match the frozen schema")
    raw_counts = payload["selected_motif_counts"]
    if not isinstance(raw_counts, list):
        raise LearnedMotifProtocolError("selected_motif_counts must be a list")
    counts: list[tuple[str, int]] = []
    for item in raw_counts:
        if not isinstance(item, dict) or set(item) != {"count", "motif_id"}:
            raise LearnedMotifProtocolError("selected motif count entries are invalid")
        counts.append(
            (
                _exact_string(
                    item["motif_id"],
                    label="selected motif count motif_id",
                ),
                _exact_int(
                    item["count"],
                    label="selected motif count",
                    minimum=1,
                ),
            )
        )
    if counts != sorted(counts) or len({motif_id for motif_id, _ in counts}) != len(counts):
        raise LearnedMotifProtocolError(
            "selected motif counts must be unique and canonically ordered"
        )
    try:
        summary = SplitMDLSummary(
            processed_count=_exact_int(
                payload["processed_count"],
                label="summary processed_count",
            ),
            success_count=_exact_int(
                payload["success_count"],
                label="summary success_count",
            ),
            reconstruction_failure_count=_exact_int(
                payload["reconstruction_failure_count"],
                label="summary reconstruction_failure_count",
            ),
            baseline_dictionary_bits=_exact_int(
                payload["baseline_dictionary_bits"],
                label="summary baseline_dictionary_bits",
            ),
            baseline_data_bits=_exact_int(
                payload["baseline_data_bits"],
                label="summary baseline_data_bits",
            ),
            baseline_total_bits=_exact_int(
                payload["baseline_total_bits"],
                label="summary baseline_total_bits",
            ),
            dictionary_bits=_exact_int(
                payload["dictionary_bits"],
                label="summary dictionary_bits",
            ),
            conditional_data_bits=_exact_int(
                payload["conditional_data_bits"],
                label="summary conditional_data_bits",
            ),
            total_mdl_bits=_exact_int(
                payload["total_mdl_bits"],
                label="summary total_mdl_bits",
            ),
            framing_bits=_exact_int(
                payload["framing_bits"],
                label="summary framing_bits",
            ),
            residual_bits=_exact_int(
                payload["residual_bits"],
                label="summary residual_bits",
            ),
            occurrence_bits=_exact_int(
                payload["occurrence_bits"],
                label="summary occurrence_bits",
            ),
            candidate_occurrence_count=_exact_int(
                payload["candidate_occurrence_count"],
                label="summary candidate_occurrence_count",
            ),
            selected_occurrence_count=_exact_int(
                payload["selected_occurrence_count"],
                label="summary selected_occurrence_count",
            ),
            selected_motif_counts=tuple(counts),
        )
    except (TypeError, ValueError) as error:
        raise LearnedMotifProtocolError("held-out summary failed invariant validation") from error
    if payload["savings_bits"] != summary.savings_bits:
        raise LearnedMotifProtocolError("held-out summary savings_bits is inconsistent")
    _require_canonical_payload(
        payload,
        _summary_payload(summary),
        label="held-out summary payload",
    )
    return summary


def _evaluation_from_payload(payload: object) -> VocabularyEvaluation:
    expected = {
        "arm_id",
        "budget",
        "method",
        "random_seed",
        "reconstruction_failures",
        "ridge_lambda",
        "source_row_count",
        "split",
        "summary",
        "total_failure_count",
        "upstream_failures",
        "vocabulary_id",
        "vocabulary_payload_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LearnedMotifProtocolError("held-out evaluation fields do not match the frozen schema")
    raw_upstream = payload["upstream_failures"]
    raw_reconstruction = payload["reconstruction_failures"]
    if not isinstance(raw_upstream, list) or not isinstance(
        raw_reconstruction,
        list,
    ):
        raise LearnedMotifProtocolError("held-out failures must be persisted as lists")
    try:
        split = CorpusSplit(payload["split"])
        method = ExperimentMethod(payload["method"])
        upstream = tuple(
            GraphInputFailure(
                expression_id=_exact_string(
                    item["expression_id"],
                    label="upstream failure expression_id",
                ),
                split=split,
                stage=_exact_string(
                    item["stage"],
                    label="upstream failure stage",
                ),
                error_type=_exact_string(
                    item["error_type"],
                    label="upstream failure error_type",
                ),
                error_message=_exact_string(
                    item["error_message"],
                    label="upstream failure error_message",
                ),
            )
            for item in raw_upstream
            if isinstance(item, dict)
            and set(item)
            == {
                "error_message",
                "error_type",
                "expression_id",
                "stage",
            }
        )
        reconstruction = tuple(
            EvaluationFailure(
                expression_id=_exact_string(
                    item["expression_id"],
                    label="reconstruction failure expression_id",
                ),
                error_type=_exact_string(
                    item["error_type"],
                    label="reconstruction failure error_type",
                ),
                error_message=_exact_string(
                    item["error_message"],
                    label="reconstruction failure error_message",
                ),
                attempted_selected_occurrence_count=_exact_int(
                    item["attempted_selected_occurrence_count"],
                    label="attempted selected occurrence count",
                ),
            )
            for item in raw_reconstruction
            if isinstance(item, dict)
            and set(item)
            == {
                "attempted_selected_occurrence_count",
                "error_message",
                "error_type",
                "expression_id",
            }
        )
        if len(upstream) != len(raw_upstream) or len(reconstruction) != len(raw_reconstruction):
            raise ValueError("failure record fields are invalid")
        evaluation = VocabularyEvaluation(
            split=split,
            arm_id=_exact_string(payload["arm_id"], label="evaluation arm_id"),
            method=method,
            budget=_exact_int(payload["budget"], label="evaluation budget"),
            vocabulary_id=payload["vocabulary_id"],
            vocabulary_payload_sha256=payload["vocabulary_payload_sha256"],
            summary=_summary_from_payload(payload["summary"]),
            upstream_failures=upstream,
            reconstruction_failures=reconstruction,
            ridge_lambda=(
                None
                if payload["ridge_lambda"] is None
                else _exact_float(
                    payload["ridge_lambda"],
                    label="evaluation ridge_lambda",
                )
            ),
            random_seed=(
                None
                if payload["random_seed"] is None
                else _exact_int(
                    payload["random_seed"],
                    label="evaluation random_seed",
                )
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LearnedMotifProtocolError(
            "held-out evaluation failed invariant validation"
        ) from error
    if payload["source_row_count"] != evaluation.source_row_count:
        raise LearnedMotifProtocolError("held-out source row denominator is inconsistent")
    if payload["total_failure_count"] != evaluation.total_failure_count:
        raise LearnedMotifProtocolError("held-out total failure count is inconsistent")
    _require_canonical_payload(
        payload,
        _evaluation_payload(evaluation),
        label="held-out evaluation payload",
    )
    return evaluation


def _heldout_from_payload(payload: object) -> HeldoutSplitReport:
    expected = {
        "frequent",
        "learned",
        "learned_beats_all_baselines",
        "macro",
        "random",
        "random_median_total_mdl_bits",
        "split",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LearnedMotifProtocolError("held-out report fields do not match the frozen schema")
    raw_random = payload["random"]
    if not isinstance(raw_random, list):
        raise LearnedMotifProtocolError("held-out random evaluations must be a list")
    try:
        report = HeldoutSplitReport(
            split=CorpusSplit(payload["split"]),
            learned=_evaluation_from_payload(payload["learned"]),
            frequent=_evaluation_from_payload(payload["frequent"]),
            random=tuple(_evaluation_from_payload(item) for item in raw_random),
            macro=_evaluation_from_payload(payload["macro"]),
            random_median_total_mdl_bits=_exact_float(
                payload["random_median_total_mdl_bits"],
                label="held-out random median",
            ),
            learned_beats_all_baselines=payload["learned_beats_all_baselines"],
        )
    except (TypeError, ValueError) as error:
        raise LearnedMotifProtocolError("held-out report failed invariant validation") from error
    if (
        report.learned.split is not report.split
        or report.frequent.split is not report.split
        or report.macro.split is not report.split
        or any(evaluation.split is not report.split for evaluation in report.random)
    ):
        raise LearnedMotifProtocolError("held-out report contains a split mismatch")
    if (
        report.learned.method is not ExperimentMethod.LEARNED
        or report.frequent.method is not ExperimentMethod.FREQUENT
        or report.macro.method is not ExperimentMethod.MACRO
        or any(evaluation.method is not ExperimentMethod.RANDOM for evaluation in report.random)
    ):
        raise LearnedMotifProtocolError("held-out report contains a method mismatch")
    recomputed_median = statistics.median(
        evaluation.summary.total_mdl_bits for evaluation in report.random
    )
    if report.random_median_total_mdl_bits != recomputed_median:
        raise LearnedMotifProtocolError("held-out random median is inconsistent")
    _require_canonical_payload(
        payload,
        _heldout_payload(report),
        label="held-out report payload",
    )
    return report


def _write_immutable_bytes(data: bytes, path: Path, *, resume: bool) -> None:
    if not isinstance(resume, bool):
        raise ValueError("resume must be a boolean")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-learned-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not resume:
                raise FileExistsError(f"immutable artifact already exists: {path}") from None
            if not path.is_file() or path.read_bytes() != data:
                raise LearnedMotifProtocolError(
                    f"existing artifact differs from resumed output: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_artifact(
    payload: object,
    path: Path,
    *,
    root: Path,
    resume: bool,
) -> ArtifactReference:
    data = _canonical_json_bytes(payload) + b"\n"
    _write_immutable_bytes(data, path, resume=resume)
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return ArtifactReference(path=relative, sha256=_sha256_bytes(data))


def _ref_payload(reference: ArtifactReference) -> dict[str, str]:
    return {"path": reference.path, "sha256": reference.sha256}


@dataclass(frozen=True, slots=True)
class LockedArtifactPaths:
    """Artifacts guaranteed to exist before any held-out loader call."""

    selector: ArtifactReference
    training_targets: ArtifactReference
    learned_vocabulary: ArtifactReference
    frequent_vocabulary: ArtifactReference
    validation_results: ArtifactReference
    selection_lock: ArtifactReference


@dataclass(frozen=True, slots=True)
class HeldoutArtifactPaths:
    """Crash-safe immutable receipts written immediately after each test split."""

    test_iid: ArtifactReference
    test_ood: ArtifactReference


def _selection_lock_payload(
    locked: LockedExperiment,
    *,
    selector: ArtifactReference,
    training_targets: ArtifactReference,
    learned_vocabulary: ArtifactReference,
    frequent_vocabulary: ArtifactReference,
    validation_results: ArtifactReference,
) -> dict[str, object]:
    """Build the one canonical persisted selection-lock payload."""

    return {
        "artifact_version": LEARNED_ARTIFACT_VERSION,
        "artifacts": {
            "frequent_vocabulary": _ref_payload(frequent_vocabulary),
            "learned_vocabulary": _ref_payload(learned_vocabulary),
            "selector": _ref_payload(selector),
            "training_targets": _ref_payload(training_targets),
            "validation_results": _ref_payload(validation_results),
        },
        "budget": locked.learned_arm.budget,
        "candidate_pool_digest": candidate_pool_digest(locked.candidate_pool),
        "candidate_prefilter": (
            _candidate_prefilter_payload(locked.training_target_audit.candidate_prefilter)
            if locked.training_target_audit.candidate_prefilter is not None
            else None
        ),
        "config_digest": locked.config_digest,
        "frequent_sweep_provenance": (
            None
            if locked.frequent_provenance is None
            else _frequent_provenance_payload(locked.frequent_provenance)
        ),
        "heldout_artifacts_absent_at_lock": True,
        "implementation_digest": locked.implementation_digest,
        "frequent_arm_id": locked.frequent_arm.arm_id,
        "frequent_vocabulary_id": locked.selected_frequent.vocabulary_id,
        "frequent_vocabulary_payload_sha256": vocabulary_payload_digest(locked.selected_frequent),
        "learned_arm_id": locked.learned_arm.arm_id,
        "learned_vocabulary_id": locked.learned_arm.vocabulary.vocabulary_id,
        "learned_vocabulary_payload_sha256": vocabulary_payload_digest(
            locked.learned_arm.vocabulary
        ),
        "lock_digest": locked.lock_digest,
        "random_arms": [
            {
                "arm_id": arm.arm_id,
                "motif_ids": [template.motif_id for template in arm.vocabulary.templates],
                "seed": arm.random_seed,
                "vocabulary_id": arm.vocabulary.vocabulary_id,
                "vocabulary_payload_sha256": vocabulary_payload_digest(arm.vocabulary),
            }
            for arm in locked.random_arms
        ],
        "ridge_lambda": locked.selector.ridge_lambda,
        "schema_version": LEARNED_LOCK_VERSION,
        "selector_payload_sha256": selector_payload_digest(locked.selector),
        "source_graph_caches": (
            None
            if locked.source_graph_cache_references is None
            else {
                split.value: _ref_payload(locked.source_graph_cache_references[split])
                for split in CorpusSplit
            }
        ),
        "target_computation": {
            "base_singleton_evaluation_count": (
                locked.training_target_audit.base_singleton_evaluation_count
            ),
            "sparse_singleton_evaluation_count": (
                locked.training_target_audit.sparse_singleton_evaluation_count
            ),
            "union_match_pass_count": (locked.training_target_audit.union_match_pass_count),
        },
        "train_partition_digest": locked.training_target_audit.partition.digest,
        "validation_total_mdl_bits": locked.validation_total_mdl_bits,
    }


def persist_locked_experiment(
    locked: LockedExperiment,
    validation: ValidationReport,
    output_dir: str | Path,
    *,
    resume: bool,
) -> LockedArtifactPaths:
    """Persist every selected choice, publishing the lock marker last."""

    root = Path(output_dir)
    lock_path = root / "selection.lock.json"
    if not lock_path.exists():
        forbidden_before_lock = (
            root / "heldout.test_iid.json",
            root / "heldout.test_ood.json",
            root / "heldout_results.json",
            root / "experiment.result.json",
            root / "run.complete.json",
        )
        stale = [path.name for path in forbidden_before_lock if path.exists()]
        if stale:
            raise LearnedMotifProtocolError(
                "held-out/completion artifacts exist before selection lock: "
                + ", ".join(sorted(stale))
            )
    selector_ref = _write_json_artifact(
        selector_payload(locked.selector),
        root / "selector.json",
        root=root,
        resume=resume,
    )
    targets_ref = _write_json_artifact(
        _target_payload(locked.training_target_audit),
        root / "training_targets.json",
        root=root,
        resume=resume,
    )
    learned_ref = _write_json_artifact(
        vocabulary_payload(locked.learned_arm.vocabulary),
        root / "selected_learned.vocabulary.json",
        root=root,
        resume=resume,
    )
    frequent_ref = _write_json_artifact(
        vocabulary_payload(locked.selected_frequent),
        root / "selected_frequent.vocabulary.json",
        root=root,
        resume=resume,
    )
    validation_ref = _write_json_artifact(
        _validation_payload(validation),
        root / "validation_results.json",
        root=root,
        resume=resume,
    )
    lock_payload = _selection_lock_payload(
        locked,
        selector=selector_ref,
        training_targets=targets_ref,
        learned_vocabulary=learned_ref,
        frequent_vocabulary=frequent_ref,
        validation_results=validation_ref,
    )
    lock_ref = _write_json_artifact(
        lock_payload,
        lock_path,
        root=root,
        resume=resume,
    )
    return LockedArtifactPaths(
        selector=selector_ref,
        training_targets=targets_ref,
        learned_vocabulary=learned_ref,
        frequent_vocabulary=frequent_ref,
        validation_results=validation_ref,
        selection_lock=lock_ref,
    )


def _heldout_receipt_path(root: Path, split: CorpusSplit) -> Path:
    if split not in {CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD}:
        raise ValueError("held-out receipts require TEST_IID or TEST_OOD")
    return root / f"heldout.{split.value}.json"


def _source_cache_payload(
    source_reference: ArtifactReference | None,
) -> dict[str, str] | None:
    return None if source_reference is None else _ref_payload(source_reference)


def _heldout_receipt_payload(
    report: HeldoutSplitReport,
    *,
    locked: LockedExperiment,
    selection_lock: ArtifactReference,
    source_reference: ArtifactReference | None,
    source_expression_ids: Sequence[str],
) -> dict[str, object]:
    return {
        "artifact_version": LEARNED_ARTIFACT_VERSION,
        "config_digest": locked.config_digest,
        "implementation_digest": locked.implementation_digest,
        "lock_digest": locked.lock_digest,
        "report": _heldout_payload(report),
        "schema_version": LEARNED_HELDOUT_SPLIT_VERSION,
        "selection_lock": _ref_payload(selection_lock),
        "source_graph_cache": _source_cache_payload(source_reference),
        "source_expression_ids": list(source_expression_ids),
        "split": report.split.value,
    }


def _load_heldout_receipt(
    path: Path,
    *,
    root: Path,
    split: CorpusSplit,
    locked: LockedExperiment,
    locked_artifacts: LockedArtifactPaths,
    source_reference: ArtifactReference | None,
) -> tuple[HeldoutSplitReport, ArtifactReference, tuple[str, ...]]:
    data = path.read_bytes()
    payload = _json_object(data, label=f"{split.value} held-out receipt")
    expected = {
        "artifact_version": LEARNED_ARTIFACT_VERSION,
        "config_digest": locked.config_digest,
        "implementation_digest": locked.implementation_digest,
        "lock_digest": locked.lock_digest,
        "schema_version": LEARNED_HELDOUT_SPLIT_VERSION,
        "selection_lock": _ref_payload(locked_artifacts.selection_lock),
        "source_graph_cache": _source_cache_payload(source_reference),
        "split": split.value,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise LearnedMotifProtocolError(
                f"{split.value} held-out receipt has incompatible {name}"
            )
    if set(payload) != {*expected, "report", "source_expression_ids"}:
        raise LearnedMotifProtocolError(
            f"{split.value} held-out receipt fields do not match the schema"
        )
    report = _heldout_from_payload(payload["report"])
    if report.split is not split:
        raise LearnedMotifProtocolError(f"{split.value} held-out receipt contains the wrong split")
    _validate_heldout_against_locked(report, locked)
    raw_expression_ids = payload["source_expression_ids"]
    if (
        not isinstance(raw_expression_ids, list)
        or any(
            not isinstance(expression_id, str) or not expression_id.strip()
            for expression_id in raw_expression_ids
        )
        or raw_expression_ids != sorted(raw_expression_ids)
        or len(set(raw_expression_ids)) != len(raw_expression_ids)
        or len(raw_expression_ids) != report.learned.source_row_count
    ):
        raise LearnedMotifProtocolError(
            f"{split.value} held-out receipt identity evidence is invalid"
        )
    return (
        report,
        ArtifactReference(
            path=path.resolve().relative_to(root.resolve()).as_posix(),
            sha256=_sha256_bytes(data),
        ),
        tuple(raw_expression_ids),
    )


def _evaluate_or_load_heldout(
    split_loader: SplitGraphLoader,
    *,
    split: CorpusSplit,
    locked: LockedExperiment,
    locked_artifacts: LockedArtifactPaths,
    output_dir: str | Path,
    source_reference: ArtifactReference | None,
    resume: bool,
    comparison_expression_ids: frozenset[str],
) -> tuple[HeldoutSplitReport, ArtifactReference, tuple[str, ...]]:
    root = Path(output_dir)
    path = _heldout_receipt_path(root, split)
    if path.exists():
        if not resume:
            raise FileExistsError(f"immutable held-out receipt already exists: {path}")
        _LOGGER.info("authenticating existing %s receipt", split.value)
        report, reference, expression_ids = _load_heldout_receipt(
            path,
            root=root,
            split=split,
            locked=locked,
            locked_artifacts=locked_artifacts,
            source_reference=source_reference,
        )
        overlap = comparison_expression_ids.intersection(expression_ids)
        if overlap:
            raise LearnedMotifProtocolError(f"held-out receipt reuses expression {min(overlap)!r}")
        return report, reference, expression_ids

    _LOGGER.info("loading %s graphs after selection lock", split.value)
    batch = split_loader(split)
    _validate_graph_batches((batch,), locked.candidate_pool)
    expression_ids = _batch_expression_ids(batch)
    overlap = comparison_expression_ids.intersection(expression_ids)
    if overlap:
        raise LearnedMotifProtocolError(f"held-out batch reuses expression {min(overlap)!r}")
    _LOGGER.info(
        "evaluating %s across %d locked motif arms and the macro baseline",
        split.value,
        2 + len(locked.random_arms),
    )
    report = evaluate_locked_heldout(locked, batch)
    _LOGGER.info("persisting immutable %s receipt", split.value)
    reference = _write_json_artifact(
        _heldout_receipt_payload(
            report,
            locked=locked,
            selection_lock=locked_artifacts.selection_lock,
            source_reference=source_reference,
            source_expression_ids=expression_ids,
        ),
        path,
        root=root,
        resume=resume,
    )
    return report, reference, expression_ids


def _heldout_aggregate_payload(
    *,
    test_iid: HeldoutSplitReport,
    test_ood: HeldoutSplitReport,
    artifacts: HeldoutArtifactPaths,
) -> dict[str, object]:
    return {
        "schema_version": "geml-goal5-learned-heldout-results-v1",
        "split_receipts": {
            CorpusSplit.TEST_IID.value: _ref_payload(artifacts.test_iid),
            CorpusSplit.TEST_OOD.value: _ref_payload(artifacts.test_ood),
        },
        "test_iid": _heldout_payload(test_iid),
        "test_ood": _heldout_payload(test_ood),
    }


def _experiment_result_payload(
    result: LearnedMotifExperimentResult,
) -> dict[str, object]:
    return {
        "artifact_version": LEARNED_ARTIFACT_VERSION,
        "claim_message": result.claim_message,
        "claim_status": result.claim_status.value,
        "config_digest": result.locked.config_digest,
        "frequent_sweep_provenance": (
            None
            if result.locked.frequent_provenance is None
            else _frequent_provenance_payload(result.locked.frequent_provenance)
        ),
        "implementation_digest": result.locked.implementation_digest,
        "locked_selection_digest": result.locked.lock_digest,
        "schema_version": LEARNED_RESULT_VERSION,
        "test_iid": _heldout_payload(result.test_iid),
        "test_ood": _heldout_payload(result.test_ood),
        "validation": _validation_payload(result.validation),
    }


def _completion_payload(
    result: LearnedMotifExperimentResult,
    *,
    locked_artifacts: LockedArtifactPaths,
    heldout_artifacts: HeldoutArtifactPaths,
    heldout_results: ArtifactReference,
    experiment_result: ArtifactReference,
    reproduction_command: str,
) -> dict[str, object]:
    return {
        "artifact_version": LEARNED_ARTIFACT_VERSION,
        "artifacts": {
            "experiment_result": _ref_payload(experiment_result),
            "frequent_vocabulary": _ref_payload(locked_artifacts.frequent_vocabulary),
            "heldout_results": _ref_payload(heldout_results),
            "test_iid_receipt": _ref_payload(heldout_artifacts.test_iid),
            "test_ood_receipt": _ref_payload(heldout_artifacts.test_ood),
            "learned_vocabulary": _ref_payload(locked_artifacts.learned_vocabulary),
            "selection_lock": _ref_payload(locked_artifacts.selection_lock),
            "selector": _ref_payload(locked_artifacts.selector),
            "training_targets": _ref_payload(locked_artifacts.training_targets),
            "validation_results": _ref_payload(locked_artifacts.validation_results),
        },
        "claim_status": result.claim_status.value,
        "config_digest": result.locked.config_digest,
        "frequent_sweep_provenance": (
            None
            if result.locked.frequent_provenance is None
            else _frequent_provenance_payload(result.locked.frequent_provenance)
        ),
        "implementation_digest": result.locked.implementation_digest,
        "locked_selection_digest": result.locked.lock_digest,
        "reproduction_command": reproduction_command,
        "schema_version": LEARNED_COMPLETE_VERSION,
    }


def finalize_experiment(
    result: LearnedMotifExperimentResult,
    locked_artifacts: LockedArtifactPaths,
    heldout_artifacts: HeldoutArtifactPaths,
    output_dir: str | Path,
    *,
    reproduction_command: str,
    resume: bool,
) -> ArtifactReference:
    """Write held-out evidence and publish the completion marker last."""

    if not isinstance(reproduction_command, str) or not reproduction_command.strip():
        raise ValueError("reproduction_command must be nonblank")
    root = Path(output_dir)
    heldout_ref = _write_json_artifact(
        _heldout_aggregate_payload(
            test_iid=result.test_iid,
            test_ood=result.test_ood,
            artifacts=heldout_artifacts,
        ),
        root / "heldout_results.json",
        root=root,
        resume=resume,
    )
    result_ref = _write_json_artifact(
        _experiment_result_payload(result),
        root / "experiment.result.json",
        root=root,
        resume=resume,
    )
    return _write_json_artifact(
        _completion_payload(
            result,
            locked_artifacts=locked_artifacts,
            heldout_artifacts=heldout_artifacts,
            heldout_results=heldout_ref,
            experiment_result=result_ref,
            reproduction_command=reproduction_command,
        ),
        root / "run.complete.json",
        root=root,
        resume=resume,
    )


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LearnedMotifProtocolError(f"{label} must be lowercase SHA-256 hex")
    return value


def _scientific_lock_digest_from_persisted(
    lock: Mapping[str, object],
) -> str:
    raw_random_arms = lock.get("random_arms")
    if not isinstance(raw_random_arms, list) or any(
        not isinstance(arm, dict) for arm in raw_random_arms
    ):
        raise LearnedMotifProtocolError("persisted lock random arms are invalid")
    payload = {
        "budget": lock.get("budget"),
        "candidate_pool_digest": lock.get("candidate_pool_digest"),
        "candidate_prefilter": lock.get("candidate_prefilter"),
        "config_digest": lock.get("config_digest"),
        "frequent_vocabulary_sha256": lock.get("frequent_vocabulary_payload_sha256"),
        "frequent_sweep_provenance": lock.get("frequent_sweep_provenance"),
        "implementation_digest": lock.get("implementation_digest"),
        "learned_vocabulary_sha256": lock.get("learned_vocabulary_payload_sha256"),
        "partition_digest": lock.get("train_partition_digest"),
        "random_arms": [
            {
                "seed": arm.get("seed"),
                "vocabulary_sha256": arm.get("vocabulary_payload_sha256"),
            }
            for arm in raw_random_arms
        ],
        "ridge_lambda": lock.get("ridge_lambda"),
        "selector_sha256": lock.get("selector_payload_sha256"),
        "source_graph_caches": lock.get("source_graph_caches"),
        "target_computation": lock.get("target_computation"),
        "validation_total_mdl_bits": lock.get("validation_total_mdl_bits"),
        "version": lock.get("schema_version"),
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _frequent_provenance_from_payload(
    payload: object,
) -> FrequentSweepProvenance | None:
    if payload is None:
        return None
    expected = {
        "config_digest",
        "implementation_digest",
        "input_manifest_sha256",
        "run_directory",
        "run_complete_sha256",
        "selected_configuration_digest",
        "selection_lock_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise LearnedMotifProtocolError("frequent sweep provenance fields do not match the schema")
    try:
        return FrequentSweepProvenance(**payload)
    except (TypeError, ValueError) as error:
        raise LearnedMotifProtocolError("frequent sweep provenance failed validation") from error


@dataclass(frozen=True, slots=True)
class LearnedMotifReplayInputs:
    """Explicit source evidence for non-production completed-run verification.

    Production runs always replay from recursively authenticated issue 5-5
    graph caches.  Tiny tests and direct API callers without 5-5 provenance
    must supply the same config, source pool, and split loader explicitly so
    the completed loader never silently downgrades scientific verification.
    """

    config: Goal5LearnedMotifConfig
    candidate_pool: MotifVocabulary
    split_loader: Callable[[CorpusSplit], SplitGraphBatch]
    train_partition: TrainPartition | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, Goal5LearnedMotifConfig):
            raise TypeError("config must be a Goal5LearnedMotifConfig")
        if not isinstance(self.candidate_pool, MotifVocabulary):
            raise TypeError("candidate_pool must be a MotifVocabulary")
        if not callable(self.split_loader):
            raise TypeError("split_loader must be callable")
        if self.train_partition is not None and not isinstance(
            self.train_partition,
            TrainPartition,
        ):
            raise TypeError("train_partition must be None or TrainPartition")


def _learned_config_from_completion(
    root: Path,
    completion: Mapping[str, object],
    *,
    inputs: FrequentSweepInputs,
    implementation_digest: str,
) -> LoadedLearnedMotifConfig:
    """Recursively authenticate the checked-in config named by the command."""

    command = completion.get("reproduction_command")
    if not isinstance(command, str) or not command.strip():
        raise LearnedMotifProtocolError("learned completion reproduction_command must be nonblank")
    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise LearnedMotifProtocolError(
            "learned completion reproduction_command is invalid"
        ) from error
    expected_prefix = [
        "python",
        "-m",
        "geml.experiments.goal5.learned_motifs",
    ]
    if (
        tokens[:3] != expected_prefix
        or len(tokens) != 7
        or tokens[3] != "--config"
        or tokens[5] != "--frequent-sweep-run-dir"
    ):
        raise LearnedMotifProtocolError(
            "learned completion must contain the exact production reproduction command"
        )
    repository_root = _repository_root(root)
    config_path = _resolve_inside(
        repository_root,
        tokens[4],
        label="learned reproduction config",
    )
    try:
        config_path.relative_to((repository_root / "configs").resolve())
    except ValueError as error:
        raise LearnedMotifProtocolError(
            "learned reproduction config must be stored in the checked-in config directory"
        ) from error
    source_run = _resolve_inside(
        repository_root,
        tokens[6],
        label="learned reproduction frequent run",
    )
    if source_run != inputs.run_dir.resolve():
        raise LearnedMotifProtocolError("learned reproduction command names the wrong frequent run")
    try:
        loaded = load_learned_motif_config(
            config_path,
            frequent_sweep_run_dir=source_run,
            require_inputs=True,
        )
    except (OSError, TypeError, ValueError) as error:
        raise LearnedMotifProtocolError(
            "learned reproduction config failed recursive authentication"
        ) from error
    if (
        loaded.config_digest != completion.get("config_digest")
        or loaded.frequent_sweep_run_dir != source_run
        or _learned_run_directory(
            loaded,
            inputs,
            implementation_digest=implementation_digest,
        ).resolve()
        != root
    ):
        raise LearnedMotifProtocolError(
            "learned checked-in config, source run, or derived run identity is inconsistent"
        )
    return loaded


def _replay_completed_scientific_evidence(
    *,
    config: Goal5LearnedMotifConfig,
    candidate_pool: MotifVocabulary,
    selected_frequent: MotifVocabulary,
    split_loader: Callable[[CorpusSplit], SplitGraphBatch],
    train_partition: TrainPartition | None,
    frequent_inputs: FrequentSweepInputs | None,
    config_digest: str,
    implementation_digest: str,
    frequent_provenance: FrequentSweepProvenance | None,
    source_graph_cache_references: Mapping[CorpusSplit, ArtifactReference] | None,
    references: Mapping[str, ArtifactReference],
    artifact_data: Mapping[str, bytes],
    completion: Mapping[str, object],
    completion_data: bytes,
) -> None:
    """Deterministically replay every fitted and evaluated scientific value."""

    if frequent_inputs is not None:
        train_partition = load_frequent_train_partition(
            frequent_inputs,
            fraction=config.selector.train_fit_fraction,
            seed=config.runtime.seed,
            maximum_graphs=config.selector.maximum_target_graphs,
        )
    train_batch = (
        split_loader(CorpusSplit.TRAIN) if train_partition is None else train_partition.batch
    )
    validation_batch = split_loader(CorpusSplit.VALIDATION)
    replayed_locked, replayed_validation = fit_and_lock_selection(
        train_batch,
        validation_batch,
        candidate_pool=candidate_pool,
        selected_frequent=selected_frequent,
        config=config,
        config_digest=config_digest,
        implementation_digest=implementation_digest,
        frequent_provenance=frequent_provenance,
        train_partition=train_partition,
        source_graph_cache_references=source_graph_cache_references,
    )
    train_validation_ids = frozenset(
        replayed_locked.training_target_audit.partition.source_expression_ids
    ).union(_batch_expression_ids(validation_batch))
    if len(train_validation_ids) != (
        len(replayed_locked.training_target_audit.partition.source_expression_ids)
        + validation_batch.source_row_count
    ):
        raise LearnedMotifProtocolError(
            "replayed TRAIN and VALIDATION reuse an expression identity"
        )

    locked_artifacts = LockedArtifactPaths(
        selector=references["selector"],
        training_targets=references["training_targets"],
        learned_vocabulary=references["learned_vocabulary"],
        frequent_vocabulary=references["frequent_vocabulary"],
        validation_results=references["validation_results"],
        selection_lock=references["selection_lock"],
    )
    _require_exact_artifact(
        artifact_data["selector"],
        selector_payload(replayed_locked.selector),
        label="persisted ridge selector",
    )
    _require_exact_artifact(
        artifact_data["training_targets"],
        _target_payload(replayed_locked.training_target_audit),
        label="persisted exact TRAIN targets",
    )
    _require_exact_artifact(
        artifact_data["learned_vocabulary"],
        vocabulary_payload(replayed_locked.learned_arm.vocabulary),
        label="persisted learned vocabulary",
    )
    _require_exact_artifact(
        artifact_data["frequent_vocabulary"],
        vocabulary_payload(replayed_locked.selected_frequent),
        label="persisted frequent vocabulary",
    )
    _require_exact_artifact(
        artifact_data["validation_results"],
        _validation_payload(replayed_validation),
        label="persisted VALIDATION results",
    )
    _require_exact_artifact(
        artifact_data["selection_lock"],
        _selection_lock_payload(
            replayed_locked,
            selector=locked_artifacts.selector,
            training_targets=locked_artifacts.training_targets,
            learned_vocabulary=locked_artifacts.learned_vocabulary,
            frequent_vocabulary=locked_artifacts.frequent_vocabulary,
            validation_results=locked_artifacts.validation_results,
        ),
        label="persisted selection lock",
    )
    del validation_batch

    replayed_reports: dict[CorpusSplit, HeldoutSplitReport] = {}
    compared_ids = train_validation_ids
    for split in (CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD):
        batch = split_loader(split)
        expression_ids = _batch_expression_ids(batch)
        overlap = compared_ids.intersection(expression_ids)
        if overlap:
            raise LearnedMotifProtocolError(
                f"replayed {split.value} reuses expression {min(overlap)!r}"
            )
        report = evaluate_locked_heldout(replayed_locked, batch)
        receipt_name = f"{split.value}_receipt"
        source_reference = (
            None if source_graph_cache_references is None else source_graph_cache_references[split]
        )
        _require_exact_artifact(
            artifact_data[receipt_name],
            _heldout_receipt_payload(
                report,
                locked=replayed_locked,
                selection_lock=locked_artifacts.selection_lock,
                source_reference=source_reference,
                source_expression_ids=expression_ids,
            ),
            label=f"persisted {split.value} receipt",
        )
        replayed_reports[split] = report
        compared_ids = compared_ids.union(expression_ids)
        del batch

    heldout_artifacts = HeldoutArtifactPaths(
        test_iid=references["test_iid_receipt"],
        test_ood=references["test_ood_receipt"],
    )
    test_iid = replayed_reports[CorpusSplit.TEST_IID]
    test_ood = replayed_reports[CorpusSplit.TEST_OOD]
    _require_exact_artifact(
        artifact_data["heldout_results"],
        _heldout_aggregate_payload(
            test_iid=test_iid,
            test_ood=test_ood,
            artifacts=heldout_artifacts,
        ),
        label="persisted held-out aggregate",
    )
    claim_status, claim_message = _claim_from_iid(test_iid)
    replayed_result = LearnedMotifExperimentResult(
        locked=replayed_locked,
        validation=replayed_validation,
        test_iid=test_iid,
        test_ood=test_ood,
        claim_status=claim_status,
        claim_message=claim_message,
    )
    _require_exact_artifact(
        artifact_data["experiment_result"],
        _experiment_result_payload(replayed_result),
        label="persisted experiment result",
    )
    _require_exact_artifact(
        completion_data,
        _completion_payload(
            replayed_result,
            locked_artifacts=locked_artifacts,
            heldout_artifacts=heldout_artifacts,
            heldout_results=references["heldout_results"],
            experiment_result=references["experiment_result"],
            reproduction_command=str(completion["reproduction_command"]),
        ),
        label="persisted run completion",
    )


@dataclass(frozen=True, slots=True)
class CompletedLearnedMotifRun:
    """Authenticated public handoff from issue 5-6 to downstream stages."""

    run_dir: Path
    learned_vocabulary: MotifVocabulary
    frequent_vocabulary: MotifVocabulary
    selection_lock_path: Path
    heldout_results_path: Path
    experiment_result_path: Path
    completion_path: Path
    claim_status: HeldoutClaimStatus
    config_digest: str
    implementation_digest: str
    frequent_provenance: FrequentSweepProvenance | None


def load_completed_learned_motif_run(
    run_dir: str | Path,
    *,
    expected_config_digest: str | None = None,
    expected_implementation_digest: str | None = None,
    expected_frequent_provenance: FrequentSweepProvenance | None = None,
    require_current_implementation: bool = True,
    replay_inputs: LearnedMotifReplayInputs | None = None,
) -> CompletedLearnedMotifRun:
    """Replay and authenticate a complete learned run and every scientific result."""

    if not isinstance(require_current_implementation, bool):
        raise TypeError("require_current_implementation must be a boolean")
    if replay_inputs is not None and not isinstance(replay_inputs, LearnedMotifReplayInputs):
        raise TypeError("replay_inputs must be None or LearnedMotifReplayInputs")
    if expected_config_digest is not None:
        _require_sha256(expected_config_digest, label="expected config digest")
    if expected_implementation_digest is not None:
        _require_sha256(
            expected_implementation_digest,
            label="expected implementation digest",
        )
    if expected_frequent_provenance is not None and not isinstance(
        expected_frequent_provenance,
        FrequentSweepProvenance,
    ):
        raise TypeError("expected_frequent_provenance must be None or FrequentSweepProvenance")

    # Authenticate the completion envelope before following any artifact reference.
    root = Path(run_dir).resolve()
    completion_path = root / "run.complete.json"
    if not completion_path.is_file():
        raise LearnedMotifConfigurationError(
            f"learned run completion does not exist: {completion_path}"
        )
    completion_data = completion_path.read_bytes()
    complete = _json_object(
        completion_data,
        label="learned run completion",
    )
    complete_fields = {
        "artifact_version",
        "artifacts",
        "claim_status",
        "config_digest",
        "frequent_sweep_provenance",
        "implementation_digest",
        "locked_selection_digest",
        "reproduction_command",
        "schema_version",
    }
    if set(complete) != complete_fields:
        raise LearnedMotifProtocolError("learned completion fields do not match the frozen schema")
    if complete["schema_version"] != LEARNED_COMPLETE_VERSION:
        raise LearnedMotifProtocolError("unsupported learned completion schema")
    if complete["artifact_version"] != LEARNED_ARTIFACT_VERSION:
        raise LearnedMotifProtocolError("learned completion artifact version mismatch")
    config_digest = _require_sha256(
        complete["config_digest"],
        label="learned config_digest",
    )
    implementation_digest = _require_sha256(
        complete["implementation_digest"],
        label="learned implementation_digest",
    )
    lock_digest = _require_sha256(
        complete["locked_selection_digest"],
        label="learned locked_selection_digest",
    )
    if expected_config_digest is not None and config_digest != expected_config_digest:
        raise LearnedMotifProtocolError("learned completion has an unexpected config digest")
    if (
        expected_implementation_digest is not None
        and implementation_digest != expected_implementation_digest
    ):
        raise LearnedMotifProtocolError(
            "learned completion has an unexpected implementation digest"
        )
    repository_root = _repository_root(Path(__file__))
    if (
        require_current_implementation
        and learned_implementation_digest(repository_root) != implementation_digest
    ):
        raise LearnedMotifProtocolError("learned run was produced by a stale implementation")
    # Reopen the immutable 5-5 source and recover the exact replay configuration.
    provenance = _frequent_provenance_from_payload(complete["frequent_sweep_provenance"])
    if expected_frequent_provenance is not None and provenance != expected_frequent_provenance:
        raise LearnedMotifProtocolError("learned completion has unexpected frequent-run provenance")
    verified_frequent_inputs: FrequentSweepInputs | None = None
    if provenance is not None:
        source_run_dir = _resolve_inside(
            repository_root,
            provenance.run_directory,
            label="frequent provenance run_directory",
        )
        try:
            verified_frequent_inputs = load_frequent_sweep_inputs(source_run_dir)
        except (OSError, TypeError, ValueError) as error:
            raise LearnedMotifProtocolError(
                "learned completion's frequent source failed recursive authentication"
            ) from error
        if verified_frequent_inputs.provenance != provenance:
            raise LearnedMotifProtocolError(
                "learned completion's frequent source provenance is inconsistent"
            )
    if verified_frequent_inputs is not None:
        loaded_replay_config = _learned_config_from_completion(
            root,
            complete,
            inputs=verified_frequent_inputs,
            implementation_digest=implementation_digest,
        )
        replay_config = loaded_replay_config.config
        replay_candidate_pool = verified_frequent_inputs.candidate_pool
        replay_split_loader = frequent_graph_split_loader(verified_frequent_inputs)
        replay_train_partition = None
    else:
        if replay_inputs is None:
            raise LearnedMotifProtocolError(
                "a run without authenticated 5-5 provenance requires explicit replay_inputs"
            )
        replay_config = replay_inputs.config
        replay_candidate_pool = replay_inputs.candidate_pool
        replay_split_loader = replay_inputs.split_loader
        replay_train_partition = replay_inputs.train_partition
        if learned_config_digest(replay_config) != config_digest:
            raise LearnedMotifProtocolError(
                "explicit replay config does not match the completed config digest"
            )
    if (
        not isinstance(complete["reproduction_command"], str)
        or not complete["reproduction_command"].strip()
    ):
        raise LearnedMotifProtocolError("learned completion reproduction_command must be nonblank")
    try:
        claim_status = HeldoutClaimStatus(complete["claim_status"])
    except (TypeError, ValueError) as error:
        raise LearnedMotifProtocolError("learned completion has an invalid claim_status") from error

    # Verify the complete artifact set and every content-addressed reference.
    raw_artifacts = complete["artifacts"]
    expected_artifact_names = {
        "experiment_result",
        "frequent_vocabulary",
        "heldout_results",
        "learned_vocabulary",
        "selection_lock",
        "selector",
        "test_iid_receipt",
        "test_ood_receipt",
        "training_targets",
        "validation_results",
    }
    if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != expected_artifact_names:
        raise LearnedMotifProtocolError(
            "learned completion artifact keys do not match the frozen schema"
        )
    references = {
        name: _artifact_ref(value, label=f"learned artifact {name}")
        for name, value in raw_artifacts.items()
    }
    artifact_data = {
        name: _read_artifact(root, reference) for name, reference in references.items()
    }

    # Validate the selection lock before inspecting post-selection evidence.
    lock = _json_object(
        artifact_data["selection_lock"],
        label="learned selection lock",
    )
    expected_lock_fields = {
        "artifact_version",
        "artifacts",
        "budget",
        "candidate_pool_digest",
        "candidate_prefilter",
        "config_digest",
        "frequent_sweep_provenance",
        "frequent_arm_id",
        "frequent_vocabulary_id",
        "frequent_vocabulary_payload_sha256",
        "heldout_artifacts_absent_at_lock",
        "implementation_digest",
        "learned_arm_id",
        "learned_vocabulary_id",
        "learned_vocabulary_payload_sha256",
        "lock_digest",
        "random_arms",
        "ridge_lambda",
        "schema_version",
        "selector_payload_sha256",
        "source_graph_caches",
        "target_computation",
        "train_partition_digest",
        "validation_total_mdl_bits",
    }
    if set(lock) != expected_lock_fields:
        raise LearnedMotifProtocolError(
            "learned selection lock fields do not match the frozen schema"
        )
    for name, value in {
        "artifact_version": LEARNED_ARTIFACT_VERSION,
        "config_digest": config_digest,
        "frequent_sweep_provenance": (
            None if provenance is None else _frequent_provenance_payload(provenance)
        ),
        "heldout_artifacts_absent_at_lock": True,
        "implementation_digest": implementation_digest,
        "lock_digest": lock_digest,
        "schema_version": LEARNED_LOCK_VERSION,
    }.items():
        if lock.get(name) != value:
            raise LearnedMotifProtocolError(f"learned selection lock has incompatible {name}")
    if _scientific_lock_digest_from_persisted(lock) != lock_digest:
        raise LearnedMotifProtocolError(
            "learned selection lock digest does not authenticate its scientific choices"
        )
    raw_source_caches = lock.get("source_graph_caches")
    if verified_frequent_inputs is None:
        if raw_source_caches is not None:
            raise LearnedMotifProtocolError(
                "learned selection lock names source caches without frequent provenance"
            )
        source_cache_references: dict[CorpusSplit, ArtifactReference] = {}
    else:
        if not isinstance(raw_source_caches, dict) or set(raw_source_caches) != {
            split.value for split in CorpusSplit
        }:
            raise LearnedMotifProtocolError(
                "learned selection lock must bind all four source graph caches"
            )
        source_cache_references = {
            split: _artifact_ref(
                raw_source_caches[split.value],
                label=f"locked {split.value} source graph cache",
            )
            for split in CorpusSplit
        }
        if source_cache_references != dict(verified_frequent_inputs.graph_cache_references):
            raise LearnedMotifProtocolError(
                "learned selection lock source caches disagree with the authenticated frequent run"
            )
    # Prove split identity disjointness directly from the authenticated graph caches.
    source_expression_ids_by_split: dict[CorpusSplit, tuple[str, ...]] = {}
    if verified_frequent_inputs is not None:
        observed_source_ids: set[str] = set()
        for split in CorpusSplit:
            expression_ids = _graph_cache_expression_ids(
                verified_frequent_inputs.graph_cache_descriptors[split]
            )
            overlap = observed_source_ids.intersection(expression_ids)
            if overlap:
                raise LearnedMotifProtocolError(
                    f"authenticated source caches reuse expression {min(overlap)!r}"
                )
            observed_source_ids.update(expression_ids)
            source_expression_ids_by_split[split] = expression_ids
    raw_locked_artifacts = lock.get("artifacts")
    locked_artifact_names = {
        "frequent_vocabulary",
        "learned_vocabulary",
        "selector",
        "training_targets",
        "validation_results",
    }
    if (
        not isinstance(raw_locked_artifacts, dict)
        or set(raw_locked_artifacts) != locked_artifact_names
    ):
        raise LearnedMotifProtocolError("learned selection lock artifact keys are invalid")
    for name in locked_artifact_names:
        if (
            _artifact_ref(
                raw_locked_artifacts[name],
                label=f"locked learned artifact {name}",
            )
            != references[name]
        ):
            raise LearnedMotifProtocolError(f"completion and selection lock disagree on {name}")

    # Decode and bind the selected vocabularies and fitted selector to the lock.
    try:
        learned_vocabulary = vocabulary_from_payload(
            _json_object(
                artifact_data["learned_vocabulary"],
                label="selected learned vocabulary",
            )
        )
        frequent_vocabulary = vocabulary_from_payload(
            _json_object(
                artifact_data["frequent_vocabulary"],
                label="selected frequent vocabulary",
            )
        )
        selector = selector_from_payload(
            _json_object(
                artifact_data["selector"],
                label="learned selector",
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LearnedMotifProtocolError(
            "learned selection artifacts failed structural validation"
        ) from error
    for prefix, vocabulary in (
        ("learned", learned_vocabulary),
        ("frequent", frequent_vocabulary),
    ):
        if lock.get(f"{prefix}_vocabulary_id") != vocabulary.vocabulary_id:
            raise LearnedMotifProtocolError(f"locked {prefix} vocabulary ID is inconsistent")
        if lock.get(f"{prefix}_vocabulary_payload_sha256") != vocabulary_payload_digest(vocabulary):
            raise LearnedMotifProtocolError(f"locked {prefix} vocabulary digest is inconsistent")
    if lock.get("selector_payload_sha256") != selector_payload_digest(selector):
        raise LearnedMotifProtocolError("locked selector payload digest is inconsistent")
    if lock.get("candidate_pool_digest") != selector.candidate_pool_digest:
        raise LearnedMotifProtocolError("locked selector and candidate pool digests disagree")
    budget = lock.get("budget")
    if (
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget < 1
        or budget != len(learned_vocabulary.templates)
        or budget != len(frequent_vocabulary.templates)
    ):
        raise LearnedMotifProtocolError("locked learned/frequent vocabulary budget is inconsistent")
    replay_size_pool, replay_budget = align_frequent_candidate_pool(
        replay_candidate_pool,
        frequent_vocabulary,
    )
    replay_eligible_pool, replay_prefilter = prefilter_train_candidates(
        replay_size_pool,
        frequent_vocabulary,
        maximum_candidate_motifs=(replay_config.selector.maximum_candidate_motifs),
    )
    if (
        replay_budget != budget
        or candidate_pool_digest(replay_eligible_pool) != selector.candidate_pool_digest
        or _candidate_prefilter_payload(replay_prefilter) != lock.get("candidate_prefilter")
    ):
        raise LearnedMotifProtocolError(
            "locked candidate pool cannot be reproduced from TRAIN evidence"
        )
    # Reconstruct the thirty equal-budget random arms from their frozen seeds.
    raw_random_arms = lock.get("random_arms")
    if not isinstance(raw_random_arms, list) or len(raw_random_arms) != 30:
        raise LearnedMotifProtocolError("learned selection lock requires thirty random arms")
    random_seeds: set[int] = set()
    for arm in raw_random_arms:
        if (
            not isinstance(arm, dict)
            or set(arm)
            != {
                "arm_id",
                "motif_ids",
                "seed",
                "vocabulary_id",
                "vocabulary_payload_sha256",
            }
            or not isinstance(arm["motif_ids"], list)
            or any(
                not isinstance(motif_id, str) or not motif_id.strip()
                for motif_id in arm["motif_ids"]
            )
            or len(arm["motif_ids"]) != budget
            or len(set(arm["motif_ids"])) != budget
            or isinstance(arm["seed"], bool)
            or not isinstance(arm["seed"], int)
            or arm["seed"] < 0
        ):
            raise LearnedMotifProtocolError("locked random arm is inconsistent")
        _exact_string(arm["arm_id"], label="random arm ID")
        _exact_string(arm["vocabulary_id"], label="random vocabulary ID")
        _require_sha256(
            arm["vocabulary_payload_sha256"],
            label="random vocabulary payload digest",
        )
        random_seeds.add(arm["seed"])
    if len(random_seeds) != 30:
        raise LearnedMotifProtocolError("locked random arm seeds must be unique")
    ordered_random_seeds = [arm["seed"] for arm in raw_random_arms]
    if (
        lock.get("learned_arm_id") != _lambda_arm_id(selector.ridge_lambda)
        or lock.get("frequent_arm_id") != "frequent"
        or lock.get("ridge_lambda") != selector.ridge_lambda
        or ordered_random_seeds
        != list(range(ordered_random_seeds[0], ordered_random_seeds[0] + 30))
        or any(
            arm["arm_id"] != f"random:{index:02d}:seed={arm['seed']}"
            for index, arm in enumerate(raw_random_arms)
        )
    ):
        raise LearnedMotifProtocolError("locked arm identities are inconsistent")
    # Reproduce every locked vocabulary from authenticated TRAIN-only inputs.
    random_dictionary_bits: dict[int, int] = {}
    if verified_frequent_inputs is not None:
        if vocabulary_payload(frequent_vocabulary) != vocabulary_payload(
            verified_frequent_inputs.selected_frequent
        ):
            raise LearnedMotifProtocolError(
                "persisted frequent baseline differs from the authenticated 5-5 winner"
            )
        size_eligible_pool, verified_budget = align_frequent_candidate_pool(
            verified_frequent_inputs.candidate_pool,
            frequent_vocabulary,
        )
        raw_prefilter = lock.get("candidate_prefilter")
        if not isinstance(raw_prefilter, dict):
            raise LearnedMotifProtocolError("locked candidate prefilter evidence is missing")
        maximum_candidates = raw_prefilter.get("maximum_candidate_motifs")
        if isinstance(maximum_candidates, bool) or not isinstance(maximum_candidates, int):
            raise LearnedMotifProtocolError("locked candidate prefilter cap is invalid")
        eligible_pool, expected_prefilter = prefilter_train_candidates(
            size_eligible_pool,
            frequent_vocabulary,
            maximum_candidate_motifs=maximum_candidates,
        )
        expected_learned = _selected_vocabulary(
            eligible_pool,
            select_learned_templates(
                eligible_pool,
                selector,
                budget=verified_budget,
            ),
            budget=verified_budget,
        )
        if (
            verified_budget != budget
            or _candidate_prefilter_payload(expected_prefilter) != raw_prefilter
            or candidate_pool_digest(eligible_pool) != lock.get("candidate_pool_digest")
            or vocabulary_payload(expected_learned) != vocabulary_payload(learned_vocabulary)
        ):
            raise LearnedMotifProtocolError(
                "persisted learned selection cannot be reproduced from authenticated TRAIN inputs"
            )
        for arm in raw_random_arms:
            expected_random = _selected_vocabulary(
                eligible_pool,
                select_random_templates(
                    eligible_pool,
                    budget=budget,
                    seed=arm["seed"],
                ),
                budget=budget,
            )
            random_dictionary_bits[arm["seed"]] = vocabulary_mdl_bits(expected_random.templates)
            if (
                arm["motif_ids"] != [template.motif_id for template in expected_random.templates]
                or arm["vocabulary_id"] != expected_random.vocabulary_id
                or arm["vocabulary_payload_sha256"] != vocabulary_payload_digest(expected_random)
            ):
                raise LearnedMotifProtocolError(
                    "locked random vocabulary cannot be reproduced from authenticated TRAIN inputs"
                )
    # Authenticate exact singleton-MDL targets and their retained failures.
    targets = _json_object(
        artifact_data["training_targets"],
        label="learned training targets",
    )
    if (
        set(targets)
        != {
            "candidate_pool_digest",
            "candidate_prefilter",
            "computation",
            "partition",
            "schema_version",
            "targets",
            "total_failure_count",
        }
        or targets.get("schema_version") != "geml-goal5-exact-training-targets-v1"
        or targets.get("candidate_pool_digest") != selector.candidate_pool_digest
        or not isinstance(targets.get("targets"), list)
        or len(targets["targets"]) != selector.training_example_count
    ):
        raise LearnedMotifProtocolError("learned training target artifact is inconsistent")
    raw_computation = targets["computation"]
    if (
        not isinstance(raw_computation, dict)
        or set(raw_computation)
        != {
            "base_singleton_evaluation_count",
            "brute_force_evaluation_count",
            "sparse_singleton_evaluation_count",
            "union_match_pass_count",
        }
        or targets["candidate_prefilter"] != lock.get("candidate_prefilter")
        or {
            "base_singleton_evaluation_count": raw_computation.get(
                "base_singleton_evaluation_count"
            ),
            "sparse_singleton_evaluation_count": raw_computation.get(
                "sparse_singleton_evaluation_count"
            ),
            "union_match_pass_count": raw_computation.get("union_match_pass_count"),
        }
        != lock.get("target_computation")
        or _exact_int(
            targets["total_failure_count"],
            label="training target total_failure_count",
        )
        != 0
    ):
        raise LearnedMotifProtocolError("learned training target audit metadata is inconsistent")
    for name in raw_computation:
        _exact_int(
            raw_computation[name],
            label=f"training target computation {name}",
        )
    raw_target_rows = targets["targets"]
    target_examples: list[SelectorTrainingExample] = []
    target_summaries: list[SplitMDLSummary] = []
    target_upstream_failure_counts: list[int] = []
    target_reconstruction_failure_counts: list[int] = []
    expected_target_fields = {
        "mdl_gain_bits",
        "motif_id",
        "reconstruction_failure_count",
        "reconstruction_failures",
        "source_row_count",
        "split",
        "summary",
        "upstream_failure_count",
    }
    for target in raw_target_rows:
        if not isinstance(target, dict) or set(target) != expected_target_fields:
            raise LearnedMotifProtocolError("learned training target row fields are invalid")
        try:
            target_examples.append(
                SelectorTrainingExample(
                    motif_id=target["motif_id"],
                    split=CorpusSplit(target["split"]),
                    mdl_gain_bits=target["mdl_gain_bits"],
                )
            )
            summary = _summary_from_payload(target["summary"])
        except (TypeError, ValueError) as error:
            raise LearnedMotifProtocolError(
                "learned training target row failed validation"
            ) from error
        target_summaries.append(summary)
        upstream_failure_count = _exact_int(
            target["upstream_failure_count"],
            label="target upstream_failure_count",
        )
        reconstruction_failure_count = _exact_int(
            target["reconstruction_failure_count"],
            label="target reconstruction_failure_count",
        )
        target_upstream_failure_counts.append(upstream_failure_count)
        target_reconstruction_failure_counts.append(reconstruction_failure_count)
        raw_reconstruction_failures = target["reconstruction_failures"]
        if not isinstance(raw_reconstruction_failures, list):
            raise LearnedMotifProtocolError("target reconstruction_failures must be a list")
        for failure in raw_reconstruction_failures:
            if not isinstance(failure, dict) or set(failure) != {
                "attempted_selected_occurrence_count",
                "error_message",
                "error_type",
                "expression_id",
            }:
                raise LearnedMotifProtocolError("target reconstruction failure fields are invalid")
            EvaluationFailure(
                expression_id=_exact_string(
                    failure["expression_id"],
                    label="target failure expression_id",
                ),
                error_type=_exact_string(
                    failure["error_type"],
                    label="target failure error_type",
                ),
                error_message=_exact_string(
                    failure["error_message"],
                    label="target failure error_message",
                ),
                attempted_selected_occurrence_count=_exact_int(
                    failure["attempted_selected_occurrence_count"],
                    label="target failure attempted occurrence count",
                ),
            )
        if (
            target["mdl_gain_bits"] != summary.savings_bits
            or reconstruction_failure_count != len(raw_reconstruction_failures)
            or reconstruction_failure_count != summary.reconstruction_failure_count
            or _exact_int(
                target["source_row_count"],
                label="target source_row_count",
            )
            != summary.processed_count + upstream_failure_count
        ):
            raise LearnedMotifProtocolError("learned training target row is inconsistent")
    target_tuple = tuple(target_examples)
    if training_target_digest(target_tuple) != selector.training_target_digest or {
        example.motif_id for example in target_tuple
    } != {template.motif_id for template in replay_eligible_pool.templates}:
        raise LearnedMotifProtocolError("persisted training targets do not match the selector")
    # Rebuild the deterministic TRAIN sub-partition and all of its denominators.
    raw_partition = targets.get("partition")
    expected_partition_fields = {
        "digest",
        "fraction",
        "fractional_graph_count",
        "full_train_graph_count",
        "maximum_graphs",
        "seed",
        "selected_graph_count",
        "source_expression_count",
        "source_identity_digest",
        "upstream_failure_count",
        "upstream_failures",
    }
    if (
        not isinstance(raw_partition, dict)
        or set(raw_partition) != expected_partition_fields
        or raw_partition.get("digest") != selector.train_partition_digest
        or lock.get("train_partition_digest") != selector.train_partition_digest
        or _require_sha256(
            raw_partition.get("source_identity_digest"),
            label="TRAIN source identity digest",
        )
        != raw_partition.get("source_identity_digest")
        or _exact_int(
            raw_partition.get("full_train_graph_count"),
            label="TRAIN full_train_graph_count",
            minimum=1,
        )
        < 1
        or _exact_int(
            raw_partition.get("selected_graph_count"),
            label="TRAIN selected_graph_count",
            minimum=1,
        )
        > raw_partition["full_train_graph_count"]
        or _exact_int(
            raw_partition.get("source_expression_count"),
            label="TRAIN source_expression_count",
            minimum=1,
        )
        < 1
        or raw_partition["source_expression_count"]
        != raw_partition["full_train_graph_count"] + raw_partition.get("upstream_failure_count", -1)
        or not isinstance(raw_partition.get("upstream_failures"), list)
        or raw_partition.get("upstream_failure_count") != len(raw_partition["upstream_failures"])
    ):
        raise LearnedMotifProtocolError("persisted TRAIN partition does not match the selector")
    partition_fraction = _exact_float(
        raw_partition["fraction"],
        label="TRAIN partition fraction",
    )
    partition_maximum_graphs = _exact_int(
        raw_partition["maximum_graphs"],
        label="TRAIN partition maximum_graphs",
        minimum=1,
    )
    partition_seed = _exact_int(
        raw_partition["seed"],
        label="TRAIN partition seed",
    )
    fractional_graph_count = _exact_int(
        raw_partition["fractional_graph_count"],
        label="TRAIN fractional_graph_count",
        minimum=1,
    )
    upstream_failure_count = _exact_int(
        raw_partition["upstream_failure_count"],
        label="TRAIN upstream_failure_count",
    )
    expected_fractional_count = max(
        1,
        math.ceil(raw_partition["full_train_graph_count"] * partition_fraction),
    )
    if (
        partition_fraction != replay_config.selector.train_fit_fraction
        or partition_maximum_graphs != replay_config.selector.maximum_target_graphs
        or partition_seed != replay_config.runtime.seed
        or fractional_graph_count != expected_fractional_count
        or raw_partition["selected_graph_count"]
        != min(fractional_graph_count, partition_maximum_graphs)
        or raw_computation["union_match_pass_count"] != raw_partition["selected_graph_count"]
        or raw_computation["base_singleton_evaluation_count"]
        != raw_partition["selected_graph_count"]
        or raw_computation["brute_force_evaluation_count"]
        != raw_partition["selected_graph_count"] * len(raw_target_rows)
        or raw_computation["sparse_singleton_evaluation_count"]
        > raw_computation["brute_force_evaluation_count"]
        or any(count != upstream_failure_count for count in target_upstream_failure_counts)
        or upstream_failure_count != 0
        or any(count != 0 for count in target_reconstruction_failure_counts)
    ):
        raise LearnedMotifProtocolError(
            "persisted TRAIN target counts or partition policy are inconsistent"
        )
    if any(
        target.split is not CorpusSplit.TRAIN
        or target_summary.processed_count != raw_partition["selected_graph_count"]
        for target, target_summary in zip(
            target_examples,
            target_summaries,
            strict=True,
        )
    ):
        raise LearnedMotifProtocolError(
            "persisted TRAIN targets use an inconsistent selected denominator"
        )
    if source_expression_ids_by_split:
        train_descriptor = verified_frequent_inputs.graph_cache_descriptors[CorpusSplit.TRAIN]
        train_source_ids = source_expression_ids_by_split[CorpusSplit.TRAIN]
        if (
            raw_partition["full_train_graph_count"] != train_descriptor.success_count
            or raw_partition["source_expression_count"] != train_descriptor.processed_count
            or raw_partition["upstream_failure_count"] != train_descriptor.failure_count
            or raw_partition["source_identity_digest"] != _split_identity_digest(train_source_ids)
            or raw_partition["upstream_failures"]
            != [
                {
                    "error_message": failure.error_message,
                    "error_type": failure.error_type,
                    "expression_id": failure.expression_id,
                    "stage": failure.stage,
                }
                for failure in sorted(
                    _graph_cache_failures(
                        train_descriptor.failures,
                        split=CorpusSplit.TRAIN,
                    ),
                    key=lambda item: item.expression_id,
                )
            ]
        ):
            raise LearnedMotifProtocolError(
                "persisted TRAIN partition disagrees with the authenticated source cache"
            )
    # Refit every lambda and replay validation-only model selection.
    refitted_selectors = tuple(
        fit_ridge_selector(
            replay_eligible_pool,
            target_tuple,
            ridge_lambda=ridge_lambda,
            train_partition_digest=selector.train_partition_digest,
            singular_value_rcond=replay_config.selector.singular_value_rcond,
            training_seed=replay_config.runtime.seed,
        )
        for ridge_lambda in replay_config.selector.ridge_lambdas
    )
    refitted_arms = tuple(
        VocabularyArm(
            arm_id=_lambda_arm_id(refitted.ridge_lambda),
            method=ExperimentMethod.LEARNED,
            vocabulary=_selected_vocabulary(
                replay_eligible_pool,
                select_learned_templates(
                    replay_eligible_pool,
                    refitted,
                    budget=budget,
                ),
                budget=budget,
            ),
            budget=budget,
            ridge_lambda=refitted.ridge_lambda,
        )
        for refitted in refitted_selectors
    )
    # Validate the selected candidate and all equal-budget validation baselines.
    validation = _json_object(
        artifact_data["validation_results"],
        label="learned validation results",
    )
    if (
        set(validation)
        != {
            "candidates",
            "frequent",
            "macro",
            "random",
            "schema_version",
            "selected_candidate_index",
        }
        or validation.get("schema_version") != "geml-goal5-learned-validation-results-v1"
        or not isinstance(validation.get("candidates"), list)
        or not isinstance(validation.get("random"), list)
        or len(validation["random"]) != 30
    ):
        raise LearnedMotifProtocolError("learned validation artifact is inconsistent")
    raw_candidates = validation["candidates"]
    selected_index = validation.get("selected_candidate_index")
    if (
        not raw_candidates
        or len(raw_candidates) != len(refitted_selectors)
        or isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or not 0 <= selected_index < len(raw_candidates)
    ):
        raise LearnedMotifProtocolError("learned validation selection index is invalid")
    candidate_evaluations: list[VocabularyEvaluation] = []
    candidate_selector_digests: list[str] = []
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict) or set(candidate) != {
            "evaluation",
            "selector_payload_sha256",
        }:
            raise LearnedMotifProtocolError("learned validation candidate fields are invalid")
        candidate_digest = _require_sha256(
            candidate["selector_payload_sha256"],
            label="validation selector payload digest",
        )
        expected_selector = refitted_selectors[index]
        if candidate_digest != selector_payload_digest(expected_selector):
            raise LearnedMotifProtocolError(
                "validation selector digest does not match exact TRAIN refit"
            )
        candidate_selector_digests.append(candidate_digest)
        evaluation = _evaluation_from_payload(candidate["evaluation"])
        expected_arm = refitted_arms[index]
        _validate_evaluation_arm_binding(evaluation, expected_arm)
        if evaluation.split is not CorpusSplit.VALIDATION:
            raise LearnedMotifProtocolError("learned candidate evaluation is not VALIDATION")
        candidate_evaluations.append(evaluation)
    selected_evaluation = candidate_evaluations[selected_index]
    eligible_candidate_indexes = [
        index for index, evaluation in enumerate(candidate_evaluations) if evaluation.eligible
    ]
    if not eligible_candidate_indexes:
        raise LearnedMotifProtocolError("persisted validation has no eligible learned candidate")
    expected_selected_index = min(
        eligible_candidate_indexes,
        key=lambda index: (
            candidate_evaluations[index].summary.total_mdl_bits,
            refitted_selectors[index].ridge_lambda,
            candidate_selector_digests[index],
        ),
    )
    if (
        selected_index != expected_selected_index
        or selected_evaluation.split is not CorpusSplit.VALIDATION
        or selected_evaluation.summary.total_mdl_bits != lock.get("validation_total_mdl_bits")
        or candidate_selector_digests[selected_index] != selector_payload_digest(selector)
        or refitted_selectors[selected_index] != selector
    ):
        raise LearnedMotifProtocolError(
            "selected validation candidate is inconsistent with the lock"
        )
    frequent_validation = _evaluation_from_payload(validation.get("frequent"))
    random_validation = tuple(_evaluation_from_payload(item) for item in validation["random"])
    macro_validation = _evaluation_from_payload(validation.get("macro"))
    _validate_evaluation_against_lock_record(
        selected_evaluation,
        arm_id=lock.get("learned_arm_id"),
        method=ExperimentMethod.LEARNED,
        budget=budget,
        vocabulary_id=lock.get("learned_vocabulary_id"),
        vocabulary_payload_sha256=lock.get("learned_vocabulary_payload_sha256"),
        ridge_lambda=selector.ridge_lambda,
        random_seed=None,
        motif_ids=tuple(template.motif_id for template in learned_vocabulary.templates),
        dictionary_bits=vocabulary_mdl_bits(learned_vocabulary.templates),
    )
    _validate_evaluation_against_lock_record(
        frequent_validation,
        arm_id=lock.get("frequent_arm_id"),
        method=ExperimentMethod.FREQUENT,
        budget=budget,
        vocabulary_id=lock.get("frequent_vocabulary_id"),
        vocabulary_payload_sha256=lock.get("frequent_vocabulary_payload_sha256"),
        ridge_lambda=None,
        random_seed=None,
        motif_ids=tuple(template.motif_id for template in frequent_vocabulary.templates),
        dictionary_bits=vocabulary_mdl_bits(frequent_vocabulary.templates),
    )
    for evaluation, arm in zip(random_validation, raw_random_arms, strict=True):
        _validate_evaluation_against_lock_record(
            evaluation,
            arm_id=arm["arm_id"],
            method=ExperimentMethod.RANDOM,
            budget=budget,
            vocabulary_id=arm["vocabulary_id"],
            vocabulary_payload_sha256=arm["vocabulary_payload_sha256"],
            ridge_lambda=None,
            random_seed=arm["seed"],
            motif_ids=arm["motif_ids"],
            dictionary_bits=random_dictionary_bits.get(arm["seed"]),
        )
    if (
        frequent_validation.split is not CorpusSplit.VALIDATION
        or any(evaluation.split is not CorpusSplit.VALIDATION for evaluation in random_validation)
        or macro_validation.split is not CorpusSplit.VALIDATION
        or macro_validation.method is not ExperimentMethod.MACRO
    ):
        raise LearnedMotifProtocolError("validation baselines are inconsistent with the lock")
    _validate_shared_evaluation_evidence(
        split=CorpusSplit.VALIDATION,
        reference=selected_evaluation,
        motif_evaluations=(
            *candidate_evaluations,
            frequent_validation,
            *random_validation,
        ),
        macro=macro_validation,
    )
    if verified_frequent_inputs is not None:
        validation_descriptor = verified_frequent_inputs.graph_cache_descriptors[
            CorpusSplit.VALIDATION
        ]
        expected_validation_failures = tuple(
            sorted(
                _graph_cache_failures(
                    validation_descriptor.failures,
                    split=CorpusSplit.VALIDATION,
                ),
                key=lambda failure: failure.expression_id,
            )
        )
        if (
            selected_evaluation.source_row_count != validation_descriptor.processed_count
            or selected_evaluation.upstream_failures != expected_validation_failures
        ):
            raise LearnedMotifProtocolError(
                "VALIDATION evidence disagrees with the authenticated source cache"
            )

    # Authenticate IID and OOD receipts independently after the selection lock.
    receipt_reports: dict[CorpusSplit, HeldoutSplitReport] = {}
    receipt_expression_ids: dict[CorpusSplit, tuple[str, ...]] = {}
    for split in (CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD):
        receipt_name = f"{split.value}_receipt"
        receipt = _json_object(
            artifact_data[receipt_name],
            label=f"{split.value} learned receipt",
        )
        for name, value in {
            "artifact_version": LEARNED_ARTIFACT_VERSION,
            "config_digest": config_digest,
            "implementation_digest": implementation_digest,
            "lock_digest": lock_digest,
            "schema_version": LEARNED_HELDOUT_SPLIT_VERSION,
            "selection_lock": _ref_payload(references["selection_lock"]),
            "split": split.value,
        }.items():
            if receipt.get(name) != value:
                raise LearnedMotifProtocolError(f"{split.value} receipt has incompatible {name}")
        if set(receipt) != {
            "artifact_version",
            "config_digest",
            "implementation_digest",
            "lock_digest",
            "report",
            "schema_version",
            "selection_lock",
            "source_graph_cache",
            "source_expression_ids",
            "split",
        }:
            raise LearnedMotifProtocolError(f"{split.value} receipt fields are invalid")
        expected_source_cache = source_cache_references.get(split)
        if expected_source_cache is None:
            if receipt["source_graph_cache"] is not None:
                raise LearnedMotifProtocolError(
                    f"{split.value} receipt names an unexpected source cache"
                )
        elif (
            _artifact_ref(
                receipt["source_graph_cache"],
                label=f"{split.value} source graph cache",
            )
            != expected_source_cache
        ):
            raise LearnedMotifProtocolError(
                f"{split.value} receipt references the wrong source cache"
            )
        report = _heldout_from_payload(receipt["report"])
        _validate_report_against_lock_payload(
            report,
            lock=lock,
            budget=budget,
            learned_vocabulary=learned_vocabulary,
            frequent_vocabulary=frequent_vocabulary,
            random_dictionary_bits=random_dictionary_bits,
        )
        raw_expression_ids = receipt["source_expression_ids"]
        if (
            not isinstance(raw_expression_ids, list)
            or any(
                not isinstance(expression_id, str) or not expression_id.strip()
                for expression_id in raw_expression_ids
            )
            or raw_expression_ids != sorted(raw_expression_ids)
            or len(set(raw_expression_ids)) != len(raw_expression_ids)
            or len(raw_expression_ids) != report.learned.source_row_count
        ):
            raise LearnedMotifProtocolError(f"{split.value} receipt identity evidence is invalid")
        if verified_frequent_inputs is not None:
            descriptor = verified_frequent_inputs.graph_cache_descriptors[split]
            expected_failures = tuple(
                sorted(
                    _graph_cache_failures(
                        descriptor.failures,
                        split=split,
                    ),
                    key=lambda failure: failure.expression_id,
                )
            )
            if (
                tuple(raw_expression_ids) != source_expression_ids_by_split[split]
                or report.learned.source_row_count != descriptor.processed_count
                or report.learned.upstream_failures != expected_failures
            ):
                raise LearnedMotifProtocolError(
                    f"{split.value} receipt disagrees with the authenticated source cache"
                )
        receipt_reports[split] = report
        receipt_expression_ids[split] = tuple(raw_expression_ids)

    heldout_overlap = set(receipt_expression_ids[CorpusSplit.TEST_IID]).intersection(
        receipt_expression_ids[CorpusSplit.TEST_OOD]
    )
    if heldout_overlap:
        raise LearnedMotifProtocolError(
            f"held-out receipts reuse expression {min(heldout_overlap)!r}"
        )

    # Bind the aggregate held-out artifact to the two authenticated receipts.
    heldout = _json_object(
        artifact_data["heldout_results"],
        label="learned held-out aggregate",
    )
    expected_receipts = {
        CorpusSplit.TEST_IID.value: _ref_payload(references["test_iid_receipt"]),
        CorpusSplit.TEST_OOD.value: _ref_payload(references["test_ood_receipt"]),
    }
    if (
        set(heldout) != {"schema_version", "split_receipts", "test_iid", "test_ood"}
        or heldout.get("schema_version") != "geml-goal5-learned-heldout-results-v1"
        or heldout.get("split_receipts") != expected_receipts
        or heldout.get("test_iid") != _heldout_payload(receipt_reports[CorpusSplit.TEST_IID])
        or heldout.get("test_ood") != _heldout_payload(receipt_reports[CorpusSplit.TEST_OOD])
    ):
        raise LearnedMotifProtocolError("learned held-out aggregate is inconsistent")

    # Validate the final public result and its honest held-out claim status.
    result_payload = _json_object(
        artifact_data["experiment_result"],
        label="learned experiment result",
    )
    for name, value in {
        "artifact_version": LEARNED_ARTIFACT_VERSION,
        "claim_status": claim_status.value,
        "config_digest": config_digest,
        "frequent_sweep_provenance": (
            None if provenance is None else _frequent_provenance_payload(provenance)
        ),
        "implementation_digest": implementation_digest,
        "locked_selection_digest": lock_digest,
        "schema_version": LEARNED_RESULT_VERSION,
        "test_iid": _heldout_payload(receipt_reports[CorpusSplit.TEST_IID]),
        "test_ood": _heldout_payload(receipt_reports[CorpusSplit.TEST_OOD]),
        "validation": validation,
    }.items():
        if result_payload.get(name) != value:
            raise LearnedMotifProtocolError(f"learned experiment result has incompatible {name}")
    if (
        set(result_payload)
        != {
            "artifact_version",
            "claim_message",
            "claim_status",
            "config_digest",
            "frequent_sweep_provenance",
            "implementation_digest",
            "locked_selection_digest",
            "schema_version",
            "test_iid",
            "test_ood",
            "validation",
        }
        or not isinstance(result_payload["claim_message"], str)
        or not result_payload["claim_message"].strip()
    ):
        raise LearnedMotifProtocolError("learned experiment result fields are invalid")
    expected_claim = (
        HeldoutClaimStatus.SUPPORTED
        if receipt_reports[CorpusSplit.TEST_IID].learned_beats_all_baselines
        else HeldoutClaimStatus.NULL_RESULT
    )
    if claim_status is not expected_claim:
        raise LearnedMotifProtocolError("learned completion claim status is inconsistent")

    # Rerun the full scientific pipeline and require byte-identical evidence.
    _replay_completed_scientific_evidence(
        config=replay_config,
        candidate_pool=replay_candidate_pool,
        selected_frequent=frequent_vocabulary,
        split_loader=replay_split_loader,
        train_partition=replay_train_partition,
        frequent_inputs=verified_frequent_inputs,
        config_digest=config_digest,
        implementation_digest=implementation_digest,
        frequent_provenance=provenance,
        source_graph_cache_references=(
            None if verified_frequent_inputs is None else source_cache_references
        ),
        references=references,
        artifact_data=artifact_data,
        completion=complete,
        completion_data=completion_data,
    )

    # Expose only paths and objects that survived the complete replay.
    return CompletedLearnedMotifRun(
        run_dir=root,
        learned_vocabulary=learned_vocabulary,
        frequent_vocabulary=frequent_vocabulary,
        selection_lock_path=_artifact_path(
            root,
            references["selection_lock"],
        ),
        heldout_results_path=_artifact_path(
            root,
            references["heldout_results"],
        ),
        experiment_result_path=_artifact_path(
            root,
            references["experiment_result"],
        ),
        completion_path=completion_path,
        claim_status=claim_status,
        config_digest=config_digest,
        implementation_digest=implementation_digest,
        frequent_provenance=provenance,
    )


type SplitGraphLoader = Callable[[CorpusSplit], SplitGraphBatch]


def run_learned_motif_experiment(
    split_loader: SplitGraphLoader,
    *,
    candidate_pool: MotifVocabulary,
    selected_frequent: MotifVocabulary,
    config: Goal5LearnedMotifConfig,
    config_digest: str,
    output_dir: str | Path,
    reproduction_command: str,
    implementation_digest: str | None = None,
    frequent_provenance: FrequentSweepProvenance | None = None,
    source_graph_cache_references: (Mapping[CorpusSplit, ArtifactReference] | None) = None,
    train_partition: TrainPartition | None = None,
) -> LearnedMotifExperimentResult:
    """Run and persist 5-6 while enforcing lock-before-held-out load order."""

    _LOGGER.info("loading TRAIN selection partition")
    train = split_loader(CorpusSplit.TRAIN) if train_partition is None else train_partition.batch
    _LOGGER.info("loading VALIDATION graphs")
    validation_batch = split_loader(CorpusSplit.VALIDATION)
    source_references = (
        {} if source_graph_cache_references is None else dict(source_graph_cache_references)
    )
    if source_graph_cache_references is not None and set(source_references) != {
        split for split in CorpusSplit
    }:
        raise ValueError("source_graph_cache_references must cover all four splits")
    _LOGGER.info("fitting selector and evaluating locked arms on VALIDATION")
    locked, validation = fit_and_lock_selection(
        train,
        validation_batch,
        candidate_pool=candidate_pool,
        selected_frequent=selected_frequent,
        config=config,
        config_digest=config_digest,
        implementation_digest=implementation_digest,
        frequent_provenance=frequent_provenance,
        train_partition=train_partition,
        source_graph_cache_references=source_graph_cache_references,
    )
    train_validation_ids = frozenset(
        locked.training_target_audit.partition.source_expression_ids
    ).union(_batch_expression_ids(validation_batch))
    if len(train_validation_ids) != (
        len(locked.training_target_audit.partition.source_expression_ids)
        + validation_batch.source_row_count
    ):
        raise LearnedMotifProtocolError(
            "TRAIN and VALIDATION reuse at least one expression identity"
        )
    _LOGGER.info("persisting selection lock before held-out access")
    locked_artifacts = persist_locked_experiment(
        locked,
        validation,
        output_dir,
        resume=config.runtime.resume,
    )

    del validation_batch

    # Each immutable receipt is published immediately after its split is
    # evaluated, and is authenticated instead of repeating test work on resume.
    test_iid, test_iid_ref, test_iid_ids = _evaluate_or_load_heldout(
        split_loader,
        split=CorpusSplit.TEST_IID,
        locked=locked,
        locked_artifacts=locked_artifacts,
        output_dir=output_dir,
        source_reference=source_references.get(CorpusSplit.TEST_IID),
        resume=config.runtime.resume,
        comparison_expression_ids=train_validation_ids,
    )
    _LOGGER.info("TEST_IID receipt checkpoint is complete")
    test_ood, test_ood_ref, _ = _evaluate_or_load_heldout(
        split_loader,
        split=CorpusSplit.TEST_OOD,
        locked=locked,
        locked_artifacts=locked_artifacts,
        output_dir=output_dir,
        source_reference=source_references.get(CorpusSplit.TEST_OOD),
        resume=config.runtime.resume,
        comparison_expression_ids=train_validation_ids.union(
            test_iid_ids,
        ),
    )
    _LOGGER.info("TEST_OOD receipt checkpoint is complete")
    claim_status, claim_message = _claim_from_iid(test_iid)
    result = LearnedMotifExperimentResult(
        locked=locked,
        validation=validation,
        test_iid=test_iid,
        test_ood=test_ood,
        claim_status=claim_status,
        claim_message=claim_message,
    )
    finalize_experiment(
        result,
        locked_artifacts,
        HeldoutArtifactPaths(
            test_iid=test_iid_ref,
            test_ood=test_ood_ref,
        ),
        output_dir,
        reproduction_command=reproduction_command,
        resume=config.runtime.resume,
    )
    _LOGGER.info("learned-motif completion checkpoint is complete")
    return result


def _learned_run_directory(
    loaded: LoadedLearnedMotifConfig,
    inputs: FrequentSweepInputs,
    *,
    implementation_digest: str,
) -> Path:
    return loaded.output_root / (
        f"{loaded.config_digest[:12]}-"
        f"{inputs.selection_lock_sha256[:12]}-"
        f"{implementation_digest[:12]}"
    )


def run_learned_motifs(
    config_path: str | Path,
    *,
    frequent_sweep_run_dir: str | Path | None = None,
) -> CompletedLearnedMotifRun:
    """Execute issue 5-6 from one authenticated completed issue 5-5 run."""

    loaded = load_learned_motif_config(
        config_path,
        frequent_sweep_run_dir=frequent_sweep_run_dir,
        require_inputs=True,
    )
    if loaded.frequent_sweep_run_dir is None:  # pragma: no cover - loader proves it
        raise LearnedMotifConfigurationError("frequent sweep run directory is required")
    inputs = load_frequent_sweep_inputs(loaded.frequent_sweep_run_dir)
    implementation_digest = learned_implementation_digest(loaded.repository_root)
    run_dir = _learned_run_directory(
        loaded,
        inputs,
        implementation_digest=implementation_digest,
    )
    completion_path = run_dir / "run.complete.json"
    if completion_path.is_file():
        return load_completed_learned_motif_run(
            run_dir,
            expected_config_digest=loaded.config_digest,
            expected_implementation_digest=implementation_digest,
            expected_frequent_provenance=inputs.provenance,
        )

    config_relative = loaded.config_path.relative_to(loaded.repository_root).as_posix()
    frequent_relative = inputs.run_dir.relative_to(loaded.repository_root).as_posix()
    reproduction_command = (
        "python -m geml.experiments.goal5.learned_motifs "
        f"--config {config_relative} "
        f"--frequent-sweep-run-dir {frequent_relative}"
    )
    train_partition = load_frequent_train_partition(
        inputs,
        fraction=loaded.config.selector.train_fit_fraction,
        seed=loaded.config.runtime.seed,
        maximum_graphs=loaded.config.selector.maximum_target_graphs,
    )
    run_learned_motif_experiment(
        frequent_graph_split_loader(inputs),
        candidate_pool=inputs.candidate_pool,
        selected_frequent=inputs.selected_frequent,
        config=loaded.config,
        config_digest=loaded.config_digest,
        output_dir=run_dir,
        reproduction_command=reproduction_command,
        implementation_digest=implementation_digest,
        frequent_provenance=inputs.provenance,
        source_graph_cache_references=inputs.graph_cache_references,
        train_partition=train_partition,
    )
    return load_completed_learned_motif_run(
        run_dir,
        expected_config_digest=loaded.config_digest,
        expected_implementation_digest=implementation_digest,
        expected_frequent_provenance=inputs.provenance,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/goal5_learned_motifs.yaml",
        help="Goal 5 learned-motif YAML configuration",
    )
    parser.add_argument(
        "--frequent-sweep-run-dir",
        default=None,
        help=("completed issue 5-5 run directory; overrides the config value"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    arguments = _parser().parse_args(argv)
    result = run_learned_motifs(
        arguments.config,
        frequent_sweep_run_dir=arguments.frequent_sweep_run_dir,
    )
    print(result.completion_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
