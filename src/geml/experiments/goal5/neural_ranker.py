"""Run the issue 5-7 neural e-graph candidate-ranking baseline.

The experiment authenticates Goal 4 rows and their source corpus, deterministically
replays every cycle-safe candidate group, fits only on TRAIN, selects ridge only on
VALIDATION, then reports TEST_IID and TEST_OOD once from the locked model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import shlex
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import psutil
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

from geml.contracts.corpus import CorpusSplit
from geml.data.egraph_candidate_dataset import (
    DatasetSummary,
    Goal4RunContract,
    Goal4Unit,
    iter_replayed_candidate_groups,
    load_goal4_run_contract,
    load_goal4_units,
    load_required_source_records,
    summarize_candidate_groups,
)
from geml.learning.egraph_ranker import (
    MODEL_VERSION,
    CandidateGroup,
    EGraphRanker,
    RankerFitResult,
    RankingMethod,
    SplitEvaluation,
    evaluate_candidate_groups,
    fit_egraph_ranker,
    heuristics_outperform_neural,
    neural_outperforms_all_heuristics,
)

CONFIG_VERSION = "geml-goal5-neural-ranker-config-v1"
RUN_VERSION = "geml-goal5-neural-ranker-run-v1"
LOCK_VERSION = "geml-goal5-neural-ranker-selection-lock-v1"
REPORT_VERSION = "geml-goal5-neural-ranker-report-v1"
COMPLETE_VERSION = "geml-goal5-neural-ranker-complete-v1"

_LOGGER = logging.getLogger(__name__)

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_PositiveInt = Annotated[StrictInt, Field(gt=0)]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class NeuralRankerConfigurationError(ValueError):
    """Configuration or authenticated input is invalid."""


class NeuralRankerProtocolError(ValueError):
    """A persistence, grouping, or leakage invariant is violated."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )


class Goal4InputConfig(_FrozenModel):
    """Immutable Goal 4 and source-corpus identities."""

    rows_path: _NonBlankStr
    rows_sha256: _Sha256
    run_manifest_path: _NonBlankStr
    run_manifest_sha256: _Sha256
    run_id: _Sha256
    corpus_manifest_path: _NonBlankStr
    corpus_manifest_sha256: _Sha256


class DatasetConfig(_FrozenModel):
    """Bounded replay and checkpoint policy."""

    method: Literal["fixed_cycle_safe_goal4_replay_v1"]
    exact_group_key: Literal["expression_id_and_rewrite_mode"]
    label: Literal["official_v4_pure_eml_dag_node_count"]
    retain_failed_candidates: Literal[True]
    require_summary_match: Literal[True]
    worker_processes: Annotated[StrictInt, Field(ge=1, le=8)]
    worker_chunksize: _PositiveInt
    checkpoint_every_groups: _PositiveInt
    log_every_groups: _PositiveInt


class ModelConfig(_FrozenModel):
    """Lightweight deterministic neural model settings."""

    method: Literal["geml-egraph-random-feature-ranker-v1"] = MODEL_VERSION
    seed: _NonNegativeInt
    hidden_units: Annotated[StrictInt, Field(ge=2, le=256)]
    ridge_values: tuple[Annotated[StrictFloat, Field(ge=0.0)], ...] = Field(min_length=1)
    target_transform: Literal["log1p"]
    group_equal_weighting: Literal[True]
    validation_selects_ridge: Literal[True]

    @field_validator("ridge_values", mode="before")
    @classmethod
    def normalize_yaml_sequence(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("ridge_values")
    @classmethod
    def require_increasing_ridges(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if tuple(sorted(set(values))) != values:
            raise ValueError("ridge_values must be unique and increasing")
        return values


class EvaluationConfig(_FrozenModel):
    """Required baselines, held-out policy, and metric definitions."""

    methods: tuple[
        Literal[
            "exact_official_eml_dag",
            "neural_ranker",
            "estimated_eml_tree_cost",
            "ast_dag_cost",
            "deterministic_random",
        ],
        ...,
    ]
    random_seed: _NonNegativeInt
    primary_test_split: Literal["test_iid"]
    report_test_ood_separately: Literal[True]
    speedup_scope: Literal["official_eml_dag_cost_scoring_only"]
    retain_failed_selections: Literal[True]
    heuristic_comparison_order: Literal["validation_rate_then_mean_regret_then_exact_best_match"]

    @field_validator("methods", mode="before")
    @classmethod
    def normalize_yaml_methods(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("methods")
    @classmethod
    def require_all_methods(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        expected = tuple(method.value for method in RankingMethod)
        if values != expected:
            raise ValueError(f"methods must be exactly {expected!r}")
        return values


class RuntimeConfig(_FrozenModel):
    output_root: _NonBlankStr
    resume: StrictBool
    atomic_finalization: Literal[True]


class Goal5NeuralRankerConfig(_FrozenModel):
    """Complete strict issue 5-7 protocol."""

    schema_version: Literal["geml-goal5-neural-ranker-config-v1"] = CONFIG_VERSION
    goal4: Goal4InputConfig
    dataset: DatasetConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig


@dataclass(frozen=True, slots=True)
class LoadedNeuralRankerConfig:
    config: Goal5NeuralRankerConfig
    repository_root: Path
    config_path: Path
    rows_path: Path
    run_manifest_path: Path
    corpus_manifest_path: Path
    output_root: Path
    config_digest: str


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    path: str
    sha256: str
    byte_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CompletedNeuralRankerRun:
    run_dir: Path
    completion_path: Path
    report_path: Path
    model_path: Path
    groups_path: Path
    report: Mapping[str, object]


@dataclass(slots=True)
class _ResourceMonitor:
    """Sample parent-plus-worker resident memory without affecting model inputs."""

    peak_process_tree_rss_bytes: int = 0
    sample_count: int = 0

    def sample(self) -> int:
        processes = [psutil.Process(os.getpid())]
        with suppress(psutil.Error, OSError):
            processes.extend(processes[0].children(recursive=True))
        observed = 0
        for process in processes:
            try:
                observed += process.memory_info().rss
            except (psutil.Error, OSError):
                continue
        self.peak_process_tree_rss_bytes = max(
            self.peak_process_tree_rss_bytes,
            observed,
        )
        self.sample_count += 1
        return observed


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
        raise NeuralRankerProtocolError(f"value is not strict JSON: {error}") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neural_ranker_config_digest(config: Goal5NeuralRankerConfig) -> str:
    return _sha256_bytes(_canonical_json_bytes(config.model_dump(mode="json")))


def neural_ranker_implementation_digest(repository_root: str | Path) -> str:
    """Fingerprint exactly the issue-owned Python implementation."""

    root = Path(repository_root).resolve()
    relative_paths = (
        Path("src/geml/data/egraph_candidate_dataset.py"),
        Path("src/geml/experiments/goal5/neural_ranker.py"),
        Path("src/geml/learning/egraph_ranker.py"),
    )
    digest = hashlib.sha256(b"geml-goal5-neural-ranker-implementation-v1\0")
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise NeuralRankerConfigurationError(f"missing implementation source: {path}")
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_neural_ranker_config(
    path: str | Path,
    *,
    rows_path: str | Path | None = None,
    run_manifest_path: str | Path | None = None,
    corpus_manifest_path: str | Path | None = None,
    require_inputs: bool = True,
) -> LoadedNeuralRankerConfig:
    """Load strict YAML and resolve optional authenticated input overrides."""

    source = Path(path).resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise NeuralRankerConfigurationError(f"invalid neural-ranker config: {error}") from error
    if not isinstance(raw, dict):
        raise NeuralRankerConfigurationError("neural-ranker config must be a mapping")
    try:
        config = Goal5NeuralRankerConfig.model_validate(raw)
    except Exception as error:
        raise NeuralRankerConfigurationError(f"invalid neural-ranker config: {error}") from error
    repository_root = Path(__file__).resolve().parents[4]
    resolved_rows = _resolve_input_path(
        repository_root,
        rows_path if rows_path is not None else config.goal4.rows_path,
    )
    resolved_run = _resolve_input_path(
        repository_root,
        run_manifest_path if run_manifest_path is not None else config.goal4.run_manifest_path,
    )
    resolved_corpus = _resolve_input_path(
        repository_root,
        corpus_manifest_path
        if corpus_manifest_path is not None
        else config.goal4.corpus_manifest_path,
    )
    output_root = _resolve_input_path(repository_root, config.runtime.output_root)
    if require_inputs:
        for label, input_path in (
            ("Goal 4 rows", resolved_rows),
            ("Goal 4 run manifest", resolved_run),
            ("source corpus manifest", resolved_corpus),
        ):
            if not input_path.is_file():
                raise NeuralRankerConfigurationError(f"missing {label}: {input_path}")
    return LoadedNeuralRankerConfig(
        config=config,
        repository_root=repository_root,
        config_path=source,
        rows_path=resolved_rows,
        run_manifest_path=resolved_run,
        corpus_manifest_path=resolved_corpus,
        output_root=output_root,
        config_digest=neural_ranker_config_digest(config),
    )


def _resolve_input_path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _reproduction_path(path: Path, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return str(path)


def _authenticate_inputs(loaded: LoadedNeuralRankerConfig) -> Goal4RunContract:
    expected = loaded.config.goal4
    for label, path, expected_digest in (
        ("Goal 4 rows", loaded.rows_path, expected.rows_sha256),
        ("Goal 4 run manifest", loaded.run_manifest_path, expected.run_manifest_sha256),
        ("source corpus manifest", loaded.corpus_manifest_path, expected.corpus_manifest_sha256),
    ):
        observed = _sha256_file(path)
        if observed != expected_digest:
            raise NeuralRankerConfigurationError(
                f"{label} SHA-256 mismatch: expected {expected_digest}, observed {observed}"
            )
    run = load_goal4_run_contract(loaded.run_manifest_path)
    if (
        run.run_id != expected.run_id
        or run.source_manifest_sha256 != expected.corpus_manifest_sha256
    ):
        raise NeuralRankerConfigurationError(
            "Goal 4 run identity does not match the configured source corpus"
        )
    return run


def _run_directory(
    loaded: LoadedNeuralRankerConfig,
    *,
    implementation_digest: str,
) -> Path:
    return loaded.output_root / (
        f"{loaded.config_digest[:12]}-"
        f"{loaded.config.goal4.rows_sha256[:12]}-"
        f"{implementation_digest[:12]}"
    )


def _run_manifest_payload(
    loaded: LoadedNeuralRankerConfig,
    run: Goal4RunContract,
    *,
    implementation_digest: str,
    run_id: str,
    reproduction_command: str,
) -> dict[str, object]:
    return {
        "config": loaded.config.model_dump(mode="json"),
        "config_digest": loaded.config_digest,
        "goal4": {
            "config_sha256": run.config_sha256,
            "implementation_commit": run.implementation_commit,
            "rows_sha256": loaded.config.goal4.rows_sha256,
            "run_id": run.run_id,
            "run_manifest_sha256": loaded.config.goal4.run_manifest_sha256,
            "source_manifest_sha256": run.source_manifest_sha256,
        },
        "implementation_digest": implementation_digest,
        "platform": platform.platform(),
        "python_version": sys.version,
        "reproduction_command": reproduction_command,
        "run_id": run_id,
        "schema_version": RUN_VERSION,
    }


def _run_id(
    loaded: LoadedNeuralRankerConfig,
    run: Goal4RunContract,
    implementation_digest: str,
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "config_digest": loaded.config_digest,
                "goal4_run_id": run.run_id,
                "implementation_digest": implementation_digest,
                "rows_sha256": loaded.config.goal4.rows_sha256,
                "schema_version": RUN_VERSION,
            }
        )
    )


def _load_existing_group_prefix(
    path: Path,
    units: Sequence[Goal4Unit],
) -> tuple[CandidateGroup, ...]:
    if not path.exists():
        return ()
    groups: list[CandidateGroup] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                group = CandidateGroup.from_dict(json.loads(line))
                if line_number > len(units):
                    raise NeuralRankerProtocolError("candidate artifact has excess groups")
                unit = units[line_number - 1]
                if (
                    group.expression_id != unit.expression_id
                    or group.rewrite_mode != unit.rewrite_mode.value
                ):
                    raise NeuralRankerProtocolError(
                        "resumed candidate artifact is not a canonical unit prefix"
                    )
                groups.append(group)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, NeuralRankerProtocolError):
            raise
        raise NeuralRankerProtocolError(
            f"invalid resumed candidate artifact at {path}: {error}"
        ) from error
    return tuple(groups)


def _append_groups(
    path: Path,
    groups: Iterable[CandidateGroup],
    *,
    start_count: int,
    total_count: int,
    checkpoint_path: Path,
    checkpoint_every: int,
    log_every: int,
    run_id: str,
    replay_started: float,
    prior_replay_wall_seconds: float,
    resource_monitor: _ResourceMonitor,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = start_count
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for group in groups:
            stream.write(_canonical_json_bytes(group.as_dict()).decode())
            stream.write("\n")
            completed += 1
            if completed % checkpoint_every == 0:
                stream.flush()
                os.fsync(stream.fileno())
                resource_monitor.sample()
                _atomic_replace_json(
                    checkpoint_path,
                    _checkpoint_payload(
                        run_id,
                        completed,
                        total_count,
                        replay_active_wall_seconds=(
                            prior_replay_wall_seconds + time.perf_counter() - replay_started
                        ),
                        peak_process_tree_rss_bytes=(resource_monitor.peak_process_tree_rss_bytes),
                    ),
                )
            if completed % log_every == 0 or completed == total_count:
                _LOGGER.info(
                    "replayed %d/%d candidate groups",
                    completed,
                    total_count,
                )
        stream.flush()
        os.fsync(stream.fileno())
    resource_monitor.sample()
    _atomic_replace_json(
        checkpoint_path,
        _checkpoint_payload(
            run_id,
            completed,
            total_count,
            replay_active_wall_seconds=(
                prior_replay_wall_seconds + time.perf_counter() - replay_started
            ),
            peak_process_tree_rss_bytes=resource_monitor.peak_process_tree_rss_bytes,
        ),
    )
    return completed


def _checkpoint_payload(
    run_id: str,
    completed: int,
    total: int,
    *,
    replay_active_wall_seconds: float,
    peak_process_tree_rss_bytes: int,
) -> dict[str, object]:
    return {
        "completed_group_count": completed,
        "peak_process_tree_rss_bytes": peak_process_tree_rss_bytes,
        "replay_active_wall_seconds": replay_active_wall_seconds,
        "run_id": run_id,
        "schema_version": "geml-goal5-neural-ranker-checkpoint-v1",
        "total_group_count": total,
    }


def _checkpoint_resources(path: Path, run_id: str) -> tuple[float, int]:
    if not path.is_file():
        return 0.0, 0
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NeuralRankerProtocolError(f"invalid replay checkpoint: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "geml-goal5-neural-ranker-checkpoint-v1"
        or value.get("run_id") != run_id
    ):
        raise NeuralRankerProtocolError("replay checkpoint identity is incompatible")
    elapsed = value.get("replay_active_wall_seconds")
    peak_rss = value.get("peak_process_tree_rss_bytes")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, int | float)
        or not math.isfinite(elapsed)
        or elapsed < 0
        or isinstance(peak_rss, bool)
        or not isinstance(peak_rss, int)
        or peak_rss < 0
    ):
        raise NeuralRankerProtocolError("replay checkpoint resources are malformed")
    return float(elapsed), peak_rss


def _load_all_groups(path: Path) -> tuple[CandidateGroup, ...]:
    groups: list[CandidateGroup] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                groups.append(CandidateGroup.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as error:
                raise NeuralRankerProtocolError(
                    f"invalid candidate group at {path}:{line_number}: {error}"
                ) from error
    return tuple(groups)


def _fit_and_evaluate(
    groups: tuple[CandidateGroup, ...],
    config: Goal5NeuralRankerConfig,
    run_dir: Path,
    *,
    run_id: str,
) -> tuple[
    RankerFitResult,
    SplitEvaluation,
    SplitEvaluation,
    SplitEvaluation,
    ArtifactReference,
]:
    usable = tuple(group for group in groups if group.replay_status == "matched")
    training = tuple(group for group in usable if group.split is CorpusSplit.TRAIN)
    validation = tuple(group for group in usable if group.split is CorpusSplit.VALIDATION)
    fit = fit_egraph_ranker(
        training,
        validation,
        seed=config.model.seed,
        hidden_units=config.model.hidden_units,
        ridge_values=config.model.ridge_values,
    )
    methods = tuple(RankingMethod(value) for value in config.evaluation.methods)
    validation_evaluation = evaluate_candidate_groups(
        validation,
        split=CorpusSplit.VALIDATION,
        model=fit.model,
        random_seed=config.evaluation.random_seed,
        methods=methods,
    )

    model_ref = _write_json_artifact(run_dir, "model.json", fit.model.as_dict())
    validation_ref = _write_evaluation_artifacts(
        run_dir,
        validation_evaluation,
    )
    lock_ref = _write_json_artifact(
        run_dir,
        "selection.lock.json",
        {
            "heldout_metrics_absent_at_lock": True,
            "model": model_ref.as_dict(),
            "run_id": run_id,
            "schema_version": LOCK_VERSION,
            "selected_ridge": fit.model.ridge,
            "selection_split": CorpusSplit.VALIDATION.value,
            "validation": validation_ref.as_dict(),
        },
    )

    test_iid_groups = tuple(group for group in usable if group.split is CorpusSplit.TEST_IID)
    test_ood_groups = tuple(group for group in usable if group.split is CorpusSplit.TEST_OOD)
    _require_expression_disjointness(training, validation, test_iid_groups, test_ood_groups)
    test_iid = evaluate_candidate_groups(
        test_iid_groups,
        split=CorpusSplit.TEST_IID,
        model=fit.model,
        random_seed=config.evaluation.random_seed,
        methods=methods,
    )
    _write_evaluation_artifacts(run_dir, test_iid)
    test_ood = evaluate_candidate_groups(
        test_ood_groups,
        split=CorpusSplit.TEST_OOD,
        model=fit.model,
        random_seed=config.evaluation.random_seed,
        methods=methods,
    )
    _write_evaluation_artifacts(run_dir, test_ood)
    return fit, validation_evaluation, test_iid, test_ood, lock_ref


def _require_expression_disjointness(*partitions: tuple[CandidateGroup, ...]) -> None:
    seen: set[str] = set()
    for partition in partitions:
        current = {group.expression_id for group in partition}
        overlap = seen & current
        if overlap:
            raise NeuralRankerProtocolError(
                f"corpus split leakage detected for {len(overlap)} expressions"
            )
        seen.update(current)


def _write_evaluation_artifacts(
    run_dir: Path,
    evaluation: SplitEvaluation,
) -> ArtifactReference:
    split_name = evaluation.split.value
    _write_jsonl_artifact(
        run_dir,
        f"{split_name}.outcomes.jsonl",
        (outcome.as_dict() for outcome in evaluation.outcomes),
    )
    return _write_json_artifact(
        run_dir,
        f"{split_name}.summary.json",
        evaluation.as_dict(),
    )


def _scientific_report(
    *,
    run_id: str,
    dataset: DatasetSummary,
    fit: RankerFitResult,
    validation: SplitEvaluation,
    test_iid: SplitEvaluation,
    test_ood: SplitEvaluation,
    reproduction_command: str,
    runtime_evidence: Mapping[str, object],
) -> dict[str, object]:
    iid_outperformers = heuristics_outperform_neural(test_iid)
    ood_outperformers = heuristics_outperform_neural(test_ood)
    iid_neural_wins = neural_outperforms_all_heuristics(test_iid)
    ood_neural_wins = neural_outperforms_all_heuristics(test_ood)
    iid_names = [method.value for method in iid_outperformers]
    ood_names = [method.value for method in ood_outperformers]
    iid_statement = (
        "On TEST_IID, no structural heuristic outperformed the neural model."
        if not iid_names
        else f"On TEST_IID, heuristics outperformed the neural model: {', '.join(iid_names)}."
    )
    ood_statement = (
        "On TEST_OOD, no structural heuristic outperformed the neural model."
        if not ood_names
        else f"On TEST_OOD, heuristics outperformed the neural model: {', '.join(ood_names)}."
    )
    return {
        "dataset": dataset.as_dict(),
        "fit": fit.as_dict(),
        "heuristic_comparison": {
            "comparison_order": (
                "validation rate (higher), then mean regret over valid selections "
                "(lower), then exact-best match rate (higher)"
            ),
            "test_iid": {
                "claim_status": "supported" if iid_neural_wins else "null_result",
                "heuristics_outperform_neural": iid_names,
                "neural_beats_all_heuristics": iid_neural_wins,
                "statement": iid_statement,
            },
            "test_ood": {
                "claim_status": "supported" if ood_neural_wins else "null_result",
                "heuristics_outperform_neural": ood_names,
                "neural_beats_all_heuristics": ood_neural_wins,
                "statement": ood_statement,
            },
        },
        "metric_definitions": {
            "exact_best_match": (
                "selected structural signature equals the minimum valid official "
                "OFFICIAL_V4 EML-DAG-cost signature under Goal 4 tie-breaking"
            ),
            "regret": (
                "selected valid official EML-DAG cost minus exact-best cost; failed "
                "selections are retained separately and excluded from mean regret"
            ),
            "speedup": (
                "official pure EML-DAG cost-scoring work only; enumeration, feature "
                "extraction, model inference, and semantic validation are excluded"
            ),
            "validation_rate": (
                "semantically valid selected candidates divided by all evaluable "
                "candidate groups; failed selections remain in the denominator"
            ),
        },
        "reproduction_command": reproduction_command,
        "runtime": dict(runtime_evidence),
        "run_id": run_id,
        "schema_version": REPORT_VERSION,
        "scientific_scope": (
            "compression-cost ranking baseline only; no claim of mathematical truth, "
            "reasoning ability, or final GNN encoding"
        ),
        "test_iid": test_iid.as_dict(),
        "test_ood": test_ood.as_dict(),
        "validation": validation.as_dict(),
    }


def _write_json_artifact(
    root: Path,
    relative: str,
    payload: object,
) -> ArtifactReference:
    data = _canonical_json_bytes(payload) + b"\n"
    path = root / relative
    _publish_immutable(path, data)
    return ArtifactReference(relative, _sha256_bytes(data), len(data))


def _write_jsonl_artifact(
    root: Path,
    relative: str,
    rows: Iterable[Mapping[str, object]],
) -> ArtifactReference:
    path = root / relative
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".neural-ranker-",
        suffix=".tmp",
        dir=root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for row in rows:
                stream.write(_canonical_json_bytes(row))
                stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        data_digest = _sha256_file(temporary)
        size = temporary.stat().st_size
        _publish_temporary(path, temporary)
    finally:
        temporary.unlink(missing_ok=True)
    return ArtifactReference(relative, data_digest, size)


def _publish_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".neural-ranker-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_temporary(path, temporary)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_temporary(path: Path, temporary: Path) -> None:
    try:
        os.link(temporary, path)
    except FileExistsError:
        if not path.is_file() or _sha256_file(path) != _sha256_file(temporary):
            raise NeuralRankerProtocolError(f"immutable resumed artifact differs: {path}") from None


def _atomic_replace_json(path: Path, payload: object) -> None:
    data = _canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".neural-ranker-checkpoint-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_reference(root: Path, relative: str) -> ArtifactReference:
    path = root / relative
    return ArtifactReference(relative, _sha256_file(path), path.stat().st_size)


def _completion_payload(
    *,
    run_id: str,
    config_digest: str,
    implementation_digest: str,
    run_dir: Path,
    dataset_summary: DatasetSummary,
    lock_ref: ArtifactReference,
    report_ref: ArtifactReference,
) -> dict[str, object]:
    required = (
        "candidate_groups.jsonl",
        "dataset.summary.json",
        "model.json",
        "selection.lock.json",
        "validation.outcomes.jsonl",
        "validation.summary.json",
        "test_iid.outcomes.jsonl",
        "test_iid.summary.json",
        "test_ood.outcomes.jsonl",
        "test_ood.summary.json",
        "report.json",
        "run.manifest.json",
        "runtime.json",
    )
    artifacts = {
        relative: _artifact_reference(run_dir, relative).as_dict() for relative in required
    }
    if artifacts["selection.lock.json"] != lock_ref.as_dict():
        raise NeuralRankerProtocolError("selection-lock artifact identity changed")
    if artifacts["report.json"] != report_ref.as_dict():
        raise NeuralRankerProtocolError("report artifact identity changed")
    return {
        "artifacts": artifacts,
        "config_digest": config_digest,
        "dataset": dataset_summary.as_dict(),
        "implementation_digest": implementation_digest,
        "run_id": run_id,
        "schema_version": COMPLETE_VERSION,
    }


def load_completed_neural_ranker_run(
    run_dir: str | Path,
    *,
    expected_config_digest: str | None = None,
    expected_implementation_digest: str | None = None,
) -> CompletedNeuralRankerRun:
    """Checksum-authenticate a completed issue 5-7 run."""

    root = Path(run_dir).resolve()
    completion_path = root / "run.complete.json"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NeuralRankerProtocolError(f"invalid completion artifact: {error}") from error
    if not isinstance(completion, dict) or completion.get("schema_version") != COMPLETE_VERSION:
        raise NeuralRankerProtocolError("completion schema is incompatible")
    if (
        expected_config_digest is not None
        and completion.get("config_digest") != expected_config_digest
    ):
        raise NeuralRankerProtocolError("completion config digest is incompatible")
    if (
        expected_implementation_digest is not None
        and completion.get("implementation_digest") != expected_implementation_digest
    ):
        raise NeuralRankerProtocolError("completion implementation digest is incompatible")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, dict):
        raise NeuralRankerProtocolError("completion artifacts must be an object")
    for relative, raw_reference in artifacts.items():
        if (
            not isinstance(relative, str)
            or not isinstance(raw_reference, dict)
            or set(raw_reference) != {"byte_count", "path", "sha256"}
            or raw_reference["path"] != relative
        ):
            raise NeuralRankerProtocolError("completion artifact reference is malformed")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise NeuralRankerProtocolError("completion artifact escapes run directory") from error
        if (
            not path.is_file()
            or path.stat().st_size != raw_reference["byte_count"]
            or _sha256_file(path) != raw_reference["sha256"]
        ):
            raise NeuralRankerProtocolError(f"completion artifact failed checksum: {relative}")
    model = EGraphRanker.from_dict(json.loads((root / "model.json").read_text(encoding="utf-8")))
    del model
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_version") != REPORT_VERSION:
        raise NeuralRankerProtocolError("completed report schema is incompatible")
    return CompletedNeuralRankerRun(
        run_dir=root,
        completion_path=completion_path,
        report_path=root / "report.json",
        model_path=root / "model.json",
        groups_path=root / "candidate_groups.jsonl",
        report=report,
    )


def run_neural_ranker(
    config_path: str | Path,
    *,
    rows_path: str | Path | None = None,
    run_manifest_path: str | Path | None = None,
    corpus_manifest_path: str | Path | None = None,
) -> CompletedNeuralRankerRun:
    """Execute and authenticate the complete issue 5-7 experiment."""

    invocation_started = time.perf_counter()
    loaded = load_neural_ranker_config(
        config_path,
        rows_path=rows_path,
        run_manifest_path=run_manifest_path,
        corpus_manifest_path=corpus_manifest_path,
        require_inputs=True,
    )
    run = _authenticate_inputs(loaded)
    implementation_digest = neural_ranker_implementation_digest(loaded.repository_root)
    run_id = _run_id(loaded, run, implementation_digest)
    run_dir = _run_directory(loaded, implementation_digest=implementation_digest)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    prior_replay_wall_seconds, prior_peak_rss = _checkpoint_resources(
        checkpoint_path,
        run_id,
    )
    resource_monitor = _ResourceMonitor(peak_process_tree_rss_bytes=prior_peak_rss)
    resource_monitor.sample()
    completion_path = run_dir / "run.complete.json"
    if completion_path.is_file():
        return load_completed_neural_ranker_run(
            run_dir,
            expected_config_digest=loaded.config_digest,
            expected_implementation_digest=implementation_digest,
        )

    config_argument = _reproduction_path(loaded.config_path, loaded.repository_root)
    reproduction_command = (
        "python -m geml.experiments.goal5.neural_ranker "
        f"--config {shlex.quote(config_argument)} "
        f"--rows-path {shlex.quote(str(loaded.rows_path))} "
        f"--run-manifest-path {shlex.quote(str(loaded.run_manifest_path))} "
        f"--corpus-manifest-path {shlex.quote(str(loaded.corpus_manifest_path))}"
    )
    _write_json_artifact(
        run_dir,
        "run.manifest.json",
        _run_manifest_payload(
            loaded,
            run,
            implementation_digest=implementation_digest,
            run_id=run_id,
            reproduction_command=reproduction_command,
        ),
    )

    replay_started = time.perf_counter()
    _LOGGER.info("loading and validating Goal 4 compact rows")
    units = load_goal4_units(loaded.rows_path, run)
    groups_path = run_dir / "candidate_groups.jsonl"
    existing = _load_existing_group_prefix(groups_path, units)
    pending_units = units[len(existing) :]
    if pending_units:
        _LOGGER.info(
            "loading %d required source expressions for %d pending groups",
            len({unit.expression_id for unit in pending_units}),
            len(pending_units),
        )
        records = load_required_source_records(loaded.corpus_manifest_path, pending_units)
        replayed = iter_replayed_candidate_groups(
            pending_units,
            records,
            include_optional_domain_rules=run.include_optional_domain_rules,
            worker_processes=loaded.config.dataset.worker_processes,
            chunksize=loaded.config.dataset.worker_chunksize,
        )
        completed = _append_groups(
            groups_path,
            replayed,
            start_count=len(existing),
            total_count=len(units),
            checkpoint_path=checkpoint_path,
            checkpoint_every=loaded.config.dataset.checkpoint_every_groups,
            log_every=loaded.config.dataset.log_every_groups,
            run_id=run_id,
            replay_started=replay_started,
            prior_replay_wall_seconds=prior_replay_wall_seconds,
            resource_monitor=resource_monitor,
        )
        if completed != len(units):
            raise NeuralRankerProtocolError("candidate replay ended with missing groups")
        replay_active_wall_seconds = (
            prior_replay_wall_seconds + time.perf_counter() - replay_started
        )
    else:
        replay_active_wall_seconds = prior_replay_wall_seconds

    _LOGGER.info("authenticating complete grouped candidate dataset")
    groups = _load_all_groups(groups_path)
    resource_monitor.sample()
    if len(groups) != len(units):
        raise NeuralRankerProtocolError("candidate dataset does not cover every Goal 4 unit")
    dataset_summary = summarize_candidate_groups(groups)
    if loaded.config.dataset.require_summary_match and dataset_summary.replay_mismatch_count != 0:
        raise NeuralRankerProtocolError(
            f"{dataset_summary.replay_mismatch_count} groups differ from Goal 4 summaries"
        )
    _write_json_artifact(
        run_dir,
        "dataset.summary.json",
        {
            **dataset_summary.as_dict(),
            "candidate_groups_sha256": _sha256_file(groups_path),
            "goal4_rows_sha256": loaded.config.goal4.rows_sha256,
            "label_provenance": (
                "official OFFICIAL_V4 pure EML-DAG cost via "
                "geml.interfaces.eml_dag_cost.compute_eml_dag_cost"
            ),
        },
    )

    _LOGGER.info("fitting TRAIN model and selecting ridge on VALIDATION")
    fitting_started = time.perf_counter()
    fit, validation, test_iid, test_ood, lock_ref = _fit_and_evaluate(
        groups,
        loaded.config,
        run_dir,
        run_id=run_id,
    )
    fitting_wall_seconds = time.perf_counter() - fitting_started
    resource_monitor.sample()
    candidate_cost_timings = [
        candidate.official_cost_scoring_seconds
        for group in groups
        for candidate in group.candidates
        if candidate.official_cost_scoring_seconds is not None
    ]
    runtime_evidence = {
        "candidate_cost_scoring_observation_count": len(candidate_cost_timings),
        "candidate_cost_scoring_total_seconds": sum(candidate_cost_timings),
        "candidate_replay_active_wall_seconds": replay_active_wall_seconds,
        "finalizing_invocation_wall_seconds_before_report": (
            time.perf_counter() - invocation_started
        ),
        "memory_scope": (
            "sampled resident bytes of the coordinator and all recursive replay workers"
        ),
        "model_fit_and_evaluation_wall_seconds": fitting_wall_seconds,
        "peak_process_tree_rss_bytes": resource_monitor.peak_process_tree_rss_bytes,
        "rss_sample_count": resource_monitor.sample_count,
        "rss_sampling_policy": (
            "at startup, every replay checkpoint, after dataset load, and after evaluation"
        ),
        "speedup_scope": "official_eml_dag_cost_scoring_only",
        "worker_processes": loaded.config.dataset.worker_processes,
    }
    _write_json_artifact(run_dir, "runtime.json", runtime_evidence)
    report = _scientific_report(
        run_id=run_id,
        dataset=dataset_summary,
        fit=fit,
        validation=validation,
        test_iid=test_iid,
        test_ood=test_ood,
        reproduction_command=reproduction_command,
        runtime_evidence=runtime_evidence,
    )
    report_ref = _write_json_artifact(run_dir, "report.json", report)
    completion = _completion_payload(
        run_id=run_id,
        config_digest=loaded.config_digest,
        implementation_digest=implementation_digest,
        run_dir=run_dir,
        dataset_summary=dataset_summary,
        lock_ref=lock_ref,
        report_ref=report_ref,
    )
    _write_json_artifact(run_dir, "run.complete.json", completion)
    return load_completed_neural_ranker_run(
        run_dir,
        expected_config_digest=loaded.config_digest,
        expected_implementation_digest=implementation_digest,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/goal5_neural_ranker.yaml",
    )
    parser.add_argument("--rows-path", default=None)
    parser.add_argument("--run-manifest-path", default=None)
    parser.add_argument("--corpus-manifest-path", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    arguments = _parser().parse_args(argv)
    completed = run_neural_ranker(
        arguments.config,
        rows_path=arguments.rows_path,
        run_manifest_path=arguments.run_manifest_path,
        corpus_manifest_path=arguments.corpus_manifest_path,
    )
    print(completed.completion_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
