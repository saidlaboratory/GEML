"""Goal 9 guided symbolic-regression runner (issue 9-1).

Drives the matched-budget EML-guided and AST-guided arms over a frozen benchmark manifest,
one atomic checkpoint per (task, method, seed). The runner is deliberately thin: the fair
comparison contract lives in ``geml.learning.sr.guided_search`` so that the baselines in
issue 9-2 share it rather than re-implementing it.

Phase A runs only tiny fixture cells with injected scorers. The production sweep is deferred
until the shared compact model and harness merge and the Goal 9 verification-scope decision
is made. Until then a sweep refuses to run on the untrained fixture scorer unless the
configuration explicitly marks it as a fixture smoke (``allow_fixture_scorers``).
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from geml.data.sr.benchmark import (
    FROZEN_SEEDS,
    BenchmarkManifest,
    EGraphFragmentEquivalenceVerifier,
    EquivalenceVerifier,
    ManifestStatus,
    ObservationRole,
    ObservationSet,
    SRSplitRole,
    SRTask,
    load_manifest,
    load_observation_sets,
    load_tasks,
)
from geml.learning.sr.guided_search import (
    FIXTURE_SCORER_ID_PREFIX,
    CandidateScorer,
    ErrorPriorityScorer,
    RunStatus,
    SearchBudget,
    SRMethod,
    SRMethodResult,
    SRRepresentation,
    assert_matched_budgets,
    run_guided_search,
    write_result,
)

PRODUCTION_OUTPUT_ROOT = "outputs/final/goal9/guided_search"

_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

#: The two controlled arms and the representation each one materialises.
CONTROLLED_ARMS: tuple[tuple[SRMethod, SRRepresentation], ...] = (
    (SRMethod.EML_GUIDED, SRRepresentation.PURE_EML),
    (SRMethod.AST_GUIDED, SRRepresentation.AST),
)


class Goal9RunError(ValueError):
    """A run configuration or input manifest was invalid."""


class UntrainedScorerError(Goal9RunError):
    """A non-fixture sweep was asked to run without a trained guidance model."""


class GuidedSearchConfig(BaseModel):
    """Configuration for the matched EML/AST guided-search comparison."""

    model_config = _FROZEN

    schema_version: str = "geml-goal9-sr-config-v1"
    output_root: str = PRODUCTION_OUTPUT_ROOT
    seeds: tuple[int, ...] = FROZEN_SEEDS
    budget: SearchBudget
    resume: bool = True
    verifier_timeout_seconds: float = Field(default=5.0, gt=0.0)
    require_benchmark_test_split: bool = True
    allow_unfrozen_benchmark: bool = False
    allow_fixture_scorers: bool = False

    @model_validator(mode="after")
    def _seeds_are_distinct(self) -> "GuidedSearchConfig":
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        return self


class RunEnvelope(BaseModel):
    """Minimal reproduction envelope for one Goal 9 guided-search sweep.

    The canonical ``RunEnvelopeV1`` is owned by issues 6-1/6-4. This is a compatibility shim
    carrying only the fields this runner can honestly fill on its own; it is marked for
    deletion once that contract merges.
    """

    model_config = _FROZEN

    schema_version: str = "geml-goal9-sr-run-envelope-shim-v1"
    compatibility_shim: bool = True
    delete_after_integration_with: str = "RunEnvelopeV1 (issues 6-1 / 6-4)"
    config_hash: str
    config_path: str
    benchmark_id: str
    benchmark_status: str
    budget_digest: str
    seeds: tuple[int, ...]
    python_version: str
    platform: str
    package_versions: dict[str, str]
    started_at: str
    finished_at: str
    wall_seconds: float = Field(ge=0.0)
    cells_planned: int = Field(ge=0)
    cells_completed: int = Field(ge=0)
    cells_resumed: int = Field(ge=0)
    cells_failed: int = Field(ge=0)
    reproduction_command: str


def load_config(path: str | Path) -> tuple[GuidedSearchConfig, str]:
    """Load a YAML configuration and return it with the SHA-256 of the exact file bytes."""

    raw = Path(path).read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8")) or {}
    return GuidedSearchConfig.model_validate(payload), hashlib.sha256(raw).hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("sympy", "pydantic", "numpy", "PyYAML"):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:  # a missing version is recorded, not fatal
            versions[name] = "unknown"
    return versions


def select_tasks(
    tasks: Sequence[SRTask], manifest: BenchmarkManifest, config: GuidedSearchConfig
) -> tuple[SRTask, ...]:
    """Return the tasks this sweep may score.

    Guards against two silent errors: scoring a manifest that is not frozen, and scoring a
    development task that was only ever meant to select a configuration.
    """

    if manifest.status is not ManifestStatus.COMPLETE and not config.allow_unfrozen_benchmark:
        raise Goal9RunError(
            f"benchmark manifest status is {manifest.status.value}; refusing to score an "
            "unfrozen benchmark. Set allow_unfrozen_benchmark only for fixture smokes."
        )
    selected = tuple(
        task
        for task in tasks
        if task.task_id in set(manifest.task_ids)
        and (
            not config.require_benchmark_test_split or task.split_role is SRSplitRole.BENCHMARK_TEST
        )
    )
    if not selected:
        raise Goal9RunError("no benchmark-test tasks were selected from the manifest")
    return selected


def _observation_index(
    observation_sets: Sequence[ObservationSet],
) -> dict[tuple[str, ObservationRole], ObservationSet]:
    return {(item.task_id, item.role): item for item in observation_sets}


def build_scorers(
    factory: "object | None" = None,
    *,
    allow_fixture: bool = False,
) -> dict[SRRepresentation, CandidateScorer]:
    """Return one scorer per controlled representation.

    ``factory`` lets a caller inject the shared compact model once Workstream 2 merges.
    Without a factory the deterministic Phase-A fixture scorer is returned only when
    ``allow_fixture`` marks the sweep as a fixture smoke; a production sweep must never fall
    back silently to an untrained heuristic.
    """

    if factory is None:
        if not allow_fixture:
            raise UntrainedScorerError(
                "no trained-scorer factory was injected; refusing to sweep with the "
                "untrained fixture scorer. Set allow_fixture_scorers only for fixture smokes."
            )
        return {
            representation: ErrorPriorityScorer(representation)
            for _method, representation in CONTROLLED_ARMS
        }
    return {
        representation: factory(representation)  # type: ignore[operator]
        for _method, representation in CONTROLLED_ARMS
    }


def run_sweep(
    *,
    tasks: Sequence[SRTask],
    observation_sets: Sequence[ObservationSet],
    config: GuidedSearchConfig,
    output_root: str | Path,
    scorers: dict[SRRepresentation, CandidateScorer] | None = None,
    verifier: EquivalenceVerifier | None = None,
) -> tuple[tuple[SRMethodResult, ...], dict[str, int]]:
    """Run every (task, arm, seed) cell under one shared budget.

    Both arms are validated to have received a byte-identical budget before any cell runs,
    and the shared digest is stamped on every emitted row. Unless the configuration
    explicitly marks the sweep as a fixture smoke, every scorer — injected or not — must
    declare a trained identity, so a production sweep cannot run on an untrained heuristic.
    """

    budgets = {method.value: config.budget for method, _ in CONTROLLED_ARMS}
    digest = assert_matched_budgets(budgets)
    active_scorers = scorers or build_scorers(allow_fixture=config.allow_fixture_scorers)
    if not config.allow_fixture_scorers:
        for scorer in active_scorers.values():
            descriptor = scorer.descriptor()
            if (
                descriptor.scorer_id.startswith(FIXTURE_SCORER_ID_PREFIX)
                or descriptor.training_data_policy == "not_trained"
            ):
                raise UntrainedScorerError(
                    f"scorer {descriptor.scorer_id!r} declares training_data_policy="
                    f"{descriptor.training_data_policy!r}; a sweep without "
                    "allow_fixture_scorers requires trained scorers"
                )
    active_verifier = verifier or EGraphFragmentEquivalenceVerifier()
    index = _observation_index(observation_sets)

    results: list[SRMethodResult] = []
    counters = {"planned": 0, "completed": 0, "resumed": 0, "failed": 0}

    for task in tasks:
        fit = index.get((task.task_id, ObservationRole.FIT))
        evaluation = index.get((task.task_id, ObservationRole.EVALUATION))
        if fit is None:
            raise Goal9RunError(f"task {task.task_id} has no fit observation set")
        for method, representation in CONTROLLED_ARMS:
            for seed in config.seeds:
                counters["planned"] += 1
                target = Path(output_root) / method.value / f"{task.task_id}.seed{seed}.json"
                if config.resume and target.exists():
                    reloaded = SRMethodResult.model_validate_json(
                        target.read_text(encoding="utf-8")
                    )
                    if not config.allow_fixture_scorers:
                        # Rows persisted before training_data_policy existed carry only
                        # the fixture id prefix; newer rows also persist the policy, so
                        # both signals are checked.
                        fixture_ids = sorted(
                            {
                                scorer_id
                                for scorer_id in (
                                    reloaded.scorer_id,
                                    *(row.proposal_scorer_id for row in reloaded.candidates),
                                )
                                if scorer_id.startswith(FIXTURE_SCORER_ID_PREFIX)
                            }
                        )
                        if fixture_ids:
                            raise UntrainedScorerError(
                                f"resumed cell {target} carries fixture scorer ids "
                                f"{fixture_ids}; a sweep without allow_fixture_scorers "
                                "refuses to reload fixture-scored evidence"
                            )
                        if reloaded.training_data_policy == "not_trained" or any(
                            row.training_data_policy == "not_trained" for row in reloaded.candidates
                        ):
                            raise UntrainedScorerError(
                                f"resumed cell {target} persists training_data_policy="
                                "'not_trained'; a sweep without allow_fixture_scorers "
                                "refuses to reload untrained-scorer evidence"
                            )
                    counters["resumed"] += 1
                    results.append(reloaded)
                    continue
                result = run_guided_search(
                    task=task,
                    fit_observations=fit,
                    evaluation_observations=evaluation,
                    method=method,
                    representation=representation,
                    scorer=active_scorers[representation],
                    budget=config.budget,
                    budget_digest=digest,
                    seed=seed,
                    verifier=active_verifier,
                )
                write_result(result, output_root)
                results.append(result)
                counters["completed"] += 1
                if result.status is not RunStatus.COMPLETE:
                    counters["failed"] += 1
    return tuple(results), counters


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI shell
    """Run the guided-search sweep described by ``--config`` over a frozen benchmark."""

    parser = argparse.ArgumentParser(description="Run Goal 9 guided SR search")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args(argv)

    config, config_hash = load_config(arguments.config)
    benchmark_dir = Path(arguments.benchmark_dir)
    manifest = load_manifest(benchmark_dir / "benchmark.manifest.json")
    tasks = load_tasks(benchmark_dir / "tasks.jsonl")
    observation_sets = load_observation_sets(benchmark_dir / "observations.jsonl")
    selected = select_tasks(tasks, manifest, config)

    started = time.time()
    results, counters = run_sweep(
        tasks=selected,
        observation_sets=observation_sets,
        config=config,
        output_root=arguments.output_dir,
    )
    finished = time.time()

    envelope = RunEnvelope(
        config_hash=config_hash,
        config_path=str(arguments.config),
        benchmark_id=manifest.benchmark_id,
        benchmark_status=manifest.status.value,
        budget_digest=config.budget.digest(),
        seeds=config.seeds,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        package_versions=_package_versions(),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
        wall_seconds=finished - started,
        cells_planned=counters["planned"],
        cells_completed=counters["completed"],
        cells_resumed=counters["resumed"],
        cells_failed=counters["failed"],
        reproduction_command=(
            "python -m geml.experiments.goal9.run_sr "
            f"--config {arguments.config} --benchmark-dir {arguments.benchmark_dir} "
            f"--output-dir {arguments.output_dir}"
        ),
    )
    envelope_path = Path(arguments.output_dir) / "run.envelope.json"
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    envelope_path.write_text(
        json.dumps(envelope.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        f"cells planned={counters['planned']} completed={counters['completed']} "
        f"resumed={counters['resumed']} failed={counters['failed']} "
        f"rows={len(results)}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module executable entry point
    raise SystemExit(main())
