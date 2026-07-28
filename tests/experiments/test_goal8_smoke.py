"""Phase-A end-to-end tests for the fixed-budget Goal 8 ATP runner."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import geml.experiments.goal8.run_atp as atp_runner
from geml.experiments.goal8.run_atp import (
    ATP_CONFIG_SCHEMA,
    ATPCellStatus,
    ATPConfig,
    ATPExecutionAttestation,
    ATPMethod,
    ATPMethodConfig,
    ATPProblem,
    ATPProtocolError,
    ATPRuntimeIdentity,
    ReplayEvidence,
    SearchExecution,
    load_atp_config,
    run_atp_shard,
)

_DIGESTS = {letter: letter * 64 for letter in "abcdef"}


def test_runtime_processor_identity_has_truthful_architecture_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atp_runner.platform, "machine", lambda: "test-arch")
    monkeypatch.setattr(atp_runner.platform, "processor", lambda: "")
    monkeypatch.setattr(
        atp_runner.platform,
        "uname",
        lambda: SimpleNamespace(machine="test-arch", processor=""),
    )
    monkeypatch.delenv("PROCESSOR_IDENTIFIER", raising=False)
    monkeypatch.setattr(atp_runner.Path, "read_text", lambda *_args, **_kwargs: "")

    assert atp_runner._hardware_identity() == (
        "test-arch",
        "architecture:test-arch",
    )


def test_production_runtime_provenance_is_strict_and_checked_before_output(
    tmp_path: Path,
) -> None:
    valid = ATPRuntimeIdentity(
        git_commit="ab" * 20,
        python_version="3.12.4",
        platform="Linux-6.8-x86_64",
        machine="x86_64",
        processor="AMD EPYC",
        package_versions={
            "geml": "0.1.0",
            "pydantic": "2.11.0",
            "pyyaml": "6.0.2",
        },
    )
    valid.require_production_ready()
    for invalid_commit in ("fixture", "a" * 40, "AB" * 20):
        with pytest.raises(ATPProtocolError, match="git_commit"):
            replace(valid, git_commit=invalid_commit).require_production_ready()
    with pytest.raises(ATPProtocolError, match="package_versions"):
        replace(
            valid,
            package_versions={"geml": "0.1.0", "pydantic": "unknown"},
        ).require_production_ready()

    output_root = tmp_path / "must-not-exist"
    production = _config(tmp_path).model_copy(
        update={"stage": "production", "output_root": str(output_root)}
    )
    with pytest.raises(ATPProtocolError, match="git_commit"):
        run_atp_shard(
            config=production,
            problems=_problems(),
            shard_index=0,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=_runtime(),
        )
    assert not output_root.exists()


def _config(
    root: Path,
    *,
    expected_problem_count: int = 2,
    shard_count: int = 1,
    missing_policy: bool = False,
) -> ATPConfig:
    benchmark_manifest = root / "fixture-benchmark.json"
    benchmark_payload = {
        "schema_version": "geml-goal8-atp-problem-projection-v1",
        "problems": [problem.identity_payload() for problem in _problems(expected_problem_count)],
    }
    benchmark_bytes = json.dumps(
        benchmark_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    benchmark_manifest.write_bytes(benchmark_bytes)
    return ATPConfig.model_validate(
        {
            "schema_version": ATP_CONFIG_SCHEMA,
            "stage": "fixture",
            "output_root": str(root),
            "benchmark_manifest": str(benchmark_manifest),
            "benchmark_manifest_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
            "expected_problem_count": expected_problem_count,
            "seeds": (20260726, 20260727, 20260728),
            "methods": (
                {
                    "method": ATPMethod.UNIFORM,
                    "stochastic": True,
                    "checkpoint_selection_split": "not_applicable",
                },
                {
                    "method": ATPMethod.POLICY,
                    "stochastic": False,
                    "checkpoint_selection_split": "validation",
                    "policy_checkpoint_sha256": (None if missing_policy else _DIGESTS["b"]),
                },
                {
                    "method": ATPMethod.POLICY_VALUE,
                    "stochastic": False,
                    "checkpoint_selection_split": "validation",
                    "policy_checkpoint_sha256": _DIGESTS["b"],
                    "value_checkpoint_sha256": _DIGESTS["c"],
                },
                {
                    "method": ATPMethod.TRANSFORMER,
                    "stochastic": False,
                    "checkpoint_selection_split": "validation",
                    "transformer_checkpoint_sha256": _DIGESTS["d"],
                },
            ),
            "budget": {
                "beam_width": 2,
                "expanded_node_budget": 11,
                "generated_state_budget": 17,
                "proof_depth_limit": 3,
                "wall_time_seconds": 1.0,
                "verifier_call_budget": 19,
            },
            "shard_count": shard_count,
            "rule_set_sha256": _DIGESTS["e"],
            "verifier_sha256": _DIGESTS["f"],
            "implementation_sha256": _DIGESTS["a"],
            "reproduction_command": "fixture-atp --shard-index {shard_index}",
        }
    )


def _problems(count: int = 2) -> tuple[ATPProblem, ...]:
    return tuple(
        ATPProblem(
            problem_id=f"problem-{index}",
            source_signature=f"source-{index}",
            goal_signature=f"goal-{index}",
            group_id=f"group-{index}",
            difficulty_tier="easy" if index % 2 == 0 else "hard",
            witness_length_tier="short" if index % 2 == 0 else "long",
            rule_diversity_tier="single" if index % 2 == 0 else "high",
            ood_tier=("length_family_in_distribution" if index % 2 == 0 else "length_ood"),
            length_ood=index % 2 == 1,
            family="algebraic_core",
            domain_mode="safe_real",
            assumptions=("real(x)",),
        )
        for index in range(count)
    )


def _runtime() -> ATPRuntimeIdentity:
    return ATPRuntimeIdentity(
        git_commit="fixture",
        python_version="3.12.fixture",
        platform="fixture",
        machine="fixture",
        processor="fixture",
        package_versions={"geml": "fixture"},
    )


def _attestation(
    method: ATPMethod,
    budget,
) -> ATPExecutionAttestation:
    payloads = {
        ATPMethod.UNIFORM: {
            "method": method,
            "stochastic": True,
            "checkpoint_selection_split": "not_applicable",
        },
        ATPMethod.POLICY: {
            "method": method,
            "stochastic": False,
            "checkpoint_selection_split": "validation",
            "policy_checkpoint_sha256": _DIGESTS["b"],
        },
        ATPMethod.POLICY_VALUE: {
            "method": method,
            "stochastic": False,
            "checkpoint_selection_split": "validation",
            "policy_checkpoint_sha256": _DIGESTS["b"],
            "value_checkpoint_sha256": _DIGESTS["c"],
        },
        ATPMethod.TRANSFORMER: {
            "method": method,
            "stochastic": False,
            "checkpoint_selection_split": "validation",
            "transformer_checkpoint_sha256": _DIGESTS["d"],
        },
    }
    method_config = ATPMethodConfig.model_validate(payloads[method])
    return ATPExecutionAttestation(
        method=method,
        checkpoint_digest=method_config.checkpoint_digest,
        rule_set_sha256=_DIGESTS["e"],
        verifier_sha256=_DIGESTS["f"],
        implementation_sha256=_DIGESTS["a"],
        budget_digest=budget.digest,
    )


class _SuccessfulExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ATPMethod, int, int, bool]] = []

    def __call__(
        self,
        problem,
        method,
        seed,
        budget,
        *,
        checkpoint_path,
        resume,
    ):
        self.calls.append((problem.problem_id, method, seed, id(budget), resume))
        return SearchExecution(
            status="success",
            termination_reason="exact_target_reached",
            attestation=_attestation(method, budget),
            claimed_success=True,
            exact_target_reached=True,
            terminal_signature=problem.goal_signature,
            proof_trace=(
                {
                    "source_signature": problem.source_signature,
                    "successor_signature": problem.goal_signature,
                    "verified": True,
                },
            ),
            expanded_count=1,
            generated_count=2,
            valid_count=1,
            invalid_count=1,
            duplicate_count=0,
            verifier_call_count=2,
            verifier_error_count=0,
            verifier_timeout_count=0,
            frontier_peak=2,
            search_depth_reached=1,
            proof_length=1,
            wall_time_seconds=0.01,
            peak_host_memory_bytes=1024,
        )


class _ExactReplayer:
    def __init__(self, *, reject_problem: str | None = None) -> None:
        self.reject_problem = reject_problem
        self.calls: list[str] = []

    def __call__(self, problem, proof_trace):
        self.calls.append(problem.problem_id)
        accepted = problem.problem_id != self.reject_problem
        return ReplayEvidence(
            transition_count=len(proof_trace),
            all_transitions_verified=accepted,
            terminal_signature=problem.goal_signature if accepted else "wrong-target",
            terminal_verified=accepted,
            status="verified" if accepted else "rejected",
            rule_set_sha256=_DIGESTS["e"] if accepted else None,
            verifier_sha256=_DIGESTS["f"] if accepted else None,
            error_type=None if accepted else "ReplayRejected",
            error_message=None if accepted else "fixture replay rejected",
        )


def _cell_rows(root: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("cells/**/*.json"))
    ]


def test_repository_config_loads_as_production_pending() -> None:
    config = load_atp_config(Path("configs/goal8_atp.yaml"))
    assert config.stage == "production"
    assert config.expected_problem_count == 256
    with pytest.raises(ATPProtocolError, match="not runnable"):
        config.require_runnable()


def test_config_freezes_exact_four_methods_and_validation_only_checkpoints(
    tmp_path: Path,
) -> None:
    payload = _config(tmp_path).model_dump(mode="python")
    payload["methods"] = payload["methods"][:-1]
    with pytest.raises(ValidationError, match="at least 4"):
        ATPConfig.model_validate(payload)

    payload = _config(tmp_path).model_dump(mode="python")
    payload["methods"][1]["checkpoint_selection_split"] = "test_iid"
    with pytest.raises(ValidationError, match="must be 'validation'"):
        ATPConfig.model_validate(payload)

    payload = _config(tmp_path).model_dump(mode="python")
    payload["methods"][0]["value_checkpoint_sha256"] = _DIGESTS["c"]
    with pytest.raises(ValidationError, match="disallowed checkpoints"):
        ATPConfig.model_validate(payload)

    payload = _config(tmp_path).model_dump(mode="python")
    payload["reproduction_command"] = "fixture-atp --shard-index {other}"
    with pytest.raises(ValidationError, match="unsupported template token"):
        ATPConfig.model_validate(payload)


def test_equal_budget_four_method_run_has_explicit_denominators_and_replay(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    executor = _SuccessfulExecutor()
    replayer = _ExactReplayer()

    receipt = run_atp_shard(
        config=config,
        problems=_problems(),
        shard_index=0,
        executor=executor,
        replayer=replayer,
        runtime=_runtime(),
    )

    assert receipt.expected_count == receipt.attempted_count == 12
    assert receipt.status_counts == {ATPCellStatus.SUCCESS.value: 12}
    assert len(executor.calls) == 12
    assert {call[1] for call in executor.calls} == set(ATPMethod)
    assert {call[2] for call in executor.calls if call[1] is ATPMethod.UNIFORM} == set(config.seeds)
    assert {call[2] for call in executor.calls if call[1] is not ATPMethod.UNIFORM} == {
        config.canonical_seed
    }
    assert len({call[3] for call in executor.calls}) == 1
    assert len(replayer.calls) == 12

    rows = _cell_rows(tmp_path)
    assert len(rows) == 12
    assert {row["budget_digest"] for row in rows} == {config.budget.digest}
    assert all(
        set(row["checkpoint_identities"])
        == {
            "policy_checkpoint_sha256",
            "value_checkpoint_sha256",
            "transformer_checkpoint_sha256",
        }
        for row in rows
    )
    assert {row["problem"]["difficulty_tier"] for row in rows} == {"easy", "hard"}
    assert {row["problem"]["witness_length_tier"] for row in rows} == {"short", "long"}
    assert {row["problem"]["rule_diversity_tier"] for row in rows} == {"single", "high"}
    assert {row["problem"]["ood_tier"] for row in rows} == {
        "length_family_in_distribution",
        "length_ood",
    }
    assert {row["problem"]["length_ood"] for row in rows} == {False, True}
    assert all(row["verified_success"] is True for row in rows)
    assert Counter(row["seed_policy"] for row in rows) == {
        "canonical_seed_deterministic": 6,
        "three_seed_stochastic": 6,
    }
    assert {row["reproduction_command"] for row in rows} == {"fixture-atp --shard-index 0"}
    completion = json.loads(receipt.completion_path.read_text(encoding="utf-8"))
    assert completion["attempted_count"] == completion["expected_count"] == 12
    assert len(completion["cell_content_digests"]) == 12
    assert completion["reproduction_command"] == "fixture-atp --shard-index 0"


def test_runner_measures_search_wall_budget_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, expected_problem_count=1)
    method = config.methods[0]
    clock = iter((0.0, 0.0, 2.0, 2.0))
    monkeypatch.setattr(atp_runner.time, "perf_counter", lambda: next(clock))

    row = atp_runner._execute_cell(
        config=config,
        run_id="fixture-run",
        benchmark_projection_digest="a" * 64,
        problem=_problems(1)[0],
        method_config=method,
        seed=config.seeds[0],
        cell_id="b" * 64,
        executor=_SuccessfulExecutor(),
        replayer=_ExactReplayer(),
        runtime=_runtime(),
        search_checkpoint=tmp_path / "search.json",
        shard_index=0,
    )

    assert row["status"] == ATPCellStatus.WALL_TIMEOUT
    assert row["search_wall_time_seconds"] == 0.01
    assert row["measured_search_wall_time_seconds"] == 2.0
    assert row["runner_wall_time_seconds"] == 2.0
    assert row["error_type"] == "MeasuredSearchWallBudgetExceeded"


def test_search_execution_rejects_non_json_or_spoofed_telemetry(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_problem_count=1)
    execution = _SuccessfulExecutor()(
        _problems(1)[0],
        ATPMethod.UNIFORM,
        config.seeds[0],
        config.budget,
        checkpoint_path=tmp_path / "search.json",
        resume=False,
    )

    with pytest.raises(ValueError, match="non-finite"):
        replace(execution, proof_trace=({"nested": [float("nan")]},))
    with pytest.raises(ValueError, match="reserved metrics"):
        replace(execution, extra_telemetry={"peak_host_memory_bytes": 1})
    with pytest.raises(ValueError, match="non-string mapping key"):
        replace(execution, extra_telemetry={1: "spoof"})
    with pytest.raises(ValueError, match="nonnegative exact integer"):
        replace(execution, peak_gpu_memory_bytes=-1)


def test_claimed_success_is_invalid_until_exact_trace_replays(tmp_path: Path) -> None:
    receipt = run_atp_shard(
        config=_config(tmp_path),
        problems=_problems(),
        shard_index=0,
        executor=_SuccessfulExecutor(),
        replayer=_ExactReplayer(reject_problem="problem-1"),
        runtime=_runtime(),
    )

    assert receipt.status_counts == {
        ATPCellStatus.REPLAY_FAILED.value: 6,
        ATPCellStatus.SUCCESS.value: 6,
    }
    rejected = [row for row in _cell_rows(tmp_path) if row["problem"]["problem_id"] == "problem-1"]
    assert all(row["claimed_success"] is True for row in rejected)
    assert all(row["verified_success"] is False for row in rejected)
    assert all(row["status"] == ATPCellStatus.REPLAY_FAILED for row in rejected)
    assert all(row["error_type"] == "ReplayRejected" for row in rejected)


def test_contradictory_replay_status_cannot_count_as_success(tmp_path: Path) -> None:
    class ContradictoryReplayer:
        def __call__(self, problem, proof_trace):
            del problem
            return ReplayEvidence(
                transition_count=len(proof_trace),
                all_transitions_verified=True,
                terminal_signature="wrong",
                terminal_verified=True,
                status="invalid",
                rule_set_sha256=None,
                verifier_sha256=None,
            )

    receipt = run_atp_shard(
        config=_config(tmp_path),
        problems=_problems(),
        shard_index=0,
        executor=_SuccessfulExecutor(),
        replayer=ContradictoryReplayer(),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {ATPCellStatus.REPLAY_FAILED.value: 12}
    assert all(row["verified_success"] is False for row in _cell_rows(tmp_path))


@pytest.mark.parametrize("fault", ["budget", "attestation"])
def test_search_budget_and_component_spoofs_are_retained_as_invalid(
    tmp_path: Path,
    fault: str,
) -> None:
    class SpoofingExecutor(_SuccessfulExecutor):
        def __call__(self, problem, method, seed, budget, **kwargs):
            execution = super().__call__(
                problem,
                method,
                seed,
                budget,
                **kwargs,
            )
            if fault == "budget":
                return replace(
                    execution,
                    expanded_count=budget.expanded_node_budget + 1,
                )
            return replace(
                execution,
                attestation=replace(
                    execution.attestation,
                    implementation_sha256=_DIGESTS["d"],
                ),
            )

    replayer = _ExactReplayer()
    receipt = run_atp_shard(
        config=_config(tmp_path),
        problems=_problems(),
        shard_index=0,
        executor=SpoofingExecutor(),
        replayer=replayer,
        runtime=_runtime(),
    )
    assert receipt.status_counts == {ATPCellStatus.INVALID.value: 12}
    assert replayer.calls == []
    assert all(row["verified_success"] is False for row in _cell_rows(tmp_path))


def test_replay_component_identity_mismatch_cannot_count(tmp_path: Path) -> None:
    class WrongIdentityReplayer:
        def __call__(self, problem, proof_trace):
            return ReplayEvidence(
                transition_count=len(proof_trace),
                all_transitions_verified=True,
                terminal_signature=problem.goal_signature,
                terminal_verified=True,
                status="verified",
                rule_set_sha256=_DIGESTS["d"],
                verifier_sha256=_DIGESTS["f"],
            )

    receipt = run_atp_shard(
        config=_config(tmp_path),
        problems=_problems(),
        shard_index=0,
        executor=_SuccessfulExecutor(),
        replayer=WrongIdentityReplayer(),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {ATPCellStatus.REPLAY_FAILED.value: 12}


def test_missing_checkpoint_fails_before_output_or_search(tmp_path: Path) -> None:
    executor = _SuccessfulExecutor()
    with pytest.raises(ATPProtocolError, match="policy_checkpoint_sha256"):
        run_atp_shard(
            config=_config(tmp_path, missing_policy=True),
            problems=_problems(),
            shard_index=0,
            executor=executor,
            replayer=_ExactReplayer(),
            runtime=_runtime(),
        )
    assert executor.calls == []
    assert not tuple(tmp_path.rglob("cells/**/*.json"))


def test_interrupted_runner_resumes_without_reexecuting_committed_cells(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    executor = _SuccessfulExecutor()

    def interrupt(committed: int) -> None:
        if committed == 5:
            raise RuntimeError("fixture interruption")

    with pytest.raises(RuntimeError, match="fixture interruption"):
        run_atp_shard(
            config=config,
            problems=_problems(),
            shard_index=0,
            executor=executor,
            replayer=_ExactReplayer(),
            runtime=_runtime(),
            on_cell_committed=interrupt,
        )
    assert len(executor.calls) == 5

    receipt = run_atp_shard(
        config=config,
        problems=_problems(),
        shard_index=0,
        executor=executor,
        replayer=_ExactReplayer(),
        runtime=_runtime(),
    )
    assert receipt.attempted_count == 12
    assert len(executor.calls) == 12
    assert len(_cell_rows(tmp_path)) == 12

    def unexpected_search(*_args, **_kwargs):
        raise AssertionError("completed shards must not re-run search")

    resumed = run_atp_shard(
        config=config,
        problems=_problems(),
        shard_index=0,
        executor=unexpected_search,
        replayer=_ExactReplayer(),
        runtime=_runtime(),
    )
    assert resumed == receipt


def test_shards_are_disjoint_and_cover_every_problem_method_seed_cell(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, shard_count=5)
    cell_sets: list[set[str]] = []
    for shard_index in range(config.shard_count):
        receipt = run_atp_shard(
            config=config,
            problems=_problems(),
            shard_index=shard_index,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=_runtime(),
        )
        completion = json.loads(receipt.completion_path.read_text(encoding="utf-8"))
        cell_sets.append(set(completion["expected_cell_ids"]))

    assert sum(len(ids) for ids in cell_sets) == 12
    assert len(set().union(*cell_sets)) == 12
    for index, left in enumerate(cell_sets):
        for right in cell_sets[index + 1 :]:
            assert left.isdisjoint(right)


class _FailingExecutor:
    def __call__(
        self,
        problem,
        method,
        seed,
        budget,
        *,
        checkpoint_path,
        resume,
    ):
        del problem, seed, checkpoint_path, resume
        if method is ATPMethod.UNIFORM:
            raise TimeoutError("retained timeout")
        if method is ATPMethod.POLICY:
            raise ValueError("retained search failure")
        return SearchExecution(
            status="unsupported" if method is ATPMethod.TRANSFORMER else "exhausted",
            termination_reason="fixture terminal",
            attestation=_attestation(method, budget),
            claimed_success=False,
            exact_target_reached=False,
            terminal_signature=None,
            proof_trace=(),
            expanded_count=2,
            generated_count=3,
            valid_count=1,
            invalid_count=2,
            duplicate_count=0,
            verifier_call_count=3,
            verifier_error_count=0,
            verifier_timeout_count=0,
            frontier_peak=2,
            search_depth_reached=1,
            proof_length=None,
            wall_time_seconds=0.01,
        )


def test_every_timeout_error_unsupported_and_exhausted_cell_is_retained(
    tmp_path: Path,
) -> None:
    receipt = run_atp_shard(
        config=_config(tmp_path),
        problems=_problems(),
        shard_index=0,
        executor=_FailingExecutor(),
        replayer=_ExactReplayer(),
        runtime=_runtime(),
    )
    assert receipt.attempted_count == 12
    assert receipt.status_counts == {
        ATPCellStatus.EXHAUSTED.value: 2,
        ATPCellStatus.SEARCH_ERROR.value: 2,
        ATPCellStatus.WALL_TIMEOUT.value: 6,
        ATPCellStatus.UNSUPPORTED.value: 2,
    }
    rows = _cell_rows(tmp_path)
    assert Counter(row["status"] for row in rows) == receipt.status_counts
    assert all(row["verified_success"] is False for row in rows)


def test_verifier_timeout_is_distinct_from_wall_timeout(tmp_path: Path) -> None:
    class VerifierTimeoutExecutor:
        def __call__(
            self,
            problem,
            method,
            seed,
            budget,
            *,
            checkpoint_path,
            resume,
        ):
            del problem, seed, checkpoint_path, resume
            return SearchExecution(
                status="verifier_timeout",
                termination_reason="verifier_call_timeout",
                attestation=_attestation(method, budget),
                claimed_success=False,
                exact_target_reached=False,
                terminal_signature=None,
                proof_trace=(),
                expanded_count=1,
                generated_count=1,
                valid_count=0,
                invalid_count=1,
                duplicate_count=0,
                verifier_call_count=1,
                verifier_error_count=0,
                verifier_timeout_count=1,
                frontier_peak=1,
                search_depth_reached=1,
                proof_length=None,
                wall_time_seconds=0.01,
            )

    receipt = run_atp_shard(
        config=_config(tmp_path),
        problems=_problems(),
        shard_index=0,
        executor=VerifierTimeoutExecutor(),
        replayer=_ExactReplayer(),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {ATPCellStatus.VERIFIER_TIMEOUT.value: 12}
    completion = json.loads(receipt.completion_path.read_text(encoding="utf-8"))
    assert completion["verifier_timeout_count"] == 12
    assert completion["wall_timeout_count"] == 0


def test_corrupt_retained_cell_is_never_silently_replaced(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_atp_shard(
        config=config,
        problems=_problems(),
        shard_index=0,
        executor=_SuccessfulExecutor(),
        replayer=_ExactReplayer(),
        runtime=_runtime(),
    )
    cell = next(tmp_path.rglob("cells/**/*.json"))
    payload = json.loads(cell.read_text(encoding="utf-8"))
    payload["budget_digest"] = "0" * 64
    cell.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ATPProtocolError, match=r"identity mismatch|digest mismatch"):
        run_atp_shard(
            config=config,
            problems=_problems(),
            shard_index=0,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=_runtime(),
        )


def test_completed_shard_rejects_a_different_supplied_runtime(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_atp_shard(
        config=config,
        problems=_problems(),
        shard_index=0,
        executor=_SuccessfulExecutor(),
        replayer=_ExactReplayer(),
        runtime=_runtime(),
    )

    with pytest.raises(ATPProtocolError, match=r"runtime|identity mismatch"):
        run_atp_shard(
            config=config,
            problems=_problems(),
            shard_index=0,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=replace(_runtime(), processor="different-runtime"),
        )


def test_resigned_invalid_resource_telemetry_is_rejected_on_resume(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_atp_shard(
        config=config,
        problems=_problems(),
        shard_index=0,
        executor=_SuccessfulExecutor(),
        replayer=_ExactReplayer(),
        runtime=_runtime(),
    )
    cell = next(tmp_path.rglob("cells/**/*.json"))
    payload = json.loads(cell.read_text(encoding="utf-8"))
    payload["resource_telemetry"]["peak_host_memory_bytes"] = -1
    content = {key: value for key, value in payload.items() if key != "content_digest"}
    payload["content_digest"] = atp_runner._payload_digest(content)
    cell.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ATPProtocolError, match="typed evidence is invalid"):
        run_atp_shard(
            config=config,
            problems=_problems(),
            shard_index=0,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=_runtime(),
        )


def test_completion_derived_counts_are_recomputed_from_cells(tmp_path: Path) -> None:
    config = _config(tmp_path)
    receipt = run_atp_shard(
        config=config,
        problems=_problems(),
        shard_index=0,
        executor=_SuccessfulExecutor(),
        replayer=_ExactReplayer(),
        runtime=_runtime(),
    )
    payload = json.loads(receipt.completion_path.read_text(encoding="utf-8"))
    payload["status_counts"] = {"invalid": 12}
    payload["success_count"] = 0
    payload["invalid_count"] = 12
    receipt.completion_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ATPProtocolError, match="derived fields"):
        run_atp_shard(
            config=config,
            problems=_problems(),
            shard_index=0,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=_runtime(),
        )


def test_checkpoint_replace_retries_only_bounded_windows_sharing_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_replace = atp_runner.atomic_replace_json
    calls = 0

    def sharing_violation() -> PermissionError:
        error = PermissionError("fixture sharing violation")
        error.winerror = 32
        return error

    def flaky(path, payload):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise sharing_violation()
        return real_replace(path, payload)

    monkeypatch.setattr(atp_runner.sys, "platform", "win32")
    monkeypatch.setattr(atp_runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(atp_runner, "atomic_replace_json", flaky)
    checkpoint = tmp_path / "checkpoint.json"
    atp_runner._replace_checkpoint_json(checkpoint, {"attempt": "fixture"})
    assert calls == 3
    assert checkpoint.is_file()

    calls = 0
    monkeypatch.setattr(
        atp_runner,
        "atomic_replace_json",
        lambda _path, _payload: (_ for _ in ()).throw(sharing_violation()),
    )
    with pytest.raises(ATPProtocolError, match="remained blocked after 4 attempts"):
        atp_runner._replace_checkpoint_json(checkpoint, {"attempt": "persistent"})


def test_interrupted_resume_rejects_nonidentity_cell_tampering(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def interrupt(committed: int) -> None:
        if committed == 1:
            raise RuntimeError("fixture interruption")

    with pytest.raises(RuntimeError, match="fixture interruption"):
        run_atp_shard(
            config=config,
            problems=_problems(),
            shard_index=0,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=_runtime(),
            on_cell_committed=interrupt,
        )
    cell = next(tmp_path.rglob("cells/**/*.json"))
    payload = json.loads(cell.read_text(encoding="utf-8"))
    payload["counts"]["expanded"] = 999
    cell.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ATPProtocolError, match="content digest mismatch"):
        run_atp_shard(
            config=config,
            problems=_problems(),
            shard_index=0,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=_runtime(),
        )


def test_manifest_bytes_and_problem_projection_are_authenticated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = Path(config.benchmark_manifest)
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ATPProtocolError, match="checksum mismatch"):
        run_atp_shard(
            config=config,
            problems=_problems(),
            shard_index=0,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=_runtime(),
        )


def test_production_manifest_is_rechecked_after_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geml.data.proofs import benchmark as benchmark_module

    manifest = tmp_path / "benchmark.json"
    authenticated = b'{"authenticated":true}'
    manifest.write_bytes(authenticated)

    def mutate_during_load(path: Path) -> object:
        path.write_bytes(b'{"changed":true}')
        return object()

    monkeypatch.setattr(
        benchmark_module,
        "load_benchmark_manifest",
        mutate_during_load,
    )
    with pytest.raises(ATPProtocolError, match="changed while"):
        atp_runner._load_authenticated_benchmark_manifest(manifest, authenticated)

    config = _config(tmp_path)
    different = list(_problems())
    different[0] = replace(different[0], source_signature="different-source")
    with pytest.raises(ATPProtocolError, match="does not match"):
        run_atp_shard(
            config=config,
            problems=tuple(different),
            shard_index=0,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=_runtime(),
        )


def test_population_and_shard_contracts_fail_loudly(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ATPProtocolError, match="expected 2"):
        run_atp_shard(
            config=config,
            problems=_problems(1),
            shard_index=0,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=_runtime(),
        )
    with pytest.raises(ATPProtocolError, match="shard_index"):
        run_atp_shard(
            config=config,
            problems=_problems(),
            shard_index=1,
            executor=_SuccessfulExecutor(),
            replayer=_ExactReplayer(),
            runtime=_runtime(),
        )
