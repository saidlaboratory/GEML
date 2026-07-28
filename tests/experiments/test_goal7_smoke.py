from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import geml.experiments.goal7.run_grid as grid_module
from geml.analysis.goal7.summary import (
    GatePolicyV1,
    GateVerdict,
    build_goal7_summary,
    write_goal7_summary,
)
from geml.experiments.goal7.run_grid import (
    CONFIG_SCHEMA_VERSION,
    PRODUCTION_SEEDS,
    STEP_MANIFEST_AUTH_SCHEMA_VERSION,
    UNIFORM_DRAW_AUDIT_SCHEMA_VERSION,
    BudgetConsumptionV1,
    BudgetStopReason,
    Goal7GridConfig,
    Goal7ProtocolError,
    GraphChannelSpec,
    GridBudgetV1,
    GridCellExecution,
    GridCellStatus,
    GridStage,
    UniformDrawAuditV1,
    compute_step_population_digest,
    current_fixture_run_envelope,
    enumerate_grid_cells,
    fixture_run_envelope_adapter,
    load_goal7_grid_config,
    load_goal7_run_evidence,
    run_goal7_grid,
    uniform_valid_order,
)
from geml.learning.eval.step_metrics import (
    FAMILY_PARTITION_EVIDENCE_SCHEMA_VERSION,
    STEP_METRIC_SCHEMA_VERSION,
    ActionIdentityV1,
    CandidateMetricOutcomeV1,
    CandidateMetricStatus,
    ExampleMetricStatus,
    FamilyGeneralization,
    FamilyPartitionEvidenceV1,
    LegalityStatus,
    ReplayStatus,
    StepMetricOutcomeV1,
    VerificationStatus,
)
from geml.learning.policy.head import (
    ActionInventoryStatus,
    compute_legal_mask_digest,
)
from geml.plots.goal7 import build_plot_data, render_goal7_plots

_RULES = ("rule.a", "rule.b", "rule.c")
_RULE_DIRECTIONS = tuple((rule_id, "forward") for rule_id in _RULES)
_FAMILY_EVIDENCE = FamilyPartitionEvidenceV1(
    schema_version=FAMILY_PARTITION_EVIDENCE_SCHEMA_VERSION,
    step_manifest_digest=hashlib.sha256(b"steps").hexdigest(),
    training_family_ids=("algebra",),
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _rehash_payload(payload: dict[str, object], domain: bytes) -> None:
    content = {key: value for key, value in payload.items() if key != "content_digest"}
    payload["content_digest"] = hashlib.sha256(
        domain + grid_module._canonical_json(content)
    ).hexdigest()


def _config(root: Path) -> Goal7GridConfig:
    return Goal7GridConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        stage=GridStage.FIXTURE,
        output_root=str(root),
        seeds=PRODUCTION_SEEDS,
        expected_step_count=25,
        step_manifest="fixture-steps.json",
        step_manifest_sha256=_digest("steps"),
        rule_registry_sha256=_digest("rules"),
        verifier_sha256=_digest("verifier"),
        shared_harness_sha256=_digest("harness"),
        shared_gnn_architecture_sha256=_digest("gnn"),
        transformer_architecture_sha256=_digest("transformer"),
        compute_reference_sha256=_digest("reference workload"),
        implementation_sha256=_digest("implementation"),
        training_config_sha256=_digest("training config"),
        training_family_inventory_sha256=_FAMILY_EVIDENCE.inventory_digest,
        step_population_sha256=_fixture_step_population_digest(),
        analysis_reproduction_command=("python -m geml.analysis.goal7.summary --run-id {run_id}"),
        channel_contract_resolved=True,
        channels=(
            GraphChannelSpec("ast_dag", "source_ast_dag", True),
            GraphChannelSpec(
                "frequent_macro_motif_dag",
                "macro_dag_with_frequent_macro_motifs",
                True,
            ),
            GraphChannelSpec(
                "motif_ast_fair_control",
                "train_only_ast_motifs_equal_budget",
                True,
            ),
            GraphChannelSpec(
                "pure_eml_dag",
                "strict_pure_eml_official_v4_dag",
                True,
            ),
        ),
        budget=GridBudgetV1(
            maximum_epochs=2,
            early_stopping_patience=1,
            maximum_optimizer_steps=3,
            node_edge_batch_budget=64,
            wall_time_seconds=30.0,
            top_k=(1, 3, 5),
            parameter_match_tolerance_fraction=0.05,
            flop_match_tolerance_fraction=0.10,
            comparison_unit="fixture_optimizer_step",
        ),
        reproduction_command="pytest goal7 --cell-id {cell_id}",
    )


def _action(
    rule_id: str,
    *,
    label: str,
    path: tuple[int, ...],
    direction: str = "forward",
) -> ActionIdentityV1:
    return ActionIdentityV1(
        rule_id=rule_id,
        direction=direction,
        occurrence_path=path,
        ordered_arguments_json=("1",),
        action_digest=_digest(label),
    )


def _inventory_actions(index: int) -> tuple[ActionIdentityV1, ...]:
    rule_id = _RULES[index % 2]
    return (
        _action(
            rule_id,
            label=f"demonstration-{index}",
            path=(index % 3,),
        ),
        _action(
            rule_id,
            label=f"alternative-a-{index}",
            path=((index + 1) % 3,),
        ),
        _action(
            rule_id,
            label=f"alternative-b-{index}",
            path=((index + 2) % 3,),
        ),
    )


def _metric_row(
    index: int,
    *,
    arm_id: str,
    seed: int,
) -> StepMetricOutcomeV1:
    record_id = _digest(f"record-{index}")
    alternatives = _inventory_actions(index)
    demonstration = alternatives[0]
    primary = arm_id == "gnn:pure_eml_dag"
    if arm_id == "uniform_valid":
        ranking = uniform_valid_order(
            (0, 1, 2),
            seed=seed,
            record_id=record_id,
        )
    elif primary:
        ranking = (0, 1, 2)
    else:
        ranking = (1, 0, 2)
    target_signature = _digest(f"target-{index}")
    exact_successor_indices = (
        {0, 1}
        if arm_id == "uniform_valid" or primary or (arm_id.startswith("gnn:") and index % 2 == 0)
        else {0}
    )
    candidates = tuple(
        CandidateMetricOutcomeV1(
            rank=rank,
            action=alternatives[action_index],
            status=CandidateMetricStatus.VERIFIED_VALID,
            exact_demonstration_action=action_index == 0,
            exact_successor_structure=action_index in exact_successor_indices,
            verifier_confirmed_valid=True,
            successor_signature=(
                target_signature
                if action_index in exact_successor_indices
                else _digest(f"{arm_id}-{seed}-next-{index}-{action_index}")
            ),
            legality_status=LegalityStatus.LEGAL,
            replay_status=ReplayStatus.SUCCEEDED,
            verifier_status=VerificationStatus.ACCEPTED,
            legality_detail="fixture registry accepted the action",
            replay_detail="fixture replay produced the retained successor",
            verifier_detail="fixture verifier accepted the transition",
            parse_error=None,
        )
        for rank, action_index in enumerate(ranking, start=1)
    )
    held_out = index >= 20
    source_group = f"group-{index // 5:02d}"
    current_signature = _digest(f"current-{index}")
    goal_signature = _digest(f"goal-{index}")
    legal_mask = (True, True, True)
    return StepMetricOutcomeV1(
        schema_version=STEP_METRIC_SCHEMA_VERSION,
        record_id=record_id,
        trace_id=_digest(f"trace-{index // 5}"),
        source_group=source_group,
        lineage_group_ids=tuple(sorted((f"eclass-{index // 5:02d}", source_group))),
        authoritative_split="test_ood" if held_out else "test_iid",
        current_signature=current_signature,
        goal_signature=goal_signature,
        target_successor_signature=target_signature,
        current_family="algebra",
        goal_family="held_out_family" if held_out else "algebra",
        evaluation_views=("family_holdout",) if held_out else ("iid",),
        family_generalization=(
            FamilyGeneralization.HELD_OUT if held_out else FamilyGeneralization.SEEN
        ),
        family_evidence_manifest_digest=_FAMILY_EVIDENCE.step_manifest_digest,
        training_family_inventory_digest=_FAMILY_EVIDENCE.inventory_digest,
        unseen_family_roles=("goal",) if held_out else (),
        remaining_witness_steps=(index % 5) + 1,
        trace_length=5,
        demonstration_action=demonstration,
        registered_rule_ids=_RULES,
        registered_rule_directions=_RULE_DIRECTIONS,
        rule_registry_digest=_digest("rules"),
        legal_mask_digest=compute_legal_mask_digest(
            action_digests=tuple(action.action_digest for action in alternatives),
            legal_mask=legal_mask,
            current_signature=current_signature,
            goal_signature=goal_signature,
            registry_digest=_digest("rules"),
            status=ActionInventoryStatus.READY,
        ),
        requested_top_ks=(1, 3, 5),
        legal_action_count=3,
        proposal_candidate_count=3,
        status=ExampleMetricStatus.EVALUATED,
        detail="count-25 fixture metric row",
        candidates=candidates,
    )


def _uniform_audit(
    index: int,
    *,
    outcome: StepMetricOutcomeV1,
    seed: int,
) -> UniformDrawAuditV1:
    inventory = _inventory_actions(index)
    action_digests = tuple(action.action_digest for action in inventory)
    ranking = uniform_valid_order(
        tuple(range(len(inventory))),
        seed=seed,
        record_id=outcome.record_id,
    )
    return UniformDrawAuditV1(
        schema_version=UNIFORM_DRAW_AUDIT_SCHEMA_VERSION,
        record_id=outcome.record_id,
        inventory_status=ActionInventoryStatus.READY.value,
        inventory_action_digests=action_digests,
        legal_mask=tuple(True for _ in inventory),
        ranked_action_digests=tuple(action_digests[index] for index in ranking),
    )


def _fixture_step_population_digest(*, lineage_bridge: bool = False) -> str:
    rows = [
        _metric_row(
            index,
            arm_id="gnn:pure_eml_dag",
            seed=PRODUCTION_SEEDS[0],
        ).as_dict()
        for index in range(25)
    ]
    if lineage_bridge:
        bridge_records = {_digest("record-0"), _digest("record-5")}
        for row in rows:
            if row["record_id"] in bridge_records:
                row["lineage_group_ids"] = sorted(
                    {*row["lineage_group_ids"], "shared-lineage-bridge"}
                )
    return compute_step_population_digest(rows)


class _FixtureExecutor:
    def __init__(self, *, timeout_arm_seed: tuple[str, int] | None = None) -> None:
        self.timeout_arm_seed = timeout_arm_seed
        self.calls: list[str] = []

    def __call__(self, request) -> GridCellExecution:
        self.calls.append(request.cell_id)
        if (request.arm.arm_id, request.seed) == self.timeout_arm_seed:
            raise TimeoutError("fixture timeout retained by the runner")
        outcomes = tuple(
            _metric_row(index, arm_id=request.arm.arm_id, seed=request.seed) for index in range(25)
        )
        rows = tuple(outcome.as_dict() for outcome in outcomes)
        learned = request.arm.arm_id != "uniform_valid"
        uniform_audits = (
            ()
            if learned
            else tuple(
                _uniform_audit(index, outcome=outcome, seed=request.seed)
                for index, outcome in enumerate(outcomes)
            )
        )
        return GridCellExecution(
            status=GridCellStatus.COMPLETE,
            metric_rows=rows,
            checkpoint_sha256=_digest(request.cell_id) if learned else None,
            parameter_count=96_000 if learned else 0,
            estimated_flops=1_000_000.0 if learned else 0.0,
            wall_time_seconds=0.25,
            peak_host_memory_bytes=1024,
            peak_device_memory_bytes=2048 if learned else 0,
            uniform_draw_audits=uniform_audits,
            run_envelope=current_fixture_run_envelope(exact_command=request.reproduction_command),
            budget_consumption=BudgetConsumptionV1(
                epochs_completed=1 if learned else 0,
                optimizer_steps_completed=2 if learned else 0,
                maximum_observed_node_edge_batch=32 if learned else 0,
                early_stopping_bad_epochs=0,
                stop_reason=(
                    BudgetStopReason.COMPLETED if learned else BudgetStopReason.NOT_APPLICABLE
                ),
            ),
        )


class _MaskMismatchExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        if request.arm.arm_id != "transformer" or request.seed != PRODUCTION_SEEDS[0]:
            return execution
        rows = [dict(row) for row in execution.metric_rows]
        rows[0]["legal_mask_digest"] = _digest("different legal mask")
        return GridCellExecution(
            status=execution.status,
            metric_rows=tuple(rows),
            checkpoint_sha256=execution.checkpoint_sha256,
            parameter_count=execution.parameter_count,
            estimated_flops=execution.estimated_flops,
            wall_time_seconds=execution.wall_time_seconds,
            peak_host_memory_bytes=execution.peak_host_memory_bytes,
            peak_device_memory_bytes=execution.peak_device_memory_bytes,
            run_envelope=execution.run_envelope,
        )


class _MetadataMismatchExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        if request.arm.arm_id != "transformer" or request.seed != PRODUCTION_SEEDS[0]:
            return execution
        rows = [dict(row) for row in execution.metric_rows]
        rows[0]["authoritative_split"] = "test_ood"
        return GridCellExecution(
            status=execution.status,
            metric_rows=tuple(rows),
            checkpoint_sha256=execution.checkpoint_sha256,
            parameter_count=execution.parameter_count,
            estimated_flops=execution.estimated_flops,
            wall_time_seconds=execution.wall_time_seconds,
            peak_host_memory_bytes=execution.peak_host_memory_bytes,
            peak_device_memory_bytes=execution.peak_device_memory_bytes,
            run_envelope=execution.run_envelope,
        )


class _CheatingUniformExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        if request.arm.arm_id != "uniform_valid" or request.seed != PRODUCTION_SEEDS[0]:
            return execution
        audits = list(execution.uniform_draw_audits)
        first = audits[0]
        ranking = list(first.ranked_action_digests)
        ranking[0], ranking[1] = ranking[1], ranking[0]
        audits[0] = replace(first, ranked_action_digests=tuple(ranking))
        return replace(execution, uniform_draw_audits=tuple(audits))


class _LearnedAuditExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        if request.arm.arm_id != "transformer" or request.seed != PRODUCTION_SEEDS[0]:
            return execution
        outcome = StepMetricOutcomeV1.from_dict(dict(execution.metric_rows[0]))
        audit = _uniform_audit(0, outcome=outcome, seed=request.seed)
        return replace(execution, uniform_draw_audits=(audit,))


class _LineageBridgeExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        bridge_records = {_digest("record-0"), _digest("record-5")}
        rows = [dict(row) for row in execution.metric_rows]
        for row in rows:
            if row["record_id"] in bridge_records:
                row["lineage_group_ids"] = sorted(
                    {*row["lineage_group_ids"], "shared-lineage-bridge"}
                )
        return replace(execution, metric_rows=tuple(rows))


class _CrossSeedComputeMismatchExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        if request.arm.arm_id != "uniform_valid" and request.seed == PRODUCTION_SEEDS[1]:
            return replace(execution, parameter_count=960_000)
        return execution


class _OverBudgetExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        if request.arm.arm_id == "transformer" and request.seed == PRODUCTION_SEEDS[0]:
            return replace(
                execution,
                wall_time_seconds=request.budget.wall_time_seconds + 0.001,
            )
        return execution


class _OptimizerBudgetExceededExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        if request.arm.arm_id == "transformer" and request.seed == PRODUCTION_SEEDS[0]:
            assert execution.budget_consumption is not None
            return replace(
                execution,
                budget_consumption=replace(
                    execution.budget_consumption,
                    optimizer_steps_completed=(request.budget.maximum_optimizer_steps + 1),
                ),
            )
        return execution


class _UnsupportedDirectionExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        if request.arm.arm_id != "gnn:pure_eml_dag":
            return execution
        rows = [dict(row) for row in execution.metric_rows]
        parsed = StepMetricOutcomeV1.from_dict(rows[0])
        unsupported = CandidateMetricOutcomeV1(
            rank=3,
            action=_action(
                "rule.c",
                label="unsupported-rule-c-backward",
                path=(99,),
                direction="backward",
            ),
            status=CandidateMetricStatus.UNSUPPORTED,
            exact_demonstration_action=False,
            exact_successor_structure=False,
            verifier_confirmed_valid=False,
            successor_signature=None,
            legality_status=LegalityStatus.UNSUPPORTED,
            replay_status=None,
            verifier_status=None,
            legality_detail="direction is absent from the frozen registry",
            replay_detail=None,
            verifier_detail=None,
            parse_error=None,
        )
        rows[0] = replace(
            parsed,
            candidates=(*parsed.candidates[:2], unsupported),
        ).as_dict()
        return replace(execution, metric_rows=tuple(rows))


class _RegisteredInvalidRuleExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        if request.arm.arm_id != "gnn:pure_eml_dag":
            return execution
        rows = [dict(row) for row in execution.metric_rows]
        parsed = StepMetricOutcomeV1.from_dict(rows[0])
        invalid = CandidateMetricOutcomeV1(
            rank=3,
            action=_action(
                "rule.c",
                label="invalid-registered-rule-c",
                path=(99,),
            ),
            status=CandidateMetricStatus.INVALID_SITE,
            exact_demonstration_action=False,
            exact_successor_structure=False,
            verifier_confirmed_valid=False,
            successor_signature=None,
            legality_status=LegalityStatus.INVALID_SITE,
            replay_status=None,
            verifier_status=None,
            legality_detail="the registered rule is not legal at this occurrence",
            replay_detail=None,
            verifier_detail=None,
            parse_error=None,
        )
        rows[0] = replace(
            parsed,
            candidates=(*parsed.candidates[:2], invalid),
        ).as_dict()
        return replace(execution, metric_rows=tuple(rows))


class _OutOfInventoryLegalCandidateExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        if request.arm.arm_id != "transformer" or request.seed != PRODUCTION_SEEDS[0]:
            return execution
        rows = [dict(row) for row in execution.metric_rows]
        parsed = StepMetricOutcomeV1.from_dict(rows[0])
        candidate = parsed.candidates[2]
        assert candidate.action is not None
        rows[0] = replace(
            parsed,
            candidates=(
                *parsed.candidates[:2],
                replace(
                    candidate,
                    action=replace(
                        candidate.action,
                        action_digest=_digest("registered but outside inventory"),
                    ),
                ),
            ),
        ).as_dict()
        return replace(execution, metric_rows=tuple(rows))


class _ProductionFixtureExecutor(_FixtureExecutor):
    def __call__(self, request) -> GridCellExecution:
        execution = super().__call__(request)
        envelope = dict(execution.run_envelope or {})
        envelope.update(
            {
                "cell_id": request.cell_id,
                "config_digest": request.config_digest,
                "exact_command": request.reproduction_command,
                "rule_registry_sha256": request.rule_registry_sha256,
                "seed": request.seed,
                "step_manifest_sha256": request.step_manifest_sha256,
                "training_config_sha256": _digest("training config"),
                "training_family_inventory_sha256": (_FAMILY_EVIDENCE.inventory_digest),
                "step_population_sha256": _fixture_step_population_digest(),
                "budget_consumption": execution.budget_consumption.as_dict(),
                "wall_time_seconds": execution.wall_time_seconds,
                "verifier_sha256": request.verifier_sha256,
            }
        )
        return replace(execution, run_envelope=envelope)


def _production_envelope_adapter(envelope, *, stage):
    assert stage is GridStage.PRODUCTION
    assert isinstance(envelope, dict)
    return envelope


def _manifest_authenticator(
    reference,
    *,
    expected_sha256,
    expected_step_count,
    expected_training_family_inventory_sha256,
    expected_step_population_sha256,
):
    return {
        "accepted_step_count": expected_step_count,
        "manifest_reference": reference,
        "manifest_sha256": expected_sha256,
        "training_family_inventory_sha256": (expected_training_family_inventory_sha256),
        "step_population_sha256": expected_step_population_sha256,
        "schema_version": STEP_MANIFEST_AUTH_SCHEMA_VERSION,
        "status": "authenticated",
    }


def _run(config: Goal7GridConfig, executor: _FixtureExecutor, **kwargs):
    return run_goal7_grid(
        config,
        executor=executor,
        run_envelope=current_fixture_run_envelope(exact_command="pytest goal7"),
        envelope_adapter=fixture_run_envelope_adapter,
        **kwargs,
    )


def test_count_25_fixture_runs_all_arms_and_seeds_and_renders(tmp_path: Path) -> None:
    config = _config(tmp_path / "grid")
    requests = enumerate_grid_cells(config)
    assert len(requests) == 18
    assert {request.arm.arm_id for request in requests} == {
        "gnn:ast_dag",
        "gnn:frequent_macro_motif_dag",
        "gnn:motif_ast_fair_control",
        "gnn:pure_eml_dag",
        "transformer",
        "uniform_valid",
    }
    assert {request.seed for request in requests} == set(PRODUCTION_SEEDS)
    assert len({request.budget.digest for request in requests}) == 1

    executor = _FixtureExecutor()
    receipt = _run(config, executor)
    assert receipt.complete
    assert receipt.retained_cell_count == 18
    assert receipt.status_counts == (("complete", 18),)
    assert len(executor.calls) == 18

    summary = build_goal7_summary(
        receipt.run_directory,
        expected_config_digest=config.digest,
        expected_step_manifest_sha256=config.step_manifest_sha256,
        expected_rule_registry_sha256=config.rule_registry_sha256,
    )
    payload = summary.as_dict()
    assert payload["gate"]["verdict"] == GateVerdict.INSUFFICIENT_EVIDENCE.value
    assert len(payload["raw_seed_rows"]) == 18
    assert payload["primary_rule_coverage"]["dead_rule_ids"] == ["rule.c"]
    contrasts = {row["metric"]: row for row in payload["paired_contrasts"]}
    assert contrasts["demonstration_action_match"]["pair_count"] == 75
    assert contrasts["demonstration_action_match"]["cluster_count"] == 5
    assert contrasts["demonstration_action_match"]["margin"] > 0

    json_path = tmp_path / "report" / "goal7.json"
    markdown_path = tmp_path / "report" / "goal7.md"
    write_goal7_summary(summary, json_path=json_path, markdown_path=markdown_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["content_digest"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "insufficient_evidence" in markdown
    assert markdown.count("| demonstration_action_match |") == 1
    assert markdown.count("| exact_successor_structure_match |") == 1
    assert markdown.count("| verifier_valid_success |") == 1
    plot_paths = render_goal7_plots(build_plot_data(summary), tmp_path / "plots")
    assert all(path.stat().st_size > 0 for path in plot_paths)


def test_resume_skips_immutable_cells_without_reexecution(tmp_path: Path) -> None:
    config = _config(tmp_path / "grid")
    executor = _FixtureExecutor()
    partial = _run(config, executor, interrupt_after_new_cells=3)
    assert not partial.complete
    assert partial.retained_cell_count == 3
    assert len(executor.calls) == 3
    incomplete = build_goal7_summary(partial.run_directory).as_dict()
    assert incomplete["gate"]["verdict"] == "insufficient_evidence"
    assert incomplete["retained_cell_count"] == 3
    assert len(incomplete["missing_cell_ids"]) == 15

    resumed = _run(config, executor)
    assert resumed.complete
    assert resumed.resumed_cell_count == 3
    assert len(executor.calls) == 18

    final = _run(config, executor)
    assert final.complete
    assert final.resumed_cell_count == 18
    assert len(executor.calls) == 18


def test_failure_is_retained_and_reported_without_rerun(tmp_path: Path) -> None:
    config = _config(tmp_path / "grid")
    failed_cell = ("transformer", PRODUCTION_SEEDS[1])
    executor = _FixtureExecutor(timeout_arm_seed=failed_cell)
    first = _run(config, executor)
    assert first.complete
    assert first.status_counts == (("complete", 17), ("timeout", 1))

    replacement = _FixtureExecutor()
    second = _run(config, replacement)
    assert second.resumed_cell_count == 18
    assert replacement.calls == []
    summary = build_goal7_summary(second.run_directory)
    raw_failures = [
        row for row in summary.as_dict()["raw_seed_rows"] if row["status"] != "complete"
    ]
    assert len(raw_failures) == 1
    assert raw_failures[0]["error_type"] == "TimeoutError"
    assert summary.as_dict()["gate"]["verdict"] == "insufficient_evidence"


def test_authenticated_loader_rejects_hash_mismatch_and_tampering(tmp_path: Path) -> None:
    config = _config(tmp_path / "grid")
    receipt = _run(config, _FixtureExecutor())
    with pytest.raises(Goal7ProtocolError, match="config_digest"):
        load_goal7_run_evidence(
            receipt.run_directory,
            expected_config_digest=_digest("wrong config"),
        )
    with pytest.raises(Goal7ProtocolError, match="step_manifest_sha256"):
        load_goal7_run_evidence(
            receipt.run_directory,
            expected_step_manifest_sha256=_digest("wrong steps"),
        )
    with pytest.raises(Goal7ProtocolError, match="rule_registry_sha256"):
        load_goal7_run_evidence(
            receipt.run_directory,
            expected_rule_registry_sha256=_digest("wrong rules"),
        )

    evidence = load_goal7_run_evidence(receipt.run_directory)
    first_cell = evidence.cells[0]
    cell_id = first_cell["cell_id"]
    assert isinstance(cell_id, str)
    cell_path = receipt.run_directory / "cells" / cell_id[:2] / f"{cell_id}.json"
    payload = json.loads(cell_path.read_text(encoding="utf-8"))
    payload["seed"] += 1
    cell_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Goal7ProtocolError, match="content digest"):
        load_goal7_run_evidence(receipt.run_directory)


def test_loader_rejects_semantically_invalid_self_consistent_cell(
    tmp_path: Path,
) -> None:
    receipt = _run(_config(tmp_path / "grid"), _FixtureExecutor())
    evidence = load_goal7_run_evidence(receipt.run_directory)
    uniform = next(cell for cell in evidence.cells if cell["arm_id"] == "uniform_valid")
    cell_id = uniform["cell_id"]
    assert isinstance(cell_id, str)
    cell_path = receipt.run_directory / "cells" / cell_id[:2] / f"{cell_id}.json"
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    cell["checkpoint_sha256"] = _digest("forged uniform checkpoint")
    cell["parameter_count"] = 1
    cell["estimated_flops"] = 1.0
    _rehash_payload(cell, grid_module._CELL_CONTENT_DOMAIN)
    cell_path.write_text(json.dumps(cell), encoding="utf-8")

    assert receipt.completion_path is not None
    completion = json.loads(receipt.completion_path.read_text(encoding="utf-8"))
    completion["cell_content_digests"][cell_id] = cell["content_digest"]
    _rehash_payload(completion, grid_module._RUN_CONTENT_DOMAIN)
    receipt.completion_path.write_text(json.dumps(completion), encoding="utf-8")

    with pytest.raises(Goal7ProtocolError, match="uniform-valid"):
        load_goal7_run_evidence(receipt.run_directory)


def test_all_cells_without_completion_ledger_remain_insufficient(tmp_path: Path) -> None:
    receipt = _run(_config(tmp_path / "grid"), _FixtureExecutor())
    assert receipt.completion_path is not None
    receipt.completion_path.unlink()
    summary = build_goal7_summary(receipt.run_directory).as_dict()
    assert summary["retained_cell_count"] == 18
    assert summary["missing_cell_ids"] == []
    assert summary["run_complete"] is False
    assert summary["gate"]["verdict"] == "insufficient_evidence"
    assert any("completion ledger" in reason for reason in summary["gate"]["reasons"])


def test_checked_in_production_config_exposes_blockers() -> None:
    config = load_goal7_grid_config("configs/goal7_grid.yaml")
    blockers = config.production_blockers()
    assert config.stage is GridStage.PRODUCTION
    assert config.expected_step_count is None
    assert "issue #56 four-channel contract is unresolved" in blockers
    assert "expected_step_count is unresolved" in blockers
    assert "training_family_inventory_sha256 is unresolved" in blockers
    assert "step_population_sha256 is unresolved" in blockers
    assert "analysis_reproduction_command is unresolved" in blockers
    with pytest.raises(Goal7ProtocolError, match="not runnable"):
        enumerate_grid_cells(config)


def test_production_requires_exact_step_manifest_authentication(
    tmp_path: Path,
) -> None:
    config = replace(_config(tmp_path / "grid"), stage=GridStage.PRODUCTION)
    executor = _ProductionFixtureExecutor()
    envelope = current_fixture_run_envelope(exact_command="pytest goal7 production")
    with pytest.raises(Goal7ProtocolError, match="manifest authenticator"):
        run_goal7_grid(
            config,
            executor=executor,
            run_envelope=envelope,
            envelope_adapter=_production_envelope_adapter,
        )

    def wrong_count(
        reference,
        *,
        expected_sha256,
        expected_step_count,
        expected_training_family_inventory_sha256,
        expected_step_population_sha256,
    ):
        evidence = _manifest_authenticator(
            reference,
            expected_sha256=expected_sha256,
            expected_step_count=expected_step_count,
            expected_training_family_inventory_sha256=(expected_training_family_inventory_sha256),
            expected_step_population_sha256=expected_step_population_sha256,
        )
        evidence["accepted_step_count"] = expected_step_count + 1
        return evidence

    with pytest.raises(Goal7ProtocolError, match="authentication evidence"):
        run_goal7_grid(
            config,
            executor=executor,
            run_envelope=envelope,
            envelope_adapter=_production_envelope_adapter,
            step_manifest_authenticator=wrong_count,
        )

    def wrong_family_inventory(
        reference,
        *,
        expected_sha256,
        expected_step_count,
        expected_training_family_inventory_sha256,
        expected_step_population_sha256,
    ):
        evidence = _manifest_authenticator(
            reference,
            expected_sha256=expected_sha256,
            expected_step_count=expected_step_count,
            expected_training_family_inventory_sha256=(expected_training_family_inventory_sha256),
            expected_step_population_sha256=expected_step_population_sha256,
        )
        evidence["training_family_inventory_sha256"] = _digest("wrong training-family inventory")
        return evidence

    with pytest.raises(Goal7ProtocolError, match="authentication evidence"):
        run_goal7_grid(
            config,
            executor=executor,
            run_envelope=envelope,
            envelope_adapter=_production_envelope_adapter,
            step_manifest_authenticator=wrong_family_inventory,
        )

    receipt = run_goal7_grid(
        config,
        executor=executor,
        run_envelope=envelope,
        envelope_adapter=_production_envelope_adapter,
        step_manifest_authenticator=_manifest_authenticator,
    )
    assert receipt.complete
    assert receipt.status_counts == (("complete", 18),)

    with pytest.raises(Goal7ProtocolError, match="post-hoc"):
        build_goal7_summary(
            receipt.run_directory,
            gate_policy=GatePolicyV1(minimum_exact_action_margin=0.0),
        )

    analysis_envelope = {
        "config_digest": config.digest,
        "exact_command": config.analysis_reproduction_command.format(run_id=receipt.run_id),
        "implementation_sha256": config.implementation_sha256,
        "rule_registry_sha256": config.rule_registry_sha256,
        "run_id": receipt.run_id,
        "step_manifest_sha256": config.step_manifest_sha256,
        "training_config_sha256": config.training_config_sha256,
        "training_family_inventory_sha256": (config.training_family_inventory_sha256),
        "step_population_sha256": config.step_population_sha256,
        "verifier_sha256": config.verifier_sha256,
    }
    with pytest.raises(Goal7ProtocolError, match="mismatched bindings"):
        build_goal7_summary(
            receipt.run_directory,
            analysis_run_envelope={**analysis_envelope, "config_digest": _digest("wrong")},
            analysis_envelope_adapter=_production_envelope_adapter,
        )
    with pytest.raises(Goal7ProtocolError, match="exact_command"):
        build_goal7_summary(
            receipt.run_directory,
            analysis_run_envelope={**analysis_envelope, "exact_command": "wrong"},
            analysis_envelope_adapter=_production_envelope_adapter,
        )
    summary = build_goal7_summary(
        receipt.run_directory,
        analysis_run_envelope=analysis_envelope,
        analysis_envelope_adapter=_production_envelope_adapter,
    )
    assert summary.as_dict()["gate"]["verdict"] == "insufficient_evidence"


def test_metric_family_inventory_must_match_the_frozen_config(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path / "grid"),
        training_family_inventory_sha256=_digest("wrong inventory"),
    )
    receipt = _run(config, _FixtureExecutor())
    assert receipt.status_counts == (("invalid", 18),)
    evidence = load_goal7_run_evidence(receipt.run_directory)
    assert all(
        "different training-family inventory" in str(cell["error_message"])
        for cell in evidence.cells
    )


def test_complete_cells_must_match_the_authenticated_step_population(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path / "grid"),
        step_population_sha256=_digest("substituted same-size population"),
    )
    receipt = _run(config, _FixtureExecutor())
    assert receipt.status_counts == (("invalid", 18),)
    evidence = load_goal7_run_evidence(receipt.run_directory)
    assert all(
        "authenticated step population" in str(cell["error_message"]) for cell in evidence.cells
    )


def test_uniform_valid_order_uses_only_the_shared_mask_and_is_stable() -> None:
    legal = (1, 3, 4, 8, 10, 12)
    first = uniform_valid_order(legal, seed=PRODUCTION_SEEDS[0], record_id=_digest("row"))
    second = uniform_valid_order(legal, seed=PRODUCTION_SEEDS[0], record_id=_digest("row"))
    assert first == second
    assert tuple(sorted(first)) == legal
    row = _metric_row(7, arm_id="uniform_valid", seed=PRODUCTION_SEEDS[0])
    selected = uniform_valid_order(
        (0, 1, 2),
        seed=PRODUCTION_SEEDS[0],
        record_id=row.record_id,
    )[0]
    assert row.candidates[0].exact_demonstration_action is (selected == 0)
    assert row.candidates[0].exact_successor_structure is (selected in {0, 1})
    with pytest.raises(TypeError, match="sorted unique"):
        uniform_valid_order((3, 1), seed=PRODUCTION_SEEDS[0], record_id=_digest("row"))


def test_uniform_cell_rejects_a_permutation_that_did_not_use_the_frozen_draw(
    tmp_path: Path,
) -> None:
    receipt = _run(_config(tmp_path / "grid"), _CheatingUniformExecutor())
    assert receipt.status_counts == (("complete", 17), ("invalid", 1))
    evidence = load_goal7_run_evidence(receipt.run_directory)
    rejected = next(
        cell
        for cell in evidence.cells
        if cell["arm_id"] == "uniform_valid" and cell["seed"] == PRODUCTION_SEEDS[0]
    )
    assert rejected["status"] == "invalid"
    assert rejected["uniform_draw_audit_count"] == 0
    assert rejected["rejected_uniform_draw_audit_count"] == 25
    assert rejected["rejected_metric_row_count"] == 25
    assert "deterministic draw" in rejected["error_message"]


def test_loader_recomputes_uniform_draw_in_self_consistent_cell(
    tmp_path: Path,
) -> None:
    receipt = _run(_config(tmp_path / "grid"), _FixtureExecutor())
    evidence = load_goal7_run_evidence(receipt.run_directory)
    uniform = next(
        cell
        for cell in evidence.cells
        if cell["arm_id"] == "uniform_valid" and cell["seed"] == PRODUCTION_SEEDS[0]
    )
    cell_id = uniform["cell_id"]
    assert isinstance(cell_id, str)
    cell_path = receipt.run_directory / "cells" / cell_id[:2] / f"{cell_id}.json"
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    ranking = cell["uniform_draw_audits"][0]["ranked_action_digests"]
    ranking[0], ranking[1] = ranking[1], ranking[0]
    cell["uniform_draw_audits_digest"] = grid_module._sha256_json(cell["uniform_draw_audits"])
    _rehash_payload(cell, grid_module._CELL_CONTENT_DOMAIN)
    cell_path.write_text(json.dumps(cell), encoding="utf-8")

    assert receipt.completion_path is not None
    completion = json.loads(receipt.completion_path.read_text(encoding="utf-8"))
    completion["cell_content_digests"][cell_id] = cell["content_digest"]
    _rehash_payload(completion, grid_module._RUN_CONTENT_DOMAIN)
    receipt.completion_path.write_text(json.dumps(completion), encoding="utf-8")

    with pytest.raises(Goal7ProtocolError, match="deterministic draw"):
        load_goal7_run_evidence(receipt.run_directory)


def test_learned_cell_rejects_uniform_draw_evidence(tmp_path: Path) -> None:
    receipt = _run(_config(tmp_path / "grid"), _LearnedAuditExecutor())
    assert receipt.status_counts == (("complete", 17), ("invalid", 1))
    evidence = load_goal7_run_evidence(receipt.run_directory)
    rejected = next(
        cell
        for cell in evidence.cells
        if cell["arm_id"] == "transformer" and cell["seed"] == PRODUCTION_SEEDS[0]
    )
    assert rejected["status"] == "invalid"
    assert rejected["uniform_draw_audit_count"] == 0
    assert rejected["rejected_uniform_draw_audit_count"] == 1
    assert "learned cells cannot carry" in rejected["error_message"]


def test_independent_cells_merge_despite_distinct_cell_run_envelopes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "grid")
    requests = enumerate_grid_cells(config)
    executor = _FixtureExecutor()
    first = run_goal7_grid(
        config,
        executor=executor,
        run_envelope=current_fixture_run_envelope(exact_command=requests[0].reproduction_command),
        envelope_adapter=fixture_run_envelope_adapter,
        requested_cell_ids=(requests[0].cell_id,),
    )
    assert not first.complete
    second = run_goal7_grid(
        config,
        executor=executor,
        run_envelope=current_fixture_run_envelope(exact_command=requests[1].reproduction_command),
        envelope_adapter=fixture_run_envelope_adapter,
        requested_cell_ids=(requests[1].cell_id,),
    )
    assert not second.complete
    assert second.retained_cell_count == 2
    evidence = load_goal7_run_evidence(second.run_directory, allow_incomplete=True)
    assert {cell["cell_id"] for cell in evidence.cells} == {
        requests[0].cell_id,
        requests[1].cell_id,
    }
    assert len(evidence.missing_cell_ids) == 16


def test_completed_distributed_grid_is_idempotent_across_cell_commands(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "grid")
    requests = enumerate_grid_cells(config)
    executor = _FixtureExecutor()
    receipt = None
    for request in requests:
        receipt = run_goal7_grid(
            config,
            executor=executor,
            run_envelope=current_fixture_run_envelope(exact_command=request.reproduction_command),
            envelope_adapter=fixture_run_envelope_adapter,
            requested_cell_ids=(request.cell_id,),
        )
    assert receipt is not None
    assert receipt.complete
    resumed = run_goal7_grid(
        config,
        executor=executor,
        run_envelope=current_fixture_run_envelope(exact_command=requests[0].reproduction_command),
        envelope_adapter=fixture_run_envelope_adapter,
        requested_cell_ids=(requests[0].cell_id,),
    )
    assert resumed.complete
    assert len(executor.calls) == 18


def test_grid_retains_cross_arm_legal_mask_mismatch_as_invalid_evidence(
    tmp_path: Path,
) -> None:
    receipt = _run(_config(tmp_path / "grid"), _MaskMismatchExecutor())
    assert receipt.status_counts == (("complete", 17), ("invalid", 1))
    summary = build_goal7_summary(receipt.run_directory).as_dict()
    invalid = [row for row in summary["raw_seed_rows"] if row["status"] == "invalid"]
    assert len(invalid) == 1
    assert invalid[0]["rejected_metric_row_count"] == 25
    assert invalid[0]["error_type"] == "Goal7ProtocolError"
    assert summary["gate"]["verdict"] == "insufficient_evidence"


def test_grid_retains_cross_arm_metadata_mismatch_as_invalid_evidence(
    tmp_path: Path,
) -> None:
    receipt = _run(_config(tmp_path / "grid"), _MetadataMismatchExecutor())
    assert receipt.status_counts == (("complete", 17), ("invalid", 1))
    summary = build_goal7_summary(receipt.run_directory).as_dict()
    invalid = [row for row in summary["raw_seed_rows"] if row["status"] == "invalid"]
    assert len(invalid) == 1
    assert invalid[0]["rejected_metric_row_count"] == 25
    assert invalid[0]["error_type"] == "Goal7ProtocolError"
    assert summary["gate"]["verdict"] == "insufficient_evidence"


def test_group_bootstrap_merges_overlapping_lineage_components(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path / "grid"),
        step_population_sha256=_fixture_step_population_digest(lineage_bridge=True),
    )
    receipt = _run(config, _LineageBridgeExecutor())
    summary = build_goal7_summary(receipt.run_directory).as_dict()
    contrasts = summary["paired_contrasts"]
    assert all(row["cluster_count"] == 4 for row in contrasts)


def test_compute_matching_rejects_cross_seed_architecture_drift(
    tmp_path: Path,
) -> None:
    receipt = _run(
        _config(tmp_path / "grid"),
        _CrossSeedComputeMismatchExecutor(),
    )
    summary = build_goal7_summary(receipt.run_directory).as_dict()
    compute = summary["compute_matching"]
    assert all(row["status"] == "matched" for row in compute["per_seed"])
    assert compute["status"] == "insufficient"
    assert all(row["status"] == "outside_tolerance" for row in compute["per_arm_across_seeds"])


def test_over_budget_complete_cell_is_retained_as_invalid(tmp_path: Path) -> None:
    receipt = _run(_config(tmp_path / "grid"), _OverBudgetExecutor())
    assert receipt.status_counts == (("complete", 17), ("invalid", 1))
    summary = build_goal7_summary(receipt.run_directory).as_dict()
    invalid = [row for row in summary["raw_seed_rows"] if row["status"] == "invalid"]
    assert len(invalid) == 1
    assert invalid[0]["rejected_metric_row_count"] == 25
    assert "wall-time budget" in invalid[0]["error_message"]


def test_optimizer_budget_violation_is_retained_as_invalid(tmp_path: Path) -> None:
    receipt = _run(
        _config(tmp_path / "grid"),
        _OptimizerBudgetExceededExecutor(),
    )
    assert receipt.status_counts == (("complete", 17), ("invalid", 1))
    evidence = load_goal7_run_evidence(receipt.run_directory)
    invalid = [cell for cell in evidence.cells if cell["status"] == "invalid"]
    assert len(invalid) == 1
    assert "optimizer-step budget" in str(invalid[0]["error_message"])
    assert invalid[0]["budget_consumption"]["optimizer_steps_completed"] == 4


def test_unsupported_direction_does_not_revive_a_dead_rule(tmp_path: Path) -> None:
    receipt = _run(_config(tmp_path / "grid"), _UnsupportedDirectionExecutor())
    coverage = build_goal7_summary(receipt.run_directory).as_dict()["primary_rule_coverage"]
    assert "rule.c" in coverage["dead_rule_ids"]
    assert ["rule.c", "backward"] in coverage["unregistered_proposed_rule_directions"]
    assert coverage["masked_or_unregistered_candidate_count"] == 3


def test_registered_illegal_candidate_does_not_revive_a_dead_rule(tmp_path: Path) -> None:
    receipt = _run(_config(tmp_path / "grid"), _RegisteredInvalidRuleExecutor())
    coverage = build_goal7_summary(receipt.run_directory).as_dict()["primary_rule_coverage"]
    assert "rule.c" in coverage["dead_rule_ids"]
    assert ["rule.c", "forward"] not in coverage["proposed_rule_directions"]
    assert coverage["masked_or_unregistered_candidate_count"] == 3


def test_legal_candidate_must_belong_to_the_shared_inventory(
    tmp_path: Path,
) -> None:
    with pytest.raises(Goal7ProtocolError, match="absent from the shared legal inventory"):
        _run(
            _config(tmp_path / "grid"),
            _OutOfInventoryLegalCandidateExecutor(),
        )
