"""Fixture coverage for scripts/fill_goal7_identities.py against a real published dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from geml.data.steps.extract import (
    ActionDirectionV1,
    CanonicalJson,
    NormalizedActionV1,
    NormalizedStateV1,
    NormalizedTraceV1,
    ReplayResultV1,
    ReplayStatusV1,
    SplitV1,
    StepDatasetConfigV1,
    StepRecordV1,
    StepRuntimeIdentityV1,
    VerificationResultV1,
    VerificationStatusV1,
    extract_step_dataset,
    load_step_rows,
    write_step_dataset,
)
from geml.data.steps.stratify import RuleRegistryEntryV1, RuleRegistrySnapshotV1
from geml.experiments.goal7.run_grid import compute_step_population_digest
from geml.learning.eval.step_metrics import (
    FAMILY_PARTITION_EVIDENCE_SCHEMA_VERSION,
    FamilyPartitionEvidenceV1,
)
from geml.learning.harness.config import config_digest, load_harness_config
from scripts.fill_goal7_identities import compute_identities, main, tree_sha256

RULE_DIGEST = hashlib.sha256(b"fixture-rule-set").hexdigest()
VERIFIER_DIGEST = hashlib.sha256(b"fixture-verifier").hexdigest()
SOURCE_ROOT = Path("src/geml")


def _digest(value: object) -> str:
    text = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _state(name: str, family: str) -> NormalizedStateV1:
    payload = {"fixture_expression": name}
    return NormalizedStateV1(
        state=CanonicalJson.from_value(payload),
        structural_signature=_digest({"state": payload}),
        family=family,
    )


def _action(source: NormalizedStateV1, successor: NormalizedStateV1) -> NormalizedActionV1:
    # Nonempty path and argument so the population digest exercises the
    # ordered_arguments_json / occurrence_path adaptation, not just [].
    payload = {
        "direction": "forward",
        "fixture_replay_status": "applied",
        "fixture_successor": successor.state.to_value(),
        "fixture_successor_signature": successor.structural_signature,
        "fixture_verification_status": "accepted",
        "occurrence_path": [0],
        "ordered_arguments": [{"slot": "x"}],
        "rule_id": "R-ADD-ZERO",
    }
    return NormalizedActionV1(
        action=CanonicalJson.from_value(payload),
        action_digest=_digest({"fixture-action": payload}),
        rule_id="R-ADD-ZERO",
        direction=ActionDirectionV1.FORWARD,
        occurrence_path=(0,),
        ordered_arguments=(CanonicalJson.from_value({"slot": "x"}),),
        source_signature=source.structural_signature,
        successor_signature=successor.structural_signature,
        assumptions=("real",),
        domain_mode="safe_real",
        stored_verification_status=VerificationStatusV1.ACCEPTED,
        stored_verifier_digest=VERIFIER_DIGEST,
        stored_verification_evidence_digest=_digest({"stored-verification": payload}),
    )


@dataclass(frozen=True)
class _FixtureInput:
    source_bytes: bytes
    normalized: NormalizedTraceV1


def _trace(
    trace_id: str,
    states: tuple[NormalizedStateV1, ...],
    actions: tuple[NormalizedActionV1, ...],
    *,
    split: SplitV1,
) -> _FixtureInput:
    source_bytes = (
        json.dumps(
            {
                "actions": [action.action.to_value() for action in actions],
                "split": split.value,
                "states": [state.state.to_value() for state in states],
                "trace_id": trace_id,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    input_digest = hashlib.sha256(source_bytes).hexdigest()
    return _FixtureInput(
        source_bytes=source_bytes,
        normalized=NormalizedTraceV1(
            trace_schema_version="fixture-rewrite-trace-v1",
            trace_id=trace_id,
            trace_digest=_digest({"authenticated-trace": source_bytes.decode()}),
            input_record_digest=input_digest,
            pair_id=f"pair-{trace_id}",
            source_id=f"source-{trace_id}",
            source_group=f"group-{trace_id}",
            lineage_group_ids=(f"group-{trace_id}",),
            authoritative_split=split,
            evaluation_views=("fixture",),
            source_family=states[0].family,
            domain_mode="safe_real",
            rewrite_mode="safe_real",
            rule_set_digest=RULE_DIGEST,
            authentication_evidence_digest=_digest({"authenticated": input_digest}),
            stored_replay_verified=True,
            states=states,
            actions=actions,
        ),
    )


class _FixtureAdapter:
    def input_record_digest(self, trace: object) -> str:
        assert isinstance(trace, _FixtureInput)
        return hashlib.sha256(trace.source_bytes).hexdigest()

    def normalize_and_authenticate(self, trace: object) -> NormalizedTraceV1:
        assert isinstance(trace, _FixtureInput)
        return trace.normalized

    def structural_signature(self, state: CanonicalJson) -> str:
        return _digest({"state": state.to_value()})


class _FixtureReplayer:
    def apply(self, current_state: CanonicalJson, action: NormalizedActionV1) -> ReplayResultV1:
        del current_state
        payload = action.action.to_value()
        assert isinstance(payload, dict)
        return ReplayResultV1(
            status=ReplayStatusV1.APPLIED,
            reason="fixture action applied",
            successor_state=CanonicalJson.from_value(payload["fixture_successor"]),
            successor_signature=payload["fixture_successor_signature"],
        )


class _FixtureVerifier:
    verifier_digest = VERIFIER_DIGEST

    def verify(self, current_state, successor_state, action, *, assumptions, domain_mode):
        del current_state, successor_state, assumptions, domain_mode
        return VerificationResultV1(
            status=VerificationStatusV1.ACCEPTED,
            reason="fixture verifier accepted",
            verifier_digest=VERIFIER_DIGEST,
            evidence_digest=_digest({"verified": action.action_digest}),
        )


def _publish(root: Path) -> None:
    train_current = _state("x + 0", "algebraic_core")
    train_middle = _state("x", "algebraic_core")
    train_goal = _state("0 + x", "trig_hyperbolic")
    iid_current = _state("exp(y)", "exp_log")
    iid_goal = _state("e**y", "exp_log")
    inputs = (
        _trace(
            "train-0",
            (train_current, train_middle, train_goal),
            (_action(train_current, train_middle), _action(train_middle, train_goal)),
            split=SplitV1.TRAIN,
        ),
        _trace(
            "iid-0",
            (iid_current, iid_goal),
            (_action(iid_current, iid_goal),),
            split=SplitV1.TEST_IID,
        ),
    )
    result = extract_step_dataset(
        inputs,
        adapter=_FixtureAdapter(),
        replayer=_FixtureReplayer(),
        verifier=_FixtureVerifier(),
        expected_verifier_digest=VERIFIER_DIGEST,
        expected_rule_set_digest=RULE_DIGEST,
    )
    config = StepDatasetConfigV1(
        seed=20260726,
        shard_size=2,
        expected_input_digest=result.input_digest,
        expected_rule_set_digest=RULE_DIGEST,
        expected_verifier_digest=VERIFIER_DIGEST,
        exact_command="python -m geml.data.steps.extract --config configs/goal7_steps.yaml",
        runtime=StepRuntimeIdentityV1(
            git_commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe",
            python_version="3.12.fixture",
            hardware="fixture-cpu",
            package_versions=(("geml", "0.1.0"),),
            deterministic_settings=("fixture_no_randomness",),
        ),
    )
    registry = RuleRegistrySnapshotV1(
        authoritative_rule_set_digest=RULE_DIGEST,
        source_schema_version="fixture-rule-registry-v1",
        source_content_digest=_digest(["R-ADD-ZERO"]),
        entries=(RuleRegistryEntryV1(rule_id="R-ADD-ZERO", direction="forward", supported=True),),
    )
    write_step_dataset(result, output_root=root, config=config, frozen_registry=registry)


def _expected_population_record(row: StepRecordV1) -> dict[str, object]:
    # Deliberately re-states the frozen adaptation instead of reusing the script's helper.
    return {
        "record_id": row.record_id,
        "trace_id": row.trace_id,
        "source_group": row.source_group,
        "lineage_group_ids": list(row.lineage_group_ids),
        "authoritative_split": row.authoritative_split.value,
        "current_signature": row.current_signature,
        "goal_signature": row.goal_signature,
        "target_successor_signature": row.next_signature,
        "current_family": row.current_family,
        "goal_family": row.goal_family,
        "evaluation_views": list(row.evaluation_views),
        "remaining_witness_steps": row.remaining_witness_steps,
        "trace_length": row.trace_length,
        "demonstration_action": {
            "action_digest": row.action_digest,
            "direction": row.direction.value,
            "occurrence_path": list(row.occurrence_path),
            "ordered_arguments_json": [argument.text for argument in row.ordered_arguments],
            "rule_id": row.rule_id,
        },
    }


def test_identities_match_the_published_dataset(tmp_path) -> None:
    root = tmp_path / "steps"
    _publish(root)
    values = compute_identities(root, source_root=SOURCE_ROOT)

    manifest_path = root / "manifest.json"
    assert values["expected_step_count"] == 3
    assert values["step_manifest"] == str(manifest_path)
    assert values["step_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert values["rule_registry_sha256"] == RULE_DIGEST
    assert values["verifier_sha256"] == VERIFIER_DIGEST

    accepted = [row for row in load_step_rows(manifest_path) if isinstance(row, StepRecordV1)]
    assert values["step_population_sha256"] == compute_step_population_digest(
        [_expected_population_record(row) for row in accepted]
    )
    # The inventory covers current/goal families of train rows only; exp_log is test_iid.
    assert values["training_family_ids"] == ["algebraic_core", "trig_hyperbolic"]
    assert (
        values["training_family_inventory_sha256"]
        == FamilyPartitionEvidenceV1(
            schema_version=FAMILY_PARTITION_EVIDENCE_SCHEMA_VERSION,
            step_manifest_digest=values["step_manifest_sha256"],
            training_family_ids=("algebraic_core", "trig_hyperbolic"),
        ).inventory_digest
    )

    gin = SOURCE_ROOT / "learning" / "backbones" / "gin.py"
    assert values["shared_gnn_architecture_sha256"] == hashlib.sha256(gin.read_bytes()).hexdigest()
    assert values["shared_harness_sha256"] == tree_sha256(SOURCE_ROOT / "learning" / "harness")
    assert values["implementation_sha256"] == tree_sha256(SOURCE_ROOT)


def test_tree_hash_pins_names_and_bytes(tmp_path) -> None:
    (tmp_path / "a.py").write_bytes(b"x = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_bytes(b"y = 2\n")
    first = tree_sha256(tmp_path)
    assert first == tree_sha256(tmp_path)
    (tmp_path / "sub" / "b.py").write_bytes(b"y = 3\n")
    assert tree_sha256(tmp_path) != first
    (tmp_path / "sub" / "b.py").rename(tmp_path / "sub" / "c.py")
    renamed = tree_sha256(tmp_path)
    (tmp_path / "sub" / "c.py").write_bytes(b"y = 2\n")
    assert tree_sha256(tmp_path) not in (first, renamed)


def test_main_prints_json_and_hashes_the_training_config(tmp_path, capsys) -> None:
    root = tmp_path / "steps"
    _publish(root)
    payload = {
        "run_name": "goal7-production",
        "output_root": "outputs/final/goal7",
        "seed": 20260726,
    }
    config_path = tmp_path / "training_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        main(
            [
                "--steps-root",
                str(root),
                "--training-config",
                str(config_path),
                "--source-root",
                str(SOURCE_ROOT),
            ]
        )
        == 0
    )
    values = json.loads(capsys.readouterr().out)
    assert values["training_config_sha256"] == config_digest(load_harness_config(payload))
    assert values["expected_step_count"] == 3


def test_missing_dataset_is_a_typed_refusal(tmp_path, capsys) -> None:
    assert main(["--steps-root", str(tmp_path / "nowhere")]) == 2
    err = capsys.readouterr().err
    assert "refusing to derive identities" in err
    assert "manifest.json" in err


def test_tampered_manifest_is_refused(tmp_path, capsys) -> None:
    root = tmp_path / "steps"
    _publish(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["accepted_count"] = 999
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n")
    assert main(["--steps-root", str(root)]) == 2
    assert "refusing to derive identities" in capsys.readouterr().err


def test_missing_train_rows_refuse_an_empty_inventory(tmp_path) -> None:
    root = tmp_path / "steps"
    current = _state("exp(y)", "exp_log")
    goal = _state("e**y", "exp_log")
    inputs = (_trace("iid-0", (current, goal), (_action(current, goal),), split=SplitV1.TEST_IID),)
    result = extract_step_dataset(
        inputs,
        adapter=_FixtureAdapter(),
        replayer=_FixtureReplayer(),
        verifier=_FixtureVerifier(),
        expected_verifier_digest=VERIFIER_DIGEST,
        expected_rule_set_digest=RULE_DIGEST,
    )
    config = StepDatasetConfigV1(
        seed=20260726,
        shard_size=2,
        expected_input_digest=result.input_digest,
        expected_rule_set_digest=RULE_DIGEST,
        expected_verifier_digest=VERIFIER_DIGEST,
        exact_command="python -m geml.data.steps.extract --config configs/goal7_steps.yaml",
        runtime=StepRuntimeIdentityV1(
            git_commit="41e0dc8cc6315fcc5593355a38c7e317d88eeafe",
            python_version="3.12.fixture",
            hardware="fixture-cpu",
            package_versions=(("geml", "0.1.0"),),
            deterministic_settings=("fixture_no_randomness",),
        ),
    )
    registry = RuleRegistrySnapshotV1(
        authoritative_rule_set_digest=RULE_DIGEST,
        source_schema_version="fixture-rule-registry-v1",
        source_content_digest=_digest(["R-ADD-ZERO"]),
        entries=(RuleRegistryEntryV1(rule_id="R-ADD-ZERO", direction="forward", supported=True),),
    )
    write_step_dataset(result, output_root=root, config=config, frozen_registry=registry)
    with pytest.raises(ValueError, match="no accepted train-split rows"):
        compute_identities(root, source_root=SOURCE_ROOT)
