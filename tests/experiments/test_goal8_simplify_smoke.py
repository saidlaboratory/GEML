"""Phase-A tests for target-free, verifier-gated Goal 8 simplification."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import geml.experiments.goal8.run_simplify as simplify_runner
from geml.experiments.goal8.run_simplify import (
    GOAL3_COST_OBJECTIVE,
    SIMPLIFY_CONFIG_SCHEMA,
    ExactCostEvidence,
    NeighborhoodExecution,
    NeighborhoodExecutionAttestation,
    SemanticVerification,
    SimplificationCandidate,
    SimplificationConfig,
    SimplificationMethod,
    SimplificationMethodConfig,
    SimplificationProtocolError,
    SimplificationRuntimeIdentity,
    SimplificationState,
    SimplificationStatus,
    SympyExecution,
    freeze_simplification_sample,
    load_frozen_sample,
    load_simplification_config,
    run_simplification_shard,
    run_sympy_simplify,
)

_DIGESTS = {letter: letter * 64 for letter in "abcdef"}


def test_runtime_processor_identity_has_truthful_architecture_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simplify_runner.platform, "machine", lambda: "test-arch")
    monkeypatch.setattr(simplify_runner.platform, "processor", lambda: "")
    monkeypatch.setattr(
        simplify_runner.platform,
        "uname",
        lambda: SimpleNamespace(machine="test-arch", processor=""),
    )
    monkeypatch.delenv("PROCESSOR_IDENTIFIER", raising=False)
    monkeypatch.setattr(simplify_runner.Path, "read_text", lambda *_args, **_kwargs: "")

    assert simplify_runner._hardware_identity() == (
        "test-arch",
        "architecture:test-arch",
    )


def test_production_runtime_provenance_is_strict_and_checked_before_output(
    tmp_path: Path,
) -> None:
    valid = SimplificationRuntimeIdentity(
        git_commit="ab" * 20,
        python_version="3.12.4",
        platform="Linux-6.8-x86_64",
        machine="x86_64",
        processor="AMD EPYC",
        package_versions={
            "geml": "0.1.0",
            "sympy": "1.14.0",
            "pydantic": "2.11.0",
            "pyyaml": "6.0.2",
        },
    )
    valid.require_production_ready()
    for invalid_commit in ("fixture", "a" * 40, "AB" * 20):
        with pytest.raises(SimplificationProtocolError, match="git_commit"):
            replace(valid, git_commit=invalid_commit).require_production_ready()
    with pytest.raises(SimplificationProtocolError, match="package_versions"):
        replace(
            valid,
            package_versions={"geml": "0.1.0", "sympy": "not-installed"},
        ).require_production_ready()

    output_root = tmp_path / "must-not-exist"
    production = _config(tmp_path).model_copy(
        update={"stage": "production", "output_root": str(output_root)}
    )
    with pytest.raises(SimplificationProtocolError, match="git_commit"):
        run_simplification_shard(
            config=production,
            sample_manifest_path=Path(production.sample_manifest),
            candidates=_candidates(),
            shard_index=0,
            explorer=_Explorer(),
            cost_oracle=_CostOracle(),
            verifier=_Verifier(),
            runtime=_runtime(),
            sympy_comparator=_Comparator(),
        )
    assert not output_root.exists()


def _expression_id(domain_mode: str, sympy_srepr: str) -> str:
    payload = f"geml-expression-v1\0{domain_mode}\0{sympy_srepr}".encode()
    return hashlib.sha256(payload).hexdigest()


def _config(
    root: Path,
    *,
    expected_sample_count: int = 2,
    shard_count: int = 1,
    missing_checkpoint: bool = False,
) -> SimplificationConfig:
    root.mkdir(parents=True, exist_ok=True)
    source_manifest = root / "fixture-corpus.manifest.json"
    source_bytes = json.dumps(
        {
            "schema_version": "geml-goal8-simplify-source-fixture-v1",
            "records": [candidate.identity_payload() for candidate in _candidates()],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    source_manifest.write_bytes(source_bytes)
    exclusion_manifest = root / "fixture-exclusions.manifest.json"
    exclusion_bytes = json.dumps(
        {
            "schema_version": "geml-goal8-simplify-exclusion-fixture-v1",
            "forbidden_group_ids": ["excluded-group"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    exclusion_manifest.write_bytes(exclusion_bytes)
    return SimplificationConfig.model_validate(
        {
            "schema_version": SIMPLIFY_CONFIG_SCHEMA,
            "stage": "fixture",
            "output_root": str(root / "results"),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "learned_exclusion_manifest": str(exclusion_manifest),
            "learned_exclusion_manifest_sha256": hashlib.sha256(exclusion_bytes).hexdigest(),
            "sample_manifest": str(root / "sample.manifest.json"),
            "expected_sample_count": expected_sample_count,
            "sample": {
                "seed": 20260726,
                "strata_axes": (
                    "family",
                    "depth_bucket",
                    "size_bucket",
                    "domain_mode",
                    "split",
                ),
                "depth_bucket_edges": (2, 4, 8),
                "size_bucket_edges": (3, 7, 15),
            },
            "seeds": (20260726, 20260727, 20260728),
            "methods": (
                {
                    "method": SimplificationMethod.UNIFORM,
                    "learned": False,
                    "stochastic": True,
                    "guide_role": "proposal_prioritization_only",
                    "checkpoint_selection_split": "not_applicable",
                },
                {
                    "method": SimplificationMethod.POLICY,
                    "learned": True,
                    "stochastic": False,
                    "guide_role": "proposal_prioritization_only",
                    "checkpoint_selection_split": "validation",
                    "policy_checkpoint_sha256": (None if missing_checkpoint else _DIGESTS["b"]),
                },
            ),
            "budget": {
                "beam_width": 2,
                "expanded_node_budget": 11,
                "generated_state_budget": 17,
                "search_depth_limit": 3,
                "wall_time_seconds": 1.0,
                "verifier_call_budget": 19,
            },
            "sympy_wall_time_seconds": 1.0,
            "sympy_implementation_sha256": _DIGESTS["f"],
            "cost_objective": GOAL3_COST_OBJECTIVE,
            "shard_count": shard_count,
            "rule_set_sha256": _DIGESTS["d"],
            "verifier_sha256": _DIGESTS["e"],
            "implementation_sha256": _DIGESTS["f"],
            "reproduction_command": "fixture-simplify --shard-index {shard_index}",
        }
    )


def _candidates(count: int = 6) -> tuple[SimplificationCandidate, ...]:
    rows = []
    for index in range(count):
        sympy_srepr = f"Add(Symbol('x{index}', real=True), Integer(0))"
        domain_mode = "safe_real" if index % 2 == 0 else "positive_real"
        rows.append(
            SimplificationCandidate(
                expression_id=_expression_id(domain_mode, sympy_srepr),
                sympy_srepr=sympy_srepr,
                source_signature=f"source-{index}",
                group_id=("excluded-group" if index == count - 1 else f"group-{index}"),
                family=("algebraic_core" if index % 2 == 0 else "exp_log"),
                ast_depth=1,
                ast_size=3,
                domain_mode=domain_mode,
                split=("test_iid" if index % 2 == 0 else "test_ood"),
                assumptions=(f"real(x{index})",),
            )
        )
    return tuple(rows)


def _runtime() -> SimplificationRuntimeIdentity:
    return SimplificationRuntimeIdentity(
        git_commit="fixture",
        python_version="3.12.fixture",
        platform="fixture",
        machine="fixture",
        processor="fixture",
        package_versions={"geml": "fixture", "sympy": "1.14.0"},
    )


def _attestation(method, budget) -> NeighborhoodExecutionAttestation:
    method_config = SimplificationMethodConfig.model_validate(
        {
            "method": method,
            "learned": method is SimplificationMethod.POLICY,
            "stochastic": method is SimplificationMethod.UNIFORM,
            "guide_role": "proposal_prioritization_only",
            "checkpoint_selection_split": (
                "validation" if method is SimplificationMethod.POLICY else "not_applicable"
            ),
            "policy_checkpoint_sha256": (
                _DIGESTS["b"] if method is SimplificationMethod.POLICY else None
            ),
        }
    )
    return NeighborhoodExecutionAttestation(
        method=method,
        checkpoint_digest=method_config.checkpoint_digest,
        rule_set_sha256=_DIGESTS["d"],
        verifier_sha256=_DIGESTS["e"],
        implementation_sha256=_DIGESTS["f"],
        budget_digest=budget.digest,
    )


def _freeze(
    root: Path,
    config: SimplificationConfig,
    candidates: tuple[SimplificationCandidate, ...],
) -> Path:
    return freeze_simplification_sample(
        config=config,
        candidates=candidates,
        manifest_path=root / "sample.manifest.json",
        results_root=root / "results",
    )


class _Explorer:
    def __init__(
        self,
        *,
        omit_source: bool = False,
        duplicate: bool = False,
        timeout: bool = False,
        terminal_status: str = "complete",
        events: list[str] | None = None,
    ) -> None:
        self.omit_source = omit_source
        self.duplicate = duplicate
        self.timeout = timeout
        self.terminal_status = terminal_status
        self.calls: list[tuple[str, SimplificationMethod, int, int]] = []
        self.events = events

    def __call__(
        self,
        source,
        method,
        seed,
        budget,
        *,
        checkpoint_path,
        resume,
    ):
        del checkpoint_path, resume
        if self.events is not None:
            self.events.append(f"geml:{source.sympy_srepr}")
        self.calls.append((source.signature, method, seed, id(budget)))
        if self.timeout:
            raise TimeoutError("retained GEML timeout")
        cheap = SimplificationState(
            signature=f"a-cheap-{source.signature}",
            sympy_srepr=source.sympy_srepr.replace("Integer(0)", "Integer(1)"),
            ast_size=max(1, source.ast_size - 1),
            ast_depth=source.ast_depth,
            path_verified=True,
        )
        states = (cheap,) if self.omit_source else (source, cheap)
        if self.duplicate:
            states = (*states, cheap)
        return NeighborhoodExecution(
            status=self.terminal_status,
            termination_reason="budget_exhausted",
            attestation=_attestation(method, budget),
            visited_states=states,
            expanded_count=2,
            generated_count=3,
            valid_count=2,
            invalid_count=1,
            duplicate_count=0,
            verifier_call_count=3,
            verifier_error_count=0,
            verifier_timeout_count=0,
            frontier_peak=2,
            search_depth_reached=1,
            wall_time_seconds=0.01,
            peak_host_memory_bytes=2048,
        )


class _FaultExplorer(_Explorer):
    def __init__(self, fault: str) -> None:
        super().__init__()
        self.fault = fault

    def __call__(self, source, method, seed, budget, **kwargs):
        execution = super().__call__(source, method, seed, budget, **kwargs)
        if self.fault == "source_spoof":
            spoof = replace(
                source,
                sympy_srepr="Integer(999)",
                ast_size=1,
                ast_depth=0,
            )
            return replace(execution, visited_states=(spoof, *execution.visited_states[1:]))
        if self.fault == "budget":
            return replace(
                execution,
                generated_count=budget.generated_state_budget + 1,
            )
        if self.fault == "attestation":
            return replace(
                execution,
                attestation=replace(
                    execution.attestation,
                    implementation_sha256=_DIGESTS["a"],
                ),
            )
        if self.fault == "verifier_timeout":
            return replace(
                execution,
                status="verifier_timeout",
                verifier_timeout_count=1,
            )
        raise AssertionError(f"unknown fixture fault: {self.fault}")


class _CostOracle:
    def __init__(self, *, unavailable: bool = False, tie: bool = False) -> None:
        self.unavailable = unavailable
        self.tie = tie

    def __call__(self, state, *, domain_mode):
        del domain_mode
        if self.unavailable:
            return ExactCostEvidence(
                status="unsupported",
                objective=GOAL3_COST_OBJECTIVE,
                value=None,
                representation_mode=None,
                evidence_digest=None,
                error_type="Unsupported",
                error_message="retained",
            )
        value = 3 if "cheap" in state.signature else 10
        if self.tie and "cheap" in state.signature:
            value = 1
        return ExactCostEvidence(
            status="success",
            objective=GOAL3_COST_OBJECTIVE,
            value=value,
            representation_mode="pure_eml:official_v4",
            evidence_digest=_DIGESTS["a"],
        )


class _EqualCostOracle:
    def __call__(self, state, *, domain_mode):
        del state, domain_mode
        return ExactCostEvidence(
            status="success",
            objective=GOAL3_COST_OBJECTIVE,
            value=10,
            representation_mode="pure_eml:official_v4",
            evidence_digest=_DIGESTS["a"],
        )


class _Verifier:
    def __init__(
        self,
        *,
        valid: bool | None = True,
        status: str | None = None,
    ) -> None:
        self.valid = valid
        self.status = status
        self.calls: list[tuple[str, str, str, tuple[str, ...]]] = []

    def __call__(
        self,
        source,
        result,
        *,
        domain_mode,
        assumptions,
    ):
        self.calls.append((source.signature, result.signature, domain_mode, assumptions))
        return SemanticVerification(
            status=self.status or ("verified" if self.valid else "rejected"),
            valid=self.valid,
            evidence_digest=_DIGESTS["b"] if self.valid else None,
            rule_set_sha256=_DIGESTS["d"] if self.valid else None,
            verifier_sha256=_DIGESTS["e"] if self.valid else None,
            error_type=None if self.valid else "VerificationRejected",
            error_message=None if self.valid else "retained",
        )


class _Comparator:
    def __init__(
        self,
        *,
        timeout: bool = False,
        events: list[str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.calls: list[tuple[str, float]] = []
        self.events = events

    def __call__(self, source_srepr, *, timeout_seconds):
        self.calls.append((source_srepr, timeout_seconds))
        if self.events is not None:
            self.events.append("sympy")
        if self.timeout:
            return SympyExecution(
                status="wall_timeout",
                result_srepr=None,
                result_ast_size=None,
                result_ast_depth=None,
                wall_time_seconds=timeout_seconds,
                implementation_sha256=_DIGESTS["f"],
                wall_time_budget_seconds=timeout_seconds,
                error_type="TimeoutError",
                error_message="retained",
            )
        return SympyExecution(
            status="complete",
            result_srepr=source_srepr,
            result_ast_size=3,
            result_ast_depth=1,
            wall_time_seconds=0.01,
            implementation_sha256=_DIGESTS["f"],
            wall_time_budget_seconds=timeout_seconds,
        )


def _rows(root: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "results").rglob("cells/**/*.json"))
    ]


def test_repository_config_loads_as_production_pending() -> None:
    config = load_simplification_config(Path("configs/goal8_simplify.yaml"))
    assert config.stage == "production"
    assert config.expected_sample_count == 1000
    with pytest.raises(SimplificationProtocolError, match="not runnable"):
        config.require_runnable()


def test_config_freezes_cost_objective_modes_and_guide_role(tmp_path: Path) -> None:
    payload = _config(tmp_path).model_dump(mode="python")
    payload["cost_objective"] = "ad_hoc_ast_size"
    with pytest.raises(ValidationError, match="cost_objective"):
        SimplificationConfig.model_validate(payload)

    payload = _config(tmp_path).model_dump(mode="python")
    payload["methods"][1]["guide_role"] = "final_learned_cost"
    with pytest.raises(ValidationError, match="proposal_prioritization_only"):
        SimplificationConfig.model_validate(payload)

    parameters = inspect.signature(_Explorer.__call__).parameters
    assert "target" not in parameters
    assert "goal" not in parameters

    with pytest.raises(ValueError, match="pure_eml:official_v4"):
        ExactCostEvidence(
            status="success",
            objective=GOAL3_COST_OBJECTIVE,
            value=1,
            representation_mode="ast",
            evidence_digest=_DIGESTS["a"],
        )
    with pytest.raises(ValueError, match="status and valid flag"):
        SemanticVerification(
            status="invalid",
            valid=True,
            evidence_digest=_DIGESTS["a"],
            rule_set_sha256=None,
            verifier_sha256=None,
        )
    with pytest.raises(ValueError, match="partial success evidence"):
        ExactCostEvidence(
            status="unsupported",
            objective=GOAL3_COST_OBJECTIVE,
            value=1,
            representation_mode=None,
            evidence_digest=None,
            error_type="Unsupported",
            error_message="fixture",
        )
    with pytest.raises(ValueError, match="unsupported SymPy status"):
        SympyExecution(
            status="timeout",
            result_srepr=None,
            result_ast_size=None,
            result_ast_depth=None,
            wall_time_seconds=1.0,
            implementation_sha256=_DIGESTS["f"],
            wall_time_budget_seconds=1.0,
            error_type="TimeoutError",
            error_message="fixture",
        )


def test_candidate_identity_and_assumptions_are_canonical() -> None:
    candidate = _candidates(1)[0]
    with pytest.raises(ValueError, match="expression_id"):
        SimplificationCandidate(
            **{
                **candidate.identity_payload(),
                "expression_id": "0" * 64,
                "assumptions": candidate.assumptions,
            }
        )
    with pytest.raises(ValueError, match="canonically sorted"):
        SimplificationCandidate(
            **{
                **candidate.identity_payload(),
                "assumptions": ("real(z)", "real(a)"),
            }
        )


def test_sample_freeze_is_deterministic_stratified_and_input_order_independent(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, expected_sample_count=4)
    candidates = _candidates()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_config = config.model_copy(update={"sample_manifest": str(first_path)})
    second_config = config.model_copy(update={"sample_manifest": str(second_path)})
    first = freeze_simplification_sample(
        config=first_config,
        candidates=candidates,
        manifest_path=first_path,
    )
    second = freeze_simplification_sample(
        config=second_config,
        candidates=tuple(reversed(candidates)),
        manifest_path=second_path,
    )
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))

    assert first_payload == second_payload
    assert first_payload["sample_size"] == 4
    assert len(set(first_payload["ordered_expression_ids"])) == 4
    assert all(len(record["stratum"]) == 5 for record in first_payload["records"])
    assert "excluded-group" not in {record["group_id"] for record in first_payload["records"]}
    assert load_frozen_sample(first, config=first_config, candidates=candidates)


def test_sample_must_precede_outputs_and_authenticate_full_source_population(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    candidates = _candidates()
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "premature.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SimplificationProtocolError, match="after simplification outputs"):
        _freeze(tmp_path, config, candidates)

    (tmp_path / "results" / "premature.json").unlink()
    manifest = _freeze(tmp_path, config, candidates)
    changed = list(candidates)
    changed_srepr = "Integer(1)"
    changed[0] = SimplificationCandidate(
        **{
            **changed[0].identity_payload(),
            "expression_id": _expression_id(changed[0].domain_mode, changed_srepr),
            "sympy_srepr": changed_srepr,
            "assumptions": changed[0].assumptions,
        }
    )
    with pytest.raises(
        SimplificationProtocolError,
        match=r"candidate projection does not match|population changed",
    ):
        load_frozen_sample(manifest, config=config, candidates=tuple(changed))

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["content_digest"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SimplificationProtocolError, match="failed authentication"):
        load_frozen_sample(manifest, config=config, candidates=candidates)


def test_self_rehashed_sample_cannot_change_the_deterministic_selection(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["ordered_expression_ids"].reverse()
    payload["records"].reverse()
    content = {key: value for key, value in payload.items() if key != "content_digest"}
    payload["content_digest"] = simplify_runner.sha256_hex(
        simplify_runner._SAMPLE_CONTENT_DOMAIN
        + simplify_runner.canonical_json(content).encode("utf-8")
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SimplificationProtocolError, match="deterministic selection"):
        load_frozen_sample(manifest, config=config, candidates=candidates)


def test_run_reads_the_authenticated_sample_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    real_load_json = simplify_runner.load_json
    sample_reads = 0

    def counted_load_json(path, *, label):
        nonlocal sample_reads
        if Path(path).resolve() == manifest.resolve():
            sample_reads += 1
        return real_load_json(path, label=label)

    monkeypatch.setattr(simplify_runner, "load_json", counted_load_json)
    run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        runtime=_runtime(),
        sympy_comparator=_Comparator(),
    )

    assert sample_reads == 1


def test_configured_sample_and_input_manifest_bytes_are_mandatory(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    candidates = _candidates()
    with pytest.raises(SimplificationProtocolError, match="must equal configured path"):
        freeze_simplification_sample(
            config=config,
            candidates=candidates,
            manifest_path=tmp_path / "wrong-sample.json",
        )

    Path(config.source_manifest).write_text("{}", encoding="utf-8")
    with pytest.raises(SimplificationProtocolError, match="source manifest checksum mismatch"):
        _freeze(tmp_path, config, candidates)

    config = _config(tmp_path)
    Path(config.learned_exclusion_manifest).write_text("{}", encoding="utf-8")
    with pytest.raises(
        SimplificationProtocolError,
        match="learned exclusion manifest checksum mismatch",
    ):
        _freeze(tmp_path, config, candidates)


def test_source_manifest_artifacts_root_expands_only_at_file_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    candidates = _candidates()
    artifacts_root = tmp_path / "artifacts"
    resolved_manifest = artifacts_root / "corpus" / "manifest.json"
    resolved_manifest.parent.mkdir(parents=True)
    resolved_manifest.write_bytes(Path(config.source_manifest).read_bytes())
    configured_path = "${GEML_ARTIFACTS_ROOT}/corpus/manifest.json"
    config = config.model_copy(update={"source_manifest": configured_path})
    monkeypatch.setenv("GEML_ARTIFACTS_ROOT", str(artifacts_root))

    manifest = _freeze(tmp_path, config, candidates)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert config.source_manifest == configured_path
    assert payload["source_manifest"] == configured_path


def test_reproduction_command_rejects_unresolved_template_tokens(tmp_path: Path) -> None:
    payload = _config(tmp_path).model_dump(mode="python")
    payload["reproduction_command"] = "fixture-simplify --shard-index {other}"
    with pytest.raises(ValidationError, match="unsupported template token"):
        SimplificationConfig.model_validate(payload)


@pytest.mark.parametrize("environment_value", [None, "${UNRESOLVED_ROOT}"])
def test_source_manifest_artifacts_root_fails_closed_when_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_value: str | None,
) -> None:
    config = _config(tmp_path).model_copy(
        update={"source_manifest": "${GEML_ARTIFACTS_ROOT}/corpus/manifest.json"}
    )
    if environment_value is None:
        monkeypatch.delenv("GEML_ARTIFACTS_ROOT", raising=False)
    else:
        monkeypatch.setenv("GEML_ARTIFACTS_ROOT", environment_value)

    with pytest.raises(SimplificationProtocolError, match="GEML_ARTIFACTS_ROOT"):
        _freeze(tmp_path, config, _candidates())


def test_source_projector_must_match_authenticated_candidate_projection(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    candidates = _candidates()
    changed = list(candidates)
    changed[0] = replace(changed[0], group_id="spoofed-group")
    with pytest.raises(SimplificationProtocolError, match="candidate projection does not match"):
        freeze_simplification_sample(
            config=config,
            candidates=candidates,
            manifest_path=Path(config.sample_manifest),
            source_projector=lambda _path: tuple(changed),
        )


def test_default_configured_output_root_prevents_late_sample_freeze(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    results = Path(config.output_root)
    results.mkdir(parents=True)
    (results / "premature.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SimplificationProtocolError, match="after simplification outputs"):
        freeze_simplification_sample(
            config=config,
            candidates=_candidates(),
            manifest_path=Path(config.sample_manifest),
        )


def test_learned_sample_requires_manifest_derived_excluded_groups(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    exclusion = Path(config.learned_exclusion_manifest)
    empty_bytes = json.dumps(
        {
            "schema_version": "geml-goal8-simplify-exclusion-fixture-v1",
            "forbidden_group_ids": [],
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    exclusion.write_bytes(empty_bytes)
    config = config.model_copy(
        update={"learned_exclusion_manifest_sha256": hashlib.sha256(empty_bytes).hexdigest()}
    )
    with pytest.raises(SimplificationProtocolError, match="nonempty manifest-derived"):
        freeze_simplification_sample(
            config=config,
            candidates=_candidates(),
            manifest_path=Path(config.sample_manifest),
        )


def test_run_authenticates_the_exact_frozen_exclusion_set(tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    with pytest.raises(SimplificationProtocolError, match="forbidden groups do not match"):
        run_simplification_shard(
            config=config,
            sample_manifest_path=manifest,
            candidates=candidates,
            shard_index=0,
            explorer=_Explorer(),
            cost_oracle=_CostOracle(),
            verifier=_Verifier(),
            runtime=_runtime(),
            exclusion_projector=lambda _path: ("different-group",),
            sympy_comparator=_Comparator(),
        )
    assert not Path(config.output_root).exists()


def test_target_free_run_selects_exact_cost_and_keeps_ids_separate(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    events: list[str] = []
    explorer = _Explorer(events=events)
    comparator = _Comparator(events=events)
    verifier = _Verifier()
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=explorer,
        cost_oracle=_CostOracle(),
        verifier=verifier,
        sympy_comparator=comparator,
        runtime=_runtime(),
    )

    assert receipt.expected_count == receipt.attempted_count == 10
    assert receipt.status_counts == {
        SimplificationStatus.COMPLETE.value: 8,
        SimplificationStatus.NO_CHANGE.value: 2,
    }
    assert len(explorer.calls) == 8
    assert len({call[3] for call in explorer.calls}) == 1
    assert len(comparator.calls) == 2
    assert events[:4] == [events[0]] * 4
    assert events[4] == "sympy"

    geml_rows = [row for row in _rows(tmp_path) if row["kind"] == "geml"]
    assert all("cheap" in row["selected_state"]["signature"] for row in geml_rows)
    assert all(row["selected_exact_cost"]["value"] == 3 for row in geml_rows)
    assert all(row["source_exact_cost"]["value"] == 10 for row in geml_rows)
    assert all(row["exact_cost_delta"] == -7 for row in geml_rows)
    assert all(row["source_depth_bucket"] for row in geml_rows)
    assert all(row["source_size_bucket"] for row in geml_rows)
    assert all(len(row["sample_stratum"]) == 5 for row in geml_rows)
    assert Counter(row["seed_policy"] for row in geml_rows) == {
        "canonical_seed_deterministic": 2,
        "three_seed_stochastic": 6,
    }
    assert all(row["semantic_verification"]["valid"] is True for row in geml_rows)
    assert all(row["source_expression_id"] != row["method_result_id"] for row in geml_rows)
    assert all(row["verification_evidence_id"] != row["method_result_id"] for row in geml_rows)
    assert all(len(row["visited_states"]) == 2 for row in geml_rows)
    assert all(
        state["result_expression_id"] is not None
        for row in geml_rows
        for state in row["visited_states"]
    )
    sympy_rows = [row for row in _rows(tmp_path) if row["kind"] == "sympy"]
    assert all(row["seed"] is None for row in sympy_rows)
    assert all(row["seed_policy"] == "deterministic_unseeded_comparator" for row in sympy_rows)
    assert all(row["source_exact_cost"]["status"] == "success" for row in sympy_rows)
    assert {row["reproduction_command"] for row in (*geml_rows, *sympy_rows)} == {
        "fixture-simplify --shard-index 0"
    }
    completion = json.loads(receipt.completion_path.read_text(encoding="utf-8"))
    assert completion["reproduction_command"] == "fixture-simplify --shard-index 0"


def test_runner_measures_neighborhood_wall_budget_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidate = _candidates()[0]
    method = config.methods[0]
    cell = {
        **simplify_runner._cell_identity(
            run_id="fixture-run",
            candidate=candidate,
            kind="geml",
            method=method.method.value,
            seed=config.seeds[0],
            checkpoint_digest=method.checkpoint_digest,
            method_config=method,
        ),
        "_shard_index": 0,
    }
    clock = iter((0.0, 0.0, 2.0, 2.0))
    monkeypatch.setattr(simplify_runner.time, "perf_counter", lambda: next(clock))

    row = simplify_runner._run_geml_cell(
        config=config,
        run_id="fixture-run",
        sample_digest="a" * 64,
        cell=cell,
        candidate=candidate,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        runtime=_runtime(),
        search_checkpoint=tmp_path / "search.json",
    )

    assert row["status"] == SimplificationStatus.WALL_TIMEOUT
    assert row["search_wall_time_seconds"] == 0.01
    assert row["measured_search_wall_time_seconds"] == 2.0
    assert row["runner_wall_time_seconds"] == 2.0
    assert row["error_type"] == "MeasuredSearchWallBudgetExceeded"


def test_neighborhood_execution_rejects_invalid_peak_memory(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    method = config.methods[0]
    execution = _Explorer()(
        simplify_runner._source_state(_candidates()[0]),
        method.method,
        config.seeds[0],
        config.budget,
        checkpoint_path=tmp_path / "search.json",
        resume=False,
    )
    with pytest.raises(ValueError, match="nonnegative exact integer"):
        replace(execution, peak_host_memory_bytes=-1)


@pytest.mark.parametrize(
    ("fault", "expected_status"),
    [
        ("source_spoof", SimplificationStatus.INVALID),
        ("budget", SimplificationStatus.INVALID),
        ("attestation", SimplificationStatus.INVALID),
        ("verifier_timeout", SimplificationStatus.VERIFIER_TIMEOUT),
    ],
)
def test_source_spoofs_budgets_identities_and_timeout_types_fail_closed(
    tmp_path: Path,
    fault: str,
    expected_status: SimplificationStatus,
) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=_freeze(tmp_path, config, candidates),
        candidates=candidates,
        shard_index=0,
        explorer=_FaultExplorer(fault),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {
        expected_status.value: 4,
        SimplificationStatus.NO_CHANGE.value: 1,
    }
    failed = [row for row in _rows(tmp_path) if row["kind"] == "geml"]
    assert all(row["method_result_id"] is None for row in failed)


def test_equal_exact_cost_always_selects_the_source(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=_freeze(tmp_path, config, candidates),
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_EqualCostOracle(),
        verifier=_Verifier(),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {SimplificationStatus.NO_CHANGE.value: 5}
    geml_rows = [row for row in _rows(tmp_path) if row["kind"] == "geml"]
    assert all(row["selected_state"] == row["source_state"] for row in geml_rows)
    assert all(row["method_result_id"] == row["source_expression_id"] for row in geml_rows)


def test_verifier_and_sympy_component_spoofs_cannot_count(
    tmp_path: Path,
) -> None:
    class WrongVerifier(_Verifier):
        def __call__(self, *args, **kwargs):
            evidence = super().__call__(*args, **kwargs)
            return replace(evidence, verifier_sha256=_DIGESTS["a"])

    class WrongComparator(_Comparator):
        def __call__(self, *args, **kwargs):
            execution = super().__call__(*args, **kwargs)
            return replace(execution, implementation_sha256=_DIGESTS["a"])

    class OverBudgetComparator(_Comparator):
        def __call__(self, *args, **kwargs):
            execution = super().__call__(*args, **kwargs)
            return replace(execution, wall_time_seconds=2.0)

    candidates = _candidates()
    verifier_root = tmp_path / "verifier"
    config = _config(verifier_root, expected_sample_count=1)
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=_freeze(verifier_root, config, candidates),
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(),
        verifier=WrongVerifier(),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {SimplificationStatus.VERIFICATION_FAILED.value: 5}

    sympy_root = tmp_path / "sympy"
    config = _config(sympy_root, expected_sample_count=1)
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=_freeze(sympy_root, config, candidates),
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        sympy_comparator=WrongComparator(),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {
        SimplificationStatus.COMPLETE.value: 4,
        SimplificationStatus.INVALID.value: 1,
    }

    budget_root = tmp_path / "sympy-budget"
    config = _config(budget_root, expected_sample_count=1)
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=_freeze(budget_root, config, candidates),
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        sympy_comparator=OverBudgetComparator(),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {
        SimplificationStatus.COMPLETE.value: 4,
        SimplificationStatus.WALL_TIMEOUT.value: 1,
    }


@pytest.mark.parametrize("fault", ["missing_source", "duplicate"])
def test_invalid_neighborhoods_are_retained_not_selected(
    tmp_path: Path,
    fault: str,
) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(
            omit_source=fault == "missing_source",
            duplicate=fault == "duplicate",
        ),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {
        SimplificationStatus.INVALID.value: 4,
        SimplificationStatus.NO_CHANGE.value: 1,
    }
    invalid = [row for row in _rows(tmp_path) if row["status"] == "invalid"]
    assert len(invalid) == 4
    assert all(row["method_result_id"] is None for row in invalid)


def test_unavailable_exact_cost_and_failed_verification_remain_explicit(
    tmp_path: Path,
) -> None:
    candidates = _candidates()

    unavailable_root = tmp_path / "unavailable"
    config = _config(unavailable_root, expected_sample_count=1)
    manifest = _freeze(unavailable_root, config, candidates)
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(unavailable=True),
        verifier=_Verifier(),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    assert receipt.status_counts[SimplificationStatus.COST_UNAVAILABLE] == 4
    sympy_row = next(row for row in _rows(unavailable_root) if row["kind"] == "sympy")
    assert sympy_row["source_exact_cost"]["status"] == "unsupported"
    assert sympy_row["selected_exact_cost"]["status"] == "unsupported"
    assert sympy_row["exact_cost_delta"] is None

    rejected_root = tmp_path / "rejected"
    config = _config(rejected_root, expected_sample_count=1)
    manifest = _freeze(rejected_root, config, candidates)
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(valid=False),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {SimplificationStatus.VERIFICATION_FAILED.value: 5}


def test_geml_and_sympy_timeouts_are_retained_for_every_attempt(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=_freeze(tmp_path, config, candidates),
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(timeout=True),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        sympy_comparator=_Comparator(timeout=True),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {SimplificationStatus.WALL_TIMEOUT.value: 5}
    assert len(_rows(tmp_path)) == 5


def test_unsupported_search_and_verification_remain_typed(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=_freeze(tmp_path, config, candidates),
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(terminal_status="unsupported"),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(valid=None, status="unsupported"),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    assert receipt.status_counts == {SimplificationStatus.UNSUPPORTED.value: 5}


class _SourceCostUnavailable:
    def __call__(self, state, *, domain_mode):
        del domain_mode
        if state.signature.startswith("source-"):
            return ExactCostEvidence(
                status="unsupported",
                objective=GOAL3_COST_OBJECTIVE,
                value=None,
                representation_mode=None,
                evidence_digest=None,
                error_type="Unsupported",
                error_message="source cost unavailable",
            )
        return ExactCostEvidence(
            status="success",
            objective=GOAL3_COST_OBJECTIVE,
            value=1,
            representation_mode="pure_eml:official_v4",
            evidence_digest=_DIGESTS["a"],
        )


def test_geml_never_selects_when_source_exact_cost_is_unavailable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=_freeze(tmp_path, config, candidates),
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_SourceCostUnavailable(),
        verifier=_Verifier(),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    assert receipt.status_counts[SimplificationStatus.COST_UNAVAILABLE] == 4
    geml_rows = [row for row in _rows(tmp_path) if row["kind"] == "geml"]
    assert all(row["selected_state"] is None for row in geml_rows)


def test_interrupted_shard_resumes_without_repeating_committed_cells(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    explorer = _Explorer()
    comparator = _Comparator()

    def interrupt(committed: int) -> None:
        if committed == 5:
            raise RuntimeError("fixture interruption")

    with pytest.raises(RuntimeError, match="fixture interruption"):
        run_simplification_shard(
            config=config,
            sample_manifest_path=manifest,
            candidates=candidates,
            shard_index=0,
            explorer=explorer,
            cost_oracle=_CostOracle(),
            verifier=_Verifier(),
            sympy_comparator=comparator,
            runtime=_runtime(),
            on_cell_committed=interrupt,
        )
    assert len(explorer.calls) + len(comparator.calls) == 5

    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=explorer,
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        sympy_comparator=comparator,
        runtime=_runtime(),
    )
    assert receipt.attempted_count == 10
    assert len(explorer.calls) + len(comparator.calls) == 10

    def unexpected(*_args, **_kwargs):
        raise AssertionError("completed shard must not execute methods")

    resumed = run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=unexpected,
        cost_oracle=unexpected,
        verifier=unexpected,
        sympy_comparator=unexpected,
        runtime=_runtime(),
    )
    assert resumed == receipt


def test_completed_shard_revalidates_every_immutable_cell(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    cell = next((tmp_path / "results").rglob("cells/**/*.json"))
    payload = json.loads(cell.read_text(encoding="utf-8"))
    payload["status"] = "tampered"
    cell.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SimplificationProtocolError, match="digest mismatch"):
        run_simplification_shard(
            config=config,
            sample_manifest_path=manifest,
            candidates=candidates,
            shard_index=0,
            explorer=_Explorer(),
            cost_oracle=_CostOracle(),
            verifier=_Verifier(),
            sympy_comparator=_Comparator(),
            runtime=_runtime(),
        )


def test_completed_shard_rejects_a_different_supplied_runtime(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )

    with pytest.raises(SimplificationProtocolError, match=r"runtime|identity mismatch"):
        run_simplification_shard(
            config=config,
            sample_manifest_path=manifest,
            candidates=candidates,
            shard_index=0,
            explorer=_Explorer(),
            cost_oracle=_CostOracle(),
            verifier=_Verifier(),
            sympy_comparator=_Comparator(),
            runtime=replace(_runtime(), processor="different-runtime"),
        )


def test_resigned_invalid_resource_telemetry_is_rejected_on_resume(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    cells = (tmp_path / "results").rglob("cells/**/*.json")
    cell = next(
        path for path in cells if json.loads(path.read_text(encoding="utf-8"))["kind"] == "geml"
    )
    payload = json.loads(cell.read_text(encoding="utf-8"))
    payload["resource_telemetry"]["peak_host_memory_bytes"] = -1
    content = {key: value for key, value in payload.items() if key != "content_digest"}
    payload["content_digest"] = simplify_runner._payload_digest(content)
    cell.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SimplificationProtocolError, match="execution evidence is invalid"):
        run_simplification_shard(
            config=config,
            sample_manifest_path=manifest,
            candidates=candidates,
            shard_index=0,
            explorer=_Explorer(),
            cost_oracle=_CostOracle(),
            verifier=_Verifier(),
            sympy_comparator=_Comparator(),
            runtime=_runtime(),
        )


def test_resigned_scientific_cell_spoof_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    cell = next((tmp_path / "results").rglob("cells/**/*.json"))
    payload = json.loads(cell.read_text(encoding="utf-8"))
    payload["source_state"]["sympy_srepr"] = "Integer(999)"
    content = {key: value for key, value in payload.items() if key != "content_digest"}
    payload["content_digest"] = simplify_runner._payload_digest(content)
    cell.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SimplificationProtocolError, match="exact source payload"):
        run_simplification_shard(
            config=config,
            sample_manifest_path=manifest,
            candidates=candidates,
            shard_index=0,
            explorer=_Explorer(),
            cost_oracle=_CostOracle(),
            verifier=_Verifier(),
            sympy_comparator=_Comparator(),
            runtime=_runtime(),
        )


def test_completion_fields_are_recomputed_from_scientific_cells(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, expected_sample_count=1)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    receipt = run_simplification_shard(
        config=config,
        sample_manifest_path=manifest,
        candidates=candidates,
        shard_index=0,
        explorer=_Explorer(),
        cost_oracle=_CostOracle(),
        verifier=_Verifier(),
        sympy_comparator=_Comparator(),
        runtime=_runtime(),
    )
    payload = json.loads(receipt.completion_path.read_text(encoding="utf-8"))
    payload["status_counts"] = {"invalid": 5}
    payload["valid_count"] = 0
    payload["failure_count"] = 5
    receipt.completion_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SimplificationProtocolError, match="derived fields"):
        run_simplification_shard(
            config=config,
            sample_manifest_path=manifest,
            candidates=candidates,
            shard_index=0,
            explorer=_Explorer(),
            cost_oracle=_CostOracle(),
            verifier=_Verifier(),
            sympy_comparator=_Comparator(),
            runtime=_runtime(),
        )


def test_checkpoint_replace_retries_only_bounded_windows_sharing_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_replace = simplify_runner.atomic_replace_json
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

    monkeypatch.setattr(simplify_runner.sys, "platform", "win32")
    monkeypatch.setattr(simplify_runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(simplify_runner, "atomic_replace_json", flaky)
    checkpoint = tmp_path / "checkpoint.json"
    simplify_runner._replace_checkpoint_json(checkpoint, {"attempt": "fixture"})
    assert calls == 3
    assert checkpoint.is_file()

    monkeypatch.setattr(
        simplify_runner,
        "atomic_replace_json",
        lambda _path, _payload: (_ for _ in ()).throw(sharing_violation()),
    )
    with pytest.raises(
        SimplificationProtocolError,
        match="remained blocked after 4 attempts",
    ):
        simplify_runner._replace_checkpoint_json(
            checkpoint,
            {"attempt": "persistent"},
        )


def test_shards_partition_expression_method_cells_exactly_once(tmp_path: Path) -> None:
    config = _config(tmp_path, expected_sample_count=2, shard_count=4)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    sets: list[set[str]] = []
    for shard_index in range(config.shard_count):
        receipt = run_simplification_shard(
            config=config,
            sample_manifest_path=manifest,
            candidates=candidates,
            shard_index=shard_index,
            explorer=_Explorer(),
            cost_oracle=_CostOracle(),
            verifier=_Verifier(),
            sympy_comparator=_Comparator(),
            runtime=_runtime(),
        )
        payload = json.loads(receipt.completion_path.read_text(encoding="utf-8"))
        sets.append(set(payload["expected_cell_ids"]))

    assert sum(map(len, sets)) == 10
    assert len(set().union(*sets)) == 10
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            assert left.isdisjoint(right)


def test_missing_learned_checkpoint_fails_before_any_method_output(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, missing_checkpoint=True)
    candidates = _candidates()
    manifest = _freeze(tmp_path, config, candidates)
    with pytest.raises(SimplificationProtocolError, match="policy_checkpoint_sha256"):
        run_simplification_shard(
            config=config,
            sample_manifest_path=manifest,
            candidates=candidates,
            shard_index=0,
            explorer=_Explorer(),
            cost_oracle=_CostOracle(),
            verifier=_Verifier(),
            sympy_comparator=_Comparator(),
            runtime=_runtime(),
        )
    assert not (tmp_path / "results").exists()


def test_default_sympy_comparator_is_independent_and_structurally_accounted() -> None:
    result = run_sympy_simplify(
        "Add(Symbol('x', real=True), Integer(0))",
        timeout_seconds=10.0,
    )
    assert result.status == "complete"
    assert result.result_srepr == "Symbol('x', real=True)"
    assert (result.result_ast_size, result.result_ast_depth) == (1, 0)
