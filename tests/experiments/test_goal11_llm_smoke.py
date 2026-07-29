"""Fully mocked tests for the Goal 11 external LLM reference panel (issue 11-3).

No test here requires an API key, network access, a provider SDK, a GPU, or any production
artifact. Every HTTP call goes through an injected transport.
"""

import json

import pytest

from geml.experiments.goal11.llm_reference import (
    ATTEMPTS_PER_MODEL,
    NEVER_GUESS_MODEL_IDS,
    PREFERRED_MODEL_ANCHORS,
    PROOF_OUTPUT_SCHEMA,
    PROOF_SYSTEM_PROMPT,
    PROVIDER_DEPENDENCY_NOTE,
    PROVIDER_ENDPOINTS,
    SR_OUTPUT_SCHEMA,
    SR_SYSTEM_PROMPT,
    AttemptStatus,
    CostGuard,
    HttpResponse,
    ModelSpec,
    PanelConfig,
    Provider,
    Track,
    VerificationStatus,
    build_request,
    extract_model_ids,
    freeze_task_selection,
    load_attempts,
    parse_response,
    plan_attempts,
    preflight,
    prompt_hash,
    redact,
    run_attempt,
    run_panel,
    summarize_panel,
)

_KEY = "sk-test-secret-value"
_ENVIRONMENT = {
    "OPENAI_API_KEY": _KEY,
    "ANTHROPIC_API_KEY": _KEY,
    "GEMINI_API_KEY": _KEY,
    "MOONSHOT_API_KEY": _KEY,
}


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _spec(provider: Provider = Provider.OPENAI, **overrides) -> ModelSpec:
    defaults = {
        "provider": provider,
        "model_id": PREFERRED_MODEL_ANCHORS[provider],
        "reasoning_field": "reasoning.effort",
        "reasoning_value": "high",
        "max_output_tokens": 256,
        "request_timeout_seconds": 5.0,
        "input_cost_per_million": 1.0,
        "output_cost_per_million": 2.0,
    }
    defaults.update(overrides)
    return ModelSpec(**defaults)


def _paid_config(**overrides) -> PanelConfig:
    defaults = {
        "models": (_spec(),),
        "proof_task_count": 2,
        "sr_task_count": 2,
        "max_retries": 1,
        "retry_backoff_seconds": 0.0,
        "allow_paid_calls": True,
        "spend_confirmed": True,
        "cost_guard": CostGuard(max_total_usd=100.0, max_per_model_usd=50.0),
    }
    defaults.update(overrides)
    return PanelConfig(**defaults)


def _openai_body(model: str = "gpt-5.6-sol", payload: dict | None = None) -> str:
    content = payload if payload is not None else {"expression": "x**2", "claimed_correct": True}
    return json.dumps(
        {
            "model": model,
            "output_text": json.dumps(content),
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "status": "completed",
        }
    )


def _transport(responses):
    """Return a transport that yields the supplied responses in order and records calls."""

    calls: list[tuple[str, dict, dict]] = []
    queue = list(responses)

    def transport(url, headers, body, timeout):
        del timeout
        decoded = json.loads(body.decode("utf-8")) if body else {}
        calls.append((url, dict(headers), decoded))
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


@pytest.fixture
def selection():
    """A tiny frozen selection: two proof tasks and two SR tasks."""

    return freeze_task_selection(
        proof_candidates=[("p1", "a"), ("p2", "b"), ("p3", "a")],
        sr_candidates=[("s1", "a"), ("s2", "b"), ("s3", "b")],
        config=_paid_config(),
        proof_source_manifest="proof-manifest",
        sr_source_manifest="sr-manifest",
    )


def _prompt(track: Track, task_id: str) -> str:
    return f"{track.value}:{task_id}"


# --------------------------------------------------------------------------------------
# Frozen workload
# --------------------------------------------------------------------------------------


def test_the_production_workload_is_exactly_two_hundred_attempts_per_model():
    assert ATTEMPTS_PER_MODEL == 200
    config = PanelConfig(
        models=(_spec(Provider.OPENAI), _spec(Provider.ANTHROPIC)),
    )
    proof = [(f"p{index}", "s") for index in range(150)]
    sr = [(f"s{index}", "s") for index in range(150)]
    frozen = freeze_task_selection(proof_candidates=proof, sr_candidates=sr, config=config)
    assert len(frozen.proof_task_ids) == 100
    assert len(frozen.sr_task_ids) == 100

    plan = plan_attempts(frozen, config.models)
    assert len(plan) == 2 * ATTEMPTS_PER_MODEL
    for spec in config.models:
        assert sum(1 for entry in plan if entry[0] is spec) == ATTEMPTS_PER_MODEL


def test_every_model_receives_the_identical_frozen_task_lists(selection):
    config = _paid_config(models=(_spec(Provider.OPENAI), _spec(Provider.ANTHROPIC)))
    plan = plan_attempts(selection, config.models)
    by_model: dict[str, list[str]] = {}
    for spec, track, task_id in plan:
        by_model.setdefault(spec.cell_id, []).append(f"{track.value}:{task_id}")
    assert len(by_model) == 2
    assert len(set(map(tuple, by_model.values()))) == 1


def test_task_ids_and_hashes_are_frozen_before_any_call(selection):
    assert selection.proof_ids_hash != selection.sr_ids_hash
    assert len(selection.proof_ids_hash) == 64
    repeated = freeze_task_selection(
        proof_candidates=[("p1", "a"), ("p2", "b"), ("p3", "a")],
        sr_candidates=[("s1", "a"), ("s2", "b"), ("s3", "b")],
        config=_paid_config(),
    )
    assert repeated.proof_task_ids == selection.proof_task_ids
    assert repeated.proof_ids_hash == selection.proof_ids_hash


def test_a_short_candidate_population_is_an_error():
    with pytest.raises(ValueError, match="needs exactly"):
        freeze_task_selection(
            proof_candidates=[("p1", "a")],
            sr_candidates=[("s1", "a"), ("s2", "a")],
            config=_paid_config(),
        )


# --------------------------------------------------------------------------------------
# Exact model identity
# --------------------------------------------------------------------------------------


def test_a_guessed_kimi_k3_identifier_is_refused():
    assert "kimi-k3" in NEVER_GUESS_MODEL_IDS
    with pytest.raises(ValueError, match="never-guess"):
        ModelSpec(provider=Provider.MOONSHOT, model_id="kimi-k3")


def test_preflight_refuses_a_model_absent_from_the_live_list():
    listing = HttpResponse(
        status=200, body=json.dumps({"data": [{"id": "gpt-4o"}, {"id": "gpt-5.5"}]})
    )
    (result,) = preflight(
        _paid_config(), list_transport=_transport([listing]), environment=_ENVIRONMENT
    )
    assert result.ok is False
    assert result.model_listed is False
    assert "not present" in result.detail
    assert result.listed_models_sample == ("gpt-4o", "gpt-5.5")


def test_preflight_accepts_only_the_exact_string():
    listing = HttpResponse(
        status=200,
        body=json.dumps({"data": [{"id": "gpt-5.6-sol"}, {"id": "gpt-5.6"}]}),
    )
    (result,) = preflight(
        _paid_config(), list_transport=_transport([listing]), environment=_ENVIRONMENT
    )
    assert result.ok is True
    assert result.anchor_matches is True


def test_preflight_without_a_transport_refuses_rather_than_assuming():
    (result,) = preflight(_paid_config(), list_transport=None, environment=_ENVIRONMENT)
    assert result.ok is False
    assert "could not be validated" in result.detail


def test_preflight_reports_a_missing_credential_without_leaking_anything():
    (result,) = preflight(_paid_config(), list_transport=_transport([]), environment={})
    assert result.ok is False
    assert result.credential_present is False
    assert "OPENAI_API_KEY" in result.detail
    assert _KEY not in result.detail


def test_a_silently_substituted_model_is_a_typed_mismatch_row(selection):
    transport = _transport([HttpResponse(status=200, body=_openai_body(model="gpt-5.5"))])
    record = run_attempt(
        spec=_spec(),
        track=Track.SYMBOLIC_REGRESSION,
        task_id="s1",
        user_prompt="observations",
        config=_paid_config(),
        transport=transport,
        environment=_ENVIRONMENT,
    )
    assert record.status is AttemptStatus.MODEL_IDENTITY_MISMATCH
    assert record.returned_model == "gpt-5.5"
    assert record.requested_model == "gpt-5.6-sol"
    assert "silent alias" in record.detail


@pytest.mark.parametrize("provider", list(Provider))
def test_model_list_extraction_works_for_every_provider(provider):
    if provider is Provider.GOOGLE:
        payload = {"models": [{"name": "models/gemini-3.6-flash"}]}
        expected = ("gemini-3.6-flash",)
    else:
        payload = {"data": [{"id": PREFERRED_MODEL_ANCHORS[provider]}]}
        expected = (PREFERRED_MODEL_ANCHORS[provider],)
    assert extract_model_ids(_spec(provider), payload) == expected


# --------------------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("provider", list(Provider))
def test_every_adapter_builds_a_request_and_parses_a_response(provider):
    spec = _spec(provider, reasoning_field="", reasoning_value="")
    url, body = build_request(
        spec,
        system_prompt=SR_SYSTEM_PROMPT,
        user_prompt="observations",
        schema=SR_OUTPUT_SCHEMA,
    )
    assert url.startswith("https://")
    assert PROVIDER_ENDPOINTS[provider].api_key_env.endswith("_API_KEY")
    if provider is Provider.GOOGLE:
        assert spec.model_id in url
        payload = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "{}"}]},
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": spec.model_id,
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3},
        }
    elif provider is Provider.ANTHROPIC:
        assert body["max_tokens"] == spec.max_output_tokens
        payload = {
            "model": spec.model_id,
            "content": [{"type": "text", "text": "{}"}],
            "usage": {"input_tokens": 7, "output_tokens": 3},
            "stop_reason": "end_turn",
        }
    elif provider is Provider.OPENAI:
        assert body["instructions"] == SR_SYSTEM_PROMPT
        payload = json.loads(_openai_body(spec.model_id, {}))
        payload["usage"] = {"input_tokens": 7, "output_tokens": 3}
        payload["output_text"] = "{}"
    else:
        assert body["messages"][0]["role"] == "system"
        payload = {
            "model": spec.model_id,
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
    parsed = parse_response(spec, payload)
    assert parsed.served_model == spec.model_id
    assert parsed.input_tokens == 7
    assert parsed.output_tokens == 3


def test_reasoning_settings_are_explicit_per_provider():
    openai_url, openai_body = build_request(
        _spec(Provider.OPENAI, reasoning_field="reasoning.effort", reasoning_value="high"),
        system_prompt=PROOF_SYSTEM_PROMPT,
        user_prompt="p",
        schema=PROOF_OUTPUT_SCHEMA,
    )
    del openai_url
    assert openai_body["reasoning"] == {"effort": "high"}

    _url, google_body = build_request(
        _spec(
            Provider.GOOGLE,
            reasoning_field="thinkingConfig.thinkingLevel",
            reasoning_value="high",
        ),
        system_prompt=PROOF_SYSTEM_PROMPT,
        user_prompt="p",
        schema=PROOF_OUTPUT_SCHEMA,
    )
    assert google_body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}


def test_prompts_forbid_tools_and_require_a_replayable_trace():
    for prompt in (PROOF_SYSTEM_PROMPT, SR_SYSTEM_PROMPT):
        assert "web browsing" in prompt
        assert "code execution" in prompt
    assert "Restating or echoing the target expression is not a proof" in PROOF_SYSTEM_PROMPT
    assert "replayable action trace" in PROOF_SYSTEM_PROMPT


def test_prompt_hash_binds_the_prompt_and_the_schema():
    first = prompt_hash(SR_SYSTEM_PROMPT, "u", SR_OUTPUT_SCHEMA)
    assert first == prompt_hash(SR_SYSTEM_PROMPT, "u", SR_OUTPUT_SCHEMA)
    assert first != prompt_hash(SR_SYSTEM_PROMPT, "u", PROOF_OUTPUT_SCHEMA)
    assert first != prompt_hash(SR_SYSTEM_PROMPT, "v", SR_OUTPUT_SCHEMA)


# --------------------------------------------------------------------------------------
# Typed failure capture
# --------------------------------------------------------------------------------------


def _run(response, **config_overrides):
    return run_attempt(
        spec=_spec(),
        track=Track.SYMBOLIC_REGRESSION,
        task_id="s1",
        user_prompt="observations",
        config=_paid_config(**config_overrides),
        transport=_transport([response]),
        environment=_ENVIRONMENT,
        sleep=lambda _seconds: None,
    )


def test_a_successful_attempt_records_usage_cost_and_claimed_correctness():
    record = _run(HttpResponse(status=200, body=_openai_body()))
    assert record.status is AttemptStatus.SUCCESS
    assert record.claimed_correct is True
    assert record.verification is VerificationStatus.NOT_RUN
    assert record.input_tokens == 100
    assert record.output_tokens == 50
    assert record.estimated_cost_usd == pytest.approx(100 / 1e6 + 100 / 1e6)
    assert record.parsed_response == {"expression": "x**2", "claimed_correct": True}
    assert record.raw_response


def test_a_refusal_is_its_own_status():
    body = json.loads(_openai_body())
    body["status"] = "refusal"
    record = _run(HttpResponse(status=200, body=json.dumps(body)))
    assert record.status is AttemptStatus.REFUSAL


def test_a_malformed_structured_output_is_a_parse_failure_with_the_raw_body_kept():
    body = json.loads(_openai_body())
    body["output_text"] = "not json at all"
    record = _run(HttpResponse(status=200, body=json.dumps(body)))
    assert record.status is AttemptStatus.PARSE_FAILURE
    assert "not json at all" in record.raw_response
    assert record.parsed_response is None


def test_a_non_json_body_is_a_parse_failure():
    record = _run(HttpResponse(status=200, body="<html>gateway</html>"))
    assert record.status is AttemptStatus.PARSE_FAILURE
    assert "<html>" in record.raw_response


def test_a_timeout_is_typed_and_retried_as_nested_evidence():
    record = run_attempt(
        spec=_spec(),
        track=Track.PROOF,
        task_id="p1",
        user_prompt="problem",
        config=_paid_config(max_retries=2),
        transport=_transport([TimeoutError("read timed out")]),
        environment=_ENVIRONMENT,
        sleep=lambda _seconds: None,
    )
    assert record.status is AttemptStatus.TIMEOUT
    assert len(record.retries) == 3
    assert {retry.index for retry in record.retries} == {1, 2, 3}


def test_rate_limiting_retries_then_records_one_attempt():
    transport = _transport(
        [
            HttpResponse(status=429, body="slow down"),
            HttpResponse(status=429, body="slow down"),
        ]
    )
    record = run_attempt(
        spec=_spec(),
        track=Track.PROOF,
        task_id="p1",
        user_prompt="problem",
        config=_paid_config(max_retries=1),
        transport=transport,
        environment=_ENVIRONMENT,
        sleep=lambda _seconds: None,
    )
    assert record.status is AttemptStatus.RATE_LIMITED
    assert len(record.retries) == 1
    assert record.retries[0].http_status == 429


def test_an_unsupported_parameter_is_detected_rather_than_ignored():
    record = _run(
        HttpResponse(
            status=400,
            body='{"error":{"message":"Unrecognized request argument supplied: seed"}}',
        )
    )
    assert record.status is AttemptStatus.UNSUPPORTED_PARAMETER


def test_a_server_error_is_an_api_error_row():
    record = _run(HttpResponse(status=500, body="internal"), max_retries=0)
    assert record.status is AttemptStatus.API_ERROR
    assert record.http_status == 500


# --------------------------------------------------------------------------------------
# Cost guard, credentials, spend confirmation
# --------------------------------------------------------------------------------------


def test_the_cost_guard_produces_a_typed_row_instead_of_a_silent_stop():
    record = run_attempt(
        spec=_spec(),
        track=Track.PROOF,
        task_id="p1",
        user_prompt="problem",
        config=_paid_config(cost_guard=CostGuard(max_total_usd=1.0)),
        transport=_transport([HttpResponse(status=200, body=_openai_body())]),
        spent_so_far_usd=1.5,
        environment=_ENVIRONMENT,
    )
    assert record.status is AttemptStatus.COST_GUARD_TRIPPED
    assert "spend guard" in record.detail


def test_paid_calls_require_an_explicit_spend_confirmation():
    with pytest.raises(ValueError, match="spend_confirmed"):
        PanelConfig(models=(_spec(),), allow_paid_calls=True, spend_confirmed=False)


def test_the_default_configuration_plans_attempts_without_calling_anything():
    record = run_attempt(
        spec=_spec(),
        track=Track.PROOF,
        task_id="p1",
        user_prompt="problem",
        config=PanelConfig(models=(_spec(),)),
        transport=_transport([HttpResponse(status=200, body=_openai_body())]),
        environment=_ENVIRONMENT,
    )
    assert record.status is AttemptStatus.NOT_ATTEMPTED
    assert record.estimated_cost_usd == 0.0
    assert record.prompt_hash


def test_a_missing_credential_never_leaks_and_is_an_api_error():
    record = run_attempt(
        spec=_spec(),
        track=Track.PROOF,
        task_id="p1",
        user_prompt="problem",
        config=_paid_config(),
        transport=_transport([HttpResponse(status=200, body=_openai_body())]),
        environment={},
    )
    assert record.status is AttemptStatus.API_ERROR
    assert _KEY not in record.detail


def test_secrets_are_redacted_from_stored_text():
    assert _KEY not in redact(f"error with {_KEY} inside", [_KEY])
    body = json.dumps({"error": f"bad key {_KEY}"})
    record = _run(HttpResponse(status=403, body=body))
    assert _KEY not in record.raw_response
    assert "***redacted***" in record.raw_response


def test_no_persisted_attempt_contains_a_credential(tmp_path, selection):
    records, _summary = run_panel(
        config=_paid_config(),
        config_hash="0" * 64,
        selection=selection,
        prompt_builder=_prompt,
        transport=_transport([HttpResponse(status=200, body=_openai_body())]),
        output_root=tmp_path,
        environment=_ENVIRONMENT,
        sleep=lambda _seconds: None,
    )
    assert records
    for path in tmp_path.rglob("*.json"):
        assert _KEY not in path.read_text()


# --------------------------------------------------------------------------------------
# Panel, resume, idempotency, denominators
# --------------------------------------------------------------------------------------


def test_the_panel_writes_one_row_per_frozen_attempt(tmp_path, selection):
    config = _paid_config()
    records, summary = run_panel(
        config=config,
        config_hash="0" * 64,
        selection=selection,
        prompt_builder=_prompt,
        transport=_transport([HttpResponse(status=200, body=_openai_body())]),
        output_root=tmp_path,
        environment=_ENVIRONMENT,
        sleep=lambda _seconds: None,
    )
    expected = config.proof_task_count + config.sr_task_count
    assert len(records) == expected
    assert summary.total_attempts == expected
    assert summary.attempts_per_model == expected
    assert len(load_attempts(tmp_path)) == expected


def test_resume_reloads_completed_attempts_without_recalling(tmp_path, selection):
    config = _paid_config()
    transport = _transport([HttpResponse(status=200, body=_openai_body())])
    arguments = {
        "config": config,
        "config_hash": "0" * 64,
        "selection": selection,
        "prompt_builder": _prompt,
        "transport": transport,
        "output_root": tmp_path,
        "environment": _ENVIRONMENT,
        "sleep": lambda _seconds: None,
    }
    run_panel(**arguments)
    first_calls = len(transport.calls)
    records, _summary = run_panel(**arguments)
    assert len(transport.calls) == first_calls
    assert len(records) == config.proof_task_count + config.sr_task_count


def test_retries_do_not_increase_the_attempt_denominator(tmp_path, selection):
    config = _paid_config(max_retries=2)
    transport = _transport(
        [
            HttpResponse(status=429, body="slow down"),
            HttpResponse(status=200, body=_openai_body()),
        ]
    )
    records, summary = run_panel(
        config=config,
        config_hash="0" * 64,
        selection=selection,
        prompt_builder=_prompt,
        transport=transport,
        output_root=tmp_path,
        environment=_ENVIRONMENT,
        sleep=lambda _seconds: None,
    )
    assert summary.total_attempts == config.proof_task_count + config.sr_task_count
    assert any(record.retries for record in records)
    assert sum(len(record.retries) for record in records) > 0


def test_claimed_and_verified_correctness_are_reported_separately(tmp_path, selection):
    records, summary = run_panel(
        config=_paid_config(),
        config_hash="0" * 64,
        selection=selection,
        prompt_builder=_prompt,
        transport=_transport([HttpResponse(status=200, body=_openai_body())]),
        output_root=tmp_path,
        environment=_ENVIRONMENT,
        sleep=lambda _seconds: None,
    )
    cell = records[0].cell_id
    assert summary.claimed_correct[cell] == len(records)
    assert summary.verified_correct[cell] == 0
    assert summary.external_reference_only is True
    assert "never enter a controlled" in summary.note


def test_a_disagreement_between_claimed_and_verified_is_representable():
    record = _run(HttpResponse(status=200, body=_openai_body()))
    disagreeing = record.model_copy(
        update={
            "verification": VerificationStatus.VERIFIED_INCORRECT,
            "verification_detail": "the verifier rejected the trace",
        }
    )
    assert disagreeing.claimed_correct is True
    assert disagreeing.verification is VerificationStatus.VERIFIED_INCORRECT


def test_only_a_successful_attempt_can_be_verified_correct():
    record = _run(HttpResponse(status=500, body="internal"), max_retries=0)
    with pytest.raises(ValueError, match="verifier-confirmed"):
        record.model_copy(
            update={"verification": VerificationStatus.VERIFIED_CORRECT}
        ).model_validate(
            record.model_copy(
                update={"verification": VerificationStatus.VERIFIED_CORRECT}
            ).model_dump()
        )


def test_summary_counts_every_status(tmp_path, selection):
    summary = summarize_panel(
        config_hash="0" * 64,
        selection=selection,
        config=_paid_config(),
        attempts=[
            _run(HttpResponse(status=200, body=_openai_body())),
            _run(HttpResponse(status=500, body="internal"), max_retries=0),
        ],
    )
    assert summary.status_counts == {"api_error": 1, "success": 1}


def test_the_provider_dependency_blocker_is_documented_in_code():
    assert "pyproject.toml" in PROVIDER_DEPENDENCY_NOTE
    assert "standard library" in PROVIDER_DEPENDENCY_NOTE
