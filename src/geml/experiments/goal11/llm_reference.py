"""Verifier-normalized frontier-LLM external reference panel (issue 11-3).

This is **external context, not a controlled baseline**. Nothing here may enter Gates G6-G11.

Phase A implements the whole machinery and makes no paid call: provider adapters over
standard-library HTTP, frozen task selection, raw and parsed result schemas, cost guards,
preflight, resume, and typed failure capture. Production requires credentials plus a separate
explicit spend confirmation.

Design commitments
------------------
* **Exact model IDs only.** Every model is validated against the provider's own model-list
  endpoint immediately before any paid call. A missing ID, an unrequested alias, a silent
  fallback, a provider substitution, or a guessed identifier is refused, not worked around.
* **No SDK dependency.** Adapters use ``urllib.request`` from the standard library. No entry
  in ``pyproject.toml`` is touched, because no assigned issue owns a ``[llm]`` optional
  dependency. See :data:`PROVIDER_DEPENDENCY_NOTE`.
* **Credentials never appear in a record.** Keys are read from environment variables at call
  time, are never logged, and never enter a persisted row or an error message.
* **Retries are nested evidence.** A retry lives inside the attempt that provoked it. It
  never creates a new attempt and never replaces the original failure, so the 200-attempt
  denominator per model is exact.
* **Claimed correctness and verified correctness are separate fields**, always.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

LLM_ATTEMPT_SCHEMA_VERSION = "geml-llm-attempt-v1"
LLM_PANEL_SCHEMA_VERSION = "geml-llm-panel-v1"
LLM_SELECTION_SCHEMA_VERSION = "geml-llm-task-selection-v1"

PRODUCTION_OUTPUT_ROOT = "outputs/final/goal11/llm_reference"

#: Frozen workload. Exactly 100 proof tasks and 100 SR tasks, identical for every model.
PROOF_TASK_COUNT = 100
SR_TASK_COUNT = 100
ATTEMPTS_PER_MODEL = PROOF_TASK_COUNT + SR_TASK_COUNT

PROVIDER_DEPENDENCY_NOTE = (
    "No assigned issue owns a [llm] optional-dependency edit in pyproject.toml, so these "
    "adapters use only urllib from the standard library. Each adapter is a small request "
    "builder plus a small response parser. If a provider ever requires behaviour that cannot "
    "stay small and auditable here (streaming, SSE reconnection, signed requests), stop and "
    "request an explicitly owned optional provider-dependency contract rather than growing a "
    "generic client framework inside this file."
)

_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class LLMReferenceError(ValueError):
    """A panel configuration, model identity, or workload input was invalid."""


class SpendConfirmationError(LLMReferenceError):
    """A paid run was requested without explicit spend confirmation."""


# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------


class Provider(StrEnum):
    """The four configured providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MOONSHOT = "moonshot"


class Track(StrEnum):
    """The two frozen tracks."""

    PROOF = "proof"
    SYMBOLIC_REGRESSION = "symbolic_regression"


class AttemptStatus(StrEnum):
    """Typed outcome of one attempt. Every attempt has exactly one."""

    SUCCESS = "success"
    REFUSAL = "refusal"
    PARSE_FAILURE = "parse_failure"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    API_ERROR = "api_error"
    MODEL_IDENTITY_MISMATCH = "model_identity_mismatch"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    COST_GUARD_TRIPPED = "cost_guard_tripped"
    NOT_ATTEMPTED = "not_attempted"


class VerificationStatus(StrEnum):
    """Verifier-confirmed correctness, always reported alongside claimed correctness."""

    VERIFIED_CORRECT = "verified_correct"
    VERIFIED_INCORRECT = "verified_incorrect"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    NOT_RUN = "not_run"


# --------------------------------------------------------------------------------------
# Provider endpoint facts
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderEndpoints:
    """Endpoint, header, and field facts for one provider.

    All values were read from the providers' own current documentation on 2026-07-26 and are
    re-validated at execution time by :func:`preflight`, because model IDs and API surfaces
    change.
    """

    provider: Provider
    generate_url: str
    models_url: str
    auth_header: str
    auth_format: str
    api_key_env: str
    extra_headers: tuple[tuple[str, str], ...] = ()
    documentation_url: str = ""


PROVIDER_ENDPOINTS: Mapping[Provider, ProviderEndpoints] = {
    Provider.OPENAI: ProviderEndpoints(
        provider=Provider.OPENAI,
        generate_url="https://api.openai.com/v1/responses",
        models_url="https://api.openai.com/v1/models",
        auth_header="Authorization",
        auth_format="Bearer {key}",
        api_key_env="OPENAI_API_KEY",
        documentation_url="https://developers.openai.com/api/reference/resources/responses",
    ),
    Provider.ANTHROPIC: ProviderEndpoints(
        provider=Provider.ANTHROPIC,
        generate_url="https://api.anthropic.com/v1/messages",
        models_url="https://api.anthropic.com/v1/models",
        auth_header="x-api-key",
        auth_format="{key}",
        api_key_env="ANTHROPIC_API_KEY",
        extra_headers=(("anthropic-version", "2023-06-01"),),
        documentation_url="https://platform.claude.com/docs/en/api/messages",
    ),
    Provider.GOOGLE: ProviderEndpoints(
        provider=Provider.GOOGLE,
        generate_url=(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        ),
        models_url="https://generativelanguage.googleapis.com/v1beta/models",
        auth_header="x-goog-api-key",
        auth_format="{key}",
        api_key_env="GEMINI_API_KEY",
        documentation_url="https://ai.google.dev/api/generate-content",
    ),
    Provider.MOONSHOT: ProviderEndpoints(
        provider=Provider.MOONSHOT,
        generate_url="https://api.moonshot.ai/v1/chat/completions",
        models_url="https://api.moonshot.ai/v1/models",
        auth_header="Authorization",
        auth_format="Bearer {key}",
        api_key_env="MOONSHOT_API_KEY",
        documentation_url="https://platform.kimi.ai/docs/api/chat",
    ),
}

#: Preferred anchors recorded from provider documentation on 2026-07-26. These are
#: *preferences*, never assumptions: :func:`preflight` refuses to proceed unless the exact
#: string is present in the provider's live model list. A newer pinned model may be
#: substituted only by editing the configuration, never by the code choosing one.
PREFERRED_MODEL_ANCHORS: Mapping[Provider, str] = {
    Provider.OPENAI: "gpt-5.6-sol",
    Provider.ANTHROPIC: "claude-opus-5",
    Provider.GOOGLE: "gemini-3.6-flash",
    Provider.MOONSHOT: "kimi-k2.5",
}

#: Model identifiers this code must never emit unless the live model list contains them.
#: ``kimi-k3`` is listed explicitly because the shared brief warns against inventing it; if
#: the provider genuinely exposes it, it must be selected through configuration and
#: confirmed by preflight, never guessed here.
NEVER_GUESS_MODEL_IDS: frozenset[str] = frozenset({"kimi-k3", "Kimi K3", "kimi-K3"})


# --------------------------------------------------------------------------------------
# Frozen prompts
# --------------------------------------------------------------------------------------

PROOF_SYSTEM_PROMPT = (
    "You are given an equational rewriting problem over a fixed real-valued expression "
    "grammar. Produce a complete, ordered, replayable action trace that transforms the "
    "source expression into the exact target expression. Each action must name a rewrite "
    "rule, a direction, an ordered occurrence path given as child-slot integers from the "
    "root, and the ordered arguments. Restating or echoing the target expression is not a "
    "proof: the target is already supplied in the problem. Do not use web browsing, external "
    "tools, retrieval, or code execution. Reply only with JSON matching the given schema."
)

SR_SYSTEM_PROMPT = (
    "You are given numeric observations of an unknown expression together with the ordered "
    "variable names and their real domains. Propose a single closed-form expression in the "
    "given grammar that reproduces the observations. Use only the listed operators and exact "
    "integer or rational constants. Do not use web browsing, external tools, retrieval, or "
    "code execution. Reply only with JSON matching the given schema."
)

PROOF_OUTPUT_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["actions", "claimed_correct"],
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rule_id", "direction", "occurrence_path", "arguments"],
                "properties": {
                    "rule_id": {"type": "string"},
                    "direction": {"type": "string", "enum": ["forward", "reverse"]},
                    "occurrence_path": {"type": "array", "items": {"type": "integer"}},
                    "arguments": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "claimed_correct": {"type": "boolean"},
    },
}

SR_OUTPUT_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["expression", "claimed_correct"],
    "properties": {
        "expression": {"type": "string"},
        "claimed_correct": {"type": "boolean"},
    },
}


def prompt_hash(system_prompt: str, user_prompt: str, schema: Mapping[str, object]) -> str:
    """Return a stable digest binding the system prompt, user prompt, and output schema."""

    hasher = hashlib.sha256()
    hasher.update(b"geml-llm-prompt-v1\0")
    for part in (
        system_prompt,
        user_prompt,
        json.dumps(schema, sort_keys=True, separators=(",", ":")),
    ):
        encoded = part.encode("utf-8")
        hasher.update(f"{len(encoded)}:".encode("ascii"))
        hasher.update(encoded)
        hasher.update(b"\0")
    return hasher.hexdigest()


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """One configured provider/model pair."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    provider: Provider
    model_id: str = Field(min_length=1)
    reasoning_field: str = ""
    reasoning_value: str = ""
    max_output_tokens: int = Field(default=8192, ge=1)
    temperature: float | None = None
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)
    determinism_note: str = ""

    @model_validator(mode="after")
    def _reject_guessed_identifiers(self) -> "ModelSpec":
        if self.model_id in NEVER_GUESS_MODEL_IDS:
            raise ValueError(
                f"model id {self.model_id!r} is on the never-guess list; it may only be used "
                "after the provider's live model list confirms the exact string"
            )
        return self

    @property
    def cell_id(self) -> str:
        """Return the stable provider/model cell identifier."""

        return f"{self.provider.value}:{self.model_id}"


class CostGuard(BaseModel):
    """Hard spend ceilings, checked before and during a run."""

    model_config = _FROZEN

    max_total_usd: float = Field(default=0.0, ge=0.0)
    max_per_model_usd: float = Field(default=0.0, ge=0.0)
    max_attempts_per_model: int = Field(default=ATTEMPTS_PER_MODEL, ge=1)


class PanelConfig(BaseModel):
    """Complete configuration for the external reference panel."""

    model_config = _FROZEN

    schema_version: str = "geml-goal11-llm-config-v1"
    output_root: str = PRODUCTION_OUTPUT_ROOT
    models: tuple[ModelSpec, ...] = ()
    proof_task_count: int = Field(default=PROOF_TASK_COUNT, ge=1)
    sr_task_count: int = Field(default=SR_TASK_COUNT, ge=1)
    selection_seed: int = 20260726
    max_concurrency: int = Field(default=2, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_seconds: float = Field(default=2.0, ge=0.0)
    cost_guard: CostGuard = CostGuard()
    allow_paid_calls: bool = False
    spend_confirmed: bool = False
    resume: bool = True

    @model_validator(mode="after")
    def _paid_calls_require_confirmation(self) -> "PanelConfig":
        if self.allow_paid_calls and not self.spend_confirmed:
            raise ValueError(
                "allow_paid_calls requires spend_confirmed; production spend needs a separate "
                "explicit confirmation"
            )
        return self


def load_config(path: str | Path) -> tuple[PanelConfig, str]:
    """Load a YAML configuration and return it with the SHA-256 of the exact file bytes."""

    raw = Path(path).read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8")) or {}
    return PanelConfig.model_validate(payload), hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------------------
# Frozen task selection
# --------------------------------------------------------------------------------------


class TaskSelection(BaseModel):
    """The two frozen ordered ID lists, shared by every provider and model."""

    model_config = _FROZEN

    schema_version: str = LLM_SELECTION_SCHEMA_VERSION
    proof_task_ids: tuple[str, ...]
    sr_task_ids: tuple[str, ...]
    proof_ids_hash: str
    sr_ids_hash: str
    selection_seed: int
    proof_source_manifest: str = ""
    sr_source_manifest: str = ""
    strata_note: str = ""

    @model_validator(mode="after")
    def _lists_are_unique(self) -> "TaskSelection":
        for name, ids in (
            ("proof_task_ids", self.proof_task_ids),
            ("sr_task_ids", self.sr_task_ids),
        ):
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name} contains duplicates")
        return self

    def ids_for(self, track: Track) -> tuple[str, ...]:
        """Return the frozen ordered id list for one track."""

        return self.proof_task_ids if track is Track.PROOF else self.sr_task_ids


def _ids_hash(track: Track, ids: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"geml-llm-task-ids-v1\0{track.value}\0".encode())
    for identifier in ids:
        encoded = identifier.encode("utf-8")
        hasher.update(f"{len(encoded)}:".encode("ascii"))
        hasher.update(encoded)
        hasher.update(b"\0")
    return hasher.hexdigest()


def freeze_task_selection(
    *,
    proof_candidates: Sequence[tuple[str, str]],
    sr_candidates: Sequence[tuple[str, str]],
    config: PanelConfig,
    proof_source_manifest: str = "",
    sr_source_manifest: str = "",
) -> TaskSelection:
    """Freeze one stratified selection per track, shared by every provider and model.

    Each candidate is a ``(task_id, stratum)`` pair. Selection is round-robin over strata,
    ordered by task id inside each stratum, so it is a pure function of the candidate
    population and cannot be nudged after seeing any response. Both ordered lists and their
    hashes are persisted before any call is made.
    """

    def _select(candidates: Sequence[tuple[str, str]], count: int, track: Track) -> tuple[str, ...]:
        if len(candidates) < count:
            raise LLMReferenceError(
                f"{track.value} track needs exactly {count} tasks but only "
                f"{len(candidates)} candidates were supplied"
            )
        strata: dict[str, list[str]] = {}
        for task_id, stratum in candidates:
            strata.setdefault(stratum, []).append(task_id)
        for bucket in strata.values():
            bucket.sort()
        selected: list[str] = []
        keys = sorted(strata)
        while len(selected) < count:
            progressed = False
            for key in keys:
                if not strata[key]:
                    continue
                selected.append(strata[key].pop(0))
                progressed = True
                if len(selected) == count:
                    break
            if not progressed:  # pragma: no cover - guarded by the length check above
                break
        return tuple(selected)

    proof_ids = _select(proof_candidates, config.proof_task_count, Track.PROOF)
    sr_ids = _select(sr_candidates, config.sr_task_count, Track.SYMBOLIC_REGRESSION)
    return TaskSelection(
        proof_task_ids=proof_ids,
        sr_task_ids=sr_ids,
        proof_ids_hash=_ids_hash(Track.PROOF, proof_ids),
        sr_ids_hash=_ids_hash(Track.SYMBOLIC_REGRESSION, sr_ids),
        selection_seed=config.selection_seed,
        proof_source_manifest=proof_source_manifest,
        sr_source_manifest=sr_source_manifest,
        strata_note="round-robin over declared strata, ordered by task id within each stratum",
    )


# --------------------------------------------------------------------------------------
# HTTP transport
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """A raw HTTP response, kept exactly as received."""

    status: int
    body: str
    headers: tuple[tuple[str, str], ...] = ()


#: The transport seam. Tests inject a callable with this signature; production uses
#: :func:`urllib_transport`. Keeping this one function wide means no mocked test ever needs
#: network access, keys, or a provider SDK.
Transport = Callable[[str, Mapping[str, str], bytes, float], HttpResponse]


def urllib_transport(
    url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> HttpResponse:  # pragma: no cover - exercised only against live providers
    """Perform one POST with the standard library. Never used by the test suite."""

    request = urllib.request.Request(url, data=body, method="POST")
    for name, value in headers.items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status=response.status,
                body=response.read().decode("utf-8", errors="replace"),
                headers=tuple(response.headers.items()),
            )
    except urllib.error.HTTPError as error:
        return HttpResponse(
            status=error.code,
            body=error.read().decode("utf-8", errors="replace"),
            headers=tuple(error.headers.items()) if error.headers else (),
        )


def urllib_get_transport(
    url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> HttpResponse:  # pragma: no cover - exercised only against live providers
    """Perform one GET with the standard library, used for model listing."""

    del body
    request = urllib.request.Request(url, method="GET")
    for name, value in headers.items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status=response.status,
                body=response.read().decode("utf-8", errors="replace"),
                headers=tuple(response.headers.items()),
            )
    except urllib.error.HTTPError as error:
        return HttpResponse(
            status=error.code,
            body=error.read().decode("utf-8", errors="replace"),
            headers=tuple(error.headers.items()) if error.headers else (),
        )


def _api_key(spec: ModelSpec, environment: Mapping[str, str] | None = None) -> str:
    """Read the provider key from the environment. The value is never logged or persisted."""

    source = os.environ if environment is None else environment
    endpoints = PROVIDER_ENDPOINTS[spec.provider]
    key = source.get(endpoints.api_key_env, "")
    if not key:
        raise LLMReferenceError(
            f"environment variable {endpoints.api_key_env} is not set for "
            f"{spec.provider.value}; credentials come only from the environment or secret "
            "infrastructure and are never committed"
        )
    return key


def _headers(spec: ModelSpec, key: str) -> dict[str, str]:
    endpoints = PROVIDER_ENDPOINTS[spec.provider]
    headers = {
        "Content-Type": "application/json",
        endpoints.auth_header: endpoints.auth_format.format(key=key),
    }
    headers.update(dict(endpoints.extra_headers))
    return headers


def redact(text: str, secrets: Sequence[str]) -> str:
    """Remove any supplied secret from a string before it is stored or reported."""

    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "***redacted***")
    return cleaned


# --------------------------------------------------------------------------------------
# Provider adapters
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    """The provider-agnostic view of one response."""

    text: str
    served_model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


def build_request(
    spec: ModelSpec,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    """Return the request URL and JSON body for one provider.

    Each branch is a direct transcription of that provider's documented request shape. Only
    fields the provider documents are sent, because an unsupported parameter that is silently
    ignored would quietly break the fairness contract.
    """

    endpoints = PROVIDER_ENDPOINTS[spec.provider]

    if spec.provider is Provider.OPENAI:
        body: dict[str, object] = {
            "model": spec.model_id,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": spec.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "geml_reference",
                    "schema": dict(schema),
                    "strict": True,
                }
            },
        }
        if spec.reasoning_field == "reasoning.effort" and spec.reasoning_value:
            body["reasoning"] = {"effort": spec.reasoning_value}
        if spec.temperature is not None:
            body["temperature"] = spec.temperature
        return endpoints.generate_url, body

    if spec.provider is Provider.ANTHROPIC:
        body = {
            "model": spec.model_id,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": spec.max_output_tokens,
            "output_config": {
                "format": {"type": "json_schema", "schema": dict(schema)},
            },
        }
        if spec.reasoning_field == "output_config.effort" and spec.reasoning_value:
            body["output_config"]["effort"] = spec.reasoning_value  # type: ignore[index]
        elif spec.reasoning_field == "thinking.type" and spec.reasoning_value:
            body["thinking"] = {"type": spec.reasoning_value}
        if spec.temperature is not None:
            body["temperature"] = spec.temperature
        return endpoints.generate_url, body

    if spec.provider is Provider.GOOGLE:
        generation_config: dict[str, object] = {
            "maxOutputTokens": spec.max_output_tokens,
            "responseMimeType": "application/json",
            "responseSchema": dict(schema),
        }
        if spec.reasoning_field == "thinkingConfig.thinkingLevel" and spec.reasoning_value:
            generation_config["thinkingConfig"] = {"thinkingLevel": spec.reasoning_value}
        elif spec.reasoning_field == "thinkingConfig.thinkingBudget" and spec.reasoning_value:
            generation_config["thinkingConfig"] = {"thinkingBudget": int(spec.reasoning_value)}
        if spec.temperature is not None:
            generation_config["temperature"] = spec.temperature
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        }
        return endpoints.generate_url.format(model=spec.model_id), body

    body = {
        "model": spec.model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": spec.max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "geml_reference",
                "schema": dict(schema),
                "strict": True,
            },
        },
    }
    if spec.reasoning_field == "reasoning_effort" and spec.reasoning_value:
        body["reasoning_effort"] = spec.reasoning_value
    elif spec.reasoning_field == "thinking.type" and spec.reasoning_value:
        body["thinking"] = {"type": spec.reasoning_value}
    if spec.temperature is not None:
        body["temperature"] = spec.temperature
    return endpoints.generate_url, body


def parse_response(spec: ModelSpec, payload: Mapping[str, object]) -> ParsedResponse:
    """Extract text, served model, usage, and finish reason from one provider payload."""

    if spec.provider is Provider.OPENAI:
        text = str(payload.get("output_text", ""))
        if not text:
            chunks: list[str] = []
            for item in payload.get("output", []) or []:
                for content in (item or {}).get("content", []) or []:
                    if content.get("type") in {"output_text", "text"}:
                        chunks.append(str(content.get("text", "")))
            text = "".join(chunks)
        usage = payload.get("usage", {}) or {}
        return ParsedResponse(
            text=text,
            served_model=str(payload.get("model", "")),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            finish_reason=str(payload.get("status", "")),
        )

    if spec.provider is Provider.ANTHROPIC:
        chunks = [
            str(block.get("text", ""))
            for block in payload.get("content", []) or []
            if block.get("type") == "text"
        ]
        usage = payload.get("usage", {}) or {}
        return ParsedResponse(
            text="".join(chunks),
            served_model=str(payload.get("model", "")),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            finish_reason=str(payload.get("stop_reason", "")),
        )

    if spec.provider is Provider.GOOGLE:
        candidates = payload.get("candidates", []) or []
        first = candidates[0] if candidates else {}
        parts = ((first or {}).get("content", {}) or {}).get("parts", []) or []
        usage = payload.get("usageMetadata", {}) or {}
        return ParsedResponse(
            text="".join(str(part.get("text", "")) for part in parts),
            served_model=str(payload.get("modelVersion", "")),
            input_tokens=int(usage.get("promptTokenCount", 0) or 0),
            output_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
            finish_reason=str((first or {}).get("finishReason", "")),
        )

    choices = payload.get("choices", []) or []
    first = choices[0] if choices else {}
    usage = payload.get("usage", {}) or {}
    return ParsedResponse(
        text=str(((first or {}).get("message", {}) or {}).get("content", "")),
        served_model=str(payload.get("model", "")),
        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("completion_tokens", 0) or 0),
        finish_reason=str((first or {}).get("finish_reason", "")),
    )


def extract_model_ids(spec: ModelSpec, payload: Mapping[str, object]) -> tuple[str, ...]:
    """Extract the exact model identifiers from one provider's model-list payload."""

    if spec.provider is Provider.GOOGLE:
        return tuple(
            str(item.get("name", "")).removeprefix("models/")
            for item in payload.get("models", []) or []
        )
    return tuple(str(item.get("id", "")) for item in payload.get("data", []) or [])


# --------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------


class PreflightResult(BaseModel):
    """Per-model preflight verdict. No paid call happens unless every model is ``ok``."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    provider: Provider
    model_id: str
    ok: bool
    credential_present: bool
    model_listed: bool
    listed_models_sample: tuple[str, ...] = ()
    detail: str = ""
    preferred_anchor: str = ""
    anchor_matches: bool = False


def preflight(
    config: PanelConfig,
    *,
    list_transport: Transport | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[PreflightResult, ...]:
    """Validate credentials and exact model availability without any paid generation call.

    A model is usable only when the provider's own live model list contains the exact
    requested string. An alias, a near match, a differently-cased variant, or an absent id is
    a refusal. Listing endpoints are free; this never invokes generation.
    """

    results: list[PreflightResult] = []
    for spec in config.models:
        endpoints = PROVIDER_ENDPOINTS[spec.provider]
        anchor = PREFERRED_MODEL_ANCHORS.get(spec.provider, "")
        try:
            key = _api_key(spec, environment)
            credential_present = True
        except LLMReferenceError as error:
            results.append(
                PreflightResult(
                    provider=spec.provider,
                    model_id=spec.model_id,
                    ok=False,
                    credential_present=False,
                    model_listed=False,
                    detail=str(error),
                    preferred_anchor=anchor,
                    anchor_matches=spec.model_id == anchor,
                )
            )
            continue

        if list_transport is None:
            results.append(
                PreflightResult(
                    provider=spec.provider,
                    model_id=spec.model_id,
                    ok=False,
                    credential_present=credential_present,
                    model_listed=False,
                    detail=(
                        "no model-list transport was supplied, so the exact model id could "
                        "not be validated; refusing to treat the model as available"
                    ),
                    preferred_anchor=anchor,
                    anchor_matches=spec.model_id == anchor,
                )
            )
            continue

        response = list_transport(
            endpoints.models_url, _headers(spec, key), b"", spec.request_timeout_seconds
        )
        if response.status != 200:
            results.append(
                PreflightResult(
                    provider=spec.provider,
                    model_id=spec.model_id,
                    ok=False,
                    credential_present=credential_present,
                    model_listed=False,
                    detail=f"model list returned HTTP {response.status}",
                    preferred_anchor=anchor,
                    anchor_matches=spec.model_id == anchor,
                )
            )
            continue

        try:
            payload = json.loads(response.body)
        except json.JSONDecodeError as error:
            results.append(
                PreflightResult(
                    provider=spec.provider,
                    model_id=spec.model_id,
                    ok=False,
                    credential_present=credential_present,
                    model_listed=False,
                    detail=f"model list was not JSON: {error}",
                    preferred_anchor=anchor,
                    anchor_matches=spec.model_id == anchor,
                )
            )
            continue

        listed = extract_model_ids(spec, payload)
        exact = spec.model_id in listed
        results.append(
            PreflightResult(
                provider=spec.provider,
                model_id=spec.model_id,
                ok=exact,
                credential_present=credential_present,
                model_listed=exact,
                listed_models_sample=tuple(sorted(listed)[:12]),
                detail=(
                    ""
                    if exact
                    else (
                        f"exact model id {spec.model_id!r} is not present in the provider's "
                        "live model list; refusing an alias, a fallback, or a substitution"
                    )
                ),
                preferred_anchor=anchor,
                anchor_matches=spec.model_id == anchor,
            )
        )
    return tuple(results)


# --------------------------------------------------------------------------------------
# Attempt records
# --------------------------------------------------------------------------------------


class RetryRecord(BaseModel):
    """One retry, nested under the attempt that provoked it."""

    model_config = _FROZEN

    index: int = Field(ge=1)
    status: AttemptStatus
    http_status: int | None = None
    latency_seconds: float = Field(ge=0.0)
    detail: str = ""


class AttemptRecord(BaseModel):
    """One frozen (model, task) attempt. Exactly 200 of these exist per configured model."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    schema_version: str = LLM_ATTEMPT_SCHEMA_VERSION
    provider: Provider
    requested_model: str
    returned_model: str = ""
    access_date: str
    track: Track
    task_id: str
    task_hash: str
    prompt_hash: str
    system_prompt: str
    user_prompt: str
    reasoning_field: str = ""
    reasoning_value: str = ""
    max_output_tokens: int = Field(ge=0)
    temperature: float | None = None
    request_timeout_seconds: float = Field(ge=0.0)
    determinism_note: str = ""
    http_status: int | None = None
    raw_response: str = ""
    parsed_response: dict[str, object] | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    retries: tuple[RetryRecord, ...] = ()
    status: AttemptStatus
    claimed_correct: bool | None = None
    verification: VerificationStatus = VerificationStatus.NOT_RUN
    verification_detail: str = ""
    detail: str = ""

    @model_validator(mode="after")
    def _identity_is_explicit(self) -> "AttemptRecord":
        if (
            self.status is AttemptStatus.SUCCESS
            and self.returned_model
            and self.returned_model != self.requested_model
        ):
            raise ValueError(
                "a successful attempt must have been served by the exact requested model; "
                f"requested {self.requested_model!r} but the provider returned "
                f"{self.returned_model!r}"
            )
        if (
            self.verification is VerificationStatus.VERIFIED_CORRECT
            and self.status is not AttemptStatus.SUCCESS
        ):
            raise ValueError("only a successful attempt can be verifier-confirmed correct")
        return self

    @property
    def cell_id(self) -> str:
        """Return the provider/model cell this attempt belongs to."""

        return f"{self.provider.value}:{self.requested_model}"


class PanelSummary(BaseModel):
    """Denominators for the whole panel, with claimed and verified always paired."""

    model_config = _FROZEN

    schema_version: str = LLM_PANEL_SCHEMA_VERSION
    config_hash: str
    selection: TaskSelection
    models: tuple[str, ...]
    attempts_per_model: int = Field(ge=0)
    total_attempts: int = Field(ge=0)
    status_counts: dict[str, int]
    claimed_correct: dict[str, int]
    verified_correct: dict[str, int]
    estimated_cost_usd: float = Field(ge=0.0)
    paid_calls_made: bool = False
    external_reference_only: bool = True
    note: str = (
        "External reference panel. These rows are context and never enter a controlled "
        "Gate G6-G11 comparison."
    )


# --------------------------------------------------------------------------------------
# Running one attempt
# --------------------------------------------------------------------------------------

_RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def _task_hash(track: Track, task_id: str, payload: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"geml-llm-task-v1\0")
    for part in (track.value, task_id, payload):
        encoded = part.encode("utf-8")
        hasher.update(f"{len(encoded)}:".encode("ascii"))
        hasher.update(encoded)
        hasher.update(b"\0")
    return hasher.hexdigest()


def estimate_cost(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    """Return the estimated USD cost of one call from the configured per-million rates."""

    return (
        input_tokens * spec.input_cost_per_million / 1_000_000.0
        + output_tokens * spec.output_cost_per_million / 1_000_000.0
    )


def run_attempt(
    *,
    spec: ModelSpec,
    track: Track,
    task_id: str,
    user_prompt: str,
    config: PanelConfig,
    transport: Transport,
    spent_so_far_usd: float = 0.0,
    environment: Mapping[str, str] | None = None,
    access_date: str | None = None,
    sleep: Callable[[float], None] | None = None,
) -> AttemptRecord:
    """Run one frozen (model, task) attempt, returning a row whatever happens.

    Retries are recorded inside this single attempt. The function never returns more than one
    record and never raises for a provider-side problem: an unreachable provider, a refusal, a
    malformed body, or a model mismatch each produce a typed row.
    """

    system_prompt = PROOF_SYSTEM_PROMPT if track is Track.PROOF else SR_SYSTEM_PROMPT
    schema = PROOF_OUTPUT_SCHEMA if track is Track.PROOF else SR_OUTPUT_SCHEMA
    digest = prompt_hash(system_prompt, user_prompt, schema)
    stamp = access_date or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    waiter = sleep if sleep is not None else time.sleep

    base = {
        "provider": spec.provider,
        "requested_model": spec.model_id,
        "access_date": stamp,
        "track": track,
        "task_id": task_id,
        "task_hash": _task_hash(track, task_id, user_prompt),
        "prompt_hash": digest,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "reasoning_field": spec.reasoning_field,
        "reasoning_value": spec.reasoning_value,
        "max_output_tokens": spec.max_output_tokens,
        "temperature": spec.temperature,
        "request_timeout_seconds": spec.request_timeout_seconds,
        "determinism_note": spec.determinism_note,
    }

    guard = config.cost_guard
    if guard.max_total_usd and spent_so_far_usd >= guard.max_total_usd:
        return AttemptRecord(
            **base,
            status=AttemptStatus.COST_GUARD_TRIPPED,
            detail=(
                f"spend guard reached: {spent_so_far_usd:.4f} USD already spent against a "
                f"ceiling of {guard.max_total_usd:.4f} USD"
            ),
        )

    if not config.allow_paid_calls:
        return AttemptRecord(
            **base,
            status=AttemptStatus.NOT_ATTEMPTED,
            detail=(
                "allow_paid_calls is false; the attempt was planned and recorded but no paid "
                "call was made"
            ),
        )
    if not config.spend_confirmed:  # pragma: no cover - blocked by PanelConfig validation
        raise SpendConfirmationError("paid calls require an explicit spend confirmation")

    try:
        key = _api_key(spec, environment)
    except LLMReferenceError as error:
        return AttemptRecord(**base, status=AttemptStatus.API_ERROR, detail=str(error))

    url, body = build_request(
        spec, system_prompt=system_prompt, user_prompt=user_prompt, schema=schema
    )
    encoded = json.dumps(body, sort_keys=True).encode("utf-8")
    headers = _headers(spec, key)
    retries: list[RetryRecord] = []
    response: HttpResponse | None = None
    latency = 0.0

    for attempt_index in range(config.max_retries + 1):
        started = time.perf_counter()
        try:
            response = transport(url, headers, encoded, spec.request_timeout_seconds)
            latency = time.perf_counter() - started
        except TimeoutError as error:
            latency = time.perf_counter() - started
            retries.append(
                RetryRecord(
                    index=attempt_index + 1,
                    status=AttemptStatus.TIMEOUT,
                    latency_seconds=latency,
                    detail=redact(str(error), [key]),
                )
            )
            if attempt_index >= config.max_retries:
                return AttemptRecord(
                    **base,
                    status=AttemptStatus.TIMEOUT,
                    latency_seconds=latency,
                    retries=tuple(retries),
                    detail="request timed out after every retry",
                )
            waiter(config.retry_backoff_seconds * (attempt_index + 1))
            continue
        except Exception as error:
            latency = time.perf_counter() - started
            return AttemptRecord(
                **base,
                status=AttemptStatus.API_ERROR,
                latency_seconds=latency,
                retries=tuple(retries),
                detail=redact(f"{type(error).__name__}: {error}", [key]),
            )

        if response.status in _RETRYABLE_STATUSES and attempt_index < config.max_retries:
            retries.append(
                RetryRecord(
                    index=attempt_index + 1,
                    status=(
                        AttemptStatus.RATE_LIMITED
                        if response.status == 429
                        else AttemptStatus.API_ERROR
                    ),
                    http_status=response.status,
                    latency_seconds=latency,
                    detail=redact(response.body[:500], [key]),
                )
            )
            waiter(config.retry_backoff_seconds * (attempt_index + 1))
            continue
        break

    assert response is not None
    raw = redact(response.body, [key])

    if response.status == 429:
        return AttemptRecord(
            **base,
            status=AttemptStatus.RATE_LIMITED,
            http_status=response.status,
            raw_response=raw,
            latency_seconds=latency,
            retries=tuple(retries),
            detail="rate limited after every retry",
        )
    if response.status == 400 and "unrecognized" in raw.lower():
        return AttemptRecord(
            **base,
            status=AttemptStatus.UNSUPPORTED_PARAMETER,
            http_status=response.status,
            raw_response=raw,
            latency_seconds=latency,
            retries=tuple(retries),
            detail="the provider rejected a request parameter",
        )
    if response.status != 200:
        return AttemptRecord(
            **base,
            status=AttemptStatus.API_ERROR,
            http_status=response.status,
            raw_response=raw,
            latency_seconds=latency,
            retries=tuple(retries),
            detail=f"HTTP {response.status}",
        )

    try:
        payload = json.loads(response.body)
    except json.JSONDecodeError as error:
        return AttemptRecord(
            **base,
            status=AttemptStatus.PARSE_FAILURE,
            http_status=response.status,
            raw_response=raw,
            latency_seconds=latency,
            retries=tuple(retries),
            detail=f"response body was not JSON: {error}",
        )

    parsed = parse_response(spec, payload)
    cost = estimate_cost(spec, parsed.input_tokens, parsed.output_tokens)

    if parsed.served_model and parsed.served_model != spec.model_id:
        return AttemptRecord(
            **base,
            returned_model=parsed.served_model,
            status=AttemptStatus.MODEL_IDENTITY_MISMATCH,
            http_status=response.status,
            raw_response=raw,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            retries=tuple(retries),
            detail=(
                f"requested {spec.model_id!r} but the provider served "
                f"{parsed.served_model!r}; refusing a silent alias or fallback"
            ),
        )

    if _looks_like_refusal(parsed):
        return AttemptRecord(
            **base,
            returned_model=parsed.served_model,
            status=AttemptStatus.REFUSAL,
            http_status=response.status,
            raw_response=raw,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            retries=tuple(retries),
            detail=f"finish reason {parsed.finish_reason!r}",
        )

    try:
        structured = json.loads(parsed.text)
        if not isinstance(structured, dict):
            raise TypeError("structured output was not a JSON object")
    except (json.JSONDecodeError, TypeError) as error:
        return AttemptRecord(
            **base,
            returned_model=parsed.served_model,
            status=AttemptStatus.PARSE_FAILURE,
            http_status=response.status,
            raw_response=raw,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            latency_seconds=latency,
            estimated_cost_usd=cost,
            retries=tuple(retries),
            detail=f"structured output did not match the frozen schema: {error}",
        )

    claimed = structured.get("claimed_correct")
    return AttemptRecord(
        **base,
        returned_model=parsed.served_model,
        status=AttemptStatus.SUCCESS,
        http_status=response.status,
        raw_response=raw,
        parsed_response=structured,
        input_tokens=parsed.input_tokens,
        output_tokens=parsed.output_tokens,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        retries=tuple(retries),
        claimed_correct=bool(claimed) if isinstance(claimed, bool) else None,
        verification=VerificationStatus.NOT_RUN,
        verification_detail="verification runs separately; no output is repaired first",
    )


def _looks_like_refusal(parsed: ParsedResponse) -> bool:
    return parsed.finish_reason.lower() in {"refusal", "content_filter", "safety", "blocked"}


# --------------------------------------------------------------------------------------
# Panel planning, persistence, resume
# --------------------------------------------------------------------------------------


def plan_attempts(
    selection: TaskSelection, models: Sequence[ModelSpec]
) -> tuple[tuple[ModelSpec, Track, str], ...]:
    """Return the complete frozen attempt plan.

    Exactly ``len(proof_task_ids) + len(sr_task_ids)`` attempts per model, in a fixed order,
    with the same task ids for every provider and model.
    """

    plan: list[tuple[ModelSpec, Track, str]] = []
    for spec in models:
        for task_id in selection.proof_task_ids:
            plan.append((spec, Track.PROOF, task_id))
        for task_id in selection.sr_task_ids:
            plan.append((spec, Track.SYMBOLIC_REGRESSION, task_id))
    return tuple(plan)


def attempt_path(root: str | Path, spec: ModelSpec, track: Track, task_id: str) -> Path:
    """Return the atomic per-attempt path used for persistence and resume."""

    safe_model = spec.model_id.replace("/", "_")
    return Path(root) / spec.provider.value / safe_model / track.value / f"{task_id}.json"


def write_attempt(record: AttemptRecord, root: str | Path) -> Path:
    """Write one attempt atomically, including its raw response."""

    spec_path = (
        Path(root)
        / record.provider.value
        / record.requested_model.replace("/", "_")
        / record.track.value
        / f"{record.task_id}.json"
    )
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    temporary = spec_path.with_suffix(".json.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(spec_path)
    return spec_path


def load_attempts(root: str | Path) -> tuple[AttemptRecord, ...]:
    """Load every persisted attempt under ``root`` in a stable order."""

    return tuple(
        AttemptRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(Path(root).rglob("*.json"))
    )


def summarize_panel(
    *,
    config_hash: str,
    selection: TaskSelection,
    config: PanelConfig,
    attempts: Sequence[AttemptRecord],
) -> PanelSummary:
    """Summarise the panel with claimed and verified correctness always paired."""

    status_counts: dict[str, int] = {}
    claimed: dict[str, int] = {}
    verified: dict[str, int] = {}
    for record in attempts:
        status_counts[record.status.value] = status_counts.get(record.status.value, 0) + 1
        cell = record.cell_id
        claimed.setdefault(cell, 0)
        verified.setdefault(cell, 0)
        if record.claimed_correct:
            claimed[cell] += 1
        if record.verification is VerificationStatus.VERIFIED_CORRECT:
            verified[cell] += 1

    return PanelSummary(
        config_hash=config_hash,
        selection=selection,
        models=tuple(sorted(spec.cell_id for spec in config.models)),
        attempts_per_model=config.proof_task_count + config.sr_task_count,
        total_attempts=len(attempts),
        status_counts=dict(sorted(status_counts.items())),
        claimed_correct=dict(sorted(claimed.items())),
        verified_correct=dict(sorted(verified.items())),
        estimated_cost_usd=sum(record.estimated_cost_usd for record in attempts),
        paid_calls_made=any(
            record.status is not AttemptStatus.NOT_ATTEMPTED for record in attempts
        ),
    )


def run_panel(
    *,
    config: PanelConfig,
    config_hash: str,
    selection: TaskSelection,
    prompt_builder: Callable[[Track, str], str],
    transport: Transport,
    output_root: str | Path,
    environment: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> tuple[tuple[AttemptRecord, ...], PanelSummary]:
    """Run the complete frozen plan, resuming completed attempts.

    A resumed attempt is reloaded, never re-called. The cost guard is checked before each
    call using the running total, and a tripped guard produces a typed row rather than a
    silent stop.
    """

    plan = plan_attempts(selection, config.models)
    records: list[AttemptRecord] = []
    spent = 0.0
    per_model_spent: dict[str, float] = {}

    for spec, track, task_id in plan:
        target = attempt_path(output_root, spec, track, task_id)
        if config.resume and target.exists():
            existing = AttemptRecord.model_validate_json(target.read_text(encoding="utf-8"))
            records.append(existing)
            spent += existing.estimated_cost_usd
            per_model_spent[spec.cell_id] = (
                per_model_spent.get(spec.cell_id, 0.0) + existing.estimated_cost_usd
            )
            continue

        cell_spent = per_model_spent.get(spec.cell_id, 0.0)
        guard = config.cost_guard
        effective_spent = spent
        if guard.max_per_model_usd and cell_spent >= guard.max_per_model_usd:
            effective_spent = max(spent, guard.max_total_usd or cell_spent)

        record = run_attempt(
            spec=spec,
            track=track,
            task_id=task_id,
            user_prompt=prompt_builder(track, task_id),
            config=config,
            transport=transport,
            spent_so_far_usd=effective_spent,
            environment=environment,
            sleep=sleep,
        )
        write_attempt(record, output_root)
        records.append(record)
        spent += record.estimated_cost_usd
        per_model_spent[spec.cell_id] = cell_spent + record.estimated_cost_usd

    summary = summarize_panel(
        config_hash=config_hash,
        selection=selection,
        config=config,
        attempts=records,
    )
    return tuple(records), summary


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI shell
    """Run the credential and exact-model-id preflight. Never makes a paid call."""

    parser = argparse.ArgumentParser(
        description="Goal 11 frontier-LLM external reference panel preflight"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args(argv)

    config, config_hash = load_config(arguments.config)
    results = preflight(config)
    for result in results:
        state = "ok" if result.ok else "refused"
        print(
            f"{result.provider.value}:{result.model_id} {state} "
            f"credential={result.credential_present} listed={result.model_listed} "
            f"{result.detail}"
        )
    if arguments.output_dir:
        root = Path(arguments.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "preflight.json").write_text(
            json.dumps(
                {
                    "config_hash": config_hash,
                    "provider_dependency_note": PROVIDER_DEPENDENCY_NOTE,
                    "results": [result.model_dump(mode="json") for result in results],
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":  # pragma: no cover - module executable entry point
    raise SystemExit(main())
