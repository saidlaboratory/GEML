# Frontier-LLM external reference panel

**Owner:** issue [11-3] (#82) · **Status:** `phase_a_implemented`, production pending
**Implementation:** `src/geml/experiments/goal11/llm_reference.py` ·
**Configuration:** `configs/goal11_llm.yaml` ·
**Tests:** `tests/experiments/test_goal11_llm_smoke.py` (43 tests, fully mocked)

> **This panel is external context, not a controlled baseline.** No row it produces may enter
> Gate G6-G11. `geml.analysis.goal9.summary.CONTROLLED_METHODS` contains no LLM method, so an
> LLM row cannot reach a controlled aggregate even by accident; ingested rows are counted
> separately as `external_reference_rows` and labelled external.

**No paid call has been made.** `configs/goal11_llm.yaml` ships with
`allow_paid_calls: false`, `spend_confirmed: false`, and a zero cost guard.

---

## 1. Frozen workload

| Item | Value |
|---|---|
| Proof tasks per model | exactly 100, stratified |
| SR tasks per model | exactly 100, stratified |
| Attempts per model | exactly 200 |
| Configured models | 4 (one per provider) |
| Retained success/failure rows | 800 |

The **same two ordered ID lists** are used by every provider and every model. They come from
the frozen #67 proof benchmark and the frozen #71 SR benchmark, and both lists plus their
SHA-256 hashes are persisted by `freeze_task_selection()` **before any call is made**.
Selection is round-robin over declared strata, ordered by task id inside each stratum, so it
is a pure function of the candidate population and cannot be nudged after seeing a response.

`plan_attempts()` produces the complete attempt plan up front; the tests assert that the plan
is exactly 200 entries per model and that every model receives an identical task sequence.

---

## 2. Provider adapters

Four adapters, standard-library HTTP only.

| Provider | Generation endpoint | Model list | Auth | Key env |
|---|---|---|---|---|
| OpenAI | `POST /v1/responses` | `GET /v1/models` | `Authorization: Bearer …` | `OPENAI_API_KEY` |
| Anthropic | `POST /v1/messages` | `GET /v1/models` | `x-api-key` + `anthropic-version: 2023-06-01` | `ANTHROPIC_API_KEY` |
| Google | `POST /v1beta/models/{model}:generateContent` | `GET /v1beta/models` | `x-goog-api-key` | `GEMINI_API_KEY` |
| Moonshot | `POST /v1/chat/completions` | `GET /v1/models` | `Authorization: Bearer …` | `MOONSHOT_API_KEY` |

Endpoint, header, and field facts were read from each provider's own current documentation on
2026-07-26 and are recorded in `PROVIDER_ENDPOINTS` with the documentation URL. Because model
IDs and API surfaces change, every fact that matters for spend is re-validated by
`preflight()` immediately before any paid call.

Per-provider field mapping implemented in `build_request` / `parse_response`:

| Concern | OpenAI | Anthropic | Google | Moonshot |
|---|---|---|---|---|
| System prompt | `instructions` | `system` | `systemInstruction` | `messages[role=system]` |
| User prompt | `input` | `messages[role=user]` | `contents[].parts[].text` | `messages[role=user]` |
| Output token cap | `max_output_tokens` | `max_tokens` (required) | `generationConfig.maxOutputTokens` | `max_completion_tokens` |
| Structured output | `text.format.json_schema` | `output_config.format` | `generationConfig.responseSchema` | `response_format.json_schema` |
| Reasoning control | `reasoning.effort` | `output_config.effort` | `generationConfig.thinkingConfig.thinkingLevel` | `reasoning_effort` / `thinking.type` |
| Response text | `output_text` / `output[].content[].text` | `content[].text` | `candidates[0].content.parts[].text` | `choices[0].message.content` |
| Usage | `usage.input_tokens` / `output_tokens` | `usage.input_tokens` / `output_tokens` | `usageMetadata.promptTokenCount` / `candidatesTokenCount` | `usage.prompt_tokens` / `completion_tokens` |
| Served model | `model` | `model` | `modelVersion` | `model` |
| Finish reason | `status` | `stop_reason` | `candidates[0].finishReason` | `choices[0].finish_reason` |

---

## 3. Exact model identity

Preferred anchors recorded on 2026-07-26:

```text
OpenAI     gpt-5.6-sol
Anthropic  claude-opus-5   (claude-fable-5 only as a deliberate higher-cost choice)
Google     gemini-3.6-flash
Moonshot   kimi-k2.5
```

These are **preferences, not assumptions**. `preflight()` refuses to mark a model usable
unless the exact string appears in that provider's own live model-list response. Refused
outright:

* a model id absent from the live list;
* an alias or near match the caller did not request;
* a silent fallback or provider substitution;
* any never-guess identifier. `kimi-k3` is on `NEVER_GUESS_MODEL_IDS`, so `ModelSpec` raises
  if it is configured; if a provider genuinely exposes it, it must be introduced by editing
  the configuration after confirming the exact string, never by the code choosing one.
* preflight with **no model-list transport at all** — an unvalidated model is refused rather
  than assumed available.

At response time, if the provider's returned `model` / `modelVersion` differs from the
requested id, the attempt is recorded as `model_identity_mismatch` and is not counted as a
success. `AttemptRecord` additionally refuses to validate a `success` row whose returned
model differs from the requested one.

**Lifecycle note.** During Phase-A research the Moonshot documentation flagged `kimi-k2.5`
for platform sunset and listed newer ids. Nothing was changed on that basis: the anchor stays
`kimi-k2.5` and preflight must confirm it immediately before any paid call.

---

## 4. Prompt fairness

One frozen system prompt and one frozen structured-output schema per track
(`PROOF_SYSTEM_PROMPT` / `PROOF_OUTPUT_SCHEMA`, `SR_SYSTEM_PROMPT` / `SR_OUTPUT_SCHEMA`).
`prompt_hash()` binds the system prompt, the user prompt, and the schema, and the digest is
recorded on every attempt.

* **A proof answer must be a complete, ordered, replayable action trace** reaching the exact
  structural goal — rule id, direction, ordered occurrence path as child-slot integers, and
  ordered arguments. The prompt states explicitly that restating or echoing the target is not
  a proof, because the target is already supplied in the problem.
* **An SR answer must be a single closed-form expression** under the frozen output grammar.
* Web browsing, external tools, retrieval, and code execution are disabled in the prompt text
  and no tool definitions are ever sent.
* Token and time budgets are configured per model and recorded per attempt.
* Reasoning/thinking/effort is set explicitly per provider and recorded per attempt.
* **No output is manually repaired before verification.**
* Concurrency and retries are bounded by configuration.

**Determinism is documented, not claimed.** None of the four providers exposes a usable seed
for reasoning models; each `ModelSpec` carries a `determinism_note` recording exactly why,
and the note is persisted on every attempt.

---

## 5. Every attempt is persisted

`AttemptRecord` (`geml-llm-attempt-v1`) stores: provider; requested and returned exact model;
access date; track; task id and task hash; prompt hash; the full system and user prompts; the
reasoning field and value; token and time budgets; supported sampling fields; the raw
response; the parsed response; usage; latency; the nested retry history; estimated cost;
typed status; claimed correctness; and verifier-confirmed correctness with its detail.

**Retries are nested evidence.** `RetryRecord` entries live inside the attempt that provoked
them. A retry never creates a new attempt, never replaces the original failure, and never
increases the 200-attempt denominator. This is asserted by
`test_retries_do_not_increase_the_attempt_denominator`.

`AttemptStatus` values: `success`, `refusal`, `parse_failure`, `timeout`, `rate_limited`,
`api_error`, `model_identity_mismatch`, `unsupported_parameter`, `cost_guard_tripped`,
`not_attempted`.

`VerificationStatus` values: `verified_correct`, `verified_incorrect`, `unknown`,
`unsupported`, `not_run`. **Claimed correctness and verifier-confirmed correctness are
separate fields and are always reported as a pair with explicit denominators.**
`AttemptRecord` refuses to validate `verified_correct` on a non-successful attempt.

---

## 6. Credentials, cost, and spend

* Credentials come **only** from environment variables (or secret infrastructure). They are
  never logged, never committed, and never enter a persisted row.
* `redact()` strips any supplied secret from raw bodies and error text before storage.
  `test_no_persisted_attempt_contains_a_credential` walks every written file to confirm it.
* `CostGuard` sets `max_total_usd`, `max_per_model_usd`, and `max_attempts_per_model`. A
  tripped guard produces a typed `cost_guard_tripped` row, not a silent stop.
* Two independent interlocks: `allow_paid_calls` and `spend_confirmed`. Setting the first
  without the second is rejected by the config validator. With `allow_paid_calls: false` the
  runner still builds and records the full plan, writing `not_attempted` rows, so the
  workload is auditable before a single cent is spent.
* `resume: true` reloads completed attempts instead of re-calling them.

---

## 7. Ownership blocker: provider dependencies

**No assigned issue owns a `[llm]` optional-dependency edit in `pyproject.toml`.** This
implementation therefore uses only `urllib` from the standard library, and touches no root
metadata. Each adapter is a small request builder plus a small response parser; there is no
generic client framework, no retry library, and no vendored SDK.

`PROVIDER_DEPENDENCY_NOTE` records the stop condition in code: if a provider ever requires
behaviour that cannot stay small and auditable here — streaming, SSE reconnection, signed
requests — **stop and request an explicitly owned optional provider-dependency contract**
rather than growing this file.

---

## 8. Running it

Preflight (free, no generation call):

```bash
python -m geml.experiments.goal11.llm_reference \
  --config configs/goal11_llm.yaml \
  --output-dir outputs/final/goal11/llm_reference
```

The production panel additionally requires, in order:

1. the frozen #67 proof task ids and the frozen #71 SR task ids;
2. provider credentials in the environment;
3. a green preflight for all four exact model ids;
4. non-zero cost guards;
5. `allow_paid_calls: true` **and** `spend_confirmed: true`;
6. a separate explicit spend confirmation from the user.

Production root: `outputs/final/goal11/llm_reference/`, one atomic JSON file per attempt at
`<provider>/<model>/<track>/<task_id>.json`.

---

## 9. Phase-A test coverage

43 mocked tests, no key, no network, no GPU, no production artifact. They cover: success;
refusal; timeout; rate limiting with retry; malformed and non-JSON responses; alias and
fallback mismatch; a model missing from the live list; preflight without a transport; an
unsupported parameter; the cost guard; resume and idempotency; raw-response preservation;
secret redaction across every persisted file; claimed-versus-verified disagreement; the
"only a success can be verified correct" rule; exactly 200 planned attempts per model; and
per-provider request building and response parsing for all four adapters.
