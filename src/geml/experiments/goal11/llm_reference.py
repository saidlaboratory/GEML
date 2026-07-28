"""Credential-free contracts for verifier-normalized external LLM reference attempts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MOONSHOT = "moonshot"


class AttemptStatus(StrEnum):
    COMPLETE = "complete"
    INVALID_FORMAT = "invalid_format"
    REFUSAL = "refusal"
    TIMEOUT = "timeout"
    API_FAILURE = "api_failure"


@dataclass(frozen=True, slots=True)
class LLMModelPinV1:
    provider: Provider
    model_id: str
    access_date: str
    prompt_hash: str

    def __post_init__(self) -> None:
        if not self.model_id or not self.access_date or not self.prompt_hash.startswith("sha256:"):
            raise ValueError("provider model ID/date/prompt checksum must be explicit")


@dataclass(frozen=True, slots=True)
class LLMReferenceAttemptV1:
    task_id: str
    track: str
    model: LLMModelPinV1
    status: AttemptStatus
    claimed_correct: bool | None
    verifier_confirmed: bool | None
    raw_response: str | None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status is AttemptStatus.COMPLETE and self.raw_response is None:
            raise ValueError("complete LLM attempt must retain raw response")
        if self.verifier_confirmed is True and self.claimed_correct is not True:
            raise ValueError("verifier-confirmed correctness requires a claimed response")
        if self.status is not AttemptStatus.COMPLETE and not self.error:
            raise ValueError("noncomplete LLM attempts require explicit error detail")


def freeze_panel_ids(
    proof_ids: Iterable[str],
    sr_ids: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Freeze exactly 100 IDs from each pre-stratified source list before any API request."""

    proof = tuple(sorted(set(proof_ids)))
    sr = tuple(sorted(set(sr_ids)))
    if len(proof) != 100 or len(sr) != 100:
        raise ValueError("each external LLM panel track requires exactly 100 frozen IDs")
    return proof, sr


def prompt_hash(system_prompt: str, user_prompt: str) -> str:
    return "sha256:" + hashlib.sha256(f"{system_prompt}\0{user_prompt}".encode()).hexdigest()


def mocked_attempt(
    *,
    task_id: str,
    track: str,
    model: LLMModelPinV1,
    call: Callable[[], str],
    verifier: Callable[[str], bool],
) -> LLMReferenceAttemptV1:
    """Run an injected mock call; production requires independent spend authorization."""

    try:
        raw = call()
    except TimeoutError:
        return LLMReferenceAttemptV1(
            task_id, track, model, AttemptStatus.TIMEOUT, None, None, None, "timeout"
        )
    except Exception as error:
        return LLMReferenceAttemptV1(
            task_id, track, model, AttemptStatus.API_FAILURE, None, None, None, str(error)
        )
    claimed = bool(raw.strip())
    try:
        confirmed = verifier(raw) if claimed else False
    except Exception as error:
        return LLMReferenceAttemptV1(
            task_id, track, model, AttemptStatus.INVALID_FORMAT, claimed, None, raw, str(error)
        )
    return LLMReferenceAttemptV1(
        task_id, track, model, AttemptStatus.COMPLETE, claimed, confirmed, raw
    )
