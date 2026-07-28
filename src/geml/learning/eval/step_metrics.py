"""Verifier-gated, denominator-complete metrics for Goal 7 rewrite proposals."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from geml.data.pairs.generate import ReplayStatus, RewriteActionV1, TransitionVerificationV1
from geml.data.steps.extract import RewriteStepRecordV1
from geml.learning.policy.head import PolicyOutputV1, ProposalStatus


class StepMetricError(ValueError):
    """A verifier metric row cannot retain its exact structural/replay semantics."""


class ProposalOutcomeStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    NO_ACTION = "no_action"


ActionVerifier = Callable[[RewriteStepRecordV1, RewriteActionV1], TransitionVerificationV1]


@dataclass(frozen=True, slots=True)
class StepMetricOutcomeV1:
    """Per-example evidence from exactly one prediction/verification attempt."""

    step_id: str
    rule_id: str
    operator_family: str
    proposal_status: ProposalStatus
    attempted: bool
    exact_action_top_k: bool
    verifier_valid_top_k: bool
    outcome_statuses: tuple[ProposalOutcomeStatus, ...]

    def __post_init__(self) -> None:
        if not self.step_id or not self.rule_id or not self.operator_family:
            raise StepMetricError("step_id, rule_id, and operator_family must be nonblank")
        if self.proposal_status is ProposalStatus.NO_LEGAL_ACTION and self.attempted:
            raise StepMetricError("a no-legal-action output cannot attempt verifier replay")
        if self.verifier_valid_top_k and ProposalOutcomeStatus.VALID not in self.outcome_statuses:
            raise StepMetricError("valid top-k must retain at least one valid verifier outcome")
        if self.exact_action_top_k and not self.attempted:
            raise StepMetricError("exact top-k match requires a submitted action")


@dataclass(frozen=True, slots=True)
class StepMetricsSummaryV1:
    """Aggregate counts reconstructible from retained per-example outcomes."""

    attempted_examples: int
    verifier_attempted_examples: int
    exact_action_top_k_count: int
    verifier_valid_top_k_count: int
    invalid_proposal_count: int
    timeout_count: int
    failed_verifier_count: int
    unsupported_count: int
    no_action_count: int
    macro_rule_exact_top_k: float | None
    macro_rule_verifier_valid_top_k: float | None
    per_rule: tuple[tuple[str, dict[str, int]], ...]
    unseen_family: tuple[tuple[str, dict[str, int]], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "attempted_examples": self.attempted_examples,
            "exact_action_top_k_count": self.exact_action_top_k_count,
            "failed_verifier_count": self.failed_verifier_count,
            "invalid_proposal_count": self.invalid_proposal_count,
            "macro_rule_exact_top_k": self.macro_rule_exact_top_k,
            "macro_rule_verifier_valid_top_k": self.macro_rule_verifier_valid_top_k,
            "no_action_count": self.no_action_count,
            "per_rule": dict(self.per_rule),
            "timeout_count": self.timeout_count,
            "unseen_family": dict(self.unseen_family),
            "unsupported_count": self.unsupported_count,
            "verifier_attempted_examples": self.verifier_attempted_examples,
            "verifier_valid_top_k_count": self.verifier_valid_top_k_count,
        }


def evaluate_step(
    record: RewriteStepRecordV1,
    output: PolicyOutputV1,
    *,
    verifier: ActionVerifier,
) -> StepMetricOutcomeV1:
    """Evaluate one typed proposal only through concrete verifier replay."""

    if output.state_structural_signature != record.state_structural_signature:
        return StepMetricOutcomeV1(
            step_id=record.step_id,
            rule_id=record.action.rule_id,
            operator_family=record.operator_family,
            proposal_status=ProposalStatus.INVALID_STATE,
            attempted=False,
            exact_action_top_k=False,
            verifier_valid_top_k=False,
            outcome_statuses=(ProposalOutcomeStatus.INVALID,),
        )
    if output.status is ProposalStatus.NO_LEGAL_ACTION:
        return StepMetricOutcomeV1(
            step_id=record.step_id,
            rule_id=record.action.rule_id,
            operator_family=record.operator_family,
            proposal_status=output.status,
            attempted=False,
            exact_action_top_k=False,
            verifier_valid_top_k=False,
            outcome_statuses=(ProposalOutcomeStatus.NO_ACTION,),
        )
    if output.status is not ProposalStatus.PROPOSED:
        return StepMetricOutcomeV1(
            step_id=record.step_id,
            rule_id=record.action.rule_id,
            operator_family=record.operator_family,
            proposal_status=output.status,
            attempted=False,
            exact_action_top_k=False,
            verifier_valid_top_k=False,
            outcome_statuses=(ProposalOutcomeStatus.INVALID,),
        )
    exact = any(
        item.action.semantic_digest == record.action.semantic_digest for item in output.proposals
    )
    statuses: list[ProposalOutcomeStatus] = []
    for proposal in output.proposals:
        try:
            verification = verifier(record, proposal.action)
        except TimeoutError:
            statuses.append(ProposalOutcomeStatus.TIMEOUT)
            continue
        except Exception:
            statuses.append(ProposalOutcomeStatus.FAILED)
            continue
        if (
            verification.source_structural_signature != record.state_structural_signature
            or verification.successor_structural_signature != record.next_state_structural_signature
        ):
            statuses.append(ProposalOutcomeStatus.INVALID)
        elif verification.status is ReplayStatus.PASSED:
            statuses.append(ProposalOutcomeStatus.VALID)
        elif verification.status is ReplayStatus.TIMEOUT:
            statuses.append(ProposalOutcomeStatus.TIMEOUT)
        elif verification.status is ReplayStatus.UNSUPPORTED:
            statuses.append(ProposalOutcomeStatus.UNSUPPORTED)
        else:
            statuses.append(ProposalOutcomeStatus.INVALID)
    return StepMetricOutcomeV1(
        step_id=record.step_id,
        rule_id=record.action.rule_id,
        operator_family=record.operator_family,
        proposal_status=output.status,
        attempted=True,
        exact_action_top_k=exact,
        verifier_valid_top_k=ProposalOutcomeStatus.VALID in statuses,
        outcome_statuses=tuple(statuses),
    )


def summarize_step_outcomes(
    outcomes: Iterable[StepMetricOutcomeV1],
    *,
    registered_rule_ids: Iterable[str],
    unseen_families: Iterable[str],
) -> StepMetricsSummaryV1:
    """Aggregate all retained outcomes, including no-action and verifier failures."""

    rows = tuple(outcomes)
    rules = tuple(sorted(set(registered_rule_ids)))
    if not rules:
        raise ValueError("registered_rule_ids must be nonempty")
    unknown_rules = sorted({row.rule_id for row in rows} - set(rules))
    if unknown_rules:
        raise StepMetricError(f"outcomes include unregistered target rules: {unknown_rules}")
    per_rule: dict[str, dict[str, int]] = {
        rule_id: {"attempted": 0, "exact": 0, "valid": 0} for rule_id in rules
    }
    unseen = set(unseen_families)
    by_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempted": 0, "exact": 0, "valid": 0}
    )
    terminal_counts: Counter[ProposalOutcomeStatus] = Counter()
    for row in rows:
        per_rule[row.rule_id]["attempted"] += 1
        per_rule[row.rule_id]["exact"] += int(row.exact_action_top_k)
        per_rule[row.rule_id]["valid"] += int(row.verifier_valid_top_k)
        if row.operator_family in unseen:
            by_family[row.operator_family]["attempted"] += 1
            by_family[row.operator_family]["exact"] += int(row.exact_action_top_k)
            by_family[row.operator_family]["valid"] += int(row.verifier_valid_top_k)
        terminal_counts.update(set(row.outcome_statuses))
    active_rules = [counts for counts in per_rule.values() if counts["attempted"]]
    macro_exact = None
    macro_valid = None
    if active_rules:
        macro_exact = sum(item["exact"] / item["attempted"] for item in active_rules) / len(
            active_rules
        )
        macro_valid = sum(item["valid"] / item["attempted"] for item in active_rules) / len(
            active_rules
        )
    return StepMetricsSummaryV1(
        attempted_examples=len(rows),
        verifier_attempted_examples=sum(row.attempted for row in rows),
        exact_action_top_k_count=sum(row.exact_action_top_k for row in rows),
        verifier_valid_top_k_count=sum(row.verifier_valid_top_k for row in rows),
        invalid_proposal_count=terminal_counts[ProposalOutcomeStatus.INVALID],
        timeout_count=terminal_counts[ProposalOutcomeStatus.TIMEOUT],
        failed_verifier_count=terminal_counts[ProposalOutcomeStatus.FAILED],
        unsupported_count=terminal_counts[ProposalOutcomeStatus.UNSUPPORTED],
        no_action_count=terminal_counts[ProposalOutcomeStatus.NO_ACTION],
        macro_rule_exact_top_k=macro_exact,
        macro_rule_verifier_valid_top_k=macro_valid,
        per_rule=tuple(sorted(per_rule.items())),
        unseen_family=tuple(sorted(by_family.items())),
    )
