"""Deterministic orchestration for typed GEML source-expression records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from functools import cache
from pathlib import Path
from random import Random
from typing import Annotated, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator
from sympy import srepr, sstr

from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.data.generation.difficulty import (
    DifficultyProfile,
    DifficultyTarget,
    freeze_mapping,
    sample_difficulty_target,
)
from geml.data.generation.grammar import (
    LOG_ARGUMENT_CLASSES,
    TAN_ARGUMENT_CLASSES,
    TRIVIALITY_FEATURES,
    GrammarGenerationError,
    GrammarPolicy,
    TrivialityPolicy,
    generate_tree,
    maximum_leaves_with_arity,
    maximum_leaves_with_arity_count,
    target_is_label_feasible,
    triviality_violations,
)
from geml.spec.corpus_families import (
    CORPUS_FAMILIES,
    CORPUS_FAMILY_REGISTRY,
    FINAL_CORPUS_SIZE,
    CorpusPolicyKind,
    FamilyGenerationBlockedError,
    blocked_operators,
    require_family_generation_ready,
)
from geml.spec.domains import DOMAIN_REGISTRY
from geml.spec.operators import OPERATOR_REGISTRY, EMLConstructionStatus

PositiveInt = Annotated[StrictInt, Field(ge=1)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
NonNegativeWeight = Annotated[StrictFloat, Field(ge=0)]

DEFAULT_CONFIG_PATH = Path(__file__).parents[4] / "configs" / "goal1_generator_final.yaml"
_EXPRESSION_ID_PREFIX = "geml-expression-v1"
_SEED_PREFIX = "geml-generator-seed-v1"
_REQUIRED_SMOKE_COUNT = 10_000


class _HashDigest(Protocol):
    def update(self, data: bytes) -> None: ...


class GeneratorConfigurationError(ValueError):
    """The checked-in generator configuration conflicts with frozen policy."""


class GeneratorPolicyBlockedError(ValueError):
    """A requested family or domain is not generation-approved."""


@cache
def _family_operator_candidates(family_id: str) -> tuple[str, ...]:
    family = CORPUS_FAMILY_REGISTRY[family_id]
    if family.eligible_operators:
        return family.eligible_operators
    return tuple(
        name
        for name, operator in OPERATOR_REGISTRY.items()
        if operator.operator_family in family.operator_family_constraints
    )


class GenerationExhaustedError(RuntimeError):
    """All deterministic attempts for one requested record were retained as failures."""

    def __init__(
        self,
        *,
        expression_index: int,
        family_id: str,
        profile_name: str,
        domain_mode: str,
        stress_criterion: str | None,
        expression_seed: int,
        attempts: int,
        target: DifficultyTarget,
        rejection_reasons: Mapping[str, int],
        labeling_attempts: int,
        labeling_rejection_reasons: Mapping[str, int],
    ) -> None:
        self.expression_index = expression_index
        self.family_id = family_id
        self.profile_name = profile_name
        self.domain_mode = domain_mode
        self.stress_criterion = stress_criterion
        self.expression_seed = expression_seed
        self.attempts = attempts
        self.target = target
        self.rejection_reasons = dict(sorted(rejection_reasons.items()))
        self.labeling_attempts = labeling_attempts
        self.labeling_rejection_reasons = dict(sorted(labeling_rejection_reasons.items()))
        reasons = ", ".join(f"{reason}={count}" for reason, count in self.rejection_reasons.items())
        super().__init__(
            f"generation exhausted for index={expression_index}, family={family_id!r}, "
            f"profile={profile_name!r}, domain={domain_mode!r}, attempts={attempts}; "
            f"{reasons or 'no rejection reason recorded'}"
        )


class FamilyGenerationPolicy(BaseModel):
    """Configurable sampling policy layered on one frozen registry family."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    difficulty_profile: str
    minimum_target_size: PositiveInt
    domain_weights: Mapping[str, NonNegativeWeight] = Field(min_length=1)
    operator_weights: Mapping[str, NonNegativeWeight] = Field(min_length=1)
    required_any_operators: tuple[str, ...]
    required_operator_groups: tuple[tuple[str, ...], ...] = ()
    stress_criterion: str | None = None

    @model_validator(mode="after")
    def validate_weights(self) -> FamilyGenerationPolicy:
        if sum(self.domain_weights.values()) <= 0:
            raise ValueError("domain weights must have positive total weight")
        if sum(self.operator_weights.values()) <= 0:
            raise ValueError("operator weights must have positive total weight")
        if len(self.required_any_operators) != len(set(self.required_any_operators)):
            raise ValueError("required-any operators must be unique")
        if self.required_any_operators and not any(
            self.operator_weights.get(operator, 0.0) > 0 for operator in self.required_any_operators
        ):
            raise ValueError("at least one required-any operator must have positive weight")
        if any(not group for group in self.required_operator_groups):
            raise ValueError("required operator groups must be nonempty")
        if any(len(group) != len(set(group)) for group in self.required_operator_groups):
            raise ValueError("operators within each required group must be unique")
        all_groups = (
            *((self.required_any_operators,) if self.required_any_operators else ()),
            *self.required_operator_groups,
        )
        flattened = tuple(operator for group in all_groups for operator in group)
        if len(flattened) != len(set(flattened)):
            raise ValueError("an operator may belong to only one required group")
        if any(
            not any(self.operator_weights.get(operator, 0.0) > 0 for operator in group)
            for group in self.required_operator_groups
        ):
            raise ValueError("every required operator group needs a positively weighted operator")
        if self.operator_weights.get("symbol", 0.0) <= 0:
            raise ValueError("symbol must have positive weight because every target has variables")
        if self.stress_criterion is not None:
            normalized_criterion = self.stress_criterion.strip()
            if not normalized_criterion:
                raise ValueError("stress criterion must be nonblank when provided")
            object.__setattr__(self, "stress_criterion", normalized_criterion)
        object.__setattr__(self, "domain_weights", freeze_mapping(self.domain_weights))
        object.__setattr__(self, "operator_weights", freeze_mapping(self.operator_weights))
        return self


class SmokePolicy(BaseModel):
    """Enabled-only in-memory smoke schedule; it never redefines final quotas."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    split: CorpusSplit
    family_counts: Mapping[str, NonNegativeInt] = Field(min_length=1)

    @model_validator(mode="after")
    def freeze_family_counts(self) -> SmokePolicy:
        object.__setattr__(self, "family_counts", freeze_mapping(self.family_counts))
        return self


class GeneratorConfig(BaseModel):
    """Validated complete policy for deterministic source generation."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str
    run_seed: StrictInt
    planned_total_count: PositiveInt
    family_quotas: Mapping[str, NonNegativeInt]
    difficulty_profiles: Mapping[str, DifficultyProfile]
    families: Mapping[str, FamilyGenerationPolicy]
    grammar: GrammarPolicy
    triviality: TrivialityPolicy
    maximum_attempts_per_record: PositiveInt
    smoke: SmokePolicy

    @model_validator(mode="after")
    def validate_against_frozen_registries(self) -> GeneratorConfig:
        if self.planned_total_count != FINAL_CORPUS_SIZE:
            raise ValueError(f"planned total must remain {FINAL_CORPUS_SIZE}")
        expected_quotas = {family.family_id: family.quota for family in CORPUS_FAMILIES}
        if self.family_quotas != expected_quotas:
            raise ValueError("configured family quotas must exactly match the frozen registry")
        if set(self.families) != set(CORPUS_FAMILY_REGISTRY):
            raise ValueError("configuration must retain every frozen final family")

        for family_id, policy in self.families.items():
            family = CORPUS_FAMILY_REGISTRY[family_id]
            if policy.difficulty_profile not in self.difficulty_profiles:
                raise ValueError(f"unknown difficulty profile for family {family_id!r}")
            if not set(policy.domain_weights) <= set(family.allowed_domain_modes):
                raise ValueError(f"invalid domain weight for family {family_id!r}")
            if not set(policy.operator_weights) <= set(OPERATOR_REGISTRY):
                raise ValueError(f"unknown operator weight for family {family_id!r}")
            eligible_operators = set(_family_operator_candidates(family_id))
            if not set(policy.operator_weights) <= eligible_operators:
                raise ValueError(f"operator weight outside frozen family {family_id!r}")
            if not set(policy.required_any_operators) <= set(policy.operator_weights):
                raise ValueError(f"required operator lacks a weight in family {family_id!r}")
            grouped_required = {
                operator for group in policy.required_operator_groups for operator in group
            }
            if not grouped_required <= set(policy.operator_weights):
                raise ValueError(f"required operator group lacks a weight in family {family_id!r}")
            if (
                policy.operator_weights.get("power", 0.0) > 0
                and policy.operator_weights.get("integer", 0.0) <= 0
            ):
                raise ValueError(
                    f"power generation requires a positive integer weight in family {family_id!r}"
                )
            if (
                family.policy_kind is CorpusPolicyKind.OOD_STRESS
                and policy.stress_criterion is None
            ):
                raise ValueError(f"OOD family {family_id!r} requires a stress criterion")
            if (
                family.policy_kind is not CorpusPolicyKind.OOD_STRESS
                and policy.stress_criterion is not None
            ):
                raise ValueError(f"IID family {family_id!r} cannot declare a stress criterion")

        if not set(self.smoke.family_counts) <= set(self.families):
            raise ValueError("smoke schedule references an unknown family")
        if sum(self.smoke.family_counts.values()) != _REQUIRED_SMOKE_COUNT:
            raise ValueError(
                f"final smoke schedule must request exactly {_REQUIRED_SMOKE_COUNT} records"
            )
        blocked_smoke_families = {}
        for family_id, count in self.smoke.family_counts.items():
            blockers = blocked_operators(family_id)
            if count > 0 and blockers:
                blocked_smoke_families[family_id] = blockers
        if blocked_smoke_families:
            names = ", ".join(sorted(blocked_smoke_families))
            raise ValueError(f"smoke schedule contains blocked final families: {names}")

        object.__setattr__(self, "family_quotas", freeze_mapping(self.family_quotas))
        object.__setattr__(
            self,
            "difficulty_profiles",
            freeze_mapping(self.difficulty_profiles),
        )
        object.__setattr__(self, "families", freeze_mapping(self.families))
        return self


def load_generator_config(path: str | Path = DEFAULT_CONFIG_PATH) -> GeneratorConfig:
    """Load YAML policy without mutating registries or global random state."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return GeneratorConfig.model_validate(payload)


def _digest_int(prefix: str, *parts: object) -> int:
    digest = hashlib.sha256()
    for part in (prefix, *parts):
        _update_framed_digest(digest, str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), byteorder="big", signed=False)


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_nonblank_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def derive_expression_seed(
    *,
    run_seed: int,
    expression_index: int,
    family_id: str,
    profile_name: str,
    domain_mode: str,
    stress_criterion: str | None = None,
) -> int:
    """Derive a stable per-expression seed from all generation identities."""

    expression_index = _require_nonnegative_integer(
        expression_index,
        name="expression index",
    )
    run_seed = _require_integer(run_seed, name="run seed")
    family_id = _require_nonblank_string(family_id, name="family id")
    profile_name = _require_nonblank_string(profile_name, name="profile name")
    domain_mode = _require_nonblank_string(domain_mode, name="domain mode")
    stable_stress_criterion = (
        ""
        if stress_criterion is None
        else _require_nonblank_string(stress_criterion, name="stress criterion")
    )
    return _digest_int(
        _SEED_PREFIX,
        run_seed,
        expression_index,
        family_id,
        profile_name,
        domain_mode,
        stable_stress_criterion,
    )


def derive_expression_id(*, domain_mode: str, sympy_srepr: str) -> str:
    """Return the project-authoritative lowercase SHA-256 expression identity."""

    domain_mode = _require_nonblank_string(domain_mode, name="domain mode")
    sympy_srepr = _require_nonblank_string(sympy_srepr, name="SymPy srepr")
    canonical_payload = f"{_EXPRESSION_ID_PREFIX}\0{domain_mode}\0{sympy_srepr}"
    payload = canonical_payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_record_bytes(record: ExpressionRecord) -> bytes:
    """Serialize one record deterministically for reproducibility comparisons."""

    return json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _update_framed_digest(digest: _HashDigest, payload: bytes) -> None:
    """Update a hashlib-compatible digest with one length-framed payload."""

    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def canonical_records_hash(records: Iterable[ExpressionRecord]) -> str:
    """Hash an ordered record stream with unambiguous length framing."""

    digest = hashlib.sha256()
    for record in records:
        _update_framed_digest(digest, canonical_record_bytes(record))
    return digest.hexdigest()


def _weighted_name(rng: Random, weights: Mapping[str, float]) -> str:
    choices = tuple(name for name, weight in sorted(weights.items()) if weight > 0)
    if not choices:
        raise GeneratorConfigurationError("weighted policy has no positive entry")
    return rng.choices(
        choices,
        weights=tuple(weights[name] for name in choices),
        k=1,
    )[0]


@cache
def _resolve_approved_operators(family_id: str, domain_mode: str) -> tuple[str, ...]:
    try:
        family = require_family_generation_ready(family_id)
    except FamilyGenerationBlockedError as error:
        raise GeneratorPolicyBlockedError(str(error)) from error

    if domain_mode not in family.allowed_domain_modes:
        raise GeneratorPolicyBlockedError(
            f"domain {domain_mode!r} is not allowed for family {family_id!r}"
        )
    domain = DOMAIN_REGISTRY[domain_mode]
    if not domain.enabled_for_generation:
        raise GeneratorPolicyBlockedError(
            f"domain {domain_mode!r} is disabled for result-bearing generation"
        )

    candidates = _family_operator_candidates(family_id)
    approved = tuple(
        name
        for name in candidates
        if OPERATOR_REGISTRY[name].enabled_for_generation
        and OPERATOR_REGISTRY[name].eml_construction_status is EMLConstructionStatus.APPROVED
        and domain_mode in OPERATOR_REGISTRY[name].domain_modes
    )
    if not approved:
        raise GeneratorPolicyBlockedError(
            f"family {family_id!r} has no approved operators for domain {domain_mode!r}"
        )
    return approved


def preflight_family(
    config: GeneratorConfig,
    family_id: str,
    *,
    domain_mode: str | None = None,
) -> tuple[str, ...]:
    """Return approved operators or raise a complete family/domain policy error."""

    if family_id not in config.families:
        raise GeneratorConfigurationError(f"unknown configured family {family_id!r}")
    policy = config.families[family_id]
    domains = (
        (domain_mode,)
        if domain_mode is not None
        else tuple(name for name, weight in policy.domain_weights.items() if weight > 0)
    )
    weighted = {name for name, weight in policy.operator_weights.items() if weight > 0}
    approved_union: set[str] = set()
    domain_failures: list[str] = []
    for selected_domain in domains:
        approved = set(_resolve_approved_operators(family_id, selected_domain))
        approved_union.update(approved)
        unavailable_weighted = weighted - approved
        if unavailable_weighted:
            names = ", ".join(sorted(unavailable_weighted))
            domain_failures.append(f"{selected_domain}: unavailable weighted operators {names}")
    if domain_failures:
        raise GeneratorPolicyBlockedError(
            f"family {family_id!r} is invalid by domain: {'; '.join(domain_failures)}"
        )
    return tuple(sorted(approved_union))


def _select_domain(
    config: GeneratorConfig,
    *,
    expression_index: int,
    family_id: str,
    profile_name: str,
    domain_mode: str | None,
) -> str:
    policy = config.families[family_id]
    if domain_mode is not None:
        if domain_mode not in policy.domain_weights:
            raise GeneratorConfigurationError(
                f"domain {domain_mode!r} has no configured weight for family {family_id!r}"
            )
        return domain_mode
    selector_seed = _digest_int(
        "geml-generator-domain-v1",
        config.run_seed,
        expression_index,
        family_id,
        profile_name,
    )
    return _weighted_name(Random(selector_seed), policy.domain_weights)


def _rejection_key(error: GrammarGenerationError) -> str:
    return f"grammar:{error.code}"


def _active_required_operator_groups(
    policy: FamilyGenerationPolicy,
) -> tuple[tuple[str, ...], ...]:
    configured_groups = (
        *((policy.required_any_operators,) if policy.required_any_operators else ()),
        *policy.required_operator_groups,
    )
    return tuple(
        tuple(operator for operator in group if policy.operator_weights.get(operator, 0.0) > 0)
        for group in configured_groups
    )


def _minimum_leaves_for_operator(
    operator_name: str,
    *,
    variable_count: int,
    domain_mode: str,
) -> int:
    """Return a conservative leaf requirement for one distinguishing operator."""

    arity = OPERATOR_REGISTRY[operator_name].arity
    if arity == 0:
        return variable_count if operator_name == "symbol" else variable_count + 1
    if arity == 1:
        return variable_count
    if operator_name == "divide" and domain_mode != "positive_real":
        return variable_count + 1
    if operator_name == "power":
        return variable_count + 1
    return max(2, variable_count)


def _target_supports_required_operator(
    target: DifficultyTarget,
    required_operators: tuple[str, ...],
    domain_mode: str,
) -> bool:
    if not required_operators:
        return True
    return any(
        maximum_leaves_with_arity(
            size=target.target_size,
            depth=target.target_depth,
            arity=OPERATOR_REGISTRY[operator_name].arity,
        )
        >= _minimum_leaves_for_operator(
            operator_name,
            variable_count=target.variable_count,
            domain_mode=domain_mode,
        )
        for operator_name in required_operators
    )


def _target_supports_required_operator_groups(
    target: DifficultyTarget,
    required_groups: tuple[tuple[str, ...], ...],
    domain_mode: str,
    *,
    allowed_operators: tuple[str, ...],
    operator_weights: Mapping[str, float],
    grammar_policy: GrammarPolicy,
) -> bool:
    if not all(
        _target_supports_required_operator(target, group, domain_mode) for group in required_groups
    ):
        return False

    uniform_arity_groups: dict[int, list[tuple[str, ...]]] = {}
    for group in required_groups:
        arities = {OPERATOR_REGISTRY[operator].arity for operator in group}
        if len(arities) == 1:
            uniform_arity_groups.setdefault(arities.pop(), []).append(group)
    for arity, groups in uniform_arity_groups.items():
        minimum_leaves = max(
            min(
                _minimum_leaves_for_operator(
                    operator,
                    variable_count=target.variable_count,
                    domain_mode=domain_mode,
                )
                for operator in group
            )
            for group in groups
        )
        if (
            maximum_leaves_with_arity_count(
                size=target.target_size,
                depth=target.target_depth,
                arity=arity,
                minimum_count=len(groups),
            )
            < minimum_leaves
        ):
            return False
    return target_is_label_feasible(
        target=target,
        domain_mode=domain_mode,
        allowed_operators=allowed_operators,
        operator_weights=operator_weights,
        policy=grammar_policy,
        required_operator_groups=required_groups,
    )


def _minimum_leaves_for_required_groups(
    *,
    variable_count: int,
    required_groups: tuple[tuple[str, ...], ...],
    domain_mode: str,
) -> int:
    if not required_groups:
        return variable_count
    return max(
        variable_count,
        *(
            min(
                _minimum_leaves_for_operator(
                    operator_name,
                    variable_count=variable_count,
                    domain_mode=domain_mode,
                )
                for operator_name in group
            )
            for group in required_groups
        ),
    )


def _minimum_required_arity_counts(
    required_groups: tuple[tuple[str, ...], ...],
) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for group in required_groups:
        arities = {OPERATOR_REGISTRY[operator].arity for operator in group}
        if len(arities) == 1:
            counts[arities.pop()] += 1
    return dict(sorted(counts.items()))


def generate_expression(
    config: GeneratorConfig,
    *,
    expression_index: int,
    family_id: str,
    split: CorpusSplit | str,
    profile_name: str | None = None,
    domain_mode: str | None = None,
    target: DifficultyTarget | None = None,
    stress_criterion: str | None = None,
) -> ExpressionRecord:
    """Generate one record independently of every other requested index."""

    expression_index = _require_nonnegative_integer(
        expression_index,
        name="expression index",
    )
    if family_id not in config.families:
        raise GeneratorConfigurationError(f"unknown configured family {family_id!r}")
    family_policy = config.families[family_id]
    family_spec = CORPUS_FAMILY_REGISTRY[family_id]
    if target is not None and target.target_size < family_policy.minimum_target_size:
        raise GeneratorConfigurationError(
            f"target size {target.target_size} is below family {family_id!r} minimum "
            f"{family_policy.minimum_target_size}"
        )
    selected_profile = family_policy.difficulty_profile if profile_name is None else profile_name
    if selected_profile not in config.difficulty_profiles:
        raise GeneratorConfigurationError(f"unknown difficulty profile {selected_profile!r}")
    if stress_criterion is not None:
        stress_criterion = stress_criterion.strip()
        if not stress_criterion:
            raise GeneratorConfigurationError("stress criterion must be nonblank")
    if family_spec.policy_kind is CorpusPolicyKind.OOD_STRESS:
        is_policy_override = (
            selected_profile != family_policy.difficulty_profile or target is not None
        )
        if stress_criterion is None and is_policy_override:
            raise GeneratorConfigurationError(
                "OOD profile or target overrides require an explicit stress criterion"
            )
        selected_stress_criterion = stress_criterion or family_policy.stress_criterion
        if selected_stress_criterion is None:
            raise GeneratorConfigurationError("OOD generation requires a stress criterion")
    else:
        if stress_criterion is not None:
            raise GeneratorConfigurationError(
                f"IID family {family_id!r} cannot use a stress criterion"
            )
        selected_stress_criterion = None
    selected_domain = _select_domain(
        config,
        expression_index=expression_index,
        family_id=family_id,
        profile_name=selected_profile,
        domain_mode=domain_mode,
    )
    approved_operators = preflight_family(
        config,
        family_id,
        domain_mode=selected_domain,
    )
    required_groups = _active_required_operator_groups(family_policy)

    expression_seed = derive_expression_seed(
        run_seed=config.run_seed,
        expression_index=expression_index,
        family_id=family_id,
        profile_name=selected_profile,
        domain_mode=selected_domain,
        stress_criterion=selected_stress_criterion,
    )
    if target is None:
        difficulty_rng = Random(_digest_int("geml-generator-difficulty-v1", expression_seed))
        try:
            selected_target = sample_difficulty_target(
                config.difficulty_profiles[selected_profile],
                difficulty_rng,
                minimum_size=family_policy.minimum_target_size,
                target_predicate=lambda candidate: _target_supports_required_operator_groups(
                    candidate,
                    required_groups,
                    selected_domain,
                    allowed_operators=approved_operators,
                    operator_weights=family_policy.operator_weights,
                    grammar_policy=config.grammar,
                ),
            )
        except ValueError as error:
            raise GeneratorConfigurationError(
                f"difficulty profile {selected_profile!r} cannot realize the required operators"
            ) from error
    else:
        selected_target = target
        if not _target_supports_required_operator_groups(
            selected_target,
            required_groups,
            selected_domain,
            allowed_operators=approved_operators,
            operator_weights=family_policy.operator_weights,
            grammar_policy=config.grammar,
        ):
            raise GeneratorConfigurationError(
                f"target size={selected_target.target_size}, "
                f"depth={selected_target.target_depth} cannot realize every required "
                f"operator group for family {family_id!r}"
            )
    variable_names = config.grammar.variable_names[: selected_target.variable_count]
    minimum_leaf_count = _minimum_leaves_for_required_groups(
        variable_count=selected_target.variable_count,
        required_groups=required_groups,
        domain_mode=selected_domain,
    )
    minimum_arity_counts = _minimum_required_arity_counts(required_groups)
    rejection_reasons: Counter[str] = Counter()
    labeling_attempts = 0
    labeling_rejection_reasons: Counter[str] = Counter()

    for attempt in range(1, config.maximum_attempts_per_record + 1):
        attempt_seed = _digest_int("geml-generator-attempt-v1", expression_seed, attempt)
        try:
            tree = generate_tree(
                target=selected_target,
                domain_mode=selected_domain,
                variable_names=variable_names,
                allowed_operators=approved_operators,
                operator_weights=family_policy.operator_weights,
                policy=config.grammar,
                rng=Random(attempt_seed),
                minimum_leaf_count=minimum_leaf_count,
                minimum_operator_arity_counts=minimum_arity_counts,
            )
        except GrammarGenerationError as error:
            rejection_reasons[_rejection_key(error)] += 1
            labeling_attempts += error.labeling_attempts
            labeling_rejection_reasons.update(error.rejection_reasons)
            continue

        operator_counts = dict(tree.operator_counts)
        labeling_attempts += tree.labeling_attempts
        labeling_rejection_reasons.update(dict(tree.labeling_rejection_reasons))
        missing_groups = tuple(
            group
            for group in required_groups
            if not any(operator_counts.get(operator, 0) > 0 for operator in group)
        )
        if missing_groups:
            for group in missing_groups:
                group_name = "|".join(group)
                rejection_reasons[f"missing_required_operator_group:{group_name}"] += 1
            continue

        triviality_counts = dict(tree.triviality_counts)
        violations = triviality_violations(triviality_counts, config.triviality)
        if violations:
            for feature in violations:
                rejection_reasons[f"triviality_cap:{feature}"] += 1
            continue

        authoritative_srepr = srepr(tree.expression, order="none")
        expression_id = derive_expression_id(
            domain_mode=selected_domain,
            sympy_srepr=authoritative_srepr,
        )
        tan_argument_classes = dict(tree.tan_argument_classes)
        metadata = {
            "generator": "geml.data.generation.generator",
            "generator_schema_version": config.schema_version,
            "expression_index": expression_index,
            "difficulty_profile": selected_profile,
            "stress_criterion": selected_stress_criterion,
            "target_source": "sampled" if target is None else "caller_supplied",
            "target_intermediate_leaf_probability": selected_target.intermediate_leaf_probability,
            "attempts": attempt,
            "rejected_attempts": attempt - 1,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "labeling_attempts": labeling_attempts,
            "labeling_rejected_attempts": sum(labeling_rejection_reasons.values()),
            "labeling_rejection_reasons": dict(sorted(labeling_rejection_reasons.items())),
            "achieved_source_ast_size": tree.source_size,
            "achieved_source_depth": tree.source_depth,
            "achieved_source_leaf_count": tree.source_leaf_count,
            "intermediate_leaf_count": tree.intermediate_leaf_count,
            "metric_status": "generator_logical_targets_not_parser_verified",
            "sympy_printer_order": "none",
            "operator_counts": operator_counts,
            "required_operator_groups": [list(group) for group in required_groups],
            "log_argument_classes": dict(tree.log_argument_classes),
            "tan_argument_classes": {
                name: tan_argument_classes.get(name, 0) for name in TAN_ARGUMENT_CLASSES
            },
            "triviality_counts": triviality_counts,
            "corpus_triviality_rate_caps": dict(config.triviality.corpus_rate_caps),
            "domain_guards": {
                "log_arguments": "positive_expression_grammar",
                "division_denominators": "positive_expression_grammar",
                "negative_power_bases": "positive_expression_grammar",
                "tan_arguments": "closed_unit_interval_structural_grammar",
            },
        }
        return ExpressionRecord(
            expression_id=expression_id,
            sympy_srepr=authoritative_srepr,
            display_text=sstr(tree.expression, order="none"),
            latex_text=None,
            split=split,
            operator_family=family_id,
            domain_mode=selected_domain,
            variables=variable_names,
            target_ast_size=selected_target.target_size,
            target_depth=selected_target.target_depth,
            generator_seed=expression_seed,
            generator_metadata=metadata,
        )

    raise GenerationExhaustedError(
        expression_index=expression_index,
        family_id=family_id,
        profile_name=selected_profile,
        domain_mode=selected_domain,
        stress_criterion=selected_stress_criterion,
        expression_seed=expression_seed,
        attempts=config.maximum_attempts_per_record,
        target=selected_target,
        rejection_reasons=rejection_reasons,
        labeling_attempts=labeling_attempts,
        labeling_rejection_reasons=labeling_rejection_reasons,
    )


def iter_expressions(
    config: GeneratorConfig,
    *,
    count: int,
    family_id: str,
    split: CorpusSplit | str,
    start_index: int = 0,
    profile_name: str | None = None,
    domain_mode: str | None = None,
    stress_criterion: str | None = None,
) -> Iterator[ExpressionRecord]:
    """Yield a caller-sized in-memory stream without assigning split policy."""

    count = _require_nonnegative_integer(count, name="count")
    start_index = _require_nonnegative_integer(start_index, name="start index")
    if family_id not in config.families:
        raise GeneratorConfigurationError(f"unknown configured family {family_id!r}")
    for expression_index in range(start_index, start_index + count):
        yield generate_expression(
            config,
            expression_index=expression_index,
            family_id=family_id,
            split=split,
            profile_name=profile_name,
            domain_mode=domain_mode,
            stress_criterion=stress_criterion,
        )


def _validated_family_counts(
    config: GeneratorConfig,
    family_counts: Mapping[str, int],
    *,
    require_positive_total: bool,
) -> dict[str, int]:
    """Validate an entire family schedule before any record is generated."""

    validated: dict[str, int] = {}
    for family_id, count in family_counts.items():
        if family_id not in config.families:
            raise GeneratorConfigurationError(f"unknown configured family {family_id!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise GeneratorConfigurationError(
                f"family count for {family_id!r} must be a nonnegative integer"
            )
        validated[family_id] = count
    if require_positive_total and sum(validated.values()) <= 0:
        raise GeneratorConfigurationError("smoke schedule must request at least one record")
    return validated


def iter_quota_fixture(
    config: GeneratorConfig,
    *,
    family_counts: Mapping[str, int],
    split: CorpusSplit | str,
    start_index: int = 0,
) -> Iterator[ExpressionRecord]:
    """Yield exact requested fixture counts in mapping order; no final split allocation."""

    start_index = _require_nonnegative_integer(start_index, name="start index")
    requested_counts = _validated_family_counts(
        config,
        family_counts,
        require_positive_total=False,
    )
    expression_index = start_index
    for family_id, count in requested_counts.items():
        for _ in range(count):
            yield generate_expression(
                config,
                expression_index=expression_index,
                family_id=family_id,
                split=split,
            )
            expression_index += 1


def _sorted_counter(counter: Counter[object]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def run_smoke(
    config: GeneratorConfig,
    *,
    family_counts: Mapping[str, int] | None = None,
    split: CorpusSplit | str | None = None,
) -> dict[str, object]:
    """Run an in-memory deterministic smoke and return complete JSON-compatible accounting."""

    requested_counts = _validated_family_counts(
        config,
        config.smoke.family_counts if family_counts is None else family_counts,
        require_positive_total=True,
    )
    selected_split = config.smoke.split if split is None else split
    attempted = sum(requested_counts.values())
    successful = 0
    failures: list[dict[str, object]] = []
    family_distribution: Counter[str] = Counter()
    domain_distribution: Counter[str] = Counter()
    target_depths: Counter[int] = Counter()
    achieved_depths: Counter[int] = Counter()
    target_sizes: Counter[int] = Counter()
    achieved_sizes: Counter[int] = Counter()
    variable_counts: Counter[int] = Counter()
    stress_criteria: Counter[str] = Counter()
    operator_usage: Counter[str] = Counter()
    log_argument_classes: Counter[str] = Counter()
    tan_argument_classes: Counter[str] = Counter()
    triviality_events: Counter[str] = Counter({name: 0 for name in TRIVIALITY_FEATURES})
    triviality_records: Counter[str] = Counter({name: 0 for name in TRIVIALITY_FEATURES})
    generation_attempts = 0
    rejected_attempts = 0
    attempt_rejection_reasons: Counter[str] = Counter()
    labeling_attempts = 0
    labeling_rejected_attempts = 0
    labeling_rejection_reasons: Counter[str] = Counter()
    expression_id_counts: Counter[str] = Counter()
    digest = hashlib.sha256()

    expression_index = 0
    for family_id, count in requested_counts.items():
        for _ in range(count):
            try:
                record = generate_expression(
                    config,
                    expression_index=expression_index,
                    family_id=family_id,
                    split=selected_split,
                )
            except (
                GenerationExhaustedError,
                GeneratorConfigurationError,
                GeneratorPolicyBlockedError,
            ) as error:
                if isinstance(error, GenerationExhaustedError):
                    generation_attempts += error.attempts
                    rejected_attempts += error.attempts
                    attempt_rejection_reasons.update(error.rejection_reasons)
                    labeling_attempts += error.labeling_attempts
                    labeling_rejected_attempts += sum(error.labeling_rejection_reasons.values())
                    labeling_rejection_reasons.update(error.labeling_rejection_reasons)
                    failure_details: dict[str, object] = {
                        "profile_name": error.profile_name,
                        "domain_mode": error.domain_mode,
                        "stress_criterion": error.stress_criterion,
                        "generator_seed": error.expression_seed,
                        "attempts": error.attempts,
                        "target": error.target.model_dump(mode="json"),
                        "rejection_reasons": error.rejection_reasons,
                        "labeling_attempts": error.labeling_attempts,
                        "labeling_rejection_reasons": error.labeling_rejection_reasons,
                    }
                else:
                    failure_details = {}
                failures.append(
                    {
                        "expression_index": expression_index,
                        "family_id": family_id,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        **failure_details,
                    }
                )
            else:
                successful += 1
                expression_id_counts[record.expression_id] += 1
                _update_framed_digest(digest, canonical_record_bytes(record))
                metadata = record.generator_metadata
                record_attempts = int(metadata["attempts"])
                generation_attempts += record_attempts
                rejected_attempts += int(metadata["rejected_attempts"])
                attempt_rejection_reasons.update(metadata["rejection_reasons"])
                labeling_attempts += int(metadata["labeling_attempts"])
                labeling_rejected_attempts += int(metadata["labeling_rejected_attempts"])
                labeling_rejection_reasons.update(metadata["labeling_rejection_reasons"])
                family_distribution[record.operator_family] += 1
                domain_distribution[record.domain_mode] += 1
                target_depths[record.target_depth] += 1
                achieved_depths[int(metadata["achieved_source_depth"])] += 1
                target_sizes[record.target_ast_size] += 1
                achieved_sizes[int(metadata["achieved_source_ast_size"])] += 1
                variable_counts[len(record.variables)] += 1
                stress_criterion_value = metadata.get("stress_criterion")
                if isinstance(stress_criterion_value, str):
                    stress_criteria[stress_criterion_value] += 1
                operator_usage.update(metadata["operator_counts"])
                log_argument_classes.update(metadata["log_argument_classes"])
                tan_argument_classes.update(metadata["tan_argument_classes"])
                counts = metadata["triviality_counts"]
                for feature in TRIVIALITY_FEATURES:
                    count_value = int(counts[feature])
                    triviality_events[feature] += count_value
                    if count_value:
                        triviality_records[feature] += 1
            expression_index += 1

    rates = {
        feature: (triviality_records[feature] / successful if successful else 0.0)
        for feature in TRIVIALITY_FEATURES
    }
    rate_cap_violations = {
        feature: {
            "observed": rates[feature],
            "cap": config.triviality.corpus_rate_caps[feature],
        }
        for feature in TRIVIALITY_FEATURES
        if rates[feature] > config.triviality.corpus_rate_caps[feature]
    }
    final_family_blockers = {}
    for family in CORPUS_FAMILIES:
        blockers = blocked_operators(family)
        if blockers:
            final_family_blockers[family.family_id] = list(blockers)
    repeated_expression_id_count = sum(count > 1 for count in expression_id_counts.values())
    repeated_expression_occurrences = successful - len(expression_id_counts)
    return {
        "attempted": attempted,
        "successful": successful,
        "failure_count": len(failures),
        "failures": failures,
        "generation_attempts": generation_attempts,
        "rejected_attempts": rejected_attempts,
        "attempt_rejection_reasons": _sorted_counter(attempt_rejection_reasons),
        "labeling_attempts": labeling_attempts,
        "labeling_rejected_attempts": labeling_rejected_attempts,
        "labeling_rejection_reasons": _sorted_counter(labeling_rejection_reasons),
        "family_counts": _sorted_counter(family_distribution),
        "domain_counts": _sorted_counter(domain_distribution),
        "target_depth_distribution": _sorted_counter(target_depths),
        "achieved_depth_distribution": _sorted_counter(achieved_depths),
        "target_size_distribution": _sorted_counter(target_sizes),
        "achieved_size_distribution": _sorted_counter(achieved_sizes),
        "variable_count_distribution": _sorted_counter(variable_counts),
        "stress_criterion_counts": _sorted_counter(stress_criteria),
        "operator_usage": _sorted_counter(operator_usage),
        "log_argument_class_distribution": {
            name: log_argument_classes[name] for name in LOG_ARGUMENT_CLASSES
        },
        "tan_argument_class_distribution": {
            name: tan_argument_classes[name] for name in TAN_ARGUMENT_CLASSES
        },
        "triviality_event_counts": _sorted_counter(triviality_events),
        "triviality_record_rates": rates,
        "triviality_rate_cap_violations": rate_cap_violations,
        "unique_expression_ids": len(expression_id_counts),
        "repeated_expression_id_count": repeated_expression_id_count,
        "repeated_expression_occurrences": repeated_expression_occurrences,
        "maximum_expression_id_multiplicity": max(
            expression_id_counts.values(),
            default=0,
        ),
        "canonical_hash": digest.hexdigest(),
        "blocked_final_families": final_family_blockers,
    }
