"""Production binding of the Goal 4 rule engine to the Goal 6 pair contracts.

This is a pure adapter.  A concrete trace state carries the authoritative
``sympy_srepr`` text; the adapter parses it through the frozen safe srepr
parser, converts the binary AST into the e-graph IR, and drives the real
pattern matcher, guards, and appliers at every ordered child-slot occurrence.
Successor states are re-emitted as canonical srepr text and re-parsed, so
``neg``/``sub``/``div`` right-hand sides are lowered to the same ``Add``/
``Mul``/``Pow`` subset the corpus uses and every signature is computed by the
repo-wide ``structural_signature(build_ast(...))`` rule.

The transition verifier is the rule-engine replay adapter from the frozen
pair-generation contract: the 2-6 verification package audits fixed sourced
operator constructions and has no API for arbitrary state transitions, so
replay through the same frozen rules is the authoritative transition evidence
(``VerificationTier.REPLAYED_RULE_ENGINE``).

Nothing here reinterprets Goal 4 saturation provenance as a trace, and no
engine or verifier module is modified.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction

from geml.ast.builder import build_ast_from_parsed
from geml.ast.statistics import structural_signature
from geml.data.pairs.generate import (
    ConcreteActionApplier,
    ConcreteActionEnumerator,
    OrderedBindingV1,
    ReplayStatus,
    RewriteActionV1,
    TraceStateV1,
    TransitionVerificationV1,
    TransitionVerifier,
    canonical_json_bytes,
    make_transition_verifier,
    sha256_digest,
)
from geml.egraph.core import EGraph
from geml.egraph.ir import EClassId, Expr, Operator, add, const, exp, log, mul, power, var
from geml.egraph.patterns import Pattern, PatternVar, match_in_eclass
from geml.egraph.policy import RewriteMode
from geml.egraph.rewrite_engine import (
    Applier,
    AssumptionEnvironment,
    PatternApplier,
    RewriteContext,
    RewriteRule,
    RuleSet,
)
from geml.egraph.rules_domain import domain_rules
from geml.egraph.rules_safe import SAFE_RULES, ConstantFoldApplier
from geml.parsing.srepr import ParsedSreprNode, parse_srepr

RULE_SET_DIGEST_SCHEMA_VERSION = "geml-goal6-rule-set-digest-v1"
VERIFIER_VERSION = "geml-goal6-rule-engine-replay-verifier-v1"

_EMIT_CONSTRUCTORS = {
    Operator.ADD: "Add",
    Operator.MUL: "Mul",
    Operator.POW: "Pow",
    Operator.EXP: "exp",
    Operator.LOG: "log",
}


class ProviderBindingError(ValueError):
    """The production provider binding cannot be constructed as requested."""


@dataclass(frozen=True, slots=True)
class ProductionProviders:
    """The bound engine/verifier callables the production pair build injects."""

    enumerate_actions: ConcreteActionEnumerator
    apply_action: ConcreteActionApplier
    transition_verifier: TransitionVerifier
    verifier_version: str
    rule_set_digests: Mapping[str, str]
    supported_domain_modes: tuple[str, ...]


def _rewrite_mode(domain_mode: str) -> RewriteMode:
    try:
        return RewriteMode(domain_mode)
    except ValueError as error:
        raise NotImplementedError(
            f"domain mode {domain_mode!r} is not a rewrite-engine capability"
        ) from error


def _rules_for(mode: RewriteMode) -> RuleSet:
    # Frozen Goal 4 selection: SAFE_REAL runs the safe rules only, the formal mode
    # adds the guarded domain rules; the OPTIONAL tier stays out, as in the 4-x runs.
    if mode is RewriteMode.SAFE_REAL:
        return SAFE_RULES.enabled_for(mode)
    return SAFE_RULES.merged_with(domain_rules()).enabled_for(mode)


def _symbol_assumptions(parsed: ParsedSreprNode) -> dict[str, tuple[str, ...]]:
    found: dict[str, tuple[str, ...]] = {}
    stack = [parsed]
    while stack:
        node = stack.pop()
        if node.constructor == "Symbol":
            name = str(node.value)
            flags = tuple(flag for flag, enabled in node.assumptions if enabled)
            if found.get(name, flags) != flags:
                raise NotImplementedError(f"symbol {name!r} declares conflicting assumptions")
            found[name] = flags
        stack.extend(node.children)
    return found


def _parsed_to_expr(node: ParsedSreprNode) -> Expr:
    if node.constructor == "Symbol":
        return var(str(node.value))
    if node.constructor == "Integer":
        return const(int(node.value))  # type: ignore[arg-type]
    if node.constructor == "Rational":
        numerator, denominator = node.value  # type: ignore[misc]
        return const(Fraction(numerator, denominator))
    children = [_parsed_to_expr(child) for child in node.children]
    if node.constructor in ("Add", "Mul"):
        build = add if node.constructor == "Add" else mul
        folded = children[0]
        for child in children[1:]:
            folded = build(folded, child)
        return folded
    if node.constructor == "Pow":
        return power(children[0], children[1])
    if node.constructor == "exp":
        return exp(children[0])
    if node.constructor == "log":
        return log(children[0])
    raise NotImplementedError(f"source operator {node.constructor!r} has no e-graph representation")


def _emit_srepr(expr: Expr, assumptions: Mapping[str, tuple[str, ...]]) -> str:
    """Serialize an e-graph tree back into the canonical parser subset."""

    if expr.op is Operator.VARIABLE:
        name = str(expr.payload)
        declared = assumptions.get(name)
        if not declared:
            raise NotImplementedError(f"symbol {name!r} has no declared srepr assumptions")
        flags = ", ".join(f"{flag}=True" for flag in declared)
        return f"Symbol('{name}', {flags})"
    if expr.op is Operator.CONSTANT:
        value = expr.payload
        if not isinstance(value, Fraction):  # pragma: no cover - IR invariant
            raise NotImplementedError("constant payload must be an exact Fraction")
        if value.denominator == 1:
            return f"Integer({value.numerator})"
        return f"Rational({value.numerator}, {value.denominator})"
    children = [_emit_srepr(child, assumptions) for child in expr.children]
    if expr.op is Operator.NEG:
        return f"Mul(Integer(-1), {children[0]})"
    if expr.op is Operator.SUB:
        return f"Add({children[0]}, Mul(Integer(-1), {children[1]}))"
    if expr.op is Operator.DIV:
        return f"Mul({children[0]}, Pow({children[1]}, Integer(-1)))"
    return f"{_EMIT_CONSTRUCTORS[expr.op]}({', '.join(children)})"


def _canonical_state(expr: Expr, assumptions: Mapping[str, tuple[str, ...]]) -> TraceStateV1:
    return trace_state_from_srepr(_emit_srepr(expr, assumptions))


def trace_state_from_srepr(
    sympy_srepr: str,
    *,
    expression_id: str | None = None,
) -> TraceStateV1:
    """Build a concrete trace state with the authoritative structural signature."""

    tree = build_ast_from_parsed(parse_srepr(sympy_srepr), expression_id="state")
    signature = structural_signature(tree)
    return TraceStateV1(
        expression_id=expression_id or f"derived-{signature}",
        sympy_srepr=sympy_srepr,
        structural_signature=signature,
    )


def _occurrences(tree: Expr) -> Iterator[tuple[tuple[int, ...], Expr]]:
    stack: list[tuple[tuple[int, ...], Expr]] = [((), tree)]
    while stack:
        path, node = stack.pop()
        yield path, node
        for slot in range(len(node.children) - 1, -1, -1):
            stack.append(((*path, slot), node.children[slot]))


def _splice(tree: Expr, path: tuple[int, ...], replacement: Expr) -> Expr:
    if not path:
        return replacement
    children = list(tree.children)
    children[path[0]] = _splice(children[path[0]], path[1:], replacement)
    return Expr(op=tree.op, children=tuple(children), payload=tree.payload)


def _class_expr(egraph: EGraph, eclass: EClassId) -> Expr:
    """Extract the unique tree of a merge-free e-class; every class holds one node."""

    nodes = egraph.nodes_of(eclass)
    if len(nodes) != 1:
        raise NotImplementedError("concrete replay requires a merge-free e-graph")
    node = nodes[0]
    children = tuple(_class_expr(egraph, child) for child in node.children)
    return Expr(op=node.op, children=children, payload=node.payload)


def _matched_rewrite(
    egraph: EGraph,
    site: EClassId,
    rule: RewriteRule,
    context: RewriteContext,
) -> tuple[tuple[tuple[str, EClassId], ...], EClassId] | None:
    """Match, guard, and apply one rule at one concrete site; None when it cannot fire."""

    substitutions = match_in_eclass(egraph, rule.lhs, site)
    if not substitutions:
        return None
    if len(substitutions) > 1:
        raise NotImplementedError("a concrete tree site produced an ambiguous match")
    substitution = substitutions[0]
    if rule.guard is not None and not rule.guard(egraph, substitution, context):
        return None
    result = rule.applier(egraph, substitution, context)
    if result.eclass is None:
        return None
    return substitution.bindings, result.eclass


def enumerate_concrete_actions(
    state: TraceStateV1,
    derived_seed: int,
    domain_mode: str,
) -> tuple[RewriteActionV1, ...]:
    """Enumerate every enabled rule at every ordered occurrence of the concrete state."""

    del derived_seed  # action selection randomness lives in the generation loop
    mode = _rewrite_mode(domain_mode)
    parsed = parse_srepr(state.sympy_srepr)
    assumptions = _symbol_assumptions(parsed)
    tree = _parsed_to_expr(parsed)
    source_signature = structural_signature(build_ast_from_parsed(parsed, expression_id="state"))
    context = RewriteContext(mode=mode, assumptions=AssumptionEnvironment.of(**assumptions))
    rules = _rules_for(mode)
    egraph = EGraph()
    actions: list[RewriteActionV1] = []
    for path, subtree in _occurrences(tree):
        site = egraph.add(subtree)
        for rule in rules:
            fired = _matched_rewrite(egraph, site, rule, context)
            if fired is None:
                continue
            bindings, produced = fired
            successor = _canonical_state(
                _splice(tree, path, _class_expr(egraph, produced)), assumptions
            )
            actions.append(
                RewriteActionV1.create(
                    rule_id=rule.rule_id,
                    direction=rule.direction.value,
                    occurrence_path=path,
                    bindings=tuple(
                        OrderedBindingV1(
                            argument_index=index,
                            name=name,
                            value=_emit_srepr(_class_expr(egraph, bound), assumptions),
                        )
                        for index, (name, bound) in enumerate(bindings)
                    ),
                    source_structural_signature=source_signature,
                    successor_structural_signature=successor.structural_signature,
                    rule_assumptions=tuple(sorted(rule.policy.assumptions)),
                    domain_mode=domain_mode,
                )
            )
    return tuple(actions)


def apply_concrete_action(
    state: TraceStateV1,
    action: RewriteActionV1,
    domain_mode: str,
) -> TraceStateV1 | None:
    """Replay one stored action against a concrete state through the real engine."""

    mode = _rewrite_mode(domain_mode)
    parsed = parse_srepr(state.sympy_srepr)
    assumptions = _symbol_assumptions(parsed)
    tree = _parsed_to_expr(parsed)
    subtree = tree
    for slot in action.occurrence_path:
        if slot >= len(subtree.children):
            return None
        subtree = subtree.children[slot]
    context = RewriteContext(mode=mode, assumptions=AssumptionEnvironment.of(**assumptions))
    for rule in _rules_for(mode):
        if rule.rule_id != action.rule_id or rule.direction.value != action.direction:
            continue
        egraph = EGraph()
        fired = _matched_rewrite(egraph, egraph.add(subtree), rule, context)
        if fired is None:
            return None
        _bindings, produced = fired
        return _canonical_state(
            _splice(tree, action.occurrence_path, _class_expr(egraph, produced)),
            assumptions,
        )
    return None


def _replay_transition_verifier() -> TransitionVerifier:
    """Wrap the contract's replay adapter so verifier failures stay typed evidence."""

    inner = make_transition_verifier(apply_concrete_action, verifier_version=VERIFIER_VERSION)

    def verify(
        state: TraceStateV1,
        action: RewriteActionV1,
        expected_successor: TraceStateV1,
        *,
        domain_mode: str,
    ) -> TransitionVerificationV1:
        try:
            return inner(state, action, expected_successor, domain_mode=domain_mode)
        except Exception as error:  # replay evidence must retain its failure
            if isinstance(error, TimeoutError):
                status = ReplayStatus.TIMEOUT
            elif isinstance(error, NotImplementedError):
                status = ReplayStatus.UNSUPPORTED
            else:
                status = ReplayStatus.FAILED
            evidence = {
                "action_digest": action.semantic_digest,
                "domain_mode": domain_mode,
                "error": f"{type(error).__name__}: {error}",
                "expected": expected_successor.model_dump(mode="json"),
                "verifier_version": VERIFIER_VERSION,
            }
            return TransitionVerificationV1(
                action_digest=action.semantic_digest,
                source_structural_signature=state.structural_signature,
                successor_structural_signature=expected_successor.structural_signature,
                verifier_version=VERIFIER_VERSION,
                status=status,
                evidence_digest=sha256_digest(canonical_json_bytes(evidence)),
                detail=f"concrete replay raised {type(error).__name__}",
            )

    return verify


def _pattern_payload(pattern: Pattern) -> dict[str, object]:
    if isinstance(pattern, PatternVar):
        return {"kind": pattern.kind.value, "var": pattern.name}
    payload = pattern.payload
    if isinstance(payload, Fraction):
        rendered: str | None = f"{payload.numerator}/{payload.denominator}"
    else:
        rendered = payload
    return {
        "children": [_pattern_payload(child) for child in pattern.children],
        "op": pattern.op.value,
        "payload": rendered,
    }


def _applier_payload(applier: Applier) -> dict[str, object]:
    if isinstance(applier, PatternApplier):
        return {"rhs": _pattern_payload(applier.rhs)}
    if isinstance(applier, ConstantFoldApplier):
        return {
            "fold": applier.operation.value,
            "max_digits": applier.bound.max_digits,
            "variables": list(applier.variables),
        }
    return {"applier": type(applier).__name__}


def rule_set_digest(mode: RewriteMode) -> str:
    """Digest the complete executable rule content enabled for one rewrite mode."""

    rows = [
        {
            "applier": _applier_payload(rule.applier),
            "assumptions": sorted(rule.policy.assumptions),
            "direction": rule.direction.value,
            "guard": None if rule.guard is None else rule.guard.name,
            "lhs": _pattern_payload(rule.lhs),
            "rule_id": rule.rule_id,
            "tier": rule.policy.tier.value,
        }
        for rule in _rules_for(mode)
    ]
    payload = {
        "mode": mode.value,
        "rules": rows,
        "schema_version": RULE_SET_DIGEST_SCHEMA_VERSION,
    }
    return sha256_digest(canonical_json_bytes(payload))


def bind_production_providers(
    *,
    required_domain_modes: tuple[str, ...] = tuple(mode.value for mode in RewriteMode),
) -> ProductionProviders:
    """Bind the merged engine and replay verifier, refusing typed on any missing capability."""

    modes: list[RewriteMode] = []
    for value in required_domain_modes:
        try:
            mode = RewriteMode(value)
        except ValueError:
            raise ProviderBindingError(
                f"required domain mode {value!r} is not a rewrite-engine capability; "
                f"supported modes are {tuple(item.value for item in RewriteMode)}"
            ) from None
        if len(_rules_for(mode)) == 0:
            raise ProviderBindingError(
                f"no executable rule is enabled for rewrite mode {mode.value!r}"
            )
        modes.append(mode)
    if not modes:
        raise ProviderBindingError("at least one required domain mode must be declared")
    return ProductionProviders(
        enumerate_actions=enumerate_concrete_actions,
        apply_action=apply_concrete_action,
        transition_verifier=_replay_transition_verifier(),
        verifier_version=VERIFIER_VERSION,
        rule_set_digests={mode.value: rule_set_digest(mode) for mode in modes},
        supported_domain_modes=tuple(mode.value for mode in modes),
    )
