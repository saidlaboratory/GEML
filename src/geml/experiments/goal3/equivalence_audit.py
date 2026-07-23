"""
equivalence_audit.py - compares 3-3's post-hoc DAG path against 3-4's
direct DAG path across a stratified set of cases

owned by 3-5

honesty note: only exp/ln and compositions of them are actually
runnable right now, since those are the only two verified eml
constructions available (see 3-3/3-4). the audit set below includes
one deliberately blocked case for "add" to prove the harness reports
missing coverage explicitly instead of silently skipping it - matches
the acceptance criterion "match exactly or are reported as blockers"
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from geml.graph.schema import compute_statistics
from geml.graph.signatures import compute_signature
from geml.dag.eml import EmlNode, eml_to_dag, make_exp, make_ln, evaluate_dag, validate_eml_purity
from geml.dag.hashcons import HashConsTable
from geml.dag.direct_eml import emit_var, emit_exp, emit_ln, compile_with_stats


@dataclass
class AuditCase:
    name: str
    family: str        # "exp" | "ln" | "composed" | "repeated" | "add" (blocked)
    size_bucket: str    # "shallow" | "medium" | "deep"
    split: str          # stand-in for a real corpus split concept - "core" | "composed" | "repeated" | "domain"
    build_direct: Callable[[HashConsTable], str] | None = None
    build_posthoc: Callable[[], EmlNode] | None = None
    eval_bindings: dict[str, float] = field(default_factory=dict)
    blocked_reason: str | None = None  # if set, this case is a known gap, not run


@dataclass
class AuditResult:
    case_name: str
    family: str
    size_bucket: str
    split: str
    blocked: bool = False
    blocked_reason: str = ""
    signature_match: bool = False
    node_count_match: bool = False
    edge_count_match: bool = False
    depth_match: bool = False
    evaluation_match: bool = False
    direct_purity_ok: bool = False
    posthoc_purity_ok: bool = False
    mismatch_details: list[str] = field(default_factory=list)

    @property
    def all_match(self) -> bool:
        if self.blocked:
            return False
        return all([
            self.signature_match, self.node_count_match, self.edge_count_match,
            self.depth_match, self.evaluation_match,
            self.direct_purity_ok, self.posthoc_purity_ok,
        ])


def audit_one(case: AuditCase) -> AuditResult:
    if case.blocked_reason:
        return AuditResult(
            case_name=case.name, family=case.family, size_bucket=case.size_bucket,
            split=case.split, blocked=True, blocked_reason=case.blocked_reason,
            mismatch_details=[case.blocked_reason],
        )

    mismatches: list[str] = []

    direct_graph, direct_root, _ = compile_with_stats(case.build_direct)
    posthoc_tree = case.build_posthoc()
    posthoc_graph = eml_to_dag(posthoc_tree)
    posthoc_root = posthoc_graph.roots[0]

    direct_sig = compute_signature(direct_graph, direct_root)
    posthoc_sig = compute_signature(posthoc_graph, posthoc_root)
    sig_match = direct_sig == posthoc_sig
    if not sig_match:
        mismatches.append(f"signature: direct={direct_sig!r} posthoc={posthoc_sig!r}")

    direct_stats = compute_statistics(direct_graph)
    posthoc_stats = compute_statistics(posthoc_graph)

    node_match = direct_stats.node_count == posthoc_stats.node_count
    if not node_match:
        mismatches.append(f"node_count: direct={direct_stats.node_count} posthoc={posthoc_stats.node_count}")

    edge_match = direct_stats.edge_count == posthoc_stats.edge_count
    if not edge_match:
        mismatches.append(f"edge_count: direct={direct_stats.edge_count} posthoc={posthoc_stats.edge_count}")

    depth_match = direct_stats.max_depth == posthoc_stats.max_depth
    if not depth_match:
        mismatches.append(f"depth: direct={direct_stats.max_depth} posthoc={posthoc_stats.max_depth}")

    direct_val = evaluate_dag(direct_graph, direct_root, case.eval_bindings)
    posthoc_val = evaluate_dag(posthoc_graph, posthoc_root, case.eval_bindings)
    eval_match = abs(direct_val - posthoc_val) < 1e-9
    if not eval_match:
        mismatches.append(f"evaluation: direct={direct_val} posthoc={posthoc_val}")

    direct_purity = validate_eml_purity(direct_graph).valid
    posthoc_purity = validate_eml_purity(posthoc_graph).valid
    if not direct_purity:
        mismatches.append("direct graph failed purity check")
    if not posthoc_purity:
        mismatches.append("posthoc graph failed purity check")

    return AuditResult(
        case_name=case.name, family=case.family, size_bucket=case.size_bucket, split=case.split,
        signature_match=sig_match, node_count_match=node_match, edge_count_match=edge_match,
        depth_match=depth_match, evaluation_match=eval_match,
        direct_purity_ok=direct_purity, posthoc_purity_ok=posthoc_purity,
        mismatch_details=mismatches,
    )


def run_audit(cases: list[AuditCase]) -> list[AuditResult]:
    return [audit_one(c) for c in cases]


# stratified audit set: covers both verified families (exp, ln), shallow
# and composed sizes, a repeated-subexpression case, and one deliberately
# blocked case proving gaps get reported, not hidden
STRATIFIED_AUDIT_SET: list[AuditCase] = [
    AuditCase(
        name="exp_shallow_core",
        family="exp", size_bucket="shallow", split="core",
        build_direct=lambda t: emit_exp(t, emit_var(t, "x")),
        build_posthoc=lambda: make_exp(EmlNode("Var", value="x")),
        eval_bindings={"x": 2.0},
    ),
    AuditCase(
        name="ln_shallow_core",
        family="ln", size_bucket="shallow", split="core",
        build_direct=lambda t: emit_ln(t, emit_var(t, "z")),
        build_posthoc=lambda: make_ln(EmlNode("Var", value="z")),
        eval_bindings={"z": 3.5},
    ),
    AuditCase(
        name="ln_shallow_domain_large",
        family="ln", size_bucket="shallow", split="domain",
        build_direct=lambda t: emit_ln(t, emit_var(t, "z")),
        build_posthoc=lambda: make_ln(EmlNode("Var", value="z")),
        eval_bindings={"z": 150.0},  # different magnitude - domain coverage
    ),
    AuditCase(
        name="ln_of_exp_medium_composed",
        family="composed", size_bucket="medium", split="composed",
        build_direct=lambda t: emit_ln(t, emit_exp(t, emit_var(t, "x"))),
        build_posthoc=lambda: make_ln(make_exp(EmlNode("Var", value="x"))),
        eval_bindings={"x": 1.3},
    ),
    AuditCase(
        name="exp_of_exp_medium_composed",
        family="composed", size_bucket="medium", split="composed",
        build_direct=lambda t: emit_exp(t, emit_exp(t, emit_var(t, "x"))),
        build_posthoc=lambda: make_exp(make_exp(EmlNode("Var", value="x"))),
        eval_bindings={"x": 0.4},  # kept small, exp(exp(x)) grows fast
    ),
    AuditCase(
        name="repeated_exp_deep_repeated",
        family="exp", size_bucket="deep", split="repeated",
        build_direct=lambda t: t.intern_binary(
            "eml", emit_exp(t, emit_var(t, "x")), emit_exp(t, emit_var(t, "x"))
        ),
        build_posthoc=lambda: EmlNode("eml", (
            make_exp(EmlNode("Var", value="x")), make_exp(EmlNode("Var", value="x"))
        )),
        eval_bindings={"x": 1.1},
    ),
    AuditCase(
        name="add_family_blocked",
        family="add", size_bucket="shallow", split="core",
        blocked_reason="add/mul/pow constructors not available yet - "
                       "pending 2-2/2-3/2-4's real compiler formulas",
    ),
]
