from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Optional


class RewriteMode(str, Enum):

    SAFE_REAL = "safe_real"
    POSITIVE_REAL_FORMAL = "positive_real_formal"


class RuleTier(str, Enum):

    ALWAYS_SAFE = "always_safe"
    GUARDED = "guarded"
    VERIFIED_GUARDED = "verified_guarded"
    OPTIONAL = "optional"
    EXPERIMENTAL = "experimental"
    UNSAFE = "unsafe"
    UNCLASSIFIED = "unclassified"


class ExtractionStatus(str, Enum):

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NODE_LIMIT = "node_limit"
    ITERATION_LIMIT = "iteration_limit"


@dataclass(frozen=True)
class RulePolicy:

    rule_id: str
    name: str
    tier: RuleTier
    assumptions: FrozenSet[str] = field(default_factory=frozenset)
    justification: str = ""
    enabled_in: FrozenSet[RewriteMode] = field(default_factory=frozenset)
    branch_sensitive: bool = False
    verifier_required: bool = True


@dataclass(frozen=True)
class ResourceLimits:

    max_iterations: int = 10000
    max_egraph_nodes: int = 1_000_000
    timeout_seconds: int = 60
    max_memory_mb: Optional[int] = None


@dataclass(frozen=True)
class SaturationReport:

    iterations: int
    rewrites_attempted: int
    rewrites_applied: int
    status: ExtractionStatus
    saturated: bool
    reason: str = ""

FORBIDDEN_SHORTCUTS = {
    "global_simplification": (
        "sympy.simplify",
        "sympy.nsimplify",
    ),

    "algebraic": (
        "sympy.expand",
        "sympy.factor",
        "sympy.cancel",
        "sympy.collect",
        "sympy.together",
        "sympy.apart",
    ),

    "power_and_radicals": (
        "sympy.powsimp",
        "sympy.powdenest",
        "sympy.sqrtdenest",
        "sympy.radsimp",
    ),

    "trigonometric": (
        "sympy.trigsimp",
        "sympy.expand_trig",
    ),

    "logarithmic": (
        "sympy.expand_log",
        "sympy.logcombine",
    ),

    "generic_rewrite": (
        "Expr.rewrite",
        "Expr.replace",
        "Expr.subs",
        "Expr.xreplace",
    ),
}