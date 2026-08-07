from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RewriteMode(StrEnum):
    SAFE_REAL = "safe_real"
    POSITIVE_REAL_FORMAL = "positive_real_formal"


class RuleTier(StrEnum):
    ALWAYS_SAFE = "always_safe"
    GUARDED = "guarded"
    VERIFIED_GUARDED = "verified_guarded"
    OPTIONAL = "optional"
    EXPERIMENTAL = "experimental"
    UNSAFE = "unsafe"
    UNCLASSIFIED = "unclassified"


class ExtractionStatus(StrEnum):
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
    assumptions: frozenset[str] = field(default_factory=frozenset)
    justification: str = ""
    enabled_in: frozenset[RewriteMode] = field(default_factory=frozenset)
    branch_sensitive: bool = False
    verifier_required: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("rule_id must be a non-blank string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("rule name must be a non-blank string")
        if not isinstance(self.tier, RuleTier):
            raise TypeError("tier must be a RuleTier")
        if not isinstance(self.assumptions, frozenset) or any(
            not isinstance(value, str) or not value.strip() for value in self.assumptions
        ):
            raise TypeError("assumptions must be a frozenset of non-blank strings")
        if not isinstance(self.enabled_in, frozenset) or any(
            not isinstance(mode, RewriteMode) for mode in self.enabled_in
        ):
            raise TypeError("enabled_in must be a frozenset of RewriteMode values")
        if type(self.branch_sensitive) is not bool or type(self.verifier_required) is not bool:
            raise TypeError("rule policy flags must be booleans")


@dataclass(frozen=True)
class ResourceLimits:
    max_iterations: int = 10000
    max_egraph_nodes: int = 1_000_000
    timeout_seconds: float = 60.0
    max_memory_mb: int | None = None
    max_rewrite_attempts: int = 1_000_000

    def __post_init__(self) -> None:
        for name in ("max_iterations", "max_egraph_nodes", "max_rewrite_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int | float)
            or self.timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be a nonnegative finite number")
        if self.timeout_seconds != self.timeout_seconds or self.timeout_seconds == float("inf"):
            raise ValueError("timeout_seconds must be a nonnegative finite number")
        if self.max_memory_mb is not None and (
            type(self.max_memory_mb) is not int or self.max_memory_mb < 1
        ):
            raise ValueError("max_memory_mb must be null or a positive integer")


@dataclass(frozen=True)
class SaturationReport:
    iterations: int
    rewrites_attempted: int
    rewrites_applied: int
    status: ExtractionStatus
    saturated: bool
    reason: str = ""

    def __post_init__(self) -> None:
        for name in ("iterations", "rewrites_attempted", "rewrites_applied"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.rewrites_applied > self.rewrites_attempted:
            raise ValueError("rewrites_applied cannot exceed rewrites_attempted")
        if not isinstance(self.status, ExtractionStatus):
            raise TypeError("status must be an ExtractionStatus")
        if type(self.saturated) is not bool:
            raise TypeError("saturated must be a bool")
        if self.saturated and self.status is not ExtractionStatus.SUCCESS:
            raise ValueError("only a successful saturation report may be saturated")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")


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
