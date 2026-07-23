# GEML Goal 4 — E-Graph Rewrite Policy

## Purpose
Goal 4 specifies the **rewrite policy** governing the semantic
canonicalization stage of GEML.

This document defines:

- rewrite modes
- rule safety classifications
- provenance requirements
- reporting requirements
- resource limits
- forbidden implementation shortcuts

# Design Principles

1. Pure EML remains the canonical representation.
2. Goal 4 is an optional semantic optimization stage.
3. Every rewrite must preserve semantic equivalence.
4. Every rewrite must be reproducible.
5. Every rewrite must be traceable through provenance metadata.
6. Unsupported or failed rewrites must never be silently ignored.



# Rewrite Modes

## SAFE_REAL

The default rewrite mode.

Only identities valid over the supported real-number semantics may be
applied.

This mode explicitly excludes branch-sensitive identities and any rule
requiring additional assumptions.

Examples of excluded identities include:

- sqrt(x²) → x
- log(xy) → log(x)+log(y)
- asin(sin(x)) → x
- acos(cos(x)) → x
- x/x → 1


## POSITIVE_REAL_FORMAL

An optional rewrite mode.

Additional assumptions may be introduced only when they are explicitly
provided by the caller or verified by the semantic verifier.

Typical assumptions include

- x > 0
- y > 0
- x ≥ 0
- y ≠ 0

Every rewrite performed under this mode must record its assumptions.

# Rule Classification

## ALWAYS_SAFE

Universally valid identities.

No assumptions required.

Examples

- x + 0 → x
- x * 1 → x
- a + b → b + a



## GUARDED

Rules requiring explicit assumptions.

Example

sqrt(x²) → x

requires

x ≥ 0


## VERIFIED_GUARDED

Rules requiring

- explicit assumptions
- successful semantic verification

before acceptance.

## OPTIONAL

Sound rewrite rules disabled by default.

These may be enabled explicitly for experimental workflows.


## EXPERIMENTAL

Research-only rewrite rules.

These are never enabled by default and should not affect reproducible
benchmark results.



## UNSAFE

Known mathematically unsound rewrites.

These must never execute.

They exist only for documentation or testing.



## UNCLASSIFIED

Rules awaiting review.

These may not participate in optimization.



# Rule Provenance

Every successful rewrite must record:

- Rule ID
- Rule name
- Rewrite tier
- Rewrite mode
- Required assumptions
- Justification
- Whether semantic verification was required

This metadata enables complete reconstruction of every optimization.



# Saturation Reporting

Optimization reports must include

- iterations
- rewrites attempted
- rewrites applied
- extraction status
- saturation reached
- termination reason

Possible extraction states include

- success
- partial_success
- failed
- timeout
- node_limit
- iteration_limit



# Resource Limits

Every optimization run shall enforce

- maximum iterations
- maximum e-graph node count
- timeout
- optional memory limit

Optimization must terminate once any configured limit is exceeded.



# Reporting Policy

The reporting denominator is

attempted rewrites

This includes

- successful rewrites
- rejected rewrites
- unsupported rewrites
- failed rewrites

Nothing may be silently omitted.



# Extraction Requirements

Every extracted expression must

1. preserve semantic equivalence

2. compile through the official Pure EML compiler

3. be evaluated using the Goal 3 exact DAG cost model



# Forbidden Shortcuts

Goal 4 is an e-graph rewrite system.

General symbolic simplifiers must not be used as replacements for
documented rewrite rules.

Forbidden categories include

## Global simplification

- sympy.simplify
- sympy.nsimplify

## Algebraic rewriting

- sympy.expand
- sympy.factor
- sympy.cancel
- sympy.collect
- sympy.together
- sympy.apart

## Power simplification

- sympy.powsimp
- sympy.powdenest
- sympy.sqrtdenest
- sympy.radsimp

## Trigonometric simplification

- sympy.trigsimp
- sympy.expand_trig

## Logarithmic simplification

- sympy.expand_log
- sympy.logcombine

## Generic symbolic rewrite APIs

- Expr.rewrite
- Expr.replace
- Expr.subs
- Expr.xreplace

These APIs must not be used to bypass documented rewrite rules or rule
provenance.



# Branch and Domain Caveats

The following identities are **not universally valid**

- sqrt(x²)
- log(xy)
- asin(sin(x))
- acos(cos(x))
- atan(tan(x))
- x/x

These identities require explicit assumptions or semantic verification
before use.

They are excluded from SAFE_REAL mode.