# GEML Goal 4 - E-Graph Domain Rules

This document specifies the guarded, domain-restricted rewrite rules implemented in
`geml.egraph.rules_domain`. It complements `EGRAPH_SEMANTICS.md`, which defines the
rewrite modes, safety tiers, provenance requirements, and resource policy that this
document assumes.

## Scope warning

**No identity in this document is a universal identity of the complex logarithm, the
complex exponential, or the complex power function.** Each one is domain restricted.

Every rule here is conditional. Each is stated together with the assumption it requires
and at least one concrete counterexample showing what breaks when the assumption is
dropped. A reader looking for assumption-free identities should read
`geml.egraph.rules_safe` instead; that module and this one are disjoint by construction.

These rules are **not** enabled in `SAFE_REAL` mode. They run only when:

1. the rewrite mode is `POSITIVE_REAL_FORMAL`, and
2. a guard confirms the caller explicitly declared the required assumption.

Both conditions must hold. Setting the mode alone changes nothing; an undeclared variable
causes every guarded rule to decline.

## Assumption model

Assumptions are **declared, never inferred**. The caller supplies an
`AssumptionEnvironment` naming source variables and the properties it asserts about them:

```python
AssumptionEnvironment.of(x=("positive",), y=("nonzero",))
```

The environment is closed under these implications and no others:

| Declared | Implies |
| --- | --- |
| `positive` | `positive`, `nonnegative`, `nonzero`, `real` |
| `nonnegative` | `nonnegative`, `real` |
| `nonzero` | `nonzero`, `real` |
| `real` | `real` |

No property is derived from the syntax of an expression. In particular the engine does
**not** conclude that `exp(x)` is positive, that `x**2` is nonnegative, or that a sum of
positive terms is positive. Each of those is a true statement about real numbers, and each
is deliberately outside the guard's reach, because admitting one opens the question of
where syntactic inference stops.

### Guards are local

A guard inspects only the e-class **directly bound by the match**. It accepts on exactly
two grounds:

- the class denotes an exact rational constant satisfying the assumption, or
- the class denotes a source variable the caller declared to satisfy it.

A consequence worth stating plainly: `log(exp(x + y))` does not rewrite even when both `x`
and `y` are declared real, because the bound class is the compound sum `x + y`, not a
declared variable. This is conservative, not sound-critical - the guard declines rewrites
that are in fact valid.

Every decline is recorded in the provenance log as a `guard_rejected` attempt with the
guard's name. Declines are counted in the "attempted rewrites" denominator required by
`EGRAPH_SEMANTICS.md`. Nothing is silently skipped.

## Rule catalogue

### Default guarded rules

These are returned by `domain_rules()`.

#### `DOMAIN-LOG-EXP` - logarithm of an exponential

```
log(exp(x)) -> x        requires: x is real
```

Tier `GUARDED`, branch sensitive.

On the real line this identity is unconditional. It is fenced here because the principal
complex logarithm satisfies `log(exp(x)) = x` only when `Im(x)` lies in `(-pi, pi]`.

**Counterexample.** With `x = 3*i*pi`, the principal branch gives `log(exp(3*i*pi)) =
log(-1) = i*pi`, not `3*i*pi`.

The `EML_SOURCE_LEDGER.md` entry for `EML-PAPER-2603.21852-V2` records (Section 4.1) that
the EML construction admits complex intermediates and that principal-log branches require
care. A rewrite that is safe on the reals is therefore not automatically safe along the
compilation route GEML uses, which is why this rule requires an explicit declaration.

#### `DOMAIN-EXP-LOG` - exponential of a logarithm

```
exp(log(x)) -> x        requires: x > 0
```

Tier `GUARDED`, branch sensitive.

**Counterexample.** At `x = 0` the left side is undefined. At `x = -1` the left side is not
real valued, while the right side is `-1`; applying the rewrite would silently extend the
domain of the expression.

#### `DOMAIN-LOG-PRODUCT` - logarithm of a product

```
log(a*b) <-> log(a) + log(b)        requires: a > 0 and b > 0
```

Tier `GUARDED`, branch sensitive, bidirectional.

**Counterexample.** With `a = b = -1`, the left side is `log(1) = 0`. The right side is
undefined over the reals; on the principal complex branch it is `i*pi + i*pi = 2*i*pi`.

Both orientations carry the same guard. The backward orientation is not a weaker
requirement: combining two logarithms of possibly negative arguments is unsound for the
same reason.

#### `DOMAIN-EXP-SUM` - exponential of a sum

```
exp(a + b) <-> exp(a) * exp(b)      requires: a is real and b is real
```

Tier `GUARDED`, **not** branch sensitive, bidirectional.

This identity holds for all finite complex arguments - the exponential is a group
homomorphism from addition to multiplication - so it carries `branch_sensitive=False`.

It is nonetheless fenced to the formal mode because extended-real arguments break it.
**Counterexample.** `exp(inf + (-inf))` is undefined, while `exp(inf) * exp(-inf)` is an
indeterminate product `inf * 0`. Requiring a declared finite real argument keeps the rule
inside the domain where both sides agree.

#### `DOMAIN-LOG-POW` - logarithm of a power

```
log(a**b) -> b * log(a)             requires: a > 0
```

Tier `GUARDED`, branch sensitive.

**Counterexample.** With `a = -1` and `b = 2`, the left side is `log(1) = 0` while the
right side is `2 * log(-1)`, which is not real valued.

### Optional rules

These are excluded from `domain_rules()` by default and are returned only by
`domain_rules(include_optional=True)`. Per `EGRAPH_SEMANTICS.md`, tier `OPTIONAL` means
"sound under its guard, but disabled by default so that reproducible benchmark results are
not affected without an explicit decision."

#### `DOMAIN-DIV-SELF` - a value divided by itself

```
x / x -> 1                          requires: x != 0
```

Tier `OPTIONAL`, not branch sensitive.

**Counterexample.** At `x = 0` the quotient is undefined while the right side is `1`. The
rewrite would extend the domain of the expression, which changes what the expression
denotes even though it never produces a wrong finite value.

#### `DOMAIN-POW-POW` - a power raised to a power

```
(a**b)**c -> a**(b*c)               requires: a > 0
```

Tier `OPTIONAL`, branch sensitive.

**Counterexample.** With `a = -1`, `b = 2`, `c = 1/2`, the left side is
`((-1)**2)**(1/2) = 1**(1/2) = 1` while the right side is `(-1)**1 = -1`.

#### `DOMAIN-POW-MUL` - a product of powers of one base

```
a**b * a**c <-> a**(b + c)          requires: a > 0
```

Tier `OPTIONAL`, branch sensitive, bidirectional.

**Counterexample.** For a negative base and fractional exponents the individual factors
need not be real valued, so the two sides do not agree as real-valued expressions.

## Provenance requirements

Every application of a rule in this module records, through
`geml.egraph.provenance.ProvenanceLog`:

- the rule identifier, name, and safety tier,
- the rewrite mode the run executed under,
- the orientation applied (`forward` or `backward`),
- the guard outcome (`not_required`, `passed`, or `failed`),
- the assumption set declared by the rule policy,
- the `branch_sensitive` flag,
- the source and result e-classes and the substitution.

`ProvenanceLog.assumptions_used()` returns the union of assumptions relied on by rewrites
that actually changed the e-graph. A run whose log reports an empty assumption set applied
no domain rule, regardless of the mode it was configured with.

## Reproducibility

Rule construction is static and ordered, guards are pure functions of the e-graph and the
declared environment, and no rule consults wall-clock time or randomness. Two runs over
the same expression, mode, and assumption environment produce identical e-graph
signatures, identical saturation reports, and identical provenance record sequences.

## What is not implemented

The following are named in `EGRAPH_SEMANTICS.md` as branch sensitive and are **not**
implemented in any mode, because the Goal 4 operator vocabulary has no trigonometric
operators:

- `sqrt(x**2) -> x`
- `asin(sin(x)) -> x`
- `acos(cos(x)) -> x`
- `atan(tan(x)) -> x`

If trigonometric operators are added to `geml.egraph.ir.Operator`, each of these requires
its own entry in this document, with its assumption and counterexample, before any
implementation.
