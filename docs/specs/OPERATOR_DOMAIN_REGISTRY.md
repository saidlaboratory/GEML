# Operator, domain, and corpus-family registry

The modules under `geml.spec` are the single machine-readable policy source for
Goal 1. They contain immutable metadata, not constructors, samplers, semantic
predicates, or EML formulas. `sympy_srepr` remains the authoritative expression
representation defined by the expression contract.

## Domain modes

| Mode | Generation | Policy summary |
|---|---:|---|
| `safe_real` | enabled | Finite real variables; log arguments are constructed positive, division denominators nonzero, and branch-sensitive power cases restricted. |
| `positive_real` | enabled | Variables are strictly positive; individual grammar productions still prove guarded operands rather than assuming every composite is positive. |
| `nonzero_real` | enabled | Variables are real and nonzero; composite denominators need an independent nonzero guarantee and logs still need positivity. |
| `complex` | disabled | Reserved future principal-branch policy pending explicit branch, singularity, and verification rules. |

Numeric probe notes are policy metadata only. Later verification must report
invalid points, timeouts, unsupported cases, and validation errors rather than
silently removing them.

## Operator fields and statuses

Each operator record has a stable name and arity, a SymPy structural encoding,
an operator-family ID, allowed domain modes, a generation flag, an EML
construction status, source-ledger IDs, and encoding notes. Arity counts
expression children; integer and rational values are leaf payload and therefore
have arity zero.

Statuses mean:

- `approved`: the pinned primary evidence supports source generation and the
  construction within the declared real-domain guards and the paper's
  almost-everywhere/extended-complex scope. It does not promise that every
  finite-precision pure-EML execution path is total at every real point;
- `pending_verification`: a candidate construction exists but project-level
  domain or mathematical verification remains open;
- `unsupported`: evidence shows the operation is outside the approved policy;
- `reserved`: intentionally held for a future project/domain expansion.

The non-negotiable generation gate is:

```text
enabled_for_generation implies eml_construction_status == approved
```

Pending, unsupported, and reserved entries cannot enter result-bearing data.

## SymPy encoding policy

SymPy uses `Add`, `Mul`, and `Pow` rather than dedicated source classes for
subtraction, division, and negation. The registry consequently records their
lowering explicitly: subtraction is an `Add` with a negated right operand,
division is a `Mul` with the denominator raised to exact `-1`, and negation is a
`Mul` by exact `-1`. Constructors should use `evaluate=False` where specified to
preserve intended source structure. No custom SymPy node type is authorized.

Power is approved only as a bounded, exact-exponent production: non-integer
exponents require a positive base and negative exponents require a nonzero
base. Every real-mode `log` argument must come from a positive-expression
grammar; `nonzero_real` alone is not sufficient.

All six trigonometric and hyperbolic operators are approved for real source
arguments by the pinned official compiler. This is a source-corpus eligibility
decision plus official construction evidence—not a claim of strict pointwise
totality for every expanded pure-EML execution. The paper permits isolated
exceptional/extended-real paths; local verification retains them explicitly
(including the official-v4 `sinh(0)` and `tanh(0)` high-precision outcomes)
rather than silently treating them as passes. Internal complex intermediates do
not enable complex source variables or the `I` leaf. To keep generated `tan`
expressions uniformly away from real poles, every argument is structurally
certified to lie in the closed interval `[-1, 1]`; the generator records the
certificate class for each occurrence.

## Operator table

| Operator | Arity | SymPy encoding | Family | Domains | Generation | EML status |
|---|---:|---|---|---|---:|---|
| `symbol` | 0 | `Symbol` with mode assumption | `leaf` | three real modes | yes | `approved` |
| `one` | 0 | `Integer(1)` | `source_constant` | three real modes | yes | `approved` |
| `integer` | 0 | `Integer(value)` | `exact_number` | three real modes | yes | `approved` |
| `rational` | 0 | `Rational(p, q)` | `exact_number` | three real modes | yes | `approved` |
| `add` | 2 | `Add(left, right, evaluate=False)` | `arithmetic` | three real modes | yes | `approved` |
| `subtract` | 2 | lowered `Add`/`Mul(-1, ...)` | `arithmetic` | three real modes | yes | `approved` |
| `multiply` | 2 | `Mul(left, right, evaluate=False)` | `arithmetic` | three real modes | yes | `approved` |
| `divide` | 2 | lowered `Mul`/`Pow(..., -1)` | `arithmetic` | three real modes | yes | `approved` |
| `negate` | 1 | lowered `Mul(-1, operand)` | `arithmetic` | three real modes | yes | `approved` |
| `power` | 2 | `Pow(base, exponent, evaluate=False)` | `power` | three real modes | yes | `approved` |
| `exp` | 1 | `exp(argument, evaluate=False)` | `exp_log` | three real modes | yes | `approved` |
| `log` | 1 | `log(argument, evaluate=False)` | `exp_log` | three real modes | yes | `approved` |
| `sin` | 1 | `sin(argument, evaluate=False)` | `trigonometric` | three real modes | yes | `approved` |
| `cos` | 1 | `cos(argument, evaluate=False)` | `trigonometric` | three real modes | yes | `approved` |
| `tan` | 1 | `tan(argument, evaluate=False)` | `trigonometric` | three real modes | yes | `approved` |
| `sinh` | 1 | `sinh(argument, evaluate=False)` | `hyperbolic` | three real modes | yes | `approved` |
| `cosh` | 1 | `cosh(argument, evaluate=False)` | `hyperbolic` | three real modes | yes | `approved` |
| `tanh` | 1 | `tanh(argument, evaluate=False)` | `hyperbolic` | three real modes | yes | `approved` |
| `e` | 0 | `E` | `source_constant` | three real modes | no | `pending_verification` |
| `pi` | 0 | `pi` | `source_constant` | three real modes | no | `pending_verification` |
| `imaginary_unit` | 0 | `I` | `source_constant` | `complex` | no | `reserved` |

There are currently no `unsupported` registry entries. The status remains part
of the closed vocabulary so future negative evidence can be recorded without a
schema change.

The six trig/hyperbolic entries are approved from the pinned official compiler's
construction and test coverage under the declared real-source guards. `e` and
`pi` remain pending source leaves because the project has not approved them for
Goal 1. `imaginary_unit` is reserved because source-complex support is outside
the current pipeline.

## Final corpus families

| Family | Quota | Kind | Eligible scope | Generation state |
|---|---:|---|---|---|
| `algebraic_core` | 70,000 | IID | exact leaves and algebraic core | ready |
| `powers_division_rationals` | 40,000 | IID | algebraic core, rationals, division, bounded power | ready |
| `exp_log` | 40,000 | IID | preceding operators plus `exp` and guarded `log` | ready |
| `trig_hyperbolic` | 40,000 | IID | algebraic core plus approved trig/hyperbolic operators | ready |
| `mixed_elementary` | 35,000 | IID | algebraic, power, exp/log, trig, and hyperbolic | ready |
| `ood_stress` | 25,000 | OOD policy | enabled leaf, exact-number, arithmetic, power, and exp/log families under a declared stress criterion | ready |
| **Total** | **250,000** |  |  |  |

`ood_stress` is a policy family, not an ordinary operator family. Its consumer
must declare the held-out depth, size, variable-count, or composition criterion;
split assignment remains outside this specification.

Quotas are final targets and are never redistributed to bypass pending
approvals. `require_family_generation_ready` exposes a clear preflight gate and
lists every blocking operator.

## Compiler-verification boundary

Goal 1 approval records that the pinned official compiler can lower these six
real-source operators; it does not copy formulas into the generator. The local
EML compiler/verifier is a separate evidence layer with structural, exact
fingerprint, symbolic-diagnostic, and two-backend numeric checks with complete
failure accounting. It neither widens the source vocabulary nor converts an
isolated EML evaluation exception into a source-domain restriction. Complex
source support and optional constants remain separate decisions requiring
explicit domain policy and ledger updates; no generator may carry a hidden
formula or approval override.
