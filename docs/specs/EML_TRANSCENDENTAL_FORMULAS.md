# Verified official trig and hyperbolic pure-EML constructions

## Authority and provenance

These clean-room constructions use two public authorities recorded in
`EML_SOURCE_LEDGER.md`:

- `EML-PAPER-2603.21852-V2`, Part II of the supplementary information for
  [*All elementary functions from a single binary operator*](https://arxiv.org/abs/2603.21852v2);
- `EML-COMPILER-B3DA1482`, the official public
  [`eml_compiler_v4.py`](https://github.com/VA00/SymbolicRegressionPackage/blob/b3da148261199b46247306dfd92068f589778260/EML_toolkit/EmL_compiler/eml_compiler_v4.py)
  at immutable commit `b3da148261199b46247306dfd92068f589778260`.

At the 2026-07-20 ledger retrieval, the compiler file had Git blob
`ce697c7767221429d87876a3d04cdca5113ee75b` and retrieved-file SHA-256
`7b147b564a952c8ef24f1b1f6bb2b443a68ce44100ad019d5222df9547b6da62`.
The hyperbolic functions are direct upstream macros. The trig functions are
instead the stable result of the upstream compiler's complete-expression
normalization through SymPy 1.14 `log`, `exp`, and `Pow`, followed by its
ordered `Add` and `Mul` folds. The relevant SymPy structure authority is pinned
as `SYMPY-1.14-STRUCTURE` in the source ledger.

## Public API and compiler modes

The public constructors and immutable dispatch maps are:

```text
compiler_trig.py:
  eml_sin, eml_cos, eml_tan, TRIG_COMPILERS

compiler_transcendental.py:
  eml_sinh, eml_cosh, eml_tanh, HYPERBOLIC_COMPILERS
```

Every constructor defaults to `CompilerMode.OFFICIAL_V4`.
`CompilerMode.CLEAN_NEGATION` is an explicit opt-in variant derived from the
public companion `eml_compiler_clean_math_v0.py` at the same commit
(retrieved-file SHA-256
`85be0e39271856dfa26c6368ba031f5eb83aa35a7beea6aba472e5dca6448f03`).
The modes propagate through all core, arithmetic, exact-number, and internal
constant helpers. Clean-negation output is always labeled separately and must
not be mixed into official-v4 fingerprints or structural metrics.

## Pure grammar and source boundary

Every returned tree is an expanded tree—not a shared-child DAG—in the grammar:

```text
P ::= variable | 1 | eml(P, P)
```

There are no final `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `exp`, `log`,
`Add`, `Mul`, `Div`, `Pow`, integer/rational, named-constant, macro, or hidden
compound leaves. Repeated mathematical subexpressions are expanded so each
node has at most one parent.

Trig formulas obtain an internal imaginary-unit branch from the frozen
`eml_internal_i_branch` construction documented in
`EML_ARITHMETIC_FORMULAS.md`. It is a pure subtree derived from primitive `1`,
not a source `I` leaf. Internal compiler use of this branch does not approve
complex-valued source expressions or source leaves `I`, `E`, or `pi`.

The frozen registry state is deliberately unchanged:

| Source constant | EML status | Source generation |
|---|---|---|
| `e` | `pending_verification` | disabled |
| `pi` | `pending_verification` | disabled |
| `imaginary_unit` | `reserved` | disabled |

## Direct hyperbolic formulas and order

The private doubling step is exactly `eml_add(z, z, mode=mode)`; no `Double`
node exists. The direct pinned formulas are:

```text
sinh(z) = (exp(2z) - 1) / (2 exp(z))
cosh(z) = (exp(2z) + 1) / (2 exp(z))
tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
```

For `sinh` and `cosh`, construction first creates `exp(2z)`, then `exp(z)`,
then the ordered denominator `multiply(integer(2), exp(z))`, then the
subtraction/addition numerator, and finally `divide(numerator, denominator)`.
For `tanh`, `exp(2z)` is created once conceptually, the ordered
`subtract(exp(2z), 1)` numerator and `add(exp(2z), 1)` denominator are formed,
and the numerator is divided by the denominator. Pure arithmetic composition
expands any repeated occurrence in the returned tree.

## Normalization-derived trig formulas and order

All three trig constructors first form these shared conceptual terms in order:

```text
i
-1
exp(i * z)
exp(((-1) * i) * z)
```

The factor nesting is significant: the negative exponent is
`multiply(multiply(-1, i), z)`, not a commuted or reassociated equivalent.
The pinned formulas and project fold orders are:

```text
sin(z) = -(i/2) (exp(i z) - exp(-i z))
  difference  = add(multiply(-1, exp(-i z)), exp(i z))
  coefficient = multiply(rational(-1, 2), i)
  result       = multiply(coefficient, difference)

cos(z) = (exp(-i z) + exp(i z)) / 2
  negative_term = multiply(rational(1, 2), exp(-i z))
  positive_term = multiply(rational(1, 2), exp(i z))
  result        = add(negative_term, positive_term)

tan(z) = i (exp(-i z) - exp(i z)) / (exp(-i z) + exp(i z))
  denominator = add(exp(-i z), exp(i z))
  reciprocal  = power(denominator, -1)
  numerator   = add(exp(-i z), multiply(-1, exp(i z)))
  result      = multiply(multiply(i, reciprocal), numerator)
```

These orders are structural requirements. Commuting source-equivalent factors
or terms produces a different pure-EML byte string and fingerprint.

## Exact atomic offline conformance

SHA-256 is over the UTF-8 official-style `EML[left,right]` emission for the
canonical atomic input `Variable("x")`. Counts are expanded-tree occurrences,
leaves have depth zero, and every row has `reused_object_count == 0`.

| Operator | official-v4 SHA-256 | nodes | leaves | depth | clean-negation SHA-256 | nodes | leaves | depth |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `sin` | `d9fa0e691922aee5a0c57ac75e101e63abf9b9ee7f56841ff1820a4aa8cd6571` | 799 | 400 | 63 | `d834c494688fdbfa764964a3762c02803afe3d76e26476d20b64d0ef545130a0` | 1,583 | 792 | 93 |
| `cos` | `1a0c5493b625c1fda4d4e4436e7e4c677eff304e01af1119e5c1c4fca530695f` | 687 | 344 | 55 | `d3b861ffd36f2c027eab1f6be6b7cbe8b3721e767b263f83894f00f7e7ba98b1` | 1,331 | 666 | 81 |
| `tan` | `20c4f5fa49f4f62955d507c1da34cd5d898e61bda5155f5c3a3e61982dcf45b1` | 1,183 | 592 | 75 | `7597f6360dcec6f277eb807fdd88dbab0868320afde639051a35ed1a7541938b` | 2,331 | 1,166 | 105 |
| `sinh` | `888c44fb76f939795e4943b500ad32c8cb223f903abc18bd008eede550213aa2` | 171 | 86 | 31 | `2f079ef578337e7fafb1a2633cbcbbfa691688bb6ea647fb92763939f8c50e95` | 311 | 156 | 45 |
| `cosh` | `f40e6f5e9d6bc3db56f7b0ebc20df2e747b107acf6a06b83874cf10b8ec2609e` | 187 | 94 | 31 | `33ead62ee5d3a657df52ca36e387561b59f78de1cfc3cb95c46652a8595a04ac` | 355 | 178 | 45 |
| `tanh` | `03aa8d0795c63db5c86c202342ce4f130a5b6858198fd76ec889cb3f79e0a943` | 157 | 79 | 28 | `3d7ca7a475154e7ed9c8b8139dad855c92666867d101fb9938294830cc91b36a` | 297 | 149 | 42 |

This byte-conformance claim is limited to the canonical atomic fixtures.
Upstream first normalizes a complete SymPy source expression, while GEML's
public constructors accept an already-lowered pure-EML child. Compound-child
composition follows the sourced formulas and is tested for purity and
determinism, but it is not claimed byte-identical to upstream whole-source
normalization.

## Domain and branch policy

The current registry permits each of the six source operators in
`safe_real`, `positive_real`, and `nonzero_real` modes. Source inputs remain
finite real values. `cosh` is positive on real inputs, but that property does
not widen any source operator or domain entry. Internal principal-complex
`Log` evaluation is a compiler concern only and does not establish unrestricted
complex-domain correctness.

`tan` additionally requires `cos(argument) != 0`. The constructor composes a
tree; it does not prove that precondition. Corpus generation must provide the
structural pole-safety proof, currently the conservative
`|argument| <= 1`, strictly inside the nearest real poles at `+-pi/2`.
Near-pole probes are a separate stress diagnostic and are never used to claim
a uniform error bound.

Finite-precision evaluation at the internal negative-real branch cut can
select paired opposite signs. Exact structural conformance and source
provenance are therefore the primary evidence, not isolated intermediate
values.

## Numerical, symbolic, and regression evidence

The read-only local verifier supplies four separately labeled layers:

1. exact fingerprints plus strict pure-tree validation;
2. SymPy exponential-identity diagnostics, explicitly not proofs;
3. principal-complex `mpmath` evaluation at 100 decimal digits;
4. independent NumPy IEEE `complex128` evaluation.

Every requested operator/point/backend retains a result row and terminal
status. Domain errors, overflow, nonfinite results, mismatches, and
extended-real intermediates are not dropped, clamped, or relabeled. Sample
agreement supports the sourced formulas; it is not a global branch proof.

One retained backend limitation is that official-v4 `sinh(0)` and `tanh(0)`
can terminate at `+inf` in the 100-digit mpmath path after nonfinite
intermediates. NumPy complex128 reaches the correct finite zero while recording
`pass_with_extended_intermediate`. Both outcomes remain visible.

Negative probes at `-1` and `-0.5` explicitly verify negative `sinh` and
`tanh` outputs. They guard against obsolete wrong-sign witnesses from an older
upstream verifier; GEML uses the later pinned direct formulas, not those
witnesses. Tan evidence is stratified into ordinary corpus-safe points,
near-pole diagnostics, and exact-pole source-domain errors.

## Stress-size observations

All clean-negation atomic fixtures remain below 2,500 nodes and depth 128; the
largest is `tan(x)` at 2,331 nodes and depth 105. Composing the small pure child
`eml_exp(x)` remains deterministic and strictly pure in both modes:

| Operator | official nodes/leaves/depth | clean nodes/leaves/depth |
|---|---|---|
| `sin` | 803 / 402 / 63 | 1,587 / 794 / 93 |
| `cos` | 691 / 346 / 55 | 1,335 / 668 / 81 |
| `tan` | 1,191 / 596 / 75 | 2,339 / 1,170 / 105 |
| `sinh` | 177 / 89 / 31 | 317 / 159 / 45 |
| `cosh` | 193 / 97 / 31 | 361 / 181 / 45 |
| `tanh` | 165 / 83 / 28 | 305 / 153 / 42 |

These bounded fixtures are CI evidence only; this issue does not materialize
production-scale corpus trees or replace repeated subtrees with macros.

Issue 2-6 owns the broader domain-aware verifier. This issue's audit is
deliberately scoped to exact construction conformance, strict purity, bounded
stress fixtures, and honest row-complete diagnostics for the six approved
operators.
