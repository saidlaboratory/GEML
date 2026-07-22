# Verified official trig and hyperbolic pure-EML constructions

## Authority and fixed variants

The constructions are a clean-room native implementation of the official EML
compiler and the identities in Part II of the supplementary information for
[*All elementary functions from a single operator*](https://arxiv.org/abs/2603.21852v2).
The implementation is pinned to official compiler commit
`b3da148261199b46247306dfd92068f589778260`; tests never fetch network content.

Two explicit variants are retained:

- `official_v4` exactly matches the pinned `eml_compiler_v4.py` emitter for the
  canonical atomic `f(x)` fixtures;
- `clean_negation` exactly matches the pinned companion clean-negation compiler
  for those same fixtures.

## Sourced formulas

The hyperbolic functions use the upstream explicit macros:

```text
sinh(z) = (exp(2z) - 1) / (2 exp(z))
cosh(z) = (exp(2z) + 1) / (2 exp(z))
tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
```

For trigonometric functions, the implementation freezes the official
SymPy-1.14 normalization and ordered Add/Mul folds:

```text
sin(z) = -i (exp(i z) - exp(-i z)) / 2
cos(z) =     (exp(i z) + exp(-i z)) / 2
tan(z) =  i (-exp(i z) + exp(-i z)) / (exp(i z) + exp(-i z))
```

Here exact `i` follows the supplement's reconstructed-log convention described
in `EML_ARITHMETIC_FORMULAS.md`; it is not a source `I` leaf. Every helper is
recursively compiled to primitive `1`. Finite-precision evaluation at the
internal negative-real branch cut can select the paired opposite signs, so
direct helper values are retained as diagnostics rather than treated as exact
identities.

## Exact offline conformance

SHA-256 is over the UTF-8 official-style `EML[left,right]` string for the
canonical atomic input variable `x`. Counts are expanded-tree occurrences;
leaves have depth zero. These fixtures prove exact conformance for `f(x)`.
Upstream first normalizes a complete SymPy source expression, whereas GEML's
constructors accept an already-lowered child; compound compositions are
formula-equivalent but are not claimed to be byte-identical to upstream
whole-source normalization.

| Operator | v4 SHA-256 | nodes | leaves | depth | clean SHA-256 | clean nodes | clean depth |
|---|---|---:|---:|---:|---|---:|---:|
| `sin` | `d9fa0e691922aee5a0c57ac75e101e63abf9b9ee7f56841ff1820a4aa8cd6571` | 799 | 400 | 63 | `d834c494688fdbfa764964a3762c02803afe3d76e26476d20b64d0ef545130a0` | 1,583 | 93 |
| `cos` | `1a0c5493b625c1fda4d4e4436e7e4c677eff304e01af1119e5c1c4fca530695f` | 687 | 344 | 55 | `d3b861ffd36f2c027eab1f6be6b7cbe8b3721e767b263f83894f00f7e7ba98b1` | 1,331 | 81 |
| `tan` | `20c4f5fa49f4f62955d507c1da34cd5d898e61bda5155f5c3a3e61982dcf45b1` | 1,183 | 592 | 75 | `7597f6360dcec6f277eb807fdd88dbab0868320afde639051a35ed1a7541938b` | 2,331 | 105 |
| `sinh` | `888c44fb76f939795e4943b500ad32c8cb223f903abc18bd008eede550213aa2` | 171 | 86 | 31 | `2f079ef578337e7fafb1a2633cbcbbfa691688bb6ea647fb92763939f8c50e95` | 311 | 45 |
| `cosh` | `f40e6f5e9d6bc3db56f7b0ebc20df2e747b107acf6a06b83874cf10b8ec2609e` | 187 | 94 | 31 | `33ead62ee5d3a657df52ca36e387561b59f78de1cfc3cb95c46652a8595a04ac` | 355 | 45 |
| `tanh` | `03aa8d0795c63db5c86c202342ce4f130a5b6858198fd76ec889cb3f79e0a943` | 157 | 79 | 28 | `3d7ca7a475154e7ed9c8b8139dad855c92666867d101fb9938294830cc91b36a` | 297 | 42 |

The strict validator proves only structural purity: result trees contain no
Add, Mul, Pow, Exp, Log, trig, hyperbolic, named-constant, macro, or compound
leaf nodes.

## Domain policy

- `sin`, `cos`, `sinh`, `cosh`, and `tanh` accept finite real source inputs.
- `tan` additionally requires `cos(argument) != 0`. Corpus generation must use
  a structural pole-safe argument proof; the agreed conservative policy is
  `|argument| <= 1`, strictly inside the nearest poles at `+-pi/2`.
- Principal complex `Log` is fixed for **internal compiler evaluation only**.
  This does not enable complex-valued source expressions or the source leaves
  `E`, `pi`, or `I`.

## Verification and limitations

The local audit has four independent layers:

1. exact pure-tree validation and the pinned fingerprints above;
2. symbolic exponential-identity diagnostics (explicitly not labelled proofs);
3. principal-complex `mpmath` evaluation at 100 decimal digits;
4. independent NumPy IEEE `complex128` evaluation.

Every requested probe produces a row containing its assumptions, method,
errors, and terminal status. Source-domain errors, overflow, nonfinite results,
mismatches, and extended-real intermediates are not dropped. In particular,
official-v4 `sinh(0)` and `tanh(0)` can end as `+inf` in the 100-digit mpmath
execution path after nonfinite intermediates, while NumPy's IEEE complex128 path
reaches the correct finite zero and records
`pass_with_extended_intermediate`. This backend/representation limitation is
retained as evidence rather than being renamed a successful high-precision
probe.

Historical regression note: the upstream verifier chain at revision
[`f5d4f3c`](https://github.com/VA00/SymbolicRegressionPackage/commit/f5d4f3cc92925a16d8b5b298630e79ccbd6c41d8)
had wrong-sign `sinh`/`tanh` witnesses for negative inputs; the
[upstream report](https://github.com/VA00/SymbolicRegressionPackage/issues/4#issuecomment-4273735397)
and [maintainer confirmation](https://github.com/VA00/SymbolicRegressionPackage/issues/4#issuecomment-4274333595)
identify the old `sinhEML` witness as incorrect. GEML does not use that witness:
it reproduces the later pinned v4 direct exponential formulas, and negative
probes at `-1` and `-0.5` are retained as sign-regression tests.

Near-pole tan points are a separate stress class and are never used to claim a
uniform error bound. Numerical agreement cannot establish global branch
correctness; it supports the sourced construction together with the exact
structural and provenance evidence.
