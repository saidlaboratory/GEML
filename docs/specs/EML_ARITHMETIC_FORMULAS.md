# Official arithmetic and exact-number formulas

## Authority and scope

This module independently expresses the arithmetic constructions in the
official public `eml_compiler_v4.py` pinned by source-ledger entry
`EML-COMPILER-B3DA1482`:

- repository: `VA00/SymbolicRegressionPackage`;
- commit: `b3da148261199b46247306dfd92068f589778260`;
- path: `EML_toolkit/EmL_compiler/eml_compiler_v4.py`;
- Git blob: `ce697c7767221429d87876a3d04cdca5113ee75b`;
- retrieved-file SHA-256:
  `7b147b564a952c8ef24f1b1f6bb2b443a68ce44100ad019d5222df9547b6da62`;
- ledger retrieval date: 2026-07-20.

No upstream source is vendored. `EML_CORE_FORMULAS.md` defines the frozen core
macros used below. This issue does not parse source ASTs or choose source-domain
guards; those remain separate integration responsibilities.

## Public API and compiler modes

`geml.eml.compiler_arithmetic` is authoritative for these public constructors:

- `eml_inverse`, `eml_multiply`, `eml_divide`, and `eml_power`;
- `eml_integer`, `eml_rational`, and `eml_decimal`.

`geml.eml.compiler_constants` re-exports the three exact-number constructors as
the same Python objects. It does not contain a second implementation.

Every constructor defaults to `CompilerMode.OFFICIAL_V4`. The separately
labeled `CompilerMode.CLEAN_NEGATION` variant is opt-in and propagates through
every nested addition or negation. Its trees must never be reported as
official-v4 representations or mixed into official-v4 structural metrics.

## Exact constructions

Argument order is structural and must be preserved.

| Public constructor | Exact construction |
|---|---|
| `eml_inverse(z)` | `exp(negate(log(z)))` |
| `eml_multiply(a, b)` | `exp(add(log(a), log(b)))` |
| `eml_divide(a, b)` | `multiply(a, inverse(b))` |
| `eml_power(a, b)` | `exp(multiply(b, log(a)))` |

The power construction deliberately places the exponent before `log(base)`.
Swapping those child positions changes the pure-EML tree even where an external
algebraic interpretation might suggest commutativity.

### Integers

The exact integer constructor follows the pinned binary doubling/addition
algorithm:

1. `1` is one fresh primitive-one leaf.
2. `0` is `zero()` (therefore `log(1)`).
3. A negative integer is the selected-mode negation of its positive magnitude.
4. A positive integer scans the binary expansion from least-significant bit to
   most-significant bit. The current term starts at `1`, doubles with ordered
   `add(term, term)`, and contributes to an accumulator when its bit is set.

The implementation omits only the pinned loop's final doubled term after no
bits remain, because that term is discarded and cannot affect the returned
structure.

Each syntactic occurrence is copied into a fresh recursive IR object. The result
is a tree, not a DAG: neither binary doubling nor repeated public operands may
create shared children or multiple-parent structure.

### Rationals

`eml_rational(p, q)` accepts canonical exact input only:

- `p` and `q` are integers but never booleans;
- `q` is positive and nonzero;
- `gcd(abs(p), q) == 1`;
- zero is represented only as `0/1`;
- `q == 1` delegates directly to `eml_integer(p)`.

Otherwise the constructor builds
`multiply(integer(abs(p)), inverse(integer(q)))` and applies the selected-mode
negation only when `p < 0`. It does not silently normalize malformed or
noncanonical input.

### Decimals

`eml_decimal(value)` accepts only `str`, `Decimal`, or `float`, excluding
booleans. A non-`Decimal` input first becomes `Decimal(str(value))`; this is the
pinned decimal-string policy and intentionally avoids compiling a float's
hidden binary expansion. Finite values use `Decimal.as_integer_ratio()` and
delegate to the canonical rational constructor. Invalid, unsupported, NaN, and
infinite values fail explicitly.

## Pure-tree and structural contract

All outputs validate against the recursive pure-EML grammar only:

```text
term := 1 | source_variable | eml(term, term)
```

Helper names, arithmetic labels, named constants, numeric leaves other than
primitive `1`, and hidden operators never appear in final trees. Constructors
validate public term inputs, do not mutate them, and expand any physical input
reuse into fresh syntactic occurrences. A leaf has depth zero.

The pinned `OFFICIAL_V4` structures have these mandatory occurrence metrics:

| Construction | Nodes | Depth |
|---|---:|---:|
| `integer(0)` | 7 | 3 |
| `integer(1)` | 1 | 0 |
| `integer(-1)` | 17 | 7 |
| `integer(2)` | 27 | 9 |
| `integer(3)` | 53 | 13 |
| `rational(1, 2)` | 91 | 23 |
| `inverse(x)` | 25 | 8 |
| `multiply(x, y)` | 41 | 10 |
| `divide(x, y)` | 65 | 16 |
| `power(x, integer(2))` | 75 | 18 |

Tests lock these values and compare complete recursive structures against a
test-local literal oracle built only from the IR grammar. SHA-256 emission
fingerprints provide an additional regression signal. A separate test-local
principal-complex evaluator checks numeric probes and explicitly records
extended-real intermediates rather than hiding them.

## Domain and branch caveats

These constructors reproduce sourced structures; they do not prove unrestricted
complex identities. Inverse and division retain their nonzero requirements.
Power retains the registry's guarded base/exponent policy. Multiplication,
division, negation, and exact negative values can traverse logarithmic branch or
extended-real intermediates even when the final value is finite. Generator and
verification layers must preserve the governing real-domain assumptions and
retain failures or nonfinite observations.

The existing internal helpers `eml_internal_e`, `eml_internal_i_branch`, and
`eml_internal_pi` remain compiler-only compatibility dependencies for the
approved trig normalization. They lower completely to primitive `1` and do not
authorize named source leaves. Registry policy is unchanged:

- primitive `one`, integers, and rationals are approved;
- source `e` and `pi` leaves remain `pending_verification`;
- source `imaginary_unit` remains `reserved` with complex mode disabled;
- `GoldenRatio` is not an approved registry entry.

Issue 2-6 consumes these compiler outputs in the full domain-aware semantic
verifier; it does not own source AST dispatch. A separate integration layer must
map source AST nodes to these constructors, preserve ordered folds for
multi-child `Add` and `Mul`, propagate the selected mode, and enforce registry
domain guards. It must not silently enable pending constants.
