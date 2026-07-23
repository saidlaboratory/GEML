# Official core pure-EML formulas

## Authority and representation

The authoritative implementation evidence is source-ledger entry
`EML-COMPILER-B3DA1482`: repository `VA00/SymbolicRegressionPackage`, path
`EML_toolkit/EmL_compiler/eml_compiler_v4.py`, immutable commit
`b3da148261199b46247306dfd92068f589778260`. At the ledger retrieval date,
2026-07-20, that file had Git blob SHA `ce697c7767221429d87876a3d04cdca5113ee75b`
and retrieved-file SHA-256
`7b147b564a952c8ef24f1b1f6bb2b443a68ce44100ad019d5222df9547b6da62`.

The primitive is

\[
F(a,b)=\operatorname{EML}(a,b)=\exp(a)-\operatorname{Log}(b).
\]

Every result obeys the pure grammar

```text
P ::= variable | 1 | EML[P,P]
```

Python names such as `eml_log` and `eml_add` are compile-time constructors,
not result nodes. The repository's compatibility API names the primitive
constructor `primitive` and the subtraction and negation constructors
`eml_subtract` and `eml_negate`.

## Exact OFFICIAL_V4 constructions

`CompilerMode.OFFICIAL_V4` is the default and the sole representation for the
official raw pure-EML baseline. Its formulas are the pinned v4 definitions:

| Python constructor | Exact construction | Fully expanded example |
|---|---|---|
| `primitive(a,b)` | `F(a,b)` | `EML[a,b]` |
| `eml_exp(x)` | `F(x,1)` | `EML[x,1]` |
| `eml_log(x)` | `F(1,F(F(1,x),1))` | `EML[1,EML[EML[1,x],1]]` |
| `eml_zero()` | `eml_log(1)` | `EML[1,EML[EML[1,1],1]]` |
| `eml_subtract(a,b)` | `F(eml_log(a),eml_exp(b))` | `EML[EML[1,EML[EML[1,a],1]],EML[b,1]]` |
| `eml_negate(x)` | `eml_subtract(eml_zero(),x)` | `EML[EML[1,EML[EML[1,EML[1,EML[EML[1,1],1]]],1]],EML[x,1]]` |
| `eml_add(a,b)` | `eml_subtract(a,eml_negate(b))` | `EML[EML[1,EML[EML[1,a],1]],EML[EML[EML[1,EML[EML[1,EML[1,EML[EML[1,1],1]]],1]],EML[b,1]],1]]` |

For positive finite inputs, direct algebra gives

\[
F(x,1)=\exp(x),
\]

\[
F(1,F(F(1,x),1))
=e-\log(\exp(e-\log x))
=\log x,
\]

and therefore

\[
F(\operatorname{eml\_log}(a),\operatorname{eml\_exp}(b))=a-b.
\]

Zero, negation, and addition follow structurally as `log(1)`, `0-x`, and
`a-(-b)`. No constructor evaluates, simplifies, hash-conses into a DAG, or
introduces a derived leaf.

## Separately labeled CLEAN_NEGATION mode

`CompilerMode.CLEAN_NEGATION` is opt-in only. It is independently labeled and
must never be included in an `OFFICIAL_V4` fingerprint, alpha metric, or
structural metric. It uses the public companion source
`EML_toolkit/EmL_compiler/eml_compiler_clean_math_v0.py` at the same pinned
commit (retrieved-file SHA-256
`85be0e39271856dfa26c6368ba031f5eb83aa35a7beea6aba472e5dca6448f03`):

```text
e            = eml_exp(1)
e_minus_one  = eml_subtract(e, 1)
one_plus_x   = eml_subtract(e, eml_subtract(e_minus_one, x))
negate_clean = eml_subtract(1, one_plus_x)
```

Within the core constructors, the mode changes the negation expansion used by
negation and addition. Later compilers propagate the same explicit mode label
to any formula that depends on negation. Clean mode never activates implicitly,
and both modes still emit strictly pure EML trees.

## Domain and numerical caveats

The reconstructed `eml_log` agrees with principal `Log` on positive reals, but
the supplement specifies a different negative-real-axis convention. It is not
an unrestricted principal-complex-log identity.

Official zero, negation, and addition can pass through exact zero and extended
real intermediates, including `log(0) = -infinity`, `log(+infinity) =
+infinity`, and `exp(-infinity) = 0`. Structural correctness is primary here.
Finite positive-real probes directly support `exp`, `log`, and suitable
subtraction cases; zero, negation, and addition require an independent
extended-real-aware or formal audit. Evaluators must not clamp zero, replace it
with epsilon, or hide undefined `infinity - infinity`. Passing a finite sample
grid is supporting evidence, not a proof of universal complex-domain validity.
Issue 2-6 owns the full domain-aware semantic verification.

Multiplication, inverse, division, powers, exact-number compilation, named
constants, source dispatch, trigonometric functions, and hyperbolic functions
are outside these core constructors.
