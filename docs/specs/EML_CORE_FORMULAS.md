# Official core pure-EML formulas

This module implements the primitive constructions from the official compiler
[`eml_compiler_v4.py`](https://github.com/VA00/SymbolicRegressionPackage/blob/b3da148261199b46247306dfd92068f589778260/EML_toolkit/EmL_compiler/eml_compiler_v4.py)
at immutable commit `b3da148261199b46247306dfd92068f589778260`.
The retrieved source's SHA-256 is
`7b147b564a952c8ef24f1b1f6bb2b443a68ce44100ad019d5222df9547b6da62`.

Pure result trees contain only ordered binary `EML[left,right]`, source-variable
occurrences, and the constant `1`. Names below are compile-time macros, not
result nodes. With `F(a,b) = exp(a) - Log(b)` and principal complex `Log`, the
pinned constructions are:

| Macro | Construction |
|---|---|
| `exp(z)` | `F(z, 1)` |
| `log(z)` | `F(1, F(F(1,z),1))` |
| `zero()` | `log(1)` |
| `subtract(a,b)` | `F(log(a), exp(b))` |
| `negate(z)` | `subtract(zero(), z)` |
| `add(a,b)` | `subtract(a, negate(b))` |

The macro named `log` reconstructs the paper's `L`; it agrees with principal
`Log` on positive reals but has the supplement's explicitly different
negative-real-axis convention. It must not be treated as an unrestricted
principal-log identity.

The default `official_v4` mode preserves these formulas byte for byte. The
separate `clean_negation` mode implements the companion official
[`eml_compiler_clean_math_v0.py`](https://github.com/VA00/SymbolicRegressionPackage/blob/b3da148261199b46247306dfd92068f589778260/EML_toolkit/EmL_compiler/eml_compiler_clean_math_v0.py),
source SHA-256
`85be0e39271856dfa26c6368ba031f5eb83aa35a7beea6aba472e5dca6448f03`:

```text
e             = exp(1)
e_minus_one   = subtract(e, 1)
one_plus_z    = subtract(e, subtract(e_minus_one, z))
negate_clean  = subtract(1, one_plus_z)
```

This clean construction removes the direct `Log(0)` route from negation. It
does not promise that a larger compiled tree never encounters an extended-real
intermediate: official multiplication and division themselves are expressed
through logarithms. Verification therefore records nonfinite intermediates and
backend behavior rather than silently dropping affected sample points.

These identities have principal-log and extended-real caveats. A numeric match
at a sample grid is evidence, not a proof that every nested branch-sensitive
identity is globally valid.
