# Official arithmetic and internal-constant formulas

The arithmetic compiler composes the core macros documented in
`EML_CORE_FORMULAS.md`. Its authority is the pinned official v4 compiler at
commit `b3da148261199b46247306dfd92068f589778260`.

| Macro | Construction |
|---|---|
| `inverse(z)` | `exp(negate(log(z)))` |
| `multiply(a,b)` | `exp(add(log(a),log(b)))` |
| `divide(a,b)` | `multiply(a,inverse(b))` |
| `power(a,b)` | `exp(multiply(b,log(a)))` |

Integers use the official deterministic binary doubling/addition algorithm.
Zero compiles to `log(1)`, one remains the primitive `1`, and negative values
apply the selected official negation mode. A rational `p/q` compiles the
canonical exact integers, inverts the denominator, multiplies, and then applies
negation when `p < 0`. Noncanonical fractions and zero denominators fail
explicitly. Finite decimal inputs first become the exact ratio of their base-10
spelling; no binary float leaf enters pure EML.

The compiler also has three internal helpers required by the official trig
normalization:

```text
e_internal  = exp(1)
i_branch    = -exp(log(-1) / 2)
pi_internal = i_branch * log(-1)
```

The paper distinguishes the primitive principal `Log` from the reconstructed
`L`. With `Arg(z)` in `(-pi, pi]`, it defines the negative-real-axis value as
`L(x) = Log(x) - 2*pi*i`, hence exactly `L(-1) = -i*pi`. The pinned helper is
therefore `-exp(-i*pi/2) = +i`, and `i_branch * L(-1) = pi`.

This construction lies exactly on a logarithm branch cut. A finite-precision
evaluator can round an intermediate to either side of that cut and observe the
paired values `i_branch = -i` and `L(-1) = +i*pi`; their product still approaches
positive `pi`. GEML reports that as branch-sensitive numeric evidence and never
uses it to redefine the exact sourced convention. The final six real-source
functions pass the independent audits documented in
`EML_TRANSCENDENTAL_FORMULAS.md`.

These helpers are fully expanded back to `1`; they are not derived leaves.
They do **not** approve `E`, `pi`, or `I` as source-corpus constants. Source
complex mode and all three named source constants remain disabled.

Power, inverse, multiplication, and division retain their natural domain and
principal-branch limitations. Generator-side positive/nonzero guards remain
authoritative for the source corpus; the compiler does not manufacture a domain
proof.
