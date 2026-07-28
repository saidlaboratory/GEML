# Goal 9 symbolic-regression task contract

The frozen benchmark requires exactly 256 synthetic in-grammar tasks and a separately documented
restricted Feynman-style subset. Each task binds the target identity, variables, real domain,
observation seed, operator family, and complexity. Unsupported Feynman candidates remain in an
exclusion ledger rather than being approximated with hidden operators.

Numeric fit and exact recovery are distinct. A fit—even a zero observed error—may only be labeled
`numeric_only` unless an approved symbolic verifier confirms it under the recorded domain. The
current Goal 4 verifier is not a full-v1 arbitrary-SR exact-recovery verifier; therefore no
production exact-recovery percentage may be generated until that gap is resolved.
