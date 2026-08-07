# GEML clean-room rules

Goals 1–5 in this repository are a clean-room production rebuild. The previous GEML prototype
may be mentioned as historical motivation, but its code, tests, schemas, helper functions,
internal architecture, and commit history are forbidden implementation sources. Do not inspect
or reuse them.

Allowed implementation sources are limited to:

- specifications and frozen contracts in this repository;
- the complete GitHub issue assigned to the work;
- authoritative public mathematical sources identified by the project source ledger; and
- official documentation for project dependencies.

If a shared interface or scientific requirement is ambiguous, stop and ask. Do not invent a
competing contract or edit files owned by another issue.

Tests must use small hand-written or temporary fixtures and must run on a fresh clone without an
`outputs/` directory. Production corpora and generated artifacts cannot be test dependencies.

Failures, unsupported inputs, timeouts, and validation errors must remain visible in accounting;
they may not be silently discarded. Future representation code must not hide unsupported
operators in derived leaves, introduce hidden operators, or relabel macro or motif nodes as pure
EML in order to improve reported metrics.
