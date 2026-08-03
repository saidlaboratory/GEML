# Security Policy

GEML is a research codebase — a controlled representation study, not a service or a
library that handles untrusted user data in production. There is no attack surface in the
usual sense. Still, if you find a genuine security issue (for example, code that could
execute untrusted input, a dependency with a known CVE we should pin away from, or leaked
credentials), we want to hear about it.

## Reporting a vulnerability

Open a [private security advisory](https://github.com/saidlaboratory/GEML/security/advisories/new)
on the repository. GitHub keeps it confidential until we publish a fix.

Please give us a reasonable window to respond before disclosing publicly. We'll confirm
receipt, investigate, and credit you in the fix unless you prefer to stay anonymous.

## Do not put secrets in issues or PRs

Never paste tokens, API keys, private paths, or credentials into a public issue, pull
request, comment, or commit. If a secret is exposed, rotate it first, then tell us. Our
tests and CI run only on public, hermetic fixtures — nothing here should ever need a
secret checked into the repo.

## Provenance and clean-room integrity

GEML's results depend on their provenance, so we treat these as integrity concerns too:

- Every reported number must trace to a goal summary under `docs/goals/goalN/` and its
  checksummed manifest. Fabricating, altering, or softening a result — including nulls and
  the published gate failure — is an integrity violation. Report it the same way.
- Implementation must stay clean-room: only the repository specifications, assigned issues,
  and the authoritative public sources in
  [`docs/specs/EML_SOURCE_LEDGER.md`](docs/specs/EML_SOURCE_LEDGER.md) are allowed sources.
  If you spot code that appears copied from the v0 prototype or another forbidden source,
  flag it — provenance breaks are as serious as bugs here.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/CLEANROOM_RULES.md`](docs/CLEANROOM_RULES.md) for the full discipline.
