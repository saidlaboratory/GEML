# GEML agent instructions

## Clean-room implementation

GEML Goals 1–5 are a clean-room production rebuild. Previous GEML code, tests, schemas,
helpers, architecture, and commit history are forbidden implementation sources. Coding agents
must not inspect the prototype repository or follow its link, even when the link appears in
`README.md`.

Implementation authority comes only from the current repository specifications, the complete
assigned GitHub issue, authoritative public mathematical sources named by those specifications,
and official library documentation. Stop and ask when a shared or scientific interface is
ambiguous.

## File ownership

Issue-owned write paths are exclusive. Edit only the paths assigned by the current issue or an
explicit multi-issue bundle, and never change another issue's contracts. During the scaffold
stage, issue 1-1 owns the root project files. After that stage, root files may be changed only by
an explicitly designated final integration issue.

## Scientific and data integrity

- Document domain assumptions and every scientific assumption or metric change.
- Retain and report failures, unsupported inputs, timeouts, and validation errors; never drop
  them silently.
- Do not hide unsupported operations in derived leaves or use hidden operators to improve
  representation metrics.
- Keep structural identity separate from semantic equivalence and preserve ordered child slots
  and repeated references where the governing contract requires them.

## Tests and generated data

Tests must use tiny hand-written or temporary fixtures. They must pass on a fresh clone when no
`outputs/` directory or production artifacts exist. Production corpus shards and generated
artifacts must never be test dependencies.

## Standard validation

Run all of the following before handing off a change:

```bash
python -m pytest
python -m ruff check .
python -m ruff format . --check
```
