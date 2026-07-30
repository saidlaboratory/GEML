<!--
One issue per PR. If this touches more than one issue, split it.
Keep the description casual and to the point.
-->

## Summary

<!-- A sentence or two on what this PR does. -->

Closes #

## Result

<!--
Only if there are real, measured numbers to report. State what produced them
(command/script, device, dataset/config) so they're verifiable. If there's
nothing measured, delete this section — don't invent metrics.
-->

## Notes

<!-- Caveats, follow-ups, open questions, reviewers to @-ping. Delete if empty. -->

## Checklist

- [ ] Stayed inside this issue's exclusive write scope — no other issue's contracts touched
- [ ] Clean-room: implemented only from repo specs / this issue / the sources in `docs/specs/EML_SOURCE_LEDGER.md` (no v0 prototype)
- [ ] Every number traces to a `docs/goals/goalN/` summary + its checksummed manifest
- [ ] Failures and null results are reported plainly, not softened
- [ ] `python -m pytest` passes
- [ ] `python -m ruff check .` and `python -m ruff format . --check` pass
- [ ] Tests run on a fresh clone with no `outputs/` or production artifacts
