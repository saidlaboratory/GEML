# Goal 8 proof benchmark contract

The production manifest must contain exactly 256 unique accepted positive Goal 6 pairs. Before
selection, every candidate’s known concrete trace is replayed through the read-only verifier and
every source/e-class-relative group that occurs in training or validation is excluded. A frozen
family-by-tier quota table is selected deterministically in pair-ID order; shortages fail rather
than being silently backfilled.

Each row binds source/target IDs, source family, domain mode, distance tier, complete group
closure, pair ID, and concrete trace digest. The manifest content digest covers the ordered rows.
No post-result replacement or benchmark expansion is permitted.
