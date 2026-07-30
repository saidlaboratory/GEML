#!/usr/bin/env bash
# Ordered Phase-B production pipeline for the 4xH100 node.
#
# Run one named step at a time, in the order documented in
# docs/specs/H100_RUNBOOK.md. Steps whose production entry point or frozen
# input does not exist yet REFUSE with exit 3 and name the blocker instead of
# improvising; see the runbook's PREREQUISITES section.
#
# All wall/RAM numbers printed here are labeled measured / estimate / pending;
# only measured numbers came from a real executed run (source noted inline).

set -euo pipefail

STEP="${1:-}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH=src

usage() {
    cat <<'EOF'
usage: scripts/h100/run_pipeline.sh <step>

steps, in execution order:
  verify-artifacts   authenticate the delivered Goals 1-5 artifact tree (recommended path)
  corpus-dev         regen fallback only: Goal 1 development stage (1k rows)
  corpus-pilot       regen fallback only: Goal 1 pilot stage (10k rows, 2 deterministic runs)
  corpus-final       regen fallback only: Goal 1 final stage (250k rows, ~28 GB RAM)
  pairs              production pair build (65k pairs from the frozen corpus)
  channels           materialize the four channel shard sets       [BLOCKED: no driver]
  width-preflight    goal6 encoder width freeze                    [BLOCKED: sibling branch]
  goal6-pilot        30-60 min 2-GPU throughput pilot (required before 4-GPU use)
  goal6-grid         six-arm x three-seed grid across 4 GPUs
  goal7-validate     validate the goal7 grid config (works today; prints blockers)
  goal7-grid         goal7 grid across 4 GPUs                      [BLOCKED: null identities]
  analysis           goal6/goal7 aggregation, gate evaluation, final report binding
EOF
}

note()    { printf '\n== %s\n' "$*"; }
budget()  { printf '   budget: %s\n' "$*"; }
blocked() { printf 'BLOCKED: %s\n' "$*" >&2; exit 3; }

need_artifacts_root() {
    [ -n "${GEML_ARTIFACTS_ROOT:-}" ] \
        || blocked "GEML_ARTIFACTS_ROOT is not set; deliver and verify the Goals 1-5 artifacts first"
}

case "${STEP}" in

verify-artifacts)
    note "authenticate the delivered Goals 1-5 artifact tree (Option A, recommended)"
    budget "download 35.8 GB tar + 80 GB free before extraction; wall depends on network (not measured)"
    DELIVERY_DIR="${GEML_DELIVERY_DIR:?set GEML_DELIVERY_DIR to the downloaded delivery directory}"
    python3 -m scripts.repro verify-delivery "${DELIVERY_DIR}"
    python3 -m scripts.repro preflight-archive "${DELIVERY_DIR}/GEML_artifacts_goals_1-5_2026-07-25.tar"
    tar -xf "${DELIVERY_DIR}/GEML_artifacts_goals_1-5_2026-07-25.tar" -C "${GEML_ARTIFACTS_PARENT:?set GEML_ARTIFACTS_PARENT to the extraction target}"
    python3 -m scripts.repro verify-tree "${GEML_ARTIFACTS_PARENT}/GEML_artifacts"
    echo "export GEML_ARTIFACTS_ROOT=${GEML_ARTIFACTS_PARENT}/GEML_artifacts"
    ;;

corpus-dev)
    note "Goal 1 corpus regen, development stage (Option B fallback; see runbook before using regen)"
    budget "1,000 rows; wall/RAM: pending (never measured)"
    python3 -m geml.experiments.goal1.run --config configs/goal1_final.yaml --stage development
    ;;

corpus-pilot)
    note "Goal 1 corpus regen, pilot stage (two deterministic runs + comparison)"
    budget "10,000 rows x2 runs; wall/RAM: pending (never measured)"
    python3 -m geml.experiments.goal1.run --config configs/goal1_final.yaml --stage pilot
    ;;

corpus-final)
    note "Goal 1 corpus regen, final stage"
    budget "250,000 rows; RAM ~28 GB (measured: memory-gated on the 16 GB M1 dev laptop,"
    budget "fits this node); wall: pending (never completed anywhere)"
    python3 -m geml.experiments.goal1.run --config configs/goal1_final.yaml --stage final
    ;;

pairs)
    note "production pair build: 50k train + 5k/5k/5k eval pairs from the frozen 250k corpus"
    budget "measured pilot: 34 min / 10k rows on M1 CPU (single process)."
    budget "65k target rows => ~3.7 h at that rate; on this node's server CPU: estimate 2-4 h,"
    budget "not measured. RAM: modest at pilot scale, estimate < 8 GB, not measured. No GPU."
    need_artifacts_root
    python3 -m geml.data.pairs --config configs/goal6_pairs.yaml
    ;;

channels)
    note "materialize ast_dag / pure_eml_dag / frequent_macro_motif_dag / motif_ast_fair_control shards"
    blocked "no production materialization driver exists; geml.learning.datasets.materialize \
exposes materialize_graph/write_channel_shard as APIs only (runbook PREREQUISITES item 4)"
    ;;

width-preflight)
    note "goal6 compact-encoder width freeze (permitted widths 64/96 + virtual-node choice)"
    blocked "the preflight tool is being built on the sibling branch h100/width-preflight; \
after it lands, freeze hidden_width/use_virtual_node in configs/goal6_grid.yaml, then re-run"
    ;;

goal6-pilot)
    note "mandatory 30-60 min throughput pilot on 2 GPUs (ML_REPRODUCIBILITY: 4 GPUs are"
    note "conditional on this recorded pilot; record GPU-hours and wall time separately)"
    budget "30-60 min wall by definition; per-cell throughput is the measurement output"
    blocked "no per-cell goal6 production entry point exists yet to hand to schedule_cells.py \
(runbook PREREQUISITES item 4); once it lands: \
python3 scripts/h100/schedule_cells.py --manifest outputs/final/goal6/grid/grid.manifest.json \
--gpus 0,1 --cmd '<per-cell command> --arm {arm_id} --seed {seed}'"
    ;;

goal6-grid)
    note "goal6 six-arm x three-seed grid (18 cells) across 4 GPUs"
    budget "per-cell wall: pending the goal6-pilot measurement; caps: 30 epochs / 20k optimizer"
    budget "steps / patience 5 per cell (configs/goal6_harness_defaults.yaml)"
    blocked "same per-cell entry point gap as goal6-pilot, and 4-GPU use is not authorized \
until the pilot is recorded; once both hold: \
python3 scripts/h100/schedule_cells.py --manifest outputs/final/goal6/grid/grid.manifest.json \
--gpus 0,1,2,3 --cmd '<per-cell command> --arm {arm_id} --seed {seed}'"
    ;;

goal7-validate)
    note "validate the goal7 grid config and print its production blockers (runs today)"
    budget "seconds; CPU only"
    python3 -m geml.experiments.goal7.run_grid --config configs/goal7_grid.yaml --validate-only
    ;;

goal7-grid)
    note "goal7 grid (4 gnn arms + transformer + uniform_valid, x3 seeds) across 4 GPUs"
    budget "hard cap: wall_time_seconds=7200 per cell => <= 36 GPU-hours for 18 cells by"
    budget "construction (a cap, not an estimate); actual per-cell wall: pending"
    blocked "configs/goal7_grid.yaml still carries null identity hashes and null reproduction \
commands, and goal7 run_grid main() only implements --validate-only (runbook PREREQUISITES \
item 3); once resolved: \
python3 scripts/h100/schedule_cells.py --manifest <output_root>/<run_id>/run.plan.json \
--gpus 0,1,2,3 --cmd '<reproduction_command from the plan, with {cell_id}>'"
    ;;

analysis)
    note "goal6/goal7 aggregation, Gate G6/G7 evaluation, final report binding"
    budget "CPU only; minutes (estimate, not measured)"
    echo "goal6: geml.analysis.goal6.summary.aggregate + evaluate_gate_g6 (Python API; the"
    echo "  frozen analysis entry command is recorded with the grid outputs)"
    echo "goal7: geml.analysis.goal7.summary.build_goal7_summary + write_goal7_summary via the"
    echo "  analysis_reproduction_command frozen into configs/goal7_grid.yaml"
    echo "final report (CLI exists today):"
    echo "  python3 -m geml.analysis.final.report --manifest <manifest> --sections <sections> \\"
    echo "      --artifact-root \${GEML_ARTIFACTS_ROOT} --markdown-out <out.md>"
    blocked "gate evaluation consumes complete authenticated grid rows; run after goal6-grid \
and goal7-grid finish"
    ;;

""|-h|--help|help)
    usage
    ;;

*)
    usage
    echo "unknown step: ${STEP}" >&2
    exit 2
    ;;
esac
