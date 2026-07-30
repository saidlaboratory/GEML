#!/usr/bin/env bash
# Bootstrap a bare Ubuntu 4xH100 node into a ready GEML Phase-B environment.
#
# Idempotent: safe to re-run after a partial failure. Every install step is a
# no-op when already satisfied. See docs/specs/H100_RUNBOOK.md for the full
# execution plan and prerequisites.
#
# Required environment:
#   GEML_REPO_URL   clone URL (e.g. git@github.com:saidlaboratory/GEML.git)
#   GEML_COMMIT     the release commit to check out (release/mathnlp-2026 tip;
#                   it must be pushed to the remote first -- see the runbook
#                   PREREQUISITES section)
# Optional:
#   GEML_DIR        checkout location (default: $HOME/geml)
#   GEML_BOOTSTRAP_FULL_TESTS=1  also run the full CPU test suite at the end

set -euo pipefail

GEML_REPO_URL="${GEML_REPO_URL:?set GEML_REPO_URL to the GEML clone URL}"
GEML_COMMIT="${GEML_COMMIT:?set GEML_COMMIT to the frozen release commit SHA}"
GEML_DIR="${GEML_DIR:-$HOME/geml}"

log() { printf '[bootstrap] %s\n' "$*"; }

# --- 1. GPU sanity: the node must expose H100s before anything is installed ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found: install the NVIDIA driver (>= 550.54 for the CUDA 12.4" >&2
    echo "runtime bundled in the torch 2.5.1 wheel) and re-run." >&2
    exit 1
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
log "visible GPUs: ${GPU_COUNT}"
if [ "${GPU_COUNT}" -lt 1 ]; then
    echo "no GPU visible; refusing to continue" >&2
    exit 1
fi

# --- 2. System packages (Ubuntu 24.04 ships python3.12; 22.04 needs deadsnakes) ---
if ! command -v python3.12 >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
    log "installing git/python3.12 via apt"
    sudo apt-get update
    sudo apt-get install -y git python3.12 python3.12-venv || {
        echo "apt could not provide python3.12; on Ubuntu 22.04 add deadsnakes first:" >&2
        echo "  sudo add-apt-repository ppa:deadsnakes/ppa" >&2
        exit 1
    }
fi
# python3.12-venv can be missing even when python3.12 exists
python3.12 -m venv --help >/dev/null 2>&1 || sudo apt-get install -y python3.12-venv

# --- 3. Clone at the exact release commit (detached; never a moving branch) ---
if [ ! -d "${GEML_DIR}/.git" ]; then
    log "cloning ${GEML_REPO_URL} -> ${GEML_DIR}"
    git clone "${GEML_REPO_URL}" "${GEML_DIR}"
fi
cd "${GEML_DIR}"
git fetch --all --tags
git checkout --detach "${GEML_COMMIT}"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [ "${ACTUAL_COMMIT}" != "${GEML_COMMIT}" ]; then
    echo "checkout mismatch: wanted ${GEML_COMMIT}, got ${ACTUAL_COMMIT}" >&2
    exit 1
fi
log "checked out ${ACTUAL_COMMIT}"

# --- 4. Python 3.12 venv + locked core dependencies ---
if [ ! -x .venv/bin/python ]; then
    python3.12 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

# --- 5. Pinned [ml] CUDA wheels per configs/ml_env.yaml ---
# The default-index (PyPI) linux x86_64 wheel for torch==2.5.1 bundles the CUDA
# 12.4 runtime and includes sm_90 (H100) kernels, and reports version exactly
# "2.5.1" -- which the pin test requires. Do NOT install from
# download.pytorch.org/whl/cu124: those wheels report "2.5.1+cu124" and fail
# the pin. torch-geometric 2.6.1 is pure Python.
python -m pip install torch==2.5.1 torch-geometric==2.6.1
# torch 2.5.1's metadata pins sympy==1.13.1, so the line above silently
# DOWNGRADES the locked sympy==1.14.0 (verified with pip: "Attempting
# uninstall: sympy ... installed sympy-1.13.1"). The core pin wins: every
# frozen Goals 1-5 artifact and the pair build parse sympy content under the
# requirements-lock.txt / pyproject pin, and the harness never calls
# torch.compile, so torch's sympy use stays inert (torch 2.5.1 imports and
# trains with sympy 1.14.0; verified on the dev laptop). Restore the lock pin;
# `pip check` will afterwards report exactly this one torch-metadata conflict,
# which is expected and accepted -- see H100_RUNBOOK prerequisite 7.
python -m pip install sympy==1.14.0

# --- 6. Editable install of GEML itself, no dependency resolution ---
python -m pip install -e . --no-deps

# --- 7. CUDA / sm90 self-check on the real device ---
python - <<'PY'
import torch

assert torch.__version__ == "2.5.1", f"torch pin violated: {torch.__version__}"
assert torch.cuda.is_available(), "torch cannot see CUDA; check driver >= 550.54"
archs = torch.cuda.get_arch_list()
assert "sm_90" in archs, f"wheel lacks sm_90 (H100) kernels: {archs}"
for index in range(torch.cuda.device_count()):
    capability = torch.cuda.get_device_capability(index)
    name = torch.cuda.get_device_name(index)
    print(f"cuda:{index} {name} capability {capability}")
    assert capability == (9, 0), f"cuda:{index} is not sm90 hardware: {capability}"
assert torch.cuda.is_bf16_supported(), "bf16 unsupported; H100 production precision is bf16"
print(f"cuda runtime {torch.version.cuda}; {torch.cuda.device_count()} GPU(s); bf16 OK")
PY

# --- 8. Environment self-check + pin tests (must PASS here, no skips) ---
# tests/learning/test_ml_env.py includes
# test_ml_environment_versions_when_optional_extra_is_installed, which skips
# when torch is absent and fails on a mismatched pin. On this node it must
# actually pass, so assert the pins directly as well.
PYTHONPATH=src python -m pytest tests/learning/test_ml_env.py -q -ra
python - <<'PY'
from importlib import metadata

assert metadata.version("torch") == "2.5.1"
assert metadata.version("torch-geometric") == "2.6.1"
assert metadata.version("sympy") == "1.14.0", "torch install clobbered the core sympy pin"
print("pins verified: torch 2.5.1, torch-geometric 2.6.1, sympy 1.14.0")
PY

if [ "${GEML_BOOTSTRAP_FULL_TESTS:-0}" = "1" ]; then
    log "running the full CPU test suite"
    PYTHONPATH=src python -m pytest -q
fi

log "bootstrap complete: ${GEML_DIR} @ ${ACTUAL_COMMIT}"
log "next: scripts/h100/run_pipeline.sh (see docs/specs/H100_RUNBOOK.md)"
