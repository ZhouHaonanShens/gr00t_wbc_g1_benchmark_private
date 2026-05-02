#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

WBC_PY=".envs/wbc/bin/python"
LOG_DIR="agent/runtime_logs/p0_ladder_phase0"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/stage_b_p0_ladder_runner_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ ! -x "${WBC_PY}" ]]; then
  echo "missing executable WBC python: ${WBC_PY}" | tee -a "${LOG_FILE}" >&2
  exit 127
fi

export PYTHONUNBUFFERED=1
export GR00T_SKIP_WBC_REEXEC=1
export NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/submodules/Isaac-GR00T:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobosuite:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobocasa:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/robocasa:${PYTHONPATH:-}"

echo "[P0_RUNNER] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) command=$*" | tee -a "${LOG_FILE}"
echo "[P0_RUNNER] python=${WBC_PY}" | tee -a "${LOG_FILE}"
timeout --preserve-status 21600 "${WBC_PY}" work/recap/stage_b/p0_eval_protocol_runner.py "$@" 2>&1 | tee -a "${LOG_FILE}"
