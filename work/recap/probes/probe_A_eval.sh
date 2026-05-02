#!/usr/bin/env bash
# Probe A — formal_eval on the new pure-SFT checkpoint.
# Runs after Probe A training completes. 30 seeds positive×30, n_envs=1, GPU2.
set -euo pipefail

REPO_ROOT="/home/howard/Projects/gr00t_wbc_g1_benchmark"
WBC_PY="${REPO_ROOT}/submodules/Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"

CKPT_DIR="${REPO_ROOT}/agent/artifacts/probes/probe_A_pure_sft_control/training_run_20260501T134222Z/checkpoint-3300"
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "ERROR: training checkpoint not yet at ${CKPT_DIR}" >&2
  exit 2
fi

UTC="$(date -u +%Y%m%dT%H%M%SZ)"
PID_TAG="$$"
OUT_DIR="${REPO_ROOT}/agent/artifacts/probes/probe_A_pure_sft_control/formal_eval/${UTC}_${PID_TAG}"
RUNTIME_DIR="${OUT_DIR}/runtime"
LOG_FILE="${REPO_ROOT}/agent/runtime_logs/probes/probe_A/${UTC}_eval.log"

mkdir -p "${OUT_DIR}" "${RUNTIME_DIR}" "$(dirname "${LOG_FILE}")"

CUDA_VISIBLE_DEVICES=2 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
PYTHONUNBUFFERED=1 \
GR00T_SKIP_WBC_REEXEC=1 \
NO_ALBUMENTATIONS_UPDATE=1 \
PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/submodules/Isaac-GR00T:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobosuite:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobocasa:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/robocasa:${PYTHONPATH:-}" \
"${WBC_PY}" "${REPO_ROOT}/work/recap/scripts/gr00t_g3_formal_eval.py" \
  --checkpoint "${CKPT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --runtime-log-dir "${RUNTIME_DIR}" \
  --server-host 127.0.0.1 \
  --server-port 5005 \
  --seed-base 20000 \
  --episode-count 30 \
  --indicator-modes positive \
  --required-cuda-visible-devices 2 \
  >> "${LOG_FILE}" 2>&1

ec=$?
echo "{\"output_dir\":\"${OUT_DIR}\",\"log\":\"${LOG_FILE}\",\"exit_code\":${ec},\"finished_at_utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
  > "${OUT_DIR}/probe_A_eval_status.json"
exit ${ec}
