#!/usr/bin/env bash
# Probe WI — formal_eval on each interpolated rescue checkpoint, sequential.
# 30 seeds positive×30, n_envs=1, GPU2, server-port 5005.
set -euo pipefail

REPO_ROOT="/home/howard/Projects/gr00t_wbc_g1_benchmark"
WBC_PY="${REPO_ROOT}/submodules/Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"

RESCUE_ROOT="${REPO_ROOT}/agent/artifacts/probes/probe_WI_rescue"
EVAL_ROOT="${RESCUE_ROOT}/per_alpha_eval"
LOG_ROOT="${REPO_ROOT}/agent/runtime_logs/probes/probe_WI"

ALPHAS=(0.25 0.5 0.75)

mkdir -p "${EVAL_ROOT}" "${LOG_ROOT}"

overall_ec=0

for ALPHA in "${ALPHAS[@]}"; do
  CKPT_DIR="${RESCUE_ROOT}/checkpoint_alpha_${ALPHA}"
  if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "ERROR: rescue checkpoint missing for alpha=${ALPHA}: ${CKPT_DIR}" >&2
    overall_ec=2
    continue
  fi

  UTC="$(date -u +%Y%m%dT%H%M%SZ)"
  PID_TAG="$$"
  OUT_DIR="${EVAL_ROOT}/${ALPHA}"
  RUNTIME_DIR="${OUT_DIR}/runtime"
  LOG_FILE="${LOG_ROOT}/${UTC}_eval_alpha_${ALPHA}.log"

  mkdir -p "${OUT_DIR}" "${RUNTIME_DIR}"

  echo "[probe_WI_eval] alpha=${ALPHA} ckpt=${CKPT_DIR}"
  echo "[probe_WI_eval] log=${LOG_FILE} out=${OUT_DIR}"

  attempt=1
  ec_alpha=0
  while [[ ${attempt} -le 2 ]]; do
    set +e
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
    ec_alpha=$?
    set -e

    if [[ ${ec_alpha} -eq 0 ]]; then
      break
    fi
    if [[ ${attempt} -ge 2 ]]; then
      echo "[probe_WI_eval] alpha=${ALPHA} failed after retry (ec=${ec_alpha}); continuing"
      break
    fi
    echo "[probe_WI_eval] alpha=${ALPHA} attempt ${attempt} failed (ec=${ec_alpha}); cleaning up port 5005 and retrying"
    pkill -f 'run_gr00t_server.py' 2>/dev/null || true
    sleep 5
    attempt=$((attempt + 1))
  done

  echo "{\"alpha\":${ALPHA},\"output_dir\":\"${OUT_DIR}\",\"log\":\"${LOG_FILE}\",\"exit_code\":${ec_alpha},\"finished_at_utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
    > "${OUT_DIR}/probe_WI_eval_status.json"

  if [[ ${ec_alpha} -ne 0 && ${overall_ec} -eq 0 ]]; then
    overall_ec=${ec_alpha}
  fi
done

exit ${overall_ec}
