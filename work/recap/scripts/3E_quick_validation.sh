#!/usr/bin/env bash

# =====================
# USER Config（仅需改这里）
# - 只改变量值即可；不要改后面的 python CLI flags（避免破坏可复现性与证据链）。
# =====================

usage() {
  cat <<'EOF'
Usage:
  bash agent/run/3E_quick_validation.sh [--help]

What it does:
  Runs the 3E Phase 1 quick validation loop once via
  `work/recap/scripts/3A_recap_multi_iter_loop.py`.

Key defaults in this script:
  N_ITERATIONS=1
  COLLECT_EPISODES=2
  EVAL_EPISODES=1
  COLLECT_MAX_POLICY_STEPS=10
  EVAL_MAX_POLICY_STEPS=50
  FINETUNE_MAX_STEPS=10
  SERVER_PORT=5800

Safe help behavior:
  `--help` / `-h` prints this message and exits 0.
  No log directory is created and no validation run is started.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

set -euxo pipefail

RUN_ID="recap_3E_quick_validation_$(date +%Y%m%d_%H%M%S)"

N_ITERATIONS=1
COLLECT_EPISODES=2
EVAL_EPISODES=1
FINETUNE_MAX_STEPS=10
SEED=42
FIXED_EVAL_SEED=5042
SERVER_PORT=5800

COLLECT_MAX_POLICY_STEPS=10
EVAL_MAX_POLICY_STEPS=50

TIMEOUT_COLLECT_S=180
TIMEOUT_CRITIC_S=120
TIMEOUT_LABEL_S=60
TIMEOUT_EXPORT_S=60
TIMEOUT_FINETUNE_S=180
TIMEOUT_EVAL_S=120

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUNTIME_LOG_DIR="agent/runtime_logs/p3E"
mkdir -p "${RUNTIME_LOG_DIR}"

LOG_FILE="${RUNTIME_LOG_DIR}/${RUN_ID}.log"

SECONDS=0
START_ISO="$(date -Is)"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[3E] Launching quick validation (phase 1)"
echo "[3E] RUN_ID=${RUN_ID}"
echo "[3E] start_time=${START_ISO}"
echo "[3E] log_file=${LOG_FILE}"

PER_ITER_TIMEOUT_S="$((TIMEOUT_COLLECT_S + TIMEOUT_CRITIC_S + TIMEOUT_LABEL_S + TIMEOUT_EXPORT_S + TIMEOUT_FINETUNE_S + (2 * TIMEOUT_EVAL_S)))"
EST_TIMEOUT_S="$((N_ITERATIONS * PER_ITER_TIMEOUT_S))"
echo "[3E] estimated_timeout_upper_bound_s=${EST_TIMEOUT_S} (~$((EST_TIMEOUT_S / 60)) min)"

set +e
python3 work/recap/scripts/3A_recap_multi_iter_loop.py \
  --run-id "${RUN_ID}" \
  --n-iterations "${N_ITERATIONS}" \
  --collect-episodes "${COLLECT_EPISODES}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --finetune-max-steps "${FINETUNE_MAX_STEPS}" \
  --finetune-save-steps "${FINETUNE_MAX_STEPS}" \
  --seed "${SEED}" \
  --fixed-eval-seed "${FIXED_EVAL_SEED}" \
  --server-port "${SERVER_PORT}" \
  --collect-max-policy-steps "${COLLECT_MAX_POLICY_STEPS}" \
  --eval-max-policy-steps "${EVAL_MAX_POLICY_STEPS}" \
  --mixdone \
  --mixdone-short-episodes 1 \
  --mixdone-long-episodes 1 \
  --finetune-tune-projector \
  --no-finetune-tune-diffusion-model \
  --no-require-git-clean \
  --write-repro-snapshot \
  --timeout-collect-s "${TIMEOUT_COLLECT_S}" \
  --timeout-critic-s "${TIMEOUT_CRITIC_S}" \
  --timeout-label-s "${TIMEOUT_LABEL_S}" \
  --timeout-export-s "${TIMEOUT_EXPORT_S}" \
  --timeout-finetune-s "${TIMEOUT_FINETUNE_S}" \
  --timeout-eval-s "${TIMEOUT_EVAL_S}"
PY_RC=$?
set -e

END_ISO="$(date -Is)"
WALL_S="${SECONDS}"

echo ""
echo "=========================================="
if [[ "${PY_RC}" -eq 0 ]]; then
  echo "[3E] status=COMPLETED"
else
  echo "[3E] status=FAILED"
  echo "[3E] failed_command=work/recap/scripts/3A_recap_multi_iter_loop.py"
  echo "[3E] rc=${PY_RC}"
fi
echo "[3E] RUN_ID: ${RUN_ID}"
echo "RUN_ID=${RUN_ID}"
echo "[3E] end_time=${END_ISO}"
echo "[3E] wall_time_s=${WALL_S}"
echo "[3E] log_file=${LOG_FILE}"
echo "=========================================="

exit "${PY_RC}"
