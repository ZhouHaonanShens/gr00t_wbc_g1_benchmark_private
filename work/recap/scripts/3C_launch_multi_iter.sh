#!/usr/bin/env bash

# =====================
# USER Config（仅需改这里）
# - 本脚本负责启动 3C 多轮（multi-iteration）RECAP 回环；实际执行逻辑在 `work/recap/scripts/3A_recap_multi_iter_loop.py`。
# - 下方变量是“用户可调参数”；默认值是当前推荐的最小可复现实用配置。
# - 只改变量值即可；不要改后面的 python CLI flags（避免破坏可复现性与计划约束）。
#
# 可调变量说明：
# - RUN_ID: 本次运行的唯一标识（也用于日志文件名）。
# - N_ITERATIONS: 回环迭代轮数。
# - COLLECT_EPISODES: 每轮收集的 episode 数。
# - FINETUNE_MAX_STEPS: 每轮 finetune 的最大 step；同时用于 `--finetune-save-steps`（每轮只保留一个 checkpoint）。
# - SEED: 随机种子。
# - SERVER_PORT: policy server 监听端口。
# - TIMEOUT_*_S: 各阶段硬超时（秒）；由 python 端分别应用，避免无限卡死。
# =====================

set -euxo pipefail

RUN_ID="recap_3C_formal_$(date +%Y%m%d_%H%M%S)"

N_ITERATIONS=5
COLLECT_EPISODES=50
FINETUNE_MAX_STEPS=200
SEED=42
SERVER_PORT=5800

TIMEOUT_COLLECT_S=2700
TIMEOUT_CRITIC_S=1800
TIMEOUT_LABEL_S=900
TIMEOUT_EXPORT_S=900
TIMEOUT_FINETUNE_S=2700
TIMEOUT_EVAL_S=1800

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUNTIME_LOG_DIR="agent/runtime_logs/p3C"
mkdir -p "${RUNTIME_LOG_DIR}"

LOG_FILE="${RUNTIME_LOG_DIR}/${RUN_ID}.log"

SECONDS=0
START_ISO="$(date -Is)"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[3C] Launching multi-iteration RECAP run"
echo "[3C] RUN_ID=${RUN_ID}"
echo "[3C] start_time=${START_ISO}"
echo "[3C] log_file=${LOG_FILE}"

PER_ITER_TIMEOUT_S="$((TIMEOUT_COLLECT_S + TIMEOUT_CRITIC_S + TIMEOUT_LABEL_S + TIMEOUT_EXPORT_S + TIMEOUT_FINETUNE_S + TIMEOUT_EVAL_S))"
EST_TIMEOUT_S="$((N_ITERATIONS * PER_ITER_TIMEOUT_S))"
echo "[3C] estimated_timeout_upper_bound_s=${EST_TIMEOUT_S} (~$((EST_TIMEOUT_S / 60)) min)"

python3 work/recap/scripts/3A_recap_multi_iter_loop.py \
  --run-id "${RUN_ID}" \
  --n-iterations "${N_ITERATIONS}" \
  --collect-episodes "${COLLECT_EPISODES}" \
  --finetune-max-steps "${FINETUNE_MAX_STEPS}" \
  --finetune-save-steps "${FINETUNE_MAX_STEPS}" \
  --seed "${SEED}" \
  --server-port "${SERVER_PORT}" \
  --mixdone \
  --no-require-git-clean \
  --write-repro-snapshot \
  --timeout-collect-s "${TIMEOUT_COLLECT_S}" \
  --timeout-critic-s "${TIMEOUT_CRITIC_S}" \
  --timeout-label-s "${TIMEOUT_LABEL_S}" \
  --timeout-export-s "${TIMEOUT_EXPORT_S}" \
  --timeout-finetune-s "${TIMEOUT_FINETUNE_S}" \
  --timeout-eval-s "${TIMEOUT_EVAL_S}"

END_ISO="$(date -Is)"
WALL_S="${SECONDS}"

echo ""
echo "========================================"
echo "[3C] DONE"
echo "[3C] RUN_ID: ${RUN_ID}"
echo "RUN_ID=${RUN_ID}"
echo "[3C] end_time=${END_ISO}"
echo "[3C] wall_time_s=${WALL_S}"
echo "[3C] log_file=${LOG_FILE}"
echo "========================================"
