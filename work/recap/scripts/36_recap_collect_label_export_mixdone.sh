#!/usr/bin/env bash
set -euxo pipefail

# USER Config (edit)
ITER_TAG="recap_iter_002_mixdone_20260226_024757"
ENV_NAME="gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
MODEL_PATH="nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
EMBODIMENT_TAG="UNITREE_G1"

SERVER_HOST="127.0.0.1"
SERVER_PORT=5556

N_EPISODES_SHORT=20
N_EPISODES_LONG=20
MAX_POLICY_STEPS=10

MAX_EPISODE_STEPS_SHORT=60
MAX_EPISODE_STEPS_LONG=1440

N_ACTION_STEPS_CONFIG=30
MUJOCO_GL="egl"

LABEL_VALUE_BASELINE="t_mean_return"
LABEL_EPS_QUANTILE=0.7

EXPORT_MAX_EPISODES=40

TOTAL_TIMEOUT_COLLECT_S=1800
TOTAL_TIMEOUT_LABEL_S=300
TOTAL_TIMEOUT_EXPORT_S=600

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="agent/runtime_logs/${ITER_TAG}"
mkdir -p "$LOG_DIR"

LOG_SYS="$LOG_DIR/00_system_probe.log"
LOG_COLLECT_SHORT="$LOG_DIR/10_collect_short_max_episode_steps.log"
LOG_COLLECT_LONG="$LOG_DIR/11_collect_long_max_episode_steps.log"
LOG_COLLECT_SANITY="$LOG_DIR/12_collect_sanity.log"
LOG_LABEL="$LOG_DIR/20_label.log"
LOG_LABEL_RETRY_Q50="$LOG_DIR/22_label_retry_q50.log"
LOG_LABEL_FALLBACK_MEAN="$LOG_DIR/23_label_fallback_mean_return.log"
LOG_LABEL_STATS="$LOG_DIR/21_label_stats.log"
LOG_EXPORT="$LOG_DIR/30_export_lerobot_v2.log"

{
  echo "[INFO] ts=$(date -Is)"
  echo "[INFO] repo_root=$REPO_ROOT"
  echo "[INFO] iter_tag=$ITER_TAG"
  echo "[INFO] env_name=$ENV_NAME"
  echo "[INFO] model_path=$MODEL_PATH"
  echo "[INFO] embodiment_tag=$EMBODIMENT_TAG"
  echo "[INFO] server=${SERVER_HOST}:${SERVER_PORT}"
  echo "[INFO] max_policy_steps=$MAX_POLICY_STEPS"
  echo "[INFO] max_episode_steps_short=$MAX_EPISODE_STEPS_SHORT"
  echo "[INFO] max_episode_steps_long=$MAX_EPISODE_STEPS_LONG"
  echo "[INFO] n_action_steps_config=$N_ACTION_STEPS_CONFIG"
  echo "[INFO] mujoco_gl=$MUJOCO_GL"
  echo "[INFO] label_value_baseline=$LABEL_VALUE_BASELINE"
  echo "[INFO] label_eps_quantile=$LABEL_EPS_QUANTILE"
} | tee -a "$LOG_SYS"

(nvidia-smi || true) 2>&1 | tee -a "$LOG_SYS"
python3 -V 2>&1 | tee -a "$LOG_SYS"

timeout --signal=INT --kill-after=20s "${TOTAL_TIMEOUT_COLLECT_S}s" \
  python3 work/recap/scripts/31_recap_collect_rollouts.py \
    --iter-tag "$ITER_TAG" \
    --env-name "$ENV_NAME" \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --server-host "$SERVER_HOST" \
    --server-port "$SERVER_PORT" \
    --n-episodes "$N_EPISODES_SHORT" \
    --max-policy-steps "$MAX_POLICY_STEPS" \
    --max-episode-steps "$MAX_EPISODE_STEPS_SHORT" \
    --n-action-steps-config "$N_ACTION_STEPS_CONFIG" \
    --seed 0 \
    --mujoco-gl "$MUJOCO_GL" \
    --offscreen \
    --no-onscreen \
    --total-timeout-s "$TOTAL_TIMEOUT_COLLECT_S" \
    --kill-server-on-exit \
  2>&1 | tee -a "$LOG_COLLECT_SHORT"

timeout --signal=INT --kill-after=20s "${TOTAL_TIMEOUT_COLLECT_S}s" \
  python3 work/recap/scripts/31_recap_collect_rollouts.py \
    --iter-tag "$ITER_TAG" \
    --env-name "$ENV_NAME" \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --server-host "$SERVER_HOST" \
    --server-port "$SERVER_PORT" \
    --n-episodes "$N_EPISODES_LONG" \
    --max-policy-steps "$MAX_POLICY_STEPS" \
    --max-episode-steps "$MAX_EPISODE_STEPS_LONG" \
    --n-action-steps-config "$N_ACTION_STEPS_CONFIG" \
    --seed 1000 \
    --mujoco-gl "$MUJOCO_GL" \
    --offscreen \
    --no-onscreen \
    --total-timeout-s "$TOTAL_TIMEOUT_COLLECT_S" \
    --kill-server-on-exit \
  2>&1 | tee -a "$LOG_COLLECT_LONG"

ITER_TAG="$ITER_TAG" python3 - <<'PY' 2>&1 | tee -a "$LOG_COLLECT_SANITY"
import json
import pathlib
import collections

import os
iter_tag = os.environ.get("ITER_TAG") or ""
assert iter_tag, "ITER_TAG env missing"
p = pathlib.Path("agent/artifacts/recap_datasets") / iter_tag / "episodes.jsonl"
lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
eps = [json.loads(l) for l in lines]
print("episodes", len(eps))
print("done_true", sum(bool(e.get("done")) for e in eps))
print("success_true", sum(bool(e.get("success_episode")) for e in eps))
lens = [int(e.get("n_policy_steps", -1)) for e in eps]
print("n_policy_steps_counts", dict(collections.Counter(lens)))
PY

timeout --signal=INT --kill-after=20s "${TOTAL_TIMEOUT_LABEL_S}s" \
  python3 work/recap/scripts/32_recap_label_dataset.py \
    --iter-tag "$ITER_TAG" \
    --value-baseline "$LABEL_VALUE_BASELINE" \
    --epsilon-strategy quantile \
    --epsilon-quantile "$LABEL_EPS_QUANTILE" \
    --total-timeout-s "$TOTAL_TIMEOUT_LABEL_S" \
  2>&1 | tee -a "$LOG_LABEL"

ITER_TAG="$ITER_TAG" python3 - <<'PY' 2>&1 | tee -a "$LOG_LABEL_STATS"
import json
import pathlib
import os

iter_tag = os.environ.get("ITER_TAG") or ""
assert iter_tag, "ITER_TAG env missing"
s = pathlib.Path("agent/artifacts/recap_datasets") / iter_tag / "m2_labels" / "stats.json"
obj = json.loads(s.read_text(encoding="utf-8"))
print(obj)
PY

POS_RATIO=$(ITER_TAG="$ITER_TAG" python3 - <<'PY'
import json, os, pathlib
iter_tag = os.environ.get("ITER_TAG") or ""
assert iter_tag
s = pathlib.Path("agent/artifacts/recap_datasets") / iter_tag / "m2_labels" / "stats.json"
obj = json.loads(s.read_text(encoding="utf-8"))
v = obj.get("pos_ratio")
print("" if v is None else v)
PY
)

if [[ "${POS_RATIO:-}" == "0" || "${POS_RATIO:-}" == "0.0" ]]; then
  timeout --signal=INT --kill-after=20s "${TOTAL_TIMEOUT_LABEL_S}s" \
    python3 work/recap/scripts/32_recap_label_dataset.py \
      --iter-tag "$ITER_TAG" \
      --value-baseline "$LABEL_VALUE_BASELINE" \
      --epsilon-strategy quantile \
      --epsilon-quantile 0.5 \
      --total-timeout-s "$TOTAL_TIMEOUT_LABEL_S" \
    2>&1 | tee -a "$LOG_LABEL_RETRY_Q50"

  POS_RATIO=$(ITER_TAG="$ITER_TAG" python3 - <<'PY'
import json, os, pathlib
iter_tag = os.environ.get("ITER_TAG") or ""
assert iter_tag
s = pathlib.Path("agent/artifacts/recap_datasets") / iter_tag / "m2_labels" / "stats.json"
obj = json.loads(s.read_text(encoding="utf-8"))
v = obj.get("pos_ratio")
print("" if v is None else v)
PY
  )
fi

if [[ "${POS_RATIO:-}" == "0" || "${POS_RATIO:-}" == "0.0" ]]; then
  timeout --signal=INT --kill-after=20s "${TOTAL_TIMEOUT_LABEL_S}s" \
    python3 work/recap/scripts/32_recap_label_dataset.py \
      --iter-tag "$ITER_TAG" \
      --value-baseline mean_return \
      --epsilon-strategy quantile \
      --epsilon-quantile "$LABEL_EPS_QUANTILE" \
      --total-timeout-s "$TOTAL_TIMEOUT_LABEL_S" \
    2>&1 | tee -a "$LOG_LABEL_FALLBACK_MEAN"
fi

timeout --signal=INT --kill-after=20s "${TOTAL_TIMEOUT_EXPORT_S}s" \
  python3 work/recap/scripts/33_recap_export_lerobot_v2_dataset.py \
    --iter-tag "$ITER_TAG" \
    --max-episodes "$EXPORT_MAX_EPISODES" \
    --total-timeout-s "$TOTAL_TIMEOUT_EXPORT_S" \
  2>&1 | tee -a "$LOG_EXPORT"

echo "[INFO] done" | tee -a "$LOG_EXPORT"
