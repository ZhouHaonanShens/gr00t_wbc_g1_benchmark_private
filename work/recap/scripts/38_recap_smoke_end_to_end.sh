#!/usr/bin/env bash
set -euxo pipefail

 # USER Config (edit)
ENV_NAME="gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
MODEL_PATH="nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
EMBODIMENT_TAG="UNITREE_G1"
SERVER_HOST="127.0.0.1"
SERVER_PORT=5555

N_ACTION_STEPS_CONFIG=30
MUJOCO_GL="egl"

N_EPISODES_SHORT=1
N_EPISODES_LONG=1
MAX_POLICY_STEPS=6
MAX_EPISODE_STEPS_SHORT=60
MAX_EPISODE_STEPS_LONG=1440

LABEL_EPS_QUANTILE=0.7
EXPORT_MAX_EPISODES=2

FINETUNE_MAX_STEPS=1
FINETUNE_SAVE_STEPS=1
FINETUNE_GLOBAL_BATCH_SIZE=1
FINETUNE_GRAD_ACCUM_STEPS=1
FINETUNE_DATALOADER_NUM_WORKERS=2

TIMEOUT_COLLECT_S=1200
TIMEOUT_LABEL_S=180
TIMEOUT_EXPORT_S=300
TIMEOUT_FINETUNE_S=3600
TIMEOUT_EVAL_S=1200

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

TS_TAG="$(date +%Y%m%d_%H%M%S)"
ITER_TAG="recap_smoke_${TS_TAG}"
LOG_DIR="agent/runtime_logs/${ITER_TAG}"
mkdir -p "$LOG_DIR"

OUT_DIR="agent/artifacts/checkpoints/${ITER_TAG}/finetune_full_diffusion_smoke"
mkdir -p "$OUT_DIR"
SELECTED_CHECKPOINT_PATH=""

WBC_PY_DEFAULT=".envs/wbc/bin/python"
WBC_PY="${WBC_PY:-$WBC_PY_DEFAULT}"

{
  echo "[INFO] ts=$(date -Is)"
  echo "[INFO] iter_tag=$ITER_TAG"
  echo "[INFO] env_name=$ENV_NAME"
  echo "[INFO] model_path=$MODEL_PATH"
  echo "[INFO] embodiment_tag=$EMBODIMENT_TAG"
  echo "[INFO] server=${SERVER_HOST}:${SERVER_PORT}"
  echo "[INFO] wbc_py=$WBC_PY"
} | tee -a "$LOG_DIR/00_header.log"

(nvidia-smi || true) 2>&1 | tee -a "$LOG_DIR/01_nvidia_smi.log"

timeout --signal=INT --kill-after=20s "${TIMEOUT_COLLECT_S}s" \
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
    --total-timeout-s "$TIMEOUT_COLLECT_S" \
    --kill-server-on-exit \
  2>&1 | tee -a "$LOG_DIR/10_collect_short.log"

timeout --signal=INT --kill-after=20s "${TIMEOUT_COLLECT_S}s" \
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
    --total-timeout-s "$TIMEOUT_COLLECT_S" \
    --kill-server-on-exit \
  2>&1 | tee -a "$LOG_DIR/11_collect_long.log"

timeout --signal=INT --kill-after=20s "${TIMEOUT_LABEL_S}s" \
  python3 work/recap/scripts/32_recap_label_dataset.py \
    --iter-tag "$ITER_TAG" \
    --value-baseline t_mean_return \
    --value-source baseline \
    --epsilon-strategy quantile \
    --epsilon-quantile "$LABEL_EPS_QUANTILE" \
    --total-timeout-s "$TIMEOUT_LABEL_S" \
  2>&1 | tee -a "$LOG_DIR/20_label.log"

timeout --signal=INT --kill-after=20s "${TIMEOUT_LABEL_S}s" \
  python3 - "$ITER_TAG" <<'PY' \
  2>&1 | tee -a "$LOG_DIR/25_advantage_contract.log"
import importlib
import json
import sys
from pathlib import Path

iter_tag = str(sys.argv[1])
repo_root = Path.cwd().resolve()
dataset_dir = repo_root / "agent" / "artifacts" / "recap_datasets" / iter_tag
labels_path = dataset_dir / "m2_labels" / "labels.jsonl"
if not labels_path.is_file():
    raise FileNotFoundError(f"labels.jsonl not found: {labels_path}")

label_rows = []
with labels_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict):
            raise ValueError(f"labels.jsonl row is not an object: {type(rec).__name__}")
        label_rows.append(rec)
if not label_rows:
    raise ValueError(f"labels.jsonl contains no records: {labels_path}")

advantage_mod = importlib.import_module("work.recap.advantage")
compute_sign_scales = getattr(advantage_mod, "compute_sign_aware_advantage_scales")
build_contract = getattr(advantage_mod, "build_advantage_contract_metadata")

advantage_values = [float(rec["advantage_A"]) for rec in label_rows]
sign_scale_summary = compute_sign_scales(
    advantage_values,
    context=f"38.{iter_tag}.continuous_advantage_contract",
)
positive_scale = sign_scale_summary.get("positive_scale")
negative_scale_abs = sign_scale_summary.get("negative_scale_abs")
if positive_scale is None or negative_scale_abs is None:
    raise ValueError(
        "continuous advantage contract requires both positive and negative scales"
    )

contract = build_contract(
    source_iter_tag=iter_tag,
    n_samples=len(label_rows),
    positive_scale=float(positive_scale),
    negative_scale_abs=float(negative_scale_abs),
    critic_dir=None,
    critic_include_t=False,
    advantage_stats={"value_source": "baseline"},
    sign_scale_summary=dict(sign_scale_summary),
)
contract_path = dataset_dir / "m2_labels" / "continuous_advantage_contract.json"
contract_path.parent.mkdir(parents=True, exist_ok=True)
with contract_path.open("w", encoding="utf-8") as f:
    json.dump(contract, f, ensure_ascii=True, indent=2, sort_keys=True)
    f.write("\n")

print(f"[INFO] contract_path: {contract_path}")
print(f"[INFO] contract_n_samples: {len(label_rows)}")
print(f"[INFO] positive_scale: {float(positive_scale)}")
print(f"[INFO] negative_scale_abs: {float(negative_scale_abs)}")
PY

timeout --signal=INT --kill-after=20s "${TIMEOUT_EXPORT_S}s" \
  python3 work/recap/scripts/39_recap_export_lerobot_v2_with_video.py \
    --iter-tag "$ITER_TAG" \
    --max-episodes "$EXPORT_MAX_EPISODES" \
    --dual-task-text \
    --total-timeout-s "$TIMEOUT_EXPORT_S" \
  2>&1 | tee -a "$LOG_DIR/30_export.log"

export PYTHONPATH="submodules/Isaac-GR00T${PYTHONPATH:+:${PYTHONPATH}}"

timeout --signal=INT --kill-after=20s "${TIMEOUT_FINETUNE_S}s" \
  "$WBC_PY" submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
    --base-model-path "$MODEL_PATH" \
    --dataset-path "agent/artifacts/lerobot_datasets/${ITER_TAG}" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --tune-projector \
    --no-tune-diffusion-model \
    --no-use-wandb \
    --output-dir "$OUT_DIR" \
    --max-steps "$FINETUNE_MAX_STEPS" \
    --save-steps "$FINETUNE_SAVE_STEPS" \
    --save-total-limit 1 \
    --global-batch-size "$FINETUNE_GLOBAL_BATCH_SIZE" \
    --gradient-accumulation-steps "$FINETUNE_GRAD_ACCUM_STEPS" \
    --dataloader-num-workers "$FINETUNE_DATALOADER_NUM_WORKERS" \
  2>&1 | tee -a "$LOG_DIR/40_finetune.log"

SELECTED_CHECKPOINT_PATH="$(python3 - "$OUT_DIR" <<'PY'
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
if not output_dir.is_dir():
    raise FileNotFoundError(output_dir)

best_step = -1
best_path = None
for p in sorted(output_dir.glob("checkpoint-*")):
    if not p.is_dir():
        continue
    if not (p / "trainer_state.json").is_file():
        continue
    try:
        step = int(p.name.split("checkpoint-", 1)[-1])
    except Exception:
        step = -1
    if step > best_step:
        best_step = step
        best_path = p

if best_path is None:
    raise RuntimeError(f"No checkpoint-* with trainer_state.json found under: {output_dir}")

print(best_path.as_posix())
PY
)"
printf '%s\n' "[INFO] selected_checkpoint_path=${SELECTED_CHECKPOINT_PATH}" | tee -a "$LOG_DIR/45_selected_checkpoint.log" -a "$LOG_DIR/40_finetune.log"

timeout --signal=INT --kill-after=20s "${TIMEOUT_EVAL_S}s" \
  python3 work/recap/scripts/31_recap_collect_rollouts.py \
    --iter-tag "${ITER_TAG}_eval_baseline" \
    --env-name "$ENV_NAME" \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --server-host "$SERVER_HOST" \
    --server-port "$SERVER_PORT" \
    --n-episodes 1 \
    --max-policy-steps 2 \
    --max-episode-steps 1440 \
    --n-action-steps-config "$N_ACTION_STEPS_CONFIG" \
    --seed 2000 \
    --mujoco-gl "$MUJOCO_GL" \
    --offscreen \
    --no-onscreen \
    --total-timeout-s "$TIMEOUT_EVAL_S" \
    --kill-server-on-exit \
  2>&1 | tee -a "$LOG_DIR/50_eval_baseline.log"

timeout --signal=INT --kill-after=20s "${TIMEOUT_EVAL_S}s" \
  python3 work/recap/scripts/31_recap_collect_rollouts.py \
    --iter-tag "${ITER_TAG}_eval_finetuned" \
    --env-name "$ENV_NAME" \
    --model-path "$SELECTED_CHECKPOINT_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --server-host "$SERVER_HOST" \
    --server-port "$SERVER_PORT" \
    --n-episodes 1 \
    --max-policy-steps 2 \
    --max-episode-steps 1440 \
    --n-action-steps-config "$N_ACTION_STEPS_CONFIG" \
    --seed 3000 \
    --mujoco-gl "$MUJOCO_GL" \
    --offscreen \
    --no-onscreen \
    --total-timeout-s "$TIMEOUT_EVAL_S" \
    --kill-server-on-exit \
  2>&1 | tee -a "$LOG_DIR/51_eval_finetuned.log"

echo "[INFO] smoke finished iter_tag=$ITER_TAG" | tee -a "$LOG_DIR/99_done.log"
