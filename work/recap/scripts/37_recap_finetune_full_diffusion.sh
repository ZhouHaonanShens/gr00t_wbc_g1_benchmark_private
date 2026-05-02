#!/usr/bin/env bash
set -euxo pipefail

# USER Config (edit)
ITER_TAG="recap_iter_002_mixdone_20260226_024757_with_video_001"
BASE_MODEL="nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
EMBODIMENT_TAG="UNITREE_G1"
DATASET_DIR_REL="agent/artifacts/lerobot_datasets"

MAX_STEPS=20
SAVE_STEPS=10
GLOBAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
DATALOADER_NUM_WORKERS=0

TOTAL_TIMEOUT_S=7200

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="agent/runtime_logs/${ITER_TAG}"
OUT_DIR="agent/artifacts/checkpoints/${ITER_TAG}/finetune_full_diffusion_002"
DATASET_DIR="${DATASET_DIR_REL}/${ITER_TAG}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

WBC_PY_DEFAULT=".envs/wbc/bin/python"
WBC_PY="${WBC_PY:-$WBC_PY_DEFAULT}"

LOG_SYS="$LOG_DIR/40_pretrain_system_probe.log"
LOG_TRAIN="$LOG_DIR/50_finetune_tune_diffusion.log"

{
  echo "[INFO] ts=$(date -Is)"
  echo "[INFO] repo_root=$REPO_ROOT"
  echo "[INFO] iter_tag=$ITER_TAG"
  echo "[INFO] base_model=$BASE_MODEL"
  echo "[INFO] embodiment_tag=$EMBODIMENT_TAG"
  echo "[INFO] dataset_dir=$DATASET_DIR"
  echo "[INFO] out_dir=$OUT_DIR"
  echo "[INFO] wbc_py=$WBC_PY"
  echo "[INFO] max_steps=$MAX_STEPS save_steps=$SAVE_STEPS"
  echo "[INFO] global_batch_size=$GLOBAL_BATCH_SIZE grad_accum_steps=$GRAD_ACCUM_STEPS"
  echo "[INFO] dataloader_num_workers=$DATALOADER_NUM_WORKERS"
} | tee -a "$LOG_SYS"

(nvidia-smi || true) 2>&1 | tee -a "$LOG_SYS"

"$WBC_PY" - <<'PY' 2>&1 | tee -a "$LOG_SYS"
import os, sys
print("python:", sys.version.replace("\n", " "))
print("sys.executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu:", torch.cuda.get_device_name(0))
        print("total_mem_gb:", round(p.total_memory / 1e9, 2))
        try:
            print("arch_list:", torch.cuda.get_arch_list())
        except Exception as e:
            print("arch_list_error:", repr(e))
except Exception as e:
    print("torch_import_error:", repr(e))
print("env.MUJOCO_GL:", os.environ.get("MUJOCO_GL"))
PY

export PYTHONPATH="submodules/Isaac-GR00T${PYTHONPATH:+:${PYTHONPATH}}"

echo "[INFO] finetune_cmd=launch_finetune.py --tune-diffusion-model --no-tune-projector" | tee -a "$LOG_TRAIN"

timeout --signal=INT --kill-after=20s "${TOTAL_TIMEOUT_S}s" \
  "$WBC_PY" submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_DIR" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --tune-diffusion-model \
    --no-tune-projector \
    --no-use-wandb \
    --output-dir "$OUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit 1 \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM_STEPS" \
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  2>&1 | tee -a "$LOG_TRAIN"

echo "[INFO] done" | tee -a "$LOG_TRAIN"
