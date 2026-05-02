#!/usr/bin/env bash
# Probe A — Pure-SFT control fine-tune of GR00T-N1.6-G1-PnPAppleToPlate
# Diagnostic for post-RECAP G3 ckpt-6600 0/30 lifts vs 17/30 base.
# Pure SFT: ablate conditioning via --indicator-dropout-p 1.0, no continuation.
# Constraint: GPU2 only. Repo-root dataset path (no /media/* substrings).

set -euo pipefail

# ---- Pinned absolute paths (no /media/ substring permitted in any command line) ----
REPO_ROOT="/home/howard/Projects/gr00t_wbc_g1_benchmark"
VENV_PY="${REPO_ROOT}/submodules/Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"
WRAPPER="${REPO_ROOT}/work/recap/scripts/34b_recap_numeric_adv_smoke.py"

DATASET_REL="agent/artifacts/lerobot_datasets/recap_stage3_iter_002"

# Run-scoped UTC stamp; allow override via env for retries
RUN_UTC="${PROBE_A_RUN_UTC:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Canonical training output MUST live under the v2 full-update authority root
# (work.recap.finetune_full.resolve_full_update_authority_output_dir gate).
# We stash the run under the gr00t_recap_live authority root in a probes/ subdir
# and expose it via a symlink at the user-facing probes namespace below.
AUTHORITY_PROBES_REL="agent/artifacts/gr00t_recap_live/single_gpu_v2_full_update/probes/probe_A_pure_sft_control"
OUTPUT_DIR_REL="${AUTHORITY_PROBES_REL}/training_run_${RUN_UTC}"
RUNTIME_LOG_DIR_REL="agent/runtime_logs/probes/probe_A/${RUN_UTC}"
SUMMARY_JSON_REL="${OUTPUT_DIR_REL}/probe_A_pure_sft_training_summary.json"
PATCHED_OUT_REL="agent/artifacts/gr00t_recap_live/hf_patches"

# User-facing namespace (per US-007); symlink-only redirect into the authority root.
PROBES_NAMESPACE_DIR="${REPO_ROOT}/agent/artifacts/probes/probe_A_pure_sft_control"
PROBES_NAMESPACE_LINK="${PROBES_NAMESPACE_DIR}/training_run_${RUN_UTC}"

mkdir -p "${REPO_ROOT}/${OUTPUT_DIR_REL}" "${REPO_ROOT}/${RUNTIME_LOG_DIR_REL}" "${PROBES_NAMESPACE_DIR}"
# Expose the authority output under the user-facing probes namespace via symlink.
ln -snf "${REPO_ROOT}/${OUTPUT_DIR_REL}" "${PROBES_NAMESPACE_LINK}"

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
# Threadripper / RTX PRO 6000 Blackwell: keep DataLoader workers=0 (per original) so
# torch thread budget below is honored.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "[probe_A] RUN_UTC=${RUN_UTC}"
echo "[probe_A] OUTPUT_DIR(authority)=${REPO_ROOT}/${OUTPUT_DIR_REL}"
echo "[probe_A] PROBES_NAMESPACE_LINK=${PROBES_NAMESPACE_LINK}"
echo "[probe_A] RUNTIME_LOG_DIR=${REPO_ROOT}/${RUNTIME_LOG_DIR_REL}"
echo "[probe_A] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Pure-SFT command: derived from G3 conditioned 6600 delegate_cmd. Differences:
#   * --indicator-dropout-p 1.0           (full ablation; route validates but is dropped every step)
#   * --no-condition-focused-continuation (start from base ckpt, not continuation-4400)
#   * --continuation-checkpoint-path ''   (unused; required to be empty for the no-flag branch)
#   * --max-steps 3300                    (halved from 6600 to fit ~6h wall-clock budget)
#   * --seed 20260501                     (date-stamp seed for this run)
#   * dataset & output paths              (repo-root, not /media)
#   * --runtime-log-prefix probe_A_pure_sft_train
exec "${VENV_PY}" "${WRAPPER}" \
  --delegate-mode \
  --dataset-path "${DATASET_REL}" \
  --output-dir "${OUTPUT_DIR_REL}" \
  --label-semantics-output-dir "" \
  --summary-json "${SUMMARY_JSON_REL}" \
  --runtime-log-dir "${RUNTIME_LOG_DIR_REL}" \
  --runtime-log-prefix probe_A_pure_sft_train \
  --python "${VENV_PY}" \
  --base-model nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
  --base-model-revision "" \
  --hf-hub-cache-dir "" \
  --patched-out-root "${PATCHED_OUT_REL}" \
  --no-force-top-llm-layers-zero \
  --conditioning-route text_indicator_v1 \
  --runtime-indicator-mode positive \
  --indicator-dropout-p 1.0 \
  --text-indicator-prompt-raw-column recap_m2.prompt_raw \
  --text-indicator-step-text-fallback \
  --embodiment-tag UNITREE_G1 \
  --max-steps 3300 \
  --save-steps 1100 \
  --save-total-limit 1 \
  --global-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 0 \
  --learning-rate 1e-05 \
  --recap-train-scope strict_full \
  --no-balanced-advantage-batches \
  --no-write-conditioning-functional-probe \
  --no-write-paired-action-probe \
  --no-write-label-semantics-audit \
  --no-write-shuffled-advantage-negative-control \
  --positive-oversample-factor 1 \
  --no-positive-curriculum \
  --negative-retain-probability 1.0 \
  --positive-curriculum-seed -1 \
  --no-late-stage-positive-emphasis \
  --late-stage-threshold 0.8 \
  --no-condition-focused-continuation \
  --continuation-checkpoint-path "" \
  --condition-hot-lr-scale 1.0 \
  --diffusion-trunk-lr-scale 1.0 \
  --num-gpus 1 \
  --seed 20260501 \
  --tune-top-llm-layers 0 \
  --no-tune-projector \
  --tune-diffusion-model \
  --tune-vlln \
  --no-use-wandb \
  --transformers-local-files-only \
  --emit-optimizer-param-group-report \
  --emit-in-memory-delta-report \
  --emit-saved-checkpoint-delta-report
