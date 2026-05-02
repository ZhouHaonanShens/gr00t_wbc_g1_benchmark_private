#!/usr/bin/env bash
set -euo pipefail

# =====================
# USER Config (edit)
# =====================

ITER_TAG_WAS_SET=0
if [[ -v ITER_TAG ]]; then
  ITER_TAG_WAS_SET=1
fi
DATASET_DIR_REL_WAS_SET=0
if [[ -v DATASET_DIR_REL ]]; then
  DATASET_DIR_REL_WAS_SET=1
fi
RUNTIME_LOGS_REL_WAS_SET=0
if [[ -v RUNTIME_LOGS_REL ]]; then
  RUNTIME_LOGS_REL_WAS_SET=1
fi

: "${ITER_TAG:=recap_iter_000}"
: "${WBC_PY_REL:=submodules/Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python}"
: "${DATASET_DIR_REL:=agent/artifacts/lerobot_datasets/${ITER_TAG}}"
: "${RUNTIME_LOGS_REL:=agent/runtime_logs/${ITER_TAG}}"
: "${TOTAL_TIMEOUT_S:=120}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

AUTO_SELECTED_DATASET_REL=""

select_latest_dataset_rel() {
  REPO_ROOT="${REPO_ROOT}" python3 - <<'PY'
from pathlib import Path
import os
import sys

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
base = repo_root / "agent" / "artifacts" / "lerobot_datasets"
required = [
    Path("meta/info.json"),
    Path("meta/modality.json"),
    Path("meta/stats.json"),
    Path("meta/tasks.jsonl"),
    Path("meta/episodes.jsonl"),
]

if not base.is_dir():
    sys.exit(1)

candidates: list[tuple[int, int, str]] = []
for entry in base.iterdir():
    if not entry.is_dir():
        continue
    name = entry.name
    if not name.startswith("recap_"):
        continue
    if "_eval_" in name or name.startswith("recap_reward_"):
        continue
    if not all((entry / rel).exists() for rel in required):
        continue
    try:
        mtime = int(entry.stat().st_mtime_ns)
    except FileNotFoundError:
        continue
    candidates.append((1 if "_k0" in name else 0, mtime, name))

if not candidates:
    sys.exit(1)

candidates.sort(key=lambda item: (item[0], item[1], item[2]))
chosen = candidates[-1][2]
print(f"agent/artifacts/lerobot_datasets/{chosen}")
PY
}

if [[ ${ITER_TAG_WAS_SET} -eq 0 && ${DATASET_DIR_REL_WAS_SET} -eq 0 ]]; then
  DEFAULT_DATASET_DIR_ABS="${REPO_ROOT}/${DATASET_DIR_REL}"
  if [[ ! -d "${DEFAULT_DATASET_DIR_ABS}" ]]; then
    AUTO_SELECTED_DATASET_REL="$(select_latest_dataset_rel || true)"
    if [[ -n "${AUTO_SELECTED_DATASET_REL}" ]]; then
      DATASET_DIR_REL="${AUTO_SELECTED_DATASET_REL}"
      ITER_TAG="$(basename -- "${DATASET_DIR_REL}")"
      if [[ ${RUNTIME_LOGS_REL_WAS_SET} -eq 0 ]]; then
        RUNTIME_LOGS_REL="agent/runtime_logs/${ITER_TAG}"
      fi
    fi
  fi
fi

WBC_PY="${REPO_ROOT}/${WBC_PY_REL}"
DATASET_DIR_ABS="${REPO_ROOT}/${DATASET_DIR_REL}"
RUNTIME_DIR_ABS="${REPO_ROOT}/${RUNTIME_LOGS_REL}"
LOG_PATH="${RUNTIME_DIR_ABS}/m4_finetune_smoke.log"

usage() {
  cat <<EOF
Usage:
  bash agent/run/34_recap_finetune_smoke.sh [--help]

What it does (no training, no model downloads):
  1) timeout 30s "\$WBC_PY" submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py --help
  2) PYTHONPATH=submodules/Isaac-GR00T timeout 60s "\$WBC_PY" -c '<dataset preflight>'

Key vars (defaults set in script; can override via env):
  ITER_TAG=${ITER_TAG}
  WBC_PY_REL=${WBC_PY_REL}
  DATASET_DIR_REL=${DATASET_DIR_REL}
  RUNTIME_LOGS_REL=${RUNTIME_LOGS_REL}
  TOTAL_TIMEOUT_S=${TOTAL_TIMEOUT_S}

Default fallback behavior:
  If ITER_TAG/DATASET_DIR_REL are not explicitly set and recap_iter_000 is missing,
  auto-select the newest valid local recap dataset under agent/artifacts/lerobot_datasets/.

Derived paths:
  REPO_ROOT=${REPO_ROOT}
  WBC_PY=${WBC_PY}
  DATASET_DIR_ABS=${DATASET_DIR_ABS}
  LOG_PATH=${LOG_PATH}
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" != "--_no_total_timeout" ]]; then
  if command -v timeout >/dev/null 2>&1; then
    exec timeout --signal=TERM --kill-after=5s "${TOTAL_TIMEOUT_S}s" bash "${SCRIPT_PATH}" --_no_total_timeout "$@"
  fi
fi
if [[ "${1:-}" == "--_no_total_timeout" ]]; then
  shift
fi

mkdir -p "${RUNTIME_DIR_ABS}"

{
  echo "[INFO] repo_root: ${REPO_ROOT}"
  echo "[INFO] iter_tag: ${ITER_TAG}"
  echo "[INFO] wbc_py: ${WBC_PY}"
  echo "[INFO] dataset_dir_abs: ${DATASET_DIR_ABS}"
  echo "[INFO] log_path: ${LOG_PATH}"
  if [[ -n "${AUTO_SELECTED_DATASET_REL}" ]]; then
    echo "[INFO] auto_selected_dataset_rel: ${AUTO_SELECTED_DATASET_REL}"
  fi

  if ! command -v timeout >/dev/null 2>&1; then
    echo "[ERROR] missing dependency: timeout (coreutils)" >&2
    exit 1
  fi
  if [[ ! -x "${WBC_PY}" ]]; then
    echo "[ERROR] WBC python not found or not executable: ${WBC_PY}" >&2
    exit 1
  fi
  if [[ ! -d "${DATASET_DIR_ABS}" ]]; then
    echo "[ERROR] dataset dir not found: ${DATASET_DIR_ABS}" >&2
    exit 1
  fi

  cd "${REPO_ROOT}"

  echo "[RUN] launch_finetune.py --help"
  PYTHONPATH=submodules/Isaac-GR00T \
    timeout 30s "${WBC_PY}" submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py --help

  echo "[RUN] dataset preflight (LeRobotEpisodeLoader readback)"
  PYTHONPATH=submodules/Isaac-GR00T \
    timeout 60s "${WBC_PY}" -c 'import pathlib, numpy as np; from gr00t.configs.base_config import get_default_config; from gr00t.data.types import ModalityConfig; from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader; ds=pathlib.Path("'"${DATASET_DIR_REL}"'"); must=[ds/"meta/info.json", ds/"meta/modality.json", ds/"meta/stats.json", ds/"meta/tasks.jsonl", ds/"meta/episodes.jsonl"]; missing=[str(p) for p in must if not p.exists()]; assert not missing, missing; embodiment_tag="unitree_g1"; cfg=get_default_config().load_dict({"data": {"datasets": [{"dataset_paths": [str(ds)], "mix_ratio": 1.0, "embodiment_tag": embodiment_tag}]}}); cfg.validate(); mcfg={"state": ModalityConfig(delta_indices=[0], modality_keys=["wbc_state"]), "action": ModalityConfig(delta_indices=[0], modality_keys=["wbc_action"]), "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description"])}; loader=LeRobotEpisodeLoader(str(ds), mcfg); df=loader[0]; assert len(df)>0; s=df["state.wbc_state"].iloc[0]; a=df["action.wbc_action"].iloc[0]; lang=df["language.annotation.human.action.task_description"].iloc[0]; assert isinstance(s,np.ndarray) and s.ndim==1; assert isinstance(a,np.ndarray) and a.ndim==1; assert isinstance(lang,str) and len(lang)>0; print("ok",len(df),s.shape,a.shape,lang[:80])'

  echo "[OK] smoke complete"
} 2>&1 | tee -a "${LOG_PATH}"
