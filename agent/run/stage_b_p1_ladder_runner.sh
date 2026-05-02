#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly SCRIPT_NAME
readonly WBC_PY_DEFAULT="submodules/Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"
readonly STAGE_B_DIR_DEFAULT="agent/artifacts/stage_B_controller_seam_20260501T045341Z_precheck_gate"
readonly P0_REL="prechecks/P0_eval_protocol_determinism"
readonly BASE_MODEL_DEFAULT="nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
readonly SEED_BASE="20000"
readonly P0A_EPISODES_PER_MODE="10"
readonly P0B_EPISODES="30"
readonly N_ENVS="1"
readonly MAX_EPISODE_STEPS="1440"
readonly N_ACTION_STEPS="20"
readonly TOTAL_TIMEOUT_S="21600"
readonly CONNECT_TIMEOUT_S="1200"
readonly GPU_HEADROOM_FLOOR_MIB="4096"
readonly DEV_SHM_FLOOR_MIB="30720"
readonly P0A_GPU="1"
readonly P0B_GPU="2"
readonly P0A_PORT="5564"
readonly P0B_PORT="5565"

STAGE_B_DIR="${STAGE_B_DIR:-${STAGE_B_DIR_DEFAULT}}"
WBC_PY="${WBC_PY:-${WBC_PY_DEFAULT}}"
BASE_MODEL_ID="${BASE_MODEL_ID:-${BASE_MODEL_DEFAULT}}"
POST_CKPT_PATH="${POST_CKPT_PATH:-}"

P0_DIR="${REPO_ROOT}/${STAGE_B_DIR}/${P0_REL}"
CELLS_DIR="${P0_DIR}/cells"
P0A_CELL_DIR="${CELLS_DIR}/P0a_post_recap_nenvs_1"
P0B_CELL_DIR="${CELLS_DIR}/P0b_base_reference_nenvs_1"
LOG_DIR="${REPO_ROOT}/agent/runtime_logs/p1_ladder_phase1"
LOCK_FILE="${LOG_DIR}/.lock"

usage() {
  cat <<'EOF'
Usage:
  agent/run/stage_b_p1_ladder_runner.sh <subcommand> [flags]

Subcommands:
  preflight        Archive Phase 0 log, move P0 gate PENDING_EXEC->RUNNING, create Phase 1 dirs.
  dryrun           Run base positive x1 on GPU2 with fresh dryrun_<UTC>_<PID> output dir.
  phase1           Launch P0a GPU1 and P0b GPU2 formal eval cells with fresh run dirs.
  aggregate        Call p1_phase1_aggregator.py for final summary when available.
  stop --reason X  Write stop_record.json and kill known sibling process trees.

Environment:
  POST_CKPT_PATH   Required for phase1 P0a post-RECAP cell.
  BASE_MODEL_ID    Base checkpoint directory or HF model id. Default: nvidia/GR00T-N1.6-G1-PnPAppleToPlate.
  STAGE_B_DIR      Default: agent/artifacts/stage_B_controller_seam_20260501T045341Z_precheck_gate.
  WBC_PY           Default WBC venv python path.

Invariant:
  Every gr00t_g3_formal_eval.py invocation receives a fresh UTC+PID-stamped --output-dir.
EOF
}

die() {
  echo "[${SCRIPT_NAME}] ERROR: $*" >&2
  exit 1
}

utc_compact() {
  date -u +%Y%m%dT%H%M%SZ
}

utc_iso() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

ensure_dirs() {
  mkdir -p "${LOG_DIR}" "${P0A_CELL_DIR}" "${P0B_CELL_DIR}"
}

log_path() {
  local label="$1"
  printf '%s/%s_%s.log' "${LOG_DIR}" "${label}" "$(utc_compact)"
}

run_id() {
  printf '%s_%s' "$(utc_compact)" "$$"
}

require_wbc_python() {
  [[ -x "${WBC_PY}" ]] || die "missing executable WBC python: ${WBC_PY}"
}

require_phase1_post_ckpt() {
  [[ -n "${POST_CKPT_PATH}" ]] || die "POST_CKPT_PATH is required for phase1"
  [[ -d "${POST_CKPT_PATH}" ]] || die "POST_CKPT_PATH is not a directory: ${POST_CKPT_PATH}"
}

resolve_checkpoint() {
  local raw="$1"
  if [[ -d "${raw}" ]]; then
    realpath "${raw}"
    return 0
  fi
  if [[ "${raw}" == /* || "${raw}" == ./* || "${raw}" == ../* ]]; then
    die "checkpoint directory not found: ${raw}"
  fi

  local hf_home="${HF_HOME:-${HOME}/.cache/huggingface}"
  local model_dir="${hf_home}/hub/models--${raw//\//--}"
  local revision=""
  if [[ -f "${model_dir}/refs/main" ]]; then
    revision="$(tr -d '[:space:]' < "${model_dir}/refs/main")"
  fi
  if [[ -n "${revision}" && -d "${model_dir}/snapshots/${revision}" ]]; then
    realpath "${model_dir}/snapshots/${revision}"
    return 0
  fi
  local snapshot=""
  snapshot="$(find "${model_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1 {print $2}')"
  [[ -n "${snapshot}" && -d "${snapshot}" ]] || die "checkpoint/model id is not available as a local directory or HF cache snapshot: ${raw}"
  realpath "${snapshot}"
}

write_stop_record() {
  local stop_code="$1"
  local cell="${2:-GLOBAL}"
  local evidence="${3:-}"
  mkdir -p "${P0_DIR}"
  STOP_CODE="${stop_code}" CELL="${cell}" EVIDENCE="${evidence}" P0_DIR="${P0_DIR}" python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

p0_dir = Path(os.environ["P0_DIR"])
evidence = [item for item in os.environ.get("EVIDENCE", "").split(":") if item]
payload = {
    "schema_version": "p1_stop_record_v1",
    "cell": os.environ["CELL"],
    "triggered_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "stop_code": os.environ["STOP_CODE"],
    "completed_episodes": 0,
    "success_count": 0,
    "evidence_paths": evidence,
    "no_retry": True,
    "leader_action_required": True,
}
path = p0_dir / "stop_record.json"
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
print(path)
PY
}

append_gate_history() {
  local from_decision="$1"
  local to_decision="$2"
  local reason="$3"
  mkdir -p "${P0_DIR}"
  FROM_DECISION="${from_decision}" TO_DECISION="${to_decision}" REASON="${reason}" P0_DIR="${P0_DIR}" python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["P0_DIR"]) / "p0_gate_decision_history.jsonl"
payload = {
    "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "from": os.environ["FROM_DECISION"],
    "to": os.environ["TO_DECISION"],
    "reason": os.environ["REASON"],
    "actor": "stage_b_p1_ladder_runner.sh",
}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
print(path)
PY
}

transition_p0_gate() {
  local expected="$1"
  local target="$2"
  local reason="$3"
  local gate="${P0_DIR}/p0_gate_decision.json"
  [[ -f "${gate}" ]] || die "missing P0 gate: ${gate}"
  EXPECTED="${expected}" TARGET="${target}" REASON="${reason}" GATE="${gate}" python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["GATE"])
payload = json.loads(path.read_text(encoding="utf-8"))
current = payload.get("decision")
expected = os.environ["EXPECTED"]
target = os.environ["TARGET"]
if current == target:
    print(f"{path}: already {target}")
    raise SystemExit(0)
if current != expected:
    raise SystemExit(f"{path}: expected decision {expected!r}, got {current!r}")
now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
payload["decision"] = target
payload["updated_at_utc"] = now
payload["phase"] = "P0_ladder_phase1"
payload["reason"] = os.environ["REASON"]
payload["training_allowed"] = False
payload["checkpoint_update_allowed"] = False
payload["continue_to_p2"] = False
payload["continue_to_runtime_probes"] = False
payload["method_claim_allowed"] = False
downstream = dict(payload.get("downstream_blocks") or {})
downstream.update(
    training_allowed=False,
    checkpoint_update_allowed=False,
    p2_allowed=False,
    phase2_allowed=False,
    runtime_probe_allowed=False,
    method_claim_allowed=False,
)
payload["downstream_blocks"] = downstream
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
print(f"{path}: {current} -> {target}")
PY
  append_gate_history "${expected}" "${target}" "${reason}" >/dev/null
}

archive_phase0_server_log() {
  local src="${P0_DIR}/runtime_cells/post_recap_gpu1/g3_formal_server.log"
  local archive="${P0_DIR}/runtime_cells/post_recap_gpu1/g3_formal_server.phase0_archived.log"
  local sha="${P0_DIR}/runtime_cells/post_recap_gpu1/g3_formal_server.phase0_archived.sha256"
  [[ -f "${src}" ]] || die "missing Phase 0 server log: ${src}"
  mkdir -p "$(dirname "${archive}")"

  if [[ -f "${archive}" && -f "${sha}" ]]; then
    local pinned
    local archive_hash
    local src_hash
    pinned="$(awk '{print $1}' "${sha}")"
    archive_hash="$(sha256sum "${archive}" | awk '{print $1}')"
    src_hash="$(sha256sum "${src}" | awk '{print $1}')"
    if [[ "${archive_hash}" != "${pinned}" || "${src_hash}" != "${pinned}" ]]; then
      write_stop_record "STOP_PHASE0_LOG_DRIFT" "P0a_post_recap_nenvs_1" "${src}:${archive}:${sha}" >/dev/null
      die "Phase 0 server log drift: source/archive SHA does not match pinned SHA"
    fi
    echo "[P1_PREFLIGHT] Phase 0 log archive already pinned: ${sha}"
    return 0
  fi

  cp "${src}" "${archive}"
  sha256sum "${archive}" > "${sha}"
  echo "[P1_PREFLIGHT] archived Phase 0 server log: ${archive}"
  echo "[P1_PREFLIGHT] sha256 pin: ${sha}"
}

check_no_aggregate_gate_mutation_target() {
  local aggregate_gate="${REPO_ROOT}/${STAGE_B_DIR}/prechecks/precheck_gate_decision.json"
  if [[ -f "${aggregate_gate}" ]]; then
    echo "[P1_PREFLIGHT] aggregate gate observed but not mutated: ${aggregate_gate}"
  fi
}

check_dev_shm() {
  local available
  available="$(df -m /dev/shm | awk 'NR==2 {print $4}')"
  [[ -n "${available}" ]] || die "cannot read /dev/shm free space"
  if (( available < DEV_SHM_FLOOR_MIB )); then
    write_stop_record "STOP_SERVER_FAIL" "GLOBAL" "/dev/shm" >/dev/null
    die "/dev/shm free ${available} MiB below floor ${DEV_SHM_FLOOR_MIB} MiB"
  fi
  echo "[P1_PREFLIGHT] /dev/shm free ${available} MiB"
}

check_gpu_headroom() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[P1_PREFLIGHT] nvidia-smi not found; GPU headroom check skipped" >&2
    return 0
  fi
  local row total used free
  row="$(nvidia-smi --id="${gpu}" --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits | head -n 1)"
  IFS=',' read -r total used free <<<"${row}"
  total="${total//[[:space:]]/}"
  used="${used//[[:space:]]/}"
  free="${free//[[:space:]]/}"
  [[ -n "${total}" && -n "${used}" ]] || die "cannot parse nvidia-smi memory row for GPU${gpu}: ${row}"
  if (( used > total - GPU_HEADROOM_FLOOR_MIB )); then
    write_stop_record "STOP_VRAM_HEADROOM" "GPU${gpu}" "nvidia-smi" >/dev/null
    die "GPU${gpu} used ${used} MiB exceeds total-${GPU_HEADROOM_FLOOR_MIB} headroom"
  fi
  echo "[P1_PREFLIGHT] GPU${gpu} memory total=${total} MiB used=${used} MiB free=${free} MiB"
}

preflight() {
  ensure_dirs
  require_wbc_python
  check_no_aggregate_gate_mutation_target
  [[ -f "${REPO_ROOT}/agent/artifacts/handoff/p0_phase0_invalid_incident_record.md" ]] \
    || echo "[P1_PREFLIGHT] incident record not found yet; W3 owns this artifact" >&2
  archive_phase0_server_log
  transition_p0_gate "P0_PENDING_EXEC" "P0_RUNNING" "Phase 1 preflight started by stage_b_p1_ladder_runner.sh"
  check_dev_shm
  check_gpu_headroom "${P0A_GPU}"
  check_gpu_headroom "${P0B_GPU}"
  mkdir -p "${P0A_CELL_DIR}/runs" "${P0B_CELL_DIR}/runs" "${P0B_CELL_DIR}/dryruns"
  echo "[P1_PREFLIGHT] ready"
}

formal_eval_cmd() {
  local checkpoint="$1"
  local output_dir="$2"
  local runtime_log_dir="$3"
  local gpu="$4"
  local port="$5"
  shift 5
  CUDA_VISIBLE_DEVICES="${gpu}" \
  MUJOCO_GL="${MUJOCO_GL:-egl}" \
  PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
  PYTHONUNBUFFERED=1 \
  GR00T_SKIP_WBC_REEXEC=1 \
  NO_ALBUMENTATIONS_UPDATE=1 \
  PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/submodules/Isaac-GR00T:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobosuite:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobocasa:${REPO_ROOT}/submodules/Isaac-GR00T/external_dependencies/robocasa:${PYTHONPATH:-}" \
  timeout --preserve-status "${TOTAL_TIMEOUT_S}" \
  "${WBC_PY}" work/recap/scripts/gr00t_g3_formal_eval.py \
    --checkpoint "${checkpoint}" \
    --output-dir "${output_dir}" \
    --runtime-log-dir "${runtime_log_dir}" \
    --server-host "127.0.0.1" \
    --server-port "${port}" \
    --seed-base "${SEED_BASE}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --n-action-steps "${N_ACTION_STEPS}" \
    --connect-timeout-s "${CONNECT_TIMEOUT_S}" \
    --total-timeout-s "${TOTAL_TIMEOUT_S}" \
    --required-cuda-visible-devices "${gpu}" \
    "$@"
}

wait_for_ready_probe() {
  local log_file="$1"
  local port="$2"
  local timeout_s="${3:-120}"
  local started
  started="$(date +%s)"
  while true; do
    if grep -q '\[SERVER_READY\]' "${log_file}" 2>/dev/null \
      && python3 - "${port}" <<'PY'
import socket
import sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect(("127.0.0.1", port))
finally:
    sock.close()
PY
    then
      echo "[P1_READY] port=${port} log=${log_file}"
      return 0
    fi
    if (( $(date +%s) - started >= timeout_s )); then
      write_stop_record "STOP_SERVER_FAIL" "GLOBAL" "${log_file}" >/dev/null
      die "readiness probe timed out for port ${port}; log=${log_file}"
    fi
    sleep 2
  done
}

run_dryrun() {
  ensure_dirs
  require_wbc_python
  check_dev_shm
  check_gpu_headroom "${P0B_GPU}"
  local rid checkpoint output_dir runtime_dir log_file
  rid="$(run_id)"
  checkpoint="$(resolve_checkpoint "${BASE_MODEL_ID}")"
  output_dir="${P0B_CELL_DIR}/dryrun_${rid}"
  runtime_dir="${output_dir}/runtime"
  log_file="$(log_path "P0b_base_dryrun_gpu${P0B_GPU}_nenvs${N_ENVS}")"
  mkdir -p "${output_dir}" "${runtime_dir}"
  echo "[P1_DRYRUN] output_dir=${output_dir}" | tee -a "${log_file}"
  if formal_eval_cmd "${checkpoint}" "${output_dir}" "${runtime_dir}" "${P0B_GPU}" "${P0B_PORT}" \
      --indicator-modes positive \
      --episode-count 1 2>&1 | tee -a "${log_file}"; then
    maybe_run_aggregator "dryrun" "${output_dir}" "${log_file}"
    echo "[P1_DRYRUN] PASS output_dir=${output_dir}"
  else
    write_stop_record "STOP_DRYRUN_BLOCKER" "P0b_base_reference_nenvs_1" "${log_file}:${output_dir}/formal_eval_summary.json" >/dev/null
    write_dryrun_blocker "${output_dir}" "${log_file}"
    transition_p0_gate "P0_RUNNING" "P0_PENDING_EXEC" "Phase 1 dryrun failed; fail-fast no retry" || true
    return 2
  fi
}

write_dryrun_blocker() {
  local output_dir="$1"
  local log_file="$2"
  local blocker="${P0_DIR}/phase1_dryrun_blocker.md"
  cat > "${blocker}" <<EOF
# P0 Phase 1 dry-run blocker

- created_at_utc: $(utc_iso)
- output_dir: ${output_dir}
- log_file: ${log_file}
- stop_code: STOP_DRYRUN_BLOCKER
- no_retry: true
- next_route: wait_for_leader_decision
EOF
  echo "[P1_DRYRUN] blocker written: ${blocker}"
}

maybe_run_aggregator() {
  local mode="$1"
  local output_dir="$2"
  local log_file="$3"
  local aggregator="work/recap/stage_b/p1_phase1_aggregator.py"
  if [[ ! -f "${aggregator}" ]]; then
    echo "[P1_AGGREGATOR] ${aggregator} not present yet; skipping ${mode} validation" | tee -a "${log_file}"
    return 0
  fi
  "${WBC_PY}" "${aggregator}" \
    --mode "${mode}" \
    --p0-dir "${P0_DIR}" \
    --output-dir "${output_dir}" \
    --launcher-log "${log_file}" 2>&1 | tee -a "${log_file}"
}

launch_formal_background() {
  local label="$1"
  local checkpoint="$2"
  local output_dir="$3"
  local runtime_dir="$4"
  local gpu="$5"
  local port="$6"
  local episodes="$7"
  shift 7
  local log_file
  log_file="$(log_path "${label}_gpu${gpu}_nenvs${N_ENVS}")"
  mkdir -p "${output_dir}" "${runtime_dir}"
  (
    echo "[P1_PHASE1] launch label=${label} output_dir=${output_dir} log=${log_file}"
    formal_eval_cmd "${checkpoint}" "${output_dir}" "${runtime_dir}" "${gpu}" "${port}" \
      --indicator-modes "$@" \
      --episode-count "${episodes}" 2>&1 | tee -a "${log_file}"
  ) &
  local pid=$!
  printf '%s:%s:%s:%s\n' "${label}" "${pid}" "${log_file}" "${output_dir}" >> "${LOG_DIR}/phase1_children_${RUN_GROUP_ID}.txt"
  echo "${pid}"
}

kill_child_tree() {
  local pid="$1"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  pkill -TERM -P "${pid}" 2>/dev/null || true
  kill -TERM "${pid}" 2>/dev/null || true
  sleep 5
  pkill -KILL -P "${pid}" 2>/dev/null || true
  kill -KILL "${pid}" 2>/dev/null || true
}

run_phase1() {
  ensure_dirs
  require_wbc_python
  require_phase1_post_ckpt
  check_dev_shm
  check_gpu_headroom "${P0A_GPU}"
  check_gpu_headroom "${P0B_GPU}"
  local base_ckpt p0a_output p0b_output p0a_runtime p0b_runtime p0a_pid p0b_pid children_file
  RUN_GROUP_ID="$(run_id)"
  export RUN_GROUP_ID
  children_file="${LOG_DIR}/phase1_children_${RUN_GROUP_ID}.txt"
  : > "${children_file}"
  base_ckpt="$(resolve_checkpoint "${BASE_MODEL_ID}")"
  p0a_output="${P0A_CELL_DIR}/runs/${RUN_GROUP_ID}"
  p0b_output="${P0B_CELL_DIR}/runs/${RUN_GROUP_ID}"
  p0a_runtime="${p0a_output}/runtime"
  p0b_runtime="${p0b_output}/runtime"

  p0a_pid="$(launch_formal_background "P0a_post_recap" "${POST_CKPT_PATH}" "${p0a_output}" "${p0a_runtime}" "${P0A_GPU}" "${P0A_PORT}" "${P0A_EPISODES_PER_MODE}" positive omit negative)"
  local p0a_log
  p0a_log="$(awk -F: -v pid="${p0a_pid}" '$2 == pid {print $3}' "${children_file}")"
  wait_for_ready_probe "${p0a_log}" "${P0A_PORT}" 120

  p0b_pid="$(launch_formal_background "P0b_base_reference" "${base_ckpt}" "${p0b_output}" "${p0b_runtime}" "${P0B_GPU}" "${P0B_PORT}" "${P0B_EPISODES}" positive)"
  local p0b_log
  p0b_log="$(awk -F: -v pid="${p0b_pid}" '$2 == pid {print $3}' "${children_file}")"
  wait_for_ready_probe "${p0b_log}" "${P0B_PORT}" 120

  local first_status=0 finished_pid=""
  wait -n -p finished_pid "${p0a_pid}" "${p0b_pid}" || first_status=$?
  if (( first_status != 0 )); then
    write_stop_record "STOP_SERVER_FAIL" "GLOBAL" "${p0a_log}:${p0b_log}" >/dev/null
    kill_child_tree "${p0a_pid}"
    kill_child_tree "${p0b_pid}"
    return "${first_status}"
  fi
  if [[ "${finished_pid}" == "${p0a_pid}" ]]; then
    wait "${p0b_pid}" || first_status=$?
  else
    wait "${p0a_pid}" || first_status=$?
  fi
  if (( first_status != 0 )); then
    write_stop_record "STOP_SERVER_FAIL" "GLOBAL" "${p0a_log}:${p0b_log}" >/dev/null
    return "${first_status}"
  fi
  maybe_run_aggregator "phase1" "${P0_DIR}" "${p0a_log}"
  echo "[P1_PHASE1] PASS run_group=${RUN_GROUP_ID}"
}

run_aggregate() {
  ensure_dirs
  local log_file
  log_file="$(log_path "P0_phase1_aggregate")"
  maybe_run_aggregator "aggregate" "${P0_DIR}" "${log_file}"
  transition_p0_gate "P0_RUNNING" "P0_AWAITING_VERIFIER" "Phase 1 aggregate completed; awaiting verifier"
}

run_stop() {
  local reason=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --reason)
        reason="${2:-}"
        shift 2
        ;;
      *)
        die "unknown stop flag: $1"
        ;;
    esac
  done
  [[ -n "${reason}" ]] || die "stop requires --reason <CODE>"
  ensure_dirs
  write_stop_record "${reason}" "GLOBAL" "${LOG_DIR}" >/dev/null
  if compgen -G "${LOG_DIR}/phase1_children_*.txt" >/dev/null; then
    while IFS=: read -r _label pid _log _out; do
      kill_child_tree "${pid}"
    done < <(cat "${LOG_DIR}"/phase1_children_*.txt)
  fi
  echo "[P1_STOP] ${reason}"
}

with_lock() {
  ensure_dirs
  exec 9>"${LOCK_FILE}"
  flock 9
  "$@"
}

main() {
  local command="${1:-}"
  if [[ -z "${command}" || "${command}" == "-h" || "${command}" == "--help" ]]; then
    usage
    return 0
  fi
  shift
  case "${command}" in
    preflight) with_lock preflight "$@" ;;
    dryrun) with_lock run_dryrun "$@" ;;
    phase1) with_lock run_phase1 "$@" ;;
    aggregate) with_lock run_aggregate "$@" ;;
    stop) run_stop "$@" ;;
    *) usage >&2; return 2 ;;
  esac
}

main "$@"
