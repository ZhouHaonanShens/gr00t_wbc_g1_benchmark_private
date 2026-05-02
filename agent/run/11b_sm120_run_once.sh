#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PY="${REPO_ROOT}/submodules/Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"

SERVER_LOG="${REPO_ROOT}/agent/runtime_logs/server/11b_server_gr00t_sm120_ok.log"
PING_LOG="${REPO_ROOT}/agent/runtime_logs/eval/11b_wait_for_ping.log"
ROLLOUT_LOG="${REPO_ROOT}/agent/runtime_logs/eval/11b_rollout_sm120_ok.log"
VIDEO_ARCHIVE_DIR="${REPO_ROOT}/agent/artifacts/videos"

mkdir -p "${REPO_ROOT}/agent/runtime_logs/server" "${REPO_ROOT}/agent/runtime_logs/eval"
mkdir -p "${VIDEO_ARCHIVE_DIR}"

kill_server_graceful() {
  timeout 10s "${PY}" - <<'PY' >/dev/null 2>&1 || true
from gr00t.policy.server_client import PolicyClient
c = PolicyClient(host='127.0.0.1', port=5555)
try:
    c.kill_server()
except Exception:
    pass
PY
}

kill_server_port_5555_if_gr00t() {
  local line pid cmd
  line="$(ss -ltnp 2>/dev/null | awk '$4 ~ /:5555$/ {print}')" || true
  if [[ -z "${line}" ]]; then
    return 0
  fi
  pid="$(echo "${line}" | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n1)"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  cmd="$(ps -p "${pid}" -o cmd= 2>/dev/null || true)"
  case "${cmd}" in
    *submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py*)
      kill "${pid}" 2>/dev/null || true
      ;;
    *)
      echo "[WARN] 5555 端口被非 GR00T 进程占用，拒绝 kill: pid=${pid} cmd=${cmd}" >&2
      ;;
  esac
}

cleanup() {
  set +e
  kill_server_graceful
  kill_server_port_5555_if_gr00t
  if [[ -n "${SERVER_PIPE_PID:-}" ]]; then
    kill "${SERVER_PIPE_PID}" 2>/dev/null || true
    wait "${SERVER_PIPE_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

kill_server_graceful
kill_server_port_5555_if_gr00t

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${REPO_ROOT}/submodules/Isaac-GR00T"

rm -f "${SERVER_LOG}" "${PING_LOG}" "${ROLLOUT_LOG}"

(
  PYTHONUNBUFFERED=1 "${PY}" "${REPO_ROOT}/submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py" \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 \
    --use-sim-policy-wrapper \
    --host 127.0.0.1 \
    --port 5555 \
    2>&1 | tee "${SERVER_LOG}"
) &
SERVER_PIPE_PID=$!

timeout 620s "${PY}" - <<'PY' 2>&1 | tee "${PING_LOG}"
import time
from gr00t.policy.server_client import PolicyClient
c = PolicyClient(host='127.0.0.1', port=5555)
t0 = time.time()
while True:
    if c.ping():
        print('ping ok')
        break
    if time.time() - t0 > 600:
        raise SystemExit('timeout waiting for ping')
    time.sleep(1)
PY

timeout --signal=INT --kill-after=20s 900s env PYTHONUNBUFFERED=1 \
  "${PY}" "${REPO_ROOT}/submodules/Isaac-GR00T/gr00t/eval/rollout_policy.py" \
  --n_episodes 1 \
  --n_envs 1 \
  --max_episode_steps 50 \
  --n_action_steps 30 \
  --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
  --policy_client_host 127.0.0.1 \
  --policy_client_port 5555 \
  2>&1 | tee "${ROLLOUT_LOG}"

kill_server_graceful

VIDEO_DIR="$(sed -n 's/^Video saved to:[[:space:]]*//p' "${ROLLOUT_LOG}" | tail -n 1 | xargs || true)"
if [[ -n "${VIDEO_DIR}" && -d "${VIDEO_DIR}" ]]; then
  DEST_DIR="${VIDEO_ARCHIVE_DIR}/$(basename "${VIDEO_DIR}")"
  rm -rf "${DEST_DIR}" || true
  cp -a "${VIDEO_DIR}" "${DEST_DIR}"
  echo "OK: ${DEST_DIR}"
else
  echo "[WARN] Video dir not found in log or missing on disk: '${VIDEO_DIR}'" >&2
fi

echo "OK: ${SERVER_LOG}"
echo "OK: ${PING_LOG}"
echo "OK: ${ROLLOUT_LOG}"
