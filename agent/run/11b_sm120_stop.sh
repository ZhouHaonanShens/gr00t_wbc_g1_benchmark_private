#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PY="${REPO_ROOT}/submodules/Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"

timeout 10s "${PY}" - <<'PY' >/dev/null 2>&1 || true
from gr00t.policy.server_client import PolicyClient
c = PolicyClient(host='127.0.0.1', port=5555)
try:
    c.kill_server()
except Exception:
    pass
PY

line="$(ss -ltnp 2>/dev/null | awk '$4 ~ /:5555$/ {print}')" || true
if [[ -n "${line}" ]]; then
  pid="$(echo "${line}" | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n1)"
  if [[ -n "${pid}" ]]; then
    cmd="$(ps -p "${pid}" -o cmd= 2>/dev/null || true)"
    case "${cmd}" in
      *submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py*)
        kill "${pid}" 2>/dev/null || true
        ;;
      *)
        echo "[WARN] 5555 端口被非 GR00T 进程占用，拒绝 kill: pid=${pid} cmd=${cmd}" >&2
        ;;
    esac
  fi
fi

if ss -ltnp | grep -q ':5555'; then
  echo "port 5555 still busy" >&2
  ss -ltnp | grep ':5555' || true
  exit 1
fi

echo "port 5555 free"
