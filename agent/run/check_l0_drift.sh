#!/usr/bin/env bash
# check_l0_drift.sh — 比对 AGENTS.md L0 声明值与现场探针读出值，仅警告，不阻塞。
# 用法: bash agent/run/check_l0_drift.sh （可从任何 cwd 调用）
# 依赖: bash, readlink, nvidia-smi, grep, POSIX coreutils (head, wc, printf)

set -uo pipefail

REPO_ROOT_EXPECTED="/home/howard/Projects/gr00t_wbc_g1_benchmark"
HDD_LIVE_ROOT_EXPECTED="/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_live/agent"
ARCHIVES_ROOT_EXPECTED="/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_archives"
HW_KEYWORD="RTX PRO 6000"
GPU_DEFAULT_KEYWORD="GPU1 / GPU2"

# Anchor at the script's repo root so probes are cwd-independent.
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_ROOT_ACTUAL="$(cd "${SCRIPT_DIR}/../.." && pwd)"

drift=0
ok()    { printf '[OK]    %s\n' "$1"; }
warn()  { printf '[DRIFT] %s\n' "$1"; drift=$((drift+1)); }
note()  { printf '[NOTE]  %s\n' "$1"; }

echo "=== AGENTS.md L0 drift check (non-blocking) ==="

# 1. canonical project root (resolved from script location, not cwd)
if [[ "${REPO_ROOT_ACTUAL}" == "${REPO_ROOT_EXPECTED}" ]]; then
  ok "project root: ${REPO_ROOT_ACTUAL}"
else
  warn "project root: actual=${REPO_ROOT_ACTUAL} expected=${REPO_ROOT_EXPECTED}"
fi

# 2. HDD live root reachable directly (catches unmounted HDD where readlink still resolves)
if [[ -d "${HDD_LIVE_ROOT_EXPECTED}" ]]; then
  ok "HDD live root reachable: ${HDD_LIVE_ROOT_EXPECTED}"
else
  warn "HDD live root not reachable: ${HDD_LIVE_ROOT_EXPECTED} (HDD unmounted?)"
fi

# 3. agent/runtime_logs and agent/artifacts symlinks resolve to HDD live root
for sub in runtime_logs artifacts; do
  link="${REPO_ROOT_ACTUAL}/agent/${sub}"
  if [[ -L "${link}" ]]; then
    target="$(readlink -f "${link}" 2>/dev/null || true)"
    expected="${HDD_LIVE_ROOT_EXPECTED}/${sub}"
    if [[ "${target}" == "${expected}" ]]; then
      ok "agent/${sub} -> ${target}"
    else
      warn "agent/${sub}: actual=${target:-<unresolved>} expected=${expected}"
    fi
  else
    warn "agent/${sub} is not a symlink (expected symlink to HDD live root)"
  fi
done

# 4. archives root reachable
if [[ -d "${ARCHIVES_ROOT_EXPECTED}" ]]; then
  ok "archives root: ${ARCHIVES_ROOT_EXPECTED}"
else
  warn "archives root not reachable: ${ARCHIVES_ROOT_EXPECTED}"
fi

# 5. hardware via nvidia-smi
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_line="$(nvidia-smi -L 2>/dev/null | head -n 1 || true)"
  if [[ "${gpu_line}" == *"${HW_KEYWORD}"* ]]; then
    ok "GPU[0] hardware: ${gpu_line}"
  else
    warn "GPU[0] hardware: ${gpu_line:-<no output>} (expected to contain '${HW_KEYWORD}')"
  fi
  gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l 2>/dev/null || echo 0)"
  note "visible GPU count: ${gpu_count}"
else
  warn "nvidia-smi not available on PATH"
fi

# 6. AGENTS.md L0 self-declarations still present
agents_md="${REPO_ROOT_ACTUAL}/AGENTS.md"
if [[ -f "${agents_md}" ]]; then
  for needle in "${REPO_ROOT_EXPECTED}" "${HDD_LIVE_ROOT_EXPECTED}" "${ARCHIVES_ROOT_EXPECTED}" "${GPU_DEFAULT_KEYWORD}" "${HW_KEYWORD}"; do
    if grep -qF -- "${needle}" "${agents_md}"; then
      ok "AGENTS.md mentions: ${needle}"
    else
      warn "AGENTS.md missing declaration: ${needle}"
    fi
  done
else
  warn "AGENTS.md not found at ${agents_md}"
fi

echo "=== summary: ${drift} drift item(s) ==="
exit 0
