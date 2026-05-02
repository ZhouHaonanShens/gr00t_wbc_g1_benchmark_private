#!/usr/bin/env bash
set -euo pipefail
# targets are relative to repo root (script cd's there first); do not move this script without updating paths
cd "$(git rev-parse --show-toplevel)"
mkdir -p .envs
# Idempotency across directory-vs-symlink transitions: ln -sfn does NOT replace a real directory
# (it creates a nested link inside). Detect and remove pre-existing real directories at each alias.
for alias in main wbc openpi; do
  if [ -e ".envs/$alias" ] && [ ! -L ".envs/$alias" ]; then rm -rf ".envs/$alias"; fi
done
ln -sfn ../submodules/Isaac-GR00T/.venv .envs/main
ln -sfn ../submodules/Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv .envs/wbc
ln -sfn ../submodules/openpi/.venv .envs/openpi
echo "[setup_envs] linked: $(ls -l .envs/)"
