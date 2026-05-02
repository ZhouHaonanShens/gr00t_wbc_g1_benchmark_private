#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

exec bash "${REPO_ROOT}/work/recap/scripts/38_recap_smoke_end_to_end.sh" "$@"
