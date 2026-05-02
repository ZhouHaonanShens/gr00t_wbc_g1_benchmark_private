#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/iter9_canonical_real_training_common.sh"

iter9_launch_canonical_lane "X" "3" "recap_variant_shuffle_diag" "$@"
