#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/iter9_canonical_real_training_common.sh"

iter9_launch_canonical_lane "B" "1" "control_no_recap_shuffled_adversarial_relabel" "$@"
