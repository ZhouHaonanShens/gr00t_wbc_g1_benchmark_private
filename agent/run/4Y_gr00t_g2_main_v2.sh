#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
exec python3 work/recap/scripts/4Y_gr00t_g2_main_v2.py "$@"
