#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.stage3_baseline_train_gate import DEFAULT_FORMAL_OUTPUT_DIR_REL
from work.recap.stage3_baseline_train_gate import DEFAULT_SINGLE_GPU_SMOKE_VERDICT_REL
from work.recap.stage3_baseline_train_gate import LIVE_LAUNCH_FAMILY
from work.recap.stage3_baseline_train_gate import run_stage3_baseline_train_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="30b_stage3_baseline_train_gate.py",
        description=(
            "Run the Wave-3 single baseline-train gate for iter_002: perform RTX PRO 6000 Blackwell Max-Q preflight, "
            "execute the only allowed meaningful baseline-train attempt, then persist the gate results back into the stage3 iteration manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--iteration-manifest",
        type=Path,
        default=None,
        help="Optional override for the frozen stage3 iteration manifest path.",
    )
    _ = parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Only refresh the stage3 hardware/delegate preflight authority and manifest summary; "
            "do not start meaningful baseline-train and do not consume the single attempt budget."
        ),
    )
    _ = parser.add_argument(
        "--launch-family",
        type=str,
        default=LIVE_LAUNCH_FAMILY,
        help="Live baseline authority family. Task 6 formal path must use single_gpu_v1.",
    )
    _ = parser.add_argument(
        "--single-gpu-smoke-verdict",
        type=Path,
        default=DEFAULT_SINGLE_GPU_SMOKE_VERDICT_REL,
        help="Path to the stage3_single_gpu_smoke_verdict_v1 JSON used as live launch authority.",
    )
    _ = parser.add_argument(
        "--formal-output-dir",
        type=Path,
        default=DEFAULT_FORMAL_OUTPUT_DIR_REL,
        help="Formal single-GPU output namespace for the baseline finetune run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_stage3_baseline_train_gate(
        repo_root=REPO_ROOT,
        manifest_path=args.iteration_manifest,
        preflight_only=bool(args.preflight_only),
        launch_family=str(args.launch_family),
        single_gpu_smoke_verdict_path=args.single_gpu_smoke_verdict,
        formal_output_dir=args.formal_output_dir,
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return int(payload.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
