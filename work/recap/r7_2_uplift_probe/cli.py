from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

from work.recap.r7_2_uplift_probe.contract import R7UpliftError, RecipePreset, TrialRequest, preset_to_recipe_flags

TRIAL_IDS = ("trial-1", "trial-2", "trial-3")
RECIPE_PRESETS = ("full_C1_C2_C5", "subset_C1_C5_no_dropout", "subset_C1_only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m work.recap.r7_2_uplift_probe")
    subcommands = parser.add_subparsers(dest="command", required=True)
    trial = subcommands.add_parser("trial", help="Run one R7.2 LoRA uplift trial.")
    trial.add_argument("--trial-id", required=True)
    trial.add_argument("--base-ckpt", required=True)
    trial.add_argument("--recipe-preset", required=True, choices=RECIPE_PRESETS)
    trial.add_argument("--output-root", required=True)
    trial.add_argument("--leader-approval-token", required=True)
    trial.add_argument("--gpu", type=int, required=True)
    trial.add_argument("--max-steps", type=int, default=2000)
    trial.add_argument("--probe-interval", type=int, default=200)
    trial.add_argument("--lora-rank", type=int, default=16)
    trial.add_argument("--lora-alpha", type=int, default=32)
    trial.add_argument("--lora-target-modules-top-k-layers", type=int, default=4)
    trial.add_argument("--budget-gpu-minutes", type=int, default=240)
    trial.set_defaults(func=run_trial_command)
    return parser


def run_trial_command(args: argparse.Namespace) -> int:
    request = request_from_args(args)
    from work.recap.r7_2_uplift_probe.trial_runner import run_trial

    report = run_trial(request)
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


def request_from_args(args: argparse.Namespace) -> TrialRequest:
    if args.trial_id not in TRIAL_IDS:
        raise R7UpliftError(f"trial_id must be one of {TRIAL_IDS}")
    base_ckpt = Path(str(args.base_ckpt))
    if not base_ckpt.is_dir():
        raise R7UpliftError(f"base_ckpt must be an existing directory: {base_ckpt}")
    token = str(args.leader_approval_token)
    if len(token) != 64 or any(char not in "0123456789abcdefABCDEF" for char in token):
        raise R7UpliftError("leader_approval_token must be 64-character hex")
    if int(args.gpu) not in {1, 2}:
        raise R7UpliftError("--gpu accepts only 1 or 2; GPU 0/3 are rejected")
    if args.trial_id == "trial-1" and int(args.gpu) != 1:
        raise R7UpliftError("trial-1 must use GPU 1")
    recipe_preset: RecipePreset = args.recipe_preset
    if recipe_preset not in RECIPE_PRESETS:
        raise R7UpliftError(f"recipe_preset must be one of {RECIPE_PRESETS}")
    output_root = Path(str(args.output_root))
    if output_root.exists():
        raise R7UpliftError(f"output_root must not exist: {output_root}")
    return TrialRequest(
        str(args.trial_id),
        str(base_ckpt.resolve()),
        preset_to_recipe_flags(recipe_preset),
        recipe_preset,
        str(output_root),
        token,
        int(args.gpu),
        int(args.max_steps),
        int(args.probe_interval),
        int(args.lora_rank),
        int(args.lora_alpha),
        int(args.lora_target_modules_top_k_layers),
        int(args.budget_gpu_minutes),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except R7UpliftError as exc:
        print(f"R7.2 failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
