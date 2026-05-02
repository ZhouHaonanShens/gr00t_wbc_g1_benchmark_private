#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from .iteration_workflow import (  # noqa: E402
    CRITIC_RETRAIN_ROUTE_ID,
    DEFAULT_REPAIRED_MATRIX_SUMMARY_PATH,
    DEFAULT_TRACKED_SUMMARY_PATH,
    ITERATION_EVAL_ROUTE_ID,
    POLICY_RETRAIN_ROUTE_ID,
    EvalVariantBundle,
    EvalVariantSpec,
    IterationConfig,
    IterationWorkflow,
    build_train_runtime_dir,
    libero_recap_train,
    libero_rollout_eval_v21,
    run_iteration,
    train_recap_critic,
)
from work.openpi.sources.libero_official.validate import DEFAULT_DATASET_DIR  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libero_recap_iteration.py",
        description=(
            "Run one repaired RECAP iteration smoke over collect -> critic retrain -> policy retrain -> eval. "
            "If --critic-config is omitted, the script stays in compatibility collect+merge mode."
        ),
    )
    _ = parser.add_argument("--iter-id", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    _ = checkpoint_group.add_argument("--seed-policy-checkpoint")
    _ = checkpoint_group.add_argument("--policy-checkpoint")
    _ = parser.add_argument("--critic-checkpoint", default=None)
    _ = parser.add_argument(
        "--indicator-mode",
        default="positive",
        choices=("positive", "negative", "omit", "cfg"),
    )
    _ = parser.add_argument("--critic-config", default=None)
    _ = parser.add_argument("--task-suite-name", required=True)
    _ = parser.add_argument("--task-ids", required=True)
    _ = parser.add_argument("--episodes", required=True, type=int)
    _ = parser.add_argument("--output-dir", required=True)
    _ = parser.add_argument("--demo-dir", default=str(DEFAULT_DATASET_DIR))
    _ = parser.add_argument("--correction-dir", default=None)
    _ = parser.add_argument(
        "--repaired-matrix-summary",
        default=str(DEFAULT_REPAIRED_MATRIX_SUMMARY_PATH),
    )
    _ = parser.add_argument(
        "--tracked-summary-path",
        default=str(DEFAULT_TRACKED_SUMMARY_PATH),
    )
    return parser


def _build_config(args: argparse.Namespace) -> IterationConfig:
    seed_policy_checkpoint = str(
        cast(object, getattr(args, "seed_policy_checkpoint", ""))
        or cast(object, getattr(args, "policy_checkpoint", ""))
        or ""
    ).strip()
    if not seed_policy_checkpoint:
        raise ValueError("--seed-policy-checkpoint or --policy-checkpoint is required")
    demo_dir = str(cast(object, getattr(args, "demo_dir", ""))).strip()
    return IterationConfig(
        iter_id=str(cast(object, args.iter_id)).strip(),
        seed_policy_checkpoint=Path(seed_policy_checkpoint).expanduser().resolve(),
        critic_checkpoint=(
            Path(cast(str, args.critic_checkpoint)).expanduser().resolve()
            if isinstance(args.critic_checkpoint, str)
            and str(args.critic_checkpoint).strip()
            else None
        ),
        indicator_mode=str(cast(object, args.indicator_mode)).strip(),
        task_suite_name=str(cast(object, args.task_suite_name)).strip(),
        task_ids=str(cast(object, args.task_ids)).strip(),
        episodes=int(cast(int, args.episodes)),
        output_dir=Path(cast(str, args.output_dir)).expanduser().resolve(),
        demo_dir=Path(demo_dir).expanduser().resolve(),
        correction_dir=(
            Path(cast(str, args.correction_dir)).expanduser().resolve()
            if isinstance(args.correction_dir, str) and str(args.correction_dir).strip()
            else None
        ),
        critic_config=(
            Path(cast(str, args.critic_config)).expanduser().resolve()
            if isinstance(args.critic_config, str) and str(args.critic_config).strip()
            else None
        ),
        repaired_matrix_summary_path=Path(cast(str, args.repaired_matrix_summary))
        .expanduser()
        .resolve(),
        tracked_summary_path=Path(cast(str, args.tracked_summary_path))
        .expanduser()
        .resolve(),
    )


def main(argv: list[str] | None = None) -> int:
    _ = run_iteration(_build_config(build_parser().parse_args(argv)))
    return 0


__all__ = [
    "CRITIC_RETRAIN_ROUTE_ID",
    "EvalVariantBundle",
    "EvalVariantSpec",
    "ITERATION_EVAL_ROUTE_ID",
    "IterationConfig",
    "IterationWorkflow",
    "POLICY_RETRAIN_ROUTE_ID",
    "build_parser",
    "build_train_runtime_dir",
    "libero_recap_train",
    "libero_rollout_eval_v21",
    "main",
    "run_iteration",
    "train_recap_critic",
]


if __name__ == "__main__":
    raise SystemExit(main())
