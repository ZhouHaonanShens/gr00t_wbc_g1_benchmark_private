#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.dataset_aggregation import parse_task_ids_csv  # noqa: E402
from work.openpi.sources.libero_official.validate import (  # noqa: E402
    BLOCKED_EXIT_CODE,
    DEFAULT_DATASET_DIR,
)
from .collect_workflow import (  # noqa: E402
    CollectConfig,
    CollectWorkflow,
    libero_rollout_eval_v21,
    run_collection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libero_recap_collect.py",
        description=(
            "Collect autonomous LIBERO recap trials into the task-9 OpenPI collection schema."
        ),
    )
    _ = parser.add_argument("--policy-checkpoint", required=True)
    _ = parser.add_argument("--critic-checkpoint", default=None)
    _ = parser.add_argument(
        "--indicator-mode",
        required=True,
        choices=("positive", "negative", "omit", "cfg"),
    )
    _ = parser.add_argument("--task-suite-name", required=True)
    _ = parser.add_argument("--task-ids", required=True)
    _ = parser.add_argument("--episodes", required=True, type=int)
    _ = parser.add_argument("--output-dir", required=True)
    _ = parser.add_argument("--demo-dir", default=str(DEFAULT_DATASET_DIR))
    return parser


def _build_config(args: argparse.Namespace) -> CollectConfig:
    task_suite_name = str(cast(object, args.task_suite_name)).strip()
    if task_suite_name != "libero_spatial":
        raise ValueError(
            f"task-9 collect requires --task-suite-name libero_spatial, got {task_suite_name!r}"
        )
    episodes = int(cast(int, args.episodes))
    if episodes <= 0:
        raise ValueError(f"--episodes must be > 0, got {episodes}")
    return CollectConfig(
        policy_checkpoint=Path(cast(str, args.policy_checkpoint))
        .expanduser()
        .resolve(),
        critic_checkpoint=(
            Path(cast(str, args.critic_checkpoint)).expanduser().resolve()
            if isinstance(args.critic_checkpoint, str)
            and str(args.critic_checkpoint).strip()
            else None
        ),
        indicator_mode=str(cast(object, args.indicator_mode)).strip(),
        task_suite_name=task_suite_name,
        task_ids=parse_task_ids_csv(cast(str, args.task_ids)),
        episodes=episodes,
        output_dir=Path(cast(str, args.output_dir)).expanduser().resolve(),
        demo_dir=Path(cast(str, args.demo_dir)).expanduser().resolve(),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _ = run_collection(_build_config(args))
    except RuntimeError as exc:
        if str(exc).startswith("canonical demo source blocked:"):
            return BLOCKED_EXIT_CODE
        raise
    return 0


__all__ = [
    "CollectConfig",
    "CollectWorkflow",
    "build_parser",
    "libero_rollout_eval_v21",
    "main",
    "run_collection",
]


if __name__ == "__main__":
    raise SystemExit(main())
