#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.dataset_aggregation import (  # noqa: E402
    LOOP_MANIFEST_NAME,
    build_loop_manifest,
    write_json,
)
from work.openpi.pipelines.recap.iteration import (  # noqa: E402
    IterationConfig,
    run_iteration,
)
from work.openpi.pipelines.recap.iteration_workflow import (  # noqa: E402
    DEFAULT_REPAIRED_MATRIX_SUMMARY_PATH,
    DEFAULT_TRACKED_SUMMARY_PATH,
)
from work.openpi.sources.libero_official.validate import (  # noqa: E402
    DEFAULT_DATASET_DIR,
)


@dataclass(frozen=True)
class LoopConfig:
    iterations: int
    iter_prefix: str
    policy_checkpoint: Path
    critic_checkpoint: Path | None
    indicator_mode: str
    task_suite_name: str
    task_ids: str
    episodes_per_iteration: int
    output_dir: Path
    demo_dir: Path
    correction_dir: Path | None


class LoopWorkflow:
    def __init__(self, config: LoopConfig) -> None:
        self.config: LoopConfig = config

    def run(self) -> dict[str, object]:
        if self.config.iterations <= 0:
            raise ValueError(f"--iterations must be > 0, got {self.config.iterations}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        manifests: list[dict[str, object]] = []
        for index in range(self.config.iterations):
            iter_id = f"{self.config.iter_prefix}{index}"
            iteration_output_dir = self.config.output_dir / iter_id
            manifests.append(
                run_iteration(
                    IterationConfig(
                        iter_id=iter_id,
                        seed_policy_checkpoint=self.config.policy_checkpoint,
                        critic_checkpoint=self.config.critic_checkpoint,
                        indicator_mode=self.config.indicator_mode,
                        task_suite_name=self.config.task_suite_name,
                        task_ids=self.config.task_ids,
                        episodes=self.config.episodes_per_iteration,
                        output_dir=iteration_output_dir,
                        demo_dir=self.config.demo_dir,
                        correction_dir=self.config.correction_dir,
                        critic_config=None,
                        repaired_matrix_summary_path=DEFAULT_REPAIRED_MATRIX_SUMMARY_PATH,
                        tracked_summary_path=DEFAULT_TRACKED_SUMMARY_PATH,
                    )
                )
            )
        loop_manifest = build_loop_manifest(
            output_dir=self.config.output_dir,
            iteration_manifests=manifests,
        )
        _ = write_json(self.config.output_dir / LOOP_MANIFEST_NAME, loop_manifest)
        return loop_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libero_recap_loop.py",
        description="Run a task-9 offline iteration loop scaffold over collect+merge stages.",
    )
    _ = parser.add_argument("--iterations", required=True, type=int)
    _ = parser.add_argument("--iter-prefix", default="iter")
    _ = parser.add_argument("--policy-checkpoint", required=True)
    _ = parser.add_argument("--critic-checkpoint", default=None)
    _ = parser.add_argument(
        "--indicator-mode",
        required=True,
        choices=("positive", "negative", "omit", "cfg"),
    )
    _ = parser.add_argument("--task-suite-name", required=True)
    _ = parser.add_argument("--task-ids", required=True)
    _ = parser.add_argument("--episodes-per-iteration", required=True, type=int)
    _ = parser.add_argument("--output-dir", required=True)
    _ = parser.add_argument("--demo-dir", default=str(DEFAULT_DATASET_DIR))
    _ = parser.add_argument("--correction-dir", default=None)
    return parser


def _build_config(args: argparse.Namespace) -> LoopConfig:
    return LoopConfig(
        iterations=int(cast(int, args.iterations)),
        iter_prefix=str(cast(object, args.iter_prefix)).strip() or "iter",
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
        task_suite_name=str(cast(object, args.task_suite_name)).strip(),
        task_ids=str(cast(object, args.task_ids)).strip(),
        episodes_per_iteration=int(cast(int, args.episodes_per_iteration)),
        output_dir=Path(cast(str, args.output_dir)).expanduser().resolve(),
        demo_dir=Path(cast(str, args.demo_dir)).expanduser().resolve(),
        correction_dir=(
            Path(cast(str, args.correction_dir)).expanduser().resolve()
            if isinstance(args.correction_dir, str) and str(args.correction_dir).strip()
            else None
        ),
    )


def run_loop(config: LoopConfig) -> dict[str, object]:
    return LoopWorkflow(config).run()


def main(argv: list[str] | None = None) -> int:
    _ = run_loop(_build_config(build_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
