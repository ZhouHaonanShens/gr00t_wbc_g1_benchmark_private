#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.critic_vlm.train import run_libero_recap_critic_training  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Task-5 LIBERO RECAP critic on the official/native 8D source.",
    )
    _ = parser.add_argument("--config", required=True)
    _ = parser.add_argument("--dataset-dir", required=True)
    _ = parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _ = run_libero_recap_critic_training(
        repo_root=REPO_ROOT,
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
