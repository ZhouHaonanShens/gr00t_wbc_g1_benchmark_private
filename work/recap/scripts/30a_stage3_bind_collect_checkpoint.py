#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.stage3_collect_checkpoint_binding import bind_collect_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="30a_stage3_bind_collect_checkpoint.py",
        description=(
            "Bind the iter_002 stage3 collect checkpoint on the manifest-v3 surface, "
            "emit checkpoint/run-manifest gates, and refresh the contract precondition gate."
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
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit manual-pinned checkpoint candidate.",
    )
    _ = parser.add_argument(
        "--checkpoint-success-rate",
        type=float,
        default=None,
        help="Optional numeric preliminary success rate for the explicit manual checkpoint.",
    )
    _ = parser.add_argument(
        "--checkpoint-success-count",
        type=int,
        default=None,
        help="Optional preliminary success count for the explicit manual checkpoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = bind_collect_checkpoint(
        repo_root=REPO_ROOT,
        manifest_path=args.iteration_manifest,
        manual_checkpoint_path=args.checkpoint_path,
        manual_checkpoint_success_rate=args.checkpoint_success_rate,
        manual_checkpoint_success_count=args.checkpoint_success_count,
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
