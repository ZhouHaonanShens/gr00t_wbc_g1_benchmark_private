from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m work.recap.r7_2_uplift_probe.lora_train_worker")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-step-count", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
