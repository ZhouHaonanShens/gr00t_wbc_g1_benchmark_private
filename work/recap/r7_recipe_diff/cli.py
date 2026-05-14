from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from work.recap.r7_recipe_diff.analyzer import OPENPI_REPORT_PATH, analyze_all
from work.recap.r7_recipe_diff.contract import R7DiffError
from work.recap.r7_recipe_diff.reports import (
    recipe_diff_payload,
    render_recipe_diff_markdown,
    write_recipe_diff_json,
)

TRAINING_RECIPE_DIFF_JSON = "training_recipe_diff.json"
RECIPE_DIFF_REPORT_MD = "R7_RECIPE_DIFF_REPORT.md"


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_audit(args: argparse.Namespace) -> int:
    generated_at_utc = _timestamp_utc()
    output_root = Path(args.output_root)
    run_dir = output_root / f"{generated_at_utc}_run"
    deltas = analyze_all(str(args.base_cell))
    payload = recipe_diff_payload(
        base_cell=str(args.base_cell),
        generated_at_utc=generated_at_utc,
        source_openpi_report=str(OPENPI_REPORT_PATH),
        deltas=deltas,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_recipe_diff_json(run_dir / TRAINING_RECIPE_DIFF_JSON, payload)
    (run_dir / RECIPE_DIFF_REPORT_MD).write_text(
        render_recipe_diff_markdown(payload),
        encoding="utf-8",
    )
    print(run_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m work.recap.r7_recipe_diff")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Run R7.0 training recipe diff audit.")
    audit.add_argument("--base-cell", required=True)
    audit.add_argument("--output-root", required=True)
    audit.set_defaults(func=run_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except R7DiffError as exc:
        print(f"R7 recipe diff failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
