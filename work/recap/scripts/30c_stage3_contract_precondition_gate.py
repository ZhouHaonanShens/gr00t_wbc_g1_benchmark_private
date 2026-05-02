#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.stage3_contract_precondition_gate import (
    run_stage3_contract_precondition_gate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="30c_stage3_contract_precondition_gate.py",
        description=(
            "Inspect the manifest-bound checkpoint weight map, auto-select the prelim eval surface, "
            "and atomically persist the stage3 contract-precondition gate outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--iteration-manifest",
        type=Path,
        default=None,
        help="Optional override for the iter_002 stage3 iteration manifest path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_stage3_contract_precondition_gate(
        repo_root=REPO_ROOT,
        manifest_path=args.iteration_manifest,
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return int(payload.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
