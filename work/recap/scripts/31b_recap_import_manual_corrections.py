#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.formal_branch_resolution import (  # noqa: E402
    BLOCKED_EXIT_CODE,
    FormalBranchResolutionBlocked,
    import_external_manual_correction_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="31b_recap_import_manual_corrections.py",
        description=(
            "Import the external manual correction M1 bundle into recap_stage3_iter_001_train."
        ),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Optional override for the external manual correction bundle directory.",
    )
    parser.add_argument(
        "--iteration-manifest",
        type=Path,
        default=None,
        help="Optional override for the frozen stage3 iteration manifest path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = import_external_manual_correction_bundle(
            REPO_ROOT,
            bundle_dir=args.bundle_dir,
            manifest_path=args.iteration_manifest,
        )
    except FormalBranchResolutionBlocked as exc:
        print(json.dumps(exc.to_machine_payload(), ensure_ascii=True, sort_keys=True))
        return BLOCKED_EXIT_CODE
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
