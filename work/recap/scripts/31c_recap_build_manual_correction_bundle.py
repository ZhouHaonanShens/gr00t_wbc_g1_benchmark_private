#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.manual_correction_bundle import (  # noqa: E402
    build_manual_correction_bundle,
    scaffold_manual_correction_spec,
    validate_manual_correction_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="31c_recap_build_manual_correction_bundle.py",
        description=(
            "Scaffold, validate, or build a strict stage3 manual correction bundle from real M1 datasets."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scaffold = subparsers.add_parser("scaffold")
    scaffold.add_argument("--spec", type=Path, required=True)
    scaffold.add_argument("--iteration-manifest", type=Path, default=None)
    scaffold.add_argument("--nominal-dataset-dir", type=Path, default=None)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--spec", type=Path, required=True)
    validate.add_argument("--iteration-manifest", type=Path, default=None)

    build = subparsers.add_parser("build")
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--bundle-dir", type=Path, required=True)
    build.add_argument("--iteration-manifest", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scaffold":
        payload = scaffold_manual_correction_spec(
            REPO_ROOT,
            spec_path=args.spec,
            manifest_path=args.iteration_manifest,
            nominal_dataset_dir=args.nominal_dataset_dir,
        )
    elif args.command == "validate":
        payload = validate_manual_correction_spec(
            REPO_ROOT,
            spec_path=args.spec,
            manifest_path=args.iteration_manifest,
        )
    else:
        payload = build_manual_correction_bundle(
            REPO_ROOT,
            spec_path=args.spec,
            bundle_dir=args.bundle_dir,
            manifest_path=args.iteration_manifest,
        )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
