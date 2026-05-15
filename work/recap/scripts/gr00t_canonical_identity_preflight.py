#!/usr/bin/env python3
"""CLI for GR00T canonical identity preflight.

This script is Phase-1 only: it does not train and does not load model weights.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.identity.gr00t_canonical_identity_contract import (  # noqa: E402
    DEFAULT_TASK_PROMPT,
    PreflightMode,
    build_preflight_report,
    launcher_inventory_rows,
    read_json,
    repo_rel,
    resolve_path,
    write_preflight_outputs,
)


def _write_inventory(path: Path) -> None:
    rows = launcher_inventory_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "kind", "purpose", "phase1_requirement", "exists"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GR00T canonical identity preflight before eval/training/server entrypoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Candidate checkpoint/surface directory.")
    parser.add_argument("--canonical", required=True, help="Canonical identity directory with canonical_identity_manifest.json.")
    parser.add_argument("--mode", choices=[m.value for m in PreflightMode], default=PreflightMode.STRICT_PROMOTION.value)
    parser.add_argument("--entrypoint", default="manual_preflight")
    parser.add_argument("--entrypoint-kind", default="diagnostic")
    parser.add_argument("--prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--matrix-id", default="", help="Required for SURFACE_CAUSALITY_DIAGNOSTIC mode.")
    parser.add_argument("--diagnostic-manifest-json", default="", help="Required for SURFACE_CAUSALITY_DIAGNOSTIC mode.")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--launcher-inventory-csv", default="", help="Optional launcher inventory output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = PreflightMode(str(args.mode))
    diagnostic_manifest: dict[str, Any] | None = None
    if str(args.diagnostic_manifest_json).strip():
        diagnostic_manifest = read_json(resolve_path(str(args.diagnostic_manifest_json)))
    report = build_preflight_report(
        checkpoint=resolve_path(str(args.checkpoint)),
        canonical=resolve_path(str(args.canonical)),
        mode=mode,
        entrypoint=str(args.entrypoint),
        entrypoint_kind=str(args.entrypoint_kind),
        prompt=str(args.prompt),
        matrix_id=str(args.matrix_id).strip() or None,
        diagnostic_manifest=diagnostic_manifest,
    )
    report_json = resolve_path(str(args.report_json))
    report_md = resolve_path(str(args.report_md))
    write_preflight_outputs(report, report_json=report_json, report_md=report_md)
    if str(args.launcher_inventory_csv).strip():
        _write_inventory(resolve_path(str(args.launcher_inventory_csv)))
    print(json.dumps({
        "status": report["verdict"],
        "reason_code": report["reason_code"],
        "report_json": repo_rel(report_json),
        "report_md": repo_rel(report_md),
    }, sort_keys=True))
    return 0 if report["verdict"] in {"PASS", "DIAGNOSTIC_ONLY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
