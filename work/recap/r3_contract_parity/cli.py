from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from work.recap.r2_authentic_eval import exclusion
from work.recap.r3_contract_parity import collectors
from work.recap.r3_contract_parity.contract import PARITY_AXES, R3AuditError, _MISSING, ParityCellReport
from work.recap.r3_contract_parity.diff import compare_all
from work.recap.r3_contract_parity.gates import cell_overall_verdict, collect_pattern_hits
from work.recap.r3_contract_parity.reports.cell_report import render_cell_report
from work.recap.r3_contract_parity.reports.summary_report import render_summary_report, summary_data


def _json_safe(value: Any) -> Any:
    if value is _MISSING or type(value) is object:
        return "__MISSING__"
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _cell_dict(report: ParityCellReport) -> dict[str, Any]:
    return _json_safe(asdict(report))


def _write_cell_outputs(report: ParityCellReport, output_root: Path) -> ParityCellReport:
    cell_dir = output_root / report.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cell_dir / "cell_parity_manifest.json"
    report_path = cell_dir / "cell_parity_report.md"
    report = ParityCellReport(report.cell_id, report.ckpt_abs_path, report.verdict, report.axes, report.pattern_hits, str(manifest_path), str(report_path))
    manifest_path.write_text(json.dumps(_cell_dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_cell_report(report), encoding="utf-8")
    return report


def audit_cell(cell_id: str, output_root: Path | str | None = None, r2_run_root: Path | str | None = None) -> ParityCellReport:
    ckpt = collectors.resolve_cell_ckpt(cell_id)
    train_snapshot = collectors.load_train_snapshot(ckpt)
    eval_snapshot = collectors.load_eval_snapshot(cell_id, Path(r2_run_root) if r2_run_root else None)
    axes = compare_all(train_snapshot, eval_snapshot, PARITY_AXES)
    report = ParityCellReport(cell_id, str(ckpt), cell_overall_verdict(axes), axes, collect_pattern_hits(axes))
    return _write_cell_outputs(report, Path(output_root)) if output_root is not None else report


def _write_summary(reports: tuple[ParityCellReport, ...], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "r3_parity_summary.json").write_text(
        json.dumps(_json_safe(summary_data(reports)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "r3_parity_summary.md").write_text(render_summary_report(reports), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m work.recap.r3_contract_parity")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    group = audit.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell")
    group.add_argument("--all", action="store_true")
    audit.add_argument("--output-root", required=True)
    audit.add_argument("--r2-run-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cells = exclusion.EVIDENCE_GRADE_CELL_IDS if args.all else (args.cell,)
        reports = tuple(audit_cell(cell, args.output_root, args.r2_run_root) for cell in cells)
        if args.all:
            _write_summary(reports, Path(args.output_root))
    except R3AuditError as exc:
        print(f"R3 audit failed: {exc}", file=sys.stderr)
        return 2
    return 0
