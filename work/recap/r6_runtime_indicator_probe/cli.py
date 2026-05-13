from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from work.recap.r6_runtime_indicator_probe.contract import CellProbeReport, R6Error
from work.recap.r6_runtime_indicator_probe.reports.cell_probe_report import render_cell_report
from work.recap.r6_runtime_indicator_probe.reports.runtime_probe_report import render_runtime_probe_report
from work.recap.r6_runtime_indicator_probe.reports.summary_report import render_summary_report
from work.recap.r6_runtime_indicator_probe.synthesis import compose_final
from work.recap.r6_runtime_indicator_probe.wiring_graph import CELL_IDS, trace_wiring

_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_FORCED_ONLY_CELL = "A.2"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_json_safe(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_cell(root: Path, report: CellProbeReport) -> None:
    cell_dir = root / report.cell_id
    _write_json(cell_dir / "wiring_graph.json", report.static)
    if report.runtime is not None:
        _write_json(cell_dir / "runtime_trace.json", report.runtime)
    (cell_dir / "cell_probe_report.md").write_text(render_cell_report(report), encoding="utf-8")


def _write_forced_runtime(root: Path, runtime: Any, counterfactual: Any, budget: Any, token: str, negative_runtime: Any = None) -> None:
    if counterfactual is None:
        raise R6Error("--forced probe requires counterfactual output")
    cell_dir = root / runtime.cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "prompt_at_tokenizer_step0.txt").write_text(runtime.prompt_text_at_tokenizer[:500], encoding="utf-8")
    _write_json(cell_dir / "runtime_trace.json", runtime)
    _write_json(cell_dir / "counterfactual.json", counterfactual)
    token_sha = hashlib.sha256(str(token).encode("utf-8")).hexdigest()
    report = render_runtime_probe_report(runtime=runtime, counterfactual=counterfactual, negative_runtime=negative_runtime, budget=budget, leader_token_sha256=token_sha)
    (cell_dir / "FIX_R2_A1_LOAD_06_R6_RUNTIME_PROBE_REPORT.md").write_text(report, encoding="utf-8")


def _selected_cells(args: argparse.Namespace) -> tuple[str, ...]:
    if bool(getattr(args, "all", False)):
        return CELL_IDS
    cell = str(getattr(args, "cell", "")).strip().upper()
    if cell not in CELL_IDS:
        raise R6Error(f"unsupported R6 cell: {cell!r}; expected A.2|A.3|A.4|A.5")
    return (cell,)


def run_trace(args: argparse.Namespace) -> int:
    root = Path(args.output_root)
    reports = []
    for cell in _selected_cells(args):
        static = trace_wiring(cell)
        report = CellProbeReport(cell, static, None, compose_final(static, None))
        _write_cell(root, report)
        reports.append(report)
    if getattr(args, "all", False):
        (root / "FIX_R2_A1_LOAD_05_R6_WIRING_REPORT.md").write_text(render_summary_report(reports), encoding="utf-8")
        _write_json(root / "r6_wiring_matrix.json", {"schema_version": "r6_wiring_matrix_v1", "cells": reports})
    return 0


def _validated_probe_args(args: argparse.Namespace) -> tuple[str, str, bool]:
    cell = str(args.cell).strip().upper()
    forced = bool(getattr(args, "forced", False))
    if forced and cell != _FORCED_ONLY_CELL:
        raise R6Error("--forced probe accepts only cell A.2")
    if cell not in CELL_IDS:
        raise R6Error(f"unsupported R6.1 cell: {cell!r}")
    token = str(args.leader_approval_token or "")
    if not _TOKEN_RE.fullmatch(token):
        raise R6Error("--leader-approval-token must be a 64-character SHA-256 hex string")
    if forced and int(args.gpu) != 1:
        raise R6Error("--forced probe is locked to GPU 1")
    if not forced and int(args.gpu) not in {1, 2}:
        raise R6Error("--gpu accepts only 1 or 2; GPU 0/3 are rejected")
    return cell, token, forced


def run_probe(args: argparse.Namespace) -> int:
    cell, token, forced = _validated_probe_args(args)
    from work.recap.r6_runtime_indicator_probe.runtime_probe import ProbeBudget, get_last_negative_trace, run_runtime_probe

    static = trace_wiring(cell)
    budget = ProbeBudget(gpu_id=int(args.gpu))
    runtime, counterfactual = run_runtime_probe(cell, budget, token, forced=forced, counterfactual=not bool(args.no_counterfactual))
    report = CellProbeReport(cell, static, runtime, compose_final(static, runtime, counterfactual))
    if forced:
        _write_forced_runtime(Path(args.output_root), runtime, counterfactual, budget, token, get_last_negative_trace())
    else:
        _write_cell(Path(args.output_root), report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m work.recap.r6_runtime_indicator_probe")
    sub = parser.add_subparsers(dest="command", required=True)
    trace = sub.add_parser("trace", help="Run R6.0 static AST/import wiring trace.")
    group = trace.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell")
    group.add_argument("--all", action="store_true")
    trace.add_argument("--output-root", required=True)
    trace.set_defaults(func=run_trace)
    probe = sub.add_parser("probe", help="Run approved bounded R6.1 runtime probe for one cell.")
    probe.add_argument("--cell", required=True)
    probe.add_argument("--leader-approval-token", required=True)
    probe.add_argument("--gpu", type=int, default=1)
    probe.add_argument("--output-root", required=True)
    probe.add_argument("--forced", action="store_true")
    probe.add_argument("--no-counterfactual", action="store_true")
    probe.set_defaults(func=run_probe)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if "torch" in sys.modules:
            raise R6Error("R6 CLI must not import torch in main process")
        return int(args.func(args))
    except R6Error as exc:
        print(f"R6 failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
