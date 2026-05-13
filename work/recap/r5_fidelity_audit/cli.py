from __future__ import annotations

import argparse, json, sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from work.recap.r5_fidelity_audit.contract import FIDELITY_QUESTIONS, FULL_REPORT_FILENAME

QUESTION_MANIFEST_FILENAME = "fidelity_question_manifest.json"
QUESTION_REPORT_FILENAME = "fidelity_question_report.md"
MATRIX_REPORT_FILENAME = "gr00t_recap_fidelity_matrix.json"
from work.recap.r5_fidelity_audit.reports.question_report import render_question_report
from work.recap.r5_fidelity_audit.reports.summary_report import matrix_data, render_summary

VALID_QIDS = tuple(f"Q{i}" for i in range(1, 10))

class CliError(RuntimeError): pass

def _json_safe(v: Any) -> Any:
    if is_dataclass(v) and not isinstance(v, type): return _json_safe(asdict(v))
    if isinstance(v, Path): return str(v)
    if isinstance(v, (tuple, list)): return [_json_safe(x) for x in v]
    if isinstance(v, Mapping): return {str(k): _json_safe(x) for k, x in v.items()}
    if hasattr(v, "__dict__"): return {str(k): _json_safe(x) for k, x in vars(v).items() if not str(k).startswith("_")}
    return v

def _get(obj: Any, key: str, default: Any = "") -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)

def _qid(obj: Any) -> str:
    q = _get(obj, "qid") or _get(obj, "question_id") or obj
    return str(q).upper() if str(q).upper() in VALID_QIDS else str(q)

def _result_qid(result: Any) -> str:
    q = _get(result, "qid") or _get(result, "question_id") or _get(result, "question")
    return _qid(q)

def _select(question_id: str | None, all_questions: bool) -> tuple[Any, ...]:
    questions = {_qid(q): q for q in FIDELITY_QUESTIONS}
    if all_questions: return tuple(questions[q] for q in VALID_QIDS)
    qid = str(question_id or "").strip().upper()
    if qid not in VALID_QIDS or qid not in questions: raise CliError(f"unsupported fidelity question: {question_id}")
    return (questions[qid],)

def _write_question(result: Any, root: Path) -> dict[str, Any]:
    qid = _result_qid(result); qdir = root / qid; qdir.mkdir(parents=True, exist_ok=True)
    manifest, report = qdir / QUESTION_MANIFEST_FILENAME, qdir / QUESTION_REPORT_FILENAME
    data = _json_safe(result); data = dict(data) if isinstance(data, Mapping) else {"result": data}
    data.setdefault("qid", qid); data["manifest_path"] = str(manifest); data["report_path"] = str(report)
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.write_text(render_question_report(result), encoding="utf-8")
    return data

def run_audit(args: argparse.Namespace) -> int:
    from work.recap.r5_fidelity_audit.analyzers import audit_question
    from work.recap.r5_fidelity_audit.verdicts import overall_fidelity_label
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    results = tuple(audit_question(q) for q in _select(args.question, bool(args.all)))
    for result in results: _write_question(result, root)
    if args.all:
        label = overall_fidelity_label(results)
        (root / MATRIX_REPORT_FILENAME).write_text(json.dumps(matrix_data(results, label), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (root / FULL_REPORT_FILENAME).write_text(render_summary(results, label), encoding="utf-8")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m work.recap.r5_fidelity_audit"); sub = p.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Run the static GR00T RECAP fidelity audit."); group = audit.add_mutually_exclusive_group(required=True)
    group.add_argument("--question"); group.add_argument("--all", action="store_true"); audit.add_argument("--output-root", required=True); audit.set_defaults(func=run_audit)
    return p

def main(argv: list[str] | None = None) -> int:
    parser = build_parser(); args = parser.parse_args(argv)
    try: return int(args.func(args))
    except CliError as exc: print(f"R5 audit failed: {exc}", file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())
