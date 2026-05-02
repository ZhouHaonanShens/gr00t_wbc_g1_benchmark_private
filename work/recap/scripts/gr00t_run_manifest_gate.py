from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_DIR = Path("agent/artifacts/gr00t_run_manifest")

RUN_MANIFEST_JSON_NAME = "run_manifest.json"
RUN_MANIFEST_REPORT_JSON_NAME = "run_manifest_gate_report.json"
FAILURE_NOTE_MARKDOWN_NAME = "run_manifest_gate_failure_note.md"

REPORT_SCHEMA_VERSION = "gr00t_run_manifest_gate_report_v1"
REPORT_ARTIFACT_KIND = "gr00t_run_manifest_gate_report"
RUN_MANIFEST_GATE_NAME = "GR00TRunManifestGate"
OK_REASON_CODE = "ok"
INVALID_RUN_MANIFEST_REASON_CODE = "invalid_run_manifest"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.run_manifest import build_run_manifest_from_sources
from work.recap.run_manifest import validate_run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_run_manifest_gate.py",
        description=(
            "Build or validate a minimal-core fail-closed run manifest for the RECAP "
            "mainline and emit a machine-readable gate report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Validate an existing run manifest JSON instead of adapting live source payloads.",
    )
    parser.add_argument(
        "--state-conditioned-metadata",
        type=Path,
        default=None,
        help="Existing state-conditioned run metadata JSON carrying comparable_run_spec.",
    )
    parser.add_argument(
        "--finetune-summary",
        type=Path,
        default=None,
        help="Existing classic finetune summary JSON exposing selected_checkpoint_path.",
    )
    parser.add_argument(
        "--eval-summary",
        type=Path,
        default=None,
        help="Existing eval summary JSON exposing execution_surface_contract and server_provenance.",
    )
    parser.add_argument(
        "--server-provenance-json",
        type=Path,
        default=None,
        help="Optional standalone server provenance JSON if not nested in eval summary.",
    )
    parser.add_argument(
        "--controller-audit-json",
        type=Path,
        default=None,
        help="Optional controller audit JSON exposing controller_provenance / action contract.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="",
        help="Optional explicit branch/embodiment label override.",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="",
        help="Optional explicit git commit override. When omitted during adaptation, git rev-parse HEAD is used if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory that receives run_manifest.json, run_manifest_gate_report.json, "
            "and a failure note markdown when the gate blocks."
        ),
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)
    return path


def _validate_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output-dir must be a directory path: {resolved}")
    if not resolved.exists() and resolved.suffix:
        raise ValueError(f"output-dir must be a directory path: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"JSON path does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"JSON payload at {resolved} must be an object, got {type(payload).__name__}"
        )
    return dict(payload)


def _git_commit(repo_root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value if value else None


def _build_failure_note(report: Mapping[str, object]) -> str:
    issues = report.get("issues")
    rendered_issues = []
    if isinstance(issues, list):
        for item in issues:
            if isinstance(item, Mapping):
                rendered_issues.append(
                    "  - "
                    + f"`{item.get('code', 'unknown')}` @ `{item.get('field_path', '?')}`: "
                    + str(item.get("message", ""))
                )
            else:
                rendered_issues.append(f"  - `{item}`")
    if not rendered_issues:
        rendered_issues = ["  - `no issues captured`"]
    core = report.get("core")
    core_branch = None
    core_commit = None
    checkpoint_selected = None
    checkpoint_loaded = None
    if isinstance(core, Mapping):
        core_branch = core.get("branch")
        core_commit = core.get("commit")
        checkpoint_selected = core.get("checkpoint_selected")
        checkpoint_loaded = core.get("checkpoint_loaded")
    lines = [
        "# GR00T run manifest gate 失败说明",
        "",
        f"- formal_eligibility: `{report.get('formal_eligibility', 'BLOCK')}`",
        f"- reason_code: `{report.get('reason_code', INVALID_RUN_MANIFEST_REASON_CODE)}`",
        f"- branch: `{core_branch}`",
        f"- commit: `{core_commit}`",
        f"- checkpoint_selected: `{checkpoint_selected}`",
        f"- checkpoint_loaded: `{checkpoint_loaded}`",
        f"- manifest_path: `{report.get('manifest_path')}`",
        f"- report_path: `{report.get('artifact_path')}`",
        "- issues:",
        *rendered_issues,
        "",
        "该 run_manifest 未能通过最小核心 schema 与 checkpoint/provenance 绑定校验，因此按 fail-closed 规则不得作为 RECAP mainline 的正式 authority 产物。",
        "",
    ]
    return "\n".join(lines)


def _write_failure_note(path: Path, report: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_build_failure_note(report), encoding="utf-8")
    tmp.replace(path)
    return path


def _build_internal_error_report(
    *,
    output_dir: Path,
    manifest_path: Path,
    message: str,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "gate_name": RUN_MANIFEST_GATE_NAME,
        "status": "FAIL",
        "formal_eligibility": "BLOCK",
        "reason_code": INVALID_RUN_MANIFEST_REASON_CODE,
        "manifest_path": str(manifest_path),
        "artifact_path": str(output_dir / RUN_MANIFEST_REPORT_JSON_NAME),
        "core": {},
        "core_digest": None,
        "issues": [
            {
                "code": "internal_error",
                "field_path": "$",
                "message": str(message),
            }
        ],
        "checkpoint_binding": {},
        "failure_note_path": None,
    }


def _build_or_load_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest_json is not None:
        return _read_json(args.manifest_json)
    state_conditioned_metadata = (
        None
        if args.state_conditioned_metadata is None
        else _read_json(args.state_conditioned_metadata)
    )
    finetune_summary = (
        None if args.finetune_summary is None else _read_json(args.finetune_summary)
    )
    eval_summary = None if args.eval_summary is None else _read_json(args.eval_summary)
    server_provenance = (
        None
        if args.server_provenance_json is None
        else _read_json(args.server_provenance_json)
    )
    controller_audit = (
        None
        if args.controller_audit_json is None
        else _read_json(args.controller_audit_json)
    )
    if (
        state_conditioned_metadata is None
        and finetune_summary is None
        and eval_summary is None
        and server_provenance is None
        and controller_audit is None
    ):
        raise ValueError(
            "either --manifest-json or at least one adapter input JSON is required"
        )
    explicit_commit = str(args.commit).strip() or _git_commit(REPO_ROOT)
    explicit_branch = str(args.branch).strip() or None
    return build_run_manifest_from_sources(
        state_conditioned_metadata=state_conditioned_metadata,
        finetune_summary=finetune_summary,
        eval_summary=eval_summary,
        server_provenance=server_provenance,
        controller_audit=controller_audit,
        branch=explicit_branch,
        commit=explicit_commit,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _validate_output_dir(args.output_dir)
    manifest_path = output_dir / RUN_MANIFEST_JSON_NAME
    try:
        manifest_payload = _build_or_load_manifest(args)
        validation = validate_run_manifest(manifest_payload, repo_root=REPO_ROOT)
        normalized_manifest = dict(validation["normalized_manifest"])
        normalized_manifest["core_digest"] = validation["core_digest"]
        manifest_path = _write_json(manifest_path, normalized_manifest)
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_kind": REPORT_ARTIFACT_KIND,
            "gate_name": RUN_MANIFEST_GATE_NAME,
            "status": "PASS" if validation["formal_eligibility"] == "ALLOW" else "FAIL",
            "formal_eligibility": validation["formal_eligibility"],
            "reason_code": OK_REASON_CODE
            if validation["formal_eligibility"] == "ALLOW"
            else INVALID_RUN_MANIFEST_REASON_CODE,
            "manifest_path": str(manifest_path),
            "artifact_path": str(output_dir / RUN_MANIFEST_REPORT_JSON_NAME),
            "core": dict(normalized_manifest.get("core", {})),
            "core_digest": validation["core_digest"],
            "issues": list(validation["issues"]),
            "checkpoint_binding": dict(validation["checkpoint_binding"]),
            "failure_note_path": None,
        }
    except Exception as exc:
        report = _build_internal_error_report(
            output_dir=output_dir,
            manifest_path=manifest_path,
            message=_exception_message(exc),
        )

    report_path = _write_json(output_dir / RUN_MANIFEST_REPORT_JSON_NAME, report)
    report["artifact_path"] = str(report_path)
    if str(report.get("formal_eligibility")) == "BLOCK":
        failure_note_path = _write_failure_note(
            output_dir / FAILURE_NOTE_MARKDOWN_NAME,
            report,
        )
        report["failure_note_path"] = str(failure_note_path)
        report_path = _write_json(report_path, report)
        report["artifact_path"] = str(report_path)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if str(report.get("formal_eligibility")) == "ALLOW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
