#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


from work.recap.advantage import (
    VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE,
    build_diagnostic_surface_metadata,
)


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


DEFAULT_BASE_NONE_JSON = "agent/artifacts/vlm_critic_relabel/base_none.json"
DEFAULT_FINETUNED_NONE_JSON = "agent/artifacts/vlm_critic_relabel/finetuned_none.json"
DEFAULT_FINETUNED_ZERO_JSON = "agent/artifacts/vlm_critic_relabel/finetuned_zero.json"
DEFAULT_FINETUNED_POS_JSON = "agent/artifacts/vlm_critic_relabel/finetuned_pos.json"
DEFAULT_FINETUNE_JSON = "agent/artifacts/vlm_critic_relabel/finetune_smoke.json"
DEFAULT_CRITIC_AUDIT_JSON = (
    "agent/artifacts/vlm_critic_offline_gate/task7_formal_gate_v2.json"
)
DEFAULT_OUTPUT_JSON = "agent/artifacts/vlm_critic_relabel/downstream_gate.json"
MAX_RETENTION_DROP = 0.05
EXPECTED_TASK_TEXT_FIELD = "prompt_raw"
EXPECTED_CONTRACT_VERSION = "full_recap_continuous_adv_v1"
EXPECTED_INJECTION_RULE = "sign_consistent"
PASS_SENTINEL = "VLM_CRITIC_DOWNSTREAM_GATE_OK"
UPGRADE_PENDING = "temporal_critic_review"
DIAGNOSTIC_GATE_NAME = "vlm_critic_downstream_diagnostic_gate"
DIAGNOSTIC_GATE_ROUTE = "vlm_critic_downstream_diagnostic_gate"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return dict(data)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _summary_rate(summary: dict[str, Any], *, name: str, reasons: list[str]) -> float:
    raw = summary.get("success_rate")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        reasons.append(f"{name}_success_rate_missing_or_invalid")
        return 0.0
    return float(raw)


def _audit_finetune_summary(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if summary.get("error"):
        reasons.append(f"finetune_error: {summary.get('error')}")
    if str(summary.get("wrapper_status", "ok")) != "ok":
        reasons.append(f"finetune_wrapper_status={summary.get('wrapper_status')!r}")
    if str(summary.get("upgrade_pending")) != UPGRADE_PENDING:
        reasons.append(
            f"finetune_upgrade_pending_mismatch: {summary.get('upgrade_pending')!r}"
        )
    selected_checkpoint = summary.get("selected_checkpoint_path")
    if not isinstance(selected_checkpoint, str) or not selected_checkpoint.strip():
        reasons.append("finetune_selected_checkpoint_missing")
    if summary.get("selected_checkpoint_exists") is not True:
        reasons.append(
            f"finetune_selected_checkpoint_exists={summary.get('selected_checkpoint_exists')!r}"
        )
    upstream_summary = summary.get("upstream_summary")
    if not isinstance(upstream_summary, dict):
        reasons.append("finetune_upstream_summary_missing")
        return reasons
    completed_steps = upstream_summary.get("completed_steps")
    max_steps = upstream_summary.get("max_steps")
    if not isinstance(completed_steps, int) or completed_steps <= 0:
        reasons.append(f"finetune_completed_steps={completed_steps!r}")
    if not isinstance(max_steps, int) or max_steps <= 0:
        reasons.append(f"finetune_max_steps={max_steps!r}")
    if isinstance(completed_steps, int) and isinstance(max_steps, int):
        if completed_steps != max_steps:
            reasons.append(
                f"finetune_completed_steps_mismatch: completed={completed_steps} max={max_steps}"
            )
    return reasons


def _audit_eval_summary(summary: dict[str, Any], *, name: str) -> list[str]:
    reasons: list[str] = []
    if summary.get("error"):
        reasons.append(f"{name}_error: {summary.get('error')}")
    if str(summary.get("wrapper_status", "ok")) != "ok":
        reasons.append(f"{name}_wrapper_status={summary.get('wrapper_status')!r}")
    if str(summary.get("upgrade_pending")) != UPGRADE_PENDING:
        reasons.append(
            f"{name}_upgrade_pending_mismatch: {summary.get('upgrade_pending')!r}"
        )
    contract = summary.get("advantage_contract_version")
    if contract not in (None, EXPECTED_CONTRACT_VERSION):
        reasons.append(f"{name}_contract_version={contract!r}")
    task_text_field = summary.get("task_text_field")
    if task_text_field not in (None, EXPECTED_TASK_TEXT_FIELD):
        reasons.append(f"{name}_task_text_field={task_text_field!r}")
    injection = summary.get("advantage_injection_rule")
    if injection not in (None, EXPECTED_INJECTION_RULE):
        reasons.append(f"{name}_advantage_injection_rule={injection!r}")
    return reasons


def _critic_gate_reasons(critic_audit: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(critic_audit.get("reintegrate_verdict")) != "ALLOW":
        reasons.append(
            f"critic_reintegrate_verdict={critic_audit.get('reintegrate_verdict')!r}"
        )
    if str(critic_audit.get("reintegrate_status")) != "REINTEGRATE_ALLOWED":
        reasons.append(
            f"critic_reintegrate_status={critic_audit.get('reintegrate_status')!r}"
        )
    prompt_shortcut_risk = critic_audit.get("prompt_shortcut_risk")
    if prompt_shortcut_risk not in (None, False):
        reasons.append(f"critic_prompt_shortcut_risk={prompt_shortcut_risk!r}")
    return reasons


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="45e_vlm_critic_downstream_gate.py",
        description=(
            "Summarize T10 downstream smoke outputs into one machine-readable gate JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-none-json", type=str, default=DEFAULT_BASE_NONE_JSON)
    parser.add_argument(
        "--finetuned-none-json", type=str, default=DEFAULT_FINETUNED_NONE_JSON
    )
    parser.add_argument(
        "--finetuned-zero-json", type=str, default=DEFAULT_FINETUNED_ZERO_JSON
    )
    parser.add_argument(
        "--finetuned-pos-json", type=str, default=DEFAULT_FINETUNED_POS_JSON
    )
    parser.add_argument("--finetune-json", type=str, default=DEFAULT_FINETUNE_JSON)
    parser.add_argument(
        "--critic-audit-json", type=str, default=DEFAULT_CRITIC_AUDIT_JSON
    )
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--max-retention-drop",
        type=float,
        default=float(MAX_RETENTION_DROP),
        help="Allowed success-rate drop for finetuned_none relative to base_none.",
    )
    return parser


def build_downstream_gate_payload(
    *,
    base_rate: float,
    none_rate: float,
    zero_rate: float,
    pos_rate: float,
    retention_drop: float,
    gate_reasons: list[str],
    retention_passed: bool,
    controllability_passed: bool,
    critic_passed: bool,
    finetune_reasons: list[str],
    critic_audit: dict[str, Any],
    finetune_summary: dict[str, Any],
    args: argparse.Namespace,
    base_none_path: Path,
    finetuned_none_path: Path,
    finetuned_zero_path: Path,
    finetuned_pos_path: Path,
    finetune_path: Path,
    critic_audit_path: Path,
    gate_passed: bool,
) -> dict[str, Any]:
    blocker_reason = None if gate_passed else "; ".join(gate_reasons[:8])
    payload: dict[str, Any] = {
        "timestamp": __import__("datetime")
        .datetime.now()
        .isoformat(timespec="seconds"),
        "wrapper": "45e_vlm_critic_downstream_gate.py",
        "sentinel": PASS_SENTINEL,
        "gate_name": DIAGNOSTIC_GATE_NAME,
        "gate_passed": bool(gate_passed),
        "gate_status": "DIAGNOSTIC_PASS" if gate_passed else "DIAGNOSTIC_BLOCK",
        "gate_semantics": "diagnostic_only_non_release_gate",
        "release_gate": False,
        "blocker_reason": blocker_reason,
        "next_required_task": (
            None
            if gate_passed
            else "resolve_downstream_smoke_blocker_before_final_verification"
        ),
        "retention_passed": bool(retention_passed),
        "controllability_passed": bool(controllability_passed),
        "finetune_passed": not finetune_reasons,
        "critic_passed": bool(critic_passed),
        "base_none_success_rate": float(base_rate),
        "finetuned_none_success_rate": float(none_rate),
        "finetuned_zero_success_rate": float(zero_rate),
        "finetuned_pos_success_rate": float(pos_rate),
        "retention_drop": float(retention_drop),
        "max_retention_drop": float(args.max_retention_drop),
        "gate_reasons": gate_reasons,
        "upgrade_pending": UPGRADE_PENDING,
        "critic_audit_summary": {
            "critic_dir": critic_audit.get("critic_dir"),
            "reintegrate_verdict": critic_audit.get("reintegrate_verdict"),
            "reintegrate_status": critic_audit.get("reintegrate_status"),
            "full_input_auc": critic_audit.get("full_input_auc"),
            "baseline_delta_auc": critic_audit.get("baseline_delta_auc"),
            "prompt_shortcut_risk": critic_audit.get("prompt_shortcut_risk"),
        },
        "inputs": {
            "base_none_json": str(base_none_path),
            "finetuned_none_json": str(finetuned_none_path),
            "finetuned_zero_json": str(finetuned_zero_path),
            "finetuned_pos_json": str(finetuned_pos_path),
            "finetune_json": str(finetune_path),
            "critic_audit_json": str(critic_audit_path),
        },
        "finetune_summary": {
            "output_dir": finetune_summary.get("output_dir"),
            "selected_checkpoint_path": finetune_summary.get(
                "selected_checkpoint_path"
            ),
            "wrapper_status": finetune_summary.get("wrapper_status"),
            "upstream_returncode": finetune_summary.get("upstream_returncode"),
            "completed_steps": (
                finetune_summary.get("upstream_summary", {}).get("completed_steps")
                if isinstance(finetune_summary.get("upstream_summary"), dict)
                else None
            ),
            "max_steps": (
                finetune_summary.get("upstream_summary", {}).get("max_steps")
                if isinstance(finetune_summary.get("upstream_summary"), dict)
                else None
            ),
        },
    }
    payload.update(
        build_diagnostic_surface_metadata(
            surface_route=DIAGNOSTIC_GATE_ROUTE,
            authority_scope=VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE,
            surface_kind="vlm_critic_downstream_gate",
        )
    )
    return payload


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = _repo_root()

    base_none_path = _resolve_path(repo_root, str(args.base_none_json))
    finetuned_none_path = _resolve_path(repo_root, str(args.finetuned_none_json))
    finetuned_zero_path = _resolve_path(repo_root, str(args.finetuned_zero_json))
    finetuned_pos_path = _resolve_path(repo_root, str(args.finetuned_pos_json))
    finetune_path = _resolve_path(repo_root, str(args.finetune_json))
    critic_audit_path = _resolve_path(repo_root, str(args.critic_audit_json))
    output_json = _resolve_path(repo_root, str(args.output_json))

    base_none = _read_json(base_none_path)
    finetuned_none = _read_json(finetuned_none_path)
    finetuned_zero = _read_json(finetuned_zero_path)
    finetuned_pos = _read_json(finetuned_pos_path)
    finetune_summary = _read_json(finetune_path)
    critic_audit = _read_json(critic_audit_path)

    gate_reasons: list[str] = []
    finetune_reasons = _audit_finetune_summary(finetune_summary)
    gate_reasons.extend(finetune_reasons)
    gate_reasons.extend(_audit_eval_summary(base_none, name="base_none"))
    gate_reasons.extend(_audit_eval_summary(finetuned_none, name="finetuned_none"))
    gate_reasons.extend(_audit_eval_summary(finetuned_zero, name="finetuned_zero"))
    gate_reasons.extend(_audit_eval_summary(finetuned_pos, name="finetuned_pos"))
    gate_reasons.extend(_critic_gate_reasons(critic_audit))

    rate_reasons: list[str] = []
    base_rate = _summary_rate(base_none, name="base_none", reasons=rate_reasons)
    none_rate = _summary_rate(
        finetuned_none, name="finetuned_none", reasons=rate_reasons
    )
    zero_rate = _summary_rate(
        finetuned_zero, name="finetuned_zero", reasons=rate_reasons
    )
    pos_rate = _summary_rate(finetuned_pos, name="finetuned_pos", reasons=rate_reasons)
    gate_reasons.extend(rate_reasons)

    retention_drop = float(base_rate - none_rate)
    retention_passed = bool(retention_drop <= float(args.max_retention_drop))
    controllability_passed = bool(pos_rate > none_rate and pos_rate > zero_rate)
    critic_passed = not _critic_gate_reasons(critic_audit)

    if not retention_passed:
        gate_reasons.append(
            "retention_drop_exceeded: "
            f"drop={retention_drop:.6f} max={float(args.max_retention_drop):.6f}"
        )
    if not controllability_passed:
        gate_reasons.append(
            "controllability_failed: require finetuned_pos success_rate > finetuned_none and > finetuned_zero"
        )

    gate_passed = bool(
        retention_passed
        and controllability_passed
        and critic_passed
        and not rate_reasons
        and not _audit_eval_summary(base_none, name="base_none")
        and not _audit_eval_summary(finetuned_none, name="finetuned_none")
        and not _audit_eval_summary(finetuned_zero, name="finetuned_zero")
        and not _audit_eval_summary(finetuned_pos, name="finetuned_pos")
    )
    payload = build_downstream_gate_payload(
        base_rate=base_rate,
        none_rate=none_rate,
        zero_rate=zero_rate,
        pos_rate=pos_rate,
        retention_drop=retention_drop,
        gate_reasons=gate_reasons,
        retention_passed=retention_passed,
        controllability_passed=controllability_passed,
        critic_passed=critic_passed,
        finetune_reasons=finetune_reasons,
        critic_audit=critic_audit,
        finetune_summary=finetune_summary,
        args=args,
        base_none_path=base_none_path,
        finetuned_none_path=finetuned_none_path,
        finetuned_zero_path=finetuned_zero_path,
        finetuned_pos_path=finetuned_pos_path,
        finetune_path=finetune_path,
        critic_audit_path=critic_audit_path,
        gate_passed=gate_passed,
    )
    _write_json(output_json, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    print(f"SENTINEL:{PASS_SENTINEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
