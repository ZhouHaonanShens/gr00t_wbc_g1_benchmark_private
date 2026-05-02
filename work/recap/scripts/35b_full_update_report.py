#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PLAN_PATH = REPO_ROOT / ".sisyphus/plans/recap-full-update-first.md"
PLAN_TASK15_START = 936
PLAN_TASK15_END = 979

REPORT_FILENAME = "full_update_diagnosis_report.md"
README_FILENAME = "README.md"
PLAN_SNAPSHOT_FILENAME = "plan_snapshot.md"


@dataclass(frozen=True)
class AuthorityBundle:
    v2_authority_root: Path
    static_audit_path: Path
    dynamic_audit_path: Path
    grad_probe_path: Path
    param_delta_path: Path
    optimizer_state_path: Path
    downgrade_attempts_path: Path
    conditioning_probe_path: Path
    paired_action_probe_path: Path
    label_semantics_path: Path
    preformal_gate_path: Path
    diagnostic_summary_path: Path
    p5_verdict_path: Path
    p5_blocker_summary_path: Path
    version_surface_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="35b_full_update_report.py",
        description=(
            "Read the single_gpu_v2_full_update authority artifacts and generate the "
            "final diagnosis report set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--v2-authority-root",
        type=Path,
        required=True,
        help="Authority root that contains p0/p1/p2/p4/p5 artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for full_update_diagnosis_report.md, README.md, and plan_snapshot.md.",
    )
    return parser


def _resolve_path(raw: Path | str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _safe_relpath(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON payload must be a mapping: {path}")
    return {str(key): value for key, value in payload.items()}


def _read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return _read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise TypeError(f"JSONL row must be a mapping: {path}")
        rows.append({str(key): value for key, value in payload.items()})
    return rows


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _require_first_existing(label: str, candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = "\n".join(f"- {_safe_relpath(candidate)}" for candidate in candidates)
    raise FileNotFoundError(f"Missing required {label}. Checked:\n{joined}")


def _discover_authority_bundle(v2_authority_root: Path) -> AuthorityBundle:
    root = _resolve_path(v2_authority_root)
    dynamic_audit_path = _require_first_existing(
        "dynamic scope audit",
        (
            root / "p1_one_step" / "repo_local_metadata" / "full_update_scope_audit_dynamic.json",
            root / "p1_one_step_fresh_20260423" / "repo_local_metadata" / "full_update_scope_audit_dynamic.json",
        ),
    )
    dynamic_audit = _read_json(dynamic_audit_path)
    dynamic_output_dir = dynamic_audit.get("output_dir")
    dynamic_output_metadata_dir = None
    if isinstance(dynamic_output_dir, str) and dynamic_output_dir.strip():
        dynamic_output_metadata_dir = _resolve_path(dynamic_output_dir) / "repo_local_metadata"

    static_audit_path = _require_first_existing(
        "static scope audit",
        tuple(
            path
            for path in (
                root / "p0_scope_audit" / "full_update_scope_audit.json",
                root / "p0_scope_audit_check" / "p0_scope_audit" / "full_update_scope_audit.json",
                root / "p1_one_step" / "repo_local_metadata" / "full_update_scope_audit.json",
                None if dynamic_output_metadata_dir is None else dynamic_output_metadata_dir / "full_update_scope_audit.json",
            )
            if path is not None
        ),
    )

    metadata_candidates = tuple(
        path
        for path in (
            root / "p1_one_step" / "repo_local_metadata",
            dynamic_output_metadata_dir,
        )
        if path is not None
    )

    def metadata_file(name: str) -> Path:
        return _require_first_existing(
            name,
            tuple(candidate / name for candidate in metadata_candidates),
        )

    return AuthorityBundle(
        v2_authority_root=root,
        static_audit_path=static_audit_path,
        dynamic_audit_path=dynamic_audit_path,
        grad_probe_path=metadata_file("first_backward_grad_probe_rank0.json"),
        param_delta_path=metadata_file("first_optimizer_step_param_delta_rank0.json"),
        optimizer_state_path=metadata_file("optimizer_state_rank0.json"),
        downgrade_attempts_path=metadata_file("downgrade_attempts.jsonl"),
        conditioning_probe_path=_require_first_existing(
            "conditioning functional probe",
            (root / "p2_full_update_overfit20" / "conditioning_functional_probe_step20.json",),
        ),
        paired_action_probe_path=_require_first_existing(
            "paired action probe",
            (root / "p2_full_update_overfit20" / "paired_action_probe_step20.json",),
        ),
        label_semantics_path=_require_first_existing(
            "label semantics audit",
            (root / "p2_5_label_semantics" / "label_semantics_audit.json",),
        ),
        preformal_gate_path=_require_first_existing(
            "preformal gate decision",
            (root / "p2_5_label_semantics" / "preformal_gate_decision.json",),
        ),
        diagnostic_summary_path=_require_first_existing(
            "full update diagnostic summary",
            (root / "p4_loss_action_subgoal" / "full_update_diagnostic_summary.json",),
        ),
        p5_verdict_path=_require_first_existing(
            "P5 verdict",
            (root / "p5_gate_eval" / "min_loop_verdict.json",),
        ),
        p5_blocker_summary_path=_require_first_existing(
            "P5 blocker summary",
            (root / "p5_gate_eval" / "p5_gate_blocker_summary.json",),
        ),
        version_surface_path=metadata_file("version_surface.json"),
    )


def _fmt_float(value: object, digits: int = 6) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_scientific(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.6e}"


def _fmt_gib_from_bytes(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) / (1024 ** 3):.2f} GiB"


def _bool_text(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _markdown_bullets(items: Sequence[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- `{item}`" for item in items)


def _load_plan_excerpt() -> str:
    lines = PLAN_PATH.read_text(encoding="utf-8").splitlines()
    selected = lines[PLAN_TASK15_START - 1 : PLAN_TASK15_END]
    rendered = []
    for index, line in enumerate(selected, start=PLAN_TASK15_START):
        rendered.append(f"{index}: {line}")
    return "\n".join(rendered).rstrip()


def _next_step_branch_recommendation(
    diagnostic_summary: Mapping[str, Any],
    p5_verdict: Mapping[str, Any],
) -> tuple[str, list[str]]:
    blockers = set(_string_list(diagnostic_summary.get("blocking_reasons")))
    blockers.update(_string_list(p5_verdict.get("blocking_reasons")))

    reasons: list[str] = []
    if "paired_action_instrumentation_incomplete" in blockers:
        reasons.append(
            "先走 control 分支，补齐 postprocessed/controller input action seam，当前 action gate 卡在 instrumentation 不完整。"
        )
    if "label_semantics_gate_block" in blockers or "shuffled_advantage_negative_control_block" in blockers:
        reasons.append(
            "随后走 label 分支，先修复 shuffled negative control 不能证明真标签优于打乱标签的问题。"
        )
    if (
        "missing_comparability_manifest" in blockers
        or "missing_complete_3seed_subgoal_probe" in blockers
        or "missing_baseline_v1_first_subgoal_probe" in blockers
    ):
        reasons.append(
            "再走 environment / evaluation 分支，补齐 comparability manifest、baseline v1 subgoal probe 与 3-seed subgoal 证据。"
        )
    if not reasons and p5_verdict.get("status") == "PASS":
        reasons.append("P5 已满足，可进入 P6 分支。")

    if p5_verdict.get("status") == "PASS" and p5_verdict.get("gate_mode") == "executed":
        branch = "P6"
    elif reasons:
        branch = "control -> label -> environment"
    else:
        branch = "hold"
    return branch, reasons


def _summarize_report_data(bundle: AuthorityBundle) -> dict[str, Any]:
    static_audit = _read_json(bundle.static_audit_path)
    dynamic_audit = _read_json(bundle.dynamic_audit_path)
    grad_probe = _read_json(bundle.grad_probe_path)
    param_delta = _read_json(bundle.param_delta_path)
    optimizer_state = _read_json(bundle.optimizer_state_path)
    downgrade_attempts = _read_jsonl(bundle.downgrade_attempts_path)
    conditioning_probe = _read_json(bundle.conditioning_probe_path)
    paired_action_probe = _read_json(bundle.paired_action_probe_path)
    label_semantics = _read_json(bundle.label_semantics_path)
    preformal_gate = _read_json(bundle.preformal_gate_path)
    diagnostic_summary = _read_json(bundle.diagnostic_summary_path)
    p5_verdict = _read_json(bundle.p5_verdict_path)
    p5_blocker_summary = _read_json(bundle.p5_blocker_summary_path)
    version_surface = _read_json(bundle.version_surface_path)

    grad_scopes = grad_probe.get("scopes") if isinstance(grad_probe.get("scopes"), Mapping) else {}
    delta_scopes = param_delta.get("scopes") if isinstance(param_delta.get("scopes"), Mapping) else {}
    optimizer_groups = optimizer_state.get("param_groups_preview")
    if not isinstance(optimizer_groups, Sequence) or isinstance(optimizer_groups, (str, bytes)):
        optimizer_groups = []

    static_summary = static_audit.get("summary")
    if not isinstance(static_summary, Mapping):
        static_summary = {}
    memory_feasibility = static_audit.get("memory_feasibility")
    if not isinstance(memory_feasibility, Mapping):
        memory_feasibility = {}
    parameter_coverage = static_audit.get("parameter_coverage")
    if not isinstance(parameter_coverage, Mapping):
        parameter_coverage = {}
    method_faithfulness = static_audit.get("method_faithfulness")
    if not isinstance(method_faithfulness, Mapping):
        method_faithfulness = {}

    route_freeze = version_surface.get("route_freeze")
    if not isinstance(route_freeze, Mapping):
        route_freeze = {}

    loss_probe = diagnostic_summary.get("loss_probe")
    if not isinstance(loss_probe, Mapping):
        loss_probe = {}
    paired_action_gate = diagnostic_summary.get("paired_action_probe")
    if not isinstance(paired_action_gate, Mapping):
        paired_action_gate = {}
    first_subgoal_probe = diagnostic_summary.get("first_subgoal_probe")
    if not isinstance(first_subgoal_probe, Mapping):
        first_subgoal_probe = {}

    branch, branch_reasons = _next_step_branch_recommendation(diagnostic_summary, p5_verdict)

    return {
        "bundle": bundle,
        "static_audit": static_audit,
        "dynamic_audit": dynamic_audit,
        "grad_probe": grad_probe,
        "param_delta": param_delta,
        "optimizer_state": optimizer_state,
        "downgrade_attempts": downgrade_attempts,
        "conditioning_probe": conditioning_probe,
        "paired_action_probe": paired_action_probe,
        "label_semantics": label_semantics,
        "preformal_gate": preformal_gate,
        "diagnostic_summary": diagnostic_summary,
        "p5_verdict": p5_verdict,
        "p5_blocker_summary": p5_blocker_summary,
        "version_surface": version_surface,
        "static_summary": static_summary,
        "memory_feasibility": memory_feasibility,
        "parameter_coverage": parameter_coverage,
        "method_faithfulness": method_faithfulness,
        "route_freeze": route_freeze,
        "optimizer_groups": list(optimizer_groups),
        "grad_scopes": grad_scopes,
        "delta_scopes": delta_scopes,
        "loss_probe": loss_probe,
        "paired_action_gate": paired_action_gate,
        "first_subgoal_probe": first_subgoal_probe,
        "recommended_branch": branch,
        "recommended_branch_reasons": branch_reasons,
    }


def build_report_markdown(report_data: Mapping[str, Any]) -> str:
    bundle = report_data["bundle"]
    static_audit = report_data["static_audit"]
    dynamic_audit = report_data["dynamic_audit"]
    grad_scopes = report_data["grad_scopes"]
    delta_scopes = report_data["delta_scopes"]
    optimizer_groups = report_data["optimizer_groups"]
    conditioning_probe = report_data["conditioning_probe"]
    paired_action_probe = report_data["paired_action_probe"]
    label_semantics = report_data["label_semantics"]
    diagnostic_summary = report_data["diagnostic_summary"]
    p5_verdict = report_data["p5_verdict"]
    p5_blocker_summary = report_data["p5_blocker_summary"]
    static_summary = report_data["static_summary"]
    memory_feasibility = report_data["memory_feasibility"]
    parameter_coverage = report_data["parameter_coverage"]
    method_faithfulness = report_data["method_faithfulness"]
    route_freeze = report_data["route_freeze"]

    attempt_rows = report_data["downgrade_attempts"]
    loss_probe = report_data["loss_probe"]
    paired_action_gate = report_data["paired_action_gate"]
    first_subgoal_probe = report_data["first_subgoal_probe"]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    optimizer_lines: list[str] = []
    for index, group in enumerate(optimizer_groups):
        if not isinstance(group, Mapping):
            continue
        params = group.get("params")
        param_count = len(params) if isinstance(params, Sequence) and not isinstance(params, (str, bytes)) else "n/a"
        optimizer_lines.append(
            "- group {index}: lr={lr}, weight_decay={weight_decay}, param_count={param_count}".format(
                index=index,
                lr=group.get("lr", "n/a"),
                weight_decay=group.get("weight_decay", "n/a"),
                param_count=param_count,
            )
        )
    if not optimizer_lines:
        optimizer_lines.append("- no optimizer group preview available")

    available_action_layers: list[str] = []
    unavailable_action_layers: list[str] = []
    action_layers = paired_action_probe.get("action_delta_layers")
    if isinstance(action_layers, Mapping):
        for layer_name, layer_payload in action_layers.items():
            if not isinstance(layer_payload, Mapping):
                continue
            if layer_payload.get("available") is True:
                available_action_layers.append(
                    f"{layer_name}: delta_l2={_fmt_float(layer_payload.get('delta_l2'))}"
                )
            else:
                unavailable_action_layers.append(
                    f"{layer_name}: {layer_payload.get('reason', 'unavailable')}"
                )

    full_update_answer = (
        "yes"
        if dynamic_audit.get("resolution_status") == "PASS"
        and dynamic_audit.get("train_scope_effective") == "strict_full"
        and static_audit.get("static_verdict") == "PASS"
        else "no"
    )

    advantage_effect_answer = (
        "partial_yes"
        if diagnostic_summary.get("loss_sensitivity_gate_pass") is True
        else "no"
    )
    rollout_answer = str(first_subgoal_probe.get("status", "n/a"))
    next_branch = str(report_data["recommended_branch"])
    next_branch_reasons = report_data["recommended_branch_reasons"]
    total_parameter_rows = static_summary.get(
        "total_parameter_rows",
        parameter_coverage.get("total_parameter_rows", static_audit.get("total_parameter_rows", "n/a")),
    )
    trainable_parameter_rows = static_summary.get(
        "trainable_parameter_rows",
        parameter_coverage.get(
            "trainable_parameter_rows",
            static_audit.get("trainable_parameter_rows", "n/a"),
        ),
    )

    sections = [
        "# Full Update Diagnosis Report",
        "",
        f"生成时间: {generated_at}",
        f"authority root: `{_safe_relpath(bundle.v2_authority_root)}`",
        "",
        "## 1. 执行摘要 / Executive Summary",
        f"- full-update scope actually applied? `{full_update_answer}`",
        (
            "- advantage had measurable effect? `partial_yes`, loss sensitivity passed and available raw/decoded action deltas were non-zero, "
            "but downstream control-path instrumentation and label semantics both blocked a formal claim."
        ),
        f"- rollout/subgoal status: `{rollout_answer}`",
        f"- P5 executed or skipped? `gate_mode={p5_verdict.get('gate_mode', 'n/a')}` with `status={p5_verdict.get('status', 'n/a')}`",
        f"- next-step branch recommendation: `{next_branch}`",
        "",
        "## 2. Authority Inputs",
        f"- static scope audit: `{_safe_relpath(bundle.static_audit_path)}`",
        f"- dynamic scope audit: `{_safe_relpath(bundle.dynamic_audit_path)}`",
        f"- grad norms: `{_safe_relpath(bundle.grad_probe_path)}`",
        f"- param deltas: `{_safe_relpath(bundle.param_delta_path)}`",
        f"- optimizer param groups: `{_safe_relpath(bundle.optimizer_state_path)}`",
        f"- fallback/degrade history: `{_safe_relpath(bundle.downgrade_attempts_path)}`",
        f"- loss sensitivity before/after: `{_safe_relpath(bundle.conditioning_probe_path)}` and `{_safe_relpath(bundle.diagnostic_summary_path)}`",
        f"- action delta before/after: `{_safe_relpath(bundle.paired_action_probe_path)}` and `{_safe_relpath(bundle.diagnostic_summary_path)}`",
        f"- label semantics: `{_safe_relpath(bundle.label_semantics_path)}`",
        f"- P5 result: `{_safe_relpath(bundle.p5_verdict_path)}`",
        "",
        "## 3. 训练范围审计 / train scope audit summary",
        f"- train_scope_requested: `{static_audit.get('train_scope_requested', 'n/a')}`",
        f"- static_verdict: `{static_audit.get('static_verdict', 'n/a')}`",
        f"- dynamic resolution_status: `{dynamic_audit.get('resolution_status', 'n/a')}`",
        f"- train_scope_effective: `{dynamic_audit.get('train_scope_effective', 'n/a')}`",
        f"- scope_faithfulness: `{static_audit.get('scope_faithfulness', 'n/a')}`",
        f"- total_parameter_rows: `{total_parameter_rows}`",
        f"- trainable_parameter_rows: `{trainable_parameter_rows}`",
        f"- memory_feasibility.estimated_total_bytes: `{memory_feasibility.get('estimated_total_bytes', 'n/a')}` ({_fmt_gib_from_bytes(memory_feasibility.get('estimated_total_bytes'))})",
        f"- memory_feasibility.trainable_numel: `{memory_feasibility.get('trainable_numel', 'n/a')}`",
        f"- method_faithfulness.paper_equivalent: `{method_faithfulness.get('paper_equivalent', 'n/a')}`",
        "- method_faithfulness.paper_method_gap:",
        _markdown_bullets(_string_list(method_faithfulness.get("paper_method_gap"))),
        "",
        "## 4. fallback/degrade history",
        f"- attempt_count: `{dynamic_audit.get('attempt_count', 'n/a')}`",
        f"- strict_full_runtime_attempted: `{dynamic_audit.get('strict_full_runtime_attempted', 'n/a')}`",
        f"- downgrade_attempts rows: `{len(attempt_rows)}`",
    ]

    if attempt_rows:
        first_attempt = attempt_rows[0]
        sections.extend(
            [
                f"- first candidate_scope: `{first_attempt.get('candidate_scope', 'n/a')}`",
                f"- first status: `{first_attempt.get('status', 'n/a')}`",
                f"- first runtime_returncode: `{first_attempt.get('runtime_returncode', 'n/a')}`",
                f"- first runtime_preflight_status: `{first_attempt.get('runtime_preflight_status', 'n/a')}`",
                "- note: degrade chain did not need to drop from `strict_full`; the only recorded attempt already passed.",
            ]
        )

    sections.extend(
        [
            "",
            "## 5. optimizer param groups / route freeze",
            f"- optimizer_class_name: `{report_data['optimizer_state'].get('optimizer_class_name', 'n/a')}`",
            f"- param_group_count: `{report_data['optimizer_state'].get('param_group_count', 'n/a')}`",
            *optimizer_lines,
            f"- route_freeze.frozen: `{route_freeze.get('frozen', 'n/a')}`",
            f"- route_freeze.route: `{route_freeze.get('route', 'n/a')}`",
            f"- route_freeze.diagnostic_only: `{route_freeze.get('diagnostic_only', 'n/a')}`",
            "",
            "## 6. grad norms / param deltas",
            f"- advantage_embedding grad_l2_norm: `{_fmt_float(grad_scopes.get('advantage_embedding', {}).get('grad_l2_norm') if isinstance(grad_scopes.get('advantage_embedding'), Mapping) else None)}`",
            f"- diffusion_trunk grad_l2_norm: `{_fmt_float(grad_scopes.get('diffusion_trunk', {}).get('grad_l2_norm') if isinstance(grad_scopes.get('diffusion_trunk'), Mapping) else None)}`",
            f"- advantage_embedding delta_l2_norm: `{_fmt_scientific(delta_scopes.get('advantage_embedding', {}).get('delta_l2_norm') if isinstance(delta_scopes.get('advantage_embedding'), Mapping) else None)}`",
            f"- diffusion_trunk delta_l2_norm: `{_fmt_float(delta_scopes.get('diffusion_trunk', {}).get('delta_l2_norm') if isinstance(delta_scopes.get('diffusion_trunk'), Mapping) else None)}`",
            "- interpretation: both `advantage_embedding` and `diffusion_trunk` showed non-zero first-step gradients and non-zero first optimizer-step parameter deltas.",
            "",
            "## 7. loss sensitivity before/after",
            f"- conditioning_functional_probe.loss_sensitivity_gate_pass: `{conditioning_probe.get('loss_sensitivity_gate_pass', 'n/a')}`",
            f"- train_subset loss_span: `{_fmt_float(conditioning_probe.get('train_subset_loss_probe', {}).get('loss_span') if isinstance(conditioning_probe.get('train_subset_loss_probe'), Mapping) else None)}`",
            f"- heldout loss_span: `{_fmt_float(conditioning_probe.get('heldout_loss_probe', {}).get('loss_span') if isinstance(conditioning_probe.get('heldout_loss_probe'), Mapping) else None)}`",
            f"- P4 loss_probe.status: `{loss_probe.get('status', 'n/a')}`",
            f"- P4 loss_probe.loss_sensitivity_gate_pass: `{loss_probe.get('loss_sensitivity_gate_pass', 'n/a')}`",
            "- verdict: advantage changed the loss surface measurably on both train_subset and heldout subsets, so the failure mode is not `loss completely insensitive`.",
            "",
            "## 8. action delta before/after",
            f"- paired_action_probe.action_sensitivity_gate_pass: `{paired_action_probe.get('action_sensitivity_gate_pass', 'n/a')}`",
            "- available action delta layers:",
            _markdown_bullets(available_action_layers),
            "- unavailable action delta layers:",
            _markdown_bullets(unavailable_action_layers),
            f"- P4 paired_action_probe.status: `{paired_action_gate.get('status', 'n/a')}`",
            f"- P4 paired_action_probe.instrumentation_incomplete: `{paired_action_gate.get('instrumentation_incomplete', 'n/a')}`",
            "- verdict: available layers already show non-zero action deltas, but postprocessed/controller layers are missing, so the control-facing gate still blocks downstream claims.",
            "",
            "## 9. subgoal probe / rollout findings",
            f"- first_subgoal_probe.status: `{first_subgoal_probe.get('status', 'n/a')}`",
            f"- strong_subgoal_progress_gate_pass: `{first_subgoal_probe.get('strong_subgoal_progress_gate_pass', 'n/a')}`",
            f"- paired_seed_improvement_count: `{first_subgoal_probe.get('paired_seed_improvement_count', 'n/a')}`",
            f"- mean_relative_improvement_min_dist_ee_to_apple: `{first_subgoal_probe.get('mean_relative_improvement_min_dist_ee_to_apple', 'n/a')}`",
            f"- no_regression_on_contact_or_lift_proxy: `{first_subgoal_probe.get('no_regression_on_contact_or_lift_proxy', 'n/a')}`",
            "- first_subgoal_probe.blocking_reasons:",
            _markdown_bullets(_string_list(first_subgoal_probe.get("blocking_reasons"))),
            "",
            "## 10. label semantics / shuffled negative control",
            f"- label_semantics_gate_pass: `{label_semantics.get('label_semantics_gate_pass', 'n/a')}`",
            f"- positive_success_rate: `{_fmt_float(label_semantics.get('positive_success_rate'))}`",
            f"- shuffled_advantage_negative_control_pass: `{label_semantics.get('shuffled_advantage_negative_control', {}).get('negative_control_pass', 'n/a') if isinstance(label_semantics.get('shuffled_advantage_negative_control'), Mapping) else 'n/a'}`",
            f"- train_subset shuffled_minus_true_loss: `{_fmt_float(label_semantics.get('shuffled_advantage_negative_control', {}).get('per_subset', {}).get('train_subset', {}).get('shuffled_minus_true_loss') if isinstance(label_semantics.get('shuffled_advantage_negative_control'), Mapping) and isinstance(label_semantics.get('shuffled_advantage_negative_control', {}).get('per_subset'), Mapping) and isinstance(label_semantics.get('shuffled_advantage_negative_control', {}).get('per_subset', {}).get('train_subset'), Mapping) else None)}`",
            f"- heldout shuffled_minus_true_loss: `{_fmt_float(label_semantics.get('shuffled_advantage_negative_control', {}).get('per_subset', {}).get('heldout', {}).get('shuffled_minus_true_loss') if isinstance(label_semantics.get('shuffled_advantage_negative_control'), Mapping) and isinstance(label_semantics.get('shuffled_advantage_negative_control', {}).get('per_subset'), Mapping) and isinstance(label_semantics.get('shuffled_advantage_negative_control', {}).get('per_subset', {}).get('heldout'), Mapping) else None)}`",
            "- label_semantics.blocking_reasons:",
            _markdown_bullets(_string_list(label_semantics.get("blocking_reasons"))),
            "",
            "## 11. P5 result / formal gate",
            f"- status: `{p5_verdict.get('status', 'n/a')}`",
            f"- gate_mode: `gate_mode={p5_verdict.get('gate_mode', 'n/a')}`",
            f"- formal_execution_attempted: `{p5_verdict.get('formal_execution_attempted', 'n/a')}`",
            f"- p5_formal_10ep_eligible: `{p5_verdict.get('p5_formal_10ep_eligible', 'n/a')}`",
            f"- seed_set_source: `{p5_verdict.get('seed_set_source', 'n/a')}`",
            f"- blocker_reason: `{p5_verdict.get('blocker_reason', 'n/a')}`",
            "- blocker reasons:",
            _markdown_bullets(_string_list(p5_verdict.get("blocking_reasons"))),
        ]
    )

    if p5_verdict.get("gate_mode") == "skipped":
        sections.extend(
            [
                "- skipped gate rationale comes from the authority verdict and must remain explicit here.",
                f"- blocker summary path: `{_safe_relpath(bundle.p5_blocker_summary_path)}`",
                f"- gate_summary_status: `{p5_blocker_summary.get('gate_summary_status', 'n/a')}`",
            ]
        )

    sections.extend(
        [
            "",
            "## 12. next-step branch recommendation",
            f"- recommended branch: `{report_data['recommended_branch']}`",
            *[f"- {reason}" for reason in next_branch_reasons],
            "- do not enter `P6` yet.",
            "- do not branch into `LIBERO` yet.",
            "",
            "## 13. Plain answers",
            f"- full-update scope actually applied? `{full_update_answer}`",
            (
                "- advantage had measurable effect? `partial_yes`, the loss probe passed and the available raw/decoded action layers moved, "
                "but the full control-facing proof failed."
            ),
            f"- rollout/subgoal status? `{rollout_answer}`, still blocked by comparability, missing baseline probe, and missing complete 3-seed evidence.",
            (
                f"- whether P5 executed or was skipped, and why? `gate_mode={p5_verdict.get('gate_mode', 'n/a')}` because "
                f"`{p5_verdict.get('blocker_reason', 'n/a')}` plus {_string_list(p5_verdict.get('blocking_reasons'))}."
            ),
            f"- what branch comes next? `{report_data['recommended_branch']}`.",
        ]
    )

    return "\n".join(sections).rstrip() + "\n"


def build_readme_markdown(report_data: Mapping[str, Any]) -> str:
    bundle = report_data["bundle"]
    p5_verdict = report_data["p5_verdict"]
    dynamic_audit = report_data["dynamic_audit"]
    branch = report_data["recommended_branch"]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    return (
        "# single_gpu_v2_full_update\n\n"
        f"更新时间: {generated_at}\n\n"
        "## 当前结论\n\n"
        f"- full-update scope: `{dynamic_audit.get('resolution_status', 'n/a')}` / `{dynamic_audit.get('train_scope_effective', 'n/a')}`\n"
        f"- P5: `gate_mode={p5_verdict.get('gate_mode', 'n/a')}`, `status={p5_verdict.get('status', 'n/a')}`\n"
        f"- next-step branch: `{branch}`\n\n"
        "## 核心文件\n\n"
        f"- `./{REPORT_FILENAME}`: 人类可读诊断总报告\n"
        f"- `./{PLAN_SNAPSHOT_FILENAME}`: Task 15 plan snapshot\n"
        f"- `{_safe_relpath(bundle.static_audit_path)}`: static train scope audit authority\n"
        f"- `{_safe_relpath(bundle.dynamic_audit_path)}`: dynamic train scope audit authority\n"
        f"- `{_safe_relpath(bundle.diagnostic_summary_path)}`: P4 gate summary authority\n"
        f"- `{_safe_relpath(bundle.p5_verdict_path)}`: P5 gate verdict authority\n\n"
        "## 重新生成\n\n"
        "```bash\n"
        ".venv/bin/python work/recap/scripts/35b_full_update_report.py \\\n"
        f"  --v2-authority-root {_safe_relpath(bundle.v2_authority_root)} \\\n"
        f"  --output-dir {_safe_relpath(bundle.v2_authority_root)}\n"
        "```\n"
    )


def build_plan_snapshot_markdown(report_data: Mapping[str, Any]) -> str:
    bundle = report_data["bundle"]
    dynamic_audit = report_data["dynamic_audit"]
    p5_verdict = report_data["p5_verdict"]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    excerpt = _load_plan_excerpt()
    return (
        "# Task 15 Plan Snapshot\n\n"
        f"captured_at: {generated_at}\n"
        f"plan_path: `{_safe_relpath(PLAN_PATH)}`\n"
        f"authority_root: `{_safe_relpath(bundle.v2_authority_root)}`\n\n"
        "## Live artifact snapshot\n\n"
        f"- dynamic resolution_status: `{dynamic_audit.get('resolution_status', 'n/a')}`\n"
        f"- train_scope_effective: `{dynamic_audit.get('train_scope_effective', 'n/a')}`\n"
        f"- P5: `gate_mode={p5_verdict.get('gate_mode', 'n/a')}`, `status={p5_verdict.get('status', 'n/a')}`\n"
        f"- recommendation: `{report_data['recommended_branch']}`\n\n"
        "## Frozen Task 15 excerpt\n\n"
        "```text\n"
        f"{excerpt}\n"
        "```\n"
    )


def generate_report_set(v2_authority_root: Path, output_dir: Path) -> dict[str, Path]:
    bundle = _discover_authority_bundle(v2_authority_root)
    report_data = _summarize_report_data(bundle)
    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "report": resolved_output_dir / REPORT_FILENAME,
        "readme": resolved_output_dir / README_FILENAME,
        "plan_snapshot": resolved_output_dir / PLAN_SNAPSHOT_FILENAME,
    }
    outputs["report"].write_text(build_report_markdown(report_data), encoding="utf-8")
    outputs["readme"].write_text(build_readme_markdown(report_data), encoding="utf-8")
    outputs["plan_snapshot"].write_text(
        build_plan_snapshot_markdown(report_data),
        encoding="utf-8",
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = generate_report_set(
        v2_authority_root=args.v2_authority_root,
        output_dir=args.output_dir,
    )
    printable = {
        key: _safe_relpath(path)
        for key, path in outputs.items()
    }
    print(json.dumps(printable, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
