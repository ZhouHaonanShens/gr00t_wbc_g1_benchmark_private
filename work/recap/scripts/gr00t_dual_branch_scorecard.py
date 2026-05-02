from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

REPO_ROOT = Path(__file__).resolve().parents[3]

REPORT_SCHEMA_VERSION = "gr00t_dual_branch_scorecard_v1"
REPORT_ARTIFACT_KIND = "gr00t_dual_branch_scorecard"

DEFAULT_OUTPUT = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/dual_branch_scorecard.json"
)
FAILURE_NOTE_MARKDOWN_NAME = "dual_branch_scorecard_failure_note.md"

DEFAULT_TASK4_PUBLIC_ANCHOR_EVIDENCE = Path(
    ".sisyphus/evidence/task-4-public-anchor.json"
)
DEFAULT_TASK5_UNITREE_AUDIT_EVIDENCE = Path(
    ".sisyphus/evidence/task-5-unitree-audit.json"
)
DEFAULT_TASK6_NEW_EMBODIMENT_AUDIT_EVIDENCE = Path(
    ".sisyphus/evidence/task-6-new-embodiment-audit.json"
)
DEFAULT_TASK7_ACTION_TELEMETRY_EVIDENCE = Path(
    ".sisyphus/evidence/task-7-action-telemetry.json"
)
DEFAULT_TASK8_CONDITION_FLIP_EVIDENCE = Path(
    ".sisyphus/evidence/task-8-condition-flip.json"
)
DEFAULT_TASK9_TEACHER_STUDENT_GAP_EVIDENCE = Path(
    ".sisyphus/evidence/task-9-teacher-student-gap.json"
)
DEFAULT_TASK10_TEACHER_REACHABILITY_EVIDENCE = Path(
    ".sisyphus/evidence/task-10-teacher-reachability.json"
)

BRANCH_UNITREE_G1 = "UNITREE_G1"
BRANCH_NEW_EMBODIMENT = "NEW_EMBODIMENT"

BRANCH_KEY_UNITREE_G1 = "unitree_g1"
BRANCH_KEY_NEW_EMBODIMENT = "new_embodiment"
BRANCH_ORDER: tuple[str, str] = (
    BRANCH_KEY_UNITREE_G1,
    BRANCH_KEY_NEW_EMBODIMENT,
)

CHECK_PUBLIC_ANCHOR = "public_anchor"
CHECK_CONTROLLER_AUDIT = "controller_audit"
CHECK_ACTION_TELEMETRY = "action_telemetry"
CHECK_CONDITION_FLIP = "condition_flip"
CHECK_TEACHER_STUDENT_GAP = "teacher_student_gap"
CHECK_TEACHER_REACHABILITY = "teacher_reachability"

CHECK_STATUS_PASS = "PASS"
CHECK_STATUS_FAIL = "FAIL"
CHECK_STATUS_MISSING = "MISSING"
CHECK_STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"

DIAGNOSTIC_METRICS: tuple[str, ...] = (
    "public_anchor_success_rate",
    "public_anchor_systemic_break_flags",
    "condition_flip_min_response_ratio",
    "condition_flip_passing_variants",
    "action_telemetry_controller_absorbed_groups",
    "action_telemetry_model_insensitive_groups",
    "action_telemetry_zero_motion_groups",
    "teacher_reachable_scene_count",
    "teacher_unreachable_family_count",
    "teacher_student_branch_match_rate",
    "teacher_student_action_group_gap_count",
)


@dataclass(frozen=True)
class BranchSpec:
    branch_key: str
    embodiment_tag: str
    branch_scope: str
    official_comparable_line: bool
    internal_only_comparable_line: bool
    controller_audit_evidence_path: Path
    controller_audit_field: str
    action_telemetry_field: str
    condition_flip_field: str
    teacher_student_gap_field: str
    teacher_reachability_field: str
    public_anchor_required: bool


BRANCH_SPECS: dict[str, BranchSpec] = {
    BRANCH_KEY_UNITREE_G1: BranchSpec(
        branch_key=BRANCH_KEY_UNITREE_G1,
        embodiment_tag=BRANCH_UNITREE_G1,
        branch_scope="official_public_anchor_line",
        official_comparable_line=True,
        internal_only_comparable_line=False,
        controller_audit_evidence_path=DEFAULT_TASK5_UNITREE_AUDIT_EVIDENCE,
        controller_audit_field="report_artifact",
        action_telemetry_field="unitree_output",
        condition_flip_field="unitree_output",
        teacher_student_gap_field="unitree_output",
        teacher_reachability_field="unitree_output",
        public_anchor_required=True,
    ),
    BRANCH_KEY_NEW_EMBODIMENT: BranchSpec(
        branch_key=BRANCH_KEY_NEW_EMBODIMENT,
        embodiment_tag=BRANCH_NEW_EMBODIMENT,
        branch_scope="branch_internal_only",
        official_comparable_line=False,
        internal_only_comparable_line=True,
        controller_audit_evidence_path=DEFAULT_TASK6_NEW_EMBODIMENT_AUDIT_EVIDENCE,
        controller_audit_field="report_artifact",
        action_telemetry_field="new_embodiment_output",
        condition_flip_field="new_embodiment_output",
        teacher_student_gap_field="new_embodiment_output",
        teacher_reachability_field="new_embodiment_output",
        public_anchor_required=False,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_dual_branch_scorecard.py",
        description=(
            "Aggregate tasks 4-10 into one dual-branch scorecard that keeps "
            "UNITREE_G1 officially comparable and NEW_EMBODIMENT internal-only."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument(
        "--task4-public-anchor-evidence",
        type=Path,
        default=DEFAULT_TASK4_PUBLIC_ANCHOR_EVIDENCE,
    )
    _ = parser.add_argument(
        "--task5-unitree-audit-evidence",
        type=Path,
        default=DEFAULT_TASK5_UNITREE_AUDIT_EVIDENCE,
    )
    _ = parser.add_argument(
        "--task6-new-embodiment-audit-evidence",
        type=Path,
        default=DEFAULT_TASK6_NEW_EMBODIMENT_AUDIT_EVIDENCE,
    )
    _ = parser.add_argument(
        "--task7-action-telemetry-evidence",
        type=Path,
        default=DEFAULT_TASK7_ACTION_TELEMETRY_EVIDENCE,
    )
    _ = parser.add_argument(
        "--task8-condition-flip-evidence",
        type=Path,
        default=DEFAULT_TASK8_CONDITION_FLIP_EVIDENCE,
    )
    _ = parser.add_argument(
        "--task9-teacher-student-gap-evidence",
        type=Path,
        default=DEFAULT_TASK9_TEACHER_STUDENT_GAP_EVIDENCE,
    )
    _ = parser.add_argument(
        "--task10-teacher-reachability-evidence",
        type=Path,
        default=DEFAULT_TASK10_TEACHER_REACHABILITY_EVIDENCE,
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _resolve_path(path: Path | str) -> Path:
    raw = Path(path).expanduser()
    return raw if raw.is_absolute() else (REPO_ROOT / raw)


def _validate_output_path(path: Path) -> Path:
    resolved = _resolve_path(path).resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(
            f"output must be a file path, got existing directory: {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _rel_repo(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        _ = f.write("\n")
    _ = tmp_path.replace(path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in {path}, got {type(payload).__name__}"
        )
    return cast(dict[str, Any], payload)


def _as_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected object for {context}, got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _as_list(value: object, *, context: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected list for {context}, got {type(value).__name__}")
    return list(value)


def _as_str(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for {context}, got {value!r}")
    return str(value)


def _as_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected float-like value for {context}, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(
        f"Expected float-like value for {context}, got {type(value).__name__}"
    )


def _as_bool(value: object) -> bool:
    return bool(value)


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        return None
    return _read_json(resolved)


def _artifact_from_evidence(
    evidence_path: Path,
    *,
    field: str,
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    resolved_evidence = _resolve_path(evidence_path)
    if not resolved_evidence.is_file():
        return None, None, f"missing_evidence:{_rel_repo(resolved_evidence)}"
    evidence = _read_json(resolved_evidence)
    implementation = _as_mapping(
        evidence.get("implementation", {}),
        context=f"{resolved_evidence}.implementation",
    )
    raw_artifact = implementation.get(field)
    if raw_artifact is None:
        return None, None, f"missing_evidence_field:{resolved_evidence.name}:{field}"
    artifact_path = _resolve_path(Path(_as_str(raw_artifact, context=f"{field}")))
    if not artifact_path.is_file():
        return artifact_path, None, f"missing_artifact:{_rel_repo(artifact_path)}"
    return artifact_path, _read_json(artifact_path), None


def _build_check(
    *,
    name: str,
    required: bool,
    status: str,
    reason_code: str,
    summary: Mapping[str, Any],
    artifact_path: Path | None,
    missing_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "required": bool(required),
        "status": status,
        "reason_code": reason_code,
        "artifact_path": _rel_repo(artifact_path)
        if artifact_path is not None
        else None,
        "missing_reason": missing_reason,
        "summary": dict(summary),
    }


def _public_anchor_check(evidence_path: Path, spec: BranchSpec) -> dict[str, Any]:
    if not spec.public_anchor_required:
        return _build_check(
            name=CHECK_PUBLIC_ANCHOR,
            required=False,
            status=CHECK_STATUS_NOT_APPLICABLE,
            reason_code="internal_only_no_public_anchor",
            summary={
                "public_anchor_comparable": False,
                "status": "NOT_APPLICABLE",
                "note": "NEW_EMBODIMENT has no official public anchor comparability line.",
            },
            artifact_path=None,
        )

    resolved_evidence = _resolve_path(evidence_path)
    if not resolved_evidence.is_file():
        return _build_check(
            name=CHECK_PUBLIC_ANCHOR,
            required=True,
            status=CHECK_STATUS_MISSING,
            reason_code="missing_public_anchor_evidence",
            summary={"public_anchor_comparable": True},
            artifact_path=resolved_evidence,
            missing_reason=f"missing_evidence:{_rel_repo(resolved_evidence)}",
        )

    evidence = _read_json(resolved_evidence)
    implementation = _as_mapping(
        evidence.get("implementation", {}), context="task4.implementation"
    )
    formal_artifact_path = _resolve_path(
        Path(
            _as_str(
                implementation.get("formal_artifact"), context="task4.formal_artifact"
            )
        )
    )
    sanity_gate_path = _resolve_path(
        Path(
            _as_str(
                implementation.get("sanity_gate_artifact"),
                context="task4.sanity_gate_artifact",
            )
        )
    )
    if not formal_artifact_path.is_file() or not sanity_gate_path.is_file():
        missing_parts = []
        if not formal_artifact_path.is_file():
            missing_parts.append(_rel_repo(formal_artifact_path))
        if not sanity_gate_path.is_file():
            missing_parts.append(_rel_repo(sanity_gate_path))
        return _build_check(
            name=CHECK_PUBLIC_ANCHOR,
            required=True,
            status=CHECK_STATUS_MISSING,
            reason_code="missing_public_anchor_artifact",
            summary={"public_anchor_comparable": True},
            artifact_path=formal_artifact_path,
            missing_reason=",".join(missing_parts),
        )

    formal = _read_json(formal_artifact_path)
    gate = _read_json(sanity_gate_path)
    systemic_break_flags = [
        str(item)
        for item in _as_list(
            formal.get("systemic_break_flags", []),
            context="task4.formal.systemic_break_flags",
        )
    ]
    continue_to_audit = _as_bool(gate.get("continue_to_audit", False))
    public_anchor_comparable = _as_bool(gate.get("public_anchor_comparable", True))
    passed = continue_to_audit and public_anchor_comparable and not systemic_break_flags
    return _build_check(
        name=CHECK_PUBLIC_ANCHOR,
        required=True,
        status=CHECK_STATUS_PASS if passed else CHECK_STATUS_FAIL,
        reason_code=(
            "public_anchor_continue_to_audit"
            if passed
            else str(gate.get("gate_reason", "public_anchor_blocked"))
        ),
        summary={
            "public_anchor_comparable": public_anchor_comparable,
            "continue_to_audit": continue_to_audit,
            "sanity_status": gate.get("sanity_status"),
            "success_rate": _as_float(
                formal.get("success_rate", 0.0), context="task4.success_rate"
            ),
            "success_count": int(
                _as_float(
                    formal.get("success_count", 0.0), context="task4.success_count"
                )
            ),
            "systemic_break_flags": systemic_break_flags,
        },
        artifact_path=formal_artifact_path,
    )


def _controller_audit_check(spec: BranchSpec) -> dict[str, Any]:
    artifact_path, payload, missing_reason = _artifact_from_evidence(
        spec.controller_audit_evidence_path,
        field=spec.controller_audit_field,
    )
    if payload is None:
        return _build_check(
            name=CHECK_CONTROLLER_AUDIT,
            required=True,
            status=CHECK_STATUS_MISSING,
            reason_code="missing_controller_audit_artifact",
            summary={"public_anchor_comparable": spec.official_comparable_line},
            artifact_path=artifact_path,
            missing_reason=missing_reason,
        )

    mismatch_fields = [
        str(item)
        for item in _as_list(
            payload.get("mismatch_fields", []),
            context="controller_audit.mismatch_fields",
        )
    ]
    public_anchor_comparable = _as_bool(
        payload.get("public_anchor_comparable", spec.official_comparable_line)
    )
    if spec.branch_key == BRANCH_KEY_UNITREE_G1:
        passed = (
            bool(payload.get("equivalent_to_official_unitree_g1", False))
            and not mismatch_fields
        )
        reason_code = (
            "unitree_controller_equivalent"
            if passed
            else "unitree_controller_not_equivalent"
        )
        summary = {
            "equivalent_to_official_unitree_g1": bool(
                payload.get("equivalent_to_official_unitree_g1", False)
            ),
            "mismatch_fields": mismatch_fields,
            "policy_horizon_expected": payload.get("policy_horizon_expected"),
            "policy_horizon_runtime": payload.get("policy_horizon_runtime"),
            "public_anchor_comparable": public_anchor_comparable,
        }
    else:
        blockers = [
            str(item)
            for item in _as_list(
                payload.get("formal_branch_blockers", []),
                context="controller_audit.formal_branch_blockers",
            )
        ]
        passed = (
            str(payload.get("formal_branch_eligibility", "")) == "ALLOW"
            and str(payload.get("reason_code", "")) == "OK"
            and not blockers
            and not public_anchor_comparable
        )
        reason_code = (
            "new_embodiment_branch_contract_allow"
            if passed
            else str(payload.get("reason_code", "new_embodiment_branch_contract_block"))
        )
        summary = {
            "formal_branch_eligibility": payload.get("formal_branch_eligibility"),
            "reason_code": payload.get("reason_code"),
            "formal_branch_blockers": blockers,
            "branch_manifest_path": payload.get("branch_manifest_path"),
            "public_anchor_comparable": public_anchor_comparable,
        }
    return _build_check(
        name=CHECK_CONTROLLER_AUDIT,
        required=True,
        status=CHECK_STATUS_PASS if passed else CHECK_STATUS_FAIL,
        reason_code=reason_code,
        summary=summary,
        artifact_path=artifact_path,
    )


def _comparability_matches(payload: Mapping[str, Any], spec: BranchSpec) -> bool:
    return (
        bool(payload.get("public_anchor_comparable", False))
        == spec.official_comparable_line
    )


def _action_telemetry_check(evidence_path: Path, spec: BranchSpec) -> dict[str, Any]:
    artifact_path, payload, missing_reason = _artifact_from_evidence(
        evidence_path,
        field=spec.action_telemetry_field,
    )
    if payload is None:
        return _build_check(
            name=CHECK_ACTION_TELEMETRY,
            required=True,
            status=CHECK_STATUS_MISSING,
            reason_code="missing_action_telemetry_artifact",
            summary={},
            artifact_path=artifact_path,
            missing_reason=missing_reason,
        )

    zero_motion_flags = _as_mapping(
        payload.get("zero_motion_flags", {}),
        context="action_telemetry.zero_motion_flags",
    )
    comparability_ok = _comparability_matches(payload, spec)
    passed = comparability_ok
    return _build_check(
        name=CHECK_ACTION_TELEMETRY,
        required=True,
        status=CHECK_STATUS_PASS if passed else CHECK_STATUS_FAIL,
        reason_code=(
            "action_telemetry_available"
            if passed
            else "action_telemetry_comparability_mismatch"
        ),
        summary={
            "public_anchor_comparable": bool(
                payload.get("public_anchor_comparable", False)
            ),
            "controller_absorbed_upstream_difference": bool(
                payload.get("controller_absorbed_upstream_difference", False)
            ),
            "controller_absorbed_groups": [
                str(item)
                for item in _as_list(
                    payload.get("controller_absorbed_groups", []),
                    context="action_telemetry.controller_absorbed_groups",
                )
            ],
            "model_insensitive_groups": [
                str(item)
                for item in _as_list(
                    payload.get("model_insensitive_groups", []),
                    context="action_telemetry.model_insensitive_groups",
                )
            ],
            "zero_motion_groups": [
                str(item)
                for item in _as_list(
                    zero_motion_flags.get("all_zero_in_both_groups", []),
                    context="action_telemetry.zero_motion_flags.all_zero_in_both_groups",
                )
            ],
        },
        artifact_path=artifact_path,
    )


def _condition_flip_check(evidence_path: Path, spec: BranchSpec) -> dict[str, Any]:
    artifact_path, payload, missing_reason = _artifact_from_evidence(
        evidence_path,
        field=spec.condition_flip_field,
    )
    if payload is None:
        return _build_check(
            name=CHECK_CONDITION_FLIP,
            required=True,
            status=CHECK_STATUS_MISSING,
            reason_code="missing_condition_flip_artifact",
            summary={},
            artifact_path=artifact_path,
            missing_reason=missing_reason,
        )

    gate_details = _as_mapping(
        payload.get("gate_details", {}), context="condition_flip.gate_details"
    )
    passed = str(
        payload.get("pass_fail_gate", "")
    ) == "PASS" and _comparability_matches(payload, spec)
    return _build_check(
        name=CHECK_CONDITION_FLIP,
        required=True,
        status=CHECK_STATUS_PASS if passed else CHECK_STATUS_FAIL,
        reason_code=(
            str(
                gate_details.get("reason_code", "semantic_condition_branching_detected")
            )
            if not passed
            else "condition_flip_pass"
        ),
        summary={
            "pass_fail_gate": payload.get("pass_fail_gate"),
            "public_anchor_comparable": bool(
                payload.get("public_anchor_comparable", False)
            ),
            "paired_scene_id": payload.get("paired_scene_id"),
            "min_ratio_across_semantic_flips": _as_float(
                _as_mapping(
                    payload.get("response_ratio", {}),
                    context="condition_flip.response_ratio",
                ).get("min_ratio_across_semantic_flips", 0.0),
                context="condition_flip.min_ratio_across_semantic_flips",
            ),
            "passing_variants": [
                str(item)
                for item in _as_list(
                    gate_details.get("passing_variants", []),
                    context="condition_flip.passing_variants",
                )
            ],
            "reason_code": gate_details.get("reason_code"),
        },
        artifact_path=artifact_path,
    )


def _teacher_student_gap_check(evidence_path: Path, spec: BranchSpec) -> dict[str, Any]:
    artifact_path, payload, missing_reason = _artifact_from_evidence(
        evidence_path,
        field=spec.teacher_student_gap_field,
    )
    if payload is None:
        return _build_check(
            name=CHECK_TEACHER_STUDENT_GAP,
            required=True,
            status=CHECK_STATUS_MISSING,
            reason_code="missing_teacher_student_gap_artifact",
            summary={},
            artifact_path=artifact_path,
            missing_reason=missing_reason,
        )

    summary = _as_mapping(
        payload.get("summary", {}), context="teacher_student_gap.summary"
    )
    passed = str(payload.get("status", "")) == "ALLOW" and _comparability_matches(
        payload, spec
    )
    return _build_check(
        name=CHECK_TEACHER_STUDENT_GAP,
        required=True,
        status=CHECK_STATUS_PASS if passed else CHECK_STATUS_FAIL,
        reason_code=(
            "teacher_student_gap_allow"
            if passed
            else str(payload.get("case_code", "teacher_student_gap_block"))
        ),
        summary={
            "status": payload.get("status"),
            "public_anchor_comparable": bool(
                payload.get("public_anchor_comparable", False)
            ),
            "case_code": payload.get("case_code"),
            "teacher_case_code": payload.get("teacher_case_code"),
            "scene_pool_status": payload.get("scene_pool_status"),
            "student_branch_match_rate": _as_float(
                payload.get("student_branch_match_rate", 0.0),
                context="teacher_student_gap.student_branch_match_rate",
            ),
            "action_group_gap_count": int(
                _as_float(
                    summary.get("action_group_gap_count", 0.0),
                    context="teacher_student_gap.summary.action_group_gap_count",
                )
            ),
            "teacher_reachable_families": [
                str(item)
                for item in _as_list(
                    payload.get("teacher_reachable_families", []),
                    context="teacher_student_gap.teacher_reachable_families",
                )
            ],
            "teacher_unreachable_families": [
                str(item)
                for item in _as_list(
                    payload.get("teacher_unreachable_families", []),
                    context="teacher_student_gap.teacher_unreachable_families",
                )
            ],
        },
        artifact_path=artifact_path,
    )


def _teacher_reachability_check(
    evidence_path: Path, spec: BranchSpec
) -> dict[str, Any]:
    artifact_path, payload, missing_reason = _artifact_from_evidence(
        evidence_path,
        field=spec.teacher_reachability_field,
    )
    if payload is None:
        return _build_check(
            name=CHECK_TEACHER_REACHABILITY,
            required=True,
            status=CHECK_STATUS_MISSING,
            reason_code="missing_teacher_reachability_artifact",
            summary={},
            artifact_path=artifact_path,
            missing_reason=missing_reason,
        )

    teacher_upper_bound = _as_mapping(
        payload.get("teacher_upper_bound", {}),
        context="teacher_reachability.teacher_upper_bound",
    )
    branch_prerequisites = _as_mapping(
        payload.get("branch_prerequisites", {}),
        context="teacher_reachability.branch_prerequisites",
    )
    blocking_reasons = [
        str(item)
        for item in _as_list(
            payload.get("blocking_reasons", []),
            context="teacher_reachability.blocking_reasons",
        )
    ]
    passed = (
        bool(payload.get("allow_formal_ladders", False))
        and str(payload.get("status", "")) == "ALLOW"
        and _comparability_matches(payload, spec)
    )
    return _build_check(
        name=CHECK_TEACHER_REACHABILITY,
        required=True,
        status=CHECK_STATUS_PASS if passed else CHECK_STATUS_FAIL,
        reason_code=(
            "teacher_reachability_allow"
            if passed
            else str(
                payload.get(
                    "reason_code",
                    payload.get("scene_pool_status", "teacher_reachability_block"),
                )
            )
        ),
        summary={
            "status": payload.get("status"),
            "allow_formal_ladders": bool(payload.get("allow_formal_ladders", False)),
            "public_anchor_comparable": bool(
                payload.get("public_anchor_comparable", False)
            ),
            "scene_pool_status": payload.get("scene_pool_status"),
            "teacher_case_code": payload.get("teacher_case_code"),
            "replay_case_code": payload.get("replay_case_code"),
            "reachable_scene_ids": [
                str(item)
                for item in _as_list(
                    payload.get("reachable_scene_ids", []),
                    context="teacher_reachability.reachable_scene_ids",
                )
            ],
            "blocking_reasons": blocking_reasons,
            "teacher_reachable_rate": _as_float(
                teacher_upper_bound.get("teacher_reachable_rate", 0.0),
                context="teacher_reachability.teacher_upper_bound.teacher_reachable_rate",
            ),
            "teacher_success_count": int(
                _as_float(
                    teacher_upper_bound.get("teacher_success_count", 0.0),
                    context="teacher_reachability.teacher_upper_bound.teacher_success_count",
                )
            ),
            "branch_prerequisite_ok": bool(
                branch_prerequisites.get("branch_prerequisite_ok", False)
            ),
            "branch_blockers": [
                str(item)
                for item in _as_list(
                    branch_prerequisites.get("branch_blockers", []),
                    context="teacher_reachability.branch_prerequisites.branch_blockers",
                )
            ],
        },
        artifact_path=artifact_path,
    )


def _recommended_next_step(
    *,
    missing_checks: Sequence[str],
    failed_checks: Sequence[str],
    has_watchlist: bool,
) -> str:
    if missing_checks:
        return "restore_missing_prerequisite_artifacts_before_task_12"
    if CHECK_PUBLIC_ANCHOR in failed_checks or CHECK_CONTROLLER_AUDIT in failed_checks:
        return "return_to_stack_or_controller_fix_before_ladders"
    if CHECK_TEACHER_REACHABILITY in failed_checks:
        return "fix_teacher_reachability_or_scene_pool_before_ladders"
    if CHECK_CONDITION_FLIP in failed_checks:
        return "fix_condition_interface_before_ladders"
    if CHECK_TEACHER_STUDENT_GAP in failed_checks:
        return "fix_teacher_student_gap_diagnostics_before_ladders"
    if CHECK_ACTION_TELEMETRY in failed_checks:
        return "fix_branch_comparability_diagnostics_before_ladders"
    if has_watchlist:
        return "proceed_to_task_12_ladder_policy_gate_with_diagnostics_watchlist"
    return "proceed_to_task_12_ladder_policy_gate"


def _build_branch_report(
    *,
    spec: BranchSpec,
    task4_public_anchor_evidence: Path,
    task7_action_telemetry_evidence: Path,
    task8_condition_flip_evidence: Path,
    task9_teacher_student_gap_evidence: Path,
    task10_teacher_reachability_evidence: Path,
) -> dict[str, Any]:
    checks = [
        _public_anchor_check(task4_public_anchor_evidence, spec),
        _controller_audit_check(spec),
        _action_telemetry_check(task7_action_telemetry_evidence, spec),
        _condition_flip_check(task8_condition_flip_evidence, spec),
        _teacher_student_gap_check(task9_teacher_student_gap_evidence, spec),
        _teacher_reachability_check(task10_teacher_reachability_evidence, spec),
    ]
    checks_by_name = {str(item["name"]): item for item in checks}

    missing_checks = [
        str(item["name"])
        for item in checks
        if bool(item["required"]) and str(item["status"]) == CHECK_STATUS_MISSING
    ]
    failed_checks = [
        str(item["name"])
        for item in checks
        if bool(item["required"]) and str(item["status"]) == CHECK_STATUS_FAIL
    ]
    passed_checks = [
        str(item["name"])
        for item in checks
        if bool(item["required"]) and str(item["status"]) == CHECK_STATUS_PASS
    ]
    required_checks = [str(item["name"]) for item in checks if bool(item["required"])]
    not_applicable_checks = [
        str(item["name"])
        for item in checks
        if str(item["status"]) == CHECK_STATUS_NOT_APPLICABLE
    ]

    prerequisite_ok = not missing_checks and not failed_checks
    action_summary = _as_mapping(
        checks_by_name[CHECK_ACTION_TELEMETRY]["summary"],
        context="branch.action_telemetry.summary",
    )
    teacher_gap_summary = _as_mapping(
        checks_by_name[CHECK_TEACHER_STUDENT_GAP]["summary"],
        context="branch.teacher_student_gap.summary",
    )
    reachability_summary = _as_mapping(
        checks_by_name[CHECK_TEACHER_REACHABILITY]["summary"],
        context="branch.teacher_reachability.summary",
    )
    public_anchor_summary = _as_mapping(
        checks_by_name[CHECK_PUBLIC_ANCHOR]["summary"],
        context="branch.public_anchor.summary",
    )
    condition_flip_summary = _as_mapping(
        checks_by_name[CHECK_CONDITION_FLIP]["summary"],
        context="branch.condition_flip.summary",
    )

    controller_absorbed_groups = [
        str(item)
        for item in _as_list(
            action_summary.get("controller_absorbed_groups", []),
            context="branch.controller_absorbed_groups",
        )
    ]
    model_insensitive_groups = [
        str(item)
        for item in _as_list(
            action_summary.get("model_insensitive_groups", []),
            context="branch.model_insensitive_groups",
        )
    ]
    zero_motion_groups = [
        str(item)
        for item in _as_list(
            action_summary.get("zero_motion_groups", []),
            context="branch.zero_motion_groups",
        )
    ]
    watchlist = bool(
        controller_absorbed_groups or model_insensitive_groups or zero_motion_groups
    )
    recommended_next_step = _recommended_next_step(
        missing_checks=missing_checks,
        failed_checks=failed_checks,
        has_watchlist=watchlist,
    )

    teacher_reachable_families = [
        str(item)
        for item in _as_list(
            teacher_gap_summary.get("teacher_reachable_families", []),
            context="branch.teacher_reachable_families",
        )
    ]
    teacher_unreachable_families = [
        str(item)
        for item in _as_list(
            teacher_gap_summary.get("teacher_unreachable_families", []),
            context="branch.teacher_unreachable_families",
        )
    ]
    reachable_scene_ids = [
        str(item)
        for item in _as_list(
            reachability_summary.get("reachable_scene_ids", []),
            context="branch.reachable_scene_ids",
        )
    ]
    systemic_break_flags = [
        str(item)
        for item in _as_list(
            public_anchor_summary.get("systemic_break_flags", []),
            context="branch.systemic_break_flags",
        )
    ]
    if not spec.public_anchor_required:
        systemic_break_flags.extend(
            str(item)
            for item in _as_list(
                _as_mapping(
                    checks_by_name[CHECK_CONTROLLER_AUDIT]["summary"],
                    context="branch.controller.summary",
                ).get("formal_branch_blockers", []),
                context="branch.formal_branch_blockers",
            )
        )

    prerequisite_status = {
        "status": "PASS" if prerequisite_ok else "BLOCK",
        "required_checks": required_checks,
        "not_applicable_checks": not_applicable_checks,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "missing_checks": missing_checks,
        "blocking_reasons": [
            str(checks_by_name[name]["reason_code"])
            for name in [*missing_checks, *failed_checks]
        ],
    }

    branch_report = {
        "branch_key": spec.branch_key,
        "embodiment_tag": spec.embodiment_tag,
        "branch_scope": spec.branch_scope,
        "official_comparable_line": spec.official_comparable_line,
        "internal_only_comparable_line": spec.internal_only_comparable_line,
        "public_anchor_comparable": spec.official_comparable_line,
        "prerequisite_status": prerequisite_status,
        "public_anchor_status": checks_by_name[CHECK_PUBLIC_ANCHOR],
        "controller_equivalence": checks_by_name[CHECK_CONTROLLER_AUDIT],
        "action_telemetry": checks_by_name[CHECK_ACTION_TELEMETRY],
        "condition_flip": checks_by_name[CHECK_CONDITION_FLIP],
        "teacher_student_gap": checks_by_name[CHECK_TEACHER_STUDENT_GAP],
        "teacher_reachability": checks_by_name[CHECK_TEACHER_REACHABILITY],
        "diagnostic_summary": {
            "public_anchor_success_rate": public_anchor_summary.get("success_rate"),
            "public_anchor_systemic_break_flags": systemic_break_flags,
            "condition_flip_min_response_ratio": condition_flip_summary.get(
                "min_ratio_across_semantic_flips"
            ),
            "condition_flip_passing_variants": condition_flip_summary.get(
                "passing_variants"
            ),
            "action_telemetry_controller_absorbed_groups": controller_absorbed_groups,
            "action_telemetry_model_insensitive_groups": model_insensitive_groups,
            "action_telemetry_zero_motion_groups": zero_motion_groups,
            "teacher_reachable_scene_count": len(reachable_scene_ids),
            "teacher_unreachable_family_count": len(teacher_unreachable_families),
            "teacher_student_branch_match_rate": teacher_gap_summary.get(
                "student_branch_match_rate"
            ),
            "teacher_student_action_group_gap_count": teacher_gap_summary.get(
                "action_group_gap_count"
            ),
            "teacher_reachable_families": teacher_reachable_families,
            "teacher_unreachable_families": teacher_unreachable_families,
            "reachable_scene_ids": reachable_scene_ids,
        },
        "systemic_break_flags": systemic_break_flags,
        "allow_p_ladder": prerequisite_ok,
        "allow_d_ladder": prerequisite_ok,
        "recommended_next_step": recommended_next_step,
    }
    return branch_report


def build_dual_branch_scorecard(
    *,
    output_path: Path,
    task4_public_anchor_evidence: Path = DEFAULT_TASK4_PUBLIC_ANCHOR_EVIDENCE,
    task5_unitree_audit_evidence: Path = DEFAULT_TASK5_UNITREE_AUDIT_EVIDENCE,
    task6_new_embodiment_audit_evidence: Path = DEFAULT_TASK6_NEW_EMBODIMENT_AUDIT_EVIDENCE,
    task7_action_telemetry_evidence: Path = DEFAULT_TASK7_ACTION_TELEMETRY_EVIDENCE,
    task8_condition_flip_evidence: Path = DEFAULT_TASK8_CONDITION_FLIP_EVIDENCE,
    task9_teacher_student_gap_evidence: Path = DEFAULT_TASK9_TEACHER_STUDENT_GAP_EVIDENCE,
    task10_teacher_reachability_evidence: Path = DEFAULT_TASK10_TEACHER_REACHABILITY_EVIDENCE,
) -> dict[str, Any]:
    branch_specs = {
        BRANCH_KEY_UNITREE_G1: replace(
            BRANCH_SPECS[BRANCH_KEY_UNITREE_G1],
            controller_audit_evidence_path=task5_unitree_audit_evidence,
        ),
        BRANCH_KEY_NEW_EMBODIMENT: replace(
            BRANCH_SPECS[BRANCH_KEY_NEW_EMBODIMENT],
            controller_audit_evidence_path=task6_new_embodiment_audit_evidence,
        ),
    }

    branches = [
        _build_branch_report(
            spec=branch_specs[branch_key],
            task4_public_anchor_evidence=task4_public_anchor_evidence,
            task7_action_telemetry_evidence=task7_action_telemetry_evidence,
            task8_condition_flip_evidence=task8_condition_flip_evidence,
            task9_teacher_student_gap_evidence=task9_teacher_student_gap_evidence,
            task10_teacher_reachability_evidence=task10_teacher_reachability_evidence,
        )
        for branch_key in BRANCH_ORDER
    ]
    prerequisite_status = {
        str(branch["branch_key"]): dict(
            _as_mapping(
                branch["prerequisite_status"], context="branch.prerequisite_status"
            )
        )
        for branch in branches
    }
    allow_p_ladder = {
        str(branch["branch_key"]): bool(branch["allow_p_ladder"]) for branch in branches
    }
    allow_d_ladder = {
        str(branch["branch_key"]): bool(branch["allow_d_ladder"]) for branch in branches
    }
    recommended_next_step = {
        str(branch["branch_key"]): str(branch["recommended_next_step"])
        for branch in branches
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "output_path": _rel_repo(output_path),
        "failure_note_path": None,
        "branch_order": list(BRANCH_ORDER),
        "official_comparable_line": BRANCH_KEY_UNITREE_G1,
        "internal_only_comparable_line": BRANCH_KEY_NEW_EMBODIMENT,
        "comparability_semantics": {
            BRANCH_KEY_UNITREE_G1: "official_public_anchor_line",
            BRANCH_KEY_NEW_EMBODIMENT: "branch_internal_only",
        },
        "prerequisite_status": prerequisite_status,
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "allow_p_ladder": allow_p_ladder,
        "allow_d_ladder": allow_d_ladder,
        "recommended_next_step": recommended_next_step,
        "branches": branches,
    }


def _build_failure_note(report: Mapping[str, Any]) -> str:
    branches = _as_list(report.get("branches", []), context="scorecard.branches")
    blocked = [
        cast(Mapping[str, Any], branch)
        for branch in branches
        if not bool(cast(Mapping[str, Any], branch).get("allow_p_ladder", False))
    ]
    lines = [
        "# GR00T dual-branch scorecard failure note",
        "",
        f"- output_path: `{report.get('output_path')}`",
        f"- blocked_branch_count: `{len(blocked)}`",
        f"- official_comparable_line: `{report.get('official_comparable_line')}`",
        f"- internal_only_comparable_line: `{report.get('internal_only_comparable_line')}`",
        "",
    ]
    for branch in blocked:
        prerequisite_status = _as_mapping(
            branch.get("prerequisite_status", {}),
            context="failure_note.prerequisite_status",
        )
        lines.extend(
            [
                f"## {branch.get('branch_key')}",
                f"- embodiment_tag: `{branch.get('embodiment_tag')}`",
                f"- branch_scope: `{branch.get('branch_scope')}`",
                f"- prerequisite_status: `{prerequisite_status.get('status')}`",
                f"- missing_checks: `{json.dumps(prerequisite_status.get('missing_checks', []), ensure_ascii=True)}`",
                f"- failed_checks: `{json.dumps(prerequisite_status.get('failed_checks', []), ensure_ascii=True)}`",
                f"- recommended_next_step: `{branch.get('recommended_next_step')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_scorecard_artifacts(report: dict[str, Any], *, output_path: Path) -> Path:
    failure_note_path = output_path.with_name(FAILURE_NOTE_MARKDOWN_NAME)
    blocked = any(
        not bool(value)
        for value in cast(Mapping[str, Any], report["allow_p_ladder"]).values()
    )
    if blocked:
        failure_note_path.write_text(_build_failure_note(report), encoding="utf-8")
        report["failure_note_path"] = _rel_repo(failure_note_path)
    elif failure_note_path.exists():
        failure_note_path.unlink()
        report["failure_note_path"] = None
    return _write_json(output_path, report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = _validate_output_path(args.output)
        report = build_dual_branch_scorecard(
            output_path=output_path,
            task4_public_anchor_evidence=args.task4_public_anchor_evidence,
            task5_unitree_audit_evidence=args.task5_unitree_audit_evidence,
            task6_new_embodiment_audit_evidence=args.task6_new_embodiment_audit_evidence,
            task7_action_telemetry_evidence=args.task7_action_telemetry_evidence,
            task8_condition_flip_evidence=args.task8_condition_flip_evidence,
            task9_teacher_student_gap_evidence=args.task9_teacher_student_gap_evidence,
            task10_teacher_reachability_evidence=args.task10_teacher_reachability_evidence,
        )
        _ = write_scorecard_artifacts(report, output_path=output_path)
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
