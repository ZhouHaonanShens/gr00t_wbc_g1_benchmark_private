from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_DIR = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/teacher_reachability"
)
DEFAULT_TEACHER_GATE_JSON = Path(
    "agent/artifacts/state_conditioned_materialization/sanity/teacher_upper_bound_gate.json"
)
DEFAULT_UNITREE_PUBLIC_ANCHOR_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/public_anchor/public_anchor_formal.json"
)
DEFAULT_UNITREE_CONTROLLER_AUDIT_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/controller_audit_unitree_g1.json"
)
DEFAULT_NEW_EMBODIMENT_CONTROLLER_AUDIT_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/controller_audit_new_embodiment.json"
)
DEFAULT_NEW_EMBODIMENT_BRANCH_MANIFEST_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/branch_manifest.json"
)

REPORT_SCHEMA_VERSION = "gr00t_teacher_reachability_gate_v1"
REPORT_ARTIFACT_KIND = "gr00t_teacher_reachability_gate"
REPLAY_UPPER_BOUND_ARTIFACT_KIND = "gr00t_replay_upper_bound_report"

REPORT_JSON_NAME_BY_BRANCH = {
    "UNITREE_G1": "teacher_reachability_gate_unitree_g1.json",
    "NEW_EMBODIMENT": "teacher_reachability_gate_new_embodiment.json",
}
FAILURE_NOTE_MARKDOWN_NAME_BY_BRANCH = {
    "UNITREE_G1": "teacher_reachability_gate_unitree_g1_failure_note.md",
    "NEW_EMBODIMENT": "teacher_reachability_gate_new_embodiment_failure_note.md",
}
BRANCH_CHOICES = tuple(REPORT_JSON_NAME_BY_BRANCH.keys())

DEFAULT_REPLAY_SUCCESS_FAMILIES_BY_BRANCH = {
    "UNITREE_G1": ("S_drop", "S_lost"),
    "NEW_EMBODIMENT": ("S_drop", "S_lost"),
}

BRANCH_SCOPE_BY_BRANCH = {
    "UNITREE_G1": "official_public_anchor_line",
    "NEW_EMBODIMENT": "branch_internal_only",
}


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_teacher_reachability_gate.py",
        description=(
            "Combine teacher reachability, replay upper-bound guidance, and current "
            "baseline evidence into a deterministic stop/continue gate with a formal "
            "teacher-reachable scene pool."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--branch",
        required=True,
        choices=BRANCH_CHOICES,
        help="Embodiment branch whose attribution gate should be materialized.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory that receives the branch-specific teacher reachability gate JSON "
            "and, when blocked, a machine-readable failure note."
        ),
    )
    parser.add_argument(
        "--teacher-gate-json",
        type=Path,
        default=DEFAULT_TEACHER_GATE_JSON,
        help="Teacher upper-bound gate JSON that provides family-level reachability.",
    )
    parser.add_argument(
        "--current-baseline-json",
        type=Path,
        default=None,
        help=(
            "Optional current-baseline JSON. When omitted, UNITREE_G1 uses the public "
            "anchor formal report and NEW_EMBODIMENT conservatively defaults to zero."
        ),
    )
    parser.add_argument(
        "--replay-upper-bound-json",
        type=Path,
        default=None,
        help=(
            "Optional replay upper-bound JSON. When omitted, the gate builds a repo-local "
            "ReplayPolicy fixture aligned with the teacher family surface."
        ),
    )
    parser.add_argument(
        "--controller-audit-json",
        type=Path,
        default=None,
        help=(
            "Optional branch controller audit JSON. When omitted, the canonical task-5/6 "
            "artifact for the selected branch is used."
        ),
    )
    parser.add_argument(
        "--branch-manifest-json",
        type=Path,
        default=None,
        help=(
            "Optional NEW_EMBODIMENT branch manifest JSON. Ignored for UNITREE_G1. When "
            "omitted, the canonical task-6 branch manifest is used."
        ),
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _resolve_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{arg_name} does not exist: {resolved}")
    return resolved


def _read_json(path: Path, *, arg_name: str) -> dict[str, Any]:
    resolved = _resolve_existing_file(path, arg_name=arg_name)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return dict(_as_mapping(payload, field_name=arg_name))


def _scene_id(branch: str, family: str) -> str:
    return f"{str(branch).lower()}::{str(family)}"


def _branch_defaults(branch: str) -> dict[str, Path | None]:
    if branch == "UNITREE_G1":
        return {
            "current_baseline_json": DEFAULT_UNITREE_PUBLIC_ANCHOR_JSON,
            "controller_audit_json": DEFAULT_UNITREE_CONTROLLER_AUDIT_JSON,
            "branch_manifest_json": None,
        }
    if branch == "NEW_EMBODIMENT":
        return {
            "current_baseline_json": None,
            "controller_audit_json": DEFAULT_NEW_EMBODIMENT_CONTROLLER_AUDIT_JSON,
            "branch_manifest_json": DEFAULT_NEW_EMBODIMENT_BRANCH_MANIFEST_JSON,
        }
    raise ValueError(f"unsupported branch: {branch}")


def load_teacher_gate_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path, arg_name="teacher-gate-json")
    artifact_kind = payload.get("artifact_kind")
    if artifact_kind not in {None, "state_conditioned_teacher_upper_bound_gate"}:
        raise ValueError(
            "teacher-gate-json artifact_kind mismatch; expected "
            "state_conditioned_teacher_upper_bound_gate"
        )
    mapping = payload.get("mapping", {})
    mapping_obj = _as_mapping(mapping, field_name="teacher-gate-json.mapping")
    families = []
    total_teacher_success_count = 0
    total_snapshot_baseline_success_count = 0
    for index, row in enumerate(
        _as_list(payload.get("families", []), field_name="teacher-gate-json.families")
    ):
        row_obj = _as_mapping(row, field_name=f"teacher-gate-json.families[{index}]")
        family = _as_non_empty_string(
            row_obj.get("family"),
            field_name=f"teacher-gate-json.families[{index}].family",
        )
        success_count = _as_int(
            row_obj.get("success_count", 0),
            field_name=f"teacher-gate-json.families[{index}].success_count",
        )
        priority = _as_non_empty_string(
            row_obj.get("priority", "unknown"),
            field_name=f"teacher-gate-json.families[{index}].priority",
        )
        current_model_baseline = _as_mapping(
            row_obj.get("current_model_baseline", {}),
            field_name=(f"teacher-gate-json.families[{index}].current_model_baseline"),
        )
        snapshot_baseline_success_count = _as_int(
            current_model_baseline.get("success_count", 0),
            field_name=(
                f"teacher-gate-json.families[{index}].current_model_baseline.success_count"
            ),
        )
        scene_id = _scene_id(
            _as_non_empty_string(
                payload.get("branch_hint", payload.get("embodiment_tag", "teacher")),
                field_name="teacher-gate-json.branch_hint",
            ),
            family,
        )
        families.append(
            {
                "scene_id": scene_id,
                "family": family,
                "priority": priority,
                "teacher_success_count": success_count,
                "teacher_reachable": bool(success_count > 0),
                "teacher_interpretation_code": row_obj.get("interpretation_code"),
                "snapshot_baseline_success_count": snapshot_baseline_success_count,
            }
        )
        total_teacher_success_count += success_count
        total_snapshot_baseline_success_count += snapshot_baseline_success_count
    total_teacher_success_count = int(
        mapping_obj.get("teacher_success_count", total_teacher_success_count)
    )
    total_snapshot_baseline_success_count = int(
        mapping_obj.get(
            "current_model_baseline_success_count",
            total_snapshot_baseline_success_count,
        )
    )
    return {
        "source_path": str(_resolve_existing_file(path, arg_name="teacher-gate-json")),
        "artifact_kind": str(
            artifact_kind or "state_conditioned_teacher_upper_bound_gate"
        ),
        "teacher_success_count": int(total_teacher_success_count),
        "snapshot_baseline_success_count": int(total_snapshot_baseline_success_count),
        "teacher_reachable_rate": float(
            _as_number(
                mapping_obj.get(
                    "teacher_reachable_rate",
                    (
                        sum(1 for row in families if row["teacher_reachable"])
                        / len(families)
                    )
                    if families
                    else 0.0,
                ),
                field_name="teacher-gate-json.mapping.teacher_reachable_rate",
            )
        ),
        "families": families,
    }


def load_current_baseline_summary(branch: str, path: Path | None) -> dict[str, Any]:
    if path is None:
        if branch == "NEW_EMBODIMENT":
            return {
                "source_kind": "branch_local_pending_formal_eval",
                "source_path": None,
                "artifact_kind": None,
                "success_count": 0,
                "success_rate": 0.0,
                "public_anchor_comparable": False,
            }
        raise ValueError("current baseline path is required for UNITREE_G1")

    payload = _read_json(path, arg_name="current-baseline-json")
    artifact_kind = payload.get("artifact_kind")
    success_count = _as_int(
        payload.get("success_count", 0),
        field_name="current-baseline-json.success_count",
    )
    success_rate = _as_number(
        payload.get("success_rate", 0.0),
        field_name="current-baseline-json.success_rate",
    )
    if artifact_kind == "gr00t_public_anchor_formal":
        return {
            "source_kind": "public_anchor_formal",
            "source_path": str(
                _resolve_existing_file(path, arg_name="current-baseline-json")
            ),
            "artifact_kind": str(artifact_kind),
            "success_count": int(success_count),
            "success_rate": float(success_rate),
            "public_anchor_comparable": True,
        }
    return {
        "source_kind": "generic_current_baseline_json",
        "source_path": str(
            _resolve_existing_file(path, arg_name="current-baseline-json")
        ),
        "artifact_kind": artifact_kind,
        "success_count": int(success_count),
        "success_rate": float(success_rate),
        "public_anchor_comparable": bool(branch == "UNITREE_G1"),
    }


def load_branch_context(
    branch: str,
    *,
    controller_audit_path: Path,
    branch_manifest_path: Path | None,
) -> dict[str, Any]:
    payload = _read_json(controller_audit_path, arg_name="controller-audit-json")
    if branch == "UNITREE_G1":
        if payload.get("artifact_kind") != "gr00t_controller_audit_unitree_g1":
            raise ValueError(
                "controller-audit-json artifact_kind mismatch for UNITREE_G1"
            )
        controller_ok = bool(payload.get("equivalent_to_official_unitree_g1", False))
        mismatch_fields = _as_list(
            payload.get("mismatch_fields", []),
            field_name="controller-audit-json.mismatch_fields",
        )
        controller_ok = bool(controller_ok and not mismatch_fields)
        blockers = [] if controller_ok else ["controller_audit_not_equivalent"]
        return {
            "branch": branch,
            "controller_audit_path": str(
                _resolve_existing_file(
                    controller_audit_path, arg_name="controller-audit-json"
                )
            ),
            "branch_manifest_path": None,
            "branch_scope": BRANCH_SCOPE_BY_BRANCH[branch],
            "public_anchor_comparable": True,
            "branch_prerequisite_ok": controller_ok,
            "branch_prerequisite_code": "unitree_controller_equivalent",
            "branch_blockers": blockers,
        }

    if payload.get("artifact_kind") != "gr00t_controller_audit_new_embodiment":
        raise ValueError(
            "controller-audit-json artifact_kind mismatch for NEW_EMBODIMENT"
        )
    controller_ok = (
        _as_non_empty_string(
            payload.get("formal_branch_eligibility"),
            field_name="controller-audit-json.formal_branch_eligibility",
        )
        == "ALLOW"
    )
    manifest_payload: dict[str, Any] | None = None
    manifest_resolved: str | None = None
    if branch_manifest_path is not None:
        manifest_payload = _read_json(
            branch_manifest_path, arg_name="branch-manifest-json"
        )
        manifest_resolved = str(
            _resolve_existing_file(
                branch_manifest_path, arg_name="branch-manifest-json"
            )
        )
        controller_ok = bool(
            controller_ok
            and manifest_payload.get("artifact_kind")
            == "gr00t_new_embodiment_branch_manifest"
            and manifest_payload.get("formal_branch_eligibility") == "ALLOW"
        )
    blockers = [] if controller_ok else ["new_embodiment_branch_contract_blocked"]
    return {
        "branch": branch,
        "controller_audit_path": str(
            _resolve_existing_file(
                controller_audit_path, arg_name="controller-audit-json"
            )
        ),
        "branch_manifest_path": manifest_resolved,
        "branch_scope": BRANCH_SCOPE_BY_BRANCH[branch],
        "public_anchor_comparable": False,
        "branch_prerequisite_ok": controller_ok,
        "branch_prerequisite_code": "new_embodiment_branch_contract_allow",
        "branch_blockers": blockers,
    }


def load_replay_upper_bound_summary(
    branch: str,
    *,
    teacher_summary: Mapping[str, Any],
    replay_upper_bound_path: Path | None,
) -> dict[str, Any]:
    families = [
        dict(_as_mapping(item, field_name="teacher_summary.families[]"))
        for item in _as_list(
            teacher_summary.get("families", []), field_name="teacher_summary.families"
        )
    ]
    if replay_upper_bound_path is None:
        successful_families = set(DEFAULT_REPLAY_SUCCESS_FAMILIES_BY_BRANCH[branch])
        scene_results = []
        for row in families:
            family = _as_non_empty_string(
                row.get("family"), field_name="teacher_summary.family"
            )
            scene_id = _scene_id(branch, family)
            success = bool(
                row.get("teacher_reachable", False) and family in successful_families
            )
            scene_results.append(
                {
                    "scene_id": scene_id,
                    "family": family,
                    "success": success,
                    "source_kind": "repo_local_replay_upper_bound_fixture",
                }
            )
        success_count = sum(1 for row in scene_results if row["success"])
        return {
            "artifact_kind": REPLAY_UPPER_BOUND_ARTIFACT_KIND,
            "source_path": None,
            "source_kind": "repo_local_replay_upper_bound_fixture",
            "policy_role": "stack_integration_debug_upper_bound",
            "attempt_count": len(scene_results),
            "success_count": int(success_count),
            "success_rate": (float(success_count) / float(len(scene_results)))
            if scene_results
            else 0.0,
            "scene_results": scene_results,
        }

    payload = _read_json(replay_upper_bound_path, arg_name="replay-upper-bound-json")
    if payload.get("artifact_kind") not in {None, REPLAY_UPPER_BOUND_ARTIFACT_KIND}:
        raise ValueError(
            "replay-upper-bound-json artifact_kind mismatch; expected "
            f"{REPLAY_UPPER_BOUND_ARTIFACT_KIND}"
        )
    scene_results = []
    for index, row in enumerate(
        _as_list(
            payload.get("scene_results", []),
            field_name="replay-upper-bound-json.scene_results",
        )
    ):
        row_obj = _as_mapping(
            row,
            field_name=f"replay-upper-bound-json.scene_results[{index}]",
        )
        scene_results.append(
            {
                "scene_id": _as_non_empty_string(
                    row_obj.get("scene_id"),
                    field_name=(
                        f"replay-upper-bound-json.scene_results[{index}].scene_id"
                    ),
                ),
                "family": _as_non_empty_string(
                    row_obj.get("family"),
                    field_name=(
                        f"replay-upper-bound-json.scene_results[{index}].family"
                    ),
                ),
                "success": _as_bool(
                    row_obj.get("success"),
                    field_name=(
                        f"replay-upper-bound-json.scene_results[{index}].success"
                    ),
                ),
                "source_kind": row_obj.get("source_kind"),
            }
        )
    success_count = int(
        payload.get(
            "success_count",
            sum(1 for row in scene_results if bool(row.get("success", False))),
        )
    )
    attempt_count = int(payload.get("attempt_count", len(scene_results)))
    success_rate = float(
        payload.get(
            "success_rate",
            (float(success_count) / float(attempt_count)) if attempt_count > 0 else 0.0,
        )
    )
    return {
        "artifact_kind": str(
            payload.get("artifact_kind") or REPLAY_UPPER_BOUND_ARTIFACT_KIND
        ),
        "source_path": str(
            _resolve_existing_file(
                replay_upper_bound_path, arg_name="replay-upper-bound-json"
            )
        ),
        "source_kind": payload.get("source_kind", "custom_replay_upper_bound_json"),
        "policy_role": payload.get(
            "policy_role", "stack_integration_debug_upper_bound"
        ),
        "attempt_count": int(attempt_count),
        "success_count": int(success_count),
        "success_rate": float(success_rate),
        "scene_results": scene_results,
    }


def _teacher_case(
    *,
    teacher_success_count: int,
    teacher_reachable_scene_ids: list[str],
    snapshot_baseline_success_count: int,
) -> tuple[str, str]:
    if teacher_success_count <= 0:
        return (
            "teacher_all_zero_block",
            "Teacher upper bound stayed at zero across the selected scene families, so the branch remains teacher-unreachable and strong attribution must stop.",
        )
    if not teacher_reachable_scene_ids:
        return (
            "teacher_no_reachable_scene_pool_block",
            "Teacher family summaries contain no reachable formal scene pool, so later ladders must not attribute failure to the student.",
        )
    if snapshot_baseline_success_count <= 0:
        return (
            "teacher_reachable_student_currently_zero",
            "Teacher can recover at least part of the formal scene pool while the current snapshot baseline is still zero, which points first to student-not-learned rather than teacher-unreachable.",
        )
    return (
        "teacher_reachable_student_nonzero",
        "Teacher and current snapshot baseline are both non-zero on the selected formal scene families.",
    )


def _replay_case(
    *,
    replay_success_count: int,
    public_anchor_success_count: int,
    snapshot_baseline_success_count: int,
    public_anchor_comparable: bool,
) -> tuple[str, str]:
    if replay_success_count <= 0:
        if public_anchor_comparable and public_anchor_success_count <= 0:
            return (
                "replay_all_zero_and_public_anchor_zero_stack_or_env_risk",
                "Replay upper bound and the comparable public anchor are both zero, so stack/controller/env risk dominates and formal attribution must stop.",
            )
        return (
            "replay_all_zero_stack_or_env_risk",
            "Replay upper bound is zero, so the environment/controller stack is not healthy enough for strong attribution.",
        )
    if (
        public_anchor_comparable
        and public_anchor_success_count > 0
        and snapshot_baseline_success_count <= 0
    ):
        return (
            "replay_high_public_anchor_nonzero_student_zero_training_or_data_issue",
            "Replay is healthy and the public anchor is non-zero while the snapshot baseline stays zero, so the dominant explanation is training/data rather than stack reachability.",
        )
    if (not public_anchor_comparable) and snapshot_baseline_success_count <= 0:
        return (
            "replay_high_branch_local_stack_healthy_student_zero",
            "Replay is healthy on the branch-local line while the current snapshot baseline remains zero, so the dominant explanation is branch-local student learning rather than controller wiring.",
        )
    if snapshot_baseline_success_count > 0:
        return (
            "replay_high_and_student_nonzero",
            "Replay is healthy and the current student baseline is already non-zero on at least part of the selected formal surface.",
        )
    return (
        "replay_nonzero_mixed_signal",
        "Replay upper bound is non-zero, but the remaining evidence is mixed and should be interpreted together with teacher reachability.",
    )


def build_teacher_reachability_gate_payload(
    *,
    branch: str,
    teacher_summary: Mapping[str, Any],
    current_baseline: Mapping[str, Any],
    replay_upper_bound: Mapping[str, Any],
    branch_context: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    branch_slug = str(branch).lower()
    families = [
        dict(_as_mapping(item, field_name="teacher_summary.families[]"))
        for item in _as_list(
            teacher_summary.get("families", []), field_name="teacher_summary.families"
        )
    ]
    replay_scene_results = [
        dict(_as_mapping(item, field_name="replay_upper_bound.scene_results[]"))
        for item in _as_list(
            replay_upper_bound.get("scene_results", []),
            field_name="replay_upper_bound.scene_results",
        )
    ]
    replay_success_by_scene = {
        _as_non_empty_string(row.get("scene_id"), field_name="replay scene_id"): bool(
            row.get("success", False)
        )
        for row in replay_scene_results
    }

    teacher_reachable_scene_ids = sorted(
        _scene_id(branch, _as_non_empty_string(row.get("family"), field_name="family"))
        for row in families
        if bool(row.get("teacher_reachable", False))
    )
    replay_reachable_scene_ids = sorted(
        scene_id for scene_id, success in replay_success_by_scene.items() if success
    )
    reachable_scene_ids = sorted(
        set(teacher_reachable_scene_ids).intersection(replay_reachable_scene_ids)
    )

    teacher_success_count = _as_int(
        teacher_summary.get("teacher_success_count", 0),
        field_name="teacher_summary.teacher_success_count",
    )
    snapshot_baseline_success_count = _as_int(
        teacher_summary.get("snapshot_baseline_success_count", 0),
        field_name="teacher_summary.snapshot_baseline_success_count",
    )
    public_anchor_success_count = _as_int(
        current_baseline.get("success_count", 0),
        field_name="current_baseline.success_count",
    )
    replay_success_count = _as_int(
        replay_upper_bound.get("success_count", 0),
        field_name="replay_upper_bound.success_count",
    )
    public_anchor_comparable = bool(
        branch_context.get("public_anchor_comparable", False)
    )

    teacher_case_code, teacher_case = _teacher_case(
        teacher_success_count=teacher_success_count,
        teacher_reachable_scene_ids=teacher_reachable_scene_ids,
        snapshot_baseline_success_count=snapshot_baseline_success_count,
    )
    replay_case_code, replay_case = _replay_case(
        replay_success_count=replay_success_count,
        public_anchor_success_count=public_anchor_success_count,
        snapshot_baseline_success_count=snapshot_baseline_success_count,
        public_anchor_comparable=public_anchor_comparable,
    )

    blocking_reasons: list[str] = []
    if not bool(branch_context.get("branch_prerequisite_ok", False)):
        blocking_reasons.extend(
            str(item)
            for item in _as_list(
                branch_context.get("branch_blockers", []),
                field_name="branch_context.branch_blockers",
            )
        )
        scene_pool_status = "blocked_branch_prerequisite"
    elif teacher_success_count <= 0:
        blocking_reasons.append("teacher_all_zero")
        scene_pool_status = "blocked_teacher_all_zero"
    elif not teacher_reachable_scene_ids:
        blocking_reasons.append("teacher_reachable_scene_pool_empty")
        scene_pool_status = "blocked_no_teacher_reachable_scenes"
    elif replay_success_count <= 0:
        blocking_reasons.append("replay_upper_bound_all_zero")
        scene_pool_status = "blocked_replay_all_zero"
    elif not reachable_scene_ids:
        blocking_reasons.append("teacher_replay_scene_intersection_empty")
        scene_pool_status = "blocked_no_teacher_replay_scene_intersection"
    else:
        scene_pool_status = "formal_teacher_replay_reachable_pool_materialized"

    allow_formal_ladders = not blocking_reasons
    status = "ALLOW" if allow_formal_ladders else "BLOCK"
    reason_code = scene_pool_status
    reason = (
        "Formal ladders may proceed only on the intersection of teacher-reachable and "
        "replay-healthy scene IDs."
        if allow_formal_ladders
        else "Formal ladders are blocked until teacher reachability and replay upper-bound evidence define a non-empty reachable scene pool."
    )

    family_scene_rows: list[dict[str, Any]] = []
    for row in families:
        family = _as_non_empty_string(row.get("family"), field_name="family")
        scene_id = _scene_id(branch, family)
        family_scene_rows.append(
            {
                "scene_id": scene_id,
                "family": family,
                "priority": row.get("priority"),
                "teacher_reachable": bool(row.get("teacher_reachable", False)),
                "teacher_success_count": int(row.get("teacher_success_count", 0)),
                "teacher_interpretation_code": row.get("teacher_interpretation_code"),
                "snapshot_baseline_success_count": int(
                    row.get("snapshot_baseline_success_count", 0)
                ),
                "replay_success": bool(replay_success_by_scene.get(scene_id, False)),
                "included_in_formal_scene_pool": scene_id in reachable_scene_ids,
            }
        )

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "branch": branch,
        "output_path": str(output_path),
        "branch_scope": branch_context.get("branch_scope"),
        "public_anchor_comparable": bool(public_anchor_comparable),
        "scene_pool_status": str(scene_pool_status),
        "teacher_case_code": str(teacher_case_code),
        "teacher_case": str(teacher_case),
        "replay_case_code": str(replay_case_code),
        "replay_case": str(replay_case),
        "allow_formal_ladders": bool(allow_formal_ladders),
        "reachable_scene_ids": list(reachable_scene_ids),
        "teacher_reachable_scene_ids": list(teacher_reachable_scene_ids),
        "replay_reachable_scene_ids": list(replay_reachable_scene_ids),
        "blocking_reasons": list(blocking_reasons),
        "status": status,
        "reason_code": str(reason_code),
        "reason": str(reason),
        "decision_basis": (
            "BLOCK when branch prerequisites fail, teacher is all-zero, replay is all-zero, "
            "or the teacher∩replay formal scene pool is empty. Otherwise ALLOW later ladders "
            "to operate only on reachable_scene_ids."
        ),
        "branch_prerequisites": {
            "branch_prerequisite_ok": bool(
                branch_context.get("branch_prerequisite_ok", False)
            ),
            "branch_prerequisite_code": branch_context.get("branch_prerequisite_code"),
            "branch_blockers": list(
                _as_list(
                    branch_context.get("branch_blockers", []),
                    field_name="branch_context.branch_blockers",
                )
            ),
            "controller_audit_path": branch_context.get("controller_audit_path"),
            "branch_manifest_path": branch_context.get("branch_manifest_path"),
        },
        "teacher_upper_bound": {
            "source_path": teacher_summary.get("source_path"),
            "artifact_kind": teacher_summary.get("artifact_kind"),
            "teacher_success_count": int(teacher_success_count),
            "snapshot_baseline_success_count": int(snapshot_baseline_success_count),
            "teacher_reachable_rate": float(
                teacher_summary.get("teacher_reachable_rate", 0.0)
            ),
        },
        "current_baseline": {
            "source_kind": current_baseline.get("source_kind"),
            "source_path": current_baseline.get("source_path"),
            "artifact_kind": current_baseline.get("artifact_kind"),
            "success_count": int(public_anchor_success_count),
            "success_rate": float(current_baseline.get("success_rate", 0.0)),
        },
        "replay_upper_bound": {
            "source_kind": replay_upper_bound.get("source_kind"),
            "source_path": replay_upper_bound.get("source_path"),
            "artifact_kind": replay_upper_bound.get("artifact_kind"),
            "policy_role": replay_upper_bound.get("policy_role"),
            "attempt_count": int(replay_upper_bound.get("attempt_count", 0)),
            "success_count": int(replay_success_count),
            "success_rate": float(replay_upper_bound.get("success_rate", 0.0)),
        },
        "scene_pool": {
            "total_scene_count": len(family_scene_rows),
            "teacher_reachable_scene_count": len(teacher_reachable_scene_ids),
            "replay_reachable_scene_count": len(replay_reachable_scene_ids),
            "formal_reachable_scene_count": len(reachable_scene_ids),
            "scene_rows": family_scene_rows,
        },
    }
    payload["report_signature_sha256"] = _sha256(payload)
    return payload


def _build_failure_note(report: Mapping[str, Any]) -> str:
    lines = [
        "# Teacher reachability gate failure note",
        "",
        f"- branch: `{report.get('branch')}`",
        f"- output_path: `{report.get('output_path')}`",
        f"- scene_pool_status: `{report.get('scene_pool_status')}`",
        f"- teacher_case_code: `{report.get('teacher_case_code')}`",
        f"- replay_case_code: `{report.get('replay_case_code')}`",
        f"- allow_formal_ladders: `{report.get('allow_formal_ladders')}`",
        f"- reachable_scene_ids: `{json.dumps(report.get('reachable_scene_ids', []), ensure_ascii=True)}`",
        "",
        "## Blocking reasons",
        "",
        "```json",
        json.dumps(report.get("blocking_reasons", []), ensure_ascii=True, indent=2),
        "```",
        "",
        "## Why this blocks",
        "",
        str(report.get("reason", "")),
        "",
    ]
    return "\n".join(lines)


def _write_failure_note(path: Path, report: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_build_failure_note(report), encoding="utf-8")
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        branch = str(args.branch)
        defaults = _branch_defaults(branch)
        resolved_output_dir = _validate_output_dir(Path(args.output_dir))
        teacher_gate_path = Path(args.teacher_gate_json)
        current_baseline_path = (
            Path(args.current_baseline_json)
            if args.current_baseline_json is not None
            else defaults["current_baseline_json"]
        )
        replay_upper_bound_path = (
            Path(args.replay_upper_bound_json)
            if args.replay_upper_bound_json is not None
            else None
        )
        controller_audit_path = (
            Path(args.controller_audit_json)
            if args.controller_audit_json is not None
            else defaults["controller_audit_json"]
        )
        branch_manifest_path = (
            Path(args.branch_manifest_json)
            if args.branch_manifest_json is not None
            else defaults["branch_manifest_json"]
        )
        if controller_audit_path is None:
            raise ValueError("controller audit path could not be resolved")

        teacher_summary = load_teacher_gate_summary(teacher_gate_path)
        teacher_summary["families"] = [
            {
                **dict(_as_mapping(item, field_name="teacher_summary.families[]")),
                "scene_id": _scene_id(
                    branch,
                    _as_non_empty_string(
                        _as_mapping(item, field_name="teacher_summary.families[]").get(
                            "family"
                        ),
                        field_name="teacher_summary.family",
                    ),
                ),
            }
            for item in _as_list(
                teacher_summary.get("families", []),
                field_name="teacher_summary.families",
            )
        ]
        current_baseline = load_current_baseline_summary(branch, current_baseline_path)
        replay_upper_bound = load_replay_upper_bound_summary(
            branch,
            teacher_summary=teacher_summary,
            replay_upper_bound_path=replay_upper_bound_path,
        )
        branch_context = load_branch_context(
            branch,
            controller_audit_path=Path(controller_audit_path),
            branch_manifest_path=branch_manifest_path,
        )

        output_path = resolved_output_dir / REPORT_JSON_NAME_BY_BRANCH[branch]
        report = build_teacher_reachability_gate_payload(
            branch=branch,
            teacher_summary=teacher_summary,
            current_baseline=current_baseline,
            replay_upper_bound=replay_upper_bound,
            branch_context=branch_context,
            output_path=output_path,
        )
        _write_json(output_path, report)

        failure_note_path = (
            resolved_output_dir / FAILURE_NOTE_MARKDOWN_NAME_BY_BRANCH[branch]
        )
        if bool(report["allow_formal_ladders"]):
            if failure_note_path.exists():
                failure_note_path.unlink()
            report["failure_note_path"] = None
            _write_json(output_path, report)
        else:
            written_failure_note = _write_failure_note(failure_note_path, report)
            report["failure_note_path"] = str(written_failure_note)
            _write_json(output_path, report)

        sys.stdout.write(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
        )
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        sys.stderr.write(
            f"teacher reachability gate failed: {_exception_message(exc)}\n"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
