from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

REPORT_SCHEMA_VERSION = "gr00t_teacher_student_gap_scorecard_v1"
REPORT_ARTIFACT_KIND = "gr00t_teacher_student_gap_scorecard"

BRANCH_UNITREE_G1 = "UNITREE_G1"
BRANCH_NEW_EMBODIMENT = "NEW_EMBODIMENT"
ALL_BRANCHES = (BRANCH_UNITREE_G1, BRANCH_NEW_EMBODIMENT)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import gr00t_action_chain_telemetry
from work.recap import gr00t_teacher_reachability_gate
from work.recap import state_conditioned_bucket_a_import


DEFAULT_OUTPUTS: dict[str, Path] = {
    BRANCH_UNITREE_G1: REPO_ROOT
    / "agent"
    / "artifacts"
    / "gr00t_anchor_controller_recap"
    / "unitree_g1"
    / "teacher_student_gap_scorecard_unitree_g1.json",
    BRANCH_NEW_EMBODIMENT: REPO_ROOT
    / "agent"
    / "artifacts"
    / "gr00t_anchor_controller_recap"
    / "new_embodiment"
    / "teacher_student_gap_scorecard_new_embodiment.json",
}

FAILURE_NOTE_MARKDOWN_NAME_BY_BRANCH = {
    BRANCH_UNITREE_G1: "teacher_student_gap_scorecard_unitree_g1_failure_note.md",
    BRANCH_NEW_EMBODIMENT: "teacher_student_gap_scorecard_new_embodiment_failure_note.md",
}

DEFAULT_TEACHER_REACHABILITY_JSON_BY_BRANCH: dict[str, Path] = {
    BRANCH_UNITREE_G1: REPO_ROOT
    / "agent"
    / "artifacts"
    / "gr00t_anchor_controller_recap"
    / "teacher_reachability"
    / "teacher_reachability_gate_unitree_g1.json",
    BRANCH_NEW_EMBODIMENT: REPO_ROOT
    / "agent"
    / "artifacts"
    / "gr00t_anchor_controller_recap"
    / "teacher_reachability"
    / "teacher_reachability_gate_new_embodiment.json",
}

DEFAULT_ACTION_TELEMETRY_JSON_BY_BRANCH: dict[str, Path] = {
    BRANCH_UNITREE_G1: REPO_ROOT
    / "agent"
    / "artifacts"
    / "gr00t_anchor_controller_recap"
    / "unitree_g1"
    / "action_chain_telemetry_unitree_g1.json",
    BRANCH_NEW_EMBODIMENT: REPO_ROOT
    / "agent"
    / "artifacts"
    / "gr00t_anchor_controller_recap"
    / "new_embodiment"
    / "action_chain_telemetry_new_embodiment.json",
}

DEFAULT_OPEN_LOOP_AGREEMENT_JSON = (
    REPO_ROOT
    / "agent"
    / "artifacts"
    / "state_conditioned_materialization"
    / "sanity"
    / "open_loop_agreement_report.json"
)

FORMAL_BRANCH_MATCH_THRESHOLD = 0.60
ZERO_MOTION_SCORE_CAP = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_teacher_student_gap.py",
        description=(
            "Combine task-10 reachable-scene semantics, task-7 action-chain telemetry, "
            "and the state-conditioned open-loop agreement report into a teacher-vs-"
            "student conditional gap scorecard."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--branch",
        required=True,
        choices=list(ALL_BRANCHES),
        help="Branch whose teacher-student conditional gap scorecard should be materialized.",
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to branch-specific teacher_student_gap_scorecard_*.json.",
    )
    _ = parser.add_argument(
        "--teacher-reachability-json",
        type=Path,
        default=None,
        help="Optional task-10 teacher reachability gate JSON. Defaults to the accepted branch artifact.",
    )
    _ = parser.add_argument(
        "--action-telemetry-json",
        type=Path,
        default=None,
        help="Optional task-7 action telemetry JSON. Defaults to the accepted branch artifact.",
    )
    _ = parser.add_argument(
        "--open-loop-agreement-json",
        type=Path,
        default=DEFAULT_OPEN_LOOP_AGREEMENT_JSON,
        help="Open-loop agreement report used as shared student open-loop evidence.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(
            f"output must be a file path, got existing directory: {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _rel_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


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


def _round_float(value: float, *, digits: int = 8) -> float:
    return float(round(float(value), digits))


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _resolve_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{arg_name} does not exist: {resolved}")
    return resolved


def _read_json(path: Path, *, arg_name: str) -> dict[str, Any]:
    payload = json.loads(
        _resolve_existing_file(path, arg_name=arg_name).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{arg_name} must contain a JSON object")
    return dict(payload)


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


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
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_number(value: object, *, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def default_output_path_for_branch(branch: str) -> Path:
    if branch not in DEFAULT_OUTPUTS:
        raise KeyError(f"unsupported branch for default output: {branch}")
    return DEFAULT_OUTPUTS[branch]


def _default_teacher_reachability_path(branch: str) -> Path:
    if branch not in DEFAULT_TEACHER_REACHABILITY_JSON_BY_BRANCH:
        raise KeyError(
            f"unsupported branch for default teacher reachability input: {branch}"
        )
    return DEFAULT_TEACHER_REACHABILITY_JSON_BY_BRANCH[branch]


def _default_action_telemetry_path(branch: str) -> Path:
    if branch not in DEFAULT_ACTION_TELEMETRY_JSON_BY_BRANCH:
        raise KeyError(
            f"unsupported branch for default action telemetry input: {branch}"
        )
    return DEFAULT_ACTION_TELEMETRY_JSON_BY_BRANCH[branch]


def load_teacher_reachability_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path, arg_name="teacher-reachability-json")
    artifact_kind = payload.get("artifact_kind")
    if artifact_kind not in {
        None,
        gr00t_teacher_reachability_gate.REPORT_ARTIFACT_KIND,
    }:
        raise ValueError(
            "teacher-reachability-json artifact_kind mismatch; expected "
            f"{gr00t_teacher_reachability_gate.REPORT_ARTIFACT_KIND}"
        )

    scene_pool = _as_mapping(
        payload.get("scene_pool", {}), field_name="teacher-reachability-json.scene_pool"
    )
    scene_rows = [
        dict(
            _as_mapping(
                item, field_name="teacher-reachability-json.scene_pool.scene_rows[]"
            )
        )
        for item in _as_list(
            scene_pool.get("scene_rows", []),
            field_name="teacher-reachability-json.scene_pool.scene_rows",
        )
    ]
    return {
        "source_path": str(
            _resolve_existing_file(path, arg_name="teacher-reachability-json")
        ),
        "artifact_kind": str(
            artifact_kind or gr00t_teacher_reachability_gate.REPORT_ARTIFACT_KIND
        ),
        "branch": payload.get("branch"),
        "branch_scope": payload.get("branch_scope"),
        "public_anchor_comparable": bool(
            payload.get("public_anchor_comparable", False)
        ),
        "status": payload.get("status"),
        "allow_formal_ladders": bool(payload.get("allow_formal_ladders", False)),
        "scene_pool_status": payload.get("scene_pool_status"),
        "teacher_case_code": payload.get("teacher_case_code"),
        "teacher_case": payload.get("teacher_case"),
        "replay_case_code": payload.get("replay_case_code"),
        "replay_case": payload.get("replay_case"),
        "reachable_scene_ids": [
            _as_non_empty_string(
                item, field_name="teacher-reachability-json.reachable_scene_ids[]"
            )
            for item in _as_list(
                payload.get("reachable_scene_ids", []),
                field_name="teacher-reachability-json.reachable_scene_ids",
            )
        ],
        "teacher_reachable_scene_ids": [
            _as_non_empty_string(
                item,
                field_name="teacher-reachability-json.teacher_reachable_scene_ids[]",
            )
            for item in _as_list(
                payload.get("teacher_reachable_scene_ids", []),
                field_name="teacher-reachability-json.teacher_reachable_scene_ids",
            )
        ],
        "replay_reachable_scene_ids": [
            _as_non_empty_string(
                item,
                field_name="teacher-reachability-json.replay_reachable_scene_ids[]",
            )
            for item in _as_list(
                payload.get("replay_reachable_scene_ids", []),
                field_name="teacher-reachability-json.replay_reachable_scene_ids",
            )
        ],
        "blocking_reasons": [
            str(item)
            for item in _as_list(
                payload.get("blocking_reasons", []),
                field_name="teacher-reachability-json.blocking_reasons",
            )
        ],
        "scene_rows": scene_rows,
        "teacher_upper_bound": dict(
            _as_mapping(
                payload.get("teacher_upper_bound", {}),
                field_name="teacher-reachability-json.teacher_upper_bound",
            )
        ),
        "branch_prerequisites": dict(
            _as_mapping(
                payload.get("branch_prerequisites", {}),
                field_name="teacher-reachability-json.branch_prerequisites",
            )
        ),
        "report_signature_sha256": payload.get("report_signature_sha256"),
    }


def load_action_telemetry_summary(branch: str, path: Path) -> dict[str, Any]:
    payload = _read_json(path, arg_name="action-telemetry-json")
    artifact_kind = payload.get("artifact_kind")
    if artifact_kind not in {None, gr00t_action_chain_telemetry.REPORT_ARTIFACT_KIND}:
        raise ValueError(
            "action-telemetry-json artifact_kind mismatch; expected "
            f"{gr00t_action_chain_telemetry.REPORT_ARTIFACT_KIND}"
        )
    payload_branch = _as_non_empty_string(
        payload.get("branch"), field_name="action-telemetry-json.branch"
    )
    if payload_branch != branch:
        raise ValueError(
            f"action-telemetry-json branch mismatch: expected {branch}, got {payload_branch}"
        )
    per_group_stats = {
        str(key): dict(
            _as_mapping(
                value, field_name=f"action-telemetry-json.per_group_stats.{key}"
            )
        )
        for key, value in _as_mapping(
            payload.get("per_group_stats", {}),
            field_name="action-telemetry-json.per_group_stats",
        ).items()
    }
    return {
        "source_path": str(
            _resolve_existing_file(path, arg_name="action-telemetry-json")
        ),
        "artifact_kind": str(
            artifact_kind or gr00t_action_chain_telemetry.REPORT_ARTIFACT_KIND
        ),
        "branch": payload_branch,
        "public_anchor_comparable": bool(
            payload.get("public_anchor_comparable", False)
        ),
        "action_order": [
            _as_non_empty_string(
                item, field_name="action-telemetry-json.action_order[]"
            )
            for item in _as_list(
                payload.get("action_order", []),
                field_name="action-telemetry-json.action_order",
            )
        ],
        "per_group_stats": per_group_stats,
        "controller_absorbed_groups": [
            _as_non_empty_string(
                item, field_name="action-telemetry-json.controller_absorbed_groups[]"
            )
            for item in _as_list(
                payload.get("controller_absorbed_groups", []),
                field_name="action-telemetry-json.controller_absorbed_groups",
            )
        ],
        "model_insensitive_groups": [
            _as_non_empty_string(
                item, field_name="action-telemetry-json.model_insensitive_groups[]"
            )
            for item in _as_list(
                payload.get("model_insensitive_groups", []),
                field_name="action-telemetry-json.model_insensitive_groups",
            )
        ],
        "clip_rate": dict(
            _as_mapping(
                payload.get("clip_rate", {}),
                field_name="action-telemetry-json.clip_rate",
            )
        ),
        "saturation_rate": dict(
            _as_mapping(
                payload.get("saturation_rate", {}),
                field_name="action-telemetry-json.saturation_rate",
            )
        ),
    }


def load_open_loop_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path, arg_name="open-loop-agreement-json")
    artifact_kind = payload.get("artifact_kind")
    if artifact_kind not in {None, "state_conditioned_open_loop_agreement_report"}:
        raise ValueError(
            "open-loop-agreement-json artifact_kind mismatch; expected state_conditioned_open_loop_agreement_report"
        )
    checks = {
        str(key): dict(
            _as_mapping(value, field_name=f"open-loop-agreement-json.checks.{key}")
        )
        for key, value in _as_mapping(
            payload.get("checks", {}), field_name="open-loop-agreement-json.checks"
        ).items()
    }
    total_check_count = max(len(checks), 1)
    passed_check_count = sum(
        1
        for value in checks.values()
        if bool(value.get("passed", False)) or str(value.get("status")) == "PASS"
    )
    history_condition_response = dict(
        _as_mapping(
            checks.get("history_condition_response", {}),
            field_name="open-loop-agreement-json.checks.history_condition_response",
        )
    )
    valid_mask_effectiveness = dict(
        _as_mapping(
            checks.get("valid_mask_effectiveness", {}),
            field_name="open-loop-agreement-json.checks.valid_mask_effectiveness",
        )
    )
    action_range = dict(
        _as_mapping(
            checks.get("action_range", {}),
            field_name="open-loop-agreement-json.checks.action_range",
        )
    )
    return {
        "source_path": str(
            _resolve_existing_file(path, arg_name="open-loop-agreement-json")
        ),
        "artifact_kind": str(
            artifact_kind or "state_conditioned_open_loop_agreement_report"
        ),
        "status": str(payload.get("status", "UNKNOWN")),
        "passed_check_count": int(passed_check_count),
        "total_check_count": int(total_check_count),
        "check_pass_rate": _round_float(
            float(passed_check_count) / float(total_check_count)
        ),
        "history_condition_response": history_condition_response,
        "valid_mask_effectiveness": valid_mask_effectiveness,
        "action_range": action_range,
        "telemetry": dict(
            _as_mapping(
                payload.get("telemetry", {}),
                field_name="open-loop-agreement-json.telemetry",
            )
        ),
        "summary": dict(
            _as_mapping(
                payload.get("summary", {}),
                field_name="open-loop-agreement-json.summary",
            )
        ),
        "failure": payload.get("failure"),
        "checks": checks,
    }


def _group_case_code(
    *,
    model_insensitive: bool,
    controller_absorbed: bool,
    all_zero_in_both: bool,
    saturation_rate: float,
    controller_clip_rate: float,
) -> str:
    if all_zero_in_both:
        return "student_zero_motion_group"
    if model_insensitive:
        return "student_model_insensitive_group"
    if controller_absorbed:
        return "student_controller_absorbed_group"
    if saturation_rate > 0.0:
        return "student_saturated_group"
    if controller_clip_rate > 0.0:
        return "student_clipped_group"
    return "student_branch_matched_group"


def build_per_action_group_gap(
    telemetry_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    per_group_stats = {
        str(key): dict(
            _as_mapping(value, field_name=f"telemetry.per_group_stats.{key}")
        )
        for key, value in _as_mapping(
            telemetry_summary.get("per_group_stats", {}),
            field_name="telemetry.per_group_stats",
        ).items()
    }
    action_order = [
        _as_non_empty_string(item, field_name="telemetry.action_order[]")
        for item in _as_list(
            telemetry_summary.get("action_order", []),
            field_name="telemetry.action_order",
        )
    ]
    if not action_order:
        raise ValueError("action telemetry report is missing action_order")

    per_action_group_gap: dict[str, Any] = {}
    group_scores: list[float] = []
    for key in action_order:
        group_payload = dict(
            _as_mapping(
                per_group_stats[key], field_name=f"telemetry.per_group_stats.{key}"
            )
        )
        difference_metrics = dict(
            _as_mapping(
                group_payload.get("difference_metrics", {}),
                field_name=f"telemetry.per_group_stats.{key}.difference_metrics",
            )
        )
        zero_motion_flags = dict(
            _as_mapping(
                group_payload.get("zero_motion_flags", {}),
                field_name=f"telemetry.per_group_stats.{key}.zero_motion_flags",
            )
        )
        clip_rate = dict(
            _as_mapping(
                group_payload.get("clip_rate", {}),
                field_name=f"telemetry.per_group_stats.{key}.clip_rate",
            )
        )

        model_insensitive = bool(difference_metrics.get("model_insensitive", False))
        controller_absorbed = bool(
            difference_metrics.get("controller_absorbed_upstream_difference", False)
        )
        all_zero_in_both = bool(zero_motion_flags.get("all_zero_in_both", False))
        saturation_rate = _as_number(
            group_payload.get("saturation_rate", 0.0),
            field_name=f"telemetry.per_group_stats.{key}.saturation_rate",
        )
        controller_clip_rate = _as_number(
            clip_rate.get("controller_input", 0.0),
            field_name=f"telemetry.per_group_stats.{key}.clip_rate.controller_input",
        )
        decoded_clip_rate = _as_number(
            clip_rate.get("decoded_action", 0.0),
            field_name=f"telemetry.per_group_stats.{key}.clip_rate.decoded_action",
        )

        score = 1.0
        if model_insensitive:
            score -= 0.70
        if controller_absorbed:
            score -= 0.35
        score -= min(0.20, float(saturation_rate))
        score -= min(0.10, float(controller_clip_rate))
        if all_zero_in_both:
            score = min(score, ZERO_MOTION_SCORE_CAP)
        score = _clamp(score)
        gap_score = _round_float(1.0 - score)
        group_scores.append(score)

        per_action_group_gap[key] = {
            "group": key,
            "telemetry_case_code": _group_case_code(
                model_insensitive=model_insensitive,
                controller_absorbed=controller_absorbed,
                all_zero_in_both=all_zero_in_both,
                saturation_rate=float(saturation_rate),
                controller_clip_rate=float(controller_clip_rate),
            ),
            "branch_match_score": _round_float(score),
            "gap_score": gap_score,
            "difference_disappeared_at": difference_metrics.get(
                "difference_disappeared_at"
            ),
            "model_insensitive": model_insensitive,
            "controller_absorbed_upstream_difference": controller_absorbed,
            "raw_action_l2": _round_float(
                _as_number(
                    difference_metrics.get("raw_action_l2", 0.0),
                    field_name=f"telemetry.per_group_stats.{key}.difference_metrics.raw_action_l2",
                )
            ),
            "decoded_action_l2": _round_float(
                _as_number(
                    difference_metrics.get("decoded_action_l2", 0.0),
                    field_name=f"telemetry.per_group_stats.{key}.difference_metrics.decoded_action_l2",
                )
            ),
            "absolute_action_l2": _round_float(
                _as_number(
                    difference_metrics.get("absolute_action_l2", 0.0),
                    field_name=f"telemetry.per_group_stats.{key}.difference_metrics.absolute_action_l2",
                )
            ),
            "controller_input_l2": _round_float(
                _as_number(
                    difference_metrics.get("controller_input_l2", 0.0),
                    field_name=f"telemetry.per_group_stats.{key}.difference_metrics.controller_input_l2",
                )
            ),
            "decoded_clip_rate": _round_float(decoded_clip_rate),
            "controller_clip_rate": _round_float(controller_clip_rate),
            "saturation_rate": _round_float(saturation_rate),
            "zero_motion_flags": zero_motion_flags,
        }

    telemetry_group_match_rate = _round_float(
        float(sum(group_scores)) / float(len(group_scores)) if group_scores else 0.0
    )
    return per_action_group_gap, telemetry_group_match_rate


def _formal_family_case_code(
    *,
    teacher_reachable: bool,
    included_in_formal_scene_pool: bool,
    student_success_count: int,
    teacher_success_count: int,
    student_branch_match_rate: float,
) -> str:
    if not teacher_reachable:
        return "teacher_unreachable_excluded_from_student_gap"
    if not included_in_formal_scene_pool:
        return "teacher_reachable_scene_pool_excluded"
    if student_success_count <= 0:
        if student_branch_match_rate >= FORMAL_BRANCH_MATCH_THRESHOLD:
            return "teacher_reachable_student_zero_branch_consistent"
        return "teacher_reachable_student_zero_branch_inconsistent"
    if student_success_count < teacher_success_count:
        if student_branch_match_rate >= FORMAL_BRANCH_MATCH_THRESHOLD:
            return "teacher_reachable_student_partial_gap_branch_consistent"
        return "teacher_reachable_student_partial_gap_branch_inconsistent"
    return "teacher_reachable_student_matched"


def build_per_family_gap(
    *,
    teacher_summary: Mapping[str, Any],
    student_branch_match_rate: float,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    scene_rows = [
        dict(_as_mapping(item, field_name="teacher-summary.scene_rows[]"))
        for item in _as_list(
            teacher_summary.get("scene_rows", []),
            field_name="teacher-summary.scene_rows",
        )
    ]
    reachable_families: list[str] = []
    teacher_unreachable_families: list[str] = []
    scene_pool_excluded_families: list[str] = []
    per_family_gap: list[dict[str, Any]] = []
    for row in scene_rows:
        family = _as_non_empty_string(
            row.get("family"), field_name="teacher-summary.scene_rows[].family"
        )
        scene_id = _as_non_empty_string(
            row.get("scene_id"), field_name="teacher-summary.scene_rows[].scene_id"
        )
        teacher_reachable = bool(row.get("teacher_reachable", False))
        included_in_formal_scene_pool = bool(
            row.get("included_in_formal_scene_pool", False)
        )
        teacher_success_count = _as_int(
            row.get("teacher_success_count", 0),
            field_name=f"teacher-summary.scene_rows[{family}].teacher_success_count",
        )
        student_success_count = _as_int(
            row.get("snapshot_baseline_success_count", 0),
            field_name=f"teacher-summary.scene_rows[{family}].snapshot_baseline_success_count",
        )
        if teacher_reachable and included_in_formal_scene_pool:
            reachable_families.append(family)
        elif not teacher_reachable:
            teacher_unreachable_families.append(family)
        else:
            scene_pool_excluded_families.append(family)

        teacher_student_success_gap_count: int | None
        teacher_student_success_gap_rate: float | None
        overall_gap_score: float | None
        blame_bucket: str
        if not teacher_reachable:
            teacher_student_success_gap_count = None
            teacher_student_success_gap_rate = None
            overall_gap_score = None
            blame_bucket = "teacher_unreachable"
        elif not included_in_formal_scene_pool:
            teacher_student_success_gap_count = None
            teacher_student_success_gap_rate = None
            overall_gap_score = None
            blame_bucket = "scene_pool_excluded_before_student_blame"
        else:
            gap_count = max(int(teacher_success_count - student_success_count), 0)
            gap_rate = float(gap_count) / float(max(teacher_success_count, 1))
            teacher_student_success_gap_count = int(gap_count)
            teacher_student_success_gap_rate = _round_float(gap_rate)
            overall_gap_score = _round_float(
                (gap_rate + (1.0 - student_branch_match_rate)) / 2.0
            )
            if student_success_count <= 0:
                blame_bucket = "student_not_learned"
            elif gap_count > 0:
                blame_bucket = "student_partial_gap"
            else:
                blame_bucket = "student_aligned"

        per_family_gap.append(
            {
                "family": family,
                "scene_id": scene_id,
                "priority": row.get("priority"),
                "teacher_reachable": teacher_reachable,
                "included_in_formal_scene_pool": included_in_formal_scene_pool,
                "teacher_interpretation_code": row.get("teacher_interpretation_code"),
                "teacher_success_count": int(teacher_success_count),
                "student_success_count": int(student_success_count),
                "teacher_student_success_gap_count": teacher_student_success_gap_count,
                "teacher_student_success_gap_rate": teacher_student_success_gap_rate,
                "student_branch_match_rate": _round_float(student_branch_match_rate),
                "overall_gap_score": overall_gap_score,
                "case_code": _formal_family_case_code(
                    teacher_reachable=teacher_reachable,
                    included_in_formal_scene_pool=included_in_formal_scene_pool,
                    student_success_count=student_success_count,
                    teacher_success_count=teacher_success_count,
                    student_branch_match_rate=student_branch_match_rate,
                ),
                "blame_bucket": blame_bucket,
            }
        )
    return (
        per_family_gap,
        reachable_families,
        teacher_unreachable_families,
        scene_pool_excluded_families,
    )


def _scorecard_case_code(
    *,
    teacher_summary: Mapping[str, Any],
    formal_reachable_families: Sequence[str],
    per_family_gap: Sequence[Mapping[str, Any]],
    student_branch_match_rate: float,
) -> tuple[str, str]:
    if not bool(teacher_summary.get("allow_formal_ladders", False)):
        scene_pool_status = str(teacher_summary.get("scene_pool_status"))
        if scene_pool_status in {
            "blocked_teacher_all_zero",
            "blocked_no_teacher_reachable_scenes",
        }:
            return (
                "teacher_unreachable_gap_block",
                "Teacher reachability blocked the formal scene pool, so the scorecard must stop before assigning blame to the student.",
            )
        return (
            "formal_scene_pool_blocked_before_student_blame",
            "The formal scene pool is blocked by replay or branch prerequisites, so student gap analysis remains informational only.",
        )
    if not formal_reachable_families:
        return (
            "no_formal_teacher_reachable_families",
            "No families survived the task-10 reachable-scene semantics, so there is no valid formal teacher-student gap surface.",
        )

    formal_rows = [
        row
        for row in per_family_gap
        if bool(row.get("included_in_formal_scene_pool", False))
    ]
    if formal_rows and all(
        int(row.get("student_success_count", 0)) <= 0 for row in formal_rows
    ):
        if student_branch_match_rate >= FORMAL_BRANCH_MATCH_THRESHOLD:
            return (
                "teacher_reachable_student_zero_branch_consistent",
                "Teacher reaches the formal families and the student's branch semantics still look consistent, so the dominant gap points to training or data rather than teacher-unreachable scenes.",
            )
        return (
            "teacher_reachable_student_zero_branch_inconsistent",
            "Teacher reaches the formal families but the student's action branch is not consistently preserved to controller input, so both student learning and branch wiring remain suspect.",
        )

    if any(
        float(row.get("teacher_student_success_gap_rate", 0.0) or 0.0) > 0.0
        for row in formal_rows
    ):
        if student_branch_match_rate >= FORMAL_BRANCH_MATCH_THRESHOLD:
            return (
                "teacher_reachable_student_partial_gap_branch_consistent",
                "Teacher and student both register some formal-family progress, but a remaining teacher-student gap persists even though the student's branch semantics are reasonably consistent.",
            )
        return (
            "teacher_reachable_student_partial_gap_branch_inconsistent",
            "Teacher and student both show some formal-family progress, but the remaining gap still coincides with branch-level inconsistency in the student's action chain.",
        )

    return (
        "teacher_student_formal_branch_aligned",
        "On the task-10 formal scene pool, the student no longer shows a material gap relative to the teacher summaries or the branch-consistency diagnostics.",
    )


def build_teacher_student_gap_scorecard(
    branch: str,
    *,
    output_path: Path | None = None,
    teacher_reachability_json: Path | None = None,
    action_telemetry_json: Path | None = None,
    open_loop_agreement_json: Path = DEFAULT_OPEN_LOOP_AGREEMENT_JSON,
) -> dict[str, Any]:
    resolved_output_path = (
        output_path
        if output_path is not None
        else default_output_path_for_branch(branch)
    )
    teacher_summary = load_teacher_reachability_summary(
        teacher_reachability_json or _default_teacher_reachability_path(branch)
    )
    telemetry_summary = load_action_telemetry_summary(
        branch,
        action_telemetry_json or _default_action_telemetry_path(branch),
    )
    open_loop_summary = load_open_loop_summary(open_loop_agreement_json)
    per_action_group_gap, telemetry_group_match_rate = build_per_action_group_gap(
        telemetry_summary
    )
    open_loop_check_pass_rate = _round_float(
        _as_number(
            open_loop_summary.get("check_pass_rate", 0.0),
            field_name="open-loop-summary.check_pass_rate",
        )
    )
    student_branch_match_rate = _round_float(
        telemetry_group_match_rate * open_loop_check_pass_rate
    )
    (
        per_family_gap,
        formal_reachable_families,
        teacher_unreachable_families,
        scene_pool_excluded_families,
    ) = build_per_family_gap(
        teacher_summary=teacher_summary,
        student_branch_match_rate=student_branch_match_rate,
    )
    case_code, case = _scorecard_case_code(
        teacher_summary=teacher_summary,
        formal_reachable_families=formal_reachable_families,
        per_family_gap=per_family_gap,
        student_branch_match_rate=student_branch_match_rate,
    )
    failure_note_path = (
        str(
            resolved_output_path.with_name(FAILURE_NOTE_MARKDOWN_NAME_BY_BRANCH[branch])
        )
        if not bool(teacher_summary.get("allow_formal_ladders", False))
        else None
    )

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "branch": branch,
        "branch_scope": teacher_summary.get("branch_scope"),
        "public_anchor_comparable": bool(
            teacher_summary.get("public_anchor_comparable", False)
        ),
        "output_path": _rel_repo(resolved_output_path),
        "failure_note_path": _rel_repo(Path(failure_note_path))
        if failure_note_path
        else None,
        "status": "ALLOW"
        if bool(teacher_summary.get("allow_formal_ladders", False))
        else "BLOCK",
        "case_code": case_code,
        "case": case,
        "teacher_case_code": teacher_summary.get("teacher_case_code"),
        "teacher_case": teacher_summary.get("teacher_case"),
        "replay_case_code": teacher_summary.get("replay_case_code"),
        "replay_case": teacher_summary.get("replay_case"),
        "scene_pool_status": teacher_summary.get("scene_pool_status"),
        "blocking_reasons": list(teacher_summary.get("blocking_reasons", [])),
        "reachable_scene_ids": list(teacher_summary.get("reachable_scene_ids", [])),
        "teacher_reachable_scene_ids": list(
            teacher_summary.get("teacher_reachable_scene_ids", [])
        ),
        "replay_reachable_scene_ids": list(
            teacher_summary.get("replay_reachable_scene_ids", [])
        ),
        "teacher_reachable_families": list(formal_reachable_families),
        "teacher_unreachable_families": list(teacher_unreachable_families),
        "scene_pool_excluded_families": list(scene_pool_excluded_families),
        "student_branch_match_rate": student_branch_match_rate,
        "student_branch_match_threshold": _round_float(FORMAL_BRANCH_MATCH_THRESHOLD),
        "student_branch_consistency": {
            "label": (
                "branch_consistent"
                if student_branch_match_rate >= FORMAL_BRANCH_MATCH_THRESHOLD
                else "branch_inconsistent"
            ),
            "telemetry_group_match_rate": telemetry_group_match_rate,
            "open_loop_check_pass_rate": open_loop_check_pass_rate,
            "student_branch_match_rate": student_branch_match_rate,
            "open_loop_status": open_loop_summary.get("status"),
            "open_loop_passed_check_count": open_loop_summary.get("passed_check_count"),
            "open_loop_total_check_count": open_loop_summary.get("total_check_count"),
            "history_condition_response": open_loop_summary.get(
                "history_condition_response"
            ),
            "valid_mask_effectiveness": open_loop_summary.get(
                "valid_mask_effectiveness"
            ),
            "action_range": open_loop_summary.get("action_range"),
        },
        "per_family_gap": per_family_gap,
        "per_action_group_gap": per_action_group_gap,
        "summary": {
            "formal_reachable_family_count": len(formal_reachable_families),
            "teacher_unreachable_family_count": len(teacher_unreachable_families),
            "scene_pool_excluded_family_count": len(scene_pool_excluded_families),
            "formal_teacher_student_zero_family_count": sum(
                1
                for row in per_family_gap
                if bool(row.get("included_in_formal_scene_pool", False))
                and int(row.get("student_success_count", 0)) <= 0
            ),
            "action_group_gap_count": sum(
                1
                for row in per_action_group_gap.values()
                if float(dict(row).get("gap_score", 0.0)) > 0.0
            ),
        },
        "source_artifacts": {
            "teacher_reachability_gate": teacher_summary.get("source_path"),
            "action_telemetry": telemetry_summary.get("source_path"),
            "open_loop_agreement": open_loop_summary.get("source_path"),
            "task_7_action_telemetry_evidence": ".sisyphus/evidence/task-7-action-telemetry.json",
            "task_10_teacher_reachability_evidence": ".sisyphus/evidence/task-10-teacher-reachability.json",
        },
        "decision_basis": (
            "Use task-10 reachable_scene_ids / scene_rows as the only formal family surface; "
            "treat teacher-unreachable families as excluded from student blame; compute "
            "student_branch_match_rate from task-7 action telemetry and the shared open-loop "
            "agreement pass rate; then report family-level success gaps and action-group-level "
            "branch gaps side by side."
        ),
    }
    report["report_signature_sha256"] = _sha256(report)
    return report


def _build_failure_note(report: Mapping[str, Any]) -> str:
    lines = [
        "# GR00T teacher-student conditional gap failure note",
        "",
        f"- branch: `{report.get('branch')}`",
        f"- status: `{report.get('status')}`",
        f"- case_code: `{report.get('case_code')}`",
        f"- scene_pool_status: `{report.get('scene_pool_status')}`",
        f"- teacher_case_code: `{report.get('teacher_case_code')}`",
        f"- replay_case_code: `{report.get('replay_case_code')}`",
        f"- teacher_unreachable_families: `{json.dumps(report.get('teacher_unreachable_families', []), ensure_ascii=True)}`",
        "",
        "## Why attribution is blocked",
        "",
        str(report.get("case")),
        "",
        "## Blocking reasons",
        "",
        "```json",
        json.dumps(
            report.get("blocking_reasons", []),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Student branch consistency snapshot",
        "",
        "```json",
        json.dumps(
            report.get("student_branch_consistency", {}),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_scorecard_artifacts(report: Mapping[str, Any], *, output_path: Path) -> Path:
    written = _write_json(output_path, report)
    failure_note_path = output_path.with_name(
        FAILURE_NOTE_MARKDOWN_NAME_BY_BRANCH[str(report["branch"])]
    )
    if str(report.get("status")) != "ALLOW":
        failure_note_path.write_text(_build_failure_note(report), encoding="utf-8")
    elif failure_note_path.exists():
        failure_note_path.unlink()
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        branch = str(args.branch)
        output_path = _validate_output_path(
            args.output
            if args.output is not None
            else default_output_path_for_branch(branch)
        )
        report = build_teacher_student_gap_scorecard(
            branch,
            output_path=output_path,
            teacher_reachability_json=args.teacher_reachability_json,
            action_telemetry_json=args.action_telemetry_json,
            open_loop_agreement_json=args.open_loop_agreement_json,
        )
        _ = write_scorecard_artifacts(report, output_path=output_path)
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
