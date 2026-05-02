from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_ROOT = Path("agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/p")
DEFAULT_P_LADDER_POLICY_GATE_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/ladder_policy/p_ladder_policy_gate_unitree_g1.json"
)
DEFAULT_DUAL_BRANCH_SCORECARD_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/dual_branch_scorecard.json"
)
DEFAULT_CHECKPOINT_PROVENANCE_JSON = Path(
    "agent/artifacts/gr00t_checkpoint_provenance/checkpoint_provenance_report.json"
)
DEFAULT_CONDITION_FLIP_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/condition_flip_scorecard_unitree_g1.json"
)
DEFAULT_TEACHER_STUDENT_GAP_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/teacher_student_gap_scorecard_unitree_g1.json"
)
DEFAULT_ACTION_TELEMETRY_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/action_chain_telemetry_unitree_g1.json"
)
DEFAULT_TEACHER_REACHABILITY_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/teacher_reachability/teacher_reachability_gate_unitree_g1.json"
)
DEFAULT_TASK2_PREFLIGHT_EVIDENCE_JSON = Path(".sisyphus/evidence/task-2-preflight.json")

REPORT_SCHEMA_VERSION = "gr00t_p_ladder_unitree_g1_v1"
MANIFEST_ARTIFACT_KIND = "gr00t_p_ladder_unitree_g1_manifest"
SCORECARD_ARTIFACT_KIND = "gr00t_p_ladder_unitree_g1_scorecard"

BRANCH = "UNITREE_G1"
BRANCH_KEY = "unitree_g1"
AXIS = "P"

RUNG_ORDER: tuple[str, ...] = ("P0", "P1", "P2", "P3")
POSITIVE_SLOPE_METRIC_EPS = 1e-9

DEFAULT_DATASET_PATH = "unitree_g1.LMPnPAppleToPlateDC"
DEFAULT_DATASET_MIX: tuple[str, ...] = ("unitree_g1.LMPnPAppleToPlateDC:1.0",)
DEFAULT_ADMISSION_POLICY_VERSION = "unitree_g1_formal_p_ladder_v1"
DEFAULT_NORMALIZATION_POLICY = "unitree_g1_branch_specific_stats_v1"
DEFAULT_CONDITION_SCHEMA = "gr00t_policy_condition_v1"
DEFAULT_CONDITION_INJECTION = "task_text_only"
DEFAULT_CONTROLLER_FAMILY = "GR00T-WholeBodyControl"
DEFAULT_RELATIVE_ACTION_POLICY = "unitree_g1_arm_relative_else_absolute"
DEFAULT_ACTION_KEYS: tuple[str, ...] = (
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
    "navigate_command",
    "base_height_command",
)
DEFAULT_STATE_KEYS: tuple[str, ...] = DEFAULT_ACTION_KEYS


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import gr00t_checkpoint_provenance_gate
from work.recap import gr00t_eval_contract_gate
from work.recap import gr00t_ladder_policy_gate
from work.recap import state_conditioned_bucket_a_import


BASE_TRAINING_SURFACE: dict[str, Any] = {
    "parameter_update": {
        "visual_unfreeze": False,
        "lora_enabled": False,
        "lora_rank": 0,
        "selective_unfreeze_modules": [],
        "tune_visual": False,
        "tune_projector": False,
        "tune_diffusion_model": False,
        "tune_llm": False,
    },
    "optimizer": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "gradient_clip_norm": 1.0,
    },
    "schedule": {
        "max_steps": 100,
        "save_steps": 100,
        "save_total_limit": 1,
        "warmup_ratio": 0.05,
        "global_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_gpus": 1,
        "dataloader_num_workers": 0,
    },
}

RUNG_SURFACE_PATCHES: dict[str, dict[str, Any]] = {
    "P0": {},
    "P1": {
        "parameter_update": {
            "visual_unfreeze": True,
            "tune_visual": True,
        }
    },
    "P2": {
        "parameter_update": {
            "visual_unfreeze": True,
            "tune_visual": True,
            "lora_enabled": True,
            "lora_rank": 16,
            "selective_unfreeze_modules": ["llm.layers.31"],
            "tune_llm": True,
        }
    },
    "P3": {
        "parameter_update": {
            "visual_unfreeze": True,
            "tune_visual": True,
            "lora_enabled": True,
            "lora_rank": 16,
            "selective_unfreeze_modules": ["llm.layers.30", "llm.layers.31"],
            "tune_llm": True,
        },
        "schedule": {
            "max_steps": 160,
            "save_steps": 160,
        },
    },
}

RUNG_DESCRIPTIONS = {
    "P0": "current_setting_frozen_parameter_surface",
    "P1": "visual_unfreeze_only",
    "P2": "visual_unfreeze_plus_top_layer_peft_or_selective_unfreeze",
    "P3": "higher_budget_parameter_rung_only_if_positive_slope_detected",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_p_ladder_unitree_g1.py",
        description=(
            "Materialize the UNITREE_G1 P0-P3 parameter ladder manifests and scorecards "
            "under a frozen formal protocol."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--rung", required=True, choices=list(RUNG_ORDER))
    _ = parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _ = parser.add_argument(
        "--p-ladder-policy-gate-json",
        type=Path,
        default=DEFAULT_P_LADDER_POLICY_GATE_JSON,
    )
    _ = parser.add_argument(
        "--dual-branch-scorecard-json",
        type=Path,
        default=DEFAULT_DUAL_BRANCH_SCORECARD_JSON,
    )
    _ = parser.add_argument(
        "--checkpoint-provenance-json",
        type=Path,
        default=DEFAULT_CHECKPOINT_PROVENANCE_JSON,
    )
    _ = parser.add_argument(
        "--condition-flip-json",
        type=Path,
        default=DEFAULT_CONDITION_FLIP_JSON,
    )
    _ = parser.add_argument(
        "--teacher-student-gap-json",
        type=Path,
        default=DEFAULT_TEACHER_STUDENT_GAP_JSON,
    )
    _ = parser.add_argument(
        "--action-telemetry-json",
        type=Path,
        default=DEFAULT_ACTION_TELEMETRY_JSON,
    )
    _ = parser.add_argument(
        "--teacher-reachability-json",
        type=Path,
        default=DEFAULT_TEACHER_REACHABILITY_JSON,
    )
    _ = parser.add_argument(
        "--task2-preflight-evidence-json",
        type=Path,
        default=DEFAULT_TASK2_PREFLIGHT_EVIDENCE_JSON,
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _resolve_path(path: Path | str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    return raw.resolve()


def _resolve_existing_file(path: Path | str, *, arg_name: str) -> Path:
    resolved = _resolve_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{arg_name} does not exist: {resolved}")
    return resolved


def _validate_rung(rung: str) -> str:
    normalized = str(rung).strip().upper()
    if normalized not in RUNG_ORDER:
        raise ValueError(f"unsupported rung: {rung}")
    return normalized


def _rel_repo(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: Path | str, *, arg_name: str) -> dict[str, Any]:
    resolved = _resolve_existing_file(path, arg_name=arg_name)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{arg_name} must contain a JSON object")
    return cast(dict[str, Any], dict(payload))


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
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric, got {type(value).__name__}")
    return float(value)


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _round_float(value: float, *, digits: int = 8) -> float:
    return float(round(float(value), digits))


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _signature_from_payload(payload: Mapping[str, Any]) -> str:
    existing = payload.get("report_signature_sha256")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    checksum = payload.get("checksum_or_signature")
    if isinstance(checksum, str) and checksum.strip():
        return checksum.strip()
    return _sha256(payload)


def load_preflight_prerequisite_proof(
    task2_preflight_evidence_json: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    evidence_path = _resolve_path(task2_preflight_evidence_json)
    evidence_payload = _read_json(
        evidence_path, arg_name="task2-preflight-evidence-json"
    )
    if str(evidence_payload.get("artifact_kind")) != "task_2_preflight_evidence":
        raise ValueError("task2-preflight-evidence-json artifact_kind mismatch")

    verification = _as_mapping(
        evidence_payload.get("verification", {}),
        field_name="task2_preflight.verification",
    )
    success_run = _as_mapping(
        verification.get("success_run", {}),
        field_name="task2_preflight.verification.success_run",
    )
    report_path_raw = success_run.get("default_report_path") or success_run.get(
        "verification_report_path"
    )
    if report_path_raw is None:
        raise ValueError(
            "task2-preflight-evidence-json missing success_run.default_report_path"
        )
    report_path = _resolve_existing_file(
        Path(
            _as_non_empty_string(report_path_raw, field_name="task2_preflight.report")
        ),
        arg_name="task2-preflight-report",
    )
    report_payload = _read_json(report_path, arg_name="task2-preflight-report")
    runtime_log_path = _resolve_existing_file(
        Path(
            _as_non_empty_string(
                success_run.get("runtime_log"),
                field_name="task2_preflight.verification.success_run.runtime_log",
            )
        ),
        arg_name="task2-preflight-runtime-log",
    )
    env_resolution = _as_mapping(
        report_payload.get("env_resolution", {}),
        field_name="task2_preflight_report.env_resolution",
    )
    policy_ping = _as_mapping(
        report_payload.get("policy_ping", {}),
        field_name="task2_preflight_report.policy_ping",
    )
    action_horizon_check = _as_mapping(
        report_payload.get("action_horizon_check", {}),
        field_name="task2_preflight_report.action_horizon_check",
    )
    smoke = _as_mapping(
        report_payload.get("smoke", {}), field_name="task2_preflight_report.smoke"
    )
    system_break_flags = _as_mapping(
        report_payload.get("system_break_flags", {}),
        field_name="task2_preflight_report.system_break_flags",
    )
    proof: dict[str, Any] = {
        "status": str(report_payload.get("status", "UNKNOWN")),
        "reason_code": str(report_payload.get("reason_code", "unknown")),
        "evidence_path": _rel_repo(evidence_path),
        "evidence_artifact_kind": str(evidence_payload.get("artifact_kind")),
        "evidence_signature_sha256": _signature_from_payload(evidence_payload),
        "preflight_report_path": _rel_repo(report_path),
        "preflight_report_signature_sha256": _signature_from_payload(report_payload),
        "runtime_log": _rel_repo(runtime_log_path),
        "env_resolution_ok": bool(env_resolution.get("ok", False)),
        "policy_ping_ok": bool(policy_ping.get("ok", False)),
        "action_horizon_check_ok": bool(action_horizon_check.get("ok", False)),
        "smoke_step_ok": bool(smoke.get("step_ok", False)),
        "system_break_flags": [
            str(item)
            for item in _as_list(
                system_break_flags.get("active_breaks", []),
                field_name="task2_preflight_report.system_break_flags.active_breaks",
            )
        ],
    }
    prerequisite_hash: dict[str, str] = {
        "path": _rel_repo(report_path) or str(report_path),
        "signature": str(proof["preflight_report_signature_sha256"]),
        "evidence_path": str(proof["evidence_path"] or str(evidence_path)),
    }
    return proof, prerequisite_hash


def _merge_nested(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_nested(
                cast(Mapping[str, Any], merged[key]),
                cast(Mapping[str, Any], value),
            )
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _deep_get(payload: Mapping[str, Any], dotted_path: str) -> object | None:
    current: object = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _mean_formal_teacher_student_gap(payload: Mapping[str, Any]) -> float:
    formal_rows: list[float] = []
    for index, raw in enumerate(
        _as_list(
            payload.get("per_family_gap", []), field_name="teacher_gap.per_family_gap"
        )
    ):
        row = _as_mapping(raw, field_name=f"teacher_gap.per_family_gap[{index}]")
        if not bool(row.get("included_in_formal_scene_pool", False)):
            continue
        gap_value = row.get("teacher_student_success_gap_rate")
        if gap_value is None:
            continue
        formal_rows.append(
            _as_number(
                gap_value,
                field_name=(
                    f"teacher_gap.per_family_gap[{index}].teacher_student_success_gap_rate"
                ),
            )
        )
    if formal_rows:
        return _round_float(sum(formal_rows) / float(len(formal_rows)))
    student_branch_match_rate = _as_number(
        payload.get("student_branch_match_rate", 0.0),
        field_name="teacher_gap.student_branch_match_rate",
    )
    return _round_float(max(0.0, 1.0 - student_branch_match_rate))


def _unitree_branch_entry(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    branches = _as_list(payload.get("branches", []), field_name="dual_branch.branches")
    for index, raw in enumerate(branches):
        row = _as_mapping(raw, field_name=f"dual_branch.branches[{index}]")
        if str(row.get("branch_key")) == BRANCH_KEY:
            return row
    raise ValueError("dual-branch scorecard missing unitree_g1 branch entry")


def _build_base_metrics(*, prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    public_anchor = _as_mapping(
        _deep_get(
            prerequisites, "dual_branch.unitree_branch.public_anchor_status.summary"
        )
        or {},
        field_name="dual_branch.unitree_branch.public_anchor_status.summary",
    )
    condition_flip = _as_mapping(
        prerequisites.get("condition_flip", {}), field_name="condition_flip"
    )
    condition_response = _as_mapping(
        condition_flip.get("response_ratio", {}),
        field_name="condition_flip.response_ratio",
    )
    teacher_gap = _as_mapping(
        prerequisites.get("teacher_student_gap", {}), field_name="teacher_student_gap"
    )
    action_telemetry = _as_mapping(
        prerequisites.get("action_telemetry", {}), field_name="action_telemetry"
    )
    controller_absorbed_groups = {
        str(item)
        for item in _as_list(
            action_telemetry.get("controller_absorbed_groups", []),
            field_name="action_telemetry.controller_absorbed_groups",
        )
    }
    model_insensitive_groups = {
        str(item)
        for item in _as_list(
            action_telemetry.get("model_insensitive_groups", []),
            field_name="action_telemetry.model_insensitive_groups",
        )
    }
    zero_motion_groups = {
        str(item)
        for item in _as_list(
            action_telemetry.get("zero_motion_groups", []),
            field_name="action_telemetry.zero_motion_groups",
        )
    }
    action_problem_groups = sorted(
        controller_absorbed_groups | model_insensitive_groups | zero_motion_groups
    )

    return {
        "success_count": _as_int(
            public_anchor.get("success_count", 0),
            field_name="dual_branch.unitree_branch.public_anchor_status.summary.success_count",
        ),
        "success_rate": _round_float(
            _as_number(
                public_anchor.get("success_rate", 0.0),
                field_name=(
                    "dual_branch.unitree_branch.public_anchor_status.summary.success_rate"
                ),
            )
        ),
        "condition_flip_response_ratio": _round_float(
            _as_number(
                condition_response.get("min_ratio_across_semantic_flips", 0.0),
                field_name="condition_flip.response_ratio.min_ratio_across_semantic_flips",
            )
        ),
        "teacher_student_gap": _mean_formal_teacher_student_gap(teacher_gap),
        "action_chain_problem_group_count": int(len(action_problem_groups)),
        "controller_absorbed_group_count": int(len(controller_absorbed_groups)),
        "model_insensitive_group_count": int(len(model_insensitive_groups)),
        "zero_motion_group_count": int(len(zero_motion_groups)),
        "systemic_break_flags": [
            str(item)
            for item in _as_list(
                public_anchor.get("systemic_break_flags", []),
                field_name=(
                    "dual_branch.unitree_branch.public_anchor_status.summary.systemic_break_flags"
                ),
            )
        ],
        "provenance_status": str(
            _as_mapping(
                prerequisites.get("checkpoint_provenance", {}),
                field_name="checkpoint_provenance",
            ).get("formal_eligibility", "BLOCK")
        ),
    }


def _apply_metric_overrides(
    base_metrics: Mapping[str, Any],
    *,
    rung: str,
    metrics_overrides_by_rung: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base_metrics))
    if metrics_overrides_by_rung is None:
        return merged
    override = metrics_overrides_by_rung.get(rung)
    if override is None:
        return merged
    for key, value in override.items():
        merged[str(key)] = copy.deepcopy(value)
    return merged


def load_prerequisites(
    *,
    p_ladder_policy_gate_json: Path,
    dual_branch_scorecard_json: Path,
    checkpoint_provenance_json: Path,
    condition_flip_json: Path,
    teacher_student_gap_json: Path,
    action_telemetry_json: Path,
    teacher_reachability_json: Path,
    task2_preflight_evidence_json: Path,
) -> dict[str, Any]:
    gate_payload = _read_json(
        p_ladder_policy_gate_json, arg_name="p-ladder-policy-gate-json"
    )
    dual_branch_payload = _read_json(
        dual_branch_scorecard_json, arg_name="dual-branch-scorecard-json"
    )
    checkpoint_payload = _read_json(
        checkpoint_provenance_json, arg_name="checkpoint-provenance-json"
    )
    condition_flip_payload = _read_json(
        condition_flip_json, arg_name="condition-flip-json"
    )
    teacher_gap_payload = _read_json(
        teacher_student_gap_json, arg_name="teacher-student-gap-json"
    )
    action_telemetry_payload = _read_json(
        action_telemetry_json, arg_name="action-telemetry-json"
    )
    teacher_reachability_payload = _read_json(
        teacher_reachability_json, arg_name="teacher-reachability-json"
    )
    preflight_prerequisite_proof, preflight_prerequisite_hash = (
        load_preflight_prerequisite_proof(task2_preflight_evidence_json)
    )

    if (
        gate_payload.get("artifact_kind")
        != gr00t_ladder_policy_gate.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("p-ladder-policy-gate-json artifact_kind mismatch")
    if (
        str(gate_payload.get("branch")) != BRANCH
        or str(gate_payload.get("ladder_axis")) != AXIS
    ):
        raise ValueError("p-ladder-policy-gate-json must target UNITREE_G1 P ladder")

    if str(dual_branch_payload.get("artifact_kind")) != "gr00t_dual_branch_scorecard":
        raise ValueError("dual-branch-scorecard-json artifact_kind mismatch")
    if (
        str(checkpoint_payload.get("artifact_kind"))
        != gr00t_checkpoint_provenance_gate.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("checkpoint-provenance-json artifact_kind mismatch")
    if (
        str(condition_flip_payload.get("artifact_kind"))
        != "gr00t_condition_flip_scorecard"
    ):
        raise ValueError("condition-flip-json artifact_kind mismatch")
    if (
        str(teacher_gap_payload.get("artifact_kind"))
        != "gr00t_teacher_student_gap_scorecard"
    ):
        raise ValueError("teacher-student-gap-json artifact_kind mismatch")
    if (
        str(action_telemetry_payload.get("artifact_kind"))
        != "gr00t_action_chain_telemetry"
    ):
        raise ValueError("action-telemetry-json artifact_kind mismatch")
    if (
        str(teacher_reachability_payload.get("artifact_kind"))
        != "gr00t_teacher_reachability_gate"
    ):
        raise ValueError("teacher-reachability-json artifact_kind mismatch")

    unitree_branch = _unitree_branch_entry(dual_branch_payload)
    prerequisite_hashes = {
        "p_ladder_policy_gate": {
            "path": _rel_repo(_resolve_path(p_ladder_policy_gate_json)),
            "signature": _signature_from_payload(gate_payload),
        },
        "dual_branch_scorecard": {
            "path": _rel_repo(_resolve_path(dual_branch_scorecard_json)),
            "signature": _signature_from_payload(dual_branch_payload),
        },
        "checkpoint_provenance": {
            "path": _rel_repo(_resolve_path(checkpoint_provenance_json)),
            "signature": _signature_from_payload(checkpoint_payload),
        },
        "condition_flip": {
            "path": _rel_repo(_resolve_path(condition_flip_json)),
            "signature": _signature_from_payload(condition_flip_payload),
        },
        "teacher_student_gap": {
            "path": _rel_repo(_resolve_path(teacher_student_gap_json)),
            "signature": _signature_from_payload(teacher_gap_payload),
        },
        "action_telemetry": {
            "path": _rel_repo(_resolve_path(action_telemetry_json)),
            "signature": _signature_from_payload(action_telemetry_payload),
        },
        "teacher_reachability": {
            "path": _rel_repo(_resolve_path(teacher_reachability_json)),
            "signature": _signature_from_payload(teacher_reachability_payload),
        },
        "preflight": dict(preflight_prerequisite_hash),
    }
    prerequisites: dict[str, Any] = {
        "p_ladder_policy_gate": gate_payload,
        "dual_branch": {
            "payload": dual_branch_payload,
            "unitree_branch": dict(unitree_branch),
            "allow_p_ladder": bool(
                cast(
                    Mapping[str, Any], dual_branch_payload.get("allow_p_ladder", {})
                ).get(BRANCH_KEY, False)
            ),
        },
        "checkpoint_provenance": checkpoint_payload,
        "condition_flip": condition_flip_payload,
        "teacher_student_gap": teacher_gap_payload,
        "action_telemetry": action_telemetry_payload,
        "teacher_reachability": teacher_reachability_payload,
        "preflight_prerequisite_proof": preflight_prerequisite_proof,
        "diagnostics_prerequisite_hashes": prerequisite_hashes,
    }
    prerequisites["base_metrics"] = _build_base_metrics(prerequisites=prerequisites)
    return prerequisites


def build_training_surface_for_rung(rung: str) -> dict[str, Any]:
    normalized = _validate_rung(rung)
    return _merge_nested(BASE_TRAINING_SURFACE, RUNG_SURFACE_PATCHES[normalized])


def build_frozen_data_surface() -> dict[str, Any]:
    dataset_surface = {
        "dataset_path": DEFAULT_DATASET_PATH,
        "dataset_mix": list(DEFAULT_DATASET_MIX),
        "admission": {
            "branch_inclusion": [BRANCH],
            "dataset_source_ids": [DEFAULT_DATASET_PATH],
            "dataset_fingerprints": [DEFAULT_DATASET_PATH],
            "admission_policy_version": DEFAULT_ADMISSION_POLICY_VERSION,
        },
        "normalization": {
            "explicit_stats_policy": DEFAULT_NORMALIZATION_POLICY,
            "stats_fingerprint": "unitree_g1_stats_fingerprint_v1",
            "stats_owner": BRANCH_KEY,
            "explicit_diff_reason": "none",
            "hidden_stats_fingerprint": "unitree_g1_hidden_stats_fingerprint_v1",
            "implicit_cross_branch_stats_reuse": False,
        },
        "sampling": {
            "seed_policy": "repo_local_formal_seed_manifest_v1",
            "episode_sampling_policy": "teacher_reachable_scene_pool_fixed_equal_weight_v1",
        },
    }
    dataset_surface["dataset_fingerprint"] = _sha256(dataset_surface)
    return dataset_surface


def build_comparison_surface(rung: str) -> dict[str, Any]:
    training_surface = build_training_surface_for_rung(rung)
    data_surface = build_frozen_data_surface()
    return {
        "branch": {
            "branch_key": BRANCH_KEY,
            "branch_scope": gr00t_ladder_policy_gate.BRANCH_SPECS[BRANCH].branch_scope,
            "public_anchor_comparable": True,
        },
        "embodiment": {
            "embodiment_tag": BRANCH,
            "modality_config_path": "builtin::UNITREE_G1",
            "modality_config_digest": "builtin::unitree_g1_post_train_contract_v1",
        },
        "controller": {
            "controller_family": DEFAULT_CONTROLLER_FAMILY,
            "action_horizon": int(
                gr00t_eval_contract_gate.DEFAULT_POLICY_HORIZON_EXPECTED
            ),
            "relative_action_policy": DEFAULT_RELATIVE_ACTION_POLICY,
            "action_keys": list(DEFAULT_ACTION_KEYS),
            "state_keys": list(DEFAULT_STATE_KEYS),
        },
        "prompt_interface": {
            "prompt_template_id": gr00t_eval_contract_gate.DEFAULT_PROMPT_TEMPLATE_ID,
            "condition_injection": DEFAULT_CONDITION_INJECTION,
            "condition_schema": DEFAULT_CONDITION_SCHEMA,
        },
        "dataset": data_surface,
        "training": training_surface,
    }


def _new_systemic_break(
    baseline_flags: Sequence[str], current_flags: Sequence[str]
) -> tuple[bool, list[str]]:
    baseline = {str(item) for item in baseline_flags}
    current = {str(item) for item in current_flags}
    new_flags = sorted(current - baseline)
    return bool(new_flags), new_flags


def _build_action_chain_delta(
    baseline_metrics: Mapping[str, Any], current_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "problem_group_count_delta": int(
            current_metrics["action_chain_problem_group_count"]
            - baseline_metrics["action_chain_problem_group_count"]
        ),
        "controller_absorbed_group_count_delta": int(
            current_metrics["controller_absorbed_group_count"]
            - baseline_metrics["controller_absorbed_group_count"]
        ),
        "model_insensitive_group_count_delta": int(
            current_metrics["model_insensitive_group_count"]
            - baseline_metrics["model_insensitive_group_count"]
        ),
        "zero_motion_group_count_delta": int(
            current_metrics["zero_motion_group_count"]
            - baseline_metrics["zero_motion_group_count"]
        ),
    }


def _diagnostics_not_regressing(
    baseline_metrics: Mapping[str, Any], current_metrics: Mapping[str, Any]
) -> bool:
    if (
        float(current_metrics["teacher_student_gap"])
        > float(baseline_metrics["teacher_student_gap"]) + POSITIVE_SLOPE_METRIC_EPS
    ):
        return False
    if float(
        current_metrics["condition_flip_response_ratio"]
    ) + POSITIVE_SLOPE_METRIC_EPS < float(
        baseline_metrics["condition_flip_response_ratio"]
    ):
        return False
    if int(current_metrics["action_chain_problem_group_count"]) > int(
        baseline_metrics["action_chain_problem_group_count"]
    ):
        return False
    return True


def _positive_slope_report(
    *,
    baseline_metrics: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    success_improved = int(current_metrics["success_count"]) > int(
        baseline_metrics["success_count"]
    )
    condition_flip_improved = (
        float(current_metrics["condition_flip_response_ratio"])
        > float(baseline_metrics["condition_flip_response_ratio"])
        + POSITIVE_SLOPE_METRIC_EPS
    )
    teacher_gap_improved = float(
        current_metrics["teacher_student_gap"]
    ) + POSITIVE_SLOPE_METRIC_EPS < float(baseline_metrics["teacher_student_gap"])
    qualifying_metric_names = [
        name
        for name, improved in (
            ("condition_flip_response_ratio", condition_flip_improved),
            ("teacher_student_gap", teacher_gap_improved),
            ("success_count", success_improved),
        )
        if improved
    ]
    provenance_regression = (
        str(current_metrics["provenance_status"])
        != str(baseline_metrics["provenance_status"])
        or str(current_metrics["provenance_status"]) != "ALLOW"
    )
    new_systemic_break, new_flags = _new_systemic_break(
        cast(Sequence[str], baseline_metrics["systemic_break_flags"]),
        cast(Sequence[str], current_metrics["systemic_break_flags"]),
    )
    positive_slope_detected = (
        bool(qualifying_metric_names)
        and not provenance_regression
        and not new_systemic_break
    )
    return {
        "positive_slope_detected": positive_slope_detected,
        "qualifying_metric_names": qualifying_metric_names,
        "metrics_checked": {
            "condition_flip_response_ratio": {
                "direction": "higher_is_better",
                "baseline": _round_float(
                    float(baseline_metrics["condition_flip_response_ratio"])
                ),
                "current": _round_float(
                    float(current_metrics["condition_flip_response_ratio"])
                ),
                "improved": condition_flip_improved,
            },
            "teacher_student_gap": {
                "direction": "lower_is_better",
                "baseline": _round_float(
                    float(baseline_metrics["teacher_student_gap"])
                ),
                "current": _round_float(float(current_metrics["teacher_student_gap"])),
                "improved": teacher_gap_improved,
            },
            "success_count": {
                "direction": "higher_is_better",
                "baseline": int(baseline_metrics["success_count"]),
                "current": int(current_metrics["success_count"]),
                "improved": success_improved,
            },
        },
        "provenance_regression": provenance_regression,
        "new_systemic_break": new_systemic_break,
        "new_systemic_break_flags": new_flags,
    }


def _promotion_evidence(
    *,
    baseline_metrics: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "fixed_replicate_policy": True,
        "fixed_seed_policy": True,
        "no_systemic_break": len(
            cast(Sequence[str], current_metrics["systemic_break_flags"])
        )
        == 0,
        "provenance_pass": str(current_metrics["provenance_status"]) == "ALLOW",
        "diagnostics_not_regressing": _diagnostics_not_regressing(
            baseline_metrics, current_metrics
        ),
        "single_lucky_seed_only_improvement": False,
        "frozen_formal_scene_pool": bool(
            _as_mapping(
                prerequisites.get("teacher_reachability", {}),
                field_name="teacher_reachability",
            ).get("allow_formal_ladders", False)
        ),
    }


def _load_existing_scorecard(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in prior scorecard {path}")
    return cast(dict[str, Any], dict(payload))


def _p3_positive_slope_from_prior_rungs(output_root: Path) -> dict[str, Any]:
    prior_reports: dict[str, dict[str, Any] | None] = {}
    for rung in ("P1", "P2"):
        prior_reports[rung] = _load_existing_scorecard(
            output_root / rung / "scorecard.json"
        )

    missing_rungs = sorted(
        rung for rung, payload in prior_reports.items() if payload is None
    )
    qualifying_rungs: list[str] = []
    qualifying_metrics_by_rung: dict[str, list[str]] = {}
    for rung, payload in prior_reports.items():
        if payload is None:
            continue
        positive = _as_mapping(
            payload.get("positive_slope_report", {}),
            field_name=f"prior_scorecard[{rung}].positive_slope_report",
        )
        if bool(positive.get("positive_slope_detected", False)):
            qualifying_rungs.append(rung)
            qualifying_metrics_by_rung[rung] = [
                str(item)
                for item in _as_list(
                    positive.get("qualifying_metric_names", []),
                    field_name=f"prior_scorecard[{rung}].positive_slope_report.qualifying_metric_names",
                )
            ]

    positive_slope_detected = bool(qualifying_rungs)
    blocking_reasons: list[str] = []
    if missing_rungs:
        blocking_reasons.append("prior_positive_slope_evidence_missing")
    if not positive_slope_detected:
        blocking_reasons.append("positive_slope_required_for_p3")
    return {
        "positive_slope_detected": positive_slope_detected,
        "qualifying_rungs": qualifying_rungs,
        "qualifying_metrics_by_rung": qualifying_metrics_by_rung,
        "missing_rungs": missing_rungs,
        "blocking_reasons": blocking_reasons,
    }


def build_rung_report(
    *,
    rung: str,
    prerequisites: Mapping[str, Any],
    output_root: Path,
    metrics_overrides_by_rung: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_rung = _validate_rung(rung)
    output_dir = (output_root / normalized_rung).resolve()
    manifest_path = output_dir / "manifest.json"
    scorecard_path = output_dir / "scorecard.json"

    p_gate_payload = _as_mapping(
        prerequisites.get("p_ladder_policy_gate", {}),
        field_name="prerequisites.p_ladder_policy_gate",
    )
    baseline_surface = build_comparison_surface("P0")
    rung_surface = build_comparison_surface(normalized_rung)
    comparability = gr00t_ladder_policy_gate.build_ladder_diff_report(
        p_gate_payload,
        baseline_surface,
        rung_surface,
    )

    base_metrics = _as_mapping(
        prerequisites.get("base_metrics", {}), field_name="prerequisites.base_metrics"
    )
    baseline_metrics = _apply_metric_overrides(
        base_metrics,
        rung="P0",
        metrics_overrides_by_rung=metrics_overrides_by_rung,
    )
    current_metrics = _apply_metric_overrides(
        base_metrics,
        rung=normalized_rung,
        metrics_overrides_by_rung=metrics_overrides_by_rung,
    )
    current_metrics["success_rate"] = _round_float(
        float(current_metrics["success_rate"])
    )
    current_metrics["condition_flip_response_ratio"] = _round_float(
        float(current_metrics["condition_flip_response_ratio"])
    )
    current_metrics["teacher_student_gap"] = _round_float(
        float(current_metrics["teacher_student_gap"])
    )

    promotion_evidence = _promotion_evidence(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        prerequisites=prerequisites,
    )
    promotion_report = gr00t_ladder_policy_gate.build_promotion_report(
        p_gate_payload,
        promotion_evidence,
    )
    positive_slope_report = _positive_slope_report(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
    )

    generic_blocking_reasons: list[str] = []
    failure_reasons = list(
        cast(Sequence[str], promotion_report.get("failure_reasons", []))
    )
    if str(comparability.get("comparability_status")) != "PASS":
        generic_blocking_reasons.extend(
            [str(item) for item in comparability.get("blocking_reasons", [])]
        )
    dual_branch = _as_mapping(
        prerequisites.get("dual_branch", {}), field_name="dual_branch"
    )
    if not bool(dual_branch.get("allow_p_ladder", False)):
        generic_blocking_reasons.append("dual_branch_prerequisite_blocks_p_ladder")
    checkpoint_provenance = _as_mapping(
        prerequisites.get("checkpoint_provenance", {}),
        field_name="checkpoint_provenance",
    )
    if str(checkpoint_provenance.get("formal_eligibility")) != "ALLOW":
        generic_blocking_reasons.append("checkpoint_provenance_blocked")
    teacher_reachability = _as_mapping(
        prerequisites.get("teacher_reachability", {}), field_name="teacher_reachability"
    )
    preflight_prerequisite_proof = _as_mapping(
        prerequisites.get("preflight_prerequisite_proof", {}),
        field_name="preflight_prerequisite_proof",
    )
    if not bool(teacher_reachability.get("allow_formal_ladders", False)):
        generic_blocking_reasons.append("teacher_reachability_blocks_formal_scene_pool")

    status = "PASS"
    blocking_reasons = list(dict.fromkeys(generic_blocking_reasons))
    p3_gate_report: dict[str, Any] | None = None
    if normalized_rung == "P3":
        p3_gate_report = _p3_positive_slope_from_prior_rungs(output_root)
        blocking_reasons.extend(
            [str(item) for item in p3_gate_report.get("blocking_reasons", [])]
        )
        if blocking_reasons:
            status = "BLOCK"
    elif blocking_reasons:
        status = "BLOCK"

    baseline_flags = [str(item) for item in baseline_metrics["systemic_break_flags"]]
    current_flags = [str(item) for item in current_metrics["systemic_break_flags"]]
    new_systemic_break, new_systemic_break_flags = _new_systemic_break(
        baseline_flags,
        current_flags,
    )

    scorecard: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": SCORECARD_ARTIFACT_KIND,
        "branch": BRANCH,
        "branch_key": BRANCH_KEY,
        "branch_scope": gr00t_ladder_policy_gate.BRANCH_SPECS[BRANCH].branch_scope,
        "public_anchor_comparable": True,
        "axis": AXIS,
        "rung": normalized_rung,
        "rung_order": list(RUNG_ORDER),
        "rung_description": RUNG_DESCRIPTIONS[normalized_rung],
        "status": status,
        "promotion_status": "BLOCK"
        if status == "BLOCK"
        else str(promotion_report.get("promotion_status", "PASS")),
        "promotion_allowed": bool(status != "BLOCK")
        and bool(promotion_report.get("promotion_allowed", False)),
        "blocking_reasons": sorted(dict.fromkeys(blocking_reasons)),
        "failure_reasons": sorted(dict.fromkeys(failure_reasons)),
        "manifest_path": _rel_repo(manifest_path),
        "output_path": _rel_repo(scorecard_path),
        "comparability": comparability,
        "promotion_gate": promotion_report,
        "p3_gate": p3_gate_report,
        "baseline_rung": "P0",
        "frozen_formal_protocol": {
            "seed_values": [
                int(seed)
                for seed in gr00t_eval_contract_gate.DEFAULT_FORMAL_SEED_VALUES
            ],
            "scene_pool_identifier": gr00t_eval_contract_gate.DEFAULT_SCENE_POOL_IDENTIFIER,
            "env_name": gr00t_eval_contract_gate.DEFAULT_ENV_NAME,
            "max_episode_steps": int(
                gr00t_eval_contract_gate.DEFAULT_MAX_EPISODE_STEPS
            ),
            "n_action_steps": int(gr00t_eval_contract_gate.DEFAULT_N_ACTION_STEPS),
            "policy_horizon_expected": int(
                gr00t_eval_contract_gate.DEFAULT_POLICY_HORIZON_EXPECTED
            ),
            "n_episodes": int(gr00t_eval_contract_gate.DEFAULT_N_EPISODES),
            "n_envs": int(gr00t_eval_contract_gate.DEFAULT_N_ENVS),
            "formal_scene_ids": [
                str(item)
                for item in _as_list(
                    teacher_reachability.get("reachable_scene_ids", []),
                    field_name="teacher_reachability.reachable_scene_ids",
                )
            ],
        },
        "success_count": int(current_metrics["success_count"]),
        "success_rate": _round_float(float(current_metrics["success_rate"])),
        "condition_flip_response_ratio": _round_float(
            float(current_metrics["condition_flip_response_ratio"])
        ),
        "teacher_student_gap": _round_float(
            float(current_metrics["teacher_student_gap"])
        ),
        "action_chain_problem_group_count": int(
            current_metrics["action_chain_problem_group_count"]
        ),
        "provenance_status": str(current_metrics["provenance_status"]),
        "condition_flip_delta": _round_float(
            float(current_metrics["condition_flip_response_ratio"])
            - float(baseline_metrics["condition_flip_response_ratio"])
        ),
        "teacher_student_gap_delta": _round_float(
            float(current_metrics["teacher_student_gap"])
            - float(baseline_metrics["teacher_student_gap"])
        ),
        "action_chain_delta": _build_action_chain_delta(
            baseline_metrics,
            current_metrics,
        ),
        "positive_slope_report": positive_slope_report,
        "systemic_break_flags": current_flags,
        "new_systemic_break_detected": new_systemic_break,
        "new_systemic_break_flags": new_systemic_break_flags,
        "baseline_metrics": {
            "success_count": int(baseline_metrics["success_count"]),
            "success_rate": _round_float(float(baseline_metrics["success_rate"])),
            "condition_flip_response_ratio": _round_float(
                float(baseline_metrics["condition_flip_response_ratio"])
            ),
            "teacher_student_gap": _round_float(
                float(baseline_metrics["teacher_student_gap"])
            ),
            "action_chain_problem_group_count": int(
                baseline_metrics["action_chain_problem_group_count"]
            ),
            "provenance_status": str(baseline_metrics["provenance_status"]),
        },
        "source_artifacts": {
            key: dict(value)
            for key, value in cast(
                Mapping[str, Mapping[str, Any]],
                prerequisites.get("diagnostics_prerequisite_hashes", {}),
            ).items()
        },
        "preflight_prerequisite_proof": copy.deepcopy(
            dict(preflight_prerequisite_proof)
        ),
    }
    scorecard["report_signature_sha256"] = _sha256(scorecard)

    comparison_surface_digest = _sha256(rung_surface)
    manifest: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": MANIFEST_ARTIFACT_KIND,
        "branch": BRANCH,
        "branch_key": BRANCH_KEY,
        "branch_scope": gr00t_ladder_policy_gate.BRANCH_SPECS[BRANCH].branch_scope,
        "public_anchor_comparable": True,
        "axis": AXIS,
        "rung": normalized_rung,
        "rung_order": list(RUNG_ORDER),
        "status": status,
        "promotion_status": str(scorecard["promotion_status"]),
        "blocking_reasons": list(scorecard["blocking_reasons"]),
        "failure_reasons": list(scorecard["failure_reasons"]),
        "output_path": _rel_repo(manifest_path),
        "scorecard_path": _rel_repo(scorecard_path),
        "scorecard_signature_sha256": str(scorecard["report_signature_sha256"]),
        "rung_description": RUNG_DESCRIPTIONS[normalized_rung],
        "p_ladder_policy_gate_path": cast(
            Mapping[str, Any], prerequisites["diagnostics_prerequisite_hashes"]
        )["p_ladder_policy_gate"]["path"],
        "allowed_variable_parameter_fields": list(
            cast(Sequence[str], p_gate_payload.get("allowed_difference_paths", []))
        ),
        "comparison_surface_digest": comparison_surface_digest,
        "comparability": comparability,
        "checkpoint_provenance": {
            "formal_eligibility": checkpoint_provenance.get("formal_eligibility"),
            "status": checkpoint_provenance.get("status"),
            "selected_checkpoint_path": checkpoint_provenance.get(
                "selected_checkpoint_path"
            ),
            "loadability_status": checkpoint_provenance.get("loadability_status"),
            "checksum_or_signature": checkpoint_provenance.get("checksum_or_signature"),
        },
        "training_budget": copy.deepcopy(rung_surface["training"]),
        "frozen_data_surface": {
            "dataset": copy.deepcopy(rung_surface["dataset"]),
            "prompt_interface": copy.deepcopy(rung_surface["prompt_interface"]),
            "controller": copy.deepcopy(rung_surface["controller"]),
            "embodiment": copy.deepcopy(rung_surface["embodiment"]),
        },
        "seed_manifest": {
            "seed_values": [
                int(seed)
                for seed in gr00t_eval_contract_gate.DEFAULT_FORMAL_SEED_VALUES
            ],
            "seed_count": int(len(gr00t_eval_contract_gate.DEFAULT_FORMAL_SEED_VALUES)),
            "seed_policy": "fixed_formal_seed_manifest",
        },
        "scene_pool": {
            "scene_pool_identifier": gr00t_eval_contract_gate.DEFAULT_SCENE_POOL_IDENTIFIER,
            "reachable_scene_ids": [
                str(item)
                for item in _as_list(
                    teacher_reachability.get("reachable_scene_ids", []),
                    field_name="teacher_reachability.reachable_scene_ids",
                )
            ],
            "scene_rows": [
                dict(
                    _as_mapping(
                        item,
                        field_name="teacher_reachability.scene_pool.scene_rows[]",
                    )
                )
                for item in _as_list(
                    _as_mapping(
                        teacher_reachability.get("scene_pool", {}),
                        field_name="teacher_reachability.scene_pool",
                    ).get("scene_rows", []),
                    field_name="teacher_reachability.scene_pool.scene_rows",
                )
            ],
            "scene_pool_status": teacher_reachability.get("scene_pool_status"),
        },
        "diagnostics_prerequisite_hashes": copy.deepcopy(
            prerequisites.get("diagnostics_prerequisite_hashes", {})
        ),
        "preflight_prerequisite_proof": copy.deepcopy(
            dict(preflight_prerequisite_proof)
        ),
        "positive_slope_report": copy.deepcopy(scorecard["positive_slope_report"]),
        "p3_gate": copy.deepcopy(scorecard["p3_gate"]),
        "data_axis_frozen": True,
        "data_axis_non_drift_fields": [
            "dataset.dataset_path",
            "dataset.dataset_fingerprint",
            "dataset.dataset_mix",
            "prompt_interface.prompt_template_id",
            "prompt_interface.condition_injection",
            "controller.controller_family",
            "controller.action_horizon",
            "embodiment.embodiment_tag",
        ],
        "decision_basis": (
            "Use task-12 P-axis whitelist as the only allowed change surface; freeze dataset, prompt, "
            "controller, embodiment, seeds, scene pool, and prerequisite hashes across all UNITREE_G1 P rungs; "
            "only unlock P3 when P1 or P2 shows positive slope under the fixed formal protocol."
        ),
    }
    manifest["report_signature_sha256"] = _sha256(manifest)
    return {
        "manifest": manifest,
        "scorecard": scorecard,
        "manifest_path": manifest_path,
        "scorecard_path": scorecard_path,
    }


def materialize_p_ladder_rung(
    *,
    rung: str,
    output_root: Path,
    p_ladder_policy_gate_json: Path,
    dual_branch_scorecard_json: Path,
    checkpoint_provenance_json: Path,
    condition_flip_json: Path,
    teacher_student_gap_json: Path,
    action_telemetry_json: Path,
    teacher_reachability_json: Path,
    task2_preflight_evidence_json: Path,
    metrics_overrides_by_rung: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_output_root = _validate_output_dir(output_root)
    prerequisites = load_prerequisites(
        p_ladder_policy_gate_json=p_ladder_policy_gate_json,
        dual_branch_scorecard_json=dual_branch_scorecard_json,
        checkpoint_provenance_json=checkpoint_provenance_json,
        condition_flip_json=condition_flip_json,
        teacher_student_gap_json=teacher_student_gap_json,
        action_telemetry_json=action_telemetry_json,
        teacher_reachability_json=teacher_reachability_json,
        task2_preflight_evidence_json=task2_preflight_evidence_json,
    )
    report = build_rung_report(
        rung=rung,
        prerequisites=prerequisites,
        output_root=resolved_output_root,
        metrics_overrides_by_rung=metrics_overrides_by_rung,
    )
    manifest_path = cast(Path, report["manifest_path"])
    scorecard_path = cast(Path, report["scorecard_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _ = _write_json(manifest_path, cast(Mapping[str, Any], report["manifest"]))
    _ = _write_json(scorecard_path, cast(Mapping[str, Any], report["scorecard"]))
    scorecard = cast(Mapping[str, Any], report["scorecard"])
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": SCORECARD_ARTIFACT_KIND,
        "branch": BRANCH,
        "rung": _validate_rung(rung),
        "status": scorecard["status"],
        "promotion_status": scorecard["promotion_status"],
        "manifest_path": str(manifest_path),
        "scorecard_path": str(scorecard_path),
        "blocking_reasons": list(scorecard.get("blocking_reasons", [])),
        "failure_reasons": list(scorecard.get("failure_reasons", [])),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = materialize_p_ladder_rung(
            rung=str(args.rung),
            output_root=args.output_root,
            p_ladder_policy_gate_json=args.p_ladder_policy_gate_json,
            dual_branch_scorecard_json=args.dual_branch_scorecard_json,
            checkpoint_provenance_json=args.checkpoint_provenance_json,
            condition_flip_json=args.condition_flip_json,
            teacher_student_gap_json=args.teacher_student_gap_json,
            action_telemetry_json=args.action_telemetry_json,
            teacher_reachability_json=args.teacher_reachability_json,
            task2_preflight_evidence_json=args.task2_preflight_evidence_json,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if str(result.get("status")) != "BLOCK" else 1


__all__ = [
    "AXIS",
    "BASE_TRAINING_SURFACE",
    "BRANCH",
    "BRANCH_KEY",
    "DEFAULT_ACTION_TELEMETRY_JSON",
    "DEFAULT_CHECKPOINT_PROVENANCE_JSON",
    "DEFAULT_CONDITION_FLIP_JSON",
    "DEFAULT_DUAL_BRANCH_SCORECARD_JSON",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_P_LADDER_POLICY_GATE_JSON",
    "DEFAULT_TASK2_PREFLIGHT_EVIDENCE_JSON",
    "DEFAULT_TEACHER_REACHABILITY_JSON",
    "DEFAULT_TEACHER_STUDENT_GAP_JSON",
    "MANIFEST_ARTIFACT_KIND",
    "REPORT_SCHEMA_VERSION",
    "RUNG_DESCRIPTIONS",
    "RUNG_ORDER",
    "RUNG_SURFACE_PATCHES",
    "SCORECARD_ARTIFACT_KIND",
    "build_comparison_surface",
    "build_frozen_data_surface",
    "build_parser",
    "build_rung_report",
    "build_training_surface_for_rung",
    "load_prerequisites",
    "load_preflight_prerequisite_proof",
    "main",
    "materialize_p_ladder_rung",
]


if __name__ == "__main__":
    raise SystemExit(main())
