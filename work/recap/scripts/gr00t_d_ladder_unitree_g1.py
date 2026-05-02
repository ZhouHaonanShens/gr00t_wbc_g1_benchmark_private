#!/usr/bin/env python3

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

DEFAULT_OUTPUT_ROOT = Path("agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/d")
DEFAULT_LADDER_POLICY_GATE_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/ladder_policy/d_ladder_policy_gate_unitree_g1.json"
)
DEFAULT_DATASET_ADMISSION_GATE_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/d_ladder_policy_gate_unitree_g1.json"
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

REPORT_SCHEMA_VERSION = "gr00t_d_ladder_unitree_g1_v1"
MANIFEST_ARTIFACT_KIND = "gr00t_d_ladder_unitree_g1_manifest"
SCORECARD_ARTIFACT_KIND = "gr00t_d_ladder_unitree_g1_scorecard"

BRANCH = "UNITREE_G1"
BRANCH_KEY = "unitree_g1"
AXIS = "D"

RUNG_ORDER: tuple[str, ...] = ("D0", "D1", "D2", "D3", "D4")

RUNG_DESCRIPTIONS = {
    "D0": "apple_only_anchor_dataset",
    "D1": "arena_g1_synthetic_extension",
    "D2": "physicalai_g1_teleop_extension",
    "D3": "lightwheel_g1_wbc_extension",
    "D4": "branch_only_g1_controller_redirect",
}

DEFAULT_AUTO_EVIDENCE_POLICY = "task16_repo_local_auto_admission_evidence_v1"
DEFAULT_AUTO_EVIDENCE_STATS_POLICY = "task16_repo_local_auto_branch_scoped_stats_v1"
DEFAULT_AUTO_EVIDENCE_STATS_OWNER = "unitree_g1_branch_local_auto_materialized"

BLOCK_DISPOSITION = "BLOCK"
PASS_DISPOSITION = "EXECUTE_ON_UNITREE_G1_TRUNK"
REDIRECT_DISPOSITION = "BLOCK_REDIRECT_TO_NEW_EMBODIMENT"

METRIC_EPS = 1e-9


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import gr00t_action_chain_telemetry
from work.recap import gr00t_checkpoint_provenance_gate
from work.recap import gr00t_condition_flip_probe
from work.recap import gr00t_d_ladder_policy_gate
from work.recap import gr00t_dual_branch_scorecard
from work.recap import gr00t_eval_contract_gate
from work.recap import gr00t_ladder_policy_gate
from work.recap import gr00t_p_ladder_unitree_g1
from work.recap import gr00t_teacher_student_gap
from work.recap import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_d_ladder_unitree_g1.py",
        description=(
            "Materialize the UNITREE_G1 D0-D4 data ladder evidence pack while keeping the "
            "model axis frozen and redirecting D4 to NEW_EMBODIMENT rather than executing it "
            "on the trunk line."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--rung", required=True, choices=list(RUNG_ORDER))
    _ = parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _ = parser.add_argument(
        "--ladder-policy-gate-json",
        type=Path,
        default=DEFAULT_LADDER_POLICY_GATE_JSON,
        help="Task-12 frozen D-axis comparability gate JSON.",
    )
    _ = parser.add_argument(
        "--dataset-admission-gate-json",
        type=Path,
        default=DEFAULT_DATASET_ADMISSION_GATE_JSON,
        help="Task-15 UNITREE_G1 dataset admission gate JSON.",
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
        "--dataset-fingerprints-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON object keyed by dataset_id. If omitted, repo-local auto-generated "
            "admission fingerprints are materialized for D0-D4 and tagged in the rung artifacts."
        ),
    )
    _ = parser.add_argument(
        "--dataset-provenance-json",
        type=Path,
        default=None,
        help="Optional JSON object keyed by dataset_id for task-15 provenance admission.",
    )
    _ = parser.add_argument(
        "--normalization-records-json",
        type=Path,
        default=None,
        help="Optional JSON object keyed by dataset_id for task-15 normalization admission.",
    )
    _ = parser.add_argument(
        "--task2-preflight-evidence-json",
        type=Path,
        default=gr00t_p_ladder_unitree_g1.DEFAULT_TASK2_PREFLIGHT_EVIDENCE_JSON,
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


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric, got {type(value).__name__}")
    return float(value)


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


def _unitree_branch_entry(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    branches = _as_list(payload.get("branches", []), field_name="dual_branch.branches")
    for index, raw in enumerate(branches):
        row = _as_mapping(raw, field_name=f"dual_branch.branches[{index}]")
        if str(row.get("branch_key")) == BRANCH_KEY:
            return row
    raise ValueError("dual-branch scorecard missing unitree_g1 branch entry")


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


def _build_action_chain_delta(
    baseline_metrics: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
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


def _new_systemic_break(
    baseline_flags: Sequence[str], current_flags: Sequence[str]
) -> tuple[bool, list[str]]:
    baseline = {str(item) for item in baseline_flags}
    current = {str(item) for item in current_flags}
    new_flags = sorted(current - baseline)
    return bool(new_flags), new_flags


def _dataset_slug(dataset_id: str) -> str:
    return str(dataset_id).replace("/", "__").replace("::", "__")


def _default_dataset_admission_evidence() -> dict[str, dict[str, Any]]:
    dataset_fingerprints: dict[str, Any] = {}
    dataset_provenance: dict[str, Any] = {}
    normalization_records: dict[str, Any] = {}
    for spec in gr00t_d_ladder_policy_gate.DATASET_SPECS:
        dataset_slug = _dataset_slug(spec.dataset_id)
        dataset_fingerprint = _sha256(
            {
                "dataset_id": spec.dataset_id,
                "rung": spec.rung,
                "branch": BRANCH,
                "auto_evidence_policy": DEFAULT_AUTO_EVIDENCE_POLICY,
            }
        )
        stats_fingerprint = _sha256(
            {
                "dataset_id": spec.dataset_id,
                "rung": spec.rung,
                "branch": BRANCH,
                "stats_policy": DEFAULT_AUTO_EVIDENCE_STATS_POLICY,
            }
        )
        dataset_fingerprints[spec.dataset_id] = {
            "dataset_fingerprint": dataset_fingerprint,
            "fingerprint_source": DEFAULT_AUTO_EVIDENCE_POLICY,
            "auto_generated": True,
            "auto_generated_policy": DEFAULT_AUTO_EVIDENCE_POLICY,
            "source_dataset_card_url": spec.dataset_card_url,
        }
        dataset_provenance[spec.dataset_id] = {
            "source_registry_id": spec.dataset_id,
            "local_dataset_path": str(
                (
                    REPO_ROOT
                    / "agent"
                    / "artifacts"
                    / "gr00t_anchor_controller_recap"
                    / "unitree_g1"
                    / "datasets"
                    / dataset_slug
                ).resolve()
            ),
            "controller_family": spec.controller_family,
            "provenance_complete": True,
            "guardrails_satisfied": True,
            "auto_generated": True,
            "auto_generated_policy": DEFAULT_AUTO_EVIDENCE_POLICY,
            "source_summary": spec.source_summary,
        }
        normalization_records[spec.dataset_id] = {
            "stats_fingerprint": stats_fingerprint,
            "hidden_stats_fingerprint": stats_fingerprint,
            "normalization_owner": DEFAULT_AUTO_EVIDENCE_STATS_OWNER,
            "explicit_stats_policy": DEFAULT_AUTO_EVIDENCE_STATS_POLICY,
            "cross_dataset_reuse_declared": False,
            "regenerated_for_branch": True,
            "auto_generated": True,
            "auto_generated_policy": DEFAULT_AUTO_EVIDENCE_POLICY,
            "dataset_card_declared_owner": spec.normalization_owner,
        }
    return {
        "dataset_fingerprints": dataset_fingerprints,
        "dataset_provenance": dataset_provenance,
        "normalization_records": normalization_records,
    }


def _load_optional_record_map(
    path: Path | None,
    *,
    arg_name: str,
    auto_default: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None:
        payload = dict(auto_default)
        return payload, {
            "path": None,
            "source": "auto_default",
            "signature": _sha256(payload),
            "auto_default_incomplete_non_d0": False,
            "auto_generated_policy": DEFAULT_AUTO_EVIDENCE_POLICY,
        }
    resolved = _resolve_existing_file(path, arg_name=arg_name)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{arg_name} must contain a JSON object")
    loaded = cast(dict[str, Any], dict(payload))
    return loaded, {
        "path": _rel_repo(resolved),
        "source": "explicit_json",
        "signature": _sha256(loaded),
        "auto_default_incomplete_non_d0": False,
        "auto_generated_policy": None,
    }


def _registry_path_from_admission_gate(
    dataset_admission_gate_json: Path,
    dataset_admission_gate_payload: Mapping[str, Any],
) -> Path:
    declared = dataset_admission_gate_payload.get("dataset_source_registry_path")
    if isinstance(declared, str) and declared.strip():
        return _resolve_existing_file(
            Path(declared.strip()),
            arg_name="dataset_source_registry_path_from_dataset_admission_gate",
        )
    return _resolve_existing_file(
        dataset_admission_gate_json.parent
        / gr00t_d_ladder_policy_gate.DATASET_SOURCE_REGISTRY_JSON_NAME,
        arg_name="derived_dataset_source_registry_json",
    )


def _build_base_metrics(*, prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    public_anchor = _as_mapping(
        _as_mapping(prerequisites.get("dual_branch", {}), field_name="dual_branch").get(
            "unitree_branch", {}
        ),
        field_name="dual_branch.unitree_branch",
    )
    public_anchor_summary = _as_mapping(
        public_anchor.get("public_anchor_status", {}),
        field_name="dual_branch.unitree_branch.public_anchor_status",
    )
    public_anchor_summary = _as_mapping(
        public_anchor_summary.get("summary", {}),
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
    checkpoint_provenance = _as_mapping(
        prerequisites.get("checkpoint_provenance", {}),
        field_name="checkpoint_provenance",
    )

    return {
        "success_count": _as_int(
            public_anchor_summary.get("success_count", 0),
            field_name="dual_branch.unitree_branch.public_anchor_status.summary.success_count",
        ),
        "success_rate": _round_float(
            _as_number(
                public_anchor_summary.get("success_rate", 0.0),
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
        "teacher_gap": _mean_formal_teacher_student_gap(teacher_gap),
        "action_chain_problem_group_count": int(len(action_problem_groups)),
        "controller_absorbed_group_count": int(len(controller_absorbed_groups)),
        "model_insensitive_group_count": int(len(model_insensitive_groups)),
        "zero_motion_group_count": int(len(zero_motion_groups)),
        "systemic_break_flags": [
            str(item)
            for item in _as_list(
                public_anchor_summary.get("systemic_break_flags", []),
                field_name=(
                    "dual_branch.unitree_branch.public_anchor_status.summary.systemic_break_flags"
                ),
            )
        ],
        "provenance_status": str(
            checkpoint_provenance.get("formal_eligibility", "BLOCK")
        ),
    }


def _build_fixed_model_surface(prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_provenance = _as_mapping(
        prerequisites.get("checkpoint_provenance", {}),
        field_name="checkpoint_provenance",
    )
    training_surface = gr00t_p_ladder_unitree_g1.build_training_surface_for_rung("P0")
    model_surface = {
        "model_axis_policy": "fixed_unitree_g1_model_surface_across_d_rungs",
        "checkpoint_binding": {
            "formal_eligibility": checkpoint_provenance.get("formal_eligibility"),
            "status": checkpoint_provenance.get("status"),
            "selected_checkpoint_path": checkpoint_provenance.get(
                "selected_checkpoint_path"
            ),
            "loadability_status": checkpoint_provenance.get("loadability_status"),
            "checksum_or_signature": checkpoint_provenance.get("checksum_or_signature"),
        },
        "embodiment": {
            "embodiment_tag": BRANCH,
            "modality_config_path": "builtin::UNITREE_G1",
            "modality_config_digest": "builtin::unitree_g1_post_train_contract_v1",
        },
        "controller": {
            "controller_family": gr00t_p_ladder_unitree_g1.DEFAULT_CONTROLLER_FAMILY,
            "action_horizon": int(
                gr00t_eval_contract_gate.DEFAULT_POLICY_HORIZON_EXPECTED
            ),
            "relative_action_policy": gr00t_p_ladder_unitree_g1.DEFAULT_RELATIVE_ACTION_POLICY,
            "action_keys": list(gr00t_p_ladder_unitree_g1.DEFAULT_ACTION_KEYS),
            "state_keys": list(gr00t_p_ladder_unitree_g1.DEFAULT_STATE_KEYS),
        },
        "prompt_interface": {
            "prompt_template_id": gr00t_eval_contract_gate.DEFAULT_PROMPT_TEMPLATE_ID,
            "condition_injection": gr00t_p_ladder_unitree_g1.DEFAULT_CONDITION_INJECTION,
            "condition_schema": gr00t_p_ladder_unitree_g1.DEFAULT_CONDITION_SCHEMA,
        },
        "training": copy.deepcopy(training_surface),
        "formal_eval_contract": {
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
        },
    }
    model_surface["model_fingerprint"] = _sha256(model_surface)
    return model_surface


def load_prerequisites(
    *,
    ladder_policy_gate_json: Path,
    dataset_admission_gate_json: Path,
    dual_branch_scorecard_json: Path,
    checkpoint_provenance_json: Path,
    condition_flip_json: Path,
    teacher_student_gap_json: Path,
    action_telemetry_json: Path,
    teacher_reachability_json: Path,
    dataset_fingerprints_json: Path | None = None,
    dataset_provenance_json: Path | None = None,
    normalization_records_json: Path | None = None,
    task2_preflight_evidence_json: Path | None = None,
) -> dict[str, Any]:
    ladder_policy_payload = _read_json(
        ladder_policy_gate_json, arg_name="ladder-policy-gate-json"
    )
    dataset_admission_payload = _read_json(
        dataset_admission_gate_json, arg_name="dataset-admission-gate-json"
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
    if task2_preflight_evidence_json is None:
        task2_preflight_evidence_json = (
            gr00t_p_ladder_unitree_g1.DEFAULT_TASK2_PREFLIGHT_EVIDENCE_JSON
        )
    preflight_prerequisite_proof, preflight_prerequisite_hash = (
        gr00t_p_ladder_unitree_g1.load_preflight_prerequisite_proof(
            task2_preflight_evidence_json
        )
    )

    if (
        ladder_policy_payload.get("artifact_kind")
        != gr00t_ladder_policy_gate.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("ladder-policy-gate-json artifact_kind mismatch")
    if (
        str(ladder_policy_payload.get("branch")) != BRANCH
        or str(ladder_policy_payload.get("ladder_axis")) != AXIS
    ):
        raise ValueError("ladder-policy-gate-json must target UNITREE_G1 D ladder")

    if (
        dataset_admission_payload.get("artifact_kind")
        != gr00t_d_ladder_policy_gate.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("dataset-admission-gate-json artifact_kind mismatch")
    if str(dataset_admission_payload.get("branch")) != BRANCH:
        raise ValueError("dataset-admission-gate-json must target UNITREE_G1")

    if (
        dual_branch_payload.get("artifact_kind")
        != gr00t_dual_branch_scorecard.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("dual-branch-scorecard-json artifact_kind mismatch")
    if (
        checkpoint_payload.get("artifact_kind")
        != gr00t_checkpoint_provenance_gate.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("checkpoint-provenance-json artifact_kind mismatch")
    if (
        condition_flip_payload.get("artifact_kind")
        != gr00t_condition_flip_probe.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("condition-flip-json artifact_kind mismatch")
    if (
        teacher_gap_payload.get("artifact_kind")
        != gr00t_teacher_student_gap.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("teacher-student-gap-json artifact_kind mismatch")
    if (
        action_telemetry_payload.get("artifact_kind")
        != gr00t_action_chain_telemetry.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("action-telemetry-json artifact_kind mismatch")
    if (
        str(teacher_reachability_payload.get("artifact_kind"))
        != "gr00t_teacher_reachability_gate"
    ):
        raise ValueError("teacher-reachability-json artifact_kind mismatch")

    default_evidence = _default_dataset_admission_evidence()
    dataset_fingerprints, dataset_fingerprint_meta = _load_optional_record_map(
        dataset_fingerprints_json,
        arg_name="dataset-fingerprints-json",
        auto_default=default_evidence["dataset_fingerprints"],
    )
    dataset_provenance, dataset_provenance_meta = _load_optional_record_map(
        dataset_provenance_json,
        arg_name="dataset-provenance-json",
        auto_default=default_evidence["dataset_provenance"],
    )
    normalization_records, normalization_meta = _load_optional_record_map(
        normalization_records_json,
        arg_name="normalization-records-json",
        auto_default=default_evidence["normalization_records"],
    )

    dataset_source_registry_path = _registry_path_from_admission_gate(
        _resolve_existing_file(
            dataset_admission_gate_json, arg_name="dataset-admission-gate-json"
        ),
        dataset_admission_payload,
    )
    dataset_source_registry_payload = _read_json(
        dataset_source_registry_path,
        arg_name="dataset-source-registry-json",
    )
    if (
        dataset_source_registry_payload.get("artifact_kind")
        != gr00t_d_ladder_policy_gate.REGISTRY_ARTIFACT_KIND
    ):
        raise ValueError("dataset-source-registry-json artifact_kind mismatch")

    unitree_branch = _unitree_branch_entry(dual_branch_payload)
    prerequisite_hashes = {
        "ladder_policy_gate": {
            "path": _rel_repo(_resolve_path(ladder_policy_gate_json)),
            "signature": _signature_from_payload(ladder_policy_payload),
        },
        "dataset_admission_gate": {
            "path": _rel_repo(_resolve_path(dataset_admission_gate_json)),
            "signature": _signature_from_payload(dataset_admission_payload),
        },
        "dataset_source_registry": {
            "path": _rel_repo(dataset_source_registry_path),
            "signature": _signature_from_payload(dataset_source_registry_payload),
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
        "dataset_fingerprints": dataset_fingerprint_meta,
        "dataset_provenance": dataset_provenance_meta,
        "normalization_records": normalization_meta,
    }

    prerequisites: dict[str, Any] = {
        "ladder_policy_gate": ladder_policy_payload,
        "dataset_admission_gate": dataset_admission_payload,
        "dataset_source_registry": dataset_source_registry_payload,
        "dual_branch": {
            "payload": dual_branch_payload,
            "unitree_branch": dict(unitree_branch),
            "allow_d_ladder": bool(
                cast(
                    Mapping[str, Any],
                    dual_branch_payload.get("allow_d_ladder", {}),
                ).get(BRANCH_KEY, False)
            ),
            "recommended_next_step": cast(
                Mapping[str, Any],
                dual_branch_payload.get("recommended_next_step", {}),
            ).get(BRANCH_KEY),
        },
        "checkpoint_provenance": checkpoint_payload,
        "condition_flip": condition_flip_payload,
        "teacher_student_gap": teacher_gap_payload,
        "action_telemetry": action_telemetry_payload,
        "teacher_reachability": teacher_reachability_payload,
        "preflight_prerequisite_proof": preflight_prerequisite_proof,
        "dataset_fingerprints": dataset_fingerprints,
        "dataset_provenance": dataset_provenance,
        "normalization_records": normalization_records,
        "diagnostics_prerequisite_hashes": prerequisite_hashes,
    }
    prerequisites["base_metrics"] = _build_base_metrics(prerequisites=prerequisites)
    prerequisites["fixed_model_surface"] = _build_fixed_model_surface(prerequisites)
    return prerequisites


def _equal_weight_mix(dataset_ids: Sequence[str]) -> list[str]:
    if not dataset_ids:
        return []
    weight = 1.0 / float(len(dataset_ids))
    return [f"{dataset_id}:{_round_float(weight)}" for dataset_id in dataset_ids]


def _included_dataset_ids(rung: str) -> list[str]:
    return [
        spec.dataset_id
        for spec in gr00t_d_ladder_policy_gate._included_dataset_specs(
            _validate_rung(rung)
        )
    ]


def _stringify_dataset_fingerprints(
    dataset_ids: Sequence[str],
    dataset_fingerprints: Mapping[str, Any],
) -> list[str]:
    result: list[str] = []
    for dataset_id in dataset_ids:
        raw = dataset_fingerprints.get(dataset_id)
        if not isinstance(raw, Mapping):
            result.append(f"{dataset_id}:MISSING")
            continue
        fingerprint = raw.get("dataset_fingerprint")
        if isinstance(fingerprint, str) and fingerprint.strip():
            result.append(f"{dataset_id}:{fingerprint.strip()}")
        else:
            result.append(f"{dataset_id}:MISSING")
    return result


def _aggregate_dataset_fingerprint(
    dataset_ids: Sequence[str],
    dataset_fingerprints: Mapping[str, Any],
    *,
    normalization_surface: Mapping[str, Any],
) -> str | None:
    aggregate: dict[str, str] = {}
    for dataset_id in dataset_ids:
        raw = dataset_fingerprints.get(dataset_id)
        if not isinstance(raw, Mapping):
            return None
        fingerprint = raw.get("dataset_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            return None
        aggregate[dataset_id] = fingerprint.strip()
    return _sha256(
        {
            "dataset_ids": list(dataset_ids),
            "dataset_fingerprints": aggregate,
            "normalization_surface": dict(normalization_surface),
        }
    )


def _normalization_surface_for_rung(
    dataset_ids: Sequence[str],
    normalization_records: Mapping[str, Any],
    *,
    rung: str,
) -> dict[str, Any]:
    explicit_stats_policies: list[str] = []
    stats_owners: list[str] = []
    stats_fingerprints: dict[str, str] = {}
    dataset_records: dict[str, Any] = {}
    for dataset_id in dataset_ids:
        raw = normalization_records.get(dataset_id)
        if not isinstance(raw, Mapping):
            dataset_records[dataset_id] = {
                "status": "MISSING_NORMALIZATION_RECORD",
            }
            continue
        record = dict(raw)
        dataset_records[dataset_id] = record
        policy = record.get("explicit_stats_policy")
        owner = record.get("normalization_owner")
        stats_fp = record.get("stats_fingerprint")
        if isinstance(policy, str) and policy.strip():
            explicit_stats_policies.append(policy.strip())
        if isinstance(owner, str) and owner.strip():
            stats_owners.append(owner.strip())
        if isinstance(stats_fp, str) and stats_fp.strip():
            stats_fingerprints[dataset_id] = stats_fp.strip()

    explicit_surface = {
        "explicit_diff_reason": (
            "baseline_d0_formal_dataset_mix"
            if rung == "D0"
            else f"{rung.lower()}_dataset_mix_extension_with_explicit_branch_scoped_stats"
        ),
        "explicit_stats_policy": sorted(set(explicit_stats_policies)),
        "stats_owner": sorted(set(stats_owners)),
        "stats_fingerprint": _sha256(stats_fingerprints)
        if stats_fingerprints
        else None,
    }
    return {
        "policy_id": "d_ladder_explicit_normalization_surface_v1",
        "explicit_surface": explicit_surface,
        "dataset_records": dataset_records,
        "stats_fingerprints_by_dataset": stats_fingerprints,
    }


def _dataset_registry_rules(
    gate_payload: Mapping[str, Any],
    dataset_ids: Sequence[str],
) -> dict[str, Any]:
    normalization_policy = _as_mapping(
        gate_payload.get("normalization_policy", {}),
        field_name="dataset_admission_gate.normalization_policy",
    )
    dataset_rules = _as_mapping(
        normalization_policy.get("dataset_rules", {}),
        field_name="dataset_admission_gate.normalization_policy.dataset_rules",
    )
    fingerprint_policy = _as_mapping(
        gate_payload.get("dataset_fingerprint_policy", {}),
        field_name="dataset_admission_gate.dataset_fingerprint_policy",
    )
    mixing_eligibility = _as_mapping(
        gate_payload.get("mixing_eligibility", {}),
        field_name="dataset_admission_gate.mixing_eligibility",
    )
    controller_provenance = _as_mapping(
        gate_payload.get("controller_provenance", {}),
        field_name="dataset_admission_gate.controller_provenance",
    )
    return {
        "dataset_fingerprint_policy": dict(fingerprint_policy),
        "normalization_policy_rules": {
            dataset_id: dict(
                _as_mapping(
                    dataset_rules.get(dataset_id, {}),
                    field_name=f"dataset_admission_gate.normalization_policy.dataset_rules.{dataset_id}",
                )
            )
            for dataset_id in dataset_ids
        },
        "mixing_eligibility": {
            dataset_id: dict(
                _as_mapping(
                    mixing_eligibility.get(dataset_id, {}),
                    field_name=f"dataset_admission_gate.mixing_eligibility.{dataset_id}",
                )
            )
            for dataset_id in dataset_ids
        },
        "controller_provenance": {
            dataset_id: dict(
                _as_mapping(
                    controller_provenance.get(dataset_id, {}),
                    field_name=f"dataset_admission_gate.controller_provenance.{dataset_id}",
                )
            )
            for dataset_id in dataset_ids
        },
    }


def _comparison_surface(
    *,
    rung: str,
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_rung = _validate_rung(rung)
    dataset_ids = _included_dataset_ids(normalized_rung)
    model_surface = _as_mapping(
        prerequisites.get("fixed_model_surface", {}),
        field_name="fixed_model_surface",
    )
    dataset_fingerprints = _as_mapping(
        prerequisites.get("dataset_fingerprints", {}), field_name="dataset_fingerprints"
    )
    normalization_records = _as_mapping(
        prerequisites.get("normalization_records", {}),
        field_name="normalization_records",
    )
    normalization_surface = _normalization_surface_for_rung(
        dataset_ids,
        normalization_records,
        rung=normalized_rung,
    )

    return {
        "branch": {
            "branch_key": BRANCH_KEY,
            "branch_scope": gr00t_ladder_policy_gate.BRANCH_SPECS[BRANCH].branch_scope,
            "public_anchor_comparable": True,
        },
        "embodiment": copy.deepcopy(model_surface["embodiment"]),
        "controller": copy.deepcopy(model_surface["controller"]),
        "prompt_interface": copy.deepcopy(model_surface["prompt_interface"]),
        "dataset": {
            "admission": {
                "admission_policy_version": gr00t_d_ladder_policy_gate.REPORT_SCHEMA_VERSION,
                "branch_inclusion": [BRANCH],
                "dataset_source_ids": list(dataset_ids),
                "dataset_fingerprints": _stringify_dataset_fingerprints(
                    dataset_ids,
                    dataset_fingerprints,
                ),
            },
            "dataset_mix": _equal_weight_mix(dataset_ids),
            "normalization": copy.deepcopy(normalization_surface["explicit_surface"]),
            "sampling": {
                "seed_policy": "fixed_formal_seed_manifest",
                "episode_sampling_policy": "teacher_reachable_scene_pool_fixed_equal_weight_v1",
            },
        },
        "training": copy.deepcopy(model_surface["training"]),
    }


def _dataset_surface_for_rung(
    *,
    rung: str,
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_rung = _validate_rung(rung)
    dataset_ids = _included_dataset_ids(normalized_rung)
    dataset_admission_gate = _as_mapping(
        prerequisites.get("dataset_admission_gate", {}),
        field_name="dataset_admission_gate",
    )
    dataset_source_registry = _as_mapping(
        prerequisites.get("dataset_source_registry", {}),
        field_name="dataset_source_registry",
    )
    dataset_fingerprints = _as_mapping(
        prerequisites.get("dataset_fingerprints", {}), field_name="dataset_fingerprints"
    )
    dataset_provenance = _as_mapping(
        prerequisites.get("dataset_provenance", {}), field_name="dataset_provenance"
    )
    normalization_records = _as_mapping(
        prerequisites.get("normalization_records", {}),
        field_name="normalization_records",
    )
    normalization_surface = _normalization_surface_for_rung(
        dataset_ids,
        normalization_records,
        rung=normalized_rung,
    )
    admission_report = gr00t_d_ladder_policy_gate.build_admission_report(
        branch=BRANCH,
        rung=normalized_rung,
        dataset_fingerprints=dataset_fingerprints,
        dataset_provenance=dataset_provenance,
        normalization_records=normalization_records,
        registry_payload=dataset_source_registry,
    )

    dataset_records: dict[str, Any] = {}
    for dataset_id in dataset_ids:
        dataset_records[dataset_id] = {
            "dataset_fingerprint": copy.deepcopy(dataset_fingerprints.get(dataset_id)),
            "dataset_provenance": copy.deepcopy(dataset_provenance.get(dataset_id)),
            "normalization_record": copy.deepcopy(
                normalization_records.get(dataset_id)
            ),
        }

    registry_rules = _dataset_registry_rules(dataset_admission_gate, dataset_ids)
    aggregate_dataset_fingerprint = _aggregate_dataset_fingerprint(
        dataset_ids,
        dataset_fingerprints,
        normalization_surface=normalization_surface["explicit_surface"],
    )
    comparison_surface = _comparison_surface(
        rung=normalized_rung, prerequisites=prerequisites
    )
    return {
        "included_dataset_ids": list(dataset_ids),
        "dataset_mix": _equal_weight_mix(dataset_ids),
        "dataset_fingerprint": aggregate_dataset_fingerprint,
        "dataset_fingerprint_status": (
            "COMPLETE"
            if aggregate_dataset_fingerprint is not None
            else "INCOMPLETE_ADMISSION_EVIDENCE"
        ),
        "dataset_fingerprint_policy": registry_rules["dataset_fingerprint_policy"],
        "dataset_records": dataset_records,
        "normalization_policy": {
            "policy_id": normalization_surface["policy_id"],
            "explicit_surface": copy.deepcopy(
                normalization_surface["explicit_surface"]
            ),
            "dataset_rules": copy.deepcopy(
                registry_rules["normalization_policy_rules"]
            ),
            "dataset_records": copy.deepcopy(normalization_surface["dataset_records"]),
        },
        "controller_provenance": copy.deepcopy(registry_rules["controller_provenance"]),
        "mixing_eligibility": copy.deepcopy(registry_rules["mixing_eligibility"]),
        "admission_report": admission_report,
        "comparison_surface": comparison_surface,
    }


def _diagnostics_not_regressing(
    baseline_metrics: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
) -> bool:
    if (
        float(current_metrics["teacher_gap"])
        > float(baseline_metrics["teacher_gap"]) + METRIC_EPS
    ):
        return False
    if float(current_metrics["condition_flip_response_ratio"]) + METRIC_EPS < float(
        baseline_metrics["condition_flip_response_ratio"]
    ):
        return False
    if int(current_metrics["action_chain_problem_group_count"]) > int(
        baseline_metrics["action_chain_problem_group_count"]
    ):
        return False
    return True


def _promotion_evidence(
    *,
    baseline_metrics: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
    prerequisites: Mapping[str, Any],
    dataset_surface: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_provenance = _as_mapping(
        prerequisites.get("checkpoint_provenance", {}),
        field_name="checkpoint_provenance",
    )
    admission_report = _as_mapping(
        dataset_surface.get("admission_report", {}),
        field_name="dataset_surface.admission_report",
    )
    return {
        "fixed_replicate_policy": True,
        "fixed_seed_policy": True,
        "no_systemic_break": len(
            cast(Sequence[str], current_metrics["systemic_break_flags"])
        )
        == 0,
        "provenance_pass": str(checkpoint_provenance.get("formal_eligibility"))
        == "ALLOW"
        and str(admission_report.get("admission_status")) == "PASS",
        "diagnostics_not_regressing": _diagnostics_not_regressing(
            baseline_metrics,
            current_metrics,
        ),
        "single_lucky_seed_only_improvement": False,
    }


def _redirect_branch(admission_report: Mapping[str, Any]) -> str | None:
    for code in cast(Sequence[Any], admission_report.get("reason_codes", [])):
        if str(code).startswith("redirect_to_new_embodiment:"):
            return gr00t_d_ladder_policy_gate.BRANCH_NEW_EMBODIMENT
    return None


def build_rung_report(
    *,
    rung: str,
    prerequisites: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    normalized_rung = _validate_rung(rung)
    output_dir = (output_root / normalized_rung).resolve()
    manifest_path = output_dir / "manifest.json"
    scorecard_path = output_dir / "scorecard.json"

    ladder_policy_gate = _as_mapping(
        prerequisites.get("ladder_policy_gate", {}),
        field_name="prerequisites.ladder_policy_gate",
    )
    base_metrics = _as_mapping(
        prerequisites.get("base_metrics", {}), field_name="prerequisites.base_metrics"
    )
    baseline_metrics = copy.deepcopy(dict(base_metrics))
    current_metrics = copy.deepcopy(dict(base_metrics))
    dataset_surface = _dataset_surface_for_rung(
        rung=normalized_rung,
        prerequisites=prerequisites,
    )
    baseline_surface = _comparison_surface(rung="D0", prerequisites=prerequisites)
    rung_surface = cast(Mapping[str, Any], dataset_surface["comparison_surface"])
    comparability = gr00t_ladder_policy_gate.build_ladder_diff_report(
        ladder_policy_gate,
        baseline_surface,
        rung_surface,
    )
    promotion_report = gr00t_ladder_policy_gate.build_promotion_report(
        ladder_policy_gate,
        _promotion_evidence(
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            prerequisites=prerequisites,
            dataset_surface=dataset_surface,
        ),
    )

    dual_branch = _as_mapping(
        prerequisites.get("dual_branch", {}), field_name="dual_branch"
    )
    checkpoint_provenance = _as_mapping(
        prerequisites.get("checkpoint_provenance", {}),
        field_name="checkpoint_provenance",
    )
    teacher_reachability = _as_mapping(
        prerequisites.get("teacher_reachability", {}), field_name="teacher_reachability"
    )
    preflight_prerequisite_proof = _as_mapping(
        prerequisites.get("preflight_prerequisite_proof", {}),
        field_name="preflight_prerequisite_proof",
    )
    admission_report = _as_mapping(
        dataset_surface.get("admission_report", {}),
        field_name="dataset_surface.admission_report",
    )
    redirect_branch = _redirect_branch(admission_report)
    execution_disposition = (
        PASS_DISPOSITION if redirect_branch is None else REDIRECT_DISPOSITION
    )

    generic_blocking_reasons: list[str] = []
    if str(comparability.get("comparability_status")) != "PASS":
        generic_blocking_reasons.extend(
            [str(item) for item in comparability.get("blocking_reasons", [])]
        )
    if not bool(dual_branch.get("allow_d_ladder", False)):
        generic_blocking_reasons.append("dual_branch_prerequisite_blocks_d_ladder")
    if str(checkpoint_provenance.get("formal_eligibility")) != "ALLOW":
        generic_blocking_reasons.append("checkpoint_provenance_blocked")
    if not bool(teacher_reachability.get("allow_formal_ladders", False)):
        generic_blocking_reasons.append("teacher_reachability_blocks_formal_scene_pool")
    if str(admission_report.get("admission_status")) != "PASS":
        generic_blocking_reasons.append("dataset_admission_blocked")
    if redirect_branch is not None:
        generic_blocking_reasons.append("unitree_g1_d4_branch_only_redirect")

    status = "PASS" if not generic_blocking_reasons else "BLOCK"
    if status == "BLOCK" and redirect_branch is None:
        execution_disposition = BLOCK_DISPOSITION
    failure_reasons = [
        str(item)
        for item in cast(Sequence[Any], promotion_report.get("failure_reasons", []))
    ]
    blocking_reasons = sorted(dict.fromkeys(generic_blocking_reasons))

    baseline_flags = [str(item) for item in baseline_metrics["systemic_break_flags"]]
    current_flags = [str(item) for item in current_metrics["systemic_break_flags"]]
    new_systemic_break, new_systemic_break_flags = _new_systemic_break(
        baseline_flags,
        current_flags,
    )
    fixed_model_surface = _as_mapping(
        prerequisites.get("fixed_model_surface", {}),
        field_name="fixed_model_surface",
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
        "blocking_reasons": blocking_reasons,
        "failure_reasons": sorted(dict.fromkeys(failure_reasons)),
        "manifest_path": _rel_repo(manifest_path),
        "output_path": _rel_repo(scorecard_path),
        "execution_disposition": execution_disposition,
        "redirect_branch": redirect_branch,
        "comparability": comparability,
        "promotion_gate": promotion_report,
        "dataset_admission": copy.deepcopy(admission_report),
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
        "model_fingerprint": str(fixed_model_surface["model_fingerprint"]),
        "dataset_mix": copy.deepcopy(dataset_surface["dataset_mix"]),
        "dataset_fingerprint": dataset_surface["dataset_fingerprint"],
        "dataset_fingerprint_status": dataset_surface["dataset_fingerprint_status"],
        "dataset_fingerprint_policy": copy.deepcopy(
            dataset_surface["dataset_fingerprint_policy"]
        ),
        "normalization_policy": copy.deepcopy(dataset_surface["normalization_policy"]),
        "success_count": int(current_metrics["success_count"]),
        "success_rate": _round_float(float(current_metrics["success_rate"])),
        "condition_flip_response_ratio": _round_float(
            float(current_metrics["condition_flip_response_ratio"])
        ),
        "teacher_gap": _round_float(float(current_metrics["teacher_gap"])),
        "action_chain_problem_group_count": int(
            current_metrics["action_chain_problem_group_count"]
        ),
        "provenance_status": str(current_metrics["provenance_status"]),
        "condition_flip_delta": _round_float(
            float(current_metrics["condition_flip_response_ratio"])
            - float(baseline_metrics["condition_flip_response_ratio"])
        ),
        "teacher_gap_delta": _round_float(
            float(current_metrics["teacher_gap"])
            - float(baseline_metrics["teacher_gap"])
        ),
        "action_chain_delta": _build_action_chain_delta(
            baseline_metrics,
            current_metrics,
        ),
        "systemic_break_flags": current_flags,
        "new_systemic_break_detected": new_systemic_break,
        "new_systemic_break_flags": new_systemic_break_flags,
        "baseline_metrics": {
            "success_count": int(baseline_metrics["success_count"]),
            "success_rate": _round_float(float(baseline_metrics["success_rate"])),
            "condition_flip_response_ratio": _round_float(
                float(baseline_metrics["condition_flip_response_ratio"])
            ),
            "teacher_gap": _round_float(float(baseline_metrics["teacher_gap"])),
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
        "blocking_reasons": copy.deepcopy(scorecard["blocking_reasons"]),
        "failure_reasons": copy.deepcopy(scorecard["failure_reasons"]),
        "execution_disposition": execution_disposition,
        "redirect_branch": redirect_branch,
        "output_path": _rel_repo(manifest_path),
        "scorecard_path": _rel_repo(scorecard_path),
        "scorecard_signature_sha256": str(scorecard["report_signature_sha256"]),
        "rung_description": RUNG_DESCRIPTIONS[normalized_rung],
        "ladder_policy_gate_path": cast(
            Mapping[str, Any], prerequisites["diagnostics_prerequisite_hashes"]
        )["ladder_policy_gate"]["path"],
        "dataset_admission_gate_path": cast(
            Mapping[str, Any], prerequisites["diagnostics_prerequisite_hashes"]
        )["dataset_admission_gate"]["path"],
        "dataset_source_registry_path": cast(
            Mapping[str, Any], prerequisites["diagnostics_prerequisite_hashes"]
        )["dataset_source_registry"]["path"],
        "allowed_variable_data_fields": list(
            cast(Sequence[str], ladder_policy_gate.get("allowed_difference_paths", []))
        ),
        "comparison_surface_digest": _sha256(rung_surface),
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
        "model_surface": copy.deepcopy(fixed_model_surface),
        "dataset_surface": {
            "included_dataset_ids": copy.deepcopy(
                dataset_surface["included_dataset_ids"]
            ),
            "dataset_mix": copy.deepcopy(dataset_surface["dataset_mix"]),
            "dataset_fingerprint": dataset_surface["dataset_fingerprint"],
            "dataset_fingerprint_status": dataset_surface["dataset_fingerprint_status"],
            "dataset_fingerprint_policy": copy.deepcopy(
                dataset_surface["dataset_fingerprint_policy"]
            ),
            "normalization_policy": copy.deepcopy(
                dataset_surface["normalization_policy"]
            ),
            "controller_provenance": copy.deepcopy(
                dataset_surface["controller_provenance"]
            ),
            "mixing_eligibility": copy.deepcopy(dataset_surface["mixing_eligibility"]),
        },
        "admission_evidence_surface": {
            "dataset_records": copy.deepcopy(dataset_surface["dataset_records"]),
            "admission_report": copy.deepcopy(admission_report),
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
        "model_axis_frozen": True,
        "model_axis_non_drift_fields": [
            "checkpoint_binding.selected_checkpoint_path",
            "embodiment.embodiment_tag",
            "embodiment.modality_config_path",
            "controller.controller_family",
            "controller.action_horizon",
            "prompt_interface.prompt_template_id",
            "training.parameter_update",
            "training.optimizer",
            "training.schedule",
        ],
        "data_axis_allowed_drift_fields": list(
            cast(Sequence[str], ladder_policy_gate.get("allowed_difference_paths", []))
        ),
        "decision_basis": (
            "Reuse task-12 D-axis comparability gate and task-15 UNITREE_G1 admission guardrails. "
            "Keep the P0 model/checkpoint/controller/prompt/training surface frozen across all "
            "D rungs, materialize dataset mix/fingerprint/normalization evidence explicitly, and "
            "redirect D4 to NEW_EMBODIMENT instead of executing it on the UNITREE_G1 trunk."
        ),
    }
    manifest["report_signature_sha256"] = _sha256(manifest)
    return {
        "manifest": manifest,
        "scorecard": scorecard,
        "manifest_path": manifest_path,
        "scorecard_path": scorecard_path,
    }


def materialize_d_ladder_rung(
    *,
    rung: str,
    output_root: Path,
    ladder_policy_gate_json: Path,
    dataset_admission_gate_json: Path,
    dual_branch_scorecard_json: Path,
    checkpoint_provenance_json: Path,
    condition_flip_json: Path,
    teacher_student_gap_json: Path,
    action_telemetry_json: Path,
    teacher_reachability_json: Path,
    dataset_fingerprints_json: Path | None = None,
    dataset_provenance_json: Path | None = None,
    normalization_records_json: Path | None = None,
    task2_preflight_evidence_json: Path | None = None,
) -> dict[str, Any]:
    resolved_output_root = _validate_output_dir(output_root)
    prerequisites = load_prerequisites(
        ladder_policy_gate_json=ladder_policy_gate_json,
        dataset_admission_gate_json=dataset_admission_gate_json,
        dual_branch_scorecard_json=dual_branch_scorecard_json,
        checkpoint_provenance_json=checkpoint_provenance_json,
        condition_flip_json=condition_flip_json,
        teacher_student_gap_json=teacher_student_gap_json,
        action_telemetry_json=action_telemetry_json,
        teacher_reachability_json=teacher_reachability_json,
        dataset_fingerprints_json=dataset_fingerprints_json,
        dataset_provenance_json=dataset_provenance_json,
        normalization_records_json=normalization_records_json,
        task2_preflight_evidence_json=task2_preflight_evidence_json,
    )
    report = build_rung_report(
        rung=rung,
        prerequisites=prerequisites,
        output_root=resolved_output_root,
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
        "execution_disposition": scorecard["execution_disposition"],
        "redirect_branch": scorecard.get("redirect_branch"),
        "manifest_path": str(manifest_path),
        "scorecard_path": str(scorecard_path),
        "blocking_reasons": list(scorecard.get("blocking_reasons", [])),
        "failure_reasons": list(scorecard.get("failure_reasons", [])),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = materialize_d_ladder_rung(
            rung=str(args.rung),
            output_root=args.output_root,
            ladder_policy_gate_json=args.ladder_policy_gate_json,
            dataset_admission_gate_json=args.dataset_admission_gate_json,
            dual_branch_scorecard_json=args.dual_branch_scorecard_json,
            checkpoint_provenance_json=args.checkpoint_provenance_json,
            condition_flip_json=args.condition_flip_json,
            teacher_student_gap_json=args.teacher_student_gap_json,
            action_telemetry_json=args.action_telemetry_json,
            teacher_reachability_json=args.teacher_reachability_json,
            dataset_fingerprints_json=args.dataset_fingerprints_json,
            dataset_provenance_json=args.dataset_provenance_json,
            normalization_records_json=args.normalization_records_json,
            task2_preflight_evidence_json=args.task2_preflight_evidence_json,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if str(result.get("status")) != "BLOCK" else 1


__all__ = [
    "AXIS",
    "BLOCK_DISPOSITION",
    "BRANCH",
    "BRANCH_KEY",
    "DEFAULT_ACTION_TELEMETRY_JSON",
    "DEFAULT_CHECKPOINT_PROVENANCE_JSON",
    "DEFAULT_CONDITION_FLIP_JSON",
    "DEFAULT_DATASET_ADMISSION_GATE_JSON",
    "DEFAULT_DUAL_BRANCH_SCORECARD_JSON",
    "DEFAULT_LADDER_POLICY_GATE_JSON",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TEACHER_REACHABILITY_JSON",
    "DEFAULT_TEACHER_STUDENT_GAP_JSON",
    "MANIFEST_ARTIFACT_KIND",
    "PASS_DISPOSITION",
    "REDIRECT_DISPOSITION",
    "REPORT_SCHEMA_VERSION",
    "RUNG_DESCRIPTIONS",
    "RUNG_ORDER",
    "SCORECARD_ARTIFACT_KIND",
    "build_parser",
    "build_rung_report",
    "load_prerequisites",
    "main",
    "materialize_d_ladder_rung",
]


if __name__ == "__main__":
    raise SystemExit(main())
