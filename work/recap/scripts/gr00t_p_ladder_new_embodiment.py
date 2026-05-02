from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_ROOT = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/p"
)
DEFAULT_P_LADDER_POLICY_GATE_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/ladder_policy/p_ladder_policy_gate_new_embodiment.json"
)
DEFAULT_DUAL_BRANCH_SCORECARD_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/dual_branch_scorecard.json"
)
DEFAULT_CHECKPOINT_PROVENANCE_JSON = Path(
    "agent/artifacts/gr00t_checkpoint_provenance/checkpoint_provenance_report.json"
)
DEFAULT_CONDITION_FLIP_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/condition_flip_scorecard_new_embodiment.json"
)
DEFAULT_TEACHER_STUDENT_GAP_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/teacher_student_gap_scorecard_new_embodiment.json"
)
DEFAULT_ACTION_TELEMETRY_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/action_chain_telemetry_new_embodiment.json"
)
DEFAULT_TEACHER_REACHABILITY_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/teacher_reachability/teacher_reachability_gate_new_embodiment.json"
)
DEFAULT_MODALITY_CONFIG_PATH = Path("work/configs/new_embodiment/modality_config.json")
DEFAULT_BRANCH_MANIFEST_PATH = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/branch_manifest.json"
)
DEFAULT_CONTROLLER_AUDIT_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/controller_audit_new_embodiment.json"
)

REPORT_SCHEMA_VERSION = "gr00t_p_ladder_new_embodiment_v1"
MANIFEST_ARTIFACT_KIND = "gr00t_p_ladder_new_embodiment_manifest"
SCORECARD_ARTIFACT_KIND = "gr00t_p_ladder_new_embodiment_scorecard"

BRANCH = "NEW_EMBODIMENT"
BRANCH_KEY = "new_embodiment"
AXIS = "P"

RUNG_ORDER: tuple[str, ...] = ("P0", "P1", "P2", "P3")


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import gr00t_checkpoint_provenance_gate
from work.recap import gr00t_controller_audit_new_embodiment
from work.recap import gr00t_eval_contract_gate
from work.recap import gr00t_ladder_policy_gate
from work.recap import gr00t_p_ladder_unitree_g1


BASE_TRAINING_SURFACE = copy.deepcopy(gr00t_p_ladder_unitree_g1.BASE_TRAINING_SURFACE)
RUNG_SURFACE_PATCHES = copy.deepcopy(gr00t_p_ladder_unitree_g1.RUNG_SURFACE_PATCHES)
RUNG_DESCRIPTIONS = copy.deepcopy(gr00t_p_ladder_unitree_g1.RUNG_DESCRIPTIONS)
POSITIVE_SLOPE_METRIC_EPS = gr00t_p_ladder_unitree_g1.POSITIVE_SLOPE_METRIC_EPS

DEFAULT_DATASET_PATH = "new_embodiment.branch_local_custom_embodiment_dataset_contract"
DEFAULT_ADMISSION_POLICY_VERSION = "new_embodiment_formal_p_ladder_v1"
DEFAULT_CONDITION_SCHEMA = gr00t_p_ladder_unitree_g1.DEFAULT_CONDITION_SCHEMA
DEFAULT_CONDITION_INJECTION = gr00t_p_ladder_unitree_g1.DEFAULT_CONDITION_INJECTION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_p_ladder_new_embodiment.py",
        description=(
            "Materialize the NEW_EMBODIMENT P0-P3 parameter ladder manifests and scorecards "
            "under a frozen internal-only formal protocol."
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
        "--modality-config-path",
        type=Path,
        default=DEFAULT_MODALITY_CONFIG_PATH,
    )
    _ = parser.add_argument(
        "--branch-manifest-path",
        type=Path,
        default=DEFAULT_BRANCH_MANIFEST_PATH,
    )
    _ = parser.add_argument(
        "--controller-audit-json",
        type=Path,
        default=DEFAULT_CONTROLLER_AUDIT_JSON,
    )
    _ = parser.add_argument(
        "--task2-preflight-evidence-json",
        type=Path,
        default=gr00t_p_ladder_unitree_g1.DEFAULT_TASK2_PREFLIGHT_EVIDENCE_JSON,
    )
    return parser


def _normalization_source_signature(normalization_source: Mapping[str, Any]) -> str:
    return gr00t_p_ladder_unitree_g1._sha256(dict(normalization_source))


def _relative_action_policy_signature(relative_action_policy: Mapping[str, Any]) -> str:
    return gr00t_p_ladder_unitree_g1._sha256(dict(relative_action_policy))


def _new_embodiment_branch_entry(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    branches = gr00t_p_ladder_unitree_g1._as_list(
        payload.get("branches", []), field_name="dual_branch.branches"
    )
    for index, raw in enumerate(branches):
        row = gr00t_p_ladder_unitree_g1._as_mapping(
            raw, field_name=f"dual_branch.branches[{index}]"
        )
        if str(row.get("branch_key")) == BRANCH_KEY:
            return row
    raise ValueError("dual-branch scorecard missing new_embodiment branch entry")


def _branch_contract_validation(
    *,
    modality_config_path: Path,
    modality_contract: Mapping[str, Any],
    branch_manifest_payload: Mapping[str, Any],
    controller_audit_payload: Mapping[str, Any],
    branch_manifest_path: Path,
    controller_audit_path: Path,
) -> dict[str, Any]:
    expected_rel_config_path = gr00t_p_ladder_unitree_g1._rel_repo(modality_config_path)
    expected_rel_manifest_path = gr00t_p_ladder_unitree_g1._rel_repo(
        branch_manifest_path
    )
    expected_fingerprint = str(modality_contract["payload_fingerprint_sha256"])

    blocking_reasons: list[str] = []
    mismatch_fields: list[str] = []

    if str(branch_manifest_payload.get("artifact_kind")) != (
        gr00t_controller_audit_new_embodiment.BRANCH_MANIFEST_ARTIFACT_KIND
    ):
        blocking_reasons.append("branch_manifest_artifact_kind_mismatch")
        mismatch_fields.append("branch_manifest.artifact_kind")
    if str(controller_audit_payload.get("artifact_kind")) != (
        gr00t_controller_audit_new_embodiment.REPORT_ARTIFACT_KIND
    ):
        blocking_reasons.append("controller_audit_artifact_kind_mismatch")
        mismatch_fields.append("controller_audit.artifact_kind")

    if str(branch_manifest_payload.get("branch_tag")) != BRANCH:
        blocking_reasons.append("branch_manifest_branch_tag_mismatch")
        mismatch_fields.append("branch_manifest.branch_tag")
    if str(controller_audit_payload.get("branch_tag")) != BRANCH:
        blocking_reasons.append("controller_audit_branch_tag_mismatch")
        mismatch_fields.append("controller_audit.branch_tag")

    if branch_manifest_payload.get("public_anchor_comparable") is not False:
        blocking_reasons.append(
            "branch_manifest_public_anchor_comparable_must_be_false"
        )
        mismatch_fields.append("branch_manifest.public_anchor_comparable")
    if controller_audit_payload.get("public_anchor_comparable") is not False:
        blocking_reasons.append(
            "controller_audit_public_anchor_comparable_must_be_false"
        )
        mismatch_fields.append("controller_audit.public_anchor_comparable")

    if str(branch_manifest_payload.get("formal_branch_eligibility")) != "ALLOW":
        blocking_reasons.append("branch_manifest_eligibility_blocked")
        mismatch_fields.append("branch_manifest.formal_branch_eligibility")
    if str(controller_audit_payload.get("formal_branch_eligibility")) != "ALLOW":
        blocking_reasons.append("controller_audit_eligibility_blocked")
        mismatch_fields.append("controller_audit.formal_branch_eligibility")

    if branch_manifest_payload.get("modality_config_path") != expected_rel_config_path:
        blocking_reasons.append("branch_manifest_modality_config_path_mismatch")
        mismatch_fields.append("branch_manifest.modality_config_path")
    if controller_audit_payload.get("modality_config_path") != expected_rel_config_path:
        blocking_reasons.append("controller_audit_modality_config_path_mismatch")
        mismatch_fields.append("controller_audit.modality_config_path")

    if (
        branch_manifest_payload.get("modality_config_fingerprint_sha256")
        != expected_fingerprint
    ):
        blocking_reasons.append("branch_manifest_modality_config_fingerprint_mismatch")
        mismatch_fields.append("branch_manifest.modality_config_fingerprint_sha256")
    if (
        controller_audit_payload.get("modality_config_fingerprint_sha256")
        != expected_fingerprint
    ):
        blocking_reasons.append("controller_audit_modality_config_fingerprint_mismatch")
        mismatch_fields.append("controller_audit.modality_config_fingerprint_sha256")

    branch_manifest_normalization = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_manifest_payload.get("normalization_source", {}),
        field_name="branch_manifest.normalization_source",
    )
    controller_audit_normalization = gr00t_p_ladder_unitree_g1._as_mapping(
        controller_audit_payload.get("normalization_source", {}),
        field_name="controller_audit.normalization_source",
    )
    if dict(branch_manifest_normalization) != dict(controller_audit_normalization):
        blocking_reasons.append("branch_manifest_normalization_source_mismatch")
        mismatch_fields.append("normalization_source")
    if branch_manifest_normalization.get("cross_branch_reuse_allowed") is not False:
        blocking_reasons.append(
            "normalization_source.cross_branch_reuse_allowed_must_be_false"
        )
        mismatch_fields.append("normalization_source.cross_branch_reuse_allowed")

    if controller_audit_payload.get("branch_manifest_path") not in {
        expected_rel_manifest_path,
        str(branch_manifest_path),
        str(branch_manifest_path.resolve()),
    }:
        blocking_reasons.append("controller_audit_branch_manifest_path_mismatch")
        mismatch_fields.append("controller_audit.branch_manifest_path")

    return {
        "status": "PASS" if not blocking_reasons else "BLOCK",
        "blocking_reasons": sorted(set(blocking_reasons)),
        "mismatch_fields": sorted(set(mismatch_fields)),
        "modality_config_path": expected_rel_config_path,
        "branch_manifest_path": expected_rel_manifest_path,
        "controller_audit_path": gr00t_p_ladder_unitree_g1._rel_repo(
            controller_audit_path
        ),
        "modality_config_fingerprint_sha256": expected_fingerprint,
        "normalization_source_signature_sha256": _normalization_source_signature(
            branch_manifest_normalization
        ),
    }


def _build_base_metrics(*, prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    teacher_reachability = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("teacher_reachability", {}), field_name="teacher_reachability"
    )
    preflight_prerequisite_proof = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("preflight_prerequisite_proof", {}),
        field_name="preflight_prerequisite_proof",
    )
    current_baseline = gr00t_p_ladder_unitree_g1._as_mapping(
        teacher_reachability.get("current_baseline", {}),
        field_name="teacher_reachability.current_baseline",
    )
    condition_flip = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("condition_flip", {}), field_name="condition_flip"
    )
    condition_response = gr00t_p_ladder_unitree_g1._as_mapping(
        condition_flip.get("response_ratio", {}),
        field_name="condition_flip.response_ratio",
    )
    teacher_gap = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("teacher_student_gap", {}), field_name="teacher_student_gap"
    )
    action_telemetry = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("action_telemetry", {}), field_name="action_telemetry"
    )
    controller_absorbed_groups = {
        str(item)
        for item in gr00t_p_ladder_unitree_g1._as_list(
            action_telemetry.get("controller_absorbed_groups", []),
            field_name="action_telemetry.controller_absorbed_groups",
        )
    }
    model_insensitive_groups = {
        str(item)
        for item in gr00t_p_ladder_unitree_g1._as_list(
            action_telemetry.get("model_insensitive_groups", []),
            field_name="action_telemetry.model_insensitive_groups",
        )
    }
    zero_motion_flags = gr00t_p_ladder_unitree_g1._as_mapping(
        action_telemetry.get("zero_motion_flags", {}),
        field_name="action_telemetry.zero_motion_flags",
    )
    zero_motion_groups = {
        str(item)
        for item in gr00t_p_ladder_unitree_g1._as_list(
            zero_motion_flags.get("all_zero_in_both_groups", []),
            field_name="action_telemetry.zero_motion_flags.all_zero_in_both_groups",
        )
    }
    action_problem_groups = sorted(
        controller_absorbed_groups | model_insensitive_groups | zero_motion_groups
    )
    return {
        "success_count": gr00t_p_ladder_unitree_g1._as_int(
            current_baseline.get("success_count", 0),
            field_name="teacher_reachability.current_baseline.success_count",
        ),
        "success_rate": gr00t_p_ladder_unitree_g1._round_float(
            gr00t_p_ladder_unitree_g1._as_number(
                current_baseline.get("success_rate", 0.0),
                field_name="teacher_reachability.current_baseline.success_rate",
            )
        ),
        "condition_flip_response_ratio": gr00t_p_ladder_unitree_g1._round_float(
            gr00t_p_ladder_unitree_g1._as_number(
                condition_response.get("min_ratio_across_semantic_flips", 0.0),
                field_name=(
                    "condition_flip.response_ratio.min_ratio_across_semantic_flips"
                ),
            )
        ),
        "teacher_student_gap": gr00t_p_ladder_unitree_g1._mean_formal_teacher_student_gap(
            teacher_gap
        ),
        "action_chain_problem_group_count": int(len(action_problem_groups)),
        "controller_absorbed_group_count": int(len(controller_absorbed_groups)),
        "model_insensitive_group_count": int(len(model_insensitive_groups)),
        "zero_motion_group_count": int(len(zero_motion_groups)),
        "systemic_break_flags": [
            str(item)
            for item in gr00t_p_ladder_unitree_g1._as_list(
                teacher_reachability.get("blocking_reasons", []),
                field_name="teacher_reachability.blocking_reasons",
            )
        ],
        "provenance_status": str(
            gr00t_p_ladder_unitree_g1._as_mapping(
                prerequisites.get("checkpoint_provenance", {}),
                field_name="checkpoint_provenance",
            ).get("formal_eligibility", "BLOCK")
        ),
    }


def load_prerequisites(
    *,
    p_ladder_policy_gate_json: Path,
    dual_branch_scorecard_json: Path,
    checkpoint_provenance_json: Path,
    condition_flip_json: Path,
    teacher_student_gap_json: Path,
    action_telemetry_json: Path,
    teacher_reachability_json: Path,
    modality_config_path: Path,
    branch_manifest_path: Path,
    controller_audit_json: Path,
    task2_preflight_evidence_json: Path,
) -> dict[str, Any]:
    gate_payload = gr00t_p_ladder_unitree_g1._read_json(
        p_ladder_policy_gate_json, arg_name="p-ladder-policy-gate-json"
    )
    dual_branch_payload = gr00t_p_ladder_unitree_g1._read_json(
        dual_branch_scorecard_json, arg_name="dual-branch-scorecard-json"
    )
    checkpoint_payload = gr00t_p_ladder_unitree_g1._read_json(
        checkpoint_provenance_json, arg_name="checkpoint-provenance-json"
    )
    condition_flip_payload = gr00t_p_ladder_unitree_g1._read_json(
        condition_flip_json, arg_name="condition-flip-json"
    )
    teacher_gap_payload = gr00t_p_ladder_unitree_g1._read_json(
        teacher_student_gap_json, arg_name="teacher-student-gap-json"
    )
    action_telemetry_payload = gr00t_p_ladder_unitree_g1._read_json(
        action_telemetry_json, arg_name="action-telemetry-json"
    )
    teacher_reachability_payload = gr00t_p_ladder_unitree_g1._read_json(
        teacher_reachability_json, arg_name="teacher-reachability-json"
    )
    preflight_prerequisite_proof, preflight_prerequisite_hash = (
        gr00t_p_ladder_unitree_g1.load_preflight_prerequisite_proof(
            task2_preflight_evidence_json
        )
    )
    resolved_modality_config_path = gr00t_p_ladder_unitree_g1._resolve_existing_file(
        modality_config_path, arg_name="modality-config-path"
    )
    resolved_branch_manifest_path = gr00t_p_ladder_unitree_g1._resolve_existing_file(
        branch_manifest_path, arg_name="branch-manifest-path"
    )
    resolved_controller_audit_path = gr00t_p_ladder_unitree_g1._resolve_existing_file(
        controller_audit_json, arg_name="controller-audit-json"
    )
    modality_contract = gr00t_controller_audit_new_embodiment.load_modality_contract(
        resolved_modality_config_path
    )
    branch_manifest_payload = gr00t_p_ladder_unitree_g1._read_json(
        resolved_branch_manifest_path, arg_name="branch-manifest-path"
    )
    controller_audit_payload = gr00t_p_ladder_unitree_g1._read_json(
        resolved_controller_audit_path, arg_name="controller-audit-json"
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
        raise ValueError(
            "p-ladder-policy-gate-json must target NEW_EMBODIMENT P ladder"
        )
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

    branch_contract_validation = _branch_contract_validation(
        modality_config_path=resolved_modality_config_path,
        modality_contract=modality_contract,
        branch_manifest_payload=branch_manifest_payload,
        controller_audit_payload=controller_audit_payload,
        branch_manifest_path=resolved_branch_manifest_path,
        controller_audit_path=resolved_controller_audit_path,
    )
    branch_manifest_hash = gr00t_p_ladder_unitree_g1._sha256(branch_manifest_payload)
    normalization_source = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_manifest_payload.get("normalization_source", {}),
        field_name="branch_manifest.normalization_source",
    )
    relative_action_policy = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_manifest_payload.get("relative_action_policy", {}),
        field_name="branch_manifest.relative_action_policy",
    )
    dataset_provenance = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_manifest_payload.get("dataset_provenance", {}),
        field_name="branch_manifest.dataset_provenance",
    )
    controller_provenance = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_manifest_payload.get("controller_provenance", {}),
        field_name="branch_manifest.controller_provenance",
    )

    prerequisite_hashes = {
        "p_ladder_policy_gate": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(
                gr00t_p_ladder_unitree_g1._resolve_path(p_ladder_policy_gate_json)
            ),
            "signature": gr00t_p_ladder_unitree_g1._signature_from_payload(
                gate_payload
            ),
        },
        "dual_branch_scorecard": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(
                gr00t_p_ladder_unitree_g1._resolve_path(dual_branch_scorecard_json)
            ),
            "signature": gr00t_p_ladder_unitree_g1._signature_from_payload(
                dual_branch_payload
            ),
        },
        "checkpoint_provenance": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(
                gr00t_p_ladder_unitree_g1._resolve_path(checkpoint_provenance_json)
            ),
            "signature": gr00t_p_ladder_unitree_g1._signature_from_payload(
                checkpoint_payload
            ),
        },
        "condition_flip": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(
                gr00t_p_ladder_unitree_g1._resolve_path(condition_flip_json)
            ),
            "signature": gr00t_p_ladder_unitree_g1._signature_from_payload(
                condition_flip_payload
            ),
        },
        "teacher_student_gap": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(
                gr00t_p_ladder_unitree_g1._resolve_path(teacher_student_gap_json)
            ),
            "signature": gr00t_p_ladder_unitree_g1._signature_from_payload(
                teacher_gap_payload
            ),
        },
        "action_telemetry": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(
                gr00t_p_ladder_unitree_g1._resolve_path(action_telemetry_json)
            ),
            "signature": gr00t_p_ladder_unitree_g1._signature_from_payload(
                action_telemetry_payload
            ),
        },
        "teacher_reachability": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(
                gr00t_p_ladder_unitree_g1._resolve_path(teacher_reachability_json)
            ),
            "signature": gr00t_p_ladder_unitree_g1._signature_from_payload(
                teacher_reachability_payload
            ),
        },
        "preflight": dict(preflight_prerequisite_hash),
        "controller_audit": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(resolved_controller_audit_path),
            "signature": gr00t_p_ladder_unitree_g1._signature_from_payload(
                controller_audit_payload
            ),
        },
        "branch_manifest": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(resolved_branch_manifest_path),
            "signature": branch_manifest_hash,
        },
        "modality_config": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(resolved_modality_config_path),
            "signature": gr00t_p_ladder_unitree_g1._sha256(
                modality_contract["payload"]
            ),
        },
    }

    dual_branch = _new_embodiment_branch_entry(dual_branch_payload)
    prerequisites: dict[str, Any] = {
        "p_ladder_policy_gate": gate_payload,
        "dual_branch": {
            "payload": dual_branch_payload,
            "new_embodiment_branch": dict(dual_branch),
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
        "controller_audit": controller_audit_payload,
        "branch_manifest": branch_manifest_payload,
        "modality_contract": modality_contract,
        "branch_contract": {
            "modality_config_path": gr00t_p_ladder_unitree_g1._rel_repo(
                resolved_modality_config_path
            ),
            "modality_config_fingerprint_sha256": str(
                modality_contract["payload_fingerprint_sha256"]
            ),
            "branch_manifest_path": gr00t_p_ladder_unitree_g1._rel_repo(
                resolved_branch_manifest_path
            ),
            "branch_manifest_hash": branch_manifest_hash,
            "normalization_source": dict(normalization_source),
            "normalization_source_signature_sha256": _normalization_source_signature(
                normalization_source
            ),
            "relative_action_policy_signature_sha256": _relative_action_policy_signature(
                relative_action_policy
            ),
            "dataset_provenance": dict(dataset_provenance),
            "controller_provenance": dict(controller_provenance),
            "public_anchor_comparable": False,
            "validation": branch_contract_validation,
        },
        "diagnostics_prerequisite_hashes": prerequisite_hashes,
    }
    prerequisites["base_metrics"] = _build_base_metrics(prerequisites=prerequisites)
    return prerequisites


def build_training_surface_for_rung(rung: str) -> dict[str, Any]:
    normalized = gr00t_p_ladder_unitree_g1._validate_rung(rung)
    return gr00t_p_ladder_unitree_g1._merge_nested(
        BASE_TRAINING_SURFACE, RUNG_SURFACE_PATCHES[normalized]
    )


def build_frozen_data_surface(*, prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    branch_contract = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("branch_contract", {}), field_name="branch_contract"
    )
    dataset_provenance = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_contract.get("dataset_provenance", {}),
        field_name="branch_contract.dataset_provenance",
    )
    normalization_source = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_contract.get("normalization_source", {}),
        field_name="branch_contract.normalization_source",
    )
    dataset_path = str(dataset_provenance.get("dataset_lineage", DEFAULT_DATASET_PATH))
    dataset_mix = [f"{dataset_path}:1.0"]
    dataset_fingerprint = gr00t_p_ladder_unitree_g1._sha256(
        {
            "dataset_provenance": dict(dataset_provenance),
            "modality_config_fingerprint_sha256": branch_contract.get(
                "modality_config_fingerprint_sha256"
            ),
            "branch_manifest_hash": branch_contract.get("branch_manifest_hash"),
        }
    )
    dataset_surface = {
        "dataset_path": dataset_path,
        "dataset_mix": dataset_mix,
        "dataset_fingerprint": dataset_fingerprint,
        "admission": {
            "branch_inclusion": [BRANCH],
            "dataset_source_ids": [dataset_path],
            "dataset_fingerprints": [dataset_fingerprint],
            "admission_policy_version": DEFAULT_ADMISSION_POLICY_VERSION,
        },
        "normalization": {
            "explicit_stats_policy": str(normalization_source.get("policy", "unknown")),
            "stats_fingerprint": branch_contract[
                "normalization_source_signature_sha256"
            ],
            "stats_owner": str(normalization_source.get("owner", "unknown")),
            "explicit_diff_reason": "none",
            "hidden_stats_fingerprint": str(branch_contract["branch_manifest_hash"]),
            "implicit_cross_branch_stats_reuse": bool(
                normalization_source.get("cross_branch_reuse_allowed", False)
            ),
        },
        "sampling": {
            "seed_policy": "repo_local_formal_seed_manifest_v1",
            "episode_sampling_policy": "teacher_reachable_scene_pool_fixed_equal_weight_v1",
        },
    }
    return dataset_surface


def build_comparison_surface(
    rung: str, *, prerequisites: Mapping[str, Any]
) -> dict[str, Any]:
    training_surface = build_training_surface_for_rung(rung)
    data_surface = build_frozen_data_surface(prerequisites=prerequisites)
    branch_contract = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("branch_contract", {}), field_name="branch_contract"
    )
    modality_contract = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("modality_contract", {}), field_name="modality_contract"
    )
    controller_provenance = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_contract.get("controller_provenance", {}),
        field_name="branch_contract.controller_provenance",
    )
    return {
        "branch": {
            "branch_key": BRANCH_KEY,
            "branch_scope": gr00t_ladder_policy_gate.BRANCH_SPECS[BRANCH].branch_scope,
            "public_anchor_comparable": False,
        },
        "embodiment": {
            "embodiment_tag": BRANCH,
            "modality_config_path": str(branch_contract["modality_config_path"]),
            "modality_config_digest": str(
                branch_contract["modality_config_fingerprint_sha256"]
            ),
        },
        "controller": {
            "controller_family": str(
                controller_provenance.get(
                    "controller_family", "custom_non_official_whole_body_controller"
                )
            ),
            "action_horizon": int(modality_contract["policy_horizon_expected"]),
            "relative_action_policy": str(
                branch_contract["relative_action_policy_signature_sha256"]
            ),
            "action_keys": list(modality_contract["action_order_expected"]),
            "state_keys": list(modality_contract["state_order_expected"]),
        },
        "prompt_interface": {
            "prompt_template_id": gr00t_eval_contract_gate.DEFAULT_PROMPT_TEMPLATE_ID,
            "condition_injection": DEFAULT_CONDITION_INJECTION,
            "condition_schema": DEFAULT_CONDITION_SCHEMA,
        },
        "dataset": data_surface,
        "training": training_surface,
    }


def _load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in prior manifest {path}")
    return cast(dict[str, Any], dict(payload))


def _branch_contract_stability_report(
    *,
    output_root: Path,
    current_rung: str,
    branch_contract: Mapping[str, Any],
) -> dict[str, Any]:
    compared_rungs: list[str] = []
    blocking_reasons: list[str] = []
    mismatches_by_rung: dict[str, list[str]] = {}
    expected_modality_path = str(branch_contract["modality_config_path"])
    expected_modality_fingerprint = str(
        branch_contract["modality_config_fingerprint_sha256"]
    )
    expected_branch_manifest_hash = str(branch_contract["branch_manifest_hash"])
    expected_normalization_signature = str(
        branch_contract["normalization_source_signature_sha256"]
    )

    for rung in RUNG_ORDER:
        if rung == current_rung:
            continue
        manifest = _load_existing_manifest(output_root / rung / "manifest.json")
        if manifest is None:
            continue
        compared_rungs.append(rung)
        rung_mismatches: list[str] = []
        if str(manifest.get("branch_manifest_hash")) != expected_branch_manifest_hash:
            blocking_reasons.append("branch_manifest_hash_drift_blocks_comparison")
            rung_mismatches.append("branch_manifest_hash")
        if (
            str(manifest.get("modality_config_path")) != expected_modality_path
            or str(manifest.get("modality_config_fingerprint_sha256"))
            != expected_modality_fingerprint
        ):
            blocking_reasons.append("modality_config_drift_blocks_comparison")
            rung_mismatches.append("modality_config")
        if str(manifest.get("normalization_source_signature_sha256")) != (
            expected_normalization_signature
        ):
            blocking_reasons.append("normalization_source_drift_blocks_comparison")
            rung_mismatches.append("normalization_source")
        if manifest.get("public_anchor_comparable") is not False:
            blocking_reasons.append(
                "public_anchor_comparability_drift_blocks_comparison"
            )
            rung_mismatches.append("public_anchor_comparable")
        if rung_mismatches:
            mismatches_by_rung[rung] = rung_mismatches

    return {
        "status": "PASS" if not blocking_reasons else "BLOCK",
        "compared_rungs": compared_rungs,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "mismatches_by_rung": mismatches_by_rung,
        "expected": {
            "modality_config_path": expected_modality_path,
            "modality_config_fingerprint_sha256": expected_modality_fingerprint,
            "branch_manifest_hash": expected_branch_manifest_hash,
            "normalization_source_signature_sha256": expected_normalization_signature,
            "public_anchor_comparable": False,
        },
    }


def _build_branch_contract_snapshot(
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    branch_contract = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("branch_contract", {}), field_name="branch_contract"
    )
    return {
        "modality_config_path": str(branch_contract["modality_config_path"]),
        "modality_config_fingerprint_sha256": str(
            branch_contract["modality_config_fingerprint_sha256"]
        ),
        "branch_manifest_path": str(branch_contract["branch_manifest_path"]),
        "branch_manifest_hash": str(branch_contract["branch_manifest_hash"]),
        "normalization_source": copy.deepcopy(branch_contract["normalization_source"]),
        "normalization_source_signature_sha256": str(
            branch_contract["normalization_source_signature_sha256"]
        ),
        "public_anchor_comparable": False,
    }


def build_rung_report(
    *,
    rung: str,
    prerequisites: Mapping[str, Any],
    output_root: Path,
    metrics_overrides_by_rung: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_rung = gr00t_p_ladder_unitree_g1._validate_rung(rung)
    output_dir = (output_root / normalized_rung).resolve()
    manifest_path = output_dir / "manifest.json"
    scorecard_path = output_dir / "scorecard.json"

    p_gate_payload = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("p_ladder_policy_gate", {}),
        field_name="prerequisites.p_ladder_policy_gate",
    )
    baseline_surface = build_comparison_surface("P0", prerequisites=prerequisites)
    rung_surface = build_comparison_surface(
        normalized_rung, prerequisites=prerequisites
    )
    comparability = gr00t_ladder_policy_gate.build_ladder_diff_report(
        p_gate_payload,
        baseline_surface,
        rung_surface,
    )

    base_metrics = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("base_metrics", {}), field_name="prerequisites.base_metrics"
    )
    baseline_metrics = gr00t_p_ladder_unitree_g1._apply_metric_overrides(
        base_metrics,
        rung="P0",
        metrics_overrides_by_rung=metrics_overrides_by_rung,
    )
    current_metrics = gr00t_p_ladder_unitree_g1._apply_metric_overrides(
        base_metrics,
        rung=normalized_rung,
        metrics_overrides_by_rung=metrics_overrides_by_rung,
    )
    current_metrics["success_rate"] = gr00t_p_ladder_unitree_g1._round_float(
        float(current_metrics["success_rate"])
    )
    current_metrics["condition_flip_response_ratio"] = (
        gr00t_p_ladder_unitree_g1._round_float(
            float(current_metrics["condition_flip_response_ratio"])
        )
    )
    current_metrics["teacher_student_gap"] = gr00t_p_ladder_unitree_g1._round_float(
        float(current_metrics["teacher_student_gap"])
    )

    promotion_evidence = gr00t_p_ladder_unitree_g1._promotion_evidence(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        prerequisites=prerequisites,
    )
    promotion_report = gr00t_ladder_policy_gate.build_promotion_report(
        p_gate_payload,
        promotion_evidence,
    )
    positive_slope_report = gr00t_p_ladder_unitree_g1._positive_slope_report(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
    )

    branch_contract = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("branch_contract", {}), field_name="branch_contract"
    )
    branch_contract_validation = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_contract.get("validation", {}), field_name="branch_contract.validation"
    )
    branch_contract_stability = _branch_contract_stability_report(
        output_root=output_root,
        current_rung=normalized_rung,
        branch_contract=branch_contract,
    )

    generic_blocking_reasons: list[str] = []
    failure_reasons = list(
        cast(Sequence[str], promotion_report.get("failure_reasons", []))
    )
    if str(comparability.get("comparability_status")) != "PASS":
        generic_blocking_reasons.extend(
            [str(item) for item in comparability.get("blocking_reasons", [])]
        )
    if str(branch_contract_validation.get("status")) != "PASS":
        generic_blocking_reasons.extend(
            [
                str(item)
                for item in branch_contract_validation.get("blocking_reasons", [])
            ]
        )
    if str(branch_contract_stability.get("status")) != "PASS":
        generic_blocking_reasons.extend(
            [
                str(item)
                for item in branch_contract_stability.get("blocking_reasons", [])
            ]
        )

    dual_branch = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("dual_branch", {}), field_name="dual_branch"
    )
    if not bool(dual_branch.get("allow_p_ladder", False)):
        generic_blocking_reasons.append("dual_branch_prerequisite_blocks_p_ladder")
    checkpoint_provenance = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("checkpoint_provenance", {}),
        field_name="checkpoint_provenance",
    )
    if str(checkpoint_provenance.get("formal_eligibility")) != "ALLOW":
        generic_blocking_reasons.append("checkpoint_provenance_blocked")
    teacher_reachability = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("teacher_reachability", {}), field_name="teacher_reachability"
    )
    preflight_prerequisite_proof = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("preflight_prerequisite_proof", {}),
        field_name="preflight_prerequisite_proof",
    )
    if not bool(teacher_reachability.get("allow_formal_ladders", False)):
        generic_blocking_reasons.append("teacher_reachability_blocks_formal_scene_pool")

    status = "PASS"
    blocking_reasons = list(dict.fromkeys(generic_blocking_reasons))
    p3_gate_report: dict[str, Any] | None = None
    if normalized_rung == "P3":
        p3_gate_report = gr00t_p_ladder_unitree_g1._p3_positive_slope_from_prior_rungs(
            output_root
        )
        blocking_reasons.extend(
            [str(item) for item in p3_gate_report.get("blocking_reasons", [])]
        )
        if blocking_reasons:
            status = "BLOCK"
    elif blocking_reasons:
        status = "BLOCK"

    baseline_flags = [str(item) for item in baseline_metrics["systemic_break_flags"]]
    current_flags = [str(item) for item in current_metrics["systemic_break_flags"]]
    new_systemic_break, new_systemic_break_flags = (
        gr00t_p_ladder_unitree_g1._new_systemic_break(baseline_flags, current_flags)
    )

    branch_contract_snapshot = _build_branch_contract_snapshot(prerequisites)

    scorecard: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": SCORECARD_ARTIFACT_KIND,
        "branch": BRANCH,
        "branch_key": BRANCH_KEY,
        "branch_scope": gr00t_ladder_policy_gate.BRANCH_SPECS[BRANCH].branch_scope,
        "public_anchor_comparable": False,
        "official_benchmark_interpretation": "internal_only_no_public_anchor_threshold",
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
        "manifest_path": str(manifest_path),
        "output_path": str(scorecard_path),
        "comparability": comparability,
        "promotion_gate": promotion_report,
        "branch_contract_validation": branch_contract_validation,
        "branch_contract_stability": branch_contract_stability,
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
                for item in gr00t_p_ladder_unitree_g1._as_list(
                    teacher_reachability.get("reachable_scene_ids", []),
                    field_name="teacher_reachability.reachable_scene_ids",
                )
            ],
        },
        "success_count": int(current_metrics["success_count"]),
        "success_rate": gr00t_p_ladder_unitree_g1._round_float(
            float(current_metrics["success_rate"])
        ),
        "condition_flip_response_ratio": gr00t_p_ladder_unitree_g1._round_float(
            float(current_metrics["condition_flip_response_ratio"])
        ),
        "teacher_student_gap": gr00t_p_ladder_unitree_g1._round_float(
            float(current_metrics["teacher_student_gap"])
        ),
        "action_chain_problem_group_count": int(
            current_metrics["action_chain_problem_group_count"]
        ),
        "provenance_status": str(current_metrics["provenance_status"]),
        "condition_flip_delta": gr00t_p_ladder_unitree_g1._round_float(
            float(current_metrics["condition_flip_response_ratio"])
            - float(baseline_metrics["condition_flip_response_ratio"])
        ),
        "teacher_student_gap_delta": gr00t_p_ladder_unitree_g1._round_float(
            float(current_metrics["teacher_student_gap"])
            - float(baseline_metrics["teacher_student_gap"])
        ),
        "action_chain_delta": gr00t_p_ladder_unitree_g1._build_action_chain_delta(
            baseline_metrics,
            current_metrics,
        ),
        "positive_slope_report": positive_slope_report,
        "systemic_break_flags": current_flags,
        "new_systemic_break_detected": new_systemic_break,
        "new_systemic_break_flags": new_systemic_break_flags,
        "baseline_metrics": {
            "success_count": int(baseline_metrics["success_count"]),
            "success_rate": gr00t_p_ladder_unitree_g1._round_float(
                float(baseline_metrics["success_rate"])
            ),
            "condition_flip_response_ratio": gr00t_p_ladder_unitree_g1._round_float(
                float(baseline_metrics["condition_flip_response_ratio"])
            ),
            "teacher_student_gap": gr00t_p_ladder_unitree_g1._round_float(
                float(baseline_metrics["teacher_student_gap"])
            ),
            "action_chain_problem_group_count": int(
                baseline_metrics["action_chain_problem_group_count"]
            ),
            "provenance_status": str(baseline_metrics["provenance_status"]),
        },
        "modality_config_path": branch_contract_snapshot["modality_config_path"],
        "modality_config_fingerprint_sha256": branch_contract_snapshot[
            "modality_config_fingerprint_sha256"
        ],
        "branch_manifest_path": branch_contract_snapshot["branch_manifest_path"],
        "branch_manifest_hash": branch_contract_snapshot["branch_manifest_hash"],
        "normalization_source": branch_contract_snapshot["normalization_source"],
        "normalization_source_signature_sha256": branch_contract_snapshot[
            "normalization_source_signature_sha256"
        ],
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
    scorecard["report_signature_sha256"] = gr00t_p_ladder_unitree_g1._sha256(scorecard)

    frozen_data_surface = {
        "dataset": copy.deepcopy(rung_surface["dataset"]),
        "prompt_interface": copy.deepcopy(rung_surface["prompt_interface"]),
        "controller": copy.deepcopy(rung_surface["controller"]),
        "embodiment": copy.deepcopy(rung_surface["embodiment"]),
        "branch_contract": copy.deepcopy(branch_contract_snapshot),
    }
    manifest: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": MANIFEST_ARTIFACT_KIND,
        "branch": BRANCH,
        "branch_key": BRANCH_KEY,
        "branch_scope": gr00t_ladder_policy_gate.BRANCH_SPECS[BRANCH].branch_scope,
        "public_anchor_comparable": False,
        "official_benchmark_interpretation": "internal_only_no_public_anchor_threshold",
        "axis": AXIS,
        "rung": normalized_rung,
        "rung_order": list(RUNG_ORDER),
        "status": status,
        "promotion_status": str(scorecard["promotion_status"]),
        "blocking_reasons": list(scorecard["blocking_reasons"]),
        "failure_reasons": list(scorecard["failure_reasons"]),
        "output_path": str(manifest_path),
        "scorecard_path": str(scorecard_path),
        "scorecard_signature_sha256": str(scorecard["report_signature_sha256"]),
        "rung_description": RUNG_DESCRIPTIONS[normalized_rung],
        "p_ladder_policy_gate_path": cast(
            Mapping[str, Any], prerequisites["diagnostics_prerequisite_hashes"]
        )["p_ladder_policy_gate"]["path"],
        "allowed_variable_parameter_fields": list(
            cast(Sequence[str], p_gate_payload.get("allowed_difference_paths", []))
        ),
        "comparison_surface_digest": gr00t_p_ladder_unitree_g1._sha256(rung_surface),
        "comparability": comparability,
        "branch_contract_validation": branch_contract_validation,
        "branch_contract_stability": branch_contract_stability,
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
        "frozen_data_surface": frozen_data_surface,
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
                for item in gr00t_p_ladder_unitree_g1._as_list(
                    teacher_reachability.get("reachable_scene_ids", []),
                    field_name="teacher_reachability.reachable_scene_ids",
                )
            ],
            "scene_rows": [
                dict(
                    gr00t_p_ladder_unitree_g1._as_mapping(
                        item,
                        field_name="teacher_reachability.scene_pool.scene_rows[]",
                    )
                )
                for item in gr00t_p_ladder_unitree_g1._as_list(
                    gr00t_p_ladder_unitree_g1._as_mapping(
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
        "modality_config_path": branch_contract_snapshot["modality_config_path"],
        "modality_config_fingerprint_sha256": branch_contract_snapshot[
            "modality_config_fingerprint_sha256"
        ],
        "branch_manifest_path": branch_contract_snapshot["branch_manifest_path"],
        "branch_manifest_hash": branch_contract_snapshot["branch_manifest_hash"],
        "normalization_source": branch_contract_snapshot["normalization_source"],
        "normalization_source_signature_sha256": branch_contract_snapshot[
            "normalization_source_signature_sha256"
        ],
        "condition_flip_delta": scorecard["condition_flip_delta"],
        "teacher_student_gap_delta": scorecard["teacher_student_gap_delta"],
        "action_chain_delta": copy.deepcopy(scorecard["action_chain_delta"]),
        "data_axis_frozen": True,
        "data_axis_non_drift_fields": [
            "dataset.dataset_path",
            "dataset.dataset_fingerprint",
            "dataset.dataset_mix",
            "dataset.normalization.explicit_stats_policy",
            "dataset.normalization.stats_fingerprint",
            "dataset.normalization.stats_owner",
            "prompt_interface.prompt_template_id",
            "prompt_interface.condition_injection",
            "controller.controller_family",
            "controller.action_horizon",
            "embodiment.embodiment_tag",
            "embodiment.modality_config_path",
            "branch_contract.branch_manifest_hash",
            "branch_contract.normalization_source_signature_sha256",
        ],
        "decision_basis": (
            "Use the task-12 P-axis whitelist as the only allowed change surface; freeze "
            "NEW_EMBODIMENT modality config, branch manifest hash, normalization ownership, "
            "controller provenance, dataset provenance, prompt interface, scene pool, and "
            "prerequisite hashes across all parameter rungs; only unlock P3 when P1 or P2 "
            "shows positive slope under the fixed formal protocol; never interpret the branch "
            "against the public UNITREE_G1 anchor."
        ),
    }
    manifest["report_signature_sha256"] = gr00t_p_ladder_unitree_g1._sha256(manifest)
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
    modality_config_path: Path,
    branch_manifest_path: Path,
    controller_audit_json: Path,
    task2_preflight_evidence_json: Path,
    metrics_overrides_by_rung: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_output_root = gr00t_p_ladder_unitree_g1._validate_output_dir(output_root)
    prerequisites = load_prerequisites(
        p_ladder_policy_gate_json=p_ladder_policy_gate_json,
        dual_branch_scorecard_json=dual_branch_scorecard_json,
        checkpoint_provenance_json=checkpoint_provenance_json,
        condition_flip_json=condition_flip_json,
        teacher_student_gap_json=teacher_student_gap_json,
        action_telemetry_json=action_telemetry_json,
        teacher_reachability_json=teacher_reachability_json,
        modality_config_path=modality_config_path,
        branch_manifest_path=branch_manifest_path,
        controller_audit_json=controller_audit_json,
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
    gr00t_p_ladder_unitree_g1._write_json(
        manifest_path, cast(Mapping[str, Any], report["manifest"])
    )
    gr00t_p_ladder_unitree_g1._write_json(
        scorecard_path, cast(Mapping[str, Any], report["scorecard"])
    )
    scorecard = cast(Mapping[str, Any], report["scorecard"])
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": SCORECARD_ARTIFACT_KIND,
        "branch": BRANCH,
        "rung": gr00t_p_ladder_unitree_g1._validate_rung(rung),
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
            modality_config_path=args.modality_config_path,
            branch_manifest_path=args.branch_manifest_path,
            controller_audit_json=args.controller_audit_json,
            task2_preflight_evidence_json=args.task2_preflight_evidence_json,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(gr00t_p_ladder_unitree_g1._exception_message(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if str(result.get("status")) != "BLOCK" else 1


__all__ = [
    "AXIS",
    "BASE_TRAINING_SURFACE",
    "BRANCH",
    "BRANCH_KEY",
    "DEFAULT_ACTION_TELEMETRY_JSON",
    "DEFAULT_BRANCH_MANIFEST_PATH",
    "DEFAULT_CHECKPOINT_PROVENANCE_JSON",
    "DEFAULT_CONDITION_FLIP_JSON",
    "DEFAULT_CONTROLLER_AUDIT_JSON",
    "DEFAULT_DUAL_BRANCH_SCORECARD_JSON",
    "DEFAULT_MODALITY_CONFIG_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_P_LADDER_POLICY_GATE_JSON",
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
    "main",
    "materialize_p_ladder_rung",
]


if __name__ == "__main__":
    raise SystemExit(main())
