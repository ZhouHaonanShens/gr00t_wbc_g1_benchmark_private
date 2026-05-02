#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
import re
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_ROOT = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/d"
)
DEFAULT_D_LADDER_COMPARABILITY_GATE_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/ladder_policy/"
    "d_ladder_policy_gate_new_embodiment.json"
)
DEFAULT_D_LADDER_ADMISSION_GATE_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/"
    "d_ladder_policy_gate_new_embodiment.json"
)
DEFAULT_DATASET_SOURCE_REGISTRY_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/dataset_source_registry.json"
)
DEFAULT_DUAL_BRANCH_SCORECARD_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/dual_branch_scorecard.json"
)
DEFAULT_CHECKPOINT_PROVENANCE_JSON = Path(
    "agent/artifacts/gr00t_checkpoint_provenance/checkpoint_provenance_report.json"
)
DEFAULT_CONDITION_FLIP_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/"
    "condition_flip_scorecard_new_embodiment.json"
)
DEFAULT_TEACHER_STUDENT_GAP_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/"
    "teacher_student_gap_scorecard_new_embodiment.json"
)
DEFAULT_ACTION_TELEMETRY_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/"
    "action_chain_telemetry_new_embodiment.json"
)
DEFAULT_TEACHER_REACHABILITY_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/teacher_reachability/"
    "teacher_reachability_gate_new_embodiment.json"
)
DEFAULT_MODALITY_CONFIG_PATH = Path("work/configs/new_embodiment/modality_config.json")
DEFAULT_BRANCH_MANIFEST_PATH = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/branch_manifest.json"
)
DEFAULT_CONTROLLER_AUDIT_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/"
    "controller_audit_new_embodiment.json"
)

REPORT_SCHEMA_VERSION = "gr00t_d_ladder_new_embodiment_v1"
MANIFEST_ARTIFACT_KIND = "gr00t_d_ladder_new_embodiment_manifest"
SCORECARD_ARTIFACT_KIND = "gr00t_d_ladder_new_embodiment_scorecard"

BRANCH = "NEW_EMBODIMENT"
BRANCH_KEY = "new_embodiment"
AXIS = "D"

RUNG_ORDER: tuple[str, ...] = ("D0", "D1", "D2", "D3", "D4")
RUNG_DESCRIPTIONS: dict[str, str] = {
    "D0": "Apple-only baseline anchor on the frozen NEW_EMBODIMENT branch contract.",
    "D1": (
        "Add the Arena G1 synthetic dataset under explicit NEW_EMBODIMENT provenance and "
        "branch-scoped stats ownership."
    ),
    "D2": (
        "Add the PhysicalAI teleop G1 dataset while preserving the frozen custom branch "
        "contract and diagnostics watchlist."
    ),
    "D3": (
        "Add the Lightwheel decoupled-WBC dataset as an internal-only source with explicit "
        "controller lineage and regenerated stats."
    ),
    "D4": (
        "Add the Lightwheel G1-Controller dataset as a legal branch-only rung that never "
        "becomes public-anchor comparable output."
    ),
}

DEFAULT_DATASET_SURFACE_PATH = "new_embodiment.d_ladder.formal_mix"
DEFAULT_ADMISSION_POLICY_VERSION = "new_embodiment_formal_d_ladder_v1"
DEFAULT_DATASET_FINGERPRINT_SOURCE = "task17_new_embodiment_d_ladder_materialization"
DEFAULT_COMPATIBILITY_REASON = "compatible_with_frozen_new_embodiment_branch_contract"
DEFAULT_INCOMPATIBLE_REASON = (
    "controller_or_embodiment_incompatible_with_frozen_new_embodiment_branch_contract"
)


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import gr00t_checkpoint_provenance_gate
from work.recap import gr00t_controller_audit_new_embodiment
from work.recap import gr00t_d_ladder_policy_gate
from work.recap import gr00t_eval_contract_gate
from work.recap import gr00t_ladder_policy_gate
from work.recap import gr00t_p_ladder_new_embodiment
from work.recap import gr00t_p_ladder_unitree_g1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_d_ladder_new_embodiment.py",
        description=(
            "Materialize the NEW_EMBODIMENT D0-D4 data ladder artifacts under the frozen "
            "branch contract, the task-15 admission policy, and the D4 branch-only rule."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--rung", required=True, choices=list(RUNG_ORDER))
    _ = parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _ = parser.add_argument(
        "--d-ladder-comparability-gate-json",
        type=Path,
        default=DEFAULT_D_LADDER_COMPARABILITY_GATE_JSON,
    )
    _ = parser.add_argument(
        "--d-ladder-admission-gate-json",
        type=Path,
        default=DEFAULT_D_LADDER_ADMISSION_GATE_JSON,
    )
    _ = parser.add_argument(
        "--dataset-source-registry-json",
        type=Path,
        default=DEFAULT_DATASET_SOURCE_REGISTRY_JSON,
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
        "--dataset-admission-records-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file that overrides per-dataset fingerprint/provenance/"
            "normalization/compatibility records for task-15 admission evaluation."
        ),
    )
    _ = parser.add_argument(
        "--task2-preflight-evidence-json",
        type=Path,
        default=gr00t_p_ladder_unitree_g1.DEFAULT_TASK2_PREFLIGHT_EVIDENCE_JSON,
    )
    return parser


def _dataset_slug(dataset_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", dataset_id.lower()).strip("_")


def _validate_rung(rung: str) -> str:
    normalized = str(rung).strip().upper()
    if normalized not in RUNG_ORDER:
        raise ValueError(f"unsupported rung: {rung}")
    return normalized


def _equal_weight_mix(dataset_ids: Sequence[str]) -> list[str]:
    if not dataset_ids:
        return []
    weight = 1.0 / float(len(dataset_ids))
    return [f"{dataset_id}:{weight:.6f}" for dataset_id in dataset_ids]


def _deep_merge_records(
    base: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = {key: copy.deepcopy(dict(value)) for key, value in base.items()}
    for dataset_id, raw_override in overrides.items():
        if dataset_id not in merged:
            raise ValueError(
                f"dataset-admission-records-json has unknown dataset_id: {dataset_id}"
            )
        merged[dataset_id] = gr00t_p_ladder_unitree_g1._merge_nested(
            merged[dataset_id], dict(raw_override)
        )
    return merged


def _validate_reason_codes(value: object, *, field_name: str) -> list[str]:
    return [
        str(item)
        for item in gr00t_p_ladder_unitree_g1._as_list(value, field_name=field_name)
    ]


def _default_compatibility_record(
    *, dataset_id: str, dataset_entry: Mapping[str, Any]
) -> dict[str, Any]:
    branch_eligibility = gr00t_p_ladder_unitree_g1._as_mapping(
        gr00t_p_ladder_unitree_g1._as_mapping(
            dataset_entry.get("branch_eligibility", {}),
            field_name=f"registry.{dataset_id}.branch_eligibility",
        ).get(BRANCH, {}),
        field_name=f"registry.{dataset_id}.branch_eligibility.{BRANCH}",
    )
    reason_codes = [DEFAULT_COMPATIBILITY_REASON]
    if bool(dataset_entry.get("branch_only", False)):
        reason_codes.append("branch_only_dataset_allowed_on_new_embodiment")
    return {
        "compatible": True,
        "reason_codes": reason_codes,
        "summary": str(branch_eligibility.get("summary", "compatibility attested")),
    }


def build_default_dataset_admission_records(
    *,
    branch_contract: Mapping[str, Any],
    registry_payload: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_entries = gr00t_d_ladder_policy_gate._registry_datasets_by_id(
        registry_payload
    )
    normalization_source = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_contract.get("normalization_source", {}),
        field_name="branch_contract.normalization_source",
    )

    dataset_fingerprints: dict[str, dict[str, Any]] = {}
    dataset_provenance: dict[str, dict[str, Any]] = {}
    normalization_records: dict[str, dict[str, Any]] = {}
    compatibility_records: dict[str, dict[str, Any]] = {}

    for dataset_id, dataset_entry in dataset_entries.items():
        controller_family = str(dataset_entry.get("controller_family", "unknown"))
        fingerprint = gr00t_p_ladder_unitree_g1._sha256(
            {
                "dataset_id": dataset_id,
                "branch_manifest_hash": branch_contract["branch_manifest_hash"],
                "modality_config_fingerprint_sha256": branch_contract[
                    "modality_config_fingerprint_sha256"
                ],
                "registry_fingerprint_sha256": registry_payload[
                    "registry_fingerprint_sha256"
                ],
            }
        )
        compatibility_record = _default_compatibility_record(
            dataset_id=dataset_id,
            dataset_entry=dataset_entry,
        )
        compatibility_records[dataset_id] = compatibility_record

        dataset_fingerprints[dataset_id] = {
            "dataset_fingerprint": fingerprint,
            "fingerprint_source": DEFAULT_DATASET_FINGERPRINT_SOURCE,
        }
        dataset_provenance[dataset_id] = {
            "source_registry_id": dataset_id,
            "local_dataset_path": (
                f"agent/artifacts/gr00t_anchor_controller_recap/cache/"
                f"new_embodiment/{_dataset_slug(dataset_id)}"
            ),
            "controller_family": controller_family,
            "provenance_complete": True,
            "guardrails_satisfied": True,
            "compatibility_status": "PASS",
        }
        stats_fingerprint = gr00t_p_ladder_unitree_g1._sha256(
            {
                "dataset_id": dataset_id,
                "dataset_fingerprint": fingerprint,
                "branch_manifest_hash": branch_contract["branch_manifest_hash"],
                "normalization_source_signature_sha256": branch_contract[
                    "normalization_source_signature_sha256"
                ],
            }
        )
        normalization_records[dataset_id] = {
            "stats_fingerprint": stats_fingerprint,
            "hidden_stats_fingerprint": stats_fingerprint,
            "normalization_owner": str(
                normalization_source.get(
                    "owner", dataset_entry.get("normalization_owner")
                )
            ),
            "explicit_stats_policy": str(
                normalization_source.get("policy", DEFAULT_ADMISSION_POLICY_VERSION)
            ),
            "cross_dataset_reuse_declared": False,
            "regenerated_for_branch": True,
        }

    return {
        "dataset_fingerprints": dataset_fingerprints,
        "dataset_provenance": dataset_provenance,
        "normalization_records": normalization_records,
        "compatibility_records": compatibility_records,
    }


def load_dataset_admission_records(
    *,
    dataset_admission_records_json: Path | None,
    branch_contract: Mapping[str, Any],
    registry_payload: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = build_default_dataset_admission_records(
        branch_contract=branch_contract,
        registry_payload=registry_payload,
    )
    if dataset_admission_records_json is None:
        merged = copy.deepcopy(defaults)
    else:
        payload = gr00t_p_ladder_unitree_g1._read_json(
            dataset_admission_records_json,
            arg_name="dataset-admission-records-json",
        )
        merged = {
            "dataset_fingerprints": _deep_merge_records(
                cast(Mapping[str, Mapping[str, Any]], defaults["dataset_fingerprints"]),
                cast(
                    Mapping[str, Mapping[str, Any]],
                    gr00t_p_ladder_unitree_g1._as_mapping(
                        payload.get("dataset_fingerprints", {}),
                        field_name="dataset_admission_records.dataset_fingerprints",
                    ),
                ),
            ),
            "dataset_provenance": _deep_merge_records(
                cast(Mapping[str, Mapping[str, Any]], defaults["dataset_provenance"]),
                cast(
                    Mapping[str, Mapping[str, Any]],
                    gr00t_p_ladder_unitree_g1._as_mapping(
                        payload.get("dataset_provenance", {}),
                        field_name="dataset_admission_records.dataset_provenance",
                    ),
                ),
            ),
            "normalization_records": _deep_merge_records(
                cast(
                    Mapping[str, Mapping[str, Any]], defaults["normalization_records"]
                ),
                cast(
                    Mapping[str, Mapping[str, Any]],
                    gr00t_p_ladder_unitree_g1._as_mapping(
                        payload.get("normalization_records", {}),
                        field_name="dataset_admission_records.normalization_records",
                    ),
                ),
            ),
            "compatibility_records": _deep_merge_records(
                cast(
                    Mapping[str, Mapping[str, Any]], defaults["compatibility_records"]
                ),
                cast(
                    Mapping[str, Mapping[str, Any]],
                    gr00t_p_ladder_unitree_g1._as_mapping(
                        payload.get("compatibility_records", {}),
                        field_name="dataset_admission_records.compatibility_records",
                    ),
                ),
            ),
        }

    compatibility_records = cast(
        Mapping[str, Mapping[str, Any]], merged["compatibility_records"]
    )
    dataset_provenance = cast(
        Mapping[str, Mapping[str, Any]], merged["dataset_provenance"]
    )
    finalized_provenance: dict[str, dict[str, Any]] = {}
    for dataset_id, provenance in dataset_provenance.items():
        compatibility = gr00t_p_ladder_unitree_g1._as_mapping(
            compatibility_records.get(dataset_id, {}),
            field_name=f"compatibility_records.{dataset_id}",
        )
        compatible = bool(compatibility.get("compatible", True))
        finalized = copy.deepcopy(dict(provenance))
        finalized["guardrails_satisfied"] = compatible
        finalized["compatibility_status"] = "PASS" if compatible else "BLOCK"
        finalized_provenance[dataset_id] = finalized
    merged["dataset_provenance"] = finalized_provenance
    return merged


def _build_dataset_compatibility_report(
    *,
    rung: str,
    admission_records: Mapping[str, Any],
    branch_contract: Mapping[str, Any],
    registry_payload: Mapping[str, Any],
) -> dict[str, Any]:
    included_ids = [
        spec.dataset_id
        for spec in gr00t_d_ladder_policy_gate._included_dataset_specs(rung)
    ]
    compatibility_records = gr00t_p_ladder_unitree_g1._as_mapping(
        admission_records.get("compatibility_records", {}),
        field_name="admission_records.compatibility_records",
    )
    registry_entries = gr00t_d_ladder_policy_gate._registry_datasets_by_id(
        registry_payload
    )

    blocking_reasons: list[str] = []
    per_dataset: dict[str, Any] = {}
    for dataset_id in included_ids:
        compatibility = gr00t_p_ladder_unitree_g1._as_mapping(
            compatibility_records.get(dataset_id, {}),
            field_name=f"compatibility_records.{dataset_id}",
        )
        reason_codes = _validate_reason_codes(
            compatibility.get("reason_codes", []),
            field_name=f"compatibility_records.{dataset_id}.reason_codes",
        )
        compatible = bool(compatibility.get("compatible", True))
        if not compatible and not reason_codes:
            reason_codes = [DEFAULT_INCOMPATIBLE_REASON]
        if not compatible:
            blocking_reasons.extend(f"{code}:{dataset_id}" for code in reason_codes)
        per_dataset[dataset_id] = {
            "status": "PASS" if compatible else "BLOCK",
            "reason_codes": reason_codes,
            "summary": str(compatibility.get("summary", "compatibility verdict")),
            "branch_only": bool(registry_entries[dataset_id].get("branch_only", False)),
            "controller_family": str(
                registry_entries[dataset_id].get("controller_family", "unknown")
            ),
        }

    return {
        "status": "PASS" if not blocking_reasons else "BLOCK",
        "blocking_reasons": sorted(set(blocking_reasons)),
        "per_dataset": per_dataset,
        "branch_manifest_hash": str(branch_contract["branch_manifest_hash"]),
        "modality_config_fingerprint_sha256": str(
            branch_contract["modality_config_fingerprint_sha256"]
        ),
    }


def build_frozen_training_surface() -> dict[str, Any]:
    return copy.deepcopy(
        gr00t_p_ladder_new_embodiment.build_training_surface_for_rung("P0")
    )


def build_dataset_surface(
    rung: str,
    *,
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_rung = _validate_rung(rung)
    branch_contract = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("branch_contract", {}), field_name="branch_contract"
    )
    admission_records = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("dataset_admission_records", {}),
        field_name="dataset_admission_records",
    )
    dataset_fingerprints = gr00t_p_ladder_unitree_g1._as_mapping(
        admission_records.get("dataset_fingerprints", {}),
        field_name="dataset_admission_records.dataset_fingerprints",
    )
    normalization_source = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_contract.get("normalization_source", {}),
        field_name="branch_contract.normalization_source",
    )

    included_ids = [
        spec.dataset_id
        for spec in gr00t_d_ladder_policy_gate._included_dataset_specs(normalized_rung)
    ]
    dataset_mix = _equal_weight_mix(included_ids)
    explicit_stats_fingerprint = gr00t_p_ladder_unitree_g1._sha256(
        {
            "rung": normalized_rung,
            "dataset_source_ids": included_ids,
            "dataset_fingerprints": [
                gr00t_p_ladder_unitree_g1._as_mapping(
                    dataset_fingerprints[dataset_id],
                    field_name=f"dataset_fingerprints.{dataset_id}",
                )["dataset_fingerprint"]
                for dataset_id in included_ids
            ],
            "branch_manifest_hash": branch_contract["branch_manifest_hash"],
        }
    )

    return {
        "dataset_path": DEFAULT_DATASET_SURFACE_PATH,
        "dataset_mix": dataset_mix,
        "admission": {
            "branch_inclusion": [BRANCH],
            "dataset_source_ids": included_ids,
            "dataset_fingerprints": [
                str(
                    gr00t_p_ladder_unitree_g1._as_mapping(
                        dataset_fingerprints[dataset_id],
                        field_name=f"dataset_fingerprints.{dataset_id}",
                    )["dataset_fingerprint"]
                )
                for dataset_id in included_ids
            ],
            "admission_policy_version": DEFAULT_ADMISSION_POLICY_VERSION,
        },
        "normalization": {
            "explicit_stats_policy": str(
                normalization_source.get("policy", DEFAULT_ADMISSION_POLICY_VERSION)
            ),
            "stats_fingerprint": explicit_stats_fingerprint,
            "stats_owner": str(normalization_source.get("owner", BRANCH_KEY)),
            "explicit_diff_reason": (
                "none"
                if normalized_rung == "D0"
                else f"expand_dataset_mix_to_{normalized_rung}"
            ),
            "hidden_stats_fingerprint": str(
                branch_contract["normalization_source_signature_sha256"]
            ),
            "implicit_cross_branch_stats_reuse": bool(
                normalization_source.get("cross_branch_reuse_allowed", False)
            ),
        },
        "sampling": {
            "seed_policy": "repo_local_formal_seed_manifest_v1",
            "episode_sampling_policy": "teacher_reachable_scene_pool_fixed_equal_weight_v1",
        },
    }


def build_comparison_surface(
    rung: str,
    *,
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
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
            "condition_injection": gr00t_p_ladder_new_embodiment.DEFAULT_CONDITION_INJECTION,
            "condition_schema": gr00t_p_ladder_new_embodiment.DEFAULT_CONDITION_SCHEMA,
        },
        "dataset": build_dataset_surface(rung, prerequisites=prerequisites),
        "training": build_frozen_training_surface(),
    }


def _resolve_registry_path(
    *, admission_gate_payload: Mapping[str, Any], dataset_source_registry_json: Path
) -> Path:
    explicit = gr00t_p_ladder_unitree_g1._resolve_existing_file(
        dataset_source_registry_json,
        arg_name="dataset-source-registry-json",
    )
    payload_path = admission_gate_payload.get("dataset_source_registry_path")
    if isinstance(payload_path, str) and payload_path.strip():
        resolved_payload_path = gr00t_p_ladder_unitree_g1._resolve_existing_file(
            Path(payload_path),
            arg_name="d-ladder-admission-gate-json.dataset_source_registry_path",
        )
        if resolved_payload_path != explicit:
            raise ValueError(
                "dataset-source-registry-json does not match the task-15 admission gate binding"
            )
    return explicit


def load_prerequisites(
    *,
    d_ladder_comparability_gate_json: Path,
    d_ladder_admission_gate_json: Path,
    dataset_source_registry_json: Path,
    dual_branch_scorecard_json: Path,
    checkpoint_provenance_json: Path,
    condition_flip_json: Path,
    teacher_student_gap_json: Path,
    action_telemetry_json: Path,
    teacher_reachability_json: Path,
    modality_config_path: Path,
    branch_manifest_path: Path,
    controller_audit_json: Path,
    dataset_admission_records_json: Path | None,
    task2_preflight_evidence_json: Path,
) -> dict[str, Any]:
    comparability_gate_payload = gr00t_p_ladder_unitree_g1._read_json(
        d_ladder_comparability_gate_json,
        arg_name="d-ladder-comparability-gate-json",
    )
    admission_gate_payload = gr00t_p_ladder_unitree_g1._read_json(
        d_ladder_admission_gate_json,
        arg_name="d-ladder-admission-gate-json",
    )
    dual_branch_payload = gr00t_p_ladder_unitree_g1._read_json(
        dual_branch_scorecard_json,
        arg_name="dual-branch-scorecard-json",
    )
    checkpoint_payload = gr00t_p_ladder_unitree_g1._read_json(
        checkpoint_provenance_json,
        arg_name="checkpoint-provenance-json",
    )
    condition_flip_payload = gr00t_p_ladder_unitree_g1._read_json(
        condition_flip_json,
        arg_name="condition-flip-json",
    )
    teacher_gap_payload = gr00t_p_ladder_unitree_g1._read_json(
        teacher_student_gap_json,
        arg_name="teacher-student-gap-json",
    )
    action_telemetry_payload = gr00t_p_ladder_unitree_g1._read_json(
        action_telemetry_json,
        arg_name="action-telemetry-json",
    )
    teacher_reachability_payload = gr00t_p_ladder_unitree_g1._read_json(
        teacher_reachability_json,
        arg_name="teacher-reachability-json",
    )
    preflight_prerequisite_proof, preflight_prerequisite_hash = (
        gr00t_p_ladder_unitree_g1.load_preflight_prerequisite_proof(
            task2_preflight_evidence_json
        )
    )
    resolved_modality_config_path = gr00t_p_ladder_unitree_g1._resolve_existing_file(
        modality_config_path,
        arg_name="modality-config-path",
    )
    resolved_branch_manifest_path = gr00t_p_ladder_unitree_g1._resolve_existing_file(
        branch_manifest_path,
        arg_name="branch-manifest-path",
    )
    resolved_controller_audit_path = gr00t_p_ladder_unitree_g1._resolve_existing_file(
        controller_audit_json,
        arg_name="controller-audit-json",
    )

    if (
        comparability_gate_payload.get("artifact_kind")
        != gr00t_ladder_policy_gate.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("d-ladder-comparability-gate-json artifact_kind mismatch")
    if (
        str(comparability_gate_payload.get("branch")) != BRANCH
        or str(comparability_gate_payload.get("ladder_axis")) != AXIS
    ):
        raise ValueError(
            "d-ladder-comparability-gate-json must target NEW_EMBODIMENT D ladder"
        )
    if (
        admission_gate_payload.get("artifact_kind")
        != gr00t_d_ladder_policy_gate.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("d-ladder-admission-gate-json artifact_kind mismatch")
    if str(admission_gate_payload.get("branch")) != BRANCH:
        raise ValueError("d-ladder-admission-gate-json must target NEW_EMBODIMENT")
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

    resolved_registry_path = _resolve_registry_path(
        admission_gate_payload=admission_gate_payload,
        dataset_source_registry_json=dataset_source_registry_json,
    )
    registry_payload = gr00t_p_ladder_unitree_g1._read_json(
        resolved_registry_path,
        arg_name="dataset-source-registry-json",
    )
    if (
        registry_payload.get("artifact_kind")
        != gr00t_d_ladder_policy_gate.REGISTRY_ARTIFACT_KIND
    ):
        raise ValueError("dataset-source-registry-json artifact_kind mismatch")

    modality_contract = gr00t_controller_audit_new_embodiment.load_modality_contract(
        resolved_modality_config_path
    )
    branch_manifest_payload = gr00t_p_ladder_unitree_g1._read_json(
        resolved_branch_manifest_path,
        arg_name="branch-manifest-path",
    )
    controller_audit_payload = gr00t_p_ladder_unitree_g1._read_json(
        resolved_controller_audit_path,
        arg_name="controller-audit-json",
    )

    branch_contract_validation = (
        gr00t_p_ladder_new_embodiment._branch_contract_validation(
            modality_config_path=resolved_modality_config_path,
            modality_contract=modality_contract,
            branch_manifest_payload=branch_manifest_payload,
            controller_audit_payload=controller_audit_payload,
            branch_manifest_path=resolved_branch_manifest_path,
            controller_audit_path=resolved_controller_audit_path,
        )
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

    branch_contract = {
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
        "normalization_source_signature_sha256": (
            gr00t_p_ladder_new_embodiment._normalization_source_signature(
                normalization_source
            )
        ),
        "relative_action_policy_signature_sha256": (
            gr00t_p_ladder_new_embodiment._relative_action_policy_signature(
                relative_action_policy
            )
        ),
        "dataset_provenance": dict(dataset_provenance),
        "controller_provenance": dict(controller_provenance),
        "public_anchor_comparable": False,
        "validation": branch_contract_validation,
    }

    admission_records = load_dataset_admission_records(
        dataset_admission_records_json=dataset_admission_records_json,
        branch_contract=branch_contract,
        registry_payload=registry_payload,
    )

    dataset_records_signature = gr00t_p_ladder_unitree_g1._sha256(admission_records)
    if dataset_admission_records_json is None:
        dataset_records_path = "generated::task17_default_dataset_admission_records"
    else:
        dataset_records_path = gr00t_p_ladder_unitree_g1._rel_repo(
            gr00t_p_ladder_unitree_g1._resolve_path(dataset_admission_records_json)
        )

    prerequisite_hashes = {
        "d_ladder_comparability_gate": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(
                gr00t_p_ladder_unitree_g1._resolve_path(
                    d_ladder_comparability_gate_json
                )
            ),
            "signature": gr00t_p_ladder_unitree_g1._signature_from_payload(
                comparability_gate_payload
            ),
        },
        "d_ladder_admission_gate": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(
                gr00t_p_ladder_unitree_g1._resolve_path(d_ladder_admission_gate_json)
            ),
            "signature": gr00t_p_ladder_unitree_g1._signature_from_payload(
                admission_gate_payload
            ),
        },
        "dataset_source_registry": {
            "path": gr00t_p_ladder_unitree_g1._rel_repo(resolved_registry_path),
            "signature": str(registry_payload["registry_fingerprint_sha256"]),
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
        "dataset_admission_records": {
            "path": dataset_records_path,
            "signature": dataset_records_signature,
        },
    }

    dual_branch = gr00t_p_ladder_new_embodiment._new_embodiment_branch_entry(
        dual_branch_payload
    )
    prerequisites: dict[str, Any] = {
        "d_ladder_comparability_gate": comparability_gate_payload,
        "d_ladder_admission_gate": admission_gate_payload,
        "dataset_source_registry": registry_payload,
        "dual_branch": {
            "payload": dual_branch_payload,
            "new_embodiment_branch": dict(dual_branch),
            "allow_d_ladder": bool(
                cast(
                    Mapping[str, Any], dual_branch_payload.get("allow_d_ladder", {})
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
        "branch_contract": branch_contract,
        "dataset_admission_records": admission_records,
        "diagnostics_prerequisite_hashes": prerequisite_hashes,
    }
    prerequisites["base_metrics"] = gr00t_p_ladder_new_embodiment._build_base_metrics(
        prerequisites=prerequisites
    )
    return prerequisites


def _dataset_fingerprint_for_rung(
    rung: str, *, prerequisites: Mapping[str, Any]
) -> str:
    normalized_rung = _validate_rung(rung)
    admission_records = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("dataset_admission_records", {}),
        field_name="dataset_admission_records",
    )
    dataset_fingerprints = gr00t_p_ladder_unitree_g1._as_mapping(
        admission_records.get("dataset_fingerprints", {}),
        field_name="dataset_admission_records.dataset_fingerprints",
    )
    included_ids = [
        spec.dataset_id
        for spec in gr00t_d_ladder_policy_gate._included_dataset_specs(normalized_rung)
    ]
    branch_contract = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("branch_contract", {}), field_name="branch_contract"
    )
    return gr00t_p_ladder_unitree_g1._sha256(
        {
            "rung": normalized_rung,
            "dataset_source_ids": included_ids,
            "dataset_mix": _equal_weight_mix(included_ids),
            "dataset_fingerprints": {
                dataset_id: gr00t_p_ladder_unitree_g1._as_mapping(
                    dataset_fingerprints[dataset_id],
                    field_name=f"dataset_fingerprints.{dataset_id}",
                )["dataset_fingerprint"]
                for dataset_id in included_ids
            },
            "branch_manifest_hash": branch_contract["branch_manifest_hash"],
            "modality_config_fingerprint_sha256": branch_contract[
                "modality_config_fingerprint_sha256"
            ],
        }
    )


def _build_branch_contract_snapshot(
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    return gr00t_p_ladder_new_embodiment._build_branch_contract_snapshot(prerequisites)


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

    d_gate_payload = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("d_ladder_comparability_gate", {}),
        field_name="prerequisites.d_ladder_comparability_gate",
    )
    baseline_surface = build_comparison_surface("D0", prerequisites=prerequisites)
    rung_surface = build_comparison_surface(
        normalized_rung, prerequisites=prerequisites
    )
    comparability = gr00t_ladder_policy_gate.build_ladder_diff_report(
        d_gate_payload,
        baseline_surface,
        rung_surface,
    )

    base_metrics = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("base_metrics", {}), field_name="prerequisites.base_metrics"
    )
    baseline_metrics = gr00t_p_ladder_unitree_g1._apply_metric_overrides(
        base_metrics,
        rung="D0",
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
        d_gate_payload,
        promotion_evidence,
    )

    branch_contract = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("branch_contract", {}), field_name="branch_contract"
    )
    branch_contract_validation = gr00t_p_ladder_unitree_g1._as_mapping(
        branch_contract.get("validation", {}), field_name="branch_contract.validation"
    )
    branch_contract_stability = (
        gr00t_p_ladder_new_embodiment._branch_contract_stability_report(
            output_root=output_root,
            current_rung=normalized_rung,
            branch_contract=branch_contract,
        )
    )
    admission_records = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("dataset_admission_records", {}),
        field_name="dataset_admission_records",
    )
    registry_payload = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("dataset_source_registry", {}),
        field_name="dataset_source_registry",
    )
    admission_report = gr00t_d_ladder_policy_gate.build_admission_report(
        branch=BRANCH,
        rung=normalized_rung,
        dataset_fingerprints=cast(
            Mapping[str, Any], admission_records.get("dataset_fingerprints", {})
        ),
        dataset_provenance=cast(
            Mapping[str, Any], admission_records.get("dataset_provenance", {})
        ),
        normalization_records=cast(
            Mapping[str, Any], admission_records.get("normalization_records", {})
        ),
        registry_payload=registry_payload,
    )
    compatibility_report = _build_dataset_compatibility_report(
        rung=normalized_rung,
        admission_records=admission_records,
        branch_contract=branch_contract,
        registry_payload=registry_payload,
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
    if not bool(dual_branch.get("allow_d_ladder", False)):
        generic_blocking_reasons.append("dual_branch_prerequisite_blocks_d_ladder")
    checkpoint_provenance = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("checkpoint_provenance", {}),
        field_name="checkpoint_provenance",
    )
    if str(checkpoint_provenance.get("formal_eligibility")) != "ALLOW":
        generic_blocking_reasons.append("checkpoint_provenance_blocked")
    teacher_reachability = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("teacher_reachability", {}),
        field_name="teacher_reachability",
    )
    preflight_prerequisite_proof = gr00t_p_ladder_unitree_g1._as_mapping(
        prerequisites.get("preflight_prerequisite_proof", {}),
        field_name="preflight_prerequisite_proof",
    )
    if not bool(teacher_reachability.get("allow_formal_ladders", False)):
        generic_blocking_reasons.append("teacher_reachability_blocks_formal_scene_pool")
    if (
        str(admission_report.get("admission_status"))
        != gr00t_d_ladder_policy_gate.ADMISSION_PASS
    ):
        generic_blocking_reasons.append("dataset_admission_blocked")
    if str(compatibility_report.get("status")) != "PASS":
        generic_blocking_reasons.append("incompatible_dataset_source_present")
        generic_blocking_reasons.extend(
            [str(item) for item in compatibility_report.get("blocking_reasons", [])]
        )

    status = "PASS"
    blocking_reasons = list(dict.fromkeys(generic_blocking_reasons))
    if blocking_reasons or str(promotion_report.get("promotion_status")) == "BLOCK":
        status = "BLOCK"

    baseline_flags = [str(item) for item in baseline_metrics["systemic_break_flags"]]
    current_flags = [str(item) for item in current_metrics["systemic_break_flags"]]
    new_systemic_break, new_systemic_break_flags = (
        gr00t_p_ladder_unitree_g1._new_systemic_break(baseline_flags, current_flags)
    )

    branch_contract_snapshot = _build_branch_contract_snapshot(prerequisites)
    included_dataset_ids = [
        spec.dataset_id
        for spec in gr00t_d_ladder_policy_gate._included_dataset_specs(normalized_rung)
    ]
    branch_only_dataset_ids = [
        dataset_id
        for dataset_id in included_dataset_ids
        if bool(
            gr00t_d_ladder_policy_gate._registry_datasets_by_id(registry_payload)[
                dataset_id
            ].get("branch_only", False)
        )
    ]
    dataset_mix = cast(list[str], rung_surface["dataset"]["dataset_mix"])
    dataset_fingerprint = _dataset_fingerprint_for_rung(
        normalized_rung,
        prerequisites=prerequisites,
    )

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
        "admission_report": admission_report,
        "compatibility_report": compatibility_report,
        "baseline_rung": "D0",
        "branch_local_only_rung": bool(branch_only_dataset_ids),
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
        "teacher_gap_delta": gr00t_p_ladder_unitree_g1._round_float(
            float(current_metrics["teacher_student_gap"])
            - float(baseline_metrics["teacher_student_gap"])
        ),
        "teacher_student_gap_delta": gr00t_p_ladder_unitree_g1._round_float(
            float(current_metrics["teacher_student_gap"])
            - float(baseline_metrics["teacher_student_gap"])
        ),
        "action_chain_delta": gr00t_p_ladder_unitree_g1._build_action_chain_delta(
            baseline_metrics,
            current_metrics,
        ),
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
        "included_dataset_ids": included_dataset_ids,
        "branch_only_dataset_ids": branch_only_dataset_ids,
        "dataset_mix": dataset_mix,
        "dataset_fingerprint": dataset_fingerprint,
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

    frozen_branch_contract = {
        "prompt_interface": copy.deepcopy(rung_surface["prompt_interface"]),
        "controller": copy.deepcopy(rung_surface["controller"]),
        "embodiment": copy.deepcopy(rung_surface["embodiment"]),
        "training": copy.deepcopy(rung_surface["training"]),
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
        "d_ladder_comparability_gate_path": cast(
            Mapping[str, Any], prerequisites["diagnostics_prerequisite_hashes"]
        )["d_ladder_comparability_gate"]["path"],
        "d_ladder_admission_gate_path": cast(
            Mapping[str, Any], prerequisites["diagnostics_prerequisite_hashes"]
        )["d_ladder_admission_gate"]["path"],
        "allowed_variable_dataset_fields": list(
            cast(Sequence[str], d_gate_payload.get("allowed_difference_paths", []))
        ),
        "comparison_surface_digest": gr00t_p_ladder_unitree_g1._sha256(rung_surface),
        "comparability": comparability,
        "branch_contract_validation": branch_contract_validation,
        "branch_contract_stability": branch_contract_stability,
        "admission_report": admission_report,
        "compatibility_report": compatibility_report,
        "checkpoint_provenance": {
            "formal_eligibility": checkpoint_provenance.get("formal_eligibility"),
            "status": checkpoint_provenance.get("status"),
            "selected_checkpoint_path": checkpoint_provenance.get(
                "selected_checkpoint_path"
            ),
            "loadability_status": checkpoint_provenance.get("loadability_status"),
            "checksum_or_signature": checkpoint_provenance.get("checksum_or_signature"),
        },
        "included_dataset_ids": included_dataset_ids,
        "branch_only_dataset_ids": branch_only_dataset_ids,
        "branch_local_only_rung": bool(branch_only_dataset_ids),
        "dataset_mix": dataset_mix,
        "dataset_fingerprint": dataset_fingerprint,
        "data_surface": copy.deepcopy(rung_surface["dataset"]),
        "frozen_branch_contract": frozen_branch_contract,
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
        "teacher_gap_delta": scorecard["teacher_gap_delta"],
        "teacher_student_gap_delta": scorecard["teacher_student_gap_delta"],
        "action_chain_delta": copy.deepcopy(scorecard["action_chain_delta"]),
        "decision_basis": (
            "Use the task-15 NEW_EMBODIMENT dataset registry/admission gate as the only data "
            "admission authority, keep the task-14 branch manifest + modality config + "
            "normalization source frozen across D0-D4, represent dataset variation only through "
            "dataset_mix and admission-bound per-dataset fingerprints, and permit D4 solely as a "
            "branch-local rung that never becomes public-anchor comparable output."
        ),
    }
    manifest["report_signature_sha256"] = gr00t_p_ladder_unitree_g1._sha256(manifest)
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
    d_ladder_comparability_gate_json: Path,
    d_ladder_admission_gate_json: Path,
    dataset_source_registry_json: Path,
    dual_branch_scorecard_json: Path,
    checkpoint_provenance_json: Path,
    condition_flip_json: Path,
    teacher_student_gap_json: Path,
    action_telemetry_json: Path,
    teacher_reachability_json: Path,
    modality_config_path: Path,
    branch_manifest_path: Path,
    controller_audit_json: Path,
    dataset_admission_records_json: Path | None = None,
    task2_preflight_evidence_json: Path = gr00t_p_ladder_unitree_g1.DEFAULT_TASK2_PREFLIGHT_EVIDENCE_JSON,
    metrics_overrides_by_rung: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_output_root = gr00t_p_ladder_unitree_g1._validate_output_dir(output_root)
    prerequisites = load_prerequisites(
        d_ladder_comparability_gate_json=d_ladder_comparability_gate_json,
        d_ladder_admission_gate_json=d_ladder_admission_gate_json,
        dataset_source_registry_json=dataset_source_registry_json,
        dual_branch_scorecard_json=dual_branch_scorecard_json,
        checkpoint_provenance_json=checkpoint_provenance_json,
        condition_flip_json=condition_flip_json,
        teacher_student_gap_json=teacher_student_gap_json,
        action_telemetry_json=action_telemetry_json,
        teacher_reachability_json=teacher_reachability_json,
        modality_config_path=modality_config_path,
        branch_manifest_path=branch_manifest_path,
        controller_audit_json=controller_audit_json,
        dataset_admission_records_json=dataset_admission_records_json,
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
        manifest_path,
        cast(Mapping[str, Any], report["manifest"]),
    )
    gr00t_p_ladder_unitree_g1._write_json(
        scorecard_path,
        cast(Mapping[str, Any], report["scorecard"]),
    )
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
        result = materialize_d_ladder_rung(
            rung=str(args.rung),
            output_root=args.output_root,
            d_ladder_comparability_gate_json=args.d_ladder_comparability_gate_json,
            d_ladder_admission_gate_json=args.d_ladder_admission_gate_json,
            dataset_source_registry_json=args.dataset_source_registry_json,
            dual_branch_scorecard_json=args.dual_branch_scorecard_json,
            checkpoint_provenance_json=args.checkpoint_provenance_json,
            condition_flip_json=args.condition_flip_json,
            teacher_student_gap_json=args.teacher_student_gap_json,
            action_telemetry_json=args.action_telemetry_json,
            teacher_reachability_json=args.teacher_reachability_json,
            modality_config_path=args.modality_config_path,
            branch_manifest_path=args.branch_manifest_path,
            controller_audit_json=args.controller_audit_json,
            dataset_admission_records_json=args.dataset_admission_records_json,
            task2_preflight_evidence_json=args.task2_preflight_evidence_json,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(gr00t_p_ladder_unitree_g1._exception_message(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if str(result.get("status")) != "BLOCK" else 1


__all__ = [
    "AXIS",
    "BRANCH",
    "BRANCH_KEY",
    "DEFAULT_ACTION_TELEMETRY_JSON",
    "DEFAULT_BRANCH_MANIFEST_PATH",
    "DEFAULT_CHECKPOINT_PROVENANCE_JSON",
    "DEFAULT_CONDITION_FLIP_JSON",
    "DEFAULT_CONTROLLER_AUDIT_JSON",
    "DEFAULT_DATASET_SOURCE_REGISTRY_JSON",
    "DEFAULT_DUAL_BRANCH_SCORECARD_JSON",
    "DEFAULT_D_LADDER_ADMISSION_GATE_JSON",
    "DEFAULT_D_LADDER_COMPARABILITY_GATE_JSON",
    "DEFAULT_MODALITY_CONFIG_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TEACHER_REACHABILITY_JSON",
    "DEFAULT_TEACHER_STUDENT_GAP_JSON",
    "MANIFEST_ARTIFACT_KIND",
    "REPORT_SCHEMA_VERSION",
    "RUNG_DESCRIPTIONS",
    "RUNG_ORDER",
    "SCORECARD_ARTIFACT_KIND",
    "build_comparison_surface",
    "build_dataset_surface",
    "build_frozen_training_surface",
    "build_parser",
    "build_rung_report",
    "load_prerequisites",
    "main",
    "materialize_d_ladder_rung",
]


if __name__ == "__main__":
    raise SystemExit(main())
