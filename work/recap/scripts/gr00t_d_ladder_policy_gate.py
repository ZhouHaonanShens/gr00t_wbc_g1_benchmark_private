from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_DIR = Path("agent/artifacts/gr00t_anchor_controller_recap")

REPORT_SCHEMA_VERSION = "gr00t_d_ladder_policy_gate_v1"
REPORT_ARTIFACT_KIND = "gr00t_d_ladder_policy_gate"
REGISTRY_SCHEMA_VERSION = "gr00t_dataset_source_registry_v1"
REGISTRY_ARTIFACT_KIND = "gr00t_dataset_source_registry"
ADMISSION_REPORT_SCHEMA_VERSION = "gr00t_d_ladder_admission_report_v1"
ADMISSION_REPORT_ARTIFACT_KIND = "gr00t_d_ladder_admission_report"
GATE_NAME = "GR00TDLadderPolicyGate"
DATASET_SOURCE_REGISTRY_JSON_NAME = "dataset_source_registry.json"

BRANCH_UNITREE_G1 = "UNITREE_G1"
BRANCH_NEW_EMBODIMENT = "NEW_EMBODIMENT"
BRANCH_CHOICES = (BRANCH_UNITREE_G1, BRANCH_NEW_EMBODIMENT)

RUNG_D0 = "D0"
RUNG_D1 = "D1"
RUNG_D2 = "D2"
RUNG_D3 = "D3"
RUNG_D4 = "D4"
RUNG_ORDER = (RUNG_D0, RUNG_D1, RUNG_D2, RUNG_D3, RUNG_D4)

GATE_JSON_NAME_BY_BRANCH = {
    BRANCH_UNITREE_G1: "d_ladder_policy_gate_unitree_g1.json",
    BRANCH_NEW_EMBODIMENT: "d_ladder_policy_gate_new_embodiment.json",
}

ELIGIBILITY_ALLOW = "ALLOW"
ELIGIBILITY_ALLOW_WITH_GUARDRAILS = "ALLOW_WITH_GUARDRAILS"
ELIGIBILITY_BLOCK = "BLOCK"

ADMISSION_PASS = "PASS"
ADMISSION_BLOCK = "BLOCK"
ADMISSION_REQUIRES_EVIDENCE = "REQUIRES_ADMISSION_EVIDENCE"

ALLOWED_DIFFERENCE_PATHS: tuple[str, ...] = (
    "dataset.admission.admission_policy_version",
    "dataset.admission.branch_inclusion",
    "dataset.admission.dataset_fingerprints",
    "dataset.admission.dataset_source_ids",
    "dataset.dataset_mix",
    "dataset.normalization.explicit_diff_reason",
    "dataset.normalization.explicit_stats_policy",
    "dataset.normalization.stats_fingerprint",
    "dataset.normalization.stats_owner",
)

FORBIDDEN_DIFFERENCE_PATHS: tuple[str, ...] = (
    "branch.branch_key",
    "branch.branch_scope",
    "branch.public_anchor_comparable",
    "controller.action_horizon",
    "controller.action_keys",
    "controller.controller_family",
    "controller.relative_action_policy",
    "controller.state_keys",
    "dataset.normalization.hidden_stats_fingerprint",
    "dataset.normalization.implicit_cross_branch_stats_reuse",
    "dataset.sampling.episode_sampling_policy",
    "dataset.sampling.seed_policy",
    "embodiment.embodiment_tag",
    "embodiment.modality_config_digest",
    "embodiment.modality_config_path",
    "prompt_interface.condition_injection",
    "prompt_interface.condition_schema",
    "prompt_interface.prompt_template_id",
    "training.optimizer.betas",
    "training.optimizer.eps",
    "training.optimizer.gradient_clip_norm",
    "training.optimizer.learning_rate",
    "training.optimizer.weight_decay",
    "training.parameter_update.lora_enabled",
    "training.parameter_update.lora_rank",
    "training.parameter_update.selective_unfreeze_modules",
    "training.parameter_update.tune_diffusion_model",
    "training.parameter_update.tune_llm",
    "training.parameter_update.tune_projector",
    "training.parameter_update.tune_visual",
    "training.parameter_update.visual_unfreeze",
    "training.schedule.dataloader_num_workers",
    "training.schedule.global_batch_size",
    "training.schedule.gradient_accumulation_steps",
    "training.schedule.max_steps",
    "training.schedule.num_gpus",
    "training.schedule.save_steps",
    "training.schedule.save_total_limit",
    "training.schedule.warmup_ratio",
)

FINGERPRINT_REQUIRED_FIELDS: tuple[str, ...] = (
    "dataset_fingerprint",
    "fingerprint_source",
)

PROVENANCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "source_registry_id",
    "local_dataset_path",
    "controller_family",
    "provenance_complete",
    "guardrails_satisfied",
)

NORMALIZATION_REQUIRED_FIELDS: tuple[str, ...] = (
    "stats_fingerprint",
    "hidden_stats_fingerprint",
    "normalization_owner",
    "explicit_stats_policy",
    "cross_dataset_reuse_declared",
    "regenerated_for_branch",
)

COMMON_ADMISSION_REQUIREMENTS: tuple[str, ...] = (
    "dataset_fingerprint_required_for_every_included_dataset",
    "dataset_provenance_required_for_every_included_dataset",
    "normalization_record_required_for_every_included_dataset",
    "hidden_normalization_drift_blocks_formal_admission",
    "implicit_cross_dataset_or_cross_branch_stats_reuse_is_forbidden",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import


@dataclass(frozen=True)
class BranchSpec:
    branch: str
    branch_key: str
    branch_scope: str
    public_anchor_comparable: bool
    official_comparable_line: bool
    internal_only_comparable_line: bool


@dataclass(frozen=True)
class DatasetSourceSpec:
    rung: str
    dataset_id: str
    dataset_label: str
    dataset_card_url: str | None
    controller_family: str
    branch_only: bool
    branch_only_allowed_branches: tuple[str, ...]
    language_object_diversity_tags: tuple[str, ...]
    source_summary: str
    controller_summary: str
    normalization_owner: str
    normalization_summary: str
    source_refs: tuple[str, ...]


BRANCH_SPECS: dict[str, BranchSpec] = {
    BRANCH_UNITREE_G1: BranchSpec(
        branch=BRANCH_UNITREE_G1,
        branch_key="unitree_g1",
        branch_scope="official_public_anchor_line",
        public_anchor_comparable=True,
        official_comparable_line=True,
        internal_only_comparable_line=False,
    ),
    BRANCH_NEW_EMBODIMENT: BranchSpec(
        branch=BRANCH_NEW_EMBODIMENT,
        branch_key="new_embodiment",
        branch_scope="branch_internal_only",
        public_anchor_comparable=False,
        official_comparable_line=False,
        internal_only_comparable_line=True,
    ),
}


DATASET_SPECS: tuple[DatasetSourceSpec, ...] = (
    DatasetSourceSpec(
        rung=RUNG_D0,
        dataset_id="repo_local::apple_only_base",
        dataset_label="Apple-only baseline data",
        dataset_card_url=None,
        controller_family="GR00T-WholeBodyControl_anchor_protocol",
        branch_only=False,
        branch_only_allowed_branches=(),
        language_object_diversity_tags=("apple_only", "single_task_prompt"),
        source_summary=(
            "Repo-local apple-only baseline rung used as the D0 anchor for both branches."
        ),
        controller_summary=(
            "Represents the canonical GR00T-WholeBodyControl benchmark-side data line."
        ),
        normalization_owner="branch_declared_formal_stats",
        normalization_summary=(
            "Formal D ladders must still record explicit stats ownership and fingerprint, even "
            "for the apple-only baseline."
        ),
        source_refs=(
            ".sisyphus/plans/gr00t-wbc-anchor-controller-recap-attribution.md:1289-1299",
        ),
    ),
    DatasetSourceSpec(
        rung=RUNG_D1,
        dataset_id="nvidia/Arena-G1-Loco-Manipulation-Task",
        dataset_label="Arena G1 loco-manipulation synthetic dataset",
        dataset_card_url=(
            "https://huggingface.co/datasets/nvidia/Arena-G1-Loco-Manipulation-Task"
        ),
        controller_family="unknown",
        branch_only=False,
        branch_only_allowed_branches=(),
        language_object_diversity_tags=(
            "synthetic",
            "g1_loco_manipulation",
            "apple_related_task_family_unproven",
        ),
        source_summary=(
            "Official card proves the exact dataset ID and card URL, but does not prove GR00T "
            "trunk controller family, embodiment tag, or normalization ownership."
        ),
        controller_summary=(
            "Controller lineage is unproven from the dataset card alone; trunk eligibility "
            "therefore remains conditional."
        ),
        normalization_owner="unknown_requires_local_regeneration",
        normalization_summary=(
            "No authoritative stats owner is published; formal mixing requires explicit local "
            "stats regeneration or equally explicit ownership evidence."
        ),
        source_refs=(
            ".sisyphus/notepads/gr00t-wbc-anchor-controller-recap-attribution/learnings.md:5",
            ".sisyphus/notepads/gr00t-wbc-anchor-controller-recap-attribution/issues.md:4",
        ),
    ),
    DatasetSourceSpec(
        rung=RUNG_D2,
        dataset_id="nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1",
        dataset_label="PhysicalAI Robotics GR00T teleop G1 dataset",
        dataset_card_url=(
            "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1"
        ),
        controller_family="upper_body_control_unproven_gr00t_wbc_lineage",
        branch_only=False,
        branch_only_allowed_branches=(),
        language_object_diversity_tags=("real_robot", "teleop", "upper_body_control"),
        source_summary=(
            "Official card proves a real Unitree G1 teleop dataset with 43-dim signals, but "
            "does not prove collection by the GR00T-WholeBodyControl repo or formal trunk stats."
        ),
        controller_summary=(
            "Real G1 teleop provenance is meaningful but trunk equivalence is still unproven "
            "without explicit collection-stack evidence."
        ),
        normalization_owner="unknown_requires_local_regeneration",
        normalization_summary=(
            "Normalization owner is not declared on the card, so no hidden reuse of UNITREE_G1 "
            "trunk stats is allowed."
        ),
        source_refs=(
            ".sisyphus/notepads/gr00t-wbc-anchor-controller-recap-attribution/learnings.md:6",
            ".sisyphus/notepads/gr00t-wbc-anchor-controller-recap-attribution/issues.md:5",
        ),
    ),
    DatasetSourceSpec(
        rung=RUNG_D3,
        dataset_id="LightwheelAI/Lightwheel-Tasks-G1-WBC",
        dataset_label="Lightwheel Tasks G1 WBC dataset",
        dataset_card_url=(
            "https://huggingface.co/datasets/LightwheelAI/Lightwheel-Tasks-G1-WBC"
        ),
        controller_family="G1-Controller-DecoupledWBC",
        branch_only=False,
        branch_only_allowed_branches=(),
        language_object_diversity_tags=("lightwheel", "multi_task", "wbc_variant"),
        source_summary=(
            "Official card proves a Lightwheel WBC dataset with separate controller lineage. "
            "It cannot be silently treated as the NVIDIA UNITREE_G1 trunk data line."
        ),
        controller_summary=(
            "Separate controller provenance requires explicit adapter/mixing proof before the "
            "dataset can be admitted into any formal rung."
        ),
        normalization_owner="unknown_requires_local_regeneration",
        normalization_summary=(
            "Normalization compatibility with the public UNITREE_G1 line is unknown; formal use "
            "must declare branch-local stats ownership and forbid implicit reuse."
        ),
        source_refs=(
            ".sisyphus/notepads/gr00t-wbc-anchor-controller-recap-attribution/learnings.md:7",
            ".sisyphus/notepads/gr00t-wbc-anchor-controller-recap-attribution/issues.md:6",
        ),
    ),
    DatasetSourceSpec(
        rung=RUNG_D4,
        dataset_id="LightwheelAI/Lightwheel-Tasks-G1-Controller",
        dataset_label="Lightwheel Tasks G1 Controller dataset",
        dataset_card_url=(
            "https://huggingface.co/datasets/LightwheelAI/Lightwheel-Tasks-G1-Controller"
        ),
        controller_family="G1-Controller",
        branch_only=True,
        branch_only_allowed_branches=(BRANCH_NEW_EMBODIMENT,),
        language_object_diversity_tags=("lightwheel", "multi_task", "controller_only"),
        source_summary=(
            "Official card proves a distinct G1-Controller dataset. This source is D4 and must "
            "remain branch-only rather than contaminating the WBC trunk."
        ),
        controller_summary=(
            "Uses G1-Controller rather than the public benchmark-side WBC protocol, so formal "
            "admission is restricted to a separate embodiment branch."
        ),
        normalization_owner="branch_only_controller_specific_stats",
        normalization_summary=(
            "Requires branch-specific stats ownership; no cross-branch or cross-controller stats "
            "reuse is permitted."
        ),
        source_refs=(
            ".sisyphus/notepads/gr00t-wbc-anchor-controller-recap-attribution/learnings.md:8-9",
            ".sisyphus/notepads/gr00t-wbc-anchor-controller-recap-attribution/issues.md:6-7",
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_d_ladder_policy_gate.py",
        description=(
            "Materialize the dual-branch D0-D4 data ladder registry and branch-scoped "
            "admission / normalization guardrails."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--branch",
        required=True,
        choices=BRANCH_CHOICES,
        help="Branch whose D-ladder policy gate should be written.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives the dataset source registry and branch-scoped gate JSON.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _rel_repo(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_of_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_required_record(
    mapping: Mapping[str, Any],
    *,
    key: str,
    field_name: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"missing required {field_name}.{key}")
    return _as_mapping(value, field_name=f"{field_name}.{key}")


def _branch_spec(branch: str) -> BranchSpec:
    try:
        return BRANCH_SPECS[str(branch)]
    except KeyError as exc:
        raise ValueError(f"unsupported branch: {branch}") from exc


def _dataset_spec_by_id(dataset_id: str) -> DatasetSourceSpec:
    for spec in DATASET_SPECS:
        if spec.dataset_id == dataset_id:
            return spec
    raise ValueError(f"unknown dataset_id: {dataset_id}")


def _validate_rung(rung: str) -> str:
    normalized = _non_empty_string(rung, field_name="rung")
    if normalized not in RUNG_ORDER:
        raise ValueError(f"unsupported rung: {normalized}")
    return normalized


def _included_dataset_specs(rung: str) -> list[DatasetSourceSpec]:
    normalized_rung = _validate_rung(rung)
    max_index = RUNG_ORDER.index(normalized_rung)
    return [spec for spec in DATASET_SPECS if RUNG_ORDER.index(spec.rung) <= max_index]


def _local_cache_root_policy(spec: DatasetSourceSpec) -> dict[str, Any]:
    return {
        "local_cache_root": None,
        "authority": "repo_policy_operator_config",
        "upstream_authoritative": False,
        "note": (
            "No canonical upstream cache path is asserted here; operators may configure a repo-"
            "local mirror path later, but this registry must not pretend it is an external fact."
        ),
        "suggested_repo_local_subdir": f"agent/artifacts/gr00t_anchor_controller_recap/cache/{spec.rung.lower()}",
    }


def _normalization_policy_for_spec(spec: DatasetSourceSpec) -> dict[str, Any]:
    requires_regeneration = spec.normalization_owner != "branch_declared_formal_stats"
    return {
        "owner": spec.normalization_owner,
        "summary": spec.normalization_summary,
        "cross_dataset_reuse_allowed": False,
        "cross_branch_reuse_allowed": False,
        "delta_indices_bound_to_stats": True,
        "requires_explicit_stats_regeneration": requires_regeneration,
        "missing_owner_blocks": requires_regeneration,
        "hidden_stats_fingerprint_blocks": True,
        "implicit_reuse_blocks": True,
    }


def _branch_eligibility_for_spec(spec: DatasetSourceSpec) -> dict[str, Any]:
    common_allow = {
        "branch_only": False,
        "redirect_branch": None,
    }
    if spec.rung == RUNG_D0:
        return {
            BRANCH_UNITREE_G1: {
                **common_allow,
                "status": ELIGIBILITY_ALLOW,
                "reason_codes": [],
                "summary": "D0 apple-only baseline is the shared data-ladder anchor.",
            },
            BRANCH_NEW_EMBODIMENT: {
                **common_allow,
                "status": ELIGIBILITY_ALLOW,
                "reason_codes": [],
                "summary": "D0 apple-only baseline is also allowed as the branch-local starting point.",
            },
        }
    if spec.rung == RUNG_D1:
        reasons = [
            "controller_family_unproven",
            "embodiment_tag_unproven",
            "normalization_owner_unknown",
        ]
        return {
            BRANCH_UNITREE_G1: {
                **common_allow,
                "status": ELIGIBILITY_ALLOW_WITH_GUARDRAILS,
                "reason_codes": reasons,
                "summary": (
                    "Can enter only after explicit provenance and normalization evidence proves "
                    "no hidden trunk contamination."
                ),
            },
            BRANCH_NEW_EMBODIMENT: {
                **common_allow,
                "status": ELIGIBILITY_ALLOW_WITH_GUARDRAILS,
                "reason_codes": reasons,
                "summary": (
                    "May be admitted to the internal branch only with explicit provenance and "
                    "branch-scoped stats evidence."
                ),
            },
        }
    if spec.rung == RUNG_D2:
        reasons = [
            "gr00t_wholebodycontrol_collection_unproven",
            "embodiment_tag_unproven",
            "normalization_owner_unknown",
        ]
        return {
            BRANCH_UNITREE_G1: {
                **common_allow,
                "status": ELIGIBILITY_ALLOW_WITH_GUARDRAILS,
                "reason_codes": reasons,
                "summary": (
                    "Real G1 teleop data is admissible only if collection-stack provenance and "
                    "stats ownership are made explicit."
                ),
            },
            BRANCH_NEW_EMBODIMENT: {
                **common_allow,
                "status": ELIGIBILITY_ALLOW_WITH_GUARDRAILS,
                "reason_codes": reasons,
                "summary": (
                    "Internal branch may use it only after controller lineage and branch-scoped "
                    "normalization are explicitly recorded."
                ),
            },
        }
    if spec.rung == RUNG_D3:
        return {
            BRANCH_UNITREE_G1: {
                **common_allow,
                "status": ELIGIBILITY_ALLOW_WITH_GUARDRAILS,
                "reason_codes": [
                    "different_controller_provenance_requires_explicit_adapter",
                    "normalization_compatibility_unknown",
                    "must_not_mix_unlabeled_into_trunk",
                ],
                "summary": (
                    "Separate WBC controller provenance may only be admitted with explicit adapter "
                    "and stats proof; unlabeled trunk mixing is forbidden."
                ),
            },
            BRANCH_NEW_EMBODIMENT: {
                **common_allow,
                "status": ELIGIBILITY_ALLOW_WITH_GUARDRAILS,
                "reason_codes": [
                    "separate_controller_provenance",
                    "branch_specific_stats_required",
                    "normalization_compatibility_unknown",
                ],
                "summary": (
                    "Internal branch may use the dataset, but only as a separately attested data "
                    "source with explicit stats ownership."
                ),
            },
        }
    if spec.rung == RUNG_D4:
        return {
            BRANCH_UNITREE_G1: {
                "branch_only": True,
                "redirect_branch": BRANCH_NEW_EMBODIMENT,
                "status": ELIGIBILITY_BLOCK,
                "reason_codes": [
                    "branch_only_dataset",
                    "different_controller_family",
                    "redirect_to_new_embodiment",
                ],
                "summary": (
                    "G1-Controller data is D4 branch-only and cannot enter the UNITREE_G1 WBC trunk."
                ),
            },
            BRANCH_NEW_EMBODIMENT: {
                "branch_only": True,
                "redirect_branch": None,
                "status": ELIGIBILITY_ALLOW_WITH_GUARDRAILS,
                "reason_codes": [
                    "branch_only_dataset",
                    "requires_branch_manifest",
                    "requires_branch_specific_normalization",
                ],
                "summary": (
                    "Allowed only on the separate branch, with explicit branch manifest and "
                    "controller-specific stats evidence."
                ),
            },
        }
    raise ValueError(f"unsupported dataset rung: {spec.rung}")


def _registry_entry(spec: DatasetSourceSpec) -> dict[str, Any]:
    return {
        "rung": spec.rung,
        "dataset_id": spec.dataset_id,
        "dataset_label": spec.dataset_label,
        "dataset_card_url": spec.dataset_card_url,
        "local_cache_root": None,
        "local_cache_root_policy": _local_cache_root_policy(spec),
        "dataset_fingerprint": None,
        "dataset_fingerprint_policy": {
            "required_for_formal_admission": True,
            "fingerprint_source": "operator_capture_or_task16_task17_materialization",
            "missing_fingerprint_status": ADMISSION_BLOCK,
            "comparison_rule": (
                "every included dataset_id must provide a stable fingerprint before a formal D rung is admitted"
            ),
        },
        "controller_family": spec.controller_family,
        "controller_provenance": {
            "controller_family": spec.controller_family,
            "summary": spec.controller_summary,
            "branch_equivalence_proven": spec.rung == RUNG_D0,
        },
        "dataset_provenance": {
            "summary": spec.source_summary,
            "source_refs": list(spec.source_refs),
            "provenance_complete_by_registry": spec.rung == RUNG_D0,
            "missing_runtime_provenance_blocks": True,
        },
        "language_object_diversity_tags": list(spec.language_object_diversity_tags),
        "normalization_owner": spec.normalization_owner,
        "normalization_policy": _normalization_policy_for_spec(spec),
        "branch_eligibility": _branch_eligibility_for_spec(spec),
        "branch_only": spec.branch_only,
        "branch_only_allowed_branches": list(spec.branch_only_allowed_branches),
        "mixing_eligibility": {
            branch: {
                "status": details["status"],
                "branch_only": details["branch_only"],
                "reason_codes": list(details["reason_codes"]),
                "summary": details["summary"],
                "requires_explicit_normalization_record": True,
                "requires_explicit_provenance_record": True,
            }
            for branch, details in _branch_eligibility_for_spec(spec).items()
        },
    }


def build_dataset_source_registry(
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    datasets = [_registry_entry(spec) for spec in DATASET_SPECS]
    payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "artifact_kind": REGISTRY_ARTIFACT_KIND,
        "output_path": _rel_repo(output_path),
        "rung_order": list(RUNG_ORDER),
        "branch_order": list(BRANCH_CHOICES),
        "required_dataset_fields": [
            "dataset_id",
            "dataset_card_url",
            "local_cache_root",
            "controller_family",
            "branch_eligibility",
            "normalization_owner",
        ],
        "local_cache_root_policy": {
            "authority": "repo_policy_operator_config",
            "upstream_authoritative": False,
            "note": (
                "local_cache_root is intentionally not asserted as an upstream fact; null means "
                "the operator must provide any repo-local mirror path explicitly."
            ),
        },
        "normalization_guardrails": {
            "policy_id": "branch_scoped_dataset_normalization_guardrails_v1",
            "hidden_stats_fingerprint_blocks": True,
            "missing_dataset_provenance_blocks": True,
            "cross_dataset_reuse_default": "FORBIDDEN_UNLESS_EXPLICITLY_DECLARED_AND_ATTESTED",
            "delta_indices_bound_to_stats": True,
        },
        "datasets": datasets,
        "dataset_ids": [entry["dataset_id"] for entry in datasets],
    }
    payload["registry_fingerprint_sha256"] = _sha256_of_payload(
        {
            "schema_version": payload["schema_version"],
            "artifact_kind": payload["artifact_kind"],
            "rung_order": payload["rung_order"],
            "datasets": payload["datasets"],
        }
    )
    return payload


def _registry_datasets_by_id(
    registry_payload: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    datasets = registry_payload.get("datasets")
    if not isinstance(datasets, list):
        raise TypeError("registry_payload.datasets must be a list")
    dataset_map: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(datasets):
        dataset = _as_mapping(item, field_name=f"registry_payload.datasets[{index}]")
        dataset_id = _non_empty_string(
            dataset.get("dataset_id"),
            field_name=f"registry_payload.datasets[{index}].dataset_id",
        )
        dataset_map[dataset_id] = dataset
    return dataset_map


def _static_admission_decision(
    *,
    branch: str,
    included_datasets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reason_codes: list[str] = []
    blockers: list[str] = []
    for dataset in included_datasets:
        dataset_id = _non_empty_string(
            dataset.get("dataset_id"), field_name="included_dataset.dataset_id"
        )
        branch_eligibility = _as_mapping(
            _as_mapping(dataset.get("branch_eligibility"), field_name=dataset_id).get(
                branch
            ),
            field_name=f"{dataset_id}.branch_eligibility.{branch}",
        )
        status = _non_empty_string(
            branch_eligibility.get("status"),
            field_name=f"{dataset_id}.branch_eligibility.{branch}.status",
        )
        reason_codes.extend(
            str(item)
            for item in cast(list[Any], branch_eligibility.get("reason_codes", []))
        )
        if status == ELIGIBILITY_BLOCK:
            blockers.extend(
                [
                    f"{code}:{dataset_id}"
                    for code in branch_eligibility.get("reason_codes", [])
                ]
            )

    if blockers:
        return {
            "status": ADMISSION_BLOCK,
            "reason_codes": sorted(set(blockers)),
            "summary": (
                "This rung is statically blocked on the selected branch before runtime evidence is even considered."
            ),
        }
    return {
        "status": ADMISSION_REQUIRES_EVIDENCE,
        "reason_codes": sorted(
            {
                *COMMON_ADMISSION_REQUIREMENTS,
                *reason_codes,
            }
        ),
        "summary": (
            "This rung is policy-admissible only after explicit dataset fingerprint, provenance, and "
            "normalization evidence are supplied."
        ),
    }


def _branch_only_rungs(registry_payload: Mapping[str, Any]) -> list[str]:
    datasets = _registry_datasets_by_id(registry_payload).values()
    return sorted(
        {
            _non_empty_string(item.get("rung"), field_name="dataset.rung")
            for item in datasets
            if bool(item.get("branch_only", False))
        },
        key=RUNG_ORDER.index,
    )


def build_branch_gate_payload(
    *,
    branch: str,
    registry_payload: Mapping[str, Any] | None = None,
    output_path: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    spec = _branch_spec(branch)
    registry = (
        build_dataset_source_registry(output_path=registry_path)
        if registry_payload is None
        else dict(registry_payload)
    )
    dataset_map = _registry_datasets_by_id(registry)
    dataset_rungs: list[dict[str, Any]] = []
    for rung in RUNG_ORDER:
        included_specs = _included_dataset_specs(rung)
        included_dataset_ids = [item.dataset_id for item in included_specs]
        included_datasets = [dataset_map[item] for item in included_dataset_ids]
        decision = _static_admission_decision(
            branch=branch,
            included_datasets=included_datasets,
        )
        dataset_rungs.append(
            {
                "rung": rung,
                "included_dataset_ids": included_dataset_ids,
                "dataset_fingerprints": {
                    dataset_id: {
                        "required": True,
                        "runtime_capture_required": True,
                        "missing_blocks": True,
                        "fingerprint_source": "runtime_admission_artifact",
                    }
                    for dataset_id in included_dataset_ids
                },
                "controller_provenance": {
                    dataset_id: dict(dataset_map[dataset_id]["controller_provenance"])
                    for dataset_id in included_dataset_ids
                },
                "normalization_policy": {
                    dataset_id: dict(dataset_map[dataset_id]["normalization_policy"])
                    for dataset_id in included_dataset_ids
                },
                "mixing_eligibility": {
                    dataset_id: dict(dataset_map[dataset_id]["mixing_eligibility"])[
                        branch
                    ]
                    for dataset_id in included_dataset_ids
                },
                "branch_only_datasets": [
                    dataset_id
                    for dataset_id in included_dataset_ids
                    if bool(dataset_map[dataset_id].get("branch_only", False))
                ],
                "admission_status": decision["status"],
                "admission_reason_codes": list(decision["reason_codes"]),
                "admission_summary": decision["summary"],
                "admission_requirements": list(COMMON_ADMISSION_REQUIREMENTS),
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "gate_name": GATE_NAME,
        "branch": spec.branch,
        "branch_key": spec.branch_key,
        "branch_scope": spec.branch_scope,
        "public_anchor_comparable": spec.public_anchor_comparable,
        "official_comparable_line": spec.official_comparable_line,
        "internal_only_comparable_line": spec.internal_only_comparable_line,
        "output_path": _rel_repo(output_path),
        "dataset_source_registry_path": _rel_repo(registry_path),
        "dataset_source_registry_fingerprint_sha256": registry[
            "registry_fingerprint_sha256"
        ],
        "change_policy": "DATA_ONLY_WITH_EXPLICIT_PROVENANCE_AND_NORMALIZATION_DECLARATIONS",
        "allowed_difference_paths": list(ALLOWED_DIFFERENCE_PATHS),
        "forbidden_difference_paths": list(FORBIDDEN_DIFFERENCE_PATHS),
        "dataset_rung_order": list(RUNG_ORDER),
        "branch_only_rungs": _branch_only_rungs(registry),
        "dataset_fingerprint_policy": {
            "required_for_formal_admission": True,
            "comparison_scope": "every_included_dataset_id",
            "missing_dataset_fingerprint_blocks": True,
            "registry_binding_required": True,
        },
        "controller_provenance": {
            entry["dataset_id"]: dict(entry["controller_provenance"])
            for entry in cast(list[dict[str, Any]], registry["datasets"])
        },
        "normalization_policy": {
            "policy_id": "d_ladder_branch_scoped_normalization_guardrails_v1",
            "cross_dataset_reuse_allowed": False,
            "cross_branch_reuse_allowed": False,
            "hidden_stats_fingerprint_blocks": True,
            "missing_dataset_provenance_blocks": True,
            "dataset_rules": {
                entry["dataset_id"]: dict(entry["normalization_policy"])
                for entry in cast(list[dict[str, Any]], registry["datasets"])
            },
        },
        "mixing_eligibility": {
            entry["dataset_id"]: dict(entry["mixing_eligibility"])[branch]
            for entry in cast(list[dict[str, Any]], registry["datasets"])
        },
        "dataset_rungs": dataset_rungs,
    }


def _collect_field_errors(
    record: Mapping[str, Any],
    *,
    required_fields: Sequence[str],
    record_name: str,
    dataset_id: str,
) -> list[str]:
    errors: list[str] = []
    for field in required_fields:
        value = record.get(field)
        if value is None:
            errors.append(f"missing_{record_name}_{field}:{dataset_id}")
            continue
        if isinstance(value, str) and not value.strip():
            errors.append(f"missing_{record_name}_{field}:{dataset_id}")
    return errors


def build_admission_report(
    *,
    branch: str,
    rung: str,
    dataset_fingerprints: Mapping[str, Any],
    dataset_provenance: Mapping[str, Any],
    normalization_records: Mapping[str, Any],
    registry_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = _branch_spec(branch)
    normalized_rung = _validate_rung(rung)
    registry = (
        build_dataset_source_registry(output_path=None)
        if registry_payload is None
        else dict(registry_payload)
    )
    dataset_map = _registry_datasets_by_id(registry)
    included_specs = _included_dataset_specs(normalized_rung)
    included_ids = [item.dataset_id for item in included_specs]

    reason_codes: list[str] = []
    per_dataset_checks: dict[str, Any] = {}
    for dataset_id in included_ids:
        dataset_entry = _as_mapping(dataset_map[dataset_id], field_name=dataset_id)
        branch_eligibility = _as_mapping(
            _as_mapping(
                dataset_entry.get("branch_eligibility"),
                field_name=f"{dataset_id}.branch_eligibility",
            ).get(branch),
            field_name=f"{dataset_id}.branch_eligibility.{branch}",
        )
        eligibility_status = _non_empty_string(
            branch_eligibility.get("status"),
            field_name=f"{dataset_id}.branch_eligibility.{branch}.status",
        )
        dataset_reasons: list[str] = []
        if eligibility_status == ELIGIBILITY_BLOCK:
            dataset_reasons.extend(
                f"{code}:{dataset_id}"
                for code in branch_eligibility.get("reason_codes", [])
            )

        fingerprint_record = dataset_fingerprints.get(dataset_id)
        if fingerprint_record is None:
            dataset_reasons.append(f"missing_dataset_fingerprint:{dataset_id}")
            parsed_fingerprint: Mapping[str, Any] | None = None
        else:
            parsed_fingerprint = _as_mapping(
                fingerprint_record,
                field_name=f"dataset_fingerprints.{dataset_id}",
            )
            dataset_reasons.extend(
                _collect_field_errors(
                    parsed_fingerprint,
                    required_fields=FINGERPRINT_REQUIRED_FIELDS,
                    record_name="fingerprint",
                    dataset_id=dataset_id,
                )
            )

        provenance_record = dataset_provenance.get(dataset_id)
        if provenance_record is None:
            dataset_reasons.append(f"missing_dataset_provenance:{dataset_id}")
            parsed_provenance: Mapping[str, Any] | None = None
        else:
            parsed_provenance = _as_mapping(
                provenance_record,
                field_name=f"dataset_provenance.{dataset_id}",
            )
            dataset_reasons.extend(
                _collect_field_errors(
                    parsed_provenance,
                    required_fields=PROVENANCE_REQUIRED_FIELDS,
                    record_name="provenance",
                    dataset_id=dataset_id,
                )
            )
            if str(parsed_provenance.get("source_registry_id", "")) != dataset_id:
                dataset_reasons.append(f"source_registry_id_mismatch:{dataset_id}")
            if str(parsed_provenance.get("controller_family", "")) != str(
                dataset_entry.get("controller_family", "")
            ):
                dataset_reasons.append(f"controller_family_mismatch:{dataset_id}")
            if not bool(parsed_provenance.get("provenance_complete", False)):
                dataset_reasons.append(f"dataset_provenance_incomplete:{dataset_id}")
            if eligibility_status == ELIGIBILITY_ALLOW_WITH_GUARDRAILS and not bool(
                parsed_provenance.get("guardrails_satisfied", False)
            ):
                dataset_reasons.append(f"guardrails_not_satisfied:{dataset_id}")

        normalization_record = normalization_records.get(dataset_id)
        if normalization_record is None:
            dataset_reasons.append(f"missing_normalization_record:{dataset_id}")
            parsed_normalization: Mapping[str, Any] | None = None
        else:
            parsed_normalization = _as_mapping(
                normalization_record,
                field_name=f"normalization_records.{dataset_id}",
            )
            dataset_reasons.extend(
                _collect_field_errors(
                    parsed_normalization,
                    required_fields=NORMALIZATION_REQUIRED_FIELDS,
                    record_name="normalization",
                    dataset_id=dataset_id,
                )
            )
            declared_stats_fingerprint = str(
                parsed_normalization.get("stats_fingerprint", "")
            )
            hidden_stats_fingerprint = str(
                parsed_normalization.get(
                    "hidden_stats_fingerprint", declared_stats_fingerprint
                )
            )
            if declared_stats_fingerprint != hidden_stats_fingerprint:
                dataset_reasons.append(f"hidden_normalization_drift:{dataset_id}")
            if bool(parsed_normalization.get("cross_dataset_reuse_declared", False)):
                dataset_reasons.append(f"implicit_stats_reuse_forbidden:{dataset_id}")

            registry_norm = _as_mapping(
                dataset_entry.get("normalization_policy"),
                field_name=f"{dataset_id}.normalization_policy",
            )
            if bool(
                registry_norm.get("requires_explicit_stats_regeneration", False)
            ) and not bool(parsed_normalization.get("regenerated_for_branch", False)):
                dataset_reasons.append(
                    f"explicit_stats_regeneration_not_attested:{dataset_id}"
                )

        per_dataset_checks[dataset_id] = {
            "eligibility_status": eligibility_status,
            "reason_codes": sorted(set(dataset_reasons)),
            "branch_only": bool(dataset_entry.get("branch_only", False)),
        }
        reason_codes.extend(dataset_reasons)

    admission_status = ADMISSION_PASS if not reason_codes else ADMISSION_BLOCK
    return {
        "schema_version": ADMISSION_REPORT_SCHEMA_VERSION,
        "artifact_kind": ADMISSION_REPORT_ARTIFACT_KIND,
        "branch": spec.branch,
        "branch_scope": spec.branch_scope,
        "public_anchor_comparable": spec.public_anchor_comparable,
        "rung": normalized_rung,
        "included_dataset_ids": included_ids,
        "admission_status": admission_status,
        "reason_codes": sorted(set(reason_codes)),
        "checks": {
            "dataset_fingerprint_complete": not any(
                code.startswith("missing_dataset_fingerprint:")
                or code.startswith("missing_fingerprint_")
                for code in reason_codes
            ),
            "dataset_provenance_complete": not any(
                code.startswith("missing_dataset_provenance:")
                or code.startswith("missing_provenance_")
                or code.startswith("dataset_provenance_incomplete:")
                for code in reason_codes
            ),
            "normalization_guardrails_pass": not any(
                code.startswith("hidden_normalization_drift:")
                or code.startswith("missing_normalization_")
                or code.startswith("implicit_stats_reuse_forbidden:")
                or code.startswith("explicit_stats_regeneration_not_attested:")
                for code in reason_codes
            ),
            "branch_only_enforced": not any(
                code.startswith("branch_only_dataset:")
                or code.startswith("redirect_to_new_embodiment:")
                for code in reason_codes
            )
            if branch == BRANCH_NEW_EMBODIMENT
            else not any(
                code.startswith("branch_only_dataset:") for code in reason_codes
            ),
        },
        "per_dataset_checks": per_dataset_checks,
    }


def materialize_branch_gate(*, branch: str, output_dir: Path) -> dict[str, Any]:
    resolved_output_dir = _validate_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    registry_path = (resolved_output_dir / DATASET_SOURCE_REGISTRY_JSON_NAME).resolve()
    gate_path = (resolved_output_dir / GATE_JSON_NAME_BY_BRANCH[branch]).resolve()

    registry_payload = build_dataset_source_registry(output_path=registry_path)
    gate_payload = build_branch_gate_payload(
        branch=branch,
        registry_payload=registry_payload,
        output_path=gate_path,
        registry_path=registry_path,
    )

    _write_json(registry_path, registry_payload)
    _write_json(gate_path, gate_payload)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "branch": branch,
        "branch_scope": gate_payload["branch_scope"],
        "public_anchor_comparable": gate_payload["public_anchor_comparable"],
        "dataset_source_registry_path": str(registry_path),
        "d_ladder_policy_gate_path": str(gate_path),
        "branch_only_rungs": list(gate_payload["branch_only_rungs"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = materialize_branch_gate(
            branch=str(args.branch),
            output_dir=cast(Path, args.output_dir),
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ADMISSION_BLOCK",
    "ADMISSION_PASS",
    "ADMISSION_REQUIRES_EVIDENCE",
    "ADMISSION_REPORT_ARTIFACT_KIND",
    "ADMISSION_REPORT_SCHEMA_VERSION",
    "ALLOWED_DIFFERENCE_PATHS",
    "BRANCH_CHOICES",
    "BRANCH_NEW_EMBODIMENT",
    "BRANCH_SPECS",
    "BRANCH_UNITREE_G1",
    "COMMON_ADMISSION_REQUIREMENTS",
    "DATASET_SOURCE_REGISTRY_JSON_NAME",
    "DEFAULT_OUTPUT_DIR",
    "ELIGIBILITY_ALLOW",
    "ELIGIBILITY_ALLOW_WITH_GUARDRAILS",
    "ELIGIBILITY_BLOCK",
    "FORBIDDEN_DIFFERENCE_PATHS",
    "FINGERPRINT_REQUIRED_FIELDS",
    "GATE_JSON_NAME_BY_BRANCH",
    "GATE_NAME",
    "NORMALIZATION_REQUIRED_FIELDS",
    "PROVENANCE_REQUIRED_FIELDS",
    "REGISTRY_ARTIFACT_KIND",
    "REGISTRY_SCHEMA_VERSION",
    "REPORT_ARTIFACT_KIND",
    "REPORT_SCHEMA_VERSION",
    "RUNG_D0",
    "RUNG_D1",
    "RUNG_D2",
    "RUNG_D3",
    "RUNG_D4",
    "RUNG_ORDER",
    "build_admission_report",
    "build_branch_gate_payload",
    "build_dataset_source_registry",
    "build_parser",
    "main",
    "materialize_branch_gate",
]


if __name__ == "__main__":
    raise SystemExit(main())
