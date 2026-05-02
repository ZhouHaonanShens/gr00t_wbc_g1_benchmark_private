from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_DIR = Path("agent/artifacts/gr00t_anchor_controller_recap/ladder_policy")

REPORT_SCHEMA_VERSION = "gr00t_ladder_policy_gate_v1"
REPORT_ARTIFACT_KIND = "gr00t_ladder_policy_gate"
GATE_NAME = "GR00TLadderPolicyGate"

BRANCH_UNITREE_G1 = "UNITREE_G1"
BRANCH_NEW_EMBODIMENT = "NEW_EMBODIMENT"
BRANCH_CHOICES = (BRANCH_UNITREE_G1, BRANCH_NEW_EMBODIMENT)

AXIS_P = "P"
AXIS_D = "D"
AXIS_CHOICES = (AXIS_P, AXIS_D)

AXIS_LABELS = {
    AXIS_P: "parameter_ladder",
    AXIS_D: "data_ladder",
}

GATE_JSON_NAME_BY_BRANCH: dict[str, dict[str, str]] = {
    BRANCH_UNITREE_G1: {
        AXIS_P: "p_ladder_policy_gate_unitree_g1.json",
        AXIS_D: "d_ladder_policy_gate_unitree_g1.json",
    },
    BRANCH_NEW_EMBODIMENT: {
        AXIS_P: "p_ladder_policy_gate_new_embodiment.json",
        AXIS_D: "d_ladder_policy_gate_new_embodiment.json",
    },
}

PROMOTION_REQUIREMENT_ORDER: tuple[str, ...] = (
    "fixed_replicate_policy",
    "fixed_seed_policy",
    "no_systemic_break",
    "provenance_pass",
    "diagnostics_not_regressing",
    "no_single_lucky_seed_promotion",
)

PROMOTION_REQUIREMENTS = {
    "fixed_replicate_policy": {
        "required": True,
        "rule": "replicate_count_and_accounting_must_remain_fixed_across_compared_rungs",
        "why": "promotion cannot rely on a one-off rerun with different replicate accounting",
    },
    "fixed_seed_policy": {
        "required": True,
        "rule": "seed_manifest_or_fixed_seed_protocol_must_match_across_compared_rungs",
        "why": "seed changes would blur attribution between parameter/data edits and sampling luck",
    },
    "no_systemic_break": {
        "required": True,
        "rule": "systemic_break_flags_must_be_empty_before_promotion",
        "why": "formal ladder promotion is invalid if the stack or controller is already broken",
    },
    "provenance_pass": {
        "required": True,
        "rule": "checkpoint_dataset_and_branch_provenance_gates_must_all_pass",
        "why": "promotion is blocked when provenance cannot prove the compared artifact belongs to the same branch line",
    },
    "diagnostics_not_regressing": {
        "required": True,
        "rule": "diagnostic_watchlists_may_exist_but_must_not_regress_when_promoting",
        "why": "the plan allows diagnostics watchlists, but promotion cannot hide a worse controller/condition/profile state",
    },
    "no_single_lucky_seed_promotion": {
        "required": True,
        "rule": "single_lucky_seed_only_improvement_must_not_trigger_promotion",
        "why": "the ladder must reject accidental wins that do not survive fixed-seed replication",
    },
}

COMMON_FORBIDDEN_IO_SURFACES: tuple[str, ...] = (
    "branch.branch_key",
    "branch.branch_scope",
    "branch.public_anchor_comparable",
    "embodiment.embodiment_tag",
    "embodiment.modality_config_path",
    "embodiment.modality_config_digest",
    "controller.controller_family",
    "controller.action_horizon",
    "controller.relative_action_policy",
    "controller.action_keys",
    "controller.state_keys",
    "prompt_interface.prompt_template_id",
    "prompt_interface.condition_injection",
    "prompt_interface.condition_schema",
)

P_ALLOWED_DIFFERENCE_PATHS: tuple[str, ...] = (
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

P_FORBIDDEN_DIFFERENCE_PATHS: tuple[str, ...] = COMMON_FORBIDDEN_IO_SURFACES + (
    "dataset.admission.admission_policy_version",
    "dataset.admission.branch_inclusion",
    "dataset.admission.dataset_fingerprints",
    "dataset.admission.dataset_source_ids",
    "dataset.dataset_mix",
    "dataset.normalization.explicit_diff_reason",
    "dataset.normalization.explicit_stats_policy",
    "dataset.normalization.hidden_stats_fingerprint",
    "dataset.normalization.implicit_cross_branch_stats_reuse",
    "dataset.normalization.stats_fingerprint",
    "dataset.normalization.stats_owner",
    "dataset.sampling.episode_sampling_policy",
    "dataset.sampling.seed_policy",
)

D_ALLOWED_DIFFERENCE_PATHS: tuple[str, ...] = (
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

D_FORBIDDEN_DIFFERENCE_PATHS: tuple[str, ...] = COMMON_FORBIDDEN_IO_SURFACES + (
    "dataset.normalization.hidden_stats_fingerprint",
    "dataset.normalization.implicit_cross_branch_stats_reuse",
    "dataset.sampling.episode_sampling_policy",
    "dataset.sampling.seed_policy",
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

REGRESSION_BLOCKERS_BY_AXIS = {
    AXIS_P: (
        "controller_embodiment_prompt_interface_drift",
        "dataset_admission_or_normalization_drift",
        "unexpected_unclassified_difference_surface",
        "single_lucky_seed_only_improvement",
        "systemic_break_flags_present",
        "provenance_gate_failed",
        "diagnostics_regressed",
    ),
    AXIS_D: (
        "parameter_or_training_scope_drift",
        "controller_embodiment_prompt_interface_drift",
        "hidden_normalization_or_sampling_drift",
        "unexpected_unclassified_difference_surface",
        "single_lucky_seed_only_improvement",
        "systemic_break_flags_present",
        "provenance_gate_failed",
        "diagnostics_regressed",
    ),
}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_ladder_policy_gate.py",
        description=(
            "Freeze machine-readable P-ladder and D-ladder comparability policies for "
            "UNITREE_G1 and NEW_EMBODIMENT."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--branch",
        required=True,
        choices=BRANCH_CHOICES,
        help="Embodiment branch whose ladder policies should be materialized.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives branch-scoped P/D ladder policy gate JSON files.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _rel_repo(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _flatten_differences(
    left: object,
    right: object,
    *,
    prefix: str = "",
) -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        keys = sorted({*left.keys(), *right.keys()})
        for key in keys:
            key_str = str(key)
            child_prefix = key_str if not prefix else f"{prefix}.{key_str}"
            if key not in left or key not in right:
                differences.append(child_prefix)
                continue
            differences.extend(
                _flatten_differences(left[key], right[key], prefix=child_prefix)
            )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if left == right:
            return []
        return [prefix]
    if left != right:
        return [prefix]
    return []


def _branch_spec(branch: str) -> BranchSpec:
    try:
        return BRANCH_SPECS[str(branch)]
    except KeyError as exc:
        raise ValueError(f"unsupported branch: {branch}") from exc


def _axis_policy(
    axis: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    if axis == AXIS_P:
        return (
            P_ALLOWED_DIFFERENCE_PATHS,
            P_FORBIDDEN_DIFFERENCE_PATHS,
            REGRESSION_BLOCKERS_BY_AXIS[AXIS_P],
            "PARAMETER_ONLY_WHITELIST",
        )
    if axis == AXIS_D:
        return (
            D_ALLOWED_DIFFERENCE_PATHS,
            D_FORBIDDEN_DIFFERENCE_PATHS,
            REGRESSION_BLOCKERS_BY_AXIS[AXIS_D],
            "DATA_ONLY_WHITELIST_WITH_EXPLICIT_NORMALIZATION_DIFFS",
        )
    raise ValueError(f"unsupported ladder axis: {axis}")


def _gate_path(output_dir: Path, *, branch: str, axis: str) -> Path:
    return (output_dir / GATE_JSON_NAME_BY_BRANCH[branch][axis]).resolve()


def build_ladder_policy_gate(
    *,
    branch: str,
    axis: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    spec = _branch_spec(branch)
    allowed_paths, forbidden_paths, regression_blockers, change_policy = _axis_policy(
        axis
    )
    comparability_summary = (
        "P-ladder allows only parameter-update scope plus directly related training/optimizer fields."
        if axis == AXIS_P
        else "D-ladder allows only dataset mix, admission/branch inclusion, and explicit normalization diffs."
    )
    gate_payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "gate_name": GATE_NAME,
        "ladder_axis": axis,
        "ladder_axis_name": AXIS_LABELS[axis],
        "branch": spec.branch,
        "branch_key": spec.branch_key,
        "branch_scope": spec.branch_scope,
        "public_anchor_comparable": spec.public_anchor_comparable,
        "official_comparable_line": spec.official_comparable_line,
        "internal_only_comparable_line": spec.internal_only_comparable_line,
        "comparability_status": "PASS",
        "blocking_reasons": [],
        "change_policy": change_policy,
        "comparability_summary": comparability_summary,
        "allowed_difference_paths": list(allowed_paths),
        "forbidden_difference_paths": list(forbidden_paths),
        "promotion_requirements": {
            key: dict(PROMOTION_REQUIREMENTS[key])
            for key in PROMOTION_REQUIREMENT_ORDER
        },
        "regression_blockers": list(regression_blockers),
        "output_path": _rel_repo(output_path),
    }
    return gate_payload


def _blocker_for_path(axis: str, path: str) -> str:
    if (
        path.startswith("controller.")
        or path.startswith("embodiment.")
        or path.startswith("prompt_interface.")
        or path.startswith("branch.")
    ):
        return "controller_embodiment_prompt_interface_drift"
    if axis == AXIS_P and path.startswith("dataset."):
        return "dataset_admission_or_normalization_drift"
    if axis == AXIS_D and path.startswith("training."):
        return "parameter_or_training_scope_drift"
    if axis == AXIS_D and (
        path.startswith("dataset.normalization.hidden")
        or path.startswith("dataset.normalization.implicit")
        or path.startswith("dataset.sampling.")
    ):
        return "hidden_normalization_or_sampling_drift"
    return "unexpected_unclassified_difference_surface"


def build_ladder_diff_report(
    gate_payload: Mapping[str, Any],
    reference_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> dict[str, Any]:
    axis = str(gate_payload.get("ladder_axis"))
    if axis not in AXIS_CHOICES:
        raise ValueError(
            f"gate_payload.ladder_axis must be one of {AXIS_CHOICES}, got {axis!r}"
        )
    _ = _branch_spec(str(gate_payload.get("branch")))

    observed_difference_paths = sorted(
        _flatten_differences(reference_payload, candidate_payload)
    )
    allowed = {str(path) for path in gate_payload.get("allowed_difference_paths", [])}
    forbidden = {
        str(path) for path in gate_payload.get("forbidden_difference_paths", [])
    }

    offending_field_paths = [
        path for path in observed_difference_paths if path in forbidden
    ]
    unexpected_difference_paths = [
        path
        for path in observed_difference_paths
        if path not in allowed and path not in forbidden
    ]
    triggered_regression_blockers = sorted(
        {
            _blocker_for_path(axis, path)
            for path in [*offending_field_paths, *unexpected_difference_paths]
        }
    )
    blocking_reasons: list[str] = []
    if offending_field_paths:
        blocking_reasons.append("forbidden_difference_paths_present")
    if unexpected_difference_paths:
        blocking_reasons.append("unexpected_difference_paths_present")

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "gate_name": str(gate_payload.get("gate_name", GATE_NAME)),
        "ladder_axis": axis,
        "ladder_axis_name": str(
            gate_payload.get("ladder_axis_name", AXIS_LABELS[axis])
        ),
        "branch": str(gate_payload.get("branch")),
        "branch_scope": str(gate_payload.get("branch_scope")),
        "public_anchor_comparable": bool(gate_payload.get("public_anchor_comparable")),
        "comparability_status": "PASS" if not blocking_reasons else "BLOCK",
        "allowed_difference_paths": sorted(allowed),
        "forbidden_difference_paths": sorted(forbidden),
        "observed_difference_paths": observed_difference_paths,
        "offending_field_paths": offending_field_paths,
        "unexpected_difference_paths": unexpected_difference_paths,
        "triggered_regression_blockers": triggered_regression_blockers,
        "blocking_reasons": blocking_reasons,
    }


def build_promotion_report(
    gate_payload: Mapping[str, Any],
    promotion_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    requirement_spec = _as_mapping(
        gate_payload.get("promotion_requirements", {}),
        field_name="gate_payload.promotion_requirements",
    )
    checks = {
        "fixed_replicate_policy": _as_bool(
            promotion_evidence.get("fixed_replicate_policy", False),
            field_name="promotion_evidence.fixed_replicate_policy",
        ),
        "fixed_seed_policy": _as_bool(
            promotion_evidence.get("fixed_seed_policy", False),
            field_name="promotion_evidence.fixed_seed_policy",
        ),
        "no_systemic_break": _as_bool(
            promotion_evidence.get("no_systemic_break", False),
            field_name="promotion_evidence.no_systemic_break",
        ),
        "provenance_pass": _as_bool(
            promotion_evidence.get("provenance_pass", False),
            field_name="promotion_evidence.provenance_pass",
        ),
        "diagnostics_not_regressing": _as_bool(
            promotion_evidence.get("diagnostics_not_regressing", False),
            field_name="promotion_evidence.diagnostics_not_regressing",
        ),
        "no_single_lucky_seed_promotion": not _as_bool(
            promotion_evidence.get("single_lucky_seed_only_improvement", False),
            field_name="promotion_evidence.single_lucky_seed_only_improvement",
        ),
    }
    failure_reasons = [
        f"{name}_required"
        for name in PROMOTION_REQUIREMENT_ORDER
        if not bool(checks[name])
        and bool(
            _as_mapping(requirement_spec[name], field_name=name).get("required", False)
        )
    ]
    return {
        "ladder_axis": str(gate_payload.get("ladder_axis")),
        "branch": str(gate_payload.get("branch")),
        "branch_scope": str(gate_payload.get("branch_scope")),
        "promotion_allowed": len(failure_reasons) == 0,
        "promotion_status": "PASS" if not failure_reasons else "BLOCK",
        "checks": checks,
        "failure_reasons": failure_reasons,
    }


def materialize_ladder_policy_gates(*, branch: str, output_dir: Path) -> dict[str, Any]:
    resolved_output_dir = _validate_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    p_path = _gate_path(resolved_output_dir, branch=branch, axis=AXIS_P)
    d_path = _gate_path(resolved_output_dir, branch=branch, axis=AXIS_D)

    p_payload = build_ladder_policy_gate(branch=branch, axis=AXIS_P, output_path=p_path)
    d_payload = build_ladder_policy_gate(branch=branch, axis=AXIS_D, output_path=d_path)

    _write_json(p_path, p_payload)
    _write_json(d_path, d_payload)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "branch": branch,
        "branch_scope": p_payload["branch_scope"],
        "public_anchor_comparable": p_payload["public_anchor_comparable"],
        "p_ladder_policy_gate_path": str(p_path),
        "d_ladder_policy_gate_path": str(d_path),
        "comparability_status": "PASS",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = materialize_ladder_policy_gates(
            branch=str(args.branch),
            output_dir=args.output_dir,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AXIS_CHOICES",
    "AXIS_D",
    "AXIS_LABELS",
    "AXIS_P",
    "BRANCH_CHOICES",
    "BRANCH_NEW_EMBODIMENT",
    "BRANCH_SPECS",
    "BRANCH_UNITREE_G1",
    "D_ALLOWED_DIFFERENCE_PATHS",
    "D_FORBIDDEN_DIFFERENCE_PATHS",
    "DEFAULT_OUTPUT_DIR",
    "GATE_JSON_NAME_BY_BRANCH",
    "GATE_NAME",
    "P_ALLOWED_DIFFERENCE_PATHS",
    "P_FORBIDDEN_DIFFERENCE_PATHS",
    "PROMOTION_REQUIREMENTS",
    "PROMOTION_REQUIREMENT_ORDER",
    "REGRESSION_BLOCKERS_BY_AXIS",
    "REPORT_ARTIFACT_KIND",
    "REPORT_SCHEMA_VERSION",
    "build_ladder_diff_report",
    "build_ladder_policy_gate",
    "build_parser",
    "build_promotion_report",
    "main",
    "materialize_ladder_policy_gates",
]


if __name__ == "__main__":
    raise SystemExit(main())
