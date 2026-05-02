from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from work.demo_utils import paths as demo_paths
from work.recap.run_manifest import build_run_manifest_from_sources
from work.recap.run_manifest import INDICATOR_SOURCE_FIELD
from work.recap.run_manifest import PROMPT_SOURCE_FIELD
from work.recap.run_manifest import TEXT_CARRIER_ROUTE
from work.recap.run_manifest import TEXT_CARRIER_SCHEMA_VERSION
from work.recap.run_manifest import validate_run_manifest
from work.recap.stage3_contract_precondition_gate import (
    run_stage3_contract_precondition_gate,
)
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts.gr00t_checkpoint_provenance_gate import (
    build_checkpoint_provenance_report,
)
from work.recap.scripts.gr00t_run_manifest_gate import INVALID_RUN_MANIFEST_REASON_CODE
from work.recap.scripts.gr00t_run_manifest_gate import (
    OK_REASON_CODE as RUN_MANIFEST_OK_REASON_CODE,
)
from work.recap.scripts.state_conditioned_common import read_json
from work.recap.scripts.state_conditioned_common import write_json


STAGE3_ITERATION_MANIFEST_V3 = "stage3_iteration_manifest_v3"
STAGE3_ITERATION_MANIFEST_V2 = STAGE3_ITERATION_MANIFEST_V3
STAGE3_CHECKPOINT_PROVENANCE_GATE_SCHEMA_VERSION = (
    "stage3_collect_checkpoint_provenance_gate_v1"
)
STAGE3_CHECKPOINT_PROVENANCE_GATE_ARTIFACT_KIND = (
    "stage3_collect_checkpoint_provenance_gate"
)
STAGE3_RUN_MANIFEST_GATE_SCHEMA_VERSION = "stage3_collect_run_manifest_gate_v1"
STAGE3_RUN_MANIFEST_GATE_ARTIFACT_KIND = "stage3_collect_run_manifest_gate"
STAGE3_COLLECT_POLICY_CKPT_PROVENANCE_SCHEMA_VERSION = (
    "stage3_collect_policy_ckpt_provenance_v1"
)

SUCCESS_GATE_THRESHOLD_COUNT = 3
HISTORICAL_SUCCESS_RATE_THRESHOLD = 0.3
OFFICIAL_TASK_ANCHOR_MODEL = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_STAGE3_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
DEFAULT_STAGE3_ITERATION_MANIFEST_REL = Path(
    "agent/artifacts/stage3_iteration/recap_stage3_iter_002/iteration_manifest.json"
)

CHECKPOINT_PROVENANCE_GATE_JSON_NAME = "checkpoint_provenance_gate.json"
RUN_MANIFEST_GATE_JSON_NAME = "run_manifest_gate.json"

DEFAULT_BRANCH = "UNITREE_G1"
DEFAULT_POLICY_HORIZON = 30
DEFAULT_N_ACTION_STEPS = 20
DEFAULT_TRAINABLE_MODULE_REGEX = "stage3_collect_checkpoint_binding"
DEFAULT_EVAL_OVERLAY_REGEX = "stage3_collect_checkpoint_binding"
DEFAULT_WBC_POLICY_CLASS = "G1DecoupledWholeBodyPolicy"
DEFAULT_RELATIVE_ABSOLUTE_ACTION_CONTRACT: dict[str, Any] = {
    "relative_action_keys": ["left_arm", "right_arm"],
    "absolute_action_keys": ["left_hand", "right_hand", "waist"],
    "action_representation_by_key": {
        "left_arm": "RELATIVE",
        "right_arm": "RELATIVE",
        "left_hand": "ABSOLUTE",
        "right_hand": "ABSOLUTE",
        "waist": "ABSOLUTE",
    },
    "must_not_conflate_horizon_and_execution": True,
}

BASELINE_FREEZE_MATRIX_REL = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/baseline_freeze/baseline_freeze_matrix.json"
)
TASK4_PUBLIC_ANCHOR_EVIDENCE_REL = Path(".sisyphus/evidence/task-4-public-anchor.json")
TASK10_B0_SUITE_EVIDENCE_REL = Path(".sisyphus/evidence/task-10-b0-suite.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = demo_paths.remap_legacy_project_root(path)
    return path.resolve()


def _abspath_preserve_symlink(repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = demo_paths.remap_legacy_project_root(path)
    return Path(os.path.abspath(str(path)))


def _repo_relative_path(repo_root: Path, path: Path | str) -> str:
    resolved = _resolve_path(repo_root, path)
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _default_manifest_path(repo_root: Path) -> Path:
    return _resolve_path(repo_root, DEFAULT_STAGE3_ITERATION_MANIFEST_REL)


def _artifact_root(
    repo_root: Path, manifest_payload: Mapping[str, Any], manifest_path: Path
) -> Path:
    artifact_root_raw = str(manifest_payload.get("artifact_root") or "").strip()
    if artifact_root_raw:
        return _resolve_path(repo_root, artifact_root_raw)
    return manifest_path.parent.resolve()


def _require_manifest_string(
    repo_root: Path,
    manifest_payload: Mapping[str, Any],
    *,
    field_name: str,
    manifest_path: Path,
) -> str:
    raw = str(manifest_payload.get(field_name) or "").strip()
    if not raw:
        raise ValueError(
            f"iteration manifest {manifest_path} missing non-empty {field_name!r}"
        )
    return str(_abspath_preserve_symlink(repo_root, raw))


def _load_training_python_contract_from_manifest(
    repo_root: Path,
    manifest_payload: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, str]:
    return {
        "manifest_path": str(manifest_path.resolve()),
        "orchestrator_python": _require_manifest_string(
            repo_root,
            manifest_payload,
            field_name="orchestrator_python",
            manifest_path=manifest_path,
        ),
        "delegate_runtime_python": _require_manifest_string(
            repo_root,
            manifest_payload,
            field_name="delegate_runtime_python",
            manifest_path=manifest_path,
        ),
    }


def _manifest_env_name(manifest_payload: Mapping[str, Any]) -> str:
    raw = str(manifest_payload.get("env_name") or "").strip()
    return raw if raw else DEFAULT_STAGE3_ENV_NAME


def _manifest_threshold_float(
    manifest_payload: Mapping[str, Any], *, field_name: str, default: float
) -> float:
    value = _maybe_float(manifest_payload.get(field_name))
    return float(default) if value is None else float(value)


def _manifest_threshold_int(
    manifest_payload: Mapping[str, Any], *, field_name: str, default: int
) -> int:
    value = _maybe_int(manifest_payload.get(field_name))
    return int(default) if value is None else int(value)


def _file_entry_like(repo_root: Path, relative_path: str) -> dict[str, object]:
    resolved_path = _resolve_path(repo_root, relative_path)
    entry: dict[str, object] = {
        "relative_path": str(relative_path),
        "resolved_path": str(resolved_path),
        "exists": resolved_path.exists(),
        "path_kind": "missing",
        "content_sha256": None,
        "size_bytes": None,
    }
    if resolved_path.is_file():
        entry["path_kind"] = "file"
        entry["content_sha256"] = apple_recap_execution_contract._sha256_file(
            resolved_path
        )
        entry["size_bytes"] = int(resolved_path.stat().st_size)
    elif resolved_path.is_dir():
        entry["path_kind"] = "directory"
    return entry


def _maybe_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _maybe_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and value.strip() and value.strip().isdigit():
        return int(value.strip())
    return None


def _base_controller_audit(*, env_name: str) -> dict[str, Any]:
    return {
        "controller_provenance": {
            "embodiment_tag": DEFAULT_BRANCH,
            "official_env_name": str(env_name),
            "wbc_policy_class": DEFAULT_WBC_POLICY_CLASS,
        }
    }


def _state_conditioned_metadata(
    *,
    checkpoint_path: str | None,
    base_model_path: str,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    return {
        "comparable_run_spec": {
            "dataset_fingerprint": str(dataset_fingerprint),
            "carrier_schema_version": TEXT_CARRIER_SCHEMA_VERSION,
            "carrier_route": TEXT_CARRIER_ROUTE,
            "prompt_source_field": PROMPT_SOURCE_FIELD,
            "indicator_source": INDICATOR_SOURCE_FIELD,
            "stable_base": {
                "base_model": str(base_model_path),
                "embodiment_tag": DEFAULT_BRANCH,
            },
            "checkpoint_rule": {
                "selected_checkpoint_path": checkpoint_path,
            },
        }
    }


def _finetune_summary(*, checkpoint_path: str | None) -> dict[str, Any]:
    return {
        "selected_checkpoint_path": checkpoint_path,
        "effective_config": {
            "trainable_module_regex": DEFAULT_TRAINABLE_MODULE_REGEX,
        },
    }


def _eval_summary(
    *,
    checkpoint_path: str | None,
    base_model_path: str,
    eval_uses_finetuned: bool,
    server_load_path: str,
) -> dict[str, Any]:
    return {
        "execution_surface_contract": {
            "policy_horizon_expected": DEFAULT_POLICY_HORIZON,
            "n_action_steps": DEFAULT_N_ACTION_STEPS,
            **json.loads(
                json.dumps(DEFAULT_RELATIVE_ABSOLUTE_ACTION_CONTRACT, ensure_ascii=True)
            ),
        },
        "evaluation_binding": {
            "eval_uses_finetuned": bool(eval_uses_finetuned),
            "server_load_mode": "model_path",
            "server_load_path": str(server_load_path),
            "base_model_path": str(base_model_path),
        },
        "server_provenance": {
            "policy_model_path": str(server_load_path),
            "base_model_path": str(base_model_path),
            "overlay_include_regex": DEFAULT_EVAL_OVERLAY_REGEX,
        },
        "selected_checkpoint_path": checkpoint_path,
    }


def _build_preliminary_run_manifest(
    *,
    checkpoint_path: str | None,
    base_model_path: str,
    eval_uses_finetuned: bool,
    dataset_fingerprint: str,
    env_name: str,
) -> dict[str, Any]:
    server_load_path = checkpoint_path if checkpoint_path else base_model_path
    return build_run_manifest_from_sources(
        state_conditioned_metadata=_state_conditioned_metadata(
            checkpoint_path=checkpoint_path,
            base_model_path=base_model_path,
            dataset_fingerprint=dataset_fingerprint,
        ),
        finetune_summary=_finetune_summary(checkpoint_path=checkpoint_path),
        eval_summary=_eval_summary(
            checkpoint_path=checkpoint_path,
            base_model_path=base_model_path,
            eval_uses_finetuned=eval_uses_finetuned,
            server_load_path=server_load_path,
        ),
        controller_audit=_base_controller_audit(env_name=env_name),
        branch=DEFAULT_BRANCH,
        commit=f"stage3_ckpt_binding::{dataset_fingerprint}",
    )


def build_local_candidate_spec(
    *,
    candidate_id: str,
    checkpoint_path: str | Path,
    success_rate: float,
    success_count: int,
    base_model_path: str = OFFICIAL_TASK_ANCHOR_MODEL,
    source_label: str = "historical_fixture",
) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate_id),
        "candidate_tier": "historical_local",
        "decision_on_select": "historical_best",
        "checkpoint_path": str(checkpoint_path),
        "base_model_path": str(base_model_path),
        "success_rate": float(success_rate),
        "success_count": int(success_count),
        "success_rate_source": f"{source_label}.success_rate",
        "success_count_source": f"{source_label}.success_count",
        "candidate_notes": [f"candidate sourced from {source_label}"],
    }


def _build_candidate_provenance_metadata(
    *,
    checkpoint_path: str | None,
    base_model_path: str,
    eval_uses_finetuned: bool,
) -> dict[str, Any]:
    server_load_path = checkpoint_path if checkpoint_path else base_model_path
    return {
        "comparable_run_spec": {
            "stable_base": {
                "base_model": str(base_model_path),
                "embodiment_tag": DEFAULT_BRANCH,
            },
            "checkpoint_rule": {
                "selected_checkpoint_path": checkpoint_path,
            },
        },
        "evaluation_binding": {
            "eval_uses_finetuned": bool(eval_uses_finetuned),
            "server_load_mode": "model_path",
            "server_load_path": str(server_load_path),
            "base_model_path": str(base_model_path),
        },
    }


def _evaluate_candidate(
    *,
    repo_root: Path,
    env_name: str,
    output_dir: Path,
    candidate: Mapping[str, Any],
    success_rate_threshold: float,
    success_count_threshold: int,
    require_success_rate_threshold: bool = True,
    require_success_count_threshold: bool = True,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "unnamed_candidate").strip()
    candidate_tier = str(candidate.get("candidate_tier") or "historical_local").strip()
    decision_on_select = str(
        candidate.get("decision_on_select")
        or (
            "historical_best"
            if candidate_tier == "historical_local"
            else candidate_tier
        )
    ).strip()
    checkpoint_path_raw = candidate.get("checkpoint_path")
    checkpoint_path = None
    if isinstance(checkpoint_path_raw, str) and checkpoint_path_raw.strip():
        checkpoint_path = str(_resolve_path(repo_root, checkpoint_path_raw))
    base_model_path = str(
        candidate.get("base_model_path") or OFFICIAL_TASK_ANCHOR_MODEL
    )
    eval_uses_finetuned = bool(
        candidate.get("eval_uses_finetuned")
        if "eval_uses_finetuned" in candidate
        else checkpoint_path is not None
    )
    dataset_fingerprint = str(
        candidate.get("dataset_fingerprint")
        or f"stage3_collect_binding::{candidate_id}"
    )
    run_manifest_payload = candidate.get("run_manifest_payload")
    if isinstance(run_manifest_payload, Mapping):
        candidate_run_manifest = dict(run_manifest_payload)
    else:
        candidate_run_manifest = _build_preliminary_run_manifest(
            checkpoint_path=checkpoint_path,
            base_model_path=base_model_path,
            eval_uses_finetuned=eval_uses_finetuned,
            dataset_fingerprint=dataset_fingerprint,
            env_name=env_name,
        )

    run_manifest_validation = validate_run_manifest(
        candidate_run_manifest,
        repo_root=repo_root,
    )
    normalized_manifest = dict(run_manifest_validation["normalized_manifest"])
    provenance_metadata = _build_candidate_provenance_metadata(
        checkpoint_path=checkpoint_path,
        base_model_path=base_model_path,
        eval_uses_finetuned=eval_uses_finetuned,
    )
    provenance_report = build_checkpoint_provenance_report(
        metadata=provenance_metadata,
        metadata_path=output_dir / "candidate_provenance_input.json",
        repo_root=repo_root,
        output_dir=output_dir,
    )
    numeric_success_rate = _maybe_float(candidate.get("success_rate"))
    numeric_success_count = _maybe_int(candidate.get("success_count"))
    explicit_checkpoint_identity_present = checkpoint_path is not None
    success_rate_pass = (
        numeric_success_rate is not None
        and numeric_success_rate >= float(success_rate_threshold)
    )
    success_count_pass = (
        numeric_success_count is not None
        and numeric_success_count >= int(success_count_threshold)
    )
    run_manifest_pass = run_manifest_validation["formal_eligibility"] == "ALLOW"
    provenance_pass = provenance_report["formal_eligibility"] == "ALLOW"
    viable = bool(
        explicit_checkpoint_identity_present
        and run_manifest_pass
        and provenance_pass
        and (success_rate_pass or not require_success_rate_threshold)
        and (success_count_pass or not require_success_count_threshold)
    )
    return {
        "candidate_id": candidate_id,
        "candidate_tier": candidate_tier,
        "decision_on_select": decision_on_select,
        "checkpoint_path": checkpoint_path,
        "base_model_path": base_model_path,
        "explicit_checkpoint_identity_present": explicit_checkpoint_identity_present,
        "success_rate": numeric_success_rate,
        "success_count": numeric_success_count,
        "success_rate_source": candidate.get("success_rate_source"),
        "success_count_source": candidate.get("success_count_source"),
        "success_rate_threshold": float(success_rate_threshold),
        "success_count_threshold": int(success_count_threshold),
        "require_success_rate_threshold": bool(require_success_rate_threshold),
        "require_success_count_threshold": bool(require_success_count_threshold),
        "success_rate_threshold_pass": success_rate_pass,
        "success_count_threshold_pass": success_count_pass,
        "candidate_notes": [
            str(item)
            for item in candidate.get("candidate_notes", [])
            if str(item).strip()
        ],
        "authority_refs": [
            dict(item)
            for item in candidate.get("authority_refs", [])
            if isinstance(item, Mapping)
        ],
        "run_manifest_validation": {
            "formal_eligibility": run_manifest_validation["formal_eligibility"],
            "issues": [dict(item) for item in run_manifest_validation["issues"]],
            "core_digest": run_manifest_validation["core_digest"],
            "normalized_manifest": normalized_manifest,
            "checkpoint_binding": dict(run_manifest_validation["checkpoint_binding"]),
        },
        "checkpoint_provenance": dict(provenance_report),
        "viable": viable,
    }


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, int, str]:
    success_rate = _maybe_float(candidate.get("success_rate"))
    success_count = _maybe_int(candidate.get("success_count"))
    candidate_id = str(candidate.get("candidate_id") or "")
    return (
        -1.0 if success_rate is None else float(success_rate),
        -1 if success_count is None else int(success_count),
        candidate_id,
    )


def _discover_local_historical_candidates(
    *,
    repo_root: Path,
    contract_env_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _ = contract_env_name
    return [], {
        "schema_version": "stage3_historical_candidate_scan_v1",
        "mode": "repo_fixed_mapping_v1",
        "repo_root": str(repo_root),
        "historical_candidates_found": 0,
        "selection_ready_candidates": 0,
        "notes": [
            "No repo-local historical checkpoint is currently wired with both explicit checkpoint identity and numeric success-rate evidence for the stage3 iteration env.",
            "T0b therefore treats local historical selection as unavailable unless an explicit candidate surface is injected programmatically or via a future dedicated evidence surface.",
        ],
    }


def _load_official_anchor_candidate(
    *,
    repo_root: Path,
    env_name: str,
    override_paths: Mapping[str, Path | str] | None = None,
) -> dict[str, Any]:
    resolved_override_paths = dict(override_paths or {})
    baseline_freeze_path = _resolve_path(
        repo_root,
        resolved_override_paths.get(
            "baseline_freeze_matrix", BASELINE_FREEZE_MATRIX_REL
        ),
    )
    task4_path = _resolve_path(
        repo_root,
        resolved_override_paths.get(
            "task4_public_anchor_evidence", TASK4_PUBLIC_ANCHOR_EVIDENCE_REL
        ),
    )
    task10_path = _resolve_path(
        repo_root,
        resolved_override_paths.get(
            "task10_b0_suite_evidence", TASK10_B0_SUITE_EVIDENCE_REL
        ),
    )
    baseline_freeze = _read_json(baseline_freeze_path)
    task4 = _read_json(task4_path)
    task10 = _read_json(task10_path)

    b0_baseline = (
        baseline_freeze.get("baselines", {})
        if isinstance(baseline_freeze.get("baselines"), Mapping)
        else {}
    )
    b0_summary = (
        b0_baseline.get("g1_b0_public_anchor", {})
        if isinstance(b0_baseline.get("g1_b0_public_anchor"), Mapping)
        else {}
    )
    b0_summary_payload = (
        b0_summary.get("summary", {})
        if isinstance(b0_summary.get("summary"), Mapping)
        else {}
    )
    task4_success_run = (
        task4.get("verification", {}).get("success_run", {})
        if isinstance(task4.get("verification"), Mapping)
        and isinstance(task4.get("verification", {}).get("success_run"), Mapping)
        else {}
    )
    task10_suite = (
        task10.get("baseline_suite", {}).get("official_comparable_10ep", {})
        if isinstance(task10.get("baseline_suite"), Mapping)
        and isinstance(
            task10.get("baseline_suite", {}).get("official_comparable_10ep"), Mapping
        )
        else {}
    )

    success_rate = _maybe_float(task4_success_run.get("formal_success_rate"))
    if success_rate is None:
        success_rate = _maybe_float(task10_suite.get("success_rate"))
    if success_rate is None:
        success_rate = _maybe_float(
            b0_summary_payload.get("public_anchor_success_rate")
        )

    success_count = _maybe_int(task4_success_run.get("formal_success_count"))
    if success_count is None:
        success_count = _maybe_int(task10_suite.get("success_count"))

    authority_refs = [
        apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="baseline_freeze_matrix",
            authority_role="official_task_anchor_authority",
            relative_path=baseline_freeze_path,
            reject_noncanonical_parts=False,
        ),
        apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="task4_public_anchor_evidence",
            authority_role="official_task_anchor_authority",
            relative_path=task4_path,
            reject_noncanonical_parts=False,
        ),
        apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="task10_b0_suite_evidence",
            authority_role="official_task_anchor_authority",
            relative_path=task10_path,
            reject_noncanonical_parts=False,
        ),
    ]

    candidate = {
        "candidate_id": "official_task_anchor",
        "candidate_tier": "official_task_anchor",
        "decision_on_select": "baseline_train_required",
        "checkpoint_path": None,
        "base_model_path": OFFICIAL_TASK_ANCHOR_MODEL,
        "eval_uses_finetuned": False,
        "dataset_fingerprint": "stage3_collect_binding::official_task_anchor_authority_only",
        "success_rate": success_rate,
        "success_count": success_count,
        "success_rate_source": "task4.formal_success_rate|task10.official_comparable_10ep.success_rate|baseline_freeze_matrix.public_anchor_success_rate",
        "success_count_source": "task4.formal_success_count|task10.official_comparable_10ep.success_count",
        "candidate_notes": [
            "Official task anchor authority is strong locally, but it is evaluated here as an authority surface rather than a repo-local loadable checkpoint binding.",
            f"env_name={env_name}",
        ],
        "authority_refs": authority_refs,
    }
    return candidate


def _build_checkpoint_provenance_gate_payload(
    *,
    repo_root: Path,
    gate_path: Path,
    candidate_eval: Mapping[str, Any],
) -> dict[str, Any]:
    raw_report = dict(candidate_eval["checkpoint_provenance"])
    raw_report["artifact_path"] = str(gate_path.resolve())
    is_base_fallback = raw_report.get("is_base_fallback")
    pass_flag = (
        raw_report.get("formal_eligibility") == "ALLOW" and is_base_fallback is False
    )
    if is_base_fallback is True:
        reason_code = "base_fallback_forbidden"
    elif is_base_fallback is False:
        reason_code = raw_report.get("reason_code")
    else:
        reason_code = "is_base_fallback_missing_fail_closed"
    return {
        "schema_version": STAGE3_CHECKPOINT_PROVENANCE_GATE_SCHEMA_VERSION,
        "artifact_kind": STAGE3_CHECKPOINT_PROVENANCE_GATE_ARTIFACT_KIND,
        "generated_at": _now_iso(),
        "artifact_path": str(gate_path.resolve()),
        "candidate_id": candidate_eval["candidate_id"],
        "candidate_tier": candidate_eval["candidate_tier"],
        "decision_on_select": candidate_eval["decision_on_select"],
        "pass": bool(pass_flag),
        "status": "PASS" if pass_flag else "FAIL",
        "formal_eligibility": "ALLOW" if pass_flag else "BLOCK",
        "reason_code": reason_code,
        "selected_checkpoint_path": candidate_eval.get("checkpoint_path"),
        "is_base_fallback": is_base_fallback,
        "success_rate": candidate_eval.get("success_rate"),
        "success_count": candidate_eval.get("success_count"),
        "explicit_checkpoint_identity_present": candidate_eval.get(
            "explicit_checkpoint_identity_present"
        ),
        "report": raw_report,
        "authority_refs": [
            dict(item)
            for item in candidate_eval.get("authority_refs", [])
            if isinstance(item, Mapping)
        ],
        "report_source": _repo_relative_path(repo_root, gate_path),
    }


def _build_run_manifest_gate_payload(
    *,
    repo_root: Path,
    gate_path: Path,
    candidate_eval: Mapping[str, Any],
) -> dict[str, Any]:
    validation = dict(candidate_eval["run_manifest_validation"])
    pass_flag = validation.get("formal_eligibility") == "ALLOW"
    return {
        "schema_version": STAGE3_RUN_MANIFEST_GATE_SCHEMA_VERSION,
        "artifact_kind": STAGE3_RUN_MANIFEST_GATE_ARTIFACT_KIND,
        "generated_at": _now_iso(),
        "artifact_path": str(gate_path.resolve()),
        "candidate_id": candidate_eval["candidate_id"],
        "candidate_tier": candidate_eval["candidate_tier"],
        "decision_on_select": candidate_eval["decision_on_select"],
        "pass": bool(pass_flag),
        "status": "PASS" if pass_flag else "FAIL",
        "formal_eligibility": validation.get("formal_eligibility"),
        "reason_code": RUN_MANIFEST_OK_REASON_CODE
        if pass_flag
        else INVALID_RUN_MANIFEST_REASON_CODE,
        "selected_checkpoint_path": candidate_eval.get("checkpoint_path"),
        "success_rate": candidate_eval.get("success_rate"),
        "success_count": candidate_eval.get("success_count"),
        "explicit_checkpoint_identity_present": candidate_eval.get(
            "explicit_checkpoint_identity_present"
        ),
        "core_digest": validation.get("core_digest"),
        "core": dict(
            validation.get("normalized_manifest", {}).get("core", {})
            if isinstance(validation.get("normalized_manifest"), Mapping)
            else {}
        ),
        "issues": [
            dict(item)
            for item in validation.get("issues", [])
            if isinstance(item, Mapping)
        ],
        "checkpoint_binding": dict(validation.get("checkpoint_binding", {})),
        "authority_refs": [
            dict(item)
            for item in candidate_eval.get("authority_refs", [])
            if isinstance(item, Mapping)
        ],
        "report_source": _repo_relative_path(repo_root, gate_path),
    }


def _build_collect_policy_ckpt_provenance(
    *,
    repo_root: Path,
    selected_at: str,
    selected_candidate: Mapping[str, Any],
    checkpoint_gate_path: Path,
    run_manifest_gate_path: Path,
    historical_scan: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_gate_payload = read_json(checkpoint_gate_path)
    run_manifest_gate_payload = read_json(run_manifest_gate_path)
    return {
        "schema_version": STAGE3_COLLECT_POLICY_CKPT_PROVENANCE_SCHEMA_VERSION,
        "selected_at": str(selected_at),
        "selected_candidate_id": selected_candidate["candidate_id"],
        "selected_candidate_tier": selected_candidate["candidate_tier"],
        "decision_on_select": selected_candidate["decision_on_select"],
        "selected_checkpoint_path": selected_candidate.get("checkpoint_path"),
        "success_rate": selected_candidate.get("success_rate"),
        "success_count": selected_candidate.get("success_count"),
        "success_rate_source": selected_candidate.get("success_rate_source"),
        "success_count_source": selected_candidate.get("success_count_source"),
        "explicit_checkpoint_identity_present": selected_candidate.get(
            "explicit_checkpoint_identity_present"
        ),
        "success_rate_threshold_pass": selected_candidate.get(
            "success_rate_threshold_pass"
        ),
        "success_count_threshold_pass": selected_candidate.get(
            "success_count_threshold_pass"
        ),
        "checkpoint_provenance_gate": {
            "path": _repo_relative_path(repo_root, checkpoint_gate_path),
            "pass": checkpoint_gate_payload.get("pass"),
            "formal_eligibility": checkpoint_gate_payload.get("formal_eligibility"),
            "reason_code": checkpoint_gate_payload.get("reason_code"),
        },
        "run_manifest_gate": {
            "path": _repo_relative_path(repo_root, run_manifest_gate_path),
            "pass": run_manifest_gate_payload.get("pass"),
            "formal_eligibility": run_manifest_gate_payload.get("formal_eligibility"),
            "reason_code": run_manifest_gate_payload.get("reason_code"),
        },
        "historical_candidate_scan": dict(historical_scan),
        "authority_refs": [
            dict(item)
            for item in selected_candidate.get("authority_refs", [])
            if isinstance(item, Mapping)
        ],
        "candidate_notes": list(selected_candidate.get("candidate_notes", [])),
        "selection_order": [
            "manual_pinned",
            "historical_best",
            "official_task_anchor",
            "baseline_train_required",
        ],
        "viable": bool(selected_candidate.get("viable", False)),
    }


def _upgrade_manifest_payload(
    *,
    manifest_payload: Mapping[str, Any],
    selected_at: str,
    selected_candidate: Mapping[str, Any],
    collect_policy_ckpt_path: str | None,
    collect_policy_ckpt_decision: str,
    collect_policy_ckpt_provenance: Mapping[str, Any],
    checkpoint_gate_payload: Mapping[str, Any],
    run_manifest_gate_payload: Mapping[str, Any],
    historical_success_rate_threshold: float,
    success_gate_threshold_count: int,
) -> dict[str, Any]:
    upgraded = dict(manifest_payload)
    upgraded["schema_version"] = STAGE3_ITERATION_MANIFEST_V3
    upgraded["collect_policy_ckpt_path"] = collect_policy_ckpt_path
    upgraded["collect_policy_ckpt_selected_at"] = str(selected_at)
    upgraded["collect_policy_ckpt_decision"] = str(collect_policy_ckpt_decision)
    upgraded["collect_policy_ckpt_selected_candidate_id"] = str(
        selected_candidate["candidate_id"]
    )
    upgraded["collect_policy_ckpt_selected_candidate_tier"] = str(
        selected_candidate["candidate_tier"]
    )
    upgraded["collect_policy_ckpt_explicit_checkpoint_identity_present"] = bool(
        selected_candidate.get("explicit_checkpoint_identity_present")
    )
    upgraded["collect_policy_ckpt_checkpoint_provenance_gate"] = {
        "json_path": str(checkpoint_gate_payload.get("report_source") or "").strip(),
        "pass": bool(checkpoint_gate_payload.get("pass")),
        "status": str(checkpoint_gate_payload.get("status") or "").strip(),
        "formal_eligibility": str(
            checkpoint_gate_payload.get("formal_eligibility") or ""
        ).strip(),
        "reason_code": str(checkpoint_gate_payload.get("reason_code") or "").strip(),
    }
    upgraded["collect_policy_ckpt_run_manifest_gate"] = {
        "json_path": str(run_manifest_gate_payload.get("report_source") or "").strip(),
        "pass": bool(run_manifest_gate_payload.get("pass")),
        "status": str(run_manifest_gate_payload.get("status") or "").strip(),
        "formal_eligibility": str(
            run_manifest_gate_payload.get("formal_eligibility") or ""
        ).strip(),
        "reason_code": str(run_manifest_gate_payload.get("reason_code") or "").strip(),
    }
    upgraded["collect_policy_ckpt_provenance"] = dict(collect_policy_ckpt_provenance)
    upgraded["success_gate_threshold_count"] = int(success_gate_threshold_count)
    upgraded["historical_success_rate_threshold"] = float(
        historical_success_rate_threshold
    )
    upgraded["official_task_anchor_model"] = OFFICIAL_TASK_ANCHOR_MODEL
    return upgraded


def bind_collect_checkpoint(
    *,
    repo_root: Path,
    manifest_path: Path | None = None,
    manual_checkpoint_path: Path | None = None,
    manual_checkpoint_success_rate: float | None = None,
    manual_checkpoint_success_count: int | None = None,
    historical_candidates: Sequence[Mapping[str, Any]] | None = None,
    official_anchor_paths: Mapping[str, Path | str] | None = None,
) -> dict[str, Any]:
    resolved_repo_root = Path(repo_root).resolve()
    resolved_manifest_path = (
        _resolve_path(resolved_repo_root, manifest_path)
        if manifest_path is not None
        else _default_manifest_path(resolved_repo_root)
    )
    manifest_payload = _read_json(resolved_manifest_path)
    training_python_contract = _load_training_python_contract_from_manifest(
        resolved_repo_root,
        manifest_payload,
        manifest_path=resolved_manifest_path,
    )
    env_name = _manifest_env_name(manifest_payload)
    historical_success_rate_threshold = _manifest_threshold_float(
        manifest_payload,
        field_name="historical_success_rate_threshold",
        default=HISTORICAL_SUCCESS_RATE_THRESHOLD,
    )
    success_gate_threshold_count = _manifest_threshold_int(
        manifest_payload,
        field_name="success_gate_threshold_count",
        default=SUCCESS_GATE_THRESHOLD_COUNT,
    )
    iteration_artifact_root = _artifact_root(
        resolved_repo_root,
        manifest_payload,
        resolved_manifest_path,
    )
    iteration_artifact_root.mkdir(parents=True, exist_ok=True)
    checkpoint_gate_path = (
        iteration_artifact_root / CHECKPOINT_PROVENANCE_GATE_JSON_NAME
    )
    run_manifest_gate_path = iteration_artifact_root / RUN_MANIFEST_GATE_JSON_NAME
    selected_at = _now_iso()

    detected_historical_candidates, historical_scan = (
        _discover_local_historical_candidates(
            repo_root=resolved_repo_root,
            contract_env_name=env_name,
        )
    )
    candidate_specs: list[dict[str, Any]] = [
        dict(item) for item in detected_historical_candidates
    ]
    if historical_candidates is not None:
        candidate_specs = [dict(item) for item in historical_candidates]

    manual_candidate_eval: dict[str, Any] | None = None
    if manual_checkpoint_path is not None:
        manual_candidate_eval = _evaluate_candidate(
            repo_root=resolved_repo_root,
            env_name=env_name,
            output_dir=iteration_artifact_root,
            candidate={
                "candidate_id": "manual_pinned",
                "candidate_tier": "manual_pinned",
                "decision_on_select": "manual_pinned",
                "checkpoint_path": str(manual_checkpoint_path),
                "base_model_path": OFFICIAL_TASK_ANCHOR_MODEL,
                "success_rate": manual_checkpoint_success_rate,
                "success_count": manual_checkpoint_success_count,
                "success_rate_source": "manual_cli.success_rate",
                "success_count_source": "manual_cli.success_count",
                "candidate_notes": [
                    "manual checkpoint candidate injected via 30a CLI arguments"
                ],
            },
            success_rate_threshold=historical_success_rate_threshold,
            success_count_threshold=success_gate_threshold_count,
            require_success_rate_threshold=False,
            require_success_count_threshold=False,
        )

    historical_evals = [
        _evaluate_candidate(
            repo_root=resolved_repo_root,
            env_name=env_name,
            output_dir=iteration_artifact_root,
            candidate=candidate,
            success_rate_threshold=historical_success_rate_threshold,
            success_count_threshold=success_gate_threshold_count,
            require_success_rate_threshold=True,
            require_success_count_threshold=False,
        )
        for candidate in candidate_specs
    ]
    viable_historical_evals = [item for item in historical_evals if item["viable"]]
    viable_historical_evals.sort(key=_candidate_sort_key, reverse=True)

    selected_candidate: dict[str, Any]
    collect_policy_ckpt_decision: str
    if manual_candidate_eval is not None and bool(manual_candidate_eval.get("viable")):
        selected_candidate = manual_candidate_eval
        collect_policy_ckpt_decision = "manual_pinned"
    elif viable_historical_evals:
        selected_candidate = viable_historical_evals[0]
        collect_policy_ckpt_decision = "historical_best"
    else:
        selected_candidate = _evaluate_candidate(
            repo_root=resolved_repo_root,
            env_name=env_name,
            output_dir=iteration_artifact_root,
            candidate=_load_official_anchor_candidate(
                repo_root=resolved_repo_root,
                env_name=env_name,
                override_paths=official_anchor_paths,
            ),
            success_rate_threshold=historical_success_rate_threshold,
            success_count_threshold=success_gate_threshold_count,
        )
        collect_policy_ckpt_decision = "baseline_train_required"

    checkpoint_gate_payload = _build_checkpoint_provenance_gate_payload(
        repo_root=resolved_repo_root,
        gate_path=checkpoint_gate_path,
        candidate_eval=selected_candidate,
    )
    _ = write_json(checkpoint_gate_path, checkpoint_gate_payload)

    run_manifest_gate_payload = _build_run_manifest_gate_payload(
        repo_root=resolved_repo_root,
        gate_path=run_manifest_gate_path,
        candidate_eval=selected_candidate,
    )
    _ = write_json(run_manifest_gate_path, run_manifest_gate_payload)

    collect_policy_ckpt_provenance = _build_collect_policy_ckpt_provenance(
        repo_root=resolved_repo_root,
        selected_at=selected_at,
        selected_candidate=selected_candidate,
        checkpoint_gate_path=checkpoint_gate_path,
        run_manifest_gate_path=run_manifest_gate_path,
        historical_scan={
            **dict(historical_scan),
            "manual_candidate_viable": None
            if manual_candidate_eval is None
            else bool(manual_candidate_eval.get("viable", False)),
            "evaluated_historical_candidates": [
                {
                    "candidate_id": item["candidate_id"],
                    "viable": item["viable"],
                    "success_rate": item["success_rate"],
                    "success_count": item["success_count"],
                    "run_manifest_pass": item["run_manifest_validation"][
                        "formal_eligibility"
                    ]
                    == "ALLOW",
                    "checkpoint_provenance_pass": item["checkpoint_provenance"][
                        "formal_eligibility"
                    ]
                    == "ALLOW",
                }
                for item in historical_evals
            ],
        },
    )
    upgraded_manifest = _upgrade_manifest_payload(
        manifest_payload=manifest_payload,
        selected_at=selected_at,
        selected_candidate=selected_candidate,
        collect_policy_ckpt_path=(
            selected_candidate.get("checkpoint_path")
            if collect_policy_ckpt_decision in {"historical_best", "manual_pinned"}
            else None
        ),
        collect_policy_ckpt_decision=collect_policy_ckpt_decision,
        collect_policy_ckpt_provenance=collect_policy_ckpt_provenance,
        checkpoint_gate_payload=checkpoint_gate_payload,
        run_manifest_gate_payload=run_manifest_gate_payload,
        historical_success_rate_threshold=historical_success_rate_threshold,
        success_gate_threshold_count=success_gate_threshold_count,
    )
    _ = write_json(resolved_manifest_path, upgraded_manifest)
    contract_precondition_gate_payload = run_stage3_contract_precondition_gate(
        repo_root=resolved_repo_root,
        manifest_path=resolved_manifest_path,
    )

    return {
        "manifest_path": str(resolved_manifest_path),
        "checkpoint_provenance_gate_path": str(checkpoint_gate_path),
        "run_manifest_gate_path": str(run_manifest_gate_path),
        "collect_policy_ckpt_decision": collect_policy_ckpt_decision,
        "collect_policy_ckpt_path": upgraded_manifest.get("collect_policy_ckpt_path"),
        "selected_candidate_id": selected_candidate["candidate_id"],
        "selected_candidate_tier": selected_candidate["candidate_tier"],
        "selected_candidate_viable": bool(selected_candidate.get("viable", False)),
        "checkpoint_provenance_gate_pass": bool(checkpoint_gate_payload["pass"]),
        "run_manifest_gate_pass": bool(run_manifest_gate_payload["pass"]),
        "contract_precondition_gate_pass": bool(
            contract_precondition_gate_payload.get("pass")
        ),
        "official_task_anchor_model": OFFICIAL_TASK_ANCHOR_MODEL,
        "training_python_contract": training_python_contract,
    }


__all__ = [
    "CHECKPOINT_PROVENANCE_GATE_JSON_NAME",
    "HISTORICAL_SUCCESS_RATE_THRESHOLD",
    "OFFICIAL_TASK_ANCHOR_MODEL",
    "RUN_MANIFEST_GATE_JSON_NAME",
    "STAGE3_ITERATION_MANIFEST_V2",
    "STAGE3_ITERATION_MANIFEST_V3",
    "SUCCESS_GATE_THRESHOLD_COUNT",
    "bind_collect_checkpoint",
    "build_local_candidate_spec",
]
