#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap import launch_finetune_use_ddp as trainability_entrypoint
from work.recap import finetune_full
from work.recap.train_scope_audit import FULL_UPDATE_SCOPE_NAMES
from work.recap.train_scope_audit import parse_scope_flag


ATTEMPT_STATUS_CHOICES: tuple[str, ...] = (
    "OOM",
    "MEMORY_ESTIMATOR_BLOCK",
    "PASS",
    "BLOCK",
)
RESOLUTION_STATUS_CHOICES: tuple[str, ...] = ("PASS", "DEGRADE", "BLOCK")
TRAIN_SCOPE_EFFECTIVE_CHOICES: tuple[str, ...] = FULL_UPDATE_SCOPE_NAMES
DYNAMIC_AUDIT_SCHEMA_VERSION = "recap_full_update_scope_audit_dynamic_v1"
DYNAMIC_AUDIT_ARTIFACT_KIND = "recap_full_update_scope_audit_dynamic"
P0_BLOCK_REPORT_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/p0_scope_audit/p0_block_report.md"
)
P0_BLOCK_EVIDENCE_REL = Path(".sisyphus/evidence/task-p0-block.md")
_OOM_SIGNATURES: tuple[str, ...] = (
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "out of memory",
)
_REQUIRED_SUCCESS_EVIDENCE_SCOPES: tuple[str, ...] = (
    "diffusion_trunk",
    "advantage_embedding",
)
_TASK11_FORMAL_SKIP_FILENAME = "formal_run_skipped.json"
_TASK11_FORMAL_CONDITIONING_PROBE_FILENAME = "conditioning_functional_probe.json"
_TASK11_SCOPE_DRIFT_FAILURE_REASON = "P3_SCOPE_DRIFT_BLOCK"
_TASK11_ARTIFACT_STAGING_FAILURE_REASON = "FORMAL_LANE_ARTIFACT_STAGING_BLOCK"
_TASK12_PARITY_BLOCK_FAILURE_REASON = "CONTINUATION_PARITY_BLOCK"
_TASK12_CONDITIONED_PEER_SKIP_FILENAME = "formal_run_skipped.json"


class _ContinuationParityError(RuntimeError):
    pass


def _repo_root() -> Path:
    return REPO_ROOT


def _load_smoke_module() -> Any:
    return importlib.import_module("work.recap.scripts.34b_recap_numeric_adv_smoke")


def _write_text_atomic(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def _resolve_summary_json_path(summary_json: object) -> Path | None:
    raw = str(summary_json or "").strip()
    if raw == "":
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path.resolve()


def _write_optional_summary_json(args: Any, payload: Mapping[str, Any]) -> Path | None:
    path = _resolve_summary_json_path(getattr(args, "summary_json", ""))
    if path is None:
        return None
    return _write_text_atomic(
        path,
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    return _write_text_atomic(
        path,
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _task11_formal_gate_enabled(args: Any) -> bool:
    return bool(getattr(args, "require_p3_formal_eligible", False))


def _task11_formal_entrypoint(args: Any) -> bool:
    return str(getattr(args, "entrypoint", "")).strip() == "conditioned"


def _task12_formal_entrypoint(args: Any) -> bool:
    return str(getattr(args, "entrypoint", "")).strip() == "continuation"


def _formal_entrypoint_name(args: Any) -> str:
    raw = str(getattr(args, "entrypoint", "")).strip()
    return raw if raw else "standard"


def _task11_effective_args(
    args: Any,
    *,
    continuation_checkpoint_path: str,
) -> Any:
    payload = dict(vars(args))
    payload["condition_focused_continuation"] = True
    payload["continuation_checkpoint_path"] = str(continuation_checkpoint_path)
    return SimpleNamespace(**payload)


def _task12_effective_args(
    args: Any,
    *,
    continuation_checkpoint_path: str,
) -> Any:
    payload = dict(vars(args))
    payload["condition_focused_continuation"] = False
    payload["continuation_checkpoint_path"] = str(continuation_checkpoint_path)
    return SimpleNamespace(**payload)


def _resolve_formal_lane_warm_start_checkpoint(
    args: Any,
    *,
    gate_summary: Mapping[str, Any],
) -> Path:
    return _load_smoke_module().resolve_task11_conditioned_warm_start_checkpoint(
        _repo_root(),
        gate_summary_payload=gate_summary,
        continuation_checkpoint_path=str(
            getattr(args, "continuation_checkpoint_path", "") or ""
        ),
    )


def _emit_task11_comparability_manifest(
    *,
    args: Any,
    output_dir: Path,
    gate_summary: Mapping[str, Any],
    train_scope_effective: str,
) -> tuple[Path, dict[str, Any]]:
    warm_start_checkpoint = _resolve_formal_lane_warm_start_checkpoint(
        args,
        gate_summary=gate_summary,
    )
    comparability_manifest = finetune_full.emit_conditioned_formal_lane_comparability_manifest(
        repo_root=_repo_root(),
        output_dir=output_dir,
        warm_start_checkpoint=warm_start_checkpoint,
        global_batch_size=int(getattr(args, "global_batch_size", 1)),
        gradient_accumulation_steps=int(
            getattr(args, "gradient_accumulation_steps", 1)
        ),
        num_gpus=int(getattr(args, "num_gpus", 1)),
        dataset_path=(str(getattr(args, "dataset_path", "")).strip() or None),
        train_scope_requested=str(getattr(args, "recap_train_scope", train_scope_effective)),
        train_scope_effective=train_scope_effective,
    )
    return warm_start_checkpoint, comparability_manifest


def _emit_task12_comparability_manifest(
    *,
    args: Any,
    output_dir: Path,
    gate_summary: Mapping[str, Any],
    train_scope_effective: str,
) -> tuple[Path, dict[str, Any]]:
    warm_start_checkpoint = _resolve_formal_lane_warm_start_checkpoint(
        args,
        gate_summary=gate_summary,
    )
    comparability_manifest = finetune_full.emit_continuation_formal_lane_comparability_manifest(
        repo_root=_repo_root(),
        output_dir=output_dir,
        warm_start_checkpoint=warm_start_checkpoint,
        global_batch_size=int(getattr(args, "global_batch_size", 1)),
        gradient_accumulation_steps=int(
            getattr(args, "gradient_accumulation_steps", 1)
        ),
        num_gpus=int(getattr(args, "num_gpus", 1)),
        dataset_path=(str(getattr(args, "dataset_path", "")).strip() or None),
        train_scope_requested=str(getattr(args, "recap_train_scope", train_scope_effective)),
        train_scope_effective=train_scope_effective,
    )
    return warm_start_checkpoint, comparability_manifest


def _write_formal_lane_skipped_manifest(
    *,
    output_dir: Path,
    requested_scope: str,
    best_scope_authority: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
    schema_version: str,
    artifact_kind: str,
    comparability_manifest_path: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": schema_version,
        "artifact_kind": artifact_kind,
        "status": "skipped",
        "output_dir": str(output_dir),
        "train_scope_requested": requested_scope,
        "train_scope_effective": str(best_scope_authority.get("train_scope_effective", "")),
        "best_scope_authority_path": best_scope_authority.get("path"),
        "gate_summary_path": gate_summary.get("path"),
        "p3_formal_training_eligible": False,
        "p3_skip_reason": str(
            gate_summary.get("p3_skip_reason") or "p3_formal_training_ineligible"
        ),
        "blocking_reasons": list(gate_summary.get("blocking_reasons", [])),
    }
    if comparability_manifest_path is not None:
        payload["comparability_manifest_path"] = comparability_manifest_path
    _write_json_atomic(output_dir / _TASK11_FORMAL_SKIP_FILENAME, payload)
    return payload


def _task11_write_skipped_manifest(
    *,
    output_dir: Path,
    requested_scope: str,
    best_scope_authority: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
    comparability_manifest_path: str | None = None,
) -> dict[str, Any]:
    return _write_formal_lane_skipped_manifest(
        output_dir=output_dir,
        requested_scope=requested_scope,
        best_scope_authority=best_scope_authority,
        gate_summary=gate_summary,
        schema_version="task11_formal_run_skipped_v1",
        artifact_kind="conditioned_formal_run_skipped",
        comparability_manifest_path=comparability_manifest_path,
    )


def _task12_write_skipped_manifest(
    *,
    output_dir: Path,
    requested_scope: str,
    best_scope_authority: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
    comparability_manifest_path: str | None = None,
) -> dict[str, Any]:
    return _write_formal_lane_skipped_manifest(
        output_dir=output_dir,
        requested_scope=requested_scope,
        best_scope_authority=best_scope_authority,
        gate_summary=gate_summary,
        schema_version="task12_formal_run_skipped_v1",
        artifact_kind="continuation_formal_run_skipped",
        comparability_manifest_path=comparability_manifest_path,
    )


def _task11_copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _task11_stage_formal_lane_artifacts(
    *,
    args: Any,
    output_dir: Path,
    best_scope_authority: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
    train_scope_effective: str,
) -> dict[str, Any]:
    conditioning_probe_path = Path(
        str(gate_summary.get("conditioning_probe_path", "")).strip()
    ).expanduser()
    if not conditioning_probe_path.is_file():
        raise FileNotFoundError(
            f"Task 11 conditioning probe missing: "
            f"{conditioning_probe_path or gate_summary.get('conditioning_probe_path')}"
        )
    staged_conditioning_probe = _task11_copy_file(
        conditioning_probe_path.resolve(),
        output_dir / _TASK11_FORMAL_CONDITIONING_PROBE_FILENAME,
    )
    warm_start_checkpoint, comparability_manifest = _emit_task11_comparability_manifest(
        args=args,
        output_dir=output_dir,
        gate_summary=gate_summary,
        train_scope_effective=train_scope_effective,
    )
    if str(comparability_manifest.get("status")) != "ok":
        raise RuntimeError(
            f"Task 11 comparability manifest blocked: "
            f"{comparability_manifest.get('blocker_code') or comparability_manifest.get('reason') or 'unknown'}"
        )
    return {
        "best_scope_authority_path": best_scope_authority.get("path"),
        "gate_summary_path": gate_summary.get("path"),
        "warm_start_checkpoint": str(warm_start_checkpoint),
        "comparability_manifest_path": str(
            output_dir / finetune_full.COMPARABILITY_MANIFEST_FILENAME
        ),
        "conditioning_functional_probe_path": str(staged_conditioning_probe),
    }


def _read_json_mapping_required(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive surface
        raise _ContinuationParityError(f"{label} unreadable: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise _ContinuationParityError(f"{label} must be a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def _task12_conditioned_peer_manifest_path() -> Path:
    return (
        _repo_root()
        / finetune_full.TASK11_CONDITIONED_FORMAL_RUN_REL
        / finetune_full.COMPARABILITY_MANIFEST_FILENAME
    ).resolve()


def _task12_conditioned_peer_skip_path() -> Path:
    return (
        _repo_root()
        / finetune_full.TASK11_CONDITIONED_FORMAL_RUN_REL
        / _TASK12_CONDITIONED_PEER_SKIP_FILENAME
    ).resolve()


def _task12_stage_formal_lane_artifacts(
    *,
    args: Any,
    output_dir: Path,
    best_scope_authority: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
    train_scope_effective: str,
) -> dict[str, Any]:
    warm_start_checkpoint, comparability_manifest = _emit_task12_comparability_manifest(
        args=args,
        output_dir=output_dir,
        gate_summary=gate_summary,
        train_scope_effective=train_scope_effective,
    )
    if str(comparability_manifest.get("status")) != "ok":
        raise _ContinuationParityError(
            "Task 12 continuation comparability manifest blocked: "
            f"{comparability_manifest.get('blocker_code') or comparability_manifest.get('reason') or 'unknown'}"
        )

    conditioned_manifest_path = _task12_conditioned_peer_manifest_path()
    if not conditioned_manifest_path.is_file():
        conditioned_skip_path = _task12_conditioned_peer_skip_path()
        if conditioned_skip_path.is_file():
            raise _ContinuationParityError(
                "Task 12 conditioned peer comparability manifest missing because the conditioned "
                f"lane was skipped: {conditioned_skip_path}"
            )
        raise _ContinuationParityError(
            "Task 12 conditioned peer comparability manifest missing: "
            f"{conditioned_manifest_path}"
        )
    conditioned_manifest = _read_json_mapping_required(
        conditioned_manifest_path,
        label="Task 12 conditioned peer comparability manifest",
    )
    parity_verdict = finetune_full.validate_full_update_comparability_manifests(
        conditioned_manifest,
        comparability_manifest,
    )
    if str(parity_verdict.get("status")) != "pass":
        raise _ContinuationParityError(
            json.dumps(parity_verdict, ensure_ascii=True, sort_keys=True)
        )
    return {
        "best_scope_authority_path": best_scope_authority.get("path"),
        "gate_summary_path": gate_summary.get("path"),
        "warm_start_checkpoint": str(warm_start_checkpoint),
        "comparability_manifest_path": str(
            output_dir / finetune_full.COMPARABILITY_MANIFEST_FILENAME
        ),
        "conditioned_peer_comparability_manifest_path": str(
            conditioned_manifest_path
        ),
        "continuation_parity_verdict": parity_verdict,
    }


def _run_continuation_single_scope(
    args: Any,
    *,
    requested_scope_override: str,
) -> tuple[int, dict[str, Any]]:
    control = importlib.import_module(
        "work.recap.scripts.30i_stage3_baseline_continuation_control"
    )
    output_dir = Path(str(getattr(args, "output_dir", ""))).expanduser()
    if not output_dir.is_absolute():
        output_dir = (_repo_root() / output_dir).resolve()
    repo_relative_output_dir: str | None = None
    try:
        repo_relative_output_dir = str(output_dir.relative_to(_repo_root()))
    except ValueError:
        repo_relative_output_dir = None
    contract = finetune_full.FROZEN_DIRECT_FINETUNE_OUTPUT_CONTRACTS.get(
        repo_relative_output_dir or "",
        {},
    )
    runtime_log_dir = contract.get("runtime_log_dir")
    resolved_runtime_log_dir = (
        (_repo_root() / str(runtime_log_dir)).resolve()
        if isinstance(runtime_log_dir, str) and runtime_log_dir.strip()
        else (output_dir / "runtime_logs").resolve()
    )
    summary_json = (
        trainability_entrypoint.resolve_repo_local_metadata_dir_for_output_dir(output_dir)
        / "continuation_control_summary.json"
    )
    argv = [
        "--dataset-path",
        str(getattr(args, "dataset_path")),
        "--continuation-checkpoint",
        str(getattr(args, "continuation_checkpoint_path")),
        "--output-dir",
        str(output_dir),
        "--runtime-log-dir",
        str(resolved_runtime_log_dir),
        "--summary-json",
        str(summary_json),
        "--max-steps",
        str(int(getattr(args, "max_steps", 1))),
        "--save-steps",
        str(int(getattr(args, "save_steps", 1))),
        "--save-total-limit",
        str(int(getattr(args, "save_total_limit", 1))),
        "--global-batch-size",
        str(int(getattr(args, "global_batch_size", 1))),
        "--gradient-accumulation-steps",
        str(int(getattr(args, "gradient_accumulation_steps", 1))),
        "--dataloader-num-workers",
        str(int(getattr(args, "dataloader_num_workers", 0))),
        "--learning-rate",
        str(float(getattr(args, "learning_rate", 1e-5))),
        "--recap-train-scope",
        requested_scope_override,
        "--num-gpus",
        str(int(getattr(args, "num_gpus", 1))),
        "--visible-device",
        "1",
        "--embodiment-tag",
        str(getattr(args, "embodiment_tag", "UNITREE_G1")),
        "--python",
        str(getattr(args, "python", "") or ""),
        "--tune-projector"
        if bool(getattr(args, "tune_projector", False))
        else "--no-tune-projector",
        "--tune-diffusion-model"
        if bool(getattr(args, "tune_diffusion_model", True))
        else "--no-tune-diffusion-model",
        "--use-wandb"
        if bool(getattr(args, "use_wandb", False))
        else "--no-use-wandb",
    ]
    rc = int(control.main(argv))
    if not summary_json.is_file():
        raise FileNotFoundError(
            f"Task 12 continuation control summary missing after launch: {summary_json}"
        )
    return rc, _read_json_mapping_required(
        summary_json,
        label="Task 12 continuation control summary",
    )


def _task11_copy_dynamic_audit_reference(output_dir: Path) -> Path:
    metadata_dir = trainability_entrypoint.resolve_repo_local_metadata_dir_for_output_dir(
        output_dir
    )
    source_path = metadata_dir / trainability_entrypoint.FULL_UPDATE_SCOPE_AUDIT_DYNAMIC_FILENAME
    if not source_path.is_file():
        raise FileNotFoundError(f"Task 11 dynamic scope audit missing: {source_path}")
    return _task11_copy_file(
        source_path,
        output_dir / trainability_entrypoint.FULL_UPDATE_SCOPE_AUDIT_DYNAMIC_FILENAME,
    )


def build_candidate_scope_chain(requested_scope: object) -> list[str]:
    normalized_scope = parse_scope_flag(requested_scope)
    if normalized_scope not in FULL_UPDATE_SCOPE_NAMES:
        raise ValueError(
            "full-update scope supervisor only supports strict_full/full_policy/full_action"
        )
    start_index = FULL_UPDATE_SCOPE_NAMES.index(normalized_scope)
    return list(FULL_UPDATE_SCOPE_NAMES[start_index:])


def _allow_downgrade(args: Any) -> bool:
    return bool(getattr(args, "allow_downgrade", True))


def _coerce_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _flatten_numeric_values(value: object) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Mapping):
        flattened: list[float] = []
        for nested in value.values():
            flattened.extend(_flatten_numeric_values(nested))
        return flattened
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        flattened = []
        for nested in value:
            flattened.extend(_flatten_numeric_values(nested))
        return flattened
    return []


def _all_zero_metric(value: object) -> bool:
    flattened = _flatten_numeric_values(value)
    return bool(flattened) and all(abs(item) == 0.0 for item in flattened)


def _scope_reason_token(scope_name: str) -> str:
    return str(scope_name).upper()


def _scope_metric_present_and_positive(summary: Mapping[str, Any], scope_name: str) -> bool:
    if scope_name not in summary:
        return False
    flattened = _flatten_numeric_values(summary.get(scope_name))
    return bool(flattened) and any(abs(item) > 0.0 for item in flattened)


def _required_scope_metric_block_reasons(
    *,
    payload: Mapping[str, Any],
    probe_key: str,
    summary_key: str,
    reason_prefix: str,
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(payload.get(probe_key), Mapping):
        reasons.append(f"MISSING_{reason_prefix}_PROBE")

    summary = payload.get(summary_key)
    if not isinstance(summary, Mapping):
        for scope_name in _REQUIRED_SUCCESS_EVIDENCE_SCOPES:
            reasons.append(
                f"MISSING_{reason_prefix}_{_scope_reason_token(scope_name)}"
            )
        return reasons

    if _all_zero_metric(summary):
        return reasons

    for scope_name in _REQUIRED_SUCCESS_EVIDENCE_SCOPES:
        if scope_name not in summary:
            reasons.append(
                f"MISSING_{reason_prefix}_{_scope_reason_token(scope_name)}"
            )
            continue
        if not _scope_metric_present_and_positive(summary, scope_name):
            reasons.append(
                f"NON_POSITIVE_{reason_prefix}_{_scope_reason_token(scope_name)}"
            )
    return reasons


def _read_log_text(path_value: object) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        return ""
    path = Path(path_value).expanduser()
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _payload_is_oom(payload: Mapping[str, Any]) -> bool:
    haystacks = [
        str(payload.get("error", "")),
        _read_log_text(payload.get("runtime_log_path")),
    ]
    lowered = "\n".join(haystacks).lower()
    return any(signature in lowered for signature in _OOM_SIGNATURES)


def _runtime_block_reasons(payload: Mapping[str, Any]) -> list[str]:
    block_reasons: list[str] = []
    if payload.get("selected_checkpoint_exists") is not True:
        block_reasons.append("MISSING_CHECKPOINT")
    if int(payload.get("trainer_global_step", 0) or 0) != 1:
        block_reasons.append("TRAINER_GLOBAL_STEP_NOT_ONE")
    block_reasons.extend(
        _required_scope_metric_block_reasons(
            payload=payload,
            probe_key="grad_probe_after_backward",
            summary_key="all_major_grad_norms",
            reason_prefix="GRAD_EVIDENCE",
        )
    )
    block_reasons.extend(
        _required_scope_metric_block_reasons(
            payload=payload,
            probe_key="param_delta_after_step",
            summary_key="all_major_param_delta",
            reason_prefix="PARAM_DELTA_EVIDENCE",
        )
    )
    if _all_zero_metric(payload.get("all_major_grad_norms")):
        block_reasons.append("ALL_ZERO_GRAD")
    if _all_zero_metric(payload.get("all_major_param_delta")):
        block_reasons.append("ALL_ZERO_PARAM_DELTA")
    return block_reasons


def _memory_estimator_block(audit: Mapping[str, Any]) -> bool:
    memory = audit.get("memory_feasibility")
    if not isinstance(memory, Mapping):
        return False
    return memory.get("fits_available_memory") is False


def _load_runtime_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    embedded = preflight.get("runtime_preflight")
    if isinstance(embedded, Mapping):
        return {str(key): value for key, value in embedded.items()}

    path_value = preflight.get("runtime_preflight_path")
    if not isinstance(path_value, str) or not path_value.strip():
        return {}
    path = Path(path_value).expanduser()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _coerce_mapping(payload)


def _runtime_preflight_block(runtime_preflight: Mapping[str, Any]) -> bool:
    return str(runtime_preflight.get("status", "")).strip() == "BLOCK"


def _runtime_preflight_memory_estimator_block(
    runtime_preflight: Mapping[str, Any],
) -> bool:
    if (
        str(runtime_preflight.get("strict_full_runtime_skipped_reason", "")).strip()
        == "memory_estimator_block"
    ):
        return True
    memory = runtime_preflight.get("memory_feasibility_estimate")
    if not isinstance(memory, Mapping):
        return False
    return str(memory.get("risk", "")).strip() == "BLOCK"


def _build_attempt_record(
    *,
    requested_scope: str,
    candidate_scope: str,
    static_audit: Mapping[str, Any],
    static_audit_path: str,
    metadata_dir: str,
    runtime_preflight: Mapping[str, Any] | None = None,
    runtime_preflight_path: str | None = None,
) -> dict[str, Any]:
    memory = _coerce_mapping(static_audit.get("memory_feasibility"))
    normalized_runtime_preflight = _coerce_mapping(runtime_preflight or {})
    return {
        "train_scope_requested": requested_scope,
        "candidate_scope": candidate_scope,
        "static_audit_path": static_audit_path,
        "metadata_dir": metadata_dir,
        "runtime_preflight_path": runtime_preflight_path,
        "runtime_preflight_status": normalized_runtime_preflight.get("status"),
        "runtime_preflight_block_reasons": list(
            normalized_runtime_preflight.get("hard_block_reasons", [])
        ),
        "strict_full_runtime_skipped_reason": normalized_runtime_preflight.get(
            "strict_full_runtime_skipped_reason"
        ),
        "static_verdict": str(static_audit.get("static_verdict", "PASS")),
        "static_block_reasons": list(static_audit.get("static_block_reasons", [])),
        "memory_feasibility": memory,
        "runtime_attempted": False,
        "status": None,
        "failure_reason": None,
        "runtime_returncode": None,
        "runtime_log_path": None,
        "wrapper_status": None,
        "delegate_cmd": None,
        "delegate_cmd_shell": None,
        "checkpoint_load_report_path": None,
        "selected_checkpoint_path": None,
        "selected_checkpoint_asset_path": None,
        "selected_checkpoint_exists": None,
        "trainer_global_step": None,
        "advantage_embedding_keys_present": None,
        "advantage_embedding_missing_keys": [],
        "all_major_grad_norms": None,
        "all_major_param_delta": None,
        "dynamic_block_reasons": [],
    }


def _record_attempt(output_dir: Path, attempt: Mapping[str, Any]) -> Path:
    return trainability_entrypoint.append_repo_local_runtime_metadata_jsonl(
        output_dir=output_dir,
        filename=trainability_entrypoint.FULL_UPDATE_DOWNGRADE_ATTEMPTS_FILENAME,
        payload=attempt,
    )


def _write_dynamic_audit(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    return trainability_entrypoint.write_repo_local_runtime_metadata_json(
        output_dir=output_dir,
        filename=trainability_entrypoint.FULL_UPDATE_SCOPE_AUDIT_DYNAMIC_FILENAME,
        payload=payload,
    )


def _emit_p0_block_artifacts(
    *,
    requested_scope: str,
    output_dir: Path,
    static_audit_path: str,
    static_audit: Mapping[str, Any],
) -> dict[str, str]:
    report_path = output_dir / "p0_scope_audit" / "p0_block_report.md"
    block_reasons = [str(reason) for reason in static_audit.get("static_block_reasons", [])]
    report_lines = [
        "# P0 Static Scope Audit Block",
        "",
        f"- requested_scope: `{requested_scope}`",
        f"- static_verdict: `{static_audit.get('static_verdict', 'BLOCK')}`",
        f"- static_audit_path: `{static_audit_path}`",
        f"- output_dir: `{output_dir}`",
        "- static_block_reasons:",
    ]
    report_lines.extend(f"  - `{reason}`" for reason in block_reasons)
    _write_text_atomic(report_path, "\n".join(report_lines) + "\n")

    evidence_path = _repo_root() / P0_BLOCK_EVIDENCE_REL
    evidence_lines = [
        "# Task P0 Block Evidence",
        "",
        f"- requested_scope: `{requested_scope}`",
        f"- static_audit_path: `{static_audit_path}`",
        f"- p0_block_report: `{report_path}`",
        f"- static_block_reasons: {json.dumps(block_reasons, ensure_ascii=True)}",
    ]
    _write_text_atomic(evidence_path, "\n".join(evidence_lines) + "\n")
    return {
        "p0_block_report_path": str(report_path),
        "p0_block_evidence_path": str(evidence_path),
    }


def _finalize_dynamic_audit(
    *,
    requested_scope: str,
    output_dir: Path,
    attempts: Sequence[Mapping[str, Any]],
    resolution_status: str,
    train_scope_effective: str | None,
    failure_reason: str | None,
    static_block_paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    strict_full_runtime_attempted = any(
        str(attempt.get("candidate_scope", "")) == "strict_full"
        and attempt.get("runtime_attempted") is True
        for attempt in attempts
    )
    payload: dict[str, Any] = {
        "schema_version": DYNAMIC_AUDIT_SCHEMA_VERSION,
        "artifact_kind": DYNAMIC_AUDIT_ARTIFACT_KIND,
        "audit_phase": "dynamic",
        "train_scope_requested": requested_scope,
        "resolution_status": resolution_status,
        "runtime_supervisor_used": True,
        "strict_full_runtime_attempted": strict_full_runtime_attempted,
        "best_scope_authority": (
            resolution_status in {"PASS", "DEGRADE"}
            and train_scope_effective is not None
        ),
        "degrade_chain_attempted": [dict(attempt) for attempt in attempts],
        "attempt_count": len(attempts),
        "output_dir": str(output_dir),
        "downgrade_attempts_path": str(
            trainability_entrypoint.resolve_repo_local_metadata_dir_for_output_dir(output_dir)
            / trainability_entrypoint.FULL_UPDATE_DOWNGRADE_ATTEMPTS_FILENAME
        ),
        "failure_reason": failure_reason,
    }
    if train_scope_effective is not None:
        payload["train_scope_effective"] = train_scope_effective
    if static_block_paths:
        payload.update({str(key): value for key, value in static_block_paths.items()})
    return payload


def run_numeric_adv_scope_supervisor(args: Any) -> int:
    requested_scope = parse_scope_flag(getattr(args, "recap_train_scope", ""))
    candidate_chain = build_candidate_scope_chain(requested_scope)
    if not _allow_downgrade(args):
        candidate_chain = candidate_chain[:1]
    smoke = _load_smoke_module()
    formal_entrypoint_name = _formal_entrypoint_name(args)
    output_dir = Path(
        smoke._resolve_full_update_output_dir(_repo_root(), str(getattr(args, "output_dir")))
    )
    attempts: list[dict[str, Any]] = []
    formal_gate_enabled = _task11_formal_gate_enabled(args)
    best_scope_authority: dict[str, Any] | None = None
    gate_summary: dict[str, Any] | None = None
    effective_args = args

    if formal_gate_enabled:
        best_scope_authority = smoke.load_task11_best_scope_authority(
            _repo_root(),
            best_scope_audit=str(getattr(args, "best_scope_audit", "") or ""),
        )
        if best_scope_authority is None:
            raise RuntimeError("Task 11 best scope authority failed to load")
        authoritative_best_scope = parse_scope_flag(
            best_scope_authority["train_scope_effective"]
        )
        if authoritative_best_scope != requested_scope:
            dynamic_payload = _finalize_dynamic_audit(
                requested_scope=requested_scope,
                output_dir=output_dir,
                attempts=attempts,
                resolution_status="BLOCK",
                train_scope_effective=None,
                failure_reason=_TASK11_SCOPE_DRIFT_FAILURE_REASON,
            )
            dynamic_payload.update(
                {
                    "best_scope_authority_path": best_scope_authority.get("path"),
                    "best_scope_authority_requested": authoritative_best_scope,
                }
            )
            _write_dynamic_audit(output_dir, dynamic_payload)
            _write_optional_summary_json(args, dynamic_payload)
            return 1

        gate_summary = smoke.load_task11_preformal_gate_summary(
            _repo_root(),
            gate_summary=str(getattr(args, "gate_summary", "") or ""),
        )
        if gate_summary is None:
            raise RuntimeError("Task 11 preformal gate summary failed to load")
        if gate_summary.get("p3_formal_training_eligible") is not True:
            if _task12_formal_entrypoint(args):
                _, comparability_manifest = _emit_task12_comparability_manifest(
                    args=effective_args,
                    output_dir=output_dir,
                    gate_summary=gate_summary,
                    train_scope_effective=requested_scope,
                )
                skipped_payload = _task12_write_skipped_manifest(
                    output_dir=output_dir,
                    requested_scope=requested_scope,
                    best_scope_authority=best_scope_authority,
                    gate_summary=gate_summary,
                    comparability_manifest_path=str(
                        output_dir / finetune_full.COMPARABILITY_MANIFEST_FILENAME
                    ),
                )
            else:
                _, comparability_manifest = _emit_task11_comparability_manifest(
                    args=effective_args,
                    output_dir=output_dir,
                    gate_summary=gate_summary,
                    train_scope_effective=requested_scope,
                )
                skipped_payload = _task11_write_skipped_manifest(
                    output_dir=output_dir,
                    requested_scope=requested_scope,
                    best_scope_authority=best_scope_authority,
                    gate_summary=gate_summary,
                    comparability_manifest_path=str(
                        output_dir / finetune_full.COMPARABILITY_MANIFEST_FILENAME
                    ),
                )
            _write_optional_summary_json(args, skipped_payload)
            return 0

        if _task11_formal_entrypoint(args):
            warm_start_checkpoint = smoke.resolve_task11_conditioned_warm_start_checkpoint(
                _repo_root(),
                gate_summary_payload=gate_summary,
                continuation_checkpoint_path=str(
                    getattr(args, "continuation_checkpoint_path", "") or ""
                ),
            )
            effective_args = _task11_effective_args(
                args,
                continuation_checkpoint_path=str(warm_start_checkpoint),
            )
        elif _task12_formal_entrypoint(args):
            warm_start_checkpoint = smoke.resolve_task11_conditioned_warm_start_checkpoint(
                _repo_root(),
                gate_summary_payload=gate_summary,
                continuation_checkpoint_path=str(
                    getattr(args, "continuation_checkpoint_path", "") or ""
                ),
            )
            effective_args = _task12_effective_args(
                args,
                continuation_checkpoint_path=str(warm_start_checkpoint),
            )

    for candidate_scope in candidate_chain:
        preflight = smoke.run_numeric_adv_static_scope_audit(
            effective_args,
            requested_scope_override=candidate_scope,
        )
        static_audit = _coerce_mapping(preflight.get("static_audit"))
        runtime_preflight = _load_runtime_preflight(preflight)
        attempt = _build_attempt_record(
            requested_scope=requested_scope,
            candidate_scope=candidate_scope,
            static_audit=static_audit,
            static_audit_path=str(preflight.get("static_audit_path", "")),
            metadata_dir=str(preflight.get("metadata_dir", "")),
            runtime_preflight=runtime_preflight,
            runtime_preflight_path=str(preflight.get("runtime_preflight_path", "") or ""),
        )

        if attempt["static_verdict"] == "BLOCK":
            attempt["status"] = "BLOCK"
            attempt["failure_reason"] = "STATIC_AUDIT_BLOCK"
            attempts.append(attempt)
            _record_attempt(output_dir, attempt)
            static_block_paths = _emit_p0_block_artifacts(
                requested_scope=requested_scope,
                output_dir=output_dir,
                static_audit_path=str(preflight.get("static_audit_path", "")),
                static_audit=static_audit,
            )
            dynamic_payload = _finalize_dynamic_audit(
                requested_scope=requested_scope,
                output_dir=output_dir,
                attempts=attempts,
                resolution_status="BLOCK",
                train_scope_effective=None,
                failure_reason="STATIC_AUDIT_BLOCK",
                static_block_paths=static_block_paths,
            )
            _write_dynamic_audit(output_dir, dynamic_payload)
            _write_optional_summary_json(args, dynamic_payload)
            return 1

        if _runtime_preflight_block(runtime_preflight):
            attempt["status"] = "BLOCK"
            attempt["failure_reason"] = "RUNTIME_PREFLIGHT_BLOCK"
            attempts.append(attempt)
            _record_attempt(output_dir, attempt)
            dynamic_payload = _finalize_dynamic_audit(
                requested_scope=requested_scope,
                output_dir=output_dir,
                attempts=attempts,
                resolution_status="BLOCK",
                train_scope_effective=None,
                failure_reason="RUNTIME_PREFLIGHT_BLOCK",
            )
            _write_dynamic_audit(output_dir, dynamic_payload)
            _write_optional_summary_json(args, dynamic_payload)
            return 1

        if _runtime_preflight_memory_estimator_block(runtime_preflight) or _memory_estimator_block(static_audit):
            attempt["status"] = "MEMORY_ESTIMATOR_BLOCK"
            attempt["failure_reason"] = "MEMORY_ESTIMATOR_BLOCK"
            attempts.append(attempt)
            _record_attempt(output_dir, attempt)
            continue

        if _task12_formal_entrypoint(args):
            rc, runtime_payload = _run_continuation_single_scope(
                effective_args,
                requested_scope_override=candidate_scope,
            )
        else:
            rc, runtime_payload = smoke.run_numeric_adv_single_scope(
                effective_args,
                requested_scope_override=candidate_scope,
                summary_json_override=None,
                emit_summary=False,
            )
        attempt["runtime_attempted"] = True
        attempt["runtime_returncode"] = int(rc)
        attempt["runtime_log_path"] = runtime_payload.get("runtime_log_path")
        attempt["wrapper_status"] = runtime_payload.get("wrapper_status")
        attempt["delegate_cmd"] = runtime_payload.get("delegate_cmd")
        attempt["delegate_cmd_shell"] = runtime_payload.get("delegate_cmd_shell")
        attempt["checkpoint_load_report_path"] = runtime_payload.get(
            "checkpoint_load_report_path"
        )
        attempt["selected_checkpoint_path"] = runtime_payload.get("selected_checkpoint_path")
        attempt["selected_checkpoint_asset_path"] = runtime_payload.get(
            "selected_checkpoint_asset_path"
        )
        attempt["selected_checkpoint_exists"] = runtime_payload.get(
            "selected_checkpoint_exists"
        )
        attempt["trainer_global_step"] = runtime_payload.get("trainer_global_step")
        attempt["advantage_embedding_keys_present"] = runtime_payload.get(
            "advantage_embedding_keys_present"
        )
        attempt["advantage_embedding_missing_keys"] = list(
            runtime_payload.get("advantage_embedding_missing_keys", [])
        )
        attempt["all_major_grad_norms"] = runtime_payload.get("all_major_grad_norms")
        attempt["all_major_param_delta"] = runtime_payload.get("all_major_param_delta")
        attempt["dynamic_block_reasons"] = _runtime_block_reasons(runtime_payload)

        if attempt["dynamic_block_reasons"]:
            attempt["status"] = "BLOCK"
            attempt["failure_reason"] = "DYNAMIC_RUNTIME_BLOCK"
            attempts.append(attempt)
            _record_attempt(output_dir, attempt)
            dynamic_payload = _finalize_dynamic_audit(
                requested_scope=requested_scope,
                output_dir=output_dir,
                attempts=attempts,
                resolution_status="BLOCK",
                train_scope_effective=None,
                failure_reason="DYNAMIC_RUNTIME_BLOCK",
            )
            _write_dynamic_audit(output_dir, dynamic_payload)
            _write_optional_summary_json(args, dynamic_payload)
            return 1

        if int(rc) == 0 and str(runtime_payload.get("wrapper_status", "")) == "ok":
            staged_artifacts: dict[str, Any] = {}
            if formal_gate_enabled and best_scope_authority is not None and gate_summary is not None:
                try:
                    if _task12_formal_entrypoint(args):
                        staged_artifacts = _task12_stage_formal_lane_artifacts(
                            args=effective_args,
                            output_dir=output_dir,
                            best_scope_authority=best_scope_authority,
                            gate_summary=gate_summary,
                            train_scope_effective=candidate_scope,
                        )
                    else:
                        staged_artifacts = _task11_stage_formal_lane_artifacts(
                            args=effective_args,
                            output_dir=output_dir,
                            best_scope_authority=best_scope_authority,
                            gate_summary=gate_summary,
                            train_scope_effective=candidate_scope,
                        )
                except _ContinuationParityError as exc:
                    attempt["status"] = "BLOCK"
                    attempt["failure_reason"] = _TASK12_PARITY_BLOCK_FAILURE_REASON
                    attempt["dynamic_block_reasons"] = [str(exc)]
                    attempts.append(attempt)
                    _record_attempt(output_dir, attempt)
                    dynamic_payload = _finalize_dynamic_audit(
                        requested_scope=requested_scope,
                        output_dir=output_dir,
                        attempts=attempts,
                        resolution_status="BLOCK",
                        train_scope_effective=None,
                        failure_reason=_TASK12_PARITY_BLOCK_FAILURE_REASON,
                    )
                    dynamic_payload.update(
                        {
                            "best_scope_authority_path": best_scope_authority.get("path"),
                            "gate_summary_path": gate_summary.get("path"),
                            "formal_entrypoint": formal_entrypoint_name,
                            "artifact_staging_error": str(exc),
                        }
                    )
                    _write_dynamic_audit(output_dir, dynamic_payload)
                    _write_optional_summary_json(args, dynamic_payload)
                    return 1
                except Exception as exc:
                    attempt["status"] = "BLOCK"
                    attempt["failure_reason"] = _TASK11_ARTIFACT_STAGING_FAILURE_REASON
                    attempt["dynamic_block_reasons"] = [str(exc)]
                    attempts.append(attempt)
                    _record_attempt(output_dir, attempt)
                    dynamic_payload = _finalize_dynamic_audit(
                        requested_scope=requested_scope,
                        output_dir=output_dir,
                        attempts=attempts,
                        resolution_status="BLOCK",
                        train_scope_effective=None,
                        failure_reason=_TASK11_ARTIFACT_STAGING_FAILURE_REASON,
                    )
                    dynamic_payload.update(
                        {
                            "best_scope_authority_path": best_scope_authority.get("path"),
                            "gate_summary_path": gate_summary.get("path"),
                            "formal_entrypoint": formal_entrypoint_name,
                            "artifact_staging_error": str(exc),
                        }
                    )
                    _write_dynamic_audit(output_dir, dynamic_payload)
                    _write_optional_summary_json(args, dynamic_payload)
                    return 1
            attempt["status"] = "PASS"
            attempts.append(attempt)
            _record_attempt(output_dir, attempt)
            resolution_status = "PASS" if candidate_scope == requested_scope else "DEGRADE"
            dynamic_payload = _finalize_dynamic_audit(
                requested_scope=requested_scope,
                output_dir=output_dir,
                attempts=attempts,
                resolution_status=resolution_status,
                train_scope_effective=candidate_scope,
                failure_reason=None,
            )
            if staged_artifacts:
                dynamic_payload.update(staged_artifacts)
            _write_dynamic_audit(output_dir, dynamic_payload)
            if staged_artifacts:
                dynamic_payload["full_update_scope_audit_dynamic_path"] = str(
                    _task11_copy_dynamic_audit_reference(output_dir)
                )
            _write_optional_summary_json(args, dynamic_payload)
            return 0

        if _payload_is_oom(runtime_payload):
            attempt["status"] = "OOM"
            attempt["failure_reason"] = "CUDA_OOM"
            attempts.append(attempt)
            _record_attempt(output_dir, attempt)
            continue

        attempt["status"] = "BLOCK"
        attempt["failure_reason"] = str(runtime_payload.get("error") or "RUNTIME_BLOCK")
        attempts.append(attempt)
        _record_attempt(output_dir, attempt)
        dynamic_payload = _finalize_dynamic_audit(
            requested_scope=requested_scope,
            output_dir=output_dir,
            attempts=attempts,
            resolution_status="BLOCK",
            train_scope_effective=None,
            failure_reason=str(runtime_payload.get("error") or "RUNTIME_BLOCK"),
        )
        _write_dynamic_audit(output_dir, dynamic_payload)
        _write_optional_summary_json(args, dynamic_payload)
        return 1

    dynamic_payload = _finalize_dynamic_audit(
        requested_scope=requested_scope,
        output_dir=output_dir,
        attempts=attempts,
        resolution_status="BLOCK",
        train_scope_effective=None,
        failure_reason="EXHAUSTED_DOWNGRADE_CHAIN",
    )
    _write_dynamic_audit(output_dir, dynamic_payload)
    _write_optional_summary_json(args, dynamic_payload)
    return 1


def main() -> int:
    smoke = _load_smoke_module()
    parser = smoke._build_parser()
    parser.prog = "34c_full_update_scope_supervisor.py"
    args = parser.parse_args()
    if str(getattr(args, "entrypoint", "")).strip() not in {"", "conditioned", "continuation"}:
        parser.error(
            "34c_full_update_scope_supervisor.py only supports --entrypoint conditioned|continuation"
        )
    if str(getattr(args, "recap_train_scope", "")) not in FULL_UPDATE_SCOPE_NAMES:
        parser.error(
            "34c_full_update_scope_supervisor.py only supports --recap-train-scope "
            "strict_full|full_policy|full_action"
        )
    return run_numeric_adv_scope_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
