#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Mapping
import functools
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import types
from io import StringIO
from pathlib import Path
from typing import Any
import csv


REPO_ROOT = Path(__file__).resolve().parents[2]
ISAAC_GR00T_ROOT = REPO_ROOT / "submodules" / "Isaac-GR00T"
REPO_LOCAL_METADATA_DIRNAME = "repo_local_metadata"
RUNTIME_PREFLIGHT_FILENAME = "runtime_preflight.json"
FULL_UPDATE_SCOPE_AUDIT_DYNAMIC_FILENAME = "full_update_scope_audit_dynamic.json"
FULL_UPDATE_DOWNGRADE_ATTEMPTS_FILENAME = "downgrade_attempts.jsonl"
FIRST_BACKWARD_GRAD_PROBE_FILENAME = "first_backward_grad_probe_rank0.json"
FIRST_OPTIMIZER_STEP_PARAM_DELTA_FILENAME = (
    "first_optimizer_step_param_delta_rank0.json"
)
NORMALIZATION_STATS_HASH_JSONL_FILENAME = (
    "normalization_stats_hash_per_batch_rank0.jsonl"
)
NORMALIZATION_STATS_HASH_SUMMARY_FILENAME = "normalization_stats_hash_summary_rank0.json"
REPO_LOCAL_CENSUS_PREVIEW_LIMIT = 32
REPO_LOCAL_TRAINABILITY_AUTHORITY_FIELD = "repo_local_trainability_authority"
REPO_LOCAL_TRAINABILITY_AUTHORITY_SCHEMA_VERSION = (
    "repo_local_trainability_authority_v1"
)
REPO_LOCAL_ROUTE_FREEZE_SCHEMA_VERSION = "repo_local_route_freeze_v1"
REPO_LOCAL_TRAINABILITY_AUTHORITY_OWNER = "training_entrypoint"
REPO_LOCAL_CONDITION_HOT_PARAM_PREFIXES = (
    "action_head.advantage_embedding.",
    "action_head.vlln.",
)
REPO_LOCAL_DIFFUSION_TRUNK_PARAM_PREFIX = "action_head.model."
REPO_LOCAL_FULL_ACTION_PARAM_PREFIXES = (
    "action_head.",
)
REPO_LOCAL_FULL_POLICY_PARAM_PREFIXES = (
    "action_head.",
    "projector.",
    "action_head.projector.",
    "vla_action_interface.",
)
REPO_LOCAL_SUCCESS_PROBE_SCOPE_PREFIXES: dict[str, str] = {
    "diffusion_trunk": REPO_LOCAL_DIFFUSION_TRUNK_PARAM_PREFIX,
    "advantage_embedding": "action_head.advantage_embedding.",
}
DEFAULT_PATCH_B1_TRIGGER_CANDIDATE_PATH = (
    REPO_ROOT
    / "agent/artifacts/stage3_ddp_smoke/run_c_gpu12_attempt01/green_smoke_candidate.json"
)
DEFAULT_PATCH_B2_TRIGGER_CANDIDATE_PATH = DEFAULT_PATCH_B1_TRIGGER_CANDIDATE_PATH
REPO_LOCAL_CENSUS_HOOK_STATE: dict[str, Any] = {
    "metadata_dir": None,
    "torch_module": None,
    "pre_forward_written": False,
    "first_backward_written": False,
    "first_optimizer_step_written": False,
    "first_optimizer_step_pending_snapshot": None,
    "pipeline_setup_patched": False,
    "trainer_compute_loss_patched": False,
    "trainer_training_step_patched": False,
    "trainer_create_optimizer_patched": False,
    "trainer_train_patched": False,
    "original_pipeline_setup": None,
    "original_trainer_compute_loss": None,
    "original_trainer_training_step": None,
    "original_trainer_create_optimizer": None,
    "original_trainer_train": None,
    "stats_hash_batch_index": 0,
    "stats_hash_last_payload": None,
    "stats_hash_jsonl_path": None,
}
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.finetune_full import resolve_full_update_authority_output_dir
from work.recap.train_scope_audit import STATIC_SCOPE_AUDIT_FILENAME
from work.recap.train_scope_audit import compute_static_scope_audit
from work.recap.train_scope_audit import estimate_memory_feasibility
from work.recap.train_scope_audit import emit_scope_audit_json
from work.recap.train_scope_audit import normalize_train_scope_payload
from work.recap.train_scope_audit import parse_scope_flag


def load_modality_config(modality_config_path: str) -> None:
    import importlib

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
        return
    raise FileNotFoundError(
        f"Modality config path does not exist: {modality_config_path}"
    )


def resolve_repo_local_use_ddp(num_gpus: int | None) -> bool:
    if num_gpus is None:
        return False
    return int(num_gpus) > 1


def _env_int(name: str, default: int = 0) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _compute_effective_output_dir(config: Any) -> Path:
    experiment_name = getattr(config.training, "experiment_name", None)
    output_dir = resolve_full_update_authority_output_dir(
        REPO_ROOT,
        str(config.training.output_dir),
        require_v2_authority=False,
    )
    if experiment_name is None:
        return output_dir
    return output_dir / str(experiment_name)


def _repo_local_metadata_dir(config: Any) -> Path:
    return _compute_effective_output_dir(config) / REPO_LOCAL_METADATA_DIRNAME


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
    return path


def _append_jsonl_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=True, sort_keys=True) + "\n")
    return path


def _json_safe_for_hash(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_for_hash(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_for_hash(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe_for_hash(tolist())
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe_for_hash(item())
        except Exception:
            pass

    return repr(value)


def _stable_stats_sha256(payload: Any) -> str | None:
    if payload is None:
        return None
    canonical = json.dumps(
        _json_safe_for_hash(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def resolve_repo_local_metadata_dir_for_output_dir(output_dir: str | Path) -> Path:
    resolved_output_dir = resolve_full_update_authority_output_dir(
        REPO_ROOT,
        str(output_dir),
        require_v2_authority=False,
    )
    return resolved_output_dir / REPO_LOCAL_METADATA_DIRNAME


def resolve_repo_local_runtime_preflight_path_for_output_dir(
    output_dir: str | Path,
) -> Path:
    resolved_output_dir = resolve_full_update_authority_output_dir(
        REPO_ROOT,
        str(output_dir),
        require_v2_authority=False,
    )
    return resolved_output_dir / "p0_scope_audit" / RUNTIME_PREFLIGHT_FILENAME


def write_repo_local_runtime_metadata_json(
    *,
    output_dir: str | Path,
    filename: str,
    payload: Mapping[str, Any],
) -> Path:
    metadata_dir = resolve_repo_local_metadata_dir_for_output_dir(output_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return _write_json_atomic(metadata_dir / filename, dict(payload))


def append_repo_local_runtime_metadata_jsonl(
    *,
    output_dir: str | Path,
    filename: str,
    payload: Mapping[str, Any],
) -> Path:
    metadata_dir = resolve_repo_local_metadata_dir_for_output_dir(output_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return _append_jsonl_atomic(metadata_dir / filename, payload)


def write_repo_local_runtime_preflight_json(
    *,
    output_dir: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    preflight_path = resolve_repo_local_runtime_preflight_path_for_output_dir(output_dir)
    return _write_json_atomic(preflight_path, dict(payload))


def _coerce_optional_int(raw: object) -> int | None:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned == "":
            return None
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def _coerce_optional_float(raw: object) -> float | None:
    if isinstance(raw, bool):
        return float(int(raw))
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _expected_cuda_visible_devices(*, requested_num_gpus: int) -> str:
    if int(requested_num_gpus) <= 1:
        return "1"
    return "1,2"


def probe_repo_local_transformers_flash_attn() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "transformers_flash_attn_import_ok": False,
        "flash_attn_2_available": False,
    }
    try:
        from transformers.utils import is_flash_attn_2_available
    except Exception as exc:
        payload["transformers_flash_attn_import_error"] = _safe_exception_reason(exc)
        return payload

    payload["transformers_flash_attn_import_ok"] = True
    try:
        payload["flash_attn_2_available"] = bool(is_flash_attn_2_available())
    except Exception as exc:
        payload["flash_attn_2_available"] = False
        payload["flash_attn_2_available_error"] = _safe_exception_reason(exc)
    return payload


def probe_repo_local_nvidia_smi_gpu_snapshot(*, gpu_index: int = 1) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    result = _safe_subprocess_capture(command)
    payload: dict[str, Any] = {
        "ok": bool(result.get("ok")),
        "gpu_index": int(gpu_index),
        "query": result,
        "row": None,
        "available_memory_bytes": None,
    }
    if not bool(result.get("ok")):
        return payload

    rows = _parse_csv_noheader_rows(
        str(result.get("stdout", "")),
        [
            "index",
            "uuid",
            "name",
            "memory_total_mb",
            "memory_used_mb",
            "utilization_gpu_percent",
            "utilization_memory_percent",
        ],
    )
    if not rows:
        return payload

    row = dict(rows[0])
    total_mb = _coerce_optional_int(row.get("memory_total_mb"))
    used_mb = _coerce_optional_int(row.get("memory_used_mb"))
    available_bytes = None
    if total_mb is not None and used_mb is not None:
        available_bytes = max(total_mb - used_mb, 0) * 1024 * 1024
    payload["row"] = row
    payload["available_memory_bytes"] = available_bytes
    return payload


def _classify_repo_local_memory_risk(memory_estimate: Mapping[str, Any]) -> str:
    if memory_estimate.get("fits_available_memory") is False:
        return "BLOCK"

    estimated_total_bytes = _coerce_optional_int(memory_estimate.get("estimated_total_bytes"))
    available_memory_bytes = _coerce_optional_int(
        memory_estimate.get("available_memory_bytes")
    )
    if (
        estimated_total_bytes is None
        or available_memory_bytes is None
        or available_memory_bytes <= 0
    ):
        return "MEDIUM"

    usage_ratio = float(estimated_total_bytes) / float(available_memory_bytes)
    if usage_ratio >= 0.75:
        return "HIGH"
    if usage_ratio >= 0.5:
        return "MEDIUM"
    return "LOW"


def build_repo_local_runtime_preflight_payload(
    *,
    output_dir: str | Path,
    python_path: str | Path,
    requested_num_gpus: int,
    requested_scope: str,
    static_audit: Mapping[str, Any],
    torch_module: Any | None = None,
) -> dict[str, Any]:
    resolved_python_path = Path(str(python_path))
    if torch_module is None:
        import torch as imported_torch

        torch_module = imported_torch

    try:
        torch_cuda_arch_list = [
            str(arch) for arch in list(torch_module.cuda.get_arch_list())
        ]
    except Exception:
        torch_cuda_arch_list = []

    flash_attn_probe = probe_repo_local_transformers_flash_attn()
    gpu_snapshot = probe_repo_local_nvidia_smi_gpu_snapshot(gpu_index=1)
    memory_estimate = estimate_memory_feasibility(
        audit=static_audit,
        available_memory_bytes=_coerce_optional_int(
            gpu_snapshot.get("available_memory_bytes")
        ),
    )
    memory_estimate["risk"] = _classify_repo_local_memory_risk(memory_estimate)

    hard_block_reasons: list[str] = []
    if not resolved_python_path.exists():
        hard_block_reasons.append("venv_symlink_invalid")
    if not bool(flash_attn_probe.get("transformers_flash_attn_import_ok")):
        hard_block_reasons.append("transformers_flash_attn_import_failed")
    if not bool(flash_attn_probe.get("flash_attn_2_available")):
        hard_block_reasons.append("flash_attn_2_unavailable")

    payload: dict[str, Any] = {
        "output_dir": str(
            resolve_full_update_authority_output_dir(
                REPO_ROOT,
                str(output_dir),
                require_v2_authority=False,
            )
        ),
        "python_path": str(resolved_python_path),
        "venv_symlink_valid": bool(resolved_python_path.exists()),
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "torch_cuda_arch_list": torch_cuda_arch_list,
        "flash_attn_2_available": bool(flash_attn_probe.get("flash_attn_2_available")),
        "transformers_flash_attn_import_ok": bool(
            flash_attn_probe.get("transformers_flash_attn_import_ok")
        ),
        "nvidia_smi_gpu1_snapshot": gpu_snapshot,
        "cuda_visible_devices_expected": _expected_cuda_visible_devices(
            requested_num_gpus=int(requested_num_gpus)
        ),
        "memory_feasibility_estimate": memory_estimate,
        "status": "BLOCK" if hard_block_reasons else "PASS",
    }
    if hard_block_reasons:
        payload["hard_block_reasons"] = hard_block_reasons
    if str(requested_scope) == "strict_full" and memory_estimate["risk"] == "BLOCK":
        payload["strict_full_runtime_skipped_reason"] = "memory_estimator_block"
    return payload


def _copy_mapping(value: Mapping[str, Any] | object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _normalize_scope_summary(
    *,
    requested_scope: object | None,
    scope_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    normalized_requested_scope = (
        None if requested_scope is None else parse_scope_flag(requested_scope)
    )
    normalized_scope_summary = None
    if scope_summary is not None:
        normalized_scope_summary = normalize_train_scope_payload(scope_summary)
    elif normalized_requested_scope is not None:
        normalized_scope_summary = normalize_train_scope_payload(normalized_requested_scope)

    if (
        normalized_requested_scope is not None
        and normalized_scope_summary is not None
        and str(normalized_scope_summary.get("train_scope_requested", ""))
        != normalized_requested_scope
    ):
        raise ValueError(
            "conflicting_scope_override: requested_scope "
            + f"{normalized_requested_scope!r} conflicts with scope_summary "
            + f"{normalized_scope_summary.get('train_scope_requested')!r}"
        )
    return normalized_scope_summary


def _scope_requires_advantage_embedding(scope_summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(scope_summary, Mapping):
        return False
    required_families = scope_summary.get("required_trainable_families", [])
    if not isinstance(required_families, list):
        return False
    families = [str(family) for family in required_families]
    return any(
        "advantage_embedding" in family
        or family == "action_head.*"
        or family == "model.named_parameters()"
        for family in families
    )


def build_repo_local_route_freeze(
    *,
    route: object,
    indicator_mode: object | None = None,
) -> dict[str, Any]:
    from work.recap import policy as recap_policy

    policy_spec = recap_policy.build_runtime_policy_spec(
        route=route,
        indicator_mode=indicator_mode,
    )
    return {
        "schema_version": REPO_LOCAL_ROUTE_FREEZE_SCHEMA_VERSION,
        "frozen": True,
        "authority_owner": REPO_LOCAL_TRAINABILITY_AUTHORITY_OWNER,
        "route": str(policy_spec["route"]),
        "carrier_route": policy_spec.get("carrier_route"),
        "carrier_schema_version": policy_spec.get("carrier_schema_version"),
        "indicator_mode": policy_spec.get("indicator_mode"),
        "policy_class_name": str(policy_spec["policy_class_name"]),
        "mainline_authority": bool(policy_spec["mainline_authority"]),
        "diagnostic_only": bool(policy_spec["diagnostic_only"]),
        "runtime_indicator_mode_required": bool(
            policy_spec["runtime_indicator_mode_required"]
        ),
        "runtime_supported_indicator_modes": list(
            policy_spec["runtime_supported_indicator_modes"]
        ),
    }


def build_repo_local_trainability_authority(
    *,
    requested_scope: object | None,
    scope_summary: Mapping[str, Any] | None = None,
    condition_focused_continuation: bool,
    condition_hot_lr_scale: float,
    diffusion_trunk_lr_scale: float,
    route: object | None = None,
    indicator_mode: object | None = None,
) -> dict[str, Any]:
    normalized_scope_summary = _normalize_scope_summary(
        requested_scope=requested_scope,
        scope_summary=scope_summary,
    )
    route_freeze = (
        None
        if route is None
        else build_repo_local_route_freeze(route=route, indicator_mode=indicator_mode)
    )
    return {
        "schema_version": REPO_LOCAL_TRAINABILITY_AUTHORITY_SCHEMA_VERSION,
        "authority_owner": REPO_LOCAL_TRAINABILITY_AUTHORITY_OWNER,
        "requested_scope": (
            None
            if normalized_scope_summary is None
            else str(normalized_scope_summary["train_scope_requested"])
        ),
        "scope_summary": (
            None if normalized_scope_summary is None else dict(normalized_scope_summary)
        ),
        "advantage_embedding_required_by_scope": _scope_requires_advantage_embedding(
            normalized_scope_summary
        ),
        "condition_focused_continuation": bool(condition_focused_continuation),
        "condition_hot_lr_scale": float(condition_hot_lr_scale),
        "diffusion_trunk_lr_scale": float(diffusion_trunk_lr_scale),
        "diffusion_trunk_mode": (
            "frozen" if float(diffusion_trunk_lr_scale) <= 0.0 else "downscaled"
        ),
        "condition_hot_param_prefixes": list(REPO_LOCAL_CONDITION_HOT_PARAM_PREFIXES),
        "diffusion_trunk_param_prefix": REPO_LOCAL_DIFFUSION_TRUNK_PARAM_PREFIX,
        "route_freeze": route_freeze,
    }


def attach_repo_local_trainability_authority(*, config: Any, authority: Mapping[str, Any]) -> None:
    training = getattr(config, "training", None)
    if training is None:
        raise AttributeError("config.training is required for repo-local trainability authority")
    setattr(training, REPO_LOCAL_TRAINABILITY_AUTHORITY_FIELD, dict(authority))


def get_repo_local_trainability_authority(*, config: Any) -> dict[str, Any] | None:
    training = getattr(config, "training", None)
    if training is None:
        return None
    return _copy_mapping(getattr(training, REPO_LOCAL_TRAINABILITY_AUTHORITY_FIELD, None))


def _named_parameter_stats(
    named_params: list[tuple[str, Any]],
    *,
    sample_limit: int = 6,
) -> dict[str, Any]:
    return {
        "tensors": len(named_params),
        "numel": sum(int(param.numel()) for _, param in named_params),
        "sample_names": [name for name, _ in named_params[:sample_limit]],
    }


def _coerce_parameter_stats_fields(
    stats: Mapping[str, object],
) -> tuple[int, int, list[str]]:
    raw_tensors = stats.get("tensors", 0)
    raw_numel = stats.get("numel", 0)
    raw_sample_names = stats.get("sample_names", [])
    tensors = raw_tensors if isinstance(raw_tensors, int) else 0
    numel = raw_numel if isinstance(raw_numel, int) else 0
    sample_names = (
        [str(name) for name in raw_sample_names]
        if isinstance(raw_sample_names, list)
        else []
    )
    return tensors, numel, sample_names


def _repo_local_success_probe_fp32_dtype(*, torch_module: Any | None) -> Any:
    if torch_module is None:
        import torch as imported_torch

        torch_module = imported_torch
    target_dtype = getattr(torch_module, "float32", None)
    if target_dtype is None:
        raise RuntimeError("torch module does not expose float32 dtype")
    return target_dtype


def _repo_local_tensor_is_floating_point(*, tensor: Any, torch_module: Any | None) -> bool:
    is_floating_point = getattr(tensor, "is_floating_point", None)
    if callable(is_floating_point):
        try:
            return bool(is_floating_point())
        except Exception:
            pass
    if torch_module is not None:
        torch_is_floating_point = getattr(torch_module, "is_floating_point", None)
        if callable(torch_is_floating_point):
            try:
                return bool(torch_is_floating_point(tensor))
            except Exception:
                pass
    return False


def cast_repo_local_success_probe_trainable_params_fp32(
    *,
    model: Any,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    target_dtype = _repo_local_success_probe_fp32_dtype(torch_module=torch_module)
    converted_named_params: list[tuple[str, Any]] = []
    already_target_dtype_named_params: list[tuple[str, Any]] = []

    for _scope_name, canonical_name, param in _iter_repo_local_success_probe_named_parameters(model):
        if not bool(getattr(param, "requires_grad", False)):
            continue
        tensor = getattr(param, "data", None)
        if tensor is None:
            continue
        if not _repo_local_tensor_is_floating_point(tensor=tensor, torch_module=torch_module):
            continue
        current_dtype = getattr(tensor, "dtype", getattr(param, "dtype", None))
        if current_dtype == target_dtype:
            already_target_dtype_named_params.append((canonical_name, param))
            continue
        try:
            converted_tensor = tensor.to(dtype=target_dtype)
        except TypeError:
            converted_tensor = tensor.to(target_dtype)
        param.data = converted_tensor
        converted_named_params.append((canonical_name, param))

    return {
        "target_dtype": str(target_dtype),
        "converted": _named_parameter_stats(converted_named_params),
        "already_target_dtype": _named_parameter_stats(already_target_dtype_named_params),
    }


def logical_repo_local_optimizer_bucket_name(name: str) -> str:
    if name.startswith(REPO_LOCAL_CONDITION_HOT_PARAM_PREFIXES):
        return "condition_hot"
    if name.startswith(REPO_LOCAL_DIFFUSION_TRUNK_PARAM_PREFIX):
        return "diffusion_trunk_cold"
    return "default"


def _resolve_repo_local_requested_scope(
    authority_payload: Mapping[str, Any],
) -> str | None:
    scope_summary = authority_payload.get("scope_summary")
    if isinstance(scope_summary, Mapping):
        raw_scope = scope_summary.get("train_scope_requested")
        if raw_scope is not None:
            return parse_scope_flag(raw_scope)
    raw_scope = authority_payload.get("requested_scope")
    if raw_scope is None:
        return None
    return parse_scope_flag(raw_scope)


def _repo_local_should_disable_one_step_grad_clipping(
    *,
    authority: Mapping[str, Any] | None,
    max_steps: object,
) -> bool:
    requested_scope = _resolve_repo_local_requested_scope(_copy_mapping(authority) or {})
    if requested_scope not in {"full_action", "full_policy", "strict_full"}:
        return False
    if isinstance(max_steps, bool):
        return int(max_steps) == 1
    if isinstance(max_steps, int):
        return max_steps == 1
    if isinstance(max_steps, float):
        return int(max_steps) == 1
    if isinstance(max_steps, str):
        cleaned = max_steps.strip()
        if cleaned == "":
            return False
        try:
            return int(cleaned) == 1
        except ValueError:
            return False
    return False


def build_repo_local_effective_tuning_flags(
    *,
    requested_scope: object | None,
    tune_projector: bool,
    tune_diffusion_model: bool,
    tune_top_llm_layers: int,
    tune_vlln: bool,
    condition_focused_continuation: bool,
) -> dict[str, Any]:
    normalized_scope = (
        None if requested_scope is None else parse_scope_flag(requested_scope)
    )
    flags = {
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": bool(tune_projector),
        "tune_diffusion_model": bool(tune_diffusion_model),
        "tune_top_llm_layers": int(tune_top_llm_layers),
        "tune_vlln": bool(tune_vlln) or bool(condition_focused_continuation),
        "scope_authority_override_active": False,
        "requested_scope": normalized_scope,
    }
    if normalized_scope in {"full_action", "full_policy"}:
        flags.update(
            {
                "tune_projector": True,
                "tune_diffusion_model": True,
                "tune_top_llm_layers": 0,
                "tune_vlln": True,
                "scope_authority_override_active": True,
            }
        )
    elif normalized_scope == "strict_full":
        flags.update(
            {
                "tune_llm": True,
                "tune_visual": True,
                "tune_projector": True,
                "tune_diffusion_model": True,
                "tune_top_llm_layers": 0,
                "tune_vlln": True,
                "scope_authority_override_active": True,
            }
        )
    return flags


def _scope_trainability_target(
    *,
    requested_scope: str | None,
    parameter_name: str,
) -> bool | None:
    if requested_scope is None or requested_scope == "current_partial":
        return None
    canonical_name = _canonical_repo_local_param_name(parameter_name)
    if requested_scope == "strict_full":
        return True
    if requested_scope == "full_action":
        return canonical_name.startswith(REPO_LOCAL_FULL_ACTION_PARAM_PREFIXES)
    if requested_scope == "full_policy":
        return canonical_name.startswith(REPO_LOCAL_FULL_POLICY_PARAM_PREFIXES)
    return None


def apply_repo_local_trainability_authority(
    *,
    model: Any,
    authority: Mapping[str, Any] | None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    authority_payload = _copy_mapping(authority) or {
        "schema_version": REPO_LOCAL_TRAINABILITY_AUTHORITY_SCHEMA_VERSION,
        "authority_owner": REPO_LOCAL_TRAINABILITY_AUTHORITY_OWNER,
    }
    summary: dict[str, Any] = dict(authority_payload)
    summary.setdefault("schema_version", REPO_LOCAL_TRAINABILITY_AUTHORITY_SCHEMA_VERSION)
    summary.setdefault("authority_owner", REPO_LOCAL_TRAINABILITY_AUTHORITY_OWNER)
    summary["forced_trainable"] = _named_parameter_stats([])
    summary["forced_frozen"] = _named_parameter_stats([])

    forced_trainable: list[tuple[str, Any]] = []
    forced_frozen: list[tuple[str, Any]] = []
    requested_scope = _resolve_repo_local_requested_scope(authority_payload)
    summary["effective_requested_scope"] = requested_scope
    summary["scope_authority_override_active"] = requested_scope in {
        "full_action",
        "full_policy",
        "strict_full",
    }
    condition_focused_continuation = bool(
        authority_payload.get("condition_focused_continuation", False)
    )
    freeze_diffusion_trunk = (
        float(authority_payload.get("diffusion_trunk_lr_scale", 1.0)) <= 0.0
    )
    requires_advantage_embedding = bool(
        authority_payload.get("advantage_embedding_required_by_scope", False)
    )

    for name, param in model.named_parameters():
        scope_target = _scope_trainability_target(
            requested_scope=requested_scope,
            parameter_name=name,
        )
        if scope_target is True and not bool(getattr(param, "requires_grad", False)):
            param.requires_grad_(True)
            forced_trainable.append((name, param))
            continue
        if scope_target is False and bool(getattr(param, "requires_grad", False)):
            param.requires_grad_(False)
            forced_frozen.append((name, param))
            continue
        if scope_target is not None:
            continue

        should_force_trainable = False
        if requires_advantage_embedding and name.startswith(
            "action_head.advantage_embedding."
        ):
            should_force_trainable = True
        if condition_focused_continuation and name.startswith(
            REPO_LOCAL_CONDITION_HOT_PARAM_PREFIXES
        ):
            should_force_trainable = True

        if should_force_trainable and not bool(getattr(param, "requires_grad", False)):
            param.requires_grad_(True)
            forced_trainable.append((name, param))

        if (
            condition_focused_continuation
            and freeze_diffusion_trunk
            and name.startswith(REPO_LOCAL_DIFFUSION_TRUNK_PARAM_PREFIX)
            and bool(getattr(param, "requires_grad", False))
        ):
            param.requires_grad_(False)
            forced_frozen.append((name, param))

    summary["forced_trainable"] = _named_parameter_stats(forced_trainable)
    summary["forced_frozen"] = _named_parameter_stats(forced_frozen)
    summary["success_probe_trainable_params_fp32"] = (
        cast_repo_local_success_probe_trainable_params_fp32(
            model=model,
            torch_module=torch_module,
        )
    )
    setattr(model, "_repo_local_trainability_authority_summary", summary)
    return summary


def build_repo_local_optimizer_group_plan(
    *,
    trainer: Any,
    authority: Mapping[str, Any] | None,
    sample_limit: int = 6,
) -> dict[str, Any] | None:
    authority_payload = _copy_mapping(authority) or {}
    if not bool(authority_payload.get("condition_focused_continuation", False)):
        return None

    opt_model = getattr(trainer, "model_wrapped", None) or trainer.model
    decay_parameters = set(trainer.get_decay_parameter_names(opt_model))
    base_lr = float(trainer.args.learning_rate)
    logical_group_lrs = {
        "condition_hot": base_lr * float(authority_payload["condition_hot_lr_scale"]),
        "diffusion_trunk_cold": base_lr
        * float(authority_payload["diffusion_trunk_lr_scale"]),
        "default": base_lr,
    }
    logical_named_params: dict[str, list[tuple[str, Any]]] = {
        "condition_hot": [],
        "diffusion_trunk_cold": [],
        "default": [],
    }
    for name, param in opt_model.named_parameters():
        if not bool(getattr(param, "requires_grad", False)):
            continue
        logical_named_params[logical_repo_local_optimizer_bucket_name(name)].append(
            (name, param)
        )

    optimizer_grouped_parameters: list[dict[str, Any]] = []
    logical_group_summaries: dict[str, dict[str, Any]] = {}
    diffusion_trunk_mode = str(authority_payload.get("diffusion_trunk_mode", "frozen"))
    authority_summary = _copy_mapping(
        getattr(opt_model, "_repo_local_trainability_authority_summary", None)
    ) or {}

    for logical_name, named_params in logical_named_params.items():
        decay_named = [
            (name, param) for name, param in named_params if name in decay_parameters
        ]
        no_decay_named = [
            (name, param) for name, param in named_params if name not in decay_parameters
        ]
        if decay_named:
            optimizer_grouped_parameters.append(
                {
                    "params": [param for _, param in decay_named],
                    "weight_decay": float(trainer.args.weight_decay),
                    "lr": float(logical_group_lrs[logical_name]),
                }
            )
        if no_decay_named:
            optimizer_grouped_parameters.append(
                {
                    "params": [param for _, param in no_decay_named],
                    "weight_decay": 0.0,
                    "lr": float(logical_group_lrs[logical_name]),
                }
            )

        stats = _named_parameter_stats(named_params, sample_limit=sample_limit)
        mode = "trainable"
        if logical_name == "condition_hot":
            mode = "boosted"
        elif logical_name == "diffusion_trunk_cold":
            if not named_params:
                frozen_summary = _copy_mapping(authority_summary.get("forced_frozen")) or {}
                if frozen_summary:
                    stats = {
                        "tensors": int(frozen_summary.get("tensors", 0)),
                        "numel": int(frozen_summary.get("numel", 0)),
                        "sample_names": list(frozen_summary.get("sample_names", [])),
                    }
                mode = diffusion_trunk_mode
            else:
                mode = "downscaled"

        logical_group_summaries[logical_name] = {
            "lr": float(logical_group_lrs[logical_name]),
            "mode": mode,
            **stats,
        }

    if not optimizer_grouped_parameters:
        raise RuntimeError(
            "Condition-focused continuation produced no optimizer parameter groups."
        )

    setattr(opt_model, "_repo_local_optimizer_bucket_summary", logical_group_summaries)
    return {
        "optimizer_grouped_parameters": optimizer_grouped_parameters,
        "logical_group_summaries": logical_group_summaries,
    }


def create_repo_local_optimizer_for_trainer(
    *,
    trainer: Any,
    authority: Mapping[str, Any] | None,
    emit_info_line: Any = None,
) -> Any | None:
    if trainer.optimizer is not None:
        return trainer.optimizer
    plan = build_repo_local_optimizer_group_plan(trainer=trainer, authority=authority)
    if plan is None:
        return None

    log = emit_info_line
    if log is None:
        log = lambda message: print(f"[INFO] {message}")
    for logical_name, stats in plan["logical_group_summaries"].items():
        tensors, numel, sample_names = _coerce_parameter_stats_fields(stats)
        log(
            "optimizer_group "
            f"{logical_name} "
            f"lr={float(stats['lr']):.6g} "
            f"mode={stats['mode']} "
            f"tensors={tensors} "
            f"numel={numel} "
            f"sample_names={sample_names}"
        )

    from transformers.trainer import Trainer

    opt_model = getattr(trainer, "model_wrapped", None) or trainer.model
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
        trainer.args,
        opt_model,
    )
    trainer.optimizer = optimizer_cls(
        plan["optimizer_grouped_parameters"],
        **optimizer_kwargs,
    )
    setattr(trainer, "_repo_local_optimizer_bucket_summary", plan["logical_group_summaries"])
    return trainer.optimizer


def emit_repo_local_static_scope_audit(
    *,
    trainer: Any,
    authority: Mapping[str, Any] | None,
    requested_scope: object | None = None,
) -> Path | None:
    if not _is_repo_local_rank0():
        return None

    metadata_dir = REPO_LOCAL_CENSUS_HOOK_STATE.get("metadata_dir")
    if metadata_dir is None:
        return None

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        return None

    model = getattr(trainer, "model_wrapped", None) or getattr(trainer, "model", None)
    if model is None:
        return None

    authority_payload = _copy_mapping(authority) or {}
    scope_input: object | None = requested_scope
    scope_summary = authority_payload.get("scope_summary")
    if isinstance(scope_summary, Mapping):
        scope_input = scope_summary
    elif scope_input is None:
        scope_input = authority_payload.get("requested_scope")
    if scope_input is None:
        return None

    audit = compute_static_scope_audit(
        model=model,
        optimizer=optimizer,
        scope_requested=scope_input,
    )
    setattr(trainer, "_repo_local_static_scope_audit", audit)
    setattr(model, "_repo_local_static_scope_audit", audit)
    return emit_scope_audit_json(Path(metadata_dir) / STATIC_SCOPE_AUDIT_FILENAME, audit)


def _safe_cuda_current_device(torch_module: Any) -> int | None:
    try:
        return int(torch_module.cuda.current_device())
    except Exception:
        return None


def _safe_tensor_shape(tensor: Any) -> list[int]:
    try:
        return [int(dim) for dim in tensor.shape]
    except Exception:
        return []


def _safe_tensor_numel(tensor: Any) -> int:
    try:
        return int(tensor.numel())
    except Exception:
        return 0


def _safe_exception_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _is_repo_local_rank0() -> bool:
    return _env_int("RANK", default=0) == 0


def _repo_local_rank0_metadata_path(filename: str) -> Path:
    metadata_dir = REPO_LOCAL_CENSUS_HOOK_STATE["metadata_dir"]
    if metadata_dir is None:
        raise RuntimeError("repo-local census hook state is not initialized")
    return Path(metadata_dir) / filename


def _stats_hash_payload_for_trainer(*, trainer: Any, model: Any) -> dict[str, Any]:
    train_dataset = getattr(trainer, "train_dataset", None)
    processor = getattr(train_dataset, "processor", None)
    processor_statistics = getattr(processor, "statistics", None)
    processor_norm_params = getattr(processor, "norm_params", None)
    train_dataset_global_stats = getattr(train_dataset, "global_stats", None)
    if train_dataset_global_stats is None and hasattr(train_dataset, "get_dataset_statistics"):
        try:
            train_dataset_global_stats = train_dataset.get_dataset_statistics()
        except Exception as exc:
            train_dataset_global_stats = {
                "unavailable_reason": _safe_exception_reason(exc),
            }

    state = REPO_LOCAL_CENSUS_HOOK_STATE
    batch_index = int(state.get("stats_hash_batch_index", 0) or 0) + 1
    state["stats_hash_batch_index"] = batch_index

    global_step = getattr(trainer, "state", None)
    global_step_value = getattr(global_step, "global_step", None)
    try:
        global_step_int = int(global_step_value)
    except Exception:
        global_step_int = None

    payload = {
        "schema_version": "repo_local_normalization_stats_hash_per_batch_v1",
        "event": "before_compute_loss",
        "batch_index": batch_index,
        "trainer_global_step": global_step_int,
        "rank": _env_int("RANK", default=0),
        "local_rank": _env_int("LOCAL_RANK", default=0),
        "world_size": _env_int("WORLD_SIZE", default=0),
        "train_dataset_type": (
            None
            if train_dataset is None
            else f"{type(train_dataset).__module__}.{type(train_dataset).__qualname__}"
        ),
        "processor_type": (
            None
            if processor is None
            else f"{type(processor).__module__}.{type(processor).__qualname__}"
        ),
        "model_type": (
            None if model is None else f"{type(model).__module__}.{type(model).__qualname__}"
        ),
        "override_pretraining_statistics": getattr(
            train_dataset,
            "override_pretraining_statistics",
            None,
        ),
        "processor_statistics_sha256": _stable_stats_sha256(processor_statistics),
        "processor_norm_params_sha256": _stable_stats_sha256(processor_norm_params),
        "train_dataset_global_stats_sha256": _stable_stats_sha256(
            train_dataset_global_stats
        ),
        "processor_statistics_embodiments": (
            []
            if not isinstance(processor_statistics, Mapping)
            else sorted(str(key) for key in processor_statistics.keys())
        ),
        "train_dataset_global_stats_embodiments": (
            []
            if not isinstance(train_dataset_global_stats, Mapping)
            else sorted(str(key) for key in train_dataset_global_stats.keys())
        ),
        "actual_normalization_authority": (
            "processor.statistics_and_norm_params"
            if processor_statistics is not None and processor_norm_params is not None
            else "unavailable"
        ),
    }
    payload["processor_stats_match_train_dataset_global_stats"] = (
        payload["processor_statistics_sha256"]
        == payload["train_dataset_global_stats_sha256"]
    )
    state["stats_hash_last_payload"] = payload
    return payload


def _write_rank0_normalization_stats_hash_event(
    *,
    trainer: Any,
    model: Any,
) -> Path | None:
    if not _is_repo_local_rank0():
        return None
    try:
        event = _stats_hash_payload_for_trainer(trainer=trainer, model=model)
        path = _repo_local_rank0_metadata_path(NORMALIZATION_STATS_HASH_JSONL_FILENAME)
        _append_jsonl_atomic(path, event)
        REPO_LOCAL_CENSUS_HOOK_STATE["stats_hash_jsonl_path"] = path
        print(
            "[INFO] repo_local_normalization_stats_hash "
            f"batch_index={event['batch_index']} "
            f"global_step={event['trainer_global_step']} "
            f"processor_statistics_sha256={event['processor_statistics_sha256']} "
            f"norm_params_sha256={event['processor_norm_params_sha256']} "
            f"dataset_global_stats_sha256={event['train_dataset_global_stats_sha256']} "
            f"jsonl={path}",
            flush=True,
        )
        return path
    except Exception as exc:
        reason = _safe_exception_reason(exc)
        print(f"[WARN] repo_local_normalization_stats_hash_failed={reason}", flush=True)
        return None


def _write_rank0_normalization_stats_hash_summary() -> Path | None:
    if not _is_repo_local_rank0():
        return None
    last_payload = REPO_LOCAL_CENSUS_HOOK_STATE.get("stats_hash_last_payload")
    jsonl_path = REPO_LOCAL_CENSUS_HOOK_STATE.get("stats_hash_jsonl_path")
    if not isinstance(last_payload, Mapping):
        return None
    payload = {
        "schema_version": "repo_local_normalization_stats_hash_summary_v1",
        "artifact_kind": "normalization_stats_hash_summary",
        "per_batch_jsonl": str(jsonl_path) if jsonl_path is not None else None,
        "batch_records_emitted": int(
            REPO_LOCAL_CENSUS_HOOK_STATE.get("stats_hash_batch_index", 0) or 0
        ),
        "last_batch": dict(last_payload),
        "contract_note": (
            "Hashes are emitted immediately before Gr00tTrainer.compute_loss; "
            "processor.statistics and processor.norm_params are the repo-local "
            "authority for state/action normalization in this training path."
        ),
    }
    path = _repo_local_rank0_metadata_path(NORMALIZATION_STATS_HASH_SUMMARY_FILENAME)
    return _write_json_atomic(path, payload)


def _safe_subprocess_capture(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return {
            "ok": False,
            "command": command,
            "reason": _safe_exception_reason(exc),
            "stdout": "",
            "stderr": "",
        }
    return {
        "ok": True,
        "command": command,
        "reason": None,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _parse_csv_noheader_rows(stdout: str, field_names: list[str]) -> list[dict[str, str]]:
    cleaned_stdout = stdout.strip()
    if cleaned_stdout == "":
        return []

    rows: list[dict[str, str]] = []
    reader = csv.reader(StringIO(cleaned_stdout), skipinitialspace=True)
    for raw_row in reader:
        if len(raw_row) < len(field_names):
            raw_row = list(raw_row) + [""] * (len(field_names) - len(raw_row))
        rows.append(
            {
                field_name: str(raw_row[index]).strip()
                for index, field_name in enumerate(field_names)
            }
        )
    return rows


def _write_rank0_torch_cuda_memory_snapshot(*, torch_module: Any) -> Path | None:
    if not _is_repo_local_rank0():
        return None

    payload: dict[str, Any] = {
        "available": False,
        "rank": _env_int("RANK", default=0),
        "local_rank": _env_int("LOCAL_RANK", default=0),
        "world_size": _env_int("WORLD_SIZE", default=0),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        "runtime_pid": int(os.getpid()),
    }

    try:
        if not bool(torch_module.cuda.is_available()):
            payload["reason"] = "torch.cuda.is_available() returned False"
        else:
            current_device = _safe_cuda_current_device(torch_module)
            payload.update(
                {
                    "available": True,
                    "reason": None,
                    "current_device": current_device,
                    "device_count": int(torch_module.cuda.device_count()),
                    "device_name": (
                        str(torch_module.cuda.get_device_name(current_device))
                        if current_device is not None
                        else None
                    ),
                    "memory_allocated_bytes": int(torch_module.cuda.memory_allocated()),
                    "memory_reserved_bytes": int(torch_module.cuda.memory_reserved()),
                    "max_memory_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
                    "max_memory_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
                }
            )
            free_bytes, total_bytes = torch_module.cuda.mem_get_info()
            payload["mem_get_info_bytes"] = {
                "free": int(free_bytes),
                "total": int(total_bytes),
            }
    except Exception as exc:
        payload["available"] = False
        payload["reason"] = _safe_exception_reason(exc)

    return _write_json_atomic(
        _repo_local_rank0_metadata_path("torch_cuda_memory_rank0.json"),
        payload,
    )


def _write_rank0_nvidia_smi_active_sampling() -> Path | None:
    if not _is_repo_local_rank0():
        return None

    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    compute_apps_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    gpu_result = _safe_subprocess_capture(gpu_command)
    compute_apps_result = _safe_subprocess_capture(compute_apps_command)
    current_pid = int(os.getpid())
    payload: dict[str, Any] = {
        "available": False,
        "rank": _env_int("RANK", default=0),
        "local_rank": _env_int("LOCAL_RANK", default=0),
        "world_size": _env_int("WORLD_SIZE", default=0),
        "runtime_pid": current_pid,
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        "current_device": None,
        "gpu_query": gpu_result,
        "compute_apps_query": compute_apps_result,
    }

    torch_module = REPO_LOCAL_CENSUS_HOOK_STATE["torch_module"]
    if torch_module is not None:
        payload["current_device"] = _safe_cuda_current_device(torch_module)

    if not bool(gpu_result["ok"]):
        payload["reason"] = f"gpu_query_failed: {gpu_result['reason']}"
    elif not bool(compute_apps_result["ok"]):
        payload["reason"] = f"compute_apps_query_failed: {compute_apps_result['reason']}"
    else:
        payload["available"] = True
        payload["reason"] = None
        payload["gpu_rows"] = _parse_csv_noheader_rows(
            str(gpu_result["stdout"]),
            [
                "index",
                "uuid",
                "name",
                "memory_total_mb",
                "memory_used_mb",
                "utilization_gpu_percent",
                "utilization_memory_percent",
            ],
        )
        payload["active_compute_apps"] = _parse_csv_noheader_rows(
            str(compute_apps_result["stdout"]),
            ["gpu_uuid", "pid", "process_name", "used_gpu_memory_mb"],
        )
        payload["matching_runtime_pid_entries"] = [
            entry
            for entry in payload["active_compute_apps"]
            if str(entry.get("pid", "")) == str(current_pid)
        ]

    return _write_json_atomic(
        _repo_local_rank0_metadata_path("nvidia_smi_active_sampling_rank0.json"),
        payload,
    )


def _build_optimizer_state_rank0_payload(*, trainer: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": False,
        "rank": _env_int("RANK", default=0),
        "local_rank": _env_int("LOCAL_RANK", default=0),
        "world_size": _env_int("WORLD_SIZE", default=0),
        "runtime_pid": int(os.getpid()),
    }

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        payload["reason"] = "trainer.optimizer is None at train-end hook"
        return payload

    try:
        state_dict = optimizer.state_dict()
    except Exception as exc:
        payload["reason"] = f"optimizer.state_dict() failed: {_safe_exception_reason(exc)}"
        return payload

    if not isinstance(state_dict, dict):
        payload["reason"] = "optimizer.state_dict() did not return a dict"
        return payload

    raw_state = state_dict.get("state")
    raw_param_groups = state_dict.get("param_groups")
    if not isinstance(raw_state, dict):
        payload["reason"] = "optimizer.state_dict()['state'] is unavailable or not a dict"
        return payload
    if not isinstance(raw_param_groups, list):
        payload["reason"] = "optimizer.state_dict()['param_groups'] is unavailable or not a list"
        return payload

    tensor_slot_count = 0
    tensor_slot_total_numel = 0
    scalar_slot_count = 0
    distinct_tensor_dtypes: set[str] = set()
    state_entry_key_counts: dict[str, int] = {}
    state_entries_preview: list[dict[str, Any]] = []

    for param_id, param_state in raw_state.items():
        if not isinstance(param_state, dict):
            scalar_slot_count += 1
            continue

        preview_entry: dict[str, Any] = {
            "param_id": str(param_id),
            "keys": sorted(str(key) for key in param_state.keys()),
            "tensor_keys": [],
            "scalar_keys": [],
        }
        for key, value in param_state.items():
            key_name = str(key)
            state_entry_key_counts[key_name] = int(state_entry_key_counts.get(key_name, 0)) + 1
            if hasattr(value, "shape") and hasattr(value, "numel"):
                tensor_slot_count += 1
                tensor_slot_total_numel += _safe_tensor_numel(value)
                distinct_tensor_dtypes.add(str(getattr(value, "dtype", "unknown")))
                if len(preview_entry["tensor_keys"]) < REPO_LOCAL_CENSUS_PREVIEW_LIMIT:
                    preview_entry["tensor_keys"].append(
                        {
                            "name": key_name,
                            "shape": _safe_tensor_shape(value),
                            "dtype": str(getattr(value, "dtype", "unknown")),
                            "numel": _safe_tensor_numel(value),
                        }
                    )
            else:
                scalar_slot_count += 1
                if len(preview_entry["scalar_keys"]) < REPO_LOCAL_CENSUS_PREVIEW_LIMIT:
                    preview_entry["scalar_keys"].append(key_name)

        if len(state_entries_preview) < REPO_LOCAL_CENSUS_PREVIEW_LIMIT:
            state_entries_preview.append(preview_entry)

    payload.update(
        {
            "available": True,
            "reason": None,
            "optimizer_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "optimizer_class_name": type(optimizer).__name__,
            "state_entry_count": int(len(raw_state)),
            "param_group_count": int(len(raw_param_groups)),
            "tensor_slot_count": int(tensor_slot_count),
            "tensor_slot_total_numel": int(tensor_slot_total_numel),
            "scalar_slot_count": int(scalar_slot_count),
            "distinct_tensor_dtypes": sorted(distinct_tensor_dtypes),
            "state_entry_key_counts": state_entry_key_counts,
            "state_entries_preview": state_entries_preview,
            "param_groups_preview": raw_param_groups[:REPO_LOCAL_CENSUS_PREVIEW_LIMIT],
            "trainer_global_step": int(getattr(getattr(trainer, "state", None), "global_step", 0)),
        }
    )
    return payload


def _write_rank0_optimizer_state_snapshot(*, trainer: Any) -> Path | None:
    if not _is_repo_local_rank0():
        return None

    payload = _build_optimizer_state_rank0_payload(trainer=trainer)
    return _write_json_atomic(
        _repo_local_rank0_metadata_path("optimizer_state_rank0.json"),
        payload,
    )


def _canonical_repo_local_param_name(name: str) -> str:
    if name.startswith("module."):
        return name[len("module.") :]
    return name


def _repo_local_success_probe_scope_name(name: str) -> str | None:
    canonical_name = _canonical_repo_local_param_name(name)
    for scope_name, prefix in REPO_LOCAL_SUCCESS_PROBE_SCOPE_PREFIXES.items():
        if canonical_name.startswith(prefix):
            return scope_name
    return None


def _empty_repo_local_success_probe_scope_summary(
    *,
    scope_name: str,
) -> dict[str, Any]:
    return {
        "scope": scope_name,
        "param_prefix": REPO_LOCAL_SUCCESS_PROBE_SCOPE_PREFIXES[scope_name],
        "present": False,
        "tensor_count": 0,
        "numel": 0,
        "sample_names": [],
    }


def _iter_repo_local_success_probe_named_parameters(model: Any):
    for raw_name, param in model.named_parameters():
        canonical_name = _canonical_repo_local_param_name(str(raw_name))
        scope_name = _repo_local_success_probe_scope_name(canonical_name)
        if scope_name is None:
            continue
        yield scope_name, canonical_name, param


def _repo_local_success_probe_model_candidates(*, trainer: Any) -> list[Any]:
    candidates: list[Any] = []
    for model in (getattr(trainer, "model_wrapped", None), getattr(trainer, "model", None)):
        if model is None:
            continue
        if any(existing is model for existing in candidates):
            continue
        candidates.append(model)
    return candidates


def _build_repo_local_success_probe_name_lookup(*, trainer: Any) -> dict[int, tuple[str, str]]:
    name_lookup: dict[int, tuple[str, str]] = {}
    for model in _repo_local_success_probe_model_candidates(trainer=trainer):
        named_parameters = getattr(model, "named_parameters", None)
        if not callable(named_parameters):
            continue
        for scope_name, canonical_name, param in _iter_repo_local_success_probe_named_parameters(model):
            name_lookup.setdefault(id(param), (scope_name, canonical_name))
    return name_lookup


def _iter_repo_local_optimizer_param_groups(optimizer: Any):
    current = optimizer
    visited_optimizer_ids: set[int] = set()
    while current is not None and id(current) not in visited_optimizer_ids:
        visited_optimizer_ids.add(id(current))
        raw_param_groups = getattr(current, "param_groups", None)
        if isinstance(raw_param_groups, (list, tuple)):
            for group in raw_param_groups:
                if isinstance(group, Mapping):
                    yield group
            return
        current = getattr(current, "optimizer", None)


def _iter_repo_local_success_probe_optimizer_named_parameters(*, trainer: Any, optimizer: Any):
    name_lookup = _build_repo_local_success_probe_name_lookup(trainer=trainer)
    seen_param_ids: set[int] = set()
    for param_group in _iter_repo_local_optimizer_param_groups(optimizer):
        raw_params = param_group.get("params")
        if not isinstance(raw_params, (list, tuple)):
            continue
        for param in raw_params:
            param_id = id(param)
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)
            metadata = name_lookup.get(param_id)
            if metadata is None:
                continue
            scope_name, canonical_name = metadata
            yield scope_name, canonical_name, param


def _capture_repo_local_first_step_parameter_snapshot(
    *,
    trainer: Any,
    optimizer: Any,
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {
        scope_name: {} for scope_name in REPO_LOCAL_SUCCESS_PROBE_SCOPE_PREFIXES
    }
    for scope_name, canonical_name, param in _iter_repo_local_success_probe_optimizer_named_parameters(
        trainer=trainer,
        optimizer=optimizer,
    ):
        snapshots[scope_name][canonical_name] = param.detach().float().cpu().clone()
    return snapshots


def _capture_repo_local_success_probe_parameter_snapshot_from_trainer_models(
    *,
    trainer: Any,
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {
        scope_name: {} for scope_name in REPO_LOCAL_SUCCESS_PROBE_SCOPE_PREFIXES
    }
    for model in _repo_local_success_probe_model_candidates(trainer=trainer):
        named_parameters = getattr(model, "named_parameters", None)
        if not callable(named_parameters):
            continue
        for scope_name, canonical_name, param in _iter_repo_local_success_probe_named_parameters(model):
            snapshots[scope_name][canonical_name] = param.detach().float().cpu().clone()
    return snapshots


def _resolve_repo_local_checkpoint_dir_for_trainer(*, trainer: Any) -> Path | None:
    metadata_dir = REPO_LOCAL_CENSUS_HOOK_STATE.get("metadata_dir")
    if metadata_dir is None:
        return None
    trainer_global_step = _coerce_optional_int(
        getattr(getattr(trainer, "state", None), "global_step", None)
    )
    if trainer_global_step is None or trainer_global_step < 1:
        return None
    return Path(metadata_dir).parent / f"checkpoint-{trainer_global_step}"


def _capture_repo_local_success_probe_parameter_snapshot_from_checkpoint(
    *,
    checkpoint_dir: Path,
    pre_step_snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]] | None:
    try:
        from safetensors import safe_open
    except Exception:
        return None

    snapshots: dict[str, dict[str, Any]] = {
        scope_name: {} for scope_name in REPO_LOCAL_SUCCESS_PROBE_SCOPE_PREFIXES
    }
    shard_to_entries: dict[Path, list[tuple[str, str]]] = {}
    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_shard_path = checkpoint_dir / "model.safetensors"

    if index_path.is_file():
        index_payload = _load_json_file(index_path)
        if index_payload is None:
            return None
        raw_weight_map = index_payload.get("weight_map")
        if not isinstance(raw_weight_map, Mapping):
            return None
        for scope_name, scope_snapshot in pre_step_snapshot.items():
            for canonical_name in dict(scope_snapshot).keys():
                shard_rel = raw_weight_map.get(canonical_name)
                if not isinstance(shard_rel, str) or shard_rel.strip() == "":
                    continue
                shard_to_entries.setdefault(checkpoint_dir / shard_rel, []).append(
                    (str(scope_name), str(canonical_name))
                )
    elif single_shard_path.is_file():
        for scope_name, scope_snapshot in pre_step_snapshot.items():
            for canonical_name in dict(scope_snapshot).keys():
                shard_to_entries.setdefault(single_shard_path, []).append(
                    (str(scope_name), str(canonical_name))
                )
    else:
        return None

    for shard_path, entries in shard_to_entries.items():
        if not shard_path.is_file():
            continue
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            available_keys = set(handle.keys())
            for scope_name, canonical_name in entries:
                if canonical_name not in available_keys:
                    continue
                snapshots[scope_name][canonical_name] = handle.get_tensor(canonical_name).float().cpu()
    return snapshots


def _build_rank0_first_backward_grad_probe_payload(*, trainer: Any, model: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": False,
        "rank": _env_int("RANK", default=0),
        "local_rank": _env_int("LOCAL_RANK", default=0),
        "world_size": _env_int("WORLD_SIZE", default=0),
        "runtime_pid": int(os.getpid()),
        "trainer_global_step": int(getattr(getattr(trainer, "state", None), "global_step", 0)),
        "scopes": {
            scope_name: {
                **_empty_repo_local_success_probe_scope_summary(scope_name=scope_name),
                "grad_present_tensor_count": 0,
                "nonzero_grad_tensor_count": 0,
                "grad_abs_max": 0.0,
                "grad_l2_norm": 0.0,
            }
            for scope_name in REPO_LOCAL_SUCCESS_PROBE_SCOPE_PREFIXES
        },
    }
    if model is None:
        payload["reason"] = "trainer model is unavailable at first backward probe"
        return payload

    grad_sq_sums = {scope_name: 0.0 for scope_name in REPO_LOCAL_SUCCESS_PROBE_SCOPE_PREFIXES}
    for scope_name, canonical_name, param in _iter_repo_local_success_probe_named_parameters(model):
        scope_payload = payload["scopes"][scope_name]
        scope_payload["present"] = True
        scope_payload["tensor_count"] += 1
        scope_payload["numel"] += int(param.numel())
        if len(scope_payload["sample_names"]) < REPO_LOCAL_CENSUS_PREVIEW_LIMIT:
            scope_payload["sample_names"].append(canonical_name)
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        detached_grad = grad.detach().float()
        scope_payload["grad_present_tensor_count"] += 1
        if detached_grad.numel() == 0:
            continue
        grad_abs_max = float(detached_grad.abs().max().item())
        grad_sq_sums[scope_name] += float(detached_grad.square().sum().item())
        scope_payload["grad_abs_max"] = max(float(scope_payload["grad_abs_max"]), grad_abs_max)
        if grad_abs_max > 0.0:
            scope_payload["nonzero_grad_tensor_count"] += 1

    for scope_name, sq_sum in grad_sq_sums.items():
        payload["scopes"][scope_name]["grad_l2_norm"] = float(math.sqrt(sq_sum))

    payload["available"] = True
    payload["reason"] = None
    return payload


def _write_rank0_first_backward_grad_probe(*, trainer: Any, model: Any) -> Path | None:
    if not _is_repo_local_rank0():
        return None
    payload = _build_rank0_first_backward_grad_probe_payload(trainer=trainer, model=model)
    return _write_json_atomic(
        _repo_local_rank0_metadata_path(FIRST_BACKWARD_GRAD_PROBE_FILENAME),
        payload,
    )


def _build_rank0_first_optimizer_step_param_delta_payload(
    *,
    trainer: Any,
    optimizer: Any,
    pre_step_snapshot: Mapping[str, Mapping[str, Any]],
    current_snapshot: Mapping[str, Mapping[str, Any]] | None = None,
    trainer_global_step_override: int | None = None,
) -> dict[str, Any]:
    trainer_global_step = (
        int(getattr(getattr(trainer, "state", None), "global_step", 0)) + 1
        if trainer_global_step_override is None
        else int(trainer_global_step_override)
    )
    payload: dict[str, Any] = {
        "available": False,
        "rank": _env_int("RANK", default=0),
        "local_rank": _env_int("LOCAL_RANK", default=0),
        "world_size": _env_int("WORLD_SIZE", default=0),
        "runtime_pid": int(os.getpid()),
        "trainer_global_step": trainer_global_step,
        "scopes": {
            scope_name: {
                **_empty_repo_local_success_probe_scope_summary(scope_name=scope_name),
                "snapshot_tensor_count": len(dict(pre_step_snapshot.get(scope_name, {}))),
                "nonzero_delta_tensor_count": 0,
                "delta_abs_max": 0.0,
                "delta_l2_norm": 0.0,
            }
            for scope_name in REPO_LOCAL_SUCCESS_PROBE_SCOPE_PREFIXES
        },
    }
    if current_snapshot is None and optimizer is None:
        payload["reason"] = "trainer optimizer is unavailable at first optimizer-step probe"
        return payload

    resolved_current_snapshot = (
        _capture_repo_local_first_step_parameter_snapshot(trainer=trainer, optimizer=optimizer)
        if current_snapshot is None
        else {
            scope_name: {
                str(canonical_name): tensor
                for canonical_name, tensor in dict(scope_snapshot).items()
            }
            for scope_name, scope_snapshot in dict(current_snapshot).items()
        }
    )
    delta_sq_sums = {scope_name: 0.0 for scope_name in REPO_LOCAL_SUCCESS_PROBE_SCOPE_PREFIXES}
    for scope_name, scope_snapshot in resolved_current_snapshot.items():
        scope_payload = payload["scopes"][scope_name]
        for canonical_name, current_tensor in dict(scope_snapshot).items():
            scope_payload["present"] = True
            scope_payload["tensor_count"] += 1
            scope_payload["numel"] += int(current_tensor.numel())
            if len(scope_payload["sample_names"]) < REPO_LOCAL_CENSUS_PREVIEW_LIMIT:
                scope_payload["sample_names"].append(canonical_name)
            before_tensor = dict(pre_step_snapshot.get(scope_name, {})).get(canonical_name)
            if before_tensor is None:
                continue
            delta = current_tensor.detach().float().cpu() - before_tensor
            if delta.numel() == 0:
                continue
            delta_abs_max = float(delta.abs().max().item())
            delta_sq_sums[scope_name] += float(delta.square().sum().item())
            scope_payload["delta_abs_max"] = max(
                float(scope_payload["delta_abs_max"]), delta_abs_max
            )
            if delta_abs_max > 0.0:
                scope_payload["nonzero_delta_tensor_count"] += 1

    for scope_name, sq_sum in delta_sq_sums.items():
        payload["scopes"][scope_name]["delta_l2_norm"] = float(math.sqrt(sq_sum))

    payload["available"] = True
    payload["reason"] = None
    return payload


def _write_rank0_first_optimizer_step_param_delta_probe(
    *,
    trainer: Any,
    optimizer: Any,
    pre_step_snapshot: Mapping[str, Mapping[str, Any]],
    payload_override: Mapping[str, Any] | None = None,
) -> Path | None:
    if not _is_repo_local_rank0():
        return None
    payload = (
        dict(payload_override)
        if payload_override is not None
        else _build_rank0_first_optimizer_step_param_delta_payload(
            trainer=trainer,
            optimizer=optimizer,
            pre_step_snapshot=pre_step_snapshot,
        )
    )
    return _write_json_atomic(
        _repo_local_rank0_metadata_path(FIRST_OPTIMIZER_STEP_PARAM_DELTA_FILENAME),
        payload,
    )


def _rank0_first_optimizer_step_payload_all_zero(payload: Mapping[str, Any]) -> bool:
    raw_scopes = payload.get("scopes")
    if not isinstance(raw_scopes, Mapping):
        return True
    for scope_name in REPO_LOCAL_SUCCESS_PROBE_SCOPE_PREFIXES:
        scope_payload = raw_scopes.get(scope_name)
        if not isinstance(scope_payload, Mapping):
            continue
        nonzero_delta_tensor_count = _coerce_optional_int(
            scope_payload.get("nonzero_delta_tensor_count", 0)
        )
        if nonzero_delta_tensor_count is not None and nonzero_delta_tensor_count > 0:
            return False
        delta_abs_max = _coerce_optional_float(scope_payload.get("delta_abs_max", 0.0))
        if delta_abs_max is not None and delta_abs_max > 0.0:
            return False
        delta_l2_norm = _coerce_optional_float(scope_payload.get("delta_l2_norm", 0.0))
        if delta_l2_norm is not None and delta_l2_norm > 0.0:
            return False
    return True


def _repo_local_optimizer_step_will_sync_gradients(*, optimizer: Any) -> bool:
    gradient_state = getattr(optimizer, "gradient_state", None)
    sync_gradients = getattr(gradient_state, "sync_gradients", None)
    if isinstance(sync_gradients, bool):
        return sync_gradients
    return True


def _repo_local_should_defer_first_optimizer_step_probe_to_train_end(*, trainer: Any) -> bool:
    trainer_args = getattr(trainer, "args", None)
    max_steps = _coerce_optional_int(
        None if trainer_args is None else getattr(trainer_args, "max_steps", None)
    )
    return max_steps == 1


def _maybe_finalize_repo_local_first_optimizer_step_probe_at_train_end(
    *,
    trainer: Any,
) -> Path | None:
    if bool(REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"]):
        return None

    raw_snapshot = REPO_LOCAL_CENSUS_HOOK_STATE.get(
        "first_optimizer_step_pending_snapshot"
    )
    if not isinstance(raw_snapshot, Mapping):
        return None
    normalized_snapshot = {
        str(scope_name): {
            str(canonical_name): tensor
            for canonical_name, tensor in dict(scope_snapshot).items()
        }
        for scope_name, scope_snapshot in raw_snapshot.items()
        if isinstance(scope_snapshot, Mapping)
    }

    trainer_global_step = int(getattr(getattr(trainer, "state", None), "global_step", 0))
    if trainer_global_step < 1:
        return None

    optimizer = getattr(trainer, "optimizer", None)
    payload = _build_rank0_first_optimizer_step_param_delta_payload(
        trainer=trainer,
        optimizer=optimizer,
        pre_step_snapshot=normalized_snapshot,
        trainer_global_step_override=trainer_global_step,
    )
    if _rank0_first_optimizer_step_payload_all_zero(payload):
        model_snapshot = _capture_repo_local_success_probe_parameter_snapshot_from_trainer_models(
            trainer=trainer
        )
        model_payload = _build_rank0_first_optimizer_step_param_delta_payload(
            trainer=trainer,
            optimizer=optimizer,
            pre_step_snapshot=normalized_snapshot,
            current_snapshot=model_snapshot,
            trainer_global_step_override=trainer_global_step,
        )
        if not _rank0_first_optimizer_step_payload_all_zero(model_payload):
            payload = model_payload
    if _rank0_first_optimizer_step_payload_all_zero(payload):
        checkpoint_dir = _resolve_repo_local_checkpoint_dir_for_trainer(trainer=trainer)
        if checkpoint_dir is not None:
            checkpoint_snapshot = (
                _capture_repo_local_success_probe_parameter_snapshot_from_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    pre_step_snapshot=normalized_snapshot,
                )
            )
            if checkpoint_snapshot is not None:
                checkpoint_payload = _build_rank0_first_optimizer_step_param_delta_payload(
                    trainer=trainer,
                    optimizer=optimizer,
                    pre_step_snapshot=normalized_snapshot,
                    current_snapshot=checkpoint_snapshot,
                    trainer_global_step_override=trainer_global_step,
                )
                if not _rank0_first_optimizer_step_payload_all_zero(checkpoint_payload):
                    payload = checkpoint_payload
    REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"] = True
    REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_pending_snapshot"] = None
    return _write_rank0_first_optimizer_step_param_delta_probe(
        trainer=trainer,
        optimizer=optimizer,
        pre_step_snapshot=normalized_snapshot,
        payload_override=payload,
    )


def _maybe_wrap_repo_local_optimizer_first_step_probe(*, trainer: Any, optimizer: Any) -> Any:
    if optimizer is None:
        return None
    if bool(getattr(optimizer, "_repo_local_first_step_probe_wrapped", False)):
        return optimizer

    original_step = getattr(optimizer, "step", None)
    if not callable(original_step):
        return optimizer

    original_step_func = getattr(original_step, "__func__", None)
    wraps_target = original_step_func if callable(original_step_func) else original_step

    @functools.wraps(wraps_target)
    def repo_local_optimizer_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        should_capture = (
            not bool(REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"])
            and _repo_local_optimizer_step_will_sync_gradients(optimizer=optimizer)
        )
        pre_step_snapshot = (
            None
            if not should_capture
            else _capture_repo_local_first_step_parameter_snapshot(
                trainer=trainer,
                optimizer=optimizer,
            )
        )
        if callable(original_step_func):
            result = original_step_func(self, *args, **kwargs)
        else:
            result = original_step(*args, **kwargs)
        if should_capture:
            if pre_step_snapshot is not None:
                if _repo_local_should_defer_first_optimizer_step_probe_to_train_end(
                    trainer=trainer
                ):
                    REPO_LOCAL_CENSUS_HOOK_STATE[
                        "first_optimizer_step_pending_snapshot"
                    ] = pre_step_snapshot
                    probe_path = None
                else:
                    REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"] = True
                    probe_path = _write_rank0_first_optimizer_step_param_delta_probe(
                        trainer=trainer,
                        optimizer=optimizer,
                        pre_step_snapshot=pre_step_snapshot,
                    )
                if probe_path is not None:
                    print(
                        "[INFO] repo_local_first_optimizer_step_param_delta_probe_path="
                        f"{probe_path}"
                    )
        return result

    optimizer.step = types.MethodType(repo_local_optimizer_step, optimizer)
    setattr(optimizer, "_repo_local_first_step_probe_wrapped", True)
    return optimizer


def _resolve_patch_b1_trigger_candidate_path() -> Path:
    raw_path = str(os.environ.get("REPO_LOCAL_B1_TRIGGER_CANDIDATE_PATH", "")).strip()
    if raw_path == "":
        return DEFAULT_PATCH_B1_TRIGGER_CANDIDATE_PATH
    return Path(raw_path)


def _resolve_patch_b2_trigger_candidate_path() -> Path:
    raw_path = str(os.environ.get("REPO_LOCAL_B2_TRIGGER_CANDIDATE_PATH", "")).strip()
    if raw_path == "":
        return DEFAULT_PATCH_B2_TRIGGER_CANDIDATE_PATH
    return Path(raw_path)


def _load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def detect_patch_b1_trigger(*, candidate_path: Path | None = None) -> dict[str, Any]:
    resolved_candidate_path = (
        candidate_path if candidate_path is not None else _resolve_patch_b1_trigger_candidate_path()
    )
    payload = _load_json_file(resolved_candidate_path)
    if payload is None:
        return {
            "activate": False,
            "candidate_path": str(resolved_candidate_path),
            "after_build_paths": [],
            "skip_reason": "candidate_missing_or_invalid",
        }

    census_checks = payload.get("census_checks")
    if not isinstance(census_checks, dict):
        return {
            "activate": False,
            "candidate_path": str(resolved_candidate_path),
            "after_build_paths": [],
            "skip_reason": "candidate_missing_census_checks",
        }

    raw_files = census_checks.get("files")
    if not isinstance(raw_files, list):
        return {
            "activate": False,
            "candidate_path": str(resolved_candidate_path),
            "after_build_paths": [],
            "skip_reason": "candidate_missing_census_files",
        }

    after_build_entries = [
        entry
        for entry in raw_files
        if isinstance(entry, dict) and str(entry.get("phase", "")) == "after_model_build"
    ]
    triggered_paths = [
        str(entry.get("path", ""))
        for entry in after_build_entries
        if bool(entry.get("exists")) and bool(entry.get("former_parameters_present"))
    ]
    activate = len(after_build_entries) >= 2 and len(triggered_paths) == len(after_build_entries)
    skip_reason = None
    if not activate:
        skip_reason = "former_parameters_not_present_on_all_after_build_ranks"

    return {
        "activate": activate,
        "candidate_path": str(resolved_candidate_path),
        "after_build_paths": [str(entry.get("path", "")) for entry in after_build_entries],
        "triggered_paths": triggered_paths,
        "skip_reason": skip_reason,
    }


def _first_parameter_tensor(model: Any) -> Any | None:
    try:
        return next(iter(model.parameters()))
    except StopIteration:
        return None
    except Exception:
        return None


def _first_former_parameter_tensor(model: Any) -> Any | None:
    former_parameters = getattr(model, "_former_parameters", None)
    if former_parameters is None or not hasattr(former_parameters, "items"):
        return None
    try:
        for _name, tensor in former_parameters.items():
            if tensor is not None:
                return tensor
    except Exception:
        return None
    return None


def _first_buffer_tensor(model: Any) -> Any | None:
    try:
        return next(iter(model.buffers()))
    except StopIteration:
        return None
    except Exception:
        return None


def _safe_dtype_fallback(torch_module: Any) -> Any:
    get_default_dtype = getattr(torch_module, "get_default_dtype", None)
    if callable(get_default_dtype):
        try:
            return get_default_dtype()
        except Exception:
            pass

    for attr_name in ("float32", "bfloat16", "float16"):
        if hasattr(torch_module, attr_name):
            return getattr(torch_module, attr_name)

    raise RuntimeError("torch module does not expose a safe dtype fallback")


def resolve_repo_local_model_device(model: Any, *, torch_module: Any) -> tuple[Any, str]:
    for source, tensor in (
        ("parameters", _first_parameter_tensor(model)),
        ("_former_parameters", _first_former_parameter_tensor(model)),
        ("buffers", _first_buffer_tensor(model)),
    ):
        if tensor is not None:
            return getattr(tensor, "device"), source

    local_rank = _env_int("LOCAL_RANK", default=0)
    return torch_module.device(f"cuda:{local_rank}"), "local_rank_fallback"


def resolve_repo_local_model_dtype(model: Any, *, torch_module: Any) -> tuple[Any, str]:
    for source, tensor in (
        ("parameters", _first_parameter_tensor(model)),
        ("_former_parameters", _first_former_parameter_tensor(model)),
        ("buffers", _first_buffer_tensor(model)),
    ):
        if tensor is None:
            continue
        dtype = getattr(tensor, "dtype", None)
        if dtype is not None:
            return dtype, source

    return _safe_dtype_fallback(torch_module), "safe_dtype_fallback"


def _install_repo_local_device_dtype_properties_for_class(
    *,
    target_cls: type[Any],
    torch_module: Any,
) -> bool:
    if bool(getattr(target_cls, "_repo_local_b1_device_dtype_patch", False)):
        return False

    @property
    def repo_local_device(self: Any) -> Any:
        return resolve_repo_local_model_device(self, torch_module=torch_module)[0]

    @property
    def repo_local_dtype(self: Any) -> Any:
        return resolve_repo_local_model_dtype(self, torch_module=torch_module)[0]

    target_cls.device = repo_local_device
    target_cls.dtype = repo_local_dtype
    target_cls._repo_local_b1_device_dtype_patch = True
    return True


def maybe_install_patch_b1(*, torch_module: Any) -> dict[str, Any]:
    trigger = detect_patch_b1_trigger()
    trigger["patched_classes"] = []
    if not bool(trigger["activate"]):
        return trigger

    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import (  # pyright: ignore[reportMissingImports]
        Gr00tN1d6,
        Gr00tN1d6ActionHead,
    )

    if _install_repo_local_device_dtype_properties_for_class(
        target_cls=Gr00tN1d6,
        torch_module=torch_module,
    ):
        trigger["patched_classes"].append("Gr00tN1d6")
    if _install_repo_local_device_dtype_properties_for_class(
        target_cls=Gr00tN1d6ActionHead,
        torch_module=torch_module,
    ):
        trigger["patched_classes"].append("Gr00tN1d6ActionHead")
    return trigger


def inspect_init_process_group_device_id_support(*, torch_dist_module: Any) -> dict[str, Any]:
    init_process_group = getattr(torch_dist_module, "init_process_group", None)
    if not callable(init_process_group):
        return {
            "callable": False,
            "signature": None,
            "has_device_id": False,
            "parameter_names": [],
            "skip_reason": "init_process_group_missing",
        }

    try:
        signature = inspect.signature(init_process_group)
    except (TypeError, ValueError):
        return {
            "callable": True,
            "signature": None,
            "has_device_id": False,
            "parameter_names": [],
            "skip_reason": "init_process_group_signature_unavailable",
        }

    parameter_names = list(signature.parameters.keys())
    has_device_id = "device_id" in signature.parameters
    return {
        "callable": True,
        "signature": str(signature),
        "has_device_id": has_device_id,
        "parameter_names": parameter_names,
        "skip_reason": None if has_device_id else "init_process_group_missing_device_id_parameter",
    }


def detect_patch_b2_trigger(
    *,
    candidate_path: Path | None = None,
    torch_dist_module: Any,
) -> dict[str, Any]:
    resolved_candidate_path = (
        candidate_path if candidate_path is not None else _resolve_patch_b2_trigger_candidate_path()
    )
    support = inspect_init_process_group_device_id_support(
        torch_dist_module=torch_dist_module
    )
    trigger = {
        "activate": False,
        "candidate_path": str(resolved_candidate_path),
        "signature": support["signature"],
        "has_device_id": bool(support["has_device_id"]),
        "parameter_names": support["parameter_names"],
        "device_unknown_tokens": [],
        "failure_tokens": [],
        "skip_reason": support["skip_reason"],
    }

    payload = _load_json_file(resolved_candidate_path)
    if payload is None:
        trigger["skip_reason"] = "candidate_missing_or_invalid"
        return trigger

    if bool(payload.get("pass")):
        trigger["skip_reason"] = "candidate_already_green"
        return trigger

    if str(payload.get("candidate_status", "")) != "blocked":
        trigger["skip_reason"] = "candidate_status_not_blocked"
        return trigger

    log_scan = payload.get("log_scan")
    if not isinstance(log_scan, dict):
        trigger["skip_reason"] = "candidate_missing_log_scan"
        return trigger

    extra_findings = log_scan.get("extra_findings")
    forbidden_tokens = log_scan.get("forbidden_tokens")
    if not isinstance(extra_findings, dict) or not isinstance(forbidden_tokens, dict):
        trigger["skip_reason"] = "candidate_missing_log_scan_sections"
        return trigger

    device_unknown_tokens = [
        token_name
        for token_name in ("device_unknown_rank0", "device_unknown_rank1")
        if bool(extra_findings.get(token_name, {}).get("present"))
    ]
    trigger["device_unknown_tokens"] = device_unknown_tokens
    if len(device_unknown_tokens) < 2:
        trigger["skip_reason"] = "candidate_missing_device_unknown_warning"
        return trigger

    failure_tokens = [
        token_name
        for token_name in ("illegal_memory_access", "ChildFailedError")
        if bool(forbidden_tokens.get(token_name, {}).get("present"))
    ]
    trigger["failure_tokens"] = failure_tokens
    if len(failure_tokens) < 2:
        trigger["skip_reason"] = "candidate_missing_nccl_failure_tokens"
        return trigger

    if not bool(support["has_device_id"]):
        return trigger

    trigger["activate"] = True
    trigger["skip_reason"] = None
    return trigger


def _resolve_patch_b2_device_id(*, torch_module: Any, current_device: int | None) -> tuple[Any, str]:
    if current_device is not None:
        return torch_module.device(f"cuda:{int(current_device)}"), "current_device"

    local_rank = _env_int("LOCAL_RANK", default=0)
    return torch_module.device(f"cuda:{local_rank}"), "local_rank_fallback"


def maybe_install_patch_b2(
    *,
    torch_module: Any,
    torch_dist_module: Any,
    current_device: int | None,
    candidate_path: Path | None = None,
) -> dict[str, Any]:
    trigger = detect_patch_b2_trigger(
        candidate_path=candidate_path,
        torch_dist_module=torch_dist_module,
    )
    trigger["patched"] = False
    trigger["injected_device_id"] = None
    trigger["device_id_source"] = None

    if not bool(trigger["activate"]):
        return trigger

    original_init_process_group = getattr(torch_dist_module, "init_process_group", None)
    if not callable(original_init_process_group):
        trigger["activate"] = False
        trigger["skip_reason"] = "init_process_group_missing"
        return trigger

    if bool(getattr(original_init_process_group, "_repo_local_b2_device_id_patch", False)):
        trigger["patched"] = True
        trigger["injected_device_id"] = str(
            getattr(original_init_process_group, "_repo_local_b2_injected_device_id", None)
        )
        trigger["device_id_source"] = str(
            getattr(original_init_process_group, "_repo_local_b2_device_id_source", "unknown")
        )
        return trigger

    injected_device_id, device_id_source = _resolve_patch_b2_device_id(
        torch_module=torch_module,
        current_device=current_device,
    )

    @functools.wraps(original_init_process_group)
    def repo_local_init_process_group(*args: Any, **kwargs: Any) -> Any:
        patched_kwargs = dict(kwargs)
        if (
            _env_int("WORLD_SIZE", default=0) > 1
            and patched_kwargs.get("device_id") is None
        ):
            patched_kwargs["device_id"] = injected_device_id
            print(
                f"[INFO] repo_local_patch_b2_injected_device_id={patched_kwargs['device_id']}"
            )
        return original_init_process_group(*args, **patched_kwargs)

    setattr(repo_local_init_process_group, "_repo_local_b2_device_id_patch", True)
    setattr(
        repo_local_init_process_group,
        "_repo_local_b2_injected_device_id",
        injected_device_id,
    )
    setattr(
        repo_local_init_process_group,
        "_repo_local_b2_device_id_source",
        device_id_source,
    )
    torch_dist_module.init_process_group = repo_local_init_process_group
    trigger["patched"] = True
    trigger["injected_device_id"] = str(injected_device_id)
    trigger["device_id_source"] = device_id_source
    return trigger


def _build_named_tensor_entry(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": str(name),
        "shape": _safe_tensor_shape(tensor),
        "device": str(getattr(tensor, "device", "unknown")),
        "dtype": str(getattr(tensor, "dtype", "unknown")),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        "is_meta": bool(getattr(tensor, "is_meta", False)),
        "numel": _safe_tensor_numel(tensor),
    }


def _summarize_named_tensors(named_tensors: Any) -> dict[str, Any]:
    preview: list[dict[str, Any]] = []
    distinct_devices: set[str] = set()
    empty_names: list[str] = []
    meta_names: list[str] = []
    tensor_count = 0
    total_numel = 0
    trainable_numel = 0
    trainable_tensor_count = 0

    for name, tensor in named_tensors:
        tensor_count += 1
        entry = _build_named_tensor_entry(name, tensor)
        total_numel += int(entry["numel"])
        if bool(entry["requires_grad"]):
            trainable_numel += int(entry["numel"])
            trainable_tensor_count += 1
        distinct_devices.add(str(entry["device"]))
        if int(entry["numel"]) == 0:
            empty_names.append(str(entry["name"]))
        if bool(entry["is_meta"]):
            meta_names.append(str(entry["name"]))
        if len(preview) < REPO_LOCAL_CENSUS_PREVIEW_LIMIT:
            preview.append(entry)

    return {
        "tensor_count": tensor_count,
        "total_numel": total_numel,
        "trainable_numel": trainable_numel,
        "trainable_tensor_count": trainable_tensor_count,
        "preview": preview,
        "distinct_devices": sorted(distinct_devices),
        "empty_names": empty_names,
        "meta_names": meta_names,
    }


def _summarize_former_parameters(model: Any) -> dict[str, Any]:
    former_parameters = getattr(model, "_former_parameters", None)
    summary: dict[str, Any] = {
        "present": former_parameters is not None,
        "type": None,
        "count": 0,
        "names_preview": [],
        "entries_preview": [],
    }
    if former_parameters is None:
        return summary

    summary["type"] = f"{type(former_parameters).__module__}.{type(former_parameters).__qualname__}"
    if hasattr(former_parameters, "items"):
        items = list(former_parameters.items())
        summary["count"] = len(items)
        summary["names_preview"] = [str(name) for name, _ in items[:REPO_LOCAL_CENSUS_PREVIEW_LIMIT]]
        summary["entries_preview"] = [
            _build_named_tensor_entry(str(name), tensor)
            for name, tensor in items[:REPO_LOCAL_CENSUS_PREVIEW_LIMIT]
        ]
        return summary

    summary["count"] = None
    summary["repr"] = repr(former_parameters)
    return summary


def _summarize_model_view(*, model: Any, view_name: str, torch_module: Any) -> dict[str, Any]:
    if model is None:
        return {
            "view_name": view_name,
            "present": False,
            "model_type": None,
            "has_module_attr": False,
            "torch_cuda_current_device": _safe_cuda_current_device(torch_module),
            "total_params": 0,
            "trainable_params": 0,
            "parameter_tensor_count": 0,
            "trainable_parameter_tensor_count": 0,
            "buffer_count": 0,
            "named_parameters_preview": [],
            "distinct_parameter_devices": [],
            "distinct_buffer_devices": [],
            "empty_parameter_names": [],
            "meta_parameter_names": [],
            "former_parameters": {
                "present": False,
                "type": None,
                "count": 0,
                "names_preview": [],
                "entries_preview": [],
            },
            "trainability_authority": None,
            "route_freeze": None,
            "optimizer_bucket_summary": None,
        }

    parameter_summary = _summarize_named_tensors(model.named_parameters())
    buffer_summary = _summarize_named_tensors(model.named_buffers())
    trainability_authority = _copy_mapping(
        getattr(model, "_repo_local_trainability_authority_summary", None)
    )
    return {
        "view_name": view_name,
        "present": True,
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "has_module_attr": hasattr(model, "module"),
        "torch_cuda_current_device": _safe_cuda_current_device(torch_module),
        "total_params": int(parameter_summary["total_numel"]),
        "trainable_params": int(parameter_summary["trainable_numel"]),
        "parameter_tensor_count": int(parameter_summary["tensor_count"]),
        "trainable_parameter_tensor_count": int(parameter_summary["trainable_tensor_count"]),
        "buffer_count": int(buffer_summary["tensor_count"]),
        "named_parameters_preview": parameter_summary["preview"],
        "distinct_parameter_devices": parameter_summary["distinct_devices"],
        "distinct_buffer_devices": buffer_summary["distinct_devices"],
        "empty_parameter_names": parameter_summary["empty_names"],
        "meta_parameter_names": parameter_summary["meta_names"],
        "former_parameters": _summarize_former_parameters(model),
        "trainability_authority": trainability_authority,
        "route_freeze": (
            None
            if trainability_authority is None
            else _copy_mapping(trainability_authority.get("route_freeze"))
        ),
        "optimizer_bucket_summary": _copy_mapping(
            getattr(model, "_repo_local_optimizer_bucket_summary", None)
        ),
    }


def _build_rank_census_payload(
    *,
    event: str,
    model_summary: dict[str, Any],
    metadata_dir: Path,
) -> dict[str, Any]:
    rank = _env_int("RANK", default=0)
    local_rank = _env_int("LOCAL_RANK", default=0)
    world_size = _env_int("WORLD_SIZE", default=0)
    return {
        "event": event,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "metadata_dir": str(metadata_dir),
        "total_params": int(model_summary["total_params"]),
        "trainable_params": int(model_summary["trainable_params"]),
        "buffer_count": int(model_summary["buffer_count"]),
        "named_parameters_preview": model_summary["named_parameters_preview"],
        "distinct_parameter_devices": model_summary["distinct_parameter_devices"],
        "distinct_buffer_devices": model_summary["distinct_buffer_devices"],
        "empty_parameter_names": model_summary["empty_parameter_names"],
        "meta_parameter_names": model_summary["meta_parameter_names"],
        "former_parameters": model_summary["former_parameters"],
        "torch_cuda_current_device": model_summary["torch_cuda_current_device"],
        "trainability_authority": model_summary["trainability_authority"],
        "route_freeze": model_summary["route_freeze"],
        "optimizer_bucket_summary": model_summary["optimizer_bucket_summary"],
    }


def _write_after_model_build_census(*, pipeline: Any) -> Path:
    metadata_dir = REPO_LOCAL_CENSUS_HOOK_STATE["metadata_dir"]
    torch_module = REPO_LOCAL_CENSUS_HOOK_STATE["torch_module"]
    if metadata_dir is None or torch_module is None:
        raise RuntimeError("repo-local census hook state is not initialized")

    model_summary = _summarize_model_view(
        model=getattr(pipeline, "model", None),
        view_name="pipeline_model",
        torch_module=torch_module,
    )
    payload = _build_rank_census_payload(
        event="after_model_build",
        model_summary=model_summary,
        metadata_dir=metadata_dir,
    )
    payload["pipeline_type"] = f"{type(pipeline).__module__}.{type(pipeline).__qualname__}"
    payload["observed_pipeline_attrs"] = {
        "model": hasattr(pipeline, "model"),
        "train_dataset": hasattr(pipeline, "train_dataset"),
        "eval_dataset": hasattr(pipeline, "eval_dataset"),
        "collator": hasattr(pipeline, "collator"),
        "data_collator": hasattr(pipeline, "data_collator"),
    }
    payload["model"] = model_summary
    path = metadata_dir / f"census_after_model_build_rank{payload['rank']}.json"
    return _write_json_atomic(path, payload)


def _write_before_first_forward_census(*, model: Any) -> Path:
    metadata_dir = REPO_LOCAL_CENSUS_HOOK_STATE["metadata_dir"]
    torch_module = REPO_LOCAL_CENSUS_HOOK_STATE["torch_module"]
    if metadata_dir is None or torch_module is None:
        raise RuntimeError("repo-local census hook state is not initialized")

    wrapper_summary = _summarize_model_view(
        model=model,
        view_name="wrapper_model",
        torch_module=torch_module,
    )
    module_model = getattr(model, "module", None)
    module_summary = None
    if module_model is not None:
        module_summary = _summarize_model_view(
            model=module_model,
            view_name="module_model",
            torch_module=torch_module,
        )

    payload = _build_rank_census_payload(
        event="before_first_forward",
        model_summary=wrapper_summary,
        metadata_dir=metadata_dir,
    )
    payload["wrapper_has_module_attr"] = hasattr(model, "module")
    payload["module_is_distinct_object"] = module_model is not None and module_model is not model
    payload["wrapper_model"] = wrapper_summary
    payload["module_model"] = module_summary
    path = metadata_dir / f"census_before_first_forward_rank{payload['rank']}.json"
    return _write_json_atomic(path, payload)


def install_repo_local_rank_census_hooks(*, config: Any, torch_module: Any) -> None:
    metadata_dir = _repo_local_metadata_dir(config)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    REPO_LOCAL_CENSUS_HOOK_STATE["metadata_dir"] = metadata_dir
    REPO_LOCAL_CENSUS_HOOK_STATE["torch_module"] = torch_module
    REPO_LOCAL_CENSUS_HOOK_STATE["pre_forward_written"] = False
    REPO_LOCAL_CENSUS_HOOK_STATE["first_backward_written"] = False
    REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_written"] = False
    REPO_LOCAL_CENSUS_HOOK_STATE["first_optimizer_step_pending_snapshot"] = None
    REPO_LOCAL_CENSUS_HOOK_STATE["stats_hash_batch_index"] = 0
    REPO_LOCAL_CENSUS_HOOK_STATE["stats_hash_last_payload"] = None
    REPO_LOCAL_CENSUS_HOOK_STATE["stats_hash_jsonl_path"] = None

    from gr00t.experiment.trainer import Gr00tTrainer  # pyright: ignore[reportMissingImports]
    from gr00t.model.gr00t_n1d6.setup import (  # pyright: ignore[reportMissingImports]
        Gr00tN1d6Pipeline,
    )

    if not bool(REPO_LOCAL_CENSUS_HOOK_STATE["pipeline_setup_patched"]):
        original_pipeline_setup = Gr00tN1d6Pipeline.setup
        REPO_LOCAL_CENSUS_HOOK_STATE["original_pipeline_setup"] = original_pipeline_setup

        @functools.wraps(original_pipeline_setup)
        def repo_local_pipeline_setup(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_pipeline_setup(self, *args, **kwargs)
            census_path = _write_after_model_build_census(pipeline=self)
            print(f"[INFO] repo_local_after_model_build_census_path={census_path}")
            return result

        Gr00tN1d6Pipeline.setup = repo_local_pipeline_setup
        REPO_LOCAL_CENSUS_HOOK_STATE["pipeline_setup_patched"] = True

    if not bool(REPO_LOCAL_CENSUS_HOOK_STATE["trainer_compute_loss_patched"]):
        original_trainer_compute_loss = Gr00tTrainer.compute_loss
        REPO_LOCAL_CENSUS_HOOK_STATE["original_trainer_compute_loss"] = original_trainer_compute_loss

        @functools.wraps(original_trainer_compute_loss)
        def repo_local_compute_loss(self: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
            _write_rank0_normalization_stats_hash_event(trainer=self, model=model)
            if not bool(REPO_LOCAL_CENSUS_HOOK_STATE["pre_forward_written"]):
                census_path = _write_before_first_forward_census(model=model)
                REPO_LOCAL_CENSUS_HOOK_STATE["pre_forward_written"] = True
                print(f"[INFO] repo_local_before_first_forward_census_path={census_path}")
                torch_cuda_memory_path = _write_rank0_torch_cuda_memory_snapshot(
                    torch_module=torch_module
                )
                if torch_cuda_memory_path is not None:
                    print(
                        "[INFO] repo_local_torch_cuda_memory_rank0_path="
                        f"{torch_cuda_memory_path}"
                    )
                nvidia_smi_sampling_path = _write_rank0_nvidia_smi_active_sampling()
                if nvidia_smi_sampling_path is not None:
                    print(
                        "[INFO] repo_local_nvidia_smi_active_sampling_rank0_path="
                        f"{nvidia_smi_sampling_path}"
                    )
            return original_trainer_compute_loss(self, model, *args, **kwargs)

        Gr00tTrainer.compute_loss = repo_local_compute_loss
        REPO_LOCAL_CENSUS_HOOK_STATE["trainer_compute_loss_patched"] = True

    if not bool(REPO_LOCAL_CENSUS_HOOK_STATE["trainer_training_step_patched"]):
        original_trainer_training_step = Gr00tTrainer.training_step
        REPO_LOCAL_CENSUS_HOOK_STATE["original_trainer_training_step"] = (
            original_trainer_training_step
        )

        @functools.wraps(original_trainer_training_step)
        def repo_local_training_step(
            self: Any,
            model: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            result = original_trainer_training_step(self, model, *args, **kwargs)
            if not bool(REPO_LOCAL_CENSUS_HOOK_STATE["first_backward_written"]):
                grad_probe_path = _write_rank0_first_backward_grad_probe(
                    trainer=self,
                    model=model,
                )
                REPO_LOCAL_CENSUS_HOOK_STATE["first_backward_written"] = True
                if grad_probe_path is not None:
                    print(
                        "[INFO] repo_local_first_backward_grad_probe_path="
                        f"{grad_probe_path}"
                    )
            return result

        Gr00tTrainer.training_step = repo_local_training_step
        REPO_LOCAL_CENSUS_HOOK_STATE["trainer_training_step_patched"] = True

    if not bool(REPO_LOCAL_CENSUS_HOOK_STATE["trainer_create_optimizer_patched"]):
        original_trainer_create_optimizer = Gr00tTrainer.create_optimizer
        REPO_LOCAL_CENSUS_HOOK_STATE["original_trainer_create_optimizer"] = (
            original_trainer_create_optimizer
        )

        @functools.wraps(original_trainer_create_optimizer)
        def repo_local_create_optimizer(self: Any, *args: Any, **kwargs: Any) -> Any:
            optimizer = original_trainer_create_optimizer(self, *args, **kwargs)
            return _maybe_wrap_repo_local_optimizer_first_step_probe(
                trainer=self,
                optimizer=optimizer,
            )

        Gr00tTrainer.create_optimizer = repo_local_create_optimizer
        REPO_LOCAL_CENSUS_HOOK_STATE["trainer_create_optimizer_patched"] = True

    if not bool(REPO_LOCAL_CENSUS_HOOK_STATE["trainer_train_patched"]):
        original_trainer_train = Gr00tTrainer.train
        REPO_LOCAL_CENSUS_HOOK_STATE["original_trainer_train"] = original_trainer_train

        @functools.wraps(original_trainer_train)
        def repo_local_train(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return original_trainer_train(self, *args, **kwargs)
            finally:
                first_step_probe_path = (
                    _maybe_finalize_repo_local_first_optimizer_step_probe_at_train_end(
                        trainer=self
                    )
                )
                if first_step_probe_path is not None:
                    print(
                        "[INFO] repo_local_first_optimizer_step_param_delta_probe_path="
                        f"{first_step_probe_path}"
                    )
                stats_hash_summary_path = _write_rank0_normalization_stats_hash_summary()
                if stats_hash_summary_path is not None:
                    print(
                        "[INFO] repo_local_normalization_stats_hash_summary_path="
                        f"{stats_hash_summary_path}"
                    )
                optimizer_state_path = _write_rank0_optimizer_state_snapshot(trainer=self)
                if optimizer_state_path is not None:
                    print(
                        "[INFO] repo_local_optimizer_state_rank0_path="
                        f"{optimizer_state_path}"
                    )

        Gr00tTrainer.train = repo_local_train
        REPO_LOCAL_CENSUS_HOOK_STATE["trainer_train_patched"] = True


def _load_python_contract() -> dict[str, str]:
    try:
        from work.demo_utils.paths import load_stage3_training_python_contract

        contract = load_stage3_training_python_contract(REPO_ROOT)
    except Exception:
        current_python = str(sys.executable)
        return {
            "orchestrator_python": current_python,
            "delegate_runtime_python": current_python,
        }
    return {
        "orchestrator_python": str(contract["orchestrator_python"]),
        "delegate_runtime_python": str(contract["delegate_runtime_python"]),
    }


def _read_git_commit(repo_path: Path) -> str:
    if not repo_path.is_dir():
        return "missing"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def _write_version_surface(
    *,
    config: Any,
    torch_module: Any,
    current_device: int | None,
    requested_num_gpus: int,
) -> Path:
    output_dir = _compute_effective_output_dir(config)
    metadata_dir = _repo_local_metadata_dir(config)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    python_contract = _load_python_contract()
    trainability_authority = get_repo_local_trainability_authority(config=config)
    payload = {
        "repo_local_launcher": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "orchestrator_python": str(python_contract["orchestrator_python"]),
        "delegate_runtime_python": str(python_contract["delegate_runtime_python"]),
        "delegate_python_version": platform.python_version(),
        "runtime_python": str(sys.executable),
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "torch_cuda_version": str(getattr(getattr(torch_module, "version", None), "cuda", "unknown")),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        "rank_env": {
            "RANK": str(os.environ.get("RANK", "")),
            "LOCAL_RANK": str(os.environ.get("LOCAL_RANK", "")),
            "WORLD_SIZE": str(os.environ.get("WORLD_SIZE", "")),
        },
        "torch_cuda_current_device": current_device,
        "requested_num_gpus": int(requested_num_gpus),
        "isaac_gr00t_commit": _read_git_commit(ISAAC_GR00T_ROOT),
        "trainability_authority": trainability_authority,
        "route_freeze": (
            None
            if trainability_authority is None
            else trainability_authority.get("route_freeze")
        ),
    }
    version_surface_path = metadata_dir / "version_surface.json"
    return _write_json_atomic(version_surface_path, payload)


def _pre_bind_and_collect_runtime_surface(*, requested_num_gpus: int) -> int | None:
    if requested_num_gpus > 2:
        raise RuntimeError(
            f"repo-local DDP launcher only supports up to 2 GPUs, got num_gpus={requested_num_gpus}"
        )

    import torch

    world_size = _env_int("WORLD_SIZE", default=0)
    local_rank = _env_int("LOCAL_RANK", default=0)
    rank = _env_int("RANK", default=0)

    if requested_num_gpus > 1 and world_size <= 1:
        raise RuntimeError(
            "num_gpus > 1 requires a multi-process DDP launch with WORLD_SIZE > 1; "
            "refusing single-process multi-visible-card execution"
        )

    if world_size > 1:
        torch.cuda.set_device(local_rank)

    current_device: int | None = None
    try:
        current_device = int(torch.cuda.current_device())
    except Exception:
        current_device = None

    print(f"[INFO] repo_local_rank_env={rank}")
    print(f"[INFO] repo_local_local_rank_env={local_rank}")
    print(f"[INFO] repo_local_world_size_env={world_size}")
    print(f"[INFO] repo_local_cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"[INFO] repo_local_torch_cuda_current_device={current_device}")

    return current_device


def apply_finetune_overrides(*, config: Any, ft_config: Any) -> bool:
    embodiment_tag = ft_config.embodiment_tag.value

    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_ddp = resolve_repo_local_use_ddp(ft_config.num_gpus)
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    if not hasattr(config.training, "max_grad_norm"):
        config.training.max_grad_norm = 1.0
    config.training.wandb_project = "finetune-gr00t-n1d6"

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    extra_authority = _copy_mapping(
        getattr(ft_config, REPO_LOCAL_TRAINABILITY_AUTHORITY_FIELD, None)
    )
    if extra_authority is not None:
        attach_repo_local_trainability_authority(
            config=config,
            authority=extra_authority,
        )
        if _repo_local_should_disable_one_step_grad_clipping(
            authority=extra_authority,
            max_steps=ft_config.max_steps,
        ):
            config.training.max_grad_norm = 0.0
    return bool(config.training.use_ddp)


def build_runtime_config(ft_config: Any) -> tuple[Any, bool]:
    from gr00t.configs.base_config import (  # pyright: ignore[reportMissingImports]
        get_default_config,
    )

    embodiment_tag = ft_config.embodiment_tag.value
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    use_ddp = apply_finetune_overrides(config=config, ft_config=ft_config)
    return config, use_ddp


def main() -> int:
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    import torch
    import tyro

    from gr00t.configs.finetune_config import (  # pyright: ignore[reportMissingImports]
        FinetuneConfig,
    )
    from gr00t.experiment.experiment import run  # pyright: ignore[reportMissingImports]

    ft_config = tyro.cli(
        FinetuneConfig,
        description=(
            "Repo-local finetune launcher. Mirrors upstream launch_finetune.py, but "
            "auto-enables config.training.use_ddp when num_gpus > 1 so stage3 pre-T3b "
            "avoids the DeepSpeed branch without modifying submodules/."
        ),
    )

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    requested_num_gpus = int(ft_config.num_gpus)
    current_device = _pre_bind_and_collect_runtime_surface(
        requested_num_gpus=requested_num_gpus
    )
    patch_b1 = maybe_install_patch_b1(torch_module=torch)
    patch_b2 = maybe_install_patch_b2(
        torch_module=torch,
        torch_dist_module=torch.distributed,
        current_device=current_device,
    )
    config, use_ddp = build_runtime_config(ft_config)
    version_surface_path = _write_version_surface(
        config=config,
        torch_module=torch,
        current_device=current_device,
        requested_num_gpus=requested_num_gpus,
    )
    print(f"[INFO] repo_local_launcher={Path(__file__).resolve()}")
    print(f"[INFO] repo_local_num_gpus={requested_num_gpus}")
    print(f"[INFO] repo_local_use_ddp={bool(use_ddp)}")
    print(f"[INFO] repo_local_version_surface_path={version_surface_path}")
    print(f"[INFO] repo_local_patch_b1_candidate_path={patch_b1['candidate_path']}")
    print(f"[INFO] repo_local_patch_b1_activated={bool(patch_b1['activate'])}")
    print(f"[INFO] repo_local_patch_b1_skip_reason={patch_b1['skip_reason']}")
    print(
        "[INFO] repo_local_patch_b1_patched_classes="
        + ",".join(patch_b1["patched_classes"])
    )
    print(f"[INFO] repo_local_patch_b2_candidate_path={patch_b2['candidate_path']}")
    print(f"[INFO] repo_local_patch_b2_signature={patch_b2['signature']}")
    print(f"[INFO] repo_local_patch_b2_has_device_id={bool(patch_b2['has_device_id'])}")
    print(f"[INFO] repo_local_patch_b2_activated={bool(patch_b2['activate'])}")
    print(f"[INFO] repo_local_patch_b2_skip_reason={patch_b2['skip_reason']}")
    print(f"[INFO] repo_local_patch_b2_patched={bool(patch_b2['patched'])}")
    print(
        f"[INFO] repo_local_patch_b2_injected_device_id={patch_b2['injected_device_id']}"
    )
    print(
        f"[INFO] repo_local_patch_b2_device_id_source={patch_b2['device_id_source']}"
    )
    print(
        "[INFO] repo_local_use_ddp_reason="
        + ("num_gpus_gt_1" if bool(use_ddp) else "num_gpus_le_1")
    )
    install_repo_local_rank_census_hooks(config=config, torch_module=torch)
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
