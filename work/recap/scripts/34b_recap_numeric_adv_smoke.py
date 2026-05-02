#!/usr/bin/env python3
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import datetime as _dt
import hashlib
import json
import logging
import math
import os
import random
import shlex
import signal
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ISAAC_GR00T_ROOT = REPO_ROOT / "submodules" / "Isaac-GR00T"
if ISAAC_GR00T_ROOT.is_dir() and str(ISAAC_GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAAC_GR00T_ROOT))


from work.recap import launch_finetune_use_ddp as trainability_entrypoint
from work.recap.scope_experiment import build_v2_train_scope_shim_metadata
from work.recap.train_scope_audit import FULL_UPDATE_SCOPE_NAMES
from work.recap.train_scope_audit import add_scope_flag_argument
from work.recap.train_scope_audit import build_scope_summary
from work.recap.train_scope_audit import parse_scope_flag


DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1"
DEFAULT_RUNTIME_LOG_DIR = "agent/runtime_logs/34b_recap_numeric_adv_smoke"
DEFAULT_RUNTIME_LOG_PREFIX = "34b_recap_numeric_adv_smoke"
DEFAULT_PATCHED_OUT_ROOT = "agent/artifacts/hf_patches"
DEFAULT_UPSTREAM_PY_REL = ".envs/wbc/bin/python"
DEFAULT_MAX_STEPS = 10
DEFAULT_SAVE_STEPS = 10
DEFAULT_SAVE_TOTAL_LIMIT = 1
DEFAULT_GLOBAL_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_DATALOADER_NUM_WORKERS = 0
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_RECAP_TRAIN_SCOPE = "full_action"
DEFAULT_NUM_GPUS = 1
DEFAULT_SEED = 42
DEFAULT_USE_WANDB = False
DEFAULT_CONDITION_FOCUSED_CONTINUATION = False
DEFAULT_CONDITION_HOT_LR_SCALE = 3.0
DEFAULT_DIFFUSION_TRUNK_LR_SCALE = 0.0
DEFAULT_POSITIVE_OVERSAMPLE_FACTOR = 1
DEFAULT_POSITIVE_CURRICULUM = False
DEFAULT_NEGATIVE_RETAIN_PROBABILITY = 1.0
DEFAULT_POSITIVE_CURRICULUM_SEED = -1
DEFAULT_LATE_STAGE_POSITIVE_EMPHASIS = False
DEFAULT_LATE_STAGE_THRESHOLD = 0.8
LATE_STAGE_POSITIVE_RULE = "advantage_input>0_and_t_norm>=threshold"
DEFAULT_TUNE_LLM = False
DEFAULT_TUNE_VISUAL = False
DEFAULT_TUNE_PROJECTOR = False
DEFAULT_TUNE_DIFFUSION_MODEL = True
DEFAULT_TUNE_TOP_LLM_LAYERS = 0
DEFAULT_TUNE_VLLN = True
DEFAULT_BACKBONE_LORA_ENABLED = False
DEFAULT_WRITE_CONDITIONING_FUNCTIONAL_PROBE = False
DEFAULT_WRITE_PAIRED_ACTION_PROBE = False
DEFAULT_WRITE_LABEL_SEMANTICS_AUDIT = False
DEFAULT_WRITE_SHUFFLED_ADVANTAGE_NEGATIVE_CONTROL = False
DEFAULT_BALANCED_ADVANTAGE_BATCHES = False
DEFAULT_EMIT_OPTIMIZER_PARAM_GROUP_REPORT = False
DEFAULT_EMIT_IN_MEMORY_DELTA_REPORT = False
DEFAULT_EMIT_SAVED_CHECKPOINT_DELTA_REPORT = False
CONDITIONING_ROUTE_NUMERIC_ADVANTAGE = "numeric_advantage_v2"
CONDITIONING_ROUTE_TEXT_INDICATOR = "text_indicator_v1"
CONDITIONING_ROUTE_CHOICES = (
    CONDITIONING_ROUTE_NUMERIC_ADVANTAGE,
    CONDITIONING_ROUTE_TEXT_INDICATOR,
)
DEFAULT_CONDITIONING_ROUTE = CONDITIONING_ROUTE_NUMERIC_ADVANTAGE
DEFAULT_RUNTIME_INDICATOR_MODE = "positive"
DEFAULT_TEXT_INDICATOR_PROMPT_RAW_COLUMN = "recap_m2.prompt_raw"
DEFAULT_TEXT_INDICATOR_DROPOUT_P = 0.0
DEFAULT_TEXT_INDICATOR_STEP_TEXT_FALLBACK = True
DEFAULT_BYPASS_SCOPE_SUPERVISOR = False
NUMERIC_ADV_MODEL_CLASS = "work.recap.model.GR00TRecapModel"
TEXT_INDICATOR_MODEL_CLASS = "gr00t.model.gr00t_n1d6.modeling_gr00t_n1_6.GR00T_N1_6"
ADVANTAGE_WEIGHT_KEY = "action_head.advantage_embedding.weight"
ADVANTAGE_BIAS_KEY = "action_head.advantage_embedding.bias"
CONDITIONED_INITIAL_ADVANTAGE_SNAPSHOT_FILENAME = (
    "conditioned_initial_advantage_embedding_snapshot.json"
)
NUMERIC_ADV_BATCH_LOG_PREFIX = "numeric-adv batch advantage shape="
NUMERIC_ADV_BATCH_LOG_LIMIT = 3
NUMERIC_ADV_BATCH_PREVIEW_LIMIT = 8
TRAINABLE_PARAM_LOG_LIMIT = 24
OPTIMIZER_GROUP_SAMPLE_LIMIT = 6
CHECKPOINT_LOAD_REPORT_ENV = "GR00T_NUMERIC_ADV_SMOKE_LOAD_REPORT_JSON"
ALLOWED_MISMATCHED_KEY_PREFIXES = ("action_head.",)
TRAINER_STATE_FILENAME = "trainer_state.json"
CONDITION_HOT_PARAM_PREFIXES = (
    "action_head.advantage_embedding.",
    "action_head.vlln.",
)
DIFFUSION_TRUNK_PARAM_PREFIX = "action_head.model."
SMOKE_CONFIG_ALIGNMENT_KEYS = (
    "max_state_dim",
    "max_action_dim",
    "action_horizon",
    "apply_sincos_state_encoding",
    "use_relative_action",
)
_ARGS_OVERRIDE_UNSET = object()
_TASK8_ADVANTAGES_TESTED: tuple[float | None, ...] = (None, -1.0, 0.0, 0.6666667, 1.0)
_TASK8_ILLEGAL_EXTRAPOLATION_ADVANTAGES: tuple[float, ...] = (3.0, 10.0)
_TASK8_REQUIRED_ACTION_DELTA_LAYERS: tuple[str, ...] = (
    "raw_normalized_action_delta",
    "decoded_action_delta",
    "postprocessed_action_delta",
    "controller_input_delta",
)
_TASK8_DEFAULT_SUBSET_SAMPLE_COUNT = 16
_TASK8_PROBE_SAMPLE_SEED_OFFSET = 1000
_TASK8_NEGATIVE_CONTROL_SEED_OFFSET = 2000
_TASK8_STEP0_PROBE_FILENAME = "conditioning_functional_probe_step0.json"
_TASK8_STEP1_PROBE_FILENAME = "conditioning_functional_probe_step1.json"
_TASK8_STEP20_PROBE_FILENAME = "conditioning_functional_probe_step20.json"
_TASK8_PAIRED_STEP0_PROBE_FILENAME = "paired_action_probe_step0.json"
_TASK8_PAIRED_STEP20_PROBE_FILENAME = "paired_action_probe_step20.json"
_TASK8_PROBE_OBS_MANIFEST_FILENAME = "probe_obs_manifest.json"
_TASK8_LABEL_SEMANTICS_AUDIT_FILENAME = "label_semantics_audit.json"
_TASK8_PREFORMAL_GATE_DECISION_FILENAME = "preformal_gate_decision.json"
_TASK10_SCOPE_AUDIT_DYNAMIC_FILENAME = "full_update_scope_audit_dynamic.json"
OPTIMIZER_PARAM_GROUP_REPORT_FILENAME = "optimizer_param_group_report.json"
IN_MEMORY_DELTA_REPORT_FILENAME = "in_memory_delta_report.json"
SAVED_CHECKPOINT_RELOAD_DELTA_REPORT_FILENAME = (
    "saved_checkpoint_reload_delta_report.json"
)
_DELEGATE_DELTA_AUDIT_STATE: dict[str, Any] = {
    "first_step_snapshot": None,
    "first_step_written": False,
}
_TASK10_GATE_STATUS_PASS = "PASS"
_TASK10_GATE_STATUS_BLOCK = "BLOCK"
_TASK10_GATE_STATUS_SKIPPED = "SKIPPED"
_TASK10_ROUTING_ROUTE_P3 = "route_p3_formal_training"
_TASK10_ROUTING_ROUTE_P5_PROBE = "route_p5_probe"
_TASK10_ROUTING_ROUTE_P5_FORMAL = "route_p5_formal_10ep"
_TASK10_ROUTING_ROUTE_P6 = "route_p6_semantic_branch"
_TASK10_ROUTING_BLOCKED = "block_downstream"
_TASK11_P1_BEST_SCOPE_AUDIT_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
    "p1_one_step/full_update_scope_audit_dynamic.json"
)
_TASK11_P1_BEST_SCOPE_AUDIT_METADATA_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
    "p1_one_step/repo_local_metadata/full_update_scope_audit_dynamic.json"
)
_TASK11_PREFORMAL_GATE_SUMMARY_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
    "p2_5_label_semantics/preformal_gate_decision.json"
)


def _parse_bool_flag(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, got {raw!r}. Use true/false."
    )


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "work").is_dir() and (parent / "agent").is_dir():
            return parent
    return Path.cwd().resolve()


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolve_path_preserve_symlink(repo_root: Path, raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return Path(os.path.abspath(str(path)))


def _resolve_full_update_output_dir(repo_root: Path, raw: str) -> Path:
    from work.recap.finetune_full import resolve_full_update_authority_output_dir

    return resolve_full_update_authority_output_dir(
        repo_root,
        raw,
        require_v2_authority=True,
    )


def _resolve_label_semantics_output_dir(
    repo_root: Path,
    *,
    output_dir: Path,
    raw: str,
) -> Path:
    raw_value = str(raw).strip()
    if raw_value:
        return _resolve_path(repo_root, raw_value)
    return output_dir.parent / "p2_5_label_semantics"


def _default_delegate_python() -> str:
    current = str(sys.executable).strip()
    if current:
        return current
    return DEFAULT_UPSTREAM_PY_REL


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _load_json_if_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): value for key, value in payload.items()}


def _load_repo_local_runtime_metadata(
    output_dir: Path,
    *,
    filename: str,
) -> dict[str, Any] | None:
    metadata_dir = trainability_entrypoint.resolve_repo_local_metadata_dir_for_output_dir(
        output_dir
    )
    return _load_json_if_dict(metadata_dir / filename)


def _normalize_reason_list(raw: object) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _resolve_task11_best_scope_audit_candidates(
    repo_root: Path,
    raw: str,
) -> list[Path]:
    raw_value = str(raw).strip()
    if raw_value:
        return [_resolve_path(repo_root, raw_value)]
    return [
        (repo_root / _TASK11_P1_BEST_SCOPE_AUDIT_REL).resolve(),
        (repo_root / _TASK11_P1_BEST_SCOPE_AUDIT_METADATA_REL).resolve(),
    ]


def load_task11_best_scope_authority(
    repo_root: Path,
    *,
    best_scope_audit: str = "",
) -> dict[str, Any]:
    candidate_paths = _resolve_task11_best_scope_audit_candidates(
        repo_root,
        best_scope_audit,
    )
    for candidate_path in candidate_paths:
        payload = _load_json_if_dict(candidate_path)
        if payload is None:
            continue
        best_scope = str(payload.get("train_scope_effective", "")).strip()
        if not best_scope:
            raise ValueError(
                f"Task 11 best-scope authority missing train_scope_effective: {candidate_path}"
            )
        return {
            "path": str(candidate_path),
            "train_scope_effective": best_scope,
            "payload": payload,
        }
    raise FileNotFoundError(
        "Task 11 best-scope authority not found. Checked: "
        + ", ".join(str(path) for path in candidate_paths)
    )


def resolve_task11_gate_summary_path(repo_root: Path, raw: str = "") -> Path:
    raw_value = str(raw).strip()
    if raw_value:
        return _resolve_path(repo_root, raw_value)
    return (repo_root / _TASK11_PREFORMAL_GATE_SUMMARY_REL).resolve()


def load_task11_preformal_gate_summary(
    repo_root: Path,
    *,
    gate_summary: str = "",
) -> dict[str, Any]:
    gate_path = resolve_task11_gate_summary_path(repo_root, gate_summary)
    payload = _load_json_if_dict(gate_path)
    if payload is None:
        raise FileNotFoundError(f"Task 11 gate summary not found or unreadable: {gate_path}")
    blocking_reasons = _normalize_reason_list(payload.get("blocking_reasons"))
    if "p3_formal_training_eligible" in payload:
        p3_formal_training_eligible = bool(payload.get("p3_formal_training_eligible"))
    else:
        p3_formal_training_eligible = bool(payload.get("formal_claim_allowed", False))
    p3_skip_reason = payload.get("p3_skip_reason")
    if p3_skip_reason is None and not p3_formal_training_eligible:
        p3_skip_reason = blocking_reasons[0] if blocking_reasons else "p3_formal_training_ineligible"
    return {
        **payload,
        "path": str(gate_path),
        "blocking_reasons": blocking_reasons,
        "p3_formal_training_eligible": bool(p3_formal_training_eligible),
        "p3_skip_reason": None if p3_formal_training_eligible else None if p3_skip_reason is None else str(p3_skip_reason),
    }


def resolve_task11_conditioned_warm_start_checkpoint(
    repo_root: Path,
    *,
    gate_summary_payload: Mapping[str, Any] | None,
    continuation_checkpoint_path: str = "",
) -> Path:
    explicit_checkpoint = str(continuation_checkpoint_path).strip()
    if explicit_checkpoint:
        return _resolve_path(repo_root, explicit_checkpoint)

    candidate_dirs: list[Path] = []
    if isinstance(gate_summary_payload, Mapping):
        conditioning_probe_path = str(
            gate_summary_payload.get("conditioning_probe_path", "")
        ).strip()
        if conditioning_probe_path:
            candidate_dirs.append(Path(conditioning_probe_path).expanduser().resolve().parent)
        gate_path = str(gate_summary_payload.get("path", "")).strip()
        if gate_path:
            candidate_dirs.append(Path(gate_path).expanduser().resolve().parent.parent / "p2_full_update_overfit20")

    for candidate_dir in candidate_dirs:
        latest_checkpoint = _latest_checkpoint(candidate_dir)
        if latest_checkpoint is not None:
            return latest_checkpoint
    raise FileNotFoundError(
        "Task 11 conditioned formal lane could not resolve a warm-start checkpoint from "
        "p2_full_update_overfit20"
    )


def _extract_probe_scope_metric_summary(
    probe_payload: Mapping[str, Any] | None,
    *,
    metric_key: str,
) -> dict[str, float] | None:
    if not isinstance(probe_payload, Mapping):
        return None
    scopes = probe_payload.get("scopes")
    if not isinstance(scopes, Mapping):
        return None
    summary: dict[str, float] = {}
    for scope_name, raw_scope_payload in scopes.items():
        if not isinstance(raw_scope_payload, Mapping):
            continue
        metric_value = raw_scope_payload.get(metric_key)
        if isinstance(metric_value, bool):
            continue
        if isinstance(metric_value, (int, float)):
            summary[str(scope_name)] = float(metric_value)
    return summary


def _extract_probe_trainer_global_step(
    probe_payload: Mapping[str, Any] | None,
) -> int | None:
    if not isinstance(probe_payload, Mapping):
        return None
    raw_value = probe_payload.get("trainer_global_step")
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int):
        return raw_value
    return None


def _tensor_snapshot_payload(tensor: Any, *, preview_limit: int = 8) -> dict[str, Any]:
    import torch

    detached = tensor.detach().float().cpu()
    flat = detached.reshape(-1)
    return {
        "shape": _shape_list(detached),
        "dtype": str(getattr(tensor, "dtype", detached.dtype)),
        "numel": int(flat.numel()),
        "min": float(flat.min().item()) if flat.numel() else None,
        "max": float(flat.max().item()) if flat.numel() else None,
        "mean": float(flat.mean().item()) if flat.numel() else None,
        "std": float(flat.std(unbiased=False).item()) if flat.numel() else None,
        "preview": [
            float(value)
            for value in flat[: int(preview_limit)].tolist()
        ],
    }


def _write_initial_advantage_embedding_snapshot(
    *,
    model: Any,
    output_dir: Path,
    continuation_checkpoint_path: Path | None,
    loading_info: Mapping[str, Any],
) -> Path:
    snapshot_path = output_dir / CONDITIONED_INITIAL_ADVANTAGE_SNAPSHOT_FILENAME
    advantage_embedding = getattr(getattr(model, "action_head", None), "advantage_embedding", None)
    if advantage_embedding is None:
        raise AttributeError("model.action_head.advantage_embedding is missing")
    weight = getattr(advantage_embedding, "weight", None)
    bias = getattr(advantage_embedding, "bias", None)
    if weight is None or bias is None:
        raise AttributeError(
            "model.action_head.advantage_embedding missing weight or bias"
        )
    missing_keys = {
        str(key) for key in loading_info.get("missing_keys", [])
    }
    payload = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "snapshot_kind": "conditioned_initial_advantage_embedding",
        "continuation_checkpoint_path": (
            None
            if continuation_checkpoint_path is None
            else str(continuation_checkpoint_path)
        ),
        "advantage_embedding_loaded_from_checkpoint": (
            ADVANTAGE_WEIGHT_KEY not in missing_keys
            and ADVANTAGE_BIAS_KEY not in missing_keys
        ),
        "missing_keys_at_load": sorted(missing_keys),
        "tensors": {
            ADVANTAGE_WEIGHT_KEY: _tensor_snapshot_payload(weight),
            ADVANTAGE_BIAS_KEY: _tensor_snapshot_payload(bias),
        },
    }
    _write_json(snapshot_path, payload)
    return snapshot_path


def _jsonify_cmd(cmd: list[str]) -> list[str]:
    return [str(part) for part in cmd]


def _effective_tuning_flags(args: argparse.Namespace) -> dict[str, Any]:
    return trainability_entrypoint.build_repo_local_effective_tuning_flags(
        requested_scope=args.recap_train_scope,
        tune_projector=bool(args.tune_projector),
        tune_diffusion_model=bool(args.tune_diffusion_model),
        tune_top_llm_layers=int(args.tune_top_llm_layers),
        tune_vlln=bool(args.tune_vlln),
        condition_focused_continuation=_condition_focused_continuation_enabled(args),
    )


def _training_scope_payload(
    args: argparse.Namespace, *, scope_summary: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "priority_order": [
            "full_action_head_with_diffusion_model_by_default",
            "backbone_lora_or_top_llm_only_if_explicitly_enabled",
            "freeze_visual_and_generic_backbone_by_default",
        ],
        "requested_scope": str(scope_summary["train_scope_requested"]),
        "scope_faithfulness": str(scope_summary["scope_faithfulness"]),
        "method_faithfulness": dict(scope_summary["method_faithfulness"]),
        "required_trainable_families": list(
            scope_summary["required_trainable_families"]
        ),
        "explicit_exclusions": list(scope_summary["explicit_exclusions"]),
        "action_head_trainable": True,
        "action_head_default_trainable_components": _trainable_scope_prefixes(
            args,
            scope_summary=scope_summary,
        ),
        "backbone_lora_enabled": bool(DEFAULT_BACKBONE_LORA_ENABLED),
        "condition_focused_continuation": _condition_focused_continuation_enabled(args),
        "condition_hot_lr_scale": float(args.condition_hot_lr_scale),
        "diffusion_trunk_lr_scale": float(args.diffusion_trunk_lr_scale),
        "positive_curriculum_enabled": bool(args.positive_curriculum),
        "negative_retain_probability": float(args.negative_retain_probability),
        "positive_curriculum_seed": int(_resolve_positive_curriculum_seed(args)),
        "late_stage_positive_enabled": bool(args.late_stage_positive_emphasis),
        "late_stage_threshold": float(args.late_stage_threshold),
        "late_stage_rule": "advantage_input>0_and_t_norm>=threshold",
        "positive_oversample_factor": int(args.positive_oversample_factor),
        **_effective_tuning_flags(args),
    }


def _build_trainability_authority_from_args(
    args: argparse.Namespace,
    *,
    scope_summary: Mapping[str, Any],
) -> dict[str, Any]:
    from work.recap import policy as recap_policy

    route = (
        recap_policy.MAINLINE_RUNTIME_ROUTE
        if _text_indicator_route_enabled(args)
        else recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE
    )
    indicator_mode = (
        str(getattr(args, "runtime_indicator_mode", DEFAULT_RUNTIME_INDICATOR_MODE))
        if _text_indicator_route_enabled(args)
        else None
    )
    return trainability_entrypoint.build_repo_local_trainability_authority(
        requested_scope=args.recap_train_scope,
        scope_summary=scope_summary,
        condition_focused_continuation=_condition_focused_continuation_enabled(args),
        condition_hot_lr_scale=float(args.condition_hot_lr_scale),
        diffusion_trunk_lr_scale=float(args.diffusion_trunk_lr_scale),
        route=route,
        indicator_mode=indicator_mode,
    )


def _emit_summary(payload: dict[str, Any], *, summary_json_path: Path | None) -> None:
    if summary_json_path is not None:
        _write_json(summary_json_path, payload)
    json.dump(payload, sys.stdout, ensure_ascii=True, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _emit_info_line(message: str) -> None:
    print(f"[INFO] {message}")


def _shape_list(value: Any) -> list[int]:
    shape = getattr(value, "shape", ())
    return [int(dim) for dim in tuple(shape)]


def _trainable_scope_prefixes(
    args: argparse.Namespace, *, scope_summary: Mapping[str, Any]
) -> list[str]:
    requested_scope = str(scope_summary["train_scope_requested"])
    if requested_scope == "full_action":
        return ["action_head.*"]
    if requested_scope == "full_policy":
        return ["action_head.*", "projector.*", "vla_action_interface.*"]
    if requested_scope == "strict_full":
        return ["model.named_parameters()"]
    prefixes = ["action_head.advantage_embedding"]
    if bool(args.tune_diffusion_model):
        prefixes.append("action_head.model")
    if bool(args.tune_vlln) or _condition_focused_continuation_enabled(args):
        prefixes.append("action_head.vlln")
    if bool(args.tune_projector):
        prefixes.append("action_head.projector")
    return prefixes


def _log_trainable_action_head_parameters(model: Any) -> None:
    scope_totals: dict[str, dict[str, int]] = {
        "action_head.advantage_embedding": {"tensors": 0, "numel": 0},
        "action_head.model": {"tensors": 0, "numel": 0},
        "action_head.vlln": {"tensors": 0, "numel": 0},
    }
    diffusion_logged = 0

    for name, param in model.named_parameters():
        if not bool(getattr(param, "requires_grad", False)):
            continue
        numel = int(param.numel())
        for scope in tuple(scope_totals.keys()):
            if name.startswith(scope + "."):
                scope_totals[scope]["tensors"] += 1
                scope_totals[scope]["numel"] += numel
                break
        if (
            name.startswith("action_head.model.")
            and diffusion_logged < TRAINABLE_PARAM_LOG_LIMIT
        ):
            print(
                "[INFO] numeric-adv smoke trainable_param="
                f"{name} shape={_shape_list(param)} numel={numel}",
            )
            diffusion_logged += 1

    for scope, totals in scope_totals.items():
        print(
            "[INFO] numeric-adv smoke trainable_prefix="
            f"{scope} tensors={int(totals['tensors'])} numel={int(totals['numel'])}",
        )


def _condition_focused_continuation_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "condition_focused_continuation", False))


def _conditioning_route(args: argparse.Namespace) -> str:
    route = str(
        getattr(args, "conditioning_route", DEFAULT_CONDITIONING_ROUTE)
        or DEFAULT_CONDITIONING_ROUTE
    ).strip()
    if route not in CONDITIONING_ROUTE_CHOICES:
        raise ValueError(
            f"--conditioning-route must be one of {list(CONDITIONING_ROUTE_CHOICES)}, got {route!r}"
        )
    return route


def _text_indicator_route_enabled(args: argparse.Namespace) -> bool:
    return _conditioning_route(args) == CONDITIONING_ROUTE_TEXT_INDICATOR


def _model_class_for_route(args: argparse.Namespace) -> str:
    return (
        TEXT_INDICATOR_MODEL_CLASS
        if _text_indicator_route_enabled(args)
        else NUMERIC_ADV_MODEL_CLASS
    )


def _hf_snapshot_overrides_for_route(args: argparse.Namespace) -> dict[str, Any]:
    if _text_indicator_route_enabled(args):
        return {"formalize_language": False}
    return {}


def _scope_summary_for_args(args: argparse.Namespace) -> dict[str, Any]:
    scope_summary = build_scope_summary(
        args.recap_train_scope,
        legacy_scope_bridge=build_v2_train_scope_shim_metadata(args.recap_train_scope),
    )
    if _text_indicator_route_enabled(args):
        scope_summary["method_faithfulness"] = {
            "recap_method_contract": "binary_text_indicator_v1",
            "paper_equivalent": False,
            "paper_method_gap": [
                "distributional_critic_not_yet_authoritative_for_this_gr00t_dataset",
                "runtime_conditioned_inference_not_yet_g3_validated",
                "cfg_policy_extraction_not_yet_validated",
            ],
            "paper_aligned_components": [
                "binarized_advantage_indicator",
                "indicator_placement_after_language_before_action",
                "advantage_condition_dropout_configurable",
            ],
        }
    return scope_summary


def _resolve_positive_curriculum_seed(args: argparse.Namespace) -> int:
    raw_seed = int(
        getattr(args, "positive_curriculum_seed", DEFAULT_POSITIVE_CURRICULUM_SEED)
    )
    if raw_seed >= 0:
        return raw_seed
    return int(args.seed)


def _resolve_optional_checkpoint_path(
    repo_root: Path, args: argparse.Namespace
) -> Path | None:
    raw = str(getattr(args, "continuation_checkpoint_path", "")).strip()
    if not raw:
        return None
    return _resolve_path(repo_root, raw)


def _checkpoint_resume_step(checkpoint_path: Path) -> int:
    trainer_state_path = checkpoint_path / TRAINER_STATE_FILENAME
    if trainer_state_path.is_file():
        payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        raw_step = payload.get("global_step")
        if isinstance(raw_step, int):
            return int(raw_step)
    suffix = checkpoint_path.name.split("checkpoint-", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return 0


def _continuation_requested_steps(args: argparse.Namespace) -> int:
    return int(args.max_steps)


def _load_state_dict_file(path: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    if path.suffix == ".safetensors":
        return dict(load_file(str(path), device="cpu"))
    loaded = torch.load(
        str(path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(loaded, dict):
        raise TypeError(f"Checkpoint file did not load as a state_dict mapping: {path}")
    return dict(loaded)


def _checkpoint_load_report_path() -> Path | None:
    raw = str(os.environ.get(CHECKPOINT_LOAD_REPORT_ENV, "")).strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _write_checkpoint_load_report(payload: dict[str, Any]) -> None:
    report_path = _checkpoint_load_report_path()
    if report_path is None:
        return
    _write_json(report_path, payload)


def _load_checkpoint_model_config_payload(
    checkpoint_path: Path,
) -> tuple[Path | None, dict[str, Any]]:
    config_path = (
        checkpoint_path / "config.json"
        if checkpoint_path.is_dir()
        else checkpoint_path.parent / "config.json"
    )
    if not config_path.is_file():
        return None, {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(
            f"Checkpoint config.json did not load as a mapping: {config_path}"
        )
    return config_path, payload


def _align_smoke_model_config_from_checkpoint(
    model_config: Any,
    checkpoint_path: Path,
) -> dict[str, Any]:
    config_path, payload = _load_checkpoint_model_config_payload(checkpoint_path)
    aligned_fields: dict[str, Any] = {}
    for key in SMOKE_CONFIG_ALIGNMENT_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        setattr(model_config, key, value)
        aligned_fields[str(key)] = value
    return {
        "config_path": None if config_path is None else str(config_path),
        "aligned_fields": aligned_fields,
    }


def _add_bool_group(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    dest: str,
    default: bool,
    help_text: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        name,
        dest=dest,
        nargs="?",
        const=True,
        type=_parse_bool_flag,
        metavar="BOOL",
        help=f"{help_text} Accepts optional explicit BOOL=true/false for plan-command compatibility.",
    )
    group.add_argument(
        name.replace("--", "--no-", 1),
        dest=dest,
        action="store_false",
        help=f"Disable {help_text.lower()}",
    )
    parser.set_defaults(**{dest: bool(default)})


def _selected_checkpoint_asset(checkpoint_dir: Path | None) -> Path | None:
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return None
    candidates = [
        checkpoint_dir / "model.safetensors.index.json",
        checkpoint_dir / "model.safetensors",
        checkpoint_dir / "pytorch_model.bin.index.json",
        checkpoint_dir / "pytorch_model.bin",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _safetensors_key_names(asset_path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(asset_path), framework="pt", device="cpu") as f:
        return list(f.keys())


def _ensure_single_safetensors_index(asset_path: Path) -> Path:
    index_path = asset_path.with_name(asset_path.name + ".index.json")
    if index_path.is_file():
        return index_path
    weight_map = {key: asset_path.name for key in _safetensors_key_names(asset_path)}
    payload = {
        "metadata": {"total_size": int(asset_path.stat().st_size)},
        "weight_map": weight_map,
    }
    _write_json(index_path, payload)
    return index_path


def _latest_checkpoint(output_dir: Path) -> Path | None:
    latest: tuple[int, float, Path] | None = None
    for path in sorted(output_dir.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        suffix = path.name.split("checkpoint-", 1)[-1]
        if not suffix.isdigit():
            continue
        candidate = (int(suffix), float(path.stat().st_mtime), path)
        if latest is None or candidate[:2] > latest[:2]:
            latest = candidate
    return None if latest is None else latest[2]


def _checkpoint_advantage_embedding_status(
    checkpoint_dir: Path | None,
) -> tuple[bool, list[str], Path | None]:
    asset_path = _selected_checkpoint_asset(checkpoint_dir)
    if asset_path is None:
        return False, [ADVANTAGE_WEIGHT_KEY, ADVANTAGE_BIAS_KEY], None
    if asset_path.name.endswith(".index.json"):
        data = json.loads(asset_path.read_text(encoding="utf-8"))
        weight_map_raw = data.get("weight_map")
        weight_map = dict(weight_map_raw) if isinstance(weight_map_raw, dict) else {}
        missing = [
            key
            for key in (ADVANTAGE_WEIGHT_KEY, ADVANTAGE_BIAS_KEY)
            if key not in weight_map
        ]
        return len(missing) == 0, missing, asset_path
    if asset_path.suffix == ".safetensors":
        key_names = set(_safetensors_key_names(asset_path))
        missing = [
            key
            for key in (ADVANTAGE_WEIGHT_KEY, ADVANTAGE_BIAS_KEY)
            if key not in key_names
        ]
        return len(missing) == 0, missing, _ensure_single_safetensors_index(asset_path)
    return False, [ADVANTAGE_WEIGHT_KEY, ADVANTAGE_BIAS_KEY], asset_path


def _run_with_tee(
    cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stderr.write(line)
                log_fp.write(line)
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait()
            return 130
    return int(proc.returncode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="34b_recap_numeric_adv_smoke.py",
        description=(
            "Smoke-only numeric-adv finetune wrapper. It patches the active upstream "
            "Gr00tN1d6Pipeline so training instantiates work.recap.model.GR00TRecapModel, "
            "then runs a tiny finetune to verify the resulting checkpoint index contains "
            "action_head.advantage_embedding.* keys while defaulting to a full action-head "
            "training scope that includes the diffusion model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="LeRobot dataset directory containing recap_m2.advantage_input.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Checkpoint output directory for the tiny smoke finetune.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional path to persist wrapper summary JSON. Always printed to stdout.",
    )
    parser.add_argument(
        "--runtime-log-dir",
        type=str,
        default=DEFAULT_RUNTIME_LOG_DIR,
        help="Directory for delegate runtime logs.",
    )
    parser.add_argument(
        "--runtime-log-prefix",
        type=str,
        default=DEFAULT_RUNTIME_LOG_PREFIX,
        help="Filename prefix for the delegate runtime log.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="",
        help=(
            "Python executable used for the delegated training process. "
            "Defaults to the current wrapper interpreter when omitted."
        ),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="HF repo_id used as the base checkpoint before local patching.",
    )
    parser.add_argument(
        "--base-model-revision",
        type=str,
        default="",
        help="Optional HF snapshot revision for the base model.",
    )
    parser.add_argument(
        "--hf-hub-cache-dir",
        type=str,
        default="",
        help="Optional HF hub cache root used while patching the base model directory.",
    )
    parser.add_argument(
        "--patched-out-root",
        type=str,
        default=DEFAULT_PATCHED_OUT_ROOT,
        help="Repo-relative output root for patched local HF model directories.",
    )
    _add_bool_group(
        parser,
        name="--force-top-llm-layers-zero",
        dest="force_top_llm_layers_zero",
        default=True,
        help_text=(
            "Preserve the legacy patched-HF config behavior that forces tune_top_llm_layers=0. "
            "Disable only for explicit full-scope main RECAP experiments."
        ),
    )
    parser.add_argument(
        "--conditioning-route",
        type=str,
        choices=CONDITIONING_ROUTE_CHOICES,
        default=DEFAULT_CONDITIONING_ROUTE,
        help=(
            "Training conditioning surface. numeric_advantage_v2 preserves the legacy "
            "diagnostic scalar route; text_indicator_v1 uses carrier_text_v1-style "
            "Advantage: positive/negative text in the VLA prefix."
        ),
    )
    parser.add_argument(
        "--runtime-indicator-mode",
        type=str,
        default=DEFAULT_RUNTIME_INDICATOR_MODE,
        help="Runtime indicator mode frozen into route metadata for text_indicator_v1.",
    )
    parser.add_argument(
        "--indicator-dropout-p",
        type=float,
        default=float(DEFAULT_TEXT_INDICATOR_DROPOUT_P),
        help="Probability of dropping positive/negative text indicators to omit during text_indicator_v1 training.",
    )
    parser.add_argument(
        "--text-indicator-prompt-raw-column",
        type=str,
        default=DEFAULT_TEXT_INDICATOR_PROMPT_RAW_COLUMN,
        help=(
            "Optional parquet column for raw task text. If absent and step-text fallback is enabled, "
            "the upstream extracted task text is used."
        ),
    )
    _add_bool_group(
        parser,
        name="--text-indicator-step-text-fallback",
        dest="text_indicator_step_text_fallback",
        default=bool(DEFAULT_TEXT_INDICATOR_STEP_TEXT_FALLBACK),
        help_text="Allow text_indicator_v1 training to use the upstream extracted task text when prompt_raw column is absent.",
    )
    parser.add_argument(
        "--bypass-scope-supervisor",
        action="store_true",
        default=bool(DEFAULT_BYPASS_SCOPE_SUPERVISOR),
        help="Run full-update scopes directly instead of the legacy one-step downgrade supervisor.",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default=DEFAULT_EMBODIMENT_TAG,
        help="Embodiment tag forwarded into the upstream training config.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(DEFAULT_MAX_STEPS),
        help="Tiny smoke max steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=int(DEFAULT_SAVE_STEPS),
        help="Checkpoint cadence for the smoke run.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=int(DEFAULT_SAVE_TOTAL_LIMIT),
        help="Checkpoint retention budget. This smoke wrapper requires exactly one retained checkpoint.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=int(DEFAULT_GLOBAL_BATCH_SIZE),
        help="Upstream training global batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(DEFAULT_GRADIENT_ACCUMULATION_STEPS),
        help="Upstream training gradient accumulation steps.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=int(DEFAULT_DATALOADER_NUM_WORKERS),
        help="Upstream training dataloader workers.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(DEFAULT_LEARNING_RATE),
        help="Upstream training learning rate.",
    )
    add_scope_flag_argument(
        parser,
        default=DEFAULT_RECAP_TRAIN_SCOPE,
        help_text=(
            "Public full-update-first scope taxonomy. This wrapper only accepts the additive "
            "v2 values current_partial|full_action|full_policy|strict_full and keeps legacy "
            "S1/S2/S3 semantics unchanged via a reporting shim."
        ),
    )
    parser.add_argument(
        "--requested-scope",
        dest="recap_train_scope",
        type=parse_scope_flag,
        choices=FULL_UPDATE_SCOPE_NAMES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-downgrade",
        type=_parse_bool_flag,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--entrypoint",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bf16",
        type=_parse_bool_flag,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        type=_parse_bool_flag,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--best-scope-audit",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gate-summary",
        type=str,
        default=str(_TASK11_PREFORMAL_GATE_SUMMARY_REL),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--require-p3-formal-eligible",
        type=_parse_bool_flag,
        default=False,
        help=argparse.SUPPRESS,
    )
    _add_bool_group(
        parser,
        name="--balanced-advantage-batches",
        dest="balanced_advantage_batches",
        default=bool(DEFAULT_BALANCED_ADVANTAGE_BATCHES),
        help_text=(
            "Freeze train/heldout diagnostic subsets with a positive/non-positive balance when the dataset permits it. "
            "This only affects Task 8 probe sample selection, not the training dataloader."
        ),
    )
    _add_bool_group(
        parser,
        name="--write-conditioning-functional-probe",
        dest="write_conditioning_functional_probe",
        default=bool(DEFAULT_WRITE_CONDITIONING_FUNCTIONAL_PROBE),
        help_text=(
            "Write Task 8 conditioning_functional_probe_step*.json artifacts from the local post-training reducer."
        ),
    )
    _add_bool_group(
        parser,
        name="--write-paired-action-probe",
        dest="write_paired_action_probe",
        default=bool(DEFAULT_WRITE_PAIRED_ACTION_PROBE),
        help_text=(
            "Write Task 8 paired_action_probe_step*.json artifacts from local checkpoint inference seams."
        ),
    )
    _add_bool_group(
        parser,
        name="--write-label-semantics-audit",
        dest="write_label_semantics_audit",
        default=bool(DEFAULT_WRITE_LABEL_SEMANTICS_AUDIT),
        help_text=(
            "Write Task 8 label_semantics_audit.json and preformal_gate_decision.json artifacts."
        ),
    )
    _add_bool_group(
        parser,
        name="--write-shuffled-advantage-negative-control",
        dest="write_shuffled_advantage_negative_control",
        default=bool(DEFAULT_WRITE_SHUFFLED_ADVANTAGE_NEGATIVE_CONTROL),
        help_text=(
            "Enable the Task 8 shuffled-advantage negative control inside label semantics audit generation."
        ),
    )
    parser.add_argument(
        "--emit-optimizer-param-group-report",
        action="store_true",
        default=bool(DEFAULT_EMIT_OPTIMIZER_PARAM_GROUP_REPORT),
        help="Write optimizer_param_group_report.json under output_dir/repo_local_metadata after live optimizer creation.",
    )
    parser.add_argument(
        "--emit-in-memory-delta-report",
        action="store_true",
        default=bool(DEFAULT_EMIT_IN_MEMORY_DELTA_REPORT),
        help="Write in_memory_delta_report.json under output_dir/repo_local_metadata after the first optimizer step.",
    )
    parser.add_argument(
        "--emit-saved-checkpoint-delta-report",
        action="store_true",
        default=bool(DEFAULT_EMIT_SAVED_CHECKPOINT_DELTA_REPORT),
        help="Write saved_checkpoint_reload_delta_report.json under output_dir/repo_local_metadata after reloading the latest saved checkpoint.",
    )
    parser.add_argument(
        "--label-semantics-output-dir",
        type=str,
        default="",
        help=(
            "Optional output directory for Task 8 label semantics artifacts. "
            "Defaults to a sibling p2_5_label_semantics directory next to --output-dir."
        ),
    )
    parser.add_argument(
        "--positive-oversample-factor",
        type=int,
        default=int(DEFAULT_POSITIVE_OVERSAMPLE_FACTOR),
        help=(
            "Local dataset overlay duplication factor for positive-advantage training units. "
            "Use 1 to disable; values >1 duplicate units whose underlying advantage_input contains values > 0."
        ),
    )
    _add_bool_group(
        parser,
        name="--positive-curriculum",
        dest="positive_curriculum",
        default=bool(DEFAULT_POSITIVE_CURRICULUM),
        help_text=(
            "Enable a positive-heavy dataset curriculum that retains all positive-advantage steps "
            "while probabilistically downsampling non-positive steps before any positive oversampling."
        ),
    )
    parser.add_argument(
        "--negative-retain-probability",
        type=float,
        default=float(DEFAULT_NEGATIVE_RETAIN_PROBABILITY),
        help=(
            "Retention probability for non-positive steps when --positive-curriculum is enabled. "
            "Use 1.0 to keep all negatives, 0.0 to drop them all."
        ),
    )
    parser.add_argument(
        "--positive-curriculum-seed",
        type=int,
        default=int(DEFAULT_POSITIVE_CURRICULUM_SEED),
        help=(
            "Seed for positive curriculum negative-step retention. "
            "Use -1 to reuse --seed."
        ),
    )
    _add_bool_group(
        parser,
        name="--late-stage-positive-emphasis",
        dest="late_stage_positive_emphasis",
        default=bool(DEFAULT_LATE_STAGE_POSITIVE_EMPHASIS),
        help_text=(
            "Restrict positive oversampling/curriculum emphasis to positive steps that also satisfy "
            "the late-stage proxy t_norm >= --late-stage-threshold."
        ),
    )
    parser.add_argument(
        "--late-stage-threshold",
        type=float,
        default=float(DEFAULT_LATE_STAGE_THRESHOLD),
        help=(
            "Late-stage proxy threshold applied as t_norm >= threshold when "
            "--late-stage-positive-emphasis is enabled."
        ),
    )
    _add_bool_group(
        parser,
        name="--condition-focused-continuation",
        dest="condition_focused_continuation",
        default=bool(DEFAULT_CONDITION_FOCUSED_CONTINUATION),
        help_text=(
            "Warm-start a fresh run from a local checkpoint directory while rebalancing optimizer groups so "
            "conditioning parameters stay hot and the diffusion trunk is frozen or downweighted."
        ),
    )
    parser.add_argument(
        "--continuation-checkpoint-path",
        type=str,
        default="",
        help=(
            "Local checkpoint directory whose model weights are used as warm-start input when "
            "--condition-focused-continuation is enabled "
            "(for example a checkpoint-500 directory)."
        ),
    )
    parser.add_argument(
        "--condition-hot-lr-scale",
        type=float,
        default=float(DEFAULT_CONDITION_HOT_LR_SCALE),
        help=(
            "Per-group LR multiplier applied to conditioning parameters during condition-focused continuation "
            "(action_head.advantage_embedding.* and action_head.vlln.*)."
        ),
    )
    parser.add_argument(
        "--diffusion-trunk-lr-scale",
        type=float,
        default=float(DEFAULT_DIFFUSION_TRUNK_LR_SCALE),
        help=(
            "Per-group LR multiplier for action_head.model.* during condition-focused continuation. "
            "Use 0 to freeze the diffusion trunk."
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=int(DEFAULT_NUM_GPUS),
        help="Number of GPUs for the smoke run. Keep this at 1 for local monkeypatching.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(DEFAULT_SEED),
        help="Training seed.",
    )
    _add_bool_group(
        parser,
        name="--tune-projector",
        dest="tune_projector",
        default=bool(DEFAULT_TUNE_PROJECTOR),
        help_text="Tune the projector module during finetuning.",
    )
    _add_bool_group(
        parser,
        name="--tune-diffusion-model",
        dest="tune_diffusion_model",
        default=bool(DEFAULT_TUNE_DIFFUSION_MODEL),
        help_text="Tune the diffusion model during finetuning.",
    )
    parser.add_argument(
        "--tune-top-llm-layers",
        type=int,
        default=int(DEFAULT_TUNE_TOP_LLM_LAYERS),
        help=(
            "Number of top backbone LLM layers to adapt. The smoke default keeps this at 0 "
            "so the default path stays inside the action head."
        ),
    )
    _add_bool_group(
        parser,
        name="--tune-vlln",
        dest="tune_vlln",
        default=bool(DEFAULT_TUNE_VLLN),
        help_text="Tune the action-head VLLN block during finetuning.",
    )
    _add_bool_group(
        parser,
        name="--use-wandb",
        dest="use_wandb",
        default=bool(DEFAULT_USE_WANDB),
        help_text="Enable Weights & Biases logging.",
    )
    _add_bool_group(
        parser,
        name="--transformers-local-files-only",
        dest="transformers_local_files_only",
        default=True,
        help_text="Restrict transformers loading to local files only.",
    )
    parser.add_argument(
        "--delegate-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    _ = _conditioning_route(args)
    if str(getattr(args, "entrypoint", "")).strip() not in {"", "conditioned"}:
        raise ValueError(
            "--entrypoint only accepts 'conditioned' for the Task 7 compatibility surface. "
            f"Got {args.entrypoint!r}."
        )
    if int(args.max_steps) <= 0:
        raise ValueError(f"--max-steps must be > 0, got {args.max_steps!r}")
    if int(args.save_steps) <= 0:
        raise ValueError(f"--save-steps must be > 0, got {args.save_steps!r}")
    if int(args.save_total_limit) != int(DEFAULT_SAVE_TOTAL_LIMIT):
        raise ValueError(
            "This wrapper enforces single-checkpoint retention. "
            f"Expected --save-total-limit={DEFAULT_SAVE_TOTAL_LIMIT}, got {args.save_total_limit}."
        )
    if int(args.num_gpus) != 1:
        raise ValueError(
            "This smoke wrapper only supports --num-gpus=1 so the monkeypatch stays in-process. "
            f"Got {args.num_gpus}."
        )
    if int(args.tune_top_llm_layers) < 0:
        raise ValueError(
            f"--tune-top-llm-layers must be >= 0, got {args.tune_top_llm_layers!r}"
        )
    if args.per_device_batch_size is not None and int(args.per_device_batch_size) <= 0:
        raise ValueError(
            "--per-device-batch-size must be > 0 when provided. "
            f"Got {args.per_device_batch_size!r}."
        )
    if float(args.condition_hot_lr_scale) <= 0.0:
        raise ValueError(
            "--condition-hot-lr-scale must be > 0 so conditioning parameters remain trainable. "
            f"Got {args.condition_hot_lr_scale!r}."
        )
    if int(args.positive_oversample_factor) < 1:
        raise ValueError(
            "--positive-oversample-factor must be >= 1. "
            f"Got {args.positive_oversample_factor!r}."
        )
    if not 0.0 <= float(args.negative_retain_probability) <= 1.0:
        raise ValueError(
            "--negative-retain-probability must be in [0, 1]. "
            f"Got {args.negative_retain_probability!r}."
        )
    if not 0.0 <= float(args.late_stage_threshold) <= 1.0:
        raise ValueError(
            "--late-stage-threshold must be in [0, 1]. "
            f"Got {args.late_stage_threshold!r}."
        )
    if float(args.diffusion_trunk_lr_scale) < 0.0:
        raise ValueError(
            "--diffusion-trunk-lr-scale must be >= 0. Use 0 to freeze the diffusion trunk. "
            f"Got {args.diffusion_trunk_lr_scale!r}."
        )
    if not 0.0 <= float(args.indicator_dropout_p) <= 1.0:
        raise ValueError(
            "--indicator-dropout-p must be in [0, 1]. "
            f"Got {args.indicator_dropout_p!r}."
        )
    if _text_indicator_route_enabled(args):
        from work.recap.text_indicator import normalize_indicator_mode

        normalize_indicator_mode(
            str(args.runtime_indicator_mode),
            field_name="runtime_indicator_mode",
        )
    if _condition_focused_continuation_enabled(args):
        repo_root = _repo_root()
        checkpoint_path = _resolve_optional_checkpoint_path(repo_root, args)
        if checkpoint_path is None:
            raise ValueError(
                "--condition-focused-continuation requires --continuation-checkpoint-path."
            )
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"Continuation checkpoint directory not found: {checkpoint_path}"
            )
        checkpoint_asset = _selected_checkpoint_asset(checkpoint_path)
        if checkpoint_asset is None:
            raise FileNotFoundError(
                f"Continuation checkpoint is missing model weights: {checkpoint_path}"
            )


def _build_delegate_cmd(
    *,
    script_path: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    output_dir: Path,
    delegate_python: Path,
) -> list[str]:
    cmd = [
        str(delegate_python),
        str(script_path),
        "--delegate-mode",
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--label-semantics-output-dir",
        str(args.label_semantics_output_dir),
        "--summary-json",
        str(args.summary_json),
        "--runtime-log-dir",
        str(args.runtime_log_dir),
        "--runtime-log-prefix",
        str(args.runtime_log_prefix),
        "--python",
        str(delegate_python),
        "--base-model",
        str(args.base_model),
        "--base-model-revision",
        str(args.base_model_revision),
        "--hf-hub-cache-dir",
        str(args.hf_hub_cache_dir),
        "--patched-out-root",
        str(args.patched_out_root),
        (
            "--force-top-llm-layers-zero"
            if bool(args.force_top_llm_layers_zero)
            else "--no-force-top-llm-layers-zero"
        ),
        "--conditioning-route",
        str(args.conditioning_route),
        "--runtime-indicator-mode",
        str(args.runtime_indicator_mode),
        "--indicator-dropout-p",
        str(float(args.indicator_dropout_p)),
        "--text-indicator-prompt-raw-column",
        str(args.text_indicator_prompt_raw_column),
        (
            "--text-indicator-step-text-fallback"
            if bool(args.text_indicator_step_text_fallback)
            else "--no-text-indicator-step-text-fallback"
        ),
        "--embodiment-tag",
        str(args.embodiment_tag),
        "--max-steps",
        str(int(args.max_steps)),
        "--save-steps",
        str(int(args.save_steps)),
        "--save-total-limit",
        str(int(args.save_total_limit)),
        "--global-batch-size",
        str(int(args.global_batch_size)),
        "--gradient-accumulation-steps",
        str(int(args.gradient_accumulation_steps)),
        "--dataloader-num-workers",
        str(int(args.dataloader_num_workers)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--recap-train-scope",
        str(args.recap_train_scope),
        (
            "--balanced-advantage-batches"
            if bool(args.balanced_advantage_batches)
            else "--no-balanced-advantage-batches"
        ),
        (
            "--write-conditioning-functional-probe"
            if bool(args.write_conditioning_functional_probe)
            else "--no-write-conditioning-functional-probe"
        ),
        (
            "--write-paired-action-probe"
            if bool(args.write_paired_action_probe)
            else "--no-write-paired-action-probe"
        ),
        (
            "--write-label-semantics-audit"
            if bool(args.write_label_semantics_audit)
            else "--no-write-label-semantics-audit"
        ),
        (
            "--write-shuffled-advantage-negative-control"
            if bool(args.write_shuffled_advantage_negative_control)
            else "--no-write-shuffled-advantage-negative-control"
        ),
        "--positive-oversample-factor",
        str(int(args.positive_oversample_factor)),
        "--positive-curriculum"
        if bool(args.positive_curriculum)
        else "--no-positive-curriculum",
        "--negative-retain-probability",
        str(float(args.negative_retain_probability)),
        "--positive-curriculum-seed",
        str(int(args.positive_curriculum_seed)),
        (
            "--late-stage-positive-emphasis"
            if bool(args.late_stage_positive_emphasis)
            else "--no-late-stage-positive-emphasis"
        ),
        "--late-stage-threshold",
        str(float(args.late_stage_threshold)),
        (
            "--condition-focused-continuation"
            if bool(args.condition_focused_continuation)
            else "--no-condition-focused-continuation"
        ),
        "--continuation-checkpoint-path",
        str(args.continuation_checkpoint_path),
        "--condition-hot-lr-scale",
        str(float(args.condition_hot_lr_scale)),
        "--diffusion-trunk-lr-scale",
        str(float(args.diffusion_trunk_lr_scale)),
        "--num-gpus",
        str(int(args.num_gpus)),
        "--seed",
        str(int(args.seed)),
        "--tune-top-llm-layers",
        str(int(args.tune_top_llm_layers)),
        "--tune-projector" if bool(args.tune_projector) else "--no-tune-projector",
        (
            "--tune-diffusion-model"
            if bool(args.tune_diffusion_model)
            else "--no-tune-diffusion-model"
        ),
        "--tune-vlln" if bool(args.tune_vlln) else "--no-tune-vlln",
        "--use-wandb" if bool(args.use_wandb) else "--no-use-wandb",
        (
            "--transformers-local-files-only"
            if bool(args.transformers_local_files_only)
            else "--no-transformers-local-files-only"
        ),
    ]
    if bool(args.emit_optimizer_param_group_report):
        cmd.append("--emit-optimizer-param-group-report")
    if bool(args.emit_in_memory_delta_report):
        cmd.append("--emit-in-memory-delta-report")
    if bool(args.emit_saved_checkpoint_delta_report):
        cmd.append("--emit-saved-checkpoint-delta-report")
    return cmd


def _rebuild_model_config_for_smoke(
    model_config: Any, *, tuning_flags: Mapping[str, Any]
) -> Any:
    to_filtered_dict = getattr(model_config, "to_filtered_dict", None)
    payload: dict[str, Any]
    if callable(to_filtered_dict):
        raw_payload = to_filtered_dict(exclude_augment=False)
        if not isinstance(raw_payload, dict):
            raise TypeError(
                "model_config.to_filtered_dict(exclude_augment=False) did not return a mapping"
            )
        payload = {str(key): value for key, value in raw_payload.items()}
    else:
        raw_payload = getattr(model_config, "__dict__", None)
        if not isinstance(raw_payload, dict):
            raise TypeError("model_config.__dict__ did not return a mapping")
        payload = {str(key): value for key, value in raw_payload.items()}
    payload.update(
        {
            "tune_llm": bool(tuning_flags["tune_llm"]),
            "tune_visual": bool(tuning_flags["tune_visual"]),
            "tune_top_llm_layers": int(tuning_flags["tune_top_llm_layers"]),
            "tune_projector": bool(tuning_flags["tune_projector"]),
            "tune_diffusion_model": bool(tuning_flags["tune_diffusion_model"]),
            "tune_vlln": bool(tuning_flags["tune_vlln"]),
        }
    )
    return model_config.__class__(**payload)


def _clone_args_with_overrides(
    args: argparse.Namespace,
    *,
    requested_scope_override: str | None = None,
    summary_json_override: object = _ARGS_OVERRIDE_UNSET,
) -> argparse.Namespace:
    payload = dict(vars(args))
    if requested_scope_override is not None:
        payload["recap_train_scope"] = str(requested_scope_override)
    if summary_json_override is not _ARGS_OVERRIDE_UNSET:
        payload["summary_json"] = "" if summary_json_override is None else str(summary_json_override)
    return argparse.Namespace(**payload)


def _resolve_one_step_verifier_training_overrides(
    args: argparse.Namespace,
    *,
    requested_additional_steps: int,
) -> dict[str, Any]:
    if (
        str(args.recap_train_scope) not in FULL_UPDATE_SCOPE_NAMES
        or int(requested_additional_steps) != 1
    ):
        return {
            "override_active": False,
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "lr_scheduler_type": None,
            "warmup_ratio": None,
        }
    return {
        "override_active": True,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
    }


def _prepare_wrapper_context(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    repo_root = _repo_root()
    script_path = Path(__file__).resolve()
    summary_json_path = (
        _resolve_path(repo_root, str(args.summary_json))
        if str(args.summary_json).strip()
        else None
    )
    dataset_path = _resolve_path(repo_root, str(args.dataset_path))
    output_dir = _resolve_full_update_output_dir(repo_root, str(args.output_dir))
    label_semantics_output_dir = _resolve_label_semantics_output_dir(
        repo_root,
        output_dir=output_dir,
        raw=str(getattr(args, "label_semantics_output_dir", "")),
    )
    runtime_log_dir = _resolve_path(repo_root, str(args.runtime_log_dir))
    continuation_checkpoint_path = _resolve_optional_checkpoint_path(repo_root, args)
    python_raw = str(args.python).strip() or _default_delegate_python()
    python_path = _resolve_path_preserve_symlink(repo_root, python_raw)
    scope_summary = _scope_summary_for_args(args)
    trainability_authority = _build_trainability_authority_from_args(
        args,
        scope_summary=scope_summary,
    )
    return {
        "args": args,
        "repo_root": repo_root,
        "script_path": script_path,
        "summary_json_path": summary_json_path,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "label_semantics_output_dir": label_semantics_output_dir,
        "runtime_log_dir": runtime_log_dir,
        "continuation_checkpoint_path": continuation_checkpoint_path,
        "python_path": python_path,
        "scope_summary": scope_summary,
        "trainability_authority": trainability_authority,
    }


def _build_wrapper_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    args = context["args"]
    dataset_path = context["dataset_path"]
    output_dir = context["output_dir"]
    label_semantics_output_dir = context["label_semantics_output_dir"]
    runtime_log_dir = context["runtime_log_dir"]
    continuation_checkpoint_path = context["continuation_checkpoint_path"]
    summary_json_path = context["summary_json_path"]
    scope_summary = context["scope_summary"]
    trainability_authority = context["trainability_authority"]
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "wrapper": "34b_recap_numeric_adv_smoke.py",
        "wrapper_status": "ok",
        "formal_mode": _condition_focused_continuation_enabled(args),
        "delegate_mode": "self-delegate monkeypatch -> gr00t.experiment.run",
        "conditioning_route": _conditioning_route(args),
        "runtime_indicator_mode": str(
            getattr(args, "runtime_indicator_mode", DEFAULT_RUNTIME_INDICATOR_MODE)
        ),
        "indicator_dropout_p": float(args.indicator_dropout_p),
        "text_indicator_prompt_raw_column": str(args.text_indicator_prompt_raw_column),
        "text_indicator_step_text_fallback": bool(
            args.text_indicator_step_text_fallback
        ),
        "numeric_adv_model_class": _model_class_for_route(args),
        "numeric_advantage_path_active": not _text_indicator_route_enabled(args),
        "text_indicator_path_active": _text_indicator_route_enabled(args),
        "force_top_llm_layers_zero": bool(args.force_top_llm_layers_zero),
        "dataset_path": str(dataset_path),
        "dataset_exists": bool(dataset_path.is_dir()),
        "output_dir": str(output_dir),
        "runtime_log_dir": str(runtime_log_dir),
        "label_semantics_output_dir": str(label_semantics_output_dir),
        "summary_json": None if summary_json_path is None else str(summary_json_path),
        "condition_focused_continuation": _condition_focused_continuation_enabled(args),
        "positive_curriculum_enabled": bool(args.positive_curriculum),
        "negative_retain_probability": float(args.negative_retain_probability),
        "positive_curriculum_seed": int(_resolve_positive_curriculum_seed(args)),
        "late_stage_positive_enabled": bool(args.late_stage_positive_emphasis),
        "late_stage_threshold": float(args.late_stage_threshold),
        "late_stage_rule": "advantage_input>0_and_t_norm>=threshold",
        "positive_oversample_enabled": int(args.positive_oversample_factor) > 1,
        "positive_oversample_factor": int(args.positive_oversample_factor),
        "continuation_checkpoint_path": (
            None
            if continuation_checkpoint_path is None
            else str(continuation_checkpoint_path)
        ),
        "continuation_checkpoint_exists": (
            False
            if continuation_checkpoint_path is None
            else bool(continuation_checkpoint_path.is_dir())
        ),
        "continuation_resume_step": (
            None
            if continuation_checkpoint_path is None
            else int(_checkpoint_resume_step(continuation_checkpoint_path))
        ),
        "selected_checkpoint_path": None,
        "selected_checkpoint_asset_path": None,
        "selected_checkpoint_exists": False,
        "selected_checkpoint_step": None,
        "advantage_embedding_keys_present": False,
        "advantage_embedding_missing_keys": [ADVANTAGE_WEIGHT_KEY, ADVANTAGE_BIAS_KEY],
        "train_scope_requested": str(scope_summary["train_scope_requested"]),
        "scope_faithfulness": str(scope_summary["scope_faithfulness"]),
        "method_faithfulness": dict(scope_summary["method_faithfulness"]),
        "train_scope_taxonomy": dict(scope_summary),
        "trainability_authority": dict(trainability_authority),
        "route_freeze": trainability_authority.get("route_freeze"),
        "delegate_cmd": None,
        "delegate_cmd_shell": None,
        "delegate_returncode": None,
        "runtime_log_path": None,
        "checkpoint_load_report_path": None,
        "checkpoint_load_report_exists": False,
        "conditioned_initial_advantage_embedding_snapshot_path": str(
            output_dir / CONDITIONED_INITIAL_ADVANTAGE_SNAPSHOT_FILENAME
        ),
        "conditioned_initial_advantage_embedding_snapshot_exists": False,
        "max_steps": int(args.max_steps),
        "save_total_limit": int(args.save_total_limit),
        "effective_tuning_flags": _effective_tuning_flags(args),
        "training_scope": _training_scope_payload(args, scope_summary=scope_summary),
        "checkpoint_load_missing_keys": [],
        "checkpoint_load_unexpected_keys": [],
        "skipped_mismatched_keys": [],
        "blocked_mismatched_keys": [],
        "trainer_global_step": None,
        "grad_probe_after_backward": None,
        "param_delta_after_step": None,
        "all_major_grad_norms": None,
        "all_major_param_delta": None,
        "error": None,
    }


def run_numeric_adv_single_scope(
    args: argparse.Namespace,
    *,
    requested_scope_override: str | None = None,
    summary_json_override: object = _ARGS_OVERRIDE_UNSET,
    emit_summary: bool = False,
) -> tuple[int, dict[str, Any]]:
    effective_args = _clone_args_with_overrides(
        args,
        requested_scope_override=requested_scope_override,
        summary_json_override=summary_json_override,
    )
    context = _prepare_wrapper_context(effective_args)
    repo_root = context["repo_root"]
    script_path = context["script_path"]
    summary_json_path = context["summary_json_path"]
    dataset_path = context["dataset_path"]
    output_dir = context["output_dir"]
    runtime_log_dir = context["runtime_log_dir"]
    python_path = context["python_path"]
    payload = _build_wrapper_payload(context)

    try:
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        if not python_path.is_file():
            raise FileNotFoundError(f"Delegate python not found: {python_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        runtime_log_dir.mkdir(parents=True, exist_ok=True)
        runtime_log_path = runtime_log_dir / (
            f"{str(effective_args.runtime_log_prefix).strip() or DEFAULT_RUNTIME_LOG_PREFIX}_{_timestamp()}.log"
        )
        checkpoint_load_report_path = runtime_log_path.with_name(
            runtime_log_path.stem + "_checkpoint_load_report.json"
        )
        delegate_cmd = _build_delegate_cmd(
            script_path=script_path,
            args=effective_args,
            dataset_path=dataset_path,
            output_dir=output_dir,
            delegate_python=python_path,
        )
        env = dict(os.environ)
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        env[CHECKPOINT_LOAD_REPORT_ENV] = str(checkpoint_load_report_path)
        payload["delegate_cmd"] = _jsonify_cmd(delegate_cmd)
        payload["delegate_cmd_shell"] = shlex.join(delegate_cmd)
        payload["runtime_log_path"] = str(runtime_log_path)
        payload["checkpoint_load_report_path"] = str(checkpoint_load_report_path)

        rc = _run_with_tee(
            delegate_cmd,
            cwd=repo_root,
            log_path=runtime_log_path,
            env=env,
        )
        payload["delegate_returncode"] = int(rc)
        if checkpoint_load_report_path.is_file():
            checkpoint_load_report = json.loads(
                checkpoint_load_report_path.read_text(encoding="utf-8")
            )
            payload["checkpoint_load_report_exists"] = True
            payload["checkpoint_load_missing_keys"] = list(
                checkpoint_load_report.get("missing_keys", [])
            )
            payload["checkpoint_load_unexpected_keys"] = list(
                checkpoint_load_report.get("unexpected_keys", [])
            )
            payload["skipped_mismatched_keys"] = list(
                checkpoint_load_report.get("skipped_mismatched_keys", [])
            )
            payload["blocked_mismatched_keys"] = list(
                checkpoint_load_report.get("blocked_mismatched_keys", [])
            )
        grad_probe_after_backward = _load_repo_local_runtime_metadata(
            output_dir,
            filename=trainability_entrypoint.FIRST_BACKWARD_GRAD_PROBE_FILENAME,
        )
        if grad_probe_after_backward is not None:
            payload["grad_probe_after_backward"] = grad_probe_after_backward
            payload["all_major_grad_norms"] = _extract_probe_scope_metric_summary(
                grad_probe_after_backward,
                metric_key="grad_l2_norm",
            )
            trainer_global_step = _extract_probe_trainer_global_step(
                grad_probe_after_backward
            )
            if trainer_global_step is not None:
                payload["trainer_global_step"] = trainer_global_step
        param_delta_after_step = _load_repo_local_runtime_metadata(
            output_dir,
            filename=trainability_entrypoint.FIRST_OPTIMIZER_STEP_PARAM_DELTA_FILENAME,
        )
        if param_delta_after_step is not None:
            payload["param_delta_after_step"] = param_delta_after_step
            payload["all_major_param_delta"] = _extract_probe_scope_metric_summary(
                param_delta_after_step,
                metric_key="delta_l2_norm",
            )
            trainer_global_step = _extract_probe_trainer_global_step(
                param_delta_after_step
            )
            if trainer_global_step is not None:
                payload["trainer_global_step"] = trainer_global_step
        snapshot_path = output_dir / CONDITIONED_INITIAL_ADVANTAGE_SNAPSHOT_FILENAME
        payload["conditioned_initial_advantage_embedding_snapshot_exists"] = bool(
            snapshot_path.is_file()
        )
        if rc != 0:
            payload["wrapper_status"] = "blocked"
            payload["error"] = f"delegate_failed: returncode={rc}"

        selected_checkpoint = _latest_checkpoint(output_dir)
        if _text_indicator_route_enabled(effective_args):
            selected_asset = _selected_checkpoint_asset(selected_checkpoint)
            advantage_ok = None
            missing_keys = []
        else:
            advantage_ok, missing_keys, selected_asset = _checkpoint_advantage_embedding_status(
                selected_checkpoint
            )
        payload["selected_checkpoint_path"] = (
            str(selected_checkpoint)
            if selected_checkpoint is not None and selected_asset is not None
            else None
        )
        payload["selected_checkpoint_asset_path"] = (
            str(selected_asset) if selected_asset is not None else None
        )
        payload["selected_checkpoint_exists"] = bool(selected_asset is not None)
        selected_checkpoint_step = (
            None
            if selected_checkpoint is None
            else int(_checkpoint_resume_step(selected_checkpoint))
        )
        payload["selected_checkpoint_step"] = selected_checkpoint_step
        if selected_checkpoint_step is not None:
            payload["trainer_global_step"] = max(
                int(payload["trainer_global_step"] or 0),
                int(selected_checkpoint_step),
            )
        payload["advantage_embedding_keys_present"] = (
            None if advantage_ok is None else bool(advantage_ok)
        )
        payload["advantage_embedding_missing_keys"] = list(missing_keys)
        if selected_asset is None and payload["error"] is None:
            payload["wrapper_status"] = "blocked"
            payload["error"] = "missing_checkpoint_asset"
        elif advantage_ok is False and payload["error"] is None:
            payload["wrapper_status"] = "blocked"
            payload["error"] = "checkpoint_missing_advantage_embedding_keys"

        final_rc = 0 if payload["wrapper_status"] == "ok" else 1
        if emit_summary:
            _emit_summary(payload, summary_json_path=summary_json_path)
        return final_rc, payload
    except Exception as exc:
        payload["wrapper_status"] = "blocked"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        if emit_summary:
            _emit_summary(payload, summary_json_path=summary_json_path)
        return 1, payload


def _task8_probes_requested(args: argparse.Namespace) -> bool:
    return any(
        bool(getattr(args, name, False))
        for name in (
            "write_conditioning_functional_probe",
            "write_paired_action_probe",
            "write_label_semantics_audit",
            "write_shuffled_advantage_negative_control",
        )
    )


def _task8_advantage_key(value: float | None) -> str:
    if value is None:
        return "null"
    return format(float(value), ".7g")


def _task8_json_ready(value: object) -> object:
    import numpy as np
    import torch

    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _task8_json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_task8_json_ready(item) for item in value]
    return value


def _task8_hash_payload(value: object) -> str:
    payload = json.dumps(
        _task8_json_ready(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _task8_clone_model_inputs(value: object) -> object:
    import torch

    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {str(key): _task8_clone_model_inputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_task8_clone_model_inputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_task8_clone_model_inputs(item) for item in value)
    return copy.deepcopy(value)


def _task8_infer_batch_size(collated_inputs: Mapping[str, Any]) -> int:
    import torch

    stack: list[object] = [collated_inputs]
    while stack:
        current = stack.pop()
        if torch.is_tensor(current) and current.ndim >= 1:
            return int(current.shape[0])
        if isinstance(current, Mapping):
            stack.extend(current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            stack.extend(current)
    return 1


def _task8_move_inputs_to_device(value: object, *, device: Any) -> object:
    import torch

    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {
            str(key): _task8_move_inputs_to_device(item, device=device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_task8_move_inputs_to_device(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_task8_move_inputs_to_device(item, device=device) for item in value)
    return value


def _task8_override_advantage_inputs(
    collated_inputs: Mapping[str, Any],
    *,
    advantage_value: float | None | Sequence[float],
    model_device: Any,
) -> dict[str, Any]:
    import torch

    from work.recap.advantage import validate_advantage_input_value

    cloned = _task8_clone_model_inputs(collated_inputs)
    if not isinstance(cloned, dict):
        raise TypeError("collated_inputs clone must remain a dict")
    target = cloned
    if isinstance(cloned.get("inputs"), Mapping):
        target = dict(cloned["inputs"])
        cloned["inputs"] = target
    target.pop("advantage", None)
    if advantage_value is None:
        return cloned
    batch_size = _task8_infer_batch_size(cloned)
    if isinstance(advantage_value, Sequence) and not isinstance(
        advantage_value, (str, bytes, bytearray)
    ):
        values = [
            float(
                validate_advantage_input_value(
                    raw_value,
                    context=f"task8_advantage[{idx}]",
                )
            )
            for idx, raw_value in enumerate(advantage_value)
        ]
        if len(values) != batch_size:
            raise ValueError(
                f"task8 advantage vector length must match batch size: {len(values)} != {batch_size}"
            )
        target["advantage"] = torch.tensor(
            [[value] for value in values],
            dtype=torch.bfloat16,
            device=model_device,
        )
        return cloned
    resolved_advantage = float(
        validate_advantage_input_value(
            advantage_value,
            context="task8_advantage",
        )
    )
    target["advantage"] = torch.full(
        (batch_size, 1),
        resolved_advantage,
        dtype=torch.bfloat16,
        device=model_device,
    )
    return cloned


def _task8_collect_dataset_inventory(trainer: Any) -> dict[str, Any]:
    from work.recap.advantage import ADVANTAGE_INPUT_COLUMN

    import numpy as np

    train_dataset = getattr(trainer, "train_dataset", None)
    datasets = list(getattr(train_dataset, "datasets", []) or [])
    if not datasets:
        raise RuntimeError("task8 probes require trainer.train_dataset.datasets")
    total_refs = 0
    positive_count = 0
    nonpositive_count = 0
    zero_count = 0
    min_advantage: float | None = None
    max_advantage: float | None = None
    positive_refs: list[dict[str, Any]] = []
    nonpositive_refs: list[dict[str, Any]] = []
    sequential_refs: list[dict[str, Any]] = []
    unique_values_preview: list[float] = []
    preview_limit = _TASK8_DEFAULT_SUBSET_SAMPLE_COUNT * 4
    for dataset_index, dataset in enumerate(datasets):
        episode_loader = getattr(dataset, "episode_loader", None)
        if episode_loader is None:
            continue
        dataset_path = Path(str(getattr(dataset, "dataset_path", "")))
        resolve_advantage = getattr(dataset, "resolve_advantage", None)
        for episode_index in range(len(episode_loader)):
            episode_data = episode_loader[episode_index]
            effective_length = int(dataset.get_effective_episode_length(episode_index))
            for step_index in range(effective_length):
                if callable(resolve_advantage):
                    raw_advantage = resolve_advantage(episode_data, step_index)
                    advantage_value = float(np.asarray(raw_advantage).reshape(-1)[0])
                else:
                    advantage_value = float(
                        episode_data[ADVANTAGE_INPUT_COLUMN].iloc[int(step_index)]
                    )
                ref = {
                    "dataset_index": int(dataset_index),
                    "dataset_path": str(dataset_path),
                    "episode_index": int(episode_index),
                    "step_index": int(step_index),
                    "advantage": float(advantage_value),
                    "sign_bucket": (
                        "positive" if float(advantage_value) > 0.0 else "nonpositive"
                    ),
                }
                total_refs += 1
                min_advantage = (
                    float(advantage_value)
                    if min_advantage is None
                    else min(float(min_advantage), float(advantage_value))
                )
                max_advantage = (
                    float(advantage_value)
                    if max_advantage is None
                    else max(float(max_advantage), float(advantage_value))
                )
                if abs(float(advantage_value)) <= 1e-9:
                    zero_count += 1
                if float(advantage_value) > 0.0:
                    positive_count += 1
                    if len(positive_refs) < preview_limit:
                        positive_refs.append(ref)
                else:
                    nonpositive_count += 1
                    if len(nonpositive_refs) < preview_limit:
                        nonpositive_refs.append(ref)
                if len(sequential_refs) < preview_limit * 2:
                    sequential_refs.append(ref)
                rounded_advantage = round(float(advantage_value), 7)
                if rounded_advantage not in unique_values_preview and len(unique_values_preview) < 16:
                    unique_values_preview.append(rounded_advantage)
    summary = {
        "total_refs": total_refs,
        "positive_count": positive_count,
        "nonpositive_count": nonpositive_count,
        "zero_count": zero_count,
        "min_advantage": min_advantage,
        "max_advantage": max_advantage,
        "unique_advantages_preview": unique_values_preview,
    }
    return {
        "summary": summary,
        "positive_refs": positive_refs,
        "nonpositive_refs": nonpositive_refs,
        "sequential_refs": sequential_refs,
        "datasets": datasets,
    }


def _task8_pick_probe_refs(
    inventory: Mapping[str, Any],
    *,
    balanced: bool,
) -> dict[str, Any]:
    sample_count = _TASK8_DEFAULT_SUBSET_SAMPLE_COUNT
    positive_refs = list(inventory.get("positive_refs", []))
    nonpositive_refs = list(inventory.get("nonpositive_refs", []))
    sequential_refs = list(inventory.get("sequential_refs", []))
    used: set[tuple[int, int, int]] = set()
    balanced_budget_blocking_reasons: list[str] = []
    balanced_budget_supported = bool(
        len(positive_refs) >= sample_count and len(nonpositive_refs) >= sample_count
    )
    if balanced and len(positive_refs) < sample_count:
        balanced_budget_blocking_reasons.append(
            "insufficient_positive_refs_for_balanced_16_per_subset"
        )
    if balanced and len(nonpositive_refs) < sample_count:
        balanced_budget_blocking_reasons.append(
            "insufficient_nonpositive_refs_for_balanced_16_per_subset"
        )
    if balanced and len(sequential_refs) < sample_count * 2:
        balanced_budget_blocking_reasons.append(
            "insufficient_total_refs_for_two_16_sample_subsets"
        )

    def _consume(refs: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for raw_ref in refs:
            key = (
                int(raw_ref["dataset_index"]),
                int(raw_ref["episode_index"]),
                int(raw_ref["step_index"]),
            )
            if key in used:
                continue
            used.add(key)
            out.append(dict(raw_ref))
            if len(out) >= count:
                break
        return out

    selection_strategy = "sequential_fallback"
    train_subset_refs: list[dict[str, Any]] = []
    heldout_refs: list[dict[str, Any]] = []
    if balanced and balanced_budget_supported:
        half = sample_count // 2
        train_subset_refs.extend(_consume(positive_refs, half))
        train_subset_refs.extend(_consume(nonpositive_refs, sample_count - len(train_subset_refs)))
        heldout_refs.extend(_consume(positive_refs, half))
        heldout_refs.extend(_consume(nonpositive_refs, sample_count - len(heldout_refs)))
        selection_strategy = "balanced_positive_nonpositive"
    elif balanced:
        selection_strategy = "balanced_budget_insufficient_sequential_fallback"
    if len(train_subset_refs) < sample_count:
        train_subset_refs.extend(_consume(sequential_refs, sample_count - len(train_subset_refs)))
    if len(heldout_refs) < sample_count:
        heldout_refs.extend(_consume(sequential_refs, sample_count - len(heldout_refs)))
    if not train_subset_refs or not heldout_refs:
        raise RuntimeError("task8 probes require at least one train_subset and one heldout ref")
    return {
        "selection_strategy": selection_strategy,
        "train_subset_refs": train_subset_refs,
        "heldout_refs": heldout_refs,
        "obs_probe_ref": dict(heldout_refs[0]),
        "selection_metadata": {
            "target_subset_sample_count": int(sample_count),
            "balanced_selection_requested": bool(balanced),
            "balanced_selection_applied": bool(
                selection_strategy == "balanced_positive_nonpositive"
            ),
            "balanced_subset_budget_supported": bool(balanced_budget_supported),
            "balanced_subset_budget_blocking_reasons": (
                balanced_budget_blocking_reasons
            ),
            "positive_refs_available": int(len(positive_refs)),
            "nonpositive_refs_available": int(len(nonpositive_refs)),
            "total_refs_available": int(len(sequential_refs)),
            "train_subset_selected_count": int(len(train_subset_refs)),
            "heldout_subset_selected_count": int(len(heldout_refs)),
        },
    }


def _task8_build_sample_from_ref(
    datasets: Sequence[Any],
    *,
    ref: Mapping[str, Any],
) -> dict[str, Any]:
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType
    import numpy as np

    dataset = datasets[int(ref["dataset_index"])]
    episode_loader = getattr(dataset, "episode_loader", None)
    if episode_loader is None:
        raise RuntimeError("task8 sample build requires dataset.episode_loader")
    episode_index = int(ref["episode_index"])
    step_index = int(ref["step_index"])
    episode_data = episode_loader[episode_index]
    vla_step_data = extract_step_data(
        episode_data,
        step_index,
        dataset.modality_configs,
        dataset.embodiment_tag,
        getattr(dataset, "allow_padding", False),
    )
    datapoint = dataset.get_datapoint(episode_data, step_index)
    advantage_tensor = datapoint.get("advantage")
    if advantage_tensor is None:
        raise KeyError("task8 sample datapoint missing advantage")
    processed = {
        str(key): value
        for key, value in datapoint.items()
    }
    return {
        "ref": dict(ref),
        "dataset_path": str(getattr(dataset, "dataset_path", "")),
        "embodiment_tag": dataset.embodiment_tag,
        "vla_step_data": vla_step_data,
        "processed": processed,
        "advantage": float(np.asarray(advantage_tensor).reshape(-1)[0]),
    }


def _task8_build_sample_bundle(trainer: Any, refs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    samples = [
        _task8_build_sample_from_ref(getattr(trainer.train_dataset, "datasets", []), ref=ref)
        for ref in refs
    ]
    collated_inputs = trainer.data_collator([sample["processed"] for sample in samples])
    return {
        "samples": samples,
        "collated_inputs": collated_inputs,
        "native_advantages": [float(sample["advantage"]) for sample in samples],
        "sample_refs": [dict(sample["ref"]) for sample in samples],
        "sample_count": int(len(samples)),
    }


def _task8_seeded_non_identity_permutation(
    length: int,
    *,
    seed: int,
) -> dict[str, Any]:
    indices = list(range(int(length)))
    if len(indices) < 2:
        return {
            "permutation_indices": indices,
            "non_identity": False,
            "shuffle_seed": int(seed),
            "shuffle_method": "seeded_non_identity_shuffle",
            "identity_collision_resolved": False,
            "blocking_reasons": ["subset_requires_at_least_two_samples_for_shuffle"],
        }
    shuffled_indices = list(indices)
    rng = random.Random(int(seed))
    rng.shuffle(shuffled_indices)
    identity_collision_resolved = False
    if shuffled_indices == indices:
        rotate_by = 1 + rng.randrange(len(indices) - 1)
        shuffled_indices = shuffled_indices[rotate_by:] + shuffled_indices[:rotate_by]
        identity_collision_resolved = True
    return {
        "permutation_indices": shuffled_indices,
        "non_identity": bool(shuffled_indices != indices),
        "shuffle_seed": int(seed),
        "shuffle_method": "seeded_non_identity_shuffle",
        "identity_collision_resolved": bool(identity_collision_resolved),
        "blocking_reasons": [],
    }


def _task8_model_loss(
    *,
    model: Any,
    collated_inputs: Mapping[str, Any],
    advantage_override: float | None | Sequence[float],
    seed: int,
) -> float:
    import torch

    from work.recap import policy as recap_policy

    prepared_inputs = _task8_override_advantage_inputs(
        collated_inputs,
        advantage_value=advantage_override,
        model_device=getattr(model, "device", None),
    )
    prepared_inputs = recap_policy._rec_to_dtype(prepared_inputs, dtype=torch.bfloat16)
    was_training = bool(getattr(model, "training", False))
    fork_devices: list[int] = []
    model_device = getattr(model, "device", None)
    if getattr(model_device, "type", None) == "cuda":
        device_index = getattr(model_device, "index", None)
        if device_index is None:
            device_index = torch.cuda.current_device()
        fork_devices = [int(device_index)]
    model.eval()
    try:
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            autocast_device_type = (
                "cuda"
                if getattr(model_device, "type", None) == "cuda"
                else "cpu"
            )
            with torch.inference_mode():
                with torch.autocast(
                    device_type=autocast_device_type,
                    dtype=torch.bfloat16,
                    enabled=(autocast_device_type == "cuda"),
                ):
                    outputs = model(**prepared_inputs)
        return float(outputs["loss"].detach().float().cpu().item())
    finally:
        if was_training:
            model.train()


def _task8_tensor_l2(value: object) -> float | None:
    import numpy as np
    import torch

    if value is None:
        return None
    if torch.is_tensor(value):
        arr = value.detach().float().cpu().numpy()
    else:
        arr = np.asarray(value, dtype=np.float32)
    if arr.size <= 0:
        return 0.0
    return float(np.linalg.norm(arr.reshape(-1), ord=2))


def _task8_conditioning_preview(
    *,
    model: Any,
    collated_inputs: Mapping[str, Any],
    advantage_override: float | None,
) -> dict[str, Any]:
    from transformers.feature_extraction_utils import BatchFeature

    prepared_inputs = _task8_override_advantage_inputs(
        collated_inputs,
        advantage_value=advantage_override,
        model_device=getattr(model, "device", None),
    )
    prepared_inputs = _task8_move_inputs_to_device(
        prepared_inputs,
        device=getattr(model, "device", None),
    )
    if not isinstance(prepared_inputs, Mapping):
        raise TypeError("task8 conditioning preview requires mapping collated inputs")
    action_payload = (
        prepared_inputs.get("inputs")
        if isinstance(prepared_inputs.get("inputs"), Mapping)
        else prepared_inputs
    )
    if not isinstance(action_payload, Mapping):
        raise TypeError("task8 conditioning preview requires mapping model inputs")
    action_input = BatchFeature(
        data={str(key): value for key, value in action_payload.items()}
    )
    preview = model.action_head.preview_advantage_conditioning(action_input)
    advantage_features = preview.get("advantage_features")
    state_features = preview.get("state_features")
    conditioned_state_features = preview.get("conditioned_state_features")
    conditioned_delta = None
    if state_features is not None and conditioned_state_features is not None:
        conditioned_delta = conditioned_state_features - state_features
    return {
        "advantage": None if advantage_override is None else float(advantage_override),
        "state_feature_l2": _task8_tensor_l2(state_features),
        "advantage_feature_l2": _task8_tensor_l2(advantage_features),
        "conditioned_state_delta_l2": _task8_tensor_l2(conditioned_delta),
    }


def _task8_stage_action_delta(
    baseline_stage: Mapping[str, Any],
    probe_stage: Mapping[str, Any],
    *,
    layer_name: str,
) -> dict[str, Any]:
    import numpy as np

    baseline_available = bool(baseline_stage.get("available", False))
    probe_available = bool(probe_stage.get("available", False))
    if not baseline_available or not probe_available:
        reasons = [
            reason
            for reason in (baseline_stage.get("reason"), probe_stage.get("reason"))
            if isinstance(reason, str) and reason.strip()
        ]
        return {
            "layer_name": layer_name,
            "available": False,
            "reason": "; ".join(reasons) or f"{layer_name}_unavailable",
            "delta_l2": None,
            "delta": None,
        }
    baseline_action = baseline_stage.get("action")
    probe_action = probe_stage.get("action")
    if isinstance(baseline_action, Mapping) and isinstance(probe_action, Mapping):
        delta: dict[str, Any] = {}
        flat: list[Any] = []
        for key in sorted(set(baseline_action) | set(probe_action)):
            if key not in baseline_action or key not in probe_action:
                return {
                    "layer_name": layer_name,
                    "available": False,
                    "reason": f"{layer_name}_key_mismatch:{key}",
                    "delta_l2": None,
                    "delta": None,
                }
            delta_arr = np.asarray(probe_action[key], dtype=np.float32) - np.asarray(
                baseline_action[key], dtype=np.float32
            )
            delta[str(key)] = delta_arr.tolist()
            flat.append(delta_arr.reshape(-1))
        merged = np.concatenate(flat, axis=0) if flat else np.zeros((0,), dtype=np.float32)
        return {
            "layer_name": layer_name,
            "available": True,
            "reason": None,
            "delta_l2": float(np.linalg.norm(merged, ord=2)),
            "delta": delta,
        }
    delta_arr = np.asarray(probe_action, dtype=np.float32) - np.asarray(
        baseline_action,
        dtype=np.float32,
    )
    return {
        "layer_name": layer_name,
        "available": True,
        "reason": None,
        "delta_l2": float(np.linalg.norm(delta_arr.reshape(-1), ord=2)),
        "delta": delta_arr.tolist(),
    }


def _evaluate_paired_action_probe_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    layers = payload.get("action_delta_layers")
    if not isinstance(layers, Mapping):
        return {
            "required_layers": list(_TASK8_REQUIRED_ACTION_DELTA_LAYERS),
            "missing_layers": list(_TASK8_REQUIRED_ACTION_DELTA_LAYERS),
            "unavailable_layers": [],
            "instrumentation_incomplete": True,
            "action_sensitivity_gate_pass": False,
        }
    missing_layers = [
        layer_name
        for layer_name in _TASK8_REQUIRED_ACTION_DELTA_LAYERS
        if layer_name not in layers
    ]
    unavailable_layers: list[str] = []
    positive_delta_present = False
    for layer_name in _TASK8_REQUIRED_ACTION_DELTA_LAYERS:
        layer_payload = layers.get(layer_name)
        if not isinstance(layer_payload, Mapping):
            continue
        if layer_payload.get("available") is not True:
            unavailable_layers.append(layer_name)
            continue
        delta_l2 = layer_payload.get("delta_l2")
        if isinstance(delta_l2, (int, float)) and float(delta_l2) > 0.0:
            positive_delta_present = True
    instrumentation_incomplete = bool(missing_layers or unavailable_layers)
    return {
        "required_layers": list(_TASK8_REQUIRED_ACTION_DELTA_LAYERS),
        "missing_layers": missing_layers,
        "unavailable_layers": unavailable_layers,
        "instrumentation_incomplete": instrumentation_incomplete,
        "action_sensitivity_gate_pass": bool(
            positive_delta_present and not instrumentation_incomplete
        ),
    }


def _task8_build_subset_loss_probe(
    *,
    subset_name: str,
    model: Any,
    subset_bundle: Mapping[str, Any],
    probe_seed: int,
) -> dict[str, Any]:
    loss_by_advantage: dict[str, float] = {}
    collated_inputs = subset_bundle["collated_inputs"]
    for offset, advantage_value in enumerate(_TASK8_ADVANTAGES_TESTED):
        loss_by_advantage[_task8_advantage_key(advantage_value)] = _task8_model_loss(
            model=model,
            collated_inputs=collated_inputs,
            advantage_override=advantage_value,
            seed=probe_seed + offset,
        )
    native_advantages = list(subset_bundle["native_advantages"])
    native_label_loss = _task8_model_loss(
        model=model,
        collated_inputs=collated_inputs,
        advantage_override=native_advantages,
        seed=probe_seed + 100,
    )
    values = list(loss_by_advantage.values())
    span = max(values) - min(values) if values else 0.0
    return {
        "subset_name": subset_name,
        "sample_refs": list(subset_bundle["sample_refs"]),
        "native_advantages": native_advantages,
        "loss_by_advantage": loss_by_advantage,
        "native_label_loss": float(native_label_loss),
        "loss_span": float(span),
        "loss_sensitive": bool(span > 1e-9),
    }


def _evaluate_shuffled_negative_control(payload: Mapping[str, Any]) -> dict[str, Any]:
    per_subset = payload.get("per_subset")
    if not isinstance(per_subset, Mapping) or not per_subset:
        return {
            "negative_control_pass": False,
            "blocking_reasons": ["missing_per_subset"],
        }
    blocking_reasons: list[str] = []
    for subset_name, subset_payload in per_subset.items():
        if not isinstance(subset_payload, Mapping):
            blocking_reasons.append(f"{subset_name}_not_mapping")
            continue
        delta = subset_payload.get("shuffled_minus_true_loss")
        if not isinstance(delta, (int, float)):
            blocking_reasons.append(f"{subset_name}_missing_shuffled_minus_true_loss")
            continue
        if float(delta) <= 1e-9:
            blocking_reasons.append(f"{subset_name}_shuffled_control_not_worse_than_true")
    return {
        "negative_control_pass": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
    }


def _evaluate_label_semantics_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    blocking_reasons: list[str] = []
    if payload.get("value_is_constant") is True:
        blocking_reasons.append("value_is_constant")
    if payload.get("all_returns_negative") is True:
        blocking_reasons.append("all_returns_negative")
    if payload.get("positive_subgoal_evidence_available") is not True:
        blocking_reasons.append("positive_subgoal_evidence_unavailable")
    shuffled_control = payload.get("shuffled_advantage_negative_control")
    if not isinstance(shuffled_control, Mapping):
        blocking_reasons.append("missing_shuffled_advantage_negative_control")
    elif shuffled_control.get("negative_control_pass") is not True:
        blocking_reasons.append("shuffled_advantage_negative_control_failed")
    gate_pass = not blocking_reasons
    return {
        "label_semantics_gate_pass": gate_pass,
        "formal_claim_allowed": gate_pass,
        "blocking_reasons": blocking_reasons,
    }


def _task10_load_full_update_scope_gate(output_dir: Path) -> dict[str, Any]:
    payload = _load_repo_local_runtime_metadata(
        output_dir,
        filename=_TASK10_SCOPE_AUDIT_DYNAMIC_FILENAME,
    )
    metadata_path = (
        trainability_entrypoint.resolve_repo_local_metadata_dir_for_output_dir(output_dir)
        / _TASK10_SCOPE_AUDIT_DYNAMIC_FILENAME
    )
    if not isinstance(payload, Mapping):
        return {
            "full_update_scope_gate_pass": False,
            "scope_gate_path": str(metadata_path),
            "scope_gate_reason": "missing_full_update_scope_audit_dynamic",
            "resolution_status": None,
            "best_scope_authority": None,
        }
    resolution_status = str(payload.get("resolution_status") or "").upper()
    best_scope_authority = payload.get("best_scope_authority") is True
    scope_gate_pass = bool(
        resolution_status == _TASK10_GATE_STATUS_PASS and best_scope_authority
    )
    if scope_gate_pass:
        scope_gate_reason = None
    elif resolution_status != _TASK10_GATE_STATUS_PASS:
        scope_gate_reason = "full_update_scope_resolution_not_pass"
    else:
        scope_gate_reason = "best_scope_authority_false"
    return {
        "full_update_scope_gate_pass": scope_gate_pass,
        "scope_gate_path": str(metadata_path),
        "scope_gate_reason": scope_gate_reason,
        "resolution_status": resolution_status or None,
        "best_scope_authority": best_scope_authority,
    }


def _task10_build_loss_probe_contract(
    *,
    conditioning_probe: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    blocking_reasons: list[str] = []
    train_subset_probe = conditioning_probe.get("train_subset_loss_probe")
    heldout_probe = conditioning_probe.get("heldout_loss_probe")
    if not isinstance(train_subset_probe, Mapping):
        blocking_reasons.append("missing_train_subset_loss_probe")
    if not isinstance(heldout_probe, Mapping):
        blocking_reasons.append("missing_heldout_loss_probe")
    loss_sensitivity_gate_pass = bool(
        conditioning_probe.get("loss_sensitivity_gate_pass", False)
    ) and not blocking_reasons
    if not blocking_reasons and not loss_sensitivity_gate_pass:
        blocking_reasons.append("loss_sensitivity_gate_block")
    instrumentation_incomplete = bool(blocking_reasons) and any(
        reason.startswith("missing_") for reason in blocking_reasons
    )
    return {
        "path": str(output_dir / _TASK8_STEP20_PROBE_FILENAME),
        "status": (
            _TASK10_GATE_STATUS_PASS
            if loss_sensitivity_gate_pass
            else _TASK10_GATE_STATUS_BLOCK
        ),
        "loss_sensitivity_gate_pass": loss_sensitivity_gate_pass,
        "instrumentation_incomplete": instrumentation_incomplete,
        "blocking_reasons": blocking_reasons,
    }


def _task10_build_paired_action_probe_contract(
    *,
    paired_probe: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    paired_contract = _evaluate_paired_action_probe_contract(paired_probe)
    blocking_reasons: list[str] = []
    if paired_contract["instrumentation_incomplete"]:
        blocking_reasons.append("paired_action_instrumentation_incomplete")
    if not paired_contract["action_sensitivity_gate_pass"]:
        blocking_reasons.append("action_sensitivity_gate_block")
    return {
        "path": str(output_dir / _TASK8_PAIRED_STEP20_PROBE_FILENAME),
        "status": (
            _TASK10_GATE_STATUS_PASS
            if paired_contract["action_sensitivity_gate_pass"]
            and not paired_contract["instrumentation_incomplete"]
            else _TASK10_GATE_STATUS_BLOCK
        ),
        "action_sensitivity_gate_pass": bool(
            paired_contract["action_sensitivity_gate_pass"]
        ),
        "instrumentation_incomplete": bool(
            paired_contract["instrumentation_incomplete"]
        ),
        "required_layers": list(paired_contract["required_layers"]),
        "missing_layers": list(paired_contract["missing_layers"]),
        "unavailable_layers": list(paired_contract["unavailable_layers"]),
        "blocking_reasons": blocking_reasons,
    }


def _task10_build_first_subgoal_probe_contract(
    first_subgoal_probe: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(first_subgoal_probe, Mapping):
        return {
            "status": _TASK10_GATE_STATUS_SKIPPED,
            "skip_reason": "p4_rollout_probe_not_run",
            "strong_subgoal_progress_gate_pass": False,
            "weak_distance_only": False,
            "blocking_reasons": [],
        }
    blocking_reasons = [
        str(reason)
        for reason in first_subgoal_probe.get("blocking_reasons", [])
        if isinstance(reason, str) and reason.strip()
    ]
    raw_status = str(first_subgoal_probe.get("status") or "").upper()
    strong_subgoal_progress_gate_pass = bool(
        first_subgoal_probe.get("strong_subgoal_progress_gate_pass", False)
    )
    weak_distance_only = bool(first_subgoal_probe.get("weak_distance_only", False))
    if raw_status not in {
        _TASK10_GATE_STATUS_PASS,
        _TASK10_GATE_STATUS_BLOCK,
        _TASK10_GATE_STATUS_SKIPPED,
    }:
        raw_status = (
            _TASK10_GATE_STATUS_PASS
            if strong_subgoal_progress_gate_pass
            else _TASK10_GATE_STATUS_BLOCK
        )
    return {
        "status": raw_status,
        "skip_reason": (
            str(first_subgoal_probe.get("skip_reason"))
            if raw_status == _TASK10_GATE_STATUS_SKIPPED
            and first_subgoal_probe.get("skip_reason") is not None
            else None
        ),
        "strong_subgoal_progress_gate_pass": strong_subgoal_progress_gate_pass,
        "weak_distance_only": weak_distance_only,
        "blocking_reasons": blocking_reasons,
    }


def _task10_p3_skip_reason(
    *,
    full_update_scope_gate_pass: bool,
    instrumentation_incomplete: bool,
    p2_not_dead: bool,
    shuffled_advantage_negative_control_pass: bool,
) -> str | None:
    if full_update_scope_gate_pass is not True:
        return "full_update_scope_gate_block"
    if instrumentation_incomplete:
        return "instrumentation_incomplete"
    if not p2_not_dead:
        return "p2_not_dead_gate_block"
    if not shuffled_advantage_negative_control_pass:
        return "shuffled_advantage_negative_control_block"
    return None


def _task10_routing_decision(
    *,
    p3_formal_training_eligible: bool,
    p5_probe_eligible: bool,
    p5_formal_10ep_eligible: bool,
    p6_branch_eligible: bool,
) -> str:
    if p3_formal_training_eligible:
        return _TASK10_ROUTING_ROUTE_P3
    if p5_formal_10ep_eligible:
        return _TASK10_ROUTING_ROUTE_P5_FORMAL
    if p6_branch_eligible:
        return _TASK10_ROUTING_ROUTE_P6
    if p5_probe_eligible:
        return _TASK10_ROUTING_ROUTE_P5_PROBE
    return _TASK10_ROUTING_BLOCKED


def _task10_build_gate_contract(
    *,
    conditioning_probe: Mapping[str, Any],
    paired_probe: Mapping[str, Any],
    label_semantics_audit: Mapping[str, Any],
    output_dir: Path,
    label_semantics_output_dir: Path,
    full_update_scope_gate_pass: bool | None,
    comparability_manifest_pass: bool,
    comparability_blocker_reason: str | None,
    first_subgoal_probe: Mapping[str, Any] | None,
    continuous_numeric_advantage_dead_after_full_update: bool,
) -> dict[str, Any]:
    loss_probe_contract = _task10_build_loss_probe_contract(
        conditioning_probe=conditioning_probe,
        output_dir=output_dir,
    )
    paired_action_probe_contract = _task10_build_paired_action_probe_contract(
        paired_probe=paired_probe,
        output_dir=output_dir,
    )
    first_subgoal_probe_contract = _task10_build_first_subgoal_probe_contract(
        first_subgoal_probe
    )
    scope_gate = _task10_load_full_update_scope_gate(output_dir)
    if full_update_scope_gate_pass is None:
        full_update_scope_gate_pass = bool(scope_gate["full_update_scope_gate_pass"])

    strong_subgoal_progress_gate_pass = bool(
        first_subgoal_probe_contract["strong_subgoal_progress_gate_pass"]
    )
    weak_distance_only = bool(first_subgoal_probe_contract["weak_distance_only"])
    label_semantics_gate_pass = bool(
        label_semantics_audit.get("label_semantics_gate_pass", False)
    )
    shuffled_advantage_negative_control_pass = bool(
        isinstance(
            label_semantics_audit.get("shuffled_advantage_negative_control"),
            Mapping,
        )
        and label_semantics_audit["shuffled_advantage_negative_control"].get(
            "negative_control_pass"
        )
        is True
    )
    instrumentation_incomplete = bool(
        loss_probe_contract["instrumentation_incomplete"]
        or paired_action_probe_contract["instrumentation_incomplete"]
    )
    loss_sensitivity_gate_pass = bool(loss_probe_contract["loss_sensitivity_gate_pass"])
    action_sensitivity_gate_pass = bool(
        paired_action_probe_contract["action_sensitivity_gate_pass"]
    )
    p2_not_dead = bool(loss_sensitivity_gate_pass or action_sensitivity_gate_pass)
    p3_formal_training_eligible = bool(
        full_update_scope_gate_pass
        and not instrumentation_incomplete
        and p2_not_dead
        and shuffled_advantage_negative_control_pass
    )
    p5_probe_eligible = bool(
        full_update_scope_gate_pass
        and comparability_manifest_pass
        and not instrumentation_incomplete
        and (
            loss_sensitivity_gate_pass
            or action_sensitivity_gate_pass
            or strong_subgoal_progress_gate_pass
            or weak_distance_only
        )
    )
    p5_formal_10ep_eligible = bool(
        full_update_scope_gate_pass
        and comparability_manifest_pass
        and not instrumentation_incomplete
        and shuffled_advantage_negative_control_pass
        and (
            strong_subgoal_progress_gate_pass
            or (
                loss_sensitivity_gate_pass
                and action_sensitivity_gate_pass
                and label_semantics_gate_pass
            )
        )
    )
    p6_branch_eligible = bool(
        full_update_scope_gate_pass
        and comparability_manifest_pass
        and not instrumentation_incomplete
        and (
            not label_semantics_gate_pass
            or continuous_numeric_advantage_dead_after_full_update
            or weak_distance_only
        )
    )
    routing_reasons: list[str] = []
    routing_reasons.extend(loss_probe_contract["blocking_reasons"])
    routing_reasons.extend(paired_action_probe_contract["blocking_reasons"])
    routing_reasons.extend(first_subgoal_probe_contract["blocking_reasons"])
    if full_update_scope_gate_pass is not True:
        routing_reasons.append(
            str(scope_gate.get("scope_gate_reason") or "full_update_scope_gate_block")
        )
    if comparability_manifest_pass is not True:
        routing_reasons.append(
            str(comparability_blocker_reason or "comparability_manifest_block")
        )
    if not shuffled_advantage_negative_control_pass:
        routing_reasons.append("shuffled_advantage_negative_control_block")
    if not label_semantics_gate_pass:
        routing_reasons.append("label_semantics_gate_block")
    if continuous_numeric_advantage_dead_after_full_update:
        routing_reasons.append("continuous_numeric_advantage_dead_after_full_update")
    if weak_distance_only:
        routing_reasons.append("weak_distance_only")
    deduped_routing_reasons = list(dict.fromkeys(routing_reasons))
    return {
        "schema_version": "task10_preformal_gate_decision_v2",
        "artifact_kind": "preformal_gate_decision",
        "conditioning_probe_path": str(output_dir / _TASK8_STEP20_PROBE_FILENAME),
        "paired_action_probe_path": str(output_dir / _TASK8_PAIRED_STEP20_PROBE_FILENAME),
        "label_semantics_audit_path": str(
            label_semantics_output_dir / _TASK8_LABEL_SEMANTICS_AUDIT_FILENAME
        ),
        "loss_probe": loss_probe_contract,
        "paired_action_probe": paired_action_probe_contract,
        "first_subgoal_probe": first_subgoal_probe_contract,
        "loss_sensitivity_gate_pass": loss_sensitivity_gate_pass,
        "action_sensitivity_gate_pass": action_sensitivity_gate_pass,
        "instrumentation_incomplete": instrumentation_incomplete,
        "label_semantics_gate_pass": label_semantics_gate_pass,
        "shuffled_advantage_negative_control_pass": (
            shuffled_advantage_negative_control_pass
        ),
        "full_update_scope_gate_pass": bool(full_update_scope_gate_pass),
        "full_update_scope_gate_path": scope_gate["scope_gate_path"],
        "full_update_scope_gate_reason": scope_gate["scope_gate_reason"],
        "comparability_manifest_pass": bool(comparability_manifest_pass),
        "comparability_blocker_reason": (
            None if comparability_manifest_pass else comparability_blocker_reason
        ),
        "strong_subgoal_progress_gate_pass": strong_subgoal_progress_gate_pass,
        "weak_distance_only": weak_distance_only,
        "continuous_numeric_advantage_dead_after_full_update": bool(
            continuous_numeric_advantage_dead_after_full_update
        ),
        "p3_formal_training_eligible": p3_formal_training_eligible,
        "p3_skip_reason": _task10_p3_skip_reason(
            full_update_scope_gate_pass=bool(full_update_scope_gate_pass),
            instrumentation_incomplete=instrumentation_incomplete,
            p2_not_dead=p2_not_dead,
            shuffled_advantage_negative_control_pass=(
                shuffled_advantage_negative_control_pass
            ),
        ),
        "p5_probe_eligible": p5_probe_eligible,
        "p5_formal_10ep_eligible": p5_formal_10ep_eligible,
        "p6_branch_eligible": p6_branch_eligible,
        "routing_decision": _task10_routing_decision(
            p3_formal_training_eligible=p3_formal_training_eligible,
            p5_probe_eligible=p5_probe_eligible,
            p5_formal_10ep_eligible=p5_formal_10ep_eligible,
            p6_branch_eligible=p6_branch_eligible,
        ),
        "routing_reasons": deduped_routing_reasons,
    }


def _identity_postprocess_action(
    action_mapping: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str | None]:
    return copy.deepcopy(dict(action_mapping)), "identity_postprocess_passthrough"


def _make_controller_input_transform(
) -> Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], str | None]]:
    def _controller_input_transform(
        action_mapping: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], str | None]:
        return copy.deepcopy(dict(action_mapping)), "identity_controller_input_passthrough"

    return _controller_input_transform


def _task8_build_negative_control(
    *,
    model: Any,
    train_subset_bundle: Mapping[str, Any],
    heldout_bundle: Mapping[str, Any],
    seed: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        payload = {
            "enabled": False,
            "same_obs_action_batch": True,
            "only_advantage_shuffled": True,
            "per_subset": {},
        }
        payload.update(_evaluate_shuffled_negative_control(payload))
        return payload
    per_subset: dict[str, Any] = {}
    for offset, (subset_name, subset_bundle) in enumerate(
        (("train_subset", train_subset_bundle), ("heldout", heldout_bundle))
    ):
        true_advantages = list(subset_bundle["native_advantages"])
        shuffle_seed = int(seed) + int(offset)
        shuffle_plan = _task8_seeded_non_identity_permutation(
            len(true_advantages),
            seed=shuffle_seed,
        )
        shuffled_advantages = [
            true_advantages[index]
            for index in shuffle_plan["permutation_indices"]
        ]
        eval_seed = int(seed) + int(offset)
        true_loss = _task8_model_loss(
            model=model,
            collated_inputs=subset_bundle["collated_inputs"],
            advantage_override=true_advantages,
            seed=eval_seed,
        )
        shuffled_loss = _task8_model_loss(
            model=model,
            collated_inputs=subset_bundle["collated_inputs"],
            advantage_override=shuffled_advantages,
            seed=eval_seed,
        )
        per_subset[subset_name] = {
            "sample_refs": list(subset_bundle["sample_refs"]),
            "sample_count": int(subset_bundle.get("sample_count", len(true_advantages))),
            "true_advantages": true_advantages,
            "shuffled_advantages": shuffled_advantages,
            "true_label_loss": float(true_loss),
            "shuffled_loss": float(shuffled_loss),
            "shuffled_minus_true_loss": float(shuffled_loss - true_loss),
            "same_obs_action_batch": True,
            "only_advantage_shuffled": True,
            **shuffle_plan,
        }
    payload = {
        "enabled": True,
        "same_obs_action_batch": True,
        "only_advantage_shuffled": True,
        "shuffle_method": "seeded_non_identity_shuffle",
        "per_subset": per_subset,
    }
    payload.update(_evaluate_shuffled_negative_control(payload))
    return payload


def _task8_build_paired_action_probe(
    *,
    model: Any,
    obs_probe_sample: Mapping[str, Any],
    probe_seed: int,
    step_index: int,
) -> dict[str, Any]:
    from work.recap import policy as recap_policy

    collated_inputs = obs_probe_sample["collated_inputs"]
    baseline_capture = recap_policy.capture_local_diagnostic_action_stages(
        model=model,
        collated_inputs=collated_inputs,
        processor=obs_probe_sample["processor"],
        embodiment_tag=obs_probe_sample["embodiment_tag"],
        batched_states=obs_probe_sample["batched_states"],
        advantage_value=-1.0,
        seed=probe_seed,
        postprocess_action=_identity_postprocess_action,
        controller_input_transform=_make_controller_input_transform(),
    )
    probe_capture = recap_policy.capture_local_diagnostic_action_stages(
        model=model,
        collated_inputs=collated_inputs,
        processor=obs_probe_sample["processor"],
        embodiment_tag=obs_probe_sample["embodiment_tag"],
        batched_states=obs_probe_sample["batched_states"],
        advantage_value=1.0,
        seed=probe_seed,
        postprocess_action=_identity_postprocess_action,
        controller_input_transform=_make_controller_input_transform(),
    )
    layers = {
        "raw_normalized_action_delta": _task8_stage_action_delta(
            baseline_capture["raw_normalized_action"],
            probe_capture["raw_normalized_action"],
            layer_name="raw_normalized_action_delta",
        ),
        "decoded_action_delta": _task8_stage_action_delta(
            baseline_capture["decoded_action"],
            probe_capture["decoded_action"],
            layer_name="decoded_action_delta",
        ),
        "postprocessed_action_delta": _task8_stage_action_delta(
            baseline_capture["postprocessed_action"],
            probe_capture["postprocessed_action"],
            layer_name="postprocessed_action_delta",
        ),
        "controller_input_delta": _task8_stage_action_delta(
            baseline_capture["controller_input"],
            probe_capture["controller_input"],
            layer_name="controller_input_delta",
        ),
    }
    contract = _evaluate_paired_action_probe_contract({"action_delta_layers": layers})
    return {
        "schema_version": "task8_paired_action_probe_v1",
        "artifact_kind": "paired_action_probe",
        "step_index": int(step_index),
        "baseline_advantage": -1.0,
        "probe_advantage": 1.0,
        "sample_ref": dict(obs_probe_sample["sample_ref"]),
        "action_delta_layers": layers,
        **contract,
    }


def _task8_build_conditioning_probe(
    *,
    model: Any,
    obs_probe_sample: Mapping[str, Any],
    train_subset_bundle: Mapping[str, Any],
    heldout_bundle: Mapping[str, Any],
    step_index: int,
    probe_seed: int,
) -> dict[str, Any]:
    from work.recap import policy as recap_policy

    collated_inputs = obs_probe_sample["collated_inputs"]
    base_capture = recap_policy.capture_local_diagnostic_action_stages(
        model=model,
        collated_inputs=collated_inputs,
        processor=obs_probe_sample["processor"],
        embodiment_tag=obs_probe_sample["embodiment_tag"],
        batched_states=obs_probe_sample["batched_states"],
        advantage_value=None,
        seed=probe_seed,
        postprocess_action=_identity_postprocess_action,
        controller_input_transform=_make_controller_input_transform(),
    )
    conditioning_variants: dict[str, Any] = {}
    for offset, advantage_value in enumerate(_TASK8_ADVANTAGES_TESTED):
        variant_capture = recap_policy.capture_local_diagnostic_action_stages(
            model=model,
            collated_inputs=collated_inputs,
            processor=obs_probe_sample["processor"],
            embodiment_tag=obs_probe_sample["embodiment_tag"],
            batched_states=obs_probe_sample["batched_states"],
            advantage_value=advantage_value,
            seed=probe_seed,
            postprocess_action=_identity_postprocess_action,
            controller_input_transform=_make_controller_input_transform(),
        )
        conditioning_variants[_task8_advantage_key(advantage_value)] = {
            "advantage": advantage_value,
            "conditioning_preview": _task8_conditioning_preview(
                model=model,
                collated_inputs=collated_inputs,
                advantage_override=advantage_value,
            ),
            "raw_normalized_action_delta_vs_null": _task8_stage_action_delta(
                base_capture["raw_normalized_action"],
                variant_capture["raw_normalized_action"],
                layer_name="raw_normalized_action_delta_vs_null",
            ),
            "decoded_action_delta_vs_null": _task8_stage_action_delta(
                base_capture["decoded_action"],
                variant_capture["decoded_action"],
                layer_name="decoded_action_delta_vs_null",
            ),
        }
    train_subset_loss_probe = _task8_build_subset_loss_probe(
        subset_name="train_subset",
        model=model,
        subset_bundle=train_subset_bundle,
        probe_seed=probe_seed + 200,
    )
    heldout_loss_probe = _task8_build_subset_loss_probe(
        subset_name="heldout",
        model=model,
        subset_bundle=heldout_bundle,
        probe_seed=probe_seed + 400,
    )
    loss_sensitivity_gate_pass = bool(
        train_subset_loss_probe["loss_sensitive"] and heldout_loss_probe["loss_sensitive"]
    )
    raw_action_delta = conditioning_variants["1"]["raw_normalized_action_delta_vs_null"]
    decoded_action_delta = conditioning_variants["1"]["decoded_action_delta_vs_null"]
    action_sensitivity_gate_pass = bool(
        any(
            isinstance(layer.get("delta_l2"), (int, float)) and float(layer["delta_l2"]) > 0.0
            for layer in (raw_action_delta, decoded_action_delta)
        )
    )
    return {
        "schema_version": "task8_conditioning_functional_probe_v1",
        "artifact_kind": "conditioning_functional_probe",
        "step_index": int(step_index),
        "sample_ref": dict(obs_probe_sample["sample_ref"]),
        "advantages_tested": [
            _task8_json_ready(value) for value in _TASK8_ADVANTAGES_TESTED
        ],
        "conditioning_variants": conditioning_variants,
        "train_subset_loss_probe": train_subset_loss_probe,
        "heldout_loss_probe": heldout_loss_probe,
        "loss_sensitivity_gate_pass": loss_sensitivity_gate_pass,
        "action_sensitivity_gate_pass": action_sensitivity_gate_pass,
    }


def _task8_build_obs_probe_sample(trainer: Any, ref: Mapping[str, Any]) -> dict[str, Any]:
    sample = _task8_build_sample_from_ref(getattr(trainer.train_dataset, "datasets", []), ref=ref)
    collated_inputs = trainer.data_collator([sample["processed"]])
    preprocessing_hash = _task8_hash_payload(
        {
            "sample_ref": sample["ref"],
            "states": sample["vla_step_data"].states,
            "language": sample["vla_step_data"].text,
            "processed_keys": sorted(sample["processed"].keys()),
            "processed_shapes": {
                str(key): getattr(value, "shape", None)
                for key, value in sample["processed"].items()
            },
        }
    )
    return {
        "sample_ref": dict(sample["ref"]),
        "collated_inputs": collated_inputs,
        "processor": getattr(trainer.train_dataset.datasets[int(ref["dataset_index"])], "processor", None),
        "embodiment_tag": sample["embodiment_tag"],
        "batched_states": {
            str(key): value[None, ...]
            for key, value in sample["vla_step_data"].states.items()
        },
        "preprocessing_hash": preprocessing_hash,
    }


def _task8_probe_obs_manifest(
    *,
    args: argparse.Namespace,
    inventory: Mapping[str, Any],
    selected_refs: Mapping[str, Any],
    obs_probe_sample: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "task8_probe_obs_manifest_v1",
        "artifact_kind": "probe_obs_manifest",
        "obs_source": dict(obs_probe_sample["sample_ref"]),
        "seed": int(args.seed),
        "preprocessing_hash": str(obs_probe_sample["preprocessing_hash"]),
        "selection_strategy": str(selected_refs["selection_strategy"]),
        "selection_metadata": dict(selected_refs.get("selection_metadata", {})),
        "train_subset_refs": list(selected_refs["train_subset_refs"]),
        "heldout_refs": list(selected_refs["heldout_refs"]),
        "inventory_summary": dict(inventory["summary"]),
        "advantages_tested": [_task8_json_ready(value) for value in _TASK8_ADVANTAGES_TESTED],
        "illegal_extrapolation_advantages": list(_TASK8_ILLEGAL_EXTRAPOLATION_ADVANTAGES),
        "illegal_extrapolation_used_for_formal_gate": False,
    }


def _task8_write_json(output_dir: Path, filename: str, payload: Mapping[str, Any]) -> Path:
    path = output_dir / filename
    _write_json(path, dict(payload))
    return path


def _task8_conditioning_filename(step_index: int) -> str:
    if step_index == 0:
        return _TASK8_STEP0_PROBE_FILENAME
    if step_index == 1:
        return _TASK8_STEP1_PROBE_FILENAME
    return f"conditioning_functional_probe_step{int(step_index)}.json"


def _task8_paired_filename(step_index: int) -> str:
    if step_index == 0:
        return _TASK8_PAIRED_STEP0_PROBE_FILENAME
    return f"paired_action_probe_step{int(step_index)}.json"


def _task8_build_probe_session(*, trainer: Any, args: argparse.Namespace) -> dict[str, Any]:
    inventory = _task8_collect_dataset_inventory(trainer)
    selected_refs = _task8_pick_probe_refs(
        inventory,
        balanced=bool(getattr(args, "balanced_advantage_batches", False)),
    )
    train_subset_bundle = _task8_build_sample_bundle(
        trainer,
        selected_refs["train_subset_refs"],
    )
    heldout_bundle = _task8_build_sample_bundle(
        trainer,
        selected_refs["heldout_refs"],
    )
    obs_probe_sample = _task8_build_obs_probe_sample(
        trainer,
        selected_refs["obs_probe_ref"],
    )
    output_dir = _resolve_full_update_output_dir(_repo_root(), str(args.output_dir))
    label_semantics_output_dir = _resolve_label_semantics_output_dir(
        _repo_root(),
        output_dir=output_dir,
        raw=str(getattr(args, "label_semantics_output_dir", "")),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    label_semantics_output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _task8_probe_obs_manifest(
        args=args,
        inventory=inventory,
        selected_refs=selected_refs,
        obs_probe_sample=obs_probe_sample,
    )
    manifest_path = _task8_write_json(
        output_dir,
        _TASK8_PROBE_OBS_MANIFEST_FILENAME,
        manifest,
    )
    return {
        "args": args,
        "inventory": inventory,
        "selected_refs": selected_refs,
        "train_subset_bundle": train_subset_bundle,
        "heldout_bundle": heldout_bundle,
        "obs_probe_sample": obs_probe_sample,
        "output_dir": output_dir,
        "label_semantics_output_dir": label_semantics_output_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "step1_written": False,
    }


def _task8_write_step_probe(
    session: Mapping[str, Any],
    *,
    model: Any,
    step_index: int,
) -> dict[str, Path]:
    args = session["args"]
    written: dict[str, Path] = {}
    if bool(getattr(args, "write_conditioning_functional_probe", False)):
        conditioning_probe = _task8_build_conditioning_probe(
            model=model,
            obs_probe_sample=session["obs_probe_sample"],
            train_subset_bundle=session["train_subset_bundle"],
            heldout_bundle=session["heldout_bundle"],
            step_index=step_index,
            probe_seed=int(args.seed) + _TASK8_PROBE_SAMPLE_SEED_OFFSET + int(step_index),
        )
        written["conditioning"] = _task8_write_json(
            session["output_dir"],
            _task8_conditioning_filename(step_index),
            conditioning_probe,
        )
    if bool(getattr(args, "write_paired_action_probe", False)) and step_index in {0, int(args.max_steps)}:
        paired_probe = _task8_build_paired_action_probe(
            model=model,
            obs_probe_sample=session["obs_probe_sample"],
            probe_seed=int(args.seed) + _TASK8_PROBE_SAMPLE_SEED_OFFSET + int(step_index),
            step_index=step_index,
        )
        written["paired"] = _task8_write_json(
            session["output_dir"],
            _task8_paired_filename(step_index),
            paired_probe,
        )
    return written


def _task8_build_label_semantics_audit(
    session: Mapping[str, Any],
    *,
    model: Any,
    final_step_index: int,
) -> dict[str, Any]:
    args = session["args"]
    inventory_summary = dict(session["inventory"]["summary"])
    positive_count = int(inventory_summary.get("positive_count", 0))
    total_refs = int(inventory_summary.get("total_refs", 0))
    positive_success_rate = (
        float(positive_count) / float(total_refs) if total_refs > 0 else 0.0
    )
    shuffled_control = _task8_build_negative_control(
        model=model,
        train_subset_bundle=session["train_subset_bundle"],
        heldout_bundle=session["heldout_bundle"],
        seed=int(args.seed) + _TASK8_NEGATIVE_CONTROL_SEED_OFFSET + int(final_step_index),
        enabled=bool(getattr(args, "write_shuffled_advantage_negative_control", False)),
    )
    payload = {
        "schema_version": "task8_label_semantics_audit_v1",
        "artifact_kind": "label_semantics_audit",
        "step_index": int(final_step_index),
        "inventory_summary": inventory_summary,
        "probe_selection": dict(
            session["selected_refs"].get("selection_metadata", {})
        ),
        "value_is_constant": bool(
            inventory_summary.get("min_advantage") == inventory_summary.get("max_advantage")
        ),
        "all_returns_negative": bool(
            (inventory_summary.get("max_advantage") is not None)
            and float(inventory_summary.get("max_advantage") or 0.0) < 0.0
        ),
        "positive_success_rate": float(positive_success_rate),
        "positive_subgoal_evidence_available": bool(positive_count > 0),
        "shuffled_advantage_negative_control": shuffled_control,
    }
    payload.update(_evaluate_label_semantics_gate(payload))
    return payload


def _task8_build_preformal_gate_decision(
    *,
    conditioning_probe: Mapping[str, Any],
    paired_probe: Mapping[str, Any],
    label_semantics_audit: Mapping[str, Any],
    output_dir: Path,
    label_semantics_output_dir: Path,
    full_update_scope_gate_pass: bool | None = None,
    comparability_manifest_pass: bool = False,
    comparability_blocker_reason: str | None = "comparability_manifest_block",
    first_subgoal_probe: Mapping[str, Any] | None = None,
    continuous_numeric_advantage_dead_after_full_update: bool = False,
) -> dict[str, Any]:
    blocking_reasons: list[str] = []
    if conditioning_probe.get("loss_sensitivity_gate_pass") is not True:
        blocking_reasons.append("loss_sensitivity_gate_block")
    if paired_probe.get("instrumentation_incomplete") is True:
        blocking_reasons.append("paired_action_instrumentation_incomplete")
    if paired_probe.get("action_sensitivity_gate_pass") is not True:
        blocking_reasons.append("action_sensitivity_gate_block")
    if label_semantics_audit.get("label_semantics_gate_pass") is not True:
        blocking_reasons.append("label_semantics_gate_block")
    payload = _task10_build_gate_contract(
        conditioning_probe=conditioning_probe,
        paired_probe=paired_probe,
        label_semantics_audit=label_semantics_audit,
        output_dir=output_dir,
        label_semantics_output_dir=label_semantics_output_dir,
        full_update_scope_gate_pass=full_update_scope_gate_pass,
        comparability_manifest_pass=comparability_manifest_pass,
        comparability_blocker_reason=comparability_blocker_reason,
        first_subgoal_probe=first_subgoal_probe,
        continuous_numeric_advantage_dead_after_full_update=(
            continuous_numeric_advantage_dead_after_full_update
        ),
    )
    payload.update(
        {
            "illegal_extrapolation_used_for_formal_gate": False,
            "formal_claim_allowed": not blocking_reasons,
            "blocking_reasons": blocking_reasons,
        }
    )
    return payload


def _task8_write_semantics_outputs(session: Mapping[str, Any], *, model: Any, final_step_index: int) -> dict[str, Path]:
    if not bool(getattr(session["args"], "write_label_semantics_audit", False)):
        return {}
    label_semantics_audit = _task8_build_label_semantics_audit(
        session,
        model=model,
        final_step_index=final_step_index,
    )
    label_path = _task8_write_json(
        session["label_semantics_output_dir"],
        _TASK8_LABEL_SEMANTICS_AUDIT_FILENAME,
        label_semantics_audit,
    )
    conditioning_probe = _load_json_if_dict(
        session["output_dir"] / _TASK8_STEP20_PROBE_FILENAME
    ) or {}
    paired_probe = _load_json_if_dict(
        session["output_dir"] / _TASK8_PAIRED_STEP20_PROBE_FILENAME
    ) or {"instrumentation_incomplete": True, "action_sensitivity_gate_pass": False}
    preformal_gate = _task8_build_preformal_gate_decision(
        conditioning_probe=conditioning_probe,
        paired_probe=paired_probe,
        label_semantics_audit=label_semantics_audit,
        output_dir=session["output_dir"],
        label_semantics_output_dir=session["label_semantics_output_dir"],
    )
    gate_path = _task8_write_json(
        session["label_semantics_output_dir"],
        _TASK8_PREFORMAL_GATE_DECISION_FILENAME,
        preformal_gate,
    )
    return {
        "label_semantics_audit": label_path,
        "preformal_gate_decision": gate_path,
    }


def _build_training_arguments_for_preflight(config: Any) -> Any:
    from transformers import TrainingArguments

    if config.training.num_gpus > 1 and not config.training.use_ddp:
        deepspeed_config = config.get_deepspeed_config()
    else:
        deepspeed_config = None

    if config.training.batch_size is None:
        per_device_train_batch_size = config.training.global_batch_size // config.training.num_gpus
    else:
        per_device_train_batch_size = config.training.batch_size

    return TrainingArguments(
        output_dir=str(trainability_entrypoint._compute_effective_output_dir(config)),
        max_steps=config.training.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=config.training.tf32,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to="wandb" if config.training.use_wandb else "none",
        seed=config.data.seed,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=config.training.ddp_bucket_cap_mb,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        batch_eval_metrics=True,
        remove_unused_columns=config.training.remove_unused_columns,
        ignore_data_skip=True,
    )


def run_numeric_adv_static_scope_audit(
    args: argparse.Namespace,
    *,
    requested_scope_override: str | None = None,
) -> dict[str, Any]:
    effective_args = _clone_args_with_overrides(
        args,
        requested_scope_override=requested_scope_override,
        summary_json_override=None,
    )
    context = _prepare_wrapper_context(effective_args)
    repo_root = context["repo_root"]
    dataset_path = context["dataset_path"]
    output_dir = context["output_dir"]
    python_path = context["python_path"]

    _ensure_repo_imports(repo_root)

    from importlib import import_module
    import torch

    from work.recap.hf_snapshot_patch import make_patched_base_model_dir

    _install_transformers_compat_hooks()
    _install_local_no_wandb_import_shim(use_wandb=bool(effective_args.use_wandb))

    registry_module = import_module("gr00t.model.registry")
    trainer_module = import_module("gr00t.experiment.trainer")

    patched_dir = make_patched_base_model_dir(
        repo_id=str(effective_args.base_model),
        revision=str(effective_args.base_model_revision).strip() or None,
        out_root=str(effective_args.patched_out_root),
        overrides=_hf_snapshot_overrides_for_route(effective_args),
        hf_hub_cache_dir=(str(effective_args.hf_hub_cache_dir).strip() or None),
        emit_evidence=True,
    )
    _install_numeric_adv_monkeypatch(args=effective_args, repo_root=repo_root)
    config = _build_training_config(
        repo_root=repo_root,
        dataset_path=dataset_path,
        base_model_path=Path(str(patched_dir)).resolve(),
        args=effective_args,
    )
    config.validate()
    metadata_dir = trainability_entrypoint._repo_local_metadata_dir(config)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    save_cfg_dir = trainability_entrypoint._compute_effective_output_dir(config) / "experiment_cfg"
    save_cfg_dir.mkdir(parents=True, exist_ok=True)
    trainability_entrypoint.install_repo_local_rank_census_hooks(
        config=config,
        torch_module=torch,
    )

    model_registry = getattr(registry_module, "MODEL_REGISTRY")
    Gr00tTrainer = getattr(trainer_module, "Gr00tTrainer")
    pipeline = model_registry.get(type(config.model))(config, save_cfg_dir)
    pipeline.setup()
    model = pipeline.return_model()
    train_dataset, eval_dataset = pipeline.return_dataset()
    data_collator = pipeline.return_collator()
    trainer = Gr00tTrainer(
        model=model,
        args=_build_training_arguments_for_preflight(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        multiprocessing_context=config.data.multiprocessing_context,
    )
    trainer.create_optimizer()
    audit = getattr(trainer, "_repo_local_static_scope_audit", None) or getattr(
        model,
        "_repo_local_static_scope_audit",
        None,
    )
    audit_path = metadata_dir / trainability_entrypoint.STATIC_SCOPE_AUDIT_FILENAME
    if not isinstance(audit, Mapping):
        if not audit_path.is_file():
            raise RuntimeError("static_scope_audit_missing_after_preflight")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    runtime_preflight = trainability_entrypoint.build_repo_local_runtime_preflight_payload(
        output_dir=output_dir,
        python_path=python_path,
        requested_num_gpus=int(effective_args.num_gpus),
        requested_scope=str(effective_args.recap_train_scope),
        static_audit=dict(audit),
    )
    runtime_preflight_path = trainability_entrypoint.write_repo_local_runtime_preflight_json(
        output_dir=output_dir,
        payload=runtime_preflight,
    )
    return {
        "train_scope_requested": str(effective_args.recap_train_scope),
        "output_dir": str(output_dir),
        "metadata_dir": str(metadata_dir),
        "static_audit_path": str(audit_path),
        "static_audit": dict(audit),
        "runtime_preflight_path": str(runtime_preflight_path),
        "runtime_preflight": dict(runtime_preflight),
    }


def _wrapper_main(args: argparse.Namespace) -> int:
    if (
        str(args.recap_train_scope) in FULL_UPDATE_SCOPE_NAMES
        and not bool(getattr(args, "bypass_scope_supervisor", False))
    ):
        import importlib

        supervisor = importlib.import_module(
            "work.recap.scripts.34c_full_update_scope_supervisor"
        )
        return int(supervisor.run_numeric_adv_scope_supervisor(args))

    rc, payload = run_numeric_adv_single_scope(args, emit_summary=True)
    return 0 if int(rc) == 0 and payload["wrapper_status"] == "ok" else 1


def _ensure_repo_imports(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    gr00t_root_str = str((repo_root / "submodules" / "Isaac-GR00T").resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    if gr00t_root_str not in sys.path:
        sys.path.insert(0, gr00t_root_str)


def _install_transformers_compat_hooks() -> None:
    from work.recap import transformers_compat as transformers_compat

    transformers_compat.install_transformers_image_processor_fast_compat()

    from transformers.models.auto import auto_factory

    if getattr(auto_factory, "_gr00t_siglip2_dynamic_modeling_compat_installed", False):
        return

    for helper_name in (
        "_install_eagle3_dynamic_config_compat",
        "_install_eagle3_dynamic_modeling_compat",
        "_install_eagle3_dynamic_processor_compat",
        "_install_siglip2_dynamic_modeling_compat",
    ):
        helper = getattr(transformers_compat, helper_name, None)
        if callable(helper):
            helper()


def _resolve_upstream_embodiment_tag(raw_tag: str) -> str:
    from importlib import import_module

    embodiment_configs = import_module("gr00t.configs.data.embodiment_configs")
    embodiment_tags = import_module("gr00t.data.embodiment_tags")
    modality_configs = getattr(embodiment_configs, "MODALITY_CONFIGS")
    embodiment_tag_enum = getattr(embodiment_tags, "EmbodimentTag")

    tag = str(raw_tag).strip()
    if not tag:
        return tag
    if tag in modality_configs:
        return tag
    try:
        return embodiment_tag_enum[tag].value
    except KeyError:
        return tag


def _build_training_config(
    *,
    repo_root: Path,
    dataset_path: Path,
    base_model_path: Path,
    args: argparse.Namespace,
) -> Any:
    from importlib import import_module

    base_config = import_module("gr00t.configs.base_config")
    embodiment_configs = import_module("gr00t.configs.data.embodiment_configs")
    get_default_config = getattr(base_config, "get_default_config")
    modality_configs = getattr(embodiment_configs, "MODALITY_CONFIGS")

    embodiment_tag = _resolve_upstream_embodiment_tag(str(args.embodiment_tag))

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(dataset_path)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    config.data.modality_configs = dict(modality_configs)
    scope_summary = _scope_summary_for_args(args)
    trainability_authority = _build_trainability_authority_from_args(
        args,
        scope_summary=scope_summary,
    )
    trainability_entrypoint.attach_repo_local_trainability_authority(
        config=config,
        authority=trainability_authority,
    )
    effective_tuning_flags = _effective_tuning_flags(args)

    config.model = _rebuild_model_config_for_smoke(
        config.model,
        tuning_flags=effective_tuning_flags,
    )
    config.model.tune_llm = bool(effective_tuning_flags["tune_llm"])
    config.model.tune_visual = bool(effective_tuning_flags["tune_visual"])
    config.model.tune_top_llm_layers = int(
        effective_tuning_flags["tune_top_llm_layers"]
    )
    config.model.tune_projector = bool(effective_tuning_flags["tune_projector"])
    config.model.tune_diffusion_model = bool(
        effective_tuning_flags["tune_diffusion_model"]
    )
    config.model.tune_vlln = bool(effective_tuning_flags["tune_vlln"])
    config.model.state_dropout_prob = getattr(config.model, "state_dropout_prob", 0.0)
    if _text_indicator_route_enabled(args):
        config.model.formalize_language = False
    config.model.random_rotation_angle = getattr(
        config.model, "random_rotation_angle", None
    )
    config.model.color_jitter_params = getattr(
        config.model, "color_jitter_params", None
    )

    # Preserve the upstream Eagle default here. The numeric-adv smoke only swaps
    # the model construction path, and forcing fp32 loading trips Eagle's
    # required bfloat16 assertion before training can start.
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    continuation_checkpoint_path = _resolve_optional_checkpoint_path(repo_root, args)
    continuation_source_step = 0
    requested_additional_steps = _continuation_requested_steps(args)
    one_step_verifier_overrides = _resolve_one_step_verifier_training_overrides(
        args,
        requested_additional_steps=requested_additional_steps,
    )
    if _condition_focused_continuation_enabled(args):
        assert continuation_checkpoint_path is not None
        continuation_source_step = _checkpoint_resume_step(continuation_checkpoint_path)
        config.training.start_from_checkpoint = str(continuation_checkpoint_path)
        _emit_info_line(
            "condition-focused continuation enabled "
            "mode=warm_start_weights_only "
            f"warm_start_checkpoint={continuation_checkpoint_path} "
            f"source_global_step={continuation_source_step} "
            f"requested_additional_steps={requested_additional_steps} "
            f"fresh_run_max_steps={requested_additional_steps}"
        )
    else:
        config.training.start_from_checkpoint = str(base_model_path)
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = int(args.global_batch_size)
    config.training.batch_size = (
        None
        if args.per_device_batch_size is None
        else int(args.per_device_batch_size)
    )
    config.training.dataloader_num_workers = int(args.dataloader_num_workers)
    config.training.learning_rate = float(args.learning_rate)
    config.training.gradient_accumulation_steps = int(
        one_step_verifier_overrides["gradient_accumulation_steps"]
    )
    config.training.output_dir = str(
        _resolve_full_update_output_dir(repo_root, str(args.output_dir))
    )
    config.training.save_steps = int(args.save_steps)
    config.training.save_total_limit = int(args.save_total_limit)
    config.training.num_gpus = int(args.num_gpus)
    config.training.use_wandb = bool(args.use_wandb)
    config.training.max_steps = int(requested_additional_steps)
    if args.bf16 is not None:
        config.training.bf16 = bool(args.bf16)
    if args.gradient_checkpointing is not None:
        config.training.gradient_checkpointing = bool(args.gradient_checkpointing)
    config.training.weight_decay = getattr(config.training, "weight_decay", 0.0)
    config.training.lr_scheduler_type = str(
        one_step_verifier_overrides["lr_scheduler_type"]
        or getattr(config.training, "lr_scheduler_type", "cosine")
    )
    config.training.warmup_ratio = float(
        one_step_verifier_overrides["warmup_ratio"]
        if one_step_verifier_overrides["warmup_ratio"] is not None
        else getattr(config.training, "warmup_ratio", 0.03)
    )
    config.training.wandb_project = "finetune-gr00t-n1d6"
    config.training.transformers_local_files_only = bool(
        args.transformers_local_files_only
    )
    config.training.transformers_trust_remote_code = True
    config.training.transformers_cache_dir = (
        None
        if not str(args.hf_hub_cache_dir).strip()
        else str(args.hf_hub_cache_dir).strip()
    )

    config.data.seed = int(args.seed)
    config.data.shard_size = getattr(config.data, "shard_size", 128)
    config.data.episode_sampling_rate = getattr(
        config.data, "episode_sampling_rate", 1.0
    )
    config.data.num_shards_per_epoch = getattr(
        config.data, "num_shards_per_epoch", None
    )
    logging.info(
        "numeric-adv smoke effective_tuning_flags=%s", _effective_tuning_flags(args)
    )
    if _condition_focused_continuation_enabled(args):
        _emit_info_line(
            "condition-focused continuation config_override "
            "mode=warm_start_weights_only "
            f"start_from_checkpoint={config.training.start_from_checkpoint} "
            f"tune_vlln={bool(config.model.tune_vlln)} "
            f"max_steps_additional={int(config.training.max_steps)} "
            "fresh_optimizer_state=True"
        )
    if bool(one_step_verifier_overrides["override_active"]):
        _emit_info_line(
            "one-step verifier config_override "
            f"train_scope={args.recap_train_scope} "
            f"gradient_accumulation_steps={config.training.gradient_accumulation_steps} "
            f"lr_scheduler_type={config.training.lr_scheduler_type} "
            f"warmup_ratio={config.training.warmup_ratio}"
        )
    return config


def _load_checkpoint_into_recap_model(
    model: Any, checkpoint_path: Path
) -> dict[str, Any]:
    import json

    import torch

    target_state_dict = model.state_dict()
    loaded_key_names: set[str] = set()
    checkpoint_files: list[Path] = []

    if checkpoint_path.is_dir():
        safe_index = checkpoint_path / "model.safetensors.index.json"
        torch_index = checkpoint_path / "pytorch_model.bin.index.json"
        if safe_index.is_file() or torch_index.is_file():
            index_path = safe_index if safe_index.is_file() else torch_index
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map_raw = index_payload.get("weight_map")
            if not isinstance(weight_map_raw, dict) or not weight_map_raw:
                raise ValueError(
                    f"Checkpoint index missing non-empty weight_map: {index_path}"
                )
            shard_names = sorted({str(name) for name in weight_map_raw.values()})
            checkpoint_files = [
                checkpoint_path / shard_name for shard_name in shard_names
            ]
        single_candidates = [
            checkpoint_path / "model.safetensors",
            checkpoint_path / "pytorch_model.bin",
        ]
        if not checkpoint_files:
            for candidate in single_candidates:
                if candidate.is_file():
                    checkpoint_files = [candidate]
                    break
        if not checkpoint_files:
            raise FileNotFoundError(
                f"Cannot locate checkpoint weights under directory: {checkpoint_path}"
            )
    else:
        checkpoint_files = [checkpoint_path]

    report: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_files": [str(path) for path in checkpoint_files],
        "loaded_keys": [],
        "loaded_key_count": 0,
        "missing_keys": [],
        "unexpected_keys": [],
        "skipped_mismatched_keys": [],
        "skipped_mismatched_key_details": [],
        "blocked_mismatched_keys": [],
        "blocked_mismatched_key_details": [],
        "allowed_mismatched_key_prefixes": list(ALLOWED_MISMATCHED_KEY_PREFIXES),
    }

    with torch.no_grad():
        for state_file in checkpoint_files:
            if not state_file.is_file():
                raise FileNotFoundError(f"Checkpoint shard not found: {state_file}")
            state_dict = _load_state_dict_file(state_file)
            for key, value in state_dict.items():
                if key not in target_state_dict:
                    report["unexpected_keys"].append(str(key))
                    continue
                model_value = target_state_dict[key]
                checkpoint_shape = _shape_list(value)
                model_shape = _shape_list(model_value)
                if checkpoint_shape != model_shape:
                    report["skipped_mismatched_keys"].append(str(key))
                    report["skipped_mismatched_key_details"].append(
                        {
                            "key": str(key),
                            "checkpoint_shape": checkpoint_shape,
                            "model_shape": model_shape,
                        }
                    )
                    continue
                model_value.copy_(value)
                loaded_key_names.add(str(key))

    report["loaded_keys"] = sorted(loaded_key_names)
    report["loaded_key_count"] = len(loaded_key_names)
    report["missing_keys"] = sorted(
        key for key in target_state_dict.keys() if key not in loaded_key_names
    )
    disallowed_mismatches = [
        entry
        for entry in report["skipped_mismatched_key_details"]
        if not str(entry.get("key", "")).startswith(ALLOWED_MISMATCHED_KEY_PREFIXES)
    ]
    report["skipped_mismatched_keys"] = sorted(report["skipped_mismatched_keys"])
    report["skipped_mismatched_key_details"] = sorted(
        report["skipped_mismatched_key_details"],
        key=lambda entry: str(entry.get("key", "")),
    )
    report["blocked_mismatched_keys"] = sorted(
        str(entry.get("key", "")) for entry in disallowed_mismatches
    )
    report["blocked_mismatched_key_details"] = list(disallowed_mismatches)
    if hasattr(model, "last_load_incompatible_keys"):
        model.last_load_incompatible_keys = report
    _write_checkpoint_load_report(report)
    if disallowed_mismatches:
        mismatch_summary = ", ".join(
            (
                f"{entry['key']} ckpt={entry['checkpoint_shape']} "
                f"model={entry['model_shape']}"
            )
            for entry in disallowed_mismatches
        )
        raise RuntimeError(
            "Checkpoint shape mismatches escaped the allowed action-head scope: "
            f"{mismatch_summary}"
        )
    return report


def _delegate_audit_env_int(name: str, *, default: int = 0) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _delegate_audit_is_rank0() -> bool:
    return _delegate_audit_env_int("RANK", default=0) == 0


def _delegate_audit_metadata_dir(trainer: Any) -> Path | None:
    metadata_dir = trainability_entrypoint.REPO_LOCAL_CENSUS_HOOK_STATE.get(
        "metadata_dir"
    )
    if metadata_dir is not None:
        return Path(metadata_dir)

    config = getattr(trainer, "config", None)
    training = None if config is None else getattr(config, "training", None)
    output_dir = None if training is None else getattr(training, "output_dir", None)
    if output_dir is None:
        trainer_args = getattr(trainer, "args", None)
        output_dir = (
            None if trainer_args is None else getattr(trainer_args, "output_dir", None)
        )
    if output_dir is None:
        return None
    return trainability_entrypoint.resolve_repo_local_metadata_dir_for_output_dir(
        Path(str(output_dir))
    )


def _delegate_audit_output_dir(trainer: Any) -> Path | None:
    metadata_dir = _delegate_audit_metadata_dir(trainer)
    if metadata_dir is None:
        return None
    return metadata_dir.parent


def _delegate_audit_report_path(trainer: Any, filename: str) -> Path | None:
    metadata_dir = _delegate_audit_metadata_dir(trainer)
    if metadata_dir is None:
        return None
    return metadata_dir / filename


def _delegate_audit_canonical_name(name: object) -> str:
    normalized = str(name)
    while normalized.startswith("module."):
        normalized = normalized[len("module.") :]
    return normalized


def _delegate_audit_model_candidates(trainer: Any) -> list[Any]:
    candidates = [
        getattr(trainer, "model", None),
        getattr(trainer, "model_wrapped", None),
    ]
    for candidate in tuple(candidates):
        candidates.append(getattr(candidate, "module", None))
    seen: set[int] = set()
    unique: list[Any] = []
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        unique.append(candidate)
    return unique


def _delegate_audit_named_parameter_records(trainer: Any) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for model in _delegate_audit_model_candidates(trainer):
        named_parameters = getattr(model, "named_parameters", None)
        if not callable(named_parameters):
            continue
        for raw_name, param in named_parameters():
            param_id = id(param)
            if param_id in records:
                continue
            records[param_id] = {
                "name": _delegate_audit_canonical_name(raw_name),
                "param": param,
            }
    return records


def _delegate_audit_trainable_named_parameters(trainer: Any) -> dict[str, Any]:
    trainable: dict[str, Any] = {}
    for record in _delegate_audit_named_parameter_records(trainer).values():
        param = record["param"]
        if not bool(getattr(param, "requires_grad", False)):
            continue
        trainable[str(record["name"])] = param
    return trainable


def _delegate_audit_tensor_shape(tensor: Any) -> list[int]:
    return _shape_list(tensor)


def _delegate_audit_tensor_l2_norm(tensor: Any) -> float | None:
    if tensor is None:
        return None
    detached = tensor.detach().float()
    if int(detached.numel()) == 0:
        return 0.0
    return float(math.sqrt(float(detached.square().sum().item())))


def _delegate_audit_module_bucket(name: str) -> str:
    prefixes = (
        "action_head.advantage_embedding",
        "action_head.model",
        "action_head.vlln",
        "action_head.projector",
        "projector",
        "vla_action_interface",
    )
    for prefix in prefixes:
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    parts = name.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0] if parts and parts[0] else "<unnamed>"


def _delegate_audit_optimizer_param_group_report(
    *, trainer: Any, optimizer: Any
) -> dict[str, Any]:
    named_records = _delegate_audit_named_parameter_records(trainer)
    name_by_id = {
        param_id: str(record["name"]) for param_id, record in named_records.items()
    }
    trainable_params = _delegate_audit_trainable_named_parameters(trainer)
    optimizer_param_ids: set[int] = set()
    groups: list[dict[str, Any]] = []

    for group_index, group in enumerate(list(getattr(optimizer, "param_groups", []))):
        raw_params = list(group.get("params", []))
        matched_names = [name_by_id.get(id(param), "<unnamed>") for param in raw_params]
        optimizer_param_ids.update(id(param) for param in raw_params)
        trainable_count = sum(
            1 for param in raw_params if bool(getattr(param, "requires_grad", False))
        )
        matched_modules = sorted(
            {
                _delegate_audit_module_bucket(name)
                for name in matched_names
                if name != "<unnamed>"
            }
        )
        groups.append(
            {
                "group_id": int(group_index),
                "lr": None if group.get("lr") is None else float(group.get("lr")),
                "weight_decay": (
                    None
                    if group.get("weight_decay") is None
                    else float(group.get("weight_decay"))
                ),
                "param_count": int(len(raw_params)),
                "trainable_param_count": int(trainable_count),
                "matched_modules": matched_modules,
                "sample_parameter_names": matched_names[:OPTIMIZER_GROUP_SAMPLE_LIMIT],
            }
        )

    missing_trainable_names = sorted(
        name
        for name, param in trainable_params.items()
        if id(param) not in optimizer_param_ids
    )
    optimizer_named_trainable_names = sorted(
        name_by_id[param_id]
        for param_id in optimizer_param_ids
        if param_id in name_by_id
        and bool(getattr(named_records[param_id]["param"], "requires_grad", False))
    )
    return {
        "schema_version": "gr00t_optimizer_param_group_report_v1",
        "status": "PASS",
        "measurement_mode": "live_delegate_optimizer_inspection",
        "rank": _delegate_audit_env_int("RANK", default=0),
        "local_rank": _delegate_audit_env_int("LOCAL_RANK", default=0),
        "world_size": _delegate_audit_env_int("WORLD_SIZE", default=0),
        "runtime_pid": int(os.getpid()),
        "trainer_global_step": int(
            getattr(getattr(trainer, "state", None), "global_step", 0) or 0
        ),
        "optimizer_param_groups": groups,
        "optimizer_param_group_count": int(len(groups)),
        "named_trainable_param_count": int(len(trainable_params)),
        "optimizer_named_trainable_param_count": int(
            len(optimizer_named_trainable_names)
        ),
        "trainable_params_missing_from_optimizer_count": int(
            len(missing_trainable_names)
        ),
        "trainable_params_missing_from_optimizer": missing_trainable_names,
    }


def _delegate_audit_emit_optimizer_param_group_report(
    *, trainer: Any, optimizer: Any
) -> Path | None:
    if not _delegate_audit_is_rank0():
        return None
    report_path = _delegate_audit_report_path(
        trainer,
        OPTIMIZER_PARAM_GROUP_REPORT_FILENAME,
    )
    if report_path is None:
        return None
    payload = _delegate_audit_optimizer_param_group_report(
        trainer=trainer,
        optimizer=optimizer,
    )
    _write_json(report_path, payload)
    return report_path


def _delegate_audit_capture_trainable_pre_step_snapshot(
    *, trainer: Any
) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, param in _delegate_audit_trainable_named_parameters(trainer).items():
        grad = getattr(param, "grad", None)
        snapshot[name] = {
            "shape": _delegate_audit_tensor_shape(param),
            "dtype": str(getattr(param, "dtype", "")),
            "requires_grad": bool(getattr(param, "requires_grad", False)),
            "grad_norm_after_backward": _delegate_audit_tensor_l2_norm(grad),
            "tensor": param.detach().float().cpu().clone(),
        }
    return snapshot


def _delegate_audit_delta_metrics(
    *, current_tensor: Any, before_tensor: Any
) -> dict[str, Any]:
    delta = current_tensor.detach().float().cpu() - before_tensor
    if int(delta.numel()) == 0:
        return {
            "delta_l2": 0.0,
            "delta_linf": 0.0,
            "delta_abs_sum": 0.0,
            "nonzero_delta_count": 0,
        }
    abs_delta = delta.abs()
    return {
        "delta_l2": float(math.sqrt(float(delta.square().sum().item()))),
        "delta_linf": float(abs_delta.max().item()),
        "delta_abs_sum": float(abs_delta.sum().item()),
        "nonzero_delta_count": int(delta.count_nonzero().item()),
    }


def _delegate_audit_build_in_memory_delta_report(
    *,
    trainer: Any,
    pre_step_snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    current_params = _delegate_audit_trainable_named_parameters(trainer)
    parameter_deltas: list[dict[str, Any]] = []
    delta_l2_square_sum = 0.0
    delta_abs_sum = 0.0
    delta_linf_max = 0.0
    nonzero_param_count = 0
    nonzero_element_count = 0

    for name in sorted(pre_step_snapshot.keys()):
        before_record = dict(pre_step_snapshot[name])
        before_tensor = before_record.get("tensor")
        current_param = current_params.get(name)
        record: dict[str, Any] = {
            "name": name,
            "shape": list(before_record.get("shape", [])),
            "dtype": str(before_record.get("dtype", "")),
            "requires_grad": bool(before_record.get("requires_grad", False)),
            "grad_norm_after_backward": before_record.get("grad_norm_after_backward"),
            "delta_l2_after_optimizer_step": None,
            "delta_linf_after_optimizer_step": None,
            "nonzero_delta_count": None,
            "current_parameter_found": current_param is not None,
        }
        if current_param is not None and before_tensor is not None:
            metrics = _delegate_audit_delta_metrics(
                current_tensor=current_param,
                before_tensor=before_tensor,
            )
            record["delta_l2_after_optimizer_step"] = metrics["delta_l2"]
            record["delta_linf_after_optimizer_step"] = metrics["delta_linf"]
            record["delta_sum_abs_after_optimizer_step"] = metrics["delta_abs_sum"]
            record["nonzero_delta_count"] = metrics["nonzero_delta_count"]
            delta_l2_square_sum += float(metrics["delta_l2"]) ** 2
            delta_abs_sum += float(metrics["delta_abs_sum"])
            delta_linf_max = max(delta_linf_max, float(metrics["delta_linf"]))
            if int(metrics["nonzero_delta_count"]) > 0:
                nonzero_param_count += 1
                nonzero_element_count += int(metrics["nonzero_delta_count"])
        parameter_deltas.append(record)

    return {
        "schema_version": "gr00t_in_memory_delta_report_v1",
        "status": "PASS",
        "measurement_mode": "live_delegate_full_named_trainable_parameter_delta",
        "rank": _delegate_audit_env_int("RANK", default=0),
        "local_rank": _delegate_audit_env_int("LOCAL_RANK", default=0),
        "world_size": _delegate_audit_env_int("WORLD_SIZE", default=0),
        "runtime_pid": int(os.getpid()),
        "trainer_global_step": int(
            getattr(getattr(trainer, "state", None), "global_step", 0) or 0
        )
        + 1,
        "parameter_deltas": parameter_deltas,
        "aggregate_summary": {
            "parameter_record_count": int(len(parameter_deltas)),
            "nonzero_delta_parameter_count": int(nonzero_param_count),
            "nonzero_delta_element_count": int(nonzero_element_count),
            "delta_l2_global": float(math.sqrt(delta_l2_square_sum)),
            "delta_linf_max": float(delta_linf_max),
            "delta_abs_sum": float(delta_abs_sum),
        },
    }


def _delegate_audit_checkpoint_weight_locations(
    checkpoint_dir: Path,
    names: set[str],
) -> dict[Path, set[str]]:
    shard_to_names: dict[Path, set[str]] = {}
    safe_index = checkpoint_dir / "model.safetensors.index.json"
    torch_index = checkpoint_dir / "pytorch_model.bin.index.json"
    if safe_index.is_file() or torch_index.is_file():
        index_path = safe_index if safe_index.is_file() else torch_index
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, Mapping):
            return {}
        for name in names:
            shard_name = weight_map.get(name)
            if isinstance(shard_name, str) and shard_name.strip():
                shard_to_names.setdefault(checkpoint_dir / shard_name, set()).add(name)
        return shard_to_names

    single_safetensors = checkpoint_dir / "model.safetensors"
    single_torch = checkpoint_dir / "pytorch_model.bin"
    if single_safetensors.is_file():
        shard_to_names[single_safetensors] = set(names)
    elif single_torch.is_file():
        shard_to_names[single_torch] = set(names)
    return shard_to_names


def _delegate_audit_checkpoint_tensors(
    checkpoint_dir: Path,
    names: set[str],
) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for shard_path, shard_names in _delegate_audit_checkpoint_weight_locations(
        checkpoint_dir,
        names,
    ).items():
        if not shard_path.is_file():
            continue
        if shard_path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for name in sorted(shard_names):
                    if name in available:
                        tensors[name] = handle.get_tensor(name).float().cpu()
            continue
        state_dict = _load_state_dict_file(shard_path)
        for name in sorted(shard_names):
            value = state_dict.get(name)
            if value is not None:
                tensors[name] = value.detach().float().cpu()
    return tensors


def _delegate_audit_empty_checkpoint_delta_report(
    *, status: str, reason: str, checkpoint_dir: Path | None
) -> dict[str, Any]:
    return {
        "schema_version": "gr00t_saved_checkpoint_reload_delta_report_v1",
        "status": status,
        "measurement_mode": "live_delegate_checkpoint_reload_delta",
        "blocking_reasons": [reason],
        "checkpoint_path": None if checkpoint_dir is None else str(checkpoint_dir),
        "parameter_deltas": [],
        "aggregate_summary": {
            "parameter_record_count": 0,
            "nonzero_delta_parameter_count": 0,
            "nonzero_delta_element_count": 0,
            "delta_l2_global": 0.0,
            "delta_linf_max": 0.0,
            "delta_abs_sum": 0.0,
        },
    }


def _delegate_audit_build_saved_checkpoint_delta_report(
    *,
    trainer: Any,
    pre_step_snapshot: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    output_dir = _delegate_audit_output_dir(trainer)
    checkpoint_dir = None if output_dir is None else _latest_checkpoint(output_dir)
    if pre_step_snapshot is None:
        return _delegate_audit_empty_checkpoint_delta_report(
            status="BLOCK",
            reason="first_optimizer_step_pre_snapshot_missing",
            checkpoint_dir=checkpoint_dir,
        )
    if checkpoint_dir is None:
        return _delegate_audit_empty_checkpoint_delta_report(
            status="BLOCK",
            reason="saved_checkpoint_missing",
            checkpoint_dir=None,
        )

    names = {str(name) for name in pre_step_snapshot.keys()}
    checkpoint_tensors = _delegate_audit_checkpoint_tensors(checkpoint_dir, names)
    parameter_deltas: list[dict[str, Any]] = []
    delta_l2_square_sum = 0.0
    delta_abs_sum = 0.0
    delta_linf_max = 0.0
    nonzero_param_count = 0
    nonzero_element_count = 0

    for name in sorted(names):
        before_record = dict(pre_step_snapshot[name])
        before_tensor = before_record.get("tensor")
        checkpoint_tensor = checkpoint_tensors.get(name)
        record: dict[str, Any] = {
            "name": name,
            "shape": list(before_record.get("shape", [])),
            "dtype": str(before_record.get("dtype", "")),
            "requires_grad": bool(before_record.get("requires_grad", False)),
            "checkpoint_key_found": checkpoint_tensor is not None,
            "delta_l2_after_checkpoint_reload": None,
            "delta_linf_after_checkpoint_reload": None,
            "nonzero_delta_count": None,
        }
        if checkpoint_tensor is not None and before_tensor is not None:
            metrics = _delegate_audit_delta_metrics(
                current_tensor=checkpoint_tensor,
                before_tensor=before_tensor,
            )
            record["checkpoint_dtype"] = str(getattr(checkpoint_tensor, "dtype", ""))
            record["delta_l2_after_checkpoint_reload"] = metrics["delta_l2"]
            record["delta_linf_after_checkpoint_reload"] = metrics["delta_linf"]
            record["delta_sum_abs_after_checkpoint_reload"] = metrics["delta_abs_sum"]
            record["nonzero_delta_count"] = metrics["nonzero_delta_count"]
            delta_l2_square_sum += float(metrics["delta_l2"]) ** 2
            delta_abs_sum += float(metrics["delta_abs_sum"])
            delta_linf_max = max(delta_linf_max, float(metrics["delta_linf"]))
            if int(metrics["nonzero_delta_count"]) > 0:
                nonzero_param_count += 1
                nonzero_element_count += int(metrics["nonzero_delta_count"])
        parameter_deltas.append(record)

    selected_asset = _selected_checkpoint_asset(checkpoint_dir)
    return {
        "schema_version": "gr00t_saved_checkpoint_reload_delta_report_v1",
        "status": "PASS",
        "measurement_mode": "live_delegate_checkpoint_reload_delta",
        "rank": _delegate_audit_env_int("RANK", default=0),
        "local_rank": _delegate_audit_env_int("LOCAL_RANK", default=0),
        "world_size": _delegate_audit_env_int("WORLD_SIZE", default=0),
        "runtime_pid": int(os.getpid()),
        "trainer_global_step": int(
            getattr(getattr(trainer, "state", None), "global_step", 0) or 0
        ),
        "checkpoint_path": str(checkpoint_dir),
        "checkpoint_asset_path": None if selected_asset is None else str(selected_asset),
        "parameter_deltas": parameter_deltas,
        "aggregate_summary": {
            "parameter_record_count": int(len(parameter_deltas)),
            "checkpoint_key_found_count": int(len(checkpoint_tensors)),
            "checkpoint_key_missing_count": int(len(names) - len(checkpoint_tensors)),
            "nonzero_delta_parameter_count": int(nonzero_param_count),
            "nonzero_delta_element_count": int(nonzero_element_count),
            "delta_l2_global": float(math.sqrt(delta_l2_square_sum)),
            "delta_linf_max": float(delta_linf_max),
            "delta_abs_sum": float(delta_abs_sum),
        },
    }


def _delegate_audit_write_in_memory_delta_report(
    *,
    trainer: Any,
    pre_step_snapshot: Mapping[str, Mapping[str, Any]],
) -> Path | None:
    if not _delegate_audit_is_rank0():
        return None
    report_path = _delegate_audit_report_path(
        trainer,
        IN_MEMORY_DELTA_REPORT_FILENAME,
    )
    if report_path is None:
        return None
    payload = _delegate_audit_build_in_memory_delta_report(
        trainer=trainer,
        pre_step_snapshot=pre_step_snapshot,
    )
    _write_json(report_path, payload)
    return report_path


def _delegate_audit_write_saved_checkpoint_delta_report(
    *, trainer: Any
) -> Path | None:
    if not _delegate_audit_is_rank0():
        return None
    report_path = _delegate_audit_report_path(
        trainer,
        SAVED_CHECKPOINT_RELOAD_DELTA_REPORT_FILENAME,
    )
    if report_path is None:
        return None
    snapshot = _DELEGATE_DELTA_AUDIT_STATE.get("first_step_snapshot")
    payload = _delegate_audit_build_saved_checkpoint_delta_report(
        trainer=trainer,
        pre_step_snapshot=snapshot if isinstance(snapshot, Mapping) else None,
    )
    _write_json(report_path, payload)
    return report_path


def _delegate_audit_should_capture_first_step(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "emit_in_memory_delta_report", False)) or bool(
        getattr(args, "emit_saved_checkpoint_delta_report", False)
    )


def _delegate_audit_maybe_wrap_first_step(
    *, trainer: Any, optimizer: Any, args: argparse.Namespace
) -> Any:
    if optimizer is None or not _delegate_audit_should_capture_first_step(args):
        return optimizer
    if bool(getattr(optimizer, "_delegate_delta_audit_step_wrapped", False)):
        return optimizer
    original_step = getattr(optimizer, "step", None)
    if not callable(original_step):
        return optimizer

    def _delegate_delta_audit_step(*step_args: Any, **step_kwargs: Any) -> Any:
        should_capture = (
            _delegate_audit_is_rank0()
            and not bool(_DELEGATE_DELTA_AUDIT_STATE["first_step_written"])
        )
        pre_step_snapshot = (
            _delegate_audit_capture_trainable_pre_step_snapshot(trainer=trainer)
            if should_capture
            else None
        )
        result = original_step(*step_args, **step_kwargs)
        if should_capture and pre_step_snapshot is not None:
            _DELEGATE_DELTA_AUDIT_STATE["first_step_snapshot"] = pre_step_snapshot
            _DELEGATE_DELTA_AUDIT_STATE["first_step_written"] = True
            if bool(getattr(args, "emit_in_memory_delta_report", False)):
                report_path = _delegate_audit_write_in_memory_delta_report(
                    trainer=trainer,
                    pre_step_snapshot=pre_step_snapshot,
                )
                if report_path is not None:
                    _emit_info_line(f"in_memory_delta_report_path={report_path}")
        return result

    optimizer.step = _delegate_delta_audit_step
    setattr(optimizer, "_delegate_delta_audit_step_wrapped", True)
    return optimizer


def _iter_advantage_batch_entries(
    value: object,
    *,
    path: str = "",
    depth: int = 0,
) -> list[tuple[str, object]]:
    if depth > 4:
        return []

    matches: list[tuple[str, object]] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if "advantage" in key.lower():
                matches.append((child_path, child))
            matches.extend(
                _iter_advantage_batch_entries(
                    child,
                    path=child_path,
                    depth=depth + 1,
                )
            )
        return matches

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value[:4]):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            matches.extend(
                _iter_advantage_batch_entries(
                    child,
                    path=child_path,
                    depth=depth + 1,
                )
            )
    return matches


def _advantage_preview_values(value: object, *, limit: int) -> list[float]:
    preview: list[float] = []

    def _append(child: object) -> None:
        if len(preview) >= limit:
            return

        try:
            import torch

            if torch.is_tensor(child):
                tensor = child.detach()
                if tensor.numel() == 0:
                    return
                flat_preview = tensor.reshape(-1)[: limit - len(preview)].cpu().tolist()
                preview.extend(float(item) for item in flat_preview)
                return
        except Exception:
            pass

        if isinstance(child, bool):
            preview.append(float(child))
            return
        if isinstance(child, int | float):
            preview.append(float(child))
            return
        if isinstance(child, Sequence) and not isinstance(
            child, (str, bytes, bytearray)
        ):
            for grandchild in child:
                _append(grandchild)
                if len(preview) >= limit:
                    break

    _append(value)
    return preview[:limit]


def _format_advantage_batch_log(source_key: str, value: object) -> str:
    shape = _shape_list(value)
    dtype = str(getattr(value, "dtype", type(value).__name__))
    device = str(getattr(value, "device", "cpu"))
    preview = _advantage_preview_values(value, limit=NUMERIC_ADV_BATCH_PREVIEW_LIMIT)
    count = len(preview)
    min_text = "n/a"
    max_text = "n/a"
    mean_text = "n/a"

    try:
        import torch

        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.numel() > 0:
                numeric_tensor = tensor.float().cpu()
                min_text = f"{float(numeric_tensor.min().item()):.6g}"
                max_text = f"{float(numeric_tensor.max().item()):.6g}"
                mean_text = f"{float(numeric_tensor.mean().item()):.6g}"
    except Exception:
        pass

    if min_text == "n/a" and preview:
        min_text = f"{min(preview):.6g}"
        max_text = f"{max(preview):.6g}"
        mean_text = f"{(sum(preview) / len(preview)):.6g}"

    return (
        f"{NUMERIC_ADV_BATCH_LOG_PREFIX}{shape} "
        f"dtype={dtype} device={device} source_key={source_key} "
        f"preview_count={count} min={min_text} max={max_text} mean={mean_text} "
        f"preview={preview}"
    )


def _log_numeric_adv_batch_once(trainer: object, inputs: object) -> None:
    already_logged = int(getattr(trainer, "_numeric_adv_smoke_advantage_log_count", 0))
    if already_logged >= NUMERIC_ADV_BATCH_LOG_LIMIT:
        return

    entries = _iter_advantage_batch_entries(inputs)
    if entries:
        for source_key, value in entries[:2]:
            logging.info("%s", _format_advantage_batch_log(source_key, value))
    elif isinstance(inputs, Mapping):
        available_keys = sorted(str(key) for key in inputs.keys())
        logging.info(
            "%ss<missing> dtype=<missing> device=<missing> source_key=<none> preview_count=0 "
            "min=n/a max=n/a mean=n/a preview=[] available_keys=%s",
            NUMERIC_ADV_BATCH_LOG_PREFIX,
            available_keys,
        )
    else:
        logging.info(
            "%ss<missing> dtype=<missing> device=<missing> source_key=<none> preview_count=0 "
            "min=n/a max=n/a mean=n/a preview=[] batch_type=%s",
            NUMERIC_ADV_BATCH_LOG_PREFIX,
            type(inputs).__name__,
        )
    setattr(trainer, "_numeric_adv_smoke_advantage_log_count", already_logged + 1)


def _install_local_no_wandb_import_shim(*, use_wandb: bool) -> None:
    if bool(use_wandb):
        return
    if "wandb" in sys.modules:
        return

    from importlib.util import find_spec

    if find_spec("wandb") is not None:
        return

    class _DisabledWandbRun:
        config: dict[str, Any] = {}

        def log(self, *args: Any, **kwargs: Any) -> None:
            return None

        def finish(self, *args: Any, **kwargs: Any) -> None:
            return None

        def watch(self, *args: Any, **kwargs: Any) -> None:
            return None

    def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    shim = types.ModuleType("wandb")
    setattr(shim, "init", lambda *args, **kwargs: _DisabledWandbRun())
    setattr(shim, "log", _noop)
    setattr(shim, "finish", _noop)
    setattr(shim, "login", _noop)
    setattr(shim, "watch", _noop)
    setattr(shim, "config", {})
    setattr(shim, "run", None)
    shim.__dict__["_numeric_adv_smoke_no_wandb_shim"] = True
    sys.modules["wandb"] = shim
    logging.info(
        "numeric-adv smoke installed local wandb shim because use_wandb=False and wandb is unavailable"
    )


def _install_numeric_adv_monkeypatch(
    *, args: argparse.Namespace, repo_root: Path
) -> None:
    from importlib import import_module

    from termcolor import colored
    import torch

    from work.recap.dataset import (
        RecapAdvantageShardedSingleStepDataset,
        configure_late_stage_positive_emphasis,
        configure_positive_curriculum,
        configure_positive_oversampling,
    )
    from work.recap.model import GR00TRecapModel
    from work.recap.text_indicator import TextIndicatorShardedSingleStepDataset

    dataset_factory = import_module("gr00t.data.dataset.factory")
    dist_utils = import_module("gr00t.experiment.dist_utils")
    gr00t_trainer_module = import_module("gr00t.experiment.trainer")
    gr00t_setup = import_module("gr00t.model.gr00t_n1d6.setup")
    trainer_module = import_module("transformers.trainer")
    get_rank = getattr(dist_utils, "get_rank")
    Gr00tTrainer = getattr(gr00t_trainer_module, "Gr00tTrainer")
    Gr00tN1d6Pipeline = getattr(gr00t_setup, "Gr00tN1d6Pipeline")
    Trainer = getattr(trainer_module, "Trainer")
    continuation_checkpoint_path = _resolve_optional_checkpoint_path(repo_root, args)
    scope_summary = _scope_summary_for_args(args)
    trainability_authority = _build_trainability_authority_from_args(
        args,
        scope_summary=scope_summary,
    )
    oversample_config = configure_positive_oversampling(
        factor=int(args.positive_oversample_factor)
    )
    positive_curriculum_config = configure_positive_curriculum(
        enabled=bool(args.positive_curriculum),
        negative_retain_probability=float(args.negative_retain_probability),
        seed=int(_resolve_positive_curriculum_seed(args)),
    )
    late_stage_positive_config = configure_late_stage_positive_emphasis(
        enabled=bool(args.late_stage_positive_emphasis),
        threshold=float(args.late_stage_threshold),
    )
    _emit_info_line(
        "numeric-adv positive oversample "
        f"enabled={bool(oversample_config['enabled'])} "
        f"factor={int(oversample_config['factor'])}"
    )
    _emit_info_line(
        "numeric-adv positive curriculum "
        f"enabled={bool(positive_curriculum_config['enabled'])} "
        f"negative retain probability={float(positive_curriculum_config['negative_retain_probability']):.6f} "
        f"seed={int(positive_curriculum_config['seed'])}"
    )
    _emit_info_line(
        "numeric-adv late-stage positive config "
        f"late_stage_positive_enabled={bool(late_stage_positive_config['enabled'])} "
        f"late_stage_threshold={float(late_stage_positive_config['threshold']):.6f} "
        f"late_stage_rule={str(late_stage_positive_config['rule'])}"
    )

    if _text_indicator_route_enabled(args):

        class ConfiguredTextIndicatorShardedSingleStepDataset(
            TextIndicatorShardedSingleStepDataset
        ):
            def __init__(self, *ds_args: Any, **ds_kwargs: Any):
                ds_kwargs.setdefault(
                    "prompt_raw_column", str(args.text_indicator_prompt_raw_column)
                )
                ds_kwargs.setdefault("indicator_dropout_p", float(args.indicator_dropout_p))
                ds_kwargs.setdefault("indicator_dropout_seed", int(args.seed))
                ds_kwargs.setdefault(
                    "fallback_to_step_text",
                    bool(args.text_indicator_step_text_fallback),
                )
                super().__init__(*ds_args, **ds_kwargs)

        dataset_patch_cls: Any = ConfiguredTextIndicatorShardedSingleStepDataset
        dataset_patch_name = (
            "work.recap.text_indicator.TextIndicatorShardedSingleStepDataset"
        )
    else:
        dataset_patch_cls = RecapAdvantageShardedSingleStepDataset
        dataset_patch_name = "work.recap.dataset.RecapAdvantageShardedSingleStepDataset"

    if not getattr(
        dataset_factory, "_numeric_adv_smoke_dataset_patch_installed", False
    ):
        original_dataset_cls = getattr(dataset_factory, "ShardedSingleStepDataset")
        setattr(
            dataset_factory,
            "ShardedSingleStepDataset",
            dataset_patch_cls,
        )
        setattr(dataset_factory, "_numeric_adv_smoke_dataset_patch_installed", True)
        setattr(
            dataset_factory,
            "_numeric_adv_smoke_original_sharded_single_step_dataset",
            original_dataset_cls,
        )
        logging.info(
            "numeric-adv smoke patched gr00t.data.dataset.factory.ShardedSingleStepDataset -> %s",
            dataset_patch_name,
        )

    if not getattr(Trainer, "_numeric_adv_smoke_prepare_inputs_patch_installed", False):
        original_prepare_inputs = Trainer._prepare_inputs

        def _patched_prepare_inputs(self: object, inputs: object) -> object:
            _log_numeric_adv_batch_once(self, inputs)
            return original_prepare_inputs(self, inputs)

        Trainer._prepare_inputs = _patched_prepare_inputs
        Trainer._numeric_adv_smoke_prepare_inputs_patch_installed = True
        Trainer._numeric_adv_smoke_original_prepare_inputs = original_prepare_inputs
        logging.info(
            "numeric-adv smoke patched transformers.Trainer._prepare_inputs for pre-model advantage logging"
        )

    if not getattr(Gr00tTrainer, "_numeric_adv_smoke_train_patch_installed", False):
        original_train = Gr00tTrainer.train

        def _patched_train(
            self: Any,
            resume_from_checkpoint: Any = None,
            **kwargs: Any,
        ) -> Any:
            probe_session: dict[str, Any] | None = None
            if _condition_focused_continuation_enabled(args):
                assert continuation_checkpoint_path is not None
                _emit_info_line(
                    "condition-focused continuation train_fresh_state "
                    "mode=warm_start_weights_only "
                    f"warm_start_checkpoint={continuation_checkpoint_path} "
                    f"requested_additional_steps={_continuation_requested_steps(args)} "
                    "resume_from_checkpoint=False optimizer_state=reset "
                    "lr_scheduler_state=reset trainer_state=reset"
                )
                resume_from_checkpoint = False
            if _task8_probes_requested(args):
                probe_session = _task8_build_probe_session(trainer=self, args=args)
                setattr(self, "_task8_probe_session", probe_session)
                _task8_write_step_probe(
                    probe_session,
                    model=self.model,
                    step_index=0,
                )
            try:
                return original_train(
                    self,
                    resume_from_checkpoint=resume_from_checkpoint,
                    **kwargs,
                )
            finally:
                final_probe_session = getattr(self, "_task8_probe_session", None)
                if isinstance(final_probe_session, Mapping):
                    final_step_index = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
                    if final_step_index > 0:
                        _task8_write_step_probe(
                            final_probe_session,
                            model=self.model,
                            step_index=final_step_index,
                        )
                        _task8_write_semantics_outputs(
                            final_probe_session,
                            model=self.model,
                            final_step_index=final_step_index,
                        )
                if bool(getattr(args, "emit_saved_checkpoint_delta_report", False)):
                    report_path = _delegate_audit_write_saved_checkpoint_delta_report(
                        trainer=self,
                    )
                    if report_path is not None:
                        _emit_info_line(
                            f"saved_checkpoint_reload_delta_report_path={report_path}"
                        )

        Gr00tTrainer.train = _patched_train
        Gr00tTrainer._numeric_adv_smoke_train_patch_installed = True
        Gr00tTrainer._numeric_adv_smoke_original_train = original_train

    if not getattr(Gr00tTrainer, "_numeric_adv_smoke_optimizer_patch_installed", False):
        original_create_optimizer = Gr00tTrainer.create_optimizer

        def _patched_create_optimizer(self: Any) -> Any:
            if not _condition_focused_continuation_enabled(args):
                optimizer = original_create_optimizer(self)
            else:
                optimizer = trainability_entrypoint.create_repo_local_optimizer_for_trainer(
                    trainer=self,
                    authority=trainability_authority,
                    emit_info_line=_emit_info_line,
                )
                if optimizer is None:
                    optimizer = original_create_optimizer(self)
            scope_audit_path = trainability_entrypoint.emit_repo_local_static_scope_audit(
                trainer=self,
                authority=trainability_authority,
                requested_scope=args.recap_train_scope,
            )
            if scope_audit_path is not None:
                _emit_info_line(f"repo_local_static_scope_audit_path={scope_audit_path}")
            if bool(getattr(args, "emit_optimizer_param_group_report", False)):
                optimizer_report_path = _delegate_audit_emit_optimizer_param_group_report(
                    trainer=self,
                    optimizer=optimizer,
                )
                if optimizer_report_path is not None:
                    _emit_info_line(
                        f"optimizer_param_group_report_path={optimizer_report_path}"
                    )
            original_step = getattr(optimizer, "step", None)
            if callable(original_step) and not getattr(
                optimizer, "_task8_step_probe_patch_installed", False
            ):

                def _task8_probe_step(*step_args: Any, **step_kwargs: Any) -> Any:
                    result = original_step(*step_args, **step_kwargs)
                    probe_session = getattr(self, "_task8_probe_session", None)
                    if isinstance(probe_session, dict) and not bool(
                        probe_session.get("step1_written", False)
                    ):
                        _task8_write_step_probe(
                            probe_session,
                            model=self.model,
                            step_index=1,
                        )
                        probe_session["step1_written"] = True
                    return result

                optimizer.step = _task8_probe_step
                optimizer._task8_step_probe_patch_installed = True
            _delegate_audit_maybe_wrap_first_step(
                trainer=self,
                optimizer=optimizer,
                args=args,
            )
            return optimizer

        Gr00tTrainer.create_optimizer = _patched_create_optimizer
        Gr00tTrainer._numeric_adv_smoke_optimizer_patch_installed = True
        Gr00tTrainer._numeric_adv_smoke_original_create_optimizer = (
            original_create_optimizer
        )

    if getattr(Gr00tN1d6Pipeline, "_numeric_adv_smoke_patch_installed", False):
        return

    original_create_model = Gr00tN1d6Pipeline._create_model

    def _patched_create_model(self: Any) -> Any:
        if _text_indicator_route_enabled(args):
            model = original_create_model(self)
            authority_summary = trainability_entrypoint.apply_repo_local_trainability_authority(
                model=model,
                authority=trainability_authority,
                torch_module=torch,
            )
            _emit_info_line(
                "text-indicator mainline param_policy "
                f"route={_conditioning_route(args)} "
                f"indicator_dropout_p={float(args.indicator_dropout_p):.6g} "
                f"scope={args.recap_train_scope} "
                f"forced_trainable_tensors={int(authority_summary['forced_trainable']['tensors'])} "
                f"forced_frozen_tensors={int(authority_summary['forced_frozen']['tensors'])}"
            )
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(
                "Text-indicator trainable parameters: %s/%s (%.2f%%)",
                f"{trainable_params:,}",
                f"{total_params:,}",
                100 * trainable_params / total_params,
            )
            _log_trainable_action_head_parameters(model)
            return model

        checkpoint_raw = self.config.training.start_from_checkpoint
        checkpoint_path: Path | None = None
        alignment_info = {"config_path": None, "aligned_fields": {}}
        if checkpoint_raw is not None:
            checkpoint_path = Path(str(checkpoint_raw)).expanduser().resolve()
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Patched base checkpoint path does not exist: {checkpoint_path}"
                )
            alignment_info = _align_smoke_model_config_from_checkpoint(
                self.config.model,
                checkpoint_path,
            )
            if alignment_info["aligned_fields"]:
                logging.info(
                    "numeric-adv smoke aligned model config from %s: %s",
                    alignment_info["config_path"],
                    alignment_info["aligned_fields"],
                )
        _install_transformers_compat_hooks()
        model = GR00TRecapModel(
            self.config.model,
            transformers_loading_kwargs=self.transformers_loading_kwargs,
        )
        loading_info = {"missing_keys": [], "unexpected_keys": []}
        if checkpoint_path is not None:
            loading_info = _load_checkpoint_into_recap_model(model, checkpoint_path)
            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)
            if mask_token_missing and model.action_head.mask_token is not None:
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")
            logging.info(
                "numeric-adv smoke checkpoint load missing=%d unexpected=%d skipped_mismatched=%d",
                len(missing_keys),
                len(loading_info.get("unexpected_keys", [])),
                len(loading_info.get("skipped_mismatched_keys", [])),
            )

        authority_summary = trainability_entrypoint.apply_repo_local_trainability_authority(
            model=model,
            authority=trainability_authority,
        )
        _emit_info_line(
            "condition-focused continuation param_policy "
            f"condition_hot_lr_scale={float(authority_summary.get('condition_hot_lr_scale', 0.0)):.6g} "
            f"diffusion_trunk_lr_scale={float(authority_summary.get('diffusion_trunk_lr_scale', 0.0)):.6g} "
            f"diffusion_trunk_mode={authority_summary.get('diffusion_trunk_mode', 'frozen')}"
        )
        _emit_info_line(
            "condition-focused continuation forced_trainable "
            f"tensors={int(authority_summary['forced_trainable']['tensors'])} "
            f"numel={int(authority_summary['forced_trainable']['numel'])} "
            f"sample_names={authority_summary['forced_trainable']['sample_names']}"
        )
        _emit_info_line(
            "condition-focused continuation forced_frozen "
            f"tensors={int(authority_summary['forced_frozen']['tensors'])} "
            f"numel={int(authority_summary['forced_frozen']['numel'])} "
            f"sample_names={authority_summary['forced_frozen']['sample_names']}"
        )
        if _condition_focused_continuation_enabled(args):
            snapshot_path = _write_initial_advantage_embedding_snapshot(
                model=model,
                output_dir=Path(str(self.config.training.output_dir)).resolve(),
                continuation_checkpoint_path=checkpoint_path,
                loading_info=loading_info,
            )
            logging.info(
                "numeric-adv smoke wrote initial advantage embedding snapshot to %s",
                snapshot_path,
            )

        print(colored(f"Model Config: {model.config}", "yellow"))
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            "Trainable parameters: %s (%.2f%%)",
            f"{trainable_params:,}",
            100 * trainable_params / total_params,
        )
        _log_trainable_action_head_parameters(model)
        print("Model: ", model)
        return model

    Gr00tN1d6Pipeline._create_model = _patched_create_model
    Gr00tN1d6Pipeline._numeric_adv_smoke_patch_installed = True
    Gr00tN1d6Pipeline._numeric_adv_smoke_original_create_model = original_create_model


def _delegate_main(args: argparse.Namespace) -> int:
    _validate_args(args)
    repo_root = _repo_root()
    _ensure_repo_imports(repo_root)

    from importlib import import_module
    import torch

    from work.recap.hf_snapshot_patch import make_patched_base_model_dir

    _install_transformers_compat_hooks()

    _install_local_no_wandb_import_shim(use_wandb=bool(args.use_wandb))

    experiment = import_module("gr00t.experiment.experiment")
    run = getattr(experiment, "run")

    dataset_path = _resolve_path(repo_root, str(args.dataset_path))
    output_dir = _resolve_full_update_output_dir(repo_root, str(args.output_dir))
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    patched_dir = make_patched_base_model_dir(
        repo_id=str(args.base_model),
        revision=str(args.base_model_revision).strip() or None,
        out_root=str(args.patched_out_root),
        overrides=_hf_snapshot_overrides_for_route(args),
        hf_hub_cache_dir=(str(args.hf_hub_cache_dir).strip() or None),
        emit_evidence=True,
        force_tune_top_llm_layers_zero=bool(args.force_top_llm_layers_zero),
    )
    print(f"[INFO] conditioning_route={_conditioning_route(args)}")
    print(f"[INFO] recap_training_model_class={_model_class_for_route(args)}")
    print(
        "[INFO] patched_pipeline_target=gr00t.model.gr00t_n1d6.setup.Gr00tN1d6Pipeline._create_model"
    )
    print(f"[INFO] patched_base_model_dir={patched_dir}")

    _install_numeric_adv_monkeypatch(args=args, repo_root=repo_root)
    config = _build_training_config(
        repo_root=repo_root,
        dataset_path=dataset_path,
        base_model_path=Path(str(patched_dir)).resolve(),
        args=args,
    )
    current_device = trainability_entrypoint._safe_cuda_current_device(torch)
    version_surface_path = trainability_entrypoint._write_version_surface(
        config=config,
        torch_module=torch,
        current_device=current_device,
        requested_num_gpus=int(args.num_gpus),
    )
    print(f"[INFO] numeric_adv_version_surface_path={version_surface_path}")
    trainability_entrypoint.install_repo_local_rank_census_hooks(
        config=config,
        torch_module=torch,
    )
    run(config)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if bool(args.delegate_mode):
        return _delegate_main(args)
    return _wrapper_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
