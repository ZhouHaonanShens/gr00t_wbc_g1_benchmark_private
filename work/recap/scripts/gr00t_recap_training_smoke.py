#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import shlex
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import advantage
from work.recap import finetune_full as finetune_full_workflow
from work.demo_utils import paths as demo_paths


DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1"
DEFAULT_PATCHED_OUT_ROOT = "agent/artifacts/hf_patches"
DEFAULT_CPU_CONFIG_MODULE = "configs.apple_recap.flux.train_smoke_cpu"
DEFAULT_GPU_CONFIG_MODULE = "configs.apple_recap.flux.train_smoke_gpu_lora"
DEFAULT_CONFIG_MODULE_BY_SMOKE_MODE = {
    "cpu": DEFAULT_CPU_CONFIG_MODULE,
    "gpu_lora_or_head_only": DEFAULT_GPU_CONFIG_MODULE,
}
SMOKE_MODE_ALIASES = {
    "gpu_lora": "gpu_lora_or_head_only",
}
DEFAULT_DELEGATE_SCRIPT = "work/recap/finetune_full.py"
DEFAULT_DELEGATE_SUMMARY_NAME = "delegate_finetune_summary.json"
DEFAULT_FINAL_MODEL_CONFIG_NAME = "final_model_config.json"
DEFAULT_STAGE3_OUTPUT_ROOT = "agent/artifacts/stage3_prereq_smoke"
DEFAULT_STAGE3_RUNTIME_LOG_ROOT = "agent/runtime_logs/stage3_prereq_smoke"
DEFAULT_TRAINABILITY_GATE_NAME = "trainability_gate.json"
SMOKE_SUMMARY_ARTIFACT_KIND = "flux_gr00t_training_smoke_summary"
SMOKE_SURFACE_ROUTE = "flux_gr00t_training_smoke_diagnostic"
SMOKE_AUTHORITY_SCOPE = "flux_training_smoke_diagnostic_lane"
GPU_LORA_UNAVAILABLE_REASON = "current_live_launch_surface_has_no_explicit_lora_knobs; using validated head-only fallback"
HEAD_ONLY_SURFACE = "head_only_fallback"
CPU_DRY_RUN_SURFACE = "head_only_dry_run"


def _normalize_smoke_mode(raw: object, *, field_name: str = "smoke_mode") -> str:
    requested = str(raw or "").strip()
    normalized = SMOKE_MODE_ALIASES.get(requested, requested)
    if normalized in DEFAULT_CONFIG_MODULE_BY_SMOKE_MODE:
        return normalized
    supported = sorted(
        [*DEFAULT_CONFIG_MODULE_BY_SMOKE_MODE.keys(), *SMOKE_MODE_ALIASES.keys()]
    )
    raise ValueError(f"{field_name} must be one of {supported!r}, got {requested!r}")


def _parse_smoke_mode(raw: str) -> str:
    try:
        return _normalize_smoke_mode(raw, field_name="--smoke-mode")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_flux_train_smoke.py",
        description=(
            "Materialize the Flux-specific training smoke lane. This wrapper stays "
            "diagnostic-only, keeps save_total_limit=1, and fail-closes if the GPU "
            "smoke lane would expand beyond LoRA-first/head-only semantics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="LeRobot dataset directory used for the Flux smoke lane.",
    )
    parser.add_argument(
        "--smoke-mode",
        type=_parse_smoke_mode,
        choices=tuple(DEFAULT_CONFIG_MODULE_BY_SMOKE_MODE.keys()),
        default="cpu",
        help=(
            "Convenience selector for the built-in Flux smoke config surfaces. "
            "Canonical alias: gpu_lora -> gpu_lora_or_head_only."
        ),
    )
    parser.add_argument(
        "--config-module",
        type=str,
        default="",
        help=(
            "Optional import path overriding --smoke-mode. The module must expose "
            "CONFIG and/or build_config()."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional override for the smoke output directory.",
    )
    parser.add_argument(
        "--runtime-log-dir",
        type=str,
        default="",
        help="Optional override for the wrapper runtime log directory.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional override for the final wrapper summary JSON path.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base model repo_id or local path passed through to the delegated smoke run.",
    )
    parser.add_argument(
        "--base-model-revision",
        type=str,
        default="",
        help="Optional base-model revision forwarded to the delegated smoke run.",
    )
    parser.add_argument(
        "--hf-hub-cache-dir",
        type=str,
        default="",
        help="Optional HuggingFace hub cache root used by the delegated smoke run.",
    )
    parser.add_argument(
        "--patched-out-root",
        type=str,
        default=DEFAULT_PATCHED_OUT_ROOT,
        help="Repo-relative root for patched base-model snapshots.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="",
        help="Python executable used to spawn the delegated smoke run.",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default=DEFAULT_EMBODIMENT_TAG,
        help="Embodiment tag forwarded to the delegated smoke run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Force the selected smoke profile into dry-run mode. The CPU profile is "
            "already dry-run by default; the GPU profile can use this for cheap "
            "planner/contract verification."
        ),
    )
    parser.add_argument(
        "--delegate-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _resolve_repo_path(repo_root: Path, raw: str | Path) -> Path:
    return finetune_full_workflow._resolve_path(repo_root, str(raw))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    finetune_full_workflow._write_json(path, dict(payload))


def _emit_summary(
    payload: Mapping[str, Any], *, summary_json_path: Path | None
) -> None:
    if summary_json_path is not None:
        _write_json(summary_json_path, payload)
    json.dump(dict(payload), sys.stdout, ensure_ascii=True, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _jsonify_cmd(cmd: Sequence[str]) -> list[str]:
    return [str(part) for part in cmd]


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(cast(Mapping[str, Any], payload))


def _load_config_payload(module_name: str) -> dict[str, Any]:
    module = import_module(str(module_name).strip())
    builder = getattr(module, "build_config", None)
    if callable(builder):
        payload = builder()
    else:
        payload = getattr(module, "CONFIG", None)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"{module_name} must expose CONFIG or build_config() returning an object"
        )
    return deepcopy(dict(cast(Mapping[str, Any], payload)))


def load_training_smoke_config(
    *,
    smoke_mode: str,
    config_module: str,
) -> tuple[str, dict[str, Any]]:
    normalized_smoke_mode = _normalize_smoke_mode(
        smoke_mode,
        field_name="smoke_mode",
    )
    resolved_module = (
        str(config_module).strip()
        if str(config_module).strip()
        else DEFAULT_CONFIG_MODULE_BY_SMOKE_MODE[normalized_smoke_mode]
    )
    return resolved_module, _load_config_payload(resolved_module)


def _profile_output_dir(
    repo_root: Path,
    *,
    config_name: str,
    override: str,
) -> Path:
    if str(override).strip():
        return _resolve_repo_path(repo_root, override)
    return _resolve_repo_path(repo_root, f"{DEFAULT_STAGE3_OUTPUT_ROOT}/{config_name}")


def _profile_runtime_log_dir(
    repo_root: Path,
    *,
    config_name: str,
    override: str,
) -> Path:
    if str(override).strip():
        return _resolve_repo_path(repo_root, override)
    return _resolve_repo_path(
        repo_root,
        f"{DEFAULT_STAGE3_RUNTIME_LOG_ROOT}/{config_name}",
    )


def _profile_summary_json(
    repo_root: Path,
    *,
    config_name: str,
    override: str,
) -> Path | None:
    if str(override).strip():
        return _resolve_repo_path(repo_root, override)
    return _resolve_repo_path(
        repo_root,
        f"{DEFAULT_STAGE3_OUTPUT_ROOT}/{config_name}_summary.json",
    )


def _resolve_two_layer_python_contract(
    repo_root: Path,
    *,
    delegate_runtime_python_flag: str,
) -> dict[str, str | None]:
    contract = demo_paths.load_stage3_training_python_contract(repo_root)
    launcher_python = str(demo_paths.current_python_abspath_preserve_symlink())
    orchestrator_python = str(contract["orchestrator_python"])
    if not demo_paths.same_abspath_preserve_symlink(
        launcher_python, orchestrator_python
    ):
        raise ValueError(
            "two-layer interpreter contract violated: "
            f"expected orchestrator_python={orchestrator_python}, got {launcher_python}"
        )

    requested_delegate_runtime = str(delegate_runtime_python_flag or "").strip()
    delegate_runtime_python = str(contract["delegate_runtime_python"])
    if requested_delegate_runtime and not demo_paths.same_abspath_preserve_symlink(
        requested_delegate_runtime,
        delegate_runtime_python,
    ):
        raise ValueError(
            "two-layer interpreter contract violated: "
            f"expected delegate_runtime_python={delegate_runtime_python}, got {requested_delegate_runtime}"
        )
    return {
        "contract_manifest_path": str(contract["manifest_path"]),
        "launcher_python": launcher_python,
        "orchestrator_python": orchestrator_python,
        "delegate_runtime_python": delegate_runtime_python,
        "delegate_runtime_python_requested": (
            None if not requested_delegate_runtime else requested_delegate_runtime
        ),
    }


def _trainability_gate_path(repo_root: Path, *, summary_json_path: Path | None) -> Path:
    if summary_json_path is not None:
        return summary_json_path.parent / DEFAULT_TRAINABILITY_GATE_NAME
    return (
        demo_paths.stage3_prereq_smoke_artifact_root(repo_root)
        / DEFAULT_TRAINABILITY_GATE_NAME
    )


def _dependency_blocked(payload: Mapping[str, Any]) -> bool:
    error_texts = [str(payload.get("error") or "")]
    delegate_summary = payload.get("delegate_summary")
    if isinstance(delegate_summary, Mapping):
        error_texts.append(str(delegate_summary.get("error") or ""))
    merged = "\n".join(part for part in error_texts if part).lower()
    return any(
        token in merged
        for token in (
            "modulenotfounderror",
            "importerror",
            "no module named",
            "unable to import required dependencies",
        )
    )


def _build_trainability_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    delegate_summary = payload.get("delegate_summary")
    delegate_mapping = (
        dict(cast(Mapping[str, Any], delegate_summary))
        if isinstance(delegate_summary, Mapping)
        else None
    )
    return {
        "schema_version": "gr00t_training_lane_trainability_gate_v1",
        "status": "PASS" if str(payload.get("wrapper_status")) == "ok" else "FAIL",
        "wrapper_status": str(payload.get("wrapper_status")),
        "dependency_blocked": bool(_dependency_blocked(payload)),
        "contract_manifest_path": payload.get("contract_manifest_path"),
        "launcher_python": payload.get("launcher_python"),
        "orchestrator_python": payload.get("orchestrator_python"),
        "delegate_runtime_python": payload.get("delegate_runtime_python"),
        "delegate_runtime_python_requested": payload.get(
            "delegate_runtime_python_requested"
        ),
        "delegate_summary_path": payload.get("delegate_summary_path"),
        "delegate_summary_exists": bool(payload.get("delegate_summary_exists")),
        "delegate_wrapper_status": None
        if delegate_mapping is None
        else delegate_mapping.get("wrapper_status"),
        "delegate_launcher_python": None
        if delegate_mapping is None
        else delegate_mapping.get("launcher_python"),
        "delegate_orchestrator_python": None
        if delegate_mapping is None
        else delegate_mapping.get("orchestrator_python"),
        "delegate_runtime_python_reported": None
        if delegate_mapping is None
        else delegate_mapping.get("delegate_runtime_python"),
        "dataset_path": payload.get("dataset_path"),
        "output_dir": payload.get("output_dir"),
        "summary_json": payload.get("summary_json"),
        "error": payload.get("error"),
    }


def _require_smoke_bool(raw: object, *, field_name: str) -> bool:
    if not isinstance(raw, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(raw).__name__}")
    return bool(raw)


def _require_smoke_int(raw: object, *, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{field_name} must be an int, got {type(raw).__name__}")
    return int(raw)


def _require_smoke_float(raw: object, *, field_name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{field_name} must be float-like, got {type(raw).__name__}")
    return float(raw)


def _requested_trainable_surface(profile: Mapping[str, Any]) -> str:
    raw = cast(Mapping[str, Any], profile.get("trainable_surface", {})).get("preferred")
    value = str(raw or "").strip()
    if value not in {"head_only", "lora"}:
        raise ValueError(
            "training smoke config trainable_surface.preferred must be 'head_only' or 'lora'"
        )
    return value


def _head_only_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    surface = cast(Mapping[str, Any], profile.get("trainable_surface", {}))
    head_only = cast(Mapping[str, Any], surface.get("head_only", {}))
    contract = {
        "tune_llm": _require_smoke_bool(
            head_only.get("tune_llm"), field_name="trainable_surface.head_only.tune_llm"
        ),
        "tune_visual": _require_smoke_bool(
            head_only.get("tune_visual"),
            field_name="trainable_surface.head_only.tune_visual",
        ),
        "tune_top_llm_layers": _require_smoke_int(
            head_only.get("tune_top_llm_layers"),
            field_name="trainable_surface.head_only.tune_top_llm_layers",
        ),
        "tune_projector": _require_smoke_bool(
            head_only.get("tune_projector"),
            field_name="trainable_surface.head_only.tune_projector",
        ),
        "tune_diffusion_model": _require_smoke_bool(
            head_only.get("tune_diffusion_model"),
            field_name="trainable_surface.head_only.tune_diffusion_model",
        ),
        "tune_vlln": _require_smoke_bool(
            head_only.get("tune_vlln"),
            field_name="trainable_surface.head_only.tune_vlln",
        ),
    }
    if contract["tune_llm"]:
        raise ValueError("head-only smoke contract must keep tune_llm=false")
    if contract["tune_visual"]:
        raise ValueError("head-only smoke contract must keep tune_visual=false")
    if contract["tune_top_llm_layers"] != 0:
        raise ValueError(
            "head-only smoke contract must keep tune_top_llm_layers=0 to avoid backbone escalation"
        )
    if contract["tune_projector"]:
        raise ValueError("head-only smoke contract must keep tune_projector=false")
    if contract["tune_diffusion_model"]:
        raise ValueError(
            "head-only smoke contract must keep tune_diffusion_model=false"
        )
    if not contract["tune_vlln"]:
        raise ValueError(
            "head-only smoke contract must keep tune_vlln=true so the fallback still trains a minimal head surface"
        )
    return contract


def resolve_smoke_plan(
    *,
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    config_module: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    execution = cast(Mapping[str, Any], profile.get("execution", {}))
    smoke_mode = str(profile.get("smoke_mode") or "").strip()
    if smoke_mode not in DEFAULT_CONFIG_MODULE_BY_SMOKE_MODE:
        raise ValueError(f"unsupported smoke_mode in config: {smoke_mode!r}")
    config_name = str(profile.get("config_name") or "").strip()
    if not config_name:
        raise ValueError("training smoke config must define config_name")

    dataset_path = _resolve_repo_path(repo_root, str(args.dataset_path))
    output_dir = _profile_output_dir(
        repo_root,
        config_name=config_name,
        override=str(args.output_dir),
    )
    runtime_log_dir = _profile_runtime_log_dir(
        repo_root,
        config_name=config_name,
        override=str(args.runtime_log_dir),
    )
    summary_json_path = _profile_summary_json(
        repo_root,
        config_name=config_name,
        override=str(args.summary_json),
    )
    python_contract = _resolve_two_layer_python_contract(
        repo_root,
        delegate_runtime_python_flag=str(args.python).strip(),
    )
    requested_surface = _requested_trainable_surface(profile)
    head_only = _head_only_contract(profile)
    fallback_surface = str(
        cast(Mapping[str, Any], profile.get("trainable_surface", {})).get("fallback")
        or ""
    ).strip()

    dry_run = _require_smoke_bool(
        execution.get("dry_run"), field_name="execution.dry_run"
    ) or bool(args.dry_run)
    max_steps = _require_smoke_int(
        execution.get("max_steps"), field_name="execution.max_steps"
    )
    save_steps = _require_smoke_int(
        execution.get("save_steps"), field_name="execution.save_steps"
    )
    save_total_limit = _require_smoke_int(
        execution.get("save_total_limit"), field_name="execution.save_total_limit"
    )
    num_gpus = _require_smoke_int(
        execution.get("num_gpus"), field_name="execution.num_gpus"
    )
    global_batch_size = _require_smoke_int(
        execution.get("global_batch_size"),
        field_name="execution.global_batch_size",
    )
    gradient_accumulation_steps = _require_smoke_int(
        execution.get("gradient_accumulation_steps"),
        field_name="execution.gradient_accumulation_steps",
    )
    dataloader_num_workers = _require_smoke_int(
        execution.get("dataloader_num_workers"),
        field_name="execution.dataloader_num_workers",
    )
    learning_rate = _require_smoke_float(
        execution.get("learning_rate"), field_name="execution.learning_rate"
    )
    use_wandb = _require_smoke_bool(
        execution.get("use_wandb"), field_name="execution.use_wandb"
    )

    if save_total_limit != 1:
        raise ValueError(
            "Flux smoke lane enforces save_total_limit=1 for single-checkpoint retention"
        )
    if max_steps <= 0:
        raise ValueError("Flux smoke lane requires max_steps > 0")
    if save_steps != max_steps:
        raise ValueError(
            "Flux smoke lane requires save_steps == max_steps so only the final smoke checkpoint is retained"
        )
    if global_batch_size != 1:
        raise ValueError("Flux smoke lane requires global_batch_size=1")
    if gradient_accumulation_steps != 1:
        raise ValueError("Flux smoke lane requires gradient_accumulation_steps=1")
    if dataloader_num_workers != 0:
        raise ValueError("Flux smoke lane requires dataloader_num_workers=0")
    if use_wandb:
        raise ValueError("Flux smoke lane requires use_wandb=false")

    lora_supported = False
    lora_reason = GPU_LORA_UNAVAILABLE_REASON
    fallback_applied = False
    if requested_surface == "lora":
        if fallback_surface != "head_only":
            raise ValueError(
                "GPU LoRA smoke requires trainable_surface.fallback='head_only' when the LoRA path is unavailable"
            )
        effective_surface = HEAD_ONLY_SURFACE
        fallback_applied = True
    else:
        effective_surface = CPU_DRY_RUN_SURFACE if dry_run else "head_only"

    if smoke_mode == "cpu":
        if not dry_run:
            raise ValueError("CPU smoke profile must remain dry_run=true")
        if num_gpus != 0:
            raise ValueError("CPU smoke profile must declare num_gpus=0")
    else:
        if num_gpus != 1:
            raise ValueError("GPU smoke profile must declare num_gpus=1")
        if max_steps > 16:
            raise ValueError(
                "GPU smoke profile exceeded the repo-local smoke budget; max_steps must be <= 16"
            )

    diagnostics = cast(Mapping[str, Any], profile.get("diagnostic", {}))
    gate_semantics = str(
        diagnostics.get("gate_semantics") or "diagnostic_only_non_release_gate"
    ).strip()
    if not gate_semantics:
        raise ValueError("diagnostic.gate_semantics must be a non-empty string")

    return {
        "config_name": config_name,
        "config_module": str(config_module),
        "smoke_mode": smoke_mode,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "runtime_log_dir": str(runtime_log_dir),
        "summary_json_path": (
            None if summary_json_path is None else str(summary_json_path)
        ),
        "delegate_summary_path": str(output_dir / DEFAULT_DELEGATE_SUMMARY_NAME),
        "final_model_config_path": str(output_dir / DEFAULT_FINAL_MODEL_CONFIG_NAME),
        "trainability_gate_path": str(
            _trainability_gate_path(repo_root, summary_json_path=summary_json_path)
        ),
        "dry_run": bool(dry_run),
        "base_model": str(args.base_model),
        "base_model_revision": str(args.base_model_revision).strip(),
        "hf_hub_cache_dir": str(args.hf_hub_cache_dir).strip(),
        "patched_out_root": str(args.patched_out_root).strip(),
        "contract_manifest_path": str(python_contract["contract_manifest_path"]),
        "launcher_python": str(python_contract["launcher_python"]),
        "orchestrator_python": str(python_contract["orchestrator_python"]),
        "delegate_runtime_python": str(python_contract["delegate_runtime_python"]),
        "delegate_runtime_python_requested": python_contract[
            "delegate_runtime_python_requested"
        ],
        "embodiment_tag": str(args.embodiment_tag),
        "max_steps": int(max_steps),
        "save_steps": int(save_steps),
        "save_total_limit": int(save_total_limit),
        "num_gpus": int(num_gpus),
        "global_batch_size": int(global_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "dataloader_num_workers": int(dataloader_num_workers),
        "learning_rate": float(learning_rate),
        "use_wandb": bool(use_wandb),
        "requested_trainable_surface": requested_surface,
        "trainable_surface": effective_surface,
        "head_only_contract": head_only,
        "lora_supported": bool(lora_supported),
        "lora_support_reason": lora_reason,
        "fallback_applied": bool(fallback_applied),
        "gate_semantics": gate_semantics,
    }


def _build_delegate_cmd(
    *,
    repo_root: Path,
    plan: Mapping[str, Any],
) -> list[str]:
    delegate_script = (repo_root / DEFAULT_DELEGATE_SCRIPT).resolve()
    cmd = [
        str(plan["launcher_python"]),
        str(delegate_script),
        "--dataset-path",
        str(plan["dataset_path"]),
        "--output-dir",
        str(plan["output_dir"]),
        "--runtime-log-dir",
        str(plan["runtime_log_dir"]),
        "--summary-json",
        str(plan["delegate_summary_path"]),
        "--base-model",
        str(plan["base_model"]),
        "--embodiment-tag",
        str(plan["embodiment_tag"]),
        "--max-steps",
        str(int(plan["max_steps"])),
        "--save-steps",
        str(int(plan["save_steps"])),
        "--save-total-limit",
        str(int(plan["save_total_limit"])),
        "--global-batch-size",
        str(int(plan["global_batch_size"])),
        "--gradient-accumulation-steps",
        str(int(plan["gradient_accumulation_steps"])),
        "--dataloader-num-workers",
        str(int(plan["dataloader_num_workers"])),
        "--learning-rate",
        str(float(plan["learning_rate"])),
        "--no-tune-projector",
        "--no-tune-diffusion-model",
        "--no-use-wandb",
        "--python",
        str(plan["delegate_runtime_python"]),
    ]
    if int(plan["num_gpus"]) > 0:
        cmd.extend(["--num-gpus", str(int(plan["num_gpus"]))])
    if str(plan["base_model_revision"]).strip():
        cmd.extend(["--base-model-revision", str(plan["base_model_revision"])])
    if str(plan["hf_hub_cache_dir"]).strip():
        cmd.extend(["--hf-hub-cache-dir", str(plan["hf_hub_cache_dir"])])
    if str(plan["patched_out_root"]).strip():
        cmd.extend(["--patched-out-root", str(plan["patched_out_root"])])
    if bool(plan["dry_run"]):
        cmd.append("--dry-run")
    return cmd


def _validate_dataset_path(plan: Mapping[str, Any]) -> None:
    dataset_path = Path(str(plan["dataset_path"]))
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"dataset path not found: {dataset_path}")


def _validate_delegate_summary(
    *,
    delegate_summary: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    if str(delegate_summary.get("wrapper_status")) != "ok":
        raise ValueError("delegated finetune wrapper did not report wrapper_status=ok")
    if str(delegate_summary.get("dataset_path")) != str(plan["dataset_path"]):
        raise ValueError("delegated finetune drifted dataset_path")
    if str(delegate_summary.get("output_dir")) != str(plan["output_dir"]):
        raise ValueError("delegated finetune drifted output_dir")
    if bool(delegate_summary.get("dry_run")) != bool(plan["dry_run"]):
        raise ValueError("delegated finetune drifted dry_run flag")
    if str(delegate_summary.get("orchestrator_python")) != str(
        plan["orchestrator_python"]
    ):
        raise ValueError("delegated finetune drifted orchestrator_python")
    if str(delegate_summary.get("delegate_runtime_python")) != str(
        plan["delegate_runtime_python"]
    ):
        raise ValueError("delegated finetune drifted delegate_runtime_python")

    effective = cast(Mapping[str, Any], delegate_summary.get("effective_config", {}))
    expected_pairs = {
        "max_steps": int(plan["max_steps"]),
        "save_steps": int(plan["save_steps"]),
        "save_total_limit": int(plan["save_total_limit"]),
        "global_batch_size": int(plan["global_batch_size"]),
        "gradient_accumulation_steps": int(plan["gradient_accumulation_steps"]),
        "dataloader_num_workers": int(plan["dataloader_num_workers"]),
        "learning_rate": float(plan["learning_rate"]),
        "num_gpus": None if int(plan["num_gpus"]) == 0 else int(plan["num_gpus"]),
        "tune_projector": False,
        "tune_diffusion_model": False,
        "use_wandb": False,
    }
    drifted = [
        key
        for key, expected in expected_pairs.items()
        if effective.get(key) != expected
    ]
    if drifted:
        raise ValueError(
            "delegated finetune drifted smoke-only fields: " + ", ".join(drifted)
        )


def _validate_head_only_final_model_config(
    *,
    final_model_config: Mapping[str, Any],
) -> None:
    expected = {
        "tune_top_llm_layers": 0,
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": False,
        "tune_diffusion_model": False,
        "tune_vlln": True,
    }
    drifted = [
        key
        for key, expected_value in expected.items()
        if final_model_config.get(key) != expected_value
    ]
    if drifted:
        raise ValueError(
            "final_model_config escaped head-only smoke fencing: " + ", ".join(drifted)
        )


def materialize_flux_training_smoke(
    *,
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    config_module: str,
    repo_root: Path = REPO_ROOT,
    delegate_runner: Callable[[Sequence[str], Path, Path], int] | None = None,
) -> dict[str, Any]:
    plan = resolve_smoke_plan(
        args=args,
        profile=profile,
        config_module=config_module,
        repo_root=repo_root,
    )
    summary_json_path = (
        None
        if plan["summary_json_path"] is None
        else Path(str(plan["summary_json_path"]))
    )
    output_dir = Path(str(plan["output_dir"]))
    runtime_log_dir = Path(str(plan["runtime_log_dir"]))
    delegate_summary_path = Path(str(plan["delegate_summary_path"]))
    final_model_config_path = Path(str(plan["final_model_config_path"]))
    runtime_log_path = runtime_log_dir / "gr00t_flux_train_smoke.log"
    delegate_script = (repo_root / DEFAULT_DELEGATE_SCRIPT).resolve()
    payload: dict[str, Any] = {
        "timestamp": __import__("datetime")
        .datetime.now()
        .isoformat(timespec="seconds"),
        "wrapper": "gr00t_recap_training_smoke.py",
        "artifact_kind": SMOKE_SUMMARY_ARTIFACT_KIND,
        "wrapper_status": "ok",
        "config_name": str(plan["config_name"]),
        "config_module": str(plan["config_module"]),
        "smoke_mode": str(plan["smoke_mode"]),
        "dry_run": bool(plan["dry_run"]),
        "dataset_path": str(plan["dataset_path"]),
        "dataset_exists": Path(str(plan["dataset_path"])).is_dir(),
        "output_dir": str(output_dir),
        "runtime_log_dir": str(runtime_log_dir),
        "runtime_log_path": None if bool(plan["dry_run"]) else str(runtime_log_path),
        "summary_json": None if summary_json_path is None else str(summary_json_path),
        "delegate_script": str(delegate_script),
        "delegate_summary_path": str(delegate_summary_path),
        "delegate_summary_exists": False,
        "delegate_summary": None,
        "delegate_cmd": None,
        "delegate_cmd_shell": None,
        "delegate_returncode": None,
        "trainability_gate_path": str(plan["trainability_gate_path"]),
        "contract_manifest_path": str(plan["contract_manifest_path"]),
        "launcher_python": str(plan["launcher_python"]),
        "orchestrator_python": str(plan["orchestrator_python"]),
        "delegate_runtime_python": str(plan["delegate_runtime_python"]),
        "delegate_runtime_python_requested": plan["delegate_runtime_python_requested"],
        "requested_trainable_surface": str(plan["requested_trainable_surface"]),
        "trainable_surface": str(plan["trainable_surface"]),
        "fallback_applied": bool(plan["fallback_applied"]),
        "lora_supported": bool(plan["lora_supported"]),
        "lora_support_reason": str(plan["lora_support_reason"]),
        "parameter_update": {
            "tune_llm": False,
            "tune_visual": False,
            "tune_top_llm_layers": 0,
            "tune_projector": False,
            "tune_diffusion_model": False,
            "tune_vlln": True,
            "lora_requested": str(plan["requested_trainable_surface"]) == "lora",
            "lora_supported": bool(plan["lora_supported"]),
        },
        "max_steps": int(plan["max_steps"]),
        "num_gpus": int(plan["num_gpus"]),
        "save_total_limit": int(plan["save_total_limit"]),
        "selected_checkpoint_path": None,
        "selected_checkpoint_exists": False,
        "selected_checkpoint_asset_path": None,
        "selected_checkpoint_asset_exists": False,
        "final_model_config_path": None,
        "final_model_config_exists": False,
        "final_model_config": None,
        "main_verdict_eligible": False,
        "external_reference_only": True,
        "gate_semantics": str(plan["gate_semantics"]),
        "release_gate": False,
        "error": None,
    }
    payload.update(
        advantage.build_diagnostic_surface_metadata(
            surface_route=SMOKE_SURFACE_ROUTE,
            authority_scope=SMOKE_AUTHORITY_SCOPE,
            compatibility_fields=(
                "wrapper_status",
                "smoke_mode",
                "max_steps",
                "num_gpus",
                "save_total_limit",
                "selected_checkpoint_path",
                "selected_checkpoint_asset_path",
                "trainable_surface",
            ),
            surface_kind=SMOKE_SUMMARY_ARTIFACT_KIND,
        )
    )

    runner = delegate_runner or (
        lambda cmd, cwd, log_path: finetune_full_workflow._run_with_tee(
            list(cmd), cwd=cwd, log_path=log_path
        )
    )

    try:
        _validate_dataset_path(plan)
        delegate_cmd = _build_delegate_cmd(repo_root=repo_root, plan=plan)
        payload["delegate_cmd"] = _jsonify_cmd(delegate_cmd)
        payload["delegate_cmd_shell"] = shlex.join(delegate_cmd)

        if bool(plan["dry_run"]):
            return payload

        output_dir.mkdir(parents=True, exist_ok=True)
        runtime_log_dir.mkdir(parents=True, exist_ok=True)
        rc = int(runner(delegate_cmd, repo_root, runtime_log_path))
        payload["delegate_returncode"] = rc
        if rc != 0:
            raise RuntimeError(f"delegated_finetune_failed: returncode={rc}")

        if not delegate_summary_path.is_file():
            raise FileNotFoundError(
                f"delegated finetune did not write {delegate_summary_path.name}"
            )
        delegate_summary = _read_json_object(delegate_summary_path)
        payload["delegate_summary_exists"] = True
        payload["delegate_summary"] = delegate_summary
        _validate_delegate_summary(delegate_summary=delegate_summary, plan=plan)

        if not final_model_config_path.is_file():
            raise FileNotFoundError(
                f"delegated finetune did not materialize {final_model_config_path.name}"
            )
        final_model_config = _read_json_object(final_model_config_path)
        _validate_head_only_final_model_config(final_model_config=final_model_config)
        payload["final_model_config_path"] = str(final_model_config_path)
        payload["final_model_config_exists"] = True
        payload["final_model_config"] = final_model_config

        selected_checkpoint = finetune_full_workflow._latest_checkpoint(output_dir)
        selected_asset = finetune_full_workflow._selected_checkpoint_asset(
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
        payload["selected_checkpoint_asset_exists"] = bool(selected_asset is not None)
        if selected_asset is None:
            raise FileNotFoundError(
                "smoke run finished without a retained checkpoint asset"
            )
        return payload
    except Exception as exc:
        payload["wrapper_status"] = "blocked"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload


def _emit_and_exit(
    payload: Mapping[str, Any], *, summary_json_path: Path | None
) -> int:
    _write_json(
        Path(str(payload["trainability_gate_path"])),
        _build_trainability_gate(payload),
    )
    _emit_summary(payload, summary_json_path=summary_json_path)
    return 0 if str(payload.get("wrapper_status")) == "ok" else 1


def _delegate_main(argv: Sequence[str] | None = None) -> int:
    del argv
    raise RuntimeError(
        "delegate-mode should not be called directly on gr00t_recap_training_smoke.py; "
        "the wrapper delegates to work/recap/finetune_full.py instead"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.delegate_mode):
        return _delegate_main(argv)
    config_module, profile = load_training_smoke_config(
        smoke_mode=str(args.smoke_mode),
        config_module=str(args.config_module),
    )
    plan = resolve_smoke_plan(
        args=args,
        profile=profile,
        config_module=config_module,
        repo_root=REPO_ROOT,
    )
    summary_json_path = (
        None
        if plan["summary_json_path"] is None
        else Path(str(plan["summary_json_path"]))
    )
    payload = materialize_flux_training_smoke(
        args=args,
        profile=profile,
        config_module=config_module,
        repo_root=REPO_ROOT,
    )
    return _emit_and_exit(payload, summary_json_path=summary_json_path)


if __name__ == "__main__":
    raise SystemExit(main())
