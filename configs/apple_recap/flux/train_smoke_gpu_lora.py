from __future__ import annotations

from copy import deepcopy
from typing import Any


CONFIG = {
    "config_name": "flux_train_smoke_gpu_lora",
    "smoke_mode": "gpu_lora_or_head_only",
    "execution": {
        "dry_run": False,
        "max_steps": 4,
        "save_steps": 4,
        "save_total_limit": 1,
        "num_gpus": 1,
        "global_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "dataloader_num_workers": 0,
        "learning_rate": 1e-5,
        "use_wandb": False,
    },
    "trainable_surface": {
        "preferred": "lora",
        "fallback": "head_only",
        "lora": {
            "use_backbone_lora": 16,
            "use_llm_lora": 16,
            "intent": "lora_first_if_live_stack_exposes_explicit_lora_knobs",
        },
        "head_only": {
            "tune_llm": False,
            "tune_visual": False,
            "tune_top_llm_layers": 0,
            "tune_projector": False,
            "tune_diffusion_model": False,
            "tune_vlln": True,
        },
    },
    "artifacts": {
        "default_output_dir": "agent/artifacts/apple_recap_flux_graft/train_smoke_gpu_lora",
        "default_runtime_log_dir": "agent/runtime_logs/gr00t_flux_train_smoke/gpu_lora",
        "default_summary_json": "agent/artifacts/apple_recap_flux_graft/train_smoke_gpu_lora/train_smoke_summary.json",
    },
    "diagnostic": {
        "gate_semantics": "diagnostic_only_non_release_gate",
    },
}


def build_config() -> dict[str, Any]:
    return deepcopy(CONFIG)


__all__ = ["CONFIG", "build_config"]
