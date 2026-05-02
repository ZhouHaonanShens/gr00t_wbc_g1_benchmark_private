from __future__ import annotations

from copy import deepcopy
from typing import Any


CONFIG = {
    "config_name": "flux_train_smoke_cpu",
    "smoke_mode": "cpu",
    "execution": {
        "dry_run": True,
        "max_steps": 1,
        "save_steps": 1,
        "save_total_limit": 1,
        "num_gpus": 0,
        "global_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "dataloader_num_workers": 0,
        "learning_rate": 1e-5,
        "use_wandb": False,
    },
    "trainable_surface": {
        "preferred": "head_only",
        "fallback": "head_only",
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
        "default_output_dir": "agent/artifacts/apple_recap_flux_graft/train_smoke_cpu",
        "default_runtime_log_dir": "agent/runtime_logs/gr00t_flux_train_smoke/cpu",
        "default_summary_json": "agent/artifacts/apple_recap_flux_graft/train_smoke_cpu/train_smoke_summary.json",
    },
    "diagnostic": {
        "gate_semantics": "diagnostic_only_non_release_gate",
    },
}


def build_config() -> dict[str, Any]:
    return deepcopy(CONFIG)


__all__ = ["CONFIG", "build_config"]
