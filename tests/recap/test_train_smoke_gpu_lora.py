from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from collections.abc import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from configs.apple_recap.flux import train_smoke_gpu_lora
from work.demo_utils import paths as demo_paths
from work.recap.scripts import gr00t_recap_training_smoke


def _build_args(*cli_args: str):
    return gr00t_recap_training_smoke.build_parser().parse_args(list(cli_args))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_flag_value(cmd: Sequence[str], flag: str) -> str:
    for index, token in enumerate(cmd):
        if token == flag:
            return str(cmd[index + 1])
        if token.startswith(flag + "="):
            return str(token.split("=", 1)[1])
    raise AssertionError(f"missing flag {flag!r} in delegated command: {cmd!r}")


def _python_contract() -> dict[str, str]:
    return demo_paths.load_stage3_training_python_contract(REPO_ROOT)


def _fake_success_runner(cmd: Sequence[str], cwd: Path, log_path: Path) -> int:
    del cwd
    output_dir = Path(_parse_flag_value(cmd, "--output-dir")).resolve()
    summary_path = Path(_parse_flag_value(cmd, "--summary-json")).resolve()
    runtime_log_dir = Path(_parse_flag_value(cmd, "--runtime-log-dir")).resolve()
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text("fake gpu smoke run\n", encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint-4"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")
    _write_json(
        summary_path,
        {
            "wrapper_status": "ok",
            "dry_run": False,
            "launcher_python": _python_contract()["orchestrator_python"],
            "orchestrator_python": _python_contract()["orchestrator_python"],
            "delegate_runtime_python": _python_contract()["delegate_runtime_python"],
            "dataset_path": _parse_flag_value(cmd, "--dataset-path"),
            "output_dir": str(output_dir),
            "selected_checkpoint_path": str(checkpoint_dir),
            "selected_checkpoint_asset_path": str(checkpoint_dir / "model.safetensors"),
            "effective_config": {
                "max_steps": 4,
                "save_steps": 4,
                "save_total_limit": 1,
                "global_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "dataloader_num_workers": 0,
                "learning_rate": 1e-5,
                "num_gpus": 1,
                "tune_projector": False,
                "tune_diffusion_model": False,
                "use_wandb": False,
            },
        },
    )
    _write_json(
        output_dir / "final_model_config.json",
        {
            "tune_top_llm_layers": 0,
            "tune_llm": False,
            "tune_visual": False,
            "tune_projector": False,
            "tune_diffusion_model": False,
            "tune_vlln": True,
        },
    )
    return 0


def _fake_escalating_runner(cmd: Sequence[str], cwd: Path, log_path: Path) -> int:
    del cwd
    output_dir = Path(_parse_flag_value(cmd, "--output-dir")).resolve()
    summary_path = Path(_parse_flag_value(cmd, "--summary-json")).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("fake gpu smoke run\n", encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint-4"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model.safetensors").write_text("weights\n", encoding="utf-8")
    _write_json(
        summary_path,
        {
            "wrapper_status": "ok",
            "dry_run": False,
            "launcher_python": _python_contract()["orchestrator_python"],
            "orchestrator_python": _python_contract()["orchestrator_python"],
            "delegate_runtime_python": _python_contract()["delegate_runtime_python"],
            "dataset_path": _parse_flag_value(cmd, "--dataset-path"),
            "output_dir": str(output_dir),
            "effective_config": {
                "max_steps": 4,
                "save_steps": 4,
                "save_total_limit": 1,
                "global_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "dataloader_num_workers": 0,
                "learning_rate": 1e-5,
                "num_gpus": 1,
                "tune_projector": False,
                "tune_diffusion_model": False,
                "use_wandb": False,
            },
        },
    )
    _write_json(
        output_dir / "final_model_config.json",
        {
            "tune_top_llm_layers": 2,
            "tune_llm": True,
            "tune_visual": False,
            "tune_projector": False,
            "tune_diffusion_model": False,
            "tune_vlln": True,
        },
    )
    return 0


def test_gpu_config_requests_lora_first_but_keeps_smoke_budget_tiny() -> None:
    config: dict[str, Any] = train_smoke_gpu_lora.build_config()

    assert config["smoke_mode"] == "gpu_lora_or_head_only"
    assert config["execution"]["max_steps"] == 4
    assert config["execution"]["num_gpus"] == 1
    assert config["execution"]["save_total_limit"] == 1
    assert config["execution"]["use_wandb"] is False
    assert config["trainable_surface"]["preferred"] == "lora"
    assert config["trainable_surface"]["fallback"] == "head_only"


def test_gpu_cli_alias_normalizes_to_existing_gpu_surface(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    args = _build_args(
        "--smoke-mode",
        "gpu_lora",
        "--dataset-path",
        str(dataset_dir),
        "--output-dir",
        str(tmp_path / "output"),
        "--runtime-log-dir",
        str(tmp_path / "runtime_logs"),
    )

    assert args.smoke_mode == "gpu_lora_or_head_only"

    config_module, profile = gr00t_recap_training_smoke.load_training_smoke_config(
        smoke_mode="gpu_lora",
        config_module="",
    )
    plan = gr00t_recap_training_smoke.resolve_smoke_plan(
        args=args,
        profile=profile,
        config_module=config_module,
        repo_root=REPO_ROOT,
    )

    assert config_module == "configs.apple_recap.flux.train_smoke_gpu_lora"
    assert plan["smoke_mode"] == "gpu_lora_or_head_only"
    assert plan["requested_trainable_surface"] == "lora"
    assert plan["trainable_surface"] == "head_only_fallback"


def test_gpu_plan_falls_back_to_head_only_when_live_lora_surface_is_unavailable(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    args = _build_args(
        "--smoke-mode",
        "gpu_lora_or_head_only",
        "--dataset-path",
        str(dataset_dir),
        "--output-dir",
        str(tmp_path / "output"),
        "--runtime-log-dir",
        str(tmp_path / "runtime_logs"),
    )
    config_module, profile = gr00t_recap_training_smoke.load_training_smoke_config(
        smoke_mode=str(args.smoke_mode),
        config_module=str(args.config_module),
    )

    plan = gr00t_recap_training_smoke.resolve_smoke_plan(
        args=args,
        profile=profile,
        config_module=config_module,
        repo_root=REPO_ROOT,
    )

    assert plan["requested_trainable_surface"] == "lora"
    assert plan["trainable_surface"] == "head_only_fallback"
    assert plan["fallback_applied"] is True
    assert plan["lora_supported"] is False
    assert "lora" in plan["lora_support_reason"]
    assert plan["orchestrator_python"] == _python_contract()["orchestrator_python"]
    assert (
        plan["delegate_runtime_python"] == _python_contract()["delegate_runtime_python"]
    )


def test_gpu_materialization_records_checkpoint_and_diagnostic_fence(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    args = _build_args(
        "--smoke-mode",
        "gpu_lora_or_head_only",
        "--dataset-path",
        str(dataset_dir),
        "--output-dir",
        str(tmp_path / "output"),
        "--runtime-log-dir",
        str(tmp_path / "runtime_logs"),
        "--summary-json",
        str(tmp_path / "summary.json"),
    )
    config_module, profile = gr00t_recap_training_smoke.load_training_smoke_config(
        smoke_mode=str(args.smoke_mode),
        config_module=str(args.config_module),
    )

    payload = gr00t_recap_training_smoke.materialize_flux_training_smoke(
        args=args,
        profile=profile,
        config_module=config_module,
        repo_root=REPO_ROOT,
        delegate_runner=_fake_success_runner,
    )

    assert payload["wrapper_status"] == "ok"
    assert payload["smoke_mode"] == "gpu_lora_or_head_only"
    assert payload["trainable_surface"] == "head_only_fallback"
    assert payload["requested_trainable_surface"] == "lora"
    assert payload["max_steps"] == 4
    assert payload["num_gpus"] == 1
    assert payload["save_total_limit"] == 1
    assert payload["selected_checkpoint_path"] is not None
    assert payload["selected_checkpoint_asset_path"] is not None
    assert payload["diagnostic_only"] is True
    assert payload["mainline_authority"] is False
    assert payload["main_verdict_eligible"] is False
    assert payload["gate_semantics"] == "diagnostic_only_non_release_gate"
    assert payload["final_model_config_exists"] is True
    assert payload["orchestrator_python"] == _python_contract()["orchestrator_python"]
    assert (
        payload["delegate_runtime_python"]
        == _python_contract()["delegate_runtime_python"]
    )
    assert payload["trainability_gate_path"].endswith("trainability_gate.json")


def test_gpu_materialization_fail_closes_on_backbone_escalation(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    args = _build_args(
        "--smoke-mode",
        "gpu_lora_or_head_only",
        "--dataset-path",
        str(dataset_dir),
        "--output-dir",
        str(tmp_path / "output"),
        "--runtime-log-dir",
        str(tmp_path / "runtime_logs"),
    )
    config_module, profile = gr00t_recap_training_smoke.load_training_smoke_config(
        smoke_mode=str(args.smoke_mode),
        config_module=str(args.config_module),
    )

    payload = gr00t_recap_training_smoke.materialize_flux_training_smoke(
        args=args,
        profile=profile,
        config_module=config_module,
        repo_root=REPO_ROOT,
        delegate_runner=_fake_escalating_runner,
    )

    assert payload["wrapper_status"] == "blocked"
    assert "head-only smoke fencing" in str(payload["error"])
