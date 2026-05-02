from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import cast

from .common import as_float, as_int

from .data import build_libero_recap_dataloader, build_libero_recap_dataset_bundle
from .targets import (
    StepValueTargetInput,
    ValueTarget,
    build_task_max_steps,
    build_value_targets,
    build_value_targets_from_step_inputs,
    encode_value_to_bin_index,
    empirical_return_from_episode_outcome,
    normalize_empirical_return,
)
from .training import PublicWarmstartSample, TrainConfig, TrainResult, WarmstartPlan
from .training.artifacts import TrainingArtifactWriter
from .training.data import TrainingDataBundle
from .training.runtime import load_modeling_module, select_device, set_seed
from .training.workflow import CriticModelFactory, EpochRunner


DEFAULT_CONFIG = {
    "base_model": "Qwen/Qwen3-VL-2B-Instruct",
    "device": "auto",
    "batch_size": 1,
    "warmstart_epochs": 0,
    "formal_epochs": 1,
    "lr_head": 1e-4,
    "lr_lora": 5e-5,
    "seed": 7,
    "top_n_lora_blocks": 4,
    "prompt_text_mode": "manifest",
    "use_proprio": True,
    "use_t_norm": True,
    "bin_centers": None,
    "max_train_samples": 8,
    "max_val_samples": 4,
}


def _load_config_payload(config_path: str | Path) -> dict[str, object]:
    path = Path(config_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ValueError(
                f"unsupported config format for {path}; install PyYAML or provide JSON-compatible YAML: {exc}"
            ) from exc
        parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"critic config at {path} must be a mapping")
    payload: dict[str, object] = {
        str(key): value for key, value in DEFAULT_CONFIG.items()
    }
    payload.update({str(key): cast(object, value) for key, value in parsed.items()})
    return dict(payload)


def _sanitize_tag(raw: str) -> str:
    filtered = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw]
    return "".join(filtered).strip("._-") or "recap_critic"


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _copy_sidecar(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _effective_side_channels(*, use_proprio: bool, use_t_norm: bool) -> list[str]:
    side_channels = ["t_norm"] if bool(use_t_norm) else []
    if bool(use_proprio):
        side_channels = ["proprio", *side_channels]
    return side_channels


def _warmstart_plan() -> WarmstartPlan:
    return WarmstartPlan(
        phase_done=True,
        phase_used_data=False,
        available_local_roots=[],
        used_dataset_roots=[],
        public_sample_count=0,
        note="public_warmstart_disabled_for_official_native_8d_smoke",
    )


def run_libero_recap_critic_training(
    *,
    repo_root: Path,
    config_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    repo_root = Path(repo_root).resolve()
    output_dir_path = Path(output_dir).expanduser().resolve()
    if output_dir_path.exists():
        shutil.rmtree(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir_path / "manifests"

    payload = _load_config_payload(config_path)
    dataset_bundle = build_libero_recap_dataset_bundle(
        dataset_dir,
        manifest_dir=manifest_dir,
        bin_centers=cast(
            list[float] | tuple[float, ...] | None, payload.get("bin_centers")
        ),
        max_train_samples=cast(int | None, payload.get("max_train_samples")),
        max_val_samples=cast(int | None, payload.get("max_val_samples")),
    )
    critic_tag = _sanitize_tag(output_dir_path.name)
    train_config = TrainConfig(
        train_manifest=dataset_bundle.train.manifest_path,
        val_manifest=dataset_bundle.val.manifest_path,
        public_warmstart_manifest=dataset_bundle.public_warmstart_manifest_path,
        critic_tag=critic_tag,
        base_model=str(payload["base_model"]),
        device=str(payload["device"]),
        batch_size=as_int(payload["batch_size"], context="config.batch_size"),
        warmstart_epochs=as_int(
            payload["warmstart_epochs"], context="config.warmstart_epochs"
        ),
        formal_epochs=as_int(payload["formal_epochs"], context="config.formal_epochs"),
        lr_head=as_float(payload["lr_head"], context="config.lr_head"),
        lr_lora=as_float(payload["lr_lora"], context="config.lr_lora"),
        seed=as_int(payload["seed"], context="config.seed"),
        top_n_lora_blocks=as_int(
            payload["top_n_lora_blocks"], context="config.top_n_lora_blocks"
        ),
        attn_implementation=(
            str(payload.get("attn_implementation", "")).strip() or None
        ),
        prompt_text_mode=str(payload["prompt_text_mode"]),
        use_proprio=bool(payload["use_proprio"]),
        use_t_norm=bool(payload["use_t_norm"]),
        bin_centers=tuple(dataset_bundle.bin_centers),
        remediation_diagnosis_json=None,
        max_warmstart_samples=0,
        max_train_samples=as_int(
            payload["max_train_samples"], context="config.max_train_samples"
        ),
        max_val_samples=as_int(
            payload["max_val_samples"], context="config.max_val_samples"
        ),
    )
    support = load_modeling_module().inspect_qwen3_vl_environment()
    if support.blocker:
        raise RuntimeError(support.blocker)
    device = select_device(train_config.device)
    set_seed(train_config.seed)
    processor, critic = CriticModelFactory(train_config).build(
        bin_centers=dataset_bundle.bin_centers,
        device=device,
    )
    train_loader = build_libero_recap_dataloader(
        examples=dataset_bundle.train.examples,
        processor=processor,
        batch_size=train_config.batch_size,
        prompt_text_mode=train_config.prompt_text_mode,
        use_proprio=train_config.use_proprio,
        use_t_norm=train_config.use_t_norm,
        shuffle=True,
    )
    val_loader = build_libero_recap_dataloader(
        examples=dataset_bundle.val.examples,
        processor=processor,
        batch_size=train_config.batch_size,
        prompt_text_mode=train_config.prompt_text_mode,
        use_proprio=train_config.use_proprio,
        use_t_norm=train_config.use_t_norm,
        shuffle=False,
    )
    data_bundle = TrainingDataBundle(
        train_manifest=dataset_bundle.train.manifest,
        val_manifest=dataset_bundle.val.manifest,
        train_loader=train_loader,
        val_loader=val_loader,
        public_warmstart_loader=None,
        public_warmstart_samples=[],
        warmstart_plan=_warmstart_plan(),
        train_sample_count=len(dataset_bundle.train.examples),
        val_sample_count=len(dataset_bundle.val.examples),
    )
    history = EpochRunner(critic=critic, device=device).execute(
        data_bundle=data_bundle,
        config=train_config,
    )
    if history.best_state is not None:
        load_modeling_module().load_partial_state_dict(critic, history.best_state)
    artifact_writer = TrainingArtifactWriter(
        repo_root=repo_root,
        config=train_config,
        train_manifest=dataset_bundle.train.manifest,
        val_manifest=dataset_bundle.val.manifest,
        support=support,
        device=device,
        warmstart_plan=data_bundle.warmstart_plan,
        effective_side_channels=_effective_side_channels(
            use_proprio=train_config.use_proprio,
            use_t_norm=train_config.use_t_norm,
        ),
        bin_centers=dataset_bundle.bin_centers,
    )
    result = artifact_writer.write(
        processor=processor,
        critic=critic,
        train_sample_count=data_bundle.train_sample_count,
        val_sample_count=data_bundle.val_sample_count,
        warmstart_history=history.warmstart_history,
        formal_history=history.formal_history,
        best_val_loss=history.best_val_loss,
    )
    checkpoint_dir = output_dir_path / "best"
    _copy_tree(result.critic_dir, checkpoint_dir)
    _copy_sidecar(
        checkpoint_dir / "metrics.json", checkpoint_dir / "critic_metrics.json"
    )
    _copy_sidecar(
        checkpoint_dir / "provenance.json",
        checkpoint_dir / "critic_provenance.json",
    )
    _copy_sidecar(
        checkpoint_dir / "metrics.json", output_dir_path / "critic_metrics.json"
    )
    _copy_sidecar(
        checkpoint_dir / "provenance.json",
        output_dir_path / "critic_provenance.json",
    )
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "critic_dir": str(checkpoint_dir),
        "train_manifest": str(dataset_bundle.train.manifest_path),
        "val_manifest": str(dataset_bundle.val.manifest_path),
        "public_warmstart_manifest": str(dataset_bundle.public_warmstart_manifest_path),
        "train_sample_count": len(dataset_bundle.train.examples),
        "val_sample_count": len(dataset_bundle.val.examples),
        "metrics": result.metrics,
        "provenance": result.provenance,
    }
    with (output_dir_path / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return dict(summary)


__all__ = [
    "PublicWarmstartSample",
    "StepValueTargetInput",
    "TrainConfig",
    "TrainResult",
    "ValueTarget",
    "WarmstartPlan",
    "build_task_max_steps",
    "build_value_targets",
    "build_value_targets_from_step_inputs",
    "empirical_return_from_episode_outcome",
    "encode_value_to_bin_index",
    "normalize_empirical_return",
    "run_libero_recap_critic_training",
]
