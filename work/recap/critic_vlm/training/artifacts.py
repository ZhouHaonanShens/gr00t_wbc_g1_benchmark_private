from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..common import JsonObject, as_int, write_json
from ..manifest import VlmCriticManifest
from ..schema import (
    ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1,
    QWEN3_VL_LATE_FUSION_BACKEND_V1,
)
from .contracts import (
    FUSION_HIDDEN_DIM,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    PROPRIO_DIM,
    PROPRIO_HIDDEN_DIM,
    RANKING_LOSS_WEIGHT,
    T_HIDDEN_DIM,
    UPGRADE_PENDING,
    VALUE_SCALE_TASK_NORMALIZED_RETURN,
    TrainConfig,
    TrainResult,
    WarmstartPlan,
)
from .runtime import load_modeling_module, runtime_import_torch, safe_float


@dataclass
class TrainingArtifactWriter:
    repo_root: Path
    config: TrainConfig
    train_manifest: VlmCriticManifest
    val_manifest: VlmCriticManifest
    support: Any
    device: str
    warmstart_plan: WarmstartPlan
    effective_side_channels: list[str]
    bin_centers: list[float]

    def critic_dir(self) -> Path:
        return (
            self.repo_root / "agent" / "artifacts" / "critics" / self.config.critic_tag
        ).resolve()

    def _prompt_template_kind(self) -> str:
        if str(self.config.prompt_text_mode) == "constant_query_only":
            return "constant_critic_query_only"
        return "fixed_critic_query_plus_optional_prompt_raw"

    def _runtime_summary(self) -> JsonObject:
        return {
            "support": self.support.to_json(),
            "device": self.device,
            "train_manifest": self.train_manifest.to_json(),
            "val_manifest": self.val_manifest.to_json(),
            "warmstart_plan": self.warmstart_plan.to_json(),
            "effective_input_policy": {
                "prompt_text_mode": str(self.config.prompt_text_mode),
                "use_proprio": bool(self.config.use_proprio),
                "use_t_norm": bool(self.config.use_t_norm),
                "effective_side_channels": list(self.effective_side_channels),
            },
        }

    def _processor_config(self) -> JsonObject:
        return {
            "task_text_field": "prompt_raw",
            "frame_policy": "current_step_index",
            "allow_future_frames": False,
            "side_channels": list(self.effective_side_channels),
            "hf_processor_subdir": "hf_processor",
            "input_prompt_template_kind": self._prompt_template_kind(),
            "prompt_text_mode": str(self.config.prompt_text_mode),
            "sample_mode": self.train_manifest.sample_mode,
        }

    def _config_json(self) -> JsonObject:
        return {
            "artifact_version": ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1,
            "critic_type": ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1,
            "base_model": self.config.base_model,
            "value_scale": VALUE_SCALE_TASK_NORMALIZED_RETURN,
            "upgrade_pending": UPGRADE_PENDING,
            "smoke_backend": QWEN3_VL_LATE_FUSION_BACKEND_V1,
            "top_n_lora_blocks": int(self.config.top_n_lora_blocks),
            "prompt_text_mode": str(self.config.prompt_text_mode),
            "use_proprio": bool(self.config.use_proprio),
            "use_t_norm": bool(self.config.use_t_norm),
            "bin_centers": [float(x) for x in self.bin_centers],
        }

    def build_metrics(
        self,
        *,
        train_sample_count: int,
        val_sample_count: int,
        warmstart_history: list[JsonObject],
        formal_history: list[JsonObject],
        best_val_loss: float,
    ) -> JsonObject:
        return {
            "warmstart_phase_done": True,
            "warmstart_phase_used_data": bool(self.warmstart_plan.phase_used_data),
            "warmstart_note": self.warmstart_plan.note,
            "warmstart_public_sample_count": int(
                self.warmstart_plan.public_sample_count
            ),
            "warmstart_used_dataset_roots": list(
                self.warmstart_plan.used_dataset_roots
            ),
            "formal_task_fit_done": True,
            "lora_rank": int(LORA_RANK),
            "lora_alpha": int(LORA_ALPHA),
            "lora_dropout": float(LORA_DROPOUT),
            "top_n_lora_blocks": int(self.config.top_n_lora_blocks),
            "ranking_loss_weight": float(RANKING_LOSS_WEIGHT),
            "upgrade_pending": UPGRADE_PENDING,
            "prompt_text_mode": str(self.config.prompt_text_mode),
            "use_proprio": bool(self.config.use_proprio),
            "use_t_norm": bool(self.config.use_t_norm),
            "effective_side_channels": list(self.effective_side_channels),
            "train_sample_count": int(train_sample_count),
            "val_sample_count": int(val_sample_count),
            "warmstart_history": warmstart_history,
            "formal_history": formal_history,
            "best_val_loss": safe_float(best_val_loss),
            "device": self.device,
            "environment": self.support.to_json(),
        }

    def build_provenance(self) -> JsonObject:
        return {
            "task": "vlm_critic_train",
            "artifact_backend": QWEN3_VL_LATE_FUSION_BACKEND_V1,
            "base_model": self.config.base_model,
            "dataset_path": str(self.train_manifest.dataset_path),
            "train_manifest": str(self.config.train_manifest),
            "val_manifest": str(self.config.val_manifest),
            "public_warmstart_manifest": str(self.config.public_warmstart_manifest),
            "warmstart_phase_used_data": bool(self.warmstart_plan.phase_used_data),
            "warmstart_plan": self.warmstart_plan.to_json(),
            "train_manifest_summary": self.train_manifest.to_json(),
            "val_manifest_summary": self.val_manifest.to_json(),
            "target_contract": {
                "value_source": "critic",
                "value_scale": VALUE_SCALE_TASK_NORMALIZED_RETURN,
                "normalization": "per_task_max_steps_from_formal_manifests",
                "episode_outcome_source": "manifest_empirical_return_G",
                "bin_encoding": "nearest_center",
                "bin_centers": [float(x) for x in self.bin_centers],
            },
            "runtime_summary": self._runtime_summary(),
            "training_hparams": {
                "batch_size": int(self.config.batch_size),
                "warmstart_epochs": int(self.config.warmstart_epochs),
                "formal_epochs": int(self.config.formal_epochs),
                "lr_head": float(self.config.lr_head),
                "lr_lora": float(self.config.lr_lora),
                "seed": int(self.config.seed),
                "attn_implementation": self.config.attn_implementation,
                "top_n_lora_blocks": int(self.config.top_n_lora_blocks),
                "prompt_text_mode": str(self.config.prompt_text_mode),
                "use_proprio": bool(self.config.use_proprio),
                "use_t_norm": bool(self.config.use_t_norm),
                "max_warmstart_samples": int(self.config.max_warmstart_samples)
                if self.config.max_warmstart_samples is not None
                else None,
                "max_train_samples": int(self.config.max_train_samples)
                if self.config.max_train_samples is not None
                else None,
                "max_val_samples": int(self.config.max_val_samples)
                if self.config.max_val_samples is not None
                else None,
            },
            "input_remediation": {
                "t7e_binding_applied": True,
                "diagnosis_json": (
                    str(self.config.remediation_diagnosis_json)
                    if self.config.remediation_diagnosis_json is not None
                    else None
                ),
                "prompt_text_mode": str(self.config.prompt_text_mode),
                "effective_side_channels": list(self.effective_side_channels),
                "use_proprio": bool(self.config.use_proprio),
                "use_t_norm": bool(self.config.use_t_norm),
            },
        }

    def build_split_manifest_ref(self) -> JsonObject:
        return {
            "train_manifest": str(self.config.train_manifest),
            "val_manifest": str(self.config.val_manifest),
            "train_source_build_json": str(self.train_manifest.source_build_json),
            "val_source_build_json": str(self.val_manifest.source_build_json),
            "formal_eval_scope": "isaac_only",
            "public_warmstart_manifest": str(self.config.public_warmstart_manifest),
        }

    def _save_processor_artifact(
        self,
        *,
        processor: Any,
        processor_dir: Path,
        processor_config: JsonObject,
    ) -> None:
        hf_dir = processor_dir / "hf_processor"
        hf_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(hf_dir)
        write_json(processor_dir / "processor_config.json", processor_config)

    def _save_model_artifact(
        self,
        *,
        critic: Any,
        model_path: Path,
        config_json: JsonObject,
    ) -> None:
        torch = runtime_import_torch()
        payload = {
            "backend_name": QWEN3_VL_LATE_FUSION_BACKEND_V1,
            "artifact_version": ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1,
            "base_model": config_json["base_model"],
            "architecture": {
                "proprio_dim": PROPRIO_DIM,
                "proprio_hidden_dim": PROPRIO_HIDDEN_DIM,
                "t_hidden_dim": T_HIDDEN_DIM,
                "fusion_hidden_dim": FUSION_HIDDEN_DIM,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "top_n_lora_blocks": as_int(
                    config_json["top_n_lora_blocks"],
                    context="config.top_n_lora_blocks",
                ),
                "use_proprio": bool(config_json["use_proprio"]),
                "use_t_norm": bool(config_json["use_t_norm"]),
                "bin_count": len(cast(list[float], config_json["bin_centers"])),
            },
            "trainable_state_dict": load_modeling_module().select_trainable_state_dict(
                critic
            ),
        }
        tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        _ = tmp_path.replace(model_path)

    def write(
        self,
        *,
        processor: Any,
        critic: Any,
        train_sample_count: int,
        val_sample_count: int,
        warmstart_history: list[JsonObject],
        formal_history: list[JsonObject],
        best_val_loss: float,
    ) -> TrainResult:
        config_json = self._config_json()
        processor_config = self._processor_config()
        metrics = self.build_metrics(
            train_sample_count=train_sample_count,
            val_sample_count=val_sample_count,
            warmstart_history=warmstart_history,
            formal_history=formal_history,
            best_val_loss=best_val_loss,
        )
        provenance = self.build_provenance()
        split_manifest_ref = self.build_split_manifest_ref()
        critic_dir = self.critic_dir()
        critic_dir.mkdir(parents=True, exist_ok=True)
        write_json(critic_dir / "config.json", config_json)
        write_json(
            critic_dir / "bin_centers.json",
            {"bin_centers": config_json["bin_centers"]},
        )
        write_json(critic_dir / "metrics.json", metrics)
        write_json(critic_dir / "provenance.json", provenance)
        write_json(critic_dir / "split_manifest_ref.json", split_manifest_ref)
        self._save_processor_artifact(
            processor=processor,
            processor_dir=critic_dir / "processor",
            processor_config=processor_config,
        )
        self._save_model_artifact(
            critic=critic,
            model_path=critic_dir / "model.pt",
            config_json=config_json,
        )
        return TrainResult(
            critic_dir=critic_dir,
            metrics=metrics,
            provenance=provenance,
        )
