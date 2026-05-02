from __future__ import annotations

import importlib
from dataclasses import dataclass

from ..common import as_float, as_int, as_str
from ..schema import (
    POSITIVE_PATH_QWEN3_VL_LATE_FUSION_LOCAL,
    CriticArtifact,
    CriticInferenceResult,
    DatasetSample,
)
from .common import (
    ensure_finite_outputs,
    load_video_frame,
    move_batch_to_device,
    validate_processor_contract,
)


@dataclass
class QwenLateFusionInferenceService:
    artifact: CriticArtifact

    def run(self, sample: DatasetSample) -> CriticInferenceResult:
        frame_policy = validate_processor_contract(self.artifact)
        if self.artifact.model_payload is None:
            raise ValueError(
                "artifact_backend_unavailable: qwen3_vl_late_fusion_v1 requires torch model payload"
            )

        processor_subdir = as_str(
            self.artifact.processor_config.get("hf_processor_subdir"),
            context="processor.hf_processor_subdir",
        )
        processor_dir = self.artifact.paths.critic_dir / "processor" / processor_subdir
        if not processor_dir.exists() or not processor_dir.is_dir():
            raise FileNotFoundError(
                f"artifact_backend_unavailable: missing HF processor directory {processor_dir}"
            )

        architecture = self.artifact.model_payload.get("architecture")
        if not isinstance(architecture, dict):
            raise ValueError(
                "artifact_shape_invalid: qwen backend model payload missing architecture object"
            )
        trainable_state_dict = self.artifact.model_payload.get("trainable_state_dict")
        if not isinstance(trainable_state_dict, dict) or not trainable_state_dict:
            raise ValueError(
                "artifact_shape_invalid: qwen backend model payload missing non-empty trainable_state_dict"
            )

        modeling_mod = importlib.import_module("work.recap.critic_vlm.modeling")
        torch = importlib.import_module("torch")
        device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
        processor = modeling_mod.load_qwen3_vl_processor(str(processor_dir))
        backbone = modeling_mod.load_qwen3_vl_backbone(
            base_model=self.artifact.base_model,
            torch_dtype=modeling_mod.resolve_torch_dtype(device),
            attn_implementation=None,
        )
        critic = modeling_mod.Qwen3VLLateFusionCritic(
            backbone=backbone,
            hidden_size=modeling_mod.resolve_hidden_size(backbone),
            bin_centers=self.artifact.bin_centers,
            proprio_dim=as_int(
                architecture.get("proprio_dim"), context="architecture.proprio_dim"
            ),
            proprio_hidden_dim=as_int(
                architecture.get("proprio_hidden_dim"),
                context="architecture.proprio_hidden_dim",
            ),
            t_hidden_dim=as_int(
                architecture.get("t_hidden_dim"), context="architecture.t_hidden_dim"
            ),
            fusion_hidden_dim=as_int(
                architecture.get("fusion_hidden_dim"),
                context="architecture.fusion_hidden_dim",
            ),
            use_proprio=bool(architecture.get("use_proprio", True)),
            use_t_norm=bool(architecture.get("use_t_norm", True)),
        )
        critic.freeze_backbone()
        critic.unfreeze_trainable_modules()
        critic.backbone = modeling_mod.apply_top_block_lora(
            critic.backbone,
            top_n=as_int(
                architecture.get("top_n_lora_blocks"),
                context="architecture.top_n_lora_blocks",
            ),
            lora_rank=as_int(
                architecture.get("lora_rank"), context="architecture.lora_rank"
            ),
            lora_alpha=as_int(
                architecture.get("lora_alpha"), context="architecture.lora_alpha"
            ),
            lora_dropout=as_float(
                architecture.get("lora_dropout"),
                context="architecture.lora_dropout",
            ),
        )
        critic.unfreeze_trainable_modules()
        critic = critic.to(device)
        critic.keep_trainable_path_fp32()
        modeling_mod.load_partial_state_dict(critic, trainable_state_dict)
        critic.eval()

        if sample.video_rel is None:
            raise ValueError(
                "artifact_backend_unavailable: dataset sample missing video_rel"
            )
        video_abs = sample.dataset_path / sample.video_rel
        image = load_video_frame(video_abs, sample.frame_index)
        text = modeling_mod.build_prompt_text(
            prompt_raw=sample.prompt_raw, use_prompt=True
        )
        model_inputs = modeling_mod.prepare_processor_inputs(
            processor=processor,
            texts=[text],
            images=[image],
        )
        model_inputs = move_batch_to_device(model_inputs, device)
        proprio_dim = as_int(
            architecture.get("proprio_dim"), context="architecture.proprio_dim"
        )
        proprio = torch.zeros((1, int(proprio_dim)), dtype=torch.float32, device=device)
        t_norm_den = max(1, int(sample.episode_length - 1))
        t_norm_value = float(float(sample.t) / float(t_norm_den))
        t_norm = torch.tensor(
            [[float(t_norm_value)]], dtype=torch.float32, device=device
        )

        with torch.no_grad():
            output = critic(model_inputs=model_inputs, proprio=proprio, t_norm=t_norm)
        logits_t = output["logits"].detach().float().cpu().reshape(-1)
        probs_t = output["probs"].detach().float().cpu().reshape(-1)
        value_t = output["value_V_raw"].detach().float().cpu().reshape(-1)
        logits = [float(value) for value in logits_t.tolist()]
        probs = [float(value) for value in probs_t.tolist()]
        value_v = float(value_t[0].item())
        ensure_finite_outputs(logits=logits, probs=probs, value_v=value_v)
        return CriticInferenceResult(
            critic_type=self.artifact.critic_type,
            artifact_version=self.artifact.artifact_version,
            bin_logits=logits,
            bin_probs=probs,
            value_V_raw=float(value_v),
            positive_path_kind=POSITIVE_PATH_QWEN3_VL_LATE_FUSION_LOCAL,
            processor_frame_policy=str(frame_policy),
        )
