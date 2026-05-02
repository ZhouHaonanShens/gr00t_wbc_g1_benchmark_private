from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, cast

from .common import JsonObject, as_float, as_int, as_str
from .data import (
    LiberoCriticExample,
    build_libero_recap_examples,
    build_task_max_steps_from_official_native_8d,
    official_image_payload_to_pil,
)
from .loader import load_critic_artifact
from .schema import CriticArtifact


DEFAULT_EVAL_SAMPLE_LIMIT = 8
PUBLIC_VALUE_SCALE = "raw_return"
INTERNAL_TASK_NORMALIZED_VALUE_SCALE = "task_normalized_return"


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _copy_sidecar(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _adapt_to_raw_return(
    *,
    artifact: CriticArtifact,
    bin_logits: list[float],
    bin_probs: list[float],
    value_v_internal: float,
    task_max_steps: int,
) -> dict[str, object]:
    if str(artifact.value_scale) == PUBLIC_VALUE_SCALE:
        factor = 1.0
    elif str(artifact.value_scale) == INTERNAL_TASK_NORMALIZED_VALUE_SCALE:
        if int(task_max_steps) <= 0:
            raise ValueError(f"task_max_steps must be > 0, got {task_max_steps}")
        factor = float(task_max_steps)
    else:
        raise ValueError(
            "unsupported critic value_scale for raw_return adaptation: "
            + f"{artifact.value_scale!r}"
        )
    return {
        "bin_centers": [float(center) * factor for center in artifact.bin_centers],
        "bin_logits": [float(value) for value in bin_logits],
        "bin_probs": [float(value) for value in bin_probs],
        "value_V_raw": float(value_v_internal) * factor,
        "internal_value": float(value_v_internal),
        "internal_value_scale": str(artifact.value_scale),
    }


def _build_critic_metadata(artifact: CriticArtifact) -> dict[str, object]:
    critic_dir = artifact.paths.critic_dir.resolve()
    canonical_metrics_path = critic_dir / "critic_metrics.json"
    canonical_provenance_path = critic_dir / "critic_provenance.json"
    adapter_name = (
        "identity"
        if str(artifact.value_scale) == PUBLIC_VALUE_SCALE
        else "task_normalized_return_to_raw_return"
    )
    return {
        "value_source": "critic",
        "value_scale": PUBLIC_VALUE_SCALE,
        "critic_dir": str(critic_dir),
        "critic_checkpoint_ref": str(critic_dir),
        "critic_metrics_path": str(
            canonical_metrics_path
            if canonical_metrics_path.is_file()
            else critic_dir / "metrics.json"
        ),
        "critic_provenance_path": str(
            canonical_provenance_path
            if canonical_provenance_path.is_file()
            else critic_dir / "provenance.json"
        ),
        "critic_internal_value_scale": str(artifact.value_scale),
        "value_adapter": adapter_name,
    }


def _load_qwen_runtime(artifact: CriticArtifact) -> tuple[Any, Any, str, Any, Any]:
    if artifact.model_payload is None:
        raise ValueError("critic artifact is missing torch model payload")
    architecture = artifact.model_payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("critic artifact model payload is missing architecture")
    trainable_state_dict = artifact.model_payload.get("trainable_state_dict")
    if not isinstance(trainable_state_dict, dict) or not trainable_state_dict:
        raise ValueError(
            "critic artifact model payload is missing trainable_state_dict"
        )
    import importlib

    modeling_mod = importlib.import_module("work.recap.critic_vlm.modeling")
    torch = importlib.import_module("torch")
    device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    processor_subdir = as_str(
        artifact.processor_config.get("hf_processor_subdir"),
        context="processor.hf_processor_subdir",
    )
    processor_dir = artifact.paths.critic_dir / "processor" / processor_subdir
    processor = modeling_mod.load_qwen3_vl_processor(str(processor_dir))
    backbone = modeling_mod.load_qwen3_vl_backbone(
        base_model=artifact.base_model,
        torch_dtype=modeling_mod.resolve_torch_dtype(device),
        attn_implementation=None,
    )
    critic = modeling_mod.Qwen3VLLateFusionCritic(
        backbone=backbone,
        hidden_size=modeling_mod.resolve_hidden_size(backbone),
        bin_centers=artifact.bin_centers,
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
    return processor, critic, device, torch, modeling_mod


def score_libero_recap_examples(
    *, checkpoint_dir: str | Path, examples: list[LiberoCriticExample]
) -> list[dict[str, object]]:
    artifact = load_critic_artifact(checkpoint_dir)
    processor, critic, device, torch, modeling_mod = _load_qwen_runtime(artifact)
    scored: list[dict[str, object]] = []
    for example in examples:
        text = modeling_mod.build_prompt_text(
            prompt_raw=example.prompt_raw,
            use_prompt=True,
        )
        model_inputs = modeling_mod.prepare_processor_inputs(
            processor=processor,
            texts=[text],
            images=[official_image_payload_to_pil(example.image_payload)],
        )
        model_inputs = {
            str(key): value.to(device) if hasattr(value, "to") else value
            for key, value in cast(dict[str, Any], model_inputs).items()
        }
        proprio = torch.tensor([example.proprio], dtype=torch.float32, device=device)
        t_norm = torch.tensor([[example.t_norm]], dtype=torch.float32, device=device)
        with torch.no_grad():
            output = critic(model_inputs=model_inputs, proprio=proprio, t_norm=t_norm)
        logits_t = output["logits"].detach().float().cpu().reshape(-1)
        probs_t = output["probs"].detach().float().cpu().reshape(-1)
        value_t = output["value_V_raw"].detach().float().cpu().reshape(-1)
        adapted = _adapt_to_raw_return(
            artifact=artifact,
            bin_logits=[float(value) for value in logits_t.tolist()],
            bin_probs=[float(value) for value in probs_t.tolist()],
            value_v_internal=float(value_t[0].item()),
            task_max_steps=int(example.task_max_steps),
        )
        scored.append(
            {
                "sample_id": example.sample_id,
                "episode_index": example.episode_index,
                "frame_index": example.frame_index,
                "prompt_raw": example.prompt_raw,
                "task_key": example.task_key,
                "task_max_steps": example.task_max_steps,
                "target_empirical_return": example.empirical_return,
                "target_normalized_return": example.normalized_return,
                "target_bin_index": example.target_bin_index,
                "predicted_raw_return": adapted["value_V_raw"],
                "predicted_internal_value": adapted["internal_value"],
                "predicted_internal_value_scale": adapted["internal_value_scale"],
                "bin_centers_raw_return": adapted["bin_centers"],
                "bin_logits": adapted["bin_logits"],
                "bin_probs": adapted["bin_probs"],
            }
        )
    return scored


def build_episode_value_predictions(
    *,
    checkpoint_dir: str | Path,
    dataset_dir: str | Path,
    episode_indices: list[int] | tuple[int, ...],
) -> dict[int, list[float]]:
    task_max_steps = build_task_max_steps_from_official_native_8d(dataset_dir)
    examples, _ = build_libero_recap_examples(
        dataset_dir,
        bin_centers=load_critic_artifact(checkpoint_dir).bin_centers,
        split_name="materialize",
        episode_indices=episode_indices,
        task_max_steps=task_max_steps,
    )
    scored = score_libero_recap_examples(
        checkpoint_dir=checkpoint_dir, examples=examples
    )
    out: dict[int, list[float]] = {}
    for row in scored:
        row_payload = cast(dict[str, object], row)
        episode_index = as_int(
            row_payload["episode_index"], context="scored.episode_index"
        )
        out.setdefault(episode_index, []).append(
            as_float(
                row_payload["predicted_raw_return"],
                context="scored.predicted_raw_return",
            )
        )
    return out


def run_libero_recap_critic_evaluation(
    *,
    checkpoint_dir: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    sample_limit: int | None = None,
) -> dict[str, object]:
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact = load_critic_artifact(checkpoint_root)
    effective_sample_limit = as_int(
        sample_limit
        or artifact.metrics.get("val_sample_count", DEFAULT_EVAL_SAMPLE_LIMIT),
        context="eval.sample_limit",
    )
    examples, task_max_steps = build_libero_recap_examples(
        dataset_dir,
        bin_centers=artifact.bin_centers,
        split_name="val",
        sample_limit=effective_sample_limit,
        task_max_steps=build_task_max_steps_from_official_native_8d(dataset_dir),
    )
    scored = score_libero_recap_examples(
        checkpoint_dir=checkpoint_root, examples=examples
    )
    raw_abs_errors = [
        abs(
            as_float(
                cast(dict[str, object], row)["predicted_raw_return"],
                context="scored.predicted_raw_return",
            )
            - as_float(
                cast(dict[str, object], row)["target_empirical_return"],
                context="scored.target_empirical_return",
            )
        )
        for row in scored
    ]
    normalized_abs_errors = [
        abs(
            as_float(
                cast(dict[str, object], row)["predicted_raw_return"],
                context="scored.predicted_raw_return",
            )
            / float(
                as_int(
                    cast(dict[str, object], row)["task_max_steps"],
                    context="scored.task_max_steps",
                )
            )
            - as_float(
                cast(dict[str, object], row)["target_normalized_return"],
                context="scored.target_normalized_return",
            )
        )
        for row in scored
    ]
    metrics: JsonObject = {
        "schema_version": "openpi_recap_critic_eval_metrics_v1",
        "sample_count": int(len(scored)),
        "split_name": "val",
        "public_value_scale": "raw_return",
        "internal_value_scale": artifact.value_scale,
        "mean_abs_error_raw_return": float(
            sum(raw_abs_errors) / max(1, len(raw_abs_errors))
        ),
        "mean_abs_error_task_normalized_return": float(
            sum(normalized_abs_errors) / max(1, len(normalized_abs_errors))
        ),
        "critic_checkpoint_ref": str(checkpoint_root),
    }
    provenance: JsonObject = {
        "schema_version": "openpi_recap_critic_eval_provenance_v1",
        "checkpoint_dir": str(checkpoint_root),
        "dataset_dir": str(Path(dataset_dir).expanduser().resolve()),
        "sample_limit": int(effective_sample_limit),
        "sample_count": int(len(scored)),
        "public_value_scale": "raw_return",
        "internal_value_scale": artifact.value_scale,
        "critic_checkpoint_ref": str(checkpoint_root),
        "critic_metadata": _build_critic_metadata(artifact),
    }
    _write_json(output_root / "metrics.json", metrics)
    _write_json(output_root / "provenance.json", provenance)
    _copy_sidecar(output_root / "metrics.json", output_root / "critic_metrics.json")
    _copy_sidecar(
        output_root / "provenance.json", output_root / "critic_provenance.json"
    )
    _write_json(output_root / "predictions.json", {"predictions": scored})
    return {
        "metrics": metrics,
        "provenance": provenance,
        "predictions": scored,
        "task_max_steps": task_max_steps,
    }


__all__ = [
    "build_episode_value_predictions",
    "run_libero_recap_critic_evaluation",
    "score_libero_recap_examples",
]
