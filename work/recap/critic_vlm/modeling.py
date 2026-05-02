from __future__ import annotations

from dataclasses import dataclass
from typing import Any


QWEN3_VL_LATE_FUSION_BACKEND_V1 = "qwen3_vl_late_fusion_v1"
DEFAULT_CRITIC_QUERY = "Estimate the raw return of the current observation."


@dataclass(frozen=True)
class Qwen3VLEnvironmentSupport:
    torch_version: str
    transformers_version: str
    peft_version: str | None
    qwen3vl_supported: bool
    blocker: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "peft_version": self.peft_version,
            "qwen3vl_supported": self.qwen3vl_supported,
            "blocker": self.blocker,
        }


def inspect_qwen3_vl_environment() -> Qwen3VLEnvironmentSupport:
    import importlib

    torch_mod = importlib.import_module("torch")
    transformers_mod = importlib.import_module("transformers")
    peft_version: str | None = None
    try:
        peft_mod = importlib.import_module("peft")
        peft_version = str(getattr(peft_mod, "__version__", "unknown"))
    except Exception:
        peft_version = None
    qwen_cls = getattr(transformers_mod, "Qwen3VLForConditionalGeneration", None)
    blocker: str | None = None
    if qwen_cls is None:
        blocker = (
            "qwen3_vl_transformers_support_missing: installed transformers="
            f"{getattr(transformers_mod, '__version__', 'unknown')} does not expose "
            "Qwen3VLForConditionalGeneration required by Qwen/Qwen3-VL-2B-Instruct; "
            "the public model card currently recommends transformers from source / >=4.57"
        )
    return Qwen3VLEnvironmentSupport(
        torch_version=str(getattr(torch_mod, "__version__", "unknown")),
        transformers_version=str(getattr(transformers_mod, "__version__", "unknown")),
        peft_version=peft_version,
        qwen3vl_supported=qwen_cls is not None,
        blocker=blocker,
    )


def load_qwen3_vl_processor(processor_name_or_path: str) -> Any:
    import importlib

    transformers_mod = importlib.import_module("transformers")
    auto_processor = getattr(transformers_mod, "AutoProcessor")
    try:
        return auto_processor.from_pretrained(
            processor_name_or_path, trust_remote_code=True
        )
    except (ImportError, ValueError) as exc:
        message = str(exc)
        if "Torchvision" not in message and "torchvision" not in message:
            raise
        if "requires `torchvision` to be installed" not in message and (
            "Qwen3VLVideoProcessor" not in message
            and "BaseVideoProcessor" not in message
            and "torchvision" not in message
            and "Torchvision" not in message
        ):
            raise
        auto_image_processor = getattr(transformers_mod, "AutoImageProcessor")
        auto_tokenizer = getattr(transformers_mod, "AutoTokenizer")
        qwen_processor_cls = getattr(transformers_mod, "Qwen3VLProcessor")
        transformers_utils_mod = importlib.import_module("transformers.utils")
        push_to_hub_mixin_cls = getattr(transformers_utils_mod, "PushToHubMixin")
        exported_base_video_processor_cls = getattr(
            transformers_mod, "BaseVideoProcessor"
        )

        class _ImageOnlyVideoProcessor(
            exported_base_video_processor_cls, push_to_hub_mixin_cls
        ):
            model_input_names = ["pixel_values_videos"]

            def __init__(self) -> None:
                try:
                    super().__init__()
                except ImportError as exc:
                    message = str(exc)
                    if "Torchvision" not in message and "torchvision" not in message:
                        raise
                self._auto_class = None
                self.video_processor_type = "image_only_fallback"

            def to_dict(self) -> dict[str, object]:
                return {"video_processor_type": self.video_processor_type}

        image_processor = auto_image_processor.from_pretrained(
            processor_name_or_path,
            trust_remote_code=True,
        )
        tokenizer = auto_tokenizer.from_pretrained(
            processor_name_or_path,
            trust_remote_code=True,
        )
        return qwen_processor_cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=_ImageOnlyVideoProcessor(),
        )


def load_qwen3_vl_backbone(
    *,
    base_model: str,
    torch_dtype: object,
    attn_implementation: str | None,
) -> Any:
    import importlib

    transformers_mod = importlib.import_module("transformers")
    qwen_cls = getattr(transformers_mod, "Qwen3VLForConditionalGeneration", None)
    if qwen_cls is None:
        raise RuntimeError(
            inspect_qwen3_vl_environment().blocker or "qwen3_vl_unavailable"
        )
    kwargs: dict[str, object] = {
        "pretrained_model_name_or_path": base_model,
        "trust_remote_code": True,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return qwen_cls.from_pretrained(**kwargs)


def resolve_torch_dtype(device: str) -> object:
    import torch

    if str(device).startswith("cuda"):
        return torch.float16
    return torch.float32


def resolve_hidden_size(backbone: Any) -> int:
    candidates = [
        getattr(getattr(backbone, "config", None), "hidden_size", None),
        getattr(
            getattr(getattr(backbone, "config", None), "text_config", None),
            "hidden_size",
            None,
        ),
        getattr(
            getattr(getattr(backbone, "model", None), "config", None),
            "hidden_size",
            None,
        ),
    ]
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return int(value)
    raise RuntimeError(
        "qwen3_vl_hidden_size_unresolved: could not infer language hidden size from backbone config"
    )


def locate_text_decoder_layers(backbone: Any) -> list[Any]:
    candidates = [
        getattr(
            getattr(getattr(backbone, "model", None), "language_model", None),
            "layers",
            None,
        ),
        getattr(getattr(backbone, "model", None), "layers", None),
        getattr(
            getattr(getattr(backbone, "language_model", None), "model", None),
            "layers",
            None,
        ),
        getattr(getattr(backbone, "language_model", None), "layers", None),
    ]
    for value in candidates:
        if value is not None and hasattr(value, "__len__"):
            return list(value)
    raise RuntimeError(
        "qwen3_vl_layers_unresolved: could not locate decoder layers for top-block LoRA"
    )


def apply_top_block_lora(
    backbone: Any,
    *,
    top_n: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    layers = locate_text_decoder_layers(backbone)
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    top_n = min(int(top_n), len(layers))
    layers_to_transform = list(range(len(layers) - top_n, len(layers)))
    config = LoraConfig(
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        layers_to_transform=layers_to_transform,
        layers_pattern="layers",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(backbone, config)


def build_prompt_text(*, prompt_raw: str, use_prompt: bool) -> str:
    prompt = str(prompt_raw).strip()
    if use_prompt and prompt:
        return f"Task: {prompt}\n{DEFAULT_CRITIC_QUERY}"
    return DEFAULT_CRITIC_QUERY


def _build_chat_template_input(
    processor: Any,
    *,
    text: str,
    has_image: bool,
) -> str:
    def _resolve_token(name: str, *, default: str) -> str:
        value = getattr(processor, name, None)
        if isinstance(value, str) and value.strip():
            return value
        tokenizer = getattr(processor, "tokenizer", None)
        token_value = getattr(tokenizer, name, None) if tokenizer is not None else None
        if isinstance(token_value, str) and token_value.strip():
            return token_value
        config = getattr(getattr(processor, "tokenizer", None), "config", None)
        token_id_name = f"{name}_id"
        token_id = getattr(config, token_id_name, None) if config is not None else None
        if tokenizer is not None and isinstance(token_id, int):
            converted = tokenizer.convert_ids_to_tokens(int(token_id))
            if isinstance(converted, str) and converted.strip():
                return converted
        return default

    def _raw_text_fallback() -> str:
        if not has_image:
            return text
        vision_start = _resolve_token(
            "vision_start_token",
            default="<|vision_start|>",
        )
        image_token = _resolve_token("image_token", default="<|image_pad|>")
        vision_end = _resolve_token(
            "vision_end_token",
            default="<|vision_end|>",
        )
        return f"{vision_start}{image_token}{vision_end}{text}"

    chat_template = getattr(processor, "chat_template", None)
    if hasattr(processor, "apply_chat_template") and isinstance(chat_template, str):
        if not chat_template.strip():
            return _raw_text_fallback()
        content: list[dict[str, object]] = []
        if has_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        try:
            rendered = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=False,
            )
        except ValueError as exc:
            message = str(exc)
            if "does not have a chat template" in message:
                return _raw_text_fallback()
            raise
        if isinstance(rendered, str) and rendered.strip():
            return rendered
    return _raw_text_fallback()


def prepare_processor_inputs(
    *,
    processor: Any,
    texts: list[str],
    images: list[object | None],
) -> dict[str, Any]:
    rendered_texts = [
        _build_chat_template_input(processor, text=text, has_image=image is not None)
        for text, image in zip(texts, images, strict=True)
    ]
    image_payload = None if all(image is None for image in images) else images
    batch = processor(
        text=rendered_texts,
        images=image_payload,
        padding=True,
        return_tensors="pt",
    )
    if not isinstance(batch, dict):
        batch = dict(batch)
    return batch


class _TwoLayerMlp:
    def __new__(
        cls,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> Any:
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.GELU(),
        )


class _OneLayerMlp:
    def __new__(cls, in_dim: int, out_dim: int) -> Any:
        import torch.nn as nn

        return nn.Sequential(nn.Linear(int(in_dim), int(out_dim)), nn.GELU())


class Qwen3VLLateFusionCritic(__import__("torch").nn.Module):
    def __init__(
        self,
        *,
        backbone: Any,
        hidden_size: int,
        bin_centers: list[float],
        proprio_dim: int,
        proprio_hidden_dim: int,
        t_hidden_dim: int,
        fusion_hidden_dim: int,
        use_proprio: bool,
        use_t_norm: bool,
    ) -> None:
        import torch
        import torch.nn as nn

        super().__init__()
        self.backbone = backbone
        self.use_proprio = bool(use_proprio)
        self.use_t_norm = bool(use_t_norm)
        self.proprio_mlp = (
            _TwoLayerMlp(proprio_dim, proprio_hidden_dim, proprio_hidden_dim)
            if self.use_proprio
            else None
        )
        self.t_mlp = _OneLayerMlp(1, t_hidden_dim) if self.use_t_norm else None
        fusion_in = int(hidden_size)
        if self.proprio_mlp is not None:
            fusion_in += int(proprio_hidden_dim)
        if self.t_mlp is not None:
            fusion_in += int(t_hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, int(fusion_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(fusion_hidden_dim), int(fusion_hidden_dim)),
            nn.GELU(),
        )
        self.value_head = nn.Linear(int(fusion_hidden_dim), len(bin_centers))
        self.register_buffer(
            "bin_centers",
            torch.tensor([float(x) for x in bin_centers], dtype=torch.float32),
            persistent=True,
        )

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_trainable_modules(self) -> None:
        for module in (self.proprio_mlp, self.t_mlp, self.fusion, self.value_head):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = True

    def keep_trainable_path_fp32(self) -> None:
        for module in (self.proprio_mlp, self.t_mlp, self.fusion, self.value_head):
            if module is None:
                continue
            _ = module.float()

    def _pool_hidden(self, hidden_states: Any, attention_mask: Any | None) -> Any:
        import torch

        if attention_mask is None:
            return hidden_states[:, -1, :]
        mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    def _cast_like_tensor(self, value: Any, ref_tensor: Any) -> Any:
        return value.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    def forward(
        self,
        *,
        model_inputs: dict[str, Any],
        proprio: Any | None,
        t_norm: Any | None,
    ) -> dict[str, Any]:
        import torch

        outputs = self.backbone(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not isinstance(hidden_states, (list, tuple)) or not hidden_states:
            raise RuntimeError(
                "qwen3_vl_hidden_states_missing: backbone forward did not return hidden_states"
            )
        last_hidden = hidden_states[-1]
        pooled = self._pool_hidden(last_hidden, model_inputs.get("attention_mask"))
        fusion_ref = self.value_head.weight
        pooled = self._cast_like_tensor(pooled, fusion_ref)
        parts = [pooled]
        if self.proprio_mlp is not None:
            if proprio is None:
                raise RuntimeError("proprio_missing_for_fusion")
            parts.append(self.proprio_mlp(self._cast_like_tensor(proprio, fusion_ref)))
        if self.t_mlp is not None:
            if t_norm is None:
                raise RuntimeError("t_norm_missing_for_fusion")
            parts.append(self.t_mlp(self._cast_like_tensor(t_norm, fusion_ref)))
        fused = self.fusion(torch.cat(parts, dim=-1))
        logits = self.value_head(fused)
        probs = torch.softmax(logits, dim=-1)
        bin_centers = self.bin_centers.to(device=probs.device, dtype=probs.dtype)
        value_v = (probs * bin_centers.reshape(1, -1)).sum(dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "value_V_raw": value_v,
        }


def select_trainable_state_dict(model: Any) -> dict[str, Any]:
    state_dict = model.state_dict()
    trainable = {
        name
        for name, param in model.named_parameters()
        if bool(getattr(param, "requires_grad", False))
    }
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if name in trainable
    }


def load_partial_state_dict(model: Any, trainable_state_dict: dict[str, Any]) -> None:
    missing, unexpected = model.load_state_dict(trainable_state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            "artifact_shape_invalid: unexpected keys when loading trainable_state_dict: "
            + ", ".join(str(x) for x in unexpected[:16])
        )
    critical_missing = [
        key
        for key in missing
        if key.startswith("fusion")
        or key.startswith("value_head")
        or key.startswith("proprio_mlp")
        or key.startswith("t_mlp")
        or ".lora_" in key
    ]
    if critical_missing:
        raise RuntimeError(
            "artifact_shape_invalid: missing critical trainable keys when loading artifact: "
            + ", ".join(str(x) for x in critical_missing[:16])
        )
