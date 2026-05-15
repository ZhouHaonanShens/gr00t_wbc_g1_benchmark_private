# pyright: reportMissingImports=false, reportPossiblyUnboundVariable=false, reportRedeclaration=false, reportGeneralTypeIssues=false
from __future__ import annotations

import warnings
from typing import Any, Protocol

_import_error: ModuleNotFoundError | None = None

try:
    import torch
    import torch.nn.functional as F
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6 import gr00t_n1d6 as _gr00t_n1d6_module
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6, Gr00tN1d6ActionHead
    from torch import nn
    from transformers.feature_extraction_utils import BatchFeature
    from work.recap.dual_loss import (
        DEFAULT_ALPHA_DUAL_LOSS_COEFFICIENT,
        DEFAULT_DUAL_LOSS_DROPOUT_P,
        DualLossConfig,
        combine_alpha_dual_loss,
    )
except ModuleNotFoundError as exc:
    _import_error = exc


class SupportsRecapForward(Protocol):
    def forward(self, *args: object, **kwargs: object) -> object: ...

def resolve_r7_1_recipe_flags(config: Any) -> Any | None:
    from work.recap.r7_1_recipe_plumbing.flags import RecipeFlags

    recipe_flags = getattr(config, "r7_1_recipe_flags", None)
    if isinstance(recipe_flags, RecipeFlags) and not recipe_flags.is_default():
        return recipe_flags
    return None


def build_r7_1_dual_loss_metadata(config: Any) -> dict[str, object]:
    from work.recap.r7_1_recipe_plumbing.dual_loss_wiring import build_dual_loss_kwargs

    recipe_flags = resolve_r7_1_recipe_flags(config)
    if recipe_flags is None or not recipe_flags.enable_dual_loss:
        return {}
    return build_dual_loss_kwargs(recipe_flags)


if _import_error is not None:

    class GR00TRecapActionHead:
        def __init__(self, config: Any):
            del config
            raise ModuleNotFoundError(
                "GR00TRecapActionHead requires torch, transformers, and gr00t to be installed"
            ) from _import_error

    class GR00TRecapModel:
        def __init__(self, config: Any, transformers_loading_kwargs: Any | None = None):
            del config, transformers_loading_kwargs
            raise ModuleNotFoundError(
                "GR00TRecapModel requires torch, transformers, and gr00t to be installed"
            ) from _import_error

else:

    def _is_meta_tensor(value: object) -> bool:
        return bool(getattr(value, "is_meta", False))

    def _compute_all_tied_weights_keys_compat(model: Gr00tN1d6) -> dict[str, str]:
        def _normalize_tied_keys(raw: object) -> dict[str, str]:
            if not isinstance(raw, dict):
                return {}
            return {str(key): str(value) for key, value in raw.items()}

        expand_tied_keys = getattr(model, "get_expanded_tied_weights_keys", None)
        if callable(expand_tied_keys):
            try:
                return _normalize_tied_keys(expand_tied_keys(all_submodels=True) or {})
            except Exception:
                pass
        return _normalize_tied_keys(getattr(model, "_tied_weights_keys", None) or {})

    def install_gr00t_transformers_tied_weights_compat() -> None:
        if getattr(Gr00tN1d6, "_gr00t_tied_weights_compat", False):
            return

        original_init = Gr00tN1d6.__init__

        def _compat_init(self: Gr00tN1d6, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = _compute_all_tied_weights_keys_compat(self)

        Gr00tN1d6.__init__ = _compat_init
        Gr00tN1d6._gr00t_tied_weights_compat = True

    def install_gr00t_meta_init_beta_compat() -> None:
        beta_ctor = getattr(_gr00t_n1d6_module, "Beta", None)
        if beta_ctor is None:
            return
        if getattr(beta_ctor, "_gr00t_meta_init_compat", False):
            return

        class _MetaInitCompatibleBeta(beta_ctor):
            _gr00t_meta_init_compat = True

            def __init__(
                self,
                concentration1: object,
                concentration0: object,
                validate_args: bool | None = None,
            ):
                if validate_args is None and (
                    _is_meta_tensor(concentration1) or _is_meta_tensor(concentration0)
                ):
                    validate_args = False
                super().__init__(
                    concentration1,
                    concentration0,
                    validate_args=validate_args,
                )

        _MetaInitCompatibleBeta.__name__ = getattr(beta_ctor, "__name__", "Beta")
        _MetaInitCompatibleBeta.__qualname__ = getattr(
            beta_ctor,
            "__qualname__",
            _MetaInitCompatibleBeta.__qualname__,
        )
        _gr00t_n1d6_module.Beta = _MetaInitCompatibleBeta

    install_gr00t_transformers_tied_weights_compat()
    install_gr00t_meta_init_beta_compat()

    class GR00TRecapActionHead(Gr00tN1d6ActionHead):
        def __init__(self, config: Gr00tN1d6Config):
            super().__init__(config)
            advantage_dim = int(config.input_embedding_dim)
            self.advantage_embedding = nn.Linear(1, advantage_dim)
            nn.init.normal_(self.advantage_embedding.weight, mean=0.0, std=0.01)
            self._reset_advantage_embedding_bias_to_zero()

        def _reset_advantage_embedding_bias_to_zero(self) -> None:
            bias = getattr(self.advantage_embedding, "bias", None)
            if bias is None:
                return
            with torch.no_grad():
                bias.zero_()

        def freeze_advantage_embedding_bias_to_zero(self) -> None:
            self._reset_advantage_embedding_bias_to_zero()
            bias = getattr(self.advantage_embedding, "bias", None)
            if bias is not None:
                bias.requires_grad_(False)

        def set_trainable_parameters(
            self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
        ):
            super().set_trainable_parameters(
                tune_projector=tune_projector,
                tune_diffusion_model=tune_diffusion_model,
                tune_vlln=tune_vlln,
            )

        def set_frozen_modules_to_eval_mode(self):
            super().set_frozen_modules_to_eval_mode()
            if self.training and hasattr(self, "advantage_embedding"):
                if not any(
                    p.requires_grad for p in self.advantage_embedding.parameters()
                ):
                    self.advantage_embedding.eval()

        def _extract_advantage(self, action_input: BatchFeature) -> torch.Tensor | None:
            advantage = getattr(action_input, "advantage", None)
            if advantage is None and isinstance(action_input, dict):
                advantage = action_input.get("advantage")
            if advantage is None:
                return None
            advantage_tensor = torch.as_tensor(advantage)
            if advantage_tensor.ndim == 0:
                return advantage_tensor.reshape(1, 1)
            if advantage_tensor.ndim == 1:
                return advantage_tensor.unsqueeze(-1)
            flattened = advantage_tensor.reshape(advantage_tensor.shape[0], -1)
            if flattened.shape[-1] != 1:
                raise ValueError(
                    "Expected advantage with last dim 1, "
                    f"but got shape {tuple(advantage_tensor.shape)}"
                )
            return flattened

        def _apply_advantage_conditioning(
            self, state_features: torch.Tensor, action_input: BatchFeature
        ) -> torch.Tensor:
            advantage = self._extract_advantage(action_input)
            if advantage is None:
                return state_features
            if advantage.shape[0] != state_features.shape[0]:
                raise ValueError(
                    "Advantage batch dimension must match state features: "
                    f"{advantage.shape[0]} != {state_features.shape[0]}"
                )
            advantage = advantage.to(
                device=state_features.device, dtype=state_features.dtype
            )
            advantage_features = self.advantage_embedding(advantage)
            if advantage_features.shape[-1] != state_features.shape[-1]:
                raise ValueError(
                    "Advantage embedding dimension must match state feature dimension: "
                    f"{advantage_features.shape[-1]} != {state_features.shape[-1]}"
                )
            return state_features + advantage_features[:, None, :]

        def preview_advantage_conditioning(
            self,
            action_input: BatchFeature,
        ) -> dict[str, torch.Tensor | None]:
            embodiment_id = action_input.embodiment_id
            state_features = self.state_encoder(action_input.state, embodiment_id)
            advantage = self._extract_advantage(action_input)
            if advantage is None:
                return {
                    "advantage": None,
                    "state_features": state_features,
                    "advantage_features": None,
                    "conditioned_state_features": state_features,
                }
            if advantage.shape[0] != state_features.shape[0]:
                raise ValueError(
                    "Advantage batch dimension must match state features: "
                    f"{advantage.shape[0]} != {state_features.shape[0]}"
                )
            normalized_advantage = advantage.to(
                device=state_features.device,
                dtype=state_features.dtype,
            )
            advantage_features = self.advantage_embedding(normalized_advantage)
            if advantage_features.shape[-1] != state_features.shape[-1]:
                raise ValueError(
                    "Advantage embedding dimension must match state feature dimension: "
                    f"{advantage_features.shape[-1]} != {state_features.shape[-1]}"
                )
            conditioned_state_features = state_features + advantage_features[:, None, :]
            return {
                "advantage": normalized_advantage,
                "state_features": state_features,
                "advantage_features": advantage_features,
                "conditioned_state_features": conditioned_state_features,
            }

        def _encode_features(
            self, backbone_output: BatchFeature, action_input: BatchFeature
        ) -> BatchFeature:
            features = super()._encode_features(backbone_output, action_input)
            conditioned_state_features = self._apply_advantage_conditioning(
                features.state_features, action_input
            )
            return BatchFeature(
                data={
                    "backbone_features": features.backbone_features,
                    "state_features": conditioned_state_features,
                }
            )

        def _apply_shared_state_training_transforms(
            self,
            unconditioned_state_features: torch.Tensor,
            conditioned_state_features: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if self.state_dropout_prob > 0:
                do_dropout = (
                    torch.rand(
                        unconditioned_state_features.shape[0],
                        device=unconditioned_state_features.device,
                    )
                    < self.state_dropout_prob
                )
                do_dropout = do_dropout[:, None, None].to(
                    dtype=unconditioned_state_features.dtype
                )
                unconditioned_state_features = (
                    unconditioned_state_features * (1 - do_dropout)
                    + self.mask_token * do_dropout
                )
                conditioned_state_features = (
                    conditioned_state_features * (1 - do_dropout)
                    + self.mask_token * do_dropout
                )

            if self.training and self.state_additive_noise_scale > 0:
                warnings.warn(
                    (
                        "Adding Gaussian noise to state features with scale "
                        f"{self.state_additive_noise_scale}"
                    ),
                    category=RuntimeWarning,
                    stacklevel=2,
                )
                noise = (
                    torch.randn_like(unconditioned_state_features)
                    * self.state_additive_noise_scale
                )
                unconditioned_state_features = unconditioned_state_features + noise
                conditioned_state_features = conditioned_state_features + noise

            return unconditioned_state_features, conditioned_state_features

        def _decode_action_loss(
            self,
            *,
            backbone_output: BatchFeature,
            action_input: BatchFeature,
            state_features: torch.Tensor,
            noise: torch.Tensor,
            t: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            vl_embeds = backbone_output.backbone_features
            device = vl_embeds.device
            actions = action_input.action
            embodiment_id = action_input.embodiment_id
            t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
            noisy_trajectory = (1 - t) * noise + t * actions
            velocity = actions - noise
            action_features = self.action_encoder(
                noisy_trajectory, t_discretized, embodiment_id
            )

            if self.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device
                )
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, action_features), dim=1)
            vl_attn_mask = backbone_output.backbone_attention_mask

            if self.config.use_alternate_vl_dit:
                image_mask = backbone_output.image_mask
                backbone_attention_mask = backbone_output.backbone_attention_mask
                model_output, _ = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    encoder_attention_mask=vl_attn_mask,
                    timestep=t_discretized,
                    return_all_hidden_states=True,
                    image_mask=image_mask,
                    backbone_attention_mask=backbone_attention_mask,
                )
            else:
                model_output, _ = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    encoder_attention_mask=vl_attn_mask,
                    timestep=t_discretized,
                    return_all_hidden_states=True,
                )

            pred = self.action_decoder(model_output, embodiment_id)
            pred_actions = pred[:, -actions.shape[1] :]

            action_mask = action_input.action_mask
            action_loss = (
                F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
            )
            loss = action_loss.sum() / (action_mask.sum() + 1e-6)
            return {
                "loss": loss,
                "action_loss": action_loss,
                "action_mask": action_mask,
                "pred_actions": pred_actions,
            }

        def _loss_decomposition(self, loss: torch.Tensor) -> dict[str, torch.Tensor]:
            zero = loss.new_zeros(())
            return {
                "flow_loss": loss,
                "discrete_action_ce": zero,
                "text_ce": zero,
                "total_loss": loss,
            }

        def _dual_loss_config(self) -> DualLossConfig:
            recipe_flags = resolve_r7_1_recipe_flags(self.config)
            if recipe_flags is not None and recipe_flags.enable_dual_loss:
                return DualLossConfig(
                    alpha=float(recipe_flags.dual_loss_alpha),
                    dropout_p=float(recipe_flags.indicator_dropout_p),
                )
            alpha = float(
                getattr(
                    self.config,
                    "dual_loss_alpha",
                    DEFAULT_ALPHA_DUAL_LOSS_COEFFICIENT,
                )
            )
            dropout_p = float(
                getattr(
                    self.config,
                    "dual_loss_dropout_p",
                    DEFAULT_DUAL_LOSS_DROPOUT_P,
                )
            )
            return DualLossConfig(alpha=alpha, dropout_p=dropout_p)

        def forward(
            self, backbone_output: BatchFeature, action_input: BatchFeature
        ) -> dict[str, torch.Tensor]:
            self.set_frozen_modules_to_eval_mode()

            backbone_output = self.process_backbone_output(backbone_output)

            vl_embeds = backbone_output.backbone_features

            embodiment_id = action_input.embodiment_id
            unconditioned_state_features = self.state_encoder(
                action_input.state, embodiment_id
            )
            conditioned_state_features = self._apply_advantage_conditioning(
                unconditioned_state_features, action_input
            )
            (
                unconditioned_state_features,
                conditioned_state_features,
            ) = self._apply_shared_state_training_transforms(
                unconditioned_state_features,
                conditioned_state_features,
            )

            actions = action_input.action
            noise = torch.randn(
                actions.shape, device=actions.device, dtype=actions.dtype
            )
            t = self.sample_time(
                actions.shape[0], device=actions.device, dtype=actions.dtype
            )
            t = t[:, None, None]

            unconditioned_loss = self._decode_action_loss(
                backbone_output=backbone_output,
                action_input=action_input,
                state_features=unconditioned_state_features,
                noise=noise,
                t=t,
            )
            conditioned_loss = self._decode_action_loss(
                backbone_output=backbone_output,
                action_input=action_input,
                state_features=conditioned_state_features,
                noise=noise,
                t=t,
            )
            dual = combine_alpha_dual_loss(
                unconditioned=self._loss_decomposition(unconditioned_loss["loss"]),
                conditioned=self._loss_decomposition(conditioned_loss["loss"]),
                config=self._dual_loss_config(),
            )

            result = {
                "loss": dual["total_loss"],
                "action_loss": conditioned_loss["action_loss"],
                "action_mask": conditioned_loss["action_mask"],
                "backbone_features": vl_embeds,
                "state_features": conditioned_state_features,
                "unconditioned_state_features": unconditioned_state_features,
                "loss_unconditioned": unconditioned_loss["loss"],
                "loss_advantage_conditioned": conditioned_loss["loss"],
                "alpha_dual_loss": dual,
            }
            recipe_metadata = build_r7_1_dual_loss_metadata(self.config)
            if recipe_metadata:
                result["r7_1_dual_loss_kwargs"] = recipe_metadata
            return result

    class GR00TRecapModel(Gr00tN1d6):
        def __init__(
            self,
            config: Gr00tN1d6Config,
            transformers_loading_kwargs: dict[str, Any] | None = None,
        ):
            if transformers_loading_kwargs is None:
                transformers_loading_kwargs = {"trust_remote_code": True}
            super().__init__(
                config=config,
                transformers_loading_kwargs=transformers_loading_kwargs,
            )
            self.action_head = GR00TRecapActionHead(config)
            self.last_load_incompatible_keys: dict[str, list[str]] = {
                "missing_keys": [],
                "unexpected_keys": [],
            }

        def load_base_checkpoint_compatible(
            self,
            state_dict: dict[str, torch.Tensor],
        ) -> dict[str, list[str]]:
            incompatible = self.load_state_dict(state_dict, strict=False)
            report = {
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
            }
            self.last_load_incompatible_keys = report
            return report


__all__ = [
    "build_r7_1_dual_loss_metadata",
    "resolve_r7_1_recipe_flags",
    "GR00TRecapActionHead",
    "GR00TRecapModel",
    "SupportsRecapForward",
    "install_gr00t_meta_init_beta_compat",
]
