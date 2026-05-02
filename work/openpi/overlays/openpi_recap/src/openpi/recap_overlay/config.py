from __future__ import annotations

from dataclasses import dataclass

from work.recap.dual_loss import (
    DEFAULT_ALPHA_DUAL_LOSS_COEFFICIENT,
    DEFAULT_DUAL_LOSS_DROPOUT_P,
    DUAL_LOSS_FORMULA,
)

PI05_LIBERO_RECAP_CONFIG_NAME = "pi05_libero_recap"
BASE_CONFIG_NAME = "pi05_libero"
DEFAULT_CFG_SCALE = 1.5


@dataclass(frozen=True)
class RecapOverlayConfig:
    name: str
    base_config_name: str
    condition_on_advantage_text: bool
    conditioning_position: str
    flow_loss_key: str = "flow_loss"
    discrete_action_loss_key: str = "discrete_action_ce"
    text_loss_key: str = "text_ce"
    total_loss_key: str = "total_loss"
    supports_cfg: bool = True
    cfg_default_scale: float = DEFAULT_CFG_SCALE
    indicator_dropout_p: float = DEFAULT_DUAL_LOSS_DROPOUT_P
    dual_loss_enabled: bool = True
    dual_loss_alpha: float = DEFAULT_ALPHA_DUAL_LOSS_COEFFICIENT
    dual_loss_formula: str = DUAL_LOSS_FORMULA


PI05_LIBERO_RECAP_POLICY_METADATA = {
    "recap_enabled": True,
    "recap_base_config": BASE_CONFIG_NAME,
    "recap_cfg_supported": True,
    "recap_conditioning_position": "after_text_before_actions",
    "recap_loss_keys": ["flow_loss", "discrete_action_ce", "text_ce", "total_loss"],
    "recap_dual_loss_enabled": True,
    "recap_dual_loss_alpha": DEFAULT_ALPHA_DUAL_LOSS_COEFFICIENT,
    "recap_dual_loss_formula": DUAL_LOSS_FORMULA,
}


def build_recap_policy_metadata() -> dict[str, object]:
    return dict(PI05_LIBERO_RECAP_POLICY_METADATA)


def build_recap_config(config_name: str) -> RecapOverlayConfig:
    normalized_name = str(config_name).strip()
    if normalized_name != PI05_LIBERO_RECAP_CONFIG_NAME:
        raise ValueError(
            f"unsupported recap overlay config {config_name!r}; expected {PI05_LIBERO_RECAP_CONFIG_NAME!r}"
        )
    return RecapOverlayConfig(
        name=PI05_LIBERO_RECAP_CONFIG_NAME,
        base_config_name=BASE_CONFIG_NAME,
        condition_on_advantage_text=True,
        conditioning_position="after_text_before_actions",
    )
