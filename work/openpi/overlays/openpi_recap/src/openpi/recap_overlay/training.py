from __future__ import annotations

from dataclasses import asdict

from .config import build_recap_config
from .modeling import (
    build_alpha_dual_loss_decomposition,
    build_cfg_loss_decomposition,
    build_loss_decomposition,
)


DEFAULT_PROMPT_TEXT = "place the mug on the coaster"


def build_smoke_forward_report(
    *,
    config_name: str,
    checkpoint_source: str,
    cfg_scale: float | None = None,
    prompt_text: str = DEFAULT_PROMPT_TEXT,
) -> dict[str, object]:
    config = build_recap_config(config_name)
    effective_cfg_scale = (
        float(config.cfg_default_scale) if cfg_scale is None else float(cfg_scale)
    )
    conditioned = build_loss_decomposition(
        config=config,
        checkpoint_source=str(checkpoint_source),
        prompt_text=str(prompt_text),
        indicator_mode="positive",
    )
    unconditioned = build_loss_decomposition(
        config=config,
        checkpoint_source=str(checkpoint_source),
        prompt_text=str(prompt_text),
        indicator_mode="omit",
    )
    cfg = build_cfg_loss_decomposition(
        config=config,
        checkpoint_source=str(checkpoint_source),
        prompt_text=str(prompt_text),
        conditioned_indicator_mode="positive",
        cfg_scale=effective_cfg_scale,
    )
    dual = build_alpha_dual_loss_decomposition(
        config=config,
        unconditioned=unconditioned,
        conditioned=conditioned,
    )
    return {
        "schema_version": "openpi_recap_smoke_forward_v1",
        "config_name": config.name,
        "checkpoint_source": str(checkpoint_source),
        "prompt_text": str(prompt_text),
        "config": asdict(config),
        "conditioned": conditioned,
        "unconditioned": unconditioned,
        "cfg": cfg,
        "dual": dual,
    }
