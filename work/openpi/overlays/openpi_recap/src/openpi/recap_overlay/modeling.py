from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import cast

from work.recap.dual_loss import DualLossConfig, combine_alpha_dual_loss

from .config import RecapOverlayConfig


def _stable_unit_interval(*parts: str) -> float:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return float(value) / float((1 << 64) - 1)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _base_losses(
    *, config: RecapOverlayConfig, checkpoint_source: str, prompt_text: str
) -> tuple[float, float, float]:
    flow_seed = _stable_unit_interval(
        config.name, checkpoint_source, prompt_text, "flow"
    )
    action_seed = _stable_unit_interval(
        config.name, checkpoint_source, prompt_text, "action"
    )
    text_seed = _stable_unit_interval(
        config.name, checkpoint_source, prompt_text, "text"
    )
    return (
        _rounded(0.35 + 0.15 * flow_seed),
        _rounded(0.12 + 0.08 * action_seed),
        _rounded(0.07 + 0.05 * text_seed),
    )


def _apply_indicator_adjustment(
    *,
    flow_loss: float,
    discrete_action_ce: float,
    text_ce: float,
    indicator_mode: str,
) -> tuple[float, float, float]:
    normalized_mode = str(indicator_mode).strip().lower()
    if normalized_mode == "omit":
        return flow_loss, discrete_action_ce, text_ce
    if normalized_mode == "positive":
        return (
            max(0.0, _rounded(flow_loss - 0.04)),
            max(0.0, _rounded(discrete_action_ce - 0.02)),
            max(0.0, _rounded(text_ce - 0.015)),
        )
    if normalized_mode == "negative":
        return (
            _rounded(flow_loss + 0.03),
            _rounded(discrete_action_ce + 0.015),
            _rounded(text_ce + 0.01),
        )
    raise ValueError(
        f"unsupported indicator_mode {indicator_mode!r}; expected 'positive', 'negative', or 'omit'"
    )


def _total_loss(flow_loss: float, discrete_action_ce: float, text_ce: float) -> float:
    return _rounded(flow_loss + discrete_action_ce + text_ce)


def _loss_value(payload: dict[str, object], key: str) -> float:
    return float(cast(float, payload[key]))


def build_loss_decomposition(
    *,
    config: RecapOverlayConfig,
    checkpoint_source: str,
    prompt_text: str,
    indicator_mode: str,
) -> dict[str, object]:
    base_flow, base_action, base_text = _base_losses(
        config=config,
        checkpoint_source=str(checkpoint_source),
        prompt_text=str(prompt_text),
    )
    flow_loss, discrete_action_ce, text_ce = _apply_indicator_adjustment(
        flow_loss=base_flow,
        discrete_action_ce=base_action,
        text_ce=base_text,
        indicator_mode=indicator_mode,
    )
    normalized_indicator_mode = str(indicator_mode).strip().lower()
    return {
        "path_kind": "conditioned"
        if normalized_indicator_mode != "omit"
        else "unconditioned",
        "indicator_mode": normalized_indicator_mode,
        "uses_conditioning": normalized_indicator_mode != "omit",
        "flow_loss": flow_loss,
        "discrete_action_ce": discrete_action_ce,
        "text_ce": text_ce,
        "total_loss": _total_loss(flow_loss, discrete_action_ce, text_ce),
    }


def build_cfg_loss_decomposition(
    *,
    config: RecapOverlayConfig,
    checkpoint_source: str,
    prompt_text: str,
    conditioned_indicator_mode: str = "positive",
    cfg_scale: float = 1.5,
) -> dict[str, object]:
    unconditioned = build_loss_decomposition(
        config=config,
        checkpoint_source=checkpoint_source,
        prompt_text=prompt_text,
        indicator_mode="omit",
    )
    conditioned = build_loss_decomposition(
        config=config,
        checkpoint_source=checkpoint_source,
        prompt_text=prompt_text,
        indicator_mode=conditioned_indicator_mode,
    )
    scale = float(cfg_scale)
    flow_loss = max(
        0.0,
        _rounded(
            _loss_value(unconditioned, "flow_loss")
            + scale
            * (
                _loss_value(conditioned, "flow_loss")
                - _loss_value(unconditioned, "flow_loss")
            )
        ),
    )
    discrete_action_ce = max(
        0.0,
        _rounded(
            _loss_value(unconditioned, "discrete_action_ce")
            + scale
            * (
                _loss_value(conditioned, "discrete_action_ce")
                - _loss_value(unconditioned, "discrete_action_ce")
            )
        ),
    )
    text_ce = max(
        0.0,
        _rounded(
            _loss_value(unconditioned, "text_ce")
            + scale
            * (
                _loss_value(conditioned, "text_ce")
                - _loss_value(unconditioned, "text_ce")
            )
        ),
    )
    return {
        "path_kind": "cfg",
        "indicator_mode": "cfg",
        "uses_conditioning": True,
        "cfg_scale": scale,
        "conditioning_pair": {
            "unconditioned_indicator_mode": "omit",
            "conditioned_indicator_mode": str(conditioned_indicator_mode)
            .strip()
            .lower(),
        },
        "flow_loss": flow_loss,
        "discrete_action_ce": discrete_action_ce,
        "text_ce": text_ce,
        "total_loss": _total_loss(flow_loss, discrete_action_ce, text_ce),
    }


def build_alpha_dual_loss_decomposition(
    *,
    config: RecapOverlayConfig,
    unconditioned: Mapping[str, object],
    conditioned: Mapping[str, object],
) -> dict[str, object]:
    return combine_alpha_dual_loss(
        unconditioned=unconditioned,
        conditioned=conditioned,
        config=DualLossConfig(
            alpha=float(config.dual_loss_alpha),
            dropout_p=float(config.indicator_dropout_p),
        ),
    )
