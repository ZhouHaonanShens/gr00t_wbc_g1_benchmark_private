# pyright: reportImportCycles=false
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from work.openpi.prompting.routes import (
    FIXEDADV_CONSTANT_CONSUMER_MODE,
    RECAP_RELABEL_CONSUMER_MODE,
    build_phase1_prompt_provenance,
    build_phase1_prompt_route,
    build_runtime_prompt_route,
)
from work.openpi.recap.indicator_modes import (
    INDICATOR_MODE_TRAIN_INFORMATIVE,
    INDICATOR_MODE_TRAIN_SHUFFLED,
    indicator_mode_train_from_consumer_mode,
)
from work.openpi.serve.provenance import resolve_critic_checkpoint_ref
from work.recap import text_indicator


RUNTIME_INDICATOR_CFG = "cfg"
RUNTIME_INDICATOR_CLI_MODES: tuple[str, ...] = (
    text_indicator.TEXT_INDICATOR_POSITIVE,
    text_indicator.TEXT_INDICATOR_NEGATIVE,
    text_indicator.TEXT_INDICATOR_OMIT,
    RUNTIME_INDICATOR_CFG,
)


@dataclass(frozen=True)
class PromptSurfaceBundle:
    prompt_text: str
    prompt_provenance: dict[str, str]
    indicator_mode: str
    indicator_source: str
    prompt_text_surface: str
    consumer_mode: str
    fixed_indicator_mode: str | None
    critic_checkpoint_ref: str
    authoritative_carrier_text: str | None
    authoritative_carrier_source: str
    authoritative_carrier_matches_prompt_text: bool


@dataclass(frozen=True)
class RuntimeIndicatorConfig:
    requested_indicator_mode: str
    indicator_mode: str
    indicator_source: str
    consumer_mode: str
    fixed_indicator_mode: str | None
    critic_checkpoint_ref: str


def _as_mapping(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        return {}
    return cast(Mapping[str, object], raw)


def _extract_training_route(
    train_manifest: Mapping[str, object] | None,
    checkpoint_provenance: Mapping[str, object] | None,
) -> Mapping[str, object]:
    for payload, key in (
        (checkpoint_provenance, "variant_derivation"),
        (train_manifest, "training_route"),
        (checkpoint_provenance, "training_route"),
    ):
        if payload is None:
            continue
        route = _as_mapping(payload.get(key))
        if route:
            return route
    return {}


def normalize_runtime_indicator_mode(
    raw: object,
    *,
    field_name: str = "indicator_mode",
) -> str:
    if raw is None:
        raise ValueError(
            f"{field_name} is required; expected positive|negative|omit|cfg"
        )
    if not isinstance(raw, str):
        raise TypeError(
            f"{field_name} must be a string; expected positive|negative|omit|cfg, got {type(raw).__name__}"
        )
    mode = raw.strip().lower()
    if mode == RUNTIME_INDICATOR_CFG:
        return mode
    return text_indicator.normalize_indicator_mode(mode, field_name=field_name)


def _cfg_variant_default(variant: str) -> tuple[str, str, str | None]:
    normalized_variant = str(variant).strip().lower()
    if normalized_variant == "stock_libero_ref_v1" or normalized_variant == "stock":
        return text_indicator.TEXT_INDICATOR_OMIT, "cfg.stock_default", None
    if "fixedadv" in normalized_variant:
        return (
            text_indicator.TEXT_INDICATOR_OMIT,
            "cfg.variant_fixedadv_default",
            "omit",
        )
    return (
        text_indicator.TEXT_INDICATOR_POSITIVE,
        "cfg.recap_runtime_default_positive",
        None,
    )


def resolve_runtime_indicator_config(
    *,
    requested_indicator_mode: object,
    variant: str,
    train_manifest: Mapping[str, object] | None = None,
    checkpoint_provenance: Mapping[str, object] | None = None,
) -> RuntimeIndicatorConfig:
    requested_mode = normalize_runtime_indicator_mode(
        requested_indicator_mode,
        field_name="indicator_mode",
    )
    consumer_mode = RECAP_RELABEL_CONSUMER_MODE
    fixed_indicator_mode: str | None = None
    if requested_mode != RUNTIME_INDICATOR_CFG:
        training_route = _extract_training_route(train_manifest, checkpoint_provenance)
        consumer_mode = (
            str(
                training_route.get("consumer_mode", RECAP_RELABEL_CONSUMER_MODE)
            ).strip()
            or RECAP_RELABEL_CONSUMER_MODE
        )
        return RuntimeIndicatorConfig(
            requested_indicator_mode=requested_mode,
            indicator_mode=requested_mode,
            indicator_source="cli.indicator_mode",
            consumer_mode=consumer_mode,
            fixed_indicator_mode=fixed_indicator_mode,
            critic_checkpoint_ref=resolve_critic_checkpoint_ref(
                variant=variant,
                train_manifest=train_manifest,
                checkpoint_provenance=checkpoint_provenance,
            ),
        )

    training_route = _extract_training_route(train_manifest, checkpoint_provenance)
    consumer_mode = (
        str(training_route.get("consumer_mode", RECAP_RELABEL_CONSUMER_MODE)).strip()
        or RECAP_RELABEL_CONSUMER_MODE
    )
    indicator_mode_train = indicator_mode_train_from_consumer_mode(consumer_mode)
    raw_fixed_indicator_mode = training_route.get("fixed_indicator_mode")
    fixed_text = str(raw_fixed_indicator_mode or "").strip()
    if fixed_text:
        fixed_indicator_mode = text_indicator.normalize_indicator_mode(
            fixed_text,
            field_name="fixed_indicator_mode",
        )
        indicator_mode = fixed_indicator_mode
        indicator_source = "cfg.fixed_indicator_mode"
    elif consumer_mode == FIXEDADV_CONSTANT_CONSUMER_MODE:
        indicator_mode = text_indicator.TEXT_INDICATOR_OMIT
        fixed_indicator_mode = text_indicator.TEXT_INDICATOR_OMIT
        indicator_source = "cfg.consumer_mode.fixedadv_constant"
    elif indicator_mode_train in {
        INDICATOR_MODE_TRAIN_INFORMATIVE,
        INDICATOR_MODE_TRAIN_SHUFFLED,
    }:
        indicator_mode = text_indicator.TEXT_INDICATOR_POSITIVE
        indicator_source = f"cfg.consumer_mode.{consumer_mode}"
    else:
        indicator_mode, indicator_source, fixed_indicator_mode = _cfg_variant_default(
            variant
        )
    return RuntimeIndicatorConfig(
        requested_indicator_mode=requested_mode,
        indicator_mode=indicator_mode,
        indicator_source=indicator_source,
        consumer_mode=consumer_mode,
        fixed_indicator_mode=fixed_indicator_mode,
        critic_checkpoint_ref=resolve_critic_checkpoint_ref(
            variant=variant,
            train_manifest=train_manifest,
            checkpoint_provenance=checkpoint_provenance,
        ),
    )


def build_training_prompt_bundle(
    label_row: Mapping[str, object],
    *,
    consumer_mode: str,
    fixed_indicator_mode: str | None,
    critic_checkpoint_ref: str = "not_applicable",
) -> PromptSurfaceBundle:
    prompt_spec = build_phase1_prompt_route(
        label_row,
        consumer_mode=consumer_mode,
        fixed_indicator_mode=fixed_indicator_mode,
    )
    prompt_provenance = build_phase1_prompt_provenance(prompt_spec)
    return PromptSurfaceBundle(
        prompt_text=prompt_spec.prompt_text,
        prompt_provenance=prompt_provenance,
        indicator_mode=prompt_spec.indicator_mode,
        indicator_source=prompt_spec.indicator_source,
        prompt_text_surface=prompt_spec.prompt_text_surface,
        consumer_mode=prompt_spec.consumer_mode,
        fixed_indicator_mode=prompt_spec.fixed_indicator_mode,
        critic_checkpoint_ref=str(critic_checkpoint_ref).strip() or "not_applicable",
        authoritative_carrier_text=prompt_spec.authoritative_carrier_text,
        authoritative_carrier_source=prompt_spec.authoritative_carrier_source,
        authoritative_carrier_matches_prompt_text=(
            prompt_spec.authoritative_carrier_matches_prompt_text
        ),
    )


def build_runtime_prompt_from_inputs(
    prompt_raw: object,
    *,
    indicator_mode: object,
) -> str:
    return text_indicator.build_authoritative_carrier_text_v1(
        prompt_raw,
        text_indicator.normalize_indicator_mode(
            indicator_mode,
            field_name="indicator_mode",
        ),
    )


def build_runtime_prompt_bundle(
    prompt_raw: object,
    *,
    config: RuntimeIndicatorConfig,
) -> PromptSurfaceBundle:
    prompt_spec = build_runtime_prompt_route(
        prompt_raw=prompt_raw,
        indicator_mode=config.indicator_mode,
        indicator_source=config.indicator_source,
        consumer_mode=config.consumer_mode,
        fixed_indicator_mode=config.fixed_indicator_mode,
    )
    prompt_provenance = build_phase1_prompt_provenance(prompt_spec)
    return PromptSurfaceBundle(
        prompt_text=prompt_spec.prompt_text,
        prompt_provenance=prompt_provenance,
        indicator_mode=prompt_spec.indicator_mode,
        indicator_source=prompt_spec.indicator_source,
        prompt_text_surface=prompt_spec.prompt_text_surface,
        consumer_mode=prompt_spec.consumer_mode,
        fixed_indicator_mode=prompt_spec.fixed_indicator_mode,
        critic_checkpoint_ref=config.critic_checkpoint_ref,
        authoritative_carrier_text=prompt_spec.authoritative_carrier_text,
        authoritative_carrier_source=prompt_spec.authoritative_carrier_source,
        authoritative_carrier_matches_prompt_text=(
            prompt_spec.authoritative_carrier_matches_prompt_text
        ),
    )


__all__ = [
    "PromptSurfaceBundle",
    "RUNTIME_INDICATOR_CFG",
    "RUNTIME_INDICATOR_CLI_MODES",
    "RuntimeIndicatorConfig",
    "build_runtime_prompt_from_inputs",
    "build_runtime_prompt_bundle",
    "build_training_prompt_bundle",
    "normalize_runtime_indicator_mode",
    "resolve_runtime_indicator_config",
]
