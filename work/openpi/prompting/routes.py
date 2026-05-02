from __future__ import annotations

from collections.abc import Mapping

from work.openpi.recap.indicator_modes import (
    INDICATOR_MODE_TRAIN_FIXED_NEGATIVE,
    INDICATOR_MODE_TRAIN_FIXED_POSITIVE,
    INDICATOR_MODE_TRAIN_INFORMATIVE,
    INDICATOR_MODE_TRAIN_OMIT,
    INDICATOR_MODE_TRAIN_SHUFFLED,
    build_shuffled_indicator_mode,
    indicator_mode_train_from_consumer_mode,
)
from work.openpi.recap.prompt_builder import (
    CONDITIONING_MODE,
    PHASE1_PROMPT_ROUTE,
    PROMPT_TEXT_SURFACE_CANONICAL,
    PROMPT_TEXT_SURFACE_PROMPT_RAW_ONLY,
    PromptRouteSpec,
    build_phase1_prompt_text,
    build_runtime_prompt_route as _build_runtime_prompt_route,
    build_training_prompt_route,
    prompt_provenance_from_spec,
    prompt_text_surface_for_indicator_mode,
)
from work.recap import text_indicator


RECAP_RELABEL_CONSUMER_MODE = "informative_adv"
LEGACY_RECAP_RELABEL_CONSUMER_MODE = "recap_relabel"
FIXEDADV_CONSTANT_CONSUMER_MODE = "fixedadv_constant"
SHUFFLED_ADV_DIAG_CONSUMER_MODE = "shuffled_adv_diag"


def normalize_consumer_mode(raw: object) -> str:
    if raw is None:
        return RECAP_RELABEL_CONSUMER_MODE
    if not isinstance(raw, str):
        raise TypeError(
            "consumer_mode must be a string; expected informative|fixed_positive|fixed_negative|omit|shuffled or legacy aliases, "
            + f"got {type(raw).__name__}"
        )
    mode = raw.strip().lower()
    if mode == LEGACY_RECAP_RELABEL_CONSUMER_MODE:
        return RECAP_RELABEL_CONSUMER_MODE
    if mode not in {
        INDICATOR_MODE_TRAIN_INFORMATIVE,
        INDICATOR_MODE_TRAIN_FIXED_POSITIVE,
        INDICATOR_MODE_TRAIN_FIXED_NEGATIVE,
        INDICATOR_MODE_TRAIN_OMIT,
        INDICATOR_MODE_TRAIN_SHUFFLED,
        RECAP_RELABEL_CONSUMER_MODE,
        FIXEDADV_CONSTANT_CONSUMER_MODE,
        SHUFFLED_ADV_DIAG_CONSUMER_MODE,
    }:
        raise ValueError(
            "unknown consumer_mode "
            + f"{raw!r}; expected informative|fixed_positive|fixed_negative|omit|shuffled or {RECAP_RELABEL_CONSUMER_MODE}|{LEGACY_RECAP_RELABEL_CONSUMER_MODE}|{FIXEDADV_CONSTANT_CONSUMER_MODE}|{SHUFFLED_ADV_DIAG_CONSUMER_MODE}"
        )
    return mode


def resolve_fixed_indicator_mode(
    *, consumer_mode: str, fixed_indicator_mode: object | None
) -> str | None:
    indicator_mode_train = indicator_mode_train_from_consumer_mode(consumer_mode)
    if fixed_indicator_mode is None:
        if consumer_mode == FIXEDADV_CONSTANT_CONSUMER_MODE:
            return text_indicator.TEXT_INDICATOR_OMIT
        if indicator_mode_train == INDICATOR_MODE_TRAIN_FIXED_POSITIVE:
            return text_indicator.TEXT_INDICATOR_POSITIVE
        if indicator_mode_train == INDICATOR_MODE_TRAIN_FIXED_NEGATIVE:
            return text_indicator.TEXT_INDICATOR_NEGATIVE
        if indicator_mode_train == INDICATOR_MODE_TRAIN_OMIT:
            return text_indicator.TEXT_INDICATOR_OMIT
        return None
    mode = text_indicator.normalize_indicator_mode(
        fixed_indicator_mode,
        field_name="fixed_indicator_mode",
    )
    if (
        consumer_mode == FIXEDADV_CONSTANT_CONSUMER_MODE
        and mode != text_indicator.TEXT_INDICATOR_OMIT
    ):
        raise ValueError(
            "fixedadv_constant requires fixed_indicator_mode='omit' to neutralize the text carrier"
        )
    return mode


def build_shuffled_adv_diag_indicator_mode(label_row: Mapping[str, object]) -> str:
    return build_shuffled_indicator_mode(label_row)


def build_phase1_prompt_route(
    label_row: Mapping[str, object],
    *,
    consumer_mode: str = RECAP_RELABEL_CONSUMER_MODE,
    fixed_indicator_mode: str | None = None,
) -> PromptRouteSpec:
    normalized_consumer_mode = normalize_consumer_mode(consumer_mode)
    resolved_fixed_indicator_mode = resolve_fixed_indicator_mode(
        consumer_mode=normalized_consumer_mode,
        fixed_indicator_mode=fixed_indicator_mode,
    )
    return build_training_prompt_route(
        label_row,
        consumer_mode=normalized_consumer_mode,
        fixed_indicator_mode=resolved_fixed_indicator_mode,
    )


def build_runtime_prompt_route(
    *,
    prompt_raw: object,
    indicator_mode: object,
    indicator_source: object,
    consumer_mode: str = RECAP_RELABEL_CONSUMER_MODE,
    fixed_indicator_mode: str | None = None,
) -> PromptRouteSpec:
    normalized_consumer_mode = normalize_consumer_mode(consumer_mode)
    resolved_fixed_indicator_mode = resolve_fixed_indicator_mode(
        consumer_mode=normalized_consumer_mode,
        fixed_indicator_mode=fixed_indicator_mode,
    )
    return _build_runtime_prompt_route(
        prompt_raw=prompt_raw,
        indicator_mode=indicator_mode,
        indicator_source=indicator_source,
        consumer_mode=normalized_consumer_mode,
        fixed_indicator_mode=resolved_fixed_indicator_mode,
    )


def build_phase1_prompt_provenance(spec: PromptRouteSpec) -> dict[str, str]:
    return prompt_provenance_from_spec(spec)
