from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from work.openpi.recap.indicator_modes import (
    DEFAULT_INDICATOR_DROPOUT_P,
    indicator_mode_train_from_consumer_mode,
    resolve_emitted_indicator_mode,
    should_apply_indicator_dropout,
)
from work.openpi.recap.thresholds import build_epsilon_source
from work.recap import text_indicator


PHASE1_PROMPT_ROUTE = "recap_conditioned_prompt_token_v1"
CONDITIONING_MODE = "prompt_text_only"
PROMPT_TEXT_SURFACE_CANONICAL = "canonical_text_indicator"
PROMPT_TEXT_SURFACE_PROMPT_RAW_ONLY = "prompt_raw_only"


@dataclass(frozen=True)
class PromptRouteSpec:
    prompt_text: str
    prompt_route: str
    conditioning_mode: str
    indicator_mode: str
    source_prompt_field: str
    consumer_mode: str
    indicator_mode_train: str
    fixed_indicator_mode: str | None
    indicator_source: str
    prompt_text_surface: str
    per_sample_indicator_consumption: bool
    indicator_dropout_p: float
    indicator_dropout_applied: bool
    epsilon_source: str
    human_correction_override: bool
    prompt_conditioned_dependency: bool
    advantage_input_dependency: bool
    authoritative_carrier_text: str | None
    authoritative_carrier_source: str
    authoritative_carrier_matches_prompt_text: bool


def build_phase1_prompt_text(prompt_raw: object, indicator_value: object) -> str:
    indicator_mode = text_indicator.indicator_mode_from_indicator_value(
        indicator_value,
        field_name="recap_m2.indicator_I",
    )
    return text_indicator.build_authoritative_carrier_text_v1(
        prompt_raw, indicator_mode
    )


def prompt_text_surface_for_indicator_mode(indicator_mode: object) -> str:
    mode = text_indicator.normalize_indicator_mode(
        indicator_mode,
        field_name="indicator_mode",
    )
    if mode == text_indicator.TEXT_INDICATOR_OMIT:
        return PROMPT_TEXT_SURFACE_PROMPT_RAW_ONLY
    return PROMPT_TEXT_SURFACE_CANONICAL


def _require_prompt_raw(label_row: Mapping[str, object]) -> str:
    return text_indicator.require_prompt_raw(
        label_row.get("prompt_raw"),
        field_name="prompt_raw",
    )


def _resolve_epsilon_source(label_row: Mapping[str, object]) -> str:
    for key in ("epsilon_source", "recap_m2.epsilon_source"):
        value = str(label_row.get(key, "")).strip()
        if value:
            return value
    return build_epsilon_source()


def _validate_authoritative_prompt_conditioned(
    *,
    label_row: Mapping[str, object],
    prompt_text: str,
) -> None:
    available_keys = set(label_row.keys())
    prompt_conditioned = (
        label_row["prompt_conditioned"]
        if "prompt_conditioned" in available_keys
        else None
    )
    if isinstance(prompt_conditioned, str) and prompt_conditioned.strip():
        if prompt_conditioned != prompt_text:
            raise ValueError(
                "Phase 1 prompt route forbids non-canonical prompt_conditioned text; "
                + "authoritative carrier must be prompt_raw + canonical newline indicator"
            )
    dual_task_text = (
        label_row["dual_task_text"] if "dual_task_text" in available_keys else False
    )
    if bool(dual_task_text):
        raise ValueError(
            "Phase 1 prompt route forbids dual_task_text because it creates multiple prompt semantics"
        )
    advantage_input = (
        label_row["advantage_input"] if "advantage_input" in available_keys else None
    )
    if advantage_input is not None:
        raise ValueError(
            "Phase 1 prompt route forbids numeric advantage passthrough in the same sample"
        )


def _resolve_mainline_authoritative_carrier_text(
    *,
    label_row: Mapping[str, object],
    prompt_raw: str,
    indicator_mode: str,
    indicator_source: str,
) -> tuple[str | None, str]:
    if indicator_source != "recap_m2.indicator_I":
        return None, "not_applicable_non_mainline_indicator_source"
    carrier_text_v1 = label_row.get(text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD)
    if carrier_text_v1 is None or not str(carrier_text_v1).strip():
        return (
            text_indicator.build_authoritative_carrier_text_v1(
                prompt_raw, indicator_mode
            ),
            "prompt_raw+indicator_I",
        )
    return (
        text_indicator.require_authoritative_carrier_text_v1(
            carrier_text_v1,
            prompt_raw=prompt_raw,
            indicator_mode=indicator_mode,
        ),
        text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD,
    )


def build_training_prompt_route(
    label_row: Mapping[str, object],
    *,
    consumer_mode: str,
    fixed_indicator_mode: str | None,
    indicator_dropout_p: float = DEFAULT_INDICATOR_DROPOUT_P,
) -> PromptRouteSpec:
    prompt_raw = _require_prompt_raw(label_row)
    indicator_mode_train = indicator_mode_train_from_consumer_mode(consumer_mode)
    indicator_mode, indicator_source, per_sample_consumption = (
        resolve_emitted_indicator_mode(
            label_row=label_row,
            indicator_mode_train=indicator_mode_train,
        )
    )
    authoritative_carrier_text, authoritative_carrier_source = (
        _resolve_mainline_authoritative_carrier_text(
            label_row=label_row,
            prompt_raw=prompt_raw,
            indicator_mode=indicator_mode,
            indicator_source=indicator_source,
        )
    )
    base_prompt_text = (
        authoritative_carrier_text
        or text_indicator.build_authoritative_carrier_text_v1(
            prompt_raw,
            indicator_mode,
        )
    )
    _validate_authoritative_prompt_conditioned(
        label_row=label_row,
        prompt_text=base_prompt_text,
    )
    dropout_applied = should_apply_indicator_dropout(
        label_row,
        indicator_mode_train=indicator_mode_train,
        dropout_p=indicator_dropout_p,
    )
    if dropout_applied:
        indicator_mode = text_indicator.TEXT_INDICATOR_OMIT
    prompt_text = text_indicator.build_authoritative_carrier_text_v1(
        prompt_raw, indicator_mode
    )
    return PromptRouteSpec(
        prompt_text=prompt_text,
        prompt_route=PHASE1_PROMPT_ROUTE,
        conditioning_mode=CONDITIONING_MODE,
        indicator_mode=indicator_mode,
        source_prompt_field="prompt_raw",
        consumer_mode=str(consumer_mode),
        indicator_mode_train=indicator_mode_train,
        fixed_indicator_mode=fixed_indicator_mode,
        indicator_source=indicator_source,
        prompt_text_surface=prompt_text_surface_for_indicator_mode(indicator_mode),
        per_sample_indicator_consumption=bool(per_sample_consumption),
        indicator_dropout_p=float(indicator_dropout_p),
        indicator_dropout_applied=bool(dropout_applied),
        epsilon_source=_resolve_epsilon_source(label_row),
        human_correction_override=True,
        prompt_conditioned_dependency=False,
        advantage_input_dependency=False,
        authoritative_carrier_text=authoritative_carrier_text,
        authoritative_carrier_source=authoritative_carrier_source,
        authoritative_carrier_matches_prompt_text=(
            authoritative_carrier_text == prompt_text
            if authoritative_carrier_text is not None
            else False
        ),
    )


def build_runtime_prompt_route(
    *,
    prompt_raw: object,
    indicator_mode: object,
    indicator_source: object,
    consumer_mode: str,
    fixed_indicator_mode: str | None,
) -> PromptRouteSpec:
    normalized_indicator_mode = text_indicator.normalize_indicator_mode(
        indicator_mode,
        field_name="indicator_mode",
    )
    authoritative_carrier_text = text_indicator.build_authoritative_carrier_text_v1(
        prompt_raw,
        normalized_indicator_mode,
    )
    return PromptRouteSpec(
        prompt_text=authoritative_carrier_text,
        prompt_route=PHASE1_PROMPT_ROUTE,
        conditioning_mode=CONDITIONING_MODE,
        indicator_mode=normalized_indicator_mode,
        source_prompt_field="prompt_raw",
        consumer_mode=str(consumer_mode),
        indicator_mode_train=indicator_mode_train_from_consumer_mode(consumer_mode),
        fixed_indicator_mode=fixed_indicator_mode,
        indicator_source=str(indicator_source).strip() or "runtime_indicator_mode",
        prompt_text_surface=prompt_text_surface_for_indicator_mode(
            normalized_indicator_mode
        ),
        per_sample_indicator_consumption=False,
        indicator_dropout_p=0.0,
        indicator_dropout_applied=False,
        epsilon_source="runtime_not_applicable",
        human_correction_override=True,
        prompt_conditioned_dependency=False,
        advantage_input_dependency=False,
        authoritative_carrier_text=authoritative_carrier_text,
        authoritative_carrier_source="prompt_raw+indicator_mode",
        authoritative_carrier_matches_prompt_text=True,
    )


def prompt_provenance_from_spec(spec: PromptRouteSpec) -> dict[str, str]:
    return {
        "prompt_route": spec.prompt_route,
        "conditioning_mode": spec.conditioning_mode,
        "indicator_mode": spec.indicator_mode,
        "source_prompt_field": spec.source_prompt_field,
        "consumer_mode": spec.consumer_mode,
        "indicator_mode_train": spec.indicator_mode_train,
        "fixed_indicator_mode": spec.fixed_indicator_mode or "",
        "indicator_source": spec.indicator_source,
        "prompt_text_surface": spec.prompt_text_surface,
        "per_sample_indicator_consumption": str(
            spec.per_sample_indicator_consumption
        ).lower(),
        "indicator_dropout_p": format(float(spec.indicator_dropout_p), ".1f"),
        "indicator_dropout_applied": str(spec.indicator_dropout_applied).lower(),
        "epsilon_source": spec.epsilon_source,
        "human_correction_override": str(spec.human_correction_override).lower(),
        "prompt_conditioned_dependency": str(
            spec.prompt_conditioned_dependency
        ).lower(),
        "advantage_input_dependency": str(spec.advantage_input_dependency).lower(),
        "authoritative_carrier_field": text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD,
        "authoritative_carrier_schema_version": text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION,
        "authoritative_carrier_source": spec.authoritative_carrier_source,
        "authoritative_carrier_matches_prompt_text": str(
            spec.authoritative_carrier_matches_prompt_text
        ).lower(),
    }


__all__ = [
    "CONDITIONING_MODE",
    "PHASE1_PROMPT_ROUTE",
    "PROMPT_TEXT_SURFACE_CANONICAL",
    "PROMPT_TEXT_SURFACE_PROMPT_RAW_ONLY",
    "PromptRouteSpec",
    "build_phase1_prompt_text",
    "build_runtime_prompt_route",
    "build_training_prompt_route",
    "prompt_provenance_from_spec",
    "prompt_text_surface_for_indicator_mode",
]
