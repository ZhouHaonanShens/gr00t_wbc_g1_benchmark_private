from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


ADVANTAGE_CONTRACT_VERSION = "full_recap_continuous_adv_v2"
LEGACY_ADVANTAGE_CONTRACT_VERSION = "full_recap_continuous_adv_v1"
ADVANTAGE_RAW_COLUMN = "recap_m2.advantage_A"
ADVANTAGE_INPUT_COLUMN = "recap_m2.advantage_input"
ADVANTAGE_VALUE_COLUMN = "recap_m2.value_V"
ADVANTAGE_RETURN_COLUMN = "recap_m2.return_G"
ADVANTAGE_INPUT_CLIP_RANGE = 1.0
ADVANTAGE_SCALE_EPS = 1e-6
ADVANTAGE_SCALE_QUANTILE = 0.95
ADVANTAGE_SCALE_RULE = "sign_aware_quantile_by_sign_v1"
MAINLINE_TASK_TEXT_FIELD = "prompt_raw"
NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE = "continuous_advantage_diagnostic_lane"
VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE = "vlm_critic_diagnostic_lane"
CONTINUOUS_ADVANTAGE_CONTRACT_DIAGNOSTIC_ROUTE = (
    "continuous_advantage_contract_diagnostic"
)
NUMERIC_ADVANTAGE_EVAL_DIAGNOSTIC_ROUTE = "numeric_advantage_eval_diagnostic"
VLM_CRITIC_EVAL_SMOKE_DIAGNOSTIC_ROUTE = "vlm_critic_eval_smoke_diagnostic"
GENERIC_DIAGNOSTIC_COMPATIBILITY_FIELDS: tuple[str, ...] = (
    "success_rate",
    "success_count",
    "episodes",
    "episode_telemetry_jsonl",
    "step_telemetry_jsonl",
    "wrapper_status",
)


def build_diagnostic_surface_metadata(
    *,
    surface_route: str,
    authority_scope: str,
    compatibility_fields: Sequence[str] = (),
    surface_kind: str | None = None,
) -> dict[str, Any]:
    normalized_compatibility_fields = [
        str(field)
        for field in dict.fromkeys(str(field) for field in compatibility_fields)
        if str(field).strip()
    ]
    payload: dict[str, Any] = {
        "surface_route": str(surface_route),
        "diagnostic_only": True,
        "mainline_authority": False,
        "authority_scope": str(authority_scope),
        "authority_status": "diagnostic_only",
    }
    if surface_kind is not None:
        payload["surface_kind"] = str(surface_kind)
    if normalized_compatibility_fields:
        payload["compatibility_preserved_fields"] = normalized_compatibility_fields
    return payload


def diagnostic_surface_violations(
    payload: Mapping[str, Any],
    *,
    expected_route: str | None = None,
    expected_authority_scope: str | None = None,
    required_compatibility_fields: Sequence[str] = (),
) -> list[str]:
    violations: list[str] = []
    if payload.get("diagnostic_only") is not True:
        violations.append(
            f"diagnostic_only must be true, got {payload.get('diagnostic_only')!r}"
        )
    if payload.get("mainline_authority") is not False:
        violations.append(
            "mainline_authority must be false, "
            f"got {payload.get('mainline_authority')!r}"
        )
    if payload.get("authority_status") != "diagnostic_only":
        violations.append(
            "authority_status must be 'diagnostic_only', "
            f"got {payload.get('authority_status')!r}"
        )
    if expected_route is not None and payload.get("surface_route") != expected_route:
        violations.append(
            f"surface_route must be {expected_route!r}, got {payload.get('surface_route')!r}"
        )
    if (
        expected_authority_scope is not None
        and payload.get("authority_scope") != expected_authority_scope
    ):
        violations.append(
            "authority_scope must be "
            f"{expected_authority_scope!r}, got {payload.get('authority_scope')!r}"
        )
    compatibility_raw = payload.get("compatibility_preserved_fields")
    compatibility_fields: set[str] = (
        {str(field) for field in compatibility_raw}
        if isinstance(compatibility_raw, list)
        else set()
    )
    for field in required_compatibility_fields:
        if str(field) not in compatibility_fields:
            violations.append(
                f"compatibility_preserved_fields must include {str(field)!r}"
            )
    return violations


def _coerce_finite_float(raw: Any, *, context: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"Expected finite float-like value, got bool ({context})")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected float-like value for {context}, got {type(raw).__name__}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"Expected finite float for {context}, got {value!r}")
    return float(value)


def validate_advantage_input_value(value: Any, *, context: str = "advantage") -> float:
    out = _coerce_finite_float(value, context=context)
    lo = -1.0 * float(ADVANTAGE_INPUT_CLIP_RANGE)
    hi = float(ADVANTAGE_INPUT_CLIP_RANGE)
    if out < lo or out > hi:
        raise ValueError(f"{context} must be within [{lo}, {hi}], got {out!r}")
    return float(out)


def _linear_quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q!r}")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = float(q) * float(len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight_hi = pos - float(lo)
    weight_lo = 1.0 - weight_hi
    return float(weight_lo * ordered[lo] + weight_hi * ordered[hi])


def _coerce_positive_scale_or_none(raw: Any, *, context: str) -> float | None:
    if raw is None:
        return None
    value = _coerce_finite_float(raw, context=context)
    if value <= float(ADVANTAGE_SCALE_EPS):
        raise ValueError(f"{context} must be > {ADVANTAGE_SCALE_EPS}, got {value}")
    return float(value)


def compute_sign_aware_advantage_scales(
    advantage_values: list[Any],
    *,
    quantile: float = ADVANTAGE_SCALE_QUANTILE,
    context: str = "compute_sign_aware_advantage_scales",
) -> dict[str, Any]:
    if not advantage_values:
        raise ValueError("advantage values must be non-empty")
    if not (0.0 <= float(quantile) <= 1.0):
        raise ValueError(f"{context}.quantile must be in [0, 1], got {quantile!r}")
    clean = [
        _coerce_finite_float(raw, context=f"{context}[{idx}]")
        for idx, raw in enumerate(advantage_values)
    ]
    positive_values = [float(v) for v in clean if float(v) > 0.0]
    negative_values = [float(v) for v in clean if float(v) < 0.0]
    negative_abs_values = [abs(float(v)) for v in negative_values]
    zero_count = int(len(clean) - len(positive_values) - len(negative_values))
    positive_scale = (
        _linear_quantile(positive_values, float(quantile)) if positive_values else None
    )
    negative_scale_abs = (
        _linear_quantile(negative_abs_values, float(quantile))
        if negative_abs_values
        else None
    )
    positive_scale = _coerce_positive_scale_or_none(
        positive_scale,
        context=f"{context}.positive_scale",
    )
    negative_scale_abs = _coerce_positive_scale_or_none(
        negative_scale_abs,
        context=f"{context}.negative_scale_abs",
    )
    total = float(len(clean))
    return {
        "scale_rule": str(ADVANTAGE_SCALE_RULE),
        "quantile": float(quantile),
        "positive_quantile": float(quantile),
        "negative_quantile": float(quantile),
        "positive_scale": float(positive_scale) if positive_scale is not None else None,
        "negative_scale_abs": float(negative_scale_abs)
        if negative_scale_abs is not None
        else None,
        "positive_quantile_value": float(positive_scale)
        if positive_scale is not None
        else None,
        "negative_quantile_abs_value": float(negative_scale_abs)
        if negative_scale_abs is not None
        else None,
        "positive_count": int(len(positive_values)),
        "negative_count": int(len(negative_values)),
        "zero_count": int(zero_count),
        "positive_fraction": float(len(positive_values)) / total,
        "negative_fraction": float(len(negative_values)) / total,
        "zero_fraction": float(zero_count) / total,
        "positive_min": float(min(positive_values)) if positive_values else None,
        "positive_max": float(max(positive_values)) if positive_values else None,
        "negative_min": float(min(negative_values)) if negative_values else None,
        "negative_max": float(max(negative_values)) if negative_values else None,
    }


def scale_advantage_input(
    advantage_A: Any,
    *,
    p95_abs_advantage: Any | None = None,
    positive_scale: Any | None = None,
    negative_scale_abs: Any | None = None,
    context: str = "scale_advantage_input",
) -> float:
    raw = _coerce_finite_float(advantage_A, context=f"{context}.advantage_A")
    shared_scale = _coerce_positive_scale_or_none(
        p95_abs_advantage,
        context=f"{context}.p95_abs_advantage",
    )
    positive_sign_scale = _coerce_positive_scale_or_none(
        positive_scale,
        context=f"{context}.positive_scale",
    )
    negative_sign_scale = _coerce_positive_scale_or_none(
        negative_scale_abs,
        context=f"{context}.negative_scale_abs",
    )
    if raw > 0.0:
        scale = positive_sign_scale if positive_sign_scale is not None else shared_scale
        if scale is None:
            raise ValueError(
                f"{context} requires positive_scale for positive advantage {raw}"
            )
    elif raw < 0.0:
        scale = negative_sign_scale if negative_sign_scale is not None else shared_scale
        if scale is None:
            raise ValueError(
                f"{context} requires negative_scale_abs for negative advantage {raw}"
            )
    else:
        return 0.0
    scaled = raw / scale
    clipped = max(
        -1.0 * float(ADVANTAGE_INPUT_CLIP_RANGE),
        min(float(ADVANTAGE_INPUT_CLIP_RANGE), scaled),
    )
    return float(clipped)


def normalize_advantage_to_input(
    advantage_A: Any,
    *,
    p95_abs_advantage: Any | None = None,
    positive_scale: Any | None = None,
    negative_scale_abs: Any | None = None,
    clip_range: float = ADVANTAGE_INPUT_CLIP_RANGE,
) -> float:
    if float(clip_range) != float(ADVANTAGE_INPUT_CLIP_RANGE):
        raise ValueError(
            "Mainline clip_range is fixed: "
            f"expected {ADVANTAGE_INPUT_CLIP_RANGE} got {clip_range}"
        )
    return float(
        scale_advantage_input(
            advantage_A,
            p95_abs_advantage=p95_abs_advantage,
            positive_scale=positive_scale,
            negative_scale_abs=negative_scale_abs,
            context="normalize_advantage_to_input",
        )
    )


def build_advantage_contract_metadata(
    *,
    source_iter_tag: str,
    n_samples: int,
    p95_abs_advantage: float | None = None,
    positive_scale: float | None = None,
    negative_scale_abs: float | None = None,
    clip_range: float = ADVANTAGE_INPUT_CLIP_RANGE,
    critic_dir: str | None,
    critic_include_t: bool,
    advantage_stats: dict[str, Any] | None = None,
    raw_summary: dict[str, Any] | None = None,
    scaled_summary: dict[str, Any] | None = None,
    sign_scale_summary: dict[str, Any] | None = None,
    scale_rule: str = ADVANTAGE_SCALE_RULE,
) -> dict[str, object]:
    if float(clip_range) != float(ADVANTAGE_INPUT_CLIP_RANGE):
        raise ValueError(
            "Mainline clip_range is fixed: "
            f"expected {ADVANTAGE_INPUT_CLIP_RANGE} got {clip_range}"
        )
    stats_dict = dict(advantage_stats or {})
    if raw_summary is None:
        raw_summary = {
            k.removeprefix("raw_"): float(v)
            for k, v in stats_dict.items()
            if k.startswith("raw_")
        }
    if scaled_summary is None:
        scaled_summary = {
            k.removeprefix("scaled_"): float(v)
            for k, v in stats_dict.items()
            if k.startswith("scaled_")
        }
    if not raw_summary:
        raw_summary = {
            k: float(v)
            for k, v in stats_dict.items()
            if k not in {"value_source"} and not k.startswith("scaled_")
        }
    if sign_scale_summary is None:
        raw_sign_summary = stats_dict.get("sign_scale_summary")
        if isinstance(raw_sign_summary, dict):
            sign_scale_summary = dict(raw_sign_summary)
    value_source = str(stats_dict.get("value_source") or "critic")
    positive_scale_value = (
        _coerce_positive_scale_or_none(
            positive_scale,
            context="build_advantage_contract_metadata.positive_scale",
        )
        if positive_scale is not None
        else None
    )
    negative_scale_value = (
        _coerce_positive_scale_or_none(
            negative_scale_abs,
            context="build_advantage_contract_metadata.negative_scale_abs",
        )
        if negative_scale_abs is not None
        else None
    )
    if positive_scale_value is None and p95_abs_advantage is not None:
        positive_scale_value = _coerce_positive_scale_or_none(
            p95_abs_advantage,
            context="build_advantage_contract_metadata.p95_abs_advantage_positive",
        )
    if negative_scale_value is None and p95_abs_advantage is not None:
        negative_scale_value = _coerce_positive_scale_or_none(
            p95_abs_advantage,
            context="build_advantage_contract_metadata.p95_abs_advantage_negative",
        )
    contract: dict[str, Any] = {
        "contract_version": str(ADVANTAGE_CONTRACT_VERSION),
        "raw_columns": [
            str(ADVANTAGE_RETURN_COLUMN),
            str(ADVANTAGE_VALUE_COLUMN),
            str(ADVANTAGE_RAW_COLUMN),
        ],
        "model_advantage_column": str(ADVANTAGE_INPUT_COLUMN),
        "task_text_field": str(MAINLINE_TASK_TEXT_FIELD),
        "value_source": str(value_source),
        "critic_dir": str(critic_dir) if critic_dir else None,
        "critic_include_t": bool(critic_include_t),
        "scale_rule": str(scale_rule),
        "positive_scale_quantile": float(ADVANTAGE_SCALE_QUANTILE),
        "negative_scale_quantile": float(ADVANTAGE_SCALE_QUANTILE),
        "positive_scale": float(positive_scale_value)
        if positive_scale_value is not None
        else None,
        "negative_scale_abs": float(negative_scale_value)
        if negative_scale_value is not None
        else None,
        "normalization_formulas": {
            "positive": "clip(advantage_A / positive_scale, -1.0, 1.0)",
            "negative": "clip(advantage_A / negative_scale_abs, -1.0, 1.0)",
            "zero": 0.0,
        },
        "clip_min": float(-1.0 * ADVANTAGE_INPUT_CLIP_RANGE),
        "clip_max": float(ADVANTAGE_INPUT_CLIP_RANGE),
        "raw_summary": raw_summary,
        "scaled_summary": scaled_summary or None,
        "sign_scale_summary": sign_scale_summary or None,
        "source_iter_tag": str(source_iter_tag),
        "n_samples": int(n_samples),
        "failure_policy": "hard_error_on_missing_nan_inf_or_out_of_range",
        "legacy_stats_snapshot": stats_dict or None,
    }
    contract.update(
        build_diagnostic_surface_metadata(
            surface_route=CONTINUOUS_ADVANTAGE_CONTRACT_DIAGNOSTIC_ROUTE,
            authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
            surface_kind="continuous_advantage_contract",
        )
    )
    return contract


def extract_advantage_contract(info_obj: dict[str, Any]) -> dict[str, Any]:
    raw = info_obj.get("recap_advantage_input_contract")
    if not isinstance(raw, dict):
        raise KeyError("Missing info.json key 'recap_advantage_input_contract'")
    version = raw.get("contract_version") or raw.get("version")
    if version not in {
        ADVANTAGE_CONTRACT_VERSION,
        LEGACY_ADVANTAGE_CONTRACT_VERSION,
    }:
        raise ValueError(
            "Unsupported advantage contract version: "
            f"expected one of {[ADVANTAGE_CONTRACT_VERSION, LEGACY_ADVANTAGE_CONTRACT_VERSION]!r} got {version!r}"
        )
    model_facing_column = raw.get("model_advantage_column") or raw.get(
        "model_facing_column"
    )
    if model_facing_column != ADVANTAGE_INPUT_COLUMN:
        raise ValueError(
            "Unexpected model-facing advantage column: "
            f"expected {ADVANTAGE_INPUT_COLUMN!r} got {model_facing_column!r}"
        )
    return dict(raw)


__all__ = [
    "ADVANTAGE_CONTRACT_VERSION",
    "ADVANTAGE_INPUT_CLIP_RANGE",
    "ADVANTAGE_INPUT_COLUMN",
    "ADVANTAGE_RAW_COLUMN",
    "ADVANTAGE_RETURN_COLUMN",
    "ADVANTAGE_SCALE_EPS",
    "ADVANTAGE_SCALE_QUANTILE",
    "ADVANTAGE_SCALE_RULE",
    "ADVANTAGE_VALUE_COLUMN",
    "CONTINUOUS_ADVANTAGE_CONTRACT_DIAGNOSTIC_ROUTE",
    "GENERIC_DIAGNOSTIC_COMPATIBILITY_FIELDS",
    "LEGACY_ADVANTAGE_CONTRACT_VERSION",
    "MAINLINE_TASK_TEXT_FIELD",
    "NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE",
    "NUMERIC_ADVANTAGE_EVAL_DIAGNOSTIC_ROUTE",
    "VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE",
    "VLM_CRITIC_EVAL_SMOKE_DIAGNOSTIC_ROUTE",
    "build_advantage_contract_metadata",
    "build_diagnostic_surface_metadata",
    "compute_sign_aware_advantage_scales",
    "diagnostic_surface_violations",
    "extract_advantage_contract",
    "normalize_advantage_to_input",
    "scale_advantage_input",
    "validate_advantage_input_value",
]
