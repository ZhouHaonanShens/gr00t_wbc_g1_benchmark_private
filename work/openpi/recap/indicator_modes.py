from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from work.recap import text_indicator


INDICATOR_MODE_TRAIN_INFORMATIVE = "informative"
INDICATOR_MODE_TRAIN_FIXED_POSITIVE = "fixed_positive"
INDICATOR_MODE_TRAIN_FIXED_NEGATIVE = "fixed_negative"
INDICATOR_MODE_TRAIN_OMIT = "omit"
INDICATOR_MODE_TRAIN_SHUFFLED = "shuffled"
DEFAULT_INDICATOR_DROPOUT_P = 0.3

CANONICAL_INDICATOR_MODE_TRAIN_VALUES: tuple[str, ...] = (
    INDICATOR_MODE_TRAIN_INFORMATIVE,
    INDICATOR_MODE_TRAIN_FIXED_POSITIVE,
    INDICATOR_MODE_TRAIN_FIXED_NEGATIVE,
    INDICATOR_MODE_TRAIN_OMIT,
    INDICATOR_MODE_TRAIN_SHUFFLED,
)

LEGACY_CONSUMER_MODE_ALIASES: dict[str, str] = {
    "informative_adv": INDICATOR_MODE_TRAIN_INFORMATIVE,
    "recap_relabel": INDICATOR_MODE_TRAIN_INFORMATIVE,
    "fixedadv_constant": INDICATOR_MODE_TRAIN_OMIT,
    "shuffled_adv_diag": INDICATOR_MODE_TRAIN_SHUFFLED,
}


def normalize_indicator_mode_train(
    raw: object,
    *,
    field_name: str = "indicator_mode_train",
) -> str:
    if raw is None:
        return INDICATOR_MODE_TRAIN_INFORMATIVE
    if not isinstance(raw, str):
        raise TypeError(f"{field_name} must be a string; got {type(raw).__name__}")
    mode = raw.strip().lower()
    canonical = LEGACY_CONSUMER_MODE_ALIASES.get(mode, mode)
    if canonical not in CANONICAL_INDICATOR_MODE_TRAIN_VALUES:
        raise ValueError(
            f"unknown {field_name} {raw!r}; expected {CANONICAL_INDICATOR_MODE_TRAIN_VALUES!r}"
        )
    return canonical


def indicator_mode_train_from_consumer_mode(raw: object) -> str:
    return normalize_indicator_mode_train(raw, field_name="consumer_mode")


def _json_hash_ready(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_hash_ready(tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _json_hash_ready(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_hash_ready(item) for item in value]
    return repr(value)


def _deterministic_probability(label_row: Mapping[str, object], *, salt: str) -> float:
    sample_key: dict[str, object] = {"salt": salt}
    for source_key in (
        "prompt_raw",
        "episode_index",
        "recap_m2.t",
        "step_index",
        "observation.state",
        "observation/state",
        "action",
        "actions",
    ):
        if source_key in label_row:
            sample_key[source_key] = _json_hash_ready(label_row[source_key])
    seed_material = json.dumps(
        sample_key,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return float(value) / float(2**64)


def build_shuffled_indicator_mode(label_row: Mapping[str, object]) -> str:
    probability = _deterministic_probability(label_row, salt="indicator_shuffle_v1")
    if probability < 0.5:
        return text_indicator.TEXT_INDICATOR_NEGATIVE
    return text_indicator.TEXT_INDICATOR_POSITIVE


def should_apply_indicator_dropout(
    label_row: Mapping[str, object],
    *,
    indicator_mode_train: str,
    dropout_p: float = DEFAULT_INDICATOR_DROPOUT_P,
) -> bool:
    mode = normalize_indicator_mode_train(
        indicator_mode_train,
        field_name="indicator_mode_train",
    )
    probability = float(dropout_p)
    if probability <= 0.0:
        return False
    if probability > 1.0:
        raise ValueError(f"dropout_p must be within [0, 1], got {dropout_p!r}")
    if mode not in {
        INDICATOR_MODE_TRAIN_INFORMATIVE,
        INDICATOR_MODE_TRAIN_SHUFFLED,
    }:
        return False
    if "episode_index" not in label_row and "recap_m2.t" not in label_row:
        return False
    return (
        _deterministic_probability(label_row, salt="indicator_dropout_v1") < probability
    )


def resolve_emitted_indicator_mode(
    *,
    label_row: Mapping[str, object],
    indicator_mode_train: str,
) -> tuple[str, str, bool]:
    mode = normalize_indicator_mode_train(
        indicator_mode_train,
        field_name="indicator_mode_train",
    )
    if mode == INDICATOR_MODE_TRAIN_FIXED_POSITIVE:
        return text_indicator.TEXT_INDICATOR_POSITIVE, "fixed_positive", False
    if mode == INDICATOR_MODE_TRAIN_FIXED_NEGATIVE:
        return text_indicator.TEXT_INDICATOR_NEGATIVE, "fixed_negative", False
    if mode == INDICATOR_MODE_TRAIN_OMIT:
        return text_indicator.TEXT_INDICATOR_OMIT, "fixed_indicator_mode", False
    if mode == INDICATOR_MODE_TRAIN_SHUFFLED:
        return (
            build_shuffled_indicator_mode(label_row),
            "deterministic_shuffled_sample_key",
            True,
        )
    indicator_value = label_row.get("recap_m2.indicator_I")
    return (
        text_indicator.indicator_mode_from_indicator_value(
            indicator_value,
            field_name="recap_m2.indicator_I",
        ),
        "recap_m2.indicator_I",
        True,
    )


__all__ = [
    "CANONICAL_INDICATOR_MODE_TRAIN_VALUES",
    "DEFAULT_INDICATOR_DROPOUT_P",
    "INDICATOR_MODE_TRAIN_FIXED_NEGATIVE",
    "INDICATOR_MODE_TRAIN_FIXED_POSITIVE",
    "INDICATOR_MODE_TRAIN_INFORMATIVE",
    "INDICATOR_MODE_TRAIN_OMIT",
    "INDICATOR_MODE_TRAIN_SHUFFLED",
    "LEGACY_CONSUMER_MODE_ALIASES",
    "build_shuffled_indicator_mode",
    "indicator_mode_train_from_consumer_mode",
    "normalize_indicator_mode_train",
    "resolve_emitted_indicator_mode",
    "should_apply_indicator_dropout",
]
