from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, cast

from work.recap.phase_thresholds import (
    DEFAULT_EPSILON_THRESHOLD_PHASE,
    FINE_TUNING_EPSILON_QUANTILE,
    PRETRAINING_EPSILON_QUANTILE,
    PhaseThresholdPolicy,
    build_phase_threshold_metadata,
    normalize_threshold_phase,
    resolve_epsilon_quantile,
    resolve_phase_threshold_policy,
)

DEFAULT_EPSILON_QUANTILE = PRETRAINING_EPSILON_QUANTILE
DEFAULT_THRESHOLD_GROUP_FIELD = "prompt_raw"


@dataclass(frozen=True)
class ThresholdEstimate:
    task_key: str
    epsilon_l: float
    sample_count: int
    quantile: float


def _coerce_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{context} must be float-like, got {value!r}")
    out = float(cast(Any, value))
    if not math.isfinite(out):
        raise ValueError(f"{context} must be finite, got {value!r}")
    return float(out)


def _quantile_linear(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError(f"q must be within [0,1], got {q!r}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = float(q) * float(len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - float(lo)
    return float((1.0 - weight) * ordered[lo] + weight * ordered[hi])


def build_epsilon_source(
    *,
    group_field: str = DEFAULT_THRESHOLD_GROUP_FIELD,
    quantile: float = DEFAULT_EPSILON_QUANTILE,
) -> str:
    quantile_text = format(float(quantile), ".4f").rstrip("0").rstrip(".")
    return f"per_task_quantile:{group_field}:q={quantile_text}"


def estimate_per_task_epsilons(
    records: Iterable[Mapping[str, object]],
    *,
    group_field: str = DEFAULT_THRESHOLD_GROUP_FIELD,
    advantage_field: str = "advantage_A",
    quantile: float = DEFAULT_EPSILON_QUANTILE,
) -> dict[str, ThresholdEstimate]:
    grouped_advantages: dict[str, list[float]] = defaultdict(list)
    for index, record in enumerate(records):
        task_key = str(record.get(group_field, "")).strip()
        if not task_key:
            raise ValueError(f"record#{index} missing non-empty {group_field}")
        grouped_advantages[task_key].append(
            _coerce_float(
                record.get(advantage_field),
                context=f"record#{index}.{advantage_field}",
            )
        )
    if not grouped_advantages:
        raise ValueError("cannot estimate per-task epsilons from empty record set")
    return {
        task_key: ThresholdEstimate(
            task_key=task_key,
            epsilon_l=_quantile_linear(values, float(quantile)),
            sample_count=len(values),
            quantile=float(quantile),
        )
        for task_key, values in sorted(grouped_advantages.items())
    }


__all__ = [
    "DEFAULT_EPSILON_QUANTILE",
    "DEFAULT_EPSILON_THRESHOLD_PHASE",
    "DEFAULT_THRESHOLD_GROUP_FIELD",
    "FINE_TUNING_EPSILON_QUANTILE",
    "PRETRAINING_EPSILON_QUANTILE",
    "PhaseThresholdPolicy",
    "ThresholdEstimate",
    "build_phase_threshold_metadata",
    "build_epsilon_source",
    "estimate_per_task_epsilons",
    "normalize_threshold_phase",
    "resolve_epsilon_quantile",
    "resolve_phase_threshold_policy",
]
