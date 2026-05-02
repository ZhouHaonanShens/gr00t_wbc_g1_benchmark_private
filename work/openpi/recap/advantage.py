from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

from work.openpi.recap.indicator_modes import (
    CANONICAL_INDICATOR_MODE_TRAIN_VALUES,
    DEFAULT_INDICATOR_DROPOUT_P,
    INDICATOR_MODE_TRAIN_INFORMATIVE,
)
from work.openpi.recap.thresholds import (
    DEFAULT_EPSILON_THRESHOLD_PHASE,
    DEFAULT_THRESHOLD_GROUP_FIELD,
    ThresholdEstimate,
    build_phase_threshold_metadata,
    build_epsilon_source,
    estimate_per_task_epsilons,
    resolve_epsilon_quantile,
    resolve_phase_threshold_policy,
)
from work.recap.advantage import (
    build_advantage_contract_metadata,
    compute_sign_aware_advantage_scales,
    normalize_advantage_to_input,
    validate_advantage_input_value,
)
from work.recap import text_indicator


@dataclass(frozen=True)
class AdvantageRecord:
    episode_index: int
    step_index: int
    task_key: str
    prompt_raw: str
    return_G: float
    value_V: float
    advantage_A: float
    advantage_input: float
    epsilon_l: float
    indicator_I: int
    is_correction: bool
    human_correction_override_applied: bool
    prompt_conditioned: str


def _coerce_episode_map(
    payload: Mapping[int, Sequence[float]] | Mapping[int, Sequence[bool]],
    *,
    context: str,
) -> dict[int, Sequence[float] | Sequence[bool]]:
    out: dict[int, Sequence[float] | Sequence[bool]] = {}
    for key, value in payload.items():
        out[int(key)] = value
    if not out:
        raise ValueError(f"{context} must be non-empty")
    return out


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("summary requires non-empty values")
    ordered = sorted(float(value) for value in values)
    count = float(len(ordered))
    return {
        "count": count,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / count),
        "p50": float(ordered[len(ordered) // 2]),
        "p95": float(
            ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))]
        ),
        "zero_ratio": float(
            sum(1 for value in ordered if abs(value) <= 1e-12) / len(ordered)
        ),
    }


def _record_float(record: Mapping[str, object], key: str) -> float:
    return float(cast(Any, record[key]))


def _record_int(record: Mapping[str, object], key: str) -> int:
    return int(cast(Any, record[key]))


def emit_binary_indicator(
    *,
    advantage_A: float,
    epsilon_l: float,
    is_correction: bool,
) -> tuple[int, bool]:
    indicator = 1 if float(advantage_A) > float(epsilon_l) else 0
    override_applied = False
    if bool(is_correction):
        indicator = 1
        override_applied = True
    return int(indicator), bool(override_applied)


def build_advantage_generation_plan(
    *,
    prompts_by_episode: Mapping[int, str],
    returns_by_episode: Mapping[int, Sequence[float]],
    values_by_episode: Mapping[int, Sequence[float]],
    critic_metadata: Mapping[str, object],
    corrections_by_episode: Mapping[int, Sequence[bool]] | None = None,
    epsilon_quantile: float | None = None,
    threshold_phase: str = DEFAULT_EPSILON_THRESHOLD_PHASE,
    threshold_group_field: str = DEFAULT_THRESHOLD_GROUP_FIELD,
    indicator_mode_train: str = INDICATOR_MODE_TRAIN_INFORMATIVE,
    indicator_dropout_p: float = DEFAULT_INDICATOR_DROPOUT_P,
) -> dict[str, object]:
    value_source = str(critic_metadata.get("value_source", "")).strip()
    if value_source != "critic":
        raise ValueError(
            "task-6 advantage generation requires critic-derived values; "
            + f"got value_source={value_source or 'missing'!r}"
        )

    prompt_map = {int(key): str(value) for key, value in prompts_by_episode.items()}
    returns_map = _coerce_episode_map(returns_by_episode, context="returns_by_episode")
    values_map = _coerce_episode_map(values_by_episode, context="values_by_episode")
    corrections_map = (
        {int(key): value for key, value in corrections_by_episode.items()}
        if corrections_by_episode is not None
        else {}
    )

    unthresholded_records: list[dict[str, object]] = []
    raw_advantages: list[float] = []
    for episode_index, returns in returns_map.items():
        if episode_index not in values_map:
            raise ValueError(f"episode_index={episode_index} missing critic values")
        if episode_index not in prompt_map:
            raise ValueError(f"episode_index={episode_index} missing prompt_raw")
        values = values_map[episode_index]
        corrections = corrections_map.get(episode_index)
        if len(returns) != len(values):
            raise ValueError(
                f"episode_index={episode_index} return/value length mismatch: {len(returns)} vs {len(values)}"
            )
        if corrections is not None and len(corrections) != len(returns):
            raise ValueError(
                f"episode_index={episode_index} correction length mismatch: {len(corrections)} vs {len(returns)}"
            )
        prompt_raw = prompt_map[episode_index]
        for step_index, (return_g, value_v) in enumerate(
            zip(returns, values, strict=True)
        ):
            advantage_A = float(float(return_g) - float(value_v))
            raw_advantages.append(float(advantage_A))
            unthresholded_records.append(
                {
                    "episode_index": int(episode_index),
                    "step_index": int(step_index),
                    threshold_group_field: prompt_raw,
                    "prompt_raw": prompt_raw,
                    "return_G": float(return_g),
                    "value_V": float(value_v),
                    "advantage_A": float(advantage_A),
                    "is_correction": bool(corrections[step_index])
                    if corrections is not None
                    else False,
                }
            )

    scale_metadata = compute_sign_aware_advantage_scales(raw_advantages)
    positive_scale = scale_metadata.get("positive_scale")
    negative_scale_abs = scale_metadata.get("negative_scale_abs")
    if positive_scale is None or negative_scale_abs is None:
        raise ValueError(
            "task-6 advantage generation requires both positive and negative scales; "
            + f"observed={scale_metadata!r}"
        )

    threshold_policy = resolve_phase_threshold_policy(threshold_phase)
    effective_epsilon_quantile = resolve_epsilon_quantile(
        threshold_phase=threshold_policy.phase,
        epsilon_quantile=epsilon_quantile,
    )
    threshold_estimates = estimate_per_task_epsilons(
        unthresholded_records,
        group_field=threshold_group_field,
        advantage_field="advantage_A",
        quantile=effective_epsilon_quantile,
    )
    epsilon_source = build_epsilon_source(
        group_field=threshold_group_field,
        quantile=effective_epsilon_quantile,
    )

    records_by_episode: dict[int, list[dict[str, object]]] = {}
    scaled_advantages: list[float] = []
    for record in unthresholded_records:
        prompt_raw = str(record["prompt_raw"])
        threshold_key = str(record[threshold_group_field])
        epsilon_estimate = threshold_estimates[threshold_key]
        advantage_input = validate_advantage_input_value(
            normalize_advantage_to_input(
                _record_float(record, "advantage_A"),
                positive_scale=positive_scale,
                negative_scale_abs=negative_scale_abs,
            ),
            context=(
                f"episode_index={record['episode_index']}.step_index={record['step_index']}.advantage_input"
            ),
        )
        indicator_I, override_applied = emit_binary_indicator(
            advantage_A=_record_float(record, "advantage_A"),
            epsilon_l=float(epsilon_estimate.epsilon_l),
            is_correction=bool(record["is_correction"]),
        )
        prompt_conditioned = text_indicator.build_canonical_text_indicator(
            prompt_raw,
            text_indicator.indicator_mode_from_indicator_value(indicator_I),
        )
        result = asdict(
            AdvantageRecord(
                episode_index=_record_int(record, "episode_index"),
                step_index=_record_int(record, "step_index"),
                task_key=prompt_raw,
                prompt_raw=prompt_raw,
                return_G=_record_float(record, "return_G"),
                value_V=_record_float(record, "value_V"),
                advantage_A=_record_float(record, "advantage_A"),
                advantage_input=float(advantage_input),
                epsilon_l=float(epsilon_estimate.epsilon_l),
                indicator_I=int(indicator_I),
                is_correction=bool(record["is_correction"]),
                human_correction_override_applied=bool(override_applied),
                prompt_conditioned=prompt_conditioned,
            )
        )
        records_by_episode.setdefault(_record_int(record, "episode_index"), []).append(
            result
        )
        scaled_advantages.append(float(advantage_input))

    raw_summary = _summary(raw_advantages)
    scaled_summary = _summary(scaled_advantages)
    threshold_summary = {
        task_key: {
            "epsilon_l": float(estimate.epsilon_l),
            "sample_count": int(estimate.sample_count),
            "quantile": float(estimate.quantile),
        }
        for task_key, estimate in threshold_estimates.items()
    }
    advantage_contract = build_advantage_contract_metadata(
        source_iter_tag="official_native_8d_recap_relabels_v1",
        n_samples=len(raw_advantages),
        positive_scale=float(positive_scale),
        negative_scale_abs=float(negative_scale_abs),
        critic_dir=str(critic_metadata.get("critic_dir") or ""),
        critic_include_t=False,
        advantage_stats={"value_source": "critic"},
        raw_summary=raw_summary,
        scaled_summary=scaled_summary,
        sign_scale_summary=scale_metadata,
    )
    advantage_contract.update(
        {str(key): value for key, value in critic_metadata.items()}
    )
    advantage_contract.update(
        {
            "epsilon_source": epsilon_source,
            "epsilon_group_field": threshold_group_field,
            "epsilon_quantile": float(effective_epsilon_quantile),
            "epsilon_threshold_phase": str(threshold_policy.phase),
            "epsilon_target_positive_fraction": float(
                threshold_policy.target_positive_fraction
            ),
            "epsilon_threshold_policy": build_phase_threshold_metadata(
                threshold_phase=threshold_policy.phase,
                epsilon_quantile=epsilon_quantile,
            ),
            "epsilon_by_task": threshold_summary,
            "indicator_formula": "I = 1[A > epsilon_l(task)]",
            "indicator_mode_train": str(indicator_mode_train),
            "indicator_modes_supported": list(CANONICAL_INDICATOR_MODE_TRAIN_VALUES),
            "indicator_dropout_p": float(indicator_dropout_p),
            "indicator_dropout_scope": "training_only",
            "human_correction_override": True,
            "prompt_text_surface": "canonical_text_indicator",
            "prompt_conditioned_authority": "canonical_text_indicator_sidecar_only",
        }
    )

    return {
        "records_by_episode": records_by_episode,
        "threshold_estimates": {
            task_key: asdict(estimate)
            for task_key, estimate in threshold_estimates.items()
        },
        "epsilon_source": epsilon_source,
        "epsilon_quantile": float(effective_epsilon_quantile),
        "epsilon_threshold_phase": str(threshold_policy.phase),
        "epsilon_threshold_policy": build_phase_threshold_metadata(
            threshold_phase=threshold_policy.phase,
            epsilon_quantile=epsilon_quantile,
        ),
        "scale_metadata": scale_metadata,
        "raw_summary": raw_summary,
        "scaled_summary": scaled_summary,
        "advantage_contract": advantage_contract,
    }


__all__ = [
    "AdvantageRecord",
    "build_advantage_generation_plan",
    "emit_binary_indicator",
]
