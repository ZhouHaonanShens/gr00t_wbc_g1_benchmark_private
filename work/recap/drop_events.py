from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DROP_EVENT_ROW_SCHEMA_VERSION = "gr00t_drop_event_row_v1"
DROP_EVENT_ROW_ARTIFACT_KIND = "gr00t_drop_event_row"
DROP_SIDECAR_SUMMARY_SCHEMA_VERSION = "gr00t_drop_sidecar_summary_v1"
DROP_SIDECAR_SUMMARY_ARTIFACT_KIND = "gr00t_drop_sidecar_summary"
DROP_DETECTOR_EVAL_SCHEMA_VERSION = "gr00t_drop_detector_eval_v1"
DROP_DETECTOR_EVAL_ARTIFACT_KIND = "gr00t_drop_detector_eval"
COUNTERFACTUAL_REWARD_ROW_SCHEMA_VERSION = "gr00t_counterfactual_reward_row_v1"
COUNTERFACTUAL_REWARD_ROW_ARTIFACT_KIND = "gr00t_counterfactual_reward_row"
COUNTERFACTUAL_REWARD_SUMMARY_SCHEMA_VERSION = "gr00t_counterfactual_reward_summary_v1"
COUNTERFACTUAL_REWARD_SUMMARY_ARTIFACT_KIND = "gr00t_counterfactual_reward_summary"
REWARD_RECOMMENDATION_SCHEMA_VERSION = "gr00t_reward_recommendation_v1"
REWARD_RECOMMENDATION_ARTIFACT_KIND = "gr00t_reward_recommendation"
REWARD_RERUN_GATE_SCHEMA_VERSION = "gr00t_reward_rerun_gate_v1"
REWARD_RERUN_GATE_ARTIFACT_KIND = "gr00t_reward_rerun_gate"
REWARD_RERUN_GATE_NAME = "GR00TMainlineRewardRerunGate"

RECOMMENDATION_KEEP_OFFLINE = "keep_offline"
RECOMMENDATION_SHIP_SIDECAR_ONLY = "ship_sidecar_only"
RECOMMENDATION_ELIGIBLE_FOR_MAINLINE = "eligible_for_mainline"
REWARD_RECOMMENDATION_VALUES: tuple[str, ...] = (
    RECOMMENDATION_KEEP_OFFLINE,
    RECOMMENDATION_SHIP_SIDECAR_ONLY,
    RECOMMENDATION_ELIGIBLE_FOR_MAINLINE,
)

COUNTERFACTUAL_VARIANT_V0 = "V0_baseline_authority_reference"
COUNTERFACTUAL_VARIANT_V1 = "V1_equivalent_tail_early_termination"
COUNTERFACTUAL_VARIANT_V2 = "V2_add_drop_penalty"
COUNTERFACTUAL_VARIANT_V3 = "V3_early_termination_plus_drop_penalty"
COUNTERFACTUAL_VARIANTS: tuple[str, ...] = (
    COUNTERFACTUAL_VARIANT_V0,
    COUNTERFACTUAL_VARIANT_V1,
    COUNTERFACTUAL_VARIANT_V2,
    COUNTERFACTUAL_VARIANT_V3,
)

DROP_SIGNAL_SOURCE_DIRECT = "direct_drop_flag"
DROP_SIGNAL_SOURCE_IN_HAND_TRANSITION = "apple_in_hand_transition"
DROP_SIGNAL_SOURCE_IN_HAND_OBSERVATION = "apple_in_hand_observation"
DROP_SIGNAL_SOURCE_MISSING = "missing"
DROP_SIGNAL_SOURCE_VALUES: tuple[str, ...] = (
    DROP_SIGNAL_SOURCE_DIRECT,
    DROP_SIGNAL_SOURCE_IN_HAND_TRANSITION,
    DROP_SIGNAL_SOURCE_IN_HAND_OBSERVATION,
    DROP_SIGNAL_SOURCE_MISSING,
)

_PHASE_ALIASES = {
    "SEARCH": "SEARCH",
    "SEARCHING": "SEARCH",
    "APPROACH": "APPROACH",
    "APPLE_VISIBLE_APPROACH": "APPROACH",
    "GRASP": "GRASP",
    "GRASPING": "GRASP",
    "VERIFY_HOLD": "VERIFY_HOLD",
    "VERIFYING_HOLD": "VERIFY_HOLD",
    "TRANSPORT": "TRANSPORT",
    "TRANSPORTING": "TRANSPORT",
    "PLACE": "PLACE",
    "PLACING": "PLACE",
    "RECOVERY_REACQUIRE": "APPROACH",
    "RECOVERY_REGRASP": "GRASP",
}
DROP_ELIGIBLE_PHASES = frozenset({"VERIFY_HOLD", "TRANSPORT", "PLACE"})

MAINLINE_DROP_DETECTOR_PRECISION_THRESHOLD = 1.0
MAINLINE_DROP_DETECTOR_RECALL_THRESHOLD = 1.0
MIN_MAINLINE_POSITIVE_SUPPORT = 1
MIN_MAINLINE_NEGATIVE_SUPPORT = 1

RECOMMENDATION_REQUIRED_EVIDENCE_KEYS: tuple[str, ...] = (
    "drop_sidecar_jsonl",
    "drop_sidecar_summary_json",
    "drop_detector_eval_json",
    "counterfactual_rows_jsonl",
    "counterfactual_summary_json",
    "reward_counterfactual_report_md",
)


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    return _as_bool(value, field_name=field_name)


def _as_optional_path_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _deep_get(payload: Mapping[str, Any], dotted_path: str) -> object | None:
    current: object = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_field(record: Mapping[str, Any], *candidate_paths: str) -> object | None:
    for candidate in candidate_paths:
        if candidate in record:
            return record[candidate]
        resolved = _deep_get(record, candidate)
        if resolved is not None:
            return resolved
    return None


def _resolve_optional_bool(
    record: Mapping[str, Any], *candidate_paths: str
) -> bool | None:
    value = _resolve_field(record, *candidate_paths)
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    raise TypeError(
        f"{candidate_paths[0]} must resolve to a bool or null, got {type(value).__name__}"
    )


def _normalize_phase(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if not normalized:
        return None
    mapped = _PHASE_ALIASES.get(normalized)
    if mapped is None:
        raise ValueError(f"unsupported phase-like value: {value!r}")
    return mapped


def _sequence_length(value: object, *, field_name: str) -> int:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence, got {type(value).__name__}")
    return int(len(value))


def _compute_returns_mc_gamma1(rewards: Sequence[float]) -> list[float]:
    out = [0.0] * len(rewards)
    acc = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        acc += float(rewards[index])
        out[index] = float(acc)
    return out


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _dedupe_reasons(reasons: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for reason in reasons:
        normalized = str(reason).strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _record_join_key(record: Mapping[str, Any]) -> tuple[str, int]:
    episode_id = _as_non_empty_string(record.get("episode_id"), field_name="episode_id")
    t = _as_int(record.get("t"), field_name=f"episode_id={episode_id}.t")
    return episode_id, int(t)


def validate_drop_event_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(_as_mapping(row, field_name="drop_event_row"))
    if payload.get("schema_version") != DROP_EVENT_ROW_SCHEMA_VERSION:
        raise ValueError(
            f"drop_event_row.schema_version must equal {DROP_EVENT_ROW_SCHEMA_VERSION!r}"
        )
    if payload.get("artifact_kind") != DROP_EVENT_ROW_ARTIFACT_KIND:
        raise ValueError(
            f"drop_event_row.artifact_kind must equal {DROP_EVENT_ROW_ARTIFACT_KIND!r}"
        )
    normalized: dict[str, Any] = {
        "schema_version": DROP_EVENT_ROW_SCHEMA_VERSION,
        "artifact_kind": DROP_EVENT_ROW_ARTIFACT_KIND,
        "episode_id": _as_non_empty_string(
            payload.get("episode_id"), field_name="drop_event_row.episode_id"
        ),
        "t": _as_int(payload.get("t"), field_name="drop_event_row.t"),
        "reward_online": _as_number(
            payload.get("reward_online"), field_name="drop_event_row.reward_online"
        ),
        "success_step": _as_bool(
            payload.get("success_step"), field_name="drop_event_row.success_step"
        ),
        "inner_reward_count": _as_int(
            payload.get("inner_reward_count"),
            field_name="drop_event_row.inner_reward_count",
        ),
        "episode_return_online": _as_number(
            payload.get("episode_return_online"),
            field_name="drop_event_row.episode_return_online",
        ),
        "success_episode": _as_bool(
            payload.get("success_episode"), field_name="drop_event_row.success_episode"
        ),
        "phase": _normalize_phase(payload.get("phase")),
        "transport_context": _as_bool(
            payload.get("transport_context"),
            field_name="drop_event_row.transport_context",
        ),
        "detector_evidence_available": _as_bool(
            payload.get("detector_evidence_available"),
            field_name="drop_event_row.detector_evidence_available",
        ),
        "detector_signal_source": _as_non_empty_string(
            payload.get("detector_signal_source"),
            field_name="drop_event_row.detector_signal_source",
        ),
        "direct_drop_flag": _as_optional_bool(
            payload.get("direct_drop_flag"),
            field_name="drop_event_row.direct_drop_flag",
        ),
        "had_in_hand_previously": _as_bool(
            payload.get("had_in_hand_previously"),
            field_name="drop_event_row.had_in_hand_previously",
        ),
        "apple_in_hand": _as_optional_bool(
            payload.get("apple_in_hand"),
            field_name="drop_event_row.apple_in_hand",
        ),
        "drop_detected": _as_bool(
            payload.get("drop_detected"), field_name="drop_event_row.drop_detected"
        ),
        "drop_during_transport": _as_bool(
            payload.get("drop_during_transport"),
            field_name="drop_event_row.drop_during_transport",
        ),
    }
    if normalized["detector_signal_source"] not in DROP_SIGNAL_SOURCE_VALUES:
        raise ValueError(
            "drop_event_row.detector_signal_source must be one of "
            + f"{DROP_SIGNAL_SOURCE_VALUES!r}"
        )
    if normalized["inner_reward_count"] <= 0:
        raise ValueError("drop_event_row.inner_reward_count must be > 0")
    return normalized


def build_drop_episode_index(
    sidecar_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_row in sidecar_rows:
        row = validate_drop_event_row(raw_row)
        grouped.setdefault(str(row["episode_id"]), []).append(row)

    index: dict[str, dict[str, Any]] = {}
    for episode_id, rows in grouped.items():
        ordered_rows = sorted(rows, key=lambda item: int(item["t"]))
        success_values = {bool(row["success_episode"]) for row in ordered_rows}
        if len(success_values) != 1:
            raise ValueError(
                f"sidecar rows disagree on success_episode for episode {episode_id!r}"
            )
        return_values = {float(row["episode_return_online"]) for row in ordered_rows}
        if len(return_values) != 1:
            raise ValueError(
                f"sidecar rows disagree on episode_return_online for episode {episode_id!r}"
            )
        first_transport_drop_t = next(
            (
                int(row["t"])
                for row in ordered_rows
                if bool(row["drop_during_transport"])
            ),
            None,
        )
        rows_missing_evidence = sum(
            1 for row in ordered_rows if not bool(row["detector_evidence_available"])
        )
        index[episode_id] = {
            "episode_id": episode_id,
            "transition_count": int(len(ordered_rows)),
            "success_episode": bool(success_values.pop()),
            "episode_return_online": float(return_values.pop()),
            "has_drop": any(bool(row["drop_detected"]) for row in ordered_rows),
            "has_transport_drop": any(
                bool(row["drop_during_transport"]) for row in ordered_rows
            ),
            "first_transport_drop_t": first_transport_drop_t,
            "evidence_complete": rows_missing_evidence == 0,
            "rows_missing_detector_evidence": int(rows_missing_evidence),
        }
    return index


def summarize_drop_sidecar(
    sidecar_rows: Sequence[Mapping[str, Any]],
    *,
    expected_transition_keys: Sequence[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    rows = [validate_drop_event_row(row) for row in sidecar_rows]
    episode_index = build_drop_episode_index(rows)
    row_keys = {(_record_join_key(row)) for row in rows}
    expected_keys = set(expected_transition_keys or row_keys)
    missing_keys = sorted(expected_keys - row_keys)
    extra_keys = sorted(row_keys - expected_keys)
    joined_key_count = len(expected_keys & row_keys)
    coverage_ratio = _safe_rate(joined_key_count, len(expected_keys))
    return {
        "schema_version": DROP_SIDECAR_SUMMARY_SCHEMA_VERSION,
        "artifact_kind": DROP_SIDECAR_SUMMARY_ARTIFACT_KIND,
        "episode_count": int(len(episode_index)),
        "transition_row_count": int(len(rows)),
        "episodes_with_drop": int(
            sum(1 for item in episode_index.values() if bool(item["has_drop"]))
        ),
        "episodes_with_transport_drop": int(
            sum(
                1 for item in episode_index.values() if bool(item["has_transport_drop"])
            )
        ),
        "rows_with_missing_detector_evidence": int(
            sum(1 for row in rows if not bool(row["detector_evidence_available"]))
        ),
        "episodes_with_missing_detector_evidence": int(
            sum(
                1
                for item in episode_index.values()
                if not bool(item["evidence_complete"])
            )
        ),
        "expected_join_key_count": int(len(expected_keys)),
        "joined_key_count": int(joined_key_count),
        "coverage_ratio": float(coverage_ratio),
        "missing_join_keys": [[episode_id, t] for episode_id, t in missing_keys],
        "extra_join_keys": [[episode_id, t] for episode_id, t in extra_keys],
        "first_transport_drop_episode_ids": sorted(
            episode_id
            for episode_id, payload in episode_index.items()
            if payload["first_transport_drop_t"] is not None
        ),
    }


def build_drop_sidecar_rows(
    *,
    episodes: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes_by_id: dict[str, dict[str, Any]] = {}
    for raw_episode in episodes:
        episode = dict(_as_mapping(raw_episode, field_name="episode"))
        episode_id = _as_non_empty_string(
            episode.get("episode_id"), field_name="episode.episode_id"
        )
        if episode_id in episodes_by_id:
            raise ValueError(f"duplicate episode_id in episodes: {episode_id!r}")
        episodes_by_id[episode_id] = {
            "episode_id": episode_id,
            "episode_return_online": _as_number(
                episode.get("episode_return_online"),
                field_name=f"episode_id={episode_id}.episode_return_online",
            ),
            "success_episode": _as_bool(
                episode.get("success_episode"),
                field_name=f"episode_id={episode_id}.success_episode",
            ),
        }

    grouped_transitions: dict[str, list[dict[str, Any]]] = {}
    for raw_transition in transitions:
        transition = dict(_as_mapping(raw_transition, field_name="transition"))
        episode_id, _ = _record_join_key(transition)
        grouped_transitions.setdefault(episode_id, []).append(transition)

    rows: list[dict[str, Any]] = []
    expected_keys: list[tuple[str, int]] = []
    for episode_id, episode_transitions in sorted(grouped_transitions.items()):
        episode = episodes_by_id.get(episode_id)
        if episode is None:
            raise ValueError(
                f"transition references unknown episode_id: {episode_id!r}"
            )
        prior_in_hand = False
        for transition in sorted(episode_transitions, key=lambda item: int(item["t"])):
            t = _as_int(transition.get("t"), field_name=f"episode_id={episode_id}.t")
            expected_keys.append((episode_id, int(t)))
            inner_reward_count = _sequence_length(
                transition.get("inner_rewards"),
                field_name=f"episode_id={episode_id} t={t} inner_rewards",
            )
            reward_online = _as_number(
                transition.get("reward_online"),
                field_name=f"episode_id={episode_id} t={t} reward_online",
            )
            success_step = _as_bool(
                transition.get("success_step"),
                field_name=f"episode_id={episode_id} t={t} success_step",
            )
            phase = _normalize_phase(
                _resolve_field(
                    transition,
                    "analysis_only.semantic_state",
                    "semantic_state",
                    "policy_condition.phase",
                    "phase",
                )
            )
            direct_drop_flag = _resolve_optional_bool(
                transition,
                "drop_during_transport",
                "analysis_only.drop_during_transport",
                "diagnostic.drop_during_transport",
            )
            apple_in_hand = _resolve_optional_bool(
                transition,
                "privileged.apple_in_hand",
                "apple_in_hand",
                "analysis_only.apple_in_hand",
            )
            detector_evidence_available = (
                direct_drop_flag is not None or apple_in_hand is not None
            )
            drop_detected = False
            signal_source = DROP_SIGNAL_SOURCE_MISSING
            if direct_drop_flag is True:
                drop_detected = True
                signal_source = DROP_SIGNAL_SOURCE_DIRECT
            elif apple_in_hand is not None and prior_in_hand and not apple_in_hand:
                drop_detected = True
                signal_source = DROP_SIGNAL_SOURCE_IN_HAND_TRANSITION
            elif direct_drop_flag is False:
                signal_source = DROP_SIGNAL_SOURCE_DIRECT
            elif apple_in_hand is not None:
                signal_source = DROP_SIGNAL_SOURCE_IN_HAND_OBSERVATION
            transport_context = bool(
                phase in DROP_ELIGIBLE_PHASES or direct_drop_flag is True
            )
            row = {
                "schema_version": DROP_EVENT_ROW_SCHEMA_VERSION,
                "artifact_kind": DROP_EVENT_ROW_ARTIFACT_KIND,
                "episode_id": episode_id,
                "t": int(t),
                "reward_online": float(reward_online),
                "success_step": bool(success_step),
                "inner_reward_count": int(inner_reward_count),
                "episode_return_online": float(episode["episode_return_online"]),
                "success_episode": bool(episode["success_episode"]),
                "phase": phase,
                "transport_context": bool(transport_context),
                "detector_evidence_available": bool(detector_evidence_available),
                "detector_signal_source": signal_source,
                "direct_drop_flag": direct_drop_flag,
                "had_in_hand_previously": bool(prior_in_hand),
                "apple_in_hand": apple_in_hand,
                "drop_detected": bool(drop_detected),
                "drop_during_transport": bool(drop_detected and transport_context),
            }
            rows.append(validate_drop_event_row(row))
            if apple_in_hand is not None:
                prior_in_hand = bool(apple_in_hand)

    summary = summarize_drop_sidecar(rows, expected_transition_keys=expected_keys)
    return rows, summary


def _normalize_diagnostic_pool_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(_as_mapping(row, field_name="diagnostic_pool_row"))
    episode_id = _as_non_empty_string(
        payload.get("episode_id"), field_name="diagnostic_pool_row.episode_id"
    )
    raw_label = None
    for field_name in (
        "expected_drop_during_transport",
        "drop_during_transport",
        "expected_label",
        "label",
    ):
        if field_name in payload:
            raw_label = payload[field_name]
            break
    if isinstance(raw_label, bool):
        expected = bool(raw_label)
    elif isinstance(raw_label, str):
        normalized = raw_label.strip().lower()
        if normalized in {"drop_during_transport", "drop", "true", "1", "yes"}:
            expected = True
        elif normalized in {
            "no_drop_during_transport",
            "nominal",
            "false",
            "0",
            "no",
            "success",
        }:
            expected = False
        else:
            raise ValueError(
                f"diagnostic_pool_row.label is unsupported for episode {episode_id!r}: {raw_label!r}"
            )
    else:
        raise ValueError(
            f"diagnostic_pool_row for episode {episode_id!r} is missing a bool-like drop label"
        )
    return {
        "episode_id": episode_id,
        "expected_drop_during_transport": bool(expected),
    }


def evaluate_drop_detector_against_diagnostic_pool(
    *,
    sidecar_rows: Sequence[Mapping[str, Any]],
    diagnostic_pool_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sidecar_summary = summarize_drop_sidecar(sidecar_rows)
    episode_index = build_drop_episode_index(sidecar_rows)
    normalized_pool = [
        _normalize_diagnostic_pool_row(row) for row in diagnostic_pool_rows
    ]
    tp = fp = tn = fn = 0
    missing_sidecar_episode_ids: list[str] = []
    missing_evidence_episode_ids: list[str] = []
    evaluated_episode_count = 0
    positive_support = 0
    negative_support = 0
    predicted_positive_episode_ids: list[str] = []
    expected_positive_episode_ids: list[str] = []
    for row in normalized_pool:
        episode_id = str(row["episode_id"])
        episode_payload = episode_index.get(episode_id)
        if episode_payload is None:
            missing_sidecar_episode_ids.append(episode_id)
            continue
        if not bool(episode_payload["evidence_complete"]):
            missing_evidence_episode_ids.append(episode_id)
            continue
        expected_positive = bool(row["expected_drop_during_transport"])
        predicted_positive = bool(episode_payload["has_transport_drop"])
        evaluated_episode_count += 1
        if expected_positive:
            positive_support += 1
            expected_positive_episode_ids.append(episode_id)
        else:
            negative_support += 1
        if predicted_positive:
            predicted_positive_episode_ids.append(episode_id)
        if expected_positive and predicted_positive:
            tp += 1
        elif (not expected_positive) and predicted_positive:
            fp += 1
        elif expected_positive and (not predicted_positive):
            fn += 1
        else:
            tn += 1

    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    accuracy = _safe_rate(tp + tn, evaluated_episode_count)
    support_underflow = (
        positive_support < MIN_MAINLINE_POSITIVE_SUPPORT
        or negative_support < MIN_MAINLINE_NEGATIVE_SUPPORT
    )
    sidecar_publishable = (
        not missing_sidecar_episode_ids
        and not missing_evidence_episode_ids
        and int(sidecar_summary["episodes_with_missing_detector_evidence"]) == 0
        and evaluated_episode_count == len(normalized_pool)
    )
    thresholds_passed = (
        precision >= MAINLINE_DROP_DETECTOR_PRECISION_THRESHOLD
        and recall >= MAINLINE_DROP_DETECTOR_RECALL_THRESHOLD
    )
    mainline_stable = bool(
        sidecar_publishable and (not support_underflow) and thresholds_passed
    )
    failure_reasons: list[str] = []
    if missing_sidecar_episode_ids:
        failure_reasons.append("diagnostic_pool_episode_missing_from_sidecar")
    if missing_evidence_episode_ids:
        failure_reasons.append("diagnostic_pool_episode_missing_detector_evidence")
    if support_underflow:
        failure_reasons.append("diagnostic_pool_support_underflow")
    if sidecar_publishable and not thresholds_passed:
        failure_reasons.append("drop_detector_not_stable_enough_for_mainline")
    if mainline_stable:
        reward_recommendation = RECOMMENDATION_ELIGIBLE_FOR_MAINLINE
    elif sidecar_publishable and evaluated_episode_count > 0:
        reward_recommendation = RECOMMENDATION_SHIP_SIDECAR_ONLY
    else:
        reward_recommendation = RECOMMENDATION_KEEP_OFFLINE
    return {
        "schema_version": DROP_DETECTOR_EVAL_SCHEMA_VERSION,
        "artifact_kind": DROP_DETECTOR_EVAL_ARTIFACT_KIND,
        "status": "PASS" if sidecar_publishable else "FAIL",
        "formal_eligibility": (
            "ALLOW"
            if reward_recommendation == RECOMMENDATION_ELIGIBLE_FOR_MAINLINE
            else "BLOCK"
        ),
        "reward_recommendation": reward_recommendation,
        "sidecar_publishable": bool(sidecar_publishable),
        "mainline_stable": bool(mainline_stable),
        "support_underflow": bool(support_underflow),
        "evaluated_episode_count": int(evaluated_episode_count),
        "diagnostic_pool_episode_count": int(len(normalized_pool)),
        "positive_support": int(positive_support),
        "negative_support": int(negative_support),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "predicted_positive_episode_ids": sorted(predicted_positive_episode_ids),
        "expected_positive_episode_ids": sorted(expected_positive_episode_ids),
        "missing_sidecar_episode_ids": sorted(missing_sidecar_episode_ids),
        "missing_detector_evidence_episode_ids": sorted(missing_evidence_episode_ids),
        "threshold_policy": {
            "precision": float(MAINLINE_DROP_DETECTOR_PRECISION_THRESHOLD),
            "recall": float(MAINLINE_DROP_DETECTOR_RECALL_THRESHOLD),
            "min_positive_support": int(MIN_MAINLINE_POSITIVE_SUPPORT),
            "min_negative_support": int(MIN_MAINLINE_NEGATIVE_SUPPORT),
        },
        "failure_reasons": _dedupe_reasons(failure_reasons),
    }


def _assert_counterfactual_variant(variant: str) -> str:
    normalized = _as_non_empty_string(variant, field_name="counterfactual_variant")
    if normalized not in COUNTERFACTUAL_VARIANTS:
        raise ValueError(
            f"counterfactual_variant must be one of {COUNTERFACTUAL_VARIANTS!r}"
        )
    return normalized


def _apply_counterfactual_variant(
    *,
    authority_rewards: Sequence[float],
    drop_t: int | None,
    c_fail: float,
    variant: str,
) -> list[float]:
    rewards = [float(value) for value in authority_rewards]
    normalized_variant = _assert_counterfactual_variant(variant)
    if drop_t is None:
        return list(rewards)
    if drop_t < 0 or drop_t >= len(rewards):
        raise ValueError(f"drop_t out of range: drop_t={drop_t} len={len(rewards)}")
    if normalized_variant == COUNTERFACTUAL_VARIANT_V0:
        return list(rewards)
    if normalized_variant == COUNTERFACTUAL_VARIANT_V1:
        tail_sum = float(sum(rewards[drop_t:]))
        updated = list(rewards)
        updated[drop_t] = float(tail_sum)
        for index in range(drop_t + 1, len(updated)):
            updated[index] = 0.0
        return updated
    if normalized_variant == COUNTERFACTUAL_VARIANT_V2:
        updated = list(rewards)
        updated[drop_t] = float(updated[drop_t] - float(c_fail))
        return updated
    updated = _apply_counterfactual_variant(
        authority_rewards=rewards,
        drop_t=drop_t,
        c_fail=c_fail,
        variant=COUNTERFACTUAL_VARIANT_V1,
    )
    updated[drop_t] = float(updated[drop_t] - float(c_fail))
    return updated


def validate_counterfactual_reward_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(_as_mapping(row, field_name="counterfactual_reward_row"))
    if payload.get("schema_version") != COUNTERFACTUAL_REWARD_ROW_SCHEMA_VERSION:
        raise ValueError(
            "counterfactual_reward_row.schema_version must equal "
            + f"{COUNTERFACTUAL_REWARD_ROW_SCHEMA_VERSION!r}"
        )
    if payload.get("artifact_kind") != COUNTERFACTUAL_REWARD_ROW_ARTIFACT_KIND:
        raise ValueError(
            "counterfactual_reward_row.artifact_kind must equal "
            + f"{COUNTERFACTUAL_REWARD_ROW_ARTIFACT_KIND!r}"
        )
    normalized = {
        "schema_version": COUNTERFACTUAL_REWARD_ROW_SCHEMA_VERSION,
        "artifact_kind": COUNTERFACTUAL_REWARD_ROW_ARTIFACT_KIND,
        "variant": _assert_counterfactual_variant(str(payload.get("variant"))),
        "episode_id": _as_non_empty_string(
            payload.get("episode_id"), field_name="counterfactual_reward_row.episode_id"
        ),
        "t": _as_int(payload.get("t"), field_name="counterfactual_reward_row.t"),
        "reward_online_authority": _as_number(
            payload.get("reward_online_authority"),
            field_name="counterfactual_reward_row.reward_online_authority",
        ),
        "reward_counterfactual": _as_number(
            payload.get("reward_counterfactual"),
            field_name="counterfactual_reward_row.reward_counterfactual",
        ),
        "return_G_authority": _as_number(
            payload.get("return_G_authority"),
            field_name="counterfactual_reward_row.return_G_authority",
        ),
        "return_G_counterfactual": _as_number(
            payload.get("return_G_counterfactual"),
            field_name="counterfactual_reward_row.return_G_counterfactual",
        ),
        "delta_reward": _as_number(
            payload.get("delta_reward"),
            field_name="counterfactual_reward_row.delta_reward",
        ),
        "delta_return_G": _as_number(
            payload.get("delta_return_G"),
            field_name="counterfactual_reward_row.delta_return_G",
        ),
        "success_step": _as_bool(
            payload.get("success_step"),
            field_name="counterfactual_reward_row.success_step",
        ),
        "first_transport_drop_t": payload.get("first_transport_drop_t"),
        "drop_during_transport_episode": _as_bool(
            payload.get("drop_during_transport_episode"),
            field_name="counterfactual_reward_row.drop_during_transport_episode",
        ),
        "counterfactual_applied": _as_bool(
            payload.get("counterfactual_applied"),
            field_name="counterfactual_reward_row.counterfactual_applied",
        ),
        "episode_return_online_authority": _as_number(
            payload.get("episode_return_online_authority"),
            field_name="counterfactual_reward_row.episode_return_online_authority",
        ),
        "episode_return_online_counterfactual": _as_number(
            payload.get("episode_return_online_counterfactual"),
            field_name="counterfactual_reward_row.episode_return_online_counterfactual",
        ),
    }
    if normalized["first_transport_drop_t"] is not None:
        normalized["first_transport_drop_t"] = _as_int(
            normalized["first_transport_drop_t"],
            field_name="counterfactual_reward_row.first_transport_drop_t",
        )
    return normalized


def relabel_counterfactual_rewards(
    *,
    episodes: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    variants: Sequence[str] = COUNTERFACTUAL_VARIANTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes_by_id: dict[str, dict[str, Any]] = {}
    for raw_episode in episodes:
        episode = dict(_as_mapping(raw_episode, field_name="episode"))
        episode_id = _as_non_empty_string(
            episode.get("episode_id"), field_name="episode.episode_id"
        )
        episodes_by_id[episode_id] = {
            "episode_id": episode_id,
            "episode_return_online": _as_number(
                episode.get("episode_return_online"),
                field_name=f"episode_id={episode_id}.episode_return_online",
            ),
            "C_fail": _as_number(
                episode.get("C_fail"), field_name=f"episode_id={episode_id}.C_fail"
            ),
            "success_episode": _as_bool(
                episode.get("success_episode"),
                field_name=f"episode_id={episode_id}.success_episode",
            ),
        }

    grouped_transitions: dict[str, list[dict[str, Any]]] = {}
    for raw_transition in transitions:
        transition = dict(_as_mapping(raw_transition, field_name="transition"))
        episode_id, _ = _record_join_key(transition)
        grouped_transitions.setdefault(episode_id, []).append(transition)

    sidecar_index = build_drop_episode_index(sidecar_rows)
    sidecar_keys = {_record_join_key(row) for row in sidecar_rows}
    transition_keys = {_record_join_key(row) for row in transitions}
    if sidecar_keys != transition_keys:
        missing = sorted(transition_keys - sidecar_keys)
        extra = sorted(sidecar_keys - transition_keys)
        raise ValueError(
            "drop sidecar join mismatch: " + f"missing={missing[:8]} extra={extra[:8]}"
        )

    rows: list[dict[str, Any]] = []
    variant_shift_values: dict[str, list[float]] = {
        _assert_counterfactual_variant(variant): [] for variant in variants
    }
    variant_affected_episode_ids: dict[str, list[str]] = {
        variant: [] for variant in variant_shift_values
    }
    for episode_id, episode_payload in sorted(episodes_by_id.items()):
        episode_transitions = sorted(
            grouped_transitions.get(episode_id, []), key=lambda item: int(item["t"])
        )
        if not episode_transitions:
            continue
        sidecar_episode = sidecar_index.get(episode_id)
        if sidecar_episode is None:
            raise ValueError(f"missing sidecar episode coverage for {episode_id!r}")
        authority_rewards = [
            _as_number(
                transition.get("reward_online"),
                field_name=f"episode_id={episode_id} t={transition.get('t')} reward_online",
            )
            for transition in episode_transitions
        ]
        for transition in episode_transitions:
            _ = _sequence_length(
                transition.get("inner_rewards"),
                field_name=f"episode_id={episode_id} t={transition.get('t')} inner_rewards",
            )
        authority_total = float(sum(authority_rewards))
        recorded_total = float(episode_payload["episode_return_online"])
        if abs(authority_total - recorded_total) > 1e-6:
            raise ValueError(
                f"episode_id={episode_id} reward authority mismatch: transitions sum to {authority_total} but episode_return_online={recorded_total}"
            )
        authority_returns = _compute_returns_mc_gamma1(authority_rewards)
        raw_drop_t = sidecar_episode.get("first_transport_drop_t")
        drop_t = None if raw_drop_t is None else int(raw_drop_t)
        for variant in variant_shift_values:
            counterfactual_rewards = _apply_counterfactual_variant(
                authority_rewards=authority_rewards,
                drop_t=None if not sidecar_episode["has_transport_drop"] else drop_t,
                c_fail=float(episode_payload["C_fail"]),
                variant=variant,
            )
            counterfactual_returns = _compute_returns_mc_gamma1(counterfactual_rewards)
            episode_return_counterfactual = float(sum(counterfactual_rewards))
            episode_shift = float(episode_return_counterfactual - authority_total)
            variant_shift_values[variant].append(float(episode_shift))
            if abs(episode_shift) > 1e-9 or counterfactual_rewards != authority_rewards:
                variant_affected_episode_ids[variant].append(episode_id)
            for (
                transition,
                authority_g,
                counterfactual_g,
                reward_authority,
                reward_cf,
            ) in zip(
                episode_transitions,
                authority_returns,
                counterfactual_returns,
                authority_rewards,
                counterfactual_rewards,
                strict=True,
            ):
                row = {
                    "schema_version": COUNTERFACTUAL_REWARD_ROW_SCHEMA_VERSION,
                    "artifact_kind": COUNTERFACTUAL_REWARD_ROW_ARTIFACT_KIND,
                    "variant": variant,
                    "episode_id": episode_id,
                    "t": _as_int(transition.get("t"), field_name="transition.t"),
                    "reward_online_authority": float(reward_authority),
                    "reward_counterfactual": float(reward_cf),
                    "return_G_authority": float(authority_g),
                    "return_G_counterfactual": float(counterfactual_g),
                    "delta_reward": float(reward_cf - reward_authority),
                    "delta_return_G": float(counterfactual_g - authority_g),
                    "success_step": _as_bool(
                        transition.get("success_step"),
                        field_name=f"episode_id={episode_id} success_step",
                    ),
                    "first_transport_drop_t": drop_t,
                    "drop_during_transport_episode": bool(
                        sidecar_episode["has_transport_drop"]
                    ),
                    "counterfactual_applied": bool(
                        counterfactual_rewards != authority_rewards
                    ),
                    "episode_return_online_authority": float(authority_total),
                    "episode_return_online_counterfactual": float(
                        episode_return_counterfactual
                    ),
                }
                rows.append(validate_counterfactual_reward_row(row))

    variant_summaries: dict[str, dict[str, Any]] = {}
    for variant in variant_shift_values:
        variant_rows = [row for row in rows if str(row["variant"]) == variant]
        shifts = variant_shift_values[variant]
        variant_summaries[variant] = {
            "row_count": int(len(variant_rows)),
            "episode_count": int(len(shifts)),
            "affected_episode_count": int(
                len(set(variant_affected_episode_ids[variant]))
            ),
            "affected_episode_ids": sorted(set(variant_affected_episode_ids[variant])),
            "mean_episode_return_shift": (
                float(sum(shifts) / float(len(shifts))) if shifts else 0.0
            ),
            "min_episode_return_shift": float(min(shifts)) if shifts else 0.0,
            "max_episode_return_shift": float(max(shifts)) if shifts else 0.0,
            "counterfactual_applied": any(
                bool(row["counterfactual_applied"]) for row in variant_rows
            ),
        }

    summary = {
        "schema_version": COUNTERFACTUAL_REWARD_SUMMARY_SCHEMA_VERSION,
        "artifact_kind": COUNTERFACTUAL_REWARD_SUMMARY_ARTIFACT_KIND,
        "row_count": int(len(rows)),
        "variant_count": int(len(variant_summaries)),
        "episode_count": int(len(episodes_by_id)),
        "mainline_candidate_variant": COUNTERFACTUAL_VARIANT_V1,
        "variants": variant_summaries,
    }
    return rows, summary


def _validate_sidecar_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(_as_mapping(payload, field_name="drop_sidecar_summary"))
    if summary.get("schema_version") != DROP_SIDECAR_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            "drop_sidecar_summary.schema_version must equal "
            + f"{DROP_SIDECAR_SUMMARY_SCHEMA_VERSION!r}"
        )
    if summary.get("artifact_kind") != DROP_SIDECAR_SUMMARY_ARTIFACT_KIND:
        raise ValueError(
            "drop_sidecar_summary.artifact_kind must equal "
            + f"{DROP_SIDECAR_SUMMARY_ARTIFACT_KIND!r}"
        )
    return summary


def _validate_detector_eval(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(_as_mapping(payload, field_name="drop_detector_eval"))
    if summary.get("schema_version") != DROP_DETECTOR_EVAL_SCHEMA_VERSION:
        raise ValueError(
            "drop_detector_eval.schema_version must equal "
            + f"{DROP_DETECTOR_EVAL_SCHEMA_VERSION!r}"
        )
    if summary.get("artifact_kind") != DROP_DETECTOR_EVAL_ARTIFACT_KIND:
        raise ValueError(
            "drop_detector_eval.artifact_kind must equal "
            + f"{DROP_DETECTOR_EVAL_ARTIFACT_KIND!r}"
        )
    return summary


def _validate_counterfactual_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(_as_mapping(payload, field_name="counterfactual_reward_summary"))
    if summary.get("schema_version") != COUNTERFACTUAL_REWARD_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            "counterfactual_reward_summary.schema_version must equal "
            + f"{COUNTERFACTUAL_REWARD_SUMMARY_SCHEMA_VERSION!r}"
        )
    if summary.get("artifact_kind") != COUNTERFACTUAL_REWARD_SUMMARY_ARTIFACT_KIND:
        raise ValueError(
            "counterfactual_reward_summary.artifact_kind must equal "
            + f"{COUNTERFACTUAL_REWARD_SUMMARY_ARTIFACT_KIND!r}"
        )
    variants = _as_mapping(
        summary.get("variants"), field_name="counterfactual_summary.variants"
    )
    if COUNTERFACTUAL_VARIANT_V1 not in variants:
        raise ValueError(
            "counterfactual_summary.variants missing V1 mainline candidate"
        )
    return summary


def build_reward_recommendation(
    *,
    sidecar_summary: Mapping[str, Any],
    counterfactual_summary: Mapping[str, Any],
    detector_eval: Mapping[str, Any] | None,
    evidence_paths: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    failure_reasons: list[str] = []
    normalized_sidecar = _validate_sidecar_summary(sidecar_summary)
    normalized_counterfactual = _validate_counterfactual_summary(counterfactual_summary)
    normalized_detector: dict[str, Any] | None = None
    if detector_eval is None:
        failure_reasons.append("drop_detector_eval_missing")
    else:
        normalized_detector = _validate_detector_eval(detector_eval)

    rows_missing_evidence = _as_int(
        normalized_sidecar.get("rows_with_missing_detector_evidence", 0),
        field_name="drop_sidecar_summary.rows_with_missing_detector_evidence",
    )
    if rows_missing_evidence > 0:
        failure_reasons.append("drop_sidecar_detector_evidence_incomplete")

    v1_summary = _as_mapping(
        _as_mapping(
            normalized_counterfactual.get("variants"), field_name="variants"
        ).get(COUNTERFACTUAL_VARIANT_V1),
        field_name="variants.V1",
    )
    if _as_int(v1_summary.get("row_count"), field_name="variants.V1.row_count") <= 0:
        failure_reasons.append("counterfactual_v1_rows_missing")

    sidecar_publishable = False
    mainline_stable = False
    support_underflow = True
    precision = 0.0
    recall = 0.0
    if normalized_detector is not None:
        sidecar_publishable = _as_bool(
            normalized_detector.get("sidecar_publishable"),
            field_name="drop_detector_eval.sidecar_publishable",
        )
        mainline_stable = _as_bool(
            normalized_detector.get("mainline_stable"),
            field_name="drop_detector_eval.mainline_stable",
        )
        support_underflow = _as_bool(
            normalized_detector.get("support_underflow"),
            field_name="drop_detector_eval.support_underflow",
        )
        precision = _as_number(
            normalized_detector.get("precision"),
            field_name="drop_detector_eval.precision",
        )
        recall = _as_number(
            normalized_detector.get("recall"), field_name="drop_detector_eval.recall"
        )
        failure_reasons.extend(
            str(reason)
            for reason in normalized_detector.get("failure_reasons", [])
            if str(reason).strip()
        )

    if (
        normalized_detector is not None
        and sidecar_publishable
        and mainline_stable
        and not failure_reasons
    ):
        recommendation = RECOMMENDATION_ELIGIBLE_FOR_MAINLINE
    elif normalized_detector is not None and sidecar_publishable:
        recommendation = RECOMMENDATION_SHIP_SIDECAR_ONLY
    else:
        recommendation = RECOMMENDATION_KEEP_OFFLINE

    normalized_evidence_paths: dict[str, str | None] = {
        key: None for key in RECOMMENDATION_REQUIRED_EVIDENCE_KEYS
    }
    if evidence_paths is not None:
        for key in RECOMMENDATION_REQUIRED_EVIDENCE_KEYS:
            normalized_evidence_paths[key] = _as_optional_path_string(
                evidence_paths.get(key)
            )

    return {
        "schema_version": REWARD_RECOMMENDATION_SCHEMA_VERSION,
        "artifact_kind": REWARD_RECOMMENDATION_ARTIFACT_KIND,
        "reward_recommendation": recommendation,
        "status": (
            "PASS" if recommendation == RECOMMENDATION_ELIGIBLE_FOR_MAINLINE else "FAIL"
        ),
        "formal_eligibility": (
            "ALLOW"
            if recommendation == RECOMMENDATION_ELIGIBLE_FOR_MAINLINE
            else "BLOCK"
        ),
        "mainline_reward_rerun_allowed": (
            recommendation == RECOMMENDATION_ELIGIBLE_FOR_MAINLINE
        ),
        "ship_sidecar_publish_allowed": recommendation
        in {RECOMMENDATION_SHIP_SIDECAR_ONLY, RECOMMENDATION_ELIGIBLE_FOR_MAINLINE},
        "diagnostic_only": recommendation != RECOMMENDATION_ELIGIBLE_FOR_MAINLINE,
        "mainline_candidate_variant": COUNTERFACTUAL_VARIANT_V1,
        "failure_reasons": _dedupe_reasons(failure_reasons),
        "detector_snapshot": {
            "available": normalized_detector is not None,
            "sidecar_publishable": bool(sidecar_publishable),
            "mainline_stable": bool(mainline_stable),
            "support_underflow": bool(support_underflow),
            "precision": float(precision),
            "recall": float(recall),
        },
        "counterfactual_snapshot": {
            "variant_count": _as_int(
                normalized_counterfactual.get("variant_count"),
                field_name="counterfactual_summary.variant_count",
            ),
            "row_count": _as_int(
                normalized_counterfactual.get("row_count"),
                field_name="counterfactual_summary.row_count",
            ),
            "mainline_candidate_variant": str(
                normalized_counterfactual.get("mainline_candidate_variant")
            ),
            "v1_affected_episode_count": _as_int(
                v1_summary.get("affected_episode_count"),
                field_name="counterfactual_summary.variants.V1.affected_episode_count",
            ),
        },
        "evidence_paths": normalized_evidence_paths,
    }


def validate_reward_recommendation_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(_as_mapping(payload, field_name="reward_recommendation"))
    if normalized.get("schema_version") != REWARD_RECOMMENDATION_SCHEMA_VERSION:
        raise ValueError(
            "reward_recommendation.schema_version must equal "
            + f"{REWARD_RECOMMENDATION_SCHEMA_VERSION!r}"
        )
    if normalized.get("artifact_kind") != REWARD_RECOMMENDATION_ARTIFACT_KIND:
        raise ValueError(
            "reward_recommendation.artifact_kind must equal "
            + f"{REWARD_RECOMMENDATION_ARTIFACT_KIND!r}"
        )
    recommendation = _as_non_empty_string(
        normalized.get("reward_recommendation"),
        field_name="reward_recommendation.reward_recommendation",
    )
    if recommendation not in REWARD_RECOMMENDATION_VALUES:
        raise ValueError(
            "reward_recommendation.reward_recommendation must be one of "
            + f"{REWARD_RECOMMENDATION_VALUES!r}"
        )
    _ = _as_bool(
        normalized.get("mainline_reward_rerun_allowed"),
        field_name="reward_recommendation.mainline_reward_rerun_allowed",
    )
    _ = _as_bool(
        normalized.get("ship_sidecar_publish_allowed"),
        field_name="reward_recommendation.ship_sidecar_publish_allowed",
    )
    _ = _as_bool(
        normalized.get("diagnostic_only"),
        field_name="reward_recommendation.diagnostic_only",
    )
    _assert_counterfactual_variant(
        _as_non_empty_string(
            normalized.get("mainline_candidate_variant"),
            field_name="reward_recommendation.mainline_candidate_variant",
        )
    )
    failure_reasons = normalized.get("failure_reasons")
    if not isinstance(failure_reasons, list) or any(
        not isinstance(item, str) or not item.strip() for item in failure_reasons
    ):
        raise TypeError(
            "reward_recommendation.failure_reasons must be a non-empty string list"
        )
    evidence_paths = _as_mapping(
        normalized.get("evidence_paths"),
        field_name="reward_recommendation.evidence_paths",
    )
    for key in RECOMMENDATION_REQUIRED_EVIDENCE_KEYS:
        _ = _as_optional_path_string(evidence_paths.get(key))
    return normalized


def build_mainline_reward_rerun_precondition_report(
    recommendation_payload: Mapping[str, Any],
    *,
    require_existing_paths: bool = True,
) -> dict[str, Any]:
    failure_reasons: list[str] = []
    try:
        normalized = validate_reward_recommendation_payload(recommendation_payload)
    except Exception as exc:
        normalized = {
            "reward_recommendation": RECOMMENDATION_KEEP_OFFLINE,
            "evidence_paths": {
                key: None for key in RECOMMENDATION_REQUIRED_EVIDENCE_KEYS
            },
            "failure_reasons": [str(exc)],
            "mainline_candidate_variant": COUNTERFACTUAL_VARIANT_V1,
        }
        failure_reasons.append("invalid_reward_recommendation_payload")

    reward_recommendation = str(normalized.get("reward_recommendation"))
    failure_reasons.extend(
        str(reason)
        for reason in normalized.get("failure_reasons", [])
        if str(reason).strip()
    )
    if reward_recommendation != RECOMMENDATION_ELIGIBLE_FOR_MAINLINE:
        failure_reasons.append(f"reward_recommendation_{reward_recommendation}")

    evidence_paths = _as_mapping(
        normalized.get("evidence_paths", {}), field_name="evidence_paths"
    )
    if require_existing_paths:
        for key in RECOMMENDATION_REQUIRED_EVIDENCE_KEYS:
            raw_path = _as_optional_path_string(evidence_paths.get(key))
            if raw_path is None:
                failure_reasons.append(f"missing_evidence_path:{key}")
                continue
            if not Path(raw_path).expanduser().exists():
                failure_reasons.append(f"evidence_path_not_found:{key}")

    deduped_reasons = _dedupe_reasons(failure_reasons)
    formal_eligibility = "ALLOW" if not deduped_reasons else "BLOCK"
    return {
        "schema_version": REWARD_RERUN_GATE_SCHEMA_VERSION,
        "artifact_kind": REWARD_RERUN_GATE_ARTIFACT_KIND,
        "gate_name": REWARD_RERUN_GATE_NAME,
        "status": "PASS" if formal_eligibility == "ALLOW" else "FAIL",
        "formal_eligibility": formal_eligibility,
        "reward_recommendation": reward_recommendation,
        "mainline_reward_rerun_allowed": formal_eligibility == "ALLOW",
        "mainline_candidate_variant": str(
            normalized.get("mainline_candidate_variant", COUNTERFACTUAL_VARIANT_V1)
        ),
        "failure_reasons": deduped_reasons,
        "evidence_paths": {
            key: _as_optional_path_string(evidence_paths.get(key))
            for key in RECOMMENDATION_REQUIRED_EVIDENCE_KEYS
        },
    }


def render_reward_counterfactual_report(
    *,
    sidecar_summary: Mapping[str, Any],
    detector_eval: Mapping[str, Any] | None,
    counterfactual_summary: Mapping[str, Any],
    recommendation: Mapping[str, Any],
) -> str:
    sidecar = _validate_sidecar_summary(sidecar_summary)
    counterfactual = _validate_counterfactual_summary(counterfactual_summary)
    reco = validate_reward_recommendation_payload(recommendation)
    detector_lines: list[str] = []
    if detector_eval is None:
        detector_lines.append("- detector audit: 缺失（fail-closed）")
    else:
        detector = _validate_detector_eval(detector_eval)
        detector_lines.extend(
            [
                f"- detector audit status: `{detector.get('status')}`",
                f"- precision / recall / accuracy: `{detector.get('precision')}` / `{detector.get('recall')}` / `{detector.get('accuracy')}`",
                f"- support underflow: `{detector.get('support_underflow')}`",
                f"- sidecar publishable: `{detector.get('sidecar_publishable')}`",
                f"- mainline stable: `{detector.get('mainline_stable')}`",
            ]
        )
    variant_lines: list[str] = []
    variants = _as_mapping(counterfactual.get("variants"), field_name="variants")
    for variant in COUNTERFACTUAL_VARIANTS:
        payload = variants.get(variant)
        if not isinstance(payload, Mapping):
            continue
        variant_lines.append(
            "- "
            + f"{variant}: affected_episode_count=`{payload.get('affected_episode_count')}`, "
            + f"min_shift=`{payload.get('min_episode_return_shift')}`, "
            + f"max_shift=`{payload.get('max_episode_return_shift')}`"
        )
    failure_reasons = reco.get("failure_reasons", [])
    rendered_reasons = [
        f"  - `{reason}`" for reason in failure_reasons if isinstance(reason, str)
    ] or ["  - `(none)`"]
    lines = [
        "# Reward / Drop Counterfactual Report",
        "",
        "## Summary",
        f"- reward recommendation: `{reco.get('reward_recommendation')}`",
        f"- formal_eligibility: `{reco.get('formal_eligibility')}`",
        f"- ship_sidecar_publish_allowed: `{reco.get('ship_sidecar_publish_allowed')}`",
        f"- mainline_reward_rerun_allowed: `{reco.get('mainline_reward_rerun_allowed')}`",
        f"- sidecar coverage_ratio: `{sidecar.get('coverage_ratio')}`",
        f"- episodes_with_transport_drop: `{sidecar.get('episodes_with_transport_drop')}`",
        f"- rows_with_missing_detector_evidence: `{sidecar.get('rows_with_missing_detector_evidence')}`",
        "",
        "## Detector audit",
        *detector_lines,
        "",
        "## Counterfactual variants",
        *variant_lines,
        "",
        "## Fail-closed reasons",
        *rendered_reasons,
        "",
        "说明：该产物是 authority reward contract 之外的离线第二表面；即使 sidecar / audit 可发布，只有 recommendation=`eligible_for_mainline` 时才允许 reward-aware mainline rerun。",
        "",
    ]
    return "\n".join(lines)
