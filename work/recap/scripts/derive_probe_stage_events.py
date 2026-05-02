from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


REPORT_SCHEMA_VERSION = "exact_probe_stage_events_v1"
REPORT_ARTIFACT_KIND = "exact_probe_stage_events"
DEFAULT_OUTPUT_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/exact_probe_stage_events.json"
)

APPROACH_DISTANCE_THRESHOLD_M = 0.10
APPLE_LIFT_THRESHOLD_M = 0.03
NEAR_PLATE_DISTANCE_THRESHOLD_M = 0.12
PLATE_PROGRESS_THRESHOLD_M = 0.05

DIRECT_SIGNAL_CONFIDENCE = 0.95
GEOMETRY_SIGNAL_CONFIDENCE = 0.75
HEURISTIC_SIGNAL_CONFIDENCE = 0.45
LOW_SIGNAL_CONFIDENCE = 0.20

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import


_MISSING = object()

PHASE_ALIASES = {
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

SEMANTIC_STAGE_ORDER = ("APPROACH", "GRASP", "VERIFY_HOLD", "TRANSPORT", "PLACE")

FAILURE_STAGE_GUESS_TO_STAGE_REASON = {
    "never_reached_apple": "approach_failed",
    "reached_apple_not_lifted": "grasp_failed",
    "lifted_not_brought_to_plate": "transport_failed",
    "near_plate_but_not_success": "place_failed",
    "SEARCH": "approach_failed",
    "APPROACH": "approach_failed",
    "GRASP": "grasp_failed",
    "VERIFY_HOLD": "grasp_failed",
    "TRANSPORT": "transport_failed",
    "PLACE": "place_failed",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="derive_probe_stage_events.py",
        description=(
            "Derive fail-soft exact-probe stage events from optional episode, step, drop, "
            "and runtime-trace artifacts using direct signals first, then geometry, then "
            "heuristic fallback."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--episode-json",
        type=Path,
        default=None,
        help="Optional episode-level JSON payload carrying failure_reason/failure_stage_guess.",
    )
    parser.add_argument(
        "--steps-jsonl",
        type=Path,
        default=None,
        help="Optional step-level JSONL payload from exact-probe-like telemetry.",
    )
    parser.add_argument(
        "--runtime-trace-json",
        type=Path,
        default=None,
        help="Optional runtime-trace JSON with prompt/token/stage delta diagnostics.",
    )
    parser.add_argument(
        "--drop-episode-json",
        type=Path,
        default=None,
        help="Optional drop detector episode summary JSON payload.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Optional output JSON path. When empty-like, the report is only printed.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"out must be a file path, got directory: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object in {path}, got {type(payload).__name__}")
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"expected JSON object in {path}:{line_number}, got {type(payload).__name__}"
            )
        rows.append(dict(payload))
    return rows


def _deep_get(payload: Mapping[str, Any], dotted_path: str) -> object:
    current: object = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _mapping_containers(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    containers = [record]
    for key in (
        "intermediate_signals",
        "analysis_only",
        "privileged",
        "diagnostic",
        "runtime_trace",
        "debug_probe",
    ):
        candidate = record.get(key)
        if isinstance(candidate, Mapping):
            containers.append(candidate)
    return containers


def resolve_field_with_source(
    record: Mapping[str, Any], *candidate_paths: str
) -> tuple[object | None, str | None]:
    containers = _mapping_containers(record)
    for candidate in candidate_paths:
        for container in containers:
            if candidate in container:
                return container[candidate], candidate
            resolved = _deep_get(container, candidate)
            if resolved is not _MISSING:
                return resolved, candidate
    return None, None


def resolve_field(record: Mapping[str, Any], *candidate_paths: str) -> object | None:
    value, _ = resolve_field_with_source(record, *candidate_paths)
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return None


def _float_sequence(value: object) -> list[float] | None:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return None
    out: list[float] = []
    for item in value:
        numeric = _optional_float(item)
        if numeric is None:
            return None
        out.append(float(numeric))
    return out


def normalize_phase(value: object) -> str | None:
    text = _optional_string(value)
    if text is None:
        return None
    return PHASE_ALIASES.get(text.upper())


def normalize_semantic_state(value: object) -> str | None:
    text = _optional_string(value)
    if text is None:
        return None
    normalized = text.upper()
    return normalized if normalized in PHASE_ALIASES else None


def normalize_failure_stage_guess(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        label = _optional_string(value.get("label"))
        if label is None:
            return None
        payload = dict(value)
    else:
        label = _optional_string(value)
        if label is None:
            return None
        payload = {"label": label}
    normalized_label = label.strip()
    return {
        "label": normalized_label,
        "mapped_stage_level_reason": FAILURE_STAGE_GUESS_TO_STAGE_REASON.get(
            normalized_label
        ),
        "payload": payload,
    }


def _distance_from_rel_pose(
    record: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    value, source = resolve_field_with_source(
        record,
        "privileged.apple_to_plate_rel_pose",
        "apple_to_plate_rel_pose",
    )
    rel_pose = _float_sequence(value)
    if rel_pose is None or len(rel_pose) < 3:
        return None, None
    return math.sqrt(sum(float(component) ** 2 for component in rel_pose[:3])), source


def _plate_distance(record: Mapping[str, Any]) -> tuple[float | None, str | None]:
    rel_pose_distance, rel_pose_source = _distance_from_rel_pose(record)
    if rel_pose_distance is not None:
        return rel_pose_distance, rel_pose_source
    value, source = resolve_field_with_source(record, "apple_to_plate_l2")
    return _optional_float(value), source


def _step_index(record: Mapping[str, Any], fallback: int) -> int:
    for field_name in ("t", "outer_step"):
        numeric = _optional_int(record.get(field_name))
        if numeric is not None:
            return int(numeric)
    return int(fallback)


def _semantic_stage(semantic_state: str | None) -> str | None:
    if semantic_state is None:
        return None
    return PHASE_ALIASES.get(semantic_state)


def _normalize_step_record(
    record: Mapping[str, Any], fallback_index: int
) -> dict[str, Any]:
    phase_value, phase_source = resolve_field_with_source(
        record,
        "policy_condition.phase",
        "phase",
        "canonical_phase",
    )
    semantic_value, semantic_source = resolve_field_with_source(
        record,
        "semantic_state",
        "analysis_only.semantic_state",
    )
    apple_in_hand_value, apple_in_hand_source = resolve_field_with_source(
        record,
        "apple_in_hand",
        "privileged.apple_in_hand",
    )
    drop_value, drop_source = resolve_field_with_source(
        record,
        "drop_during_transport",
        "analysis_only.drop_during_transport",
        "diagnostic.drop_during_transport",
    )
    hand_distance_value, hand_distance_source = resolve_field_with_source(
        record,
        "apple_to_right_eef_l2",
    )
    apple_height_value, apple_height_source = resolve_field_with_source(
        record,
        "apple_height_z",
    )
    plate_distance, plate_distance_source = _plate_distance(record)
    success_step_value, success_step_source = resolve_field_with_source(
        record, "success_step"
    )

    semantic_state = normalize_semantic_state(semantic_value)
    semantic_stage = _semantic_stage(semantic_state)
    phase = normalize_phase(phase_value)
    phase_from_direct_signal = phase is not None
    if phase is None:
        phase = semantic_stage
        phase_source = semantic_source

    return {
        "step_index": _step_index(record, fallback=fallback_index),
        "phase": phase,
        "phase_source": phase_source,
        "phase_from_direct_signal": bool(phase_from_direct_signal),
        "semantic_state": semantic_state,
        "semantic_state_source": semantic_source,
        "apple_in_hand": _optional_bool(apple_in_hand_value),
        "apple_in_hand_source": apple_in_hand_source,
        "drop_during_transport": _optional_bool(drop_value),
        "drop_during_transport_source": drop_source,
        "apple_to_right_eef_l2": _optional_float(hand_distance_value),
        "apple_to_right_eef_l2_source": hand_distance_source,
        "apple_height_z": _optional_float(apple_height_value),
        "apple_height_z_source": apple_height_source,
        "apple_to_plate_distance": plate_distance,
        "apple_to_plate_distance_source": plate_distance_source,
        "success_step": _optional_bool(success_step_value),
        "success_step_source": success_step_source,
        "raw": dict(record),
    }


def _event_payload(
    *,
    event_name: str,
    stage_name: str,
    step_index: int | None,
    source: str,
    confidence: float,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_name": event_name,
        "stage_name": stage_name,
        "step_index": step_index,
        "source": source,
        "confidence": float(confidence),
    }
    if details:
        payload["details"] = dict(details)
    return payload


def _append_event_once(
    events: list[dict[str, Any]],
    seen_event_names: set[str],
    payload: dict[str, Any] | None,
) -> None:
    if payload is None:
        return
    event_name = str(payload["event_name"])
    if event_name in seen_event_names:
        return
    seen_event_names.add(event_name)
    events.append(payload)


def _first_available_value(
    normalized_steps: Sequence[Mapping[str, Any]], field_name: str
) -> float | None:
    for row in normalized_steps:
        value = _optional_float(row.get(field_name))
        if value is not None:
            return float(value)
    return None


def _plate_progress_improvement(
    normalized_steps: Sequence[Mapping[str, Any]],
) -> float | None:
    initial_distance = _first_available_value(
        normalized_steps, "apple_to_plate_distance"
    )
    if initial_distance is None:
        return None
    distances = [
        float(value)
        for row in normalized_steps
        for value in [_optional_float(row.get("apple_to_plate_distance"))]
        if value is not None
    ]
    if not distances:
        return None
    return float(initial_distance - min(distances))


def _initial_height_and_max_lift(
    normalized_steps: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float | None]:
    initial_height = _first_available_value(normalized_steps, "apple_height_z")
    if initial_height is None:
        return None, None
    heights = [
        float(value)
        for row in normalized_steps
        for value in [_optional_float(row.get("apple_height_z"))]
        if value is not None
    ]
    if not heights:
        return initial_height, None
    return float(initial_height), float(max(heights) - initial_height)


def _missing_signals(
    normalized_steps: Sequence[Mapping[str, Any]],
    *,
    failure_stage_guess: Mapping[str, Any] | None,
    runtime_trace: Mapping[str, Any] | None,
    drop_episode_summary: Mapping[str, Any] | None,
) -> list[str]:
    missing: list[str] = []
    if not any(
        row.get("phase_from_direct_signal") is True
        or row.get("semantic_state") is not None
        for row in normalized_steps
    ):
        missing.append("policy_condition.phase_or_semantic_state")
    if not any(row.get("apple_in_hand") is not None for row in normalized_steps):
        missing.append("apple_in_hand")
    has_drop_signal = any(
        row.get("drop_during_transport") is not None for row in normalized_steps
    ) or isinstance(drop_episode_summary, Mapping)
    if not has_drop_signal:
        missing.append("drop_during_transport")
    if not any(
        row.get("apple_to_plate_distance") is not None for row in normalized_steps
    ):
        missing.append("apple_to_plate_rel_pose_or_apple_to_plate_l2")
    if not any(row.get("apple_height_z") is not None for row in normalized_steps):
        missing.append("apple_height_z")
    if not any(
        row.get("apple_to_right_eef_l2") is not None for row in normalized_steps
    ):
        missing.append("apple_to_right_eef_l2")
    if failure_stage_guess is None:
        missing.append("failure_stage_guess")
    if runtime_trace is None:
        missing.append("runtime_trace")
    return missing


def derive_probe_stage_events(
    *,
    episode_record: Mapping[str, Any] | None = None,
    step_records: Sequence[Mapping[str, Any]] | None = None,
    runtime_trace: Mapping[str, Any] | None = None,
    drop_episode_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    episode_payload = (
        dict(episode_record) if isinstance(episode_record, Mapping) else {}
    )
    raw_steps = [
        dict(step) for step in (step_records or []) if isinstance(step, Mapping)
    ]
    normalized_steps = [
        _normalize_step_record(record, fallback_index=index)
        for index, record in enumerate(raw_steps)
    ]
    normalized_steps.sort(key=lambda row: int(row["step_index"]))

    failure_stage_guess = normalize_failure_stage_guess(
        episode_payload.get("failure_stage_guess")
    )
    missing_signals = _missing_signals(
        normalized_steps,
        failure_stage_guess=failure_stage_guess,
        runtime_trace=runtime_trace,
        drop_episode_summary=drop_episode_summary,
    )

    events: list[dict[str, Any]] = []
    seen_event_names: set[str] = set()

    initial_height, max_lift = _initial_height_and_max_lift(normalized_steps)
    plate_progress_improvement = _plate_progress_improvement(normalized_steps)
    had_in_hand = False
    near_plate_seen = False
    place_seen = False

    for row in normalized_steps:
        phase = _optional_string(row.get("phase"))
        semantic_state = _optional_string(row.get("semantic_state"))
        apple_in_hand = row.get("apple_in_hand")
        drop_during_transport = row.get("drop_during_transport")
        hand_distance = _optional_float(row.get("apple_to_right_eef_l2"))
        plate_distance = _optional_float(row.get("apple_to_plate_distance"))
        step_index = int(row["step_index"])

        direct_approach = phase in set(SEMANTIC_STAGE_ORDER)
        if direct_approach:
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="approach_observed",
                    stage_name="APPROACH",
                    step_index=step_index,
                    source=str(row.get("phase_source") or "policy_condition.phase"),
                    confidence=DIRECT_SIGNAL_CONFIDENCE,
                    details={"phase": phase},
                ),
            )
        elif semantic_state == "APPLE_VISIBLE_APPROACH":
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="approach_observed",
                    stage_name="APPROACH",
                    step_index=step_index,
                    source=str(row.get("semantic_state_source") or "semantic_state"),
                    confidence=DIRECT_SIGNAL_CONFIDENCE,
                    details={"semantic_state": semantic_state},
                ),
            )
        elif (
            hand_distance is not None and hand_distance <= APPROACH_DISTANCE_THRESHOLD_M
        ):
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="approach_observed",
                    stage_name="APPROACH",
                    step_index=step_index,
                    source=str(
                        row.get("apple_to_right_eef_l2_source")
                        or "apple_to_right_eef_l2"
                    ),
                    confidence=GEOMETRY_SIGNAL_CONFIDENCE,
                    details={"apple_to_right_eef_l2": float(hand_distance)},
                ),
            )

        if phase in {"GRASP", "VERIFY_HOLD", "TRANSPORT", "PLACE"}:
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="grasp_observed",
                    stage_name="GRASP",
                    step_index=step_index,
                    source=str(row.get("phase_source") or "policy_condition.phase"),
                    confidence=DIRECT_SIGNAL_CONFIDENCE,
                    details={"phase": phase},
                ),
            )
        elif semantic_state in {"GRASPING", "VERIFYING_HOLD", "RECOVERY_REGRASP"}:
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="grasp_observed",
                    stage_name="GRASP",
                    step_index=step_index,
                    source=str(row.get("semantic_state_source") or "semantic_state"),
                    confidence=DIRECT_SIGNAL_CONFIDENCE,
                    details={"semantic_state": semantic_state},
                ),
            )
        elif apple_in_hand is True:
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="grasp_observed",
                    stage_name="GRASP",
                    step_index=step_index,
                    source=str(row.get("apple_in_hand_source") or "apple_in_hand"),
                    confidence=DIRECT_SIGNAL_CONFIDENCE,
                    details={"apple_in_hand": True},
                ),
            )
        elif max_lift is not None and max_lift >= APPLE_LIFT_THRESHOLD_M:
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="grasp_observed",
                    stage_name="GRASP",
                    step_index=step_index,
                    source=str(row.get("apple_height_z_source") or "apple_height_z"),
                    confidence=GEOMETRY_SIGNAL_CONFIDENCE,
                    details={
                        "initial_height_z": initial_height,
                        "max_lift_z": float(max_lift),
                    },
                ),
            )

        if phase == "VERIFY_HOLD" or semantic_state == "VERIFYING_HOLD":
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="verify_hold_observed",
                    stage_name="VERIFY_HOLD",
                    step_index=step_index,
                    source=str(
                        row.get("phase_source")
                        or row.get("semantic_state_source")
                        or "verify_hold"
                    ),
                    confidence=DIRECT_SIGNAL_CONFIDENCE,
                    details={
                        "phase": phase,
                        "semantic_state": semantic_state,
                    },
                ),
            )

        if phase in {"TRANSPORT", "PLACE"} or semantic_state == "TRANSPORTING":
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="transport_observed",
                    stage_name="TRANSPORT",
                    step_index=step_index,
                    source=str(
                        row.get("phase_source")
                        or row.get("semantic_state_source")
                        or "policy_condition.phase"
                    ),
                    confidence=DIRECT_SIGNAL_CONFIDENCE,
                    details={
                        "phase": phase,
                        "semantic_state": semantic_state,
                    },
                ),
            )
        elif (
            apple_in_hand is True
            and plate_progress_improvement is not None
            and plate_progress_improvement >= PLATE_PROGRESS_THRESHOLD_M
        ):
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="transport_observed",
                    stage_name="TRANSPORT",
                    step_index=step_index,
                    source=str(
                        row.get("apple_to_plate_distance_source")
                        or row.get("apple_in_hand_source")
                        or "apple_to_plate_rel_pose_or_apple_to_plate_l2"
                    ),
                    confidence=GEOMETRY_SIGNAL_CONFIDENCE,
                    details={
                        "plate_progress_improvement_m": float(
                            plate_progress_improvement
                        ),
                        "apple_in_hand": True,
                    },
                ),
            )

        if phase == "PLACE" or semantic_state == "PLACING":
            near_plate_seen = True
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="near_plate_observed",
                    stage_name="PLACE",
                    step_index=step_index,
                    source=str(
                        row.get("phase_source")
                        or row.get("semantic_state_source")
                        or "policy_condition.phase"
                    ),
                    confidence=DIRECT_SIGNAL_CONFIDENCE,
                    details={"phase": phase, "semantic_state": semantic_state},
                ),
            )
            place_seen = True
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="place_observed",
                    stage_name="PLACE",
                    step_index=step_index,
                    source=str(
                        row.get("phase_source")
                        or row.get("semantic_state_source")
                        or "policy_condition.phase"
                    ),
                    confidence=DIRECT_SIGNAL_CONFIDENCE,
                    details={"phase": phase, "semantic_state": semantic_state},
                ),
            )
        elif (
            plate_distance is not None
            and plate_distance <= NEAR_PLATE_DISTANCE_THRESHOLD_M
        ):
            near_plate_seen = True
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="near_plate_observed",
                    stage_name="PLACE",
                    step_index=step_index,
                    source=str(
                        row.get("apple_to_plate_distance_source")
                        or "apple_to_plate_rel_pose_or_apple_to_plate_l2"
                    ),
                    confidence=GEOMETRY_SIGNAL_CONFIDENCE,
                    details={"apple_to_plate_distance": float(plate_distance)},
                ),
            )

        if apple_in_hand is True:
            had_in_hand = True
        if (
            had_in_hand
            and near_plate_seen
            and apple_in_hand is False
            and not place_seen
        ):
            place_seen = True
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="place_observed",
                    stage_name="PLACE",
                    step_index=step_index,
                    source="apple_in_hand_transition_near_plate",
                    confidence=GEOMETRY_SIGNAL_CONFIDENCE,
                    details={
                        "apple_in_hand": False,
                        "near_plate_seen": True,
                    },
                ),
            )

        if drop_during_transport is True:
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="drop_during_transport_observed",
                    stage_name="TRANSPORT",
                    step_index=step_index,
                    source=str(
                        row.get("drop_during_transport_source")
                        or "drop_during_transport"
                    ),
                    confidence=DIRECT_SIGNAL_CONFIDENCE,
                    details={"drop_during_transport": True},
                ),
            )
        elif had_in_hand and apple_in_hand is False and phase == "TRANSPORT":
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="drop_during_transport_observed",
                    stage_name="TRANSPORT",
                    step_index=step_index,
                    source="apple_in_hand_transition_during_transport",
                    confidence=DIRECT_SIGNAL_CONFIDENCE,
                    details={"phase": phase, "apple_in_hand": False},
                ),
            )

    if isinstance(drop_episode_summary, Mapping) and bool(
        drop_episode_summary.get("has_transport_drop")
    ):
        _append_event_once(
            events,
            seen_event_names,
            _event_payload(
                event_name="drop_during_transport_observed",
                stage_name="TRANSPORT",
                step_index=_optional_int(
                    drop_episode_summary.get("first_transport_drop_t")
                ),
                source="drop_episode_summary.has_transport_drop",
                confidence=DIRECT_SIGNAL_CONFIDENCE,
                details={
                    "has_transport_drop": True,
                    "first_transport_drop_t": drop_episode_summary.get(
                        "first_transport_drop_t"
                    ),
                },
            ),
        )

    if failure_stage_guess is not None and not events:
        mapped = failure_stage_guess.get("mapped_stage_level_reason")
        label = str(failure_stage_guess["label"])
        if isinstance(mapped, str):
            _append_event_once(
                events,
                seen_event_names,
                _event_payload(
                    event_name="failure_stage_guess_fallback",
                    stage_name=label,
                    step_index=None,
                    source="failure_stage_guess",
                    confidence=HEURISTIC_SIGNAL_CONFIDENCE,
                    details={
                        "label": label,
                        "mapped_stage_level_reason": mapped,
                    },
                ),
            )

    observed_stage_sequence: list[str] = []
    for event in events:
        stage_name = str(event["stage_name"])
        if stage_name not in observed_stage_sequence:
            observed_stage_sequence.append(stage_name)

    direct_event_count = sum(
        1 for event in events if float(event["confidence"]) >= DIRECT_SIGNAL_CONFIDENCE
    )
    geometry_event_count = sum(
        1
        for event in events
        if DIRECT_SIGNAL_CONFIDENCE
        > float(event["confidence"])
        >= GEOMETRY_SIGNAL_CONFIDENCE
    )
    if direct_event_count > 0:
        confidence = DIRECT_SIGNAL_CONFIDENCE
    elif geometry_event_count > 0:
        confidence = GEOMETRY_SIGNAL_CONFIDENCE
    elif failure_stage_guess is not None:
        confidence = HEURISTIC_SIGNAL_CONFIDENCE
    else:
        confidence = LOW_SIGNAL_CONFIDENCE
    confidence = max(
        LOW_SIGNAL_CONFIDENCE, float(confidence - 0.03 * min(len(missing_signals), 5))
    )

    progress_flags = {
        "approach_seen": "approach_observed" in seen_event_names,
        "grasp_seen": "grasp_observed" in seen_event_names,
        "verify_hold_seen": "verify_hold_observed" in seen_event_names,
        "transport_seen": "transport_observed" in seen_event_names,
        "near_plate_seen": "near_plate_observed" in seen_event_names,
        "place_seen": "place_observed" in seen_event_names,
        "drop_during_transport_seen": "drop_during_transport_observed"
        in seen_event_names,
        "success_step_seen": any(
            row.get("success_step") is True for row in normalized_steps
        ),
    }

    geometry_metrics = {
        "initial_apple_height_z": initial_height,
        "max_apple_lift_z": max_lift,
        "min_apple_to_right_eef_l2": min(
            (
                float(row["apple_to_right_eef_l2"])
                for row in normalized_steps
                if _optional_float(row.get("apple_to_right_eef_l2")) is not None
            ),
            default=None,
        ),
        "min_apple_to_plate_distance": min(
            (
                float(row["apple_to_plate_distance"])
                for row in normalized_steps
                if _optional_float(row.get("apple_to_plate_distance")) is not None
            ),
            default=None,
        ),
        "plate_progress_improvement_m": plate_progress_improvement,
    }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "step_count": int(len(normalized_steps)),
        "event_count": int(len(events)),
        "stage_events": events,
        "observed_stage_sequence": observed_stage_sequence,
        "progress_flags": progress_flags,
        "geometry_metrics": geometry_metrics,
        "normalized_failure_stage_guess": failure_stage_guess,
        "missing_signals": missing_signals,
        "confidence": float(round(confidence, 4)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        episode_record = _read_json(args.episode_json) if args.episode_json else None
        step_records = _read_jsonl(args.steps_jsonl) if args.steps_jsonl else None
        runtime_trace = (
            _read_json(args.runtime_trace_json) if args.runtime_trace_json else None
        )
        drop_episode_summary = (
            _read_json(args.drop_episode_json) if args.drop_episode_json else None
        )
        payload = derive_probe_stage_events(
            episode_record=episode_record,
            step_records=step_records,
            runtime_trace=runtime_trace,
            drop_episode_summary=drop_episode_summary,
        )
        output_path = _validate_output_path(args.out)
        if output_path is not None:
            payload["output_path"] = str(output_path)
            _ = _write_json(output_path, payload)
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
