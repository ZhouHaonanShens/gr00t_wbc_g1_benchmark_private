from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import derive_probe_stage_events
from work.recap.scripts import gr00t_same_checkpoint_triplet_eval
from work.recap.transitions import transition_schema


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


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


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n", ""}:
            return False
    return None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


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


def _plate_distance(record: Mapping[str, Any]) -> float | None:
    rel_pose = _float_sequence(
        _resolve_field(
            record,
            "privileged.apple_to_plate_rel_pose",
            "apple_to_plate_rel_pose",
        )
    )
    if rel_pose is not None and len(rel_pose) >= 3:
        return math.sqrt(sum(float(component) ** 2 for component in rel_pose[:3]))
    return _optional_float(_resolve_field(record, "apple_to_plate_l2"))


def _normalized_stage(record: Mapping[str, Any]) -> str:
    phase = transition_schema.normalize_phase(
        _resolve_field(record, "policy_condition.phase", "phase"),
        field_name="phase",
    )
    semantic_state = transition_schema.normalize_semantic_state(
        _resolve_field(record, "semantic_state", "analysis_only.semantic_state"),
        field_name="semantic_state",
    )
    normalized = None
    if phase is not None:
        normalized = derive_probe_stage_events.normalize_phase(phase)
    if normalized is None and semantic_state is not None:
        normalized_semantic = derive_probe_stage_events.normalize_semantic_state(
            semantic_state
        )
        if normalized_semantic is not None:
            normalized = derive_probe_stage_events.PHASE_ALIASES.get(
                normalized_semantic
            )
    if normalized is None:
        raise ValueError(
            "transition labeling requires phase or semantic_state from the frozen live vocab"
        )
    stage = transition_schema.normalize_stage(normalized, field_name="normalized_stage")
    if stage is None:
        raise ValueError("normalized_stage must not be null after normalization")
    return stage


def _transition_label_for_record(
    *,
    record: Mapping[str, Any],
    previous_record: Mapping[str, Any] | None,
) -> str:
    normalized_stage = _normalized_stage(record)
    apple_in_hand = _optional_bool(
        _resolve_field(record, "privileged.apple_in_hand", "apple_in_hand")
    )
    previous_in_hand = _optional_bool(
        None
        if previous_record is None
        else _resolve_field(
            previous_record,
            "privileged.apple_in_hand",
            "apple_in_hand",
        )
    )
    drop_during_transport = _optional_bool(
        _resolve_field(
            record,
            "drop_during_transport",
            "analysis_only.drop_during_transport",
            "diagnostic.drop_during_transport",
        )
    )
    plate_distance = _plate_distance(record)
    near_plate = bool(
        plate_distance is not None
        and plate_distance
        <= float(derive_probe_stage_events.NEAR_PLATE_DISTANCE_THRESHOLD_M)
    )

    if drop_during_transport is True:
        return "drop"
    if previous_in_hand is True and apple_in_hand is False:
        if normalized_stage in {"PLACE", "TRANSPORT"} and near_plate:
            return "release"
        return "drop"
    if normalized_stage == "SEARCH":
        return "approach"
    if normalized_stage == "APPROACH":
        return "approach"
    if normalized_stage == "GRASP":
        return "grasp"
    if normalized_stage == "VERIFY_HOLD":
        return "stable_grasp"
    if normalized_stage == "TRANSPORT":
        if near_plate:
            return "near_plate"
        return "transport"
    if normalized_stage == "PLACE":
        if apple_in_hand is False and previous_in_hand is True:
            return "release"
        return "place"
    raise ValueError(f"unsupported normalized stage: {normalized_stage!r}")


def build_transition_sidecar_row(
    *,
    record: Mapping[str, Any],
    previous_record: Mapping[str, Any] | None = None,
    runtime_trace: Mapping[str, Any] | None = None,
    execution_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    episode_id, t = transition_schema.record_join_key(record)
    phase = transition_schema.normalize_phase(
        _resolve_field(record, "policy_condition.phase", "phase"),
        field_name="phase",
    )
    mode = transition_schema.normalize_mode(
        _resolve_field(record, "policy_condition.mode", "mode"),
        field_name="mode",
    )
    semantic_state = transition_schema.normalize_semantic_state(
        _resolve_field(record, "semantic_state", "analysis_only.semantic_state"),
        field_name="semantic_state",
    )
    normalized_stage = _normalized_stage(record)
    transition_label = _transition_label_for_record(
        record=record,
        previous_record=previous_record,
    )

    runtime_trace_payload = (
        dict(_as_mapping(runtime_trace, field_name="runtime_trace"))
        if isinstance(runtime_trace, Mapping)
        else {}
    )
    execution_payload = (
        dict(_as_mapping(execution_audit, field_name="execution_audit"))
        if isinstance(execution_audit, Mapping)
        else {}
    )
    controller_output_available = _optional_bool(
        runtime_trace_payload.get("controller_output_available")
    )
    terminal_stage_used = _optional_string(
        runtime_trace_payload.get("terminal_stage_used")
    )
    if terminal_stage_used is None and controller_output_available is not None:
        terminal_stage_used = (
            gr00t_same_checkpoint_triplet_eval.resolve_triplet_terminal_stage(
                controller_output_available=controller_output_available,
            )
        )

    row = {
        "schema_version": transition_schema.RTC_TRANSITION_ROW_SCHEMA_VERSION,
        "artifact_kind": transition_schema.RTC_TRANSITION_ROW_ARTIFACT_KIND,
        "episode_id": episode_id,
        "t": int(t),
        "transition_label": transition_label,
        "phase": phase,
        "mode": mode,
        "semantic_state": semantic_state,
        "normalized_stage": normalized_stage,
        "controller_output_available": controller_output_available,
        "controller_output_unavailable_reason": _optional_string(
            runtime_trace_payload.get("controller_output_unavailable_reason")
        ),
        "terminal_stage_used": terminal_stage_used,
        "execution_surface_verdict": _optional_string(execution_payload.get("verdict")),
        "authority_boundary": transition_schema.RTC_AUTHORITY_BOUNDARY,
        "mainline_authority": transition_schema.RTC_MAINLINE_AUTHORITY,
        "diagnostic_only": transition_schema.RTC_DIAGNOSTIC_ONLY,
    }
    return transition_schema.validate_transition_sidecar_row(row)


def build_transition_sidecar_rows(
    *,
    records: Sequence[Mapping[str, Any]],
    runtime_trace_by_episode: Mapping[str, Mapping[str, Any]] | None = None,
    execution_audit_by_episode: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    previous_by_episode: dict[str, Mapping[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for record in records:
        episode_id, _ = transition_schema.record_join_key(record)
        row = build_transition_sidecar_row(
            record=record,
            previous_record=previous_by_episode.get(episode_id),
            runtime_trace=(runtime_trace_by_episode or {}).get(episode_id),
            execution_audit=(execution_audit_by_episode or {}).get(episode_id),
        )
        rows.append(row)
        previous_by_episode[episode_id] = dict(record)
    return transition_schema.validate_transition_sidecar_rows(rows)


def summarize_transition_sidecar(
    rows: Sequence[Mapping[str, Any]],
    *,
    path: str | Path | None = None,
    expected_join_keys: Sequence[Sequence[object]] | None = None,
) -> dict[str, Any]:
    normalized_rows = transition_schema.validate_transition_sidecar_rows(
        rows,
        expected_join_keys=expected_join_keys,
    )
    return transition_schema.build_transition_sidecar_summary(
        status=transition_schema.RTC_TRANSITION_STATUS_AVAILABLE,
        reason_code="ok",
        path=None if path is None else str(path),
        episodes_covered=len({str(row["episode_id"]) for row in normalized_rows}),
        steps_labeled=len(normalized_rows),
        label_counts=transition_schema.build_transition_label_counts(normalized_rows),
    )


def build_transition_sidecar_context(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    path: str | Path | None = None,
    expected_join_keys: Sequence[Sequence[object]] | None = None,
) -> dict[str, Any]:
    if rows is None:
        return transition_schema.build_transition_sidecar_summary(
            status=transition_schema.RTC_TRANSITION_STATUS_NOT_AVAILABLE,
            reason_code="not_available",
            path=None if path is None else str(path),
            episodes_covered=0,
            steps_labeled=0,
            label_counts=transition_schema.build_transition_label_counts(),
        )
    try:
        return summarize_transition_sidecar(
            rows,
            path=path,
            expected_join_keys=expected_join_keys,
        )
    except (TypeError, ValueError) as exc:
        return transition_schema.build_transition_sidecar_summary(
            status=transition_schema.RTC_TRANSITION_STATUS_INVALID,
            reason_code=f"invalid:{_exception_message(exc)}",
            path=None if path is None else str(path),
            episodes_covered=0,
            steps_labeled=0,
            label_counts=transition_schema.build_transition_label_counts(),
        )


def validate_transition_context_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return transition_schema.validate_transition_sidecar_summary(payload)


__all__ = [
    "build_transition_sidecar_context",
    "build_transition_sidecar_row",
    "build_transition_sidecar_rows",
    "summarize_transition_sidecar",
    "validate_transition_context_summary",
]
