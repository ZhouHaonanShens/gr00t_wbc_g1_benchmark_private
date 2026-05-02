from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


RTC_TRANSITION_ROW_SCHEMA_VERSION = "rtc_transition_sidecar_row_v1"
RTC_TRANSITION_ROW_ARTIFACT_KIND = "rtc_transition_sidecar_row"
RTC_TRANSITION_SUMMARY_SCHEMA_VERSION = "rtc_transition_sidecar_summary_v1"
RTC_TRANSITION_SUMMARY_ARTIFACT_KIND = "rtc_transition_sidecar_summary"
RTC_TRANSITION_LABEL_VOCAB_VERSION = "rtc_transition_label_vocab_v1"

RTC_TRANSITION_STATUS_AVAILABLE = "available"
RTC_TRANSITION_STATUS_NOT_AVAILABLE = "not_available"
RTC_TRANSITION_STATUS_INVALID = "invalid"
RTC_TRANSITION_STATUS_VALUES: tuple[str, ...] = (
    RTC_TRANSITION_STATUS_AVAILABLE,
    RTC_TRANSITION_STATUS_NOT_AVAILABLE,
    RTC_TRANSITION_STATUS_INVALID,
)

RTC_AUTHORITY_BOUNDARY = "optional_runtime_context_only"
RTC_MAINLINE_AUTHORITY = False
RTC_DIAGNOSTIC_ONLY = True

TRANSITION_LABEL_VOCAB: tuple[str, ...] = (
    "approach",
    "grasp",
    "stable_grasp",
    "transport",
    "near_plate",
    "place",
    "release",
    "drop",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import audit_g1_execution_surface
from work.recap.scripts import derive_probe_stage_events
from work.recap.scripts import gr00t_same_checkpoint_triplet_eval
from work.recap.scripts import state_conditioned_bucket_a_sidecar
from work.recap.scripts import state_conditioned_contract_gate


PHASE_VOCAB: tuple[str, ...] = tuple(state_conditioned_contract_gate.PHASE_VOCAB)
MODE_VOCAB: tuple[str, ...] = tuple(state_conditioned_contract_gate.MODE_VOCAB)
SEMANTIC_STATE_VALUES: tuple[str, ...] = tuple(
    state_conditioned_bucket_a_sidecar.SEMANTIC_STATE_VALUES
)
STAGE_VOCAB: tuple[str, ...] = (
    "SEARCH",
    *derive_probe_stage_events.SEMANTIC_STAGE_ORDER,
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


def _as_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _as_non_empty_string(value, field_name=field_name)


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    return _as_bool(value, field_name=field_name)


def _as_list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


def _as_sequence(value: object, *, field_name: str) -> list[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence, got {type(value).__name__}")
    return [item for item in value]


def build_transition_label_counts(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    counts = {label: 0 for label in TRANSITION_LABEL_VOCAB}
    if rows is None:
        return counts
    for row in rows:
        label = normalize_transition_label(
            row.get("transition_label"), field_name="transition_label"
        )
        counts[label] += 1
    return counts


def normalize_transition_label(
    value: object,
    *,
    field_name: str = "transition_label",
) -> str:
    normalized = _as_non_empty_string(value, field_name=field_name).strip().lower()
    if normalized not in TRANSITION_LABEL_VOCAB:
        raise ValueError(f"{field_name} must be one of {TRANSITION_LABEL_VOCAB!r}")
    return normalized


def normalize_phase(value: object, *, field_name: str = "phase") -> str | None:
    if value is None:
        return None
    normalized = _as_non_empty_string(value, field_name=field_name).upper()
    if normalized not in PHASE_VOCAB:
        raise ValueError(f"{field_name} must be one of {PHASE_VOCAB!r}")
    return normalized


def normalize_mode(value: object, *, field_name: str = "mode") -> str | None:
    if value is None:
        return None
    normalized = _as_non_empty_string(value, field_name=field_name).upper()
    if normalized not in MODE_VOCAB:
        raise ValueError(f"{field_name} must be one of {MODE_VOCAB!r}")
    return normalized


def normalize_semantic_state(
    value: object,
    *,
    field_name: str = "semantic_state",
) -> str | None:
    if value is None:
        return None
    normalized = _as_non_empty_string(value, field_name=field_name).upper()
    if normalized not in SEMANTIC_STATE_VALUES:
        raise ValueError(f"{field_name} must be one of {SEMANTIC_STATE_VALUES!r}")
    return normalized


def normalize_stage(
    value: object,
    *,
    field_name: str = "normalized_stage",
) -> str | None:
    if value is None:
        return None
    normalized = derive_probe_stage_events.normalize_phase(value)
    if normalized is None:
        semantic_state = derive_probe_stage_events.normalize_semantic_state(value)
        if semantic_state is not None:
            normalized = derive_probe_stage_events.PHASE_ALIASES.get(semantic_state)
    if normalized is None:
        raise ValueError(f"{field_name} could not be normalized from {value!r}")
    if normalized not in STAGE_VOCAB:
        raise ValueError(f"{field_name} must be one of {STAGE_VOCAB!r}")
    return normalized


def record_join_key(record: Mapping[str, Any]) -> tuple[str, int]:
    episode_id = _as_non_empty_string(
        record.get("episode_id"), field_name="transition_row.episode_id"
    )
    t = _as_int(record.get("t"), field_name=f"transition_row[{episode_id!r}].t")
    return episode_id, int(t)


def validate_transition_sidecar_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(_as_mapping(row, field_name="transition_sidecar_row"))
    if payload.get("schema_version") != RTC_TRANSITION_ROW_SCHEMA_VERSION:
        raise ValueError(
            "transition_sidecar_row.schema_version must equal "
            + repr(RTC_TRANSITION_ROW_SCHEMA_VERSION)
        )
    if payload.get("artifact_kind") != RTC_TRANSITION_ROW_ARTIFACT_KIND:
        raise ValueError(
            "transition_sidecar_row.artifact_kind must equal "
            + repr(RTC_TRANSITION_ROW_ARTIFACT_KIND)
        )

    episode_id, t = record_join_key(payload)
    normalized: dict[str, Any] = {
        "schema_version": RTC_TRANSITION_ROW_SCHEMA_VERSION,
        "artifact_kind": RTC_TRANSITION_ROW_ARTIFACT_KIND,
        "episode_id": episode_id,
        "t": int(t),
        "transition_label": normalize_transition_label(payload.get("transition_label")),
        "phase": normalize_phase(payload.get("phase"), field_name="phase"),
        "mode": normalize_mode(payload.get("mode"), field_name="mode"),
        "semantic_state": normalize_semantic_state(
            payload.get("semantic_state"), field_name="semantic_state"
        ),
        "normalized_stage": normalize_stage(
            payload.get("normalized_stage"), field_name="normalized_stage"
        ),
        "controller_output_available": _as_optional_bool(
            payload.get("controller_output_available"),
            field_name="controller_output_available",
        ),
        "controller_output_unavailable_reason": _as_optional_string(
            payload.get("controller_output_unavailable_reason"),
            field_name="controller_output_unavailable_reason",
        ),
        "terminal_stage_used": _as_optional_string(
            payload.get("terminal_stage_used"),
            field_name="terminal_stage_used",
        ),
        "execution_surface_verdict": _as_optional_string(
            payload.get("execution_surface_verdict"),
            field_name="execution_surface_verdict",
        ),
        "authority_boundary": _as_non_empty_string(
            payload.get("authority_boundary"), field_name="authority_boundary"
        ),
        "mainline_authority": _as_bool(
            payload.get("mainline_authority"), field_name="mainline_authority"
        ),
        "diagnostic_only": _as_bool(
            payload.get("diagnostic_only"), field_name="diagnostic_only"
        ),
    }
    if normalized["authority_boundary"] != RTC_AUTHORITY_BOUNDARY:
        raise ValueError(f"authority_boundary must equal {RTC_AUTHORITY_BOUNDARY!r}")
    if normalized["mainline_authority"] is not RTC_MAINLINE_AUTHORITY:
        raise ValueError("mainline_authority must remain false for RTC sidecars")
    if normalized["diagnostic_only"] is not RTC_DIAGNOSTIC_ONLY:
        raise ValueError("diagnostic_only must remain true for RTC sidecars")

    execution_surface_verdict = normalized["execution_surface_verdict"]
    if execution_surface_verdict is not None:
        if execution_surface_verdict not in audit_g1_execution_surface.VERDICT_ENUM:
            raise ValueError(
                "execution_surface_verdict must be one of "
                + f"{audit_g1_execution_surface.VERDICT_ENUM!r}"
            )

    terminal_stage_used = normalized["terminal_stage_used"]
    if terminal_stage_used is not None and terminal_stage_used not in {
        *gr00t_same_checkpoint_triplet_eval.MACHINE_CHECKPOINT_ORDER,
    }:
        raise ValueError(
            "terminal_stage_used must be one of "
            + f"{gr00t_same_checkpoint_triplet_eval.MACHINE_CHECKPOINT_ORDER!r}"
        )

    if normalized["controller_output_available"] is True and (
        normalized["terminal_stage_used"] not in (None, "controller_output")
    ):
        raise ValueError(
            "terminal_stage_used must resolve to controller_output when controller_output_available is true"
        )
    if normalized["controller_output_available"] is False and (
        normalized["terminal_stage_used"] not in (None, "controller_input")
    ):
        raise ValueError(
            "terminal_stage_used must fail soft to controller_input when controller_output_available is false"
        )
    return normalized


def build_transition_sidecar_summary(
    *,
    status: str,
    reason_code: str,
    path: str | None,
    episodes_covered: int,
    steps_labeled: int,
    label_counts: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    normalized_status = _as_non_empty_string(status, field_name="status")
    if normalized_status not in RTC_TRANSITION_STATUS_VALUES:
        raise ValueError(f"status must be one of {RTC_TRANSITION_STATUS_VALUES!r}")
    normalized_reason_code = _as_non_empty_string(reason_code, field_name="reason_code")
    normalized_path = _as_optional_string(path, field_name="path")
    normalized_episodes = _as_int(episodes_covered, field_name="episodes_covered")
    normalized_steps = _as_int(steps_labeled, field_name="steps_labeled")
    if normalized_episodes < 0:
        raise ValueError("episodes_covered must be >= 0")
    if normalized_steps < 0:
        raise ValueError("steps_labeled must be >= 0")
    counts_source = dict(label_counts or {})
    normalized_counts: dict[str, int] = {}
    for label in TRANSITION_LABEL_VOCAB:
        raw_count = counts_source.get(label, 0)
        count = _as_int(raw_count, field_name=f"label_counts[{label!r}]")
        if count < 0:
            raise ValueError(f"label_counts[{label!r}] must be >= 0")
        normalized_counts[label] = count
    extra_labels = sorted(set(counts_source) - set(TRANSITION_LABEL_VOCAB))
    if extra_labels:
        raise ValueError(f"label_counts contains unsupported labels: {extra_labels!r}")
    return {
        "status": normalized_status,
        "reason_code": normalized_reason_code,
        "path": normalized_path,
        "schema_version": RTC_TRANSITION_SUMMARY_SCHEMA_VERSION,
        "artifact_kind": RTC_TRANSITION_SUMMARY_ARTIFACT_KIND,
        "label_vocab_version": RTC_TRANSITION_LABEL_VOCAB_VERSION,
        "episodes_covered": normalized_episodes,
        "steps_labeled": normalized_steps,
        "label_counts": normalized_counts,
        "authority_boundary": RTC_AUTHORITY_BOUNDARY,
        "mainline_authority": RTC_MAINLINE_AUTHORITY,
    }


def validate_transition_sidecar_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = dict(_as_mapping(payload, field_name="transition_sidecar_summary"))
    if summary.get("schema_version") != RTC_TRANSITION_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            "transition_sidecar_summary.schema_version must equal "
            + repr(RTC_TRANSITION_SUMMARY_SCHEMA_VERSION)
        )
    if summary.get("artifact_kind") != RTC_TRANSITION_SUMMARY_ARTIFACT_KIND:
        raise ValueError(
            "transition_sidecar_summary.artifact_kind must equal "
            + repr(RTC_TRANSITION_SUMMARY_ARTIFACT_KIND)
        )
    validated = build_transition_sidecar_summary(
        status=_as_non_empty_string(summary.get("status"), field_name="status"),
        reason_code=_as_non_empty_string(
            summary.get("reason_code"), field_name="reason_code"
        ),
        path=summary.get("path"),
        episodes_covered=_as_int(
            summary.get("episodes_covered"), field_name="episodes_covered"
        ),
        steps_labeled=_as_int(summary.get("steps_labeled"), field_name="steps_labeled"),
        label_counts=summary.get("label_counts"),
    )
    if validated["label_vocab_version"] != _as_non_empty_string(
        summary.get("label_vocab_version"), field_name="label_vocab_version"
    ):
        raise ValueError("label_vocab_version mismatch")
    if (
        _as_non_empty_string(
            summary.get("authority_boundary"), field_name="authority_boundary"
        )
        != RTC_AUTHORITY_BOUNDARY
    ):
        raise ValueError(f"authority_boundary must equal {RTC_AUTHORITY_BOUNDARY!r}")
    if _as_bool(summary.get("mainline_authority"), field_name="mainline_authority"):
        raise ValueError("mainline_authority must remain false for RTC summaries")
    return validated


def validate_transition_sidecar_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_join_keys: Sequence[Sequence[object]] | None = None,
) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    observed_keys: set[tuple[str, int]] = set()
    for index, raw_row in enumerate(rows):
        row = validate_transition_sidecar_row(raw_row)
        join_key = record_join_key(row)
        if join_key in observed_keys:
            raise ValueError(f"duplicate RTC transition join key: {join_key!r}")
        observed_keys.add(join_key)
        normalized_rows.append(row)
    if expected_join_keys is None:
        return normalized_rows
    expected_keys: set[tuple[str, int]] = set()
    for index, pair in enumerate(expected_join_keys):
        values = _as_sequence(pair, field_name=f"expected_join_keys[{index}]")
        if len(values) != 2:
            raise ValueError(
                f"expected_join_keys[{index}] must contain [episode_id, t]"
            )
        episode_id = _as_non_empty_string(
            values[0], field_name=f"expected_join_keys[{index}][0]"
        )
        t = _as_int(values[1], field_name=f"expected_join_keys[{index}][1]")
        expected_keys.add((episode_id, int(t)))
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"RTC transition sidecar join mismatch: missing={missing!r} extra={extra!r}"
        )
    return normalized_rows


__all__ = [
    "MODE_VOCAB",
    "PHASE_VOCAB",
    "RTC_AUTHORITY_BOUNDARY",
    "RTC_DIAGNOSTIC_ONLY",
    "RTC_MAINLINE_AUTHORITY",
    "RTC_TRANSITION_LABEL_VOCAB_VERSION",
    "RTC_TRANSITION_ROW_ARTIFACT_KIND",
    "RTC_TRANSITION_ROW_SCHEMA_VERSION",
    "RTC_TRANSITION_STATUS_AVAILABLE",
    "RTC_TRANSITION_STATUS_INVALID",
    "RTC_TRANSITION_STATUS_NOT_AVAILABLE",
    "RTC_TRANSITION_STATUS_VALUES",
    "RTC_TRANSITION_SUMMARY_ARTIFACT_KIND",
    "RTC_TRANSITION_SUMMARY_SCHEMA_VERSION",
    "SEMANTIC_STATE_VALUES",
    "STAGE_VOCAB",
    "TRANSITION_LABEL_VOCAB",
    "build_transition_label_counts",
    "build_transition_sidecar_summary",
    "normalize_mode",
    "normalize_phase",
    "normalize_semantic_state",
    "normalize_stage",
    "normalize_transition_label",
    "record_join_key",
    "validate_transition_sidecar_row",
    "validate_transition_sidecar_rows",
    "validate_transition_sidecar_summary",
]
