from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_BUCKET_DIR = Path("agent/artifacts/state_conditioned_materialization/bucket_a")
DEFAULT_OUTPUT_DIR = DEFAULT_BUCKET_DIR
DEFAULT_HISTORY_K = 8

BUCKET_A_SIDECAR_JSON_NAME = "bucket_A_sidecar.jsonl"
BUCKET_A_JOIN_COVERAGE_JSON_NAME = "bucket_A_join_coverage.json"
BUCKET_A_EXPORTER_MANIFEST_JSON_NAME = "bucket_A_exporter_manifest.json"
EXPECTED_ACCEPTED_EPISODE_COUNT = 24

SEMANTIC_STATE_VALUES: tuple[str, ...] = (
    "SEARCHING",
    "APPLE_VISIBLE_APPROACH",
    "GRASPING",
    "VERIFYING_HOLD",
    "TRANSPORTING",
    "PLACING",
    "RECOVERY_REACQUIRE",
    "RECOVERY_REGRASP",
)

MEMORY_COMMIT_CAUSE_VALUES: tuple[str, ...] = (
    "none",
    "nominal_visual_confirmation",
    "contact_confirmation",
    "grasp_confirmation",
    "hold_loss_detected",
    "recovery_entry",
    "recovery_exit",
)

SUMMARY_TEMPLATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")

PRIVILEGED_FIELD_NAMES: tuple[str, ...] = (
    "privileged.apple_pose_world",
    "privileged.hand_to_apple_rel_pose",
    "privileged.apple_to_plate_rel_pose",
    "privileged.contact_flag",
    "privileged.apple_in_hand",
    "privileged.apple_visible",
    "privileged.last_seen_dt",
    "privileged.last_in_hand_dt",
)

ANALYSIS_ONLY_FIELD_NAMES: tuple[str, ...] = (
    "semantic_state",
    "memory_commit_mask",
    "memory_commit_cause",
    "recovery_entry_step",
    "recovery_exit_step",
    "summary_template",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap import state_conditioned_bucket_a_import
from work.recap.lerobot_export import dataset_export as lerobot_v2_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate canonical Bucket A into a history-aware bucket-level sidecar "
            "after Gate A is ready."
        )
    )
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        default=DEFAULT_BUCKET_DIR,
        help="Canonical Bucket A directory containing gate and manifest artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives bucket_A_sidecar.jsonl and companion artifacts.",
    )
    parser.add_argument(
        "--history-k",
        type=int,
        default=DEFAULT_HISTORY_K,
        help="Frozen history window length; only the canonical value is accepted.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _validate_existing_dir(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"{arg_name} directory does not exist: {resolved}")
    return resolved


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    return state_conditioned_bucket_a_import._as_non_empty_string(
        value, field_name=field_name
    )


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, field_name=field_name)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _as_list(
    value: object, *, field_name: str, expected_len: int | None = None
) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    result = list(value)
    if expected_len is not None and len(result) != int(expected_len):
        raise ValueError(
            f"{field_name} must have length {expected_len}, got {len(result)}"
        )
    return result


_MISSING = object()


def _deep_get(source: Mapping[str, Any], dotted_path: str) -> object:
    current: object = source
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _resolve_optional_field(field_name: str, *sources: object) -> object | None:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        if field_name in source:
            return source[field_name]
        nested = _deep_get(source, field_name)
        if nested is not _MISSING:
            return nested
    return None


def _resolve_required_field(field_name: str, *sources: object) -> object:
    value = _resolve_optional_field(field_name, *sources)
    if value is None:
        raise ValueError(
            f"missing required field {field_name!r} in source sidecar data"
        )
    return value


def _normalize_semantic_state(value: object) -> str:
    normalized = _as_non_empty_string(value, field_name="semantic_state").upper()
    if normalized not in SEMANTIC_STATE_VALUES:
        raise ValueError(
            "semantic_state must be one of "
            + f"{SEMANTIC_STATE_VALUES!r}, got {normalized!r}"
        )
    return normalized


def _normalize_memory_commit_cause(value: object) -> str:
    normalized = _as_non_empty_string(value, field_name="memory_commit_cause")
    if normalized not in MEMORY_COMMIT_CAUSE_VALUES:
        raise ValueError(
            "memory_commit_cause must be one of "
            + f"{MEMORY_COMMIT_CAUSE_VALUES!r}, got {normalized!r}"
        )
    return normalized


def _normalize_memory_commit_mask(value: object) -> bool | list[bool]:
    if isinstance(value, bool):
        return bool(value)
    items = _as_list(value, field_name="memory_commit_mask")
    return [_as_bool(item, field_name="memory_commit_mask[]") for item in items]


def _normalize_summary_template(value: object) -> str | None:
    if value is None:
        return None
    normalized = _as_non_empty_string(value, field_name="summary_template")
    if not SUMMARY_TEMPLATE_ID_RE.match(normalized):
        raise ValueError(
            "summary_template must be a template id or null, not rendered natural language"
        )
    return normalized


def _normalize_history_indices(
    base_row: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    window = _as_list(
        base_row.get("prehistory_window"),
        field_name="prehistory_window",
        expected_len=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
    )
    t_std_indices: list[int] = []
    t_raw_indices: list[int] = []
    for index, raw_item in enumerate(window):
        item = _as_mapping(raw_item, field_name=f"prehistory_window[{index}]")
        t_std = _as_int(
            item.get("t_std"), field_name=f"prehistory_window[{index}].t_std"
        )
        raw_value = item.get("t_raw", item.get("t_raw_index", t_std))
        t_raw = _as_int(raw_value, field_name=f"prehistory_window[{index}].t_raw")
        t_std_indices.append(int(t_std))
        t_raw_indices.append(int(t_raw))
    return t_std_indices, t_raw_indices


def _build_history_timestamp_s(
    *,
    history_valid_mask: list[bool],
    history_t_raw_indices: list[int],
    transition_by_t: Mapping[int, Mapping[str, Any]],
) -> list[float | None]:
    history_timestamp_s: list[float | None] = []
    for index, is_valid in enumerate(history_valid_mask):
        if not bool(is_valid):
            history_timestamp_s.append(None)
            continue
        t_raw = int(history_t_raw_indices[index])
        transition = transition_by_t.get(t_raw)
        if transition is None:
            raise ValueError(
                f"missing transition row for history_t_raw_indices[{index}]={t_raw}"
            )
        raw_timestamp = transition.get("timestamp_s", transition.get("timestamp"))
        if raw_timestamp is None:
            history_timestamp_s.append(float(t_raw))
            continue
        history_timestamp_s.append(
            _as_number(raw_timestamp, field_name=f"history_timestamp_s[{index}]")
        )
    return history_timestamp_s


def _normalize_history_timestamp_s(
    value: object, *, expected_len: int
) -> list[float | None]:
    items = _as_list(value, field_name="history_timestamp_s", expected_len=expected_len)
    normalized: list[float | None] = []
    for index, item in enumerate(items):
        if item is None:
            normalized.append(None)
            continue
        normalized.append(_as_number(item, field_name=f"history_timestamp_s[{index}]"))
    return normalized


def validate_consolidated_sidecar_row(row: Mapping[str, Any]) -> dict[str, Any]:
    state_conditioned_bucket_a_import.validate_sidecar_row_for_gate(row)

    history_valid_mask = _as_list(
        row.get("history_valid_mask"),
        field_name="history_valid_mask",
        expected_len=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
    )
    normalized_valid_mask = [
        _as_bool(item, field_name=f"history_valid_mask[{index}]")
        for index, item in enumerate(history_valid_mask)
    ]

    history_t_std_indices = [
        _as_int(item, field_name=f"history_t_std_indices[{index}]")
        for index, item in enumerate(
            _as_list(
                row.get("history_t_std_indices"),
                field_name="history_t_std_indices",
                expected_len=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
            )
        )
    ]
    history_t_raw_indices = [
        _as_int(item, field_name=f"history_t_raw_indices[{index}]")
        for index, item in enumerate(
            _as_list(
                row.get("history_t_raw_indices"),
                field_name="history_t_raw_indices",
                expected_len=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
            )
        )
    ]
    history_timestamp_s = _normalize_history_timestamp_s(
        row.get("history_timestamp_s"),
        expected_len=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
    )

    for index, is_valid in enumerate(normalized_valid_mask):
        if bool(is_valid) and history_timestamp_s[index] is None:
            raise ValueError(
                f"history_timestamp_s[{index}] missing for valid history slot"
            )

    normalized_row = dict(row)
    normalized_row["history_t_std_indices"] = history_t_std_indices
    normalized_row["history_t_raw_indices"] = history_t_raw_indices
    normalized_row["history_timestamp_s"] = history_timestamp_s

    for field_name in PRIVILEGED_FIELD_NAMES:
        if normalized_row.get(field_name) is None:
            raise ValueError(f"{field_name} is required in consolidated sidecar rows")

    normalized_row["semantic_state"] = _normalize_semantic_state(
        normalized_row.get("semantic_state")
    )
    normalized_row["memory_commit_mask"] = _normalize_memory_commit_mask(
        normalized_row.get("memory_commit_mask")
    )
    normalized_row["memory_commit_cause"] = _normalize_memory_commit_cause(
        normalized_row.get("memory_commit_cause")
    )
    normalized_row["recovery_entry_step"] = _as_optional_int(
        normalized_row.get("recovery_entry_step"), field_name="recovery_entry_step"
    )
    normalized_row["recovery_exit_step"] = _as_optional_int(
        normalized_row.get("recovery_exit_step"), field_name="recovery_exit_step"
    )
    normalized_row["summary_template"] = _normalize_summary_template(
        normalized_row.get("summary_template")
    )
    return normalized_row


def _analysis_only_sources(
    *,
    base_row: Mapping[str, Any],
    transition: Mapping[str, Any],
    label: Mapping[str, Any],
    episode_record: Mapping[str, Any],
) -> tuple[object, ...]:
    metadata = _as_mapping(
        episode_record.get("metadata", {}), field_name="episode.metadata"
    )
    episode_analysis = _as_mapping(
        metadata.get("analysis_only", {}), field_name="episode.metadata.analysis_only"
    )
    return (
        base_row,
        _as_mapping(base_row.get("analysis_only", {}), field_name="row.analysis_only"),
        transition,
        _as_mapping(
            transition.get("analysis_only", {}), field_name="transition.analysis_only"
        ),
        label,
        _as_mapping(label.get("analysis_only", {}), field_name="label.analysis_only"),
        episode_analysis,
        metadata,
        episode_record,
    )


def _build_transition_lookup(
    records: Sequence[Mapping[str, Any]], *, record_name: str
) -> dict[int, Mapping[str, Any]]:
    by_t: dict[int, Mapping[str, Any]] = {}
    for record in records:
        t_value = _as_int(record.get("t"), field_name=f"{record_name}.t")
        if t_value in by_t:
            raise ValueError(f"duplicate {record_name} row for t={t_value}")
        by_t[t_value] = record
    return by_t


def _build_consolidated_row(
    *,
    episode_record: Mapping[str, Any],
    base_row: Mapping[str, Any],
    transition_by_t: Mapping[int, Mapping[str, Any]],
    label_by_t: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    state_conditioned_bucket_a_import.validate_sidecar_row_for_gate(base_row)
    t_value = _as_int(base_row.get("t"), field_name="t")
    transition = transition_by_t.get(t_value)
    if transition is None:
        raise ValueError(f"missing transition for episode step t={t_value}")
    label = label_by_t.get(t_value)
    if label is None:
        raise ValueError(f"missing label for episode step t={t_value}")

    analysis_sources = _analysis_only_sources(
        base_row=base_row,
        transition=transition,
        label=label,
        episode_record=episode_record,
    )
    privileged_sources: tuple[object, ...] = (
        base_row,
        transition,
        label,
        episode_record,
    )

    history_valid_mask = [
        _as_bool(item, field_name=f"history_valid_mask[{index}]")
        for index, item in enumerate(
            _as_list(
                base_row.get("history_valid_mask"),
                field_name="history_valid_mask",
                expected_len=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
            )
        )
    ]
    history_t_std_indices, history_t_raw_indices = _normalize_history_indices(base_row)

    row = dict(base_row)
    row["history_t_std_indices"] = history_t_std_indices
    row["history_t_raw_indices"] = history_t_raw_indices
    row["history_timestamp_s"] = _build_history_timestamp_s(
        history_valid_mask=history_valid_mask,
        history_t_raw_indices=history_t_raw_indices,
        transition_by_t=transition_by_t,
    )

    optional_event = _resolve_optional_field("event", base_row, transition, label)
    if optional_event is not None:
        row["event"] = optional_event
    optional_recovery_needed = _resolve_optional_field(
        "recovery_needed", base_row, transition, label
    )
    if optional_recovery_needed is not None:
        row["recovery_needed"] = optional_recovery_needed

    for field_name in PRIVILEGED_FIELD_NAMES:
        row[field_name] = _resolve_required_field(field_name, *privileged_sources)

    row["semantic_state"] = _resolve_required_field("semantic_state", *analysis_sources)
    row["memory_commit_mask"] = _resolve_required_field(
        "memory_commit_mask", *analysis_sources
    )
    row["memory_commit_cause"] = _resolve_required_field(
        "memory_commit_cause", *analysis_sources
    )
    row["recovery_entry_step"] = _resolve_optional_field(
        "recovery_entry_step", *analysis_sources
    )
    row["recovery_exit_step"] = _resolve_optional_field(
        "recovery_exit_step", *analysis_sources
    )
    row["summary_template"] = _resolve_optional_field(
        "summary_template", *analysis_sources
    )
    return validate_consolidated_sidecar_row(row)


def _load_ready_gate(
    bucket_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    gate_path = bucket_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME
    manifest_path = bucket_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME
    if not gate_path.is_file():
        raise ValueError(f"missing Gate A artifact: {gate_path}")
    if not manifest_path.is_file():
        raise ValueError(f"missing canonical manifest: {manifest_path}")
    gate = state_conditioned_bucket_a_import._read_json(gate_path)
    manifest = state_conditioned_bucket_a_import._read_json(manifest_path)
    if not bool(gate.get("ready", False)):
        raise ValueError(
            "canonical Bucket A sidecar refuses to run until "
            "bucket_A_gate_a_ready.json.ready == true"
        )
    return gate, manifest, gate_path, manifest_path


def _accepted_canonical_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    accepted_entries: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    for raw_entry in manifest.get("episodes", []):
        entry = dict(_as_mapping(raw_entry, field_name="manifest.episodes[]"))
        if not bool(entry.get("accepted", False)):
            continue
        if not bool(entry.get("fresh_nominal_recollection", False)):
            continue
        if bool(entry.get("debug_only", False)):
            continue
        if bool(entry.get("reused_existing_live_dataset", True)):
            continue
        episode_id = _as_non_empty_string(
            entry.get("episode_id"), field_name="episode_id"
        )
        if episode_id in seen_episode_ids:
            raise ValueError(
                f"duplicate canonical episode_id in manifest: {episode_id}"
            )
        seen_episode_ids.add(episode_id)
        accepted_entries.append(entry)
    return accepted_entries


def _build_exporter_manifest(
    *,
    accepted_episode_ids: list[str],
    manifest_path: Path,
    gate_path: Path,
    sidecar_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed_field_names = sorted(
        {
            str(field_name)
            for row in sidecar_rows
            for field_name in row.keys()
            if str(field_name).strip()
        }
    )
    field_groups = lerobot_v2_export.build_state_conditioned_field_groups(
        observed_field_names
    )
    return {
        "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
        "artifact_kind": "bucket_A_exporter_manifest",
        "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
        "canonical_manifest_path": str(manifest_path),
        "gate_path": str(gate_path),
        "accepted_episode_count": int(len(accepted_episode_ids)),
        "accepted_episode_ids": list(accepted_episode_ids),
        "field_groups": field_groups,
    }


def materialize_bucket_a_sidecar(
    *,
    bucket_dir: Path,
    output_dir: Path,
    history_k: int = DEFAULT_HISTORY_K,
) -> dict[str, Any]:
    bucket_dir = _validate_existing_dir(bucket_dir, arg_name="bucket-dir")
    output_dir = _validate_output_dir(output_dir)
    if int(history_k) != int(
        state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K
    ):
        raise ValueError(
            "history-k is frozen at "
            + str(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K)
        )

    gate, manifest, gate_path, manifest_path = _load_ready_gate(bucket_dir)
    accepted_entries = _accepted_canonical_entries(manifest)
    required_count = int(
        gate.get(
            "required_distinct_accepted_episode_count",
            manifest.get(
                "required_distinct_episode_count", EXPECTED_ACCEPTED_EPISODE_COUNT
            ),
        )
    )
    if int(required_count) != int(EXPECTED_ACCEPTED_EPISODE_COUNT):
        raise ValueError(
            "canonical Bucket A sidecar expects exactly 24 fresh accepted episodes; got "
            + str(required_count)
        )
    if len(accepted_entries) != int(required_count):
        raise ValueError(
            "canonical manifest does not contain the required number of accepted fresh episodes: "
            + f"expected {required_count}, got {len(accepted_entries)}"
        )

    sidecar_rows: list[dict[str, Any]] = []
    all_transitions: list[Mapping[str, Any]] = []
    all_labels: list[Mapping[str, Any]] = []
    accepted_episode_ids: list[str] = []

    for entry in accepted_entries:
        episode_id = _as_non_empty_string(
            entry.get("episode_id"), field_name="episode_id"
        )
        dataset_dir = _validate_existing_dir(
            Path(
                _as_non_empty_string(
                    entry.get("source_dataset_dir"), field_name="source_dataset_dir"
                )
            ),
            arg_name=f"source_dataset_dir for {episode_id}",
        )
        dataset_records = state_conditioned_bucket_a_import._load_dataset_records(
            dataset_dir
        )
        episodes_by_id = dict(dataset_records["episodes_by_id"])
        if episode_id not in episodes_by_id:
            raise ValueError(
                f"canonical manifest episode {episode_id!r} missing from source dataset {dataset_dir}"
            )
        episode_record = dict(episodes_by_id[episode_id])
        transitions = list(
            dataset_records["transitions_by_episode"].get(episode_id, [])
        )
        labels = list(dataset_records["labels_by_episode"].get(episode_id, []))
        base_sidecar_rows = list(
            dataset_records["sidecar_by_episode"].get(episode_id, [])
        )
        if not base_sidecar_rows:
            raise ValueError(
                f"canonical manifest episode {episode_id!r} is missing history-aware source sidecar rows"
            )
        transition_by_t = _build_transition_lookup(
            transitions, record_name="transition"
        )
        label_by_t = _build_transition_lookup(labels, record_name="label")
        episode_rows = [
            _build_consolidated_row(
                episode_record=episode_record,
                base_row=row,
                transition_by_t=transition_by_t,
                label_by_t=label_by_t,
            )
            for row in sorted(
                base_sidecar_rows,
                key=lambda item: _as_int(item.get("t"), field_name="sidecar.t"),
            )
        ]
        sidecar_rows.extend(episode_rows)
        all_transitions.extend(transitions)
        all_labels.extend(labels)
        accepted_episode_ids.append(episode_id)

    sidecar_path = output_dir / BUCKET_A_SIDECAR_JSON_NAME
    state_conditioned_bucket_a_import._write_jsonl(sidecar_path, sidecar_rows)

    expected_join_keys = [
        [episode_id, t_value]
        for episode_id, t_value in sorted(
            state_conditioned_bucket_a_import._record_join_key(record)
            for record in all_transitions
        )
    ]
    round_trip = state_conditioned_bucket_a_import.validate_sidecar_round_trip(
        sidecar_path=sidecar_path,
        expected_join_keys=expected_join_keys,
    )
    join_coverage = state_conditioned_bucket_a_import.compute_episode_join_coverage(
        all_transitions,
        all_labels,
        sidecar_rows,
    )
    if float(join_coverage["coverage_ratio"]) < float(
        state_conditioned_bucket_a_import.JOIN_COVERAGE_THRESHOLD
    ):
        raise ValueError(
            "bucket-level join coverage below threshold: "
            + str(join_coverage["coverage_ratio"])
        )

    join_coverage_payload = {
        "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
        "artifact_kind": "bucket_A_join_coverage",
        "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
        "coverage_threshold": float(
            state_conditioned_bucket_a_import.JOIN_COVERAGE_THRESHOLD
        ),
        "accepted_episode_count": int(len(accepted_episode_ids)),
        "accepted_episode_ids": list(accepted_episode_ids),
        **join_coverage,
        "sidecar_round_trip": dict(round_trip),
    }
    join_coverage_path = output_dir / BUCKET_A_JOIN_COVERAGE_JSON_NAME
    state_conditioned_bucket_a_import._write_json(
        join_coverage_path, join_coverage_payload
    )

    exporter_manifest = _build_exporter_manifest(
        accepted_episode_ids=accepted_episode_ids,
        manifest_path=manifest_path,
        gate_path=gate_path,
        sidecar_rows=sidecar_rows,
    )
    exporter_manifest_path = output_dir / BUCKET_A_EXPORTER_MANIFEST_JSON_NAME
    state_conditioned_bucket_a_import._write_json(
        exporter_manifest_path,
        exporter_manifest,
    )
    return {
        "accepted_episode_count": int(len(accepted_episode_ids)),
        "sidecar_path": str(sidecar_path),
        "join_coverage_path": str(join_coverage_path),
        "exporter_manifest_path": str(exporter_manifest_path),
        "coverage_ratio": float(join_coverage["coverage_ratio"]),
        "field_groups": exporter_manifest["field_groups"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = materialize_bucket_a_sidecar(
            bucket_dir=args.bucket_dir,
            output_dir=args.output_dir,
            history_k=int(args.history_k),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ANALYSIS_ONLY_FIELD_NAMES",
    "BUCKET_A_EXPORTER_MANIFEST_JSON_NAME",
    "BUCKET_A_JOIN_COVERAGE_JSON_NAME",
    "BUCKET_A_SIDECAR_JSON_NAME",
    "EXPECTED_ACCEPTED_EPISODE_COUNT",
    "MEMORY_COMMIT_CAUSE_VALUES",
    "PRIVILEGED_FIELD_NAMES",
    "SEMANTIC_STATE_VALUES",
    "build_parser",
    "materialize_bucket_a_sidecar",
    "validate_consolidated_sidecar_row",
]


if __name__ == "__main__":
    raise SystemExit(main())
