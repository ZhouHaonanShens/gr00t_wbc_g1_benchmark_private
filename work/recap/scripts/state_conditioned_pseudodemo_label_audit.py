#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
import time
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_SNAPSHOT = Path(
    "agent/artifacts/state_conditioned_materialization/harvest/snapshot_candidates.jsonl"
)
DEFAULT_MANIFEST = Path(
    "agent/artifacts/state_conditioned_materialization/harvest/local_recovery_pseudodemo_manifest.json"
)
DEFAULT_LABELS = Path(
    "agent/artifacts/state_conditioned_materialization/training/state_conditioned_sft_labels.jsonl"
)
DEFAULT_OUTPUT = Path(
    "agent/artifacts/state_conditioned_materialization/audit/pseudodemo_label_audit.json"
)

SCHEMA_VERSION = "g1_state_conditioned_pseudodemo_label_audit_v1"
CHECK_ORDER: tuple[str, ...] = (
    "missing_field_rate",
    "family_source_mismatch",
    "temporal_drift",
    "history_payload_incompleteness",
    "reset_boundary_violations",
    "invalid_mask_anomalies",
    "teacher_target_null_degradation",
    "nominal_recovery_confusion",
    "m2_label_missing_rate",
    "formal_label_coverage",
)
M2_EXPECTED_FIELDS: tuple[str, ...] = (
    "return_G",
    "value_V",
    "advantage_A",
    "epsilon_l",
    "indicator_I",
)
NEGATIVE_MODE_LABEL_NULL_TEACHER_TARGET = "label_null_teacher_target"
NEGATIVE_MODE_SNAPSHOT_RESET_BOUNDARY = "snapshot_reset_boundary_cross_episode"
NEGATIVE_MODE_VALUES: tuple[str, ...] = (
    NEGATIVE_MODE_LABEL_NULL_TEACHER_TARGET,
    NEGATIVE_MODE_SNAPSHOT_RESET_BOUNDARY,
)
EXAMPLE_LIMIT = 8


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap import state_conditioned_build_training_set
from work.recap import episode_writer as recap_episode_writer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit pseudodemo snapshot/manifest/training-label alignment and write a "
            "machine-readable JSON report without mutating source artifacts."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="snapshot_candidates.jsonl emitted by formal harvest feasibility.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="local_recovery_pseudodemo_manifest.json emitted by formal harvest.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="state_conditioned_sft_labels.jsonl emitted by unified training-set build.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the machine-readable audit JSON output.",
    )
    parser.add_argument(
        "--deliberate-bad-sample",
        choices=NEGATIVE_MODE_VALUES,
        default=None,
        help=(
            "Inject a synthetic in-memory corruption to prove the auditor fails closed "
            "or captures the anomaly explicitly. Source artifacts remain untouched."
        ),
    )
    return parser


def _validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"missing required {arg_name}: {resolved}")
    return resolved


def _validate_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(
            f"output must be a file path, got existing directory: {resolved}"
        )
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"output must be a file path: {resolved}")
    if not resolved.parent.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"expected JSON object in {path}, got {type(payload).__name__}"
        )
    return dict(payload)


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{lineno}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"expected JSON object in {path}:{lineno}, got {type(payload).__name__}"
                )
            records.append(dict(payload))
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)
    return path


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    return state_conditioned_bucket_a_import._as_non_empty_string(
        value,
        field_name=field_name,
    )


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_list(
    value: object,
    *,
    field_name: str,
    expected_len: int | None = None,
) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    items = list(value)
    if expected_len is not None and len(items) != int(expected_len):
        raise ValueError(
            f"{field_name} must have length {expected_len}, got {len(items)}"
        )
    return items


def _safe_trace(raw: Mapping[str, Any]) -> dict[str, Any]:
    trace: dict[str, Any] = {}
    for key in (
        "snapshot_id",
        "episode_id",
        "source_episode_id",
        "anchor_episode_id",
        "source_snapshot_id",
        "source_snapshot_family",
        "family",
        "training_view",
        "source_bucket",
        "source_kind",
        "sample_id",
        "source_t",
        "anchor_t",
        "t",
    ):
        if key in raw:
            trace[key] = raw.get(key)
    return trace


def _new_check(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS",
        "anomaly_count": 0,
        "denominator": 0,
        "anomaly_rate": 0.0,
        "examples": [],
    }


def _new_missing_field_check() -> dict[str, Any]:
    check = _new_check("missing_field_rate")
    check.update(
        {
            "required_cell_count": 0,
            "missing_field_count": 0,
            "missing_record_count": 0,
            "missing_rate": 0.0,
            "field_counts": {},
            "artifact_breakdown": {},
        }
    )
    return check


def _ensure_artifact_breakdown(
    check: dict[str, Any], artifact_name: str
) -> dict[str, Any]:
    artifact_breakdown = check.setdefault("artifact_breakdown", {})
    if artifact_name not in artifact_breakdown:
        artifact_breakdown[artifact_name] = {
            "record_count": 0,
            "required_cell_count": 0,
            "missing_field_count": 0,
            "missing_record_count": 0,
            "missing_rate": 0.0,
        }
    return dict(artifact_breakdown[artifact_name])


def _set_artifact_breakdown(
    check: dict[str, Any], artifact_name: str, payload: Mapping[str, Any]
) -> None:
    artifact_breakdown = check.setdefault("artifact_breakdown", {})
    artifact_breakdown[artifact_name] = dict(payload)


def _append_example(check: dict[str, Any], example: Mapping[str, Any]) -> None:
    examples = list(check.setdefault("examples", []))
    if len(examples) >= EXAMPLE_LIMIT:
        return
    normalized = dict(example)
    if normalized not in examples:
        examples.append(normalized)
        examples.sort(
            key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True)
        )
        check["examples"] = examples[:EXAMPLE_LIMIT]


def _record_anomaly(
    checks: dict[str, dict[str, Any]],
    check_name: str,
    *,
    trace: Mapping[str, Any],
    message: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    check = checks[check_name]
    check["status"] = "FAIL"
    check["anomaly_count"] = int(check.get("anomaly_count", 0)) + 1
    payload = {"message": str(message), **dict(trace)}
    if extra:
        payload.update(dict(extra))
    _append_example(check, payload)


def _record_missing_fields(
    checks: dict[str, dict[str, Any]],
    *,
    artifact_name: str,
    record: Mapping[str, Any],
    required_fields: Sequence[str],
    conditional_required_fields: Sequence[str] = (),
) -> list[str]:
    check = checks["missing_field_rate"]
    breakdown = _ensure_artifact_breakdown(check, artifact_name)
    breakdown["record_count"] = int(breakdown["record_count"]) + 1
    breakdown["required_cell_count"] = int(breakdown["required_cell_count"]) + int(
        len(required_fields) + len(conditional_required_fields)
    )
    check["denominator"] = int(check.get("denominator", 0)) + 1
    check["required_cell_count"] = int(check.get("required_cell_count", 0)) + int(
        len(required_fields) + len(conditional_required_fields)
    )

    trace = _safe_trace(record)
    missing_fields: list[str] = []
    for field_name in list(required_fields) + list(conditional_required_fields):
        value_missing = field_name not in record
        if not value_missing:
            value = record.get(field_name)
            if value is None:
                value_missing = True
            elif isinstance(value, str) and not value.strip():
                value_missing = True
        if not value_missing:
            continue
        missing_fields.append(field_name)
        check["status"] = "FAIL"
        check["missing_field_count"] = int(check.get("missing_field_count", 0)) + 1
        breakdown["missing_field_count"] = int(breakdown["missing_field_count"]) + 1
        field_counts = dict(check.get("field_counts", {}))
        field_key = f"{artifact_name}.{field_name}"
        field_counts[field_key] = int(field_counts.get(field_key, 0)) + 1
        check["field_counts"] = field_counts
        _append_example(
            check,
            {
                **trace,
                "artifact": artifact_name,
                "missing_field": field_name,
            },
        )
    if missing_fields:
        check["anomaly_count"] = int(check.get("anomaly_count", 0)) + 1
        check["missing_record_count"] = int(check.get("missing_record_count", 0)) + 1
        breakdown["missing_record_count"] = int(breakdown["missing_record_count"]) + 1
    _set_artifact_breakdown(check, artifact_name, breakdown)
    return missing_fields


def _finalize_checks(checks: dict[str, dict[str, Any]]) -> None:
    for check_name, check in checks.items():
        denominator = int(check.get("denominator", 0))
        anomaly_count = int(check.get("anomaly_count", 0))
        check["anomaly_rate"] = (
            float(anomaly_count) / float(denominator) if denominator > 0 else 0.0
        )
        if check_name == "missing_field_rate":
            required_cell_count = int(check.get("required_cell_count", 0))
            missing_field_count = int(check.get("missing_field_count", 0))
            check["missing_rate"] = (
                float(missing_field_count) / float(required_cell_count)
                if required_cell_count > 0
                else 0.0
            )
            artifact_breakdown = dict(check.get("artifact_breakdown", {}))
            normalized_breakdown: dict[str, Any] = {}
            for artifact_name in sorted(artifact_breakdown):
                item = dict(artifact_breakdown[artifact_name])
                required_count = int(item.get("required_cell_count", 0))
                missing_count = int(item.get("missing_field_count", 0))
                item["missing_rate"] = (
                    float(missing_count) / float(required_count)
                    if required_count > 0
                    else 0.0
                )
                normalized_breakdown[artifact_name] = item
            check["artifact_breakdown"] = normalized_breakdown
            check["field_counts"] = {
                key: check["field_counts"][key]
                for key in sorted(check.get("field_counts", {}))
            }


def _snapshot_history_signature(snapshot_row: Mapping[str, Any]) -> dict[str, Any]:
    prehistory_window = [
        dict(_as_mapping(item, field_name="prehistory_window[]"))
        for item in _as_list(
            snapshot_row.get("prehistory_window"),
            field_name="prehistory_window",
            expected_len=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
        )
    ]
    return {
        "history_k": _as_int(snapshot_row.get("history_k"), field_name="history_k"),
        "history_stride": _as_int(
            snapshot_row.get("history_stride"), field_name="history_stride"
        ),
        "history_valid_mask": list(
            _as_list(
                snapshot_row.get("history_valid_mask"),
                field_name="history_valid_mask",
                expected_len=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
            )
        ),
        "history_t_std_indices": [
            _as_int(item.get("t_std"), field_name=f"prehistory_window[{index}].t_std")
            for index, item in enumerate(prehistory_window)
        ],
        "history_t_raw_indices": [
            _as_int(
                item.get("t_raw", item.get("t_raw_index", item.get("t_std"))),
                field_name=f"prehistory_window[{index}].t_raw",
            )
            for index, item in enumerate(prehistory_window)
        ],
        "history_timestamp_s": [
            None
            if not bool(snapshot_row["history_valid_mask"][index])
            else float(
                _as_int(
                    item.get("t_raw", item.get("t_raw_index", item.get("t_std"))),
                    field_name=f"prehistory_window[{index}].t_raw",
                )
            )
            for index, item in enumerate(prehistory_window)
        ],
        "deployable.previous_action_history": list(
            _as_list(
                snapshot_row.get("deployable.previous_action_history"),
                field_name="deployable.previous_action_history",
                expected_len=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
            )
        ),
        "deployable.proprio_history": list(
            _as_list(
                snapshot_row.get("deployable.proprio_history"),
                field_name="deployable.proprio_history",
                expected_len=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
            )
        ),
        "deployable.short_visual_history_refs": list(
            _as_list(
                snapshot_row.get("deployable.short_visual_history_refs"),
                field_name="deployable.short_visual_history_refs",
                expected_len=state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
            )
        ),
    }


def _validate_mask_and_history_payload(
    checks: dict[str, dict[str, Any]],
    *,
    record: Mapping[str, Any],
    history_k: int,
    history_valid_mask: Sequence[Any],
    history_t_std_indices: Sequence[Any] | None,
    history_t_raw_indices: Sequence[Any] | None,
    history_timestamp_s: Sequence[Any] | None,
    previous_action_history: Sequence[Any] | None,
    proprio_history: Sequence[Any] | None,
    short_visual_history_refs: Sequence[Any] | None,
    source_t: int | None,
) -> None:
    trace = _safe_trace(record)
    mask_values = list(history_valid_mask)
    if len(mask_values) != int(history_k):
        _record_anomaly(
            checks,
            "history_payload_incompleteness",
            trace=trace,
            message=(
                f"history_valid_mask length mismatch: expected {history_k}, "
                f"got {len(mask_values)}"
            ),
        )
        return

    normalized_mask: list[bool] = []
    for index, value in enumerate(mask_values):
        if not isinstance(value, bool):
            _record_anomaly(
                checks,
                "invalid_mask_anomalies",
                trace=trace,
                message=f"history_valid_mask[{index}] must be bool, got {type(value).__name__}",
            )
            return
        normalized_mask.append(bool(value))
    if not any(normalized_mask):
        _record_anomaly(
            checks,
            "invalid_mask_anomalies",
            trace=trace,
            message="history_valid_mask contains no valid slots",
        )
        return
    seen_valid = False
    seen_invalid_after_valid = False
    for index, is_valid in enumerate(normalized_mask):
        if is_valid:
            if seen_invalid_after_valid:
                _record_anomaly(
                    checks,
                    "invalid_mask_anomalies",
                    trace=trace,
                    message=(
                        "history_valid_mask must be left-padded then contiguous true; "
                        f"found true again at index {index}"
                    ),
                )
                return
            seen_valid = True
        elif seen_valid:
            seen_invalid_after_valid = True

    payload_specs = (
        ("history_t_std_indices", history_t_std_indices),
        ("history_t_raw_indices", history_t_raw_indices),
        ("history_timestamp_s", history_timestamp_s),
        ("deployable.previous_action_history", previous_action_history),
        ("deployable.proprio_history", proprio_history),
        ("deployable.short_visual_history_refs", short_visual_history_refs),
    )
    normalized_payloads: dict[str, list[Any]] = {}
    for field_name, payload in payload_specs:
        if payload is None:
            _record_anomaly(
                checks,
                "history_payload_incompleteness",
                trace=trace,
                message=f"{field_name} missing while history_valid_mask is present",
            )
            return
        if len(payload) != int(history_k):
            _record_anomaly(
                checks,
                "history_payload_incompleteness",
                trace=trace,
                message=(
                    f"{field_name} length mismatch: expected {history_k}, got {len(payload)}"
                ),
            )
            return
        normalized_payloads[field_name] = list(payload)

    valid_indices = [
        index for index, is_valid in enumerate(normalized_mask) if is_valid
    ]
    previous_valid_t_std: int | None = None
    previous_valid_t_raw: int | None = None
    for index, is_valid in enumerate(normalized_mask):
        t_std = normalized_payloads["history_t_std_indices"][index]
        t_raw = normalized_payloads["history_t_raw_indices"][index]
        ts = normalized_payloads["history_timestamp_s"][index]
        prev_action = normalized_payloads["deployable.previous_action_history"][index]
        proprio = normalized_payloads["deployable.proprio_history"][index]
        visual = normalized_payloads["deployable.short_visual_history_refs"][index]
        if is_valid:
            if prev_action is None or proprio is None or visual is None:
                _record_anomaly(
                    checks,
                    "history_payload_incompleteness",
                    trace=trace,
                    message=f"valid history slot {index} is missing deployable payload",
                )
                return
            if ts is None:
                _record_anomaly(
                    checks,
                    "history_payload_incompleteness",
                    trace=trace,
                    message=f"valid history slot {index} is missing history_timestamp_s",
                )
                return
            if not isinstance(t_std, int) or isinstance(t_std, bool):
                _record_anomaly(
                    checks,
                    "history_payload_incompleteness",
                    trace=trace,
                    message=f"history_t_std_indices[{index}] must be int for valid slot",
                )
                return
            if not isinstance(t_raw, int) or isinstance(t_raw, bool):
                _record_anomaly(
                    checks,
                    "history_payload_incompleteness",
                    trace=trace,
                    message=f"history_t_raw_indices[{index}] must be int for valid slot",
                )
                return
            if (
                previous_valid_t_std is not None
                and int(t_std) - previous_valid_t_std != 1
            ):
                _record_anomaly(
                    checks,
                    "temporal_drift",
                    trace=trace,
                    message=(
                        f"history_t_std_indices stride drift at index {index}: "
                        f"{previous_valid_t_std} -> {t_std}"
                    ),
                )
                return
            if (
                previous_valid_t_raw is not None
                and int(t_raw) - previous_valid_t_raw != 1
            ):
                _record_anomaly(
                    checks,
                    "temporal_drift",
                    trace=trace,
                    message=(
                        f"history_t_raw_indices stride drift at index {index}: "
                        f"{previous_valid_t_raw} -> {t_raw}"
                    ),
                )
                return
            previous_valid_t_std = int(t_std)
            previous_valid_t_raw = int(t_raw)
        else:
            if (
                prev_action is not None
                or proprio is not None
                or visual is not None
                or ts is not None
            ):
                _record_anomaly(
                    checks,
                    "invalid_mask_anomalies",
                    trace=trace,
                    message=(
                        f"invalid history slot {index} carries payload despite history_valid_mask=false"
                    ),
                )
                return

    if source_t is not None:
        last_valid_index = valid_indices[-1]
        last_t_std = normalized_payloads["history_t_std_indices"][last_valid_index]
        last_t_raw = normalized_payloads["history_t_raw_indices"][last_valid_index]
        last_ts = normalized_payloads["history_timestamp_s"][last_valid_index]
        if int(last_t_std) != int(source_t) or int(last_t_raw) != int(source_t):
            _record_anomaly(
                checks,
                "temporal_drift",
                trace=trace,
                message=(
                    "last valid history index must land on source_t: "
                    f"std={last_t_std} raw={last_t_raw} source_t={source_t}"
                ),
            )
            return
        if float(last_ts) != float(source_t):
            _record_anomaly(
                checks,
                "temporal_drift",
                trace=trace,
                message=(
                    f"last history_timestamp_s must equal source_t: {last_ts} vs {source_t}"
                ),
            )


def _apply_negative_input(
    *,
    snapshot_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    label_rows: Sequence[Mapping[str, Any]],
    mode: str | None,
) -> tuple[
    list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None
]:
    snapshots = [dict(row) for row in snapshot_rows]
    manifest_copy = dict(manifest)
    manifest_copy["pseudodemos"] = [
        dict(_as_mapping(item, field_name="manifest.pseudodemos[]"))
        for item in list(manifest.get("pseudodemos", []))
    ]
    labels = [dict(row) for row in label_rows]
    if mode is None:
        return snapshots, manifest_copy, labels, None

    if mode == NEGATIVE_MODE_LABEL_NULL_TEACHER_TARGET:
        for row in labels:
            if (
                row.get("source_bucket")
                == state_conditioned_build_training_set.SOURCE_BUCKET_FORMAL_PSEUDODEMO
                and row.get("training_view")
                == state_conditioned_build_training_set.VIEW_C1
            ):
                corrupted = dict(row)
                corrupted["sample_id"] = f"{row['sample_id']}::deliberate_bad"
                corrupted["policy_condition.phase"] = (
                    state_conditioned_build_training_set.NULL_PHASE_TOKEN
                )
                corrupted["policy_condition.mode"] = (
                    state_conditioned_build_training_set.NULL_MODE_TOKEN
                )
                corrupted["policy_condition_text"] = (
                    state_conditioned_build_training_set.build_null_policy_condition_text()
                )
                corrupted["source_t"] = int(corrupted["source_t"])
                labels.append(corrupted)
                return (
                    snapshots,
                    manifest_copy,
                    labels,
                    {
                        "mode": mode,
                        "status": "injected",
                        "target": "training_labels",
                        "sample_id": corrupted["sample_id"],
                    },
                )
        raise ValueError(
            "could not inject deliberate bad label: no formal C1 row found"
        )

    if mode == NEGATIVE_MODE_SNAPSHOT_RESET_BOUNDARY:
        if not snapshots:
            raise ValueError(
                "could not inject deliberate bad snapshot: no snapshot rows found"
            )
        corrupted = dict(snapshots[0])
        episode_ids = list(corrupted.get("history_episode_ids", []))
        if len(episode_ids) != int(
            state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K
        ):
            raise ValueError(
                "could not inject deliberate bad snapshot: history_episode_ids has unexpected length"
            )
        episode_ids[-1] = f"{episode_ids[-1]}__cross_episode_corruption"
        corrupted["history_episode_ids"] = episode_ids
        snapshots = [corrupted, *snapshots[1:]]
        return (
            snapshots,
            manifest_copy,
            labels,
            {
                "mode": mode,
                "status": "injected",
                "target": "snapshot_candidates",
                "snapshot_id": corrupted.get("snapshot_id"),
            },
        )

    raise ValueError(f"unsupported deliberate-bad-sample mode: {mode!r}")


def materialize_pseudodemo_label_audit(
    *,
    snapshot_path: Path,
    manifest_path: Path,
    labels_path: Path,
    output_path: Path,
    deliberate_bad_sample: str | None = None,
) -> dict[str, Any]:
    raw_snapshot_rows = _read_jsonl_dicts(snapshot_path)
    raw_manifest = _read_json(manifest_path)
    raw_label_rows = _read_jsonl_dicts(labels_path)
    snapshot_rows, manifest, label_rows, negative_input_info = _apply_negative_input(
        snapshot_rows=raw_snapshot_rows,
        manifest=raw_manifest,
        label_rows=raw_label_rows,
        mode=deliberate_bad_sample,
    )

    checks: dict[str, dict[str, Any]] = {
        name: (
            _new_missing_field_check()
            if name == "missing_field_rate"
            else _new_check(name)
        )
        for name in CHECK_ORDER
    }

    snapshot_required_fields = (
        "snapshot_id",
        "family",
        "anchor_episode_id",
        "anchor_t",
        "history_k",
        "history_stride",
        "history_valid_mask",
        "history_episode_ids",
        "anchor_mujoco_state_ref",
        "prehistory_window",
        "reset_boundary",
        "policy_condition.phase",
        "policy_condition.mode",
        "policy_condition_text",
        "deployable.previous_action_history",
        "deployable.proprio_history",
        "deployable.short_visual_history_refs",
    )
    manifest_required_fields = tuple(
        recap_episode_writer.LOCAL_RECOVERY_PSEUDODEMO_REQUIRED_KEYS
    ) + (
        "policy_condition.phase",
        "policy_condition.mode",
        "policy_condition_text",
        "anchor_episode_id",
        "anchor_t",
    )
    label_required_fields = tuple(
        field_name
        for field_name in state_conditioned_build_training_set.SAFE_LABEL_FIELD_ORDER
        if field_name != "source_snapshot_id"
    )

    snapshot_index: dict[str, dict[str, Any]] = {}
    snapshot_history_index: dict[str, dict[str, Any]] = {}
    snapshot_trace_index: dict[str, dict[str, Any]] = {}
    snapshot_family_counter: Counter[str] = Counter()
    for raw_row in sorted(
        snapshot_rows,
        key=lambda item: str(item.get("snapshot_id", "")),
    ):
        row = dict(_as_mapping(raw_row, field_name="snapshot_candidates[]"))
        checks["family_source_mismatch"]["denominator"] = (
            int(checks["family_source_mismatch"].get("denominator", 0)) + 1
        )
        checks["temporal_drift"]["denominator"] = (
            int(checks["temporal_drift"].get("denominator", 0)) + 1
        )
        checks["history_payload_incompleteness"]["denominator"] = (
            int(checks["history_payload_incompleteness"].get("denominator", 0)) + 1
        )
        checks["reset_boundary_violations"]["denominator"] = (
            int(checks["reset_boundary_violations"].get("denominator", 0)) + 1
        )
        checks["invalid_mask_anomalies"]["denominator"] = (
            int(checks["invalid_mask_anomalies"].get("denominator", 0)) + 1
        )
        checks["nominal_recovery_confusion"]["denominator"] = (
            int(checks["nominal_recovery_confusion"].get("denominator", 0)) + 1
        )
        _record_missing_fields(
            checks,
            artifact_name="snapshot_candidates",
            record=row,
            required_fields=snapshot_required_fields,
        )
        trace = _safe_trace(row)
        snapshot_id = str(row.get("snapshot_id", "")).strip()
        family = str(row.get("family", "")).strip()
        if snapshot_id:
            if snapshot_id in snapshot_index:
                _record_anomaly(
                    checks,
                    "family_source_mismatch",
                    trace=trace,
                    message=f"duplicate snapshot_id observed: {snapshot_id}",
                )
            snapshot_index[snapshot_id] = row
            snapshot_trace_index[snapshot_id] = trace
            if family:
                snapshot_family_counter[family] += 1
        try:
            phase, mode, text = (
                state_conditioned_bucket_a_import.validate_state_conditioned_policy_condition(
                    phase=row.get("policy_condition.phase"),
                    mode=row.get("policy_condition.mode"),
                    policy_condition_text=row.get("policy_condition_text"),
                )
            )
            if mode != "RECOVERY":
                _record_anomaly(
                    checks,
                    "nominal_recovery_confusion",
                    trace=trace,
                    message=f"formal snapshot must be RECOVERY, got mode={mode!r}",
                    extra={"observed_policy_condition_text": text, "phase": phase},
                )
        except (TypeError, ValueError) as exc:
            _record_anomaly(
                checks,
                "nominal_recovery_confusion",
                trace=trace,
                message=f"snapshot policy_condition invalid: {exc}",
            )
        try:
            state_conditioned_bucket_a_import.validate_state_conditioned_history_contract(
                anchor_episode_id=row.get("anchor_episode_id"),
                history_episode_ids=row.get("history_episode_ids"),
                history_valid_mask=row.get("history_valid_mask"),
                anchor_mujoco_state_ref=row.get("anchor_mujoco_state_ref"),
                prehistory_window=row.get("prehistory_window"),
                history_k=row.get("history_k"),
                history_stride=row.get("history_stride"),
                reset_boundary=row.get("reset_boundary"),
            )
        except (TypeError, ValueError) as exc:
            message = str(exc)
            target_check = (
                "reset_boundary_violations"
                if "cross-episode" in message or "reset_boundary" in message
                else "history_payload_incompleteness"
            )
            _record_anomaly(checks, target_check, trace=trace, message=message)
        try:
            history_k = _as_int(row.get("history_k"), field_name="history_k")
            _validate_mask_and_history_payload(
                checks,
                record=row,
                history_k=history_k,
                history_valid_mask=_as_list(
                    row.get("history_valid_mask"),
                    field_name="history_valid_mask",
                    expected_len=history_k,
                ),
                history_t_std_indices=[
                    _as_int(
                        dict(item).get("t_std"),
                        field_name=f"prehistory_window[{index}].t_std",
                    )
                    for index, item in enumerate(
                        _as_list(
                            row.get("prehistory_window"),
                            field_name="prehistory_window",
                            expected_len=history_k,
                        )
                    )
                ],
                history_t_raw_indices=[
                    _as_int(
                        dict(item).get(
                            "t_raw",
                            dict(item).get("t_raw_index", dict(item).get("t_std")),
                        ),
                        field_name=f"prehistory_window[{index}].t_raw",
                    )
                    for index, item in enumerate(
                        _as_list(
                            row.get("prehistory_window"),
                            field_name="prehistory_window",
                            expected_len=history_k,
                        )
                    )
                ],
                history_timestamp_s=[
                    None
                    if not bool(
                        _as_list(
                            row.get("history_valid_mask"),
                            field_name="history_valid_mask",
                            expected_len=history_k,
                        )[index]
                    )
                    else float(
                        _as_int(
                            dict(item).get(
                                "t_raw",
                                dict(item).get("t_raw_index", dict(item).get("t_std")),
                            ),
                            field_name=f"prehistory_window[{index}].t_raw",
                        )
                    )
                    for index, item in enumerate(
                        _as_list(
                            row.get("prehistory_window"),
                            field_name="prehistory_window",
                            expected_len=history_k,
                        )
                    )
                ],
                previous_action_history=_as_list(
                    row.get("deployable.previous_action_history"),
                    field_name="deployable.previous_action_history",
                    expected_len=history_k,
                ),
                proprio_history=_as_list(
                    row.get("deployable.proprio_history"),
                    field_name="deployable.proprio_history",
                    expected_len=history_k,
                ),
                short_visual_history_refs=_as_list(
                    row.get("deployable.short_visual_history_refs"),
                    field_name="deployable.short_visual_history_refs",
                    expected_len=history_k,
                ),
                source_t=_as_int(row.get("anchor_t"), field_name="anchor_t"),
            )
            if snapshot_id:
                snapshot_history_index[snapshot_id] = _snapshot_history_signature(row)
        except (TypeError, ValueError) as exc:
            _record_anomaly(
                checks,
                "history_payload_incompleteness",
                trace=trace,
                message=f"snapshot history payload could not be normalized: {exc}",
            )

    manifest_pseudodemos = [
        dict(_as_mapping(item, field_name="manifest.pseudodemos[]"))
        for item in list(raw_manifest.get("pseudodemos", []))
    ]
    if not isinstance(manifest.get("pseudodemos"), list):
        raise ValueError(
            "local_recovery_pseudodemo_manifest.pseudodemos must be a list"
        )
    manifest_index: dict[tuple[str, str], dict[str, Any]] = {}
    manifest_episode_index: dict[str, dict[str, Any]] = {}
    manifest_family_counter: Counter[str] = Counter()
    teacher_producer_counter: Counter[str] = Counter()
    for raw_record in sorted(
        manifest_pseudodemos,
        key=lambda item: (
            str(item.get("source_snapshot_id", "")),
            str(item.get("episode_id", "")),
        ),
    ):
        record = dict(raw_record)
        checks["family_source_mismatch"]["denominator"] = (
            int(checks["family_source_mismatch"].get("denominator", 0)) + 1
        )
        checks["temporal_drift"]["denominator"] = (
            int(checks["temporal_drift"].get("denominator", 0)) + 1
        )
        checks["teacher_target_null_degradation"]["denominator"] = (
            int(checks["teacher_target_null_degradation"].get("denominator", 0)) + 1
        )
        checks["nominal_recovery_confusion"]["denominator"] = (
            int(checks["nominal_recovery_confusion"].get("denominator", 0)) + 1
        )
        _record_missing_fields(
            checks,
            artifact_name="pseudodemo_manifest",
            record=record,
            required_fields=manifest_required_fields,
        )
        trace = _safe_trace(record)
        episode_id = str(record.get("episode_id", "")).strip()
        source_snapshot_id = str(record.get("source_snapshot_id", "")).strip()
        source_snapshot_family = str(record.get("source_snapshot_family", "")).strip()
        producer = str(record.get("producer", "")).strip()
        if episode_id:
            manifest_episode_index[episode_id] = record
        if episode_id and source_snapshot_id:
            manifest_index[(episode_id, source_snapshot_id)] = record
        if source_snapshot_family:
            manifest_family_counter[source_snapshot_family] += 1
        if producer:
            teacher_producer_counter[producer] += 1

        try:
            recap_episode_writer.validate_local_recovery_pseudodemo_record(record)
        except (KeyError, TypeError, ValueError) as exc:
            _record_anomaly(
                checks,
                "teacher_target_null_degradation",
                trace=trace,
                message=f"manifest pseudodemo record invalid: {exc}",
            )

        snapshot_row = snapshot_index.get(source_snapshot_id)
        if snapshot_row is None:
            _record_anomaly(
                checks,
                "family_source_mismatch",
                trace=trace,
                message=f"manifest source_snapshot_id not found in snapshot candidates: {source_snapshot_id!r}",
            )
            continue

        snapshot_trace = snapshot_trace_index.get(source_snapshot_id, {})
        if source_snapshot_family != str(snapshot_row.get("family")):
            _record_anomaly(
                checks,
                "family_source_mismatch",
                trace={**snapshot_trace, **trace},
                message=(
                    "source_snapshot_family mismatch between manifest and snapshot candidates: "
                    f"manifest={source_snapshot_family!r} snapshot={snapshot_row.get('family')!r}"
                ),
            )

        anchor_episode_id = record.get("anchor_episode_id")
        anchor_t = record.get("anchor_t")
        if anchor_episode_id != snapshot_row.get("anchor_episode_id"):
            _record_anomaly(
                checks,
                "temporal_drift",
                trace={**snapshot_trace, **trace},
                message=(
                    "anchor_episode_id drift between manifest and snapshot: "
                    f"manifest={anchor_episode_id!r} snapshot={snapshot_row.get('anchor_episode_id')!r}"
                ),
            )
        if anchor_t != snapshot_row.get("anchor_t"):
            _record_anomaly(
                checks,
                "temporal_drift",
                trace={**snapshot_trace, **trace},
                message=(
                    "anchor_t drift between manifest and snapshot: "
                    f"manifest={anchor_t!r} snapshot={snapshot_row.get('anchor_t')!r}"
                ),
            )

        for prefix_name in ("failure_prefix", "recovery_suffix"):
            range_field = f"{prefix_name}_source_t_range"
            count_field = f"{prefix_name}_step_count"
            episode_field = f"{prefix_name}_source_episode_id"
            try:
                t_range = _as_list(
                    record.get(range_field), field_name=range_field, expected_len=2
                )
                start_t = _as_int(t_range[0], field_name=f"{range_field}[0]")
                end_t = _as_int(t_range[1], field_name=f"{range_field}[1]")
                step_count = _as_int(record.get(count_field), field_name=count_field)
                if int(end_t) - int(start_t) + 1 != int(step_count):
                    _record_anomaly(
                        checks,
                        "temporal_drift",
                        trace=trace,
                        message=(
                            f"{count_field} inconsistent with {range_field}: "
                            f"step_count={step_count} range={[start_t, end_t]}"
                        ),
                    )
                if record.get(episode_field) != episode_id:
                    _record_anomaly(
                        checks,
                        "family_source_mismatch",
                        trace=trace,
                        message=(
                            f"{episode_field} must match local rollout episode_id for pseudodemo traceability"
                        ),
                    )
            except (TypeError, ValueError) as exc:
                _record_anomaly(
                    checks,
                    "temporal_drift",
                    trace=trace,
                    message=f"{prefix_name} temporal summary invalid: {exc}",
                )

        observed_phase = record.get("policy_condition.phase")
        observed_mode = record.get("policy_condition.mode")
        observed_text = record.get("policy_condition_text")
        if (
            observed_phase is None
            or observed_mode is None
            or observed_text is None
            or (isinstance(observed_phase, str) and not observed_phase.strip())
            or (isinstance(observed_mode, str) and not observed_mode.strip())
            or (isinstance(observed_text, str) and not observed_text.strip())
        ):
            _record_anomaly(
                checks,
                "teacher_target_null_degradation",
                trace={**snapshot_trace, **trace},
                message="manifest policy_condition payload is null/empty for formal pseudodemo",
            )
        else:
            try:
                phase, mode, text = (
                    state_conditioned_bucket_a_import.validate_state_conditioned_policy_condition(
                        phase=observed_phase,
                        mode=observed_mode,
                        policy_condition_text=observed_text,
                    )
                )
                if mode != "RECOVERY":
                    _record_anomaly(
                        checks,
                        "nominal_recovery_confusion",
                        trace={**snapshot_trace, **trace},
                        message=f"manifest formal pseudodemo must stay RECOVERY, got {mode!r}",
                        extra={"phase": phase, "policy_condition_text": text},
                    )
                if (
                    phase != snapshot_row.get("policy_condition.phase")
                    or mode != snapshot_row.get("policy_condition.mode")
                    or text != snapshot_row.get("policy_condition_text")
                ):
                    _record_anomaly(
                        checks,
                        "nominal_recovery_confusion",
                        trace={**snapshot_trace, **trace},
                        message="manifest policy_condition drifted from snapshot canonical target",
                    )
            except (TypeError, ValueError) as exc:
                _record_anomaly(
                    checks,
                    "teacher_target_null_degradation",
                    trace={**snapshot_trace, **trace},
                    message=f"manifest policy_condition invalid: {exc}",
                )

    all_label_rows = [
        dict(_as_mapping(item, field_name="state_conditioned_sft_labels[]"))
        for item in label_rows
    ]
    label_view_counter: Counter[str] = Counter()
    label_source_bucket_counter: Counter[str] = Counter()
    sample_view_map: dict[str, set[str]] = {}
    formal_sample_view_map: dict[str, set[str]] = {}
    observed_formal_sample_ids: set[str] = set()
    overall_m2_rows_missing_any = 0
    overall_m2_rows_missing_all = 0
    formal_m2_rows_missing_all = 0
    formal_label_rows: list[dict[str, Any]] = []
    for raw_row in sorted(
        all_label_rows,
        key=lambda item: (
            str(item.get("sample_id", "")),
            str(item.get("training_view", "")),
        ),
    ):
        row = dict(raw_row)
        source_bucket = str(row.get("source_bucket", ""))
        training_view = str(row.get("training_view", ""))
        label_view_counter[training_view] += 1
        label_source_bucket_counter[source_bucket] += 1
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            sample_view_map.setdefault(sample_id, set()).add(training_view)
        checks["temporal_drift"]["denominator"] = (
            int(checks["temporal_drift"].get("denominator", 0)) + 1
        )
        checks["history_payload_incompleteness"]["denominator"] = (
            int(checks["history_payload_incompleteness"].get("denominator", 0)) + 1
        )
        checks["invalid_mask_anomalies"]["denominator"] = (
            int(checks["invalid_mask_anomalies"].get("denominator", 0)) + 1
        )
        checks["m2_label_missing_rate"]["denominator"] = (
            int(checks["m2_label_missing_rate"].get("denominator", 0)) + 1
        )
        missing_conditional_fields: tuple[str, ...] = ()
        if (
            source_bucket
            == state_conditioned_build_training_set.SOURCE_BUCKET_FORMAL_PSEUDODEMO
        ):
            formal_label_rows.append(row)
            checks["family_source_mismatch"]["denominator"] = (
                int(checks["family_source_mismatch"].get("denominator", 0)) + 1
            )
            checks["teacher_target_null_degradation"]["denominator"] = (
                int(checks["teacher_target_null_degradation"].get("denominator", 0)) + 1
            )
            checks["nominal_recovery_confusion"]["denominator"] = (
                int(checks["nominal_recovery_confusion"].get("denominator", 0)) + 1
            )
            checks["formal_label_coverage"]["denominator"] = (
                int(checks["formal_label_coverage"].get("denominator", 0)) + 1
            )
            missing_conditional_fields = ("source_snapshot_id",)
            if sample_id:
                formal_sample_view_map.setdefault(sample_id, set()).add(training_view)
                observed_formal_sample_ids.add(sample_id)
        _record_missing_fields(
            checks,
            artifact_name="training_labels",
            record=row,
            required_fields=label_required_fields,
            conditional_required_fields=missing_conditional_fields,
        )
        trace = _safe_trace(row)
        m2_missing_fields = [field for field in M2_EXPECTED_FIELDS if field not in row]
        if m2_missing_fields:
            overall_m2_rows_missing_any += 1
        if len(m2_missing_fields) == len(M2_EXPECTED_FIELDS):
            overall_m2_rows_missing_all += 1
            if (
                source_bucket
                == state_conditioned_build_training_set.SOURCE_BUCKET_FORMAL_PSEUDODEMO
            ):
                formal_m2_rows_missing_all += 1
            _record_anomaly(
                checks,
                "m2_label_missing_rate",
                trace=trace,
                message="training label row is missing all expected M2 label fields",
                extra={"missing_m2_fields": list(M2_EXPECTED_FIELDS)},
            )

        try:
            history_k = _as_int(row.get("history_k"), field_name="history_k")
            history_valid_mask = _as_list(
                row.get("history_valid_mask"),
                field_name="history_valid_mask",
                expected_len=history_k,
            )
            _validate_mask_and_history_payload(
                checks,
                record=row,
                history_k=history_k,
                history_valid_mask=history_valid_mask,
                history_t_std_indices=_as_list(
                    row.get("history_t_std_indices"),
                    field_name="history_t_std_indices",
                    expected_len=history_k,
                ),
                history_t_raw_indices=_as_list(
                    row.get("history_t_raw_indices"),
                    field_name="history_t_raw_indices",
                    expected_len=history_k,
                ),
                history_timestamp_s=_as_list(
                    row.get("history_timestamp_s"),
                    field_name="history_timestamp_s",
                    expected_len=history_k,
                ),
                previous_action_history=_as_list(
                    row.get("deployable.previous_action_history"),
                    field_name="deployable.previous_action_history",
                    expected_len=history_k,
                ),
                proprio_history=_as_list(
                    row.get("deployable.proprio_history"),
                    field_name="deployable.proprio_history",
                    expected_len=history_k,
                ),
                short_visual_history_refs=_as_list(
                    row.get("deployable.short_visual_history_refs"),
                    field_name="deployable.short_visual_history_refs",
                    expected_len=history_k,
                ),
                source_t=None
                if row.get("source_t") is None
                else _as_int(row.get("source_t"), field_name="source_t"),
            )
        except (TypeError, ValueError) as exc:
            _record_anomaly(
                checks,
                "history_payload_incompleteness",
                trace=trace,
                message=f"training label history payload invalid: {exc}",
            )

        try:
            state_conditioned_build_training_set.validate_view_policy_condition_text(
                training_view=training_view,
                phase=row.get("policy_condition.phase"),
                mode=row.get("policy_condition.mode"),
                policy_condition_text=row.get("policy_condition_text"),
            )
        except (TypeError, ValueError) as exc:
            target_check = (
                "teacher_target_null_degradation"
                if source_bucket
                == state_conditioned_build_training_set.SOURCE_BUCKET_FORMAL_PSEUDODEMO
                and training_view == state_conditioned_build_training_set.VIEW_C1
                else "nominal_recovery_confusion"
            )
            _record_anomaly(
                checks,
                target_check,
                trace=trace,
                message=f"training label policy_condition invalid: {exc}",
            )

        if (
            source_bucket
            != state_conditioned_build_training_set.SOURCE_BUCKET_FORMAL_PSEUDODEMO
        ):
            continue

        source_snapshot_id = str(row.get("source_snapshot_id", "")).strip()
        if not source_snapshot_id or source_snapshot_id not in snapshot_index:
            _record_anomaly(
                checks,
                "family_source_mismatch",
                trace=trace,
                message=(
                    "formal pseudodemo label is missing snapshot context or points to unknown snapshot_id"
                ),
            )
            continue
        snapshot_row = snapshot_index[source_snapshot_id]
        manifest_record = manifest_index.get(
            (str(row.get("source_episode_id")), source_snapshot_id)
        )
        if manifest_record is None:
            _record_anomaly(
                checks,
                "formal_label_coverage",
                trace={**snapshot_trace_index.get(source_snapshot_id, {}), **trace},
                message="formal label row has no matching pseudodemo manifest record",
            )
        source_t = row.get("source_t")
        if source_t != snapshot_row.get("anchor_t"):
            _record_anomaly(
                checks,
                "temporal_drift",
                trace={**snapshot_trace_index.get(source_snapshot_id, {}), **trace},
                message=(
                    "formal training label source_t drifted from snapshot anchor_t: "
                    f"label={source_t!r} snapshot={snapshot_row.get('anchor_t')!r}"
                ),
            )

        try:
            expected_history_signature = snapshot_history_index[source_snapshot_id]
            observed_history_signature = {
                "history_k": row.get("history_k"),
                "history_stride": row.get("history_stride"),
                "history_valid_mask": row.get("history_valid_mask"),
                "history_t_std_indices": row.get("history_t_std_indices"),
                "history_t_raw_indices": row.get("history_t_raw_indices"),
                "history_timestamp_s": row.get("history_timestamp_s"),
                "deployable.previous_action_history": row.get(
                    "deployable.previous_action_history"
                ),
                "deployable.proprio_history": row.get("deployable.proprio_history"),
                "deployable.short_visual_history_refs": row.get(
                    "deployable.short_visual_history_refs"
                ),
            }
            if observed_history_signature != expected_history_signature:
                _record_anomaly(
                    checks,
                    "temporal_drift",
                    trace={**snapshot_trace_index.get(source_snapshot_id, {}), **trace},
                    message="formal label history payload drifted from snapshot context",
                )
        except KeyError:
            _record_anomaly(
                checks,
                "history_payload_incompleteness",
                trace={**snapshot_trace_index.get(source_snapshot_id, {}), **trace},
                message="snapshot history signature missing for formal label linkage",
            )

        if training_view == state_conditioned_build_training_set.VIEW_C1:
            observed_phase = row.get("policy_condition.phase")
            observed_mode = row.get("policy_condition.mode")
            observed_text = row.get("policy_condition_text")
            if (
                observed_phase == state_conditioned_build_training_set.NULL_PHASE_TOKEN
                or observed_mode == state_conditioned_build_training_set.NULL_MODE_TOKEN
                or observed_text
                == state_conditioned_build_training_set.build_null_policy_condition_text()
            ):
                _record_anomaly(
                    checks,
                    "teacher_target_null_degradation",
                    trace={**snapshot_trace_index.get(source_snapshot_id, {}), **trace},
                    message="formal C1 row degraded to null teacher target",
                )
            elif (
                observed_phase != snapshot_row.get("policy_condition.phase")
                or observed_mode != snapshot_row.get("policy_condition.mode")
                or observed_text != snapshot_row.get("policy_condition_text")
            ):
                _record_anomaly(
                    checks,
                    "nominal_recovery_confusion",
                    trace={**snapshot_trace_index.get(source_snapshot_id, {}), **trace},
                    message="formal C1 policy_condition drifted from snapshot canonical recovery target",
                )
            elif observed_mode != "RECOVERY":
                _record_anomaly(
                    checks,
                    "nominal_recovery_confusion",
                    trace={**snapshot_trace_index.get(source_snapshot_id, {}), **trace},
                    message=f"formal C1 row must remain RECOVERY, got {observed_mode!r}",
                )
        elif training_view == state_conditioned_build_training_set.VIEW_C0:
            if (
                row.get("policy_condition.phase")
                != state_conditioned_build_training_set.NULL_PHASE_TOKEN
                or row.get("policy_condition.mode")
                != state_conditioned_build_training_set.NULL_MODE_TOKEN
                or row.get("policy_condition_text")
                != state_conditioned_build_training_set.build_null_policy_condition_text()
            ):
                _record_anomaly(
                    checks,
                    "nominal_recovery_confusion",
                    trace={**snapshot_trace_index.get(source_snapshot_id, {}), **trace},
                    message="formal C0 row leaked non-null recovery conditioning",
                )

    checks["formal_label_coverage"]["denominator"] = int(
        checks["formal_label_coverage"].get("denominator", 0)
    ) + int(len(manifest_pseudodemos))
    expected_formal_sample_ids = {
        (
            f"{state_conditioned_build_training_set.SOURCE_BUCKET_FORMAL_PSEUDODEMO}::"
            f"{record['episode_id']}:{record['source_snapshot_id']}::repeat{repeat_index}"
        )
        for record in manifest_pseudodemos
        for repeat_index in range(
            int(state_conditioned_build_training_set.RECOVERY_OVERSAMPLE_FACTOR)
        )
    }
    missing_formal_sample_ids = sorted(
        expected_formal_sample_ids - observed_formal_sample_ids
    )
    unexpected_formal_sample_ids = sorted(
        observed_formal_sample_ids - expected_formal_sample_ids
    )
    for sample_id in missing_formal_sample_ids[:EXAMPLE_LIMIT]:
        _record_anomaly(
            checks,
            "formal_label_coverage",
            trace={"sample_id": sample_id},
            message="manifest-backed formal sample_id missing from training labels",
        )
    for sample_id in unexpected_formal_sample_ids[:EXAMPLE_LIMIT]:
        _record_anomaly(
            checks,
            "formal_label_coverage",
            trace={"sample_id": sample_id},
            message="training labels contain unexpected formal sample_id",
        )
    for sample_id in sorted(formal_sample_view_map):
        observed_views = sorted(formal_sample_view_map[sample_id])
        if observed_views != [
            state_conditioned_build_training_set.VIEW_C0,
            state_conditioned_build_training_set.VIEW_C1,
        ]:
            _record_anomaly(
                checks,
                "formal_label_coverage",
                trace={"sample_id": sample_id, "observed_views": observed_views},
                message="formal sample_id does not have exactly paired C0/C1 rows",
            )

    checks["m2_label_missing_rate"]["overall_rows_missing_any_m2_fields"] = int(
        overall_m2_rows_missing_any
    )
    checks["m2_label_missing_rate"]["overall_rows_missing_all_m2_fields"] = int(
        overall_m2_rows_missing_all
    )
    checks["m2_label_missing_rate"]["formal_rows_missing_all_m2_fields"] = int(
        formal_m2_rows_missing_all
    )
    checks["m2_label_missing_rate"]["expected_m2_fields"] = list(M2_EXPECTED_FIELDS)

    _finalize_checks(checks)

    overall_verdict = (
        "PASS"
        if all(checks[name]["status"] == "PASS" for name in CHECK_ORDER)
        else "FAIL"
    )
    manifest_counts = dict(
        _as_mapping(manifest.get("counts", {}), field_name="manifest.counts")
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_pseudodemo_label_audit",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_verdict": overall_verdict,
        "input_paths": {
            "snapshot": str(snapshot_path),
            "manifest": str(manifest_path),
            "labels": str(labels_path),
            "output": str(output_path),
        },
        "negative_input": negative_input_info,
        "counts": {
            "snapshot_candidate_count": int(len(snapshot_rows)),
            "snapshot_family_counts": {
                key: snapshot_family_counter[key]
                for key in sorted(snapshot_family_counter)
            },
            "manifest_pseudodemo_count": int(len(manifest_pseudodemos)),
            "manifest_family_counts": {
                key: manifest_family_counter[key]
                for key in sorted(manifest_family_counter)
            },
            "manifest_producer_counts": {
                key: teacher_producer_counter[key]
                for key in sorted(teacher_producer_counter)
            },
            "manifest_successful_pseudodemo_count_reported": manifest_counts.get(
                "successful_pseudodemo_count"
            ),
            "label_row_count": int(len(all_label_rows)),
            "formal_label_row_count": int(len(formal_label_rows)),
            "label_view_counts": {
                key: label_view_counter[key] for key in sorted(label_view_counter)
            },
            "label_source_bucket_counts": {
                key: label_source_bucket_counter[key]
                for key in sorted(label_source_bucket_counter)
            },
            "paired_sample_id_count": int(len(sample_view_map)),
            "formal_paired_sample_id_count": int(len(formal_sample_view_map)),
            "expected_formal_sample_id_count": int(len(expected_formal_sample_ids)),
            "observed_formal_sample_id_count": int(len(observed_formal_sample_ids)),
        },
        "checks": {name: checks[name] for name in CHECK_ORDER},
    }
    _write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        snapshot_path = _validate_existing_file(args.snapshot, arg_name="snapshot")
        manifest_path = _validate_existing_file(args.manifest, arg_name="manifest")
        labels_path = _validate_existing_file(args.labels, arg_name="labels")
        output_path = _validate_output_path(args.output)
        materialize_pseudodemo_label_audit(
            snapshot_path=snapshot_path,
            manifest_path=manifest_path,
            labels_path=labels_path,
            output_path=output_path,
            deliberate_bad_sample=args.deliberate_bad_sample,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
