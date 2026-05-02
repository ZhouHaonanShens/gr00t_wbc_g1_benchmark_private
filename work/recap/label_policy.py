from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any


from .text_indicator import CANONICAL_NEGATIVE_LINE
from .text_indicator import CANONICAL_POSITIVE_LINE
from .text_indicator import RECAP_TEXT_INDICATOR_CARRIER_FIELD
from .text_indicator import TEXT_INDICATOR_NEGATIVE
from .text_indicator import TEXT_INDICATOR_OMIT
from .text_indicator import TEXT_INDICATOR_POSITIVE


LABEL_DASHBOARD_SCHEMA_VERSION = "gr00t_label_dashboard_v1"
LABEL_DASHBOARD_ARTIFACT_KIND = "gr00t_label_dashboard"
LABEL_POLICY_SCHEMA_VERSION = "gr00t_label_policy_v1"
LABEL_POLICY_ARTIFACT_KIND = "gr00t_label_policy"
LABEL_POLICY_EXTENSION_KEY = "label_policy"

LABEL_DASHBOARD_JSON_NAME = "gr00t_label_dashboard.json"
LABEL_POLICY_JSON_NAME = "gr00t_label_policy.json"

TASK_SURFACE_FIELD = RECAP_TEXT_INDICATOR_CARRIER_FIELD
PHASE_FIELD = "policy_condition.phase"
MODE_FIELD = "policy_condition.mode"
TRAINING_VIEW_FIELD = "training_view"
SAMPLE_ID_FIELD = "sample_id"
SOURCE_SAMPLE_KEY_FIELD = "source_sample_key"
EPSILON_FIELD = "epsilon_l"
INDICATOR_FIELD = "indicator_I"
REPEAT_INDEX_FIELD = "repeat_index"
RECOVERY_OVERSAMPLE_FACTOR_FIELD = "recovery_oversample_factor"


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    text = value.strip()
    return text or None


def _coerce_indicator(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        indicator = 1 if value else 0
    elif isinstance(value, int):
        indicator = int(value)
    elif isinstance(value, float):
        if not float(value).is_integer():
            raise ValueError(f"{field_name} must be 0 or 1, got {value!r}")
        indicator = int(value)
    elif isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"1", "1.0", "true", TEXT_INDICATOR_POSITIVE}:
            indicator = 1
        elif stripped in {
            "0",
            "0.0",
            "false",
            TEXT_INDICATOR_NEGATIVE,
            TEXT_INDICATOR_OMIT,
        }:
            indicator = 0
        else:
            raise ValueError(f"{field_name} must be 0/1-like, got {value!r}")
    else:
        raise TypeError(f"{field_name} must be int-like, got {type(value).__name__}")
    if indicator not in (0, 1):
        raise ValueError(f"{field_name} must be 0 or 1, got {indicator!r}")
    return indicator


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a float-like number")
    return float(value)


def _coerce_non_negative_int(
    value: object,
    *,
    field_name: str,
    default: int,
) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    if int(value) < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value!r}")
    return int(value)


def strip_indicator_suffix_from_carrier_text(carrier_text: object) -> str:
    normalized = _require_non_empty_string(carrier_text, field_name=TASK_SURFACE_FIELD)
    for suffix in (
        CANONICAL_POSITIVE_LINE,
        CANONICAL_NEGATIVE_LINE,
    ):
        marker = "\n" + suffix
        if normalized.endswith(marker):
            base_task = normalized[: -len(marker)]
            return _require_non_empty_string(
                base_task,
                field_name=f"{TASK_SURFACE_FIELD}_base_task",
            )
    return normalized


def indicator_mode_from_carrier_text(carrier_text: object) -> str:
    normalized = _require_non_empty_string(carrier_text, field_name=TASK_SURFACE_FIELD)
    if normalized.endswith("\n" + CANONICAL_POSITIVE_LINE):
        return TEXT_INDICATOR_POSITIVE
    if normalized.endswith("\n" + CANONICAL_NEGATIVE_LINE):
        return TEXT_INDICATOR_NEGATIVE
    return TEXT_INDICATOR_OMIT


def _dedupe_sorted_strings(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value).strip()})


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("numeric summary requires at least one value")
    normalized = [float(value) for value in values]
    distinct_values = sorted({float(value) for value in normalized})
    return {
        "count": int(len(normalized)),
        "min": float(min(normalized)),
        "max": float(max(normalized)),
        "mean": float(sum(normalized) / float(len(normalized))),
        "distinct_count": int(len(distinct_values)),
        "distinct_values": distinct_values,
        "all_equal": len(distinct_values) == 1,
        "constant_value": distinct_values[0] if len(distinct_values) == 1 else None,
    }


def _int_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise ValueError("int summary requires at least one value")
    normalized = [int(value) for value in values]
    distinct_values = sorted({int(value) for value in normalized})
    return {
        "count": int(len(normalized)),
        "min": int(min(normalized)),
        "max": int(max(normalized)),
        "distinct_count": int(len(distinct_values)),
        "distinct_values": distinct_values,
        "all_equal": len(distinct_values) == 1,
        "constant_value": distinct_values[0] if len(distinct_values) == 1 else None,
    }


def _collapse_checks(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = int(len(rows))
    positive_count = int(sum(int(row[INDICATOR_FIELD]) for row in rows))
    negative_count = int(total - positive_count)
    tasks = {str(row["task"]) for row in rows}
    phases = {str(row["phase"]) for row in rows}
    task_phase_pairs = {(str(row["task"]), str(row["phase"])) for row in rows}
    epsilon_values = {float(row[EPSILON_FIELD]) for row in rows}
    return {
        "all_positive_indicator": total > 0 and positive_count == total,
        "all_negative_indicator": total > 0 and negative_count == total,
        "mixed_indicator": positive_count > 0 and negative_count > 0,
        "single_task_bucket": len(tasks) == 1,
        "single_phase_bucket": len(phases) == 1,
        "single_task_phase_bucket": len(task_phase_pairs) == 1,
        "epsilon_collapsed": len(epsilon_values) == 1,
    }


def _normalize_label_rows(
    label_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(label_rows):
        row = _require_mapping(raw_row, field_name=f"label_rows[{row_index}]")
        carrier_text = _require_non_empty_string(
            row.get(TASK_SURFACE_FIELD),
            field_name=f"label_rows[{row_index}].{TASK_SURFACE_FIELD}",
        )
        task = strip_indicator_suffix_from_carrier_text(carrier_text)
        normalized_rows.append(
            {
                TASK_SURFACE_FIELD: carrier_text,
                "task": task,
                "carrier_indicator_mode": indicator_mode_from_carrier_text(
                    carrier_text
                ),
                "phase": _require_non_empty_string(
                    row.get(PHASE_FIELD),
                    field_name=f"label_rows[{row_index}].{PHASE_FIELD}",
                ),
                "mode": _require_non_empty_string(
                    row.get(MODE_FIELD),
                    field_name=f"label_rows[{row_index}].{MODE_FIELD}",
                ),
                TRAINING_VIEW_FIELD: _optional_string(row.get(TRAINING_VIEW_FIELD)),
                SAMPLE_ID_FIELD: _optional_string(row.get(SAMPLE_ID_FIELD)),
                SOURCE_SAMPLE_KEY_FIELD: _optional_string(
                    row.get(SOURCE_SAMPLE_KEY_FIELD)
                ),
                EPSILON_FIELD: _coerce_float(
                    row.get(EPSILON_FIELD),
                    field_name=f"label_rows[{row_index}].{EPSILON_FIELD}",
                ),
                INDICATOR_FIELD: _coerce_indicator(
                    row.get(INDICATOR_FIELD),
                    field_name=f"label_rows[{row_index}].{INDICATOR_FIELD}",
                ),
                REPEAT_INDEX_FIELD: _coerce_non_negative_int(
                    row.get(REPEAT_INDEX_FIELD),
                    field_name=f"label_rows[{row_index}].{REPEAT_INDEX_FIELD}",
                    default=0,
                ),
                RECOVERY_OVERSAMPLE_FACTOR_FIELD: _coerce_non_negative_int(
                    row.get(RECOVERY_OVERSAMPLE_FACTOR_FIELD),
                    field_name=(
                        f"label_rows[{row_index}].{RECOVERY_OVERSAMPLE_FACTOR_FIELD}"
                    ),
                    default=1,
                ),
            }
        )
    if not normalized_rows:
        raise ValueError("label_rows must contain at least one label row")
    return normalized_rows


def _summary_for_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row_count = int(len(rows))
    positive_count = int(sum(int(row[INDICATOR_FIELD]) for row in rows))
    negative_count = int(row_count - positive_count)
    return {
        "row_count": row_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_ratio": float(positive_count / float(row_count)),
        "negative_ratio": float(negative_count / float(row_count)),
        "carrier_text_variants": _dedupe_sorted_strings(
            str(row[TASK_SURFACE_FIELD]) for row in rows
        ),
        "distinct_phase_values": _dedupe_sorted_strings(
            str(row["phase"]) for row in rows
        ),
        "distinct_mode_values": _dedupe_sorted_strings(
            str(row["mode"]) for row in rows
        ),
        "distinct_training_views": _dedupe_sorted_strings(
            str(row[TRAINING_VIEW_FIELD])
            for row in rows
            if row.get(TRAINING_VIEW_FIELD) is not None
        ),
        "epsilon_summary": _numeric_summary(
            [float(row[EPSILON_FIELD]) for row in rows]
        ),
        "recovery_oversample_factor_summary": _int_summary(
            [int(row[RECOVERY_OVERSAMPLE_FACTOR_FIELD]) for row in rows]
        ),
        "repeat_index_summary": _int_summary(
            [int(row[REPEAT_INDEX_FIELD]) for row in rows]
        ),
        "duplicated_extra_row_count": int(
            sum(1 for row in rows if int(row[REPEAT_INDEX_FIELD]) > 0)
        ),
        "collapse_checks": _collapse_checks(rows),
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
) -> list[tuple[tuple[str, ...], list[dict[str, Any]]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field_name]) for field_name in key_fields)
        grouped[key].append(dict(row))
    return sorted(grouped.items(), key=lambda item: item[0])


def _source_artifacts(
    *,
    stats: Mapping[str, Any] | None,
    labels_jsonl_path: Path | None,
    stats_json_path: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "labels_jsonl_path": str(labels_jsonl_path)
        if labels_jsonl_path is not None
        else None,
        "stats_json_path": str(stats_json_path)
        if stats_json_path is not None
        else None,
    }
    if stats is not None:
        payload["stats_schema_version"] = stats.get("schema_version")
        payload["stats_artifact_kind"] = stats.get("artifact_kind")
        counts = stats.get("counts")
        if isinstance(counts, Mapping):
            payload["stats_counts"] = dict(counts)
    return payload


def build_label_dashboard(
    *,
    label_rows: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any] | None = None,
    labels_jsonl_path: Path | None = None,
    stats_json_path: Path | None = None,
) -> dict[str, Any]:
    normalized_rows = _normalize_label_rows(label_rows)
    task_groups = _group_rows(normalized_rows, key_fields=("task",))
    phase_groups = _group_rows(normalized_rows, key_fields=("phase",))
    task_phase_groups = _group_rows(normalized_rows, key_fields=("task", "phase"))
    overall_summary = _summary_for_rows(normalized_rows)
    overall_summary["task_bucket_count"] = int(len(task_groups))
    overall_summary["phase_bucket_count"] = int(len(phase_groups))
    overall_summary["task_phase_bucket_count"] = int(len(task_phase_groups))
    return {
        "schema_version": LABEL_DASHBOARD_SCHEMA_VERSION,
        "artifact_kind": LABEL_DASHBOARD_ARTIFACT_KIND,
        "task_grouping_authority": {
            "task_surface_field": TASK_SURFACE_FIELD,
            "task_grouping_key": "carrier_text_v1_without_indicator_suffix",
            "task_indicator_suffix_lines": [
                CANONICAL_NEGATIVE_LINE,
                CANONICAL_POSITIVE_LINE,
            ],
            "phase_field": PHASE_FIELD,
            "mode_field": MODE_FIELD,
            "epsilon_field": EPSILON_FIELD,
            "indicator_field": INDICATOR_FIELD,
        },
        "source_artifacts": _source_artifacts(
            stats=stats,
            labels_jsonl_path=labels_jsonl_path,
            stats_json_path=stats_json_path,
        ),
        "summaries": {
            "overall": overall_summary,
            "per_task": [
                {
                    "task": task,
                    **_summary_for_rows(group_rows),
                }
                for (task,), group_rows in task_groups
            ],
            "per_phase": [
                {
                    "phase": phase,
                    **_summary_for_rows(group_rows),
                }
                for (phase,), group_rows in phase_groups
            ],
            "per_task_phase": [
                {
                    "task": task,
                    "phase": phase,
                    **_summary_for_rows(group_rows),
                }
                for (task, phase), group_rows in task_phase_groups
            ],
        },
    }


def _epsilon_view_item(
    *,
    rows: Sequence[Mapping[str, Any]],
    extra_fields: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _summary_for_rows(rows)
    return {
        **dict(extra_fields),
        "row_count": int(summary["row_count"]),
        "positive_count": int(summary["positive_count"]),
        "negative_count": int(summary["negative_count"]),
        "positive_ratio": float(summary["positive_ratio"]),
        "epsilon_summary": dict(summary["epsilon_summary"]),
        "collapse_checks": dict(summary["collapse_checks"]),
    }


def _build_positive_duplication_policy(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    duplicated_rows = [row for row in rows if int(row[REPEAT_INDEX_FIELD]) > 0]
    duplicated_positive_rows = [
        row for row in duplicated_rows if int(row[INDICATOR_FIELD]) == 1
    ]
    duplicated_negative_rows = [
        row for row in duplicated_rows if int(row[INDICATOR_FIELD]) == 0
    ]
    distinct_factors = sorted(
        {
            int(row[RECOVERY_OVERSAMPLE_FACTOR_FIELD])
            for row in duplicated_positive_rows
            if int(row[RECOVERY_OVERSAMPLE_FACTOR_FIELD]) > 0
        }
    )
    return {
        "policy_name": "positive_duplication_control_v1",
        "enabled": bool(duplicated_positive_rows),
        "policy_source_fields": {
            "factor_field": RECOVERY_OVERSAMPLE_FACTOR_FIELD,
            "repeat_index_field": REPEAT_INDEX_FIELD,
            "indicator_field": INDICATOR_FIELD,
            "phase_field": PHASE_FIELD,
            "mode_field": MODE_FIELD,
        },
        "factor": int(max(distinct_factors)) if distinct_factors else 1,
        "distinct_factor_values": distinct_factors,
        "target_condition": {
            "indicator_I": 1,
            "phase_values": _dedupe_sorted_strings(
                str(row["phase"]) for row in duplicated_positive_rows
            ),
            "mode_values": _dedupe_sorted_strings(
                str(row["mode"]) for row in duplicated_positive_rows
            ),
            "task_values": _dedupe_sorted_strings(
                str(row["task"]) for row in duplicated_positive_rows
            ),
        },
        "observed_duplicate_row_count": int(len(duplicated_rows)),
        "observed_positive_duplicate_row_count": int(len(duplicated_positive_rows)),
        "observed_negative_duplicate_row_count": int(len(duplicated_negative_rows)),
        "duplicate_rows_only_positive": bool(duplicated_rows)
        and len(duplicated_negative_rows) == 0,
    }


def build_label_policy(
    *,
    label_rows: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any] | None = None,
    labels_jsonl_path: Path | None = None,
    stats_json_path: Path | None = None,
) -> dict[str, Any]:
    normalized_rows = _normalize_label_rows(label_rows)
    task_groups = _group_rows(normalized_rows, key_fields=("task",))
    phase_groups = _group_rows(normalized_rows, key_fields=("phase",))
    task_phase_groups = _group_rows(normalized_rows, key_fields=("task", "phase"))
    overall_summary = _summary_for_rows(normalized_rows)
    payload = {
        "schema_version": LABEL_POLICY_SCHEMA_VERSION,
        "artifact_kind": LABEL_POLICY_ARTIFACT_KIND,
        "task_grouping_authority": {
            "task_surface_field": TASK_SURFACE_FIELD,
            "task_grouping_key": "carrier_text_v1_without_indicator_suffix",
            "phase_field": PHASE_FIELD,
            "mode_field": MODE_FIELD,
        },
        "source_artifacts": _source_artifacts(
            stats=stats,
            labels_jsonl_path=labels_jsonl_path,
            stats_json_path=stats_json_path,
        ),
        "policy_source_fields": {
            "epsilon_field": EPSILON_FIELD,
            "indicator_field": INDICATOR_FIELD,
            "duplication_factor_field": RECOVERY_OVERSAMPLE_FACTOR_FIELD,
            "duplication_repeat_index_field": REPEAT_INDEX_FIELD,
            "task_surface_field": TASK_SURFACE_FIELD,
            "phase_field": PHASE_FIELD,
            "mode_field": MODE_FIELD,
        },
        "overall_epsilon_summary": dict(overall_summary["epsilon_summary"]),
        "task_aware_epsilon_view": [
            _epsilon_view_item(rows=group_rows, extra_fields={"task": task})
            for (task,), group_rows in task_groups
        ],
        "phase_aware_epsilon_view": [
            _epsilon_view_item(rows=group_rows, extra_fields={"phase": phase})
            for (phase,), group_rows in phase_groups
        ],
        "task_phase_aware_epsilon_view": [
            _epsilon_view_item(
                rows=group_rows,
                extra_fields={"task": task, "phase": phase},
            )
            for (task, phase), group_rows in task_phase_groups
        ],
        "positive_duplication_policy": _build_positive_duplication_policy(
            normalized_rows
        ),
        "collapse_checks": dict(overall_summary["collapse_checks"]),
    }
    return normalize_label_policy_payload(payload)


def normalize_label_policy_payload(
    payload: object,
    *,
    field_name: str = LABEL_POLICY_EXTENSION_KEY,
) -> dict[str, Any]:
    mapping = _require_mapping(payload, field_name=field_name)
    schema_version = mapping.get("schema_version")
    if schema_version != LABEL_POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"{field_name}.schema_version must equal {LABEL_POLICY_SCHEMA_VERSION!r}"
        )
    artifact_kind = mapping.get("artifact_kind")
    if artifact_kind != LABEL_POLICY_ARTIFACT_KIND:
        raise ValueError(
            f"{field_name}.artifact_kind must equal {LABEL_POLICY_ARTIFACT_KIND!r}"
        )
    for required_field in (
        "task_grouping_authority",
        "source_artifacts",
        "policy_source_fields",
        "overall_epsilon_summary",
        "task_aware_epsilon_view",
        "phase_aware_epsilon_view",
        "task_phase_aware_epsilon_view",
        "positive_duplication_policy",
        "collapse_checks",
    ):
        if required_field not in mapping:
            raise ValueError(f"{field_name}.{required_field} is required")
    for mapping_field in (
        "task_grouping_authority",
        "source_artifacts",
        "policy_source_fields",
        "overall_epsilon_summary",
        "positive_duplication_policy",
        "collapse_checks",
    ):
        _require_mapping(
            mapping.get(mapping_field), field_name=f"{field_name}.{mapping_field}"
        )
    for list_field in (
        "task_aware_epsilon_view",
        "phase_aware_epsilon_view",
        "task_phase_aware_epsilon_view",
    ):
        value = mapping.get(list_field)
        if not isinstance(value, list):
            raise TypeError(f"{field_name}.{list_field} must be a list")
    return dict(mapping)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"expected JSON object line at {path}:{line_number}, got {type(payload).__name__}"
                )
            records.append(dict(payload))
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(
            payload, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path.replace(path)


def write_label_dashboard_sidecars(
    *,
    labels_jsonl_path: Path,
    stats_json_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    resolved_labels_path = Path(labels_jsonl_path).expanduser().resolve()
    resolved_stats_path = Path(stats_json_path).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    label_rows = _read_jsonl(resolved_labels_path)
    stats = _read_json(resolved_stats_path)
    dashboard_payload = build_label_dashboard(
        label_rows=label_rows,
        stats=stats,
        labels_jsonl_path=resolved_labels_path,
        stats_json_path=resolved_stats_path,
    )
    label_policy_payload = build_label_policy(
        label_rows=label_rows,
        stats=stats,
        labels_jsonl_path=resolved_labels_path,
        stats_json_path=resolved_stats_path,
    )
    dashboard_path = _write_json(
        resolved_output_dir / LABEL_DASHBOARD_JSON_NAME,
        dashboard_payload,
    )
    label_policy_path = _write_json(
        resolved_output_dir / LABEL_POLICY_JSON_NAME,
        label_policy_payload,
    )
    return {
        "label_dashboard_path": str(dashboard_path),
        "label_policy_path": str(label_policy_path),
        "label_row_count": int(len(label_rows)),
    }


__all__ = [
    "LABEL_DASHBOARD_ARTIFACT_KIND",
    "LABEL_DASHBOARD_JSON_NAME",
    "LABEL_DASHBOARD_SCHEMA_VERSION",
    "LABEL_POLICY_ARTIFACT_KIND",
    "LABEL_POLICY_EXTENSION_KEY",
    "LABEL_POLICY_JSON_NAME",
    "LABEL_POLICY_SCHEMA_VERSION",
    "build_label_dashboard",
    "build_label_policy",
    "indicator_mode_from_carrier_text",
    "normalize_label_policy_payload",
    "strip_indicator_suffix_from_carrier_text",
    "write_label_dashboard_sidecars",
]
