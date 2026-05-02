from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REQUIRED_SPLITS: tuple[str, ...] = ("train", "val", "test")
SPLIT_ORDER = {name: index for index, name in enumerate(REQUIRED_SPLITS)}

SCHEMA_VERSION = "critic_prediction_dump_v1"
ARTIFACT_KIND = "critic_prediction_dump"
COMBINED_CSV_NAME = "score_rows_all_splits_v1.csv"
MANIFEST_JSON_NAME = "critic_prediction_dump_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "agent/artifacts/vlm_critic_scorecard_all_splits"

CANONICAL_ROW_FIELDNAMES: tuple[str, ...] = (
    "sample_id",
    "split_name",
    "episode_index",
    "recap_episode_id",
    "episode_length",
    "episode_t_max",
    "local_index",
    "t",
    "t_norm",
    "frame_index",
    "prompt_raw",
    "return_G",
    "pred_ev",
    "abs_error",
    "error_signed",
    "video_path",
    "parquet_path",
    "success",
    "trace_hint",
)


CanonicalRow = dict[str, object]


def timestamp_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload_raw: object = json.load(handle)
    if not isinstance(payload_raw, dict):
        raise ValueError(
            f"Expected JSON object in {path}, got {type(payload_raw).__name__}"
        )
    return {str(key): value for key, value in payload_raw.items()}


def write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        _ = handle.write("\n")
    _ = tmp_path.replace(path)
    return path


def serialize_csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANONICAL_ROW_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: serialize_csv_value(row.get(field))
                    for field in CANONICAL_ROW_FIELDNAMES
                }
            )
    _ = tmp_path.replace(path)
    return path


def _require_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected object for {context}, got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _as_non_empty_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for {context}, got {value!r}")
    return str(value)


def _as_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    normalized = str(value).strip()
    return normalized or None


def _as_int(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Expected int-like value for {context}, got bool")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"Expected integer-valued float for {context}, got {value!r}"
            )
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(
        f"Expected int-like value for {context}, got {type(value).__name__}"
    )


def _as_optional_int(value: object, *, context: str) -> int | None:
    if value in (None, ""):
        return None
    return _as_int(value, context=context)


def _as_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected float-like value for {context}, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(
        f"Expected float-like value for {context}, got {type(value).__name__}"
    )


def _as_optional_bool(value: object, *, context: str) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Expected bool-like value for {context}, got {value!r}")


def _pick_first(mapping: Mapping[str, object], keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in mapping:
            candidate = mapping.get(key)
            if candidate not in (None, ""):
                return candidate
    return None


def row_str(row: Mapping[str, object], key: str, *, default: str | None = None) -> str:
    value = row.get(key, default)
    if value is None:
        if default is not None:
            return str(default)
        raise ValueError(f"missing required string field: {key}")
    return _as_non_empty_string(value, context=f"row.{key}")


def row_optional_str(row: Mapping[str, object], key: str) -> str | None:
    return _as_optional_string(row.get(key))


def row_int(row: Mapping[str, object], key: str, *, default: int | None = None) -> int:
    value = row.get(key, default)
    if value is None:
        if default is not None:
            return int(default)
        raise ValueError(f"missing required int field: {key}")
    return _as_int(value, context=f"row.{key}")


def row_optional_int(row: Mapping[str, object], key: str) -> int | None:
    return _as_optional_int(row.get(key), context=f"row.{key}")


def row_float(
    row: Mapping[str, object], key: str, *, default: float | None = None
) -> float:
    value = row.get(key, default)
    if value is None:
        if default is not None:
            return float(default)
        raise ValueError(f"missing required float field: {key}")
    return _as_float(value, context=f"row.{key}")


def row_optional_bool(row: Mapping[str, object], key: str) -> bool | None:
    return _as_optional_bool(row.get(key), context=f"row.{key}")


def _split_rows_from_payload(
    payload: Mapping[str, object], *, path: Path
) -> list[dict[str, object]]:
    rows_raw: object = payload.get("rows")
    if rows_raw is None:
        rows_raw = payload.get("records")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError(f"rows must be a non-empty list in {path}")
    rows: list[dict[str, object]] = []
    for index, item in enumerate(list(rows_raw)):
        rows.append(_require_mapping(item, context=f"{path}.rows[{index}]"))
    return rows


def _normalize_prediction_row(
    raw_row: Mapping[str, object], *, split_name: str, row_index: int
) -> CanonicalRow:
    context = f"{split_name}[{row_index}]"
    video_mapping = raw_row.get("video")
    nested_video: dict[str, object] = (
        {str(key): value for key, value in video_mapping.items()}
        if isinstance(video_mapping, Mapping)
        else {}
    )

    pred_ev = _as_float(
        _pick_first(
            raw_row, ("pred_ev", "predicted_value", "value_V_raw", "full_score")
        ),
        context=f"{context}.pred_ev",
    )
    return_g = _as_float(raw_row.get("return_G"), context=f"{context}.return_G")
    episode_index = _as_int(
        _pick_first(raw_row, ("episode_index",)), context=f"{context}.episode_index"
    )
    t = _as_int(_pick_first(raw_row, ("t", "timestep")), context=f"{context}.t")
    episode_t_max = _as_optional_int(
        _pick_first(raw_row, ("episode_t_max",)), context=f"{context}.episode_t_max"
    )
    episode_length = _as_optional_int(
        _pick_first(raw_row, ("episode_length", "n_policy_steps")),
        context=f"{context}.episode_length",
    )
    local_index = _as_optional_int(
        raw_row.get("local_index"), context=f"{context}.local_index"
    )

    resolved_episode_t_max = episode_t_max
    resolved_episode_length = episode_length
    if resolved_episode_t_max is None and resolved_episode_length is None:
        resolved_episode_t_max = int(t)
        resolved_episode_length = int(t) + 1
    elif resolved_episode_t_max is None:
        assert resolved_episode_length is not None
        resolved_episode_t_max = max(0, int(resolved_episode_length) - 1)
    elif resolved_episode_length is None:
        resolved_episode_length = int(resolved_episode_t_max) + 1
    assert resolved_episode_t_max is not None
    assert resolved_episode_length is not None

    t_norm_raw = _pick_first(raw_row, ("t_norm",))
    if t_norm_raw is None:
        t_norm = (
            0.0
            if int(resolved_episode_t_max) <= 0
            else float(t) / float(resolved_episode_t_max)
        )
    else:
        t_norm = _as_float(t_norm_raw, context=f"{context}.t_norm")

    video_path = _as_optional_string(
        _pick_first(raw_row, ("video_path", "video_rel", "video_abs"))
        or _pick_first(nested_video, ("video_path", "video_rel", "video_abs"))
    )
    parquet_path = _as_optional_string(
        _pick_first(raw_row, ("parquet_path", "parquet_rel"))
    )
    frame_index = _as_optional_int(
        _pick_first(raw_row, ("frame_index",)), context=f"{context}.frame_index"
    )
    success = _as_optional_bool(
        _pick_first(raw_row, ("success", "success_episode")),
        context=f"{context}.success",
    )
    trace_hint = _as_optional_string(
        _pick_first(
            raw_row, ("trace_hint", "trace_category", "failure_cause", "episode_tag")
        )
    )

    error_signed = float(pred_ev - return_g)
    return {
        "sample_id": _as_non_empty_string(
            raw_row.get("sample_id"), context=f"{context}.sample_id"
        ),
        "split_name": str(split_name),
        "episode_index": int(episode_index),
        "recap_episode_id": _as_non_empty_string(
            _pick_first(raw_row, ("recap_episode_id", "episode_id")),
            context=f"{context}.recap_episode_id",
        ),
        "episode_length": int(resolved_episode_length),
        "episode_t_max": int(resolved_episode_t_max),
        "local_index": int(t if local_index is None else local_index),
        "t": int(t),
        "t_norm": float(t_norm),
        "frame_index": frame_index,
        "prompt_raw": _as_non_empty_string(
            _pick_first(raw_row, ("prompt_raw", "task_text")),
            context=f"{context}.prompt_raw",
        ),
        "return_G": float(return_g),
        "pred_ev": float(pred_ev),
        "abs_error": abs(error_signed),
        "error_signed": float(error_signed),
        "video_path": video_path,
        "parquet_path": parquet_path,
        "success": success,
        "trace_hint": trace_hint,
    }


def row_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    split_name = str(row.get("split_name"))
    return (
        SPLIT_ORDER.get(split_name, len(REQUIRED_SPLITS)),
        row_int(row, "episode_index", default=0),
        str(row.get("recap_episode_id", "")),
        row_int(row, "t", default=0),
        str(row.get("sample_id", "")),
    )


def sort_canonical_rows(rows: Sequence[Mapping[str, object]]) -> list[CanonicalRow]:
    return [dict(row) for row in sorted(rows, key=row_sort_key)]


def require_all_splits(rows: Sequence[Mapping[str, object]]) -> None:
    present = {str(row.get("split_name")) for row in rows}
    missing = [
        split_name for split_name in REQUIRED_SPLITS if split_name not in present
    ]
    if missing:
        raise ValueError("missing required split(s): " + ", ".join(missing))


def _normalize_split_payload(path: Path, *, expected_split: str) -> list[CanonicalRow]:
    payload = _read_json(path)
    split_name = _as_non_empty_string(
        payload.get("split_name"), context=f"{path}.split_name"
    )
    if split_name != expected_split:
        raise ValueError(
            f"split_name mismatch for {path}: expected {expected_split!r}, got {split_name!r}"
        )
    rows = _split_rows_from_payload(payload, path=path)
    return [
        _normalize_prediction_row(row, split_name=expected_split, row_index=index)
        for index, row in enumerate(rows)
    ]


def load_canonical_rows_csv(
    path: str | Path, *, require_complete_splits: bool = True
) -> list[CanonicalRow]:
    resolved_path = resolve_path(path)
    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != list(CANONICAL_ROW_FIELDNAMES):
            raise ValueError(
                f"unexpected CSV fieldnames in {resolved_path}: {reader.fieldnames!r}"
            )
        rows: list[CanonicalRow] = []
        for index, raw_row in enumerate(reader):
            raw_row_dict = cast(dict[str, str], raw_row)
            split_name = _as_non_empty_string(
                raw_row_dict.get("split_name"),
                context=f"{resolved_path}[{index}].split_name",
            )
            rows.append(
                {
                    "sample_id": _as_non_empty_string(
                        raw_row_dict.get("sample_id"),
                        context=f"{resolved_path}[{index}].sample_id",
                    ),
                    "split_name": split_name,
                    "episode_index": _as_int(
                        raw_row_dict.get("episode_index"),
                        context=f"{resolved_path}[{index}].episode_index",
                    ),
                    "recap_episode_id": _as_non_empty_string(
                        raw_row_dict.get("recap_episode_id"),
                        context=f"{resolved_path}[{index}].recap_episode_id",
                    ),
                    "episode_length": _as_int(
                        raw_row_dict.get("episode_length"),
                        context=f"{resolved_path}[{index}].episode_length",
                    ),
                    "episode_t_max": _as_int(
                        raw_row_dict.get("episode_t_max"),
                        context=f"{resolved_path}[{index}].episode_t_max",
                    ),
                    "local_index": _as_int(
                        raw_row_dict.get("local_index"),
                        context=f"{resolved_path}[{index}].local_index",
                    ),
                    "t": _as_int(
                        raw_row_dict.get("t"), context=f"{resolved_path}[{index}].t"
                    ),
                    "t_norm": _as_float(
                        raw_row_dict.get("t_norm"),
                        context=f"{resolved_path}[{index}].t_norm",
                    ),
                    "frame_index": _as_optional_int(
                        raw_row_dict.get("frame_index"),
                        context=f"{resolved_path}[{index}].frame_index",
                    ),
                    "prompt_raw": _as_non_empty_string(
                        raw_row_dict.get("prompt_raw"),
                        context=f"{resolved_path}[{index}].prompt_raw",
                    ),
                    "return_G": _as_float(
                        raw_row_dict.get("return_G"),
                        context=f"{resolved_path}[{index}].return_G",
                    ),
                    "pred_ev": _as_float(
                        raw_row_dict.get("pred_ev"),
                        context=f"{resolved_path}[{index}].pred_ev",
                    ),
                    "abs_error": _as_float(
                        raw_row_dict.get("abs_error"),
                        context=f"{resolved_path}[{index}].abs_error",
                    ),
                    "error_signed": _as_float(
                        raw_row_dict.get("error_signed"),
                        context=f"{resolved_path}[{index}].error_signed",
                    ),
                    "video_path": _as_optional_string(raw_row_dict.get("video_path")),
                    "parquet_path": _as_optional_string(
                        raw_row_dict.get("parquet_path")
                    ),
                    "success": _as_optional_bool(
                        raw_row_dict.get("success"),
                        context=f"{resolved_path}[{index}].success",
                    ),
                    "trace_hint": _as_optional_string(raw_row_dict.get("trace_hint")),
                }
            )
    if require_complete_splits:
        require_all_splits(rows)
    return sort_canonical_rows(rows)


def rows_for_split(
    rows: Sequence[Mapping[str, object]], split_name: str
) -> list[CanonicalRow]:
    selected = [dict(row) for row in rows if str(row.get("split_name")) == split_name]
    if not selected:
        raise ValueError(f"missing rows for split {split_name!r}")
    return sort_canonical_rows(selected)


def group_rows_by_episode(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[CanonicalRow]]:
    grouped: dict[str, list[CanonicalRow]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("recap_episode_id")), []).append(dict(row))
    for episode_rows in grouped.values():
        episode_rows.sort(key=row_sort_key)
    return grouped


def materialize_prediction_dump(
    *,
    train_json: str | Path,
    val_json: str | Path,
    test_json: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    generated_at: str | None = None,
) -> dict[str, object]:
    output_root = resolve_path(output_dir)
    split_inputs = {
        "train": resolve_path(train_json),
        "val": resolve_path(val_json),
        "test": resolve_path(test_json),
    }
    rows_by_split = {
        split_name: _normalize_split_payload(path, expected_split=split_name)
        for split_name, path in split_inputs.items()
    }
    combined_rows = sort_canonical_rows(
        [row for split_rows in rows_by_split.values() for row in split_rows]
    )
    require_all_splits(combined_rows)

    combined_csv_path = _write_rows_csv(output_root / COMBINED_CSV_NAME, combined_rows)
    split_csv_paths = {
        split_name: _write_rows_csv(
            output_root / f"{split_name}_rows_v1.csv", split_rows
        )
        for split_name, split_rows in rows_by_split.items()
    }

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "generated_at": generated_at or timestamp_now(),
        "split_order": list(REQUIRED_SPLITS),
        "row_fieldnames": list(CANONICAL_ROW_FIELDNAMES),
        "input_paths": {
            split_name: str(path) for split_name, path in split_inputs.items()
        },
        "row_count": int(len(combined_rows)),
        "split_summaries": {
            split_name: {
                "row_count": int(len(split_rows)),
                "episode_count": int(
                    len({row_str(row, "recap_episode_id") for row in split_rows})
                ),
            }
            for split_name, split_rows in rows_by_split.items()
        },
        "artifacts": {
            "combined_csv": str(combined_csv_path),
            **{
                f"{split_name}_csv": str(path)
                for split_name, path in split_csv_paths.items()
            },
        },
    }
    _ = write_json(output_root / MANIFEST_JSON_NAME, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize train/val/test critic prediction fixtures into a stable all-splits CSV surface."
        )
    )
    _ = parser.add_argument("--train-json", type=Path, required=True)
    _ = parser.add_argument("--val-json", type=Path, required=True)
    _ = parser.add_argument("--test-json", type=Path, required=True)
    _ = parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = materialize_prediction_dump(
            train_json=cast(Path, args.train_json),
            val_json=cast(Path, args.val_json),
            test_json=cast(Path, args.test_json),
            output_dir=cast(Path, args.output_dir),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ARTIFACT_KIND",
    "CANONICAL_ROW_FIELDNAMES",
    "COMBINED_CSV_NAME",
    "DEFAULT_OUTPUT_DIR",
    "MANIFEST_JSON_NAME",
    "REQUIRED_SPLITS",
    "SCHEMA_VERSION",
    "build_parser",
    "group_rows_by_episode",
    "load_canonical_rows_csv",
    "main",
    "materialize_prediction_dump",
    "require_all_splits",
    "resolve_path",
    "row_float",
    "row_int",
    "row_optional_bool",
    "row_optional_int",
    "row_optional_str",
    "row_sort_key",
    "row_str",
    "rows_for_split",
    "serialize_csv_value",
    "sort_canonical_rows",
    "timestamp_now",
    "write_json",
]


if __name__ == "__main__":
    raise SystemExit(main())
