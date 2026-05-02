from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import critic_dump_predictions


SCHEMA_VERSION = "critic_episode_traces_v1"
ARTIFACT_KIND = "critic_episode_traces"
DEFAULT_DUMP_CSV = (
    critic_dump_predictions.DEFAULT_OUTPUT_DIR
    / critic_dump_predictions.COMBINED_CSV_NAME
)
DEFAULT_OUTPUT_JSON = (
    critic_dump_predictions.DEFAULT_OUTPUT_DIR / "critic_episode_traces_v1.json"
)
DEFAULT_SAMPLE_SPLIT = "test"
SUCCESS_CATEGORY = "success"
TRANSPORT_DROP_CATEGORY = "transport-drop-or-proxy"
PRE_PLACE_CATEGORY = "pre-place-or-proxy"
UNKNOWN_NOT_MATERIALIZED = "UNKNOWN / not materialized"
TRANSPORT_PROXY_MIN_T_NORM = 0.66


def _timestamp_now() -> str:
    return critic_dump_predictions.timestamp_now()


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return critic_dump_predictions.write_json(path, payload)


def _normalize_hint(raw_hint: object) -> str | None:
    if not isinstance(raw_hint, str) or not raw_hint.strip():
        return None
    return raw_hint.strip().lower().replace("_", "-")


def _episode_success(rows: Sequence[Mapping[str, object]]) -> bool:
    explicit_values = [
        row.get("success") for row in rows if row.get("success") is not None
    ]
    if explicit_values:
        return bool(explicit_values[-1])
    return critic_dump_predictions.row_float(rows[-1], "return_G") >= 0.0


def _episode_label(rows: Sequence[Mapping[str, object]]) -> tuple[str, str, str]:
    if _episode_success(rows):
        return SUCCESS_CATEGORY, "inferred", "success_episode"
    normalized_hints = [_normalize_hint(row.get("trace_hint")) for row in rows]
    hints = [hint for hint in normalized_hints if hint is not None]
    if any("transport" in hint or "drop" in hint for hint in hints):
        return TRANSPORT_DROP_CATEGORY, "explicit", "explicit_trace_hint"
    if any("pre-place" in hint or "preplace" in hint for hint in hints):
        return PRE_PLACE_CATEGORY, "explicit", "explicit_trace_hint"
    terminal_t_norm = max(
        critic_dump_predictions.row_float(row, "t_norm") for row in rows
    )
    if terminal_t_norm >= TRANSPORT_PROXY_MIN_T_NORM:
        return (
            TRANSPORT_DROP_CATEGORY,
            "proxy",
            f"late_failure_proxy_t_norm_gte_{TRANSPORT_PROXY_MIN_T_NORM:.2f}",
        )
    return (
        PRE_PLACE_CATEGORY,
        "proxy",
        f"early_failure_proxy_t_norm_lt_{TRANSPORT_PROXY_MIN_T_NORM:.2f}",
    )


def _episode_selection_key(episode_payload: Mapping[str, object]) -> tuple[object, ...]:
    selection_kind = str(episode_payload["selection_kind"])
    return (
        0 if selection_kind == "explicit" else 1,
        0 if selection_kind == "inferred" else 1,
        -1 * critic_dump_predictions.row_int(episode_payload, "trace_row_count"),
        critic_dump_predictions.row_int(episode_payload, "episode_index"),
        critic_dump_predictions.row_str(episode_payload, "recap_episode_id"),
    )


def _build_episode_trace_payload(
    rows: Sequence[Mapping[str, object]],
    *,
    category_name: str,
    selection_kind: str,
    selection_reason: str,
) -> dict[str, object]:
    sorted_rows = critic_dump_predictions.sort_canonical_rows(rows)
    ego_video_path = next(
        (
            critic_dump_predictions.row_str(row, "video_path")
            for row in sorted_rows
            if isinstance(row.get("video_path"), str)
            and str(row.get("video_path")).strip()
        ),
        UNKNOWN_NOT_MATERIALIZED,
    )
    success_episode = _episode_success(sorted_rows)
    trace_rows = []
    last_index = len(sorted_rows) - 1
    for index, row in enumerate(sorted_rows):
        trace_rows.append(
            {
                "timestep_index": int(index),
                "t": critic_dump_predictions.row_int(row, "t"),
                "t_norm": critic_dump_predictions.row_float(row, "t_norm"),
                "return_G": critic_dump_predictions.row_float(row, "return_G"),
                "pred_ev": critic_dump_predictions.row_float(row, "pred_ev"),
                "abs_error": critic_dump_predictions.row_float(row, "abs_error"),
                "frame_index": critic_dump_predictions.row_optional_int(
                    row, "frame_index"
                ),
                "done": bool(index == last_index),
                "success": bool(success_episode),
                "failure_cause": "success"
                if success_episode
                else str(selection_reason),
            }
        )
    return {
        "category_name": str(category_name),
        "selection_kind": str(selection_kind),
        "selection_reason": str(selection_reason),
        "split_name": critic_dump_predictions.row_str(sorted_rows[0], "split_name"),
        "episode_index": critic_dump_predictions.row_int(
            sorted_rows[0], "episode_index"
        ),
        "recap_episode_id": critic_dump_predictions.row_str(
            sorted_rows[0], "recap_episode_id"
        ),
        "trace_row_count": int(len(trace_rows)),
        "ego_video_path": ego_video_path,
        "wrist_video_path": UNKNOWN_NOT_MATERIALIZED,
        "modality_availability": {
            "ego_view": "materialized via video_path",
            "wrist_view": UNKNOWN_NOT_MATERIALIZED,
        },
        "trace_rows": trace_rows,
    }


def build_episode_traces(
    rows: Sequence[Mapping[str, object]],
    *,
    sample_split: str = DEFAULT_SAMPLE_SPLIT,
    source_dump_csv: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    ordered_rows = critic_dump_predictions.sort_canonical_rows(rows)
    critic_dump_predictions.require_all_splits(ordered_rows)
    split_rows = critic_dump_predictions.rows_for_split(ordered_rows, sample_split)
    episodes = critic_dump_predictions.group_rows_by_episode(split_rows)
    categorized: dict[str, list[dict[str, object]]] = {
        SUCCESS_CATEGORY: [],
        TRANSPORT_DROP_CATEGORY: [],
        PRE_PLACE_CATEGORY: [],
    }
    for episode_rows in episodes.values():
        category_name, selection_kind, selection_reason = _episode_label(episode_rows)
        categorized[category_name].append(
            _build_episode_trace_payload(
                episode_rows,
                category_name=category_name,
                selection_kind=selection_kind,
                selection_reason=selection_reason,
            )
        )
    selected_traces: list[dict[str, object]] = []
    for category_name in (
        SUCCESS_CATEGORY,
        TRANSPORT_DROP_CATEGORY,
        PRE_PLACE_CATEGORY,
    ):
        candidates = sorted(categorized[category_name], key=_episode_selection_key)
        if not candidates:
            raise ValueError(
                f"missing episode candidate for category {category_name!r}"
            )
        selected_traces.append(candidates[0])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "generated_at": generated_at or _timestamp_now(),
        "source_dump_csv": None if source_dump_csv is None else str(source_dump_csv),
        "sample_split": str(sample_split),
        "trace_categories": selected_traces,
    }


def materialize_episode_traces(
    *,
    dump_csv: str | Path = DEFAULT_DUMP_CSV,
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
    sample_split: str = DEFAULT_SAMPLE_SPLIT,
    generated_at: str | None = None,
) -> dict[str, object]:
    resolved_dump_csv = critic_dump_predictions.resolve_path(dump_csv)
    rows = critic_dump_predictions.load_canonical_rows_csv(resolved_dump_csv)
    payload = build_episode_traces(
        rows,
        sample_split=sample_split,
        source_dump_csv=resolved_dump_csv,
        generated_at=generated_at,
    )
    _ = _write_json(critic_dump_predictions.resolve_path(output_json), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit deterministic success / transport-drop-or-proxy / pre-place-or-proxy critic traces."
    )
    _ = parser.add_argument("--dump-csv", type=Path, default=DEFAULT_DUMP_CSV)
    _ = parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    _ = parser.add_argument("--sample-split", type=str, default=DEFAULT_SAMPLE_SPLIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = materialize_episode_traces(
            dump_csv=cast(Path, args.dump_csv),
            output_json=cast(Path, args.output_json),
            sample_split=cast(str, args.sample_split),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ARTIFACT_KIND",
    "DEFAULT_DUMP_CSV",
    "DEFAULT_OUTPUT_JSON",
    "DEFAULT_SAMPLE_SPLIT",
    "PRE_PLACE_CATEGORY",
    "SCHEMA_VERSION",
    "SUCCESS_CATEGORY",
    "TRANSPORT_DROP_CATEGORY",
    "TRANSPORT_PROXY_MIN_T_NORM",
    "UNKNOWN_NOT_MATERIALIZED",
    "build_episode_traces",
    "build_parser",
    "main",
    "materialize_episode_traces",
]


if __name__ == "__main__":
    raise SystemExit(main())
