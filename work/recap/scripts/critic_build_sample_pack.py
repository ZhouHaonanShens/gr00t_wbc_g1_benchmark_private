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


SCHEMA_VERSION = "critic_sample_pack_v1"
ARTIFACT_KIND = "critic_sample_pack"
DEFAULT_DUMP_CSV = (
    critic_dump_predictions.DEFAULT_OUTPUT_DIR
    / critic_dump_predictions.COMBINED_CSV_NAME
)
DEFAULT_OUTPUT_JSON = (
    critic_dump_predictions.DEFAULT_OUTPUT_DIR / "critic_sample_pack_v1.json"
)
DEFAULT_SAMPLE_SPLIT = "test"
UNKNOWN_NOT_MATERIALIZED = "UNKNOWN / not materialized"


def _timestamp_now() -> str:
    return critic_dump_predictions.timestamp_now()


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return critic_dump_predictions.write_json(path, payload)


def _selection_payload(
    row: Mapping[str, object], *, category_name: str, selection_rank: int
) -> dict[str, object]:
    return {
        "pack_category": category_name,
        "selection_rank": int(selection_rank),
        "sample_id": critic_dump_predictions.row_str(row, "sample_id"),
        "split_name": critic_dump_predictions.row_str(row, "split_name"),
        "episode_index": critic_dump_predictions.row_int(row, "episode_index"),
        "recap_episode_id": critic_dump_predictions.row_str(row, "recap_episode_id"),
        "t": critic_dump_predictions.row_int(row, "t"),
        "frame_index": critic_dump_predictions.row_optional_int(row, "frame_index"),
        "prompt_raw": critic_dump_predictions.row_str(row, "prompt_raw"),
        "return_G": critic_dump_predictions.row_float(row, "return_G"),
        "pred_ev": critic_dump_predictions.row_float(row, "pred_ev"),
        "abs_error": critic_dump_predictions.row_float(row, "abs_error"),
        "error_signed": critic_dump_predictions.row_float(row, "error_signed"),
        "video_path": critic_dump_predictions.row_optional_str(row, "video_path"),
        "parquet_path": critic_dump_predictions.row_optional_str(row, "parquet_path"),
        "wrist_video_path": UNKNOWN_NOT_MATERIALIZED,
        "wrist_frame_index": UNKNOWN_NOT_MATERIALIZED,
    }


def _require_video_frame_linkage(
    row: Mapping[str, object], *, category_name: str
) -> None:
    if (
        not isinstance(row.get("video_path"), str)
        or not str(row.get("video_path")).strip()
    ):
        raise ValueError(
            f"sample pack requires video_path linkage for category {category_name}: {critic_dump_predictions.row_str(row, 'sample_id')}"
        )
    if row.get("frame_index") is None:
        raise ValueError(
            f"sample pack requires frame_index linkage for category {category_name}: {critic_dump_predictions.row_str(row, 'sample_id')}"
        )


def _take_unique_episode_rows(
    rows: Sequence[Mapping[str, object]], *, count: int
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen_episode_ids: set[str] = set()
    for row in rows:
        episode_id = critic_dump_predictions.row_str(row, "recap_episode_id")
        if episode_id in seen_episode_ids:
            continue
        selected.append(dict(row))
        seen_episode_ids.add(episode_id)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError(f"need at least {count} unique episodes, got {len(selected)}")
    return selected


def build_sample_pack(
    rows: Sequence[Mapping[str, object]],
    *,
    sample_split: str = DEFAULT_SAMPLE_SPLIT,
    source_dump_csv: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    ordered_rows = critic_dump_predictions.sort_canonical_rows(rows)
    critic_dump_predictions.require_all_splits(ordered_rows)
    split_rows = critic_dump_predictions.rows_for_split(ordered_rows, sample_split)

    good_rows = _take_unique_episode_rows(
        sorted(
            split_rows,
            key=lambda row: (
                critic_dump_predictions.row_float(row, "abs_error"),
                critic_dump_predictions.row_int(row, "episode_index"),
                critic_dump_predictions.row_int(row, "t"),
                critic_dump_predictions.row_str(row, "sample_id"),
            ),
        ),
        count=4,
    )
    overestimate_rows = [
        dict(row)
        for row in sorted(
            [
                row
                for row in split_rows
                if critic_dump_predictions.row_float(row, "error_signed") > 0.0
            ],
            key=lambda row: (
                -1.0 * critic_dump_predictions.row_float(row, "error_signed"),
                critic_dump_predictions.row_int(row, "episode_index"),
                critic_dump_predictions.row_int(row, "t"),
                critic_dump_predictions.row_str(row, "sample_id"),
            ),
        )[:4]
    ]
    underestimate_rows = [
        dict(row)
        for row in sorted(
            [
                row
                for row in split_rows
                if critic_dump_predictions.row_float(row, "error_signed") < 0.0
            ],
            key=lambda row: (
                critic_dump_predictions.row_float(row, "error_signed"),
                critic_dump_predictions.row_int(row, "episode_index"),
                critic_dump_predictions.row_int(row, "t"),
                critic_dump_predictions.row_str(row, "sample_id"),
            ),
        )[:4]
    ]
    if len(overestimate_rows) < 4:
        raise ValueError("need at least 4 overestimate rows in selected split")
    if len(underestimate_rows) < 4:
        raise ValueError("need at least 4 underestimate rows in selected split")

    for category_name, category_rows in (
        ("good", good_rows),
        ("overestimate", overestimate_rows),
        ("underestimate", underestimate_rows),
    ):
        for row in category_rows:
            _require_video_frame_linkage(row, category_name=category_name)

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "generated_at": generated_at or _timestamp_now(),
        "source_dump_csv": None if source_dump_csv is None else str(source_dump_csv),
        "sample_split": str(sample_split),
        "selection_policy": {
            "good": "4 lowest abs_error rows, unique by recap_episode_id, ordered by abs_error then episode/t/sample_id.",
            "overestimate": "4 largest positive pred_ev - return_G rows, ordered deterministically by signed error then episode/t/sample_id.",
            "underestimate": "4 largest positive return_G - pred_ev rows, ordered deterministically by signed error then episode/t/sample_id.",
        },
        "categories": {
            "good": [
                _selection_payload(row, category_name="good", selection_rank=index)
                for index, row in enumerate(good_rows, start=1)
            ],
            "overestimate": [
                _selection_payload(
                    row, category_name="overestimate", selection_rank=index
                )
                for index, row in enumerate(overestimate_rows, start=1)
            ],
            "underestimate": [
                _selection_payload(
                    row, category_name="underestimate", selection_rank=index
                )
                for index, row in enumerate(underestimate_rows, start=1)
            ],
        },
    }


def materialize_sample_pack(
    *,
    dump_csv: str | Path = DEFAULT_DUMP_CSV,
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
    sample_split: str = DEFAULT_SAMPLE_SPLIT,
    generated_at: str | None = None,
) -> dict[str, object]:
    resolved_dump_csv = critic_dump_predictions.resolve_path(dump_csv)
    rows = critic_dump_predictions.load_canonical_rows_csv(resolved_dump_csv)
    payload = build_sample_pack(
        rows,
        sample_split=sample_split,
        source_dump_csv=resolved_dump_csv,
        generated_at=generated_at,
    )
    _ = _write_json(critic_dump_predictions.resolve_path(output_json), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic 12-sample critic pack: 4 good, 4 overestimate, 4 underestimate."
    )
    _ = parser.add_argument("--dump-csv", type=Path, default=DEFAULT_DUMP_CSV)
    _ = parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    _ = parser.add_argument("--sample-split", type=str, default=DEFAULT_SAMPLE_SPLIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = materialize_sample_pack(
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
    "SCHEMA_VERSION",
    "UNKNOWN_NOT_MATERIALIZED",
    "build_parser",
    "build_sample_pack",
    "main",
    "materialize_sample_pack",
]


if __name__ == "__main__":
    raise SystemExit(main())
