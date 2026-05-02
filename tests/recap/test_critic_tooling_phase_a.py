from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import cast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import critic_build_episode_traces
from work.recap.scripts import critic_build_sample_pack
from work.recap.scripts import critic_dump_predictions
from work.recap.scripts import critic_scorecard_all_splits


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(critic_dump_predictions.CANONICAL_ROW_FIELDNAMES)
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: critic_dump_predictions.serialize_csv_value(row.get(field))
                    for field in critic_dump_predictions.CANONICAL_ROW_FIELDNAMES
                }
            )
    return path


def _row(
    *,
    sample_id: str,
    split_name: str,
    episode_index: int,
    recap_episode_id: str,
    t: int,
    t_norm: float,
    return_G: float,
    pred_ev: float,
    video_path: str | None = "/tmp/video.mp4",
    frame_index: int | None = 10,
    success: bool | None = None,
    trace_hint: str | None = None,
) -> dict[str, object]:
    error_signed = float(pred_ev - return_G)
    return {
        "sample_id": sample_id,
        "split_name": split_name,
        "episode_index": episode_index,
        "recap_episode_id": recap_episode_id,
        "episode_length": 3,
        "episode_t_max": 2,
        "local_index": t,
        "t": t,
        "t_norm": t_norm,
        "frame_index": frame_index,
        "prompt_raw": "pick up the apple and place it on the plate",
        "return_G": return_G,
        "pred_ev": pred_ev,
        "abs_error": abs(error_signed),
        "error_signed": error_signed,
        "video_path": video_path,
        "parquet_path": f"/tmp/{sample_id}.parquet",
        "success": success,
        "trace_hint": trace_hint,
    }


def _dump_input_row(
    *,
    sample_id: str,
    episode_index: int,
    recap_episode_id: str,
    t: int,
    t_norm: float,
    return_G: float,
    pred_ev: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "episode_index": episode_index,
        "recap_episode_id": recap_episode_id,
        "episode_length": 3,
        "episode_t_max": 2,
        "local_index": t,
        "t": t,
        "t_norm": t_norm,
        "frame_index": t * 15,
        "prompt_raw": "pick up the apple and place it on the plate",
        "return_G": return_G,
        "pred_ev": pred_ev,
        "video_path": f"/tmp/{sample_id}.mp4",
    }


def _all_split_rows() -> list[dict[str, object]]:
    rows = [
        _row(
            sample_id="train-early",
            split_name="train",
            episode_index=1,
            recap_episode_id="train_ep_001",
            t=0,
            t_norm=0.0,
            return_G=-10.0,
            pred_ev=-9.0,
        ),
        _row(
            sample_id="train-late",
            split_name="train",
            episode_index=1,
            recap_episode_id="train_ep_001",
            t=2,
            t_norm=1.0,
            return_G=-8.0,
            pred_ev=-7.5,
        ),
        _row(
            sample_id="val-middle",
            split_name="val",
            episode_index=2,
            recap_episode_id="val_ep_001",
            t=1,
            t_norm=0.5,
            return_G=-6.0,
            pred_ev=-6.2,
        ),
        _row(
            sample_id="val-late",
            split_name="val",
            episode_index=2,
            recap_episode_id="val_ep_001",
            t=2,
            t_norm=1.0,
            return_G=-5.0,
            pred_ev=-4.5,
        ),
    ]
    for index, abs_error in enumerate((0.10, 0.20, 0.30, 0.40), start=1):
        rows.append(
            _row(
                sample_id=f"good-{index}",
                split_name="test",
                episode_index=100 + index,
                recap_episode_id=f"good_ep_{index:03d}",
                t=1,
                t_norm=0.5,
                return_G=-10.0,
                pred_ev=-10.0 + abs_error,
                success=False,
            )
        )
    for index, signed_error in enumerate((50.0, 49.0, 48.0, 47.0), start=1):
        rows.append(
            _row(
                sample_id=f"over-{index}",
                split_name="test",
                episode_index=200 + index,
                recap_episode_id=f"over_ep_{index:03d}",
                t=0,
                t_norm=0.0,
                return_G=-100.0,
                pred_ev=-100.0 + signed_error,
                success=False,
            )
        )
    for index, signed_error in enumerate((-60.0, -59.0, -58.0, -57.0), start=1):
        rows.append(
            _row(
                sample_id=f"under-{index}",
                split_name="test",
                episode_index=300 + index,
                recap_episode_id=f"under_ep_{index:03d}",
                t=0,
                t_norm=0.0,
                return_G=-10.0,
                pred_ev=-10.0 + signed_error,
                success=False,
            )
        )
    rows.extend(
        [
            _row(
                sample_id="trace-success-0",
                split_name="test",
                episode_index=401,
                recap_episode_id="trace_success_ep",
                t=0,
                t_norm=0.0,
                return_G=-2.0,
                pred_ev=-4.0,
                success=True,
            ),
            _row(
                sample_id="trace-success-1",
                split_name="test",
                episode_index=401,
                recap_episode_id="trace_success_ep",
                t=1,
                t_norm=0.5,
                return_G=-1.0,
                pred_ev=-2.0,
                success=True,
            ),
            _row(
                sample_id="trace-success-2",
                split_name="test",
                episode_index=401,
                recap_episode_id="trace_success_ep",
                t=2,
                t_norm=1.0,
                return_G=0.0,
                pred_ev=-0.5,
                success=True,
            ),
            _row(
                sample_id="trace-late-0",
                split_name="test",
                episode_index=402,
                recap_episode_id="trace_late_fail_ep",
                t=0,
                t_norm=0.0,
                return_G=-15.0,
                pred_ev=-8.0,
                success=False,
            ),
            _row(
                sample_id="trace-late-1",
                split_name="test",
                episode_index=402,
                recap_episode_id="trace_late_fail_ep",
                t=1,
                t_norm=0.7,
                return_G=-14.0,
                pred_ev=-9.0,
                success=False,
            ),
            _row(
                sample_id="trace-late-2",
                split_name="test",
                episode_index=402,
                recap_episode_id="trace_late_fail_ep",
                t=2,
                t_norm=0.9,
                return_G=-13.0,
                pred_ev=-10.0,
                success=False,
            ),
            _row(
                sample_id="trace-early-0",
                split_name="test",
                episode_index=403,
                recap_episode_id="trace_early_fail_ep",
                t=0,
                t_norm=0.0,
                return_G=-30.0,
                pred_ev=-20.0,
                success=False,
            ),
            _row(
                sample_id="trace-early-1",
                split_name="test",
                episode_index=403,
                recap_episode_id="trace_early_fail_ep",
                t=1,
                t_norm=0.2,
                return_G=-29.0,
                pred_ev=-21.0,
                success=False,
            ),
            _row(
                sample_id="trace-early-2",
                split_name="test",
                episode_index=403,
                recap_episode_id="trace_early_fail_ep",
                t=2,
                t_norm=0.3,
                return_G=-28.0,
                pred_ev=-22.0,
                success=False,
            ),
        ]
    )
    return rows


def test_dump_predictions_and_scorecard_emit_all_splits_and_temporal_slices(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    train_json = _write_json(
        input_dir / "train.json",
        {
            "split_name": "train",
            "rows": [
                _dump_input_row(
                    sample_id="train-0",
                    episode_index=1,
                    recap_episode_id="train_ep_1",
                    t=0,
                    t_norm=0.0,
                    return_G=-10.0,
                    pred_ev=-9.0,
                ),
                _dump_input_row(
                    sample_id="train-1",
                    episode_index=1,
                    recap_episode_id="train_ep_1",
                    t=1,
                    t_norm=0.5,
                    return_G=-8.0,
                    pred_ev=-7.0,
                ),
            ],
        },
    )
    val_json = _write_json(
        input_dir / "val.json",
        {
            "split_name": "val",
            "rows": [
                _dump_input_row(
                    sample_id="val-0",
                    episode_index=2,
                    recap_episode_id="val_ep_1",
                    t=0,
                    t_norm=0.1,
                    return_G=-12.0,
                    pred_ev=-11.5,
                ),
                _dump_input_row(
                    sample_id="val-1",
                    episode_index=2,
                    recap_episode_id="val_ep_1",
                    t=2,
                    t_norm=0.9,
                    return_G=-9.0,
                    pred_ev=-9.2,
                ),
            ],
        },
    )
    test_json = _write_json(
        input_dir / "test.json",
        {
            "split_name": "test",
            "rows": [
                _dump_input_row(
                    sample_id="test-0",
                    episode_index=3,
                    recap_episode_id="test_ep_1",
                    t=0,
                    t_norm=0.0,
                    return_G=-20.0,
                    pred_ev=-19.0,
                ),
                _dump_input_row(
                    sample_id="test-1",
                    episode_index=3,
                    recap_episode_id="test_ep_1",
                    t=1,
                    t_norm=0.5,
                    return_G=-18.0,
                    pred_ev=-17.0,
                ),
                _dump_input_row(
                    sample_id="test-2",
                    episode_index=3,
                    recap_episode_id="test_ep_1",
                    t=2,
                    t_norm=1.0,
                    return_G=-16.0,
                    pred_ev=-15.0,
                ),
            ],
        },
    )

    dump_payload = critic_dump_predictions.materialize_prediction_dump(
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        output_dir=output_dir,
        generated_at="2026-04-12T00:00:00+00:00",
    )
    dump_csv = output_dir / critic_dump_predictions.COMBINED_CSV_NAME
    loaded_rows = critic_dump_predictions.load_canonical_rows_csv(dump_csv)

    assert dump_payload["schema_version"] == critic_dump_predictions.SCHEMA_VERSION
    assert dump_payload["row_fieldnames"] == list(
        critic_dump_predictions.CANONICAL_ROW_FIELDNAMES
    )
    assert [row["split_name"] for row in loaded_rows[:2]] == ["train", "train"]

    scorecard = critic_scorecard_all_splits.build_scorecard(
        loaded_rows,
        source_dump_csv=dump_csv,
        generated_at="2026-04-12T00:00:00+00:00",
    )
    by_split = cast(dict[str, object], scorecard["by_split"])
    overall = cast(dict[str, object], scorecard["overall"])
    temporal_slices = cast(dict[str, object], overall["temporal_slices"])
    early = cast(dict[str, object], temporal_slices["early"])
    middle = cast(dict[str, object], temporal_slices["middle"])
    late = cast(dict[str, object], temporal_slices["late"])
    early_metrics = cast(dict[str, object], early["metrics"])
    middle_metrics = cast(dict[str, object], middle["metrics"])
    late_metrics = cast(dict[str, object], late["metrics"])

    assert scorecard["schema_version"] == critic_scorecard_all_splits.SCHEMA_VERSION
    assert set(by_split.keys()) == {"train", "val", "test"}
    assert cast(int, early_metrics["row_count"]) > 0
    assert cast(int, middle_metrics["row_count"]) > 0
    assert cast(int, late_metrics["row_count"]) > 0


def test_missing_split_fails_closed(tmp_path: Path) -> None:
    rows = [row for row in _all_split_rows() if row["split_name"] != "val"]
    dump_csv = _write_csv(tmp_path / "score_rows.csv", rows)
    with pytest.raises(ValueError, match="missing required split"):
        _ = critic_dump_predictions.load_canonical_rows_csv(dump_csv)


def test_sample_pack_fails_closed_without_video_frame_linkage(tmp_path: Path) -> None:
    rows = _all_split_rows()
    for row in rows:
        if row["sample_id"] == "good-1":
            row["video_path"] = None
            row["frame_index"] = None
            break
    dump_csv = _write_csv(tmp_path / "score_rows.csv", rows)
    loaded_rows = critic_dump_predictions.load_canonical_rows_csv(dump_csv)
    with pytest.raises(ValueError, match="video_path linkage"):
        _ = critic_build_sample_pack.build_sample_pack(
            loaded_rows, source_dump_csv=dump_csv
        )


def test_sample_pack_selection_is_deterministic(tmp_path: Path) -> None:
    rows = _all_split_rows()
    dump_csv = _write_csv(tmp_path / "score_rows.csv", rows)
    loaded_rows = critic_dump_predictions.load_canonical_rows_csv(dump_csv)
    payload_a = critic_build_sample_pack.build_sample_pack(
        loaded_rows, source_dump_csv=dump_csv
    )
    payload_b = critic_build_sample_pack.build_sample_pack(
        list(reversed(loaded_rows)), source_dump_csv=dump_csv
    )
    categories_a = cast(dict[str, list[dict[str, object]]], payload_a["categories"])
    categories_b = cast(dict[str, list[dict[str, object]]], payload_b["categories"])

    assert categories_a == categories_b
    assert [sample["sample_id"] for sample in categories_a["good"]] == [
        "good-1",
        "good-2",
        "good-3",
        "good-4",
    ]
    assert [sample["sample_id"] for sample in categories_a["overestimate"]] == [
        "over-1",
        "over-2",
        "over-3",
        "over-4",
    ]
    assert [sample["sample_id"] for sample in categories_a["underestimate"]] == [
        "under-1",
        "under-2",
        "under-3",
        "under-4",
    ]


def test_episode_trace_selection_is_deterministic_and_uses_proxy_handling(
    tmp_path: Path,
) -> None:
    rows = _all_split_rows()
    dump_csv = _write_csv(tmp_path / "score_rows.csv", rows)
    loaded_rows = critic_dump_predictions.load_canonical_rows_csv(dump_csv)
    payload_a = critic_build_episode_traces.build_episode_traces(
        loaded_rows, source_dump_csv=dump_csv
    )
    payload_b = critic_build_episode_traces.build_episode_traces(
        list(reversed(loaded_rows)), source_dump_csv=dump_csv
    )
    categories = cast(list[dict[str, object]], payload_a["trace_categories"])

    assert payload_a == payload_b
    assert [entry["category_name"] for entry in categories] == [
        critic_build_episode_traces.SUCCESS_CATEGORY,
        critic_build_episode_traces.TRANSPORT_DROP_CATEGORY,
        critic_build_episode_traces.PRE_PLACE_CATEGORY,
    ]
    assert categories[0]["recap_episode_id"] == "trace_success_ep"
    assert categories[1]["recap_episode_id"] == "trace_late_fail_ep"
    assert categories[1]["selection_kind"] == "proxy"
    assert categories[2]["recap_episode_id"] == "trace_early_fail_ep"
    assert categories[2]["selection_kind"] == "proxy"
    assert (
        categories[1]["wrist_video_path"]
        == critic_build_episode_traces.UNKNOWN_NOT_MATERIALIZED
    )
    modality_availability = cast(
        dict[str, object], categories[2]["modality_availability"]
    )
    assert (
        modality_availability["wrist_view"]
        == critic_build_episode_traces.UNKNOWN_NOT_MATERIALIZED
    )
