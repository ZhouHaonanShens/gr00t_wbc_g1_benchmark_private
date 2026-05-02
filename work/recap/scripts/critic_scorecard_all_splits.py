from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import critic_dump_predictions


SCHEMA_VERSION = "critic_scorecard_all_splits_v1"
ARTIFACT_KIND = "critic_scorecard_all_splits"
DEFAULT_DUMP_CSV = (
    critic_dump_predictions.DEFAULT_OUTPUT_DIR
    / critic_dump_predictions.COMBINED_CSV_NAME
)
DEFAULT_OUTPUT_JSON = (
    critic_dump_predictions.DEFAULT_OUTPUT_DIR / "critic_scorecard_all_splits_v1.json"
)
TEMPORAL_SLICE_RANGES: tuple[tuple[str, tuple[float, float]], ...] = (
    ("early", (0.0, 0.2)),
    ("middle", (0.2, 0.8)),
    ("late", (0.8, 1.0)),
)


def _timestamp_now() -> str:
    return critic_dump_predictions.timestamp_now()


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return critic_dump_predictions.write_json(path, payload)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(float(value) for value in values) / float(len(values)))


def _rankdata(values: Sequence[float]) -> list[float]:
    indexed = sorted(
        enumerate(float(value) for value in values), key=lambda item: item[1]
    )
    ranks = [0.0] * len(indexed)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        avg_rank = (float(cursor + 1) + float(end)) / 2.0
        for idx in range(cursor, end):
            ranks[indexed[idx][0]] = avg_rank
        cursor = end
    return ranks


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    if x_mean is None or y_mean is None:
        return None
    cov = sum(
        (float(x) - float(x_mean)) * (float(y) - float(y_mean))
        for x, y in zip(x_values, y_values, strict=True)
    )
    x_var = sum((float(x) - float(x_mean)) ** 2 for x in x_values)
    y_var = sum((float(y) - float(y_mean)) ** 2 for y in y_values)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    return float(cov / math.sqrt(x_var * y_var))


def _spearman(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    return _pearson(_rankdata(x_values), _rankdata(y_values))


def _r2(pred_values: Sequence[float], target_values: Sequence[float]) -> float | None:
    if len(pred_values) != len(target_values) or not pred_values:
        return None
    target_mean = _mean(target_values)
    if target_mean is None:
        return None
    sse = sum(
        (float(pred) - float(target)) ** 2
        for pred, target in zip(pred_values, target_values, strict=True)
    )
    sst = sum((float(target) - float(target_mean)) ** 2 for target in target_values)
    if sst <= 0.0:
        return None
    return float(1.0 - (sse / sst))


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    pred_values = [critic_dump_predictions.row_float(row, "pred_ev") for row in rows]
    target_values = [critic_dump_predictions.row_float(row, "return_G") for row in rows]
    errors = [critic_dump_predictions.row_float(row, "error_signed") for row in rows]
    abs_errors = [critic_dump_predictions.row_float(row, "abs_error") for row in rows]
    mse = _mean([error * error for error in errors])
    return {
        "row_count": int(len(rows)),
        "episode_count": int(
            len(
                {
                    critic_dump_predictions.row_str(row, "recap_episode_id")
                    for row in rows
                }
            )
        ),
        "pred_ev_mean": _mean(pred_values),
        "return_G_mean": _mean(target_values),
        "mae": _mean(abs_errors),
        "rmse": None if mse is None else float(math.sqrt(mse)),
        "bias": _mean(errors),
        "pearson": _pearson(pred_values, target_values),
        "spearman": _spearman(pred_values, target_values),
        "r2": _r2(pred_values, target_values),
    }


def _slice_rows(
    rows: Sequence[Mapping[str, object]], *, lo: float, hi: float, name: str
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        t_norm = critic_dump_predictions.row_float(row, "t_norm")
        if name == "late":
            include = lo <= t_norm <= hi
        elif name == "middle":
            include = lo <= t_norm < hi
        else:
            include = lo <= t_norm <= hi
        if include:
            selected.append(dict(row))
    return selected


def _temporal_slices(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, (lo, hi) in TEMPORAL_SLICE_RANGES:
        slice_rows = _slice_rows(rows, lo=lo, hi=hi, name=name)
        payload[name] = {
            "t_norm_range": [float(lo), float(hi)],
            "metrics": _metrics(slice_rows),
        }
    return payload


def build_scorecard(
    rows: Sequence[Mapping[str, object]],
    *,
    source_dump_csv: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    ordered_rows = critic_dump_predictions.sort_canonical_rows(rows)
    critic_dump_predictions.require_all_splits(ordered_rows)
    by_split = {
        split_name: critic_dump_predictions.rows_for_split(ordered_rows, split_name)
        for split_name in critic_dump_predictions.REQUIRED_SPLITS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "generated_at": generated_at or _timestamp_now(),
        "source_dump_csv": None if source_dump_csv is None else str(source_dump_csv),
        "split_order": list(critic_dump_predictions.REQUIRED_SPLITS),
        "overall": {
            "metrics": _metrics(ordered_rows),
            "temporal_slices": _temporal_slices(ordered_rows),
        },
        "by_split": {
            split_name: {
                "metrics": _metrics(split_rows),
                "temporal_slices": _temporal_slices(split_rows),
            }
            for split_name, split_rows in by_split.items()
        },
    }


def materialize_scorecard(
    *,
    dump_csv: str | Path = DEFAULT_DUMP_CSV,
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
    generated_at: str | None = None,
) -> dict[str, object]:
    resolved_dump_csv = critic_dump_predictions.resolve_path(dump_csv)
    rows = critic_dump_predictions.load_canonical_rows_csv(resolved_dump_csv)
    payload = build_scorecard(
        rows, source_dump_csv=resolved_dump_csv, generated_at=generated_at
    )
    _ = _write_json(critic_dump_predictions.resolve_path(output_json), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute unified all-splits critic metrics and early/middle/late temporal slices."
    )
    _ = parser.add_argument("--dump-csv", type=Path, default=DEFAULT_DUMP_CSV)
    _ = parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = materialize_scorecard(
            dump_csv=cast(Path, args.dump_csv), output_json=cast(Path, args.output_json)
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
    "SCHEMA_VERSION",
    "TEMPORAL_SLICE_RANGES",
    "build_parser",
    "build_scorecard",
    "main",
    "materialize_scorecard",
]


if __name__ == "__main__":
    raise SystemExit(main())
