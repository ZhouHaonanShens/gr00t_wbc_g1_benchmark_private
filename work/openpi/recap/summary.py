from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import cast

from .checkpoint import read_json
from .protocol import (
    FrozenComparisonManifest,
    PAIRED_SUMMARY_SCHEMA_VERSION,
    RECAP_ONLY_VARIANT,
    STOCK_SUMMARY_PATH,
    STOCK_VARIANT,
    SUMMARY_SCHEMA_VERSION,
)


SUMMARY_FIELDS: tuple[str, ...] = (
    "variant",
    "checkpoint_source",
    "checkpoint_dir",
    "suite",
    "task_ids",
    "seed_manifest",
    "num_trials_per_task",
    "episode_count",
    "success_rate",
    "failure_count",
    "deviation_notes",
)


def _require_mapping(raw: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


def _normalize_rate(raw: object, *, context: str) -> float:
    value = _coerce_float_like(raw, context=context)
    if math.isnan(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{context} must be within [0, 1], got {value!r}")
    return float(value)


def _coerce_float_like(raw: object, *, context: str) -> float:
    if isinstance(raw, bool) or raw is None:
        raise TypeError(f"{context} must be float-like, got {raw!r}")
    if not isinstance(raw, (int, float, str)):
        raise TypeError(f"{context} must be float-like, got {type(raw).__name__}")
    return float(raw)


def _coerce_int_like(raw: object, *, context: str) -> int:
    if isinstance(raw, bool) or raw is None:
        raise TypeError(f"{context} must be int-like, got {raw!r}")
    if not isinstance(raw, (int, float, str)):
        raise TypeError(f"{context} must be int-like, got {type(raw).__name__}")
    return int(raw)


def _coerce_int_list(raw: object, *, context: str) -> list[int]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(f"{context} must be an integer sequence")
    return [_coerce_int_like(value, context=f"{context}[]") for value in raw]


def _failure_count(*, episode_count: int, success_rate: float) -> int:
    successes = int(round(float(success_rate) * float(episode_count)))
    return max(0, int(episode_count) - successes)


def build_recap_only_summary(
    checkpoint_dir: str | Path,
    *,
    manifest: FrozenComparisonManifest,
) -> dict[str, object]:
    checkpoint_dir_path = Path(checkpoint_dir).resolve()
    checkpoint_payload = read_json(checkpoint_dir_path / "checkpoint.json")
    checkpoint_provenance = read_json(
        checkpoint_dir_path / "checkpoint_provenance.json"
    )
    success_rate = _normalize_rate(
        checkpoint_payload.get("offline_success_proxy", 0.0),
        context="checkpoint.offline_success_proxy",
    )
    summary = {
        "variant": RECAP_ONLY_VARIANT,
        "checkpoint_source": str(checkpoint_provenance["checkpoint_source"]),
        "checkpoint_dir": str(checkpoint_dir_path),
        "suite": manifest.suite,
        "task_ids": [int(value) for value in manifest.task_ids],
        "seed_manifest": [int(value) for value in manifest.seed_manifest],
        "num_trials_per_task": int(manifest.num_trials_per_task),
        "episode_count": int(manifest.episode_count),
        "success_rate": float(success_rate),
        "failure_count": _failure_count(
            episode_count=manifest.episode_count,
            success_rate=success_rate,
        ),
        "deviation_notes": [
            "当前 Task 7 评测是离线 comparative summary；success_rate 来自 checkpoint 中的 offline positive-indicator proxy，而不是 MuJoCo rollout。"
        ],
    }
    return validate_summary_fields(summary)


def build_stock_summary(
    stock_source: str | Path | None,
    *,
    manifest: FrozenComparisonManifest,
) -> dict[str, object]:
    source_path = (
        STOCK_SUMMARY_PATH if stock_source is None else Path(stock_source).resolve()
    )
    summary_path = source_path / "summary.json" if source_path.is_dir() else source_path
    stock_summary = read_json(summary_path)
    provenance = _require_mapping(
        stock_summary.get("provenance", {}), context="stock.provenance"
    )
    client = _require_mapping(stock_summary.get("client", {}), context="stock.client")
    success_rate = _normalize_rate(
        client.get("success_rate", 0.0), context="stock.client.success_rate"
    )
    deviation_notes = [
        "stock 结果直接复用 Task 4 native smoke artifact；当前 summary 字段固定对齐 Task 7 comparison manifest，不会在评测时内联重跑 stock。"
    ]
    source_task_ids = provenance.get("task_ids")
    source_seeds = provenance.get("seed_manifest")
    source_trials = provenance.get("num_trials_per_task")
    if (
        tuple(
            _coerce_int_list(source_task_ids or [], context="stock.provenance.task_ids")
        )
        != tuple(manifest.task_ids)
        or tuple(
            _coerce_int_list(
                source_seeds or [],
                context="stock.provenance.seed_manifest",
            )
        )
        != tuple(manifest.seed_manifest)
        or _coerce_int_like(
            source_trials or 0,
            context="stock.provenance.num_trials_per_task",
        )
        != int(manifest.num_trials_per_task)
    ):
        deviation_notes.append(
            "source stock artifact 与 Task 7 comparison tier 不同；success_rate 仍保留原 artifact 数值，仅作为 frozen stock reference。"
        )
    summary = {
        "variant": STOCK_VARIANT,
        "checkpoint_source": str(provenance.get("checkpoint_source", "")),
        "checkpoint_dir": str(source_path),
        "suite": manifest.suite,
        "task_ids": [int(value) for value in manifest.task_ids],
        "seed_manifest": [int(value) for value in manifest.seed_manifest],
        "num_trials_per_task": int(manifest.num_trials_per_task),
        "episode_count": int(manifest.episode_count),
        "success_rate": float(success_rate),
        "failure_count": _failure_count(
            episode_count=manifest.episode_count,
            success_rate=success_rate,
        ),
        "deviation_notes": deviation_notes,
    }
    return validate_summary_fields(summary)


def build_paired_summary(
    *,
    recap_summary: dict[str, object],
    stock_summary: dict[str, object],
) -> dict[str, object]:
    stock = validate_summary_fields(stock_summary)
    recap = validate_summary_fields(recap_summary)
    return {
        "schema_version": PAIRED_SUMMARY_SCHEMA_VERSION,
        "summary_fields": list(SUMMARY_FIELDS),
        "paired_summary": [stock, recap],
        "delta_success_rate": _coerce_float_like(
            recap["success_rate"],
            context="recap.success_rate",
        )
        - _coerce_float_like(
            stock["success_rate"],
            context="stock.success_rate",
        ),
    }


def build_eval_wrapper(
    *,
    variant: str,
    summary: dict[str, object],
    paired_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "variant": str(variant),
        "summary": validate_summary_fields(summary),
    }
    if paired_summary is not None:
        payload["paired_summary"] = dict(paired_summary)
    return payload


def validate_summary_fields(summary: Mapping[str, object]) -> dict[str, object]:
    missing = [field for field in SUMMARY_FIELDS if field not in summary]
    if missing:
        raise ValueError(f"summary is missing required fields: {missing!r}")
    deviation_notes = summary.get("deviation_notes")
    if not isinstance(deviation_notes, list):
        raise TypeError("summary.deviation_notes must always be a list")
    return {
        "variant": str(summary["variant"]),
        "checkpoint_source": str(summary["checkpoint_source"]),
        "checkpoint_dir": str(summary["checkpoint_dir"]),
        "suite": str(summary["suite"]),
        "task_ids": _coerce_int_list(summary["task_ids"], context="summary.task_ids"),
        "seed_manifest": _coerce_int_list(
            summary["seed_manifest"],
            context="summary.seed_manifest",
        ),
        "num_trials_per_task": _coerce_int_like(
            summary["num_trials_per_task"],
            context="summary.num_trials_per_task",
        ),
        "episode_count": _coerce_int_like(
            summary["episode_count"],
            context="summary.episode_count",
        ),
        "success_rate": _coerce_float_like(
            summary["success_rate"],
            context="summary.success_rate",
        ),
        "failure_count": _coerce_int_like(
            summary["failure_count"],
            context="summary.failure_count",
        ),
        "deviation_notes": [str(item) for item in deviation_notes],
    }


__all__ = [
    "PAIRED_SUMMARY_SCHEMA_VERSION",
    "SUMMARY_FIELDS",
    "SUMMARY_SCHEMA_VERSION",
    "build_eval_wrapper",
    "build_paired_summary",
    "build_recap_only_summary",
    "build_stock_summary",
    "validate_summary_fields",
]
