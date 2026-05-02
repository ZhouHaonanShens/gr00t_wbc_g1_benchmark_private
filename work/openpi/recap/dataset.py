from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from work.openpi.data.contract_mapping import build_phase1_dataset_mapping_spec
from work.openpi.prompting.routes import (
    FIXEDADV_CONSTANT_CONSUMER_MODE,
    RECAP_RELABEL_CONSUMER_MODE,
    SHUFFLED_ADV_DIAG_CONSUMER_MODE,
    build_phase1_prompt_provenance,
    build_phase1_prompt_route,
)
from work.recap.advantage import validate_advantage_input_value

from .protocol import RECAP_RECORD_SCHEMA_VERSION, REPO_ROOT


DATASET_ROOT = REPO_ROOT / "agent" / "artifacts" / "lerobot_datasets"
PREFERRED_DATASET_NAMES: tuple[str, ...] = (
    "physical_intelligence_libero_official_8d_recap_relabels_v1",
    "recap_reward_approved_v1",
    "recap_reward_baseline_v1",
    "openpi_phase05_smoke_contract_v1",
)
REQUIRED_RECAP_COLUMNS: tuple[str, ...] = (
    "action",
    "episode_index",
    "observation.state",
    "recap_m2.advantage_A",
    "recap_m2.advantage_input",
    "recap_m2.indicator_I",
    "recap_m2.prompt_conditioned",
    "recap_m2.prompt_raw",
    "recap_m2.return_G",
    "recap_m2.value_V",
)


@dataclass(frozen=True)
class RecapDatasetBundle:
    dataset_dir: Path
    dataset_name: str
    parquet_files: tuple[Path, ...]
    total_rows: int
    prompt_route: str
    conditioning_mode: str
    source_prompt_field: str
    indicator_positive_fraction: float
    indicator_positive_count: int
    indicator_negative_count: int
    advantage_input_mean: float
    advantage_input_abs_mean: float
    action_dim: int
    state_dim: int
    record_preview: tuple[dict[str, object], ...]
    recap_contract: dict[str, object]
    consumer_mode: str = RECAP_RELABEL_CONSUMER_MODE
    fixed_indicator_mode: str | None = None


def _read_json(path: Path) -> dict[str, object]:
    data = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}, got {type(data).__name__}")
    return {str(key): value for key, value in cast(dict[object, object], data).items()}


def _candidate_dataset_dirs() -> list[Path]:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(DATASET_ROOT)
    preferred = [DATASET_ROOT / name for name in PREFERRED_DATASET_NAMES]
    remaining = sorted(
        [
            path
            for path in DATASET_ROOT.iterdir()
            if path.is_dir() and path.name not in PREFERRED_DATASET_NAMES
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [path for path in preferred if path.is_dir()] + remaining


def _validate_dataset_columns(frame: pd.DataFrame, *, dataset_dir: Path) -> None:
    missing = [
        column for column in REQUIRED_RECAP_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"dataset {dataset_dir} missing required recap columns: {missing!r}"
        )


def _convert_vector(raw: object) -> list[float]:
    if hasattr(raw, "tolist"):
        value = cast(Any, raw).tolist()
    else:
        value = raw
    if not isinstance(value, list):
        raise TypeError(f"expected list-like vector, got {type(raw).__name__}")
    return [float(item) for item in value]


def _build_preview_records(
    frame: pd.DataFrame,
    *,
    limit: int,
    consumer_mode: str,
    fixed_indicator_mode: str | None,
) -> tuple[dict[str, object], ...]:
    preview: list[dict[str, object]] = []
    for _, row in frame.head(limit).iterrows():
        prompt_raw = str(row["recap_m2.prompt_raw"])
        label_row: dict[str, object] = {
            "prompt_raw": prompt_raw,
            "episode_index": int(row["episode_index"]),
            "observation.state": row["observation.state"],
        }
        if consumer_mode in {
            RECAP_RELABEL_CONSUMER_MODE,
            SHUFFLED_ADV_DIAG_CONSUMER_MODE,
        }:
            label_row["recap_m2.indicator_I"] = row["recap_m2.indicator_I"]
        prompt_spec = build_phase1_prompt_route(
            label_row,
            consumer_mode=consumer_mode,
            fixed_indicator_mode=fixed_indicator_mode,
        )
        prompt_provenance = build_phase1_prompt_provenance(prompt_spec)
        record: dict[str, object] = {
            "schema_version": RECAP_RECORD_SCHEMA_VERSION,
            "observation/image": "from observation.images.ego_view",
            "observation/wrist_image": "duplicate observation.images.ego_view",
            "observation/state": _convert_vector(row["observation.state"]),
            "action": _convert_vector(row["action"]),
            "prompt": getattr(prompt_spec, "prompt_text"),
            "prompt_raw": prompt_raw,
            "training_prompt_text": getattr(prompt_spec, "prompt_text"),
            "recap_m2.return_G": float(row["recap_m2.return_G"]),
            "recap_m2.value_V": float(row["recap_m2.value_V"]),
            "recap_m2.advantage_A": float(row["recap_m2.advantage_A"]),
            "prompt_route": prompt_provenance["prompt_route"],
            "conditioning_mode": prompt_provenance["conditioning_mode"],
            "source_prompt_field": prompt_provenance["source_prompt_field"],
            "consumer_mode": prompt_provenance["consumer_mode"],
            "fixed_indicator_mode": prompt_provenance["fixed_indicator_mode"],
            "indicator_mode": prompt_provenance["indicator_mode"],
            "indicator_source": prompt_provenance["indicator_source"],
            "prompt_text_surface": prompt_provenance["prompt_text_surface"],
            "per_sample_indicator_consumption": prompt_provenance[
                "per_sample_indicator_consumption"
            ],
            "prompt_conditioned_dependency": prompt_provenance[
                "prompt_conditioned_dependency"
            ],
            "advantage_input_dependency": prompt_provenance[
                "advantage_input_dependency"
            ],
        }
        if consumer_mode != FIXEDADV_CONSTANT_CONSUMER_MODE:
            record["prompt_conditioned"] = str(row["recap_m2.prompt_conditioned"])
            record["recap_m2.advantage_input"] = validate_advantage_input_value(
                row["recap_m2.advantage_input"],
                context="record_preview.recap_m2.advantage_input",
            )
            record["recap_m2.indicator_I"] = int(row["recap_m2.indicator_I"])
        preview.append(record)
    return tuple(preview)


def _load_parquet_frame(parquet_files: Iterable[Path]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(
            path,
            columns=list(REQUIRED_RECAP_COLUMNS),
        )
        for path in parquet_files
    ]
    if not frames:
        raise ValueError("compatible recap dataset is missing parquet episode files")
    return pd.concat(frames, ignore_index=True)


def resolve_recap_dataset(
    dataset_dir: str | Path,
    *,
    preview_limit: int = 3,
    consumer_mode: str = RECAP_RELABEL_CONSUMER_MODE,
    fixed_indicator_mode: str | None = None,
) -> RecapDatasetBundle:
    dataset_dir_path = Path(dataset_dir).resolve()
    mapping = build_phase1_dataset_mapping_spec(dataset_dir_path)
    parquet_files = tuple(
        sorted(dataset_dir_path.glob("data/chunk-*/episode_*.parquet"))
    )
    frame = _load_parquet_frame(parquet_files)
    _validate_dataset_columns(frame, dataset_dir=dataset_dir_path)
    if frame.empty:
        raise ValueError(f"dataset {dataset_dir_path} contains no recap rows")
    preview = _build_preview_records(
        frame,
        limit=preview_limit,
        consumer_mode=consumer_mode,
        fixed_indicator_mode=fixed_indicator_mode,
    )
    prompt_provenance = {
        "prompt_route": str(preview[0]["prompt_route"]),
        "conditioning_mode": str(preview[0]["conditioning_mode"]),
        "source_prompt_field": str(preview[0]["source_prompt_field"]),
    }
    info = _read_json(dataset_dir_path / "meta" / "info.json")
    recap_contract = cast(
        dict[str, object], info.get("recap_advantage_input_contract", {})
    )
    indicator_series = frame["recap_m2.indicator_I"].astype(int)
    advantage_series = frame["recap_m2.advantage_input"].astype(float)
    return RecapDatasetBundle(
        dataset_dir=dataset_dir_path,
        dataset_name=dataset_dir_path.name,
        parquet_files=tuple(path.resolve() for path in parquet_files),
        total_rows=int(len(frame)),
        prompt_route=prompt_provenance["prompt_route"],
        conditioning_mode=prompt_provenance["conditioning_mode"],
        source_prompt_field=prompt_provenance["source_prompt_field"],
        indicator_positive_fraction=float(indicator_series.mean()),
        indicator_positive_count=int(indicator_series.sum()),
        indicator_negative_count=int(
            len(indicator_series) - int(indicator_series.sum())
        ),
        advantage_input_mean=float(advantage_series.mean()),
        advantage_input_abs_mean=float(advantage_series.abs().mean()),
        action_dim=int(mapping.action_dim),
        state_dim=int(mapping.state_dim),
        record_preview=preview,
        recap_contract=recap_contract,
        consumer_mode=str(preview[0]["consumer_mode"]),
        fixed_indicator_mode=(
            str(preview[0]["fixed_indicator_mode"])
            if str(preview[0]["fixed_indicator_mode"])
            else None
        ),
    )


def resolve_default_recap_dataset(
    *,
    preview_limit: int = 3,
    consumer_mode: str = RECAP_RELABEL_CONSUMER_MODE,
    fixed_indicator_mode: str | None = None,
) -> RecapDatasetBundle:
    failures: list[str] = []
    for dataset_dir in _candidate_dataset_dirs():
        try:
            return resolve_recap_dataset(
                dataset_dir,
                preview_limit=preview_limit,
                consumer_mode=consumer_mode,
                fixed_indicator_mode=fixed_indicator_mode,
            )
        except Exception as exc:
            failures.append(f"{dataset_dir.name}: {exc}")
    raise RuntimeError(
        "unable to locate a compatible RECAP dataset: " + " | ".join(failures)
    )


def dataset_bundle_to_dict(bundle: RecapDatasetBundle) -> dict[str, object]:
    return {
        "dataset_dir": str(bundle.dataset_dir),
        "dataset_name": bundle.dataset_name,
        "parquet_files": [str(path) for path in bundle.parquet_files],
        "total_rows": int(bundle.total_rows),
        "prompt_route": bundle.prompt_route,
        "conditioning_mode": bundle.conditioning_mode,
        "source_prompt_field": bundle.source_prompt_field,
        "indicator_positive_fraction": float(bundle.indicator_positive_fraction),
        "indicator_positive_count": int(bundle.indicator_positive_count),
        "indicator_negative_count": int(bundle.indicator_negative_count),
        "advantage_input_mean": float(bundle.advantage_input_mean),
        "advantage_input_abs_mean": float(bundle.advantage_input_abs_mean),
        "action_dim": int(bundle.action_dim),
        "state_dim": int(bundle.state_dim),
        "record_preview": list(bundle.record_preview),
        "recap_advantage_input_contract": dict(bundle.recap_contract),
        "consumer_mode": bundle.consumer_mode,
        "fixed_indicator_mode": bundle.fixed_indicator_mode,
    }


__all__ = [
    "RecapDatasetBundle",
    "dataset_bundle_to_dict",
    "resolve_recap_dataset",
    "resolve_default_recap_dataset",
]
