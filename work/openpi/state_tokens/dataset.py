from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from work.openpi.data.contract_mapping import build_phase1_dataset_mapping_spec
from work.openpi.prompting.routes import (
    build_phase1_prompt_provenance,
    build_phase1_prompt_route,
)
from work.recap.advantage import validate_advantage_input_value

from work.openpi.recap.dataset import (
    RecapDatasetBundle,
    dataset_bundle_to_dict as recap_dataset_bundle_to_dict,
)

from .protocol import (
    BLOCKER_CODE_INVALID_TRAINING_SOURCE,
    BLOCKER_CODE_MISSING_NATIVE_8D_STATE,
    OFFICIAL_NATIVE_DATASET_DIR,
    OFFICIAL_NATIVE_DATASET_NAME,
    OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_DIR,
    OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME,
    OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
    REQUIRED_NATIVE_STATE_DIM,
    SOURCE_STATE,
    SOURCE_STATE_PADDING,
    StateTokenContractError,
    STATE_TOKEN_ROUTE,
    STATE_TOKEN_SEMANTICS,
    TRANSFORM_ORDER,
    build_blocker_report,
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
class OfficialNativeLiberoDatasetBundle:
    dataset_dir: Path
    dataset_name: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    fps: int
    state_dim: int
    action_dim: int
    task_texts: tuple[str, ...]
    source_dataset_dir: Path
    source_dataset_name: str
    schema_version: str
    route_id: str


@dataclass(frozen=True)
class StateTokenDatasetBundle:
    source_bundle: OfficialNativeLiberoDatasetBundle
    recap_bundle: RecapDatasetBundle
    aligned_record_count: int
    state_token_route: str
    source_state: str
    source_state_padding: str
    transform_order: str
    state_token_semantics: str
    discrete_state_input: bool
    observed_dataset_state_dim: int

    @property
    def dataset_dir(self) -> Path:
        return self.source_bundle.dataset_dir

    @property
    def dataset_name(self) -> str:
        return self.source_bundle.dataset_name

    @property
    def total_rows(self) -> int:
        return int(self.aligned_record_count)


def _read_json(path: Path) -> dict[str, object]:
    data = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}, got {type(data).__name__}")
    return {str(key): value for key, value in cast(dict[object, object], data).items()}


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = cast(object, json.loads(line))
        if not isinstance(data, dict):
            raise ValueError(
                f"expected JSON object line at {path}, got {type(data).__name__}"
            )
        rows.append(
            {str(key): value for key, value in cast(dict[object, object], data).items()}
        )
    return tuple(rows)


def _coerce_int_like(raw: object, *, context: str) -> int:
    if isinstance(raw, bool) or raw is None:
        raise TypeError(f"{context} must be int-like, got {raw!r}")
    if not isinstance(raw, (int, float, str)):
        raise TypeError(f"{context} must be int-like, got {type(raw).__name__}")
    return int(raw)


def _coerce_path(raw: object, *, context: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise TypeError(f"{context} must be a non-empty path string")
    return Path(raw).resolve()


def _coerce_text(raw: object, *, context: str) -> str:
    if not isinstance(raw, str):
        raise TypeError(f"{context} must be a string")
    return raw.strip()


def _build_training_source_blocker(
    *,
    dataset_dir: Path,
    observed_dataset_state_dim: int,
    reason: str,
    info: dict[str, object],
) -> StateTokenContractError:
    return StateTokenContractError(
        reason,
        payload=build_blocker_report(
            stage="train_preflight",
            blocker_code=BLOCKER_CODE_INVALID_TRAINING_SOURCE,
            reason=reason,
            source_dataset_dir=dataset_dir,
            observed_dataset_state_dim=observed_dataset_state_dim,
            next_action=(
                "Use physical_intelligence_libero_official_8d_recap_relabels_v1 as the only state-token training source and keep the official/native 8D provenance intact."
            ),
            extra_payload={
                "observed_dataset_name": dataset_dir.name,
                "observed_schema_version": str(info.get("schema_version", "")),
                "observed_route_id": str(info.get("route_id", "")),
                "observed_source_dataset_name": str(
                    info.get("source_dataset_name", "")
                ),
                "observed_source_dataset_dir": str(info.get("source_dataset_dir", "")),
                "required_training_dataset_name": OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME,
                "required_training_route_id": OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
                "required_official_source_dataset_name": OFFICIAL_NATIVE_DATASET_NAME,
            },
        ),
    )


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
    frame: pd.DataFrame, *, limit: int
) -> tuple[dict[str, object], ...]:
    preview: list[dict[str, object]] = []
    for _, row in frame.head(limit).iterrows():
        prompt_raw = str(row["recap_m2.prompt_raw"])
        indicator_value = row["recap_m2.indicator_I"]
        prompt_spec = build_phase1_prompt_route(
            {
                "prompt_raw": prompt_raw,
                "recap_m2.indicator_I": indicator_value,
            }
        )
        prompt_provenance = build_phase1_prompt_provenance(prompt_spec)
        preview.append(
            {
                "schema_version": "openpi_libero_recap_record_v1",
                "observation/image": "from observation.images.ego_view",
                "observation/wrist_image": "duplicate observation.images.ego_view",
                "observation/state": _convert_vector(row["observation.state"]),
                "action": _convert_vector(row["action"]),
                "prompt": getattr(prompt_spec, "prompt_text"),
                "prompt_raw": prompt_raw,
                "prompt_conditioned": str(row["recap_m2.prompt_conditioned"]),
                "training_prompt_text": getattr(prompt_spec, "prompt_text"),
                "recap_m2.return_G": float(row["recap_m2.return_G"]),
                "recap_m2.value_V": float(row["recap_m2.value_V"]),
                "recap_m2.advantage_A": float(row["recap_m2.advantage_A"]),
                "recap_m2.advantage_input": validate_advantage_input_value(
                    row["recap_m2.advantage_input"],
                    context="record_preview.recap_m2.advantage_input",
                ),
                "recap_m2.indicator_I": int(row["recap_m2.indicator_I"]),
                "prompt_route": prompt_provenance["prompt_route"],
                "conditioning_mode": prompt_provenance["conditioning_mode"],
                "source_prompt_field": prompt_provenance["source_prompt_field"],
            }
        )
    return tuple(preview)


def _load_parquet_frame(parquet_files: Iterable[Path]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path, columns=list(REQUIRED_RECAP_COLUMNS))
        for path in parquet_files
    ]
    if not frames:
        raise ValueError(
            "compatible state-token dataset is missing parquet episode files"
        )
    return pd.concat(frames, ignore_index=True)


def _build_recap_bundle(
    dataset_dir: Path, *, preview_limit: int
) -> tuple[RecapDatasetBundle, dict[str, object]]:
    mapping = build_phase1_dataset_mapping_spec(dataset_dir)
    parquet_files = tuple(sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet")))
    frame = _load_parquet_frame(parquet_files)
    _validate_dataset_columns(frame, dataset_dir=dataset_dir)
    if frame.empty:
        raise ValueError(f"dataset {dataset_dir} contains no recap rows")
    preview = _build_preview_records(frame, limit=preview_limit)
    prompt_provenance = {
        "prompt_route": str(preview[0]["prompt_route"]),
        "conditioning_mode": str(preview[0]["conditioning_mode"]),
        "source_prompt_field": str(preview[0]["source_prompt_field"]),
    }
    info = _read_json(dataset_dir / "meta" / "info.json")
    recap_contract = cast(
        dict[str, object], info.get("recap_advantage_input_contract", {})
    )
    indicator_series = frame["recap_m2.indicator_I"].astype(int)
    advantage_series = frame["recap_m2.advantage_input"].astype(float)
    return (
        RecapDatasetBundle(
            dataset_dir=dataset_dir.resolve(),
            dataset_name=dataset_dir.name,
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
        ),
        info,
    )


def resolve_official_native_8d_dataset() -> OfficialNativeLiberoDatasetBundle:
    dataset_dir = OFFICIAL_NATIVE_DATASET_DIR.resolve()
    info = _read_json(dataset_dir / "meta" / "info.json")
    features = cast(dict[str, object], info.get("features", {}))
    state = cast(dict[str, object], features.get("state", {}))
    actions = cast(dict[str, object], features.get("actions", {}))
    state_shape = cast(list[object], state.get("shape", []))
    action_shape = cast(list[object], actions.get("shape", []))
    state_dim = (
        _coerce_int_like(state_shape[0], context="official.features.state.shape[0]")
        if len(state_shape) == 1
        else -1
    )
    action_dim = (
        _coerce_int_like(action_shape[0], context="official.features.actions.shape[0]")
        if len(action_shape) == 1
        else -1
    )
    if state_dim != REQUIRED_NATIVE_STATE_DIM:
        reason = f"official/native dataset {dataset_dir} must expose state.shape=[8], got {state_shape!r}."
        raise StateTokenContractError(
            reason,
            payload=build_blocker_report(
                stage="train_preflight",
                blocker_code=BLOCKER_CODE_MISSING_NATIVE_8D_STATE,
                reason=reason,
                source_dataset_dir=dataset_dir,
                observed_dataset_state_dim=state_dim,
            ),
        )
    tasks = _read_jsonl(dataset_dir / "meta" / "tasks.jsonl")
    return OfficialNativeLiberoDatasetBundle(
        dataset_dir=dataset_dir,
        dataset_name=OFFICIAL_NATIVE_DATASET_NAME,
        total_episodes=_coerce_int_like(
            info.get("total_episodes", 0), context="official.total_episodes"
        ),
        total_frames=_coerce_int_like(
            info.get("total_frames", 0), context="official.total_frames"
        ),
        total_tasks=_coerce_int_like(
            info.get("total_tasks", 0), context="official.total_tasks"
        ),
        fps=_coerce_int_like(info.get("fps", 0), context="official.fps"),
        state_dim=state_dim,
        action_dim=action_dim,
        task_texts=tuple(
            str(row.get("task", "")).strip()
            for row in tasks
            if str(row.get("task", "")).strip()
        ),
        source_dataset_dir=dataset_dir,
        source_dataset_name=OFFICIAL_NATIVE_DATASET_NAME,
        schema_version=str(info.get("schema_version", "")),
        route_id=str(info.get("route_id", "official_native_8d_source")),
    )


def resolve_state_token_dataset(
    dataset_dir: str | Path | None = None, *, preview_limit: int = 3
) -> StateTokenDatasetBundle:
    dataset_dir_path = (
        OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_DIR.resolve()
        if dataset_dir is None
        else Path(dataset_dir).resolve()
    )
    recap_bundle, info = _build_recap_bundle(
        dataset_dir_path, preview_limit=preview_limit
    )
    mapping = build_phase1_dataset_mapping_spec(dataset_dir_path)
    observed_state_dim = int(mapping.state_dim)
    if observed_state_dim != REQUIRED_NATIVE_STATE_DIM:
        reason = f"state-token training dataset {dataset_dir_path} must expose observation.state.shape=[8], got {observed_state_dim!r}."
        raise StateTokenContractError(
            reason,
            payload=build_blocker_report(
                stage="train_preflight",
                blocker_code=BLOCKER_CODE_MISSING_NATIVE_8D_STATE,
                reason=reason,
                source_dataset_dir=dataset_dir_path,
                observed_dataset_state_dim=observed_state_dim,
            ),
        )
    if int(mapping.action_dim) != 7:
        reason = f"state-token training dataset {dataset_dir_path} must preserve action.shape=[7], got {mapping.action_dim!r}."
        raise _build_training_source_blocker(
            dataset_dir=dataset_dir_path,
            observed_dataset_state_dim=observed_state_dim,
            reason=reason,
            info=info,
        )
    schema_version = _coerce_text(
        info.get("schema_version", ""), context="training.info.schema_version"
    )
    route_id = _coerce_text(info.get("route_id", ""), context="training.info.route_id")
    source_dataset_name = _coerce_text(
        info.get("source_dataset_name", ""),
        context="training.info.source_dataset_name",
    )
    source_dataset_dir = _coerce_path(
        info.get("source_dataset_dir", ""),
        context="training.info.source_dataset_dir",
    )
    if dataset_dir_path.name != OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME:
        reason = (
            "Task 9D only allows the relabeled official/native 8D source dataset as the state-token training input, "
            + f"got dataset_dir.name={dataset_dir_path.name!r}."
        )
        raise _build_training_source_blocker(
            dataset_dir=dataset_dir_path,
            observed_dataset_state_dim=observed_state_dim,
            reason=reason,
            info=info,
        )
    if route_id != OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID:
        reason = (
            "state-token training dataset route_id must remain official_native_8d_recap_relabels_v1, "
            + f"got {route_id!r}."
        )
        raise _build_training_source_blocker(
            dataset_dir=dataset_dir_path,
            observed_dataset_state_dim=observed_state_dim,
            reason=reason,
            info=info,
        )
    if source_dataset_name != OFFICIAL_NATIVE_DATASET_NAME:
        reason = (
            "state-token training dataset must preserve official/native LIBERO provenance in meta/info.json, "
            + f"got source_dataset_name={source_dataset_name!r}."
        )
        raise _build_training_source_blocker(
            dataset_dir=dataset_dir_path,
            observed_dataset_state_dim=observed_state_dim,
            reason=reason,
            info=info,
        )
    tasks = _read_jsonl(dataset_dir_path / "meta" / "tasks.jsonl")
    source_bundle = OfficialNativeLiberoDatasetBundle(
        dataset_dir=dataset_dir_path,
        dataset_name=dataset_dir_path.name,
        total_episodes=_coerce_int_like(
            info.get("total_episodes", 0), context="training.total_episodes"
        ),
        total_frames=_coerce_int_like(
            info.get("total_frames", 0), context="training.total_frames"
        ),
        total_tasks=_coerce_int_like(
            info.get("total_tasks", 0), context="training.total_tasks"
        ),
        fps=_coerce_int_like(info.get("fps", 0), context="training.fps"),
        state_dim=observed_state_dim,
        action_dim=int(mapping.action_dim),
        task_texts=tuple(
            str(row.get("task", "")).strip()
            for row in tasks
            if str(row.get("task", "")).strip()
        ),
        source_dataset_dir=source_dataset_dir,
        source_dataset_name=source_dataset_name,
        schema_version=schema_version,
        route_id=route_id,
    )
    return StateTokenDatasetBundle(
        source_bundle=source_bundle,
        recap_bundle=recap_bundle,
        aligned_record_count=int(recap_bundle.total_rows),
        state_token_route=STATE_TOKEN_ROUTE,
        source_state=SOURCE_STATE,
        source_state_padding=SOURCE_STATE_PADDING,
        transform_order=TRANSFORM_ORDER,
        state_token_semantics=STATE_TOKEN_SEMANTICS,
        discrete_state_input=True,
        observed_dataset_state_dim=observed_state_dim,
    )


def resolve_default_state_token_dataset(
    *, preview_limit: int = 3
) -> StateTokenDatasetBundle:
    return resolve_state_token_dataset(
        OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_DIR,
        preview_limit=preview_limit,
    )


def dataset_bundle_to_dict(bundle: StateTokenDatasetBundle) -> dict[str, object]:
    payload = recap_dataset_bundle_to_dict(bundle.recap_bundle)
    payload.update(
        {
            "dataset_dir": str(bundle.source_bundle.dataset_dir),
            "dataset_name": bundle.source_bundle.dataset_name,
            "parquet_files": [str(path) for path in bundle.recap_bundle.parquet_files],
            "total_rows": int(bundle.aligned_record_count),
            "state_dim": int(bundle.source_bundle.state_dim),
            "action_dim": int(bundle.source_bundle.action_dim),
            "official_native_source": {
                "dataset_dir": str(bundle.source_bundle.source_dataset_dir),
                "dataset_name": bundle.source_bundle.source_dataset_name,
                "required_source_dataset_dir": str(OFFICIAL_NATIVE_DATASET_DIR),
                "required_source_dataset_name": OFFICIAL_NATIVE_DATASET_NAME,
            },
            "source_dataset_schema_version": bundle.source_bundle.schema_version,
            "source_dataset_route_id": bundle.source_bundle.route_id,
            "recap_label_source": recap_dataset_bundle_to_dict(bundle.recap_bundle),
            "state_token_route": bundle.state_token_route,
            "source_state": bundle.source_state,
            "source_state_padding": bundle.source_state_padding,
            "transform_order": bundle.transform_order,
            "state_token_semantics": bundle.state_token_semantics,
            "discrete_state_input": bool(bundle.discrete_state_input),
            "observed_dataset_state_dim": int(bundle.observed_dataset_state_dim),
        }
    )
    return payload


__all__ = [
    "OfficialNativeLiberoDatasetBundle",
    "StateTokenDatasetBundle",
    "dataset_bundle_to_dict",
    "resolve_default_state_token_dataset",
    "resolve_official_native_8d_dataset",
    "resolve_state_token_dataset",
]
