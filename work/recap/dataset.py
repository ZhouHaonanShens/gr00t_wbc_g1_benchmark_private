# pyright: reportMissingImports=false, reportRedeclaration=false, reportAssignmentType=false
"""Repo-local RECAP dataset overlays for numeric advantage wiring."""

from __future__ import annotations

import logging
import math
import numbers
import random
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd

    Float32Array = npt.NDArray[np.float32]
else:
    Float32Array = object

from .advantage import (
    ADVANTAGE_INPUT_COLUMN,
    NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
    build_diagnostic_surface_metadata,
    validate_advantage_input_value,
)

logger = logging.getLogger(__name__)

_POSITIVE_OVERSAMPLING_CONFIG: dict[str, int] = {"factor": 1}
_POSITIVE_CURRICULUM_CONFIG: dict[str, bool | float | int] = {
    "enabled": False,
    "negative_retain_probability": 1.0,
    "seed": 42,
}
_LATE_STAGE_POSITIVE_CONFIG: dict[str, bool | float | str] = {
    "enabled": False,
    "threshold": 0.8,
    "rule": "advantage_input>0_and_t_norm>=threshold",
}
_STEP_REF_EPISODE_KEYS = ("episode_index", "episode_idx", "episode_id")
_STEP_REF_STEP_KEYS = ("step_index", "step_idx", "timestep", "frame_index", "index")
_T_NORM_COLUMN_CANDIDATES = ("recap_m2.t_norm", "t_norm")
_LATE_STAGE_POSITIVE_RULE = "advantage_input>0_and_t_norm>=threshold"


def configure_positive_oversampling(*, factor: int = 1) -> dict[str, Any]:
    factor_i = int(factor)
    if factor_i < 1:
        raise ValueError(f"positive oversample factor must be >= 1, got {factor!r}")
    _POSITIVE_OVERSAMPLING_CONFIG["factor"] = factor_i
    payload: dict[str, Any] = {"enabled": factor_i > 1, "factor": factor_i}
    payload.update(
        build_diagnostic_surface_metadata(
            surface_route="numeric_advantage_positive_oversample_config_diagnostic",
            authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
            surface_kind="numeric_advantage_dataset_sampling_config",
        )
    )
    logger.info(
        "numeric-adv positive oversample config enabled=%s factor=%d",
        bool(payload["enabled"]),
        factor_i,
    )
    return payload


def configure_positive_curriculum(
    *,
    enabled: bool = False,
    negative_retain_probability: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    retain_prob = float(negative_retain_probability)
    if not 0.0 <= retain_prob <= 1.0:
        raise ValueError(
            "negative retain probability must be in [0, 1], "
            f"got {negative_retain_probability!r}"
        )
    seed_i = int(seed)
    enabled_flag = bool(enabled)
    _POSITIVE_CURRICULUM_CONFIG["enabled"] = enabled_flag
    _POSITIVE_CURRICULUM_CONFIG["negative_retain_probability"] = retain_prob
    _POSITIVE_CURRICULUM_CONFIG["seed"] = seed_i
    payload: dict[str, Any] = {
        "enabled": enabled_flag,
        "negative_retain_probability": retain_prob,
        "seed": seed_i,
    }
    payload.update(
        build_diagnostic_surface_metadata(
            surface_route="numeric_advantage_positive_curriculum_config_diagnostic",
            authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
            surface_kind="numeric_advantage_dataset_sampling_config",
        )
    )
    logger.info(
        "numeric-adv positive curriculum enabled=%s negative retain probability=%.6f seed=%d",
        enabled_flag,
        retain_prob,
        seed_i,
    )
    return payload


def configure_late_stage_positive_emphasis(
    *,
    enabled: bool = False,
    threshold: float = 0.8,
) -> dict[str, Any]:
    threshold_f = float(threshold)
    if not 0.0 <= threshold_f <= 1.0:
        raise ValueError(f"late-stage threshold must be in [0, 1], got {threshold!r}")
    enabled_flag = bool(enabled)
    rule = _LATE_STAGE_POSITIVE_RULE
    _LATE_STAGE_POSITIVE_CONFIG["enabled"] = enabled_flag
    _LATE_STAGE_POSITIVE_CONFIG["threshold"] = threshold_f
    _LATE_STAGE_POSITIVE_CONFIG["rule"] = rule
    payload: dict[str, Any] = {
        "enabled": enabled_flag,
        "threshold": threshold_f,
        "rule": rule,
    }
    payload.update(
        build_diagnostic_surface_metadata(
            surface_route="numeric_advantage_late_stage_positive_config_diagnostic",
            authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
            surface_kind="numeric_advantage_dataset_sampling_config",
        )
    )
    logger.info(
        "numeric-adv late-stage positive config late_stage_positive_enabled=%s late_stage_threshold=%.6f late_stage_rule=%s",
        enabled_flag,
        threshold_f,
        rule,
    )
    return payload


def _coerce_integral_pair(first: Any, second: Any) -> tuple[int, int] | None:
    if not isinstance(first, numbers.Integral) or not isinstance(
        second, numbers.Integral
    ):
        return None
    return int(first), int(second)


def _mapping_step_ref(value: Any) -> tuple[int, int] | None:
    if not hasattr(value, "get"):
        return None

    episode_value = None
    step_value = None
    for key in _STEP_REF_EPISODE_KEYS:
        candidate = value.get(key)
        if candidate is not None:
            episode_value = candidate
            break
    for key in _STEP_REF_STEP_KEYS:
        candidate = value.get(key)
        if candidate is not None:
            step_value = candidate
            break
    if episode_value is None or step_value is None:
        return None
    return _coerce_integral_pair(episode_value, step_value)


def _object_step_ref(value: Any) -> tuple[int, int] | None:
    episode_value = None
    step_value = None
    for key in _STEP_REF_EPISODE_KEYS:
        if hasattr(value, key):
            episode_value = getattr(value, key)
            break
    for key in _STEP_REF_STEP_KEYS:
        if hasattr(value, key):
            step_value = getattr(value, key)
            break
    if episode_value is None or step_value is None:
        return None
    return _coerce_integral_pair(episode_value, step_value)


ImportedLeRobotEpisodeLoader: Any = object
ImportedShardedSingleStepDataset: Any = object
imported_extract_step_data: Any = None
ImportedMessageType: Any = None
np = None

_import_error: ModuleNotFoundError | None = None
try:
    import numpy as np
    import pandas as pd
    from gr00t.data.dataset.lerobot_episode_loader import (
        LeRobotEpisodeLoader as ImportedLeRobotEpisodeLoader,
    )
    from gr00t.data.dataset.sharded_single_step_dataset import (
        ShardedSingleStepDataset as ImportedShardedSingleStepDataset,
        extract_step_data as imported_extract_step_data,
    )
    from gr00t.data.types import MessageType as ImportedMessageType
except ModuleNotFoundError as exc:
    _import_error = exc


if _import_error is not None:

    class _RecapLeRobotEpisodeLoaderUnavailable:
        def __init__(self, *args: Any, **kwargs: Any):
            del args, kwargs
            raise ModuleNotFoundError(
                "RecapLeRobotEpisodeLoader requires numpy, pandas, and gr00t"
            ) from _import_error

    class _RecapAdvantageShardedSingleStepDatasetUnavailable:
        def __init__(self, *args: Any, **kwargs: Any):
            del args, kwargs
            raise ModuleNotFoundError(
                "RecapAdvantageShardedSingleStepDataset requires numpy, pandas, and gr00t"
            ) from _import_error

    RecapLeRobotEpisodeLoader: Any = _RecapLeRobotEpisodeLoaderUnavailable
    RecapAdvantageShardedSingleStepDataset: Any = (
        _RecapAdvantageShardedSingleStepDatasetUnavailable
    )

else:
    assert np is not None
    runtime_np = cast(Any, np)

    class RecapLeRobotEpisodeLoader(ImportedLeRobotEpisodeLoader):
        def __init__(
            self,
            dataset_path: str | Path,
            modality_configs: dict[str, Any],
            video_backend: str = "torchcodec",
            video_backend_kwargs: dict[str, Any] | None = None,
            *,
            passthrough_scalar_columns: tuple[str, ...] = (ADVANTAGE_INPUT_COLUMN,),
            optional_scalar_columns: tuple[str, ...] = _T_NORM_COLUMN_CANDIDATES,
        ) -> None:
            self.passthrough_scalar_columns = tuple(
                dict.fromkeys(str(col) for col in passthrough_scalar_columns)
            )
            self.optional_scalar_columns = tuple(
                column
                for column in dict.fromkeys(str(col) for col in optional_scalar_columns)
                if column not in self.passthrough_scalar_columns
            )
            super().__init__(
                dataset_path=dataset_path,
                modality_configs=modality_configs,
                video_backend=video_backend,
                video_backend_kwargs=video_backend_kwargs,
            )

        def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
            loaded_df = super()._load_parquet_data(episode_index)
            if not self.passthrough_scalar_columns and not self.optional_scalar_columns:
                return loaded_df

            chunk_idx = episode_index // self.chunk_size
            parquet_filename = self.data_path_pattern.format(
                episode_chunk=chunk_idx,
                episode_index=episode_index,
            )
            parquet_path = self.dataset_path / parquet_filename
            if self.passthrough_scalar_columns:
                raw_scalar_df = pd.read_parquet(
                    parquet_path,
                    columns=list(self.passthrough_scalar_columns),
                )
                for column in self.passthrough_scalar_columns:
                    if column not in raw_scalar_df.columns:
                        raise KeyError(
                            f"Missing required RECAP passthrough column {column!r} in {parquet_path}"
                        )
                    loaded_df[column] = raw_scalar_df[column]

            for column in self.optional_scalar_columns:
                if column in loaded_df.columns:
                    continue
                try:
                    optional_scalar_df = pd.read_parquet(parquet_path, columns=[column])
                except Exception as exc:
                    logger.debug(
                        "recap optional scalar passthrough skipped column=%s parquet_path=%s reason=%s:%s",
                        column,
                        parquet_path,
                        type(exc).__name__,
                        exc,
                    )
                    continue
                if column in optional_scalar_df.columns:
                    loaded_df[column] = optional_scalar_df[column]
            return loaded_df

    class RecapAdvantageShardedSingleStepDataset(ImportedShardedSingleStepDataset):
        def __init__(
            self,
            dataset_path: str | Path,
            embodiment_tag: Any,
            modality_configs: dict[str, Any],
            video_backend: str = "torchcodec",
            video_backend_kwargs: dict[str, Any] | None = None,
            shard_size: int = 2**10,
            episode_sampling_rate: float = 0.1,
            seed: int = 42,
            allow_padding: bool = False,
            *,
            advantage_column: str = ADVANTAGE_INPUT_COLUMN,
            positive_oversample_factor: int | None = None,
            positive_curriculum_enabled: bool | None = None,
            negative_retain_probability: float | None = None,
            positive_curriculum_seed: int | None = None,
            late_stage_positive_enabled: bool | None = None,
            late_stage_threshold: float | None = None,
        ) -> None:
            self.advantage_column = str(advantage_column)
            configured_factor = _POSITIVE_OVERSAMPLING_CONFIG.get("factor", 1)
            configured_curriculum_enabled = bool(
                _POSITIVE_CURRICULUM_CONFIG.get("enabled", False)
            )
            configured_negative_retain_probability = float(
                _POSITIVE_CURRICULUM_CONFIG.get("negative_retain_probability", 1.0)
            )
            configured_curriculum_seed = int(
                _POSITIVE_CURRICULUM_CONFIG.get("seed", 42)
            )
            configured_late_stage_enabled = bool(
                _LATE_STAGE_POSITIVE_CONFIG.get("enabled", False)
            )
            configured_late_stage_threshold = float(
                _LATE_STAGE_POSITIVE_CONFIG.get("threshold", 0.8)
            )
            self.positive_oversample_factor = int(
                configured_factor
                if positive_oversample_factor is None
                else positive_oversample_factor
            )
            self.positive_curriculum_enabled = bool(
                configured_curriculum_enabled
                if positive_curriculum_enabled is None
                else positive_curriculum_enabled
            )
            self.negative_retain_probability = float(
                configured_negative_retain_probability
                if negative_retain_probability is None
                else negative_retain_probability
            )
            self.positive_curriculum_seed = int(
                configured_curriculum_seed
                if positive_curriculum_seed is None
                else positive_curriculum_seed
            )
            self.late_stage_positive_enabled = bool(
                configured_late_stage_enabled
                if late_stage_positive_enabled is None
                else late_stage_positive_enabled
            )
            self.late_stage_threshold = float(
                configured_late_stage_threshold
                if late_stage_threshold is None
                else late_stage_threshold
            )
            if self.positive_oversample_factor < 1:
                raise ValueError(
                    "positive_oversample_factor must be >= 1; "
                    f"got {self.positive_oversample_factor!r}"
                )
            if not 0.0 <= self.negative_retain_probability <= 1.0:
                raise ValueError(
                    "negative_retain_probability must be in [0, 1]; "
                    f"got {self.negative_retain_probability!r}"
                )
            if not 0.0 <= self.late_stage_threshold <= 1.0:
                raise ValueError(
                    "late_stage_threshold must be in [0, 1]; "
                    f"got {self.late_stage_threshold!r}"
                )
            self._positive_oversample_index_lookup: tuple[int, ...] | None = None
            self._recap_episode_loader_ready = False
            super().__init__(
                dataset_path=dataset_path,
                embodiment_tag=embodiment_tag,
                modality_configs=modality_configs,
                video_backend=video_backend,
                video_backend_kwargs=video_backend_kwargs,
                shard_size=shard_size,
                episode_sampling_rate=episode_sampling_rate,
                seed=seed,
                allow_padding=allow_padding,
            )
            self.episode_loader = RecapLeRobotEpisodeLoader(
                dataset_path=dataset_path,
                modality_configs=modality_configs,
                video_backend=video_backend,
                video_backend_kwargs=video_backend_kwargs,
                passthrough_scalar_columns=(self.advantage_column,),
                optional_scalar_columns=_T_NORM_COLUMN_CANDIDATES,
            )
            self._advantage_debug_log_budget = 12
            self._recap_episode_loader_ready = True
            self.shard_dataset()

        def _episode_data_for_scan(
            self,
            episode_cache: dict[int, pd.DataFrame],
            episode_index: int,
        ) -> pd.DataFrame:
            cached = episode_cache.get(episode_index)
            if cached is not None:
                return cached
            loaded = self.episode_loader._load_parquet_data(episode_index)
            episode_cache[episode_index] = loaded
            return loaded

        def _scan_step_curriculum(
            self,
            *,
            shard_index: int,
            episode_index: int,
            episode_data: pd.DataFrame,
            step_indices: Any,
        ) -> tuple[list[int], list[int], int, int, int, int, int, int, int]:
            selected_positive_step_indices: list[int] = []
            retained_step_indices: list[int] = []
            total_step_refs = 0
            positive_step_refs = 0
            negative_step_refs = 0
            late_stage_positive_step_refs = 0
            selected_positive_step_refs = 0
            retained_negative_step_refs = 0
            retained_positive_nonselected_step_refs = 0
            curriculum_rng = random.Random(
                int(getattr(self, "positive_curriculum_seed", 42)) * 1_000_003
                + int(shard_index) * 10_007
                + int(episode_index)
            )
            late_stage_positive_enabled = bool(
                getattr(self, "late_stage_positive_enabled", False)
            )
            curriculum_enabled = bool(
                getattr(self, "positive_curriculum_enabled", False)
            )
            negative_retain_probability = float(
                getattr(self, "negative_retain_probability", 1.0)
            )

            for raw_step_index in step_indices:
                step_index = int(raw_step_index)
                raw_value = self._get_row_value(
                    episode_data, step_index, self.advantage_column
                )
                advantage_value = validate_advantage_input_value(
                    raw_value,
                    context=self.advantage_column,
                )
                total_step_refs += 1
                is_positive = float(advantage_value) > 0.0
                is_late_stage_positive = False
                if is_positive:
                    positive_step_refs += 1
                    is_late_stage_positive = self._is_late_stage_positive_step(
                        episode_data,
                        step_index,
                        advantage_value=float(advantage_value),
                    )
                    if is_late_stage_positive:
                        late_stage_positive_step_refs += 1
                else:
                    negative_step_refs += 1

                is_selected_positive = is_positive and (
                    not late_stage_positive_enabled or is_late_stage_positive
                )
                if is_selected_positive:
                    selected_positive_step_refs += 1
                    selected_positive_step_indices.append(step_index)

                if not curriculum_enabled:
                    retained_step_indices.append(step_index)
                    if not is_positive:
                        retained_negative_step_refs += 1
                    elif not is_selected_positive:
                        retained_positive_nonselected_step_refs += 1
                    continue

                if is_selected_positive:
                    retained_step_indices.append(step_index)
                    continue

                if (
                    negative_retain_probability >= 1.0
                    or curriculum_rng.random() < float(negative_retain_probability)
                ):
                    retained_step_indices.append(step_index)
                    if not is_positive:
                        retained_negative_step_refs += 1
                    else:
                        retained_positive_nonselected_step_refs += 1

            return (
                selected_positive_step_indices,
                retained_step_indices,
                total_step_refs,
                positive_step_refs,
                negative_step_refs,
                late_stage_positive_step_refs,
                selected_positive_step_refs,
                retained_negative_step_refs,
                retained_positive_nonselected_step_refs,
            )

        def _oversample_step_indices(
            self,
            *,
            shard_index: int,
            episode_index: int,
            step_indices: Any,
            positive_step_indices: list[int],
        ) -> Any:
            base_indices = [int(step_index) for step_index in step_indices]
            if self.positive_oversample_factor <= 1 or not positive_step_indices:
                return runtime_np.asarray(base_indices, dtype=runtime_np.int64)

            oversampled_indices = list(base_indices)
            oversampled_indices.extend(
                int(step_index)
                for _ in range(self.positive_oversample_factor - 1)
                for step_index in positive_step_indices
            )
            random.Random(
                int(getattr(self, "seed", 42)) * 1_000_003
                + int(shard_index) * 10_007
                + int(episode_index)
            ).shuffle(oversampled_indices)
            return runtime_np.asarray(oversampled_indices, dtype=runtime_np.int64)

        def _positive_shard_scan(
            self, sharded_episodes: list[list[tuple[int, Any]]]
        ) -> tuple[list[list[tuple[int, Any]]], dict[str, int | float]]:
            episode_cache: dict[int, pd.DataFrame] = {}
            oversampled_shards: list[list[tuple[int, Any]]] = []
            total_shards_before = len(sharded_episodes)
            positive_shards_before = 0
            total_shards_after = 0
            positive_shards_after = 0
            total_step_refs_before = 0
            positive_step_refs_before = 0
            negative_step_refs_before = 0
            late_stage_positive_step_refs_before = 0
            selected_positive_step_refs_before = 0
            total_episode_refs_before = 0
            positive_episode_refs_before = 0
            total_step_refs_after = 0
            positive_step_refs_after = 0
            negative_step_refs_after = 0
            late_stage_positive_step_refs_after = 0
            selected_positive_step_refs_after = 0
            total_episode_refs_after = 0
            positive_episode_refs_after = 0
            factor = int(self.positive_oversample_factor)

            for shard_index, shard in enumerate(sharded_episodes):
                oversampled_shard: list[tuple[int, Any]] = []
                shard_has_positive_before = False
                shard_has_positive_after = False
                for episode_index, split_step_indices in shard:
                    episode_data = self._episode_data_for_scan(
                        episode_cache, episode_index
                    )
                    (
                        selected_positive_step_indices,
                        retained_step_indices,
                        seq_total_step_refs,
                        seq_positive_step_refs,
                        seq_negative_step_refs,
                        seq_late_stage_positive_step_refs,
                        seq_selected_positive_step_refs,
                        seq_retained_negative_step_refs,
                        seq_retained_positive_nonselected_step_refs,
                    ) = self._scan_step_curriculum(
                        shard_index=shard_index,
                        episode_index=int(episode_index),
                        episode_data=episode_data,
                        step_indices=split_step_indices,
                    )
                    total_episode_refs_before += 1
                    total_step_refs_before += seq_total_step_refs
                    positive_step_refs_before += seq_positive_step_refs
                    negative_step_refs_before += seq_negative_step_refs
                    late_stage_positive_step_refs_before += (
                        seq_late_stage_positive_step_refs
                    )
                    selected_positive_step_refs_before += (
                        seq_selected_positive_step_refs
                    )
                    if seq_positive_step_refs > 0:
                        positive_episode_refs_before += 1
                        shard_has_positive_before = True
                    oversampled_step_indices = self._oversample_step_indices(
                        shard_index=shard_index,
                        episode_index=int(episode_index),
                        step_indices=retained_step_indices,
                        positive_step_indices=selected_positive_step_indices,
                    )
                    if int(len(oversampled_step_indices)) <= 0:
                        continue
                    oversampled_shard.append(
                        (int(episode_index), oversampled_step_indices)
                    )
                    total_episode_refs_after += 1
                    total_step_refs_after += int(len(oversampled_step_indices))
                    positive_step_refs_after += int(
                        seq_selected_positive_step_refs * factor
                        + seq_retained_positive_nonselected_step_refs
                    )
                    selected_positive_step_refs_after += int(
                        seq_selected_positive_step_refs * factor
                    )
                    late_stage_positive_step_refs_after += int(
                        seq_late_stage_positive_step_refs * factor
                    )
                    negative_step_refs_after += int(seq_retained_negative_step_refs)
                    if seq_positive_step_refs > 0:
                        positive_episode_refs_after += 1
                        shard_has_positive_after = True

                if shard_has_positive_before:
                    positive_shards_before += 1
                if oversampled_shard:
                    total_shards_after += 1
                    if shard_has_positive_after:
                        positive_shards_after += 1
                    oversampled_shards.append(oversampled_shard)

            return oversampled_shards, {
                "total_shards_before": total_shards_before,
                "positive_shards_before": positive_shards_before,
                "positive_shard_ratio_before": (
                    float(positive_shards_before) / float(total_shards_before)
                    if total_shards_before > 0
                    else 0.0
                ),
                "total_shards_after": total_shards_after,
                "positive_shards_after": positive_shards_after,
                "positive_shard_ratio_after": (
                    float(positive_shards_after) / float(total_shards_after)
                    if total_shards_after > 0
                    else 0.0
                ),
                "total_episode_refs_before": total_episode_refs_before,
                "positive_episode_refs_before": positive_episode_refs_before,
                "positive_episode_ratio_before": (
                    float(positive_episode_refs_before)
                    / float(total_episode_refs_before)
                    if total_episode_refs_before > 0
                    else 0.0
                ),
                "total_episode_refs_after": total_episode_refs_after,
                "positive_episode_refs_after": positive_episode_refs_after,
                "positive_episode_ratio_after": (
                    float(positive_episode_refs_after) / float(total_episode_refs_after)
                    if total_episode_refs_after > 0
                    else 0.0
                ),
                "total_step_refs_before": total_step_refs_before,
                "positive_step_refs_before": positive_step_refs_before,
                "negative_step_refs_before": negative_step_refs_before,
                "late_stage_positive_step_refs_before": late_stage_positive_step_refs_before,
                "selected_positive_step_refs_before": selected_positive_step_refs_before,
                "positive_step_ratio_before": (
                    float(positive_step_refs_before) / float(total_step_refs_before)
                    if total_step_refs_before > 0
                    else 0.0
                ),
                "negative_step_ratio_before": (
                    float(negative_step_refs_before) / float(total_step_refs_before)
                    if total_step_refs_before > 0
                    else 0.0
                ),
                "late_stage_positive_step_ratio_before": (
                    float(late_stage_positive_step_refs_before)
                    / float(total_step_refs_before)
                    if total_step_refs_before > 0
                    else 0.0
                ),
                "selected_positive_step_ratio_before": (
                    float(selected_positive_step_refs_before)
                    / float(total_step_refs_before)
                    if total_step_refs_before > 0
                    else 0.0
                ),
                "late_stage_positive_within_positive_ratio_before": (
                    float(late_stage_positive_step_refs_before)
                    / float(positive_step_refs_before)
                    if positive_step_refs_before > 0
                    else 0.0
                ),
                "selected_positive_within_positive_ratio_before": (
                    float(selected_positive_step_refs_before)
                    / float(positive_step_refs_before)
                    if positive_step_refs_before > 0
                    else 0.0
                ),
                "total_step_refs_after": total_step_refs_after,
                "positive_step_refs_after": positive_step_refs_after,
                "negative_step_refs_after": negative_step_refs_after,
                "late_stage_positive_step_refs_after": late_stage_positive_step_refs_after,
                "selected_positive_step_refs_after": selected_positive_step_refs_after,
                "positive_step_ratio_after": (
                    float(positive_step_refs_after) / float(total_step_refs_after)
                    if total_step_refs_after > 0
                    else 0.0
                ),
                "negative_step_ratio_after": (
                    float(negative_step_refs_after) / float(total_step_refs_after)
                    if total_step_refs_after > 0
                    else 0.0
                ),
                "late_stage_positive_step_ratio_after": (
                    float(late_stage_positive_step_refs_after)
                    / float(total_step_refs_after)
                    if total_step_refs_after > 0
                    else 0.0
                ),
                "selected_positive_step_ratio_after": (
                    float(selected_positive_step_refs_after)
                    / float(total_step_refs_after)
                    if total_step_refs_after > 0
                    else 0.0
                ),
                "late_stage_positive_within_positive_ratio_after": (
                    float(late_stage_positive_step_refs_after)
                    / float(positive_step_refs_after)
                    if positive_step_refs_after > 0
                    else 0.0
                ),
                "selected_positive_within_positive_ratio_after": (
                    float(selected_positive_step_refs_after)
                    / float(positive_step_refs_after)
                    if positive_step_refs_after > 0
                    else 0.0
                ),
            }

        def _resolve_step_t_norm(
            self,
            episode_data: pd.DataFrame,
            step_index: int,
        ) -> float:
            columns = getattr(episode_data, "columns", None)
            if columns is not None:
                for column in _T_NORM_COLUMN_CANDIDATES:
                    if column not in columns:
                        continue
                    raw_value = self._get_row_value(episode_data, step_index, column)
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value) and 0.0 <= value <= 1.0:
                        return value

            last_step_index = len(episode_data) - 1
            if last_step_index <= 0:
                return 1.0
            return float(int(step_index)) / float(last_step_index)

        def _is_late_stage_positive_step(
            self,
            episode_data: pd.DataFrame,
            step_index: int,
            *,
            advantage_value: float,
        ) -> bool:
            if float(advantage_value) <= 0.0:
                return False
            t_norm = self._resolve_step_t_norm(episode_data, int(step_index))
            return t_norm >= float(getattr(self, "late_stage_threshold", 0.8))

        def shard_dataset(self) -> None:
            super().shard_dataset()
            self._positive_oversample_index_lookup = None

            factor = int(getattr(self, "positive_oversample_factor", 1))
            curriculum_enabled = bool(
                getattr(self, "positive_curriculum_enabled", False)
            )
            negative_retain_probability = float(
                getattr(self, "negative_retain_probability", 1.0)
            )
            curriculum_seed = int(getattr(self, "positive_curriculum_seed", 42))
            late_stage_positive_enabled = bool(
                getattr(self, "late_stage_positive_enabled", False)
            )
            late_stage_threshold = float(getattr(self, "late_stage_threshold", 0.8))
            late_stage_rule = str(
                _LATE_STAGE_POSITIVE_CONFIG.get("rule", _LATE_STAGE_POSITIVE_RULE)
            )
            if not bool(getattr(self, "_recap_episode_loader_ready", False)):
                return
            if factor <= 1 and not curriculum_enabled:
                logger.info(
                    "numeric-adv positive oversample enabled=false factor=%d late_stage_positive_enabled=%s late_stage_threshold=%.6f late_stage_rule=%s reason=disabled",
                    factor,
                    late_stage_positive_enabled,
                    late_stage_threshold,
                    late_stage_rule,
                )
                logger.info(
                    "numeric-adv positive curriculum enabled=false negative retain probability=%.6f seed=%d late_stage_positive_enabled=%s late_stage_threshold=%.6f late_stage_rule=%s reason=disabled",
                    negative_retain_probability,
                    curriculum_seed,
                    late_stage_positive_enabled,
                    late_stage_threshold,
                    late_stage_rule,
                )
                return

            sharded_episodes = getattr(self, "sharded_episodes", None)
            if not isinstance(sharded_episodes, list):
                logger.warning(
                    "numeric-adv positive oversample enabled=%s factor=%d positive curriculum enabled=%s negative retain probability=%.6f seed=%d late_stage_positive_enabled=%s late_stage_threshold=%.6f late_stage_rule=%s reason=missing_sharded_episodes",
                    factor > 1,
                    factor,
                    curriculum_enabled,
                    negative_retain_probability,
                    curriculum_seed,
                    late_stage_positive_enabled,
                    late_stage_threshold,
                    late_stage_rule,
                )
                return

            oversampled_shards, stats = self._positive_shard_scan(sharded_episodes)
            if int(stats["positive_step_refs_before"]) <= 0:
                logger.info(
                    "numeric-adv positive oversample enabled=%s factor=%d positive curriculum enabled=%s negative retain probability=%.6f seed=%d late_stage_positive_enabled=%s late_stage_threshold=%.6f late_stage_rule=%s structure=sharded_episodes positive_step_refs_before=0 reason=no_positive_steps_found",
                    factor > 1,
                    factor,
                    curriculum_enabled,
                    negative_retain_probability,
                    curriculum_seed,
                    late_stage_positive_enabled,
                    late_stage_threshold,
                    late_stage_rule,
                )
                return

            if int(stats["total_step_refs_after"]) <= 0:
                logger.warning(
                    "numeric-adv positive oversample enabled=%s factor=%d positive curriculum enabled=%s negative retain probability=%.6f seed=%d late_stage_positive_enabled=%s late_stage_threshold=%.6f late_stage_rule=%s structure=sharded_episodes total_step_refs_after=0 reason=no_retained_steps_after_curriculum",
                    factor > 1,
                    factor,
                    curriculum_enabled,
                    negative_retain_probability,
                    curriculum_seed,
                    late_stage_positive_enabled,
                    late_stage_threshold,
                    late_stage_rule,
                )
                return

            self.sharded_episodes = oversampled_shards
            self.shard_lengths = runtime_np.asarray(
                [
                    sum(int(len(step_indices)) for _, step_indices in shard)
                    for shard in oversampled_shards
                ],
                dtype=runtime_np.int64,
            )
            mode = "duplicate_positive_steps"
            if curriculum_enabled and factor > 1:
                mode = "positive_heavy_curriculum_then_duplicate_positive_steps"
            elif curriculum_enabled:
                mode = "positive_heavy_curriculum"
            if late_stage_positive_enabled and curriculum_enabled and factor > 1:
                mode = "late_stage_positive_curriculum_then_duplicate_selected_positive_steps"
            elif late_stage_positive_enabled and curriculum_enabled:
                mode = "late_stage_positive_curriculum"
            elif late_stage_positive_enabled and factor > 1:
                mode = "duplicate_selected_late_stage_positive_steps"
            logger.info(
                "numeric-adv positive oversample enabled=%s factor=%d positive curriculum enabled=%s negative retain probability=%.6f seed=%d late_stage_positive_enabled=%s late_stage_threshold=%.6f late_stage_rule=%s structure=sharded_episodes mode=%s total_shards_before=%d positive_shards_before=%d positive_shard_ratio_before=%.6f total_shards_after=%d positive_shards_after=%d positive_shard_ratio_after=%.6f total_episode_refs_before=%d positive_episode_refs_before=%d positive_episode_ratio_before=%.6f total_episode_refs_after=%d positive_episode_refs_after=%d positive_episode_ratio_after=%.6f total_step_refs_before=%d positive_step_refs_before=%d negative_step_refs_before=%d late_stage_positive_step_refs_before=%d selected_positive_step_refs_before=%d positive_step_ratio_before=%.6f negative_step_ratio_before=%.6f late_stage_positive_step_ratio_before=%.6f selected_positive_step_ratio_before=%.6f late_stage_positive_within_positive_ratio_before=%.6f selected_positive_within_positive_ratio_before=%.6f total_step_refs_after=%d positive_step_refs_after=%d negative_step_refs_after=%d late_stage_positive_step_refs_after=%d selected_positive_step_refs_after=%d positive_step_ratio_after=%.6f negative_step_ratio_after=%.6f late_stage_positive_step_ratio_after=%.6f selected_positive_step_ratio_after=%.6f late_stage_positive_within_positive_ratio_after=%.6f selected_positive_within_positive_ratio_after=%.6f",
                factor > 1,
                factor,
                curriculum_enabled,
                negative_retain_probability,
                curriculum_seed,
                late_stage_positive_enabled,
                late_stage_threshold,
                late_stage_rule,
                mode,
                int(stats["total_shards_before"]),
                int(stats["positive_shards_before"]),
                float(stats["positive_shard_ratio_before"]),
                int(stats["total_shards_after"]),
                int(stats["positive_shards_after"]),
                float(stats["positive_shard_ratio_after"]),
                int(stats["total_episode_refs_before"]),
                int(stats["positive_episode_refs_before"]),
                float(stats["positive_episode_ratio_before"]),
                int(stats["total_episode_refs_after"]),
                int(stats["positive_episode_refs_after"]),
                float(stats["positive_episode_ratio_after"]),
                int(stats["total_step_refs_before"]),
                int(stats["positive_step_refs_before"]),
                int(stats["negative_step_refs_before"]),
                int(stats["late_stage_positive_step_refs_before"]),
                int(stats["selected_positive_step_refs_before"]),
                float(stats["positive_step_ratio_before"]),
                float(stats["negative_step_ratio_before"]),
                float(stats["late_stage_positive_step_ratio_before"]),
                float(stats["selected_positive_step_ratio_before"]),
                float(stats["late_stage_positive_within_positive_ratio_before"]),
                float(stats["selected_positive_within_positive_ratio_before"]),
                int(stats["total_step_refs_after"]),
                int(stats["positive_step_refs_after"]),
                int(stats["negative_step_refs_after"]),
                int(stats["late_stage_positive_step_refs_after"]),
                int(stats["selected_positive_step_refs_after"]),
                float(stats["positive_step_ratio_after"]),
                float(stats["negative_step_ratio_after"]),
                float(stats["late_stage_positive_step_ratio_after"]),
                float(stats["selected_positive_step_ratio_after"]),
                float(stats["late_stage_positive_within_positive_ratio_after"]),
                float(stats["selected_positive_within_positive_ratio_after"]),
            )

        def _get_row_value(
            self, episode_data: pd.DataFrame, step_index: int, column: str
        ) -> Any:
            columns = getattr(episode_data, "columns", None)
            if columns is None:
                raise TypeError(
                    "episode_data must be a pandas DataFrame with columns; "
                    f"got {type(episode_data).__name__}"
                )
            if column not in columns:
                raise KeyError(
                    f"Missing required RECAP advantage column {column!r}; "
                    f"available columns: {list(columns)}"
                )
            if not isinstance(step_index, numbers.Integral):
                raise TypeError(
                    f"step_index must be integral for RECAP advantage lane, got {type(step_index).__name__}"
                )
            step_i = int(step_index)
            if step_i < 0 or step_i >= len(episode_data):
                raise IndexError(
                    f"step_index out of range for RECAP advantage lane: {step_i} not in [0, {len(episode_data)})"
                )
            return episode_data[column].iloc[step_i]

        def resolve_advantage(
            self, episode_data: pd.DataFrame, step_index: int
        ) -> Float32Array:
            raw = self._get_row_value(episode_data, step_index, self.advantage_column)
            value = validate_advantage_input_value(raw, context=self.advantage_column)
            return runtime_np.asarray([value], dtype=runtime_np.float32)

        def _log_advantage_item(
            self,
            *,
            step_index: int,
            advantage: Float32Array,
        ) -> None:
            remaining = int(getattr(self, "_advantage_debug_log_budget", 0))
            if remaining <= 0:
                return
            self._advantage_debug_log_budget = remaining - 1
            logger.info(
                "numeric-adv dataset item advantage shape=%s step_index=%d value=%.6f",
                list(advantage.shape),
                int(step_index),
                float(advantage.reshape(-1)[0]),
            )

        def get_datapoint(
            self, episode_data: pd.DataFrame, step_index: int
        ) -> dict[str, Any]:
            assert self.processor is not None, (
                "Processor must be set before getting datapoints"
            )
            vla_step_data = imported_extract_step_data(
                episode_data,
                step_index,
                self.modality_configs,
                self.embodiment_tag,
                self.allow_padding,
            )
            messages = [
                {
                    "type": ImportedMessageType.EPISODE_STEP.value,
                    "content": vla_step_data,
                }
            ]
            datapoint = self.processor(messages)
            advantage = self.resolve_advantage(episode_data, step_index)
            datapoint["advantage"] = advantage
            self._log_advantage_item(step_index=int(step_index), advantage=advantage)
            return datapoint


__all__ = [
    "RecapAdvantageShardedSingleStepDataset",
    "RecapLeRobotEpisodeLoader",
    "configure_late_stage_positive_emphasis",
    "configure_positive_curriculum",
    "configure_positive_oversampling",
]
