# pyright: reportMissingImports=false
from __future__ import annotations

import math
import numbers
import random
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

_dataset_import_error: ModuleNotFoundError | None = None
try:
    from .dataset import RecapLeRobotEpisodeLoader
except ModuleNotFoundError as exc:
    _dataset_import_error = exc

    class RecapLeRobotEpisodeLoader:
        def __init__(self, *args: Any, **kwargs: Any):
            del args, kwargs
            raise ModuleNotFoundError(
                "TextIndicatorShardedSingleStepDataset requires work.recap.dataset"
            ) from _dataset_import_error


TEXT_INDICATOR_OMIT = "omit"
TEXT_INDICATOR_NEGATIVE = "negative"
TEXT_INDICATOR_POSITIVE = "positive"
RECAP_TEXT_INDICATOR_SCHEMA_VERSION = "recap_text_indicator_v1"
RECAP_TEXT_INDICATOR_AUTHORITY_NAME = "recap_text_indicator_v1"
RECAP_TEXT_INDICATOR_CARRIER_FIELD = "carrier_text_v1"
RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD = "prompt_raw"
CANONICAL_TEXT_INDICATOR_STATES: tuple[str, str, str] = (
    TEXT_INDICATOR_OMIT,
    TEXT_INDICATOR_NEGATIVE,
    TEXT_INDICATOR_POSITIVE,
)
CANONICAL_NEGATIVE_LINE = "Advantage: negative"
CANONICAL_POSITIVE_LINE = "Advantage: positive"
DEFAULT_INDICATOR_DROPOUT_P = 0.0
PAPER_RECAP_INDICATOR_DROPOUT_P = 0.3


class SupportsTextIndicatorDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...


def normalize_indicator_mode(raw: Any, *, field_name: str = "indicator_mode") -> str:
    if raw is None:
        raise ValueError(f"{field_name} is required; expected omit|negative|positive")
    if not isinstance(raw, str):
        raise TypeError(
            f"{field_name} must be a string; expected omit|negative|positive, got {type(raw).__name__}"
        )
    mode = raw.strip().lower()
    if mode not in CANONICAL_TEXT_INDICATOR_STATES:
        expected = "|".join(CANONICAL_TEXT_INDICATOR_STATES)
        raise ValueError(f"Unknown {field_name}: {raw!r}; expected {expected}")
    return mode


def normalize_indicator_dropout_p(
    raw: Any,
    *,
    field_name: str = "indicator_dropout_p",
) -> float:
    if isinstance(raw, bool):
        raise TypeError(f"{field_name} must be a float in [0, 1], got bool")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a float in [0, 1], got {raw!r}") from exc
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {raw!r}")
    return value


def apply_indicator_dropout(
    indicator_mode: Any,
    *,
    dropout_p: Any = DEFAULT_INDICATOR_DROPOUT_P,
    seed: int = 0,
    sample_key: Any = "",
) -> str:
    mode = normalize_indicator_mode(indicator_mode, field_name="indicator_mode")
    probability = normalize_indicator_dropout_p(dropout_p)
    if mode == TEXT_INDICATOR_OMIT or probability <= 0.0:
        return mode
    if probability >= 1.0:
        return TEXT_INDICATOR_OMIT
    rng = random.Random(f"{int(seed)}::{sample_key}")
    return TEXT_INDICATOR_OMIT if rng.random() < probability else mode


def require_prompt_raw(prompt_raw: Any, *, field_name: str = "prompt_raw") -> str:
    if prompt_raw is None:
        raise ValueError(
            f"{field_name} is missing; canonical text-indicator lane requires prompt_raw"
        )
    if isinstance(prompt_raw, numbers.Real) and math.isnan(float(prompt_raw)):
        raise ValueError(
            f"{field_name} is NaN; canonical text-indicator lane rejects NaN prompt text"
        )
    text = str(prompt_raw)
    if not text.strip():
        raise ValueError(
            f"{field_name} is empty; canonical text-indicator lane requires non-empty task text"
        )
    return text


def indicator_mode_from_indicator_value(
    raw: Any, *, field_name: str = "recap_m2.indicator_I"
) -> str:
    if raw is None:
        raise ValueError(
            f"{field_name} is missing; canonical text-indicator lane rejects missing indicator state"
        )
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in CANONICAL_TEXT_INDICATOR_STATES:
            return normalize_indicator_mode(s, field_name=field_name)
        if s in {"0", "0.0", "false"}:
            return TEXT_INDICATOR_NEGATIVE
        if s in {"1", "1.0", "true"}:
            return TEXT_INDICATOR_POSITIVE
        raise ValueError(
            f"{field_name} has unknown string value {raw!r}; expected 0/1 or omit|negative|positive"
        )
    if isinstance(raw, bool):
        return TEXT_INDICATOR_POSITIVE if raw else TEXT_INDICATOR_NEGATIVE
    if not isinstance(raw, numbers.Real):
        raise TypeError(
            f"{field_name} must be numeric or string-like; got {type(raw).__name__}"
        )
    value = float(raw)
    if math.isnan(value):
        raise ValueError(
            f"{field_name} is NaN; canonical text-indicator lane rejects NaN indicator state"
        )
    if value == 0.0:
        return TEXT_INDICATOR_NEGATIVE
    if value == 1.0:
        return TEXT_INDICATOR_POSITIVE
    raise ValueError(f"{field_name} must be exactly 0 or 1, got {value!r}")


def build_canonical_text_indicator(prompt_raw: Any, indicator_mode: Any) -> str:
    prompt = require_prompt_raw(prompt_raw, field_name="prompt_raw")
    mode = normalize_indicator_mode(indicator_mode, field_name="indicator_mode")
    if mode == TEXT_INDICATOR_OMIT:
        return prompt
    if mode == TEXT_INDICATOR_NEGATIVE:
        return f"{prompt}\n{CANONICAL_NEGATIVE_LINE}"
    return f"{prompt}\n{CANONICAL_POSITIVE_LINE}"


def build_authoritative_carrier_text_v1(prompt_raw: Any, indicator_mode: Any) -> str:
    authority_record = build_recap_text_indicator_v1_record(prompt_raw, indicator_mode)
    return require_prompt_raw(
        authority_record[RECAP_TEXT_INDICATOR_CARRIER_FIELD],
        field_name=RECAP_TEXT_INDICATOR_CARRIER_FIELD,
    )


def require_authoritative_carrier_text_v1(
    carrier_text_v1: Any,
    *,
    prompt_raw: Any,
    indicator_mode: Any,
    field_name: str = RECAP_TEXT_INDICATOR_CARRIER_FIELD,
) -> str:
    carrier_text = require_prompt_raw(carrier_text_v1, field_name=field_name)
    expected = build_authoritative_carrier_text_v1(prompt_raw, indicator_mode)
    if carrier_text != expected:
        raise ValueError(
            f"{field_name} must match the canonical prompt_raw + indicator_I text-indicator carrier"
        )
    return carrier_text


def _has_non_empty_text(raw: Any) -> bool:
    if raw is None:
        return False
    return bool(str(raw).strip())


def validate_recap_text_indicator_v1_authority(
    *,
    carrier_text_v1: Any,
    prompt_conditioned: Any = None,
    policy_condition_text: Any = None,
    advantage_input: Any = None,
    dual_task_text: Any = None,
) -> None:
    authority_text = require_prompt_raw(
        carrier_text_v1,
        field_name=RECAP_TEXT_INDICATOR_CARRIER_FIELD,
    )
    if _has_non_empty_text(prompt_conditioned):
        prompt_conditioned_text = str(prompt_conditioned)
        if prompt_conditioned_text != authority_text:
            raise ValueError(
                "recap_text_indicator_v1 forbids non-canonical prompt_conditioned authority; "
                + "carrier_text_v1 must remain the only authoritative text carrier"
            )
    if _has_non_empty_text(policy_condition_text):
        raise ValueError(
            "recap_text_indicator_v1 forbids policy_condition_text in the same authority record; "
            + "that text belongs to a separate state-conditioned lane"
        )
    if advantage_input is not None:
        raise ValueError(
            "recap_text_indicator_v1 forbids numeric advantage passthrough in the authority carrier"
        )
    if bool(dual_task_text):
        raise ValueError(
            "recap_text_indicator_v1 forbids dual_task_text authority because it creates multiple text carriers"
        )


def build_recap_text_indicator_v1_record(
    prompt_raw: Any,
    indicator_mode: Any,
    *,
    prompt_conditioned: Any = None,
    policy_condition_text: Any = None,
    advantage_input: Any = None,
    dual_task_text: Any = None,
) -> dict[str, Any]:
    prompt = require_prompt_raw(
        prompt_raw,
        field_name=RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD,
    )
    mode = normalize_indicator_mode(indicator_mode, field_name="indicator_mode")
    carrier_text_v1 = build_canonical_text_indicator(prompt, mode)
    validate_recap_text_indicator_v1_authority(
        carrier_text_v1=carrier_text_v1,
        prompt_conditioned=prompt_conditioned,
        policy_condition_text=policy_condition_text,
        advantage_input=advantage_input,
        dual_task_text=dual_task_text,
    )
    return {
        "schema_version": RECAP_TEXT_INDICATOR_SCHEMA_VERSION,
        "authority_name": RECAP_TEXT_INDICATOR_AUTHORITY_NAME,
        "carrier_field": RECAP_TEXT_INDICATOR_CARRIER_FIELD,
        RECAP_TEXT_INDICATOR_CARRIER_FIELD: carrier_text_v1,
        "indicator_mode": mode,
        "source_prompt_field": RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD,
        "source_prompt_text": prompt,
        "prompt_conditioned_role": "non_authority_sidecar_only",
        "policy_condition_text_role": "separate_state_conditioned_lane_not_authority",
        "advantage_input_role": "numeric_sidecar_not_authority",
        "dual_task_text_role": "multi_text_legacy_not_authority",
    }


def canonical_text_indicator_metadata(indicator_mode: Any) -> dict[str, Any]:
    mode = normalize_indicator_mode(indicator_mode, field_name="indicator_mode")
    return {
        "schema_version": RECAP_TEXT_INDICATOR_SCHEMA_VERSION,
        "authority_name": RECAP_TEXT_INDICATOR_AUTHORITY_NAME,
        "carrier_field": RECAP_TEXT_INDICATOR_CARRIER_FIELD,
        "source_prompt_field": RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD,
        "carrier": "text/conditioning-info",
        "carrier_mode": "text_indicator",
        "indicator_mode": mode,
        "prompt_conditioned_role": "non_authority_sidecar_only",
        "policy_condition_text_role": "separate_state_conditioned_lane_not_authority",
        "advantage_input_role": "numeric_sidecar_not_authority",
        "dual_task_text_role": "multi_text_legacy_not_authority",
        "deviation_from_public_official_contract": False,
        "formalize_language": False,
        "prompt_conditioned_dependency": False,
        "dual_task_text_dependency": False,
        "mix50_dependency": False,
        "numeric_advantage_dependency": False,
        "surface_form": (
            "prompt_raw"
            if mode == TEXT_INDICATOR_OMIT
            else CANONICAL_NEGATIVE_LINE
            if mode == TEXT_INDICATOR_NEGATIVE
            else CANONICAL_POSITIVE_LINE
        ),
    }


def require_formalize_language_false(processor: Any) -> None:
    value = getattr(processor, "formalize_language", None)
    if value is not False:
        raise ValueError(
            "Canonical text-indicator lane requires processor.formalize_language=False; "
            f"got {value!r}"
        )


ImportedShardedSingleStepDataset: Any = object
imported_extract_step_data: Any = None
ImportedMessageType: Any = None

_import_error: ModuleNotFoundError | None = None
try:
    from gr00t.data.dataset.sharded_single_step_dataset import (
        ShardedSingleStepDataset as ImportedShardedSingleStepDataset,
        extract_step_data as imported_extract_step_data,
    )
    from gr00t.data.types import MessageType as ImportedMessageType
except ModuleNotFoundError as exc:
    _import_error = exc


if _import_error is not None or _dataset_import_error is not None:

    class _UnavailableTextIndicatorShardedSingleStepDataset:
        def __init__(self, *args: Any, **kwargs: Any):
            del args, kwargs
            cause = (
                _dataset_import_error
                if _dataset_import_error is not None
                else _import_error
            )
            message = (
                "TextIndicatorShardedSingleStepDataset requires gr00t and "
                "work.recap.dataset to be installed"
            )
            raise ModuleNotFoundError(message) from cause

    TextIndicatorShardedSingleStepDataset = (
        _UnavailableTextIndicatorShardedSingleStepDataset
    )

else:

    class _InstalledTextIndicatorShardedSingleStepDataset(
        ImportedShardedSingleStepDataset
    ):
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
            indicator_column: str = "recap_m2.indicator_I",
            prompt_raw_column: str = "recap_m2.prompt_raw",
            fixed_indicator_mode: str | None = None,
            indicator_dropout_p: float = DEFAULT_INDICATOR_DROPOUT_P,
            indicator_dropout_seed: int = 0,
            fallback_to_step_text: bool = True,
        ):
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
            self.indicator_column = str(indicator_column)
            self.prompt_raw_column = str(prompt_raw_column)
            self.indicator_dropout_p = normalize_indicator_dropout_p(
                indicator_dropout_p,
                field_name="indicator_dropout_p",
            )
            self.indicator_dropout_seed = int(indicator_dropout_seed)
            self.fallback_to_step_text = bool(fallback_to_step_text)
            self.fixed_indicator_mode = (
                normalize_indicator_mode(
                    fixed_indicator_mode, field_name="fixed_indicator_mode"
                )
                if fixed_indicator_mode is not None
                else None
            )
            self.episode_loader = RecapLeRobotEpisodeLoader(
                dataset_path=dataset_path,
                modality_configs=modality_configs,
                video_backend=video_backend,
                video_backend_kwargs=video_backend_kwargs,
                passthrough_scalar_columns=(self.indicator_column,),
                optional_scalar_columns=(
                    () if not self.prompt_raw_column else (self.prompt_raw_column,)
                ),
            )
            self.shard_dataset()

        def _has_column(self, episode_data: pd.DataFrame, column: str) -> bool:
            columns = getattr(episode_data, "columns", None)
            return columns is not None and column in columns

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
                    f"Missing required canonical text-indicator column {column!r}; "
                    f"available columns: {list(columns)}"
                )
            if not isinstance(step_index, numbers.Integral):
                raise TypeError(
                    f"step_index must be integral for canonical text-indicator lane, got {type(step_index).__name__}"
                )
            step_i = int(step_index)
            if step_i < 0 or step_i >= len(episode_data):
                raise IndexError(
                    f"step_index out of range for canonical text-indicator lane: {step_i} not in [0, {len(episode_data)})"
                )
            return episode_data[column].iloc[step_i]

        def _optional_row_value(
            self, episode_data: pd.DataFrame, step_index: int, column: str
        ) -> Any:
            if not column or not self._has_column(episode_data, column):
                return None
            return self._get_row_value(episode_data, step_index, column)

        def _dropout_sample_key(
            self,
            episode_data: pd.DataFrame,
            step_index: int,
            raw_mode: str,
        ) -> str:
            episode_index = self._optional_row_value(
                episode_data, step_index, "episode_index"
            )
            if episode_index is None:
                episode_index = self._optional_row_value(
                    episode_data, step_index, "episode_idx"
                )
            global_index = self._optional_row_value(episode_data, step_index, "index")
            return (
                f"episode={episode_index if episode_index is not None else 'unknown'}"
                f"|index={global_index if global_index is not None else 'unknown'}"
                f"|step={int(step_index)}|raw={raw_mode}"
            )

        def resolve_indicator_mode(
            self, episode_data: pd.DataFrame, step_index: int
        ) -> str:
            if self.fixed_indicator_mode is not None:
                raw_mode = self.fixed_indicator_mode
            else:
                raw = self._get_row_value(episode_data, step_index, self.indicator_column)
                raw_mode = indicator_mode_from_indicator_value(
                    raw, field_name=self.indicator_column
                )
            return apply_indicator_dropout(
                raw_mode,
                dropout_p=self.indicator_dropout_p,
                seed=self.indicator_dropout_seed,
                sample_key=self._dropout_sample_key(episode_data, step_index, raw_mode),
            )

        def resolve_prompt_raw(
            self,
            episode_data: pd.DataFrame,
            step_index: int,
            *,
            fallback_prompt_raw: Any = None,
        ) -> str:
            raw = self._optional_row_value(
                episode_data, step_index, self.prompt_raw_column
            )
            if raw is None and self.fallback_to_step_text:
                raw = fallback_prompt_raw
            return require_prompt_raw(raw, field_name=self.prompt_raw_column)

        def build_step_text(
            self,
            episode_data: pd.DataFrame,
            step_index: int,
            *,
            fallback_prompt_raw: Any = None,
        ) -> str:
            prompt_raw = self.resolve_prompt_raw(
                episode_data,
                step_index,
                fallback_prompt_raw=fallback_prompt_raw,
            )
            indicator_mode = self.resolve_indicator_mode(episode_data, step_index)
            return build_canonical_text_indicator(prompt_raw, indicator_mode)

        def get_datapoint(
            self, episode_data: pd.DataFrame, step_index: int
        ) -> dict[str, Any]:
            assert self.processor is not None, (
                "Processor must be set before getting datapoints"
            )
            require_formalize_language_false(self.processor)
            vla_step_data = imported_extract_step_data(
                episode_data,
                step_index,
                self.modality_configs,
                self.embodiment_tag,
                self.allow_padding,
            )
            vla_step_data.text = self.build_step_text(
                episode_data,
                step_index,
                fallback_prompt_raw=getattr(vla_step_data, "text", None),
            )
            messages = [
                {
                    "type": ImportedMessageType.EPISODE_STEP.value,
                    "content": vla_step_data,
                }
            ]
            return self.processor(messages)

    TextIndicatorShardedSingleStepDataset = (
        _InstalledTextIndicatorShardedSingleStepDataset
    )


__all__ = [
    "TEXT_INDICATOR_OMIT",
    "TEXT_INDICATOR_NEGATIVE",
    "TEXT_INDICATOR_POSITIVE",
    "RECAP_TEXT_INDICATOR_SCHEMA_VERSION",
    "RECAP_TEXT_INDICATOR_AUTHORITY_NAME",
    "RECAP_TEXT_INDICATOR_CARRIER_FIELD",
    "RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD",
    "CANONICAL_TEXT_INDICATOR_STATES",
    "CANONICAL_NEGATIVE_LINE",
    "CANONICAL_POSITIVE_LINE",
    "DEFAULT_INDICATOR_DROPOUT_P",
    "PAPER_RECAP_INDICATOR_DROPOUT_P",
    "SupportsTextIndicatorDataset",
    "normalize_indicator_mode",
    "normalize_indicator_dropout_p",
    "apply_indicator_dropout",
    "require_prompt_raw",
    "indicator_mode_from_indicator_value",
    "build_canonical_text_indicator",
    "build_authoritative_carrier_text_v1",
    "require_authoritative_carrier_text_v1",
    "validate_recap_text_indicator_v1_authority",
    "build_recap_text_indicator_v1_record",
    "canonical_text_indicator_metadata",
    "require_formalize_language_false",
    "TextIndicatorShardedSingleStepDataset",
]
