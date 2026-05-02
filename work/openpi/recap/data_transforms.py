from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from work.openpi.prompting.routes import (
    RECAP_RELABEL_CONSUMER_MODE,
    SHUFFLED_ADV_DIAG_CONSUMER_MODE,
)
from work.openpi.sources.libero_official.relabels import (
    materialize_dataset,
)

from . import dataset_aggregation
from .dataset import RecapDatasetBundle, resolve_recap_dataset
from .runtime_prompt import PromptSurfaceBundle, build_training_prompt_bundle
from .train_config import RepairedStageConfig, STAGE_RECAP_INFORMATIVE


DEFAULT_SOURCE_SMOKE_EPISODE_LIMIT = 8
INFORMATIVE_POSITIVE_REWEIGHT_KEY = "informative_positive_reweight"
INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME = (
    "informative_positive_episode_duplication_v2"
)
INFORMATIVE_POSITIVE_REWEIGHT_DUPLICATES_PER_EPISODE = 3
INFORMATIVE_POSITIVE_REWEIGHT_DIR_SUFFIX = (
    "_recap_informative_positive_episode_reweight_v2"
)
INFORMATIVE_POSITIVE_REWEIGHT_SKIP_REASON_CORRECTION_AUGMENTED = (
    "correction_augmented_trainer_surface"
)


@dataclass(frozen=True)
class PreparedStageDataset:
    dataset_dir: Path
    dataset_bundle: RecapDatasetBundle
    source_dataset_dir: Path
    materialization_report_path: Path | None
    prepared_from_source: bool


@dataclass(frozen=True)
class RepairedStagePromptTransform:
    stage: str
    consumer_mode: str
    fixed_indicator_mode: str | None
    critic_checkpoint_ref: str = "not_applicable"

    @staticmethod
    def _unwrap_scalar(value: Any) -> Any:
        if isinstance(value, (str, bytes)):
            return value
        item = getattr(value, "item", None)
        if not callable(item):
            return value
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            return value

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        prompt_bundle = build_training_prompt_bundle(
            {
                "prompt_raw": self._unwrap_scalar(data["prompt_raw"]),
                "recap_m2.indicator_I": self._unwrap_scalar(
                    data.get("recap_m2.indicator_I")
                ),
                "recap_m2.t": self._unwrap_scalar(data.get("recap_m2.t")),
                "episode_index": self._unwrap_scalar(data.get("episode_index")),
                "step_index": self._unwrap_scalar(data.get("step_index")),
                "action": self._unwrap_scalar(data.get("action")),
                "observation.state": self._unwrap_scalar(
                    data.get("observation.state", data.get("observation/state"))
                ),
                "prompt_conditioned": self._unwrap_scalar(
                    data.get("prompt_conditioned")
                ),
            },
            consumer_mode=self.consumer_mode,
            fixed_indicator_mode=self.fixed_indicator_mode,
            critic_checkpoint_ref=self.critic_checkpoint_ref,
        )
        updated = dict(data)
        updated["prompt"] = np.asarray(prompt_bundle.prompt_text)
        return updated


def _load_info(dataset_dir: Path) -> dict[str, object]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        return {}
    payload = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}


def _mapping_from_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _load_materialization_report(dataset_dir: Path) -> dict[str, object]:
    report_path = dataset_dir / "materialization_report.json"
    if not report_path.is_file():
        return {}
    payload = dataset_aggregation.read_json(report_path)
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}


def _import_pandas() -> Any:
    import pandas as pd  # type: ignore

    return pd


def _as_int_like(value: object, *, context: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{context} must be int-like, got {value!r}")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{context} must be integer-valued, got {value!r}")
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"{context} must be int-like, got {type(value).__name__}")


def _is_iteration_merged_dataset(dataset_dir: Path) -> bool:
    info = _load_info(dataset_dir)
    route_id = str(info.get("route_id", "")).strip()
    merged_route_id = str(info.get("merged_dataset_route_id", "")).strip()
    return (
        route_id == dataset_aggregation.MERGED_DATASET_ROUTE_ID
        or merged_route_id == dataset_aggregation.MERGED_DATASET_ROUTE_ID
    )


def _coerce_nonnegative_int(value: object, *, context: str) -> int | None:
    if value is None or value == "":
        return None
    resolved = _as_int_like(value, context=context)
    if resolved < 0:
        raise ValueError(f"{context} must be non-negative, got {resolved}")
    return int(resolved)


def _safe_fraction(*, numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return float(int(numerator) / int(denominator))


def _symlink_or_copy_file(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()
    try:
        destination_path.symlink_to(source_path.resolve())
    except OSError:
        _ = shutil.copy2(source_path, destination_path)


def _data_path_template_from_info(info: dict[str, object]) -> str:
    return str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    ).strip()


def _resolve_episode_data_path(
    *,
    dataset_dir: Path,
    data_path_template: str,
    chunks_size: int,
    episode_index: int,
    resolve: bool = True,
) -> Path:
    path = dataset_dir / data_path_template.format(
        episode_chunk=int(episode_index) // int(chunks_size),
        episode_index=int(episode_index),
    )
    if resolve:
        return path.resolve()
    return path


def _informative_positive_reweight_policy(
    dataset_dir: str | Path,
) -> dict[str, object] | None:
    info = _load_info(Path(dataset_dir).expanduser().resolve())
    contract = _mapping_from_object(info.get("recap_advantage_input_contract"))
    policy = _mapping_from_object(contract.get(INFORMATIVE_POSITIVE_REWEIGHT_KEY))
    if not policy:
        return None
    if bool(policy.get("enabled", True)) is False:
        return None
    if (
        str(policy.get("policy_name", "")).strip()
        != INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME
    ):
        return None
    if str(policy.get("applies_to_stage", "")).strip() != STAGE_RECAP_INFORMATIVE:
        return None
    duplicates_per_positive_episode = _coerce_nonnegative_int(
        policy.get("duplicates_per_positive_episode"),
        context=(
            f"recap_advantage_input_contract.{INFORMATIVE_POSITIVE_REWEIGHT_KEY}.duplicates_per_positive_episode"
        ),
    )
    if (
        duplicates_per_positive_episode
        != INFORMATIVE_POSITIVE_REWEIGHT_DUPLICATES_PER_EPISODE
    ):
        return None
    correction_signal = _mapping_from_object(policy.get("correction_signal"))
    source_dataset_dir = str(policy.get("source_dataset_dir", "")).strip()
    if source_dataset_dir:
        source_signal = _correction_signal_summary(
            Path(source_dataset_dir).expanduser().resolve()
        )
        if source_signal is not None and correction_signal != source_signal:
            return None
    return policy


def _resolve_informative_positive_reweight_source_dataset_dir(
    dataset_dir: str | Path,
) -> Path | None:
    policy = _informative_positive_reweight_policy(dataset_dir)
    if policy is None:
        return None
    raw_source = str(policy.get("source_dataset_dir", "")).strip()
    if not raw_source:
        return None
    return Path(raw_source).expanduser().resolve()


def _has_complete_informative_positive_reweight_surface(
    dataset_dir: str | Path,
) -> bool:
    resolved_dir = Path(dataset_dir).expanduser().resolve()
    if _informative_positive_reweight_policy(resolved_dir) is None:
        return False
    info = _load_info(resolved_dir)
    episodes_path = resolved_dir / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        return False
    try:
        chunks_size = _as_int_like(
            info.get("chunks_size", 1000), context="info.chunks_size"
        )
        episode_rows = dataset_aggregation.read_jsonl(episodes_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    data_path_template = _data_path_template_from_info(info)
    for row in episode_rows:
        try:
            episode_index = _as_int_like(
                row.get("episode_index", -1),
                context="episodes[].episode_index",
            )
        except ValueError:
            return False
        parquet_path = _resolve_episode_data_path(
            dataset_dir=resolved_dir,
            data_path_template=data_path_template,
            chunks_size=chunks_size,
            episode_index=episode_index,
            resolve=False,
        )
        if parquet_path.is_symlink() and not parquet_path.exists():
            return False
        if not parquet_path.is_file():
            return False
    return True


def _is_informative_positive_reweight_dataset(dataset_dir: str | Path) -> bool:
    return _has_complete_informative_positive_reweight_surface(dataset_dir)


def build_informative_positive_reweight_dataset_dir(
    source_dataset_dir: str | Path,
) -> Path:
    resolved_source = Path(source_dataset_dir).expanduser().resolve()
    return resolved_source.parent / (
        f"{resolved_source.name}{INFORMATIVE_POSITIVE_REWEIGHT_DIR_SUFFIX}"
    )


def _materialize_informative_positive_reweight_dataset(
    *,
    source_dataset_dir: str | Path,
    output_dataset_dir: str | Path,
) -> Path:
    source_dir = Path(source_dataset_dir).expanduser().resolve()
    output_dir = Path(output_dataset_dir).expanduser().resolve()
    if _has_complete_informative_positive_reweight_surface(output_dir):
        return output_dir

    if output_dir.exists() or output_dir.is_symlink():
        if output_dir.is_symlink() or output_dir.is_file():
            output_dir.unlink()
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_meta_dir = source_dir / "meta"
    target_meta_dir = output_dir / "meta"
    target_meta_dir.mkdir(parents=True, exist_ok=True)
    for meta_path in sorted(source_meta_dir.iterdir()):
        if meta_path.is_file():
            _ = shutil.copy2(meta_path, target_meta_dir / meta_path.name)

    source_info = _load_info(source_dir)
    source_contract = _mapping_from_object(
        source_info.get("recap_advantage_input_contract")
    )
    source_report_path = source_dir / "materialization_report.json"
    source_report = (
        dataset_aggregation.read_json(source_report_path)
        if source_report_path.is_file()
        else {}
    )
    correction_signal = _correction_signal_summary(source_dir)
    source_episode_rows = dataset_aggregation.read_jsonl(
        source_dir / "meta" / "episodes.jsonl"
    )
    data_path_template = _data_path_template_from_info(source_info)
    chunks_size = _as_int_like(
        source_info.get("chunks_size", 1000), context="info.chunks_size"
    )

    pd = _import_pandas()
    next_global_index = _coerce_nonnegative_int(
        source_info.get("total_frames"), context="info.total_frames"
    )
    if next_global_index is None:
        next_global_index = 0
    source_total_rows = 0
    source_positive_rows = 0
    source_positive_episode_count = 0
    source_positive_correction_episode_count = 0
    duplicated_total_rows = 0
    duplicated_positive_rows = 0
    duplicated_positive_episode_count = 0
    duplicated_correction_episode_count = 0
    duplicates_per_positive_episode = (
        INFORMATIVE_POSITIVE_REWEIGHT_DUPLICATES_PER_EPISODE
    )
    effective_episode_rows = [dict(row) for row in source_episode_rows]
    next_episode_index = 1 + max(
        _as_int_like(row.get("episode_index", -1), context="episodes[].episode_index")
        for row in source_episode_rows
    )
    for row in source_episode_rows:
        episode_index = _as_int_like(
            row.get("episode_index", -1), context="episodes[].episode_index"
        )
        parquet_path = _resolve_episode_data_path(
            dataset_dir=source_dir,
            data_path_template=data_path_template,
            chunks_size=chunks_size,
            episode_index=episode_index,
        )
        output_parquet_path = _resolve_episode_data_path(
            dataset_dir=output_dir,
            data_path_template=data_path_template,
            chunks_size=chunks_size,
            episode_index=episode_index,
        )
        frame = pd.read_parquet(parquet_path)
        positive_mask = frame["recap_m2.indicator_I"].astype(int) == 1
        positive_rows = int(positive_mask.sum())
        is_correction_episode = str(
            row.get("source_kind", "")
        ).strip() == "correction_segment" or bool(row.get("is_correction", False))
        source_total_rows += int(len(frame))
        source_positive_rows += int(positive_rows)
        if positive_rows > 0:
            source_positive_episode_count += 1
            if is_correction_episode:
                source_positive_correction_episode_count += 1

        _symlink_or_copy_file(parquet_path, output_parquet_path)
        if positive_rows <= 0:
            continue

        for _ in range(duplicates_per_positive_episode):
            duplicated_episode_frame = frame.copy()
            duplicated_episode_index = int(next_episode_index)
            if "episode_index" in duplicated_episode_frame.columns:
                duplicated_episode_frame.loc[:, "episode_index"] = [
                    duplicated_episode_index
                ] * len(duplicated_episode_frame)
            if "index" in duplicated_episode_frame.columns:
                duplicated_episode_frame.loc[:, "index"] = list(
                    range(
                        int(next_global_index),
                        int(next_global_index) + int(len(duplicated_episode_frame)),
                    )
                )
            next_global_index += int(len(duplicated_episode_frame))
            duplicated_total_rows += int(len(duplicated_episode_frame))
            duplicated_positive_rows += int(positive_rows)
            duplicated_positive_episode_count += 1
            if is_correction_episode:
                duplicated_correction_episode_count += 1
            duplicate_episode_row = dict(row)
            duplicate_episode_row["episode_index"] = duplicated_episode_index
            duplicate_episode_row["informative_positive_reweight_duplicate"] = True
            duplicate_episode_row[
                "informative_positive_reweight_source_episode_index"
            ] = episode_index
            effective_episode_rows.append(duplicate_episode_row)
            duplicate_parquet_path = _resolve_episode_data_path(
                dataset_dir=output_dir,
                data_path_template=data_path_template,
                chunks_size=chunks_size,
                episode_index=duplicated_episode_index,
            )
            duplicate_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            duplicated_episode_frame.to_parquet(
                duplicate_parquet_path,
                engine="pyarrow",
                index=False,
            )
            next_episode_index += 1

    effective_total_rows = int(source_total_rows + duplicated_total_rows)
    effective_positive_rows = int(source_positive_rows + duplicated_positive_rows)
    effective_total_episodes = int(
        len(source_episode_rows) + duplicated_positive_episode_count
    )
    effective_positive_episode_count = int(
        source_positive_episode_count + duplicated_positive_episode_count
    )
    source_positive_fraction = _safe_fraction(
        numerator=source_positive_rows,
        denominator=source_total_rows,
    )
    effective_positive_fraction = _safe_fraction(
        numerator=effective_positive_rows,
        denominator=effective_total_rows,
    )
    reweight_policy: dict[str, object] = {
        "applied": True,
        "enabled": True,
        "policy_name": INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME,
        "applies_to_stage": STAGE_RECAP_INFORMATIVE,
        "positive_indicator_value": 1,
        "duplication_unit": "episode",
        "positive_episode_selection": "episode_contains_positive_indicator_row",
        "duplicates_per_positive_episode": duplicates_per_positive_episode,
        "source_dataset_dir": str(source_dir),
        "source_total_episodes": int(len(source_episode_rows)),
        "source_positive_episode_count": int(source_positive_episode_count),
        "source_total_rows": int(source_total_rows),
        "source_positive_indicator_count": int(source_positive_rows),
        "source_positive_indicator_fraction": float(source_positive_fraction),
        "effective_total_episodes": int(effective_total_episodes),
        "effective_positive_episode_count": int(effective_positive_episode_count),
        "effective_total_rows": int(effective_total_rows),
        "effective_positive_indicator_count": int(effective_positive_rows),
        "effective_positive_indicator_fraction": float(effective_positive_fraction),
    }
    if correction_signal is not None:
        reweight_policy["correction_aware"] = True
        reweight_policy["correction_signal"] = correction_signal
        reweight_policy["source_positive_correction_episode_count"] = int(
            source_positive_correction_episode_count
        )
        reweight_policy["duplicated_correction_episode_count"] = int(
            duplicated_correction_episode_count
        )

    updated_contract = dict(source_contract)
    updated_contract[INFORMATIVE_POSITIVE_REWEIGHT_KEY] = reweight_policy
    updated_info = dict(source_info)
    updated_info["total_episodes"] = int(effective_total_episodes)
    updated_info["total_frames"] = int(effective_total_rows)
    updated_info["total_chunks"] = int(
        1 + ((effective_total_episodes - 1) // int(chunks_size))
    )
    updated_info["splits"] = {"train": f"0:{effective_total_episodes}"}
    updated_info["recap_advantage_input_contract"] = updated_contract
    _ = dataset_aggregation.write_json(target_meta_dir / "info.json", updated_info)
    _ = dataset_aggregation.write_jsonl(
        target_meta_dir / "episodes.jsonl",
        effective_episode_rows,
    )

    updated_report = dict(source_report)
    updated_report.setdefault(
        "schema_version", "openpi_libero_official_8d_recap_relabels_report_v1"
    )
    updated_report.setdefault("route_id", str(source_info.get("route_id", "")).strip())
    updated_report["final_status"] = "materialized"
    updated_report["source_dataset_dir"] = str(source_dir)
    updated_report["output_dataset_dir"] = str(output_dir)
    updated_report["selected_episode_count"] = int(effective_total_episodes)
    updated_report["selected_frame_count"] = int(effective_total_rows)
    updated_report["positive_indicator_count"] = int(effective_positive_rows)
    updated_report["positive_indicator_fraction"] = float(effective_positive_fraction)
    updated_report[INFORMATIVE_POSITIVE_REWEIGHT_KEY] = reweight_policy
    _ = dataset_aggregation.write_json(
        output_dir / "materialization_report.json", updated_report
    )

    for stale_meta in (
        target_meta_dir / "dataset_fingerprint.json",
        target_meta_dir / "episode_universe_hash.txt",
    ):
        if stale_meta.is_file():
            stale_meta.unlink()
    return output_dir


def _info_reports_autonomous_or_correction_rows(info: dict[str, object]) -> bool:
    episodes_added = _coerce_nonnegative_int(
        info.get("episodes_added"), context="info.episodes_added"
    )
    if episodes_added is not None and episodes_added > 0:
        return True
    corrections_added = _coerce_nonnegative_int(
        info.get("corrections_added"), context="info.corrections_added"
    )
    if corrections_added is not None and corrections_added > 0:
        return True
    dataset_mix = info.get("dataset_mix")
    if not isinstance(dataset_mix, dict):
        return False
    autonomous = dataset_mix.get("autonomous")
    if isinstance(autonomous, dict):
        autonomous_episodes = _coerce_nonnegative_int(
            autonomous.get("episodes"),
            context="info.dataset_mix.autonomous.episodes",
        )
        if autonomous_episodes is not None and autonomous_episodes > 0:
            return True
    correction = dataset_mix.get("correction")
    if isinstance(correction, dict):
        correction_segments = _coerce_nonnegative_int(
            correction.get("segments"),
            context="info.dataset_mix.correction.segments",
        )
        if correction_segments is not None and correction_segments > 0:
            return True
    return False


def _info_reports_correction_rows(info: dict[str, object]) -> bool:
    corrections_added = _coerce_nonnegative_int(
        info.get("corrections_added"), context="info.corrections_added"
    )
    if corrections_added is not None and corrections_added > 0:
        return True
    dataset_mix = info.get("dataset_mix")
    if not isinstance(dataset_mix, dict):
        return False
    correction = dataset_mix.get("correction")
    if not isinstance(correction, dict):
        return False
    correction_segments = _coerce_nonnegative_int(
        correction.get("segments"),
        context="info.dataset_mix.correction.segments",
    )
    return correction_segments is not None and correction_segments > 0


def _episodes_surface_has_autonomous_or_correction_rows(dataset_dir: Path) -> bool:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        return False
    for row in dataset_aggregation.read_jsonl(episodes_path):
        source_kind = str(row.get("source_kind", "")).strip()
        if source_kind in {"autonomous_trial", "correction_segment"}:
            return True
        if bool(row.get("is_correction", False)):
            return True
    return False


def _episodes_surface_has_correction_rows(dataset_dir: Path) -> bool:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        return False
    for row in dataset_aggregation.read_jsonl(episodes_path):
        if str(row.get("source_kind", "")).strip() == "correction_segment":
            return True
        if bool(row.get("is_correction", False)):
            return True
    return False


def _correction_signal_summary(dataset_dir: Path) -> dict[str, object] | None:
    info = _load_info(dataset_dir)
    report = _load_materialization_report(dataset_dir)
    signal: dict[str, object] = {}

    corrections_added = _coerce_nonnegative_int(
        info.get("corrections_added"), context="info.corrections_added"
    )
    if corrections_added is not None and corrections_added > 0:
        signal["corrections_added"] = int(corrections_added)

    dataset_mix = info.get("dataset_mix")
    if isinstance(dataset_mix, dict):
        correction = dataset_mix.get("correction")
        if isinstance(correction, dict):
            correction_segments = _coerce_nonnegative_int(
                correction.get("segments"),
                context="info.dataset_mix.correction.segments",
            )
            if correction_segments is not None and correction_segments > 0:
                signal["dataset_mix_correction_segments"] = int(correction_segments)

    merged_correction_override_count = _coerce_nonnegative_int(
        report.get("merged_correction_override_count"),
        context="materialization_report.merged_correction_override_count",
    )
    if (
        merged_correction_override_count is not None
        and merged_correction_override_count > 0
    ):
        signal["merged_correction_override_count"] = int(
            merged_correction_override_count
        )

    if _episodes_surface_has_correction_rows(dataset_dir):
        signal["episode_level_correction_rows_present"] = True

    if not signal:
        return None
    return signal


def _ensure_informative_positive_reweight_skip_metadata_for_corrections(
    dataset_dir: Path,
) -> dict[str, object] | None:
    correction_signal = _correction_signal_summary(dataset_dir)
    if correction_signal is None:
        return None

    skip_metadata: dict[str, object] = {
        "enabled": False,
        "applied": False,
        "policy_name": INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME,
        "applies_to_stage": STAGE_RECAP_INFORMATIVE,
        "skip_reason": (INFORMATIVE_POSITIVE_REWEIGHT_SKIP_REASON_CORRECTION_AUGMENTED),
        "source_dataset_dir": str(dataset_dir),
        "correction_signal": correction_signal,
    }

    info = _load_info(dataset_dir)
    contract = _mapping_from_object(info.get("recap_advantage_input_contract"))
    if contract.get(INFORMATIVE_POSITIVE_REWEIGHT_KEY) != skip_metadata:
        updated_contract = dict(contract)
        updated_contract[INFORMATIVE_POSITIVE_REWEIGHT_KEY] = skip_metadata
        updated_info = dict(info)
        updated_info["recap_advantage_input_contract"] = updated_contract
        _ = dataset_aggregation.write_json(
            dataset_dir / "meta" / "info.json", updated_info
        )

    report_path = dataset_dir / "materialization_report.json"
    if report_path.is_file():
        report = _load_materialization_report(dataset_dir)
        if report.get(INFORMATIVE_POSITIVE_REWEIGHT_KEY) != skip_metadata:
            updated_report = dict(report)
            updated_report[INFORMATIVE_POSITIVE_REWEIGHT_KEY] = skip_metadata
            _ = dataset_aggregation.write_json(report_path, updated_report)
    return skip_metadata


def source_dataset_uses_full_trainer_surface(dataset_dir: str | Path) -> bool:
    resolved_dir = Path(dataset_dir).expanduser().resolve()
    info = _load_info(resolved_dir)
    return (
        _is_iteration_merged_dataset(resolved_dir)
        or _info_reports_autonomous_or_correction_rows(info)
        or _episodes_surface_has_autonomous_or_correction_rows(resolved_dir)
    )


def resolve_prebuilt_training_dataset_dir(dataset_dir: str | Path) -> Path | None:
    resolved_dir = Path(dataset_dir).expanduser().resolve()
    if is_recap_training_dataset(resolved_dir):
        return resolved_dir
    info = _load_info(resolved_dir)
    raw_ref = str(
        info.get(dataset_aggregation.MERGED_RECAP_READY_DATASET_REF_KEY, "")
    ).strip()
    if not raw_ref:
        return None
    candidate = Path(raw_ref).expanduser().resolve()
    if not candidate.is_dir() or not is_recap_training_dataset(candidate):
        return None
    return candidate


def resolve_prepare_episode_limit(
    dataset_dir: str | Path,
    episode_limit: int | None,
) -> int | None:
    if episode_limit is not None:
        return int(episode_limit)
    if source_dataset_uses_full_trainer_surface(dataset_dir):
        return None
    return DEFAULT_SOURCE_SMOKE_EPISODE_LIMIT


def _prepared_parquet_path(dataset_dir: Path, episode_index: int) -> Path:
    info = _load_info(dataset_dir)
    data_path = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    ).strip()
    chunks_size = _as_int_like(
        info.get("chunks_size", 1000), context="info.chunks_size"
    )
    return (
        dataset_dir
        / data_path.format(
            episode_chunk=int(episode_index) // int(chunks_size),
            episode_index=int(episode_index),
        )
    ).resolve()


def _override_materialized_merged_episode_semantics(
    *,
    source_dataset_dir: Path,
    prepared_dataset_dir: Path,
    critic_checkpoint_dir: Path,
) -> None:
    episodes = dataset_aggregation.read_jsonl(
        source_dataset_dir / "meta" / "episodes.jsonl"
    )
    pd = _import_pandas()
    overridden_episode_count = 0
    correction_episode_count = 0
    for row in episodes:
        source_kind = str(row.get("source_kind", "")).strip()
        if source_kind not in {"autonomous_trial", "correction_segment"}:
            continue
        episode_index = _as_int_like(
            row.get("episode_index", -1), context="merged_episode.episode_index"
        )
        if episode_index < 0:
            raise ValueError(f"merged episode row missing episode_index: {row}")
        parquet_path = _prepared_parquet_path(prepared_dataset_dir, episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        frame = pd.read_parquet(parquet_path)
        if len(frame) <= 0:
            raise ValueError(
                f"prepared merged episode parquet is empty: {parquet_path}"
            )
        tasks = row.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise ValueError(
                f"merged episode tasks must be single-item list, got {tasks!r}"
            )
        prompt_raw = str(tasks[0]).strip()
        if not prompt_raw:
            raise ValueError(f"merged episode prompt_raw must be non-empty: {row}")
        indicator = (
            1
            if source_kind == "correction_segment"
            else _as_int_like(
                row.get("indicator_I", 0), context="merged_episode.indicator_I"
            )
        )
        prompt_bundle = build_training_prompt_bundle(
            {"prompt_raw": prompt_raw, "recap_m2.indicator_I": indicator},
            consumer_mode=RECAP_RELABEL_CONSUMER_MODE,
            fixed_indicator_mode=None,
            critic_checkpoint_ref=str(critic_checkpoint_dir),
        )
        signed_value = 1.0 if int(indicator) == 1 else -1.0
        if "recap_m2.indicator_I" in frame.columns:
            frame["recap_m2.indicator_I"] = [int(indicator)] * len(frame)
        if "recap_m2.prompt_raw" in frame.columns:
            frame["recap_m2.prompt_raw"] = [prompt_raw] * len(frame)
        if "recap_m2.prompt_conditioned" in frame.columns:
            frame["recap_m2.prompt_conditioned"] = [prompt_bundle.prompt_text] * len(
                frame
            )
        if "recap_m2.advantage_input" in frame.columns:
            frame["recap_m2.advantage_input"] = [float(signed_value)] * len(frame)
        if "recap_m2.advantage_A" in frame.columns:
            frame["recap_m2.advantage_A"] = [float(signed_value)] * len(frame)
        frame.to_parquet(parquet_path, engine="pyarrow", index=False)
        overridden_episode_count += 1
        if source_kind == "correction_segment":
            correction_episode_count += 1
    report_path = prepared_dataset_dir / "materialization_report.json"
    if report_path.is_file():
        report = dataset_aggregation.read_json(report_path)
        report["merged_dataset_route_id"] = dataset_aggregation.MERGED_DATASET_ROUTE_ID
        report["source_dataset_dir"] = str(source_dataset_dir)
        report["output_dataset_dir"] = str(prepared_dataset_dir)
        report["merged_episode_override_applied"] = True
        report["merged_episode_override_count"] = int(overridden_episode_count)
        report["merged_correction_override_count"] = int(correction_episode_count)
        dataset_aggregation.write_json(report_path, report)


def is_recap_training_dataset(dataset_dir: str | Path) -> bool:
    resolved_dir = Path(dataset_dir).expanduser().resolve()
    info = _load_info(resolved_dir)
    contract = info.get("recap_advantage_input_contract")
    features = info.get("features")
    if not isinstance(contract, dict) or not contract:
        return False
    if not isinstance(features, dict):
        return False
    core_keys = {
        "observation.images.ego_view",
        "observation.state",
        "action",
        "recap_m2.indicator_I",
    }
    feature_keys = {str(key) for key in features.keys()}
    if not core_keys.issubset(feature_keys):
        return False
    if {"recap_m2.prompt_raw", "recap_m2.prompt_conditioned"}.issubset(feature_keys):
        return True
    parquet_files = tuple(sorted(resolved_dir.glob("data/chunk-*/episode_*.parquet")))
    if not parquet_files:
        return False
    pd = _import_pandas()
    try:
        sample_frame = pd.read_parquet(
            parquet_files[0],
            columns=["recap_m2.prompt_raw", "recap_m2.prompt_conditioned"],
        )
    except Exception:
        return False
    sample_columns = {str(column) for column in sample_frame.columns}
    return {"recap_m2.prompt_raw", "recap_m2.prompt_conditioned"}.issubset(
        sample_columns
    )


def build_default_prepared_dataset_dir(
    source_dataset_dir: str | Path,
    *,
    critic_checkpoint_dir: str | Path,
    episode_limit: int | None,
) -> Path:
    resolved_source = Path(source_dataset_dir).expanduser().resolve()
    resolved_critic = Path(critic_checkpoint_dir).expanduser().resolve()
    critic_hash = hashlib.sha256(str(resolved_critic).encode("utf-8")).hexdigest()[:12]
    limit_token = (
        f"episodes_{int(episode_limit)}"
        if episode_limit is not None
        else "episodes_full"
    )
    return (
        resolved_source.parent
        / f"{resolved_source.name}_recap_task7_cache_{limit_token}_{critic_hash}"
    )


def build_stage_prompt_bundle(
    *,
    stage_config: RepairedStageConfig,
    label_row: dict[str, object],
    critic_checkpoint_ref: str = "not_applicable",
) -> PromptSurfaceBundle:
    return build_training_prompt_bundle(
        label_row,
        consumer_mode=stage_config.consumer_mode,
        fixed_indicator_mode=stage_config.fixed_indicator_mode,
        critic_checkpoint_ref=critic_checkpoint_ref,
    )


def _preview_consumer_mode(stage_config: RepairedStageConfig) -> str:
    if stage_config.indicator_mode_train == "informative":
        return RECAP_RELABEL_CONSUMER_MODE
    if stage_config.indicator_mode_train == "shuffled":
        return SHUFFLED_ADV_DIAG_CONSUMER_MODE
    return stage_config.consumer_mode


def prepare_stage_training_dataset(
    *,
    dataset_dir: str | Path,
    stage_config: RepairedStageConfig,
    critic_checkpoint_dir: str | Path,
    prepared_dataset_dir: str | Path | None = None,
    episode_limit: int | None = None,
) -> PreparedStageDataset:
    resolved_input = Path(dataset_dir).expanduser().resolve()
    prebuilt_training_dataset_dir = resolve_prebuilt_training_dataset_dir(
        resolved_input
    )
    if prebuilt_training_dataset_dir is not None:
        ready_dir = prebuilt_training_dataset_dir
        report_path = ready_dir / "materialization_report.json"
        prepared_from_source = False
    else:
        needs_full_trainer_surface = source_dataset_uses_full_trainer_surface(
            resolved_input
        )
        effective_episode_limit = resolve_prepare_episode_limit(
            resolved_input,
            episode_limit,
        )
        ready_dir = (
            Path(prepared_dataset_dir).expanduser().resolve()
            if prepared_dataset_dir is not None
            else build_default_prepared_dataset_dir(
                resolved_input,
                critic_checkpoint_dir=critic_checkpoint_dir,
                episode_limit=effective_episode_limit,
            )
        )
        if not is_recap_training_dataset(ready_dir):
            _ = materialize_dataset(
                official_dataset_dir=resolved_input,
                output_dir=ready_dir,
                episode_limit=effective_episode_limit,
                critic_checkpoint_dir=critic_checkpoint_dir,
            )
            if needs_full_trainer_surface:
                _override_materialized_merged_episode_semantics(
                    source_dataset_dir=resolved_input,
                    prepared_dataset_dir=ready_dir,
                    critic_checkpoint_dir=Path(critic_checkpoint_dir)
                    .expanduser()
                    .resolve(),
                )
        report_path = ready_dir / "materialization_report.json"
        prepared_from_source = True
    if stage_config.stage == STAGE_RECAP_INFORMATIVE:
        reweight_source_dir = _resolve_informative_positive_reweight_source_dataset_dir(
            ready_dir
        )
        if reweight_source_dir is not None:
            ready_dir = _materialize_informative_positive_reweight_dataset(
                source_dataset_dir=reweight_source_dir,
                output_dataset_dir=ready_dir,
            )
        elif not _is_informative_positive_reweight_dataset(ready_dir):
            ready_dir = _materialize_informative_positive_reweight_dataset(
                source_dataset_dir=ready_dir,
                output_dataset_dir=build_informative_positive_reweight_dataset_dir(
                    ready_dir
                ),
            )
        report_path = ready_dir / "materialization_report.json"
    dataset_bundle = resolve_recap_dataset(
        ready_dir,
        consumer_mode=_preview_consumer_mode(stage_config),
        fixed_indicator_mode=stage_config.fixed_indicator_mode,
    )
    dataset_bundle = replace(
        dataset_bundle,
        consumer_mode=stage_config.consumer_mode,
        fixed_indicator_mode=stage_config.fixed_indicator_mode,
    )
    return PreparedStageDataset(
        dataset_dir=ready_dir,
        dataset_bundle=dataset_bundle,
        source_dataset_dir=resolved_input,
        materialization_report_path=report_path if report_path.is_file() else None,
        prepared_from_source=prepared_from_source,
    )


__all__ = [
    "DEFAULT_SOURCE_SMOKE_EPISODE_LIMIT",
    "INFORMATIVE_POSITIVE_REWEIGHT_DIR_SUFFIX",
    "INFORMATIVE_POSITIVE_REWEIGHT_KEY",
    "INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME",
    "PreparedStageDataset",
    "RepairedStagePromptTransform",
    "build_default_prepared_dataset_dir",
    "build_informative_positive_reweight_dataset_dir",
    "build_stage_prompt_bundle",
    "is_recap_training_dataset",
    "prepare_stage_training_dataset",
    "resolve_prebuilt_training_dataset_dir",
    "resolve_prepare_episode_limit",
    "source_dataset_uses_full_trainer_surface",
]
