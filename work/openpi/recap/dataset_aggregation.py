from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from work.openpi.prompting.routes import RECAP_RELABEL_CONSUMER_MODE

from .runtime_prompt import build_training_prompt_bundle
from .thresholds import DEFAULT_EPSILON_QUANTILE, build_epsilon_source


SOURCE_IMAGE_COLUMN = "image"
SOURCE_WRIST_IMAGE_COLUMN = "wrist_image"
SOURCE_STATE_COLUMN = "state"
SOURCE_ACTION_COLUMN = "actions"
REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    SOURCE_IMAGE_COLUMN,
    SOURCE_WRIST_IMAGE_COLUMN,
    SOURCE_STATE_COLUMN,
    SOURCE_ACTION_COLUMN,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


COLLECTION_ROUTE_ID = "openpi_libero_recap_task9_collect_v1"
COLLECTION_SCHEMA_VERSION = "openpi_libero_recap_collection_v1"
COLLECTION_EPISODE_SCHEMA_VERSION = "openpi_libero_recap_collection_episode_v1"
CORRECTION_SEGMENT_SCHEMA_VERSION = "openpi_libero_recap_correction_segment_v1"
MERGED_DATASET_ROUTE_ID = "openpi_libero_recap_task9_merge_v1"
MERGED_DATASET_SCHEMA_VERSION = "openpi_libero_recap_iteration_dataset_v1"
MERGED_EPISODE_SCHEMA_VERSION = "openpi_libero_recap_iteration_episode_v1"
MERGED_EPISODE_LINEAGE_NAME = "merged_episode_lineage.jsonl"
MERGED_RECAP_READY_DATASET_DIRNAME = "_trainer_compatible_recap_surface"
MERGED_RECAP_READY_DATASET_REF_KEY = "trainer_compatible_recap_dataset_ref"
OFFICIAL_RECAP_RELABEL_ROUTE_ID = "official_native_8d_recap_relabels_v1"
ITERATION_MANIFEST_SCHEMA_VERSION = "openpi_libero_recap_iteration_manifest_v1"
LOOP_MANIFEST_SCHEMA_VERSION = "openpi_libero_recap_loop_manifest_v1"

AUTONOMOUS_TRIALS_NAME = "autonomous_trials.jsonl"
COLLECTION_MANIFEST_NAME = "collection_manifest.json"
CORRECTION_SEGMENTS_NAME = "correction_segments.jsonl"
CANONICAL_SOURCE_CHECK_NAME = "canonical_source_check.json"
MERGE_MANIFEST_NAME = "merge_manifest.json"
ITERATION_MANIFEST_NAME = "iteration_manifest.json"
LOOP_MANIFEST_NAME = "loop_manifest.json"


@dataclass(frozen=True)
class CheckpointLineage:
    policy_checkpoint_ref: str
    critic_checkpoint_ref: str
    stage: str | None
    source_dataset_dir: str | None
    prepared_dataset_dir: str | None
    train_manifest_ref: str | None
    checkpoint_provenance_ref: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_checkpoint_ref": self.policy_checkpoint_ref,
            "critic_checkpoint_ref": self.critic_checkpoint_ref,
            "stage": self.stage,
            "source_dataset_dir": self.source_dataset_dir,
            "prepared_dataset_dir": self.prepared_dataset_dir,
            "train_manifest_ref": self.train_manifest_ref,
            "checkpoint_provenance_ref": self.checkpoint_provenance_ref,
        }


@dataclass(frozen=True)
class CollectionBundle:
    output_dir: Path
    manifest_path: Path
    autonomous_trials_path: Path
    correction_segments_path: Path
    canonical_source_check_path: Path
    manifest: dict[str, object]
    autonomous_trials: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class MergedDatasetBundle:
    output_dir: Path
    merge_manifest_path: Path
    info_path: Path
    episodes_path: Path
    episode_lineage_path: Path
    canonical_source_check_path: Path
    merge_manifest: dict[str, object]


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _resolve_path(raw_path: str | Path) -> Path:
    return Path(raw_path).expanduser().resolve()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    output_path = _resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(
            _json_ready(dict(payload)), ensure_ascii=True, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path.replace(output_path)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, object]]) -> Path:
    output_path = _resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(_json_ready(dict(record)), ensure_ascii=True, sort_keys=True)
            )
            handle.write("\n")
    return tmp_path.replace(output_path)


def read_json(path: str | Path) -> dict[str, object]:
    payload = cast(object, json.loads(_resolve_path(path).read_text(encoding="utf-8")))
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"expected JSON object at {path}, got {type(payload).__name__}"
        )
    mapping_payload = cast(Mapping[object, object], payload)
    return {str(key): value for key, value in mapping_payload.items()}


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    resolved_path = _resolve_path(path)
    rows: list[dict[str, object]] = []
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    for line_number, line in enumerate(
        resolved_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text:
            continue
        payload = cast(object, json.loads(text))
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"expected JSON object in {resolved_path}:{line_number}, got {type(payload).__name__}"
            )
        mapping_payload = cast(Mapping[object, object], payload)
        rows.append({str(key): value for key, value in mapping_payload.items()})
    return rows


def read_demo_metadata(
    demo_dir: str | Path,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    demo_root = _resolve_path(demo_dir)
    meta_dir = demo_root / "meta"
    return (
        read_json(meta_dir / "info.json"),
        read_jsonl(meta_dir / "tasks.jsonl"),
        read_jsonl(meta_dir / "episodes.jsonl"),
    )


def parse_task_ids_csv(raw_task_ids: str) -> tuple[int, ...]:
    values = [chunk.strip() for chunk in str(raw_task_ids).split(",") if chunk.strip()]
    if not values:
        raise ValueError("task ids must be a non-empty comma-separated integer list")
    return tuple(int(value) for value in values)


def compute_selection_source_hash(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            _json_ready(dict(payload)), ensure_ascii=True, sort_keys=True
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _maybe_read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return read_json(path)


def _import_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"merged_dataset_requires_pandas: {exc}") from exc
    return pd


def _first_text(*values: object) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _nested_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    raw = mapping.get(key)
    if not isinstance(raw, Mapping):
        return {}
    nested = cast(Mapping[object, object], raw)
    return {str(child_key): value for child_key, value in nested.items()}


def resolve_checkpoint_lineage(
    policy_checkpoint: str | Path,
    *,
    explicit_critic_checkpoint: str | Path | None = None,
) -> CheckpointLineage:
    checkpoint_dir = _resolve_path(policy_checkpoint)
    candidate_dirs = [checkpoint_dir]
    if checkpoint_dir.parent not in candidate_dirs:
        candidate_dirs.append(checkpoint_dir.parent)

    train_manifest: dict[str, object] | None = None
    train_manifest_ref: str | None = None
    checkpoint_provenance: dict[str, object] | None = None
    checkpoint_provenance_ref: str | None = None
    for candidate_dir in candidate_dirs:
        if train_manifest is None:
            candidate_path = candidate_dir / "train_manifest.json"
            train_manifest = _maybe_read_json(candidate_path)
            if train_manifest is not None:
                train_manifest_ref = str(candidate_path.resolve())
        if checkpoint_provenance is None:
            candidate_path = candidate_dir / "checkpoint_provenance.json"
            checkpoint_provenance = _maybe_read_json(candidate_path)
            if checkpoint_provenance is not None:
                checkpoint_provenance_ref = str(candidate_path.resolve())

    training_route = (
        _nested_mapping(train_manifest, "training_route")
        if train_manifest is not None
        else {}
    )
    variant_derivation = (
        _nested_mapping(checkpoint_provenance, "variant_derivation")
        if checkpoint_provenance is not None
        else {}
    )

    inferred_critic_ref = _first_text(
        training_route.get("critic_checkpoint_ref"),
        variant_derivation.get("critic_checkpoint_ref"),
        (train_manifest or {}).get("critic_checkpoint_ref"),
        (checkpoint_provenance or {}).get("critic_checkpoint_ref"),
    )
    explicit_critic_ref = (
        str(_resolve_path(explicit_critic_checkpoint))
        if explicit_critic_checkpoint is not None
        else None
    )
    if explicit_critic_ref is not None and inferred_critic_ref is not None:
        resolved_inferred = str(_resolve_path(inferred_critic_ref))
        if resolved_inferred != explicit_critic_ref:
            raise ValueError(
                "critic checkpoint lineage mismatch between explicit CLI value and checkpoint provenance: "
                + f"explicit={explicit_critic_ref!r} inferred={resolved_inferred!r}"
            )
    critic_checkpoint_ref = (
        explicit_critic_ref
        or (
            str(_resolve_path(inferred_critic_ref))
            if inferred_critic_ref is not None
            else None
        )
        or "not_applicable"
    )
    return CheckpointLineage(
        policy_checkpoint_ref=str(checkpoint_dir),
        critic_checkpoint_ref=critic_checkpoint_ref,
        stage=_first_text(
            (train_manifest or {}).get("stage"),
            training_route.get("stage"),
            (checkpoint_provenance or {}).get("stage"),
            variant_derivation.get("stage"),
        ),
        source_dataset_dir=_first_text(
            training_route.get("source_dataset_dir"),
            variant_derivation.get("source_dataset_dir"),
        ),
        prepared_dataset_dir=_first_text(
            training_route.get("prepared_dataset_dir"),
            variant_derivation.get("prepared_dataset_dir"),
        ),
        train_manifest_ref=train_manifest_ref,
        checkpoint_provenance_ref=checkpoint_provenance_ref,
    )


def normalize_trace_row(
    row: Mapping[str, object],
    *,
    runtime_prompting: Mapping[str, object],
    policy_lineage: CheckpointLineage,
) -> dict[str, object]:
    task_id = int(cast(Any, row.get("task_id")))
    seed = int(cast(Any, row.get("seed")))
    trial_idx = int(cast(Any, row.get("trial_idx")))
    success = bool(cast(Any, row.get("success")))
    trial_id = f"task{task_id}_seed{seed}_trial{trial_idx}"
    prompt_text = str(runtime_prompting.get("prompt_text", "")).strip()
    return {
        "schema_version": COLLECTION_EPISODE_SCHEMA_VERSION,
        "trial_id": trial_id,
        "task_suite_name": str(row.get("task_suite_name", "libero_spatial")),
        "task_id": task_id,
        "seed": seed,
        "trial_idx": trial_idx,
        "success": success,
        "label": "success" if success else "failure",
        "indicator_I": 1 if success else 0,
        "is_correction": False,
        "forced_positive_indicator": False,
        "first_success_step": row.get("first_success_step"),
        "executed_steps": int(cast(Any, row.get("executed_steps"))),
        "max_steps_resolved": int(cast(Any, row.get("max_steps_resolved"))),
        "success_within_50pct_budget": bool(
            cast(Any, row.get("success_within_50pct_budget"))
        ),
        "success_within_75pct_budget": bool(
            cast(Any, row.get("success_within_75pct_budget"))
        ),
        "timeout_flag": bool(cast(Any, row.get("timeout_flag"))),
        "deviation_notes": [
            str(value)
            for value in cast(Sequence[object], row.get("deviation_notes", []))
        ],
        "episode_status": str(row.get("episode_status", "ok")),
        "error": str(row.get("error", "")),
        "indicator_mode": str(runtime_prompting.get("indicator_mode", "")),
        "indicator_source": str(runtime_prompting.get("indicator_source", "")),
        "prompt_text_surface": str(runtime_prompting.get("prompt_text_surface", "")),
        "prompt_text": prompt_text,
        "policy_checkpoint_ref": policy_lineage.policy_checkpoint_ref,
        "critic_checkpoint_ref": str(
            runtime_prompting.get("critic_checkpoint_ref")
            or policy_lineage.critic_checkpoint_ref
        ),
        "policy_stage": policy_lineage.stage,
    }


def build_collection_manifest(
    *,
    output_dir: Path,
    canonical_source_report: Mapping[str, object],
    policy_lineage: CheckpointLineage,
    task_suite_name: str,
    task_ids: Sequence[int],
    episodes_requested: int,
    episodes_materialized: int,
    rollout_manifest_path: Path,
    rollout_output_dir: Path,
    rollout_input_summary: Mapping[str, object],
    autonomous_trials_path: Path,
    correction_segments_path: Path,
) -> dict[str, object]:
    runtime_prompting = _nested_mapping(
        cast(Mapping[str, object], rollout_input_summary), "runtime_prompting"
    )
    return {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "route_id": COLLECTION_ROUTE_ID,
        "created_at": _now_iso(),
        "output_dir": str(output_dir),
        "task_suite_name": task_suite_name,
        "task_ids": [int(task_id) for task_id in task_ids],
        "episodes_requested": int(episodes_requested),
        "episodes_materialized": int(episodes_materialized),
        "policy_checkpoint_ref": policy_lineage.policy_checkpoint_ref,
        "critic_checkpoint_ref": policy_lineage.critic_checkpoint_ref,
        "policy_stage": policy_lineage.stage,
        "source_dataset_dir": policy_lineage.source_dataset_dir,
        "prepared_dataset_dir": policy_lineage.prepared_dataset_dir,
        "train_manifest_ref": policy_lineage.train_manifest_ref,
        "checkpoint_provenance_ref": policy_lineage.checkpoint_provenance_ref,
        "canonical_demo_source": dict(canonical_source_report),
        "rollout_manifest_ref": str(rollout_manifest_path),
        "rollout_output_dir": str(rollout_output_dir),
        "rollout_input_summary_ref": str(
            rollout_output_dir / "_staging" / "rollout_input_summary.json"
        ),
        "autonomous_trials_ref": str(autonomous_trials_path),
        "correction_segments_ref": str(correction_segments_path),
        "indicator_mode_requested": str(
            runtime_prompting.get("indicator_mode_requested", "")
        ),
        "indicator_mode": str(runtime_prompting.get("indicator_mode", "")),
        "indicator_source": str(runtime_prompting.get("indicator_source", "")),
        "prompt_text_surface": str(runtime_prompting.get("prompt_text_surface", "")),
        "prompt_text": str(runtime_prompting.get("prompt_text", "")),
    }


def materialize_collection_bundle(
    *,
    output_dir: str | Path,
    canonical_source_report: Mapping[str, object],
    policy_lineage: CheckpointLineage,
    task_suite_name: str,
    task_ids: Sequence[int],
    episodes_requested: int,
    rollout_manifest_path: str | Path,
    rollout_output_dir: str | Path,
    rollout_input_summary: Mapping[str, object],
    trace_rows: Sequence[Mapping[str, object]],
) -> CollectionBundle:
    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    runtime_prompting = _nested_mapping(
        cast(Mapping[str, object], rollout_input_summary), "runtime_prompting"
    )
    selected_trace_rows = list(trace_rows)[: int(episodes_requested)]
    normalized_rows = tuple(
        normalize_trace_row(
            row,
            runtime_prompting=runtime_prompting,
            policy_lineage=policy_lineage,
        )
        for row in selected_trace_rows
    )
    autonomous_trials_path = write_jsonl(
        resolved_output_dir / AUTONOMOUS_TRIALS_NAME,
        normalized_rows,
    )
    correction_segments_path = write_jsonl(
        resolved_output_dir / CORRECTION_SEGMENTS_NAME,
        [],
    )
    canonical_source_check_path = write_json(
        resolved_output_dir / CANONICAL_SOURCE_CHECK_NAME,
        dict(canonical_source_report),
    )
    manifest = build_collection_manifest(
        output_dir=resolved_output_dir,
        canonical_source_report=canonical_source_report,
        policy_lineage=policy_lineage,
        task_suite_name=task_suite_name,
        task_ids=task_ids,
        episodes_requested=episodes_requested,
        episodes_materialized=len(normalized_rows),
        rollout_manifest_path=_resolve_path(rollout_manifest_path),
        rollout_output_dir=_resolve_path(rollout_output_dir),
        rollout_input_summary=rollout_input_summary,
        autonomous_trials_path=autonomous_trials_path,
        correction_segments_path=correction_segments_path,
    )
    manifest_path = write_json(resolved_output_dir / COLLECTION_MANIFEST_NAME, manifest)
    return CollectionBundle(
        output_dir=resolved_output_dir,
        manifest_path=manifest_path,
        autonomous_trials_path=autonomous_trials_path,
        correction_segments_path=correction_segments_path,
        canonical_source_check_path=canonical_source_check_path,
        manifest=manifest,
        autonomous_trials=normalized_rows,
    )


def load_collection_bundle(collection_dir: str | Path) -> CollectionBundle:
    resolved_dir = _resolve_path(collection_dir)
    manifest = read_json(resolved_dir / COLLECTION_MANIFEST_NAME)
    autonomous_trials = tuple(read_jsonl(resolved_dir / AUTONOMOUS_TRIALS_NAME))
    return CollectionBundle(
        output_dir=resolved_dir,
        manifest_path=resolved_dir / COLLECTION_MANIFEST_NAME,
        autonomous_trials_path=resolved_dir / AUTONOMOUS_TRIALS_NAME,
        correction_segments_path=resolved_dir / CORRECTION_SEGMENTS_NAME,
        canonical_source_check_path=resolved_dir / CANONICAL_SOURCE_CHECK_NAME,
        manifest=manifest,
        autonomous_trials=autonomous_trials,
    )


def normalize_correction_segment(
    record: Mapping[str, object],
    *,
    default_policy_checkpoint_ref: str,
    default_critic_checkpoint_ref: str,
) -> dict[str, object]:
    correction_id = str(record.get("correction_id", "")).strip()
    if not correction_id:
        raise ValueError("correction segment requires correction_id")
    task_id_raw = record.get("task_id")
    if task_id_raw is None:
        raise ValueError(f"correction_id={correction_id} missing task_id")
    source_trial_id = str(record.get("source_trial_id", "")).strip() or None
    start_step = record.get("start_step")
    end_step = record.get("end_step")
    return {
        "schema_version": CORRECTION_SEGMENT_SCHEMA_VERSION,
        "correction_id": correction_id,
        "source_trial_id": source_trial_id,
        "task_suite_name": str(record.get("task_suite_name", "libero_spatial")),
        "task_id": int(cast(Any, task_id_raw)),
        "prompt_text": str(record.get("prompt_text", "")).strip(),
        "start_step": int(cast(Any, start_step)) if start_step is not None else None,
        "end_step": int(cast(Any, end_step)) if end_step is not None else None,
        "indicator_I": 1,
        "is_correction": True,
        "forced_positive_indicator": True,
        "human_correction_override_applied": True,
        "label": "forced_positive_correction",
        "policy_checkpoint_ref": str(
            record.get("policy_checkpoint_ref") or default_policy_checkpoint_ref
        ),
        "critic_checkpoint_ref": str(
            record.get("critic_checkpoint_ref") or default_critic_checkpoint_ref
        ),
    }


def load_correction_segments(
    correction_dir: str | Path | None,
    *,
    default_policy_checkpoint_ref: str,
    default_critic_checkpoint_ref: str,
) -> tuple[dict[str, object], ...]:
    if correction_dir is None:
        return tuple()
    resolved_dir = _resolve_path(correction_dir)
    correction_path = resolved_dir / CORRECTION_SEGMENTS_NAME
    if not correction_path.is_file():
        raise FileNotFoundError(correction_path)
    return tuple(
        normalize_correction_segment(
            record,
            default_policy_checkpoint_ref=default_policy_checkpoint_ref,
            default_critic_checkpoint_ref=default_critic_checkpoint_ref,
        )
        for record in read_jsonl(correction_path)
    )


def _copy_tasks_file(demo_dir: Path, output_dir: Path) -> Path:
    tasks = read_jsonl(demo_dir / "meta" / "tasks.jsonl")
    return write_jsonl(output_dir / "meta" / "tasks.jsonl", tasks)


def _copy_optional_meta_file(
    demo_dir: Path, output_dir: Path, name: str
) -> Path | None:
    source_path = demo_dir / "meta" / name
    if not source_path.is_file():
        return None
    destination_path = output_dir / "meta" / name
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    _ = shutil.copy2(source_path, destination_path)
    return destination_path


def _as_int(value: object, *, context: str) -> int:
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


def _episode_prompt_text(row: Mapping[str, object], *, context: str) -> str:
    tasks = row.get("tasks")
    if (
        not isinstance(tasks, Sequence)
        or isinstance(tasks, (str, bytes))
        or len(tasks) != 1
    ):
        raise ValueError(f"{context}.tasks must be a single-item list, got {tasks!r}")
    prompt = str(tasks[0]).strip()
    if not prompt:
        raise ValueError(f"{context}.tasks[0] must be non-empty")
    return prompt


def _task_prompt_map(task_rows: Sequence[Mapping[str, object]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for index, row in enumerate(task_rows):
        task_index = _as_int(
            row.get("task_index"), context=f"tasks[{index}].task_index"
        )
        task_text = str(row.get("task", "")).strip()
        if not task_text:
            raise ValueError(f"tasks[{index}].task must be non-empty")
        mapping[int(task_index)] = task_text
    return mapping


def _data_path_template(info: Mapping[str, object]) -> str:
    template = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    ).strip()
    if not template:
        raise ValueError("source dataset info.json is missing data_path")
    return template


def _chunks_size(info: Mapping[str, object]) -> int:
    return _as_int(info.get("chunks_size", 1000), context="info.chunks_size")


def _resolve_episode_parquet_path(
    dataset_dir: Path,
    info: Mapping[str, object],
    episode_index: int,
) -> Path:
    chunks_size = _chunks_size(info)
    rel_path = _data_path_template(info).format(
        episode_chunk=int(episode_index) // int(chunks_size),
        episode_index=int(episode_index),
    )
    return (dataset_dir / rel_path).resolve()


def _materialize_canonical_data_surface(
    *,
    demo_dir: Path,
    demo_info: Mapping[str, object],
    demo_episodes: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    data_root = output_dir / "data"
    if data_root.exists() or data_root.is_symlink():
        _replace_path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    for row in demo_episodes:
        episode_index = _as_int(
            row.get("episode_index"), context="demo_episodes[].episode_index"
        )
        source_path = _resolve_episode_parquet_path(demo_dir, demo_info, episode_index)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        relative_path = source_path.relative_to(demo_dir)
        destination_path = output_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists() or destination_path.is_symlink():
            _replace_path(destination_path)
        try:
            destination_path.symlink_to(source_path)
        except OSError:
            _ = shutil.copy2(source_path, destination_path)


def _demo_episode_surface_rows(
    demo_episodes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in demo_episodes:
        episode_index = _as_int(
            row.get("episode_index"), context="demo_episodes[].episode_index"
        )
        rows.append(
            {
                **dict(row),
                "episode_index": int(episode_index),
                "source_kind": "canonical_demo",
                "merge_entry_id": f"demo:{episode_index:06d}",
                "indicator_I": None,
                "success": True,
                "is_correction": False,
                "forced_positive_indicator": False,
                "policy_checkpoint_ref": None,
                "critic_checkpoint_ref": None,
            }
        )
    return rows


def _template_episode_index_by_prompt(
    demo_episodes: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, row in enumerate(demo_episodes):
        prompt = _episode_prompt_text(row, context=f"demo_episodes[{index}]")
        mapping.setdefault(
            prompt,
            _as_int(
                row.get("episode_index"),
                context=f"demo_episodes[{index}].episode_index",
            ),
        )
    return mapping


def _load_source_template_row(
    *,
    demo_dir: Path,
    demo_info: Mapping[str, object],
    episode_index: int,
) -> dict[str, object]:
    pd = _import_pandas()
    parquet_path = _resolve_episode_parquet_path(
        demo_dir, demo_info, int(episode_index)
    )
    frame = pd.read_parquet(parquet_path)
    missing = [
        column for column in REQUIRED_SOURCE_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"template episode {episode_index} missing required source columns: {missing}"
        )
    if len(frame) <= 0:
        raise ValueError(f"template episode {episode_index} parquet is empty")
    first_row = frame.iloc[0].to_dict()
    return {str(key): value for key, value in first_row.items()}


def _resolve_prompt_for_task(
    *,
    task_id: int,
    task_prompts: Mapping[int, str],
    fallback_prompt: str,
) -> str:
    prompt = str(task_prompts.get(int(task_id), fallback_prompt)).strip()
    if not prompt:
        raise ValueError(f"unable to resolve prompt text for task_id={task_id}")
    return prompt


def _write_extra_episode_parquet(
    *,
    output_dir: Path,
    data_path_template: str,
    chunks_size: int,
    episode_index: int,
    task_index: int,
    global_index: int,
    template_row: Mapping[str, object],
) -> None:
    pd = _import_pandas()
    payload: dict[str, list[object]] = {}
    for column in REQUIRED_SOURCE_COLUMNS:
        payload[column] = [template_row.get(column)]
    payload["timestamp"] = [0.0]
    payload["frame_index"] = [0]
    payload["episode_index"] = [int(episode_index)]
    payload["index"] = [int(global_index)]
    payload["task_index"] = [int(task_index)]
    frame = pd.DataFrame(payload)
    output_path = output_dir / data_path_template.format(
        episode_chunk=int(episode_index) // int(chunks_size),
        episode_index=int(episode_index),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, engine="pyarrow", index=False)


def _replace_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _link_optional_data_dir(demo_dir: Path, output_dir: Path) -> Path | None:
    source_dir = demo_dir / "data"
    if not source_dir.is_dir():
        return None
    destination_dir = output_dir / "data"
    if destination_dir.exists() or destination_dir.is_symlink():
        _replace_path(destination_dir)
    try:
        destination_dir.symlink_to(source_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(source_dir, destination_dir)
    return destination_dir


def _is_recap_ready_dataset_dir(dataset_dir: Path) -> bool:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        return False
    info = read_json(info_path)
    contract = info.get("recap_advantage_input_contract")
    features = info.get("features")
    if not isinstance(contract, Mapping) or not contract:
        return False
    if not isinstance(features, Mapping):
        return False
    required_core_features = {
        "observation.images.ego_view",
        "observation.state",
        "action",
        "recap_m2.indicator_I",
    }
    feature_keys = {str(key) for key in features.keys()}
    if not required_core_features.issubset(feature_keys):
        return False
    if {
        "recap_m2.prompt_raw",
        "recap_m2.prompt_conditioned",
    }.issubset(feature_keys):
        return True
    parquet_files = tuple(sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet")))
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
    return {
        "recap_m2.prompt_raw",
        "recap_m2.prompt_conditioned",
    }.issubset(sample_columns)


def _resolve_recap_relabel_source_dir(demo_dir: Path) -> Path | None:
    candidate = demo_dir.parent / f"{demo_dir.name}_recap_relabels_v1"
    if candidate.is_dir() and _is_recap_ready_dataset_dir(candidate):
        return candidate.resolve()
    return None


def _parse_chunk_index(chunk_dir: Path) -> int:
    suffix = chunk_dir.name.split("-")[-1]
    return int(suffix)


def _symlink_or_copy_dir(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(source_dir.iterdir()):
        destination_child = destination_dir / child.name
        if child.is_dir():
            _symlink_or_copy_dir(child, destination_child)
            continue
        try:
            destination_child.symlink_to(child)
        except OSError:
            _ = shutil.copy2(child, destination_child)


def _load_recap_template_row(
    *,
    recap_source_dir: Path,
    recap_info: Mapping[str, object],
    episode_index: int,
) -> dict[str, object]:
    pd = _import_pandas()
    parquet_path = _resolve_episode_parquet_path(
        recap_source_dir, recap_info, int(episode_index)
    )
    frame = pd.read_parquet(parquet_path)
    if len(frame) <= 0:
        raise ValueError(f"recap template episode {episode_index} parquet is empty")
    first_row = frame.iloc[0].to_dict()
    return {str(key): value for key, value in first_row.items()}


def _complete_recap_advantage_input_contract(
    *,
    recap_source_dir: Path,
    recap_info: Mapping[str, object],
) -> dict[str, object]:
    source_contract_raw = recap_info.get("recap_advantage_input_contract")
    source_contract = (
        {str(key): value for key, value in source_contract_raw.items()}
        if isinstance(source_contract_raw, Mapping)
        else {}
    )
    report_path = recap_source_dir / "materialization_report.json"
    report_payload = read_json(report_path) if report_path.is_file() else {}
    report_contract_raw = report_payload.get("advantage_contract")
    report_contract = (
        {str(key): value for key, value in report_contract_raw.items()}
        if isinstance(report_contract_raw, Mapping)
        else {}
    )

    contract = dict(report_contract)
    contract.update(source_contract)

    if "epsilon_source" not in contract:
        epsilon_source = report_payload.get("epsilon_source")
        if epsilon_source is None or str(epsilon_source).strip() == "":
            epsilon_group_field = str(
                contract.get("epsilon_group_field")
                or contract.get("task_text_field")
                or "prompt_raw"
            ).strip()
            epsilon_quantile_raw = (
                report_payload.get("epsilon_quantile")
                if report_payload.get("epsilon_quantile") is not None
                else contract.get("epsilon_quantile", DEFAULT_EPSILON_QUANTILE)
            )
            try:
                epsilon_quantile = float(cast(Any, epsilon_quantile_raw))
            except (TypeError, ValueError):
                epsilon_quantile = float(DEFAULT_EPSILON_QUANTILE)
            epsilon_source = build_epsilon_source(
                group_field=epsilon_group_field,
                quantile=epsilon_quantile,
            )
        contract["epsilon_source"] = str(epsilon_source)

    if "indicator_dropout_p" not in contract:
        contract["indicator_dropout_p"] = 0.3
    if "human_correction_override" not in contract:
        contract["human_correction_override"] = True
    if (
        report_payload.get("epsilon_quantile") is not None
        and "epsilon_quantile" not in contract
    ):
        contract["epsilon_quantile"] = report_payload.get("epsilon_quantile")
    if report_payload.get("epsilon_l") is not None and "epsilon_l" not in contract:
        contract["epsilon_l"] = report_payload.get("epsilon_l")
    return contract


def _write_recap_extra_episode_parquet(
    *,
    output_dir: Path,
    data_path_template: str,
    chunks_size: int,
    episode_index: int,
    task_index: int,
    global_index: int,
    prompt_raw: str,
    indicator: int,
    critic_checkpoint_ref: str,
    template_row: Mapping[str, object],
) -> None:
    pd = _import_pandas()
    prompt_bundle = build_training_prompt_bundle(
        {"prompt_raw": prompt_raw, "recap_m2.indicator_I": int(indicator)},
        consumer_mode=RECAP_RELABEL_CONSUMER_MODE,
        fixed_indicator_mode=None,
        critic_checkpoint_ref=critic_checkpoint_ref,
    )
    signed_value = 1.0 if int(indicator) == 1 else -1.0
    payload = {str(column): [value] for column, value in template_row.items()}
    payload["timestamp"] = [0.0]
    payload["frame_index"] = [0]
    payload["episode_index"] = [int(episode_index)]
    payload["index"] = [int(global_index)]
    payload["task_index"] = [int(task_index)]
    if "recap_m2.indicator_I" in payload:
        payload["recap_m2.indicator_I"] = [int(indicator)]
    if "recap_m2.prompt_raw" in payload:
        payload["recap_m2.prompt_raw"] = [prompt_raw]
    if "recap_m2.prompt_conditioned" in payload:
        payload["recap_m2.prompt_conditioned"] = [prompt_bundle.prompt_text]
    if "recap_m2.advantage_input" in payload:
        payload["recap_m2.advantage_input"] = [float(signed_value)]
    if "recap_m2.advantage_A" in payload:
        payload["recap_m2.advantage_A"] = [float(signed_value)]
    output_path = output_dir / data_path_template.format(
        episode_chunk=int(episode_index) // int(chunks_size),
        episode_index=int(episode_index),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(payload).to_parquet(output_path, engine="pyarrow", index=False)


def _materialize_prebuilt_merged_recap_surface(
    *,
    demo_dir: Path,
    output_dir: Path,
    task_rows: Sequence[Mapping[str, object]],
    collection_bundle: CollectionBundle,
    correction_segments: Sequence[Mapping[str, object]],
    dataset_mix: Mapping[str, object],
    episode_lineage_path: Path,
) -> Path | None:
    recap_source_dir = _resolve_recap_relabel_source_dir(demo_dir)
    if recap_source_dir is None:
        return None
    surface_dir = output_dir / MERGED_RECAP_READY_DATASET_DIRNAME
    if surface_dir.exists() or surface_dir.is_symlink():
        _replace_path(surface_dir)
    surface_dir.mkdir(parents=True, exist_ok=True)

    recap_info, _, recap_episodes = read_demo_metadata(recap_source_dir)
    recap_contract = _complete_recap_advantage_input_contract(
        recap_source_dir=recap_source_dir,
        recap_info=recap_info,
    )
    source_meta_dir = recap_source_dir / "meta"
    target_meta_dir = surface_dir / "meta"
    target_meta_dir.mkdir(parents=True, exist_ok=True)
    for name in ("stats.json", "modality.json", "tasks.jsonl"):
        source_path = source_meta_dir / name
        if source_path.is_file():
            _ = shutil.copy2(source_path, target_meta_dir / name)

    source_data_dir = recap_source_dir / "data"
    target_data_dir = surface_dir / "data"
    target_data_dir.mkdir(parents=True, exist_ok=True)
    recap_total_frames = _as_int(
        recap_info.get("total_frames", 0), context="recap_info.total_frames"
    )
    data_path_template = _data_path_template(recap_info)
    chunks_size = _chunks_size(recap_info)
    total_extra_rows = len(collection_bundle.autonomous_trials) + len(
        correction_segments
    )
    base_episode_count = len(recap_episodes)
    touched_chunks = {
        int((base_episode_count + offset) // int(chunks_size))
        for offset in range(total_extra_rows)
    }
    for chunk_dir in sorted(source_data_dir.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        chunk_index = _parse_chunk_index(chunk_dir)
        destination_dir = target_data_dir / chunk_dir.name
        if chunk_index in touched_chunks:
            shutil.copytree(chunk_dir, destination_dir)
        else:
            _symlink_or_copy_dir(chunk_dir, destination_dir)

    critic_checkpoint_ref = str(
        collection_bundle.manifest.get("critic_checkpoint_ref", "not_applicable")
    )
    task_prompts = _task_prompt_map(task_rows)
    template_by_prompt = _template_episode_index_by_prompt(recap_episodes)
    fallback_prompt = _episode_prompt_text(
        recap_episodes[0], context="recap_episodes[0]"
    )
    merged_episode_rows = [dict(row) for row in recap_episodes]
    next_episode_index = len(recap_episodes)
    next_global_index = recap_total_frames

    for row in collection_bundle.autonomous_trials:
        task_id = _as_int(row.get("task_id"), context="autonomous_trials[].task_id")
        prompt_raw = _resolve_prompt_for_task(
            task_id=task_id,
            task_prompts=task_prompts,
            fallback_prompt=fallback_prompt,
        )
        template_episode_index = int(
            template_by_prompt.get(prompt_raw, next(iter(template_by_prompt.values())))
        )
        template_row = _load_recap_template_row(
            recap_source_dir=recap_source_dir,
            recap_info=recap_info,
            episode_index=template_episode_index,
        )
        _write_recap_extra_episode_parquet(
            output_dir=surface_dir,
            data_path_template=data_path_template,
            chunks_size=chunks_size,
            episode_index=next_episode_index,
            task_index=task_id,
            global_index=next_global_index,
            prompt_raw=prompt_raw,
            indicator=_as_int(
                row.get("indicator_I", 0), context="autonomous_trials[].indicator_I"
            ),
            critic_checkpoint_ref=critic_checkpoint_ref,
            template_row=template_row,
        )
        merged_episode_rows.append(
            {
                "episode_index": int(next_episode_index),
                "tasks": [prompt_raw],
                "length": 1,
                "source_kind": "autonomous_trial",
                "merge_entry_id": str(
                    row.get("trial_id", f"autonomous:{next_episode_index:06d}")
                ),
                "task_id": int(task_id),
                "seed": _as_int(row.get("seed"), context="autonomous_trials[].seed"),
                "trial_idx": _as_int(
                    row.get("trial_idx"), context="autonomous_trials[].trial_idx"
                ),
                "success": bool(cast(Any, row.get("success", False))),
                "indicator_I": _as_int(
                    row.get("indicator_I", 0), context="autonomous_trials[].indicator_I"
                ),
                "is_correction": False,
                "forced_positive_indicator": False,
                "policy_checkpoint_ref": str(row.get("policy_checkpoint_ref", "")),
                "critic_checkpoint_ref": str(row.get("critic_checkpoint_ref", "")),
            }
        )
        next_episode_index += 1
        next_global_index += 1

    for row in correction_segments:
        task_id = _as_int(row.get("task_id"), context="correction_segments[].task_id")
        prompt_raw = _resolve_prompt_for_task(
            task_id=task_id,
            task_prompts=task_prompts,
            fallback_prompt=fallback_prompt,
        )
        template_episode_index = int(
            template_by_prompt.get(prompt_raw, next(iter(template_by_prompt.values())))
        )
        template_row = _load_recap_template_row(
            recap_source_dir=recap_source_dir,
            recap_info=recap_info,
            episode_index=template_episode_index,
        )
        _write_recap_extra_episode_parquet(
            output_dir=surface_dir,
            data_path_template=data_path_template,
            chunks_size=chunks_size,
            episode_index=next_episode_index,
            task_index=task_id,
            global_index=next_global_index,
            prompt_raw=prompt_raw,
            indicator=1,
            critic_checkpoint_ref=str(
                row.get("critic_checkpoint_ref", critic_checkpoint_ref)
            ),
            template_row=template_row,
        )
        merged_episode_rows.append(
            {
                "episode_index": int(next_episode_index),
                "tasks": [prompt_raw],
                "length": 1,
                "source_kind": "correction_segment",
                "merge_entry_id": str(
                    row.get("correction_id", f"correction:{next_episode_index:06d}")
                ),
                "task_id": int(task_id),
                "source_trial_id": row.get("source_trial_id"),
                "success": True,
                "indicator_I": 1,
                "is_correction": True,
                "forced_positive_indicator": True,
                "human_correction_override_applied": True,
                "policy_checkpoint_ref": str(row.get("policy_checkpoint_ref", "")),
                "critic_checkpoint_ref": str(row.get("critic_checkpoint_ref", "")),
            }
        )
        next_episode_index += 1
        next_global_index += 1

    episodes_path = write_jsonl(target_meta_dir / "episodes.jsonl", merged_episode_rows)
    target_info = {
        **dict(recap_info),
        "recap_advantage_input_contract": recap_contract,
        "source_dataset_dir": str(output_dir),
        "source_dataset_name": output_dir.name,
        "merged_dataset_route_id": MERGED_DATASET_ROUTE_ID,
        "merged_dataset_ref": str(output_dir),
        MERGED_RECAP_READY_DATASET_REF_KEY: str(surface_dir),
        "episodes_added": int(len(collection_bundle.autonomous_trials)),
        "corrections_added": int(len(correction_segments)),
        "dataset_mix": dict(dataset_mix),
        "total_episodes": int(len(merged_episode_rows)),
        "total_frames": int(
            recap_total_frames
            + len(collection_bundle.autonomous_trials)
            + len(correction_segments)
        ),
        "total_chunks": int(1 + ((len(merged_episode_rows) - 1) // int(chunks_size))),
        "splits": {"train": f"0:{len(merged_episode_rows)}"},
        "episodes_jsonl": str(episodes_path),
        "episode_lineage_jsonl": str(episode_lineage_path),
    }
    write_json(target_meta_dir / "info.json", target_info)

    report_path = recap_source_dir / "materialization_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        report["source_dataset_dir"] = str(output_dir)
        report["output_dataset_dir"] = str(surface_dir)
        report["merged_dataset_route_id"] = MERGED_DATASET_ROUTE_ID
        report["merged_dataset_ref"] = str(output_dir)
        report["selected_episode_count"] = int(len(merged_episode_rows))
        report["selected_frame_count"] = _as_int(
            target_info.get("total_frames", 0), context="target_info.total_frames"
        )
        report["merged_episode_override_applied"] = True
        report["merged_episode_override_count"] = int(total_extra_rows)
        report["merged_correction_override_count"] = int(len(correction_segments))
        write_json(surface_dir / "materialization_report.json", report)

    for stale_meta in (
        target_meta_dir / "dataset_fingerprint.json",
        target_meta_dir / "episode_universe_hash.txt",
    ):
        if stale_meta.is_file():
            stale_meta.unlink()
    return surface_dir


def _demo_episode_lineage_rows(
    demo_episodes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in demo_episodes:
        episode_index = int(cast(Any, row.get("episode_index")))
        rows.append(
            {
                "schema_version": MERGED_EPISODE_SCHEMA_VERSION,
                "merge_entry_id": f"demo:{episode_index:06d}",
                "source_kind": "canonical_demo",
                "episode_index": episode_index,
                "tasks": list(cast(Sequence[object], row.get("tasks", []))),
                "length": int(cast(Any, row.get("length", 0))),
                "is_correction": False,
                "forced_positive_indicator": False,
                "indicator_I": None,
                "policy_checkpoint_ref": None,
                "critic_checkpoint_ref": None,
            }
        )
    return rows


def _autonomous_episode_lineage_rows(
    autonomous_trials: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in autonomous_trials:
        rows.append(
            {
                "schema_version": MERGED_EPISODE_SCHEMA_VERSION,
                "merge_entry_id": str(row["trial_id"]),
                "source_kind": "autonomous_trial",
                "task_suite_name": str(row.get("task_suite_name", "libero_spatial")),
                "task_id": int(cast(Any, row.get("task_id"))),
                "seed": int(cast(Any, row.get("seed"))),
                "trial_idx": int(cast(Any, row.get("trial_idx"))),
                "label": str(row.get("label", "failure")),
                "success": bool(cast(Any, row.get("success"))),
                "indicator_I": int(cast(Any, row.get("indicator_I", 0))),
                "is_correction": False,
                "forced_positive_indicator": False,
                "policy_checkpoint_ref": str(row.get("policy_checkpoint_ref", "")),
                "critic_checkpoint_ref": str(row.get("critic_checkpoint_ref", "")),
            }
        )
    return rows


def _correction_lineage_rows(
    correction_segments: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in correction_segments:
        rows.append(
            {
                "schema_version": MERGED_EPISODE_SCHEMA_VERSION,
                "merge_entry_id": str(row["correction_id"]),
                "source_kind": "correction_segment",
                "task_suite_name": str(row.get("task_suite_name", "libero_spatial")),
                "task_id": int(cast(Any, row.get("task_id"))),
                "source_trial_id": row.get("source_trial_id"),
                "label": "forced_positive_correction",
                "indicator_I": 1,
                "is_correction": True,
                "forced_positive_indicator": True,
                "human_correction_override_applied": True,
                "policy_checkpoint_ref": str(row.get("policy_checkpoint_ref", "")),
                "critic_checkpoint_ref": str(row.get("critic_checkpoint_ref", "")),
            }
        )
    return rows


def build_dataset_mix(
    *,
    demo_episode_count: int,
    autonomous_trials: Sequence[Mapping[str, object]],
    correction_segments: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    success_count = sum(
        1 for row in autonomous_trials if bool(cast(Any, row.get("success", False)))
    )
    autonomous_count = len(autonomous_trials)
    correction_count = len(correction_segments)
    return {
        "canonical_demo": {
            "episodes": int(demo_episode_count),
            "source": "official_native_8d",
        },
        "autonomous": {
            "episodes": int(autonomous_count),
            "successes": int(success_count),
            "failures": int(autonomous_count - success_count),
        },
        "correction": {
            "segments": int(correction_count),
            "forced_positive": True,
        },
    }


def build_merge_manifest(
    *,
    output_dir: Path,
    demo_dir: Path,
    canonical_source_report: Mapping[str, object],
    demo_info: Mapping[str, object],
    collection_bundle: CollectionBundle,
    correction_dir: Path | None,
    correction_segments: Sequence[Mapping[str, object]],
    dataset_mix: Mapping[str, object],
    trainer_compatible_recap_dataset_ref: Path | None,
) -> dict[str, object]:
    return {
        "schema_version": MERGED_DATASET_SCHEMA_VERSION,
        "route_id": MERGED_DATASET_ROUTE_ID,
        "created_at": _now_iso(),
        "output_dir": str(output_dir),
        "canonical_demo_source_dir": str(demo_dir),
        "canonical_demo_source": dict(canonical_source_report),
        "canonical_demo_source_schema_version": demo_info.get("schema_version"),
        "source_dataset_name": demo_dir.name,
        "collection_manifest_ref": str(collection_bundle.manifest_path),
        "autonomous_trials_ref": str(collection_bundle.autonomous_trials_path),
        "correction_segments_ref": (
            str(_resolve_path(correction_dir) / CORRECTION_SEGMENTS_NAME)
            if correction_dir is not None
            else None
        ),
        "dataset_mix": dict(dataset_mix),
        "episodes_added": int(len(collection_bundle.autonomous_trials)),
        "corrections_added": int(len(correction_segments)),
        "trainer_visible_total_episodes": int(
            _as_int(
                demo_info.get("total_episodes", 0), context="demo_info.total_episodes"
            )
            + len(collection_bundle.autonomous_trials)
            + len(correction_segments)
        ),
        MERGED_RECAP_READY_DATASET_REF_KEY: (
            str(trainer_compatible_recap_dataset_ref)
            if trainer_compatible_recap_dataset_ref is not None
            else None
        ),
        "policy_checkpoint_ref": collection_bundle.manifest.get(
            "policy_checkpoint_ref"
        ),
        "critic_checkpoint_ref": collection_bundle.manifest.get(
            "critic_checkpoint_ref"
        ),
        "policy_stage": collection_bundle.manifest.get("policy_stage"),
        "source_dataset_dir": collection_bundle.manifest.get("source_dataset_dir"),
        "prepared_dataset_dir": collection_bundle.manifest.get("prepared_dataset_dir"),
    }


def materialize_merged_dataset(
    *,
    demo_dir: str | Path,
    canonical_source_report: Mapping[str, object],
    collection_bundle: CollectionBundle,
    output_dir: str | Path,
    correction_dir: str | Path | None = None,
) -> MergedDatasetBundle:
    resolved_demo_dir = _resolve_path(demo_dir)
    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in (
        resolved_output_dir / "meta",
        resolved_output_dir / "data",
        resolved_output_dir / MERGED_RECAP_READY_DATASET_DIRNAME,
        resolved_output_dir / AUTONOMOUS_TRIALS_NAME,
        resolved_output_dir / CORRECTION_SEGMENTS_NAME,
        resolved_output_dir / CANONICAL_SOURCE_CHECK_NAME,
        resolved_output_dir / MERGE_MANIFEST_NAME,
    ):
        if stale_path.exists() or stale_path.is_symlink():
            _replace_path(stale_path)
    demo_info, task_rows, demo_episodes = read_demo_metadata(resolved_demo_dir)
    tasks_path = _copy_tasks_file(resolved_demo_dir, resolved_output_dir)
    _ = _copy_optional_meta_file(resolved_demo_dir, resolved_output_dir, "stats.json")
    _ = _copy_optional_meta_file(
        resolved_demo_dir, resolved_output_dir, "modality.json"
    )
    _materialize_canonical_data_surface(
        demo_dir=resolved_demo_dir,
        demo_info=demo_info,
        demo_episodes=demo_episodes,
        output_dir=resolved_output_dir,
    )
    correction_segments = load_correction_segments(
        correction_dir,
        default_policy_checkpoint_ref=str(
            collection_bundle.manifest.get("policy_checkpoint_ref", "")
        ),
        default_critic_checkpoint_ref=str(
            collection_bundle.manifest.get("critic_checkpoint_ref", "")
        ),
    )
    dataset_mix = build_dataset_mix(
        demo_episode_count=len(demo_episodes),
        autonomous_trials=collection_bundle.autonomous_trials,
        correction_segments=correction_segments,
    )
    merged_lineage_rows = (
        _demo_episode_lineage_rows(demo_episodes)
        + _autonomous_episode_lineage_rows(collection_bundle.autonomous_trials)
        + _correction_lineage_rows(correction_segments)
    )
    task_prompts = _task_prompt_map(task_rows)
    template_by_prompt = _template_episode_index_by_prompt(demo_episodes)
    fallback_prompt = _episode_prompt_text(demo_episodes[0], context="demo_episodes[0]")
    data_path_template = _data_path_template(demo_info)
    chunks_size = _chunks_size(demo_info)
    demo_total_frames = _as_int(
        demo_info.get("total_frames", 0), context="demo_info.total_frames"
    )
    merged_episode_rows = _demo_episode_surface_rows(demo_episodes)
    next_episode_index = len(demo_episodes)
    next_global_index = demo_total_frames
    for row in collection_bundle.autonomous_trials:
        task_id = _as_int(row.get("task_id"), context="autonomous_trials[].task_id")
        prompt_raw = _resolve_prompt_for_task(
            task_id=task_id,
            task_prompts=task_prompts,
            fallback_prompt=fallback_prompt,
        )
        template_episode_index = int(
            template_by_prompt.get(prompt_raw, next(iter(template_by_prompt.values())))
        )
        template_row = _load_source_template_row(
            demo_dir=resolved_demo_dir,
            demo_info=demo_info,
            episode_index=template_episode_index,
        )
        task_index = _as_int(
            task_id if task_id in task_prompts else template_row.get("task_index", 0),
            context="autonomous_trials[].task_index",
        )
        _write_extra_episode_parquet(
            output_dir=resolved_output_dir,
            data_path_template=data_path_template,
            chunks_size=chunks_size,
            episode_index=next_episode_index,
            task_index=task_index,
            global_index=next_global_index,
            template_row=template_row,
        )
        merged_episode_rows.append(
            {
                "episode_index": int(next_episode_index),
                "tasks": [prompt_raw],
                "length": 1,
                "source_kind": "autonomous_trial",
                "merge_entry_id": str(
                    row.get("trial_id", f"autonomous:{next_episode_index:06d}")
                ),
                "task_id": int(task_id),
                "seed": _as_int(row.get("seed"), context="autonomous_trials[].seed"),
                "trial_idx": _as_int(
                    row.get("trial_idx"), context="autonomous_trials[].trial_idx"
                ),
                "success": bool(cast(Any, row.get("success", False))),
                "indicator_I": _as_int(
                    row.get("indicator_I", 0), context="autonomous_trials[].indicator_I"
                ),
                "is_correction": False,
                "forced_positive_indicator": False,
                "policy_checkpoint_ref": str(row.get("policy_checkpoint_ref", "")),
                "critic_checkpoint_ref": str(row.get("critic_checkpoint_ref", "")),
            }
        )
        next_episode_index += 1
        next_global_index += 1
    for row in correction_segments:
        task_id = _as_int(row.get("task_id"), context="correction_segments[].task_id")
        prompt_raw = _resolve_prompt_for_task(
            task_id=task_id,
            task_prompts=task_prompts,
            fallback_prompt=fallback_prompt,
        )
        template_episode_index = int(
            template_by_prompt.get(prompt_raw, next(iter(template_by_prompt.values())))
        )
        template_row = _load_source_template_row(
            demo_dir=resolved_demo_dir,
            demo_info=demo_info,
            episode_index=template_episode_index,
        )
        task_index = _as_int(
            task_id if task_id in task_prompts else template_row.get("task_index", 0),
            context="correction_segments[].task_index",
        )
        _write_extra_episode_parquet(
            output_dir=resolved_output_dir,
            data_path_template=data_path_template,
            chunks_size=chunks_size,
            episode_index=next_episode_index,
            task_index=task_index,
            global_index=next_global_index,
            template_row=template_row,
        )
        merged_episode_rows.append(
            {
                "episode_index": int(next_episode_index),
                "tasks": [prompt_raw],
                "length": 1,
                "source_kind": "correction_segment",
                "merge_entry_id": str(
                    row.get("correction_id", f"correction:{next_episode_index:06d}")
                ),
                "task_id": int(task_id),
                "source_trial_id": row.get("source_trial_id"),
                "success": True,
                "indicator_I": 1,
                "is_correction": True,
                "forced_positive_indicator": True,
                "human_correction_override_applied": True,
                "policy_checkpoint_ref": str(row.get("policy_checkpoint_ref", "")),
                "critic_checkpoint_ref": str(row.get("critic_checkpoint_ref", "")),
            }
        )
        next_episode_index += 1
        next_global_index += 1
    episode_lineage_path = write_jsonl(
        resolved_output_dir / "meta" / MERGED_EPISODE_LINEAGE_NAME,
        merged_lineage_rows,
    )
    trainer_compatible_recap_dataset_ref = _materialize_prebuilt_merged_recap_surface(
        demo_dir=resolved_demo_dir,
        output_dir=resolved_output_dir,
        task_rows=task_rows,
        collection_bundle=collection_bundle,
        correction_segments=correction_segments,
        dataset_mix=dataset_mix,
        episode_lineage_path=episode_lineage_path,
    )
    episodes_path = write_jsonl(
        resolved_output_dir / "meta" / "episodes.jsonl",
        merged_episode_rows,
    )
    _ = write_jsonl(
        resolved_output_dir / AUTONOMOUS_TRIALS_NAME,
        collection_bundle.autonomous_trials,
    )
    _ = write_jsonl(
        resolved_output_dir / CORRECTION_SEGMENTS_NAME,
        correction_segments,
    )
    canonical_source_check_path = write_json(
        resolved_output_dir / CANONICAL_SOURCE_CHECK_NAME,
        dict(canonical_source_report),
    )
    merge_manifest = build_merge_manifest(
        output_dir=resolved_output_dir,
        demo_dir=resolved_demo_dir,
        canonical_source_report=canonical_source_report,
        demo_info=demo_info,
        collection_bundle=collection_bundle,
        correction_dir=_resolve_path(correction_dir)
        if correction_dir is not None
        else None,
        correction_segments=correction_segments,
        dataset_mix=dataset_mix,
        trainer_compatible_recap_dataset_ref=trainer_compatible_recap_dataset_ref,
    )
    info_payload = {
        **dict(demo_info),
        "schema_version": MERGED_DATASET_SCHEMA_VERSION,
        "route_id": MERGED_DATASET_ROUTE_ID,
        "merged_dataset_schema_version": MERGED_DATASET_SCHEMA_VERSION,
        "merged_dataset_route_id": MERGED_DATASET_ROUTE_ID,
        "source_dataset_name": resolved_demo_dir.name,
        "canonical_demo_source_dir": str(resolved_demo_dir),
        "dataset_mix": dict(dataset_mix),
        "episodes_added": int(len(collection_bundle.autonomous_trials)),
        "corrections_added": int(len(correction_segments)),
        "total_episodes": int(len(merged_episode_rows)),
        "total_frames": int(
            demo_total_frames
            + len(collection_bundle.autonomous_trials)
            + len(correction_segments)
        ),
        "total_chunks": int(1 + ((len(merged_episode_rows) - 1) // int(chunks_size))),
        "splits": {"train": f"0:{len(merged_episode_rows)}"},
        "collection_manifest_ref": str(collection_bundle.manifest_path),
        "merge_manifest_ref": str(resolved_output_dir / MERGE_MANIFEST_NAME),
        "tasks_jsonl": str(tasks_path),
        "episodes_jsonl": str(episodes_path),
        "episode_lineage_jsonl": str(episode_lineage_path),
        "canonical_demo_source": dict(canonical_source_report),
        MERGED_RECAP_READY_DATASET_REF_KEY: (
            str(trainer_compatible_recap_dataset_ref)
            if trainer_compatible_recap_dataset_ref is not None
            else None
        ),
    }
    info_path = write_json(resolved_output_dir / "meta" / "info.json", info_payload)
    merge_manifest_path = write_json(
        resolved_output_dir / MERGE_MANIFEST_NAME,
        merge_manifest,
    )
    return MergedDatasetBundle(
        output_dir=resolved_output_dir,
        merge_manifest_path=merge_manifest_path,
        info_path=info_path,
        episodes_path=episodes_path,
        episode_lineage_path=episode_lineage_path,
        canonical_source_check_path=canonical_source_check_path,
        merge_manifest=merge_manifest,
    )


def build_iteration_manifest(
    *,
    iter_id: str,
    collection_bundle: CollectionBundle,
    merged_dataset: MergedDatasetBundle,
    critic_retrain: Mapping[str, object] | None = None,
    policy_retrain: Mapping[str, object] | None = None,
    iteration_eval: Mapping[str, object] | None = None,
) -> dict[str, object]:
    dataset_mix = cast(
        Mapping[str, object], merged_dataset.merge_manifest["dataset_mix"]
    )
    canonical_source = cast(
        Mapping[str, object], merged_dataset.merge_manifest["canonical_demo_source"]
    )
    merged_surface_info = read_json(merged_dataset.info_path)
    stage_lineage: dict[str, object] = {
        "collect_route_id": collection_bundle.manifest.get("route_id"),
        "merge_route_id": merged_dataset.merge_manifest.get("route_id"),
        "policy_stage": collection_bundle.manifest.get("policy_stage"),
    }
    if critic_retrain is not None:
        stage_lineage["critic_retrain_route_id"] = critic_retrain.get("route_id")
    if policy_retrain is not None:
        stage_lineage["policy_retrain_route_id"] = policy_retrain.get("route_id")
    if iteration_eval is not None:
        stage_lineage["iteration_eval_route_id"] = iteration_eval.get("route_id")

    manifest = {
        "schema_version": ITERATION_MANIFEST_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "iter_id": iter_id,
        "policy_checkpoint_ref": collection_bundle.manifest.get(
            "policy_checkpoint_ref"
        ),
        "critic_checkpoint_ref": collection_bundle.manifest.get(
            "critic_checkpoint_ref"
        ),
        "dataset_mix": dict(dataset_mix),
        "episodes_added": merged_dataset.merge_manifest.get("episodes_added"),
        "corrections_added": merged_dataset.merge_manifest.get("corrections_added"),
        "canonical_demo_source_root_proof": dict(canonical_source),
        "collection_manifest_ref": str(collection_bundle.manifest_path),
        "merged_dataset_ref": str(merged_dataset.output_dir),
        "merge_manifest_ref": str(merged_dataset.merge_manifest_path),
        "merged_dataset_surface": {
            "total_episodes": merged_surface_info.get("total_episodes"),
            "total_frames": merged_surface_info.get("total_frames"),
            "episodes_added": merged_surface_info.get("episodes_added"),
            "corrections_added": merged_surface_info.get("corrections_added"),
            "dataset_mix": merged_surface_info.get("dataset_mix"),
            "episodes_jsonl": str(merged_dataset.episodes_path),
            "episode_lineage_jsonl": str(merged_dataset.episode_lineage_path),
        },
        "policy_lineage": {
            "policy_stage": collection_bundle.manifest.get("policy_stage"),
            "source_dataset_dir": collection_bundle.manifest.get("source_dataset_dir"),
            "prepared_dataset_dir": collection_bundle.manifest.get(
                "prepared_dataset_dir"
            ),
            "train_manifest_ref": collection_bundle.manifest.get("train_manifest_ref"),
            "checkpoint_provenance_ref": collection_bundle.manifest.get(
                "checkpoint_provenance_ref"
            ),
        },
        "stage_lineage": stage_lineage,
    }
    if critic_retrain is not None:
        manifest["critic_retrain"] = dict(critic_retrain)
    if policy_retrain is not None:
        manifest["policy_retrain"] = dict(policy_retrain)
    if iteration_eval is not None:
        manifest["iteration_eval"] = dict(iteration_eval)
    return manifest


def build_loop_manifest(
    *,
    output_dir: str | Path,
    iteration_manifests: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": LOOP_MANIFEST_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "output_dir": str(_resolve_path(output_dir)),
        "iterations": [dict(manifest) for manifest in iteration_manifests],
    }


__all__ = [
    "AUTONOMOUS_TRIALS_NAME",
    "CANONICAL_SOURCE_CHECK_NAME",
    "COLLECTION_MANIFEST_NAME",
    "COLLECTION_ROUTE_ID",
    "COLLECTION_SCHEMA_VERSION",
    "COLLECTION_EPISODE_SCHEMA_VERSION",
    "CollectionBundle",
    "CORRECTION_SEGMENT_SCHEMA_VERSION",
    "CORRECTION_SEGMENTS_NAME",
    "CheckpointLineage",
    "ITERATION_MANIFEST_NAME",
    "ITERATION_MANIFEST_SCHEMA_VERSION",
    "LOOP_MANIFEST_NAME",
    "LOOP_MANIFEST_SCHEMA_VERSION",
    "MERGED_DATASET_ROUTE_ID",
    "MERGED_DATASET_SCHEMA_VERSION",
    "MERGED_EPISODE_SCHEMA_VERSION",
    "MERGED_EPISODE_LINEAGE_NAME",
    "MERGED_RECAP_READY_DATASET_DIRNAME",
    "MERGED_RECAP_READY_DATASET_REF_KEY",
    "MERGE_MANIFEST_NAME",
    "MergedDatasetBundle",
    "OFFICIAL_RECAP_RELABEL_ROUTE_ID",
    "build_collection_manifest",
    "build_dataset_mix",
    "build_iteration_manifest",
    "build_loop_manifest",
    "compute_selection_source_hash",
    "load_collection_bundle",
    "load_correction_segments",
    "materialize_collection_bundle",
    "materialize_merged_dataset",
    "normalize_correction_segment",
    "parse_task_ids_csv",
    "read_demo_metadata",
    "read_json",
    "read_jsonl",
    "resolve_checkpoint_lineage",
    "write_json",
    "write_jsonl",
]
