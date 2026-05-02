from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .common import (
    JsonObject,
    as_float,
    as_int,
    as_str,
    parquet_read_table,
    read_json,
    read_jsonl,
)
from .manifest import VlmCriticManifest, VlmCriticSample
from .targets import (
    StepValueTargetInput,
    ValueTarget,
    build_task_max_steps,
    build_value_targets,
    build_value_targets_from_step_inputs,
    encode_value_to_bin_index,
    normalize_empirical_return,
    resolve_effective_bin_centers,
)
from .training.contracts import PROPRIO_DIM
from .training.data import (
    CriticTrainingDataLoaderBuilder,
    CriticTrainingDataService,
    LoadedManifestSet,
    TrainingDataBundle,
    WarmstartDataService,
    load_video_frame,
    resolve_sample_proprio,
)
from .training.runtime import load_modeling_module, runtime_import_torch_utils


DEFAULT_FORMAL_EVAL_SCOPE = "official_native_8d_success_demo"
DEFAULT_SAMPLE_MODE = "current_step_parquet_image_ego_view"
DEFAULT_BIN_COUNT = 21


@dataclass(frozen=True)
class OfficialLiberoEpisode:
    episode_index: int
    prompt_raw: str
    episode_length: int
    split_name: str
    episode_key: str
    task_index: int | None
    episode_success: bool


@dataclass(frozen=True)
class LiberoCriticExample:
    sample_id: str
    split_name: str
    dataset_path: Path
    prompt_raw: str
    task_key: str
    episode_key: str
    episode_index: int
    episode_length: int
    local_index: int
    frame_index: int
    task_max_steps: int
    episode_success: bool
    empirical_return: float
    normalized_return: float
    target_bin_index: int
    proprio: list[float]
    t_norm: float
    image_payload: object

    def to_vlm_critic_sample(self) -> VlmCriticSample:
        return VlmCriticSample(
            sample_id=str(self.sample_id),
            dataset_path=self.dataset_path,
            split_name=str(self.split_name),
            episode_index=int(self.episode_index),
            episode_length=int(self.episode_length),
            local_index=int(self.local_index),
            frame_index=int(self.frame_index),
            t=int(self.local_index),
            t_norm=float(self.t_norm),
            recap_episode_id=str(self.episode_key),
            prompt_raw=str(self.prompt_raw),
            return_g=float(self.empirical_return),
            target_bin_index=int(self.target_bin_index),
            video_rel=None,
            video_key=None,
            input_use_prompt=True,
            input_use_video=False,
            input_use_side_channels=True,
        )


@dataclass(frozen=True)
class LiberoManifestArtifacts:
    manifest_path: Path
    source_build_json_path: Path
    manifest: VlmCriticManifest
    examples: list[LiberoCriticExample]


@dataclass(frozen=True)
class LiberoCriticDatasetBundle:
    dataset_path: Path
    task_max_steps: dict[str, int]
    bin_centers: list[float]
    train: LiberoManifestArtifacts
    val: LiberoManifestArtifacts
    public_warmstart_manifest_path: Path


def build_default_task_normalized_bin_centers(
    *, bin_count: int = DEFAULT_BIN_COUNT
) -> list[float]:
    count = int(bin_count)
    if count <= 1:
        raise ValueError(f"bin_count must be > 1, got {bin_count}")
    step = 1.0 / float(count - 1)
    return [float(-1.0 + (step * idx)) for idx in range(count)]


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _episode_hash(episode_index: int) -> str:
    return hashlib.sha256(f"episode::{int(episode_index)}".encode("utf-8")).hexdigest()


def _split_counts(total_episodes: int) -> tuple[int, int, int]:
    if total_episodes <= 0:
        raise ValueError(f"Expected total_episodes > 0, got {total_episodes}")
    if total_episodes == 1:
        return 1, 0, 0
    if total_episodes == 2:
        return 1, 1, 0
    if total_episodes < 10:
        return total_episodes - 2, 1, 1
    train_n = int(total_episodes * 0.8)
    val_n = int(total_episodes * 0.1)
    test_n = int(total_episodes - train_n - val_n)
    if val_n <= 0:
        val_n = 1
        train_n -= 1
    if test_n <= 0:
        test_n = 1
        train_n -= 1
    if train_n <= 0:
        raise RuntimeError(
            f"split_count_invalid: failed to allocate train split for total_episodes={total_episodes}"
        )
    return train_n, val_n, test_n


def _prompt_task_mapping(dataset_path: Path) -> dict[str, int]:
    tasks = read_jsonl(dataset_path / "meta" / "tasks.jsonl")
    mapping: dict[str, int] = {}
    for row in tasks:
        task_text = as_str(row.get("task"), context="meta/tasks.jsonl task")
        mapping[task_text] = as_int(row.get("task_index"), context="task_index")
    return mapping


def split_official_native_8d_episodes(
    dataset_path: str | Path,
) -> dict[str, list[OfficialLiberoEpisode]]:
    root = Path(dataset_path).expanduser().resolve()
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    prompt_to_task_index = _prompt_task_mapping(root)
    records: list[OfficialLiberoEpisode] = []
    for row in episodes:
        tasks_raw = row.get("tasks")
        if not isinstance(tasks_raw, list) or len(tasks_raw) != 1:
            raise ValueError(
                f"episode tasks must be single-item list, got {tasks_raw!r}"
            )
        prompt_raw = as_str(tasks_raw[0], context="episodes.tasks[0]")
        episode_index = as_int(
            row.get("episode_index"), context="episodes.episode_index"
        )
        episode_length = as_int(row.get("length"), context="episodes.length")
        records.append(
            OfficialLiberoEpisode(
                episode_index=int(episode_index),
                prompt_raw=str(prompt_raw),
                episode_length=int(episode_length),
                split_name="",
                episode_key=f"episode::{int(episode_index)}",
                task_index=prompt_to_task_index.get(prompt_raw),
                episode_success=True,
            )
        )
    ordered = sorted(
        records,
        key=lambda item: (_episode_hash(item.episode_index), item.episode_index),
    )
    train_n, val_n, test_n = _split_counts(len(ordered))
    split_map = {
        "train": ordered[:train_n],
        "val": ordered[train_n : train_n + val_n],
        "test": ordered[train_n + val_n : train_n + val_n + test_n],
    }
    return {
        split_name: [
            OfficialLiberoEpisode(
                episode_index=item.episode_index,
                prompt_raw=item.prompt_raw,
                episode_length=item.episode_length,
                split_name=split_name,
                episode_key=item.episode_key,
                task_index=item.task_index,
                episode_success=item.episode_success,
            )
            for item in split_items
        ]
        for split_name, split_items in split_map.items()
    }


def build_task_max_steps_from_official_native_8d(
    dataset_path: str | Path,
) -> dict[str, int]:
    split_map = split_official_native_8d_episodes(dataset_path)
    task_max_steps: dict[str, int] = {}
    for split_items in split_map.values():
        for item in split_items:
            current = task_max_steps.get(str(item.prompt_raw), 0)
            task_max_steps[str(item.prompt_raw)] = max(
                int(current), int(item.episode_length)
            )
    if not task_max_steps:
        raise ValueError("official/native 8D dataset is empty")
    return task_max_steps


def official_image_payload_to_pil(image_payload: object) -> Any:
    from PIL import Image

    if not isinstance(image_payload, dict):
        raise TypeError(
            f"official image payload must be a mapping, got {type(image_payload).__name__}"
        )
    raw_bytes = cast(dict[str, object], image_payload).get("bytes")
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise TypeError("official image payload is missing bytes")
    with Image.open(io.BytesIO(bytes(raw_bytes))) as image:
        return image.convert("RGB")


def _pad_state_to_proprio(raw_state: object) -> list[float]:
    tolist = getattr(raw_state, "tolist", None)
    values = tolist() if callable(tolist) else raw_state
    if not isinstance(values, list):
        raise TypeError(f"expected list-like state, got {type(raw_state).__name__}")
    out = [float(value) for value in values]
    if len(out) < PROPRIO_DIM:
        out.extend([0.0] * int(PROPRIO_DIM - len(out)))
    return [float(value) for value in out[:PROPRIO_DIM]]


def _resolve_parquet_path(dataset_path: Path, episode_index: int) -> Path:
    info = read_json(dataset_path / "meta" / "info.json")
    data_path_template = as_str(
        info.get("data_path"), context="meta/info.json data_path"
    )
    chunks_size = as_int(info.get("chunks_size"), context="meta/info.json chunks_size")
    chunk_idx = int(episode_index) // int(chunks_size)
    return (
        dataset_path
        / data_path_template.format(
            episode_chunk=int(chunk_idx),
            episode_index=int(episode_index),
        )
    ).resolve()


def build_libero_recap_examples(
    dataset_path: str | Path,
    *,
    bin_centers: list[float] | tuple[float, ...],
    split_name: str | None = None,
    episode_indices: list[int] | tuple[int, ...] | None = None,
    sample_limit: int | None = None,
    split_map: dict[str, list[OfficialLiberoEpisode]] | None = None,
    task_max_steps: dict[str, int] | None = None,
) -> tuple[list[LiberoCriticExample], dict[str, int]]:
    root = Path(dataset_path).expanduser().resolve()
    effective_split_map = (
        split_official_native_8d_episodes(root) if split_map is None else split_map
    )
    effective_task_max_steps = (
        build_task_max_steps_from_official_native_8d(root)
        if task_max_steps is None
        else {str(key): int(value) for key, value in task_max_steps.items()}
    )
    centers = [float(center) for center in bin_centers]
    selected_episodes: list[OfficialLiberoEpisode]
    if episode_indices is not None:
        index_set = {int(value) for value in episode_indices}
        selected_episodes = [
            episode
            for split_items in effective_split_map.values()
            for episode in split_items
            if int(episode.episode_index) in index_set
        ]
        selected_episodes.sort(key=lambda item: int(item.episode_index))
        effective_split_name = str(split_name or "selected")
    else:
        if split_name is None:
            raise ValueError("split_name is required when episode_indices is omitted")
        if split_name not in effective_split_map:
            raise ValueError(f"unsupported split_name {split_name!r}")
        selected_episodes = list(effective_split_map[split_name])
        effective_split_name = str(split_name)

    step_inputs: list[StepValueTargetInput] = []
    step_records: list[
        tuple[OfficialLiberoEpisode, int, object, object, int | None]
    ] = []
    for episode in selected_episodes:
        parquet_path = _resolve_parquet_path(root, episode.episode_index)
        table = parquet_read_table(
            parquet_path,
            columns=["image", "state", "frame_index", "episode_index", "task_index"],
        )
        row_count = int(getattr(table, "num_rows", 0))
        if row_count != int(episode.episode_length):
            raise ValueError(
                "official/native 8D episode length mismatch: "
                + f"episode_index={episode.episode_index} meta={episode.episode_length} parquet={row_count}"
            )
        for row_index, row in enumerate(table.to_pylist()):
            sample_id = (
                f"{effective_split_name}:episode_{int(episode.episode_index):06d}:"
                f"t{int(row_index):06d}"
            )
            task_index = row.get("task_index")
            frame_index = as_int(row.get("frame_index"), context="frame_index")
            episode_index_in_row = as_int(
                row.get("episode_index"), context="episode_index"
            )
            if episode_index_in_row != int(episode.episode_index):
                raise ValueError(
                    "official/native 8D episode_index mismatch: "
                    + f"expected {episode.episode_index}, got {episode_index_in_row}"
                )
            if frame_index != int(row_index):
                raise ValueError(
                    "official/native 8D frame_index mismatch: "
                    + f"episode_index={episode.episode_index} row_index={row_index} frame_index={frame_index}"
                )
            step_inputs.append(
                StepValueTargetInput(
                    sample_id=str(sample_id),
                    task_key=str(episode.prompt_raw),
                    episode_key=str(episode.episode_key),
                    step_index=int(row_index),
                    episode_length=int(episode.episode_length),
                    episode_success=bool(episode.episode_success),
                )
            )
            step_records.append(
                (
                    episode,
                    int(row_index),
                    row.get("image"),
                    row.get("state"),
                    task_index,
                )
            )
            if sample_limit is not None and len(step_inputs) >= int(sample_limit):
                break
        if sample_limit is not None and len(step_inputs) >= int(sample_limit):
            break

    value_targets = build_value_targets_from_step_inputs(
        step_inputs, bin_centers=centers
    )
    examples: list[LiberoCriticExample] = []
    for episode, row_index, image_payload, raw_state, task_index in step_records:
        sample_id = (
            f"{effective_split_name}:episode_{int(episode.episode_index):06d}:"
            f"t{int(row_index):06d}"
        )
        target = value_targets[sample_id]
        if episode.task_index is not None and task_index is not None:
            observed_task_index = as_int(task_index, context="task_index")
            if observed_task_index != int(episode.task_index):
                raise ValueError(
                    "official/native 8D task_index mismatch: "
                    + f"episode_index={episode.episode_index} expected={episode.task_index} got={observed_task_index}"
                )
        examples.append(
            LiberoCriticExample(
                sample_id=str(sample_id),
                split_name=str(effective_split_name),
                dataset_path=root,
                prompt_raw=str(episode.prompt_raw),
                task_key=str(target.task_key),
                episode_key=str(target.episode_key),
                episode_index=int(episode.episode_index),
                episode_length=int(episode.episode_length),
                local_index=int(row_index),
                frame_index=int(row_index),
                task_max_steps=int(target.task_max_steps),
                episode_success=bool(target.episode_success),
                empirical_return=float(target.empirical_return),
                normalized_return=float(target.normalized_return),
                target_bin_index=int(target.target_bin_index),
                proprio=_pad_state_to_proprio(raw_state),
                t_norm=float(row_index / max(1, int(episode.episode_length) - 1)),
                image_payload=image_payload,
            )
        )
    return examples, effective_task_max_steps


def build_libero_recap_dataloader(
    *,
    examples: list[LiberoCriticExample],
    processor: Any,
    batch_size: int,
    prompt_text_mode: str,
    use_proprio: bool,
    use_t_norm: bool,
    shuffle: bool,
) -> Any:
    torch, _, dataset_mods = runtime_import_torch_utils()
    DataLoader, Dataset = dataset_mods

    class _ExampleDataset(Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, index: int) -> dict[str, object]:
            example = examples[int(index)]
            use_prompt = str(prompt_text_mode) != "constant_query_only"
            return {
                "example": example,
                "text": load_modeling_module().build_prompt_text(
                    prompt_raw=example.prompt_raw,
                    use_prompt=use_prompt,
                ),
                "image": official_image_payload_to_pil(example.image_payload),
                "proprio": example.proprio
                if bool(use_proprio)
                else [0.0] * PROPRIO_DIM,
                "t_norm": float(example.t_norm) if bool(use_t_norm) else 0.0,
            }

    def _collate(batch_items: list[dict[str, object]]) -> dict[str, Any]:
        texts = [as_str(item.get("text"), context="batch.text") for item in batch_items]
        images = [item.get("image") for item in batch_items]
        model_inputs = load_modeling_module().prepare_processor_inputs(
            processor=processor,
            texts=texts,
            images=images,
        )
        proprio = torch.tensor(
            [cast(list[float], item["proprio"]) for item in batch_items],
            dtype=torch.float32,
        )
        t_norm = torch.tensor(
            [
                [as_float(item.get("t_norm", 0.0), context="batch.t_norm")]
                for item in batch_items
            ],
            dtype=torch.float32,
        )
        target_bin = torch.tensor(
            [
                cast(LiberoCriticExample, item["example"]).target_bin_index
                for item in batch_items
            ],
            dtype=torch.long,
        )
        target_return = torch.tensor(
            [
                cast(LiberoCriticExample, item["example"]).normalized_return
                for item in batch_items
            ],
            dtype=torch.float32,
        )
        episode_index = torch.tensor(
            [
                cast(LiberoCriticExample, item["example"]).episode_index
                for item in batch_items
            ],
            dtype=torch.long,
        )
        return {
            "model_inputs": model_inputs,
            "proprio": proprio,
            "t_norm": t_norm,
            "target_bin": target_bin,
            "target_return": target_return,
            "episode_index": episode_index,
        }

    return DataLoader(
        _ExampleDataset(),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        collate_fn=_collate,
    )


def _build_manifest_payload(
    *,
    dataset_path: Path,
    split_name: str,
    source_build_json_path: Path,
    examples: list[LiberoCriticExample],
    bin_centers: list[float],
) -> JsonObject:
    samples = []
    for example in examples:
        samples.append(
            {
                "sample_id": str(example.sample_id),
                "split_name": str(split_name),
                "episode_index": int(example.episode_index),
                "episode_length": int(example.episode_length),
                "local_index": int(example.local_index),
                "frame_index": int(example.frame_index),
                "t": int(example.local_index),
                "t_norm": float(example.t_norm),
                "recap_episode_id": str(example.episode_key),
                "prompt_raw": str(example.prompt_raw),
                "video": {},
                "target": {
                    "return_G": float(example.empirical_return),
                    "normalized_return": float(example.normalized_return),
                    "task_max_steps": int(example.task_max_steps),
                    "episode_success": bool(example.episode_success),
                    "target_bin_index": int(example.target_bin_index),
                },
            }
        )
    return {
        "schema_version": "vlm_critic_dataset_build_v1",
        "builder_entrypoint": "work/recap/critic_vlm/data.py",
        "dataset_path": str(dataset_path),
        "split_name": str(split_name),
        "sample_mode": DEFAULT_SAMPLE_MODE,
        "allow_future_frames": False,
        "task_text_field": "prompt_raw",
        "frame_index_field": "frame_index",
        "frame_source_index_field": "frame_index",
        "step_field": "frame_index",
        "episode_length_semantics": "meta_episodes_jsonl.length",
        "policy_step_horizon_semantics": "official_native_success_demo_episode_length",
        "target_mode": "distributional_task_normalized_return",
        "sample_identity_mode": "episode_index_frame_index",
        "sample_identity_fields": ["episode_index", "frame_index"],
        "side_channels": ["proprio", "t_norm"],
        "side_channel_status": {
            "proprio": {
                "available_in_dataset": True,
                "source_columns": ["state"],
            },
            "t_norm": {
                "derived_from": ["frame_index", "meta/episodes.jsonl:length"],
            },
        },
        "video_keys_available": ["parquet_image"],
        "primary_video_key": "parquet_image",
        "primary_video_original_key": "image",
        "formal_eval_scope": DEFAULT_FORMAL_EVAL_SCOPE,
        "split_granularity": "episode",
        "sample_count": int(len(samples)),
        "episode_count": int(len({example.episode_index for example in examples})),
        "bin_spec": {
            "bin_count": int(len(bin_centers)),
            "g_min": float(min(bin_centers)),
            "g_max": float(max(bin_centers)),
            "degenerate_range_adjusted": False,
            "bin_centers": [float(center) for center in bin_centers],
            "index_policy": "nearest_center_round",
        },
        "alignment_report_path": str(
            source_build_json_path.with_suffix(".alignment_report.json")
        ),
        "current_step_policy": {
            "policy_name": "current_row_frame_index",
            "sample_identity_mode": "episode_index_frame_index",
            "require_local_index_match": True,
            "require_step_match": True,
            "step_alignment_semantics": "frame row carries current policy step label",
            "allow_future_frames": False,
        },
        "samples": samples,
        "ablation_manifests": {
            "full_input": str(
                source_build_json_path.with_name(
                    f"{split_name}_manifest.full_input.json"
                )
            )
        },
    }


def _build_ablation_payload(
    *,
    dataset_path: Path,
    source_build_json_path: Path,
    examples: list[LiberoCriticExample],
) -> JsonObject:
    return {
        "schema_version": "vlm_critic_ablation_manifest_v1",
        "source_build_json": str(source_build_json_path),
        "ablation_name": "full_input",
        "dataset_path": str(dataset_path),
        "sample_count": int(len(examples)),
        "input_mode": {
            "sample_mode": DEFAULT_SAMPLE_MODE,
            "use_prompt": True,
            "use_video": False,
            "use_image_columns": True,
            "use_side_channels": True,
            "allow_future_frames": False,
        },
        "sample_ids": [str(example.sample_id) for example in examples],
        "alignment_report_path": str(
            source_build_json_path.with_suffix(".alignment_report.json")
        ),
    }


def _build_manifest_dataclass(
    *,
    manifest_path: Path,
    source_build_json_path: Path,
    dataset_path: Path,
    split_name: str,
    examples: list[LiberoCriticExample],
    bin_centers: list[float],
) -> VlmCriticManifest:
    return VlmCriticManifest(
        manifest_path=manifest_path.resolve(),
        dataset_path=dataset_path.resolve(),
        split_name=str(split_name),
        sample_mode=DEFAULT_SAMPLE_MODE,
        formal_eval_scope=DEFAULT_FORMAL_EVAL_SCOPE,
        task_text_field="prompt_raw",
        allow_future_frames=False,
        side_channels=["proprio", "t_norm"],
        side_channel_status={
            "proprio": {"available_in_dataset": True, "source_columns": ["state"]},
            "t_norm": {"derived_from": ["frame_index", "meta/episodes.jsonl:length"]},
        },
        input_mode={
            "sample_mode": DEFAULT_SAMPLE_MODE,
            "use_prompt": True,
            "use_video": False,
            "use_image_columns": True,
            "use_side_channels": True,
            "allow_future_frames": False,
        },
        bin_centers=[float(center) for center in bin_centers],
        samples=[example.to_vlm_critic_sample() for example in examples],
        source_build_json=source_build_json_path.resolve(),
    )


def write_libero_recap_manifest_artifacts(
    *,
    dataset_path: str | Path,
    manifest_dir: str | Path,
    split_name: str,
    examples: list[LiberoCriticExample],
    bin_centers: list[float],
) -> LiberoManifestArtifacts:
    dataset_root = Path(dataset_path).expanduser().resolve()
    manifest_root = Path(manifest_dir).expanduser().resolve()
    source_build_json_path = manifest_root / f"{split_name}_build.json"
    manifest_path = manifest_root / f"{split_name}_manifest.full_input.json"
    build_payload = _build_manifest_payload(
        dataset_path=dataset_root,
        split_name=split_name,
        source_build_json_path=source_build_json_path,
        examples=examples,
        bin_centers=bin_centers,
    )
    ablation_payload = _build_ablation_payload(
        dataset_path=dataset_root,
        source_build_json_path=source_build_json_path,
        examples=examples,
    )
    alignment_report_path = source_build_json_path.with_suffix(".alignment_report.json")
    _write_json(source_build_json_path, build_payload)
    _write_json(manifest_path, ablation_payload)
    _write_json(
        alignment_report_path,
        {
            "schema_version": "vlm_critic_alignment_report_v1",
            "dataset_path": str(dataset_root),
            "split_name": str(split_name),
            "checked_episodes": int(
                len({example.episode_index for example in examples})
            ),
            "checked_samples": int(len(examples)),
            "checked_rows": int(len(examples)),
            "pass": True,
            "image_source": "parquet.image bytes",
        },
    )
    return LiberoManifestArtifacts(
        manifest_path=manifest_path,
        source_build_json_path=source_build_json_path,
        manifest=_build_manifest_dataclass(
            manifest_path=manifest_path,
            source_build_json_path=source_build_json_path,
            dataset_path=dataset_root,
            split_name=split_name,
            examples=examples,
            bin_centers=bin_centers,
        ),
        examples=list(examples),
    )


def build_libero_recap_dataset_bundle(
    dataset_path: str | Path,
    *,
    manifest_dir: str | Path,
    bin_centers: list[float] | tuple[float, ...] | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> LiberoCriticDatasetBundle:
    dataset_root = Path(dataset_path).expanduser().resolve()
    centers = (
        build_default_task_normalized_bin_centers()
        if not bin_centers
        else [float(center) for center in bin_centers]
    )
    split_map = split_official_native_8d_episodes(dataset_root)
    task_max_steps = build_task_max_steps_from_official_native_8d(dataset_root)
    train_examples, _ = build_libero_recap_examples(
        dataset_root,
        bin_centers=centers,
        split_name="train",
        sample_limit=max_train_samples,
        split_map=split_map,
        task_max_steps=task_max_steps,
    )
    val_examples, _ = build_libero_recap_examples(
        dataset_root,
        bin_centers=centers,
        split_name="val",
        sample_limit=max_val_samples,
        split_map=split_map,
        task_max_steps=task_max_steps,
    )
    train_artifacts = write_libero_recap_manifest_artifacts(
        dataset_path=dataset_root,
        manifest_dir=manifest_dir,
        split_name="train",
        examples=train_examples,
        bin_centers=centers,
    )
    val_artifacts = write_libero_recap_manifest_artifacts(
        dataset_path=dataset_root,
        manifest_dir=manifest_dir,
        split_name="val",
        examples=val_examples,
        bin_centers=centers,
    )
    public_warmstart_manifest_path = (
        Path(manifest_dir).expanduser().resolve() / "public_warmstart_manifest.json"
    )
    _write_json(public_warmstart_manifest_path, {"manifest": {"sources": []}})
    return LiberoCriticDatasetBundle(
        dataset_path=dataset_root,
        task_max_steps=task_max_steps,
        bin_centers=centers,
        train=train_artifacts,
        val=val_artifacts,
        public_warmstart_manifest_path=public_warmstart_manifest_path,
    )


__all__ = [
    "CriticTrainingDataLoaderBuilder",
    "CriticTrainingDataService",
    "LiberoCriticDatasetBundle",
    "LiberoCriticExample",
    "LiberoManifestArtifacts",
    "LoadedManifestSet",
    "OfficialLiberoEpisode",
    "TrainingDataBundle",
    "ValueTarget",
    "WarmstartDataService",
    "build_default_task_normalized_bin_centers",
    "build_libero_recap_dataloader",
    "build_libero_recap_dataset_bundle",
    "build_libero_recap_examples",
    "build_task_max_steps",
    "build_task_max_steps_from_official_native_8d",
    "build_value_targets",
    "build_value_targets_from_step_inputs",
    "encode_value_to_bin_index",
    "load_video_frame",
    "normalize_empirical_return",
    "official_image_payload_to_pil",
    "resolve_effective_bin_centers",
    "resolve_sample_proprio",
    "split_official_native_8d_episodes",
    "write_libero_recap_manifest_artifacts",
]
