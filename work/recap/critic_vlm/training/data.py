from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ..common import (
    JsonObject,
    as_float,
    as_int,
    as_str,
    parquet_read_table,
    read_json,
    table_scalar,
)
from ..manifest import (
    VlmCriticManifest,
    VlmCriticSample,
    collect_existing_public_roots,
    load_public_warmstart_manifest,
    load_vlm_critic_manifest,
)
from ..targets import resolve_effective_bin_centers
from .contracts import (
    PROPRIO_DIM,
    PUBLIC_WARMSTART_PROMPTS,
    PublicWarmstartSample,
    TrainConfig,
    WarmstartPlan,
)
from .runtime import load_modeling_module, runtime_import_torch_utils


@dataclass(frozen=True)
class LoadedManifestSet:
    train_manifest: VlmCriticManifest
    val_manifest: VlmCriticManifest
    discovery_plan: WarmstartPlan


@dataclass(frozen=True)
class TrainingDataBundle:
    train_manifest: VlmCriticManifest
    val_manifest: VlmCriticManifest
    train_loader: Any
    val_loader: Any
    public_warmstart_loader: Any | None
    public_warmstart_samples: list[PublicWarmstartSample]
    warmstart_plan: WarmstartPlan
    train_sample_count: int
    val_sample_count: int


def _resolve_dataset_meta(dataset_path: Path) -> tuple[str, int]:
    info = read_json(dataset_path / "meta" / "info.json")
    data_path = as_str(info.get("data_path"), context="meta/info.json data_path")
    chunks_size = as_int(info.get("chunks_size"), context="meta/info.json chunks_size")
    return data_path, chunks_size


def _resolve_parquet_path(dataset_path: Path, episode_index: int) -> Path:
    data_path_tmpl, chunks_size = _resolve_dataset_meta(dataset_path)
    chunk_idx = int(episode_index) // int(chunks_size)
    parquet_rel = data_path_tmpl.format(
        episode_chunk=int(chunk_idx),
        episode_index=int(episode_index),
    )
    return (dataset_path / parquet_rel).resolve()


def _as_float_list(value: object, *, context: str) -> list[float]:
    to_list = getattr(value, "tolist", None)
    raw_value = to_list() if callable(to_list) else value
    if isinstance(raw_value, tuple):
        raw_value = list(raw_value)
    if not isinstance(raw_value, list):
        raise ValueError(f"Expected list ({context}), got {type(value).__name__}")
    return [as_float(v, context=f"{context}[{idx}]") for idx, v in enumerate(raw_value)]


def _load_proprio_from_columns(
    *,
    dataset_path: Path,
    sample: VlmCriticSample,
    source_columns: list[str],
) -> list[float]:
    parquet_path = _resolve_parquet_path(dataset_path, sample.episode_index)
    table = parquet_read_table(parquet_path, columns=source_columns)
    values: list[float] = []
    for column in source_columns:
        raw_value = table_scalar(table, column, sample.local_index)
        if isinstance(raw_value, list):
            values.extend(_as_float_list(raw_value, context=column))
        else:
            values.append(as_float(raw_value, context=column))
    if not values:
        raise RuntimeError(
            "proprio_contract_invalid: source_columns produced empty feature vector"
        )
    if len(values) < PROPRIO_DIM:
        values.extend([0.0] * int(PROPRIO_DIM - len(values)))
    return [float(x) for x in values[:PROPRIO_DIM]]


def resolve_sample_proprio(
    sample: VlmCriticSample, manifest: VlmCriticManifest
) -> list[float]:
    status_obj = manifest.side_channel_status.get("proprio")
    if not isinstance(status_obj, dict):
        return [0.0] * PROPRIO_DIM
    source_columns_raw = status_obj.get("source_columns", [])
    source_columns = (
        [
            as_str(v, context="side_channel_status.proprio.source_columns[]")
            for v in source_columns_raw
        ]
        if isinstance(source_columns_raw, list)
        else []
    )
    if not bool(status_obj.get("available_in_dataset", False)) or not source_columns:
        return [0.0] * PROPRIO_DIM
    return _load_proprio_from_columns(
        dataset_path=sample.dataset_path,
        sample=sample,
        source_columns=source_columns,
    )


def load_video_frame(video_path: Path, frame_index: int) -> Any:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"video_decode_open_failed: {video_path}")
    _ = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"video_decode_frame_failed: frame_index={int(frame_index)} path={video_path}"
        )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _ensure_formal_manifest_allowed(
    manifest: VlmCriticManifest,
    *,
    public_roots: list[Path],
) -> None:
    if manifest.formal_eval_scope != "isaac_only":
        raise RuntimeError(
            "formal_split_public_data_forbidden: train/val manifests must declare formal_eval_scope=isaac_only"
        )
    if manifest.task_text_field != "prompt_raw":
        raise RuntimeError(
            f"formal_split_contract_invalid: task_text_field must be 'prompt_raw', got {manifest.task_text_field!r}"
        )
    if manifest.allow_future_frames:
        raise RuntimeError(
            "formal_split_contract_invalid: allow_future_frames must be false"
        )
    dataset_root = manifest.dataset_path.resolve()
    for public_root in public_roots:
        try:
            dataset_root.relative_to(public_root)
        except ValueError:
            continue
        raise RuntimeError(
            "formal_split_public_data_forbidden: formal manifest dataset_path resolves under public warm-start root "
            + str(public_root)
        )


def _resolve_nearest_bin_index(*, value: float, bin_centers: list[float]) -> int:
    best_idx = 0
    best_distance = float("inf")
    for idx, center in enumerate(bin_centers):
        distance = abs(float(center) - float(value))
        if distance < best_distance:
            best_idx = int(idx)
            best_distance = float(distance)
    return int(best_idx)


def _discover_public_dataset_dirs(root: Path) -> list[Path]:
    direct_info = root / "meta" / "info.json"
    if direct_info.exists() and direct_info.is_file():
        return [root.resolve()]
    discovered: list[Path] = []
    for child in sorted(root.iterdir()):
        info_path = child / "meta" / "info.json"
        if child.is_dir() and info_path.exists() and info_path.is_file():
            discovered.append(child.resolve())
    return discovered


def _resolve_public_prompt(*, dataset_root: Path, source_name: str) -> tuple[str, bool]:
    prompt = PUBLIC_WARMSTART_PROMPTS.get(dataset_root.name)
    if prompt:
        return str(prompt), True
    source_prompt = PUBLIC_WARMSTART_PROMPTS.get(source_name)
    if source_prompt:
        return str(source_prompt), True
    return "", False


def _resolve_public_video_key(info: JsonObject) -> str | None:
    features = info.get("features")
    if not isinstance(features, dict):
        return None
    for key, feature in features.items():
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            return str(key)
    return None


def _resolve_public_parquet_path(
    dataset_root: Path,
    info: JsonObject,
    episode_index: int,
) -> Path:
    data_path_tmpl = as_str(info.get("data_path"), context="public.info.data_path")
    chunks_size = as_int(info.get("chunks_size"), context="public.info.chunks_size")
    chunk_idx = int(episode_index) // int(chunks_size)
    return (
        dataset_root
        / data_path_tmpl.format(
            episode_chunk=int(chunk_idx),
            episode_index=int(episode_index),
        )
    ).resolve()


def _resolve_public_video_path(
    dataset_root: Path,
    info: JsonObject,
    *,
    episode_index: int,
    video_key: str,
) -> Path:
    video_path_tmpl = as_str(info.get("video_path"), context="public.info.video_path")
    chunks_size = as_int(info.get("chunks_size"), context="public.info.chunks_size")
    chunk_idx = int(episode_index) // int(chunks_size)
    return (
        dataset_root
        / video_path_tmpl.format(
            episode_chunk=int(chunk_idx),
            episode_index=int(episode_index),
            video_key=str(video_key),
        )
    ).resolve()


def _select_public_row_positions(row_count: int) -> list[int]:
    if row_count <= 0:
        return []
    candidates = [0, row_count // 2, row_count - 1]
    seen: set[int] = set()
    out: list[int] = []
    for raw in candidates:
        idx = int(max(0, min(row_count - 1, int(raw))))
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _limit_samples(samples: list[Any], limit: int | None) -> list[Any]:
    if limit is None or limit <= 0:
        return list(samples)
    return list(samples[: int(limit)])


@dataclass
class WarmstartDataService:
    public_manifest_path: Path

    def _manifest_obj(self) -> JsonObject:
        return load_public_warmstart_manifest(self.public_manifest_path)

    def available_roots(self) -> list[Path]:
        return collect_existing_public_roots(self._manifest_obj())

    def build_discovery_plan(self) -> WarmstartPlan:
        roots = self.available_roots()
        if not roots:
            return WarmstartPlan(
                phase_done=True,
                phase_used_data=False,
                available_local_roots=[],
                used_dataset_roots=[],
                public_sample_count=0,
                note="public_warmstart_unavailable_locally: no approved public warm-start roots are present on this machine",
            )
        return WarmstartPlan(
            phase_done=True,
            phase_used_data=False,
            available_local_roots=[str(x) for x in roots],
            used_dataset_roots=[],
            public_sample_count=0,
            note="public_warmstart_available_but_not_consumed: Task 6 formal trainer consumes Task 4 build manifests; public root discovery is recorded honestly without claiming public samples were used",
        )

    def _resolve_public_sources(self) -> list[tuple[str, Path]]:
        sources = self._manifest_obj().get("sources")
        if not isinstance(sources, list):
            return []
        resolved: list[tuple[str, Path]] = []
        seen: set[tuple[str, str]] = set()
        for idx, raw_source in enumerate(sources):
            if not isinstance(raw_source, dict):
                continue
            source_name = as_str(
                raw_source.get("source_name", f"public_source_{idx}"),
                context=f"public_warmstart.sources[{idx}].source_name",
            )
            for key in ("approved_subset_local_root", "local_root"):
                raw_root = raw_source.get(key)
                exists_flag = bool(
                    raw_source.get(
                        f"{key}_exists",
                        raw_source.get("local_root_exists", False),
                    )
                )
                if (
                    not isinstance(raw_root, str)
                    or not raw_root.strip()
                    or not exists_flag
                ):
                    continue
                root = Path(raw_root).expanduser().resolve()
                if not root.exists() or not root.is_dir():
                    continue
                dedupe_key = (source_name, str(root))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                resolved.append((source_name, root))
        return resolved

    def collect_samples(
        self,
        *,
        bin_centers: list[float],
        limit: int | None,
    ) -> tuple[list[PublicWarmstartSample], list[Path]]:
        sample_limit = None if limit is None or limit <= 0 else int(limit)
        samples: list[PublicWarmstartSample] = []
        used_dataset_roots: list[Path] = []
        used_root_set: set[str] = set()
        episode_uid = 0
        for source_name, source_root in self._resolve_public_sources():
            dataset_dirs = _discover_public_dataset_dirs(source_root)
            if not dataset_dirs:
                continue
            for dataset_root in dataset_dirs:
                if sample_limit is not None and len(samples) >= sample_limit:
                    break
                info_path = dataset_root / "meta" / "info.json"
                if not info_path.exists():
                    continue
                info = read_json(info_path)
                video_key = _resolve_public_video_key(info)
                if video_key is None:
                    continue
                total_episodes = as_int(
                    info.get("total_episodes"),
                    context="public.info.total_episodes",
                )
                discarded_raw = info.get("discarded_episode_indices", [])
                discarded = (
                    {
                        as_int(
                            v,
                            context="public.info.discarded_episode_indices[]",
                        )
                        for v in discarded_raw
                    }
                    if isinstance(discarded_raw, list)
                    else set()
                )
                prompt_raw, use_prompt = _resolve_public_prompt(
                    dataset_root=dataset_root,
                    source_name=source_name,
                )
                for episode_index in range(int(total_episodes)):
                    if sample_limit is not None and len(samples) >= sample_limit:
                        break
                    if episode_index in discarded:
                        continue
                    parquet_path = _resolve_public_parquet_path(
                        dataset_root,
                        info,
                        int(episode_index),
                    )
                    if not parquet_path.exists():
                        continue
                    video_path = _resolve_public_video_path(
                        dataset_root,
                        info,
                        episode_index=int(episode_index),
                        video_key=video_key,
                    )
                    if not video_path.exists():
                        continue
                    table = parquet_read_table(
                        parquet_path,
                        columns=["frame_index", "observation.state"],
                    )
                    row_count = int(getattr(table, "num_rows", 0))
                    if row_count <= 0:
                        continue
                    dataset_root_key = str(dataset_root.resolve())
                    if dataset_root_key not in used_root_set:
                        used_root_set.add(dataset_root_key)
                        used_dataset_roots.append(dataset_root.resolve())
                    for row_index in _select_public_row_positions(row_count):
                        if sample_limit is not None and len(samples) >= sample_limit:
                            break
                        frame_index = as_int(
                            table_scalar(table, "frame_index", int(row_index)),
                            context="public.frame_index",
                        )
                        proprio_value = table_scalar(
                            table,
                            "observation.state",
                            int(row_index),
                        )
                        proprio = _as_float_list(
                            proprio_value,
                            context="public.observation.state",
                        )
                        if len(proprio) < PROPRIO_DIM:
                            proprio.extend([0.0] * int(PROPRIO_DIM - len(proprio)))
                        proprio = [float(x) for x in proprio[:PROPRIO_DIM]]
                        t_norm = float(row_index / max(1, row_count - 1))
                        target_return = float(t_norm)
                        samples.append(
                            PublicWarmstartSample(
                                source_name=str(source_name),
                                dataset_root=dataset_root.resolve(),
                                prompt_raw=str(prompt_raw),
                                use_prompt=bool(
                                    use_prompt and bool(prompt_raw.strip())
                                ),
                                video_path=video_path.resolve(),
                                frame_index=int(frame_index),
                                proprio=proprio,
                                t_norm=float(t_norm),
                                target_return=float(target_return),
                                target_bin_index=_resolve_nearest_bin_index(
                                    value=float(target_return),
                                    bin_centers=bin_centers,
                                ),
                                episode_uid=int(episode_uid),
                            )
                        )
                    episode_uid += 1
        return samples, used_dataset_roots

    def build_effective_plan(
        self,
        *,
        requested_epochs: int,
        used_dataset_roots: list[Path],
        public_sample_count: int,
    ) -> WarmstartPlan:
        roots = self.available_roots()
        if not roots:
            return WarmstartPlan(
                phase_done=True,
                phase_used_data=False,
                available_local_roots=[],
                used_dataset_roots=[],
                public_sample_count=0,
                note="public_warmstart_unavailable_locally: no approved public warm-start roots are present on this machine",
            )
        used_root_strings = [str(x.resolve()) for x in used_dataset_roots]
        if requested_epochs <= 0:
            note = (
                "public_warmstart_available_but_disabled: localized public roots exist, "
                "but warmstart_epochs<=0 so no public samples were consumed"
            )
        elif public_sample_count <= 0:
            note = (
                "public_warmstart_available_but_no_compatible_samples: localized public roots exist, "
                "but no compatible LeRobot public samples were discovered for warm-start"
            )
        else:
            note = (
                "public_warmstart_consumed_localized_data: warm-start consumed localized public LeRobot samples "
                f"from {len(used_root_strings)} dataset root(s)"
            )
        return WarmstartPlan(
            phase_done=True,
            phase_used_data=bool(requested_epochs > 0 and public_sample_count > 0),
            available_local_roots=[str(x) for x in roots],
            used_dataset_roots=used_root_strings,
            public_sample_count=int(public_sample_count),
            note=note,
        )


@dataclass
class CriticTrainingDataLoaderBuilder:
    processor: Any
    batch_size: int
    prompt_text_mode: str
    use_proprio: bool
    use_t_norm: bool

    def _build_text(self, *, prompt_raw: str, use_prompt: bool) -> str:
        effective_use_prompt = bool(use_prompt)
        if str(self.prompt_text_mode) == "constant_query_only":
            effective_use_prompt = False
        return load_modeling_module().build_prompt_text(
            prompt_raw=prompt_raw,
            use_prompt=effective_use_prompt,
        )

    def build_manifest_loader(
        self,
        *,
        manifest: VlmCriticManifest,
        shuffle: bool,
        limit: int | None,
    ) -> tuple[Any, int]:
        torch, _, dataset_mods = runtime_import_torch_utils()
        DataLoader, Dataset = dataset_mods
        selected_samples = cast(
            list[VlmCriticSample], _limit_samples(manifest.samples, limit)
        )

        class _ManifestDataset(Dataset):
            def __len__(self) -> int:
                return len(selected_samples)

            def __getitem__(self, index: int) -> dict[str, object]:
                sample = selected_samples[int(index)]
                image = None
                if sample.input_use_video:
                    if not sample.video_rel:
                        raise RuntimeError(f"video_rel_missing: {sample.sample_id}")
                    image = load_video_frame(
                        sample.dataset_path / sample.video_rel,
                        sample.frame_index,
                    )
                proprio = (
                    resolve_sample_proprio(sample, manifest)
                    if bool(self_outer.use_proprio)
                    else [0.0] * PROPRIO_DIM
                )
                if not sample.input_use_side_channels:
                    proprio = [0.0] * PROPRIO_DIM
                    t_norm = 0.0
                else:
                    t_norm = (
                        float(sample.t_norm) if bool(self_outer.use_t_norm) else 0.0
                    )
                return {
                    "sample": sample,
                    "text": self_outer._build_text(
                        prompt_raw=sample.prompt_raw,
                        use_prompt=bool(sample.input_use_prompt),
                    ),
                    "image": image,
                    "proprio": proprio,
                    "t_norm": t_norm,
                }

        def _collate(batch_items: list[dict[str, object]]) -> dict[str, Any]:
            texts = [
                as_str(item.get("text"), context="batch.text") for item in batch_items
            ]
            images = [item.get("image") for item in batch_items]
            model_inputs = load_modeling_module().prepare_processor_inputs(
                processor=self_outer.processor,
                texts=texts,
                images=images,
            )
            proprio = torch.tensor(
                [cast(list[float], item["proprio"]) for item in batch_items],
                dtype=torch.float32,
            )
            t_norm = torch.tensor(
                [
                    [as_float(item.get("t_norm"), context="batch.t_norm")]
                    for item in batch_items
                ],
                dtype=torch.float32,
            )
            target_bin = torch.tensor(
                [
                    cast(VlmCriticSample, item["sample"]).target_bin_index
                    for item in batch_items
                ],
                dtype=torch.long,
            )
            target_return = torch.tensor(
                [
                    cast(VlmCriticSample, item["sample"]).return_g
                    for item in batch_items
                ],
                dtype=torch.float32,
            )
            episode_index = torch.tensor(
                [
                    cast(VlmCriticSample, item["sample"]).episode_index
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

        self_outer = self
        loader = DataLoader(
            _ManifestDataset(),
            batch_size=int(self.batch_size),
            shuffle=bool(shuffle),
            num_workers=0,
            collate_fn=_collate,
        )
        return loader, len(selected_samples)

    def build_public_warmstart_loader(
        self,
        *,
        public_samples: list[PublicWarmstartSample],
    ) -> Any:
        torch, _, dataset_mods = runtime_import_torch_utils()
        DataLoader, Dataset = dataset_mods

        class _PublicDataset(Dataset):
            def __len__(self) -> int:
                return len(public_samples)

            def __getitem__(self, index: int) -> dict[str, object]:
                sample = public_samples[int(index)]
                return {
                    "text": self_outer._build_text(
                        prompt_raw=sample.prompt_raw,
                        use_prompt=bool(sample.use_prompt),
                    ),
                    "image": load_video_frame(sample.video_path, sample.frame_index),
                    "proprio": sample.proprio
                    if bool(self_outer.use_proprio)
                    else [0.0] * PROPRIO_DIM,
                    "t_norm": float(sample.t_norm)
                    if bool(self_outer.use_t_norm)
                    else 0.0,
                    "target_bin": sample.target_bin_index,
                    "target_return": sample.target_return,
                    "episode_index": sample.episode_uid,
                }

        def _collate(batch_items: list[dict[str, object]]) -> dict[str, Any]:
            texts = [
                as_str(item.get("text"), context="public_batch.text")
                for item in batch_items
            ]
            images = [item.get("image") for item in batch_items]
            model_inputs = load_modeling_module().prepare_processor_inputs(
                processor=self_outer.processor,
                texts=texts,
                images=images,
            )
            proprio = torch.tensor(
                [cast(list[float], item["proprio"]) for item in batch_items],
                dtype=torch.float32,
            )
            t_norm = torch.tensor(
                [
                    [as_float(item.get("t_norm"), context="public_batch.t_norm")]
                    for item in batch_items
                ],
                dtype=torch.float32,
            )
            target_bin = torch.tensor(
                [
                    as_int(item.get("target_bin"), context="public_batch.target_bin")
                    for item in batch_items
                ],
                dtype=torch.long,
            )
            target_return = torch.tensor(
                [
                    as_float(
                        item.get("target_return"),
                        context="public_batch.target_return",
                    )
                    for item in batch_items
                ],
                dtype=torch.float32,
            )
            episode_index = torch.tensor(
                [
                    as_int(
                        item.get("episode_index"),
                        context="public_batch.episode_index",
                    )
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

        self_outer = self
        return DataLoader(
            _PublicDataset(),
            batch_size=int(self.batch_size),
            shuffle=True,
            num_workers=0,
            collate_fn=_collate,
        )


@dataclass
class CriticTrainingDataService:
    config: TrainConfig
    warmstart_service: WarmstartDataService = field(init=False)

    def __post_init__(self) -> None:
        self.warmstart_service = WarmstartDataService(
            self.config.public_warmstart_manifest
        )

    def load_manifests(self) -> LoadedManifestSet:
        train_manifest = load_vlm_critic_manifest(self.config.train_manifest)
        val_manifest = load_vlm_critic_manifest(self.config.val_manifest)
        discovery_plan = self.warmstart_service.build_discovery_plan()
        public_roots = [Path(p).resolve() for p in discovery_plan.available_local_roots]
        _ensure_formal_manifest_allowed(train_manifest, public_roots=public_roots)
        _ensure_formal_manifest_allowed(val_manifest, public_roots=public_roots)
        if train_manifest.bin_centers != val_manifest.bin_centers:
            raise RuntimeError(
                "manifest_bin_spec_mismatch: train/val bin_centers differ"
            )
        return LoadedManifestSet(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            discovery_plan=discovery_plan,
        )

    def build_data_bundle(
        self,
        *,
        processor: Any,
        manifests: LoadedManifestSet,
    ) -> TrainingDataBundle:
        effective_bin_centers = resolve_effective_bin_centers(
            configured_bin_centers=self.config.bin_centers,
            manifest_bin_centers=manifests.train_manifest.bin_centers,
        )
        loader_builder = CriticTrainingDataLoaderBuilder(
            processor=processor,
            batch_size=self.config.batch_size,
            prompt_text_mode=self.config.prompt_text_mode,
            use_proprio=self.config.use_proprio,
            use_t_norm=self.config.use_t_norm,
        )
        train_loader, train_sample_count = loader_builder.build_manifest_loader(
            manifest=manifests.train_manifest,
            shuffle=True,
            limit=self.config.max_train_samples,
        )
        val_loader, val_sample_count = loader_builder.build_manifest_loader(
            manifest=manifests.val_manifest,
            shuffle=False,
            limit=self.config.max_val_samples,
        )
        public_samples, used_public_dataset_roots = (
            self.warmstart_service.collect_samples(
                bin_centers=effective_bin_centers,
                limit=self.config.max_warmstart_samples,
            )
        )
        warmstart_plan = self.warmstart_service.build_effective_plan(
            requested_epochs=int(self.config.warmstart_epochs),
            used_dataset_roots=used_public_dataset_roots,
            public_sample_count=len(public_samples),
        )
        public_loader = None
        if public_samples and int(self.config.warmstart_epochs) > 0:
            public_loader = loader_builder.build_public_warmstart_loader(
                public_samples=public_samples
            )
        return TrainingDataBundle(
            train_manifest=manifests.train_manifest,
            val_manifest=manifests.val_manifest,
            train_loader=train_loader,
            val_loader=val_loader,
            public_warmstart_loader=public_loader,
            public_warmstart_samples=public_samples,
            warmstart_plan=warmstart_plan,
            train_sample_count=int(train_sample_count),
            val_sample_count=int(val_sample_count),
        )
