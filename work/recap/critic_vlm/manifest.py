from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .common import JsonObject, as_float, as_int, as_str, read_json


@dataclass(frozen=True)
class VlmCriticSample:
    sample_id: str
    dataset_path: Path
    split_name: str
    episode_index: int
    episode_length: int
    local_index: int
    frame_index: int
    t: int
    t_norm: float
    recap_episode_id: str
    prompt_raw: str
    return_g: float
    target_bin_index: int
    video_rel: str | None
    video_key: str | None
    input_use_prompt: bool
    input_use_video: bool
    input_use_side_channels: bool

    def to_json(self) -> JsonObject:
        return {
            "sample_id": self.sample_id,
            "dataset_path": str(self.dataset_path),
            "split_name": self.split_name,
            "episode_index": self.episode_index,
            "episode_length": self.episode_length,
            "local_index": self.local_index,
            "frame_index": self.frame_index,
            "t": self.t,
            "t_norm": self.t_norm,
            "recap_episode_id": self.recap_episode_id,
            "prompt_raw": self.prompt_raw,
            "return_G": self.return_g,
            "target_bin_index": self.target_bin_index,
            "video_rel": self.video_rel,
            "video_key": self.video_key,
            "input_use_prompt": self.input_use_prompt,
            "input_use_video": self.input_use_video,
            "input_use_side_channels": self.input_use_side_channels,
        }


@dataclass(frozen=True)
class VlmCriticManifest:
    manifest_path: Path
    dataset_path: Path
    split_name: str
    sample_mode: str
    formal_eval_scope: str
    task_text_field: str
    allow_future_frames: bool
    side_channels: list[str]
    side_channel_status: JsonObject
    input_mode: JsonObject
    bin_centers: list[float]
    samples: list[VlmCriticSample]
    source_build_json: Path

    def to_json(self) -> JsonObject:
        return {
            "manifest_path": str(self.manifest_path),
            "dataset_path": str(self.dataset_path),
            "split_name": self.split_name,
            "sample_mode": self.sample_mode,
            "formal_eval_scope": self.formal_eval_scope,
            "task_text_field": self.task_text_field,
            "allow_future_frames": self.allow_future_frames,
            "side_channels": [str(x) for x in self.side_channels],
            "side_channel_status": dict(self.side_channel_status),
            "input_mode": dict(self.input_mode),
            "bin_count": len(self.bin_centers),
            "source_build_json": str(self.source_build_json),
            "sample_count": len(self.samples),
        }


def _resolve_path(base: Path, raw_value: object, *, context: str) -> Path:
    path_str = as_str(raw_value, context=context)
    path = Path(path_str)
    return path if path.is_absolute() else (base / path)


def _json_obj(value: object, *, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object ({context}), got {type(value).__name__}"
        )
    return cast(JsonObject, value)


def _json_list(value: object, *, context: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list ({context}), got {type(value).__name__}")
    out: list[JsonObject] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected JSON object ({context}[{idx}]), got {type(item).__name__}"
            )
        out.append(cast(JsonObject, item))
    return out


def _resolve_build_manifest(
    manifest_path: Path,
    obj: JsonObject,
) -> tuple[Path, JsonObject, JsonObject]:
    schema_version = as_str(
        obj.get("schema_version"), context="manifest.schema_version"
    )
    if schema_version == "vlm_critic_dataset_build_v1":
        return (
            manifest_path,
            obj,
            {
                "use_prompt": True,
                "use_video": True,
                "use_side_channels": True,
                "allow_future_frames": False,
            },
        )
    if schema_version != "vlm_critic_ablation_manifest_v1":
        raise ValueError(
            "manifest_schema_invalid: expected dataset build or ablation manifest, "
            f"got {schema_version!r}"
        )
    source_build_json = _resolve_path(
        manifest_path.parent,
        obj.get("source_build_json"),
        context="ablation.source_build_json",
    )
    build_obj = read_json(source_build_json)
    build_schema = as_str(
        build_obj.get("schema_version"), context="source_build.schema_version"
    )
    if build_schema != "vlm_critic_dataset_build_v1":
        raise ValueError(
            "manifest_schema_invalid: ablation source_build_json must point to "
            f"vlm_critic_dataset_build_v1, got {build_schema!r}"
        )
    input_mode = _json_obj(obj.get("input_mode"), context="ablation.input_mode")
    sample_ids = obj.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError(
            "manifest_schema_invalid: ablation manifest must contain sample_ids"
        )
    wanted_ids = {as_str(v, context="ablation.sample_ids[]") for v in sample_ids}
    build_samples = _json_list(build_obj.get("samples"), context="source_build.samples")
    filtered = [
        sample
        for sample in build_samples
        if as_str(sample.get("sample_id"), context="sample_id") in wanted_ids
    ]
    if len(filtered) != len(wanted_ids):
        filtered_ids = {
            as_str(sample.get("sample_id"), context="sample.sample_id")
            for sample in filtered
        }
        missing = sorted(wanted_ids - filtered_ids)
        raise ValueError(
            "manifest_schema_invalid: ablation sample_ids missing from source build: "
            + ", ".join(missing[:8])
        )
    build_obj = dict(build_obj)
    build_obj["samples"] = filtered
    return source_build_json, cast(JsonObject, build_obj), input_mode


def load_vlm_critic_manifest(manifest_path: str | Path) -> VlmCriticManifest:
    resolved_path = Path(manifest_path).expanduser().resolve()
    manifest_obj = read_json(resolved_path)
    source_build_json, build_obj, input_mode = _resolve_build_manifest(
        resolved_path, manifest_obj
    )

    dataset_path = _resolve_path(
        source_build_json.parent,
        build_obj.get("dataset_path"),
        context="build.dataset_path",
    ).resolve()
    split_name = as_str(build_obj.get("split_name"), context="build.split_name")
    sample_mode = as_str(build_obj.get("sample_mode"), context="build.sample_mode")
    formal_eval_scope = as_str(
        build_obj.get("formal_eval_scope"), context="build.formal_eval_scope"
    )
    task_text_field = as_str(
        build_obj.get("task_text_field"), context="build.task_text_field"
    )
    allow_future_frames = bool(build_obj.get("allow_future_frames", False))

    raw_side_channels = build_obj.get("side_channels", [])
    if not isinstance(raw_side_channels, list):
        raise ValueError("manifest_schema_invalid: build.side_channels must be a list")
    side_channels = [
        as_str(v, context="build.side_channels[]") for v in raw_side_channels
    ]
    side_channel_status = _json_obj(
        build_obj.get("side_channel_status", {}), context="build.side_channel_status"
    )

    bin_spec = _json_obj(build_obj.get("bin_spec"), context="build.bin_spec")
    raw_bin_centers = bin_spec.get("bin_centers")
    if not isinstance(raw_bin_centers, list) or not raw_bin_centers:
        raise ValueError(
            "manifest_schema_invalid: build.bin_spec.bin_centers must be non-empty"
        )
    bin_centers = [
        as_float(v, context=f"build.bin_spec.bin_centers[{idx}]")
        for idx, v in enumerate(raw_bin_centers)
    ]

    use_prompt = bool(input_mode.get("use_prompt", True))
    use_video = bool(input_mode.get("use_video", True))
    use_side_channels = bool(input_mode.get("use_side_channels", True))
    samples_raw = _json_list(build_obj.get("samples"), context="build.samples")
    samples: list[VlmCriticSample] = []
    for idx, sample in enumerate(samples_raw):
        video_obj = _json_obj(sample.get("video", {}), context=f"sample[{idx}].video")
        target = _json_obj(sample.get("target"), context=f"sample[{idx}].target")
        video_rel: str | None = None
        video_key: str | None = None
        if use_video:
            video_rel = as_str(
                video_obj.get("video_rel"), context=f"sample[{idx}].video_rel"
            )
            video_key = as_str(
                video_obj.get("video_key"), context=f"sample[{idx}].video_key"
            )

        samples.append(
            VlmCriticSample(
                sample_id=as_str(
                    sample.get("sample_id"), context=f"sample[{idx}].sample_id"
                ),
                dataset_path=dataset_path,
                split_name=as_str(
                    sample.get("split_name"), context=f"sample[{idx}].split_name"
                ),
                episode_index=as_int(
                    sample.get("episode_index"), context=f"sample[{idx}].episode_index"
                ),
                episode_length=as_int(
                    sample.get("episode_length"),
                    context=f"sample[{idx}].episode_length",
                ),
                local_index=as_int(
                    sample.get("local_index"), context=f"sample[{idx}].local_index"
                ),
                frame_index=as_int(
                    sample.get("frame_index"), context=f"sample[{idx}].frame_index"
                ),
                t=as_int(sample.get("t"), context=f"sample[{idx}].t"),
                t_norm=as_float(
                    sample.get("t_norm", 0.0), context=f"sample[{idx}].t_norm"
                ),
                recap_episode_id=as_str(
                    sample.get("recap_episode_id"),
                    context=f"sample[{idx}].recap_episode_id",
                ),
                prompt_raw=as_str(
                    sample.get("prompt_raw"), context=f"sample[{idx}].prompt_raw"
                ),
                return_g=as_float(
                    sample.get("return_G"), context=f"sample[{idx}].return_G"
                ),
                target_bin_index=as_int(
                    target.get("target_bin_index"),
                    context=f"sample[{idx}].target.target_bin_index",
                ),
                video_rel=video_rel,
                video_key=video_key,
                input_use_prompt=use_prompt,
                input_use_video=use_video,
                input_use_side_channels=use_side_channels,
            )
        )

    return VlmCriticManifest(
        manifest_path=resolved_path,
        dataset_path=dataset_path,
        split_name=split_name,
        sample_mode=sample_mode,
        formal_eval_scope=formal_eval_scope,
        task_text_field=task_text_field,
        allow_future_frames=allow_future_frames,
        side_channels=side_channels,
        side_channel_status=side_channel_status,
        input_mode=dict(input_mode),
        bin_centers=bin_centers,
        samples=samples,
        source_build_json=source_build_json,
    )


def load_public_warmstart_manifest(path: str | Path) -> JsonObject:
    resolved = Path(path).expanduser().resolve()
    obj = read_json(resolved)
    manifest_obj = obj.get("manifest", obj)
    if not isinstance(manifest_obj, dict):
        raise ValueError(
            "public_warmstart_manifest_invalid: top-level manifest object missing"
        )
    return cast(JsonObject, manifest_obj)


def collect_existing_public_roots(public_manifest: JsonObject) -> list[Path]:
    sources = public_manifest.get("sources")
    if not isinstance(sources, list):
        return []
    roots: list[Path] = []
    for idx, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        for key in ("local_root", "approved_subset_local_root"):
            raw_root = source.get(key)
            exists_flag = bool(
                source.get(f"{key}_exists", source.get("local_root_exists", False))
            )
            if isinstance(raw_root, str) and raw_root.strip() and exists_flag:
                roots.append(Path(raw_root).expanduser().resolve())
    return roots
