from __future__ import annotations

import importlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from .common import as_float, as_int, as_str
from .loader import load_critic_artifact
from .schema import (
    ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1,
    QWEN3_VL_LATE_FUSION_BACKEND_V1,
    CriticArtifact,
)


JsonRecord = dict[str, object]

_EPISODE_MP4_NAME_RE = re.compile(r"^(?P<episode_id>.+)_s(?P<stream_idx>\d+)\.mp4$")
_DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class EpisodeVideoInfo:
    episode_id: str
    video_path: Path
    n_policy_steps: int
    frame_count: int


@dataclass
class QwenLabelerRuntime:
    artifact: CriticArtifact
    processor: Any
    critic: Any
    device: str
    torch: Any
    modeling: Any
    architecture: dict[str, object]
    use_prompt: bool
    use_proprio: bool
    use_t_norm: bool
    batch_size: int


class EpisodeVideoReader:
    def __init__(self, video_path: Path) -> None:
        self.video_path = Path(video_path)
        self._cv2 = importlib.import_module("cv2")
        pil_image_mod = importlib.import_module("PIL.Image")
        self._pil_image = getattr(pil_image_mod, "fromarray")
        self._cap = self._cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(
                f"artifact_backend_unavailable: failed to open video {self.video_path}"
            )

    def read_frame(self, frame_index: int) -> object:
        if int(frame_index) < 0:
            raise ValueError(f"negative_frame_index: {frame_index}")
        _ = self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"artifact_backend_unavailable: failed to decode frame_index={frame_index} from {self.video_path}"
            )
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return self._pil_image(rgb)

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass


def _episode_id_from_mp4_name(mp4_path: Path) -> str | None:
    match = _EPISODE_MP4_NAME_RE.match(mp4_path.name)
    if not match:
        return None
    episode_id = match.group("episode_id")
    return str(episode_id) if episode_id else None


def _resolve_batch_size() -> int:
    raw = os.environ.get("RECAP_VLM_CRITIC_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE))
    try:
        value = int(str(raw).strip())
    except Exception:
        value = _DEFAULT_BATCH_SIZE
    return max(1, int(value))


def _resolve_device(torch: Any) -> str:
    requested = str(os.environ.get("RECAP_VLM_CRITIC_DEVICE", "auto")).strip()
    if requested and requested != "auto":
        return requested
    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _get_frame_count(video_path: Path) -> int:
    cv2 = importlib.import_module("cv2")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"artifact_backend_unavailable: failed to open video {video_path}"
        )
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if int(frame_count) <= 0:
        raise RuntimeError(
            f"artifact_backend_unavailable: invalid frame count for {video_path}: {frame_count}"
        )
    return int(frame_count)


def _preview_items(items: list[str], *, limit: int = 5) -> str:
    preview = [str(item) for item in items[:limit]]
    if len(items) > limit:
        preview.append(f"...(+{len(items) - limit} more)")
    return repr(preview)


def _collect_episode_video_infos(
    *,
    all_episode_ids_in_order: list[str],
    episode_by_id: Mapping[str, JsonRecord],
    transitions_by_episode: Mapping[str, list[JsonRecord]],
) -> dict[str, EpisodeVideoInfo]:
    if not all_episode_ids_in_order:
        return {}

    episode_ids_by_video_dir: dict[str, list[str]] = {}
    for episode_id in all_episode_ids_in_order:
        episode = episode_by_id.get(episode_id)
        if not isinstance(episode, dict):
            raise ValueError(f"episode_id={episode_id} missing episode record")
        video_dir_archived = episode.get("video_dir_archived")
        if not isinstance(video_dir_archived, str) or not video_dir_archived:
            raise ValueError(
                f"episode_id={episode_id} missing video_dir_archived for multimodal critic backend"
            )
        episode_ids_by_video_dir.setdefault(video_dir_archived, []).append(episode_id)

    out: dict[str, EpisodeVideoInfo] = {}
    for video_dir_raw, episode_ids_for_dir in episode_ids_by_video_dir.items():
        video_dir_archived = Path(video_dir_raw).expanduser().resolve()
        if not video_dir_archived.is_dir():
            raise FileNotFoundError(
                f"multimodal_video_dir_missing: {video_dir_archived} is not a directory"
            )

        src_mp4s = [p for p in sorted(video_dir_archived.glob("*.mp4")) if p.is_file()]
        expected_ids = set(episode_ids_for_dir)
        parsed_in_expected: dict[str, list[Path]] = {}
        parsed_not_expected: dict[str, list[Path]] = {}
        unparsed_names: list[str] = []
        for mp4_path in src_mp4s:
            parsed_episode_id = _episode_id_from_mp4_name(mp4_path)
            if parsed_episode_id is None:
                unparsed_names.append(mp4_path.name)
                continue
            if parsed_episode_id in expected_ids:
                parsed_in_expected.setdefault(parsed_episode_id, []).append(mp4_path)
            else:
                parsed_not_expected.setdefault(parsed_episode_id, []).append(mp4_path)

        ordered_mp4s: list[Path]
        duplicate_ids = sorted(
            episode_id
            for episode_id, paths in parsed_in_expected.items()
            if len(paths) != 1
        )
        missing_ids = sorted(expected_ids - set(parsed_in_expected.keys()))

        if len(src_mp4s) != len(episode_ids_for_dir):
            if duplicate_ids or missing_ids:
                raise RuntimeError(
                    "multimodal_episode_video_subset_unresolved: "
                    f"episodes={len(episode_ids_for_dir)} mp4s={len(src_mp4s)} "
                    f"matched={len(parsed_in_expected)} parsed_not_expected={len(parsed_not_expected)} "
                    f"unparsed={len(unparsed_names)} video_dir_archived={video_dir_archived} "
                    f"missing_preview={_preview_items(missing_ids)} "
                    f"duplicate_preview={_preview_items(duplicate_ids)} "
                    f"unparsed_preview={_preview_items(sorted(unparsed_names))}"
                )
            ordered_mp4s = [
                parsed_in_expected[episode_id][0] for episode_id in episode_ids_for_dir
            ]
        elif parsed_in_expected:
            if unparsed_names or parsed_not_expected:
                raise RuntimeError(
                    "multimodal_episode_video_pairing_ambiguous: mixed parsed/unparsed mp4 naming detected"
                )
            if duplicate_ids:
                raise RuntimeError(
                    "multimodal_episode_video_pairing_duplicate: " + repr(duplicate_ids)
                )
            if missing_ids:
                raise RuntimeError(
                    "multimodal_episode_video_pairing_missing: " + repr(missing_ids)
                )
            ordered_mp4s = [
                parsed_in_expected[episode_id][0] for episode_id in episode_ids_for_dir
            ]
        else:
            ordered_mp4s = sorted(
                src_mp4s, key=lambda path: (int(path.stat().st_mtime_ns), path.name)
            )

        for episode_id, video_path in zip(
            episode_ids_for_dir, ordered_mp4s, strict=True
        ):
            transitions = transitions_by_episode.get(episode_id)
            if not isinstance(transitions, list) or not transitions:
                raise ValueError(
                    f"episode_id={episode_id} missing transitions for multimodal critic backend"
                )
            frame_count = _get_frame_count(video_path)
            out[episode_id] = EpisodeVideoInfo(
                episode_id=str(episode_id),
                video_path=Path(video_path).resolve(),
                n_policy_steps=int(len(transitions)),
                frame_count=int(frame_count),
            )
    return out


def _policy_step_to_frame_index(
    *, t: int, n_policy_steps: int, frame_count: int
) -> int:
    if int(n_policy_steps) <= 0:
        raise ValueError(f"invalid_n_policy_steps: {n_policy_steps}")
    if int(frame_count) <= 0:
        raise ValueError(f"invalid_frame_count: {frame_count}")
    if int(t) < 0 or int(t) >= int(n_policy_steps):
        raise ValueError(
            f"policy_step_out_of_range: t={t} n_policy_steps={n_policy_steps}"
        )
    frame_index = int((int(t) * int(frame_count)) // int(n_policy_steps))
    if frame_index >= int(frame_count):
        frame_index = int(frame_count - 1)
    return max(0, int(frame_index))


def _load_qwen_runtime(artifact: CriticArtifact) -> QwenLabelerRuntime:
    modeling = importlib.import_module("work.recap.critic_vlm.modeling")
    torch = importlib.import_module("torch")
    model_payload = artifact.model_payload
    if not isinstance(model_payload, dict):
        raise ValueError("artifact_backend_unavailable: qwen model payload missing")
    architecture = model_payload.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("artifact_shape_invalid: qwen architecture missing")
    trainable_state_dict = model_payload.get("trainable_state_dict")
    if not isinstance(trainable_state_dict, dict) or not trainable_state_dict:
        raise ValueError("artifact_shape_invalid: qwen trainable_state_dict missing")

    processor_subdir = as_str(
        artifact.processor_config.get("hf_processor_subdir"),
        context="processor.hf_processor_subdir",
    )
    processor_dir = artifact.paths.critic_dir / "processor" / processor_subdir
    device = _resolve_device(torch)
    processor = modeling.load_qwen3_vl_processor(str(processor_dir))
    backbone = modeling.load_qwen3_vl_backbone(
        base_model=artifact.base_model,
        torch_dtype=modeling.resolve_torch_dtype(device),
        attn_implementation=None,
    )
    critic = modeling.Qwen3VLLateFusionCritic(
        backbone=backbone,
        hidden_size=modeling.resolve_hidden_size(backbone),
        bin_centers=artifact.bin_centers,
        proprio_dim=as_int(
            architecture.get("proprio_dim"), context="architecture.proprio_dim"
        ),
        proprio_hidden_dim=as_int(
            architecture.get("proprio_hidden_dim"),
            context="architecture.proprio_hidden_dim",
        ),
        t_hidden_dim=as_int(
            architecture.get("t_hidden_dim"), context="architecture.t_hidden_dim"
        ),
        fusion_hidden_dim=as_int(
            architecture.get("fusion_hidden_dim"),
            context="architecture.fusion_hidden_dim",
        ),
        use_proprio=bool(architecture.get("use_proprio", True)),
        use_t_norm=bool(architecture.get("use_t_norm", True)),
    )
    critic.freeze_backbone()
    critic.unfreeze_trainable_modules()
    critic.backbone = modeling.apply_top_block_lora(
        critic.backbone,
        top_n=as_int(
            architecture.get("top_n_lora_blocks"),
            context="architecture.top_n_lora_blocks",
        ),
        lora_rank=as_int(
            architecture.get("lora_rank"), context="architecture.lora_rank"
        ),
        lora_alpha=as_int(
            architecture.get("lora_alpha"), context="architecture.lora_alpha"
        ),
        lora_dropout=as_float(
            architecture.get("lora_dropout"), context="architecture.lora_dropout"
        ),
    )
    critic.unfreeze_trainable_modules()
    critic = critic.to(device)
    critic.keep_trainable_path_fp32()
    modeling.load_partial_state_dict(critic, trainable_state_dict)
    critic.eval()

    prompt_text_mode = str(
        artifact.processor_config.get("prompt_text_mode", "manifest")
    ).strip()
    use_prompt = prompt_text_mode != "constant_query_only"
    return QwenLabelerRuntime(
        artifact=artifact,
        processor=processor,
        critic=critic,
        device=device,
        torch=torch,
        modeling=modeling,
        architecture=cast(dict[str, object], architecture),
        use_prompt=bool(use_prompt),
        use_proprio=bool(architecture.get("use_proprio", True)),
        use_t_norm=bool(architecture.get("use_t_norm", True)),
        batch_size=_resolve_batch_size(),
    )


def _validate_artifact_for_labeler(artifact: CriticArtifact) -> None:
    if artifact.artifact_version != ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1:
        raise ValueError(
            "unknown_critic_backend: "
            f"artifact_version={artifact.artifact_version!r} is not supported by labeler multimodal backend"
        )
    task_text_field = as_str(
        artifact.processor_config.get("task_text_field"),
        context="processor.task_text_field",
    )
    if task_text_field != "prompt_raw":
        raise ValueError(
            "artifact_contract_invalid: processor.task_text_field must be 'prompt_raw', "
            f"got {task_text_field!r}"
        )
    frame_policy = as_str(
        artifact.processor_config.get("frame_policy"),
        context="processor.frame_policy",
    )
    if frame_policy != "current_step_index":
        raise ValueError(
            "artifact_contract_invalid: processor.frame_policy must be 'current_step_index', "
            f"got {frame_policy!r}"
        )
    if bool(artifact.processor_config.get("allow_future_frames", False)):
        raise ValueError("artifact_contract_invalid: allow_future_frames must be false")
    if artifact.backend_name != QWEN3_VL_LATE_FUSION_BACKEND_V1:
        raise ValueError(
            "unknown_critic_backend: "
            f"artifact_version={artifact.artifact_version!r} critic_type={artifact.critic_type!r} "
            f"backend_name={artifact.backend_name!r}"
        )


def _predict_episode_values(
    runtime: QwenLabelerRuntime,
    *,
    episode_id: str,
    episode_records: list[JsonRecord],
    video_info: EpisodeVideoInfo,
) -> list[float]:
    if runtime.use_proprio:
        raise ValueError(
            "multimodal_labeler_proprio_unsupported: current M1 labeler backend only supports artifacts with use_proprio=false"
        )

    reader = EpisodeVideoReader(video_info.video_path)
    try:
        out: list[float] = []
        expected_t = 0
        denom = max(1, int(video_info.n_policy_steps) - 1)
        for batch_start in range(0, len(episode_records), int(runtime.batch_size)):
            batch_records = episode_records[
                batch_start : batch_start + int(runtime.batch_size)
            ]
            texts: list[str] = []
            images: list[object | None] = []
            t_norm_values: list[list[float]] = []
            for rec in batch_records:
                t = as_int(rec.get("t"), context=f"episode_id={episode_id} label.t")
                if int(t) != int(expected_t):
                    raise ValueError(
                        f"episode_id={episode_id} invalid transition order for multimodal critic: expected t={expected_t} but got t={t}"
                    )
                expected_t += 1
                frame_index = _policy_step_to_frame_index(
                    t=int(t),
                    n_policy_steps=int(video_info.n_policy_steps),
                    frame_count=int(video_info.frame_count),
                )
                prompt_raw = as_str(
                    rec.get("prompt_raw", ""),
                    context=f"episode_id={episode_id} prompt_raw",
                )
                texts.append(
                    runtime.modeling.build_prompt_text(
                        prompt_raw=prompt_raw,
                        use_prompt=bool(runtime.use_prompt),
                    )
                )
                images.append(reader.read_frame(frame_index))
                if runtime.use_t_norm:
                    t_norm_values.append([[float(t) / float(denom)][0]])

            model_inputs = runtime.modeling.prepare_processor_inputs(
                processor=runtime.processor,
                texts=texts,
                images=images,
            )
            moved_inputs = {
                key: value.to(runtime.device) if hasattr(value, "to") else value
                for key, value in dict(model_inputs).items()
            }
            t_norm = None
            if runtime.use_t_norm:
                t_norm = runtime.torch.tensor(
                    [[float(v[0])] for v in t_norm_values],
                    dtype=runtime.torch.float32,
                    device=runtime.device,
                )

            with runtime.torch.no_grad():
                output = runtime.critic(
                    model_inputs=moved_inputs,
                    proprio=None,
                    t_norm=t_norm,
                )

            values = output["value_V_raw"].detach().float().cpu().reshape(-1).tolist()
            if len(values) != len(batch_records):
                raise RuntimeError(
                    f"episode_id={episode_id} multimodal prediction size mismatch"
                )
            for value in values:
                value_f = float(value)
                if not math.isfinite(value_f):
                    raise ValueError(
                        f"episode_id={episode_id} multimodal critic returned non-finite value_V: {value_f}"
                    )
                out.append(float(value_f))
        return out
    finally:
        reader.close()


def predict_labeler_values(
    dataset: Mapping[str, object],
    *,
    base_records: list[JsonRecord],
    transitions_by_episode: Mapping[str, list[JsonRecord]],
    episode_by_id: Mapping[str, JsonRecord],
    critic_dir: str | Path,
) -> list[float]:
    _ = dataset
    artifact = load_critic_artifact(critic_dir)
    _validate_artifact_for_labeler(artifact)
    runtime = _load_qwen_runtime(artifact)
    print(
        "[INFO] critic_backend_runtime="
        + (
            f"{artifact.critic_type} backend_name={artifact.backend_name} "
            f"device={runtime.device} batch_size={runtime.batch_size} "
            f"use_prompt={runtime.use_prompt} use_t_norm={runtime.use_t_norm}"
        )
    )

    episode_ids_in_order: list[str] = []
    records_by_episode: dict[str, list[JsonRecord]] = {}
    for rec in base_records:
        episode_id = as_str(rec.get("episode_id"), context="label.episode_id")
        if episode_id not in records_by_episode:
            episode_ids_in_order.append(episode_id)
            records_by_episode[episode_id] = []
        records_by_episode[episode_id].append(rec)

    video_infos = _collect_episode_video_infos(
        all_episode_ids_in_order=list(episode_by_id.keys()),
        episode_by_id=episode_by_id,
        transitions_by_episode=transitions_by_episode,
    )

    out: list[float] = []
    total_episodes = len(episode_ids_in_order)
    for episode_index, episode_id in enumerate(episode_ids_in_order, start=1):
        video_info = video_infos[episode_id]
        print(
            "[INFO] critic_label_episode="
            + (
                f"{episode_index}/{total_episodes} episode_id={episode_id} "
                f"n_policy_steps={video_info.n_policy_steps} frame_count={video_info.frame_count}"
            )
        )
        episode_values = _predict_episode_values(
            runtime,
            episode_id=episode_id,
            episode_records=records_by_episode[episode_id],
            video_info=video_info,
        )
        out.extend(episode_values)
    return out


def predict_m1_values(
    dataset: Mapping[str, object],
    *,
    base_records: list[JsonRecord],
    transitions_by_episode: Mapping[str, list[JsonRecord]],
    episode_by_id: Mapping[str, JsonRecord],
    critic_dir: str | Path,
) -> list[float]:
    return predict_labeler_values(
        dataset,
        base_records=base_records,
        transitions_by_episode=transitions_by_episode,
        episode_by_id=episode_by_id,
        critic_dir=critic_dir,
    )


__all__ = ["predict_labeler_values", "predict_m1_values"]
