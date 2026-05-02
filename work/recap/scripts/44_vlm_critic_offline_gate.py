#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

DEFAULT_DATASET_PATH = "/home/howard/Projects/gr00t_wbc_g1_benchmark/agent/artifacts/lerobot_datasets/recap_mainline_fresh_20260311_121500_k0_wvideo_contract_v1"
DEFAULT_EPISODES_JSONL = "/home/howard/Projects/gr00t_wbc_g1_benchmark/agent/artifacts/recap_datasets/recap_mainline_fresh_20260311_121500_k0/episodes.jsonl"
DEFAULT_OUTPUT_JSON_REL = (
    "agent/artifacts/vlm_critic_offline_gate/task7_offline_gate.json"
)
DEFAULT_EARLY_MAX_T = 5
DEFAULT_QWEN_BATCH_SIZE = 4
DEFAULT_QWEN_DEVICE = "auto"
PASS_SENTINEL = "OFFLINE_GATE_OK"
FAIL_SENTINEL = "OFFLINE_GATE_FAIL"
VERDICT_ALLOW = "ALLOW"
VERDICT_BLOCK = "BLOCK"
REINTEGRATE_ALLOWED = "REINTEGRATE_ALLOWED"
REINTEGRATE_BLOCKED = "REINTEGRATE_BLOCKED"

THRESHOLD_AUC_ALL = 0.75
THRESHOLD_BASELINE_DELTA_AUC = 0.05
THRESHOLD_PROMPT_MARGIN = 0.03
DEGENERATE_RANGE_EPS = 1e-6


@dataclass(frozen=True)
class EvalRow:
    sample_id: str
    split_name: str
    episode_index: int
    recap_episode_id: str
    episode_length: int
    episode_t_max: int
    local_index: int
    frame_index: int
    t: int
    prompt_raw: str
    return_G: float
    state_vec: list[float]
    parquet_rel: str
    video_rel: str
    video_abs: str
    video_decode_backend: str


@dataclass(frozen=True)
class AblationMode:
    name: str
    use_prompt: bool
    use_video: bool
    use_side_channels: bool
    manifest_path: str | None
    source_build_json_path: str | None
    sample_count: int | None
    manifest_input_mode: dict[str, object]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_path(raw_path: str | None, *, default_rel: str) -> Path:
    value = str(raw_path or default_rel).strip()
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        _ = f.write("\n")
    _ = tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return cast(dict[str, object], obj)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object in {path} line {line_no}, got {type(obj).__name__}"
                )
            out.append(cast(dict[str, object], obj))
    return out


def _as_int(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Expected int-like value ({context}), got bool")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"Expected integer-valued float ({context}), got {value!r}"
            )
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Expected int-like value ({context}), got {type(value).__name__}")


def _as_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected float-like value ({context}), got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(
        f"Expected float-like value ({context}), got {type(value).__name__}"
    )


def _as_str(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string ({context}), got {value!r}")
    return str(value)


def _bool_from_mapping(mapping: Mapping[str, object], key: str, default: bool) -> bool:
    raw = mapping.get(key, default)
    if isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Expected bool-like value for {key}, got {raw!r}")


def _log_progress(stage: str, **fields: object) -> None:
    parts = [f"[PROGRESS] {stage}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    print(" ".join(parts))


def _emit_result(
    *, sentinel: str, output_json: Path | None, payload: Mapping[str, object]
) -> None:
    if output_json is not None:
        _log_progress("writing_final_json", path=output_json)
        _write_json(output_json, payload)
        print(f"[INFO] wrote_json: {output_json}")
    verdict_text = payload.get("reintegrate_status")
    if isinstance(verdict_text, str) and verdict_text.strip():
        print(verdict_text)
    print(f"SENTINEL:{sentinel}")


def _auc_binary(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("labels/scores length mismatch")
    pairs = [
        (int(label), float(score)) for label, score in zip(labels, scores, strict=True)
    ]
    pos_n = sum(1 for label, _ in pairs if label == 1)
    neg_n = sum(1 for label, _ in pairs if label == 0)
    if pos_n <= 0 or neg_n <= 0:
        return None
    indexed = sorted(enumerate(pairs), key=lambda item: item[1][1])
    ranks = [0.0] * len(indexed)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1][1] == indexed[cursor][1][1]:
            end += 1
        avg_rank = (float(cursor + 1) + float(end)) / 2.0
        for idx in range(cursor, end):
            original_index = indexed[idx][0]
            ranks[original_index] = avg_rank
        cursor = end
    rank_sum_pos = sum(
        rank for rank, (label, _) in zip(ranks, pairs, strict=True) if label == 1
    )
    auc = (rank_sum_pos - float(pos_n * (pos_n + 1)) / 2.0) / float(pos_n * neg_n)
    return float(auc)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(float(v) for v in values) / float(len(values)))


def _min_max_prob(score: float, *, lower: float, upper: float) -> float:
    if not math.isfinite(score):
        return float("nan")
    if not math.isfinite(lower) or not math.isfinite(upper):
        return float("nan")
    if upper <= lower:
        return 0.5
    clipped = min(max(float(score), float(lower)), float(upper))
    return float((clipped - float(lower)) / float(upper - lower))


def _load_test_manifest(
    test_manifest_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = _read_json(test_manifest_path)
    records_raw = manifest.get("records")
    if not isinstance(records_raw, list) or not records_raw:
        raise ValueError("test_manifest_invalid: records must be a non-empty list")
    records = [
        cast(dict[str, object], rec) for rec in records_raw if isinstance(rec, dict)
    ]
    if len(records) != len(records_raw):
        raise ValueError(
            "test_manifest_invalid: records must contain only JSON objects"
        )
    split_name = _as_str(manifest.get("split_name"), context="test_manifest.split_name")
    if split_name != "test":
        raise ValueError(
            f"test_manifest_invalid: expected split_name='test', got {split_name!r}"
        )
    return manifest, records


def _load_episode_outcomes(episodes_jsonl_path: Path) -> dict[str, dict[str, object]]:
    _log_progress("episode_outcomes_load_start", path=episodes_jsonl_path)
    outcome_by_episode: dict[str, dict[str, object]] = {}
    for rec in _read_jsonl(episodes_jsonl_path):
        episode_id = _as_str(rec.get("episode_id"), context="episodes_jsonl.episode_id")
        outcome_by_episode[episode_id] = {
            "success_episode": bool(rec.get("success_episode", False)),
            "episode_return_online": _as_float(
                rec.get("episode_return_online", 0.0),
                context=f"{episode_id}.episode_return_online",
            ),
            "episode_return_wrapper": _as_float(
                rec.get("episode_return_wrapper", 0.0),
                context=f"{episode_id}.episode_return_wrapper",
            ),
            "n_policy_steps": _as_int(
                rec.get("n_policy_steps", 0), context=f"{episode_id}.n_policy_steps"
            ),
        }
    _log_progress(
        "episode_outcomes_load_end",
        path=episodes_jsonl_path,
        episode_count=len(outcome_by_episode),
    )
    return outcome_by_episode


def _load_ablation_mode(
    path: Path | None, *, name: str, use_prompt: bool, use_video: bool
) -> AblationMode:
    if path is None:
        return AblationMode(
            name=name,
            use_prompt=bool(use_prompt),
            use_video=bool(use_video),
            use_side_channels=True,
            manifest_path=None,
            source_build_json_path=None,
            sample_count=None,
            manifest_input_mode={
                "allow_future_frames": False,
                "sample_mode": "current_step_single_view_ego_view",
                "use_prompt": bool(use_prompt),
                "use_side_channels": True,
                "use_video": bool(use_video),
            },
        )
    obj = _read_json(path)
    input_mode = obj.get("input_mode")
    if not isinstance(input_mode, dict):
        raise ValueError(f"ablation_manifest_invalid: missing input_mode in {path}")
    input_mode_obj = cast(dict[str, object], input_mode)
    source_build_json_path = obj.get("source_build_json")
    sample_count = obj.get("sample_count")
    return AblationMode(
        name=name,
        use_prompt=_bool_from_mapping(input_mode_obj, "use_prompt", use_prompt),
        use_video=_bool_from_mapping(input_mode_obj, "use_video", use_video),
        use_side_channels=_bool_from_mapping(input_mode_obj, "use_side_channels", True),
        manifest_path=str(path),
        source_build_json_path=(
            _as_str(source_build_json_path, context=f"{path}.source_build_json")
            if source_build_json_path is not None
            else None
        ),
        sample_count=(
            _as_int(sample_count, context=f"{path}.sample_count")
            if sample_count is not None
            else None
        ),
        manifest_input_mode=dict(input_mode_obj),
    )


def _import_vlm_modules() -> tuple[Any, Any, Any, Any, Any]:
    common = importlib.import_module("work.recap.critic_vlm.common")
    schema = importlib.import_module("work.recap.critic_vlm.schema")
    loader = importlib.import_module("work.recap.critic_vlm.loader")
    inference = importlib.import_module("work.recap.critic_vlm.inference")
    modeling = importlib.import_module("work.recap.critic_vlm.modeling")
    return common, schema, loader, inference, modeling


def _load_video_frame(video_path: Path, frame_index: int) -> object:
    cv2 = importlib.import_module("cv2")
    pil_image = getattr(importlib.import_module("PIL.Image"), "fromarray")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    _ = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"failed to decode frame_index={frame_index} from {video_path}"
        )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pil_image(rgb)


@dataclass
class QwenRuntime:
    artifact: Any
    processor: Any
    critic: Any
    device: str
    torch: Any
    modeling: Any
    architecture: dict[str, object]


def _load_qwen_runtime(artifact: Any, *, requested_device: str) -> QwenRuntime:
    _, _, _, _, modeling = _import_vlm_modules()
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

    if requested_device == "auto":
        device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    else:
        device = str(requested_device)
    processor_subdir = _as_str(
        artifact.processor_config.get("hf_processor_subdir"),
        context="processor.hf_processor_subdir",
    )
    processor_dir = artifact.paths.critic_dir / "processor" / processor_subdir
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
        proprio_dim=_as_int(
            architecture.get("proprio_dim"), context="architecture.proprio_dim"
        ),
        proprio_hidden_dim=_as_int(
            architecture.get("proprio_hidden_dim"),
            context="architecture.proprio_hidden_dim",
        ),
        t_hidden_dim=_as_int(
            architecture.get("t_hidden_dim"), context="architecture.t_hidden_dim"
        ),
        fusion_hidden_dim=_as_int(
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
        top_n=_as_int(
            architecture.get("top_n_lora_blocks"),
            context="architecture.top_n_lora_blocks",
        ),
        lora_rank=_as_int(
            architecture.get("lora_rank"), context="architecture.lora_rank"
        ),
        lora_alpha=_as_int(
            architecture.get("lora_alpha"), context="architecture.lora_alpha"
        ),
        lora_dropout=_as_float(
            architecture.get("lora_dropout"), context="architecture.lora_dropout"
        ),
    )
    critic.unfreeze_trainable_modules()
    critic = critic.to(device)
    critic.keep_trainable_path_fp32()
    modeling.load_partial_state_dict(critic, trainable_state_dict)
    critic.eval()
    return QwenRuntime(
        artifact=artifact,
        processor=processor,
        critic=critic,
        device=device,
        torch=torch,
        modeling=modeling,
        architecture=cast(dict[str, object], architecture),
    )


def _validate_ablation_sample_ids(rows: Sequence[EvalRow], mode: AblationMode) -> None:
    if not mode.manifest_path:
        return
    manifest = _read_json(Path(mode.manifest_path))
    sample_ids_raw = manifest.get("sample_ids")
    if not isinstance(sample_ids_raw, list):
        raise ValueError(
            f"ablation_manifest_invalid: sample_ids missing in {mode.manifest_path}"
        )
    expected = [str(x) for x in sample_ids_raw]
    actual = [row.sample_id for row in rows]
    if expected != actual:
        raise ValueError(
            f"ablation_manifest_mismatch: sample_ids in {mode.manifest_path} do not match derived held-out rows"
        )


def _select_primary_manifest_mode(
    modes: Sequence[AblationMode],
) -> AblationMode | None:
    for preferred_name in ("full_input", "prompt_only", "vision_only"):
        for mode in modes:
            if mode.name == preferred_name and mode.manifest_path:
                return mode
    return None


def _build_eval_rows_from_build_manifest(
    *,
    primary_mode: AblationMode,
    dataset_path: Path,
    test_manifest_path: Path,
) -> list[EvalRow]:
    if not primary_mode.manifest_path or not primary_mode.source_build_json_path:
        raise ValueError(
            "build_manifest_primary_invalid: missing manifest_path/source_build_json"
        )
    common, _, _, _, _ = _import_vlm_modules()
    _log_progress(
        "baseline_rows_load_start",
        source="build_manifest",
        manifest=primary_mode.manifest_path,
        source_build_json=primary_mode.source_build_json_path,
    )
    manifest_obj = _read_json(Path(primary_mode.manifest_path))
    manifest_dataset_path = manifest_obj.get("dataset_path")
    if isinstance(manifest_dataset_path, str) and manifest_dataset_path.strip():
        candidate = Path(manifest_dataset_path).expanduser().resolve()
        if candidate != dataset_path:
            raise ValueError(
                f"dataset_path mismatch: ablation manifest points to {candidate}, cli resolved {dataset_path}"
            )
    manifest_split_path = manifest_obj.get("split_manifest_path")
    if isinstance(manifest_split_path, str) and manifest_split_path.strip():
        candidate = Path(manifest_split_path).expanduser().resolve()
        if candidate != test_manifest_path:
            raise ValueError(
                f"split_manifest mismatch: ablation manifest points to {candidate}, cli resolved {test_manifest_path}"
            )
    source_build_json_path = (
        Path(primary_mode.source_build_json_path).expanduser().resolve()
    )
    build_obj = _read_json(source_build_json_path)
    build_dataset_path = build_obj.get("dataset_path")
    if isinstance(build_dataset_path, str) and build_dataset_path.strip():
        candidate = Path(build_dataset_path).expanduser().resolve()
        if candidate != dataset_path:
            raise ValueError(
                f"dataset_path mismatch: source build points to {candidate}, cli resolved {dataset_path}"
            )
    build_split_path = build_obj.get("split_manifest_path")
    if isinstance(build_split_path, str) and build_split_path.strip():
        candidate = Path(build_split_path).expanduser().resolve()
        if candidate != test_manifest_path:
            raise ValueError(
                f"split_manifest mismatch: source build points to {candidate}, cli resolved {test_manifest_path}"
            )
    if bool(build_obj.get("allow_future_frames", True)):
        raise ValueError("source_build_invalid: allow_future_frames must be false")

    manifest_sample_ids_raw = manifest_obj.get("sample_ids")
    if not isinstance(manifest_sample_ids_raw, list) or not manifest_sample_ids_raw:
        raise ValueError(
            f"ablation_manifest_invalid: sample_ids missing in {primary_mode.manifest_path}"
        )
    manifest_sample_ids = [str(item) for item in manifest_sample_ids_raw]
    samples_raw = build_obj.get("samples")
    if not isinstance(samples_raw, list) or not samples_raw:
        raise ValueError(
            f"source_build_invalid: samples missing in {source_build_json_path}"
        )
    samples = [
        cast(dict[str, object], item) for item in samples_raw if isinstance(item, dict)
    ]
    if len(samples) != len(samples_raw):
        raise ValueError("source_build_invalid: samples must contain only JSON objects")
    if len(samples) != len(manifest_sample_ids):
        raise ValueError(
            f"source_build_invalid: sample count mismatch source_build={len(samples)} ablation_manifest={len(manifest_sample_ids)}"
        )

    _, test_records = _load_test_manifest(test_manifest_path)
    test_record_by_episode: dict[tuple[int, str], dict[str, object]] = {}
    for record in test_records:
        record_episode_index = _as_int(
            record.get("episode_index"), context="test.records[*].episode_index"
        )
        record_episode_id = _as_str(
            record.get("recap_episode_id"),
            context=f"test.records[{record_episode_index}].recap_episode_id",
        )
        test_record_by_episode[(record_episode_index, record_episode_id)] = record

    required_columns = [
        "recap_m2.t",
        "recap_m2.prompt_raw",
        "recap_m2.return_G",
        "observation.state",
    ]
    parquet_cache: dict[tuple[int, str], dict[str, list[object]]] = {}
    rows: list[EvalRow] = []
    for sample_index, sample in enumerate(samples):
        sample_id = _as_str(
            sample.get("sample_id"), context="source_build.samples[*].sample_id"
        )
        if sample_id != manifest_sample_ids[sample_index]:
            raise ValueError(
                "source_build_invalid: sample_ids do not align with ablation manifest ordering"
            )
        episode_index = _as_int(
            sample.get("episode_index"), context=f"{sample_id}.episode_index"
        )
        recap_episode_id = _as_str(
            sample.get("recap_episode_id"), context=f"{sample_id}.recap_episode_id"
        )
        sample_split_name = _as_str(
            sample.get("split_name"), context=f"{sample_id}.split_name"
        )
        if sample_split_name != "test":
            raise ValueError(
                f"source_build_invalid: expected split_name='test', got {sample_split_name!r} for {sample_id}"
            )
        record_key = (episode_index, recap_episode_id)
        episode_record = test_record_by_episode.get(record_key)
        if episode_record is None:
            raise ValueError(
                f"source_build_invalid: sample episode not found in test manifest for {sample_id}"
            )
        parquet_rel = _as_str(
            episode_record.get("parquet_rel"), context=f"{sample_id}.parquet_rel"
        )
        video_info = cast(dict[str, object], sample.get("video") or {})
        video_rel = _as_str(
            video_info.get("video_rel") or episode_record.get("video_rel"),
            context=f"{sample_id}.video_rel",
        )
        cache_key = (episode_index, recap_episode_id)
        cached = parquet_cache.get(cache_key)
        if cached is None:
            parquet_abs = dataset_path / parquet_rel
            table = common.parquet_read_table(parquet_abs, columns=required_columns)
            cached = {
                "t": list(table.column("recap_m2.t").to_pylist()),
                "prompt_raw": list(table.column("recap_m2.prompt_raw").to_pylist()),
                "return_G": list(table.column("recap_m2.return_G").to_pylist()),
                "state": list(table.column("observation.state").to_pylist()),
            }
            parquet_cache[cache_key] = cached
        local_index = _as_int(
            sample.get("local_index"), context=f"{sample_id}.local_index"
        )
        if local_index < 0 or local_index >= len(cached["state"]):
            raise ValueError(
                f"source_build_invalid: local_index out of range for {sample_id}: {local_index}"
            )
        t_i = _as_int(sample.get("t"), context=f"{sample_id}.t")
        t_from_parquet = _as_int(
            cached["t"][local_index], context=f"{sample_id}.parquet.t"
        )
        if t_from_parquet != t_i:
            raise ValueError(
                f"source_build_invalid: t mismatch for {sample_id}: build={t_i} parquet={t_from_parquet}"
            )
        prompt_raw = _as_str(
            sample.get("prompt_raw"), context=f"{sample_id}.prompt_raw"
        )
        prompt_from_parquet = _as_str(
            cached["prompt_raw"][local_index], context=f"{sample_id}.parquet.prompt_raw"
        )
        if prompt_from_parquet != prompt_raw:
            raise ValueError(
                f"source_build_invalid: prompt_raw mismatch for {sample_id}"
            )
        return_g = _as_float(sample.get("return_G"), context=f"{sample_id}.return_G")
        return_from_parquet = _as_float(
            cached["return_G"][local_index], context=f"{sample_id}.parquet.return_G"
        )
        if abs(float(return_from_parquet) - float(return_g)) > 1e-6:
            raise ValueError(
                f"source_build_invalid: return_G mismatch for {sample_id}: build={return_g} parquet={return_from_parquet}"
            )
        state_raw = cached["state"][local_index]
        if not isinstance(state_raw, Sequence) or isinstance(state_raw, (str, bytes)):
            raise ValueError(
                f"source_build_invalid: observation.state must be a sequence for {sample_id}"
            )
        state_vec = [
            _as_float(item, context=f"{sample_id}.state[{idx}]")
            for idx, item in enumerate(list(state_raw))
        ]
        sample_identity = cast(dict[str, object], sample.get("sample_identity") or {})
        episode_t_max = _as_int(
            sample_identity.get("episode_t_max"),
            context=f"{sample_id}.sample_identity.episode_t_max",
        )
        episode_length = _as_int(
            sample.get("episode_length"), context=f"{sample_id}.episode_length"
        )
        frame_index = _as_int(
            sample.get("frame_index"), context=f"{sample_id}.frame_index"
        )
        decode_backend = _as_str(
            video_info.get("decode_backend") or "validated_by_episode_decode_probes",
            context=f"{sample_id}.video.decode_backend",
        )
        rows.append(
            EvalRow(
                sample_id=sample_id,
                split_name=sample_split_name,
                episode_index=episode_index,
                recap_episode_id=recap_episode_id,
                episode_length=episode_length,
                episode_t_max=episode_t_max,
                local_index=local_index,
                frame_index=frame_index,
                t=t_i,
                prompt_raw=prompt_raw,
                return_G=return_g,
                state_vec=state_vec,
                parquet_rel=str(Path(parquet_rel).as_posix()),
                video_rel=str(Path(video_rel).as_posix()),
                video_abs=str(dataset_path / video_rel),
                video_decode_backend=decode_backend,
            )
        )
    _log_progress(
        "baseline_rows_load_end",
        source="build_manifest",
        sample_count=len(rows),
        episode_count=len({row.recap_episode_id for row in rows}),
        source_build_json=source_build_json_path,
    )
    return rows


def _build_eval_rows(
    *,
    dataset_path: Path,
    test_manifest_path: Path,
    outcome_by_episode: Mapping[str, Mapping[str, object]],
    max_rows_per_episode: int | None,
    probe_video: bool,
) -> list[EvalRow]:
    common, _, _, _, _ = _import_vlm_modules()
    manifest, records = _load_test_manifest(test_manifest_path)
    manifest_dataset_path = manifest.get("dataset_path")
    if isinstance(manifest_dataset_path, str) and manifest_dataset_path.strip():
        candidate = Path(manifest_dataset_path).expanduser().resolve()
        if candidate != dataset_path:
            raise ValueError(
                f"dataset_path mismatch: manifest points to {candidate}, cli resolved {dataset_path}"
            )
    rows: list[EvalRow] = []
    required_columns = [
        "episode_index",
        "index",
        "recap_m2.t",
        "recap_m2.prompt_raw",
        "recap_m2.return_G",
        "observation.state",
    ]
    for rec in records:
        episode_index = _as_int(
            rec.get("episode_index"), context="test.records[*].episode_index"
        )
        recap_episode_id = _as_str(
            rec.get("recap_episode_id"),
            context=f"test.records[{episode_index}].recap_episode_id",
        )
        episode_length = _as_int(
            rec.get("episode_length"),
            context=f"test.records[{episode_index}].episode_length",
        )
        parquet_rel = _as_str(
            rec.get("parquet_rel"), context=f"test.records[{episode_index}].parquet_rel"
        )
        video_rel = _as_str(
            rec.get("video_rel"), context=f"test.records[{episode_index}].video_rel"
        )
        if recap_episode_id not in outcome_by_episode:
            raise ValueError(
                f"missing outcome label for recap_episode_id={recap_episode_id}"
            )
        parquet_abs = dataset_path / parquet_rel
        video_abs = dataset_path / video_rel
        table = common.parquet_read_table(parquet_abs, columns=required_columns)
        episode_indexes = list(table.column("episode_index").to_pylist())
        frame_indexes = list(table.column("index").to_pylist())
        t_values = list(table.column("recap_m2.t").to_pylist())
        prompts = list(table.column("recap_m2.prompt_raw").to_pylist())
        returns = list(table.column("recap_m2.return_G").to_pylist())
        states = list(table.column("observation.state").to_pylist())
        if not frame_indexes:
            raise ValueError(
                f"heldout_rows_invalid: episode_index={episode_index} has zero parquet rows"
            )
        episode_t_max = max(
            _as_int(t_raw_any, context=f"episode_index={episode_index} recap_m2.t[*]")
            for t_raw_any in t_values
        )
        for local_index, values in enumerate(
            zip(
                episode_indexes,
                frame_indexes,
                t_values,
                prompts,
                returns,
                states,
                strict=True,
            )
        ):
            if max_rows_per_episode is not None and local_index >= int(
                max_rows_per_episode
            ):
                break
            (
                episode_index_in_row,
                frame_index_raw,
                t_raw,
                prompt_raw,
                return_g_raw,
                state_raw,
            ) = values
            row_episode_index = _as_int(
                episode_index_in_row,
                context=f"episode_index={episode_index} row[{local_index}].episode_index",
            )
            if row_episode_index != episode_index:
                raise ValueError(
                    f"heldout_rows_invalid: row episode_index mismatch expected={episode_index} got={row_episode_index}"
                )
            frame_index = _as_int(
                frame_index_raw,
                context=f"episode_index={episode_index} row[{local_index}].index",
            )
            t_i = _as_int(
                t_raw, context=f"episode_index={episode_index} row[{local_index}].t"
            )
            if frame_index < 0 or t_i < 0:
                raise ValueError(
                    "heldout_rows_invalid: negative frame index or t is forbidden"
                )
            decode_backend = (
                common.probe_frame_decode(video_abs, frame_index)
                if probe_video
                else "probe_skipped_for_smoke"
            )
            if isinstance(state_raw, Sequence) and not isinstance(
                state_raw, (str, bytes)
            ):
                state_vec = [
                    _as_float(x, context=f"state[{i}]")
                    for i, x in enumerate(list(state_raw))
                ]
            else:
                raise ValueError(
                    f"heldout_rows_invalid: observation.state row must be a sequence, got {type(state_raw).__name__}"
                )
            rows.append(
                EvalRow(
                    sample_id=f"test:{recap_episode_id}:i{int(frame_index):06d}:t{int(t_i):06d}",
                    split_name="test",
                    episode_index=int(episode_index),
                    recap_episode_id=str(recap_episode_id),
                    episode_length=int(episode_length),
                    episode_t_max=int(episode_t_max),
                    local_index=int(local_index),
                    frame_index=int(frame_index),
                    t=int(t_i),
                    prompt_raw=_as_str(
                        prompt_raw,
                        context=f"episode_index={episode_index} row[{local_index}].prompt_raw",
                    ),
                    return_G=_as_float(
                        return_g_raw,
                        context=f"episode_index={episode_index} row[{local_index}].return_G",
                    ),
                    state_vec=state_vec,
                    parquet_rel=str(Path(parquet_rel).as_posix()),
                    video_rel=str(Path(video_rel).as_posix()),
                    video_abs=str(video_abs),
                    video_decode_backend=str(decode_backend),
                )
            )
    return rows


def _predict_baseline_scores(
    rows: Sequence[EvalRow], *, critic_dir: Path
) -> list[float]:
    _log_progress(
        "baseline_scoring_start", critic_dir=critic_dir, sample_count=len(rows)
    )
    recap_mod = importlib.import_module(
        "agent.archive.recap_legacy_state_only_critic.critic_dist_bins"
    )
    predictor = recap_mod.load_critic(str(critic_dir))
    np = importlib.import_module("numpy")
    state_matrix = np.asarray([row.state_vec for row in rows], dtype=np.float32)
    if bool(getattr(predictor.config, "include_t", False)):
        t_values = np.asarray([row.t for row in rows], dtype=np.float32)
        scores = recap_mod.predict_value_V(predictor, state_matrix, t_values)
    else:
        scores = recap_mod.predict_value_V(predictor, state_matrix, None)
    if not isinstance(scores, list) or len(scores) != len(rows):
        raise RuntimeError("baseline prediction size mismatch")
    _log_progress("baseline_scoring_end", critic_dir=critic_dir, sample_count=len(rows))
    return [float(score) for score in scores]


def _predict_multimodal_scores(
    rows: Sequence[EvalRow],
    *,
    artifact: Any,
    mode: AblationMode,
    dataset_path: Path,
    batch_size: int,
    device: str,
) -> list[float]:
    _log_progress(
        f"{mode.name}_forward_start",
        sample_count=len(rows),
        batch_size=batch_size,
        use_prompt=mode.use_prompt,
        use_video=mode.use_video,
    )
    _, schema, _, inference, _ = _import_vlm_modules()
    if artifact.backend_name != "qwen3_vl_late_fusion_v1":
        scores: list[float] = []
        for row in rows:
            sample = schema.DatasetSample(
                dataset_path=dataset_path,
                episode_index=row.episode_index,
                episode_length=row.episode_length,
                recap_episode_id=row.recap_episode_id,
                sample_index=row.local_index,
                local_index=row.local_index,
                frame_index=row.frame_index,
                t=row.t,
                prompt_raw=row.prompt_raw if mode.use_prompt else "",
                return_G=row.return_G,
                parquet_rel=row.parquet_rel,
                video_rel=row.video_rel,
                video_decode_backend=row.video_decode_backend,
            )
            if mode.name == "full_input":
                inference_result = inference.run_critic_inference(artifact, sample)
                scores.append(float(inference_result.value_V_raw))
            else:
                raise ValueError(
                    f"artifact_backend_unavailable: backend={artifact.backend_name!r} does not support ablation mode={mode.name!r}"
                )
        return scores

    runtime = _load_qwen_runtime(artifact, requested_device=device)
    torch = runtime.torch
    architecture = runtime.architecture
    use_proprio = bool(architecture.get("use_proprio", True))
    use_t_norm = bool(architecture.get("use_t_norm", True))
    proprio_dim = _as_int(
        architecture.get("proprio_dim"), context="architecture.proprio_dim"
    )
    out: list[float] = []
    total_batches = max(1, math.ceil(float(len(rows)) / float(max(1, int(batch_size)))))
    for batch_index, batch_start in enumerate(
        range(0, len(rows), int(batch_size)), start=1
    ):
        batch_rows = list(rows[batch_start : batch_start + int(batch_size)])
        batch_end = batch_start + len(batch_rows)
        _log_progress(
            f"{mode.name}_forward_batch",
            batch=batch_index,
            total_batches=total_batches,
            processed=batch_end,
            sample_count=len(rows),
        )
        texts: list[str] = []
        images: list[object | None] = []
        t_norm_values: list[list[float]] = []
        for row in batch_rows:
            text = runtime.modeling.build_prompt_text(
                prompt_raw=row.prompt_raw,
                use_prompt=bool(mode.use_prompt),
            )
            texts.append(text)
            if mode.use_video:
                images.append(
                    _load_video_frame(dataset_path / row.video_rel, row.frame_index)
                )
            else:
                images.append(None)
            t_norm_den = max(1, int(row.episode_t_max))
            t_norm_values.append([float(float(row.t) / float(t_norm_den))])
        batch_inputs = runtime.modeling.prepare_processor_inputs(
            processor=runtime.processor,
            texts=texts,
            images=images,
        )
        moved_batch = {
            key: value.to(runtime.device) if hasattr(value, "to") else value
            for key, value in dict(batch_inputs).items()
        }
        proprio = None
        if bool(mode.use_side_channels) and use_proprio:
            proprio = torch.zeros(
                (len(batch_rows), int(proprio_dim)),
                dtype=torch.float32,
                device=runtime.device,
            )
        t_norm = None
        if use_t_norm:
            t_norm = torch.tensor(
                t_norm_values, dtype=torch.float32, device=runtime.device
            )
        with torch.no_grad():
            output = runtime.critic(
                model_inputs=moved_batch, proprio=proprio, t_norm=t_norm
            )
        values = output["value_V_raw"].detach().float().cpu().reshape(-1).tolist()
        if len(values) != len(batch_rows):
            raise RuntimeError("multimodal prediction size mismatch")
        for value in values:
            out.append(float(value))
    _log_progress(
        f"{mode.name}_forward_end", sample_count=len(rows), batch_size=batch_size
    )
    return out


def _collect_labels(
    rows: Sequence[EvalRow], outcome_by_episode: Mapping[str, Mapping[str, object]]
) -> list[int]:
    labels: list[int] = []
    for row in rows:
        outcome = outcome_by_episode.get(row.recap_episode_id)
        if outcome is None:
            raise ValueError(
                f"missing outcome for recap_episode_id={row.recap_episode_id}"
            )
        labels.append(1 if bool(outcome.get("success_episode", False)) else 0)
    return labels


def _cap_rows_per_episode(
    rows: Sequence[EvalRow], *, max_rows_per_episode: int
) -> list[EvalRow]:
    counts: dict[str, int] = {}
    out: list[EvalRow] = []
    for row in rows:
        count = counts.get(row.recap_episode_id, 0)
        if count >= int(max_rows_per_episode):
            continue
        counts[row.recap_episode_id] = int(count + 1)
        out.append(row)
    return out


def _mode_metrics(
    *,
    rows: Sequence[EvalRow],
    scores: Sequence[float],
    labels: Sequence[int],
    early_max_t: int,
    return_min: float,
    return_max: float,
) -> dict[str, object]:
    if len(rows) != len(scores) or len(rows) != len(labels):
        raise ValueError("metrics input size mismatch")
    finite_scores = [float(score) for score in scores if math.isfinite(float(score))]
    score_range = (
        max(finite_scores) - min(finite_scores) if finite_scores else float("nan")
    )
    success_scores = [
        float(score) for score, label in zip(scores, labels, strict=True) if label == 1
    ]
    fail_scores = [
        float(score) for score, label in zip(scores, labels, strict=True) if label == 0
    ]
    direction_correct = (
        bool(_mean(success_scores) > _mean(fail_scores))
        if success_scores and fail_scores
        else False
    )
    early_indices = [
        idx for idx, row in enumerate(rows) if int(row.t) <= int(early_max_t)
    ]
    early_labels = [int(labels[idx]) for idx in early_indices]
    early_scores = [float(scores[idx]) for idx in early_indices]
    calibration_probs = [
        _min_max_prob(float(score), lower=float(return_min), upper=float(return_max))
        for score in scores
    ]
    calibration_errors = [
        abs(float(prob) - float(label))
        for prob, label in zip(calibration_probs, labels, strict=True)
        if math.isfinite(float(prob))
    ]
    return {
        "n_samples": int(len(rows)),
        "n_success_samples": int(sum(labels)),
        "n_fail_samples": int(len(labels) - sum(labels)),
        "n_early_samples": int(len(early_indices)),
        "degenerate": bool(
            (not finite_scores) or float(score_range) <= float(DEGENERATE_RANGE_EPS)
        ),
        "direction_correct": bool(direction_correct),
        "auc_t0_or_early": _auc_binary(early_labels, early_scores),
        "auc_all": _auc_binary(labels, scores),
        "success_fail_gap": (
            float(_mean(success_scores) - _mean(fail_scores))
            if success_scores and fail_scores
            else None
        ),
        "calibration_mae": (
            float(_mean(calibration_errors)) if calibration_errors else None
        ),
        "success_score_mean": float(_mean(success_scores)) if success_scores else None,
        "fail_score_mean": float(_mean(fail_scores)) if fail_scores else None,
        "score_min": float(min(finite_scores)) if finite_scores else None,
        "score_max": float(max(finite_scores)) if finite_scores else None,
        "score_range": float(score_range) if finite_scores else None,
        "early_scope": {"kind": "t_leq", "max_t": int(early_max_t)},
    }


def _smoke_skipped_metrics(
    *, early_max_t: int, n_samples: int, n_labels: int
) -> dict[str, object]:
    return {
        "n_samples": int(n_samples),
        "n_success_samples": int(n_labels),
        "n_fail_samples": int(max(0, n_samples - n_labels)),
        "n_early_samples": 0,
        "degenerate": True,
        "direction_correct": False,
        "auc_t0_or_early": None,
        "auc_all": None,
        "success_fail_gap": None,
        "calibration_mae": None,
        "success_score_mean": None,
        "fail_score_mean": None,
        "score_min": None,
        "score_max": None,
        "score_range": None,
        "smoke_skipped": True,
        "early_scope": {"kind": "t_leq", "max_t": int(early_max_t)},
    }


def generate_offline_gate_result(
    *,
    critic_dir: Path,
    baseline_critic_dir: Path,
    test_manifest_path: Path,
    dataset_path: Path,
    episodes_jsonl_path: Path,
    prompt_only_manifest_path: Path | None,
    vision_only_manifest_path: Path | None,
    full_input_manifest_path: Path | None,
    early_max_t: int,
    qwen_batch_size: int,
    qwen_device: str,
    max_rows: int | None,
    max_rows_per_episode: int | None,
    smoke_skip_multimodal_forward: bool,
) -> dict[str, object]:
    _, _, loader, _, _ = _import_vlm_modules()
    prompt_mode = _load_ablation_mode(
        prompt_only_manifest_path,
        name="prompt_only",
        use_prompt=True,
        use_video=False,
    )
    vision_mode = _load_ablation_mode(
        vision_only_manifest_path,
        name="vision_only",
        use_prompt=False,
        use_video=True,
    )
    full_mode = _load_ablation_mode(
        full_input_manifest_path,
        name="full_input",
        use_prompt=True,
        use_video=True,
    )
    outcome_by_episode = _load_episode_outcomes(episodes_jsonl_path)
    primary_mode = _select_primary_manifest_mode((prompt_mode, vision_mode, full_mode))
    if primary_mode is not None:
        rows = _build_eval_rows_from_build_manifest(
            primary_mode=primary_mode,
            dataset_path=dataset_path,
            test_manifest_path=test_manifest_path,
        )
    else:
        _log_progress(
            "heldout_rows_load_start",
            source="raw_split",
            test_manifest=test_manifest_path,
            probe_video=not smoke_skip_multimodal_forward,
        )
        rows = _build_eval_rows(
            dataset_path=dataset_path,
            test_manifest_path=test_manifest_path,
            outcome_by_episode=outcome_by_episode,
            max_rows_per_episode=max_rows_per_episode,
            probe_video=not smoke_skip_multimodal_forward,
        )
        _log_progress(
            "heldout_rows_load_end",
            source="raw_split",
            sample_count=len(rows),
            episode_count=len({row.recap_episode_id for row in rows}),
        )
    if max_rows is not None:
        rows = list(rows[: int(max_rows)])
    if not rows:
        raise ValueError("heldout_rows_invalid: no rows selected for offline gate")
    for mode in (prompt_mode, vision_mode, full_mode):
        if mode.manifest_path:
            _log_progress(
                f"{mode.name}_build_manifest_load_start",
                manifest=mode.manifest_path,
                source_build_json=mode.source_build_json_path,
                declared_sample_count=mode.sample_count,
            )
            if primary_mode is not None and mode.source_build_json_path:
                primary_source = primary_mode.source_build_json_path
                if (
                    primary_source
                    and Path(mode.source_build_json_path).expanduser().resolve()
                    != Path(primary_source).expanduser().resolve()
                ):
                    raise ValueError(
                        f"ablation_manifest_invalid: {mode.manifest_path} points to a different source_build_json than the primary manifest"
                    )
        _validate_ablation_sample_ids(rows, mode)
        if mode.manifest_path:
            _log_progress(
                f"{mode.name}_build_manifest_load_end",
                manifest=mode.manifest_path,
                sample_count=len(rows),
            )

    labels = _collect_labels(rows, outcome_by_episode)
    return_min = min(float(row.return_G) for row in rows)
    return_max = max(float(row.return_G) for row in rows)

    heldout_episode_ids = {row.recap_episode_id for row in rows}
    baseline_scores = _predict_baseline_scores(rows, critic_dir=baseline_critic_dir)
    multimodal_artifact = loader.load_critic_artifact(str(critic_dir))

    baseline_metrics = _mode_metrics(
        rows=rows,
        scores=baseline_scores,
        labels=labels,
        early_max_t=int(early_max_t),
        return_min=float(return_min),
        return_max=float(return_max),
    )
    if smoke_skip_multimodal_forward:
        prompt_metrics = _smoke_skipped_metrics(
            early_max_t=int(early_max_t),
            n_samples=len(rows),
            n_labels=sum(labels),
        )
        vision_metrics = _smoke_skipped_metrics(
            early_max_t=int(early_max_t),
            n_samples=len(rows),
            n_labels=sum(labels),
        )
        full_metrics = _smoke_skipped_metrics(
            early_max_t=int(early_max_t),
            n_samples=len(rows),
            n_labels=sum(labels),
        )
    else:
        prompt_scores = _predict_multimodal_scores(
            rows,
            artifact=multimodal_artifact,
            mode=prompt_mode,
            dataset_path=dataset_path,
            batch_size=int(qwen_batch_size),
            device=qwen_device,
        )
        vision_scores = _predict_multimodal_scores(
            rows,
            artifact=multimodal_artifact,
            mode=vision_mode,
            dataset_path=dataset_path,
            batch_size=int(qwen_batch_size),
            device=qwen_device,
        )
        full_scores = _predict_multimodal_scores(
            rows,
            artifact=multimodal_artifact,
            mode=full_mode,
            dataset_path=dataset_path,
            batch_size=int(qwen_batch_size),
            device=qwen_device,
        )
        prompt_metrics = _mode_metrics(
            rows=rows,
            scores=prompt_scores,
            labels=labels,
            early_max_t=int(early_max_t),
            return_min=float(return_min),
            return_max=float(return_max),
        )
        vision_metrics = _mode_metrics(
            rows=rows,
            scores=vision_scores,
            labels=labels,
            early_max_t=int(early_max_t),
            return_min=float(return_min),
            return_max=float(return_max),
        )
        full_metrics = _mode_metrics(
            rows=rows,
            scores=full_scores,
            labels=labels,
            early_max_t=int(early_max_t),
            return_min=float(return_min),
            return_max=float(return_max),
        )

    baseline_auc_all = cast(float | None, baseline_metrics.get("auc_all"))
    full_auc_all = cast(float | None, full_metrics.get("auc_all"))
    prompt_auc_all = cast(float | None, prompt_metrics.get("auc_all"))
    baseline_delta_auc = (
        float(full_auc_all - baseline_auc_all)
        if baseline_auc_all is not None and full_auc_all is not None
        else None
    )
    prompt_margin = (
        float(full_auc_all - prompt_auc_all)
        if prompt_auc_all is not None and full_auc_all is not None
        else None
    )

    threshold_checks: dict[str, dict[str, object]] = {
        "degenerate_false": {
            "passed": not bool(full_metrics.get("degenerate", True)),
            "actual": bool(full_metrics.get("degenerate", True)),
        },
        "direction_correct_true": {
            "passed": bool(full_metrics.get("direction_correct", False)),
            "actual": bool(full_metrics.get("direction_correct", False)),
        },
        "auc_all_gte_0_75": {
            "passed": full_auc_all is not None
            and float(full_auc_all) >= float(THRESHOLD_AUC_ALL),
            "actual": full_auc_all,
            "threshold": float(THRESHOLD_AUC_ALL),
        },
        "baseline_delta_auc_gte_0_05": {
            "passed": baseline_delta_auc is not None
            and float(baseline_delta_auc) >= float(THRESHOLD_BASELINE_DELTA_AUC),
            "actual": baseline_delta_auc,
            "threshold": float(THRESHOLD_BASELINE_DELTA_AUC),
        },
        "prompt_only_margin_gte_0_03": {
            "passed": prompt_margin is not None
            and float(prompt_margin) >= float(THRESHOLD_PROMPT_MARGIN),
            "actual": prompt_margin,
            "threshold": float(THRESHOLD_PROMPT_MARGIN),
        },
    }
    blocking_reasons: list[str] = []
    if bool(full_metrics.get("degenerate", True)):
        blocking_reasons.append("degenerate_predictions")
    if not bool(full_metrics.get("direction_correct", False)):
        blocking_reasons.append("direction_incorrect")
    if full_auc_all is None or float(full_auc_all) < float(THRESHOLD_AUC_ALL):
        blocking_reasons.append("auc_all_below_threshold")
    if baseline_delta_auc is None or float(baseline_delta_auc) < float(
        THRESHOLD_BASELINE_DELTA_AUC
    ):
        blocking_reasons.append("baseline_underperforming")
    if prompt_margin is None or float(prompt_margin) < float(THRESHOLD_PROMPT_MARGIN):
        blocking_reasons.append("prompt_shortcut_risk")
    if bool(smoke_skip_multimodal_forward):
        blocking_reasons.append("multimodal_forward_skipped_for_smoke")
    verdict = VERDICT_ALLOW if not blocking_reasons else VERDICT_BLOCK
    reintegrate_status = (
        REINTEGRATE_ALLOWED if verdict == VERDICT_ALLOW else REINTEGRATE_BLOCKED
    )

    return {
        "schema_version": "vlm_critic_offline_gate_v1",
        "task": "task7_vlm_critic_offline_gate",
        "critic_dir": str(critic_dir),
        "baseline_critic_dir": str(baseline_critic_dir),
        "test_manifest": str(test_manifest_path),
        "dataset_path": str(dataset_path),
        "episodes_jsonl": str(episodes_jsonl_path),
        "sample_count": int(len(rows)),
        "episode_count": int(len({row.recap_episode_id for row in rows})),
        "positive_episode_count": int(
            sum(
                1
                for episode_id, outcome in outcome_by_episode.items()
                if episode_id in heldout_episode_ids
                and bool(outcome.get("success_episode", False))
            )
        ),
        "negative_episode_count": int(
            sum(
                1
                for episode_id, outcome in outcome_by_episode.items()
                if episode_id in heldout_episode_ids
                and not bool(outcome.get("success_episode", False))
            )
        ),
        "row_limit_applied": int(max_rows) if max_rows is not None else None,
        "row_limit_per_episode": (
            int(max_rows_per_episode) if max_rows_per_episode is not None else None
        ),
        "smoke_skip_multimodal_forward": bool(smoke_skip_multimodal_forward),
        "degenerate": bool(full_metrics.get("degenerate", True)),
        "direction_correct": bool(full_metrics.get("direction_correct", False)),
        "auc_t0_or_early": full_metrics.get("auc_t0_or_early"),
        "auc_all": full_auc_all,
        "success_fail_gap": full_metrics.get("success_fail_gap"),
        "calibration_mae": full_metrics.get("calibration_mae"),
        "baseline_delta_auc": baseline_delta_auc,
        "reintegrate_verdict": verdict,
        "reintegrate_status": reintegrate_status,
        "blocking_reasons": blocking_reasons,
        "threshold_checks": threshold_checks,
        "return_scale_anchors": {
            "min_return_G": float(return_min),
            "max_return_G": float(return_max),
        },
        "modes": {
            "baseline_state_only": {
                "critic_type": "state_only_dist_bins",
                "input_mode": {
                    "use_prompt": False,
                    "use_video": False,
                    "use_side_channels": False,
                },
                **baseline_metrics,
            },
            "prompt_only": {
                "input_mode": prompt_mode.manifest_input_mode,
                "manifest_path": prompt_mode.manifest_path,
                **prompt_metrics,
            },
            "vision_only": {
                "input_mode": vision_mode.manifest_input_mode,
                "manifest_path": vision_mode.manifest_path,
                **vision_metrics,
            },
            "full_input": {
                "input_mode": full_mode.manifest_input_mode,
                "manifest_path": full_mode.manifest_path,
                **full_metrics,
            },
        },
        "ablation_summary": {
            "prompt_only_auc": prompt_auc_all,
            "vision_only_auc": vision_metrics.get("auc_all"),
            "full_input_auc": full_auc_all,
            "full_minus_prompt_auc": prompt_margin,
            "full_minus_vision_auc": (
                float(full_auc_all - cast(float, vision_metrics.get("auc_all")))
                if full_auc_all is not None
                and vision_metrics.get("auc_all") is not None
                else None
            ),
        },
        "sample_id_preview": [row.sample_id for row in rows[:10]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="44_vlm_critic_offline_gate.py",
        description="Evaluate held-out offline gate for Task 7 against the real old state-only baseline on the same test split.",
    )
    _ = parser.add_argument("--critic-dir", type=str, required=True)
    _ = parser.add_argument("--baseline-critic-dir", type=str, required=True)
    _ = parser.add_argument("--test-manifest", type=str, required=True)
    _ = parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    _ = parser.add_argument(
        "--episodes-jsonl", type=str, default=DEFAULT_EPISODES_JSONL
    )
    _ = parser.add_argument("--prompt-only-manifest", type=str, default="")
    _ = parser.add_argument("--vision-only-manifest", type=str, default="")
    _ = parser.add_argument("--full-input-manifest", type=str, default="")
    _ = parser.add_argument("--early-max-t", type=int, default=int(DEFAULT_EARLY_MAX_T))
    _ = parser.add_argument(
        "--qwen-batch-size", type=int, default=int(DEFAULT_QWEN_BATCH_SIZE)
    )
    _ = parser.add_argument("--qwen-device", type=str, default=DEFAULT_QWEN_DEVICE)
    _ = parser.add_argument("--max-rows", type=int, default=0)
    _ = parser.add_argument("--max-rows-per-episode", type=int, default=0)
    _ = parser.add_argument("--smoke-skip-multimodal-forward", action="store_true")
    _ = parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON_REL)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_json = _resolve_path(args.output_json, default_rel=DEFAULT_OUTPUT_JSON_REL)
    try:
        result = generate_offline_gate_result(
            critic_dir=_resolve_path(args.critic_dir, default_rel=""),
            baseline_critic_dir=_resolve_path(args.baseline_critic_dir, default_rel=""),
            test_manifest_path=_resolve_path(args.test_manifest, default_rel=""),
            dataset_path=_resolve_path(
                args.dataset_path, default_rel=DEFAULT_DATASET_PATH
            ),
            episodes_jsonl_path=_resolve_path(
                args.episodes_jsonl, default_rel=DEFAULT_EPISODES_JSONL
            ),
            prompt_only_manifest_path=_resolve_path(
                args.prompt_only_manifest, default_rel=""
            )
            if str(args.prompt_only_manifest).strip()
            else None,
            vision_only_manifest_path=_resolve_path(
                args.vision_only_manifest, default_rel=""
            )
            if str(args.vision_only_manifest).strip()
            else None,
            full_input_manifest_path=_resolve_path(
                args.full_input_manifest, default_rel=""
            )
            if str(args.full_input_manifest).strip()
            else None,
            early_max_t=int(args.early_max_t),
            qwen_batch_size=int(args.qwen_batch_size),
            qwen_device=str(args.qwen_device),
            max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
            max_rows_per_episode=(
                int(args.max_rows_per_episode)
                if int(args.max_rows_per_episode) > 0
                else None
            ),
            smoke_skip_multimodal_forward=bool(args.smoke_skip_multimodal_forward),
        )
        _emit_result(sentinel=PASS_SENTINEL, output_json=output_json, payload=result)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "vlm_critic_offline_gate_v1",
            "task": "task7_vlm_critic_offline_gate",
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "reintegrate_status": REINTEGRATE_BLOCKED,
        }
        _emit_result(sentinel=FAIL_SENTINEL, output_json=output_json, payload=failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
