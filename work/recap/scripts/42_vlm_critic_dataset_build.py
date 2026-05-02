#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast


JsonObject = dict[str, object]


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


DEFAULT_OUTPUT_JSON_REL = (
    "agent/artifacts/vlm_critic_manifests/task4_dataset_build.json"
)
DEFAULT_BINS = 201
DEFAULT_SAMPLE_IDENTITY_MODE = "auto"

SAMPLE_IDENTITY_MODE_AUTO = "auto"
SAMPLE_IDENTITY_MODE_FIXTURE = "fixture_local_index_equals_index_equals_t"
SAMPLE_IDENTITY_MODE_ROW_FRAME_T = "row_frame_index_plus_t"


PASS_SENTINEL = "VLM_CRITIC_BUILD_OK"
FAIL_SENTINEL = "VLM_CRITIC_BUILD_FAIL"


class BuildBlocker(RuntimeError):
    alignment_report: dict[str, object]

    def __init__(self, message: str, *, alignment_report: Mapping[str, object]) -> None:
        super().__init__(message)
        self.alignment_report = dict(alignment_report)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(repo_root: Path, raw_path: str | None, *, default_rel: str) -> Path:
    value = str(raw_path or default_rel)
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        _ = f.write("\n")
    _ = tmp_path.replace(path)


def _read_json(path: Path) -> JsonObject:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Expected file, got {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return cast(JsonObject, obj)


def _read_jsonl(path: Path) -> list[JsonObject]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Expected file, got {path}")
    out: list[JsonObject] = []
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
            out.append(cast(JsonObject, obj))
    return out


def _json_list_of_dicts(value: object, *, context: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise ValueError(f"Expected list ({context}), got {type(value).__name__}")
    out: list[JsonObject] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected JSON object in {context}[{idx}], got {type(item).__name__}"
            )
        out.append(cast(JsonObject, item))
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
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"Expected int-like string ({context}), got {value!r}"
            ) from exc
    raise ValueError(f"Expected int-like value ({context}), got {type(value).__name__}")


def _as_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected float-like value ({context}), got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Expected float-like string ({context}), got {value!r}"
            ) from exc
    raise ValueError(
        f"Expected float-like value ({context}), got {type(value).__name__}"
    )


def _as_str(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string ({context}), got {value!r}")
    return str(value)


def _import_parquet_module() -> Any:
    try:
        return importlib.import_module("pyarrow.parquet")
    except Exception as exc:
        raise RuntimeError(f"dataset_build_missing_pyarrow: {exc}") from exc


def _parquet_read_table(parquet_path: Path, *, columns: list[str] | None = None) -> Any:
    pq = _import_parquet_module()
    try:
        return pq.read_table(str(parquet_path), columns=columns)
    except Exception as exc:
        raise RuntimeError(
            f"dataset_build_parquet_read_failed: {parquet_path}: {exc}"
        ) from exc


def _ffprobe_num_frames(video_path: Path) -> tuple[int | None, str | None]:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return None, "ffprobe_missing"
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return None, f"ffprobe_rc={proc.returncode} err={err[:240]}"
    try:
        obj = json.loads(proc.stdout)
    except Exception as exc:
        return None, f"ffprobe_json_error: {exc}"
    streams = obj.get("streams") if isinstance(obj, dict) else None
    if not isinstance(streams, list) or not streams:
        return None, "ffprobe_streams_missing"
    s0 = streams[0]
    if not isinstance(s0, dict):
        return None, "ffprobe_stream0_invalid"
    nb = s0.get("nb_frames")
    if isinstance(nb, int) and nb > 0:
        return int(nb), None
    if isinstance(nb, str):
        nb_s = nb.strip()
        if nb_s.isdigit():
            parsed = int(nb_s)
            if parsed > 0:
                return parsed, None
    return None, "ffprobe_nb_frames_missing"


def _opencv_num_frames(video_path: Path) -> tuple[int | None, str | None]:
    try:
        cv2 = importlib.import_module("cv2")
    except Exception as exc:
        return None, f"opencv_import_error: {exc}"
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, "opencv_cap_not_opened"
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except Exception as exc:
        return None, f"opencv_count_error: {exc}"
    if count <= 0:
        return None, f"opencv_count_invalid: {count}"
    return count, None


def _resolve_num_frames(video_path: Path) -> tuple[int | None, str | None]:
    ff_n, ff_err = _ffprobe_num_frames(video_path)
    if ff_n is not None:
        return ff_n, "ffprobe.nb_frames"
    cv_n, cv_err = _opencv_num_frames(video_path)
    if cv_n is not None:
        return cv_n, "opencv.frame_count"
    return None, f"ffprobe={ff_err}; opencv={cv_err}"


def _ffmpeg_frame_probe(video_path: Path, frame_index: int) -> tuple[bool, str | None]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False, "ffmpeg_missing"
    if frame_index < 0:
        return False, f"negative_frame_index: {frame_index}"
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{int(frame_index)})",
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = ((proc.stderr or b"") or (proc.stdout or b"")).decode(
            "utf-8", errors="replace"
        )
        return False, f"ffmpeg_rc={proc.returncode} err={err[:240].strip()}"
    if not proc.stdout:
        return False, "ffmpeg_empty_stdout"
    return True, None


def _opencv_frame_probe(video_path: Path, frame_index: int) -> tuple[bool, str | None]:
    try:
        cv2 = importlib.import_module("cv2")
    except Exception as exc:
        return False, f"opencv_import_error: {exc}"
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "opencv_cap_not_opened"
        _ = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        cap.release()
    except Exception as exc:
        return False, f"opencv_read_error: {exc}"
    if not ok or frame is None:
        return False, "opencv_frame_read_failed"
    return True, None


def _probe_frame_decode(video_path: Path, frame_index: int) -> str:
    ok, err = _ffmpeg_frame_probe(video_path, frame_index)
    if ok:
        return "ffmpeg.frame_probe"
    ok, cv_err = _opencv_frame_probe(video_path, frame_index)
    if ok:
        return "opencv.frame_probe"
    raise RuntimeError(
        f"video_decode_missing: cannot decode frame_index={frame_index} from {video_path}: ffmpeg={err}; opencv={cv_err}"
    )


def _table_column_pylist(table: Any, column: str) -> list[object]:
    try:
        return list(cast(list[object], table.column(column).to_pylist()))
    except Exception as exc:
        raise RuntimeError(
            f"dataset_build_column_read_failed: column={column!r}: {exc}"
        ) from exc


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    if count == 1:
        return [float(start)]
    step = (float(end) - float(start)) / float(count - 1)
    return [float(float(start) + float(i) * float(step)) for i in range(count)]


def _to_bin_index(value: float, *, g_min: float, g_max: float, bins: int) -> int:
    denom = float(g_max) - float(g_min)
    if denom <= 0.0:
        raise RuntimeError(
            f"dataset_build_invalid_bin_range: g_min={g_min} g_max={g_max}"
        )
    u = (float(value) - float(g_min)) / float(denom)
    idx_f = float(u) * float(bins - 1)
    idx_i = int(round(idx_f))
    return int(max(0, min(int(bins - 1), int(idx_i))))


def _derive_alignment_report_path(output_json: Path) -> Path:
    return output_json.with_name(f"{output_json.stem}.alignment_report.json")


def _derive_ablation_paths(output_json: Path) -> dict[str, Path]:
    return {
        "full_input": output_json.with_name(f"{output_json.stem}.full_input.json"),
        "vision_only": output_json.with_name(f"{output_json.stem}.vision_only.json"),
        "prompt_only": output_json.with_name(f"{output_json.stem}.prompt_only.json"),
    }


def _resolve_proprio_sources(features: Mapping[str, object]) -> list[str]:
    preferred = [
        "observation.state",
        "observation.proprio",
        "recap_m2.proprio",
        "proprio",
        "state",
    ]
    out: list[str] = []
    for key in preferred:
        value = features.get(key)
        if isinstance(value, dict) and value.get("dtype") != "video":
            out.append(str(key))
    return out


def _build_side_channel_status(features: Mapping[str, object]) -> JsonObject:
    proprio_sources = _resolve_proprio_sources(features)
    return {
        "proprio": {
            "declared": True,
            "available_in_dataset": bool(proprio_sources),
            "source_columns": [str(x) for x in proprio_sources],
            "status": (
                "dataset_column_present"
                if proprio_sources
                else "declared_for_downstream_contract_but_missing_in_current_dataset"
            ),
        },
        "t_norm": {
            "declared": True,
            "available_in_dataset": True,
            "derived_from": ["recap_m2.t", "episode_length"],
            "status": "derived_current_step_scalar",
        },
    }


def _honest_sample_mode(video_keys_available: list[str]) -> str:
    keys = [str(k) for k in video_keys_available]
    if len(keys) <= 1:
        if keys == ["ego_view"]:
            return "current_step_single_view_ego_view"
        return "current_step_single_view"
    return "current_step_multiview"


def _input_mode_payload(
    sample_mode: str, *, use_prompt: bool, use_video: bool
) -> JsonObject:
    return {
        "sample_mode": str(sample_mode),
        "use_prompt": bool(use_prompt),
        "use_video": bool(use_video),
        "use_side_channels": bool(use_prompt or use_video),
        "allow_future_frames": False,
    }


def _validate_split_manifest(
    split_manifest: JsonObject,
) -> tuple[str, list[JsonObject]]:
    schema_version = _as_str(
        split_manifest.get("schema_version"), context="split_manifest.schema_version"
    )
    if schema_version != "vlm_critic_split_manifest_v1":
        raise ValueError(
            f"split_manifest_invalid: expected schema_version='vlm_critic_split_manifest_v1', got {schema_version!r}"
        )
    if (
        _as_str(split_manifest.get("split_granularity"), context="split_granularity")
        != "episode"
    ):
        raise ValueError(
            "split_manifest_invalid: Task 4 requires episode-granularity split manifests"
        )
    if not bool(split_manifest.get("sample_level_split_forbidden", False)):
        raise ValueError(
            "split_manifest_invalid: sample_level_split_forbidden must be true"
        )
    split_name = _as_str(split_manifest.get("split_name"), context="split_name")
    records = _json_list_of_dicts(split_manifest.get("records"), context="records")
    if not records:
        raise ValueError("split_manifest_invalid: records must be non-empty")
    return split_name, records


def _raise_blocker(
    message: str,
    *,
    report: dict[str, object],
    violations_preview: list[str],
    decode_fail_count: int,
    future_frame_violation_count: int,
    episode_leakage_count: int,
    frame_t_mismatch_count: int,
) -> None:
    report["pass"] = False
    report["decode_fail_count"] = int(decode_fail_count)
    report["future_frame_violation_count"] = int(future_frame_violation_count)
    report["episode_leakage_count"] = int(episode_leakage_count)
    report["frame_t_mismatch_count"] = int(frame_t_mismatch_count)
    report["violations_preview"] = list(violations_preview[:16])
    report["blocker"] = str(message)
    raise BuildBlocker(str(message), alignment_report=report)


def _resolve_sample_identity_mode(
    *, requested_mode: str, row_count: int, episode_length: int
) -> str:
    mode = str(requested_mode).strip()
    if mode == SAMPLE_IDENTITY_MODE_AUTO:
        if int(row_count) == int(episode_length):
            return SAMPLE_IDENTITY_MODE_FIXTURE
        return SAMPLE_IDENTITY_MODE_ROW_FRAME_T
    if mode not in {
        SAMPLE_IDENTITY_MODE_FIXTURE,
        SAMPLE_IDENTITY_MODE_ROW_FRAME_T,
    }:
        raise ValueError(
            f"dataset_build_invalid: unsupported sample_identity_mode={mode!r}"
        )
    return mode


def _decode_probe_indices(max_frame_index: int) -> list[int]:
    if max_frame_index < 0:
        raise ValueError(f"max_frame_index must be >= 0, got {max_frame_index}")
    midpoint = int(max_frame_index // 2)
    return sorted({0, midpoint, int(max_frame_index)})


def _return_group_consistent(group_rows: list[JsonObject], *, expected: float) -> bool:
    for row in group_rows:
        value = _as_float(row.get("return_G"), context="policy_step_group.return_G")
        if not math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-6):
            return False
    return True


def _source_index_to_frame_index(
    *, source_index: int, source_horizon: int, frame_count: int | None
) -> int:
    if source_index < 0:
        raise ValueError(f"source_index must be >= 0, got {source_index}")
    if frame_count is None:
        return int(source_index)
    if frame_count <= 0:
        raise ValueError(f"frame_count must be > 0, got {frame_count}")
    if source_horizon <= 0:
        raise ValueError(f"source_horizon must be > 0, got {source_horizon}")
    if frame_count == source_horizon:
        return int(min(int(frame_count - 1), int(source_index)))
    mapped = int(
        math.floor(float(source_index) * float(frame_count) / float(source_horizon))
    )
    return int(max(0, min(int(frame_count - 1), int(mapped))))


def _build_dataset_manifest(
    *,
    dataset_path: Path,
    split_manifest_path: Path,
    output_json: Path,
    bins: int,
    sample_identity_mode: str,
) -> tuple[JsonObject, JsonObject, dict[str, JsonObject]]:
    if bins <= 1:
        raise ValueError(f"bins must be >= 2, got {bins}")
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"dataset_path is not a directory: {dataset_path}")
    if not split_manifest_path.exists() or not split_manifest_path.is_file():
        raise FileNotFoundError(f"split_manifest is not a file: {split_manifest_path}")

    meta_dir = dataset_path / "meta"
    info = _read_json(meta_dir / "info.json")
    modality = _read_json(meta_dir / "modality.json")
    episodes_meta = _read_jsonl(meta_dir / "episodes.jsonl")
    video_map = _read_json(meta_dir / "video_map.json")
    split_manifest = _read_json(split_manifest_path)
    split_name, split_records = _validate_split_manifest(split_manifest)

    total_videos = _as_int(
        info.get("total_videos"), context="meta/info.json total_videos"
    )
    if total_videos <= 0:
        raise RuntimeError(
            "video_decode_missing: dataset build requires with-video dataset (total_videos>0)"
        )

    features_raw = info.get("features")
    if not isinstance(features_raw, dict):
        raise ValueError(
            "dataset_build_invalid: meta/info.json features must be an object"
        )
    features = cast(Mapping[str, object], features_raw)
    modality_video_raw = modality.get("video")
    if not isinstance(modality_video_raw, dict) or not modality_video_raw:
        raise RuntimeError(
            "video_decode_missing: meta/modality.json missing non-empty video mapping"
        )
    video_keys_available = sorted(str(k) for k in modality_video_raw.keys())
    sample_mode = _honest_sample_mode(video_keys_available)
    primary_video_key = str(video_keys_available[0])
    primary_original_key = _as_str(
        cast(Mapping[str, object], modality_video_raw[primary_video_key]).get(
            "original_key"
        ),
        context=f"modality.video[{primary_video_key}] original_key",
    )

    data_path_template = _as_str(
        info.get("data_path"), context="meta/info.json data_path"
    )
    chunks_size = _as_int(info.get("chunks_size"), context="meta/info.json chunks_size")
    episode_meta_by_index: dict[int, JsonObject] = {}
    for ep_meta in episodes_meta:
        episode_index = _as_int(
            ep_meta.get("episode_index"), context="episodes episode_index"
        )
        if episode_index in episode_meta_by_index:
            raise ValueError(
                f"dataset_build_invalid: duplicate episode_index in meta/episodes.jsonl: {episode_index}"
            )
        episode_meta_by_index[episode_index] = ep_meta

    video_map_records = _json_list_of_dicts(
        video_map.get("records"), context="video_map.records"
    )
    video_map_by_episode: dict[int, JsonObject] = {}
    for rec in video_map_records:
        episode_index = _as_int(
            rec.get("episode_index"), context="video_map.records[*].episode_index"
        )
        if episode_index in video_map_by_episode:
            raise ValueError(
                f"dataset_build_invalid: duplicate video_map episode_index={episode_index}"
            )
        video_map_by_episode[episode_index] = rec

    alignment_report: dict[str, object] = {
        "schema_version": "vlm_critic_alignment_report_v1",
        "dataset_path": str(dataset_path),
        "split_manifest_path": str(split_manifest_path),
        "split_name": str(split_name),
        "sample_mode": str(sample_mode),
        "available_video_keys": video_keys_available,
        "primary_video_key": str(primary_video_key),
        "allow_future_frames": False,
        "frame_policy": {
            "index_column": "index",
            "step_column": "recap_m2.t",
            "sample_identity_mode": str(sample_identity_mode),
            "selected_frame_must_match_row_index": False,
            "representative_frame_selection": "resolved_per_sample_identity_mode",
            "allow_future_frames": False,
        },
        "checked_episodes": 0,
        "checked_samples": 0,
        "checked_rows": 0,
        "decode_fail_count": 0,
        "decode_probe_count": 0,
        "future_frame_violation_count": 0,
        "episode_leakage_count": 0,
        "frame_t_mismatch_count": 0,
        "violations_preview": [],
        "decode_backends": {},
        "frame_count_sources": [],
        "decode_probe_policy": "frame_count_or_boundary_midpoint_row_probes",
        "pass": False,
    }

    side_channel_status = _build_side_channel_status(features)
    decode_backends: Counter[str] = Counter()
    frame_count_sources: set[str] = set()
    violations_preview: list[str] = []
    decode_fail_count = 0
    decode_probe_count = 0
    future_frame_violation_count = 0
    episode_leakage_count = 0
    frame_t_mismatch_count = 0
    sample_records: list[JsonObject] = []
    return_values: list[float] = []
    identity_modes_seen: set[str] = set()

    required_columns = [
        "episode_index",
        "index",
        "recap_m2.t",
        "recap_m2.prompt_raw",
        "recap_m2.return_G",
    ]

    for split_rec in split_records:
        episode_index = _as_int(
            split_rec.get("episode_index"), context="split.records[*].episode_index"
        )
        split_episode_id = _as_str(
            split_rec.get("recap_episode_id"),
            context=f"split.records[{episode_index}] recap_episode_id",
        )
        split_episode_length = _as_int(
            split_rec.get("episode_length"),
            context=f"split.records[{episode_index}] episode_length",
        )
        split_parquet_rel = _as_str(
            split_rec.get("parquet_rel"),
            context=f"split.records[{episode_index}] parquet_rel",
        )
        split_video_rel = _as_str(
            split_rec.get("video_rel"),
            context=f"split.records[{episode_index}] video_rel",
        )

        ep_meta = episode_meta_by_index.get(episode_index)
        if ep_meta is None:
            episode_leakage_count += 1
            violations_preview.append(
                f"episode_leakage_detected: split references missing episode_index={episode_index}"
            )
            _raise_blocker(
                f"episode_leakage_detected: split references missing episode_index={episode_index}",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )
        ep_meta_obj = cast(JsonObject, ep_meta)
        dataset_episode_id = _as_str(
            ep_meta_obj.get("recap.episode_id"),
            context=f"meta episode_index={episode_index} recap.episode_id",
        )
        if dataset_episode_id != split_episode_id:
            episode_leakage_count += 1
            violations_preview.append(
                "episode_leakage_detected: "
                f"episode_index={episode_index} split_episode_id={split_episode_id!r} dataset_episode_id={dataset_episode_id!r}"
            )
            _raise_blocker(
                "episode_leakage_detected: split manifest recap_episode_id mismatches dataset metadata",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )

        episode_length = _as_int(
            ep_meta_obj.get("length"),
            context=f"meta episode_index={episode_index} length",
        )
        if episode_length != split_episode_length:
            frame_t_mismatch_count += 1
            violations_preview.append(
                "frame_t_alignment_mismatch: "
                f"episode_index={episode_index} split_episode_length={split_episode_length} dataset_episode_length={episode_length}"
            )
            _raise_blocker(
                "frame_t_alignment_mismatch: split manifest episode_length mismatches dataset metadata",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )

        expected_chunk = int(episode_index) // int(chunks_size)
        expected_parquet_rel = Path(
            data_path_template.format(
                episode_chunk=int(expected_chunk),
                episode_index=int(episode_index),
            )
        ).as_posix()
        if Path(split_parquet_rel).as_posix() != expected_parquet_rel:
            frame_t_mismatch_count += 1
            violations_preview.append(
                "frame_t_alignment_mismatch: "
                f"episode_index={episode_index} split_parquet_rel={split_parquet_rel!r} expected_parquet_rel={expected_parquet_rel!r}"
            )
            _raise_blocker(
                "frame_t_alignment_mismatch: split manifest parquet_rel mismatches dataset template",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )

        video_map_rec = video_map_by_episode.get(episode_index)
        if video_map_rec is None:
            episode_leakage_count += 1
            violations_preview.append(
                f"episode_leakage_detected: missing video_map record for episode_index={episode_index}"
            )
            _raise_blocker(
                f"episode_leakage_detected: missing video_map record for episode_index={episode_index}",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )
        video_map_rec_obj = cast(JsonObject, video_map_rec)
        expected_video_rel = _as_str(
            video_map_rec_obj.get("dst_mp4"),
            context=f"video_map[{episode_index}] dst_mp4",
        )
        if Path(split_video_rel).as_posix() != Path(expected_video_rel).as_posix():
            frame_t_mismatch_count += 1
            violations_preview.append(
                "frame_t_alignment_mismatch: "
                f"episode_index={episode_index} split_video_rel={split_video_rel!r} expected_video_rel={expected_video_rel!r}"
            )
            _raise_blocker(
                "frame_t_alignment_mismatch: split manifest video_rel mismatches dataset video_map",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )

        parquet_abs = dataset_path / Path(expected_parquet_rel)
        if not parquet_abs.is_file():
            raise FileNotFoundError(
                f"frame_t_alignment_mismatch: missing parquet for episode_index={episode_index}: {parquet_abs}"
            )
        video_abs = dataset_path / Path(expected_video_rel)
        if not video_abs.is_file():
            raise FileNotFoundError(
                f"video_decode_missing: missing video for episode_index={episode_index}: {video_abs}"
            )

        frame_count, frame_count_source = _resolve_num_frames(video_abs)
        if frame_count_source:
            frame_count_sources.add(str(frame_count_source))
        table = _parquet_read_table(parquet_abs, columns=required_columns)
        column_names = set(str(name) for name in getattr(table, "column_names", []))
        missing_columns = [col for col in required_columns if col not in column_names]
        if missing_columns:
            raise ValueError(
                "frame_t_alignment_mismatch: missing required parquet columns "
                f"for episode_index={episode_index}: {missing_columns}"
            )

        episode_indexes = _table_column_pylist(table, "episode_index")
        frame_indexes = _table_column_pylist(table, "index")
        t_values = _table_column_pylist(table, "recap_m2.t")
        prompts = _table_column_pylist(table, "recap_m2.prompt_raw")
        return_g_values = _table_column_pylist(table, "recap_m2.return_G")

        row_count = len(frame_indexes)
        if row_count <= 0:
            frame_t_mismatch_count += 1
            violations_preview.append(
                "frame_t_alignment_mismatch: "
                f"episode_index={episode_index} parquet_rows={row_count}"
            )
            _raise_blocker(
                "frame_t_alignment_mismatch: parquet must contain at least one row",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )

        max_source_index = max(
            _as_int(
                raw_value,
                context=f"episode_index={episode_index} parquet.index_source[{local_idx}]",
            )
            for local_idx, raw_value in enumerate(frame_indexes)
        )
        source_index_horizon = max(int(row_count), int(max_source_index + 1))

        effective_identity_mode = _resolve_sample_identity_mode(
            requested_mode=sample_identity_mode,
            row_count=int(row_count),
            episode_length=int(episode_length),
        )
        identity_modes_seen.add(str(effective_identity_mode))
        if len(identity_modes_seen) > 1:
            frame_t_mismatch_count += 1
            violations_preview.append(
                "frame_t_alignment_mismatch: "
                f"split contains mixed sample_identity_mode values={sorted(identity_modes_seen)}"
            )
            _raise_blocker(
                "frame_t_alignment_mismatch: split must resolve to a single sample_identity_mode",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )

        if (
            effective_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
            and row_count != episode_length
        ):
            frame_t_mismatch_count += 1
            violations_preview.append(
                "frame_t_alignment_mismatch: "
                f"episode_index={episode_index} parquet_rows={row_count} episode_length={episode_length} sample_identity_mode={effective_identity_mode}"
            )
            _raise_blocker(
                "frame_t_alignment_mismatch: fixture sample identity requires parquet row count == episode_length",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )

        alignment_report["sample_identity_mode"] = str(effective_identity_mode)
        alignment_report["frame_policy"] = {
            "index_column": "index",
            "step_column": "recap_m2.t",
            "sample_identity_mode": str(effective_identity_mode),
            "selected_frame_must_match_row_index": bool(
                effective_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
            ),
            "representative_frame_selection": (
                "row_local_index_equals_frame_index"
                if effective_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
                else "earliest_frame_index_within_policy_step_group"
            ),
            "allow_future_frames": False,
        }

        seen_t: set[int] = set()
        last_t: int | None = None
        max_frame_index_seen = -1
        current_group_t: int | None = None
        current_group_rows: list[JsonObject] = []
        sample_count_this_episode = 0
        previous_selected_frame_index: int | None = None
        episode_sample_start = len(sample_records)

        def _flush_policy_step_group() -> None:
            nonlocal current_group_t
            nonlocal current_group_rows
            nonlocal max_frame_index_seen
            nonlocal sample_count_this_episode
            nonlocal previous_selected_frame_index

            if current_group_t is None:
                if current_group_rows:
                    raise AssertionError(
                        "policy_step_group rows without current_group_t"
                    )
                return
            if not current_group_rows:
                raise AssertionError("policy_step_group flush without rows")

            prompt0 = _as_str(
                current_group_rows[0].get("prompt_raw"),
                context="policy_step_group.prompt_raw[0]",
            )
            return0 = _as_float(
                current_group_rows[0].get("return_G"),
                context="policy_step_group.return_G[0]",
            )
            for row in current_group_rows[1:]:
                prompt_i = _as_str(
                    row.get("prompt_raw"), context="policy_step_group.prompt_raw"
                )
                if prompt_i != prompt0:
                    raise RuntimeError(
                        "frame_t_alignment_mismatch: prompt_raw must stay constant within a policy-step group"
                    )
            if not _return_group_consistent(current_group_rows, expected=return0):
                raise RuntimeError(
                    "frame_t_alignment_mismatch: return_G must stay constant within a policy-step group"
                )

            selected_row = min(
                current_group_rows,
                key=lambda row: (
                    _as_int(
                        row.get("frame_source_index"),
                        context="policy_step_group.frame_source_index",
                    ),
                    _as_int(
                        row.get("local_index"), context="policy_step_group.local_index"
                    ),
                ),
            )
            selected_local_index = _as_int(
                selected_row.get("local_index"),
                context="policy_step_group.selected.local_index",
            )
            selected_source_index = _as_int(
                selected_row.get("frame_source_index"),
                context="policy_step_group.selected.frame_source_index",
            )
            selected_frame_index = _source_index_to_frame_index(
                source_index=int(selected_source_index),
                source_horizon=int(source_index_horizon),
                frame_count=frame_count,
            )
            min_group_source_index = min(
                _as_int(
                    row.get("frame_source_index"),
                    context="policy_step_group.frame_source_index",
                )
                for row in current_group_rows
            )
            max_group_source_index = max(
                _as_int(
                    row.get("frame_source_index"),
                    context="policy_step_group.frame_source_index",
                )
                for row in current_group_rows
            )
            min_group_row_index = min(
                _as_int(row.get("local_index"), context="policy_step_group.local_index")
                for row in current_group_rows
            )
            max_group_row_index = max(
                _as_int(row.get("local_index"), context="policy_step_group.local_index")
                for row in current_group_rows
            )

            if previous_selected_frame_index is not None and (
                selected_frame_index < previous_selected_frame_index
            ):
                raise RuntimeError(
                    "frame_t_alignment_mismatch: representative frame_index must be monotonic non-decreasing across policy steps"
                )
            previous_selected_frame_index = int(selected_frame_index)
            max_frame_index_seen = max(
                int(max_frame_index_seen), int(selected_frame_index)
            )
            sample_count_this_episode += 1

            sample_id = (
                f"{split_name}:{split_episode_id}:i{int(selected_frame_index):06d}:t{int(current_group_t):06d}"
                if effective_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
                else f"{split_name}:{split_episode_id}:t{int(current_group_t):06d}"
            )
            sample_records.append(
                {
                    "sample_id": sample_id,
                    "split_name": str(split_name),
                    "episode_index": int(episode_index),
                    "recap_episode_id": str(split_episode_id),
                    "episode_length": int(episode_length),
                    "local_index": int(selected_local_index),
                    "frame_index": int(selected_frame_index),
                    "t": int(current_group_t),
                    "t_norm": 0.0,
                    "sample_identity": {
                        "mode": str(effective_identity_mode),
                        "episode_index": int(episode_index),
                        "t": int(current_group_t),
                        "representative_row_index": int(selected_local_index),
                        "representative_source_index": int(selected_source_index),
                        "representative_frame_index": int(selected_frame_index),
                        "row_selection_policy": (
                            "local_index_equals_index_equals_t"
                            if effective_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
                            else "earliest_frame_index_within_policy_step_group"
                        ),
                        "policy_step_row_count": int(len(current_group_rows)),
                        "row_index_min": int(min_group_row_index),
                        "row_index_max": int(max_group_row_index),
                        "source_index_min": int(min_group_source_index),
                        "source_index_max": int(max_group_source_index),
                        "source_index_horizon": int(source_index_horizon),
                        "episode_length_semantics": "frame_video_horizon",
                        "policy_step_horizon_semantics": "derived_from_recap_m2_t_max_plus_one",
                    },
                    "prompt_raw": str(prompt0),
                    "return_G": float(return0),
                    "video": {
                        "video_key": str(primary_video_key),
                        "original_key": str(primary_original_key),
                        "video_rel": str(expected_video_rel),
                        "frame_policy": (
                            "current_row_frame_index"
                            if effective_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
                            else "earliest_frame_index_within_policy_step_group"
                        ),
                        "source_row_index": int(selected_local_index),
                        "frame_source_index": int(selected_source_index),
                        "frame_source_horizon": int(source_index_horizon),
                        "decode_backend": "validated_by_episode_decode_probes",
                    },
                    "target": {
                        "target_mode": "dist201_raw_return",
                        "return_G": float(return0),
                    },
                }
            )
            return_values.append(float(return0))
            current_group_t = None
            current_group_rows = []

        for local_index, (
            row_episode_index,
            frame_index_raw,
            t_raw,
            prompt_raw_value,
            return_g_raw,
        ) in enumerate(
            zip(
                episode_indexes,
                frame_indexes,
                t_values,
                prompts,
                return_g_values,
                strict=True,
            )
        ):
            episode_index_in_row = _as_int(
                row_episode_index,
                context=f"episode_index={episode_index} parquet.episode_index[{local_index}]",
            )
            if episode_index_in_row != episode_index:
                episode_leakage_count += 1
                violations_preview.append(
                    "episode_leakage_detected: "
                    f"episode_index={episode_index} local_index={local_index} row_episode_index={episode_index_in_row}"
                )
                _raise_blocker(
                    "episode_leakage_detected: parquet row episode_index mismatches split episode",
                    report=alignment_report,
                    violations_preview=violations_preview,
                    decode_fail_count=decode_fail_count,
                    future_frame_violation_count=future_frame_violation_count,
                    episode_leakage_count=episode_leakage_count,
                    frame_t_mismatch_count=frame_t_mismatch_count,
                )

            frame_index = _as_int(
                frame_index_raw,
                context=f"episode_index={episode_index} parquet.index[{local_index}]",
            )
            t_i = _as_int(
                t_raw,
                context=f"episode_index={episode_index} recap_m2.t[{local_index}]",
            )
            prompt_raw = _as_str(
                prompt_raw_value,
                context=f"episode_index={episode_index} recap_m2.prompt_raw[{local_index}]",
            )
            return_g = _as_float(
                return_g_raw,
                context=f"episode_index={episode_index} recap_m2.return_G[{local_index}]",
            )
            if not math.isfinite(return_g):
                raise RuntimeError(
                    f"frame_t_alignment_mismatch: recap_m2.return_G must be finite, got {return_g}"
                )
            if frame_index < 0 or t_i < 0:
                frame_t_mismatch_count += 1
                violations_preview.append(
                    "frame_t_alignment_mismatch: "
                    f"episode_index={episode_index} local_index={local_index} frame_index={frame_index} t={t_i}"
                )
                _raise_blocker(
                    "frame_t_alignment_mismatch: negative frame_index or recap_m2.t is forbidden",
                    report=alignment_report,
                    violations_preview=violations_preview,
                    decode_fail_count=decode_fail_count,
                    future_frame_violation_count=future_frame_violation_count,
                    episode_leakage_count=episode_leakage_count,
                    frame_t_mismatch_count=frame_t_mismatch_count,
                )
            if effective_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE:
                if frame_index > local_index:
                    future_frame_violation_count += 1
                    violations_preview.append(
                        "frame_t_alignment_mismatch: "
                        f"episode_index={episode_index} local_index={local_index} frame_index={frame_index}"
                    )
                    _raise_blocker(
                        "frame_t_alignment_mismatch: future frame access detected (frame_index > row_local_index)",
                        report=alignment_report,
                        violations_preview=violations_preview,
                        decode_fail_count=decode_fail_count,
                        future_frame_violation_count=future_frame_violation_count,
                        episode_leakage_count=episode_leakage_count,
                        frame_t_mismatch_count=frame_t_mismatch_count,
                    )
                if frame_index != local_index:
                    frame_t_mismatch_count += 1
                    violations_preview.append(
                        "frame_t_alignment_mismatch: "
                        f"episode_index={episode_index} local_index={local_index} frame_index={frame_index} t={t_i}"
                    )
                    _raise_blocker(
                        "frame_t_alignment_mismatch: current-row frame policy requires local_index == index",
                        report=alignment_report,
                        violations_preview=violations_preview,
                        decode_fail_count=decode_fail_count,
                        future_frame_violation_count=future_frame_violation_count,
                        episode_leakage_count=episode_leakage_count,
                        frame_t_mismatch_count=frame_t_mismatch_count,
                    )
            if t_i >= episode_length:
                frame_t_mismatch_count += 1
                violations_preview.append(
                    "frame_t_alignment_mismatch: "
                    f"episode_index={episode_index} local_index={local_index} t={t_i} episode_length={episode_length}"
                )
                _raise_blocker(
                    "frame_t_alignment_mismatch: recap_m2.t exceeds episode_length horizon",
                    report=alignment_report,
                    violations_preview=violations_preview,
                    decode_fail_count=decode_fail_count,
                    future_frame_violation_count=future_frame_violation_count,
                    episode_leakage_count=episode_leakage_count,
                    frame_t_mismatch_count=frame_t_mismatch_count,
                )

            if last_t is not None and t_i < last_t:
                frame_t_mismatch_count += 1
                violations_preview.append(
                    "frame_t_alignment_mismatch: "
                    f"episode_index={episode_index} local_index={local_index} prev_t={last_t} t={t_i}"
                )
                _raise_blocker(
                    "frame_t_alignment_mismatch: recap_m2.t must be monotonic non-decreasing across parquet rows",
                    report=alignment_report,
                    violations_preview=violations_preview,
                    decode_fail_count=decode_fail_count,
                    future_frame_violation_count=future_frame_violation_count,
                    episode_leakage_count=episode_leakage_count,
                    frame_t_mismatch_count=frame_t_mismatch_count,
                )
            if last_t is not None and t_i > last_t + 1:
                frame_t_mismatch_count += 1
                violations_preview.append(
                    "frame_t_alignment_mismatch: "
                    f"episode_index={episode_index} local_index={local_index} prev_t={last_t} t={t_i}"
                )
                _raise_blocker(
                    "frame_t_alignment_mismatch: recap_m2.t cannot skip unseen policy steps",
                    report=alignment_report,
                    violations_preview=violations_preview,
                    decode_fail_count=decode_fail_count,
                    future_frame_violation_count=future_frame_violation_count,
                    episode_leakage_count=episode_leakage_count,
                    frame_t_mismatch_count=frame_t_mismatch_count,
                )
            if t_i not in seen_t and t_i != len(seen_t):
                frame_t_mismatch_count += 1
                violations_preview.append(
                    "frame_t_alignment_mismatch: "
                    f"episode_index={episode_index} local_index={local_index} first_seen_t={t_i} expected_first_seen_t={len(seen_t)}"
                )
                _raise_blocker(
                    "frame_t_alignment_mismatch: first occurrence of recap_m2.t must cover policy steps contiguously from zero",
                    report=alignment_report,
                    violations_preview=violations_preview,
                    decode_fail_count=decode_fail_count,
                    future_frame_violation_count=future_frame_violation_count,
                    episode_leakage_count=episode_leakage_count,
                    frame_t_mismatch_count=frame_t_mismatch_count,
                )
            if (
                effective_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
                and t_i != local_index
            ):
                frame_t_mismatch_count += 1
                violations_preview.append(
                    "frame_t_alignment_mismatch: "
                    f"episode_index={episode_index} local_index={local_index} frame_index={frame_index} t={t_i} sample_identity_mode={effective_identity_mode}"
                )
                _raise_blocker(
                    "frame_t_alignment_mismatch: fixture sample identity requires local_index == index == recap_m2.t",
                    report=alignment_report,
                    violations_preview=violations_preview,
                    decode_fail_count=decode_fail_count,
                    future_frame_violation_count=future_frame_violation_count,
                    episode_leakage_count=episode_leakage_count,
                    frame_t_mismatch_count=frame_t_mismatch_count,
                )

            seen_t.add(int(t_i))
            last_t = int(t_i)
            if current_group_t is None:
                current_group_t = int(t_i)
            elif int(t_i) != int(current_group_t):
                _flush_policy_step_group()
                current_group_t = int(t_i)
            current_group_rows.append(
                {
                    "local_index": int(local_index),
                    "frame_source_index": int(frame_index),
                    "prompt_raw": str(prompt_raw),
                    "return_G": float(return_g),
                }
            )

        _flush_policy_step_group()

        if last_t is None:
            frame_t_mismatch_count += 1
            violations_preview.append(
                f"frame_t_alignment_mismatch: episode_index={episode_index} missing recap_m2.t rows"
            )
            _raise_blocker(
                "frame_t_alignment_mismatch: episode parquet must contain at least one recap_m2.t value",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )

        assert last_t is not None
        final_t = int(last_t)
        policy_step_count = int(final_t + 1)
        if len(seen_t) != policy_step_count:
            frame_t_mismatch_count += 1
            violations_preview.append(
                "frame_t_alignment_mismatch: "
                f"episode_index={episode_index} seen_policy_steps={len(seen_t)} expected_policy_steps={policy_step_count}"
            )
            _raise_blocker(
                "frame_t_alignment_mismatch: parquet rows do not cover every policy step implied by recap_m2.t",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )

        if policy_step_count > episode_length:
            frame_t_mismatch_count += 1
            violations_preview.append(
                "frame_t_alignment_mismatch: "
                f"episode_index={episode_index} policy_step_count={policy_step_count} episode_length={episode_length}"
            )
            _raise_blocker(
                "frame_t_alignment_mismatch: derived policy-step horizon cannot exceed frame/video horizon",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )

        t_norm_den = max(1, int(policy_step_count - 1))
        for sample in sample_records[episode_sample_start:]:
            sample_t = _as_int(sample.get("t"), context="sample.t")
            sample["t_norm"] = float(float(sample_t) / float(t_norm_den))
            sample_identity = cast(dict[str, object], sample.get("sample_identity", {}))
            sample_identity["policy_step_count"] = int(policy_step_count)
            sample_identity["episode_t_max"] = int(final_t)
            sample_identity["episode_frame_length"] = int(episode_length)

        if frame_count is not None and max_frame_index_seen >= frame_count:
            decode_fail_count += 1
            violations_preview.append(
                "video_decode_missing: "
                f"episode_index={episode_index} max_frame_index={max_frame_index_seen} frame_count={frame_count}"
            )
            _raise_blocker(
                "video_decode_missing: current-row frame_index exceeds decoded video length",
                report=alignment_report,
                violations_preview=violations_preview,
                decode_fail_count=decode_fail_count,
                future_frame_violation_count=future_frame_violation_count,
                episode_leakage_count=episode_leakage_count,
                frame_t_mismatch_count=frame_t_mismatch_count,
            )

        for probe_frame_index in _decode_probe_indices(max_frame_index_seen):
            backend = ""
            try:
                backend = _probe_frame_decode(video_abs, int(probe_frame_index))
            except RuntimeError as exc:
                decode_fail_count += 1
                violations_preview.append(
                    f"video_decode_missing: episode_index={episode_index} probe_frame_index={probe_frame_index} {exc}"
                )
                _raise_blocker(
                    str(exc),
                    report=alignment_report,
                    violations_preview=violations_preview,
                    decode_fail_count=decode_fail_count,
                    future_frame_violation_count=future_frame_violation_count,
                    episode_leakage_count=episode_leakage_count,
                    frame_t_mismatch_count=frame_t_mismatch_count,
                )
            decode_backends[str(backend)] += 1
            decode_probe_count += 1

        alignment_report["checked_episodes"] = int(
            _as_int(
                alignment_report.get("checked_episodes", 0), context="checked_episodes"
            )
            + 1
        )
        alignment_report["checked_samples"] = int(
            _as_int(
                alignment_report.get("checked_samples", 0), context="checked_samples"
            )
            + sample_count_this_episode
        )
        alignment_report["checked_rows"] = int(
            _as_int(alignment_report.get("checked_rows", 0), context="checked_rows")
            + row_count
        )

    if not sample_records:
        raise RuntimeError("frame_t_alignment_mismatch: build produced zero samples")

    if len(identity_modes_seen) != 1:
        raise RuntimeError(
            "frame_t_alignment_mismatch: build must resolve exactly one sample_identity_mode"
        )
    final_sample_identity_mode = next(iter(identity_modes_seen))

    g_min = float(min(return_values))
    g_max = float(max(return_values))
    degenerate_range = False
    if g_min == g_max:
        degenerate_range = True
        g_min = float(g_min) - 1e-3
        g_max = float(g_max) + 1e-3
    bin_centers = _linspace(g_min, g_max, bins)
    for sample in sample_records:
        target = cast(dict[str, object], sample["target"])
        return_g = _as_float(target.get("return_G"), context="target.return_G")
        target_bin_index = _to_bin_index(return_g, g_min=g_min, g_max=g_max, bins=bins)
        target["bin_count"] = int(bins)
        target["bin_range"] = {"g_min": float(g_min), "g_max": float(g_max)}
        target["target_bin_index"] = int(target_bin_index)

    alignment_report["decode_fail_count"] = int(decode_fail_count)
    alignment_report["decode_probe_count"] = int(decode_probe_count)
    alignment_report["future_frame_violation_count"] = int(future_frame_violation_count)
    alignment_report["episode_leakage_count"] = int(episode_leakage_count)
    alignment_report["frame_t_mismatch_count"] = int(frame_t_mismatch_count)
    alignment_report["violations_preview"] = list(violations_preview[:16])
    alignment_report["decode_backends"] = dict(sorted(decode_backends.items()))
    alignment_report["frame_count_sources"] = sorted(frame_count_sources)
    alignment_report["pass"] = True

    build_payload: JsonObject = {
        "schema_version": "vlm_critic_dataset_build_v1",
        "builder_entrypoint": "work/recap/scripts/42_vlm_critic_dataset_build.py",
        "dataset_path": str(dataset_path),
        "split_manifest_path": str(split_manifest_path),
        "split_name": str(split_name),
        "sample_mode": str(sample_mode),
        "allow_future_frames": False,
        "task_text_field": "prompt_raw",
        "frame_index_field": "derived_current_step_frame_index",
        "frame_source_index_field": "index",
        "step_field": "recap_m2.t",
        "episode_length_semantics": "frame_video_horizon_from_meta_episodes_jsonl_length",
        "policy_step_horizon_semantics": "derived_per_episode_from_recap_m2_t_max_plus_one",
        "target_mode": "dist201_raw_return",
        "sample_identity_mode": str(final_sample_identity_mode),
        "sample_identity_fields": (
            ["episode_index", "index", "recap_m2.t"]
            if final_sample_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
            else ["episode_index", "recap_m2.t"]
        ),
        "side_channels": ["proprio", "t_norm"],
        "side_channel_status": side_channel_status,
        "video_keys_available": video_keys_available,
        "primary_video_key": str(primary_video_key),
        "primary_video_original_key": str(primary_original_key),
        "formal_eval_scope": _as_str(
            split_manifest.get("formal_eval_scope", "isaac_only"),
            context="split_manifest.formal_eval_scope",
        ),
        "split_granularity": _as_str(
            split_manifest.get("split_granularity"),
            context="split_manifest.split_granularity",
        ),
        "sample_count": int(len(sample_records)),
        "episode_count": int(len(split_records)),
        "bin_spec": {
            "bin_count": int(bins),
            "g_min": float(g_min),
            "g_max": float(g_max),
            "degenerate_range_adjusted": bool(degenerate_range),
            "bin_centers": [float(x) for x in bin_centers],
            "index_policy": "nearest_center_round",
        },
        "alignment_report_path": str(_derive_alignment_report_path(output_json)),
        "current_step_policy": {
            "policy_name": (
                "current_row_frame_index"
                if final_sample_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
                else "earliest_frame_index_within_policy_step_group"
            ),
            "sample_identity_mode": str(final_sample_identity_mode),
            "require_local_index_match": bool(
                final_sample_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
            ),
            "require_step_match": bool(
                final_sample_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
            ),
            "step_alignment_semantics": (
                "frame_row_carries_current_policy_step_label"
                if final_sample_identity_mode == SAMPLE_IDENTITY_MODE_FIXTURE
                else "sample identity anchored on policy step t; index used only to select an earliest no-future representative frame within each t group"
            ),
            "allow_future_frames": False,
        },
        "samples": sample_records,
    }

    ablations: dict[str, JsonObject] = {}
    for name, ablation_path in _derive_ablation_paths(output_json).items():
        if name == "full_input":
            mode_payload = _input_mode_payload(
                sample_mode, use_prompt=True, use_video=True
            )
        elif name == "vision_only":
            mode_payload = _input_mode_payload(
                sample_mode, use_prompt=False, use_video=True
            )
        elif name == "prompt_only":
            mode_payload = _input_mode_payload(
                sample_mode, use_prompt=True, use_video=False
            )
        else:
            raise AssertionError(f"Unexpected ablation name: {name}")
        ablations[name] = {
            "schema_version": "vlm_critic_ablation_manifest_v1",
            "source_build_json": str(output_json),
            "ablation_name": str(name),
            "dataset_path": str(dataset_path),
            "split_manifest_path": str(split_manifest_path),
            "sample_count": int(len(sample_records)),
            "input_mode": mode_payload,
            "sample_ids": [str(sample["sample_id"]) for sample in sample_records],
            "alignment_report_path": str(_derive_alignment_report_path(output_json)),
        }
        _write_json(ablation_path, ablations[name])

    build_payload["ablation_manifests"] = {
        name: str(path) for name, path in _derive_ablation_paths(output_json).items()
    }
    return build_payload, cast(JsonObject, alignment_report), ablations


def _emit_result(
    *, sentinel: str, output_json: Path | None, payload: Mapping[str, object]
) -> None:
    if output_json is not None:
        _write_json(output_json, payload)
        print(f"[INFO] wrote_json: {output_json}")
    print(f"SENTINEL:{sentinel}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="42_vlm_critic_dataset_build.py",
        description=(
            "Build Task 4 with-video critic sample manifests plus a frame/t alignment report."
        ),
    )
    _ = parser.add_argument("--dataset-path", type=str, required=True)
    _ = parser.add_argument("--split-manifest", type=str, required=True)
    _ = parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON_REL)
    _ = parser.add_argument("--bins", type=int, default=int(DEFAULT_BINS))
    _ = parser.add_argument(
        "--sample-identity-mode",
        type=str,
        default=DEFAULT_SAMPLE_IDENTITY_MODE,
        choices=[
            SAMPLE_IDENTITY_MODE_AUTO,
            SAMPLE_IDENTITY_MODE_FIXTURE,
            SAMPLE_IDENTITY_MODE_ROW_FRAME_T,
        ],
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    dataset_path = _resolve_path(repo_root, str(args.dataset_path), default_rel="")
    split_manifest_path = _resolve_path(
        repo_root, str(args.split_manifest), default_rel=""
    )
    output_json = _resolve_path(
        repo_root, str(args.output_json), default_rel=DEFAULT_OUTPUT_JSON_REL
    )
    alignment_report_path = _derive_alignment_report_path(output_json)

    try:
        build_payload, alignment_report, _ = _build_dataset_manifest(
            dataset_path=dataset_path,
            split_manifest_path=split_manifest_path,
            output_json=output_json,
            bins=int(args.bins),
            sample_identity_mode=str(args.sample_identity_mode),
        )
        _write_json(alignment_report_path, alignment_report)
        print(f"[INFO] wrote_json: {alignment_report_path}")
        _emit_result(
            sentinel=PASS_SENTINEL, output_json=output_json, payload=build_payload
        )
        return 0
    except BuildBlocker as exc:
        report = dict(exc.alignment_report)
        report["dataset_path"] = str(dataset_path)
        report["split_manifest_path"] = str(split_manifest_path)
        _write_json(alignment_report_path, report)
        print(f"[INFO] wrote_json: {alignment_report_path}")
        failure: JsonObject = {
            "pass": False,
            "dataset_path": str(dataset_path),
            "split_manifest_path": str(split_manifest_path),
            "output_json": str(output_json),
            "alignment_report_path": str(alignment_report_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _emit_result(sentinel=FAIL_SENTINEL, output_json=output_json, payload=failure)
        return 1
    except Exception as exc:
        fallback_report: JsonObject = {
            "schema_version": "vlm_critic_alignment_report_v1",
            "dataset_path": str(dataset_path),
            "split_manifest_path": str(split_manifest_path),
            "decode_fail_count": 0,
            "future_frame_violation_count": 0,
            "episode_leakage_count": 0,
            "frame_t_mismatch_count": 0,
            "pass": False,
            "blocker": f"{type(exc).__name__}: {exc}",
        }
        _write_json(alignment_report_path, fallback_report)
        print(f"[INFO] wrote_json: {alignment_report_path}")
        failure = {
            "pass": False,
            "dataset_path": str(dataset_path),
            "split_manifest_path": str(split_manifest_path),
            "output_json": str(output_json),
            "alignment_report_path": str(alignment_report_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _emit_result(sentinel=FAIL_SENTINEL, output_json=output_json, payload=failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
