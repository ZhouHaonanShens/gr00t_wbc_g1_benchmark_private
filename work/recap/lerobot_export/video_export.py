from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from work.recap.lerobot_export import dataset_export
from work.recap.lerobot_export.contract_io import as_int as _as_int
from work.recap.lerobot_export.contract_io import read_json_object as _read_json
from work.recap.lerobot_export.contract_io import (
    read_jsonl_objects as _read_jsonl_dicts,
)
from work.recap.lerobot_export.contract_io import write_json_object as _write_json
from work.recap.lerobot_export.contract_io import (
    write_jsonl_objects as _write_jsonl_dicts,
)


@dataclass(frozen=True)
class LeRobotV2ExportWithVideoResult:
    output_dataset_dir: Path
    total_episodes: int
    total_videos: int
    video_path_template: str
    image_key: str
    original_key: str
    video_map_path: Path


DEFAULT_IMAGE_KEY = "ego_view"
DEFAULT_ORIGINAL_KEY = "observation.images.ego_view"
DEFAULT_VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/observation.images.ego_view/episode_{episode_index:06d}.mp4"


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


_EPISODE_MP4_NAME_RE = re.compile(r"^(?P<episode_id>.+)_s(?P<stream_idx>\d+)\.mp4$")


def _preview_ids(ids: list[str], *, limit: int = 12) -> str:
    if not ids:
        return "<empty>"
    uniq = sorted(set([str(x) for x in ids]))
    head = uniq[: int(limit)]
    if len(uniq) <= int(limit):
        return ", ".join(head)
    return ", ".join(head) + f", ... (+{len(uniq) - int(limit)} more)"


def _episode_id_from_mp4_name(mp4_path: Path) -> str | None:
    m = _EPISODE_MP4_NAME_RE.match(mp4_path.name)
    if not m:
        return None
    episode_id = m.group("episode_id")
    if not episode_id:
        return None
    return str(episode_id)


def resolve_episode_video_path(
    *, video_dir_archived: str | Path, episode_id: str
) -> Path:
    video_dir = Path(video_dir_archived).expanduser().resolve()
    if not video_dir.is_dir():
        raise FileNotFoundError(f"video_dir_archived is not a directory: {video_dir}")
    ordered, _summary = _pair_root_mp4s_to_episode_ids(
        video_dir_archived=video_dir,
        episode_ids_in_order=[str(episode_id)],
    )
    if len(ordered) != 1:
        raise RuntimeError(
            "Expected exactly one video for source episode: "
            f"episode_id={episode_id} video_dir_archived={video_dir} matched={len(ordered)}"
        )
    return ordered[0].resolve()


def resolve_video_frame_count(video_path: str | Path) -> int:
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video file does not exist: {video}")

    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        try:
            ff_meta = _ffprobe_video_meta(ffprobe_path, video)
            ff_nb = _parse_ffprobe_nb_frames(ff_meta)
            if ff_nb is not None:
                return int(ff_nb)
        except Exception:
            pass

    torchcodec_n, _torchcodec_err = _torchcodec_num_frames(video)
    if torchcodec_n is not None:
        return int(torchcodec_n)

    opencv_n, _opencv_err = _opencv_num_frames(video)
    if opencv_n is not None:
        return int(opencv_n)

    raise RuntimeError(f"unable to resolve video frame count: {video}")


def _copy_or_trim_episode_video(
    *,
    ffmpeg_path: str,
    src_mp4: Path,
    dst_mp4: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> None:
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if end_frame < start_frame:
        raise ValueError(
            f"end_frame must be >= start_frame, got start={start_frame} end={end_frame}"
        )

    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        "select='between(n\\,"
        + str(int(start_frame))
        + "\\,"
        + str(int(end_frame))
        + ")',setpts=N/FRAME_RATE/TB"
    )
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(src_mp4),
        "-vf",
        vf,
        "-an",
        "-r",
        str(float(fps)),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dst_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            "ffmpeg failed while exporting LeRobot video clip: "
            f"src={src_mp4} dst={dst_mp4} start_frame={start_frame} end_frame={end_frame} "
            f"rc={proc.returncode} err={detail[:400]}"
        )
    if not dst_mp4.is_file() or dst_mp4.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg wrote no video data: {dst_mp4}")


def attach_videos_to_existing_lerobot_dataset(
    *,
    output_dataset_dir: str | Path,
    episode_video_specs: list[dict[str, Any]],
    fps: float = 30.0,
    chunk_size: int = 1000,
    image_key: str = DEFAULT_IMAGE_KEY,
    original_key: str = DEFAULT_ORIGINAL_KEY,
    require_ffmpeg: bool = True,
) -> LeRobotV2ExportWithVideoResult:
    if not isinstance(episode_video_specs, list) or not episode_video_specs:
        raise ValueError("episode_video_specs must be a non-empty list")
    if float(fps) <= 0.0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")

    output_dataset_dir_path = Path(output_dataset_dir).expanduser().resolve()
    if not output_dataset_dir_path.is_dir():
        raise FileNotFoundError(
            f"LeRobot dataset directory does not exist: {output_dataset_dir_path}"
        )

    ffprobe_path = shutil.which("ffprobe")
    ffmpeg_path = shutil.which("ffmpeg")
    if require_ffmpeg:
        ffprobe_path = _require_tool("ffprobe")
        ffmpeg_path = _require_tool("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "attach_videos_to_existing_lerobot_dataset requires ffmpeg to export clips"
        )

    video_path_template = DEFAULT_VIDEO_PATH_TEMPLATE
    videos_root = output_dataset_dir_path / "videos"
    meta_dir = output_dataset_dir_path / "meta"
    info_path = meta_dir / "info.json"
    modality_path = meta_dir / "modality.json"
    episodes_meta_path = meta_dir / "episodes.jsonl"
    video_map_path = meta_dir / "video_map.json"

    info = _read_json(info_path)
    modality = _read_json(modality_path)
    episodes_meta = _read_jsonl_dicts(episodes_meta_path)

    if len(episodes_meta) != len(episode_video_specs):
        raise RuntimeError(
            "LeRobot episode/video spec length mismatch: "
            f"episodes_meta={len(episodes_meta)} specs={len(episode_video_specs)}"
        )

    features = info.get("features")
    if not isinstance(features, dict):
        features = {}
        info["features"] = features

    video_feature = features.get(original_key)
    if not isinstance(video_feature, dict):
        video_feature = {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
        }
        features[original_key] = video_feature

    probe_meta: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []

    for raw_spec in episode_video_specs:
        if not isinstance(raw_spec, dict):
            raise TypeError(
                "episode_video_specs entries must be dicts, got "
                + type(raw_spec).__name__
            )
        episode_index = _as_int(
            raw_spec.get("episode_index"),
            context="episode_video_specs[*].episode_index",
        )
        if episode_index < 0 or episode_index >= len(episodes_meta):
            raise IndexError(
                f"episode_index out of range for LeRobot meta/episodes.jsonl: {episode_index}"
            )

        src_mp4 = (
            Path(
                _require_str(
                    raw_spec, "src_mp4", context=f"episode_index={episode_index}"
                )
            )
            .expanduser()
            .resolve()
        )
        start_frame = _as_int(
            raw_spec.get("start_frame"), context=f"episode_index={episode_index}"
        )
        end_frame = _as_int(
            raw_spec.get("end_frame"), context=f"episode_index={episode_index}"
        )

        chunk_idx = int(episode_index) // int(chunk_size)
        dst_rel = Path(
            video_path_template.format(
                episode_chunk=int(chunk_idx),
                video_key=str(original_key),
                episode_index=int(episode_index),
            )
        )
        dst_abs = output_dataset_dir_path / dst_rel
        _copy_or_trim_episode_video(
            ffmpeg_path=str(ffmpeg_path),
            src_mp4=src_mp4,
            dst_mp4=dst_abs,
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            fps=float(fps),
        )

        ff_meta: dict[str, Any] | None = None
        if ffprobe_path:
            try:
                ff_meta = _ffprobe_video_meta(str(ffprobe_path), dst_abs)
            except Exception:
                ff_meta = None
        if probe_meta is None:
            probe_meta = ff_meta

        if probe_meta is not None:
            streams = probe_meta.get("streams")
            if isinstance(streams, list) and streams:
                s0 = streams[0]
                if isinstance(s0, dict):
                    w = s0.get("width")
                    h = s0.get("height")
                    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                        video_feature["shape"] = [int(h), int(w), 3]

        records.append(
            {
                "episode_index": int(episode_index),
                "source_episode_id": raw_spec.get("source_episode_id"),
                "source_t": raw_spec.get("source_t"),
                "source_n_policy_steps": raw_spec.get("source_n_policy_steps"),
                "source_video_frame_count": raw_spec.get("source_video_frame_count"),
                "desired_frames": raw_spec.get("desired_frames"),
                "src_mp4": str(src_mp4),
                "dst_mp4": str(dst_rel.as_posix()),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "ffprobe": ff_meta,
            }
        )

    clamped_episodes = 0
    for rec in records:
        ep_idx = _as_int(
            rec.get("episode_index"), context="video_map.records[*].episode_index"
        )
        ep_meta = episodes_meta[int(ep_idx)]
        old_length = _as_int(
            ep_meta.get("length"),
            context=f"output episodes.jsonl episode_index={ep_idx} length",
        )
        dst_mp4_rel = _require_str(rec, "dst_mp4", context=f"episode_index={ep_idx}")
        dst_abs = output_dataset_dir_path / Path(dst_mp4_rel)
        nb_frames = resolve_video_frame_count(dst_abs)
        new_length = int(min(int(old_length), int(nb_frames)))
        if new_length <= 0:
            raise RuntimeError(
                f"invalid clipped video length for episode_index={ep_idx}: old={old_length} nb_frames={nb_frames}"
            )
        if new_length != int(old_length):
            clamped_episodes += 1
        ep_meta["length"] = int(new_length)
        rec["old_length"] = int(old_length)
        rec["new_length"] = int(new_length)
        rec["nb_frames"] = int(nb_frames)
        rec["length_clamp_method"] = "min(old_length, nb_frames)"

    if "video" not in modality or not isinstance(modality.get("video"), dict):
        modality["video"] = {}
    modality_video = modality["video"]
    assert isinstance(modality_video, dict)
    modality_video[str(image_key)] = {"original_key": str(original_key)}

    info["video_path"] = str(video_path_template)
    info["total_videos"] = int(len(records))

    _write_jsonl_dicts(episodes_meta_path, episodes_meta)
    _write_json(info_path, info)
    _write_json(modality_path, modality)
    _write_json(
        video_map_path,
        {
            "dest": {
                "dataset_dir": str(output_dataset_dir_path),
                "videos_root": str(videos_root),
                "video_path_template": str(video_path_template),
                "image_key": str(image_key),
                "original_key": str(original_key),
            },
            "length_clamp": {
                "meta_episodes_jsonl": str(episodes_meta_path),
                "clamped_episodes": int(clamped_episodes),
                "total_episodes": int(len(records)),
            },
            "probe": {"ffprobe": probe_meta},
            "records": records,
        },
    )

    return LeRobotV2ExportWithVideoResult(
        output_dataset_dir=output_dataset_dir_path,
        total_episodes=int(len(records)),
        total_videos=int(len(records)),
        video_path_template=str(video_path_template),
        image_key=str(image_key),
        original_key=str(original_key),
        video_map_path=video_map_path,
    )


def _parse_ffprobe_nb_frames(ffprobe_meta: dict[str, Any] | None) -> int | None:
    if not isinstance(ffprobe_meta, dict):
        return None
    streams = ffprobe_meta.get("streams")
    if not isinstance(streams, list) or not streams:
        return None
    s0 = streams[0]
    if not isinstance(s0, dict):
        return None
    nb = s0.get("nb_frames")
    if nb is None:
        return None
    if isinstance(nb, str):
        nb_s = nb.strip()
        if not nb_s or nb_s.upper() == "N/A":
            return None
        if nb_s.isdigit():
            v = int(nb_s)
            return v if v > 0 else None
        return None
    if isinstance(nb, int) and not isinstance(nb, bool):
        return int(nb) if int(nb) > 0 else None
    if isinstance(nb, float) and nb.is_integer():
        v = int(nb)
        return v if v > 0 else None
    return None


def _parquet_num_rows(parquet_path: Path) -> int:
    try:
        import importlib

        pq = importlib.import_module("pyarrow.parquet")
    except Exception as e:
        raise RuntimeError(
            f"Failed to import pyarrow.parquet for {parquet_path}: {e}"
        ) from e
    try:
        pf = pq.ParquetFile(str(parquet_path))
        md = pf.metadata
        if md is None:
            raise RuntimeError("Missing parquet metadata")
        n = int(md.num_rows)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet num_rows for {parquet_path}: {e}"
        ) from e
    if n <= 0:
        raise RuntimeError(f"Invalid parquet num_rows for {parquet_path}: {n}")
    return n


def _torchcodec_num_frames(video_path: Path) -> tuple[int | None, str | None]:
    try:
        import importlib

        torchcodec = importlib.import_module("torchcodec")
    except Exception as e:
        return None, f"import_error: {e}"
    try:
        dec = torchcodec.decoders.VideoDecoder(
            str(video_path),
            device="cpu",
            dimension_order="NHWC",
            num_ffmpeg_threads=0,
        )
        n = int(len(dec))
        if n <= 0:
            return None, f"invalid_len: {n}"
        return n, None
    except Exception as e:
        return None, f"decoder_error: {e}"


def _opencv_num_frames(video_path: Path) -> tuple[int | None, str | None]:
    try:
        import importlib

        cv2 = importlib.import_module("cv2")
    except Exception as e:
        return None, f"import_error: {e}"
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, "cap_not_opened"
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n <= 0:
            return None, f"invalid_count: {n}"
        return n, None
    except Exception as e:
        return None, f"capture_error: {e}"


def _ffmpeg_probe_has_frame(
    ffmpeg_path: str, video_path: Path, frame_index: int
) -> tuple[bool, str | None]:
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
    p = subprocess.run(cmd, capture_output=True, check=False)
    if p.returncode != 0:
        err = (
            (p.stderr or b"" or p.stdout or b"")
            .decode("utf-8", errors="replace")
            .strip()
        )
        return False, f"ffmpeg_rc={p.returncode} err={err[:240]}"
    if not p.stdout:
        return False, "empty_stdout"
    return True, None


def _require_str(obj: dict[str, Any], key: str, *, context: str) -> str:
    v = obj.get(key)
    if not isinstance(v, str) or not v:
        raise ValueError(f"Missing/invalid {key} ({context}): {v!r}")
    return v


def _require_tool(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise RuntimeError(
            f"Missing required tool: {name}. Install ffmpeg (provides {name})."
        )
    return p


def _ffprobe_video_meta(ffprobe_path: str, video_path: Path) -> dict[str, Any]:
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(
            f"ffprobe failed for {video_path}: rc={p.returncode} err={err[:400]}"
        )
    try:
        obj = json.loads(p.stdout)
    except Exception as e:
        raise RuntimeError(
            f"ffprobe returned invalid JSON for {video_path}: {e}"
        ) from e
    return obj if isinstance(obj, dict) else {"raw": p.stdout}


def _pair_root_mp4s_to_episode_ids(
    *,
    video_dir_archived: Path,
    episode_ids_in_order: list[str],
) -> tuple[list[Path], dict[str, Any]]:
    src_mp4s = [p for p in sorted(video_dir_archived.glob("*.mp4")) if p.is_file()]
    if len(src_mp4s) != len(episode_ids_in_order):
        preview = "\n".join([str(p) for p in src_mp4s[:20]])
        raise RuntimeError(
            "Video count mismatch (fail-fast): "
            f"episodes={len(episode_ids_in_order)} mp4s={len(src_mp4s)} video_dir_archived={video_dir_archived}\n"
            f"mp4_preview:\n{preview}"
        )

    expected_ids = set(episode_ids_in_order)

    parsed_in_expected: dict[str, list[Path]] = {}
    parsed_not_expected: dict[str, list[Path]] = {}
    unparsed_mp4_names: list[str] = []
    for p in src_mp4s:
        parsed = _episode_id_from_mp4_name(p)
        if parsed is None:
            unparsed_mp4_names.append(p.name)
            continue
        if str(parsed) in expected_ids:
            parsed_in_expected.setdefault(str(parsed), []).append(p)
        else:
            parsed_not_expected.setdefault(str(parsed), []).append(p)

    src_mp4s_ordered: list[Path]
    pairing_mode: str
    if parsed_in_expected:
        if unparsed_mp4_names or parsed_not_expected:
            raise RuntimeError(
                "Mixed/partial episode_id filename mapping detected (fail-fast): "
                "some mp4s look episode_id-like but naming is not fully consistent, "
                "so pairing would be ambiguous. "
                f"episodes={len(episode_ids_in_order)} mp4s={len(src_mp4s)} "
                f"parsed_matches={len(parsed_in_expected)} parsed_not_expected={len(parsed_not_expected)} "
                f"unparsed={len(unparsed_mp4_names)} video_dir_archived={video_dir_archived}\n"
                f"unparsed_preview=[{_preview_ids(unparsed_mp4_names)}]\n"
                f"parsed_not_expected_preview=[{_preview_ids(list(parsed_not_expected.keys()))}]"
            )

        dup_ids = sorted(
            [eid for eid, paths in parsed_in_expected.items() if len(paths) != 1]
        )
        if dup_ids:
            example_id = dup_ids[0]
            example_paths = parsed_in_expected.get(example_id) or []
            example_preview = "\n".join([str(p) for p in example_paths[:10]])
            raise RuntimeError(
                "Multiple mp4s mapped to the same episode_id (fail-fast): "
                f"dup_episode_ids={len(dup_ids)} preview=[{_preview_ids(dup_ids)}]\n"
                f"example_episode_id={example_id}\n"
                f"example_paths:\n{example_preview}"
            )

        missing_ids = sorted(expected_ids - set(parsed_in_expected.keys()))
        extra_ids = sorted(set(parsed_in_expected.keys()) - expected_ids)
        if missing_ids or extra_ids:
            raise RuntimeError(
                "Episode<->video pairing mismatch by episode_id (fail-fast): "
                f"episodes={len(episode_ids_in_order)} mp4s={len(src_mp4s)} "
                f"missing={len(missing_ids)} extra={len(extra_ids)} video_dir_archived={video_dir_archived}\n"
                f"missing_preview=[{_preview_ids(missing_ids)}]\n"
                f"extra_preview=[{_preview_ids(extra_ids)}]"
            )

        mp4_by_episode_id: dict[str, Path] = {
            eid: paths[0] for eid, paths in parsed_in_expected.items()
        }
        src_mp4s_ordered = [mp4_by_episode_id[eid] for eid in episode_ids_in_order]
        pairing_mode = "episode_id_filename"
    else:
        src_mp4s_ordered = sorted(
            src_mp4s, key=lambda p: (int(p.stat().st_mtime_ns), p.name)
        )
        pairing_mode = "mtime_ns_name"

    return src_mp4s_ordered, {
        "video_dir_archived": str(video_dir_archived),
        "episode_count": int(len(episode_ids_in_order)),
        "mp4_count": int(len(src_mp4s)),
        "pairing_mode": str(pairing_mode),
    }


def export_recap_to_lerobot_v2_with_video(
    *,
    iter_tag: str,
    repo_root: str | Path | None = None,
    input_recap_dataset_dir: str | Path | None = None,
    output_dataset_dir: str | Path | None = None,
    max_episodes: int | None = None,
    task_text_field: str = dataset_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD,
    dual_task_text: bool = False,
    fps: float = 30.0,
    chunk_size: int = 1000,
    include_m2_label_columns: bool = True,
    require_ffmpeg: bool = False,
) -> LeRobotV2ExportWithVideoResult:
    root = _repo_root_from_here() if repo_root is None else Path(repo_root)
    root = root.resolve()

    ffprobe_path = shutil.which("ffprobe")
    ffmpeg_path = shutil.which("ffmpeg")
    if require_ffmpeg:
        ffprobe_path = _require_tool("ffprobe")
        ffmpeg_path = _require_tool("ffmpeg")
    _ = ffmpeg_path

    if input_recap_dataset_dir is None:
        input_dir = root / "agent" / "artifacts" / "recap_datasets" / iter_tag
    else:
        p = Path(input_recap_dataset_dir)
        input_dir = (root / p).resolve() if not p.is_absolute() else p.resolve()

    exp_mod = __import__(
        "work.recap.lerobot_export.dataset_export",
        fromlist=["export_recap_to_lerobot_v2"],
    )
    export_fn = getattr(exp_mod, "export_recap_to_lerobot_v2")
    resolve_out_dir_fn = getattr(exp_mod, "resolve_lerobot_v2_dataset_dir")

    expected_out_dir = resolve_out_dir_fn(iter_tag=iter_tag, repo_root=root)
    out_dir = (
        expected_out_dir if output_dataset_dir is None else Path(output_dataset_dir)
    )

    result = export_fn(
        iter_tag=str(iter_tag),
        repo_root=root,
        input_recap_dataset_dir=str(input_dir),
        output_dataset_dir=str(out_dir),
        max_episodes=max_episodes,
        task_text_field=str(task_text_field),
        dual_task_text=bool(dual_task_text),
        fps=float(fps),
        chunk_size=int(chunk_size),
        include_m2_label_columns=bool(include_m2_label_columns),
    )

    output_dataset_dir_path = Path(getattr(result, "output_dataset_dir"))
    total_episodes = int(getattr(result, "total_episodes"))

    recap_episodes_path = input_dir / "episodes.jsonl"
    recap_eps = _read_jsonl_dicts(recap_episodes_path)
    recap_eps = [e for e in recap_eps if isinstance(e.get("episode_id"), str)]
    if max_episodes is not None:
        recap_eps = recap_eps[: int(max_episodes)]
    if not recap_eps:
        raise ValueError(f"No RECAP episodes found in {recap_episodes_path}")
    if len(recap_eps) != total_episodes:
        raise ValueError(
            f"Episode count mismatch: exporter wrote {total_episodes} but input has {len(recap_eps)}"
        )

    episode_ids_in_order: list[str] = []
    dup_episode_ids: set[str] = set()
    seen_episode_ids: set[str] = set()

    episode_refs_by_video_dir: dict[str, list[tuple[int, str]]] = {}
    for i, ep in enumerate(recap_eps):
        context = f"episodes.jsonl record#{i + 1}"
        episode_id = _require_str(ep, "episode_id", context=context)
        episode_ids_in_order.append(str(episode_id))
        if str(episode_id) in seen_episode_ids:
            dup_episode_ids.add(str(episode_id))
        seen_episode_ids.add(str(episode_id))
        vd = ep.get("video_dir_archived")
        if not isinstance(vd, str) or not vd:
            raise ValueError(f"Missing video_dir_archived ({context}): {vd!r}")
        vd_resolved = str(Path(vd).resolve())
        episode_refs_by_video_dir.setdefault(vd_resolved, []).append(
            (int(i), str(episode_id))
        )

    if dup_episode_ids:
        raise ValueError(
            "Duplicate episode_id in episodes.jsonl (fail-fast): "
            f"episodes={len(recap_eps)} unique_episode_ids={len(seen_episode_ids)} "
            f"duplicates={len(dup_episode_ids)} preview=[{_preview_ids(sorted(dup_episode_ids))}]"
        )

    unique_video_dirs = list(episode_refs_by_video_dir.keys())
    root_pairing_summaries: list[dict[str, Any]] = []
    src_mp4_by_episode_index: dict[int, Path] = {}
    total_src_mp4s = 0
    for video_dir in unique_video_dirs:
        video_dir_archived = Path(video_dir).resolve()
        if not video_dir_archived.is_dir():
            raise FileNotFoundError(
                f"video_dir_archived is not a directory: {video_dir_archived}"
            )
        root_episode_refs = episode_refs_by_video_dir[video_dir]
        root_episode_ids_in_order = [episode_id for _, episode_id in root_episode_refs]
        root_src_mp4s_ordered, root_summary = _pair_root_mp4s_to_episode_ids(
            video_dir_archived=video_dir_archived,
            episode_ids_in_order=root_episode_ids_in_order,
        )
        root_pairing_summaries.append(root_summary)
        total_src_mp4s += int(len(root_src_mp4s_ordered))
        for (episode_index, _episode_id), src_mp4 in zip(
            root_episode_refs, root_src_mp4s_ordered
        ):
            if int(episode_index) in src_mp4_by_episode_index:
                raise RuntimeError(
                    "Duplicate episode_index while pairing videos across roots (fail-fast): "
                    f"episode_index={episode_index} video_dir_archived={video_dir_archived}"
                )
            src_mp4_by_episode_index[int(episode_index)] = src_mp4

    missing_episode_indices = sorted(
        set(range(len(recap_eps))) - set(src_mp4_by_episode_index.keys())
    )
    if missing_episode_indices:
        raise RuntimeError(
            "Failed to pair videos for all episode indices (fail-fast): "
            f"missing={len(missing_episode_indices)} preview=[{_preview_ids([str(i) for i in missing_episode_indices])}]"
        )
    src_mp4s_ordered = [src_mp4_by_episode_index[i] for i in range(len(recap_eps))]
    if len(src_mp4s_ordered) != len(recap_eps):
        raise RuntimeError(
            "Unexpected paired video count mismatch after root grouping: "
            f"episodes={len(recap_eps)} paired_mp4s={len(src_mp4s_ordered)}"
        )

    single_video_dir_archived = (
        str(Path(unique_video_dirs[0]).resolve())
        if len(unique_video_dirs) == 1
        else None
    )

    image_key = DEFAULT_IMAGE_KEY
    original_key = DEFAULT_ORIGINAL_KEY
    video_path_template = DEFAULT_VIDEO_PATH_TEMPLATE

    videos_root = output_dataset_dir_path / "videos"
    meta_dir = output_dataset_dir_path / "meta"

    info_path = meta_dir / "info.json"
    modality_path = meta_dir / "modality.json"
    video_map_path = meta_dir / "video_map.json"

    info = _read_json(info_path)
    info["recap_export.dual_task_text"] = bool(dual_task_text)
    if bool(dual_task_text):
        info["task_text_mode"] = "mix50"
    features = info.get("features")
    if not isinstance(features, dict):
        features = {}
        info["features"] = features

    video_feature = features.get(original_key)
    if not isinstance(video_feature, dict):
        video_feature = {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
        }
        features[original_key] = video_feature

    probe_meta: dict[str, Any] | None = None
    if ffprobe_path and src_mp4s_ordered:
        try:
            probe_meta = _ffprobe_video_meta(ffprobe_path, src_mp4s_ordered[0])
            streams = probe_meta.get("streams")
            if isinstance(streams, list) and streams:
                s0 = streams[0]
                if isinstance(s0, dict):
                    w = s0.get("width")
                    h = s0.get("height")
                    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                        video_feature["shape"] = [int(h), int(w), 3]
        except Exception:
            probe_meta = None

    info["video_path"] = str(video_path_template)
    info["total_videos"] = int(total_episodes)

    modality = _read_json(modality_path)
    if "video" not in modality or not isinstance(modality.get("video"), dict):
        modality["video"] = {}
    modality_video = modality["video"]
    assert isinstance(modality_video, dict)
    modality_video[str(image_key)] = {"original_key": str(original_key)}

    records: list[dict[str, Any]] = []
    for episode_index, ep in enumerate(recap_eps):
        episode_id = _require_str(
            ep, "episode_id", context=f"episodes.jsonl idx={episode_index}"
        )
        src = src_mp4s_ordered[int(episode_index)]
        chunk_idx = int(episode_index) // int(chunk_size)
        dst_rel = Path(
            video_path_template.format(
                episode_chunk=int(chunk_idx),
                video_key=str(original_key),
                episode_index=int(episode_index),
            )
        )
        dst_abs = output_dataset_dir_path / dst_rel
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_abs)
        if dst_abs.stat().st_size <= 0:
            raise RuntimeError(f"Copied empty video file: {dst_abs}")

        ff_meta: dict[str, Any] | None = None
        if ffprobe_path:
            try:
                ff_meta = _ffprobe_video_meta(ffprobe_path, dst_abs)
            except Exception:
                ff_meta = None

        records.append(
            {
                "episode_index": int(episode_index),
                "recap.episode_id": str(episode_id),
                "src_mp4": str(src),
                "src_video_dir_archived": str(src.parent.resolve()),
                "dst_mp4": str(dst_rel.as_posix()),
                "src_stat": {
                    "size": int(src.stat().st_size),
                    "mtime_ns": int(src.stat().st_mtime_ns),
                },
                "dst_stat": {
                    "size": int(dst_abs.stat().st_size),
                    "mtime_ns": int(dst_abs.stat().st_mtime_ns),
                },
                "ffprobe": ff_meta,
            }
        )

    episodes_meta_path = meta_dir / "episodes.jsonl"
    episodes_meta = _read_jsonl_dicts(episodes_meta_path)
    if len(episodes_meta) != int(total_episodes):
        raise RuntimeError(
            "Unexpected output meta/episodes.jsonl length: "
            f"expected={int(total_episodes)} got={len(episodes_meta)} path={episodes_meta_path}"
        )

    meta_by_idx: dict[int, dict[str, Any]] = {}
    for rec_i, ep_meta in enumerate(episodes_meta):
        if not isinstance(ep_meta, dict):
            raise ValueError(
                f"Invalid output episodes.jsonl record type at line {rec_i + 1}: {type(ep_meta).__name__}"
            )
        ep_idx = _as_int(
            ep_meta.get("episode_index"),
            context=f"output episodes.jsonl line {rec_i + 1} episode_index",
        )
        if ep_idx in meta_by_idx:
            raise ValueError(
                f"Duplicate episode_index in {episodes_meta_path}: {ep_idx}"
            )
        meta_by_idx[int(ep_idx)] = ep_meta

    data_path_template = _require_str(info, "data_path", context=str(info_path))
    chunks_size = _as_int(info.get("chunks_size"), context=f"{info_path} chunks_size")
    if chunks_size != int(chunk_size):
        raise RuntimeError(
            "Chunk size mismatch: "
            f"info.chunks_size={chunks_size} != export.chunk_size={int(chunk_size)} (iter_tag={iter_tag})"
        )

    clamped_episodes = 0
    for r in records:
        ep_idx = _as_int(
            r.get("episode_index"), context="video_map.records[*].episode_index"
        )
        ep_meta = meta_by_idx.get(int(ep_idx))
        if ep_meta is None:
            raise KeyError(f"Missing episode_index={ep_idx} in {episodes_meta_path}")

        old_length = _as_int(
            ep_meta.get("length"),
            context=f"output episodes.jsonl episode_index={ep_idx} length",
        )
        if old_length <= 0:
            raise ValueError(
                f"Invalid nominal episode length in {episodes_meta_path}: episode_index={ep_idx} length={old_length}"
            )

        chunk_idx = int(ep_idx) // int(chunk_size)
        parquet_rel = data_path_template.format(
            episode_chunk=int(chunk_idx),
            episode_index=int(ep_idx),
        )
        parquet_abs = output_dataset_dir_path / Path(parquet_rel)
        if not parquet_abs.is_file():
            raise FileNotFoundError(
                f"Missing parquet for episode_index={ep_idx}: {parquet_abs} (data_path={parquet_rel})"
            )
        parquet_rows = int(_parquet_num_rows(parquet_abs))

        dst_mp4_rel = r.get("dst_mp4")
        if not isinstance(dst_mp4_rel, str) or not dst_mp4_rel:
            raise ValueError(
                f"Missing dst_mp4 in video_map record for episode_index={ep_idx}: {dst_mp4_rel!r}"
            )
        video_abs = output_dataset_dir_path / Path(dst_mp4_rel)
        if not video_abs.is_file():
            raise FileNotFoundError(
                f"Missing video for episode_index={ep_idx}: {video_abs} (dst_mp4={dst_mp4_rel})"
            )

        nb_frames: int | None = None
        nb_frames_source: str | None = None
        torchcodec_err: str | None = None
        opencv_err: str | None = None

        ff_nb = _parse_ffprobe_nb_frames(
            r.get("ffprobe") if isinstance(r.get("ffprobe"), dict) else None
        )
        if ff_nb is not None:
            nb_frames = int(ff_nb)
            nb_frames_source = "ffprobe.nb_frames"
        else:
            tc_n, tc_err = _torchcodec_num_frames(video_abs)
            torchcodec_err = tc_err
            if tc_n is not None:
                nb_frames = int(tc_n)
                nb_frames_source = "torchcodec.len"
            else:
                oc_n, oc_err = _opencv_num_frames(video_abs)
                opencv_err = oc_err
                if oc_n is not None:
                    nb_frames = int(oc_n)
                    nb_frames_source = "opencv.frame_count"

        max_by_parquet = min(int(old_length), int(parquet_rows))
        if max_by_parquet <= 0:
            raise RuntimeError(
                "Invalid parquet clamp: "
                f"iter_tag={iter_tag} episode_index={ep_idx} old_length={old_length} parquet_rows={parquet_rows}"
            )

        new_length = int(max_by_parquet)
        clamp_method = "min(old_length, parquet_num_rows)"
        validated_by: str | None = None
        ffmpeg_probe_error: str | None = None

        if nb_frames is not None:
            new_length = int(min(int(new_length), int(nb_frames)))
            clamp_method = "min(old_length, parquet_num_rows, nb_frames)"
            validated_by = str(nb_frames_source)
        else:
            heuristic_len = int(min(int(new_length), max(1, int(parquet_rows) - 1)))
            new_length = int(heuristic_len)
            clamp_method = (
                "min(old_length, parquet_num_rows, max(1, parquet_num_rows-1))"
            )

            if ffmpeg_path:
                while True:
                    ok, probe_err = _ffmpeg_probe_has_frame(
                        str(ffmpeg_path), video_abs, int(new_length) - 1
                    )
                    ffmpeg_probe_error = probe_err
                    if ok:
                        validated_by = "ffmpeg.frame_probe"
                        break
                    new_length -= 1
                    if new_length <= 0:
                        raise RuntimeError(
                            "Cannot find any valid video frame index (fail-fast): "
                            f"iter_tag={iter_tag} episode_index={ep_idx} video={video_abs} "
                            f"old_length={old_length} parquet_rows={parquet_rows} last_probe_err={probe_err}"
                        )
            else:
                raise RuntimeError(
                    "Cannot guarantee video frame indices will be in-bounds without ffprobe/torchcodec/opencv; "
                    f"ffmpeg is also missing. iter_tag={iter_tag} episode_index={ep_idx} video={video_abs} "
                    f"old_length={old_length} parquet_rows={parquet_rows}"
                )

        if new_length <= 0:
            raise RuntimeError(
                f"Computed invalid new_length={new_length} (iter_tag={iter_tag} episode_index={ep_idx})"
            )
        if nb_frames is not None and int(new_length) > int(nb_frames):
            raise RuntimeError(
                "Length clamp failed (would still be out-of-range): "
                f"iter_tag={iter_tag} episode_index={ep_idx} new_length={new_length} nb_frames={nb_frames}"
            )

        if int(new_length) != int(old_length):
            clamped_episodes += 1

        ep_meta["length"] = int(new_length)
        r["old_length"] = int(old_length)
        r["new_length"] = int(new_length)
        r["episode_meta_length_old"] = int(old_length)
        r["episode_meta_length_new"] = int(new_length)
        r["parquet_num_rows"] = int(parquet_rows)
        if nb_frames is not None:
            r["nb_frames"] = int(nb_frames)
        r["nb_frames_source"] = nb_frames_source
        r["torchcodec_err"] = torchcodec_err
        r["opencv_err"] = opencv_err
        r["ffmpeg_probe_err"] = ffmpeg_probe_error
        r["length_clamp_method"] = str(clamp_method)
        r["length_clamp_validated_by"] = validated_by

    _write_jsonl_dicts(episodes_meta_path, episodes_meta)

    video_map_obj = {
        "iter_tag": str(iter_tag),
        "source": {
            "video_dir_archived": single_video_dir_archived,
            "video_dir_archived_roots": [
                str(Path(video_dir).resolve()) for video_dir in unique_video_dirs
            ],
            "video_dir_archived_groups": root_pairing_summaries,
            "episodes_jsonl": str(recap_episodes_path),
            "mp4_count": int(total_src_mp4s),
            "dual_task_text": bool(dual_task_text),
        },
        "dest": {
            "dataset_dir": str(output_dataset_dir_path),
            "videos_root": str(videos_root),
            "video_path_template": str(video_path_template),
            "image_key": str(image_key),
            "original_key": str(original_key),
        },
        "probe": {"ffprobe": probe_meta},
        "length_clamp": {
            "meta_episodes_jsonl": str(episodes_meta_path),
            "clamped_episodes": int(clamped_episodes),
            "total_episodes": int(total_episodes),
        },
        "records": records,
    }

    _write_json(info_path, info)
    _write_json(modality_path, modality)
    _write_json(video_map_path, video_map_obj)

    return LeRobotV2ExportWithVideoResult(
        output_dataset_dir=output_dataset_dir_path,
        total_episodes=int(total_episodes),
        total_videos=int(total_episodes),
        video_path_template=str(video_path_template),
        image_key=str(image_key),
        original_key=str(original_key),
        video_map_path=video_map_path,
    )
