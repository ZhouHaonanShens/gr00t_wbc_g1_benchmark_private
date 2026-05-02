from __future__ import annotations

from pathlib import Path

from .common import (
    JsonObject,
    as_float,
    as_int,
    as_str,
    parquet_read_table,
    probe_frame_decode,
    read_json,
    read_jsonl,
    table_scalar,
)
from .schema import DatasetSample


def load_dataset_sample(dataset_path: str | Path, sample_index: int) -> DatasetSample:
    dataset_root = Path(dataset_path).expanduser().resolve()
    if sample_index < 0:
        raise ValueError(f"sample_index must be >= 0, got {sample_index}")
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_path is not a directory: {dataset_root}")

    meta_dir = dataset_root / "meta"
    info = read_json(meta_dir / "info.json")
    if as_int(info.get("total_videos"), context="meta/info.json total_videos") <= 0:
        raise RuntimeError(
            "video_decode_missing: artifact smoke requires a with-video dataset (total_videos>0)"
        )
    data_path_template = as_str(
        info.get("data_path"), context="meta/info.json data_path"
    )
    chunks_size = as_int(info.get("chunks_size"), context="meta/info.json chunks_size")
    episodes_meta = sorted(
        read_jsonl(meta_dir / "episodes.jsonl"),
        key=lambda item: as_int(item.get("episode_index"), context="episode_index"),
    )
    video_map = read_json(meta_dir / "video_map.json")
    records_raw = video_map.get("records")
    if not isinstance(records_raw, list) or not records_raw:
        raise RuntimeError(
            "video_decode_missing: artifact smoke requires meta/video_map.json with records"
        )

    video_by_episode: dict[int, JsonObject] = {}
    for idx, rec in enumerate(records_raw):
        if not isinstance(rec, dict):
            raise ValueError(
                f"Invalid video_map record at index {idx}: {type(rec).__name__}"
            )
        ep_idx = as_int(
            rec.get("episode_index"), context=f"video_map.records[{idx}].episode_index"
        )
        video_by_episode[ep_idx] = rec

    cursor = 0
    selected_meta: JsonObject | None = None
    local_index = 0
    for ep_meta in episodes_meta:
        ep_length = as_int(ep_meta.get("length"), context="meta/episodes length")
        if sample_index < cursor + ep_length:
            selected_meta = ep_meta
            local_index = int(sample_index - cursor)
            break
        cursor += int(ep_length)
    if selected_meta is None:
        raise IndexError(
            f"sample_index out of range: sample_index={sample_index} total_rows={cursor}"
        )

    episode_index = as_int(
        selected_meta.get("episode_index"), context="selected episode_index"
    )
    episode_length = as_int(
        selected_meta.get("length"), context="selected episode length"
    )
    recap_episode_id = as_str(
        selected_meta.get("recap.episode_id"), context="selected recap.episode_id"
    )
    chunk_idx = int(episode_index) // int(chunks_size)
    parquet_rel = data_path_template.format(
        episode_chunk=int(chunk_idx),
        episode_index=int(episode_index),
    )
    parquet_abs = dataset_root / Path(parquet_rel)

    video_rec = video_by_episode.get(episode_index)
    if video_rec is None:
        raise RuntimeError(
            f"video_decode_missing: no video record for episode_index={episode_index}"
        )
    video_rel = as_str(
        video_rec.get("dst_mp4"), context=f"video_map[{episode_index}] dst_mp4"
    )
    video_abs = dataset_root / Path(video_rel)
    if not video_abs.is_file():
        raise RuntimeError(
            f"video_decode_missing: missing video file for episode_index={episode_index}: {video_abs}"
        )

    required_columns = [
        "recap_m2.prompt_raw",
        "recap_m2.return_G",
        "recap_m2.t",
        "index",
        "episode_index",
    ]
    table = parquet_read_table(parquet_abs, columns=required_columns)
    prompt_raw = as_str(
        table_scalar(table, "recap_m2.prompt_raw", local_index),
        context="sample prompt_raw",
    )
    return_g = as_float(
        table_scalar(table, "recap_m2.return_G", local_index),
        context="sample return_G",
    )
    t_i = as_int(table_scalar(table, "recap_m2.t", local_index), context="sample t")
    frame_index = as_int(
        table_scalar(table, "index", local_index),
        context="sample index",
    )
    episode_index_in_row = as_int(
        table_scalar(table, "episode_index", local_index),
        context="sample episode_index",
    )
    if episode_index_in_row != int(episode_index):
        raise RuntimeError(
            f"artifact_smoke_sample_invalid: row episode_index mismatch expected={episode_index} "
            f"got={episode_index_in_row}"
        )

    backend = probe_frame_decode(video_abs, frame_index)
    return DatasetSample(
        dataset_path=dataset_root,
        episode_index=int(episode_index),
        episode_length=int(episode_length),
        recap_episode_id=str(recap_episode_id),
        sample_index=int(sample_index),
        local_index=int(local_index),
        frame_index=int(frame_index),
        t=int(t_i),
        prompt_raw=str(prompt_raw),
        return_G=float(return_g),
        parquet_rel=str(Path(parquet_rel).as_posix()),
        video_rel=str(Path(video_rel).as_posix()),
        video_decode_backend=str(backend),
    )
