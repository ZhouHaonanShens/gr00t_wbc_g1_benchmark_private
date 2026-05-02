#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast


JsonObject = dict[str, object]


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_DIR_REL = "agent/artifacts/vlm_critic_manifests/task2_split"
DEFAULT_SPLIT_GRANULARITY = "episode"


PASS_SENTINEL = "SPLIT_MANIFEST_OK"
FAIL_SENTINEL = "SPLIT_MANIFEST_FAIL"


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


def _as_str(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string ({context}), got {value!r}")
    return str(value)


def _sorted_episode_records(dataset_path: Path) -> list[JsonObject]:
    meta_dir = dataset_path / "meta"
    info = _read_json(meta_dir / "info.json")
    if _as_int(info.get("total_videos"), context="meta/info.json total_videos") <= 0:
        raise RuntimeError(
            "video_decode_missing: split manifest requires a with-video dataset (total_videos>0)"
        )
    data_path_template = _as_str(
        info.get("data_path"), context="meta/info.json data_path"
    )
    chunks_size = _as_int(info.get("chunks_size"), context="meta/info.json chunks_size")

    episodes_meta = _read_jsonl(meta_dir / "episodes.jsonl")
    video_map = _read_json(meta_dir / "video_map.json")
    records_raw = video_map.get("records")
    if not isinstance(records_raw, list) or not records_raw:
        raise RuntimeError(
            "video_decode_missing: split manifest requires meta/video_map.json with records"
        )

    video_by_episode: dict[int, JsonObject] = {}
    for idx, rec in enumerate(records_raw):
        if not isinstance(rec, dict):
            raise ValueError(
                f"Invalid video_map record at index {idx}: {type(rec).__name__}"
            )
        episode_index = _as_int(
            rec.get("episode_index"), context=f"video_map.records[{idx}].episode_index"
        )
        video_by_episode[episode_index] = cast(JsonObject, rec)

    out: list[JsonObject] = []
    for ep_meta in episodes_meta:
        episode_index = _as_int(
            ep_meta.get("episode_index"), context="meta/episodes episode_index"
        )
        episode_id = _as_str(
            ep_meta.get("recap.episode_id"),
            context=f"episode_index={episode_index} recap.episode_id",
        )
        length = _as_int(
            ep_meta.get("length"), context=f"episode_index={episode_index} length"
        )
        chunk_idx = int(episode_index) // int(chunks_size)
        parquet_rel = data_path_template.format(
            episode_chunk=int(chunk_idx),
            episode_index=int(episode_index),
        )
        video_rec = video_by_episode.get(episode_index)
        if video_rec is None:
            raise RuntimeError(
                f"video_decode_missing: split manifest missing video_map record for episode_index={episode_index}"
            )
        video_rel = _as_str(
            video_rec.get("dst_mp4"), context=f"video_map[{episode_index}] dst_mp4"
        )
        hash_hex = hashlib.sha256(str(episode_id).encode("utf-8")).hexdigest()
        out.append(
            {
                "episode_index": int(episode_index),
                "recap_episode_id": str(episode_id),
                "episode_length": int(length),
                "episode_id_hash": str(hash_hex),
                "parquet_rel": str(Path(parquet_rel).as_posix()),
                "video_rel": str(Path(video_rel).as_posix()),
            }
        )

    out.sort(
        key=lambda item: (
            _as_str(item.get("episode_id_hash"), context="episode_id_hash"),
            _as_int(item.get("episode_index"), context="episode_index"),
        )
    )
    return out


def _split_counts(total_episodes: int) -> tuple[int, int, int, str]:
    if total_episodes <= 0:
        raise ValueError(f"Expected total_episodes > 0, got {total_episodes}")
    if total_episodes == 1:
        return 1, 0, 0, "single_episode_train_only"
    if total_episodes == 2:
        return 1, 0, 1, "two_episode_train_test"
    if total_episodes < 10:
        return total_episodes - 2, 1, 1, "small_n_holdout_guard"
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
    return train_n, val_n, test_n, "sorted_episode_id_hash_then_80_10_10_counts"


def _build_manifest(
    *,
    dataset_path: Path,
    output_dir: Path,
    split_name: str,
    records: list[JsonObject],
    source_manifest_path: Path,
) -> JsonObject:
    total_frames = sum(
        _as_int(item.get("episode_length"), context=f"{split_name}.episode_length")
        for item in records
    )
    return {
        "schema_version": "vlm_critic_split_manifest_v1",
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "split_name": str(split_name),
        "split_key": "episode_id_hash",
        "split_granularity": "episode",
        "formal_eval_scope": "isaac_only",
        "leakage_check": "passed",
        "sample_level_split_forbidden": True,
        "source_manifest_path": str(source_manifest_path),
        "counts": {
            "episodes": int(len(records)),
            "frames": int(total_frames),
        },
        "records": records,
    }


def _emit_result(
    *, sentinel: str, output_dir: Path, payload: Mapping[str, object]
) -> None:
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, payload)
    print(f"[INFO] wrote_json: {summary_path}")
    print(f"SENTINEL:{sentinel}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="41b_vlm_critic_split_manifest.py",
        description="Build deterministic episode-level train/val/test manifests for the Task 2 VLM critic bootstrap line.",
    )
    _ = parser.add_argument("--dataset-path", type=str, required=True)
    _ = parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR_REL)
    _ = parser.add_argument(
        "--split-granularity",
        type=str,
        default=DEFAULT_SPLIT_GRANULARITY,
        help="Only episode-level splitting is allowed in Task 2.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    dataset_path = _resolve_path(repo_root, str(args.dataset_path), default_rel="")
    output_dir = _resolve_path(
        repo_root, str(args.output_dir), default_rel=DEFAULT_OUTPUT_DIR_REL
    )

    try:
        split_granularity = (
            str(args.split_granularity or DEFAULT_SPLIT_GRANULARITY).strip().lower()
        )
        if split_granularity != "episode":
            raise ValueError(
                f"sample_level_split_forbidden: Task 2 requires episode-granularity or stronger splits, got {split_granularity!r}"
            )
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"dataset_path is not a directory: {dataset_path}")

        source_records = _sorted_episode_records(dataset_path)
        train_n, val_n, test_n, policy = _split_counts(len(source_records))

        train_records = source_records[:train_n]
        val_records = source_records[train_n : train_n + val_n]
        test_records = source_records[train_n + val_n : train_n + val_n + test_n]

        train_ids = {str(item["recap_episode_id"]) for item in train_records}
        val_ids = {str(item["recap_episode_id"]) for item in val_records}
        test_ids = {str(item["recap_episode_id"]) for item in test_records}
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise RuntimeError(
                "leakage_check_failed: duplicate episode ids leaked across splits"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        source_manifest_path = output_dir / "source_manifest.json"
        source_manifest: JsonObject = {
            "schema_version": "vlm_critic_split_source_manifest_v1",
            "dataset_path": str(dataset_path),
            "split_key": "episode_id_hash",
            "split_granularity": "episode",
            "split_policy": str(policy),
            "formal_eval_scope": "isaac_only",
            "leakage_check": "passed",
            "sample_level_split_forbidden": True,
            "counts": {
                "episodes": int(len(source_records)),
                "train": int(len(train_records)),
                "val": int(len(val_records)),
                "test": int(len(test_records)),
            },
            "records": source_records,
        }
        _write_json(source_manifest_path, source_manifest)

        train_manifest = _build_manifest(
            dataset_path=dataset_path,
            output_dir=output_dir,
            split_name="train",
            records=train_records,
            source_manifest_path=source_manifest_path,
        )
        val_manifest = _build_manifest(
            dataset_path=dataset_path,
            output_dir=output_dir,
            split_name="val",
            records=val_records,
            source_manifest_path=source_manifest_path,
        )
        test_manifest = _build_manifest(
            dataset_path=dataset_path,
            output_dir=output_dir,
            split_name="test",
            records=test_records,
            source_manifest_path=source_manifest_path,
        )

        _write_json(output_dir / "train.json", train_manifest)
        _write_json(output_dir / "val.json", val_manifest)
        _write_json(output_dir / "test.json", test_manifest)

        summary: JsonObject = {
            "pass": True,
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "split_key": "episode_id_hash",
            "split_granularity": "episode",
            "formal_eval_scope": "isaac_only",
            "leakage_check": "passed",
            "sample_level_split_forbidden": True,
            "counts": source_manifest["counts"],
            "artifacts": {
                "source_manifest": str(source_manifest_path),
                "train": str(output_dir / "train.json"),
                "val": str(output_dir / "val.json"),
                "test": str(output_dir / "test.json"),
            },
        }
        _emit_result(sentinel=PASS_SENTINEL, output_dir=output_dir, payload=summary)
        return 0
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure: JsonObject = {
            "pass": False,
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _emit_result(sentinel=FAIL_SENTINEL, output_dir=output_dir, payload=failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
