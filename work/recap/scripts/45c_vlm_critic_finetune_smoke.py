#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


from work.recap.advantage import (
    CONTINUOUS_ADVANTAGE_CONTRACT_DIAGNOSTIC_ROUTE,
    NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
    VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE,
    build_diagnostic_surface_metadata,
)


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


DEFAULT_MAIN_REPO_ROOT = str(Path(__file__).resolve().parents[3])
DEFAULT_MAIN_REPO_PYTHON = f"{DEFAULT_MAIN_REPO_ROOT}/.envs/wbc/bin/python"
DEFAULT_MAIN_REPO_SITE_PACKAGES_PYTHON = f"{DEFAULT_MAIN_REPO_ROOT}/.envs/main/bin/python"
DEFAULT_UPSTREAM_SCRIPT = (
    f"{DEFAULT_MAIN_REPO_ROOT}/work/recap/scripts/3D_recap_finetune_full.py"
)
DEFAULT_DATASET_PATH = (
    "agent/artifacts/lerobot_datasets/"
    "recap_mainline_fresh_20260311_121500_k0_t8_local_smoke"
)
DEFAULT_OUTPUT_JSON = "agent/artifacts/vlm_critic_relabel/finetune_smoke.json"
DEFAULT_RUNTIME_LOG_DIR = "agent/runtime_logs/task10_vlm_critic_downstream_smoke"
DEFAULT_OUTPUT_DIR_PREFIX = (
    "agent/artifacts/checkpoints/task10_vlm_critic_finetune_smoke_"
)
DEFAULT_PREPARED_DATASET_SUFFIX = "__45c_finetune_ready"
DEFAULT_BASE_MODEL_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1"
DEFAULT_MAX_STEPS = 4
DEFAULT_SAVE_STEPS = 4
DEFAULT_SEED = 42
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_GLOBAL_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_DATALOADER_NUM_WORKERS = 0
DEFAULT_DEVICE = "auto"
DEFAULT_LOG_EVERY = 1
DEFAULT_MAX_EPISODE_ROWS_FOR_PREP = 96
PASS_SENTINEL = "VLM_CRITIC_FINETUNE_SMOKE_OK"
UPGRADE_PENDING = "temporal_critic_review"
UPSTREAM_SMOKE_SHARD_SIZE = 128
UPSTREAM_EPISODE_SAMPLING_RATE = 0.1
UPSTREAM_ACTION_HORIZON = 30
ADVANTAGE_INPUT_COLUMN = "recap_m2.advantage_input"
ADVANTAGE_RAW_COLUMN = "recap_m2.advantage_A"
ADVANTAGE_CONTRACT_VERSION = "full_recap_continuous_adv_v1"
ADVANTAGE_CLIP_MIN = -1.0
ADVANTAGE_CLIP_MAX = 1.0
ADVANTAGE_SCALE_EPS = 1e-6
ADVANTAGE_SCALE_ABS_QUANTILE = 0.95


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return dict(data)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object in {path} line {line_no}, got {type(obj).__name__}"
                )
            rows.append(dict(obj))
    return rows


def _require_int_field(obj: dict[str, Any], key: str, *, context: str) -> int:
    if key not in obj:
        raise KeyError(f"Missing required int field {key!r} ({context})")
    raw = obj[key]
    if raw is None:
        raise ValueError(f"Required int field {key!r} is null ({context})")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected int-like field {key!r} for {context}, got {raw!r}"
        ) from exc
    return int(value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=True, sort_keys=True)
            f.write("\n")
    tmp.replace(path)


def _copy_text_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path)


def _coerce_finite_float(raw: Any, *, context: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"Expected finite float-like value, got bool ({context})")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected float-like value for {context}, got {type(raw).__name__}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"Expected finite float for {context}, got {value!r}")
    return float(value)


def _linear_quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q!r}")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = float(q) * float(len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight_hi = pos - float(lo)
    weight_lo = 1.0 - weight_hi
    return float(weight_lo * ordered[lo] + weight_hi * ordered[hi])


def _compute_p95_abs_advantage(values: list[float]) -> float:
    if not values:
        raise ValueError("advantage values must be non-empty")
    abs_values = [
        abs(_coerce_finite_float(v, context=ADVANTAGE_RAW_COLUMN)) for v in values
    ]
    p95 = _linear_quantile(abs_values, ADVANTAGE_SCALE_ABS_QUANTILE)
    if p95 <= float(ADVANTAGE_SCALE_EPS):
        raise ValueError(
            "p95_abs_advantage is too small for stable scaling: "
            f"{p95:.8f} <= {ADVANTAGE_SCALE_EPS}"
        )
    return float(p95)


def _scale_advantage_input(advantage_value: Any, *, p95_abs_advantage: float) -> float:
    raw = _coerce_finite_float(advantage_value, context=ADVANTAGE_RAW_COLUMN)
    scale = _coerce_finite_float(
        p95_abs_advantage, context=f"{ADVANTAGE_INPUT_COLUMN}.p95_abs_advantage"
    )
    if scale <= float(ADVANTAGE_SCALE_EPS):
        raise ValueError(
            f"p95_abs_advantage must be > {ADVANTAGE_SCALE_EPS}, got {scale}"
        )
    scaled = raw / scale
    clipped = max(float(ADVANTAGE_CLIP_MIN), min(float(ADVANTAGE_CLIP_MAX), scaled))
    return float(clipped)


def _summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("values must be non-empty")
    clean = [_coerce_finite_float(v, context="summarize_values") for v in values]
    zero_count = sum(1 for v in clean if float(v) == 0.0)
    clip_count = sum(
        1
        for v in clean
        if float(v) <= float(ADVANTAGE_CLIP_MIN)
        or float(v) >= float(ADVANTAGE_CLIP_MAX)
    )
    return {
        "count": float(len(clean)),
        "min": float(min(clean)),
        "max": float(max(clean)),
        "mean": float(sum(clean) / float(len(clean))),
        "p50": float(_linear_quantile(clean, 0.50)),
        "p95": float(_linear_quantile(clean, 0.95)),
        "zero_ratio": float(zero_count) / float(len(clean)),
        "clip_ratio": float(clip_count) / float(len(clean)),
    }


def _build_advantage_contract_metadata(
    *,
    source_iter_tag: str,
    n_samples: int,
    p95_abs_advantage: float,
    raw_advantages: list[float],
    scaled_advantages: list[float],
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "contract_version": str(ADVANTAGE_CONTRACT_VERSION),
        "raw_columns": ["recap_m2.return_G", "recap_m2.value_V", ADVANTAGE_RAW_COLUMN],
        "model_advantage_column": str(ADVANTAGE_INPUT_COLUMN),
        "task_text_field": "prompt_raw",
        "value_source": "critic",
        "critic_dir": None,
        "critic_include_t": True,
        "abs_quantile": float(ADVANTAGE_SCALE_ABS_QUANTILE),
        "p95_abs_advantage": float(p95_abs_advantage),
        "clip_min": float(ADVANTAGE_CLIP_MIN),
        "clip_max": float(ADVANTAGE_CLIP_MAX),
        "raw_summary": _summarize_values(list(raw_advantages)),
        "scaled_summary": _summarize_values(list(scaled_advantages)),
        "source_iter_tag": str(source_iter_tag),
        "n_samples": int(n_samples),
        "failure_policy": "hard_error_on_missing_nan_inf_or_out_of_range",
        "legacy_stats_snapshot": {
            "value_source": "critic",
            "raw_min": float(min(raw_advantages)),
            "raw_p50": float(_linear_quantile(raw_advantages, 0.50)),
            "raw_p95": float(_linear_quantile(raw_advantages, 0.95)),
            "raw_max": float(max(raw_advantages)),
            "raw_mean": float(sum(raw_advantages) / float(len(raw_advantages))),
            "raw_p95_abs": float(p95_abs_advantage),
            "scaled_min": float(min(scaled_advantages)),
            "scaled_p50": float(_linear_quantile(scaled_advantages, 0.50)),
            "scaled_p95": float(_linear_quantile(scaled_advantages, 0.95)),
            "scaled_max": float(max(scaled_advantages)),
            "scaled_mean": float(
                sum(scaled_advantages) / float(len(scaled_advantages))
            ),
            "scaled_zero_ratio": float(
                sum(1 for v in scaled_advantages if float(v) == 0.0)
            )
            / float(len(scaled_advantages)),
            "scaled_clip_ratio": float(
                sum(
                    1
                    for v in scaled_advantages
                    if abs(float(v)) >= float(ADVANTAGE_CLIP_MAX)
                )
            )
            / float(len(scaled_advantages)),
        },
    }
    contract.update(
        build_diagnostic_surface_metadata(
            surface_route=CONTINUOUS_ADVANTAGE_CONTRACT_DIAGNOSTIC_ROUTE,
            authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
            surface_kind="continuous_advantage_contract",
        )
    )
    return contract


def _load_sample_parquet_columns(dataset_path: Path) -> list[str]:
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError:
        return []
    parquet_paths = sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        return []
    df = pd.read_parquet(parquet_paths[0])
    return [str(col) for col in df.columns]


def _estimate_num_shards(episode_lengths: list[int]) -> int:
    total_effective_steps = sum(
        max(0, int(length) - int(UPSTREAM_ACTION_HORIZON) + 1)
        for length in episode_lengths
    )
    if total_effective_steps <= 0:
        return 0
    return int(
        math.ceil(float(total_effective_steps) / float(UPSTREAM_SMOKE_SHARD_SIZE))
    )


def _collect_preparation_reasons(
    dataset_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    info_path = dataset_path / "meta" / "info.json"
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    info = _read_json(info_path)
    episodes = _read_jsonl(episodes_path)
    features_raw = info.get("features")
    features = dict(features_raw) if isinstance(features_raw, dict) else {}
    sample_columns = _load_sample_parquet_columns(dataset_path)
    episode_lengths = [int(ep.get("length", 0)) for ep in episodes]
    estimated_num_shards = _estimate_num_shards(episode_lengths)
    num_splits = max(1, int(round(1.0 / float(UPSTREAM_EPISODE_SAMPLING_RATE))))
    max_nonempty_shards = len(episodes) * num_splits
    reasons: list[str] = []
    if ADVANTAGE_INPUT_COLUMN not in features:
        reasons.append("missing_advantage_input_feature")
    if ADVANTAGE_INPUT_COLUMN not in sample_columns:
        reasons.append("missing_advantage_input_column")
    if not isinstance(info.get("recap_advantage_input_contract"), dict):
        reasons.append("missing_advantage_input_contract")
    if estimated_num_shards > max_nonempty_shards:
        reasons.append("unsafe_for_upstream_smoke_shards")
    summary = {
        "source_dataset_path": str(dataset_path),
        "episode_count": int(len(episodes)),
        "episode_lengths": [int(v) for v in episode_lengths],
        "estimated_num_shards": int(estimated_num_shards),
        "max_nonempty_shards": int(max_nonempty_shards),
        "sample_parquet_columns": sample_columns,
        "source_has_advantage_input_feature": bool(ADVANTAGE_INPUT_COLUMN in features),
        "source_has_advantage_input_column": bool(
            ADVANTAGE_INPUT_COLUMN in sample_columns
        ),
        "source_has_advantage_contract": bool(
            isinstance(info.get("recap_advantage_input_contract"), dict)
        ),
    }
    return reasons, summary


def _prepare_finetune_ready_dataset(
    *, source_dataset_path: Path, max_episode_rows: int
) -> tuple[Path, dict[str, Any]]:
    import pandas as pd  # type: ignore

    if int(max_episode_rows) <= int(UPSTREAM_ACTION_HORIZON):
        raise ValueError(
            "max_episode_rows must be greater than upstream action horizon: "
            f"{max_episode_rows} <= {UPSTREAM_ACTION_HORIZON}"
        )

    prepared_dataset_path = source_dataset_path.parent / (
        source_dataset_path.name + DEFAULT_PREPARED_DATASET_SUFFIX
    )
    _remove_path(prepared_dataset_path)
    (prepared_dataset_path / "meta").mkdir(parents=True, exist_ok=True)
    (prepared_dataset_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    source_info = _read_json(source_dataset_path / "meta" / "info.json")
    source_episodes = _read_jsonl(source_dataset_path / "meta" / "episodes.jsonl")
    source_video_map_path = source_dataset_path / "meta" / "video_map.json"
    source_tasks_path = source_dataset_path / "meta" / "tasks.jsonl"
    source_modality_path = source_dataset_path / "meta" / "modality.json"

    prepared_episodes: list[dict[str, Any]] = []
    prepared_rows_by_episode: dict[int, int] = {}
    raw_advantages: list[float] = []
    parquet_cache: dict[int, Any] = {}

    for episode in source_episodes:
        episode_index = _require_int_field(
            episode,
            "episode_index",
            context=f"source episode in {source_dataset_path / 'meta' / 'episodes.jsonl'}",
        )
        parquet_path = (
            source_dataset_path
            / "data"
            / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"Missing source parquet for episode {episode_index}: {parquet_path}"
            )
        df = pd.read_parquet(parquet_path)
        source_length = min(int(episode.get("length", len(df))), int(len(df)))
        new_length = min(int(max_episode_rows), int(source_length))
        if new_length <= int(UPSTREAM_ACTION_HORIZON):
            raise ValueError(
                "Prepared finetune dataset would still be too short for upstream action horizon: "
                f"episode_index={episode_index} new_length={new_length}"
            )
        df = df.iloc[:new_length].copy()
        if ADVANTAGE_RAW_COLUMN not in df.columns:
            raise KeyError(
                f"Prepared finetune dataset requires {ADVANTAGE_RAW_COLUMN!r} in source parquet {parquet_path}"
            )
        raw_advantages.extend(
            [
                _coerce_finite_float(
                    v, context=f"{ADVANTAGE_RAW_COLUMN}[episode={episode_index}]"
                )
                for v in df[ADVANTAGE_RAW_COLUMN].tolist()
            ]
        )
        parquet_cache[episode_index] = df
        prepared_rows_by_episode[episode_index] = int(new_length)
        updated_episode = dict(episode)
        updated_episode["length"] = int(new_length)
        prepared_episodes.append(updated_episode)

    if not raw_advantages:
        raise ValueError("Prepared finetune dataset has no advantage_A values")

    p95_abs_advantage = _compute_p95_abs_advantage(raw_advantages)
    scaled_advantages: list[float] = []
    for episode in prepared_episodes:
        episode_index = int(episode["episode_index"])
        df = parquet_cache[episode_index]
        scaled = [
            _scale_advantage_input(v, p95_abs_advantage=p95_abs_advantage)
            for v in df[ADVANTAGE_RAW_COLUMN].tolist()
        ]
        df[ADVANTAGE_INPUT_COLUMN] = pd.Series(scaled, index=df.index, dtype="float32")
        scaled_advantages.extend(float(v) for v in scaled)
        dst = (
            prepared_dataset_path
            / "data"
            / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        df.to_parquet(dst, index=False)

    prepared_total_frames = int(sum(int(ep["length"]) for ep in prepared_episodes))
    advantage_contract = _build_advantage_contract_metadata(
        source_iter_tag=str(source_dataset_path.name),
        n_samples=int(len(raw_advantages)),
        p95_abs_advantage=float(p95_abs_advantage),
        raw_advantages=list(raw_advantages),
        scaled_advantages=list(scaled_advantages),
    )

    info_obj = dict(source_info)
    features_raw = info_obj.get("features")
    features = dict(features_raw) if isinstance(features_raw, dict) else {}
    features[ADVANTAGE_INPUT_COLUMN] = {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }
    info_obj["features"] = features
    info_obj["total_frames"] = int(prepared_total_frames)
    info_obj["total_episodes"] = int(len(prepared_episodes))
    info_obj["total_videos"] = int(len(prepared_episodes))
    info_obj["task_text_field"] = "prompt_raw"
    info_obj["task_text_mode"] = "prompt_raw"
    info_obj["recap_export.dual_task_text"] = False
    info_obj["recap_advantage_input_contract"] = advantage_contract
    info_obj["45c_finetune_smoke_prepare"] = {
        "source_dataset_path": str(source_dataset_path),
        "prepared_dataset_path": str(prepared_dataset_path),
        "max_episode_rows": int(max_episode_rows),
        "upstream_shard_size": int(UPSTREAM_SMOKE_SHARD_SIZE),
        "upstream_episode_sampling_rate": float(UPSTREAM_EPISODE_SAMPLING_RATE),
        "upstream_action_horizon": int(UPSTREAM_ACTION_HORIZON),
        "prepared_rows_by_episode": {
            str(k): int(v) for k, v in sorted(prepared_rows_by_episode.items())
        },
    }
    _write_json(prepared_dataset_path / "meta" / "info.json", info_obj)
    _write_jsonl(prepared_dataset_path / "meta" / "episodes.jsonl", prepared_episodes)
    _copy_text_file(source_tasks_path, prepared_dataset_path / "meta" / "tasks.jsonl")
    _copy_text_file(
        source_modality_path, prepared_dataset_path / "meta" / "modality.json"
    )

    if source_video_map_path.is_file():
        video_map = _read_json(source_video_map_path)
        dest = video_map.get("dest")
        if isinstance(dest, dict):
            dest = dict(dest)
            dest["dataset_dir"] = str(prepared_dataset_path)
            dest["videos_root"] = str(prepared_dataset_path / "videos")
            video_map["dest"] = dest
        video_map["iter_tag"] = str(prepared_dataset_path.name)
        video_map["45c_finetune_smoke_prepare"] = info_obj["45c_finetune_smoke_prepare"]
        _write_json(prepared_dataset_path / "meta" / "video_map.json", video_map)

    source_videos = source_dataset_path / "videos"
    prepared_videos = prepared_dataset_path / "videos"
    if source_videos.is_dir():
        os.symlink(source_videos, prepared_videos, target_is_directory=True)

    prep_info = {
        "prepared_dataset_path": str(prepared_dataset_path),
        "source_dataset_path": str(source_dataset_path),
        "max_episode_rows": int(max_episode_rows),
        "prepared_total_frames": int(prepared_total_frames),
        "prepared_total_episodes": int(len(prepared_episodes)),
        "p95_abs_advantage": float(p95_abs_advantage),
        "prepared_rows_by_episode": {
            str(k): int(v) for k, v in sorted(prepared_rows_by_episode.items())
        },
        "advantage_input_summary": _summarize_values(list(scaled_advantages)),
    }
    return prepared_dataset_path, prep_info


def _resolve_effective_dataset_path(
    *, dataset_path: Path, max_episode_rows: int
) -> tuple[Path, dict[str, Any]]:
    reasons, source_summary = _collect_preparation_reasons(dataset_path)
    if not reasons:
        return dataset_path, {
            "preparation_applied": False,
            "preparation_reasons": [],
            "source_summary": source_summary,
            "effective_dataset_path": str(dataset_path),
        }
    effective_dataset_path, prep_info = _prepare_finetune_ready_dataset(
        source_dataset_path=dataset_path,
        max_episode_rows=int(max_episode_rows),
    )
    return effective_dataset_path, {
        "preparation_applied": True,
        "preparation_reasons": list(reasons),
        "source_summary": source_summary,
        "prepared_summary": prep_info,
        "effective_dataset_path": str(effective_dataset_path),
    }


def _site_packages_path(main_repo_root: Path) -> Path:
    candidates = sorted(
        (main_repo_root / ".envs" / "main" / "lib").glob("python*/site-packages")
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate main-repo site-packages under {main_repo_root / '.envs' / 'main' / 'lib'}"
        )
    return candidates[-1]


def _probe_python_imports(python_exe: Path, env: dict[str, str]) -> tuple[bool, str]:
    probe_cmd = [
        str(python_exe),
        "-c",
        (
            "import torch, transformers; "
            "print('BRIDGE_PROBE_OK', torch.__version__, transformers.__version__)"
        ),
    ]
    proc = subprocess.run(
        probe_cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    preview = (proc.stdout or proc.stderr or "").strip()
    if len(preview) > 500:
        preview = preview[:500]
    return bool(proc.returncode == 0), preview


def _build_upstream_env(
    *, main_repo_root: Path, python_exe: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    requested = str(python_exe)
    resolved = str(python_exe.resolve())
    import_ok, preview = _probe_python_imports(python_exe, env)
    bridge_info: dict[str, Any] = {
        "python_requested": requested,
        "python_resolved": resolved,
        "probe_import_ok_before_bridge": bool(import_ok),
        "probe_preview_before_bridge": preview,
        "bridge_mode": "direct",
        "site_packages_path": None,
        "probe_import_ok_after_bridge": bool(import_ok),
        "probe_preview_after_bridge": preview,
    }
    if import_ok:
        return env, bridge_info

    site_packages = _site_packages_path(main_repo_root)
    existing = str(env.get("PYTHONPATH", "")).strip()
    env["PYTHONPATH"] = (
        f"{site_packages}{os.pathsep}{existing}" if existing else str(site_packages)
    )
    import_ok_after, preview_after = _probe_python_imports(python_exe, env)
    bridge_info.update(
        {
            "bridge_mode": "main_repo_venv_site_packages",
            "site_packages_path": str(site_packages),
            "probe_import_ok_after_bridge": bool(import_ok_after),
            "probe_preview_after_bridge": preview_after,
        }
    )
    return env, bridge_info


def _selected_checkpoint_asset(checkpoint_dir: Path | None) -> Path | None:
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return None
    candidates = [
        checkpoint_dir / "model.safetensors.index.json",
        checkpoint_dir / "model.safetensors",
        checkpoint_dir / "pytorch_model.bin.index.json",
        checkpoint_dir / "pytorch_model.bin",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _latest_checkpoint(output_dir: Path) -> Path | None:
    latest: tuple[int, float, Path] | None = None
    for path in sorted(output_dir.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        suffix = path.name.split("checkpoint-", 1)[-1]
        if not suffix.isdigit():
            continue
        candidate = (int(suffix), float(path.stat().st_mtime), path)
        if latest is None or candidate[:2] > latest[:2]:
            latest = candidate
    return None if latest is None else latest[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="45c_vlm_critic_finetune_smoke.py",
        description=(
            "Branch-local wrapper around main-repo 3D_recap_finetune_full.py. "
            "All writable outputs stay inside the delegated worktree."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--runtime-log-dir", type=str, default=DEFAULT_RUNTIME_LOG_DIR)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--base-model-path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--embodiment-tag", type=str, default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--max-steps", type=int, default=int(DEFAULT_MAX_STEPS))
    parser.add_argument("--save-steps", type=int, default=int(DEFAULT_SAVE_STEPS))
    parser.add_argument("--seed", type=int, default=int(DEFAULT_SEED))
    parser.add_argument(
        "--learning-rate", type=float, default=float(DEFAULT_LEARNING_RATE)
    )
    parser.add_argument(
        "--global-batch-size", type=int, default=int(DEFAULT_GLOBAL_BATCH_SIZE)
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(DEFAULT_GRADIENT_ACCUMULATION_STEPS),
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=int(DEFAULT_DATALOADER_NUM_WORKERS),
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--log-every", type=int, default=int(DEFAULT_LOG_EVERY))
    parser.add_argument(
        "--max-episode-rows-for-prep",
        type=int,
        default=int(DEFAULT_MAX_EPISODE_ROWS_FOR_PREP),
        help=(
            "When wrapper-side dataset preparation is needed, truncate each episode parquet "
            "to at most this many rows so upstream shard_size=128 stays non-empty."
        ),
    )
    parser.add_argument(
        "--transformers-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--main-repo-root", type=str, default=DEFAULT_MAIN_REPO_ROOT)
    parser.add_argument("--python", type=str, default=DEFAULT_MAIN_REPO_PYTHON)
    parser.add_argument("--upstream-script", type=str, default=DEFAULT_UPSTREAM_SCRIPT)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = _repo_root()

    dataset_path = _resolve_path(repo_root, str(args.dataset_path))
    output_json = _resolve_path(repo_root, str(args.output_json))
    runtime_log_dir = _resolve_path(repo_root, str(args.runtime_log_dir))
    main_repo_root = Path(str(args.main_repo_root)).expanduser().resolve()
    python_exe = Path(str(args.python)).expanduser()
    upstream_script = Path(str(args.upstream_script)).expanduser().resolve()
    requested_dataset_path = dataset_path
    output_dir = (
        _resolve_path(repo_root, str(args.output_dir))
        if str(args.output_dir).strip()
        else _resolve_path(repo_root, DEFAULT_OUTPUT_DIR_PREFIX + _timestamp())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    upstream_summary_path = output_dir / "smoke_train_summary.json"
    effective_dataset_path, dataset_preparation = _resolve_effective_dataset_path(
        dataset_path=dataset_path,
        max_episode_rows=int(args.max_episode_rows_for_prep),
    )
    upstream_env, bridge_info = _build_upstream_env(
        main_repo_root=main_repo_root,
        python_exe=python_exe,
    )

    cmd = [
        str(python_exe),
        str(upstream_script),
        "--base-model-path",
        str(args.base_model_path),
        "--dataset-path",
        str(effective_dataset_path),
        "--embodiment-tag",
        str(args.embodiment_tag),
        "--runtime-log-dir",
        str(runtime_log_dir),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(int(args.max_steps)),
        "--save-steps",
        str(int(args.save_steps)),
        "--global-batch-size",
        str(int(args.global_batch_size)),
        "--gradient-accumulation-steps",
        str(int(args.gradient_accumulation_steps)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--dataloader-num-workers",
        str(int(args.dataloader_num_workers)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--log-every",
        str(int(args.log_every)),
    ]
    cmd.append(
        "--transformers-local-files-only"
        if bool(args.transformers_local_files_only)
        else "--no-transformers-local-files-only"
    )

    upstream_summary: dict[str, Any] | None = None
    error_text: str | None = None
    rc: int | None = None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(main_repo_root),
            env=upstream_env,
            check=False,
        )
        rc = int(proc.returncode)
        if upstream_summary_path.is_file():
            upstream_summary = _read_json(upstream_summary_path)
        if rc != 0:
            error_text = f"upstream_finetune_failed: returncode={rc}"
        elif upstream_summary is None:
            error_text = "upstream_finetune_missing_summary_json"
    except Exception as exc:
        error_text = f"wrapper_exception: {type(exc).__name__}: {exc}"

    selected_checkpoint = _latest_checkpoint(output_dir)
    selected_checkpoint_asset = _selected_checkpoint_asset(selected_checkpoint)
    payload: dict[str, Any] = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "wrapper": "45c_vlm_critic_finetune_smoke.py",
        "sentinel": PASS_SENTINEL,
        "wrapper_status": "ok" if error_text is None else "blocked",
        "output_json": str(output_json),
        "main_repo_root": str(main_repo_root),
        "python": str(python_exe),
        "interpreter_bridge": bridge_info,
        "upstream_script": str(upstream_script),
        "upstream_cmd": cmd,
        "upstream_returncode": rc,
        "dataset_path": str(requested_dataset_path),
        "effective_dataset_path": str(effective_dataset_path),
        "dataset_preparation": dataset_preparation,
        "base_model_path": str(args.base_model_path),
        "embodiment_tag": str(args.embodiment_tag),
        "runtime_log_dir": str(runtime_log_dir),
        "output_dir": str(output_dir),
        "upstream_summary_path": str(upstream_summary_path),
        "upstream_summary_exists": bool(upstream_summary_path.is_file()),
        "selected_checkpoint_path": (
            str(selected_checkpoint)
            if selected_checkpoint is not None and selected_checkpoint_asset is not None
            else None
        ),
        "selected_checkpoint_exists": bool(selected_checkpoint_asset is not None),
        "selected_checkpoint_asset_path": (
            str(selected_checkpoint_asset)
            if selected_checkpoint_asset is not None
            else None
        ),
        "upgrade_pending": UPGRADE_PENDING,
        "requested_hparams": {
            "max_steps": int(args.max_steps),
            "save_steps": int(args.save_steps),
            "seed": int(args.seed),
            "learning_rate": float(args.learning_rate),
            "global_batch_size": int(args.global_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "dataloader_num_workers": int(args.dataloader_num_workers),
            "device": str(args.device),
            "log_every": int(args.log_every),
            "max_episode_rows_for_prep": int(args.max_episode_rows_for_prep),
            "transformers_local_files_only": bool(args.transformers_local_files_only),
        },
        "upstream_summary": upstream_summary,
        "error": error_text,
    }
    payload.update(
        build_diagnostic_surface_metadata(
            surface_route="vlm_critic_finetune_smoke_diagnostic",
            authority_scope=VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE,
            surface_kind="vlm_critic_finetune_smoke_summary",
        )
    )
    _write_json(output_json, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    print(f"SENTINEL:{PASS_SENTINEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
