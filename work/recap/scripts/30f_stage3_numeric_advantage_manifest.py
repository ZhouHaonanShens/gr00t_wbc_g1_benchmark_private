#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


_REPO_ROOT_FOR_IMPORT = _repo_root()
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return {str(k): v for k, v in payload.items()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write('\n')
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Materialize numeric advantage manifest from exported LeRobot dataset.')
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--seed-set', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser


def _choose_eval_advantage_scalar(advantage_series: Any, stats: dict[str, float]) -> tuple[float, str]:
    p50 = float(stats['p50'])
    if p50 > 0.0:
        return p50, 'dataset_p50_numeric_advantage_input_v1'

    positive_series = advantage_series[advantage_series > 0.0]
    if int(getattr(positive_series, 'shape', [len(positive_series)])[0]) > 0:
        positive_median = float(positive_series.quantile(0.50, interpolation='linear'))
        if positive_median > 0.0:
            return positive_median, 'dataset_positive_subset_p50_numeric_advantage_input_v1'

        positive_max = float(positive_series.max())
        if positive_max > 0.0:
            return positive_max, 'dataset_positive_subset_max_numeric_advantage_input_v1'

    return p50, 'dataset_p50_fallback_numeric_advantage_input_v1'


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    dataset_path = _resolve_path(repo_root, args.dataset_path)
    baseline_checkpoint = _resolve_path(repo_root, args.baseline_checkpoint)
    seed_set_path = _resolve_path(repo_root, args.seed_set)
    output_dir = _resolve_path(repo_root, args.output_dir)

    import pandas as pd  # type: ignore

    from work.recap.advantage import ADVANTAGE_INPUT_COLUMN, extract_advantage_contract

    info_path = dataset_path / 'meta' / 'info.json'
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = _read_json(info_path)
    contract = extract_advantage_contract(info)

    features = info.get('features')
    if not isinstance(features, dict) or ADVANTAGE_INPUT_COLUMN not in features:
        raise KeyError(f'missing {ADVANTAGE_INPUT_COLUMN!r} feature in {info_path}')

    parquet_files = tuple(sorted((dataset_path / 'data').glob('chunk-*/episode_*.parquet')))
    if not parquet_files:
        raise FileNotFoundError(f'no parquet files under {dataset_path / "data"}')

    frames: list[Any] = []
    rel_paths: list[str] = []
    for parquet_path in parquet_files:
        frame = pd.read_parquet(parquet_path)
        if ADVANTAGE_INPUT_COLUMN not in frame.columns:
            raise KeyError(f'missing {ADVANTAGE_INPUT_COLUMN!r} in {parquet_path}')
        frame = frame.copy()
        frame['source_parquet_rel'] = str(parquet_path.relative_to(dataset_path).as_posix())
        frames.append(frame)
        rel_paths.append(str(parquet_path.relative_to(dataset_path).as_posix()))

    full_frame = pd.concat(frames, ignore_index=True)
    advantage_series = pd.to_numeric(full_frame[ADVANTAGE_INPUT_COLUMN], errors='coerce').astype(float)
    all_labels_finite = bool(advantage_series.notna().all() and advantage_series.map(math.isfinite).all())
    if not all_labels_finite:
        raise ValueError('exported recap_m2.advantage_input contains NaN/Inf')

    stats = {
        'min': float(advantage_series.min()),
        'p05': float(advantage_series.quantile(0.05, interpolation='linear')),
        'p50': float(advantage_series.quantile(0.50, interpolation='linear')),
        'p75': float(advantage_series.quantile(0.75, interpolation='linear')),
        'p95': float(advantage_series.quantile(0.95, interpolation='linear')),
        'max': float(advantage_series.max()),
        'mean': float(advantage_series.mean()),
        'std': float(advantage_series.std(ddof=0)),
    }

    eval_advantage_scalar, eval_advantage_scalar_policy = _choose_eval_advantage_scalar(
        advantage_series,
        stats,
    )
    eval_advantage_scalar_in_distribution = bool(stats['min'] <= eval_advantage_scalar <= stats['max'])
    if not eval_advantage_scalar_in_distribution:
        raise ValueError('eval_advantage_scalar fell outside dataset distribution')

    checkpoint_sha_source = baseline_checkpoint / 'model.safetensors.index.json'
    if not checkpoint_sha_source.is_file():
        checkpoint_sha_source = baseline_checkpoint / 'trainer_state.json'
    if not checkpoint_sha_source.is_file():
        raise FileNotFoundError(f'no checkpoint hash source file under {baseline_checkpoint}')

    seed_set = _read_json(seed_set_path)
    seeds_raw = seed_set.get('seeds')
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError(f'invalid seed set seeds in {seed_set_path}')

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_out = output_dir / 'per_transition_advantage.parquet'
    manifest_out = output_dir / 'advantage_label_manifest.json'

    desired_cols = [
        'episode_index',
        'index',
        'timestamp',
        'recap_m2.t',
        'recap_m2.return_G',
        'recap_m2.value_V',
        'recap_m2.advantage_A',
        'recap_m2.advantage_input',
        'recap_m2.indicator_I',
        'source_parquet_rel',
    ]
    kept_cols = [col for col in desired_cols if col in full_frame.columns]
    full_frame.loc[:, kept_cols].to_parquet(parquet_out, engine='pyarrow', index=False)

    manifest = {
        'schema_version': 'stage3_numeric_advantage_manifest_v1',
        'dataset_path': str(dataset_path),
        'dataset_info_path': str(info_path),
        'label_source_mode': 'dataset_native_numeric_advantage_input',
        'advantage_input_column': ADVANTAGE_INPUT_COLUMN,
        'checkpoint_source_path': str(baseline_checkpoint),
        'checkpoint_source_sha': _sha256_file(checkpoint_sha_source),
        'checkpoint_source_sha_file': str(checkpoint_sha_source),
        'seed_set_path': str(seed_set_path),
        'seed_count': int(len(seeds_raw)),
        'formal_eval_episodes': int(seed_set.get('formal_eval_episodes', len(seeds_raw))),
        'seed_base': int(seed_set.get('seed_base', seeds_raw[0])),
        'row_count': int(len(full_frame)),
        'parquet_file_count': int(len(parquet_files)),
        'source_parquet_files': rel_paths,
        'all_labels_finite': bool(all_labels_finite),
        'advantage_stats': stats,
        'eval_advantage_scalar': float(eval_advantage_scalar),
        'eval_advantage_scalar_policy': eval_advantage_scalar_policy,
        'eval_advantage_scalar_in_distribution': bool(eval_advantage_scalar_in_distribution),
        'recap_advantage_input_contract': dict(contract),
        'output_parquet': str(parquet_out),
    }
    _write_json(manifest_out, manifest)

    print('[INFO] dataset_path:', dataset_path)
    print('[INFO] output_parquet:', parquet_out)
    print('[INFO] output_manifest:', manifest_out)
    print('[INFO] row_count:', int(len(full_frame)))
    print('[INFO] eval_advantage_scalar:', float(eval_advantage_scalar))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
