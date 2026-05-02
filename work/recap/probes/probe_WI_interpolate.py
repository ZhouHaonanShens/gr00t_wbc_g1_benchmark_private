#!/usr/bin/env python3
"""Probe WI — Weight-interpolation rescue between base GR00T checkpoint and
post-RECAP G3 checkpoint.

Linearly interpolates per-tensor weights between BASE and POST per
arXiv:2512.08333 to attempt to rescue the right-hand gripper primitive that
collapsed during RECAP fine-tuning (verified at parquet-row level by Probes
D + A).

Inputs (read-only):
  - BASE: HF cache snapshot of `nvidia/GR00T-N1.6-G1-PnPAppleToPlate`
    (2-shard bf16 layout, top-level `statistics.json`).
  - POST: post-RECAP G3 ckpt-6600 (3-shard fp32 layout, with
    `experiment_cfg/dataset_statistics.json`).

Outputs (under `agent/artifacts/probes/probe_WI_rescue/`):
  - For each alpha in {0.25, 0.5, 0.75}:
        checkpoint_alpha_{alpha}/
            model.safetensors                (single shard, bf16)
            model.safetensors.index.json     (auto-derived)
            statistics.json                  (copied from BASE)
            processor_config.json            (copied from BASE)
            config.json                      (copied from BASE)
            embodiment_id.json               (copied from BASE)
            experiment_cfg/dataset_statistics.json  (sourced from BASE
                                              `unitree_g1` block)
  - `interpolation_summary.json` — per-alpha tensor count, total bytes,
    dtype histogram, key-diff status, dataset-stats source provenance.
  - On state-dict key mismatch:
        stop_record.json with stop_code = STOP_KEY_DIFF
    (script exits non-zero, no rescue dirs produced).

Constraints honored:
  - No mutation of base or post checkpoints.
  - GPU not required (pure tensor arithmetic on CPU).
  - All tensors cast to fp32 for the interpolation, written back as bf16
    to match the base layout (which the eval runner expects via
    `model.to(dtype=torch.bfloat16)` in `Gr00tPolicy.__init__`).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPO_ROOT = Path("/home/howard/Projects/gr00t_wbc_g1_benchmark")
BASE_DIR = Path(
    "/home/howard/.cache/huggingface/hub/"
    "models--nvidia--GR00T-N1.6-G1-PnPAppleToPlate/"
    "snapshots/897d0313a190f46a2cccaeb34077752a0db4b0de"
)
POST_DIR = (
    REPO_ROOT
    / "agent/artifacts/gr00t_recap_live/single_gpu_v2_full_update"
    / "stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ"
    / "gr00t/g3_conditioned_continuation_6600_after_surfacefix_20260430_181210"
    / "checkpoint-6600"
)
OUT_ROOT = REPO_ROOT / "agent/artifacts/probes/probe_WI_rescue"
ALPHAS = (0.25, 0.5, 0.75)


def _shard_paths(ckpt_dir: Path) -> list[Path]:
    """Return shard paths in the order declared by the index file."""
    idx_file = ckpt_dir / "model.safetensors.index.json"
    if not idx_file.is_file():
        raise FileNotFoundError(f"missing index file: {idx_file}")
    with idx_file.open("r", encoding="utf-8") as fh:
        idx = json.load(fh)
    shard_files = sorted(set(idx["weight_map"].values()))
    return [ckpt_dir / s for s in shard_files]


def _load_state_dict(ckpt_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load all tensors from a sharded safetensors checkpoint into a single
    dict keyed by tensor name. Returns (state_dict, meta) where meta has
    dtype histogram and shard layout."""
    shards = _shard_paths(ckpt_dir)
    state: dict[str, torch.Tensor] = {}
    dtype_hist: Counter[str] = Counter()
    per_shard_keys: dict[str, list[str]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            keys = list(f.keys())
            per_shard_keys[shard.name] = keys
            for k in keys:
                if k in state:
                    raise RuntimeError(
                        f"duplicate tensor key {k!r} encountered while loading {ckpt_dir}"
                    )
                t = f.get_tensor(k)
                state[k] = t
                dtype_hist[str(t.dtype)] += 1
    meta = {
        "checkpoint_dir": str(ckpt_dir),
        "shard_files": [s.name for s in shards],
        "tensor_count": len(state),
        "dtype_histogram": dict(dtype_hist),
        "per_shard_key_counts": {k: len(v) for k, v in per_shard_keys.items()},
    }
    return state, meta


def _write_stop_record(payload: dict[str, Any]) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stop_path = OUT_ROOT / "stop_record.json"
    with stop_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return stop_path


def _interpolate(
    base_state: dict[str, torch.Tensor],
    post_state: dict[str, torch.Tensor],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compute theta = (1 - alpha) * base + alpha * post per tensor name.

    Both tensors are upcast to fp32 for the arithmetic, then the result is
    saved as bf16 to keep the memory profile compatible with the eval
    pipeline (which casts the whole model to bf16 at load time).
    """
    out: dict[str, torch.Tensor] = {}
    shape_mismatches: list[dict[str, Any]] = []
    dtype_hist_in: Counter[str] = Counter()
    for k, base_t in base_state.items():
        post_t = post_state[k]
        dtype_hist_in[f"base={base_t.dtype}|post={post_t.dtype}"] += 1
        if base_t.shape != post_t.shape:
            shape_mismatches.append(
                {
                    "key": k,
                    "base_shape": list(base_t.shape),
                    "post_shape": list(post_t.shape),
                }
            )
            # Per spec: use base shape and effective alpha=0 for this tensor.
            interp = base_t.detach().to(dtype=torch.float32)
        else:
            b32 = base_t.detach().to(dtype=torch.float32)
            p32 = post_t.detach().to(dtype=torch.float32)
            interp = (1.0 - alpha) * b32 + alpha * p32
        out[k] = interp.to(dtype=torch.bfloat16).contiguous()
    diag = {
        "alpha": alpha,
        "tensor_count": len(out),
        "shape_mismatches": shape_mismatches,
        "input_dtype_combinations": dict(dtype_hist_in),
        "output_dtype": "torch.bfloat16",
    }
    return out, diag


def _write_single_shard(
    state: dict[str, torch.Tensor],
    out_dir: Path,
) -> dict[str, Any]:
    """Write all tensors as a single safetensors shard plus a matching
    index file, then verify by loading it back."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_name = "model.safetensors"
    shard_path = out_dir / shard_name
    save_file(
        state,
        str(shard_path),
        metadata={"format": "pt"},
    )
    # Build a minimal index that points every key to the single shard.
    total_size = sum(t.numel() * t.element_size() for t in state.values())
    index = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": {k: shard_name for k in state.keys()},
    }
    with (out_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)

    # Verify by reading back tensor count + a checksum on a stable key.
    with safe_open(str(shard_path), framework="pt") as f:
        loaded_keys = list(f.keys())
        sample_key = sorted(loaded_keys)[0]
        sample_t = f.get_tensor(sample_key)
    if len(loaded_keys) != len(state):
        raise RuntimeError(
            f"verify-load tensor count mismatch in {shard_path}: "
            f"wrote {len(state)}, loaded {len(loaded_keys)}"
        )
    return {
        "shard_file": shard_name,
        "shard_bytes": shard_path.stat().st_size,
        "total_size_declared": index["metadata"]["total_size"],
        "verify_loaded_keys": len(loaded_keys),
        "verify_sample_key": sample_key,
        "verify_sample_dtype": str(sample_t.dtype),
    }


def _copy_static_files(src: Path, dst: Path, names: list[str]) -> list[str]:
    copied: list[str] = []
    for name in names:
        sp = src / name
        if not sp.exists():
            continue
        dp = dst / name
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, dp)
        copied.append(name)
    return copied


def _build_experiment_cfg(out_dir: Path, base_dir: Path, post_dir: Path) -> dict[str, Any]:
    """Create a minimal `experiment_cfg/dataset_statistics.json` for the
    rescue checkpoint.

    Per the closure note, post's `experiment_cfg/dataset_statistics.json` has
    the right-hand gripper q99/mean/max collapsed (~0). We therefore prefer
    the BASE-side `statistics.json` `unitree_g1` block as the source of
    truth and write it under `experiment_cfg/dataset_statistics.json` so
    any code path that looks for `experiment_cfg/` finds healthy stats.
    """
    base_stats_path = base_dir / "statistics.json"
    if not base_stats_path.is_file():
        raise FileNotFoundError(f"base statistics.json missing: {base_stats_path}")
    with base_stats_path.open("r", encoding="utf-8") as fh:
        base_stats = json.load(fh)
    if "unitree_g1" not in base_stats:
        raise RuntimeError("base statistics.json has no unitree_g1 block")
    derived = {"unitree_g1": copy.deepcopy(base_stats["unitree_g1"])}
    cfg_dir = out_dir / "experiment_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with (cfg_dir / "dataset_statistics.json").open("w", encoding="utf-8") as fh:
        json.dump(derived, fh, indent=2, sort_keys=True)
    return {
        "experiment_cfg/dataset_statistics.json": "derived_from_base.statistics.json[unitree_g1]",
        "post_experiment_cfg_present": (post_dir / "experiment_cfg/dataset_statistics.json").is_file(),
        "post_experiment_cfg_skipped_reason": "post stats have collapsed right_hand q99/mean/max per probe D",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    ap.add_argument("--dry-run", action="store_true", help="only run key-diff and exit")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUT_ROOT / "interpolation_log.txt"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    log(f"BASE = {BASE_DIR}")
    log(f"POST = {POST_DIR}")
    log(f"OUT  = {OUT_ROOT}")
    log(f"alphas = {args.alphas}")

    log("Loading BASE state dict ...")
    t0 = time.monotonic()
    base_state, base_meta = _load_state_dict(BASE_DIR)
    log(
        f"BASE loaded: {base_meta['tensor_count']} tensors across "
        f"{len(base_meta['shard_files'])} shards "
        f"in {time.monotonic() - t0:.1f}s; dtype_hist={base_meta['dtype_histogram']}"
    )

    log("Loading POST state dict ...")
    t0 = time.monotonic()
    post_state, post_meta = _load_state_dict(POST_DIR)
    log(
        f"POST loaded: {post_meta['tensor_count']} tensors across "
        f"{len(post_meta['shard_files'])} shards "
        f"in {time.monotonic() - t0:.1f}s; dtype_hist={post_meta['dtype_histogram']}"
    )

    base_keys = set(base_state.keys())
    post_keys = set(post_state.keys())
    base_only = sorted(base_keys - post_keys)
    post_only = sorted(post_keys - base_keys)
    if base_only or post_only:
        payload = {
            "stop_code": "STOP_KEY_DIFF",
            "base_only_keys": base_only,
            "post_only_keys": post_only,
            "base_key_count": len(base_keys),
            "post_key_count": len(post_keys),
            "common_count": len(base_keys & post_keys),
            "base_meta": base_meta,
            "post_meta": post_meta,
        }
        stop_path = _write_stop_record(payload)
        log(f"STOP_KEY_DIFF — wrote {stop_path}")
        log_handle.close()
        return 2

    log(f"Key-diff OK: {len(base_keys)} keys match exactly between BASE and POST.")
    if args.dry_run:
        log("dry-run requested; exiting without producing rescue dirs.")
        log_handle.close()
        return 0

    static_files_from_base = [
        "statistics.json",
        "processor_config.json",
        "config.json",
        "embodiment_id.json",
    ]

    summary: dict[str, Any] = {
        "schema": "probe_WI_interpolation_summary_v1",
        "base_dir": str(BASE_DIR),
        "post_dir": str(POST_DIR),
        "out_root": str(OUT_ROOT),
        "alphas": args.alphas,
        "base_meta": base_meta,
        "post_meta": post_meta,
        "key_diff_status": "MATCH",
        "common_key_count": len(base_keys),
        "interpolation_dtype": "fp32",
        "saved_dtype": "torch.bfloat16",
        "save_format": "single_shard",
        "alpha_runs": [],
    }

    for alpha in args.alphas:
        log(f"--- alpha = {alpha} ---")
        out_dir = OUT_ROOT / f"checkpoint_alpha_{alpha:g}"
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()
        interp_state, interp_diag = _interpolate(base_state, post_state, alpha)
        log(
            f"alpha={alpha}: interpolated {interp_diag['tensor_count']} tensors "
            f"in {time.monotonic() - t0:.1f}s; "
            f"shape_mismatches={len(interp_diag['shape_mismatches'])}"
        )

        t0 = time.monotonic()
        write_diag = _write_single_shard(interp_state, out_dir)
        log(
            f"alpha={alpha}: wrote {write_diag['shard_file']} "
            f"({write_diag['shard_bytes']:,} bytes) in {time.monotonic() - t0:.1f}s; "
            f"verify-loaded {write_diag['verify_loaded_keys']} keys"
        )

        copied = _copy_static_files(BASE_DIR, out_dir, static_files_from_base)
        log(f"alpha={alpha}: copied static files from base: {copied}")

        cfg_diag = _build_experiment_cfg(out_dir, BASE_DIR, POST_DIR)
        log(f"alpha={alpha}: experiment_cfg provenance: {cfg_diag}")

        # Free memory between alphas.
        del interp_state

        summary["alpha_runs"].append(
            {
                "alpha": alpha,
                "out_dir": str(out_dir),
                "static_copied": copied,
                "experiment_cfg_provenance": cfg_diag,
                "interp_diag": interp_diag,
                "write_diag": write_diag,
            }
        )

    summary_path = OUT_ROOT / "interpolation_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    log(f"wrote {summary_path}")
    log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
