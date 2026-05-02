#!/usr/bin/env python3
"""Probe D: replay formal_eval with full per-dim signed action logging.

Monkey-patches helper3d._summarize_action_chunk to embed full signed arrays
into the per-step `action_summary` field of the telemetry JSONL emitted by
gr00t_g3_formal_eval.

Invocation pattern (one process per checkpoint):

    CUDA_VISIBLE_DEVICES=1 python -m work.recap.probes.probe_D_action_replay \\
        --checkpoint-tag post_p0a \\
        --checkpoint-path <abs path to checkpoint dir> \\
        --episode-count 10 --indicator-modes positive,omit,negative \\
        --output-dir <abs replay root>

Action-summary record schema (after monkey patch), per modality key:
    {
      "shape": [B, T, D],
      "mean_abs": float,         # original lossy stats kept for back-compat
      "max_abs": float,
      "p95_abs": float,
      "abs_preview": [...],
      "full_signed_first_chunk": [[...] * D],  # B=0, T=0..min(8,T), all D dims, signed
      "per_dim_mean_signed": [...],            # mean signed across (B,T) per dim
      "per_dim_first_step_signed": [...],      # b=0, t=0, all D dims signed
    }

This wrapper does NOT mutate the runner script on disk. The patch lives
in this process only.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/howard/Projects/gr00t_wbc_g1_benchmark")


def _load_helper3d_module():
    spec = importlib.util.spec_from_file_location(
        "_3D_recap_eval", str(REPO_ROOT / "work" / "recap" / "scripts" / "3D_recap_eval.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load helper3d from 3D_recap_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patched_summarize_action_chunk(action: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    summary: dict[str, Any] = {}
    for key, value in action.items():
        arr = np.asarray(value, dtype=np.float32)
        flat_abs = np.abs(arr.reshape(-1)) if int(arr.size) > 0 else np.asarray([])
        preview = flat_abs[: min(6, int(flat_abs.size))].tolist() if int(flat_abs.size) > 0 else []
        rec: dict[str, Any] = {
            "shape": [int(x) for x in arr.shape],
            "mean_abs": float(np.mean(flat_abs)) if int(flat_abs.size) > 0 else 0.0,
            "max_abs": float(np.max(flat_abs)) if int(flat_abs.size) > 0 else 0.0,
            "p95_abs": float(np.quantile(flat_abs, 0.95)) if int(flat_abs.size) > 0 else 0.0,
            "abs_preview": [float(v) for v in preview],
        }
        # Probe D extension: full signed payload for downstream stats.
        if int(arr.size) > 0 and arr.ndim >= 1:
            # Treat the LAST dim as the "per-dim" axis. Most modalities are
            # shape [B, T, D]; some may be [B, D] or [D] — handle all.
            if arr.ndim == 3:
                # [B, T, D] -> first chunk = [t=0..min(8,T), D] from b=0
                B, T, D = arr.shape
                first_b = arr[0]                          # [T, D]
                first_chunk = first_b[: min(8, T), :]     # [<=8, D]
                per_dim_mean_signed = first_b.mean(axis=0).astype(float).tolist()  # [D]
                first_step = first_b[0].astype(float).tolist()                     # [D]
                rec["full_signed_first_chunk"] = first_chunk.astype(float).tolist()
                rec["per_dim_mean_signed"] = per_dim_mean_signed
                rec["per_dim_first_step_signed"] = first_step
            elif arr.ndim == 2:
                # [B, D] or [T, D] -> first row
                rec["per_dim_first_step_signed"] = arr[0].astype(float).tolist()
                rec["per_dim_mean_signed"] = arr.mean(axis=0).astype(float).tolist()
                rec["full_signed_first_chunk"] = arr[: min(8, arr.shape[0])].astype(float).tolist()
            elif arr.ndim == 1:
                rec["per_dim_first_step_signed"] = arr.astype(float).tolist()
                rec["per_dim_mean_signed"] = arr.astype(float).tolist()
                rec["full_signed_first_chunk"] = [arr.astype(float).tolist()]
        summary[str(key)] = rec
    return summary


def _patch_helper3d() -> None:
    helper = _load_helper3d_module()
    helper._summarize_action_chunk = _patched_summarize_action_chunk  # type: ignore[attr-defined]
    sys.modules["helper3d"] = helper
    sys.modules["work.recap.scripts.3D_recap_eval"] = helper
    sys.modules["3D_recap_eval"] = helper


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe D action-replay wrapper")
    parser.add_argument("--checkpoint-tag", required=True, choices=["post_p0a", "base_p0b"])
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-log-dir", required=True)
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument("--episode-count", type=int, default=10)
    parser.add_argument("--indicator-modes", default="positive")
    parser.add_argument("--required-cuda-visible-devices", default="1")
    args = parser.parse_args(argv)

    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.required_cuda_visible_devices:
        print(
            f"[probe_D] WARN: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
            f"!= required {args.required_cuda_visible_devices!r}; runner will assert.",
            flush=True,
        )

    _patch_helper3d()
    print("[probe_D] helper3d._summarize_action_chunk monkey-patched (full signed payload)", flush=True)

    # Build argv for gr00t_g3_formal_eval and exec it as __main__.
    # Note: runner CLI uses --checkpoint (not --checkpoint-path), and
    # --indicator-modes is nargs="+" (space-separated tokens).
    runner_argv = [
        str(REPO_ROOT / "work" / "recap" / "scripts" / "gr00t_g3_formal_eval.py"),
        "--seed-base", str(args.seed_base),
        "--episode-count", str(args.episode_count),
        "--indicator-modes", *args.indicator_modes.split(","),
        "--output-dir", args.output_dir,
        "--runtime-log-dir", args.runtime_log_dir,
        "--required-cuda-visible-devices", args.required_cuda_visible_devices,
        "--checkpoint", args.checkpoint_path,
    ]
    sys.argv = runner_argv
    print(f"[probe_D] launching runner with argv: {runner_argv}", flush=True)
    runpy.run_path(runner_argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
