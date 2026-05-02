#!/usr/bin/env python3
"""Probe ZERO -- dataset lineage audit (CPU-only, read-only).

Compares per-dim action statistics between
  recap_stage3_iter_001_pilot   (3 episodes, 360 frames)
  recap_stage3_iter_002         (10 episodes, 1200 frames)

For each parquet-backed dataset, this script:
  * loads meta/modality.json  -> dim->modality_key map
  * samples up to N rows per dataset (default: all rows; cap parameter is for
    safety on larger datasets)
  * computes per-dim stats (mean/std/min/max/median/p1/p99/nz_rate) on the
    full 35-D action vector, restricted in the diff table to right_arm and
    right_hand dims (per task spec).
  * writes lineage_dim_stats.csv (dataset x dim_idx)
  * writes lineage_diff_table.csv (per-dim iter001 vs iter002 with RED flag)

Usage:
  python work/recap/probes/probe_ZERO_lineage_audit.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = REPO_ROOT / "agent" / "artifacts" / "lerobot_datasets"
OUTPUT_DIR = REPO_ROOT / "agent" / "artifacts" / "probes" / "probe_ZERO_lineage_audit"

NZ_THRESHOLD = 1.0e-4

# Datasets with comparable 35-D wbc_action namespace.
# physical_intelligence_libero_official_8d_recap_*_v1 are libero-format (8D)
# and are not directly comparable; skipped per spec.
WBC_DATASETS = (
    "recap_stage3_iter_001_pilot",
    "recap_stage3_iter_002",
    "openpi_phase05_smoke_contract_v1",
)

# Modality dim ranges of primary interest. Stats are computed on all 35 dims;
# the diff table is reported only for these (right_arm + right_hand) plus
# left_hand for context.
DIM_RANGES_OF_INTEREST: dict[str, tuple[int, int]] = {
    "right_arm": (18, 25),
    "right_hand": (25, 32),
    "left_hand": (8, 15),
}


def _load_modality_map(meta_path: Path) -> dict[int, str]:
    """Build dim_idx -> 'action.<modality_key>' from meta/modality.json."""
    obj = json.loads(meta_path.read_text())
    action_groups = obj.get("action", {})
    out: dict[int, str] = {}
    for key, span in action_groups.items():
        if key == "wbc_action":
            continue
        st = int(span["start"])
        ed = int(span["end"])
        for d in range(st, ed):
            out[d] = f"action.{key}"
    return out


def _load_action_matrix(dataset_dir: Path) -> np.ndarray:
    """Concatenate all per-episode parquet `action` columns -> (N, 35) array."""
    chunk_dir = dataset_dir / "data" / "chunk-000"
    parquets = sorted(p for p in chunk_dir.iterdir() if p.suffix == ".parquet")
    if not parquets:
        raise FileNotFoundError(f"No parquet files under {chunk_dir}")
    parts: list[np.ndarray] = []
    for p in parquets:
        tbl = pq.read_table(p, columns=["action"])
        # action is list<float32>; convert to ndarray of shape (T, D)
        col = tbl.column("action").to_pylist()
        parts.append(np.asarray(col, dtype=np.float32))
    mat = np.concatenate(parts, axis=0)
    if mat.ndim != 2:
        raise ValueError(
            f"Unexpected action shape after concat: {mat.shape} from {dataset_dir}"
        )
    return mat


def _per_dim_stats(mat: np.ndarray) -> pd.DataFrame:
    """Compute per-dim stats and return a DataFrame indexed by dim_idx."""
    n_rows, n_dims = mat.shape
    rows: list[dict[str, float | int]] = []
    for d in range(n_dims):
        col = mat[:, d]
        absc = np.abs(col)
        rows.append(
            {
                "dim_idx": int(d),
                "n_rows": int(n_rows),
                "mean": float(col.mean()),
                "std": float(col.std()),
                "min": float(col.min()),
                "max": float(col.max()),
                "median": float(np.median(col)),
                "p1": float(np.percentile(col, 1.0)),
                "p99": float(np.percentile(col, 99.0)),
                "abs_mean": float(absc.mean()),
                "abs_max": float(absc.max()),
                "nz_rate": float((absc > NZ_THRESHOLD).mean()),
            }
        )
    return pd.DataFrame(rows)


def _stats_for_dataset(name: str) -> tuple[pd.DataFrame, dict[int, str]]:
    ds_dir = DATASETS_DIR / name
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory missing: {ds_dir}")
    modality_map = _load_modality_map(ds_dir / "meta" / "modality.json")
    mat = _load_action_matrix(ds_dir)
    df = _per_dim_stats(mat)
    df.insert(0, "dataset", name)
    df.insert(2, "modality_key", df["dim_idx"].map(lambda d: modality_map.get(int(d), "<unmapped>")))
    return df, modality_map


def _diff_table(
    df_a: pd.DataFrame, df_b: pd.DataFrame, modality_map: dict[int, str],
    *, focus_ranges: Iterable[tuple[int, int]],
) -> pd.DataFrame:
    """Build iter_001 vs iter_002 diff table over the focus dims."""
    a = df_a.set_index("dim_idx")
    b = df_b.set_index("dim_idx")
    rows: list[dict[str, float | str | bool | int]] = []
    for st, ed in focus_ranges:
        for d in range(st, ed):
            ma = float(a.at[d, "mean"])
            mb = float(b.at[d, "mean"])
            denom = max(abs(ma), abs(mb), 1.0e-9)
            mean_diff_ratio = abs(ma - mb) / denom
            nzra = float(a.at[d, "nz_rate"])
            nzrb = float(b.at[d, "nz_rate"])
            nz_diff = abs(nzra - nzrb)
            absmax_a = float(a.at[d, "abs_max"])
            absmax_b = float(b.at[d, "abs_max"])
            red = bool(mean_diff_ratio > 0.5 or nz_diff > 0.3)
            severity = mean_diff_ratio + nz_diff
            rows.append(
                {
                    "dim_idx": int(d),
                    "modality_key": modality_map.get(int(d), "<unmapped>"),
                    "mean_iter001": ma,
                    "mean_iter002": mb,
                    "mean_diff_ratio": float(mean_diff_ratio),
                    "abs_max_iter001": absmax_a,
                    "abs_max_iter002": absmax_b,
                    "nz_rate_iter001": nzra,
                    "nz_rate_iter002": nzrb,
                    "nz_rate_diff": float(nz_diff),
                    "RED_flag": red,
                    "severity_rank": float(severity),
                }
            )
    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["RED_flag", "severity_rank"], ascending=[False, False]
    ).reset_index(drop=True)
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] repo_root: {REPO_ROOT}")
    print(f"[INFO] datasets_dir: {DATASETS_DIR}")
    print(f"[INFO] output_dir: {out_dir}")

    all_stats: list[pd.DataFrame] = []
    modality_maps: dict[str, dict[int, str]] = {}
    n_rows_per_dataset: dict[str, int] = {}
    for name in WBC_DATASETS:
        print(f"[INFO] ingesting dataset: {name}")
        df, mmap = _stats_for_dataset(name)
        all_stats.append(df)
        modality_maps[name] = mmap
        n_rows_per_dataset[name] = int(df["n_rows"].iloc[0])
        print(f"  rows={n_rows_per_dataset[name]} dims={len(df)}")

    full = pd.concat(all_stats, axis=0, ignore_index=True)
    _write_csv(full, out_dir / "lineage_dim_stats.csv")
    print(f"[OK] wrote {out_dir / 'lineage_dim_stats.csv'}")

    # Diff table: iter_001_pilot vs iter_002, focus right_arm + right_hand + left_hand.
    df_a = next(d for d in all_stats if d["dataset"].iloc[0] == "recap_stage3_iter_001_pilot")
    df_b = next(d for d in all_stats if d["dataset"].iloc[0] == "recap_stage3_iter_002")
    mmap = modality_maps["recap_stage3_iter_002"]
    diff = _diff_table(
        df_a, df_b, mmap,
        focus_ranges=tuple(DIM_RANGES_OF_INTEREST.values()),
    )
    _write_csv(diff, out_dir / "lineage_diff_table.csv")
    print(f"[OK] wrote {out_dir / 'lineage_diff_table.csv'}")

    # Print a quick summary to stdout for the operator.
    print("\n=== diff table (focus dims, sorted by severity) ===")
    print(diff.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    red_count = int(diff["RED_flag"].sum())
    print(f"\n[RESULT] RED-flagged dim count (mean_diff_ratio>0.5 OR nz_rate_diff>0.3): {red_count}")

    # Programmatic verdict probe: are right_hand dims dead in iter_001 too?
    rh_iter001_max = float(df_a.iloc[25:32]["abs_max"].max())
    rh_iter002_max = float(df_b.iloc[25:32]["abs_max"].max())
    rh_iter001_nz = float(df_a.iloc[25:32]["nz_rate"].max())
    rh_iter002_nz = float(df_b.iloc[25:32]["nz_rate"].max())
    print(
        f"[RESULT] right_hand abs_max:  iter001={rh_iter001_max:.4f}  iter002={rh_iter002_max:.4f}"
    )
    print(
        f"[RESULT] right_hand max nz_rate: iter001={rh_iter001_nz:.4f}  iter002={rh_iter002_nz:.4f}"
    )

    # Dump a small JSON summary for downstream consumers.
    summary = {
        "n_rows_per_dataset": n_rows_per_dataset,
        "red_flag_count": red_count,
        "right_hand_iter001_abs_max": rh_iter001_max,
        "right_hand_iter002_abs_max": rh_iter002_max,
        "right_hand_iter001_nz_rate_max": rh_iter001_nz,
        "right_hand_iter002_nz_rate_max": rh_iter002_nz,
        "right_arm_iter001_abs_max": float(df_a.iloc[18:25]["abs_max"].max()),
        "right_arm_iter002_abs_max": float(df_b.iloc[18:25]["abs_max"].max()),
        "right_arm_iter001_nz_rate_min": float(df_a.iloc[18:25]["nz_rate"].min()),
        "right_arm_iter002_nz_rate_min": float(df_b.iloc[18:25]["nz_rate"].min()),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[OK] wrote {out_dir / 'audit_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
