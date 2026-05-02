#!/usr/bin/env python3
"""Build OpenPI runtime dataset/norm provenance for the formal LIBERO relabels.

This is a CPU-only pre-runtime artifact builder.  It intentionally does not
claim p1/p2 GPU runtime success.  Instead it proves that the formal dataset,
fingerprint, norm stats, and one-row loader/prompt bridge are ready for the GPU2
runtime lane and records any data-side blocker with an atomic code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.norm.policy import (  # noqa: E402
    build_phase1_norm_policy,
    build_phase1_norm_provenance,
    validate_phase1_norm_policy,
)
from work.openpi.scripts.verify_libero_official_8d_recap_relabels import (  # noqa: E402
    ROUTE_ID,
    build_report as build_materialization_reverify_report,
)
from work.recap import text_indicator  # noqa: E402


REPORT_SCHEMA_VERSION = "openpi_runtime_dataset_provenance_v1"
NORM_SCHEMA_VERSION = "openpi_norm_stats_provenance_v1"
P0_LOADER_SCHEMA_VERSION = "openpi_p0_runtime_loader_smoke_v1"
RUNTIME_LEVEL_P0 = "p0_loader_runtime_pass"
RUNTIME_LEVEL_PENDING = "materialized_dataset_runtime_pending"
CONSUMER_MODE = "informative_adv"
REQUIRED_STATS_KEYS = ("observation.state", "action")
REQUIRED_RECAP_COLUMNS = (
    "action",
    "episode_index",
    "observation.state",
    "recap_m2.advantage_A",
    "recap_m2.advantage_input",
    "recap_m2.indicator_I",
    "recap_m2.prompt_conditioned",
    "recap_m2.prompt_raw",
    "recap_m2.return_G",
    "recap_m2.t",
    "recap_m2.value_V",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return cast(dict[str, Any], payload)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(info: dict[str, Any], key: str) -> list[int]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json missing features")
    feature = features.get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"meta/info.json missing feature {key!r}")
    raw_shape = feature.get("shape")
    if not isinstance(raw_shape, list):
        raise ValueError(f"meta/info.json feature {key!r} missing shape")
    return [int(item) for item in raw_shape]


def _coerce_vector(value: object) -> list[float]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, list):
        raise TypeError(f"expected list-like vector, got {type(value).__name__}")
    return [float(item) for item in value]


def _coerce_scalar(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            return value
    return value


def _stats_vector_ok(stats_payload: dict[str, Any], key: str, dim: int) -> bool:
    entry = stats_payload.get(key)
    if not isinstance(entry, dict):
        return False
    for stat_name in ("mean", "std", "min", "max"):
        values = entry.get(stat_name)
        if not isinstance(values, list) or len(values) != dim:
            return False
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            return False
    std_values = entry.get("std")
    return isinstance(std_values, list) and all(float(value) > 0.0 for value in std_values)


def build_norm_stats_provenance(dataset_dir: Path) -> dict[str, Any]:
    info = _read_json(dataset_dir / "meta" / "info.json")
    stats_path = dataset_dir / "meta" / "stats.json"
    stats_payload = _read_json(stats_path)
    state_dim = _shape(info, "observation.state")[0]
    action_dim = _shape(info, "action")[0]
    policy = validate_phase1_norm_policy(build_phase1_norm_policy(dataset_dir))
    policy_payload = build_phase1_norm_provenance(policy)
    checks = {
        "stats_file_exists": stats_path.is_file(),
        "required_stats_keys_present": all(key in stats_payload for key in REQUIRED_STATS_KEYS),
        "state_stats_shape_matches": _stats_vector_ok(
            stats_payload, "observation.state", state_dim
        ),
        "action_stats_shape_matches": _stats_vector_ok(stats_payload, "action", action_dim),
        "policy_validates": True,
    }
    return {
        "schema_version": NORM_SCHEMA_VERSION,
        "dataset_dir": str(dataset_dir),
        "route_id": str(info.get("route_id", "")),
        "final_status": "ready" if all(checks.values()) else "blocked",
        "checks": checks,
        "blockers": [name for name, passed in checks.items() if not passed],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "stats_json_path": str(stats_path),
        "stats_json_sha256": _sha256_file(stats_path),
        "stats_json_bytes": stats_path.stat().st_size,
        "policy": policy_payload,
    }


def build_p0_loader_smoke(dataset_dir: Path, *, preview_rows: int = 1) -> dict[str, Any]:
    import pandas as pd

    info = _read_json(dataset_dir / "meta" / "info.json")
    state_dim = _shape(info, "observation.state")[0]
    action_dim = _shape(info, "action")[0]
    parquet_files = tuple(sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"missing parquet episodes under {dataset_dir}")
    first_parquet = parquet_files[0]
    frame = pd.read_parquet(first_parquet, columns=list(REQUIRED_RECAP_COLUMNS))
    if frame.empty:
        raise ValueError(f"{first_parquet}: no rows")
    row = frame.head(max(1, int(preview_rows))).iloc[0]
    state = _coerce_vector(row["observation.state"])
    action = _coerce_vector(row["action"])
    prompt_raw = _coerce_scalar(row["recap_m2.prompt_raw"])
    indicator_mode = text_indicator.indicator_mode_from_indicator_value(
        _coerce_scalar(row["recap_m2.indicator_I"])
    )
    prompt_text = text_indicator.build_authoritative_carrier_text_v1(
        prompt_raw, indicator_mode
    )
    checks = {
        "parquet_file_found": True,
        "required_columns_present": all(column in frame.columns for column in REQUIRED_RECAP_COLUMNS),
        "state_dim_matches": len(state) == int(state_dim) == 8,
        "action_dim_matches": len(action) == int(action_dim) == 7,
        "prompt_bridge_built": bool(prompt_text),
    }
    return {
        "schema_version": P0_LOADER_SCHEMA_VERSION,
        "artifact_kind": "p0_runtime_loader_smoke",
        "dataset_dir": str(dataset_dir),
        "route_id": ROUTE_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "runtime_level": RUNTIME_LEVEL_P0 if all(checks.values()) else RUNTIME_LEVEL_PENDING,
        "checks": checks,
        "blockers": [name for name, passed in checks.items() if not passed],
        "loader_bridge": {
            "mapping_source": "meta/info.json feature shapes",
            "prompt_builder": "work.recap.text_indicator.build_authoritative_carrier_text_v1",
            "norm_policy_builder": "work.openpi.norm.policy.build_phase1_norm_policy",
            "consumer_mode": CONSUMER_MODE,
        },
        "sample": {
            "source_parquet": str(first_parquet),
            "episode_index": int(_coerce_scalar(row["episode_index"])),
            "state_dim": len(state),
            "action_dim": len(action),
            "prompt_text": prompt_text,
            "prompt_provenance": {
                "prompt_route": "carrier_text_v1",
                "conditioning_mode": "text_indicator",
                "source_prompt_field": "recap_m2.prompt_raw",
                "indicator_mode": indicator_mode,
                "consumer_mode": CONSUMER_MODE,
            },
        },
        "formal_claim_allowed": False,
        "notes": "CPU p0 loader bridge only; p1/p2 GPU2 runtime evidence is still required.",
    }


def _load_or_build_reverify(
    dataset_dir: Path, materialization_reverify_path: Path | None
) -> dict[str, Any]:
    if materialization_reverify_path is not None and materialization_reverify_path.is_file():
        return _read_json(materialization_reverify_path)
    return build_materialization_reverify_report(dataset_dir)


def build_runtime_dataset_provenance(
    dataset_dir: Path, *, materialization_reverify_path: Path | None = None
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    info = _read_json(dataset_dir / "meta" / "info.json")
    materialization = _read_json(dataset_dir / "materialization_report.json")
    fingerprint = _read_json(dataset_dir / "meta" / "dataset_fingerprint.json")
    relabel_stats = _read_json(dataset_dir / "meta" / "relabel_stats_report.json")
    reverify = _load_or_build_reverify(dataset_dir, materialization_reverify_path)
    norm = build_norm_stats_provenance(dataset_dir)
    try:
        p0_loader = build_p0_loader_smoke(dataset_dir)
    except Exception as exc:
        p0_loader = {
            "schema_version": P0_LOADER_SCHEMA_VERSION,
            "artifact_kind": "p0_runtime_loader_smoke",
            "dataset_dir": str(dataset_dir),
            "status": "FAIL",
            "runtime_level": RUNTIME_LEVEL_PENDING,
            "blockers": [f"p0_loader_bridge_error:{type(exc).__name__}: {exc}"],
            "formal_claim_allowed": False,
        }

    data_blockers: list[str] = []
    if reverify.get("final_status") != "ready":
        data_blockers.append("materialization_reverify_failed")
    if norm.get("final_status") != "ready":
        data_blockers.extend(str(item) for item in norm.get("blockers", []))
    if p0_loader.get("status") != "PASS":
        data_blockers.extend(str(item) for item in p0_loader.get("blockers", []))

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_dir.name,
        "route_id": str(info.get("route_id", "")),
        "formal_dataset": dataset_dir.name
        == "physical_intelligence_libero_official_8d_recap_relabels_v1",
        "materialization": {
            "final_status": str(materialization.get("final_status", "")),
            "route_id": str(materialization.get("route_id", "")),
            "report_path": str(dataset_dir / "materialization_report.json"),
            "reverify_path": str(materialization_reverify_path or ""),
            "reverify_final_status": str(reverify.get("final_status", "")),
            "reverify_checks": dict(cast(dict[str, Any], reverify.get("checks", {}))),
        },
        "fingerprint": {
            "path": str(dataset_dir / "meta" / "dataset_fingerprint.json"),
            "fingerprint_sha256": str(fingerprint.get("fingerprint_sha256", "")),
            "stats_report_fingerprint_sha256": str(
                relabel_stats.get("dataset_fingerprint_sha256", "")
            ),
            "parquet_inventory_hash": str(fingerprint.get("parquet_inventory_hash", "")),
            "episode_universe_hash": str(fingerprint.get("episode_universe_hash", "")),
            "episode_count": int(fingerprint.get("episode_count", 0)),
            "frame_count": int(fingerprint.get("frame_count", 0)),
            "state_shape": _shape(info, "observation.state"),
            "action_shape": _shape(info, "action"),
            "fingerprint_status": str(relabel_stats.get("dataset_fingerprint_status", "")),
        },
        "norm_stats": norm,
        "p0_runtime_loader_smoke": p0_loader,
        "runtime_blocker_triage": {
            "data_side_status": "ready" if not data_blockers else "blocked",
            "data_side_blockers": data_blockers,
            "runtime_level": p0_loader.get("runtime_level", RUNTIME_LEVEL_PENDING),
            "formal_claim_allowed": False,
            "pending_gpu2_runtime_evidence": [
                "p1_one_step_probe_gpu2",
                "p2_tiny_update_or_overfit20_gpu2",
            ],
            "blocked_reason_if_no_worker3_runtime": "p1_one_step_runtime_evidence_pending",
        },
        "authority_inputs": [
            str(dataset_dir / "materialization_report.json"),
            str(dataset_dir / "meta" / "info.json"),
            str(dataset_dir / "meta" / "stats.json"),
            str(dataset_dir / "meta" / "relabel_stats_report.json"),
            str(dataset_dir / "meta" / "dataset_fingerprint.json"),
            str(materialization_reverify_path or ""),
        ],
        "notes": (
            "Worker4 CPU provenance artifact: data/norm/p0 loader bridge ready "
            "does not claim p1/p2 GPU2 runtime or benchmark success."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--materialization-reverify", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--norm-out", type=Path)
    parser.add_argument("--p0-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_runtime_dataset_provenance(
        args.dataset_dir,
        materialization_reverify_path=args.materialization_reverify,
    )
    _write_json(args.out, report)
    if args.norm_out is not None:
        _write_json(args.norm_out, report["norm_stats"])
    if args.p0_out is not None:
        _write_json(args.p0_out, report["p0_runtime_loader_smoke"])
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    triage = cast(dict[str, Any], report["runtime_blocker_triage"])
    return 0 if triage.get("data_side_status") == "ready" else 42


if __name__ == "__main__":
    raise SystemExit(main())
