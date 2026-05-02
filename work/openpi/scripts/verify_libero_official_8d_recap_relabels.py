#!/usr/bin/env python3
"""Metadata-only verifier for official/native 8D RECAP relabel materialization.

The verifier intentionally avoids parquet/GPU execution.  It checks the durable
JSON reports emitted by the materialization/statistics gates so a team worker can
prove that the formal-safe dataset reroute exists before launching OpenPI runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, cast

ROUTE_ID = "official_native_8d_recap_relabels_v1"
REPORT_SCHEMA_VERSION = "openpi_libero_official_8d_recap_relabels_verification_v1"
REQUIRED_RECAP_COLUMNS = {
    "recap_m2.t",
    "recap_m2.return_G",
    "recap_m2.value_V",
    "recap_m2.advantage_A",
    "recap_m2.advantage_input",
    "recap_m2.epsilon_l",
    "recap_m2.indicator_I",
    "recap_m2.prompt_raw",
    "recap_m2.prompt_conditioned",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return cast(dict[str, Any], payload)


def _shape(payload: dict[str, Any], key: str) -> list[int]:
    features = payload.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json missing features object")
    feature = features.get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"meta/info.json missing feature {key!r}")
    shape = feature.get("shape")
    if not isinstance(shape, list):
        raise ValueError(f"feature {key!r} missing shape list")
    return [int(item) for item in shape]


def build_report(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    materialization = _read_json(dataset_dir / "materialization_report.json")
    info = _read_json(dataset_dir / "meta" / "info.json")
    stats = _read_json(dataset_dir / "meta" / "relabel_stats_report.json")
    fingerprint = _read_json(dataset_dir / "meta" / "dataset_fingerprint.json")

    checks: dict[str, bool] = {
        "materialization_status_is_materialized": materialization.get("final_status") == "materialized",
        "materialization_route_matches": materialization.get("route_id") == ROUTE_ID,
        "info_route_matches": info.get("route_id") == ROUTE_ID,
        "stats_status_ready": stats.get("final_status") == "ready",
        "stats_route_matches": stats.get("route_id") == ROUTE_ID,
        "state_shape_is_8d": _shape(info, "observation.state") == [8],
        "action_shape_is_7d": _shape(info, "action") == [7],
        "frame_count_matches": int(materialization.get("selected_frame_count", -1)) == int(stats.get("frame_count", -2)) == int(fingerprint.get("frame_count", -3)),
        "episode_count_matches": int(materialization.get("selected_episode_count", -1)) == int(stats.get("episode_count", -2)) == int(fingerprint.get("episode_count", -3)),
        "required_recap_columns_reported": REQUIRED_RECAP_COLUMNS.issubset(set(materialization.get("required_output_label_columns", []))),
        "indicator_has_positive_and_negative": int(stats.get("indicator_positive_count", 0)) > 0 and int(stats.get("indicator_negative_count", 0)) > 0,
        "fingerprint_current": stats.get("dataset_fingerprint_status") == "current",
    }
    final_status = "ready" if all(checks.values()) else "blocked"
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "route_id": ROUTE_ID,
        "dataset_dir": str(dataset_dir),
        "final_status": final_status,
        "checks": checks,
        "blockers": blockers,
        "episode_count": int(stats.get("episode_count", 0)),
        "frame_count": int(stats.get("frame_count", 0)),
        "state_shape": _shape(info, "observation.state"),
        "action_shape": _shape(info, "action"),
        "indicator_positive_count": int(stats.get("indicator_positive_count", 0)),
        "indicator_negative_count": int(stats.get("indicator_negative_count", 0)),
        "dataset_fingerprint_sha256": str(stats.get("dataset_fingerprint_sha256", "")),
        "materialization_report": str(dataset_dir / "materialization_report.json"),
        "stats_report": str(dataset_dir / "meta" / "relabel_stats_report.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_report(args.dataset_dir)
    except Exception as exc:
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "route_id": ROUTE_ID,
            "dataset_dir": str(args.dataset_dir),
            "final_status": "blocked",
            "checks": {},
            "blockers": [f"{type(exc).__name__}: {exc}"],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if report.get("final_status") == "ready" else 42


if __name__ == "__main__":
    raise SystemExit(main())
