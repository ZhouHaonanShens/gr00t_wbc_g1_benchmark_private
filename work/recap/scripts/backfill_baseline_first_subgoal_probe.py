#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = "baseline_first_subgoal_probe_v1"
ARTIFACT_KIND = "baseline_first_subgoal_probe_backfill"
REQUIRED_SEEDS = (20260421, 20260422, 20260423)


def _repo_root() -> Path:
    return REPO_ROOT


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(path.resolve())


def _pick_number(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"JSONL row must be an object: {path}:{line_number}"
                )
            rows.append({str(key): value for key, value in payload.items()})
    return rows


def _in_requested_range(seed: int, *, seed_start: int | None, seed_end: int | None) -> bool:
    if seed_start is not None and seed < seed_start:
        return False
    if seed_end is not None and seed > seed_end:
        return False
    return True


def _extract_seed_metric(row: Mapping[str, Any]) -> dict[str, Any]:
    seed = row.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("telemetry row seed must be an int")
    stage_guess = row.get("failure_stage_guess")
    if not isinstance(stage_guess, Mapping):
        raise TypeError(f"telemetry row for seed {seed} missing failure_stage_guess")
    min_dist = _pick_number(
        stage_guess,
        "min_dist_ee_to_apple",
        "ee_to_apple_min_dist",
        "min_apple_to_right_eef_l2",
    )
    if min_dist is None:
        raise TypeError(f"telemetry row for seed {seed} missing min distance metric")
    contact_proxy = 1.0 if stage_guess.get("ever_near_apple") is True else 0.0
    lift_proxy = _pick_number(stage_guess, "lift_proxy", "max_apple_lift_z") or 0.0
    return {
        "seed": int(seed),
        "min_dist_ee_to_apple": float(min_dist),
        "contact_proxy": float(contact_proxy),
        "lift_proxy": float(lift_proxy),
        "contact_or_lift_proxy": float(max(contact_proxy, lift_proxy)),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _build_blocker_payload(
    *,
    baseline_root: Path,
    telemetry_path: Path,
    output_path: Path,
    available_seeds: Sequence[int],
    missing_required_seeds: Sequence[int],
    seed_start: int | None,
    seed_end: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": "BLOCK",
        "probe_eligible": False,
        "required_seed_metrics_present": False,
        "complete_3seed_subgoal_probe": False,
        "blocking_reasons": ["baseline_v1_seed_coverage_insufficient"],
        "required_seeds": list(REQUIRED_SEEDS),
        "available_seeds": list(sorted(available_seeds)),
        "missing_required_seeds": list(sorted(missing_required_seeds)),
        "seed_start": seed_start,
        "seed_end": seed_end,
        "selected_seeds": [],
        "seed_metrics": [],
        "baseline_root": _safe_relpath(baseline_root),
        "telemetry_path": _safe_relpath(telemetry_path),
        "output_path": _safe_relpath(output_path),
        "read_only_authority_root": True,
        "skip_reason": "baseline_v1_seed_coverage_insufficient",
    }


def _build_success_payload(
    *,
    baseline_root: Path,
    telemetry_path: Path,
    output_path: Path,
    available_seeds: Sequence[int],
    selected_metrics: Sequence[Mapping[str, Any]],
    seed_start: int | None,
    seed_end: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": "PASS",
        "probe_eligible": True,
        "required_seed_metrics_present": True,
        "complete_3seed_subgoal_probe": True,
        "blocking_reasons": [],
        "required_seeds": list(REQUIRED_SEEDS),
        "available_seeds": list(sorted(available_seeds)),
        "missing_required_seeds": [],
        "seed_start": seed_start,
        "seed_end": seed_end,
        "selected_seeds": [int(metric["seed"]) for metric in selected_metrics],
        "seed_metrics": list(selected_metrics),
        "baseline_root": _safe_relpath(baseline_root),
        "telemetry_path": _safe_relpath(telemetry_path),
        "output_path": _safe_relpath(output_path),
        "read_only_authority_root": True,
        "skip_reason": None,
    }


def run_backfill(
    *,
    baseline_root: Path,
    output_path: Path,
    seed_start: int | None = None,
    seed_end: int | None = None,
) -> dict[str, Any]:
    telemetry_path = baseline_root / "telemetry" / "eval_summary_episodes.jsonl"
    rows = _read_jsonl(telemetry_path)
    seed_metrics_by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = row.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            continue
        if not _in_requested_range(seed, seed_start=seed_start, seed_end=seed_end):
            continue
        seed_metrics_by_seed[int(seed)] = _extract_seed_metric(row)

    available_seeds = sorted(seed_metrics_by_seed)
    missing_required_seeds = [
        seed for seed in REQUIRED_SEEDS if seed not in seed_metrics_by_seed
    ]
    if missing_required_seeds:
        payload = _build_blocker_payload(
            baseline_root=baseline_root,
            telemetry_path=telemetry_path,
            output_path=output_path,
            available_seeds=available_seeds,
            missing_required_seeds=missing_required_seeds,
            seed_start=seed_start,
            seed_end=seed_end,
        )
    else:
        payload = _build_success_payload(
            baseline_root=baseline_root,
            telemetry_path=telemetry_path,
            output_path=output_path,
            available_seeds=available_seeds,
            selected_metrics=[seed_metrics_by_seed[seed] for seed in REQUIRED_SEEDS],
            seed_start=seed_start,
            seed_end=seed_end,
        )
    _write_json(output_path, payload)
    return {
        "status": payload["status"],
        "probe_eligible": payload["probe_eligible"],
        "output_path": str(output_path),
        "required_seed_metrics_present": payload["required_seed_metrics_present"],
        "complete_3seed_subgoal_probe": payload["complete_3seed_subgoal_probe"],
        "blocking_reasons": payload["blocking_reasons"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill a read-only baseline-v1 first-subgoal probe from telemetry JSONL.",
    )
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-end", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.seed_start is not None and args.seed_end is not None and args.seed_end < args.seed_start:
        parser.error("--seed-end must be >= --seed-start")
    result = run_backfill(
        baseline_root=_resolve_path(args.baseline_root),
        output_path=_resolve_path(args.output_path),
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
