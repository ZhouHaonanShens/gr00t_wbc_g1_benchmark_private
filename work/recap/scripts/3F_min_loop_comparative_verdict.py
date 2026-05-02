#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts.state_conditioned_common import write_json


DEFAULT_BASELINE_EVAL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v1/t5_baseline_formal_eval/eval_summary.json"
)
DEFAULT_CONTROL_EVAL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v1/eval_baseline_continuation_control/eval_summary.json"
)
DEFAULT_CONDITIONED_EVAL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v1/eval_numeric_advantage_conditioned/eval_summary.json"
)
DEFAULT_CONDITIONED_SURFACE_GATE = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v1/t10_conditioned_surface_gate.json"
)
DEFAULT_ADVANTAGE_EMBEDDING_DIFF = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v1/advantage_embedding_diff.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v1/min_loop_verdict.json"
)

SCHEMA_VERSION = "recap_min_loop_comparative_verdict_v1"
ARTIFACT_KIND = "recap_min_loop_comparative_verdict"
CLAIM_LEVEL_STRONG = "strong"
CLAIM_LEVEL_FUNCTIONAL_INCONCLUSIVE = "functional_inconclusive"
CLAIM_LEVEL_FAIL = "fail"
HISTORICAL_MIN_SUCCESS_COUNT = 3
HISTORICAL_MIN_SUCCESS_RATE = 0.3
COMPARATIVE_MIN_SUCCESS_DELTA = 2


def _resolve_path(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _repo_relative(path: str | Path) -> str:
    resolved = _resolve_path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _load_json_dict(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must load as a JSON object: {path}")
    return payload


def _require_bool(payload: dict[str, Any], *, field: str, label: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise TypeError(f"{label}.{field} must be a bool, got {type(value).__name__}")
    return value


def _extract_episode_seeds(episode_results: list[Any], *, label: str) -> list[int]:
    seeds: list[int] = []
    for index, entry in enumerate(episode_results, start=1):
        if not isinstance(entry, dict):
            raise TypeError(
                f"{label}.episode_results[{index}] must be an object, got {type(entry).__name__}"
            )
        seed = entry.get("seed")
        if not isinstance(seed, int):
            raise TypeError(
                f"{label}.episode_results[{index}].seed must be an int, got {type(seed).__name__}"
            )
        seeds.append(int(seed))
    return seeds


def _load_eval_summary(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_json_dict(path, label=label)
    success_count = payload.get("success_count")
    success_rate = payload.get("success_rate")
    episodes = payload.get("episodes")
    seed_base = payload.get("seed_base")
    episode_results = payload.get("episode_results")
    if not isinstance(success_count, int):
        raise TypeError(f"{label}.success_count must be an int")
    if not isinstance(success_rate, (int, float)):
        raise TypeError(f"{label}.success_rate must be numeric")
    if not isinstance(episodes, int):
        raise TypeError(f"{label}.episodes must be an int")
    if not isinstance(seed_base, int):
        raise TypeError(f"{label}.seed_base must be an int")
    if not isinstance(episode_results, list):
        raise TypeError(f"{label}.episode_results must be a list")
    seeds = _extract_episode_seeds(episode_results, label=label)
    return {
        "path": _repo_relative(path),
        "label": label,
        "success_count": int(success_count),
        "success_rate": float(success_rate),
        "episodes": int(episodes),
        "seed_base": int(seed_base),
        "episode_result_count": len(episode_results),
        "episode_seeds": seeds,
        "advantage": payload.get("advantage"),
        "advantage_mode": payload.get("advantage_mode"),
        "policy_model_path": payload.get("server_provenance", {}).get("policy_model_path")
        if isinstance(payload.get("server_provenance"), dict)
        else None,
        "log_path": payload.get("log_path"),
    }


def _same_seed_bundle(*surfaces: dict[str, Any]) -> bool:
    if not surfaces:
        return False
    first_seed_base = surfaces[0]["seed_base"]
    first_episode_count = surfaces[0]["episodes"]
    first_seeds = list(surfaces[0]["episode_seeds"])
    for surface in surfaces[1:]:
        if int(surface["seed_base"]) != int(first_seed_base):
            return False
        if int(surface["episodes"]) != int(first_episode_count):
            return False
        if list(surface["episode_seeds"]) != first_seeds:
            return False
    return True


def _append_reason(reasons: list[str], reason: str) -> None:
    normalized = str(reason).strip()
    if normalized and normalized not in reasons:
        reasons.append(normalized)


def build_payload(
    *,
    baseline_eval: Path,
    continuation_control_eval: Path,
    conditioned_eval: Path,
    conditioned_surface_gate: Path,
    advantage_embedding_diff: Path,
    output_json: Path,
) -> dict[str, Any]:
    baseline = _load_eval_summary(baseline_eval, label="baseline_eval")
    control = _load_eval_summary(
        continuation_control_eval,
        label="continuation_control_eval",
    )
    conditioned = _load_eval_summary(conditioned_eval, label="conditioned_eval")
    surface_gate_payload = _load_json_dict(
        conditioned_surface_gate,
        label="conditioned_surface_gate",
    )
    diff_payload = _load_json_dict(
        advantage_embedding_diff,
        label="advantage_embedding_diff",
    )

    conditioned_surface_gate_pass = _require_bool(
        surface_gate_payload,
        field="pass",
        label="conditioned_surface_gate",
    )
    conditioned_surface_is_valid = _require_bool(
        diff_payload,
        field="conditioned_surface_is_valid",
        label="advantage_embedding_diff",
    )
    required_semantics_pass = _require_bool(
        diff_payload,
        field="required_semantics_pass",
        label="advantage_embedding_diff",
    )
    advantage_embedding_weight_changed_from_init = _require_bool(
        diff_payload,
        field="advantage_embedding_weight_changed_from_init",
        label="advantage_embedding_diff",
    )
    advantage_embedding_bias_checked = _require_bool(
        diff_payload,
        field="advantage_embedding_bias_checked",
        label="advantage_embedding_diff",
    )
    all_tensors_finite = _require_bool(
        diff_payload,
        field="all_tensors_finite",
        label="advantage_embedding_diff",
    )

    same_seed_bundle = _same_seed_bundle(baseline, control, conditioned)
    historical_gate_pass = bool(
        conditioned["success_count"] >= HISTORICAL_MIN_SUCCESS_COUNT
        and conditioned["success_rate"] >= HISTORICAL_MIN_SUCCESS_RATE
    )
    conditioned_vs_baseline_delta = int(conditioned["success_count"] - baseline["success_count"])
    conditioned_vs_control_delta = int(conditioned["success_count"] - control["success_count"])
    comparative_gate_pass = bool(
        conditioned_vs_baseline_delta >= COMPARATIVE_MIN_SUCCESS_DELTA
        and conditioned_vs_control_delta >= COMPARATIVE_MIN_SUCCESS_DELTA
    )
    artifact_valid = bool(
        same_seed_bundle
        and conditioned_surface_gate_pass
        and conditioned_surface_is_valid
        and required_semantics_pass
        and advantage_embedding_bias_checked
        and all_tensors_finite
    )

    if artifact_valid and historical_gate_pass and comparative_gate_pass:
        claim_level = CLAIM_LEVEL_STRONG
    elif historical_gate_pass:
        claim_level = CLAIM_LEVEL_FUNCTIONAL_INCONCLUSIVE
    else:
        claim_level = CLAIM_LEVEL_FAIL

    recap_effective_claim_allowed = bool(
        artifact_valid and historical_gate_pass and comparative_gate_pass and claim_level == CLAIM_LEVEL_STRONG
    )

    blocking_reasons: list[str] = []
    if not same_seed_bundle:
        _append_reason(blocking_reasons, "eval_seed_bundle_mismatch")
    if not conditioned_surface_gate_pass:
        _append_reason(blocking_reasons, "conditioned_surface_gate_not_pass")
    if not conditioned_surface_is_valid:
        _append_reason(blocking_reasons, "conditioned_surface_invalid_in_diff")
    if not required_semantics_pass:
        _append_reason(blocking_reasons, "advantage_embedding_required_semantics_failed")
    if not advantage_embedding_weight_changed_from_init:
        _append_reason(blocking_reasons, "advantage_embedding_weight_unchanged_from_init")
    if not advantage_embedding_bias_checked:
        _append_reason(blocking_reasons, "advantage_embedding_bias_not_checked")
    if not all_tensors_finite:
        _append_reason(blocking_reasons, "advantage_embedding_tensors_not_all_finite")
    if conditioned["success_count"] < HISTORICAL_MIN_SUCCESS_COUNT:
        _append_reason(blocking_reasons, "conditioned_success_count_below_historical_threshold")
    if conditioned["success_rate"] < HISTORICAL_MIN_SUCCESS_RATE:
        _append_reason(blocking_reasons, "conditioned_success_rate_below_historical_threshold")
    if conditioned_vs_baseline_delta < COMPARATIVE_MIN_SUCCESS_DELTA:
        _append_reason(blocking_reasons, "conditioned_not_plus_2_vs_baseline")
    if conditioned_vs_control_delta < COMPARATIVE_MIN_SUCCESS_DELTA:
        _append_reason(blocking_reasons, "conditioned_not_plus_2_vs_continuation_control")

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "artifact_valid": artifact_valid,
        "historical_gate_pass": historical_gate_pass,
        "comparative_gate_pass": comparative_gate_pass,
        "claim_level": claim_level,
        "recap_effective_claim_allowed": recap_effective_claim_allowed,
        "blocking_reasons": blocking_reasons,
        "thresholds": {
            "historical_min_success_count": HISTORICAL_MIN_SUCCESS_COUNT,
            "historical_min_success_rate": HISTORICAL_MIN_SUCCESS_RATE,
            "comparative_min_success_delta": COMPARATIVE_MIN_SUCCESS_DELTA,
        },
        "comparative_metrics": {
            "conditioned_minus_baseline_success_count": conditioned_vs_baseline_delta,
            "conditioned_minus_continuation_control_success_count": conditioned_vs_control_delta,
        },
        "artifact_chain": {
            "same_seed_bundle": same_seed_bundle,
            "conditioned_surface_gate_pass": conditioned_surface_gate_pass,
            "conditioned_surface_is_valid": conditioned_surface_is_valid,
            "required_semantics_pass": required_semantics_pass,
            "advantage_embedding_weight_changed_from_init": (
                advantage_embedding_weight_changed_from_init
            ),
            "advantage_embedding_bias_checked": advantage_embedding_bias_checked,
            "all_tensors_finite": all_tensors_finite,
        },
        "baseline_eval_summary": baseline,
        "continuation_control_eval_summary": control,
        "conditioned_eval_summary": conditioned,
        "conditioned_surface_gate": {
            "path": _repo_relative(conditioned_surface_gate),
            "schema_version": surface_gate_payload.get("schema_version"),
            "pass": conditioned_surface_gate_pass,
            "surface_mode": surface_gate_payload.get("surface_mode"),
            "failure_reason_codes": list(surface_gate_payload.get("failure_reason_codes", [])),
        },
        "advantage_embedding_diff": {
            "path": _repo_relative(advantage_embedding_diff),
            "schema_version": diff_payload.get("schema_version"),
            "required_semantics_pass": required_semantics_pass,
            "conditioned_surface_is_valid": conditioned_surface_is_valid,
            "advantage_embedding_weight_changed_from_init": (
                advantage_embedding_weight_changed_from_init
            ),
            "advantage_embedding_bias_checked": advantage_embedding_bias_checked,
            "all_tensors_finite": all_tensors_finite,
            "failure_reason_codes": list(diff_payload.get("failure_reason_codes", [])),
        },
        "input_paths": {
            "baseline_eval": _repo_relative(baseline_eval),
            "continuation_control_eval": _repo_relative(continuation_control_eval),
            "conditioned_eval": _repo_relative(conditioned_eval),
            "conditioned_surface_gate": _repo_relative(conditioned_surface_gate),
            "advantage_embedding_diff": _repo_relative(advantage_embedding_diff),
        },
        "output_json_path": _repo_relative(output_json),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="3F_min_loop_comparative_verdict.py",
        description=(
            "Compare baseline, continuation-control, and numeric-advantage-conditioned eval artifacts "
            "and write the final min-loop verdict without masking artifact-chain blockers."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    parser.add_argument("--continuation-control-eval", type=Path, default=DEFAULT_CONTROL_EVAL)
    parser.add_argument("--conditioned-eval", type=Path, default=DEFAULT_CONDITIONED_EVAL)
    parser.add_argument(
        "--conditioned-surface-gate",
        type=Path,
        default=DEFAULT_CONDITIONED_SURFACE_GATE,
    )
    parser.add_argument(
        "--advantage-embedding-diff",
        type=Path,
        default=DEFAULT_ADVANTAGE_EMBEDDING_DIFF,
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_payload(
        baseline_eval=_resolve_path(args.baseline_eval),
        continuation_control_eval=_resolve_path(args.continuation_control_eval),
        conditioned_eval=_resolve_path(args.conditioned_eval),
        conditioned_surface_gate=_resolve_path(args.conditioned_surface_gate),
        advantage_embedding_diff=_resolve_path(args.advantage_embedding_diff),
        output_json=_resolve_path(args.output_json),
    )
    output_json = _resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    _ = write_json(output_json, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
