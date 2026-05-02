#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap import finetune_full
from work.recap import policy as recap_policy


SCHEMA_VERSION = "task13_full_update_rollout_probe_v1"
PARTIAL_VS_FULL_ACTION_PROBE_FILENAME = "partial_vs_full_action_probe.json"
SUBGOAL_SUMMARY_FILENAME = "subgoal_summary_3seed.json"
FULL_UPDATE_DIAGNOSTIC_SUMMARY_FILENAME = "full_update_diagnostic_summary.json"
MIN_LOOP_VERDICT_FILENAME = "min_loop_verdict.json"
P5_EXECUTION_DECISION_FILENAME = "p5_execution_decision.json"
P5_BLOCKER_SUMMARY_FILENAME = "p5_gate_blocker_summary.json"
CONDITIONING_PROBE_FILENAME = "conditioning_functional_probe_step20.json"
PAIRED_ACTION_PROBE_FILENAME = "paired_action_probe_step20.json"
LABEL_SEMANTICS_AUDIT_FILENAME = "label_semantics_audit.json"
PREFORMAL_GATE_FILENAME = "preformal_gate_decision.json"
SCOPE_AUDIT_DYNAMIC_FILENAME = "full_update_scope_audit_dynamic.json"
VERSION_SURFACE_FILENAME = "version_surface.json"
COMPARABILITY_MANIFEST_FILENAME = finetune_full.COMPARABILITY_MANIFEST_FILENAME
DEFAULT_SUBGOAL_SEED_COUNT = 3

_task10_module_cache: Any | None = None


def _repo_root() -> Path:
    return REPO_ROOT


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _safe_relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(path.resolve())


def _dedupe(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [
        str(item)
        for item in value
        if isinstance(item, str) and str(item).strip()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON payload must be a mapping: {path}")
    return {str(key): value for key, value in payload.items()}


def _read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return _read_json(path)


def _write_json(output_dir: Path, filename: str, payload: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _p4_gate_summary_path(v2_authority_root: Path) -> Path:
    return (
        v2_authority_root
        / "p4_loss_action_subgoal"
        / FULL_UPDATE_DIAGNOSTIC_SUMMARY_FILENAME
    )


def _p5_gate_summary_candidates(v2_authority_root: Path) -> list[Path]:
    """Return P4 authority candidates in Stage1-preferred order.

    Stage1 names the machine gate ``p4_refresh/p4_gate_verdict.json`` while
    older Task13/Task14 artifacts expose the same four hard P5 inputs through
    ``p4_loss_action_subgoal/full_update_diagnostic_summary.json``.  Prefer the
    Stage1 verdict when present, but keep the legacy path for existing tests
    and authority bundles.
    """

    return [
        v2_authority_root / "p4_refresh" / "p4_gate_verdict.json",
        v2_authority_root / "p4_refresh" / FULL_UPDATE_DIAGNOSTIC_SUMMARY_FILENAME,
        _p4_gate_summary_path(v2_authority_root),
    ]


def _load_task10_module() -> Any:
    global _task10_module_cache
    if _task10_module_cache is not None:
        return _task10_module_cache
    module_path = Path(__file__).with_name("34b_recap_numeric_adv_smoke.py")
    spec = importlib.util.spec_from_file_location(
        "task13_rollout_probe_task10_helpers",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _task10_module_cache = module
    return module


def _checkpoint_asset_path(checkpoint_dir: Path) -> Path | None:
    candidates = (
        checkpoint_dir / "model.safetensors",
        checkpoint_dir / "model.safetensors.index.json",
        checkpoint_dir / "pytorch_model.bin",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_checkpoint_dir(formal_root: Path) -> Path | None:
    preferred = formal_root / "checkpoint-200"
    if preferred.is_dir() and _checkpoint_asset_path(preferred) is not None:
        return preferred
    candidates = sorted(
        (
            path
            for path in formal_root.glob("checkpoint-*")
            if path.is_dir() and _checkpoint_asset_path(path) is not None
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        return None
    return candidates[-1]


def _resolve_lane_state(run_root: Path) -> dict[str, Any]:
    formal_root = run_root / "formal_run"
    effective_formal_root = formal_root if formal_root.is_dir() else run_root
    skip_candidates = (
        formal_root / "formal_run_skipped.json",
        run_root / "formal_run_skipped.json",
        formal_root / "skipped_manifest.json",
        run_root / "skipped_manifest.json",
    )
    for skip_path in skip_candidates:
        skip_payload = _read_json_if_exists(skip_path)
        if skip_payload is None:
            continue
        return {
            "resolution": "skipped",
            "run_root": str(run_root),
            "formal_root": str(effective_formal_root),
            "skip_manifest_path": str(skip_path),
            "skip_manifest": skip_payload,
            "checkpoint_path": None,
            "checkpoint_asset_path": None,
            "p3_formal_training_eligible": bool(
                skip_payload.get("p3_formal_training_eligible", False)
            ),
            "p3_skip_reason": skip_payload.get("p3_skip_reason"),
            "blocking_reasons": _string_list(skip_payload.get("blocking_reasons")),
        }
    checkpoint_dir = _resolve_checkpoint_dir(effective_formal_root)
    if checkpoint_dir is not None:
        checkpoint_asset = _checkpoint_asset_path(checkpoint_dir)
        return {
            "resolution": "checkpoint",
            "run_root": str(run_root),
            "formal_root": str(effective_formal_root),
            "skip_manifest_path": None,
            "skip_manifest": None,
            "checkpoint_path": str(checkpoint_dir),
            "checkpoint_asset_path": None if checkpoint_asset is None else str(checkpoint_asset),
            "p3_formal_training_eligible": True,
            "p3_skip_reason": None,
            "blocking_reasons": [],
        }
    return {
        "resolution": "missing",
        "run_root": str(run_root),
        "formal_root": str(effective_formal_root),
        "skip_manifest_path": None,
        "skip_manifest": None,
        "checkpoint_path": None,
        "checkpoint_asset_path": None,
        "p3_formal_training_eligible": False,
        "p3_skip_reason": "missing_formal_checkpoint_or_skip_manifest",
        "blocking_reasons": ["missing_formal_checkpoint_or_skip_manifest"],
    }


def _load_scope_gate_payload(v2_authority_root: Path, lane_states: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], Path | None]:
    candidate_paths: list[Path] = []
    for lane_state in lane_states:
        skip_manifest = lane_state.get("skip_manifest")
        if isinstance(skip_manifest, Mapping):
            best_scope_authority_path = skip_manifest.get("best_scope_authority_path")
            if isinstance(best_scope_authority_path, str) and best_scope_authority_path.strip():
                candidate_paths.append(_resolve_path(best_scope_authority_path))
    candidate_paths.extend(
        [
            v2_authority_root
            / "p1_one_step"
            / "repo_local_metadata"
            / SCOPE_AUDIT_DYNAMIC_FILENAME,
            v2_authority_root
            / "p1_one_step_fresh_20260423"
            / "repo_local_metadata"
            / SCOPE_AUDIT_DYNAMIC_FILENAME,
        ]
    )
    for candidate in candidate_paths:
        payload = _read_json_if_exists(candidate)
        if payload is not None:
            return payload, candidate
    return {}, None


def _load_route_freeze_payload(v2_authority_root: Path) -> tuple[dict[str, Any], Path | None]:
    candidate_paths = (
        v2_authority_root
        / "p2_full_update_overfit20"
        / "repo_local_metadata"
        / VERSION_SURFACE_FILENAME,
        v2_authority_root
        / "p1_one_step"
        / "repo_local_metadata"
        / VERSION_SURFACE_FILENAME,
    )
    for candidate in candidate_paths:
        payload = _read_json_if_exists(candidate)
        if payload is None:
            continue
        route_freeze = payload.get("route_freeze")
        if isinstance(route_freeze, Mapping):
            return {str(key): value for key, value in route_freeze.items()}, candidate
    return {}, None


def _evaluate_route_freeze(route_freeze: Mapping[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not route_freeze:
        return False, ["missing_route_freeze"]
    if route_freeze.get("frozen") is not True:
        blockers.append("route_freeze_not_frozen")
    if route_freeze.get("route") != recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE:
        blockers.append("route_freeze_route_mismatch")
    if route_freeze.get("diagnostic_only") is not True:
        blockers.append("route_freeze_diagnostic_only_false")
    return not blockers, blockers


def _candidate_comparability_paths(run_root: Path) -> tuple[Path, ...]:
    return (
        run_root / COMPARABILITY_MANIFEST_FILENAME,
        run_root / "formal_run" / COMPARABILITY_MANIFEST_FILENAME,
    )


def _resolve_comparability_status(
    conditioned_run_root: Path,
    continuation_run_root: Path,
) -> dict[str, Any]:
    conditioned_manifest_path = next(
        (path for path in _candidate_comparability_paths(conditioned_run_root) if path.is_file()),
        None,
    )
    continuation_manifest_path = next(
        (path for path in _candidate_comparability_paths(continuation_run_root) if path.is_file()),
        None,
    )
    if conditioned_manifest_path is None or continuation_manifest_path is None:
        missing = []
        if conditioned_manifest_path is None:
            missing.append("conditioned")
        if continuation_manifest_path is None:
            missing.append("continuation")
        return {
            "comparability_manifest_pass": False,
            "comparability_blocker_reason": "missing_comparability_manifest",
            "validation": {
                "status": "blocked",
                "blocker_code": "missing_comparability_manifest",
                "reason": "missing comparability manifest for required full-update lanes",
                "missing_lanes": missing,
            },
            "conditioned_manifest_path": None
            if conditioned_manifest_path is None
            else str(conditioned_manifest_path),
            "continuation_manifest_path": None
            if continuation_manifest_path is None
            else str(continuation_manifest_path),
        }
    conditioned_manifest = _read_json(conditioned_manifest_path)
    continuation_manifest = _read_json(continuation_manifest_path)
    validation = finetune_full.validate_full_update_comparability_manifests(
        conditioned_manifest,
        continuation_manifest,
    )
    return {
        "comparability_manifest_pass": validation.get("status") == "pass",
        "comparability_blocker_reason": None
        if validation.get("status") == "pass"
        else str(validation.get("blocker_code") or "comparability_manifest_block"),
        "validation": validation,
        "conditioned_manifest_path": str(conditioned_manifest_path),
        "continuation_manifest_path": str(continuation_manifest_path),
    }


def _load_required_task8_artifacts(v2_authority_root: Path) -> dict[str, Any]:
    p2_dir = v2_authority_root / "p2_full_update_overfit20"
    p25_dir = v2_authority_root / "p2_5_label_semantics"
    conditioning_probe_path = p2_dir / CONDITIONING_PROBE_FILENAME
    paired_action_probe_path = p2_dir / PAIRED_ACTION_PROBE_FILENAME
    label_semantics_path = p25_dir / LABEL_SEMANTICS_AUDIT_FILENAME
    preformal_gate_path = p25_dir / PREFORMAL_GATE_FILENAME
    return {
        "conditioning_probe_path": conditioning_probe_path,
        "conditioning_probe": _read_json_if_exists(conditioning_probe_path) or {},
        "paired_action_probe_path": paired_action_probe_path,
        "paired_action_probe": _read_json_if_exists(paired_action_probe_path) or {},
        "label_semantics_audit_path": label_semantics_path,
        "label_semantics_audit": _read_json_if_exists(label_semantics_path) or {},
        "preformal_gate_path": preformal_gate_path,
        "preformal_gate": _read_json_if_exists(preformal_gate_path) or {},
    }


def _baseline_reference_payload(baseline_authority_root: Path) -> dict[str, Any]:
    conditioned_summary_path = (
        baseline_authority_root / "eval_numeric_advantage_conditioned" / "eval_summary.json"
    )
    continuation_summary_path = (
        baseline_authority_root / "eval_baseline_continuation_control" / "eval_summary.json"
    )
    baseline_summary_path = (
        baseline_authority_root / "t5_baseline_formal_eval" / "eval_summary.json"
    )
    seed_bundle_path = baseline_authority_root / "eval_seed_set.json"
    min_loop_verdict_path = baseline_authority_root / "min_loop_verdict.json"
    return {
        "authority_root": str(baseline_authority_root),
        "conditioned_eval_summary_path": _safe_relpath(conditioned_summary_path),
        "conditioned_eval_summary": _read_json_if_exists(conditioned_summary_path),
        "continuation_control_eval_summary_path": _safe_relpath(continuation_summary_path),
        "continuation_control_eval_summary": _read_json_if_exists(continuation_summary_path),
        "baseline_eval_summary_path": _safe_relpath(baseline_summary_path),
        "baseline_eval_summary": _read_json_if_exists(baseline_summary_path),
        "eval_seed_set_path": _safe_relpath(seed_bundle_path),
        "eval_seed_set": _read_json_if_exists(seed_bundle_path),
        "min_loop_verdict_path": _safe_relpath(min_loop_verdict_path),
        "min_loop_verdict": _read_json_if_exists(min_loop_verdict_path),
    }


def _pick_number(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _coerce_seed_metric(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    seed = raw.get("seed")
    if not isinstance(seed, int):
        return None
    nested_metrics = raw.get("metrics")
    metric_source = nested_metrics if isinstance(nested_metrics, Mapping) else raw
    min_dist = _pick_number(metric_source, "min_dist_ee_to_apple", "ee_to_apple_min_dist")
    if min_dist is None:
        return None
    contact_or_lift_proxy = _pick_number(metric_source, "contact_or_lift_proxy")
    if contact_or_lift_proxy is None:
        contact_proxy = _pick_number(metric_source, "contact_proxy")
        lift_proxy = _pick_number(metric_source, "lift_proxy")
        if contact_proxy is None and lift_proxy is None:
            contact_or_lift_proxy = 0.0
        else:
            contact_or_lift_proxy = max(contact_proxy or 0.0, lift_proxy or 0.0)
    return {
        "seed": seed,
        "min_dist_ee_to_apple": float(min_dist),
        "contact_or_lift_proxy": float(contact_or_lift_proxy),
    }


def _normalize_lane_probe_metrics(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    candidates = payload.get("seed_metrics")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        candidates = payload.get("per_seed")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        candidates = []
    result: dict[int, dict[str, Any]] = {}
    for raw in candidates:
        if not isinstance(raw, Mapping):
            continue
        coerced = _coerce_seed_metric(raw)
        if coerced is None:
            continue
        result[int(coerced["seed"])] = coerced
    return result


def _load_lane_subgoal_probe(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _read_json_if_exists(path)


def _subgoal_probe_candidates(base_dir: Path) -> tuple[Path, ...]:
    return (
        base_dir / "first_subgoal_probe.json",
        base_dir / "first_subgoal_probe_3seed.json",
        base_dir / SUBGOAL_SUMMARY_FILENAME,
    )


def _resolve_lane_subgoal_probe_path(
    *,
    lane_name: str,
    baseline_authority_root: Path,
    run_root: Path | None,
    lane_state: Mapping[str, Any] | None,
    baseline_v1_subgoal_override: Path | None = None,
) -> Path | None:
    if lane_name == "baseline_v1":
        if baseline_v1_subgoal_override is not None and baseline_v1_subgoal_override.is_file():
            return baseline_v1_subgoal_override
        for candidate in _subgoal_probe_candidates(baseline_authority_root):
            if candidate.is_file():
                return candidate
        return None
    if run_root is None:
        return None
    search_roots: list[Path] = []
    if lane_state is not None:
        formal_root = lane_state.get("formal_root")
        if isinstance(formal_root, str) and formal_root.strip():
            search_roots.append(Path(formal_root))
    search_roots.extend([run_root / "formal_run", run_root])
    for search_root in search_roots:
        for candidate in _subgoal_probe_candidates(search_root):
            if candidate.is_file():
                return candidate
    return None


def _load_seed_bundle(baseline_reference: Mapping[str, Any]) -> list[int]:
    seed_bundle = baseline_reference.get("eval_seed_set")
    if not isinstance(seed_bundle, Mapping):
        return []
    raw_seeds = seed_bundle.get("seeds")
    if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes)):
        return []
    result: list[int] = []
    for raw_seed in raw_seeds:
        if isinstance(raw_seed, int):
            result.append(raw_seed)
    return result


def _requested_seed_bundle(seed_start: int, seed_end: int) -> list[int]:
    if int(seed_end) < int(seed_start):
        raise ValueError("--seed-end must be >= --seed-start")
    return list(range(int(seed_start), int(seed_end) + 1))


def _load_p5_gate_summary(v2_authority_root: Path) -> tuple[dict[str, Any] | None, Path]:
    candidates = _p5_gate_summary_candidates(v2_authority_root)
    for summary_path in candidates:
        payload = _read_json_if_exists(summary_path)
        if payload is not None:
            return payload, summary_path
    return None, candidates[0]


def _load_p5_seed_bundle(
    repo_root: Path,
    *,
    baseline_authority_root: Path,
) -> dict[str, Any]:
    return finetune_full.load_full_update_comparability_seed_bundle(
        repo_root,
        seed_bundle_path=baseline_authority_root / "eval_seed_set.json",
    )


def _extract_eval_summary_seed_set(
    payload: Mapping[str, Any], *, label: str
) -> list[int]:
    episode_results = payload.get("episode_results")
    if not isinstance(episode_results, Sequence) or isinstance(
        episode_results, (str, bytes)
    ):
        raise TypeError(f"{label}.episode_results must be a list")
    seeds: list[int] = []
    for index, raw in enumerate(episode_results, start=1):
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{label}.episode_results[{index}] must be an object, got {type(raw).__name__}"
            )
        seed = raw.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(
                f"{label}.episode_results[{index}].seed must be an int, got {type(seed).__name__}"
            )
        seeds.append(int(seed))
    return seeds


def _validate_eval_summary_payload(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    success_count = payload.get("success_count")
    success_rate = payload.get("success_rate")
    episodes = payload.get("episodes")
    seed_base = payload.get("seed_base")
    if not isinstance(success_count, int) or isinstance(success_count, bool):
        raise TypeError(f"{label}.success_count must be an int")
    if not isinstance(success_rate, (int, float)) or isinstance(success_rate, bool):
        raise TypeError(f"{label}.success_rate must be numeric")
    if not isinstance(episodes, int) or isinstance(episodes, bool) or int(episodes) <= 0:
        raise TypeError(f"{label}.episodes must be a positive int")
    if not isinstance(seed_base, int) or isinstance(seed_base, bool):
        raise TypeError(f"{label}.seed_base must be an int")
    episode_seeds = _extract_eval_summary_seed_set(payload, label=label)
    if len(episode_seeds) != int(episodes):
        raise ValueError(
            f"{label}.episode_results length must equal episodes ({len(episode_seeds)} != {episodes})"
        )
    return {
        "success_count": int(success_count),
        "success_rate": float(success_rate),
        "episodes": int(episodes),
        "seed_base": int(seed_base),
        "episode_seeds": episode_seeds,
    }


def _build_p5_skip_blockers(
    *,
    gate_summary: Mapping[str, Any] | None,
    extra_reasons: Sequence[str],
) -> list[str]:
    blockers = list(extra_reasons)
    if isinstance(gate_summary, Mapping):
        blockers.extend(_string_list(gate_summary.get("blocking_reasons")))
        p3_skip_reason = gate_summary.get("p3_skip_reason")
        if isinstance(p3_skip_reason, str) and p3_skip_reason.strip():
            blockers.append(p3_skip_reason)
    return _dedupe(blockers)


def _comparability_manifest_blockers(
    *,
    gate_summary: Mapping[str, Any],
    gate_summary_path: Path | None,
) -> list[str]:
    blockers: list[str] = []

    if gate_summary.get("comparability_manifest_pass") is False:
        blockers.append("comparability_manifest_not_pass")

    raw_manifest_path = gate_summary.get("comparability_manifest")
    stage1_gate = (
        str(gate_summary.get("schema_version") or "")
        == "gr00t_p4_gate_verdict_v1"
    )
    if not isinstance(raw_manifest_path, str) or not raw_manifest_path.strip():
        if stage1_gate:
            blockers.append("comparability_manifest_missing")
        return blockers

    manifest_path = Path(raw_manifest_path)
    if not manifest_path.is_absolute() and gate_summary_path is not None:
        manifest_path = gate_summary_path.parent / manifest_path

    manifest = _read_json_if_exists(manifest_path)
    if manifest is None:
        blockers.append("comparability_manifest_missing")
        return blockers

    status = str(manifest.get("status") or "").upper()
    if status != "PASS":
        blockers.append(f"comparability_manifest_status_{status.lower() or 'missing'}")

    paired_seed_total = manifest.get("paired_seed_total")
    if not isinstance(paired_seed_total, int) or isinstance(paired_seed_total, bool):
        blockers.append("comparability_manifest_paired_seed_total_missing")
    elif paired_seed_total < DEFAULT_SUBGOAL_SEED_COUNT:
        blockers.append("comparability_manifest_paired_seed_total_below_3")

    paired_seed_improvement_count = manifest.get("paired_seed_improvement_count")
    if not isinstance(paired_seed_improvement_count, int) or isinstance(
        paired_seed_improvement_count,
        bool,
    ):
        blockers.append("comparability_manifest_paired_seed_improvement_count_missing")
    elif paired_seed_improvement_count < 2:
        blockers.append("comparability_manifest_paired_seed_improvement_count_below_2")

    for hash_field in (
        "baseline_config_hash",
        "candidate_config_hash",
        "eval_condition_hash",
    ):
        value = manifest.get(hash_field)
        if not isinstance(value, str) or not value.startswith("sha256:"):
            blockers.append(f"comparability_manifest_{hash_field}_missing")

    manifest_blockers = _string_list(manifest.get("blocking_reasons"))
    if manifest_blockers:
        blockers.append("comparability_manifest_blocking_reasons_present")
        blockers.extend(manifest_blockers)

    return _dedupe(blockers)


def _p5_formal_gate_contract_blockers(
    gate_summary: Mapping[str, Any],
    *,
    gate_summary_path: Path | None = None,
) -> list[str]:
    """Return blockers that prevent P5 from executing from a P4 gate summary.

    P5 is a formal gate.  The raw ``p5_formal_10ep_eligible`` flag is not
    enough by itself: the upstream P4 summary must also be a clean formal pass
    with no blocking reasons.
    """

    blockers: list[str] = []
    status = str(gate_summary.get("status") or "").upper()
    if status != "PASS":
        blockers.append(f"p4_summary_status_{status.lower() or 'missing'}")
    if gate_summary.get("formal_claim_allowed") is not True:
        blockers.append("p4_formal_claim_not_allowed")
    p4_blocking_reasons = _string_list(gate_summary.get("blocking_reasons"))
    if p4_blocking_reasons:
        blockers.append("p4_blocking_reasons_present")
        blockers.extend(p4_blocking_reasons)
    if gate_summary.get("p5_formal_10ep_eligible") is not True:
        blockers.append("p5_formal_10ep_ineligible")
    blockers.extend(
        _comparability_manifest_blockers(
            gate_summary=gate_summary,
            gate_summary_path=gate_summary_path,
        )
    )
    return _dedupe(blockers)


def _build_p5_skip_payload(
    *,
    baseline_authority_root: Path,
    v2_authority_root: Path,
    output_dir: Path,
    gate_summary: Mapping[str, Any] | None,
    gate_summary_path: Path,
    seed_bundle: Mapping[str, Any],
    requested_seed_set: Sequence[int],
    blocker_reason: str,
    blocking_reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "full_update_p5_gate_verdict",
        "status": "SKIPPED",
        "gate_mode": "skipped",
        "blocker_reason": blocker_reason,
        "blocking_reasons": _dedupe(blocking_reasons),
        "authority_root": _safe_relpath(v2_authority_root),
        "authority_root_label": "single_gpu_v2_full_update",
        "baseline_authority_root": _safe_relpath(baseline_authority_root),
        "output_dir": _safe_relpath(output_dir),
        "output_json_path": _safe_relpath(output_dir / MIN_LOOP_VERDICT_FILENAME),
        "blocker_summary_path": _safe_relpath(output_dir / P5_BLOCKER_SUMMARY_FILENAME),
        "gate_summary_path": _safe_relpath(gate_summary_path),
        "gate_summary_found": bool(gate_summary_path.is_file()),
        "gate_summary_status": None
        if not isinstance(gate_summary, Mapping)
        else gate_summary.get("status"),
        "p5_formal_10ep_eligible": bool(
            isinstance(gate_summary, Mapping)
            and gate_summary.get("p5_formal_10ep_eligible") is True
        ),
        "p5_probe_eligible": bool(
            isinstance(gate_summary, Mapping) and gate_summary.get("p5_probe_eligible") is True
        ),
        "comparability_manifest_pass": None
        if not isinstance(gate_summary, Mapping)
        else gate_summary.get("comparability_manifest_pass"),
        "route_freeze_ok": None
        if not isinstance(gate_summary, Mapping)
        else gate_summary.get("route_freeze_ok"),
        "seed_set_source": str(seed_bundle.get("seed_set_source", "inherit_from_v1")),
        "seed_set_source_path": seed_bundle.get("seed_set_source_path"),
        "seed_bundle_status": seed_bundle.get("status"),
        "seed_bundle_blocker_code": seed_bundle.get("blocker_code"),
        "seed_set": list(seed_bundle.get("seed_set") or []),
        "requested_seed_set": list(requested_seed_set),
        "requested_episode_count": len(requested_seed_set),
        "seed_bundle_identity_pass": False,
        "formal_execution_attempted": False,
        "lane_eval_outputs": {},
    }


def _copy_eval_summary(
    *,
    output_dir: Path,
    lane_name: str,
    payload: Mapping[str, Any],
) -> Path:
    lane_dir = output_dir / lane_name
    return _write_json(lane_dir, "eval_summary.json", payload)


def _default_p5_lane_eval_summary_path(run_root: Path) -> Path | None:
    candidates = (
        run_root / "formal_run" / "eval_summary.json",
        run_root / "eval_summary.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _run_p5_eval_lane(
    *,
    lane_name: str,
    lane_state: Mapping[str, Any],
    run_root: Path,
    output_dir: Path,
    requested_seed_set: Sequence[int],
) -> dict[str, Any]:
    summary_path = _default_p5_lane_eval_summary_path(run_root)
    if summary_path is None:
        raise RuntimeError(
            f"{lane_name} requires an eval_summary.json under {run_root}/formal_run or {run_root}"
        )
    summary_payload = _read_json(summary_path)
    validated = _validate_eval_summary_payload(summary_payload, label=f"{lane_name}_eval_summary")
    if list(validated["episode_seeds"]) != list(requested_seed_set):
        raise ValueError(
            f"{lane_name} eval summary seeds do not match inherited v1 seed bundle"
        )
    output_path = _copy_eval_summary(
        output_dir=output_dir,
        lane_name=lane_name,
        payload=summary_payload,
    )
    return {
        "lane_name": lane_name,
        "status": "PASS",
        "source_summary_path": _safe_relpath(summary_path),
        "output_summary_path": _safe_relpath(output_path),
        "checkpoint_path": lane_state.get("checkpoint_path"),
        "checkpoint_asset_path": lane_state.get("checkpoint_asset_path"),
        **validated,
    }


def _build_p5_executed_payload(
    *,
    baseline_authority_root: Path,
    v2_authority_root: Path,
    output_dir: Path,
    gate_summary_path: Path,
    seed_bundle: Mapping[str, Any],
    requested_seed_set: Sequence[int],
    baseline_eval: Mapping[str, Any],
    conditioned_eval: Mapping[str, Any],
    continuation_eval: Mapping[str, Any],
) -> dict[str, Any]:
    lane_eval_outputs = {
        "baseline": dict(baseline_eval),
        "conditioned": dict(conditioned_eval),
        "continuation": dict(continuation_eval),
    }
    shared_seed_match = all(
        list(lane_payload.get("episode_seeds") or []) == list(requested_seed_set)
        for lane_payload in lane_eval_outputs.values()
    )
    blocking_reasons: list[str] = []
    if not shared_seed_match:
        blocking_reasons.append("p5_seed_bundle_identity_mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "full_update_p5_gate_verdict",
        "status": "PASS" if not blocking_reasons else "BLOCK",
        "gate_mode": "executed",
        "blocking_reasons": blocking_reasons,
        "authority_root": _safe_relpath(v2_authority_root),
        "authority_root_label": "single_gpu_v2_full_update",
        "baseline_authority_root": _safe_relpath(baseline_authority_root),
        "output_dir": _safe_relpath(output_dir),
        "output_json_path": _safe_relpath(output_dir / MIN_LOOP_VERDICT_FILENAME),
        "gate_summary_path": _safe_relpath(gate_summary_path),
        "seed_set_source": str(seed_bundle.get("seed_set_source", "inherit_from_v1")),
        "seed_set_source_path": seed_bundle.get("seed_set_source_path"),
        "seed_set": list(seed_bundle.get("seed_set") or []),
        "requested_seed_set": list(requested_seed_set),
        "requested_episode_count": len(requested_seed_set),
        "seed_bundle_identity_pass": shared_seed_match,
        "formal_execution_attempted": True,
        "lane_eval_outputs": lane_eval_outputs,
    }


def _p5_gate_inputs(gate_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(gate_summary, Mapping):
        return {
            "status": None,
            "formal_claim_allowed": None,
            "blocking_reasons": [],
            "p5_formal_10ep_eligible": None,
        }
    return {
        "status": gate_summary.get("status"),
        "formal_claim_allowed": gate_summary.get("formal_claim_allowed"),
        "blocking_reasons": _string_list(gate_summary.get("blocking_reasons")),
        "p5_formal_10ep_eligible": gate_summary.get("p5_formal_10ep_eligible"),
    }


def _build_p5_execution_decision(
    *,
    verdict: Mapping[str, Any],
    gate_summary: Mapping[str, Any] | None,
    gate_summary_path: Path,
    min_loop_verdict_path: Path,
    blocker_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    formal_execution_attempted = verdict.get("formal_execution_attempted") is True
    decision = "RUN" if formal_execution_attempted else "BLOCKED"
    pre_execution_blockers = (
        []
        if decision == "RUN"
        else _string_list(verdict.get("blocking_reasons"))
    )
    return {
        "schema_version": "gr00t_p5_execution_decision_v1",
        "decision": decision,
        "p4_gate_verdict": _safe_relpath(gate_summary_path),
        "gate_inputs": _p5_gate_inputs(gate_summary),
        "blocking_reasons": _dedupe(pre_execution_blockers),
        "min_loop_verdict": _safe_relpath(min_loop_verdict_path),
        "p5_gate_blocker_summary": _safe_relpath(blocker_summary_path),
        "output_dir": _safe_relpath(output_dir),
        "requested_episode_count": verdict.get("requested_episode_count"),
        "requested_seed_set": list(verdict.get("requested_seed_set") or []),
        "formal_execution_attempted": formal_execution_attempted,
        "min_loop_status": verdict.get("status"),
    }


def _write_p5_gate_artifacts(
    *,
    output_dir: Path,
    verdict: Mapping[str, Any],
    blocker_summary: Mapping[str, Any],
    gate_summary: Mapping[str, Any] | None,
    gate_summary_path: Path,
) -> tuple[Path, Path, Path]:
    verdict_path = _write_json(output_dir, MIN_LOOP_VERDICT_FILENAME, verdict)
    blocker_summary_path = _write_json(
        output_dir,
        P5_BLOCKER_SUMMARY_FILENAME,
        blocker_summary,
    )
    decision = _build_p5_execution_decision(
        verdict=verdict,
        gate_summary=gate_summary,
        gate_summary_path=gate_summary_path,
        min_loop_verdict_path=verdict_path,
        blocker_summary_path=blocker_summary_path,
        output_dir=output_dir,
    )
    decision_path = _write_json(output_dir, P5_EXECUTION_DECISION_FILENAME, decision)
    return verdict_path, blocker_summary_path, decision_path


def _p5_gate_result(
    *,
    output_dir: Path,
    verdict: Mapping[str, Any],
    verdict_path: Path,
    blocker_summary_path: Path,
    decision_path: Path,
    extra_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    result = {
        "mode": "p5_gate",
        "status": verdict["status"],
        "output_dir": str(output_dir),
        "min_loop_verdict_path": str(verdict_path),
        "blocker_summary_path": str(blocker_summary_path),
        "p5_execution_decision_path": str(decision_path),
    }
    if extra_paths:
        result.update(dict(extra_paths))
    return result


def _build_first_subgoal_summary(
    *,
    baseline_authority_root: Path,
    conditioned_run_root: Path,
    continuation_run_root: Path,
    baseline_v1_subgoal_override: Path | None,
    baseline_reference: Mapping[str, Any],
    conditioned_lane_state: Mapping[str, Any],
    continuation_lane_state: Mapping[str, Any],
    comparability_manifest_pass: bool,
    comparability_blocker_reason: str | None,
    route_freeze_ok: bool,
) -> dict[str, Any]:
    seed_bundle = _load_seed_bundle(baseline_reference)
    selected_seeds = seed_bundle[:DEFAULT_SUBGOAL_SEED_COUNT]
    lane_details: dict[str, Any] = {}
    blocking_reasons: list[str] = []

    if not route_freeze_ok:
        blocking_reasons.append("route_freeze_block")
    if not comparability_manifest_pass:
        blocking_reasons.append(str(comparability_blocker_reason or "comparability_manifest_block"))

    lane_specs = (
        ("baseline_v1", baseline_authority_root, None),
        ("conditioned_v2", conditioned_run_root, conditioned_lane_state),
        ("continuation_v2", continuation_run_root, continuation_lane_state),
    )
    lane_metrics: dict[str, dict[int, dict[str, Any]]] = {}
    for lane_name, lane_root, lane_state in lane_specs:
        if lane_state is not None and lane_state.get("resolution") == "skipped":
            skip_reason = str(lane_state.get("p3_skip_reason") or "formal_lane_skipped")
            lane_details[lane_name] = {
                "status": "SKIPPED",
                "skip_reason": skip_reason,
                "blocking_reasons": list(lane_state.get("blocking_reasons", [])),
                "probe_path": None,
            }
            blocking_reasons.append(f"{lane_name}_formal_lane_skipped")
            continue
        probe_path = _resolve_lane_subgoal_probe_path(
            lane_name=lane_name,
            baseline_authority_root=baseline_authority_root,
            run_root=lane_root,
            lane_state=lane_state,
            baseline_v1_subgoal_override=baseline_v1_subgoal_override,
        )
        probe_payload = _load_lane_subgoal_probe(probe_path)
        if probe_payload is None:
            lane_details[lane_name] = {
                "status": "BLOCK",
                "skip_reason": None,
                "blocking_reasons": ["missing_first_subgoal_probe"],
                "probe_path": None,
            }
            blocking_reasons.append(f"missing_{lane_name}_first_subgoal_probe")
            continue
        lane_details[lane_name] = {
            "status": str(probe_payload.get("status") or "PASS").upper(),
            "skip_reason": probe_payload.get("skip_reason"),
            "blocking_reasons": _string_list(probe_payload.get("blocking_reasons")),
            "probe_path": str(probe_path),
        }
        lane_metrics[lane_name] = _normalize_lane_probe_metrics(probe_payload)
        if not lane_metrics[lane_name]:
            blocking_reasons.append(f"missing_{lane_name}_seed_metrics")

    if not selected_seeds:
        selected_seeds = sorted(
            set(lane_metrics.get("baseline_v1", {}))
            & set(lane_metrics.get("conditioned_v2", {}))
            & set(lane_metrics.get("continuation_v2", {}))
        )[:DEFAULT_SUBGOAL_SEED_COUNT]
    if len(selected_seeds) < DEFAULT_SUBGOAL_SEED_COUNT:
        blocking_reasons.append("missing_3seed_bundle")

    per_seed_pairs: list[dict[str, Any]] = []
    relative_improvements: list[float] = []
    contact_non_regression_flags: list[bool] = []
    all_contact_values: list[float] = []
    for seed in selected_seeds:
        baseline_metric = lane_metrics.get("baseline_v1", {}).get(seed)
        conditioned_metric = lane_metrics.get("conditioned_v2", {}).get(seed)
        continuation_metric = lane_metrics.get("continuation_v2", {}).get(seed)
        if (
            baseline_metric is None
            or conditioned_metric is None
            or continuation_metric is None
        ):
            blocking_reasons.append(f"missing_seed_metric_{seed}")
            continue
        control_best_distance = min(
            float(baseline_metric["min_dist_ee_to_apple"]),
            float(continuation_metric["min_dist_ee_to_apple"]),
        )
        conditioned_distance = float(conditioned_metric["min_dist_ee_to_apple"])
        relative_improvement = (
            (control_best_distance - conditioned_distance)
            / max(control_best_distance, 1e-9)
        )
        baseline_contact = float(baseline_metric["contact_or_lift_proxy"])
        conditioned_contact = float(conditioned_metric["contact_or_lift_proxy"])
        continuation_contact = float(continuation_metric["contact_or_lift_proxy"])
        control_best_contact = max(baseline_contact, continuation_contact)
        no_regression = conditioned_contact + 1e-9 >= control_best_contact
        per_seed_pairs.append(
            {
                "seed": seed,
                "baseline_min_dist_ee_to_apple": baseline_metric["min_dist_ee_to_apple"],
                "conditioned_min_dist_ee_to_apple": conditioned_distance,
                "continuation_min_dist_ee_to_apple": continuation_metric[
                    "min_dist_ee_to_apple"
                ],
                "control_best_min_dist_ee_to_apple": control_best_distance,
                "relative_improvement_min_dist_ee_to_apple": relative_improvement,
                "baseline_contact_or_lift_proxy": baseline_contact,
                "conditioned_contact_or_lift_proxy": conditioned_contact,
                "continuation_contact_or_lift_proxy": continuation_contact,
                "no_regression_on_contact_or_lift_proxy": no_regression,
            }
        )
        relative_improvements.append(relative_improvement)
        contact_non_regression_flags.append(no_regression)
        all_contact_values.extend(
            [baseline_contact, conditioned_contact, continuation_contact]
        )

    paired_seed_improvement_count = sum(
        1 for improvement in relative_improvements if improvement > 1e-9
    )
    mean_relative_improvement = (
        float(sum(relative_improvements) / len(relative_improvements))
        if relative_improvements
        else None
    )
    no_regression_on_contact_or_lift_proxy = bool(
        contact_non_regression_flags and all(contact_non_regression_flags)
    )
    all_contact_zero = bool(all_contact_values) and all(
        abs(value) <= 1e-9 for value in all_contact_values
    )
    weak_distance_only = bool(
        len(relative_improvements) == DEFAULT_SUBGOAL_SEED_COUNT
        and paired_seed_improvement_count >= 2
        and mean_relative_improvement is not None
        and mean_relative_improvement >= 0.05
        and all_contact_zero
    )
    strong_subgoal_progress_gate_pass = bool(
        len(relative_improvements) == DEFAULT_SUBGOAL_SEED_COUNT
        and paired_seed_improvement_count >= 2
        and mean_relative_improvement is not None
        and mean_relative_improvement >= 0.05
        and no_regression_on_contact_or_lift_proxy
        and not weak_distance_only
    )

    if len(relative_improvements) != DEFAULT_SUBGOAL_SEED_COUNT:
        blocking_reasons.append("missing_complete_3seed_subgoal_probe")
    elif paired_seed_improvement_count < 2:
        blocking_reasons.append("paired_seed_improvement_count_below_2_of_3")
    elif mean_relative_improvement is not None and mean_relative_improvement < 0.05:
        blocking_reasons.append(
            "mean_relative_improvement_min_dist_ee_to_apple_below_0p05"
        )
    elif weak_distance_only:
        blocking_reasons.append("contact_or_lift_proxy_uninformative_all_zero")
    elif not no_regression_on_contact_or_lift_proxy:
        blocking_reasons.append("contact_or_lift_proxy_regression")

    status = "PASS" if strong_subgoal_progress_gate_pass else "BLOCK"
    if status == "PASS":
        blocking_reasons = []

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "task13_subgoal_summary_3seed",
        "status": status,
        "seed_bundle_source": "inherit_from_v1" if seed_bundle else "unavailable",
        "requested_seed_count": DEFAULT_SUBGOAL_SEED_COUNT,
        "selected_seeds": selected_seeds,
        "lane_status": lane_details,
        "per_seed_pairs": per_seed_pairs,
        "paired_seed_improvement_count": paired_seed_improvement_count,
        "mean_relative_improvement_min_dist_ee_to_apple": mean_relative_improvement,
        "no_regression_on_contact_or_lift_proxy": no_regression_on_contact_or_lift_proxy,
        "strong_subgoal_progress_gate_pass": strong_subgoal_progress_gate_pass,
        "weak_distance_only": weak_distance_only,
        "blocking_reasons": _dedupe(blocking_reasons),
        "skip_reason": None if status != "SKIPPED" else "subgoal_probe_skipped",
    }


def _build_partial_vs_full_action_probe(
    *,
    baseline_reference: Mapping[str, Any],
    task8_artifacts: Mapping[str, Any],
    conditioned_lane_state: Mapping[str, Any],
    continuation_lane_state: Mapping[str, Any],
    comparability: Mapping[str, Any],
    route_freeze_ok: bool,
) -> dict[str, Any]:
    task10_module = _load_task10_module()
    paired_contract = task10_module._task10_build_paired_action_probe_contract(
        paired_probe=task8_artifacts["paired_action_probe"],
        output_dir=task8_artifacts["paired_action_probe_path"].parent,
    )
    blocking_reasons = list(paired_contract["blocking_reasons"])
    if not bool(comparability.get("comparability_manifest_pass")):
        blocking_reasons.append(
            str(
                comparability.get("comparability_blocker_reason")
                or "comparability_manifest_block"
            )
        )
    if not route_freeze_ok:
        blocking_reasons.append("route_freeze_block")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "task13_partial_vs_full_action_probe",
        "status": "PASS" if not blocking_reasons else "BLOCK",
        "baseline_reference": {
            "authority_root": baseline_reference.get("authority_root"),
            "conditioned_eval_summary_path": baseline_reference.get(
                "conditioned_eval_summary_path"
            ),
            "continuation_control_eval_summary_path": baseline_reference.get(
                "continuation_control_eval_summary_path"
            ),
            "baseline_eval_summary_path": baseline_reference.get(
                "baseline_eval_summary_path"
            ),
            "eval_seed_set_path": baseline_reference.get("eval_seed_set_path"),
            "min_loop_verdict_path": baseline_reference.get("min_loop_verdict_path"),
        },
        "paired_action_probe": paired_contract,
        "conditioned_lane": {
            "resolution": conditioned_lane_state.get("resolution"),
            "p3_formal_training_eligible": conditioned_lane_state.get(
                "p3_formal_training_eligible"
            ),
            "p3_skip_reason": conditioned_lane_state.get("p3_skip_reason"),
        },
        "continuation_lane": {
            "resolution": continuation_lane_state.get("resolution"),
            "p3_formal_training_eligible": continuation_lane_state.get(
                "p3_formal_training_eligible"
            ),
            "p3_skip_reason": continuation_lane_state.get("p3_skip_reason"),
        },
        "comparability_manifest_pass": bool(
            comparability.get("comparability_manifest_pass", False)
        ),
        "comparability_blocker_reason": comparability.get(
            "comparability_blocker_reason"
        ),
        "route_freeze_ok": route_freeze_ok,
        "blocking_reasons": _dedupe(blocking_reasons),
    }


def _postprocess_gate_summary(
    *,
    summary: dict[str, Any],
    first_subgoal_summary: Mapping[str, Any],
    route_freeze_ok: bool,
    route_freeze_blockers: Sequence[str],
    conditioned_lane_state: Mapping[str, Any],
    continuation_lane_state: Mapping[str, Any],
    comparability: Mapping[str, Any],
    scope_gate_payload: Mapping[str, Any],
    scope_gate_path: Path | None,
    route_freeze_path: Path | None,
) -> dict[str, Any]:
    normalized_scope_gate_pass = bool(
        str(scope_gate_payload.get("resolution_status") or "").upper() == "PASS"
        and scope_gate_payload.get("best_scope_authority") is True
    )
    summary["artifact_kind"] = "full_update_diagnostic_summary"
    summary["schema_version"] = SCHEMA_VERSION
    summary["full_update_scope_gate_pass"] = normalized_scope_gate_pass
    if not route_freeze_ok:
        summary["p3_formal_training_eligible"] = False
        summary["p5_probe_eligible"] = False
        summary["p5_formal_10ep_eligible"] = False
        summary["p6_branch_eligible"] = False
        if summary.get("p3_skip_reason") in {None, ""}:
            summary["p3_skip_reason"] = "route_freeze_block"
        summary["routing_decision"] = "block_downstream"
    summary["route_freeze_ok"] = route_freeze_ok
    summary["route_freeze_blocking_reasons"] = list(route_freeze_blockers)
    summary["route_freeze_path"] = None if route_freeze_path is None else str(route_freeze_path)
    summary["scope_gate_resolution_status"] = scope_gate_payload.get("resolution_status")
    summary["scope_gate_best_scope_authority"] = scope_gate_payload.get(
        "best_scope_authority"
    )
    summary["scope_gate_path"] = None if scope_gate_path is None else str(scope_gate_path)
    summary.pop("full_update_scope_gate_path", None)
    summary.pop("full_update_scope_gate_reason", None)
    summary["conditioned_lane_state"] = dict(conditioned_lane_state)
    summary["continuation_lane_state"] = dict(continuation_lane_state)
    summary["comparability_validation"] = comparability.get("validation")
    summary["conditioned_comparability_manifest_path"] = comparability.get(
        "conditioned_manifest_path"
    )
    summary["continuation_comparability_manifest_path"] = comparability.get(
        "continuation_manifest_path"
    )
    if isinstance(summary.get("first_subgoal_probe"), Mapping):
        summary["first_subgoal_probe"] = {
            **dict(summary["first_subgoal_probe"]),
            "paired_seed_improvement_count": first_subgoal_summary.get(
                "paired_seed_improvement_count"
            ),
            "mean_relative_improvement_min_dist_ee_to_apple": first_subgoal_summary.get(
                "mean_relative_improvement_min_dist_ee_to_apple"
            ),
            "no_regression_on_contact_or_lift_proxy": first_subgoal_summary.get(
                "no_regression_on_contact_or_lift_proxy"
            ),
            "selected_seeds": list(first_subgoal_summary.get("selected_seeds", [])),
            "lane_status": dict(first_subgoal_summary.get("lane_status", {})),
        }
    blocking_reasons: list[str] = []
    for field_name in ("loss_probe", "paired_action_probe", "first_subgoal_probe"):
        field_payload = summary.get(field_name)
        if not isinstance(field_payload, Mapping):
            blocking_reasons.append(f"missing_{field_name}")
            continue
        status = str(field_payload.get("status") or "").upper()
        if status != "PASS":
            field_blockers = _string_list(field_payload.get("blocking_reasons"))
            if status == "SKIPPED":
                skip_reason = field_payload.get("skip_reason")
                if isinstance(skip_reason, str) and skip_reason.strip():
                    field_blockers.append(skip_reason)
            if not field_blockers:
                field_blockers.append(f"{field_name.lower()}_{status.lower()}")
            blocking_reasons.extend(field_blockers)
    if not bool(summary.get("comparability_manifest_pass", False)):
        blocking_reasons.append(
            str(
                summary.get("comparability_blocker_reason")
                or comparability.get("comparability_blocker_reason")
                or "comparability_manifest_block"
            )
        )
    if not route_freeze_ok:
        blocking_reasons.extend(route_freeze_blockers or ["route_freeze_block"])
    if not bool(summary.get("label_semantics_gate_pass", False)):
        blocking_reasons.append("label_semantics_gate_block")
    if not bool(summary.get("shuffled_advantage_negative_control_pass", False)):
        blocking_reasons.append("shuffled_advantage_negative_control_block")
    summary["blocking_reasons"] = _dedupe(blocking_reasons)
    summary["status"] = "PASS" if not summary["blocking_reasons"] else "BLOCK"
    summary["formal_claim_allowed"] = summary["status"] == "PASS"
    return summary


def run_p4_diagnostics(
    *,
    baseline_authority_root: Path,
    v2_authority_root: Path,
    conditioned_run_root: Path,
    continuation_run_root: Path,
    output_dir: Path,
    baseline_v1_subgoal_override: Path | None = None,
) -> dict[str, Any]:
    task10_module = _load_task10_module()
    baseline_reference = _baseline_reference_payload(baseline_authority_root)
    conditioned_lane_state = _resolve_lane_state(conditioned_run_root)
    continuation_lane_state = _resolve_lane_state(continuation_run_root)
    lane_states = [conditioned_lane_state, continuation_lane_state]
    task8_artifacts = _load_required_task8_artifacts(v2_authority_root)
    comparability = _resolve_comparability_status(
        conditioned_run_root,
        continuation_run_root,
    )
    scope_gate_payload, scope_gate_path = _load_scope_gate_payload(
        v2_authority_root,
        lane_states,
    )
    route_freeze_payload, route_freeze_path = _load_route_freeze_payload(v2_authority_root)
    route_freeze_ok, route_freeze_blockers = _evaluate_route_freeze(route_freeze_payload)
    scope_gate_pass = bool(
        str(scope_gate_payload.get("resolution_status") or "").upper() == "PASS"
        and scope_gate_payload.get("best_scope_authority") is True
    )
    first_subgoal_summary = _build_first_subgoal_summary(
        baseline_authority_root=baseline_authority_root,
        conditioned_run_root=conditioned_run_root,
        continuation_run_root=continuation_run_root,
        baseline_v1_subgoal_override=baseline_v1_subgoal_override,
        baseline_reference=baseline_reference,
        conditioned_lane_state=conditioned_lane_state,
        continuation_lane_state=continuation_lane_state,
        comparability_manifest_pass=bool(comparability["comparability_manifest_pass"]),
        comparability_blocker_reason=comparability["comparability_blocker_reason"],
        route_freeze_ok=route_freeze_ok,
    )
    partial_vs_full_action_probe = _build_partial_vs_full_action_probe(
        baseline_reference=baseline_reference,
        task8_artifacts=task8_artifacts,
        conditioned_lane_state=conditioned_lane_state,
        continuation_lane_state=continuation_lane_state,
        comparability=comparability,
        route_freeze_ok=route_freeze_ok,
    )
    diagnostic_summary = task10_module._task8_build_preformal_gate_decision(
        conditioning_probe=task8_artifacts["conditioning_probe"],
        paired_probe=task8_artifacts["paired_action_probe"],
        label_semantics_audit=task8_artifacts["label_semantics_audit"],
        output_dir=task8_artifacts["conditioning_probe_path"].parent,
        label_semantics_output_dir=task8_artifacts["label_semantics_audit_path"].parent,
        full_update_scope_gate_pass=scope_gate_pass,
        comparability_manifest_pass=bool(comparability["comparability_manifest_pass"]),
        comparability_blocker_reason=comparability["comparability_blocker_reason"],
        first_subgoal_probe=first_subgoal_summary,
        continuous_numeric_advantage_dead_after_full_update=False,
    )
    diagnostic_summary = _postprocess_gate_summary(
        summary=dict(diagnostic_summary),
        first_subgoal_summary=first_subgoal_summary,
        route_freeze_ok=route_freeze_ok,
        route_freeze_blockers=route_freeze_blockers,
        conditioned_lane_state=conditioned_lane_state,
        continuation_lane_state=continuation_lane_state,
        comparability=comparability,
        scope_gate_payload=scope_gate_payload,
        scope_gate_path=scope_gate_path,
        route_freeze_path=route_freeze_path,
    )

    action_probe_path = _write_json(
        output_dir,
        PARTIAL_VS_FULL_ACTION_PROBE_FILENAME,
        partial_vs_full_action_probe,
    )
    subgoal_summary_path = _write_json(
        output_dir,
        SUBGOAL_SUMMARY_FILENAME,
        first_subgoal_summary,
    )
    diagnostic_summary_path = _write_json(
        output_dir,
        FULL_UPDATE_DIAGNOSTIC_SUMMARY_FILENAME,
        diagnostic_summary,
    )
    return {
        "mode": "p4",
        "status": diagnostic_summary["status"],
        "output_dir": str(output_dir),
        "partial_vs_full_action_probe_path": str(action_probe_path),
        "subgoal_summary_3seed_path": str(subgoal_summary_path),
        "full_update_diagnostic_summary_path": str(diagnostic_summary_path),
    }


def run_p5_gate(
    *,
    baseline_authority_root: Path,
    v2_authority_root: Path,
    conditioned_run_root: Path,
    continuation_run_root: Path,
    output_dir: Path,
    seed_start: int,
    seed_end: int,
    baseline_v1_subgoal_override: Path | None = None,
) -> dict[str, Any]:
    del baseline_v1_subgoal_override
    requested_seed_set = _requested_seed_bundle(seed_start, seed_end)
    gate_summary, gate_summary_path = _load_p5_gate_summary(v2_authority_root)
    seed_bundle = _load_p5_seed_bundle(
        _repo_root(),
        baseline_authority_root=baseline_authority_root,
    )

    if gate_summary is None:
        blocking_reasons = _build_p5_skip_blockers(
            gate_summary=None,
            extra_reasons=["missing_p4_gate_summary"],
        )
        verdict = _build_p5_skip_payload(
            baseline_authority_root=baseline_authority_root,
            v2_authority_root=v2_authority_root,
            output_dir=output_dir,
            gate_summary=None,
            gate_summary_path=gate_summary_path,
            seed_bundle=seed_bundle,
            requested_seed_set=requested_seed_set,
            blocker_reason="missing_p4_gate_summary",
            blocking_reasons=blocking_reasons,
        )
        blocker_summary = {
            **verdict,
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "full_update_p5_gate_blocker_summary",
        }
        verdict_path, blocker_summary_path, decision_path = _write_p5_gate_artifacts(
            output_dir=output_dir,
            verdict=verdict,
            blocker_summary=blocker_summary,
            gate_summary=None,
            gate_summary_path=gate_summary_path,
        )
        return _p5_gate_result(
            output_dir=output_dir,
            verdict=verdict,
            verdict_path=verdict_path,
            blocker_summary_path=blocker_summary_path,
            decision_path=decision_path,
        )

    if seed_bundle.get("status") != "ok":
        blocking_reasons = _build_p5_skip_blockers(
            gate_summary=gate_summary,
            extra_reasons=["missing_v1_seed_bundle"],
        )
        verdict = _build_p5_skip_payload(
            baseline_authority_root=baseline_authority_root,
            v2_authority_root=v2_authority_root,
            output_dir=output_dir,
            gate_summary=gate_summary,
            gate_summary_path=gate_summary_path,
            seed_bundle=seed_bundle,
            requested_seed_set=requested_seed_set,
            blocker_reason="missing_v1_seed_bundle",
            blocking_reasons=blocking_reasons,
        )
        blocker_summary = {
            **verdict,
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "full_update_p5_gate_blocker_summary",
        }
        verdict_path, blocker_summary_path, decision_path = _write_p5_gate_artifacts(
            output_dir=output_dir,
            verdict=verdict,
            blocker_summary=blocker_summary,
            gate_summary=gate_summary,
            gate_summary_path=gate_summary_path,
        )
        return _p5_gate_result(
            output_dir=output_dir,
            verdict=verdict,
            verdict_path=verdict_path,
            blocker_summary_path=blocker_summary_path,
            decision_path=decision_path,
        )

    inherited_seed_set = list(seed_bundle.get("seed_set") or [])
    if inherited_seed_set != list(requested_seed_set):
        blocking_reasons = _build_p5_skip_blockers(
            gate_summary=gate_summary,
            extra_reasons=["v1_seed_bundle_mismatch"],
        )
        verdict = _build_p5_skip_payload(
            baseline_authority_root=baseline_authority_root,
            v2_authority_root=v2_authority_root,
            output_dir=output_dir,
            gate_summary=gate_summary,
            gate_summary_path=gate_summary_path,
            seed_bundle=seed_bundle,
            requested_seed_set=requested_seed_set,
            blocker_reason="v1_seed_bundle_mismatch",
            blocking_reasons=blocking_reasons,
        )
        blocker_summary = {
            **verdict,
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "full_update_p5_gate_blocker_summary",
        }
        verdict_path, blocker_summary_path, decision_path = _write_p5_gate_artifacts(
            output_dir=output_dir,
            verdict=verdict,
            blocker_summary=blocker_summary,
            gate_summary=gate_summary,
            gate_summary_path=gate_summary_path,
        )
        return _p5_gate_result(
            output_dir=output_dir,
            verdict=verdict,
            verdict_path=verdict_path,
            blocker_summary_path=blocker_summary_path,
            decision_path=decision_path,
        )

    p5_gate_blockers = _p5_formal_gate_contract_blockers(
        gate_summary,
        gate_summary_path=gate_summary_path,
    )
    if p5_gate_blockers:
        blocker_reason = (
            "p5_formal_10ep_ineligible"
            if p5_gate_blockers == ["p5_formal_10ep_ineligible"]
            else "p5_formal_gate_not_passed"
        )
        blocking_reasons = _build_p5_skip_blockers(
            gate_summary=gate_summary,
            extra_reasons=[blocker_reason, *p5_gate_blockers],
        )
        verdict = _build_p5_skip_payload(
            baseline_authority_root=baseline_authority_root,
            v2_authority_root=v2_authority_root,
            output_dir=output_dir,
            gate_summary=gate_summary,
            gate_summary_path=gate_summary_path,
            seed_bundle=seed_bundle,
            requested_seed_set=requested_seed_set,
            blocker_reason=blocker_reason,
            blocking_reasons=blocking_reasons,
        )
        blocker_summary = {
            **verdict,
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "full_update_p5_gate_blocker_summary",
        }
        verdict_path, blocker_summary_path, decision_path = _write_p5_gate_artifacts(
            output_dir=output_dir,
            verdict=verdict,
            blocker_summary=blocker_summary,
            gate_summary=gate_summary,
            gate_summary_path=gate_summary_path,
        )
        return _p5_gate_result(
            output_dir=output_dir,
            verdict=verdict,
            verdict_path=verdict_path,
            blocker_summary_path=blocker_summary_path,
            decision_path=decision_path,
        )

    baseline_reference = _baseline_reference_payload(baseline_authority_root)
    baseline_summary_payload = baseline_reference.get("baseline_eval_summary")
    if not isinstance(baseline_summary_payload, Mapping):
        raise RuntimeError("baseline authority is missing t5 baseline eval_summary.json")
    baseline_validated = _validate_eval_summary_payload(
        baseline_summary_payload,
        label="baseline_eval_summary",
    )
    if list(baseline_validated["episode_seeds"]) != list(requested_seed_set):
        raise ValueError("baseline eval summary seeds do not match inherited v1 seed bundle")
    baseline_output_path = _copy_eval_summary(
        output_dir=output_dir,
        lane_name="baseline",
        payload=baseline_summary_payload,
    )
    baseline_eval = {
        "lane_name": "baseline",
        "status": "PASS",
        "source_summary_path": baseline_reference.get("baseline_eval_summary_path"),
        "output_summary_path": _safe_relpath(baseline_output_path),
        **baseline_validated,
    }

    conditioned_lane_state = _resolve_lane_state(conditioned_run_root)
    continuation_lane_state = _resolve_lane_state(continuation_run_root)
    if conditioned_lane_state.get("resolution") != "checkpoint":
        raise RuntimeError("conditioned lane does not expose a formal checkpoint for p5_gate")
    if continuation_lane_state.get("resolution") != "checkpoint":
        raise RuntimeError("continuation lane does not expose a formal checkpoint for p5_gate")

    conditioned_eval = _run_p5_eval_lane(
        lane_name="conditioned",
        lane_state=conditioned_lane_state,
        run_root=conditioned_run_root,
        output_dir=output_dir,
        requested_seed_set=requested_seed_set,
    )
    continuation_eval = _run_p5_eval_lane(
        lane_name="continuation",
        lane_state=continuation_lane_state,
        run_root=continuation_run_root,
        output_dir=output_dir,
        requested_seed_set=requested_seed_set,
    )

    verdict = _build_p5_executed_payload(
        baseline_authority_root=baseline_authority_root,
        v2_authority_root=v2_authority_root,
        output_dir=output_dir,
        gate_summary_path=gate_summary_path,
        seed_bundle=seed_bundle,
        requested_seed_set=requested_seed_set,
        baseline_eval=baseline_eval,
        conditioned_eval=conditioned_eval,
        continuation_eval=continuation_eval,
    )
    blocker_summary = {
        **verdict,
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "full_update_p5_gate_blocker_summary",
        "blocker_reason": None,
        "blocker_summary_path": _safe_relpath(
            output_dir / P5_BLOCKER_SUMMARY_FILENAME
        ),
    }
    verdict_path, blocker_summary_path, decision_path = _write_p5_gate_artifacts(
        output_dir=output_dir,
        verdict=verdict,
        blocker_summary=blocker_summary,
        gate_summary=gate_summary,
        gate_summary_path=gate_summary_path,
    )
    return _p5_gate_result(
        output_dir=output_dir,
        verdict=verdict,
        verdict_path=verdict_path,
        blocker_summary_path=blocker_summary_path,
        decision_path=decision_path,
        extra_paths={
            "baseline_eval_summary_path": str(output_dir / "baseline" / "eval_summary.json"),
            "conditioned_eval_summary_path": str(output_dir / "conditioned" / "eval_summary.json"),
            "continuation_eval_summary_path": str(output_dir / "continuation" / "eval_summary.json"),
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate Task 13 P4 full-update diagnostics.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("p4", "p5_gate"),
        help="Diagnostic mode. Task 13 implements p4 only.",
    )
    parser.add_argument("--baseline-authority-root", required=True)
    parser.add_argument("--v2-authority-root", required=True)
    parser.add_argument("--conditioned-run-root", required=True)
    parser.add_argument("--continuation-run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-end", type=int)
    parser.add_argument(
        "--baseline-v1-subgoal-override",
        help="Optional baseline_v1-only probe path override. Default resolver behavior is unchanged when omitted.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    baseline_v1_subgoal_override = (
        None
        if not args.baseline_v1_subgoal_override
        else _resolve_path(args.baseline_v1_subgoal_override)
    )
    if args.mode == "p4":
        result = run_p4_diagnostics(
            baseline_authority_root=_resolve_path(args.baseline_authority_root),
            v2_authority_root=_resolve_path(args.v2_authority_root),
            conditioned_run_root=_resolve_path(args.conditioned_run_root),
            continuation_run_root=_resolve_path(args.continuation_run_root),
            output_dir=_resolve_path(args.output_dir),
            baseline_v1_subgoal_override=baseline_v1_subgoal_override,
        )
    else:
        if args.seed_start is None or args.seed_end is None:
            parser.error("p5_gate requires --seed-start and --seed-end")
        result = run_p5_gate(
            baseline_authority_root=_resolve_path(args.baseline_authority_root),
            v2_authority_root=_resolve_path(args.v2_authority_root),
            conditioned_run_root=_resolve_path(args.conditioned_run_root),
            continuation_run_root=_resolve_path(args.continuation_run_root),
            output_dir=_resolve_path(args.output_dir),
            seed_start=int(args.seed_start),
            seed_end=int(args.seed_end),
            baseline_v1_subgoal_override=baseline_v1_subgoal_override,
        )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
