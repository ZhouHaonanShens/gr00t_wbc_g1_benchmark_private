#!/usr/bin/env python3
"""Validate and materialize Stage1 GR00T/OpenPI benchmark artifacts.

The Stage1 validator is intentionally CPU-only.  It reads machine-checkable
artifacts from the Stage1 run root, summarizes the current state, and refuses
unsafe claim escalation (P5 before clean P4, OpenPI smoke as benchmark, or
validator migration before parity).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RUN_ID = "stage1_p4_benchmark_validator_20260424T111232Z"
STAGE0_RUN_ID = "stage0_gated_next_gate_20260424T064818Z"
SUMMARY_SCHEMA = "dual_track_stage1_summary_v1"
FINAL_REPORT_SCHEMA = "stage1_final_artifact_validator_report_v1"
RESOURCE_REPORT_SCHEMA = "stage1_resource_boundary_report_v1"
GIT_HYGIENE_SCHEMA = "stage1_git_hygiene_report_v1"
OPENPI_PRIMARY_METRIC_ID = "success_rate@0.50_budget"

FORBIDDEN_INFERENCES = [
    "GR00T P4 PASS != P5 PASS",
    "GR00T P5 skip != runtime failure",
    "OpenPI smoke PASS != benchmark PASS",
    "OpenPI benchmark materialized != RECAP/state-side success",
    "validator migration parity != active referee replacement",
    "worker message != machine-checkable artifact",
]

P4_UNLOCK_FIELDS = (
    "status",
    "formal_claim_allowed",
    "blocking_reasons",
    "p5_formal_10ep_eligible",
)

STATUS_VALUES = {"PASS", "BLOCK", "SKIPPED"}
P5_DECISIONS = {"RUN", "SKIPPED", "BLOCKED"}
OPENPI_DECISIONS = {"REUSE", "RERUN", "BLOCK"}


class Stage1ValidationError(AssertionError):
    """Raised when a Stage1 artifact violates the Stage1 contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Stage1ValidationError(f"missing JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Stage1ValidationError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Stage1ValidationError(f"{path}: top-level JSON must be an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage1ValidationError(message)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [str(value)]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _default_run_root() -> Path:
    return Path("agent") / "artifacts" / RUN_ID


def _default_runtime_log_root() -> Path:
    return Path("agent") / "runtime_logs" / RUN_ID


def _repo_root_from_run_root(run_root: Path) -> Path:
    parts = run_root.parts
    if len(parts) >= 3 and parts[-3:-1] == ("agent", "artifacts"):
        return Path(*parts[:-3]) if parts[:-3] else Path(".")
    return Path(".")


def _contains_direct_sudo(text: str) -> bool:
    return re.search(r"(^|[;&|`$()\s])sudo(\s|$)", text) is not None


def _contains_forbidden_cuda_visible_devices(text: str) -> bool:
    for match in re.finditer(r"CUDA_VISIBLE_DEVICES\s*=\s*([^\s;]+)", text):
        visible = match.group(1).strip().strip('"\'')
        tokens = [token.strip() for token in visible.split(",")]
        if any(token in {"0", "3"} for token in tokens):
            return True
    return False


def _iter_commandish_text(payload: Any, *, parent_key: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            lowered = key_text.lower()
            if isinstance(value, (str, int, float, bool)) and (
                "command" in lowered
                or "shell" in lowered
                or lowered in {"cuda_visible_devices", "runtime_log"}
            ):
                yield key_text, str(value)
            elif isinstance(value, dict) and lowered == "env":
                for env_key, env_value in value.items():
                    if str(env_key) == "CUDA_VISIBLE_DEVICES":
                        yield f"{key_text}.{env_key}", f"CUDA_VISIBLE_DEVICES={env_value}"
                yield from _iter_commandish_text(value, parent_key=key_text)
            else:
                yield from _iter_commandish_text(value, parent_key=key_text)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            yield from _iter_commandish_text(item, parent_key=f"{parent_key}[{index}]")


def _validate_boundary_text(text: str, *, label: str) -> None:
    if _contains_direct_sudo(text):
        raise Stage1ValidationError(f"{label}: direct sudo command text is forbidden")
    if _contains_forbidden_cuda_visible_devices(text):
        raise Stage1ValidationError(f"{label}: CUDA_VISIBLE_DEVICES exposes GPU0/GPU3")


def _lane_family(lane: Any) -> str | None:
    lane_text = str(lane).lower()
    if lane_text.startswith("gr00t"):
        return "gr00t"
    if lane_text.startswith("openpi"):
        return "openpi"
    return None


def validate_resource_lease(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    prefix = f"{path}: "
    _require(payload.get("schema_version") == "resource_lease_v1", prefix + "schema_version must be resource_lease_v1")
    family = _lane_family(payload.get("lane"))
    _require(family in {"gr00t", "openpi"}, prefix + "lane must start with gr00t or openpi")
    expected_gpu = 1 if family == "gr00t" else 2
    _require(str(payload.get("gpu")) == str(expected_gpu), prefix + f"{family} lease must use GPU{expected_gpu}")
    _require(isinstance(payload.get("worker"), str) and bool(payload["worker"]), prefix + "worker required")
    _require(payload.get("forbidden_gpus_visible") is False, prefix + "forbidden_gpus_visible must be false")
    _require(payload.get("sudo_used") is False, prefix + "sudo_used must be false")
    _require(
        payload.get("direct_privileged_escalation_used") in {False, None},
        prefix + "direct_privileged_escalation_used must be false when present",
    )
    _require(payload.get("returncode") == 0, prefix + "returncode must be 0")
    _require(
        isinstance(payload.get("created_at_utc") or payload.get("started_at_utc") or payload.get("start_time"), str),
        prefix + "created_at_utc/started_at_utc/start_time required",
    )
    runtime_log = payload.get("runtime_log_dir") or payload.get("runtime_log")
    _require(isinstance(runtime_log, str) and runtime_log.strip(), prefix + "runtime_log_dir/runtime_log required")
    for label, text in _iter_commandish_text(payload):
        _validate_boundary_text(text, label=prefix + label)
    return payload


def _safe_load(path: Path, blockers: list[str], label: str) -> dict[str, Any] | None:
    try:
        return _load_json(path)
    except Stage1ValidationError as exc:
        blockers.append(f"{label}: {exc}")
        return None


def _status(payload: dict[str, Any] | None, *, default: str = "BLOCK") -> str:
    if payload is None:
        return default
    value = payload.get("status")
    return str(value) if isinstance(value, str) else default


def _is_clean_p4_gate(p4_gate: dict[str, Any] | None) -> bool:
    if not p4_gate:
        return False
    return (
        p4_gate.get("status") == "PASS"
        and p4_gate.get("formal_claim_allowed") is True
        and _string_list(p4_gate.get("blocking_reasons")) == []
        and p4_gate.get("p5_formal_10ep_eligible") is True
    )


def validate_p4_gate(run_root: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    validated: list[str] = []
    p4_root = run_root / "gr00t" / "p4_refresh"
    lease_path = p4_root / "resource_lease.json"
    input_path = p4_root / "p4_input_manifest.json"
    comp_path = p4_root / "comparability_manifest.json"
    diag_path = p4_root / "full_update_diagnostic_summary.json"
    gate_path = p4_root / "p4_gate_verdict.json"

    if lease_path.exists():
        try:
            validate_resource_lease(lease_path)
            validated.append(_rel(lease_path, run_root))
        except Stage1ValidationError as exc:
            blockers.append(f"gr00t P4 resource lease invalid: {exc}")
    else:
        blockers.append("gr00t P4 missing resource lease: gr00t/p4_refresh/resource_lease.json")

    input_manifest = _safe_load(input_path, blockers, "gr00t P4 input manifest")
    if input_manifest:
        validated.append(_rel(input_path, run_root))
        _require_or_block(
            blockers,
            input_manifest.get("schema_version") == "gr00t_p4_input_manifest_v1",
            "gr00t P4 input manifest schema_version must be gr00t_p4_input_manifest_v1",
        )
        _require_or_block(
            blockers,
            input_manifest.get("stage0_authority_run_id") == STAGE0_RUN_ID,
            f"gr00t P4 input manifest must reference Stage0 authority {STAGE0_RUN_ID}",
        )
        _require_or_block(
            blockers,
            len(_string_list(input_manifest.get("seed_bundle"))) >= 3,
            "gr00t P4 input manifest seed_bundle must include at least 3 paired seeds",
        )
    comparability = _safe_load(comp_path, blockers, "gr00t P4 comparability manifest")
    if comparability:
        validated.append(_rel(comp_path, run_root))
        _require_or_block(
            blockers,
            comparability.get("schema_version") == "gr00t_p4_comparability_manifest_v1",
            "gr00t comparability schema_version must be gr00t_p4_comparability_manifest_v1",
        )
        if comparability.get("status") == "PASS":
            _require_or_block(blockers, int(comparability.get("paired_seed_total", 0)) >= 3, "gr00t comparability PASS requires paired_seed_total >= 3")
            _require_or_block(
                blockers,
                int(comparability.get("paired_seed_improvement_count", 0)) >= 2,
                "gr00t comparability PASS requires paired_seed_improvement_count >= 2",
            )
            _require_or_block(blockers, _string_list(comparability.get("blocking_reasons")) == [], "gr00t comparability PASS requires empty blocking_reasons")
            for key in ("baseline_config_hash", "candidate_config_hash", "eval_condition_hash"):
                _require_or_block(blockers, isinstance(comparability.get(key), str) and bool(comparability[key]), f"gr00t comparability PASS requires {key}")
    if diag_path.exists():
        _safe_load(diag_path, blockers, "gr00t P4 full update diagnostic summary")
        validated.append(_rel(diag_path, run_root))
    else:
        blockers.append("gr00t P4 missing full_update_diagnostic_summary.json")

    p4_gate = _safe_load(gate_path, blockers, "gr00t P4 gate verdict")
    if p4_gate:
        validated.append(_rel(gate_path, run_root))
        _require_or_block(blockers, p4_gate.get("schema_version") == "gr00t_p4_gate_verdict_v1", "gr00t P4 gate schema_version must be gr00t_p4_gate_verdict_v1")
        _require_or_block(blockers, p4_gate.get("status") in {"PASS", "BLOCK"}, "gr00t P4 gate status must be PASS or BLOCK")
        p4_gate_blockers = _string_list(p4_gate.get("blocking_reasons"))
        if p4_gate.get("status") == "BLOCK":
            if p4_gate_blockers:
                blockers.extend(f"gr00t P4 gate blocker: {reason}" for reason in p4_gate_blockers)
            else:
                blockers.append("gr00t P4 gate is BLOCK without machine-readable blocking_reasons")
        if p4_gate.get("status") == "PASS":
            _require_or_block(blockers, comparability is not None and comparability.get("status") == "PASS", "gr00t P4 PASS requires comparability_manifest.status=PASS")
            _require_or_block(blockers, _is_clean_p4_gate(p4_gate), "gr00t P4 PASS must satisfy all four P5 unlock fields")
    status = "PASS" if p4_gate and p4_gate.get("status") == "PASS" and not blockers else "BLOCK"
    summary = {
        "status": status,
        "clean_p4_gate": _is_clean_p4_gate(p4_gate),
        "p5_formal_10ep_eligible": bool(p4_gate and p4_gate.get("p5_formal_10ep_eligible") is True),
        "blocking_reasons": blockers,
        "artifacts": sorted(set(validated)),
    }
    return summary, blockers, validated


def _require_or_block(blockers: list[str], condition: bool, message: str) -> None:
    if not condition:
        blockers.append(message)


def validate_p5_gate(run_root: Path, *, clean_p4_gate: bool) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    validated: list[str] = []
    p5_root = run_root / "gr00t" / "p5_gate"
    decision_path = p5_root / "p5_execution_decision.json"
    decision = _safe_load(decision_path, blockers, "gr00t P5 execution decision")
    decision_value = "BLOCKED"
    if decision:
        validated.append(_rel(decision_path, run_root))
        decision_value = str(decision.get("decision"))
        _require_or_block(blockers, decision.get("schema_version") == "gr00t_p5_execution_decision_v1", "gr00t P5 decision schema_version must be gr00t_p5_execution_decision_v1")
        _require_or_block(blockers, decision_value in P5_DECISIONS, "gr00t P5 decision must be RUN, SKIPPED, or BLOCKED")
        gate_inputs = decision.get("gate_inputs")
        if isinstance(gate_inputs, dict):
            missing = [field for field in P4_UNLOCK_FIELDS if field not in gate_inputs]
            _require_or_block(blockers, not missing, f"gr00t P5 gate_inputs missing fields: {missing}")
        if decision_value == "RUN":
            _require_or_block(blockers, clean_p4_gate, "gr00t P5 decision RUN is forbidden unless P4 gate is clean")
            for rel in ("resource_lease.json", "min_loop_verdict.json", "p5_gate_blocker_summary.json"):
                path = p5_root / rel
                if not path.exists():
                    blockers.append(f"gr00t P5 RUN missing {rel}")
                    continue
                if rel == "resource_lease.json":
                    try:
                        validate_resource_lease(path)
                    except Stage1ValidationError as exc:
                        blockers.append(f"gr00t P5 resource lease invalid: {exc}")
                else:
                    payload = _safe_load(path, blockers, f"gr00t P5 {rel}")
                    if rel == "min_loop_verdict.json" and payload:
                        episodes = payload.get("formal_episode_count", payload.get("episode_count", payload.get("episodes")))
                        if isinstance(episodes, int):
                            _require_or_block(blockers, episodes >= 10, "gr00t P5 min_loop_verdict must record at least 10 formal episodes")
                validated.append(_rel(path, run_root))
        else:
            decision_blockers = _string_list(decision.get("blocking_reasons"))
            _require_or_block(blockers, decision_blockers != [], "gr00t P5 SKIPPED/BLOCKED requires machine-readable blocking_reasons")
            if (p5_root / "resource_lease.json").exists():
                blockers.append(f"gr00t P5 {decision_value} must not include long-runtime artifact resource_lease.json")
            min_loop_path = p5_root / "min_loop_verdict.json"
            if min_loop_path.exists():
                min_loop = _safe_load(min_loop_path, blockers, "gr00t P5 min_loop_verdict")
                if min_loop:
                    _require_or_block(
                        blockers,
                        min_loop.get("status") == "SKIPPED"
                        and min_loop.get("formal_execution_attempted") is False
                        and min_loop.get("gate_mode") == "skipped",
                        "gr00t P5 non-RUN min_loop_verdict must be a skipped/no-runtime verdict",
                    )
                    validated.append(_rel(min_loop_path, run_root))
            blocker_summary_path = p5_root / "p5_gate_blocker_summary.json"
            if blocker_summary_path.exists():
                blocker_summary = _safe_load(blocker_summary_path, blockers, "gr00t P5 p5_gate_blocker_summary")
                if blocker_summary:
                    _require_or_block(
                        blockers,
                        blocker_summary.get("status") == "SKIPPED"
                        and blocker_summary.get("formal_execution_attempted") is False,
                        "gr00t P5 blocker summary must record skipped/no-runtime status",
                    )
                    validated.append(_rel(blocker_summary_path, run_root))
            else:
                blockers.append("gr00t P5 SKIPPED/BLOCKED missing p5_gate_blocker_summary.json")
    summary = {
        "decision": decision_value if decision_value in P5_DECISIONS else "BLOCKED",
        "status": "PASS" if decision_value == "RUN" and not blockers else ("SKIPPED" if decision_value == "SKIPPED" and not blockers else "BLOCK"),
        "decision_blocking_reasons": _string_list(decision.get("blocking_reasons")) if decision else [],
        "blocking_reasons": blockers,
        "artifacts": sorted(set(validated)),
    }
    return summary, blockers, validated


def _validate_go_no_go(payload: dict[str, Any], blockers: list[str]) -> None:
    _require_or_block(blockers, payload.get("schema_version") == "openpi_libero_go_no_go_report_v21", "OpenPI go_no_go_v21 schema_version must be openpi_libero_go_no_go_report_v21")
    _require_or_block(blockers, payload.get("authority_mode") == "strong", "OpenPI go_no_go_v21 authority_mode must be strong")
    _require_or_block(blockers, payload.get("primary_metric_id") == OPENPI_PRIMARY_METRIC_ID, f"OpenPI go_no_go_v21 primary_metric_id must be {OPENPI_PRIMARY_METRIC_ID}")
    gates = payload.get("gates")
    if isinstance(gates, dict):
        present = set(gates)
    elif isinstance(gates, list):
        present = {str(item.get("gate_id") if isinstance(item, dict) else item) for item in gates}
    else:
        present = set()
    required = {f"H{idx}" for idx in range(8)}
    _require_or_block(blockers, required <= present, "OpenPI go_no_go_v21 must include H0-H7 gates")


def validate_openpi(run_root: Path) -> tuple[dict[str, Any], list[str], list[str], bool]:
    blockers: list[str] = []
    validated: list[str] = []
    root = run_root / "openpi" / "benchmark_sweep"
    guard_path = run_root / "openpi" / "formal_status_guard" / "runtime_formal_status_unchanged.json"
    decision_path = root / "v21_reuse_or_rerun_decision.json"
    lease_path = root / "resource_lease.json"
    go_path = root / "go_no_go_v21.json"
    paired_path = root / "paired_summary_abcx_v21.json"
    rollout_path = root / "rollout_or_tracked_gate_summary.json"
    claim_path = root / "benchmark_claim_artifact.json"
    gate_review_path = root / "benchmark_claim_gate_review.json"

    decision = _safe_load(decision_path, blockers, "OpenPI v21 reuse/rerun decision")
    if decision:
        validated.append(_rel(decision_path, run_root))
        _require_or_block(blockers, decision.get("schema_version") == "openpi_v21_reuse_or_rerun_decision_v1", "OpenPI v21 decision schema_version must be openpi_v21_reuse_or_rerun_decision_v1")
        _require_or_block(blockers, decision.get("decision") in OPENPI_DECISIONS, "OpenPI v21 decision must be REUSE, RERUN, or BLOCK")
        _require_or_block(blockers, decision.get("authority_mode") == "strong", "OpenPI v21 decision authority_mode must be strong")
        _require_or_block(blockers, decision.get("primary_metric_id") == OPENPI_PRIMARY_METRIC_ID, f"OpenPI v21 decision primary_metric_id must be {OPENPI_PRIMARY_METRIC_ID}")
        if decision.get("decision") == "BLOCK":
            _require_or_block(blockers, _string_list(decision.get("blocking_reasons")) != [], "OpenPI v21 BLOCK decision requires blocking_reasons")
    if lease_path.exists():
        try:
            validate_resource_lease(lease_path)
            validated.append(_rel(lease_path, run_root))
        except Stage1ValidationError as exc:
            blockers.append(f"OpenPI resource lease invalid: {exc}")
    else:
        blockers.append("OpenPI missing benchmark_sweep/resource_lease.json")
    go = _safe_load(go_path, blockers, "OpenPI go_no_go_v21")
    if go:
        validated.append(_rel(go_path, run_root))
        _validate_go_no_go(go, blockers)
    for label, path in (("OpenPI paired_summary_abcx_v21", paired_path), ("OpenPI rollout_or_tracked_gate_summary", rollout_path)):
        if path.exists():
            _safe_load(path, blockers, label)
            validated.append(_rel(path, run_root))
        else:
            blockers.append(f"missing {label}: {_rel(path, run_root)}")

    claim = _safe_load(claim_path, blockers, "OpenPI benchmark claim artifact")
    benchmark_materialized = False
    benchmark_success_claimed = False
    recap_validated = False
    informative = False
    eligible_v22 = False
    known_risks: list[str] = []
    claim_gate_review: dict[str, Any] = {}
    if claim:
        validated.append(_rel(claim_path, run_root))
        for field in (
            "formal_benchmark_materialized",
            "benchmark_success_claimed",
            "recap_validated_on_desaturated_eval",
            "informativeness_validated",
            "eligible_for_state_side_v22",
        ):
            _require_or_block(blockers, isinstance(claim.get(field), bool), f"OpenPI benchmark_claim_artifact.{field} must be bool")
        benchmark_materialized = claim.get("formal_benchmark_materialized") is True
        benchmark_success_claimed = claim.get("benchmark_success_claimed") is True
        recap_validated = claim.get("recap_validated_on_desaturated_eval") is True
        informative = claim.get("informativeness_validated") is True
        eligible_v22 = claim.get("eligible_for_state_side_v22") is True
    guard = _safe_load(guard_path, blockers, "OpenPI runtime formal status guard")
    if guard:
        validated.append(_rel(guard_path, run_root))
        _require_or_block(blockers, guard.get("schema_version") == "openpi_runtime_formal_status_guard_v1", "OpenPI runtime guard schema_version must be openpi_runtime_formal_status_guard_v1")
        _require_or_block(blockers, guard.get("runtime_level_before") == "p2_overfit_or_tiny_update_pass", "OpenPI runtime guard before level must stay p2_overfit_or_tiny_update_pass")
        _require_or_block(blockers, guard.get("runtime_level_after") == "p2_overfit_or_tiny_update_pass", "OpenPI runtime guard after level must stay p2_overfit_or_tiny_update_pass")
        _require_or_block(blockers, guard.get("benchmark_success_encoded_in_runtime_status") is False, "OpenPI runtime status must not encode benchmark success")
        _require_or_block(blockers, guard.get("status") == "PASS", "OpenPI runtime guard status must be PASS")
    if gate_review_path.exists():
        review = _safe_load(gate_review_path, blockers, "OpenPI benchmark claim gate review")
        if review:
            validated.append(_rel(gate_review_path, run_root))
            _require_or_block(
                blockers,
                review.get("schema_version") == "openpi_stage1_benchmark_claim_gate_review_v1",
                "OpenPI benchmark claim gate review schema_version must be openpi_stage1_benchmark_claim_gate_review_v1",
            )
            _require_or_block(
                blockers,
                review.get("benchmark_runtime_started_by_worker3") is False
                and review.get("gpu_runtime_started_by_worker3") is False,
                "OpenPI benchmark claim gate review must not start GPU/runtime work",
            )
            checks = review.get("checks") if isinstance(review.get("checks"), dict) else {}
            canonical = checks.get("canonical_root_paths") if isinstance(checks, dict) and isinstance(checks.get("canonical_root_paths"), dict) else {}
            legacy_count = int(canonical.get("legacy_root_occurrence_count", 0)) if isinstance(canonical.get("legacy_root_occurrence_count", 0), int) else 0
            review_blockers = _string_list(review.get("blocking_reasons"))
            non_legacy_blockers = [reason for reason in review_blockers if reason != "legacy_media_root_paths_present"]
            _require_or_block(blockers, not non_legacy_blockers, f"OpenPI claim gate review has unexpected blockers: {non_legacy_blockers}")
            if legacy_count > 0 or "legacy_media_root_paths_present" in review_blockers:
                known_risks.append(
                    f"benchmark claim held: {legacy_count} legacy /media root paths remain in OpenPI source-artifact provenance"
                )
            claim_gate_review = {
                "artifact": _rel(gate_review_path, run_root),
                "status": review.get("status"),
                "claim_gate_decision": review.get("claim_gate_decision"),
                "blocking_reasons": review_blockers,
                "legacy_root_occurrence_count": legacy_count,
                "expected_current_root": canonical.get("expected_current_root"),
                "legacy_root": canonical.get("legacy_root"),
            }
    benchmark_claim_allowed = bool(benchmark_success_claimed and not blockers and go and guard)
    if benchmark_success_claimed and not benchmark_claim_allowed:
        blockers.append("OpenPI benchmark_success_claimed=true is forbidden without strong v21 bundle, runtime guard, and final validator allowance")
    status = "PASS" if not blockers else "BLOCK"
    summary = {
        "status": status,
        "formal_benchmark_materialized": benchmark_materialized,
        "benchmark_success_claimed": benchmark_success_claimed,
        "benchmark_claim_allowed_by_final_validator": benchmark_claim_allowed,
        "recap_validated_on_desaturated_eval": recap_validated,
        "informativeness_validated": informative,
        "eligible_for_state_side_v22": eligible_v22,
        "claim_gate_review": claim_gate_review,
        "known_risks": known_risks,
        "blocking_reasons": blockers,
        "artifacts": sorted(set(validated)),
    }
    return summary, blockers, validated, benchmark_claim_allowed


def validate_migration(run_root: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    validated: list[str] = []
    root = run_root / "validator_migration"
    plan_path = root / "migration_plan.json"
    matrix_path = root / "parity_matrix.json"
    verdict_path = root / "migration_verdict.json"
    plan = _safe_load(plan_path, blockers, "validator migration plan")
    if plan:
        validated.append(_rel(plan_path, run_root))
    matrix = _safe_load(matrix_path, blockers, "validator migration parity matrix")
    overall = "BLOCK"
    if matrix:
        validated.append(_rel(matrix_path, run_root))
        _require_or_block(blockers, matrix.get("schema_version") == "stage1_validator_parity_matrix_v1", "validator parity_matrix schema_version must be stage1_validator_parity_matrix_v1")
        _require_or_block(blockers, matrix.get("old_entrypoint") == "agent/run/stage0_artifact_validator.py", "validator parity old_entrypoint must be active Stage0 referee")
        fixtures = matrix.get("fixtures")
        _require_or_block(blockers, isinstance(fixtures, list) and len(fixtures) >= 2, "validator parity requires PASS and BLOCK fixtures")
        if isinstance(fixtures, list):
            _require_or_block(blockers, any(isinstance(item, dict) and item.get("old_status") == "PASS" and item.get("new_status") == "PASS" and item.get("equivalent") is True for item in fixtures), "validator parity requires current Stage0 PASS fixture equivalence")
            _require_or_block(blockers, any(isinstance(item, dict) and item.get("old_status") == "BLOCK" and item.get("new_status") == "BLOCK" and item.get("equivalent") is True for item in fixtures), "validator parity requires at least one BLOCK fixture equivalence")
        overall = str(matrix.get("overall_status", "BLOCK"))
        _require_or_block(blockers, overall in {"PASS", "BLOCK"}, "validator parity overall_status must be PASS or BLOCK")
    verdict = _safe_load(verdict_path, blockers, "validator migration verdict")
    if verdict:
        validated.append(_rel(verdict_path, run_root))
        _require_or_block(blockers, verdict.get("status") in {"PASS", "BLOCK"}, "validator migration verdict status must be PASS or BLOCK")
        if verdict.get("status") == "PASS":
            _require_or_block(blockers, overall == "PASS", "validator migration verdict PASS requires parity_matrix.overall_status=PASS")
    active_referee = "agent/run/stage0_artifact_validator.py"
    status = "PASS" if overall == "PASS" and verdict and verdict.get("status") == "PASS" and not blockers else "BLOCK"
    summary = {
        "status": status,
        "active_referee": active_referee,
        "parity_passed": status == "PASS",
        "blocking_reasons": blockers,
        "artifacts": sorted(set(validated)),
    }
    return summary, blockers, validated


def build_stage1_summary(run_root: Path, *, created_at_utc: str | None = None) -> dict[str, Any]:
    created = created_at_utc or _utc_now()
    p4_summary, p4_blockers, p4_validated = validate_p4_gate(run_root)
    p5_summary, p5_blockers, p5_validated = validate_p5_gate(run_root, clean_p4_gate=bool(p4_summary["clean_p4_gate"]))
    openpi_summary, openpi_blockers, openpi_validated, benchmark_claim_allowed = validate_openpi(run_root)
    migration_summary, migration_blockers, migration_validated = validate_migration(run_root)
    blockers = [
        *p4_blockers,
        *p5_blockers,
        *openpi_blockers,
        *migration_blockers,
    ]
    validator_outputs = sorted(set([*p4_validated, *p5_validated, *openpi_validated, *migration_validated]))
    completion_claim_allowed = not blockers and p4_summary["status"] == "PASS" and openpi_summary["status"] == "PASS" and migration_summary["status"] == "PASS"
    if openpi_summary["benchmark_success_claimed"] and not benchmark_claim_allowed:
        completion_claim_allowed = False
    return {
        "schema_version": SUMMARY_SCHEMA,
        "run_id": RUN_ID,
        "created_at_utc": created,
        "gr00t": {
            "p4_refresh": p4_summary,
            "p5_gate": p5_summary,
        },
        "openpi": {
            "benchmark_sweep": {key: value for key, value in openpi_summary.items() if key not in {"formal_benchmark_materialized", "benchmark_success_claimed", "recap_validated_on_desaturated_eval", "eligible_for_state_side_v22"}},
            "formal_benchmark_materialized": openpi_summary["formal_benchmark_materialized"],
            "benchmark_success_claimed": openpi_summary["benchmark_success_claimed"],
            "recap_validated_on_desaturated_eval": openpi_summary["recap_validated_on_desaturated_eval"],
            "eligible_for_state_side_v22": openpi_summary["eligible_for_state_side_v22"],
        },
        "validator_migration": migration_summary,
        "completion_claim_allowed": completion_claim_allowed,
        "blocking_reasons": blockers,
        "forbidden_inferences": FORBIDDEN_INFERENCES,
        "validator_outputs": validator_outputs,
    }


def validate_stage1_summary(path: Path, *, run_root: Path) -> dict[str, Any]:
    payload = _load_json(path)
    prefix = f"{path}: "
    _require(payload.get("schema_version") == SUMMARY_SCHEMA, prefix + f"schema_version must be {SUMMARY_SCHEMA}")
    _require(payload.get("run_id") == RUN_ID, prefix + f"run_id must be {RUN_ID}")
    for forbidden in FORBIDDEN_INFERENCES:
        _require(forbidden in _string_list(payload.get("forbidden_inferences")), prefix + f"missing forbidden inference: {forbidden}")
    gr00t = payload.get("gr00t")
    openpi = payload.get("openpi")
    migration = payload.get("validator_migration")
    _require(isinstance(gr00t, dict), prefix + "gr00t must be object")
    _require(isinstance(openpi, dict), prefix + "openpi must be object")
    _require(isinstance(migration, dict), prefix + "validator_migration must be object")
    p4 = gr00t.get("p4_refresh") if isinstance(gr00t, dict) else None
    p5 = gr00t.get("p5_gate") if isinstance(gr00t, dict) else None
    _require(isinstance(p4, dict), prefix + "gr00t.p4_refresh must be object")
    _require(isinstance(p5, dict), prefix + "gr00t.p5_gate must be object")
    _require(p4.get("status") in STATUS_VALUES, prefix + "gr00t.p4_refresh.status invalid")
    _require(p5.get("decision") in P5_DECISIONS, prefix + "gr00t.p5_gate.decision invalid")
    _require(p5.get("status") in STATUS_VALUES, prefix + "gr00t.p5_gate.status invalid")
    if p5.get("decision") == "RUN":
        _require(p4.get("clean_p4_gate") is True, prefix + "P5 RUN requires gr00t.p4_refresh.clean_p4_gate=true")
    if p4.get("status") != "PASS" and p5.get("decision") == "RUN":
        raise Stage1ValidationError(prefix + "P5 RUN cannot follow non-PASS P4")
    bench = openpi.get("benchmark_sweep")
    _require(isinstance(bench, dict), prefix + "openpi.benchmark_sweep must be object")
    for field in ("formal_benchmark_materialized", "benchmark_success_claimed", "recap_validated_on_desaturated_eval", "eligible_for_state_side_v22"):
        _require(isinstance(openpi.get(field), bool), prefix + f"openpi.{field} must be bool")
    if openpi.get("benchmark_success_claimed") is True:
        _require(bench.get("benchmark_claim_allowed_by_final_validator") is True, prefix + "benchmark_success_claimed requires final-validator allowance")
    _require(migration.get("status") in {"PASS", "BLOCK"}, prefix + "validator_migration.status invalid")
    if migration.get("active_referee") != "agent/run/stage0_artifact_validator.py" and migration.get("parity_passed") is not True:
        raise Stage1ValidationError(prefix + "validator migration cannot replace active referee before parity PASS")
    blockers = _string_list(payload.get("blocking_reasons"))
    _require(isinstance(payload.get("completion_claim_allowed"), bool), prefix + "completion_claim_allowed must be bool")
    if blockers:
        _require(payload.get("completion_claim_allowed") is False, prefix + "completion_claim_allowed must be false while blockers exist")
    _require(isinstance(payload.get("validator_outputs"), list), prefix + "validator_outputs must be list")
    return payload


def _validate_runtime_log_tree(runtime_log_root: Path) -> list[str]:
    blockers: list[str] = []
    if not runtime_log_root.exists():
        return blockers
    for path in sorted(runtime_log_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".npy", ".npz", ".pt", ".pth"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            blockers.append(f"runtime log unreadable {path}: {exc}")
            continue
        try:
            _validate_boundary_text(text, label=str(path))
        except Stage1ValidationError as exc:
            blockers.append(str(exc))
    return blockers


def _is_validator_migration_output_fixture(path: Path, run_root: Path) -> bool:
    """Return true for old/new validator parity fixture outputs.

    Worker5 intentionally stores negative resource-boundary fixtures under
    ``validator_migration/*_validator_outputs``.  Those files prove the old and
    new validators still reject bad leases; they are not Stage1 runtime leases
    and must not make the independent runtime-boundary report fail.
    """

    try:
        rel_parts = path.resolve().relative_to(run_root.resolve()).parts
    except ValueError:
        return False
    return len(rel_parts) >= 2 and rel_parts[0] == "validator_migration" and rel_parts[1] in {
        "old_validator_outputs",
        "new_validator_outputs",
    }


def build_resource_boundary_report(run_root: Path, runtime_log_root: Path, *, created_at_utc: str | None = None) -> dict[str, Any]:
    blockers: list[str] = []
    validated_leases: list[str] = []
    used_gpus: list[int] = []
    for path in sorted(run_root.rglob("resource_lease.json")) if run_root.exists() else []:
        if _is_validator_migration_output_fixture(path, run_root):
            continue
        try:
            payload = validate_resource_lease(path)
            validated_leases.append(_rel(path, run_root))
            try:
                used_gpus.append(int(payload["gpu"]))
            except (KeyError, TypeError, ValueError):
                pass
        except Stage1ValidationError as exc:
            blockers.append(str(exc))
    blockers.extend(_validate_runtime_log_tree(runtime_log_root))
    return {
        "schema_version": RESOURCE_REPORT_SCHEMA,
        "run_id": RUN_ID,
        "created_at_utc": created_at_utc or _utc_now(),
        "status": "PASS" if not blockers else "BLOCK",
        "blocking_reasons": blockers,
        "validated_resource_leases": validated_leases,
        "used_gpus": sorted(set(used_gpus)),
        "forbidden_gpus_visible": False if not blockers else any("GPU0/GPU3" in item or "CUDA_VISIBLE_DEVICES" in item for item in blockers),
        "sudo_used": any("sudo" in item.lower() for item in blockers),
    }


def validate_resource_boundary_report(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    prefix = f"{path}: "
    _require(payload.get("schema_version") == RESOURCE_REPORT_SCHEMA, prefix + f"schema_version must be {RESOURCE_REPORT_SCHEMA}")
    _require(payload.get("run_id") == RUN_ID, prefix + f"run_id must be {RUN_ID}")
    _require(payload.get("status") in {"PASS", "BLOCK"}, prefix + "status must be PASS or BLOCK")
    _require(isinstance(payload.get("blocking_reasons"), list), prefix + "blocking_reasons must be list")
    return payload


def _tracked_large_files(repo_root: Path, *, max_bytes: int = 5 * 1024 * 1024) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    large: list[str] = []
    for rel in proc.stdout.splitlines():
        path = repo_root / rel
        try:
            if path.is_file() and path.stat().st_size > max_bytes:
                large.append(rel)
        except OSError:
            continue
    return large


def build_git_hygiene_report(run_root: Path, *, created_at_utc: str | None = None) -> dict[str, Any]:
    repo_root = _repo_root_from_run_root(run_root)
    large = [rel for rel in _tracked_large_files(repo_root) if rel.startswith(_rel(run_root, repo_root))]
    return {
        "schema_version": GIT_HYGIENE_SCHEMA,
        "run_id": RUN_ID,
        "created_at_utc": created_at_utc or _utc_now(),
        "status": "PASS" if not large else "BLOCK",
        "tracked_large_artifact_candidates_over_5mib": large,
        "notes": "Final clean git status is verified by command output after commit; this artifact records tracked large-file hygiene.",
    }


def validate_git_hygiene_report(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    prefix = f"{path}: "
    _require(payload.get("schema_version") == GIT_HYGIENE_SCHEMA, prefix + f"schema_version must be {GIT_HYGIENE_SCHEMA}")
    _require(payload.get("status") in {"PASS", "BLOCK"}, prefix + "status must be PASS or BLOCK")
    _require(isinstance(payload.get("tracked_large_artifact_candidates_over_5mib"), list), prefix + "tracked_large_artifact_candidates_over_5mib must be list")
    return payload


def _focused_pytest_blockers(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing focused pytest log: {_rel(path, path.parents[1]) if len(path.parents) > 1 else path}"]
    text = path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    blockers: list[str] = []
    if "pytest_exit=0" not in lowered and " passed" not in lowered:
        blockers.append("focused pytest log does not contain pytest_exit=0 or passed marker")
    for marker in ("pytest_exit=1", " failed", " error"):
        if marker in lowered:
            blockers.append(f"focused pytest log contains failure marker {marker.strip()}")
    return blockers


def build_final_validator_report(run_root: Path, runtime_log_root: Path, *, created_at_utc: str | None = None) -> dict[str, Any]:
    created = created_at_utc or _utc_now()
    verifier = run_root / "verifier"
    summary_path = verifier / "dual_track_stage1_summary.json"
    resource_path = verifier / "resource_boundary_report.json"
    focused_pytest_path = verifier / "focused_pytest.log"
    git_hygiene_path = verifier / "git_hygiene_report.json"
    blocking_reasons: list[str] = []
    validated: list[str] = []
    summary_completion_claim_allowed = False

    if summary_path.exists():
        try:
            summary = validate_stage1_summary(summary_path, run_root=run_root)
            validated.append(_rel(summary_path, run_root))
            blocking_reasons.extend(f"summary blocker: {reason}" for reason in _string_list(summary.get("blocking_reasons")))
            summary_completion_claim_allowed = summary.get("completion_claim_allowed") is True
            if not summary_completion_claim_allowed:
                blocking_reasons.append("summary completion_claim_allowed=false")
        except Stage1ValidationError as exc:
            blocking_reasons.append(str(exc))
    else:
        blocking_reasons.append("missing final summary: verifier/dual_track_stage1_summary.json")

    if resource_path.exists():
        try:
            resource = validate_resource_boundary_report(resource_path)
            validated.append(_rel(resource_path, run_root))
            blocking_reasons.extend(f"resource boundary blocker: {reason}" for reason in _string_list(resource.get("blocking_reasons")))
        except Stage1ValidationError as exc:
            blocking_reasons.append(str(exc))
    else:
        blocking_reasons.append("missing resource boundary report: verifier/resource_boundary_report.json")

    if git_hygiene_path.exists():
        try:
            hygiene = validate_git_hygiene_report(git_hygiene_path)
            validated.append(_rel(git_hygiene_path, run_root))
            for rel in _string_list(hygiene.get("tracked_large_artifact_candidates_over_5mib")):
                blocking_reasons.append(f"large tracked artifact candidate over 5MiB: {rel}")
        except Stage1ValidationError as exc:
            blocking_reasons.append(str(exc))
    else:
        blocking_reasons.append("missing git hygiene report: verifier/git_hygiene_report.json")

    if focused_pytest_path.exists():
        validated.append(_rel(focused_pytest_path, run_root))
    blocking_reasons.extend(_focused_pytest_blockers(focused_pytest_path))
    blocking_reasons.extend(_validate_runtime_log_tree(runtime_log_root))
    for path in sorted(run_root.rglob("resource_lease.json")) if run_root.exists() else []:
        if _is_validator_migration_output_fixture(path, run_root):
            continue
        try:
            validate_resource_lease(path)
            validated.append(_rel(path, run_root))
        except Stage1ValidationError as exc:
            blocking_reasons.append(str(exc))

    large = [rel for rel in _tracked_large_files(_repo_root_from_run_root(run_root)) if rel.startswith(_rel(run_root, _repo_root_from_run_root(run_root)))]
    for rel in large:
        blocking_reasons.append(f"large tracked artifact candidate over 5MiB: {rel}")

    unique_blockers = sorted(set(blocking_reasons))
    return {
        "schema_version": FINAL_REPORT_SCHEMA,
        "run_id": RUN_ID,
        "created_at_utc": created,
        "stage": "stage1_final_summary_gate",
        "status": "PASS" if not unique_blockers else "BLOCK",
        "blocking_reasons": unique_blockers,
        "summary_file": "verifier/dual_track_stage1_summary.json",
        "resource_boundary_report": "verifier/resource_boundary_report.json",
        "focused_pytest_log": "verifier/focused_pytest.log",
        "git_hygiene_report": "verifier/git_hygiene_report.json",
        "validated_artifacts": sorted(set(validated)),
        "final_claim_language": {
            "completion_claim_allowed": bool(summary_completion_claim_allowed and not unique_blockers),
            "must_not_claim": FORBIDDEN_INFERENCES if unique_blockers else [],
        },
    }


def write_stage1_summary(run_root: Path, *, created_at_utc: str | None = None) -> dict[str, Any]:
    summary = build_stage1_summary(run_root, created_at_utc=created_at_utc)
    verifier = run_root / "verifier"
    _write_json(verifier / "dual_track_stage1_summary.draft.json", summary)
    _write_json(verifier / "dual_track_stage1_summary.json", summary)
    validate_stage1_summary(verifier / "dual_track_stage1_summary.json", run_root=run_root)
    return summary


def write_resource_boundary_report(run_root: Path, runtime_log_root: Path, *, created_at_utc: str | None = None) -> dict[str, Any]:
    report = build_resource_boundary_report(run_root, runtime_log_root, created_at_utc=created_at_utc)
    path = run_root / "verifier" / "resource_boundary_report.json"
    _write_json(path, report)
    validate_resource_boundary_report(path)
    return report


def write_git_hygiene_report(run_root: Path, *, created_at_utc: str | None = None) -> dict[str, Any]:
    report = build_git_hygiene_report(run_root, created_at_utc=created_at_utc)
    path = run_root / "verifier" / "git_hygiene_report.json"
    _write_json(path, report)
    validate_git_hygiene_report(path)
    return report


def write_final_validator_report(run_root: Path, runtime_log_root: Path, *, created_at_utc: str | None = None) -> dict[str, Any]:
    report = build_final_validator_report(run_root, runtime_log_root, created_at_utc=created_at_utc)
    path = run_root / "verifier" / "final_artifact_validator_report.json"
    _write_json(path, report)
    validate_final_validator_report(path, run_root=run_root)
    return report


def validate_final_validator_report(path: Path, *, run_root: Path) -> dict[str, Any]:
    payload = _load_json(path)
    prefix = f"{path}: "
    _require(payload.get("schema_version") == FINAL_REPORT_SCHEMA, prefix + f"schema_version must be {FINAL_REPORT_SCHEMA}")
    _require(payload.get("run_id") == RUN_ID, prefix + f"run_id must be {RUN_ID}")
    _require(payload.get("stage") == "stage1_final_summary_gate", prefix + "stage must be stage1_final_summary_gate")
    _require(payload.get("status") in {"PASS", "BLOCK"}, prefix + "status must be PASS or BLOCK")
    _require(isinstance(payload.get("blocking_reasons"), list), prefix + "blocking_reasons must be list")
    _require(payload.get("summary_file") == "verifier/dual_track_stage1_summary.json", prefix + "summary_file mismatch")
    _require(payload.get("resource_boundary_report") == "verifier/resource_boundary_report.json", prefix + "resource_boundary_report mismatch")
    _require(isinstance(payload.get("validated_artifacts"), list), prefix + "validated_artifacts must be list")
    claim = payload.get("final_claim_language")
    _require(isinstance(claim, dict), prefix + "final_claim_language must be object")
    if payload.get("status") == "BLOCK":
        _require(claim.get("completion_claim_allowed") is False, prefix + "BLOCK report must set completion_claim_allowed=false")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument("--runtime-log-root", type=Path, default=_default_runtime_log_root())
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--write-resource-report", action="store_true")
    parser.add_argument("--write-git-hygiene-report", action="store_true")
    parser.add_argument("--final", action="store_true", help="write final validator report")
    parser.add_argument("--all", action="store_true", help="write summary, resource, git hygiene, and final report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.all or args.write_resource_report:
            write_resource_boundary_report(args.run_root, args.runtime_log_root)
        if args.all or args.write_git_hygiene_report:
            write_git_hygiene_report(args.run_root)
        if args.all or args.write_summary:
            write_stage1_summary(args.run_root)
        if args.all or args.final:
            report = write_final_validator_report(args.run_root, args.runtime_log_root)
            print(f"{report['status']}: wrote {args.run_root / 'verifier' / 'final_artifact_validator_report.json'}")
        return 0
    except Stage1ValidationError as exc:
        print(f"BLOCK: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
