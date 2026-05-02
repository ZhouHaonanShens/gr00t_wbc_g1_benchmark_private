#!/usr/bin/env python3
"""Validate and materialize Stage0-gated GR00T/OpenPI artifacts.

This command owns the CPU-only Worker5 gate in the Stage0 next-gate plan:

* validate W1/W4 Stage0 schema artifacts;
* write authoritative non-draft ready JSON files for W2/W3;
* validate the W6 final draft summary and write the final validator report.

It deliberately performs artifact/schema checks only.  It does not run training,
evaluation, privileged commands, or GPU workloads.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.run.dual_track_artifact_validator import (
    ValidationError,
    validate_resource_lease,
    validate_runtime_log_boundaries,
)

RUN_ID = "stage0_gated_next_gate_20260424T064818Z"
STAGE0_READY_SCHEMA = "stage0_ready_v1"
STAGE0_VALIDATOR_REPORT_SCHEMA = "stage0_validator_report_v1"
GR00T_UPDATE_SPEC_SCHEMA = "gr00t_update_probe_spec_v1"
GR00T_ROOT_CAUSE_MATRIX_SCHEMA = "gr00t_root_cause_matrix_v1"
OPENPI_BENCHMARK_GATE_CONTRACT_SCHEMA = "openpi_benchmark_gate_contract_v1"
FINAL_ARTIFACT_VALIDATOR_REPORT_SCHEMA = "final_artifact_validator_report_v1"
NEXT_STAGE_SUMMARY_SCHEMA = "dual_track_next_stage_summary_v1"
WORKER_ID = "worker-5"

GR00T_KNOWN_BLOCKERS = {"ALL_ZERO_PARAM_DELTA", "STATIC_AUDIT_BLOCK"}
GR00T_ALLOWED_CLASSIFICATIONS = {
    "optimizer_not_stepping",
    "zero_lr_param_group",
    "frozen_or_unreachable_params",
    "delta_capture_bug",
    "checkpoint_selection_bug",
    "scope_routing_bug",
    "expected_no_update",
    "inconclusive",
}
GR00T_REQUIRED_EVIDENCE_FIELDS = {
    "optimizer_step_observed",
    "optimizer_param_group_count",
    "trainable_param_count",
    "nonzero_lr_group_count",
    "requires_grad_count",
    "grad_norm_nonzero",
    "param_delta_nonzero",
    "before_after_checkpoint_hashes",
}
OPENPI_PRIMARY_METRIC_ID = "success_rate@0.50_budget"
OPENPI_PRIMARY_METRIC_ORDER = [
    "success_rate@0.50_budget",
    "success_rate@0.75_budget",
    "throughput_like_score",
]
OPENPI_REQUIRED_BENCHMARK_ARTIFACTS = {
    "go_no_go_report.json",
    "rollout_or_tracked_gate_summary.json",
    "resource_lease.json",
    "benchmark_claim_artifact.json",
}
NEXT_STAGE_FORBIDDEN_INFERENCES = (
    "GR00T min repro PASS != P5 eligibility",
    "OpenPI P2 runtime PASS != benchmark PASS",
    "worker message != machine-checkable ready artifact",
)


class Stage0ValidationError(ValidationError):
    """Raised when a Stage0 artifact violates the gated-next schema."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Stage0ValidationError(f"{path}: missing JSON artifact") from exc
    except json.JSONDecodeError as exc:
        raise Stage0ValidationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise Stage0ValidationError(f"{path}: top-level JSON must be an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage0ValidationError(message)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [str(value)]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _artifact_rel(path: Path, run_root: Path) -> str:
    try:
        return path.resolve().relative_to(run_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_artifact(path: str, run_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else run_root / candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record_error(errors: list[str], label: str, check: Callable[[], object]) -> object | None:
    try:
        return check()
    except Stage0ValidationError as exc:
        errors.append(f"{label}: {exc}")
        return None


def _require_json_object_file(path: Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    payload = _load_json(path)
    if expected_schema is not None and "schema_version" in payload:
        _require(
            payload.get("schema_version") == expected_schema,
            f"{path}: schema_version must be {expected_schema}",
        )
    return payload


def validate_gr00t_update_probe_spec(path: Path) -> dict[str, Any]:
    """Validate the minimal W1 update-probe spec wrapper."""

    payload = _require_json_object_file(path, expected_schema=GR00T_UPDATE_SPEC_SCHEMA)
    prefix = f"{path}: "
    if "schema_version" in payload:
        _require(
            payload.get("schema_version") == GR00T_UPDATE_SPEC_SCHEMA,
            prefix + f"schema_version must be {GR00T_UPDATE_SPEC_SCHEMA}",
        )
    _require(payload, prefix + "spec must not be empty")
    return payload


def validate_gr00t_root_cause_matrix(path: Path) -> dict[str, Any]:
    """Validate ``gr00t/stage0/root_cause_matrix.json``."""

    payload = _load_json(path)
    prefix = f"{path}: "
    _require(
        payload.get("schema_version") == GR00T_ROOT_CAUSE_MATRIX_SCHEMA,
        prefix + f"schema_version must be {GR00T_ROOT_CAUSE_MATRIX_SCHEMA}",
    )
    observed_blockers = _string_list(payload.get("observed_blockers"))
    _require(observed_blockers, prefix + "observed_blockers must be a non-empty list")
    absent_explanations = payload.get("absent_blocker_explanations", {})
    _require(
        isinstance(absent_explanations, dict),
        prefix + "absent_blocker_explanations must be an object when provided",
    )
    for blocker in sorted(GR00T_KNOWN_BLOCKERS - set(observed_blockers)):
        _require(
            isinstance(absent_explanations.get(blocker), str)
            and bool(absent_explanations[blocker].strip()),
            prefix + f"missing known blocker {blocker} requires absent_blocker_explanations",
        )

    candidate_findings = payload.get("candidate_findings")
    _require(isinstance(candidate_findings, list), prefix + "candidate_findings must be a list")
    _require(candidate_findings, prefix + "candidate_findings must not be empty")

    findings_by_blocker: dict[str, list[dict[str, Any]]] = {}
    for index, item in enumerate(candidate_findings):
        item_prefix = prefix + f"candidate_findings[{index}]: "
        _require(isinstance(item, dict), item_prefix + "must be an object")
        _require(
            isinstance(item.get("candidate_id"), str) and item["candidate_id"].strip(),
            item_prefix + "candidate_id required",
        )
        observed = item.get("observed_blocker")
        _require(
            isinstance(observed, str) and observed.strip(),
            item_prefix + "observed_blocker required",
        )
        classification = item.get("classification")
        _require(
            classification in GR00T_ALLOWED_CLASSIFICATIONS,
            item_prefix
            + "classification must be one of "
            + ", ".join(sorted(GR00T_ALLOWED_CLASSIFICATIONS)),
        )
        evidence = item.get("evidence")
        _require(isinstance(evidence, dict), item_prefix + "evidence must be an object")
        missing_evidence = sorted(GR00T_REQUIRED_EVIDENCE_FIELDS - set(evidence))
        _require(
            not missing_evidence,
            item_prefix + f"evidence missing required fields: {missing_evidence}",
        )
        _require(
            isinstance(evidence.get("before_after_checkpoint_hashes"), list),
            item_prefix + "evidence.before_after_checkpoint_hashes must be a list",
        )
        if observed == "STATIC_AUDIT_BLOCK":
            has_static_detail = any(
                key in evidence
                for key in (
                    "static_audit_details",
                    "static_audit_block_reasons",
                    "static_audit_status",
                )
            )
            _require(
                has_static_detail,
                item_prefix + "STATIC_AUDIT_BLOCK requires static audit details",
            )
        _require(
            isinstance(item.get("next_probe"), str) and item["next_probe"].strip(),
            item_prefix + "next_probe required",
        )
        findings_by_blocker.setdefault(str(observed), []).append(item)

    for blocker in sorted(GR00T_KNOWN_BLOCKERS & set(observed_blockers)):
        _require(
            blocker in findings_by_blocker,
            prefix + f"{blocker} requires at least one candidate finding",
        )

    assessment = payload.get("same_root_cause_assessment")
    _require(
        isinstance(assessment, dict),
        prefix + "same_root_cause_assessment must be an object",
    )
    _require(
        assessment.get("zero_lr_trainable_param_group_same_as_all_zero_param_delta")
        in {"yes", "no", "unknown"},
        prefix
        + "same_root_cause_assessment.zero_lr_trainable_param_group_same_as_all_zero_param_delta invalid",
    )
    _require(
        isinstance(assessment.get("rationale"), str) and assessment["rationale"].strip(),
        prefix + "same_root_cause_assessment.rationale required",
    )
    return payload


def validate_openpi_benchmark_gate_contract(path: Path) -> dict[str, Any]:
    """Validate ``openpi/stage0/benchmark_gate_contract.json``."""

    payload = _load_json(path)
    prefix = f"{path}: "
    _require(
        payload.get("schema_version") == OPENPI_BENCHMARK_GATE_CONTRACT_SCHEMA,
        prefix + f"schema_version must be {OPENPI_BENCHMARK_GATE_CONTRACT_SCHEMA}",
    )
    expected_literals: dict[str, Any] = {
        "runtime_prereq": "p2_overfit_or_tiny_update_pass",
        "runtime_pass_is_benchmark_pass": False,
        "benchmark_success_claimed_default": False,
        "authority_mode": "strong",
        "lite_is_advisory_only": True,
        "primary_metric_id": OPENPI_PRIMARY_METRIC_ID,
    }
    for key, expected in expected_literals.items():
        _require(payload.get(key) == expected, prefix + f"{key} must be {expected!r}")
    _require(
        payload.get("primary_metric_order") == OPENPI_PRIMARY_METRIC_ORDER,
        prefix + "primary_metric_order must preserve v21 ordering",
    )
    required_artifacts = set(_string_list(payload.get("required_artifacts")))
    _require(
        OPENPI_REQUIRED_BENCHMARK_ARTIFACTS.issubset(required_artifacts),
        prefix
        + "required_artifacts missing "
        + repr(sorted(OPENPI_REQUIRED_BENCHMARK_ARTIFACTS - required_artifacts)),
    )
    rules = payload.get("claim_placement_rules")
    _require(isinstance(rules, dict), prefix + "claim_placement_rules must be an object")
    _require(
        rules.get("must_not_modify_runtime_formal_status") is True,
        prefix + "claim_placement_rules.must_not_modify_runtime_formal_status must be true",
    )
    may_set = str(rules.get("may_set_benchmark_success_claimed", ""))
    _require(
        "validator PASS" in may_set or "validator" in may_set,
        prefix + "claim placement must require validator PASS",
    )
    return payload


def _early_paths(run_root: Path, lane: str) -> dict[str, Path]:
    if lane == "gr00t":
        stage_root = run_root / "gr00t" / "stage0"
        return {
            "stage_root": stage_root,
            "spec": stage_root / "gr00t_update_probe_spec_v1.json",
            "matrix": stage_root / "root_cause_matrix.json",
            "report": stage_root / "gr00t_stage0_validator_report.json",
            "ready": stage_root / "gr00t_stage0_ready.json",
        }
    if lane == "openpi":
        stage_root = run_root / "openpi" / "stage0"
        return {
            "stage_root": stage_root,
            "contract": stage_root / "benchmark_gate_contract.json",
            "report": stage_root / "openpi_benchmark_gate_validator_report.json",
            "ready": stage_root / "benchmark_gate_ready.json",
        }
    raise ValueError(f"unsupported lane: {lane}")


def _ready_contract(lane: str) -> tuple[str, str]:
    if lane == "gr00t":
        return "worker-2", "gpu1_min_repro"
    if lane == "openpi":
        return "worker-3", "gpu2_benchmark_candidate"
    raise ValueError(f"unsupported lane: {lane}")


def build_early_gate_payloads(
    run_root: Path, lane: str, *, created_at_utc: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build W5 early validator report and authoritative ready payloads."""

    created_at = created_at_utc or _utc_now()
    paths = _early_paths(run_root, lane)
    errors: list[str] = []
    validated_paths: list[Path] = []
    validated_schema_versions: list[str] = []
    spec_path: Path

    if lane == "gr00t":
        spec_path = paths["spec"]
        spec = _record_error(
            errors,
            "gr00t_update_probe_spec_v1",
            lambda: validate_gr00t_update_probe_spec(paths["spec"]),
        )
        if spec is not None:
            validated_paths.append(paths["spec"])
            validated_schema_versions.append(str(spec.get("schema_version", GR00T_UPDATE_SPEC_SCHEMA)))
        matrix = _record_error(
            errors,
            "root_cause_matrix",
            lambda: validate_gr00t_root_cause_matrix(paths["matrix"]),
        )
        if matrix is not None:
            validated_paths.append(paths["matrix"])
            validated_schema_versions.append(str(matrix["schema_version"]))
    else:
        spec_path = paths["contract"]
        contract = _record_error(
            errors,
            "benchmark_gate_contract",
            lambda: validate_openpi_benchmark_gate_contract(paths["contract"]),
        )
        if contract is not None:
            validated_paths.append(paths["contract"])
            validated_schema_versions.append(str(contract["schema_version"]))

    status = "PASS" if not errors else "BLOCK"
    blocking_reasons = errors
    authority_inputs = [_artifact_rel(path, run_root) for path in validated_paths] if status == "PASS" else []
    report_path = paths["report"]
    ready_path = paths["ready"]
    report_rel = _artifact_rel(report_path, run_root)
    ready_rel = _artifact_rel(ready_path, run_root)
    allowed_worker, allowed_stage = _ready_contract(lane)

    report = {
        "schema_version": STAGE0_VALIDATOR_REPORT_SCHEMA,
        "lane": lane,
        "stage": "stage0_ready_gate",
        "status": status,
        "blocking_reasons": blocking_reasons,
        "ready_file": ready_rel,
        "validated_artifacts": [_artifact_rel(path, run_root) for path in validated_paths],
        "validated_schema_versions": sorted(set(validated_schema_versions)),
        "created_by_worker": WORKER_ID,
        "created_at_utc": created_at,
    }
    ready = {
        "schema_version": STAGE0_READY_SCHEMA,
        "lane": lane,
        "ready": status == "PASS",
        "blocking_reasons": blocking_reasons,
        "authority_inputs": authority_inputs,
        "validator_outputs": [report_rel],
        "spec_hash": _sha256_file(spec_path) if spec_path.exists() else "sha256:missing",
        "created_by_worker": WORKER_ID,
        "created_at_utc": created_at,
        "finalized_by_worker": WORKER_ID,
        "allowed_next_worker": allowed_worker,
        "allowed_next_stage": allowed_stage,
    }
    return report, ready


def write_early_gate(
    run_root: Path, lane: str, *, created_at_utc: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write W5 early validator report before the authoritative ready file."""

    paths = _early_paths(run_root, lane)
    report, ready = build_early_gate_payloads(run_root, lane, created_at_utc=created_at_utc)
    _write_json(paths["report"], report)
    _write_json(paths["ready"], ready)
    validate_stage0_validator_report(paths["report"], run_root=run_root, expected_lane=lane)
    validate_stage0_ready_file(paths["ready"], run_root=run_root, expected_lane=lane)
    return report, ready


def _list_existing_artifact_paths(
    payload: dict[str, Any], key: str, run_root: Path, prefix: str, *, require_existing: bool
) -> list[Path]:
    values = payload.get(key)
    _require(isinstance(values, list), prefix + f"{key} must be a list")
    paths = []
    for index, value in enumerate(values):
        _require(isinstance(value, str) and value.strip(), prefix + f"{key}[{index}] must be a path")
        path = _resolve_artifact(value, run_root)
        if require_existing:
            _require(path.exists(), prefix + f"{key}[{index}] does not exist: {value}")
        paths.append(path)
    return paths


def validate_stage0_validator_report(
    path: Path,
    *,
    run_root: Path,
    expected_lane: str | None = None,
    check_ready: bool = True,
) -> dict[str, Any]:
    """Validate a W5 early Stage0 validator report."""

    payload = _load_json(path)
    prefix = f"{path}: "
    _require(
        payload.get("schema_version") == STAGE0_VALIDATOR_REPORT_SCHEMA,
        prefix + f"schema_version must be {STAGE0_VALIDATOR_REPORT_SCHEMA}",
    )
    lane = payload.get("lane")
    _require(lane in {"gr00t", "openpi"}, prefix + "lane must be gr00t or openpi")
    if expected_lane is not None:
        _require(lane == expected_lane, prefix + f"lane must be {expected_lane}")
    _require(payload.get("stage") == "stage0_ready_gate", prefix + "stage must be stage0_ready_gate")
    status = payload.get("status")
    _require(status in {"PASS", "BLOCK"}, prefix + "status must be PASS or BLOCK")
    blocking_reasons = _string_list(payload.get("blocking_reasons"))
    _require(
        isinstance(payload.get("blocking_reasons"), list),
        prefix + "blocking_reasons must be a list",
    )
    if status == "PASS":
        _require(not blocking_reasons, prefix + "PASS requires empty blocking_reasons")
    else:
        _require(blocking_reasons, prefix + "BLOCK requires non-empty blocking_reasons")
    _require(
        isinstance(payload.get("ready_file"), str) and payload["ready_file"].strip(),
        prefix + "ready_file required",
    )
    _list_existing_artifact_paths(
        payload,
        "validated_artifacts",
        run_root,
        prefix,
        require_existing=status == "PASS",
    )
    _require(
        isinstance(payload.get("validated_schema_versions"), list),
        prefix + "validated_schema_versions must be a list",
    )
    _require(payload.get("created_by_worker") == WORKER_ID, prefix + f"created_by_worker must be {WORKER_ID}")
    _require(
        isinstance(payload.get("created_at_utc"), str) and payload["created_at_utc"].endswith("Z"),
        prefix + "created_at_utc must be an ISO UTC string ending in Z",
    )
    if check_ready:
        ready_path = _resolve_artifact(str(payload["ready_file"]), run_root)
        if status == "PASS":
            ready = validate_stage0_ready_file(ready_path, run_root=run_root, expected_lane=str(lane))
            _require(ready.get("ready") is True, prefix + "PASS report must point to ready=true file")
        elif ready_path.exists():
            ready = validate_stage0_ready_file(ready_path, run_root=run_root, expected_lane=str(lane))
            _require(ready.get("ready") is False, prefix + "BLOCK report must not point to ready=true file")
    return payload


def _expected_ready_spec_path(run_root: Path, lane: str) -> Path:
    paths = _early_paths(run_root, lane)
    return paths["spec"] if lane == "gr00t" else paths["contract"]


def validate_stage0_ready_file(
    path: Path, *, run_root: Path, expected_lane: str | None = None
) -> dict[str, Any]:
    """Validate an authoritative Stage0 ready file."""

    payload = _load_json(path)
    prefix = f"{path}: "
    _require(
        payload.get("schema_version") == STAGE0_READY_SCHEMA,
        prefix + f"schema_version must be {STAGE0_READY_SCHEMA}",
    )
    lane = payload.get("lane")
    _require(lane in {"gr00t", "openpi"}, prefix + "lane must be gr00t or openpi")
    if expected_lane is not None:
        _require(lane == expected_lane, prefix + f"lane must be {expected_lane}")
    _require(isinstance(payload.get("ready"), bool), prefix + "ready must be bool")
    _require(isinstance(payload.get("blocking_reasons"), list), prefix + "blocking_reasons must be a list")
    blocking_reasons = _string_list(payload.get("blocking_reasons"))
    if payload["ready"]:
        _require(not blocking_reasons, prefix + "ready=true requires empty blocking_reasons")
    else:
        _require(blocking_reasons, prefix + "ready=false requires non-empty blocking_reasons")

    _list_existing_artifact_paths(
        payload,
        "authority_inputs",
        run_root,
        prefix,
        require_existing=bool(payload["ready"]),
    )
    validator_outputs = _list_existing_artifact_paths(
        payload,
        "validator_outputs",
        run_root,
        prefix,
        require_existing=True,
    )
    _require(validator_outputs, prefix + "validator_outputs must not be empty")
    for output_path in validator_outputs:
        _require(
            output_path.name.endswith("_validator_report.json"),
            prefix + "validator_outputs must reference W5 validator reports",
        )
        report = validate_stage0_validator_report(
            output_path,
            run_root=run_root,
            expected_lane=str(lane),
            check_ready=False,
        )
        if payload["ready"]:
            _require(report.get("status") == "PASS", prefix + "ready=true requires PASS validator report")
            _require(report.get("created_by_worker") == WORKER_ID, prefix + "validator report must be W5-created")

    spec_hash = payload.get("spec_hash")
    _require(
        isinstance(spec_hash, str) and spec_hash.startswith("sha256:"),
        prefix + "spec_hash must start with sha256:",
    )
    if payload["ready"]:
        spec_path = _expected_ready_spec_path(run_root, str(lane))
        _require(spec_path.exists(), prefix + f"ready=true requires spec/contract file: {spec_path}")
        _require(spec_hash == _sha256_file(spec_path), prefix + "spec_hash does not match spec/contract")

    _require(
        payload.get("created_by_worker") == WORKER_ID or payload.get("finalized_by_worker") == WORKER_ID,
        prefix + "created_by_worker or finalized_by_worker must be worker-5",
    )
    _require(
        isinstance(payload.get("created_at_utc"), str) and payload["created_at_utc"].endswith("Z"),
        prefix + "created_at_utc must be an ISO UTC string ending in Z",
    )
    allowed_worker, allowed_stage = _ready_contract(str(lane))
    _require(payload.get("allowed_next_worker") == allowed_worker, prefix + f"allowed_next_worker must be {allowed_worker}")
    _require(payload.get("allowed_next_stage") == allowed_stage, prefix + f"allowed_next_stage must be {allowed_stage}")
    return payload


def validate_stage0_dependency_ready(
    path: Path, *, run_root: Path, expected_lane: str
) -> dict[str, Any]:
    """Validate the exact non-draft ready artifact consumed by W2/W3."""

    _require(
        not path.name.endswith(".draft.json"),
        f"{path}: draft ready files must not satisfy dependency checks",
    )
    ready = validate_stage0_ready_file(path, run_root=run_root, expected_lane=expected_lane)
    _require(ready.get("ready") is True, f"{path}: dependency ready file must have ready=true")
    return ready


def _summary_stage(payload: dict[str, Any], lane: str, stage: str) -> dict[str, Any]:
    section = payload.get(lane)
    _require(isinstance(section, dict), f"summary: {lane} must be an object")
    stage_payload = section.get(stage)
    _require(isinstance(stage_payload, dict), f"summary: {lane}.{stage} must be an object")
    _require(
        isinstance(stage_payload.get("status"), str) and stage_payload["status"],
        f"summary: {lane}.{stage}.status required",
    )
    return stage_payload


def _status(stage_payload: dict[str, Any]) -> str:
    return str(stage_payload.get("status"))


def _validate_openpi_benchmark_claim(run_root: Path) -> None:
    candidate_root = run_root / "openpi" / "gpu2_benchmark_candidate"
    paths = {
        "resource_lease.json": candidate_root / "resource_lease.json",
        "go_no_go_report.json": candidate_root / "go_no_go_report.json",
        "benchmark_claim_artifact.json": candidate_root / "benchmark_claim_artifact.json",
        "rollout_or_tracked_gate_summary.json": candidate_root / "rollout_or_tracked_gate_summary.json",
    }
    for name, path in paths.items():
        _require(path.exists(), f"openpi benchmark claim requires {name}")
    _validate_stage0_resource_lease(paths["resource_lease.json"])
    _require_json_object_file(paths["go_no_go_report.json"])
    _require_json_object_file(paths["rollout_or_tracked_gate_summary.json"])
    claim = _load_json(paths["benchmark_claim_artifact.json"])
    prefix = f"{paths['benchmark_claim_artifact.json']}: "
    _require(claim.get("benchmark_success_claimed") is True, prefix + "benchmark_success_claimed must be true")
    _require(claim.get("primary_metric_id") == OPENPI_PRIMARY_METRIC_ID, prefix + "primary_metric_id mismatch")
    _require(claim.get("authority_mode") == "strong", prefix + "authority_mode must be strong")

    formal_status = run_root / "openpi" / "formal_status.json"
    if formal_status.exists():
        formal = _load_json(formal_status)
        _require(
            formal.get("benchmark_success_claimed") is not True
            and formal.get("benchmark_claim_allowed") is not True,
            f"{formal_status}: runtime formal status must not encode benchmark success",
        )


def validate_next_stage_summary_draft(path: Path, *, run_root: Path) -> dict[str, Any]:
    """Validate W6's final-summary draft before W5 writes the final report."""

    payload = _load_json(path)
    prefix = f"{path}: "
    _require(
        payload.get("schema_version") == NEXT_STAGE_SUMMARY_SCHEMA,
        prefix + f"schema_version must be {NEXT_STAGE_SUMMARY_SCHEMA}",
    )
    _require(payload.get("run_id") == RUN_ID, prefix + f"run_id must be {RUN_ID}")
    for lane, stage in (
        ("gr00t", "stage0"),
        ("gr00t", "gpu1_min_repro"),
        ("gr00t", "p4_refresh"),
        ("gr00t", "p5_gate"),
        ("openpi", "stage0"),
        ("openpi", "gpu2_benchmark_candidate"),
    ):
        _summary_stage(payload, lane, stage)
    _require(
        _status(_summary_stage(payload, "gr00t", "p4_refresh")) in {"PASS", "BLOCK", "SKIPPED"},
        prefix + "gr00t.p4_refresh.status must be PASS/BLOCK/SKIPPED",
    )
    _require(
        _status(_summary_stage(payload, "gr00t", "p5_gate")) in {"PASS", "BLOCK", "SKIPPED"},
        prefix + "gr00t.p5_gate.status must be PASS/BLOCK/SKIPPED",
    )
    openpi = payload.get("openpi")
    _require(isinstance(openpi, dict), prefix + "openpi must be an object")
    runtime_level = openpi.get("runtime_level")
    _require(isinstance(runtime_level, str) and runtime_level, prefix + "openpi.runtime_level required")
    if not runtime_level.startswith("blocked_"):
        _require(
            runtime_level == "p2_overfit_or_tiny_update_pass",
            prefix + "openpi.runtime_level must remain p2_overfit_or_tiny_update_pass unless blocked",
        )
    benchmark_success_claimed = openpi.get("benchmark_success_claimed")
    _require(isinstance(benchmark_success_claimed, bool), prefix + "openpi.benchmark_success_claimed must be bool")
    _require(
        isinstance(payload.get("completion_claim_allowed"), bool),
        prefix + "completion_claim_allowed must be bool",
    )
    _require(isinstance(payload.get("validator_outputs"), list), prefix + "validator_outputs must be a list")
    forbidden = payload.get("forbidden_inferences")
    _require(isinstance(forbidden, list), prefix + "forbidden_inferences must be a list")
    forbidden_text = "\n".join(str(item) for item in forbidden)
    for snippet in NEXT_STAGE_FORBIDDEN_INFERENCES:
        _require(snippet in forbidden_text, prefix + f"missing forbidden inference: {snippet}")

    for lane, ready_rel in (
        ("gr00t", "gr00t/stage0/gr00t_stage0_ready.json"),
        ("openpi", "openpi/stage0/benchmark_gate_ready.json"),
    ):
        stage_status = _status(_summary_stage(payload, lane, "stage0"))
        ready_path = run_root / ready_rel
        if stage_status == "PASS":
            validate_stage0_dependency_ready(ready_path, run_root=run_root, expected_lane=lane)
        elif ready_path.exists():
            ready = validate_stage0_ready_file(ready_path, run_root=run_root, expected_lane=lane)
            _require(
                ready.get("ready") is not True,
                prefix + f"{lane}.stage0 non-PASS summary cannot point at ready=true",
            )

    if benchmark_success_claimed:
        _validate_openpi_benchmark_claim(run_root)
    else:
        _require(
            payload.get("completion_claim_allowed") is not True,
            prefix + "completion_claim_allowed cannot be true without benchmark success claim",
        )
    return payload


def _default_runtime_log_root(run_root: Path) -> Path:
    if run_root.parent.name == "artifacts":
        return run_root.parent.parent / "runtime_logs" / run_root.name
    return Path("agent/runtime_logs") / run_root.name


def _find_large_files(run_root: Path, *, max_bytes: int = 5 * 1024 * 1024) -> list[str]:
    if not run_root.exists():
        return []
    return [
        _artifact_rel(path, run_root)
        for path in run_root.rglob("*")
        if path.is_file() and path.stat().st_size > max_bytes
    ]


def _focused_pytest_log_blockers(path: Path) -> list[str]:
    """Return blocking reasons from W6's focused pytest log.

    The final validator must not treat mere log presence as evidence of a
    passing test run.  W6 records a compact ``pytest_exit=<code>`` trailer when
    it can; older/smaller smoke logs may instead contain an explicit PASS token.
    """

    text = path.read_text(encoding="utf-8", errors="replace")
    blockers: list[str] = []
    pytest_exit: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("pytest_exit="):
            pytest_exit = stripped.split("=", 1)[1].strip()
    if pytest_exit is not None:
        if pytest_exit != "0":
            blockers.append(f"focused pytest log reports pytest_exit={pytest_exit}")
        return blockers

    lowered = text.lower()
    if "focused pytest pass" in lowered or "pytest pass" in lowered:
        return blockers
    if " failed" in lowered or "failed," in lowered or "error" in lowered:
        blockers.append("focused pytest log contains failure/error text without pytest_exit=0")
    else:
        blockers.append("focused pytest log lacks pytest_exit=0 or explicit PASS marker")
    return blockers


def _summary_blockers(payload: dict[str, Any]) -> list[str]:
    blockers = [
        reason
        for reason in _string_list(payload.get("blocking_reasons"))
        # W6 is required to draft before W5 writes this report.  The draft may
        # therefore contain a sequencing placeholder that becomes obsolete as
        # soon as W5 materializes the final validator report.
        if "final_artifact_validator_report.json is missing" not in reason
    ]
    return sorted(set(blockers))


def _report_status_blocker(path: Path, *, label: str) -> str | None:
    try:
        payload = _require_json_object_file(path)
    except Stage0ValidationError as exc:
        return str(exc)
    status = str(payload.get("status", "")).upper()
    if status and status != "PASS":
        return f"{label} status is {status}"
    return None


def _iter_commandish_text(payload: Any, *, parent_key: str = "") -> list[tuple[str, str]]:
    """Extract command/env-like strings from structured runtime evidence.

    Some Stage0 JSON logs include prose such as "no sudo tokens"; that should
    not be treated as executing sudo.  Boundary scanning for structured logs is
    therefore limited to command/env fields, while unstructured text logs keep
    the stricter raw-text scan.
    """

    hits: list[tuple[str, str]] = []
    key = parent_key.lower()
    commandish = any(token in key for token in ("command", "shell", "argv", "env"))
    if isinstance(payload, dict):
        for child_key, value in payload.items():
            child_label = f"{parent_key}.{child_key}" if parent_key else str(child_key)
            hits.extend(_iter_commandish_text(value, parent_key=child_label))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            child_label = f"{parent_key}[{index}]"
            hits.extend(_iter_commandish_text(value, parent_key=child_label))
    elif isinstance(payload, str) and commandish:
        hits.append((parent_key, payload))
    return hits


def _validate_stage0_runtime_log_boundaries(path: Path) -> None:
    if path.suffix == ".json":
        try:
            payload = _load_json(path)
        except Stage0ValidationError:
            validate_runtime_log_boundaries(path)
            return
        for label, text in _iter_commandish_text(payload):
            _validate_stage0_boundary_text(text, label=f"{path}:{label}")
        return
    validate_runtime_log_boundaries(path)


def _validate_runtime_log_tree(runtime_log_root: Path) -> None:
    if not runtime_log_root.exists():
        return
    for path in runtime_log_root.rglob("*"):
        if path.is_file():
            _validate_stage0_runtime_log_boundaries(path)


def _validate_stage0_boundary_text(text: str, *, label: str) -> None:
    _require(
        "sudo " not in text and "\nsudo" not in text,
        f"{label}: evidence must not run sudo directly",
    )
    _require(
        not re.search(r"CUDA_VISIBLE_DEVICES\s*=\s*(?:0|3|0,|3,|.*,0|.*,3)", text),
        f"{label}: evidence references forbidden GPU0/GPU3 CUDA visibility",
    )


def _command_text(command_payload: dict[str, Any]) -> str:
    raw_command = command_payload.get("command")
    command_shell = command_payload.get("command_shell")
    if isinstance(command_shell, str):
        return command_shell
    if isinstance(raw_command, str):
        return raw_command
    if isinstance(raw_command, list):
        return " ".join(str(part) for part in raw_command)
    return ""


def _lease_lane_family(lane: Any) -> str | None:
    lane_text = str(lane)
    if lane_text == "gr00t" or lane_text.startswith("gr00t_"):
        return "gr00t"
    if lane_text == "openpi" or lane_text.startswith("openpi_"):
        return "openpi"
    return None


def _validate_stage0_resource_lease(path: Path) -> dict[str, Any]:
    """Validate Stage0 resource leases, including command-list manifests.

    Earlier dual-track leases used one top-level ``command``/``runtime_log``.
    Stage0 GPU lanes may instead record a short command history under
    ``commands`` plus ``runtime_log_dir``.  Accept both without weakening the
    GPU/sudo boundary checks.
    """

    try:
        return validate_resource_lease(path)
    except ValidationError as original_error:
        payload = _load_json(path)
        prefix = f"{path}: "
        commands = payload.get("commands")
        has_command_history = isinstance(commands, list) and bool(commands)
        top_level_command = _command_text(payload)
        if not has_command_history and not top_level_command.strip():
            raise original_error

        _require(
            payload.get("schema_version") == "resource_lease_v1",
            prefix + "wrong schema_version",
        )
        lane = payload.get("lane")
        lane_family = _lease_lane_family(lane)
        _require(
            lane_family in {"gr00t", "openpi"},
            prefix + "lane must be gr00t/openpi or a Stage0 lane prefixed with gr00t_/openpi_",
        )
        expected_gpu = "1" if lane_family == "gr00t" else "2"
        _require(str(payload.get("gpu")) == expected_gpu, prefix + f"{lane} must use GPU{expected_gpu}")
        _require(isinstance(payload.get("worker"), str) and payload["worker"], prefix + "worker required")
        _require(payload.get("forbidden_gpus_visible") is False, prefix + "forbidden_gpus_visible must be false")
        _require(payload.get("sudo_used") is False, prefix + "sudo_used must be false")
        _require(payload.get("direct_privileged_escalation_used") in {False, None}, prefix + "direct privileged escalation must be false")
        _require(isinstance(payload.get("returncode"), int), prefix + "returncode required")
        _require(
            isinstance(payload.get("created_at_utc") or payload.get("started_at_utc"), str),
            prefix + "created_at_utc/started_at_utc required",
        )
        _require(
            isinstance(payload.get("runtime_log") or payload.get("runtime_log_dir"), str),
            prefix + "runtime_log/runtime_log_dir required",
        )

        if top_level_command.strip():
            _validate_stage0_boundary_text(top_level_command, label=prefix + "command")
            _require(
                isinstance(payload.get("ended_at_utc") or payload.get("end_time"), str)
                or has_command_history,
                prefix + "ended_at_utc/end_time required for single-command leases",
            )
            _require(
                isinstance(payload.get("timeout_s", payload.get("timeout_seconds")), (int, float))
                or has_command_history,
                prefix + "timeout_s/timeout_seconds required for single-command leases",
            )

        if has_command_history:
            assert isinstance(commands, list)
            for index, command_payload in enumerate(commands):
                item_prefix = prefix + f"commands[{index}]: "
                _require(isinstance(command_payload, dict), item_prefix + "must be an object")
                command_text = _command_text(command_payload)
                _require(command_text.strip(), item_prefix + "command or command_shell required")
                _validate_stage0_boundary_text(command_text, label=item_prefix + "command")
                env = command_payload.get("env")
                if isinstance(env, dict):
                    _validate_stage0_boundary_text(
                        " ".join(f"{key}={value}" for key, value in env.items()),
                        label=item_prefix + "env",
                    )
                _require(
                    isinstance(command_payload.get("returncode"), int)
                    or command_payload.get("timed_out") is True,
                    item_prefix + "returncode or timeout evidence required",
                )
        return payload


def _validate_resource_lease_tree(run_root: Path) -> list[str]:
    validated: list[str] = []
    for path in run_root.rglob("resource_lease.json") if run_root.exists() else []:
        _validate_stage0_resource_lease(path)
        validated.append(_artifact_rel(path, run_root))
    return validated


def build_final_validator_report(
    run_root: Path,
    *,
    runtime_log_root: Path | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build W5's final validator report against W6's draft summary."""

    runtime_logs = runtime_log_root or _default_runtime_log_root(run_root)
    created_at = created_at_utc or _utc_now()
    summary_rel = "verifier/dual_track_next_stage_summary.draft.json"
    summary_path = run_root / summary_rel
    focused_pytest_rel = "verifier/focused_pytest.log"
    focused_pytest_path = run_root / focused_pytest_rel
    resource_report_rel = "verifier/resource_boundary_report.json"
    resource_report_path = run_root / resource_report_rel
    git_hygiene_rel = "verifier/git_hygiene_report.json"
    git_hygiene_path = run_root / git_hygiene_rel

    blocking_reasons: list[str] = []
    validated_artifacts: list[str] = []

    if summary_path.exists():
        try:
            summary = validate_next_stage_summary_draft(summary_path, run_root=run_root)
            validated_artifacts.append(summary_rel)
            blocking_reasons.extend(
                f"draft summary blocker: {reason}" for reason in _summary_blockers(summary)
            )
        except Stage0ValidationError as exc:
            blocking_reasons.append(str(exc))
    else:
        blocking_reasons.append(f"missing final draft summary: {summary_rel}")

    if focused_pytest_path.exists():
        validated_artifacts.append(focused_pytest_rel)
        blocking_reasons.extend(_focused_pytest_log_blockers(focused_pytest_path))
    else:
        blocking_reasons.append(f"missing focused pytest log: {focused_pytest_rel}")

    if resource_report_path.exists():
        try:
            _require_json_object_file(resource_report_path)
            validated_artifacts.append(resource_report_rel)
            status_blocker = _report_status_blocker(
                resource_report_path, label=resource_report_rel
            )
            if status_blocker is not None:
                blocking_reasons.append(status_blocker)
        except Stage0ValidationError as exc:
            blocking_reasons.append(str(exc))
    else:
        blocking_reasons.append(f"missing resource boundary report: {resource_report_rel}")

    if git_hygiene_path.exists():
        try:
            _require_json_object_file(git_hygiene_path)
            validated_artifacts.append(git_hygiene_rel)
            status_blocker = _report_status_blocker(
                git_hygiene_path, label=git_hygiene_rel
            )
            if status_blocker is not None:
                blocking_reasons.append(status_blocker)
        except Stage0ValidationError as exc:
            blocking_reasons.append(str(exc))

    try:
        _validate_runtime_log_tree(runtime_logs)
    except ValidationError as exc:
        blocking_reasons.append(str(exc))

    try:
        validated_artifacts.extend(_validate_resource_lease_tree(run_root))
    except ValidationError as exc:
        blocking_reasons.append(str(exc))

    large_files = _find_large_files(run_root)
    if large_files:
        blocking_reasons.append("large tracked artifact candidates over 5MiB: " + ", ".join(large_files))

    return {
        "schema_version": FINAL_ARTIFACT_VALIDATOR_REPORT_SCHEMA,
        "stage": "final_summary_gate",
        "status": "BLOCK" if blocking_reasons else "PASS",
        "blocking_reasons": blocking_reasons,
        "summary_file": summary_rel,
        "validated_artifacts": sorted(set(validated_artifacts)),
        "focused_pytest_log": focused_pytest_rel,
        "resource_boundary_report": resource_report_rel,
        "created_by_worker": WORKER_ID,
        "created_at_utc": created_at,
    }


def write_final_validator_report(
    run_root: Path,
    *,
    runtime_log_root: Path | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    report = build_final_validator_report(
        run_root, runtime_log_root=runtime_log_root, created_at_utc=created_at_utc
    )
    path = run_root / "verifier" / "final_artifact_validator_report.json"
    _write_json(path, report)
    validate_final_validator_report(path, run_root=run_root)
    return report


def validate_final_validator_report(path: Path, *, run_root: Path) -> dict[str, Any]:
    payload = _load_json(path)
    prefix = f"{path}: "
    _require(
        payload.get("schema_version") == FINAL_ARTIFACT_VALIDATOR_REPORT_SCHEMA,
        prefix + f"schema_version must be {FINAL_ARTIFACT_VALIDATOR_REPORT_SCHEMA}",
    )
    _require(payload.get("stage") == "final_summary_gate", prefix + "stage must be final_summary_gate")
    _require(payload.get("status") in {"PASS", "BLOCK"}, prefix + "status must be PASS or BLOCK")
    _require(isinstance(payload.get("blocking_reasons"), list), prefix + "blocking_reasons must be a list")
    if payload.get("status") == "PASS":
        _require(not payload["blocking_reasons"], prefix + "PASS requires empty blocking_reasons")
    else:
        _require(payload["blocking_reasons"], prefix + "BLOCK requires non-empty blocking_reasons")
    _require(
        payload.get("summary_file") == "verifier/dual_track_next_stage_summary.draft.json",
        prefix + "summary_file must be verifier/dual_track_next_stage_summary.draft.json",
    )
    _list_existing_artifact_paths(
        payload,
        "validated_artifacts",
        run_root,
        prefix,
        require_existing=payload.get("status") == "PASS",
    )
    _require(
        payload.get("focused_pytest_log") == "verifier/focused_pytest.log",
        prefix + "focused_pytest_log path mismatch",
    )
    _require(
        payload.get("resource_boundary_report") == "verifier/resource_boundary_report.json",
        prefix + "resource_boundary_report path mismatch",
    )
    _require(payload.get("created_by_worker") == WORKER_ID, prefix + f"created_by_worker must be {WORKER_ID}")
    _require(
        isinstance(payload.get("created_at_utc"), str) and payload["created_at_utc"].endswith("Z"),
        prefix + "created_at_utc must be an ISO UTC string ending in Z",
    )
    return payload


__all__ = [
    "FINAL_ARTIFACT_VALIDATOR_REPORT_SCHEMA",
    "GR00T_ALLOWED_CLASSIFICATIONS",
    "GR00T_KNOWN_BLOCKERS",
    "GR00T_REQUIRED_EVIDENCE_FIELDS",
    "GR00T_ROOT_CAUSE_MATRIX_SCHEMA",
    "GR00T_UPDATE_SPEC_SCHEMA",
    "NEXT_STAGE_FORBIDDEN_INFERENCES",
    "NEXT_STAGE_SUMMARY_SCHEMA",
    "OPENPI_BENCHMARK_GATE_CONTRACT_SCHEMA",
    "OPENPI_PRIMARY_METRIC_ID",
    "OPENPI_PRIMARY_METRIC_ORDER",
    "OPENPI_REQUIRED_BENCHMARK_ARTIFACTS",
    "RUN_ID",
    "STAGE0_READY_SCHEMA",
    "STAGE0_VALIDATOR_REPORT_SCHEMA",
    "Stage0ValidationError",
    "ValidationError",
    "WORKER_ID",
    "_artifact_rel",
    "_command_text",
    "_default_runtime_log_root",
    "_early_paths",
    "_expected_ready_spec_path",
    "_find_large_files",
    "_focused_pytest_log_blockers",
    "_iter_commandish_text",
    "_lease_lane_family",
    "_list_existing_artifact_paths",
    "_load_json",
    "_parser",
    "_ready_contract",
    "_record_error",
    "_report_status_blocker",
    "_require",
    "_require_json_object_file",
    "_resolve_artifact",
    "_sha256_file",
    "_status",
    "_string_list",
    "_summary_blockers",
    "_summary_stage",
    "_utc_now",
    "_validate_openpi_benchmark_claim",
    "_validate_resource_lease_tree",
    "_validate_runtime_log_tree",
    "_validate_stage0_boundary_text",
    "_validate_stage0_resource_lease",
    "_validate_stage0_runtime_log_boundaries",
    "_write_json",
    "build_early_gate_payloads",
    "build_final_validator_report",
    "main",
    "validate_final_validator_report",
    "validate_gr00t_root_cause_matrix",
    "validate_gr00t_update_probe_spec",
    "validate_next_stage_summary_draft",
    "validate_openpi_benchmark_gate_contract",
    "validate_stage0_dependency_ready",
    "validate_stage0_ready_file",
    "validate_stage0_validator_report",
    "write_early_gate",
    "write_final_validator_report",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--runtime-log-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--early", choices=("all", "gr00t", "openpi"))
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--check-ready", type=Path)
    parser.add_argument("--lane", choices=("gr00t", "openpi"), help="Lane for --check-ready")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.early:
            lanes = ("gr00t", "openpi") if args.early == "all" else (args.early,)
            written: list[str] = []
            for lane in lanes:
                report, ready = write_early_gate(args.run_root, lane)
                written.append(f"{lane}:{report['status']}:{'ready' if ready['ready'] else 'blocked'}")
            print("PASS: wrote Stage0 early gates " + ", ".join(written))
            return 0
        if args.final:
            report = write_final_validator_report(
                args.run_root,
                runtime_log_root=args.runtime_log_root,
            )
            print(f"PASS: wrote final validator report status={report['status']}")
            return 0 if report["status"] == "PASS" else 1
        if args.check_ready is not None:
            _require(args.lane is not None, "--lane is required with --check-ready")
            validate_stage0_dependency_ready(
                args.check_ready,
                run_root=args.run_root,
                expected_lane=args.lane,
            )
            print("PASS: Stage0 dependency ready artifact is authoritative and ready=true")
            return 0
    except (Stage0ValidationError, ValidationError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
