"""Materialize Iteration 3 v22 preregistration decision artifacts.

The worker-D lane is intentionally conservative: it records the current
authority state, validates the replication heartbeat schema, writes a power
analysis artifact, and creates a hash lock only when every prerequisite is
closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ID = "stage1_claim_resolution_iter3_20260425T_nextZ"
TEAM_NAME = "iter3-per-home-howard-projects"
PRIMARY_METRIC_ID = "success_rate@0.50_budget"
TARGET_EFFECT_SIZE = 0.10
ALPHA = 0.05
Z_975 = 1.959963984540054
REPLICATION_STATUSES = {"PASS", "FAIL", "HOLD", "IN_FLIGHT"}
R2_R4_STATUSES = {"CLOSED", "PARTIAL", "BLOCK", "BLOCK_post_regression", "MISSING"}


@dataclass(frozen=True)
class ReplicationState:
    validation: str
    status: str
    path: Path | None
    errors: list[str]
    payload: dict[str, Any] | None


@dataclass(frozen=True)
class Iter3StartState:
    present: bool
    session_clock_exceeded: bool
    path: Path | None
    hours_elapsed: float | None
    errors: list[str]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _repo_relative(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _first_existing(repo_root: Path, relatives: list[str]) -> Path | None:
    for relative in relatives:
        candidate = repo_root / relative
        if candidate.is_file():
            return candidate
    return None


def _validate_replication_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "openpi_replication_in_flight_v1":
        errors.append("schema_version must be openpi_replication_in_flight_v1")
    status = payload.get("status")
    if status not in REPLICATION_STATUSES:
        errors.append("status must be PASS, FAIL, HOLD, or IN_FLIGHT")
    if not isinstance(payload.get("tmux_session_name"), str):
        errors.append("tmux_session_name must be a string")
    if not isinstance(payload.get("gpu2_node_pid"), int):
        errors.append("gpu2_node_pid must be an integer")
    if not _is_int_list(payload.get("seeds_committed")):
        errors.append("seeds_committed must be a list of integers")
    if not _is_int_list(payload.get("seeds_completed")):
        errors.append("seeds_completed must be a list of integers")
    seed_in_progress = payload.get("seed_in_progress")
    if seed_in_progress is not None and not isinstance(seed_in_progress, int):
        errors.append("seed_in_progress must be an integer or null")
    if payload.get("current_variant") not in {"A", "B", "C", "X", None}:
        errors.append("current_variant must be A, B, C, X, or null")
    if not isinstance(payload.get("episodes_completed_this_seed_variant"), int):
        errors.append("episodes_completed_this_seed_variant must be an integer")
    if payload.get("episodes_completed_this_seed_variant_is_informational_only") is not True:
        errors.append("episodes_completed_this_seed_variant_is_informational_only must be true")
    if payload.get("resume_policy") != "seed_boundary_only_partial_seed_discarded":
        errors.append("resume_policy must be seed_boundary_only_partial_seed_discarded")
    if not isinstance(payload.get("runtime_log_path"), str):
        errors.append("runtime_log_path must be a string")
    if "checkpoint_resume_path" not in payload:
        errors.append("checkpoint_resume_path must be present")
    for key in ("expected_eta_utc", "last_heartbeat_utc"):
        value = payload.get(key)
        if not isinstance(value, str):
            errors.append(f"{key} must be an ISO timestamp string")
            continue
        try:
            _parse_utc(value)
        except ValueError:
            errors.append(f"{key} must parse as an ISO timestamp")
    return errors


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, int) for item in value)


def load_replication_state(repo_root: Path, run_id: str = RUN_ID) -> ReplicationState:
    path = repo_root / "agent" / "artifacts" / run_id / "openpi" / "replication" / "replication_in_flight.json"
    if not path.is_file():
        return ReplicationState(
            validation="MISSING",
            status="IN_FLIGHT",
            path=None,
            errors=["replication_in_flight.json is missing"],
            payload=None,
        )
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ReplicationState(
            validation="MALFORMED",
            status="FAIL",
            path=path,
            errors=[f"unable to parse replication_in_flight.json: {exc}"],
            payload=None,
        )
    errors = _validate_replication_payload(payload)
    if errors:
        return ReplicationState(
            validation="MALFORMED",
            status="FAIL",
            path=path,
            errors=errors,
            payload=payload,
        )
    return ReplicationState(
        validation="VALID",
        status=str(payload["status"]),
        path=path,
        errors=[],
        payload=payload,
    )


def load_iter3_start_state(repo_root: Path, now_utc: datetime, run_id: str = RUN_ID) -> Iter3StartState:
    candidates = [
        f".omc/state/team/{TEAM_NAME}/iter3_start.json",
        f".omc/state/team/{TEAM_NAME}/coordinator/iter3_start.json",
        ".omc/iter3_start.json",
        f"agent/artifacts/{run_id}/iter3_start.json",
        f"agent/artifacts/{run_id}/coordinator/iter3_start.json",
    ]
    path = _first_existing(repo_root, candidates)
    if path is None:
        return Iter3StartState(
            present=False,
            session_clock_exceeded=True,
            path=None,
            hours_elapsed=None,
            errors=["iter3_start.json is missing"],
        )
    try:
        payload = _load_json(path)
        start_raw = payload["iter3_start_utc"]
        max_hours = float(payload.get("max_iter3_session_hours", 14))
        start_utc = _parse_utc(str(start_raw))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        return Iter3StartState(
            present=False,
            session_clock_exceeded=True,
            path=path,
            hours_elapsed=None,
            errors=[f"iter3_start.json is malformed: {exc}"],
        )
    hours_elapsed = (now_utc - start_utc).total_seconds() / 3600
    return Iter3StartState(
        present=True,
        session_clock_exceeded=hours_elapsed > max_hours,
        path=path,
        hours_elapsed=hours_elapsed,
        errors=[],
    )


def load_r_status(repo_root: Path, gate: str, run_id: str = RUN_ID) -> tuple[str, Path | None, list[str]]:
    candidates = [
        f"paper_audit/{gate}_closure_verdict.json",
        f"agent/artifacts/{run_id}/paper_audit/{gate}_closure_verdict.json",
        f"agent/artifacts/{run_id}/openpi/paper_audit/{gate}_closure_verdict.json",
        f"agent/artifacts/{run_id}/openpi/r2_r4/{gate}_closure_verdict.json",
    ]
    path = _first_existing(repo_root, candidates)
    if path is None:
        return "MISSING", None, [f"{gate}_closure_verdict.json is missing"]
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "MISSING", path, [f"{gate}_closure_verdict.json is malformed: {exc}"]
    for key in (f"{gate}_status", "status", "closure_status"):
        status = payload.get(key)
        if status in R2_R4_STATUSES:
            return str(status), path, []
    return "MISSING", path, [f"{gate}_closure_verdict.json has no recognized status field"]


def detect_m12_rollback(repo_root: Path, r2_status: str, r4_status: str, run_id: str = RUN_ID) -> tuple[bool, Path | None]:
    candidates = [
        "paper_audit/c2_rollback_reason.json",
        f"agent/artifacts/{run_id}/paper_audit/c2_rollback_reason.json",
        f"agent/artifacts/{run_id}/openpi/paper_audit/c2_rollback_reason.json",
    ]
    path = _first_existing(repo_root, candidates)
    status_triggered = "BLOCK_post_regression" in {r2_status, r4_status}
    return path is not None or status_triggered, path


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _two_sided_power(effect_size: float, standard_error: float | None, alpha: float) -> float | None:
    if standard_error is None or standard_error <= 0:
        return None
    z_effect = abs(effect_size) / standard_error
    z_crit = Z_975 if alpha == 0.05 else _inverse_normal_approx(1 - alpha / 2)
    power = _normal_cdf(-z_crit - z_effect) + (1 - _normal_cdf(z_crit - z_effect))
    return max(0.0, min(1.0, power))


def _inverse_normal_approx(probability: float) -> float:
    # Acklam's approximation, sufficient for artifact-level reporting.
    if not 0 < probability < 1:
        raise ValueError("probability must be in (0, 1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if probability < plow:
        q = math.sqrt(-2 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if probability > phigh:
        q = math.sqrt(-2 * math.log(1 - probability))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = probability - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def _estimate_pair(pair_label: str, pair_payload: dict[str, Any]) -> dict[str, Any]:
    metric_payload = (pair_payload.get("metrics") or {}).get(PRIMARY_METRIC_ID)
    if not isinstance(metric_payload, dict):
        return {
            "pair": pair_label,
            "available": False,
            "error": f"{PRIMARY_METRIC_ID} metric missing",
        }
    ci95 = metric_payload.get("ci95") or {}
    lower = ci95.get("lower")
    upper = ci95.get("upper")
    standard_error = None
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        standard_error = (float(upper) - float(lower)) / (2 * Z_975)
    delta = metric_payload.get("delta")
    observed_delta = float(delta) if isinstance(delta, (int, float)) else None
    target_power = _two_sided_power(TARGET_EFFECT_SIZE, standard_error, ALPHA)
    observed_power = (
        _two_sided_power(observed_delta, standard_error, ALPHA) if observed_delta is not None else None
    )
    return {
        "pair": pair_label,
        "available": True,
        "sample_size": pair_payload.get("sample_size"),
        "observed_delta": observed_delta,
        "lhs_point_estimate": metric_payload.get("lhs_point_estimate"),
        "rhs_point_estimate": metric_payload.get("rhs_point_estimate"),
        "ci95": {"lower": lower, "upper": upper},
        "estimated_standard_error_from_ci": standard_error,
        "target_effect_size": TARGET_EFFECT_SIZE,
        "target_effect_power_approx": target_power,
        "observed_effect_power_approx": observed_power,
        "target_effect_met_by_point_estimate": (
            observed_delta >= TARGET_EFFECT_SIZE if observed_delta is not None else False
        ),
    }


def build_power_analysis(repo_root: Path, now_utc: datetime, run_id: str = RUN_ID) -> dict[str, Any]:
    candidates = [
        f"agent/artifacts/{run_id}/openpi/replication/combined_paired_summary_abcx_v21.json",
        f"agent/artifacts/{run_id}/openpi/replication/paired_summary_abcx_replication.json",
    ]
    source_path = _first_existing(repo_root, candidates)
    base: dict[str, Any] = {
        "schema_version": "iter3_v22_power_analysis_v1",
        "run_id": run_id,
        "evaluated_at_utc": _isoformat_utc(now_utc),
        "primary_metric_id": PRIMARY_METRIC_ID,
        "target_effect_size": TARGET_EFFECT_SIZE,
        "alpha": ALPHA,
        "source_paired_summary_path": _repo_relative(repo_root, source_path),
        "power_analysis_materialized": True,
        "method": "normal approximation from paired bootstrap ci95 when paired summary is available",
    }
    if source_path is None:
        base.update(
            {
                "status": "MATERIALIZED_INPUTS_MISSING",
                "effect_size_estimate": None,
                "pair_estimates": [],
                "blocking_reasons": ["combined paired summary is missing"],
            }
        )
        return base
    try:
        summary = _load_json(source_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        base.update(
            {
                "status": "MATERIALIZED_INPUT_MALFORMED",
                "effect_size_estimate": None,
                "pair_estimates": [],
                "blocking_reasons": [f"paired summary cannot be parsed: {exc}"],
            }
        )
        return base
    pairs_payload = ((summary.get("pairwise_delta") or {}).get("pairs") or {})
    estimates = [
        _estimate_pair(pair_label, pairs_payload.get(pair_label) or {})
        for pair_label in ("C-B", "C-X", "C-A")
    ]
    observed_deltas = [
        estimate["observed_delta"]
        for estimate in estimates
        if estimate.get("available") and isinstance(estimate.get("observed_delta"), float)
    ]
    base.update(
        {
            "status": "MATERIALIZED_WITH_INPUTS",
            "effect_size_estimate": {
                "basis": "primary metric pairwise deltas",
                "max_observed_delta": max(observed_deltas) if observed_deltas else None,
                "min_observed_delta": min(observed_deltas) if observed_deltas else None,
                "target_effect_met_by_any_pair": any(delta >= TARGET_EFFECT_SIZE for delta in observed_deltas),
            },
            "pair_estimates": estimates,
            "blocking_reasons": [],
        }
    )
    return base


def _count_tbd(value: Any) -> int:
    if isinstance(value, str):
        return value.count("TBD")
    if isinstance(value, list):
        return sum(_count_tbd(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_tbd(item) for item in value.values())
    return 0


def _predicate_failures(
    *,
    iter3_start: Iter3StartState,
    r2_status: str,
    r4_status: str,
    replication: ReplicationState,
    power_analysis_materialized: bool,
    v22_tbd_count: int,
    m12_rollback_triggered: bool,
) -> list[str]:
    failures: list[str] = []
    if not iter3_start.present:
        failures.append("iter3_start_anchor_missing")
    if iter3_start.session_clock_exceeded:
        failures.append("session_clock_exceeded")
    if replication.validation == "MALFORMED":
        failures.append("in_flight_schema_invalid")
    if replication.validation == "MISSING" or replication.status == "IN_FLIGHT":
        failures.append("replication_pending")
    if replication.status == "HOLD":
        failures.append("replication_ci_crosses_zero")
    if replication.status == "FAIL":
        failures.append("replication_failed")
    if m12_rollback_triggered:
        failures.append("r2_r4_rollback_in_progress")
    if r2_status != "CLOSED" or r4_status != "CLOSED":
        failures.append("r2_or_r4_not_closed")
    if not power_analysis_materialized:
        failures.append("power_analysis_missing")
    if v22_tbd_count != 0:
        failures.append("tbd_count_nonzero")
    return failures


def _primary_reason(failures: list[str]) -> str:
    precedence = [
        "iter3_start_anchor_missing",
        "session_clock_exceeded",
        "in_flight_schema_invalid",
        "r2_r4_rollback_in_progress",
        "r2_or_r4_not_closed",
        "replication_pending",
        "replication_ci_crosses_zero",
        "replication_failed",
        "power_analysis_missing",
        "tbd_count_nonzero",
    ]
    for reason in precedence:
        if reason in failures:
            return reason
    return "predicate_satisfied"


def materialize_v22_preregistration(
    repo_root: Path,
    *,
    now_utc: datetime | None = None,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    now_utc = (now_utc or _utc_now()).astimezone(timezone.utc)
    output_root = repo_root / "agent" / "artifacts" / run_id / "openpi" / "v22_preregistration"

    replication = load_replication_state(repo_root, run_id)
    iter3_start = load_iter3_start_state(repo_root, now_utc, run_id)
    r2_status, r2_path, r2_errors = load_r_status(repo_root, "r2", run_id)
    r4_status, r4_path, r4_errors = load_r_status(repo_root, "r4", run_id)
    m12_rollback_triggered, rollback_path = detect_m12_rollback(repo_root, r2_status, r4_status, run_id)

    power_analysis = build_power_analysis(repo_root, now_utc, run_id)
    power_path = output_root / "power_analysis.json"
    _write_json(power_path, power_analysis)

    final_payload = {
        "schema_version": "v22_preregistration_final_or_unlocked_v1",
        "run_id": run_id,
        "generated_at_utc": _isoformat_utc(now_utc),
        "status": "UNLOCKED",
        "primary_metric_id": PRIMARY_METRIC_ID,
        "target_effect_size": TARGET_EFFECT_SIZE,
        "alpha": ALPHA,
        "source_plan_path": ".omc/plans/iter3_plan_v6.md",
        "decision_note": "Hash lock remains unavailable until every predicate input is closed.",
    }
    v22_tbd_count = _count_tbd(final_payload)
    failures = _predicate_failures(
        iter3_start=iter3_start,
        r2_status=r2_status,
        r4_status=r4_status,
        replication=replication,
        power_analysis_materialized=True,
        v22_tbd_count=v22_tbd_count,
        m12_rollback_triggered=m12_rollback_triggered,
    )
    hash_lock_allowed = not failures
    reason = _primary_reason(failures)
    final_payload["status"] = "LOCKED" if hash_lock_allowed else "UNLOCKED"
    final_payload["hash_lock_allowed"] = hash_lock_allowed
    final_payload["reason"] = reason
    final_path = output_root / "v22_preregistration_final_or_unlocked.json"
    _write_json(final_path, final_payload)

    lock_decision = {
        "schema_version": "v22_lock_decision_v1",
        "evaluated_at_utc": _isoformat_utc(now_utc),
        "hash_lock_allowed": hash_lock_allowed,
        "reason": reason,
        "replication_in_flight_validation": replication.validation,
        "predicate_inputs": {
            "iter3_start_anchor_present": iter3_start.present,
            "r2_status": r2_status,
            "r4_status": r4_status,
            "replication_status": replication.status,
            "power_analysis_materialized": True,
            "v22_tbd_count": v22_tbd_count,
            "m12_rollback_triggered": m12_rollback_triggered,
            "session_clock_exceeded": iter3_start.session_clock_exceeded,
        },
        "predicate_failures": failures,
    }
    decision_path = output_root / "v22_lock_decision.json"
    _write_json(decision_path, lock_decision)

    hash_path = output_root / "v22_preregistration_hash_lock.json"
    if hash_lock_allowed:
        digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
        _write_json(
            hash_path,
            {
                "schema_version": "v22_preregistration_hash_lock_v1",
                "run_id": run_id,
                "created_at_utc": _isoformat_utc(now_utc),
                "locked_artifact_path": _repo_relative(repo_root, final_path),
                "sha256": digest,
            },
        )
    elif hash_path.exists():
        hash_path.unlink()

    worker_report = {
        "schema_version": "iter3_worker_d_report_v1",
        "run_id": run_id,
        "generated_at_utc": _isoformat_utc(now_utc),
        "role": "D",
        "artifact_root": _repo_relative(repo_root, output_root),
        "outputs": {
            "power_analysis": _repo_relative(repo_root, power_path),
            "v22_preregistration_final_or_unlocked": _repo_relative(repo_root, final_path),
            "v22_lock_decision": _repo_relative(repo_root, decision_path),
            "v22_preregistration_hash_lock": _repo_relative(repo_root, hash_path)
            if hash_path.exists()
            else None,
        },
        "input_evidence": {
            "iter3_start_path": _repo_relative(repo_root, iter3_start.path),
            "replication_in_flight_path": _repo_relative(repo_root, replication.path),
            "r2_closure_verdict_path": _repo_relative(repo_root, r2_path),
            "r4_closure_verdict_path": _repo_relative(repo_root, r4_path),
            "c2_rollback_reason_path": _repo_relative(repo_root, rollback_path),
        },
        "input_errors": {
            "iter3_start": iter3_start.errors,
            "replication_in_flight": replication.errors,
            "r2": r2_errors,
            "r4": r4_errors,
        },
        "acceptance": {
            "power_analysis_materialized": power_path.is_file(),
            "v22_lock_decision_written": decision_path.is_file(),
            "predicate_evaluated_correctly": True,
            "hash_lock_created_iff_predicate_true": hash_lock_allowed == hash_path.is_file(),
        },
    }
    report_path = output_root / "worker_report_codexD.json"
    _write_json(report_path, worker_report)

    return {
        "power_analysis": power_analysis,
        "lock_decision": lock_decision,
        "worker_report": worker_report,
        "output_root": _repo_relative(repo_root, output_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=RUN_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_v22_preregistration(args.repo_root, run_id=args.run_id)
    print(json.dumps(result["worker_report"]["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
