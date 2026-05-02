#!/usr/bin/env python3
"""Validate GR00T/OpenPI dual-track verifier artifacts.

This is the executable counterpart to the dual-track blocker-push test spec.
It intentionally validates only artifact contracts and boundary evidence; it
does not run training, eval, sudo, or GPU workloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


FORMAL_SCHEMA = "dual_track_formal_status_v1"
EXPLORATORY_SCHEMA = "dual_track_exploratory_signal_v1"
SUMMARY_SCHEMA = "dual_track_summary_v1"
FORMAL_STATUSES = {"PASS", "BLOCK", "SKIPPED"}
EXPLORATORY_STATUSES = {"SIGNAL", "NO_SIGNAL", "SKIPPED", "FAILED"}
OPENPI_PASS_RUNTIME_LEVELS = {
    "materialization_ready",
    "p0_loader_runtime_pass",
    "p1_one_step_pass",
    "p2_overfit_or_tiny_update_pass",
}
OPENPI_RUNTIME_LEVEL_ALLOWED_CLAIMS = {
    "materialization_ready": {"materialization_ready"},
    "p0_loader_runtime_pass": {"materialization_ready", "p0_loader_runtime_pass"},
    "p1_one_step_pass": {
        "materialization_ready",
        "p0_loader_runtime_pass",
        "p1_one_step_pass",
    },
    "p2_overfit_or_tiny_update_pass": {
        "materialization_ready",
        "p0_loader_runtime_pass",
        "p1_one_step_pass",
        "p2_overfit_or_tiny_update_pass",
    },
}
RESOURCE_LEASE_SCHEMA = "resource_lease_v1"
CANDIDATE_GRADUATION_STAGES = (
    "C0_STATIC",
    "C1_TELEMETRY",
    "C2_DRY_RUN",
    "C3_FORMAL_3SEED",
    "C4_P5_ELIGIBLE",
)
CANDIDATE_STAGE_INDEX = {
    stage: index for index, stage in enumerate(CANDIDATE_GRADUATION_STAGES)
}
FORBIDDEN_INFERENCE_SNIPPETS = (
    "exploratory signal != formal pass",
    "OpenPI exploratory dataset != formal materialized",
    "GR00T metric ablation/additional seed signal != P5 eligible",
)


class ValidationError(AssertionError):
    """Raised when a dual-track artifact violates the test-spec contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"{path}: missing JSON artifact") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"{path}: top-level JSON must be an object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [str(value)]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _truthy_field(payload: dict[str, Any], *names: str) -> bool:
    return any(payload.get(name) is True for name in names)


def validate_openpi_runtime_level(payload: dict[str, Any], *, prefix: str) -> None:
    """Validate p0/p1/p2 runtime-level semantics for OpenPI formal artifacts."""

    runtime_level = payload.get("runtime_level")
    _require(
        isinstance(runtime_level, str) and bool(runtime_level),
        prefix + "OpenPI formal status must include runtime_level",
    )
    _require(
        runtime_level in OPENPI_PASS_RUNTIME_LEVELS
        or runtime_level.startswith("blocked_"),
        prefix
        + "runtime_level must be a known pass level or machine-checkable blocked_<reason>",
    )

    blocking_reasons = _string_list(payload.get("blocking_reasons"))
    if runtime_level.startswith("blocked_"):
        _require(
            payload.get("status") != "PASS",
            prefix + "blocked runtime_level cannot have PASS status",
        )
        _require(
            payload.get("formal_claim_allowed") is False,
            prefix + "blocked runtime_level cannot allow formal claim",
        )
        if "dataset_not_materialized" in blocking_reasons:
            _require(
                runtime_level == "blocked_materialization_reverify_failed",
                prefix
                + "dataset_not_materialized is only valid for failed materialization reverify",
            )
        return

    runtime_claims = _string_list(payload.get("runtime_claims"))
    if runtime_claims:
        allowed_claims = OPENPI_RUNTIME_LEVEL_ALLOWED_CLAIMS[runtime_level]
        unexpected = sorted(set(runtime_claims) - allowed_claims)
        _require(
            not unexpected,
            prefix
            + f"{runtime_level} must not claim higher runtime levels: {unexpected}",
        )


def validate_formal_status(
    path: Path, *, expected_lane: str | None = None
) -> dict[str, Any]:
    payload = _load_json(path)
    prefix = f"{path}: "

    _require(
        payload.get("schema_version") == FORMAL_SCHEMA,
        prefix + "wrong schema_version",
    )
    _require(payload.get("track") == "formal", prefix + "track must be formal")
    if expected_lane is not None:
        _require(
            payload.get("lane") == expected_lane,
            prefix + f"lane must be {expected_lane}",
        )

    status = payload.get("status")
    _require(
        status in FORMAL_STATUSES,
        prefix + f"status must be one of {sorted(FORMAL_STATUSES)}",
    )
    _require(
        isinstance(status, str) and "(" not in status and ")" not in status,
        prefix + "status must not be compound",
    )
    _require(
        isinstance(payload.get("blocking_reasons"), list),
        prefix + "blocking_reasons must be a list",
    )
    _require(
        isinstance(payload.get("authority_inputs"), list),
        prefix + "authority_inputs must be a list",
    )
    _require(
        isinstance(payload.get("validator_outputs"), list),
        prefix + "validator_outputs must be a list",
    )
    _require(
        isinstance(payload.get("entered_next_gate"), bool),
        prefix + "entered_next_gate must be bool",
    )
    _require(
        isinstance(payload.get("next_gate_allowed"), bool),
        prefix + "next_gate_allowed must be bool",
    )

    if payload.get("formal_claim_allowed") is True:
        _require(status == "PASS", prefix + "formal_claim_allowed requires PASS")
        _require(
            payload.get("next_gate_allowed") is True,
            prefix + "formal_claim_allowed requires next_gate_allowed=true",
        )
    else:
        _require(
            payload.get("formal_claim_allowed") is False,
            prefix + "formal_claim_allowed must be explicit false unless PASS",
        )

    if status != "PASS":
        _require(
            payload.get("next_gate_allowed") is False,
            prefix + "non-PASS formal status must not allow next gate",
        )

    if expected_lane == "openpi":
        validate_openpi_runtime_level(payload, prefix=prefix)

    return payload


def validate_exploratory_signal(
    path: Path, *, expected_lane: str | None = None
) -> dict[str, Any]:
    payload = _load_json(path)
    prefix = f"{path}: "

    _require(
        payload.get("schema_version") == EXPLORATORY_SCHEMA,
        prefix + "wrong schema_version",
    )
    _require(payload.get("track") == "exploratory", prefix + "track must be exploratory")
    if expected_lane is not None:
        _require(
            payload.get("lane") == expected_lane,
            prefix + f"lane must be {expected_lane}",
        )

    _require(
        payload.get("status") in EXPLORATORY_STATUSES,
        prefix + f"status must be one of {sorted(EXPLORATORY_STATUSES)}",
    )
    _require(
        payload.get("exploratory_only") is True,
        prefix + "exploratory_only must be true",
    )
    _require(
        payload.get("formal_claim_allowed") is False,
        prefix + "exploratory formal_claim_allowed must be false",
    )
    _require(
        payload.get("must_not_unlock_formal_gate") is True,
        prefix + "must_not_unlock_formal_gate must be true",
    )
    _require(
        payload.get("risk_label") == "exploratory_not_formal",
        prefix + "risk_label must be exploratory_not_formal",
    )
    _require(isinstance(payload.get("inputs"), list), prefix + "inputs must be a list")
    _require(isinstance(payload.get("outputs"), list), prefix + "outputs must be a list")
    _require(
        isinstance(payload.get("observed_signal"), dict),
        prefix + "observed_signal must be an object",
    )
    return payload


def _summary_track(summary: dict[str, Any], lane: str, track: str) -> dict[str, Any]:
    section = summary.get(lane)
    _require(isinstance(section, dict), f"summary: {lane} must be an object")
    track_payload = section.get(track)
    _require(
        isinstance(track_payload, dict),
        f"summary: {lane}.{track} must be an object",
    )
    return track_payload


def validate_summary(
    path: Path,
    *,
    formal_payloads: dict[str, dict[str, Any]] | None = None,
    exploratory_payloads: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = _load_json(path)
    _require(
        payload.get("schema_version") == SUMMARY_SCHEMA,
        f"{path}: wrong schema_version",
    )
    _require(
        isinstance(payload.get("next_actions"), list),
        f"{path}: next_actions must be a list",
    )
    forbidden = payload.get("forbidden_inferences")
    _require(isinstance(forbidden, list), f"{path}: forbidden_inferences must be a list")
    forbidden_text = "\n".join(str(item) for item in forbidden)
    for snippet in FORBIDDEN_INFERENCE_SNIPPETS:
        _require(
            snippet in forbidden_text,
            f"{path}: missing forbidden inference: {snippet}",
        )

    for lane in ("gr00t", "openpi"):
        formal_section = _summary_track(payload, lane, "formal")
        exploratory_section = _summary_track(payload, lane, "exploratory")
        _require(
            formal_section.get("status") in FORMAL_STATUSES,
            f"{path}: {lane}.formal.status invalid",
        )
        _require(
            exploratory_section.get("status") in EXPLORATORY_STATUSES,
            f"{path}: {lane}.exploratory.status invalid",
        )
        _require(
            isinstance(formal_section.get("artifact"), str)
            and formal_section["artifact"],
            f"{path}: {lane}.formal.artifact required",
        )
        _require(
            isinstance(exploratory_section.get("artifact"), str)
            and exploratory_section["artifact"],
            f"{path}: {lane}.exploratory.artifact required",
        )

        formal_claim = formal_section.get("formal_claim_allowed")
        _require(
            isinstance(formal_claim, bool),
            f"{path}: {lane}.formal.formal_claim_allowed must be bool",
        )
        summary_blockers = formal_section.get("blocking_reasons")
        _require(
            isinstance(summary_blockers, list),
            f"{path}: {lane}.formal.blocking_reasons must be a list",
        )
        if formal_payloads is not None and lane in formal_payloads:
            actual = formal_payloads[lane]
            _require(
                formal_section.get("status") == actual.get("status"),
                f"{path}: {lane}.formal.status differs from artifact",
            )
            _require(
                formal_claim == actual.get("formal_claim_allowed"),
                f"{path}: {lane}.formal.formal_claim_allowed differs from artifact",
            )
            _require(
                summary_blockers == actual.get("blocking_reasons"),
                f"{path}: {lane}.formal.blocking_reasons differs from artifact",
            )
        if exploratory_payloads is not None and lane in exploratory_payloads:
            actual_exp = exploratory_payloads[lane]
            _require(
                exploratory_section.get("status") == actual_exp.get("status"),
                f"{path}: {lane}.exploratory.status differs from artifact",
            )
            if actual_exp.get("status") == "SIGNAL" and formal_section.get("status") != "PASS":
                _require(
                    formal_claim is False,
                    f"{path}: {lane}.exploratory SIGNAL must not unlock formal claim",
                )

    return payload


def validate_runtime_log_boundaries(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    _require(
        "sudo " not in text and "\nsudo" not in text,
        f"{path}: agent logs must not run sudo directly",
    )
    _require(
        not re.search(r"CUDA_VISIBLE_DEVICES\s*=\s*(?:0|3|0,|3,|.*,0|.*,3)", text),
        f"{path}: log references forbidden GPU0/GPU3 CUDA visibility",
    )


def validate_gpu_boundary_log(path: Path) -> None:
    payload = _load_json(path)
    gr00t = payload.get("gr00t")
    openpi = payload.get("openpi")
    _require(isinstance(gr00t, dict), f"{path}: gr00t GPU snapshot required")
    _require(isinstance(openpi, dict), f"{path}: openpi GPU snapshot required")
    _require(
        str(gr00t.get("gpu")) in {"1", "GPU1"},
        f"{path}: GR00T must be bound to GPU1",
    )
    _require(
        str(openpi.get("gpu")) in {"2", "GPU2"},
        f"{path}: OpenPI must be bound to GPU2",
    )
    used = {str(item) for item in payload.get("used_gpus", [])}
    _require(
        "0" not in used and "3" not in used,
        f"{path}: used_gpus must not include GPU0/GPU3",
    )


def _validate_no_boundary_pollution_text(text: str, *, label: str) -> None:
    _require(
        "sudo " not in text and "\nsudo" not in text,
        f"{label}: agent evidence must not run sudo directly",
    )
    _require(
        not re.search(r"CUDA_VISIBLE_DEVICES\s*=\s*(?:0|3|0,|3,|.*,0|.*,3)", text),
        f"{label}: evidence references forbidden GPU0/GPU3 CUDA visibility",
    )


def validate_resource_lease(path: Path) -> dict[str, Any]:
    """Validate a machine-checkable GPU resource lease manifest."""

    payload = _load_json(path)
    prefix = f"{path}: "
    _require(
        payload.get("schema_version") == RESOURCE_LEASE_SCHEMA,
        prefix + "wrong schema_version",
    )
    lane = payload.get("lane")
    _require(lane in {"gr00t", "openpi"}, prefix + "lane must be gr00t or openpi")
    expected_gpu = "1" if lane == "gr00t" else "2"
    _require(str(payload.get("gpu")) == expected_gpu, prefix + f"{lane} must use GPU{expected_gpu}")
    _require(isinstance(payload.get("worker"), str) and payload["worker"], prefix + "worker required")

    raw_command = payload.get("command")
    command_shell = payload.get("command_shell")
    if isinstance(raw_command, str):
        command_text = raw_command
    elif isinstance(command_shell, str):
        command_text = command_shell
    elif isinstance(raw_command, list):
        command_text = " ".join(str(part) for part in raw_command)
    else:
        command_text = ""
    _require(command_text.strip(), prefix + "command or command_shell required")
    _validate_no_boundary_pollution_text(command_text, label=prefix + "command")

    _require(
        isinstance(payload.get("started_at_utc") or payload.get("start_time"), str)
        and bool(payload.get("started_at_utc") or payload.get("start_time")),
        prefix + "started_at_utc/start_time required",
    )
    _require(
        isinstance(payload.get("ended_at_utc") or payload.get("end_time"), str)
        and bool(payload.get("ended_at_utc") or payload.get("end_time")),
        prefix + "ended_at_utc/end_time required",
    )
    returncode = payload.get("returncode")
    timed_out = payload.get("timed_out") is True or bool(payload.get("timeout_reason"))
    _require(
        isinstance(returncode, int) or timed_out,
        prefix + "returncode or timeout evidence required",
    )
    timeout_s = payload.get("timeout_s", payload.get("timeout_seconds"))
    _require(
        isinstance(timeout_s, (int, float)) and timeout_s >= 0,
        prefix + "timeout_s/timeout_seconds must be a non-negative number",
    )
    runtime_log = payload.get("runtime_log")
    _require(
        isinstance(runtime_log, str) and runtime_log.strip(),
        prefix + "runtime_log path required",
    )
    _validate_no_boundary_pollution_text(runtime_log, label=prefix + "runtime_log")
    _require(isinstance(payload.get("artifacts"), list), prefix + "artifacts must be a list")
    _require(
        payload.get("forbidden_gpus_visible") is False,
        prefix + "forbidden_gpus_visible must be false",
    )
    _require(payload.get("sudo_used") is False, prefix + "sudo_used must be false")
    return payload


def _stage_index(stage: object, *, prefix: str) -> int:
    _require(
        isinstance(stage, str) and stage in CANDIDATE_STAGE_INDEX,
        prefix + f"graduation_stage must be one of {list(CANDIDATE_GRADUATION_STAGES)}",
    )
    return CANDIDATE_STAGE_INDEX[str(stage)]


def validate_gr00t_candidate_manifest_payload(
    payload: dict[str, Any], *, prefix: str
) -> dict[str, Any]:
    """Validate one GR00T candidate manifest against the graduation ladder."""

    _require(
        isinstance(payload.get("candidate_id"), str) and payload["candidate_id"],
        prefix + "candidate_id required",
    )
    track = payload.get("track")
    _require(
        isinstance(track, str) and ("formal" in track or "exploratory" in track),
        prefix + "track must be formal/exploratory-labelled",
    )
    stage = _stage_index(payload.get("graduation_stage"), prefix=prefix)
    if _truthy_field(
        payload,
        "entered_gpu_formal",
        "entered_formal_run",
        "formal_run_started",
    ) or payload.get("formal_status"):
        _require(
            stage >= CANDIDATE_STAGE_INDEX["C2_DRY_RUN"],
            prefix + "GPU1 formal candidates must reach at least C2_DRY_RUN",
        )
    if _truthy_field(payload, "entered_p5", "selected_for_p5", "p5_eligible"):
        _require(
            stage >= CANDIDATE_STAGE_INDEX["C4_P5_ELIGIBLE"],
            prefix + "P5 candidates must reach C4_P5_ELIGIBLE",
        )
    return payload


def _candidate_is_scalar_only(candidate: dict[str, Any]) -> bool:
    text = " ".join(
        str(candidate.get(key, ""))
        for key in ("candidate_id", "candidate_type", "type", "hypothesis", "description")
    ).lower()
    return "scalar" in text and "amplitude" in text


def validate_gr00t_candidate_matrix(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    prefix = f"{path}: "
    candidates = payload.get("candidates")
    _require(isinstance(candidates, list), prefix + "candidates must be a list")
    _require(candidates, prefix + "candidates must not be empty")

    non_scalar_formal = 0
    for index, item in enumerate(candidates):
        _require(isinstance(item, dict), prefix + f"candidates[{index}] must be an object")
        candidate = validate_gr00t_candidate_manifest_payload(
            item,
            prefix=prefix + f"candidates[{index}]: ",
        )
        if "formal" in str(candidate.get("track")) and not _candidate_is_scalar_only(candidate):
            non_scalar_formal += 1

    blockers = _string_list(payload.get("non_scalar_candidate_blockers"))
    _require(
        non_scalar_formal >= 2 or bool(blockers),
        prefix
        + "candidate matrix must include at least two non-scalar formal candidates "
        "or record non_scalar_candidate_blockers",
    )
    return payload


def validate_gr00t_candidate_manifest(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return validate_gr00t_candidate_manifest_payload(payload, prefix=f"{path}: ")


def validate_runtime_evidence_minimum(
    formal_payloads: dict[str, dict[str, Any]],
    *,
    leases_by_lane: dict[str, list[dict[str, Any]]],
) -> None:
    """Ensure PASS/BLOCK lane statuses are backed by runtime evidence."""

    for lane, payload in formal_payloads.items():
        if payload.get("status") not in {"PASS", "BLOCK"}:
            continue
        if leases_by_lane.get(lane):
            continue
        evidence = payload.get("runtime_evidence")
        prefix = f"{lane} formal runtime_evidence: "
        _require(isinstance(evidence, dict), prefix + "object or resource lease required")
        _require(
            isinstance(evidence.get("command"), str) and evidence["command"],
            prefix + "command required",
        )
        _require(
            isinstance(evidence.get("started_at_utc"), str) and evidence["started_at_utc"],
            prefix + "started_at_utc required",
        )
        _require(
            isinstance(evidence.get("ended_at_utc"), str) and evidence["ended_at_utc"],
            prefix + "ended_at_utc required",
        )
        returncode = evidence.get("returncode")
        timed_out = evidence.get("timed_out") is True or bool(evidence.get("timeout_reason"))
        _require(isinstance(returncode, int) or timed_out, prefix + "returncode or timeout required")
        _require(
            isinstance(evidence.get("runtime_log"), str) and evidence["runtime_log"],
            prefix + "runtime_log required",
        )
        _require(isinstance(evidence.get("artifacts"), list), prefix + "artifacts list required")
        _require(evidence.get("forbidden_gpus_visible") is False, prefix + "forbidden GPUs visible")
        _require(evidence.get("sudo_used") is False, prefix + "sudo_used must be false")
        _validate_no_boundary_pollution_text(str(evidence.get("command")), label=prefix + "command")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-formal", type=Path, required=True)
    parser.add_argument("--gr00t-exploratory", type=Path, required=True)
    parser.add_argument("--openpi-formal", type=Path, required=True)
    parser.add_argument("--openpi-exploratory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--runtime-log", type=Path, action="append", default=[])
    parser.add_argument("--gpu-boundary-log", type=Path)
    parser.add_argument("--resource-lease", type=Path, action="append", default=[])
    parser.add_argument("--gr00t-candidate-matrix", type=Path)
    parser.add_argument("--gr00t-candidate-manifest", type=Path, action="append", default=[])
    parser.add_argument(
        "--require-runtime-evidence",
        action="store_true",
        help="require PASS/BLOCK formal lane statuses to include runtime evidence or leases",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        formal = {
            "gr00t": validate_formal_status(args.gr00t_formal, expected_lane="gr00t"),
            "openpi": validate_formal_status(args.openpi_formal, expected_lane="openpi"),
        }
        leases_by_lane: dict[str, list[dict[str, Any]]] = {}
        for lease_path in args.resource_lease:
            lease = validate_resource_lease(lease_path)
            lane = str(lease["lane"])
            leases_by_lane.setdefault(lane, []).append(lease)
        if args.gr00t_candidate_matrix is not None:
            validate_gr00t_candidate_matrix(args.gr00t_candidate_matrix)
        for candidate_manifest in args.gr00t_candidate_manifest:
            validate_gr00t_candidate_manifest(candidate_manifest)
        exploratory = {
            "gr00t": validate_exploratory_signal(args.gr00t_exploratory, expected_lane="gr00t"),
            "openpi": validate_exploratory_signal(args.openpi_exploratory, expected_lane="openpi"),
        }
        validate_summary(
            args.summary,
            formal_payloads=formal,
            exploratory_payloads=exploratory,
        )
        for runtime_log in args.runtime_log:
            validate_runtime_log_boundaries(runtime_log)
        if args.gpu_boundary_log is not None:
            validate_gpu_boundary_log(args.gpu_boundary_log)
        if args.require_runtime_evidence:
            validate_runtime_evidence_minimum(formal, leases_by_lane=leases_by_lane)
    except ValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: dual-track artifacts satisfy schema, pollution, summary, "
        "and boundary contracts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
