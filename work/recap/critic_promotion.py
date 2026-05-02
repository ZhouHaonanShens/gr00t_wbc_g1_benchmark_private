from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CRITIC_PROMOTION_SCHEMA_VERSION = "gr00t_critic_promotion_v1"
CRITIC_PROMOTION_ARTIFACT_KIND = "gr00t_critic_promotion"
CRITIC_PROMOTION_EXTENSION_KEY = "critic_promotion"
CRITIC_PROMOTION_GATE_NAME = "vlm_critic_upgrade_gate"

CRITIC_ROLE_REVIEW_ONLY = "review_only"
CRITIC_ROLE_PRIMARY_RELABEL_SOURCE = "primary_relabel_source"

RELABEL_SOURCE_DEFAULT_MAINLINE = "default_mainline"
RELABEL_SOURCE_CRITIC = "critic"

PROMOTION_STATUS_PASS = "PASS"
PROMOTION_STATUS_BLOCK = "BLOCK"

GATES_A_F_ORDER: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

_GREEN_STATUS_VALUES = {
    "ALLOW",
    "ALLOWED",
    "COMPLETE",
    "COMPLETED",
    "DIAGNOSTIC_PASS",
    "GREEN",
    "OK",
    "PASS",
    "PASSED",
    "REINTEGRATE_ALLOWED",
    "TRUE",
}
_BLOCKED_STATUS_VALUES = {
    "BLOCK",
    "BLOCKED",
    "DIAGNOSTIC_BLOCK",
    "FAIL",
    "FAILED",
    "FALSE",
    "INCOMPLETE",
    "PENDING",
    "RED",
    "REINTEGRATE_BLOCKED",
}


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    normalized = str(value).strip()
    return normalized or None


def _coerce_bool_flag(value: object) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"TRUE", "YES", "1", "PASS", "GREEN", "ALLOW", "OK"}:
            return True
        if normalized in {
            "FALSE",
            "NO",
            "0",
            "BLOCK",
            "FAIL",
            "RED",
            "PENDING",
        }:
            return False
    return None


def _candidate_bool(
    payload: Mapping[str, Any],
    *,
    field_names: tuple[str, ...],
) -> bool | None:
    for field_name in field_names:
        if field_name not in payload:
            continue
        coerced = _coerce_bool_flag(payload.get(field_name))
        if coerced is not None:
            return coerced
    return None


def _candidate_status(
    payload: Mapping[str, Any],
    *,
    field_names: tuple[str, ...],
) -> bool | None:
    for field_name in field_names:
        status = _optional_string(payload.get(field_name))
        if status is None:
            continue
        normalized = status.upper()
        if normalized in _GREEN_STATUS_VALUES:
            return True
        if normalized in _BLOCKED_STATUS_VALUES:
            return False
    return None


def _dedupe_failure_reasons(reasons: list[str]) -> list[str]:
    deduped: list[str] = []
    for reason in reasons:
        normalized = str(reason).strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _gate_candidate_keys(letter: str) -> tuple[str, ...]:
    lower = letter.lower()
    return (
        letter,
        lower,
        f"gate_{lower}",
        f"gate_{letter}",
        f"gate{letter}",
        f"gate{lower}",
    )


def _gate_green_from_value(value: object) -> bool | None:
    coerced = _coerce_bool_flag(value)
    if coerced is not None:
        return coerced
    if isinstance(value, Mapping):
        direct_bool = _candidate_bool(
            value,
            field_names=(
                "green",
                "passed",
                "pass",
                "ok",
                "gate_passed",
                "complete",
                "completed",
            ),
        )
        if direct_bool is not None:
            return direct_bool
        status_bool = _candidate_status(
            value,
            field_names=("status", "gate_status", "promotion_status"),
        )
        if status_bool is not None:
            return status_bool
    return None


def summarize_gates_a_f_status(
    gates_a_f_bundle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(gates_a_f_bundle, Mapping):
        return {
            "gates": {gate_name: False for gate_name in GATES_A_F_ORDER},
            "gates_a_f_green": False,
            "missing_gates": list(GATES_A_F_ORDER),
            "failure_reasons": ["gates_a_f_bundle_missing"],
        }

    nested_gates = gates_a_f_bundle.get("gates")
    gate_mapping = (
        nested_gates if isinstance(nested_gates, Mapping) else gates_a_f_bundle
    )

    gate_statuses: dict[str, bool] = {}
    missing_gates: list[str] = []
    failure_reasons: list[str] = []
    for gate_name in GATES_A_F_ORDER:
        raw_value = None
        found = False
        for candidate_key in _gate_candidate_keys(gate_name):
            if candidate_key not in gate_mapping:
                continue
            raw_value = gate_mapping.get(candidate_key)
            found = True
            break
        if not found:
            gate_statuses[gate_name] = False
            missing_gates.append(gate_name)
            failure_reasons.append(f"gate_{gate_name.lower()}_missing_or_invalid")
            continue
        resolved = _gate_green_from_value(raw_value)
        if resolved is None:
            gate_statuses[gate_name] = False
            missing_gates.append(gate_name)
            failure_reasons.append(f"gate_{gate_name.lower()}_missing_or_invalid")
            continue
        gate_statuses[gate_name] = bool(resolved)
        if not resolved:
            failure_reasons.append(f"gate_{gate_name.lower()}_not_green")
    return {
        "gates": gate_statuses,
        "gates_a_f_green": bool(
            not missing_gates
            and all(gate_statuses[gate_name] for gate_name in GATES_A_F_ORDER)
        ),
        "missing_gates": missing_gates,
        "failure_reasons": _dedupe_failure_reasons(failure_reasons),
    }


def summarize_offline_audit(
    offline_audit_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(offline_audit_payload, Mapping):
        return {
            "available": False,
            "offline_gain_green": False,
            "failure_reasons": ["offline_audit_missing"],
            "source_summary": {},
        }

    offline_passed = _candidate_bool(
        offline_audit_payload,
        field_names=("pass", "gate_passed", "ok"),
    )
    if offline_passed is None:
        offline_passed = _candidate_status(
            offline_audit_payload,
            field_names=(
                "reintegrate_verdict",
                "reintegrate_status",
                "promotion_status",
            ),
        )
    if offline_passed is None:
        return {
            "available": True,
            "offline_gain_green": False,
            "failure_reasons": ["offline_audit_missing_pass_signal"],
            "source_summary": {
                "task": _optional_string(offline_audit_payload.get("task")),
                "surface_route": _optional_string(
                    offline_audit_payload.get("surface_route")
                ),
                "diagnostic_only": offline_audit_payload.get("diagnostic_only"),
                "mainline_authority": offline_audit_payload.get("mainline_authority"),
            },
        }

    failure_reasons: list[str] = []
    if not offline_passed:
        failure_reasons.append("offline_gain_not_green")
    return {
        "available": True,
        "offline_gain_green": bool(offline_passed),
        "failure_reasons": failure_reasons,
        "source_summary": {
            "task": _optional_string(offline_audit_payload.get("task")),
            "surface_route": _optional_string(
                offline_audit_payload.get("surface_route")
            ),
            "diagnostic_only": offline_audit_payload.get("diagnostic_only"),
            "mainline_authority": offline_audit_payload.get("mainline_authority"),
            "pass": offline_audit_payload.get("pass"),
            "reintegrate_verdict": offline_audit_payload.get("reintegrate_verdict"),
            "reintegrate_status": offline_audit_payload.get("reintegrate_status"),
        },
    }


def summarize_downstream_gate(
    downstream_gate_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(downstream_gate_payload, Mapping):
        return {
            "available": False,
            "downstream_gain_green": False,
            "failure_reasons": ["downstream_gate_missing"],
            "source_summary": {},
        }

    downstream_passed = _candidate_bool(
        downstream_gate_payload,
        field_names=("gate_passed", "pass", "ok"),
    )
    if downstream_passed is None:
        downstream_passed = _candidate_status(
            downstream_gate_payload,
            field_names=("gate_status", "promotion_status"),
        )
    if downstream_passed is None:
        return {
            "available": True,
            "downstream_gain_green": False,
            "failure_reasons": ["downstream_gate_missing_pass_signal"],
            "source_summary": {
                "gate_name": _optional_string(downstream_gate_payload.get("gate_name")),
                "gate_status": _optional_string(
                    downstream_gate_payload.get("gate_status")
                ),
                "gate_semantics": _optional_string(
                    downstream_gate_payload.get("gate_semantics")
                ),
                "release_gate": downstream_gate_payload.get("release_gate"),
            },
        }

    failure_reasons: list[str] = []
    if not downstream_passed:
        failure_reasons.append("downstream_gain_not_green")
    return {
        "available": True,
        "downstream_gain_green": bool(downstream_passed),
        "failure_reasons": failure_reasons,
        "source_summary": {
            "gate_name": _optional_string(downstream_gate_payload.get("gate_name")),
            "gate_status": _optional_string(downstream_gate_payload.get("gate_status")),
            "gate_semantics": _optional_string(
                downstream_gate_payload.get("gate_semantics")
            ),
            "release_gate": downstream_gate_payload.get("release_gate"),
            "gate_passed": downstream_gate_payload.get("gate_passed"),
            "critic_passed": downstream_gate_payload.get("critic_passed"),
            "retention_passed": downstream_gate_payload.get("retention_passed"),
            "controllability_passed": downstream_gate_payload.get(
                "controllability_passed"
            ),
        },
    }


def normalize_critic_promotion_payload(
    payload: Mapping[str, Any],
    *,
    field_name: str = CRITIC_PROMOTION_EXTENSION_KEY,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(payload).__name__}")
    schema_version = _optional_string(payload.get("schema_version"))
    if schema_version != CRITIC_PROMOTION_SCHEMA_VERSION:
        raise ValueError(
            f"{field_name}.schema_version must equal {CRITIC_PROMOTION_SCHEMA_VERSION!r}"
        )
    artifact_kind = _optional_string(payload.get("artifact_kind"))
    if artifact_kind != CRITIC_PROMOTION_ARTIFACT_KIND:
        raise ValueError(
            f"{field_name}.artifact_kind must equal {CRITIC_PROMOTION_ARTIFACT_KIND!r}"
        )
    promotion_allowed = payload.get("promotion_allowed")
    if not isinstance(promotion_allowed, bool):
        raise TypeError(f"{field_name}.promotion_allowed must be a bool")
    promotion_status = _optional_string(payload.get("promotion_status"))
    if promotion_status not in {PROMOTION_STATUS_PASS, PROMOTION_STATUS_BLOCK}:
        raise ValueError(
            f"{field_name}.promotion_status must be PASS/BLOCK, got {promotion_status!r}"
        )
    critic_role = _optional_string(payload.get("critic_role"))
    if critic_role not in {
        CRITIC_ROLE_REVIEW_ONLY,
        CRITIC_ROLE_PRIMARY_RELABEL_SOURCE,
    }:
        raise ValueError(f"{field_name}.critic_role is invalid: {critic_role!r}")
    relabel_source = _optional_string(payload.get("relabel_source"))
    if relabel_source not in {RELABEL_SOURCE_DEFAULT_MAINLINE, RELABEL_SOURCE_CRITIC}:
        raise ValueError(f"{field_name}.relabel_source is invalid: {relabel_source!r}")
    normalized: dict[str, Any] = dict(payload)
    failure_reasons = payload.get("failure_reasons")
    if not isinstance(failure_reasons, list) or any(
        not isinstance(item, str) or not item.strip() for item in failure_reasons
    ):
        raise TypeError(f"{field_name}.failure_reasons must be a string list")
    for field_name_bool in (
        "gates_a_f_green",
        "offline_gain_green",
        "downstream_gain_green",
    ):
        if not isinstance(payload.get(field_name_bool), bool):
            raise TypeError(f"{field_name}.{field_name_bool} must be a bool")
    return normalized


def build_critic_promotion_verdict(
    *,
    offline_audit_payload: Mapping[str, Any] | None = None,
    downstream_gate_payload: Mapping[str, Any] | None = None,
    gates_a_f_bundle: Mapping[str, Any] | None = None,
    evidence_paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gates_summary = summarize_gates_a_f_status(gates_a_f_bundle)
    offline_summary = summarize_offline_audit(offline_audit_payload)
    downstream_summary = summarize_downstream_gate(downstream_gate_payload)

    promotion_allowed = bool(
        gates_summary["gates_a_f_green"]
        and offline_summary["offline_gain_green"]
        and downstream_summary["downstream_gain_green"]
    )
    promotion_status = (
        PROMOTION_STATUS_PASS if promotion_allowed else PROMOTION_STATUS_BLOCK
    )
    critic_role = (
        CRITIC_ROLE_PRIMARY_RELABEL_SOURCE
        if promotion_allowed
        else CRITIC_ROLE_REVIEW_ONLY
    )
    relabel_source = (
        RELABEL_SOURCE_CRITIC if promotion_allowed else RELABEL_SOURCE_DEFAULT_MAINLINE
    )
    failure_reasons = _dedupe_failure_reasons(
        list(gates_summary["failure_reasons"])
        + list(offline_summary["failure_reasons"])
        + list(downstream_summary["failure_reasons"])
    )

    normalized_evidence_paths = {
        "offline_audit_json": _optional_string(
            None if evidence_paths is None else evidence_paths.get("offline_audit_json")
        ),
        "downstream_gate_json": _optional_string(
            None
            if evidence_paths is None
            else evidence_paths.get("downstream_gate_json")
        ),
        "gates_a_f_json": _optional_string(
            None if evidence_paths is None else evidence_paths.get("gates_a_f_json")
        ),
    }
    payload = {
        "schema_version": CRITIC_PROMOTION_SCHEMA_VERSION,
        "artifact_kind": CRITIC_PROMOTION_ARTIFACT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gate_name": CRITIC_PROMOTION_GATE_NAME,
        "promotion_allowed": bool(promotion_allowed),
        "promotion_status": promotion_status,
        "critic_role": critic_role,
        "relabel_source": relabel_source,
        "gates_a_f_green": bool(gates_summary["gates_a_f_green"]),
        "offline_gain_green": bool(offline_summary["offline_gain_green"]),
        "downstream_gain_green": bool(downstream_summary["downstream_gain_green"]),
        "failure_reasons": failure_reasons,
        "gates_a_f": dict(gates_summary["gates"]),
        "evidence_paths": normalized_evidence_paths,
        "input_evaluation": {
            "offline_audit": {
                "available": bool(offline_summary["available"]),
                "green": bool(offline_summary["offline_gain_green"]),
                "failure_reasons": list(offline_summary["failure_reasons"]),
                "source_summary": dict(offline_summary["source_summary"]),
            },
            "downstream_gate": {
                "available": bool(downstream_summary["available"]),
                "green": bool(downstream_summary["downstream_gain_green"]),
                "failure_reasons": list(downstream_summary["failure_reasons"]),
                "source_summary": dict(downstream_summary["source_summary"]),
            },
            "gates_a_f": {
                "green": bool(gates_summary["gates_a_f_green"]),
                "missing_gates": list(gates_summary["missing_gates"]),
                "failure_reasons": list(gates_summary["failure_reasons"]),
            },
        },
    }
    return normalize_critic_promotion_payload(payload)


def write_critic_promotion_payload(path: Path, payload: Mapping[str, Any]) -> Path:
    normalized = normalize_critic_promotion_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)
    return path


__all__ = [
    "CRITIC_PROMOTION_ARTIFACT_KIND",
    "CRITIC_PROMOTION_EXTENSION_KEY",
    "CRITIC_PROMOTION_GATE_NAME",
    "CRITIC_PROMOTION_SCHEMA_VERSION",
    "CRITIC_ROLE_PRIMARY_RELABEL_SOURCE",
    "CRITIC_ROLE_REVIEW_ONLY",
    "GATES_A_F_ORDER",
    "PROMOTION_STATUS_BLOCK",
    "PROMOTION_STATUS_PASS",
    "RELABEL_SOURCE_CRITIC",
    "RELABEL_SOURCE_DEFAULT_MAINLINE",
    "build_critic_promotion_verdict",
    "normalize_critic_promotion_payload",
    "summarize_downstream_gate",
    "summarize_gates_a_f_status",
    "summarize_offline_audit",
    "write_critic_promotion_payload",
]
