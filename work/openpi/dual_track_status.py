"""Dual-track status helpers for the OpenPI blocker-push lane.

The dual-track plan keeps formal gate evidence separate from exploratory
signals.  This module centralizes the OpenPI status shape so tests and small
execution wrappers do not accidentally encode exploratory evidence as a formal
pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast


FORMAL_STATUS_SCHEMA_VERSION = "dual_track_formal_status_v1"
EXPLORATORY_SIGNAL_SCHEMA_VERSION = "dual_track_exploratory_signal_v1"
OPENPI_LANE = "openpi"
FORMAL_STATUSES = frozenset({"PASS", "BLOCK", "SKIPPED"})
EXPLORATORY_STATUSES = frozenset({"SIGNAL", "NO_SIGNAL", "SKIPPED", "FAILED"})
OPENPI_PASS_RUNTIME_LEVELS = frozenset(
    {
        "materialization_ready",
        "p0_loader_runtime_pass",
        "p1_one_step_pass",
        "p2_overfit_or_tiny_update_pass",
    }
)
OPENPI_RUNTIME_LEVEL_ALLOWED_CLAIMS = {
    "materialization_ready": frozenset({"materialization_ready"}),
    "p0_loader_runtime_pass": frozenset({"materialization_ready", "p0_loader_runtime_pass"}),
    "p1_one_step_pass": frozenset(
        {"materialization_ready", "p0_loader_runtime_pass", "p1_one_step_pass"}
    ),
    "p2_overfit_or_tiny_update_pass": frozenset(
        {
            "materialization_ready",
            "p0_loader_runtime_pass",
            "p1_one_step_pass",
            "p2_overfit_or_tiny_update_pass",
        }
    ),
}
OPENPI_RUNTIME_LEVEL_CLAIM_ORDER = (
    "materialization_ready",
    "p0_loader_runtime_pass",
    "p1_one_step_pass",
    "p2_overfit_or_tiny_update_pass",
)
DATASET_NOT_MATERIALIZED = "dataset_not_materialized"
IDENTITY_BLOCKER_CODES = frozenset(
    {
        "missing_cross_dataset_task_identity",
        "missing_verified_episode_frame_crosswalk",
        "missing_cross_dataset_frame_identity",
        "task_text_universe_not_proven_compatible",
        "weak_key_only_join_rejected",
    }
)
LABEL_SEMANTICS_BLOCK = "label_semantics_block"
RUNTIME_LEVEL_NONE = "none"
RUNTIME_LEVEL_P0_LOADER = "p0_loader_runtime_pass"
RUNTIME_LEVEL_P1_ONE_STEP = "p1_one_step_pass"
RUNTIME_LEVEL_P2_TINY_UPDATE = "p2_overfit_or_tiny_update_pass"
RUNTIME_LEVELS = (
    RUNTIME_LEVEL_NONE,
    RUNTIME_LEVEL_P0_LOADER,
    RUNTIME_LEVEL_P1_ONE_STEP,
    RUNTIME_LEVEL_P2_TINY_UPDATE,
)
RUNTIME_LEVEL_RANK = {level: rank for rank, level in enumerate(RUNTIME_LEVELS)}
RUNTIME_EVIDENCE_PENDING_BLOCKERS = (
    "p0_loader_runtime_evidence_pending",
    "p1_one_step_runtime_evidence_pending",
    "p2_tiny_update_runtime_evidence_pending",
)
RUNTIME_EVIDENCE_MISSING = "runtime_evidence_missing"

FormalStatus = Literal["PASS", "BLOCK", "SKIPPED"]
ExploratoryStatus = Literal["SIGNAL", "NO_SIGNAL", "SKIPPED", "FAILED"]
RuntimeLevel = Literal[
    "materialization_ready",
    "p0_loader_runtime_pass",
    "p1_one_step_pass",
    "p2_overfit_or_tiny_update_pass",
]


def _string_list(raw: object) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [str(item) for item in raw]


def _is_materialized(p0_scope_audit_manifest: Mapping[str, object]) -> bool:
    raw_status = p0_scope_audit_manifest.get("dataset_join_root_status")
    if not isinstance(raw_status, Mapping):
        return False
    return raw_status.get("materialized") is True


def _join_blocker_codes(join_report: Mapping[str, object] | None) -> list[str]:
    if join_report is None:
        return []
    raw_blockers = join_report.get("hard_blockers", [])
    if not isinstance(raw_blockers, Sequence) or isinstance(
        raw_blockers, (str, bytes, bytearray)
    ):
        return []
    codes: list[str] = []
    for item in raw_blockers:
        if isinstance(item, Mapping) and item.get("code") is not None:
            codes.append(str(item["code"]))
    return codes


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def _default_runtime_level(*, materialized: bool, blockers: Sequence[str]) -> str:
    if not materialized and DATASET_NOT_MATERIALIZED in blockers:
        return "blocked_materialization_reverify_failed"
    if blockers:
        return "blocked_formal_prereq"
    return "materialization_ready"


def _runtime_claims_for_level(runtime_level: str) -> list[str]:
    if runtime_level in OPENPI_RUNTIME_LEVEL_ALLOWED_CLAIMS:
        allowed = OPENPI_RUNTIME_LEVEL_ALLOWED_CLAIMS[runtime_level]
        return [claim for claim in OPENPI_RUNTIME_LEVEL_CLAIM_ORDER if claim in allowed]
    return []


def _validate_runtime_level(runtime_level: object, runtime_claims: Sequence[str]) -> None:
    assert isinstance(runtime_level, str) and runtime_level
    assert runtime_level in OPENPI_PASS_RUNTIME_LEVELS or runtime_level.startswith("blocked_")
    if runtime_level in OPENPI_RUNTIME_LEVEL_ALLOWED_CLAIMS:
        allowed = OPENPI_RUNTIME_LEVEL_ALLOWED_CLAIMS[runtime_level]
        assert not (set(runtime_claims) - allowed)


def _runtime_rank(runtime_level: str) -> int:
    if runtime_level == "materialization_ready":
        return RUNTIME_LEVEL_RANK[RUNTIME_LEVEL_NONE]
    return RUNTIME_LEVEL_RANK.get(runtime_level, -1)


def _runtime_blockers(
    *,
    runtime_level: str | None,
    required_runtime_level: str,
    runtime_evidence: Sequence[str],
) -> list[str]:
    if required_runtime_level not in RUNTIME_LEVEL_RANK:
        raise ValueError(f"invalid required runtime level: {required_runtime_level}")

    effective_level = (
        "materialization_ready"
        if runtime_level in (None, RUNTIME_LEVEL_NONE, "materialization_ready")
        else str(runtime_level)
    )
    if effective_level.startswith("blocked_"):
        return [effective_level]
    if effective_level not in OPENPI_PASS_RUNTIME_LEVELS:
        raise ValueError(f"invalid runtime level: {effective_level}")

    blockers: list[str] = []
    if _runtime_rank(effective_level) < _runtime_rank(RUNTIME_LEVEL_P0_LOADER):
        blockers.append("p0_loader_runtime_evidence_pending")
    if _runtime_rank(effective_level) < _runtime_rank(RUNTIME_LEVEL_P1_ONE_STEP):
        blockers.append("p1_one_step_runtime_evidence_pending")
    if _runtime_rank(required_runtime_level) >= _runtime_rank(RUNTIME_LEVEL_P2_TINY_UPDATE) and (
        _runtime_rank(effective_level) < _runtime_rank(RUNTIME_LEVEL_P2_TINY_UPDATE)
    ):
        blockers.append("p2_tiny_update_runtime_evidence_pending")
    if _runtime_rank(effective_level) >= _runtime_rank(required_runtime_level) and not runtime_evidence:
        blockers.append(RUNTIME_EVIDENCE_MISSING)
    return blockers


def build_openpi_formal_status(
    *,
    p0_scope_audit_manifest: Mapping[str, object],
    authority_inputs: Sequence[str],
    join_report: Mapping[str, object] | None = None,
    label_semantics_pass: bool = False,
    runtime_level: str | None = None,
    required_runtime_level: str = RUNTIME_LEVEL_P1_ONE_STEP,
    runtime_evidence: Sequence[str] = (),
    validator_outputs: Sequence[str] = (),
    runtime_claims: Sequence[str] | None = None,
    notes: str = "",
) -> dict[str, object]:
    """Build the formal OpenPI status without consuming exploratory evidence.

    Formal pass requires a materialized p0 dataset, no identity blockers from the
    join report, a separate label-semantics pass, and runtime evidence at or above
    ``required_runtime_level``.  Any failure is reported as ``status="BLOCK"`` with
    atomic blocker codes in ``blocking_reasons``; compound strings such as
    ``BLOCK(label_semantics_block)`` are intentionally forbidden.
    """

    blockers: list[str] = []
    materialized = _is_materialized(p0_scope_audit_manifest)
    if not materialized:
        blockers.append(DATASET_NOT_MATERIALIZED)

    join_codes = _join_blocker_codes(join_report)
    blockers.extend(code for code in join_codes if code in IDENTITY_BLOCKER_CODES)

    if not label_semantics_pass:
        blockers.append(LABEL_SEMANTICS_BLOCK)

    data_prereqs_pass = not blockers
    if data_prereqs_pass:
        blockers.extend(
            _runtime_blockers(
                runtime_level=runtime_level,
                required_runtime_level=required_runtime_level,
                runtime_evidence=runtime_evidence,
            )
        )

    unique_blockers = _unique(blockers)
    if runtime_level not in (None, RUNTIME_LEVEL_NONE):
        resolved_runtime_level = str(runtime_level)
    elif not materialized and DATASET_NOT_MATERIALIZED in unique_blockers:
        resolved_runtime_level = "blocked_materialization_reverify_failed"
    elif not data_prereqs_pass:
        resolved_runtime_level = "blocked_formal_prereq"
    else:
        resolved_runtime_level = "materialization_ready"
    resolved_runtime_claims = (
        list(runtime_claims)
        if runtime_claims is not None
        else _runtime_claims_for_level(resolved_runtime_level)
    )
    _validate_runtime_level(resolved_runtime_level, resolved_runtime_claims)
    status: FormalStatus = "PASS" if not unique_blockers else "BLOCK"
    claim_allowed = status == "PASS"
    return {
        "schema_version": FORMAL_STATUS_SCHEMA_VERSION,
        "lane": OPENPI_LANE,
        "track": "formal",
        "status": status,
        "formal_claim_allowed": claim_allowed,
        "blocking_reasons": unique_blockers,
        "runtime_level": resolved_runtime_level,
        "runtime_claims": resolved_runtime_claims,
        "authority_inputs": list(authority_inputs),
        "validator_outputs": list(validator_outputs),
        "required_runtime_level": required_runtime_level,
        "runtime_evidence": list(runtime_evidence),
        "entered_next_gate": False,
        "next_gate_allowed": claim_allowed,
        "notes": notes,
    }


def build_openpi_exploratory_signal(
    *,
    status: ExploratoryStatus,
    method: str,
    inputs: Sequence[str] = (),
    outputs: Sequence[str] = (),
    observed_signal: Mapping[str, object] | None = None,
    notes: str = "",
) -> dict[str, object]:
    """Build an exploratory OpenPI signal that is barred from formal gates."""

    if status not in EXPLORATORY_STATUSES:
        raise ValueError(f"invalid exploratory status: {status}")
    return {
        "schema_version": EXPLORATORY_SIGNAL_SCHEMA_VERSION,
        "lane": OPENPI_LANE,
        "track": "exploratory",
        "status": status,
        "exploratory_only": True,
        "formal_claim_allowed": False,
        "must_not_unlock_formal_gate": True,
        "method": method,
        "risk_label": "exploratory_not_formal",
        "inputs": list(inputs),
        "outputs": list(outputs),
        "observed_signal": dict(observed_signal or {}),
        "notes": notes,
    }


def validate_formal_status(payload: Mapping[str, object], *, lane: str = OPENPI_LANE) -> None:
    """Raise ``AssertionError`` if a formal status violates the dual-track schema."""

    assert payload.get("schema_version") == FORMAL_STATUS_SCHEMA_VERSION
    assert payload.get("lane") == lane
    assert payload.get("track") == "formal"
    status = payload.get("status")
    assert status in FORMAL_STATUSES
    assert isinstance(status, str)
    assert "(" not in status and ")" not in status
    assert isinstance(payload.get("blocking_reasons"), list)
    runtime_level = payload.get("runtime_level")
    runtime_claims = _string_list(payload.get("runtime_claims"))
    _validate_runtime_level(runtime_level, runtime_claims)
    if runtime_level == "blocked_materialization_reverify_failed":
        assert DATASET_NOT_MATERIALIZED in payload.get("blocking_reasons", [])
    if payload.get("formal_claim_allowed") is True:
        required_runtime_level = payload.get("required_runtime_level", RUNTIME_LEVEL_P1_ONE_STEP)
        assert status == "PASS"
        assert payload.get("next_gate_allowed") is True
        assert _runtime_rank(str(runtime_level)) >= _runtime_rank(
            str(required_runtime_level)
        )


def validate_exploratory_signal(
    payload: Mapping[str, object], *, lane: str = OPENPI_LANE
) -> None:
    """Raise ``AssertionError`` if exploratory evidence can unlock formal gates."""

    assert payload.get("schema_version") == EXPLORATORY_SIGNAL_SCHEMA_VERSION
    assert payload.get("lane") == lane
    assert payload.get("track") == "exploratory"
    assert payload.get("status") in EXPLORATORY_STATUSES
    assert payload.get("exploratory_only") is True
    assert payload.get("formal_claim_allowed") is False
    assert payload.get("must_not_unlock_formal_gate") is True
    assert payload.get("risk_label") == "exploratory_not_formal"


def build_openpi_summary_section(
    *, formal_status: Mapping[str, object], exploratory_signal: Mapping[str, object]
) -> dict[str, object]:
    """Build the OpenPI quarter-section for ``dual_track_summary.json``."""

    validate_formal_status(formal_status)
    validate_exploratory_signal(exploratory_signal)
    return {
        "formal": {
            "status": str(formal_status["status"]),
            "formal_claim_allowed": bool(formal_status["formal_claim_allowed"]),
            "artifact": str(formal_status.get("artifact", "")),
            "runtime_level": str(formal_status.get("runtime_level", RUNTIME_LEVEL_NONE)),
        },
        "exploratory": {
            "status": str(exploratory_signal["status"]),
            "artifact": str(exploratory_signal.get("artifact", "")),
        },
    }


__all__ = [
    "DATASET_NOT_MATERIALIZED",
    "EXPLORATORY_SIGNAL_SCHEMA_VERSION",
    "EXPLORATORY_STATUSES",
    "FORMAL_STATUS_SCHEMA_VERSION",
    "FORMAL_STATUSES",
    "IDENTITY_BLOCKER_CODES",
    "LABEL_SEMANTICS_BLOCK",
    "OPENPI_LANE",
    "OPENPI_PASS_RUNTIME_LEVELS",
    "OPENPI_RUNTIME_LEVEL_ALLOWED_CLAIMS",
    "build_openpi_exploratory_signal",
    "build_openpi_formal_status",
    "build_openpi_summary_section",
    "validate_exploratory_signal",
    "validate_formal_status",
]
