from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.checkpoint import read_json  # noqa: E402
from work.openpi.recap.control_gate import (  # noqa: E402
    WORDING_RULE_FULL_AND_PAPER_FULL_ALLOWED,
    WORDING_RULE_REPAIRED_PATH_ONLY,
    build_final_gate_summary,
    build_state_side_freeze,
)


def _mapping(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise AssertionError(f"expected mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


def _sequence(raw: object) -> Sequence[object]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise AssertionError(f"expected sequence, got {type(raw).__name__}")
    return raw


def _live_gate_rows() -> list[dict[str, object]]:
    payload = _mapping(
        read_json(
            REPO_ROOT
            / "agent/artifacts/openpi_recap_loop/iter0/eval/repaired_gate_results.json"
        )
    )
    return [dict(_mapping(row)) for row in _sequence(payload["gates"])]


def _live_summary_inputs() -> tuple[dict[str, object], ...]:
    return (
        read_json(
            REPO_ROOT / "agent/artifacts/openpi_recap_v1/repaired_matrix_summary.json"
        ),
        read_json(
            REPO_ROOT / "agent/artifacts/openpi_recap_loop/iter0/eval/eval_summary.json"
        ),
        read_json(
            REPO_ROOT
            / "agent/artifacts/openpi_recap_loop/iter0/eval/repaired_gate_results.json"
        ),
        read_json(
            REPO_ROOT
            / "agent/artifacts/openpi_recap_loop/iter0/eval/blocker_verdict.json"
        ),
        read_json(
            REPO_ROOT
            / "agent/artifacts/openpi_overlay_builds/openpi_fdc03f5_recap/overlay_materialization.json"
        ),
        read_json(
            REPO_ROOT / "agent/artifacts/openpi_recap_v1/task11_verification.json"
        ),
    )


def _source_refs() -> dict[str, object]:
    return {
        "repaired_matrix_summary": str(
            REPO_ROOT / "agent/artifacts/openpi_recap_v1/repaired_matrix_summary.json"
        ),
        "eval_summary": str(
            REPO_ROOT / "agent/artifacts/openpi_recap_loop/iter0/eval/eval_summary.json"
        ),
        "repaired_gate_results": str(
            REPO_ROOT
            / "agent/artifacts/openpi_recap_loop/iter0/eval/repaired_gate_results.json"
        ),
        "blocker_verdict": str(
            REPO_ROOT
            / "agent/artifacts/openpi_recap_loop/iter0/eval/blocker_verdict.json"
        ),
        "overlay_materialization": str(
            REPO_ROOT
            / "agent/artifacts/openpi_overlay_builds/openpi_fdc03f5_recap/overlay_materialization.json"
        ),
        "task11_verification": str(
            REPO_ROOT / "agent/artifacts/openpi_recap_v1/task11_verification.json"
        ),
    }


def _approved_reviewers() -> dict[str, object]:
    return {"F1": "APPROVE", "F2": "APPROVE", "F3": "APPROVE", "F4": "APPROVE"}


def test_state_side_freeze_releases_only_when_all_required_gates_pass() -> None:
    freeze = build_state_side_freeze(
        _live_gate_rows(),
        task11_overlay_passed=True,
        task11_verification_passed=True,
        reviewer_approvals_passed=True,
        audit_complete=True,
    )

    assert freeze["state_side_frozen"] is False
    assert freeze["task11_verification_passed"] is True
    assert freeze["missing_required_gates"] == []
    assert freeze["non_pass_required_gates"] == []
    assert freeze["freeze_reason_codes"] == []


def test_state_side_freeze_fails_closed_when_any_required_gate_is_not_pass() -> None:
    gate_rows = _live_gate_rows()
    gate_rows[3]["status"] = "fail"

    freeze = build_state_side_freeze(
        gate_rows,
        task11_overlay_passed=True,
        task11_verification_passed=True,
        reviewer_approvals_passed=True,
        audit_complete=True,
    )

    assert freeze["state_side_frozen"] is True
    assert freeze["non_pass_required_gates"] == ["G3"]
    assert "required_gate_not_passed" in _sequence(freeze["freeze_reason_codes"])


def test_state_side_freeze_fails_closed_when_only_overlay_materialization_exists() -> (
    None
):
    freeze = build_state_side_freeze(
        _live_gate_rows(),
        task11_overlay_passed=True,
        task11_verification_passed=False,
        reviewer_approvals_passed=True,
        audit_complete=True,
    )

    assert freeze["state_side_frozen"] is True
    assert freeze["task11_verification_passed"] is False
    assert freeze["non_pass_required_gates"] == []
    assert freeze["freeze_reason_codes"] == ["task11_verification_not_passed"]


def test_state_side_freeze_fails_closed_when_task11_overlay_is_not_passed() -> None:
    freeze = build_state_side_freeze(
        _live_gate_rows(),
        task11_overlay_passed=False,
        task11_verification_passed=True,
        reviewer_approvals_passed=True,
        audit_complete=True,
    )

    assert freeze["state_side_frozen"] is True
    assert freeze["non_pass_required_gates"] == []
    assert freeze["freeze_reason_codes"] == ["task11_overlay_not_passed"]


def test_state_side_freeze_stays_locked_with_real_iter0_task11_verification() -> None:
    (
        repaired_matrix_summary,
        eval_summary,
        gate_results,
        blocker_verdict,
        overlay,
        task11,
    ) = _live_summary_inputs()

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=task11,
        source_refs=_source_refs(),
    )

    freeze = _mapping(summary["state_side_freeze"])
    wording_rule = _mapping(summary["wording_rule"])
    reviewer_state = _mapping(summary["reviewer_state"])

    assert freeze["state_side_frozen"] is True
    assert "reviewer_approvals_not_passed" in _sequence(freeze["freeze_reason_codes"])
    assert "audit_incomplete" in _sequence(freeze["freeze_reason_codes"])
    assert wording_rule["rule_mode"] == WORDING_RULE_REPAIRED_PATH_ONLY
    assert reviewer_state["status"] == "pending_review"


def test_state_side_freeze_releases_with_real_iter0_task11_verification_after_approvals() -> (
    None
):
    (
        repaired_matrix_summary,
        eval_summary,
        gate_results,
        blocker_verdict,
        overlay,
        task11,
    ) = _live_summary_inputs()

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=task11,
        source_refs=_source_refs(),
        reviewer_approvals=_approved_reviewers(),
    )

    freeze = _mapping(summary["state_side_freeze"])
    wording_rule = _mapping(summary["wording_rule"])
    reviewer_state = _mapping(summary["reviewer_state"])

    assert freeze["state_side_frozen"] is False
    assert freeze["freeze_reason_codes"] == []
    assert wording_rule["rule_mode"] == WORDING_RULE_FULL_AND_PAPER_FULL_ALLOWED
    assert reviewer_state["status"] == "approve_all"
    assert summary["audit_complete"] is True
