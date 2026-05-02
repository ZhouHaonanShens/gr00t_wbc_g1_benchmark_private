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
    REPAIRED_GATE_ORDER,
    TASK11_VERIFICATION_SCHEMA_VERSION,
    WORDING_RULE_FULL_AND_PAPER_FULL_ALLOWED,
    WORDING_RULE_REPAIRED_PATH_ONLY,
    canonical_json_sha256,
    build_final_gate_summary,
)


def _mapping(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise AssertionError(f"expected mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


def _sequence(raw: object) -> Sequence[object]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise AssertionError(f"expected sequence, got {type(raw).__name__}")
    return raw


def _live_inputs() -> tuple[dict[str, object], ...]:
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


def _task11_verification() -> dict[str, object]:
    _ = TASK11_VERIFICATION_SCHEMA_VERSION
    _ = canonical_json_sha256
    return read_json(
        REPO_ROOT / "agent/artifacts/openpi_recap_v1/task11_verification.json"
    )


def _approved_reviewers() -> dict[str, object]:
    return {"F1": "APPROVE", "F2": "APPROVE", "F3": "APPROVE", "F4": "APPROVE"}


def test_final_gate_summary_freezes_live_g0_g7_verdict_set() -> None:
    repaired_matrix_summary, eval_summary, gate_results, blocker_verdict, overlay = (
        _live_inputs()
    )

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=None,
        source_refs=_source_refs(),
    )

    gate_order = tuple(_sequence(summary["gate_order"]))
    final_gate_ids = tuple(_sequence(summary["final_gate_ids"]))
    gates = [_mapping(row) for row in _sequence(summary["gates"])]
    gate_statuses = _mapping(summary["gate_statuses"])
    single_source = _mapping(summary["single_source_of_truth"])
    wording_rule = _mapping(summary["wording_rule"])

    assert gate_order == REPAIRED_GATE_ORDER
    assert final_gate_ids == REPAIRED_GATE_ORDER
    assert [row["gate"] for row in gates] == list(REPAIRED_GATE_ORDER)
    assert set(gate_statuses) == set(REPAIRED_GATE_ORDER)
    assert summary["non_pass_gates"] == []
    assert single_source["gate_verdict_source"] == "repaired_gate_results"
    assert single_source["iter_id"] == "iter0"
    assert summary["run_id"] == "task10_eval_iter0_cba17fb43da8"
    assert summary["audit_complete"] is False

    assert wording_rule["rule_mode"] == WORDING_RULE_REPAIRED_PATH_ONLY
    assert wording_rule["allowed_path_scopes"] == ["repaired_path"]


def test_final_gate_summary_keeps_repaired_only_when_only_overlay_materialization_exists() -> (
    None
):
    repaired_matrix_summary, eval_summary, gate_results, blocker_verdict, overlay = (
        _live_inputs()
    )

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=None,
        source_refs=_source_refs(),
    )

    wording_rule = _mapping(summary["wording_rule"])
    task11_overlay = _mapping(summary["task11_overlay_prerequisite"])
    task11_verification = _mapping(summary["task11_verification_prerequisite"])

    assert task11_overlay["passed"] is True
    assert task11_verification["passed"] is False
    assert task11_verification["evidence_present"] is False
    assert wording_rule["rule_mode"] == WORDING_RULE_REPAIRED_PATH_ONLY
    assert wording_rule["forbidden_path_scopes"] == ["full_path", "paper_full_path"]


def test_final_gate_summary_restricts_wording_when_task11_overlay_fails() -> None:
    repaired_matrix_summary, eval_summary, gate_results, blocker_verdict, overlay = (
        _live_inputs()
    )
    mutated_overlay = dict(overlay)
    mutated_overlay["pinned_commit"] = "deadbeef"

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=mutated_overlay,
        task11_verification=None,
        source_refs=_source_refs(),
    )

    wording_rule = _mapping(summary["wording_rule"])
    task11_overlay = _mapping(summary["task11_overlay_prerequisite"])
    assert wording_rule["rule_mode"] == WORDING_RULE_REPAIRED_PATH_ONLY
    assert wording_rule["allowed_path_scopes"] == ["repaired_path"]
    assert wording_rule["forbidden_path_scopes"] == ["full_path", "paper_full_path"]
    assert task11_overlay["passed"] is False


def test_final_gate_summary_keeps_repaired_only_while_review_is_pending() -> None:
    repaired_matrix_summary, eval_summary, gate_results, blocker_verdict, overlay = (
        _live_inputs()
    )

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=_task11_verification(),
        source_refs=_source_refs(),
    )

    wording_rule = _mapping(summary["wording_rule"])
    task11_verification = _mapping(summary["task11_verification_prerequisite"])
    freeze = _mapping(summary["state_side_freeze"])

    assert task11_verification["passed"] is True
    assert task11_verification["run_id"] == "task10_eval_iter0_cba17fb43da8"
    assert wording_rule["rule_mode"] == WORDING_RULE_REPAIRED_PATH_ONLY
    assert wording_rule["allowed_path_scopes"] == ["repaired_path"]
    reviewer_state = _mapping(summary["reviewer_state"])
    assert reviewer_state["status"] == "pending_review"
    assert summary["audit_complete"] is False
    assert freeze["state_side_frozen"] is True
    assert "reviewer_approvals_not_passed" in _sequence(freeze["freeze_reason_codes"])
    assert "audit_incomplete" in _sequence(freeze["freeze_reason_codes"])


def test_final_gate_summary_rejects_task11_verification_when_binding_does_not_match() -> (
    None
):
    repaired_matrix_summary, eval_summary, gate_results, blocker_verdict, overlay = (
        _live_inputs()
    )
    task11_verification = _task11_verification()
    task11_verification["iter_id"] = "iter999"

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=task11_verification,
        source_refs=_source_refs(),
    )

    verification_gate = _mapping(summary["task11_verification_prerequisite"])
    wording_rule = _mapping(summary["wording_rule"])

    assert verification_gate["passed"] is False
    assert "iter_id_mismatch" in _sequence(verification_gate["blocking_reasons"])
    assert wording_rule["rule_mode"] == WORDING_RULE_REPAIRED_PATH_ONLY


def test_final_gate_summary_unlocks_when_all_reviewers_explicitly_approve() -> None:
    repaired_matrix_summary, eval_summary, gate_results, blocker_verdict, overlay = (
        _live_inputs()
    )

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=_task11_verification(),
        source_refs=_source_refs(),
        reviewer_approvals=_approved_reviewers(),
    )

    wording_rule = _mapping(summary["wording_rule"])
    reviewer_state = _mapping(summary["reviewer_state"])
    freeze = _mapping(summary["state_side_freeze"])

    assert reviewer_state["status"] == "approve_all"
    assert reviewer_state["approved_reviewers"] == ["F1", "F2", "F3", "F4"]
    assert reviewer_state["missing_reviewers"] == []
    assert reviewer_state["rejected_reviewers"] == []
    assert summary["audit_complete"] is True
    assert wording_rule["rule_mode"] == WORDING_RULE_FULL_AND_PAPER_FULL_ALLOWED
    assert wording_rule["allowed_path_scopes"] == [
        "repaired_path",
        "full_path",
        "paper_full_path",
    ]
    assert freeze["state_side_frozen"] is False
    assert freeze["freeze_reason_codes"] == []
