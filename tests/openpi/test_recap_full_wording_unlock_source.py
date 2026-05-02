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
    WORDING_RULE_REPAIRED_PATH_ONLY,
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


def test_full_wording_stays_locked_until_review_wave_completes() -> None:
    (
        repaired_matrix_summary,
        eval_summary,
        gate_results,
        blocker_verdict,
        overlay,
        task11,
    ) = _live_inputs()

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=task11,
        source_refs=_source_refs(),
    )

    wording_rule = _mapping(summary["wording_rule"])
    freeze = _mapping(summary["state_side_freeze"])
    reviewer_state = _mapping(summary["reviewer_state"])

    assert wording_rule["rule_mode"] == WORDING_RULE_REPAIRED_PATH_ONLY
    assert freeze["state_side_frozen"] is True
    assert reviewer_state["status"] == "pending_review"
    assert summary["audit_complete"] is False


def test_full_wording_stays_locked_when_unlock_source_mismatches_overlay_tree() -> None:
    (
        repaired_matrix_summary,
        eval_summary,
        gate_results,
        blocker_verdict,
        overlay,
        task11,
    ) = _live_inputs()
    mismatched_task11 = dict(task11)
    mismatched_source_refs = dict(_mapping(task11["source_refs"]))
    mismatched_source_refs["materialized_openpi_tree"] = str(
        REPO_ROOT / "agent/artifacts/openpi_overlay_builds/openpi_wrong_overlay"
    )
    mismatched_task11["source_refs"] = mismatched_source_refs

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=mismatched_task11,
        source_refs=_source_refs(),
    )

    verification_gate = _mapping(summary["task11_verification_prerequisite"])
    wording_rule = _mapping(summary["wording_rule"])
    freeze = _mapping(summary["state_side_freeze"])

    assert verification_gate["passed"] is False
    assert "materialized_openpi_tree_source_ref_mismatch" in _sequence(
        verification_gate["blocking_reasons"]
    )
    assert wording_rule["rule_mode"] == WORDING_RULE_REPAIRED_PATH_ONLY
    assert freeze["state_side_frozen"] is True
