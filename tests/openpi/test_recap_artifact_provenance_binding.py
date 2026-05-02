from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.checkpoint import read_json  # noqa: E402
from work.openpi.recap.control_gate import build_final_gate_summary  # noqa: E402


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


def test_final_gate_summary_fails_closed_on_eval_repaired_matrix_provenance_mismatch() -> (
    None
):
    (
        repaired_matrix_summary,
        eval_summary,
        gate_results,
        blocker_verdict,
        overlay,
        task11,
    ) = _live_inputs()
    mismatched_eval_summary = dict(eval_summary)
    mismatched_eval_summary["repaired_matrix_summary_ref"] = str(
        REPO_ROOT / "agent/artifacts/openpi_recap_v1/repaired_matrix_summary_other.json"
    )

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=mismatched_eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=task11,
        source_refs=_source_refs(),
    )

    current_run_binding = _mapping(summary["current_run_binding"])
    reviewer_state = _mapping(summary["reviewer_state"])

    assert current_run_binding["passed"] is False
    assert "eval_summary_repaired_matrix_ref_mismatch" in _sequence(
        current_run_binding["blocking_reasons"]
    )
    assert reviewer_state["status"] == "blocked"
    assert summary["audit_complete"] is False


def test_final_gate_summary_fails_closed_on_gate_results_provenance_mismatch() -> None:
    (
        repaired_matrix_summary,
        eval_summary,
        gate_results,
        blocker_verdict,
        overlay,
        task11,
    ) = _live_inputs()
    mismatched_gate_results = dict(gate_results)
    mismatched_gate_results["repaired_matrix_summary_ref"] = str(
        REPO_ROOT / "agent/artifacts/openpi_recap_v1/repaired_matrix_summary_other.json"
    )

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=mismatched_gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=task11,
        source_refs=_source_refs(),
    )

    current_run_binding = _mapping(summary["current_run_binding"])

    assert current_run_binding["passed"] is False
    assert "repaired_gate_results_matrix_ref_mismatch" in _sequence(
        current_run_binding["blocking_reasons"]
    )
