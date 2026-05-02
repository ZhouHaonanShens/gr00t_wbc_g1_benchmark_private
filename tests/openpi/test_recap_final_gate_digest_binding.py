from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.checkpoint import read_json  # noqa: E402
from work.openpi.recap.control_gate import (  # noqa: E402
    build_final_gate_summary,
    canonical_json_sha256,
)


def _mapping(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise AssertionError(f"expected mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


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


def test_final_gate_summary_binds_paths_and_digests_to_iter0_current_run() -> None:
    (
        repaired_matrix_summary,
        eval_summary,
        gate_results,
        blocker_verdict,
        overlay,
        task11,
    ) = _live_inputs()
    source_refs = _source_refs()

    summary = build_final_gate_summary(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=gate_results,
        blocker_verdict=blocker_verdict,
        overlay_materialization=overlay,
        task11_verification=task11,
        source_refs=source_refs,
    )

    input_paths = _mapping(summary["input_artifact_paths"])
    input_digests = _mapping(summary["input_artifact_digests"])
    current_run_binding = _mapping(summary["current_run_binding"])
    task11_gate = _mapping(summary["task11_verification_prerequisite"])

    assert summary["iter_id"] == "iter0"
    assert summary["run_id"] == "task10_eval_iter0_cba17fb43da8"
    assert input_paths == source_refs
    assert input_digests == {
        "repaired_matrix_summary": canonical_json_sha256(repaired_matrix_summary),
        "eval_summary": canonical_json_sha256(eval_summary),
        "repaired_gate_results": canonical_json_sha256(gate_results),
        "blocker_verdict": canonical_json_sha256(blocker_verdict),
        "overlay_materialization": canonical_json_sha256(overlay),
        "task11_verification": canonical_json_sha256(task11),
    }
    assert summary["task11_verification_digest"] == canonical_json_sha256(task11)
    assert current_run_binding["passed"] is True
    assert current_run_binding["run_id"] == "task10_eval_iter0_cba17fb43da8"
    assert current_run_binding["current_run_root"] == str(
        REPO_ROOT / "agent/artifacts/openpi_recap_loop/iter0"
    )
    assert task11_gate["run_id"] == "task10_eval_iter0_cba17fb43da8"
