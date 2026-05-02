from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_complete_cell(cell_dir: Path, *, status: str = "PASS") -> None:
    _write_json(
        cell_dir / "cell_status.json",
        {
            "schema_version": "v22_blind_calibration_cell_status_v1",
            "cell_id": cell_dir.name,
            "suite_family": "libero_goal",
            "suite_family_resolution_status": "resolved",
            "tasks": [],
            "budget_fraction": 0.20,
            "status": status,
            "variants_run_for_selection": ["A"],
            "optional_control_variants_run": ["B"],
            "forbidden_variants_absent": ["C", "X"],
            "selected_using_c_results": False,
            "blocking_reasons": [],
        },
    )
    _write_text(cell_dir / "stock_A" / "per_episode_trace.jsonl", "{}\n")
    _write_json(cell_dir / "stock_A" / "summary.json", {"status": status})
    _write_text(cell_dir / "SHA256SUMS", "0" * 64 + "  stock_A/summary.json\n")


def _candidate_cell():
    from work.openpi.pipelines.recap.blind_calibration_runtime import CandidateCell

    return CandidateCell(
        candidate_id="libero_goal__budget_0_20",
        suite_family="libero_goal",
        suite_description="",
        tasks=(),
        budget_fraction=0.20,
        calibration_variants=("A",),
        selection_inputs=("stock_A_success_rate",),
        forbidden_selection_inputs=("C_success_rate", "X_success_rate"),
    )


def test_resume_skips_completed_cell_when_predicate_true(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        build_resume_index,
        cell_skip_predicate,
    )

    cell_dir = tmp_path / "cells" / "libero_goal__budget_0_20"
    _write_complete_cell(cell_dir)

    assert cell_skip_predicate(cell_dir)["skip"] is True
    resume_index = build_resume_index(
        tmp_path,
        [_candidate_cell()],
        skip_completed=False,
    )
    assert resume_index["schema_version"] == "v22_blind_calibration_resume_index_v1"
    assert resume_index["total_cells"] == 1
    assert resume_index["completed_cells"] == ["libero_goal__budget_0_20"]
    assert resume_index["incomplete_cells"] == []
    assert resume_index["rerun_required_cells"] == []


def test_resume_marks_incomplete_when_sha256sums_missing(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        build_resume_index,
        cell_skip_predicate,
    )

    cell_dir = tmp_path / "cells" / "libero_goal__budget_0_20"
    _write_complete_cell(cell_dir)
    (cell_dir / "SHA256SUMS").unlink()

    assert cell_skip_predicate(cell_dir)["skip"] is False
    resume_index = build_resume_index(
        tmp_path,
        [_candidate_cell()],
        skip_completed=True,
    )
    assert resume_index["completed_cells"] == []
    assert resume_index["incomplete_cells"] == ["libero_goal__budget_0_20"]
    assert resume_index["rerun_required_cells"] == ["libero_goal__budget_0_20"]


def test_resume_marks_incomplete_when_per_episode_trace_missing(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        build_resume_index,
        cell_skip_predicate,
    )

    cell_dir = tmp_path / "cells" / "libero_goal__budget_0_20"
    _write_complete_cell(cell_dir)
    (cell_dir / "stock_A" / "per_episode_trace.jsonl").unlink()

    assert cell_skip_predicate(cell_dir)["skip"] is False
    resume_index = build_resume_index(
        tmp_path,
        [_candidate_cell()],
        skip_completed=True,
    )
    assert resume_index["completed_cells"] == []
    assert resume_index["incomplete_cells"] == ["libero_goal__budget_0_20"]


def test_resume_moves_partial_outputs_to_incomplete_utc_dir(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        move_incomplete_cell_outputs,
    )

    cell_dir = tmp_path / "cells" / "libero_goal__budget_0_20"
    _write_json(cell_dir / "cell_status.json", {"status": "INCOMPLETE"})
    _write_text(cell_dir / "stock_A" / "per_episode_trace.jsonl", "{}\n")

    quarantine_dir = move_incomplete_cell_outputs(
        cell_dir,
        utc_stamp="20260426T130000Z",
    )

    assert quarantine_dir is not None
    assert quarantine_dir.parent == cell_dir
    assert quarantine_dir.name.startswith("_incomplete_20260426T130000Z")
    assert (quarantine_dir / "cell_status.json").is_file()
    assert (quarantine_dir / "stock_A" / "per_episode_trace.jsonl").is_file()
    assert not (cell_dir / "cell_status.json").exists()
