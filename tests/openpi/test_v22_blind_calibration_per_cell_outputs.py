from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_CONTRACT = (
    REPO_ROOT
    / "agent/artifacts/stage1_v22_calibration_iter7_20260426T_nextZ/coordinator/w6_iter7_input_contract.json"
)
INPUT_CONTRACT_SHA256 = "3b19b6ac911c5a124972d58138b3398cc6d4585e30eea5fb42094c78fd191a1b"
EARLY_STOP_POLICY = REPO_ROOT / "work/openpi/eval/configs/v22_early_stop_policy_default.json"


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_legacy_full_rewrite_jsonl_truncates_not_appends(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        legacy_full_rewrite_jsonl,
    )

    path = tmp_path / "per_episode_rollouts.jsonl"
    first = {"episode": 1, "variant_code": "A"}
    second = {"episode": 2, "variant_code": "A"}

    legacy_full_rewrite_jsonl(path, [first])
    legacy_full_rewrite_jsonl(path, [first, second])

    assert _jsonl_rows(path) == [first, second]
    assert path.read_text(encoding="utf-8").count(json.dumps(first, sort_keys=True)) == 1
    assert list(tmp_path.glob("per_episode_rollouts.jsonl.tmp*")) == []
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_atomic_jsonl_write_uses_tempfile_then_os_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import atomic_jsonl_write

    path = tmp_path / "per_episode_trace.jsonl"
    calls: list[tuple[str, str]] = []
    original_replace = os.replace

    def replace_spy(src: object, dst: object) -> None:
        calls.append((Path(src).name, Path(dst).name))
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_spy)

    atomic_jsonl_write(path, [{"episode": 1, "variant_code": "A"}])

    assert _jsonl_rows(path) == [{"episode": 1, "variant_code": "A"}]
    assert calls
    assert calls[-1][1] == "per_episode_trace.jsonl"
    assert ".tmp-" in calls[-1][0]
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_atomic_jsonl_write_no_partial_file_on_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import atomic_jsonl_write

    path = tmp_path / "per_episode_trace.jsonl"
    path.write_text('{"episode": 0}\n', encoding="utf-8")

    def failing_replace(src: object, dst: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_jsonl_write(path, [{"episode": 1}])

    assert _jsonl_rows(path) == [{"episode": 0}]
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_cell_dir_layout_matches_per_cell_artifact_contract(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        CandidateCell,
        write_cell_artifacts,
    )

    cell = CandidateCell(
        candidate_id="libero_goal__budget_0_20",
        suite_family="libero_goal",
        suite_description="",
        tasks=(),
        budget_fraction=0.20,
        calibration_variants=("A",),
        selection_inputs=("stock_A_success_rate",),
        forbidden_selection_inputs=("C_success_rate", "X_success_rate"),
    )
    cell_dir = tmp_path / "cells" / cell.candidate_id

    write_cell_artifacts(
        cell_dir=cell_dir,
        cell=cell,
        run_id="stage1_v22_runner_surface_iter7_5_20260426T_nextZ",
        mode="smoke",
        suite_family_resolution_status="resolved",
        stock_rows=({"episode_index": 0, "variant_code": "A", "success": False},),
        control_rows=({"episode_index": 0, "variant_code": "B", "success": False},),
    )

    for relative_path in (
        "cell_manifest.json",
        "precondition_check.json",
        "stock_A/per_episode_trace.jsonl",
        "stock_A/summary.json",
        "stock_A/metric_ladder_summary.json",
        "stock_A/bootstrap_ci.json",
        "control_B/per_episode_trace.jsonl",
        "control_B/summary.json",
        "control_B/metric_ladder_summary.json",
        "control_B/bootstrap_ci.json",
        "cell_status.json",
        "SHA256SUMS",
    ):
        assert (cell_dir / relative_path).is_file(), relative_path

    status = json.loads((cell_dir / "cell_status.json").read_text(encoding="utf-8"))
    assert status["selected_using_c_results"] is False
    assert status["forbidden_variants_absent"] == ["C", "X"]
