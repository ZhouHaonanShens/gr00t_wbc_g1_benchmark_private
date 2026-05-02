from __future__ import annotations

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = (
    REPO_ROOT
    / "agent/artifacts/stage1_v22_redesign_iter6_20260425T_nextZ/openpi/v22_candidate_space_iter6/candidate_space_matrix.json"
)
REQUIRED_FLAGS = (
    "--input-contract",
    "--output-dir",
    "--runtime-log-dir",
    "--mode",
    "--max-cells",
    "--cell-id",
    "--resume",
    "--skip-completed",
    "--calibration-variants",
    "--optional-control-variants",
    "--per-cell-timeout-sec",
    "--early-stop-policy",
    "--no-c-results",
    "--no-x-results",
    "--no-sudo",
    "--episodes-per-cell-smoke",
    "--input-contract-sha256",
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def test_cli_surface_exposes_all_required_flags() -> None:
    from work.openpi.eval import v22_blind_calibration_runner as runner

    parser = runner.build_parser()
    help_text = parser.format_help()

    for flag in REQUIRED_FLAGS:
        assert flag in help_text

    mode_actions = [
        action for action in parser._actions if "--mode" in action.option_strings
    ]
    assert len(mode_actions) == 1
    assert set(mode_actions[0].choices) == {"dry-run", "smoke", "calibrate"}


def test_matrix_expansion_24_cells_keyed_verbatim() -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        load_candidate_cells,
    )

    cells = load_candidate_cells(MATRIX_PATH)
    candidate_ids = [cell.candidate_id for cell in cells]

    assert len(candidate_ids) == 24
    assert "libero_goal__budget_0_20" in candidate_ids
    assert "libero_object__budget_0_33" in candidate_ids
    assert "other_locally_supported_LIBERO_suites__budget_0_40" in candidate_ids
    assert all("__taskset_" not in candidate_id for candidate_id in candidate_ids)
    assert all("0p" not in candidate_id for candidate_id in candidate_ids)
    assert all("__budget_0_" in candidate_id for candidate_id in candidate_ids)


def test_cell_plan_lists_24_candidate_ids_with_resolution_status() -> None:
    from work.openpi.eval.v22_blind_calibration_runner import build_cell_plan
    from work.openpi.pipelines.recap.blind_calibration_runtime import load_candidate_cells

    plan = build_cell_plan(load_candidate_cells(MATRIX_PATH))

    assert plan["schema_version"] == "v22_blind_calibration_cell_plan_v1"
    assert plan["total_cells"] == 24
    assert len(plan["cells"]) == 24
    for cell in plan["cells"]:
        assert cell["candidate_id"]
        assert cell["suite_family_resolution_status"] in {
            "resolved",
            "probe_resolved_in_dry_run",
            "probe_failed_in_dry_run",
        }


def test_suite_probe_does_not_walk_artifacts_or_work_dirs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import SuiteFamilyProbe

    forbidden_roots = (
        REPO_ROOT / "agent/artifacts",
        REPO_ROOT / "agent/runtime_logs",
        REPO_ROOT / "work",
    )
    original_iterdir = Path.iterdir

    def guarded_walk(top: object, *args: object, **kwargs: object) -> object:
        path = Path(top)
        if any(_is_relative_to(path, root) for root in forbidden_roots):
            raise AssertionError(f"SuiteFamilyProbe walked forbidden root: {path}")
        return iter(())

    def guarded_iterdir(path: Path) -> object:
        if any(_is_relative_to(path, root) for root in forbidden_roots):
            raise AssertionError(f"SuiteFamilyProbe iterated forbidden root: {path}")
        return original_iterdir(path)

    monkeypatch.setattr(os, "walk", guarded_walk)
    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)

    probe = SuiteFamilyProbe(mode="cpu_introspection", repo_root=REPO_ROOT)
    result = probe.resolve_suite_family("other_locally_supported_LIBERO_suites")

    assert result["suite_family_resolution_status"] in {
        "resolved",
        "probe_resolved_in_dry_run",
        "probe_failed_in_dry_run",
    }
