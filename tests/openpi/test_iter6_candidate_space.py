from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from work.openpi.pipelines.recap.iter6_candidate_space import (
    BUDGET_GRID,
    RUN_ID,
    SUITE_CANDIDATES,
    materialize_iter6_candidate_space,
)


def test_iter6_candidate_space_emits_24_cells_and_no_c_attestation(tmp_path: Path) -> None:
    result = materialize_iter6_candidate_space(
        tmp_path,
        now_utc=datetime(2026, 4, 25, 17, 0, tzinfo=timezone.utc),
    )

    matrix = result["candidate_space_matrix"]
    assert matrix["candidate_count"] == len(BUDGET_GRID) * len(SUITE_CANDIDATES)
    assert matrix["acceptance"]["candidate_matrix_includes_budget_below_0_25"] is True
    assert matrix["acceptance"]["candidate_matrix_includes_non_spatial_or_harder_tasks"] is True

    stage = tmp_path / "agent" / "artifacts" / RUN_ID
    output_root = stage / "openpi/v22_candidate_space_iter6"
    rule_path = output_root / "blind_selection_rule_v2.json"
    sidecar_path = output_root / "blind_selection_rule_v2.sha256"
    assert rule_path.is_file()
    assert sidecar_path.is_file()
    assert sidecar_path.read_text(encoding="utf-8").split()[0] == hashlib.sha256(rule_path.read_bytes()).hexdigest()

    assert (output_root / "candidate_space_matrix.json").is_file()
    assert (output_root / "v22_candidate_space_matrix.json").is_file()
    assert (output_root / "calibration_budget_grid.json").is_file()
    assert (output_root / "budget_grid.json").is_file()
    assert (output_root / "suite_task_discovery_report.json").is_file()
    assert (output_root / "no_c_leakage_attestation.json").is_file()
    assert (stage / "coordinator/w2_blind_rule_v2_contract_assumption.json").is_file()

    report = result["worker_report"]
    assert report["acceptance"]["candidate_cells"] == 24
    assert report["acceptance"]["uses_c_results_for_selection"] is False
    assert report["acceptance"]["blind_selection_rule_sidecar_matches"] is True
