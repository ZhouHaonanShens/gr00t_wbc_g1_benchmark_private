from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from work.openpi.pipelines.recap.iter6_candidate_space import (
    BUDGET_GRID,
    RUN_ID,
    SUITE_CANDIDATES,
    materialize_iter6_candidate_space,
)


def test_iter6_v22_candidate_space_matrix_schema() -> None:
    result = materialize_iter6_candidate_space(
        Path("/tmp/iter6_candidate_space_contract"),
        now_utc=datetime(2026, 4, 25, 17, 0, tzinfo=timezone.utc),
    )
    matrix = result["candidate_space_matrix"]

    assert matrix["schema_version"] == "iter6_v22_candidate_space_matrix_v1"
    assert matrix["run_id"] == RUN_ID
    assert matrix["candidate_count"] == len(BUDGET_GRID) * len(SUITE_CANDIDATES)
    assert matrix["acceptance"]["candidate_matrix_includes_budget_below_0_25"] is True
    assert matrix["acceptance"]["candidate_matrix_includes_non_spatial_or_harder_tasks"] is True
    assert matrix["acceptance"]["blind_selection_rule_v2_uses_c_results_for_selection"] is False
    for cell in matrix["candidate_cells"]:
        assert set(cell["calibration_variants"]) == {"A", "B"}
        assert "C_success_rate" in cell["forbidden_selection_inputs"]
