from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from work.openpi.pipelines.recap.iter6_candidate_space import materialize_iter6_candidate_space


def test_iter6_blind_selection_rule_v2_schema_and_sidecar() -> None:
    repo_root = Path("/tmp/iter6_blind_rule_contract")
    result = materialize_iter6_candidate_space(
        repo_root,
        now_utc=datetime(2026, 4, 25, 17, 0, tzinfo=timezone.utc),
    )
    rule = result["blind_selection_rule_v2"]
    report = result["worker_report"]

    assert rule["schema_version"] == "iter6_blind_selection_rule_v2"
    assert rule["selection_rule_frozen_before_calibration"] is True
    assert rule["uses_c_results_for_selection"] is False
    assert rule["allowed_calibration_variant_codes"] == ["A", "B"]
    assert set(rule["forbidden_calibration_variant_codes"]) == {"C", "X"}
    assert report["acceptance"]["blind_selection_rule_sidecar_matches"] is True
