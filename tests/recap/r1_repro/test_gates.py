from __future__ import annotations

from pathlib import Path

import pytest

from work.recap.r1_repro.gates import Verdict
from work.recap.r1_repro.gates import descriptive_axis_sum_stats
from work.recap.r1_repro.gates import gate_r1_0_baseline_reproduction
from work.recap.r1_repro.gates import gate_r1_2_variant_audit
from work.recap.r1_repro.gates import newcombe_ci_on_delta
from work.recap.r1_repro.gates import wilson_ci_on_rate


def test_gate_r1_0_boundaries() -> None:
    assert gate_r1_0_baseline_reproduction(13, "PASS", 30, True) == Verdict.PASS
    assert gate_r1_0_baseline_reproduction(22, "PASS", 30, True) == Verdict.PASS
    assert gate_r1_0_baseline_reproduction(12, "PASS", 30, True) == Verdict.FAIL_LOW
    assert gate_r1_0_baseline_reproduction(23, "PASS", 30, True) == Verdict.FAIL_HIGH


def test_gate_r1_0_driver_status_and_episode_count_collapse() -> None:
    assert (
        gate_r1_0_baseline_reproduction(17, "RUNNING", 30, True)
        == Verdict.FAIL_DRIVER_STATUS
    )
    assert (
        gate_r1_0_baseline_reproduction(17, "PASS", 29, True)
        == Verdict.FAIL_DRIVER_STATUS
    )
    assert (
        gate_r1_0_baseline_reproduction(17, "PASS", 30, False)
        == Verdict.FAIL_OUT_OF_TREE_WRITE
    )


def test_gate_r1_2_path_missing_and_pass_with_risk(tmp_path: Path) -> None:
    assert gate_r1_2_variant_audit({}, {}, tmp_path / "missing") == Verdict.FAIL_PATH_MISSING
    variant = tmp_path / "variant"
    variant.mkdir()
    assert (
        gate_r1_2_variant_audit({"config.json": [("a", 1, 2)]}, {"LOW": [("a",)]}, variant)
        == Verdict.PASS_WITH_RISK
    )
    assert gate_r1_2_variant_audit({}, {}, variant) == Verdict.PASS_CLEAN


def test_wilson_ci_on_rate_boundaries_and_known_value() -> None:
    low_zero, high_zero = wilson_ci_on_rate(0, 30)
    assert low_zero == 0.0
    assert 0.0 < high_zero < 0.2
    low_full, high_full = wilson_ci_on_rate(30, 30)
    assert 0.8 < low_full < 1.0
    assert high_full == 1.0
    low_mid, high_mid = wilson_ci_on_rate(15, 30)
    assert low_mid == pytest.approx(0.331, abs=0.01)
    assert high_mid == pytest.approx(0.669, abs=0.01)


def test_newcombe_ci_on_delta_boundary_cases() -> None:
    low, high = newcombe_ci_on_delta(16, 15, 30, 30)
    assert low < 0.0 < high
    low_large, high_large = newcombe_ci_on_delta(30, 0, 30, 30)
    assert low_large > 0.7
    assert high_large == 1.0
    low_zero, high_zero = newcombe_ci_on_delta(15, 15, 30, 30)
    assert low_zero == pytest.approx(-high_zero, abs=0.02)
    low_neg, high_neg = newcombe_ci_on_delta(0, 30, 30, 30)
    assert low_neg == -1.0
    assert high_neg < -0.7


def test_descriptive_axis_sum_stats_schema_and_no_boolean() -> None:
    payload = descriptive_axis_sum_stats(
        per_axis_observed_deltas=[0, 2, -3, 8, -1],
        per_axis_baseline_successes=[17, 17, 17, 17, 17],
        baseline_n=30,
    )
    assert {
        "sum_abs_delta_observed",
        "sum_abs_delta_lower_95",
        "per_axis_newcombe_lower_99_over_5",
        "gap_p0b_minus_t81b0_observed",
        "unmodeled_axes_documented_in_dossier",
        "interpretation_note",
    } <= set(payload)
    assert payload["sum_abs_delta_lower_95"] >= 0.0
    assert any("S2_ADAPTER" in item for item in payload["unmodeled_axes_documented_in_dossier"])
    assert "axis_set_incomplete" not in payload
    assert "incomplete_axis_set" not in payload
