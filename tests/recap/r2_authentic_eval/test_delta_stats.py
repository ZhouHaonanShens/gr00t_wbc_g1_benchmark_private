"""Tests for work.recap.r2_authentic_eval.delta_stats (plan §3.1, V3-FIX-1)."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from work.recap.r1_repro.gates import (
    newcombe_ci_on_delta,
    wilson_ci_on_rate,
)
from work.recap.r2_authentic_eval.delta_stats import (
    EVIDENCE_GRADE_N_CELLS,
    R2_BASELINE_N_DEFAULT,
    R2_BASELINE_SUCC_DEFAULT,
    R2_CELL_RESULT_SCHEMA_VERSION,
    R2_DECOMPOSITION_TABLE_SCHEMA_VERSION,
    R2_SCHEMA_VERSIONS,
    R2_SUMMARY_TABLE_SCHEMA_VERSION,
    SummaryRow,
    family_wise_error_rate_at_baseline,
    newcombe_delta_ci_95,
    newcombe_half_width_at_baseline,
    per_cell_below_trigger_probability,
    wilson_ci_95,
)

# ---------------------------------------------------------------------------
# Alias tests
# ---------------------------------------------------------------------------


def test_wilson_ci_95_alias_matches_r1_wilson_ci_on_rate():
    assert wilson_ci_95(17, 30) == wilson_ci_on_rate(17, 30)
    assert wilson_ci_95(10, 30) == wilson_ci_on_rate(10, 30)
    assert wilson_ci_95(0, 30) == wilson_ci_on_rate(0, 30)


def test_newcombe_delta_ci_95_alias_matches_r1_newcombe_ci_on_delta():
    assert newcombe_delta_ci_95(17, 17, 30, 30) == newcombe_ci_on_delta(17, 17, 30, 30)
    assert newcombe_delta_ci_95(10, 17, 30, 30) == newcombe_ci_on_delta(10, 17, 30, 30)


# ---------------------------------------------------------------------------
# per_cell_below_trigger_probability
# ---------------------------------------------------------------------------


def test_per_cell_below_trigger_probability_canonical_value():
    """P(succ < threshold*n | n=30, p=17/30) ≈ 0.099 (plan §4.5 + brief D3)."""
    result = per_cell_below_trigger_probability(17, 30)
    assert abs(result - 0.0992) < 1e-3, f"Expected ≈0.099, got {result:.6f}"


# ---------------------------------------------------------------------------
# family_wise_error_rate_at_baseline
# ---------------------------------------------------------------------------


def test_family_wise_error_rate_at_baseline_canonical_value():
    """Raw-observation family-wise rate remains available when n_cells=5."""
    result = family_wise_error_rate_at_baseline(17, 30, n_cells=5)
    assert abs(result - 0.4068) < 1e-3, f"Expected ≈0.407, got {result:.6f}"


def test_family_wise_error_rate_default_uses_evidence_grade_count():
    result = family_wise_error_rate_at_baseline(17, 30)
    explicit = family_wise_error_rate_at_baseline(
        17, 30, n_cells=EVIDENCE_GRADE_N_CELLS
    )
    assert EVIDENCE_GRADE_N_CELLS == 4
    assert result == explicit
    assert abs(result - 0.3413) < 1e-3


def test_family_wise_error_rate_at_baseline_n_cells_changes_value():
    """SSOT-drift detector: different n_cells must produce different rates."""
    r4 = family_wise_error_rate_at_baseline(17, 30, n_cells=4)
    r5 = family_wise_error_rate_at_baseline(17, 30, n_cells=5)
    r6 = family_wise_error_rate_at_baseline(17, 30, n_cells=6)
    assert r4 != r5, "n_cells=4 and n_cells=5 should differ"
    assert r5 != r6, "n_cells=5 and n_cells=6 should differ"
    assert r4 != r6, "n_cells=4 and n_cells=6 should differ"
    # Monotone: more cells → higher family-wise rate
    assert r4 < r5 < r6


# ---------------------------------------------------------------------------
# newcombe_half_width_at_baseline (V3-FIX-1)
# ---------------------------------------------------------------------------


def test_newcombe_half_width_at_baseline_default_args():
    """Half-width ≈ 0.237 and matches (high-low)/2 of newcombe_ci_on_delta."""
    hw = newcombe_half_width_at_baseline(17, 30)
    assert abs(hw - 0.237) < 1e-3, f"Expected ≈0.237, got {hw:.6f}"
    # Structural equivalence
    low, high = newcombe_ci_on_delta(17, 17, 30, 30)
    expected = (high - low) / 2.0
    assert abs(hw - expected) < 1e-10


def test_newcombe_half_width_at_baseline_tracks_baseline_succ_change():
    """SSOT-drift detector: changing baseline_succ must change the half-width."""
    hw17 = newcombe_half_width_at_baseline(17, 30)
    hw18 = newcombe_half_width_at_baseline(18, 30)
    assert hw17 != hw18, "Half-width must change when baseline_succ changes"


# ---------------------------------------------------------------------------
# SummaryRow
# ---------------------------------------------------------------------------


def test_summary_row_round_trip_through_json():
    """Frozen dataclass survives json.dumps → json.loads round-trip."""
    row = SummaryRow(
        label="test_cell_A",
        training_algo="PPO",
        n_train_steps=5000,
        success_count=20,
        completed_episode_total=30,
        rate=20 / 30,
        wilson_ci_95=(0.50, 0.82),
        delta_vs_baseline=3 / 30,
        newcombe_delta_ci_95=(-0.08, 0.31),
        triggered_below_threshold=False,
    )
    d = dataclasses.asdict(row)
    json_str = json.dumps(d)
    loaded = json.loads(json_str)
    row2 = SummaryRow(
        label=loaded["label"],
        training_algo=loaded["training_algo"],
        n_train_steps=loaded["n_train_steps"],
        success_count=loaded["success_count"],
        completed_episode_total=loaded["completed_episode_total"],
        rate=loaded["rate"],
        wilson_ci_95=tuple(loaded["wilson_ci_95"]),
        delta_vs_baseline=loaded["delta_vs_baseline"],
        newcombe_delta_ci_95=tuple(loaded["newcombe_delta_ci_95"]),
        triggered_below_threshold=loaded["triggered_below_threshold"],
    )
    assert row2.label == row.label
    assert row2.training_algo == row.training_algo
    assert row2.n_train_steps == row.n_train_steps
    assert row2.success_count == row.success_count
    assert row2.completed_episode_total == row.completed_episode_total
    assert abs(row2.rate - row.rate) < 1e-10
    assert row2.wilson_ci_95 == row.wilson_ci_95
    assert abs(row2.delta_vs_baseline - row.delta_vs_baseline) < 1e-10
    assert row2.newcombe_delta_ci_95 == row.newcombe_delta_ci_95
    assert row2.triggered_below_threshold == row.triggered_below_threshold


def test_summary_row_is_frozen():
    """SummaryRow must be a frozen dataclass (immutable)."""
    row = SummaryRow(
        label="x",
        training_algo="SFT",
        n_train_steps=0,
        success_count=17,
        completed_episode_total=30,
        rate=17 / 30,
        wilson_ci_95=(0.39, 0.73),
        delta_vs_baseline=0.0,
        newcombe_delta_ci_95=(-0.24, 0.24),
        triggered_below_threshold=False,
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        row.label = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Schema version constants
# ---------------------------------------------------------------------------


def test_schema_versions_constants_present():
    """All R2_*_SCHEMA_VERSION constants must be non-empty strings."""
    assert isinstance(R2_SUMMARY_TABLE_SCHEMA_VERSION, str) and R2_SUMMARY_TABLE_SCHEMA_VERSION
    assert (
        isinstance(R2_DECOMPOSITION_TABLE_SCHEMA_VERSION, str)
        and R2_DECOMPOSITION_TABLE_SCHEMA_VERSION
    )
    assert isinstance(R2_CELL_RESULT_SCHEMA_VERSION, str) and R2_CELL_RESULT_SCHEMA_VERSION


def test_summary_table_validates_schema_version():
    """R2_SCHEMA_VERSIONS aggregates the three individual version strings."""
    assert R2_SCHEMA_VERSIONS["summary_table"] == R2_SUMMARY_TABLE_SCHEMA_VERSION
    assert R2_SCHEMA_VERSIONS["decomposition_table"] == R2_DECOMPOSITION_TABLE_SCHEMA_VERSION
    assert R2_SCHEMA_VERSIONS["cell_result"] == R2_CELL_RESULT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Source-scan guard
# ---------------------------------------------------------------------------


def test_no_t8_imports_in_delta_stats():
    """delta_stats.py must not import safe_sft or any t8_* module."""
    src = (
        Path(__file__).parent.parent.parent.parent
        / "work"
        / "recap"
        / "r2_authentic_eval"
        / "delta_stats.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "safe_sft" not in node.module, f"Forbidden import: {node.module}"
            assert not node.module.startswith("t8_"), f"Forbidden import: {node.module}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "safe_sft" not in alias.name, f"Forbidden import: {alias.name}"
                assert not alias.name.startswith("t8_"), f"Forbidden import: {alias.name}"
