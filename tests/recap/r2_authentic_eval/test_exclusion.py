"""Tests for the R2 evidence-grade exclusion SSOT."""
from __future__ import annotations

from work.recap.r2_authentic_eval.exclusion import (
    EVIDENCE_GRADE_CELL_IDS,
    EVIDENCE_GRADE_N_CELLS,
    filter_evidence_grade,
    is_excluded_cell,
)


def test_evidence_subset_is_a2_through_a5() -> None:
    assert EVIDENCE_GRADE_CELL_IDS == ("A.2", "A.3", "A.4", "A.5")


def test_evidence_grade_n_is_four() -> None:
    assert EVIDENCE_GRADE_N_CELLS == 4


def test_cell_id_excludes_a1_but_keeps_a2() -> None:
    assert is_excluded_cell({"cell_id": "A.1"})
    assert not is_excluded_cell({"cell_id": "A.2"})


def test_path_marker_excludes_even_without_a_label() -> None:
    assert is_excluded_cell({"ckpt_abs_path": "/tmp/g2_full_training/checkpoint-2200"})


def test_filter_preserves_order_after_exclusion() -> None:
    cells = [{"cell_id": "A.5"}, {"cell_id": "A.1"}, {"cell_id": "A.2"}]
    assert filter_evidence_grade(cells) == [{"cell_id": "A.5"}, {"cell_id": "A.2"}]


def test_missing_ckpt_path_is_tolerated() -> None:
    assert not is_excluded_cell({"cell_id": "A.3"})


def test_filter_does_not_mutate_input() -> None:
    cells = [{"cell_id": "A.1"}, {"cell_id": "A.4"}]
    original = [dict(cell) for cell in cells]
    filter_evidence_grade(cells)
    assert cells == original
