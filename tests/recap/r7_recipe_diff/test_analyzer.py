from __future__ import annotations

from pathlib import Path

import pytest

from work.recap.r7_recipe_diff import analyzer
from work.recap.r7_recipe_diff.contract import R7DiffError

OPENPI_TEXT = Path("agent/exchange/openpi_recap_fidelity_fact_report_v1.md").read_text(encoding="utf-8")
EXPECTED_STATES = {
    "C1_dual_loss": ("ABSENT", "ADD_LOSS_TERM"),
    "C2_indicator_dropout": ("ABSENT", "ADD_DATASET_AUG"),
    "C3_learned_value": ("ABSENT", "ADD_LOSS_TERM"),
    "C4_advantage_embedding_active": ("PARTIAL", "ENABLE_FLAG"),
    "C5_carrier_text_v1_grad_path": ("PARTIAL", "ADD_CLI_ARG"),
}


def test_public_analyzer_names_are_exact() -> None:
    for name in (
        "analyze_c1_dual_loss",
        "analyze_c2_indicator_dropout",
        "analyze_c3_learned_value",
        "analyze_c4_advantage_embedding_active",
        "analyze_c5_carrier_text_v1_grad_path",
        "analyze_all",
    ):
        assert callable(getattr(analyzer, name))


def test_analyze_all_rejects_a1_and_unknown_cells() -> None:
    for cell in ("A.1", "A.0", "A.9", ""):
        with pytest.raises(R7DiffError):
            analyzer.analyze_all(cell)


def test_analyze_all_a2_returns_five_ordered_deltas() -> None:
    deltas = analyzer.analyze_all("A.2")
    assert tuple(delta.component.component_id for delta in deltas) == tuple(EXPECTED_STATES)
    for delta in deltas:
        expected_state, expected_action = EXPECTED_STATES[delta.component.component_id]
        assert delta.current_state == expected_state
        assert delta.paper_prescribed_state == "IMPLEMENTED"
        assert delta.diff_action == expected_action
        assert delta.cli_arg_addition
        assert delta.evidence_files


def test_c1_dual_loss_helper_presence_does_not_count_as_implemented() -> None:
    delta = analyzer.analyze_c1_dual_loss("A.2", OPENPI_TEXT)
    assert "work/recap/dual_loss.py" in delta.evidence_files
    assert delta.current_state == "ABSENT"
    assert delta.diff_action == "ADD_LOSS_TERM"


def test_c2_dropout_helpers_do_not_count_as_implemented() -> None:
    delta = analyzer.analyze_c2_indicator_dropout("A.2", OPENPI_TEXT)
    assert "work/recap/text_indicator.py" in delta.evidence_files
    assert "work/recap/scripts/34b_recap_numeric_adv_smoke.py" in delta.evidence_files
    assert delta.current_state == "ABSENT"
    assert delta.diff_action == "ADD_DATASET_AUG"


def test_c3_static_value_labels_do_not_count_as_learned_value() -> None:
    delta = analyzer.analyze_c3_learned_value("A.2", OPENPI_TEXT)
    assert "work/recap/advantage.py" in delta.evidence_files
    assert delta.current_state == "ABSENT"
    assert delta.diff_action == "ADD_LOSS_TERM"


def test_c4_sidecar_presence_is_partial_not_implemented() -> None:
    delta = analyzer.analyze_c4_advantage_embedding_active("A.2", OPENPI_TEXT)
    assert "work/recap/advantage.py" in delta.evidence_files
    assert delta.current_state == "PARTIAL"
    assert delta.diff_action == "ENABLE_FLAG"


def test_c5_carrier_text_presence_is_partial_not_implemented() -> None:
    delta = analyzer.analyze_c5_carrier_text_v1_grad_path("A.2", OPENPI_TEXT)
    assert "work/recap/text_indicator.py" in delta.evidence_files
    assert delta.current_state == "PARTIAL"
    assert delta.diff_action == "ADD_CLI_ARG"


def test_missing_openpi_report_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analyzer, "OPENPI_REPORT_PATH", Path("agent/exchange/does_not_exist.md"))
    with pytest.raises(R7DiffError):
        analyzer.analyze_all("A.2")
