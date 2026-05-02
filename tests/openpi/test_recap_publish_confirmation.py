from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROUND_ROOT = REPO_ROOT / "agent/artifacts/openpi_recap_confirmation_v1"
WORDING_CONTRACT = (
    REPO_ROOT
    / "agent/exchange/openpi_recap_performance_confirmation_wording_contract_v1.md"
)
REQUIRED_A_PAPER_FULL_BINDING_SENTENCE = (
    "同时，paper_full 的 finalized verification 与当前轮 run 绑定已经完成；"
    "当前仍未确认的是性能优势本身，而不是验证链或 current-run binding 缺失。"
)


from work.openpi.scripts.openpi_recap_publish_confirmation import (  # noqa: E402
    publish_confirmation,
)
from work.openpi.scripts.openpi_recap_validate_wording import (  # noqa: E402
    validate_wording,
)


def _copy_required_json(
    source_root: Path, target_root: Path, relative_path: str
) -> None:
    source_path = source_root / relative_path
    target_path = target_root / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _ = shutil.copy2(source_path, target_path)


def _build_temp_round_root(tmp_path: Path) -> Path:
    round_root = tmp_path / "openpi_recap_confirmation_v1"
    required_paths = [
        "repaired_path/paired_summary.json",
        "repaired_path/gate_report.json",
        "repaired_path/comparisons.json",
        "paper_full/paired_summary.json",
        "paper_full/gate_report.json",
        "paper_full/comparisons.json",
        "charts/figure_01_executive_headline_repaired.json",
        "charts/figure_02_metric_ladder_repaired.json",
        "charts/figure_03_task_heatmaps.json",
        "charts/figure_04_seed_stability.json",
        "charts/figure_05_success_throughput_scatter.json",
        "charts/figure_06_gate_outcome_panel.json",
        "charts/figure_07_provenance_binding_diagram.json",
        "charts/figure_08_cross_lane_panel.json",
        "videos/repaired_side_by_side_compare/video_index.json",
        "videos/paper_full_side_by_side_compare/video_index.json",
        "videos/repaired_best_case_highlight_reel/video_index.json",
        "videos/paper_full_best_case_highlight_reel/video_index.json",
        "videos/failure_case_diagnostic_reel/video_index.json",
    ]
    for relative_path in required_paths:
        _copy_required_json(SOURCE_ROUND_ROOT, round_root, relative_path)
    return round_root


def test_publish_confirmation_writes_cross_lane_outputs_and_doc(tmp_path: Path) -> None:
    round_root = _build_temp_round_root(tmp_path)
    results_doc = tmp_path / "exchange" / "openpi_recap_performance_confirmation_v1.md"

    outputs = publish_confirmation(
        input_root=round_root,
        results_doc=results_doc,
        wording_contract=WORDING_CONTRACT,
    )

    cross_lane_summary_path = round_root / "cross_lane_summary.json"
    executive_summary_path = round_root / "executive_summary.json"
    assert cross_lane_summary_path.is_file()
    assert executive_summary_path.is_file()
    assert results_doc.is_file()
    assert (
        Path(cast(str, outputs["cross_lane_summary_path"])) == cross_lane_summary_path
    )

    cross_lane_summary = cast(
        dict[str, object],
        json.loads(cross_lane_summary_path.read_text(encoding="utf-8")),
    )
    executive_summary = cast(
        dict[str, object],
        json.loads(executive_summary_path.read_text(encoding="utf-8")),
    )
    assert cross_lane_summary["repaired_path_effect_confirmed"] is True
    assert cross_lane_summary["paper_full_effect_confirmed"] is False
    assert cross_lane_summary["same_metric_layer"] is True
    assert cross_lane_summary["both_outperform_control"] is True
    assert executive_summary["allowed_conclusion_code"] == "A"
    assert (
        executive_summary["allowed_conclusion_template"]
        == "结论 A：当前 authority 已确认 repaired-path effect；paper-full effect 仍未确认。"
    )

    results_text = results_doc.read_text(encoding="utf-8")
    assert "结论 A：repaired-path confirmed，paper-full unconfirmed" in results_text
    assert (
        "结论 A：当前 authority 已确认 repaired-path effect；paper-full effect 仍未确认。"
        in results_text
    )
    assert REQUIRED_A_PAPER_FULL_BINDING_SENTENCE in results_text
    assert "paper-full 已确认" not in results_text

    wording_report = validate_wording(
        results_doc=results_doc,
        executive_summary_path=executive_summary_path,
        wording_contract_path=WORDING_CONTRACT,
    )
    assert wording_report["passed"] is True
    assert wording_report["blocking_reasons"] == []


def test_validate_wording_rejects_paper_full_overclaim(tmp_path: Path) -> None:
    round_root = _build_temp_round_root(tmp_path)
    results_doc = tmp_path / "exchange" / "openpi_recap_performance_confirmation_v1.md"
    _ = publish_confirmation(
        input_root=round_root,
        results_doc=results_doc,
        wording_contract=WORDING_CONTRACT,
    )

    tampered = results_doc.read_text(encoding="utf-8").replace(
        "结论 A：当前 authority 已确认 repaired-path effect；paper-full effect 仍未确认。",
        "结论 B：当前 authority 已确认 repaired-path effect，且已确认 paper-full effect。",
    )
    _ = results_doc.write_text(tampered, encoding="utf-8")

    wording_report = validate_wording(
        results_doc=results_doc,
        executive_summary_path=round_root / "executive_summary.json",
        wording_contract_path=WORDING_CONTRACT,
    )

    assert wording_report["passed"] is False
    reasons = cast(list[str], wording_report["blocking_reasons"])
    assert "paper_full_overclaim_detected" in reasons
    assert "non_selected_template_present:B" in reasons


def test_validate_wording_requires_explicit_a_semantic_sentence(tmp_path: Path) -> None:
    round_root = _build_temp_round_root(tmp_path)
    results_doc = tmp_path / "exchange" / "openpi_recap_performance_confirmation_v1.md"
    _ = publish_confirmation(
        input_root=round_root,
        results_doc=results_doc,
        wording_contract=WORDING_CONTRACT,
    )

    tampered = results_doc.read_text(encoding="utf-8").replace(
        REQUIRED_A_PAPER_FULL_BINDING_SENTENCE + "\n",
        "",
    )
    _ = results_doc.write_text(tampered, encoding="utf-8")

    wording_report = validate_wording(
        results_doc=results_doc,
        executive_summary_path=round_root / "executive_summary.json",
        wording_contract_path=WORDING_CONTRACT,
    )

    assert wording_report["passed"] is False
    reasons = cast(list[str], wording_report["blocking_reasons"])
    assert (
        "missing_required_a_semantic_sentence:paper_full_verification_binding_complete"
        in reasons
    )
