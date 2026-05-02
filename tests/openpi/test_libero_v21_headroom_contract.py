from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "agent/exchange/openpi_libero_v21_headroom_contract.md"


def test_v21_headroom_contract_freezes_scope_baseline_and_variants() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO v21 headroom authority 合同",
        "eval-only phase",
        "A/B/C/X only",
        "v21 is eval-only authority lane",
        "A/B/C/X-only scope",
        "D_not_executed_in_v21=true",
        "A=stock_libero_ref_v1",
        "B=fixedadv_relabel8d_control_v1",
        "C=recap_only_relabel8d_v2",
        "X=recap_shuffledadv_diag_v1",
        "stock baseline = pi05_libero + discrete_state_input=False",
        "no training-semantic changes",
        "no D/state-token execution",
        "no G1",
        "no online loop",
        "no RL token/state leakage",
        "no submodules/openpi/** edits",
    ]
    for item in required:
        assert item in text, f"missing v21 headroom scope item: {item}"


def test_v21_headroom_contract_freezes_lite_strong_and_primary_metric_binding() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "lite=advisory",
        "strong=formal authority",
        "if lite is complete and H0/H1/H3 are not FAIL, strong must continue",
        "primary_metric_id=success_rate@0.50_budget",
        "throughput_like_score",
    ]
    for item in required:
        assert item in text, f"missing v21 headroom binding item: {item}"


def test_v21_headroom_contract_freezes_collision_table_and_v2_isolation() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "collision table",
        "collision table forbids v2 path reuse",
        "v21 must not reuse v2 path namespace",
        "v21 must not overwrite agent/artifacts/openpi_libero_v2/**",
        "v21 must not overwrite go_no_go_report_v2.json",
        "v21 must not overwrite paired_rollout_summary_abc_v2.json",
        "v21 must not overwrite agent/exchange/openpi_libero_results_v2.md",
    ]
    for item in required:
        assert item in text, f"missing v21 collision item: {item}"


def test_v21_headroom_contract_keeps_v21_naming_consistent() -> None:
    text = DOC.read_text(encoding="utf-8")
    forbidden = ["v2.1", "v2_1", "v2p1"]
    for item in forbidden:
        assert item not in text, f"unexpected alternate v21 naming found: {item}"
