from __future__ import annotations

import json
from pathlib import Path
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DOC = REPO_ROOT / "agent/exchange/openpi_libero_results_v2.md"
V3_DOC = REPO_ROOT / "agent/exchange/openpi_libero_v3_entry_prereqs.md"

GO_NO_GO_REPORT = REPO_ROOT / "agent/artifacts/openpi_libero_v2/go_no_go_report_v2.json"
PAIRED_SUMMARY = (
    REPO_ROOT / "agent/artifacts/openpi_libero_v2/paired_rollout_summary_abc_v2.json"
)
SUMMARY_A = (
    REPO_ROOT
    / "agent/artifacts/openpi_libero_v2/rollouts/stock_libero_ref_v1/rollout_strong_v2_bb2598f21b69/summary.json"
)
SUMMARY_B = (
    REPO_ROOT
    / "agent/artifacts/openpi_libero_v2/rollouts/fixedadv_relabel8d_control_v1/rollout_strong_v2_bb2598f21b69/summary.json"
)
SUMMARY_C = (
    REPO_ROOT
    / "agent/artifacts/openpi_libero_v2/rollouts/recap_only_relabel8d_v2/rollout_strong_v2_bb2598f21b69/summary.json"
)
SUMMARY_X = (
    REPO_ROOT
    / "agent/artifacts/openpi_libero_v2/rollouts/recap_shuffledadv_diag_v1/rollout_strong_v2_bb2598f21b69/summary.json"
)


def _mapping(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TypeError(f"expected dict, got {type(raw).__name__}")
    return cast(dict[str, object], raw)


def _load_json(path: Path) -> dict[str, object]:
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    return _mapping(payload)


def test_results_doc_references_only_current_v2_headline_artifacts() -> None:
    text = RESULTS_DOC.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO fresh-rollout v2 结果",
        "agent/artifacts/openpi_libero_v2/go_no_go_report_v2.json",
        "agent/artifacts/openpi_libero_v2/paired_rollout_summary_abc_v2.json",
        "agent/artifacts/openpi_libero_v2/rollouts/stock_libero_ref_v1/rollout_strong_v2_bb2598f21b69/summary.json",
        "agent/artifacts/openpi_libero_v2/rollouts/fixedadv_relabel8d_control_v1/rollout_strong_v2_bb2598f21b69/summary.json",
        "agent/artifacts/openpi_libero_v2/rollouts/recap_only_relabel8d_v2/rollout_strong_v2_bb2598f21b69/summary.json",
        "agent/artifacts/openpi_libero_v2/rollouts/recap_shuffledadv_diag_v1/rollout_strong_v2_bb2598f21b69/summary.json",
        "RECAP_NOT_VALIDATED_YET",
        "STATE_SIDE_NOT_ENTERED",
        "X.non_headline_only=true",
    ]
    for item in required:
        assert item in text, f"missing required v2 results item: {item}"

    forbidden = [
        "RECAP_VALIDATED",
        "STATE_SIDE_ENTERED_CONDITIONALLY",
        "paired_rollout_summary_abcd_v2.json",
        "agent/artifacts/openpi_libero_native/summary.json",
        "agent/artifacts/openpi_libero_recap_eval/",
        "0.29995794708646445",
        "0.2993501616432094",
    ]
    for item in forbidden:
        assert item not in text, (
            f"v2 results doc should not include legacy or state-side headline item: {item}"
        )


def test_results_doc_matches_current_machine_checkable_v2_outputs() -> None:
    text = RESULTS_DOC.read_text(encoding="utf-8")
    report = _load_json(GO_NO_GO_REPORT)
    paired_summary = _load_json(PAIRED_SUMMARY)
    summary_a = _load_json(SUMMARY_A)
    summary_b = _load_json(SUMMARY_B)
    summary_c = _load_json(SUMMARY_C)
    summary_x = _load_json(SUMMARY_X)

    manifest = _mapping(report["manifest"])
    pairwise_deltas = _mapping(paired_summary["pairwise_deltas"])
    b_minus_a = _mapping(pairwise_deltas["B_minus_A"])
    c_minus_b = _mapping(pairwise_deltas["C_minus_B"])
    c_minus_a = _mapping(pairwise_deltas["C_minus_A"])

    required = [
        f"eval_authority={report['eval_authority']}",
        f"manifest_name={manifest['manifest_name']}",
        f"task_suite_name={manifest['task_suite_name']}",
        "task_ids=[0,1]",
        "seed_manifest=[7,17,27,37,47,57]",
        f"num_trials_per_task={manifest['num_trials_per_task']}",
        f"expected_episode_count={manifest['expected_episode_count']}",
        f"A.success_rate={_mapping(summary_a['rollout_summary'])['success_rate']}",
        f"A.success_count={_mapping(summary_a['rollout_summary'])['success_count']}",
        f"A.failure_count={_mapping(summary_a['rollout_summary'])['failure_count']}",
        f"B.success_rate={_mapping(summary_b['rollout_summary'])['success_rate']}",
        f"B.success_count={_mapping(summary_b['rollout_summary'])['success_count']}",
        f"B.failure_count={_mapping(summary_b['rollout_summary'])['failure_count']}",
        f"C.success_rate={_mapping(summary_c['rollout_summary'])['success_rate']}",
        f"C.success_count={_mapping(summary_c['rollout_summary'])['success_count']}",
        f"C.failure_count={_mapping(summary_c['rollout_summary'])['failure_count']}",
        f"B_minus_A.point_estimate_pp={b_minus_a['point_estimate']}",
        "B_minus_A.ci95_pp=[-6.25,0.0]",
        f"C_minus_B.point_estimate_pp={c_minus_b['point_estimate']}",
        "C_minus_B.ci95_pp=[0.0,6.25]",
        f"C_minus_A.point_estimate_pp={c_minus_a['point_estimate']}",
        "C_minus_A.ci95_pp=[0.0,0.0]",
        "G2=PASS",
        "G3=PASS",
        "G4=HOLD",
        "G5=PASS",
        "G6=NOT_APPLICABLE",
        "G7=NOT_APPLICABLE",
        f"eligible_for_state_side={str(report['eligible_for_state_side']).lower()}",
        f"X.success_rate={_mapping(summary_x['rollout_summary'])['success_rate']}",
        f"X.success_count={_mapping(summary_x['rollout_summary'])['success_count']}",
        f"X.failure_count={_mapping(summary_x['rollout_summary'])['failure_count']}",
    ]
    for item in required:
        assert item in text, f"missing machine-checkable v2 value: {item}"

    assert report["state_side_status"] == "STATE_SIDE_NOT_ENTERED"
    assert report["eligible_for_state_side"] is False
    assert paired_summary["eval_authority"] == "fresh_rollout_v2"


def test_v3_entry_doc_binds_to_g4_through_g7_and_concludes_false() -> None:
    text = V3_DOC.read_text(encoding="utf-8")
    report = _load_json(GO_NO_GO_REPORT)
    gates = cast(list[object], report["gates"])
    gate_status = {
        str(_mapping(item)["gate"]): str(_mapping(item)["status"]) for item in gates
    }

    required = [
        "openpi LIBERO v3 state-transfer 入口前提",
        "本文只回答一个问题：`eligible_for_state_transfer_v3=true/false`。",
        "agent/artifacts/openpi_libero_v2/go_no_go_report_v2.json",
        f"G4={gate_status['G4']}",
        f"G5={gate_status['G5']}",
        f"G6={gate_status['G6']}",
        f"G7={gate_status['G7']}",
        f"state_side_status={report['state_side_status']}",
        "eligible_for_state_transfer_v3=false",
    ]
    for item in required:
        assert item in text, f"missing v3 entry prereq item: {item}"

    forbidden = [
        "STATE_SIDE_ENTERED_CONDITIONALLY",
        "设计 proposal",
        "implementation plan",
    ]
    for item in forbidden:
        assert item not in text, f"v3 entry doc should stay gate-bound only: {item}"

    assert "\n- eligible_for_state_transfer_v3=true\n" not in text

    assert gate_status == {
        "G0": "PASS",
        "G1": "PASS",
        "G2": "PASS",
        "G3": "PASS",
        "G4": "HOLD",
        "G5": "PASS",
        "G6": "NOT_APPLICABLE",
        "G7": "NOT_APPLICABLE",
    }
