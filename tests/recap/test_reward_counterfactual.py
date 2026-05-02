from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import drop_events
from work.recap.scripts import build_drop_sidecar
from work.recap.scripts import eval_drop_detector_against_diagnostic_pool
from work.recap.scripts import relabel_counterfactual_rewards


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_dataset_fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset_dir = tmp_path / "dataset"
    episodes = [
        {
            "episode_id": "episode_nominal",
            "episode_return_online": -1.0,
            "success_episode": True,
            "C_fail": 5,
        },
        {
            "episode_id": "episode_drop",
            "episode_return_online": -8.0,
            "success_episode": False,
            "C_fail": 5,
        },
    ]
    transitions = [
        {
            "episode_id": "episode_nominal",
            "t": 0,
            "reward_online": -1.0,
            "inner_rewards": [-0.5, -0.5],
            "success_step": False,
            "analysis_only": {"semantic_state": "SEARCHING"},
            "privileged": {"apple_in_hand": False},
        },
        {
            "episode_id": "episode_nominal",
            "t": 1,
            "reward_online": 0.0,
            "inner_rewards": [0.0],
            "success_step": True,
            "analysis_only": {"semantic_state": "PLACING"},
            "privileged": {"apple_in_hand": True},
        },
        {
            "episode_id": "episode_drop",
            "t": 0,
            "reward_online": -1.0,
            "inner_rewards": [-0.5, -0.5],
            "success_step": False,
            "analysis_only": {"semantic_state": "GRASPING"},
            "privileged": {"apple_in_hand": False},
        },
        {
            "episode_id": "episode_drop",
            "t": 1,
            "reward_online": -1.0,
            "inner_rewards": [-0.5, -0.5],
            "success_step": False,
            "analysis_only": {"semantic_state": "TRANSPORTING"},
            "privileged": {"apple_in_hand": True},
        },
        {
            "episode_id": "episode_drop",
            "t": 2,
            "reward_online": -1.0,
            "inner_rewards": [-0.5, -0.5],
            "success_step": False,
            "analysis_only": {"semantic_state": "TRANSPORTING"},
            "privileged": {"apple_in_hand": False},
        },
        {
            "episode_id": "episode_drop",
            "t": 3,
            "reward_online": -5.0,
            "inner_rewards": [-5.0],
            "success_step": False,
            "analysis_only": {"semantic_state": "TRANSPORTING"},
            "privileged": {"apple_in_hand": False},
        },
    ]
    diagnostic_pool_rows = [
        {"episode_id": "episode_nominal", "expected_drop_during_transport": False},
        {"episode_id": "episode_drop", "expected_drop_during_transport": True},
    ]
    _write_jsonl(dataset_dir / "episodes.jsonl", episodes)
    _write_jsonl(dataset_dir / "transitions.jsonl", transitions)
    diagnostic_pool_path = _write_jsonl(
        tmp_path / "diagnostic_pool.jsonl",
        diagnostic_pool_rows,
    )
    return dataset_dir, diagnostic_pool_path


def _build_happy_path_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_dir, diagnostic_pool_path = _build_dataset_fixture(tmp_path)
    output_dir = tmp_path / "reward_artifacts"
    build_drop_sidecar.materialize_drop_sidecar(dataset_dir, output_dir)
    eval_drop_detector_against_diagnostic_pool.materialize_drop_detector_eval(
        output_dir / build_drop_sidecar.DROP_SIDECAR_JSONL_NAME,
        diagnostic_pool_path,
        output_dir,
    )
    return dataset_dir, diagnostic_pool_path, output_dir


def test_counterfactual_happy_path_materializes_offline_artifacts_and_allows_gate(
    tmp_path: Path,
) -> None:
    dataset_dir, _diagnostic_pool_path, output_dir = _build_happy_path_artifacts(
        tmp_path
    )

    recommendation = relabel_counterfactual_rewards.materialize_counterfactual_rewards(
        dataset_dir,
        output_dir / build_drop_sidecar.DROP_SIDECAR_JSONL_NAME,
        output_dir,
        detector_eval_json=output_dir
        / eval_drop_detector_against_diagnostic_pool.DROP_DETECTOR_EVAL_JSON_NAME,
    )

    written_recommendation = _read_json(
        output_dir / relabel_counterfactual_rewards.REWARD_RECOMMENDATION_JSON_NAME
    )
    counterfactual_summary = _read_json(
        output_dir / relabel_counterfactual_rewards.COUNTERFACTUAL_SUMMARY_JSON_NAME
    )
    counterfactual_rows = _read_jsonl(
        output_dir / relabel_counterfactual_rewards.COUNTERFACTUAL_REWARDS_JSONL_NAME
    )
    report_text = (
        output_dir / relabel_counterfactual_rewards.REWARD_COUNTERFACTUAL_REPORT_MD_NAME
    ).read_text(encoding="utf-8")

    assert recommendation == written_recommendation
    assert (
        recommendation["reward_recommendation"]
        == drop_events.RECOMMENDATION_ELIGIBLE_FOR_MAINLINE
    )
    assert recommendation["formal_eligibility"] == "ALLOW"
    assert recommendation["mainline_reward_rerun_allowed"] is True
    assert "eligible_for_mainline" in report_text
    assert (
        counterfactual_summary["mainline_candidate_variant"]
        == drop_events.COUNTERFACTUAL_VARIANT_V1
    )
    assert (
        counterfactual_summary["variants"][drop_events.COUNTERFACTUAL_VARIANT_V1][
            "affected_episode_count"
        ]
        == 1
    )

    v1_rows = [
        row
        for row in counterfactual_rows
        if row["variant"] == drop_events.COUNTERFACTUAL_VARIANT_V1
        and row["episode_id"] == "episode_drop"
    ]
    row_t2 = next(row for row in v1_rows if row["t"] == 2)
    row_t3 = next(row for row in v1_rows if row["t"] == 3)
    assert row_t2["reward_counterfactual"] == -6.0
    assert row_t3["reward_counterfactual"] == 0.0

    rerun_gate = drop_events.build_mainline_reward_rerun_precondition_report(
        recommendation,
    )
    assert rerun_gate["formal_eligibility"] == "ALLOW"
    assert rerun_gate["failure_reasons"] == []


def test_ship_sidecar_only_blocks_mainline_reward_rerun(tmp_path: Path) -> None:
    dataset_dir, _diagnostic_pool_path, output_dir = _build_happy_path_artifacts(
        tmp_path
    )
    sidecar_summary = _read_json(
        output_dir / build_drop_sidecar.DROP_SIDECAR_SUMMARY_JSON_NAME
    )
    counterfactual_rows, counterfactual_summary = (
        drop_events.relabel_counterfactual_rewards(
            episodes=_read_jsonl(dataset_dir / "episodes.jsonl"),
            transitions=_read_jsonl(dataset_dir / "transitions.jsonl"),
            sidecar_rows=_read_jsonl(
                output_dir / build_drop_sidecar.DROP_SIDECAR_JSONL_NAME
            ),
        )
    )
    assert counterfactual_rows
    detector_eval_payload = {
        "schema_version": drop_events.DROP_DETECTOR_EVAL_SCHEMA_VERSION,
        "artifact_kind": drop_events.DROP_DETECTOR_EVAL_ARTIFACT_KIND,
        "status": "PASS",
        "formal_eligibility": "BLOCK",
        "reward_recommendation": drop_events.RECOMMENDATION_SHIP_SIDECAR_ONLY,
        "sidecar_publishable": True,
        "mainline_stable": False,
        "support_underflow": True,
        "evaluated_episode_count": 2,
        "diagnostic_pool_episode_count": 2,
        "positive_support": 1,
        "negative_support": 1,
        "confusion_matrix": {"tp": 1, "fp": 0, "tn": 1, "fn": 0},
        "precision": 1.0,
        "recall": 1.0,
        "accuracy": 1.0,
        "predicted_positive_episode_ids": ["episode_drop"],
        "expected_positive_episode_ids": ["episode_drop"],
        "missing_sidecar_episode_ids": [],
        "missing_detector_evidence_episode_ids": [],
        "threshold_policy": {
            "precision": 1.0,
            "recall": 1.0,
            "min_positive_support": 2,
            "min_negative_support": 1,
        },
        "failure_reasons": ["diagnostic_pool_support_underflow"],
    }
    detector_eval_path = _write_json(
        output_dir / "ship_only_detector_eval.json",
        detector_eval_payload,
    )
    evidence_files = {
        "drop_sidecar_jsonl": str(
            output_dir / build_drop_sidecar.DROP_SIDECAR_JSONL_NAME
        ),
        "drop_sidecar_summary_json": str(
            output_dir / build_drop_sidecar.DROP_SIDECAR_SUMMARY_JSON_NAME
        ),
        "drop_detector_eval_json": str(detector_eval_path),
        "counterfactual_rows_jsonl": str(
            output_dir / "counterfactual_rows_fixture.jsonl"
        ),
        "counterfactual_summary_json": str(
            output_dir / "counterfactual_summary_fixture.json"
        ),
        "reward_counterfactual_report_md": str(
            output_dir / "counterfactual_report_fixture.md"
        ),
    }
    _write_jsonl(Path(evidence_files["counterfactual_rows_jsonl"]), counterfactual_rows)
    _write_json(
        Path(evidence_files["counterfactual_summary_json"]), counterfactual_summary
    )
    Path(evidence_files["reward_counterfactual_report_md"]).write_text(
        "fixture\n", encoding="utf-8"
    )

    recommendation = drop_events.build_reward_recommendation(
        sidecar_summary=sidecar_summary,
        counterfactual_summary=counterfactual_summary,
        detector_eval=detector_eval_payload,
        evidence_paths=evidence_files,
    )

    assert (
        recommendation["reward_recommendation"]
        == drop_events.RECOMMENDATION_SHIP_SIDECAR_ONLY
    )
    gate = drop_events.build_mainline_reward_rerun_precondition_report(recommendation)
    assert gate["formal_eligibility"] == "BLOCK"
    assert "reward_recommendation_ship_sidecar_only" in gate["failure_reasons"]


def test_keep_offline_and_missing_evidence_fail_closed(tmp_path: Path) -> None:
    dataset_dir, _diagnostic_pool_path, output_dir = _build_happy_path_artifacts(
        tmp_path
    )
    sidecar_summary = _read_json(
        output_dir / build_drop_sidecar.DROP_SIDECAR_SUMMARY_JSON_NAME
    )
    _counterfactual_rows, counterfactual_summary = (
        drop_events.relabel_counterfactual_rewards(
            episodes=_read_jsonl(dataset_dir / "episodes.jsonl"),
            transitions=_read_jsonl(dataset_dir / "transitions.jsonl"),
            sidecar_rows=_read_jsonl(
                output_dir / build_drop_sidecar.DROP_SIDECAR_JSONL_NAME
            ),
        )
    )
    recommendation = drop_events.build_reward_recommendation(
        sidecar_summary=sidecar_summary,
        counterfactual_summary=counterfactual_summary,
        detector_eval=None,
        evidence_paths={
            "drop_sidecar_jsonl": str(
                output_dir / build_drop_sidecar.DROP_SIDECAR_JSONL_NAME
            ),
            "drop_sidecar_summary_json": str(
                output_dir / build_drop_sidecar.DROP_SIDECAR_SUMMARY_JSON_NAME
            ),
            "drop_detector_eval_json": None,
            "counterfactual_rows_jsonl": str(output_dir / "missing_rows.jsonl"),
            "counterfactual_summary_json": str(output_dir / "missing_summary.json"),
            "reward_counterfactual_report_md": str(output_dir / "missing_report.md"),
        },
    )

    assert (
        recommendation["reward_recommendation"]
        == drop_events.RECOMMENDATION_KEEP_OFFLINE
    )
    keep_offline_gate = drop_events.build_mainline_reward_rerun_precondition_report(
        recommendation,
    )
    assert keep_offline_gate["formal_eligibility"] == "BLOCK"
    assert "reward_recommendation_keep_offline" in keep_offline_gate["failure_reasons"]
    assert (
        "missing_evidence_path:drop_detector_eval_json"
        in keep_offline_gate["failure_reasons"]
    )
    assert (
        "evidence_path_not_found:counterfactual_rows_jsonl"
        in keep_offline_gate["failure_reasons"]
    )
