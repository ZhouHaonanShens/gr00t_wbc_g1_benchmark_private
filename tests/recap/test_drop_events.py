from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import drop_events
from work.recap.scripts import build_drop_sidecar
from work.recap.scripts import eval_drop_detector_against_diagnostic_pool


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


def test_validate_drop_event_row_fail_closed_on_missing_required_field() -> None:
    with pytest.raises((TypeError, ValueError)):
        drop_events.validate_drop_event_row(
            {
                "schema_version": drop_events.DROP_EVENT_ROW_SCHEMA_VERSION,
                "artifact_kind": drop_events.DROP_EVENT_ROW_ARTIFACT_KIND,
                "episode_id": "ep_001",
                "t": 0,
                "success_step": False,
                "inner_reward_count": 1,
                "episode_return_online": -1.0,
                "success_episode": False,
                "phase": "TRANSPORT",
                "transport_context": True,
                "detector_evidence_available": True,
                "detector_signal_source": drop_events.DROP_SIGNAL_SOURCE_DIRECT,
                "direct_drop_flag": True,
                "had_in_hand_previously": True,
                "apple_in_hand": False,
                "drop_detected": True,
                "drop_during_transport": True,
            }
        )


def test_build_drop_sidecar_and_detector_eval_happy_path(tmp_path: Path) -> None:
    dataset_dir, diagnostic_pool_path = _build_dataset_fixture(tmp_path)
    output_dir = tmp_path / "reward_artifacts"

    sidecar_result = build_drop_sidecar.materialize_drop_sidecar(
        dataset_dir, output_dir
    )

    assert sidecar_result["status"] == "PASS"
    sidecar_rows = _read_jsonl(output_dir / build_drop_sidecar.DROP_SIDECAR_JSONL_NAME)
    assert len(sidecar_rows) == 6
    drop_rows = [row for row in sidecar_rows if row["drop_during_transport"] is True]
    assert len(drop_rows) == 1
    assert drop_rows[0]["episode_id"] == "episode_drop"
    assert drop_rows[0]["t"] == 2
    assert (
        drop_rows[0]["detector_signal_source"]
        == drop_events.DROP_SIGNAL_SOURCE_IN_HAND_TRANSITION
    )

    summary = _read_json(output_dir / build_drop_sidecar.DROP_SIDECAR_SUMMARY_JSON_NAME)
    assert summary["coverage_ratio"] == 1.0
    assert summary["episodes_with_transport_drop"] == 1
    assert summary["rows_with_missing_detector_evidence"] == 0
    assert summary["first_transport_drop_episode_ids"] == ["episode_drop"]

    eval_payload = (
        eval_drop_detector_against_diagnostic_pool.materialize_drop_detector_eval(
            output_dir / build_drop_sidecar.DROP_SIDECAR_JSONL_NAME,
            diagnostic_pool_path,
            output_dir,
        )
    )
    written_eval = _read_json(
        output_dir
        / eval_drop_detector_against_diagnostic_pool.DROP_DETECTOR_EVAL_JSON_NAME
    )

    assert eval_payload == written_eval
    assert eval_payload["status"] == "PASS"
    assert (
        eval_payload["reward_recommendation"]
        == drop_events.RECOMMENDATION_ELIGIBLE_FOR_MAINLINE
    )
    assert eval_payload["sidecar_publishable"] is True
    assert eval_payload["mainline_stable"] is True
    assert eval_payload["support_underflow"] is False
    assert eval_payload["precision"] == 1.0
    assert eval_payload["recall"] == 1.0
    assert eval_payload["confusion_matrix"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
