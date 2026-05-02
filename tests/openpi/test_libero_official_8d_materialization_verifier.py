from __future__ import annotations

import json
from pathlib import Path

from work.openpi.scripts.verify_libero_official_8d_recap_relabels import (
    ROUTE_ID,
    build_report,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _dataset(root: Path, *, route_id: str = ROUTE_ID, status: str = "materialized") -> Path:
    _write_json(
        root / "materialization_report.json",
        {
            "route_id": route_id,
            "final_status": status,
            "selected_episode_count": 2,
            "selected_frame_count": 5,
            "required_output_label_columns": [
                "recap_m2.t",
                "recap_m2.return_G",
                "recap_m2.value_V",
                "recap_m2.advantage_A",
                "recap_m2.advantage_input",
                "recap_m2.epsilon_l",
                "recap_m2.indicator_I",
                "recap_m2.prompt_raw",
                "recap_m2.prompt_conditioned",
            ],
        },
    )
    _write_json(
        root / "meta/info.json",
        {
            "route_id": route_id,
            "features": {
                "observation.state": {"shape": [8]},
                "action": {"shape": [7]},
            },
        },
    )
    _write_json(
        root / "meta/relabel_stats_report.json",
        {
            "route_id": route_id,
            "final_status": "ready",
            "episode_count": 2,
            "frame_count": 5,
            "indicator_positive_count": 2,
            "indicator_negative_count": 3,
            "dataset_fingerprint_status": "current",
            "dataset_fingerprint_sha256": "abc123",
        },
    )
    _write_json(root / "meta/dataset_fingerprint.json", {"episode_count": 2, "frame_count": 5})
    return root


def test_metadata_verifier_accepts_ready_official_relabels(tmp_path: Path) -> None:
    report = build_report(_dataset(tmp_path / "dataset"))

    assert report["route_id"] == ROUTE_ID
    assert report["final_status"] == "ready"
    assert report["blockers"] == []
    assert report["state_shape"] == [8]
    assert report["action_shape"] == [7]
    assert report["frame_count"] == 5


def test_metadata_verifier_blocks_route_or_shape_drift(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset", route_id="wrong_route")
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.state"]["shape"] = [43]
    info_path.write_text(json.dumps(info), encoding="utf-8")

    report = build_report(dataset)

    assert report["final_status"] == "blocked"
    assert "materialization_route_matches" in report["blockers"]
    assert "state_shape_is_8d" in report["blockers"]
