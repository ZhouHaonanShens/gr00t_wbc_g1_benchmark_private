from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from work.openpi.scripts import build_openpi_runtime_dataset_provenance as provenance


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _dataset(root: Path) -> Path:
    _write_json(
        root / "materialization_report.json",
        {
            "route_id": "official_native_8d_recap_relabels_v1",
            "final_status": "materialized",
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
            "route_id": "official_native_8d_recap_relabels_v1",
            "features": {
                "observation.state": {"shape": [8]},
                "action": {"shape": [7]},
            },
        },
    )
    _write_json(
        root / "meta/stats.json",
        {
            "observation.state": {
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
                "min": [-1.0] * 8,
                "max": [1.0] * 8,
            },
            "action": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "min": [-1.0] * 7,
                "max": [1.0] * 7,
            },
        },
    )
    _write_json(
        root / "meta/relabel_stats_report.json",
        {
            "route_id": "official_native_8d_recap_relabels_v1",
            "final_status": "ready",
            "episode_count": 2,
            "frame_count": 5,
            "indicator_positive_count": 2,
            "indicator_negative_count": 3,
            "dataset_fingerprint_status": "current",
            "dataset_fingerprint_sha256": "abc123",
        },
    )
    _write_json(
        root / "meta/dataset_fingerprint.json",
        {
            "episode_count": 2,
            "frame_count": 5,
            "fingerprint_sha256": "abc123",
            "parquet_inventory_hash": "parquet123",
            "episode_universe_hash": "episodes123",
        },
    )
    return root


def test_runtime_dataset_provenance_records_norm_and_p0_readiness(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_dir = _dataset(
        tmp_path / "physical_intelligence_libero_official_8d_recap_relabels_v1"
    )

    def _p0_stub(_: Path) -> dict[str, Any]:
        return {
            "schema_version": "openpi_p0_runtime_loader_smoke_v1",
            "status": "PASS",
            "runtime_level": "p0_loader_runtime_pass",
            "blockers": [],
            "formal_claim_allowed": False,
        }

    monkeypatch.setattr(provenance, "build_p0_loader_smoke", _p0_stub)

    report = provenance.build_runtime_dataset_provenance(dataset_dir)

    assert report["schema_version"] == "openpi_runtime_dataset_provenance_v1"
    assert report["formal_dataset"] is True
    assert report["fingerprint"]["fingerprint_sha256"] == "abc123"
    assert report["norm_stats"]["final_status"] == "ready"
    assert report["norm_stats"]["state_dim"] == 8
    assert report["norm_stats"]["action_dim"] == 7
    assert report["runtime_blocker_triage"] == {
        "data_side_status": "ready",
        "data_side_blockers": [],
        "runtime_level": "p0_loader_runtime_pass",
        "formal_claim_allowed": False,
        "pending_gpu2_runtime_evidence": [
            "p1_one_step_probe_gpu2",
            "p2_tiny_update_or_overfit20_gpu2",
        ],
        "blocked_reason_if_no_worker3_runtime": "p1_one_step_runtime_evidence_pending",
    }


def test_norm_stats_provenance_blocks_shape_drift(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path / "dataset")
    stats_path = dataset_dir / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["action"]["std"] = [1.0] * 6
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    report = provenance.build_norm_stats_provenance(dataset_dir)

    assert report["final_status"] == "blocked"
    assert "action_stats_shape_matches" in report["blockers"]
