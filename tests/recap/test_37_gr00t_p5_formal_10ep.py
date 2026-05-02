from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_p5_module() -> Any:
    path = REPO_ROOT / "work/recap/scripts/37_gr00t_p5_formal_10ep.py"
    spec = importlib.util.spec_from_file_location("gr00t_p5_formal_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P5 = _load_p5_module()


def test_build_seed_metrics_from_step_telemetry() -> None:
    step_rows = [
        {
            "seed": 20260421,
            "episode_index": 1,
            "apple_to_right_eef_l2": 0.30,
            "apple_height_z": 0.10,
        },
        {
            "seed": 20260421,
            "episode_index": 1,
            "apple_to_right_eef_l2": 0.08,
            "apple_height_z": 0.135,
        },
    ]
    episode_rows = [{"seed": 20260421, "episode_index": 1, "success": False}]

    metrics = P5.build_seed_metrics(step_rows=step_rows, episode_rows=episode_rows)

    assert metrics[20260421]["min_dist_ee_to_apple"] == 0.08
    assert metrics[20260421]["contact_proxy"] == 1.0
    assert round(metrics[20260421]["lift_proxy"], 6) == 0.035
    assert metrics[20260421]["step_count"] == 2


def test_summary_validator_blocks_below_threshold(tmp_path: Path) -> None:
    records = [
        {
            "seed": 20260421,
            "success": False,
            "relative_improvement_min_dist_ee_to_apple": 0.25,
            "distance_improved": True,
            "p5_metric_present": True,
            "baseline_metric_present": True,
        }
    ]

    summary, validator = P5.build_summary_and_validator(
        selected_seeds=[20260421],
        per_episode_records=records,
        eval_summary={"episodes": 1},
        p4_verdict={"mean_relative_improvement_min_dist_ee_to_apple": 0.537},
        p4_blockers=[],
        gpu_boundary_ok=True,
        gpu0_or_gpu3_touched=False,
        threshold=0.5,
        output_root=tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/run/gr00t/p5",
        pointer_root=tmp_path / "agent/artifacts/run/gr00t/p5",
    )

    assert summary["status"] == "BLOCK"
    assert validator["p5_validator_status"] == "BLOCK_with_specific_reasons"
    assert "mean_relative_improvement_below_threshold" in validator["blocking_reasons"]


def test_p5_episode_records_pair_baseline_and_candidate_metrics() -> None:
    records = P5.build_p5_episode_records(
        selected_seeds=[20260421],
        p5_metrics_by_seed={
            20260421: {
                "episode_index": 1,
                "success": False,
                "min_dist_ee_to_apple": 0.05,
                "contact_proxy": 1.0,
                "lift_proxy": 0.02,
                "contact_or_lift_proxy": 1.0,
                "step_count": 72,
            }
        },
        baseline_metrics_by_seed={
            20260421: {
                "episode_index": 1,
                "success": False,
                "min_dist_ee_to_apple": 0.25,
            }
        },
    )

    assert records[0]["relative_improvement_min_dist_ee_to_apple"] == 0.8
    assert records[0]["distance_improved"] is True
