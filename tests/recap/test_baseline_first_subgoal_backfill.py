from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BACKFILL = _load_module(
    "baseline_first_subgoal_backfill_for_tests",
    "work/recap/scripts/backfill_baseline_first_subgoal_probe.py",
)
ROLLOUT35A = _load_module(
    "baseline_first_subgoal_rollout_probe_for_tests",
    "work/recap/scripts/35a_full_update_rollout_probe.py",
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_episode_jsonl(baseline_root: Path, rows: list[dict[str, object]]) -> None:
    telemetry_path = baseline_root / "telemetry" / "eval_summary_episodes.jsonl"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _episode_row(seed: int, *, min_dist: float, near_apple: bool, max_lift: float) -> dict[str, object]:
    return {
        "seed": seed,
        "failure_stage_guess": {
            "min_apple_to_right_eef_l2": min_dist,
            "ever_near_apple": near_apple,
            "max_apple_lift_z": max_lift,
        },
    }


def test_missing_seed_fail_closed(tmp_path: Path) -> None:
    baseline_root = tmp_path / "single_gpu_v1" / "t5_baseline_formal_eval"
    _write_episode_jsonl(
        baseline_root,
        [
            _episode_row(20260421, min_dist=0.33, near_apple=False, max_lift=0.0),
            _episode_row(20260422, min_dist=0.32, near_apple=True, max_lift=0.01),
        ],
    )
    output_path = tmp_path / "single_gpu_v2_full_update" / "p5_gate_eval" / "baseline_first_subgoal_probe_v1.json"

    result = BACKFILL.run_backfill(
        baseline_root=baseline_root,
        output_path=output_path,
        seed_start=20260421,
        seed_end=20260430,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "BLOCK"
    assert payload["probe_eligible"] is False
    assert payload["blocking_reasons"] == ["baseline_v1_seed_coverage_insufficient"]
    assert payload["missing_required_seeds"] == [20260423]
    assert payload["required_seed_metrics_present"] is False
    assert payload["complete_3seed_subgoal_probe"] is False


def test_baseline_override_hit(tmp_path: Path) -> None:
    baseline_authority_root = tmp_path / "single_gpu_v1"
    _write_json(
        baseline_authority_root / "first_subgoal_probe.json",
        {"status": "PASS", "seed_metrics": []},
    )
    override_path = tmp_path / "single_gpu_v2_full_update" / "p5_gate_eval" / "baseline_first_subgoal_probe_v1.json"
    _write_json(
        override_path,
        {"status": "PASS", "seed_metrics": [{"seed": 20260421, "min_dist_ee_to_apple": 0.3, "contact_or_lift_proxy": 0.0}]},
    )

    resolved = ROLLOUT35A._resolve_lane_subgoal_probe_path(
        lane_name="baseline_v1",
        baseline_authority_root=baseline_authority_root,
        run_root=None,
        lane_state=None,
        baseline_v1_subgoal_override=override_path,
    )

    assert resolved == override_path


def test_default_resolver_stability_without_override(tmp_path: Path) -> None:
    baseline_authority_root = tmp_path / "single_gpu_v1"
    expected_path = baseline_authority_root / "first_subgoal_probe_3seed.json"
    _write_json(
        expected_path,
        {"status": "PASS", "seed_metrics": [{"seed": 20260421, "min_dist_ee_to_apple": 0.3, "contact_or_lift_proxy": 0.0}]},
    )

    resolved = ROLLOUT35A._resolve_lane_subgoal_probe_path(
        lane_name="baseline_v1",
        baseline_authority_root=baseline_authority_root,
        run_root=None,
        lane_state=None,
    )

    assert resolved == expected_path
