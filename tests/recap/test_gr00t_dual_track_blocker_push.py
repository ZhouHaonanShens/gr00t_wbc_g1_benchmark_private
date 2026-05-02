from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = REPO_ROOT / "work/recap/scripts/36_gr00t_dual_track_blocker_push.py"
    spec = importlib.util.spec_from_file_location("gr00t_dual_track_blocker_push", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_formal_status_blocks_without_compound_status_and_never_enters_p5(tmp_path: Path) -> None:
    lane_root = tmp_path / "lane"
    summary = tmp_path / "full_update_diagnostic_summary.json"
    _write_json(
        summary,
        {
            "status": "BLOCK",
            "p5_formal_10ep_eligible": False,
            "blocking_reasons": [
                "paired_seed_improvement_count_below_2_of_3",
                "missing_comparability_manifest",
            ],
        },
    )

    formal = MODULE._formal_status(
        lane_root=lane_root,
        p4_summary_path=summary,
        manifest_repair={"validation": {"status": "pass"}},
    )

    assert formal["schema_version"] == "dual_track_formal_status_v1"
    assert formal["track"] == "formal"
    assert formal["status"] == "BLOCK"
    assert "(" not in formal["status"] and ")" not in formal["status"]
    assert formal["formal_claim_allowed"] is False
    assert formal["next_gate_allowed"] is False
    assert formal["entered_next_gate"] is False
    assert formal["blocking_reasons"] == [
        "paired_seed_improvement_count_below_2_of_3",
        "missing_comparability_manifest",
    ]


def test_clean_p4_requires_gated_p5_verdict_before_formal_pass(tmp_path: Path) -> None:
    lane_root = tmp_path / "lane"
    summary = tmp_path / "full_update_diagnostic_summary.json"
    _write_json(
        summary,
        {
            "status": "PASS",
            "formal_claim_allowed": True,
            "p5_formal_10ep_eligible": True,
            "blocking_reasons": [],
        },
    )

    formal = MODULE._formal_status(
        lane_root=lane_root,
        p4_summary_path=summary,
        manifest_repair={"validation": {"status": "pass"}},
    )

    assert formal["status"] == "BLOCK"
    assert formal["formal_claim_allowed"] is False
    assert formal["entered_next_gate"] is False
    assert formal["next_gate_allowed"] is False
    assert formal["blocking_reasons"] == ["p5_gate_verdict_missing"]


def test_gated_p5_pass_allows_formal_pass_and_records_entry(tmp_path: Path) -> None:
    lane_root = tmp_path / "lane"
    summary = tmp_path / "full_update_diagnostic_summary.json"
    p5_verdict = tmp_path / "p5_gate" / "min_loop_verdict.json"
    _write_json(
        summary,
        {
            "status": "PASS",
            "formal_claim_allowed": True,
            "p5_formal_10ep_eligible": True,
            "blocking_reasons": [],
        },
    )
    _write_json(
        p5_verdict,
        {
            "status": "PASS",
            "gate_mode": "executed",
            "formal_execution_attempted": True,
            "blocking_reasons": [],
        },
    )

    formal = MODULE._formal_status(
        lane_root=lane_root,
        p4_summary_path=summary,
        manifest_repair={"validation": {"status": "pass"}},
        p5_verdict_path=p5_verdict,
        p5_refresh={"returncode": 0, "lease_path": "resource_lease_p5.json"},
    )

    assert formal["status"] == "PASS"
    assert formal["formal_claim_allowed"] is True
    assert formal["entered_next_gate"] is True
    assert formal["next_gate_allowed"] is True
    assert formal["p5_status"] == "PASS"
    assert str(p5_verdict) in formal["authority_inputs"]


def test_gpu1_runner_writes_timeout_resource_lease(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.log"
    lease_path = tmp_path / "resource_lease.json"

    result = MODULE._run(
        [sys.executable, "-c", "print('gpu1 runner ok')"],
        log_path=log_path,
        lease_path=lease_path,
        env={"CUDA_VISIBLE_DEVICES": "1"},
        timeout_seconds=5,
        artifacts=[tmp_path / "artifact.json"],
    )

    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    log_text = log_path.read_text(encoding="utf-8")
    assert result["returncode"] == 0
    assert lease["schema_version"] == "resource_lease_v1"
    assert lease["lane"] == "gr00t"
    assert lease["gpu"] == 1
    assert lease["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert lease["forbidden_gpus_visible"] is False
    assert lease["sudo_used"] is False
    assert lease["timeout_seconds"] == 5
    assert "timeout_seconds=5" in log_text
    assert "CUDA_VISIBLE_DEVICES=1" in log_text


def test_exploratory_positive_seed_signal_cannot_unlock_formal(tmp_path: Path) -> None:
    subgoal = tmp_path / "subgoal_summary_3seed.json"
    _write_json(
        subgoal,
        {
            "selected_seeds": [20260421, 20260422, 20260423],
            "paired_seed_improvement_count": 1,
            "mean_relative_improvement_min_dist_ee_to_apple": -0.1,
            "no_regression_on_contact_or_lift_proxy": False,
            "per_seed_pairs": [
                {"seed": 20260421, "relative_improvement_min_dist_ee_to_apple": -0.2},
                {"seed": 20260422, "relative_improvement_min_dist_ee_to_apple": 0.3},
            ],
        },
    )

    exploratory = MODULE._exploratory_signal(subgoal_path=subgoal)

    assert exploratory["schema_version"] == "dual_track_exploratory_signal_v1"
    assert exploratory["track"] == "exploratory"
    assert exploratory["status"] == "SIGNAL"
    assert exploratory["exploratory_only"] is True
    assert exploratory["formal_claim_allowed"] is False
    assert exploratory["must_not_unlock_formal_gate"] is True
    assert exploratory["risk_label"] == "exploratory_not_formal"
    assert exploratory["observed_signal"]["positive_seed_count"] == 1
    assert exploratory["observed_signal"]["required_formal_improvement_count"] == 2
