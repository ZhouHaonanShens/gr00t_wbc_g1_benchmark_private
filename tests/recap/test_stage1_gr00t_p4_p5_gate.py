from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_rollout35a():
    path = REPO_ROOT / "work/recap/scripts/35a_full_update_rollout_probe.py"
    spec = importlib.util.spec_from_file_location(
        "stage1_gr00t_p4_p5_rollout35a",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROLLOUT35A = _load_rollout35a()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _eval_summary(*, seeds: list[int], label: str) -> dict[str, Any]:
    return {
        "episodes": len(seeds),
        "success_count": 0,
        "success_rate": 0.0,
        "seed_base": seeds[0],
        "episode_results": [
            {"seed": seed, "success": False, "episode_index": index}
            for index, seed in enumerate(seeds)
        ],
        "server_provenance": {
            "policy_model_path": f"agent/artifacts/mock/{label}/checkpoint-200",
        },
    }


def _baseline_authority(tmp_path: Path, *, seeds: list[int]) -> Path:
    baseline_root = tmp_path / "agent/artifacts/recap_min_loop/single_gpu_v1"
    _write_json(
        baseline_root / "eval_seed_set.json",
        {
            "schema_version": "recap_eval_seed_set_v1",
            "formal_eval_episodes": len(seeds),
            "seed_base": seeds[0],
            "episode_indices": list(range(len(seeds))),
            "seeds": seeds,
        },
    )
    _write_json(
        baseline_root / "t5_baseline_formal_eval/eval_summary.json",
        _eval_summary(seeds=seeds, label="baseline"),
    )
    return baseline_root


def _stage1_gate_verdict(
    stage1_gr00t_root: Path,
    *,
    status: str = "PASS",
    formal_claim_allowed: bool = True,
    blocking_reasons: list[str] | None = None,
    p5_formal_10ep_eligible: bool = True,
    comparability_manifest: dict[str, Any] | None = None,
) -> Path:
    manifest_name = "comparability_manifest.json"
    if comparability_manifest is not None:
        _write_json(stage1_gr00t_root / "p4_refresh" / manifest_name, comparability_manifest)
    return _write_json(
        stage1_gr00t_root / "p4_refresh" / "p4_gate_verdict.json",
        {
            "schema_version": "gr00t_p4_gate_verdict_v1",
            "status": status,
            "formal_claim_allowed": formal_claim_allowed,
            "blocking_reasons": list(blocking_reasons or []),
            "p5_formal_10ep_eligible": p5_formal_10ep_eligible,
            "comparability_manifest": manifest_name,
            "full_update_diagnostic_summary": "full_update_diagnostic_summary.json",
        },
    )


def _pass_comparability_manifest() -> dict[str, Any]:
    return {
        "schema_version": "gr00t_p4_comparability_manifest_v1",
        "status": "PASS",
        "paired_seed_total": 3,
        "paired_seed_improvement_count": 2,
        "baseline_config_hash": "sha256:" + "a" * 64,
        "candidate_config_hash": "sha256:" + "b" * 64,
        "eval_condition_hash": "sha256:" + "c" * 64,
        "blocking_reasons": [],
    }


def test_stage1_missing_p4_writes_blocked_execution_decision_without_lane_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seeds = list(range(20260421, 20260431))
    baseline_root = _baseline_authority(tmp_path, seeds=seeds)
    stage1_gr00t_root = tmp_path / "agent/artifacts/stage1/gr00t"
    output_dir = stage1_gr00t_root / "p5_gate"

    def _unexpected_lane_state(*args: object, **kwargs: object) -> None:
        raise AssertionError("missing P4 gate must block before formal lane lookup")

    monkeypatch.setattr(ROLLOUT35A, "_resolve_lane_state", _unexpected_lane_state)

    result = ROLLOUT35A.run_p5_gate(
        baseline_authority_root=baseline_root,
        v2_authority_root=stage1_gr00t_root,
        conditioned_run_root=stage1_gr00t_root / "conditioned",
        continuation_run_root=stage1_gr00t_root / "continuation",
        output_dir=output_dir,
        seed_start=20260421,
        seed_end=20260430,
    )

    decision = json.loads(
        Path(result["p5_execution_decision_path"]).read_text(encoding="utf-8")
    )
    assert decision["schema_version"] == "gr00t_p5_execution_decision_v1"
    assert decision["decision"] == "BLOCKED"
    assert decision["p4_gate_verdict"].endswith("p4_refresh/p4_gate_verdict.json")
    assert decision["gate_inputs"] == {
        "status": None,
        "formal_claim_allowed": None,
        "blocking_reasons": [],
        "p5_formal_10ep_eligible": None,
    }
    assert decision["blocking_reasons"] == ["missing_p4_gate_summary"]
    assert Path(result["blocker_summary_path"]).is_file()


def test_stage1_clean_flags_still_block_without_comparability_manifest_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seeds = list(range(20260421, 20260431))
    baseline_root = _baseline_authority(tmp_path, seeds=seeds)
    stage1_gr00t_root = tmp_path / "agent/artifacts/stage1/gr00t"
    _stage1_gate_verdict(
        stage1_gr00t_root,
        comparability_manifest={
            **_pass_comparability_manifest(),
            "status": "BLOCK",
            "paired_seed_improvement_count": 1,
            "blocking_reasons": ["paired_seed_improvement_count_below_2_of_3"],
        },
    )

    def _unexpected_lane_state(*args: object, **kwargs: object) -> None:
        raise AssertionError("comparability BLOCK must skip before lane lookup")

    monkeypatch.setattr(ROLLOUT35A, "_resolve_lane_state", _unexpected_lane_state)

    result = ROLLOUT35A.run_p5_gate(
        baseline_authority_root=baseline_root,
        v2_authority_root=stage1_gr00t_root,
        conditioned_run_root=stage1_gr00t_root / "conditioned",
        continuation_run_root=stage1_gr00t_root / "continuation",
        output_dir=stage1_gr00t_root / "p5_gate",
        seed_start=20260421,
        seed_end=20260430,
    )

    verdict = json.loads(Path(result["min_loop_verdict_path"]).read_text(encoding="utf-8"))
    decision = json.loads(
        Path(result["p5_execution_decision_path"]).read_text(encoding="utf-8")
    )
    assert verdict["status"] == "SKIPPED"
    assert verdict["formal_execution_attempted"] is False
    assert decision["decision"] == "BLOCKED"
    assert decision["gate_inputs"] == {
        "status": "PASS",
        "formal_claim_allowed": True,
        "blocking_reasons": [],
        "p5_formal_10ep_eligible": True,
    }
    assert "comparability_manifest_status_block" in decision["blocking_reasons"]
    assert (
        "comparability_manifest_paired_seed_improvement_count_below_2"
        in decision["blocking_reasons"]
    )
    assert "paired_seed_improvement_count_below_2_of_3" in decision["blocking_reasons"]


def test_stage1_clean_p4_writes_run_decision_and_blocker_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seeds = list(range(20260421, 20260431))
    baseline_root = _baseline_authority(tmp_path, seeds=seeds)
    stage1_gr00t_root = tmp_path / "agent/artifacts/stage1/gr00t"
    _stage1_gate_verdict(
        stage1_gr00t_root,
        comparability_manifest=_pass_comparability_manifest(),
    )

    def _checkpoint_lane_state(*args: object, **kwargs: object) -> dict[str, Any]:
        return {
            "resolution": "checkpoint",
            "checkpoint_path": str(stage1_gr00t_root / "mock-checkpoint"),
            "checkpoint_asset_path": str(stage1_gr00t_root / "mock-checkpoint/model.safetensors"),
        }

    def _fake_run_p5_eval_lane(
        *,
        lane_name: str,
        lane_state: dict[str, object],
        run_root: Path,
        output_dir: Path,
        requested_seed_set: list[int],
    ) -> dict[str, object]:
        del lane_state, run_root
        output_path = ROLLOUT35A._copy_eval_summary(
            output_dir=output_dir,
            lane_name=lane_name,
            payload=_eval_summary(seeds=list(requested_seed_set), label=lane_name),
        )
        return {
            "lane_name": lane_name,
            "status": "PASS",
            "source_summary_path": None,
            "output_summary_path": ROLLOUT35A._safe_relpath(output_path),
            "checkpoint_path": None,
            "checkpoint_asset_path": None,
            "success_count": 0,
            "success_rate": 0.0,
            "episodes": len(requested_seed_set),
            "seed_base": requested_seed_set[0],
            "episode_seeds": list(requested_seed_set),
        }

    monkeypatch.setattr(ROLLOUT35A, "_resolve_lane_state", _checkpoint_lane_state)
    monkeypatch.setattr(ROLLOUT35A, "_run_p5_eval_lane", _fake_run_p5_eval_lane)

    result = ROLLOUT35A.run_p5_gate(
        baseline_authority_root=baseline_root,
        v2_authority_root=stage1_gr00t_root,
        conditioned_run_root=stage1_gr00t_root / "conditioned",
        continuation_run_root=stage1_gr00t_root / "continuation",
        output_dir=stage1_gr00t_root / "p5_gate",
        seed_start=20260421,
        seed_end=20260430,
    )

    verdict = json.loads(Path(result["min_loop_verdict_path"]).read_text(encoding="utf-8"))
    decision = json.loads(
        Path(result["p5_execution_decision_path"]).read_text(encoding="utf-8")
    )
    blocker_summary = json.loads(
        Path(result["blocker_summary_path"]).read_text(encoding="utf-8")
    )
    assert verdict["status"] == "PASS"
    assert verdict["gate_mode"] == "executed"
    assert decision["decision"] == "RUN"
    assert decision["blocking_reasons"] == []
    assert decision["gate_inputs"] == {
        "status": "PASS",
        "formal_claim_allowed": True,
        "blocking_reasons": [],
        "p5_formal_10ep_eligible": True,
    }
    assert decision["min_loop_verdict"].endswith("p5_gate/min_loop_verdict.json")
    assert decision["p5_gate_blocker_summary"].endswith(
        "p5_gate/p5_gate_blocker_summary.json"
    )
    assert blocker_summary["artifact_kind"] == "full_update_p5_gate_blocker_summary"
    assert blocker_summary["blocking_reasons"] == []
