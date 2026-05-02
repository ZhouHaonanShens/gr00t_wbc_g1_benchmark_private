from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any, Protocol, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import stage3_collect_checkpoint_binding


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return dict(payload)


def _make_repo_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    (repo_root / "agent").mkdir(parents=True, exist_ok=True)
    (repo_root / ".sisyphus" / "evidence").mkdir(parents=True, exist_ok=True)
    (repo_root / "AGENTS.md").write_text("fixture\n", encoding="utf-8")
    manifest_path = _write_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/iteration_manifest.json",
        {
            "schema_version": "stage3_iteration_manifest_v3",
            "active_plan_id": "stage3-iter-002-bootstrap",
            "artifact_root": "agent/artifacts/stage3_iteration/recap_stage3_iter_002/",
            "collect_policy_ckpt_decision_enum": [
                "historical_best",
                "manual_pinned",
                "baseline_train_required",
                "baseline_trained",
                "iteration_hard_block",
            ],
            "critic_tag": "g1_recap_stage3_iter_002_qwen3vl2b",
            "delegate_runtime_python": sys.executable,
            "formal_eval_episodes": 10,
            "formal_eval_seed_base": 20000,
            "formal_iter_tag": "recap_stage3_iter_002",
            "historical_success_rate_threshold": 0.3,
            "mainline_text_authority": "carrier_text_v1",
            "n_action_steps": 20,
            "official_task_anchor_model": stage3_collect_checkpoint_binding.OFFICIAL_TASK_ANCHOR_MODEL,
            "orchestrator_python": sys.executable,
            "policy_horizon": 30,
            "success_gate_threshold_count": 3,
            "train_iter_tag": "recap_stage3_iter_002_train",
        },
    )
    return repo_root, manifest_path


def _checkpoint_index(
    repo_root: Path, checkpoint_rel: str, *, with_adv_pair: bool
) -> Path:
    checkpoint_dir = repo_root / checkpoint_rel
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {
        "action_head.output_proj.weight": "model-00001-of-00001.safetensors",
    }
    if with_adv_pair:
        weight_map.update(
            {
                "action_head.advantage_embedding.weight": "model-00001-of-00001.safetensors",
                "action_head.advantage_embedding.bias": "model-00001-of-00001.safetensors",
            }
        )
    _write_json(
        checkpoint_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 1},
            "weight_map": weight_map,
        },
    )
    return checkpoint_dir


def _write_official_anchor_artifacts(repo_root: Path) -> None:
    _write_json(
        repo_root
        / "agent/artifacts/gr00t_anchor_controller_recap/baseline_freeze/baseline_freeze_matrix.json",
        {
            "schema_version": "gr00t_baseline_freeze_matrix_v1",
            "artifact_kind": "gr00t_baseline_freeze_matrix",
            "baselines": {
                "g1_b0_public_anchor": {
                    "summary": {
                        "public_anchor_success_rate": 0.5,
                    }
                }
            },
        },
    )
    _write_json(
        repo_root / ".sisyphus/evidence/task-4-public-anchor.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_4_public_anchor_evidence",
            "verification": {
                "success_run": {
                    "formal_success_rate": 0.5,
                    "formal_success_count": 5,
                }
            },
        },
    )
    _write_json(
        repo_root / ".sisyphus/evidence/task-10-b0-suite.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_10_b0_suite_evidence",
            "baseline_suite": {
                "official_comparable_10ep": {
                    "success_rate": 0.5,
                    "success_count": 5,
                }
            },
        },
    )


class _WrapperModule(Protocol):
    REPO_ROOT: Path

    def main(self, argv: list[str] | None = None) -> int: ...


def _load_wrapper_module() -> _WrapperModule:
    script_path = REPO_ROOT / "work/recap/scripts/30a_stage3_bind_collect_checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "stage3_bind_collect_checkpoint_wrapper",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_WrapperModule, cast(object, module))


def test_bind_collect_checkpoint_prefers_manual_pin_over_historical_candidate(
    tmp_path: Path,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    _write_official_anchor_artifacts(repo_root)
    manual_checkpoint_dir = _checkpoint_index(
        repo_root,
        "agent/artifacts/checkpoints/manual/checkpoint-8",
        with_adv_pair=True,
    )
    historical_checkpoint_dir = _checkpoint_index(
        repo_root,
        "agent/artifacts/checkpoints/historical/checkpoint-100",
        with_adv_pair=True,
    )

    result = stage3_collect_checkpoint_binding.bind_collect_checkpoint(
        repo_root=repo_root,
        manifest_path=manifest_path,
        manual_checkpoint_path=manual_checkpoint_dir,
        historical_candidates=[
            stage3_collect_checkpoint_binding.build_local_candidate_spec(
                candidate_id="historical_candidate_a",
                checkpoint_path=historical_checkpoint_dir,
                success_rate=0.8,
                success_count=8,
            )
        ],
    )

    manifest = _read_json(manifest_path)
    checkpoint_gate = _read_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/checkpoint_provenance_gate.json"
    )
    run_manifest_gate = _read_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/run_manifest_gate.json"
    )
    contract_gate = _read_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/contract_precondition_gate.json"
    )

    assert result["collect_policy_ckpt_decision"] == "manual_pinned"
    assert manifest["schema_version"] == "stage3_iteration_manifest_v3"
    assert manifest["collect_policy_ckpt_decision"] == "manual_pinned"
    assert manifest["collect_policy_ckpt_path"] == str(manual_checkpoint_dir.resolve())
    assert manifest["collect_policy_ckpt_selected_candidate_id"] == "manual_pinned"
    assert manifest["collect_policy_ckpt_selected_candidate_tier"] == "manual_pinned"
    assert manifest["collect_policy_ckpt_explicit_checkpoint_identity_present"] is True
    assert checkpoint_gate["pass"] is True
    assert checkpoint_gate["is_base_fallback"] is False
    assert run_manifest_gate["pass"] is True
    assert contract_gate["pass"] is True
    assert contract_gate["prelim_eval_surface"]["mode"] == "adv_server_required"


def test_bind_collect_checkpoint_selects_historical_candidate_by_success_rate_only(
    tmp_path: Path,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    _write_official_anchor_artifacts(repo_root)
    historical_checkpoint_dir = _checkpoint_index(
        repo_root,
        "agent/artifacts/checkpoints/historical/checkpoint-100",
        with_adv_pair=True,
    )

    result = stage3_collect_checkpoint_binding.bind_collect_checkpoint(
        repo_root=repo_root,
        manifest_path=manifest_path,
        historical_candidates=[
            stage3_collect_checkpoint_binding.build_local_candidate_spec(
                candidate_id="historical_candidate_a",
                checkpoint_path=historical_checkpoint_dir,
                success_rate=0.4,
                success_count=0,
            )
        ],
    )

    manifest = _read_json(manifest_path)
    provenance = manifest["collect_policy_ckpt_provenance"]
    assert isinstance(provenance, Mapping)

    assert result["collect_policy_ckpt_decision"] == "historical_best"
    assert manifest["collect_policy_ckpt_decision"] == "historical_best"
    assert manifest["collect_policy_ckpt_path"] == str(
        historical_checkpoint_dir.resolve()
    )
    assert provenance["selected_candidate_id"] == "historical_candidate_a"
    assert provenance["success_rate"] == 0.4
    assert provenance["success_count"] == 0
    assert provenance["success_rate_threshold_pass"] is True
    assert provenance["success_count_threshold_pass"] is False
    run_manifest_gate = provenance["run_manifest_gate"]
    checkpoint_provenance_gate = provenance["checkpoint_provenance_gate"]
    assert isinstance(run_manifest_gate, Mapping)
    assert isinstance(checkpoint_provenance_gate, Mapping)
    assert run_manifest_gate["pass"] is True
    assert checkpoint_provenance_gate["pass"] is True


def test_bind_collect_checkpoint_records_official_anchor_without_local_checkpoint_path(
    tmp_path: Path,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    _write_official_anchor_artifacts(repo_root)

    result = stage3_collect_checkpoint_binding.bind_collect_checkpoint(
        repo_root=repo_root,
        manifest_path=manifest_path,
        historical_candidates=[],
    )

    manifest = _read_json(manifest_path)
    checkpoint_gate = _read_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/checkpoint_provenance_gate.json"
    )
    run_manifest_gate = _read_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/run_manifest_gate.json"
    )
    provenance = manifest["collect_policy_ckpt_provenance"]
    assert isinstance(provenance, Mapping)

    assert result["selected_candidate_id"] == "official_task_anchor"
    assert (
        manifest["collect_policy_ckpt_selected_candidate_id"] == "official_task_anchor"
    )
    assert (
        manifest["collect_policy_ckpt_selected_candidate_tier"]
        == "official_task_anchor"
    )
    assert manifest["collect_policy_ckpt_path"] is None
    assert provenance["selected_candidate_id"] == "official_task_anchor"
    assert provenance["selected_checkpoint_path"] is None
    assert isinstance(provenance["authority_refs"], list)
    assert len(provenance["authority_refs"]) == 3
    assert checkpoint_gate["pass"] is False
    assert checkpoint_gate["reason_code"] == "base_fallback_forbidden"
    assert run_manifest_gate["pass"] is False


def test_collect_checkpoint_binding_accepts_stage3_t3b_baseline_1gpu_path(
    tmp_path: Path,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    _write_official_anchor_artifacts(repo_root)
    formal_checkpoint_dir = _checkpoint_index(
        repo_root,
        "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/checkpoint-200",
        with_adv_pair=True,
    )

    result = stage3_collect_checkpoint_binding.bind_collect_checkpoint(
        repo_root=repo_root,
        manifest_path=manifest_path,
        manual_checkpoint_path=formal_checkpoint_dir,
        historical_candidates=[],
    )

    manifest = _read_json(manifest_path)
    checkpoint_gate = _read_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/checkpoint_provenance_gate.json"
    )
    run_manifest_gate = _read_json(
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/run_manifest_gate.json"
    )

    assert result["collect_policy_ckpt_decision"] == "manual_pinned"
    assert manifest["collect_policy_ckpt_path"] == str(formal_checkpoint_dir.resolve())
    assert checkpoint_gate["pass"] is True
    assert run_manifest_gate["pass"] is True


def test_wrapper_main_exits_zero_and_falls_back_to_baseline_train_required(
    tmp_path: Path,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    _write_official_anchor_artifacts(repo_root)
    original_manifest = _read_json(manifest_path)
    wrapper = _load_wrapper_module()
    original_repo_root = getattr(wrapper, "REPO_ROOT")
    setattr(wrapper, "REPO_ROOT", repo_root)
    try:
        exit_code = wrapper.main(["--iteration-manifest", str(manifest_path)])
    finally:
        setattr(wrapper, "REPO_ROOT", original_repo_root)

    manifest = _read_json(manifest_path)

    assert exit_code == 0
    assert manifest["schema_version"] == "stage3_iteration_manifest_v3"
    assert manifest["collect_policy_ckpt_decision"] == "baseline_train_required"
    assert manifest["orchestrator_python"] == original_manifest["orchestrator_python"]
    assert (
        manifest["delegate_runtime_python"]
        == original_manifest["delegate_runtime_python"]
    )
    assert (
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/checkpoint_provenance_gate.json"
    ).is_file()
    assert (
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/run_manifest_gate.json"
    ).is_file()
    assert (
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/contract_precondition_gate.json"
    ).is_file()
    assert not (
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/superseded_outputs.json"
    ).exists()
