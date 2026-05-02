from __future__ import annotations
import json
from pathlib import Path
from work.recap.stage_b import p0_eval_protocol_runner as runner
EXPECTED_CHECKLIST_16 = tuple("json:baseline_manifest_v1.json|json:internal_g3_checkpoint_6600/artifact_index.json|json:internal_g3_checkpoint_6600/env_lock.json|json:internal_g3_checkpoint_6600/git_provenance.json|json:public_gr00t_g1_baseline/public_repo_lock.json|json:public_gr00t_g1_baseline/public_dataset_lock.json|json:public_gr00t_g1_baseline/public_reproduction_run_summary.json|json:public_gr00t_g1_baseline/level0_server_smoke_summary.json|nonempty:baseline_manifest_v1.md|nonempty:pre_registration_v1.md|nonempty:pre_registration_seed_table_v1.csv|nonempty:final_gate_decision.md|nonempty:openpi_auxiliary_evidence/openpi_not_primary_baseline_note.md|nonempty:openpi_auxiliary_evidence/openpi_carrier_summary.md|nonempty:public_gr00t_g1_baseline/worker2_a4_a5_verification.log|nonempty:logs/worker3_a6_a7_verification_summary.md".split("|"))
def test_static_protocol_contracts() -> None:
    assert runner.CHECKLIST_16 == EXPECTED_CHECKLIST_16
    assert len(runner.verify_stage_a_checklist(runner.REPO_ROOT / runner.STAGE_A_DIR)["checks"]) == 16
    cells = runner.expand_level0_vram_cells()
    assert {(c["gpu"], c["n_envs"]) for c in cells} == {(1, 1), (1, 5), (1, 30), (2, 1), (2, 5), (2, 30)}
    assert {c["seed"] for c in cells} == {20000}
    assert all(c["checkpoint_role"] == "post_recap" and c["episode_count"] == 1 for c in cells)
    assert not runner.should_run_nenvs50({"5": 0.0, "30": 0.0})
    assert runner.should_run_nenvs50({"5": 0.5, "30": 0.0})
def test_seed_replay_and_pending_gate(tmp_path: Path, monkeypatch) -> None:
    assert runner.load_seeds(runner.REPO_ROOT / runner.STAGE_A_DIR, 30) == list(range(20000, 20030))
    p0 = tmp_path / "stage_b" / runner.P0_REL
    p0.mkdir(parents=True)
    gate = p0 / "p0_gate_decision.json"
    gate.write_text(json.dumps({"decision": "P0_BLOCKED", "blocked_by": "P1", "training_allowed": True}), encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    runner.write_pending_gate(type("Args", (), {"stage_b_dir": "stage_b"})())
    payload = json.loads(gate.read_text(encoding="utf-8"))
    assert payload["decision"] == "P0_PENDING_EXEC"
    assert payload["blocked_by"] is None
    for key in ("training_allowed", "checkpoint_update_allowed", "continue_to_p2", "continue_to_runtime_probes", "method_claim_allowed"):
        assert payload[key] is False
def test_base_failure_writes_stop_record_and_unknown_provenance(tmp_path: Path, monkeypatch) -> None:
    result = {"status": "FAIL", "error": "boom", "modality_summary": {}, "mujoco_params": {}}
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "launch_group", lambda *_args: [result])
    args = type("Args", (), {"stage_b_dir": "stage_b", "base_model": str(tmp_path / "missing"), "timeout_s": 1, "server_timeout_s": 1})()
    assert runner.run_base(args) == 2
    stop = json.loads((tmp_path / "stage_b" / runner.P0_REL / "stop_record.json").read_text(encoding="utf-8"))
    provenance = json.loads((tmp_path / "stage_b" / runner.P0_REL / "cell_runner_provenance.json").read_text(encoding="utf-8"))
    assert stop["downstream_blocks"] == {"p2_allowed": False, "runtime_probe_allowed": False, "training_allowed": False, "checkpoint_update_allowed": False, "method_claim_allowed": False}
    assert provenance["num_diffusion_steps"]["source_key"] == "NOT_FOUND"
def test_forbidden_surface_guard_and_line_budget() -> None:
    files = [runner.REPO_ROOT / "work/recap/stage_b/p0_eval_protocol_runner.py", runner.REPO_ROOT / "tests/recap/test_p0_eval_protocol_runner.py"]
    src = files[0].read_text(encoding="utf-8")
    assert all(token not in src for token in ("launch_finetune", "torchrun", "save_total_limit", "Probe A", "unconditional swap"))
    assert sum(len(path.read_text(encoding="utf-8").splitlines()) for path in files) <= 300
