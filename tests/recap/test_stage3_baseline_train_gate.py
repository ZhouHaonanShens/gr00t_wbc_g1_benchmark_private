from __future__ import annotations

import importlib.util
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any, Protocol, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import stage3_baseline_train_gate


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
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


def _read_flag_value(cmd: list[str], flag: str) -> str:
    for index, token in enumerate(cmd):
        if token == flag:
            return str(cmd[index + 1])
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    raise AssertionError(f"missing {flag} in command: {cmd}")


def _write_single_gpu_smoke_verdict(
    repo_root: Path,
    *,
    summary_path: str | None = None,
    schema_version: str = "stage3_single_gpu_smoke_verdict_v1",
    pass_verdict: bool = True,
    unlock_signal: str = "single_gpu_formal_baseline_allowed",
    launch_family: str = "single_gpu_v1",
    num_gpus: int = 1,
    use_ddp: bool = False,
    global_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    per_device_batch_size: int = 1,
    effective_update_batch: int = 4,
) -> Path:
    return _write_json(
        repo_root
        / "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/green_smoke_single_gpu_verdict.json",
        {
            "schema_version": schema_version,
            "pass": bool(pass_verdict),
            "unlock_signal": unlock_signal,
            "launch_policy": {
                "summary_path": summary_path,
                "launch_family": launch_family,
                "num_gpus": num_gpus,
                "use_ddp": use_ddp,
            },
            "launch_contract": {
                "launch_family": launch_family,
                "num_gpus": num_gpus,
                "use_ddp": use_ddp,
                "global_batch_size": global_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "per_device_batch_size": per_device_batch_size,
                "effective_update_batch": effective_update_batch,
            },
        },
    )


def _write_single_gpu_smoke_summary(repo_root: Path) -> Path:
    summary_path = (
        repo_root
        / "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/delegate_finetune_summary.json"
    )
    return _write_json(
        summary_path,
        {
            "dataset_path": str(
                repo_root
                / "agent/artifacts/lerobot_datasets/openpi_phase05_smoke_contract_v1"
            ),
            "effective_config": {
                "max_steps": 8,
                "save_steps": 8,
                "save_total_limit": 1,
                "global_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "per_device_batch_size": 1,
                "effective_update_batch": 4,
                "dataloader_num_workers": 0,
                "learning_rate": 1e-5,
                "num_gpus": 1,
                "use_ddp": False,
            },
            "delegate_runtime_python": sys.executable,
            "wrapper_status": "ok",
        },
    )


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
            "artifact_root": "agent/artifacts/stage3_iteration/recap_stage3_iter_002/",
            "formal_iter_tag": "recap_stage3_iter_002",
            "train_iter_tag": "recap_stage3_iter_002_train",
            "hardware_profile": "rtx_pro_6000_blackwell_max_q_96g_x4",
            "orchestrator_python": sys.executable,
            "delegate_runtime_python": sys.executable,
            "collect_policy_ckpt_decision": "baseline_train_required",
            "historical_success_rate_threshold": 0.3,
            "collect_policy_ckpt_provenance": {
                "authority_refs": [
                    {
                        "relative_path": ".sisyphus/evidence/task-4-public-anchor.json",
                        "authority_role": "official_task_anchor_authority",
                    }
                ]
            },
        },
    )
    _write_json(
        repo_root / ".sisyphus/evidence/task-4-public-anchor.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_4_public_anchor_evidence",
        },
    )
    _write_json(
        repo_root / "agent/artifacts/stage3_prereq_smoke/train_smoke_summary.json",
        {
            "delegate_summary": None,
            "delegate_summary_exists": False,
            "delegate_returncode": 1,
            "wrapper_status": "blocked",
        },
    )
    _write_json(
        repo_root / "agent/artifacts/stage3_prereq_smoke/trainability_gate.json",
        {
            "schema_version": "gr00t_training_lane_trainability_gate_v1",
            "status": "FAIL",
        },
    )
    dataset_dir = (
        repo_root / "agent/artifacts/lerobot_datasets/openpi_phase05_smoke_contract_v1"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    smoke_summary_path = _write_single_gpu_smoke_summary(repo_root)
    _write_single_gpu_smoke_verdict(repo_root, summary_path=str(smoke_summary_path))
    return repo_root, manifest_path


def _stub_preflight_factory(*, pass_preflight: bool) -> Any:
    def _stub(
        *,
        repo_root: Path,
        manifest_path: Path,
        manifest_payload: Mapping[str, Any],
        preflight_path: Path,
        training_python_contract: Mapping[str, str],
    ) -> dict[str, Any]:
        del repo_root, manifest_path, manifest_payload, training_python_contract
        payload = {
            "schema_version": "stage3_rtx_pro_6000_blackwell_max_q_preflight_v2",
            "artifact_kind": "stage3_rtx_pro_6000_blackwell_max_q_preflight",
            "generated_at": "2026-04-18T16:30:00+00:00",
            "artifact_path": str(preflight_path),
            "expected_hardware_profile": "rtx_pro_6000_blackwell_max_q_96g_x2_subset",
            "pass": bool(pass_preflight),
            "status": "continue" if pass_preflight else "execution_hard_block",
            "hard_block_subfamily": None
            if pass_preflight
            else "hardware_profile_mismatch",
            "next_action": None
            if pass_preflight
            else "wait_for_rtx_pro_6000_blackwell_max_q_x2_subset",
            "repair_tool_status": "healthy_noop",
            "reason_codes": []
            if pass_preflight
            else [
                "gpu_count_below_2",
                "gpu_name_not_rtx_pro_6000_blackwell_max_q",
            ],
            "delegate_runtime_health": {
                "pass": True,
                "argv0_matches_manifest": True,
                "reason_codes": [],
            },
            "hardware_profile_match": {
                "pass": bool(pass_preflight),
                "reason_codes": [] if pass_preflight else ["gpu_count_below_2"],
            },
            "evidence": {
                "delegate_runtime_repair_summary_path": "agent/runtime_logs/stage3_delegate_runtime_repair/fake.json",
                "delegate_runtime_repair_summary_sha256": "f" * 64,
                "delegate_runtime_repair_log": "agent/runtime_logs/stage3_delegate_runtime_repair/fake.log",
                "delegate_runtime_repair_log_sha256": "e" * 64,
            },
        }
        _write_json(preflight_path, payload)
        return payload

    return _stub


def _meaningful_effective_config() -> dict[str, object]:
    return {
        "max_steps": 200,
        "save_steps": 50,
        "save_total_limit": 1,
        "global_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "per_device_batch_size": 1,
        "effective_update_batch": 4,
        "dataloader_num_workers": 0,
        "learning_rate": 1e-5,
        "num_gpus": 1,
        "tune_projector": True,
        "tune_diffusion_model": True,
    }


def _write_checkpoint_dir(
    checkpoint_dir: Path,
    *,
    include_advantage_embedding: bool,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {
        "action_head.linear.weight": "model-00001-of-00001.safetensors",
    }
    if include_advantage_embedding:
        weight_map.update(
            {
                "action_head.advantage_embedding.weight": "model-00001-of-00001.safetensors",
                "action_head.advantage_embedding.bias": "model-00001-of-00001.safetensors",
            }
        )
    _write_json(
        checkpoint_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 123},
            "weight_map": weight_map,
        },
    )
    _write_json(
        checkpoint_dir / "trainer_state.json",
        {
            "global_step": 200,
            "max_steps": 200,
        },
    )


def _stub_run_logged_command_factory(
    *,
    repo_root: Path,
    manifest_success_rate: float,
    checkpoint_gate_allow: bool,
    run_manifest_gate_allow: bool,
    include_advantage_embedding: bool = True,
    train_returncode: int = 0,
    eval_returncode: int = 0,
    episodes: int = 10,
    wrapper_status: str = "ok",
) -> Any:
    calls: list[list[str]] = []

    def _stub(*, cmd: list[str], cwd: Path, log_path: Path) -> int:
        del cwd
        calls.append(list(cmd))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("stub log\n", encoding="utf-8")
        joined = " ".join(cmd)
        attempt_root = (
            repo_root
            / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/baseline_train_attempt_001"
        )
        if "work/recap/finetune_full.py" in joined:
            output_dir = Path(_read_flag_value(cmd, "--output-dir"))
            for existing_checkpoint in output_dir.glob("checkpoint-*"):
                if existing_checkpoint.is_dir():
                    for child in existing_checkpoint.iterdir():
                        if child.is_file() or child.is_symlink():
                            child.unlink()
                    existing_checkpoint.rmdir()
            checkpoint_dir = output_dir / "checkpoint-50"
            _write_checkpoint_dir(
                checkpoint_dir,
                include_advantage_embedding=include_advantage_embedding,
            )
            _write_json(
                attempt_root / "baseline_train_finetune_summary.json",
                {
                    "selected_checkpoint_path": str(checkpoint_dir),
                    "selected_checkpoint_exists": True,
                    "selected_checkpoint_asset_path": str(
                        checkpoint_dir / "model.safetensors.index.json"
                    ),
                    "effective_config": _meaningful_effective_config(),
                },
            )
            return train_returncode
        if "45d_vlm_critic_eval_smoke.py" in joined:
            _write_json(
                attempt_root / "baseline_train_prelim_eval_summary.json",
                {
                    "episodes": int(episodes),
                    "requested_episodes": 10,
                    "success_count": int(round(manifest_success_rate * 10)),
                    "success_rate": float(manifest_success_rate),
                    "wrapper_status": str(wrapper_status),
                },
            )
            return eval_returncode
        if "gr00t_checkpoint_provenance_gate.py" in joined:
            raw_dir = attempt_root / "checkpoint_provenance_gate_raw"
            _write_json(
                raw_dir / "checkpoint_provenance_report.json",
                {
                    "formal_eligibility": "ALLOW" if checkpoint_gate_allow else "BLOCK",
                    "reason_code": "ok"
                    if checkpoint_gate_allow
                    else "wrong_checkpoint_or_missing_finetune_artifact",
                },
            )
            return 0 if checkpoint_gate_allow else 1
        if "gr00t_run_manifest_gate.py" in joined:
            raw_dir = attempt_root / "run_manifest_gate_raw"
            _write_json(
                raw_dir / "run_manifest_gate_report.json",
                {
                    "formal_eligibility": "ALLOW"
                    if run_manifest_gate_allow
                    else "BLOCK",
                    "reason_code": "ok"
                    if run_manifest_gate_allow
                    else "invalid_run_manifest",
                    "core_digest": "digest-1",
                    "issues": [],
                    "checkpoint_binding": {},
                    "core": {"checkpoint_selected": "checkpoint-50"},
                },
            )
            _write_json(
                raw_dir / "run_manifest.json",
                {
                    "schema_version": "gr00t_run_manifest_v1",
                    "core": {"checkpoint_selected": "checkpoint-50"},
                },
            )
            return 0 if run_manifest_gate_allow else 1
        raise AssertionError(f"unexpected command: {cmd}")

    return _stub, calls


def test_collect_duplicate_processes_ignores_parent_shell_wrapper(
    monkeypatch: Any,
) -> None:
    ps_stdout = (
        "    PID COMMAND\n"
        '1395297 /bin/bash -c "/media/howard/Data/Projects/gr00t_wbc_g1_benchmark/.venv/bin/python" work/recap/scripts/30b_stage3_baseline_train_gate.py --iteration-manifest agent/artifacts/stage3_iteration/recap_stage3_iter_002/iteration_manifest.json\n'
    )

    def _stub_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(
            args=["ps", "-eo", "pid,args"],
            returncode=0,
            stdout=ps_stdout,
            stderr="",
        )

    monkeypatch.setattr(stage3_baseline_train_gate.subprocess, "run", _stub_run)

    duplicates = stage3_baseline_train_gate._collect_duplicate_processes(current_pid=42)

    assert duplicates == []


def test_collect_duplicate_processes_keeps_direct_python_conflict(
    monkeypatch: Any,
) -> None:
    ps_stdout = (
        "    PID COMMAND\n"
        "2001 /media/howard/Data/Projects/gr00t_wbc_g1_benchmark/.venv/bin/python work/recap/finetune_full.py --dataset-path foo --output-dir bar\n"
    )

    def _stub_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(
            args=["ps", "-eo", "pid,args"],
            returncode=0,
            stdout=ps_stdout,
            stderr="",
        )

    monkeypatch.setattr(stage3_baseline_train_gate.subprocess, "run", _stub_run)

    duplicates = stage3_baseline_train_gate._collect_duplicate_processes(current_pid=42)

    assert len(duplicates) == 1
    assert duplicates[0]["pid"] == 2001
    assert duplicates[0]["matched_patterns"] == ["work/recap/finetune_full.py"]


def test_stage3_baseline_train_gate_skips_when_condition_not_met(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    manifest = _read_json(manifest_path)
    manifest["collect_policy_ckpt_decision"] = "historical_best"
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=True),
    )

    result = stage3_baseline_train_gate.run_stage3_baseline_train_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    assert result["status"] == "skipped_by_condition"
    assert result["exit_code"] == 0
    persisted = _read_json(manifest_path)
    assert "collect_policy_ckpt_baseline_train_attempt" not in persisted


def test_stage3_baseline_train_gate_blocks_finished_second_attempt_when_marker_exists(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=True),
    )
    manifest = _read_json(manifest_path)
    manifest["collect_policy_ckpt_baseline_train_attempt"] = {
        "schema_version": "stage3_collect_policy_baseline_train_attempt_v1",
        "state": "finished",
        "attempt_number": 1,
    }
    _write_json(manifest_path, manifest)

    result = stage3_baseline_train_gate.run_stage3_baseline_train_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    assert result["status"] == "blocked_second_attempt"
    assert result["exit_code"] == 1


def test_stage3_baseline_train_gate_repairs_started_attempt_instead_of_blocking(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    stub, _calls = _stub_run_logged_command_factory(
        repo_root=repo_root,
        manifest_success_rate=0.4,
        checkpoint_gate_allow=True,
        run_manifest_gate_allow=True,
    )
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=True),
    )
    monkeypatch.setattr(stage3_baseline_train_gate, "_run_logged_command", stub)
    checkpoint_dir = (
        repo_root / "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/checkpoint-150"
    )
    _write_checkpoint_dir(checkpoint_dir, include_advantage_embedding=False)
    manifest = _read_json(manifest_path)
    manifest["collect_policy_ckpt_baseline_train_attempt"] = {
        "schema_version": "stage3_collect_policy_baseline_train_attempt_v1",
        "state": "started",
        "attempt_number": 1,
        "started_at": "2026-04-21T12:00:00+00:00",
        "finetune_output_dir": "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run",
    }
    _write_json(manifest_path, manifest)

    result = stage3_baseline_train_gate.run_stage3_baseline_train_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    persisted = _read_json(manifest_path)
    attempt = persisted["collect_policy_ckpt_baseline_train_attempt"]
    assert result["status"] == "ok"
    assert result["exit_code"] == 0
    assert isinstance(attempt, Mapping)
    assert attempt["state"] == "finished"
    assert attempt["started_at"] == "2026-04-21T12:00:00+00:00"


def test_stage3_baseline_train_gate_preflight_fail_does_not_consume_attempt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=False),
    )

    result = stage3_baseline_train_gate.run_stage3_baseline_train_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    manifest = _read_json(manifest_path)
    assert result["status"] == "execution_hard_block"
    assert result["exit_code"] == 1
    assert manifest["collect_policy_ckpt_decision"] == "baseline_train_required"
    assert "collect_policy_ckpt_baseline_train_attempt" not in manifest
    hard_block = manifest["collect_policy_ckpt_hard_block"]
    assert isinstance(hard_block, Mapping)
    assert hard_block["status_family"] == "execution_hard_block"
    assert hard_block["attempt_budget_consumed"] is False
    assert hard_block["hard_block_subfamily"] == "hardware_profile_mismatch"
    assert hard_block["next_action"] == "wait_for_rtx_pro_6000_blackwell_max_q_x2_subset"
    assert hard_block["repair_tool_status"] == "healthy_noop"
    artifact_paths = hard_block["artifact_paths"]
    assert isinstance(artifact_paths, Mapping)
    assert "finetune_summary_path" not in artifact_paths
    assert "prelim_eval_summary_path" not in artifact_paths
    preflight = manifest["rtx_pro_6000_blackwell_max_q_preflight"]
    assert isinstance(preflight, Mapping)
    assert preflight["pass"] is False
    assert (
        preflight["schema_version"]
        == "stage3_rtx_pro_6000_blackwell_max_q_preflight_v2"
    )
    assert preflight["hard_block_subfamily"] == "hardware_profile_mismatch"
    assert preflight["next_action"] == "wait_for_rtx_pro_6000_blackwell_max_q_x2_subset"
    assert manifest["next_action"] == "wait_for_rtx_pro_6000_blackwell_max_q_x2_subset"
    assert not (
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/baseline_train_attempt_001"
    ).exists()


def test_stage3_baseline_train_gate_preflight_only_refreshes_manifest_without_attempt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=True),
    )

    result = stage3_baseline_train_gate.run_stage3_baseline_train_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
        preflight_only=True,
    )

    manifest = _read_json(manifest_path)
    assert result["status"] == "preflight_ready"
    assert result["exit_code"] == 0
    assert result["attempt_budget_consumed"] is False
    assert manifest["collect_policy_ckpt_decision"] == "baseline_train_required"
    assert manifest["collect_policy_ckpt_t3_status"] == "continue"
    assert "collect_policy_ckpt_baseline_train_attempt" not in manifest
    preflight = manifest["rtx_pro_6000_blackwell_max_q_preflight"]
    assert isinstance(preflight, Mapping)
    assert preflight["pass"] is True
    assert preflight["status"] == "continue"
    assert manifest["hardware_profile"] == "rtx_pro_6000_blackwell_max_q_96g_x2_subset"
    assert manifest["next_action"] is None
    assert not (
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/baseline_train_attempt_001"
    ).exists()


def test_stage3_baseline_train_gate_writes_baseline_trained_manifest(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    stub, calls = _stub_run_logged_command_factory(
        repo_root=repo_root,
        manifest_success_rate=0.4,
        checkpoint_gate_allow=True,
        run_manifest_gate_allow=True,
    )
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=True),
    )
    monkeypatch.setattr(stage3_baseline_train_gate, "_run_logged_command", stub)

    result = stage3_baseline_train_gate.run_stage3_baseline_train_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    manifest = _read_json(manifest_path)
    assert result["collect_policy_ckpt_decision"] == "baseline_trained"
    assert result["status"] == "ok"
    assert manifest["collect_policy_ckpt_decision"] == "baseline_trained"
    assert manifest["collect_policy_ckpt_path"]
    assert manifest["collect_policy_ckpt_prelim_success_rate"] == 0.4
    assert manifest["collect_policy_ckpt_t3_status"] == "continue"
    attempt = manifest["collect_policy_ckpt_baseline_train_attempt"]
    assert isinstance(attempt, Mapping)
    assert attempt["state"] == "finished"
    assert attempt["attempt_number"] == 1
    assert attempt["checkpoint_provenance_gate_pass"] is True
    assert attempt["run_manifest_gate_pass"] is True
    assert attempt["meaningful_config_pass"] is True
    assert attempt["retained_checkpoint_count"] == 1
    assert attempt["prereq_smoke_summary_path"] == (
        "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/"
        "delegate_finetune_summary.json"
    )
    assert attempt["trainability_gate_path"] == (
        "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/"
        "green_smoke_single_gpu_verdict.json"
    )
    assert manifest["collect_policy_ckpt_hard_block"] is None
    assert len(calls) == 4
    finetune_cmd = calls[0]
    eval_cmd = calls[1]
    assert (
        _read_flag_value(finetune_cmd, "--dataset-path")
        == str(
            repo_root / "agent/artifacts/lerobot_datasets/openpi_phase05_smoke_contract_v1"
        )
    )
    assert "--max-steps" in finetune_cmd and "200" in finetune_cmd
    assert "--save-steps" in finetune_cmd and "50" in finetune_cmd
    assert "--global-batch-size" in finetune_cmd and "4" in finetune_cmd
    assert "--gradient-accumulation-steps" in finetune_cmd and "4" in finetune_cmd
    assert "--dataloader-num-workers" in finetune_cmd and "0" in finetune_cmd
    assert "--num-gpus" in finetune_cmd and "1" in finetune_cmd
    assert "--output-dir" in finetune_cmd
    assert (
        _read_flag_value(finetune_cmd, "--output-dir")
        == str(repo_root / "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run")
    )
    assert _read_flag_value(eval_cmd, "--base-model-path") == "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
    assert "--tune-projector" in finetune_cmd
    assert "--tune-diffusion-model" in finetune_cmd
    assert manifest["hardware_profile"] == "rtx_pro_6000_blackwell_max_q_96g_x2_subset"
    prelaunch_summary = attempt["prelaunch_summary"]
    assert isinstance(prelaunch_summary, Mapping)
    assert prelaunch_summary["launch_family"] == "single_gpu_v1"
    assert prelaunch_summary["single_gpu_launch_authority_pass"] is True
    assert (
        repo_root / "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/trainer_state.json"
    ).is_file()
    assert (
        repo_root
        / "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/checkpoint_provenance_gate.json"
    ).is_file()
    assert (
        repo_root
        / "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/run_manifest_gate.json"
    ).is_file()


def test_baseline_gate_accepts_single_gpu_v1_smoke_verdict(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    stub, _calls = _stub_run_logged_command_factory(
        repo_root=repo_root,
        manifest_success_rate=0.4,
        checkpoint_gate_allow=True,
        run_manifest_gate_allow=True,
    )
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=True),
    )
    monkeypatch.setattr(stage3_baseline_train_gate, "_run_logged_command", stub)

    result = stage3_baseline_train_gate.run_stage3_baseline_train_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
        launch_family="single_gpu_v1",
        single_gpu_smoke_verdict_path=repo_root
        / "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/green_smoke_single_gpu_verdict.json",
    )

    manifest = _read_json(manifest_path)
    attempt = manifest["collect_policy_ckpt_baseline_train_attempt"]
    assert result["status"] == "ok"
    assert isinstance(attempt, Mapping)
    prelaunch_summary = attempt["prelaunch_summary"]
    assert isinstance(prelaunch_summary, Mapping)
    assert prelaunch_summary["launch_family"] == "single_gpu_v1"
    assert prelaunch_summary["single_gpu_smoke_verdict_schema_version"] == (
        "stage3_single_gpu_smoke_verdict_v1"
    )
    assert prelaunch_summary["single_gpu_launch_authority_pass"] is True
    assert prelaunch_summary["single_gpu_launch_authority_reason_codes"] == []


def test_baseline_gate_accepts_real_smoke_verdict_shape_without_launch_contract(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    smoke_summary_path = (
        repo_root
        / "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/delegate_finetune_summary.json"
    )
    _write_json(
        repo_root
        / "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/green_smoke_single_gpu_verdict.json",
        {
            "schema_version": "stage3_single_gpu_smoke_verdict_v1",
            "pass": True,
            "unlock_signal": "single_gpu_formal_baseline_allowed",
            "launch_policy": {
                "summary_path": str(smoke_summary_path),
                "num_gpus": 1,
                "use_ddp": False,
            },
            "geometry_checks": {
                "global_batch_size": {"actual": 4, "expected": 4, "pass": True},
                "gradient_accumulation_steps": {
                    "actual": 4,
                    "expected": 4,
                    "pass": True,
                },
                "per_device_batch_size": {"actual": 1, "expected": 1, "pass": True},
                "effective_update_batch": {
                    "actual": 4,
                    "expected": 4,
                    "pass": True,
                },
            },
        },
    )
    stub, _calls = _stub_run_logged_command_factory(
        repo_root=repo_root,
        manifest_success_rate=0.4,
        checkpoint_gate_allow=True,
        run_manifest_gate_allow=True,
    )
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=True),
    )
    monkeypatch.setattr(stage3_baseline_train_gate, "_run_logged_command", stub)

    result = stage3_baseline_train_gate.run_stage3_baseline_train_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    assert result["status"] == "ok"


def test_load_prereq_uses_live_single_gpu_smoke_summary_instead_of_historical_prereq(
    tmp_path: Path,
) -> None:
    repo_root, _manifest_path = _make_repo_fixture(tmp_path)

    prereq = stage3_baseline_train_gate._load_prereq(
        repo_root,
        single_gpu_smoke_verdict_path=(
            repo_root
            / "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/green_smoke_single_gpu_verdict.json"
        ),
    )

    assert prereq.smoke_summary_path == (
        repo_root
        / "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/delegate_finetune_summary.json"
    )
    assert prereq.trainability_gate_path == (
        repo_root
        / "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/green_smoke_single_gpu_verdict.json"
    )
    assert prereq.dataset_path == (
        repo_root / "agent/artifacts/lerobot_datasets/openpi_phase05_smoke_contract_v1"
    )


def test_baseline_gate_rejects_task10_2gpu_live_authority_verdict(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    _write_single_gpu_smoke_verdict(
        repo_root,
        schema_version="task10_green_smoke_verdict_v1",
        launch_family="task10_2gpu_ddp_diagnostic_v1",
        num_gpus=2,
        use_ddp=True,
        gradient_accumulation_steps=1,
        per_device_batch_size=2,
        effective_update_batch=4,
    )
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=True),
    )

    result = stage3_baseline_train_gate.run_stage3_baseline_train_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    manifest = _read_json(manifest_path)
    assert result["status"] == "inconclusive_contract_mismatch"
    assert result["single_gpu_launch_authority_pass"] is False
    assert (
        "historical_task10_ddp_verdict_rejected_as_live_authority"
        in result["reason_codes"]
    )
    assert "collect_policy_ckpt_baseline_train_attempt" not in manifest
    hard_block = manifest["collect_policy_ckpt_hard_block"]
    assert isinstance(hard_block, Mapping)
    assert hard_block["attempt_budget_consumed"] is False
    assert (
        "historical_task10_ddp_verdict_rejected_as_live_authority"
        in hard_block["reason_codes"]
    )


def test_stage3_baseline_train_gate_allows_baseline_checkpoint_without_advantage_weights(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    stub, calls = _stub_run_logged_command_factory(
        repo_root=repo_root,
        manifest_success_rate=0.0,
        checkpoint_gate_allow=True,
        run_manifest_gate_allow=True,
        include_advantage_embedding=False,
    )
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=True),
    )
    monkeypatch.setattr(stage3_baseline_train_gate, "_run_logged_command", stub)

    result = stage3_baseline_train_gate.run_stage3_baseline_train_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    manifest = _read_json(manifest_path)
    assert result["status"] == "ok"
    assert result["exit_code"] == 0
    assert manifest["collect_policy_ckpt_decision"] == "baseline_trained"
    assert result["reason_codes"] == []
    assert manifest["collect_policy_ckpt_prelim_success_rate"] == 0.0
    attempt = manifest["collect_policy_ckpt_baseline_train_attempt"]
    assert isinstance(attempt, Mapping)
    features = attempt["checkpoint_weight_map_features"]
    assert isinstance(features, Mapping)
    assert features["has_advantage_embedding_pair"] is False
    assert features["baseline_like_path"] is True
    assert attempt["prelim_success_rate"] == 0.0
    eval_cmd = calls[1]
    assert "--base-model-path" not in eval_cmd
    assert (
        repo_root / "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/trainer_state.json"
    ).is_file()
    assert (
        repo_root
        / "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/checkpoint_provenance_gate.json"
    ).is_file()
    assert (
        repo_root
        / "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run/run_manifest_gate.json"
    ).is_file()


def test_stage3_baseline_train_gate_inconclusive_when_eval_contract_fails(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    stub, _calls = _stub_run_logged_command_factory(
        repo_root=repo_root,
        manifest_success_rate=0.2,
        checkpoint_gate_allow=True,
        run_manifest_gate_allow=False,
        include_advantage_embedding=True,
        episodes=0,
        wrapper_status="blocked",
    )
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=True),
    )
    monkeypatch.setattr(stage3_baseline_train_gate, "_run_logged_command", stub)

    result = stage3_baseline_train_gate.run_stage3_baseline_train_gate(
        repo_root=repo_root,
        manifest_path=manifest_path,
    )

    manifest = _read_json(manifest_path)
    assert result["status"] == "inconclusive_contract_mismatch"
    assert manifest["collect_policy_ckpt_decision"] == "baseline_train_required"
    hard_block = manifest["collect_policy_ckpt_hard_block"]
    assert isinstance(hard_block, Mapping)
    assert hard_block["status_family"] == "inconclusive_contract_mismatch"
    reason_codes = hard_block["reason_codes"]
    assert isinstance(reason_codes, list)
    assert "prelim_eval_wrapper_blocked" in reason_codes
    assert "prelim_eval_completed_episodes_below_minimum" in reason_codes
    assert "run_manifest_gate_blocked" in reason_codes


class _WrapperModule(Protocol):
    REPO_ROOT: Path

    def main(self, argv: list[str] | None = None) -> int: ...


def _load_wrapper_module() -> _WrapperModule:
    script_path = REPO_ROOT / "work/recap/scripts/30b_stage3_baseline_train_gate.py"
    spec = importlib.util.spec_from_file_location(
        "stage3_baseline_train_gate_wrapper",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_WrapperModule, cast(object, module))


def test_stage3_baseline_train_gate_wrapper_returns_exit_code_from_workflow(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path = _make_repo_fixture(tmp_path)
    wrapper = _load_wrapper_module()
    original_repo_root = getattr(wrapper, "REPO_ROOT")
    stub, _calls = _stub_run_logged_command_factory(
        repo_root=repo_root,
        manifest_success_rate=0.4,
        checkpoint_gate_allow=True,
        run_manifest_gate_allow=True,
    )
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_rtx_pro_6000_blackwell_max_q_preflight",
        _stub_preflight_factory(pass_preflight=True),
    )
    monkeypatch.setattr(stage3_baseline_train_gate, "_run_logged_command", stub)
    setattr(wrapper, "REPO_ROOT", repo_root)
    try:
        exit_code = wrapper.main(
            [
                "--iteration-manifest",
                str(manifest_path),
                "--launch-family",
                "single_gpu_v1",
                "--single-gpu-smoke-verdict",
                str(
                    repo_root
                    / "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/green_smoke_single_gpu_verdict.json"
                ),
                "--formal-output-dir",
                str(repo_root / "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run"),
            ]
        )
    finally:
        setattr(wrapper, "REPO_ROOT", original_repo_root)

    assert exit_code == 0


def test_stage3_baseline_train_gate_meaningful_contract_constants() -> None:
    assert stage3_baseline_train_gate.LIVE_LAUNCH_FAMILY == "single_gpu_v1"
    assert stage3_baseline_train_gate.MEANINGFUL_NUM_GPUS == 1
    assert stage3_baseline_train_gate.MEANINGFUL_GLOBAL_BATCH_SIZE == 4
    assert stage3_baseline_train_gate.MEANINGFUL_DATALOADER_NUM_WORKERS == 0
    assert stage3_baseline_train_gate.MEANINGFUL_GRADIENT_ACCUMULATION_STEPS == 4
    assert stage3_baseline_train_gate.MEANINGFUL_PER_DEVICE_BATCH_SIZE == 1
    assert stage3_baseline_train_gate.MEANINGFUL_EFFECTIVE_UPDATE_BATCH == 4
    assert stage3_baseline_train_gate.MEANINGFUL_MAX_STEPS == 200
    assert stage3_baseline_train_gate.MEANINGFUL_SAVE_STEPS == 50
    assert stage3_baseline_train_gate.MEANINGFUL_SAVE_TOTAL_LIMIT == 1
    assert stage3_baseline_train_gate.MEANINGFUL_LEARNING_RATE == 1e-5
    assert (
        stage3_baseline_train_gate.EXPECTED_HARDWARE_PROFILE
        == "rtx_pro_6000_blackwell_max_q_96g_x2_subset"
    )
