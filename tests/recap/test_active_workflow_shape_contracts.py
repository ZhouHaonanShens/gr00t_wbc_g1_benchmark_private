from __future__ import annotations

import importlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


state_conditioned_offline_sanity = importlib.import_module(
    "work.recap.scripts.state_conditioned_offline_sanity"
)
multi_iter_loop = importlib.import_module("work.recap.multi_iter_loop")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_module(filename: str, module_name: str):
    module_path = REPO_ROOT / "work" / "recap" / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_representative_workflows_match_config_runner_main_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_run_module("3A_recap_multi_iter_loop.py", "recap_3a_workflow_shape")
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_maybe_reexec_into_wbc_venv", lambda _repo_root: None)
    monkeypatch.setattr(
        module, "_git_head_and_dirty", lambda _repo_root: ("test-sha", False)
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "3A_recap_multi_iter_loop.py",
            "--run-id",
            "workflow_shape_p3a",
            "--n-iterations",
            "1",
            "--dry-run",
            "--no-require-git-clean",
            "--no-write-repro-snapshot",
        ],
    )

    exit_code = module.main()

    captured = capsys.readouterr()
    manifest = _read_json(
        tmp_path
        / "agent"
        / "artifacts"
        / "p3A"
        / "workflow_shape_p3a"
        / "manifest.json"
    )
    stages = manifest["iterations"][0]["stages"]

    assert exit_code == 0
    assert "[EVIDENCE] dry_run=True" in captured.out
    assert manifest["dry_run"] is True
    assert manifest["params"]["n_iterations"] == 1
    assert manifest["params"]["require_git_clean"] is False
    assert manifest["params"]["write_repro_snapshot"] is False
    assert [stage["name"] for stage in stages] == [
        "05_eval_base_advpos",
        "10_collect",
        "20_critic_cumulative",
        "30_label_value_source_critic",
        "40_export_with_video_dual_task_text",
        "50_finetune_upstream",
        "60_eval_ft_advpos",
    ]
    assert all(stage["skipped"] is True for stage in stages)
    assert {stage["skip_reason"] for stage in stages} == {"dry_run"}
    assert stages[0]["tags"]["non_fatal"] == "true"
    assert (
        stages[-1]["tags"]["eval_iter_tag"]
        == "recap_workflow_shape_p3a_k0_eval_ft_advpos"
    )


def test_3a_script_app_run_keeps_wrapper_patch_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_module("3A_recap_multi_iter_loop.py", "recap_3a_script_app_sync")
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_maybe_reexec_into_wbc_venv", lambda _repo_root: None)
    monkeypatch.setattr(
        module, "_git_head_and_dirty", lambda _repo_root: ("test-sha", False)
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "3A_recap_multi_iter_loop.py",
            "--run-id",
            "workflow_shape_p3a_script_app",
            "--n-iterations",
            "1",
            "--dry-run",
            "--no-require-git-clean",
            "--no-write-repro-snapshot",
        ],
    )

    exit_code = module.RecapMultiIterLoopScriptApp().run()

    manifest = _read_json(
        tmp_path
        / "agent"
        / "artifacts"
        / "p3A"
        / "workflow_shape_p3a_script_app"
        / "manifest.json"
    )

    assert exit_code == 0
    assert manifest["run_id"] == "workflow_shape_p3a_script_app"
    assert manifest["git"] == {"sha": "test-sha", "dirty": False}


def test_online_loop_iterate_dry_run_manifest_stays_stage_addressable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_run_module(
        "38_recap_online_loop_iterate.py", "recap_38_workflow_shape"
    )
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        module, "_git_head_and_dirty", lambda _repo_root: ("test-sha", False)
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "38_recap_online_loop_iterate.py",
            "--run-id",
            "workflow_shape_p38",
            "--dry-run",
            "--no-require-git-clean",
            "--no-write-repro-snapshot",
        ],
    )

    exit_code = module.main()

    manifest = _read_json(
        tmp_path
        / "agent"
        / "artifacts"
        / "p38"
        / "workflow_shape_p38"
        / "manifest.json"
    )
    stages = manifest["stages"]

    assert exit_code == 0
    assert manifest["run_id"] == "workflow_shape_p38"
    assert manifest["iter_tags"] == {
        "k0": "recap_workflow_shape_p38_k0",
        "k1": "recap_workflow_shape_p38_k1",
    }
    assert manifest["ports"] == {
        "base": 5800,
        "collect_k0": 5800,
        "eval": 5801,
        "collect_k1": 5802,
    }
    assert [stage["name"] for stage in stages] == [
        "10_collect_k0",
        "20_label_k0",
        "30_export_with_video_k0",
        "40_finetune_k0",
        "50_eval_k0_base_raw",
        "51_eval_k0_base_advpos",
        "52_eval_k0_ft_raw",
        "53_eval_k0_ft_advpos",
        "11_collect_k1",
    ]
    assert all(stage["skipped"] is True for stage in stages)
    assert all(stage["rc"] is None for stage in stages)


def test_multi_iter_loop_config_normalizes_defaults_and_pins(
    tmp_path: Path,
) -> None:
    args = Namespace(
        run_id="cfg_contract",
        start_policy_path="policy/start",
        n_iterations=2,
        dry_run=True,
        require_git_clean=False,
        write_repro_snapshot=False,
        env_name="env/name",
        embodiment_tag="UNITREE_G1",
        mujoco_gl="egl",
        n_action_steps_config=30,
        server_host="127.0.0.1",
        server_port=5800,
        seed=7,
        seed_offset_per_iter=10000,
        fixed_eval_seed="123",
        collect_episodes=40,
        collect_max_policy_steps=10,
        mixdone=True,
        dual_task_text=True,
        mixdone_short_episodes=10,
        mixdone_long_episodes=30,
        mixdone_short_max_episode_steps=60,
        mixdone_long_max_episode_steps=1440,
        mixdone_long_seed_offset=1000,
        critic_bins=201,
        critic_max_epochs=100,
        critic_patience=10,
        critic_lr=1e-3,
        critic_val_ratio=0.1,
        critic_device="cuda",
        finetune_max_steps=200,
        finetune_save_steps=None,
        finetune_save_total_limit=1,
        finetune_tune_projector=False,
        finetune_tune_diffusion_model=True,
        eval_episodes=10,
        eval_max_policy_steps=None,
        eval_policy_prompt_prefix="advantage positive ",
        timeout_collect_s=100.0,
        timeout_critic_s=101.0,
        timeout_label_s=102.0,
        timeout_export_s=103.0,
        timeout_finetune_s=104.0,
        timeout_eval_s=105.0,
        min_free_gb=40.0,
        archive_root=str(tmp_path / "archive_root"),
        keep_last_n_iters_local=1,
        pin_checkpoint_dir=["recap_cfg_contract_k0", "recap_cfg_contract_k0"],
    )

    config = multi_iter_loop.build_workflow_config(
        args,
        repo_root=tmp_path,
        git_sha="test-sha",
        git_dirty=False,
        stage_python="/usr/bin/python3",
        validate_tag=lambda tag, name: str(tag),
    )

    assert config.run_id == "cfg_contract"
    assert config.fixed_eval_seed == 123
    assert config.finetune_save_steps == 200
    assert config.pin_checkpoint_dirs == ("recap_cfg_contract_k0",)
    assert (
        config.runtime_dir
        == tmp_path / "agent" / "runtime_logs" / "p3A" / "cfg_contract"
    )
    assert (
        config.artifacts_dir
        == tmp_path / "agent" / "artifacts" / "p3A" / "cfg_contract"
    )


def test_state_conditioned_offline_sanity_remains_machine_readable_current_workflow(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "offline_sanity"

    exit_code = state_conditioned_offline_sanity.main(["--output-dir", str(output_dir)])

    report = _read_json(output_dir / state_conditioned_offline_sanity.REPORT_JSON_NAME)
    assert exit_code == 0
    assert report["schema_version"] == state_conditioned_offline_sanity.SCHEMA_VERSION
    assert report["artifact_kind"] == "state_conditioned_offline_sanity_report"
    assert report["status"] == "PASS"
    assert report["failure"] is None
    assert set(report["checks"].keys()) == set(
        state_conditioned_offline_sanity.CHECK_ORDER
    )
    assert report["summary"] == {
        "passed_check_count": len(state_conditioned_offline_sanity.CHECK_ORDER),
        "total_check_count": len(state_conditioned_offline_sanity.CHECK_ORDER),
    }
