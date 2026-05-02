from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import advantage


def _load_script_module(script_name: str, module_name: str):
    module_path = REPO_ROOT / "work" / "recap" / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_contract_spec_freezes_critic_smoke_as_diagnostic_only() -> None:
    checker = _load_script_module(
        "41_vlm_critic_contract_check.py", "critic_contract_check_41"
    )

    contract = checker._base_contract(REPO_ROOT)

    assert checker._validate_contract(contract) == []
    critic_smoke = contract["diagnostic_surfaces"]["critic_eval_smoke"]
    assert (
        critic_smoke["surface_route"]
        == advantage.VLM_CRITIC_EVAL_SMOKE_DIAGNOSTIC_ROUTE
    )
    assert critic_smoke["diagnostic_only"] is True
    assert critic_smoke["mainline_authority"] is False
    assert (
        critic_smoke["authority_scope"]
        == advantage.VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE
    )


def test_45d_wrapper_summary_keeps_generic_fields_and_adds_diagnostic_fence(
    tmp_path: Path,
) -> None:
    eval_smoke = _load_script_module(
        "45d_vlm_critic_eval_smoke.py", "critic_eval_smoke_45d"
    )
    checker = _load_script_module(
        "41_vlm_critic_contract_check.py", "critic_contract_check_41_for_45d"
    )

    args = argparse.Namespace(
        advantage="None",
        n_episodes=1,
        env_name="gr00tlocomanip_g1_sim/example",
        port=49637,
        max_episode_steps=240,
        seed_base=20000,
        eval_label=eval_smoke.DEFAULT_EVAL_LABEL,
        embodiment_tag="UNITREE_G1",
    )
    upstream_summary = {
        "success_rate": 1.0,
        "success_count": 1,
        "episodes": 1,
        "episode_telemetry_jsonl": str(tmp_path / "episodes.jsonl"),
        "step_telemetry_jsonl": str(tmp_path / "steps.jsonl"),
        "wrapper_status": "ok",
        "task_text_field": "prompt_raw",
    }

    merged = eval_smoke._merge_eval_smoke_summary(
        upstream_summary=upstream_summary,
        wrapper_error=None,
        summary_json=tmp_path / "summary.json",
        args=args,
        host_for_client="127.0.0.1",
        model_path=tmp_path / "model",
        model_path_is_local_source=True,
        server_model_path=str(tmp_path / "server_model"),
        server_model_path_is_local_source=True,
        unconditional_baseline_case=False,
        baseline_local_snapshot_rewrite_applied=False,
        base_model_path="",
        overlay_from="",
        overlay_include_regex=r"^action_head\..*",
        overlay_input_source="none",
        legacy_adv_embedding_from_raw="",
        stats_from_model_path="",
        require_advantage_embedding=False,
        allow_baseline_default_advantage_embedding_init=False,
        main_repo_root=REPO_ROOT,
        python_exe=Path(sys.executable),
        bridge_info={},
        wrapper_bridge_info={},
        server_script=REPO_ROOT / "work/recap/scripts/3D_recap_run_adv_server.py",
        eval_script=REPO_ROOT / "work/recap/scripts/3D_recap_eval.py",
        server_cmd=["python3", "server.py"],
        eval_cmd=["python3", "eval.py"],
        server_started=True,
        server_log=tmp_path / "server.log",
        wrapper_log=tmp_path / "wrapper.log",
        upstream_summary_path=tmp_path / "upstream_summary.json",
        eval_returncode=0,
    )

    assert merged["success_rate"] == 1.0
    assert merged["success_count"] == 1
    assert merged["episodes"] == 1
    assert merged["episode_telemetry_jsonl"].endswith("episodes.jsonl")
    assert merged["step_telemetry_jsonl"].endswith("steps.jsonl")
    assert merged["wrapper_status"] == "ok"
    assert merged["surface_route"] == advantage.VLM_CRITIC_EVAL_SMOKE_DIAGNOSTIC_ROUTE
    assert merged["diagnostic_only"] is True
    assert merged["mainline_authority"] is False
    assert checker.validate_critic_smoke_summary_contract(merged) == []


def test_contract_checker_rejects_mainline_claiming_critic_smoke_summary() -> None:
    checker = _load_script_module(
        "41_vlm_critic_contract_check.py", "critic_contract_check_41_negative"
    )
    invalid_summary = {
        "surface_route": "vlm_critic_eval_smoke_mainline",
        "diagnostic_only": False,
        "mainline_authority": True,
        "authority_scope": "mainline",
        "authority_status": "mainline",
        "success_rate": 0.5,
        "success_count": 1,
        "episodes": 2,
        "wrapper_status": "ok",
        "compatibility_preserved_fields": ["success_rate"],
    }

    violations = checker.validate_critic_smoke_summary_contract(invalid_summary)

    assert violations
    assert any("diagnostic_only must be true" in item for item in violations)
    assert any("mainline_authority must be false" in item for item in violations)


def test_45e_gate_payload_is_explicitly_diagnostic_only() -> None:
    gate = _load_script_module(
        "45e_vlm_critic_downstream_gate.py", "critic_downstream_gate_45e"
    )

    payload = gate.build_downstream_gate_payload(
        base_rate=0.5,
        none_rate=0.48,
        zero_rate=0.40,
        pos_rate=0.60,
        retention_drop=0.02,
        gate_reasons=[],
        retention_passed=True,
        controllability_passed=True,
        critic_passed=True,
        finetune_reasons=[],
        critic_audit={"critic_dir": "/tmp/critic", "reintegrate_verdict": "ALLOW"},
        finetune_summary={
            "output_dir": "/tmp/out",
            "selected_checkpoint_path": "/tmp/out/checkpoint-4",
            "wrapper_status": "ok",
            "upstream_returncode": 0,
            "upstream_summary": {"completed_steps": 4, "max_steps": 4},
        },
        args=argparse.Namespace(max_retention_drop=0.05),
        base_none_path=Path("/tmp/base_none.json"),
        finetuned_none_path=Path("/tmp/finetuned_none.json"),
        finetuned_zero_path=Path("/tmp/finetuned_zero.json"),
        finetuned_pos_path=Path("/tmp/finetuned_pos.json"),
        finetune_path=Path("/tmp/finetune.json"),
        critic_audit_path=Path("/tmp/critic_audit.json"),
        gate_passed=True,
    )

    assert payload["gate_name"] == gate.DIAGNOSTIC_GATE_NAME
    assert payload["gate_status"] == "DIAGNOSTIC_PASS"
    assert payload["gate_semantics"] == "diagnostic_only_non_release_gate"
    assert payload["release_gate"] is False
    assert payload["diagnostic_only"] is True
    assert payload["mainline_authority"] is False
    assert payload["authority_scope"] == advantage.VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE
