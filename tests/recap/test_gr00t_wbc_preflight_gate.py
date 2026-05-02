from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_wbc_preflight_gate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _success_probe(*, repo_root: Path, add_live_paths: bool) -> dict[str, Any]:
    del repo_root, add_live_paths
    return {
        "ok": True,
        "blockers": [],
        "errors": {},
        "runtime_probe": {
            "flash_attn_2_available": True,
            "gr00t_import_ok": True,
            "gr00t_wbc_import_ok": True,
            "gymnasium_import_ok": True,
            "numpy_import_ok": True,
            "policy_client_import_ok": True,
            "robocasa_import_ok": True,
            "rollout_policy_import_ok": True,
            "server_entrypoint_exists": True,
            "torch_bfloat16_cuda_ok": True,
            "torch_cuda_available": True,
            "torch_import_ok": True,
            "video_backend_probe_ok": True,
        },
        "sys_path_injected": ["/fake/live/root"],
    }


def _blocking_probe(*, repo_root: Path, add_live_paths: bool) -> dict[str, Any]:
    del repo_root, add_live_paths
    result = _success_probe(repo_root=REPO_ROOT, add_live_paths=True)
    result["ok"] = False
    result["blockers"] = ["flash_attn_2_available", "gr00t_import_ok"]
    runtime_probe = dict(result["runtime_probe"])
    runtime_probe["flash_attn_2_available"] = False
    runtime_probe["gr00t_import_ok"] = False
    result["runtime_probe"] = runtime_probe
    result["errors"] = {
        "flash_attn_2_available": "RuntimeError: transformers reported flash-attn2 unavailable",
        "gr00t_import_ok": "ModuleNotFoundError: No module named 'gr00t'",
    }
    return result


def _readiness_success(
    args: Any,
    *,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    del args, repo_root
    return {
        "sim_imports": {
            "ok": True,
            "imports": {
                "gymnasium": True,
                "robocasa": True,
                "sync_env": True,
                "base_config": True,
                "wbc_wrapper": True,
                "sim_policy_wrapper": True,
            },
            "shim_checks": {
                "check_obj_upright_available": True,
                "visuals_utls_importable": True,
                "ik_wrapper_importable": True,
            },
            "module_files": {
                "sync_env": "/fake/sync_env.py",
                "sim_policy_wrapper": "/fake/multistep_wrapper.py",
            },
            "errors": {},
            "blockers": [],
            "sys_path_injected": ["/fake/live/root"],
        },
        "env_resolution": {
            "ok": True,
            "logical_task": "apple_to_plate_g1",
            "requested_env_name": (
                "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
            ),
            "resolved_env_name": (
                "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_g1_sim_gear_wbc"
            ),
            "alias_applied": True,
            "available_close_matches": [],
            "registered_env_count": 1,
            "registered_env_count_before_import": 0,
            "registered_env_count_after_import": 1,
            "registered_env_ids_sample": [
                "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_g1_sim_gear_wbc"
            ],
            "sync_env_module": "gr00t_wbc.control.envs.robocasa.sync_env",
            "sync_env_module_file": "/fake/sync_env.py",
        },
        "policy_ping": {
            "attempted": True,
            "ok": True,
            "host": "127.0.0.1",
            "port": 5555,
            "spawned": False,
            "reused_existing": True,
            "server_log": str(output_dir / "00_server.log"),
        },
        "action_horizon_check": {
            "attempted": True,
            "ok": True,
            "expected_policy_horizon": 30,
            "requested_smoke_n_action_steps": 20,
            "server_action_horizon": 30,
            "within_smoke_budget": True,
            "modality_config_keys": ["action", "language", "state", "video"],
        },
        "smoke": {
            "attempted": True,
            "reset_ok": True,
            "step_ok": True,
            "sample_action_kind": "dict",
            "terminated": False,
            "truncated": False,
            "reward_sample": [0.0],
        },
    }


def _make_preflight_error(
    reason_code: str,
) -> gr00t_wbc_preflight_gate.PreflightGateError:
    detail: dict[str, Any]
    blockers: list[str]
    if reason_code == "state_conditioned_env_unavailable":
        detail = {
            "code": reason_code,
            "logical_task": "apple_to_plate_g1",
            "requested_env_name": (
                "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
            ),
            "available_close_matches": [
                "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_g1_fixed_base_gear_wbc"
            ],
        }
        blockers = [reason_code]
        stage = "env_resolution"
        message = "env resolution failed"
    elif reason_code == "import_shim_issue":
        detail = {
            "errors": {
                "visuals_utls_importable": "ModuleNotFoundError: missing shim target"
            }
        }
        blockers = ["visuals_utls_importable"]
        stage = "sim_imports"
        message = "sim import or shim readiness failed"
    elif reason_code == "server_entrypoint_missing":
        detail = {
            "path": "/missing/run_gr00t_server.py",
            "exists": False,
        }
        blockers = ["server_entrypoint_missing"]
        stage = "server_entrypoint"
        message = "missing server entrypoint"
    elif reason_code == "ping_timeout":
        detail = {
            "host": "127.0.0.1",
            "port": 5555,
            "timeout_s": 600,
        }
        blockers = ["policy_ping"]
        stage = "policy_ping"
        message = "timeout waiting for ping ok after 600s"
    elif reason_code == "action_horizon_mismatch":
        detail = {
            "expected_policy_horizon": 30,
            "server_action_horizon": 20,
            "requested_smoke_n_action_steps": 20,
        }
        blockers = ["server_action_horizon"]
        stage = "action_horizon_check"
        message = "server action horizon does not match the required G1 WBC contract"
    else:
        detail = {"errors": {"flash_attn_2_available": "RuntimeError: unavailable"}}
        blockers = ["flash_attn_2_available"]
        stage = "import_probe"
        message = "runtime dependencies failed"
    return gr00t_wbc_preflight_gate.PreflightGateError(
        stage,
        message,
        reason_code=reason_code,
        detail=detail,
        blockers=blockers,
    )


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_wbc_preflight_gate.main(["--help"])
    assert exc_info.value.code == 0


def test_import_only_success_writes_machine_readable_preflight_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        gr00t_wbc_preflight_gate,
        "_collect_runtime_probe",
        _success_probe,
    )
    monkeypatch.setattr(
        gr00t_wbc_preflight_gate,
        "_run_readiness_checks",
        _readiness_success,
    )

    exit_code = gr00t_wbc_preflight_gate.main(
        ["--mode", "import-only", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact = _read_json(
        tmp_path / gr00t_wbc_preflight_gate.PREFLIGHT_REPORT_JSON_NAME
    )

    assert exit_code == 0
    assert captured.err == ""
    assert payload["status"] == "PASS"
    assert payload["reason_code"] == "ok"
    assert payload["run_mode"] == "import-only"
    assert payload["live_checks_requested"] is False
    assert payload["env_resolution"]["resolved_env_name"]
    assert payload["sim_imports"]["ok"] is True
    assert payload["artifact_path"] == str(
        tmp_path / gr00t_wbc_preflight_gate.PREFLIGHT_REPORT_JSON_NAME
    )
    assert artifact == payload


def test_smoke_success_writes_required_gate_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        gr00t_wbc_preflight_gate,
        "_collect_runtime_probe",
        _success_probe,
    )
    monkeypatch.setattr(
        gr00t_wbc_preflight_gate,
        "_run_readiness_checks",
        _readiness_success,
    )

    exit_code = gr00t_wbc_preflight_gate.main(
        ["--mode", "smoke", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact = _read_json(
        tmp_path / gr00t_wbc_preflight_gate.PREFLIGHT_REPORT_JSON_NAME
    )

    assert exit_code == 0
    assert captured.err == ""
    assert payload["status"] == "PASS"
    assert payload["reason_code"] == "ok"
    assert payload["policy_ping"]["ok"] is True
    assert payload["action_horizon_check"]["ok"] is True
    assert payload["smoke"]["step_ok"] is True
    for field_name in (
        "env_resolution",
        "sim_imports",
        "server_entrypoint",
        "policy_ping",
        "timeout_policy",
        "action_horizon_check",
        "system_break_flags",
    ):
        assert field_name in payload
    assert artifact == payload


def test_runtime_probe_blocker_returns_machine_readable_fail_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        gr00t_wbc_preflight_gate,
        "_collect_runtime_probe",
        _blocking_probe,
    )

    exit_code = gr00t_wbc_preflight_gate.main(
        ["--mode", "import-only", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    artifact = _read_json(
        tmp_path / gr00t_wbc_preflight_gate.PREFLIGHT_REPORT_JSON_NAME
    )
    failure_note = (
        tmp_path / gr00t_wbc_preflight_gate.FAILURE_NOTE_MARKDOWN_NAME
    ).read_text(encoding="utf-8")

    assert exit_code == 1
    assert captured.err == ""
    assert "Traceback" not in captured.out
    assert payload["status"] == "FAIL"
    assert payload["reason_code"] == "runtime_dependency_breakage"
    assert payload["failure"]["stage"] == "import_probe"
    assert payload["failure"]["blockers"] == [
        "flash_attn_2_available",
        "gr00t_import_ok",
    ]
    assert payload["failure_note_path"] == str(
        tmp_path / gr00t_wbc_preflight_gate.FAILURE_NOTE_MARKDOWN_NAME
    )
    assert "runtime_dependency_breakage" in failure_note
    assert artifact == payload


@pytest.mark.parametrize(
    ("reason_code", "stage"),
    [
        ("state_conditioned_env_unavailable", "env_resolution"),
        ("import_shim_issue", "sim_imports"),
        ("server_entrypoint_missing", "server_entrypoint"),
        ("ping_timeout", "policy_ping"),
        ("action_horizon_mismatch", "action_horizon_check"),
    ],
)
def test_failure_reason_codes_are_preserved_in_report_and_failure_note(
    reason_code: str,
    stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        gr00t_wbc_preflight_gate,
        "_collect_runtime_probe",
        _success_probe,
    )

    def _readiness_failure(
        args: Any,
        *,
        repo_root: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        del args, repo_root, output_dir
        raise _make_preflight_error(reason_code)

    monkeypatch.setattr(
        gr00t_wbc_preflight_gate,
        "_run_readiness_checks",
        _readiness_failure,
    )

    exit_code = gr00t_wbc_preflight_gate.main(
        ["--mode", "smoke", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    failure_note = (
        tmp_path / gr00t_wbc_preflight_gate.FAILURE_NOTE_MARKDOWN_NAME
    ).read_text(encoding="utf-8")

    assert exit_code == 1
    assert payload["status"] == "FAIL"
    assert payload["reason_code"] == reason_code
    assert payload["failure"]["stage"] == stage
    assert "Traceback" not in captured.out
    assert reason_code in failure_note
