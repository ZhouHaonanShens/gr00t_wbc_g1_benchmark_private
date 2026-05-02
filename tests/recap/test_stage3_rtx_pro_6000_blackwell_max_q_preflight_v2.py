from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any


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


def _repair_summary_payload(
    *,
    repo_root: Path,
    summary_path: Path,
    log_path: Path,
    delegate_runtime_python: str,
) -> dict[str, object]:
    summary_rel = stage3_baseline_train_gate._repo_relative_path(
        repo_root, summary_path
    )
    log_rel = stage3_baseline_train_gate._repo_relative_path(repo_root, log_path)
    return {
        "artifacts": {
            "session_log": log_rel,
            "summary_json": summary_rel,
        },
        "checked_at": "2026-04-18T17:00:00+00:00",
        "delegate_runtime_python": delegate_runtime_python,
        "delegate_runtime_python_realpath": "/real/python",
        "exit_code": 0,
        "final_health": {
            "healthy": True,
            "torch_cuda_available": True,
            "torch_cuda_arch_list": ["sm_86", "sm_120"],
            "flash_attn_2_available": True,
            "flash_attn_2_cuda_import_ok": True,
        },
        "final_probe": {
            "command": [delegate_runtime_python, "-c", "print('probe')"],
            "payload": {"python_executable": delegate_runtime_python},
            "returncode": 0,
            "stderr": "",
            "stdout": "{}",
        },
        "status": "healthy_noop",
    }


def _make_preflight_context(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], Path, dict[str, Any]]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        repo_root
        / "agent/artifacts/stage3_iteration/recap_stage3_iter_002/iteration_manifest.json"
    )
    delegate_runtime_python = str(
        repo_root / "submodules/fake_delegate/.venv/bin/python"
    )
    manifest_payload = {
        "schema_version": "stage3_iteration_manifest_v3",
        "artifact_root": "agent/artifacts/stage3_iteration/recap_stage3_iter_002/",
        "hardware_profile": "rtx_pro_6000_blackwell_max_q_96g_x4",
        "delegate_runtime_python": delegate_runtime_python,
        "orchestrator_python": str(repo_root / ".venv/bin/python"),
        "collect_policy_ckpt_decision": "baseline_train_required",
    }
    _write_json(manifest_path, manifest_payload)

    preflight_path = (
        manifest_path.parent / "rtx_pro_6000_blackwell_max_q_preflight.json"
    )
    existing_preflight_payload = {
        "schema_version": "stage3_a6000_preflight_v1",
        "artifact_kind": "stage3_a6000_preflight",
        "artifact_path": stage3_baseline_train_gate._repo_relative_path(
            repo_root,
            preflight_path,
        ),
        "delegate_runtime_python": "/home/howard/.local/share/uv/python/cpython/bin/python3.10",
        "pass": False,
        "status": "execution_hard_block",
    }
    _write_json(preflight_path, existing_preflight_payload)

    runtime_log_dir = repo_root / "agent/runtime_logs/stage3_delegate_runtime_repair"
    summary_path = (
        runtime_log_dir / "stage3_delegate_runtime_repair_20260418_170000.json"
    )
    log_path = runtime_log_dir / "stage3_delegate_runtime_repair_20260418_170000.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("repair ok\n", encoding="utf-8")
    _write_json(
        summary_path,
        _repair_summary_payload(
            repo_root=repo_root,
            summary_path=summary_path,
            log_path=log_path,
            delegate_runtime_python=delegate_runtime_python,
        ),
    )
    return (
        repo_root,
        manifest_path,
        manifest_payload,
        preflight_path,
        existing_preflight_payload,
    )


def _install_live_probe_stubs(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_collect_duplicate_processes",
        lambda *, current_pid: [],
    )
    monkeypatch.setattr(
        stage3_baseline_train_gate.shutil,
        "disk_usage",
        lambda _path: type("DiskUsage", (), {"free": 120 * 1024**3})(),
    )
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_query_gpu_inventory",
        lambda: (
            0,
            {
                "cmd": ["nvidia-smi"],
                "gpus": [
                    {
                        "index": index,
                        "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                        "memory_total_mib": 97887,
                        "memory_free_mib": 64000,
                    }
                    for index in range(4)
                ],
            },
        ),
    )
    monkeypatch.setattr(
        stage3_baseline_train_gate,
        "_run_json_command",
        lambda *, cmd, cwd: (
            0,
            {
                "returncode": 0,
                "cmd": list(cmd),
                "torch_import_ok": True,
                "torch_cuda_available": True,
                "torch_cuda_arch_list": ["sm_86", "sm_120"],
                "flash_attn_2_available": True,
                "flash_attn_2_cuda_import_ok": True,
                "transformers_error": None,
                "flash_attn_2_cuda_error": None,
            },
        ),
    )


def test_run_rtx_pro_6000_blackwell_max_q_preflight_v2_tracks_current_workstation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path, manifest_payload, preflight_path, _existing = (
        _make_preflight_context(tmp_path)
    )
    _install_live_probe_stubs(monkeypatch)

    payload = stage3_baseline_train_gate._run_rtx_pro_6000_blackwell_max_q_preflight(
        repo_root=repo_root,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        preflight_path=preflight_path,
        training_python_contract={
            "manifest_path": str(manifest_path),
            "orchestrator_python": str(manifest_payload["orchestrator_python"]),
            "delegate_runtime_python": str(manifest_payload["delegate_runtime_python"]),
        },
    )

    superseded_outputs_path = preflight_path.parent / "superseded_outputs.json"
    superseded = json.loads(superseded_outputs_path.read_text(encoding="utf-8"))

    assert (
        payload["schema_version"] == "stage3_rtx_pro_6000_blackwell_max_q_preflight_v2"
    )
    assert (
        payload["delegate_runtime_python_manifest"]
        == manifest_payload["delegate_runtime_python"]
    )
    assert (
        payload["probe_commands"]["delegate_runtime_probe"][0]
        == manifest_payload["delegate_runtime_python"]
    )
    assert payload["delegate_runtime_health"]["pass"] is True
    assert payload["delegate_runtime_health"]["argv0_matches_manifest"] is True
    assert payload["hardware_profile_match"]["pass"] is True
    assert payload["repair_tool_status"] == "healthy_noop"
    assert payload["hard_block_subfamily"] is None
    assert payload["next_action"] == "user_escalation_required"
    assert payload["evidence"]["delegate_runtime_repair_summary_sha256"] == (
        stage3_baseline_train_gate._sha256_file(
            repo_root / payload["evidence"]["delegate_runtime_repair_summary_path"]
        )
    )
    assert payload["evidence"]["delegate_runtime_repair_log_sha256"] == (
        stage3_baseline_train_gate._sha256_file(
            repo_root / payload["evidence"]["delegate_runtime_repair_log"]
        )
    )
    assert superseded["schema_version"] == "stage3_superseded_outputs_v1"
    assert superseded["superseded_outputs"][0]["superseded_schema_version"] == (
        "stage3_a6000_preflight_v1"
    )


def test_validate_preflight_authority_fails_closed_on_repair_summary_sha_mismatch(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path, manifest_payload, preflight_path, existing_preflight = (
        _make_preflight_context(tmp_path)
    )
    existing_preflight_sha = stage3_baseline_train_gate._sha256_file(preflight_path)
    _install_live_probe_stubs(monkeypatch)
    payload = stage3_baseline_train_gate._run_rtx_pro_6000_blackwell_max_q_preflight(
        repo_root=repo_root,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        preflight_path=preflight_path,
        training_python_contract={
            "manifest_path": str(manifest_path),
            "orchestrator_python": str(manifest_payload["orchestrator_python"]),
            "delegate_runtime_python": str(manifest_payload["delegate_runtime_python"]),
        },
    )
    tampered = json.loads(json.dumps(payload))
    tampered["evidence"]["delegate_runtime_repair_summary_sha256"] = "0" * 64

    try:
        stage3_baseline_train_gate._validate_preflight_authority(
            repo_root=repo_root,
            manifest_path=manifest_path,
            manifest_payload=manifest_payload,
            superseded_outputs_path=preflight_path.parent / "superseded_outputs.json",
            superseded_outputs_payload=json.loads(
                (preflight_path.parent / "superseded_outputs.json").read_text(
                    encoding="utf-8"
                )
            ),
            preflight_payload=tampered,
            existing_preflight_path=preflight_path,
            existing_preflight_payload=existing_preflight,
            existing_preflight_sha256=existing_preflight_sha,
        )
    except ValueError as exc:
        assert "summary sha256 binding mismatch" in str(exc)
    else:
        raise AssertionError("expected repair summary sha256 mismatch to fail-close")


def test_validate_preflight_authority_requires_v1_supersede_before_trusting_v2(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root, manifest_path, manifest_payload, preflight_path, existing_preflight = (
        _make_preflight_context(tmp_path)
    )
    existing_preflight_sha = stage3_baseline_train_gate._sha256_file(preflight_path)
    _install_live_probe_stubs(monkeypatch)
    payload = stage3_baseline_train_gate._run_rtx_pro_6000_blackwell_max_q_preflight(
        repo_root=repo_root,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        preflight_path=preflight_path,
        training_python_contract={
            "manifest_path": str(manifest_path),
            "orchestrator_python": str(manifest_payload["orchestrator_python"]),
            "delegate_runtime_python": str(manifest_payload["delegate_runtime_python"]),
        },
    )
    _write_json(
        preflight_path.parent / "superseded_outputs.json",
        {
            "schema_version": "stage3_superseded_outputs_v1",
            "artifact_kind": "stage3_superseded_outputs",
            "superseded_outputs": [],
        },
    )

    try:
        stage3_baseline_train_gate._validate_preflight_authority(
            repo_root=repo_root,
            manifest_path=manifest_path,
            manifest_payload=manifest_payload,
            superseded_outputs_path=preflight_path.parent / "superseded_outputs.json",
            superseded_outputs_payload=json.loads(
                (preflight_path.parent / "superseded_outputs.json").read_text(
                    encoding="utf-8"
                )
            ),
            preflight_payload=payload,
            existing_preflight_path=preflight_path,
            existing_preflight_payload=existing_preflight,
            existing_preflight_sha256=existing_preflight_sha,
        )
    except ValueError as exc:
        assert "must be superseded before trusting v2" in str(exc)
    else:
        raise AssertionError("expected missing v1 supersede record to fail-close")
