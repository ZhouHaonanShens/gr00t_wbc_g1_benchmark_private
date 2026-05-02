from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    _ = sys.path.insert(0, str(REPO_ROOT))


from work.openpi.scripts import libero_native_smoke  # noqa: E402


def _write_probe_evidence(path: Path) -> None:
    payload = "\n".join(
        [
            "# OpenPI Push Evidence",
            "",
            "### Starting-state loader probe",
            "```json",
            "{",
            '  "dataset_root": "/tmp/join_dataset",',
            '  "materialized": false,',
            '  "sample_file_count": 0',
            "}",
            "```",
        ]
    )
    _ = path.write_text(payload + "\n", encoding="utf-8")


def test_main_writes_blocked_verdict_before_runtime_execution(tmp_path: Path) -> None:
    evidence_path = tmp_path / "openpi_push.md"
    _write_probe_evidence(evidence_path)
    output_root = tmp_path / "openpi_libero_recap_v2_full_update"
    runtime_root = tmp_path / "runtime_logs"
    runtime_evidence_path = tmp_path / "runtime_evidence.md"

    exit_code = libero_native_smoke.main(
        [
            "--output-root",
            str(output_root),
            "--runtime-root",
            str(runtime_root),
            "--dataset-probe-evidence",
            str(evidence_path),
            "--runtime-evidence-path",
            str(runtime_evidence_path),
        ]
    )

    assert exit_code == 0
    manifest = cast(
        dict[str, Any],
        json.loads(
            (output_root / "p0_scope_audit" / "scope_audit_manifest.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    smoke = cast(
        dict[str, Any],
        json.loads(
            (output_root / "p0_scope_audit" / "libero_single_episode_smoke.json").read_text(
                encoding="utf-8"
            )
        ),
    )

    required_manifest_keys = {
        "benchmark",
        "policy_family",
        "policy_anchor",
        "checkpoint_anchor_materialized",
        "dataset_native_root",
        "dataset_relabels_root",
        "dataset_join_root",
        "dataset_join_root_status",
        "conditioning_mode",
        "strict_full_supported",
        "train_entrypoint",
        "eval_entrypoint",
        "gpu",
        "resolved_egl_device_pci_bus_id",
        "no_submodule_modifications",
    }
    assert required_manifest_keys.issubset(manifest)
    assert manifest["benchmark"] == "LIBERO"
    assert manifest["policy_family"] == "openpi"
    assert manifest["policy_anchor"] == "pi05_libero_anchor"
    assert manifest["checkpoint_anchor_materialized"] is False
    assert manifest["conditioning_mode"] == "prompt_text_only"
    assert manifest["strict_full_supported"] is False
    assert manifest["dataset_gate"]["materialized"] is False
    assert manifest["dataset_gate"]["sample_file_count"] == 0
    assert manifest["dataset_join_root"] == "/tmp/join_dataset"
    assert manifest["dataset_join_root_status"]["materialized"] is False
    assert (
        manifest["dataset_join_root_status"]["blocking_reason"]
        == "dataset_not_materialized"
    )
    assert manifest["verdict"] == "BLOCKED(dataset_not_materialized)"
    assert manifest["no_submodule_modifications"] is True
    assert manifest["train_entrypoint"].endswith("submodules/openpi/scripts/train.py")
    assert manifest["eval_entrypoint"].endswith(
        "work/openpi/scripts/libero_native_smoke.py"
    )
    assert isinstance(manifest["resolved_egl_device_pci_bus_id"], str)
    assert manifest["scenario"]["runtime"]["artifact_root"] == str(
        output_root / "libero_native_smoke"
    )
    assert manifest["scenario"]["runtime"]["runtime_root"] == str(runtime_root)

    required_smoke_keys = {
        "status",
        "verdict",
        "skip_before_execute",
        "runtime_started",
        "blocker_code",
        "failure",
        "command_context",
        "runtime_output",
        "resolved_egl_device_pci_bus_id",
    }
    assert required_smoke_keys.issubset(smoke)
    assert smoke["status"] == "BLOCKED(dataset_not_materialized)"
    assert smoke["verdict"] == "BLOCKED(dataset_not_materialized)"
    assert smoke["skip_before_execute"] is True
    assert smoke["runtime_started"] is False
    assert smoke["server_started"] is False
    assert smoke["client_started"] is False
    assert smoke["exit_code"] == 0
    assert smoke["blocker_code"] == "dataset_not_materialized"
    assert smoke["failure"]["blocking_reason"] == "dataset_not_materialized"
    assert smoke["command_context"]["command"][0] == sys.executable
    assert smoke["command_context"]["eval_entrypoint"].endswith(
        "work/openpi/scripts/libero_native_smoke.py"
    )
    assert smoke["command_context"]["planned_runtime_env"]["MUJOCO_GL"] == "egl"
    assert smoke["command_context"]["planned_runtime_env"]["MUJOCO_EGL_DEVICE_ID"] == "2"
    assert smoke["runtime_output"]["downstream_runtime_command"] is None
    assert smoke["python"]["expected"] == str(libero_native_smoke.OPENPI_VENV_PYTHON)
