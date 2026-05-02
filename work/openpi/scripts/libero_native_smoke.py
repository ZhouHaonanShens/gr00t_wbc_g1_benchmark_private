#!/usr/bin/env python3
"""Thin OpenPI LIBERO stock-smoke wrapper for the v2 p0 lane."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.dataloader import write_json  # noqa: E402
from work.openpi.eval import DEFAULT_STOCK_SMOKE_SCENARIO, OpenPIEvalApp  # noqa: E402
from work.openpi.eval.scenarios import StockSmokeScenario  # noqa: E402


OPENPI_ROOT = REPO_ROOT / "submodules" / "openpi"
OPENPI_VENV_PYTHON = OPENPI_ROOT / ".venv" / "bin" / "python"
TRAIN_ENTRYPOINT = OPENPI_ROOT / "scripts" / "train.py"
EVAL_ENTRYPOINT = Path(__file__).resolve()
NATIVE_DATASET_ROOT = (
    REPO_ROOT / "agent" / "artifacts" / "lerobot_datasets" / "physical_intelligence_libero_official_8d"
)
RELABELS_DATASET_ROOT = (
    REPO_ROOT
    / "agent"
    / "artifacts"
    / "lerobot_datasets"
    / "physical_intelligence_libero_official_8d_recap_relabels_v1"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "agent" / "artifacts" / "openpi_libero_recap_v2_full_update"
)
DEFAULT_RUNTIME_ROOT = (
    REPO_ROOT
    / "agent"
    / "runtime_logs"
    / "openpi_libero_recap_v2_full_update"
    / "libero_native_smoke"
)
DEFAULT_EVIDENCE_PATH = (
    REPO_ROOT / ".sisyphus" / "evidence" / "openpi_push_20260423T124010Z.md"
)
P0_SCOPE_AUDIT_DIRNAME = "p0_scope_audit"
SCOPE_AUDIT_MANIFEST_NAME = "scope_audit_manifest.json"
SMOKE_VERDICT_NAME = "libero_single_episode_smoke.json"
DATASET_PROBE_HEADING = "### Starting-state loader probe"
BLOCKED_DATASET_NOT_MATERIALIZED = "BLOCKED(dataset_not_materialized)"
DATASET_NOT_MATERIALIZED_CODE = "dataset_not_materialized"
BENCHMARK_NAME = "LIBERO"
POLICY_FAMILY = "openpi"
POLICY_ANCHOR = "pi05_libero_anchor"
CONDITIONING_MODE = "prompt_text_only"
STRICT_FULL_SUPPORTED = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libero_native_smoke.py",
        description=(
            "Emit the p0 scope-audit manifest and run the native LIBERO stock smoke "
            "only when the joined dataset is materialized."
        ),
    )
    _ = parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    _ = parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    _ = parser.add_argument(
        "--dataset-probe-evidence",
        default=str(DEFAULT_EVIDENCE_PATH),
    )
    _ = parser.add_argument(
        "--runtime-evidence-path",
        default=str(DEFAULT_EVIDENCE_PATH),
    )
    _ = parser.add_argument("--gpu", type=int, default=2)
    return parser


def _p0_scope_audit_dir(output_root: Path) -> Path:
    return output_root / P0_SCOPE_AUDIT_DIRNAME


def _scope_audit_manifest_path(output_root: Path) -> Path:
    return _p0_scope_audit_dir(output_root) / SCOPE_AUDIT_MANIFEST_NAME


def _smoke_verdict_path(output_root: Path) -> Path:
    return _p0_scope_audit_dir(output_root) / SMOKE_VERDICT_NAME


def _load_starting_state_probe(evidence_path: Path) -> dict[str, object]:
    text = evidence_path.read_text(encoding="utf-8")
    match = re.search(
        rf"{re.escape(DATASET_PROBE_HEADING)}\s*```json\s*(\{{.*?\}})\s*```",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(
            f"failed to locate JSON block under {DATASET_PROBE_HEADING!r} in {evidence_path}"
        )
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset probe payload in {evidence_path} must be a JSON object")
    probe_mapping = cast(dict[object, object], payload)
    return {str(key): value for key, value in probe_mapping.items()}


def _coerce_int(raw: object, *, default: int = 0) -> int:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        cleaned = raw.strip()
        return int(cleaned) if cleaned else default
    return default


def _coerce_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(raw, (int, float)):
        return bool(raw)
    return False


def _coerce_dataset_gate(dataset_probe: dict[str, object]) -> dict[str, object]:
    sample_file_count = _coerce_int(dataset_probe.get("sample_file_count", 0))
    materialized = _coerce_bool(dataset_probe.get("materialized", False))
    blocked = (not materialized) or sample_file_count <= 0
    reason = (
        "join dataset is not materialized yet; skipping native smoke before JAX/CUDA runtime startup"
        if blocked
        else "join dataset materialized; native smoke may execute"
    )
    return {
        "dataset_root": str(dataset_probe.get("dataset_root", "")),
        "materialized": materialized,
        "sample_file_count": sample_file_count,
        "status": "blocked" if blocked else "pass",
        "blocking_reason": DATASET_NOT_MATERIALIZED_CODE if blocked else "",
        "loader_exception_summary": (
            f"materialized={materialized}, sample_file_count={sample_file_count}"
            if blocked
            else ""
        ),
        "reason": reason,
    }


def _checkpoint_anchor_materialized(checkpoint_ref: str) -> bool:
    if checkpoint_ref.startswith("gs://"):
        return False
    return Path(checkpoint_ref).exists()


def _resolve_gpu_pci_bus_id(gpu: int) -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) == 2 and parts[0] == str(gpu):
            return parts[1]
    return ""


def _dataset_join_root_status(dataset_gate: dict[str, object]) -> dict[str, object]:
    return {
        "materialized": bool(dataset_gate["materialized"]),
        "blocking_reason": str(dataset_gate.get("blocking_reason", "")),
        "loader_exception_summary": str(
            dataset_gate.get("loader_exception_summary", "")
        ),
        "status": (
            BLOCKED_DATASET_NOT_MATERIALIZED
            if str(dataset_gate["status"]) == "blocked"
            else "materialized=true"
        ),
    }


def _planned_runtime_env(gpu: int) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "JAX_PLATFORMS": "cuda",
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
    }


def _build_wrapper_command(raw_args: list[str]) -> list[str]:
    return [str(sys.executable), str(EVAL_ENTRYPOINT), *raw_args]


def _command_shell(command: list[str]) -> str:
    return shlex.join(command)


def _scenario_payload(scenario: StockSmokeScenario) -> dict[str, object]:
    return {
        "checkpoint_ref": scenario.checkpoint_ref,
        "task_suite_name": scenario.task_suite_name,
        "task_id": scenario.task_id,
        "num_trials_per_task": scenario.num_trials_per_task,
        "seed": scenario.seed,
        "host": scenario.host,
        "port": scenario.port,
        "indicator_mode": scenario.indicator_mode,
        "runtime": {
            "artifact_root": str(scenario.runtime.artifact_root),
            "runtime_root": str(scenario.runtime.runtime_root),
            "evidence_path": str(scenario.runtime.evidence_path),
            "server_ready_timeout_s": scenario.runtime.server_ready_timeout_s,
            "client_timeout_s": scenario.runtime.client_timeout_s,
            "video_fps": scenario.runtime.video_fps,
        },
    }


def _build_smoke_execution_context(
    *,
    raw_args: list[str],
    scenario: StockSmokeScenario,
    output_root: Path,
    gpu: int,
    resolved_egl_device_pci_bus_id: str,
) -> dict[str, object]:
    command = _build_wrapper_command(raw_args)
    return {
        "command": command,
        "command_shell": _command_shell(command),
        "working_directory": str(REPO_ROOT),
        "python_executable": sys.executable,
        "script_path": str(EVAL_ENTRYPOINT),
        "planned_runtime_env": _planned_runtime_env(gpu),
        "scenario": _scenario_payload(scenario),
        "output_context": {
            "output_root": str(output_root),
            "scope_audit_manifest_path": str(_scope_audit_manifest_path(output_root)),
            "smoke_json_path": str(_smoke_verdict_path(output_root)),
            "artifact_root": str(scenario.runtime.artifact_root),
            "runtime_root": str(scenario.runtime.runtime_root),
        },
        "train_entrypoint": str(TRAIN_ENTRYPOINT),
        "eval_entrypoint": str(EVAL_ENTRYPOINT),
        "gpu": gpu,
        "resolved_egl_device_pci_bus_id": resolved_egl_device_pci_bus_id,
    }


def _build_scenario(
    *,
    output_root: Path,
    runtime_root: Path,
    runtime_evidence_path: Path,
) -> StockSmokeScenario:
    runtime = dataclasses.replace(
        DEFAULT_STOCK_SMOKE_SCENARIO.runtime,
        artifact_root=output_root / "libero_native_smoke",
        runtime_root=runtime_root,
        evidence_path=runtime_evidence_path,
    )
    return dataclasses.replace(
        DEFAULT_STOCK_SMOKE_SCENARIO,
        runtime=runtime,
    )


def _build_scope_audit_manifest(
    *,
    scenario: StockSmokeScenario,
    dataset_gate: dict[str, object],
    dataset_probe_evidence: Path,
    output_root: Path,
    gpu: int,
    resolved_egl_device_pci_bus_id: str,
) -> dict[str, object]:
    return {
        "schema_version": "openpi_libero_p0_scope_audit_v1",
        "artifact_kind": "p0_scope_audit_manifest",
        "lane": "openpi_libero_recap_v2_full_update",
        "phase": P0_SCOPE_AUDIT_DIRNAME,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "benchmark": BENCHMARK_NAME,
        "policy_family": POLICY_FAMILY,
        "policy_anchor": POLICY_ANCHOR,
        "checkpoint_anchor_materialized": _checkpoint_anchor_materialized(
            scenario.checkpoint_ref
        ),
        "dataset_native_root": str(NATIVE_DATASET_ROOT),
        "dataset_relabels_root": str(RELABELS_DATASET_ROOT),
        "dataset_join_root": str(dataset_gate["dataset_root"]),
        "dataset_join_root_status": _dataset_join_root_status(dataset_gate),
        "conditioning_mode": CONDITIONING_MODE,
        "strict_full_supported": STRICT_FULL_SUPPORTED,
        "train_entrypoint": str(TRAIN_ENTRYPOINT),
        "eval_entrypoint": str(EVAL_ENTRYPOINT),
        "gpu": gpu,
        "cuda_visible_devices": str(gpu),
        "resolved_egl_device_pci_bus_id": resolved_egl_device_pci_bus_id,
        "no_submodule_modifications": True,
        "python_executable": sys.executable,
        "expected_python_executable": str(OPENPI_VENV_PYTHON),
        "output_root": str(output_root),
        "scope_audit_dir": str(_p0_scope_audit_dir(output_root)),
        "dataset_probe_evidence": str(dataset_probe_evidence),
        "dataset_gate": dataset_gate,
        "scenario": _scenario_payload(scenario),
        "verdict": (
            BLOCKED_DATASET_NOT_MATERIALIZED
            if dataset_gate["status"] == "blocked"
            else "READY(native_smoke)"
        ),
    }


def _build_blocked_smoke_payload(
    *,
    raw_args: list[str],
    scenario: StockSmokeScenario,
    dataset_gate: dict[str, object],
    output_root: Path,
    gpu: int,
    resolved_egl_device_pci_bus_id: str,
) -> dict[str, object]:
    return {
        "schema_version": "openpi_libero_single_episode_smoke_v1",
        "artifact_kind": "libero_single_episode_smoke",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": BLOCKED_DATASET_NOT_MATERIALIZED,
        "verdict": BLOCKED_DATASET_NOT_MATERIALIZED,
        "skip_before_execute": True,
        "runtime_started": False,
        "server_started": False,
        "client_started": False,
        "process_status": "blocked_before_execute",
        "exit_code": 0,
        "reason": dataset_gate["reason"],
        "blocker_code": DATASET_NOT_MATERIALIZED_CODE,
        "failure": {
            "kind": "blocker",
            "blocker_code": DATASET_NOT_MATERIALIZED_CODE,
            "blocker_reason": dataset_gate["reason"],
            "blocking_reason": dataset_gate.get("blocking_reason", ""),
            "loader_exception_summary": dataset_gate.get(
                "loader_exception_summary", ""
            ),
        },
        "gpu": gpu,
        "cuda_visible_devices": str(gpu),
        "resolved_egl_device_pci_bus_id": resolved_egl_device_pci_bus_id,
        "output_root": str(output_root),
        "dataset_gate": dataset_gate,
        "command_context": _build_smoke_execution_context(
            raw_args=raw_args,
            scenario=scenario,
            output_root=output_root,
            gpu=gpu,
            resolved_egl_device_pci_bus_id=resolved_egl_device_pci_bus_id,
        ),
        "runtime_output": {
            "stdout": "",
            "stderr": "",
            "runtime_log_path": None,
            "subprocesses_started": [],
            "downstream_runtime_command": None,
            "downstream_runtime_command_shell": None,
        },
        "python": {
            "observed": sys.executable,
            "expected": str(OPENPI_VENV_PYTHON),
        },
    }


def _build_executed_smoke_payload(
    *,
    raw_args: list[str],
    scenario: StockSmokeScenario,
    dataset_gate: dict[str, object],
    output_root: Path,
    gpu: int,
    exit_code: int,
    resolved_egl_device_pci_bus_id: str,
) -> dict[str, object]:
    status = "PASS" if exit_code == 0 else "FAIL"
    return {
        "schema_version": "openpi_libero_single_episode_smoke_v1",
        "artifact_kind": "libero_single_episode_smoke",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "verdict": status,
        "skip_before_execute": False,
        "runtime_started": True,
        "server_started": True,
        "client_started": True,
        "process_status": "executed",
        "exit_code": exit_code,
        "gpu": gpu,
        "cuda_visible_devices": str(gpu),
        "resolved_egl_device_pci_bus_id": resolved_egl_device_pci_bus_id,
        "output_root": str(output_root),
        "dataset_gate": dataset_gate,
        "command_context": _build_smoke_execution_context(
            raw_args=raw_args,
            scenario=scenario,
            output_root=output_root,
            gpu=gpu,
            resolved_egl_device_pci_bus_id=resolved_egl_device_pci_bus_id,
        ),
        "runtime_output": {
            "stdout": "",
            "stderr": "",
            "runtime_log_path": None,
            "subprocesses_started": ["openpi_eval_app"],
            "downstream_runtime_command": None,
            "downstream_runtime_command_shell": None,
        },
        "python": {
            "observed": sys.executable,
            "expected": str(OPENPI_VENV_PYTHON),
        },
    }


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else list(sys.argv[1:])
    args = _build_parser().parse_args(raw_args)
    gpu = int(args.gpu)
    if gpu != 2:
        raise ValueError("this p0 authority root is pinned to gpu=2")

    output_root = Path(str(args.output_root)).resolve()
    runtime_root = Path(str(args.runtime_root)).resolve()
    dataset_probe_evidence = Path(str(args.dataset_probe_evidence)).resolve()
    runtime_evidence_path = Path(str(args.runtime_evidence_path)).resolve()

    scenario = _build_scenario(
        output_root=output_root,
        runtime_root=runtime_root,
        runtime_evidence_path=runtime_evidence_path,
    )
    dataset_probe = _load_starting_state_probe(dataset_probe_evidence)
    dataset_gate = _coerce_dataset_gate(dataset_probe)
    resolved_egl_device_pci_bus_id = _resolve_gpu_pci_bus_id(gpu)

    manifest = _build_scope_audit_manifest(
        scenario=scenario,
        dataset_gate=dataset_gate,
        dataset_probe_evidence=dataset_probe_evidence,
        output_root=output_root,
        gpu=gpu,
        resolved_egl_device_pci_bus_id=resolved_egl_device_pci_bus_id,
    )
    write_json(_scope_audit_manifest_path(output_root), manifest)

    if dataset_gate["status"] == "blocked":
        write_json(
            _smoke_verdict_path(output_root),
            _build_blocked_smoke_payload(
                raw_args=raw_args,
                scenario=scenario,
                dataset_gate=dataset_gate,
                output_root=output_root,
                gpu=gpu,
                resolved_egl_device_pci_bus_id=resolved_egl_device_pci_bus_id,
            ),
        )
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    exit_code = OpenPIEvalApp().run_stock_smoke(scenario)
    write_json(
        _smoke_verdict_path(output_root),
        _build_executed_smoke_payload(
            raw_args=raw_args,
            scenario=scenario,
            dataset_gate=dataset_gate,
            output_root=output_root,
            gpu=gpu,
            exit_code=exit_code,
            resolved_egl_device_pci_bus_id=resolved_egl_device_pci_bus_id,
        ),
    )
    return exit_code


__all__ = [
    "BLOCKED_DATASET_NOT_MATERIALIZED",
    "DEFAULT_EVIDENCE_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_RUNTIME_ROOT",
    "OPENPI_VENV_PYTHON",
    "P0_SCOPE_AUDIT_DIRNAME",
    "SCOPE_AUDIT_MANIFEST_NAME",
    "SMOKE_VERDICT_NAME",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
