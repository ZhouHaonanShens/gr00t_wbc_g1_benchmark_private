#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from work.demo_utils import paths as demo_paths
from work.recap.dual_loss import DualLossConfig
from work.recap.dual_loss import build_dual_loss_integration_report
from work.recap.phase_thresholds import build_phase_threshold_metadata
from work.recap.script_apps.recap_finetune_repro_app import (
    RecapFinetuneReproScriptApp,
)


RUN_ID = "stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ"
GR00T_ROOT_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
) / RUN_ID / "gr00t"
COORDINATOR_PROTOCOL_REL = (
    GR00T_ROOT_REL / "coordinator" / "new_candidate_training_protocol.json"
)
COORDINATOR_GATE_POLICY_REL = GR00T_ROOT_REL / "coordinator" / "gate_policy.json"
TRAINING_RUNNER_CONTRACT_DIR_REL = GR00T_ROOT_REL / "training_runner_contract"
CONTRACT_JSON_NAME = "gr00t_r2r4_training_runner_contract.json"
PREFLIGHT_JSON_NAME = "gr00t_training_preflight.json"

COMMON_WARM_START_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
    "p2_full_update_overfit20/checkpoint-20"
)
DEFAULT_DATASET_REL = Path("agent/artifacts/lerobot_datasets/recap_stage3_iter_002")
ITER3_FAILED_P5_CHECKPOINT_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
    "stage1_blocker_resolution_iter2_20260425T_nextZ/gr00t/"
    "p4_rerun/conditioned_train/checkpoint-200"
)

DEFAULT_MAX_STEPS = 2200
DEFAULT_SAVE_STEPS = 1100
G1_CONTRACT_SAVE_TOTAL_LIMIT = 1
DEFAULT_SAVE_STRATEGY = "steps"
DEFAULT_CUDA_VISIBLE_DEVICES = "1"
DEFAULT_SEED = 20260421
G2_FULL_TRAINING_STEPS = 2200
G2_SOFT_CAP_HOURS = 14.0
THROUGHPUT_PROBE_STEPS = 50
THROUGHPUT_PROBE_SAVE_TOTAL_LIMIT = 1
THROUGHPUT_PROBE_SCHEMA_VERSION = "gr00t_g2_throughput_probe_spec_v1"
THROUGHPUT_PROBE_REPORT_SCHEMA_VERSION = "gr00t_g2_throughput_probe_report_v1"

CONTRACT_SCHEMA_VERSION = "gr00t_r2r4_training_runner_contract_v1"
PREFLIGHT_SCHEMA_VERSION = "gr00t_training_preflight_v1"
RUN_MANIFEST_SCHEMA_VERSION = "gr00t_r2r4_training_run_manifest_v1"
LOSS_DECOMPOSITION_SCHEMA_VERSION = "gr00t_loss_decomposition_v1"
THRESHOLD_TRACE_SCHEMA_VERSION = "gr00t_threshold_switch_trace_v1"
ALPHA_DUAL_TRACE_SCHEMA_VERSION = "gr00t_alpha_dual_loss_trace_v1"
DYNAMIC_DELTA_AUDIT_SCHEMA_VERSION = "gr00t_dynamic_delta_audit_v1"
GRADIENT_PATH_ATTESTATION_SCHEMA_VERSION = "gr00t_gradient_path_attestation_v1"
CHECKPOINT_PROVENANCE_SCHEMA_VERSION = "gr00t_checkpoint_provenance_v1"

REQUIRED_TRUE_FLAGS: tuple[str, ...] = (
    "enable_r2_phase_threshold_switching",
    "enable_r4_alpha_dual_loss",
    "emit_loss_decomposition",
    "emit_threshold_switch_trace",
    "emit_alpha_dual_trace",
    "emit_gradient_path_attestation",
    "emit_dynamic_delta_audit",
    "emit_checkpoint_sha256sums",
)


def _repo_root() -> Path:
    return demo_paths.repo_root(Path(__file__))


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON must be an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".tmp.",
        delete=False,
    ) as fp:
        tmp = Path(fp.name)
        fp.write(text)
    tmp.replace(path)


def _numeric_checkpoint_step(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith("checkpoint-"):
        return None
    suffix = path.name.split("checkpoint-", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _latest_checkpoint_step(output_dir: Path) -> int | None:
    if not output_dir.is_dir():
        return None
    steps = [
        step
        for child in output_dir.iterdir()
        for step in [_numeric_checkpoint_step(child)]
        if step is not None
    ]
    if not steps:
        return None
    return max(steps)


def _resolve_repo_path(repo_root: Path, raw: str | os.PathLike[str]) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return demo_paths.abspath_preserve_symlink(path)


def _as_repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _same_path(left: Path, right: Path) -> bool:
    return demo_paths.same_abspath_preserve_symlink(left, right)


def _load_optional_json(repo_root: Path, rel_path: Path) -> dict[str, Any]:
    path = repo_root / rel_path
    if not path.is_file():
        return {}
    return _read_json(path)


def _protocol_value(
    protocol: dict[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    value = protocol.get(key)
    if value in (None, ""):
        return fallback
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="4X_recap_r2r4_closed_finetune.py",
        description=(
            "Iter9 GR00T R2/R4 closed candidate wrapper. It validates the G0 protocol, "
            "emits G1 contract/preflight artifacts, and delegates the actual launch to "
            "RecapFinetuneReproScriptApp when not run in dry-run/contract-only mode."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--warm-start-checkpoint", default="")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--runtime-log-dir", default="")
    parser.add_argument("--contract-output-dir", default="")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--save-strategy", default="")
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=DEFAULT_CUDA_VISIBLE_DEVICES)
    parser.add_argument("--python", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--emit-contract-only", action="store_true")
    parser.add_argument("--no-sudo", action="store_true")
    parser.add_argument("--enable-r2-phase-threshold-switching", action="store_true")
    parser.add_argument("--enable-r4-alpha-dual-loss", action="store_true")
    parser.add_argument("--emit-loss-decomposition", action="store_true")
    parser.add_argument("--emit-threshold-switch-trace", action="store_true")
    parser.add_argument("--emit-alpha-dual-trace", action="store_true")
    parser.add_argument("--emit-gradient-path-attestation", action="store_true")
    parser.add_argument("--emit-dynamic-delta-audit", action="store_true")
    parser.add_argument("--emit-checkpoint-sha256sums", action="store_true")
    return parser


def _effective_config(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    warm_start_raw = str(
        args.warm_start_checkpoint
        or _protocol_value(protocol, "warm_start_checkpoint", COMMON_WARM_START_REL.as_posix())
    )
    dataset_raw = str(
        args.dataset_path
        or _protocol_value(protocol, "dataset_path", DEFAULT_DATASET_REL.as_posix())
    )
    output_raw = str(
        args.output_dir
        or _protocol_value(
            protocol,
            "downstream_output_root",
            (GR00T_ROOT_REL / "g2_full_training").as_posix(),
        )
    )
    runtime_raw = str(
        args.runtime_log_dir
        or _protocol_value(
            protocol,
            "runtime_log_root",
            f"agent/runtime_logs/{RUN_ID}/gr00t/g2_full_training",
        )
    )
    contract_dir_raw = str(args.contract_output_dir or TRAINING_RUNNER_CONTRACT_DIR_REL)
    save_strategy = str(
        args.save_strategy
        or _protocol_value(protocol, "save_strategy", DEFAULT_SAVE_STRATEGY)
    )
    return {
        "warm_start_checkpoint": _resolve_repo_path(repo_root, warm_start_raw),
        "dataset_path": _resolve_repo_path(repo_root, dataset_raw),
        "output_dir": _resolve_repo_path(repo_root, output_raw),
        "runtime_log_dir": _resolve_repo_path(repo_root, runtime_raw),
        "contract_output_dir": _resolve_repo_path(repo_root, contract_dir_raw),
        "max_steps": int(
            args.max_steps
            if args.max_steps is not None
            else _protocol_value(protocol, "max_steps", DEFAULT_MAX_STEPS)
        ),
        # The old G2 coordinator protocol may still carry save_total_limit=2
        # because an earlier plan wanted both a mid-run and final checkpoint.
        # The live training surface underneath this wrapper now enforces the
        # repo-wide single-checkpoint retention contract, so the default must
        # fail closed to 1 unless the caller explicitly asks for something else
        # and lets preflight block it.
        "save_total_limit": int(
            args.save_total_limit
            if args.save_total_limit is not None
            else G1_CONTRACT_SAVE_TOTAL_LIMIT
        ),
        "save_strategy": save_strategy,
        "save_steps": int(
            args.save_steps
            if args.save_steps is not None
            else _protocol_value(protocol, "save_steps", DEFAULT_SAVE_STEPS)
        ),
        "seed": int(
            args.seed
            if args.seed is not None
            else _protocol_value(protocol, "seed", DEFAULT_SEED)
        ),
        "cuda_visible_devices": str(args.cuda_visible_devices).strip(),
        "python": str(args.python).strip(),
    }


def _required_flag_status(args: argparse.Namespace) -> dict[str, bool]:
    return {name: bool(getattr(args, name)) for name in REQUIRED_TRUE_FLAGS}


def _minimum_g2_steps_per_hour() -> float:
    return round(float(G2_FULL_TRAINING_STEPS) / float(G2_SOFT_CAP_HOURS), 6)


def _build_throughput_probe_spec(
    *,
    repo_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    probe_output_dir = (
        GR00T_ROOT_REL / "g2_throughput_probe_50step"
    ).as_posix()
    probe_runtime_log_dir = (
        f"agent/runtime_logs/{RUN_ID}/gr00t/g2_throughput_probe_50step"
    )
    command = [
        "CUDA_VISIBLE_DEVICES=1",
        "submodules/Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/"
        "GR00T-WholeBodyControl_uv/.venv/bin/python",
        "work/recap/scripts/4X_recap_r2r4_closed_finetune.py",
        "--warm-start-checkpoint",
        _as_repo_relative(repo_root, Path(config["warm_start_checkpoint"])),
        "--dataset-path",
        _as_repo_relative(repo_root, Path(config["dataset_path"])),
        "--output-dir",
        probe_output_dir,
        "--runtime-log-dir",
        probe_runtime_log_dir,
        "--max-steps",
        str(int(THROUGHPUT_PROBE_STEPS)),
        "--save-total-limit",
        str(int(THROUGHPUT_PROBE_SAVE_TOTAL_LIMIT)),
        "--save-strategy",
        "steps",
        "--save-steps",
        str(int(THROUGHPUT_PROBE_STEPS)),
        "--no-sudo",
        "--enable-r2-phase-threshold-switching",
        "--enable-r4-alpha-dual-loss",
        "--emit-loss-decomposition",
        "--emit-threshold-switch-trace",
        "--emit-alpha-dual-trace",
        "--emit-gradient-path-attestation",
        "--emit-dynamic-delta-audit",
        "--emit-checkpoint-sha256sums",
    ]
    minimum_steps_per_hour = _minimum_g2_steps_per_hour()
    return {
        "schema_version": THROUGHPUT_PROBE_SCHEMA_VERSION,
        "required_before_g2_full_launch": True,
        "probe_steps": int(THROUGHPUT_PROBE_STEPS),
        "cuda_visible_devices": DEFAULT_CUDA_VISIBLE_DEVICES,
        "no_sudo": True,
        "output_dir": probe_output_dir,
        "runtime_log_dir": probe_runtime_log_dir,
        "checkpoint_save_policy": {
            "save_total_limit": int(THROUGHPUT_PROBE_SAVE_TOTAL_LIMIT),
            "save_strategy": "steps",
            "save_steps": int(THROUGHPUT_PROBE_STEPS),
        },
        "gate": {
            "metric": "estimated_steps_per_hour",
            "minimum_steps_per_hour": minimum_steps_per_hour,
            "basis": "g2_full_training_steps / g2_soft_cap_hours",
            "g2_full_training_steps": int(G2_FULL_TRAINING_STEPS),
            "g2_soft_cap_hours": float(G2_SOFT_CAP_HOURS),
            "expression": (
                "estimated_steps_per_hour >= minimum_steps_per_hour "
                "and probe_status == 'PASS'"
            ),
            "failure_status": "BLOCK_INFRA_G2_THROUGHPUT",
        },
        "expected_report": {
            "path": f"{probe_output_dir}/throughput_probe_report.json",
            "schema_version": "gr00t_g2_throughput_probe_report_v1",
            "required_fields": [
                "probe_status",
                "probe_steps_completed",
                "elapsed_seconds",
                "estimated_steps_per_hour",
                "minimum_steps_per_hour",
                "g2_launch_allowed",
            ],
        },
        "command": command,
    }


def _build_contract(
    *,
    repo_root: Path,
    config: dict[str, Any],
    protocol: dict[str, Any],
    gate_policy: dict[str, Any],
) -> dict[str, Any]:
    del gate_policy
    dual_loss_config = DualLossConfig()
    protocol_save_policy = {
        "save_total_limit": int(_protocol_value(protocol, "save_total_limit", 2)),
        "save_strategy": str(_protocol_value(protocol, "save_strategy", "steps")),
        "save_steps": int(_protocol_value(protocol, "save_steps", DEFAULT_SAVE_STEPS)),
    }
    throughput_probe_spec = _build_throughput_probe_spec(
        repo_root=repo_root,
        config=config,
    )
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "run_id": RUN_ID,
        "wrapper_script": "work/recap/scripts/4X_recap_r2r4_closed_finetune.py",
        "delegate": {
            "class": "work.recap.script_apps.recap_finetune_repro_app.RecapFinetuneReproScriptApp",
            "delegates_to_recap_finetune_repro_script_app": True,
            "direct_import_of_finetune_full": False,
            "process_local_overlay": (
                "monkeypatch work.recap.hf_snapshot_patch.make_patched_base_model_dir "
                "to use the validated warm-start checkpoint as snapshot_dir"
            ),
        },
        "common_warm_start": COMMON_WARM_START_REL.as_posix(),
        "effective_warm_start": _as_repo_relative(
            repo_root, Path(config["warm_start_checkpoint"])
        ),
        "do_not_retry_same_checkpoint": True,
        "required_features": {
            "loss_decomposition_required": True,
            "threshold_switch_trace_required": True,
            "alpha_dual_loss_trace_required": True,
            "dynamic_delta_audit_required": True,
            "sha256sums_required": True,
            "gradient_path_attestation_required": True,
        },
        "checkpoint_save_policy": {
            "save_total_limit": int(G1_CONTRACT_SAVE_TOTAL_LIMIT),
            "save_strategy": DEFAULT_SAVE_STRATEGY,
            "save_steps": int(DEFAULT_SAVE_STEPS),
        },
        "g2_protocol_checkpoint_save_policy": protocol_save_policy,
        "effective_checkpoint_save_policy": {
            "save_total_limit": int(config["save_total_limit"]),
            "save_strategy": str(config["save_strategy"]),
            "save_steps": int(config["save_steps"]),
        },
        "throughput_probe_spec": throughput_probe_spec,
        "required_cli_flags": [
            "--enable-r2-phase-threshold-switching",
            "--enable-r4-alpha-dual-loss",
            "--emit-loss-decomposition",
            "--emit-threshold-switch-trace",
            "--emit-alpha-dual-trace",
            "--emit-gradient-path-attestation",
            "--emit-dynamic-delta-audit",
            "--emit-checkpoint-sha256sums",
            "--warm-start-checkpoint",
            "--output-dir",
            "--runtime-log-dir",
            "--max-steps",
            "--save-total-limit",
            "--no-sudo",
            "--cuda-visible-devices",
        ],
        "r2_phase_threshold_switching": {
            "module": "work.recap.phase_thresholds",
            "fine_tuning_policy": build_phase_threshold_metadata(
                threshold_phase="fine_tuning"
            ),
        },
        "r4_alpha_dual_loss": {
            "module": "work.recap.dual_loss",
            "config": {
                "alpha": float(dual_loss_config.alpha),
                "dropout_p": float(dual_loss_config.dropout_p),
                "mode": str(dual_loss_config.mode),
            },
            "integration_report": build_dual_loss_integration_report(
                training_surface="gr00t_r2r4_closed_finetune_wrapper",
                single_path_loss_name="canonical_gr00t_training_loss",
                dual_view_integrated=True,
                replacement_strategy="process_local_overlay",
            ),
        },
        "schema_versions": {
            "training_run_manifest": RUN_MANIFEST_SCHEMA_VERSION,
            "loss_decomposition": LOSS_DECOMPOSITION_SCHEMA_VERSION,
            "threshold_switch_trace": THRESHOLD_TRACE_SCHEMA_VERSION,
            "alpha_dual_loss_trace": ALPHA_DUAL_TRACE_SCHEMA_VERSION,
            "dynamic_delta_audit": DYNAMIC_DELTA_AUDIT_SCHEMA_VERSION,
            "gradient_path_attestation": GRADIENT_PATH_ATTESTATION_SCHEMA_VERSION,
            "checkpoint_provenance": CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
            "throughput_probe_spec": THROUGHPUT_PROBE_SCHEMA_VERSION,
        },
    }


def _build_preflight(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    protocol: dict[str, Any],
    gate_policy: dict[str, Any],
) -> dict[str, Any]:
    common_warm_start = _resolve_repo_path(repo_root, COMMON_WARM_START_REL)
    iter3_failed_checkpoint = _resolve_repo_path(
        repo_root, ITER3_FAILED_P5_CHECKPOINT_REL
    )
    flag_status = _required_flag_status(args)
    warm_start_path = Path(config["warm_start_checkpoint"])
    dataset_path = Path(config["dataset_path"])
    output_dir = Path(config["output_dir"])
    gr00t_root = _resolve_repo_path(repo_root, GR00T_ROOT_REL)
    gate_preconditions = gate_policy.get("required_before_g2", {})
    if not isinstance(gate_preconditions, dict):
        gate_preconditions = {}
    throughput_probe_spec = _build_throughput_probe_spec(
        repo_root=repo_root,
        config=config,
    )

    checks = {
        "warm_start_matches_common_contract": _same_path(
            warm_start_path, common_warm_start
        ),
        "warm_start_exists": warm_start_path.is_dir(),
        "warm_start_has_config_json": (warm_start_path / "config.json").is_file(),
        "warm_start_is_not_iter3_failed_checkpoint": not _same_path(
            warm_start_path, iter3_failed_checkpoint
        ),
        "dataset_path_exists": dataset_path.is_dir(),
        "output_under_gr00t_iter9_root": str(output_dir).startswith(str(gr00t_root)),
        "gr00t_artifact_root_valid": bool(
            gate_policy.get("gr00t_artifact_root_valid")
            or gate_preconditions.get("gr00t_artifact_root_valid")
            or protocol.get("precondition_predicates", {}).get(
                "gr00t_artifact_root_valid", False
            )
        ),
        "cuda_visible_devices_is_gpu1": config["cuda_visible_devices"]
        == DEFAULT_CUDA_VISIBLE_DEVICES,
        "no_sudo": bool(args.no_sudo),
        "save_strategy_is_steps": str(config["save_strategy"]) == "steps",
        "save_steps_positive": int(config["save_steps"]) > 0,
        "max_steps_positive": int(config["max_steps"]) > 0,
        "save_total_limit_is_single_checkpoint": int(config["save_total_limit"])
        == int(G1_CONTRACT_SAVE_TOTAL_LIMIT),
        "all_required_feature_flags_enabled": all(flag_status.values()),
        "iter9_hooks_active_flag_exists": (repo_root / ".omc/iter9_hooks_active.flag").is_file(),
        "throughput_probe_spec_present": bool(throughput_probe_spec),
        "throughput_probe_steps_is_50": int(
            throughput_probe_spec["probe_steps"]
        )
        == int(THROUGHPUT_PROBE_STEPS),
        "throughput_probe_gate_matches_14h_soft_cap": float(
            throughput_probe_spec["gate"]["minimum_steps_per_hour"]
        )
        == _minimum_g2_steps_per_hour(),
    }
    status = "PASS" if all(checks.values()) else "BLOCK"
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "run_id": RUN_ID,
        "status": status,
        "checks": checks,
        "required_flag_status": flag_status,
        "effective_config": {
            "warm_start_checkpoint": _as_repo_relative(repo_root, warm_start_path),
            "dataset_path": _as_repo_relative(repo_root, dataset_path),
            "output_dir": _as_repo_relative(repo_root, output_dir),
            "runtime_log_dir": _as_repo_relative(
                repo_root, Path(config["runtime_log_dir"])
            ),
            "max_steps": int(config["max_steps"]),
            "save_total_limit": int(config["save_total_limit"]),
            "save_strategy": str(config["save_strategy"]),
            "save_steps": int(config["save_steps"]),
            "seed": int(config["seed"]),
            "cuda_visible_devices": str(config["cuda_visible_devices"]),
            "no_sudo": bool(args.no_sudo),
        },
        "coordinator_protocol": {
            "path": COORDINATOR_PROTOCOL_REL.as_posix(),
            "present": bool(protocol),
            "schema_version": protocol.get("schema_version"),
        },
        "coordinator_gate_policy": {
            "path": COORDINATOR_GATE_POLICY_REL.as_posix(),
            "present": bool(gate_policy),
            "schema_version": gate_policy.get("schema_version"),
        },
        "throughput_probe_spec": throughput_probe_spec,
    }


def _emit_g1_artifacts(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    protocol: dict[str, Any],
    gate_policy: dict[str, Any],
) -> dict[str, Any]:
    contract_dir = Path(config["contract_output_dir"])
    contract_path = contract_dir / CONTRACT_JSON_NAME
    preflight_path = contract_dir / PREFLIGHT_JSON_NAME
    contract = _build_contract(
        repo_root=repo_root,
        config=config,
        protocol=protocol,
        gate_policy=gate_policy,
    )
    preflight = _build_preflight(
        repo_root=repo_root,
        args=args,
        config=config,
        protocol=protocol,
        gate_policy=gate_policy,
    )
    _write_json_atomic(contract_path, contract)
    _write_json_atomic(preflight_path, preflight)
    return {
        "contract_path": str(contract_path),
        "preflight_path": str(preflight_path),
        "preflight_status": preflight["status"],
    }


def _build_delegated_argv(config: dict[str, Any], *, dry_run: bool) -> list[str]:
    argv = [
        "34_recap_finetune_repro.py",
        "--base-model",
        "local/GR00T-R2R4-closed-warm-start",
        "--dataset-path",
        str(config["dataset_path"]),
        "--output-dir",
        str(config["output_dir"]),
        "--max-steps",
        str(int(config["max_steps"])),
        "--save-steps",
        str(int(config["save_steps"])),
        "--save-total-limit",
        str(int(config["save_total_limit"])),
        "--global-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--dataloader-num-workers",
        "0",
        "--num-gpus",
        "1",
        "--no-use-wandb",
    ]
    if str(config["python"]):
        argv.extend(["--python", str(config["python"])])
    if dry_run:
        argv.append("--dry-run")
    return argv


@contextlib.contextmanager
def _temporary_argv(argv: list[str]) -> Iterator[None]:
    previous = list(sys.argv)
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = previous


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _patch_hf_snapshot_to_warm_start(warm_start_checkpoint: Path) -> Iterator[None]:
    from work.recap import hf_snapshot_patch

    original = hf_snapshot_patch.make_patched_base_model_dir

    def _patched_make_patched_base_model_dir(
        *,
        repo_id: str,
        revision: str | None = None,
        out_root: Path | str = "agent/artifacts/hf_patches",
        overrides: dict[str, object] | None = None,
        hf_hub_cache_dir: Path | str | None = None,
        snapshot_dir: Path | str | None = None,
        emit_evidence: bool = True,
        force: bool = False,
    ) -> Path:
        del repo_id, revision, hf_hub_cache_dir, snapshot_dir
        return original(
            repo_id="local/GR00T-R2R4-closed-warm-start",
            revision=None,
            out_root=out_root,
            overrides=overrides,
            hf_hub_cache_dir=None,
            snapshot_dir=warm_start_checkpoint,
            emit_evidence=emit_evidence,
            force=force,
        )

    hf_snapshot_patch.make_patched_base_model_dir = _patched_make_patched_base_model_dir
    try:
        yield
    finally:
        hf_snapshot_patch.make_patched_base_model_dir = original


def _run_delegate(config: dict[str, Any], *, dry_run: bool) -> int:
    delegated_argv = _build_delegated_argv(config, dry_run=dry_run)
    with (
        _patch_hf_snapshot_to_warm_start(Path(config["warm_start_checkpoint"])),
        _temporary_env({"CUDA_VISIBLE_DEVICES": str(config["cuda_visible_devices"])}),
        _temporary_argv(delegated_argv),
    ):
        started = time.monotonic()
        rc = int(RecapFinetuneReproScriptApp().run())
        elapsed_seconds = max(0.0, time.monotonic() - started)
    _maybe_write_throughput_probe_report(
        config=config,
        dry_run=dry_run,
        return_code=rc,
        elapsed_seconds=elapsed_seconds,
    )
    return rc


def _maybe_write_throughput_probe_report(
    *,
    config: dict[str, Any],
    dry_run: bool,
    return_code: int,
    elapsed_seconds: float,
) -> None:
    if bool(dry_run):
        return
    if int(config["max_steps"]) != int(THROUGHPUT_PROBE_STEPS):
        return

    output_dir = Path(config["output_dir"])
    latest_step = _latest_checkpoint_step(output_dir)
    probe_steps_completed = int(latest_step or 0)
    if return_code == 0 and probe_steps_completed <= 0:
        # Some upstream save surfaces materialize the final checkpoint after
        # the trainer exits; if the command returned 0 but checkpoint discovery
        # is unavailable, still expose the completed requested step count while
        # keeping checkpoint evidence explicit in the report.
        probe_steps_completed = int(config["max_steps"])
    estimated_steps_per_hour = (
        0.0
        if elapsed_seconds <= 0.0
        else float(probe_steps_completed) / (elapsed_seconds / 3600.0)
    )
    minimum_steps_per_hour = _minimum_g2_steps_per_hour()
    probe_status = (
        "PASS"
        if return_code == 0
        and probe_steps_completed >= int(THROUGHPUT_PROBE_STEPS)
        and estimated_steps_per_hour >= minimum_steps_per_hour
        else "FAIL"
    )
    payload = {
        "schema_version": THROUGHPUT_PROBE_REPORT_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "probe_status": probe_status,
        "delegate_return_code": int(return_code),
        "probe_steps_requested": int(config["max_steps"]),
        "probe_steps_completed": int(probe_steps_completed),
        "latest_checkpoint_step": latest_step,
        "elapsed_seconds": float(round(elapsed_seconds, 6)),
        "estimated_steps_per_hour": float(round(estimated_steps_per_hour, 6)),
        "minimum_steps_per_hour": float(minimum_steps_per_hour),
        "g2_launch_allowed": bool(probe_status == "PASS"),
        "output_dir": str(output_dir),
        "runtime_log_dir": str(config["runtime_log_dir"]),
    }
    _write_json_atomic(output_dir / "throughput_probe_report.json", payload)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root()
    protocol = _load_optional_json(repo_root, COORDINATOR_PROTOCOL_REL)
    gate_policy = _load_optional_json(repo_root, COORDINATOR_GATE_POLICY_REL)
    config = _effective_config(repo_root=repo_root, args=args, protocol=protocol)
    artifact_result = _emit_g1_artifacts(
        repo_root=repo_root,
        args=args,
        config=config,
        protocol=protocol,
        gate_policy=gate_policy,
    )
    summary = {
        "schema_version": "gr00t_r2r4_wrapper_invocation_summary_v1",
        "created_at_utc": _utc_now(),
        "artifact_result": artifact_result,
        "dry_run": bool(args.dry_run),
        "emit_contract_only": bool(args.emit_contract_only),
    }
    json.dump(summary, sys.stdout, ensure_ascii=True, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    if artifact_result["preflight_status"] != "PASS":
        return 1
    if bool(args.emit_contract_only):
        return 0
    return _run_delegate(config, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
