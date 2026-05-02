#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_DIR = Path("agent/artifacts/gr00t_wbc_preflight")
DEFAULT_RUNTIME_LOG_DIR = Path("agent/runtime_logs/gr00t_wbc_preflight")

DEFAULT_MODE = "smoke"
DEFAULT_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
DEFAULT_MODEL_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 5555
DEFAULT_SERVER_PYTHON = ""
DEFAULT_MUJOCO_GL = ""

DEFAULT_SERVER_READY_TIMEOUT_S = 600.0
DEFAULT_SERVER_PING_TIMEOUT_MS = 2000
DEFAULT_SERVER_PING_INTERVAL_S = 1.0
DEFAULT_TOTAL_TIMEOUT_S = 660.0

DEFAULT_EXPECTED_ACTION_HORIZON = 30
DEFAULT_SMOKE_N_ACTION_STEPS = 20
DEFAULT_MAX_EPISODE_STEPS = 60

SCHEMA_VERSION = "g1_gr00t_wbc_preflight_gate_v1"
PREFLIGHT_REPORT_JSON_NAME = "preflight_report.json"
FAILURE_NOTE_MARKDOWN_NAME = "preflight_failure_note.md"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from agent.run import state_conditioned_env_resolution
from work.recap import state_conditioned_phase0_smoke
from work.demo_utils import paths as demo_paths


class PreflightGateError(RuntimeError):
    stage: str
    reason_code: str
    detail: Mapping[str, object] | None
    blockers: list[str]

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        reason_code: str,
        detail: Mapping[str, object] | None = None,
        blockers: Sequence[str] | None = None,
    ):
        super().__init__(message)
        self.stage = str(stage)
        self.reason_code = str(reason_code)
        self.detail = dict(detail) if detail is not None else None
        self.blockers = [str(value) for value in list(blockers or [])]


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _json_text(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value]


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _server_entrypoint(repo_root: Path) -> Path:
    return repo_root / "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"


def _runtime_log_dir(repo_root: Path) -> Path:
    path = repo_root / DEFAULT_RUNTIME_LOG_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _activate_live_import_roots(repo_root: Path) -> list[str]:
    return state_conditioned_phase0_smoke._activate_live_import_roots(repo_root)


def _maybe_reexec_into_wbc_venv(repo_root: Path) -> None:
    paths_mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(paths_mod, "maybe_reexec_into_wbc_venv")
    fn(repo_root)


def _collect_runtime_probe(
    *, repo_root: Path, add_live_paths: bool
) -> dict[str, object]:
    return state_conditioned_phase0_smoke._collect_runtime_probe(
        repo_root=repo_root,
        add_live_paths=bool(add_live_paths),
    )


def _server_entrypoint_probe(repo_root: Path) -> dict[str, object]:
    path = _server_entrypoint(repo_root)
    exists = bool(path.is_file())
    source_text = ""
    if exists:
        try:
            source_text = path.read_text(encoding="utf-8")
        except Exception:
            source_text = ""
    surface = {
        "embodiment_tag": "embodiment_tag" in source_text,
        "strict": "strict" in source_text,
        "use_sim_policy_wrapper": "use_sim_policy_wrapper" in source_text,
    }
    blockers: list[str] = []
    if not exists:
        blockers.append("server_entrypoint_missing")
    return {
        "path": str(path),
        "exists": exists,
        "config_surface": surface,
        "ok": exists,
        "blockers": blockers,
    }


def _zero_action_like(value: object) -> object:
    np = importlib.import_module("numpy")
    if isinstance(value, dict):
        return {str(key): _zero_action_like(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_zero_action_like(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_zero_action_like(item) for item in value)
    if hasattr(value, "shape"):
        return np.zeros_like(value)
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0
    if isinstance(value, float):
        return 0.0
    return value


def _collect_sim_imports(*, repo_root: Path) -> dict[str, object]:
    active_paths = _activate_live_import_roots(repo_root)
    state_conditioned_phase0_smoke._install_known_live_import_shims()

    imports: dict[str, bool] = {
        "gymnasium": False,
        "robocasa": False,
        "sync_env": False,
        "base_config": False,
        "wbc_wrapper": False,
        "sim_policy_wrapper": False,
    }
    module_files: dict[str, str] = {}
    errors: dict[str, str] = {}

    module_specs = {
        "gymnasium": "gymnasium",
        "robocasa": "robocasa",
        "sync_env": "gr00t_wbc.control.envs.robocasa.sync_env",
        "base_config": "gr00t_wbc.control.main.teleop.configs.configs",
        "wbc_wrapper": "gr00t_wbc.control.utils.n1_utils",
        "sim_policy_wrapper": "gr00t.eval.sim.wrapper.multistep_wrapper",
    }
    for field_name, module_name in module_specs.items():
        try:
            module_obj = importlib.import_module(module_name)
        except Exception as exc:
            errors[field_name] = f"{exc.__class__.__name__}: {_exception_message(exc)}"
            continue
        imports[field_name] = True
        module_file = getattr(module_obj, "__file__", None)
        if module_file is not None:
            module_files[field_name] = str(module_file)

    shim_checks: dict[str, bool] = {
        "check_obj_upright_available": False,
        "visuals_utls_importable": False,
        "ik_wrapper_importable": False,
    }
    try:
        object_utils = importlib.import_module("robocasa.utils.object_utils")
        shim_checks["check_obj_upright_available"] = hasattr(
            object_utils,
            "check_obj_upright",
        )
        if not shim_checks["check_obj_upright_available"]:
            errors["check_obj_upright_available"] = (
                "RuntimeError: check_obj_upright shim unavailable"
            )
    except Exception as exc:
        errors["check_obj_upright_available"] = (
            f"{exc.__class__.__name__}: {_exception_message(exc)}"
        )

    for field_name, module_name in (
        ("visuals_utls_importable", "robocasa.utils.visuals_utls"),
        ("ik_wrapper_importable", "robocasa.wrappers.ik_wrapper"),
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors[field_name] = f"{exc.__class__.__name__}: {_exception_message(exc)}"
            continue
        shim_checks[field_name] = True

    blockers = [field_name for field_name, ok in imports.items() if not bool(ok)] + [
        field_name for field_name, ok in shim_checks.items() if not bool(ok)
    ]
    return {
        "ok": not blockers,
        "imports": imports,
        "shim_checks": shim_checks,
        "module_files": module_files,
        "errors": errors,
        "blockers": blockers,
        "sys_path_injected": active_paths,
    }


def _ensure_explicit_g1_env_registration(gym_module: Any) -> dict[str, object]:
    sync_env_module_name = "gr00t_wbc.control.envs.robocasa.sync_env"
    before_ids = state_conditioned_env_resolution.registered_g1_env_ids(gym_module)
    try:
        sync_env_mod = importlib.import_module(sync_env_module_name)
    except Exception as exc:
        raise PreflightGateError(
            "env_registry",
            "explicit G1 env registration import failed",
            reason_code="import_shim_issue",
            detail={
                "sync_env_module": sync_env_module_name,
                "registered_env_count_before_import": int(len(before_ids)),
                "error": f"{exc.__class__.__name__}: {_exception_message(exc)}",
            },
            blockers=["sync_env_import"],
        ) from exc

    after_ids = state_conditioned_env_resolution.registered_g1_env_ids(gym_module)
    module_file = str(getattr(sync_env_mod, "__file__", "<unknown>"))
    if not after_ids:
        raise PreflightGateError(
            "env_registry",
            "explicit G1 env registration left registry empty",
            reason_code="state_conditioned_env_unavailable",
            detail={
                "sync_env_module": sync_env_module_name,
                "sync_env_module_file": module_file,
                "registered_env_count_before_import": int(len(before_ids)),
                "registered_env_count_after_import": int(len(after_ids)),
                "registered_env_prefix": state_conditioned_env_resolution.ENV_REGISTRY_PREFIX,
            },
            blockers=["registered_env_count_after_import"],
        )

    return {
        "sync_env_module": sync_env_module_name,
        "sync_env_module_file": module_file,
        "registered_env_count_before_import": int(len(before_ids)),
        "registered_env_count_after_import": int(len(after_ids)),
        "registered_env_ids_sample": after_ids[: min(5, len(after_ids))],
    }


def _collect_env_resolution(*, requested_env_name: str) -> dict[str, object]:
    gym_module = importlib.import_module("gymnasium")
    env_registry = _ensure_explicit_g1_env_registration(gym_module)
    try:
        resolution = (
            state_conditioned_env_resolution.resolve_apple_to_plate_g1_env_name(
                gym_module,
                requested_env_name=str(requested_env_name),
            )
        )
    except state_conditioned_env_resolution.StateConditionedEnvResolutionError as exc:
        raise PreflightGateError(
            "env_resolution",
            _exception_message(exc),
            reason_code=str(exc.code),
            detail={
                **env_registry,
                **exc.to_machine_payload(),
            },
            blockers=[str(exc.code)],
        ) from exc

    return {
        "ok": True,
        **env_registry,
        "logical_task": str(resolution["logical_task"]),
        "requested_env_name": str(resolution["requested_env_name"]),
        "resolved_env_name": str(resolution["resolved_env_name"]),
        "alias_applied": bool(resolution["alias_applied"]),
        "available_close_matches": list(
            cast(Sequence[str], resolution["available_close_matches"])
        ),
        "registered_env_count": int(
            len(cast(Sequence[str], resolution["registered_env_ids"]))
        ),
        "registered_env_ids_sample": list(
            cast(Sequence[str], resolution["registered_env_ids"])
        )[:5],
    }


def _phase0_error_to_preflight(
    exc: state_conditioned_phase0_smoke.Phase0SmokeError,
) -> PreflightGateError:
    message = _exception_message(exc)
    reason_code = "runtime_dependency_breakage"
    if exc.stage == "policy_ping" and "timeout waiting for ping" in message:
        reason_code = "ping_timeout"
    elif exc.stage == "import_probe" and "missing server entrypoint" in message:
        reason_code = "server_entrypoint_missing"
    return PreflightGateError(
        exc.stage,
        message,
        reason_code=reason_code,
        detail=exc.detail,
        blockers=[exc.stage],
    )


def _run_smoke_checks(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    output_dir: Path,
    env_resolution: Mapping[str, object],
) -> dict[str, object]:
    del output_dir
    runtime_dir = _runtime_log_dir(repo_root)
    server_log = runtime_dir / "00_server.log"
    client: Any | None = None
    proc: Any | None = None
    started_by_me = False
    host_for_client = str(args.server_host)
    env: Any | None = None

    policy_ping: dict[str, object] = {
        "attempted": True,
        "ok": False,
        "host": state_conditioned_phase0_smoke._normalize_client_host(
            str(args.server_host)
        ),
        "port": int(args.server_port),
        "spawned": False,
        "reused_existing": False,
        "server_log": str(server_log),
    }
    action_horizon_check: dict[str, object] = {
        "attempted": False,
        "ok": False,
        "expected_policy_horizon": int(args.expected_action_horizon),
        "requested_smoke_n_action_steps": int(args.smoke_n_action_steps),
        "server_action_horizon": None,
        "within_smoke_budget": False,
        "modality_config_keys": [],
    }
    smoke: dict[str, object] = {
        "attempted": False,
        "reset_ok": False,
        "step_ok": False,
        "sample_action_kind": None,
        "terminated": None,
        "truncated": None,
        "reward_sample": None,
    }

    try:
        try:
            client, proc, started_by_me, host_for_client = (
                state_conditioned_phase0_smoke._ensure_server_ready(
                    args,
                    repo_root=repo_root,
                    server_log=server_log,
                )
            )
        except state_conditioned_phase0_smoke.Phase0SmokeError as exc:
            raise _phase0_error_to_preflight(exc) from exc
        assert client is not None
        policy_ping.update(
            {
                "ok": True,
                "host": str(host_for_client),
                "spawned": bool(started_by_me),
                "reused_existing": not bool(started_by_me),
            }
        )

        try:
            modality_cfg = cast(Mapping[str, object], client.get_modality_config())
        except BaseException as exc:
            raise PreflightGateError(
                "action_horizon_check",
                f"client.get_modality_config() failed: {_exception_message(exc)}",
                reason_code="runtime_dependency_breakage",
                blockers=["modality_config"],
            ) from exc

        action_horizon = state_conditioned_phase0_smoke._infer_action_horizon(
            modality_cfg
        )
        action_horizon_check.update(
            {
                "attempted": True,
                "server_action_horizon": int(action_horizon)
                if action_horizon is not None
                else None,
                "modality_config_keys": sorted(
                    [str(key) for key in list(modality_cfg.keys())]
                ),
                "within_smoke_budget": bool(
                    action_horizon is not None
                    and int(args.smoke_n_action_steps) <= int(action_horizon)
                ),
            }
        )
        if action_horizon is None or int(action_horizon) != int(
            args.expected_action_horizon
        ):
            raise PreflightGateError(
                "action_horizon_check",
                "server action horizon does not match the required G1 WBC contract",
                reason_code="action_horizon_mismatch",
                detail=action_horizon_check,
                blockers=["server_action_horizon"],
            )
        if int(args.smoke_n_action_steps) > int(action_horizon):
            raise PreflightGateError(
                "action_horizon_check",
                "requested smoke action steps exceed the server action horizon",
                reason_code="action_horizon_mismatch",
                detail=action_horizon_check,
                blockers=["requested_smoke_n_action_steps"],
            )
        action_horizon_check["ok"] = True

        np = importlib.import_module("numpy")
        video_delta = state_conditioned_phase0_smoke._delta_indices_from_modality(
            modality_cfg.get("video")
        )
        state_delta = state_conditioned_phase0_smoke._delta_indices_from_modality(
            modality_cfg.get("state")
        )
        video_delta_indices = (
            np.asarray(video_delta, dtype=np.int64)
            if video_delta
            else np.asarray([0], dtype=np.int64)
        )
        state_delta_indices = (
            np.asarray(state_delta, dtype=np.int64)
            if state_delta
            else np.asarray([0], dtype=np.int64)
        )

        resolved_env_name = str(env_resolution["resolved_env_name"])
        env = state_conditioned_phase0_smoke._make_live_env(
            resolved_env_name=resolved_env_name,
            n_action_steps=int(args.smoke_n_action_steps),
            max_episode_steps=int(args.max_episode_steps),
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
        )
        assert env is not None
        smoke["attempted"] = True

        try:
            _obs, _info = env.reset()
        except BaseException as exc:
            raise PreflightGateError(
                "smoke",
                f"env.reset() failed: {_exception_message(exc)}",
                reason_code="runtime_dependency_breakage",
                blockers=["env.reset"],
            ) from exc
        smoke["reset_ok"] = True

        try:
            sample_action = env.action_space.sample()
            action_to_execute = _zero_action_like(sample_action)
            _next_obs, reward, term, trunc, _step_info = env.step(action_to_execute)
        except BaseException as exc:
            raise PreflightGateError(
                "smoke",
                f"env.step(sample()) failed: {_exception_message(exc)}",
                reason_code="runtime_dependency_breakage",
                blockers=["env.step"],
            ) from exc
        smoke.update(
            {
                "step_ok": True,
                "sample_action_kind": type(sample_action).__name__,
                "executed_action_strategy": "zero_like_action_space_sample",
                "terminated": bool(term),
                "truncated": bool(trunc),
                "reward_sample": cast(Any, np.asarray(reward)).reshape(-1).tolist(),
            }
        )
        return {
            "policy_ping": policy_ping,
            "action_horizon_check": action_horizon_check,
            "smoke": smoke,
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if client is not None and started_by_me and bool(args.kill_server_on_exit):
            try:
                state_conditioned_phase0_smoke._safe_kill_server(
                    client,
                    int(args.server_ping_timeout_ms),
                )
            except Exception:
                pass
        if proc is not None and started_by_me and bool(args.kill_server_on_exit):
            state_conditioned_phase0_smoke._terminate_process(proc, timeout_s=10.0)


def _timeout_policy(args: argparse.Namespace) -> dict[str, object]:
    return {
        "hard_timeouts_enforced": True,
        "total_timeout_s": float(args.total_timeout_s),
        "server_ready_timeout_s": float(args.server_ready_timeout_s),
        "server_ping_timeout_ms": int(args.server_ping_timeout_ms),
        "server_ping_interval_s": float(args.server_ping_interval_s),
    }


def _empty_sim_imports() -> dict[str, object]:
    return {
        "ok": False,
        "imports": {
            "gymnasium": False,
            "robocasa": False,
            "sync_env": False,
            "base_config": False,
            "wbc_wrapper": False,
            "sim_policy_wrapper": False,
        },
        "shim_checks": {
            "check_obj_upright_available": False,
            "visuals_utls_importable": False,
            "ik_wrapper_importable": False,
        },
        "module_files": {},
        "errors": {},
        "blockers": [],
        "sys_path_injected": [],
    }


def _base_payload(args: argparse.Namespace, *, repo_root: Path) -> dict[str, object]:
    preferred_live_python = state_conditioned_phase0_smoke._preferred_live_python(
        repo_root
    )
    python_contract = demo_paths.load_stage3_training_python_contract(repo_root)
    server_entry = _server_entrypoint_probe(repo_root)
    requested_server_python = str(getattr(args, "server_python", "") or "").strip()
    effective_server_python = requested_server_python or preferred_live_python
    return {
        "schema_version": SCHEMA_VERSION,
        "run_mode": str(args.mode),
        "status": "FAIL",
        "reason_code": "blocked_preflight",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "failure": None,
        "live_checks_requested": bool(str(args.mode) == "smoke"),
        "live_checks_attempted": False,
        "repo_root": str(repo_root),
        "output_dir": str(args.output_dir),
        "artifact_path": None,
        "failure_note_path": None,
        "python_env": {
            "sys_executable": str(sys.executable),
            "sys_prefix": str(sys.prefix),
            "venv_active": bool(os.environ.get("VIRTUAL_ENV")),
            "venv_path": str(os.environ.get("VIRTUAL_ENV", "")),
            "contract_manifest_path": str(python_contract["manifest_path"]),
            "orchestrator_python": str(python_contract["orchestrator_python"]),
            "orchestrator_python_matches_current_process": bool(
                demo_paths.same_abspath_preserve_symlink(
                    sys.executable,
                    str(python_contract["orchestrator_python"]),
                )
            ),
            "delegate_runtime_python": str(python_contract["delegate_runtime_python"]),
            "delegate_runtime_python_matches_current_process": bool(
                demo_paths.same_abspath_preserve_symlink(
                    sys.executable,
                    str(python_contract["delegate_runtime_python"]),
                )
            ),
            "server_python_requested": requested_server_python,
            "server_python_effective": effective_server_python,
            "server_python_exists": bool(Path(effective_server_python).exists()),
            "preferred_live_python": preferred_live_python,
            "preferred_live_python_exists": bool(Path(preferred_live_python).exists()),
            "sys_path_injected": [],
        },
        "runtime_probe": state_conditioned_phase0_smoke._empty_runtime_probe(),
        "sim_imports": _empty_sim_imports(),
        "server_entrypoint": server_entry,
        "env_resolution": {
            "ok": False,
            "logical_task": state_conditioned_env_resolution.LOGICAL_TASK_APPLE_TO_PLATE_G1,
            "requested_env_name": str(args.env_name),
            "resolved_env_name": None,
            "alias_applied": False,
            "available_close_matches": [],
            "registered_env_count": 0,
            "registered_env_count_before_import": 0,
            "registered_env_count_after_import": 0,
            "registered_env_ids_sample": [],
        },
        "policy_ping": {
            "attempted": False,
            "ok": False,
            "host": str(args.server_host),
            "port": int(args.server_port),
            "spawned": False,
            "reused_existing": False,
            "server_log": None,
        },
        "timeout_policy": _timeout_policy(args),
        "action_horizon_check": {
            "attempted": False,
            "ok": False,
            "expected_policy_horizon": int(args.expected_action_horizon),
            "requested_smoke_n_action_steps": int(args.smoke_n_action_steps),
            "server_action_horizon": None,
            "within_smoke_budget": False,
            "modality_config_keys": [],
        },
        "smoke": {
            "attempted": False,
            "reset_ok": False,
            "step_ok": False,
            "sample_action_kind": None,
            "terminated": None,
            "truncated": None,
            "reward_sample": None,
        },
        "system_break_flags": {
            "active_breaks": ["preflight_blocked"],
            "runtime_dependency_breakage": True,
            "import_shim_issue": False,
            "server_entrypoint_missing": not bool(server_entry["exists"]),
            "state_conditioned_env_unavailable": False,
            "ping_timeout": False,
            "action_horizon_mismatch": False,
        },
    }


def _failure_payload(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    stage: str,
    reason_code: str,
    message: str,
    exception_type: str = "RuntimeError",
    detail: Mapping[str, object] | None = None,
    blockers: Sequence[str] | None = None,
    live_checks_attempted: bool = False,
) -> dict[str, object]:
    payload = _base_payload(args, repo_root=repo_root)
    payload["reason_code"] = str(reason_code)
    payload["live_checks_attempted"] = bool(live_checks_attempted)
    payload["failure"] = {
        "stage": str(stage),
        "type": str(exception_type),
        "message": str(message),
        "blockers": [str(value) for value in list(blockers or [])],
        "detail": dict(detail) if detail is not None else None,
    }
    return payload


def _finalize_system_break_flags(payload: dict[str, object]) -> None:
    reason_code = str(payload.get("reason_code", "blocked_preflight"))
    runtime_probe = cast(Mapping[str, object], payload.get("runtime_probe", {}))
    sim_imports = cast(Mapping[str, object], payload.get("sim_imports", {}))
    server_entrypoint = cast(Mapping[str, object], payload.get("server_entrypoint", {}))
    env_resolution = cast(Mapping[str, object], payload.get("env_resolution", {}))
    policy_ping = cast(Mapping[str, object], payload.get("policy_ping", {}))
    action_horizon_check = cast(
        Mapping[str, object],
        payload.get("action_horizon_check", {}),
    )
    runtime_blocked = any(
        not bool(runtime_probe.get(field_name, False))
        for field_name in (
            "torch_import_ok",
            "torch_cuda_available",
            "torch_bfloat16_cuda_ok",
            "flash_attn_2_available",
            "gr00t_import_ok",
            "gr00t_wbc_import_ok",
            "policy_client_import_ok",
            "rollout_policy_import_ok",
        )
    )
    active_breaks: list[str] = []
    for flag_name in (
        "runtime_dependency_breakage",
        "import_shim_issue",
        "server_entrypoint_missing",
        "state_conditioned_env_unavailable",
        "ping_timeout",
        "action_horizon_mismatch",
    ):
        if reason_code == flag_name:
            active_breaks.append(flag_name)
    if not bool(sim_imports.get("ok", False)):
        active_breaks.append("sim_imports_blocked")
    if not bool(server_entrypoint.get("exists", False)):
        active_breaks.append("server_entrypoint_missing")
    if not bool(env_resolution.get("ok", False)):
        active_breaks.append("env_resolution_blocked")
    if bool(payload.get("live_checks_attempted", False)) and not bool(
        policy_ping.get("ok", False)
    ):
        active_breaks.append("policy_ping_blocked")
    if bool(payload.get("live_checks_attempted", False)) and not bool(
        action_horizon_check.get("ok", False)
    ):
        active_breaks.append("action_horizon_check_blocked")
    if runtime_blocked:
        active_breaks.append("runtime_dependency_breakage")
    payload["system_break_flags"] = {
        "active_breaks": sorted(set(active_breaks)) or ["none"],
        "runtime_dependency_breakage": bool(
            reason_code == "runtime_dependency_breakage" or runtime_blocked
        ),
        "import_shim_issue": bool(reason_code == "import_shim_issue"),
        "server_entrypoint_missing": bool(
            reason_code == "server_entrypoint_missing"
            or not bool(server_entrypoint.get("exists", False))
        ),
        "state_conditioned_env_unavailable": bool(
            reason_code == "state_conditioned_env_unavailable"
        ),
        "ping_timeout": bool(reason_code == "ping_timeout"),
        "action_horizon_mismatch": bool(reason_code == "action_horizon_mismatch"),
    }


def _build_failure_note(report: Mapping[str, object]) -> str:
    failure = cast(Mapping[str, object], report.get("failure", {}))
    blockers = _string_list(failure.get("blockers", []))
    detail = failure.get("detail")
    lines = [
        "# GR00T WBC preflight 失败说明",
        "",
        f"- status: `{report.get('status', 'FAIL')}`",
        f"- reason_code: `{report.get('reason_code', 'blocked_preflight')}`",
        f"- stage: `{failure.get('stage', 'unknown')}`",
        f"- message: `{failure.get('message', 'unknown')}`",
        "- blockers:",
    ]
    if blockers:
        lines.extend(f"  - `{item}`" for item in blockers)
    else:
        lines.append("  - `none captured`")
    if isinstance(detail, Mapping):
        lines.extend(
            [
                "",
                "## detail",
                "```json",
                _json_text(cast(Mapping[str, object], detail)),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "请先修复该 preflight 阻塞项，再进入后续 anchor / audit / P / D 任务。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_failure_note(path: Path, report: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_build_failure_note(report), encoding="utf-8")
    tmp.replace(path)
    return path


def _run_readiness_checks(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    sim_imports = _collect_sim_imports(repo_root=repo_root)
    if not bool(sim_imports.get("ok", False)):
        raise PreflightGateError(
            "sim_imports",
            "sim import or shim readiness failed",
            reason_code="import_shim_issue",
            detail={
                "errors": dict(
                    cast(Mapping[str, object], sim_imports.get("errors", {}))
                ),
                "blockers": list(cast(Sequence[str], sim_imports.get("blockers", []))),
            },
            blockers=cast(Sequence[str], sim_imports.get("blockers", [])),
        )

    env_resolution = _collect_env_resolution(requested_env_name=str(args.env_name))
    result: dict[str, object] = {
        "sim_imports": sim_imports,
        "env_resolution": env_resolution,
    }
    if str(args.mode) != "smoke":
        return result

    smoke_checks = _run_smoke_checks(
        args,
        repo_root=repo_root,
        output_dir=output_dir,
        env_resolution=env_resolution,
    )
    result.update(smoke_checks)
    return result


def run_preflight(
    args: argparse.Namespace,
    *,
    runtime_probe_fn: Any | None = None,
    readiness_fn: Any | None = None,
) -> dict[str, object]:
    repo_root = REPO_ROOT
    output_dir = _validate_output_dir(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if runtime_probe_fn is None:
        runtime_probe_fn = _collect_runtime_probe
    if readiness_fn is None:
        readiness_fn = _run_readiness_checks

    payload = _base_payload(args, repo_root=repo_root)
    payload["output_dir"] = str(output_dir)

    probe_result = cast(
        Mapping[str, object],
        runtime_probe_fn(repo_root=repo_root, add_live_paths=True),
    )
    runtime_probe = cast(
        Mapping[str, object],
        probe_result.get(
            "runtime_probe", state_conditioned_phase0_smoke._empty_runtime_probe()
        ),
    )
    payload["runtime_probe"] = dict(runtime_probe)
    python_env = cast(Mapping[str, object], payload.get("python_env", {}))
    payload["python_env"] = {
        **dict(python_env),
        "sys_path_injected": list(
            cast(Sequence[str], probe_result.get("sys_path_injected", []))
        ),
    }
    blockers = _string_list(probe_result.get("blockers", []))
    if not bool(probe_result.get("ok", False)):
        payload = _failure_payload(
            args,
            repo_root=repo_root,
            stage="import_probe",
            reason_code="runtime_dependency_breakage",
            message="import/runtime probe failed: " + ", ".join(blockers),
            exception_type="ImportError",
            detail={
                "errors": dict(
                    cast(Mapping[str, object], probe_result.get("errors", {}))
                ),
                "runtime_probe": dict(runtime_probe),
            },
            blockers=blockers,
            live_checks_attempted=False,
        )
        payload["runtime_probe"] = dict(runtime_probe)
        payload["python_env"] = {
            **dict(cast(Mapping[str, object], payload.get("python_env", {}))),
            "sys_path_injected": list(
                cast(Sequence[str], probe_result.get("sys_path_injected", []))
            ),
        }
        _finalize_system_break_flags(payload)
        return payload

    server_entrypoint = _server_entrypoint_probe(repo_root)
    payload["server_entrypoint"] = server_entrypoint
    if not bool(server_entrypoint.get("exists", False)):
        payload = _failure_payload(
            args,
            repo_root=repo_root,
            stage="server_entrypoint",
            reason_code="server_entrypoint_missing",
            message=f"missing server entrypoint: {server_entrypoint['path']}",
            detail=server_entrypoint,
            blockers=["server_entrypoint_missing"],
            live_checks_attempted=False,
        )
        payload["runtime_probe"] = dict(runtime_probe)
        payload["python_env"] = {
            **dict(cast(Mapping[str, object], payload.get("python_env", {}))),
            "sys_path_injected": list(
                cast(Sequence[str], probe_result.get("sys_path_injected", []))
            ),
        }
        payload["server_entrypoint"] = server_entrypoint
        _finalize_system_break_flags(payload)
        return payload

    try:
        readiness_result = cast(
            Mapping[str, object],
            readiness_fn(args, repo_root=repo_root, output_dir=output_dir),
        )
    except PreflightGateError as exc:
        payload = _failure_payload(
            args,
            repo_root=repo_root,
            stage=exc.stage,
            reason_code=exc.reason_code,
            message=_exception_message(exc),
            exception_type=exc.__class__.__name__,
            detail=exc.detail,
            blockers=exc.blockers,
            live_checks_attempted=bool(str(args.mode) == "smoke"),
        )
        payload["runtime_probe"] = dict(runtime_probe)
        payload["python_env"] = {
            **dict(cast(Mapping[str, object], payload.get("python_env", {}))),
            "sys_path_injected": list(
                cast(Sequence[str], probe_result.get("sys_path_injected", []))
            ),
        }
        payload["server_entrypoint"] = server_entrypoint
        _finalize_system_break_flags(payload)
        return payload

    payload["sim_imports"] = dict(
        cast(
            Mapping[str, object],
            readiness_result.get("sim_imports", _empty_sim_imports()),
        )
    )
    payload["env_resolution"] = dict(
        cast(Mapping[str, object], readiness_result.get("env_resolution", {}))
    )
    payload["live_checks_attempted"] = bool(str(args.mode) == "smoke")
    if "policy_ping" in readiness_result:
        payload["policy_ping"] = dict(
            cast(Mapping[str, object], readiness_result.get("policy_ping", {}))
        )
    if "action_horizon_check" in readiness_result:
        payload["action_horizon_check"] = dict(
            cast(Mapping[str, object], readiness_result.get("action_horizon_check", {}))
        )
    if "smoke" in readiness_result:
        payload["smoke"] = dict(
            cast(Mapping[str, object], readiness_result.get("smoke", {}))
        )
    payload["status"] = "PASS"
    payload["reason_code"] = "ok"
    payload["failure"] = None
    _finalize_system_break_flags(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_wbc_preflight_gate.py",
        description=(
            "Unified preflight gate for GR00T G1 WBC smoke readiness with machine-readable JSON output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("import-only", "smoke"),
        default=DEFAULT_MODE,
        help="import-only checks runtime/env readiness; smoke also requires ping + reset()+step(sample()).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives preflight_report.json and failure note artifacts.",
    )
    parser.add_argument("--env-name", type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--embodiment-tag", type=str, default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--server-host", type=str, default=DEFAULT_SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=int(DEFAULT_SERVER_PORT))
    parser.add_argument(
        "--server-python",
        type=str,
        default=DEFAULT_SERVER_PYTHON,
        help="Optional explicit python executable for the GR00T server subprocess.",
    )
    parser.add_argument("--mujoco-gl", type=str, default=DEFAULT_MUJOCO_GL)
    parser.add_argument(
        "--server-ready-timeout-s",
        type=float,
        default=float(DEFAULT_SERVER_READY_TIMEOUT_S),
    )
    parser.add_argument(
        "--server-ping-timeout-ms",
        type=int,
        default=int(DEFAULT_SERVER_PING_TIMEOUT_MS),
    )
    parser.add_argument(
        "--server-ping-interval-s",
        type=float,
        default=float(DEFAULT_SERVER_PING_INTERVAL_S),
    )
    parser.add_argument(
        "--total-timeout-s",
        type=float,
        default=float(DEFAULT_TOTAL_TIMEOUT_S),
        help="Hard timeout fuse for the entire preflight gate.",
    )
    parser.add_argument(
        "--expected-action-horizon",
        type=int,
        default=int(DEFAULT_EXPECTED_ACTION_HORIZON),
        help="Required policy action horizon for the G1 WBC sim stack.",
    )
    parser.add_argument(
        "--smoke-n-action-steps",
        type=int,
        default=int(DEFAULT_SMOKE_N_ACTION_STEPS),
        help="MultiStepWrapper action chunk length used for reset()+step(sample()) smoke.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=int(DEFAULT_MAX_EPISODE_STEPS),
    )
    parser.add_argument(
        "--spawn-server-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Spawn the GR00T server subprocess when ping fails.",
    )
    parser.add_argument(
        "--kill-server-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Kill the server if this script started it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        _maybe_reexec_into_wbc_venv(REPO_ROOT)
    parser = build_parser()
    args = parser.parse_args(argv)
    state_conditioned_phase0_smoke._apply_env(str(args.mujoco_gl))
    output_dir = Path(args.output_dir)
    payload: dict[str, object]
    try:
        state_conditioned_phase0_smoke._install_alarm_timeout(
            float(args.total_timeout_s) if args.total_timeout_s else None
        )
        payload = run_preflight(args)
    except (OSError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
        payload = _failure_payload(
            args,
            repo_root=REPO_ROOT,
            stage="cli",
            reason_code="runtime_dependency_breakage",
            message=_exception_message(exc),
            exception_type=exc.__class__.__name__,
        )
    except BaseException as exc:
        payload = _failure_payload(
            args,
            repo_root=REPO_ROOT,
            stage="cli",
            reason_code="runtime_dependency_breakage",
            message=_exception_message(exc),
            exception_type=exc.__class__.__name__,
        )
    finally:
        state_conditioned_phase0_smoke._clear_alarm_timeout()

    _finalize_system_break_flags(payload)
    result_text = _json_text(payload)
    report_path: Path | None = None
    failure_note_path: Path | None = None
    try:
        resolved_output_dir = _validate_output_dir(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        report_path = resolved_output_dir / PREFLIGHT_REPORT_JSON_NAME
        _write_json(report_path, payload)
        if payload.get("status") != "PASS":
            failure_note_path = _write_failure_note(
                resolved_output_dir / FAILURE_NOTE_MARKDOWN_NAME,
                payload,
            )
        else:
            stale_failure_note = resolved_output_dir / FAILURE_NOTE_MARKDOWN_NAME
            if stale_failure_note.exists():
                stale_failure_note.unlink()
    except Exception:
        report_path = None
        failure_note_path = None

    if report_path is not None:
        payload["artifact_path"] = str(report_path)
    if failure_note_path is not None:
        payload["failure_note_path"] = str(failure_note_path)
    result_text = _json_text(payload)
    if report_path is not None:
        _write_json(report_path, payload)
    print(result_text)
    return 0 if payload.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
