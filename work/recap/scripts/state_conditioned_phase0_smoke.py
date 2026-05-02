from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_DIR = Path("agent/artifacts/state_conditioned_phase0")
DEFAULT_RUNTIME_LOG_DIR = Path("agent/runtime_logs/state_conditioned_phase0")

DEFAULT_MODE = "live"
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
DEFAULT_TOTAL_TIMEOUT_S = 300.0

DEFAULT_MAX_EPISODE_STEPS = 60
DEFAULT_MAX_OUTER_STEPS = 1
DEFAULT_N_ACTION_STEPS = 30

DEFAULT_POLICY_PHASE = "SEARCH"
DEFAULT_POLICY_MODE = "NOMINAL"

SCHEMA_VERSION = "g1_state_conditioned_phase0_smoke_v1"
IMPORT_PROBE_JSON_NAME = "import_probe_result.json"
LIVE_SMOKE_OK_JSON_NAME = "server_smoke_ok.json"
LIVE_SMOKE_FAIL_JSON_NAME = "server_smoke_fail.json"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import policy as recap_policy
from work.recap import state_conditioned_bucket_a_import
from agent.run import state_conditioned_env_resolution


PHASE_VOCAB: tuple[str, ...] = tuple(
    state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES
)
MODE_VOCAB: tuple[str, ...] = tuple(
    state_conditioned_bucket_a_import.STATE_CONDITIONED_MODES
)


class Phase0SmokeError(RuntimeError):
    stage: str
    detail: Mapping[str, object] | None

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        detail: Mapping[str, object] | None = None,
    ):
        super().__init__(message)
        self.stage = str(stage)
        self.detail = dict(detail) if detail is not None else None


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _build_live_pythonpath(repo_root: Path) -> list[str]:
    return state_conditioned_bucket_a_import._build_live_pythonpath(repo_root)


def _preferred_live_python(repo_root: Path) -> str:
    return state_conditioned_bucket_a_import._preferred_live_python(repo_root)


def _server_entrypoint(repo_root: Path) -> Path:
    return repo_root / "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"


def _runtime_log_dir(repo_root: Path) -> Path:
    path = repo_root / DEFAULT_RUNTIME_LOG_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_phase(value: object) -> str:
    normalized = str(value).strip().upper()
    if normalized not in PHASE_VOCAB:
        raise ValueError(f"policy phase must be one of {PHASE_VOCAB!r}")
    return normalized


def _normalize_mode(value: object) -> str:
    normalized = str(value).strip().upper()
    if normalized not in MODE_VOCAB:
        raise ValueError(f"policy mode must be one of {MODE_VOCAB!r}")
    return normalized


def _policy_condition_payload(phase: object, mode: object) -> dict[str, object]:
    normalized_phase = _normalize_phase(phase)
    normalized_mode = _normalize_mode(mode)
    return {
        "phase": normalized_phase,
        "phase_index": int(PHASE_VOCAB.index(normalized_phase)),
        "mode": normalized_mode,
        "mode_index": int(MODE_VOCAB.index(normalized_mode)),
        "text": state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
            normalized_phase,
            normalized_mode,
        ),
    }


def _json_text(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value]


def _batch_observation_for_policy(obs: Mapping[str, object]) -> dict[str, object]:
    np = importlib.import_module("numpy")
    sanitized_obs = recap_policy.filter_canonical_serving_observation(
        obs,
        field_name="phase0.policy_observation",
    )
    batched: dict[str, object] = {}
    for key, value in sanitized_obs.items():
        if isinstance(value, str):
            batched[key] = [value]
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(item, str) for item in value)
        ):
            batched[key] = [str(item) for item in value]
        elif hasattr(value, "shape"):
            array_value = np.asarray(value)
            if np.issubdtype(array_value.dtype, np.floating):
                array_value = array_value.astype(np.float32, copy=False)
            batched[key] = np.expand_dims(array_value, axis=0)
        else:
            batched[key] = value
    return batched


def _unbatch_policy_action(action: Mapping[str, object]) -> dict[str, object]:
    unbatched: dict[str, object] = {}
    for key, value in action.items():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) == 1:
            unbatched[key] = cast(Any, value)[0]
        else:
            unbatched[key] = value
    return unbatched


def _activate_live_import_roots(repo_root: Path) -> list[str]:
    active_paths = _build_live_pythonpath(repo_root)
    for raw_path in reversed(active_paths):
        if raw_path in sys.path:
            sys.path.remove(raw_path)
        sys.path.insert(0, raw_path)
    return active_paths


def _install_known_live_import_shims() -> None:
    import types

    try:
        obj_utils = importlib.import_module("robocasa.utils.object_utils")
        if not hasattr(obj_utils, "check_obj_upright"):
            obj_cos_fn = getattr(obj_utils, "obj_cos", None)

            def check_obj_upright(
                env: object,
                obj_name: object,
                threshold: float = 0.8,
                symmetric: bool = False,
            ) -> bool:
                if not callable(obj_cos_fn):
                    return False
                try:
                    z_alignment = float(
                        cast(Any, obj_cos_fn)(env, obj_name=obj_name, ref=(0, 0, 1))
                    )
                except Exception:
                    return False
                if bool(symmetric):
                    z_alignment = abs(z_alignment)
                return bool(z_alignment > float(threshold))

            setattr(obj_utils, "check_obj_upright", check_obj_upright)
    except Exception:
        pass

    try:
        _ = importlib.import_module("robocasa.utils.visuals_utls")
    except ModuleNotFoundError:
        module_obj = types.ModuleType("robocasa.utils.visuals_utls")

        class Gradient:
            def __init__(self, *_args: object, **_kwargs: object):
                return None

        def randomize_materials_rgba(*_args: object, **_kwargs: object) -> None:
            return None

        setattr(module_obj, "Gradient", Gradient)
        setattr(module_obj, "randomize_materials_rgba", randomize_materials_rgba)
        sys.modules["robocasa.utils.visuals_utls"] = module_obj
    except Exception:
        pass

    try:
        _ = importlib.import_module("robocasa.wrappers.ik_wrapper")
    except ModuleNotFoundError:
        wrappers_mod = types.ModuleType("robocasa.wrappers")
        ik_mod = types.ModuleType("robocasa.wrappers.ik_wrapper")

        class IKWrapper:
            def __init__(self, env: object, **_kwargs: object):
                self.env = env

            def __getattr__(self, name: str) -> object:
                return getattr(self.env, name)

        setattr(ik_mod, "IKWrapper", IKWrapper)
        sys.modules.setdefault("robocasa.wrappers", wrappers_mod)
        sys.modules["robocasa.wrappers.ik_wrapper"] = ik_mod
    except Exception:
        pass

    try:
        gym_basic_mod = importlib.import_module(
            "robocasa.utils.gym_utils.gymnasium_basic"
        )
        orig_create_env_robosuite = getattr(
            gym_basic_mod,
            "create_env_robosuite",
            None,
        )
        if callable(orig_create_env_robosuite):

            def _patched_create_env_robosuite(
                *args: object,
                **kwargs: object,
            ) -> object:
                if "offscreen" in kwargs and "enable_render" not in kwargs:
                    kwargs["enable_render"] = bool(kwargs.pop("offscreen"))
                else:
                    kwargs.pop("offscreen", None)
                kwargs.pop("onscreen", None)
                kwargs.pop("renderer", None)
                kwargs.pop("render_camera", None)
                kwargs.pop("translucent_robot", None)
                kwargs.pop("ik_indicator", None)
                kwargs.pop("control_freq", None)
                return cast(Any, orig_create_env_robosuite)(*args, **kwargs)

            setattr(
                gym_basic_mod, "create_env_robosuite", _patched_create_env_robosuite
            )
    except Exception:
        pass

    try:
        robots_mod = importlib.import_module("robocasa.models.robots")
        if not hasattr(robots_mod, "GR00T_LOCOMANIP_ENVS_ROBOTS"):
            setattr(robots_mod, "GR00T_LOCOMANIP_ENVS_ROBOTS", {"G1": "g1_sim"})
        if not hasattr(robots_mod, "remove_mimic_joints"):

            def remove_mimic_joints(_gripper: object, action: object) -> object:
                return action

            setattr(robots_mod, "remove_mimic_joints", remove_mimic_joints)
    except Exception:
        pass

    try:
        robosuite_robots_mod = importlib.import_module("robosuite.robots")
        robot_class_mapping = getattr(robosuite_robots_mod, "ROBOT_CLASS_MAPPING", None)
        if isinstance(robot_class_mapping, dict):
            if "G1" not in robot_class_mapping and "GR1" in robot_class_mapping:
                robot_class_mapping["G1"] = robot_class_mapping["GR1"]
        robot_model_mod = importlib.import_module("robosuite.models.robots.robot_model")
        registered_robots = getattr(robot_model_mod, "REGISTERED_ROBOTS", None)
        if isinstance(registered_robots, dict):
            if "G1" not in registered_robots and "GR1" in registered_robots:
                registered_robots["G1"] = registered_robots["GR1"]
    except Exception:
        pass

    try:
        controller_utils = importlib.import_module(
            "gr00t_wbc.control.envs.robocasa.utils.controller_utils"
        )
        orig_update_controller_cfg = getattr(
            controller_utils,
            "update_robosuite_controller_configs",
            None,
        )
        if callable(orig_update_controller_cfg):

            def _patched_update_robosuite_controller_configs(
                robot: str,
                wbc_version: str | None = None,
                enable_gravity_compensation: bool = False,
            ) -> Any:
                cfg = orig_update_controller_cfg(
                    robot=robot,
                    wbc_version=wbc_version,
                    enable_gravity_compensation=enable_gravity_compensation,
                )
                target_name = "default_mink_ik_g1_gear_wbc.json"
                if not str(cfg).endswith(target_name):
                    return cfg
                cfg_path = Path(str(cfg))
                if not cfg_path.is_absolute():
                    try:
                        robocasa_mod = importlib.import_module("robocasa")
                        cfg_path = (
                            Path(str(getattr(robocasa_mod, "__file__", "")))
                            .resolve()
                            .parent
                            / ".."
                            / str(cfg)
                        ).resolve()
                    except Exception:
                        cfg_path = Path(str(cfg))
                if cfg_path.is_file():
                    return cfg

                module_file = Path(
                    str(getattr(controller_utils, "__file__", ""))
                ).resolve()
                gr00t_wbc_root = None
                for parent in (module_file.parent, *module_file.parents):
                    if parent.name == "gr00t_wbc":
                        gr00t_wbc_root = parent
                        break
                if gr00t_wbc_root is None:
                    return cfg

                fallback_cfg = (
                    gr00t_wbc_root
                    / "dexmg"
                    / "gr00trobosuite"
                    / "robosuite"
                    / "examples"
                    / "third_party_controller"
                    / "default_mink_ik_gr1.json"
                )
                if not fallback_cfg.is_file():
                    return cfg
                return str(fallback_cfg)

            setattr(
                controller_utils,
                "update_robosuite_controller_configs",
                _patched_update_robosuite_controller_configs,
            )
    except Exception:
        pass


def _import_module_probe(
    module_name: str,
    *,
    errors: dict[str, str],
    field_name: str,
) -> tuple[bool, object | None]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        errors[field_name] = f"{exc.__class__.__name__}: {_exception_message(exc)}"
        return False, None
    return True, module


def _collect_runtime_probe(
    *, repo_root: Path, add_live_paths: bool
) -> dict[str, object]:
    runtime_probe: dict[str, object] = {
        "flash_attn_2_available": False,
        "gr00t_import_ok": False,
        "gr00t_wbc_import_ok": False,
        "gymnasium_import_ok": False,
        "numpy_import_ok": False,
        "policy_client_import_ok": False,
        "robocasa_import_ok": False,
        "rollout_policy_import_ok": False,
        "server_entrypoint_exists": bool(_server_entrypoint(repo_root).is_file()),
        "torch_bfloat16_cuda_ok": False,
        "torch_cuda_available": False,
        "torch_import_ok": False,
        "video_backend_probe_ok": False,
    }
    errors: dict[str, str] = {}
    active_paths: list[str] = []
    if add_live_paths:
        active_paths = _activate_live_import_roots(repo_root)

    numpy_ok, _numpy_mod = _import_module_probe(
        "numpy", errors=errors, field_name="numpy_import_ok"
    )
    runtime_probe["numpy_import_ok"] = bool(numpy_ok)

    gym_ok, _gym_mod = _import_module_probe(
        "gymnasium", errors=errors, field_name="gymnasium_import_ok"
    )
    runtime_probe["gymnasium_import_ok"] = bool(gym_ok)

    torch_ok, torch_mod = _import_module_probe(
        "torch", errors=errors, field_name="torch_import_ok"
    )
    runtime_probe["torch_import_ok"] = bool(torch_ok)
    if torch_ok and torch_mod is not None:
        try:
            torch_cuda_available = bool(cast(Any, torch_mod).cuda.is_available())
        except Exception as exc:
            torch_cuda_available = False
            errors["torch_cuda_available"] = (
                f"{exc.__class__.__name__}: {_exception_message(exc)}"
            )
        runtime_probe["torch_cuda_available"] = bool(torch_cuda_available)
        if torch_cuda_available:
            try:
                cast(Any, torch_mod).zeros(
                    (1,),
                    device="cuda",
                    dtype=cast(Any, torch_mod).bfloat16,
                )
                runtime_probe["torch_bfloat16_cuda_ok"] = True
            except Exception as exc:
                errors["torch_bfloat16_cuda_ok"] = (
                    f"{exc.__class__.__name__}: {_exception_message(exc)}"
                )
        else:
            errors.setdefault(
                "torch_bfloat16_cuda_ok",
                "RuntimeError: torch.cuda.is_available() returned False",
            )
    else:
        errors.setdefault("torch_cuda_available", "RuntimeError: torch import failed")
        errors.setdefault("torch_bfloat16_cuda_ok", "RuntimeError: torch import failed")

    transformers_ok, transformers_utils_mod = _import_module_probe(
        "transformers.utils",
        errors=errors,
        field_name="flash_attn_2_available",
    )
    if transformers_ok and transformers_utils_mod is not None:
        checker = getattr(transformers_utils_mod, "is_flash_attn_2_available", None)
        if callable(checker):
            try:
                runtime_probe["flash_attn_2_available"] = bool(checker())
                if not bool(runtime_probe["flash_attn_2_available"]):
                    errors["flash_attn_2_available"] = (
                        "RuntimeError: transformers reported flash-attn2 unavailable"
                    )
            except Exception as exc:
                errors["flash_attn_2_available"] = (
                    f"{exc.__class__.__name__}: {_exception_message(exc)}"
                )

    gr00t_ok, _gr00t_mod = _import_module_probe(
        "gr00t", errors=errors, field_name="gr00t_import_ok"
    )
    runtime_probe["gr00t_import_ok"] = bool(gr00t_ok)

    policy_client_ok, _policy_client_mod = _import_module_probe(
        "gr00t.policy.server_client",
        errors=errors,
        field_name="policy_client_import_ok",
    )
    runtime_probe["policy_client_import_ok"] = bool(policy_client_ok)

    rollout_policy_ok, _rollout_mod = _import_module_probe(
        "gr00t.eval.rollout_policy",
        errors=errors,
        field_name="rollout_policy_import_ok",
    )
    runtime_probe["rollout_policy_import_ok"] = bool(rollout_policy_ok)

    robocasa_ok, _robocasa_mod = _import_module_probe(
        "robocasa", errors=errors, field_name="robocasa_import_ok"
    )
    runtime_probe["robocasa_import_ok"] = bool(robocasa_ok)

    if add_live_paths:
        _install_known_live_import_shims()

    gr00t_wbc_ok, _gr00t_wbc_mod = _import_module_probe(
        "gr00t_wbc.control.envs.robocasa.sync_env",
        errors=errors,
        field_name="gr00t_wbc_import_ok",
    )
    runtime_probe["gr00t_wbc_import_ok"] = bool(gr00t_wbc_ok)

    video_backend_ok = False
    for module_name in ("av", "imageio_ffmpeg"):
        try:
            importlib.import_module(module_name)
            video_backend_ok = True
            break
        except Exception:
            continue
    runtime_probe["video_backend_probe_ok"] = bool(video_backend_ok)
    if not video_backend_ok:
        errors["video_backend_probe_ok"] = (
            "RuntimeError: neither av nor imageio_ffmpeg could be imported"
        )

    blocker_keys = [
        "server_entrypoint_exists",
        "numpy_import_ok",
        "gymnasium_import_ok",
        "torch_import_ok",
        "torch_cuda_available",
        "torch_bfloat16_cuda_ok",
        "flash_attn_2_available",
        "gr00t_import_ok",
        "policy_client_import_ok",
        "rollout_policy_import_ok",
        "robocasa_import_ok",
        "gr00t_wbc_import_ok",
        "video_backend_probe_ok",
    ]
    blockers = [key for key in blocker_keys if not bool(runtime_probe.get(key, False))]
    return {
        "ok": not blockers,
        "blockers": blockers,
        "errors": errors,
        "import_gr00t_ok": bool(runtime_probe["gr00t_import_ok"]),
        "runtime_probe": runtime_probe,
        "sys_path_injected": active_paths,
    }


def _normalize_client_host(host: str) -> str:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "normalize_client_host")
    return str(fn(str(host)))


def _is_tcp_port_listening(host: str, port: int, timeout_s: float = 0.2) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "is_tcp_port_listening")
    return bool(fn(str(host), int(port), timeout_s=float(timeout_s)))


def _make_policy_client(host: str, port: int, timeout_ms: int) -> Any:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "make_policy_client")
    return fn(host=str(host), port=int(port), timeout_ms=int(timeout_ms))


def _safe_ping(client: Any, timeout_ms: int) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "safe_ping")
    return bool(fn(client, int(timeout_ms)))


def _safe_kill_server(client: Any, timeout_ms: int) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "safe_kill_server")
    return bool(fn(client, int(timeout_ms)))


def _spawn_server_subprocess(
    cmd: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    env: Mapping[str, str],
) -> subprocess.Popen[str]:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "spawn_server_subprocess")
    return cast(
        subprocess.Popen[str],
        fn(list(cmd), log_path=Path(log_path), cwd=Path(cwd), env=dict(env)),
    )


def _terminate_process(proc: subprocess.Popen[str], timeout_s: float) -> None:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "terminate_process")
    fn(proc, float(timeout_s))


def _apply_env(mujoco_gl: str) -> None:
    value = str(mujoco_gl or "").strip()
    if not value:
        return
    os.environ["MUJOCO_GL"] = value
    if value.lower() == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _build_server_cmd(args: argparse.Namespace, repo_root: Path) -> list[str]:
    exe = str(getattr(args, "server_python", "") or "").strip()
    if not exe:
        exe = _preferred_live_python(repo_root)
    return [
        exe,
        str(_server_entrypoint(repo_root)),
        "--model-path",
        str(args.model_path),
        "--embodiment-tag",
        str(args.embodiment_tag),
        "--use-sim-policy-wrapper",
        "--host",
        str(args.server_host),
        "--port",
        str(int(args.server_port)),
    ]


def _ensure_server_ready(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    server_log: Path,
) -> tuple[Any, subprocess.Popen[str] | None, bool, str]:
    host_for_client = _normalize_client_host(str(args.server_host))
    client = _make_policy_client(
        host=host_for_client,
        port=int(args.server_port),
        timeout_ms=int(args.server_ping_timeout_ms),
    )
    if _safe_ping(client, int(args.server_ping_timeout_ms)):
        return client, None, False, host_for_client

    if not bool(args.spawn_server_if_missing):
        raise Phase0SmokeError(
            "policy_ping",
            "no responsive server found and spawn_server_if_missing is disabled",
        )

    if _is_tcp_port_listening(host_for_client, int(args.server_port)):
        raise Phase0SmokeError(
            "policy_ping",
            "port is occupied but PolicyClient.ping() failed; refusing to kill an unknown process",
        )

    server_py = _server_entrypoint(repo_root)
    if not server_py.is_file():
        raise Phase0SmokeError(
            "import_probe", f"missing server entrypoint: {server_py}"
        )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GR00T_SKIP_WBC_REEXEC"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(_build_live_pythonpath(repo_root))
    if str(args.mujoco_gl or "").strip():
        env["MUJOCO_GL"] = str(args.mujoco_gl)
        if str(args.mujoco_gl).lower() == "egl":
            env.setdefault("PYOPENGL_PLATFORM", "egl")

    proc = _spawn_server_subprocess(
        _build_server_cmd(args, repo_root),
        log_path=server_log,
        cwd=repo_root,
        env=env,
    )
    t0 = time.monotonic()
    while True:
        if _safe_ping(client, int(args.server_ping_timeout_ms)):
            return client, proc, True, host_for_client
        if proc.poll() is not None:
            raise Phase0SmokeError(
                "policy_ping",
                f"server subprocess exited early rc={proc.returncode}; see {server_log}",
            )
        elapsed_s = time.monotonic() - t0
        if elapsed_s > float(args.server_ready_timeout_s):
            raise Phase0SmokeError(
                "policy_ping",
                f"timeout waiting for ping ok after {int(elapsed_s)}s; see {server_log}",
            )
        time.sleep(float(args.server_ping_interval_s))


def _infer_action_horizon(modality_cfg: Mapping[str, object]) -> int | None:
    try:
        action_cfg = modality_cfg.get("action")
        if action_cfg is None:
            return None
        delta_indices = list(getattr(action_cfg, "delta_indices", []) or [])
        if not delta_indices:
            return None
        return int(len(delta_indices))
    except Exception:
        return None


def _delta_indices_from_modality(modality_cfg: object) -> list[int]:
    if modality_cfg is None:
        return []
    raw = getattr(modality_cfg, "delta_indices", None)
    if raw is None:
        return []
    try:
        return [int(x) for x in list(raw)]
    except Exception:
        try:
            return [int(x) for x in raw]
        except Exception:
            return []


def _registered_g1_env_ids(gym_module: object) -> list[str]:
    registry = getattr(getattr(gym_module, "envs", None), "registry", None)
    keys = getattr(registry, "keys", None)
    if not callable(keys):
        return []
    env_ids = list(cast(Any, keys)())
    return sorted(
        [
            str(env_id)
            for env_id in env_ids
            if str(env_id).startswith("gr00tlocomanip_g1_sim/")
        ]
    )


def _resolve_env_name(
    gym_module: object, requested_env_name: str
) -> tuple[str, bool, list[str]]:
    try:
        resolution = (
            state_conditioned_env_resolution.resolve_apple_to_plate_g1_env_name(
                gym_module,
                requested_env_name=str(requested_env_name),
            )
        )
    except state_conditioned_env_resolution.StateConditionedEnvResolutionError as exc:
        raise Phase0SmokeError(
            "env_lookup",
            _exception_message(exc),
            detail=exc.to_machine_payload(),
        ) from exc
    return (
        str(resolution["resolved_env_name"]),
        bool(resolution["alias_applied"]),
        list(cast(Sequence[str], resolution["registered_env_ids"])),
    )


def _make_live_env(
    *,
    resolved_env_name: str,
    n_action_steps: int,
    max_episode_steps: int,
    video_delta_indices: object,
    state_delta_indices: object,
) -> Any:
    sync_env_mod = importlib.import_module("gr00t_wbc.control.envs.robocasa.sync_env")
    base_cfg_mod = importlib.import_module(
        "gr00t_wbc.control.main.teleop.configs.configs"
    )
    n1_utils_mod = importlib.import_module("gr00t_wbc.control.utils.n1_utils")
    ms_mod = importlib.import_module("gr00t.eval.sim.wrapper.multistep_wrapper")

    BaseConfig = getattr(base_cfg_mod, "BaseConfig")
    WholeBodyControlWrapper = getattr(n1_utils_mod, "WholeBodyControlWrapper")
    MultiStepWrapper = getattr(ms_mod, "MultiStepWrapper")

    class_name = str(resolved_env_name).split("/", 1)[-1]
    env_class = getattr(sync_env_mod, class_name, None)
    if env_class is None:
        raise Phase0SmokeError(
            "env_lookup",
            f"missing sync_env class for resolved env {resolved_env_name!r}",
        )

    base_env = env_class(
        onscreen=False,
        offscreen=True,
        enable_waist=True,
        randomize_cameras=False,
        camera_names=["robot0_oak_egoview", "robot0_rs_tppview"],
    )
    wbc_config = BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict()
    wbc_env = WholeBodyControlWrapper(base_env, wbc_config)
    return MultiStepWrapper(
        wbc_env,
        video_delta_indices=video_delta_indices,
        state_delta_indices=state_delta_indices,
        n_action_steps=int(n_action_steps),
        max_episode_steps=int(max_episode_steps),
        terminate_on_success=True,
    )


def _run_live_checks(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    del output_dir
    runtime_dir = _runtime_log_dir(repo_root)
    server_log = runtime_dir / "00_server.log"
    client: Any | None = None
    proc: subprocess.Popen[str] | None = None
    started_by_me = False
    host_for_client = _normalize_client_host(str(args.server_host))
    env: Any | None = None
    try:
        client, proc, started_by_me, host_for_client = _ensure_server_ready(
            args,
            repo_root=repo_root,
            server_log=server_log,
        )
        assert client is not None
        modality_cfg = cast(Mapping[str, object], client.get_modality_config())
        action_horizon = _infer_action_horizon(modality_cfg)

        gym = importlib.import_module("gymnasium")
        np = importlib.import_module("numpy")
        importlib.import_module("robocasa")
        resolved_env_name, env_alias_applied, registered_env_ids = _resolve_env_name(
            gym,
            str(args.env_name),
        )

        n_action_steps = (
            int(action_horizon)
            if action_horizon is not None
            else int(args.n_action_steps)
        )
        video_delta = _delta_indices_from_modality(modality_cfg.get("video"))
        state_delta = _delta_indices_from_modality(modality_cfg.get("state"))
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

        env = cast(
            Any,
            _make_live_env(
                resolved_env_name=str(resolved_env_name),
                n_action_steps=int(n_action_steps),
                max_episode_steps=int(args.max_episode_steps),
                video_delta_indices=video_delta_indices,
                state_delta_indices=state_delta_indices,
            ),
        )
        assert env is not None
        try:
            obs, _info = env.reset()
        except Exception as exc:
            raise Phase0SmokeError(
                "env_reset",
                f"env.reset() failed: {_exception_message(exc)}",
            ) from exc

        try:
            client.reset()
        except Exception as exc:
            raise Phase0SmokeError(
                "policy_reset",
                f"client.reset() failed: {_exception_message(exc)}",
            ) from exc

        try:
            policy_obs = _batch_observation_for_policy(cast(Mapping[str, object], obs))
            action, _action_info = client.get_action(policy_obs)
        except Exception as exc:
            raise Phase0SmokeError(
                "get_action",
                f"client.get_action() failed: {_exception_message(exc)}",
            ) from exc
        if not isinstance(action, dict):
            raise Phase0SmokeError(
                "get_action",
                f"client.get_action() returned {type(action).__name__}, expected dict",
            )
        env_action = _unbatch_policy_action(action)

        try:
            _next_obs, reward, term, trunc, _step_info = env.step(env_action)
        except Exception as exc:
            raise Phase0SmokeError(
                "rollout_episode",
                f"env.step() failed: {_exception_message(exc)}",
            ) from exc

        rollout_done_observed = bool(term) or bool(trunc)
        return {
            "policy_ping_ok": True,
            "get_action_ok": True,
            "rollout_episode_ok": True,
            "rollout_done_observed": bool(rollout_done_observed),
            "rollout_outer_steps": int(min(1, int(args.max_outer_steps))),
            "rollout_stop_reason": (
                "env_done" if rollout_done_observed else "bounded_outer_steps"
            ),
            "env_name_requested": str(args.env_name),
            "env_name_resolved": str(resolved_env_name),
            "env_alias_applied": bool(env_alias_applied),
            "registered_env_ids": registered_env_ids,
            "reward_sample": cast(Any, np.asarray(reward)).reshape(-1).tolist(),
            "policy_obs_q_shape": list(
                cast(Sequence[int], getattr(policy_obs.get("q"), "shape", ()))
            )
            if isinstance(policy_obs, dict) and "q" in policy_obs
            else None,
            "env_action_left_arm_shape": list(
                cast(
                    Sequence[int],
                    getattr(env_action.get("action.left_arm"), "shape", ()),
                )
            )
            if isinstance(env_action, dict) and "action.left_arm" in env_action
            else None,
            "modality_config_keys": sorted([str(key) for key in modality_cfg.keys()]),
            "action_horizon": int(action_horizon)
            if action_horizon is not None
            else None,
            "server": {
                "host": host_for_client,
                "port": int(args.server_port),
                "spawned": bool(started_by_me),
                "reused_existing": not bool(started_by_me),
                "server_log": str(server_log),
            },
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if client is not None and started_by_me and bool(args.kill_server_on_exit):
            try:
                _safe_kill_server(client, int(args.server_ping_timeout_ms))
            except Exception:
                pass
        if proc is not None and started_by_me and bool(args.kill_server_on_exit):
            _terminate_process(proc, timeout_s=10.0)


def _install_alarm_timeout(timeout_s: float | None) -> None:
    if timeout_s is None:
        return
    try:
        timeout_i = int(float(timeout_s))
    except Exception:
        return
    if timeout_i <= 0 or not hasattr(signal, "SIGALRM"):
        return

    def _handler(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"Timed out after {timeout_i}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_i)


def _clear_alarm_timeout() -> None:
    if hasattr(signal, "SIGALRM"):
        try:
            signal.alarm(0)
        except Exception:
            pass


def _empty_runtime_probe() -> dict[str, object]:
    return {
        "flash_attn_2_available": False,
        "gr00t_import_ok": False,
        "gr00t_wbc_import_ok": False,
        "gymnasium_import_ok": False,
        "numpy_import_ok": False,
        "policy_client_import_ok": False,
        "robocasa_import_ok": False,
        "rollout_policy_import_ok": False,
        "server_entrypoint_exists": False,
        "torch_bfloat16_cuda_ok": False,
        "torch_cuda_available": False,
        "torch_import_ok": False,
        "video_backend_probe_ok": False,
    }


def _base_payload(args: argparse.Namespace, *, repo_root: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_mode": str(args.mode),
        "status": "FAIL",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "failure": None,
        "live_checks_requested": bool(str(args.mode) == "live"),
        "live_checks_attempted": False,
        "import_gr00t_ok": False,
        "policy_ping_ok": False,
        "get_action_ok": False,
        "rollout_episode_ok": False,
        "runtime_probe": _empty_runtime_probe(),
        "modality_config_keys": [],
        "action_horizon": None,
        "server": {
            "host": str(args.server_host),
            "port": int(args.server_port),
            "spawned": False,
            "reused_existing": False,
        },
        "policy_condition": _policy_condition_payload(
            args.policy_phase,
            args.policy_mode,
        ),
        "output_dir": str(args.output_dir),
        "repo_root": str(repo_root),
    }


def _failure_payload(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    stage: str,
    message: str,
    exception_type: str = "RuntimeError",
    detail: Mapping[str, object] | None = None,
    runtime_probe: Mapping[str, object] | None = None,
    import_gr00t_ok: bool = False,
    blockers: Sequence[str] | None = None,
    live_checks_attempted: bool = False,
) -> dict[str, object]:
    payload = _base_payload(args, repo_root=repo_root)
    payload["status"] = "FAIL"
    payload["live_checks_attempted"] = bool(live_checks_attempted)
    payload["runtime_probe"] = dict(runtime_probe or _empty_runtime_probe())
    payload["import_gr00t_ok"] = bool(import_gr00t_ok)
    payload["failure"] = {
        "stage": str(stage),
        "type": str(exception_type),
        "message": str(message),
        "blockers": list(blockers or []),
        "detail": dict(detail) if detail is not None else None,
    }
    return payload


def run_phase0_smoke(
    args: argparse.Namespace,
    *,
    import_probe_fn: Any | None = None,
    live_checks_fn: Any | None = None,
) -> dict[str, object]:
    repo_root = REPO_ROOT
    output_dir = _validate_output_dir(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if import_probe_fn is None:
        import_probe_fn = _collect_runtime_probe
    if live_checks_fn is None:
        live_checks_fn = _run_live_checks

    payload = _base_payload(args, repo_root=repo_root)
    payload["output_dir"] = str(output_dir)

    probe_result = cast(
        Mapping[str, object],
        import_probe_fn(repo_root=repo_root, add_live_paths=bool(args.mode == "live")),
    )
    runtime_probe = cast(
        Mapping[str, object],
        probe_result.get("runtime_probe", _empty_runtime_probe()),
    )
    payload["runtime_probe"] = dict(runtime_probe)
    payload["import_gr00t_ok"] = bool(probe_result.get("import_gr00t_ok", False))
    payload["sys_path_injected"] = _string_list(
        probe_result.get("sys_path_injected", [])
    )
    blockers = _string_list(probe_result.get("blockers", []))

    if not bool(probe_result.get("ok", False)):
        return _failure_payload(
            args,
            repo_root=repo_root,
            stage="import_probe",
            message="import/runtime probe failed: " + ", ".join(blockers),
            exception_type="ImportError",
            runtime_probe=runtime_probe,
            import_gr00t_ok=bool(probe_result.get("import_gr00t_ok", False)),
            blockers=blockers,
            live_checks_attempted=False,
        )

    if args.mode == "import-only":
        payload["status"] = "PASS"
        payload["failure"] = None
        return payload

    try:
        live_result = cast(
            Mapping[str, object],
            live_checks_fn(args, repo_root=repo_root, output_dir=output_dir),
        )
    except Phase0SmokeError as exc:
        return _failure_payload(
            args,
            repo_root=repo_root,
            stage=exc.stage,
            message=_exception_message(exc),
            exception_type=exc.__class__.__name__,
            detail=exc.detail,
            runtime_probe=runtime_probe,
            import_gr00t_ok=bool(probe_result.get("import_gr00t_ok", False)),
            live_checks_attempted=True,
        )
    except Exception as exc:
        return _failure_payload(
            args,
            repo_root=repo_root,
            stage="live_smoke",
            message=_exception_message(exc),
            exception_type=exc.__class__.__name__,
            runtime_probe=runtime_probe,
            import_gr00t_ok=bool(probe_result.get("import_gr00t_ok", False)),
            live_checks_attempted=True,
        )

    payload.update(dict(live_result))
    payload["status"] = "PASS"
    payload["failure"] = None
    payload["live_checks_attempted"] = True
    payload["policy_ping_ok"] = bool(live_result.get("policy_ping_ok", False))
    payload["get_action_ok"] = bool(live_result.get("get_action_ok", False))
    payload["rollout_episode_ok"] = bool(live_result.get("rollout_episode_ok", False))
    return payload


def _result_path(output_dir: Path, *, mode: str, status: str) -> Path:
    normalized_mode = str(mode)
    normalized_status = str(status)
    if normalized_mode == "live":
        filename = (
            LIVE_SMOKE_OK_JSON_NAME
            if normalized_status == "PASS"
            else LIVE_SMOKE_FAIL_JSON_NAME
        )
    else:
        filename = IMPORT_PROBE_JSON_NAME
    return output_dir / filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="state_conditioned_phase0_smoke.py",
        description=(
            "Task 2 phase-0 live/import smoke for GR00T runtime readiness with machine-readable JSON outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("import-only", "live"),
        default=DEFAULT_MODE,
        help="import-only probes runtime blockers only; live continues into ping/get_action/rollout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives machine-readable phase-0 JSON artifacts.",
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
        help="Hard timeout fuse for the entire smoke run.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=int(DEFAULT_MAX_EPISODE_STEPS),
    )
    parser.add_argument(
        "--max-outer-steps",
        type=int,
        default=int(DEFAULT_MAX_OUTER_STEPS),
        help="Bounded outer rollout steps; default keeps phase-0 smoke minimal.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=int(DEFAULT_N_ACTION_STEPS),
        help="Fallback action horizon when server modality_config lacks delta_indices.",
    )
    parser.add_argument(
        "--policy-phase",
        type=str,
        default=DEFAULT_POLICY_PHASE,
        help="Metadata-only canonical phase token recorded in the smoke JSON.",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        default=DEFAULT_POLICY_MODE,
        help="Metadata-only canonical mode token recorded in the smoke JSON.",
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
        default=False,
        help="Kill the server if this script started it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_env(str(args.mujoco_gl))
    output_dir = Path(args.output_dir)
    payload: dict[str, object]
    try:
        _install_alarm_timeout(
            float(args.total_timeout_s) if args.total_timeout_s else None
        )
        payload = run_phase0_smoke(args)
    except (OSError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
        payload = _failure_payload(
            args,
            repo_root=REPO_ROOT,
            stage="cli",
            message=_exception_message(exc),
            exception_type=exc.__class__.__name__,
        )
    finally:
        _clear_alarm_timeout()

    result_text = _json_text(payload)
    result_path: Path | None = None
    try:
        resolved_output_dir = _validate_output_dir(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        result_path = _result_path(
            resolved_output_dir,
            mode=str(args.mode),
            status=str(payload.get("status", "FAIL")),
        )
        _write_json(result_path, payload)
    except Exception:
        result_path = None

    if result_path is not None:
        payload["artifact_path"] = str(result_path)
        result_text = _json_text(payload)
        _write_json(result_path, payload)
    print(result_text)
    return 0 if payload.get("status") == "PASS" else 1


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "IMPORT_PROBE_JSON_NAME",
    "LIVE_SMOKE_FAIL_JSON_NAME",
    "LIVE_SMOKE_OK_JSON_NAME",
    "PHASE_VOCAB",
    "MODE_VOCAB",
    "SCHEMA_VERSION",
    "Phase0SmokeError",
    "build_parser",
    "main",
    "run_phase0_smoke",
]


if __name__ == "__main__":
    raise SystemExit(main())
