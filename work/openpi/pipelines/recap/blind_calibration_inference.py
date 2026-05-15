"""Iter8 real-inference helpers. Heavy imports (jax/torch via openpi) are lazy."""

from __future__ import annotations

import collections
import dataclasses
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from work.openpi.recap.real_variant_policy_contract import (
    LIBERO_ASSET_ID,
    POLICY_CONFIG_MISMATCH_BLOCKER,
    REAL_VARIANT_DATA_FACTORY_KIND,
    REAL_VARIANT_POLICY_CONFIG_NAME,
    extract_policy_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
OPENPI_ROOT = REPO_ROOT / "submodules/openpi"
_INITIALIZED: bool = False
LIBERO_ENV_RESOLUTION = 256
LIBERO_RESIZE_SIZE = 224
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_NUM_STEPS_WAIT = 10
LIBERO_REPLAN_STEPS = 5


@dataclass(frozen=True)
class LocalCheckpointPolicySpec:
    config_name: str
    manifest_path: Path | None = None
    policy_contract: dict[str, Any] | None = None


def _initialize_jax_runtime() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    _bootstrap_openpi_paths()
    _ensure_libero_config_path()
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    _INITIALIZED = True


def load_variant_A(authority_manifest_path: Path):
    """Load variant A (pi0_libero) policy.

    The blind-calibration path passes an authority manifest, while the v22 formal
    eval authority manifest may pass the resolved stock checkpoint directory
    directly.  Accept both forms so the runner does not depend on which
    authority layer supplied the stock pi0_libero source.
    """
    _initialize_jax_runtime()
    try:
        checkpoint_dir = _resolve_variant_a_checkpoint_dir(authority_manifest_path)
        return _load_libero_policy(checkpoint_dir, config_name="pi0_libero")
    except RuntimeError as exc:
        if str(exc).startswith("BLOCK_A_STOCK_AUTHORITY_MISSING"):
            raise
        raise RuntimeError("BLOCK_A_CHECKPOINT_LOAD_FAILED:" + str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("BLOCK_A_CHECKPOINT_LOAD_FAILED:" + str(exc)) from exc


def _resolve_variant_a_checkpoint_dir(authority_or_checkpoint_path: Path) -> Path:
    source_path = authority_or_checkpoint_path.expanduser()
    if source_path.is_dir():
        try:
            return _resolve_checkpoint_dir(source_path)
        except FileNotFoundError as exc:
            raise RuntimeError("BLOCK_A_STOCK_AUTHORITY_MISSING") from exc
    if not source_path.is_file():
        raise RuntimeError("BLOCK_A_STOCK_AUTHORITY_MISSING")

    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    local_resolved_path = manifest.get("local_resolved_path")
    if not local_resolved_path:
        raise RuntimeError("BLOCK_A_STOCK_AUTHORITY_MISSING")
    local_path = Path(str(local_resolved_path)).expanduser()
    if not local_path.exists():
        raise RuntimeError("BLOCK_A_STOCK_AUTHORITY_MISSING")
    try:
        return _resolve_checkpoint_dir(local_path)
    except FileNotFoundError as exc:
        raise RuntimeError("BLOCK_A_STOCK_AUTHORITY_MISSING") from exc


def load_variant_B_optional(local_checkpoint_path: Path):
    """Load local B control checkpoint for calibration-only sanity scan."""
    _initialize_jax_runtime()
    local_path = local_checkpoint_path.expanduser()
    if not local_path.exists():
        return None
    try:
        checkpoint_dir = _resolve_checkpoint_dir(local_path)
        policy_spec = _resolve_local_checkpoint_policy_spec(local_path, checkpoint_dir)
        return _load_libero_policy(checkpoint_dir, policy_spec=policy_spec)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("BLOCK_B_CHECKPOINT_LOAD_FAILED:" + str(exc)) from exc


def _run_real_episode(env, policy, *, max_steps: int, seed: int) -> dict:
    """Run one real LIBERO episode against a local OpenPI policy."""
    _initialize_jax_runtime()
    steps_taken = 0
    done = False
    trace_completeness = 1.0
    terminal_reason = "timeout"
    action_plan: collections.deque[Any] = collections.deque()
    obs: Any = None
    try:
        obs = _reset_episode_env(env, seed=seed)
        for _ in range(_num_steps_wait(env)):
            obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
            if done:
                success = _task_succeeded(env, done=done)
                return {
                    "seed": seed,
                    "success": bool(success),
                    "timeout_flag": False,
                    "trace_completeness": 1.0,
                    "steps_taken": steps_taken,
                    "terminal_reason": "success" if success else "env_done",
                }
        while steps_taken < max(0, int(max_steps)):
            if not action_plan:
                element = _build_policy_observation(obs, _task_description(env))
                action_chunk = policy.infer(element)["actions"]
                if len(action_chunk) < 1:
                    raise RuntimeError("policy_returned_empty_action_chunk")
                action_plan.extend(action_chunk[:LIBERO_REPLAN_STEPS])
            action = action_plan.popleft()
            obs, _reward, done, _info = env.step(_action_to_env(action))
            steps_taken += 1
            if done:
                success = _task_succeeded(env, done=done)
                terminal_reason = "success" if success else "env_done"
                return {
                    "seed": seed,
                    "success": bool(success),
                    "timeout_flag": False,
                    "trace_completeness": trace_completeness,
                    "steps_taken": steps_taken,
                    "terminal_reason": terminal_reason,
                }
        return {
            "seed": seed,
            "success": False,
            "timeout_flag": True,
            "trace_completeness": trace_completeness,
            "steps_taken": steps_taken,
            "terminal_reason": "timeout",
        }
    except Exception as exc:  # noqa: BLE001
        attempted = max(1, max(0, int(max_steps)))
        trace_completeness = min(0.99, steps_taken / attempted)
        return {
            "seed": seed,
            "success": False,
            "timeout_flag": False,
            "trace_completeness": trace_completeness,
            "steps_taken": steps_taken,
            "terminal_reason": "error:" + str(exc),
        }


@dataclass
class LiberoEpisodeEnv:
    env: Any
    task_description: str
    initial_state: Any
    num_steps_wait: int = LIBERO_NUM_STEPS_WAIT

    def reset(self, *, seed: int) -> Any:
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        self.env.reset()
        return self.env.set_init_state(self.initial_state)

    def step(self, action: Any) -> Any:
        return self.env.step(action)

    def task_succeeded(self) -> bool:
        return _task_succeeded(self.env, done=False)

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


def build_libero_episode_env(
    *,
    suite_family: str,
    tasks: tuple[str, ...],
    episode_index: int,
    seed: int,
) -> LiberoEpisodeEnv:
    benchmark_module = __import__("libero.libero.benchmark", fromlist=["benchmark"])
    libero_module = __import__("libero.libero", fromlist=["get_libero_path"])
    envs_module = __import__("libero.libero.envs", fromlist=["OffScreenRenderEnv"])

    suite_name = resolve_libero_suite_name(suite_family)
    task_suite = benchmark_module.get_benchmark_dict()[suite_name]()
    task_ids = _resolve_task_ids(task_suite, suite_family=suite_family, tasks=tasks)
    task_id = task_ids[episode_index % len(task_ids)]
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    if len(initial_states) == 0:
        raise RuntimeError(f"libero_initial_states_missing:{suite_name}:{task_id}")
    trial_index = episode_index % len(initial_states)
    task_bddl_file = (
        Path(libero_module.get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = envs_module.OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return LiberoEpisodeEnv(
        env=env,
        task_description=str(task.language),
        initial_state=initial_states[trial_index],
    )


def resolve_libero_suite_name(suite_family: str) -> str:
    aliases = {
        "libero_spatial_expanded": "libero_spatial",
        "other_locally_supported_LIBERO_suites": "libero_10",
    }
    return aliases.get(suite_family, suite_family)


def resolve_suite_max_steps(suite_family: str) -> int:
    return PER_SUITE_MAX_STEPS[resolve_libero_suite_name(suite_family)]


def _resolve_checkpoint_dir(path: Path) -> Path:
    path = path.expanduser()
    if _looks_like_openpi_checkpoint(path):
        return path
    best = path / "best"
    if _looks_like_openpi_checkpoint(best):
        return best
    raise FileNotFoundError(f"openpi_checkpoint_payload_missing:{path}")


def _looks_like_openpi_checkpoint(path: Path) -> bool:
    return path.is_dir() and (
        (path / "model.safetensors").is_file()
        or (path / "params" / "manifest.ocdbt").is_file()
        or (path / "params" / "_METADATA").is_file()
    )


def _load_libero_policy(
    checkpoint_dir: Path,
    *,
    config_name: str | None = None,
    policy_spec: LocalCheckpointPolicySpec | None = None,
):
    from openpi.policies import policy_config
    from openpi.training import config as training_config

    if policy_spec is None:
        if config_name is None:
            raise ValueError("config_name or policy_spec is required")
        policy_spec = LocalCheckpointPolicySpec(config_name=config_name)
    train_config = _build_train_config_for_policy_spec(
        training_config,
        policy_spec=policy_spec,
    )
    return policy_config.create_trained_policy(train_config, checkpoint_dir)


def _infer_local_checkpoint_config(local_path: Path, checkpoint_dir: Path) -> str:
    return _resolve_local_checkpoint_policy_spec(local_path, checkpoint_dir).config_name


def _resolve_local_checkpoint_policy_spec(
    local_path: Path,
    checkpoint_dir: Path,
) -> LocalCheckpointPolicySpec:
    manifest_path, manifest = _read_export_manifest(local_path, checkpoint_dir)
    if manifest is not None:
        policy_contract = extract_policy_contract(manifest)
        if policy_contract is not None:
            _assert_real_variant_policy_contract_compatible(policy_contract)
            return LocalCheckpointPolicySpec(
                config_name=REAL_VARIANT_POLICY_CONFIG_NAME,
                manifest_path=manifest_path,
                policy_contract=policy_contract,
            )
        schema = str(manifest.get("schema_version", ""))
        if schema.startswith("openpi_real_variant_export"):
            raise RuntimeError(
                f"{POLICY_CONFIG_MISMATCH_BLOCKER}:"
                f"real_variant_export_manifest_missing_policy_contract:{manifest_path}"
            )

    for path in (
        local_path / "checkpoint_provenance.json",
        checkpoint_dir / "checkpoint_provenance.json",
        checkpoint_dir.parent / "checkpoint_provenance.json",
        local_path / "train_manifest.json",
        checkpoint_dir / "train_manifest.json",
    ):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in _walk_config_values(payload):
            if value in {"pi0_libero", "pi05_libero"}:
                return LocalCheckpointPolicySpec(config_name=value)
    return LocalCheckpointPolicySpec(config_name="pi0_libero")


def _read_export_manifest(
    local_path: Path,
    checkpoint_dir: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    for path in (
        local_path / "export_manifest.json",
        checkpoint_dir / "export_manifest.json",
        checkpoint_dir.parent / "export_manifest.json",
        local_path / "checkpoint" / "export_manifest.json",
        local_path / "best" / "export_manifest.json",
    ):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        return path, payload
    return None, None


def _assert_real_variant_policy_contract_compatible(
    policy_contract: dict[str, Any],
) -> None:
    data_factory_kind = policy_contract.get("data_factory_kind")
    if data_factory_kind != REAL_VARIANT_DATA_FACTORY_KIND:
        raise RuntimeError(
            f"{POLICY_CONFIG_MISMATCH_BLOCKER}:"
            f"unsupported_data_factory_kind:{data_factory_kind}"
        )
    if bool(policy_contract.get("extra_delta_transform", False)):
        raise RuntimeError(
            f"{POLICY_CONFIG_MISMATCH_BLOCKER}:"
            "real_variant_declares_extra_delta_transform_true"
        )
    for graph_key in ("training_transform_graph", "inference_transform_graph"):
        graph = policy_contract.get(graph_key)
        if not isinstance(graph, dict):
            continue
        outputs = graph.get("outputs")
        if outputs is None:
            continue
        if list(outputs) != ["LiberoOutputs"]:
            raise RuntimeError(
                f"{POLICY_CONFIG_MISMATCH_BLOCKER}:"
                f"{graph_key}_outputs_mismatch:{outputs}"
            )


def _build_train_config_for_policy_spec(
    training_config: Any,
    *,
    policy_spec: LocalCheckpointPolicySpec,
):
    if policy_spec.config_name != REAL_VARIANT_POLICY_CONFIG_NAME:
        return training_config.get_config(policy_spec.config_name)
    if policy_spec.policy_contract is None:
        raise RuntimeError(
            f"{POLICY_CONFIG_MISMATCH_BLOCKER}:"
            f"{REAL_VARIANT_POLICY_CONFIG_NAME}_missing_policy_contract"
        )
    base_config_name = str(
        policy_spec.policy_contract.get("base_train_config_name", "pi0_libero")
    )
    base_config = training_config.get_config(base_config_name)
    data_factory = training_config.SimpleDataConfig(
        repo_id=LIBERO_ASSET_ID,
        assets=training_config.AssetsConfig(asset_id=LIBERO_ASSET_ID),
        base_config=training_config.DataConfig(
            prompt_from_task=False,
            action_sequence_keys=("action",),
        ),
        data_transforms=_build_recap_real_variant_inference_transforms,
    )
    return dataclasses.replace(
        base_config,
        name=REAL_VARIANT_POLICY_CONFIG_NAME,
        data=data_factory,
    )


def _build_recap_real_variant_inference_transforms(model_config: Any):
    import openpi.transforms as transforms
    from openpi.policies import libero_policy

    return transforms.Group(
        inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
        outputs=[libero_policy.LiberoOutputs()],
    )


def _walk_config_values(value: Any) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in {"config", "train_config", "train_config_name", "policy_config"}:
                found.append(str(item))
            found.extend(_walk_config_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_config_values(item))
    return tuple(found)


def _bootstrap_openpi_paths() -> None:
    for path in (
        OPENPI_ROOT / "third_party/libero",
        OPENPI_ROOT / "packages/openpi-client/src",
        OPENPI_ROOT / "src",
    ):
        if not path.exists():
            continue
        text = str(path)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    work_path = str(REPO_ROOT / "work")
    while work_path in sys.path:
        sys.path.remove(work_path)
    sys.path.append(work_path)
    module = sys.modules.get("openpi")
    module_file = str(getattr(module, "__file__", "") or "")
    if module_file.startswith(str(REPO_ROOT / "work/openpi")):
        del sys.modules["openpi"]


def _ensure_libero_config_path() -> None:
    if os.environ.get("LIBERO_CONFIG_PATH"):
        return
    benchmark_root = OPENPI_ROOT / "third_party/libero/libero/libero"
    config_dir = REPO_ROOT / "agent/runtime_logs/openpi_libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_text = "\n".join(
        [
            f"benchmark_root: {benchmark_root}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"init_states: {benchmark_root / 'init_files'}",
            f"datasets: {benchmark_root.parent / 'datasets'}",
            f"assets: {benchmark_root / 'assets'}",
            "",
        ]
    )
    (config_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def _reset_episode_env(env: Any, *, seed: int) -> Any:
    try:
        return env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        return env.reset() if hasattr(env, "reset") else None


def _num_steps_wait(env: Any) -> int:
    return int(getattr(env, "num_steps_wait", LIBERO_NUM_STEPS_WAIT))


def _task_description(env: Any) -> str:
    return str(getattr(env, "task_description", ""))


def _build_policy_observation(obs: Any, prompt: str) -> dict[str, Any]:
    import numpy as np
    from openpi_client import image_tools

    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(image, LIBERO_RESIZE_SIZE, LIBERO_RESIZE_SIZE)
    )
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, LIBERO_RESIZE_SIZE, LIBERO_RESIZE_SIZE)
    )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": prompt,
    }


def _quat2axisangle(quat: Any) -> Any:
    import numpy as np

    quat = np.asarray(quat).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _action_to_env(action: Any) -> Any:
    return action.tolist() if hasattr(action, "tolist") else action


def _task_succeeded(env: Any, *, done: bool) -> bool:
    for attr in ("task_succeeded", "_check_success", "check_success"):
        method = getattr(env, attr, None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                continue
    return bool(done)


def _resolve_task_ids(task_suite: Any, *, suite_family: str, tasks: tuple[str, ...]) -> tuple[int, ...]:
    task_count = int(task_suite.n_tasks)
    explicit: list[int] = []
    for raw in tasks:
        text = str(raw).strip()
        if text.isdigit():
            explicit.append(int(text))
            continue
        if text.startswith("task_") and text.removeprefix("task_").isdigit():
            explicit.append(int(text.removeprefix("task_")))
    explicit = [task_id for task_id in explicit if 0 <= task_id < task_count]
    if explicit:
        return tuple(dict.fromkeys(explicit))
    if suite_family == "libero_spatial_expanded" and task_count > 2:
        return tuple(range(2, task_count))
    return tuple(range(task_count))


PER_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_spatial_expanded": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
    "other_locally_supported_LIBERO_suites": 520,
}
