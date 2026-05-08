# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportMissingParameterType=false, reportUnknownLambdaType=false, reportUnannotatedClassAttribute=false

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import errno
import functools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from work.openpi.recap.real_variant_policy_contract import (
    NORM_STATS_RELATIVE_PATH,
    attach_real_variant_policy_contract,
    build_real_variant_policy_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OPENPI_ROOT = REPO_ROOT / "submodules" / "openpi"
OPENPI_ROOT = Path(
    os.environ.get("OPENPI_ROOT_OVERRIDE", str(DEFAULT_OPENPI_ROOT))
).resolve()
OPENPI_VENV_PYTHON = Path(
    os.environ.get(
        "OPENPI_VENV_PYTHON_OVERRIDE",
        str(OPENPI_ROOT / ".venv" / "bin" / "python"),
    )
)
OPENPI_SRC = Path(
    os.environ.get("OPENPI_SRC_OVERRIDE", str(OPENPI_ROOT / "src"))
).resolve()
OPENPI_CLIENT_SRC = Path(
    os.environ.get(
        "OPENPI_CLIENT_SRC_OVERRIDE",
        str(OPENPI_ROOT / "packages" / "openpi-client" / "src"),
    )
).resolve()
LIBERO_ASSET_SUBDIR = Path("assets") / "physical-intelligence" / "libero"
DEFAULT_TRAIN_CONFIG_NAME = "pi05_libero"
DEFAULT_WEIGHT_LOADER_PARAMS = "gs://openpi-assets/checkpoints/pi05_libero/params"
DEFAULT_ASSET_ID = "physical-intelligence/libero"
DEFAULT_LOCAL_SAFE_NUM_TRAIN_STEPS = 1
DEFAULT_LOCAL_SAFE_BATCH_SIZE = 1
DEFAULT_LOCAL_SAFE_SAVE_INTERVAL = 1
DEFAULT_LOCAL_SAFE_NUM_WORKERS = 0
DEFAULT_LOCAL_SAFE_FSDP_DEVICES = 1
DEFAULT_LOCAL_SAFE_JAX_PLATFORM = "cpu"
DEFAULT_LOCAL_SAFE_CUDA_VISIBLE_DEVICES = ""
DEFAULT_LOCAL_SAFE_XLA_PREALLOCATE = "false"
DEFAULT_LOCAL_SAFE_XLA_ALLOCATOR = "platform"
TRAIN_NUM_STEPS_ENV = "OPENPI_VARIANT_TRAIN_NUM_STEPS"
TRAIN_NUM_STEPS_SOURCE_ENV = "OPENPI_VARIANT_TRAIN_NUM_STEPS_SOURCE"
TRAIN_DEFAULT_NUM_STEPS_ENV = "OPENPI_VARIANT_TRAIN_DEFAULT_NUM_STEPS"
TRAIN_SAVE_INTERVAL_ENV = "OPENPI_VARIANT_TRAIN_SAVE_INTERVAL"
TRAIN_SAVE_INTERVAL_SOURCE_ENV = "OPENPI_VARIANT_TRAIN_SAVE_INTERVAL_SOURCE"
TRAIN_DEFAULT_SAVE_INTERVAL_ENV = "OPENPI_VARIANT_TRAIN_DEFAULT_SAVE_INTERVAL"
SUBPROCESS_CACHE_ROOT_DIRNAME = "subprocess_cache"
SUBPROCESS_CACHE_ROOT_ENV = "OPENPI_VARIANT_SUBPROCESS_CACHE_ROOT"
LOSS_DECOMPOSITION_REAL_SCHEMA_VERSION = "v22_variant_loss_decomposition_real_v1"
THRESHOLD_TRACE_REAL_SCHEMA_VERSION = "v22_variant_threshold_switch_trace_real_v1"
ALPHA_TRACE_REAL_SCHEMA_VERSION = "v22_variant_alpha_dual_loss_trace_real_v1"


def _prioritize_upstream_openpi_imports() -> None:
    """Ensure upstream Physical-Intelligence ``openpi`` wins over ``work/openpi``.

    This module is part of the repo-local ``work.openpi`` package. Importing it can
    place ``<repo>/work`` ahead of the upstream OpenPI source tree on ``sys.path``;
    then OpenPI's own ``scripts/train.py`` resolves ``import openpi`` to
    ``work/openpi`` instead of ``submodules/openpi/src/openpi``. The failure is
    silent until the train script imports e.g. ``openpi.models``. Before loading
    upstream train.py, pin the upstream src/client paths to the front and evict only
    the shadowed top-level ``openpi`` modules from ``sys.modules``.
    """

    desired_paths = [str(OPENPI_SRC), str(OPENPI_CLIENT_SRC)]
    for path in reversed(desired_paths):
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    work_openpi_root = (REPO_ROOT / "work" / "openpi").resolve()
    for module_name, module in list(sys.modules.items()):
        if module_name != "openpi" and not module_name.startswith("openpi."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            Path(module_file).resolve().relative_to(work_openpi_root)
        except ValueError:
            continue
        del sys.modules[module_name]


@dataclass(frozen=True)
class RealVariantExportRequest:
    variant: str
    variant_name: str
    dataset_dir: Path
    runtime_dir: Path
    consumer_mode: str
    fixed_indicator_mode: str | None
    resume: bool = False
    default_num_train_steps: int = DEFAULT_LOCAL_SAFE_NUM_TRAIN_STEPS
    default_save_interval: int = DEFAULT_LOCAL_SAFE_SAVE_INTERVAL
    probe_metrics_path: Path | None = None
    train_config_name: str = DEFAULT_TRAIN_CONFIG_NAME
    weight_loader_params: str = DEFAULT_WEIGHT_LOADER_PARAMS
    log_interval: int | None = None
    v22_trace_dir: Path | None = None
    v22_trace_run_id: str = ""
    v22_trace_variant: str = ""
    v22_emit_loss_decomposition: bool = False
    v22_emit_threshold_trace: bool = False
    v22_emit_alpha_dual_trace: bool = False
    v22_enable_r2_phase_threshold_switching: bool = False
    v22_enable_r4_alpha_dual_loss: bool = False
    v22_phase_threshold_step: int = 0
    v22_alpha_pre_phase: float = 0.0
    v22_alpha_post_phase: float = 1.0


@dataclass(frozen=True)
class RealVariantExportBundle:
    export_dir: Path
    runtime_log_path: Path


@dataclass(frozen=True)
class ResumeCheckpointPlan:
    mode: str
    latest_step: int | None = None
    resume_state_step: int | None = None
    latest_checkpoint_dir: Path | None = None
    seed_params_path: Path | None = None


class RealVariantExportBlockedError(RuntimeError):
    payload: dict[str, object]

    def __init__(self, message: str, *, payload: dict[str, object]) -> None:
        super().__init__(message)
        self.payload = payload


@dataclass(frozen=True)
class StreamedSubprocessResult:
    returncode: int
    output_tail: str


def _run_streamed_subprocess(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    runtime_log_path: Path,
) -> StreamedSubprocessResult:
    """Run a long-lived child while preserving live console output and log evidence."""
    tail: deque[str] = deque(maxlen=400)
    with runtime_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("# command\n")
        log_file.write(" ".join(command))
        log_file.write("\n\n# combined stdout/stderr\n")
        log_file.flush()
        process = subprocess.Popen(  # noqa: S603
            list(command),
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(env),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
            tail.append(line)
        returncode = process.wait()
    return StreamedSubprocessResult(
        returncode=int(returncode),
        output_tail="".join(tail)[-4000:],
    )


@dataclass(frozen=True)
class V22RealTraceConfig:
    trace_dir: Path
    run_id: str
    variant: str
    emit_loss_decomposition: bool
    emit_threshold_trace: bool
    emit_alpha_dual_trace: bool
    enable_r2_phase_threshold_switching: bool
    enable_r4_alpha_dual_loss: bool
    phase_threshold_step: int
    alpha_pre_phase: float
    alpha_post_phase: float


@dataclass(frozen=True)
class VariantPromptTransform:
    consumer_mode: str
    fixed_indicator_mode: str | None

    @staticmethod
    def _unwrap_scalar(value: Any) -> Any:
        if isinstance(value, (str, bytes)):
            return value
        item = getattr(value, "item", None)
        if not callable(item):
            return value
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            return value

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        from work.openpi.recap.runtime_prompt import build_training_prompt_bundle

        prompt_bundle = build_training_prompt_bundle(
            {
                "prompt_raw": self._unwrap_scalar(data["prompt_raw"]),
                "recap_m2.indicator_I": self._unwrap_scalar(
                    data["recap_m2.indicator_I"]
                ),
                "recap_m2.t": self._unwrap_scalar(data.get("recap_m2.t")),
                "episode_index": self._unwrap_scalar(data.get("episode_index")),
                "action": self._unwrap_scalar(data.get("action")),
                "observation.state": self._unwrap_scalar(data.get("observation/state")),
                "prompt_conditioned": self._unwrap_scalar(
                    data.get("prompt_conditioned")
                ),
            },
            consumer_mode=self.consumer_mode,
            fixed_indicator_mode=self.fixed_indicator_mode,
        )
        updated = dict(data)
        updated["prompt"] = np.asarray(prompt_bundle.prompt_text)
        return updated


def _build_variant_data_transforms(
    model: Any,
    *,
    consumer_mode: str,
    fixed_indicator_mode: str | None,
):
    import openpi.transforms as transforms
    from openpi.policies import libero_policy

    return transforms.Group(
        inputs=[
            VariantPromptTransform(
                consumer_mode=consumer_mode,
                fixed_indicator_mode=fixed_indicator_mode,
            ),
            libero_policy.LiberoInputs(model_type=model.model_type),
        ],
        outputs=[libero_policy.LiberoOutputs()],
    )


def _resolve_train_resource_settings(base_config: Any) -> dict[str, int]:
    return {
        "num_train_steps": int(
            os.environ.get(
                TRAIN_NUM_STEPS_ENV,
                str(DEFAULT_LOCAL_SAFE_NUM_TRAIN_STEPS),
            )
        ),
        "batch_size": int(
            os.environ.get(
                "OPENPI_VARIANT_TRAIN_BATCH_SIZE",
                str(DEFAULT_LOCAL_SAFE_BATCH_SIZE),
            )
        ),
        "save_interval": int(
            os.environ.get(
                TRAIN_SAVE_INTERVAL_ENV,
                str(DEFAULT_LOCAL_SAFE_SAVE_INTERVAL),
            )
        ),
        "num_workers": int(
            os.environ.get(
                "OPENPI_VARIANT_TRAIN_NUM_WORKERS",
                str(DEFAULT_LOCAL_SAFE_NUM_WORKERS),
            )
        ),
        "fsdp_devices": int(
            os.environ.get(
                "OPENPI_VARIANT_TRAIN_FSDP_DEVICES",
                str(
                    min(
                        max(int(getattr(base_config, "fsdp_devices", 1)), 1),
                        DEFAULT_LOCAL_SAFE_FSDP_DEVICES,
                    )
                ),
            )
        ),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(_json_ready(dict(row)), ensure_ascii=False) + "\n" for row in rows
    )
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _append_jsonl_row(path: Path, row: Mapping[str, object]) -> None:
    existing: list[Mapping[str, object]] = []
    if path.is_file():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    existing.append(dict(row))
    _write_jsonl_rows(path, existing)


def _coerce_probe_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if isinstance(scalar, bool):
            return float(int(scalar))
        if isinstance(scalar, (int, float)):
            return float(scalar)
    raise TypeError(f"unable to coerce probe metric {value!r} to float")


def _zero_like_actions(train_main_mod: Any, actions: Any) -> Any:
    return train_main_mod.jnp.mean(train_main_mod.jnp.asarray(actions) * 0.0)


def _component_loss_flow(
    train_main_mod: Any,
    model: Any,
    rng: Any,
    observation: Any,
    actions: Any,
) -> Any:
    chunked_loss = model.compute_loss(rng, observation, actions, train=True)
    return train_main_mod.jnp.mean(chunked_loss)


def _component_token_ce(
    train_main_mod: Any,
    model: Any,
    rng: Any,
    observation: Any,
    actions: Any,
    *,
    target_kind: str,
) -> Any:
    del actions
    if not hasattr(model, "embed_inputs") or not hasattr(model, "PaliGemma"):
        return _zero_like_actions(train_main_mod, actions=0.0)
    tokenized_prompt = getattr(observation, "tokenized_prompt", None)
    token_loss_mask = getattr(observation, "token_loss_mask", None)
    token_mask = getattr(observation, "token_mask", None)
    if tokenized_prompt is None or token_loss_mask is None:
        return _zero_like_actions(train_main_mod, actions=0.0)

    image_keys = list(getattr(observation, "images", {}).keys())
    observation = train_main_mod._model.preprocess_observation(
        rng,
        observation,
        train=True,
        image_keys=image_keys,
    )
    input_token_embeddings, input_mask, ar_mask = model.embed_inputs(observation)
    make_attn_mask = getattr(train_main_mod, "make_attn_mask", None)
    if make_attn_mask is None:
        from openpi.models.pi0_fast import make_attn_mask as make_attn_mask

    attn_mask = make_attn_mask(input_mask, ar_mask)
    targets = train_main_mod.jax.nn.one_hot(
        observation.tokenized_prompt[:, 1:],
        model.PaliGemma.llm.module.vocab_size,
    )
    pre_logits, _, _ = model.PaliGemma.llm(
        embedded_prefix=input_token_embeddings[:, :-1],
        mask=attn_mask[:, :-1, :-1],
        return_prelogits=True,
    )
    logits, _ = model.PaliGemma.llm(pre_logits=pre_logits[:, -targets.shape[1] :])
    logp = train_main_mod.jax.nn.log_softmax(logits, axis=-1)
    token_pplx = train_main_mod.jnp.sum(targets * logp, axis=-1)
    loss_mask = observation.token_loss_mask[:, 1:].astype(train_main_mod.jnp.bool_)
    if target_kind == "text":
        if token_mask is None:
            mask = train_main_mod.jnp.logical_not(loss_mask)
        else:
            mask = train_main_mod.jnp.logical_and(
                observation.token_mask[:, 1:].astype(train_main_mod.jnp.bool_),
                train_main_mod.jnp.logical_not(loss_mask),
            )
    else:
        mask = loss_mask
    denominator = train_main_mod.jnp.clip(train_main_mod.jnp.sum(mask, axis=-1), 1)
    per_item = -train_main_mod.jnp.sum(token_pplx * mask, axis=-1) / denominator
    return train_main_mod.jnp.mean(per_item)


def _component_loss_discrete_action_ce(
    train_main_mod: Any,
    model: Any,
    rng: Any,
    observation: Any,
    actions: Any,
) -> Any:
    return _component_token_ce(
        train_main_mod,
        model,
        rng,
        observation,
        actions,
        target_kind="discrete_action",
    )


def _component_loss_text_ce(
    train_main_mod: Any,
    model: Any,
    rng: Any,
    observation: Any,
    actions: Any,
) -> Any:
    return _component_token_ce(
        train_main_mod,
        model,
        rng,
        observation,
        actions,
        target_kind="text",
    )


def _step_alpha(train_main_mod: Any, config: V22RealTraceConfig, step_counter: Any) -> Any:
    if not (
        config.enable_r2_phase_threshold_switching
        or config.enable_r4_alpha_dual_loss
    ):
        return train_main_mod.jnp.asarray(0.0, dtype=train_main_mod.jnp.float32)
    return train_main_mod.jnp.where(
        train_main_mod.jnp.asarray(step_counter) >= int(config.phase_threshold_step),
        train_main_mod.jnp.asarray(config.alpha_post_phase, dtype=train_main_mod.jnp.float32),
        train_main_mod.jnp.asarray(config.alpha_pre_phase, dtype=train_main_mod.jnp.float32),
    )


def _coerce_trace_row_float(data: Mapping[str, Any], key: str) -> float:
    return _coerce_probe_float(data.get(key, 0.0))


def _trace_phase(config: V22RealTraceConfig, step: int) -> str:
    return "post_threshold" if step >= int(config.phase_threshold_step) else "pre_threshold"


def _append_v22_real_trace_rows(config: V22RealTraceConfig, data: Mapping[str, Any], step: int) -> None:
    if "loss" not in data:
        return
    trace_dir = config.trace_dir
    if config.emit_loss_decomposition:
        _append_jsonl_row(
            trace_dir / "loss_decomposition.jsonl",
            {
                "schema_version": LOSS_DECOMPOSITION_REAL_SCHEMA_VERSION,
                "run_id": config.run_id,
                "variant": config.variant,
                "step": int(step),
                "loss": _coerce_trace_row_float(data, "loss"),
                "grad_norm": _coerce_trace_row_float(data, "grad_norm"),
                "param_norm": _coerce_trace_row_float(data, "param_norm"),
                "flow_loss": _coerce_trace_row_float(data, "flow_loss"),
                "discrete_action_ce": _coerce_trace_row_float(data, "discrete_action_ce"),
                "text_ce": _coerce_trace_row_float(data, "text_ce"),
            },
        )
    if config.emit_threshold_trace:
        _append_jsonl_row(
            trace_dir / "threshold_switch_trace.jsonl",
            {
                "schema_version": THRESHOLD_TRACE_REAL_SCHEMA_VERSION,
                "run_id": config.run_id,
                "variant": config.variant,
                "step": int(step),
                "phase": _trace_phase(config, int(step)),
                "threshold_value": int(config.phase_threshold_step),
                "switch_event": int(step) == int(config.phase_threshold_step),
            },
        )
    if config.emit_alpha_dual_trace:
        alpha = (
            float(config.alpha_post_phase)
            if step >= int(config.phase_threshold_step)
            else float(config.alpha_pre_phase)
        )
        components = {
            "flow_loss": _coerce_trace_row_float(data, "flow_loss"),
            "discrete_action_ce": _coerce_trace_row_float(data, "discrete_action_ce"),
            "text_ce": _coerce_trace_row_float(data, "text_ce"),
        }
        _append_jsonl_row(
            trace_dir / "alpha_dual_loss_trace.jsonl",
            {
                "schema_version": ALPHA_TRACE_REAL_SCHEMA_VERSION,
                "run_id": config.run_id,
                "variant": config.variant,
                "step": int(step),
                "alpha": alpha,
                "dual_loss_components": components,
                "total_alpha_dual_loss": _coerce_trace_row_float(data, "loss"),
            },
        )


def _install_v22_real_training_hooks(
    train_main_mod: Any,
    *,
    trace_config: V22RealTraceConfig,
):
    original_train_step = train_main_mod.train_step
    missing_init_wandb = object()
    original_init_wandb = getattr(train_main_mod, "init_wandb", missing_init_wandb)
    missing_loss_fn = object()
    original_loss_fn = getattr(train_main_mod, "loss_fn", missing_loss_fn)
    wandb_log_delegate = {"fn": train_main_mod.wandb.log}

    def _wrapped_loss_fn(
        model: Any,
        rng: Any,
        observation: Any,
        actions: Any,
        *,
        step_counter: Any = 0,
    ) -> tuple[Any, dict[str, Any]]:
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        total_loss = train_main_mod.jnp.mean(chunked_loss)
        aux = {
            "flow_loss": _component_loss_flow(
                train_main_mod, model, rng, observation, actions
            ),
            "discrete_action_ce": _component_loss_discrete_action_ce(
                train_main_mod, model, rng, observation, actions
            ),
            "text_ce": _component_loss_text_ce(
                train_main_mod, model, rng, observation, actions
            ),
        }
        alpha = _step_alpha(train_main_mod, trace_config, step_counter)
        if trace_config.enable_r4_alpha_dual_loss:
            conditioned_total = (
                aux["flow_loss"] + aux["discrete_action_ce"] + aux["text_ce"]
            )
            total_loss = total_loss + alpha * conditioned_total
        else:
            total_loss = total_loss + alpha * train_main_mod.jnp.asarray(0.0)
        return total_loss, aux

    @train_main_mod.at.typecheck
    def _wrapped_train_step(
        config: Any,
        rng: Any,
        state: Any,
        batch: tuple[Any, Any],
    ) -> tuple[Any, dict[str, Any]]:
        model = train_main_mod.nnx.merge(state.model_def, state.params)
        model.train()
        train_rng = train_main_mod.jax.random.fold_in(rng, state.step)
        observation, actions = batch

        @train_main_mod.at.typecheck
        def _loss_for_step(
            model: Any,
            rng: Any,
            observation: Any,
            actions: Any,
        ) -> tuple[Any, dict[str, Any]]:
            return train_main_mod.loss_fn(
                model,
                rng,
                observation,
                actions,
                step_counter=state.step,
            )

        diff_state = train_main_mod.nnx.DiffState(0, config.trainable_filter)
        (loss, aux), grads = train_main_mod.nnx.value_and_grad(
            _loss_for_step,
            has_aux=True,
            argnums=diff_state,
        )(model, train_rng, observation, actions)

        params = state.params.filter(config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = train_main_mod.optax.apply_updates(params, updates)
        train_main_mod.nnx.update(model, new_params)
        new_params = train_main_mod.nnx.state(model)
        new_state = train_main_mod.dataclasses.replace(
            state,
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )
        if state.ema_decay is not None:
            new_state = train_main_mod.dataclasses.replace(
                new_state,
                ema_params=train_main_mod.jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params,
                    new_params,
                ),
            )

        kernel_params = train_main_mod.nnx.state(
            model,
            train_main_mod.nnx.All(
                train_main_mod.nnx.Param,
                train_main_mod.nnx.Not(
                    train_main_mod.nnx_utils.PathRegex(
                        ".*/(bias|scale|pos_embedding|input_embedding)"
                    )
                ),
                lambda _, x: x.value.ndim > 1,
            ),
        )
        info = {
            "loss": loss,
            "grad_norm": train_main_mod.optax.global_norm(grads),
            "param_norm": train_main_mod.optax.global_norm(kernel_params),
            "flow_loss": aux["flow_loss"],
            "discrete_action_ce": aux["discrete_action_ce"],
            "text_ce": aux["text_ce"],
        }
        return new_state, info

    def _v22_real_train_jsonl_appender(
        data: Any,
        *,
        step: Any = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(data, Mapping) and step is not None:
            _append_v22_real_trace_rows(trace_config, data, int(step))
        return wandb_log_delegate["fn"](data, step=step, **kwargs)

    def _wrapped_init_wandb(*args: Any, **kwargs: Any) -> Any:
        assert original_init_wandb is not missing_init_wandb
        assert callable(original_init_wandb)
        result = original_init_wandb(*args, **kwargs)
        # wandb.init(mode="disabled") replaces wandb.log; reinstall the appender
        # after init so per-step traces keep flowing in real runs.
        wandb_log_delegate["fn"] = train_main_mod.wandb.log
        train_main_mod.wandb.log = _v22_real_train_jsonl_appender
        return result

    train_main_mod.train_step = _wrapped_train_step
    train_main_mod.loss_fn = _wrapped_loss_fn
    train_main_mod.wandb.log = _v22_real_train_jsonl_appender
    if original_init_wandb is not missing_init_wandb:
        train_main_mod.init_wandb = _wrapped_init_wandb

    def _validate_v22_hooks() -> None:
        assert id(train_main_mod.train_step) == id(_wrapped_train_step), (
            "patch lost: train_step"
        )
        assert id(train_main_mod.loss_fn) == id(_wrapped_loss_fn), "patch lost: loss_fn"
        assert id(train_main_mod.wandb.log) == id(_v22_real_train_jsonl_appender), (
            "patch lost: wandb.log"
        )
        if original_init_wandb is not missing_init_wandb:
            assert id(train_main_mod.init_wandb) == id(_wrapped_init_wandb), (
                "patch lost: init_wandb"
            )

    def _finalize_v22_hooks() -> None:
        train_main_mod.train_step = original_train_step
        if original_init_wandb is not missing_init_wandb:
            train_main_mod.init_wandb = original_init_wandb
        if original_loss_fn is missing_loss_fn:
            delattr(train_main_mod, "loss_fn")
        else:
            train_main_mod.loss_fn = original_loss_fn
        train_main_mod.wandb.log = wandb_log_delegate["fn"]

    return _validate_v22_hooks, _finalize_v22_hooks


def _build_probe_metrics_payload(
    probe_records: list[dict[str, object]],
) -> dict[str, object]:
    loss_values: list[float] = []
    grad_norm_values: list[float] = []
    param_delta_norm_values: list[float] = []
    for record in probe_records:
        raw_loss = record.get("loss")
        if isinstance(raw_loss, float):
            loss_values.append(raw_loss)
        raw_grad_norm = record.get("grad_norm")
        if isinstance(raw_grad_norm, float):
            grad_norm_values.append(raw_grad_norm)
        raw_param_delta_norm = record.get("param_delta_norm")
        if isinstance(raw_param_delta_norm, float):
            param_delta_norm_values.append(raw_param_delta_norm)
    any_grad_nonzero = any(value > 0.0 for value in grad_norm_values)
    any_param_delta_nonzero = any(value > 0.0 for value in param_delta_norm_values)
    return {
        "schema_version": "openpi_real_variant_probe_metrics_v1",
        "probe_records": probe_records,
        "loss_values": loss_values,
        "grad_norm_values": grad_norm_values,
        "param_delta_norm_values": param_delta_norm_values,
        "any_grad_nonzero": any_grad_nonzero,
        "any_param_delta_nonzero": any_param_delta_nonzero,
        "probe_pass": bool(
            loss_values and any_grad_nonzero and any_param_delta_nonzero
        ),
    }


def _install_one_step_probe_hooks(
    train_main_mod: Any,
    *,
    probe_metrics_path: Path,
):
    original_train_step = train_main_mod.train_step
    original_init_wandb = train_main_mod.init_wandb
    wandb_log_delegate = {"fn": train_main_mod.wandb.log}
    probe_records: list[dict[str, object]] = []

    @train_main_mod.at.typecheck
    def _wrapped_train_step(
        config: Any,
        rng: Any,
        state: Any,
        batch: tuple[Any, Any],
    ) -> tuple[Any, dict[str, Any]]:
        model = train_main_mod.nnx.merge(state.model_def, state.params)
        model.train()

        @train_main_mod.at.typecheck
        def _loss_fn(model: Any, rng: Any, observation: Any, actions: Any):
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            return train_main_mod.jnp.mean(chunked_loss)

        train_rng = train_main_mod.jax.random.fold_in(rng, state.step)
        observation, actions = batch
        diff_state = train_main_mod.nnx.DiffState(0, config.trainable_filter)
        loss, grads = train_main_mod.nnx.value_and_grad(_loss_fn, argnums=diff_state)(
            model,
            train_rng,
            observation,
            actions,
        )

        params = state.params.filter(config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = train_main_mod.optax.apply_updates(params, updates)

        train_main_mod.nnx.update(model, new_params)
        new_params = train_main_mod.nnx.state(model)

        new_state = train_main_mod.dataclasses.replace(
            state,
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )
        if state.ema_decay is not None:
            new_state = train_main_mod.dataclasses.replace(
                new_state,
                ema_params=train_main_mod.jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params,
                    new_params,
                ),
            )

        kernel_params = train_main_mod.nnx.state(
            model,
            train_main_mod.nnx.All(
                train_main_mod.nnx.Param,
                train_main_mod.nnx.Not(
                    train_main_mod.nnx_utils.PathRegex(
                        ".*/(bias|scale|pos_embedding|input_embedding)"
                    )
                ),
                lambda _, x: x.value.ndim > 1,
            ),
        )
        grad_norm = train_main_mod.optax.global_norm(grads)
        param_delta_norm = train_main_mod.optax.global_norm(updates)
        info = {
            "loss": loss,
            "grad_norm": grad_norm,
            "param_norm": train_main_mod.optax.global_norm(kernel_params),
            "param_delta_norm": param_delta_norm,
        }
        return new_state, info

    def _wrapped_wandb_log(data: Any, *, step: Any = None, **kwargs: Any) -> Any:
        if isinstance(data, dict):
            probe_record: dict[str, object] = {}
            if step is not None:
                probe_record["step"] = int(step)
            for key in ("loss", "grad_norm", "param_norm", "param_delta_norm"):
                if key in data:
                    probe_record[key] = _coerce_probe_float(data[key])
            if probe_record:
                raw_grad_norm = probe_record.get("grad_norm")
                grad_norm = raw_grad_norm if isinstance(raw_grad_norm, float) else 0.0
                raw_param_delta_norm = probe_record.get("param_delta_norm")
                param_delta_norm = (
                    raw_param_delta_norm
                    if isinstance(raw_param_delta_norm, float)
                    else 0.0
                )
                probe_record["any_grad_nonzero"] = grad_norm > 0.0
                probe_record["any_param_delta_nonzero"] = param_delta_norm > 0.0
                probe_records.append(probe_record)
        return wandb_log_delegate["fn"](data, step=step, **kwargs)

    def _wrapped_init_wandb(*args: Any, **kwargs: Any) -> Any:
        result = original_init_wandb(*args, **kwargs)
        wandb_log_delegate["fn"] = train_main_mod.wandb.log
        train_main_mod.wandb.log = _wrapped_wandb_log
        return result

    train_main_mod.train_step = _wrapped_train_step
    train_main_mod.wandb.log = _wrapped_wandb_log
    train_main_mod.init_wandb = _wrapped_init_wandb

    def _finalize_probe_hooks() -> None:
        train_main_mod.train_step = original_train_step
        train_main_mod.init_wandb = original_init_wandb
        train_main_mod.wandb.log = wandb_log_delegate["fn"]
        _write_json(probe_metrics_path, _build_probe_metrics_payload(probe_records))

    return _finalize_probe_hooks


def _build_blocker_payload(
    *,
    request: RealVariantExportRequest,
    blocker_code: str,
    reason: str,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "blocked",
        "blocker_code": blocker_code,
        "variant": request.variant,
        "variant_name": request.variant_name,
        "dataset_dir": str(request.dataset_dir),
        "runtime_dir": str(request.runtime_dir),
        "reason": reason,
    }
    if details:
        payload["details"] = details
    return payload


def _ensure_real_export_layout(export_dir: Path) -> None:
    required_paths = (
        export_dir / "params" / "_METADATA",
        export_dir / LIBERO_ASSET_SUBDIR / "norm_stats.json",
    )
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "real variant export is incomplete: " + ", ".join(missing_paths)
        )


def _resolve_subprocess_cache_dirs(
    request: RealVariantExportRequest,
) -> dict[str, Path]:
    cache_root_override = os.environ.get(SUBPROCESS_CACHE_ROOT_ENV, "").strip()
    cache_root = (
        Path(cache_root_override).resolve()
        if cache_root_override
        else request.runtime_dir.resolve() / SUBPROCESS_CACHE_ROOT_DIRNAME
    )
    return {
        "HF_HOME": cache_root / "hf_home",
        "HF_DATASETS_CACHE": cache_root / "hf_home" / "datasets",
        "TRANSFORMERS_CACHE": cache_root / "hf_home" / "transformers",
        "TMPDIR": cache_root / "tmp",
    }


def _build_subprocess_env(request: RealVariantExportRequest) -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [str(REPO_ROOT), str(OPENPI_SRC)]
    if OPENPI_CLIENT_SRC.exists():
        pythonpath_entries.append(str(OPENPI_CLIENT_SRC))
    if current_pythonpath:
        pythonpath_entries.append(current_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["HF_LEROBOT_HOME"] = str(request.dataset_dir.parent)
    cache_dirs = _resolve_subprocess_cache_dirs(request)
    for env_name, directory in cache_dirs.items():
        directory.mkdir(parents=True, exist_ok=True)
        env[env_name] = str(directory)
    env["WANDB_MODE"] = "disabled"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    platform_override_present = any(
        name in env
        for name in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME", "CUDA_VISIBLE_DEVICES")
    )
    if not platform_override_present:
        env["JAX_PLATFORMS"] = DEFAULT_LOCAL_SAFE_JAX_PLATFORM
        env["JAX_PLATFORM_NAME"] = DEFAULT_LOCAL_SAFE_JAX_PLATFORM
        env["CUDA_VISIBLE_DEVICES"] = DEFAULT_LOCAL_SAFE_CUDA_VISIBLE_DEVICES
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", DEFAULT_LOCAL_SAFE_XLA_PREALLOCATE)
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", DEFAULT_LOCAL_SAFE_XLA_ALLOCATOR)
    if TRAIN_NUM_STEPS_ENV in env:
        env.setdefault(TRAIN_NUM_STEPS_SOURCE_ENV, "env_override")
    else:
        env[TRAIN_NUM_STEPS_ENV] = str(int(request.default_num_train_steps))
        env[TRAIN_NUM_STEPS_SOURCE_ENV] = "stage_default"
    env[TRAIN_DEFAULT_NUM_STEPS_ENV] = str(int(request.default_num_train_steps))
    if TRAIN_SAVE_INTERVAL_ENV in env:
        env.setdefault(TRAIN_SAVE_INTERVAL_SOURCE_ENV, "env_override")
    else:
        env[TRAIN_SAVE_INTERVAL_ENV] = str(int(request.default_save_interval))
        env[TRAIN_SAVE_INTERVAL_SOURCE_ENV] = "stage_default"
    env[TRAIN_DEFAULT_SAVE_INTERVAL_ENV] = str(int(request.default_save_interval))
    return env


def run_real_variant_training_export(
    request: RealVariantExportRequest,
) -> RealVariantExportBundle:
    runtime_dir = request.runtime_dir.resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_path = runtime_dir / "real_variant_training.log"
    export_dir = runtime_dir / "real_variant_export"

    if not OPENPI_VENV_PYTHON.is_file():
        raise RealVariantExportBlockedError(
            "missing openpi venv python for real variant training export",
            payload=_build_blocker_payload(
                request=request,
                blocker_code="missing_openpi_venv_python",
                reason=f"required interpreter missing: {OPENPI_VENV_PYTHON}",
            ),
        )

    command = [
        str(OPENPI_VENV_PYTHON),
        str(Path(__file__).resolve()),
        "--internal-run-export",
        "--variant",
        request.variant,
        "--variant-name",
        request.variant_name,
        "--dataset-dir",
        str(request.dataset_dir),
        "--runtime-dir",
        str(runtime_dir),
        "--export-dir",
        str(export_dir),
        "--consumer-mode",
        request.consumer_mode,
        "--train-config-name",
        request.train_config_name,
        "--weight-loader-params",
        request.weight_loader_params,
    ]
    if request.fixed_indicator_mode is not None:
        command.extend(["--fixed-indicator-mode", request.fixed_indicator_mode])
    if request.probe_metrics_path is not None:
        command.extend(["--probe-metrics-path", str(request.probe_metrics_path)])
    if request.log_interval is not None:
        command.extend(["--log-interval", str(int(request.log_interval))])
    if request.v22_trace_dir is not None:
        command.extend(
            [
                "--v22-trace-dir",
                str(request.v22_trace_dir),
                "--v22-trace-run-id",
                request.v22_trace_run_id,
                "--v22-trace-variant",
                request.v22_trace_variant or request.variant,
                "--v22-phase-threshold-step",
                str(int(request.v22_phase_threshold_step)),
                "--v22-alpha-pre-phase",
                str(float(request.v22_alpha_pre_phase)),
                "--v22-alpha-post-phase",
                str(float(request.v22_alpha_post_phase)),
            ]
        )
        if request.v22_emit_loss_decomposition:
            command.append("--v22-emit-loss-decomposition")
        if request.v22_emit_threshold_trace:
            command.append("--v22-emit-threshold-trace")
        if request.v22_emit_alpha_dual_trace:
            command.append("--v22-emit-alpha-dual-trace")
        if request.v22_enable_r2_phase_threshold_switching:
            command.append("--v22-enable-r2-phase-threshold-switching")
        if request.v22_enable_r4_alpha_dual_loss:
            command.append("--v22-enable-r4-alpha-dual-loss")
    if request.resume:
        command.append("--resume")

    result = _run_streamed_subprocess(
        command,
        env=_build_subprocess_env(request),
        runtime_log_path=runtime_log_path,
    )
    if result.returncode != 0:
        raise RealVariantExportBlockedError(
            "real variant training export failed",
            payload=_build_blocker_payload(
                request=request,
                blocker_code="real_variant_training_failed",
                reason="upstream training/export subprocess returned non-zero exit status",
                details={
                    "returncode": int(result.returncode),
                    "runtime_log": str(runtime_log_path),
                    "stdout_tail": result.output_tail,
                    "stderr_tail": result.output_tail,
                },
            ),
        )
    try:
        _ensure_real_export_layout(export_dir)
    except FileNotFoundError as exc:
        raise RealVariantExportBlockedError(
            str(exc),
            payload=_build_blocker_payload(
                request=request,
                blocker_code="real_variant_export_incomplete",
                reason=str(exc),
                details={"runtime_log": str(runtime_log_path)},
            ),
        ) from exc
    return RealVariantExportBundle(
        export_dir=export_dir,
        runtime_log_path=runtime_log_path,
    )


def _build_internal_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    _ = parser.add_argument("--internal-run-export", action="store_true")
    _ = parser.add_argument("--variant", required=True)
    _ = parser.add_argument("--variant-name", required=True)
    _ = parser.add_argument("--dataset-dir", required=True)
    _ = parser.add_argument("--runtime-dir", required=True)
    _ = parser.add_argument("--export-dir", required=True)
    _ = parser.add_argument("--consumer-mode", required=True)
    _ = parser.add_argument("--fixed-indicator-mode", default="")
    _ = parser.add_argument("--probe-metrics-path", default="")
    _ = parser.add_argument("--train-config-name", default=DEFAULT_TRAIN_CONFIG_NAME)
    _ = parser.add_argument("--weight-loader-params", default=DEFAULT_WEIGHT_LOADER_PARAMS)
    _ = parser.add_argument("--log-interval", type=int, default=None)
    _ = parser.add_argument("--resume", action="store_true")
    _ = parser.add_argument("--v22-trace-dir", default="")
    _ = parser.add_argument("--v22-trace-run-id", default="")
    _ = parser.add_argument("--v22-trace-variant", default="")
    _ = parser.add_argument("--v22-emit-loss-decomposition", action="store_true")
    _ = parser.add_argument("--v22-emit-threshold-trace", action="store_true")
    _ = parser.add_argument("--v22-emit-alpha-dual-trace", action="store_true")
    _ = parser.add_argument("--v22-enable-r2-phase-threshold-switching", action="store_true")
    _ = parser.add_argument("--v22-enable-r4-alpha-dual-loss", action="store_true")
    _ = parser.add_argument("--v22-phase-threshold-step", type=int, default=0)
    _ = parser.add_argument("--v22-alpha-pre-phase", type=float, default=0.0)
    _ = parser.add_argument("--v22-alpha-post-phase", type=float, default=1.0)
    return parser


def _build_variant_train_config(args: argparse.Namespace):
    import dataclasses
    import numpy as np
    import pandas as pd

    from datasets import Image, load_dataset

    from lerobot.common.datasets import lerobot_dataset as lerobot_dataset_mod
    import openpi.shared.normalize as normalize
    import openpi.training.config as train_config_mod
    import openpi.training.weight_loaders as weight_loaders
    import openpi.transforms as transforms

    dataset_dir = Path(args.dataset_dir).resolve()
    export_dir = Path(args.export_dir).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    fixed_indicator_mode = args.fixed_indicator_mode.strip() or None
    consumer_mode = str(args.consumer_mode)

    def _patched_load_hf_dataset(self):  # type: ignore[no-untyped-def]
        if self.episodes is None:
            data_dir = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=data_dir, split="train")
        else:
            files = [
                str(self.root / self.meta.get_data_file_path(ep_idx))
                for ep_idx in self.episodes
            ]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")
        for image_key in self.meta.image_keys:
            hf_dataset = hf_dataset.cast_column(image_key, Image())
        hf_dataset.set_transform(lerobot_dataset_mod.hf_transform_to_torch)
        return hf_dataset

    lerobot_dataset_mod.LeRobotDataset.load_hf_dataset = _patched_load_hf_dataset

    parquet_files = tuple(sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"missing parquet episodes under {dataset_dir}")
    state_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    for parquet_path in parquet_files:
        frame = pd.read_parquet(
            parquet_path,
            columns=["observation.state", "action"],
        )
        state_batches.append(
            np.stack(frame["observation.state"].to_list()).astype(np.float32)
        )
        action_batches.append(np.stack(frame["action"].to_list()).astype(np.float32))
    state_matrix = np.concatenate(state_batches, axis=0)
    action_matrix = np.concatenate(action_batches, axis=0)
    norm_stats = {
        "state": normalize.NormStats(
            mean=state_matrix.mean(axis=0),
            std=state_matrix.std(axis=0),
            q01=np.quantile(state_matrix, 0.01, axis=0),
            q99=np.quantile(state_matrix, 0.99, axis=0),
        ),
        "actions": normalize.NormStats(
            mean=action_matrix.mean(axis=0),
            std=action_matrix.std(axis=0),
            q01=np.quantile(action_matrix, 0.01, axis=0),
            q99=np.quantile(action_matrix, 0.99, axis=0),
        ),
    }
    training_assets_root = runtime_dir / "training_assets"
    normalize.save(training_assets_root / DEFAULT_ASSET_ID, norm_stats)

    train_config_name = str(args.train_config_name or DEFAULT_TRAIN_CONFIG_NAME)
    base_config = train_config_mod.get_config(train_config_name)
    data_factory = train_config_mod.SimpleDataConfig(
        repo_id=dataset_dir.name,
        assets=train_config_mod.AssetsConfig(
            assets_dir=str(training_assets_root),
            asset_id=DEFAULT_ASSET_ID,
        ),
        base_config=train_config_mod.DataConfig(
            prompt_from_task=False,
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "observation/image": "observation.images.ego_view",
                            "observation/wrist_image": "observation.images.wrist_view",
                            "observation/state": "observation.state",
                            "actions": "action",
                            "prompt_raw": "recap_m2.prompt_raw",
                            "recap_m2.indicator_I": "recap_m2.indicator_I",
                        }
                    )
                ]
            ),
            action_sequence_keys=("action",),
        ),
        data_transforms=functools.partial(
            _build_variant_data_transforms,
            consumer_mode=consumer_mode,
            fixed_indicator_mode=fixed_indicator_mode,
        ),
    )
    checkpoint_base_dir = runtime_dir / "upstream_train_checkpoints"
    resource_settings = _resolve_train_resource_settings(base_config)
    resume_plan = _resolve_resume_checkpoint_plan(
        checkpoint_dir=checkpoint_base_dir / train_config_name / str(args.variant_name),
        runtime_dir=runtime_dir,
        requested_resume=bool(args.resume),
    )
    _validate_resume_target_step(
        resume_plan,
        num_train_steps=resource_settings["num_train_steps"],
    )
    train_config = dataclasses.replace(
        base_config,
        exp_name=args.variant_name,
        data=data_factory,
        checkpoint_base_dir=str(checkpoint_base_dir),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(args.weight_loader_params or DEFAULT_WEIGHT_LOADER_PARAMS)
        ),
        num_train_steps=resource_settings["num_train_steps"],
        batch_size=resource_settings["batch_size"],
        save_interval=resource_settings["save_interval"],
        num_workers=resource_settings["num_workers"],
        fsdp_devices=resource_settings["fsdp_devices"],
        log_interval=(
            int(args.log_interval)
            if args.log_interval is not None
            else int(base_config.log_interval)
        ),
        overwrite=not bool(args.resume),
        resume=bool(args.resume),
        wandb_enabled=False,
    )
    if resume_plan.mode == "params_bootstrap":
        train_config = dataclasses.replace(
            train_config,
            weight_loader=weight_loaders.CheckpointWeightLoader(
                str(resume_plan.seed_params_path)
            ),
            overwrite=True,
            resume=False,
        )
    elif resume_plan.mode == "fresh_overwrite":
        train_config = dataclasses.replace(
            train_config,
            overwrite=True,
            resume=False,
        )
    return train_config, export_dir, resume_plan


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _ = shutil.copytree(src, dst)


def _materialize_tree_with_hardlinks_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    try:
        for source_path in sorted(src.rglob("*")):
            target_path = dst / source_path.relative_to(src)
            if source_path.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_path, target_path)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
            raise
        shutil.rmtree(dst, ignore_errors=True)
        _copytree(src, dst)


def _real_variant_transform_kwargs(train_config) -> dict[str, str | None]:
    data_factory = getattr(train_config, "data", None)
    data_transforms = getattr(data_factory, "data_transforms", None)
    keywords = getattr(data_transforms, "keywords", {}) or {}
    if "consumer_mode" not in keywords:
        raise RuntimeError(
            "real_variant_transform_contract_missing_consumer_mode:"
            f"{getattr(train_config, 'name', '<unknown_config>')}"
        )
    return {
        "consumer_mode": str(keywords["consumer_mode"]),
        "fixed_indicator_mode": keywords.get("fixed_indicator_mode"),
    }


def _export_latest_checkpoint(train_config, export_dir: Path) -> None:
    checkpoint_run_dir = train_config.checkpoint_dir
    step_dirs = sorted(
        [
            path
            for path in checkpoint_run_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        ],
        key=lambda path: int(path.name),
    )
    if not step_dirs:
        raise FileNotFoundError(
            f"no checkpoint step directories found under {checkpoint_run_dir}"
        )
    latest_step_dir = step_dirs[-1]
    params_dir = latest_step_dir / "params"
    assets_dir = latest_step_dir / LIBERO_ASSET_SUBDIR
    default_num_train_steps = int(
        os.environ.get(
            TRAIN_DEFAULT_NUM_STEPS_ENV,
            str(DEFAULT_LOCAL_SAFE_NUM_TRAIN_STEPS),
        )
    )
    default_save_interval = int(
        os.environ.get(
            TRAIN_DEFAULT_SAVE_INTERVAL_ENV,
            str(DEFAULT_LOCAL_SAFE_SAVE_INTERVAL),
        )
    )
    num_train_steps_source = (
        os.environ.get(TRAIN_NUM_STEPS_SOURCE_ENV, "local_safe_default").strip()
        or "local_safe_default"
    )
    save_interval_source = (
        os.environ.get(TRAIN_SAVE_INTERVAL_SOURCE_ENV, "local_safe_default").strip()
        or "local_safe_default"
    )
    _ensure_real_export_layout(latest_step_dir)
    _copytree(params_dir, export_dir / "params")
    _copytree(assets_dir, export_dir / LIBERO_ASSET_SUBDIR)
    transform_kwargs = _real_variant_transform_kwargs(train_config)
    policy_contract = build_real_variant_policy_contract(
        base_train_config_name=str(train_config.name),
        exp_name=str(train_config.exp_name),
        consumer_mode=str(transform_kwargs["consumer_mode"]),
        fixed_indicator_mode=transform_kwargs["fixed_indicator_mode"],
        norm_stats_json_path=export_dir / NORM_STATS_RELATIVE_PATH,
    )
    manifest = attach_real_variant_policy_contract(
        {
            "schema_version": "openpi_real_variant_export_v2",
            "source_checkpoint_dir": str(latest_step_dir),
            "export_dir": str(export_dir),
            "artifact_mirror_mode": "directory_copy",
            "train_config_name": train_config.name,
            "exp_name": train_config.exp_name,
            "default_num_train_steps": default_num_train_steps,
            "num_train_steps": int(train_config.num_train_steps),
            "num_train_steps_source": num_train_steps_source,
            "default_save_interval": default_save_interval,
            "save_interval": int(train_config.save_interval),
            "save_interval_source": save_interval_source,
        },
        policy_contract=policy_contract,
    )
    _write_json(export_dir / "export_manifest.json", manifest)


def _latest_checkpoint_step_dir(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.is_dir():
        return None
    step_dirs = sorted(
        [path for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    if not step_dirs:
        return None
    return step_dirs[-1]


def _resolve_resume_checkpoint_plan(
    *,
    checkpoint_dir: Path,
    runtime_dir: Path,
    requested_resume: bool,
) -> ResumeCheckpointPlan:
    if not requested_resume:
        return ResumeCheckpointPlan(mode="fresh")

    latest_step_dir = _latest_checkpoint_step_dir(checkpoint_dir)
    if latest_step_dir is None:
        if checkpoint_dir.exists():
            return ResumeCheckpointPlan(mode="fresh_overwrite")
        return ResumeCheckpointPlan(mode="fresh")

    latest_step = int(latest_step_dir.name)
    params_dir = latest_step_dir / "params"
    if (latest_step_dir / "train_state").is_dir() and params_dir.is_dir():
        return ResumeCheckpointPlan(
            mode="native_resume",
            latest_step=latest_step,
            resume_state_step=latest_step + 1,
            latest_checkpoint_dir=latest_step_dir,
        )

    if not params_dir.is_dir():
        raise FileNotFoundError(
            "resume checkpoint is missing params required for restore/bootstrap: "
            + str(latest_step_dir)
        )

    seed_root = runtime_dir / "resume_compat_seed" / latest_step_dir.name
    seed_params_path = seed_root / "params"
    _materialize_tree_with_hardlinks_or_copy(params_dir, seed_params_path)
    return ResumeCheckpointPlan(
        mode="params_bootstrap",
        latest_step=latest_step,
        resume_state_step=latest_step + 1,
        latest_checkpoint_dir=latest_step_dir,
        seed_params_path=seed_params_path,
    )


def _validate_resume_target_step(
    resume_plan: ResumeCheckpointPlan, *, num_train_steps: int
) -> None:
    if resume_plan.mode != "params_bootstrap":
        return
    if int(resume_plan.resume_state_step or 0) >= int(num_train_steps):
        raise ValueError(
            "resume compatibility bootstrap requires num_train_steps greater than restored state step; "
            + f"resume_state_step={resume_plan.resume_state_step!r} num_train_steps={num_train_steps!r}"
        )


def _install_resume_bootstrap_step_hook(train_main_mod: Any, *, resume_step: int):
    original_init_train_state = train_main_mod.init_train_state

    def _wrapped_init_train_state(
        config: Any,
        init_rng: Any,
        mesh: Any,
        *,
        resume: bool,
    ):
        state, state_sharding = original_init_train_state(
            config,
            init_rng,
            mesh,
            resume=resume,
        )
        if resume:
            return state, state_sharding
        return (
            train_main_mod.dataclasses.replace(state, step=int(resume_step)),
            state_sharding,
        )

    train_main_mod.init_train_state = _wrapped_init_train_state

    def _finalize_resume_bootstrap() -> None:
        train_main_mod.init_train_state = original_init_train_state

    return _finalize_resume_bootstrap


def _run_internal_export(args: argparse.Namespace) -> int:
    import importlib.util

    _prioritize_upstream_openpi_imports()
    train_script_path = OPENPI_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location(
        "openpi_train_script", train_script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load train.py from {train_script_path}")
    train_main_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_main_mod)

    probe_metrics_raw = str(getattr(args, "probe_metrics_path", "")).strip()
    finalize_probe_hooks = None
    if probe_metrics_raw:
        finalize_probe_hooks = _install_one_step_probe_hooks(
            train_main_mod,
            probe_metrics_path=Path(probe_metrics_raw).resolve(),
        )
    v22_trace_raw = str(getattr(args, "v22_trace_dir", "")).strip()
    validate_v22_hooks = None
    finalize_v22_hooks = None
    if v22_trace_raw:
        validate_v22_hooks, finalize_v22_hooks = _install_v22_real_training_hooks(
            train_main_mod,
            trace_config=V22RealTraceConfig(
                trace_dir=Path(v22_trace_raw).resolve(),
                run_id=str(getattr(args, "v22_trace_run_id", "")).strip(),
                variant=str(getattr(args, "v22_trace_variant", "")).strip()
                or str(getattr(args, "variant", "")).strip(),
                emit_loss_decomposition=bool(args.v22_emit_loss_decomposition),
                emit_threshold_trace=bool(args.v22_emit_threshold_trace),
                emit_alpha_dual_trace=bool(args.v22_emit_alpha_dual_trace),
                enable_r2_phase_threshold_switching=bool(
                    args.v22_enable_r2_phase_threshold_switching
                ),
                enable_r4_alpha_dual_loss=bool(args.v22_enable_r4_alpha_dual_loss),
                phase_threshold_step=int(args.v22_phase_threshold_step),
                alpha_pre_phase=float(args.v22_alpha_pre_phase),
                alpha_post_phase=float(args.v22_alpha_post_phase),
            ),
        )

    train_config, export_dir, resume_plan = _build_variant_train_config(args)
    finalize_resume_bootstrap = None
    if resume_plan.mode == "params_bootstrap":
        finalize_resume_bootstrap = _install_resume_bootstrap_step_hook(
            train_main_mod,
            resume_step=int(resume_plan.resume_state_step or 0),
        )
        _write_json(
            Path(args.runtime_dir).resolve() / "resume_compatibility.json",
            {
                "mode": resume_plan.mode,
                "latest_step": int(resume_plan.latest_step or 0),
                "resume_state_step": int(resume_plan.resume_state_step or 0),
                "latest_checkpoint_dir": str(resume_plan.latest_checkpoint_dir),
                "seed_params_path": str(resume_plan.seed_params_path),
                "reason": "resume_checkpoint_missing_train_state",
            },
        )
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    try:
        if validate_v22_hooks is not None:
            validate_v22_hooks()
        train_main_mod.main(train_config)
    finally:
        if finalize_v22_hooks is not None:
            finalize_v22_hooks()
        if finalize_resume_bootstrap is not None:
            finalize_resume_bootstrap()
        if finalize_probe_hooks is not None:
            finalize_probe_hooks()
    _export_latest_checkpoint(train_config, export_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_internal_parser()
    args = parser.parse_args(argv)
    if not args.internal_run_export:
        raise ValueError("real_variant_export.py only supports --internal-run-export")
    return _run_internal_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
