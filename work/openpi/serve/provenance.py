from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from work.openpi.prompting.routes import (
    PROMPT_TEXT_SURFACE_CANONICAL,
    PROMPT_TEXT_SURFACE_PROMPT_RAW_ONLY,
)
from work.recap import text_indicator


EXPECTED_SCHEMA_VERSION = "openpi_libero_stock_checkpoint_v1"
EXPECTED_MODEL_FAMILY = "openpi"
EXPECTED_MODEL_ANCHOR = "pi05_libero"
EXPECTED_CONFIG_NAME = "pi05_libero"
EXPECTED_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
EXPECTED_CHECKPOINT_SOURCE = "upstream_openpi_default_or_explicit_cli"
EXPECTED_ENV_MODE = "LIBERO"
EXPECTED_SIMULATOR = "MuJoCo"
EXPECTED_SUITE = "libero_spatial"
EXPECTED_ACTION_HORIZON = 10
EXPECTED_DISCRETE_STATE_INPUT = False
EXPECTED_EXTRA_DELTA_TRANSFORM = False
EXPECTED_REPLAN_STEPS = 5
EXPECTED_NUM_STEPS_WAIT = 10
EXPECTED_SMOKE_TASK_IDS: tuple[int, ...] = (0,)
EXPECTED_SMOKE_SEEDS: tuple[int, ...] = (7,)
EXPECTED_SMOKE_NUM_TRIALS = 1
EXPECTED_COMPARISON_TASK_IDS: tuple[int, ...] = (0, 1)
EXPECTED_COMPARISON_SEEDS: tuple[int, ...] = (7, 17)
EXPECTED_COMPARISON_NUM_TRIALS = 2
CRITIC_CHECKPOINT_REF_NOT_APPLICABLE = "not_applicable"
CRITIC_CHECKPOINT_REF_ADAPTER_REQUIRED = "adapter_required"
LEGACY_G1_KEYS = (
    "env_id",
    "logical_task",
    "policy_horizon",
    "executed_action_steps",
)


@dataclass(frozen=True)
class LiberoServerProvenance:
    schema_version: str
    model_family: str
    model_anchor: str
    config_name: str
    checkpoint: str
    checkpoint_source: str
    env_mode: str
    simulator: str
    prompt_route: str
    conditioning_mode: str
    indicator_mode: str
    indicator_source: str
    prompt_text_surface: str
    critic_checkpoint_ref: str
    norm_stats_source: str
    norm_stats_path: str
    asset_id: str
    suite: str
    task_ids: tuple[int, ...]
    seed_manifest: tuple[int, ...]
    num_trials_per_task: int
    evaluation_tier: str
    action_horizon: int
    discrete_state_input: bool
    extra_delta_transform: bool
    replan_steps: int
    num_steps_wait: int
    action_semantics: dict[str, object]


def build_libero_server_provenance_payload(
    *,
    prompt_provenance: Mapping[str, str],
    norm_provenance: Mapping[str, str],
    critic_checkpoint_ref: str = CRITIC_CHECKPOINT_REF_NOT_APPLICABLE,
    suite: str = EXPECTED_SUITE,
    task_ids: Sequence[int] = EXPECTED_SMOKE_TASK_IDS,
    seed_manifest: Sequence[int] = EXPECTED_SMOKE_SEEDS,
    num_trials_per_task: int = EXPECTED_SMOKE_NUM_TRIALS,
    action_horizon: int = EXPECTED_ACTION_HORIZON,
    discrete_state_input: bool = EXPECTED_DISCRETE_STATE_INPUT,
    extra_delta_transform: bool = EXPECTED_EXTRA_DELTA_TRANSFORM,
    replan_steps: int = EXPECTED_REPLAN_STEPS,
    num_steps_wait: int = EXPECTED_NUM_STEPS_WAIT,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "model_family": EXPECTED_MODEL_FAMILY,
        "model_anchor": EXPECTED_MODEL_ANCHOR,
        "config_name": EXPECTED_CONFIG_NAME,
        "checkpoint": EXPECTED_CHECKPOINT,
        "checkpoint_source": EXPECTED_CHECKPOINT_SOURCE,
        "env_mode": EXPECTED_ENV_MODE,
        "simulator": EXPECTED_SIMULATOR,
        "prompt_route": str(prompt_provenance["prompt_route"]),
        "conditioning_mode": str(prompt_provenance["conditioning_mode"]),
        "indicator_mode": str(
            prompt_provenance.get("indicator_mode", text_indicator.TEXT_INDICATOR_OMIT)
        ),
        "indicator_source": str(
            prompt_provenance.get("indicator_source", "runtime_indicator_mode")
        ),
        "prompt_text_surface": str(
            prompt_provenance.get(
                "prompt_text_surface",
                PROMPT_TEXT_SURFACE_PROMPT_RAW_ONLY,
            )
        ),
        "critic_checkpoint_ref": str(critic_checkpoint_ref).strip()
        or CRITIC_CHECKPOINT_REF_NOT_APPLICABLE,
        "norm_stats_source": str(norm_provenance["norm_stats_source"]),
        "norm_stats_path": str(norm_provenance["norm_stats_path"]),
        "asset_id": str(norm_provenance["asset_id"]),
        "suite": str(suite),
        "task_ids": [int(task_id) for task_id in task_ids],
        "seed_manifest": [int(seed) for seed in seed_manifest],
        "num_trials_per_task": int(num_trials_per_task),
        "action_horizon": int(action_horizon),
        "discrete_state_input": bool(discrete_state_input),
        "extra_delta_transform": bool(extra_delta_transform),
        "replan_steps": int(replan_steps),
        "num_steps_wait": int(num_steps_wait),
        "action_semantics": {
            "extra_delta_transform": bool(extra_delta_transform),
            "replan_steps": int(replan_steps),
            "num_steps_wait": int(num_steps_wait),
        },
    }
    validated = validate_libero_server_provenance(payload)
    payload["evaluation_tier"] = validated.evaluation_tier
    return payload


def _coerce_required_int(payload: Mapping[str, object], key: str) -> int:
    raw = payload.get(key)
    if raw is None or isinstance(raw, bool):
        raise ValueError(f"invalid {key} {raw!r}; expected integer")
    if not isinstance(raw, (int, float, str)):
        raise ValueError(f"invalid {key} {raw!r}; expected integer")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key} {raw!r}; expected integer") from exc


def _coerce_required_bool(payload: Mapping[str, object], key: str) -> bool:
    raw = payload.get(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    raise ValueError(f"invalid {key} {raw!r}; expected boolean")


def _coerce_required_non_empty_str(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing {key} in LIBERO stock provenance payload")
    return value


def _coerce_required_manifest(
    payload: Mapping[str, object], key: str
) -> tuple[int, ...]:
    raw = payload.get(key)
    if raw is None or isinstance(raw, (str, bytes, Mapping)):
        raise ValueError(f"invalid {key} {raw!r}; expected integer sequence")
    if not isinstance(raw, Sequence):
        raise ValueError(f"invalid {key} {raw!r}; expected integer sequence")
    coerced: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"invalid {key} {raw!r}; expected integer sequence")
        try:
            coerced.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid {key} {raw!r}; expected integer sequence"
            ) from exc
    if not coerced:
        raise ValueError(f"invalid {key} {raw!r}; expected non-empty integer sequence")
    return tuple(coerced)


def _reject_legacy_g1_keys(payload: Mapping[str, object]) -> None:
    for key in LEGACY_G1_KEYS:
        if key in payload:
            raise ValueError(
                f"legacy field {key!r} is not accepted in LIBERO stock provenance"
            )


def _coerce_mapping_bool(payload: Mapping[str, object], key: str) -> bool:
    raw = payload.get(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    raise ValueError(f"invalid {key} {raw!r}; expected boolean")


def _coerce_mapping_int(payload: Mapping[str, object], key: str) -> int:
    raw = payload.get(key)
    if raw is None or isinstance(raw, bool):
        raise ValueError(f"invalid {key} {raw!r}; expected integer")
    if not isinstance(raw, (int, float, str)):
        raise ValueError(f"invalid {key} {raw!r}; expected integer")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key} {raw!r}; expected integer") from exc


def _validate_action_semantics(payload: Mapping[str, object]) -> dict[str, object]:
    raw = payload.get("action_semantics")
    if not isinstance(raw, Mapping):
        raise ValueError("missing action_semantics in LIBERO stock provenance payload")
    semantics_payload = cast(Mapping[str, object], raw)

    extra_delta_transform = _coerce_mapping_bool(
        semantics_payload, "extra_delta_transform"
    )
    replan_steps = _coerce_mapping_int(semantics_payload, "replan_steps")
    num_steps_wait = _coerce_mapping_int(semantics_payload, "num_steps_wait")

    if extra_delta_transform is not EXPECTED_EXTRA_DELTA_TRANSFORM:
        raise ValueError(
            "invalid action_semantics.extra_delta_transform "
            + f"{extra_delta_transform!r}; expected {EXPECTED_EXTRA_DELTA_TRANSFORM!r}"
        )
    if replan_steps != EXPECTED_REPLAN_STEPS:
        raise ValueError(
            "invalid action_semantics.replan_steps "
            + f"{replan_steps!r}; expected {EXPECTED_REPLAN_STEPS!r}"
        )
    if num_steps_wait != EXPECTED_NUM_STEPS_WAIT:
        raise ValueError(
            "invalid action_semantics.num_steps_wait "
            + f"{num_steps_wait!r}; expected {EXPECTED_NUM_STEPS_WAIT!r}"
        )

    return {
        "extra_delta_transform": EXPECTED_EXTRA_DELTA_TRANSFORM,
        "replan_steps": EXPECTED_REPLAN_STEPS,
        "num_steps_wait": EXPECTED_NUM_STEPS_WAIT,
    }


def _validate_eval_tier(
    *,
    task_ids: tuple[int, ...],
    seed_manifest: tuple[int, ...],
    num_trials_per_task: int,
) -> str:
    if (
        task_ids == EXPECTED_SMOKE_TASK_IDS
        and seed_manifest == EXPECTED_SMOKE_SEEDS
        and num_trials_per_task == EXPECTED_SMOKE_NUM_TRIALS
    ):
        return "smoke"
    if (
        task_ids == EXPECTED_COMPARISON_TASK_IDS
        and seed_manifest == EXPECTED_COMPARISON_SEEDS
        and num_trials_per_task == EXPECTED_COMPARISON_NUM_TRIALS
    ):
        return "comparison"
    raise ValueError(
        "invalid LIBERO stock manifest "
        + f"task_ids={task_ids!r}, seed_manifest={seed_manifest!r}, "
        + f"num_trials_per_task={num_trials_per_task!r}; expected the frozen smoke or comparison tier"
    )


def _extract_training_route(
    train_manifest: Mapping[str, object] | None,
    checkpoint_provenance: Mapping[str, object] | None,
) -> Mapping[str, object]:
    for payload, key in (
        (checkpoint_provenance, "variant_derivation"),
        (train_manifest, "training_route"),
        (checkpoint_provenance, "training_route"),
    ):
        if payload is None:
            continue
        route = payload.get(key)
        if isinstance(route, Mapping):
            return cast(Mapping[str, object], route)
    return {}


def resolve_critic_checkpoint_ref(
    *,
    variant: str,
    train_manifest: Mapping[str, object] | None = None,
    checkpoint_provenance: Mapping[str, object] | None = None,
) -> str:
    for payload in (checkpoint_provenance, train_manifest):
        if payload is None:
            continue
        raw = payload.get("critic_checkpoint_ref")
        value = str(raw or "").strip()
        if value:
            return value
    training_route = _extract_training_route(train_manifest, checkpoint_provenance)
    raw_training_ref = training_route.get("critic_checkpoint_ref")
    training_ref = str(raw_training_ref or "").strip()
    if training_ref:
        return training_ref
    if str(variant).strip().lower() in {"stock", "stock_libero_ref_v1"}:
        return CRITIC_CHECKPOINT_REF_NOT_APPLICABLE
    return CRITIC_CHECKPOINT_REF_ADAPTER_REQUIRED


def validate_libero_server_provenance(
    payload: Mapping[str, object],
) -> LiberoServerProvenance:
    _reject_legacy_g1_keys(payload)

    schema_version = str(payload.get("schema_version", ""))
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"invalid schema_version {schema_version!r}; expected {EXPECTED_SCHEMA_VERSION!r}"
        )

    model_family = str(payload.get("model_family", ""))
    if model_family != EXPECTED_MODEL_FAMILY:
        raise ValueError(
            f"invalid model_family {model_family!r}; expected {EXPECTED_MODEL_FAMILY!r}"
        )

    model_anchor = str(payload.get("model_anchor", ""))
    if model_anchor != EXPECTED_MODEL_ANCHOR:
        raise ValueError(
            f"invalid model_anchor {model_anchor!r}; expected {EXPECTED_MODEL_ANCHOR!r}"
        )

    config_name = str(payload.get("config_name", ""))
    if config_name != EXPECTED_CONFIG_NAME:
        raise ValueError(
            f"invalid config_name {config_name!r}; expected {EXPECTED_CONFIG_NAME!r}"
        )

    checkpoint = str(payload.get("checkpoint", ""))
    if checkpoint != EXPECTED_CHECKPOINT:
        raise ValueError(
            f"invalid checkpoint {checkpoint!r}; expected {EXPECTED_CHECKPOINT!r}"
        )

    checkpoint_source = str(payload.get("checkpoint_source", ""))
    if checkpoint_source != EXPECTED_CHECKPOINT_SOURCE:
        raise ValueError(
            "invalid checkpoint_source "
            + f"{checkpoint_source!r}; expected {EXPECTED_CHECKPOINT_SOURCE!r}"
        )

    env_mode = str(payload.get("env_mode", ""))
    if env_mode != EXPECTED_ENV_MODE:
        raise ValueError(
            f"invalid env_mode {env_mode!r}; expected {EXPECTED_ENV_MODE!r}"
        )

    simulator = str(payload.get("simulator", ""))
    if simulator != EXPECTED_SIMULATOR:
        raise ValueError(
            f"invalid simulator {simulator!r}; expected {EXPECTED_SIMULATOR!r}"
        )

    prompt_route = str(payload.get("prompt_route", ""))
    if not prompt_route:
        raise ValueError("missing prompt_route in LIBERO stock provenance payload")

    conditioning_mode = str(payload.get("conditioning_mode", ""))
    if not conditioning_mode:
        raise ValueError("missing conditioning_mode in LIBERO stock provenance payload")

    indicator_mode = _coerce_required_non_empty_str(payload, "indicator_mode")
    indicator_mode = text_indicator.normalize_indicator_mode(
        indicator_mode,
        field_name="indicator_mode",
    )

    indicator_source = _coerce_required_non_empty_str(payload, "indicator_source")

    prompt_text_surface = _coerce_required_non_empty_str(payload, "prompt_text_surface")
    if prompt_text_surface not in {
        PROMPT_TEXT_SURFACE_CANONICAL,
        PROMPT_TEXT_SURFACE_PROMPT_RAW_ONLY,
    }:
        raise ValueError(
            "invalid prompt_text_surface "
            + f"{prompt_text_surface!r}; expected {PROMPT_TEXT_SURFACE_CANONICAL!r}|{PROMPT_TEXT_SURFACE_PROMPT_RAW_ONLY!r}"
        )

    critic_checkpoint_ref = _coerce_required_non_empty_str(
        payload,
        "critic_checkpoint_ref",
    )

    norm_stats_source = str(payload.get("norm_stats_source", ""))
    norm_stats_path = str(payload.get("norm_stats_path", ""))
    asset_id = str(payload.get("asset_id", ""))
    if not norm_stats_source or not norm_stats_path or not asset_id:
        raise ValueError(
            "LIBERO stock provenance must include norm_stats_source, norm_stats_path, and asset_id"
        )

    suite = str(payload.get("suite", ""))
    if suite != EXPECTED_SUITE:
        raise ValueError(f"invalid suite {suite!r}; expected {EXPECTED_SUITE!r}")

    task_ids = _coerce_required_manifest(payload, "task_ids")
    seed_manifest = _coerce_required_manifest(payload, "seed_manifest")
    num_trials_per_task = _coerce_required_int(payload, "num_trials_per_task")
    evaluation_tier = _validate_eval_tier(
        task_ids=task_ids,
        seed_manifest=seed_manifest,
        num_trials_per_task=num_trials_per_task,
    )

    action_horizon = _coerce_required_int(payload, "action_horizon")
    if action_horizon != EXPECTED_ACTION_HORIZON:
        raise ValueError(
            f"invalid action_horizon {action_horizon!r}; expected {EXPECTED_ACTION_HORIZON!r}"
        )

    discrete_state_input = _coerce_required_bool(payload, "discrete_state_input")
    if discrete_state_input is not EXPECTED_DISCRETE_STATE_INPUT:
        raise ValueError(
            "invalid discrete_state_input "
            + f"{discrete_state_input!r}; expected {EXPECTED_DISCRETE_STATE_INPUT!r}"
        )

    extra_delta_transform = _coerce_required_bool(payload, "extra_delta_transform")
    if extra_delta_transform is not EXPECTED_EXTRA_DELTA_TRANSFORM:
        raise ValueError(
            "invalid extra_delta_transform "
            + f"{extra_delta_transform!r}; expected {EXPECTED_EXTRA_DELTA_TRANSFORM!r}"
        )

    replan_steps = _coerce_required_int(payload, "replan_steps")
    if replan_steps != EXPECTED_REPLAN_STEPS:
        raise ValueError(
            f"invalid replan_steps {replan_steps!r}; expected {EXPECTED_REPLAN_STEPS!r}"
        )

    num_steps_wait = _coerce_required_int(payload, "num_steps_wait")
    if num_steps_wait != EXPECTED_NUM_STEPS_WAIT:
        raise ValueError(
            f"invalid num_steps_wait {num_steps_wait!r}; expected {EXPECTED_NUM_STEPS_WAIT!r}"
        )

    action_semantics = _validate_action_semantics(payload)

    return LiberoServerProvenance(
        schema_version=schema_version,
        model_family=model_family,
        model_anchor=model_anchor,
        config_name=config_name,
        checkpoint=checkpoint,
        checkpoint_source=checkpoint_source,
        env_mode=env_mode,
        simulator=simulator,
        prompt_route=prompt_route,
        conditioning_mode=conditioning_mode,
        indicator_mode=indicator_mode,
        indicator_source=indicator_source,
        prompt_text_surface=prompt_text_surface,
        critic_checkpoint_ref=critic_checkpoint_ref,
        norm_stats_source=norm_stats_source,
        norm_stats_path=norm_stats_path,
        asset_id=asset_id,
        suite=suite,
        task_ids=task_ids,
        seed_manifest=seed_manifest,
        num_trials_per_task=num_trials_per_task,
        evaluation_tier=evaluation_tier,
        action_horizon=action_horizon,
        discrete_state_input=discrete_state_input,
        extra_delta_transform=extra_delta_transform,
        replan_steps=replan_steps,
        num_steps_wait=num_steps_wait,
        action_semantics=action_semantics,
    )
