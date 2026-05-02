from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


EXPECTED_SCHEMA_VERSION = "openpi_libero_eval_protocol_v1"
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
DEFAULT_ARTIFACT_TOPIC = "openpi_libero_native"
LEGACY_G1_KEYS = (
    "env_id",
    "logical_task",
    "policy_horizon",
    "executed_action_steps",
)


@dataclass(frozen=True)
class LiberoEvalProtocol:
    schema_version: str
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


def build_libero_eval_protocol(
    *,
    suite: str = EXPECTED_SUITE,
    task_ids: Sequence[int] = EXPECTED_SMOKE_TASK_IDS,
    seed_manifest: Sequence[int] = EXPECTED_SMOKE_SEEDS,
    num_trials_per_task: int = EXPECTED_SMOKE_NUM_TRIALS,
    action_horizon: int = EXPECTED_ACTION_HORIZON,
    discrete_state_input: bool = EXPECTED_DISCRETE_STATE_INPUT,
    extra_delta_transform: bool = EXPECTED_EXTRA_DELTA_TRANSFORM,
    replan_steps: int = EXPECTED_REPLAN_STEPS,
    num_steps_wait: int = EXPECTED_NUM_STEPS_WAIT,
) -> LiberoEvalProtocol:
    protocol = LiberoEvalProtocol(
        schema_version=EXPECTED_SCHEMA_VERSION,
        suite=str(suite),
        task_ids=tuple(int(value) for value in task_ids),
        seed_manifest=tuple(int(value) for value in seed_manifest),
        num_trials_per_task=int(num_trials_per_task),
        evaluation_tier="",
        action_horizon=int(action_horizon),
        discrete_state_input=bool(discrete_state_input),
        extra_delta_transform=bool(extra_delta_transform),
        replan_steps=int(replan_steps),
        num_steps_wait=int(num_steps_wait),
    )
    return validate_libero_eval_protocol(protocol)


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


def _reject_legacy_g1_keys(payload: Mapping[str, object]) -> None:
    for key in LEGACY_G1_KEYS:
        if key in payload:
            raise ValueError(
                f"legacy field {key!r} is not accepted; use suite/task_ids/seed_manifest instead"
            )


def _as_protocol_mapping(
    protocol: LiberoEvalProtocol | Mapping[str, object],
) -> Mapping[str, object]:
    if isinstance(protocol, Mapping):
        return protocol
    return asdict(protocol)


def validate_libero_eval_protocol(
    protocol: LiberoEvalProtocol | Mapping[str, object],
) -> LiberoEvalProtocol:
    payload = _as_protocol_mapping(protocol)
    _reject_legacy_g1_keys(payload)

    schema_version = str(payload.get("schema_version", ""))
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"invalid schema_version {schema_version!r}; expected {EXPECTED_SCHEMA_VERSION!r}"
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

    return LiberoEvalProtocol(
        schema_version=schema_version,
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
    )


def build_libero_eval_artifact_paths(
    run_id: str | None = None,
    *,
    topic: str = DEFAULT_ARTIFACT_TOPIC,
) -> dict[str, Path]:
    topic_str = str(topic).strip()
    if not topic_str:
        raise ValueError("topic must be non-empty")

    run_id_str = "" if run_id is None else str(run_id).strip()
    runtime_dir = Path("agent/runtime_logs") / topic_str
    artifact_dir = Path("agent/artifacts") / topic_str
    if run_id_str:
        runtime_dir = runtime_dir / run_id_str
        artifact_dir = artifact_dir / run_id_str

    return {
        "runtime_dir": runtime_dir,
        "artifact_dir": artifact_dir,
        "summary_json": artifact_dir / "summary.json",
        "telemetry_dir": artifact_dir / "telemetry",
    }
