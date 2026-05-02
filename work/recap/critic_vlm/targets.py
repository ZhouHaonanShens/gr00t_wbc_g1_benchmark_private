from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .manifest import VlmCriticSample


@dataclass(frozen=True)
class ValueTarget:
    sample_id: str
    task_key: str
    episode_key: str
    episode_success: bool
    task_max_steps: int
    empirical_return: float
    normalized_return: float
    target_bin_index: int


@dataclass(frozen=True)
class StepValueTargetInput:
    sample_id: str
    task_key: str
    episode_key: str
    step_index: int
    episode_length: int
    episode_success: bool


def encode_value_to_bin_index(value: float, *, bin_centers: Iterable[float]) -> int:
    centers = [float(center) for center in bin_centers]
    if not centers:
        raise ValueError("bin_centers must be non-empty")
    best_index = 0
    best_distance = float("inf")
    for index, center in enumerate(centers):
        distance = abs(float(center) - float(value))
        if distance < best_distance:
            best_index = int(index)
            best_distance = float(distance)
    return int(best_index)


def empirical_return_from_episode_outcome(
    *, step_index: int, episode_length: int, episode_success: bool
) -> float:
    if int(episode_length) <= 0:
        raise ValueError(f"episode_length must be > 0, got {episode_length}")
    if int(step_index) < 0 or int(step_index) >= int(episode_length):
        raise ValueError(
            "step_index must satisfy 0 <= step_index < episode_length, got "
            + f"step_index={step_index} episode_length={episode_length}"
        )
    remaining_after_t = int(episode_length) - 1 - int(step_index)
    terminal_penalty = 0 if bool(episode_success) else 1
    return float(-1 * int(remaining_after_t + terminal_penalty))


def build_episode_empirical_returns(
    *, episode_length: int, episode_success: bool
) -> list[float]:
    return [
        empirical_return_from_episode_outcome(
            step_index=step_index,
            episode_length=episode_length,
            episode_success=episode_success,
        )
        for step_index in range(int(episode_length))
    ]


def normalize_empirical_return(
    *, empirical_return: float, task_max_steps: int
) -> float:
    if int(task_max_steps) <= 0:
        raise ValueError(f"task_max_steps must be > 0, got {task_max_steps}")
    return float(empirical_return) / float(task_max_steps)


def build_task_max_steps_from_step_inputs(
    step_inputs: Iterable[StepValueTargetInput],
) -> dict[str, int]:
    task_max_steps: dict[str, int] = {}
    for step_input in step_inputs:
        current = task_max_steps.get(str(step_input.task_key), 0)
        task_max_steps[str(step_input.task_key)] = max(
            int(current), int(step_input.episode_length)
        )
    if not task_max_steps:
        raise ValueError("at least one step input is required to build task max steps")
    return task_max_steps


def task_key_for_sample(sample: VlmCriticSample) -> str:
    prompt = str(sample.prompt_raw).strip()
    if prompt:
        return prompt
    return str(sample.split_name).strip() or "unknown_task"


def episode_key_for_sample(sample: VlmCriticSample) -> str:
    episode_id = str(sample.recap_episode_id).strip()
    if episode_id:
        return episode_id
    return f"{task_key_for_sample(sample)}::episode_index::{int(sample.episode_index)}"


def build_task_max_steps(samples: Iterable[VlmCriticSample]) -> dict[str, int]:
    task_max_steps: dict[str, int] = {}
    for sample in samples:
        task_key = task_key_for_sample(sample)
        current = task_max_steps.get(task_key, 0)
        task_max_steps[task_key] = max(int(current), int(sample.episode_length))
    if not task_max_steps:
        raise ValueError("at least one sample is required to build task max steps")
    return task_max_steps


def build_value_targets(
    samples: Iterable[VlmCriticSample], *, bin_centers: Iterable[float]
) -> dict[str, ValueTarget]:
    sample_list = list(samples)
    centers = [float(center) for center in bin_centers]
    task_max_steps = build_task_max_steps(sample_list)
    episode_returns: dict[str, list[float]] = defaultdict(list)
    for sample in sample_list:
        episode_returns[episode_key_for_sample(sample)].append(float(sample.return_g))
    targets: dict[str, ValueTarget] = {}
    for sample in sample_list:
        task_key = task_key_for_sample(sample)
        episode_key = episode_key_for_sample(sample)
        empirical_return = float(sample.return_g)
        normalized_return = normalize_empirical_return(
            empirical_return=empirical_return,
            task_max_steps=task_max_steps[task_key],
        )
        episode_success = max(episode_returns[episode_key]) >= 0.0
        targets[str(sample.sample_id)] = ValueTarget(
            sample_id=str(sample.sample_id),
            task_key=task_key,
            episode_key=episode_key,
            episode_success=bool(episode_success),
            task_max_steps=int(task_max_steps[task_key]),
            empirical_return=empirical_return,
            normalized_return=normalized_return,
            target_bin_index=encode_value_to_bin_index(
                normalized_return,
                bin_centers=centers,
            ),
        )
    return targets


def build_value_targets_from_step_inputs(
    step_inputs: Iterable[StepValueTargetInput], *, bin_centers: Iterable[float]
) -> dict[str, ValueTarget]:
    input_list = list(step_inputs)
    centers = [float(center) for center in bin_centers]
    task_max_steps = build_task_max_steps_from_step_inputs(input_list)
    episode_success_lookup: dict[str, bool] = {}
    for step_input in input_list:
        prior = episode_success_lookup.get(str(step_input.episode_key))
        current = bool(step_input.episode_success)
        if prior is not None and bool(prior) != current:
            raise ValueError(
                "episode_success must stay consistent within each episode_key: "
                + f"episode_key={step_input.episode_key!r}"
            )
        episode_success_lookup[str(step_input.episode_key)] = current
    targets: dict[str, ValueTarget] = {}
    for step_input in input_list:
        empirical_return = empirical_return_from_episode_outcome(
            step_index=int(step_input.step_index),
            episode_length=int(step_input.episode_length),
            episode_success=bool(step_input.episode_success),
        )
        normalized_return = normalize_empirical_return(
            empirical_return=empirical_return,
            task_max_steps=task_max_steps[str(step_input.task_key)],
        )
        targets[str(step_input.sample_id)] = ValueTarget(
            sample_id=str(step_input.sample_id),
            task_key=str(step_input.task_key),
            episode_key=str(step_input.episode_key),
            episode_success=bool(step_input.episode_success),
            task_max_steps=int(task_max_steps[str(step_input.task_key)]),
            empirical_return=float(empirical_return),
            normalized_return=float(normalized_return),
            target_bin_index=encode_value_to_bin_index(
                normalized_return,
                bin_centers=centers,
            ),
        )
    return targets


def resolve_effective_bin_centers(
    *,
    configured_bin_centers: tuple[float, ...] | list[float] | None,
    manifest_bin_centers: list[float],
) -> list[float]:
    if configured_bin_centers:
        return [float(center) for center in configured_bin_centers]
    return [float(center) for center in manifest_bin_centers]


__all__ = [
    "StepValueTargetInput",
    "ValueTarget",
    "build_task_max_steps",
    "build_task_max_steps_from_step_inputs",
    "build_episode_empirical_returns",
    "build_value_targets",
    "build_value_targets_from_step_inputs",
    "encode_value_to_bin_index",
    "empirical_return_from_episode_outcome",
    "episode_key_for_sample",
    "normalize_empirical_return",
    "resolve_effective_bin_centers",
    "task_key_for_sample",
]
