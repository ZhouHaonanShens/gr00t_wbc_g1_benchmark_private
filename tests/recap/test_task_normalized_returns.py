from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.critic_vlm.targets import (  # noqa: E402
    StepValueTargetInput,
    build_episode_empirical_returns,
    build_task_max_steps_from_step_inputs,
    empirical_return_from_episode_outcome,
    normalize_empirical_return,
)


def test_success_and_failure_episode_returns_follow_terminal_rule() -> None:
    assert build_episode_empirical_returns(episode_length=4, episode_success=True) == [
        -3.0,
        -2.0,
        -1.0,
        0.0,
    ]
    assert build_episode_empirical_returns(episode_length=4, episode_success=False) == [
        -4.0,
        -3.0,
        -2.0,
        -1.0,
    ]
    assert (
        empirical_return_from_episode_outcome(
            step_index=0,
            episode_length=4,
            episode_success=False,
        )
        == -4.0
    )


def test_task_normalized_returns_use_per_task_max_steps() -> None:
    inputs = [
        StepValueTargetInput(
            sample_id="a",
            task_key="task-a",
            episode_key="ep-a",
            step_index=0,
            episode_length=4,
            episode_success=True,
        ),
        StepValueTargetInput(
            sample_id="b",
            task_key="task-a",
            episode_key="ep-b",
            step_index=0,
            episode_length=6,
            episode_success=True,
        ),
        StepValueTargetInput(
            sample_id="c",
            task_key="task-b",
            episode_key="ep-c",
            step_index=0,
            episode_length=3,
            episode_success=True,
        ),
    ]

    task_max_steps = build_task_max_steps_from_step_inputs(inputs)

    assert task_max_steps == {"task-a": 6, "task-b": 3}
    assert normalize_empirical_return(empirical_return=-3.0, task_max_steps=6) == -0.5
