from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.critic_vlm.targets import (  # noqa: E402
    StepValueTargetInput,
    build_value_targets_from_step_inputs,
)


def test_value_target_builder_uses_task_max_steps_and_success_returns() -> None:
    targets = build_value_targets_from_step_inputs(
        [
            StepValueTargetInput(
                sample_id="a0",
                task_key="put the bowl on the plate",
                episode_key="ep-a",
                step_index=0,
                episode_length=5,
                episode_success=True,
            ),
            StepValueTargetInput(
                sample_id="a4",
                task_key="put the bowl on the plate",
                episode_key="ep-a",
                step_index=4,
                episode_length=5,
                episode_success=True,
            ),
            StepValueTargetInput(
                sample_id="b6",
                task_key="put the bowl on the plate",
                episode_key="ep-b",
                step_index=6,
                episode_length=7,
                episode_success=True,
            ),
        ],
        bin_centers=[-1.0, -0.5, 0.0],
    )

    assert targets["a0"].task_max_steps == 7
    assert targets["a0"].empirical_return == -4.0
    assert targets["a0"].normalized_return == (-4.0 / 7.0)
    assert targets["a0"].target_bin_index == 1
    assert targets["a4"].empirical_return == 0.0
    assert targets["a4"].normalized_return == 0.0
    assert targets["a4"].target_bin_index == 2
    assert targets["b6"].episode_success is True
