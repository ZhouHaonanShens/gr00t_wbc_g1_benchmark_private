from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bridge import _client_get_env, _client_quat2axisangle, _get_max_steps


@dataclass(frozen=True)
class LiberoEnvironmentSession:
    """Create and query a LIBERO environment session for runtime execution.

    The session wraps the small set of environment-facing helpers needed by the
    runtime bridge so upper layers do not talk to raw environment utilities
    directly.
    """

    task: Any
    resolution: int
    seed: int

    def create(self):
        return _client_get_env(self.task, self.resolution, self.seed)

    @staticmethod
    def quat_to_axis_angle(quat: Any) -> Any:
        return _client_quat2axisangle(quat)

    @staticmethod
    def max_steps(task_suite_name: str) -> int:
        return _get_max_steps(task_suite_name)
