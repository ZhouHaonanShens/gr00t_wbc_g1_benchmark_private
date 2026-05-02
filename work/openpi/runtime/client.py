from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bridge import (
    _run_runtime_episode_subprocess,
    _run_stock_episode,
    _run_stock_episode_subprocess,
)


@dataclass(frozen=True)
class RuntimeEpisodeClient:
    """Launch stock or generic runtime episodes through subprocess helpers.

    The client is a thin facade over the bridge subprocess functions so callers
    can use a stable episode-level interface without importing the raw helper
    names or recreating runtime-dir conventions themselves.
    """

    def run_stock_episode(self, **kwargs):
        return _run_stock_episode_subprocess(**kwargs)

    def run_stock_episode_direct(self, **kwargs):
        return _run_stock_episode(**kwargs)

    def run_runtime_episode(self, **kwargs):
        return _run_runtime_episode_subprocess(**kwargs)

    def build_episode_runtime_dir(
        self,
        *,
        runtime_dir: Path,
        task_id: int,
        seed: int,
        trial_index: int,
    ) -> Path:
        return runtime_dir / "episodes" / f"task{task_id}_seed{seed}_trial{trial_index}"
