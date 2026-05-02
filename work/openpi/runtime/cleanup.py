from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import subprocess

from .bridge import _close_handle, _terminate_process


@dataclass(frozen=True)
class RuntimeCleanup:
    """Centralize process and handle cleanup for the runtime bridge.

    Keeping cleanup in one tiny surface prevents workflow code from duplicating
    termination details across server, client, and harness call paths.
    """

    @staticmethod
    def close_process(proc: subprocess.Popen[str] | None) -> None:
        _terminate_process(proc)

    @staticmethod
    def close_handle(handle: Any | None) -> None:
        _close_handle(handle)
