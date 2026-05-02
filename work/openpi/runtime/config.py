from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeBridgeConfig:
    """Shared runtime configuration for stock smoke and rollout workflows.

    The object centralizes host, port, timeouts, and optional artifact/runtime
    roots so scenarios and pipelines can pass one stable runtime contract across
    the bridge boundary.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    server_ready_timeout_s: float = 150.0
    client_timeout_s: float = 80.0
    video_fps: int = 10
    artifact_root: Path | None = None
    runtime_root: Path | None = None
    evidence_path: Path | None = None


DEFAULT_RUNTIME_BRIDGE_CONFIG = RuntimeBridgeConfig()


__all__ = [
    "DEFAULT_RUNTIME_BRIDGE_CONFIG",
    "RuntimeBridgeConfig",
]
