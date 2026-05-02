from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from work.openpi.contracts import OpenPIRuntimePaths

from .api import build_runtime_paths
from .bridge import TOPIC


@dataclass(frozen=True)
class RuntimePathsBuilder:
    """Build the canonical runtime path set for one topic.

    This wrapper keeps path construction explicit at the call site while
    delegating the frozen path contract to ``_required_paths``.
    """

    topic: str = TOPIC
    evidence_path: Path | None = None
    artifact_root: Path | None = None
    runtime_root: Path | None = None

    def build(self) -> OpenPIRuntimePaths:
        return build_runtime_paths(
            topic=self.topic,
            evidence_path=self.evidence_path,
            artifact_root=self.artifact_root,
            runtime_root=self.runtime_root,
        )
