from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .source import load_checkpoint_provenance_pair, load_provenance_pair


@dataclass(frozen=True)
class CheckpointProvenanceLoader:
    def load_from_source_dir(
        self,
        source_dir: Path,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        return load_provenance_pair(source_dir)

    def load_from_checkpoint_ref(
        self,
        *,
        checkpoint_ref: str,
        raw_checkpoint_dir: str | None,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        return load_checkpoint_provenance_pair(
            checkpoint_ref=checkpoint_ref,
            raw_checkpoint_dir=raw_checkpoint_dir,
        )
