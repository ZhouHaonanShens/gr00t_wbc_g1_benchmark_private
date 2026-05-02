from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .source import resolve_checkpoint_instance_binding


@dataclass(frozen=True)
class CheckpointBindingResolver:
    """Build a stable instance binding for a checkpoint reference.

    The binding is used as an identity surface for runtime/eval artifacts, so
    callers can compare instances without reinterpreting higher-level business
    semantics.
    """

    key_files: tuple[Path, ...]
    schema_version: str

    def resolve(self, checkpoint_ref: str) -> str:
        return resolve_checkpoint_instance_binding(
            checkpoint_ref,
            key_files=self.key_files,
            schema_version=self.schema_version,
        )
