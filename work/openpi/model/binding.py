from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyModelBinding:
    variant: str
    checkpoint_ref: str
    serve_checkpoint_ref: str
    serve_checkpoint_mode: str
