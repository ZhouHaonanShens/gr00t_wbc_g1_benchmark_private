from __future__ import annotations

from dataclasses import dataclass

from .source import (
    expected_stock_checkpoint,
    normalize_checkpoint_ref,
    resolve_servable_checkpoint_ref,
)


@dataclass(frozen=True)
class CheckpointResolver:
    """Normalize checkpoint references and resolve their servable form.

    This class is the canonical entry point for turning a caller-supplied
    checkpoint ref into a stable, validated value before runtime or eval code
    decides how to serve it.
    """

    stock_variants: frozenset[str]

    def normalize(self, raw_checkpoint_dir: str) -> str:
        return normalize_checkpoint_ref(raw_checkpoint_dir)

    def expected_stock_checkpoint(self) -> str:
        return expected_stock_checkpoint()

    def resolve_servable(self, *, checkpoint_ref: str, variant: str) -> tuple[str, str]:
        return resolve_servable_checkpoint_ref(
            checkpoint_ref=checkpoint_ref,
            variant=variant,
            stock_variants=self.stock_variants,
        )
