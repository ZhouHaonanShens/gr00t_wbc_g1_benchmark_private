from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .gating import build_bootstrap_ci_v21


@dataclass(frozen=True)
class BootstrapEstimator:
    variant: str
    deterministic_seed_material: str

    def build(self, trace_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return build_bootstrap_ci_v21(
            trace_rows=trace_rows,
            variant=self.variant,
            deterministic_seed_material=self.deterministic_seed_material,
        )
