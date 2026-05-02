from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .gating import build_pairwise_delta_payload_v21


@dataclass(frozen=True)
class PairwiseDeltaBuilder:
    """Build pairwise delta payloads across multiple rollout traces.

    The builder fixes the primary metric and evaluation tier up front so eval
    workflows can emit comparable pairwise summaries with a deterministic seed
    contract.
    """

    primary_metric_id: str
    evaluation_tier: str

    def build(
        self,
        *,
        trace_rows_by_variant: Mapping[str, Sequence[Mapping[str, object]]],
        deterministic_seed_material: str,
    ) -> dict[str, object]:
        return build_pairwise_delta_payload_v21(
            trace_rows_by_variant=trace_rows_by_variant,
            primary_metric_id=self.primary_metric_id,
            deterministic_seed_material=deterministic_seed_material,
            evaluation_tier=self.evaluation_tier,
        )
