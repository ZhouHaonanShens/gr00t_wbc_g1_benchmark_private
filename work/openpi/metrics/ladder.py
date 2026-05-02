from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .gating import build_v21_metric_ladder_summary


@dataclass(frozen=True)
class MetricLadderBuilder:
    """Build the canonical metric-ladder summary for one trace bundle.

    The builder fixes the primary metric choice at construction time so callers
    can request deterministic ladder payloads without reassembling the summary
    contract on every workflow path.
    """

    primary_metric_id: str

    def build(
        self,
        *,
        trace_rows: Sequence[Mapping[str, object]],
        authority_id: str,
        variant: str,
        checkpoint_ref: str,
        metric_profile: str,
    ) -> dict[str, object]:
        return build_v21_metric_ladder_summary(
            trace_rows=trace_rows,
            authority_id=authority_id,
            variant=variant,
            checkpoint_ref=checkpoint_ref,
            metric_profile=metric_profile,
            primary_metric_id=self.primary_metric_id,
        )
