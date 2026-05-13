from __future__ import annotations

from work.recap.r6_runtime_indicator_probe.contract import WiringGraph
from work.recap.r6_runtime_indicator_probe.synthesis import compose_final


def test_wired_not_run_is_unconfirmed_runtime_not_inconclusive() -> None:
    graph = WiringGraph("A.2", (), (), (), True, "WIRED", "static path reaches action head")
    assert compose_final(graph, None) == "WIRED_STATIC_UNCONFIRMED_RUNTIME"
