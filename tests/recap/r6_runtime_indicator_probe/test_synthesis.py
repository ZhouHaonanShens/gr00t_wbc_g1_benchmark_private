from __future__ import annotations

from work.recap.r6_runtime_indicator_probe.contract import RuntimeTrace, WiringGraph
from work.recap.r6_runtime_indicator_probe.synthesis import compose_final


def _graph(verdict: str) -> WiringGraph:
    return WiringGraph("A.2", (), (), (), verdict != "BROKEN", verdict, "notes")  # type: ignore[arg-type]


def _runtime(verdict: str) -> RuntimeTrace:
    return RuntimeTrace("A.2", 0, "prompt", "p" * 64, "a" * 64, (0.0, 1.0, 2.0, 3.0, 4.0), verdict == "INDICATOR_PRESENT", verdict)  # type: ignore[arg-type]


def test_exact_final_verdict_table() -> None:
    assert compose_final(_graph("BROKEN"), None) == "ACTIVE_PATH_BROKEN_STATIC"
    assert compose_final(_graph("WIRED"), None) == "INCONCLUSIVE"
    assert compose_final(_graph("WIRED"), _runtime("INDICATOR_PRESENT")) == "ACTIVE_PATH_CONFIRMED_WIRED"
    assert compose_final(_graph("WIRED"), _runtime("INDICATOR_ABSENT")) == "ACTIVE_PATH_BROKEN_AT_RUNTIME"
    assert compose_final(_graph("AMBIGUOUS"), _runtime("INDICATOR_PRESENT")) == "INCONCLUSIVE"
