from __future__ import annotations

from work.recap.r6_runtime_indicator_probe.contract import FinalVerdict, RuntimeTrace, WiringGraph


def compose_final(static_: WiringGraph, runtime: RuntimeTrace | None) -> FinalVerdict:
    if static_.static_verdict == "BROKEN":
        return "ACTIVE_PATH_BROKEN_STATIC"
    if runtime is None:
        return "INCONCLUSIVE"
    if static_.static_verdict == "WIRED" and runtime.runtime_verdict == "INDICATOR_PRESENT":
        return "ACTIVE_PATH_CONFIRMED_WIRED"
    if static_.static_verdict == "WIRED" and runtime.runtime_verdict == "INDICATOR_ABSENT":
        return "ACTIVE_PATH_BROKEN_AT_RUNTIME"
    return "INCONCLUSIVE"
