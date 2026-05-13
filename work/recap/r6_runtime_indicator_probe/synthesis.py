from __future__ import annotations

from work.recap.r6_runtime_indicator_probe.contract import (
    FinalVerdict,
    ProbeCounterfactual,
    RuntimeTrace,
    WiringGraph,
)


def compose_final(
    static_: WiringGraph,
    runtime: RuntimeTrace | None,
    counterfactual: ProbeCounterfactual | None = None,
) -> FinalVerdict:
    _ = counterfactual
    if static_.static_verdict == "BROKEN":
        return "ACTIVE_PATH_BROKEN_STATIC"
    if static_.static_verdict == "WIRED" and runtime is None:
        return "WIRED_STATIC_UNCONFIRMED_RUNTIME"
    if static_.static_verdict == "WIRED" and runtime is not None:
        if runtime.runtime_verdict == "INDICATOR_PRESENT":
            return "ACTIVE_PATH_CONFIRMED_WIRED"
        if runtime.runtime_verdict == "INDICATOR_ABSENT":
            return "ACTIVE_PATH_BROKEN_AT_RUNTIME"
    return "INCONCLUSIVE"
