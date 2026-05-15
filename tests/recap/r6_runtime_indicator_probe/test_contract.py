from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import get_args

import pytest

import work.recap.r6_runtime_indicator_probe as r6
from work.recap.r6_runtime_indicator_probe.contract import (
    CellProbeReport,
    CounterfactualVerdict,
    FinalVerdict,
    ProbeCounterfactual,
    R6BudgetExceeded,
    R6Error,
    RuntimeTrace,
    RuntimeVerdict,
    StaticVerdict,
    WiringEdge,
    WiringGraph,
)


def test_public_exports_are_exact_and_lazy_accessible() -> None:
    assert r6.__all__ == (
        "WiringGraph",
        "RuntimeTrace",
        "CellProbeReport",
        "trace_wiring",
        "run_runtime_probe",
        "compose_final",
        "ENTRY_SYMBOLS",
    )
    for name in r6.__all__:
        assert getattr(r6, name) is not None


def test_literal_contracts_are_exact() -> None:
    assert get_args(StaticVerdict) == ("WIRED", "BROKEN", "AMBIGUOUS")
    assert get_args(RuntimeVerdict) == ("INDICATOR_PRESENT", "INDICATOR_ABSENT", "NOT_RUN")
    assert get_args(CounterfactualVerdict) == ("INDICATOR_SENSITIVE", "INDICATOR_INVARIANT")
    assert get_args(FinalVerdict) == (
        "ACTIVE_PATH_CONFIRMED_WIRED",
        "ACTIVE_PATH_BROKEN_AT_RUNTIME",
        "ACTIVE_PATH_BROKEN_STATIC",
        "WIRED_STATIC_UNCONFIRMED_RUNTIME",
        "INCONCLUSIVE",
    )


def test_dataclass_fields_are_exact_ordered_and_frozen() -> None:
    assert [f.name for f in fields(WiringEdge)] == ["src_symbol", "dst_symbol", "src_file", "src_line", "via"]
    assert [f.name for f in fields(WiringGraph)] == ["cell_id", "edges", "start_symbols", "sink_symbols", "reaches_sink", "static_verdict", "notes"]
    assert [f.name for f in fields(RuntimeTrace)] == ["cell_id", "episode_seed", "prompt_text_at_tokenizer", "prompt_tokens_sha256", "action_head_conditioning_sha256", "first_5_actions_l2", "indicator_substring_present", "runtime_verdict"]
    assert [f.name for f in fields(ProbeCounterfactual)] == ["cell_id", "seed", "positive_trace_sha256", "negative_trace_sha256", "condition_sha_equal", "first_5_actions_l2_diff", "counterfactual_verdict"]
    assert [f.name for f in fields(CellProbeReport)] == ["cell_id", "static", "runtime", "final"]
    edge = WiringEdge("a", "b", "f.py", 1, "via")
    with pytest.raises(FrozenInstanceError):
        edge.via = "mutated"  # type: ignore[misc]
    cf = ProbeCounterfactual("A.2", 1, "a" * 64, "b" * 64, False, (0, 0, 0, 0, 0), "INDICATOR_INVARIANT")
    with pytest.raises(FrozenInstanceError):
        cf.seed = 2  # type: ignore[misc]


def test_exception_hierarchy() -> None:
    assert issubclass(R6Error, RuntimeError)
    assert issubclass(R6BudgetExceeded, R6Error)
