from __future__ import annotations

import pytest

from work.recap.r6_runtime_indicator_probe.contract import R6Error
from work.recap.r6_runtime_indicator_probe.wiring_graph import (
    ENTRY_SYMBOLS,
    SINK_SYMBOLS,
    _build_call_graph,
    _expand_imports,
    _reaches,
    trace_wiring,
)


def test_trace_wiring_covers_exact_evidence_cells_with_deterministic_wired_graph() -> None:
    first = trace_wiring("A.2")
    again = trace_wiring("a.2")
    assert first == again
    assert first.start_symbols == ENTRY_SYMBOLS
    assert first.sink_symbols == SINK_SYMBOLS
    assert first.reaches_sink is True
    assert first.static_verdict == "WIRED"
    assert {edge.dst_symbol for edge in first.edges} & set(SINK_SYMBOLS)
    assert all(edge.src_file and edge.src_line > 0 for edge in first.edges)
    assert all(trace_wiring(cell).static_verdict == "WIRED" for cell in ("A.2", "A.3", "A.4", "A.5"))


def test_trace_rejects_a1_and_unknown_cells() -> None:
    for cell in ("A.1", "Q5", ""):
        with pytest.raises(R6Error):
            trace_wiring(cell)


def test_build_call_graph_and_reaches_are_public_contract_helpers() -> None:
    edges = _build_call_graph(ENTRY_SYMBOLS[0])
    graph: dict[str, list[str]] = {}
    for edge in edges:
        graph.setdefault(edge.src_symbol, []).append(edge.dst_symbol)
    assert _reaches(graph, ENTRY_SYMBOLS[0], SINK_SYMBOLS) is True
    with pytest.raises(R6Error):
        _build_call_graph("work.recap.text_indicator.not_an_entry")


def test_expand_imports_documents_wrapper_aliases_and_submodule_stub() -> None:
    assert _expand_imports("work.openpi.prompting.routes.build_runtime_prompt_route") == (
        "work.openpi.recap.prompt_builder.build_runtime_prompt_route",
    )
    assert _expand_imports("submodules.gr00t.<Transformer>.forward") == ()
