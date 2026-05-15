from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StaticVerdict = Literal["WIRED", "BROKEN", "AMBIGUOUS"]
RuntimeVerdict = Literal["INDICATOR_PRESENT", "INDICATOR_ABSENT", "NOT_RUN"]
CounterfactualVerdict = Literal["INDICATOR_SENSITIVE", "INDICATOR_INVARIANT"]
FinalVerdict = Literal[
    "ACTIVE_PATH_CONFIRMED_WIRED",
    "ACTIVE_PATH_BROKEN_AT_RUNTIME",
    "ACTIVE_PATH_BROKEN_STATIC",
    "WIRED_STATIC_UNCONFIRMED_RUNTIME",
    "INCONCLUSIVE",
]


class R6Error(RuntimeError):
    """Raised when R6 inputs, contracts, or safety gates are invalid."""


class R6BudgetExceeded(R6Error):
    """Raised when an approved R6.1 probe would exceed its fixed budget."""


@dataclass(frozen=True)
class WiringEdge:
    src_symbol: str
    dst_symbol: str
    src_file: str
    src_line: int
    via: str


@dataclass(frozen=True)
class WiringGraph:
    cell_id: str
    edges: tuple[WiringEdge, ...]
    start_symbols: tuple[str, ...]
    sink_symbols: tuple[str, ...]
    reaches_sink: bool
    static_verdict: StaticVerdict
    notes: str


@dataclass(frozen=True)
class RuntimeTrace:
    cell_id: str
    episode_seed: int
    prompt_text_at_tokenizer: str
    prompt_tokens_sha256: str
    action_head_conditioning_sha256: str
    first_5_actions_l2: tuple[float, float, float, float, float]
    indicator_substring_present: bool
    runtime_verdict: RuntimeVerdict


@dataclass(frozen=True)
class ProbeCounterfactual:
    cell_id: str
    seed: int
    positive_trace_sha256: str
    negative_trace_sha256: str
    condition_sha_equal: bool
    first_5_actions_l2_diff: tuple[float, float, float, float, float]
    counterfactual_verdict: CounterfactualVerdict


@dataclass(frozen=True)
class CellProbeReport:
    cell_id: str
    static: WiringGraph
    runtime: RuntimeTrace | None
    final: FinalVerdict
