from __future__ import annotations

from typing import Any, Mapping

from work.recap.r3_contract_parity.contract import (
    FAIL_HOT,
    NONE_PATTERN,
    PASS,
    WARN,
    _MISSING,
    ParityAxisResult,
    ParityAxisSpec,
)


def get_path(snapshot: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = snapshot
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
    return node


def compare_axis(train_snapshot: Mapping[str, Any], eval_snapshot: Mapping[str, Any], axis: ParityAxisSpec) -> ParityAxisResult:
    train_value = get_path(train_snapshot, axis.train_path)
    eval_value = get_path(eval_snapshot, axis.eval_path)
    # Verdict table is intentionally closed: both absent is WARN; one absent or unequal is hot.
    if train_value is _MISSING and eval_value is _MISSING:
        verdict, pattern_id, note = WARN, NONE_PATTERN, "both sides missing"
    elif train_value is _MISSING or eval_value is _MISSING:
        verdict, pattern_id, note = FAIL_HOT, axis.pattern_id, "one side missing"
    elif train_value == eval_value:
        verdict, pattern_id, note = PASS, NONE_PATTERN, "values match"
    else:
        verdict, pattern_id, note = FAIL_HOT, axis.pattern_id, "values differ"
    return ParityAxisResult(axis, train_value, eval_value, verdict, pattern_id, note)


def compare_all(train_snapshot: Mapping[str, Any], eval_snapshot: Mapping[str, Any], axes: tuple[ParityAxisSpec, ...]) -> tuple[ParityAxisResult, ...]:
    return tuple(compare_axis(train_snapshot, eval_snapshot, axis) for axis in axes)
