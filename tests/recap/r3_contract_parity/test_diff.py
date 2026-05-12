from __future__ import annotations

from work.recap.r3_contract_parity.contract import FAIL_HOT, NONE_PATTERN, PASS, WARN, PARITY_AXES
from work.recap.r3_contract_parity.diff import compare_axis


def test_diff_verdict_table() -> None:
    axis = PARITY_AXES[0]
    both_missing = compare_axis({}, {}, axis)
    assert (both_missing.verdict, both_missing.pattern_id) == (WARN, NONE_PATTERN)
    one_missing = compare_axis({"checkpoint": {"abs_path": "a"}}, {}, axis)
    assert (one_missing.verdict, one_missing.pattern_id) == (FAIL_HOT, axis.pattern_id)
    equal = compare_axis({"checkpoint": {"abs_path": ["a", "b"]}}, {"eval": {"checkpoint": ["a", "b"]}}, axis)
    assert (equal.verdict, equal.pattern_id) == (PASS, NONE_PATTERN)
    unequal = compare_axis({"checkpoint": {"abs_path": "a"}}, {"eval": {"checkpoint": "b"}}, axis)
    assert (unequal.verdict, unequal.pattern_id) == (FAIL_HOT, axis.pattern_id)
