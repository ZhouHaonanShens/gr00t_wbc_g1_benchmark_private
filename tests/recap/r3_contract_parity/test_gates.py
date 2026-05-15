from __future__ import annotations

import pytest

from work.recap.r3_contract_parity.contract import FAIL_HOT, PASS, WARN, NONE_PATTERN, PARITY_AXES, R3AuditError, ParityAxisResult
from work.recap.r3_contract_parity.gates import cell_overall_verdict, collect_pattern_hits


def _result(verdict: str, pattern: str = NONE_PATTERN) -> ParityAxisResult:
    return ParityAxisResult(PARITY_AXES[0], "a", "b", verdict, pattern, "note")


def test_overall_verdict_order_and_patterns() -> None:
    assert cell_overall_verdict((_result(PASS),)) == PASS
    assert cell_overall_verdict((_result(PASS), _result(WARN))) == WARN
    hot = _result(FAIL_HOT, PARITY_AXES[0].pattern_id)
    assert cell_overall_verdict((_result(WARN), hot)) == FAIL_HOT
    assert collect_pattern_hits((hot, hot, _result(PASS))) == (PARITY_AXES[0].pattern_id,)
    with pytest.raises(R3AuditError):
        cell_overall_verdict(())
