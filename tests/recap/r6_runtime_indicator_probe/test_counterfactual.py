from __future__ import annotations

from work.recap.r6_runtime_indicator_probe.contract import RuntimeTrace
from work.recap.r6_runtime_indicator_probe.runtime_probe import _build_counterfactual


def _trace(condition_sha: str, actions: tuple[float, float, float, float, float]) -> RuntimeTrace:
    return RuntimeTrace("A.2", 20000, "prompt", "p" * 64, condition_sha, actions, True, "INDICATOR_PRESENT")


def test_counterfactual_sensitive_iff_any_diff_exceeds_threshold() -> None:
    cf = _build_counterfactual("A.2", _trace("a" * 64, (0, 0, 0, 0, 0)), _trace("b" * 64, (0, 0, 0.0011, 0, 0)))
    assert cf.condition_sha_equal is False
    assert cf.first_5_actions_l2_diff == (0.0, 0.0, 0.0011, 0.0, 0.0)
    assert cf.counterfactual_verdict == "INDICATOR_SENSITIVE"


def test_counterfactual_invariant_at_exact_threshold_and_equal_condition_sha() -> None:
    cf = _build_counterfactual("A.2", _trace("a" * 64, (1, 2, 3, 4, 5)), _trace("a" * 64, (1.001, 2, 3, 4, 5)))
    assert cf.condition_sha_equal is True
    assert cf.counterfactual_verdict == "INDICATOR_INVARIANT"
