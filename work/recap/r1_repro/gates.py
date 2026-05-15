from __future__ import annotations

from enum import Enum
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any


class Verdict(str, Enum):
    PASS = "PASS"
    PASS_CLEAN = "PASS_CLEAN"
    PASS_WITH_RISK = "PASS_WITH_RISK"
    FAIL_LOW = "FAIL_LOW"
    FAIL_HIGH = "FAIL_HIGH"
    FAIL_DRIVER_STATUS = "FAIL_DRIVER_STATUS"
    FAIL_AMBIGUOUS = "FAIL_AMBIGUOUS"
    FAIL_PATH_MISSING = "FAIL_PATH_MISSING"
    FAIL_SHA_DRIFT = "FAIL_SHA_DRIFT"
    FAIL_OUT_OF_TREE_WRITE = "FAIL_OUT_OF_TREE_WRITE"
    FAIL_INCOMPLETE = "FAIL_INCOMPLETE"


def _z_for_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    return float(NormalDist().inv_cdf(1.0 - ((1.0 - level) / 2.0)))


def _validate_count(success_count: int, n: int) -> None:
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if success_count < 0 or success_count > n:
        raise ValueError(
            f"success_count must be between 0 and n, got {success_count}/{n}"
        )


def wilson_ci_on_rate(
    success_count: int,
    n: int,
    level: float = 0.95,
) -> tuple[float, float]:
    _validate_count(success_count, n)
    z = _z_for_level(level)
    phat = success_count / n
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (phat + (z2 / (2.0 * n))) / denom
    half_width = (
        z
        * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
        / denom
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def newcombe_ci_on_delta(
    swap_success: int,
    baseline_success: int,
    n_swap: int,
    n_baseline: int,
    level: float = 0.95,
) -> tuple[float, float]:
    _validate_count(swap_success, n_swap)
    _validate_count(baseline_success, n_baseline)
    swap_rate = swap_success / n_swap
    baseline_rate = baseline_success / n_baseline
    swap_low, swap_high = wilson_ci_on_rate(swap_success, n_swap, level)
    base_low, base_high = wilson_ci_on_rate(baseline_success, n_baseline, level)
    delta = swap_rate - baseline_rate
    lower = delta - math.sqrt((swap_rate - swap_low) ** 2 + (base_high - baseline_rate) ** 2)
    upper = delta + math.sqrt((swap_high - swap_rate) ** 2 + (baseline_rate - base_low) ** 2)
    return max(-1.0, lower), min(1.0, upper)


def _nonempty_diff(audit: dict[str, Any]) -> bool:
    for key, value in audit.items():
        if key in {"status", "schema_version", "generated_at_utc"}:
            continue
        if isinstance(value, dict):
            if _nonempty_diff(value):
                return True
        elif bool(value):
            return True
    return False


def _nonempty_risk(risk: dict[str, Any]) -> bool:
    return any(bool(risk.get(level)) for level in ("HIGH", "MEDIUM", "LOW"))


def gate_r1_2_variant_audit(
    audit: dict[str, Any],
    risk: dict[str, Any],
    variant_root: Path,
) -> Verdict:
    if not Path(variant_root).is_dir():
        return Verdict.FAIL_PATH_MISSING
    status = str(audit.get("status", "")).upper()
    if status in {"FAIL_AMBIGUOUS", "AMBIGUOUS"} or bool(audit.get("ambiguous")):
        return Verdict.FAIL_AMBIGUOUS
    if status in {"FAIL_SHA_DRIFT", "SHA_DRIFT"} or bool(audit.get("sha_drift")):
        return Verdict.FAIL_SHA_DRIFT
    if _nonempty_diff(audit) or _nonempty_risk(risk):
        return Verdict.PASS_WITH_RISK
    return Verdict.PASS_CLEAN


def gate_r1_0_baseline_reproduction(
    success_count: int,
    formal_eval_status: str,
    episode_count: int,
    git_diff_clean_outside_artifact_dir: bool,
) -> Verdict:
    if not git_diff_clean_outside_artifact_dir:
        return Verdict.FAIL_OUT_OF_TREE_WRITE
    if formal_eval_status != "PASS" or episode_count != 30:
        return Verdict.FAIL_DRIVER_STATUS
    if success_count < 13:
        return Verdict.FAIL_LOW
    if success_count > 22:
        return Verdict.FAIL_HIGH
    return Verdict.PASS


def gate_r1_1_axis_complete(cell_results: dict[str, Any]) -> Verdict:
    required = {"r1_1_A", "r1_1_B", "r1_1_C", "r1_1_D", "r1_1_E"}
    if set(cell_results) < required:
        return Verdict.FAIL_INCOMPLETE
    for cell in required:
        result = cell_results[cell]
        success_count = getattr(result, "success_count", None)
        rate_ci = getattr(result, "wilson_ci_on_rate", None)
        status = getattr(result, "formal_eval_summary_status", None)
        if success_count is None or rate_ci is None or status != "PASS":
            return Verdict.FAIL_INCOMPLETE
    return Verdict.PASS


def _count_delta_to_rate(delta_count: int, n: int) -> float:
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    return float(delta_count / n)


def _abs_delta_lower_bound(
    delta_count: int,
    baseline_success: int,
    baseline_n: int,
    swap_n: int,
) -> float:
    swap_success = max(0, min(swap_n, int(baseline_success + delta_count)))
    low, high = newcombe_ci_on_delta(
        swap_success,
        baseline_success,
        swap_n,
        baseline_n,
        level=0.99,
    )
    if low > 0.0:
        return max(0.0, low)
    if high < 0.0:
        return max(0.0, abs(high))
    return 0.0


def descriptive_axis_sum_stats(
    per_axis_observed_deltas: list[int],
    per_axis_baseline_successes: list[int],
    baseline_n: int,
    swap_n: int = 30,
) -> dict[str, Any]:
    if len(per_axis_observed_deltas) != len(per_axis_baseline_successes):
        raise ValueError("per-axis delta and baseline lists must have the same length")
    per_axis_lower = [
        _abs_delta_lower_bound(delta, baseline, baseline_n, swap_n)
        for delta, baseline in zip(
            per_axis_observed_deltas,
            per_axis_baseline_successes,
            strict=True,
        )
    ]
    observed_abs = sum(
        abs(_count_delta_to_rate(delta, swap_n)) for delta in per_axis_observed_deltas
    )
    gap_observed = (
        max((abs(_count_delta_to_rate(delta, swap_n)) for delta in per_axis_observed_deltas), default=0.0)
    )
    return {
        "sum_abs_delta_observed": float(observed_abs),
        "sum_abs_delta_lower_95": float(sum(max(0.0, bound) for bound in per_axis_lower)),
        "per_axis_newcombe_lower_99_over_5": [float(bound) for bound in per_axis_lower],
        "gap_p0b_minus_t81b0_observed": float(gap_observed),
        "unmodeled_axes_documented_in_dossier": [
            "S2_ADAPTER (LoRA / adapter weight delta on the variant; not in current 5-axis ablation; recommended for R2 if R1 deltas under-account for gap)",
            "env_name suffix",
            "prompt phrasing",
            "simulator determinism",
            "observation pipeline",
        ],
        "interpretation_note": (
            "if sum_abs_delta_lower_95 is less than gap_p0b_minus_t81b0_observed, "
            "the 5-axis set may be incomplete; otherwise R1 cannot rule out "
            "additivity at n=30 -- see ADR Consequences for the n=30 noise floor."
        ),
    }
