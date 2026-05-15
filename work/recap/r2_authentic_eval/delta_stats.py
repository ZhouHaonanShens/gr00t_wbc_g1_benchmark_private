"""R2 statistical-regime helpers for delta analysis and closure rendering.

Re-exports R1 CI functions under brief-naming aliases, adds three SSOT
helpers (per-cell P, family-wise rate, Newcombe half-width), and defines
the SummaryRow frozen dataclass for structured result rows.

Canonical values (computed at runtime, never hardcoded in the renderer):
  per_cell_below_trigger_probability(17, 30) ≈ 0.099
  family_wise_error_rate_at_baseline(17, 30) uses evidence-grade n=4 ≈ 0.341
  raw-observation n=5 remains available by explicitly passing n_cells=5
  newcombe_half_width_at_baseline(17, 30) ≈ 0.237
"""

from __future__ import annotations

import dataclasses
import math

try:
    import scipy.stats as _scipy_stats

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

from work.recap.r1_repro.gates import (
    newcombe_ci_on_delta as newcombe_delta_ci_95,
    wilson_ci_on_rate as wilson_ci_95,
)
from work.recap.r2_authentic_eval.exclusion import EVIDENCE_GRADE_N_CELLS

# ---------------------------------------------------------------------------
# Module-top SCREAMING_CASE constants
# ---------------------------------------------------------------------------

R2_BASELINE_SUCC_DEFAULT: int = 17
R2_BASELINE_N_DEFAULT: int = 30
R2_TRIGGER_THRESHOLD: float = 17 / 30 - 0.10

R2_SUMMARY_TABLE_SCHEMA_VERSION: str = "1.0.0"
R2_DECOMPOSITION_TABLE_SCHEMA_VERSION: str = "1.0.0"
R2_CELL_RESULT_SCHEMA_VERSION: str = "1.0.0"
R2_SCHEMA_VERSIONS: dict[str, str] = {
    "summary_table": R2_SUMMARY_TABLE_SCHEMA_VERSION,
    "decomposition_table": R2_DECOMPOSITION_TABLE_SCHEMA_VERSION,
    "cell_result": R2_CELL_RESULT_SCHEMA_VERSION,
}


# ---------------------------------------------------------------------------
# Statistical-regime SSOT helpers (V3-FIX-1)
# ---------------------------------------------------------------------------


def per_cell_below_trigger_probability(
    baseline_succ: int = R2_BASELINE_SUCC_DEFAULT,
    n: int = R2_BASELINE_N_DEFAULT,
    threshold: float = R2_TRIGGER_THRESHOLD,
) -> float:
    """P(succ/n < threshold | n, p=baseline_succ/n) via binomial CDF.

    Implements the strict-below semantics of the brief D3 trigger: a cell
    "fires" when its observed rate is *strictly below* the threshold.  This
    is equivalent to P(succ <= ceil(threshold * n) - 1), which at the
    canonical threshold 14/30 gives P(succ <= 13) ≈ 0.099.

    Uses scipy.stats.binom.cdf when scipy is available; falls back to an
    explicit summation via math.comb otherwise.
    """
    p = baseline_succ / n
    # "Strictly below" semantics: P(succ < threshold*n) = P(succ <= k_max)
    # where k_max = ceil(threshold*n) - 1.  For non-integer threshold*n this
    # equals floor(threshold*n); for the canonical boundary threshold=14/30
    # (where threshold*n=14.0 exactly) it correctly gives 13 rather than 14.
    k_max = math.ceil(threshold * n) - 1
    if _HAS_SCIPY:
        return float(_scipy_stats.binom.cdf(k_max, n, p))
    total = 0.0
    for k in range(k_max + 1):
        total += math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))
    return total


def family_wise_error_rate_at_baseline(
    baseline_succ: int = R2_BASELINE_SUCC_DEFAULT,
    n: int = R2_BASELINE_N_DEFAULT,
    n_cells: int = EVIDENCE_GRADE_N_CELLS,
) -> float:
    """Family-wise false-positive rate across n_cells independent cells.

    Returns 1 - (1 - per_cell_below_trigger_probability(baseline_succ, n))^n_cells.
    At the canonical baseline (17/30), the default evidence-grade n=4 is ≈ 0.341.
    The raw-observation n=5 comparison is explicit: pass ``n_cells=5``.
    """
    per_cell = per_cell_below_trigger_probability(baseline_succ, n)
    return 1.0 - (1.0 - per_cell) ** n_cells


def newcombe_half_width_at_baseline(
    baseline_succ: int = R2_BASELINE_SUCC_DEFAULT,
    n: int = R2_BASELINE_N_DEFAULT,
) -> float:
    """Newcombe 95% CI half-width for a delta measured against baseline.

    Computes (high - low) / 2 of newcombe_ci_on_delta(baseline_succ,
    baseline_succ, n, n) — the symmetric case (swap == baseline).
    At the canonical baseline (17/30) this is ≈ 0.237.
    """
    low, high = newcombe_delta_ci_95(baseline_succ, baseline_succ, n, n)
    return (high - low) / 2.0


# ---------------------------------------------------------------------------
# SummaryRow dataclass (plan §5.4)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SummaryRow:
    """Immutable row in the R2 summary table."""

    label: str
    training_algo: str
    n_train_steps: int
    success_count: int
    completed_episode_total: int
    rate: float
    wilson_ci_95: tuple[float, float]
    delta_vs_baseline: float
    newcombe_delta_ci_95: tuple[float, float]
    triggered_below_threshold: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # R1 re-exports (brief-name aliases)
    "wilson_ci_95",
    "newcombe_delta_ci_95",
    # SSOT helpers
    "per_cell_below_trigger_probability",
    "family_wise_error_rate_at_baseline",
    "newcombe_half_width_at_baseline",
    # Dataclass
    "SummaryRow",
    # Constants
    "R2_BASELINE_SUCC_DEFAULT",
    "R2_BASELINE_N_DEFAULT",
    "R2_TRIGGER_THRESHOLD",
    "R2_SUMMARY_TABLE_SCHEMA_VERSION",
    "R2_DECOMPOSITION_TABLE_SCHEMA_VERSION",
    "R2_CELL_RESULT_SCHEMA_VERSION",
    "R2_SCHEMA_VERSIONS",
    "EVIDENCE_GRADE_N_CELLS",
]
