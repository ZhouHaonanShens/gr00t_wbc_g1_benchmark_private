from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


THRESHOLD_PHASE_PRETRAINING = "pretraining"
THRESHOLD_PHASE_FINE_TUNING = "fine_tuning"
DEFAULT_EPSILON_THRESHOLD_PHASE = THRESHOLD_PHASE_PRETRAINING
PRETRAINING_EPSILON_QUANTILE = 0.7
FINE_TUNING_EPSILON_QUANTILE = 0.6


@dataclass(frozen=True)
class PhaseThresholdPolicy:
    phase: str
    epsilon_quantile: float
    target_positive_fraction: float
    source_ref: str

    def as_metadata(self) -> dict[str, object]:
        return asdict(self)


PHASE_THRESHOLD_POLICIES: dict[str, PhaseThresholdPolicy] = {
    THRESHOLD_PHASE_PRETRAINING: PhaseThresholdPolicy(
        phase=THRESHOLD_PHASE_PRETRAINING,
        epsilon_quantile=PRETRAINING_EPSILON_QUANTILE,
        target_positive_fraction=0.3,
        source_ref="pistar06_appendix_f_advantage_threshold_pretraining",
    ),
    THRESHOLD_PHASE_FINE_TUNING: PhaseThresholdPolicy(
        phase=THRESHOLD_PHASE_FINE_TUNING,
        epsilon_quantile=FINE_TUNING_EPSILON_QUANTILE,
        target_positive_fraction=0.4,
        source_ref="pistar06_appendix_f_advantage_threshold_fine_tuning",
    ),
}

_PHASE_ALIASES = {
    "pretrain": THRESHOLD_PHASE_PRETRAINING,
    "pre-training": THRESHOLD_PHASE_PRETRAINING,
    "pre_training": THRESHOLD_PHASE_PRETRAINING,
    "pretraining": THRESHOLD_PHASE_PRETRAINING,
    "finetune": THRESHOLD_PHASE_FINE_TUNING,
    "fine-tune": THRESHOLD_PHASE_FINE_TUNING,
    "fine_tune": THRESHOLD_PHASE_FINE_TUNING,
    "fine-tuning": THRESHOLD_PHASE_FINE_TUNING,
    "fine_tuning": THRESHOLD_PHASE_FINE_TUNING,
    "finetuning": THRESHOLD_PHASE_FINE_TUNING,
}


def normalize_threshold_phase(raw: object | None) -> str:
    if raw is None:
        return DEFAULT_EPSILON_THRESHOLD_PHASE
    if not isinstance(raw, str):
        raise TypeError(f"threshold_phase must be a string, got {type(raw).__name__}")
    normalized = raw.strip().lower()
    if not normalized:
        return DEFAULT_EPSILON_THRESHOLD_PHASE
    canonical = _PHASE_ALIASES.get(normalized, normalized)
    if canonical not in PHASE_THRESHOLD_POLICIES:
        raise ValueError(
            "unsupported threshold_phase "
            + f"{raw!r}; expected one of {tuple(PHASE_THRESHOLD_POLICIES)}"
        )
    return canonical


def resolve_phase_threshold_policy(raw: object | None) -> PhaseThresholdPolicy:
    return PHASE_THRESHOLD_POLICIES[normalize_threshold_phase(raw)]


def _validate_quantile(raw: Any, *, context: str) -> float:
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"{context} must be a numeric quantile, got {raw!r}")
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{context} must be within [0, 1], got {raw!r}")
    return float(value)


def resolve_epsilon_quantile(
    *,
    threshold_phase: object | None = None,
    epsilon_quantile: object | None = None,
) -> float:
    if epsilon_quantile is not None:
        return _validate_quantile(
            epsilon_quantile,
            context="epsilon_quantile",
        )
    return float(resolve_phase_threshold_policy(threshold_phase).epsilon_quantile)


def build_phase_threshold_metadata(
    *,
    threshold_phase: object | None = None,
    epsilon_quantile: object | None = None,
) -> dict[str, object]:
    policy = resolve_phase_threshold_policy(threshold_phase)
    effective_quantile = resolve_epsilon_quantile(
        threshold_phase=policy.phase,
        epsilon_quantile=epsilon_quantile,
    )
    metadata = policy.as_metadata()
    metadata["epsilon_quantile"] = float(effective_quantile)
    metadata["epsilon_quantile_override"] = epsilon_quantile is not None
    return metadata


__all__ = [
    "DEFAULT_EPSILON_THRESHOLD_PHASE",
    "FINE_TUNING_EPSILON_QUANTILE",
    "PHASE_THRESHOLD_POLICIES",
    "PRETRAINING_EPSILON_QUANTILE",
    "PhaseThresholdPolicy",
    "THRESHOLD_PHASE_FINE_TUNING",
    "THRESHOLD_PHASE_PRETRAINING",
    "build_phase_threshold_metadata",
    "normalize_threshold_phase",
    "resolve_epsilon_quantile",
    "resolve_phase_threshold_policy",
]
