from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


DEFAULT_ALPHA_DUAL_LOSS_COEFFICIENT = 1.0
DEFAULT_DUAL_LOSS_DROPOUT_P = 0.3
DUAL_LOSS_SCHEMA_VERSION = "recap_alpha_dual_loss_v1"
DUAL_LOSS_FORMULA = "L = L_unconditioned + alpha * L_conditioned"
LOSS_COMPONENT_KEYS: tuple[str, ...] = (
    "flow_loss",
    "discrete_action_ce",
    "text_ce",
)


@dataclass(frozen=True)
class DualLossBreakdown:
    flow_loss: object
    discrete_action_ce: object
    text_ce: object
    total_loss: object

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        context: str,
    ) -> "DualLossBreakdown":
        components = {
            key: _coerce_finite_scalar(payload.get(key), context=f"{context}.{key}")
            for key in LOSS_COMPONENT_KEYS
        }
        expected_total = _sum_components(components.values())
        total = payload.get("total_loss")
        total_value = expected_total if total is None else _coerce_finite_scalar(
            total,
            context=f"{context}.total_loss",
        )
        if not _scalars_equal(total_value, expected_total):
            raise ValueError(
                f"{context}.total_loss must equal component sum; "
                + f"expected {_scalar_to_float(expected_total)} got {_scalar_to_float(total_value)}"
            )
        return cls(
            flow_loss=components["flow_loss"],
            discrete_action_ce=components["discrete_action_ce"],
            text_ce=components["text_ce"],
            total_loss=expected_total,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "flow_loss": self.flow_loss,
            "discrete_action_ce": self.discrete_action_ce,
            "text_ce": self.text_ce,
            "total_loss": self.total_loss,
        }


@dataclass(frozen=True)
class DualLossConfig:
    alpha: float = DEFAULT_ALPHA_DUAL_LOSS_COEFFICIENT
    dropout_p: float = DEFAULT_DUAL_LOSS_DROPOUT_P
    mode: str = "explicit_alpha_dual_loss"

    def __post_init__(self) -> None:
        _coerce_non_negative_float(self.alpha, context="alpha")
        dropout = _coerce_finite_float(self.dropout_p, context="dropout_p")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout_p must be within [0, 1], got {self.dropout_p!r}")


def _is_tensor_scalar(raw: object) -> bool:
    module_name = type(raw).__module__.split(".", maxsplit=1)[0]
    return module_name == "torch" and hasattr(raw, "detach") and hasattr(raw, "numel")


def _scalar_to_float(raw: object) -> float:
    if _is_tensor_scalar(raw):
        detached = raw.detach()
        if int(detached.numel()) != 1:
            raise ValueError(f"loss component must be scalar, got shape {tuple(detached.shape)}")
        return float(detached.reshape(()).float().cpu().item())
    return float(raw)


def _coerce_finite_scalar(raw: Any, *, context: str) -> object:
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"{context} must be finite float-like, got {raw!r}")
    value = _scalar_to_float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite, got {raw!r}")
    return raw if _is_tensor_scalar(raw) else float(value)


def _coerce_finite_float(raw: Any, *, context: str) -> float:
    return float(_scalar_to_float(_coerce_finite_scalar(raw, context=context)))


def _coerce_non_negative_float(raw: Any, *, context: str) -> float:
    value = _coerce_finite_float(raw, context=context)
    if value < 0.0:
        raise ValueError(f"{context} must be non-negative, got {raw!r}")
    return float(value)


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _round_scalar(value: object) -> object:
    return value if _is_tensor_scalar(value) else _rounded(float(value))


def _sum_components(values: object) -> object:
    total: object | None = None
    for value in values:
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return _round_scalar(total)


def _scalars_equal(left: object, right: object) -> bool:
    return _rounded(_scalar_to_float(left)) == _rounded(_scalar_to_float(right))


def _combine_component(
    *,
    unconditioned: DualLossBreakdown,
    conditioned: DualLossBreakdown,
    key: str,
    alpha: float,
) -> float:
    combined = getattr(unconditioned, key) + float(alpha) * getattr(conditioned, key)
    return _round_scalar(combined)


def combine_alpha_dual_loss(
    *,
    unconditioned: Mapping[str, object],
    conditioned: Mapping[str, object],
    config: DualLossConfig | None = None,
) -> dict[str, object]:
    effective_config = config or DualLossConfig()
    alpha = _coerce_non_negative_float(effective_config.alpha, context="alpha")
    unconditioned_loss = DualLossBreakdown.from_mapping(
        unconditioned,
        context="unconditioned",
    )
    conditioned_loss = DualLossBreakdown.from_mapping(
        conditioned,
        context="conditioned",
    )
    components = {
        key: _combine_component(
            unconditioned=unconditioned_loss,
            conditioned=conditioned_loss,
            key=key,
            alpha=alpha,
        )
        for key in LOSS_COMPONENT_KEYS
    }
    total_loss = _round_scalar(
        unconditioned_loss.total_loss + alpha * conditioned_loss.total_loss
    )
    return {
        "schema_version": DUAL_LOSS_SCHEMA_VERSION,
        "path_kind": "dual_alpha",
        "mode": str(effective_config.mode),
        "uses_conditioning": True,
        "alpha": float(alpha),
        "dropout_p": float(effective_config.dropout_p),
        "formula": DUAL_LOSS_FORMULA,
        "unconditioned": unconditioned_loss.as_dict(),
        "conditioned": conditioned_loss.as_dict(),
        "components": components,
        "total_loss": total_loss,
    }


def build_dual_loss_integration_report(
    *,
    training_surface: str,
    single_path_loss_name: str,
    dual_view_integrated: bool,
    replacement_strategy: str,
) -> dict[str, object]:
    status = "CLOSED" if bool(dual_view_integrated) else "PARTIAL"
    return {
        "schema_version": "recap_dual_loss_integration_report_v1",
        "training_surface": str(training_surface),
        "single_path_loss_name": str(single_path_loss_name),
        "dual_view_integrated": bool(dual_view_integrated),
        "replacement_strategy": str(replacement_strategy),
        "status": status,
    }


__all__ = [
    "DEFAULT_ALPHA_DUAL_LOSS_COEFFICIENT",
    "DEFAULT_DUAL_LOSS_DROPOUT_P",
    "DUAL_LOSS_FORMULA",
    "DUAL_LOSS_SCHEMA_VERSION",
    "LOSS_COMPONENT_KEYS",
    "DualLossBreakdown",
    "DualLossConfig",
    "build_dual_loss_integration_report",
    "combine_alpha_dual_loss",
]
