from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, cast


METRIC_LADDER_SCHEMA_VERSION = "openpi_libero_metric_ladder_summary_v21"
BOOTSTRAP_SCHEMA_VERSION = "openpi_libero_rollout_bootstrap_ci_v21"
PAIRWISE_DELTA_SCHEMA_VERSION = "openpi_libero_rollout_pairwise_delta_v21"
DEFAULT_PRIMARY_METRIC_ID = "success_rate@0.50_budget"
HEADLINE_METRIC_ORDER: tuple[str, ...] = (
    "success_rate@0.50_budget",
    "success_rate@0.75_budget",
    "throughput_like_score",
)
COMPATIBILITY_ONLY_METRICS: tuple[str, ...] = ("success_rate@1.00_budget",)
ALLOWED_PRIMARY_METRIC_IDS: tuple[str, ...] = (
    "success_rate@0.50_budget",
    "success_rate@0.75_budget",
    "throughput_like_score",
    "undecidable_on_lite",
)
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95
LITE_PRIMARY_VARIANT = "stock_libero_ref_v1"
VARIANT_CODE_TO_ID: dict[str, str] = {
    "A": "stock_libero_ref_v1",
    "B": "fixedadv_relabel8d_control_v1",
    "C": "recap_only_relabel8d_v2",
    "X": "recap_shuffledadv_diag_v1",
}
REQUIRED_PAIR_LABELS: tuple[str, ...] = ("C-B", "C-X", "C-A", "B-A", "X-A")
PAIR_LABEL_TO_CODES: dict[str, tuple[str, str]] = {
    "C-B": ("C", "B"),
    "C-X": ("C", "X"),
    "C-A": ("C", "A"),
    "B-A": ("B", "A"),
    "X-A": ("X", "A"),
}
AGGREGATE_METRIC_IDS: tuple[str, ...] = (
    "success_rate@0.50_budget",
    "success_rate@0.75_budget",
    "success_rate@1.00_budget",
    "median_first_success_step_fraction",
    "timeout_rate",
    "throughput_like_score",
)
PAIRED_SUMMARY_SCHEMA_VERSION = "openpi_libero_paired_summary_abcx_v21"
GO_NO_GO_REPORT_SCHEMA_VERSION = "openpi_libero_go_no_go_report_v21"
ALLOWED_GATE_STATUSES: tuple[str, ...] = ("PASS", "HOLD", "FAIL", "NOT_APPLICABLE")
HEADLINE_VARIANT_CODES: tuple[str, ...] = ("A", "B", "C")
DIAGNOSTIC_VARIANT_CODES: tuple[str, ...] = ("X",)
EXPECTED_TASK_SUITE_NAME = "libero_spatial"
EXPECTED_TASK_IDS: tuple[int, ...] = (0, 1)
EXPECTED_SELECTION_POLICY = "stock_only_hard_seed_v1"
STOCK_SCAN_SUMMARY_SCHEMA_VERSION = "openpi_libero_stock_seed_scan_summary_v21"
TOP_K_HARDEST_SEEDS = 6
EXPECTED_AUTHORITY_ID_BY_MODE: dict[str, str] = {
    "lite": "fresh_rollout_v21_lite",
    "strong": "fresh_rollout_v21_strong",
}
PAIRED_SUMMARY_NAME_BY_MODE: dict[str, str] = {
    "lite": "paired_summary_abcx_v21_lite.json",
    "strong": "paired_summary_abcx_v21.json",
}
GO_NO_GO_NAME_BY_MODE: dict[str, str] = {
    "lite": "go_no_go_v21_lite.json",
    "strong": "go_no_go_v21.json",
}
REPO_ROOT = Path(__file__).resolve().parents[3]


class AggregationValidationError(ValueError):
    pass


def _require_mapping(raw: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise AggregationValidationError(
            f"{context} must be a mapping, got {type(raw).__name__}"
        )
    return cast(Mapping[str, object], raw)


def _require_sequence(raw: object, *, context: str) -> Sequence[object]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise AggregationValidationError(f"{context} must be a sequence")
    return cast(Sequence[object], raw)


def _coerce_int(raw: object, *, context: str) -> int:
    if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise AggregationValidationError(f"{context} must be integer-like, got {raw!r}")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise AggregationValidationError(
            f"{context} must be integer-like, got {raw!r}"
        ) from exc


def _coerce_float(raw: object, *, context: str) -> float:
    if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise AggregationValidationError(f"{context} must be float-like, got {raw!r}")
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise AggregationValidationError(
            f"{context} must be float-like, got {raw!r}"
        ) from exc


def _coerce_bool(raw: object, *, context: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise AggregationValidationError(f"{context} must be boolean-like, got {raw!r}")


def _coerce_optional_float(raw: object, *, context: str) -> float | None:
    if raw in {None, "", "null"}:
        return None
    return _coerce_float(raw, context=context)


def _coerce_int_sequence(raw: object, *, context: str) -> tuple[int, ...]:
    values = _require_sequence(raw, context=context)
    return tuple(
        _coerce_int(value, context=f"{context}[{index}]")
        for index, value in enumerate(values)
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _float_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _episode_key(row: Mapping[str, object], *, context: str) -> tuple[int, int, int]:
    return (
        _coerce_int(row.get("task_id"), context=f"{context}.task_id"),
        _coerce_int(row.get("seed"), context=f"{context}.seed"),
        _coerce_int(row.get("trial_idx"), context=f"{context}.trial_idx"),
    )


def _quantile_bounds(
    values: Sequence[float], *, confidence_level: float
) -> tuple[float, float]:
    sorted_values = sorted(values)
    alpha = (1.0 - confidence_level) / 2.0
    lower_index = max(0, int(len(sorted_values) * alpha))
    upper_index = min(len(sorted_values) - 1, int(len(sorted_values) * (1.0 - alpha)))
    return sorted_values[lower_index], sorted_values[upper_index]


def _assert_metric_equal(
    actual: float | None,
    expected: float | None,
    *,
    context: str,
) -> None:
    if not _float_equal(actual, expected):
        raise AggregationValidationError(
            f"{context} drifted from trace-derived value: actual={actual!r} expected={expected!r}"
        )


def _trace_metric_components(
    row: Mapping[str, object], *, context: str
) -> dict[str, object]:
    success = _coerce_bool(row.get("success"), context=f"{context}.success")
    executed_steps = _coerce_int(
        row.get("executed_steps"), context=f"{context}.executed_steps"
    )
    max_steps_resolved = _coerce_int(
        row.get("max_steps_resolved"), context=f"{context}.max_steps_resolved"
    )
    if max_steps_resolved <= 0:
        raise AggregationValidationError(
            f"{context}.max_steps_resolved must be positive"
        )
    if executed_steps < 0 or executed_steps > max_steps_resolved:
        raise AggregationValidationError(
            f"{context}.executed_steps must be within [0, max_steps_resolved]"
        )
    raw_first_success_step = row.get("first_success_step")
    first_success_step = (
        None
        if raw_first_success_step in {None, "", "null"}
        else _coerce_int(
            raw_first_success_step, context=f"{context}.first_success_step"
        )
    )
    if success and first_success_step is None:
        raise AggregationValidationError(
            f"{context}.first_success_step must be present when success=true"
        )
    if (not success) and first_success_step is not None:
        raise AggregationValidationError(
            f"{context}.first_success_step must be null when success=false"
        )
    if first_success_step is not None:
        if first_success_step < 1 or first_success_step > max_steps_resolved:
            raise AggregationValidationError(
                f"{context}.first_success_step must be within [1, max_steps_resolved]"
            )
        if first_success_step > executed_steps:
            raise AggregationValidationError(
                f"{context}.first_success_step cannot exceed executed_steps"
            )
    expected_success_50 = bool(
        first_success_step is not None
        and first_success_step <= math.floor(0.50 * max_steps_resolved)
    )
    expected_success_75 = bool(
        first_success_step is not None
        and first_success_step <= math.floor(0.75 * max_steps_resolved)
    )
    success_50 = _coerce_bool(
        row.get("success_within_50pct_budget"),
        context=f"{context}.success_within_50pct_budget",
    )
    success_75 = _coerce_bool(
        row.get("success_within_75pct_budget"),
        context=f"{context}.success_within_75pct_budget",
    )
    if success_50 != expected_success_50:
        raise AggregationValidationError(
            f"{context}.success_within_50pct_budget drifted from first_success_step"
        )
    if success_75 != expected_success_75:
        raise AggregationValidationError(
            f"{context}.success_within_75pct_budget drifted from first_success_step"
        )
    timeout_flag = _coerce_bool(
        row.get("timeout_flag"), context=f"{context}.timeout_flag"
    )
    return {
        "success": success,
        "executed_steps": executed_steps,
        "max_steps_resolved": max_steps_resolved,
        "first_success_step": first_success_step,
        "success_within_50pct_budget": success_50,
        "success_within_75pct_budget": success_75,
        "timeout_flag": timeout_flag,
    }


def metric_point_estimates_from_trace_rows_v21(
    trace_rows: Sequence[Mapping[str, object]],
) -> dict[str, float | None]:
    total_episodes = len(trace_rows)
    if total_episodes == 0:
        raise AggregationValidationError("cannot aggregate empty trace rows")
    success_50 = 0
    success_75 = 0
    successful_episodes = 0
    timeout_count = 0
    executed_steps_sum = 0
    success_fractions: list[float] = []
    for index, row in enumerate(trace_rows):
        components = _trace_metric_components(row, context=f"trace[{index}]")
        success_50 += int(bool(components["success_within_50pct_budget"]))
        success_75 += int(bool(components["success_within_75pct_budget"]))
        timeout_count += int(bool(components["timeout_flag"]))
        executed_steps = cast(int, components["executed_steps"])
        executed_steps_sum += executed_steps
        first_success_step = cast(int | None, components["first_success_step"])
        max_steps_resolved = cast(int, components["max_steps_resolved"])
        if first_success_step is not None:
            successful_episodes += 1
            success_fractions.append(first_success_step / float(max_steps_resolved))
    if executed_steps_sum <= 0 and successful_episodes > 0:
        raise AggregationValidationError(
            "throughput_like_score undefined because successful_episodes>0 but sum(executed_steps)<=0"
        )
    return {
        "success_rate@0.50_budget": float(success_50) / float(total_episodes),
        "success_rate@0.75_budget": float(success_75) / float(total_episodes),
        "success_rate@1.00_budget": float(successful_episodes) / float(total_episodes),
        "timeout_rate": float(timeout_count) / float(total_episodes),
        "throughput_like_score": (
            1000.0 * float(successful_episodes) / float(executed_steps_sum)
            if executed_steps_sum > 0
            else 0.0
        ),
        "median_first_success_step_fraction": (
            float(statistics.median(success_fractions)) if success_fractions else None
        ),
    }


def build_metric_ladder_summary_v21(
    *,
    trace_rows: Sequence[Mapping[str, object]],
    authority_id: str,
    variant: str,
    checkpoint_ref: str,
    metric_profile: str,
    primary_metric_id: str = DEFAULT_PRIMARY_METRIC_ID,
) -> dict[str, object]:
    if primary_metric_id not in ALLOWED_PRIMARY_METRIC_IDS:
        raise AggregationValidationError(
            f"primary_metric_id must be one of {ALLOWED_PRIMARY_METRIC_IDS!r}"
        )
    point_estimates = metric_point_estimates_from_trace_rows_v21(trace_rows)
    successful_episodes = sum(
        1
        for row in trace_rows
        if row.get("first_success_step") not in {None, "", "null"}
    )
    metrics = {
        metric_id: {"point_estimate": point_estimates[metric_id]}
        for metric_id in AGGREGATE_METRIC_IDS
    }
    return {
        "schema_version": METRIC_LADDER_SCHEMA_VERSION,
        "authority_id": authority_id,
        "variant": variant,
        "checkpoint_ref": checkpoint_ref,
        "metric_profile": metric_profile,
        "primary_metric_id": primary_metric_id,
        "headline_metric_order": list(HEADLINE_METRIC_ORDER),
        "compatibility_only_metrics": list(COMPATIBILITY_ONLY_METRICS),
        "total_episodes": len(trace_rows),
        "successful_episodes": successful_episodes,
        "failed_episodes": len(trace_rows) - successful_episodes,
        "metrics": metrics,
        "ladder": [
            {
                "metric_id": metric_id,
                "point_estimate": point_estimates[metric_id],
                "headline_rank": index + 1,
            }
            for index, metric_id in enumerate(HEADLINE_METRIC_ORDER)
        ],
    }


def build_bootstrap_ci_v21(
    *,
    trace_rows: Sequence[Mapping[str, object]],
    deterministic_seed_material: str,
    variant: str,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, object]:
    sample_size = len(trace_rows)
    if sample_size == 0:
        raise AggregationValidationError("cannot bootstrap empty trace rows")
    if bootstrap_iterations <= 0:
        raise AggregationValidationError("bootstrap_iterations must be positive")
    rng = random.Random(int(_sha256_text(deterministic_seed_material)[:16], 16))
    point_estimates = metric_point_estimates_from_trace_rows_v21(trace_rows)
    samples_by_metric: dict[str, list[float]] = {
        metric_id: [] for metric_id in AGGREGATE_METRIC_IDS
    }
    for _ in range(bootstrap_iterations):
        sample_rows = [
            trace_rows[rng.randrange(sample_size)] for _index in range(sample_size)
        ]
        sample_metrics = metric_point_estimates_from_trace_rows_v21(sample_rows)
        for metric_id in AGGREGATE_METRIC_IDS:
            value = sample_metrics[metric_id]
            if value is not None:
                samples_by_metric[metric_id].append(float(value))
    metric_payload: dict[str, object] = {}
    for metric_id, values in samples_by_metric.items():
        if not values:
            metric_payload[metric_id] = {
                "point_estimate": point_estimates[metric_id],
                "ci_lower": None,
                "ci_upper": None,
                "ci_width": None,
                "ci_non_degenerate": False,
                "sample_variance": 0.0,
                "sample_count": 0,
            }
            continue
        ci_lower, ci_upper = _quantile_bounds(values, confidence_level=confidence_level)
        sample_variance = float(statistics.variance(values)) if len(values) > 1 else 0.0
        ci_non_degenerate = not math.isclose(
            ci_lower, ci_upper, rel_tol=1e-12, abs_tol=1e-12
        )
        metric_payload[metric_id] = {
            "point_estimate": point_estimates[metric_id],
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "ci_width": float(ci_upper - ci_lower),
            "ci_non_degenerate": ci_non_degenerate,
            "sample_variance": sample_variance,
            "sample_count": len(values),
        }
    return {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "variant": variant,
        "sample_size": sample_size,
        "bootstrap_iterations": bootstrap_iterations,
        "confidence_level": confidence_level,
        "deterministic_seed_material": deterministic_seed_material,
        "metrics": metric_payload,
    }


def _metric_entry_point_estimate(
    metric_map: Mapping[str, object], *, metric_id: str, context: str
) -> float | None:
    metric_entry = _require_mapping(metric_map.get(metric_id), context=context)
    return _coerce_optional_float(
        metric_entry.get("point_estimate"), context=f"{context}.point_estimate"
    )


def _bootstrap_metric_entry(
    bootstrap_ci: Mapping[str, object], *, metric_id: str, context: str
) -> Mapping[str, object]:
    metrics = _require_mapping(
        bootstrap_ci.get("metrics"), context=f"{context}.metrics"
    )
    return _require_mapping(
        metrics.get(metric_id), context=f"{context}.metrics.{metric_id}"
    )


def _ci_is_non_degenerate(metric_entry: Mapping[str, object], *, context: str) -> bool:
    raw_flag = metric_entry.get("ci_non_degenerate")
    if raw_flag is not None:
        return _coerce_bool(raw_flag, context=f"{context}.ci_non_degenerate")
    ci_lower = _coerce_optional_float(
        metric_entry.get("ci_lower"), context=f"{context}.ci_lower"
    )
    ci_upper = _coerce_optional_float(
        metric_entry.get("ci_upper"), context=f"{context}.ci_upper"
    )
    return (
        ci_lower is not None
        and ci_upper is not None
        and not math.isclose(ci_lower, ci_upper, rel_tol=1e-12, abs_tol=1e-12)
    )


def select_primary_metric_id_v21(
    *, lite_variant_bundles: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    bundle = lite_variant_bundles.get(LITE_PRIMARY_VARIANT)
    if bundle is None:
        return {
            "primary_metric_id": "undecidable_on_lite",
            "selection_surface": "lite",
            "selection_variant": LITE_PRIMARY_VARIANT,
            "allowed_primary_metric_ids": list(ALLOWED_PRIMARY_METRIC_IDS),
            "candidates": [],
            "decision_reason": "lite stock baseline bundle missing",
        }
    metric_ladder_summary = _require_mapping(
        bundle.get("metric_ladder_summary"),
        context="lite_variant_bundles.stock_libero_ref_v1.metric_ladder_summary",
    )
    bootstrap_ci = _require_mapping(
        bundle.get("bootstrap_ci"),
        context="lite_variant_bundles.stock_libero_ref_v1.bootstrap_ci",
    )
    ladder_metrics = _require_mapping(
        metric_ladder_summary.get("metrics"),
        context="lite_variant_bundles.stock_libero_ref_v1.metric_ladder_summary.metrics",
    )
    candidates: list[dict[str, object]] = []
    for metric_id in HEADLINE_METRIC_ORDER:
        point_estimate = _metric_entry_point_estimate(
            ladder_metrics,
            metric_id=metric_id,
            context=f"lite_variant_bundles.stock.metric_ladder_summary.metrics.{metric_id}",
        )
        bootstrap_metric = _bootstrap_metric_entry(
            bootstrap_ci,
            metric_id=metric_id,
            context="lite_variant_bundles.stock.bootstrap_ci",
        )
        ci_non_degenerate = _ci_is_non_degenerate(
            bootstrap_metric,
            context=f"lite_variant_bundles.stock.bootstrap_ci.metrics.{metric_id}",
        )
        sample_variance = _coerce_float(
            bootstrap_metric.get("sample_variance", 0.0),
            context=f"lite_variant_bundles.stock.bootstrap_ci.metrics.{metric_id}.sample_variance",
        )
        point_estimate_le_095 = point_estimate is not None and point_estimate <= 0.95
        sample_variance_gt_zero = sample_variance > 0.0
        eligible = False
        if metric_id in {
            "success_rate@0.50_budget",
            "success_rate@0.75_budget",
        }:
            eligible = point_estimate_le_095 and ci_non_degenerate
        elif metric_id == "throughput_like_score":
            eligible = sample_variance_gt_zero and ci_non_degenerate
        candidates.append(
            {
                "metric_id": metric_id,
                "point_estimate": point_estimate,
                "ci_non_degenerate": ci_non_degenerate,
                "sample_variance": sample_variance,
                "point_estimate_le_0_95": point_estimate_le_095,
                "sample_variance_gt_zero": sample_variance_gt_zero,
                "eligible": eligible,
            }
        )
        if eligible:
            return {
                "primary_metric_id": metric_id,
                "selection_surface": "lite",
                "selection_variant": LITE_PRIMARY_VARIANT,
                "allowed_primary_metric_ids": list(ALLOWED_PRIMARY_METRIC_IDS),
                "candidates": candidates,
                "decision_reason": f"selected {metric_id} from lite stock bundle",
            }
    return {
        "primary_metric_id": "undecidable_on_lite",
        "selection_surface": "lite",
        "selection_variant": LITE_PRIMARY_VARIANT,
        "allowed_primary_metric_ids": list(ALLOWED_PRIMARY_METRIC_IDS),
        "candidates": candidates,
        "decision_reason": "all lite headline metrics degenerate or saturated",
    }


def assert_variant_aggregate_conservation_v21(
    *,
    trace_rows: Sequence[Mapping[str, object]],
    metric_ladder_summary: Mapping[str, object],
    bootstrap_ci: Mapping[str, object],
    summary: Mapping[str, object],
) -> None:
    point_estimates = metric_point_estimates_from_trace_rows_v21(trace_rows)
    total_episodes = len(trace_rows)
    successful_episodes = sum(
        1
        for row in trace_rows
        if row.get("first_success_step") not in {None, "", "null"}
    )
    timeout_count = sum(
        int(
            _coerce_bool(
                row.get("timeout_flag"), context=f"trace[{index}].timeout_flag"
            )
        )
        for index, row in enumerate(trace_rows)
    )
    if (
        _coerce_int(
            metric_ladder_summary.get("total_episodes"),
            context="metric_ladder_summary.total_episodes",
        )
        != total_episodes
    ):
        raise AggregationValidationError(
            "metric_ladder_summary.total_episodes drifted from per_episode_trace"
        )
    if (
        _coerce_int(
            metric_ladder_summary.get("successful_episodes"),
            context="metric_ladder_summary.successful_episodes",
        )
        != successful_episodes
    ):
        raise AggregationValidationError(
            "metric_ladder_summary.successful_episodes drifted from per_episode_trace"
        )
    if (
        _coerce_int(
            metric_ladder_summary.get("failed_episodes"),
            context="metric_ladder_summary.failed_episodes",
        )
        != total_episodes - successful_episodes
    ):
        raise AggregationValidationError(
            "metric_ladder_summary.failed_episodes drifted from per_episode_trace"
        )
    ladder_metrics = _require_mapping(
        metric_ladder_summary.get("metrics"), context="metric_ladder_summary.metrics"
    )
    bootstrap_metrics = _require_mapping(
        bootstrap_ci.get("metrics"), context="bootstrap_ci.metrics"
    )
    for metric_id in AGGREGATE_METRIC_IDS:
        _assert_metric_equal(
            _metric_entry_point_estimate(
                ladder_metrics,
                metric_id=metric_id,
                context=f"metric_ladder_summary.metrics.{metric_id}",
            ),
            point_estimates[metric_id],
            context=f"metric_ladder_summary.metrics.{metric_id}.point_estimate",
        )
        bootstrap_metric = _require_mapping(
            bootstrap_metrics.get(metric_id),
            context=f"bootstrap_ci.metrics.{metric_id}",
        )
        _assert_metric_equal(
            _coerce_optional_float(
                bootstrap_metric.get("point_estimate"),
                context=f"bootstrap_ci.metrics.{metric_id}.point_estimate",
            ),
            point_estimates[metric_id],
            context=f"bootstrap_ci.metrics.{metric_id}.point_estimate",
        )
    if (
        _coerce_int(bootstrap_ci.get("sample_size"), context="bootstrap_ci.sample_size")
        != total_episodes
    ):
        raise AggregationValidationError(
            "bootstrap_ci.sample_size drifted from per_episode_trace"
        )
    summary_primary_metric_id = str(summary.get("primary_metric_id"))
    ladder_primary_metric_id = str(metric_ladder_summary.get("primary_metric_id"))
    if summary_primary_metric_id != ladder_primary_metric_id:
        raise AggregationValidationError(
            "summary.primary_metric_id drifted from metric_ladder_summary.primary_metric_id"
        )
    summary_scope_audit = _require_mapping(
        summary.get("scope_audit"), context="summary.scope_audit"
    )
    if (
        _coerce_int(
            summary_scope_audit.get("observed_episode_count"),
            context="summary.scope_audit.observed_episode_count",
        )
        != total_episodes
    ):
        raise AggregationValidationError(
            "summary.scope_audit.observed_episode_count drifted from per_episode_trace"
        )
    if (
        _coerce_int(
            summary_scope_audit.get("success_count"),
            context="summary.scope_audit.success_count",
        )
        != successful_episodes
    ):
        raise AggregationValidationError(
            "summary.scope_audit.success_count drifted from per_episode_trace"
        )
    if (
        _coerce_int(
            summary_scope_audit.get("failure_count"),
            context="summary.scope_audit.failure_count",
        )
        != total_episodes - successful_episodes
    ):
        raise AggregationValidationError(
            "summary.scope_audit.failure_count drifted from per_episode_trace"
        )
    if (
        _coerce_int(
            summary_scope_audit.get("timeout_count"),
            context="summary.scope_audit.timeout_count",
        )
        != timeout_count
    ):
        raise AggregationValidationError(
            "summary.scope_audit.timeout_count drifted from per_episode_trace"
        )


def _aligned_trace_rows_by_variant(
    trace_rows_by_variant: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, list[Mapping[str, object]]]:
    aligned: dict[str, list[Mapping[str, object]]] = {}
    expected_keys: tuple[tuple[int, int, int], ...] | None = None
    for variant, rows in trace_rows_by_variant.items():
        sorted_rows = sorted(
            rows,
            key=lambda row: _episode_key(
                row, context=f"trace_rows_by_variant.{variant}"
            ),
        )
        keys = tuple(
            _episode_key(row, context=f"trace_rows_by_variant.{variant}[{index}]")
            for index, row in enumerate(sorted_rows)
        )
        if len(keys) != len(set(keys)):
            raise AggregationValidationError(
                f"trace_rows_by_variant.{variant} contains duplicate episodes"
            )
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise AggregationValidationError(
                f"trace_rows_by_variant.{variant} episode scope drifted from peer variants"
            )
        aligned[variant] = sorted_rows
    return aligned


def _paired_metric_delta_samples(
    *,
    lhs_rows: Sequence[Mapping[str, object]],
    rhs_rows: Sequence[Mapping[str, object]],
    deterministic_seed_material: str,
    bootstrap_iterations: int,
) -> dict[str, list[float]]:
    sample_size = len(lhs_rows)
    rng = random.Random(int(_sha256_text(deterministic_seed_material)[:16], 16))
    samples_by_metric: dict[str, list[float]] = {
        metric_id: [] for metric_id in AGGREGATE_METRIC_IDS
    }
    for _ in range(bootstrap_iterations):
        sample_indices = [rng.randrange(sample_size) for _index in range(sample_size)]
        lhs_sample = [lhs_rows[index] for index in sample_indices]
        rhs_sample = [rhs_rows[index] for index in sample_indices]
        lhs_metrics = metric_point_estimates_from_trace_rows_v21(lhs_sample)
        rhs_metrics = metric_point_estimates_from_trace_rows_v21(rhs_sample)
        for metric_id in AGGREGATE_METRIC_IDS:
            lhs_value = lhs_metrics[metric_id]
            rhs_value = rhs_metrics[metric_id]
            if lhs_value is None or rhs_value is None:
                continue
            samples_by_metric[metric_id].append(float(lhs_value) - float(rhs_value))
    return samples_by_metric


def build_pairwise_delta_payload_v21(
    *,
    trace_rows_by_variant: Mapping[str, Sequence[Mapping[str, object]]],
    primary_metric_id: str,
    deterministic_seed_material: str,
    evaluation_tier: str,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, object]:
    if primary_metric_id not in ALLOWED_PRIMARY_METRIC_IDS:
        raise AggregationValidationError(
            f"primary_metric_id must be one of {ALLOWED_PRIMARY_METRIC_IDS!r}"
        )
    aligned_rows = _aligned_trace_rows_by_variant(trace_rows_by_variant)
    required_variants = set(VARIANT_CODE_TO_ID.values())
    missing_variants = sorted(required_variants.difference(aligned_rows))
    if missing_variants:
        raise AggregationValidationError(
            f"pairwise delta missing required variants: {missing_variants!r}"
        )
    variant_metrics = {
        variant: metric_point_estimates_from_trace_rows_v21(rows)
        for variant, rows in aligned_rows.items()
    }
    pairs: dict[str, object] = {}
    for pair_label in REQUIRED_PAIR_LABELS:
        lhs_code, rhs_code = PAIR_LABEL_TO_CODES[pair_label]
        lhs_variant = VARIANT_CODE_TO_ID[lhs_code]
        rhs_variant = VARIANT_CODE_TO_ID[rhs_code]
        lhs_rows = aligned_rows[lhs_variant]
        rhs_rows = aligned_rows[rhs_variant]
        delta_samples = _paired_metric_delta_samples(
            lhs_rows=lhs_rows,
            rhs_rows=rhs_rows,
            deterministic_seed_material=(
                f"{deterministic_seed_material}:{pair_label}:{evaluation_tier}"
            ),
            bootstrap_iterations=bootstrap_iterations,
        )
        metrics_payload: dict[str, object] = {}
        for metric_id in AGGREGATE_METRIC_IDS:
            lhs_value = variant_metrics[lhs_variant][metric_id]
            rhs_value = variant_metrics[rhs_variant][metric_id]
            delta = (
                None
                if lhs_value is None or rhs_value is None
                else float(lhs_value) - float(rhs_value)
            )
            values = delta_samples[metric_id]
            ci95: dict[str, float | None]
            if values:
                ci_lower, ci_upper = _quantile_bounds(
                    values, confidence_level=confidence_level
                )
                ci95 = {"lower": float(ci_lower), "upper": float(ci_upper)}
                ci_non_degenerate = not math.isclose(
                    ci_lower, ci_upper, rel_tol=1e-12, abs_tol=1e-12
                )
            else:
                ci95 = {"lower": None, "upper": None}
                ci_non_degenerate = False
            metrics_payload[metric_id] = {
                "lhs_point_estimate": lhs_value,
                "rhs_point_estimate": rhs_value,
                "delta": delta,
                "ci95": ci95,
                "ci_non_degenerate": ci_non_degenerate,
            }
        pairs[pair_label] = {
            "lhs_variant": lhs_variant,
            "rhs_variant": rhs_variant,
            "sample_size": len(lhs_rows),
            "metrics": metrics_payload,
        }
    return {
        "schema_version": PAIRWISE_DELTA_SCHEMA_VERSION,
        "evaluation_tier": evaluation_tier,
        "primary_metric_id": primary_metric_id,
        "pair_labels": list(REQUIRED_PAIR_LABELS),
        "variant_code_to_id": dict(VARIANT_CODE_TO_ID),
        "bootstrap_iterations": bootstrap_iterations,
        "confidence_level": confidence_level,
        "deterministic_seed_material": deterministic_seed_material,
        "pairs": pairs,
    }


def assert_pairwise_delta_conservation_v21(
    *,
    trace_rows_by_variant: Mapping[str, Sequence[Mapping[str, object]]],
    pairwise_delta: Mapping[str, object],
) -> None:
    aligned_rows = _aligned_trace_rows_by_variant(trace_rows_by_variant)
    pair_labels = _require_sequence(
        pairwise_delta.get("pair_labels"), context="pairwise_delta.pair_labels"
    )
    observed_labels = tuple(str(item) for item in pair_labels)
    for required_label in REQUIRED_PAIR_LABELS:
        if required_label not in observed_labels:
            raise AggregationValidationError(
                f"pairwise_delta missing required pair {required_label}"
            )
    pairs = _require_mapping(
        pairwise_delta.get("pairs"), context="pairwise_delta.pairs"
    )
    variant_metrics = {
        variant: metric_point_estimates_from_trace_rows_v21(rows)
        for variant, rows in aligned_rows.items()
    }
    for pair_label in REQUIRED_PAIR_LABELS:
        pair_payload = _require_mapping(
            pairs.get(pair_label), context=f"pairwise_delta.pairs.{pair_label}"
        )
        lhs_variant = str(pair_payload.get("lhs_variant"))
        rhs_variant = str(pair_payload.get("rhs_variant"))
        lhs_rows = aligned_rows[lhs_variant]
        if rhs_variant not in aligned_rows:
            raise AggregationValidationError(
                f"pairwise_delta.pairs.{pair_label}.rhs_variant missing from trace_rows_by_variant"
            )
        rhs_rows = aligned_rows[rhs_variant]
        if _coerce_int(
            pair_payload.get("sample_size"),
            context=f"pairwise_delta.pairs.{pair_label}.sample_size",
        ) != len(lhs_rows):
            raise AggregationValidationError(
                f"pairwise_delta.pairs.{pair_label}.sample_size drifted from per_episode_trace"
            )
        if len(lhs_rows) != len(rhs_rows):
            raise AggregationValidationError(
                f"pairwise_delta.pairs.{pair_label} trace rows are not aligned"
            )
        metrics_payload = _require_mapping(
            pair_payload.get("metrics"),
            context=f"pairwise_delta.pairs.{pair_label}.metrics",
        )
        for metric_id in AGGREGATE_METRIC_IDS:
            metric_payload = _require_mapping(
                metrics_payload.get(metric_id),
                context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}",
            )
            lhs_value = variant_metrics[lhs_variant][metric_id]
            rhs_value = variant_metrics[rhs_variant][metric_id]
            expected_delta = (
                None
                if lhs_value is None or rhs_value is None
                else float(lhs_value) - float(rhs_value)
            )
            _assert_metric_equal(
                _coerce_optional_float(
                    metric_payload.get("lhs_point_estimate"),
                    context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}.lhs_point_estimate",
                ),
                lhs_value,
                context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}.lhs_point_estimate",
            )
            _assert_metric_equal(
                _coerce_optional_float(
                    metric_payload.get("rhs_point_estimate"),
                    context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}.rhs_point_estimate",
                ),
                rhs_value,
                context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}.rhs_point_estimate",
            )
            _assert_metric_equal(
                _coerce_optional_float(
                    metric_payload.get("delta"),
                    context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}.delta",
                ),
                expected_delta,
                context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}.delta",
            )


def _require_authority_mode(authority_mode: str) -> str:
    normalized = authority_mode.strip().lower()
    if normalized not in EXPECTED_AUTHORITY_ID_BY_MODE:
        raise AggregationValidationError(
            f"authority_mode must be one of {tuple(EXPECTED_AUTHORITY_ID_BY_MODE)!r}"
        )
    return normalized


def _artifact_ref(*, label: str, path: str | None, kind: str) -> dict[str, object]:
    return {"label": label, "path": path, "kind": kind}


def _bundle_authority_dir(bundle: Mapping[str, object]) -> str | None:
    raw = bundle.get("authority_dir")
    text = str(raw).strip() if raw is not None else ""
    return text or None


def _bundle_file_path(
    bundle: Mapping[str, object], file_name: str, *, fallback_key: str
) -> str | None:
    authority_dir = _bundle_authority_dir(bundle)
    if authority_dir:
        return f"{authority_dir}/{file_name}"
    raw = bundle.get(fallback_key)
    text = str(raw).strip() if raw is not None else ""
    return text or None


def _bundle_variant_id(
    bundle: Mapping[str, object], *, expected_variant_id: str, context: str
) -> str:
    candidates = (
        bundle.get("variant"),
        _require_mapping(bundle.get("summary", {}), context=f"{context}.summary").get(
            "variant"
        ),
        _require_mapping(
            bundle.get("metric_ladder_summary", {}),
            context=f"{context}.metric_ladder_summary",
        ).get("variant"),
    )
    for raw in candidates:
        text = str(raw).strip() if raw is not None else ""
        if text:
            return text
    return expected_variant_id


def _selection_variant_bundles_by_id(
    variant_authorities: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    bundles_by_id: dict[str, Mapping[str, object]] = {}
    for variant_code, variant_id in VARIANT_CODE_TO_ID.items():
        bundle = variant_authorities.get(variant_code)
        if bundle is None:
            continue
        metric_ladder_summary = bundle.get("metric_ladder_summary")
        bootstrap_ci = bundle.get("bootstrap_ci")
        if not isinstance(metric_ladder_summary, Mapping) or not isinstance(
            bootstrap_ci, Mapping
        ):
            continue
        bundles_by_id[variant_id] = {
            "metric_ladder_summary": metric_ladder_summary,
            "bootstrap_ci": bootstrap_ci,
        }
    return bundles_by_id


def _selected_metric_point_estimate(
    bundle: Mapping[str, object], *, metric_id: str, context: str
) -> float | None:
    if metric_id not in AGGREGATE_METRIC_IDS:
        return None
    metric_ladder_summary = _require_mapping(
        bundle.get("metric_ladder_summary"), context=f"{context}.metric_ladder_summary"
    )
    metrics = _require_mapping(
        metric_ladder_summary.get("metrics"),
        context=f"{context}.metric_ladder_summary.metrics",
    )
    return _metric_entry_point_estimate(
        metrics,
        metric_id=metric_id,
        context=f"{context}.metric_ladder_summary.metrics.{metric_id}",
    )


def _bundle_metric_point_estimate_or_none(
    bundle: Mapping[str, object], *, metric_id: str, context: str
) -> float | None:
    metric_ladder_summary = bundle.get("metric_ladder_summary")
    if not isinstance(metric_ladder_summary, Mapping):
        return None
    metrics = metric_ladder_summary.get("metrics")
    if not isinstance(metrics, Mapping) or metric_id not in metrics:
        return None
    return _selected_metric_point_estimate(bundle, metric_id=metric_id, context=context)


def _bundle_ci_non_degenerate_or_none(
    bundle: Mapping[str, object], *, metric_id: str, context: str
) -> bool | None:
    bootstrap_ci = bundle.get("bootstrap_ci")
    if not isinstance(bootstrap_ci, Mapping):
        return None
    metrics = bootstrap_ci.get("metrics")
    if not isinstance(metrics, Mapping) or metric_id not in metrics:
        return None
    metric_entry = _bootstrap_metric_entry(
        bootstrap_ci,
        metric_id=metric_id,
        context=f"{context}.bootstrap_ci",
    )
    return _ci_is_non_degenerate(
        metric_entry,
        context=f"{context}.bootstrap_ci.metrics.{metric_id}",
    )


def _trace_completeness_for_bundle(
    bundle: Mapping[str, object], *, context: str
) -> dict[str, object]:
    summary_present = isinstance(bundle.get("summary"), Mapping)
    trace_rows_raw = bundle.get("per_episode_trace")
    if trace_rows_raw is None:
        return {
            "status": "summary_only" if summary_present else "missing_trace",
            "row_count": 0,
            "missing_fields": [],
            "invalid_row_indices": [],
        }
    if isinstance(trace_rows_raw, (str, bytes)) or not isinstance(
        trace_rows_raw, Sequence
    ):
        return {
            "status": "invalid_trace",
            "row_count": 0,
            "missing_fields": ["per_episode_trace_not_sequence"],
            "invalid_row_indices": [],
        }
    missing_fields: set[str] = set()
    invalid_row_indices: list[int] = []
    for index, row in enumerate(trace_rows_raw):
        if not isinstance(row, Mapping):
            invalid_row_indices.append(index)
            continue
        for required_field in ("first_success_step", "executed_steps", "timeout_flag"):
            if required_field not in row:
                missing_fields.add(required_field)
    if invalid_row_indices or missing_fields:
        return {
            "status": "invalid_trace",
            "row_count": len(trace_rows_raw),
            "missing_fields": sorted(missing_fields),
            "invalid_row_indices": invalid_row_indices,
        }
    return {
        "status": "complete",
        "row_count": len(trace_rows_raw),
        "missing_fields": [],
        "invalid_row_indices": [],
    }


def _difficulty_sort_key_from_seed_record(
    record: Mapping[str, object], *, context: str
) -> tuple[float, float, float, float, int]:
    ranking_key = _require_mapping(
        record.get("ranking_key"), context=f"{context}.ranking_key"
    )
    success_50 = _coerce_float(
        ranking_key.get("success_rate@0.50_budget"),
        context=f"{context}.ranking_key.success_rate@0.50_budget",
    )
    success_75 = _coerce_float(
        ranking_key.get("success_rate@0.75_budget"),
        context=f"{context}.ranking_key.success_rate@0.75_budget",
    )
    median_or_one = _coerce_float(
        ranking_key.get("median_first_success_step_fraction_null_as_one"),
        context=(
            f"{context}.ranking_key.median_first_success_step_fraction_null_as_one"
        ),
    )
    timeout_rate = _coerce_float(
        ranking_key.get("timeout_rate"),
        context=f"{context}.ranking_key.timeout_rate",
    )
    seed = _coerce_int(ranking_key.get("seed"), context=f"{context}.ranking_key.seed")
    return (success_50, success_75, -median_or_one, -timeout_rate, seed)


def _task_seed_manifest_rows(
    task_seed_manifests: Mapping[int, tuple[int, ...]],
) -> dict[str, list[int]]:
    return {
        str(task_id): list(task_seed_manifest)
        for task_id, task_seed_manifest in sorted(task_seed_manifests.items())
    }


def _shared_seed_manifest_or_none(
    task_seed_manifests: Mapping[int, tuple[int, ...]],
) -> list[int] | None:
    manifests = list(task_seed_manifests.values())
    if not manifests:
        return None
    shared_seed_manifest = manifests[0]
    if any(
        task_seed_manifest != shared_seed_manifest
        for task_seed_manifest in manifests[1:]
    ):
        return None
    return list(shared_seed_manifest)


def _selection_seed_manifests_from_eval_manifest(
    eval_manifest: Mapping[str, object], *, context: str
) -> dict[int, tuple[int, ...]]:
    task_ids = _coerce_int_sequence(
        eval_manifest.get("task_ids"), context=f"{context}.task_ids"
    )
    raw_per_task_seed_manifest = eval_manifest.get("per_task_seed_manifest")
    if raw_per_task_seed_manifest is not None:
        per_task_seed_manifest = _require_mapping(
            raw_per_task_seed_manifest,
            context=f"{context}.per_task_seed_manifest",
        )
        normalized: dict[int, tuple[int, ...]] = {}
        for raw_task_id, raw_seed_manifest in per_task_seed_manifest.items():
            task_id = _coerce_int(
                raw_task_id, context=f"{context}.per_task_seed_manifest task_id"
            )
            normalized[task_id] = tuple(
                sorted(
                    _coerce_int(
                        value,
                        context=f"{context}.per_task_seed_manifest[{task_id}]",
                    )
                    for value in _require_sequence(
                        raw_seed_manifest,
                        context=f"{context}.per_task_seed_manifest[{task_id}]",
                    )
                )
            )
        actual_task_ids = tuple(sorted(normalized))
        expected_task_ids = tuple(sorted(task_ids))
        if actual_task_ids != expected_task_ids:
            raise AggregationValidationError(
                f"{context}.per_task_seed_manifest task coverage mismatch; expected {expected_task_ids!r}, got {actual_task_ids!r}"
            )
        return dict(sorted(normalized.items()))
    shared_seed_manifest = tuple(
        sorted(
            _coerce_int(value, context=f"{context}.seed_manifest")
            for value in _require_sequence(
                eval_manifest.get("seed_manifest"), context=f"{context}.seed_manifest"
            )
        )
    )
    return {task_id: shared_seed_manifest for task_id in sorted(task_ids)}


def _resolve_selection_source_path(selection_source: str) -> Path:
    selection_source_path = Path(selection_source)
    if not selection_source_path.is_absolute():
        selection_source_path = REPO_ROOT / selection_source_path
    return selection_source_path.resolve()


def _load_selection_source_summary(selection_source: str) -> Mapping[str, object]:
    selection_source_path = _resolve_selection_source_path(selection_source)
    try:
        payload = cast(
            object, json.loads(selection_source_path.read_text(encoding="utf-8"))
        )
    except FileNotFoundError as exc:
        raise AggregationValidationError(
            f"selection_source summary not found: {selection_source_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise AggregationValidationError(
            f"selection_source summary is not valid JSON: {selection_source_path}"
        ) from exc
    return _require_mapping(
        payload, context=f"selection_source_summary[{selection_source_path}]"
    )


def _expected_selected_seed_manifests_from_stock_scan_summary(
    summary: Mapping[str, object], *, context: str
) -> dict[int, tuple[int, ...]]:
    schema_version = str(summary.get("schema_version", "")).strip()
    if schema_version != STOCK_SCAN_SUMMARY_SCHEMA_VERSION:
        raise AggregationValidationError(
            f"{context}.schema_version must be {STOCK_SCAN_SUMMARY_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    summary_task_ids = tuple(
        sorted(
            _coerce_int_sequence(summary.get("task_ids"), context=f"{context}.task_ids")
        )
    )
    seed_records = _require_sequence(
        summary.get("seed_records"), context=f"{context}.seed_records"
    )
    seed_records_by_task: dict[int, list[Mapping[str, object]]] = {
        task_id: [] for task_id in summary_task_ids
    }
    for index, raw_record in enumerate(seed_records):
        record = _require_mapping(
            raw_record, context=f"{context}.seed_records[{index}]"
        )
        task_id = _coerce_int(
            record.get("task_id"), context=f"{context}.seed_records[{index}].task_id"
        )
        if task_id not in seed_records_by_task:
            raise AggregationValidationError(
                f"{context}.seed_records[{index}].task_id {task_id!r} is outside summary.task_ids {summary_task_ids!r}"
            )
        seed_records_by_task[task_id].append(record)
    expected_selected_seed_manifests: dict[int, tuple[int, ...]] = {}
    for task_id in summary_task_ids:
        ranked_seed_records = sorted(
            seed_records_by_task[task_id],
            key=lambda record: _difficulty_sort_key_from_seed_record(
                record, context=f"{context}.seed_records[task_id={task_id}]"
            ),
        )
        if len(ranked_seed_records) < TOP_K_HARDEST_SEEDS:
            raise AggregationValidationError(
                f"{context}.seed_records for task_id={task_id} must contain at least {TOP_K_HARDEST_SEEDS} seeds"
            )
        expected_selected_seed_manifests[task_id] = tuple(
            sorted(
                _coerce_int(
                    record.get("seed"),
                    context=f"{context}.selected_top{TOP_K_HARDEST_SEEDS}[task_id={task_id}].seed",
                )
                for record in ranked_seed_records[:TOP_K_HARDEST_SEEDS]
            )
        )
    return expected_selected_seed_manifests


def _build_variant_snapshot(
    *,
    variant_code: str,
    bundle: Mapping[str, object] | None,
    primary_metric_id: str,
) -> dict[str, object]:
    variant_id = VARIANT_CODE_TO_ID[variant_code]
    if bundle is None:
        return {
            "variant_code": variant_code,
            "variant": variant_id,
            "authority_dir": None,
            "selected_metric_point_estimate": None,
            "compatibility_metric_point_estimate": None,
            "throughput_ci_non_degenerate": None,
            "trace_completeness": {
                "status": "missing_bundle",
                "row_count": 0,
                "missing_fields": [],
                "invalid_row_indices": [],
            },
            "selection_policy": None,
            "selection_source": None,
            "selection_source_hash": None,
        }
    eval_manifest = _require_mapping(
        bundle.get("eval_manifest", {}),
        context=f"variant_authorities.{variant_code}.eval_manifest",
    )
    return {
        "variant_code": variant_code,
        "variant": _bundle_variant_id(
            bundle,
            expected_variant_id=variant_id,
            context=f"variant_authorities.{variant_code}",
        ),
        "authority_dir": _bundle_authority_dir(bundle),
        "selected_metric_point_estimate": _bundle_metric_point_estimate_or_none(
            bundle,
            metric_id=primary_metric_id,
            context=f"variant_authorities.{variant_code}",
        ),
        "compatibility_metric_point_estimate": _bundle_metric_point_estimate_or_none(
            bundle,
            metric_id="success_rate@1.00_budget",
            context=f"variant_authorities.{variant_code}",
        ),
        "throughput_ci_non_degenerate": _bundle_ci_non_degenerate_or_none(
            bundle,
            metric_id="throughput_like_score",
            context=f"variant_authorities.{variant_code}",
        ),
        "trace_completeness": _trace_completeness_for_bundle(
            bundle, context=f"variant_authorities.{variant_code}"
        ),
        "selection_policy": eval_manifest.get("selection_policy"),
        "selection_source": eval_manifest.get("selection_source"),
        "selection_source_hash": eval_manifest.get("selection_source_hash"),
    }


def _build_pairwise_delta_or_error(
    *,
    variant_authorities: Mapping[str, Mapping[str, object]],
    primary_metric_id: str,
    authority_mode: str,
) -> tuple[dict[str, object] | None, str | None]:
    trace_rows_by_variant: dict[str, Sequence[Mapping[str, object]]] = {}
    for variant_code, variant_id in VARIANT_CODE_TO_ID.items():
        bundle = variant_authorities.get(variant_code)
        if bundle is None:
            return None, f"missing required bundle for variant {variant_code}"
        trace_rows = bundle.get("per_episode_trace")
        if isinstance(trace_rows, (str, bytes)) or not isinstance(trace_rows, Sequence):
            return (
                None,
                f"variant {variant_code} per_episode_trace missing or not sequence",
            )
        trace_rows_by_variant[variant_id] = cast(
            Sequence[Mapping[str, object]], trace_rows
        )
    try:
        return (
            build_pairwise_delta_payload_v21(
                trace_rows_by_variant=trace_rows_by_variant,
                primary_metric_id=primary_metric_id,
                deterministic_seed_material=f"paired_summary:{authority_mode}:{primary_metric_id}",
                evaluation_tier=authority_mode,
            ),
            None,
        )
    except AggregationValidationError as exc:
        return None, str(exc)


def build_libero_paired_summary_v21(
    *,
    variant_authorities: Mapping[str, Mapping[str, object]],
    authority_mode: str,
    selection_variant_authorities: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    normalized_mode = _require_authority_mode(authority_mode)
    if normalized_mode == "strong":
        selection_bundles = selection_variant_authorities or {}
    else:
        selection_bundles = selection_variant_authorities or variant_authorities
    primary_metric_selection = select_primary_metric_id_v21(
        lite_variant_bundles=_selection_variant_bundles_by_id(selection_bundles)
    )
    primary_metric_id = str(primary_metric_selection["primary_metric_id"])
    pairwise_delta, pairwise_delta_error = _build_pairwise_delta_or_error(
        variant_authorities=variant_authorities,
        primary_metric_id=primary_metric_id,
        authority_mode=normalized_mode,
    )
    variants = {
        variant_code: _build_variant_snapshot(
            variant_code=variant_code,
            bundle=variant_authorities.get(variant_code),
            primary_metric_id=primary_metric_id,
        )
        for variant_code in VARIANT_CODE_TO_ID
    }
    headline_ranking: list[dict[str, object]] = []
    if primary_metric_id in AGGREGATE_METRIC_IDS:
        ranking_rows: list[tuple[float, str]] = []
        for variant_code in HEADLINE_VARIANT_CODES:
            point_estimate = cast(
                float | None, variants[variant_code]["selected_metric_point_estimate"]
            )
            if point_estimate is None:
                continue
            ranking_rows.append((float(point_estimate), variant_code))
        ranking_rows.sort(key=lambda item: (-item[0], item[1]))
        headline_ranking = [
            {
                "rank": index + 1,
                "variant_code": variant_code,
                "variant": VARIANT_CODE_TO_ID[variant_code],
                "point_estimate": point_estimate,
                "metric_id": primary_metric_id,
            }
            for index, (point_estimate, variant_code) in enumerate(ranking_rows)
        ]
    headline_winner = headline_ranking[0] if headline_ranking else None
    return {
        "schema_version": PAIRED_SUMMARY_SCHEMA_VERSION,
        "authority_mode": normalized_mode,
        "primary_metric_id": primary_metric_id,
        "primary_metric_selection": primary_metric_selection,
        "headline_variant_codes": list(HEADLINE_VARIANT_CODES),
        "diagnostic_variant_codes": list(DIAGNOSTIC_VARIANT_CODES),
        "headline_winner": headline_winner,
        "headline_ranking": headline_ranking,
        "pairwise_delta_available": pairwise_delta is not None,
        "pairwise_delta_error": pairwise_delta_error,
        "pairwise_delta": pairwise_delta,
        "variants": variants,
    }


def _gate_payload(
    *,
    gate: str,
    name: str,
    status: str,
    threshold: Mapping[str, object],
    operator: str,
    metric_inputs: Mapping[str, object],
    source_artifacts: Sequence[Mapping[str, object]],
    decision_text: str,
    next_action: str,
) -> dict[str, object]:
    if status not in ALLOWED_GATE_STATUSES:
        raise AggregationValidationError(
            f"gate status must be one of {ALLOWED_GATE_STATUSES!r}, got {status!r}"
        )
    return {
        "gate": gate,
        "name": name,
        "status": status,
        "threshold": dict(threshold),
        "operator": operator,
        "metric_inputs": dict(metric_inputs),
        "source_artifacts": [dict(item) for item in source_artifacts],
        "decision_text": decision_text,
        "next_action": next_action,
    }


def _evaluate_scope_gate(
    *, variant_authorities: Mapping[str, Mapping[str, object]], authority_mode: str
) -> dict[str, object]:
    expected_authority_id = EXPECTED_AUTHORITY_ID_BY_MODE[authority_mode]
    missing_variants = [
        variant_code
        for variant_code in VARIANT_CODE_TO_ID
        if variant_authorities.get(variant_code) is None
    ]
    scope_rows: dict[str, object] = {}
    violations: list[str] = []
    source_artifacts: list[dict[str, object]] = []
    if missing_variants:
        status = "NOT_APPLICABLE"
        decision_text = (
            "required variant bundles are missing, so scope cannot be fully evaluated"
        )
        next_action = "补齐缺失的 A/B/C/X authority bundle 后再生成 v21 reporter。"
    else:
        for variant_code, variant_id in VARIANT_CODE_TO_ID.items():
            bundle = _require_mapping(
                variant_authorities.get(variant_code),
                context=f"variant_authorities.{variant_code}",
            )
            eval_manifest = _require_mapping(
                bundle.get("eval_manifest"),
                context=f"variant_authorities.{variant_code}.eval_manifest",
            )
            manifest_task_ids = tuple(
                _coerce_int(task_id, context=f"{variant_code}.eval_manifest.task_ids[]")
                for task_id in _require_sequence(
                    eval_manifest.get("task_ids"),
                    context=f"{variant_code}.eval_manifest.task_ids",
                )
            )
            manifest_variant_scope = tuple(
                str(item).strip()
                for item in _require_sequence(
                    eval_manifest.get("variant_scope"),
                    context=f"{variant_code}.eval_manifest.variant_scope",
                )
            )
            observed = {
                "variant": _bundle_variant_id(
                    bundle,
                    expected_variant_id=variant_id,
                    context=f"variant_authorities.{variant_code}",
                ),
                "authority_id": str(eval_manifest.get("authority_id", "")),
                "task_suite_name": str(eval_manifest.get("task_suite_name", "")),
                "task_ids": list(manifest_task_ids),
                "variant_scope": list(manifest_variant_scope),
            }
            scope_rows[variant_code] = observed
            source_artifacts.append(
                _artifact_ref(
                    label=f"{variant_code}.eval_manifest",
                    path=_bundle_file_path(
                        bundle,
                        "eval_manifest.json",
                        fallback_key="eval_manifest_path",
                    ),
                    kind="eval_manifest",
                )
            )
            if observed["variant"] != variant_id:
                violations.append(
                    f"{variant_code}.variant expected {variant_id} got {observed['variant']}"
                )
            if observed["authority_id"] != expected_authority_id:
                violations.append(
                    f"{variant_code}.authority_id expected {expected_authority_id} got {observed['authority_id']}"
                )
            if observed["task_suite_name"] != EXPECTED_TASK_SUITE_NAME:
                violations.append(
                    f"{variant_code}.task_suite_name expected {EXPECTED_TASK_SUITE_NAME} got {observed['task_suite_name']}"
                )
            if manifest_task_ids != EXPECTED_TASK_IDS:
                violations.append(
                    f"{variant_code}.task_ids expected {EXPECTED_TASK_IDS!r} got {manifest_task_ids!r}"
                )
            if tuple(manifest_variant_scope) != tuple(VARIANT_CODE_TO_ID.values()):
                violations.append(
                    f"{variant_code}.variant_scope expected {tuple(VARIANT_CODE_TO_ID.values())!r} got {tuple(manifest_variant_scope)!r}"
                )
        status = "FAIL" if violations else "PASS"
        if status == "PASS":
            decision_text = "all variant bundles stay inside the frozen v21 scope"
            next_action = "继续消费当前 v21 authority bundles。"
        else:
            decision_text = "scope drift detected in one or more v21 bundles"
            next_action = (
                "先修复 authority_id/task_ids/variant_scope 越界，再重新生成 reporter。"
            )
    return _gate_payload(
        gate="H0",
        name="scope_gate",
        status=status,
        threshold={
            "authority_id": expected_authority_id,
            "task_suite_name": EXPECTED_TASK_SUITE_NAME,
            "task_ids": list(EXPECTED_TASK_IDS),
            "allowed_variants": list(VARIANT_CODE_TO_ID.values()),
        },
        operator="all required bundles must match the frozen v21 scope",
        metric_inputs={
            "missing_variants": missing_variants,
            "observed_scope": scope_rows,
            "violations": violations,
        },
        source_artifacts=source_artifacts,
        decision_text=decision_text,
        next_action=next_action,
    )


def _evaluate_trace_gate(
    *, variant_authorities: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    per_variant: dict[str, object] = {}
    invalid_variants: list[str] = []
    summary_only_variants: list[str] = []
    missing_trace_variants: list[str] = []
    source_artifacts: list[dict[str, object]] = []
    for variant_code in VARIANT_CODE_TO_ID:
        bundle = variant_authorities.get(variant_code)
        if bundle is None:
            per_variant[variant_code] = {
                "status": "missing_bundle",
                "row_count": 0,
                "missing_fields": [],
                "invalid_row_indices": [],
            }
            missing_trace_variants.append(variant_code)
            continue
        completeness = _trace_completeness_for_bundle(
            bundle, context=f"variant_authorities.{variant_code}"
        )
        per_variant[variant_code] = completeness
        source_artifacts.extend(
            [
                _artifact_ref(
                    label=f"{variant_code}.per_episode_trace",
                    path=_bundle_file_path(
                        bundle,
                        "per_episode_trace.jsonl",
                        fallback_key="per_episode_trace_path",
                    ),
                    kind="per_episode_trace",
                ),
                _artifact_ref(
                    label=f"{variant_code}.summary",
                    path=_bundle_file_path(
                        bundle,
                        "summary.json",
                        fallback_key="summary_path",
                    ),
                    kind="summary",
                ),
            ]
        )
        completeness_status = str(completeness["status"])
        if completeness_status == "invalid_trace":
            invalid_variants.append(variant_code)
        elif completeness_status == "summary_only":
            summary_only_variants.append(variant_code)
        elif completeness_status in {"missing_trace", "missing_bundle"}:
            missing_trace_variants.append(variant_code)
    if invalid_variants:
        status = "FAIL"
        decision_text = (
            "one or more trace bundles are present but missing required H1 fields"
        )
        next_action = "补齐 first_success_step/executed_steps/timeout_flag 后重新 materialize v21 authority bundle。"
    elif summary_only_variants:
        status = "HOLD"
        decision_text = "summary-only legacy bundle detected; old summary cannot satisfy v21 trace completeness"
        next_action = (
            "不要把 summary-only 旧结果当成 v21 authority；需要补齐 fresh trace。"
        )
    elif missing_trace_variants:
        status = "NOT_APPLICABLE"
        decision_text = (
            "required trace artifacts are missing, so downstream gates cannot run"
        )
        next_action = "补齐缺失 variant 的 per_episode_trace.jsonl 后再生成 reporter。"
    else:
        status = "PASS"
        decision_text = "all required v21 trace artifacts include first_success_step/executed_steps/timeout_flag"
        next_action = "继续计算 paired summary 与 H2-H7。"
    return _gate_payload(
        gate="H1",
        name="trace_completeness_gate",
        status=status,
        threshold={
            "required_trace_fields": [
                "first_success_step",
                "executed_steps",
                "timeout_flag",
            ]
        },
        operator="all required variants must provide fresh trace rows with H1 fields",
        metric_inputs={
            "per_variant": per_variant,
            "invalid_variants": invalid_variants,
            "summary_only_variants": summary_only_variants,
            "missing_trace_variants": missing_trace_variants,
        },
        source_artifacts=source_artifacts,
        decision_text=decision_text,
        next_action=next_action,
    )


def _evaluate_selection_purity_gate(
    *, variant_authorities: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    missing_variants: list[str] = []
    per_variant: dict[str, object] = {}
    selection_source_hashes: list[str] = []
    selection_sources: list[str] = []
    violations: list[str] = []
    source_artifacts: list[dict[str, object]] = []
    expected_seed_manifests_by_source: dict[str, dict[int, tuple[int, ...]]] = {}
    for variant_code in VARIANT_CODE_TO_ID:
        bundle = variant_authorities.get(variant_code)
        if bundle is None:
            missing_variants.append(variant_code)
            continue
        eval_manifest = _require_mapping(
            bundle.get("eval_manifest", {}),
            context=f"variant_authorities.{variant_code}.eval_manifest",
        )
        selection_policy = str(eval_manifest.get("selection_policy", "")).strip()
        selection_source = str(eval_manifest.get("selection_source", "")).strip()
        selection_source_hash = str(
            eval_manifest.get("selection_source_hash", "")
        ).strip()
        actual_seed_manifests = _selection_seed_manifests_from_eval_manifest(
            eval_manifest, context=f"variant_authorities.{variant_code}.eval_manifest"
        )
        expected_seed_manifests: dict[int, tuple[int, ...]] | None = None
        content_error: str | None = None
        if selection_source:
            try:
                expected_seed_manifests = expected_seed_manifests_by_source.get(
                    selection_source
                )
                if expected_seed_manifests is None:
                    expected_seed_manifests = _expected_selected_seed_manifests_from_stock_scan_summary(
                        _load_selection_source_summary(selection_source),
                        context=(
                            f"variant_authorities.{variant_code}.selection_source_summary"
                        ),
                    )
                    expected_seed_manifests_by_source[selection_source] = (
                        expected_seed_manifests
                    )
            except AggregationValidationError as exc:
                content_error = str(exc)
                violations.append(
                    f"{variant_code}.selection_source invalid: {content_error}"
                )
        selection_content_matches_expected = (
            expected_seed_manifests is not None
            and actual_seed_manifests == expected_seed_manifests
        )
        per_variant[variant_code] = {
            "selection_policy": selection_policy or None,
            "selection_source": selection_source or None,
            "selection_source_hash": selection_source_hash or None,
            "actual_selected_seed_manifest": _shared_seed_manifest_or_none(
                actual_seed_manifests
            ),
            "actual_per_task_selected_seed_manifest": _task_seed_manifest_rows(
                actual_seed_manifests
            ),
            "expected_selected_seed_manifest": (
                _shared_seed_manifest_or_none(expected_seed_manifests)
                if expected_seed_manifests is not None
                else None
            ),
            "expected_per_task_selected_seed_manifest": (
                _task_seed_manifest_rows(expected_seed_manifests)
                if expected_seed_manifests is not None
                else None
            ),
            "selection_content_matches_expected": (
                selection_content_matches_expected
                if expected_seed_manifests is not None
                else None
            ),
            "selection_content_error": content_error,
        }
        source_artifacts.append(
            _artifact_ref(
                label=f"{variant_code}.eval_manifest",
                path=_bundle_file_path(
                    bundle,
                    "eval_manifest.json",
                    fallback_key="eval_manifest_path",
                ),
                kind="eval_manifest",
            )
        )
        if selection_source:
            source_artifacts.append(
                _artifact_ref(
                    label=f"{variant_code}.selection_source_summary",
                    path=str(_resolve_selection_source_path(selection_source)),
                    kind="stock_seed_scan_summary",
                )
            )
        if selection_policy != EXPECTED_SELECTION_POLICY:
            violations.append(
                f"{variant_code}.selection_policy expected {EXPECTED_SELECTION_POLICY} got {selection_policy or '<missing>'}"
            )
        if not selection_source:
            violations.append(f"{variant_code}.selection_source missing")
        if not selection_source_hash:
            violations.append(f"{variant_code}.selection_source_hash missing")
        if selection_source:
            selection_sources.append(selection_source)
        if selection_source_hash:
            selection_source_hashes.append(selection_source_hash)
        if (
            expected_seed_manifests is not None
            and not selection_content_matches_expected
        ):
            violations.append(
                f"{variant_code}.selected_seed_manifest mismatch; expected {_task_seed_manifest_rows(expected_seed_manifests)!r} got {_task_seed_manifest_rows(actual_seed_manifests)!r}"
            )
    unique_sources = sorted(set(selection_sources))
    unique_hashes = sorted(set(selection_source_hashes))
    if len(unique_sources) > 1:
        violations.append(
            f"selection_source mismatch across variants: {unique_sources!r}"
        )
    if len(unique_hashes) > 1:
        violations.append(
            f"selection_source_hash mismatch across variants: {unique_hashes!r}"
        )
    if missing_variants:
        status = "NOT_APPLICABLE"
        decision_text = (
            "selection purity cannot be checked because required bundles are missing"
        )
        next_action = "补齐 A/B/C/X bundles 后再验证 stock-only hard-seed purity。"
    elif violations:
        status = "FAIL"
        decision_text = "hard-seed selection is not stock-only and deterministic across all variants"
        next_action = "重新生成 hard-seed manifest，确保 selection_policy=stock_only_hard_seed_v1、source hash 一致，且实际 seed 内容与 stock scan top6 完全一致。"
    else:
        status = "PASS"
        decision_text = "all bundles inherit the same stock-only hard-seed selection metadata and actual selected seeds"
        next_action = "继续使用当前 hard-seed manifest 进入 paired gates。"
    return _gate_payload(
        gate="H3",
        name="selection_purity_gate",
        status=status,
        threshold={
            "selection_policy": EXPECTED_SELECTION_POLICY,
            "selection_source_hash_must_match": True,
            "selected_seed_content_must_match_stock_scan_top6": True,
        },
        operator=(
            "all variants must share stock_only_hard_seed_v1, identical selection_source_hash, and actual selected seeds equal to the referenced stock-scan top6"
        ),
        metric_inputs={
            "missing_variants": missing_variants,
            "per_variant": per_variant,
            "unique_selection_sources": unique_sources,
            "unique_selection_source_hashes": unique_hashes,
            "violations": violations,
        },
        source_artifacts=source_artifacts,
        decision_text=decision_text,
        next_action=next_action,
    )


def _gate_sources_for_pairwise(
    *,
    pairwise_delta: Mapping[str, object] | None,
    variant_authorities: Mapping[str, Mapping[str, object]],
    variant_codes: Sequence[str],
) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for variant_code in variant_codes:
        bundle = variant_authorities.get(variant_code)
        if bundle is None:
            continue
        sources.extend(
            [
                _artifact_ref(
                    label=f"{variant_code}.metric_ladder_summary",
                    path=_bundle_file_path(
                        bundle,
                        "metric_ladder_summary.json",
                        fallback_key="metric_ladder_summary_path",
                    ),
                    kind="metric_ladder_summary",
                ),
                _artifact_ref(
                    label=f"{variant_code}.bootstrap_ci",
                    path=_bundle_file_path(
                        bundle,
                        "bootstrap_ci.json",
                        fallback_key="bootstrap_ci_path",
                    ),
                    kind="bootstrap_ci",
                ),
            ]
        )
    if pairwise_delta is not None:
        pairwise_path = str(pairwise_delta.get("path", "")).strip() or None
        sources.append(
            _artifact_ref(
                label="pairwise_delta",
                path=pairwise_path,
                kind="pairwise_delta",
            )
        )
    return sources


def _prerequisite_statuses_allow(
    prerequisite_statuses: Sequence[str],
) -> bool:
    return all(status == "PASS" for status in prerequisite_statuses)


def _pair_metric_inputs(
    *,
    pairwise_delta: Mapping[str, object],
    pair_label: str,
    metric_id: str,
) -> dict[str, object]:
    pairs = _require_mapping(
        pairwise_delta.get("pairs"), context="pairwise_delta.pairs"
    )
    pair_payload = _require_mapping(
        pairs.get(pair_label), context=f"pairwise_delta.pairs.{pair_label}"
    )
    metric_payload = _require_mapping(
        _require_mapping(
            pair_payload.get("metrics"),
            context=f"pairwise_delta.pairs.{pair_label}.metrics",
        ).get(metric_id),
        context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}",
    )
    ci95 = _require_mapping(
        metric_payload.get("ci95"),
        context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}.ci95",
    )
    return {
        "delta": _coerce_optional_float(
            metric_payload.get("delta"),
            context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}.delta",
        ),
        "ci95_lower": _coerce_optional_float(
            ci95.get("lower"),
            context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}.ci95.lower",
        ),
        "ci95_upper": _coerce_optional_float(
            ci95.get("upper"),
            context=f"pairwise_delta.pairs.{pair_label}.metrics.{metric_id}.ci95.upper",
        ),
    }


def _evaluate_headroom_gate(
    *,
    paired_summary: Mapping[str, object],
    variant_authorities: Mapping[str, Mapping[str, object]],
    prerequisite_statuses: Sequence[str],
) -> dict[str, object]:
    primary_metric_id = str(paired_summary.get("primary_metric_id", "")).strip()
    if not _prerequisite_statuses_allow(prerequisite_statuses):
        return _gate_payload(
            gate="H2",
            name="headroom_recovery_gate",
            status="NOT_APPLICABLE",
            threshold={
                "headline_complete_ceiling": False,
                "headline_throughput_ci_degenerate": False,
            },
            operator=(
                "requires H0/H1/H3 PASS before judging whether the selected primary metric recovered headroom"
            ),
            metric_inputs={
                "prerequisite_statuses": list(prerequisite_statuses),
                "primary_metric_id": primary_metric_id or None,
            },
            source_artifacts=_gate_sources_for_pairwise(
                pairwise_delta=None,
                variant_authorities=variant_authorities,
                variant_codes=HEADLINE_VARIANT_CODES,
            ),
            decision_text="prerequisite gate missing or failed; headroom recovery cannot be evaluated",
            next_action="先修复 H0/H1/H3，再判断 H2。",
        )
    if primary_metric_id not in AGGREGATE_METRIC_IDS:
        return _gate_payload(
            gate="H2",
            name="headroom_recovery_gate",
            status="NOT_APPLICABLE",
            threshold={
                "headline_complete_ceiling": False,
                "headline_throughput_ci_degenerate": False,
            },
            operator="selected primary metric must be one of the aggregate metrics",
            metric_inputs={"primary_metric_id": primary_metric_id or None},
            source_artifacts=_gate_sources_for_pairwise(
                pairwise_delta=None,
                variant_authorities=variant_authorities,
                variant_codes=HEADLINE_VARIANT_CODES,
            ),
            decision_text="lite selection did not yield a usable primary metric, so H2 is not applicable",
            next_action="保持 strong 执行，但把 H2 标记为 NOT_APPLICABLE。",
        )
    selected_metric_points: dict[str, float | None] = {}
    throughput_flags: dict[str, bool | None] = {}
    for variant_code in HEADLINE_VARIANT_CODES:
        bundle = variant_authorities.get(variant_code)
        if bundle is None:
            return _gate_payload(
                gate="H2",
                name="headroom_recovery_gate",
                status="NOT_APPLICABLE",
                threshold={
                    "headline_complete_ceiling": False,
                    "headline_throughput_ci_degenerate": False,
                },
                operator="all headline bundles must exist",
                metric_inputs={
                    "primary_metric_id": primary_metric_id,
                    "missing_variant": variant_code,
                },
                source_artifacts=_gate_sources_for_pairwise(
                    pairwise_delta=None,
                    variant_authorities=variant_authorities,
                    variant_codes=HEADLINE_VARIANT_CODES,
                ),
                decision_text="headline bundle missing; H2 cannot be evaluated",
                next_action="补齐 headline bundles 后再生成 reporter。",
            )
        selected_metric_points[variant_code] = _bundle_metric_point_estimate_or_none(
            bundle,
            metric_id=primary_metric_id,
            context=f"variant_authorities.{variant_code}",
        )
        throughput_flags[variant_code] = _bundle_ci_non_degenerate_or_none(
            bundle,
            metric_id="throughput_like_score",
            context=f"variant_authorities.{variant_code}",
        )
    if any(value is None for value in selected_metric_points.values()) or any(
        flag is None for flag in throughput_flags.values()
    ):
        status = "NOT_APPLICABLE"
        decision_text = (
            "selected metric or throughput CI payload missing, so H2 cannot be checked"
        )
        next_action = "补齐 metric_ladder_summary/bootstrap_ci 后再判断 headroom。"
    else:
        headline_complete_ceiling = primary_metric_id.startswith(
            "success_rate@"
        ) and all(
            math.isclose(
                cast(float, selected_metric_points[variant_code]),
                1.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for variant_code in HEADLINE_VARIANT_CODES
        )
        headline_throughput_ci_degenerate = all(
            not cast(bool, throughput_flags[variant_code])
            for variant_code in HEADLINE_VARIANT_CODES
        )
        if not headline_complete_ceiling:
            status = "PASS"
            decision_text = "selected primary metric is not fully saturated across headline variants"
            next_action = "继续用当前 selected primary metric 解释 C/B/A 结论。"
        elif headline_throughput_ci_degenerate:
            status = "FAIL"
            decision_text = "EVAL_SLICE_NOT_DECISION_CAPABLE_ON_TASKS_0_1"
            next_action = "发布 strong 结果，但保持 state side 冻结。"
        else:
            status = "HOLD"
            decision_text = "selected metric is still ceiling, but throughput-like CI retains residual informativeness"
            next_action = "把 H2 保持为 HOLD，并在后续结果中标注 residual throughput informativeness。"
    return _gate_payload(
        gate="H2",
        name="headroom_recovery_gate",
        status=status,
        threshold={
            "headline_complete_ceiling": True,
            "headline_throughput_ci_degenerate": True,
        },
        operator=(
            "FAIL only when selected metric remains full ceiling across A/B/C and throughput_like_score CI is degenerate for all headline variants"
        ),
        metric_inputs={
            "prerequisite_statuses": list(prerequisite_statuses),
            "primary_metric_id": primary_metric_id,
            "selected_metric_points": selected_metric_points,
            "headline_throughput_ci_non_degenerate": throughput_flags,
        },
        source_artifacts=_gate_sources_for_pairwise(
            pairwise_delta=None,
            variant_authorities=variant_authorities,
            variant_codes=HEADLINE_VARIANT_CODES,
        ),
        decision_text=decision_text,
        next_action=next_action,
    )


def _evaluate_pairwise_primary_gate(
    *,
    gate: str,
    name: str,
    pair_label: str,
    paired_summary: Mapping[str, object],
    pairwise_delta: Mapping[str, object] | None,
    variant_authorities: Mapping[str, Mapping[str, object]],
    prerequisite_statuses: Sequence[str],
    pass_text: str,
    hold_text: str,
    fail_text: str,
    next_action_pass: str,
    next_action_hold: str,
    next_action_fail: str,
) -> dict[str, object]:
    primary_metric_id = str(paired_summary.get("primary_metric_id", "")).strip()
    source_artifacts = _gate_sources_for_pairwise(
        pairwise_delta=pairwise_delta,
        variant_authorities=variant_authorities,
        variant_codes=(pair_label[0], pair_label[-1]),
    )
    if not _prerequisite_statuses_allow(prerequisite_statuses):
        return _gate_payload(
            gate=gate,
            name=name,
            status="NOT_APPLICABLE",
            threshold={"delta": 0.0, "ci95_lower": 0.0},
            operator="requires H0/H1/H3 PASS",
            metric_inputs={
                "prerequisite_statuses": list(prerequisite_statuses),
                "primary_metric_id": primary_metric_id or None,
            },
            source_artifacts=source_artifacts,
            decision_text="prerequisite gate missing or failed; pairwise gate is not applicable",
            next_action="先修复 H0/H1/H3，再评估 pairwise gate。",
        )
    if primary_metric_id not in AGGREGATE_METRIC_IDS or pairwise_delta is None:
        return _gate_payload(
            gate=gate,
            name=name,
            status="NOT_APPLICABLE",
            threshold={"delta": 0.0, "ci95_lower": 0.0},
            operator="requires selected primary metric and pairwise delta payload",
            metric_inputs={
                "primary_metric_id": primary_metric_id or None,
                "pairwise_delta_available": pairwise_delta is not None,
            },
            source_artifacts=source_artifacts,
            decision_text="selected primary metric or pairwise delta payload missing; gate is not applicable",
            next_action="补齐 pairwise delta 与 primary metric 后再评估。",
        )
    metric_inputs = _pair_metric_inputs(
        pairwise_delta=pairwise_delta,
        pair_label=pair_label,
        metric_id=primary_metric_id,
    )
    delta = cast(float | None, metric_inputs["delta"])
    ci95_lower = cast(float | None, metric_inputs["ci95_lower"])
    if delta is None:
        status = "NOT_APPLICABLE"
        decision_text = "pairwise delta missing for the selected primary metric"
        next_action = "补齐 pairwise delta 结果后重跑 reporter。"
    elif delta > 0.0 and ci95_lower is not None and ci95_lower > 0.0:
        status = "PASS"
        decision_text = pass_text
        next_action = next_action_pass
    elif delta > 0.0:
        status = "HOLD"
        decision_text = hold_text
        next_action = next_action_hold
    else:
        status = "FAIL"
        decision_text = fail_text
        next_action = next_action_fail
    return _gate_payload(
        gate=gate,
        name=name,
        status=status,
        threshold={"delta": 0.0, "ci95_lower": 0.0},
        operator="PASS if delta>0 && ci95.lower>0; HOLD if delta>0 && ci95.lower<=0; FAIL if delta<=0",
        metric_inputs={
            "prerequisite_statuses": list(prerequisite_statuses),
            "primary_metric_id": primary_metric_id,
            **metric_inputs,
        },
        source_artifacts=source_artifacts,
        decision_text=decision_text,
        next_action=next_action,
    )


def _evaluate_viability_gate(
    *,
    pairwise_delta: Mapping[str, object] | None,
    variant_authorities: Mapping[str, Mapping[str, object]],
    prerequisite_statuses: Sequence[str],
) -> dict[str, object]:
    source_artifacts = _gate_sources_for_pairwise(
        pairwise_delta=pairwise_delta,
        variant_authorities=variant_authorities,
        variant_codes=("C", "A"),
    )
    if (
        not _prerequisite_statuses_allow(prerequisite_statuses)
        or pairwise_delta is None
    ):
        return _gate_payload(
            gate="H6",
            name="viability_gate",
            status="NOT_APPLICABLE",
            threshold={"pass_min_delta": -0.025, "fail_max_delta": -0.05},
            operator="requires H0/H1/H3 PASS and pairwise delta payload",
            metric_inputs={
                "prerequisite_statuses": list(prerequisite_statuses),
                "pairwise_delta_available": pairwise_delta is not None,
            },
            source_artifacts=source_artifacts,
            decision_text="compatibility check cannot run because prerequisite payload is missing",
            next_action="先补齐 prerequisite payload，再判断 C-A compatibility。",
        )
    metric_inputs = _pair_metric_inputs(
        pairwise_delta=pairwise_delta,
        pair_label="C-A",
        metric_id="success_rate@1.00_budget",
    )
    delta = cast(float | None, metric_inputs["delta"])
    if delta is None:
        status = "NOT_APPLICABLE"
        decision_text = "compatibility delta missing, so viability is not applicable"
        next_action = "补齐 success_rate@1.00_budget 的 pairwise delta。"
    elif delta >= -0.025:
        status = "PASS"
        decision_text = "compatibility loss stays within the v21 viability tolerance"
        next_action = "继续保留 C 的 deploy-side viability。"
    elif delta > -0.05:
        status = "HOLD"
        decision_text = "compatibility loss is borderline and must stay advisory only"
        next_action = "不要进入 state side；先记录 borderline viability。"
    else:
        status = "FAIL"
        decision_text = "compatibility loss exceeds the v21 viability floor"
        next_action = "保持 state side 冻结，并先解决 C-A compatibility regression。"
    return _gate_payload(
        gate="H6",
        name="viability_gate",
        status=status,
        threshold={"pass_min_delta": -0.025, "fail_max_delta": -0.05},
        operator=(
            "PASS if C-A >= -0.025; HOLD if -0.05 < C-A < -0.025; FAIL if C-A <= -0.05 on success_rate@1.00_budget"
        ),
        metric_inputs={
            "compatibility_metric_id": "success_rate@1.00_budget",
            **metric_inputs,
        },
        source_artifacts=source_artifacts,
        decision_text=decision_text,
        next_action=next_action,
    )


def _evaluate_state_side_gate(
    *, h2_status: str, h4_status: str, h5_status: str, h6_status: str
) -> dict[str, object]:
    upstream = {
        "H2": h2_status,
        "H4": h4_status,
        "H5": h5_status,
        "H6": h6_status,
    }
    if any(status == "NOT_APPLICABLE" for status in upstream.values()):
        status = "NOT_APPLICABLE"
        eligible = False
        decision_text = "state-side eligibility cannot be evaluated until H2/H4/H5/H6 are all machine-checkable"
        next_action = "先补齐上游 gate payload，再判断是否进入 v22 state side。"
    elif all(status == "PASS" for status in upstream.values()):
        status = "PASS"
        eligible = True
        decision_text = "eligible_for_state_side_v22=true"
        next_action = (
            "v21 已满足 state-side entry prerequisites；后续阶段才允许运行 D。"
        )
    elif any(status == "FAIL" for status in upstream.values()):
        status = "FAIL"
        eligible = False
        decision_text = "eligible_for_state_side_v22=false because at least one upstream gate failed"
        next_action = "保持 D_not_executed_in_v21=true，并先修复失败 gate。"
    else:
        status = "HOLD"
        eligible = False
        decision_text = "eligible_for_state_side_v22=false because at least one upstream gate is HOLD"
        next_action = "保持 D_not_executed_in_v21=true，等待上游 HOLD 收敛。"
    return _gate_payload(
        gate="H7",
        name="state_side_entry_gate",
        status=status,
        threshold={"requires_all_pass": ["H2", "H4", "H5", "H6"]},
        operator="eligible_for_state_side_v22=true only when H2/H4/H5/H6 are all PASS",
        metric_inputs={
            "upstream_gate_statuses": upstream,
            "eligible_for_state_side_v22": eligible,
            "D_not_executed_in_v21": True,
        },
        source_artifacts=[],
        decision_text=decision_text,
        next_action=next_action,
    )


def build_libero_go_no_go_report_v21(
    *,
    paired_summary: Mapping[str, object],
    variant_authorities: Mapping[str, Mapping[str, object]],
    authority_mode: str,
) -> dict[str, object]:
    normalized_mode = _require_authority_mode(authority_mode)
    h0_gate = _evaluate_scope_gate(
        variant_authorities=variant_authorities,
        authority_mode=normalized_mode,
    )
    h1_gate = _evaluate_trace_gate(variant_authorities=variant_authorities)
    h3_gate = _evaluate_selection_purity_gate(variant_authorities=variant_authorities)
    prerequisite_statuses = [
        str(h0_gate["status"]),
        str(h1_gate["status"]),
        str(h3_gate["status"]),
    ]
    pairwise_delta_raw = paired_summary.get("pairwise_delta")
    pairwise_delta = (
        _require_mapping(pairwise_delta_raw, context="paired_summary.pairwise_delta")
        if isinstance(pairwise_delta_raw, Mapping)
        else None
    )
    h2_gate = _evaluate_headroom_gate(
        paired_summary=paired_summary,
        variant_authorities=variant_authorities,
        prerequisite_statuses=prerequisite_statuses,
    )
    h4_gate = _evaluate_pairwise_primary_gate(
        gate="H4",
        name="informativeness_gate",
        pair_label="C-X",
        paired_summary=paired_summary,
        pairwise_delta=pairwise_delta,
        variant_authorities=variant_authorities,
        prerequisite_statuses=prerequisite_statuses,
        pass_text="C beats diagnostic X on the selected primary metric with positive lower CI bound",
        hold_text="C beats diagnostic X point-wise, but the lower CI bound still crosses zero",
        fail_text="C does not beat diagnostic X on the selected primary metric",
        next_action_pass="保持 X 为 diagnostic-only；不要让 X 进入 headline winner。",
        next_action_hold="保持 X 为 diagnostic-only，并把 informativeness 记为 HOLD。",
        next_action_fail="不要把 X 当 headline winner；先解决 C 相对 X 的 informativeness 问题。",
    )
    h5_gate = _evaluate_pairwise_primary_gate(
        gate="H5",
        name="recap_gate",
        pair_label="C-B",
        paired_summary=paired_summary,
        pairwise_delta=pairwise_delta,
        variant_authorities=variant_authorities,
        prerequisite_statuses=prerequisite_statuses,
        pass_text="C beats B on the selected primary metric with positive lower CI bound",
        hold_text="C beats B point-wise, but the lower CI bound still crosses zero",
        fail_text="C does not beat B on the selected primary metric",
        next_action_pass="保留 RECAP validated 结论，但仍不得在 v21 内执行 D。",
        next_action_hold="把 RECAP gate 保持为 HOLD，等待更强证据。",
        next_action_fail="先解决 C 相对 B 的核心收益问题，不要推进 state side。",
    )
    h6_gate = _evaluate_viability_gate(
        pairwise_delta=pairwise_delta,
        variant_authorities=variant_authorities,
        prerequisite_statuses=prerequisite_statuses,
    )
    h7_gate = _evaluate_state_side_gate(
        h2_status=str(h2_gate["status"]),
        h4_status=str(h4_gate["status"]),
        h5_status=str(h5_gate["status"]),
        h6_status=str(h6_gate["status"]),
    )
    report: dict[str, object] = {
        "schema_version": GO_NO_GO_REPORT_SCHEMA_VERSION,
        "authority_mode": normalized_mode,
        "primary_metric_id": paired_summary.get("primary_metric_id"),
        "headline_winner": paired_summary.get("headline_winner"),
        "headline_variant_codes": paired_summary.get("headline_variant_codes"),
        "diagnostic_variant_codes": paired_summary.get("diagnostic_variant_codes"),
        "headroom_recovered": h2_gate["status"] == "PASS",
        "recap_validated_on_desaturated_eval": h5_gate["status"] == "PASS",
        "informativeness_validated": h4_gate["status"] == "PASS",
        "eligible_for_state_side_v22": _require_mapping(
            h7_gate.get("metric_inputs"), context="H7.metric_inputs"
        ).get("eligible_for_state_side_v22")
        is True,
        "D_not_executed_in_v21": True,
        "gate_order": ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"],
        "gates": {
            "H0": h0_gate,
            "H1": h1_gate,
            "H2": h2_gate,
            "H3": h3_gate,
            "H4": h4_gate,
            "H5": h5_gate,
            "H6": h6_gate,
            "H7": h7_gate,
        },
    }
    if normalized_mode == "lite":
        report["proceed_to_strong"] = all(
            status == "PASS" for status in prerequisite_statuses
        )
    return report


def build_libero_abcx_gate_artifacts_v21(
    *,
    variant_authorities: Mapping[str, Mapping[str, object]],
    authority_mode: str,
    selection_variant_authorities: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    paired_summary = build_libero_paired_summary_v21(
        variant_authorities=variant_authorities,
        authority_mode=authority_mode,
        selection_variant_authorities=selection_variant_authorities,
    )
    go_no_go_report = build_libero_go_no_go_report_v21(
        paired_summary=paired_summary,
        variant_authorities=variant_authorities,
        authority_mode=authority_mode,
    )
    return {
        "paired_summary": paired_summary,
        "go_no_go_report": go_no_go_report,
    }


__all__ = [
    "AGGREGATE_METRIC_IDS",
    "ALLOWED_PRIMARY_METRIC_IDS",
    "BOOTSTRAP_SCHEMA_VERSION",
    "COMPATIBILITY_ONLY_METRICS",
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_PRIMARY_METRIC_ID",
    "GO_NO_GO_NAME_BY_MODE",
    "GO_NO_GO_REPORT_SCHEMA_VERSION",
    "HEADLINE_METRIC_ORDER",
    "HEADLINE_VARIANT_CODES",
    "LITE_PRIMARY_VARIANT",
    "METRIC_LADDER_SCHEMA_VERSION",
    "PAIRWISE_DELTA_SCHEMA_VERSION",
    "PAIRED_SUMMARY_NAME_BY_MODE",
    "PAIRED_SUMMARY_SCHEMA_VERSION",
    "REQUIRED_PAIR_LABELS",
    "VARIANT_CODE_TO_ID",
    "AggregationValidationError",
    "assert_pairwise_delta_conservation_v21",
    "assert_variant_aggregate_conservation_v21",
    "build_libero_abcx_gate_artifacts_v21",
    "build_libero_go_no_go_report_v21",
    "build_libero_paired_summary_v21",
    "build_bootstrap_ci_v21",
    "build_metric_ladder_summary_v21",
    "build_pairwise_delta_payload_v21",
    "metric_point_estimates_from_trace_rows_v21",
    "select_primary_metric_id_v21",
]
