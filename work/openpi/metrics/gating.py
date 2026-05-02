from __future__ import annotations

from collections.abc import Mapping, Sequence

from work.openpi.eval.reports.go_no_go import (  # noqa: E402
    AGGREGATE_METRIC_IDS,
    BOOTSTRAP_SCHEMA_VERSION,
    COMPATIBILITY_ONLY_METRICS,
    DEFAULT_PRIMARY_METRIC_ID,
    GO_NO_GO_NAME_BY_MODE,
    HEADLINE_METRIC_ORDER,
    METRIC_LADDER_SCHEMA_VERSION,
    PAIRWISE_DELTA_SCHEMA_VERSION,
    PAIRED_SUMMARY_NAME_BY_MODE,
    REQUIRED_PAIR_LABELS,
    STOCK_SCAN_SUMMARY_SCHEMA_VERSION,
    VARIANT_CODE_TO_ID,
    AggregationValidationError,
    assert_pairwise_delta_conservation_v21,
    assert_variant_aggregate_conservation_v21,
    build_bootstrap_ci_v21,
    build_libero_abcx_gate_artifacts_v21,
    build_libero_go_no_go_report_v21,
    build_libero_paired_summary_v21,
    build_metric_ladder_summary_v21,
    build_pairwise_delta_payload_v21,
    metric_point_estimates_from_trace_rows_v21,
    select_primary_metric_id_v21,
)


def build_v21_metric_ladder_summary(
    *,
    trace_rows: Sequence[Mapping[str, object]],
    authority_id: str,
    variant: str,
    checkpoint_ref: str,
    metric_profile: str,
    primary_metric_id: str = DEFAULT_PRIMARY_METRIC_ID,
) -> dict[str, object]:
    return build_metric_ladder_summary_v21(
        trace_rows=trace_rows,
        authority_id=authority_id,
        variant=variant,
        checkpoint_ref=checkpoint_ref,
        metric_profile=metric_profile,
        primary_metric_id=primary_metric_id,
    )


__all__ = [
    "AGGREGATE_METRIC_IDS",
    "BOOTSTRAP_SCHEMA_VERSION",
    "COMPATIBILITY_ONLY_METRICS",
    "DEFAULT_PRIMARY_METRIC_ID",
    "GO_NO_GO_NAME_BY_MODE",
    "HEADLINE_METRIC_ORDER",
    "METRIC_LADDER_SCHEMA_VERSION",
    "PAIRWISE_DELTA_SCHEMA_VERSION",
    "PAIRED_SUMMARY_NAME_BY_MODE",
    "REQUIRED_PAIR_LABELS",
    "STOCK_SCAN_SUMMARY_SCHEMA_VERSION",
    "VARIANT_CODE_TO_ID",
    "AggregationValidationError",
    "assert_pairwise_delta_conservation_v21",
    "assert_variant_aggregate_conservation_v21",
    "build_bootstrap_ci_v21",
    "build_libero_abcx_gate_artifacts_v21",
    "build_libero_go_no_go_report_v21",
    "build_libero_paired_summary_v21",
    "build_pairwise_delta_payload_v21",
    "build_v21_metric_ladder_summary",
    "metric_point_estimates_from_trace_rows_v21",
    "select_primary_metric_id_v21",
]
