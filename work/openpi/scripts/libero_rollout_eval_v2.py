from __future__ import annotations

"""Compatibility shim for the retained rollout-v2 script import path.

The real implementation lives in `work.openpi.eval.workflows.tracked_gate`.
Keep this module thin so legacy imports keep working without turning the
compatibility package into an eager import hub.
"""

from work.openpi.checkpoint import resolve_servable_checkpoint_ref
from work.openpi.eval.workflows.tracked_gate import (
    BOOTSTRAP_SCHEMA_VERSION,
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    EVAL_MANIFEST_NAME,
    FailFastError,
    GO_NO_GO_CORE_SCHEMA_VERSION,
    PAIRED_DELTA_SCHEMA_VERSION,
    PER_EPISODE_NAME,
    PER_EPISODE_SCHEMA_VERSION,
    SUMMARY_NAME,
    SUMMARY_SCHEMA_VERSION,
    STOCK_VARIANTS,
    TOPIC,
    VIDEO_INDEX_NAME,
    VIDEO_INDEX_SCHEMA_VERSION,
    build_eval_manifest_id,
    derive_go_no_go_core_from_authority_bundle,
    load_rollout_eval_v2_authority_bundle,
    main,
)


def _resolve_servable_checkpoint_ref(
    checkpoint_ref: str,
    variant: str,
) -> tuple[str, str]:
    try:
        return resolve_servable_checkpoint_ref(
            checkpoint_ref=checkpoint_ref,
            variant=variant,
            stock_variants=STOCK_VARIANTS,
        )
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


__all__ = [
    "BOOTSTRAP_SCHEMA_VERSION",
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_CONFIDENCE_LEVEL",
    "EVAL_MANIFEST_NAME",
    "FailFastError",
    "GO_NO_GO_CORE_SCHEMA_VERSION",
    "PAIRED_DELTA_SCHEMA_VERSION",
    "PER_EPISODE_NAME",
    "PER_EPISODE_SCHEMA_VERSION",
    "SUMMARY_NAME",
    "SUMMARY_SCHEMA_VERSION",
    "STOCK_VARIANTS",
    "TOPIC",
    "VIDEO_INDEX_NAME",
    "VIDEO_INDEX_SCHEMA_VERSION",
    "_resolve_servable_checkpoint_ref",
    "build_eval_manifest_id",
    "derive_go_no_go_core_from_authority_bundle",
    "load_rollout_eval_v2_authority_bundle",
    "main",
]
