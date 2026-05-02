from __future__ import annotations

from collections.abc import Mapping, Sequence
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import cast

from .checkpoint import read_json, write_json
from .protocol import (
    CURRENT_BACKBONE_VARIANT_IDS,
    PAPER_FULL_VARIANT_IDS,
    RECAP_ONLY_VARIANT,
    REPAIRED_CURRENT_BACKBONE_LAYER_ID,
    REPAIRED_METRIC_PROFILE_ID,
    REPAIRED_PAPER_FULL_LAYER_ID,
    REPAIRED_PRIMARY_METRIC_ORDER,
    REPAIRED_RUNTIME_LAYER_ID,
    REPO_ROOT,
    RUNTIME_LAYER_VARIANT_IDS,
    build_repaired_future_comparisons,
    build_repaired_headline_comparisons,
    build_repaired_matrix_layers,
    build_repaired_metric_profile,
    build_repaired_variant_catalog,
)


CONTROL_GATE_SCHEMA_VERSION = "openpi_libero_recap_control_gate_v1"
CONTROL_GATE_ROUTE_ID = "task9b_recap_only_relabel8d_control_gate_v1"
RELABEL_CONTROL_OUTPUT_DIR = (
    REPO_ROOT
    / "agent"
    / "artifacts"
    / "checkpoints"
    / "openpi_libero_variants"
    / "recap_only_relabel8d_v1"
)
SOURCE_EQUIVALENCE_REPORT_NAME = "source_equivalence_report.json"
BLOCKED_EXIT_CODE = 42
REPAIRED_GATE_SCHEMA_VERSION = "openpi_recap_repaired_gate_v1"
TASK11_BLOCKER_VERDICT_SCHEMA_VERSION = "openpi_recap_task11_blocker_verdict_v1"
FINAL_GATE_SUMMARY_SCHEMA_VERSION = "openpi_recap_final_gate_summary_v1"
TASK11_OVERLAY_MATERIALIZATION_SCHEMA_VERSION = (
    "openpi_recap_overlay_materialization_v1"
)
TASK11_VERIFICATION_SCHEMA_VERSION = "openpi_recap_task11_verification_v1"
TASK11_REQUIRED_VERIFIED_PATH_SCOPES: tuple[str, ...] = (
    "full_path",
    "paper_full_path",
)
TASK11_REQUIRED_OVERLAY_FILES: tuple[str, ...] = (
    "src/openpi/policies/policy_config.py",
    "src/openpi/recap_overlay/__init__.py",
    "src/openpi/recap_overlay/config.py",
    "src/openpi/recap_overlay/modeling.py",
    "src/openpi/recap_overlay/training.py",
    "src/openpi/training/config.py",
)
TASK11_REQUIRED_SOURCE_BACKUPS: tuple[tuple[str, str], ...] = (
    (
        "src/openpi/policies/policy_config.py",
        "src/openpi/policies/_upstream_openpi_recap_policy_config.py",
    ),
    (
        "src/openpi/training/config.py",
        "src/openpi/training/_upstream_openpi_recap_training_config.py",
    ),
)
TASK11_REQUIRED_KEY_SOURCE_FILES: tuple[str, ...] = (
    "src/openpi/models/pi0.py",
    "src/openpi/policies/policy_config.py",
    "src/openpi/training/config.py",
)
FINAL_GATE_INPUT_ARTIFACT_KEYS: tuple[str, ...] = (
    "repaired_matrix_summary",
    "eval_summary",
    "repaired_gate_results",
    "blocker_verdict",
    "overlay_materialization",
    "task11_verification",
)
TASK11_CURRENT_RUN_SOURCE_REF_KEYS: tuple[str, ...] = (
    "overlay_materialization",
    "eval_summary",
    "repaired_gate_results",
    "blocker_verdict",
)
TASK11_CURRENT_RUN_VARIANT_IDS: tuple[str, ...] = (
    "B0_omit_control_v2",
    "C0_recap_informative_positiveinfer_v2",
    "C1_recap_informative_cfg_v2",
)
FINAL_GATE_REQUIRED_REVIEWERS: tuple[str, ...] = ("F1", "F2", "F3", "F4")
FINAL_GATE_APPROVE_VERDICT = "APPROVE"
REPAIRED_GATE_ORDER: tuple[str, ...] = (
    "G0",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "G7",
)
TASK11_GATE_ORDER: tuple[str, ...] = REPAIRED_GATE_ORDER[:-1]
WORDING_RULE_REPAIRED_PATH_ONLY = "repaired_path_only"
WORDING_RULE_FULL_AND_PAPER_FULL_ALLOWED = "full_and_paper_full_allowed"

CONTROL_SEMANTIC_FIELDS: tuple[str, ...] = (
    "train_stage",
    "train_consumer_mode",
    "fixed_indicator_mode",
    "prompt_text_surface",
    "per_sample_indicator_consumption",
    "runtime_indicator_mode",
)


def canonical_json_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _require_mapping(raw: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


def _require_sequence(raw: object, *, context: str) -> Sequence[object]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(f"{context} must be a sequence")
    return raw


def _coerce_path(raw: object, *, context: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise TypeError(f"{context} must be a non-empty path string")
    return Path(raw).resolve()


def _coerce_str(raw: object, *, context: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return raw.strip()


def _coerce_float(raw: object, *, context: str) -> float:
    if isinstance(raw, bool) or raw is None:
        raise TypeError(f"{context} must be float-like, got {raw!r}")
    if not isinstance(raw, (int, float, str)):
        raise TypeError(f"{context} must be float-like, got {type(raw).__name__}")
    return float(raw)


def _coerce_int(raw: object, *, context: str) -> int:
    if isinstance(raw, bool) or raw is None:
        raise TypeError(f"{context} must be int-like, got {raw!r}")
    if not isinstance(raw, (int, float, str)):
        raise TypeError(f"{context} must be int-like, got {type(raw).__name__}")
    return int(raw)


def _coerce_bool(raw: object, *, context: str) -> bool:
    if not isinstance(raw, bool):
        raise TypeError(f"{context} must be a bool, got {type(raw).__name__}")
    return raw


def _coerce_optional_path_str(raw: object, *, context: str) -> str | None:
    if raw is None:
        return None
    return str(_coerce_path(raw, context=context))


def _coerce_optional_path_mapping(
    source_refs: Mapping[str, object] | None,
    *,
    context: str,
) -> dict[str, str | None]:
    normalized: dict[str, str | None] = {
        key: None for key in FINAL_GATE_INPUT_ARTIFACT_KEYS
    }
    if source_refs is None:
        return normalized
    for key in FINAL_GATE_INPUT_ARTIFACT_KEYS:
        normalized[key] = _coerce_optional_path_str(
            source_refs.get(key),
            context=f"{context}.{key}",
        )
    return normalized


def _normalize_reviewer_approvals(
    reviewer_approvals: Mapping[str, object] | None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if reviewer_approvals is None:
        return normalized
    for reviewer in FINAL_GATE_REQUIRED_REVIEWERS:
        raw_verdict = reviewer_approvals.get(reviewer)
        if raw_verdict is None:
            continue
        verdict = _coerce_str(
            raw_verdict,
            context=f"reviewer_approvals.{reviewer}",
        ).strip()
        if verdict:
            normalized[reviewer] = verdict.upper()
    return normalized


def build_final_reviewer_state(
    *,
    current_run_binding_passed: bool,
    reviewer_approvals: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized_approvals = _normalize_reviewer_approvals(reviewer_approvals)
    completed_reviewers = [
        reviewer
        for reviewer in FINAL_GATE_REQUIRED_REVIEWERS
        if reviewer in normalized_approvals
    ]
    approved_reviewers = [
        reviewer
        for reviewer in FINAL_GATE_REQUIRED_REVIEWERS
        if normalized_approvals.get(reviewer) == FINAL_GATE_APPROVE_VERDICT
    ]
    missing_reviewers = [
        reviewer
        for reviewer in FINAL_GATE_REQUIRED_REVIEWERS
        if reviewer not in normalized_approvals
    ]
    rejected_reviewers = [
        reviewer
        for reviewer in completed_reviewers
        if normalized_approvals.get(reviewer) != FINAL_GATE_APPROVE_VERDICT
    ]
    approve_all = (
        current_run_binding_passed
        and not missing_reviewers
        and not rejected_reviewers
        and len(approved_reviewers) == len(FINAL_GATE_REQUIRED_REVIEWERS)
    )
    if not current_run_binding_passed:
        status = "blocked"
        decision_text = (
            "review wave remains blocked until current-run authority binding is valid"
        )
    elif approve_all:
        status = "approve_all"
        decision_text = "F1-F4 all approved; final review closure is complete"
    else:
        status = "pending_review"
        decision_text = (
            "F1-F4 review approvals are incomplete; final review remains pending"
        )
    return {
        "status": status,
        "required_reviewers": list(FINAL_GATE_REQUIRED_REVIEWERS),
        "completed_reviewers": completed_reviewers,
        "approved_reviewers": approved_reviewers,
        "missing_reviewers": missing_reviewers,
        "rejected_reviewers": rejected_reviewers,
        "reviewer_verdicts": {
            reviewer: normalized_approvals[reviewer] for reviewer in completed_reviewers
        },
        "decision_text": decision_text,
    }


def _coerce_metric(raw: object, *, context: str) -> float:
    if isinstance(raw, bool) or raw is None:
        raise TypeError(f"{context} must be numeric, got {raw!r}")
    if not isinstance(raw, (int, float, str)):
        raise TypeError(f"{context} must be numeric, got {type(raw).__name__}")
    return float(raw)


def _derive_task11_current_run_id(eval_summary: Mapping[str, object]) -> str:
    variant_rows = {
        _coerce_str(
            mapping.get("repaired_variant_id"),
            context="eval_summary.variant_results[].repaired_variant_id",
        ): mapping
        for mapping in (
            _require_mapping(row, context="eval_summary.variant_results[]")
            for row in _require_sequence(
                eval_summary.get("variant_results"),
                context="eval_summary.variant_results",
            )
        )
    }
    observed_run_ids: set[str] = set()
    for variant_id in TASK11_CURRENT_RUN_VARIANT_IDS:
        variant_row = variant_rows.get(variant_id)
        if variant_row is None:
            raise TypeError(
                f"missing task11 current-run variant {variant_id} in eval_summary"
            )
        summary_path = _coerce_path(
            variant_row.get("summary_ref"),
            context=f"eval_summary.variant_results.{variant_id}.summary_ref",
        )
        eval_manifest_path = _coerce_path(
            variant_row.get("eval_manifest_ref"),
            context=f"eval_summary.variant_results.{variant_id}.eval_manifest_ref",
        )
        summary_payload = read_json(summary_path)
        eval_manifest_payload = read_json(eval_manifest_path)
        summary_run_id = _coerce_str(
            summary_payload.get("eval_manifest_id"),
            context=f"{variant_id}.summary.eval_manifest_id",
        )
        manifest_run_id = _coerce_str(
            eval_manifest_payload.get("eval_manifest_id"),
            context=f"{variant_id}.eval_manifest.eval_manifest_id",
        )
        runtime_dir = _coerce_path(
            summary_payload.get("runtime_dir"),
            context=f"{variant_id}.summary.runtime_dir",
        )
        if summary_run_id != manifest_run_id:
            raise TypeError(
                f"run id mismatch between summary and eval manifest for {variant_id}"
            )
        if runtime_dir.name != summary_run_id:
            raise TypeError(f"runtime_dir basename mismatch for {variant_id}")
        observed_run_ids.add(summary_run_id)
    if len(observed_run_ids) != 1:
        raise TypeError("task11 current-run variants do not share one run_id")
    return next(iter(observed_run_ids))


def _recap_summary_row(paired_summary: Mapping[str, object]) -> Mapping[str, object]:
    rows = _require_sequence(
        paired_summary.get("paired_summary"), context="paired_summary.paired_summary"
    )
    for row in rows:
        mapping = _require_mapping(row, context="paired_summary.paired_summary[]")
        if str(mapping.get("variant", "")).strip() == RECAP_ONLY_VARIANT:
            return mapping
    raise ValueError("paired_summary must contain a recap_only row")


def _source_dataset_dirs(
    train_manifest: Mapping[str, object], checkpoint_provenance: Mapping[str, object]
) -> tuple[Path, ...]:
    train_source = _require_mapping(
        train_manifest.get("train_source"), context="train_manifest.train_source"
    )
    variant_derivation = _require_mapping(
        checkpoint_provenance.get("variant_derivation", {}),
        context="checkpoint_provenance.variant_derivation",
    )
    dirs = {
        _coerce_path(
            train_source.get("dataset_dir"),
            context="train_manifest.train_source.dataset_dir",
        ),
        _coerce_path(
            variant_derivation.get("source_dataset_dir"),
            context="checkpoint_provenance.variant_derivation.source_dataset_dir",
        ),
    }
    return tuple(sorted(dirs))


def _existing_control_consistency(
    train_manifest: Mapping[str, object],
    checkpoint_provenance: Mapping[str, object],
    paired_summary: Mapping[str, object],
) -> tuple[Path, tuple[str, ...]]:
    issues: list[str] = []
    recap_row = _recap_summary_row(paired_summary)
    manifest_variant = str(train_manifest.get("variant", "")).strip()
    provenance_variant = str(checkpoint_provenance.get("variant", "")).strip()
    summary_variant = str(recap_row.get("variant", "")).strip()
    if manifest_variant != RECAP_ONLY_VARIANT:
        issues.append(
            f"train_manifest.variant must be {RECAP_ONLY_VARIANT!r}, got {manifest_variant!r}"
        )
    if provenance_variant != RECAP_ONLY_VARIANT:
        issues.append(
            "checkpoint_provenance.variant must be "
            + f"{RECAP_ONLY_VARIANT!r}, got {provenance_variant!r}"
        )
    if summary_variant != RECAP_ONLY_VARIANT:
        issues.append(
            f"paired_summary recap row must be {RECAP_ONLY_VARIANT!r}, got {summary_variant!r}"
        )
    manifest_checkpoint_dir = _coerce_path(
        train_manifest.get("checkpoint_dir"), context="train_manifest.checkpoint_dir"
    )
    provenance_checkpoint_dir = _coerce_path(
        checkpoint_provenance.get("checkpoint_dir"),
        context="checkpoint_provenance.checkpoint_dir",
    )
    summary_checkpoint_dir = _coerce_path(
        recap_row.get("checkpoint_dir"),
        context="paired_summary.recap_only.checkpoint_dir",
    )
    if manifest_checkpoint_dir != provenance_checkpoint_dir:
        issues.append(
            "train_manifest.checkpoint_dir and checkpoint_provenance.checkpoint_dir differ"
        )
    if manifest_checkpoint_dir != summary_checkpoint_dir:
        issues.append(
            "train_manifest.checkpoint_dir and paired_summary recap_only checkpoint_dir differ"
        )
    manifest_train_source = _require_mapping(
        train_manifest.get("train_source"), context="train_manifest.train_source"
    )
    variant_derivation = _require_mapping(
        checkpoint_provenance.get("variant_derivation", {}),
        context="checkpoint_provenance.variant_derivation",
    )
    manifest_source_name = _coerce_str(
        manifest_train_source.get("dataset_name"),
        context="train_manifest.train_source.dataset_name",
    )
    provenance_source_name = _coerce_str(
        variant_derivation.get("source_dataset_name"),
        context="checkpoint_provenance.variant_derivation.source_dataset_name",
    )
    if manifest_source_name != provenance_source_name:
        issues.append(
            "train_manifest.train_source.dataset_name and checkpoint_provenance.variant_derivation.source_dataset_name differ"
        )
    return manifest_checkpoint_dir, tuple(issues)


def _load_variant_catalog(summary: Mapping[str, object]) -> Mapping[str, object]:
    raw = summary.get("variant_catalog")
    if raw is None:
        return build_repaired_variant_catalog()
    return _require_mapping(raw, context="summary.variant_catalog")


def _variant_entry(
    summary: Mapping[str, object], variant_id: str
) -> Mapping[str, object]:
    catalog = _load_variant_catalog(summary)
    if variant_id not in catalog:
        raise KeyError(f"unknown repaired matrix variant: {variant_id}")
    return _require_mapping(
        catalog[variant_id], context=f"variant_catalog.{variant_id}"
    )


def _headline_comparison_map(
    summary: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    rows = _headline_comparison_rows(summary)
    resolved: dict[str, Mapping[str, object]] = {}
    for row in rows:
        resolved[_coerce_str(row.get("comparison_id"), context="comparison_id")] = row
    return resolved


def _headline_comparison_rows(
    summary: Mapping[str, object],
) -> list[Mapping[str, object]]:
    raw = summary.get("headline_comparisons")
    if raw is None:
        return [
            {
                "comparison_id": comparison.comparison_id,
                "lhs_variant_id": comparison.lhs_variant_id,
                "rhs_variant_id": comparison.rhs_variant_id,
                "relation": comparison.relation,
                "gate": comparison.gate,
                "purpose": comparison.purpose,
                "metric_profile_id": comparison.metric_profile_id,
            }
            for comparison in build_repaired_headline_comparisons()
        ]
    return [
        _require_mapping(row, context="headline_comparisons[]")
        for row in _require_sequence(raw, context="summary.headline_comparisons")
    ]


def _future_comparison_map(
    summary: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw = summary.get("future_comparisons")
    if raw is None:
        rows = [
            {
                "comparison_id": comparison.comparison_id,
                "lhs_variant_id": comparison.lhs_variant_id,
                "rhs_variant_id": comparison.rhs_variant_id,
                "relation": comparison.relation,
                "gate": comparison.gate,
                "purpose": comparison.purpose,
                "metric_profile_id": comparison.metric_profile_id,
            }
            for comparison in build_repaired_future_comparisons()
        ]
    else:
        rows = list(_require_sequence(raw, context="summary.future_comparisons"))
    resolved: dict[str, Mapping[str, object]] = {}
    for row in rows:
        mapping = _require_mapping(row, context="future_comparisons[]")
        resolved[_coerce_str(mapping.get("comparison_id"), context="comparison_id")] = (
            mapping
        )
    return resolved


def _metric_profile(summary: Mapping[str, object]) -> Mapping[str, object]:
    raw = summary.get("metric_profile")
    if raw is None:
        return build_repaired_metric_profile()
    return _require_mapping(raw, context="summary.metric_profile")


def _observed_metrics(
    summary: Mapping[str, object],
    *,
    metrics_by_variant: Mapping[str, Mapping[str, object]] | None = None,
) -> Mapping[str, Mapping[str, object]]:
    if metrics_by_variant is not None:
        return metrics_by_variant
    raw = summary.get("observed_metrics")
    if raw is None:
        return {}
    resolved = _require_mapping(raw, context="summary.observed_metrics")
    return {
        str(key): _require_mapping(value, context=f"observed_metrics.{key}")
        for key, value in resolved.items()
    }


def _semantic_signature(
    variant_entry: Mapping[str, object],
) -> dict[str, object]:
    return {field: variant_entry.get(field) for field in CONTROL_SEMANTIC_FIELDS}


def _metric_value(
    metrics: Mapping[str, object],
    metric_id: str,
    *,
    variant_id: str,
) -> float:
    if metric_id not in metrics:
        raise KeyError(f"missing metric {metric_id!r} for {variant_id!r}")
    return _coerce_metric(metrics[metric_id], context=f"{variant_id}.{metric_id}")


def _ordered_metric_comparison(
    lhs_variant_id: str,
    rhs_variant_id: str,
    lhs_metrics: Mapping[str, object],
    rhs_metrics: Mapping[str, object],
) -> dict[str, object]:
    ordered_results: list[dict[str, object]] = []
    decision = "="
    decisive_metric_id = None
    for metric_id in REPAIRED_PRIMARY_METRIC_ORDER:
        lhs_value = _metric_value(lhs_metrics, metric_id, variant_id=lhs_variant_id)
        rhs_value = _metric_value(rhs_metrics, metric_id, variant_id=rhs_variant_id)
        if lhs_value > rhs_value:
            metric_relation = ">"
        elif lhs_value < rhs_value:
            metric_relation = "<"
        else:
            metric_relation = "="
        ordered_results.append(
            {
                "metric_id": metric_id,
                "lhs_value": lhs_value,
                "rhs_value": rhs_value,
                "relation": metric_relation,
            }
        )
        if metric_relation != "=" and decisive_metric_id is None:
            decision = metric_relation
            decisive_metric_id = metric_id
            break
    return {
        "relation": decision,
        "decisive_metric_id": decisive_metric_id,
        "ordered_results": ordered_results,
    }


def _relation_satisfied(decision: str, required_relation: str) -> bool:
    if required_relation == "!=":
        return decision != "="
    if required_relation == ">":
        return decision == ">"
    if required_relation == ">=":
        return decision in {">", "="}
    raise ValueError(f"unsupported repaired gate relation: {required_relation!r}")


def evaluate_control_semantics_gate(summary: Mapping[str, object]) -> dict[str, object]:
    lhs_variant_id = "B1_fixed_positive_sft_v2"
    rhs_variant_id = "B0_omit_control_v2"
    lhs_entry = _variant_entry(summary, lhs_variant_id)
    rhs_entry = _variant_entry(summary, rhs_variant_id)
    lhs_signature = _semantic_signature(lhs_entry)
    rhs_signature = _semantic_signature(rhs_entry)
    passed = lhs_signature != rhs_signature
    return {
        "schema_version": REPAIRED_GATE_SCHEMA_VERSION,
        "gate": "G1",
        "name": "control_semantics_gate",
        "status": "pass" if passed else "fail",
        "comparison_id": "B1_vs_B0",
        "lhs_variant_id": lhs_variant_id,
        "rhs_variant_id": rhs_variant_id,
        "required_relation": "!=",
        "semantic_fields": list(CONTROL_SEMANTIC_FIELDS),
        "lhs_signature": lhs_signature,
        "rhs_signature": rhs_signature,
        "decision_text": (
            "B1 fixed-positive SFT remains semantically distinct from B0 omit control"
            if passed
            else "B1 and B0 semantics collapsed"
        ),
    }


def evaluate_metric_comparison_gate(
    summary: Mapping[str, object],
    *,
    comparison_id: str,
    metrics_by_variant: Mapping[str, Mapping[str, object]] | None = None,
    future_only: bool = False,
) -> dict[str, object]:
    comparison_map = (
        _future_comparison_map(summary)
        if future_only
        else _headline_comparison_map(summary)
    )
    if comparison_id not in comparison_map:
        raise KeyError(f"unknown repaired matrix comparison: {comparison_id}")
    comparison = comparison_map[comparison_id]
    lhs_variant_id = _coerce_str(
        comparison.get("lhs_variant_id"), context=f"{comparison_id}.lhs_variant_id"
    )
    rhs_variant_id = _coerce_str(
        comparison.get("rhs_variant_id"), context=f"{comparison_id}.rhs_variant_id"
    )
    relation = _coerce_str(
        comparison.get("relation"), context=f"{comparison_id}.relation"
    )
    observed = _observed_metrics(summary, metrics_by_variant=metrics_by_variant)
    if future_only:
        return {
            "schema_version": REPAIRED_GATE_SCHEMA_VERSION,
            "gate": _coerce_str(
                comparison.get("gate"), context=f"{comparison_id}.gate"
            ),
            "name": "paper_full_iteration_value_gate",
            "status": "future_only",
            "comparison_id": comparison_id,
            "lhs_variant_id": lhs_variant_id,
            "rhs_variant_id": rhs_variant_id,
            "required_relation": relation,
            "metric_profile_id": REPAIRED_METRIC_PROFILE_ID,
            "decision_text": "paper-full P* layer stays separated until future execution materializes metrics",
            "observed_metric_variants": sorted(observed.keys()),
        }

    lhs_metrics = observed.get(lhs_variant_id)
    rhs_metrics = observed.get(rhs_variant_id)
    if lhs_metrics is None or rhs_metrics is None:
        return {
            "schema_version": REPAIRED_GATE_SCHEMA_VERSION,
            "gate": _coerce_str(
                comparison.get("gate"), context=f"{comparison_id}.gate"
            ),
            "name": _coerce_str(
                comparison.get("purpose"), context=f"{comparison_id}.purpose"
            ),
            "status": "pending_evidence",
            "comparison_id": comparison_id,
            "lhs_variant_id": lhs_variant_id,
            "rhs_variant_id": rhs_variant_id,
            "required_relation": relation,
            "metric_profile_id": REPAIRED_METRIC_PROFILE_ID,
            "required_metric_order": list(REPAIRED_PRIMARY_METRIC_ORDER),
            "decision_text": "comparison semantics frozen, waiting for rollout metrics",
        }

    comparison_result = _ordered_metric_comparison(
        lhs_variant_id,
        rhs_variant_id,
        lhs_metrics,
        rhs_metrics,
    )
    passed = _relation_satisfied(
        str(comparison_result["relation"]),
        relation,
    )
    return {
        "schema_version": REPAIRED_GATE_SCHEMA_VERSION,
        "gate": _coerce_str(comparison.get("gate"), context=f"{comparison_id}.gate"),
        "name": _coerce_str(
            comparison.get("purpose"), context=f"{comparison_id}.purpose"
        ),
        "status": "pass" if passed else "fail",
        "comparison_id": comparison_id,
        "lhs_variant_id": lhs_variant_id,
        "rhs_variant_id": rhs_variant_id,
        "required_relation": relation,
        "metric_profile_id": REPAIRED_METRIC_PROFILE_ID,
        "decisive_metric_id": comparison_result["decisive_metric_id"],
        "observed_relation": comparison_result["relation"],
        "ordered_results": comparison_result["ordered_results"],
        "decision_text": (
            f"{comparison_id} satisfied using budget-aware metric order"
            if passed
            else f"{comparison_id} violated under budget-aware metric order"
        ),
    }


def evaluate_informativeness_gate(
    summary: Mapping[str, object],
    *,
    metrics_by_variant: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    return evaluate_metric_comparison_gate(
        summary,
        comparison_id="C0_vs_X",
        metrics_by_variant=metrics_by_variant,
    )


def evaluate_recap_gate(
    summary: Mapping[str, object],
    *,
    metrics_by_variant: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    return evaluate_metric_comparison_gate(
        summary,
        comparison_id="C0_vs_B1",
        metrics_by_variant=metrics_by_variant,
    )


def evaluate_runtime_cfg_gate(
    summary: Mapping[str, object],
    *,
    metrics_by_variant: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    return evaluate_metric_comparison_gate(
        summary,
        comparison_id="C1_vs_C0",
        metrics_by_variant=metrics_by_variant,
    )


def evaluate_iteration_value_gate(summary: Mapping[str, object]) -> dict[str, object]:
    return evaluate_metric_comparison_gate(
        summary,
        comparison_id="P3_vs_P2",
        future_only=True,
    )


def build_repaired_gate_rows(
    summary: Mapping[str, object],
    *,
    metrics_by_variant: Mapping[str, Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    layers = summary.get("layers")
    if layers is None:
        layers_payload = list(build_repaired_matrix_layers())
    else:
        layers_payload = list(_require_sequence(layers, context="summary.layers"))

    layer_ids: list[str] = []
    variant_membership: dict[str, set[str]] = {}
    for row in layers_payload:
        mapping = _require_mapping(row, context="summary.layers[]")
        layer_id = _coerce_str(
            mapping.get("layer_id"), context="summary.layers[].layer_id"
        )
        layer_ids.append(layer_id)
        variant_membership[layer_id] = {
            _coerce_str(value, context=f"{layer_id}.variant_ids[]")
            for value in _require_sequence(
                mapping.get("variant_ids"), context=f"{layer_id}.variant_ids"
            )
        }

    expected_layers = {
        REPAIRED_RUNTIME_LAYER_ID,
        REPAIRED_CURRENT_BACKBONE_LAYER_ID,
        REPAIRED_PAPER_FULL_LAYER_ID,
    }
    g0_pass = set(layer_ids) == expected_layers and not (
        variant_membership.get(REPAIRED_RUNTIME_LAYER_ID, set())
        & variant_membership.get(REPAIRED_CURRENT_BACKBONE_LAYER_ID, set())
    )

    control_gate = evaluate_control_semantics_gate(summary)
    informativeness_gate = evaluate_informativeness_gate(
        summary, metrics_by_variant=metrics_by_variant
    )
    recap_gate = evaluate_recap_gate(summary, metrics_by_variant=metrics_by_variant)
    runtime_cfg_gate = evaluate_runtime_cfg_gate(
        summary, metrics_by_variant=metrics_by_variant
    )

    comparison_ids = [
        _coerce_str(
            row.get("comparison_id"), context="headline_comparisons[].comparison_id"
        )
        for row in _headline_comparison_map(summary).values()
    ]
    g5_pass = comparison_ids == ["C0_vs_B1", "C0_vs_X", "B1_vs_B0", "C1_vs_C0"]

    metric_profile = _metric_profile(summary)
    g6_pass = (
        _coerce_str(
            metric_profile.get("metric_profile_id"),
            context="metric_profile.metric_profile_id",
        )
        == REPAIRED_METRIC_PROFILE_ID
        and tuple(
            _coerce_str(value, context="metric_profile.primary_metric_order[]")
            for value in _require_sequence(
                metric_profile.get("primary_metric_order"),
                context="metric_profile.primary_metric_order",
            )
        )
        == REPAIRED_PRIMARY_METRIC_ORDER
        and "throughput_like_score"
        in _require_sequence(
            metric_profile.get("primary_metric_order"),
            context="metric_profile.primary_metric_order",
        )
    )

    future_gate = evaluate_iteration_value_gate(summary)
    g7_pass = (
        variant_membership.get(REPAIRED_PAPER_FULL_LAYER_ID, set())
        == set(PAPER_FULL_VARIANT_IDS)
        and variant_membership.get(REPAIRED_CURRENT_BACKBONE_LAYER_ID, set())
        == set(CURRENT_BACKBONE_VARIANT_IDS)
        and variant_membership.get(REPAIRED_RUNTIME_LAYER_ID, set())
        == set(RUNTIME_LAYER_VARIANT_IDS)
    )

    return [
        {
            "schema_version": REPAIRED_GATE_SCHEMA_VERSION,
            "gate": "G0",
            "name": "matrix_layer_separation_gate",
            "status": "pass" if g0_pass else "fail",
            "decision_text": "runtime/current-backbone/paper-full layers stay explicitly separated",
            "expected_layer_ids": sorted(expected_layers),
            "observed_layer_ids": layer_ids,
        },
        control_gate,
        informativeness_gate,
        recap_gate,
        runtime_cfg_gate,
        {
            "schema_version": REPAIRED_GATE_SCHEMA_VERSION,
            "gate": "G5",
            "name": "headline_comparison_order_gate",
            "status": "pass" if g5_pass else "fail",
            "decision_text": "headline comparisons stay frozen in repaired order",
            "headline_comparison_ids": comparison_ids,
        },
        {
            "schema_version": REPAIRED_GATE_SCHEMA_VERSION,
            "gate": "G6",
            "name": "budget_aware_metric_profile_gate",
            "status": "pass" if g6_pass else "fail",
            "decision_text": "headline/gate semantics keep budget and throughput ordering",
            "metric_profile": dict(metric_profile),
        },
        {
            "schema_version": REPAIRED_GATE_SCHEMA_VERSION,
            "gate": "G7",
            "name": "paper_full_layer_separation_gate",
            "status": "pass" if g7_pass else "fail",
            "decision_text": "paper-full P0..P7 stay separate from repaired backbone while keeping P3>P2 reserved",
            "future_gate": future_gate,
        },
    ]


def build_repaired_headline_results(
    summary: Mapping[str, object],
    *,
    metrics_by_variant: Mapping[str, Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    return [
        {
            **evaluate_metric_comparison_gate(
                summary,
                comparison_id=_coerce_str(
                    row.get("comparison_id"),
                    context="headline_comparisons[].comparison_id",
                ),
                metrics_by_variant=metrics_by_variant,
            ),
            "comparison_role": "headline",
        }
        for row in _headline_comparison_rows(summary)
    ]


def build_task11_blocker_verdict(
    gate_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    gate_map: dict[str, Mapping[str, object]] = {}
    for row in gate_rows:
        mapping = _require_mapping(row, context="gate_rows[]")
        gate = _coerce_str(mapping.get("gate"), context="gate_rows[].gate")
        gate_map[gate] = mapping

    missing_gates = [gate for gate in TASK11_GATE_ORDER if gate not in gate_map]
    passed_gates: list[str] = []
    pending_gates: list[str] = []
    blocking_gates: list[str] = []
    gate_statuses: dict[str, str] = {}
    gate_details: list[dict[str, object]] = []

    for gate in TASK11_GATE_ORDER:
        row = gate_map.get(gate)
        if row is None:
            status = "missing_gate"
            gate_statuses[gate] = status
            blocking_gates.append(gate)
            gate_details.append(
                {
                    "gate": gate,
                    "status": status,
                    "name": "missing_gate",
                    "decision_text": f"required gate {gate} is missing",
                }
            )
            continue

        status = str(row.get("status", "")).strip().lower() or "unknown"
        gate_statuses[gate] = status
        gate_details.append(
            {
                "gate": gate,
                "status": status,
                "name": row.get("name"),
                "decision_text": row.get("decision_text"),
            }
        )
        if status == "pass":
            passed_gates.append(gate)
        elif status == "pending_evidence":
            pending_gates.append(gate)
        else:
            blocking_gates.append(gate)

    ready_for_task11 = not blocking_gates and not pending_gates
    if ready_for_task11:
        decision_text = (
            "all repaired gates G0-G6 passed; task 11 may consume this verdict"
        )
    else:
        decision_parts: list[str] = []
        if blocking_gates:
            decision_parts.append(f"blocking_gates={blocking_gates}")
        if pending_gates:
            decision_parts.append(f"pending_gates={pending_gates}")
        if missing_gates:
            decision_parts.append(f"missing_gates={missing_gates}")
        decision_text = "; ".join(decision_parts) or "task 11 remains blocked"

    return {
        "schema_version": TASK11_BLOCKER_VERDICT_SCHEMA_VERSION,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "gate_order": list(TASK11_GATE_ORDER),
        "ready_for_task11": ready_for_task11,
        "verdict": (
            "ready_for_task11" if ready_for_task11 else "blocked_by_repaired_gates"
        ),
        "decision_text": decision_text,
        "passed_gates": passed_gates,
        "pending_gates": pending_gates,
        "blocking_gates": blocking_gates,
        "missing_gates": missing_gates,
        "gate_statuses": gate_statuses,
        "gate_details": gate_details,
    }


def _resolved_gate_rows(
    gate_results: Mapping[str, object],
) -> tuple[
    tuple[str, ...], list[Mapping[str, object]], dict[str, Mapping[str, object]]
]:
    gate_order = tuple(
        _coerce_str(value, context="gate_results.gate_order[]")
        for value in _require_sequence(
            gate_results.get("gate_order"), context="gate_results.gate_order"
        )
    )
    if gate_order != REPAIRED_GATE_ORDER:
        raise ValueError(
            "final gate summary requires repaired gate order "
            + f"{list(REPAIRED_GATE_ORDER)}, got {list(gate_order)}"
        )

    rows = [
        _require_mapping(row, context="gate_results.gates[]")
        for row in _require_sequence(
            gate_results.get("gates"), context="gate_results.gates"
        )
    ]
    observed_gate_order = tuple(
        _coerce_str(row.get("gate"), context="gate_results.gates[].gate")
        for row in rows
    )
    if observed_gate_order != REPAIRED_GATE_ORDER:
        raise ValueError(
            "final gate summary requires exactly one G0-G7 row in canonical order"
        )

    gate_map: dict[str, Mapping[str, object]] = {}
    for row in rows:
        gate = _coerce_str(row.get("gate"), context="gate_results.gates[].gate")
        gate_map[gate] = row
    return gate_order, rows, gate_map


def _validate_task11_blocker_consistency(
    blocker_verdict: Mapping[str, object],
    *,
    gate_map: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    blocker_gate_order = tuple(
        _coerce_str(value, context="blocker_verdict.gate_order[]")
        for value in _require_sequence(
            blocker_verdict.get("gate_order"), context="blocker_verdict.gate_order"
        )
    )
    if blocker_gate_order != TASK11_GATE_ORDER:
        raise ValueError(
            "blocker verdict must keep task-11 gate order G0-G6, "
            + f"got {list(blocker_gate_order)}"
        )

    blocker_gate_statuses = _require_mapping(
        blocker_verdict.get("gate_statuses"), context="blocker_verdict.gate_statuses"
    )
    normalized_statuses: dict[str, str] = {}
    mismatches: list[str] = []
    for gate in TASK11_GATE_ORDER:
        blocker_status = _coerce_str(
            blocker_gate_statuses.get(gate),
            context=f"blocker_verdict.gate_statuses.{gate}",
        ).lower()
        final_status = _coerce_str(
            gate_map[gate].get("status"), context=f"gate_results.{gate}.status"
        ).lower()
        normalized_statuses[gate] = blocker_status
        if blocker_status != final_status:
            mismatches.append(f"{gate}:{blocker_status}!={final_status}")

    ready_for_task11 = _coerce_bool(
        blocker_verdict.get("ready_for_task11"),
        context="blocker_verdict.ready_for_task11",
    )
    expected_ready = all(
        normalized_statuses[gate] == "pass" for gate in TASK11_GATE_ORDER
    )
    if ready_for_task11 != expected_ready:
        mismatches.append(f"ready_for_task11:{ready_for_task11}!={expected_ready}")
    if mismatches:
        raise ValueError(
            "blocker verdict disagrees with final G0-G6 gate rows: "
            + ", ".join(mismatches)
        )

    return {
        "gate_order": list(TASK11_GATE_ORDER),
        "ready_for_task11": ready_for_task11,
        "gate_statuses": normalized_statuses,
        "decision_text": blocker_verdict.get("decision_text"),
    }


def build_task11_verification_prerequisite(
    task11_verification: Mapping[str, object] | None,
    *,
    current_iter_id: str,
    current_gate_statuses: Mapping[str, str],
    overlay_pinned_commit: str,
    overlay_materialization: Mapping[str, object] | None = None,
    eval_summary: Mapping[str, object] | None = None,
    repaired_gate_results: Mapping[str, object] | None = None,
    blocker_verdict: Mapping[str, object] | None = None,
    current_source_refs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if task11_verification is None:
        return {
            "passed": False,
            "evidence_present": False,
            "schema_version": None,
            "verified_path_scopes": [],
            "blocking_reasons": ["missing_explicit_task11_verification_evidence"],
            "decision_text": (
                "explicit machine-readable task-11 verification evidence is absent; overlay materialization alone cannot unlock full/paper-full wording"
            ),
        }

    def _capture_positive_str(
        raw: object,
        *,
        context: str,
        invalid_reason: str,
    ) -> str:
        try:
            return _coerce_str(raw, context=context)
        except TypeError:
            blocking_reasons.append(invalid_reason)
            return ""

    def _capture_positive_bool(
        raw: object,
        *,
        context: str,
        invalid_reason: str,
    ) -> bool:
        try:
            return _coerce_bool(raw, context=context)
        except TypeError:
            blocking_reasons.append(invalid_reason)
            return False

    def _capture_positive_mapping(
        raw: object,
        *,
        context: str,
        invalid_reason: str,
    ) -> Mapping[str, object]:
        try:
            return _require_mapping(raw, context=context)
        except TypeError:
            blocking_reasons.append(invalid_reason)
            return {}

    def _capture_positive_string_mapping_values(
        raw: Mapping[str, object],
        *,
        value_context_prefix: str,
        invalid_reason: str,
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for key, value in raw.items():
            try:
                resolved[key] = _coerce_str(
                    value,
                    context=f"{value_context_prefix}.{key}",
                )
            except TypeError:
                blocking_reasons.append(invalid_reason)
                return {}
        return resolved

    schema_version = _coerce_str(
        task11_verification.get("schema_version"),
        context="task11_verification.schema_version",
    )
    task11_verified = _coerce_bool(
        task11_verification.get("task11_verified"),
        context="task11_verification.task11_verified",
    )
    verification_mode = _coerce_str(
        task11_verification.get("verification_mode", "synthetic_only"),
        context="task11_verification.verification_mode",
    )
    verified_path_scopes = [
        _coerce_str(value, context="task11_verification.verified_path_scopes[]")
        for value in _require_sequence(
            task11_verification.get("verified_path_scopes"),
            context="task11_verification.verified_path_scopes",
        )
    ]
    provided_blocking_reasons = [
        _coerce_str(value, context="task11_verification.blocking_reasons[]")
        for value in _require_sequence(
            task11_verification.get("blocking_reasons", []),
            context="task11_verification.blocking_reasons",
        )
    ]
    state_side_unlock_permitted = _coerce_bool(
        task11_verification.get("state_side_unlock_permitted", False),
        context="task11_verification.state_side_unlock_permitted",
    )
    artifact_kind = task11_verification.get("artifact_kind")
    checkpoint_source = task11_verification.get("checkpoint_source")
    overlay_materialization_digest = task11_verification.get(
        "overlay_materialization_digest"
    )
    authority_input_digests = task11_verification.get("authority_input_digests")
    loads_real_checkpoint = task11_verification.get("loads_real_checkpoint")
    executes_real_model_forward = task11_verification.get("executes_real_model_forward")
    training_route_verified = task11_verification.get("training_route_verified")
    policy_runtime_route_verified = task11_verification.get(
        "policy_runtime_route_verified"
    )
    conditioned_path_verified = task11_verification.get("conditioned_path_verified")
    unconditioned_path_verified = task11_verification.get("unconditioned_path_verified")
    cfg_path_verified = task11_verification.get("cfg_path_verified")
    blocking_reasons: list[str] = []
    if schema_version != TASK11_VERIFICATION_SCHEMA_VERSION:
        blocking_reasons.append("unexpected_task11_verification_schema")
    if not task11_verified:
        blocking_reasons.append("task11_not_verified")
    expected_scopes = set(TASK11_REQUIRED_VERIFIED_PATH_SCOPES)
    observed_scopes = set(verified_path_scopes)
    if observed_scopes != expected_scopes:
        if "full_path" not in observed_scopes:
            blocking_reasons.append("full_path_not_verified")
        if "paper_full_path" not in observed_scopes:
            blocking_reasons.append("paper_full_path_not_verified")
        extra_scopes = sorted(observed_scopes - expected_scopes)
        if extra_scopes:
            blocking_reasons.append("unexpected_verified_path_scopes")

    if task11_verified:
        expected_run_id = None
        if eval_summary is not None:
            try:
                expected_run_id = _derive_task11_current_run_id(eval_summary)
            except (TypeError, ValueError, FileNotFoundError):
                blocking_reasons.append("current_run_id_binding_unavailable")
        artifact_kind = _capture_positive_str(
            artifact_kind,
            context="task11_verification.artifact_kind",
            invalid_reason="missing_or_invalid_task11_artifact_kind",
        )
        run_id = _capture_positive_str(
            task11_verification.get("run_id"),
            context="task11_verification.run_id",
            invalid_reason="missing_or_invalid_run_id",
        )
        checkpoint_source = _capture_positive_str(
            checkpoint_source,
            context="task11_verification.checkpoint_source",
            invalid_reason="missing_or_invalid_checkpoint_source",
        )
        overlay_materialization_digest = _capture_positive_str(
            overlay_materialization_digest,
            context="task11_verification.overlay_materialization_digest",
            invalid_reason="missing_or_invalid_overlay_materialization_digest",
        )
        authority_input_digests_raw = _capture_positive_mapping(
            authority_input_digests,
            context="task11_verification.authority_input_digests",
            invalid_reason="missing_or_invalid_authority_input_digests",
        )
        normalized_authority_input_digests = _capture_positive_string_mapping_values(
            authority_input_digests_raw,
            value_context_prefix="task11_verification.authority_input_digests",
            invalid_reason="missing_or_invalid_authority_input_digests",
        )
        loads_real_checkpoint = _capture_positive_bool(
            loads_real_checkpoint,
            context="task11_verification.loads_real_checkpoint",
            invalid_reason="missing_or_invalid_loads_real_checkpoint",
        )
        executes_real_model_forward = _capture_positive_bool(
            executes_real_model_forward,
            context="task11_verification.executes_real_model_forward",
            invalid_reason="missing_or_invalid_executes_real_model_forward",
        )
        training_route_verified = _capture_positive_bool(
            training_route_verified,
            context="task11_verification.training_route_verified",
            invalid_reason="missing_or_invalid_training_route_verified",
        )
        policy_runtime_route_verified = _capture_positive_bool(
            policy_runtime_route_verified,
            context="task11_verification.policy_runtime_route_verified",
            invalid_reason="missing_or_invalid_policy_runtime_route_verified",
        )
        conditioned_path_verified = _capture_positive_bool(
            conditioned_path_verified,
            context="task11_verification.conditioned_path_verified",
            invalid_reason="missing_or_invalid_conditioned_path_verified",
        )
        unconditioned_path_verified = _capture_positive_bool(
            unconditioned_path_verified,
            context="task11_verification.unconditioned_path_verified",
            invalid_reason="missing_or_invalid_unconditioned_path_verified",
        )
        cfg_path_verified = _capture_positive_bool(
            cfg_path_verified,
            context="task11_verification.cfg_path_verified",
            invalid_reason="missing_or_invalid_cfg_path_verified",
        )

        if verification_mode != "current_run_bound":
            blocking_reasons.append("verification_mode_not_current_run_bound")
        if artifact_kind != "task11_current_run_verification":
            blocking_reasons.append("unexpected_task11_artifact_kind")
        verified_iter_id = _capture_positive_str(
            task11_verification.get("iter_id"),
            context="task11_verification.iter_id",
            invalid_reason="missing_or_invalid_iter_id",
        )
        if verified_iter_id != current_iter_id:
            blocking_reasons.append("iter_id_mismatch")
        if expected_run_id is not None and run_id != expected_run_id:
            blocking_reasons.append("run_id_mismatch")
        verified_overlay_pinned_commit = _capture_positive_str(
            task11_verification.get("overlay_pinned_commit"),
            context="task11_verification.overlay_pinned_commit",
            invalid_reason="missing_or_invalid_overlay_pinned_commit",
        )
        if verified_overlay_pinned_commit != overlay_pinned_commit:
            blocking_reasons.append("overlay_pinned_commit_mismatch")
        verified_gate_statuses_raw = _capture_positive_mapping(
            task11_verification.get("verified_gate_statuses"),
            context="task11_verification.verified_gate_statuses",
            invalid_reason="missing_or_invalid_verified_gate_statuses",
        )
        verified_gate_statuses = {
            gate: value.lower()
            for gate, value in _capture_positive_string_mapping_values(
                verified_gate_statuses_raw,
                value_context_prefix="task11_verification.verified_gate_statuses",
                invalid_reason="missing_or_invalid_verified_gate_statuses",
            ).items()
        }
        if verified_gate_statuses != dict(current_gate_statuses):
            blocking_reasons.append("verified_gate_statuses_mismatch")
        if not checkpoint_source:
            blocking_reasons.append("missing_checkpoint_source")
        if not loads_real_checkpoint:
            blocking_reasons.append("real_checkpoint_not_loaded")
        if not executes_real_model_forward:
            blocking_reasons.append("real_model_forward_not_executed")
        if not training_route_verified:
            blocking_reasons.append("training_runtime_route_not_verified")
        if not policy_runtime_route_verified:
            blocking_reasons.append("policy_runtime_route_not_verified")
        if not conditioned_path_verified:
            blocking_reasons.append("conditioned_path_not_verified")
        if not unconditioned_path_verified:
            blocking_reasons.append("unconditioned_path_not_verified")
        if not cfg_path_verified:
            blocking_reasons.append("cfg_path_not_verified")

        current_source_paths = _coerce_optional_path_mapping(
            current_source_refs,
            context="current_source_refs",
        )
        task11_source_refs_raw = _capture_positive_mapping(
            task11_verification.get("source_refs"),
            context="task11_verification.source_refs",
            invalid_reason="missing_or_invalid_task11_source_refs",
        )
        task11_source_refs: dict[str, str] = {}
        if task11_source_refs_raw:
            for key, value in task11_source_refs_raw.items():
                try:
                    task11_source_refs[
                        _coerce_str(key, context="task11_verification.source_refs key")
                    ] = str(
                        _coerce_path(
                            value,
                            context=f"task11_verification.source_refs.{key}",
                        )
                    )
                except TypeError:
                    blocking_reasons.append("missing_or_invalid_task11_source_refs")
                    task11_source_refs = {}
                    break

        if any(value is not None for value in current_source_paths.values()):
            for key in TASK11_CURRENT_RUN_SOURCE_REF_KEYS:
                expected_source_path = current_source_paths[key]
                observed_source_path = task11_source_refs.get(key)
                if expected_source_path is None:
                    continue
                if observed_source_path is None:
                    blocking_reasons.append(f"missing_{key}_source_ref_binding")
                elif observed_source_path != expected_source_path:
                    blocking_reasons.append(f"{key}_source_ref_mismatch")

            expected_output_tree = None
            if overlay_materialization is not None:
                expected_output_tree = str(
                    _coerce_path(
                        overlay_materialization.get("output_tree"),
                        context="overlay_materialization.output_tree",
                    )
                )
            observed_output_tree = task11_source_refs.get("materialized_openpi_tree")
            if expected_output_tree is not None:
                if observed_output_tree is None:
                    blocking_reasons.append(
                        "missing_materialized_openpi_tree_source_ref"
                    )
                elif observed_output_tree != expected_output_tree:
                    blocking_reasons.append(
                        "materialized_openpi_tree_source_ref_mismatch"
                    )

        expected_input_digests: dict[str, str] = {}
        if overlay_materialization is not None:
            expected_input_digests["overlay_materialization"] = canonical_json_sha256(
                overlay_materialization
            )
        if eval_summary is not None:
            expected_input_digests["eval_summary"] = canonical_json_sha256(eval_summary)
        if repaired_gate_results is not None:
            expected_input_digests["repaired_gate_results"] = canonical_json_sha256(
                repaired_gate_results
            )
        if blocker_verdict is not None:
            expected_input_digests["blocker_verdict"] = canonical_json_sha256(
                blocker_verdict
            )

        expected_overlay_digest = expected_input_digests.get("overlay_materialization")
        if (
            expected_overlay_digest is not None
            and overlay_materialization_digest != expected_overlay_digest
        ):
            blocking_reasons.append("overlay_materialization_digest_mismatch")

        for digest_name, expected_digest in expected_input_digests.items():
            observed_digest = normalized_authority_input_digests.get(digest_name)
            if observed_digest is None:
                blocking_reasons.append(f"missing_{digest_name}_digest_binding")
            elif observed_digest != expected_digest:
                blocking_reasons.append(f"{digest_name}_digest_mismatch")
        if not state_side_unlock_permitted:
            blocking_reasons.append("state_side_unlock_not_permitted")
        if provided_blocking_reasons:
            blocking_reasons.append("verification_contains_blocking_reasons")
    else:
        if verification_mode != "explicit_negative_overlay_smoke_only":
            blocking_reasons.append("verification_mode_not_negative_smoke")
        if not provided_blocking_reasons:
            blocking_reasons.append("missing_negative_blocking_reasons")
        else:
            blocking_reasons.extend(provided_blocking_reasons)

    normalized_blocking_reasons: list[str] = []
    for reason in blocking_reasons:
        if reason not in normalized_blocking_reasons:
            normalized_blocking_reasons.append(reason)

    passed = not normalized_blocking_reasons
    if passed:
        decision_text = "explicit task-11 verification evidence is present; final review may consume this prerequisite, but wording/state-side stay locked until approvals complete"
    else:
        decision_text = "explicit task-11 verification evidence is present but insufficient to unlock full/paper-full wording"
    return {
        "passed": passed,
        "evidence_present": True,
        "schema_version": schema_version,
        "task11_verified": task11_verified,
        "verification_mode": verification_mode,
        "run_id": task11_verification.get("run_id"),
        "verified_path_scopes": verified_path_scopes,
        "state_side_unlock_permitted": state_side_unlock_permitted,
        "artifact_kind": artifact_kind,
        "checkpoint_source": checkpoint_source,
        "overlay_materialization_digest": overlay_materialization_digest,
        "authority_input_digests": authority_input_digests,
        "blocking_reasons": normalized_blocking_reasons,
        "decision_text": decision_text,
    }


def build_final_wording_rule(
    *,
    all_final_gates_passed: bool,
    task11_overlay_passed: bool,
    task11_verification_passed: bool,
    reviewer_approvals_passed: bool,
    audit_complete: bool,
) -> dict[str, object]:
    if (
        all_final_gates_passed
        and task11_overlay_passed
        and task11_verification_passed
        and reviewer_approvals_passed
        and audit_complete
    ):
        return {
            "rule_mode": WORDING_RULE_FULL_AND_PAPER_FULL_ALLOWED,
            "all_final_gates_passed": True,
            "task11_overlay_passed": True,
            "task11_verification_passed": True,
            "reviewer_approvals_passed": True,
            "audit_complete": True,
            "allowed_path_scopes": [
                "repaired_path",
                "full_path",
                "paper_full_path",
            ],
            "forbidden_path_scopes": [],
            "decision_text": (
                "task 11 has explicit machine-readable verification evidence, so repaired/full/paper-full wording is allowed"
            ),
        }
    return {
        "rule_mode": WORDING_RULE_REPAIRED_PATH_ONLY,
        "all_final_gates_passed": all_final_gates_passed,
        "task11_overlay_passed": task11_overlay_passed,
        "task11_verification_passed": task11_verification_passed,
        "reviewer_approvals_passed": reviewer_approvals_passed,
        "audit_complete": audit_complete,
        "allowed_path_scopes": ["repaired_path"],
        "forbidden_path_scopes": ["full_path", "paper_full_path"],
        "decision_text": (
            "final review is not fully approved yet, so external wording must stay on repaired-path phrasing only"
        ),
    }


def build_state_side_freeze(
    gate_rows: Sequence[Mapping[str, object]],
    *,
    task11_overlay_passed: bool,
    task11_verification_passed: bool,
    reviewer_approvals_passed: bool,
    audit_complete: bool,
) -> dict[str, object]:
    gate_map: dict[str, Mapping[str, object]] = {}
    gate_statuses: dict[str, str] = {}
    for row in gate_rows:
        mapping = _require_mapping(row, context="gate_rows[]")
        gate = _coerce_str(mapping.get("gate"), context="gate_rows[].gate")
        status = _coerce_str(mapping.get("status"), context=f"gate_rows.{gate}.status")
        gate_map[gate] = mapping
        gate_statuses[gate] = status.lower()

    missing_required_gates = [
        gate for gate in REPAIRED_GATE_ORDER if gate not in gate_map
    ]
    non_pass_required_gates = [
        gate for gate in REPAIRED_GATE_ORDER if gate_statuses.get(gate) != "pass"
    ]
    freeze_reason_codes: list[str] = []
    if missing_required_gates or non_pass_required_gates:
        freeze_reason_codes.append("required_gate_not_passed")
    if not task11_overlay_passed:
        freeze_reason_codes.append("task11_overlay_not_passed")
    if not task11_verification_passed:
        freeze_reason_codes.append("task11_verification_not_passed")
    if not reviewer_approvals_passed:
        freeze_reason_codes.append("reviewer_approvals_not_passed")
    if not audit_complete:
        freeze_reason_codes.append("audit_incomplete")

    state_side_frozen = bool(freeze_reason_codes)
    if state_side_frozen:
        decision_text = "state-side stays frozen until G0-G7 all pass, task 11 evidence is valid, reviewer approvals complete, and audit is marked complete"
    else:
        decision_text = "state-side may unfreeze because G0-G7 all pass, task 11 evidence is valid, reviewer approvals complete, and audit is marked complete"
    return {
        "required_gate_ids": list(REPAIRED_GATE_ORDER),
        "required_gate_status": "pass",
        "gate_statuses": gate_statuses,
        "missing_required_gates": missing_required_gates,
        "non_pass_required_gates": non_pass_required_gates,
        "task11_overlay_passed": task11_overlay_passed,
        "task11_verification_passed": task11_verification_passed,
        "reviewer_approvals_passed": reviewer_approvals_passed,
        "audit_complete": audit_complete,
        "state_side_frozen": state_side_frozen,
        "fail_closed": True,
        "freeze_reason_codes": freeze_reason_codes,
        "decision_text": decision_text,
    }


def build_task11_overlay_prerequisite(
    overlay_materialization: Mapping[str, object],
) -> dict[str, object]:
    schema_version = _coerce_str(
        overlay_materialization.get("schema_version"),
        context="overlay_materialization.schema_version",
    )
    pinned_commit = _coerce_str(
        overlay_materialization.get("pinned_commit"),
        context="overlay_materialization.pinned_commit",
    )
    source_tree_commit = _coerce_str(
        overlay_materialization.get("source_tree_commit"),
        context="overlay_materialization.source_tree_commit",
    )
    overlay_file_list = [
        _coerce_str(value, context="overlay_materialization.overlay_file_list[]")
        for value in _require_sequence(
            overlay_materialization.get("overlay_file_list"),
            context="overlay_materialization.overlay_file_list",
        )
    ]
    key_source_files = [
        _require_mapping(value, context="overlay_materialization.key_source_files[]")
        for value in _require_sequence(
            overlay_materialization.get("key_source_files"),
            context="overlay_materialization.key_source_files",
        )
    ]
    source_backups = [
        _require_mapping(value, context="overlay_materialization.source_backups[]")
        for value in _require_sequence(
            overlay_materialization.get("source_backups"),
            context="overlay_materialization.source_backups",
        )
    ]

    blocking_reasons: list[str] = []
    if schema_version != TASK11_OVERLAY_MATERIALIZATION_SCHEMA_VERSION:
        blocking_reasons.append("unexpected_overlay_schema")
    if pinned_commit != source_tree_commit:
        blocking_reasons.append("pinned_commit_mismatch")
    if not overlay_file_list:
        blocking_reasons.append("missing_overlay_files")
    if not key_source_files:
        blocking_reasons.append("missing_key_source_files")
    observed_overlay_files = tuple(sorted(overlay_file_list))
    if observed_overlay_files != tuple(sorted(TASK11_REQUIRED_OVERLAY_FILES)):
        blocking_reasons.append("overlay_file_list_mismatch")
    observed_key_source_files = tuple(
        sorted(
            _coerce_str(
                value.get("relative_path"),
                context="overlay_materialization.key_source_files[].relative_path",
            )
            for value in key_source_files
        )
    )
    if observed_key_source_files != tuple(sorted(TASK11_REQUIRED_KEY_SOURCE_FILES)):
        blocking_reasons.append("key_source_file_set_mismatch")
    observed_source_backups = tuple(
        sorted(
            (
                _coerce_str(
                    value.get("source_relative_path"),
                    context="overlay_materialization.source_backups[].source_relative_path",
                ),
                _coerce_str(
                    value.get("backup_relative_path"),
                    context="overlay_materialization.source_backups[].backup_relative_path",
                ),
            )
            for value in source_backups
        )
    )
    if observed_source_backups != tuple(sorted(TASK11_REQUIRED_SOURCE_BACKUPS)):
        blocking_reasons.append("source_backups_mismatch")

    passed = not blocking_reasons
    if passed:
        decision_text = "task 11 overlay materialization is pinned and present, but explicit task-11 verification evidence is still required before any wording unlock"
    else:
        decision_text = "task 11 overlay materialization is not fully verified, wording must stay on repaired path only"
    return {
        "passed": passed,
        "schema_version": schema_version,
        "pinned_commit": pinned_commit,
        "source_tree_commit": source_tree_commit,
        "output_tree": str(
            _coerce_path(
                overlay_materialization.get("output_tree"),
                context="overlay_materialization.output_tree",
            )
        ),
        "overlay_root": str(
            _coerce_path(
                overlay_materialization.get("overlay_root"),
                context="overlay_materialization.overlay_root",
            )
        ),
        "overlay_file_count": len(overlay_file_list),
        "key_source_file_count": len(key_source_files),
        "blocking_reasons": blocking_reasons,
        "decision_text": decision_text,
    }


def build_final_gate_current_run_binding(
    *,
    repaired_matrix_summary: Mapping[str, object],
    eval_summary: Mapping[str, object],
    repaired_gate_results: Mapping[str, object],
    overlay_materialization: Mapping[str, object],
    blocker_verdict: Mapping[str, object],
    task11_verification: Mapping[str, object] | None,
    source_refs: Mapping[str, object] | None,
) -> dict[str, object]:
    input_artifact_paths = _coerce_optional_path_mapping(
        source_refs,
        context="source_refs",
    )
    input_artifact_digests = {
        "repaired_matrix_summary": canonical_json_sha256(repaired_matrix_summary),
        "eval_summary": canonical_json_sha256(eval_summary),
        "repaired_gate_results": canonical_json_sha256(repaired_gate_results),
        "blocker_verdict": canonical_json_sha256(blocker_verdict),
        "overlay_materialization": canonical_json_sha256(overlay_materialization),
    }
    task11_verification_digest = None
    if task11_verification is not None:
        task11_verification_digest = canonical_json_sha256(task11_verification)
        input_artifact_digests["task11_verification"] = task11_verification_digest

    iter_id = _coerce_str(eval_summary.get("iter_id"), context="eval_summary.iter_id")
    run_id = None
    current_run_root = None
    eval_summary_path = input_artifact_paths["eval_summary"]
    if eval_summary_path is not None:
        current_run_root = str(Path(eval_summary_path).parent.parent)

    blocking_reasons: list[str] = []
    for key in (
        "repaired_matrix_summary",
        "eval_summary",
        "repaired_gate_results",
        "blocker_verdict",
        "overlay_materialization",
    ):
        if input_artifact_paths[key] is None:
            blocking_reasons.append(f"missing_{key}_path")

    gate_results_iter_id = _coerce_str(
        repaired_gate_results.get("iter_id"),
        context="repaired_gate_results.iter_id",
    )
    if gate_results_iter_id != iter_id:
        blocking_reasons.append("repaired_gate_results_iter_id_mismatch")

    try:
        run_id = _derive_task11_current_run_id(eval_summary)
    except (TypeError, ValueError, FileNotFoundError):
        blocking_reasons.append("current_run_id_binding_unavailable")

    if task11_verification is not None:
        observed_task11_run_id: str | None = None
        try:
            observed_task11_run_id = _coerce_str(
                task11_verification.get("run_id"),
                context="task11_verification.run_id",
            )
        except TypeError:
            blocking_reasons.append("missing_or_invalid_task11_run_id")
        if run_id is not None and observed_task11_run_id != run_id:
            blocking_reasons.append("task11_verification_run_id_mismatch")

    repaired_matrix_path = input_artifact_paths["repaired_matrix_summary"]
    repaired_matrix_summary_ref = _coerce_optional_path_str(
        eval_summary.get("repaired_matrix_summary_ref"),
        context="eval_summary.repaired_matrix_summary_ref",
    )
    if (
        repaired_matrix_path is not None
        and repaired_matrix_summary_ref != repaired_matrix_path
    ):
        blocking_reasons.append("eval_summary_repaired_matrix_ref_mismatch")

    gate_results_ref = _coerce_optional_path_str(
        eval_summary.get("gate_results_ref"),
        context="eval_summary.gate_results_ref",
    )
    current_gate_results_path = input_artifact_paths["repaired_gate_results"]
    if (
        current_gate_results_path is not None
        and gate_results_ref != current_gate_results_path
    ):
        blocking_reasons.append("eval_summary_gate_results_ref_mismatch")

    blocker_verdict_ref = _coerce_optional_path_str(
        eval_summary.get("blocker_verdict_ref"),
        context="eval_summary.blocker_verdict_ref",
    )
    current_blocker_path = input_artifact_paths["blocker_verdict"]
    if current_blocker_path is not None and blocker_verdict_ref != current_blocker_path:
        blocking_reasons.append("eval_summary_blocker_verdict_ref_mismatch")

    repaired_gate_results_matrix_ref = _coerce_optional_path_str(
        repaired_gate_results.get("repaired_matrix_summary_ref"),
        context="repaired_gate_results.repaired_matrix_summary_ref",
    )
    if (
        repaired_matrix_path is not None
        and repaired_gate_results_matrix_ref != repaired_matrix_path
    ):
        blocking_reasons.append("repaired_gate_results_matrix_ref_mismatch")

    if current_run_root is not None and Path(current_run_root).name != iter_id:
        blocking_reasons.append("current_run_root_iter_id_mismatch")

    normalized_blocking_reasons: list[str] = []
    for reason in blocking_reasons:
        if reason not in normalized_blocking_reasons:
            normalized_blocking_reasons.append(reason)

    passed = not normalized_blocking_reasons
    if passed:
        decision_text = "current-run authority paths, digests, iter id, and provenance refs are consistently bound to one run root"
    else:
        decision_text = "current-run authority binding is incomplete or inconsistent; final review surfaces remain fail-closed"
    return {
        "passed": passed,
        "fail_closed": True,
        "iter_id": iter_id,
        "run_id": run_id,
        "current_run_root": current_run_root,
        "input_artifact_paths": input_artifact_paths,
        "input_artifact_digests": input_artifact_digests,
        "task11_verification_digest": task11_verification_digest,
        "blocking_reasons": normalized_blocking_reasons,
        "decision_text": decision_text,
    }


def build_final_gate_summary(
    *,
    repaired_matrix_summary: Mapping[str, object],
    eval_summary: Mapping[str, object],
    repaired_gate_results: Mapping[str, object],
    blocker_verdict: Mapping[str, object],
    overlay_materialization: Mapping[str, object],
    task11_verification: Mapping[str, object] | None = None,
    source_refs: Mapping[str, object] | None = None,
    reviewer_approvals: Mapping[str, object] | None = None,
) -> dict[str, object]:
    current_run_binding = build_final_gate_current_run_binding(
        repaired_matrix_summary=repaired_matrix_summary,
        eval_summary=eval_summary,
        repaired_gate_results=repaired_gate_results,
        overlay_materialization=overlay_materialization,
        blocker_verdict=blocker_verdict,
        task11_verification=task11_verification,
        source_refs=source_refs,
    )
    gate_order, gate_rows, gate_map = _resolved_gate_rows(repaired_gate_results)
    blocker_consistency = _validate_task11_blocker_consistency(
        blocker_verdict, gate_map=gate_map
    )
    task11_overlay = build_task11_overlay_prerequisite(overlay_materialization)
    gate_statuses = {
        gate: _coerce_str(
            gate_map[gate].get("status"), context=f"{gate}.status"
        ).lower()
        for gate in gate_order
    }
    all_final_gates_passed = all(gate_statuses[gate] == "pass" for gate in gate_order)
    overlay_pinned_commit = _coerce_str(
        task11_overlay.get("pinned_commit"), context="task11_overlay.pinned_commit"
    )
    task11_verification_gate = build_task11_verification_prerequisite(
        task11_verification,
        current_iter_id=_coerce_str(
            eval_summary.get("iter_id"), context="eval_summary.iter_id"
        ),
        current_gate_statuses=gate_statuses,
        overlay_pinned_commit=overlay_pinned_commit,
        overlay_materialization=overlay_materialization,
        eval_summary=eval_summary,
        repaired_gate_results=repaired_gate_results,
        blocker_verdict=blocker_verdict,
        current_source_refs=source_refs,
    )
    reviewer_state = build_final_reviewer_state(
        current_run_binding_passed=_coerce_bool(
            current_run_binding.get("passed"),
            context="current_run_binding.passed",
        ),
        reviewer_approvals=reviewer_approvals,
    )
    audit_complete = reviewer_state["status"] == "approve_all"
    reviewer_approvals_passed = reviewer_state["status"] == "approve_all"
    wording_rule = build_final_wording_rule(
        all_final_gates_passed=all_final_gates_passed,
        task11_overlay_passed=_coerce_bool(
            task11_overlay.get("passed"), context="task11_overlay.passed"
        ),
        task11_verification_passed=_coerce_bool(
            task11_verification_gate.get("passed"),
            context="task11_verification_gate.passed",
        ),
        reviewer_approvals_passed=reviewer_approvals_passed,
        audit_complete=audit_complete,
    )
    state_side_freeze = build_state_side_freeze(
        gate_rows,
        task11_overlay_passed=_coerce_bool(
            task11_overlay.get("passed"), context="task11_overlay.passed"
        ),
        task11_verification_passed=_coerce_bool(
            task11_verification_gate.get("passed"),
            context="task11_verification_gate.passed",
        ),
        reviewer_approvals_passed=reviewer_approvals_passed,
        audit_complete=audit_complete,
    )
    passed_gates = [gate for gate in gate_order if gate_statuses[gate] == "pass"]
    non_pass_gates = [gate for gate in gate_order if gate_statuses[gate] != "pass"]

    repaired_matrix_schema = _coerce_str(
        repaired_matrix_summary.get("schema_version"),
        context="repaired_matrix_summary.schema_version",
    )
    eval_summary_schema = _coerce_str(
        eval_summary.get("schema_version"), context="eval_summary.schema_version"
    )
    iter_id = _coerce_str(eval_summary.get("iter_id"), context="eval_summary.iter_id")
    metric_profile_id = _coerce_str(
        eval_summary.get("metric_profile_id"), context="eval_summary.metric_profile_id"
    )

    payload: dict[str, object] = {
        "schema_version": FINAL_GATE_SUMMARY_SCHEMA_VERSION,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "iter_id": iter_id,
        "run_id": current_run_binding["run_id"],
        "metric_profile_id": metric_profile_id,
        "input_artifact_paths": dict(
            cast(Mapping[str, object], current_run_binding["input_artifact_paths"])
        ),
        "input_artifact_digests": dict(
            cast(Mapping[str, object], current_run_binding["input_artifact_digests"])
        ),
        "task11_verification_digest": current_run_binding["task11_verification_digest"],
        "final_gate_ids": list(gate_order),
        "gate_order": list(gate_order),
        "gates": [dict(row) for row in gate_rows],
        "gate_statuses": gate_statuses,
        "passed_gates": passed_gates,
        "non_pass_gates": non_pass_gates,
        "single_source_of_truth": {
            "final_gate_ids": list(REPAIRED_GATE_ORDER),
            "gate_verdict_source": "repaired_gate_results",
            "repaired_matrix_schema_version": repaired_matrix_schema,
            "eval_summary_schema_version": eval_summary_schema,
            "iter_id": iter_id,
            "run_id": current_run_binding["run_id"],
            "current_run_root": current_run_binding["current_run_root"],
            "task11_verification_source": (
                "task11_verification" if task11_verification is not None else None
            ),
        },
        "current_run_binding": current_run_binding,
        "task11_blocker_consistency": blocker_consistency,
        "task11_overlay_prerequisite": task11_overlay,
        "task11_verification_prerequisite": task11_verification_gate,
        "wording_rule": wording_rule,
        "state_side_freeze": state_side_freeze,
        "reviewer_state": reviewer_state,
        "audit_complete": audit_complete,
    }
    if source_refs is not None:
        payload["source_refs"] = dict(source_refs)
    return payload


def materialize_final_gate_summary(
    *,
    repaired_matrix_summary_path: str | Path,
    eval_summary_path: str | Path,
    overlay_materialization_path: str | Path,
    output_path: str | Path,
    task11_verification_path: str | Path | None = None,
    reviewer_approvals: Mapping[str, object] | None = None,
) -> Path:
    repaired_matrix_summary_path = Path(repaired_matrix_summary_path).resolve()
    eval_summary_path = Path(eval_summary_path).resolve()
    overlay_materialization_path = Path(overlay_materialization_path).resolve()
    output_path = Path(output_path).resolve()
    resolved_task11_verification_path = (
        Path(task11_verification_path).resolve()
        if task11_verification_path is not None
        else None
    )

    eval_summary = read_json(eval_summary_path)
    repaired_gate_results_path = _coerce_path(
        eval_summary.get("gate_results_ref"), context="eval_summary.gate_results_ref"
    )
    blocker_verdict_path = _coerce_path(
        eval_summary.get("blocker_verdict_ref"),
        context="eval_summary.blocker_verdict_ref",
    )

    summary = build_final_gate_summary(
        repaired_matrix_summary=read_json(repaired_matrix_summary_path),
        eval_summary=eval_summary,
        repaired_gate_results=read_json(repaired_gate_results_path),
        blocker_verdict=read_json(blocker_verdict_path),
        overlay_materialization=read_json(overlay_materialization_path),
        task11_verification=(
            read_json(resolved_task11_verification_path)
            if resolved_task11_verification_path is not None
            else None
        ),
        source_refs={
            "repaired_matrix_summary": str(repaired_matrix_summary_path),
            "eval_summary": str(eval_summary_path),
            "repaired_gate_results": str(repaired_gate_results_path),
            "blocker_verdict": str(blocker_verdict_path),
            "overlay_materialization": str(overlay_materialization_path),
            "task11_verification": (
                str(resolved_task11_verification_path)
                if resolved_task11_verification_path is not None
                else None
            ),
        },
        reviewer_approvals=reviewer_approvals,
    )
    write_json(output_path, summary)
    return output_path


def build_source_equivalence_report(
    *,
    materialization_report: Mapping[str, object],
    train_manifest: Mapping[str, object],
    checkpoint_provenance: Mapping[str, object],
    paired_summary: Mapping[str, object],
    rerun_output_dir: str | Path = RELABEL_CONTROL_OUTPUT_DIR,
) -> dict[str, object]:
    rerun_output_dir_path = Path(rerun_output_dir).resolve()
    rerun_checkpoint_dir = rerun_output_dir_path / "best"
    rerun_artifact_exists = (rerun_checkpoint_dir / "checkpoint.json").is_file()
    authority_final_status = _coerce_str(
        materialization_report.get("final_status"),
        context="materialization_report.final_status",
    )
    relabel_source_dir = _coerce_path(
        materialization_report.get("output_dataset_dir"),
        context="materialization_report.output_dataset_dir",
    )
    relabel_source_materialized = authority_final_status == "materialized"
    source_route_id = _coerce_str(
        materialization_report.get("route_id"),
        context="materialization_report.route_id",
    )
    existing_source_dirs = _source_dataset_dirs(train_manifest, checkpoint_provenance)
    existing_checkpoint_dir, consistency_issues = _existing_control_consistency(
        train_manifest, checkpoint_provenance, paired_summary
    )
    recap_row = _recap_summary_row(paired_summary)

    source_equivalence_proven = (
        relabel_source_materialized
        and not consistency_issues
        and bool(existing_source_dirs)
        and all(path == relabel_source_dir for path in existing_source_dirs)
    )

    if source_equivalence_proven:
        status = "reuse_existing_control"
        proof_mode = "existing_control_source_matches_materialized_relabel8d_output"
        decision_reason = "现有 recap_only control 的训练来源已经与 9C 权威 relabel8d source 完全一致，可直接复用。"
        blocker: dict[str, object] | None = None
    elif relabel_source_materialized:
        status = "rerun_required"
        proof_mode = "materialized_relabel8d_exists_but_existing_control_source_differs"
        decision_reason = "9C 已物化 relabel8d source，但现有 recap_only control 不是在该 relabel8d source 上训练出来的；按硬规则必须 rerun control。"
        blocker = None
    else:
        status = "blocked"
        proof_mode = "relabel8d_authority_missing_or_not_materialized"
        decision_reason = "缺少 source-equivalence 强证明，而且 9C relabel8d source 尚未物化；因此 control 仍需 rerun，但当前 rerun 被阻塞。"
        blocker = {
            "code": "missing_materialized_relabel8d_source",
            "reason": (
                "9C authority did not provide a materialized relabel8d source; "
                "narrow recap_only rerun cannot start until the relabeled official/native 8D dataset exists."
            ),
            "authority_final_status": authority_final_status,
            "source_route_id": source_route_id,
            "next_action": (
                "先物化 9C relabel8d source，再重新评估或执行 recap_only control rerun。"
            ),
        }

    report: dict[str, object] = {
        "schema_version": CONTROL_GATE_SCHEMA_VERSION,
        "route_id": CONTROL_GATE_ROUTE_ID,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "variant": RECAP_ONLY_VARIANT,
        "status": status,
        "decision_reason": decision_reason,
        "reuse_existing_control": bool(source_equivalence_proven),
        "rerun_control": not bool(source_equivalence_proven),
        "rerun_possible_now": bool(
            (not source_equivalence_proven) and relabel_source_materialized
        ),
        "authority": {
            "relabel_final_status": authority_final_status,
            "relabel_output_dataset_dir": str(relabel_source_dir),
            "relabel_source_materialized": relabel_source_materialized,
            "source_dataset_dir": str(
                _coerce_path(
                    materialization_report.get("source_dataset_dir"),
                    context="materialization_report.source_dataset_dir",
                )
            ),
            "source_route_id": source_route_id,
            "selected_episode_count": _coerce_int(
                materialization_report.get("selected_episode_count", 0),
                context="materialization_report.selected_episode_count",
            ),
            "selected_frame_count": _coerce_int(
                materialization_report.get("selected_frame_count", 0),
                context="materialization_report.selected_frame_count",
            ),
        },
        "existing_control": {
            "checkpoint_dir": str(existing_checkpoint_dir),
            "checkpoint_source": str(
                checkpoint_provenance.get("checkpoint_source", "")
            ),
            "train_source_dataset_dirs": [str(path) for path in existing_source_dirs],
            "paired_summary_success_rate": _coerce_float(
                recap_row.get("success_rate"),
                context="paired_summary.recap_only.success_rate",
            ),
            "paired_summary_failure_count": _coerce_int(
                recap_row.get("failure_count", 0),
                context="paired_summary.recap_only.failure_count",
            ),
            "consistency_issues": list(consistency_issues),
        },
        "source_equivalence": {
            "strongly_proven": bool(source_equivalence_proven),
            "proof_mode": proof_mode,
            "required_relabel_source_dataset_dir": str(relabel_source_dir),
            "observed_existing_source_dataset_dirs": [
                str(path) for path in existing_source_dirs
            ],
        },
        "rerun_target": {
            "output_dir": str(rerun_output_dir_path),
            "checkpoint_dir": str(rerun_checkpoint_dir),
            "dataset_dir": str(relabel_source_dir),
            "dataset_route_id": source_route_id,
            "rerun_artifact_exists": rerun_artifact_exists,
        },
    }
    if blocker is not None:
        report["blocker"] = blocker
    return report


def materialize_source_equivalence_report(
    *,
    materialization_report_path: str | Path,
    train_manifest_path: str | Path,
    checkpoint_provenance_path: str | Path,
    paired_summary_path: str | Path,
    output_dir: str | Path = RELABEL_CONTROL_OUTPUT_DIR,
) -> Path:
    report = build_source_equivalence_report(
        materialization_report=read_json(Path(materialization_report_path).resolve()),
        train_manifest=read_json(Path(train_manifest_path).resolve()),
        checkpoint_provenance=read_json(Path(checkpoint_provenance_path).resolve()),
        paired_summary=read_json(Path(paired_summary_path).resolve()),
        rerun_output_dir=output_dir,
    )
    output_dir_path = Path(output_dir).resolve()
    report_path = output_dir_path / SOURCE_EQUIVALENCE_REPORT_NAME
    write_json(report_path, report)
    return report_path


__all__ = [
    "BLOCKED_EXIT_CODE",
    "CONTROL_GATE_ROUTE_ID",
    "CONTROL_GATE_SCHEMA_VERSION",
    "CONTROL_SEMANTIC_FIELDS",
    "FINAL_GATE_SUMMARY_SCHEMA_VERSION",
    "RELABEL_CONTROL_OUTPUT_DIR",
    "REPAIRED_GATE_ORDER",
    "REPAIRED_GATE_SCHEMA_VERSION",
    "SOURCE_EQUIVALENCE_REPORT_NAME",
    "TASK11_BLOCKER_VERDICT_SCHEMA_VERSION",
    "TASK11_GATE_ORDER",
    "TASK11_OVERLAY_MATERIALIZATION_SCHEMA_VERSION",
    "TASK11_VERIFICATION_SCHEMA_VERSION",
    "WORDING_RULE_FULL_AND_PAPER_FULL_ALLOWED",
    "WORDING_RULE_REPAIRED_PATH_ONLY",
    "canonical_json_sha256",
    "build_final_gate_summary",
    "build_final_wording_rule",
    "build_repaired_headline_results",
    "build_source_equivalence_report",
    "build_state_side_freeze",
    "build_task11_overlay_prerequisite",
    "build_task11_verification_prerequisite",
    "build_repaired_gate_rows",
    "build_task11_blocker_verdict",
    "evaluate_control_semantics_gate",
    "evaluate_informativeness_gate",
    "evaluate_iteration_value_gate",
    "evaluate_metric_comparison_gate",
    "evaluate_recap_gate",
    "evaluate_runtime_cfg_gate",
    "materialize_final_gate_summary",
    "materialize_source_equivalence_report",
]
