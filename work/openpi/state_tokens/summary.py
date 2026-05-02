from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import cast

from work.openpi.recap.checkpoint import read_json
from work.openpi.recap.protocol import FrozenComparisonManifest
from work.openpi.recap.summary import (
    build_recap_only_summary as build_base_recap_only_summary,
    build_stock_summary as build_base_stock_summary,
    validate_summary_fields as validate_base_summary_fields,
)

from .protocol import (
    BLOCKER_CODE_CONTROL_PARITY_NOT_SATISFIED,
    BLOCKER_CODE_INVALID_CONTROL_PARITY_REFERENCE,
    BLOCKER_CODE_INVALID_NATIVE_PROVENANCE,
    BLOCKER_CODE_MISSING_CONTROL_PARITY_ARTIFACT,
    DEFAULT_RECAP_ONLY_CONTROL_GATE_REPORT,
    NOT_APPLICABLE_STATE_TOKEN_ROUTE,
    OFFICIAL_NATIVE_DATASET_NAME,
    OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME,
    OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID,
    PAIRED_SUMMARY_SCHEMA_VERSION,
    REQUIRED_NATIVE_STATE_DIM,
    RECAP_STATE_TOKENS_VARIANT,
    SOURCE_STATE,
    StateTokenContractError,
    STATE_TOKEN_ROUTE,
    SUMMARY_SCHEMA_VERSION,
    build_blocker_report,
)


SUMMARY_FIELDS: tuple[str, ...] = (
    "variant",
    "checkpoint_source",
    "checkpoint_dir",
    "suite",
    "task_ids",
    "seed_manifest",
    "num_trials_per_task",
    "episode_count",
    "success_rate",
    "failure_count",
    "deviation_notes",
    "state_token_route",
)


def _require_mapping(raw: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


def _require_sequence(raw: object, *, context: str) -> Sequence[object]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(f"{context} must be a sequence")
    return cast(Sequence[object], raw)


def _coerce_float_like(raw: object, *, context: str) -> float:
    if isinstance(raw, bool) or raw is None:
        raise TypeError(f"{context} must be float-like, got {raw!r}")
    if not isinstance(raw, (int, float, str)):
        raise TypeError(f"{context} must be float-like, got {type(raw).__name__}")
    return float(raw)


def _normalize_rate(raw: object, *, context: str) -> float:
    value = _coerce_float_like(raw, context=context)
    if math.isnan(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{context} must be within [0, 1], got {value!r}")
    return float(value)


def _coerce_int_like(raw: object, *, context: str) -> int:
    if isinstance(raw, bool) or raw is None:
        raise TypeError(f"{context} must be int-like, got {raw!r}")
    if not isinstance(raw, (int, float, str)):
        raise TypeError(f"{context} must be int-like, got {type(raw).__name__}")
    return int(raw)


def _coerce_path(raw: object, *, context: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise TypeError(f"{context} must be a non-empty path string")
    return Path(raw).resolve()


def _coerce_text(raw: object, *, context: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return raw.strip()


def _coerce_bool(raw: object, *, context: str) -> bool:
    if not isinstance(raw, bool):
        raise TypeError(f"{context} must be bool, got {type(raw).__name__}")
    return bool(raw)


def _failure_count(*, episode_count: int, success_rate: float) -> int:
    successes = int(round(float(success_rate) * float(episode_count)))
    return max(0, int(episode_count) - successes)


def validate_summary_fields(summary: Mapping[str, object]) -> dict[str, object]:
    missing = [field for field in SUMMARY_FIELDS if field not in summary]
    if missing:
        raise ValueError(f"summary is missing required fields: {missing!r}")
    base = validate_base_summary_fields(summary)
    state_token_route = str(summary["state_token_route"])
    if not state_token_route:
        raise ValueError("summary.state_token_route must be a non-empty string")
    return {
        **base,
        "state_token_route": state_token_route,
    }


def _control_parity_error(
    *,
    stage: str,
    blocker_code: str,
    reason: str,
    subject_path: Path,
    control_gate_report_path: Path,
    extra_payload: dict[str, object] | None = None,
) -> StateTokenContractError:
    payload = build_blocker_report(
        stage=stage,
        blocker_code=blocker_code,
        reason=reason,
        checkpoint_dir=subject_path,
        next_action=(
            "Wait until Task 9B control parity is satisfied (or its required relabel8d rerun artifact exists), then rerun Task 9D."
        ),
        extra_payload={
            "control_gate_report_path": str(control_gate_report_path),
            "control_parity_required": True,
            **({} if extra_payload is None else extra_payload),
        },
    )
    return StateTokenContractError(reason, payload=payload)


def require_control_parity_ready(
    subject_path: str | Path,
    *,
    control_gate_report_path: str | Path | None = None,
    stage: str = "eval_preflight",
) -> dict[str, object]:
    subject_path_resolved = Path(subject_path).resolve()
    report_path = (
        DEFAULT_RECAP_ONLY_CONTROL_GATE_REPORT.resolve()
        if control_gate_report_path is None
        else Path(control_gate_report_path).resolve()
    )
    if not report_path.is_file():
        reason = "Task 9D three-way comparison requires the Task 9B source_equivalence_report.json artifact; eval must not bypass that gate silently."
        raise _control_parity_error(
            stage=stage,
            blocker_code=BLOCKER_CODE_MISSING_CONTROL_PARITY_ARTIFACT,
            reason=reason,
            subject_path=subject_path_resolved,
            control_gate_report_path=report_path,
        )

    report = read_json(report_path)
    status = _coerce_text(report.get("status", ""), context="control_gate.status")
    decision_reason = _coerce_text(
        report.get("decision_reason", ""), context="control_gate.decision_reason"
    )
    existing_control = _require_mapping(
        report.get("existing_control", {}), context="control_gate.existing_control"
    )
    rerun_target = _require_mapping(
        report.get("rerun_target", {}), context="control_gate.rerun_target"
    )
    rerun_control = _coerce_bool(
        report.get("rerun_control", False), context="control_gate.rerun_control"
    )
    rerun_possible_now = _coerce_bool(
        report.get("rerun_possible_now", False),
        context="control_gate.rerun_possible_now",
    )

    if status == "reuse_existing_control":
        authorized_checkpoint_dir = _coerce_path(
            existing_control.get("checkpoint_dir"),
            context="control_gate.existing_control.checkpoint_dir",
        )
    elif status == "blocked":
        blocker_payload = _require_mapping(
            report.get("blocker", {}), context="control_gate.blocker"
        )
        reason = "Task 9D is blocked because Task 9B control parity is still blocked; state-token ablation must not run until the control gate is satisfied."
        raise _control_parity_error(
            stage=stage,
            blocker_code=BLOCKER_CODE_CONTROL_PARITY_NOT_SATISFIED,
            reason=reason,
            subject_path=subject_path_resolved,
            control_gate_report_path=report_path,
            extra_payload={
                "control_gate_status": status,
                "control_gate_decision_reason": decision_reason,
                "rerun_control": rerun_control,
                "rerun_possible_now": rerun_possible_now,
                "task9b_blocker": dict(blocker_payload),
            },
        )
    elif status == "rerun_required":
        authorized_checkpoint_dir = _coerce_path(
            rerun_target.get("checkpoint_dir"),
            context="control_gate.rerun_target.checkpoint_dir",
        )
        if not (authorized_checkpoint_dir / "checkpoint.json").is_file():
            reason = "Task 9B control gate requires a relabel8d recap_only rerun checkpoint, but the authorized rerun artifact is still missing."
            raise _control_parity_error(
                stage=stage,
                blocker_code=BLOCKER_CODE_INVALID_CONTROL_PARITY_REFERENCE,
                reason=reason,
                subject_path=subject_path_resolved,
                control_gate_report_path=report_path,
                extra_payload={
                    "control_gate_status": status,
                    "authorized_checkpoint_dir": str(authorized_checkpoint_dir),
                    "rerun_control": rerun_control,
                    "rerun_possible_now": rerun_possible_now,
                },
            )
    else:
        reason = f"unsupported Task 9B control gate status {status!r}"
        raise _control_parity_error(
            stage=stage,
            blocker_code=BLOCKER_CODE_INVALID_CONTROL_PARITY_REFERENCE,
            reason=reason,
            subject_path=subject_path_resolved,
            control_gate_report_path=report_path,
            extra_payload={
                "control_gate_status": status,
                "rerun_control": rerun_control,
                "rerun_possible_now": rerun_possible_now,
            },
        )

    control_reference_mode = (
        "reuse_existing_control"
        if status == "reuse_existing_control"
        else "rerun_required_checkpoint"
    )

    if status == "reuse_existing_control":
        deviation_notes = [
            "Task 9B control gate 已满足；comparison-tier control reference 直接复用既有 recap_only control。",
            f"9B status: {status}",
            f"9B decision_reason: {decision_reason}",
        ]
    else:
        deviation_notes = [
            "Task 9B control gate 已满足；comparison-tier control reference 使用 gate 要求的 relabel8d rerun checkpoint。",
            f"9B status: {status}",
            f"9B decision_reason: {decision_reason}",
        ]

    return {
        "control_gate_report_path": str(report_path),
        "status": status,
        "decision_reason": decision_reason,
        "authorized_checkpoint_dir": str(authorized_checkpoint_dir),
        "control_reference_mode": control_reference_mode,
        "rerun_control": rerun_control,
        "rerun_possible_now": rerun_possible_now,
        "deviation_notes": deviation_notes,
    }


def resolve_control_parity_reference(
    checkpoint_dir: str | Path,
    *,
    control_gate_report_path: str | Path | None = None,
    stage: str = "eval_preflight",
) -> dict[str, object]:
    provided_checkpoint_dir = Path(checkpoint_dir).resolve()
    control_parity = require_control_parity_ready(
        provided_checkpoint_dir,
        control_gate_report_path=control_gate_report_path,
        stage=stage,
    )
    control_gate_report_resolved = _coerce_path(
        control_parity.get("control_gate_report_path"),
        context="control_parity.control_gate_report_path",
    )
    authorized_checkpoint_dir = _coerce_path(
        control_parity.get("authorized_checkpoint_dir"),
        context="control_parity.authorized_checkpoint_dir",
    )

    if provided_checkpoint_dir != authorized_checkpoint_dir:
        reason = "recap_only checkpoint for Task 9D eval does not match the checkpoint authorized by Task 9B control parity."
        raise _control_parity_error(
            stage=stage,
            blocker_code=BLOCKER_CODE_INVALID_CONTROL_PARITY_REFERENCE,
            reason=reason,
            subject_path=provided_checkpoint_dir,
            control_gate_report_path=control_gate_report_resolved,
            extra_payload={
                "control_gate_status": str(control_parity["status"]),
                "authorized_checkpoint_dir": str(authorized_checkpoint_dir),
                "provided_checkpoint_dir": str(provided_checkpoint_dir),
            },
        )

    return {
        **control_parity,
        "authorized_checkpoint_dir": str(authorized_checkpoint_dir),
        "provided_checkpoint_dir": str(provided_checkpoint_dir),
    }


def build_stock_comparison_summary(
    stock_source: str | Path | None,
    *,
    manifest: FrozenComparisonManifest,
) -> dict[str, object]:
    summary = build_base_stock_summary(stock_source, manifest=manifest)
    return validate_summary_fields(
        {
            **summary,
            "state_token_route": NOT_APPLICABLE_STATE_TOKEN_ROUTE,
        }
    )


def build_recap_only_comparison_summary(
    checkpoint_dir: str | Path,
    *,
    manifest: FrozenComparisonManifest,
    control_gate_report_path: str | Path | None = None,
) -> dict[str, object]:
    control_parity = resolve_control_parity_reference(
        checkpoint_dir,
        control_gate_report_path=control_gate_report_path,
    )
    summary = build_base_recap_only_summary(
        str(control_parity["authorized_checkpoint_dir"]),
        manifest=manifest,
    )
    deviation_notes = [
        *cast(list[str], summary["deviation_notes"]),
        *cast(list[str], control_parity["deviation_notes"]),
    ]
    return validate_summary_fields(
        {
            **summary,
            "checkpoint_dir": str(control_parity["authorized_checkpoint_dir"]),
            "deviation_notes": deviation_notes,
            "state_token_route": NOT_APPLICABLE_STATE_TOKEN_ROUTE,
        }
    )


def build_state_token_summary(
    checkpoint_dir: str | Path,
    *,
    manifest: FrozenComparisonManifest,
) -> dict[str, object]:
    checkpoint_dir_path = Path(checkpoint_dir).resolve()
    checkpoint_payload = read_json(checkpoint_dir_path / "checkpoint.json")
    checkpoint_provenance = read_json(
        checkpoint_dir_path / "checkpoint_provenance.json"
    )
    variant_derivation = _require_mapping(
        checkpoint_provenance.get("variant_derivation", {}),
        context="checkpoint_provenance.variant_derivation",
    )
    state_token_route = str(
        variant_derivation.get(
            "state_token_route",
            checkpoint_payload.get(
                "state_token_route", checkpoint_provenance.get("state_token_route", "")
            ),
        )
    )
    if state_token_route != STATE_TOKEN_ROUTE:
        raise ValueError(
            "Task 9 requires checkpoint provenance to record "
            + f"state_token_route={STATE_TOKEN_ROUTE!r}, got {state_token_route!r}"
        )
    observed_dataset_state_dim = _coerce_int_like(
        variant_derivation.get("observed_dataset_state_dim"),
        context="checkpoint_provenance.variant_derivation.observed_dataset_state_dim",
    )
    source_dataset_name = _coerce_text(
        variant_derivation.get("source_dataset_name", ""),
        context="checkpoint_provenance.variant_derivation.source_dataset_name",
    )
    source_dataset_route_id = _coerce_text(
        variant_derivation.get("source_dataset_route_id", ""),
        context="checkpoint_provenance.variant_derivation.source_dataset_route_id",
    )
    official_source_dataset_name = _coerce_text(
        variant_derivation.get("official_native_source_dataset_name", ""),
        context="checkpoint_provenance.variant_derivation.official_native_source_dataset_name",
    )
    if observed_dataset_state_dim != REQUIRED_NATIVE_STATE_DIM:
        reason = (
            f"checkpoint provenance claims {STATE_TOKEN_ROUTE} but observed_dataset_state_dim="
            + f"{observed_dataset_state_dim!r}; contract requires source state = {SOURCE_STATE}."
        )
        raise StateTokenContractError(
            reason,
            payload=build_blocker_report(
                stage="eval_preflight",
                blocker_code=BLOCKER_CODE_INVALID_NATIVE_PROVENANCE,
                reason=reason,
                checkpoint_dir=checkpoint_dir_path,
                checkpoint_provenance_path=checkpoint_dir_path
                / "checkpoint_provenance.json",
                observed_dataset_state_dim=observed_dataset_state_dim,
            ),
        )
    if source_dataset_name != OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME:
        reason = (
            "Task 9D checkpoint provenance must point to the relabeled official/native 8D training source, "
            + f"got source_dataset_name={source_dataset_name!r}."
        )
        raise StateTokenContractError(
            reason,
            payload=build_blocker_report(
                stage="eval_preflight",
                blocker_code=BLOCKER_CODE_INVALID_NATIVE_PROVENANCE,
                reason=reason,
                checkpoint_dir=checkpoint_dir_path,
                checkpoint_provenance_path=checkpoint_dir_path
                / "checkpoint_provenance.json",
                observed_dataset_state_dim=observed_dataset_state_dim,
            ),
        )
    if source_dataset_route_id != OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID:
        reason = (
            "Task 9D checkpoint provenance must preserve route_id=official_native_8d_recap_relabels_v1, "
            + f"got {source_dataset_route_id!r}."
        )
        raise StateTokenContractError(
            reason,
            payload=build_blocker_report(
                stage="eval_preflight",
                blocker_code=BLOCKER_CODE_INVALID_NATIVE_PROVENANCE,
                reason=reason,
                checkpoint_dir=checkpoint_dir_path,
                checkpoint_provenance_path=checkpoint_dir_path
                / "checkpoint_provenance.json",
                observed_dataset_state_dim=observed_dataset_state_dim,
            ),
        )
    if official_source_dataset_name != OFFICIAL_NATIVE_DATASET_NAME:
        reason = (
            "Task 9D checkpoint provenance must preserve official/native LIBERO source attribution, "
            + f"got official_native_source_dataset_name={official_source_dataset_name!r}."
        )
        raise StateTokenContractError(
            reason,
            payload=build_blocker_report(
                stage="eval_preflight",
                blocker_code=BLOCKER_CODE_INVALID_NATIVE_PROVENANCE,
                reason=reason,
                checkpoint_dir=checkpoint_dir_path,
                checkpoint_provenance_path=checkpoint_dir_path
                / "checkpoint_provenance.json",
                observed_dataset_state_dim=observed_dataset_state_dim,
            ),
        )
    success_rate = _normalize_rate(
        checkpoint_payload.get("offline_success_proxy", 0.0),
        context="checkpoint.offline_success_proxy",
    )
    deviation_notes = [
        "当前 Task 9D 评测仍是离线 comparative summary；success_rate 来自 checkpoint 中的 offline proxy，而不是 MuJoCo rollout。",
        "训练源固定为 physical_intelligence_libero_official_8d_recap_relabels_v1，并保持 route_id=official_native_8d_recap_relabels_v1 与 official/native 8D provenance。",
        "state_token_route=native_discrete_state_input_v1 仅表示原生 discrete_state_input=True carrier provenance；当前分支不包含 symbolic phase token、RL token、next-state head、自定义 vocabulary 或第二 tokenizer。",
    ]
    summary = {
        "variant": RECAP_STATE_TOKENS_VARIANT,
        "checkpoint_source": str(checkpoint_provenance["checkpoint_source"]),
        "checkpoint_dir": str(checkpoint_dir_path),
        "suite": manifest.suite,
        "task_ids": [int(value) for value in manifest.task_ids],
        "seed_manifest": [int(value) for value in manifest.seed_manifest],
        "num_trials_per_task": int(manifest.num_trials_per_task),
        "episode_count": int(manifest.episode_count),
        "success_rate": float(success_rate),
        "failure_count": _failure_count(
            episode_count=manifest.episode_count,
            success_rate=success_rate,
        ),
        "deviation_notes": deviation_notes,
        "state_token_route": state_token_route,
    }
    return validate_summary_fields(summary)


def build_three_way_paired_summary(
    *,
    stock_summary: Mapping[str, object],
    recap_only_summary: Mapping[str, object],
    state_token_summary: Mapping[str, object],
    control_parity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    stock = validate_summary_fields(stock_summary)
    recap_only = validate_summary_fields(recap_only_summary)
    state_token = validate_summary_fields(state_token_summary)
    payload: dict[str, object] = {
        "schema_version": PAIRED_SUMMARY_SCHEMA_VERSION,
        "summary_fields": list(SUMMARY_FIELDS),
        "comparison_order": [
            stock["variant"],
            recap_only["variant"],
            state_token["variant"],
        ],
        "paired_summary": [stock, recap_only, state_token],
        "delta_success_rate_vs_stock": _coerce_float_like(
            state_token["success_rate"],
            context="state_token.success_rate",
        )
        - _coerce_float_like(
            stock["success_rate"],
            context="stock.success_rate",
        ),
        "delta_success_rate_vs_recap_only": _coerce_float_like(
            state_token["success_rate"],
            context="state_token.success_rate",
        )
        - _coerce_float_like(
            recap_only["success_rate"],
            context="recap_only.success_rate",
        ),
    }
    if control_parity is not None:
        control_parity_mapping = _require_mapping(
            control_parity, context="control_parity"
        )
        payload["control_parity"] = {
            "control_gate_report_path": str(
                control_parity_mapping.get("control_gate_report_path", "")
            ),
            "status": str(control_parity_mapping.get("status", "")),
            "decision_reason": str(control_parity_mapping.get("decision_reason", "")),
            "authorized_checkpoint_dir": str(
                control_parity_mapping.get("authorized_checkpoint_dir", "")
            ),
            "provided_checkpoint_dir": str(
                control_parity_mapping.get("provided_checkpoint_dir", "")
            ),
            "deviation_notes": [
                str(item)
                for item in _require_sequence(
                    control_parity_mapping.get("deviation_notes", []),
                    context="control_parity.deviation_notes",
                )
            ],
        }
    return payload


def build_eval_wrapper(
    *,
    variant: str,
    summary: Mapping[str, object],
    paired_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "variant": str(variant),
        "summary": validate_summary_fields(summary),
    }
    if paired_summary is not None:
        payload["paired_summary"] = dict(paired_summary)
    return payload


__all__ = [
    "PAIRED_SUMMARY_SCHEMA_VERSION",
    "SUMMARY_FIELDS",
    "SUMMARY_SCHEMA_VERSION",
    "build_eval_wrapper",
    "build_recap_only_comparison_summary",
    "build_state_token_summary",
    "build_stock_comparison_summary",
    "build_three_way_paired_summary",
    "require_control_parity_ready",
    "resolve_control_parity_reference",
    "validate_summary_fields",
]
