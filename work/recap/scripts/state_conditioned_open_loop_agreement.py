#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import importlib
from pathlib import Path
import sys
import time
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_LABELS_PATH = Path(
    "agent/artifacts/state_conditioned_materialization/training/state_conditioned_sft_labels.jsonl"
)
DEFAULT_OUTPUT_PATH = Path(
    "agent/artifacts/state_conditioned_materialization/sanity/open_loop_agreement_report.json"
)
DEFAULT_TRAINING_VIEW = "C1"
DEFAULT_FIT_STEPS = 180
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_DIM = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
MIN_HISTORY_RESPONSE_RATIO = 1e-3
INVALID_SLOT_STABILITY_TOL = 1e-9
SCHEMA_VERSION = "g1_state_conditioned_open_loop_agreement_v1"
REPORT_ARTIFACT_KIND = "state_conditioned_open_loop_agreement_report"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


micro_overfit = importlib.import_module(
    "work.recap.state_conditioned_micro_overfit_sanity"
)


class OpenLoopAgreementError(RuntimeError):
    code: str
    stage: str

    def __init__(self, code: str, stage: str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.stage = str(stage)


def _not_run_check(name: str) -> dict[str, Any]:
    return {"name": name, "passed": None, "status": "NOT_RUN"}


def _pass_check(name: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": True, "status": "PASS", **details}


def _fail_check(name: str, code: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": False,
        "status": "FAIL",
        "error_code": str(code),
        "error": str(message),
        **details,
    }


CHECK_ORDER: tuple[str, ...] = (
    "teacher_label_alignment",
    "action_range",
    "valid_mask_effectiveness",
    "history_condition_response",
    "negative_extreme_action_probe",
    "negative_all_false_mask_probe",
)


def _import_numpy() -> Any:
    return micro_overfit._import_numpy()  # type: ignore[attr-defined]


def _base_payload(output_path: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "status": "FAIL",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "output_path": str(output_path),
        "failure": None,
        "checks": {name: _not_run_check(name) for name in CHECK_ORDER},
        "summary": None,
    }


def _episode_task_text(episode_meta: Mapping[str, Any]) -> str:
    tasks = list(episode_meta.get("tasks", []))
    if not tasks:
        raise OpenLoopAgreementError(
            "INVALID_EPISODES_META",
            "teacher_label_alignment",
            "lerobot episode metadata is missing tasks[]",
        )
    return str(tasks[0])


def _build_probe_row_with_invalid_slot_payload(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(row))
    history_valid_mask = [bool(value) for value in list(mutated["history_valid_mask"])]
    for index, is_valid in enumerate(history_valid_mask):
        if is_valid:
            continue
        mutated["deployable.previous_action_history"][index] = [999.0, -999.0]
        mutated["deployable.proprio_history"][index] = [777.0, -777.0]
        mutated["deployable.short_visual_history_refs"][index] = "video://invalid/probe"
        mutated["history_timestamp_s"][index] = 12345.0
    return mutated


def _build_probe_row_with_all_history_mask_false(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(row))
    history_k = len(list(mutated["history_valid_mask"]))
    mutated["history_valid_mask"] = [False] * history_k
    mutated["history_t_std_indices"] = [0] * history_k
    mutated["history_t_raw_indices"] = [0] * history_k
    mutated["history_timestamp_s"] = [None] * history_k
    mutated["deployable.previous_action_history"] = [None] * history_k
    mutated["deployable.proprio_history"] = [None] * history_k
    mutated["deployable.short_visual_history_refs"] = [None] * history_k
    return mutated


def _validate_history_mask_or_raise(row: Mapping[str, Any]) -> dict[str, Any]:
    history_valid_mask = list(row.get("history_valid_mask", []))
    valid_count = int(sum(bool(value) for value in history_valid_mask))
    if valid_count <= 0:
        raise OpenLoopAgreementError(
            "EMPTY_HISTORY_VALID_MASK",
            "negative_all_false_mask_probe",
            "history_valid_mask must expose at least one valid slot",
        )
    return {"valid_history_count": valid_count, "history_k": len(history_valid_mask)}


def _teacher_label_alignment_check(
    rows: Sequence[Mapping[str, Any]],
    context: Any,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        episode_meta = context.episodes_by_sample_id.get(sample_id)
        if episode_meta is None:
            raise OpenLoopAgreementError(
                "DATASET_ALIGNMENT_ERROR",
                "teacher_label_alignment",
                f"sample_id missing from lerobot episodes metadata: {sample_id}",
            )
        expected_text = _episode_task_text(episode_meta)
        actual_text = str(row.get("policy_condition_text", ""))
        if expected_text != actual_text:
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "episode_index": int(episode_meta.get("episode_index", -1)),
                    "expected": expected_text,
                    "actual": actual_text,
                }
            )
    if mismatches:
        preview = mismatches[:5]
        return _fail_check(
            "teacher_label_alignment",
            "TEACHER_LABEL_TEXT_MISMATCH",
            "policy_condition_text disagrees with lerobot episode task text",
            mismatch_count=len(mismatches),
            mismatches=preview,
        )
    return _pass_check(
        "teacher_label_alignment",
        checked_row_count=len(rows),
        mismatch_count=0,
    )


def _masked_values(values: Any, mask: Any) -> Any:
    np = _import_numpy()
    values_np = np.asarray(values, dtype=np.float32)
    mask_np = np.asarray(mask, dtype=np.float32)
    return values_np[mask_np > 0.0]


def _action_range_stats(values: Any) -> dict[str, float]:
    np = _import_numpy()
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise OpenLoopAgreementError(
            "EMPTY_ACTION_VALUES",
            "action_range",
            "action range stats received no valid values",
        )
    abs_flat = np.abs(flat)
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "abs_max": float(abs_flat.max()),
        "abs_p95": float(np.percentile(abs_flat, 95.0)),
        "abs_p99": float(np.percentile(abs_flat, 99.0)),
        "abs_mean": float(abs_flat.mean()),
    }


def _range_check(values: Any, *, allowed_abs_limit: float) -> dict[str, Any]:
    np = _import_numpy()
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    violating = np.flatnonzero(np.abs(flat) > float(allowed_abs_limit))
    return {
        "allowed_abs_limit": float(allowed_abs_limit),
        "violation_count": int(violating.size),
        "max_abs_value": float(np.abs(flat).max()) if flat.size else 0.0,
        "first_violation_flat_index": None
        if violating.size == 0
        else int(violating[0]),
    }


def run_open_loop_agreement(
    *,
    labels_path: Path,
    output_path: Path,
    training_view: str,
    fit_steps: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    payload = _base_payload(output_path)
    try:
        context = micro_overfit.load_training_context(labels_path)
        rows = micro_overfit.select_label_rows(
            context.labels,
            training_view=str(training_view),
        )
        spec = micro_overfit.build_feature_spec(rows)
        dataset = micro_overfit.materialize_encoded_dataset(context, rows, spec)
        fit_result = micro_overfit.train_proxy_model(
            dataset,
            seed=int(seed),
            max_steps=int(fit_steps),
            batch_size=int(batch_size),
            hidden_dim=int(hidden_dim),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        baseline_prediction = micro_overfit.predict_proxy_model(
            fit_result,
            dataset.features,
        )
        baseline_metrics = micro_overfit.masked_regression_metrics(
            prediction=baseline_prediction,
            target=dataset.targets,
            target_mask=dataset.target_mask,
        )
        teacher_values = _masked_values(dataset.targets, dataset.target_mask)
        prediction_values = _masked_values(baseline_prediction, dataset.target_mask)
        teacher_stats = _action_range_stats(teacher_values)
        prediction_stats = _action_range_stats(prediction_values)
        allowed_abs_limit = max(
            float(teacher_stats["abs_max"]) * 2.0,
            float(teacher_stats["abs_p99"]) * 3.0,
            1.0,
        )

        payload["checks"]["teacher_label_alignment"] = _teacher_label_alignment_check(
            rows,
            context,
        )

        baseline_range = _range_check(
            prediction_values,
            allowed_abs_limit=float(allowed_abs_limit),
        )
        if int(baseline_range["violation_count"]) > 0:
            payload["checks"]["action_range"] = _fail_check(
                "action_range",
                "MODEL_ACTION_OUT_OF_RANGE",
                "proxy model predictions exceed teacher-derived action range budget",
                teacher_stats=teacher_stats,
                prediction_stats=prediction_stats,
                range_check=baseline_range,
            )
        else:
            payload["checks"]["action_range"] = _pass_check(
                "action_range",
                teacher_stats=teacher_stats,
                prediction_stats=prediction_stats,
                range_check=baseline_range,
            )

        np = _import_numpy()
        probe_row_indices = [
            index
            for index, row in enumerate(rows)
            if any(not bool(value) for value in list(row.get("history_valid_mask", [])))
            and any(bool(value) for value in list(row.get("history_valid_mask", [])))
        ]
        if not probe_row_indices:
            raise OpenLoopAgreementError(
                "NO_VALID_MASK_PROBES",
                "valid_mask_effectiveness",
                "could not find rows with both valid and invalid history slots",
            )
        invalid_probe_indices = probe_row_indices[:16]
        invalid_slot_encoded = np.stack(
            [
                np.asarray(
                    micro_overfit.encode_label_row(
                        _build_probe_row_with_invalid_slot_payload(rows[index]),
                        spec,
                    ),
                    dtype=np.float32,
                )
                for index in invalid_probe_indices
            ],
            axis=0,
        )
        invalid_slot_prediction = micro_overfit.predict_proxy_model(
            fit_result,
            invalid_slot_encoded,
        )
        baseline_probe_prediction = baseline_prediction[invalid_probe_indices]
        invalid_slot_deltas = np.abs(
            invalid_slot_prediction - baseline_probe_prediction
        )
        invalid_slot_max_delta = float(invalid_slot_deltas.max())
        if invalid_slot_max_delta > float(INVALID_SLOT_STABILITY_TOL):
            payload["checks"]["valid_mask_effectiveness"] = _fail_check(
                "valid_mask_effectiveness",
                "INVALID_SLOT_LEAKAGE",
                "changing masked-out history payload altered model predictions",
                max_abs_prediction_delta=invalid_slot_max_delta,
                tolerance=float(INVALID_SLOT_STABILITY_TOL),
                probe_count=len(invalid_slot_prediction),
            )
        else:
            payload["checks"]["valid_mask_effectiveness"] = _pass_check(
                "valid_mask_effectiveness",
                max_abs_prediction_delta=invalid_slot_max_delta,
                tolerance=float(INVALID_SLOT_STABILITY_TOL),
                probe_count=len(invalid_slot_prediction),
            )

        response_row_indices = [
            index
            for index, row in enumerate(rows)
            if int(
                sum(bool(value) for value in list(row.get("history_valid_mask", [])))
            )
            >= 2
        ]
        if not response_row_indices:
            raise OpenLoopAgreementError(
                "NO_HISTORY_RESPONSE_PROBES",
                "history_condition_response",
                "could not find rows with at least two valid history slots",
            )
        response_probe_indices = response_row_indices[:24]
        ablated_encoded = np.stack(
            [
                np.asarray(
                    micro_overfit.encode_label_row(
                        _build_probe_row_with_all_history_mask_false(rows[index]),
                        spec,
                    ),
                    dtype=np.float32,
                )
                for index in response_probe_indices
            ],
            axis=0,
        )
        ablated_prediction = micro_overfit.predict_proxy_model(
            fit_result, ablated_encoded
        )
        baseline_response_prediction = baseline_prediction[response_probe_indices]
        response_target_mask = dataset.target_mask[response_probe_indices]
        response_deltas = _masked_values(
            np.abs(ablated_prediction - baseline_response_prediction),
            response_target_mask,
        )
        teacher_scale = max(float(teacher_stats["abs_mean"]), 1e-6)
        response_ratio = float(response_deltas.mean() / teacher_scale)
        if response_ratio < float(MIN_HISTORY_RESPONSE_RATIO):
            payload["checks"]["history_condition_response"] = _fail_check(
                "history_condition_response",
                "HISTORY_UNRESPONSIVE",
                "ablating history changed the open-loop prediction too little",
                response_ratio=response_ratio,
                min_response_ratio=float(MIN_HISTORY_RESPONSE_RATIO),
                probe_count=len(response_probe_indices),
            )
        else:
            payload["checks"]["history_condition_response"] = _pass_check(
                "history_condition_response",
                response_ratio=response_ratio,
                min_response_ratio=float(MIN_HISTORY_RESPONSE_RATIO),
                probe_count=len(response_probe_indices),
            )

        negative_extreme = baseline_prediction.copy()
        negative_extreme[0] = negative_extreme[0] * float(allowed_abs_limit * 10.0)
        negative_extreme_range = _range_check(
            _masked_values(negative_extreme[:1], dataset.target_mask[:1]),
            allowed_abs_limit=float(allowed_abs_limit),
        )
        if int(negative_extreme_range["violation_count"]) <= 0:
            payload["checks"]["negative_extreme_action_probe"] = _fail_check(
                "negative_extreme_action_probe",
                "NEGATIVE_PROBE_NOT_DETECTED",
                "synthetic extreme action probe was not detected",
                range_check=negative_extreme_range,
            )
        else:
            payload["checks"]["negative_extreme_action_probe"] = _pass_check(
                "negative_extreme_action_probe",
                range_check=negative_extreme_range,
                detected_violation_count=int(negative_extreme_range["violation_count"]),
            )

        try:
            negative_mask_probe = _validate_history_mask_or_raise(
                _build_probe_row_with_all_history_mask_false(
                    rows[response_probe_indices[0]]
                )
            )
        except OpenLoopAgreementError as exc:
            payload["checks"]["negative_all_false_mask_probe"] = _pass_check(
                "negative_all_false_mask_probe",
                detected_error_code=exc.code,
                detected_error_message=micro_overfit.exception_message(exc),
            )
        else:
            payload["checks"]["negative_all_false_mask_probe"] = _fail_check(
                "negative_all_false_mask_probe",
                "NEGATIVE_PROBE_NOT_DETECTED",
                "all-false history_valid_mask probe did not raise an invalid-state signal",
                probe_result=negative_mask_probe,
            )

        failed_checks = [
            name
            for name, check in dict(payload["checks"]).items()
            if dict(check).get("status") != "PASS"
        ]
        if failed_checks:
            first_failed = failed_checks[0]
            first_payload = dict(payload["checks"][first_failed])
            payload["failure"] = {
                "code": str(first_payload.get("error_code", "CHECK_FAILED")),
                "stage": first_failed,
                "type": "OpenLoopAgreementError",
                "message": str(
                    first_payload.get("error", f"check failed: {first_failed}")
                ),
            }
            payload["summary"] = {
                "passed_check_count": len(CHECK_ORDER) - len(failed_checks),
                "total_check_count": len(CHECK_ORDER),
                "train_metrics": baseline_metrics,
                "fit_steps": int(fit_result.step_count),
            }
            return payload

        payload["status"] = "PASS"
        payload["summary"] = {
            "passed_check_count": len(CHECK_ORDER),
            "total_check_count": len(CHECK_ORDER),
            "train_metrics": baseline_metrics,
            "fit_steps": int(fit_result.step_count),
            "teacher_row_count": len(rows),
        }
        payload["telemetry"] = {
            "teacher_stats": teacher_stats,
            "prediction_stats": prediction_stats,
            "allowed_abs_limit": float(allowed_abs_limit),
            "loss": {
                "first_full_loss": float(fit_result.initial_full_loss),
                "last_full_loss": float(fit_result.final_full_loss),
                "step_count": int(fit_result.step_count),
            },
        }
        return payload
    except micro_overfit.MicroOverfitError as exc:
        payload["failure"] = {
            "code": exc.code,
            "stage": exc.stage,
            "type": exc.__class__.__name__,
            "message": micro_overfit.exception_message(exc),
        }
        return payload
    except OpenLoopAgreementError as exc:
        payload["failure"] = {
            "code": exc.code,
            "stage": exc.stage,
            "type": exc.__class__.__name__,
            "message": micro_overfit.exception_message(exc),
        }
        return payload
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        payload["failure"] = {
            "code": "UNHANDLED_ERROR",
            "stage": "cli",
            "type": exc.__class__.__name__,
            "message": micro_overfit.exception_message(exc),
        }
        return payload


def materialize_open_loop_agreement(
    *,
    labels_path: Path,
    output_path: Path,
    training_view: str,
    fit_steps: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    resolved_output = output_path.expanduser().resolve()
    payload = run_open_loop_agreement(
        labels_path=labels_path,
        output_path=resolved_output,
        training_view=training_view,
        fit_steps=fit_steps,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    payload["report_path"] = str(resolved_output)
    micro_overfit.write_json(resolved_output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="state_conditioned_open_loop_agreement.py",
        description=(
            "Check teacher/label/proxy-model open-loop agreement for action range, "
            "history-valid-mask effectiveness, and history-condition response."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--training-view",
        type=str,
        default=DEFAULT_TRAINING_VIEW,
        choices=micro_overfit.SUPPORTED_VIEWS,
    )
    parser.add_argument("--fit-steps", type=int, default=DEFAULT_FIT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = materialize_open_loop_agreement(
        labels_path=Path(str(args.labels)),
        output_path=Path(str(args.output)),
        training_view=str(args.training_view),
        fit_steps=int(args.fit_steps),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    if payload.get("status") != "PASS":
        failure = dict(payload.get("failure") or {})
        print(
            str(failure.get("message", "open-loop agreement sanity failed")),
            file=sys.stderr,
        )
    print(micro_overfit.json_text(payload))
    return 0 if payload.get("status") == "PASS" else 1


__all__ = [
    "CHECK_ORDER",
    "SCHEMA_VERSION",
    "build_parser",
    "main",
    "materialize_open_loop_agreement",
    "run_open_loop_agreement",
]


if __name__ == "__main__":
    raise SystemExit(main())
