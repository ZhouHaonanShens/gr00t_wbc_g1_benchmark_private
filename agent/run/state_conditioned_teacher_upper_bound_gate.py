#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_UPPER_BOUND_REPORT = Path(
    "agent/artifacts/state_conditioned_materialization/sanity/teacher_upper_bound_report.json"
)
DEFAULT_OUTPUT = Path(
    "agent/artifacts/state_conditioned_materialization/sanity/teacher_upper_bound_gate.json"
)

SCHEMA_VERSION = "g1_state_conditioned_teacher_upper_bound_gate_v1"
ARTIFACT_KIND = "state_conditioned_teacher_upper_bound_gate"
EXPECTED_UPPER_BOUND_ARTIFACT_KIND = "state_conditioned_teacher_upper_bound_report"
EXPECTED_BASELINE_ARTIFACT_KIND = "state_conditioned_teacher_gate_report"


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts.state_conditioned_teacher_upper_bound_sanity import (
    _overall_gate,
)


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="state_conditioned_teacher_upper_bound_gate.py",
        description=(
            "Materialize a dedicated retrain gate artifact from the existing "
            "teacher upper-bound report without re-running teacher rollouts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _build_parser()
    parser.add_argument(
        "--upper-bound-report",
        type=Path,
        default=DEFAULT_UPPER_BOUND_REPORT,
        help="Existing teacher upper-bound report JSON produced by task 5.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path for the dedicated teacher upper-bound gate artifact.",
    )
    return parser


def _validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"missing required {arg_name}: {resolved}")
    return resolved


def _validate_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError("--output must be a JSON file path, not a directory")
    if resolved.suffix.lower() != ".json":
        raise ValueError("--output must be a .json file path")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object, got {type(payload).__name__}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _as_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return dict(value)


def _as_list(value: Any, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return list(value)


def _as_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _as_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _as_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, not bool")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be numeric") from exc


def _expected_status(allow_retrain: bool) -> str:
    return "ALLOW" if bool(allow_retrain) else "BLOCK"


def _family_summary(family_row: dict[str, Any]) -> dict[str, Any]:
    family = _as_non_empty_string(
        family_row.get("family"), field_name="families[].family"
    )
    baseline = _as_mapping(
        family_row.get("current_model_baseline"),
        field_name=f"families[{family}].current_model_baseline",
    )
    return {
        "family": family,
        "priority": _as_non_empty_string(
            family_row.get("priority"), field_name=f"families[{family}].priority"
        ),
        "teacher_meets_threshold": bool(
            family_row.get("teacher_meets_threshold", False)
        ),
        "attempt_count": _as_int(
            family_row.get("attempt_count"),
            field_name=f"families[{family}].attempt_count",
        ),
        "success_count": _as_int(
            family_row.get("success_count"),
            field_name=f"families[{family}].success_count",
        ),
        "reachable_rate": _as_number(
            family_row.get("reachable_rate"),
            field_name=f"families[{family}].reachable_rate",
        ),
        "interpretation_code": _as_non_empty_string(
            family_row.get("interpretation_code"),
            field_name=f"families[{family}].interpretation_code",
        ),
        "interpretation": _as_non_empty_string(
            family_row.get("interpretation"),
            field_name=f"families[{family}].interpretation",
        ),
        "current_model_baseline": {
            "attempt_count": _as_int(
                baseline.get("attempt_count"),
                field_name=f"families[{family}].current_model_baseline.attempt_count",
            ),
            "success_count": _as_int(
                baseline.get("success_count"),
                field_name=f"families[{family}].current_model_baseline.success_count",
            ),
            "success_rate": _as_number(
                baseline.get("success_rate"),
                field_name=f"families[{family}].current_model_baseline.success_rate",
            ),
            "teacher_fallback_enabled": bool(
                baseline.get("teacher_fallback_enabled", False)
            ),
        },
    }


def _baseline_teacher_total_success_count(baseline_report: dict[str, Any]) -> int:
    families = _as_list(
        baseline_report.get("families"), field_name="baseline_report.families"
    )
    return sum(
        _as_int(
            _as_mapping(row, field_name=f"baseline_report.families[{index}]").get(
                "success_count"
            ),
            field_name=f"baseline_report.families[{index}].success_count",
        )
        for index, row in enumerate(families)
    )


def _observed_case(
    *,
    allow_retrain: bool,
    teacher_all_zero: bool,
    current_model_baseline_success_count: int,
    reachable_high_priority_families: list[str],
) -> tuple[str, str]:
    if bool(teacher_all_zero):
        return (
            "teacher_all_zero_block",
            "Teacher upper bound is all-zero across the selected snapshot curriculum, so retrain remains blocked.",
        )
    if (
        int(current_model_baseline_success_count) <= 0
        and reachable_high_priority_families
    ):
        return (
            "reachable_high_priority_families_baseline_zero",
            "High-priority recovery families are teacher-reachable while the current model baseline is still zero.",
        )
    if bool(allow_retrain) and int(current_model_baseline_success_count) <= 0:
        return (
            "teacher_reachable_baseline_zero",
            "Teacher is reachable on at least part of the snapshot curriculum while the current model baseline stays zero.",
        )
    return (
        "teacher_and_model_nonzero",
        "Teacher and current model baseline are both non-zero on at least part of the snapshot curriculum.",
    )


def build_teacher_upper_bound_gate(
    *,
    upper_bound_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    resolved_upper_bound_report_path = _validate_existing_file(
        upper_bound_report_path,
        arg_name="upper-bound-report",
    )
    resolved_output_path = _validate_output_path(output_path)

    report = _read_json(resolved_upper_bound_report_path)
    artifact_kind = _as_non_empty_string(
        report.get("artifact_kind"), field_name="report.artifact_kind"
    )
    if artifact_kind != EXPECTED_UPPER_BOUND_ARTIFACT_KIND:
        raise ValueError(
            "upper-bound report artifact_kind mismatch; expected "
            + EXPECTED_UPPER_BOUND_ARTIFACT_KIND
        )

    report_gate = _as_mapping(report.get("gate"), field_name="report.gate")
    teacher_upper_bound = _as_mapping(
        report.get("teacher_upper_bound"), field_name="report.teacher_upper_bound"
    )
    current_model_baseline = _as_mapping(
        report.get("current_model_baseline"),
        field_name="report.current_model_baseline",
    )
    baseline_teacher_gate_report_path = _validate_existing_file(
        Path(
            _as_non_empty_string(
                report.get("baseline_teacher_gate_report_path"),
                field_name="report.baseline_teacher_gate_report_path",
            )
        ),
        arg_name="baseline teacher_gate_report",
    )
    baseline_teacher_gate_report = _read_json(baseline_teacher_gate_report_path)
    baseline_artifact_kind = _as_non_empty_string(
        baseline_teacher_gate_report.get("artifact_kind"),
        field_name="baseline_teacher_gate_report.artifact_kind",
    )
    if baseline_artifact_kind != EXPECTED_BASELINE_ARTIFACT_KIND:
        raise ValueError(
            "baseline teacher_gate_report artifact_kind mismatch; expected "
            + EXPECTED_BASELINE_ARTIFACT_KIND
        )

    teacher_success_count = _as_int(
        teacher_upper_bound.get("success_count"),
        field_name="report.teacher_upper_bound.success_count",
    )
    teacher_reachable_rate = _as_number(
        teacher_upper_bound.get("reachable_rate"),
        field_name="report.teacher_upper_bound.reachable_rate",
    )
    teacher_all_zero = bool(teacher_upper_bound.get("teacher_all_zero", False))
    if teacher_all_zero != (teacher_success_count <= 0):
        raise ValueError(
            "report teacher_all_zero disagrees with teacher_upper_bound.success_count"
        )

    current_model_baseline_success_count = _as_int(
        current_model_baseline.get("success_count"),
        field_name="report.current_model_baseline.success_count",
    )
    current_model_baseline_success_rate = _as_number(
        current_model_baseline.get("success_rate"),
        field_name="report.current_model_baseline.success_rate",
    )

    computed_allow_retrain, computed_reason_code, computed_reason = _overall_gate(
        total_teacher_success_count=teacher_success_count,
        total_baseline_success_count=current_model_baseline_success_count,
    )
    computed_status = _expected_status(computed_allow_retrain)

    report_allow_retrain = bool(report_gate.get("allow_retrain", False))
    report_status = _as_non_empty_string(
        report_gate.get("status"), field_name="report.gate.status"
    )
    report_reason_code = _as_non_empty_string(
        report_gate.get("reason_code"), field_name="report.gate.reason_code"
    )
    report_reason = _as_non_empty_string(
        report_gate.get("reason"), field_name="report.gate.reason"
    )
    if (
        report_allow_retrain != computed_allow_retrain
        or report_status != computed_status
        or report_reason_code != computed_reason_code
        or report_reason != computed_reason
    ):
        raise ValueError(
            "upper-bound report gate disagrees with deterministic mapping from success counts"
        )

    block_allow_retrain, block_reason_code, block_reason = _overall_gate(
        total_teacher_success_count=0,
        total_baseline_success_count=current_model_baseline_success_count,
    )
    if bool(block_allow_retrain):
        raise ValueError("teacher-all-zero block semantics drifted unexpectedly")

    family_rows = [
        _family_summary(_as_mapping(row, field_name=f"report.families[{index}]"))
        for index, row in enumerate(
            _as_list(report.get("families"), field_name="report.families")
        )
    ]
    reachable_families = [
        row["family"] for row in family_rows if int(row["success_count"]) > 0
    ]
    unreachable_families = [
        row["family"] for row in family_rows if int(row["success_count"]) <= 0
    ]
    reachable_high_priority_families = [
        row["family"]
        for row in family_rows
        if row["priority"] == "high" and int(row["success_count"]) > 0
    ]
    observed_case_code, observed_case = _observed_case(
        allow_retrain=computed_allow_retrain,
        teacher_all_zero=teacher_all_zero,
        current_model_baseline_success_count=current_model_baseline_success_count,
        reachable_high_priority_families=reachable_high_priority_families,
    )
    baseline_teacher_total_success_count = _baseline_teacher_total_success_count(
        baseline_teacher_gate_report
    )

    gate_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "upper_bound_report_path": str(resolved_upper_bound_report_path),
        "upper_bound_report_artifact_kind": artifact_kind,
        "upper_bound_report_output_path": str(
            report.get("output_path", str(resolved_upper_bound_report_path))
        ),
        "baseline_teacher_gate_report_path": str(baseline_teacher_gate_report_path),
        "teacher_threshold": _as_number(
            report.get("teacher_threshold"), field_name="report.teacher_threshold"
        ),
        "allow_retrain": bool(computed_allow_retrain),
        "status": str(computed_status),
        "reason_code": str(computed_reason_code),
        "reason": str(computed_reason),
        "decision_basis": _as_non_empty_string(
            report_gate.get("decision_basis"), field_name="report.gate.decision_basis"
        ),
        "mapping": {
            "rule": (
                "BLOCK iff teacher_upper_bound.success_count == 0; otherwise ALLOW. "
                "When ALLOW, baseline success_count == 0 maps to "
                "allow_teacher_reachable_model_currently_zero; baseline success_count > 0 maps to "
                "allow_teacher_and_model_both_nonzero."
            ),
            "teacher_success_count": int(teacher_success_count),
            "teacher_reachable_rate": float(teacher_reachable_rate),
            "current_model_baseline_success_count": int(
                current_model_baseline_success_count
            ),
            "current_model_baseline_success_rate": float(
                current_model_baseline_success_rate
            ),
            "report_gate_matches_computed_gate": True,
        },
        "current_observation": {
            "case_code": str(observed_case_code),
            "case": str(observed_case),
            "teacher_all_zero": bool(teacher_all_zero),
            "reachable_high_priority_families": list(reachable_high_priority_families),
            "reachable_families": list(reachable_families),
            "unreachable_families": list(unreachable_families),
        },
        "block_semantics": {
            "trigger": "teacher_upper_bound.success_count == 0",
            "allow_retrain": False,
            "status": "BLOCK",
            "reason_code": str(block_reason_code),
            "reason": str(block_reason),
            "baseline_teacher_gate_total_success_count": int(
                baseline_teacher_total_success_count
            ),
            "baseline_teacher_gate_all_zero": bool(
                baseline_teacher_total_success_count <= 0
            ),
        },
        "family_order": [
            _as_non_empty_string(item, field_name=f"report.family_order[{index}]")
            for index, item in enumerate(
                _as_list(report.get("family_order"), field_name="report.family_order")
            )
        ],
        "families": family_rows,
    }
    _write_json(resolved_output_path, gate_payload)
    return gate_payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gate_payload = build_teacher_upper_bound_gate(
        upper_bound_report_path=args.upper_bound_report,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output_path": str(Path(args.output).expanduser().resolve()),
                "allow_retrain": gate_payload["allow_retrain"],
                "status": gate_payload["status"],
                "reason_code": gate_payload["reason_code"],
                "case_code": gate_payload["current_observation"]["case_code"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
