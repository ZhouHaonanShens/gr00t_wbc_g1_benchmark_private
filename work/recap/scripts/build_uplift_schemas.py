#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


DEFAULT_OUTPUT_DIR = Path("agent/artifacts/apple_recap_exec/phase_a_tooling_draft")
DEFAULT_EXPERIMENT_MATRIX_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/experiment_matrix/gr00t_experiment_matrix.json"
)

FROZEN_MATRIX_JSON_NAME = "experiment_matrix_frozen.json"
LEDGER_CSV_NAME = "B0_E1_E2_run_ledger.csv"
UPLIFT_VERDICT_SCHEMA_JSON_NAME = "uplift_verdict_schema.json"

FROZEN_MATRIX_SCHEMA_VERSION = "apple_recap_experiment_matrix_frozen_v1"
FROZEN_MATRIX_ARTIFACT_KIND = "apple_recap_experiment_matrix_frozen"
UPLIFT_VERDICT_SCHEMA_VERSION = "apple_recap_uplift_verdict_schema_v1"
UPLIFT_VERDICT_ARTIFACT_KIND = "apple_recap_uplift_verdict_schema"

MAINLINE_DISPLAY_LABELS: tuple[str, ...] = ("B0", "E1", "E2", "E3", "E4")
LEDGER_COLUMNS: tuple[str, ...] = (
    "row_id",
    "execution_sha",
    "stage_status",
    "row_signal",
    "formal_success_count",
    "formal_success_rate",
    "long_success_rate",
    "comparable_to",
    "verdict",
)
LEDGER_REQUIRED_ROW_LABELS: tuple[str, ...] = ("B0", "E1", "E2")
STAGE_STATUS_ENUM: tuple[str, ...] = ("pending", "ready", "blocked")
ROW_SIGNAL_ENUM: tuple[str, ...] = (
    "screen_positive",
    "screen_flat",
    "screen_negative",
    "screen_inconclusive",
)
VERDICT_ENUM: tuple[str, ...] = (
    "accepted_uplift",
    "no_material_uplift",
    "rejected_regression",
    "inconclusive_rerun_needed",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import experiment_matrix
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_uplift_schemas.py",
        description=(
            "Freeze the Phase-A mainline experiment matrix view and emit the uplift "
            "ledger/schema skeletons without running evaluations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    _ = parser.add_argument(
        "--experiment-matrix-json",
        type=Path,
        default=DEFAULT_EXPERIMENT_MATRIX_JSON,
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _resolve_path(path: Path | str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    return raw.resolve()


def _validate_existing_file(path: Path | str, *, arg_name: str) -> Path:
    resolved = _resolve_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{arg_name} does not exist: {resolved}")
    return resolved


def _validate_output_dir(path: Path | str) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(_resolve_path(path))


def _validate_output_path(path: Path | str) -> Path:
    resolved = _resolve_path(path)
    if resolved.exists():
        raise ValueError(
            f"uplift tooling output already exists (no-overwrite): {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path | str, *, arg_name: str) -> dict[str, Any]:
    resolved = _validate_existing_file(path, arg_name=arg_name)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{arg_name} must contain a JSON object")
    return cast(dict[str, Any], dict(payload))


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _write_csv_header(path: Path, columns: Sequence[str]) -> Path:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns))
    tmp.replace(path)
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _signature_for_payload(payload: Mapping[str, Any]) -> str:
    signature_basis = {
        str(key): value
        for key, value in dict(payload).items()
        if key != "report_signature_sha256"
    }
    return apple_recap_execution_contract._sha256_payload(signature_basis)


def _row_map(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, Mapping):
        raise ValueError("experiment matrix rows must be an object")
    return cast(Mapping[str, Any], rows)


def _display_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    value = payload.get("display_rows")
    if not isinstance(value, list):
        raise ValueError("experiment matrix display_rows must be a list")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"experiment matrix display_rows[{index}] must be an object"
            )
        display_label = str(item.get("display_label", "")).strip()
        row_id = str(item.get("row_id", "")).strip()
        if not display_label or not row_id:
            raise ValueError(
                f"experiment matrix display_rows[{index}] must include display_label and row_id"
            )
        normalized.append({"display_label": display_label, "row_id": row_id})
    return normalized


def _validate_experiment_matrix_payload(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version")
        != experiment_matrix.EXPERIMENT_MATRIX_SCHEMA_VERSION
    ):
        raise ValueError("experiment matrix schema_version mismatch")
    if (
        payload.get("artifact_kind")
        != experiment_matrix.EXPERIMENT_MATRIX_ARTIFACT_KIND
    ):
        raise ValueError("experiment matrix artifact_kind mismatch")

    display_rows = _display_rows(payload)
    rows = _row_map(payload)
    display_label_map = {row["display_label"]: row["row_id"] for row in display_rows}
    missing_labels = [
        label for label in MAINLINE_DISPLAY_LABELS if label not in display_label_map
    ]
    if missing_labels:
        joined = ", ".join(missing_labels)
        raise ValueError(f"experiment matrix missing required mainline rows: {joined}")
    for label, row_id in display_label_map.items():
        if row_id not in rows:
            raise ValueError(
                f"experiment matrix display row {label!r} points to missing row_id {row_id!r}"
            )


def _freeze_row(
    *,
    row: Mapping[str, Any],
    display_label: str,
    diagnostic_only: bool,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(dict(row), ensure_ascii=True, sort_keys=True))
    payload["display_label"] = display_label
    payload["diagnostic_only"] = bool(diagnostic_only)
    payload["external_reference_only"] = bool(diagnostic_only)
    payload["main_verdict_eligible"] = not bool(diagnostic_only)
    if diagnostic_only:
        payload["mainline_authority"] = False
        payload["comparability_level"] = "external_reference_only"
        payload["diagnostic_status"] = "diagnostic_only"
        payload["diagnostic_reason"] = (
            "Visible for context only; excluded from the main uplift verdict surface."
        )
    else:
        payload["diagnostic_status"] = "mainline"
    return payload


def build_experiment_matrix_frozen(
    *,
    experiment_matrix_payload: Mapping[str, Any],
    experiment_matrix_json: Path | str,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
    execution_sha: str = apple_recap_execution_contract.UNSET_EXECUTION_SHA,
) -> dict[str, Any]:
    _validate_experiment_matrix_payload(experiment_matrix_payload)
    display_rows = _display_rows(experiment_matrix_payload)
    rows = _row_map(experiment_matrix_payload)

    mainline_display_rows: list[dict[str, str]] = []
    diagnostic_display_rows: list[dict[str, str]] = []
    mainline_rows: dict[str, dict[str, Any]] = {}
    diagnostic_rows: dict[str, dict[str, Any]] = {}

    for item in display_rows:
        display_label = item["display_label"]
        row_id = item["row_id"]
        row_payload = cast(Mapping[str, Any], rows[row_id])
        frozen_row = _freeze_row(
            row=row_payload,
            display_label=display_label,
            diagnostic_only=display_label not in MAINLINE_DISPLAY_LABELS,
        )
        if display_label in MAINLINE_DISPLAY_LABELS:
            mainline_display_rows.append(dict(item))
            mainline_rows[row_id] = frozen_row
        else:
            diagnostic_display_rows.append(dict(item))
            diagnostic_rows[row_id] = frozen_row

    mainline_row_id_order = [row["row_id"] for row in mainline_display_rows]
    diagnostic_row_id_order = [row["row_id"] for row in diagnostic_display_rows]
    core = {"commit": str(execution_sha)}
    payload: dict[str, Any] = {
        "schema_version": FROZEN_MATRIX_SCHEMA_VERSION,
        "artifact_kind": FROZEN_MATRIX_ARTIFACT_KIND,
        "generated_at": generated_at or _now_iso(),
        "execution_sha": str(execution_sha),
        "core": core,
        "core_digest": apple_recap_execution_contract.core_digest(core),
        "display_rows": mainline_display_rows,
        "row_id_order": mainline_row_id_order,
        "rows": mainline_rows,
        "diagnostic_display_rows": diagnostic_display_rows,
        "diagnostic_row_id_order": diagnostic_row_id_order,
        "diagnostic_rows": diagnostic_rows,
        "source_artifacts": [
            apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id="gr00t_experiment_matrix",
                authority_role="experiment_matrix_backpointer",
                relative_path=experiment_matrix_json,
            )
        ],
        "freeze_policy": {
            "append_only": True,
            "no_overwrite": True,
            "source_artifacts_read_only": True,
            "source_artifacts_mutated": False,
            "mainline_display_labels": list(MAINLINE_DISPLAY_LABELS),
            "diagnostic_rows_external_only": True,
            "diagnostic_rows_not_part_of_main_verdict": True,
        },
    }
    payload["report_signature_sha256"] = _signature_for_payload(payload)
    return payload


def build_uplift_verdict_schema(
    *,
    frozen_matrix_payload: Mapping[str, Any],
    generated_at: str | None = None,
    execution_sha: str = apple_recap_execution_contract.UNSET_EXECUTION_SHA,
) -> dict[str, Any]:
    mainline_labels = [
        str(item["display_label"])
        for item in _display_rows(frozen_matrix_payload)
        if str(item.get("display_label", "")).strip()
    ]
    if mainline_labels != list(MAINLINE_DISPLAY_LABELS):
        raise ValueError(
            "frozen matrix mainline rows must preserve the B0/E1/E2/E3/E4 order"
        )
    core = {"commit": str(execution_sha)}
    payload: dict[str, Any] = {
        "schema_version": UPLIFT_VERDICT_SCHEMA_VERSION,
        "artifact_kind": UPLIFT_VERDICT_ARTIFACT_KIND,
        "generated_at": generated_at or _now_iso(),
        "execution_sha": str(execution_sha),
        "core": core,
        "core_digest": apple_recap_execution_contract.core_digest(core),
        "ledger_columns": list(LEDGER_COLUMNS),
        "required_mainline_rows": list(LEDGER_REQUIRED_ROW_LABELS),
        "context_only_rows": [
            label
            for label in MAINLINE_DISPLAY_LABELS
            if label not in LEDGER_REQUIRED_ROW_LABELS
        ],
        "mainline_rows_required_for_verdict": list(MAINLINE_DISPLAY_LABELS),
        "diagnostic_rows_allowed_outside_mainline": True,
        "verdict_record": {
            "type": "object",
            "required": list(LEDGER_COLUMNS),
            "properties": {
                "row_id": {"type": "string", "enum": list(LEDGER_REQUIRED_ROW_LABELS)},
                "execution_sha": {"type": "string", "minLength": 1},
                "stage_status": {"type": "string", "enum": list(STAGE_STATUS_ENUM)},
                "row_signal": {"type": "string", "enum": list(ROW_SIGNAL_ENUM)},
                "formal_success_count": {"type": "integer", "minimum": 0},
                "formal_success_rate": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "long_success_rate": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "comparable_to": {"type": ["string", "null"]},
                "verdict": {"type": "string", "enum": list(VERDICT_ENUM)},
            },
        },
    }
    payload["report_signature_sha256"] = _signature_for_payload(payload)
    return payload


def materialize_uplift_schemas(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    experiment_matrix_json: Path | str = DEFAULT_EXPERIMENT_MATRIX_JSON,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
    execution_sha: str = apple_recap_execution_contract.UNSET_EXECUTION_SHA,
) -> dict[str, Any]:
    resolved_output_dir = _validate_output_dir(output_dir)
    resolved_matrix_json = _validate_existing_file(
        experiment_matrix_json,
        arg_name="experiment-matrix-json",
    )
    frozen_output = _validate_output_path(resolved_output_dir / FROZEN_MATRIX_JSON_NAME)
    ledger_output = _validate_output_path(resolved_output_dir / LEDGER_CSV_NAME)
    verdict_output = _validate_output_path(
        resolved_output_dir / UPLIFT_VERDICT_SCHEMA_JSON_NAME
    )

    experiment_matrix_payload = _read_json(
        resolved_matrix_json,
        arg_name="experiment-matrix-json",
    )
    frozen_payload = build_experiment_matrix_frozen(
        experiment_matrix_payload=experiment_matrix_payload,
        experiment_matrix_json=resolved_matrix_json,
        repo_root=repo_root,
        generated_at=generated_at,
        execution_sha=execution_sha,
    )
    _ = _write_json(frozen_output, frozen_payload)
    _ = _write_csv_header(ledger_output, LEDGER_COLUMNS)
    verdict_payload = build_uplift_verdict_schema(
        frozen_matrix_payload=frozen_payload,
        generated_at=generated_at,
        execution_sha=execution_sha,
    )
    _ = _write_json(verdict_output, verdict_payload)
    return {
        "experiment_matrix_frozen": str(frozen_output),
        "run_ledger_csv": str(ledger_output),
        "uplift_verdict_schema": str(verdict_output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = materialize_uplift_schemas(
            output_dir=args.output_dir,
            experiment_matrix_json=args.experiment_matrix_json,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_EXPERIMENT_MATRIX_JSON",
    "DEFAULT_OUTPUT_DIR",
    "FROZEN_MATRIX_ARTIFACT_KIND",
    "FROZEN_MATRIX_JSON_NAME",
    "FROZEN_MATRIX_SCHEMA_VERSION",
    "LEDGER_COLUMNS",
    "LEDGER_CSV_NAME",
    "MAINLINE_DISPLAY_LABELS",
    "UPLIFT_VERDICT_ARTIFACT_KIND",
    "UPLIFT_VERDICT_SCHEMA_JSON_NAME",
    "UPLIFT_VERDICT_SCHEMA_VERSION",
    "build_experiment_matrix_frozen",
    "build_parser",
    "build_uplift_verdict_schema",
    "main",
    "materialize_uplift_schemas",
]


if __name__ == "__main__":
    raise SystemExit(main())
