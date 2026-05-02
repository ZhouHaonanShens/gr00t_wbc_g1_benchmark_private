from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_DEV_DIR = Path("agent/artifacts/state_conditioned_materialization/devbench")
DEFAULT_TRAINING_DIR = Path(
    "agent/artifacts/state_conditioned_materialization/training"
)
DEFAULT_OUTPUT_DIR = Path("agent/artifacts/state_conditioned_materialization/eval")

ORACLE_CONDITIONED_DEV_SCORECARD_JSON_NAME = "oracle_conditioned_dev_scorecard.json"
ORACLE_GATE_DECISION_JSON_NAME = "oracle_gate_decision.json"
RECOVERY_BENCHMARK_SUMMARY_JSON_NAME = "recovery_benchmark_summary.json"
RESULT_SPLIT_DECISION_JSON_NAME = "result_split_decision.json"
SCHEMA_VERSION = "g1_state_conditioned_oracle_eval_v1"

LINE_BASELINE = "baseline"
LINE_C0 = "c0"
LINE_C1 = "c1"
LINE_ORDER: tuple[str, ...] = (LINE_BASELINE, LINE_C0, LINE_C1)
LINE_LABELS = {
    LINE_BASELINE: "original baseline",
    LINE_C0: "C0 history-aware equal-data control",
    LINE_C1: "C1 + dev-only oracle-supplied phase/mode",
}

NEXT_STEP_DETECTOR = "detector_candidate_next_round"
NEXT_STEP_CONDITION_INTERFACE = "condition_interface_analysis"
NEXT_STEP_FIX_CURRICULUM = "fix_snapshot_curriculum_pseudodemo_labels"
ALLOWED_NEXT_STEPS: tuple[str, ...] = (
    NEXT_STEP_DETECTOR,
    NEXT_STEP_CONDITION_INTERFACE,
    NEXT_STEP_FIX_CURRICULUM,
)
DIAGNOSTIC_METRIC_NAMES: tuple[str, ...] = (
    "max_phase_reached",
    "first_failure_phase",
    "recovery_attempted_rate",
    "valid_action_rate",
    "snapshot_family_hit_rate",
    "teacher_reachable_rate",
    "history_condition_usage_probe",
)
AB_CASE_ORDER: tuple[str, ...] = ("A", "B", "C", "D")
DEFAULT_VALID_ACTION_ABS_LIMIT = 2.2200000286102295


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap import state_conditioned_dev_manifest
from work.recap import state_conditioned_train
from work.demo_utils.paths import wbc_venv_python


EvalRunner = Callable[..., Mapping[str, Any]]
PHASE_INDEX_BY_NAME = {
    phase: index
    for index, phase in enumerate(
        state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES
    )
}
FAILURE_STAGE_LABEL_TO_PHASE = {
    "never_reached_apple": "SEARCH",
    "reached_apple_not_lifted": "GRASP",
    "lifted_not_brought_to_plate": "TRANSPORT",
    "near_plate_but_not_success": "PLACE",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="state_conditioned_oracle_eval.py",
        description=(
            "Run the T12 evaluation-only dev benchmark on the fixed T6 manifest, compare "
            "exactly three lines (baseline/C0/C1), add verify-hold + empty-hand transport "
            "+ same-failure-repeat diagnostics, and emit machine-readable gate/split JSON "
            "without executing detector or PM+EVENT unlocks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=DEFAULT_DEV_DIR,
        help="T6 devbench directory containing fixed_strata_definition.json and baseline manifest artifacts.",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=DEFAULT_TRAINING_DIR,
        help="T11 training directory containing C0/C1 run metadata and retained checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives oracle_conditioned_dev_scorecard.json and decision JSON artifacts.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _validate_existing_dir(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"{arg_name} directory does not exist: {resolved}")
    return resolved


def _validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"missing required {arg_name}: {resolved}")
    return resolved


def _validate_existing_file_preserve_path(path: Path, *, arg_name: str) -> Path:
    expanded = path.expanduser()
    if not expanded.exists() or not expanded.is_file():
        raise ValueError(f"missing required {arg_name}: {expanded}")
    return expanded


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    return state_conditioned_bucket_a_import._read_json(path)


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    return state_conditioned_bucket_a_import._read_jsonl_dicts(path)


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    expanded = path.expanduser()
    if not expanded.exists() or not expanded.is_file():
        return None
    return _read_json(expanded.resolve())


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    return state_conditioned_bucket_a_import._as_non_empty_string(
        value,
        field_name=field_name,
    )


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _normalize_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    raise TypeError(f"{field_name} must be bool-like, got {value!r}")


def _deep_get(source: Mapping[str, Any], dotted_path: str) -> object | None:
    current: object = source
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_field(record: Mapping[str, Any], *names: str) -> object | None:
    for name in names:
        if name in record:
            return record[name]
        nested = _deep_get(record, name)
        if nested is not None:
            return nested
        intermediate = record.get("intermediate_signals")
        if isinstance(intermediate, Mapping):
            if name in intermediate:
                return intermediate[name]
            nested = _deep_get(intermediate, name)
            if nested is not None:
                return nested
    return None


def _resolve_optional_string(record: Mapping[str, Any], *names: str) -> str | None:
    value = _resolve_field(record, *names)
    if value is None:
        return None
    if isinstance(value, list) and value:
        value = value[0]
    text = str(value).strip()
    return text if text else None


def _resolve_optional_bool(record: Mapping[str, Any], *names: str) -> bool | None:
    value = _resolve_field(record, *names)
    if value is None:
        return None
    if isinstance(value, list) and value:
        value = value[0]
    return _normalize_bool(value, field_name=names[0])


def _resolve_optional_int(record: Mapping[str, Any], *names: str) -> int | None:
    value = _resolve_field(record, *names)
    if value is None:
        return None
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _normalize_phase(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if not normalized:
        return None
    if normalized not in state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES:
        return None
    return normalized


def _phase_index(phase: str | None) -> int | None:
    if phase is None:
        return None
    return PHASE_INDEX_BY_NAME.get(str(phase))


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _failure_stage_phase(record: Mapping[str, Any]) -> str | None:
    failure_stage_guess = record.get("failure_stage_guess")
    if not isinstance(failure_stage_guess, Mapping):
        return None
    label = _resolve_optional_string(failure_stage_guess, "label")
    if label is None:
        return None
    return FAILURE_STAGE_LABEL_TO_PHASE.get(label)


def _compact_phase_rates(
    counter: Mapping[str, int], *, denominator: int
) -> dict[str, float]:
    return {
        phase: _safe_rate(int(counter.get(phase, 0)), denominator)
        for phase in state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES
    }


def _shared_diagnostic_snapshot(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: json.loads(json.dumps(diagnostics[key]))
        for key in DIAGNOSTIC_METRIC_NAMES
        if key in diagnostics
    }


def _normalize_checkpoint_dir(raw_path: object, *, field_name: str) -> Path:
    checkpoint_path = (
        Path(_as_non_empty_string(raw_path, field_name=field_name))
        .expanduser()
        .resolve()
    )
    if checkpoint_path.is_dir():
        checkpoint_dir = checkpoint_path
    elif checkpoint_path.is_file():
        checkpoint_dir = checkpoint_path.parent
    else:
        raise ValueError(f"{field_name} does not exist: {checkpoint_path}")
    selected_asset = state_conditioned_train._selected_checkpoint_asset(checkpoint_dir)
    if selected_asset is None:
        raise ValueError(
            f"{field_name} does not contain a retained checkpoint asset: {checkpoint_dir}"
        )
    return checkpoint_dir


def _resolve_baseline_python_override(
    baseline_dev_scorecard: Mapping[str, Any],
) -> dict[str, str | None]:
    baseline_invocation = _as_mapping(
        baseline_dev_scorecard.get("baseline_invocation", {}),
        field_name="baseline_dev_scorecard.baseline_invocation",
    )
    stale_python = _resolve_optional_string(baseline_invocation, "python")
    selected_python = _validate_existing_file_preserve_path(
        wbc_venv_python(REPO_ROOT),
        arg_name="WBC venv python",
    )
    return {
        "selected_python": str(selected_python),
        "stale_python": stale_python,
    }


def _expected_entry_count() -> int:
    return int(sum(state_conditioned_dev_manifest.EXPECTED_STRATA_COUNTS.values()))


def _sorted_manifest_entries(
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [
            dict(_as_mapping(entry, field_name="baseline_manifest.entries[]"))
            for entry in entries
        ],
        key=lambda entry: (
            str(entry.get("stratum_id", "")),
            int(entry.get("seed", 0)),
        ),
    )


def _load_prerequisites(*, dev_dir: Path, training_dir: Path) -> dict[str, Any]:
    dev_dir = _validate_existing_dir(dev_dir, arg_name="dev-dir")
    training_dir = _validate_existing_dir(training_dir, arg_name="training-dir")
    materialization_root = training_dir.parent
    sanity_dir = materialization_root / "sanity"

    fixed_strata_definition_path = _validate_existing_file(
        dev_dir / state_conditioned_dev_manifest.FIXED_STRATA_DEFINITION_JSON_NAME,
        arg_name="T6 fixed_strata_definition.json",
    )
    baseline_manifest_path = _validate_existing_file(
        dev_dir / state_conditioned_dev_manifest.BASELINE_MANIFEST_JSON_NAME,
        arg_name="T6 baseline_manifest.json",
    )
    baseline_dev_scorecard_path = _validate_existing_file(
        dev_dir / state_conditioned_dev_manifest.BASELINE_DEV_SCORECARD_JSON_NAME,
        arg_name="T6 baseline_dev_scorecard.json",
    )

    run_metadata_c0_path = _validate_existing_file(
        training_dir
        / state_conditioned_train.RUN_METADATA_BASENAME_BY_VARIANT[LINE_C0],
        arg_name="T11 run_metadata_C0_equal_data_control.json",
    )
    run_metadata_c1_path = _validate_existing_file(
        training_dir
        / state_conditioned_train.RUN_METADATA_BASENAME_BY_VARIANT[LINE_C1],
        arg_name="T11 run_metadata_C1_phase_mode.json",
    )
    diff_whitelist_path = _validate_existing_file(
        training_dir / state_conditioned_train.DIFF_WHITELIST_JSON_NAME,
        arg_name="T11 state_conditioned_training_fairness_diff_whitelist.json",
    )

    fixed_strata_definition = _read_json(fixed_strata_definition_path)
    baseline_manifest = _read_json(baseline_manifest_path)
    baseline_dev_scorecard = _read_json(baseline_dev_scorecard_path)
    run_metadata_c0 = _read_json(run_metadata_c0_path)
    run_metadata_c1 = _read_json(run_metadata_c1_path)
    diff_whitelist = _read_json(diff_whitelist_path)
    teacher_upper_bound_report_path = sanity_dir / "teacher_upper_bound_report.json"
    teacher_upper_bound_gate_path = sanity_dir / "teacher_upper_bound_gate.json"
    open_loop_agreement_report_path = sanity_dir / "open_loop_agreement_report.json"
    teacher_upper_bound_report = _read_optional_json(teacher_upper_bound_report_path)
    teacher_upper_bound_gate = _read_optional_json(teacher_upper_bound_gate_path)
    open_loop_agreement_report = _read_optional_json(open_loop_agreement_report_path)

    expected_entry_count = _expected_entry_count()
    if int(fixed_strata_definition.get("paired_seed_count", 0)) != int(
        len(state_conditioned_dev_manifest.DEFAULT_PAIRED_SEEDS)
    ):
        raise ValueError("T12 requires T6 paired_seed_count to remain frozen at 8")
    if _as_int(
        _as_mapping(
            baseline_dev_scorecard.get("counts"),
            field_name="baseline_dev_scorecard.counts",
        ).get("requested_entries"),
        field_name="baseline_dev_scorecard.counts.requested_entries",
    ) != int(expected_entry_count):
        raise ValueError(
            f"T12 requires T6 baseline_dev_scorecard requested_entries={expected_entry_count}"
        )

    entries = _sorted_manifest_entries(
        baseline_manifest.get("entries", [])
        if isinstance(baseline_manifest.get("entries"), list)
        else []
    )
    if len(entries) != int(expected_entry_count):
        raise ValueError(
            f"T12 requires T6 baseline_manifest entries={expected_entry_count}, got {len(entries)}"
        )
    stratum_counts = state_conditioned_dev_manifest._validate_manifest_entries(entries)
    if str(diff_whitelist.get("status", "")) != "PASS":
        raise ValueError(
            "T12 requires T11 state_conditioned_training_fairness_diff_whitelist.json status=PASS"
        )

    c0_checkpoint_dir = _normalize_checkpoint_dir(
        _as_mapping(
            _as_mapping(
                run_metadata_c0.get("comparable_run_spec"),
                field_name="run_metadata_c0.comparable_run_spec",
            ).get("checkpoint_rule"),
            field_name="run_metadata_c0.comparable_run_spec.checkpoint_rule",
        ).get("selected_checkpoint_path"),
        field_name="run_metadata_c0.comparable_run_spec.checkpoint_rule.selected_checkpoint_path",
    )
    c1_checkpoint_dir = _normalize_checkpoint_dir(
        _as_mapping(
            _as_mapping(
                run_metadata_c1.get("comparable_run_spec"),
                field_name="run_metadata_c1.comparable_run_spec",
            ).get("checkpoint_rule"),
            field_name="run_metadata_c1.comparable_run_spec.checkpoint_rule",
        ).get("selected_checkpoint_path"),
        field_name="run_metadata_c1.comparable_run_spec.checkpoint_rule.selected_checkpoint_path",
    )

    baseline_policy = dict(
        _as_mapping(
            baseline_manifest.get("baseline_policy"),
            field_name="baseline_manifest.baseline_policy",
        )
    )
    baseline_model_path = _as_non_empty_string(
        baseline_policy.get("model_path"),
        field_name="baseline_manifest.baseline_policy.model_path",
    )
    baseline_python_override = _resolve_baseline_python_override(baseline_dev_scorecard)

    return {
        "dev_dir": str(dev_dir),
        "training_dir": str(training_dir),
        "materialization_root": str(materialization_root),
        "sanity_dir": str(sanity_dir),
        "fixed_strata_definition_path": str(fixed_strata_definition_path),
        "baseline_manifest_path": str(baseline_manifest_path),
        "baseline_dev_scorecard_path": str(baseline_dev_scorecard_path),
        "run_metadata_c0_path": str(run_metadata_c0_path),
        "run_metadata_c1_path": str(run_metadata_c1_path),
        "diff_whitelist_path": str(diff_whitelist_path),
        "fixed_strata_definition": fixed_strata_definition,
        "baseline_manifest": baseline_manifest,
        "baseline_dev_scorecard": baseline_dev_scorecard,
        "run_metadata_c0": run_metadata_c0,
        "run_metadata_c1": run_metadata_c1,
        "diff_whitelist": diff_whitelist,
        "manifest_entries": entries,
        "stratum_counts": dict(stratum_counts),
        "baseline_model_path": baseline_model_path,
        "baseline_invocation_python": baseline_python_override["stale_python"],
        "baseline_override_python": baseline_python_override["selected_python"],
        "c0_checkpoint_dir": str(c0_checkpoint_dir),
        "c1_checkpoint_dir": str(c1_checkpoint_dir),
        "teacher_upper_bound_report_path": str(teacher_upper_bound_report_path),
        "teacher_upper_bound_gate_path": str(teacher_upper_bound_gate_path),
        "open_loop_agreement_report_path": str(open_loop_agreement_report_path),
        "teacher_upper_bound_report": teacher_upper_bound_report,
        "teacher_upper_bound_gate": teacher_upper_bound_gate,
        "open_loop_agreement_report": open_loop_agreement_report,
    }


def _manifest_index(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    by_seed_and_stratum: dict[tuple[int, str], dict[str, Any]] = {}
    by_paired_key: dict[str, dict[str, Any]] = {}
    for raw_entry in entries:
        entry = dict(_as_mapping(raw_entry, field_name="manifest_entry"))
        seed = _as_int(entry.get("seed"), field_name="manifest_entry.seed")
        stratum_id = _as_non_empty_string(
            entry.get("stratum_id"), field_name="manifest_entry.stratum_id"
        )
        paired_key = _as_non_empty_string(
            entry.get("paired_key"), field_name="manifest_entry.paired_key"
        )
        key = (int(seed), str(stratum_id))
        if key in by_seed_and_stratum or paired_key in by_paired_key:
            raise ValueError(f"duplicate manifest identity: {paired_key}")
        by_seed_and_stratum[key] = entry
        by_paired_key[paired_key] = entry
    return by_seed_and_stratum, by_paired_key


def _resolve_manifest_entry_for_record(
    record: Mapping[str, Any],
    *,
    by_seed_and_stratum: Mapping[tuple[int, str], Mapping[str, Any]],
    by_paired_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    paired_key = _resolve_optional_string(record, "paired_key")
    if paired_key is not None:
        entry = by_paired_key.get(paired_key)
        if entry is None:
            raise ValueError(f"runner returned unknown paired_key: {paired_key}")
        return dict(entry)
    seed = _resolve_optional_int(record, "seed")
    stratum_id = _resolve_optional_string(record, "stratum_id")
    if seed is None or stratum_id is None:
        raise ValueError(
            "runner records must provide paired_key or the pair (seed, stratum_id)"
        )
    key = (int(seed), str(stratum_id))
    entry = by_seed_and_stratum.get(key)
    if entry is None:
        raise ValueError(
            f"runner returned unknown manifest identity: seed={seed} stratum={stratum_id}"
        )
    return dict(entry)


def _normalize_episode_records(
    raw_records: Sequence[Mapping[str, Any]],
    *,
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_seed_and_stratum, by_paired_key = _manifest_index(entries)
    normalized: list[dict[str, Any]] = []
    seen_paired_keys: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        record = dict(_as_mapping(raw_record, field_name=f"episode_records[{index}]"))
        manifest_entry = _resolve_manifest_entry_for_record(
            record,
            by_seed_and_stratum=by_seed_and_stratum,
            by_paired_key=by_paired_key,
        )
        paired_key = str(manifest_entry["paired_key"])
        if paired_key in seen_paired_keys:
            raise ValueError(f"duplicate episode record for paired_key: {paired_key}")
        seen_paired_keys.add(paired_key)
        success_raw = _resolve_field(record, "success", "success_episode")
        if success_raw is None:
            raise ValueError(f"episode record missing success flag for {paired_key}")
        normalized.append(
            {
                **record,
                "paired_key": paired_key,
                "seed": int(manifest_entry["seed"]),
                "stratum_id": str(manifest_entry["stratum_id"]),
                "success": _normalize_bool(success_raw, field_name="success"),
                "primary_failure_family": (
                    _resolve_optional_string(
                        record, "primary_failure_family", "source_snapshot_family"
                    )
                    or (
                        None
                        if str(manifest_entry["stratum_id"]) == "nominal"
                        else str(manifest_entry["stratum_id"])
                    )
                ),
            }
        )
    expected_paired_keys = {str(entry["paired_key"]) for entry in entries}
    if seen_paired_keys != expected_paired_keys:
        missing = sorted(expected_paired_keys - seen_paired_keys)
        extra = sorted(seen_paired_keys - expected_paired_keys)
        raise ValueError(
            "runner did not evaluate the exact fixed dev manifest: "
            + json.dumps({"missing": missing, "extra": extra}, ensure_ascii=True)
        )
    return sorted(normalized, key=lambda record: str(record["paired_key"]))


def _normalize_sidecar_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_seed_and_stratum, by_paired_key = _manifest_index(entries)
    normalized: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        row = dict(_as_mapping(raw_row, field_name=f"sidecar_rows[{index}]"))
        manifest_entry = _resolve_manifest_entry_for_record(
            row,
            by_seed_and_stratum=by_seed_and_stratum,
            by_paired_key=by_paired_key,
        )
        paired_key = str(manifest_entry["paired_key"])
        t_value = _resolve_optional_int(row, "t", "outer_step")
        if t_value is None:
            raise ValueError(f"sidecar row missing t/outer_step for {paired_key}")
        if "t" not in row and "outer_step" in row:
            t_value = int(t_value) - 1
        phase = _normalize_phase(
            _resolve_optional_string(
                row,
                "phase",
                "policy_condition.phase",
                "canonical_phase",
            )
        )
        apple_in_hand = _resolve_optional_bool(
            row,
            "apple_in_hand",
            "privileged.apple_in_hand",
        )
        normalized.append(
            {
                **row,
                "paired_key": paired_key,
                "seed": int(manifest_entry["seed"]),
                "stratum_id": str(manifest_entry["stratum_id"]),
                "t": int(t_value),
                "phase": phase,
                "apple_in_hand": apple_in_hand,
                "recovery_entry_step": _resolve_optional_int(
                    row, "recovery_entry_step"
                ),
                "recovery_exit_step": _resolve_optional_int(row, "recovery_exit_step"),
                "source_snapshot_family": _resolve_optional_string(
                    row,
                    "source_snapshot_family",
                    "primary_failure_family",
                ),
            }
        )
    return sorted(
        normalized,
        key=lambda row: (str(row["paired_key"]), int(row["t"])),
    )


def _extract_sidecar_rows_from_step_records(
    step_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **dict(_as_mapping(record, field_name=f"step_records[{index}]")),
            "t": (
                _resolve_optional_int(record, "t")
                if _resolve_optional_int(record, "t") is not None
                else _resolve_optional_int(record, "outer_step")
            ),
        }
        for index, record in enumerate(step_records)
    ]


def _stratum_seed_block(
    entries: Sequence[Mapping[str, Any]], *, stratum_id: str
) -> list[int]:
    return state_conditioned_dev_manifest._stratum_seed_block(
        entries,
        stratum_id=stratum_id,
    )


def _eval_env_name(entries: Sequence[Mapping[str, Any]]) -> str:
    if not entries:
        raise ValueError("baseline manifest entries are empty")
    baseline_eval = dict(
        _as_mapping(entries[0].get("baseline_eval"), field_name="entry.baseline_eval")
    )
    return _as_non_empty_string(baseline_eval.get("env_name"), field_name="env_name")


def _eval_max_episode_steps(entries: Sequence[Mapping[str, Any]]) -> int:
    if not entries:
        raise ValueError("baseline manifest entries are empty")
    baseline_eval = dict(
        _as_mapping(entries[0].get("baseline_eval"), field_name="entry.baseline_eval")
    )
    return _as_int(
        baseline_eval.get("max_episode_steps"),
        field_name="max_episode_steps",
    )


def _run_eval_subprocess(
    *,
    line_spec: Mapping[str, Any],
    manifest_entries: Sequence[Mapping[str, Any]],
    stratum_counts: Mapping[str, int],
    output_dir: Path,
) -> Mapping[str, Any]:
    wrapper_path = REPO_ROOT / "agent" / "run" / "45d_vlm_critic_eval_smoke.py"
    server_script = REPO_ROOT / "agent" / "run" / "3D_recap_run_adv_server.py"
    eval_script = REPO_ROOT / "agent" / "run" / "3D_recap_eval.py"
    if not wrapper_path.is_file():
        raise FileNotFoundError(f"missing eval wrapper: {wrapper_path}")
    if not server_script.is_file() or not eval_script.is_file():
        raise FileNotFoundError("missing 3D eval entrypoint(s) required by T12")

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_dir = output_dir / "runtime_logs"
    artifact_dir = output_dir / "artifacts"
    telemetry_dir = output_dir / "telemetry"
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    per_stratum: dict[str, dict[str, Any]] = {}
    episode_records: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    total_success_count = 0
    total_evaluated = 0
    wrapper_python = _validate_existing_file_preserve_path(
        Path(str(line_spec.get("wrapper_python", sys.executable))).expanduser(),
        arg_name=f"{line_spec['line_key']} wrapper python",
    )
    eval_python = _validate_existing_file_preserve_path(
        Path(str(line_spec.get("eval_python", sys.executable))).expanduser(),
        arg_name=f"{line_spec['line_key']} eval python",
    )

    for stratum_id, requested_count in sorted(stratum_counts.items()):
        seed_block = _stratum_seed_block(manifest_entries, stratum_id=stratum_id)
        summary_json = output_dir / f"{line_spec['line_key']}_{stratum_id}.json"
        command = [
            str(wrapper_python),
            str(wrapper_path),
            "--main-repo-root",
            str(REPO_ROOT),
            "--python",
            str(eval_python),
            "--server-script",
            str(server_script),
            "--eval-script",
            str(eval_script),
            "--summary-json",
            str(summary_json),
            "--runtime-log-dir",
            str(runtime_log_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--telemetry-dir",
            str(telemetry_dir),
            "--model-path",
            str(line_spec["model_path"]),
            "--env-name",
            _eval_env_name(manifest_entries),
            "--advantage",
            "None",
            "--eval-label",
            f"state_conditioned_oracle_eval_{line_spec['line_key']}_{stratum_id}",
            "--n-episodes",
            str(int(requested_count)),
            "--max-episode-steps",
            str(_eval_max_episode_steps(manifest_entries)),
            "--seed-base",
            str(int(seed_block[0])),
        ]
        commands.append(list(command))
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if not summary_json.is_file():
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"{line_spec['line_key']} eval did not write summary JSON for {stratum_id}: {detail or f'returncode={completed.returncode}'}"
            )
        summary = _read_json(summary_json)
        if str(summary.get("wrapper_status", "")) != "ok":
            raise RuntimeError(
                f"{line_spec['line_key']} eval wrapper_status!=ok for {stratum_id}: {summary.get('error')!r}"
            )
        total_success_count += int(summary.get("success_count", 0))
        total_evaluated += int(summary.get("episodes", 0))
        per_stratum[stratum_id] = {
            "requested_count": int(requested_count),
            "evaluated_episodes": int(summary.get("episodes", 0)),
            "success_count": int(summary.get("success_count", 0)),
            "success_rate": float(summary.get("success_rate", 0.0)),
            "summary_json": str(summary_json),
        }
        episode_telemetry_jsonl = _resolve_optional_string(
            summary,
            "episode_telemetry_jsonl",
        )
        if episode_telemetry_jsonl:
            for record in _read_jsonl_dicts(Path(episode_telemetry_jsonl)):
                episode_records.append(
                    {
                        **record,
                        "stratum_id": stratum_id,
                    }
                )
        step_telemetry_jsonl = _resolve_optional_string(summary, "step_telemetry_jsonl")
        if step_telemetry_jsonl:
            for record in _read_jsonl_dicts(Path(step_telemetry_jsonl)):
                sidecar_rows.append(
                    {
                        **record,
                        "stratum_id": stratum_id,
                    }
                )

    aggregate_success_rate = (
        float(total_success_count) / float(total_evaluated)
        if total_evaluated > 0
        else 0.0
    )
    line_invocation = {
        "runner": str(wrapper_path),
        "line_key": str(line_spec["line_key"]),
        "line_label": str(line_spec["line_label"]),
        "model_path": str(line_spec["model_path"]),
        "oracle_phase_mode_supplied": bool(line_spec["oracle_phase_mode_supplied"]),
        "wrapper_python": str(wrapper_python),
        "eval_python": str(eval_python),
        "commands": commands,
    }
    for key in (
        "baseline_python_override_active",
        "baseline_python_override_reason",
        "baseline_python_override_source",
        "stale_baseline_invocation_python",
    ):
        if key in line_spec:
            line_invocation[str(key)] = line_spec[key]
    return {
        "line_invocation": line_invocation,
        "aggregate_metrics": {
            "requested_entries": int(
                sum(int(value) for value in stratum_counts.values())
            ),
            "evaluated_episodes": int(total_evaluated),
            "success_count": int(total_success_count),
            "success_rate": float(aggregate_success_rate),
        },
        "per_stratum": per_stratum,
        "episode_records": episode_records,
        "sidecar_rows": _extract_sidecar_rows_from_step_records(sidecar_rows),
    }


def _normalize_runner_result(
    result: Mapping[str, Any],
    *,
    line_spec: Mapping[str, Any],
    manifest_entries: Sequence[Mapping[str, Any]],
    stratum_counts: Mapping[str, int],
) -> dict[str, Any]:
    aggregate_metrics = dict(
        _as_mapping(result.get("aggregate_metrics", {}), field_name="aggregate_metrics")
    )
    raw_episode_records = list(result.get("episode_records", []))
    raw_sidecar_rows = list(result.get("sidecar_rows", []))
    raw_step_records = list(result.get("step_records", []))
    if not raw_sidecar_rows and raw_step_records:
        raw_sidecar_rows = _extract_sidecar_rows_from_step_records(raw_step_records)
    episode_records = _normalize_episode_records(
        raw_episode_records, entries=manifest_entries
    )
    sidecar_rows = _normalize_sidecar_rows(raw_sidecar_rows, entries=manifest_entries)

    if int(aggregate_metrics.get("requested_entries", 0)) not in (
        0,
        len(manifest_entries),
    ):
        raise ValueError(
            f"runner aggregate_metrics.requested_entries drifted from fixed dev manifest for {line_spec['line_key']}"
        )
    per_stratum_raw = result.get("per_stratum", {})
    per_stratum = {}
    if isinstance(per_stratum_raw, Mapping):
        for stratum_id, payload in per_stratum_raw.items():
            per_stratum[str(stratum_id)] = dict(
                _as_mapping(payload, field_name=f"per_stratum[{stratum_id}]")
            )
    else:
        raise TypeError("per_stratum must be an object")

    expected_strata = set(stratum_counts.keys())
    if set(per_stratum.keys()) != expected_strata:
        raise ValueError(
            f"runner per_stratum keys mismatch for {line_spec['line_key']}: expected {sorted(expected_strata)!r}, got {sorted(per_stratum.keys())!r}"
        )

    return {
        "line_invocation": dict(
            _as_mapping(result.get("line_invocation", {}), field_name="line_invocation")
        ),
        "aggregate_metrics": {
            "requested_entries": int(len(manifest_entries)),
            "evaluated_episodes": int(len(episode_records)),
            "success_count": int(
                sum(1 for record in episode_records if record["success"])
            ),
            "success_rate": float(
                sum(1 for record in episode_records if record["success"])
                / len(episode_records)
            )
            if episode_records
            else 0.0,
        },
        "per_stratum": per_stratum,
        "episode_records": episode_records,
        "sidecar_rows": sidecar_rows,
    }


def _group_sidecar_rows_by_episode(
    sidecar_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_row in sidecar_rows:
        row = dict(raw_row)
        grouped.setdefault(str(row["paired_key"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["t"]))
    return grouped


def _build_verify_hold_payload(
    *,
    sidecar_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    denominator = 0
    numerator = 0
    excluded_count = 0
    failed_paired_keys: list[str] = []
    for paired_key, rows in sorted(sidecar_by_episode.items()):
        normalized_rows = [dict(row) for row in rows]
        first_true_index: int | None = None
        for index, row in enumerate(normalized_rows):
            if row.get("apple_in_hand") is True:
                first_true_index = index
                break
        if first_true_index is None:
            excluded_count += 1
            continue
        denominator += 1
        trailing = normalized_rows[first_true_index : first_true_index + 8]
        if len(trailing) == 8 and all(
            row.get("apple_in_hand") is True for row in trailing
        ):
            numerator += 1
            continue
        failed_paired_keys.append(str(paired_key))
    rate = float(numerator) / float(denominator) if denominator > 0 else 0.0
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "excluded_count": int(excluded_count),
        "rate": float(rate),
        "failed_paired_keys": failed_paired_keys,
        "definition": "first apple_in_hand=true then remains true for 8 consecutive policy steps",
    }


def _build_empty_hand_transport_payload(
    *,
    sidecar_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    denominator = int(len(sidecar_by_episode))
    positive_paired_keys: list[str] = []
    for paired_key, rows in sorted(sidecar_by_episode.items()):
        if any(
            row.get("phase") == "TRANSPORT" and row.get("apple_in_hand") is False
            for row in rows
        ):
            positive_paired_keys.append(str(paired_key))
    numerator = int(len(positive_paired_keys))
    rate = float(numerator) / float(denominator) if denominator > 0 else 0.0
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": float(rate),
        "positive_paired_keys": positive_paired_keys,
        "definition": "episode-level rate where any sidecar row satisfies phase==TRANSPORT && apple_in_hand==false",
    }


def _merge_segments(segments: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(segments):
        if not merged:
            merged.append((int(start), int(end)))
            continue
        prev_start, prev_end = merged[-1]
        if int(start) <= int(prev_end) + 1:
            merged[-1] = (int(prev_start), max(int(prev_end), int(end)))
            continue
        merged.append((int(start), int(end)))
    return merged


def _build_same_failure_repeat_payload(
    *,
    episode_records: Sequence[Mapping[str, Any]],
    sidecar_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    positive_count = 0
    denominator = 0
    excluded_count = 0
    per_family_denominator: Counter[str] = Counter()
    per_family_positive: Counter[str] = Counter()
    episode_outcomes: list[dict[str, Any]] = []

    for raw_record in episode_records:
        record = dict(raw_record)
        stratum_id = str(record["stratum_id"])
        if stratum_id == "nominal":
            continue
        paired_key = str(record["paired_key"])
        family = _resolve_optional_string(
            record,
            "primary_failure_family",
            "source_snapshot_family",
        )
        rows = [dict(row) for row in sidecar_by_episode.get(paired_key, [])]
        if family is None or not rows:
            excluded_count += 1
            episode_outcomes.append(
                {
                    "paired_key": paired_key,
                    "family": family,
                    "eligible": False,
                    "repeat_positive": False,
                    "segments": [],
                }
            )
            continue
        denominator += 1
        per_family_denominator[str(family)] += 1
        raw_segments = {
            (
                int(row["recovery_entry_step"]),
                int(row["recovery_exit_step"]),
            )
            for row in rows
            if isinstance(row.get("recovery_entry_step"), int)
            and isinstance(row.get("recovery_exit_step"), int)
            and int(row["recovery_entry_step"]) <= int(row["recovery_exit_step"])
        }
        merged_segments = _merge_segments(sorted(raw_segments))
        repeat_positive = bool(len(merged_segments) >= 2)
        if repeat_positive:
            positive_count += 1
            per_family_positive[str(family)] += 1
        episode_outcomes.append(
            {
                "paired_key": paired_key,
                "family": str(family),
                "eligible": True,
                "repeat_positive": bool(repeat_positive),
                "segments": [[start, end] for start, end in merged_segments],
            }
        )

    per_family_breakdown = {
        family: {
            "denominator": int(per_family_denominator[family]),
            "repeat_positive_count": int(per_family_positive[family]),
            "repeat_positive_rate": (
                float(per_family_positive[family])
                / float(per_family_denominator[family])
                if per_family_denominator[family] > 0
                else 0.0
            ),
        }
        for family in sorted(per_family_denominator.keys())
    }
    rate = float(positive_count) / float(denominator) if denominator > 0 else 0.0
    return {
        "numerator": int(positive_count),
        "denominator": int(denominator),
        "excluded_count": int(excluded_count),
        "rate": float(rate),
        "per_family_breakdown": per_family_breakdown,
        "episode_outcomes": episode_outcomes,
        "definition": (
            "eligible off-nominal episode repeats when the same primary_failure_family has >=2 "
            "discrete recovery segments, merged over overlapping/adjacent recovery_entry_step "
            "/ recovery_exit_step ranges, with at least one non-family step between segments"
        ),
    }


def _build_empty_hand_release_payload(
    *,
    sidecar_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    denominator = int(len(sidecar_by_episode))
    numerator = sum(
        1
        for rows in sidecar_by_episode.values()
        if any(
            row.get("phase") == "PLACE" and row.get("apple_in_hand") is False
            for row in rows
        )
    )
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": float(numerator) / float(denominator) if denominator > 0 else 0.0,
    }


def _build_reacquire_attempt_payload(
    *,
    episode_records: Sequence[Mapping[str, Any]],
    sidecar_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    off_nominal_keys = [
        str(record["paired_key"])
        for record in episode_records
        if str(record["stratum_id"]) != "nominal"
    ]
    denominator = int(len(off_nominal_keys))
    numerator = sum(
        1
        for paired_key in off_nominal_keys
        if any(
            isinstance(row.get("recovery_entry_step"), int)
            for row in sidecar_by_episode.get(paired_key, [])
        )
    )
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": float(numerator) / float(denominator) if denominator > 0 else 0.0,
    }


def _episode_max_phase(
    record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> str | None:
    episode_phase_indices = [
        phase_index
        for row in rows
        for phase_index in [_phase_index(_normalize_phase(row.get("phase")))]
        if phase_index is not None
    ]
    if episode_phase_indices:
        return state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES[
            max(episode_phase_indices)
        ]
    if bool(record.get("success")):
        return "PLACE"
    return _failure_stage_phase(record)


def _build_max_phase_reached_payload(
    *,
    episode_records: Sequence[Mapping[str, Any]],
    sidecar_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    phase_counter: Counter[str] = Counter()
    phase_indices: list[int] = []
    missing_phase_signal_count = 0
    for raw_record in episode_records:
        record = dict(raw_record)
        phase = _episode_max_phase(
            record,
            [
                dict(row)
                for row in sidecar_by_episode.get(str(record["paired_key"]), [])
            ],
        )
        phase_index = _phase_index(phase)
        if phase is None or phase_index is None:
            missing_phase_signal_count += 1
            continue
        phase_counter[str(phase)] += 1
        phase_indices.append(int(phase_index))
    episodes_with_phase_signal = int(len(phase_indices))
    global_max_phase_index = max(phase_indices) if phase_indices else None
    global_max_phase = (
        state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES[
            global_max_phase_index
        ]
        if global_max_phase_index is not None
        else None
    )
    return {
        "phase_order": list(state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES),
        "episode_max_phase_counts": {
            phase: int(phase_counter.get(phase, 0))
            for phase in state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES
        },
        "episode_max_phase_rate": _compact_phase_rates(
            phase_counter,
            denominator=episodes_with_phase_signal,
        ),
        "episodes_with_phase_signal": episodes_with_phase_signal,
        "missing_phase_signal_count": int(missing_phase_signal_count),
        "global_max_phase": global_max_phase,
        "global_max_phase_index": global_max_phase_index,
        "mean_phase_index": (
            float(sum(phase_indices) / len(phase_indices)) if phase_indices else None
        ),
    }


def _first_failure_phase(
    record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> str | None:
    if bool(record.get("success")):
        return None
    had_in_hand = False
    for row in rows:
        apple_in_hand = row.get("apple_in_hand")
        phase = _normalize_phase(row.get("phase"))
        if apple_in_hand is True:
            had_in_hand = True
            continue
        if had_in_hand and apple_in_hand is False and phase is not None:
            return phase
    for row in rows:
        phase = _normalize_phase(row.get("phase"))
        if phase is not None:
            return phase
    return _failure_stage_phase(record)


def _build_first_failure_phase_payload(
    *,
    episode_records: Sequence[Mapping[str, Any]],
    sidecar_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    failure_phase_counter: Counter[str] = Counter()
    failed_episode_count = 0
    missing_phase_signal_count = 0
    for raw_record in episode_records:
        record = dict(raw_record)
        if bool(record.get("success")):
            continue
        failed_episode_count += 1
        phase = _first_failure_phase(
            record,
            [
                dict(row)
                for row in sidecar_by_episode.get(str(record["paired_key"]), [])
            ],
        )
        if phase is None:
            missing_phase_signal_count += 1
            continue
        failure_phase_counter[str(phase)] += 1
    dominant_phase = None
    if failure_phase_counter:
        dominant_phase = max(
            failure_phase_counter.items(),
            key=lambda item: (int(item[1]), -int(PHASE_INDEX_BY_NAME[item[0]])),
        )[0]
    return {
        "phase_order": list(state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES),
        "failed_episode_count": int(failed_episode_count),
        "missing_phase_signal_count": int(missing_phase_signal_count),
        "phase_counts": {
            phase: int(failure_phase_counter.get(phase, 0))
            for phase in state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES
        },
        "phase_rate": _compact_phase_rates(
            failure_phase_counter,
            denominator=int(failed_episode_count - missing_phase_signal_count),
        ),
        "dominant_phase": dominant_phase,
    }


def _build_recovery_attempted_payload(
    *,
    episode_records: Sequence[Mapping[str, Any]],
    sidecar_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    off_nominal_keys = [
        str(record["paired_key"])
        for record in episode_records
        if str(record["stratum_id"]) != "nominal"
    ]
    positive_paired_keys = [
        paired_key
        for paired_key in off_nominal_keys
        if any(
            isinstance(row.get("recovery_entry_step"), int)
            for row in sidecar_by_episode.get(paired_key, [])
        )
    ]
    denominator = int(len(off_nominal_keys))
    numerator = int(len(positive_paired_keys))
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": _safe_rate(numerator, denominator),
        "positive_paired_keys": sorted(positive_paired_keys),
    }


def _action_step_is_valid(
    row: Mapping[str, Any], *, allowed_abs_limit: float | None
) -> bool | None:
    action_summary = row.get("action_summary")
    if not isinstance(action_summary, Mapping):
        return None
    for payload in action_summary.values():
        if not isinstance(payload, Mapping):
            return False
        max_abs = payload.get("max_abs")
        if isinstance(max_abs, bool) or not isinstance(max_abs, (int, float)):
            return False
        if not math.isfinite(float(max_abs)):
            return False
        if allowed_abs_limit is not None and float(max_abs) > float(allowed_abs_limit):
            return False
    return True


def _build_valid_action_rate_payload(
    *,
    sidecar_rows: Sequence[Mapping[str, Any]],
    allowed_abs_limit: float | None,
    source_path: str | None,
) -> dict[str, Any]:
    denominator = 0
    numerator = 0
    excluded_count = 0
    first_invalid_step: dict[str, Any] | None = None
    for row in sidecar_rows:
        is_valid = _action_step_is_valid(row, allowed_abs_limit=allowed_abs_limit)
        if is_valid is None:
            excluded_count += 1
            continue
        denominator += 1
        if is_valid:
            numerator += 1
            continue
        if first_invalid_step is None:
            first_invalid_step = {
                "paired_key": str(row.get("paired_key")),
                "t": row.get("t"),
            }
    return {
        "allowed_abs_limit": float(allowed_abs_limit)
        if allowed_abs_limit is not None
        else None,
        "contract_source_path": source_path,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "excluded_count": int(excluded_count),
        "rate": _safe_rate(numerator, denominator),
        "first_invalid_step": first_invalid_step,
    }


def _build_snapshot_family_hit_rate_payload(
    *,
    episode_records: Sequence[Mapping[str, Any]],
    sidecar_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    denominator = 0
    numerator = 0
    missing_family_signal_count = 0
    positive_paired_keys: list[str] = []
    mismatch_examples: list[dict[str, Any]] = []
    for raw_record in episode_records:
        record = dict(raw_record)
        expected_family = str(record.get("stratum_id", ""))
        if expected_family == "nominal":
            continue
        denominator += 1
        paired_key = str(record["paired_key"])
        explicit_families = set()
        source_snapshot_family = _resolve_optional_string(
            record, "source_snapshot_family"
        )
        if source_snapshot_family is not None:
            explicit_families.add(source_snapshot_family)
        for row in sidecar_by_episode.get(paired_key, []):
            family = _resolve_optional_string(row, "source_snapshot_family")
            if family is not None:
                explicit_families.add(family)
        if not explicit_families:
            missing_family_signal_count += 1
            continue
        if expected_family in explicit_families:
            numerator += 1
            positive_paired_keys.append(paired_key)
            continue
        mismatch_examples.append(
            {
                "paired_key": paired_key,
                "expected_family": expected_family,
                "observed_families": sorted(explicit_families),
            }
        )
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": _safe_rate(numerator, denominator),
        "missing_family_signal_count": int(missing_family_signal_count),
        "positive_paired_keys": sorted(positive_paired_keys),
        "mismatch_examples": mismatch_examples[:5],
        "definition": (
            "off-nominal episode counts as a family hit only when explicit source_snapshot_family "
            "telemetry matches the fixed dev-manifest stratum"
        ),
    }


def _build_teacher_reachable_rate_payload(
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    report = prerequisites.get("teacher_upper_bound_report")
    gate = prerequisites.get("teacher_upper_bound_gate")
    source_path = str(prerequisites.get("teacher_upper_bound_report_path"))
    if not isinstance(report, Mapping):
        return {
            "available": False,
            "source_path": source_path,
            "reachable_rate": None,
            "family_breakdown": [],
        }
    reachable_rate = _resolve_field(
        report,
        "teacher_upper_bound.reachable_rate",
        "gate.mapping.teacher_reachable_rate",
    )
    family_breakdown: list[dict[str, Any]] = []
    families = report.get("families")
    if isinstance(families, list):
        for family in families:
            if not isinstance(family, Mapping):
                continue
            family_breakdown.append(
                {
                    "family": _resolve_optional_string(family, "family"),
                    "reachable_rate": _resolve_field(family, "reachable_rate"),
                    "teacher_meets_threshold": family.get("teacher_meets_threshold"),
                    "interpretation_code": family.get("interpretation_code"),
                }
            )
    gate_reason_code = None
    if isinstance(gate, Mapping):
        gate_reason_code = _resolve_optional_string(gate, "reason_code")
    gate_payload = report.get("gate")
    gate_status = (
        gate_payload.get("status") if isinstance(gate_payload, Mapping) else None
    )
    return {
        "available": True,
        "source_path": source_path,
        "reachable_rate": float(reachable_rate)
        if isinstance(reachable_rate, (int, float))
        and not isinstance(reachable_rate, bool)
        else None,
        "teacher_threshold": _resolve_field(report, "teacher_threshold"),
        "gate_status": gate_status,
        "gate_reason_code": gate_reason_code,
        "family_breakdown": family_breakdown,
    }


def _resolve_valid_action_abs_limit(prerequisites: Mapping[str, Any]) -> float:
    report = prerequisites.get("open_loop_agreement_report")
    if isinstance(report, Mapping):
        raw_limit = _resolve_field(
            report,
            "telemetry.allowed_abs_limit",
            "checks.action_range.range_check.allowed_abs_limit",
        )
        if isinstance(raw_limit, (int, float)) and not isinstance(raw_limit, bool):
            return float(raw_limit)
    return float(DEFAULT_VALID_ACTION_ABS_LIMIT)


def _build_history_condition_usage_probe_payload(
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    report = prerequisites.get("open_loop_agreement_report")
    source_path = str(prerequisites.get("open_loop_agreement_report_path"))
    if not isinstance(report, Mapping):
        return {
            "available": False,
            "source_path": source_path,
            "status": None,
            "history_condition_response": None,
        }
    checks = report.get("checks")
    history_condition_response: Mapping[str, Any] = {}
    valid_mask_effectiveness: Mapping[str, Any] = {}
    negative_mask_probe: Mapping[str, Any] = {}
    if isinstance(checks, Mapping):
        history_payload = checks.get("history_condition_response")
        valid_mask_payload = checks.get("valid_mask_effectiveness")
        negative_mask_payload = checks.get("negative_all_false_mask_probe")
        if isinstance(history_payload, Mapping):
            history_condition_response = history_payload
        if isinstance(valid_mask_payload, Mapping):
            valid_mask_effectiveness = valid_mask_payload
        if isinstance(negative_mask_payload, Mapping):
            negative_mask_probe = negative_mask_payload
    return {
        "available": True,
        "source_path": source_path,
        "status": report.get("status"),
        "history_condition_response": {
            "passed": history_condition_response.get("passed"),
            "status": history_condition_response.get("status"),
            "probe_count": history_condition_response.get("probe_count"),
            "response_ratio": history_condition_response.get("response_ratio"),
            "min_response_ratio": history_condition_response.get("min_response_ratio"),
        },
        "valid_mask_effectiveness": {
            "passed": valid_mask_effectiveness.get("passed"),
            "status": valid_mask_effectiveness.get("status"),
            "probe_count": valid_mask_effectiveness.get("probe_count"),
            "max_abs_prediction_delta": valid_mask_effectiveness.get(
                "max_abs_prediction_delta"
            ),
        },
        "negative_all_false_mask_probe": {
            "passed": negative_mask_probe.get("passed"),
            "status": negative_mask_probe.get("status"),
            "detected_error_code": negative_mask_probe.get("detected_error_code"),
        },
    }


def _build_drop_to_abort_latency_payload(
    *,
    episode_records: Sequence[Mapping[str, Any]],
    sidecar_by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    latencies: list[int] = []
    for record in episode_records:
        if bool(record["success"]):
            continue
        rows = [
            dict(row) for row in sidecar_by_episode.get(str(record["paired_key"]), [])
        ]
        if not rows:
            continue
        first_had_in_hand = False
        drop_t: int | None = None
        last_t = int(rows[-1]["t"])
        for row in rows:
            in_hand = row.get("apple_in_hand")
            if in_hand is True:
                first_had_in_hand = True
                continue
            if first_had_in_hand and in_hand is False:
                drop_t = int(row["t"])
                break
        if drop_t is not None:
            latencies.append(int(last_t - drop_t + 1))
    return {
        "eligible_episode_count": int(len(latencies)),
        "mean_policy_steps": (
            float(sum(latencies) / len(latencies)) if latencies else None
        ),
        "latencies": [int(value) for value in latencies],
    }


def _build_per_stratum_metrics(
    episode_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw_record in episode_records:
        record = dict(raw_record)
        grouped.setdefault(str(record["stratum_id"]), []).append(record)
    payload: dict[str, dict[str, Any]] = {}
    for stratum_id, records in sorted(grouped.items()):
        success_count = int(sum(1 for record in records if bool(record["success"])))
        payload[stratum_id] = {
            "evaluated_episodes": int(len(records)),
            "success_count": int(success_count),
            "success_rate": float(success_count) / float(len(records))
            if records
            else 0.0,
        }
    return payload


def _build_line_metrics(
    *,
    line_spec: Mapping[str, Any],
    runner_result: Mapping[str, Any],
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    episode_records = [dict(record) for record in runner_result["episode_records"]]
    sidecar_rows = [dict(row) for row in runner_result["sidecar_rows"]]
    sidecar_by_episode = _group_sidecar_rows_by_episode(sidecar_rows)
    if len(sidecar_by_episode) != len(episode_records):
        raise ValueError(
            f"line {line_spec['line_key']} sidecar coverage mismatch: "
            + f"episodes={len(episode_records)} sidecar_episode_count={len(sidecar_by_episode)}"
        )

    success_count = int(sum(1 for record in episode_records if bool(record["success"])))
    nominal_records = [
        record for record in episode_records if str(record["stratum_id"]) == "nominal"
    ]
    off_nominal_records = [
        record for record in episode_records if str(record["stratum_id"]) != "nominal"
    ]
    nominal_success_count = int(
        sum(1 for record in nominal_records if bool(record["success"]))
    )
    off_nominal_success_count = int(
        sum(1 for record in off_nominal_records if bool(record["success"]))
    )

    verify_hold = _build_verify_hold_payload(sidecar_by_episode=sidecar_by_episode)
    empty_hand_transport = _build_empty_hand_transport_payload(
        sidecar_by_episode=sidecar_by_episode
    )
    same_failure_repeat = _build_same_failure_repeat_payload(
        episode_records=episode_records,
        sidecar_by_episode=sidecar_by_episode,
    )
    empty_hand_release = _build_empty_hand_release_payload(
        sidecar_by_episode=sidecar_by_episode
    )
    reacquire_attempt = _build_reacquire_attempt_payload(
        episode_records=episode_records,
        sidecar_by_episode=sidecar_by_episode,
    )
    recovery_attempted = _build_recovery_attempted_payload(
        episode_records=episode_records,
        sidecar_by_episode=sidecar_by_episode,
    )
    drop_to_abort_latency = _build_drop_to_abort_latency_payload(
        episode_records=episode_records,
        sidecar_by_episode=sidecar_by_episode,
    )
    max_phase_reached = _build_max_phase_reached_payload(
        episode_records=episode_records,
        sidecar_by_episode=sidecar_by_episode,
    )
    first_failure_phase = _build_first_failure_phase_payload(
        episode_records=episode_records,
        sidecar_by_episode=sidecar_by_episode,
    )
    valid_action_rate = _build_valid_action_rate_payload(
        sidecar_rows=sidecar_rows,
        allowed_abs_limit=_resolve_valid_action_abs_limit(prerequisites),
        source_path=str(prerequisites.get("open_loop_agreement_report_path")),
    )
    snapshot_family_hit_rate = _build_snapshot_family_hit_rate_payload(
        episode_records=episode_records,
        sidecar_by_episode=sidecar_by_episode,
    )
    teacher_reachable_rate = _build_teacher_reachable_rate_payload(prerequisites)
    history_condition_usage_probe = _build_history_condition_usage_probe_payload(
        prerequisites
    )

    aggregate_metrics = dict(runner_result["aggregate_metrics"])
    comparable_metrics = {
        "success_rate": float(success_count) / float(len(episode_records))
        if episode_records
        else 0.0,
        "off_nominal_recovery_success_rate": float(off_nominal_success_count)
        / float(len(off_nominal_records))
        if off_nominal_records
        else 0.0,
        "empty_hand_release_rate": float(empty_hand_release["rate"]),
        "drop_to_abort_latency": drop_to_abort_latency["mean_policy_steps"],
        "reacquire_attempt_rate": float(reacquire_attempt["rate"]),
        "verify_hold_pass_rate": float(verify_hold["rate"]),
        "empty_hand_transport_rate": float(empty_hand_transport["rate"]),
        "same_failure_repeat_rate": float(same_failure_repeat["rate"]),
        "nominal_success_delta": 0.0,
    }
    diagnostics = {
        "verify_hold_pass_rate": verify_hold,
        "empty_hand_transport_rate": empty_hand_transport,
        "same_failure_repeat_rate": same_failure_repeat,
        "empty_hand_release_rate": empty_hand_release,
        "reacquire_attempt_rate": reacquire_attempt,
        "drop_to_abort_latency": drop_to_abort_latency,
        "max_phase_reached": max_phase_reached,
        "first_failure_phase": first_failure_phase,
        "recovery_attempted_rate": recovery_attempted,
        "valid_action_rate": valid_action_rate,
        "snapshot_family_hit_rate": snapshot_family_hit_rate,
        "teacher_reachable_rate": teacher_reachable_rate,
        "history_condition_usage_probe": history_condition_usage_probe,
    }
    return {
        "line_key": str(line_spec["line_key"]),
        "line_label": str(line_spec["line_label"]),
        "oracle_phase_mode_supplied": bool(line_spec["oracle_phase_mode_supplied"]),
        "line_invocation": dict(runner_result["line_invocation"]),
        "counts": {
            "evaluated_episodes": int(len(episode_records)),
            "success_count": int(success_count),
            "nominal_episodes": int(len(nominal_records)),
            "off_nominal_episodes": int(len(off_nominal_records)),
            "nominal_success_count": int(nominal_success_count),
            "off_nominal_success_count": int(off_nominal_success_count),
            "aggregate_requested_entries": int(aggregate_metrics["requested_entries"]),
        },
        "comparable_metrics": comparable_metrics,
        "diagnostics": diagnostics,
        "diagnostic_snapshot": _shared_diagnostic_snapshot(diagnostics),
        "per_stratum": _build_per_stratum_metrics(episode_records),
        "episode_records": episode_records,
    }


def _apply_nominal_success_delta(
    line_metrics_by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    baseline_nominal = float(
        line_metrics_by_key[LINE_BASELINE]["counts"]["nominal_success_count"]
    ) / float(line_metrics_by_key[LINE_BASELINE]["counts"]["nominal_episodes"])
    updated: dict[str, dict[str, Any]] = {}
    for line_key in LINE_ORDER:
        payload = json.loads(json.dumps(line_metrics_by_key[line_key]))
        nominal_rate = float(payload["counts"]["nominal_success_count"]) / float(
            payload["counts"]["nominal_episodes"]
        )
        payload["comparable_metrics"]["nominal_success_delta"] = float(
            nominal_rate - baseline_nominal
        )
        updated[line_key] = payload
    return updated


def _resolve_line_diagnostic_payload(
    line_metrics: Mapping[str, Any], diagnostic_name: str
) -> Mapping[str, Any]:
    diagnostics = line_metrics.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return {}
    payload = diagnostics.get(diagnostic_name)
    return payload if isinstance(payload, Mapping) else {}


def _resolve_line_diagnostic_number(
    line_metrics: Mapping[str, Any],
    diagnostic_name: str,
    field_name: str,
) -> float | None:
    value = _resolve_line_diagnostic_payload(line_metrics, diagnostic_name).get(
        field_name
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _history_probe_passed(line_metrics_by_key: Mapping[str, Mapping[str, Any]]) -> bool:
    history_probe = _resolve_line_diagnostic_payload(
        line_metrics_by_key[LINE_C1],
        "history_condition_usage_probe",
    )
    history_response = history_probe.get("history_condition_response")
    if not isinstance(history_response, Mapping):
        return False
    return bool(history_response.get("passed"))


def _build_ab_decision_tree_payload(
    *,
    line_metrics_by_key: Mapping[str, Mapping[str, Any]],
    tolerance: float,
    legacy_next_step: str,
) -> dict[str, Any]:
    baseline_metrics = line_metrics_by_key[LINE_BASELINE]
    c0_metrics = line_metrics_by_key[LINE_C0]
    c1_metrics = line_metrics_by_key[LINE_C1]

    baseline_success = float(baseline_metrics["comparable_metrics"]["success_rate"])
    c0_success = float(c0_metrics["comparable_metrics"]["success_rate"])
    c1_success = float(c1_metrics["comparable_metrics"]["success_rate"])

    teacher_reachable_rate = _resolve_line_diagnostic_number(
        baseline_metrics,
        "teacher_reachable_rate",
        "reachable_rate",
    )
    history_probe_positive = _history_probe_passed(line_metrics_by_key)
    baseline_phase_mean = _resolve_line_diagnostic_number(
        baseline_metrics,
        "max_phase_reached",
        "mean_phase_index",
    )
    c0_phase_mean = _resolve_line_diagnostic_number(
        c0_metrics,
        "max_phase_reached",
        "mean_phase_index",
    )
    c1_phase_mean = _resolve_line_diagnostic_number(
        c1_metrics,
        "max_phase_reached",
        "mean_phase_index",
    )
    baseline_recovery_rate = _resolve_line_diagnostic_number(
        baseline_metrics,
        "recovery_attempted_rate",
        "rate",
    )
    c0_recovery_rate = _resolve_line_diagnostic_number(
        c0_metrics,
        "recovery_attempted_rate",
        "rate",
    )
    c1_recovery_rate = _resolve_line_diagnostic_number(
        c1_metrics,
        "recovery_attempted_rate",
        "rate",
    )
    c1_valid_action_rate = _resolve_line_diagnostic_number(
        c1_metrics,
        "valid_action_rate",
        "rate",
    )
    c1_snapshot_family_hit_rate = _resolve_line_diagnostic_number(
        c1_metrics,
        "snapshot_family_hit_rate",
        "rate",
    )

    phase_gain_vs_c0 = (
        float(c1_phase_mean - c0_phase_mean)
        if c1_phase_mean is not None and c0_phase_mean is not None
        else None
    )
    phase_gain_vs_baseline = (
        float(c1_phase_mean - baseline_phase_mean)
        if c1_phase_mean is not None and baseline_phase_mean is not None
        else None
    )
    recovery_gain_vs_c0 = (
        float(c1_recovery_rate - c0_recovery_rate)
        if c1_recovery_rate is not None and c0_recovery_rate is not None
        else None
    )
    recovery_gain_vs_baseline = (
        float(c1_recovery_rate - baseline_recovery_rate)
        if c1_recovery_rate is not None and baseline_recovery_rate is not None
        else None
    )

    all_zero_success = bool(
        abs(baseline_success) <= tolerance
        and abs(c0_success) <= tolerance
        and abs(c1_success) <= tolerance
    )
    teacher_reachable = bool(
        teacher_reachable_rate is not None and teacher_reachable_rate > tolerance
    )
    teacher_zero = bool(
        teacher_reachable_rate is not None and teacher_reachable_rate <= tolerance
    )
    success_progress_detected = bool(
        c0_success > baseline_success + tolerance
        or c1_success > baseline_success + tolerance
        or c1_success > c0_success + tolerance
    )
    phase_progress_detected = bool(
        (phase_gain_vs_c0 is not None and phase_gain_vs_c0 > 0.25)
        or (phase_gain_vs_baseline is not None and phase_gain_vs_baseline > 0.25)
    )
    recovery_progress_detected = bool(
        (recovery_gain_vs_c0 is not None and recovery_gain_vs_c0 > tolerance)
        or (
            recovery_gain_vs_baseline is not None
            and recovery_gain_vs_baseline > tolerance
        )
    )
    structural_behavior_change_detected = bool(
        success_progress_detected
        or (teacher_reachable and phase_progress_detected)
        or (teacher_reachable and recovery_progress_detected)
    )
    signal_integrity_issue_detected = bool(
        not history_probe_positive
        or (
            c1_valid_action_rate is not None
            and c1_valid_action_rate < float(1.0 - tolerance)
        )
        or (
            c1_snapshot_family_hit_rate is not None
            and c1_snapshot_family_hit_rate < float(1.0 - tolerance)
        )
    )

    case_matches = {
        "A": bool(all_zero_success and teacher_zero),
        "B": False,
        "C": False,
        "D": False,
    }
    if not case_matches["A"] and structural_behavior_change_detected:
        case_matches["C"] = True
    elif (
        not case_matches["A"] and teacher_reachable and signal_integrity_issue_detected
    ):
        case_matches["B"] = True
    elif not case_matches["A"] and not case_matches["B"] and not case_matches["C"]:
        case_matches["D"] = True

    matched_cases = [case for case in AB_CASE_ORDER if case_matches[case]]
    if len(matched_cases) != 1:
        raise ValueError(
            "A/B/C/D decision tree must match exactly one case, got "
            + f"{matched_cases}"
        )
    ab_case = matched_cases[0]

    case_details = {
        "A": {
            "label": "teacher unreachable and all three lines remain zero",
            "interpretation": (
                "The bottleneck is still snapshot curriculum reachability or label semantics, "
                "not the C0/C1 training split itself."
            ),
            "recommended_focus": "fix_snapshot_curriculum_pseudodemo_labels",
            "structural_behavior_change": "no",
        },
        "B": {
            "label": "teacher reachable but the conditioned signal is not being consumed",
            "interpretation": (
                "The task is teacher-reachable, yet behavior stays flat and at least one signal-"
                "integrity probe says history/state usage or wrapper injection still needs diagnosis."
            ),
            "recommended_focus": "condition_interface_or_training_signal_diagnosis",
            "structural_behavior_change": "no",
        },
        "C": {
            "label": "label fix caused structural behavior change",
            "interpretation": (
                "The refreshed C0/C1 path changes behavior or intermediate control structure, so "
                "the pseudodemo-label refresh is having a real effect even if the benchmark may still be hard."
            ),
            "recommended_focus": "continue_state_conditioned_direction_without_unlocking_detector",
            "structural_behavior_change": "yes",
        },
        "D": {
            "label": "teacher reachable but current phase/mode/history abstraction stays too weak",
            "interpretation": (
                "The conditioning pathway looks alive and integrity checks stay clean, but the current "
                "abstraction does not separate behavior from baseline strongly enough."
            ),
            "recommended_focus": "prepare_more_explicit_state_labels_next_round",
            "structural_behavior_change": "no",
        },
    }
    selected_case = dict(case_details[ab_case])
    selected_case["case"] = ab_case

    return {
        "legacy_next_step": legacy_next_step,
        "ab_case": ab_case,
        "ab_case_label": selected_case["label"],
        "ab_case_reason": selected_case["interpretation"],
        "ab_recommended_focus": selected_case["recommended_focus"],
        "structural_behavior_change_detected": bool(
            selected_case["structural_behavior_change"] == "yes"
        ),
        "pseudodemo_label_fix_structural_behavior_change": selected_case[
            "structural_behavior_change"
        ],
        "decision_tree": {
            "tree_name": "pseudodemo_label_fix_abcd_v1",
            "legacy_next_step": legacy_next_step,
            "matched_case_count": 1,
            "matched_cases": [ab_case],
            "ordered_case_matches": [
                {"case": case, "matched": bool(case_matches[case])}
                for case in AB_CASE_ORDER
            ],
            "signals": {
                "all_zero_success": bool(all_zero_success),
                "teacher_reachable_rate": teacher_reachable_rate,
                "teacher_reachable": bool(teacher_reachable),
                "history_probe_positive": bool(history_probe_positive),
                "signal_integrity_issue_detected": bool(
                    signal_integrity_issue_detected
                ),
                "success_progress_detected": bool(success_progress_detected),
                "phase_progress_detected": bool(phase_progress_detected),
                "recovery_progress_detected": bool(recovery_progress_detected),
                "structural_behavior_change_detected": bool(
                    structural_behavior_change_detected
                ),
            },
            "metric_snapshot": {
                "success_rate": {
                    LINE_BASELINE: float(baseline_success),
                    LINE_C0: float(c0_success),
                    LINE_C1: float(c1_success),
                },
                "mean_phase_index": {
                    LINE_BASELINE: baseline_phase_mean,
                    LINE_C0: c0_phase_mean,
                    LINE_C1: c1_phase_mean,
                },
                "recovery_attempted_rate": {
                    LINE_BASELINE: baseline_recovery_rate,
                    LINE_C0: c0_recovery_rate,
                    LINE_C1: c1_recovery_rate,
                },
                "conditioning_integrity": {
                    "history_probe_positive": bool(history_probe_positive),
                    "c1_valid_action_rate": c1_valid_action_rate,
                    "c1_snapshot_family_hit_rate": c1_snapshot_family_hit_rate,
                },
            },
            "pairwise_deltas": {
                "phase_gain_c1_minus_c0": phase_gain_vs_c0,
                "phase_gain_c1_minus_baseline": phase_gain_vs_baseline,
                "recovery_gain_c1_minus_c0": recovery_gain_vs_c0,
                "recovery_gain_c1_minus_baseline": recovery_gain_vs_baseline,
            },
            "selected_case": selected_case,
        },
    }


def build_result_split_decision(
    *,
    line_metrics_by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = float(
        line_metrics_by_key[LINE_BASELINE]["comparable_metrics"]["success_rate"]
    )
    c0 = float(line_metrics_by_key[LINE_C0]["comparable_metrics"]["success_rate"])
    c1 = float(line_metrics_by_key[LINE_C1]["comparable_metrics"]["success_rate"])
    denominator = int(
        line_metrics_by_key[LINE_BASELINE]["counts"]["evaluated_episodes"]
    )
    tolerance = 1.0 / float(denominator) if denominator > 0 else 0.0
    c1_minus_c0 = float(c1 - c0)
    c0_minus_baseline = float(c0 - baseline)
    c1_minus_baseline = float(c1 - baseline)

    clearly_better_than_c0 = bool(c1_minus_c0 > tolerance)
    clearly_better_than_baseline = bool(c0_minus_baseline > tolerance)
    c1_approx_c0 = bool(abs(c1_minus_c0) <= tolerance)
    c0_approx_baseline = bool(abs(c0_minus_baseline) <= tolerance)
    c1_approx_baseline = bool(abs(c1_minus_baseline) <= tolerance)

    if clearly_better_than_c0 and clearly_better_than_baseline:
        next_step = NEXT_STEP_DETECTOR
        oracle_uplift = True
        branch_reason = "C1 > C0 > baseline on the fixed dev manifest using the primary success_rate comparator"
    elif clearly_better_than_baseline and c1_approx_c0:
        next_step = NEXT_STEP_CONDITION_INTERFACE
        oracle_uplift = False
        branch_reason = "C0 > baseline but C1 ≈ C0 on the fixed dev manifest"
    else:
        next_step = NEXT_STEP_FIX_CURRICULUM
        oracle_uplift = False
        branch_reason = "C1 ≈ C0 ≈ baseline or no clear uplift ordering was established"

    ab_decision = _build_ab_decision_tree_payload(
        line_metrics_by_key=line_metrics_by_key,
        tolerance=tolerance,
        legacy_next_step=next_step,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_result_split_decision",
        "line_order": list(LINE_ORDER),
        "allowed_next_steps": list(ALLOWED_NEXT_STEPS),
        "primary_comparator_metric": "success_rate",
        "approx_equal_tolerance": float(tolerance),
        "metric_snapshot": {
            LINE_BASELINE: float(baseline),
            LINE_C0: float(c0),
            LINE_C1: float(c1),
        },
        "pairwise_deltas": {
            "c0_minus_baseline": float(c0_minus_baseline),
            "c1_minus_c0": float(c1_minus_c0),
            "c1_minus_baseline": float(c1_minus_baseline),
        },
        "next_step": next_step,
        "legacy_next_step": str(ab_decision["legacy_next_step"]),
        "branch_reason": branch_reason,
        "oracle_uplift_clearly_established": bool(oracle_uplift),
        "ab_case": str(ab_decision["ab_case"]),
        "ab_case_label": str(ab_decision["ab_case_label"]),
        "ab_case_reason": str(ab_decision["ab_case_reason"]),
        "ab_recommended_focus": str(ab_decision["ab_recommended_focus"]),
        "structural_behavior_change_detected": bool(
            ab_decision["structural_behavior_change_detected"]
        ),
        "pseudodemo_label_fix_structural_behavior_change": str(
            ab_decision["pseudodemo_label_fix_structural_behavior_change"]
        ),
        "decision_tree": dict(ab_decision["decision_tree"]),
        "future_unlocks": {
            "pm_event_analysis_only": bool(oracle_uplift),
            "detector_candidate": bool(oracle_uplift),
        },
        "executed_actions": {
            "pm_event_analysis": False,
            "detector": False,
        },
        "this_round_execution_scope": "evaluation_and_decision_only",
    }


def _build_oracle_gate_decision(
    *,
    line_metrics_by_key: Mapping[str, Mapping[str, Any]],
    result_split_decision: Mapping[str, Any],
) -> dict[str, Any]:
    oracle_uplift = bool(result_split_decision["oracle_uplift_clearly_established"])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_oracle_gate_decision",
        "gate_name": "oracle_conditioned_dev_uplift_gate",
        "gate_status": "PASS" if oracle_uplift else "BLOCK",
        "gate_passed": bool(oracle_uplift),
        "blocker_reason": None
        if oracle_uplift
        else str(result_split_decision["branch_reason"]),
        "line_order": list(LINE_ORDER),
        "oracle_phase_mode_contract": {
            line_key: bool(line_metrics_by_key[line_key]["oracle_phase_mode_supplied"])
            for line_key in LINE_ORDER
        },
        "primary_metric_snapshot": {
            line_key: float(
                line_metrics_by_key[line_key]["comparable_metrics"]["success_rate"]
            )
            for line_key in LINE_ORDER
        },
        "legacy_next_step": str(result_split_decision["legacy_next_step"]),
        "ab_case": str(result_split_decision["ab_case"]),
        "structural_behavior_change_detected": bool(
            result_split_decision["structural_behavior_change_detected"]
        ),
        "pseudodemo_label_fix_structural_behavior_change": str(
            result_split_decision["pseudodemo_label_fix_structural_behavior_change"]
        ),
        "next_step_if_blocked": None
        if oracle_uplift
        else str(result_split_decision["next_step"]),
    }


def _build_recovery_benchmark_summary(
    *,
    prerequisites: Mapping[str, Any],
    line_metrics_by_key: Mapping[str, Mapping[str, Any]],
    result_split_decision: Mapping[str, Any],
) -> dict[str, Any]:
    shared_diagnostics = {
        "teacher_reachable_rate": _build_teacher_reachable_rate_payload(prerequisites),
        "history_condition_usage_probe": _build_history_condition_usage_probe_payload(
            prerequisites
        ),
    }
    summary_lines: list[dict[str, Any]] = []
    for line_key in LINE_ORDER:
        line_metrics = line_metrics_by_key[line_key]
        summary_lines.append(
            {
                "line_key": line_key,
                "line_label": line_metrics["line_label"],
                "oracle_phase_mode_supplied": bool(
                    line_metrics["oracle_phase_mode_supplied"]
                ),
                "comparable_metrics": dict(line_metrics["comparable_metrics"]),
                "counts": dict(line_metrics["counts"]),
                "diagnostic_snapshot": dict(line_metrics["diagnostic_snapshot"]),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_recovery_benchmark_summary",
        "fixed_dev_manifest_path": str(prerequisites["baseline_manifest_path"]),
        "line_order": list(LINE_ORDER),
        "diagnostic_metric_names": list(DIAGNOSTIC_METRIC_NAMES),
        "shared_diagnostics": shared_diagnostics,
        "summary_lines": summary_lines,
        "next_step": str(result_split_decision["next_step"]),
        "legacy_next_step": str(result_split_decision["legacy_next_step"]),
        "ab_case": str(result_split_decision["ab_case"]),
        "ab_case_reason": str(result_split_decision["ab_case_reason"]),
        "structural_behavior_change_detected": bool(
            result_split_decision["structural_behavior_change_detected"]
        ),
        "pseudodemo_label_fix_structural_behavior_change": str(
            result_split_decision["pseudodemo_label_fix_structural_behavior_change"]
        ),
        "decision_tree": dict(result_split_decision["decision_tree"]),
        "future_unlocks": dict(result_split_decision["future_unlocks"]),
    }


def materialize_state_conditioned_oracle_eval(
    *,
    dev_dir: Path,
    training_dir: Path,
    output_dir: Path,
    eval_runner: EvalRunner | None = None,
) -> dict[str, Any]:
    output_dir = state_conditioned_bucket_a_import.validate_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prerequisites = _load_prerequisites(dev_dir=dev_dir, training_dir=training_dir)
    manifest_entries = list(prerequisites["manifest_entries"])
    stratum_counts = dict(prerequisites["stratum_counts"])
    shared_eval_python = str(prerequisites["baseline_override_python"])

    line_specs = {
        LINE_BASELINE: {
            "line_key": LINE_BASELINE,
            "line_label": LINE_LABELS[LINE_BASELINE],
            "model_path": str(prerequisites["baseline_model_path"]),
            "oracle_phase_mode_supplied": False,
            "wrapper_python": shared_eval_python,
            "eval_python": shared_eval_python,
            "baseline_python_override_active": True,
            "baseline_python_override_reason": (
                "T12 ignores stale T6 baseline_invocation.python and pins the current "
                "WBC venv python for baseline command reconstruction"
            ),
            "baseline_python_override_source": "work.demo_utils.paths.wbc_venv_python",
            "stale_baseline_invocation_python": prerequisites[
                "baseline_invocation_python"
            ],
        },
        LINE_C0: {
            "line_key": LINE_C0,
            "line_label": LINE_LABELS[LINE_C0],
            "model_path": str(prerequisites["c0_checkpoint_dir"]),
            "oracle_phase_mode_supplied": False,
            "wrapper_python": shared_eval_python,
            "eval_python": shared_eval_python,
        },
        LINE_C1: {
            "line_key": LINE_C1,
            "line_label": LINE_LABELS[LINE_C1],
            "model_path": str(prerequisites["c1_checkpoint_dir"]),
            "oracle_phase_mode_supplied": True,
            "wrapper_python": shared_eval_python,
            "eval_python": shared_eval_python,
        },
    }

    runner = _run_eval_subprocess if eval_runner is None else eval_runner
    raw_line_metrics: dict[str, dict[str, Any]] = {}
    for line_key in LINE_ORDER:
        line_output_dir = output_dir / line_key
        raw_result = runner(
            line_spec=line_specs[line_key],
            manifest_entries=manifest_entries,
            stratum_counts=stratum_counts,
            output_dir=line_output_dir,
        )
        normalized_result = _normalize_runner_result(
            dict(_as_mapping(raw_result, field_name=f"{line_key}_runner_result")),
            line_spec=line_specs[line_key],
            manifest_entries=manifest_entries,
            stratum_counts=stratum_counts,
        )
        raw_line_metrics[line_key] = _build_line_metrics(
            line_spec=line_specs[line_key],
            runner_result=normalized_result,
            prerequisites=prerequisites,
        )

    line_metrics_by_key = _apply_nominal_success_delta(raw_line_metrics)
    shared_diagnostics = {
        "teacher_reachable_rate": _build_teacher_reachable_rate_payload(prerequisites),
        "history_condition_usage_probe": _build_history_condition_usage_probe_payload(
            prerequisites
        ),
    }
    scorecard = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_oracle_conditioned_dev_scorecard",
        "fixed_dev_manifest_path": str(prerequisites["baseline_manifest_path"]),
        "fixed_strata_definition_path": str(
            prerequisites["fixed_strata_definition_path"]
        ),
        "training_run_metadata": {
            LINE_C0: str(prerequisites["run_metadata_c0_path"]),
            LINE_C1: str(prerequisites["run_metadata_c1_path"]),
        },
        "line_order": list(LINE_ORDER),
        "line_labels": dict(LINE_LABELS),
        "comparable_metric_names": [
            "success_rate",
            "off_nominal_recovery_success_rate",
            "empty_hand_release_rate",
            "drop_to_abort_latency",
            "reacquire_attempt_rate",
            "verify_hold_pass_rate",
            "empty_hand_transport_rate",
            "same_failure_repeat_rate",
            "nominal_success_delta",
        ],
        "diagnostic_metric_names": list(DIAGNOSTIC_METRIC_NAMES),
        "shared_diagnostics": shared_diagnostics,
        "lines": [line_metrics_by_key[line_key] for line_key in LINE_ORDER],
    }
    result_split_decision = build_result_split_decision(
        line_metrics_by_key=line_metrics_by_key
    )
    oracle_gate_decision = _build_oracle_gate_decision(
        line_metrics_by_key=line_metrics_by_key,
        result_split_decision=result_split_decision,
    )
    recovery_benchmark_summary = _build_recovery_benchmark_summary(
        prerequisites=prerequisites,
        line_metrics_by_key=line_metrics_by_key,
        result_split_decision=result_split_decision,
    )

    scorecard_path = _write_json(
        output_dir / ORACLE_CONDITIONED_DEV_SCORECARD_JSON_NAME,
        scorecard,
    )
    gate_path = _write_json(
        output_dir / ORACLE_GATE_DECISION_JSON_NAME,
        oracle_gate_decision,
    )
    summary_path = _write_json(
        output_dir / RECOVERY_BENCHMARK_SUMMARY_JSON_NAME,
        recovery_benchmark_summary,
    )
    split_path = _write_json(
        output_dir / RESULT_SPLIT_DECISION_JSON_NAME,
        result_split_decision,
    )
    return {
        "oracle_conditioned_dev_scorecard_path": str(scorecard_path),
        "oracle_gate_decision_path": str(gate_path),
        "recovery_benchmark_summary_path": str(summary_path),
        "result_split_decision_path": str(split_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = materialize_state_conditioned_oracle_eval(
            dev_dir=args.dev_dir,
            training_dir=args.training_dir,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ALLOWED_NEXT_STEPS",
    "DIAGNOSTIC_METRIC_NAMES",
    "LINE_BASELINE",
    "LINE_C0",
    "LINE_C1",
    "LINE_LABELS",
    "ORACLE_CONDITIONED_DEV_SCORECARD_JSON_NAME",
    "ORACLE_GATE_DECISION_JSON_NAME",
    "RECOVERY_BENCHMARK_SUMMARY_JSON_NAME",
    "RESULT_SPLIT_DECISION_JSON_NAME",
    "SCHEMA_VERSION",
    "_build_empty_hand_transport_payload",
    "_build_same_failure_repeat_payload",
    "_build_verify_hold_payload",
    "build_parser",
    "build_result_split_decision",
    "main",
    "materialize_state_conditioned_oracle_eval",
]


if __name__ == "__main__":
    raise SystemExit(main())
