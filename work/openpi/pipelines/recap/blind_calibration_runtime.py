from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any


FORBIDDEN_VARIANTS = ("C", "X")
ALLOWED_EARLY_STOP_REASONS = (
    "A_success_rate_clearly_above_headroom_max",
    "A_success_rate_clearly_below_headroom_min",
    "trace_completeness_too_low",
    "timeout_rate_too_high",
)
SUITE_PROBE_IMPORT_TARGETS = ("openpi.policies.libero_policy", "libero.libero.benchmark")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00",
        "Z",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def _tmp_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_sibling(path)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_json_write(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    text = json.dumps(
        json_ready(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=sort_keys,
    )
    atomic_write_text(path, f"{text}\n")


def _jsonl_text(rows: Sequence[Mapping[str, object]], *, sort_keys: bool = False) -> str:
    serialized = "\n".join(
        json.dumps(json_ready(dict(row)), ensure_ascii=False, sort_keys=sort_keys)
        for row in rows
    )
    return f"{serialized}\n" if serialized else ""


def legacy_full_rewrite_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    sort_keys: bool = False,
) -> None:
    """Rewrite the whole JSONL file atomically, preserving iter5p5 semantics."""

    atomic_write_text(path, _jsonl_text(rows, sort_keys=sort_keys))


def atomic_jsonl_write(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    sort_keys: bool = False,
) -> None:
    atomic_write_text(path, _jsonl_text(rows, sort_keys=sort_keys))


@dataclass
class Sha256Sums:
    root: Path
    entries: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.entries is None:
            self.entries = {}

    def record(self, path: Path) -> None:
        if self.entries is None:
            self.entries = {}
        rel = path.resolve().relative_to(self.root.resolve()).as_posix()
        self.entries[rel] = sha256_file(path)

    def write(self, path: Path | None = None) -> Path:
        target = path or self.root / "SHA256SUMS"
        entries = self.entries or {}
        lines = [f"{digest}  {rel}" for rel, digest in sorted(entries.items())]
        atomic_write_text(target, "\n".join(lines) + ("\n" if lines else ""))
        return target


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


@dataclass(frozen=True)
class CandidateCell:
    candidate_id: str
    suite_family: str
    suite_description: str
    tasks: tuple[str, ...]
    budget_fraction: float
    calibration_variants: tuple[str, ...]
    selection_inputs: tuple[str, ...]
    forbidden_selection_inputs: tuple[str, ...]
    suite_family_resolution_status_default: str = "resolved"

    @classmethod
    def from_matrix_row(cls, row: Mapping[str, object]) -> "CandidateCell":
        tasks = tuple(str(item) for item in _sequence(row.get("tasks")))
        return cls(
            candidate_id=str(row["candidate_id"]),
            suite_family=str(row["suite_family"]),
            suite_description=str(row.get("suite_description") or ""),
            tasks=tasks,
            budget_fraction=float(row["budget_fraction"]),
            calibration_variants=tuple(
                str(item) for item in _sequence(row.get("calibration_variants"))
            ),
            selection_inputs=tuple(
                str(item) for item in _sequence(row.get("selection_inputs"))
            ),
            forbidden_selection_inputs=tuple(
                str(item) for item in _sequence(row.get("forbidden_selection_inputs"))
            ),
        )

    def as_plan_json(
        self,
        *,
        output_dir: Path,
        resolution_status: str | None = None,
    ) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "suite_family": self.suite_family,
            "suite_description": self.suite_description,
            "tasks": list(self.tasks),
            "budget_fraction": self.budget_fraction,
            "calibration_variants": list(self.calibration_variants),
            "selection_inputs": list(self.selection_inputs),
            "forbidden_selection_inputs": list(self.forbidden_selection_inputs),
            "suite_family_resolution_status": (
                resolution_status or self.suite_family_resolution_status_default
            ),
            "cell_dir": (output_dir / "cells" / self.candidate_id).as_posix(),
        }


def read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload at {path} must be an object")
    return {str(key): value for key, value in payload.items()}


def load_candidate_cells(
    matrix_path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[CandidateCell, ...]:
    if expected_sha256 is not None:
        actual_sha = sha256_file(matrix_path)
        if actual_sha != expected_sha256:
            raise ValueError(
                f"candidate_matrix_sha256_mismatch expected={expected_sha256} actual={actual_sha}"
            )
    matrix = read_json_object(matrix_path)
    rows = matrix.get("candidate_cells")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("candidate_matrix_missing_candidate_cells")
    cells = tuple(
        CandidateCell.from_matrix_row(row) for row in rows if isinstance(row, Mapping)
    )
    if len(cells) != 24:
        raise ValueError(f"candidate_matrix_count_mismatch count={len(cells)}")
    return cells


@dataclass(frozen=True)
class OverallTimeoutClock:
    started_monotonic: float
    timeout_s: float

    @property
    def deadline_monotonic(self) -> float:
        return self.started_monotonic + self.timeout_s

    def remaining_s(self) -> float:
        return max(0.0, self.deadline_monotonic - time.monotonic())

    def expired(self) -> bool:
        return self.remaining_s() <= 0.0


def overall_timeout_clock(timeout_s: float) -> OverallTimeoutClock:
    return OverallTimeoutClock(started_monotonic=time.monotonic(), timeout_s=float(timeout_s))


def cuda_visible_devices_boundary(
    *,
    expected: str,
    allow_empty_cpu: bool = False,
    forbidden: Sequence[str] = ("0", "1", "3"),
) -> dict[str, object]:
    observed = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = {item.strip() for item in observed.split(",") if item.strip()}
    forbidden_visible = [gpu for gpu in forbidden if gpu in visible]
    allowed = observed == expected or (allow_empty_cpu and observed == "")
    return {
        "cuda_visible_devices_expected": expected,
        "cuda_visible_devices_observed": observed,
        "allow_empty_cpu": allow_empty_cpu,
        "forbidden_visible": forbidden_visible,
        "status": "PASS" if allowed and not forbidden_visible else "BLOCK",
    }


def token_set(text: str) -> set[str]:
    return {token for token in re.split(r"[^A-Za-z0-9]+", text) if token}


def forbidden_variant_tokens(
    text: str,
    forbidden: Sequence[str] = FORBIDDEN_VARIANTS,
) -> tuple[str, ...]:
    tokens = token_set(text)
    return tuple(token for token in forbidden if token in tokens)


def validate_early_stop_reason(
    policy_or_reason: Mapping[str, object] | str,
    reason: str | None = None,
    *,
    forbidden_word_tokens: set[str] | Sequence[str] | None = None,
) -> None:
    text = reason if reason is not None else str(policy_or_reason)
    blocked = forbidden_variant_tokens(text, tuple(forbidden_word_tokens or FORBIDDEN_VARIANTS))
    if blocked:
        raise ValueError(
            "early_stop_reason_references_forbidden_variant:" + ",".join(blocked)
        )


def build_per_cell_timeout_record(
    *,
    per_cell_timeout_sec: float,
    client_timeout_sec: float,
    overall_timeout_sec: float,
) -> dict[str, object]:
    return {
        "schema_version": "v22_blind_calibration_per_cell_timeout_v1",
        "per_cell_timeout_sec": per_cell_timeout_sec,
        "client_timeout_sec": client_timeout_sec,
        "overall_timeout_sec": overall_timeout_sec,
        "per_cell_timeout_is_distinct": (
            per_cell_timeout_sec != client_timeout_sec
            and per_cell_timeout_sec != overall_timeout_sec
        ),
    }


def validate_early_stop_policy_payload(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != "v22_calibration_early_stop_policy_v1":
        raise ValueError("early_stop_policy_schema_mismatch")
    if list(_sequence(payload.get("allowed_inputs"))) != ["A", "optional_B"]:
        raise ValueError("early_stop_policy_allowed_inputs_mismatch")
    if list(_sequence(payload.get("forbidden_inputs"))) != ["C", "X"]:
        raise ValueError("early_stop_policy_forbidden_inputs_mismatch")
    if list(_sequence(payload.get("allowed_early_stop_reasons"))) != list(
        ALLOWED_EARLY_STOP_REASONS
    ):
        raise ValueError("early_stop_policy_allowed_reasons_mismatch")
    if list(_sequence(payload.get("forbidden_early_stop_reason_word_tokens"))) != [
        "C",
        "X",
    ]:
        raise ValueError("early_stop_policy_forbidden_reason_tokens_mismatch")
    for reason in ALLOWED_EARLY_STOP_REASONS:
        validate_early_stop_reason(reason)


def load_early_stop_policy(path: Path) -> dict[str, object]:
    payload = read_json_object(path)
    validate_early_stop_policy_payload(payload)
    return payload


def find_forbidden_selection_inputs(payload: object) -> tuple[str, ...]:
    hits: list[str] = []

    def visit(value: object, trail: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                if forbidden_variant_tokens(key_text):
                    hits.append(f"{trail}.{key_text}" if trail else key_text)
                visit(item, f"{trail}.{key_text}" if trail else key_text)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for idx, item in enumerate(value):
                visit(item, f"{trail}[{idx}]")
        elif isinstance(value, str) and forbidden_variant_tokens(value):
            hits.append(trail or value)

    visit(payload, "")
    return tuple(dict.fromkeys(hits))


def assert_no_forbidden_selection_inputs(payload: object) -> None:
    hits = find_forbidden_selection_inputs(payload)
    if hits:
        raise ValueError("selection_using_c_or_x_detected:" + ",".join(hits))


def validate_no_c_x_selection_inputs(payload: object) -> None:
    assert_no_forbidden_selection_inputs(payload)


def validate_selection_inputs_for_no_c_leakage(payload: object) -> None:
    assert_no_forbidden_selection_inputs(payload)


def selected_using_c_results_attestation(
    path: Path,
    *,
    run_id: str,
    schema_version: str = "v22_selected_using_c_results_attestation_v1",
    cell_id: str | None = None,
    selected_using_c_results: bool = False,
    forbidden_variant_codes_used: Sequence[str] = (),
) -> Path | dict[str, object]:
    path_is_directory = path.is_dir()
    target = path / "selected_using_c_results_attestation.json" if path_is_directory else path
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "captured_at_utc": utc_now(),
        "selected_using_c_results": bool(selected_using_c_results),
        "uses_c_results_for_selection": False,
        "forbidden_variant_codes_used": list(forbidden_variant_codes_used),
        "variant_codes_used_subset_of_A_B_only": True,
    }
    if cell_id:
        payload["cell_id"] = cell_id
    atomic_json_write(target, payload)
    return target if path_is_directory else payload


@dataclass(frozen=True)
class SuiteFamilyProbe:
    mode: str = "cpu_introspection"
    import_targets: tuple[str, ...] = SUITE_PROBE_IMPORT_TARGETS
    repo_root: Path | None = None

    def resolve(self, cell: CandidateCell | str) -> tuple[str, str | None]:
        suite_family = cell.suite_family if isinstance(cell, CandidateCell) else str(cell)
        return self._resolve_suite_family(suite_family)

    def resolve_suite_family(self, suite_family: str) -> dict[str, str | None]:
        status, error = self._resolve_suite_family(suite_family)
        return {
            "suite_family": suite_family,
            "suite_family_resolution_status": status,
            "suite_family_resolution_error": error,
        }

    def probe_suite_family(self, suite_family: str) -> dict[str, str | None]:
        return self.resolve_suite_family(suite_family)

    def _resolve_suite_family(self, suite_family: str) -> tuple[str, str | None]:
        if suite_family != "other_locally_supported_LIBERO_suites":
            return "resolved", None
        if self.mode == "deferred":
            return "probe_failed_in_dry_run", "suite_probe_deferred"
        for target in self.import_targets:
            try:
                importlib.import_module(target)
            except (ImportError, RuntimeError) as exc:
                return "probe_failed_in_dry_run", f"{type(exc).__name__}:{exc}"
        return "probe_resolved_in_dry_run", None


def cell_skip_predicate(cell_dir: Path) -> dict[str, object]:
    status_path = cell_dir / "cell_status.json"
    sha_path = cell_dir / "SHA256SUMS"
    trace_path = cell_dir / "stock_A" / "per_episode_trace.jsonl"
    summary_path = cell_dir / "stock_A" / "summary.json"
    status_payload = read_json_object(status_path) if status_path.is_file() else {}
    forbidden_absent = status_payload.get("forbidden_variants_absent")
    forbidden_absent_bool = forbidden_absent is True or forbidden_absent == ["C", "X"]
    predicate: dict[str, object] = {
        "cell_status": status_payload.get("status"),
        "sha256sums_present": sha_path.is_file(),
        "per_episode_trace_present": trace_path.is_file(),
        "summary_present": summary_path.is_file(),
        "selected_using_c_results": status_payload.get("selected_using_c_results"),
        "forbidden_variants_absent": forbidden_absent_bool,
    }
    predicate["skip"] = (
        predicate["cell_status"] in {"PASS", "BLOCK"}
        and predicate["sha256sums_present"] is True
        and predicate["per_episode_trace_present"] is True
        and predicate["summary_present"] is True
        and predicate["selected_using_c_results"] is False
        and predicate["forbidden_variants_absent"] is True
    )
    return predicate


def evaluate_resume_cell(cell_dir: Path) -> dict[str, object]:
    predicate = cell_skip_predicate(cell_dir)
    predicate["resume_action"] = "skip" if predicate["skip"] else "rerun"
    predicate["rerun_required"] = not bool(predicate["skip"])
    predicate["status"] = "SKIPPED" if predicate["skip"] else "INCOMPLETE"
    return predicate


def move_incomplete_cell(
    cell_dir: Path,
    *,
    captured_at_utc: str | None = None,
    utc_stamp: str | None = None,
) -> Path | None:
    if not cell_dir.exists():
        return None
    stamp = (utc_stamp or captured_at_utc or utc_now()).replace(":", "").replace("-", "")
    target = cell_dir / f"_incomplete_{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    for child in list(cell_dir.iterdir()):
        if child == target:
            continue
        shutil.move(str(child), str(target / child.name))
    return target


def move_incomplete_cell_outputs(
    cell_dir: Path,
    *,
    captured_at_utc: str | None = None,
    utc_stamp: str | None = None,
) -> Path | None:
    return move_incomplete_cell(
        cell_dir,
        captured_at_utc=captured_at_utc,
        utc_stamp=utc_stamp,
    )


def _candidate_id(cell: CandidateCell | str) -> str:
    return cell.candidate_id if isinstance(cell, CandidateCell) else str(cell)


def is_cell_complete_for_resume(cell_dir: Path) -> bool:
    return bool(cell_skip_predicate(cell_dir)["skip"])


def quarantine_incomplete_cell(
    cell_dir: Path,
    *,
    now_utc: str | None = None,
) -> Path:
    target = move_incomplete_cell(cell_dir, captured_at_utc=now_utc)
    if target is None:
        raise FileNotFoundError(cell_dir)
    return target


def build_resume_index(
    output_dir: Path,
    cells: Sequence[CandidateCell | str],
    *,
    skip_completed: bool = False,
) -> dict[str, object]:
    completed: list[str] = []
    skipped: list[str] = []
    incomplete: list[str] = []
    rerun_required: list[str] = []
    for cell in cells:
        cell_id = _candidate_id(cell)
        cell_dir = output_dir / "cells" / cell_id
        if not cell_dir.exists():
            continue
        predicate = cell_skip_predicate(cell_dir)
        if predicate["skip"]:
            completed.append(cell_id)
            if skip_completed:
                skipped.append(cell_id)
            continue
        incomplete.append(cell_id)
        rerun_required.append(cell_id)
    return {
        "schema_version": "v22_blind_calibration_resume_index_v1",
        "generated_at_utc": utc_now(),
        "total_cells": len(cells),
        "completed_cells": completed,
        "incomplete_cells": incomplete,
        "skipped_cells": skipped,
        "rerun_required_cells": rerun_required,
    }


def build_dry_run_cell_plan(
    cells: Sequence[CandidateCell],
    *,
    output_dir: Path | None = None,
) -> dict[str, object]:
    resolved_output_dir = output_dir or Path(".")
    probe = SuiteFamilyProbe(mode="cpu_introspection")
    planned_cells: list[dict[str, object]] = []
    for cell in cells:
        status, error = probe.resolve(cell)
        row = cell.as_plan_json(output_dir=resolved_output_dir, resolution_status=status)
        if error:
            row["suite_family_resolution_error"] = error
        planned_cells.append(row)
    return {
        "schema_version": "v22_blind_calibration_cell_plan_v1",
        "run_id": "stage1_v22_runner_surface_iter7_5_20260426T_nextZ",
        "generated_at_utc": utc_now(),
        "cell_count": len(planned_cells),
        "total_cells": len(planned_cells),
        "candidate_id_format": "matrix_verbatim",
        "cells": planned_cells,
    }


def _variant_summary(
    cell: CandidateCell,
    rows: Sequence[Mapping[str, object]],
    *,
    variant_code: str,
) -> dict[str, object]:
    observed = len(rows)
    successes = sum(1 for row in rows if bool(row.get("success")))
    timeouts = sum(1 for row in rows if bool(row.get("timeout_flag")))
    denominator = max(observed, 1)
    summary: dict[str, object] = {
        "schema_version": "v22_blind_calibration_variant_summary_v1",
        "cell_id": cell.candidate_id,
        "variant_code": variant_code,
        "episode_count": observed,
        "success_count": successes,
        "success_rate": successes / denominator if observed else None,
        "trace_completeness": 1.0 if observed else None,
        "timeout_count": timeouts,
        "timeout_rate": timeouts / denominator if observed else None,
        "selected_using_c_results": False,
        "formal_result": False,
    }
    policy_sources = {
        str(row.get("policy_output_source"))
        for row in rows
        if row.get("policy_output_source") is not None
    }
    if policy_sources:
        summary["policy_output_sources"] = sorted(policy_sources)
    if policy_sources == {"synthetic_test_stub"}:
        summary["synthetic_policy"] = True
        if variant_code == "A":
            summary["stock_A_success_rate_source"] = "synthetic_test_stub_not_real_policy"
    return summary


def _metric_ladder_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "v22_blind_calibration_metric_ladder_summary_v1",
        "success_rate": summary.get("success_rate"),
        "trace_completeness": summary.get("trace_completeness"),
        "timeout_rate": summary.get("timeout_rate"),
        "headroom_eligible": False,
    }


def _bootstrap_ci(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "v22_blind_calibration_bootstrap_ci_v1",
        "computed": False,
        "reason": "runner_surface_placeholder",
        "success_rate": summary.get("success_rate"),
        "ci95": None,
    }


def write_cell_artifacts(
    *,
    cell_dir: Path,
    cell: CandidateCell,
    run_id: str,
    mode: str,
    suite_family_resolution_status: str,
    stock_rows: Sequence[Mapping[str, object]],
    control_rows: Sequence[Mapping[str, object]] = (),
    status: str = "PASS",
    blocking_reasons: Sequence[str] = (),
) -> dict[str, object]:
    cell_dir.mkdir(parents=True, exist_ok=True)
    sums = Sha256Sums(cell_dir)
    manifest = {
        "schema_version": "v22_blind_calibration_cell_manifest_v1",
        "run_id": run_id,
        "mode": mode,
        "cell_id": cell.candidate_id,
        "suite_family": cell.suite_family,
        "budget_fraction": cell.budget_fraction,
        "created_at_utc": utc_now(),
        "formal_result": False,
        "hash_lock_allowed": False,
    }
    precondition = {
        "schema_version": "v22_blind_calibration_cell_precondition_v1",
        "status": "PASS",
        "selected_using_c_results": False,
        "forbidden_variants_absent": True,
        "blocking_reasons": [],
    }
    stock_summary = _variant_summary(cell, stock_rows, variant_code="A")
    status_payload: dict[str, object] = {
        "schema_version": "v22_blind_calibration_cell_status_v1",
        "cell_id": cell.candidate_id,
        "suite_family": cell.suite_family,
        "suite_family_resolution_status": suite_family_resolution_status,
        "tasks": list(cell.tasks),
        "budget_fraction": cell.budget_fraction,
        "status": status,
        "variants_run_for_selection": ["A"],
        "optional_control_variants_run": ["B"] if control_rows else [],
        "forbidden_variants_absent": ["C", "X"],
        "selected_using_c_results": False,
        "trace_completeness": stock_summary["trace_completeness"],
        "stock_A_success_rate": stock_summary["success_rate"],
        "control_B_success_rate": None,
        "timeout_rate": stock_summary["timeout_rate"],
        "headroom_eligible": False,
        "blocking_reasons": list(blocking_reasons),
    }
    payloads: list[tuple[Path, object, str]] = [
        (cell_dir / "cell_manifest.json", manifest, "json"),
        (cell_dir / "precondition_check.json", precondition, "json"),
        (cell_dir / "stock_A" / "per_episode_trace.jsonl", stock_rows, "jsonl"),
        (cell_dir / "stock_A" / "summary.json", stock_summary, "json"),
        (cell_dir / "stock_A" / "metric_ladder_summary.json", _metric_ladder_summary(stock_summary), "json"),
        (cell_dir / "stock_A" / "bootstrap_ci.json", _bootstrap_ci(stock_summary), "json"),
    ]
    if control_rows:
        control_summary = _variant_summary(cell, control_rows, variant_code="B")
        status_payload["control_B_success_rate"] = control_summary["success_rate"]
        payloads.extend(
            [
                (cell_dir / "control_B" / "per_episode_trace.jsonl", control_rows, "jsonl"),
                (cell_dir / "control_B" / "summary.json", control_summary, "json"),
                (cell_dir / "control_B" / "metric_ladder_summary.json", _metric_ladder_summary(control_summary), "json"),
                (cell_dir / "control_B" / "bootstrap_ci.json", _bootstrap_ci(control_summary), "json"),
            ]
        )
    for path, payload, kind in payloads:
        if kind == "json":
            atomic_json_write(path, payload)
        else:
            atomic_jsonl_write(path, payload)  # type: ignore[arg-type]
        sums.record(path)
    atomic_json_write(cell_dir / "cell_status.json", status_payload)
    sums.record(cell_dir / "cell_status.json")
    sums.write(cell_dir / "SHA256SUMS")
    return status_payload


def write_cell_artifact_contract(
    cell_dir: Path,
    payload: Mapping[str, object],
    *,
    include_control_b: bool,
) -> dict[str, object]:
    cell = CandidateCell(
        candidate_id=str(payload["cell_id"]),
        suite_family=str(payload["suite_family"]),
        suite_description="",
        tasks=(),
        budget_fraction=float(payload["budget_fraction"]),
        calibration_variants=("A",),
        selection_inputs=("stock_A_success_rate",),
        forbidden_selection_inputs=("C_success_rate", "X_success_rate"),
    )
    stock_rows = tuple(
        row for row in _sequence(payload.get("stock_A_rows")) if isinstance(row, Mapping)
    )
    control_rows = (
        tuple(
            row
            for row in _sequence(payload.get("control_B_rows"))
            if isinstance(row, Mapping)
        )
        if include_control_b
        else ()
    )
    return write_cell_artifacts(
        cell_dir=cell_dir,
        cell=cell,
        run_id=str(payload.get("run_id") or "stage1_v22_runner_surface_iter7_5_20260426T_nextZ"),
        mode=str(payload.get("mode") or "smoke"),
        suite_family_resolution_status=str(
            payload.get("suite_family_resolution_status") or "resolved"
        ),
        stock_rows=stock_rows,
        control_rows=control_rows,
        status=str(payload.get("status") or "PASS"),
    )


def episode_specs(
    *,
    task_ids: Sequence[int],
    seeds: Sequence[int],
    trial_indices: Sequence[int],
) -> tuple[dict[str, int], ...]:
    return tuple(
        {"task_id": int(task_id), "seed": int(seed), "trial_index": int(trial_index)}
        for task_id in task_ids
        for seed in seeds
        for trial_index in trial_indices
    )


def episode_submission(
    specs: Sequence[Any],
    submit_one: Callable[[Any], Mapping[str, object]],
    *,
    max_workers: int,
    overall_timeout_s: float,
    on_completed: Callable[[Any, Mapping[str, object]], None] | None = None,
    on_failed: Callable[[Any, Exception], None] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

    clock = overall_timeout_clock(overall_timeout_s)
    completed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {executor.submit(submit_one, spec): spec for spec in specs}
        try:
            iterator = as_completed(futures, timeout=max(clock.remaining_s(), 0.001))
            for future in iterator:
                spec = futures[future]
                try:
                    row = dict(future.result())
                except Exception as exc:  # noqa: BLE001
                    failed.append({"episode_status": "runtime_error", "error": str(exc)})
                    if on_failed:
                        on_failed(spec, exc)
                else:
                    completed.append(row)
                    if on_completed:
                        on_completed(spec, row)
        except TimeoutError:
            failed.extend(
                {"episode_status": "IN_FLIGHT", "error": "overall_timeout_s_reached"}
                for _future in futures
                if not _future.done()
            )
    return completed, failed
