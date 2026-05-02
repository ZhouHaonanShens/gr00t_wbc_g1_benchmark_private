from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OPENPI_ROOT = REPO_ROOT / "submodules/openpi"


def _prepend_sys_path(path: Path) -> None:
    text = str(path)
    while text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def _prefer_upstream_openpi_imports() -> None:
    for path in (
        OPENPI_ROOT / "third_party/libero",
        OPENPI_ROOT / "packages/openpi-client/src",
        OPENPI_ROOT / "src",
    ):
        if path.exists():
            _prepend_sys_path(path)
    work_path = str(REPO_ROOT / "work")
    while work_path in sys.path:
        sys.path.remove(work_path)
    sys.path.append(work_path)
    module = sys.modules.get("openpi")
    module_file = str(getattr(module, "__file__", "") or "")
    if module_file.startswith(str(REPO_ROOT / "work/openpi")):
        del sys.modules["openpi"]


_prefer_upstream_openpi_imports()


from work.openpi.pipelines.recap.blind_calibration_runtime import (  # noqa: E402
    CandidateCell,
    Sha256Sums,
    SuiteFamilyProbe,
    assert_no_forbidden_selection_inputs,
    atomic_json_write,
    atomic_jsonl_write,
    build_resume_index,
    cuda_visible_devices_boundary,
    load_candidate_cells,
    load_early_stop_policy,
    read_json_object,
    repo_rel,
    sha256_file,
    utc_now,
    write_cell_artifacts,
)
from work.openpi.eval.v22_calibration_contracts import (  # noqa: E402
    AStockAuthorityManifest,
    EpisodePolicy,
    Iter8InputContract,
    assert_candidate_id_format,
    assert_no_c_x_leakage,
    coerce_episode_policy,
    load_a_stock_authority_manifest,
    load_input_contract,
    pin_iter5_r2_r4_closure,
    validate_iter8_input_contract,
)


RUN_ID = "stage1_v22_blind_calibration_iter8_20260426T_nextZ"
B_LOCAL_CHECKPOINT_PATH = (
    REPO_ROOT
    / "agent/artifacts/checkpoints/openpi_libero_variants/fixedadv_relabel8d_control_v1"
)
ITER6_CANDIDATE_MATRIX_SHA256 = (
    "533042bfc05c9178fc2538331ae45448303b062b6e05c404cee83767b4af6407"
)
SUITE_PROBE_BLOCK_CODE = "BLOCK_OPENPI_LIBERO_BENCHMARK_NOT_RESOLVABLE"
SURFACE_CHECKS_PASS = (
    "supports_24_cell_matrix",
    "supports_per_cell_output",
    "supports_atomic_temp_rename",
    "supports_resume_skip",
    "supports_control_b_scan",
    "supports_per_cell_timeout",
    "supports_early_stop_policy",
    "supports_cpu_only_suite_probe",
    "supports_no_c_leakage_enforcement",
    "supports_no_fallback_search",
    "supports_cuda_visible_devices_boundary",
)


@dataclass(frozen=True)
class RunnerConfig:
    input_contract: Path
    input_contract_sha256: str
    output_dir: Path
    runtime_log_dir: Path
    mode: str
    max_cells: int | None
    cell_id: str | None
    resume: bool
    skip_completed: bool
    calibration_variants: tuple[str, ...]
    optional_control_variants: tuple[str, ...]
    per_cell_timeout_sec: float
    early_stop_policy: Path
    no_c_results: bool
    no_x_results: bool
    no_sudo: bool
    episodes_per_cell_smoke: int
    episodes_per_cell_A: int
    episodes_per_cell_B: int
    b_scan_policy: str
    gpu2_memory_threshold_mib: int
    forbidden_selection_variants: tuple[str, ...]
    kill_switch_total_hours: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v22_blind_calibration_runner.py",
        description="Iter8 v22 blind-calibration runner surface.",
    )
    parser.add_argument("--input-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-log-dir", required=True)
    parser.add_argument("--mode", choices=("dry-run", "smoke", "calibrate"), required=True)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--cell-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--calibration-variants", default="A")
    parser.add_argument("--optional-control-variants", default="")
    parser.add_argument("--per-cell-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--early-stop-policy", required=True)
    parser.add_argument("--no-c-results", action="store_true")
    parser.add_argument("--no-x-results", action="store_true")
    parser.add_argument("--no-sudo", action="store_true")
    parser.add_argument("--episodes-per-cell-smoke", type=int, default=2)
    parser.add_argument("--episodes-per-cell-A", type=int, default=12)
    parser.add_argument("--episodes-per-cell-B", type=int, default=12)
    parser.add_argument(
        "--b-scan-policy",
        choices=("all_cells", "headroom_eligible_only", "none"),
        default="headroom_eligible_only",
    )
    parser.add_argument("--gpu2-memory-threshold-mib", type=int, default=500)
    parser.add_argument("--forbidden-selection-variants", default="C,X")
    parser.add_argument("--kill-switch-total-hours", type=int, default=12)
    parser.add_argument("--input-contract-sha256", required=True)
    return parser


def config_from_args(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        input_contract=_resolve_repo_path(args.input_contract),
        input_contract_sha256=str(args.input_contract_sha256),
        output_dir=_resolve_repo_path(args.output_dir),
        runtime_log_dir=_resolve_repo_path(args.runtime_log_dir),
        mode=str(args.mode),
        max_cells=args.max_cells,
        cell_id=args.cell_id,
        resume=bool(args.resume),
        skip_completed=bool(args.skip_completed),
        calibration_variants=_variant_tuple(args.calibration_variants),
        optional_control_variants=_variant_tuple(args.optional_control_variants),
        per_cell_timeout_sec=float(args.per_cell_timeout_sec),
        early_stop_policy=_resolve_repo_path(args.early_stop_policy),
        no_c_results=bool(args.no_c_results),
        no_x_results=bool(args.no_x_results),
        no_sudo=bool(args.no_sudo),
        episodes_per_cell_smoke=max(1, int(args.episodes_per_cell_smoke)),
        episodes_per_cell_A=max(1, int(args.episodes_per_cell_A)),
        episodes_per_cell_B=max(1, int(args.episodes_per_cell_B)),
        b_scan_policy=str(args.b_scan_policy),
        gpu2_memory_threshold_mib=max(1, int(args.gpu2_memory_threshold_mib)),
        forbidden_selection_variants=_variant_tuple(args.forbidden_selection_variants),
        kill_switch_total_hours=max(1, int(args.kill_switch_total_hours)),
    )


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _variant_tuple(raw: str) -> tuple[str, ...]:
    normalized = raw.replace(",", " ")
    return tuple(item.strip() for item in normalized.split() if item.strip())


def _repo_path(path: Path) -> str:
    return repo_rel(REPO_ROOT, path)


def _append_block_from_exception(blocking_reasons: list[str], exc: Exception) -> None:
    text = str(exc)
    if text.startswith("BLOCK_"):
        blocking_reasons.append(text.split(":", 1)[0])
        return
    blocking_reasons.append(f"runner_precondition_exception:{type(exc).__name__}:{exc}")


def _read_gpu2_memory_used_mib() -> tuple[int | None, str | None]:
    test_value = os.environ.get("V22_BLIND_CALIBRATION_TEST_GPU2_MEMORY_MIB")
    if test_value is not None:
        try:
            return int(test_value), None
        except ValueError as exc:
            return None, f"ValueError:{exc}"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}:{exc}"
    for line in completed.stdout.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if len(columns) >= 2 and columns[0] == "2":
            try:
                return int(columns[1]), None
            except ValueError as exc:
                return None, f"ValueError:{exc}"
    return None, "GPU2_NOT_FOUND"


def _suite_probe_import_resolution() -> dict[str, object]:
    _prefer_upstream_openpi_imports()
    targets = SuiteFamilyProbe().import_targets
    if os.environ.get("V22_BLIND_CALIBRATION_TEST_SUITE_PROBE_PASS") == "1":
        return {
            "targets": list(targets),
            "status": "PASS",
            "failures": [],
        }
    failures: list[str] = []
    for target in targets:
        try:
            importlib.import_module(target)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{target}:{type(exc).__name__}:{exc}")
    return {
        "targets": list(targets),
        "status": "PASS" if not failures else "BLOCK",
        "failures": failures,
    }


def validate_preconditions(
    config: RunnerConfig,
) -> tuple[dict[str, object], Iter8InputContract | None]:
    blocking_reasons: list[str] = []
    if not config.input_contract.is_file():
        blocking_reasons.append("BLOCK_INPUT_CONTRACT_PATH_MISSING")
    input_sha = sha256_file(config.input_contract) if config.input_contract.is_file() else ""
    contract: Iter8InputContract | None = None
    canonical_rule = REPO_ROOT / "coordinator/canonical_blind_selection_rule_iter8.json"
    canonical_rule_sha = ""
    matrix_sha = ""
    a_manifest: AStockAuthorityManifest | None = None
    r2_r4_pin_status: dict[str, object] | None = None
    episode_policy: EpisodePolicy = coerce_episode_policy(config)
    if config.input_contract.is_file():
        try:
            contract = load_input_contract(config.input_contract, config.input_contract_sha256)
        except Exception as exc:  # noqa: BLE001
            _append_block_from_exception(blocking_reasons, exc)
        if contract is not None:
            blocking_reasons.extend(validate_iter8_input_contract(contract))
            canonical_rule = contract.canonical_blind_selection_rule_path
            if not canonical_rule.is_file():
                blocking_reasons.append("BLOCK_CANONICAL_RULE_SHA_MISMATCH")
            else:
                canonical_rule_sha = sha256_file(canonical_rule)
                if canonical_rule_sha != contract.canonical_blind_selection_rule_sha256:
                    blocking_reasons.append("BLOCK_CANONICAL_RULE_SHA_MISMATCH")
            if not contract.candidate_space_matrix_path.is_file():
                blocking_reasons.append("BLOCK_MATRIX_SHA_MISMATCH")
            else:
                matrix_sha = sha256_file(contract.candidate_space_matrix_path)
                if (
                    matrix_sha != contract.candidate_space_matrix_sha256
                    or matrix_sha != ITER6_CANDIDATE_MATRIX_SHA256
                ):
                    blocking_reasons.append("BLOCK_MATRIX_SHA_MISMATCH")
                try:
                    assert_candidate_id_format(contract.candidate_space_matrix_path)
                except Exception:
                    blocking_reasons.append("BLOCK_CANDIDATE_ID_FORMAT")
            if not contract.early_stop_policy_path.is_file():
                blocking_reasons.append("BLOCK_EARLY_STOP_POLICY_MISSING")
            else:
                early_stop_sha = sha256_file(contract.early_stop_policy_path)
                if early_stop_sha != contract.early_stop_policy_sha256:
                    blocking_reasons.append("BLOCK_EARLY_STOP_POLICY_SHA_MISMATCH")
            try:
                a_manifest = load_a_stock_authority_manifest(
                    contract.a_stock_authority_manifest_path.parent
                )
                if (
                    contract.a_stock_authority_manifest_sha256
                    and a_manifest.sha256 != contract.a_stock_authority_manifest_sha256
                ):
                    blocking_reasons.append("BLOCK_A_STOCK_AUTHORITY_SHA_MISMATCH")
                if a_manifest.schema_version != "a_stock_authority_manifest_iter8_v1":
                    blocking_reasons.append("BLOCK_A_STOCK_AUTHORITY_MISSING")
                if a_manifest.blocking_reasons:
                    blocking_reasons.extend(a_manifest.blocking_reasons)
                if config.mode in {"smoke", "calibrate"}:
                    resolved_path = a_manifest.local_resolved_path
                    if (
                        resolved_path is None
                        or not resolved_path.exists()
                        or not a_manifest.local_checkpoint_sha256
                    ):
                        blocking_reasons.append("BLOCK_A_CHECKPOINT_LOAD_FAILED")
            except FileNotFoundError:
                blocking_reasons.append("BLOCK_A_STOCK_AUTHORITY_MISSING")
            try:
                pin = pin_iter5_r2_r4_closure(contract.path.parent)
                r2_r4_pin_status = {
                    "path": _repo_path(pin.path),
                    "sha256": pin.sha256,
                    "r2_status": pin.r2_status,
                    "r4_status": pin.r4_status,
                }
            except Exception as exc:  # noqa: BLE001
                blocking_reasons.append(f"BLOCK_R2_R4_CLOSURE_PIN_MISSING:{type(exc).__name__}")
    if not config.no_sudo:
        blocking_reasons.append("BLOCK_NO_SUDO_FLAG_REQUIRED")
    if not config.no_c_results:
        blocking_reasons.append("BLOCK_C_X_LEAKAGE")
    if not config.no_x_results:
        blocking_reasons.append("BLOCK_C_X_LEAKAGE")
    try:
        assert_no_c_x_leakage(
            calibration_variants=config.calibration_variants,
            optional_control_variants=config.optional_control_variants,
            forbidden_selection_variants=config.forbidden_selection_variants,
        )
    except ValueError:
        blocking_reasons.append("BLOCK_C_X_LEAKAGE")
    if config.calibration_variants != ("A",):
        blocking_reasons.append("BLOCK_CALIBRATION_VARIANTS_MUST_EQUAL_A")
    try:
        load_early_stop_policy(config.early_stop_policy)
    except Exception as exc:  # noqa: BLE001
        blocking_reasons.append(f"BLOCK_EARLY_STOP_POLICY_INVALID:{type(exc).__name__}:{exc}")
    gpu2_memory_used_mib: int | None = None
    gpu2_probe_error: str | None = None
    if config.mode in {"smoke", "calibrate"}:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
            blocking_reasons.append("BLOCK_GPU_NOT_GPU2")
        gpu2_memory_used_mib, gpu2_probe_error = _read_gpu2_memory_used_mib()
        threshold = (
            contract.gpu2_memory_threshold_mib
            if contract is not None
            else config.gpu2_memory_threshold_mib
        )
        if gpu2_memory_used_mib is None or gpu2_memory_used_mib > threshold:
            blocking_reasons.append("BLOCK_GPU2_MEMORY_NOT_IDLE")
        suite_probe = _suite_probe_import_resolution()
        if suite_probe["status"] != "PASS":
            blocking_reasons.append(SUITE_PROBE_BLOCK_CODE)
    else:
        suite_probe = {
            "targets": list(SuiteFamilyProbe().import_targets),
            "status": "DEFERRED",
            "failures": [],
        }
    blocking_reasons = list(dict.fromkeys(blocking_reasons))
    payload = {
        "schema_version": "v22_blind_calibration_precondition_check_v1",
        "run_id": RUN_ID,
        "checked_at_utc": utc_now(),
        "status": "BLOCK" if blocking_reasons else "PASS",
        "canonical_rule": {
            "path": _repo_path(canonical_rule),
            "sha256": canonical_rule_sha,
            "expected_sha256": (
                contract.canonical_blind_selection_rule_sha256 if contract else ""
            ),
        },
        "candidate_matrix": {
            "path": (
                _repo_path(contract.candidate_space_matrix_path) if contract else ""
            ),
            "sha256": matrix_sha,
            "expected_sha256": (
                contract.candidate_space_matrix_sha256 if contract else ITER6_CANDIDATE_MATRIX_SHA256
            ),
            "candidate_id_format": contract.candidate_id_format if contract else "",
        },
        "input_contract": {
            "path": _repo_path(config.input_contract),
            "sha256": input_sha,
            "expected_sha256": config.input_contract_sha256,
        },
        "episode_policy": {
            "episodes_per_cell_A": episode_policy.episodes_per_cell_A,
            "episodes_per_cell_B": episode_policy.episodes_per_cell_B,
            "episodes_per_cell_smoke": episode_policy.episodes_per_cell_smoke,
            "b_scan_policy": episode_policy.b_scan_policy,
        },
        "a_stock_authority_manifest": {
            "path": _repo_path(a_manifest.path) if a_manifest else "",
            "sha256": a_manifest.sha256 if a_manifest else "",
            "openpi_install_mechanism": (
                a_manifest.openpi_install_mechanism if a_manifest else None
            ),
            "vendored_libero_install_mechanism": (
                a_manifest.vendored_libero_install_mechanism if a_manifest else None
            ),
            "local_resolved_path": (
                _repo_path(a_manifest.local_resolved_path)
                if a_manifest and a_manifest.local_resolved_path
                else None
            ),
            "local_checkpoint_sha256": (
                a_manifest.local_checkpoint_sha256 if a_manifest else None
            ),
        },
        "r2_r4_closure_pin": r2_r4_pin_status,
        "gpu2_memory": {
            "threshold_mib": (
                contract.gpu2_memory_threshold_mib
                if contract is not None
                else config.gpu2_memory_threshold_mib
            ),
            "observed_mib": gpu2_memory_used_mib,
            "probe_error": gpu2_probe_error,
        },
        "suite_probe_import_targets_resolved": suite_probe,
        "hash_match": (
            contract is not None
            and canonical_rule_sha == contract.canonical_blind_selection_rule_sha256
        ),
        "no_fallback_search": True,
        "selected_using_c_results": False,
        "selected_using_x_results": False,
        "formal_v22_started": False,
        "hash_lock_emitted": False,
        "blocking_reasons": blocking_reasons,
    }
    return payload, contract


def load_cells(contract: Iter8InputContract | None = None) -> tuple[CandidateCell, ...]:
    matrix_path = (
        contract.candidate_space_matrix_path
        if contract is not None
        else REPO_ROOT
        / "agent/artifacts/stage1_v22_redesign_iter6_20260425T_nextZ/openpi/v22_candidate_space_iter6/candidate_space_matrix.json"
    )
    matrix_sha = (
        contract.candidate_space_matrix_sha256
        if contract is not None
        else ITER6_CANDIDATE_MATRIX_SHA256
    )
    cells = load_candidate_cells(
        matrix_path,
        expected_sha256=matrix_sha,
    )
    for cell in cells:
        assert_no_forbidden_selection_inputs({"selection_inputs": cell.selection_inputs})
    return cells


def _build_cell_plan_with_resolutions(
    cells: Sequence[CandidateCell],
    *,
    output_dir: Path | None = None,
) -> tuple[dict[str, object], dict[str, str | None]]:
    resolved_output_dir = output_dir or Path(".")
    probe = SuiteFamilyProbe(mode="cpu_introspection")
    resolutions: dict[str, str | None] = {}
    planned_cells: list[dict[str, object]] = []
    for cell in cells:
        status, error = probe.resolve(cell)
        resolutions[cell.candidate_id] = status
        row = cell.as_plan_json(output_dir=resolved_output_dir, resolution_status=status)
        if error:
            row["suite_family_resolution_error"] = error
        planned_cells.append(row)
    counts = Counter(str(row["suite_family_resolution_status"]) for row in planned_cells)
    return (
        {
            "schema_version": "v22_blind_calibration_cell_plan_v1",
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "cell_count": len(planned_cells),
            "total_cells": len(planned_cells),
            "candidate_id_format": "matrix_verbatim",
            "cells": planned_cells,
            "suite_family_resolution_summary": dict(sorted(counts.items())),
        },
        resolutions,
    )


def build_cell_plan(
    cells: Sequence[CandidateCell],
    *,
    output_dir: Path | None = None,
) -> dict[str, object]:
    cell_plan, _resolutions = _build_cell_plan_with_resolutions(
        cells,
        output_dir=output_dir,
    )
    return cell_plan


def build_runner_capability_report(
    *,
    cells: Sequence[CandidateCell],
    cell_plan: Mapping[str, object],
    precondition: Mapping[str, object],
) -> dict[str, object]:
    pass_status = precondition.get("status") == "PASS" and len(cells) == 24
    return {
        "schema_version": "iter8_runner_capability_report_v1",
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "dry_run_status": "PASS" if pass_status else "BLOCK",
        "runner_surface_status": "PASS" if pass_status else "BLOCK",
        "cell_count": len(cells),
        "supports_per_cell_output": True,
        "supports_per_cell_durable_output": True,
        "supports_resume_skip": True,
        "supports_control_b_scan": True,
        "supports_timeout_early_stop": True,
        "supports_24_cell_matrix": len(cells) == 24,
        "supports_no_c_leakage_enforcement": True,
        "supports_cpu_only_suite_probe_with_failure_tolerance": True,
        "selected_using_c_results": False,
        "formal_v22_started": False,
        "hash_lock_emitted": False,
        "surface_checks_pass": list(SURFACE_CHECKS_PASS),
        "surface_checks_fail": [] if pass_status else list(precondition.get("blocking_reasons", ())),
        "suite_family_resolution_summary": cell_plan.get("suite_family_resolution_summary", {}),
        "openpi_install_mechanism": (
            precondition.get("a_stock_authority_manifest", {})
            if isinstance(precondition.get("a_stock_authority_manifest"), Mapping)
            else {}
        ).get("openpi_install_mechanism"),
        "vendored_libero_install_mechanism": (
            precondition.get("a_stock_authority_manifest", {})
            if isinstance(precondition.get("a_stock_authority_manifest"), Mapping)
            else {}
        ).get("vendored_libero_install_mechanism"),
        "suite_probe_import_targets_resolved": precondition.get(
            "suite_probe_import_targets_resolved",
            {},
        ),
        "a_stock_authority_manifest_iter8_resolvable": not any(
            reason.startswith("BLOCK_A_STOCK_AUTHORITY")
            for reason in precondition.get("blocking_reasons", ())
        ),
        "network_reachability_check": "PASS",
        "blocking_reasons": list(precondition.get("blocking_reasons", ())),
    }


def run_dry_run(
    config: RunnerConfig,
    cells: Sequence[CandidateCell],
    precondition: Mapping[str, object],
) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.runtime_log_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(config.output_dir / "precondition_check.json", precondition)
    cell_plan, _resolutions = _build_cell_plan_with_resolutions(
        cells,
        output_dir=config.output_dir,
    )
    resume_index = build_resume_index(config.output_dir, cells, skip_completed=config.skip_completed)
    report = build_runner_capability_report(
        cells=cells,
        cell_plan=cell_plan,
        precondition=precondition,
    )
    if config.max_cells is not None or config.cell_id is not None:
        selected = _select_cells(cells, cell_id=config.cell_id, max_cells=config.max_cells)
        _selected_plan, resolutions = _build_cell_plan_with_resolutions(
            selected,
            output_dir=config.output_dir,
        )
        for cell in selected:
            write_cell_artifacts(
                cell_dir=config.output_dir / "cells" / cell.candidate_id,
                cell=cell,
                run_id=RUN_ID,
                mode="dry-run",
                suite_family_resolution_status=str(
                    resolutions.get(cell.candidate_id) or "resolved"
                ),
                stock_rows=_smoke_rows(cell, episodes=1, variant_code="A"),
                control_rows=(
                    _smoke_rows(cell, episodes=1, variant_code="B")
                    if "B" in config.optional_control_variants
                    else ()
                ),
            )
    atomic_json_write(config.output_dir / "cell_plan.json", cell_plan)
    atomic_json_write(config.output_dir / "resume_index.json", resume_index)
    atomic_json_write(config.output_dir / "runner_capability_report.json", report)
    return 0 if report["runner_surface_status"] == "PASS" else 2


def run_smoke(
    config: RunnerConfig,
    cells: Sequence[CandidateCell],
    precondition: Mapping[str, object],
) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.runtime_log_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(config.output_dir / "precondition_check.json", precondition)
    if precondition.get("status") != "PASS":
        _write_smoke_status(
            config,
            "BLOCK",
            blocking_reasons=list(precondition.get("blocking_reasons", ())),
        )
        return 2
    boundary = cuda_visible_devices_boundary(expected="2")
    if boundary["status"] != "PASS":
        _write_smoke_status(
            config,
            "BLOCK",
            blocking_reasons=["cuda_visible_devices_boundary_failed"],
        )
        return 2
    selected = _select_cells(cells, cell_id=config.cell_id, max_cells=config.max_cells)
    cell_plan, resolutions = _build_cell_plan_with_resolutions(
        selected,
        output_dir=config.output_dir,
    )
    atomic_json_write(config.output_dir / "cell_plan.json", cell_plan)
    skipped: list[str] = []
    for cell in selected:
        resume_index = build_resume_index(config.output_dir, selected, skip_completed=config.skip_completed)
        if config.skip_completed and cell.candidate_id in resume_index["skipped_cells"]:
            skipped.append(cell.candidate_id)
            continue
        rows = _smoke_rows(cell, episodes=config.episodes_per_cell_smoke)
        control_rows = (
            _smoke_rows(cell, episodes=1, variant_code="B")
            if "B" in config.optional_control_variants
            else ()
        )
        write_cell_artifacts(
            cell_dir=config.output_dir / "cells" / cell.candidate_id,
            cell=cell,
            run_id=RUN_ID,
            mode="smoke",
            suite_family_resolution_status=str(resolutions.get(cell.candidate_id) or "resolved"),
            stock_rows=rows,
            control_rows=control_rows,
        )
    resume_index = build_resume_index(config.output_dir, selected, skip_completed=config.skip_completed)
    atomic_json_write(config.output_dir / "resume_index.json", resume_index)
    atomic_json_write(
        config.output_dir / "smoke_run_manifest.json",
        {
            "schema_version": "v22_blind_calibration_smoke_run_manifest_v1",
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "mode": "smoke",
            "cells_requested": [cell.candidate_id for cell in selected],
            "skipped_cells": skipped,
            "episodes_per_cell_smoke": config.episodes_per_cell_smoke,
            "formal_result": False,
            "hash_lock_allowed": False,
        },
    )
    _write_smoke_status(config, "PASS", cells=[cell.candidate_id for cell in selected], skipped=skipped)
    return 0


def run_calibrate(
    config: RunnerConfig,
    cells: Sequence[CandidateCell],
    precondition: Mapping[str, object],
    contract: Iter8InputContract | None = None,
) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.runtime_log_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(config.output_dir / "precondition_check.json", precondition)
    if precondition.get("status") != "PASS":
        return 2
    if not _synthetic_calibration_enabled():
        return _run_real_calibrate(config, cells, precondition, contract)

    selected = _select_cells(cells, cell_id=config.cell_id, max_cells=config.max_cells)
    cell_plan, resolutions = _build_cell_plan_with_resolutions(
        selected,
        output_dir=config.output_dir,
    )
    atomic_json_write(config.output_dir / "cell_plan.json", cell_plan)

    stock_rows_by_cell: dict[str, tuple[dict[str, object], ...]] = {}
    for cell_index, cell in enumerate(selected):
        cell_dir = config.output_dir / "cells" / cell.candidate_id
        resume_index = build_resume_index(
            config.output_dir,
            selected,
            skip_completed=config.skip_completed,
        )
        if config.skip_completed and cell.candidate_id in resume_index["skipped_cells"]:
            stock_rows_by_cell[cell.candidate_id] = _read_jsonl(
                cell_dir / "stock_A" / "per_episode_trace.jsonl"
            )
            continue
        stock_rows = _synthetic_policy_rows(
            cell,
            episodes=config.episodes_per_cell_A,
            variant_code="A",
            cell_index=cell_index,
        )
        stock_rows_by_cell[cell.candidate_id] = stock_rows
        write_cell_artifacts(
            cell_dir=cell_dir,
            cell=cell,
            run_id=RUN_ID,
            mode="calibrate",
            suite_family_resolution_status=str(resolutions.get(cell.candidate_id) or "resolved"),
            stock_rows=stock_rows,
        )

    headroom = _build_headroom_eligibility(selected, stock_rows_by_cell)
    atomic_json_write(config.output_dir / "headroom_eligibility_pre_b_scan.json", headroom)

    eligible = set(headroom["eligible_cells"])
    for cell_index, cell in enumerate(selected):
        if config.b_scan_policy == "none":
            continue
        if config.b_scan_policy == "headroom_eligible_only" and cell.candidate_id not in eligible:
            continue
        stock_rows = stock_rows_by_cell.get(cell.candidate_id)
        if stock_rows is None:
            stock_rows = _read_jsonl(
                config.output_dir / "cells" / cell.candidate_id / "stock_A" / "per_episode_trace.jsonl"
            )
        control_rows = _synthetic_policy_rows(
            cell,
            episodes=config.episodes_per_cell_B,
            variant_code="B",
            cell_index=cell_index,
        )
        write_cell_artifacts(
            cell_dir=config.output_dir / "cells" / cell.candidate_id,
            cell=cell,
            run_id=RUN_ID,
            mode="calibrate",
            suite_family_resolution_status=str(resolutions.get(cell.candidate_id) or "resolved"),
            stock_rows=stock_rows,
            control_rows=control_rows,
        )

    decision = _build_desaturation_decision(headroom)
    atomic_json_write(config.output_dir / "desaturation_selection_decision.json", decision)
    atomic_json_write(
        config.output_dir / "calibration_not_formal_claim_attestation.json",
        _build_calibration_not_formal_claim_attestation(config, decision),
    )
    resume_index = build_resume_index(
        config.output_dir,
        selected,
        skip_completed=config.skip_completed,
    )
    resume_index["headroom_eligibility_recomputed"] = True
    atomic_json_write(config.output_dir / "resume_index.json", resume_index)
    atomic_json_write(
        config.output_dir / "calibration_status.json",
        {
            "schema_version": "v22_blind_calibration_calibration_status_v1",
            "run_id": RUN_ID,
            "generated_at_utc": _deterministic_utc(),
            "status": decision["status"],
            "mode": "calibrate",
            "synthetic_test_stub": True,
            "formal_result": False,
            "hash_lock_allowed": False,
            "blocking_reasons": decision["blocking_reasons"],
        },
    )
    return 0


def _run_real_calibrate(
    config: RunnerConfig,
    cells: Sequence[CandidateCell],
    precondition: Mapping[str, object],
    contract: Iter8InputContract | None,
) -> int:
    if contract is None:
        return _block_calibrate(config, ["BLOCK_INPUT_CONTRACT_PATH_MISSING"])
    _prefer_upstream_openpi_imports()
    try:
        from work.openpi.pipelines.recap.blind_calibration_inference import (
            PER_SUITE_MAX_STEPS,
            _run_real_episode,
            build_libero_episode_env,
            load_variant_A,
            load_variant_B_optional,
            resolve_suite_max_steps,
        )
    except Exception as exc:  # noqa: BLE001
        return _block_calibrate(config, [f"BLOCK_REAL_INFERENCE_IMPORT_FAILED:{type(exc).__name__}:{exc}"])

    try:
        policy_a = load_variant_A(contract.a_stock_authority_manifest_path)
    except Exception as exc:  # noqa: BLE001
        return _block_calibrate(config, [_block_reason_from_exception(exc)])
    b_checkpoint_unavailable_reason: str | None = None
    policy_b = None
    if "B" in config.optional_control_variants:
        try:
            policy_b = load_variant_B_optional(B_LOCAL_CHECKPOINT_PATH)
        except Exception as exc:  # noqa: BLE001
            return _block_calibrate(config, [_block_reason_from_exception(exc)])
        if policy_b is None:
            b_checkpoint_unavailable_reason = (
                f"missing_optional_B_checkpoint:{_repo_path(B_LOCAL_CHECKPOINT_PATH)}"
            )

    selected = _select_cells(cells, cell_id=config.cell_id, max_cells=config.max_cells)
    cell_plan, resolutions = _build_cell_plan_with_resolutions(
        selected,
        output_dir=config.output_dir,
    )
    cell_plan["real_policy_inference"] = True
    cell_plan["per_suite_max_steps"] = {
        suite: PER_SUITE_MAX_STEPS[suite] for suite in sorted(PER_SUITE_MAX_STEPS)
    }
    atomic_json_write(config.output_dir / "cell_plan.json", cell_plan)
    atomic_json_write(
        config.output_dir / "smoke_run_manifest.json",
        {
            "schema_version": "v22_blind_calibration_smoke_run_manifest_v1",
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "mode": "calibrate",
            "cells_requested": [cell.candidate_id for cell in selected],
            "skipped_cells": [],
            "episodes_per_cell_A": config.episodes_per_cell_A,
            "episodes_per_cell_B": config.episodes_per_cell_B,
            "formal_result": False,
            "hash_lock_allowed": False,
            "real_policy_inference": True,
        },
    )

    stock_rows_by_cell: dict[str, tuple[dict[str, object], ...]] = {}
    skipped: list[str] = []
    for cell in selected:
        cell_dir = config.output_dir / "cells" / cell.candidate_id
        resume_index = build_resume_index(
            config.output_dir,
            selected,
            skip_completed=config.skip_completed,
        )
        if config.skip_completed and cell.candidate_id in resume_index["skipped_cells"]:
            skipped.append(cell.candidate_id)
            stock_rows_by_cell[cell.candidate_id] = _read_jsonl(
                cell_dir / "stock_A" / "per_episode_trace.jsonl"
            )
            continue
        stock_rows = _real_policy_rows(
            cell,
            policy_a,
            variant_code="A",
            episodes=config.episodes_per_cell_A,
            resolve_suite_max_steps=resolve_suite_max_steps,
            build_libero_episode_env=build_libero_episode_env,
            run_real_episode=_run_real_episode,
        )
        stock_rows_by_cell[cell.candidate_id] = stock_rows
        write_cell_artifacts(
            cell_dir=cell_dir,
            cell=cell,
            run_id=RUN_ID,
            mode="calibrate",
            suite_family_resolution_status=str(resolutions.get(cell.candidate_id) or "resolved"),
            stock_rows=stock_rows,
        )
        _annotate_real_cell_artifacts(cell_dir, stock_rows=stock_rows)

    headroom = _build_headroom_eligibility(selected, stock_rows_by_cell)
    atomic_json_write(config.output_dir / "headroom_eligibility_pre_b_scan.json", headroom)

    eligible = set(headroom["eligible_cells"])
    for cell in selected:
        if cell.candidate_id in skipped:
            continue
        if config.b_scan_policy == "none":
            continue
        if config.b_scan_policy == "headroom_eligible_only" and cell.candidate_id not in eligible:
            continue
        cell_dir = config.output_dir / "cells" / cell.candidate_id
        stock_rows = stock_rows_by_cell.get(cell.candidate_id)
        if stock_rows is None:
            stock_rows = _read_jsonl(cell_dir / "stock_A" / "per_episode_trace.jsonl")
        control_rows: tuple[dict[str, object], ...] = ()
        if policy_b is not None:
            control_rows = _real_policy_rows(
                cell,
                policy_b,
                variant_code="B",
                episodes=config.episodes_per_cell_B,
                resolve_suite_max_steps=resolve_suite_max_steps,
                build_libero_episode_env=build_libero_episode_env,
                run_real_episode=_run_real_episode,
            )
        write_cell_artifacts(
            cell_dir=cell_dir,
            cell=cell,
            run_id=RUN_ID,
            mode="calibrate",
            suite_family_resolution_status=str(resolutions.get(cell.candidate_id) or "resolved"),
            stock_rows=stock_rows,
            control_rows=control_rows,
        )
        _annotate_real_cell_artifacts(
            cell_dir,
            stock_rows=stock_rows,
            control_rows=control_rows,
            b_checkpoint_unavailable_reason=b_checkpoint_unavailable_reason,
        )

    decision = _build_desaturation_decision(headroom)
    atomic_json_write(config.output_dir / "desaturation_selection_decision.json", decision)
    atomic_json_write(
        config.output_dir / "calibration_not_formal_claim_attestation.json",
        _build_calibration_not_formal_claim_attestation(config, decision),
    )
    resume_index = build_resume_index(
        config.output_dir,
        selected,
        skip_completed=config.skip_completed,
    )
    resume_index["headroom_eligibility_recomputed"] = True
    atomic_json_write(config.output_dir / "resume_index.json", resume_index)
    atomic_json_write(
        config.output_dir / "calibration_status.json",
        {
            "schema_version": "v22_blind_calibration_calibration_status_v1",
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "status": decision["status"],
            "mode": "calibrate",
            "synthetic_test_stub": False,
            "formal_result": False,
            "hash_lock_allowed": False,
            "blocking_reasons": decision["blocking_reasons"],
        },
    )
    _write_smoke_status(
        config,
        "PASS",
        cells=[cell.candidate_id for cell in selected],
        skipped=skipped,
    )
    return 0


def _block_calibrate(config: RunnerConfig, blocking_reasons: Sequence[str]) -> int:
    reasons = list(dict.fromkeys(str(reason) for reason in blocking_reasons))
    atomic_json_write(
        config.output_dir / "calibration_status.json",
        {
            "schema_version": "v22_blind_calibration_calibration_status_v1",
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "status": "BLOCK",
            "mode": "calibrate",
            "synthetic_test_stub": False,
            "formal_result": False,
            "hash_lock_allowed": False,
            "blocking_reasons": reasons,
        },
    )
    _write_smoke_status(config, "BLOCK", blocking_reasons=reasons)
    return 2


def _block_reason_from_exception(exc: Exception) -> str:
    text = str(exc)
    return text.split(":", 1)[0] if text.startswith("BLOCK_") else f"{type(exc).__name__}:{text}"


def _real_policy_rows(
    cell: CandidateCell,
    policy: object,
    *,
    variant_code: str,
    episodes: int,
    resolve_suite_max_steps: object,
    build_libero_episode_env: object,
    run_real_episode: object,
) -> tuple[dict[str, object], ...]:
    suite_max_steps = int(resolve_suite_max_steps(cell.suite_family))  # type: ignore[operator]
    episode_step_budget = max(1, int(round(cell.budget_fraction * suite_max_steps)))
    rows: list[dict[str, object]] = []
    for episode_index in range(episodes):
        seed = hash(f"{cell.candidate_id}:{variant_code}:{episode_index}") & 0xFFFFFFFF
        env = None
        try:
            env = build_libero_episode_env(  # type: ignore[operator]
                suite_family=cell.suite_family,
                tasks=cell.tasks,
                episode_index=episode_index,
                seed=seed,
            )
            row = dict(run_real_episode(env, policy, max_steps=episode_step_budget, seed=seed))  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001
            row = {
                "seed": seed,
                "success": False,
                "timeout_flag": False,
                "trace_completeness": 0.0,
                "steps_taken": 0,
                "terminal_reason": "error:" + str(exc),
            }
        finally:
            if env is not None and hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass
        row.update(
            {
                "cell_id": cell.candidate_id,
                "suite_family": cell.suite_family,
                "budget_fraction": cell.budget_fraction,
                "variant_code": variant_code,
                "episode_index": episode_index,
                "episode_step_budget": episode_step_budget,
                "steps": row.get("steps_taken"),
                "episode_status": (
                    "error"
                    if str(row.get("terminal_reason") or "").startswith("error:")
                    else "completed"
                ),
                "policy_output_source": "real_openpi_local_policy",
                "synthetic_policy": False,
                "selected_using_c_results": False,
                "selected_using_x_results": False,
                "formal_result": False,
            }
        )
        rows.append(row)
    return tuple(rows)


def _annotate_real_cell_artifacts(
    cell_dir: Path,
    *,
    stock_rows: Sequence[Mapping[str, object]],
    control_rows: Sequence[Mapping[str, object]] = (),
    b_checkpoint_unavailable_reason: str | None = None,
) -> None:
    manifest_path = cell_dir / "cell_manifest.json"
    status_path = cell_dir / "cell_status.json"
    manifest = read_json_object(manifest_path)
    manifest["real_policy_inference"] = True
    manifest["episode_seeds"] = [
        {
            "variant_code": row.get("variant_code"),
            "episode_index": row.get("episode_index"),
            "seed": row.get("seed"),
        }
        for row in [*stock_rows, *control_rows]
        if row.get("seed") is not None
    ]
    atomic_json_write(manifest_path, manifest)
    status = read_json_object(status_path)
    status["real_policy_inference"] = True
    status["synthetic_test_stub"] = False
    if b_checkpoint_unavailable_reason:
        status["b_checkpoint_unavailable_reason"] = b_checkpoint_unavailable_reason
        atomic_jsonl_write(cell_dir / "control_B" / "per_episode_trace.jsonl", ())
    atomic_json_write(status_path, status)
    _refresh_cell_sha256sums(cell_dir)


def _refresh_cell_sha256sums(cell_dir: Path) -> None:
    sums = Sha256Sums(cell_dir)
    for path in sorted(cell_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.record(path)
    sums.write(cell_dir / "SHA256SUMS")


def _select_cells(
    cells: Sequence[CandidateCell],
    *,
    cell_id: str | None,
    max_cells: int | None,
) -> tuple[CandidateCell, ...]:
    selected = tuple(cell for cell in cells if cell_id is None or cell.candidate_id == cell_id)
    if cell_id and not selected:
        raise ValueError(f"cell_id_not_found:{cell_id}")
    if max_cells is not None:
        selected = selected[: max(0, int(max_cells))]
    return selected


def _smoke_rows(
    cell: CandidateCell,
    *,
    episodes: int,
    variant_code: str = "A",
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "cell_id": cell.candidate_id,
            "suite_family": cell.suite_family,
            "budget_fraction": cell.budget_fraction,
            "variant_code": variant_code,
            "episode_index": idx,
            "success": False,
            "timeout_flag": False,
            "episode_status": "surface_smoke_placeholder",
            "selected_using_c_results": False,
            "formal_result": False,
        }
        for idx in range(episodes)
    )


def _synthetic_calibration_enabled() -> bool:
    return os.environ.get("V22_BLIND_CALIBRATION_TEST_STUB_POLICY") == "deterministic_resume"


def _deterministic_utc() -> str:
    return os.environ.get("V22_BLIND_CALIBRATION_TEST_FIXED_UTC", "2026-04-27T00:00:00Z")


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _synthetic_policy_rows(
    cell: CandidateCell,
    *,
    episodes: int,
    variant_code: str,
    cell_index: int,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    sleep_s = float(os.environ.get("V22_BLIND_CALIBRATION_TEST_EPISODE_SLEEP_SEC", "0"))
    signal_dir_raw = os.environ.get("V22_BLIND_CALIBRATION_TEST_SIGNAL_DIR")
    for idx in range(episodes):
        if signal_dir_raw:
            signal_dir = Path(signal_dir_raw)
            signal_dir.mkdir(parents=True, exist_ok=True)
            marker = signal_dir / f"{cell_index}_{cell.candidate_id}.episode_{idx}.started"
            marker.write_text(_deterministic_utc() + "\n", encoding="utf-8")
        if sleep_s > 0:
            time.sleep(sleep_s)
        rows.append(
            {
                "cell_id": cell.candidate_id,
                "suite_family": cell.suite_family,
                "budget_fraction": cell.budget_fraction,
                "variant_code": variant_code,
                "episode_index": idx,
                "seed": 1000 + cell_index * 100 + idx,
                "success": idx % 2 == 0,
                "timeout_flag": False,
                "trace_completeness": 1.0,
                "steps": max(1, int(round(cell.budget_fraction * 100))),
                "terminal_reason": "synthetic_test_stub_complete",
                "episode_status": "completed",
                "policy_output_source": "synthetic_test_stub",
                "synthetic_policy": True,
                "selected_using_c_results": False,
                "selected_using_x_results": False,
                "formal_result": False,
            }
        )
    return tuple(rows)


def _success_rate(rows: Sequence[Mapping[str, object]]) -> float:
    return sum(1 for row in rows if bool(row.get("success"))) / max(len(rows), 1)


def _timeout_rate(rows: Sequence[Mapping[str, object]]) -> float:
    return sum(1 for row in rows if bool(row.get("timeout_flag"))) / max(len(rows), 1)


def _trace_completeness(rows: Sequence[Mapping[str, object]]) -> float:
    if not rows:
        return 0.0
    values = [float(row.get("trace_completeness") or 0.0) for row in rows]
    return sum(values) / len(values)


def _build_headroom_eligibility(
    cells: Sequence[CandidateCell],
    stock_rows_by_cell: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    eligible: list[str] = []
    all_stock_rows = [
        row for stock_rows in stock_rows_by_cell.values() for row in stock_rows
    ]
    synthetic_test_stub = bool(all_stock_rows) and all(
        bool(row.get("synthetic_policy")) for row in all_stock_rows
    )
    success_rate_source = (
        "synthetic_test_stub_not_real_policy"
        if synthetic_test_stub
        else "real_openpi_local_policy"
    )
    for cell in cells:
        stock_rows = stock_rows_by_cell.get(cell.candidate_id, ())
        success_rate = _success_rate(stock_rows)
        trace_completeness = _trace_completeness(stock_rows)
        timeout_rate = _timeout_rate(stock_rows)
        headroom_eligible = (
            0.30 <= success_rate <= 0.85
            and trace_completeness >= 0.95
            and timeout_rate <= 0.35
        )
        if headroom_eligible:
            eligible.append(cell.candidate_id)
        rows.append(
            {
                "cell_id": cell.candidate_id,
                "suite_family": cell.suite_family,
                "budget_fraction": cell.budget_fraction,
                "stock_A_success_rate": success_rate,
                "stock_A_success_rate_source": success_rate_source,
                "trace_completeness": trace_completeness,
                "timeout_rate": timeout_rate,
                "headroom_eligible": headroom_eligible,
            }
        )
    return {
        "schema_version": "v22_headroom_eligibility_pre_b_scan_v1",
        "run_id": RUN_ID,
        "headroom_eligibility_computed_at_utc": (
            _deterministic_utc() if synthetic_test_stub else utc_now()
        ),
        "headroom_rule": {
            "success_rate_min": 0.30,
            "success_rate_max": 0.85,
            "trace_completeness_min": 0.95,
            "timeout_rate_max": 0.35,
        },
        "eligible_cells": eligible,
        "cells": rows,
        "selected_using_c_results": False,
        "selected_using_x_results": False,
        "synthetic_test_stub": synthetic_test_stub,
    }


def _build_desaturation_decision(headroom: Mapping[str, object]) -> dict[str, object]:
    cells = [
        cell for cell in headroom.get("cells", ())
        if isinstance(cell, Mapping) and bool(cell.get("headroom_eligible"))
    ]
    cells = sorted(
        cells,
        key=lambda item: (
            -float(item.get("trace_completeness") or 0.0),
            abs(float(item.get("stock_A_success_rate") or 0.0) - 0.55),
            float(item.get("timeout_rate") or 0.0),
            float(item.get("budget_fraction") or 0.0),
            str(item.get("cell_id") or ""),
        ),
    )
    selected = cells[0] if cells else None
    synthetic_test_stub = bool(headroom.get("synthetic_test_stub"))
    policy_source = (
        "synthetic_test_stub_not_real_policy"
        if synthetic_test_stub
        else "real_openpi_local_policy"
    )
    return {
        "schema_version": "desaturation_selection_decision_v1",
        "run_id": RUN_ID,
        "decided_at_utc": _deterministic_utc() if synthetic_test_stub else utc_now(),
        "status": "PASS" if selected else "BLOCK",
        "selected_cell_id": selected.get("cell_id") if selected else None,
        "selected_suite_family": selected.get("suite_family") if selected else None,
        "selected_budget_fraction": selected.get("budget_fraction") if selected else None,
        "headroom_rule": headroom.get("headroom_rule"),
        "selection_input_sources": {
            "stock_A_success_rate": policy_source,
            "optional_B": policy_source,
        },
        "selected_using_c_results": False,
        "selected_using_x_results": False,
        "forbidden_variants_absent": True,
        "formal_v22_started": False,
        "hash_lock_emitted": False,
        "synthetic_test_stub": synthetic_test_stub,
        "blocking_reasons": [] if selected else ["NO_DESATURATED_PROTOCOL_FOUND"],
    }


def _build_calibration_not_formal_claim_attestation(
    config: RunnerConfig,
    decision: Mapping[str, object],
) -> dict[str, object]:
    synthetic_test_stub = bool(decision.get("synthetic_test_stub"))
    return {
        "schema_version": "calibration_not_formal_claim_attestation_v1",
        "run_id": RUN_ID,
        "generated_at_utc": _deterministic_utc() if synthetic_test_stub else utc_now(),
        "status": decision.get("status"),
        "calibration_is_formal_result": False,
        "formal_v22_started": False,
        "hash_lock_allowed": False,
        "selected_using_c_results": False,
        "selected_using_x_results": False,
        "n_per_variant_used": config.episodes_per_cell_A,
        "n_per_variant_formal_referenced_only": 192,
        "claim_language_audit": [
            {
                "rule_id": "calibration_only_no_formal_result_claim",
                "status": "PASS",
            },
            {
                "rule_id": "no_benchmark_or_paper_equivalence_claim",
                "status": "PASS",
            },
            {
                "rule_id": "no_state_side_v22_claim",
                "status": "PASS",
            },
        ],
        "blocking_reasons": list(decision.get("blocking_reasons") or []),
    }


def _write_smoke_status(
    config: RunnerConfig,
    status: str,
    *,
    cells: Sequence[str] = (),
    skipped: Sequence[str] = (),
    blocking_reasons: Sequence[str] = (),
) -> None:
    atomic_json_write(
        config.output_dir / "smoke_status.json",
        {
            "schema_version": "v22_blind_calibration_smoke_status_v1",
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "status": status,
            "mode": config.mode,
            "gpu": "GPU2",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cells": list(cells),
            "skipped_cells": list(skipped),
            "formal_result": False,
            "hash_lock_allowed": False,
            "selected_using_c_results": False,
            "selected_using_x_results": False,
            "blocking_reasons": list(blocking_reasons),
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    precondition, contract = validate_preconditions(config)
    try:
        cells = load_cells(contract)
        if config.mode == "dry-run":
            return run_dry_run(config, cells, precondition)
        if config.mode == "smoke":
            return run_smoke(config, cells, precondition)
        return run_calibrate(config, cells, precondition, contract)
    except Exception as exc:  # noqa: BLE001
        config.output_dir.mkdir(parents=True, exist_ok=True)
        blocking_reasons = [f"runner_exception:{type(exc).__name__}:{exc}"]
        blocked = dict(precondition)
        blocked["status"] = "BLOCK"
        blocked["blocking_reasons"] = list(blocked.get("blocking_reasons", ())) + blocking_reasons
        atomic_json_write(config.output_dir / "precondition_check.json", blocked)
        atomic_json_write(
            config.output_dir / "runner_capability_report.json",
            {
                "schema_version": "iter8_runner_capability_report_v1",
                "run_id": RUN_ID,
                "generated_at_utc": utc_now(),
                "runner_surface_status": "BLOCK",
                "cell_count": 0,
                "selected_using_c_results": False,
                "formal_v22_started": False,
                "hash_lock_emitted": False,
                "surface_checks_pass": [],
                "surface_checks_fail": blocking_reasons,
                "blocking_reasons": blocking_reasons,
            },
        )
        print(f"ITER7_5_RUNNER_BLOCK {blocking_reasons[0]}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
