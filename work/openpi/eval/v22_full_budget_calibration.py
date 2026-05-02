from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
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
RUN_ID = "stage1_v22_blind_calibration_iter8_20260426T_nextZ"
ART = REPO_ROOT / "agent/artifacts" / RUN_ID
OUT = ART / "openpi/v22_blind_calibration_full_budget"
VERIFIER_DIR = ART / "verifier"
A_AUTHORITY_MANIFEST = ART / "coordinator/a_stock_authority_manifest_iter8.json"
TASKS = tuple(f"task_{idx}" for idx in range(10))
SUITES = (
    ("libero_spatial", 220),
    ("libero_object", 280),
    ("libero_goal", 300),
    ("libero_10", 520),
)
EPISODES_PER_SUITE = 12
GPU2_MEMORY_THRESHOLD_MIB = 500
DESIGN_NOTE = (
    "FULL max_steps per suite; 1 cell per suite; "
    "budget grid derived post-hoc from steps_taken"
)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from work.openpi.pipelines.recap.blind_calibration_runtime import (  # noqa: E402
    Sha256Sums,
    atomic_json_write,
    atomic_jsonl_write,
    repo_rel,
    sha256_file,
    utc_now,
)
from work.openpi.pipelines.recap.blind_calibration_inference import (  # noqa: E402
    PER_SUITE_MAX_STEPS,
    _run_real_episode,
    build_libero_episode_env,
    load_variant_A,
)


def main() -> int:
    started = time.monotonic()
    _ensure_python_hash_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)

    precondition = _build_precondition_check()
    atomic_json_write(OUT / "precondition_check.json", precondition)
    manifest = _build_run_manifest()
    atomic_json_write(OUT / "full_budget_run_manifest.json", manifest)

    blocking_reasons = list(precondition.get("blocking_reasons", ()))
    all_rows: dict[str, tuple[dict[str, object], ...]] = {}
    if blocking_reasons:
        _write_terminal_artifacts(
            status="BLOCK",
            started=started,
            rows_by_suite=all_rows,
            blocking_reasons=blocking_reasons,
        )
        print(f"W4B_BLOCK precondition {blocking_reasons}", flush=True)
        return 2

    try:
        policy = load_variant_A(A_AUTHORITY_MANIFEST)
    except Exception as exc:  # noqa: BLE001
        blocking_reasons.append(_block_reason_from_exception(exc))
        _write_terminal_artifacts(
            status="BLOCK",
            started=started,
            rows_by_suite=all_rows,
            blocking_reasons=blocking_reasons,
        )
        print(f"W4B_BLOCK policy_load {blocking_reasons[-1]}", flush=True)
        return 2

    for suite, max_steps in SUITES:
        print(
            f"W4B_SUITE_START suite={suite} max_steps={max_steps} episodes={EPISODES_PER_SUITE}",
            flush=True,
        )
        rows = _run_suite(suite=suite, max_steps=max_steps, policy=policy)
        all_rows[suite] = tuple(rows)
        summary = _write_suite_artifacts(suite=suite, max_steps=max_steps, rows=rows)
        print(
            "W4B_SUITE_DONE "
            f"suite={suite} succ={summary['success_count']}/{summary['episode_count']} "
            f"timeouts={summary['timeout_count']} errors={summary['error_count']}",
            flush=True,
        )

    blocking_reasons.extend(_episode_blocking_reasons(all_rows))
    status = "PASS" if not blocking_reasons and _total_rows(all_rows) == 48 else "BLOCK"
    if status == "BLOCK" and _total_rows(all_rows) != 48:
        blocking_reasons.append("EPISODE_COUNT_MISMATCH")
    blocking_reasons = list(dict.fromkeys(blocking_reasons))
    _write_terminal_artifacts(
        status=status,
        started=started,
        rows_by_suite=all_rows,
        blocking_reasons=blocking_reasons,
    )
    print(f"W4B_STATUS {status} episodes={_total_rows(all_rows)}", flush=True)
    return 0 if status == "PASS" else 2


def _ensure_python_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") == "0":
        return
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "2"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


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


def _build_precondition_check() -> dict[str, object]:
    checked_at = utc_now()
    gpu2 = _probe_gpu2()
    openpi_probe = _probe_openpi_importable()
    max_step_mismatches = [
        {
            "suite": suite,
            "expected": expected,
            "observed": PER_SUITE_MAX_STEPS.get(suite),
        }
        for suite, expected in SUITES
        if int(PER_SUITE_MAX_STEPS.get(suite, -1)) != expected
    ]
    blocking_reasons: list[str] = []
    if gpu2.get("memory_used_mib") is None:
        blocking_reasons.append("BLOCK_GPU2_OCCUPANCY_PROBE_FAILED")
    elif int(gpu2["memory_used_mib"]) > GPU2_MEMORY_THRESHOLD_MIB:
        blocking_reasons.append("BLOCK_GPU2_MEMORY_NOT_IDLE")
    if openpi_probe["status"] != "PASS":
        blocking_reasons.append("BLOCK_OPENPI_NOT_IMPORTABLE")
    if not A_AUTHORITY_MANIFEST.is_file():
        blocking_reasons.append("BLOCK_A_STOCK_AUTHORITY_MISSING")
    if max_step_mismatches:
        blocking_reasons.append("BLOCK_PER_SUITE_MAX_STEPS_MISMATCH")
    return {
        "schema_version": "w4b_precondition_check_v1",
        "run_id": RUN_ID,
        "checked_at_utc": checked_at,
        "status": "BLOCK" if blocking_reasons else "PASS",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": "GPU2",
        "gpu2_occupancy": gpu2,
        "gpu2_memory_threshold_mib": GPU2_MEMORY_THRESHOLD_MIB,
        "openpi_importable": openpi_probe,
        "a_stock_authority_manifest": {
            "path": _repo_path(A_AUTHORITY_MANIFEST),
            "exists": A_AUTHORITY_MANIFEST.is_file(),
            "sha256": sha256_file(A_AUTHORITY_MANIFEST)
            if A_AUTHORITY_MANIFEST.is_file()
            else None,
        },
        "per_suite_max_steps": dict(SUITES),
        "per_suite_max_steps_mismatches": max_step_mismatches,
        "formal_result": False,
        "selected_using_c_results": False,
        "selected_using_x_results": False,
        "blocking_reasons": blocking_reasons,
    }


def _probe_gpu2() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "-i",
        "2",
        "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "BLOCK",
            "probe_command": " ".join(command),
            "memory_used_mib": None,
            "probe_error": f"{type(exc).__name__}:{exc}",
            "compute_apps": [],
        }
    line = next((item for item in completed.stdout.splitlines() if item.strip()), "")
    columns = [part.strip() for part in line.split(",")]
    if len(columns) < 5:
        return {
            "status": "BLOCK",
            "probe_command": " ".join(command),
            "memory_used_mib": None,
            "probe_error": f"unexpected_nvidia_smi_output:{completed.stdout.strip()}",
            "compute_apps": [],
        }
    compute_apps = _probe_gpu2_compute_apps()
    used_mib = int(columns[3])
    return {
        "status": "PASS" if used_mib <= GPU2_MEMORY_THRESHOLD_MIB else "BLOCK",
        "probe_command": " ".join(command),
        "index": int(columns[0]),
        "uuid": columns[1],
        "name": columns[2],
        "memory_used_mib": used_mib,
        "memory_total_mib": int(columns[4]),
        "compute_apps": compute_apps,
    }


def _probe_gpu2_compute_apps() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "-i",
        "2",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except Exception:
        return []
    apps: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        columns = [part.strip() for part in line.split(",")]
        if len(columns) < 3:
            continue
        apps.append(
            {
                "pid": int(columns[0]),
                "process_name": columns[1],
                "used_memory_mib": int(columns[2]),
            }
        )
    return apps


def _probe_openpi_importable() -> dict[str, object]:
    _prefer_upstream_openpi_imports()
    modules = ("openpi", "openpi.policies.policy_config")
    imported: list[dict[str, object]] = []
    failures: list[str] = []
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            imported.append(
                {
                    "module": module_name,
                    "file": str(getattr(module, "__file__", "") or ""),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{module_name}:{type(exc).__name__}:{exc}")
    return {
        "status": "PASS" if not failures else "BLOCK",
        "modules": imported,
        "failures": failures,
    }


def _build_run_manifest() -> dict[str, object]:
    return {
        "schema_version": "v22_blind_calibration_full_budget_run_manifest_v1",
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "design_note": DESIGN_NOTE,
        "output_dir": _repo_path(OUT),
        "variant_code": "A",
        "variant": "pi0_libero",
        "policy_output_source": "real_openpi_local_policy",
        "synthetic_policy": False,
        "formal_result": False,
        "selected_using_c_results": False,
        "selected_using_x_results": False,
        "seed_formula": 'hash(f"w4b:{suite}:{episode_index}") & 0xFFFFFFFF',
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
        "tasks": list(TASKS),
        "suites": [
            {
                "suite": suite,
                "cell_id": _cell_id(suite),
                "max_steps": max_steps,
                "n_episodes": EPISODES_PER_SUITE,
                "n_tasks_used": len(TASKS),
                "tasks": list(TASKS),
                "episode_plan": [
                    {
                        "episode_index": idx,
                        "task_id": _task_id_for_episode(idx),
                        "task": TASKS[_task_id_for_episode(idx)],
                        "seed": _seed_for(suite, idx),
                        "episode_step_budget": max_steps,
                    }
                    for idx in range(EPISODES_PER_SUITE)
                ],
            }
            for suite, max_steps in SUITES
        ],
    }


def _run_suite(*, suite: str, max_steps: int, policy: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    trace_path = OUT / "cells" / suite / "stock_A" / "per_episode_trace.jsonl"
    for episode_index in range(EPISODES_PER_SUITE):
        seed = _seed_for(suite, episode_index)
        task_id = _task_id_for_episode(episode_index)
        env = None
        task_description = ""
        print(
            "W4B_EPISODE_START "
            f"suite={suite} episode_index={episode_index} task_id={task_id} seed={seed}",
            flush=True,
        )
        try:
            env = build_libero_episode_env(
                suite_family=suite,
                tasks=TASKS,
                episode_index=episode_index,
                seed=seed,
            )
            task_description = str(getattr(env, "task_description", ""))
            result = dict(_run_real_episode(env, policy, max_steps=max_steps, seed=seed))
        except Exception as exc:  # noqa: BLE001
            result = {
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
        row = _episode_row(
            suite=suite,
            max_steps=max_steps,
            episode_index=episode_index,
            task_id=task_id,
            task_description=task_description,
            result=result,
        )
        rows.append(row)
        atomic_jsonl_write(trace_path, rows)
        print(
            "W4B_EPISODE_DONE "
            f"suite={suite} episode_index={episode_index} "
            f"success={row['success']} terminal_reason={row['terminal_reason']} "
            f"steps_taken={row['steps_taken']}",
            flush=True,
        )
    return rows


def _episode_row(
    *,
    suite: str,
    max_steps: int,
    episode_index: int,
    task_id: int,
    task_description: str,
    result: Mapping[str, object],
) -> dict[str, object]:
    terminal_reason = str(result.get("terminal_reason") or "error:missing_terminal_reason")
    steps_taken = int(result.get("steps_taken") or 0)
    return {
        "seed": int(result.get("seed") or _seed_for(suite, episode_index)),
        "success": bool(result.get("success")),
        "timeout_flag": bool(result.get("timeout_flag")),
        "trace_completeness": float(result.get("trace_completeness") or 0.0),
        "steps_taken": steps_taken,
        "terminal_reason": terminal_reason,
        "cell_id": _cell_id(suite),
        "suite_family": suite,
        "budget_fraction": 1.0,
        "variant_code": "A",
        "episode_index": episode_index,
        "episode_step_budget": max_steps,
        "task_id": task_id,
        "task_description": task_description,
        "policy_output_source": "real_openpi_local_policy",
        "synthetic_policy": False,
        "selected_using_c_results": False,
        "selected_using_x_results": False,
        "formal_result": False,
        "steps": steps_taken,
        "episode_status": "error" if terminal_reason.startswith("error:") else "completed",
    }


def _write_suite_artifacts(
    *,
    suite: str,
    max_steps: int,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    cell_dir = OUT / "cells" / suite
    stock_dir = cell_dir / "stock_A"
    stock_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl_write(stock_dir / "per_episode_trace.jsonl", rows)
    summary = _suite_summary(suite=suite, max_steps=max_steps, rows=rows)
    atomic_json_write(stock_dir / "summary.json", summary)
    atomic_json_write(
        cell_dir / "cell_status.json",
        {
            "schema_version": "v22_blind_calibration_full_budget_cell_status_v1",
            "run_id": RUN_ID,
            "generated_at_utc": utc_now(),
            "status": "PASS" if summary["error_count"] == 0 and len(rows) == 12 else "BLOCK",
            "cell_id": _cell_id(suite),
            "suite_family": suite,
            "budget_fraction": 1.0,
            "max_steps": max_steps,
            "n_episodes": len(rows),
            "n_tasks_used": len(TASKS),
            "tasks": list(TASKS),
            "stock_A_success_rate": summary["success_rate"],
            "timeout_rate": summary["timeout_rate"],
            "trace_completeness": summary["trace_completeness"],
            "real_policy_inference": True,
            "synthetic_test_stub": False,
            "variants_run_for_selection": ["A"],
            "forbidden_variants_absent": ["C", "X"],
            "selected_using_c_results": False,
            "selected_using_x_results": False,
            "formal_result": False,
            "blocking_reasons": (
                ["EPISODE_ERRORS_PRESENT"] if summary["error_count"] else []
            ),
        },
    )
    _refresh_cell_sha256sums(cell_dir)
    return summary


def _suite_summary(
    *,
    suite: str,
    max_steps: int,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    success_count = sum(1 for row in rows if bool(row.get("success")))
    timeout_count = sum(1 for row in rows if bool(row.get("timeout_flag")))
    error_count = sum(
        1 for row in rows if str(row.get("terminal_reason") or "").startswith("error:")
    )
    success_steps = [int(row.get("steps_taken") or 0) for row in rows if row.get("success")]
    trace_values = [float(row.get("trace_completeness") or 0.0) for row in rows]
    return {
        "schema_version": "v22_blind_calibration_full_budget_variant_summary_v1",
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "cell_id": _cell_id(suite),
        "suite_family": suite,
        "variant_code": "A",
        "budget_fraction": 1.0,
        "max_steps": max_steps,
        "episode_count": len(rows),
        "success_count": success_count,
        "success_rate": success_count / max(len(rows), 1),
        "timeout_count": timeout_count,
        "timeout_rate": timeout_count / max(len(rows), 1),
        "error_count": error_count,
        "trace_completeness": sum(trace_values) / max(len(trace_values), 1),
        "mean_steps_taken": (
            sum(int(row.get("steps_taken") or 0) for row in rows) / max(len(rows), 1)
        ),
        "mean_succ_steps": sum(success_steps) / len(success_steps) if success_steps else 0.0,
        "policy_output_sources": ["real_openpi_local_policy"] if rows else [],
        "synthetic_policy": False,
        "selected_using_c_results": False,
        "selected_using_x_results": False,
        "formal_result": False,
    }


def _write_terminal_artifacts(
    *,
    status: str,
    started: float,
    rows_by_suite: Mapping[str, Sequence[Mapping[str, object]]],
    blocking_reasons: Sequence[str],
) -> None:
    elapsed = int(round(time.monotonic() - started))
    per_suite = []
    for suite, max_steps in SUITES:
        rows = tuple(rows_by_suite.get(suite, ()))
        summary = _suite_summary(suite=suite, max_steps=max_steps, rows=rows)
        per_suite.append(
            {
                "suite": suite,
                "max_steps": max_steps,
                "succ": summary["success_count"],
                "n": len(rows),
                "rate": summary["success_rate"],
                "mean_succ_steps": summary["mean_succ_steps"],
            }
        )
    n_rows = _total_rows(rows_by_suite)
    n_success = sum(
        1 for rows in rows_by_suite.values() for row in rows if bool(row.get("success"))
    )
    smoke_status = {
        "schema_version": "w4b_full_budget_smoke_status_v1",
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "status": status,
        "mode": "full_budget_calibration",
        "gpu": "GPU2",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "design_note": DESIGN_NOTE,
        "n_episodes_total": n_rows,
        "n_success_total": n_success,
        "per_suite": per_suite,
        "formal_result": False,
        "selected_using_c_results": False,
        "selected_using_x_results": False,
        "blocking_reasons": list(blocking_reasons),
    }
    atomic_json_write(OUT / "smoke_status.json", smoke_status)
    atomic_json_write(
        VERIFIER_DIR / "w4b_status.json",
        {
            "schema_version": "w4b_status_v1",
            "worker": "worker-4b",
            "status": status,
            "completed_at_utc": utc_now(),
            "design_note": DESIGN_NOTE,
            "wall_clock_seconds": elapsed,
            "n_episodes_total": n_rows,
            "n_success_total": n_success,
            "per_suite": per_suite,
            "blocking_reasons": list(blocking_reasons),
        },
    )


def _refresh_cell_sha256sums(cell_dir: Path) -> None:
    sums = Sha256Sums(cell_dir)
    for path in sorted(cell_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.record(path)
    sums.write(cell_dir / "SHA256SUMS")


def _episode_blocking_reasons(
    rows_by_suite: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[str]:
    reasons: list[str] = []
    for suite, _max_steps in SUITES:
        rows = tuple(rows_by_suite.get(suite, ()))
        if len(rows) != EPISODES_PER_SUITE:
            reasons.append(f"{suite}:EPISODE_COUNT_MISMATCH")
        if any(str(row.get("terminal_reason") or "").startswith("error:") for row in rows):
            reasons.append(f"{suite}:EPISODE_ERRORS_PRESENT")
    return reasons


def _block_reason_from_exception(exc: Exception) -> str:
    text = str(exc)
    return text.split(":", 1)[0] if text.startswith("BLOCK_") else f"{type(exc).__name__}:{text}"


def _total_rows(rows_by_suite: Mapping[str, Sequence[Mapping[str, object]]]) -> int:
    return sum(len(rows) for rows in rows_by_suite.values())


def _seed_for(suite: str, episode_index: int) -> int:
    return hash(f"w4b:{suite}:{episode_index}") & 0xFFFFFFFF


def _task_id_for_episode(episode_index: int) -> int:
    return episode_index % len(TASKS)


def _cell_id(suite: str) -> str:
    return f"{suite}__full_budget"


def _repo_path(path: Path) -> str:
    return repo_rel(REPO_ROOT, path)


if __name__ == "__main__":
    raise SystemExit(main())
