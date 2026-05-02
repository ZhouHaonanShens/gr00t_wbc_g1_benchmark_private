from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.contracts import RuntimeServerSpec  # noqa: E402
from work.openpi.dataloader import read_json, write_json  # noqa: E402
from work.openpi.pipelines.recap.blind_calibration_runtime import (  # noqa: E402
    legacy_full_rewrite_jsonl,
)
from work.openpi.recap.runtime_prompt import resolve_runtime_indicator_config  # noqa: E402
from work.openpi.runtime import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    LIBERO_NATIVE_SMOKE_ENTRY,
    NUM_STEPS_WAIT,
    PolicyServerProcess,
    RuntimeCleanup,
    RuntimeEpisodeClient,
    RuntimePathsBuilder,
    max_steps_for_task_suite,
    pick_free_port,
    prepare_libero_config_dir,
)
from work.openpi.serve.provenance import EXPECTED_CHECKPOINT  # noqa: E402


RUN_ID = "stage1_recap_longrun_iter5_5_contract_fix_20260425T_nextZ"
CANONICAL_RULE_REL = Path(
    f"agent/artifacts/{RUN_ID}/coordinator/canonical_blind_selection_rule.json"
)
PRECONDITION_REL = Path(
    f"agent/artifacts/{RUN_ID}/coordinator/w6_precondition_check.json"
)
DEFAULT_OUTPUT_REL = Path(
    f"agent/artifacts/{RUN_ID}/openpi/v22_blind_calibration"
)
DEFAULT_RUNTIME_REL = Path("agent/runtime_logs/iter5p5_w3p5_blind_calibration")
TARGET_STATUS = "DESATURATED_" + "FOUND"
TERMINAL_STATUS_NO_HEADROOM = "NO_HEADROOM_IN_A"
TERMINAL_STATUS_INSUFFICIENT = "INSUFFICIENT_BUDGET_GRID"
STATUS_IN_FLIGHT = "IN_FLIGHT"
STATUS_BLOCK_PRECONDITION = "BLOCK_PRECONDITION"
STATUS_BLOCK_RUNTIME = "BLOCK_runtime_defect"
STATUS_BLOCK_LOGIC = "BLOCK_logic_defect"
Q8_RUNTIME_DEFECT_SIGNATURES = (
    "CUDA_OOM",
    "XID_DRIVER_FAULT",
    "SIGSEGV",
    "SIGBUS",
    "NCCL_TIMEOUT",
)


@dataclass(frozen=True)
class CalibrationConfig:
    repo_root: Path
    output_dir: Path
    runtime_dir: Path
    canonical_rule_path: Path
    precondition_path: Path
    max_workers: int
    overall_timeout_s: float
    client_timeout_s: float
    server_ready_timeout_s: float
    host: str
    port: int
    task_suite_name: str
    task_ids: tuple[int, ...]
    seeds: tuple[int, ...]
    trial_indices: tuple[int, ...]
    budget_fractions: tuple[float, ...]


@dataclass(frozen=True)
class Preflight:
    canonical_rule_sha256: str
    expected_sha256: str
    actual_sha256: str
    hash_match: bool
    precondition_pass: bool
    canonical_rule_path_only: bool
    fallback_paths_allowed: bool
    blocking_reasons: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeSpec:
    task_id: int
    seed: int
    trial_index: int

    @property
    def key(self) -> str:
        return f"task{self.task_id}_seed{self.seed}_trial{self.trial_index}"


@dataclass(frozen=True)
class RuntimeContext:
    port: int
    runtime_dir: Path
    server_log: Path
    harness_log: Path
    source_dir: Path


@dataclass(frozen=True)
class CalibrationDecision:
    calibration_status: str
    selected_candidate_id: str | None
    selected_budget: float | None
    selected_task_suite: str | None
    selected_task_ids: tuple[int, ...]
    blocking_reasons: tuple[str, ...]
    candidates: tuple[dict[str, object], ...]


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


def log_line(path: Path, message: str) -> None:
    print(message, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        handle.write("\n")


def coerce_int_sequence(raw: object, *, default: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return tuple(default)
    values: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            continue
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(values) or tuple(default)


def coerce_float_sequence(raw: object, *, default: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return tuple(default)
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool):
            continue
        try:
            values.append(round(float(item), 2))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(set(values))) or tuple(default)


def mapping_at(payload: Mapping[str, object], *keys: str) -> Mapping[str, object]:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="materialize_iter5p5_blind_calibration.py",
        description="Run the iter5.5 W3p5 OpenPI A-stock calibration lane.",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_REL))
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_REL))
    parser.add_argument("--canonical-rule", default=str(CANONICAL_RULE_REL))
    parser.add_argument("--precondition", default=str(PRECONDITION_REL))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--overall-timeout-s", type=float, default=10_800.0)
    parser.add_argument("--client-timeout-s", type=float, default=600.0)
    parser.add_argument("--server-ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def config_from_args(args: argparse.Namespace) -> CalibrationConfig:
    repo_root = Path(args.repo_root).resolve()
    canonical_rule_path = (repo_root / args.canonical_rule).resolve()
    precondition_path = (repo_root / args.precondition).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    runtime_dir = (repo_root / args.runtime_dir).resolve()
    canonical_rule = read_json(canonical_rule_path)
    protocol = mapping_at(canonical_rule, "rule", "calibration_protocol")
    task_suite_name = str(protocol.get("task_suite") or "libero_spatial").strip()
    if not task_suite_name or "or_other" in task_suite_name:
        task_suite_name = "libero_spatial"
    return CalibrationConfig(
        repo_root=repo_root,
        output_dir=output_dir,
        runtime_dir=runtime_dir,
        canonical_rule_path=canonical_rule_path,
        precondition_path=precondition_path,
        max_workers=max(1, int(args.max_workers)),
        overall_timeout_s=float(args.overall_timeout_s),
        client_timeout_s=float(args.client_timeout_s),
        server_ready_timeout_s=float(args.server_ready_timeout_s),
        host=str(args.host),
        port=int(args.port),
        task_suite_name=task_suite_name,
        task_ids=coerce_int_sequence(protocol.get("task_ids"), default=(0, 1)),
        seeds=coerce_int_sequence(
            protocol.get("seed_set"),
            default=(7, 17, 27, 37, 47, 57, 67, 77),
        ),
        trial_indices=coerce_int_sequence(protocol.get("trial_indices"), default=(0,)),
        budget_fractions=coerce_float_sequence(
            protocol.get("budget_grid"),
            default=(0.50,),
        ),
    )


def validate_preflight(config: CalibrationConfig) -> Preflight:
    blocking_reasons: list[str] = []
    if not config.canonical_rule_path.is_file():
        blocking_reasons.append("canonical_rule_missing")
    if not config.precondition_path.is_file():
        blocking_reasons.append("w6_precondition_check_missing")
    if blocking_reasons:
        return Preflight(
            canonical_rule_sha256="",
            expected_sha256="",
            actual_sha256="",
            hash_match=False,
            precondition_pass=False,
            canonical_rule_path_only=False,
            fallback_paths_allowed=True,
            blocking_reasons=tuple(blocking_reasons),
        )
    canonical_sha = sha256_file(config.canonical_rule_path)
    precondition = read_json(config.precondition_path)
    expected_sha = str(precondition.get("expected_sha256") or "").strip()
    actual_sha = canonical_sha
    canonical_path_only = bool(precondition.get("canonical_rule_path_only"))
    fallback_paths_allowed = bool(precondition.get("fallback_paths_allowed"))
    hash_match = bool(precondition.get("hash_match")) and expected_sha == actual_sha
    precondition_pass = bool(precondition.get("precondition_pass")) and hash_match
    if not precondition_pass:
        blocking_reasons.append("w6_precondition_check_not_pass")
    if not canonical_path_only:
        blocking_reasons.append("canonical_rule_path_only_false")
    if fallback_paths_allowed:
        blocking_reasons.append("fallback_paths_allowed_true")
    return Preflight(
        canonical_rule_sha256=canonical_sha,
        expected_sha256=expected_sha,
        actual_sha256=actual_sha,
        hash_match=hash_match,
        precondition_pass=precondition_pass,
        canonical_rule_path_only=canonical_path_only,
        fallback_paths_allowed=fallback_paths_allowed,
        blocking_reasons=tuple(blocking_reasons),
    )


def episode_specs(config: CalibrationConfig) -> tuple[EpisodeSpec, ...]:
    return tuple(
        EpisodeSpec(task_id=task_id, seed=seed, trial_index=trial_index)
        for task_id in config.task_ids
        for seed in config.seeds
        for trial_index in config.trial_indices
    )


def runtime_context(config: CalibrationConfig) -> RuntimeContext:
    runtime_dir = config.runtime_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return RuntimeContext(
        port=pick_free_port(config.host, config.port),
        runtime_dir=runtime_dir,
        server_log=runtime_dir / "server.log",
        harness_log=runtime_dir / "harness.log",
        source_dir=config.output_dir / "_runtime_source",
    )


def build_trace_row(
    row: Mapping[str, object],
    *,
    task_suite_name: str,
    max_steps_resolved: int,
) -> dict[str, object]:
    success = bool(row.get("success", False))
    steps_observed = int(row.get("steps_observed") or 0)
    executed_steps = min(max(steps_observed - NUM_STEPS_WAIT, 0), max_steps_resolved)
    first_success_step = max(1, executed_steps) if success else None
    return {
        "variant_code": "A",
        "variant": "stock_libero_ref_v1",
        "task_suite_name": task_suite_name,
        "task_id": int(row.get("task_id") or 0),
        "seed": int(row.get("seed") or 0),
        "trial_idx": int(row.get("trial_index") or row.get("trial_idx") or 0),
        "success": success,
        "first_success_step": first_success_step,
        "executed_steps": executed_steps,
        "max_steps_resolved": max_steps_resolved,
        "timeout_flag": bool((not success) and executed_steps >= max_steps_resolved),
        "episode_status": str(row.get("episode_status") or "ok"),
        "error": str(row.get("error") or ""),
        "video_path": str(row.get("video_path") or ""),
        "client_log": str(row.get("client_log") or ""),
        "runtime_prompting": {
            key: str(row.get(key) or "")
            for key in (
                "indicator_mode_requested",
                "indicator_mode",
                "indicator_source",
                "prompt_text_surface",
                "prompt_route",
                "conditioning_mode",
                "consumer_mode",
                "fixed_indicator_mode",
                "critic_checkpoint_ref",
            )
        },
    }


def run_episode(
    *,
    config: CalibrationConfig,
    context: RuntimeContext,
    spec: EpisodeSpec,
    openpi_root: Path,
    venv_python: Path,
    libero_config_dir: Path,
    max_steps_resolved: int,
) -> dict[str, object]:
    client = RuntimeEpisodeClient()
    raw_row = client.run_runtime_episode(
        task_suite_name=config.task_suite_name,
        task_id=spec.task_id,
        seed=spec.seed,
        trial_idx=spec.trial_index,
        video_path=context.source_dir / "videos" / f"{spec.key}.mp4",
        host=config.host,
        port=context.port,
        venv_python=venv_python,
        openpi_root=openpi_root,
        libero_config_dir=libero_config_dir,
        runtime_dir=context.runtime_dir,
        timeout_s=config.client_timeout_s,
        checkpoint_ref=EXPECTED_CHECKPOINT,
        indicator_mode_requested="omit",
        runtime_indicator_config=resolve_runtime_indicator_config(
            requested_indicator_mode="omit",
            variant="stock_libero_ref_v1",
            train_manifest=None,
            checkpoint_provenance=None,
        ),
        cli_entry=Path(LIBERO_NATIVE_SMOKE_ENTRY),
    )
    return build_trace_row(
        raw_row,
        task_suite_name=config.task_suite_name,
        max_steps_resolved=max_steps_resolved,
    )


def runtime_defect_signature(text: str) -> str | None:
    lowered = text.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "CUDA_OOM"
    if "xid" in lowered:
        return "XID_DRIVER_FAULT"
    if "sigsegv" in lowered or "segmentation fault" in lowered:
        return "SIGSEGV"
    if "sigbus" in lowered or "bus error" in lowered:
        return "SIGBUS"
    if "nccl" in lowered and "timeout" in lowered:
        return "NCCL_TIMEOUT"
    return None


def run_rollouts(config: CalibrationConfig, context: RuntimeContext) -> tuple[list[dict[str, object]], list[dict[str, object]], str | None]:
    paths = RuntimePathsBuilder(
        topic="iter5p5_w3p5_blind_calibration",
        artifact_root=config.output_dir,
        runtime_root=context.runtime_dir,
    ).build()
    libero_config_dir = prepare_libero_config_dir(paths.openpi_root, context.runtime_dir)
    max_steps_resolved = max_steps_for_task_suite(config.task_suite_name)
    server_spec = RuntimeServerSpec(
        host=config.host,
        port=context.port,
        checkpoint_dir=EXPECTED_CHECKPOINT,
        server_ready_timeout_s=config.server_ready_timeout_s,
        client_timeout_s=config.client_timeout_s,
    )
    server = PolicyServerProcess(
        spec=server_spec,
        venv_python=paths.openpi_venv_python,
        serve_policy=paths.serve_policy,
        openpi_root=paths.openpi_root,
        server_log=context.server_log,
        libero_config_dir=libero_config_dir,
        cli_entry=Path(LIBERO_NATIVE_SMOKE_ENTRY),
    )
    specs = episode_specs(config)
    completed_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    runtime_signature: str | None = None
    proc: subprocess.Popen[str] | None = None
    handle: Any | None = None
    deadline = time.monotonic() + config.overall_timeout_s
    try:
        proc, handle = server.start()
        probe = server.wait_until_ready(
            runtime_dir=context.runtime_dir,
            harness_log=context.harness_log,
        )
        log_line(
            context.harness_log,
            f"W3P5_SERVER_READY port={context.port} probe_at={probe.get('probed_at', '')}",
        )
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(
                    run_episode,
                    config=config,
                    context=context,
                    spec=spec,
                    openpi_root=paths.openpi_root,
                    venv_python=paths.openpi_venv_python,
                    libero_config_dir=libero_config_dir,
                    max_steps_resolved=max_steps_resolved,
                ): spec
                for spec in specs
            }
            for future in as_completed(futures):
                spec = futures[future]
                if time.monotonic() > deadline:
                    failed_rows.append(
                        {
                            "task_id": spec.task_id,
                            "seed": spec.seed,
                            "trial_idx": spec.trial_index,
                            "episode_status": STATUS_IN_FLIGHT,
                            "error": "overall_timeout_s_reached",
                        }
                    )
                    continue
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                    failed_rows.append(
                        {
                            "task_id": spec.task_id,
                            "seed": spec.seed,
                            "trial_idx": spec.trial_index,
                            "episode_status": "runtime_error",
                            "error": error,
                        }
                    )
                    runtime_signature = runtime_signature or runtime_defect_signature(
                        error,
                    )
                    log_line(context.harness_log, f"W3P5_EPISODE_FAIL {spec.key} {error}")
                else:
                    completed_rows.append(row)
                    legacy_full_rewrite_jsonl(
                        context.source_dir / "per_episode_rollouts.jsonl",
                        completed_rows,
                    )
                    log_line(context.harness_log, f"W3P5_EPISODE_DONE {spec.key}")
    finally:
        RuntimeCleanup.close_process(proc)
        RuntimeCleanup.close_handle(handle)
    return completed_rows, failed_rows, runtime_signature


def candidate_metrics(
    *,
    rows: Sequence[Mapping[str, object]],
    failed_rows: Sequence[Mapping[str, object]],
    specs: Sequence[EpisodeSpec],
    config: CalibrationConfig,
) -> tuple[dict[str, object], ...]:
    rows_by_task: dict[int, list[Mapping[str, object]]] = {task_id: [] for task_id in config.task_ids}
    for row in rows:
        rows_by_task.setdefault(int(row["task_id"]), []).append(row)
    expected_by_task = {
        task_id: sum(1 for spec in specs if spec.task_id == task_id)
        for task_id in config.task_ids
    }
    failed_by_task = {
        task_id: sum(1 for row in failed_rows if int(row.get("task_id") or -1) == task_id)
        for task_id in config.task_ids
    }
    candidates: list[dict[str, object]] = []
    for task_id in config.task_ids:
        task_rows = rows_by_task.get(task_id, [])
        expected = expected_by_task[task_id]
        failure_count = failed_by_task.get(task_id, 0)
        for budget in config.budget_fractions:
            successes = 0
            timeouts = 0
            for row in task_rows:
                first_success_step = row.get("first_success_step")
                max_steps = int(row.get("max_steps_resolved") or 0)
                budget_steps = int(max_steps * float(budget))
                successes += int(
                    first_success_step not in {None, "", "null"}
                    and int(first_success_step) <= budget_steps
                )
                timeouts += int(bool(row.get("timeout_flag")))
            observed = len(task_rows)
            denominator = max(expected, 1)
            success_rate = float(successes) / float(denominator)
            trace_completeness = float(observed) / float(denominator)
            timeout_rate = float(timeouts) / float(denominator)
            invalid_trace_rate = float(failure_count) / float(denominator)
            candidate_id = f"libero_spatial_task{task_id}__budget_{budget:.2f}".replace(
                ".",
                "_",
            )
            target_band_pass = 0.30 <= success_rate <= 0.85
            quality_pass = (
                trace_completeness >= 0.95
                and 0.02 <= timeout_rate <= 0.70
                and invalid_trace_rate <= 0.05
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "variant_code": "A",
                    "task_suite_name": config.task_suite_name,
                    "task_id": task_id,
                    "budget_fraction": budget,
                    "expected_episode_count": expected,
                    "observed_episode_count": observed,
                    "failed_episode_count": failure_count,
                    "stock_A_success_rate_at_budget": success_rate,
                    "trace_completeness": trace_completeness,
                    "timeout_rate": timeout_rate,
                    "invalid_trace_rate": invalid_trace_rate,
                    "target_band_pass": target_band_pass,
                    "quality_pass": quality_pass,
                    "selectable": target_band_pass and quality_pass,
                    "ranking_key": {
                        "distance_from_0_60": abs(success_rate - 0.60),
                        "negative_trace_completeness": -trace_completeness,
                        "timeout_distance_from_0_20": abs(timeout_rate - 0.20),
                        "budget_fraction": budget,
                        "task_id": task_id,
                    },
                }
            )
    return tuple(candidates)


def decide(candidates: Sequence[dict[str, object]], failed_rows: Sequence[Mapping[str, object]]) -> CalibrationDecision:
    selectable = [candidate for candidate in candidates if bool(candidate["selectable"])]
    if selectable:
        selected = sorted(
            selectable,
            key=lambda candidate: (
                float(candidate["ranking_key"]["distance_from_0_60"]),  # type: ignore[index]
                float(candidate["ranking_key"]["negative_trace_completeness"]),  # type: ignore[index]
                float(candidate["ranking_key"]["timeout_distance_from_0_20"]),  # type: ignore[index]
                float(candidate["ranking_key"]["budget_fraction"]),  # type: ignore[index]
                str(candidate["candidate_id"]),
            ),
        )[0]
        return CalibrationDecision(
            calibration_status=TARGET_STATUS,
            selected_candidate_id=str(selected["candidate_id"]),
            selected_budget=float(selected["budget_fraction"]),
            selected_task_suite=str(selected["task_suite_name"]),
            selected_task_ids=(int(selected["task_id"]),),
            blocking_reasons=(),
            candidates=tuple(candidates),
        )
    if failed_rows:
        return CalibrationDecision(
            calibration_status=TERMINAL_STATUS_INSUFFICIENT,
            selected_candidate_id=None,
            selected_budget=None,
            selected_task_suite=None,
            selected_task_ids=(),
            blocking_reasons=("incomplete_or_invalid_calibration_traces",),
            candidates=tuple(candidates),
        )
    return CalibrationDecision(
        calibration_status=TERMINAL_STATUS_NO_HEADROOM,
        selected_candidate_id=None,
        selected_budget=None,
        selected_task_suite=None,
        selected_task_ids=(),
        blocking_reasons=("no_stock_A_candidate_in_target_band",),
        candidates=tuple(candidates),
    )


def build_manifest(
    *,
    config: CalibrationConfig,
    preflight: Preflight,
    status: str,
    started_at: str,
    terminal_at: str | None,
    context: RuntimeContext | None,
    retry_attempted: bool,
    runtime_signature: str | None,
    blocking_reasons: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": "iter5p5_w3p5_blind_calibration_run_manifest_v1",
        "run_id": RUN_ID,
        "worker": "worker-3",
        "role": "W3p5",
        "task_id": "5",
        "started_at_utc": started_at,
        "terminal_at_utc": terminal_at,
        "calibration_status": status,
        "selected_using_c_results": False,
        "variant_codes_used": ["A"] if status not in {STATUS_BLOCK_PRECONDITION} else [],
        "variant_codes_used_subset_of_A_B_only": True,
        "formal_v22_execution_allowed": False,
        "retry_attempted": retry_attempted,
        "runtime_defect_signature": runtime_signature,
        "gpu_policy": {
            "cuda_visible_devices_required": "2",
            "cuda_visible_devices_observed": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu0_forbidden": True,
            "gpu1_forbidden": True,
            "gpu3_forbidden": True,
            "gpu_rollout_started": context is not None and status != STATUS_BLOCK_PRECONDITION,
            "max_rollout_env_workers_requested": config.max_workers,
        },
        "precondition_checks": {
            "canonical_rule_path": repo_rel(config.repo_root, config.canonical_rule_path),
            "canonical_rule_sha256": preflight.canonical_rule_sha256,
            "expected_sha256": preflight.expected_sha256,
            "actual_sha256": preflight.actual_sha256,
            "hash_match": preflight.hash_match,
            "precondition_pass": preflight.precondition_pass,
            "canonical_rule_path_only": preflight.canonical_rule_path_only,
            "fallback_paths_allowed": preflight.fallback_paths_allowed,
        },
        "runtime": {
            "runtime_dir": repo_rel(config.repo_root, context.runtime_dir) if context else None,
            "server_log": repo_rel(config.repo_root, context.server_log) if context else None,
            "harness_log": repo_rel(config.repo_root, context.harness_log) if context else None,
            "host": config.host,
            "port": context.port if context else None,
            "task_suite_name": config.task_suite_name,
            "task_ids": list(config.task_ids),
            "seeds": list(config.seeds),
            "trial_indices": list(config.trial_indices),
            "budget_fractions": list(config.budget_fractions),
            "client_timeout_s": config.client_timeout_s,
            "overall_timeout_s": config.overall_timeout_s,
        },
        "blocking_reasons": list(blocking_reasons),
    }


def write_outputs(
    *,
    config: CalibrationConfig,
    preflight: Preflight,
    status: str,
    started_at: str,
    terminal_at: str | None,
    context: RuntimeContext | None,
    rows: Sequence[Mapping[str, object]],
    failed_rows: Sequence[Mapping[str, object]],
    decision: CalibrationDecision | None,
    retry_attempted: bool,
    runtime_signature: str | None,
    blocking_reasons: Sequence[str],
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        config=config,
        preflight=preflight,
        status=status,
        started_at=started_at,
        terminal_at=terminal_at,
        context=context,
        retry_attempted=retry_attempted,
        runtime_signature=runtime_signature,
        blocking_reasons=blocking_reasons,
    )
    candidates = tuple(decision.candidates) if decision else ()
    decision_payload = {
        "schema_version": "iter5p5_w3p5_desaturation_selection_decision_v1",
        "run_id": RUN_ID,
        "updated_at_utc": terminal_at or utc_now(),
        "calibration_status": status,
        "selected_candidate_id": decision.selected_candidate_id if decision else None,
        "selected_protocol": "stock_A_budget_task_pair" if decision and decision.selected_candidate_id else None,
        "selected_budget": decision.selected_budget if decision else None,
        "selected_task_suite": decision.selected_task_suite if decision else None,
        "selected_task_ids": list(decision.selected_task_ids) if decision else [],
        "selected_using_c_results": False,
        "variant_codes_used": ["A"] if rows else [],
        "variant_codes_used_subset_of_A_B_only": True,
        "formal_v22_execution_allowed": False,
        "canonical_rule": {
            "path": repo_rel(config.repo_root, config.canonical_rule_path),
            "sha256": preflight.canonical_rule_sha256,
            "read_as_single_source_of_truth": True,
        },
        "blocking_reasons": list(blocking_reasons),
    }
    scan_payload = {
        "schema_version": "iter5p5_w3p5_stock_A_budget_task_scan_v1",
        "run_id": RUN_ID,
        "updated_at_utc": terminal_at or utc_now(),
        "scan_status": status,
        "calibration_status": status,
        "gpu_rollout_started": bool(rows),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
        "variant_codes_considered": ["A"],
        "variant_codes_used": ["A"] if rows else [],
        "variant_codes_used_subset_of_A_B_only": True,
        "forbidden_variant_codes_used": [],
        "selected_using_c_results": False,
        "candidate_families_scanned": [config.task_suite_name] if rows else [],
        "candidate_budgets_scanned": list(config.budget_fractions) if rows else [],
        "expected_episode_count": len(episode_specs(config)),
        "observed_episode_count": len(rows),
        "failed_episode_count": len(failed_rows),
        "candidates": list(candidates),
        "blocking_reasons": list(blocking_reasons),
    }
    results_payload = {
        "schema_version": "iter5p5_w3p5_blind_calibration_results_v1",
        "run_id": RUN_ID,
        "updated_at_utc": terminal_at or utc_now(),
        "calibration_status": status,
        "selected_using_c_results": False,
        "variant_codes_used_subset_of_A_B_only": True,
        "formal_v22_execution_allowed": False,
        "per_episode_trace_path": repo_rel(
            config.repo_root,
            config.output_dir / "per_episode_trace.jsonl",
        ),
        "failed_episode_rows": list(failed_rows),
        "candidates": list(candidates),
    }
    attestation = {
        "schema_version": "iter5p5_w3p5_calibration_not_formal_claim_attestation_v1",
        "run_id": RUN_ID,
        "captured_at_utc": terminal_at or utc_now(),
        "calibration_is_not_formal_v22": True,
        "formal_v22_execution_allowed": False,
        "selected_using_c_results": False,
        "forbidden_variant_codes_used": [],
        "machine_status_only": True,
    }
    write_json(config.output_dir / "calibration_run_manifest.json", manifest)
    write_json(config.output_dir / "stock_A_budget_task_scan.json", scan_payload)
    write_json(config.output_dir / "desaturation_selection_decision.json", decision_payload)
    write_json(config.output_dir / "blind_calibration_results.json", results_payload)
    write_json(config.output_dir / "calibration_not_formal_claim_attestation.json", attestation)
    legacy_full_rewrite_jsonl(config.output_dir / "per_episode_trace.jsonl", rows)
    legacy_full_rewrite_jsonl(config.output_dir / "failed_episode_rows.jsonl", failed_rows)
    return {
        "manifest": manifest,
        "scan": scan_payload,
        "decision": decision_payload,
        "results": results_payload,
        "attestation": attestation,
    }


@dataclass
class Iter5p5BlindCalibrationWorkflow:
    config: CalibrationConfig

    def run(self) -> int:
        started_at = utc_now()
        preflight = validate_preflight(self.config)
        if preflight.blocking_reasons:
            terminal_at = utc_now()
            write_outputs(
                config=self.config,
                preflight=preflight,
                status=STATUS_BLOCK_PRECONDITION,
                started_at=started_at,
                terminal_at=terminal_at,
                context=None,
                rows=(),
                failed_rows=(),
                decision=None,
                retry_attempted=False,
                runtime_signature=None,
                blocking_reasons=preflight.blocking_reasons,
            )
            print("ITER5P5_W3P5_BLOCK_PRECONDITION", flush=True)
            return 0

        context = runtime_context(self.config)
        log_line(
            context.harness_log,
            f"W3P5_START started_at_utc={started_at} max_workers={self.config.max_workers}",
        )
        try:
            rows, failed_rows, runtime_signature = run_rollouts(self.config, context)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            runtime_signature = runtime_defect_signature(error)
            status = (
                STATUS_BLOCK_RUNTIME
                if runtime_signature in Q8_RUNTIME_DEFECT_SIGNATURES
                else STATUS_BLOCK_LOGIC
            )
            terminal_at = utc_now()
            failed_rows = (
                {
                    "episode_status": "launch_failed",
                    "error": error,
                    "runtime_defect_signature": runtime_signature,
                },
            )
            blocking_reasons = (
                f"runtime_defect_{runtime_signature}"
                if status == STATUS_BLOCK_RUNTIME
                else "openpi_runtime_launch_failed",
            )
            write_outputs(
                config=self.config,
                preflight=preflight,
                status=status,
                started_at=started_at,
                terminal_at=terminal_at,
                context=context,
                rows=(),
                failed_rows=failed_rows,
                decision=None,
                retry_attempted=False,
                runtime_signature=runtime_signature,
                blocking_reasons=blocking_reasons,
            )
            log_line(context.harness_log, f"W3P5_DONE status={status} error={error}")
            print(f"ITER5P5_W3P5_DONE status={status}", flush=True)
            return 0
        specs = episode_specs(self.config)
        terminal_at = utc_now()
        if failed_rows and runtime_signature in Q8_RUNTIME_DEFECT_SIGNATURES:
            status = STATUS_BLOCK_RUNTIME
            decision = None
            blocking_reasons = (f"runtime_defect_{runtime_signature}",)
        elif failed_rows and not rows:
            status = STATUS_BLOCK_LOGIC
            decision = None
            blocking_reasons = ("all_calibration_episodes_failed",)
        else:
            candidates = candidate_metrics(
                rows=rows,
                failed_rows=failed_rows,
                specs=specs,
                config=self.config,
            )
            decision = decide(candidates, failed_rows)
            status = decision.calibration_status
            blocking_reasons = decision.blocking_reasons
        write_outputs(
            config=self.config,
            preflight=preflight,
            status=status,
            started_at=started_at,
            terminal_at=terminal_at,
            context=context,
            rows=rows,
            failed_rows=failed_rows,
            decision=decision,
            retry_attempted=False,
            runtime_signature=runtime_signature,
            blocking_reasons=blocking_reasons,
        )
        log_line(context.harness_log, f"W3P5_DONE status={status} rows={len(rows)} failed={len(failed_rows)}")
        print(f"ITER5P5_W3P5_DONE status={status}", flush=True)
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    return Iter5p5BlindCalibrationWorkflow(config=config).run()


if __name__ == "__main__":
    raise SystemExit(main())
