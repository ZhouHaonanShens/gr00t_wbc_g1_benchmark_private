from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OPENPI_ROOT = REPO_ROOT / "submodules/openpi"
RUN_ID = "stage1_v22_full_training_eval_iter9_20260426T_nextZ"


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


from work.openpi.eval.v22_formal_eval_contracts import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    PreregHashLock,
    VariantAuthorityManifest,
    load_prereg_hash_lock,
    load_variant_authority_manifest,
    paired_bootstrap_ci,
    protocol_equal,
    validate_hash_lock,
    validate_variant_authority_manifest,
)
from work.openpi.pipelines.recap.blind_calibration_runtime import (  # noqa: E402
    Sha256Sums,
    atomic_json_write,
    atomic_jsonl_write,
    read_json_object,
    repo_rel,
    utc_now,
)


@dataclass(frozen=True)
class FormalEvalConfig:
    prereg_hash_lock: Path
    prereg_hash_lock_sha256: str | None
    variant_authority_manifest: Path | None
    output_dir: Path
    runtime_log_dir: Path
    mode: str
    n_per_variant: int
    episodes_per_variant: int
    variants: tuple[str, ...]
    resume: bool
    skip_completed: bool
    cuda_visible_devices: str
    no_sudo: bool
    suite: str | None
    budget: float | None
    step_cap: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v22_formal_eval_runner.py",
        description="Iter9 v22 formal eval runner surface.",
    )
    parser.add_argument("--prereg-hash-lock", required=True)
    parser.add_argument("--prereg-hash-lock-sha256")
    parser.add_argument("--variant-authority-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-log-dir", required=True)
    parser.add_argument("--n-per-variant", type=int, default=192)
    parser.add_argument("--variants", default="A,B,C,X")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--no-sudo", action="store_true")
    parser.add_argument("--mode", choices=("dry-run", "smoke", "long-run"), required=True)
    parser.add_argument("--episodes-per-variant", type=int, default=192)
    parser.add_argument("--suite")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--step-cap", type=int)
    return parser


def config_from_args(args: argparse.Namespace) -> FormalEvalConfig:
    return FormalEvalConfig(
        prereg_hash_lock=_resolve_repo_path(args.prereg_hash_lock),
        prereg_hash_lock_sha256=args.prereg_hash_lock_sha256,
        variant_authority_manifest=(
            _resolve_repo_path(args.variant_authority_manifest)
            if args.variant_authority_manifest
            else None
        ),
        output_dir=_resolve_repo_path(args.output_dir),
        runtime_log_dir=_resolve_repo_path(args.runtime_log_dir),
        mode=str(args.mode),
        n_per_variant=max(1, int(args.n_per_variant)),
        episodes_per_variant=max(1, int(args.episodes_per_variant)),
        variants=_variant_tuple(args.variants),
        resume=bool(args.resume),
        skip_completed=bool(args.skip_completed),
        cuda_visible_devices=str(args.cuda_visible_devices),
        no_sudo=bool(args.no_sudo),
        suite=str(args.suite) if args.suite else None,
        budget=float(args.budget) if args.budget is not None else None,
        step_cap=int(args.step_cap) if args.step_cap is not None else None,
    )


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _variant_tuple(raw: str) -> tuple[str, ...]:
    normalized = raw.replace(",", " ")
    return tuple(item.strip() for item in normalized.split() if item.strip())


def _repo_path(path: Path) -> str:
    return repo_rel(REPO_ROOT, path)


def _append_exception_reason(reasons: list[str], exc: Exception) -> None:
    text = str(exc)
    if text.startswith("BLOCK_"):
        reasons.append(text.split(":", 1)[0])
        return
    reasons.append(f"BLOCK_FORMAL_EVAL_EXCEPTION:{type(exc).__name__}")


def validate_preconditions(
    config: FormalEvalConfig,
) -> tuple[dict[str, object], PreregHashLock | None, VariantAuthorityManifest | None]:
    blocking_reasons: list[str] = []
    lock: PreregHashLock | None = None
    manifest: VariantAuthorityManifest | None = None
    hash_lock_sha = ""
    try:
        lock = load_prereg_hash_lock(
            config.prereg_hash_lock,
            expected_sha256=config.prereg_hash_lock_sha256,
        )
        hash_lock_sha = lock.sha256
        blocking_reasons.extend(validate_hash_lock(lock))
        blocking_reasons.extend(_protocol_mutation_reasons(config, lock))
    except Exception as exc:  # noqa: BLE001
        _append_exception_reason(blocking_reasons, exc)

    if not config.no_sudo:
        blocking_reasons.append("BLOCK_NO_SUDO_FLAG_REQUIRED")
    if config.cuda_visible_devices not in {"1", "2"}:
        blocking_reasons.append("BLOCK_CUDA_VISIBLE_DEVICES_BOUNDARY")
    visible = {item.strip() for item in config.cuda_visible_devices.split(",") if item.strip()}
    if visible & {"0", "3"}:
        blocking_reasons.append("BLOCK_CUDA_VISIBLE_DEVICES_BOUNDARY")

    if config.variant_authority_manifest is None:
        blocking_reasons.append("BLOCK_VARIANT_AUTHORITY_MANIFEST_MISSING")
    elif lock is not None:
        try:
            manifest = load_variant_authority_manifest(config.variant_authority_manifest)
            blocking_reasons.extend(validate_variant_authority_manifest(manifest, lock))
        except Exception as exc:  # noqa: BLE001
            _append_exception_reason(blocking_reasons, exc)

    blocking_reasons = list(dict.fromkeys(blocking_reasons))
    payload: dict[str, object] = {
        "schema_version": "v22_formal_eval_precondition_v1",
        "run_id": RUN_ID,
        "checked_at_utc": utc_now(),
        "status": "BLOCK" if blocking_reasons else "PASS",
        "mode": config.mode,
        "hash_lock": {
            "path": _repo_path(config.prereg_hash_lock),
            "present": config.prereg_hash_lock.is_file(),
            "sha256": hash_lock_sha,
            "expected_sha256": (
                config.prereg_hash_lock_sha256 or (lock.expected_sha256 if lock else None)
            ),
            "schema_version": lock.schema_version if lock else "",
            "sha256_matches": (
                lock is not None
                and (not (config.prereg_hash_lock_sha256 or lock.expected_sha256)
                     or (config.prereg_hash_lock_sha256 or lock.expected_sha256) == lock.sha256)
            ),
        },
        "selected_protocol": dict(lock.selected_protocol) if lock else {},
        "variants": list(config.variants),
        "n_per_variant": config.n_per_variant,
        "episodes_per_variant": config.episodes_per_variant,
        "variant_authority_manifest": {
            "path": _repo_path(config.variant_authority_manifest)
            if config.variant_authority_manifest
            else None,
            "loadable": manifest is not None,
            "formal_eval_allowed": manifest.formal_eval_allowed if manifest else False,
            "selected_protocol_matches_hash_lock": (
                manifest is not None
                and lock is not None
                and (
                    not manifest.selected_protocol
                    or protocol_equal(manifest.selected_protocol, lock.selected_protocol)
                )
            ),
            "hash_lock_sha256": manifest.hash_lock_sha256 if manifest else None,
        },
        "resource_boundary": {
            "cuda_visible_devices_requested": config.cuda_visible_devices,
            "gpu0_gpu3_forbidden": True,
            "sudo_forbidden": True,
            "no_sudo": config.no_sudo,
        },
        "paired_bootstrap_ci_helper_present": True,
        "paired_bootstrap_ci_default_n_resamples": DEFAULT_BOOTSTRAP_RESAMPLES,
        "paired_bootstrap_ci_default_seed": DEFAULT_BOOTSTRAP_SEED,
        "blocking_reasons": blocking_reasons,
    }
    return payload, lock, manifest


def _protocol_mutation_reasons(
    config: FormalEvalConfig,
    lock: PreregHashLock,
) -> list[str]:
    reasons: list[str] = []
    if config.n_per_variant != lock.n_per_variant:
        reasons.append("BLOCK_PROTOCOL_N_PER_VARIANT_MUTATION")
    if config.episodes_per_variant != lock.n_per_variant:
        reasons.append("BLOCK_PROTOCOL_N_PER_VARIANT_MUTATION")
    if config.variants != lock.variants:
        reasons.append("BLOCK_PROTOCOL_VARIANTS_MUTATION")
    if config.suite is not None and config.suite != lock.suite:
        reasons.append("BLOCK_PROTOCOL_SUITE_MUTATION")
    if config.budget is not None and config.budget != lock.budget:
        reasons.append("BLOCK_PROTOCOL_BUDGET_MUTATION")
    if config.step_cap is not None and config.step_cap != lock.step_cap:
        reasons.append("BLOCK_PROTOCOL_STEP_CAP_MUTATION")
    return reasons


def build_formal_eval_plan(
    config: FormalEvalConfig,
    lock: PreregHashLock,
) -> dict[str, object]:
    return {
        "schema_version": "v22_formal_eval_plan_v1",
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "mode": config.mode,
        "selected_protocol": dict(lock.selected_protocol),
        "variants": list(lock.variants),
        "n_per_variant": lock.n_per_variant,
        "episodes_per_variant": config.episodes_per_variant,
        "total_episodes": lock.n_per_variant * len(lock.variants),
        "variant_output_dirs": {
            variant: _repo_path(_variant_dir(config.output_dir, variant))
            for variant in lock.variants
        },
        "required_outputs_per_variant": [
            "per_episode_trace.jsonl",
            "summary.json",
            "metric_ladder_summary.json",
            "bootstrap_ci.json",
            "SHA256SUMS",
        ],
        "resume": config.resume,
        "skip_completed": config.skip_completed,
        "paired_bootstrap_ci_helper_present": True,
    }


def build_variant_manifest_requirements(lock: PreregHashLock) -> dict[str, object]:
    return {
        "schema_version": "v22_variant_manifest_requirements_v1",
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "formal_eval_allowed_required": True,
        "hash_lock_sha256_required": lock.sha256,
        "selected_protocol_required": dict(lock.selected_protocol),
        "variants_required": list(lock.variants),
        "checkpoint_paths_required_for_variants": list(lock.variants),
        "missing_or_nonpassing_manifest_blocks_eval_start": True,
    }


def build_resume_index(
    output_dir: Path,
    variants: Sequence[str],
    *,
    expected_episode_count: int,
    skip_completed: bool,
) -> dict[str, object]:
    completed: list[str] = []
    skipped: list[str] = []
    incomplete: list[str] = []
    rerun_required: list[str] = []
    predicates: dict[str, object] = {}
    for variant in variants:
        predicate = variant_completion_predicate(
            _variant_dir(output_dir, variant),
            expected_episode_count=expected_episode_count,
        )
        predicates[str(variant)] = predicate
        if predicate["skip"]:
            completed.append(str(variant))
            if skip_completed:
                skipped.append(str(variant))
            continue
        incomplete.append(str(variant))
        rerun_required.append(str(variant))
    return {
        "schema_version": "v22_formal_eval_resume_index_v1",
        "generated_at_utc": utc_now(),
        "total_variants": len(variants),
        "completed_variants": completed,
        "incomplete_variants": incomplete,
        "skipped_variants": skipped,
        "rerun_required_variants": rerun_required,
        "variant_predicates": predicates,
    }


def variant_completion_predicate(
    variant_dir: Path,
    *,
    expected_episode_count: int,
) -> dict[str, object]:
    trace_path = variant_dir / "per_episode_trace.jsonl"
    summary_path = variant_dir / "summary.json"
    metric_path = variant_dir / "metric_ladder_summary.json"
    bootstrap_path = variant_dir / "bootstrap_ci.json"
    sha_path = variant_dir / "SHA256SUMS"
    row_count = _jsonl_row_count(trace_path) if trace_path.is_file() else 0
    summary = read_json_object(summary_path) if summary_path.is_file() else {}
    predicate = {
        "trace_present": trace_path.is_file(),
        "summary_present": summary_path.is_file(),
        "metric_ladder_present": metric_path.is_file(),
        "bootstrap_ci_present": bootstrap_path.is_file(),
        "sha256sums_present": sha_path.is_file(),
        "episode_count": row_count,
        "expected_episode_count": expected_episode_count,
        "summary_status": summary.get("status"),
    }
    predicate["skip"] = (
        predicate["trace_present"] is True
        and predicate["summary_present"] is True
        and predicate["metric_ladder_present"] is True
        and predicate["bootstrap_ci_present"] is True
        and predicate["sha256sums_present"] is True
        and row_count >= expected_episode_count
        and summary.get("status") in {"PASS", "BLOCK", "NEGATIVE_RESULT"}
    )
    return predicate


def run_dry_run(
    config: FormalEvalConfig,
    precondition: Mapping[str, object],
    lock: PreregHashLock | None,
) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.runtime_log_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(config.output_dir / "precondition_check.json", precondition)
    if lock is None:
        return 2
    atomic_json_write(config.output_dir / "formal_eval_plan.json", build_formal_eval_plan(config, lock))
    atomic_json_write(
        config.output_dir / "variant_manifest_requirements.json",
        build_variant_manifest_requirements(lock),
    )
    atomic_json_write(
        config.output_dir / "resume_index.json",
        build_resume_index(
            config.output_dir,
            lock.variants,
            expected_episode_count=config.episodes_per_variant,
            skip_completed=config.skip_completed,
        ),
    )
    return 0


def run_eval(
    config: FormalEvalConfig,
    precondition: Mapping[str, object],
    lock: PreregHashLock | None,
    manifest: VariantAuthorityManifest | None,
) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.runtime_log_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(config.output_dir / "precondition_check.json", precondition)
    if precondition.get("status") != "PASS" or lock is None or manifest is None:
        _write_eval_status(
            config,
            status="BLOCK",
            blocking_reasons=list(precondition.get("blocking_reasons", ())),
        )
        return 2
    if os.environ.get("V22_FORMAL_EVAL_SYNTHETIC") == "1":
        return _run_synthetic_eval(config, lock)
    return _run_real_eval(config, lock, manifest)


def _run_synthetic_eval(config: FormalEvalConfig, lock: PreregHashLock) -> int:
    skipped: list[str] = []
    for variant in lock.variants:
        resume = build_resume_index(
            config.output_dir,
            lock.variants,
            expected_episode_count=config.episodes_per_variant,
            skip_completed=config.skip_completed,
        )
        if config.skip_completed and variant in resume["skipped_variants"]:
            skipped.append(variant)
            continue
        rows = _synthetic_rows(lock, variant=variant, episodes=config.episodes_per_variant)
        _write_variant_outputs(config.output_dir, lock, variant=variant, rows=rows)
    _write_eval_status(config, status="PASS", skipped=skipped, synthetic=True)
    return 0


def _run_real_eval(
    config: FormalEvalConfig,
    lock: PreregHashLock,
    manifest: VariantAuthorityManifest,
) -> int:
    try:
        from work.openpi.pipelines.recap.blind_calibration_inference import (
            _run_real_episode,
            build_libero_episode_env,
            load_variant_A,
            load_variant_B_optional,
        )
    except Exception as exc:  # noqa: BLE001
        _write_eval_status(
            config,
            status="BLOCK",
            blocking_reasons=[f"BLOCK_REAL_EVAL_IMPORT_FAILED:{type(exc).__name__}"],
        )
        return 2
    policies: dict[str, object] = {}
    for variant in lock.variants:
        try:
            if variant == "A":
                authority_path = manifest.checkpoint_path_for("A") or manifest.path
                policies[variant] = load_variant_A(authority_path)
            else:
                checkpoint_path = manifest.checkpoint_path_for(variant)
                if checkpoint_path is None:
                    raise RuntimeError(f"BLOCK_VARIANT_CHECKPOINT_MISSING:{variant}")
                policy = load_variant_B_optional(checkpoint_path)
                if policy is None:
                    raise RuntimeError(f"BLOCK_VARIANT_CHECKPOINT_MISSING:{variant}")
                policies[variant] = policy
        except Exception as exc:  # noqa: BLE001
            _write_eval_status(
                config,
                status="BLOCK",
                blocking_reasons=[_block_reason_from_exception(exc)],
            )
            return 2

    skipped: list[str] = []
    for variant in lock.variants:
        resume = build_resume_index(
            config.output_dir,
            lock.variants,
            expected_episode_count=config.episodes_per_variant,
            skip_completed=config.skip_completed,
        )
        if config.skip_completed and variant in resume["skipped_variants"]:
            skipped.append(variant)
            continue
        rows: list[dict[str, object]] = []
        for episode_index in range(config.episodes_per_variant):
            seed = DEFAULT_BOOTSTRAP_SEED + episode_index
            env = build_libero_episode_env(
                suite_family=lock.suite,
                tasks=lock.tasks,
                episode_index=episode_index,
                seed=seed,
            )
            try:
                result = _run_real_episode(env, policies[variant], max_steps=lock.step_cap, seed=seed)
            finally:
                env.close()
            rows.append(_episode_row(lock, variant=variant, episode_index=episode_index, result=result))
        _write_variant_outputs(config.output_dir, lock, variant=variant, rows=rows)
    _write_eval_status(config, status="PASS", skipped=skipped, synthetic=False)
    return 0


def _write_variant_outputs(
    output_dir: Path,
    lock: PreregHashLock,
    *,
    variant: str,
    rows: Sequence[Mapping[str, object]],
) -> None:
    variant_dir = _variant_dir(output_dir, variant)
    variant_dir.mkdir(parents=True, exist_ok=True)
    sums = Sha256Sums(variant_dir)
    trace_path = variant_dir / "per_episode_trace.jsonl"
    summary_path = variant_dir / "summary.json"
    metric_path = variant_dir / "metric_ladder_summary.json"
    bootstrap_path = variant_dir / "bootstrap_ci.json"
    atomic_jsonl_write(trace_path, rows)
    sums.record(trace_path)
    summary = _variant_summary(lock, variant=variant, rows=rows)
    atomic_json_write(summary_path, summary)
    sums.record(summary_path)
    atomic_json_write(metric_path, _metric_ladder_summary(summary))
    sums.record(metric_path)
    atomic_json_write(bootstrap_path, _variant_bootstrap_placeholder(summary))
    sums.record(bootstrap_path)
    sums.write(variant_dir / "SHA256SUMS")


def _variant_summary(
    lock: PreregHashLock,
    *,
    variant: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    episode_count = len(rows)
    success_count = sum(1 for row in rows if bool(row.get("success")))
    timeout_count = sum(1 for row in rows if bool(row.get("timeout_flag")))
    denominator = max(episode_count, 1)
    return {
        "schema_version": "v22_formal_eval_variant_summary_v1",
        "run_id": RUN_ID,
        "variant": variant,
        "status": "PASS",
        "suite": lock.suite,
        "step_cap": lock.step_cap,
        "episode_count": episode_count,
        "success_count": success_count,
        "success_rate": success_count / denominator if episode_count else None,
        "timeout_count": timeout_count,
        "timeout_rate": timeout_count / denominator if episode_count else None,
        "trace_completeness": 1.0 if episode_count else None,
    }


def _metric_ladder_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "v22_formal_eval_metric_ladder_summary_v1",
        "success_rate": summary.get("success_rate"),
        "trace_completeness": summary.get("trace_completeness"),
        "timeout_rate": summary.get("timeout_rate"),
    }


def _variant_bootstrap_placeholder(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "v22_formal_eval_bootstrap_ci_v1",
        "computed": False,
        "reason": "paired_ci_computed_by_cross_variant_gate",
        "success_rate": summary.get("success_rate"),
    }


def _write_eval_status(
    config: FormalEvalConfig,
    *,
    status: str,
    blocking_reasons: Sequence[str] = (),
    skipped: Sequence[str] = (),
    synthetic: bool | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "v22_formal_eval_status_v1",
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "mode": config.mode,
        "status": status,
        "skipped_variants": list(skipped),
        "blocking_reasons": list(blocking_reasons),
    }
    if synthetic is not None:
        payload["synthetic_test_stub"] = synthetic
    atomic_json_write(config.output_dir / "formal_eval_status.json", payload)


def _synthetic_rows(
    lock: PreregHashLock,
    *,
    variant: str,
    episodes: int,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    variant_offset = {"A": 0, "B": 1, "C": 2, "X": 3}.get(variant, 0)
    for episode_index in range(episodes):
        success = (episode_index + variant_offset) % 3 != 0
        rows.append(
            _episode_row(
                lock,
                variant=variant,
                episode_index=episode_index,
                result={
                    "seed": DEFAULT_BOOTSTRAP_SEED + episode_index,
                    "success": success,
                    "timeout_flag": False,
                    "trace_completeness": 1.0,
                    "steps_taken": min(lock.step_cap, 5 + episode_index),
                    "terminal_reason": "synthetic_test_stub",
                },
                synthetic=True,
            )
        )
    return tuple(rows)


def _episode_row(
    lock: PreregHashLock,
    *,
    variant: str,
    episode_index: int,
    result: Mapping[str, object],
    synthetic: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "v22_formal_eval_per_episode_trace_v1",
        "run_id": RUN_ID,
        "variant": variant,
        "suite": lock.suite,
        "step_cap": lock.step_cap,
        "episode_index": episode_index,
        "pairing_key": lock.pairing_rule.key_for_episode(episode_index),
        "seed": int(result.get("seed") or DEFAULT_BOOTSTRAP_SEED + episode_index),
        "success": bool(result.get("success")),
        "timeout_flag": bool(result.get("timeout_flag")),
        "trace_completeness": float(result.get("trace_completeness") or 0.0),
        "steps_taken": int(result.get("steps_taken") or 0),
        "terminal_reason": str(result.get("terminal_reason") or ""),
        "synthetic_test_stub": synthetic,
    }


def _jsonl_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _variant_dir(output_dir: Path, variant: str) -> Path:
    return output_dir / str(variant)


def _block_reason_from_exception(exc: Exception) -> str:
    text = str(exc)
    if text.startswith("BLOCK_"):
        return text.split(":", 1)[0]
    return f"BLOCK_FORMAL_EVAL_EXCEPTION:{type(exc).__name__}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    precondition, lock, manifest = validate_preconditions(config)
    if config.mode == "dry-run":
        return run_dry_run(config, precondition, lock)
    return run_eval(config, precondition, lock, manifest)


__all__ = [
    "FormalEvalConfig",
    "build_formal_eval_plan",
    "build_parser",
    "build_resume_index",
    "config_from_args",
    "main",
    "paired_bootstrap_ci",
    "run_dry_run",
    "validate_preconditions",
    "variant_completion_predicate",
]


if __name__ == "__main__":
    raise SystemExit(main())
