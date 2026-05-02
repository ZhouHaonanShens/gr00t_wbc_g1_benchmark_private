#!/usr/bin/env python3
"""Real implementation for the LIBERO rollout eval v21 authority lane.

The public CLI and compatibility import surface stays in
``work.openpi.scripts.libero_rollout_eval_v21``. This module owns the actual
workflow, runtime/source reconciliation, trace construction, and authority
bundle materialization logic.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import datetime as dt
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import shutil
from typing import Any, Callable, cast


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.eval.protocols.manifest import (  # noqa: E402
    ROLLOUT_INPUT_SUMMARY_SCHEMA_VERSION,
    compute_rollout_manifest_hash_v21,
    load_rollout_manifest_v21,
    manifest_payload_v21,
)
from work.openpi.contracts import RuntimeServerSpec  # noqa: E402
from work.openpi.checkpoint import (  # noqa: E402
    candidate_rollout_source_dirs,
    load_checkpoint_provenance_pair,
    resolve_rollout_source_dir,
)
from work.openpi.dataloader import (  # noqa: E402
    json_ready,
    load_rollout_eval_v21_authority_bundle as load_rollout_eval_v21_authority_bundle_io,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    write_markdown,
)
from work.openpi.recap.runtime_prompt import (  # noqa: E402
    RuntimeIndicatorConfig,
    build_runtime_prompt_bundle,
    resolve_runtime_indicator_config,
)
from work.openpi.eval.reports.go_no_go import (  # noqa: E402
    BOOTSTRAP_SCHEMA_VERSION,
    COMPATIBILITY_ONLY_METRICS,
    DEFAULT_PRIMARY_METRIC_ID,
    HEADLINE_METRIC_ORDER,
    METRIC_LADDER_SCHEMA_VERSION,
    PAIRWISE_DELTA_SCHEMA_VERSION,
    AggregationValidationError,
    assert_variant_aggregate_conservation_v21,
    build_bootstrap_ci_v21,
    build_metric_ladder_summary_v21,
    metric_point_estimates_from_trace_rows_v21,
)
from work.openpi.model import (  # noqa: E402
    build_effective_runtime_spec as build_effective_runtime_spec_model,
    build_expected_runtime_prompting_payload,
    build_rollout_input_summary_v21 as build_rollout_input_summary_v21_model,
    effective_runtime_spec_from_runtime_prompting as effective_runtime_spec_from_runtime_prompting_model,
    effective_runtime_spec_hash as effective_runtime_spec_hash_model,
    effective_runtime_spec_without_request as effective_runtime_spec_without_request_model,
    effective_runtime_surface_signature as effective_runtime_surface_signature_model,
    normalize_effective_runtime_spec as normalize_effective_runtime_spec_model,
    normalize_runtime_prompting_payload as normalize_runtime_prompting_payload_model,
    observed_runtime_prompting_from_rows as observed_runtime_prompting_from_rows_model,
    resolve_runtime_prompting_payload as resolve_runtime_prompting_payload_model,
)
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
from work.openpi.runtime.api import FailFastError as RuntimeBridgeError  # noqa: E402
from work.openpi.serve.provenance import EXPECTED_CHECKPOINT  # noqa: E402


SUMMARY_SCHEMA_VERSION = "openpi_libero_rollout_eval_summary_v21"
TRACE_NAME = "per_episode_trace.jsonl"
METRIC_LADDER_NAME = "metric_ladder_summary.json"
BOOTSTRAP_NAME = "bootstrap_ci.json"
PAIRWISE_DELTA_NAME = "pairwise_delta.json"
SUMMARY_NAME = "summary.json"
EVAL_MANIFEST_NAME = "eval_manifest.json"
DEVIATION_NOTES_NAME = "deviation_notes.md"
TOPIC = "openpi_libero_v21"
PRIMARY_METRIC_ID = DEFAULT_PRIMARY_METRIC_ID
TRACE_REQUIRED_FIELDS: tuple[str, ...] = (
    "variant",
    "task_id",
    "seed",
    "trial_idx",
    "success",
    "first_success_step",
    "executed_steps",
    "max_steps_resolved",
    "success_within_50pct_budget",
    "success_within_75pct_budget",
    "timeout_flag",
    "deviation_notes",
)
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95
V2_INPUT_PER_EPISODE_NAME = "per_episode_rollouts.jsonl"
ROLLOUT_INPUT_SUMMARY_NAME = "rollout_input_summary.json"
rollout_v2 = importlib.import_module("work.openpi.eval.workflows.tracked_gate")


class FailFastError(RuntimeError):
    pass


EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION = "openpi_libero_effective_runtime_spec_v1"
CHECKPOINT_INSTANCE_BINDING_SCHEMA_VERSION = (
    "openpi_libero_checkpoint_instance_binding_v1"
)
CHECKPOINT_INSTANCE_BINDING_KEY_FILES: tuple[Path, ...] = (
    Path("checkpoint.json"),
    Path("train_manifest.json"),
    Path("checkpoint_provenance.json"),
    Path("params") / "_METADATA",
    Path("assets") / "physical-intelligence" / "libero" / "norm_stats.json",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libero_rollout_eval_v21.py",
        description=(
            "Materialize v21 trace-capable LIBERO authority bundles without "
            "changing rollout semantics."
        ),
    )
    _ = parser.add_argument("--variant", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    _ = checkpoint_group.add_argument(
        "--checkpoint-source",
        choices=("stock",),
        help="Use the frozen stock checkpoint source.",
    )
    _ = checkpoint_group.add_argument(
        "--checkpoint-dir",
        help="Local checkpoint directory or explicit remote checkpoint reference.",
    )
    _ = parser.add_argument("--manifest", required=True)
    _ = parser.add_argument("--metric-profile", required=True)
    _ = parser.add_argument("--output-dir", required=True)
    _ = parser.add_argument(
        "--indicator-mode",
        choices=("positive", "negative", "omit", "cfg"),
        default="cfg",
    )
    _ = parser.add_argument(
        "--resolved-runtime-indicator-mode",
        default="",
        help=argparse.SUPPRESS,
    )
    _ = parser.add_argument(
        "--resolved-runtime-indicator-source",
        default="",
        help=argparse.SUPPRESS,
    )
    _ = parser.add_argument(
        "--resolved-runtime-consumer-mode",
        default="",
        help=argparse.SUPPRESS,
    )
    _ = parser.add_argument(
        "--resolved-runtime-fixed-indicator-mode",
        default="",
        help=argparse.SUPPRESS,
    )
    _ = parser.add_argument(
        "--resolved-runtime-critic-checkpoint-ref",
        default="",
        help=argparse.SUPPRESS,
    )
    _ = parser.add_argument(
        "--canonical-source-dir",
        default="",
        help=argparse.SUPPRESS,
    )
    return parser


def _json_ready(value: Any) -> Any:
    return json_ready(value)


def _write_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    write_jsonl(path, rows)


def _write_markdown(path: Path, text: str) -> None:
    write_markdown(path, text)


def _stable_json_dumps(payload: Mapping[str, object]) -> str:
    return json.dumps(
        _json_ready(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        return read_json(path)
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        return read_jsonl(path)
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def _log(message: str, *, log_path: Path | None = None) -> None:
    print(message, flush=True)
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        _ = handle.write(message)
        _ = handle.write("\n")


def _require_mapping(raw: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise FailFastError(f"{context} must be a mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


def _require_sequence(raw: object, *, context: str) -> Sequence[object]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise FailFastError(f"{context} must be a sequence")
    return cast(Sequence[object], raw)


def _coerce_int(raw: object, *, context: str) -> int:
    if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise FailFastError(f"{context} must be integer-like, got {raw!r}")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise FailFastError(f"{context} must be integer-like, got {raw!r}") from exc


def _coerce_float(raw: object, *, context: str) -> float:
    if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise FailFastError(f"{context} must be float-like, got {raw!r}")
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise FailFastError(f"{context} must be float-like, got {raw!r}") from exc


def _coerce_bool(raw: object, *, context: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise FailFastError(f"{context} must be boolean-like, got {raw!r}")


def _require_non_empty_str(raw: object, *, context: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise FailFastError(f"{context} must be a non-empty string")
    return value


def _coerce_int_sequence(raw: object, *, context: str) -> tuple[int, ...]:
    sequence = _require_sequence(raw, context=context)
    return tuple(_coerce_int(value, context=f"{context}[]") for value in sequence)


def _coerce_string_list(raw: object, *, context: str) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    sequence = _require_sequence(raw, context=context)
    return [str(value).strip() for value in sequence if str(value).strip()]


def _normalize_variant(raw_variant: str) -> str:
    variant = raw_variant.strip()
    if not variant:
        raise FailFastError("--variant must be non-empty")
    return variant


def _normalize_eval_manifest_payload(eval_manifest: object) -> dict[str, object]:
    if is_dataclass(eval_manifest) and not isinstance(eval_manifest, type):
        raw_payload = asdict(eval_manifest)
        compact_payload = {
            str(key): value for key, value in raw_payload.items() if value is not None
        }
        return manifest_payload_v21(compact_payload)
    if isinstance(eval_manifest, Mapping):
        return manifest_payload_v21(eval_manifest)
    raise FailFastError(
        "eval_manifest must be a validated mapping or dataclass-like payload"
    )


def _resolve_task_seed_manifests(
    eval_manifest: Mapping[str, object], *, context: str
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    manifest = _normalize_eval_manifest_payload(eval_manifest)
    task_ids = _coerce_int_sequence(
        manifest.get("task_ids"), context=f"{context}.task_ids"
    )
    raw_per_task_seed_manifest = manifest.get("per_task_seed_manifest")
    if raw_per_task_seed_manifest is not None:
        per_task_seed_manifest = _require_mapping(
            raw_per_task_seed_manifest,
            context=f"{context}.per_task_seed_manifest",
        )
        normalized_entries: list[tuple[int, tuple[int, ...]]] = []
        for raw_task_id, raw_seed_manifest in per_task_seed_manifest.items():
            task_id = _coerce_int(
                raw_task_id,
                context=f"{context}.per_task_seed_manifest task_id",
            )
            seed_manifest = _coerce_int_sequence(
                raw_seed_manifest,
                context=f"{context}.per_task_seed_manifest[{task_id}]",
            )
            normalized_entries.append((task_id, seed_manifest))
        normalized_entries.sort(key=lambda item: item[0])
        actual_task_ids = tuple(task_id for task_id, _ in normalized_entries)
        if actual_task_ids != task_ids:
            raise FailFastError(
                f"{context}.per_task_seed_manifest task coverage mismatch; expected {task_ids!r}, got {actual_task_ids!r}"
            )
        return tuple(normalized_entries)

    shared_seed_manifest = _coerce_int_sequence(
        manifest.get("seed_manifest"), context=f"{context}.seed_manifest"
    )
    return tuple((task_id, shared_seed_manifest) for task_id in task_ids)


def build_eval_manifest_id(eval_manifest: Mapping[str, object]) -> str:
    payload = _normalize_eval_manifest_payload(eval_manifest)
    manifest_name = str(payload["manifest_name"])
    manifest_hash = compute_rollout_manifest_hash_v21(payload)
    return f"{manifest_name}_{manifest_hash[:12]}"


def _build_runtime_dir(variant: str, eval_manifest_id: str) -> Path:
    return (
        REPO_ROOT
        / "agent"
        / "runtime_logs"
        / TOPIC
        / "rollouts"
        / variant
        / eval_manifest_id
    )


def _resolve_checkpoint_input(
    args: argparse.Namespace, variant: str
) -> tuple[str, str]:
    checkpoint_source = str(getattr(args, "checkpoint_source", "") or "").strip()
    checkpoint_dir = str(getattr(args, "checkpoint_dir", "") or "").strip()
    if checkpoint_source == "stock":
        if variant != "stock_libero_ref_v1":
            raise FailFastError(
                "--checkpoint-source stock is reserved for variant='stock_libero_ref_v1'"
            )
        return EXPECTED_CHECKPOINT, "stock"
    if not checkpoint_dir:
        raise FailFastError(
            "either --checkpoint-source stock or --checkpoint-dir is required"
        )
    return rollout_v2._normalize_checkpoint_ref(checkpoint_dir), "checkpoint_dir"


def _candidate_rollout_source_dirs(
    *, checkpoint_ref: str, raw_checkpoint_dir: str | None, output_dir: Path
) -> list[Path]:
    return list(
        candidate_rollout_source_dirs(
            checkpoint_ref=checkpoint_ref,
            raw_checkpoint_dir=raw_checkpoint_dir,
            output_dir=output_dir,
        )
    )


def _resolve_rollout_source_dir(
    *, checkpoint_ref: str, raw_checkpoint_dir: str | None, output_dir: Path
) -> Path:
    try:
        return resolve_rollout_source_dir(
            checkpoint_ref=checkpoint_ref,
            raw_checkpoint_dir=raw_checkpoint_dir,
            output_dir=output_dir,
            per_episode_name=V2_INPUT_PER_EPISODE_NAME,
        )
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def _load_checkpoint_provenance_pair(
    *, checkpoint_ref: str, raw_checkpoint_dir: str | None
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    return load_checkpoint_provenance_pair(
        checkpoint_ref=checkpoint_ref,
        raw_checkpoint_dir=raw_checkpoint_dir,
    )


def _checkpoint_instance_binding_from_dir(checkpoint_dir: Path) -> str:
    return resolve_checkpoint_instance_binding(str(checkpoint_dir))


def resolve_checkpoint_instance_binding(checkpoint_ref: str) -> str:
    try:
        from work.openpi.checkpoint import (
            resolve_checkpoint_instance_binding as resolve_checkpoint_instance_binding_io,
        )

        return resolve_checkpoint_instance_binding_io(
            checkpoint_ref,
            key_files=CHECKPOINT_INSTANCE_BINDING_KEY_FILES,
            schema_version=CHECKPOINT_INSTANCE_BINDING_SCHEMA_VERSION,
        )
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def _run_runtime_episode_subprocess(
    *,
    task_suite_name: str,
    task_id: int,
    seed: int,
    trial_idx: int,
    video_path: Path,
    host: str,
    port: int,
    venv_python: Path,
    openpi_root: Path,
    libero_config_dir: Path,
    runtime_dir: Path,
    timeout_s: float,
    checkpoint_ref: str,
    indicator_mode_requested: str,
    runtime_indicator_config: RuntimeIndicatorConfig,
) -> dict[str, object]:
    try:
        client = RuntimeEpisodeClient()
        return client.run_runtime_episode(
            task_suite_name=task_suite_name,
            task_id=task_id,
            seed=seed,
            trial_idx=trial_idx,
            video_path=video_path,
            host=host,
            port=port,
            venv_python=venv_python,
            openpi_root=openpi_root,
            libero_config_dir=libero_config_dir,
            runtime_dir=runtime_dir,
            timeout_s=timeout_s,
            checkpoint_ref=checkpoint_ref,
            indicator_mode_requested=indicator_mode_requested,
            runtime_indicator_config=runtime_indicator_config,
            cli_entry=Path(LIBERO_NATIVE_SMOKE_ENTRY),
        )
    except RuntimeBridgeError as exc:
        raise FailFastError(str(exc)) from exc


def _expected_episode_keys(
    *,
    task_seed_manifests: Sequence[tuple[int, tuple[int, ...]]],
    num_trials_per_task: int,
) -> set[tuple[int, int, int]]:
    return {
        (int(task_id), int(seed), int(trial_idx))
        for task_id, seed_manifest in task_seed_manifests
        for seed in seed_manifest
        for trial_idx in range(int(num_trials_per_task))
    }


def _episode_key_from_row(
    row: Mapping[str, object],
    *,
    context: str,
) -> tuple[int, int, int]:
    return (
        _coerce_int(row.get("task_id"), context=f"{context}.task_id"),
        _coerce_int(row.get("seed"), context=f"{context}.seed"),
        _coerce_int(
            row.get("trial_idx", row.get("trial_index")),
            context=f"{context}.trial_idx",
        ),
    )


def _load_resumable_rollout_rows(
    *,
    source_dir: Path,
    expected_keys: set[tuple[int, int, int]],
    log_path: Path,
) -> list[dict[str, object]]:
    input_path = source_dir / V2_INPUT_PER_EPISODE_NAME
    if not input_path.is_file():
        return []
    rows = _read_jsonl(input_path)
    seen: set[tuple[int, int, int]] = set()
    normalized_rows: list[dict[str, object]] = []
    for index, raw_row in enumerate(rows):
        mapping = _require_mapping(raw_row, context=f"resumable_rows[{index}]")
        key = _episode_key_from_row(mapping, context=f"resumable_rows[{index}]")
        if key in seen:
            raise FailFastError(
                "resumable rollout source contains duplicate episode "
                + f"task{key[0]}_seed{key[1]}_trial{key[2]}"
            )
        if key not in expected_keys:
            raise FailFastError(
                "resumable rollout source contains unexpected episode "
                + f"task{key[0]}_seed{key[1]}_trial{key[2]}"
            )
        seen.add(key)
        normalized_rows.append(dict(mapping))
    _log(
        f"resumable rollout rows detected: {len(normalized_rows)}/{len(expected_keys)}",
        log_path=log_path,
    )
    return normalized_rows


def _is_materialized_rollout_source_complete(
    *,
    source_dir: Path,
    eval_manifest: Mapping[str, object],
    log_path: Path,
) -> bool:
    manifest = _normalize_eval_manifest_payload(eval_manifest)
    task_seed_manifests = _resolve_task_seed_manifests(
        eval_manifest,
        context="eval_manifest",
    )
    num_trials_per_task = _coerce_int(
        manifest.get("num_trials_per_task"),
        context="eval_manifest.num_trials_per_task",
    )
    expected_keys = _expected_episode_keys(
        task_seed_manifests=task_seed_manifests,
        num_trials_per_task=num_trials_per_task,
    )
    rows = _load_resumable_rollout_rows(
        source_dir=source_dir,
        expected_keys=expected_keys,
        log_path=log_path,
    )
    if len(rows) == len(expected_keys):
        _log(
            f"reusing complete rollout source: {source_dir}",
            log_path=log_path,
        )
        return True
    if rows:
        _log(
            f"found partial rollout source, will resume: {source_dir}",
            log_path=log_path,
        )
    return False


def _build_rollout_input_summary(
    *,
    variant: str,
    checkpoint_ref: str,
    serve_checkpoint_ref: str,
    serve_checkpoint_mode: str,
    task_suite_name: str,
    task_seed_manifests: Sequence[tuple[int, tuple[int, ...]]],
    manifest: Mapping[str, object],
    num_trials_per_task: int,
    server_log: Path,
    harness_log: Path,
    host: str,
    port: int,
    server_ready_timeout_s: float,
    client_timeout_s: float,
    episode_count: int,
    runtime_indicator_config: Any,
    prompt_surface_bundle: Any,
    observed_runtime_prompting: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return build_rollout_input_summary_v21_model(
        schema_version=ROLLOUT_INPUT_SUMMARY_SCHEMA_VERSION,
        variant=variant,
        checkpoint_ref=checkpoint_ref,
        serve_checkpoint_ref=serve_checkpoint_ref,
        serve_checkpoint_mode=serve_checkpoint_mode,
        task_suite_name=task_suite_name,
        task_seed_manifests=task_seed_manifests,
        manifest=manifest,
        num_trials_per_task=num_trials_per_task,
        server_spec=RuntimeServerSpec(
            host=host,
            port=port,
            checkpoint_dir=checkpoint_ref,
            server_ready_timeout_s=server_ready_timeout_s,
            client_timeout_s=client_timeout_s,
        ),
        server_log=server_log,
        harness_log=harness_log,
        episode_count=episode_count,
        runtime_indicator_config=runtime_indicator_config,
        prompt_surface_bundle=prompt_surface_bundle,
        observed_runtime_prompting=observed_runtime_prompting,
        key_files=CHECKPOINT_INSTANCE_BINDING_KEY_FILES,
        binding_schema_version=CHECKPOINT_INSTANCE_BINDING_SCHEMA_VERSION,
        runtime_spec_schema_version=EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
    )


def build_rollout_input_summary_v21(
    *,
    variant: str,
    checkpoint_ref: str,
    serve_checkpoint_ref: str,
    serve_checkpoint_mode: str,
    task_suite_name: str,
    task_seed_manifests: Sequence[tuple[int, tuple[int, ...]]],
    manifest: Mapping[str, object],
    num_trials_per_task: int,
    server_log: Path,
    harness_log: Path,
    host: str,
    port: int,
    server_ready_timeout_s: float = 150.0,
    client_timeout_s: float = 80.0,
    episode_count: int,
    runtime_indicator_config: Any,
    prompt_surface_bundle: Any,
    observed_runtime_prompting: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return _build_rollout_input_summary(
        variant=variant,
        checkpoint_ref=checkpoint_ref,
        serve_checkpoint_ref=serve_checkpoint_ref,
        serve_checkpoint_mode=serve_checkpoint_mode,
        task_suite_name=task_suite_name,
        task_seed_manifests=task_seed_manifests,
        manifest=manifest,
        num_trials_per_task=num_trials_per_task,
        server_log=server_log,
        harness_log=harness_log,
        host=host,
        port=port,
        server_ready_timeout_s=server_ready_timeout_s,
        client_timeout_s=client_timeout_s,
        episode_count=episode_count,
        runtime_indicator_config=runtime_indicator_config,
        prompt_surface_bundle=prompt_surface_bundle,
        observed_runtime_prompting=observed_runtime_prompting,
    )


def _expected_runtime_prompting_payload(
    *,
    runtime_indicator_config: Any,
    prompt_surface_bundle: Any,
) -> dict[str, str]:
    return build_expected_runtime_prompting_payload(
        runtime_indicator_config=runtime_indicator_config,
        prompt_surface_bundle=prompt_surface_bundle,
    )


def build_effective_runtime_spec(
    *,
    variant: str,
    checkpoint_ref: str,
    runtime_indicator_config: Any,
    prompt_surface_bundle: Any,
) -> dict[str, str]:
    return build_effective_runtime_spec_model(
        variant=variant,
        checkpoint_ref=checkpoint_ref,
        runtime_indicator_config=runtime_indicator_config,
        prompt_surface_bundle=prompt_surface_bundle,
        key_files=CHECKPOINT_INSTANCE_BINDING_KEY_FILES,
        binding_schema_version=CHECKPOINT_INSTANCE_BINDING_SCHEMA_VERSION,
        runtime_spec_schema_version=EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
    )


def _effective_runtime_spec_without_request(
    payload: Mapping[str, object],
) -> dict[str, str]:
    try:
        return effective_runtime_spec_without_request_model(
            payload,
            runtime_spec_schema_version=EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
        )
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def _effective_runtime_surface_signature(
    payload: Mapping[str, object],
) -> dict[str, str]:
    try:
        return effective_runtime_surface_signature_model(
            payload,
            runtime_spec_schema_version=EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
        )
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def _normalize_effective_runtime_spec(
    payload: Mapping[str, object],
    *,
    context: str,
) -> dict[str, str]:
    try:
        return normalize_effective_runtime_spec_model(
            payload,
            context=context,
            runtime_spec_schema_version=EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
        )
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def effective_runtime_spec_hash(payload: Mapping[str, object]) -> str:
    return effective_runtime_spec_hash_model(payload)


def _effective_runtime_spec_from_runtime_prompting(
    runtime_prompting: Mapping[str, object],
    *,
    variant: str,
    checkpoint_ref: str,
    context: str,
) -> dict[str, str]:
    try:
        return effective_runtime_spec_from_runtime_prompting_model(
            runtime_prompting,
            variant=variant,
            checkpoint_ref=checkpoint_ref,
            key_files=CHECKPOINT_INSTANCE_BINDING_KEY_FILES,
            binding_schema_version=CHECKPOINT_INSTANCE_BINDING_SCHEMA_VERSION,
            runtime_spec_schema_version=EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION,
            context=context,
        )
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def _normalize_runtime_prompting_payload(
    payload: Mapping[str, object],
    *,
    context: str,
) -> dict[str, str]:
    try:
        return normalize_runtime_prompting_payload_model(payload, context=context)
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def _resolve_runtime_prompting_payload(
    *,
    runtime_indicator_config: Any,
    prompt_surface_bundle: Any,
    observed_runtime_prompting: Mapping[str, object] | None,
    context: str,
) -> dict[str, str]:
    try:
        return resolve_runtime_prompting_payload_model(
            runtime_indicator_config=runtime_indicator_config,
            prompt_surface_bundle=prompt_surface_bundle,
            observed_runtime_prompting=observed_runtime_prompting,
            context=context,
        )
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def _observed_runtime_prompting_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    context: str,
) -> dict[str, str]:
    try:
        return observed_runtime_prompting_from_rows_model(rows, context=context)
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def _materialized_rollout_source_has_runtime_mismatch(
    *,
    source_dir: Path,
    runtime_indicator_config: Any,
    prompt_surface_bundle: Any,
    log_path: Path,
) -> bool:
    input_path = source_dir / V2_INPUT_PER_EPISODE_NAME
    if not input_path.is_file():
        return False
    summary_path = source_dir / ROLLOUT_INPUT_SUMMARY_NAME
    expected_runtime_prompting = _expected_runtime_prompting_payload(
        runtime_indicator_config=runtime_indicator_config,
        prompt_surface_bundle=prompt_surface_bundle,
    )
    expected_effective_runtime_spec = build_effective_runtime_spec(
        variant="runtime_prompting_source",
        checkpoint_ref="runtime_prompting_source",
        runtime_indicator_config=runtime_indicator_config,
        prompt_surface_bundle=prompt_surface_bundle,
    )
    try:
        observed_rows = _observed_runtime_prompting_from_rows(
            _read_jsonl(input_path),
            context="materialized_rollout_source.rows",
        )
    except (FailFastError, ValueError) as exc:
        if not summary_path.is_file():
            _log(
                "legacy rollout source lacks runtime_prompting metadata; skipping runtime mismatch check",
                log_path=log_path,
            )
            return False
        _log(
            f"runtime_prompting mismatch requires rematerialization: {exc}",
            log_path=log_path,
        )
        return True
    observed_effective_runtime_spec = _effective_runtime_spec_from_runtime_prompting(
        observed_rows,
        variant="runtime_prompting_source",
        checkpoint_ref="runtime_prompting_source",
        context="materialized_rollout_source.rows",
    )
    if not summary_path.is_file():
        _log(
            "missing rollout_input_summary.json for complete rollout source; rematerializing",
            log_path=log_path,
        )
        return True
    try:
        rollout_input_summary = _read_json(summary_path)
        summary_runtime_prompting = _normalize_runtime_prompting_payload(
            _require_mapping(
                rollout_input_summary.get("runtime_prompting"),
                context="rollout_input_summary.runtime_prompting",
            ),
            context="rollout_input_summary.runtime_prompting",
        )
        raw_effective_runtime_spec = rollout_input_summary.get("effective_runtime_spec")
        if isinstance(raw_effective_runtime_spec, Mapping):
            summary_effective_runtime_spec = _normalize_effective_runtime_spec(
                cast(Mapping[str, object], raw_effective_runtime_spec),
                context="rollout_input_summary.effective_runtime_spec",
            )
        else:
            summary_effective_runtime_spec = (
                _effective_runtime_spec_from_runtime_prompting(
                    summary_runtime_prompting,
                    variant="runtime_prompting_source",
                    checkpoint_ref="runtime_prompting_source",
                    context="rollout_input_summary.runtime_prompting",
                )
            )
    except (FailFastError, ValueError) as exc:
        _log(
            f"invalid rollout_input_summary runtime_prompting requires rematerialization: {exc}",
            log_path=log_path,
        )
        return True
    if summary_runtime_prompting != observed_rows:
        _log(
            "rollout_input_summary runtime_prompting disagrees with episode rows; rematerializing",
            log_path=log_path,
        )
        return True
    if _effective_runtime_surface_signature(
        summary_effective_runtime_spec
    ) != _effective_runtime_surface_signature(observed_effective_runtime_spec):
        _log(
            "rollout_input_summary effective_runtime_spec disagrees with episode rows; rematerializing",
            log_path=log_path,
        )
        return True
    if _effective_runtime_surface_signature(
        expected_effective_runtime_spec
    ) != _effective_runtime_surface_signature(summary_effective_runtime_spec):
        if summary_runtime_prompting != expected_runtime_prompting:
            _log(
                "runtime_prompting mismatch requires rematerialization because effective runtime spec also changed",
                log_path=log_path,
            )
        else:
            _log(
                "effective_runtime_spec mismatch between expected resolved config and materialized source; rematerializing",
                log_path=log_path,
            )
        return True
    return False


def _reset_materialized_rollout_source(source_dir: Path, *, log_path: Path) -> None:
    if not source_dir.exists():
        return
    for file_name in (V2_INPUT_PER_EPISODE_NAME, ROLLOUT_INPUT_SUMMARY_NAME):
        file_path = source_dir / file_name
        if file_path.exists():
            file_path.unlink()
    videos_dir = source_dir / "videos"
    if videos_dir.is_dir():
        shutil.rmtree(videos_dir)
    _log(
        f"cleared stale rollout source for rematerialization: {source_dir}",
        log_path=log_path,
    )


def _materialize_checkpoint_rollout_source_v21(
    *,
    checkpoint_ref: str,
    checkpoint_mode: str,
    raw_checkpoint_dir: str | None,
    variant: str,
    eval_manifest: Mapping[str, object],
    source_dir: Path,
    runtime_dir: Path,
    log_path: Path,
    indicator_mode_requested: str,
    runtime_indicator_config: RuntimeIndicatorConfig,
    prompt_surface_bundle: Any,
    runtime_host: str | None = None,
    runtime_port: int | None = None,
    server_ready_timeout_s: float | None = None,
    client_timeout_s: float | None = None,
) -> Path:
    manifest = _normalize_eval_manifest_payload(eval_manifest)
    task_suite_name = _require_non_empty_str(
        manifest.get("task_suite_name"), context="eval_manifest.task_suite_name"
    )
    task_seed_manifests = _resolve_task_seed_manifests(
        eval_manifest,
        context="eval_manifest",
    )
    num_trials_per_task = _coerce_int(
        manifest.get("num_trials_per_task"), context="eval_manifest.num_trials_per_task"
    )
    expected_keys = _expected_episode_keys(
        task_seed_manifests=task_seed_manifests,
        num_trials_per_task=num_trials_per_task,
    )
    paths = RuntimePathsBuilder(topic=TOPIC).build()
    host = str(runtime_host or DEFAULT_HOST)
    port = pick_free_port(
        host,
        int(runtime_port if runtime_port is not None else DEFAULT_PORT),
    )
    resolved_server_ready_timeout_s = float(
        server_ready_timeout_s if server_ready_timeout_s is not None else 150.0
    )
    resolved_client_timeout_s = float(
        client_timeout_s if client_timeout_s is not None else 80.0
    )
    serve_checkpoint_ref = checkpoint_ref
    serve_checkpoint_mode = "explicit_stock_source"
    if checkpoint_mode != "stock":
        serve_checkpoint_ref, serve_checkpoint_mode = (
            rollout_v2._resolve_servable_checkpoint_ref(
                checkpoint_ref=checkpoint_ref,
                variant=variant,
            )
        )
    source_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = _load_resumable_rollout_rows(
        source_dir=source_dir,
        expected_keys=expected_keys,
        log_path=log_path,
    )
    rows: list[dict[str, object]] = [dict(row) for row in existing_rows]
    completed_keys = {
        _episode_key_from_row(row, context=f"existing_rows[{index}]")
        for index, row in enumerate(existing_rows)
    }
    if len(completed_keys) == len(expected_keys):
        _log(
            f"rollout source already complete: {source_dir}",
            log_path=log_path,
        )
        return source_dir
    bridge_runtime_dir = runtime_dir / "native_bridge"
    bridge_runtime_dir.mkdir(parents=True, exist_ok=True)
    harness_log = bridge_runtime_dir / "harness.log"
    server_log = bridge_runtime_dir / "server.log"
    libero_config_dir = prepare_libero_config_dir(
        paths.openpi_root,
        bridge_runtime_dir,
    )
    server_spec = RuntimeServerSpec(
        host=host,
        port=port,
        checkpoint_dir=serve_checkpoint_ref,
        server_ready_timeout_s=resolved_server_ready_timeout_s,
        client_timeout_s=resolved_client_timeout_s,
    )
    proc: Any | None = None
    server_handle: Any | None = None
    try:
        _log(
            "materializing fresh rollout source "
            + f"variant={variant} requested_checkpoint_ref={checkpoint_ref} "
            + f"serve_checkpoint_ref={serve_checkpoint_ref} serve_checkpoint_mode={serve_checkpoint_mode}",
            log_path=log_path,
        )
        server = PolicyServerProcess(
            spec=server_spec,
            venv_python=paths.openpi_venv_python,
            serve_policy=paths.serve_policy,
            openpi_root=paths.openpi_root,
            server_log=server_log,
            libero_config_dir=libero_config_dir,
            cli_entry=Path(LIBERO_NATIVE_SMOKE_ENTRY),
        )
        proc, server_handle = server.start()
        probe_payload = server.wait_until_ready(
            runtime_dir=bridge_runtime_dir,
            harness_log=harness_log,
        )
        _log(
            "native rollout bridge ready "
            + f"variant={variant} host={host} port={port} probe_at={probe_payload.get('probed_at', '')}",
            log_path=log_path,
        )
        for task_id, seed_manifest in task_seed_manifests:
            for seed in seed_manifest:
                for trial_index in range(num_trials_per_task):
                    episode_key = (int(task_id), int(seed), int(trial_index))
                    if episode_key in completed_keys:
                        _log(
                            "reusing materialized rollout "
                            + f"task_id={task_id} seed={seed} trial_index={trial_index}",
                            log_path=log_path,
                        )
                        continue
                    video_path = (
                        source_dir
                        / "videos"
                        / f"task{task_id}_seed{seed}_trial{trial_index}.mp4"
                    )
                    _log(
                        "running fresh rollout "
                        + f"variant={variant} task_id={task_id} seed={seed} trial_index={trial_index}",
                        log_path=log_path,
                    )
                    row = _run_runtime_episode_subprocess(
                        task_suite_name=task_suite_name,
                        task_id=task_id,
                        seed=seed,
                        trial_idx=trial_index,
                        video_path=video_path,
                        host=host,
                        port=port,
                        venv_python=paths.openpi_venv_python,
                        openpi_root=paths.openpi_root,
                        libero_config_dir=libero_config_dir,
                        runtime_dir=bridge_runtime_dir,
                        timeout_s=float(server_spec.client_timeout_s),
                        checkpoint_ref=serve_checkpoint_ref,
                        indicator_mode_requested=runtime_indicator_config.requested_indicator_mode,
                        runtime_indicator_config=runtime_indicator_config,
                    )
                    rows.append(row)
                    completed_keys.add(episode_key)
                    _write_jsonl(source_dir / V2_INPUT_PER_EPISODE_NAME, rows)
                    observed_runtime_prompting = _observed_runtime_prompting_from_rows(
                        rows,
                        context="materialized_rollout_source.rows",
                    )
                    _write_json(
                        source_dir / ROLLOUT_INPUT_SUMMARY_NAME,
                        _build_rollout_input_summary(
                            variant=variant,
                            checkpoint_ref=checkpoint_ref,
                            serve_checkpoint_ref=serve_checkpoint_ref,
                            serve_checkpoint_mode=serve_checkpoint_mode,
                            task_suite_name=task_suite_name,
                            task_seed_manifests=task_seed_manifests,
                            manifest=manifest,
                            num_trials_per_task=num_trials_per_task,
                            server_log=server_log,
                            harness_log=harness_log,
                            host=host,
                            port=port,
                            server_ready_timeout_s=resolved_server_ready_timeout_s,
                            client_timeout_s=resolved_client_timeout_s,
                            episode_count=len(rows),
                            runtime_indicator_config=runtime_indicator_config,
                            prompt_surface_bundle=prompt_surface_bundle,
                            observed_runtime_prompting=observed_runtime_prompting,
                        ),
                    )
        _write_jsonl(source_dir / V2_INPUT_PER_EPISODE_NAME, rows)
        observed_runtime_prompting = _observed_runtime_prompting_from_rows(
            rows,
            context="materialized_rollout_source.rows",
        )
        _write_json(
            source_dir / ROLLOUT_INPUT_SUMMARY_NAME,
            _build_rollout_input_summary(
                variant=variant,
                checkpoint_ref=checkpoint_ref,
                serve_checkpoint_ref=serve_checkpoint_ref,
                serve_checkpoint_mode=serve_checkpoint_mode,
                task_suite_name=task_suite_name,
                task_seed_manifests=task_seed_manifests,
                manifest=manifest,
                num_trials_per_task=num_trials_per_task,
                server_log=server_log,
                harness_log=harness_log,
                host=host,
                port=port,
                server_ready_timeout_s=resolved_server_ready_timeout_s,
                client_timeout_s=resolved_client_timeout_s,
                episode_count=len(rows),
                runtime_indicator_config=runtime_indicator_config,
                prompt_surface_bundle=prompt_surface_bundle,
                observed_runtime_prompting=observed_runtime_prompting,
            ),
        )
        return source_dir
    finally:
        RuntimeCleanup.close_process(proc)
        RuntimeCleanup.close_handle(server_handle)


def _ensure_rollout_source_dir(
    *,
    checkpoint_ref: str,
    checkpoint_mode: str,
    raw_checkpoint_dir: str | None,
    indicator_mode_requested: str,
    variant: str,
    eval_manifest: Mapping[str, object],
    output_dir: Path,
    runtime_dir: Path,
    log_path: Path,
    runtime_indicator_config: RuntimeIndicatorConfig,
    prompt_surface_bundle: Any,
    canonical_source_dir: Path | None = None,
    runtime_host: str | None = None,
    runtime_port: int | None = None,
    server_ready_timeout_s: float | None = None,
    client_timeout_s: float | None = None,
) -> Path:
    if canonical_source_dir is not None:
        source_dir = canonical_source_dir.resolve()
    else:
        source_dir = output_dir / "_staging"
        try:
            source_dir = _resolve_rollout_source_dir(
                checkpoint_ref=checkpoint_ref,
                raw_checkpoint_dir=raw_checkpoint_dir,
                output_dir=output_dir,
            )
        except FailFastError:
            pass
        else:
            if _materialized_rollout_source_has_runtime_mismatch(
                source_dir=source_dir,
                runtime_indicator_config=runtime_indicator_config,
                prompt_surface_bundle=prompt_surface_bundle,
                log_path=log_path,
            ):
                _reset_materialized_rollout_source(source_dir, log_path=log_path)
            if _is_materialized_rollout_source_complete(
                source_dir=source_dir,
                eval_manifest=eval_manifest,
                log_path=log_path,
            ):
                return source_dir
    if canonical_source_dir is not None:
        if _materialized_rollout_source_has_runtime_mismatch(
            source_dir=source_dir,
            runtime_indicator_config=runtime_indicator_config,
            prompt_surface_bundle=prompt_surface_bundle,
            log_path=log_path,
        ):
            _reset_materialized_rollout_source(source_dir, log_path=log_path)
        if _is_materialized_rollout_source_complete(
            source_dir=source_dir,
            eval_manifest=eval_manifest,
            log_path=log_path,
        ):
            return source_dir
    _log(
        "missing or incomplete pre-materialized per_episode_rollouts.jsonl; materializing/resuming fresh rollouts now",
        log_path=log_path,
    )
    return _materialize_checkpoint_rollout_source_v21(
        checkpoint_ref=checkpoint_ref,
        checkpoint_mode=checkpoint_mode,
        raw_checkpoint_dir=raw_checkpoint_dir,
        variant=variant,
        eval_manifest=eval_manifest,
        source_dir=source_dir,
        runtime_dir=runtime_dir,
        log_path=log_path,
        indicator_mode_requested=indicator_mode_requested,
        runtime_indicator_config=runtime_indicator_config,
        prompt_surface_bundle=prompt_surface_bundle,
        runtime_host=runtime_host,
        runtime_port=runtime_port,
        server_ready_timeout_s=server_ready_timeout_s,
        client_timeout_s=client_timeout_s,
    )


def _runtime_indicator_config_from_args(
    *,
    args: argparse.Namespace,
    variant: str,
    train_manifest: Mapping[str, object] | None,
    checkpoint_provenance: Mapping[str, object] | None,
) -> RuntimeIndicatorConfig:
    explicit_indicator_mode = str(
        getattr(args, "resolved_runtime_indicator_mode", "") or ""
    ).strip()
    if not explicit_indicator_mode:
        return resolve_runtime_indicator_config(
            requested_indicator_mode=str(args.indicator_mode),
            variant=variant,
            train_manifest=train_manifest,
            checkpoint_provenance=checkpoint_provenance,
        )

    explicit_indicator_source = str(
        getattr(args, "resolved_runtime_indicator_source", "") or ""
    ).strip()
    explicit_consumer_mode = str(
        getattr(args, "resolved_runtime_consumer_mode", "") or ""
    ).strip()
    explicit_critic_checkpoint_ref = str(
        getattr(args, "resolved_runtime_critic_checkpoint_ref", "") or ""
    ).strip()
    missing_required = [
        field_name
        for field_name, value in (
            ("--resolved-runtime-indicator-source", explicit_indicator_source),
            ("--resolved-runtime-consumer-mode", explicit_consumer_mode),
            (
                "--resolved-runtime-critic-checkpoint-ref",
                explicit_critic_checkpoint_ref,
            ),
        )
        if not value
    ]
    if missing_required:
        raise FailFastError(
            "explicit resolved runtime config is incomplete; missing "
            + ", ".join(missing_required)
        )
    fixed_indicator_mode_text = str(
        getattr(args, "resolved_runtime_fixed_indicator_mode", "") or ""
    ).strip()
    return RuntimeIndicatorConfig(
        requested_indicator_mode=str(args.indicator_mode),
        indicator_mode=explicit_indicator_mode,
        indicator_source=explicit_indicator_source,
        consumer_mode=explicit_consumer_mode,
        fixed_indicator_mode=fixed_indicator_mode_text or None,
        critic_checkpoint_ref=explicit_critic_checkpoint_ref,
    )


def _episode_sort_key(row: Mapping[str, object]) -> tuple[int, int, int]:
    return (
        _coerce_int(row.get("task_id"), context="episode.task_id"),
        _coerce_int(row.get("seed"), context="episode.seed"),
        _coerce_int(row.get("trial_idx"), context="episode.trial_idx"),
    )


def _resolve_max_steps_resolved(
    row: Mapping[str, object], *, task_suite_name: str, context: str
) -> int:
    raw_value = row.get("max_steps_resolved")
    if raw_value is not None:
        value = _coerce_int(raw_value, context=f"{context}.max_steps_resolved")
    else:
        value = int(max_steps_for_task_suite(task_suite_name))
    if value <= 0:
        raise FailFastError(f"{context}.max_steps_resolved must be positive")
    return value


def _resolve_executed_steps(
    row: Mapping[str, object], *, max_steps_resolved: int, context: str
) -> int:
    raw_value = row.get("executed_steps")
    if raw_value is not None:
        executed_steps = _coerce_int(raw_value, context=f"{context}.executed_steps")
    else:
        steps_observed = _coerce_int(
            row.get("steps_observed", row.get("steps", 0)),
            context=f"{context}.steps_observed",
        )
        executed_steps = max(steps_observed - NUM_STEPS_WAIT, 0)
    if executed_steps < 0:
        raise FailFastError(f"{context}.executed_steps must be non-negative")
    return min(executed_steps, max_steps_resolved)


def _resolve_first_success_step(
    row: Mapping[str, object],
    *,
    success: bool,
    executed_steps: int,
    max_steps_resolved: int,
    context: str,
) -> int | None:
    raw_value = row.get("first_success_step")
    if raw_value in {None, "", "null"}:
        if not success:
            return None
        return max(1, executed_steps)
    first_success_step = _coerce_int(raw_value, context=f"{context}.first_success_step")
    if not success:
        raise FailFastError(
            f"{context}.first_success_step cannot be set when success=false"
        )
    if first_success_step < 1 or first_success_step > max_steps_resolved:
        raise FailFastError(
            f"{context}.first_success_step must be within [1, max_steps_resolved]"
        )
    return first_success_step


def _resolve_timeout_flag(
    row: Mapping[str, object],
    *,
    success: bool,
    executed_steps: int,
    max_steps_resolved: int,
    context: str,
) -> bool:
    raw_value = row.get("timeout_flag")
    if raw_value is not None:
        return _coerce_bool(raw_value, context=f"{context}.timeout_flag")
    return (not success) and executed_steps >= max_steps_resolved


def _resolve_row_deviation_notes(
    row: Mapping[str, object], *, context: str
) -> list[str]:
    notes = _coerce_string_list(row.get("deviation_notes"), context=context)
    episode_status = str(row.get("episode_status", "")).strip()
    error = str(row.get("error", "")).strip()
    if episode_status and episode_status != "ok":
        notes.append(f"episode_status={episode_status}")
    if error:
        notes.append(f"error={error}")
    return notes


def _validate_and_build_trace_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    eval_manifest: Mapping[str, object],
    variant: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest = _normalize_eval_manifest_payload(eval_manifest)
    task_suite_name = str(manifest["task_suite_name"])
    task_seed_manifests = _resolve_task_seed_manifests(
        eval_manifest,
        context="eval_manifest",
    )
    num_trials_per_task = _coerce_int(
        manifest.get("num_trials_per_task"), context="eval_manifest.num_trials_per_task"
    )
    expected_keys = {
        (task_id, seed, trial_idx)
        for task_id, seed_manifest in task_seed_manifests
        for seed in seed_manifest
        for trial_idx in range(num_trials_per_task)
    }
    seen: set[tuple[int, int, int]] = set()
    trace_rows: list[dict[str, object]] = []
    duplicate_episode_ids: list[str] = []
    unexpected_keys: list[str] = []

    for index, raw_row in enumerate(rows):
        mapping = _require_mapping(raw_row, context=f"per_episode_rollouts[{index}]")
        row_task_suite_name = str(
            mapping.get("task_suite_name", mapping.get("suite", task_suite_name))
        ).strip()
        if row_task_suite_name != task_suite_name:
            raise FailFastError(
                f"episode[{index}] task_suite_name mismatch; expected {task_suite_name!r}, got {row_task_suite_name!r}"
            )
        task_id = _coerce_int(
            mapping.get("task_id"), context=f"episode[{index}].task_id"
        )
        seed = _coerce_int(mapping.get("seed"), context=f"episode[{index}].seed")
        trial_idx = _coerce_int(
            mapping.get("trial_idx", mapping.get("trial_index")),
            context=f"episode[{index}].trial_idx",
        )
        success = _coerce_bool(
            mapping.get("success"), context=f"episode[{index}].success"
        )
        key = (task_id, seed, trial_idx)
        episode_id = f"task{task_id}_seed{seed}_trial{trial_idx}"
        if key in seen:
            duplicate_episode_ids.append(episode_id)
            continue
        seen.add(key)
        if key not in expected_keys:
            unexpected_keys.append(episode_id)
        max_steps_resolved = _resolve_max_steps_resolved(
            mapping,
            task_suite_name=task_suite_name,
            context=f"episode[{index}]",
        )
        executed_steps = _resolve_executed_steps(
            mapping,
            max_steps_resolved=max_steps_resolved,
            context=f"episode[{index}]",
        )
        first_success_step = _resolve_first_success_step(
            mapping,
            success=success,
            executed_steps=executed_steps,
            max_steps_resolved=max_steps_resolved,
            context=f"episode[{index}]",
        )
        success_within_50pct_budget = bool(
            first_success_step is not None
            and first_success_step <= math.floor(0.50 * max_steps_resolved)
        )
        success_within_75pct_budget = bool(
            first_success_step is not None
            and first_success_step <= math.floor(0.75 * max_steps_resolved)
        )
        timeout_flag = _resolve_timeout_flag(
            mapping,
            success=success,
            executed_steps=executed_steps,
            max_steps_resolved=max_steps_resolved,
            context=f"episode[{index}]",
        )
        trace_row: dict[str, object] = {
            "variant": variant,
            "task_id": task_id,
            "seed": seed,
            "trial_idx": trial_idx,
            "success": success,
            "first_success_step": first_success_step,
            "executed_steps": executed_steps,
            "max_steps_resolved": max_steps_resolved,
            "success_within_50pct_budget": success_within_50pct_budget,
            "success_within_75pct_budget": success_within_75pct_budget,
            "timeout_flag": timeout_flag,
            "deviation_notes": _resolve_row_deviation_notes(
                mapping, context=f"episode[{index}].deviation_notes"
            ),
        }
        if tuple(trace_row.keys()) != TRACE_REQUIRED_FIELDS:
            raise FailFastError(
                f"episode[{index}] trace field order drifted; expected {TRACE_REQUIRED_FIELDS!r}"
            )
        trace_rows.append(trace_row)

    if duplicate_episode_ids:
        raise FailFastError(
            "duplicate rollout episodes detected: "
            + ", ".join(sorted(duplicate_episode_ids))
        )
    if unexpected_keys:
        raise FailFastError(
            "rollout scope drift: unexpected episodes "
            + ", ".join(sorted(unexpected_keys))
        )
    missing_keys = sorted(expected_keys.difference(seen))
    if missing_keys:
        missing_episode_ids = [
            f"task{task_id}_seed{seed}_trial{trial_idx}"
            for task_id, seed, trial_idx in missing_keys
        ]
        raise FailFastError(
            "rollout scope incomplete: missing episodes "
            + ", ".join(missing_episode_ids)
        )

    trace_rows.sort(key=_episode_sort_key)
    episode_count = len(trace_rows)
    success_count = sum(1 for row in trace_rows if bool(row["success"]))
    timeout_count = sum(1 for row in trace_rows if bool(row["timeout_flag"]))
    scope_audit: dict[str, object] = {
        "task_suite_name": task_suite_name,
        "expected_episode_count": len(expected_keys),
        "observed_episode_count": episode_count,
        "scope_complete": episode_count == len(expected_keys),
        "unique_episode_count": len(seen),
        "duplicate_episode_count": 0,
        "unexpected_episode_count": 0,
        "missing_episode_count": 0,
        "success_count": success_count,
        "failure_count": episode_count - success_count,
        "timeout_count": timeout_count,
    }
    return trace_rows, scope_audit


def _metric_point_estimates(
    trace_rows: Sequence[Mapping[str, object]],
) -> dict[str, float | None]:
    try:
        return metric_point_estimates_from_trace_rows_v21(trace_rows)
    except AggregationValidationError as exc:
        raise FailFastError(str(exc)) from exc


def _build_metric_ladder_summary(
    *,
    trace_rows: Sequence[Mapping[str, object]],
    eval_manifest: Mapping[str, object],
    variant: str,
    checkpoint_ref: str,
) -> dict[str, object]:
    manifest = _normalize_eval_manifest_payload(eval_manifest)
    try:
        return build_metric_ladder_summary_v21(
            trace_rows=trace_rows,
            authority_id=str(manifest["authority_id"]),
            variant=variant,
            checkpoint_ref=checkpoint_ref,
            metric_profile=str(manifest["metric_profile"]),
            primary_metric_id=PRIMARY_METRIC_ID,
        )
    except AggregationValidationError as exc:
        raise FailFastError(str(exc)) from exc


def _bootstrap_ci_for_trace_rows(
    *,
    trace_rows: Sequence[Mapping[str, object]],
    eval_manifest_hash: str,
    variant: str,
) -> dict[str, object]:
    seed_material = f"{variant}:{eval_manifest_hash}"
    try:
        return build_bootstrap_ci_v21(
            trace_rows=trace_rows,
            deterministic_seed_material=seed_material,
            variant=variant,
            bootstrap_iterations=DEFAULT_BOOTSTRAP_ITERATIONS,
            confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        )
    except AggregationValidationError as exc:
        raise FailFastError(str(exc)) from exc


def _build_pairwise_delta(
    *,
    trace_rows: Sequence[Mapping[str, object]],
    variant: str,
    checkpoint_ref: str,
) -> dict[str, object]:
    point_estimates = _metric_point_estimates(trace_rows)
    metric_deltas = {
        metric_id: {
            "current": point_estimates[metric_id],
            "baseline": point_estimates[metric_id],
            "delta": 0.0,
        }
        for metric_id in (
            PRIMARY_METRIC_ID,
            "success_rate@0.75_budget",
            "success_rate@1.00_budget",
            "timeout_rate",
            "throughput_like_score",
            "median_first_success_step_fraction",
        )
    }
    return {
        "schema_version": PAIRWISE_DELTA_SCHEMA_VERSION,
        "variant": variant,
        "baseline_variant": variant,
        "checkpoint_ref": checkpoint_ref,
        "comparison_mode": "self_baseline_task3_minimum",
        "note": "Task 3 only requires a variant-local bundle; formal cross-variant deltas are deferred.",
        "metrics": metric_deltas,
    }


def _build_deviation_notes(
    *,
    variant: str,
    checkpoint_mode: str,
    checkpoint_ref: str,
    source_dir: Path,
    trace_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    notes = [
        "v21 is eval-only authority lane; this bundle stays sibling to v2 outputs.",
        "trace fields are observational only; checkpoint loading, action generation, and rollout termination semantics remain unchanged.",
        f"fresh rollout source_dir={source_dir}",
        "pairwise_delta.json is emitted in Task 3 minimum mode as a self-baseline placeholder.",
    ]
    if checkpoint_mode == "stock":
        notes.append(
            f"checkpoint-source=stock resolved to frozen stock anchor {checkpoint_ref} for variant={variant}."
        )
    deviation_row_count = sum(
        1
        for row in trace_rows
        if _coerce_string_list(
            row.get("deviation_notes"), context="trace.deviation_notes"
        )
    )
    if deviation_row_count:
        notes.append(f"trace rows with deviation notes: {deviation_row_count}")
    return notes


def _deviation_notes_markdown(notes: Sequence[str]) -> str:
    return (
        "\n".join(
            ["# libero_rollout_eval_v21 deviation notes", ""]
            + [f"- {note}" for note in notes]
        )
        + "\n"
    )


@dataclass(frozen=True)
class RequiredOutputPaths:
    per_episode_trace: Path
    metric_ladder_summary: Path
    bootstrap_ci: Path
    pairwise_delta: Path
    summary: Path
    eval_manifest: Path
    deviation_notes: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "RequiredOutputPaths":
        return cls(
            per_episode_trace=output_dir / TRACE_NAME,
            metric_ladder_summary=output_dir / METRIC_LADDER_NAME,
            bootstrap_ci=output_dir / BOOTSTRAP_NAME,
            pairwise_delta=output_dir / PAIRWISE_DELTA_NAME,
            summary=output_dir / SUMMARY_NAME,
            eval_manifest=output_dir / EVAL_MANIFEST_NAME,
            deviation_notes=output_dir / DEVIATION_NOTES_NAME,
        )

    def as_summary_mapping(self) -> dict[str, str]:
        return {
            "per_episode_trace": str(self.per_episode_trace),
            "metric_ladder_summary": str(self.metric_ladder_summary),
            "bootstrap_ci": str(self.bootstrap_ci),
            "pairwise_delta": str(self.pairwise_delta),
            "summary": str(self.summary),
            "eval_manifest": str(self.eval_manifest),
            "deviation_notes": str(self.deviation_notes),
        }


@dataclass(frozen=True)
class ResolvedCliInputs:
    variant: str
    eval_manifest: dict[str, object]
    manifest_metric_profile: str
    checkpoint_ref: str
    checkpoint_mode: str
    raw_checkpoint_dir: str | None
    output_dir: Path
    eval_manifest_hash: str
    eval_manifest_id: str
    runtime_dir: Path
    log_path: Path
    indicator_mode_requested: str
    canonical_source_dir: Path | None


@dataclass(frozen=True)
class RequestedRuntimeBinding:
    runtime_indicator_config: RuntimeIndicatorConfig
    prompt_surface_bundle: Any
    runtime_prompting: dict[str, str]
    effective_runtime_spec: dict[str, str]


@dataclass(frozen=True)
class ExecutedRuntimeBinding:
    runtime_prompting: dict[str, str]
    effective_runtime_spec: dict[str, str]


@dataclass(frozen=True)
class TraceBuildOutputs:
    trace_rows: list[dict[str, object]]
    scope_audit: dict[str, object]
    metric_ladder_summary: dict[str, object]
    bootstrap_ci: dict[str, object]
    pairwise_delta: dict[str, object]
    deviation_notes: list[str]


@dataclass(frozen=True)
class RolloutEvalV21Dependencies:
    build_runtime_dir: Callable[[str, str], Path]
    load_checkpoint_provenance_pair: Callable[
        [str, str | None], tuple[dict[str, object] | None, dict[str, object] | None]
    ]
    runtime_indicator_config_from_args: Callable[
        [
            argparse.Namespace,
            str,
            Mapping[str, object] | None,
            Mapping[str, object] | None,
        ],
        RuntimeIndicatorConfig,
    ]
    ensure_rollout_source_dir: Callable[
        [
            str,
            str,
            str | None,
            str,
            str,
            Mapping[str, object],
            Path,
            Path,
            Path,
            RuntimeIndicatorConfig,
            Any,
            Path | None,
            str | None,
            int | None,
            float | None,
            float | None,
        ],
        Path,
    ]
    validate_and_build_trace_rows: Callable[
        [Sequence[Mapping[str, object]], Mapping[str, object], str],
        tuple[list[dict[str, object]], dict[str, object]],
    ]


@dataclass
class RolloutEvalV21Workflow:
    args: argparse.Namespace
    dependencies: RolloutEvalV21Dependencies

    def run(self) -> int:
        inputs = self._resolve_cli_inputs()
        requested_runtime = self._resolve_requested_runtime_binding(inputs)
        source_dir = self._ensure_rollout_source_dir(inputs, requested_runtime)
        _log(f"source_rollout_dir={source_dir}", log_path=inputs.log_path)
        trace_outputs = self._build_trace_outputs(inputs, source_dir)
        executed_runtime = self._load_executed_runtime_binding(
            inputs,
            source_dir,
            requested_runtime,
        )
        required_outputs = RequiredOutputPaths.from_output_dir(inputs.output_dir)
        summary = self._build_summary(
            inputs=inputs,
            source_dir=source_dir,
            trace_outputs=trace_outputs,
            requested_runtime=requested_runtime,
            executed_runtime=executed_runtime,
            required_outputs=required_outputs,
        )
        self._write_authority_bundle(
            inputs=inputs,
            trace_outputs=trace_outputs,
            summary=summary,
            required_outputs=required_outputs,
        )
        return 0

    def _resolve_cli_inputs(self) -> ResolvedCliInputs:
        variant = _normalize_variant(str(self.args.variant))
        eval_manifest = _normalize_eval_manifest_payload(
            load_rollout_manifest_v21(str(self.args.manifest))
        )
        metric_profile = _require_non_empty_str(
            self.args.metric_profile,
            context="--metric-profile",
        )
        manifest_metric_profile = _require_non_empty_str(
            eval_manifest.get("metric_profile"),
            context="eval_manifest.metric_profile",
        )
        if metric_profile != manifest_metric_profile:
            raise FailFastError(
                "metric_profile mismatch: "
                + f"cli={metric_profile!r} manifest={manifest_metric_profile!r}"
            )
        variant_scope = {
            str(item).strip()
            for item in _require_sequence(
                eval_manifest.get("variant_scope"),
                context="eval_manifest.variant_scope",
            )
        }
        if variant not in variant_scope:
            raise FailFastError(
                f"variant {variant!r} is outside manifest variant_scope={sorted(variant_scope)!r}"
            )
        checkpoint_ref, checkpoint_mode = _resolve_checkpoint_input(self.args, variant)
        raw_checkpoint_dir = (
            str(getattr(self.args, "checkpoint_dir", "") or "").strip() or None
        )
        output_dir = Path(str(self.args.output_dir)).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_manifest_hash = compute_rollout_manifest_hash_v21(eval_manifest)
        eval_manifest_id = build_eval_manifest_id(eval_manifest)
        runtime_dir = self.dependencies.build_runtime_dir(variant, eval_manifest_id)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        log_path = runtime_dir / "eval.log"
        _log(
            f"[{dt.datetime.now().isoformat(timespec='seconds')}] rollout_eval_v21 variant={variant} manifest={eval_manifest_id}",
            log_path=log_path,
        )
        canonical_source_dir = (
            Path(str(getattr(self.args, "canonical_source_dir", "") or "")).resolve()
            if str(getattr(self.args, "canonical_source_dir", "") or "").strip()
            else None
        )
        return ResolvedCliInputs(
            variant=variant,
            eval_manifest=eval_manifest,
            manifest_metric_profile=manifest_metric_profile,
            checkpoint_ref=checkpoint_ref,
            checkpoint_mode=checkpoint_mode,
            raw_checkpoint_dir=raw_checkpoint_dir,
            output_dir=output_dir,
            eval_manifest_hash=eval_manifest_hash,
            eval_manifest_id=eval_manifest_id,
            runtime_dir=runtime_dir,
            log_path=log_path,
            indicator_mode_requested=str(self.args.indicator_mode),
            canonical_source_dir=canonical_source_dir,
        )

    def _resolve_requested_runtime_binding(
        self,
        inputs: ResolvedCliInputs,
    ) -> RequestedRuntimeBinding:
        train_manifest, checkpoint_provenance = (
            self.dependencies.load_checkpoint_provenance_pair(
                inputs.checkpoint_ref,
                inputs.raw_checkpoint_dir,
            )
        )
        runtime_indicator_config = self.dependencies.runtime_indicator_config_from_args(
            self.args,
            inputs.variant,
            train_manifest,
            checkpoint_provenance,
        )
        prompt_surface_bundle = build_runtime_prompt_bundle(
            "runtime prompt surface preview",
            config=runtime_indicator_config,
        )
        return RequestedRuntimeBinding(
            runtime_indicator_config=runtime_indicator_config,
            prompt_surface_bundle=prompt_surface_bundle,
            runtime_prompting=_expected_runtime_prompting_payload(
                runtime_indicator_config=runtime_indicator_config,
                prompt_surface_bundle=prompt_surface_bundle,
            ),
            effective_runtime_spec=build_effective_runtime_spec(
                variant=inputs.variant,
                checkpoint_ref=inputs.checkpoint_ref,
                runtime_indicator_config=runtime_indicator_config,
                prompt_surface_bundle=prompt_surface_bundle,
            ),
        )

    def _ensure_rollout_source_dir(
        self,
        inputs: ResolvedCliInputs,
        requested_runtime: RequestedRuntimeBinding,
    ) -> Path:
        return self.dependencies.ensure_rollout_source_dir(
            inputs.checkpoint_ref,
            inputs.checkpoint_mode,
            inputs.raw_checkpoint_dir,
            inputs.indicator_mode_requested,
            inputs.variant,
            inputs.eval_manifest,
            inputs.output_dir,
            inputs.runtime_dir,
            inputs.log_path,
            requested_runtime.runtime_indicator_config,
            requested_runtime.prompt_surface_bundle,
            inputs.canonical_source_dir,
            None,
            None,
            None,
            None,
        )

    def _build_trace_outputs(
        self,
        inputs: ResolvedCliInputs,
        source_dir: Path,
    ) -> TraceBuildOutputs:
        input_rows = _read_jsonl(source_dir / V2_INPUT_PER_EPISODE_NAME)
        trace_rows, scope_audit = self.dependencies.validate_and_build_trace_rows(
            input_rows,
            inputs.eval_manifest,
            inputs.variant,
        )
        metric_ladder_summary = _build_metric_ladder_summary(
            trace_rows=trace_rows,
            eval_manifest=inputs.eval_manifest,
            variant=inputs.variant,
            checkpoint_ref=inputs.checkpoint_ref,
        )
        bootstrap_ci = _bootstrap_ci_for_trace_rows(
            trace_rows=trace_rows,
            eval_manifest_hash=inputs.eval_manifest_hash,
            variant=inputs.variant,
        )
        pairwise_delta = _build_pairwise_delta(
            trace_rows=trace_rows,
            variant=inputs.variant,
            checkpoint_ref=inputs.checkpoint_ref,
        )
        deviation_notes = _build_deviation_notes(
            variant=inputs.variant,
            checkpoint_mode=inputs.checkpoint_mode,
            checkpoint_ref=inputs.checkpoint_ref,
            source_dir=source_dir,
            trace_rows=trace_rows,
        )
        return TraceBuildOutputs(
            trace_rows=trace_rows,
            scope_audit=scope_audit,
            metric_ladder_summary=metric_ladder_summary,
            bootstrap_ci=bootstrap_ci,
            pairwise_delta=pairwise_delta,
            deviation_notes=deviation_notes,
        )

    def _load_executed_runtime_binding(
        self,
        inputs: ResolvedCliInputs,
        source_dir: Path,
        requested_runtime: RequestedRuntimeBinding,
    ) -> ExecutedRuntimeBinding:
        source_rollout_input_summary_path = source_dir / ROLLOUT_INPUT_SUMMARY_NAME
        if not source_rollout_input_summary_path.is_file():
            return ExecutedRuntimeBinding(
                runtime_prompting=requested_runtime.runtime_prompting,
                effective_runtime_spec=requested_runtime.effective_runtime_spec,
            )
        source_rollout_input_summary = _read_json(source_rollout_input_summary_path)
        executed_runtime_prompting = _normalize_runtime_prompting_payload(
            _require_mapping(
                source_rollout_input_summary.get("runtime_prompting"),
                context="source_rollout_input_summary.runtime_prompting",
            ),
            context="source_rollout_input_summary.runtime_prompting",
        )
        raw_executed_effective_runtime_spec = source_rollout_input_summary.get(
            "effective_runtime_spec"
        )
        if isinstance(raw_executed_effective_runtime_spec, Mapping):
            executed_effective_runtime_spec = _normalize_effective_runtime_spec(
                cast(Mapping[str, object], raw_executed_effective_runtime_spec),
                context="source_rollout_input_summary.effective_runtime_spec",
            )
        else:
            executed_effective_runtime_spec = (
                _effective_runtime_spec_from_runtime_prompting(
                    executed_runtime_prompting,
                    variant=inputs.variant,
                    checkpoint_ref=inputs.checkpoint_ref,
                    context="source_rollout_input_summary.runtime_prompting",
                )
            )
        return ExecutedRuntimeBinding(
            runtime_prompting=executed_runtime_prompting,
            effective_runtime_spec=executed_effective_runtime_spec,
        )

    def _build_summary(
        self,
        *,
        inputs: ResolvedCliInputs,
        source_dir: Path,
        trace_outputs: TraceBuildOutputs,
        requested_runtime: RequestedRuntimeBinding,
        executed_runtime: ExecutedRuntimeBinding,
        required_outputs: RequiredOutputPaths,
    ) -> dict[str, object]:
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "authority_id": str(inputs.eval_manifest["authority_id"]),
            "variant": inputs.variant,
            "checkpoint_ref": inputs.checkpoint_ref,
            "checkpoint_mode": inputs.checkpoint_mode,
            "output_dir": str(inputs.output_dir),
            "runtime_dir": str(inputs.runtime_dir),
            "source_rollout_dir": str(source_dir),
            "eval_manifest_id": inputs.eval_manifest_id,
            "eval_manifest_hash": inputs.eval_manifest_hash,
            "manifest_name": str(inputs.eval_manifest["manifest_name"]),
            "task_suite_name": str(inputs.eval_manifest["task_suite_name"]),
            "metric_profile": inputs.manifest_metric_profile,
            "primary_metric_id": trace_outputs.metric_ladder_summary[
                "primary_metric_id"
            ],
            "headline_metric_order": list(HEADLINE_METRIC_ORDER),
            "compatibility_only_metrics": list(COMPATIBILITY_ONLY_METRICS),
            "scope_audit": trace_outputs.scope_audit,
            "metric_ladder_summary": trace_outputs.metric_ladder_summary,
            "runtime_prompting": executed_runtime.runtime_prompting,
            "requested_runtime_prompting": requested_runtime.runtime_prompting,
            "effective_runtime_spec": executed_runtime.effective_runtime_spec,
            "requested_effective_runtime_spec": requested_runtime.effective_runtime_spec,
            "rollout_source_binding": {
                "source_selection_mode": (
                    "explicit_canonical_source_dir"
                    if inputs.canonical_source_dir is not None
                    else "variant_output_staging"
                ),
                "requested_runtime_prompting_matches_executed": (
                    requested_runtime.runtime_prompting
                    == executed_runtime.runtime_prompting
                ),
                "effective_runtime_spec_matches_requested": (
                    _effective_runtime_surface_signature(
                        executed_runtime.effective_runtime_spec
                    )
                    == _effective_runtime_surface_signature(
                        requested_runtime.effective_runtime_spec
                    )
                ),
                "effective_runtime_spec_hash": effective_runtime_spec_hash(
                    executed_runtime.effective_runtime_spec
                ),
            },
            "required_outputs": required_outputs.as_summary_mapping(),
            "deviation_notes": list(trace_outputs.deviation_notes),
        }
        try:
            assert_variant_aggregate_conservation_v21(
                trace_rows=trace_outputs.trace_rows,
                metric_ladder_summary=trace_outputs.metric_ladder_summary,
                bootstrap_ci=trace_outputs.bootstrap_ci,
                summary=summary,
            )
        except AggregationValidationError as exc:
            raise FailFastError(str(exc)) from exc
        return summary

    def _write_authority_bundle(
        self,
        *,
        inputs: ResolvedCliInputs,
        trace_outputs: TraceBuildOutputs,
        summary: Mapping[str, object],
        required_outputs: RequiredOutputPaths,
    ) -> None:
        eval_manifest_payload = {
            **inputs.eval_manifest,
            "eval_manifest_hash": inputs.eval_manifest_hash,
            "eval_manifest_id": inputs.eval_manifest_id,
        }
        _write_json(required_outputs.eval_manifest, eval_manifest_payload)
        _write_jsonl(required_outputs.per_episode_trace, trace_outputs.trace_rows)
        _write_json(
            required_outputs.metric_ladder_summary,
            trace_outputs.metric_ladder_summary,
        )
        _write_json(required_outputs.bootstrap_ci, trace_outputs.bootstrap_ci)
        _write_json(required_outputs.pairwise_delta, trace_outputs.pairwise_delta)
        _write_markdown(
            required_outputs.deviation_notes,
            _deviation_notes_markdown(trace_outputs.deviation_notes),
        )
        _write_json(required_outputs.summary, summary)
        _log(f"summary_json={required_outputs.summary}", log_path=inputs.log_path)
        _log("LIBERO_ROLLOUT_EVAL_V21_DONE", log_path=inputs.log_path)


def run_rollout_eval_v21(
    args: argparse.Namespace,
    *,
    dependencies: RolloutEvalV21Dependencies,
) -> int:
    return RolloutEvalV21Workflow(args=args, dependencies=dependencies).run()


def _default_dependencies() -> RolloutEvalV21Dependencies:
    return RolloutEvalV21Dependencies(
        build_runtime_dir=_build_runtime_dir,
        load_checkpoint_provenance_pair=lambda checkpoint_ref,
        raw_checkpoint_dir: _load_checkpoint_provenance_pair(
            checkpoint_ref=checkpoint_ref,
            raw_checkpoint_dir=raw_checkpoint_dir,
        ),
        runtime_indicator_config_from_args=lambda args,
        variant,
        train_manifest,
        checkpoint_provenance: _runtime_indicator_config_from_args(
            args=args,
            variant=variant,
            train_manifest=train_manifest,
            checkpoint_provenance=checkpoint_provenance,
        ),
        ensure_rollout_source_dir=lambda checkpoint_ref,
        checkpoint_mode,
        raw_checkpoint_dir,
        indicator_mode_requested,
        variant,
        eval_manifest,
        output_dir,
        runtime_dir,
        log_path,
        runtime_indicator_config,
        prompt_surface_bundle,
        canonical_source_dir,
        runtime_host,
        runtime_port,
        server_ready_timeout_s,
        client_timeout_s: _ensure_rollout_source_dir(
            checkpoint_ref=checkpoint_ref,
            checkpoint_mode=checkpoint_mode,
            raw_checkpoint_dir=raw_checkpoint_dir,
            indicator_mode_requested=indicator_mode_requested,
            variant=variant,
            eval_manifest=eval_manifest,
            output_dir=output_dir,
            runtime_dir=runtime_dir,
            log_path=log_path,
            runtime_indicator_config=runtime_indicator_config,
            prompt_surface_bundle=prompt_surface_bundle,
            canonical_source_dir=canonical_source_dir,
            runtime_host=runtime_host,
            runtime_port=runtime_port,
            server_ready_timeout_s=server_ready_timeout_s,
            client_timeout_s=client_timeout_s,
        ),
        validate_and_build_trace_rows=lambda rows,
        eval_manifest,
        variant: _validate_and_build_trace_rows(
            rows=rows,
            eval_manifest=eval_manifest,
            variant=variant,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return run_rollout_eval_v21(args, dependencies=_default_dependencies())
    except FailFastError as exc:
        print(f"LIBERO_ROLLOUT_EVAL_V21_FAIL_FAST {exc}", flush=True)
        return 1


def load_rollout_eval_v21_authority_bundle(
    authority_dir: str | Path,
) -> dict[str, object]:
    try:
        return load_rollout_eval_v21_authority_bundle_io(authority_dir)
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


__all__ = [
    "BOOTSTRAP_SCHEMA_VERSION",
    "EFFECTIVE_RUNTIME_SPEC_SCHEMA_VERSION",
    "METRIC_LADDER_SCHEMA_VERSION",
    "PAIRWISE_DELTA_SCHEMA_VERSION",
    "PRIMARY_METRIC_ID",
    "RequiredOutputPaths",
    "ResolvedCliInputs",
    "RequestedRuntimeBinding",
    "ExecutedRuntimeBinding",
    "TraceBuildOutputs",
    "RolloutEvalV21Dependencies",
    "RolloutEvalV21Workflow",
    "SUMMARY_SCHEMA_VERSION",
    "TRACE_REQUIRED_FIELDS",
    "build_rollout_input_summary_v21",
    "build_effective_runtime_spec",
    "build_eval_manifest_id",
    "effective_runtime_spec_hash",
    "load_rollout_eval_v21_authority_bundle",
    "main",
    "run_rollout_eval_v21",
]


if __name__ == "__main__":
    raise SystemExit(main())
