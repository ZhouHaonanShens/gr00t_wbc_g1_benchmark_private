#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.eval.protocols.tracked_gate import (  # noqa: E402
    EXPECTED_EVAL_AUTHORITY,
    compute_rollout_eval_manifest_hash,
    load_rollout_eval_manifest_v2,
    manifest_payload_v2,
)
from work.openpi.checkpoint import (  # noqa: E402
    expected_stock_checkpoint,
    is_remote_uri,
    load_provenance_pair,
    normalize_checkpoint_ref,
    resolve_servable_checkpoint_ref,
)
from work.openpi.contracts import RuntimeServerSpec  # noqa: E402
from work.openpi.dataloader import (  # noqa: E402
    json_ready,
    load_rollout_eval_v2_authority_bundle as load_rollout_eval_v2_authority_bundle_io,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    write_markdown,
)
from work.openpi.runtime import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    LIBERO_NATIVE_SMOKE_ENTRY,
    PolicyServerProcess,
    RuntimeCleanup,
    RuntimeEpisodeClient,
    RuntimePathsBuilder,
    pick_free_port,
    prepare_libero_config_dir,
)
from work.openpi.runtime.api import FailFastError as RuntimeBridgeError  # noqa: E402


SUMMARY_SCHEMA_VERSION = "openpi_libero_rollout_eval_summary_v2"
PER_EPISODE_SCHEMA_VERSION = "openpi_libero_rollout_episode_v2"
BOOTSTRAP_SCHEMA_VERSION = "openpi_libero_rollout_bootstrap_ci_v2"
VIDEO_INDEX_SCHEMA_VERSION = "openpi_libero_rollout_video_index_v2"
PAIRED_DELTA_SCHEMA_VERSION = "openpi_libero_rollout_paired_delta_v2"
GO_NO_GO_CORE_SCHEMA_VERSION = "openpi_libero_go_no_go_core_v2"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "agent" / "artifacts" / "openpi_libero_v2"
TOPIC = "openpi_libero_v2"
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEVIATION_NOTES_NAME = "deviation_notes.md"
SUMMARY_NAME = "summary.json"
PER_EPISODE_NAME = "per_episode_rollouts.jsonl"
VIDEO_INDEX_NAME = "video_index.json"
BOOTSTRAP_NAME = "bootstrap_ci.json"
EVAL_MANIFEST_NAME = "eval_manifest.json"
STOCK_VARIANTS = frozenset({"stock", "stock_libero_ref_v1"})


class FailFastError(RuntimeError):
    pass


def _expected_stock_checkpoint() -> str:
    return expected_stock_checkpoint()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libero_rollout_eval_v2.py",
        description=(
            "Materialize fresh_rollout_v2 LIBERO authority bundles from validated "
            "per-episode rollout records."
        ),
    )
    _ = parser.add_argument("--variant", required=True)
    _ = parser.add_argument("--checkpoint-dir", required=True)
    _ = parser.add_argument("--eval-manifest", required=True)
    _ = parser.add_argument("--baseline-variant", required=True)
    _ = parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def _build_internal_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libero_rollout_eval_v2.py",
        description="Internal helpers for stock rollout bridge.",
    )
    _ = parser.add_argument(
        "--internal-mode", required=True, choices=("stock-episode",)
    )
    _ = parser.add_argument("--task-suite-name", required=True)
    _ = parser.add_argument("--task-id", required=True, type=int)
    _ = parser.add_argument("--seed", required=True, type=int)
    _ = parser.add_argument("--trial-index", required=True, type=int)
    _ = parser.add_argument("--host", required=True)
    _ = parser.add_argument("--port", required=True, type=int)
    _ = parser.add_argument("--openpi-root", required=True)
    _ = parser.add_argument("--libero-config-dir", required=True)
    _ = parser.add_argument("--video-path", required=True)
    _ = parser.add_argument("--episode-row-out", required=True)
    return parser


def _json_ready(value: Any) -> Any:
    return json_ready(value)


def _write_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    write_jsonl(path, rows, sort_keys=True)


def _write_markdown(path: Path, text: str) -> None:
    write_markdown(path, text)


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _is_remote_uri(raw: str) -> bool:
    return is_remote_uri(raw)


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


def _normalize_variant(raw_variant: str) -> str:
    variant = raw_variant.strip()
    if not variant:
        raise FailFastError("--variant must be non-empty")
    return variant


def _is_stock_variant(variant: str) -> bool:
    return variant.strip() in STOCK_VARIANTS


def _normalize_checkpoint_ref(raw_checkpoint_dir: str) -> str:
    try:
        return normalize_checkpoint_ref(raw_checkpoint_dir)
    except ValueError as exc:
        raise FailFastError(str(exc).replace("checkpoint reference", "--checkpoint-dir")) from exc


def build_eval_manifest_id(eval_manifest: Mapping[str, object]) -> str:
    payload = manifest_payload_v2(eval_manifest)
    manifest_name = str(payload["manifest_name"])
    manifest_hash = compute_rollout_eval_manifest_hash(payload)
    return f"{manifest_name}_{manifest_hash[:12]}"


def _resolve_rollouts_root(raw_output_root: str) -> Path:
    output_root = Path(raw_output_root).resolve()
    if output_root.name == "rollouts":
        return output_root
    return output_root / "rollouts"


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


def _build_artifact_paths(
    *,
    output_root: str,
    variant: str,
    eval_manifest_id: str,
    baseline_variant: str,
) -> dict[str, Path]:
    rollouts_root = _resolve_rollouts_root(output_root)
    artifact_dir = rollouts_root / variant / eval_manifest_id
    runtime_dir = _build_runtime_dir(variant, eval_manifest_id)
    paired_name = f"paired_delta_vs_{baseline_variant}.json"
    return {
        "rollouts_root": rollouts_root,
        "artifact_dir": artifact_dir,
        "runtime_dir": runtime_dir,
        "log_path": runtime_dir / "eval.log",
        "per_episode": artifact_dir / PER_EPISODE_NAME,
        "video_index": artifact_dir / VIDEO_INDEX_NAME,
        "bootstrap_ci": artifact_dir / BOOTSTRAP_NAME,
        "paired_delta": artifact_dir / paired_name,
        "eval_manifest": artifact_dir / EVAL_MANIFEST_NAME,
        "deviation_notes": artifact_dir / DEVIATION_NOTES_NAME,
        "summary": artifact_dir / SUMMARY_NAME,
    }


def _log(message: str, *, log_path: Path | None = None) -> None:
    print(message, flush=True)
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        _ = handle.write(message)
        _ = handle.write("\n")


def _reject_legacy_authority_input(raw_checkpoint_dir: str) -> None:
    checkpoint_dir = raw_checkpoint_dir.strip()
    normalized = checkpoint_dir.replace("\\", "/")
    rejection_suffix = "legacy/offline proxy rejected"
    if normalized.endswith("libero_recap_eval.py"):
        raise FailFastError(
            f"libero_recap_eval.py rejected as v2 authority; {rejection_suffix}"
        )
    if "openpi_libero_recap_eval" in normalized:
        raise FailFastError(
            f"libero_recap_eval.py rejected as v2 authority; {rejection_suffix}"
        )
    if normalized.endswith("agent/artifacts/openpi_libero_native/summary.json"):
        raise FailFastError(
            f"old stock summary rejected as v2 authority; {rejection_suffix}"
        )
    if not _is_remote_uri(checkpoint_dir):
        path = Path(checkpoint_dir).resolve()
        if path.is_file():
            if path.name == "summary.json" and "openpi_libero_native" in str(path):
                raise FailFastError(
                    f"old stock summary rejected as v2 authority; {rejection_suffix}"
                )
            payload = _read_json(path)
            schema_version = str(payload.get("schema_version", "")).strip()
            if schema_version == SUMMARY_SCHEMA_VERSION:
                return
            if path.name == "paired_summary.json" or path.name == "summary.json":
                if "libero_recap" in schema_version or "paired_summary" in payload:
                    raise FailFastError(
                        f"libero_recap_eval.py rejected as v2 authority; {rejection_suffix}"
                    )
                if "summary" in payload and "paired_summary" in payload:
                    raise FailFastError(
                        f"libero_recap_eval.py rejected as v2 authority; {rejection_suffix}"
                    )
                raise FailFastError(
                    f"old stock summary rejected as v2 authority; {rejection_suffix}"
                )


def _candidate_rollout_source_dirs(
    *,
    checkpoint_ref: str,
    output_root: str,
    variant: str,
    eval_manifest_id: str,
) -> list[Path]:
    candidates: list[Path] = []
    if not _is_remote_uri(checkpoint_ref):
        checkpoint_path = Path(checkpoint_ref)
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            candidates.extend(
                [
                    checkpoint_path / "rollout_eval_v2_input",
                    checkpoint_path.parent / "rollout_eval_v2_input",
                    checkpoint_path,
                    checkpoint_path.parent,
                ]
            )
    rollouts_root = _resolve_rollouts_root(output_root)
    candidates.extend(
        [
            rollouts_root.parent / "staging" / variant / eval_manifest_id,
            rollouts_root / variant / eval_manifest_id / "_staging",
            rollouts_root / variant / eval_manifest_id,
        ]
    )
    return candidates


def _resolve_rollout_source_dir(
    *,
    checkpoint_ref: str,
    output_root: str,
    variant: str,
    eval_manifest_id: str,
) -> Path:
    candidates = _candidate_rollout_source_dirs(
        checkpoint_ref=checkpoint_ref,
        output_root=output_root,
        variant=variant,
        eval_manifest_id=eval_manifest_id,
    )
    checked: list[str] = []
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in checked:
            continue
        checked.append(candidate_str)
        if (candidate / PER_EPISODE_NAME).is_file():
            return candidate
    raise FailFastError(
        "missing fresh rollout input bundle; expected "
        + PER_EPISODE_NAME
        + " under one of: "
        + ", ".join(checked)
    )


def _select_materialization_source_dir(
    *, checkpoint_ref: str, output_root: str, variant: str, eval_manifest_id: str
) -> Path:
    if not _is_remote_uri(checkpoint_ref):
        checkpoint_path = Path(checkpoint_ref)
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            return checkpoint_path.parent / "rollout_eval_v2_input"
    rollouts_root = _resolve_rollouts_root(output_root)
    return rollouts_root / variant / eval_manifest_id / "_staging"


def _prepend_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _prepare_stock_runtime_imports(
    *, openpi_root: Path, libero_config_dir: Path
) -> None:
    _prepend_sys_path(openpi_root / "third_party" / "libero")
    _prepend_sys_path(openpi_root / "packages" / "openpi-client" / "src")
    _prepend_sys_path(openpi_root / "src")
    os.environ["LIBERO_CONFIG_PATH"] = str(libero_config_dir)
    _ = os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def _pick_free_port(host: str, start_port: int) -> int:
    return pick_free_port(host, start_port)


def _run_stock_episode(
    *,
    task_suite_name: str,
    task_id: int,
    seed: int,
    trial_index: int,
    video_path: Path,
    host: str,
    port: int,
) -> dict[str, object]:
    try:
        client = RuntimeEpisodeClient()
        return client.run_stock_episode_direct(
            task_suite_name=task_suite_name,
            task_id=task_id,
            seed=seed,
            trial_index=trial_index,
            video_path=video_path,
            host=host,
            port=port,
        )
    except RuntimeBridgeError as exc:
        raise FailFastError(str(exc)) from exc


def _run_stock_episode_subprocess(
    *,
    task_suite_name: str,
    task_id: int,
    seed: int,
    trial_index: int,
    video_path: Path,
    host: str,
    port: int,
    venv_python: Path,
    openpi_root: Path,
    libero_config_dir: Path,
    runtime_dir: Path,
    timeout_s: float,
) -> dict[str, object]:
    try:
        client = RuntimeEpisodeClient()
        return client.run_stock_episode(
            task_suite_name=task_suite_name,
            task_id=task_id,
            seed=seed,
            trial_index=trial_index,
            video_path=video_path,
            host=host,
            port=port,
            venv_python=venv_python,
            openpi_root=openpi_root,
            libero_config_dir=libero_config_dir,
            runtime_dir=runtime_dir,
            timeout_s=timeout_s,
            episode_entry=Path(__file__).resolve(),
        )
    except RuntimeBridgeError as exc:
        raise FailFastError(str(exc)) from exc


def _run_internal_stock_episode(args: argparse.Namespace) -> int:
    _prepare_stock_runtime_imports(
        openpi_root=Path(args.openpi_root),
        libero_config_dir=Path(args.libero_config_dir),
    )
    row = _run_stock_episode(
        task_suite_name=str(args.task_suite_name),
        task_id=int(args.task_id),
        seed=int(args.seed),
        trial_index=int(args.trial_index),
        video_path=Path(args.video_path),
        host=str(args.host),
        port=int(args.port),
    )
    _write_json(Path(args.episode_row_out), row)
    print("LIBERO_ROLLOUT_EVAL_V2_STOCK_EPISODE_DONE", flush=True)
    return 0


def _materialize_checkpoint_rollout_source(
    *,
    checkpoint_ref: str,
    variant: str,
    eval_manifest: Mapping[str, object],
    source_dir: Path,
    artifact_dir: Path,
    runtime_dir: Path,
    log_path: Path,
) -> Path:
    manifest = manifest_payload_v2(eval_manifest)
    task_suite_name = _require_non_empty_str(
        manifest.get("task_suite_name"), context="eval_manifest.task_suite_name"
    )
    task_ids = _coerce_int_sequence(
        manifest.get("task_ids"), context="eval_manifest.task_ids"
    )
    seed_manifest = _coerce_int_sequence(
        manifest.get("seed_manifest"), context="eval_manifest.seed_manifest"
    )
    num_trials_per_task = _coerce_int(
        manifest.get("num_trials_per_task"), context="eval_manifest.num_trials_per_task"
    )
    paths = RuntimePathsBuilder().build()
    host = DEFAULT_HOST
    port = _pick_free_port(host, DEFAULT_PORT)
    serve_checkpoint_ref, serve_checkpoint_mode = _resolve_servable_checkpoint_ref(
        checkpoint_ref=checkpoint_ref,
        variant=variant,
    )
    source_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    bridge_runtime_dir = runtime_dir / "native_bridge"
    bridge_runtime_dir.mkdir(parents=True, exist_ok=True)
    harness_log = bridge_runtime_dir / "harness.log"
    server_log = bridge_runtime_dir / "server.log"
    libero_config_dir = prepare_libero_config_dir(
        paths.openpi_root, bridge_runtime_dir
    )
    server_spec = RuntimeServerSpec(
        host=host,
        port=port,
        checkpoint_dir=serve_checkpoint_ref,
        server_ready_timeout_s=150.0,
        client_timeout_s=80.0,
    )
    proc: subprocess.Popen[str] | None = None
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
            f"native rollout bridge ready variant={variant} host={host} port={port} probe_at={probe_payload.get('probed_at', '')}",
            log_path=log_path,
        )
        rows: list[dict[str, object]] = []
        for task_id in task_ids:
            for seed in seed_manifest:
                for trial_index in range(num_trials_per_task):
                    video_path = (
                        artifact_dir
                        / "videos"
                        / f"task{task_id}_seed{seed}_trial{trial_index}.mp4"
                    )
                    _log(
                        "running fresh rollout "
                        + f"variant={variant} task_id={task_id} seed={seed} trial_index={trial_index}",
                        log_path=log_path,
                    )
                    row = _run_stock_episode_subprocess(
                        task_suite_name=task_suite_name,
                        task_id=task_id,
                        seed=seed,
                        trial_index=trial_index,
                        video_path=video_path,
                        host=host,
                        port=port,
                        venv_python=paths.openpi_venv_python,
                        openpi_root=paths.openpi_root,
                        libero_config_dir=libero_config_dir,
                        runtime_dir=bridge_runtime_dir,
                        timeout_s=float(server_spec.client_timeout_s),
                    )
                    rows.append(row)
        _write_jsonl(source_dir / PER_EPISODE_NAME, rows)
        _write_json(
            source_dir / "rollout_input_summary.json",
            {
                "schema_version": "openpi_libero_rollout_eval_v2_input_v1",
                "variant": variant,
                "checkpoint_dir": checkpoint_ref,
                "serve_checkpoint_ref": serve_checkpoint_ref,
                "serve_checkpoint_mode": serve_checkpoint_mode,
                "task_suite_name": task_suite_name,
                "task_ids": list(task_ids),
                "seed_manifest": list(seed_manifest),
                "num_trials_per_task": num_trials_per_task,
                "server_log": str(server_log),
                "harness_log": str(harness_log),
                "host": host,
                "port": port,
                "episode_count": len(rows),
            },
        )
        return source_dir
    finally:
        RuntimeCleanup.close_process(proc)
        RuntimeCleanup.close_handle(server_handle)


def _ensure_rollout_source_dir(
    *,
    checkpoint_ref: str,
    output_root: str,
    variant: str,
    eval_manifest_id: str,
    eval_manifest: Mapping[str, object],
    artifact_dir: Path,
    runtime_dir: Path,
    log_path: Path,
) -> Path:
    try:
        return _resolve_rollout_source_dir(
            checkpoint_ref=checkpoint_ref,
            output_root=output_root,
            variant=variant,
            eval_manifest_id=eval_manifest_id,
        )
    except FailFastError:
        source_dir = _select_materialization_source_dir(
            checkpoint_ref=checkpoint_ref,
            output_root=output_root,
            variant=variant,
            eval_manifest_id=eval_manifest_id,
        )
    _log(
        "missing pre-materialized per_episode_rollouts.jsonl; materializing fresh rollouts now",
        log_path=log_path,
    )
    return _materialize_checkpoint_rollout_source(
        checkpoint_ref=checkpoint_ref,
        variant=variant,
        eval_manifest=eval_manifest,
        source_dir=source_dir,
        artifact_dir=artifact_dir,
        runtime_dir=runtime_dir,
        log_path=log_path,
    )


def _episode_sort_key(row: Mapping[str, object]) -> tuple[int, int, int]:
    return (
        _coerce_int(row.get("task_id"), context="episode.task_id"),
        _coerce_int(row.get("seed"), context="episode.seed"),
        _coerce_int(row.get("trial_index"), context="episode.trial_index"),
    )


def _validate_and_canonicalize_episode_rollouts(
    *,
    rows: Sequence[Mapping[str, object]],
    eval_manifest: Mapping[str, object],
    variant: str,
    checkpoint_ref: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest = manifest_payload_v2(eval_manifest)
    task_suite_name = str(manifest["task_suite_name"])
    manifest_name = str(manifest["manifest_name"])
    task_ids = _coerce_int_sequence(
        manifest.get("task_ids"), context="eval_manifest.task_ids"
    )
    seed_manifest = _coerce_int_sequence(
        manifest.get("seed_manifest"), context="eval_manifest.seed_manifest"
    )
    num_trials_per_task = _coerce_int(
        manifest.get("num_trials_per_task"), context="eval_manifest.num_trials_per_task"
    )
    expected_keys = {
        (task_id, seed, trial_index)
        for task_id in task_ids
        for seed in seed_manifest
        for trial_index in range(num_trials_per_task)
    }

    seen: set[tuple[int, int, int]] = set()
    canonical_rows: list[dict[str, object]] = []
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
        trial_index = _coerce_int(
            mapping.get("trial_index"), context=f"episode[{index}].trial_index"
        )
        success = _coerce_bool(
            mapping.get("success"), context=f"episode[{index}].success"
        )
        steps_observed = _coerce_int(
            mapping.get("steps_observed", mapping.get("steps", 0)),
            context=f"episode[{index}].steps_observed",
        )
        key = (task_id, seed, trial_index)
        episode_id = f"task{task_id}_seed{seed}_trial{trial_index}"
        if key in seen:
            duplicate_episode_ids.append(episode_id)
            continue
        seen.add(key)
        if key not in expected_keys:
            unexpected_keys.append(episode_id)
        video_path = str(
            mapping.get("video_path", mapping.get("video_relpath", ""))
        ).strip()
        canonical_rows.append(
            {
                "schema_version": PER_EPISODE_SCHEMA_VERSION,
                "eval_authority": EXPECTED_EVAL_AUTHORITY,
                "manifest_name": manifest_name,
                "variant": variant,
                "checkpoint_dir": checkpoint_ref,
                "task_suite_name": task_suite_name,
                "task_id": task_id,
                "seed": seed,
                "trial_index": trial_index,
                "episode_id": episode_id,
                "success": success,
                "steps_observed": steps_observed,
                "video_path": video_path,
                "episode_status": str(mapping.get("episode_status", "ok")).strip()
                or "ok",
                "error": str(mapping.get("error", "")).strip(),
            }
        )

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
            f"task{task_id}_seed{seed}_trial{trial_index}"
            for task_id, seed, trial_index in missing_keys
        ]
        raise FailFastError(
            "rollout scope incomplete: missing episodes "
            + ", ".join(missing_episode_ids)
        )

    canonical_rows.sort(key=_episode_sort_key)
    success_count = sum(1 for row in canonical_rows if bool(row["success"]))
    episode_count = len(canonical_rows)
    scope_audit: dict[str, object] = {
        "manifest_name": manifest_name,
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
        "success_rate": (float(success_count) / float(episode_count))
        if episode_count
        else 0.0,
    }
    return canonical_rows, scope_audit


def _bootstrap_success_rate(
    *,
    canonical_rows: Sequence[Mapping[str, object]],
    eval_manifest: Mapping[str, object],
    variant: str,
) -> dict[str, object]:
    outcomes = [1 if bool(row["success"]) else 0 for row in canonical_rows]
    sample_size = len(outcomes)
    if sample_size == 0:
        raise FailFastError("cannot bootstrap empty rollout set")
    point_estimate = float(sum(outcomes)) / float(sample_size)
    seed_material = f"{variant}:{compute_rollout_eval_manifest_hash(eval_manifest)}"
    rng = random.Random(int(_sha256_text(seed_material)[:16], 16))
    samples: list[float] = []
    for _ in range(DEFAULT_BOOTSTRAP_ITERATIONS):
        total = 0
        for _sample_index in range(sample_size):
            total += outcomes[rng.randrange(sample_size)]
        samples.append(float(total) / float(sample_size))
    samples.sort()
    lower_index = max(0, int(DEFAULT_BOOTSTRAP_ITERATIONS * 0.025))
    upper_index = min(
        DEFAULT_BOOTSTRAP_ITERATIONS - 1,
        int(DEFAULT_BOOTSTRAP_ITERATIONS * 0.975),
    )
    return {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "eval_authority": EXPECTED_EVAL_AUTHORITY,
        "metric": "success_rate",
        "variant": variant,
        "sample_size": sample_size,
        "bootstrap_iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "point_estimate": point_estimate,
        "ci_lower": float(samples[lower_index]),
        "ci_upper": float(samples[upper_index]),
        "deterministic_seed_material": seed_material,
    }


def _build_video_index(
    *,
    canonical_rows: Sequence[Mapping[str, object]],
    eval_manifest_id: str,
    variant: str,
) -> dict[str, object]:
    videos: list[dict[str, object]] = []
    present = 0
    for row in canonical_rows:
        video_path = str(row.get("video_path", "")).strip()
        if video_path:
            present += 1
        videos.append(
            {
                "episode_id": str(row["episode_id"]),
                "task_id": _coerce_int(row.get("task_id"), context="video.task_id"),
                "seed": _coerce_int(row.get("seed"), context="video.seed"),
                "trial_index": _coerce_int(
                    row.get("trial_index"), context="video.trial_index"
                ),
                "video_path": video_path,
                "has_video": bool(video_path),
            }
        )
    return {
        "schema_version": VIDEO_INDEX_SCHEMA_VERSION,
        "eval_authority": EXPECTED_EVAL_AUTHORITY,
        "variant": variant,
        "eval_manifest_id": eval_manifest_id,
        "video_count": present,
        "episode_count": len(canonical_rows),
        "videos": videos,
    }


def _load_provenance_pair(
    source_dir: Path,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    return load_provenance_pair(source_dir)


def _resolve_servable_checkpoint_ref(
    *, checkpoint_ref: str, variant: str
) -> tuple[str, str]:
    try:
        return resolve_servable_checkpoint_ref(
            checkpoint_ref=checkpoint_ref,
            variant=variant,
            stock_variants=STOCK_VARIANTS,
        )
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc


def _extract_train_provenance(
    *,
    variant: str,
    train_manifest: Mapping[str, object] | None,
    checkpoint_provenance: Mapping[str, object] | None,
) -> dict[str, object]:
    tracked_fields = (
        "dataset_fingerprint",
        "episode_universe_hash",
        "base_checkpoint_id",
        "train_budget_id",
        "consumer_mode",
        "gate_eval_manifest_hash",
    )
    if train_manifest is None and checkpoint_provenance is None:
        status = "not_applicable" if _is_stock_variant(variant) else "missing"
        note = (
            "stock upstream baseline has no local train_manifest/checkpoint_provenance"
            if _is_stock_variant(variant)
            else "missing train_manifest/checkpoint_provenance for non-stock variant"
        )
        return {
            "status": status,
            "note": note,
            "train_manifest_present": False,
            "checkpoint_provenance_present": False,
            "parity_ok": variant == "stock",
            "fields": {field: "" for field in tracked_fields},
        }

    source_payload = checkpoint_provenance or train_manifest
    if source_payload is None:
        raise FailFastError("internal error: missing provenance source payload")
    fields = {
        field: str(source_payload.get(field, "")).strip() for field in tracked_fields
    }
    mismatches: list[str] = []
    if train_manifest is not None and checkpoint_provenance is not None:
        for field in tracked_fields:
            left = train_manifest.get(field)
            right = checkpoint_provenance.get(field)
            if left != right:
                mismatches.append(field)
    return {
        "status": "present" if not mismatches else "mismatch",
        "note": "" if not mismatches else "train/provenance parity mismatch",
        "train_manifest_present": train_manifest is not None,
        "checkpoint_provenance_present": checkpoint_provenance is not None,
        "parity_ok": len(mismatches) == 0,
        "mismatched_fields": mismatches,
        "fields": fields,
    }


def _build_deviation_notes(
    *,
    variant: str,
    baseline_variant: str,
    source_dir: Path,
    train_provenance: Mapping[str, object],
) -> list[str]:
    notes = [
        f"本 authority bundle 明确绑定 eval_authority={EXPECTED_EVAL_AUTHORITY}。",
        f"fresh rollout source_dir={source_dir}",
    ]
    if _is_stock_variant(variant):
        notes.append(
            "stock 行必须通过 fresh_rollout_v2 重新物化 authority bundle；old stock summary rejected as v2 authority。"
        )
    if variant == baseline_variant:
        notes.append(
            "baseline_variant 与当前 variant 相同；paired delta 退化为自比较。"
        )
    provenance_status = str(train_provenance.get("status", "")).strip()
    if provenance_status == "not_applicable":
        notes.append(str(train_provenance.get("note", "")).strip())
    elif provenance_status != "present":
        notes.append(
            str(train_provenance.get("note", "missing provenance parity")).strip()
        )
    return notes


def _deviation_notes_markdown(notes: Sequence[str]) -> str:
    return (
        "\n".join(
            ["# rollout_eval_v2 deviation notes", ""] + [f"- {note}" for note in notes]
        )
        + "\n"
    )


def _summary_row(
    *,
    variant: str,
    checkpoint_ref: str,
    scope_audit: Mapping[str, object],
) -> dict[str, object]:
    return {
        "variant": variant,
        "checkpoint_dir": checkpoint_ref,
        "episode_count": _coerce_int(
            scope_audit.get("observed_episode_count"),
            context="scope_audit.observed_episode_count",
        ),
        "success_count": _coerce_int(
            scope_audit.get("success_count"), context="scope_audit.success_count"
        ),
        "failure_count": _coerce_int(
            scope_audit.get("failure_count"), context="scope_audit.failure_count"
        ),
        "success_rate": _coerce_float(
            scope_audit.get("success_rate"), context="scope_audit.success_rate"
        ),
    }


def _validate_v2_summary_payload(
    summary_payload: Mapping[str, object],
) -> dict[str, object]:
    schema_version = str(summary_payload.get("schema_version", "")).strip()
    eval_authority = str(summary_payload.get("eval_authority", "")).strip()
    if (
        schema_version != SUMMARY_SCHEMA_VERSION
        or eval_authority != EXPECTED_EVAL_AUTHORITY
    ):
        raise FailFastError("legacy/offline proxy rejected")
    return dict(summary_payload)


def _load_baseline_summary(
    *,
    paths: Mapping[str, Path],
    variant: str,
    baseline_variant: str,
    current_summary: Mapping[str, object],
) -> dict[str, object]:
    if baseline_variant == variant:
        return dict(current_summary)
    baseline_summary_path = (
        paths["rollouts_root"]
        / baseline_variant
        / paths["artifact_dir"].name
        / SUMMARY_NAME
    )
    if not baseline_summary_path.is_file():
        raise FailFastError(
            f"missing baseline authority bundle summary: {baseline_summary_path}"
        )
    return _validate_v2_summary_payload(_read_json(baseline_summary_path))


def _build_paired_delta(
    *,
    variant: str,
    baseline_variant: str,
    eval_manifest_id: str,
    checkpoint_ref: str,
    current_summary_row: Mapping[str, object],
    baseline_summary_payload: Mapping[str, object],
) -> dict[str, object]:
    baseline_rollout = _require_mapping(
        baseline_summary_payload.get("rollout_summary", {}),
        context="baseline.rollout_summary",
    )
    baseline_checkpoint_dir = _require_non_empty_str(
        baseline_summary_payload.get("checkpoint_dir", ""),
        context="baseline.checkpoint_dir",
    )
    delta_success_rate = _coerce_float(
        current_summary_row.get("success_rate"), context="current.success_rate"
    ) - _coerce_float(
        baseline_rollout.get("success_rate", 0.0), context="baseline.success_rate"
    )
    delta_success_count = _coerce_int(
        current_summary_row.get("success_count"), context="current.success_count"
    ) - _coerce_int(
        baseline_rollout.get("success_count", 0), context="baseline.success_count"
    )
    return {
        "schema_version": PAIRED_DELTA_SCHEMA_VERSION,
        "eval_authority": EXPECTED_EVAL_AUTHORITY,
        "eval_manifest_id": eval_manifest_id,
        "variant": variant,
        "baseline_variant": baseline_variant,
        "current": dict(current_summary_row),
        "baseline": {
            "variant": baseline_variant,
            "checkpoint_dir": baseline_checkpoint_dir,
            "episode_count": _coerce_int(
                baseline_rollout.get("episode_count", 0),
                context="baseline.episode_count",
            ),
            "success_count": _coerce_int(
                baseline_rollout.get("success_count", 0),
                context="baseline.success_count",
            ),
            "failure_count": _coerce_int(
                baseline_rollout.get("failure_count", 0),
                context="baseline.failure_count",
            ),
            "success_rate": _coerce_float(
                baseline_rollout.get("success_rate", 0.0),
                context="baseline.success_rate",
            ),
        },
        "delta_success_rate": delta_success_rate,
        "delta_success_count": delta_success_count,
        "checkpoint_dir": checkpoint_ref,
    }


def _required_output_paths(summary_dir: Path, baseline_variant: str) -> dict[str, Path]:
    return {
        "per_episode_rollouts": summary_dir / PER_EPISODE_NAME,
        "video_index": summary_dir / VIDEO_INDEX_NAME,
        "bootstrap_ci": summary_dir / BOOTSTRAP_NAME,
        "paired_delta": summary_dir / f"paired_delta_vs_{baseline_variant}.json",
        "eval_manifest": summary_dir / EVAL_MANIFEST_NAME,
        "deviation_notes": summary_dir / DEVIATION_NOTES_NAME,
        "summary": summary_dir / SUMMARY_NAME,
    }


def _output_presence_report(
    summary_dir: Path, baseline_variant: str
) -> dict[str, object]:
    paths = _required_output_paths(summary_dir, baseline_variant)
    present = {name: path.is_file() for name, path in paths.items()}
    return {
        "all_present": all(present.values()),
        "present": present,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def load_rollout_eval_v2_authority_bundle(
    authority_dir: str | Path,
) -> dict[str, object]:
    try:
        bundle = load_rollout_eval_v2_authority_bundle_io(authority_dir)
    except ValueError as exc:
        raise FailFastError(str(exc)) from exc
    bundle["summary"] = _validate_v2_summary_payload(
        cast(Mapping[str, object], bundle["summary"])
    )
    return bundle


def derive_go_no_go_core_from_authority_bundle(
    authority_bundle: Mapping[str, object],
) -> dict[str, object]:
    summary = _require_mapping(
        authority_bundle.get("summary", {}), context="bundle.summary"
    )
    eval_manifest = _require_mapping(
        authority_bundle.get("eval_manifest", {}), context="bundle.eval_manifest"
    )
    per_episode_rollouts = _require_sequence(
        authority_bundle.get("per_episode_rollouts", []),
        context="bundle.per_episode_rollouts",
    )
    paired_delta = _require_mapping(
        authority_bundle.get("paired_delta", {}), context="bundle.paired_delta"
    )
    bootstrap_ci = _require_mapping(
        authority_bundle.get("bootstrap_ci", {}), context="bundle.bootstrap_ci"
    )
    video_index = _require_mapping(
        authority_bundle.get("video_index", {}), context="bundle.video_index"
    )
    summary_rollout = _require_mapping(
        summary.get("rollout_summary", {}), context="bundle.summary.rollout_summary"
    )
    scope_audit = _require_mapping(
        summary.get("scope_audit", {}), context="bundle.summary.scope_audit"
    )
    train_provenance = _require_mapping(
        summary.get("train_provenance", {}), context="bundle.summary.train_provenance"
    )
    authority_dir = Path(str(authority_bundle.get("authority_dir", "")).strip())
    baseline_variant = str(summary.get("baseline_variant", "")).strip()
    presence = _output_presence_report(authority_dir, baseline_variant)
    manifest = manifest_payload_v2(eval_manifest)
    total_episodes = _coerce_int(
        summary_rollout.get("episode_count", 0),
        context="summary.rollout_summary.episode_count",
    )
    manifest_task_ids = _coerce_int_sequence(
        manifest.get("task_ids"), context="eval_manifest.task_ids"
    )
    manifest_seeds = _coerce_int_sequence(
        manifest.get("seed_manifest"), context="eval_manifest.seed_manifest"
    )
    manifest_trials = _coerce_int(
        manifest.get("num_trials_per_task"), context="eval_manifest.num_trials_per_task"
    )
    gate_rows = [
        {
            "gate": "G0",
            "name": "fresh_rollout_authority",
            "status": "pass"
            if str(summary.get("eval_authority", "")).strip() == EXPECTED_EVAL_AUTHORITY
            and str(eval_manifest.get("eval_authority", "")).strip()
            == EXPECTED_EVAL_AUTHORITY
            else "fail",
            "detail": f"eval_authority={summary.get('eval_authority', '')}",
        },
        {
            "gate": "G1",
            "name": "manifest_scope_matches_bundle",
            "status": "pass"
            if total_episodes == len(per_episode_rollouts)
            and total_episodes
            == _coerce_int(
                scope_audit.get("expected_episode_count", -1),
                context="scope_audit.expected_episode_count",
            )
            and str(summary.get("eval_manifest_id", "")).strip()
            == build_eval_manifest_id(eval_manifest)
            else "fail",
            "detail": f"expected_total={manifest_trials}x{len(manifest_task_ids)}x{len(manifest_seeds)}",
        },
        {
            "gate": "G2",
            "name": "scope_audit_complete",
            "status": "pass"
            if bool(scope_audit.get("scope_complete", False))
            else "fail",
            "detail": f"observed={scope_audit.get('observed_episode_count', 0)} expected={scope_audit.get('expected_episode_count', 0)}",
        },
    ]

    dataset_fingerprint = str(
        _require_mapping(
            train_provenance.get("fields", {}), context="train_provenance.fields"
        ).get("dataset_fingerprint", "")
    ).strip()
    episode_universe_hash = str(
        _require_mapping(
            train_provenance.get("fields", {}), context="train_provenance.fields"
        ).get("episode_universe_hash", "")
    ).strip()
    provenance_status = str(train_provenance.get("status", "")).strip()
    gate_rows.extend(
        [
            {
                "gate": "G3",
                "name": "dataset_fingerprint_bound",
                "status": (
                    "not_applicable"
                    if provenance_status == "not_applicable"
                    else ("pass" if bool(dataset_fingerprint) else "fail")
                ),
                "detail": dataset_fingerprint or str(train_provenance.get("note", "")),
            },
            {
                "gate": "G4",
                "name": "episode_universe_hash_bound",
                "status": (
                    "not_applicable"
                    if provenance_status == "not_applicable"
                    else ("pass" if bool(episode_universe_hash) else "fail")
                ),
                "detail": episode_universe_hash
                or str(train_provenance.get("note", "")),
            },
            {
                "gate": "G5",
                "name": "train_provenance_parity",
                "status": (
                    "not_applicable"
                    if provenance_status == "not_applicable"
                    else (
                        "pass"
                        if bool(train_provenance.get("parity_ok", False))
                        else "fail"
                    )
                ),
                "detail": str(train_provenance.get("note", ""))
                or "train/provenance parity ok",
            },
            {
                "gate": "G6",
                "name": "paired_delta_uses_v2_baseline",
                "status": "pass"
                if str(paired_delta.get("eval_authority", "")).strip()
                == EXPECTED_EVAL_AUTHORITY
                and str(paired_delta.get("baseline_variant", "")).strip()
                == baseline_variant
                else "fail",
                "detail": f"baseline_variant={paired_delta.get('baseline_variant', '')}",
            },
            {
                "gate": "G7",
                "name": "required_outputs_present_and_consistent",
                "status": "pass"
                if bool(presence["all_present"])
                and str(bootstrap_ci.get("eval_authority", "")).strip()
                == EXPECTED_EVAL_AUTHORITY
                and _coerce_int(
                    video_index.get("episode_count", 0),
                    context="video_index.episode_count",
                )
                == total_episodes
                else "fail",
                "detail": json.dumps(
                    presence["present"], ensure_ascii=False, sort_keys=True
                ),
            },
        ]
    )
    overall_go = all(row["status"] in {"pass", "not_applicable"} for row in gate_rows)
    return {
        "schema_version": GO_NO_GO_CORE_SCHEMA_VERSION,
        "eval_authority": EXPECTED_EVAL_AUTHORITY,
        "overall_go": overall_go,
        "gates": gate_rows,
    }


def _materialize_bundle(args: argparse.Namespace) -> int:
    variant = _normalize_variant(str(args.variant))
    baseline_variant = _normalize_variant(str(args.baseline_variant))
    raw_checkpoint_dir = str(args.checkpoint_dir)
    _reject_legacy_authority_input(raw_checkpoint_dir)
    checkpoint_ref = _normalize_checkpoint_ref(raw_checkpoint_dir)
    eval_manifest = manifest_payload_v2(
        load_rollout_eval_manifest_v2(str(args.eval_manifest))
    )
    eval_manifest_hash = compute_rollout_eval_manifest_hash(eval_manifest)
    eval_manifest_id = build_eval_manifest_id(eval_manifest)
    paths = _build_artifact_paths(
        output_root=str(args.output_root),
        variant=variant,
        eval_manifest_id=eval_manifest_id,
        baseline_variant=baseline_variant,
    )
    paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    paths["runtime_dir"].mkdir(parents=True, exist_ok=True)
    log_path = paths["log_path"]
    _log(
        f"[{dt.datetime.now().isoformat(timespec='seconds')}] rollout_eval_v2 variant={variant} manifest={eval_manifest_id}",
        log_path=log_path,
    )
    source_dir = _ensure_rollout_source_dir(
        checkpoint_ref=checkpoint_ref,
        output_root=str(args.output_root),
        variant=variant,
        eval_manifest_id=eval_manifest_id,
        eval_manifest=eval_manifest,
        artifact_dir=paths["artifact_dir"],
        runtime_dir=paths["runtime_dir"],
        log_path=log_path,
    )
    _log(f"source_rollout_dir={source_dir}", log_path=log_path)
    input_rows = _read_jsonl(source_dir / PER_EPISODE_NAME)
    canonical_rows, scope_audit = _validate_and_canonicalize_episode_rollouts(
        rows=input_rows,
        eval_manifest=eval_manifest,
        variant=variant,
        checkpoint_ref=checkpoint_ref,
    )
    train_manifest, checkpoint_provenance = _load_provenance_pair(source_dir)
    train_provenance = _extract_train_provenance(
        variant=variant,
        train_manifest=train_manifest,
        checkpoint_provenance=checkpoint_provenance,
    )
    deviation_notes = _build_deviation_notes(
        variant=variant,
        baseline_variant=baseline_variant,
        source_dir=source_dir,
        train_provenance=train_provenance,
    )
    bootstrap_ci = _bootstrap_success_rate(
        canonical_rows=canonical_rows,
        eval_manifest=eval_manifest,
        variant=variant,
    )
    video_index = _build_video_index(
        canonical_rows=canonical_rows,
        eval_manifest_id=eval_manifest_id,
        variant=variant,
    )
    current_summary_row = _summary_row(
        variant=variant,
        checkpoint_ref=checkpoint_ref,
        scope_audit=scope_audit,
    )

    eval_manifest_payload = {
        **eval_manifest,
        "eval_manifest_hash": eval_manifest_hash,
        "eval_manifest_id": eval_manifest_id,
    }
    _write_json(paths["eval_manifest"], eval_manifest_payload)
    _write_jsonl(paths["per_episode"], canonical_rows)
    _write_json(paths["video_index"], video_index)
    _write_json(paths["bootstrap_ci"], bootstrap_ci)
    _write_markdown(
        paths["deviation_notes"], _deviation_notes_markdown(deviation_notes)
    )

    baseline_summary = _load_baseline_summary(
        paths=paths,
        variant=variant,
        baseline_variant=baseline_variant,
        current_summary={
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "eval_authority": EXPECTED_EVAL_AUTHORITY,
            "checkpoint_dir": checkpoint_ref,
            "rollout_summary": current_summary_row,
        },
    )
    paired_delta = _build_paired_delta(
        variant=variant,
        baseline_variant=baseline_variant,
        eval_manifest_id=eval_manifest_id,
        checkpoint_ref=checkpoint_ref,
        current_summary_row=current_summary_row,
        baseline_summary_payload=baseline_summary,
    )
    _write_json(paths["paired_delta"], paired_delta)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "eval_authority": EXPECTED_EVAL_AUTHORITY,
        "variant": variant,
        "baseline_variant": baseline_variant,
        "checkpoint_dir": checkpoint_ref,
        "output_dir": str(paths["artifact_dir"]),
        "runtime_dir": str(paths["runtime_dir"]),
        "source_rollout_dir": str(source_dir),
        "eval_manifest_id": eval_manifest_id,
        "eval_manifest_hash": eval_manifest_hash,
        "manifest_name": str(eval_manifest["manifest_name"]),
        "task_suite_name": str(eval_manifest["task_suite_name"]),
        "rollout_summary": {
            **current_summary_row,
            "episode_count": _coerce_int(
                scope_audit.get("observed_episode_count"),
                context="scope_audit.observed_episode_count",
            ),
        },
        "scope_audit": scope_audit,
        "train_provenance": train_provenance,
        "required_outputs": {
            "per_episode_rollouts": str(paths["per_episode"]),
            "video_index": str(paths["video_index"]),
            "bootstrap_ci": str(paths["bootstrap_ci"]),
            "paired_delta": str(paths["paired_delta"]),
            "eval_manifest": str(paths["eval_manifest"]),
            "deviation_notes": str(paths["deviation_notes"]),
            "summary": str(paths["summary"]),
        },
        "deviation_notes": list(deviation_notes),
    }
    _write_json(paths["summary"], summary)
    authority_bundle = {
        "authority_dir": str(paths["artifact_dir"]),
        "summary": summary,
        "eval_manifest": eval_manifest_payload,
        "per_episode_rollouts": canonical_rows,
        "video_index": video_index,
        "bootstrap_ci": bootstrap_ci,
        "paired_delta": paired_delta,
        "deviation_notes": _deviation_notes_markdown(deviation_notes),
    }
    summary["go_no_go_core"] = derive_go_no_go_core_from_authority_bundle(
        authority_bundle
    )
    _write_json(paths["summary"], summary)
    _log(f"summary_json={paths['summary']}", log_path=log_path)
    _log("LIBERO_ROLLOUT_EVAL_V2_DONE", log_path=log_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if "--internal-mode" in argv_list:
        args = _build_internal_parser().parse_args(argv_list)
        try:
            return _run_internal_stock_episode(args)
        except FailFastError as exc:
            print(f"LIBERO_ROLLOUT_EVAL_V2_FAIL_FAST {exc}", flush=True)
            return 1
    args = _build_parser().parse_args(argv_list)
    try:
        return _materialize_bundle(args)
    except FailFastError as exc:
        print(f"LIBERO_ROLLOUT_EVAL_V2_FAIL_FAST {exc}", flush=True)
        return 1


__all__ = [
    "BOOTSTRAP_SCHEMA_VERSION",
    "GO_NO_GO_CORE_SCHEMA_VERSION",
    "PAIRED_DELTA_SCHEMA_VERSION",
    "PER_EPISODE_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "VIDEO_INDEX_SCHEMA_VERSION",
    "build_eval_manifest_id",
    "derive_go_no_go_core_from_authority_bundle",
    "load_rollout_eval_v2_authority_bundle",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
