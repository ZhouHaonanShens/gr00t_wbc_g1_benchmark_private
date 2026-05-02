#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap import (  # noqa: E402
    RECAP_ONLY_VARIANT,
    RecapDatasetBundle,
    TrainCheckpointMetadata,
    build_frozen_comparison_manifest,
    build_train_runtime_dir,
    materialize_recap_checkpoint,
    read_json,
    resolve_recap_dataset,
    resolve_default_recap_dataset,
    validate_recap_only_variant,
    write_json,
)
from work.openpi.recap.real_variant_export import (  # noqa: E402
    RealVariantExportBlockedError,
    RealVariantExportRequest,
    run_real_variant_training_export,
)
from work.openpi.state_tokens import (  # noqa: E402
    RECAP_STATE_TOKENS_VARIANT,
    StateTokenDatasetBundle,
    StateTokenContractError,
    build_train_runtime_dir as build_state_token_train_runtime_dir,
    materialize_state_token_checkpoint,
    require_control_parity_ready,
    resolve_default_state_token_dataset,
    resolve_state_token_dataset,
    validate_state_token_variant,
)
from work.openpi.prompting.routes import (  # noqa: E402
    FIXEDADV_CONSTANT_CONSUMER_MODE,
    RECAP_RELABEL_CONSUMER_MODE,
)


FIXEDADV_CONTROL_VARIANT = "fixedadv_control"
SHUFFLED_ADV_DIAG_VARIANT = "shuffled_adv_diag"
SHUFFLED_ADV_DIAG_CONSUMER_MODE = "shuffled_adv_diag"

RECAP_ONLY_VARIANT_NAME = "recap_only_relabel8d_v2"
FIXEDADV_CONTROL_VARIANT_NAME = "fixedadv_relabel8d_control_v1"
RECAP_STATE_TOKENS_VARIANT_NAME = "recap_state_tokens_relabel8d_v2"
SHUFFLED_ADV_DIAG_VARIANT_NAME = "recap_shuffledadv_diag_v1"

RECAP_ONLY_CHECKPOINT_SOURCE = (
    "repo_local_openpi_recap_only_offline_advantage_conditioned_baseline"
)
FIXEDADV_CONTROL_CHECKPOINT_SOURCE = "repo_local_openpi_fixedadv_relabel8d_control_v1"
SHUFFLED_ADV_DIAG_CHECKPOINT_SOURCE = "repo_local_openpi_recap_shuffledadv_diag_v1"

BASE_CHECKPOINT_ID = "pi05_libero_anchor"
TRAIN_BUDGET_ID = "libero_cmp_budget_v2"
REUSE_VERDICT_NEW = "materialize_new_checkpoint"
REUSE_VERDICT_REUSED = "reuse_existing_checkpoint"
DEFAULT_GATE_EVAL_MANIFEST_PATH = (
    REPO_ROOT
    / "work"
    / "openpi"
    / "eval"
    / "manifests"
    / "eval_manifest_rollout_lite_v2.json"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libero_variant_train.py",
        description="Materialize the Task 7/9 offline checkpoint bundle.",
    )
    _ = parser.add_argument("--variant", required=True)
    _ = parser.add_argument("--output-dir", required=True)
    _ = parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Explicit dataset dir for fixedadv_control, recap_only, recap_state_tokens, or shuffled_adv_diag; if omitted, each variant uses its default compatible source.",
    )
    _ = parser.add_argument("--task-suite-name", required=True)
    _ = parser.add_argument("--task-ids", required=True)
    _ = parser.add_argument("--seeds", required=True)
    _ = parser.add_argument("--num-trials-per-task", required=True, type=int)
    _ = parser.add_argument(
        "--reuse-existing-checkpoint",
        action="store_true",
        help="Require an existing checkpoint bundle and refuse reuse on any parity mismatch.",
    )
    _ = parser.add_argument(
        "--gate-eval-manifest",
        default=None,
        help="Optional lite eval manifest path used to derive gate_eval_manifest_hash; defaults to work/openpi/eval/manifests/eval_manifest_rollout_lite_v2.json.",
    )
    return parser


def _log(message: str, *, log_path: Path) -> None:
    print(message, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        _ = handle.write(message)
        _ = handle.write("\n")


def _require_str_arg(args: argparse.Namespace, name: str) -> str:
    value = cast(object, getattr(args, name))
    if not isinstance(value, str):
        raise TypeError(f"argument {name} must be a string")
    return value


def _require_int_arg(args: argparse.Namespace, name: str) -> int:
    value = cast(object, getattr(args, name))
    if not isinstance(value, int):
        raise TypeError(f"argument {name} must be an int")
    return value


def _write_blocker_reports(output_dir: Path, payload: dict[str, object]) -> None:
    write_json(output_dir / "blocker_report.json", payload)
    write_json(output_dir / "best" / "blocker_report.json", payload)


def _clear_blocker_reports(output_dir: Path) -> None:
    for path in (
        output_dir / "blocker_report.json",
        output_dir / "best" / "blocker_report.json",
    ):
        if path.exists():
            path.unlink()


def _clear_state_token_success_artifacts(output_dir: Path) -> None:
    for path in (
        output_dir / "train_manifest.json",
        output_dir / "checkpoint_provenance.json",
        output_dir / "best" / "train_manifest.json",
        output_dir / "best" / "checkpoint_provenance.json",
        output_dir / "best" / "checkpoint.json",
    ):
        if path.exists():
            path.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_required_text(path: Path, *, field_name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing required {field_name} file: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"{field_name} at {path} must be a non-empty string")
    return value


def _resolve_gate_eval_manifest_path(raw_path: object) -> Path:
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path.strip()).resolve()
    else:
        path = DEFAULT_GATE_EVAL_MANIFEST_PATH.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "missing gate eval manifest required for gate_eval_manifest_hash: "
            + f"{path}"
        )
    return path


def _load_dataset_gate_metadata(dataset_dir: Path) -> tuple[str, str, str]:
    meta_dir = dataset_dir.resolve() / "meta"
    fingerprint_payload = read_json(meta_dir / "dataset_fingerprint.json")
    dataset_route_id = str(fingerprint_payload.get("route_id", "")).strip()
    if not dataset_route_id:
        raise ValueError(
            f"dataset_fingerprint.json under {meta_dir} is missing route_id"
        )
    dataset_fingerprint = str(fingerprint_payload.get("fingerprint_sha256", "")).strip()
    if not dataset_fingerprint:
        raise ValueError(
            f"dataset_fingerprint.json under {meta_dir} is missing fingerprint_sha256"
        )
    episode_universe_hash = _read_required_text(
        meta_dir / "episode_universe_hash.txt",
        field_name="episode_universe_hash",
    )
    return dataset_route_id, dataset_fingerprint, episode_universe_hash


def _require_local_orbax_checkpoint_assets(checkpoint_dir: Path) -> None:
    required_paths = (
        checkpoint_dir / "params" / "_METADATA",
        checkpoint_dir
        / "assets"
        / "physical-intelligence"
        / "libero"
        / "norm_stats.json",
    )
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "materialized non-stock checkpoint is incomplete: "
            + ", ".join(missing_paths)
        )


def _run_real_variant_training_export_or_block(
    *,
    variant: str,
    variant_name: str,
    dataset_dir: Path,
    runtime_dir: Path,
    consumer_mode: str,
    fixed_indicator_mode: str | None,
    output_dir: Path,
    log_path: Path,
) -> Path:
    try:
        export_bundle = run_real_variant_training_export(
            RealVariantExportRequest(
                variant=variant,
                variant_name=variant_name,
                dataset_dir=dataset_dir,
                runtime_dir=runtime_dir,
                consumer_mode=consumer_mode,
                fixed_indicator_mode=fixed_indicator_mode,
            )
        )
    except RealVariantExportBlockedError as exc:
        _write_blocker_reports(output_dir, exc.payload)
        _log(str(exc), log_path=log_path)
        _log(
            f"blocker_report={output_dir / 'blocker_report.json'}",
            log_path=log_path,
        )
        raise
    _log(
        f"real_variant_export={export_bundle.export_dir} runtime_log={export_bundle.runtime_log_path}",
        log_path=log_path,
    )
    return export_bundle.export_dir


def _metadata_for_reuse_gate(
    *,
    variant: str,
    metadata: TrainCheckpointMetadata,
) -> dict[str, object]:
    return {
        "variant": variant,
        "variant_name": metadata.variant_name,
        "dataset_route_id": metadata.dataset_route_id,
        "dataset_fingerprint": metadata.dataset_fingerprint,
        "episode_universe_hash": metadata.episode_universe_hash,
        "base_checkpoint_id": metadata.base_checkpoint_id,
        "train_budget_id": metadata.train_budget_id,
        "consumer_mode": metadata.consumer_mode,
        "gate_eval_manifest_hash": metadata.gate_eval_manifest_hash,
    }


def _require_reuse_match(
    *,
    output_dir: Path,
    variant: str,
    metadata: TrainCheckpointMetadata,
) -> None:
    checkpoint_dir = output_dir / "best"
    required_paths = (
        output_dir / "train_manifest.json",
        output_dir / "checkpoint_provenance.json",
        checkpoint_dir / "checkpoint.json",
    )
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "reuse_existing_checkpoint was requested but the existing checkpoint bundle is incomplete: "
            + ", ".join(missing_paths)
        )
    expected = _metadata_for_reuse_gate(variant=variant, metadata=metadata)
    train_manifest = read_json(output_dir / "train_manifest.json")
    checkpoint_provenance = read_json(output_dir / "checkpoint_provenance.json")
    for source_name, payload in (
        ("train_manifest", train_manifest),
        ("checkpoint_provenance", checkpoint_provenance),
    ):
        for field_name, expected_value in expected.items():
            observed_value = payload.get(field_name)
            if observed_value != expected_value:
                raise ValueError(
                    "checkpoint reuse rejected: "
                    + f"{source_name}.{field_name} mismatch; "
                    + f"expected {expected_value!r}, got {observed_value!r}"
                )


def _build_checkpoint_metadata(
    *,
    variant_name: str,
    dataset_route_id: str,
    dataset_fingerprint: str,
    episode_universe_hash: str,
    consumer_mode: str,
    gate_eval_manifest_hash: str,
    reuse_existing_checkpoint: bool,
    reuse_verdict: str,
) -> TrainCheckpointMetadata:
    return TrainCheckpointMetadata(
        variant_name=variant_name,
        dataset_route_id=dataset_route_id,
        dataset_fingerprint=dataset_fingerprint,
        episode_universe_hash=episode_universe_hash,
        base_checkpoint_id=BASE_CHECKPOINT_ID,
        train_budget_id=TRAIN_BUDGET_ID,
        consumer_mode=consumer_mode,
        gate_eval_manifest_hash=gate_eval_manifest_hash,
        reuse_existing_checkpoint=reuse_existing_checkpoint,
        reuse_verdict=reuse_verdict,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    raw_variant = _require_str_arg(args, "variant").strip()
    output_dir = Path(_require_str_arg(args, "output_dir")).resolve()
    raw_dataset_dir = cast(object, getattr(args, "dataset_dir"))
    raw_gate_eval_manifest = cast(object, getattr(args, "gate_eval_manifest"))
    task_suite_name = _require_str_arg(args, "task_suite_name")
    task_ids = _require_str_arg(args, "task_ids")
    seeds = _require_str_arg(args, "seeds")
    num_trials_per_task = _require_int_arg(args, "num_trials_per_task")
    reuse_existing_checkpoint = bool(
        cast(object, getattr(args, "reuse_existing_checkpoint"))
    )

    if raw_variant == RECAP_ONLY_VARIANT:
        variant = validate_recap_only_variant(raw_variant)
        variant_name = RECAP_ONLY_VARIANT_NAME
        checkpoint_source = RECAP_ONLY_CHECKPOINT_SOURCE
        consumer_mode = RECAP_RELABEL_CONSUMER_MODE
        runtime_dir = build_train_runtime_dir(output_dir, variant=variant)
        dataset_bundle = (
            resolve_recap_dataset(
                raw_dataset_dir.strip(),
                consumer_mode=consumer_mode,
            )
            if isinstance(raw_dataset_dir, str) and raw_dataset_dir.strip()
            else resolve_default_recap_dataset(consumer_mode=consumer_mode)
        )
    elif raw_variant == FIXEDADV_CONTROL_VARIANT:
        variant = FIXEDADV_CONTROL_VARIANT
        variant_name = FIXEDADV_CONTROL_VARIANT_NAME
        checkpoint_source = FIXEDADV_CONTROL_CHECKPOINT_SOURCE
        consumer_mode = FIXEDADV_CONSTANT_CONSUMER_MODE
        runtime_dir = build_train_runtime_dir(output_dir, variant=variant)
        dataset_bundle = (
            resolve_recap_dataset(
                raw_dataset_dir.strip(),
                consumer_mode=consumer_mode,
            )
            if isinstance(raw_dataset_dir, str) and raw_dataset_dir.strip()
            else resolve_default_recap_dataset(consumer_mode=consumer_mode)
        )
    elif raw_variant == RECAP_STATE_TOKENS_VARIANT:
        variant = validate_state_token_variant(raw_variant)
        variant_name = RECAP_STATE_TOKENS_VARIANT_NAME
        checkpoint_source = (
            "repo_local_openpi_recap_state_tokens_native_discrete_state_input_v1"
        )
        consumer_mode = RECAP_RELABEL_CONSUMER_MODE
        runtime_dir = build_state_token_train_runtime_dir(output_dir, variant=variant)
        try:
            _ = require_control_parity_ready(
                output_dir / "best",
                stage="train_preflight",
            )
            dataset_dir = None
            if isinstance(raw_dataset_dir, str) and raw_dataset_dir.strip():
                dataset_dir = raw_dataset_dir.strip()
            dataset_bundle = (
                resolve_default_state_token_dataset()
                if dataset_dir is None
                else resolve_state_token_dataset(dataset_dir)
            )
        except StateTokenContractError as exc:
            runtime_dir.mkdir(parents=True, exist_ok=True)
            log_path = runtime_dir / "train.log"
            _clear_state_token_success_artifacts(output_dir)
            _write_blocker_reports(output_dir, exc.payload)
            _log(str(exc), log_path=log_path)
            _log(
                f"blocker_report={output_dir / 'blocker_report.json'}",
                log_path=log_path,
            )
            _log("LIBERO_RECAP_TRAIN_BLOCKED", log_path=log_path)
            return 2
    elif raw_variant == SHUFFLED_ADV_DIAG_VARIANT:
        variant = SHUFFLED_ADV_DIAG_VARIANT
        variant_name = SHUFFLED_ADV_DIAG_VARIANT_NAME
        checkpoint_source = SHUFFLED_ADV_DIAG_CHECKPOINT_SOURCE
        consumer_mode = SHUFFLED_ADV_DIAG_CONSUMER_MODE
        runtime_dir = build_train_runtime_dir(output_dir, variant=variant)
        dataset_bundle = (
            resolve_recap_dataset(
                raw_dataset_dir.strip(),
                consumer_mode=consumer_mode,
            )
            if isinstance(raw_dataset_dir, str) and raw_dataset_dir.strip()
            else resolve_default_recap_dataset(consumer_mode=consumer_mode)
        )
    else:
        raise ValueError(
            "unsupported --variant "
            + f"{raw_variant!r}; expected one of {(FIXEDADV_CONTROL_VARIANT, RECAP_ONLY_VARIANT, RECAP_STATE_TOKENS_VARIANT, SHUFFLED_ADV_DIAG_VARIANT)!r}"
        )

    log_path = runtime_dir / "train.log"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    gate_eval_manifest_path = _resolve_gate_eval_manifest_path(raw_gate_eval_manifest)
    gate_eval_manifest_hash = _sha256_file(gate_eval_manifest_path)
    dataset_route_id, dataset_fingerprint, episode_universe_hash = (
        _load_dataset_gate_metadata(dataset_bundle.dataset_dir)
    )

    manifest = build_frozen_comparison_manifest(
        suite=task_suite_name,
        task_ids=task_ids,
        seed_manifest=seeds,
        num_trials_per_task=num_trials_per_task,
        gate_eval_manifest=gate_eval_manifest_path,
    )
    _log(
        f"[{dt.datetime.now().isoformat(timespec='seconds')}] resolving compatible recap dataset for {variant}",
        log_path=log_path,
    )
    _log(f"selected dataset: {dataset_bundle.dataset_dir}", log_path=log_path)
    _log(
        f"gate_eval_manifest={gate_eval_manifest_path} gate_eval_manifest_hash={gate_eval_manifest_hash}",
        log_path=log_path,
    )
    _log(
        f"reuse_existing_checkpoint={str(reuse_existing_checkpoint).lower()}",
        log_path=log_path,
    )
    if reuse_existing_checkpoint:
        _require_reuse_match(
            output_dir=output_dir,
            variant=variant,
            metadata=_build_checkpoint_metadata(
                variant_name=variant_name,
                dataset_route_id=dataset_route_id,
                dataset_fingerprint=dataset_fingerprint,
                episode_universe_hash=episode_universe_hash,
                consumer_mode=consumer_mode,
                gate_eval_manifest_hash=gate_eval_manifest_hash,
                reuse_existing_checkpoint=False,
                reuse_verdict=REUSE_VERDICT_NEW,
            ),
        )
    train_metadata = _build_checkpoint_metadata(
        variant_name=variant_name,
        dataset_route_id=dataset_route_id,
        dataset_fingerprint=dataset_fingerprint,
        episode_universe_hash=episode_universe_hash,
        consumer_mode=consumer_mode,
        gate_eval_manifest_hash=gate_eval_manifest_hash,
        reuse_existing_checkpoint=reuse_existing_checkpoint,
        reuse_verdict=(
            REUSE_VERDICT_REUSED if reuse_existing_checkpoint else REUSE_VERDICT_NEW
        ),
    )
    real_checkpoint_source_dir: Path | None = None
    if variant in {
        RECAP_ONLY_VARIANT,
        FIXEDADV_CONTROL_VARIANT,
        SHUFFLED_ADV_DIAG_VARIANT,
    }:
        try:
            real_checkpoint_source_dir = _run_real_variant_training_export_or_block(
                variant=variant,
                variant_name=variant_name,
                dataset_dir=dataset_bundle.dataset_dir,
                runtime_dir=runtime_dir,
                consumer_mode=consumer_mode,
                fixed_indicator_mode=getattr(
                    dataset_bundle, "fixed_indicator_mode", None
                ),
                output_dir=output_dir,
                log_path=log_path,
            )
        except RealVariantExportBlockedError:
            _log("LIBERO_RECAP_TRAIN_BLOCKED", log_path=log_path)
            return 2
    if variant == RECAP_ONLY_VARIANT:
        checkpoint_bundle = materialize_recap_checkpoint(
            output_dir=output_dir,
            dataset_bundle=cast(RecapDatasetBundle, dataset_bundle),
            manifest=manifest,
            variant=variant,
            checkpoint_source=checkpoint_source,
            train_metadata=train_metadata,
            serveable_checkpoint_source_dir=real_checkpoint_source_dir,
        )
        _require_local_orbax_checkpoint_assets(checkpoint_bundle.checkpoint_dir)
    elif variant == FIXEDADV_CONTROL_VARIANT:
        checkpoint_bundle = materialize_recap_checkpoint(
            output_dir=output_dir,
            dataset_bundle=cast(RecapDatasetBundle, dataset_bundle),
            manifest=manifest,
            variant=variant,
            checkpoint_source=checkpoint_source,
            train_metadata=train_metadata,
            serveable_checkpoint_source_dir=real_checkpoint_source_dir,
        )
        _require_local_orbax_checkpoint_assets(checkpoint_bundle.checkpoint_dir)
    elif variant == SHUFFLED_ADV_DIAG_VARIANT:
        checkpoint_bundle = materialize_recap_checkpoint(
            output_dir=output_dir,
            dataset_bundle=cast(RecapDatasetBundle, dataset_bundle),
            manifest=manifest,
            variant=variant,
            checkpoint_source=checkpoint_source,
            train_metadata=train_metadata,
            serveable_checkpoint_source_dir=real_checkpoint_source_dir,
        )
        _require_local_orbax_checkpoint_assets(checkpoint_bundle.checkpoint_dir)
    else:
        checkpoint_bundle = materialize_state_token_checkpoint(
            output_dir=output_dir,
            dataset_bundle=cast(StateTokenDatasetBundle, dataset_bundle),
            manifest=manifest,
            train_metadata=train_metadata,
        )
    _clear_blocker_reports(output_dir)
    summary = {
        "schema_version": "openpi_libero_recap_train_run_v1",
        "variant": variant,
        "variant_name": variant_name,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_bundle.checkpoint_dir),
        "runtime_dir": str(runtime_dir),
        "train_manifest": str(checkpoint_bundle.train_manifest_path),
        "checkpoint_provenance": str(checkpoint_bundle.checkpoint_provenance_path),
        "dataset_dir": str(dataset_bundle.dataset_dir),
        "dataset_route_id": dataset_route_id,
        "dataset_fingerprint": dataset_fingerprint,
        "episode_universe_hash": episode_universe_hash,
        "base_checkpoint_id": BASE_CHECKPOINT_ID,
        "train_budget_id": TRAIN_BUDGET_ID,
        "consumer_mode": consumer_mode,
        "gate_eval_manifest_hash": gate_eval_manifest_hash,
        "reuse_existing_checkpoint": reuse_existing_checkpoint,
        "reuse_verdict": train_metadata.reuse_verdict,
        "record_count": int(dataset_bundle.total_rows),
    }
    write_json(runtime_dir / "summary.json", summary)
    _log(f"checkpoint ready: {checkpoint_bundle.checkpoint_dir}", log_path=log_path)
    _log("LIBERO_RECAP_TRAIN_DONE", log_path=log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
