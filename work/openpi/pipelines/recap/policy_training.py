#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.checkpoint import materialize_recap_checkpoint, write_json
from work.openpi.recap.checkpoint_provenance import (
    annotate_stage_checkpoint_artifacts,
)
from work.openpi.recap.data_transforms import (
    DEFAULT_SOURCE_SMOKE_EPISODE_LIMIT,
    prepare_stage_training_dataset,
    resolve_prebuilt_training_dataset_dir,
    resolve_prepare_episode_limit,
)
from work.openpi.recap.protocol import (
    build_frozen_comparison_manifest,
    build_train_runtime_dir,
)
from work.openpi.recap.real_variant_export import (
    RealVariantExportBlockedError,
    RealVariantExportRequest,
    run_real_variant_training_export,
)
from work.openpi.recap.train_config import (
    build_stage_train_metadata,
    resolve_repaired_stage_config,
    resolve_train_scope,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libero_recap_train.py",
        description=(
            "Train repaired OpenPI-compatible RECAP stages using the task-6 canonical carrier authority."
        ),
    )
    _ = parser.add_argument("--stage", required=True)
    _ = parser.add_argument(
        "--dataset-dir",
        required=True,
        help=(
            "Official/native 8D source dataset or an already-materialized recap-ready dataset."
        ),
    )
    _ = parser.add_argument(
        "--critic-checkpoint",
        required=True,
        help="Critic checkpoint dir used by task-6 canonical relabel materialization.",
    )
    _ = parser.add_argument("--output-dir", required=True)
    _ = parser.add_argument(
        "--gate-eval-manifest",
        default=None,
        help="Optional tracked rollout eval manifest; defaults to eval_manifest_rollout_lite_v2.json.",
    )
    _ = parser.add_argument("--task-suite-name", default=None)
    _ = parser.add_argument("--task-ids", default=None)
    _ = parser.add_argument("--seeds", default=None)
    _ = parser.add_argument("--num-trials-per-task", type=int, default=None)
    _ = parser.add_argument(
        "--episode-limit",
        type=int,
        default=None,
        help="Optional contiguous prefix episode count for test-only source materialization.",
    )
    _ = parser.add_argument(
        "--prepared-dataset-dir",
        default=None,
        help="Optional explicit cache dir for the recap-ready dataset prepared from the official source.",
    )
    return parser


def _log(message: str, *, log_path: Path) -> None:
    print(message, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        _ = handle.write(message)
        _ = handle.write("\n")


def _write_blocker_report(output_dir: Path, payload: dict[str, object]) -> None:
    write_json(output_dir / "blocker_report.json", payload)
    write_json(output_dir / "best" / "blocker_report.json", payload)


def _clear_blocker_report(output_dir: Path) -> None:
    for path in (
        output_dir / "blocker_report.json",
        output_dir / "best" / "blocker_report.json",
    ):
        if path.is_file():
            path.unlink()


@contextmanager
def _task7_train_lock(runtime_root: Path):
    lock_path = runtime_root / "task7_recap_train.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _require_str_arg(args: argparse.Namespace, name: str) -> str:
    value = cast(object, getattr(args, name))
    if not isinstance(value, str):
        raise TypeError(f"argument {name} must be a string")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stage_config = resolve_repaired_stage_config(_require_str_arg(args, "stage"))
    dataset_dir = Path(_require_str_arg(args, "dataset_dir")).expanduser().resolve()
    critic_checkpoint = (
        Path(_require_str_arg(args, "critic_checkpoint")).expanduser().resolve()
    )
    output_dir = Path(_require_str_arg(args, "output_dir")).expanduser().resolve()
    prepared_dataset_dir_raw = cast(object, getattr(args, "prepared_dataset_dir"))
    gate_eval_manifest_raw = cast(object, getattr(args, "gate_eval_manifest"))
    episode_limit = cast(int | None, getattr(args, "episode_limit"))
    runtime_dir = build_train_runtime_dir(output_dir, variant=stage_config.stage)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "train.log"
    runtime_root = runtime_dir.parent
    _log(
        (
            f"[{dt.datetime.now().isoformat(timespec='seconds')}] stage={stage_config.stage} "
            + f"consumer_mode={stage_config.consumer_mode} fixed_indicator_mode={stage_config.fixed_indicator_mode or ''}"
        ),
        log_path=log_path,
    )
    scope = resolve_train_scope(
        gate_eval_manifest=(
            gate_eval_manifest_raw if isinstance(gate_eval_manifest_raw, str) else None
        ),
        suite=cast(str | None, getattr(args, "task_suite_name")),
        task_ids=cast(str | None, getattr(args, "task_ids")),
        seeds=cast(str | None, getattr(args, "seeds")),
        num_trials_per_task=cast(int | None, getattr(args, "num_trials_per_task")),
    )
    _log(
        (
            f"gate_eval_manifest={scope.gate_eval_manifest_path} "
            + f"gate_eval_manifest_hash={scope.gate_eval_manifest_hash}"
        ),
        log_path=log_path,
    )
    _log(
        (
            f"preparing_dataset source={dataset_dir} episode_limit="
            + (
                str(episode_limit)
                if episode_limit is not None
                else (
                    "bypass:delta_prepared_merged_surface"
                    if (
                        resolve_prebuilt_training_dataset_dir(dataset_dir)
                        not in {None, dataset_dir}
                    )
                    else (
                        "auto:full"
                        if resolve_prepare_episode_limit(dataset_dir, None) is None
                        else f"auto:{DEFAULT_SOURCE_SMOKE_EPISODE_LIMIT}"
                    )
                )
            )
        ),
        log_path=log_path,
    )
    _log(
        f"waiting_for_task7_train_lock root={runtime_root / 'task7_recap_train.lock'}",
        log_path=log_path,
    )
    with _task7_train_lock(runtime_root) as lock_path:
        _log(f"acquired_task7_train_lock path={lock_path}", log_path=log_path)
        _clear_blocker_report(output_dir)
        prepared_dataset = prepare_stage_training_dataset(
            dataset_dir=dataset_dir,
            stage_config=stage_config,
            critic_checkpoint_dir=critic_checkpoint,
            prepared_dataset_dir=(
                prepared_dataset_dir_raw
                if isinstance(prepared_dataset_dir_raw, str)
                and prepared_dataset_dir_raw.strip()
                else None
            ),
            episode_limit=episode_limit,
        )
        _log(
            (
                f"prepared_dataset={prepared_dataset.dataset_dir} "
                + f"prepared_from_source={str(prepared_dataset.prepared_from_source).lower()}"
            ),
            log_path=log_path,
        )
        manifest = build_frozen_comparison_manifest(
            suite=scope.suite,
            task_ids=scope.task_ids,
            seed_manifest=scope.seeds,
            num_trials_per_task=scope.num_trials_per_task,
            gate_eval_manifest=scope.gate_eval_manifest_path,
        )
        train_metadata = build_stage_train_metadata(
            stage_config=stage_config,
            dataset_dir=prepared_dataset.dataset_dir,
            gate_eval_manifest_hash=scope.gate_eval_manifest_hash,
        )
        try:
            export_bundle = run_real_variant_training_export(
                RealVariantExportRequest(
                    variant=stage_config.stage,
                    variant_name=stage_config.variant_name,
                    dataset_dir=prepared_dataset.dataset_dir,
                    runtime_dir=runtime_dir,
                    consumer_mode=stage_config.consumer_mode,
                    fixed_indicator_mode=stage_config.fixed_indicator_mode,
                    default_num_train_steps=stage_config.default_num_train_steps,
                    default_save_interval=stage_config.default_save_interval,
                )
            )
        except RealVariantExportBlockedError as exc:
            _write_blocker_report(output_dir, exc.payload)
            _log(str(exc), log_path=log_path)
            _log(
                f"blocker_report={output_dir / 'blocker_report.json'}",
                log_path=log_path,
            )
            _log("LIBERO_RECAP_TRAIN_BLOCKED", log_path=log_path)
            return 2
        _log(
            f"real_variant_export={export_bundle.export_dir} runtime_log={export_bundle.runtime_log_path}",
            log_path=log_path,
        )
        checkpoint_bundle = materialize_recap_checkpoint(
            output_dir=output_dir,
            dataset_bundle=prepared_dataset.dataset_bundle,
            manifest=manifest,
            variant=stage_config.stage,
            checkpoint_source=stage_config.checkpoint_source,
            train_metadata=train_metadata,
            serveable_checkpoint_source_dir=export_bundle.export_dir,
        )
        stage_provenance = annotate_stage_checkpoint_artifacts(
            checkpoint_bundle=checkpoint_bundle,
            export_dir=export_bundle.export_dir,
            dataset_bundle=prepared_dataset.dataset_bundle,
            stage_config=stage_config,
            critic_checkpoint_ref=str(critic_checkpoint),
            source_dataset_dir=prepared_dataset.source_dataset_dir,
            prepared_dataset_dir=prepared_dataset.dataset_dir,
            materialization_report_path=prepared_dataset.materialization_report_path,
        )
        summary = {
            "schema_version": "openpi_libero_recap_train_run_v2",
            "stage": stage_config.stage,
            "variant_name": stage_config.variant_name,
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(output_dir),
            "checkpoint_dir": str(checkpoint_bundle.checkpoint_dir),
            "runtime_dir": str(runtime_dir),
            "train_manifest": str(checkpoint_bundle.train_manifest_path),
            "checkpoint_provenance": str(checkpoint_bundle.checkpoint_provenance_path),
            "export_manifest": str(export_bundle.export_dir / "export_manifest.json"),
            "dataset_dir": str(dataset_dir),
            "prepared_dataset_dir": str(prepared_dataset.dataset_dir),
            "prepared_from_source": bool(prepared_dataset.prepared_from_source),
            "critic_checkpoint_ref": stage_provenance.critic_checkpoint_ref,
            "indicator_mode_train": stage_provenance.indicator_mode_train,
            "indicator_dropout_p": float(stage_provenance.indicator_dropout_p),
            "epsilon_source": stage_provenance.epsilon_source,
            "human_correction_override": bool(
                stage_provenance.human_correction_override
            ),
            "gate_eval_manifest_hash": scope.gate_eval_manifest_hash,
            "record_count": int(prepared_dataset.dataset_bundle.total_rows),
        }
        write_json(runtime_dir / "summary.json", summary)
        _clear_blocker_report(output_dir)
        _log(f"checkpoint ready: {checkpoint_bundle.checkpoint_dir}", log_path=log_path)
        _log(
            (
                f"stage_provenance indicator_mode_train={stage_provenance.indicator_mode_train} "
                + f"epsilon_source={stage_provenance.epsilon_source}"
            ),
            log_path=log_path,
        )
        _log("LIBERO_RECAP_TRAIN_DONE", log_path=log_path)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
