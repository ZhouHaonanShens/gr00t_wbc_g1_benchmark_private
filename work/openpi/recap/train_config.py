from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from work.openpi.eval.protocols.tracked_gate import load_rollout_eval_manifest_v2

from .checkpoint import TrainCheckpointMetadata
from .indicator_modes import (
    DEFAULT_INDICATOR_DROPOUT_P,
    INDICATOR_MODE_TRAIN_FIXED_POSITIVE,
    INDICATOR_MODE_TRAIN_INFORMATIVE,
    INDICATOR_MODE_TRAIN_OMIT,
    INDICATOR_MODE_TRAIN_SHUFFLED,
)
from .protocol import REPO_ROOT


DEFAULT_GATE_EVAL_MANIFEST_PATH = (
    REPO_ROOT
    / "work"
    / "openpi"
    / "eval"
    / "manifests"
    / "eval_manifest_rollout_lite_v2.json"
)
REUSE_VERDICT_NEW = "materialize_new_checkpoint"
TRAIN_BUDGET_ID = "libero_cmp_budget_v2"
BASE_CHECKPOINT_ID = "pi05_libero_anchor"
DEFAULT_REPAIRED_STAGE_NUM_TRAIN_STEPS = 1
RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS = 24
DEFAULT_REPAIRED_STAGE_SAVE_INTERVAL = 1
RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL = RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
STAGE_SFT_FIXED_POSITIVE = "sft_fixed_positive"
STAGE_RECAP_INFORMATIVE = "recap_informative"
STAGE_SHUFFLED_INDICATOR = "shuffled_indicator"
STAGE_OMIT_CONTROL = "omit_control"
REPAIRED_STAGE_VALUES: tuple[str, ...] = (
    STAGE_SFT_FIXED_POSITIVE,
    STAGE_RECAP_INFORMATIVE,
    STAGE_SHUFFLED_INDICATOR,
    STAGE_OMIT_CONTROL,
)


@dataclass(frozen=True)
class RepairedStageConfig:
    stage: str
    variant_name: str
    checkpoint_source: str
    consumer_mode: str
    fixed_indicator_mode: str | None
    indicator_mode_train: str
    indicator_dropout_p: float = DEFAULT_INDICATOR_DROPOUT_P
    human_correction_override: bool = True
    default_num_train_steps: int = DEFAULT_REPAIRED_STAGE_NUM_TRAIN_STEPS
    default_save_interval: int = DEFAULT_REPAIRED_STAGE_SAVE_INTERVAL


@dataclass(frozen=True)
class ResolvedTrainScope:
    gate_eval_manifest_path: Path
    gate_eval_manifest_hash: str
    suite: str
    task_ids: str
    seeds: str
    num_trials_per_task: int


REPAIRED_STAGE_CONFIGS: dict[str, RepairedStageConfig] = {
    STAGE_SFT_FIXED_POSITIVE: RepairedStageConfig(
        stage=STAGE_SFT_FIXED_POSITIVE,
        variant_name=STAGE_SFT_FIXED_POSITIVE,
        checkpoint_source="repo_local_openpi_recap_sft_fixed_positive_v1",
        consumer_mode=INDICATOR_MODE_TRAIN_FIXED_POSITIVE,
        fixed_indicator_mode="positive",
        indicator_mode_train=INDICATOR_MODE_TRAIN_FIXED_POSITIVE,
    ),
    STAGE_RECAP_INFORMATIVE: RepairedStageConfig(
        stage=STAGE_RECAP_INFORMATIVE,
        variant_name=STAGE_RECAP_INFORMATIVE,
        checkpoint_source="repo_local_openpi_recap_informative_v1",
        consumer_mode=INDICATOR_MODE_TRAIN_INFORMATIVE,
        fixed_indicator_mode=None,
        indicator_mode_train=INDICATOR_MODE_TRAIN_INFORMATIVE,
        default_num_train_steps=RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS,
        default_save_interval=RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL,
    ),
    STAGE_SHUFFLED_INDICATOR: RepairedStageConfig(
        stage=STAGE_SHUFFLED_INDICATOR,
        variant_name=STAGE_SHUFFLED_INDICATOR,
        checkpoint_source="repo_local_openpi_recap_shuffled_indicator_v1",
        consumer_mode=INDICATOR_MODE_TRAIN_SHUFFLED,
        fixed_indicator_mode=None,
        indicator_mode_train=INDICATOR_MODE_TRAIN_SHUFFLED,
    ),
    STAGE_OMIT_CONTROL: RepairedStageConfig(
        stage=STAGE_OMIT_CONTROL,
        variant_name=STAGE_OMIT_CONTROL,
        checkpoint_source="repo_local_openpi_recap_omit_control_v1",
        consumer_mode=INDICATOR_MODE_TRAIN_OMIT,
        fixed_indicator_mode="omit",
        indicator_mode_train=INDICATOR_MODE_TRAIN_OMIT,
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _read_required_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_bytes()


def resolve_repaired_stage_config(raw_stage: object) -> RepairedStageConfig:
    if not isinstance(raw_stage, str):
        raise TypeError(f"stage must be a string, got {type(raw_stage).__name__}")
    stage = raw_stage.strip().lower()
    try:
        return REPAIRED_STAGE_CONFIGS[stage]
    except KeyError as exc:
        raise ValueError(
            "unsupported --stage "
            + f"{raw_stage!r}; expected one of {REPAIRED_STAGE_VALUES!r}"
        ) from exc


def resolve_train_scope(
    *,
    gate_eval_manifest: str | Path | None,
    suite: str | None,
    task_ids: str | None,
    seeds: str | None,
    num_trials_per_task: int | None,
) -> ResolvedTrainScope:
    gate_eval_manifest_path = _resolve_path(
        gate_eval_manifest or DEFAULT_GATE_EVAL_MANIFEST_PATH
    )
    if not gate_eval_manifest_path.is_file():
        raise FileNotFoundError(
            "missing gate eval manifest required for repaired recap train: "
            + str(gate_eval_manifest_path)
        )
    gate_manifest = load_rollout_eval_manifest_v2(gate_eval_manifest_path)
    resolved_suite = (
        str(suite).strip()
        if isinstance(suite, str) and suite.strip()
        else str(gate_manifest.task_suite_name)
    )
    resolved_task_ids = (
        task_ids.strip()
        if isinstance(task_ids, str) and task_ids.strip()
        else ",".join(str(value) for value in gate_manifest.task_ids)
    )
    resolved_seeds = (
        seeds.strip()
        if isinstance(seeds, str) and seeds.strip()
        else ",".join(str(value) for value in gate_manifest.seed_manifest)
    )
    resolved_num_trials = (
        int(num_trials_per_task)
        if num_trials_per_task is not None
        else int(gate_manifest.num_trials_per_task)
    )
    return ResolvedTrainScope(
        gate_eval_manifest_path=gate_eval_manifest_path,
        gate_eval_manifest_hash=_sha256_file(gate_eval_manifest_path),
        suite=resolved_suite,
        task_ids=resolved_task_ids,
        seeds=resolved_seeds,
        num_trials_per_task=resolved_num_trials,
    )


def build_dataset_identity(dataset_dir: str | Path) -> tuple[str, str, str]:
    resolved_dir = _resolve_path(dataset_dir)
    meta_dir = resolved_dir / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing dataset info.json under {meta_dir}")
    info_bytes = _read_required_bytes(info_path)
    route_id_marker = b'"route_id"'
    route_id = ""
    if route_id_marker in info_bytes:
        import json

        payload = json.loads(info_bytes.decode("utf-8"))
        route_id = str(payload.get("route_id", "")).strip()
    if not route_id:
        raise ValueError(f"dataset {resolved_dir} missing info.route_id")

    fingerprint_path = meta_dir / "dataset_fingerprint.json"
    if fingerprint_path.is_file():
        import json

        fingerprint_payload = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        dataset_fingerprint = str(
            fingerprint_payload.get("fingerprint_sha256", "")
        ).strip()
        if not dataset_fingerprint:
            raise ValueError(
                f"dataset_fingerprint.json under {meta_dir} is missing fingerprint_sha256"
            )
    else:
        digest = hashlib.sha256()
        for path in (
            meta_dir / "info.json",
            meta_dir / "tasks.jsonl",
            meta_dir / "episodes.jsonl",
            meta_dir / "stats.json",
        ):
            digest.update(_read_required_bytes(path))
        dataset_fingerprint = digest.hexdigest()

    episode_universe_hash_path = meta_dir / "episode_universe_hash.txt"
    if episode_universe_hash_path.is_file():
        episode_universe_hash = episode_universe_hash_path.read_text(
            encoding="utf-8"
        ).strip()
    else:
        episode_universe_hash = hashlib.sha256(
            _read_required_bytes(meta_dir / "episodes.jsonl")
        ).hexdigest()
    if not episode_universe_hash:
        raise ValueError(
            f"dataset {resolved_dir} did not produce a non-empty episode_universe_hash"
        )
    return route_id, dataset_fingerprint, episode_universe_hash


def build_stage_train_metadata(
    *,
    stage_config: RepairedStageConfig,
    dataset_dir: str | Path,
    gate_eval_manifest_hash: str,
    reuse_existing_checkpoint: bool = False,
) -> TrainCheckpointMetadata:
    dataset_route_id, dataset_fingerprint, episode_universe_hash = (
        build_dataset_identity(dataset_dir)
    )
    return TrainCheckpointMetadata(
        variant_name=stage_config.variant_name,
        dataset_route_id=dataset_route_id,
        dataset_fingerprint=dataset_fingerprint,
        episode_universe_hash=episode_universe_hash,
        base_checkpoint_id=BASE_CHECKPOINT_ID,
        train_budget_id=TRAIN_BUDGET_ID,
        consumer_mode=stage_config.consumer_mode,
        gate_eval_manifest_hash=gate_eval_manifest_hash,
        reuse_existing_checkpoint=bool(reuse_existing_checkpoint),
        reuse_verdict=REUSE_VERDICT_NEW,
    )


__all__ = [
    "BASE_CHECKPOINT_ID",
    "DEFAULT_GATE_EVAL_MANIFEST_PATH",
    "DEFAULT_REPAIRED_STAGE_NUM_TRAIN_STEPS",
    "DEFAULT_REPAIRED_STAGE_SAVE_INTERVAL",
    "REPAIRED_STAGE_CONFIGS",
    "REPAIRED_STAGE_VALUES",
    "RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS",
    "RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL",
    "REUSE_VERDICT_NEW",
    "ResolvedTrainScope",
    "RepairedStageConfig",
    "STAGE_OMIT_CONTROL",
    "STAGE_RECAP_INFORMATIVE",
    "STAGE_SFT_FIXED_POSITIVE",
    "STAGE_SHUFFLED_INDICATOR",
    "TRAIN_BUDGET_ID",
    "build_dataset_identity",
    "build_stage_train_metadata",
    "resolve_repaired_stage_config",
    "resolve_train_scope",
]
