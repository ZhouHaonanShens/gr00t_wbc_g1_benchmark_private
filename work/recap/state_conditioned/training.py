from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_TRAINING_SET_ROOT = Path(
    "agent/artifacts/state_conditioned_materialization/training"
)
DEFAULT_OUTPUT_DIR = Path("agent/artifacts/state_conditioned_materialization/training")
DEFAULT_RUNTIME_LOG_DIR = Path("agent/runtime_logs/state_conditioned_train")

DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1"
DEFAULT_MAX_STEPS = 100
DEFAULT_GLOBAL_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_DATALOADER_NUM_WORKERS = 0
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_NUM_GPUS = 1
DEFAULT_SHARD_SIZE = 2**10
DEFAULT_EPISODE_SAMPLING_RATE = 0.1
DEFAULT_NUM_SHARDS_PER_EPOCH = int(1e5)
DEFAULT_SAVE_TOTAL_LIMIT = 1
DEFAULT_SEED = 42
DEFAULT_TUNE_PROJECTOR = False
DEFAULT_TUNE_DIFFUSION_MODEL = False
DEFAULT_USE_WANDB = False

TRAINING_KERNEL_REL = Path("work/recap/scripts/3D_recap_finetune_full.py")
RUN_METADATA_SCHEMA_VERSION = "state_conditioned_training_run_v1"
DIFF_ARTIFACT_SCHEMA_VERSION = "state_conditioned_training_diff_v1"
RUN_METADATA_BASENAME_BY_VARIANT = {
    "c0": "run_metadata_C0_equal_data_control.json",
    "c1": "run_metadata_C1_phase_mode.json",
}
DELEGATE_SUMMARY_BASENAME_BY_VARIANT = {
    "c0": "delegate_summary_C0_equal_data_control.json",
    "c1": "delegate_summary_C1_phase_mode.json",
}
DIFF_WHITELIST_JSON_NAME = "state_conditioned_training_fairness_diff_whitelist.json"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import policy as recap_policy
from work.recap import state_conditioned_build_training_set
from work.recap.run_manifest import INDICATOR_SOURCE_FIELD
from work.recap.run_manifest import PROMPT_SOURCE_FIELD
from work.recap.run_manifest import TEXT_CARRIER_ROUTE
from work.recap.run_manifest import TEXT_CARRIER_SCHEMA_VERSION
from work.recap.scripts.state_conditioned_common import (
    exception_message as _exception_message,
)
from work.recap.scripts.state_conditioned_common import (
    validate_existing_dir as _validate_existing_dir,
)
from work.recap.scripts.state_conditioned_common import (
    validate_existing_file as _validate_existing_file,
)
from work.recap.scripts.state_conditioned_common import write_json as _write_json


MAINLINE_CARRIER_SCHEMA_VERSION = TEXT_CARRIER_SCHEMA_VERSION
MAINLINE_CARRIER_ROUTE = TEXT_CARRIER_ROUTE
MAINLINE_PROMPT_SOURCE_FIELD = PROMPT_SOURCE_FIELD
MAINLINE_INDICATOR_SOURCE_FIELD = INDICATOR_SOURCE_FIELD
MAINLINE_RUNTIME_ROUTE = recap_policy.MAINLINE_RUNTIME_ROUTE
MAINLINE_RUNTIME_POLICY_CLASS = recap_policy.MAINLINE_RUNTIME_POLICY_CLASS_NAME
MAINLINE_RUNTIME_INDICATOR_MODES = recap_policy.MAINLINE_RUNTIME_INDICATOR_MODES


@dataclass(frozen=True)
class VariantConfig:
    key: str
    output_dir_name: str
    conditioning_enabled: bool
    null_phase_mode_token_enabled: bool


VARIANT_CONFIGS: dict[str, VariantConfig] = {
    "c0": VariantConfig(
        key="c0",
        output_dir_name="checkpoint_C0_equal_data_control",
        conditioning_enabled=False,
        null_phase_mode_token_enabled=True,
    ),
    "c1": VariantConfig(
        key="c1",
        output_dir_name="checkpoint_C1_phase_mode",
        conditioning_enabled=True,
        null_phase_mode_token_enabled=False,
    ),
}

ALLOWED_DIFF_PATHS = {
    "conditioning_enabled",
    "null_phase_mode_token_enabled",
    "output_dir",
    "checkpoint_rule.selected_checkpoint_path",
}


def _repo_root() -> Path:
    return REPO_ROOT


def _resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_signature(payload: object) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return _sha256_bytes(canonical.encode("utf-8"))


def _add_bool_group(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    dest: str,
    default: bool,
    help_text: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=dest, action="store_true", help=help_text)
    group.add_argument(
        name.replace("--", "--no-", 1),
        dest=dest,
        action="store_false",
        help=f"Disable {help_text.lower()}",
    )
    parser.set_defaults(**{dest: bool(default)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="state_conditioned_train.py",
        description=(
            "Thin T11 orchestration wrapper: validate the shared T10 training-set root, "
            "freeze one run spec, then launch work/recap/scripts/3D_recap_finetune_full.py for C0 "
            "and/or C1 without creating a new training subsystem."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training-set-root",
        type=Path,
        default=DEFAULT_TRAINING_SET_ROOT,
        help="T10 training-set root containing labels and fairness/liveness audits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Root directory that will contain checkpoint_C0_equal_data_control/ and "
            "checkpoint_C1_phase_mode/."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=("all", "c0", "c1"),
        default="all",
        help="Run both variants serially or only a single variant.",
    )
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-model-revision", type=str, default="")
    parser.add_argument("--hf-hub-cache-dir", type=str, default="")
    parser.add_argument(
        "--patched-out-root",
        type=str,
        default="agent/artifacts/hf_patches",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="",
        help="Python executable forwarded to 3D_recap_finetune_full.py for upstream finetune.",
    )
    parser.add_argument("--embodiment-tag", type=str, default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="If unset, this wrapper freezes save_steps to max_steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=DEFAULT_SAVE_TOTAL_LIMIT,
        help="Must remain 1 for T11 fairness discipline.",
    )
    parser.add_argument(
        "--runtime-log-dir",
        type=Path,
        default=DEFAULT_RUNTIME_LOG_DIR,
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=DEFAULT_GLOBAL_BATCH_SIZE,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=DEFAULT_DATALOADER_NUM_WORKERS,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument(
        "--episode-sampling-rate",
        type=float,
        default=DEFAULT_EPISODE_SAMPLING_RATE,
    )
    parser.add_argument(
        "--num-shards-per-epoch",
        type=int,
        default=DEFAULT_NUM_SHARDS_PER_EPOCH,
    )
    _add_bool_group(
        parser,
        name="--tune-projector",
        dest="tune_projector",
        default=DEFAULT_TUNE_PROJECTOR,
        help_text="Tune the projector module during finetuning.",
    )
    _add_bool_group(
        parser,
        name="--tune-diffusion-model",
        dest="tune_diffusion_model",
        default=DEFAULT_TUNE_DIFFUSION_MODEL,
        help_text="Tune the diffusion model during finetuning.",
    )
    _add_bool_group(
        parser,
        name="--use-wandb",
        dest="use_wandb",
        default=DEFAULT_USE_WANDB,
        help_text="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate T10 artifacts and emit machine-readable metadata without starting training.",
    )
    return parser


def _resolve_contract_path(
    training_set_root: Path, raw_path: object, *, field_name: str
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"T10 {field_name} must be a non-empty path string")
    candidate = Path(raw_path.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (training_set_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _selected_checkpoint_asset(checkpoint_dir: Path | None) -> Path | None:
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return None
    candidates = [
        checkpoint_dir / "model.safetensors.index.json",
        checkpoint_dir / "model.safetensors",
        checkpoint_dir / "pytorch_model.bin.index.json",
        checkpoint_dir / "pytorch_model.bin",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _retained_checkpoint_dirs(output_dir: Path) -> list[Path]:
    checkpoints: list[Path] = []
    for path in sorted(output_dir.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        suffix = path.name.split("checkpoint-", 1)[-1]
        if not suffix.isdigit():
            continue
        if _selected_checkpoint_asset(path) is None:
            continue
        checkpoints.append(path.resolve())
    return checkpoints


def _ensure_single_retained_checkpoint(output_dir: Path) -> Path:
    retained = _retained_checkpoint_dirs(output_dir)
    if len(retained) != 1:
        raise ValueError(
            "expected exactly one retained checkpoint in "
            + f"{output_dir}, found {len(retained)}"
        )
    return retained[0]


def _load_training_set_contract(training_set_root: Path) -> dict[str, Any]:
    root = _validate_existing_dir(training_set_root, arg_name="training-set-root")
    labels_path = _validate_existing_file(
        root
        / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_LABELS_JSONL_NAME,
        arg_name="T10 state_conditioned_sft_labels.jsonl",
    )
    stats_path = _validate_existing_file(
        root
        / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_STATS_JSON_NAME,
        arg_name="T10 state_conditioned_sft_stats.json",
    )
    fairness_path = _validate_existing_file(
        root / state_conditioned_build_training_set.EQUAL_DATA_FAIRNESS_AUDIT_JSON_NAME,
        arg_name="T10 equal_data_fairness_audit.json",
    )
    liveness_path = _validate_existing_file(
        root
        / state_conditioned_build_training_set.CONDITIONING_CHANNEL_LIVENESS_JSON_NAME,
        arg_name="T10 conditioning_channel_liveness.json",
    )
    gate_path = _validate_existing_file(
        root / state_conditioned_build_training_set.DEV_ONLY_PROMOTION_GATE_JSON_NAME,
        arg_name="T10 dev_only_promotion_gate.json",
    )

    labels = _read_jsonl(labels_path)
    if not labels:
        raise ValueError("T10 state_conditioned_sft_labels.jsonl is empty")
    stats = _read_json(stats_path)
    fairness = _read_json(fairness_path)
    liveness = _read_json(liveness_path)
    gate = _read_json(gate_path)

    if not bool(fairness.get("overall_pass", False)):
        raise ValueError(
            "T11 requires T10 equal_data_fairness_audit.json overall_pass == true"
        )
    if not bool(liveness.get("overall_pass", False)):
        raise ValueError(
            "T11 requires T10 conditioning_channel_liveness.json overall_pass == true"
        )
    if int(stats.get("recovery_oversample_factor_min", -1)) != int(
        state_conditioned_build_training_set.RECOVERY_OVERSAMPLE_FACTOR
    ) or int(stats.get("recovery_oversample_factor_max", -1)) != int(
        state_conditioned_build_training_set.RECOVERY_OVERSAMPLE_FACTOR
    ):
        raise ValueError(
            "T10 recovery_oversample_factor drifted from the frozen value 3"
        )

    c0_rows = [
        row
        for row in labels
        if row.get("training_view") == state_conditioned_build_training_set.VIEW_C0
    ]
    c1_rows = [
        row
        for row in labels
        if row.get("training_view") == state_conditioned_build_training_set.VIEW_C1
    ]
    if not c0_rows or not c1_rows:
        raise ValueError("T10 labels must contain both C0 and C1 rows for T11")
    if len(c0_rows) != len(c1_rows):
        raise ValueError("T10 labels have mismatched C0/C1 row counts")

    c0_sample_ids = [str(row["sample_id"]) for row in c0_rows]
    c1_sample_ids = [str(row["sample_id"]) for row in c1_rows]
    if c0_sample_ids != c1_sample_ids:
        raise ValueError(
            "T10 labels lost equal-data sample alignment between C0 and C1"
        )

    for row in c0_rows:
        state_conditioned_build_training_set.validate_view_policy_condition_text(
            training_view=str(row["training_view"]),
            phase=row["policy_condition.phase"],
            mode=row["policy_condition.mode"],
            policy_condition_text=row["policy_condition_text"],
        )
    for row in c1_rows:
        state_conditioned_build_training_set.validate_view_policy_condition_text(
            training_view=str(row["training_view"]),
            phase=row["policy_condition.phase"],
            mode=row["policy_condition.mode"],
            policy_condition_text=row["policy_condition_text"],
        )

    stats_views = dict(stats.get("views", {}))
    c0_view = dict(stats_views.get(state_conditioned_build_training_set.VIEW_C0, {}))
    c1_view = dict(stats_views.get(state_conditioned_build_training_set.VIEW_C1, {}))
    if str(c0_view.get("sample_ids_hash", "")) != str(
        c1_view.get("sample_ids_hash", "")
    ):
        raise ValueError("T10 stats show different C0/C1 sample_ids hashes")

    lerobot_root = _resolve_contract_path(
        root,
        stats.get("lerobot_dataset_path"),
        field_name="state_conditioned_sft_stats.json.lerobot_dataset_path",
    )
    lerobot_info_path = _validate_existing_file(
        lerobot_root / "meta" / "info.json",
        arg_name="T10 lerobot_dataset_path/meta/info.json",
    )

    file_hashes = {
        "state_conditioned_sft_labels_jsonl": _sha256_file(labels_path),
        "state_conditioned_sft_stats_json": _sha256_file(stats_path),
        "equal_data_fairness_audit_json": _sha256_file(fairness_path),
        "conditioning_channel_liveness_json": _sha256_file(liveness_path),
        "dev_only_promotion_gate_json": _sha256_file(gate_path),
        "lerobot_meta_info_json": _sha256_file(lerobot_info_path),
    }
    dataset_fingerprint = _stable_signature(
        {
            "root": str(lerobot_root),
            "file_hashes": file_hashes,
            "c0_rows": len(c0_rows),
            "c1_rows": len(c1_rows),
        }
    )
    return {
        "training_set_root": str(root),
        "dataset_path": str(lerobot_root),
        "dataset_fingerprint": dataset_fingerprint,
        "file_hashes": file_hashes,
        "artifact_paths": {
            "state_conditioned_sft_labels_path": str(labels_path),
            "state_conditioned_sft_stats_path": str(stats_path),
            "equal_data_fairness_audit_path": str(fairness_path),
            "conditioning_channel_liveness_path": str(liveness_path),
            "dev_only_promotion_gate_path": str(gate_path),
            "lerobot_dataset_path": str(lerobot_root),
            "lerobot_dataset_info_path": str(lerobot_info_path),
        },
        "counts": {
            "c0_rows": int(len(c0_rows)),
            "c1_rows": int(len(c1_rows)),
            "unified_base_row_count": int(
                stats.get("counts", {}).get("unified_base_row_count", 0)
            ),
        },
        "deployable_observation_allowlist": list(
            stats.get("deployable_observation_allowlist", [])
        ),
        "shared_sample_ids_hash": str(c0_view.get("sample_ids_hash", "")),
        "dev_only_promotion_allowed": bool(gate.get("promotion_allowed", False)),
    }


def _save_steps(args: argparse.Namespace) -> int:
    save_steps = (
        int(args.max_steps) if args.save_steps is None else int(args.save_steps)
    )
    if save_steps <= 0:
        raise ValueError(f"--save-steps must be > 0, got {args.save_steps!r}")
    if int(args.max_steps) <= 0:
        raise ValueError(f"--max-steps must be > 0, got {args.max_steps!r}")
    if int(args.save_total_limit) != int(DEFAULT_SAVE_TOTAL_LIMIT):
        raise ValueError(
            "T11 enforces single-checkpoint retention: "
            + f"expected --save-total-limit={DEFAULT_SAVE_TOTAL_LIMIT}, got {args.save_total_limit}"
        )
    return save_steps


def _build_mainline_training_route() -> dict[str, Any]:
    return {
        "carrier_route": MAINLINE_CARRIER_ROUTE,
        "carrier_schema_version": MAINLINE_CARRIER_SCHEMA_VERSION,
        "prompt_source_field": MAINLINE_PROMPT_SOURCE_FIELD,
        "indicator_source": MAINLINE_INDICATOR_SOURCE_FIELD,
        "runtime_route": MAINLINE_RUNTIME_ROUTE,
        "runtime_policy_class": MAINLINE_RUNTIME_POLICY_CLASS,
        "runtime_indicator_mode_required": True,
        "runtime_supported_indicator_modes": list(MAINLINE_RUNTIME_INDICATOR_MODES),
        "mainline_authority": True,
        "diagnostic_only": False,
    }


def _build_shared_run_spec(
    *,
    args: argparse.Namespace,
    training_set_contract: Mapping[str, Any],
    forwarded: Sequence[str],
) -> dict[str, Any]:
    save_steps = _save_steps(args)
    return {
        "dataset_path": str(training_set_contract["dataset_path"]),
        "dataset_fingerprint": str(training_set_contract["dataset_fingerprint"]),
        "carrier_schema_version": MAINLINE_CARRIER_SCHEMA_VERSION,
        "carrier_route": MAINLINE_CARRIER_ROUTE,
        "prompt_source_field": MAINLINE_PROMPT_SOURCE_FIELD,
        "indicator_source": MAINLINE_INDICATOR_SOURCE_FIELD,
        "training_route": _build_mainline_training_route(),
        "source_data": {
            **dict(training_set_contract["artifact_paths"]),
            **dict(training_set_contract["file_hashes"]),
            "shared_sample_ids_hash": str(
                training_set_contract["shared_sample_ids_hash"]
            ),
            "unified_base_row_count": int(
                training_set_contract["counts"]["unified_base_row_count"]
            ),
            "c0_rows": int(training_set_contract["counts"]["c0_rows"]),
            "c1_rows": int(training_set_contract["counts"]["c1_rows"]),
            "dev_only_promotion_allowed": bool(
                training_set_contract["dev_only_promotion_allowed"]
            ),
        },
        "deployable_history_allowlist": list(
            training_set_contract["deployable_observation_allowlist"]
        ),
        "sampling": {
            "episode_sampling_rate": float(args.episode_sampling_rate),
            "num_shards_per_epoch": int(args.num_shards_per_epoch),
            "shard_size": int(args.shard_size),
            "seed": int(DEFAULT_SEED),
            "seed_source": "Isaac-GR00T data_config default",
        },
        "stable_base": {
            "base_model": str(args.base_model),
            "base_model_revision": str(args.base_model_revision),
            "hf_hub_cache_dir": str(args.hf_hub_cache_dir),
            "patched_out_root": str(args.patched_out_root),
            "embodiment_tag": str(args.embodiment_tag),
        },
        "training_budget": {
            "global_batch_size": int(args.global_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "max_steps": int(args.max_steps),
            "learning_rate": float(args.learning_rate),
            "dataloader_num_workers": int(args.dataloader_num_workers),
            "num_gpus": int(args.num_gpus),
            "tune_projector": bool(args.tune_projector),
            "tune_diffusion_model": bool(args.tune_diffusion_model),
            "use_wandb": bool(args.use_wandb),
        },
        "optimizer_schedule": {
            "optimizer": "adamw_torch",
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
        },
        "checkpoint_rule": {
            "save_steps": int(save_steps),
            "save_total_limit": int(DEFAULT_SAVE_TOTAL_LIMIT),
            "must_end_with_exactly_one_retained_checkpoint": True,
        },
        "baseline_training_requested": False,
        "forwarded_passthrough_args": [str(item) for item in forwarded],
    }


def _build_variant_metadata(
    *,
    variant: VariantConfig,
    output_dir: Path,
    selected_checkpoint_path: Path | None,
    shared_run_spec: Mapping[str, Any],
) -> dict[str, Any]:
    comparable_run_spec = {
        **dict(shared_run_spec),
        "conditioning_enabled": bool(variant.conditioning_enabled),
        "null_phase_mode_token_enabled": bool(variant.null_phase_mode_token_enabled),
        "output_dir": str(output_dir.resolve()),
        "checkpoint_rule": {
            **dict(shared_run_spec["checkpoint_rule"]),
            "selected_checkpoint_path": (
                None
                if selected_checkpoint_path is None
                else str(selected_checkpoint_path.resolve())
            ),
        },
    }
    return comparable_run_spec


def _build_delegate_cmd(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    shared_run_spec: Mapping[str, Any],
    variant: VariantConfig,
    output_root: Path,
) -> tuple[list[str], Path, Path, Path]:
    kernel_script = (repo_root / TRAINING_KERNEL_REL).resolve()
    if not kernel_script.is_file():
        raise FileNotFoundError(f"training kernel not found: {kernel_script}")

    output_dir = (output_root / variant.output_dir_name).resolve()
    summary_path = (
        output_root / DELEGATE_SUMMARY_BASENAME_BY_VARIANT[variant.key]
    ).resolve()
    runtime_log_dir = (
        _resolve_path(repo_root, args.runtime_log_dir) / variant.output_dir_name
    )

    cmd = [
        str(Path(sys.executable).resolve()),
        str(kernel_script),
        "--dataset-path",
        str(shared_run_spec["dataset_path"]),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(shared_run_spec["training_budget"]["max_steps"]),
        "--save-steps",
        str(shared_run_spec["checkpoint_rule"]["save_steps"]),
        "--save-total-limit",
        str(shared_run_spec["checkpoint_rule"]["save_total_limit"]),
        "--runtime-log-dir",
        str(runtime_log_dir),
        "--summary-json",
        str(summary_path),
        "--base-model",
        str(shared_run_spec["stable_base"]["base_model"]),
        "--embodiment-tag",
        str(shared_run_spec["stable_base"]["embodiment_tag"]),
        "--global-batch-size",
        str(shared_run_spec["training_budget"]["global_batch_size"]),
        "--gradient-accumulation-steps",
        str(shared_run_spec["training_budget"]["gradient_accumulation_steps"]),
        "--dataloader-num-workers",
        str(shared_run_spec["training_budget"]["dataloader_num_workers"]),
        "--learning-rate",
        str(shared_run_spec["training_budget"]["learning_rate"]),
        "--num-gpus",
        str(shared_run_spec["training_budget"]["num_gpus"]),
        "--weight-decay",
        str(shared_run_spec["optimizer_schedule"]["weight_decay"]),
        "--warmup-ratio",
        str(shared_run_spec["optimizer_schedule"]["warmup_ratio"]),
        "--shard-size",
        str(shared_run_spec["sampling"]["shard_size"]),
        "--episode-sampling-rate",
        str(shared_run_spec["sampling"]["episode_sampling_rate"]),
        "--num-shards-per-epoch",
        str(shared_run_spec["sampling"]["num_shards_per_epoch"]),
        "--tune-projector"
        if bool(shared_run_spec["training_budget"]["tune_projector"])
        else "--no-tune-projector",
        "--tune-diffusion-model"
        if bool(shared_run_spec["training_budget"]["tune_diffusion_model"])
        else "--no-tune-diffusion-model",
        "--use-wandb"
        if bool(shared_run_spec["training_budget"]["use_wandb"])
        else "--no-use-wandb",
    ]
    if str(shared_run_spec["stable_base"]["base_model_revision"]).strip():
        cmd.extend(
            [
                "--base-model-revision",
                str(shared_run_spec["stable_base"]["base_model_revision"]),
            ]
        )
    if str(shared_run_spec["stable_base"]["hf_hub_cache_dir"]).strip():
        cmd.extend(
            [
                "--hf-hub-cache-dir",
                str(shared_run_spec["stable_base"]["hf_hub_cache_dir"]),
            ]
        )
    if str(shared_run_spec["stable_base"]["patched_out_root"]).strip():
        cmd.extend(
            [
                "--patched-out-root",
                str(shared_run_spec["stable_base"]["patched_out_root"]),
            ]
        )
    if str(args.python).strip():
        cmd.extend(["--python", str(args.python).strip()])
    if bool(args.dry_run):
        cmd.append("--dry-run")
    cmd.extend(list(shared_run_spec["forwarded_passthrough_args"]))
    return cmd, output_dir, summary_path, runtime_log_dir


def _run_delegate_and_load_summary(
    cmd: Sequence[str],
    cwd: Path,
    summary_path: Path,
) -> dict[str, Any]:
    proc = subprocess.run(list(cmd), cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"delegated finetune failed with returncode={proc.returncode}"
        )
    if not summary_path.is_file():
        raise RuntimeError(f"delegate summary JSON was not written: {summary_path}")
    return _read_json(summary_path)


def _validate_delegate_summary(
    *,
    variant: VariantConfig,
    delegate_summary: Mapping[str, Any],
    shared_run_spec: Mapping[str, Any],
    output_dir: Path,
    summary_path: Path,
    dry_run: bool,
) -> Path | None:
    if str(delegate_summary.get("wrapper_status", "")) != "ok":
        raise ValueError(
            f"3D training kernel did not report wrapper_status=ok for {variant.key}"
        )
    if str(delegate_summary.get("dataset_path", "")) != str(
        shared_run_spec["dataset_path"]
    ):
        raise ValueError(
            f"{variant.key} dataset path drifted from the frozen shared training-set root"
        )
    if str(delegate_summary.get("output_dir", "")) != str(output_dir):
        raise ValueError(
            f"{variant.key} output dir drifted from the derived variant output dir"
        )

    effective_config = dict(delegate_summary.get("effective_config", {}))
    checks: dict[str, tuple[object, object]] = {
        "max_steps": (
            effective_config.get("max_steps"),
            shared_run_spec["training_budget"]["max_steps"],
        ),
        "save_steps": (
            effective_config.get("save_steps"),
            shared_run_spec["checkpoint_rule"]["save_steps"],
        ),
        "save_total_limit": (
            effective_config.get("save_total_limit"),
            shared_run_spec["checkpoint_rule"]["save_total_limit"],
        ),
        "global_batch_size": (
            effective_config.get("global_batch_size"),
            shared_run_spec["training_budget"]["global_batch_size"],
        ),
        "gradient_accumulation_steps": (
            effective_config.get("gradient_accumulation_steps"),
            shared_run_spec["training_budget"]["gradient_accumulation_steps"],
        ),
        "dataloader_num_workers": (
            effective_config.get("dataloader_num_workers"),
            shared_run_spec["training_budget"]["dataloader_num_workers"],
        ),
        "learning_rate": (
            effective_config.get("learning_rate"),
            shared_run_spec["training_budget"]["learning_rate"],
        ),
        "num_gpus": (
            effective_config.get("num_gpus"),
            shared_run_spec["training_budget"]["num_gpus"],
        ),
        "tune_projector": (
            effective_config.get("tune_projector"),
            shared_run_spec["training_budget"]["tune_projector"],
        ),
        "tune_diffusion_model": (
            effective_config.get("tune_diffusion_model"),
            shared_run_spec["training_budget"]["tune_diffusion_model"],
        ),
        "use_wandb": (
            effective_config.get("use_wandb"),
            shared_run_spec["training_budget"]["use_wandb"],
        ),
        "base_model": (
            effective_config.get("base_model"),
            shared_run_spec["stable_base"]["base_model"],
        ),
        "base_model_revision": (
            effective_config.get("base_model_revision", ""),
            shared_run_spec["stable_base"]["base_model_revision"],
        ),
        "embodiment_tag": (
            effective_config.get("embodiment_tag"),
            shared_run_spec["stable_base"]["embodiment_tag"],
        ),
    }
    drifted_fields = [
        field_name
        for field_name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if drifted_fields:
        raise ValueError(
            f"{variant.key} delegated training drifted frozen config fields: {', '.join(drifted_fields)}"
        )

    if str(summary_path) != str(summary_path.resolve()):
        summary_path = summary_path.resolve()

    if bool(dry_run):
        return None

    selected_checkpoint = _ensure_single_retained_checkpoint(output_dir)
    selected_checkpoint_path = str(delegate_summary.get("selected_checkpoint_path", ""))
    if selected_checkpoint_path and selected_checkpoint_path != str(
        selected_checkpoint
    ):
        raise ValueError(
            f"{variant.key} selected checkpoint path drifted from retained checkpoint: {selected_checkpoint_path}"
        )
    return selected_checkpoint


def _flatten_differences(
    left: object,
    right: object,
    *,
    prefix: str = "",
) -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        keys = sorted({*left.keys(), *right.keys()})
        for key in keys:
            key_str = str(key)
            child_prefix = key_str if not prefix else f"{prefix}.{key_str}"
            if key not in left or key not in right:
                differences.append(child_prefix)
                continue
            differences.extend(
                _flatten_differences(left[key], right[key], prefix=child_prefix)
            )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if left == right:
            return []
        return [prefix]
    if left != right:
        return [prefix]
    return []


def build_diff_whitelist_result(
    c0_metadata: Mapping[str, Any],
    c1_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    c0_run = dict(c0_metadata.get("comparable_run_spec", {}))
    c1_run = dict(c1_metadata.get("comparable_run_spec", {}))
    difference_paths = sorted(_flatten_differences(c0_run, c1_run))
    unexpected_paths = [
        path for path in difference_paths if str(path) not in ALLOWED_DIFF_PATHS
    ]
    return {
        "schema_version": DIFF_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_training_fairness_diff_whitelist",
        "status": "PASS" if not unexpected_paths else "FAIL",
        "allowed_difference_paths": sorted(ALLOWED_DIFF_PATHS),
        "observed_difference_paths": difference_paths,
        "unexpected_difference_paths": unexpected_paths,
        "same_equal_data_fairness_audit_path": (
            c0_run.get("source_data", {}).get("equal_data_fairness_audit_path")
            == c1_run.get("source_data", {}).get("equal_data_fairness_audit_path")
        ),
        "same_dataset_fingerprint": (
            c0_run.get("dataset_fingerprint") == c1_run.get("dataset_fingerprint")
        ),
    }


def validate_diff_whitelist_or_raise(
    c0_metadata: Mapping[str, Any],
    c1_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    result = build_diff_whitelist_result(c0_metadata, c1_metadata)
    if not bool(result["same_equal_data_fairness_audit_path"]):
        raise ValueError(
            "C0/C1 no longer reference the same equal_data_fairness_audit.json"
        )
    if not bool(result["same_dataset_fingerprint"]):
        raise ValueError("C0/C1 dataset fingerprint drifted")
    if result["unexpected_difference_paths"]:
        raise ValueError(
            "run metadata fairness drift detected: "
            + ", ".join(result["unexpected_difference_paths"])
        )
    return result


def _load_existing_run_metadata(
    output_root: Path, variant: str
) -> dict[str, Any] | None:
    path = (output_root / RUN_METADATA_BASENAME_BY_VARIANT[variant]).resolve()
    if not path.is_file():
        return None
    return _read_json(path)


def _write_pending_diff_artifact(output_root: Path, *, reason: str) -> Path:
    payload = {
        "schema_version": DIFF_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_training_fairness_diff_whitelist",
        "status": "PENDING",
        "reason": str(reason),
        "allowed_difference_paths": sorted(ALLOWED_DIFF_PATHS),
        "observed_difference_paths": [],
        "unexpected_difference_paths": [],
    }
    return _write_json((output_root / DIFF_WHITELIST_JSON_NAME).resolve(), payload)


def _run_training_variant(
    *,
    repo_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    shared_run_spec: Mapping[str, Any],
    variant: VariantConfig,
    runner: Callable[[Sequence[str], Path, Path], Mapping[str, Any]],
) -> dict[str, Any]:
    cmd, output_dir, summary_path, runtime_log_dir = _build_delegate_cmd(
        repo_root=repo_root,
        args=args,
        shared_run_spec=shared_run_spec,
        variant=variant,
        output_root=output_root,
    )
    delegate_summary = dict(runner(cmd, repo_root, summary_path))
    selected_checkpoint = _validate_delegate_summary(
        variant=variant,
        delegate_summary=delegate_summary,
        shared_run_spec=shared_run_spec,
        output_dir=output_dir,
        summary_path=summary_path,
        dry_run=bool(args.dry_run),
    )
    metadata = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_training_run_metadata",
        "variant_key": variant.key,
        "training_route": dict(shared_run_spec["training_route"]),
        "comparable_run_spec": _build_variant_metadata(
            variant=variant,
            output_dir=output_dir,
            selected_checkpoint_path=selected_checkpoint,
            shared_run_spec=shared_run_spec,
        ),
        "delegate_summary_path": str(summary_path),
        "delegate_summary_fingerprint": _sha256_file(summary_path),
        "runtime_log_dir": str(runtime_log_dir),
        "training_kernel": str((repo_root / TRAINING_KERNEL_REL).resolve()),
    }
    metadata_path = _write_json(
        (output_root / RUN_METADATA_BASENAME_BY_VARIANT[variant.key]).resolve(),
        metadata,
    )
    return {
        "metadata_path": str(metadata_path),
        "delegate_summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "selected_checkpoint_path": None
        if selected_checkpoint is None
        else str(selected_checkpoint),
    }


def _materialize_diff_whitelist(output_root: Path) -> tuple[Path, dict[str, Any]]:
    c0_metadata = _load_existing_run_metadata(output_root, "c0")
    c1_metadata = _load_existing_run_metadata(output_root, "c1")
    if c0_metadata is None or c1_metadata is None:
        diff_artifact_path = _write_pending_diff_artifact(
            output_root,
            reason="waiting_for_both_c0_and_c1_run_metadata",
        )
        return diff_artifact_path, _read_json(diff_artifact_path)

    diff_result = validate_diff_whitelist_or_raise(c0_metadata, c1_metadata)
    diff_artifact_path = _write_json(
        (output_root / DIFF_WHITELIST_JSON_NAME).resolve(),
        diff_result,
    )
    return diff_artifact_path, diff_result


@dataclass
class TrainingSetContractLoader:
    repo_root: Path

    def load(self, training_set_root: Path) -> dict[str, Any]:
        return _load_training_set_contract(
            _resolve_path(self.repo_root, training_set_root)
        )


@dataclass
class VariantTrainingRunner:
    repo_root: Path
    output_root: Path
    args: argparse.Namespace
    shared_run_spec: Mapping[str, Any]
    runner: Callable[[Sequence[str], Path, Path], Mapping[str, Any]]

    def run_variant(self, variant: VariantConfig) -> dict[str, Any]:
        return _run_training_variant(
            repo_root=self.repo_root,
            output_root=self.output_root,
            args=self.args,
            shared_run_spec=self.shared_run_spec,
            variant=variant,
            runner=self.runner,
        )


@dataclass
class DiffWhitelistManager:
    output_root: Path

    def materialize(self) -> tuple[Path, dict[str, Any]]:
        return _materialize_diff_whitelist(self.output_root)


@dataclass
class StateConditionedTrainingWorkflow:
    args: argparse.Namespace
    forwarded: Sequence[str]
    kernel_runner: Callable[[Sequence[str], Path, Path], Mapping[str, Any]] | None = (
        None
    )
    repo_root: Path = field(init=False)
    output_root: Path = field(init=False)
    contract_loader: TrainingSetContractLoader = field(init=False)
    training_set_contract: dict[str, Any] = field(init=False)
    shared_run_spec: Mapping[str, Any] = field(init=False)
    runner: Callable[[Sequence[str], Path, Path], Mapping[str, Any]] = field(init=False)
    variant_runner: VariantTrainingRunner = field(init=False)
    diff_manager: DiffWhitelistManager = field(init=False)

    def __post_init__(self) -> None:
        self.repo_root = _repo_root()
        self.output_root = _resolve_path(self.repo_root, self.args.output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.contract_loader = TrainingSetContractLoader(self.repo_root)
        self.training_set_contract = self.contract_loader.load(
            self.args.training_set_root
        )
        self.shared_run_spec = _build_shared_run_spec(
            args=self.args,
            training_set_contract=self.training_set_contract,
            forwarded=self.forwarded,
        )
        self.runner = self.kernel_runner or _run_delegate_and_load_summary
        self.variant_runner = VariantTrainingRunner(
            repo_root=self.repo_root,
            output_root=self.output_root,
            args=self.args,
            shared_run_spec=self.shared_run_spec,
            runner=self.runner,
        )
        self.diff_manager = DiffWhitelistManager(self.output_root)

    def selected_variants(self) -> list[VariantConfig]:
        variant_keys = (
            ["c0", "c1"] if self.args.variant == "all" else [str(self.args.variant)]
        )
        return [VARIANT_CONFIGS[key] for key in variant_keys]

    def run_variants(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for variant in self.selected_variants():
            results[variant.key] = self.variant_runner.run_variant(variant)
        return results

    def execute(self) -> dict[str, Any]:
        run_results = self.run_variants()
        diff_artifact_path, diff_result = self.diff_manager.materialize()
        return {
            "training_set_root": str(self.training_set_contract["training_set_root"]),
            "shared_run_spec": self.shared_run_spec,
            "run_results": run_results,
            "diff_whitelist_path": str(diff_artifact_path),
            "diff_whitelist_status": str(diff_result["status"]),
            "baseline_trained": False,
        }


def materialize_state_conditioned_training(
    *,
    args: argparse.Namespace,
    forwarded: Sequence[str],
    kernel_runner: Callable[[Sequence[str], Path, Path], Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    return StateConditionedTrainingWorkflow(
        args=args,
        forwarded=forwarded,
        kernel_runner=kernel_runner,
    ).execute()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)
    try:
        result = materialize_state_conditioned_training(
            args=args,
            forwarded=forwarded,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ALLOWED_DIFF_PATHS",
    "DEFAULT_SEED",
    "DEFAULT_TRAINING_SET_ROOT",
    "DIFF_WHITELIST_JSON_NAME",
    "MAINLINE_CARRIER_ROUTE",
    "MAINLINE_CARRIER_SCHEMA_VERSION",
    "MAINLINE_INDICATOR_SOURCE_FIELD",
    "MAINLINE_PROMPT_SOURCE_FIELD",
    "MAINLINE_RUNTIME_INDICATOR_MODES",
    "MAINLINE_RUNTIME_POLICY_CLASS",
    "MAINLINE_RUNTIME_ROUTE",
    "RUN_METADATA_BASENAME_BY_VARIANT",
    "VARIANT_CONFIGS",
    "VariantConfig",
    "build_diff_whitelist_result",
    "build_parser",
    "main",
    "materialize_state_conditioned_training",
    "validate_diff_whitelist_or_raise",
]


if __name__ == "__main__":
    raise SystemExit(main())


class StateConditionedTrainScriptApp:
    def build_parser(self):
        return build_parser()

    def materialize_training(self, **kwargs):
        return StateConditionedTrainingWorkflow(**kwargs).execute()

    def run(self, argv=None) -> int:
        return main(argv)
