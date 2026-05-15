#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
import datetime as _dt
import hashlib
import json
import signal
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.demo_utils import paths as demo_paths
from work.recap.r7_1_recipe_plumbing.flags import RecipeFlags, build_argparse_group, recipe_flags_to_cli_args


DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1"
DEFAULT_DELEGATE_SCRIPT_REL = "work/recap/scripts/34_recap_finetune_repro.py"
DEFAULT_REAL_LAUNCHER_REL = "work/recap/launch_finetune_use_ddp.py"
DEFAULT_RUNTIME_LOG_DIR = "agent/runtime_logs/3D_recap_finetune_full"
DEFAULT_PATCHED_OUT_ROOT = "agent/artifacts/hf_patches"
DEFAULT_REPO_LOCAL_METADATA_DIRNAME = "repo_local_metadata"
DEFAULT_MAX_STEPS = 100
DEFAULT_GLOBAL_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_DATALOADER_NUM_WORKERS = 0
DEFAULT_SAVE_TOTAL_LIMIT = 1
DEFAULT_TUNE_PROJECTOR = False
DEFAULT_TUNE_DIFFUSION_MODEL = True
DEFAULT_USE_WANDB = False
SINGLE_GPU_LAUNCH_FAMILY = "single_gpu_v1"
HISTORICAL_DDP_LAUNCH_FAMILY = "task10_2gpu_ddp_diagnostic_v1"
FULL_UPDATE_AUTHORITY_ROOT_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
)
LIVE_FULL_UPDATE_AUTHORITY_ROOT_REL = Path(
    "agent/artifacts/gr00t_recap_live/single_gpu_v2_full_update"
)
FULL_UPDATE_AUTHORITY_ROOT_RELS = (
    FULL_UPDATE_AUTHORITY_ROOT_REL,
    LIVE_FULL_UPDATE_AUTHORITY_ROOT_REL,
)
READONLY_BASELINE_AUTHORITY_ROOT_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v1"
)
LEGACY_HISTORICAL_OUTPUT_ROOT_PREFIXES = (
    "stage3_t10",
    "stage3_t11",
    "stage3_t12",
    "stage3_t13",
)
AUTHORITY_ROOT_BLOCKER_SCHEMA_VERSION = "recap_full_update_authority_blocker_v1"
DDP_GREEN_SMOKE_VERDICT_REL = Path("agent/artifacts/stage3_ddp_smoke/green_smoke_verdict.json")
SINGLE_GPU_SMOKE_VERDICT_REL = Path(
    "agent/artifacts/stage3_single_gpu_smoke/gpu1_formal_geometry_attempt01/green_smoke_single_gpu_verdict.json"
)
COMPARABILITY_MANIFEST_SCHEMA_VERSION = "full_update_comparability_manifest_v1"
COMPARABILITY_MANIFEST_ARTIFACT_KIND = "full_update_comparability_manifest"
COMPARABILITY_MANIFEST_FILENAME = "comparability_manifest.json"
COMPARABILITY_SEED_SET_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v1/eval_seed_set.json"
)
TASK11_CONDITIONED_FORMAL_RUN_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
    "t13_advantage_full_update_1gpu/formal_run"
)
TASK12_CONTINUATION_FORMAL_RUN_REL = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
    "t13_continuation_full_update_1gpu/formal_run"
)
COMPARABILITY_SHARED_FIELDS: tuple[str, ...] = (
    "warm_start_checkpoint",
    "optimizer_reset",
    "scheduler_reset",
    "batch_geometry",
    "dataset_fingerprint",
    "seed_set",
    "seed_set_source",
    "eval_budget",
    "launch_family",
    "policy_route",
    "train_scope_requested",
    "train_scope_effective",
)
COMPARABILITY_ALLOWED_DIFF_FIELDS = frozenset(
    {
        "advantage_consumed",
        "output_dir",
        "comparability_manifest_path",
    }
)

FROZEN_DIRECT_FINETUNE_OUTPUT_CONTRACTS: dict[str, dict[str, str | None]] = {
    "agent/artifacts/stage3_ddp_smoke/run_a_gpu1_attempt01": {
        "runtime_log_dir": "agent/runtime_logs/stage3_ddp_smoke/run_a_gpu1_attempt01",
        "visible_devices_policy": "single_gpu_gpu1_only",
        "contract_role": "smoke_attempt",
        "launch_family": SINGLE_GPU_LAUNCH_FAMILY,
        "authority_status": "smoke_only",
    },
    "agent/artifacts/stage3_ddp_smoke/run_b_gpu2_attempt01": {
        "runtime_log_dir": "agent/runtime_logs/stage3_ddp_smoke/run_b_gpu2_attempt01",
        "visible_devices_policy": "single_gpu_gpu2_only",
        "contract_role": "smoke_attempt",
        "launch_family": SINGLE_GPU_LAUNCH_FAMILY,
        "authority_status": "manual_fallback_smoke_only",
    },
    "agent/artifacts/stage3_ddp_smoke/run_c_gpu12_attempt01": {
        "runtime_log_dir": "agent/runtime_logs/stage3_ddp_smoke/run_c_gpu12_attempt01",
        "visible_devices_policy": "torchrun_gpu1_gpu2_only",
        "contract_role": "smoke_attempt",
        "launch_family": HISTORICAL_DDP_LAUNCH_FAMILY,
        "authority_status": "historical_diagnostic_only",
    },
    "agent/artifacts/stage3_ddp_smoke/run_c_gpu12_b1_attempt01": {
        "runtime_log_dir": "agent/runtime_logs/stage3_ddp_smoke/run_c_gpu12_b1_attempt01",
        "visible_devices_policy": "torchrun_gpu1_gpu2_only",
        "contract_role": "smoke_attempt",
        "launch_family": HISTORICAL_DDP_LAUNCH_FAMILY,
        "authority_status": "historical_diagnostic_only",
    },
    "agent/artifacts/stage3_ddp_smoke/run_c_gpu12_b2_attempt01": {
        "runtime_log_dir": "agent/runtime_logs/stage3_ddp_smoke/run_c_gpu12_b2_attempt01",
        "visible_devices_policy": "torchrun_gpu1_gpu2_only",
        "contract_role": "smoke_attempt",
        "launch_family": HISTORICAL_DDP_LAUNCH_FAMILY,
        "authority_status": "historical_diagnostic_only",
    },
    "agent/artifacts/stage3_t3b_baseline_1gpu/formal_dataset_gate": {
        "runtime_log_dir": None,
        "visible_devices_policy": "dataset_gate_not_a_training_launch",
        "contract_role": "formal_gate",
        "launch_family": SINGLE_GPU_LAUNCH_FAMILY,
        "authority_status": "live_authority",
    },
    "agent/artifacts/stage3_t3b_baseline_1gpu/formal_run": {
        "runtime_log_dir": "agent/runtime_logs/stage3_t3b_baseline_1gpu/formal_run",
        "visible_devices_policy": "single_gpu_gpu1_only",
        "contract_role": "formal_run",
        "launch_family": SINGLE_GPU_LAUNCH_FAMILY,
        "authority_status": "live_authority",
    },
    "agent/artifacts/stage3_t3b_baseline_2gpu/formal_dataset_gate": {
        "runtime_log_dir": None,
        "visible_devices_policy": "dataset_gate_not_a_training_launch",
        "contract_role": "formal_gate",
        "launch_family": HISTORICAL_DDP_LAUNCH_FAMILY,
        "authority_status": "historical_diagnostic_only",
    },
    "agent/artifacts/stage3_t3b_baseline_2gpu/formal_run": {
        "runtime_log_dir": "agent/runtime_logs/stage3_t3b_baseline_2gpu/formal_run",
        "visible_devices_policy": "torchrun_gpu1_gpu2_only",
        "contract_role": "formal_run",
        "launch_family": HISTORICAL_DDP_LAUNCH_FAMILY,
        "authority_status": "historical_diagnostic_only",
    },
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/t13_advantage_full_update_1gpu/formal_run": {
        "runtime_log_dir": "agent/runtime_logs/recap_full_update_first/task11_conditioned_formal_run",
        "visible_devices_policy": "single_gpu_gpu1_only",
        "contract_role": "formal_run",
        "launch_family": SINGLE_GPU_LAUNCH_FAMILY,
        "authority_status": "live_authority",
    },
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/t13_continuation_full_update_1gpu/formal_run": {
        "runtime_log_dir": "agent/runtime_logs/recap_full_update_first/task12_continuation_formal_run",
        "visible_devices_policy": "single_gpu_gpu1_only",
        "contract_role": "formal_run",
        "launch_family": SINGLE_GPU_LAUNCH_FAMILY,
        "authority_status": "live_authority",
    },
}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "work").is_dir() and (parent / "agent").is_dir():
            return parent
    return Path.cwd().resolve()


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    # `agent/artifacts` and `agent/runtime_logs` are canonical repo
    # entrypoints, but this workspace intentionally maps them to the HDD live
    # root via symlinks.  Authority checks below are based on repo-relative
    # contracts, so do not collapse those symlinks with Path.resolve().
    return demo_paths.abspath_preserve_symlink(path)


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _auto_use_ddp(num_gpus: int | None) -> bool:
    if num_gpus is None:
        return False
    return int(num_gpus) > 1


def _effective_batch_geometry(
    *,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int | None,
) -> dict[str, int]:
    resolved_num_gpus = 1 if num_gpus is None else int(num_gpus)
    divisor = resolved_num_gpus * int(gradient_accumulation_steps)
    if divisor <= 0:
        raise ValueError(
            "num_gpus * gradient_accumulation_steps must be positive to compute batch geometry"
        )
    if int(global_batch_size) % divisor != 0:
        raise ValueError(
            "global_batch_size must be evenly divisible by num_gpus * gradient_accumulation_steps"
        )
    per_device_batch_size = int(global_batch_size) // divisor
    effective_update_batch = (
        per_device_batch_size * resolved_num_gpus * int(gradient_accumulation_steps)
    )
    return {
        "per_device_batch_size": int(per_device_batch_size),
        "effective_update_batch": int(effective_update_batch),
    }


def _repo_relative_str(repo_root: Path, path: Path) -> str | None:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return None


def _stable_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_repo_path_string(repo_root: Path, raw: str | Path | None) -> str | None:
    if raw is None:
        return None
    normalized = str(raw).strip()
    if not normalized:
        return None
    resolved = _resolve_path(repo_root, normalized)
    return _repo_relative_str(repo_root, resolved) or str(resolved)


def _coerce_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _read_flag_value(argv: object, *, flag: str) -> str | None:
    if not isinstance(argv, list):
        return None
    values = [str(item) for item in argv]
    for index, token in enumerate(values):
        if token == flag:
            if index + 1 < len(values):
                return values[index + 1]
            return None
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _resolve_forwarded_train_scope(forwarded: list[str]) -> str | None:
    raw_scope = _read_flag_value(forwarded, flag="--recap-train-scope")
    if raw_scope is None:
        return None
    normalized = str(raw_scope).strip()
    return None if not normalized else normalized


def _normalize_comparability_seed_bundle(
    payload: Mapping[str, Any], *, source_path: str
) -> dict[str, Any]:
    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("eval_seed_set.seeds must be a non-empty list of ints")
    seeds: list[int] = []
    for index, raw_seed in enumerate(raw_seeds):
        if not isinstance(raw_seed, int) or isinstance(raw_seed, bool):
            raise ValueError(
                f"eval_seed_set.seeds[{index}] must be an int, got {type(raw_seed).__name__}"
            )
        seeds.append(int(raw_seed))

    formal_eval_episodes = payload.get("formal_eval_episodes")
    if (
        not isinstance(formal_eval_episodes, int)
        or isinstance(formal_eval_episodes, bool)
        or formal_eval_episodes <= 0
    ):
        raise ValueError(
            "eval_seed_set.formal_eval_episodes must be a positive int"
        )
    if formal_eval_episodes != len(seeds):
        raise ValueError(
            "eval_seed_set.formal_eval_episodes must equal len(eval_seed_set.seeds)"
        )

    raw_episode_indices = payload.get("episode_indices")
    if not isinstance(raw_episode_indices, list) or len(raw_episode_indices) != len(seeds):
        raise ValueError(
            "eval_seed_set.episode_indices must be a list aligned with eval_seed_set.seeds"
        )
    episode_indices: list[int] = []
    for index, raw_episode_index in enumerate(raw_episode_indices):
        if not isinstance(raw_episode_index, int) or isinstance(raw_episode_index, bool):
            raise ValueError(
                "eval_seed_set.episode_indices[%d] must be an int, got %s"
                % (index, type(raw_episode_index).__name__)
            )
        episode_indices.append(int(raw_episode_index))

    return {
        "status": "ok",
        "seed_set": seeds,
        "seed_set_source": "inherit_from_v1",
        "seed_set_source_path": source_path,
        "eval_budget": {
            "formal_eval_episodes": int(formal_eval_episodes),
            "episode_indices": episode_indices,
        },
    }


def load_full_update_comparability_seed_bundle(
    repo_root: Path,
    *,
    seed_bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_path = (
        _resolve_path(repo_root, str(seed_bundle_path))
        if seed_bundle_path is not None
        else (repo_root / COMPARABILITY_SEED_SET_REL).resolve()
    )
    source_path = _repo_relative_str(repo_root, resolved_path) or str(resolved_path)
    if not resolved_path.is_file():
        return {
            "status": "blocked",
            "blocker_code": "seed_bundle_missing_block",
            "reason": "single_gpu_v1/eval_seed_set.json is required for comparability and cannot be synthesized",
            "seed_set_source": "inherit_from_v1",
            "seed_set_source_path": source_path,
            "seed_set": None,
            "eval_budget": None,
        }

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"expected JSON object, got {type(payload).__name__}"
            )
        normalized = _normalize_comparability_seed_bundle(
            payload,
            source_path=source_path,
        )
        normalized["seed_bundle_path"] = source_path
        return normalized
    except Exception as exc:
        return {
            "status": "blocked",
            "blocker_code": "seed_bundle_malformed_block",
            "reason": f"malformed inherited eval seed bundle: {exc}",
            "seed_set_source": "inherit_from_v1",
            "seed_set_source_path": source_path,
            "seed_set": None,
            "eval_budget": None,
        }


def build_full_update_dataset_fingerprint(
    repo_root: Path,
    *,
    dataset_path: str | Path | None = None,
    dataset_fingerprint: object | None = None,
) -> str | None:
    if isinstance(dataset_fingerprint, str) and dataset_fingerprint.strip():
        return dataset_fingerprint.strip()
    normalized_dataset_path = _resolved_repo_path_string(repo_root, dataset_path)
    if normalized_dataset_path is None:
        return None
    return _stable_json_sha256({"dataset_path": normalized_dataset_path})


def build_full_update_comparability_manifest(
    *,
    repo_root: Path,
    output_dir: str | Path,
    warm_start_checkpoint: str | Path | None,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int | None,
    dataset_path: str | Path | None = None,
    dataset_fingerprint: object | None = None,
    launch_family: object | None = None,
    train_scope_requested: object | None = None,
    train_scope_effective: object | None = None,
    advantage_consumed: bool,
    seed_bundle_path: str | Path | None = None,
    policy_route: object | None = None,
    policy_indicator_mode: object | None = None,
) -> dict[str, Any]:
    from work.recap import policy as recap_policy

    resolved_output_dir = _resolve_path(repo_root, str(output_dir))
    manifest_path = resolved_output_dir / COMPARABILITY_MANIFEST_FILENAME
    seed_bundle = load_full_update_comparability_seed_bundle(
        repo_root,
        seed_bundle_path=seed_bundle_path,
    )
    route_freeze = recap_policy.build_comparability_policy_route_freeze(
        route=policy_route,
        indicator_mode=policy_indicator_mode,
    )
    payload: dict[str, Any] = {
        "schema_version": COMPARABILITY_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": COMPARABILITY_MANIFEST_ARTIFACT_KIND,
        "status": str(seed_bundle["status"]),
        "output_dir": _repo_relative_str(repo_root, resolved_output_dir)
        or str(resolved_output_dir),
        "comparability_manifest_path": _repo_relative_str(repo_root, manifest_path)
        or str(manifest_path),
        "warm_start_checkpoint": _resolved_repo_path_string(
            repo_root, warm_start_checkpoint
        ),
        "optimizer_reset": True,
        "scheduler_reset": True,
        "batch_geometry": {
            "global_batch_size": int(global_batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "num_gpus": 1 if num_gpus is None else int(num_gpus),
            **_effective_batch_geometry(
                global_batch_size=int(global_batch_size),
                gradient_accumulation_steps=int(gradient_accumulation_steps),
                num_gpus=num_gpus,
            ),
        },
        "dataset_fingerprint": build_full_update_dataset_fingerprint(
            repo_root,
            dataset_path=dataset_path,
            dataset_fingerprint=dataset_fingerprint,
        ),
        "seed_set": seed_bundle.get("seed_set"),
        "seed_set_source": str(seed_bundle.get("seed_set_source", "inherit_from_v1")),
        "seed_set_source_path": seed_bundle.get("seed_set_source_path"),
        "eval_budget": seed_bundle.get("eval_budget"),
        "launch_family": None if launch_family is None else str(launch_family),
        "policy_route": str(route_freeze["route"]),
        "policy_route_frozen": bool(route_freeze["frozen"]),
        "policy_route_freeze": dict(route_freeze),
        "train_scope_requested": (
            None if train_scope_requested is None else str(train_scope_requested)
        ),
        "train_scope_effective": (
            str(train_scope_requested)
            if train_scope_effective is None and train_scope_requested is not None
            else None if train_scope_effective is None else str(train_scope_effective)
        ),
        "advantage_consumed": bool(advantage_consumed),
        "optimizer_state_shared_across_lanes": False,
        "blocker_code": seed_bundle.get("blocker_code"),
        "reason": seed_bundle.get("reason"),
    }
    if payload["status"] == "ok":
        required_fields_missing = [
            field_name
            for field_name in (
                "warm_start_checkpoint",
                "dataset_fingerprint",
                "launch_family",
                "policy_route",
                "train_scope_requested",
                "train_scope_effective",
            )
            if payload.get(field_name) is None
        ]
        if required_fields_missing:
            payload["status"] = "blocked"
            payload["blocker_code"] = "comparability_manifest_incomplete_block"
            payload["reason"] = (
                "comparability manifest is incomplete; missing required frozen fields: "
                + ", ".join(required_fields_missing)
            )
    return payload


def write_full_update_comparability_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    output_dir: str | Path,
) -> Path:
    resolved_output_dir = _resolve_path(repo_root, str(output_dir))
    manifest_path = resolved_output_dir / COMPARABILITY_MANIFEST_FILENAME
    _write_json(manifest_path, dict(manifest))
    return manifest_path


def emit_full_update_comparability_manifest(
    *,
    repo_root: Path,
    output_dir: str | Path,
    warm_start_checkpoint: str | Path | None,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int | None,
    dataset_path: str | Path | None = None,
    dataset_fingerprint: object | None = None,
    launch_family: object | None = None,
    train_scope_requested: object | None = None,
    train_scope_effective: object | None = None,
    advantage_consumed: bool,
    seed_bundle_path: str | Path | None = None,
    policy_route: object | None = None,
    policy_indicator_mode: object | None = None,
) -> dict[str, Any]:
    manifest = build_full_update_comparability_manifest(
        repo_root=repo_root,
        output_dir=output_dir,
        warm_start_checkpoint=warm_start_checkpoint,
        global_batch_size=global_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_gpus=num_gpus,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        launch_family=launch_family,
        train_scope_requested=train_scope_requested,
        train_scope_effective=train_scope_effective,
        advantage_consumed=advantage_consumed,
        seed_bundle_path=seed_bundle_path,
        policy_route=policy_route,
        policy_indicator_mode=policy_indicator_mode,
    )
    write_full_update_comparability_manifest(
        manifest,
        repo_root=repo_root,
        output_dir=output_dir,
    )
    return manifest


def emit_conditioned_formal_lane_comparability_manifest(
    *,
    repo_root: Path,
    output_dir: str | Path,
    warm_start_checkpoint: str | Path,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int | None,
    dataset_path: str | Path | None = None,
    dataset_fingerprint: object | None = None,
    train_scope_requested: object | None = None,
    train_scope_effective: object | None = None,
    seed_bundle_path: str | Path | None = None,
    policy_route: object | None = None,
    policy_indicator_mode: object | None = None,
    launch_family: object | None = SINGLE_GPU_LAUNCH_FAMILY,
) -> dict[str, Any]:
    return emit_full_update_comparability_manifest(
        repo_root=repo_root,
        output_dir=output_dir,
        warm_start_checkpoint=warm_start_checkpoint,
        global_batch_size=global_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_gpus=num_gpus,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        launch_family=launch_family,
        train_scope_requested=train_scope_requested,
        train_scope_effective=train_scope_effective,
        advantage_consumed=True,
        seed_bundle_path=seed_bundle_path,
        policy_route=policy_route,
        policy_indicator_mode=policy_indicator_mode,
    )


def emit_continuation_formal_lane_comparability_manifest(
    *,
    repo_root: Path,
    output_dir: str | Path,
    warm_start_checkpoint: str | Path,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int | None,
    dataset_path: str | Path | None = None,
    dataset_fingerprint: object | None = None,
    train_scope_requested: object | None = None,
    train_scope_effective: object | None = None,
    seed_bundle_path: str | Path | None = None,
    policy_route: object | None = None,
    policy_indicator_mode: object | None = None,
    launch_family: object | None = SINGLE_GPU_LAUNCH_FAMILY,
) -> dict[str, Any]:
    return emit_full_update_comparability_manifest(
        repo_root=repo_root,
        output_dir=output_dir,
        warm_start_checkpoint=warm_start_checkpoint,
        global_batch_size=global_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_gpus=num_gpus,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        launch_family=launch_family,
        train_scope_requested=train_scope_requested,
        train_scope_effective=train_scope_effective,
        advantage_consumed=False,
        seed_bundle_path=seed_bundle_path,
        policy_route=policy_route,
        policy_indicator_mode=policy_indicator_mode,
    )


def validate_full_update_comparability_manifests(
    conditioned_manifest: Mapping[str, Any] | object,
    continuation_manifest: Mapping[str, Any] | object,
) -> dict[str, Any]:
    conditioned = _coerce_mapping(
        conditioned_manifest,
        field_name="conditioned_manifest",
    )
    continuation = _coerce_mapping(
        continuation_manifest,
        field_name="continuation_manifest",
    )
    if str(conditioned.get("status")) != "ok":
        return {
            "status": "blocked",
            "blocker_code": conditioned.get("blocker_code")
            or "conditioned_manifest_blocked",
            "reason": conditioned.get("reason"),
        }
    if str(continuation.get("status")) != "ok":
        return {
            "status": "blocked",
            "blocker_code": continuation.get("blocker_code")
            or "continuation_manifest_blocked",
            "reason": continuation.get("reason"),
        }

    mismatches: list[dict[str, Any]] = []
    blocker_code = None
    blocker_reason = None
    for field_name in COMPARABILITY_SHARED_FIELDS:
        conditioned_value = conditioned.get(field_name)
        continuation_value = continuation.get(field_name)
        if conditioned_value == continuation_value:
            continue
        mismatch = {
            "field": field_name,
            "conditioned": conditioned_value,
            "continuation": continuation_value,
        }
        mismatches.append(mismatch)
        if field_name == "policy_route":
            blocker_code = "route_mismatch_block"
            blocker_reason = "policy_route drifted between conditioned and continuation lanes"
            break
        if field_name == "warm_start_checkpoint":
            blocker_code = "warm_start_mismatch_block"
            blocker_reason = "warm_start_checkpoint drifted between conditioned and continuation lanes"
            break

    differing_fields = sorted(
        field_name
        for field_name in set(conditioned) | set(continuation)
        if conditioned.get(field_name) != continuation.get(field_name)
    )
    unexpected_diff_fields = sorted(
        field_name
        for field_name in differing_fields
        if field_name not in COMPARABILITY_ALLOWED_DIFF_FIELDS
    )
    if blocker_code is None and unexpected_diff_fields:
        blocker_code = "disallowed_diff_block"
        blocker_reason = (
            "conditioned/continuation comparability manifests drifted outside the allowed diff surface"
        )
    if blocker_code is not None:
        return {
            "status": "blocked",
            "blocker_code": blocker_code,
            "reason": blocker_reason,
            "mismatches": mismatches,
            "unexpected_diff_fields": unexpected_diff_fields,
            "allowed_diff_fields": sorted(COMPARABILITY_ALLOWED_DIFF_FIELDS),
        }
    return {
        "status": "pass",
        "allowed_diff_fields": sorted(COMPARABILITY_ALLOWED_DIFF_FIELDS),
        "differing_fields": differing_fields,
        "unexpected_diff_fields": [],
    }


class AuthorityRootBlocker(ValueError):
    def __init__(self, payload: dict[str, Any]):
        self.payload = dict(payload)
        super().__init__(json.dumps(self.payload, ensure_ascii=True, sort_keys=True))


def _is_within_repo_relative_root(path_str: str | None, root: Path) -> bool:
    if path_str is None:
        return False
    root_str = root.as_posix()
    return path_str == root_str or path_str.startswith(root_str + "/")


def _is_within_any_repo_relative_root(
    path_str: str | None,
    roots: tuple[Path, ...],
) -> bool:
    return any(_is_within_repo_relative_root(path_str, root) for root in roots)


def _artifact_root_name(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    artifact_prefix = "agent/artifacts/"
    if not path_str.startswith(artifact_prefix):
        return None
    suffix = path_str[len(artifact_prefix) :]
    if suffix == "":
        return None
    return suffix.split("/", 1)[0]


def _authority_root_blocker_payload(
    *,
    repo_root: Path,
    resolved_path: Path,
    blocker_code: str,
    reason: str,
    matched_legacy_root_prefix: str | None = None,
) -> dict[str, Any]:
    attempted_output_dir = _repo_relative_str(repo_root, resolved_path) or str(resolved_path)
    return {
        "schema_version": AUTHORITY_ROOT_BLOCKER_SCHEMA_VERSION,
        "artifact_kind": "recap_full_update_authority_root_blocker",
        "status": "blocked",
        "blocker_code": blocker_code,
        "reason": reason,
        "attempted_output_dir": attempted_output_dir,
        "resolved_output_dir": str(resolved_path),
        "required_authority_root": FULL_UPDATE_AUTHORITY_ROOT_REL.as_posix(),
        "allowed_authority_roots": [
            root.as_posix() for root in FULL_UPDATE_AUTHORITY_ROOT_RELS
        ],
        "readonly_baseline_root": READONLY_BASELINE_AUTHORITY_ROOT_REL.as_posix(),
        "historical_root_prefixes": list(LEGACY_HISTORICAL_OUTPUT_ROOT_PREFIXES),
        "matched_legacy_root_prefix": matched_legacy_root_prefix,
    }


def resolve_full_update_authority_output_dir(
    repo_root: Path,
    raw: str | Path,
    *,
    require_v2_authority: bool,
) -> Path:
    resolved_path = _resolve_path(repo_root, str(raw))
    repo_relative = _repo_relative_str(repo_root, resolved_path)
    if _is_within_repo_relative_root(
        repo_relative, READONLY_BASELINE_AUTHORITY_ROOT_REL
    ):
        raise AuthorityRootBlocker(
            _authority_root_blocker_payload(
                repo_root=repo_root,
                resolved_path=resolved_path,
                blocker_code="readonly_baseline_root_blocked",
                reason=(
                    "single_gpu_v1 is a read-only baseline authority; new full-update-first "
                    "outputs must not write there"
                ),
            )
        )

    artifact_root_name = _artifact_root_name(repo_relative)
    if artifact_root_name is not None:
        for prefix in LEGACY_HISTORICAL_OUTPUT_ROOT_PREFIXES:
            if artifact_root_name.startswith(prefix):
                raise AuthorityRootBlocker(
                    _authority_root_blocker_payload(
                        repo_root=repo_root,
                        resolved_path=resolved_path,
                        blocker_code="historical_stage3_output_root_blocked",
                        reason=(
                            "stage3_t10-stage3_t13 roots are historical references only; "
                            "new full-update-first outputs must not write there"
                        ),
                        matched_legacy_root_prefix=prefix,
                    )
                )

    if require_v2_authority and not _is_within_any_repo_relative_root(
        repo_relative, FULL_UPDATE_AUTHORITY_ROOT_RELS
    ):
        raise AuthorityRootBlocker(
            _authority_root_blocker_payload(
                repo_root=repo_root,
                resolved_path=resolved_path,
                blocker_code="non_authority_output_root_blocked",
                reason=(
                    "full-update-first outputs must stay under an allowed v2 authority root"
                ),
            )
        )
    return resolved_path


def _planned_rank_count(num_gpus: int | None, *, visible_devices_policy: str) -> int:
    if visible_devices_policy == "dataset_gate_not_a_training_launch":
        return 0
    if num_gpus is None:
        return 1
    return max(int(num_gpus), 1)


def _default_visible_devices_policy(num_gpus: int | None) -> str:
    if num_gpus is None:
        return "launcher_default_visible_devices"
    if int(num_gpus) <= 1:
        return "single_gpu_explicit_visible_device_required"
    return "torchrun_visible_devices_contract_required"


def _default_launch_family(num_gpus: int | None) -> str:
    if num_gpus is None:
        return "launcher_default_launch_family"
    if int(num_gpus) <= 1:
        return SINGLE_GPU_LAUNCH_FAMILY
    return HISTORICAL_DDP_LAUNCH_FAMILY


def _green_smoke_verdict_path_for_launch_family(launch_family: str) -> Path:
    if launch_family == SINGLE_GPU_LAUNCH_FAMILY:
        return SINGLE_GPU_SMOKE_VERDICT_REL
    return DDP_GREEN_SMOKE_VERDICT_REL


def _green_smoke_gate_status(
    repo_root: Path, *, contract_role: str, launch_family: str
) -> str:
    if contract_role == "smoke_attempt":
        return "self_green_smoke_attempt"
    if contract_role not in {"formal_gate", "formal_run"}:
        return "not_part_of_stage3_green_smoke_gate"

    verdict_path = repo_root / _green_smoke_verdict_path_for_launch_family(launch_family)
    if not verdict_path.is_file():
        return "blocked_missing_green_smoke_verdict"

    try:
        payload = json.loads(verdict_path.read_text(encoding="utf-8"))
    except Exception:
        return "blocked_unreadable_green_smoke_verdict"
    if bool(payload.get("pass")):
        return "green_smoke_passed"
    return "blocked_green_smoke_not_green"


def _build_deterministic_artifact_contract(
    repo_root: Path,
    *,
    output_dir: Path,
    num_gpus: int | None,
) -> dict[str, Any]:
    output_rel = _repo_relative_str(repo_root, output_dir)
    contract = FROZEN_DIRECT_FINETUNE_OUTPUT_CONTRACTS.get(output_rel or "")
    visible_devices_policy = (
        str(contract["visible_devices_policy"])
        if contract is not None and contract.get("visible_devices_policy") is not None
        else _default_visible_devices_policy(num_gpus)
    )
    contract_role = (
        str(contract["contract_role"])
        if contract is not None and contract.get("contract_role") is not None
        else "generic_run"
    )
    launch_family = (
        str(contract["launch_family"])
        if contract is not None and contract.get("launch_family") is not None
        else _default_launch_family(num_gpus)
    )
    authority_status = (
        str(contract["authority_status"])
        if contract is not None and contract.get("authority_status") is not None
        else (
            "live_authority"
            if _is_within_any_repo_relative_root(
                output_rel,
                FULL_UPDATE_AUTHORITY_ROOT_RELS,
            )
            else "ad_hoc"
        )
    )
    metadata_dir = output_dir / DEFAULT_REPO_LOCAL_METADATA_DIRNAME
    rank_count = _planned_rank_count(
        num_gpus,
        visible_devices_policy=visible_devices_policy,
    )
    return {
        "contract_role": contract_role,
        "live_launch_family": launch_family,
        "direct_output_contract_path": output_rel or str(output_dir),
        "direct_output_contract_status": authority_status,
        "visible_devices_policy": visible_devices_policy,
        "torchrun_invoked": bool(rank_count > 1),
        "version_surface_path": str(metadata_dir / "version_surface.json"),
        "nvidia_smi_snapshot_path": str(metadata_dir / "nvidia_smi_snapshot.json"),
        "census_after_model_build_paths": [
            str(metadata_dir / f"census_after_model_build_rank{rank}.json")
            for rank in range(rank_count)
        ],
        "census_before_first_forward_paths": [
            str(metadata_dir / f"census_before_first_forward_rank{rank}.json")
            for rank in range(rank_count)
        ],
        "green_smoke_gate_status": _green_smoke_gate_status(
            repo_root,
            contract_role=contract_role,
            launch_family=launch_family,
        ),
    }


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


def _latest_checkpoint(output_dir: Path) -> Path | None:
    latest: tuple[int, float, Path] | None = None
    for path in sorted(output_dir.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        suffix = path.name.split("checkpoint-", 1)[-1]
        if not suffix.isdigit():
            continue
        candidate = (int(suffix), float(path.stat().st_mtime), path)
        if latest is None or candidate[:2] > latest[:2]:
            latest = candidate
    return None if latest is None else latest[2]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _jsonify_cmd(cmd: list[str]) -> list[str]:
    return [str(part) for part in cmd]


def _resolve_two_layer_python_contract(
    repo_root: Path,
    *,
    delegate_runtime_python_flag: str,
) -> dict[str, Any]:
    contract = demo_paths.load_stage3_training_python_contract(repo_root)
    launcher_python = str(demo_paths.current_python_abspath_preserve_symlink())
    orchestrator_python = str(contract["orchestrator_python"])
    if not demo_paths.same_abspath_preserve_symlink(
        launcher_python, orchestrator_python
    ):
        raise ValueError(
            "two-layer interpreter contract violated: "
            f"expected orchestrator_python={orchestrator_python}, got {launcher_python}"
        )

    requested_delegate_runtime = str(delegate_runtime_python_flag or "").strip()
    delegate_runtime_python = str(contract["delegate_runtime_python"])
    if requested_delegate_runtime and not demo_paths.same_abspath_preserve_symlink(
        requested_delegate_runtime,
        delegate_runtime_python,
    ):
        raise ValueError(
            "two-layer interpreter contract violated: "
            f"expected delegate_runtime_python={delegate_runtime_python}, got {requested_delegate_runtime}"
        )

    return {
        "contract_manifest_path": str(contract["manifest_path"]),
        "launcher_python": launcher_python,
        "orchestrator_python": orchestrator_python,
        "delegate_runtime_python": delegate_runtime_python,
        "delegate_runtime_python_requested": (
            None if not requested_delegate_runtime else requested_delegate_runtime
        ),
        "orchestrator_python_exists": bool(Path(orchestrator_python).is_file()),
        "delegate_runtime_python_exists": bool(Path(delegate_runtime_python).is_file()),
    }


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="3D_recap_finetune_full.py",
        description=(
            "Thin repo-local wrapper for the T11 full finetune entrypoint. It delegates to "
            "work/recap/scripts/34_recap_finetune_repro.py, which then launches "
            "the repo-local training wrapper work/recap/launch_finetune_use_ddp.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="LeRobot dataset directory used for finetuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Checkpoint output directory for the finetune run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(DEFAULT_MAX_STEPS),
        help="Upstream finetune --max-steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help=(
            "Upstream finetune --save-steps. If unset, this wrapper uses --max-steps so "
            "the run still retains a single final checkpoint."
        ),
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=int(DEFAULT_SAVE_TOTAL_LIMIT),
        help=(
            "Checkpoint retention budget. Repo governance requires exactly one retained "
            "checkpoint for this entrypoint."
        ),
    )
    parser.add_argument(
        "--runtime-log-dir",
        type=str,
        default=DEFAULT_RUNTIME_LOG_DIR,
        help="Directory for wrapper runtime logs when the delegated launch executes.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="Optional path to persist the wrapper summary JSON. Always printed to stdout.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base HF repo_id resolved by the delegated repro wrapper.",
    )
    parser.add_argument(
        "--base-model-revision",
        type=str,
        default="",
        help="Optional HF snapshot revision passed through to the delegated repro wrapper.",
    )
    parser.add_argument(
        "--hf-hub-cache-dir",
        type=str,
        default="",
        help="Optional HF hub cache root for locating the base model snapshot.",
    )
    parser.add_argument(
        "--patched-out-root",
        type=str,
        default=DEFAULT_PATCHED_OUT_ROOT,
        help="Repo-relative output root for patched HF base-model directories.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="",
        help="Python executable used by the delegated repro wrapper to run the repo-local training launcher.",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default=DEFAULT_EMBODIMENT_TAG,
        help="Embodiment tag forwarded to upstream finetune.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=int(DEFAULT_GLOBAL_BATCH_SIZE),
        help="Upstream finetune --global-batch-size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(DEFAULT_GRADIENT_ACCUMULATION_STEPS),
        help="Upstream finetune --gradient-accumulation-steps.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=int(DEFAULT_DATALOADER_NUM_WORKERS),
        help="Upstream finetune --dataloader-num-workers.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional upstream finetune --learning-rate override.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Optional upstream finetune --num-gpus override.",
    )
    _add_bool_group(
        parser,
        name="--tune-projector",
        dest="tune_projector",
        default=bool(DEFAULT_TUNE_PROJECTOR),
        help_text="Tune the projector module during finetuning.",
    )
    _add_bool_group(
        parser,
        name="--tune-diffusion-model",
        dest="tune_diffusion_model",
        default=bool(DEFAULT_TUNE_DIFFUSION_MODEL),
        help_text="Tune the diffusion model during finetuning.",
    )
    _add_bool_group(
        parser,
        name="--use-wandb",
        dest="use_wandb",
        default=bool(DEFAULT_USE_WANDB),
        help_text="Enable Weights & Biases logging.",
    )
    build_argparse_group(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve dataset/output/delegate paths and print a machine-readable summary without "
            "starting training."
        ),
    )
    return parser


def _build_delegate_cmd(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    forwarded: list[str],
) -> tuple[list[str], dict[str, Any]]:
    save_steps = (
        int(args.max_steps) if args.save_steps is None else int(args.save_steps)
    )
    if save_steps <= 0:
        raise ValueError(
            f"--save-steps must be > 0 when provided, got {args.save_steps!r}"
        )
    if int(args.save_total_limit) != int(DEFAULT_SAVE_TOTAL_LIMIT):
        raise ValueError(
            "This wrapper enforces single-checkpoint retention. "
            f"Expected --save-total-limit={DEFAULT_SAVE_TOTAL_LIMIT}, got {args.save_total_limit}."
        )
    if int(args.max_steps) <= 0:
        raise ValueError(f"--max-steps must be > 0, got {args.max_steps!r}")

    dataset_path = _resolve_path(repo_root, str(args.dataset_path))
    output_dir = resolve_full_update_authority_output_dir(
        repo_root,
        str(args.output_dir),
        require_v2_authority=True,
    )
    runtime_log_dir = _resolve_path(repo_root, str(args.runtime_log_dir))
    delegate_script = (repo_root / DEFAULT_DELEGATE_SCRIPT_REL).resolve()
    real_launcher_script = (repo_root / DEFAULT_REAL_LAUNCHER_REL).resolve()
    python_contract = _resolve_two_layer_python_contract(
        repo_root,
        delegate_runtime_python_flag=str(args.python).strip(),
    )

    cmd = [
        str(python_contract["launcher_python"]),
        str(delegate_script),
        "--base-model",
        str(args.base_model),
        "--dataset-path",
        str(dataset_path),
        "--embodiment-tag",
        str(args.embodiment_tag),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(int(args.max_steps)),
        "--save-steps",
        str(int(save_steps)),
        "--save-total-limit",
        str(int(DEFAULT_SAVE_TOTAL_LIMIT)),
        "--global-batch-size",
        str(int(args.global_batch_size)),
        "--gradient-accumulation-steps",
        str(int(args.gradient_accumulation_steps)),
        "--dataloader-num-workers",
        str(int(args.dataloader_num_workers)),
        "--tune-projector" if bool(args.tune_projector) else "--no-tune-projector",
        (
            "--tune-diffusion-model"
            if bool(args.tune_diffusion_model)
            else "--no-tune-diffusion-model"
        ),
        "--use-wandb" if bool(args.use_wandb) else "--no-use-wandb",
    ]
    if str(args.base_model_revision).strip():
        cmd.extend(["--base-model-revision", str(args.base_model_revision).strip()])
    if str(args.hf_hub_cache_dir).strip():
        cmd.extend(["--hf-hub-cache-dir", str(args.hf_hub_cache_dir).strip()])
    if str(args.patched_out_root).strip():
        cmd.extend(["--patched-out-root", str(args.patched_out_root).strip()])
    cmd.extend(["--python", str(python_contract["delegate_runtime_python"])])
    if args.learning_rate is not None:
        cmd.extend(["--learning-rate", str(float(args.learning_rate))])
    if args.num_gpus is not None:
        cmd.extend(["--num-gpus", str(int(args.num_gpus))])
    cmd.extend(list(forwarded))

    resolved = {
        "dataset_path": str(dataset_path),
        "dataset_exists": bool(dataset_path.is_dir()),
        "output_dir": str(output_dir),
        "runtime_log_dir": str(runtime_log_dir),
        "delegate_script": str(delegate_script),
        "delegate_script_exists": bool(delegate_script.is_file()),
        "real_launcher_script": str(real_launcher_script),
        "real_launcher_script_exists": bool(real_launcher_script.is_file()),
        "save_steps": int(save_steps),
        "save_total_limit": int(DEFAULT_SAVE_TOTAL_LIMIT),
        **python_contract,
    }
    return cmd, resolved


def _run_with_tee(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stderr.write(line)
                log_fp.write(line)
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait()
            return 130
    return int(proc.returncode)


def _emit_summary(payload: dict[str, Any], *, summary_json_path: Path | None) -> None:
    if summary_json_path is not None:
        _write_json(summary_json_path, payload)
    json.dump(payload, sys.stdout, ensure_ascii=True, indent=2, sort_keys=True)
    sys.stdout.write("\n")


class RecapFinetuneFullWorkflow:
    def run(self) -> int:
        parser = _build_parser()
        args, forwarded = parser.parse_known_args()
        recipe_flags = RecipeFlags.from_argparse(args)
        recipe_forwarded = recipe_flags_to_cli_args(recipe_flags)
        forwarded = [*recipe_forwarded, *list(forwarded)]
        repo_root = _repo_root()
        wrapper_ts = _dt.datetime.now().isoformat(timespec="seconds")
        summary_json_path = (
            _resolve_path(repo_root, str(args.summary_json))
            if str(args.summary_json).strip()
            else None
        )

        try:
            delegate_cmd, resolved = _build_delegate_cmd(
                repo_root=repo_root,
                args=args,
                forwarded=list(forwarded),
            )
            runtime_log_dir = Path(str(resolved["runtime_log_dir"]))
            runtime_log_path = (
                runtime_log_dir / f"3D_recap_finetune_full_{_timestamp()}.log"
            )
            deterministic_contract = _build_deterministic_artifact_contract(
                repo_root,
                output_dir=Path(str(resolved["output_dir"])),
                num_gpus=args.num_gpus,
            )
            train_scope_requested = _resolve_forwarded_train_scope(list(forwarded))
            train_scope_effective = train_scope_requested

            payload: dict[str, Any] = {
                "timestamp": wrapper_ts,
                "wrapper": "3D_recap_finetune_full.py",
                "wrapper_status": "ok",
                "dry_run": bool(args.dry_run),
                "repo_root": str(repo_root),
                "delegate_mode": "work/recap/scripts/34_recap_finetune_repro.py -> work/recap/launch_finetune_use_ddp.py",
                "delegate_cmd": _jsonify_cmd(delegate_cmd),
                "delegate_cmd_shell": shlex.join(delegate_cmd),
                "forwarded_passthrough_args": list(forwarded),
                "dataset_path": resolved["dataset_path"],
                "dataset_exists": bool(resolved["dataset_exists"]),
                "output_dir": resolved["output_dir"],
                "runtime_log_dir": resolved["runtime_log_dir"],
                "runtime_log_path": None
                if bool(args.dry_run)
                else str(runtime_log_path),
                "summary_json": None
                if summary_json_path is None
                else str(summary_json_path),
                "delegate_script": resolved["delegate_script"],
                "delegate_script_exists": bool(resolved["delegate_script_exists"]),
                "real_launcher_script": resolved["real_launcher_script"],
                "real_launcher_script_exists": bool(
                    resolved["real_launcher_script_exists"]
                ),
                "contract_manifest_path": resolved["contract_manifest_path"],
                "launcher_python": resolved["launcher_python"],
                "orchestrator_python": resolved["orchestrator_python"],
                "orchestrator_python_exists": bool(
                    resolved["orchestrator_python_exists"]
                ),
                "delegate_runtime_python": resolved["delegate_runtime_python"],
                "delegate_runtime_python_requested": resolved[
                    "delegate_runtime_python_requested"
                ],
                "delegate_runtime_python_exists": bool(
                    resolved["delegate_runtime_python_exists"]
                ),
                "formal_lane_live_authority": bool(
                    deterministic_contract["direct_output_contract_status"]
                    == "live_authority"
                ),
                "effective_config": {
                    "base_model": str(args.base_model),
                    "base_model_revision": str(args.base_model_revision),
                    "embodiment_tag": str(args.embodiment_tag),
                    "launch_family": deterministic_contract["live_launch_family"],
                    "max_steps": int(args.max_steps),
                    "save_steps": int(resolved["save_steps"]),
                    "save_total_limit": int(resolved["save_total_limit"]),
                    "global_batch_size": int(args.global_batch_size),
                    "gradient_accumulation_steps": int(
                        args.gradient_accumulation_steps
                    ),
                    "dataloader_num_workers": int(args.dataloader_num_workers),
                    "learning_rate": (
                        None
                        if args.learning_rate is None
                        else float(args.learning_rate)
                    ),
                    "num_gpus": None if args.num_gpus is None else int(args.num_gpus),
                    "use_ddp": _auto_use_ddp(args.num_gpus),
                    "visible_devices_policy": deterministic_contract[
                        "visible_devices_policy"
                    ],
                    "torchrun_invoked": bool(
                        deterministic_contract["torchrun_invoked"]
                    ),
                    "tune_projector": bool(args.tune_projector),
                    "tune_diffusion_model": bool(args.tune_diffusion_model),
                    "use_wandb": bool(args.use_wandb),
                    **_effective_batch_geometry(
                        global_batch_size=int(args.global_batch_size),
                        gradient_accumulation_steps=int(
                            args.gradient_accumulation_steps
                        ),
                        num_gpus=args.num_gpus,
                    ),
                },
                "upstream_returncode": None,
                "selected_checkpoint_path": None,
                "selected_checkpoint_exists": False,
                "selected_checkpoint_asset_path": None,
                **deterministic_contract,
                "comparability_manifest_path": str(
                    Path(str(resolved["output_dir"])) / COMPARABILITY_MANIFEST_FILENAME
                ),
                "comparability_manifest_status": None,
                "comparability_manifest_blocker_code": None,
                "error": None,
            }

            output_dir = Path(str(resolved["output_dir"]))
            output_dir.mkdir(parents=True, exist_ok=True)
            comparability_manifest = emit_full_update_comparability_manifest(
                repo_root=repo_root,
                output_dir=output_dir,
                warm_start_checkpoint=str(args.base_model).strip(),
                global_batch_size=int(args.global_batch_size),
                gradient_accumulation_steps=int(args.gradient_accumulation_steps),
                num_gpus=args.num_gpus,
                dataset_path=str(resolved["dataset_path"]),
                launch_family=deterministic_contract["live_launch_family"],
                train_scope_requested=train_scope_requested,
                train_scope_effective=train_scope_effective,
                advantage_consumed=True,
            )
            payload["comparability_manifest_status"] = comparability_manifest["status"]
            payload["comparability_manifest_blocker_code"] = comparability_manifest.get(
                "blocker_code"
            )

            if not bool(resolved["delegate_script_exists"]):
                raise FileNotFoundError(
                    f"Delegate script not found: {resolved['delegate_script']}"
                )
            if not bool(resolved["real_launcher_script_exists"]):
                raise FileNotFoundError(
                    f"Real launcher script not found: {resolved['real_launcher_script']}"
                )
            if not bool(resolved["dataset_exists"]):
                raise FileNotFoundError(
                    f"Dataset path not found: {resolved['dataset_path']}"
                )

            if bool(args.dry_run):
                _emit_summary(payload, summary_json_path=summary_json_path)
                return 0

            runtime_log_dir.mkdir(parents=True, exist_ok=True)

            rc = _run_with_tee(delegate_cmd, cwd=repo_root, log_path=runtime_log_path)
            payload["upstream_returncode"] = int(rc)
            if rc != 0:
                payload["wrapper_status"] = "blocked"
                payload["error"] = f"delegated_finetune_failed: returncode={rc}"
            selected_checkpoint = _latest_checkpoint(output_dir)
            selected_checkpoint_asset = _selected_checkpoint_asset(selected_checkpoint)
            payload["selected_checkpoint_path"] = (
                str(selected_checkpoint)
                if selected_checkpoint is not None
                and selected_checkpoint_asset is not None
                else None
            )
            payload["selected_checkpoint_exists"] = bool(
                selected_checkpoint_asset is not None
            )
            payload["selected_checkpoint_asset_path"] = (
                str(selected_checkpoint_asset)
                if selected_checkpoint_asset is not None
                else None
            )
            _emit_summary(payload, summary_json_path=summary_json_path)
            return int(rc)
        except Exception as exc:
            payload = {
                "timestamp": wrapper_ts,
                "wrapper": "3D_recap_finetune_full.py",
                "wrapper_status": "blocked",
                "dry_run": bool(getattr(args, "dry_run", False)),
                "repo_root": str(repo_root),
                "launcher_python": str(
                    demo_paths.current_python_abspath_preserve_symlink()
                ),
                "live_launch_family": None,
                "direct_output_contract_path": None,
                "direct_output_contract_status": None,
                "formal_lane_live_authority": False,
                "visible_devices_policy": None,
                "torchrun_invoked": None,
                "version_surface_path": None,
                "nvidia_smi_snapshot_path": None,
                "census_after_model_build_paths": [],
                "census_before_first_forward_paths": [],
                "green_smoke_gate_status": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            _emit_summary(payload, summary_json_path=summary_json_path)
            return 1


def main() -> int:
    return RecapFinetuneFullWorkflow().run()


if __name__ == "__main__":
    raise SystemExit(main())


class RecapFinetuneFullScriptApp:
    def run(self) -> int:
        return main()
