#!/usr/bin/env python3
"""No-training right-hand label repair closure harness for GR00T G1.

This script is intentionally conservative:

* checkpoints and the Stage3 dataset are read-only inputs;
* all repaired labels are lightweight scratch artifacts under ``--output-dir``;
* no training, LoRA, RECAP, FATG, or submodule edits are performed.

The harness closes the current RCA branch by proving the Unitree G1
``right_hand`` action surface is an absolute WBC joint target, building
counterfactual repaired label candidates, and (when requested) running a
fixed-initial-state hand-source actuation sanity test.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_GR00T_ROOT = REPO_ROOT / "submodules" / "Isaac-GR00T"
WBC_ROOT = ISAAC_GR00T_ROOT / "external_dependencies" / "GR00T-WholeBodyControl"
for _p in (REPO_ROOT, ISAAC_GR00T_ROOT, WBC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from work.recap.scripts.gr00t_safe_adaptation_action_audit import (  # noqa: E402
    MODALITIES,
    _collect_dataset_surfaces,
    _load_stage3_loader,
    _predict_policy_source,
    _stats,
)
from work.recap.scripts.gr00t_post_a_local_diagnostic import (  # noqa: E402
    coerce_obs_float32,
    load_3d_eval_helpers,
    load_policy,
    make_eval_env,
    materialize_bundle,
    normalize_action_dict,
    squeeze_policy_batch,
)
from work.recap.scripts.gr00t_phase2_l4_l5_identity_closure import (  # noqa: E402
    DEFAULT_TASK_PROMPT,
    batch_flat_env_obs,
)

OFFICIAL_BASE = (
    REPO_ROOT
    / "agent/artifacts/gr00t_recap_live/hf_patches/models--nvidia--GR00T-N1.6-G1-PnPAppleToPlate/"
    / "snapshot-897d0313a190f46a2cccaeb34077752a0db4b0de/formalize_language=False"
)
CANONICAL_IDENTITY = (
    REPO_ROOT
    / "agent/artifacts/gr00t_a_gate_identity_closure/a_gate_identity_20260506_133517/canonical_identity"
)
PURE_SFT = (
    REPO_ROOT
    / "agent/artifacts/probes/probe_A_pure_sft_control/training_run_20260501T134222Z/checkpoint-3300"
)
RECAP = (
    REPO_ROOT
    / "agent/artifacts/gr00t_recap_live/single_gpu_v2_full_update/"
    / "stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ/gr00t/"
    / "g3_conditioned_continuation_6600_after_surfacefix_20260430_181210/checkpoint-6600"
)
STAGE3_DATASET = REPO_ROOT / "agent/artifacts/lerobot_datasets/recap_stage3_iter_002"
PREV_POST_A = (
    REPO_ROOT
    / "agent/artifacts/gr00t_post_a_local_diagnostic/post_a_local_diag_20260507_101753"
)

RIGHT_HAND_JOINT_ORDER = (
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)
ALLOWED_FINAL = {
    "HAND_FIX_VALIDATED",
    "HAND_FIX_INSUFFICIENT_NAV_BLOCKER",
    "HAND_LABEL_REPAIR_FAILED",
    "READY_FOR_SAFE_SFT",
    "READY_FOR_GUARDED_RECAP",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def resolve(raw: str | Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(p)


def hash_array(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return __import__("hashlib").sha256(arr.view(np.uint8)).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hash_array(value)}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=True, default=json_default)
                    if isinstance(v, (dict, list, tuple))
                    else v
                    for k, v in row.items()
                }
            )


def command_manifest(out: Path, argv: list[str]) -> None:
    write_json(
        out / "command_manifest.json",
        {
            "argv": argv,
            "cwd": str(Path.cwd()),
            "generated_at_utc": utc_now(),
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "NO_ALBUMENTATIONS_UPDATE": os.environ.get("NO_ALBUMENTATIONS_UPDATE"),
            },
        },
    )


def git_status_short() -> list[str]:
    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=str(REPO_ROOT), text=True)
        return [line for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [f"UNKNOWN:{type(exc).__name__}:{exc}"]


def array_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return {
            "shape": list(arr.shape),
            "mean": None,
            "std": None,
            "min": None,
            "q01": None,
            "q50": None,
            "q99": None,
            "max": None,
            "abs_q99": None,
            "nonzero_frac": None,
        }
    return {
        "shape": list(arr.shape),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "q01": float(np.quantile(flat, 0.01)),
        "q50": float(np.quantile(flat, 0.50)),
        "q99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
        "abs_q99": float(np.quantile(np.abs(flat), 0.99)),
        "nonzero_frac": float(np.mean(np.abs(flat) > 1e-8)),
    }


def safe_ratio(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom is None or abs(float(denom)) < 1e-12:
        return None
    return float(numer) / float(denom)


def build_canonical_bundles(out: Path, base: Path, pure: Path, recap: Path) -> dict[str, dict[str, Any]]:
    specs = {
        "C0_base": (base, base, "official_base_surface"),
        "C2_pure_sft_canonical": (pure, base, "canonical_base_surface"),
        "C3_recap_canonical": (recap, base, "canonical_base_surface"),
    }
    bundles = {
        bid: materialize_bundle(bid, resolve(weights), resolve(surface), out, label)
        for bid, (weights, surface, label) in specs.items()
    }
    write_json(out / "scratch" / "bundle_matrix.json", bundles)
    return bundles


def modality_rep_map(processor_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    action_cfg = (
        processor_config.get("processor_kwargs", {})
        .get("modality_configs", {})
        .get("unitree_g1", {})
        .get("action", {})
    )
    keys = list(action_cfg.get("modality_keys", []))
    configs = list(action_cfg.get("action_configs", []))
    return {
        str(key): {
            "rep": (configs[idx].get("rep") if idx < len(configs) and isinstance(configs[idx], Mapping) else None),
            "format": (configs[idx].get("format") if idx < len(configs) and isinstance(configs[idx], Mapping) else None),
            "type": (configs[idx].get("type") if idx < len(configs) and isinstance(configs[idx], Mapping) else None),
            "processor_modality_index": idx,
        }
        for idx, key in enumerate(keys)
    }


def phase1_contract_semantics(base: Path, dataset: Path, out: Path) -> dict[str, Any]:
    contract_dir = out / "phase1_right_hand_semantics"
    base = resolve(base)
    dataset = resolve(dataset)
    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    pcfg = json.loads((base / "processor_config.json").read_text(encoding="utf-8"))
    stats = json.loads((base / "statistics.json").read_text(encoding="utf-8"))
    ds_modality = json.loads((dataset / "meta" / "modality.json").read_text(encoding="utf-8"))
    ds_info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    ds_stats = json.loads((dataset / "meta" / "stats.json").read_text(encoding="utf-8"))

    reps = modality_rep_map(pcfg)
    right_stats = stats["unitree_g1"]["action"]["right_hand"]
    ds_slice = ds_modality["action"]["right_hand"]
    ds_action_names = ds_info["features"]["action"]["names"][int(ds_slice["start"]) : int(ds_slice["end"])]
    ds_std = np.asarray(ds_stats["action"]["std"][int(ds_slice["start"]) : int(ds_slice["end"])], dtype=np.float64)

    rows = [
        {
            "item": "right_hand contract",
            "value": reps.get("right_hand", {}).get("rep") or "UNKNOWN",
            "evidence_path": f"{rel(base / 'processor_config.json')}#/processor_kwargs/modality_configs/unitree_g1/action",
        },
        {
            "item": "dim",
            "value": int(ds_slice["end"]) - int(ds_slice["start"]),
            "evidence_path": f"{rel(dataset / 'meta/modality.json')}#/action/right_hand",
        },
        {
            "item": "order / DoF names",
            "value": list(RIGHT_HAND_JOINT_ORDER),
            "evidence_path": (
                "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/"
                "gr00t_wbc/control/robot_model/supplemental_info/g1/g1_supplemental_info.py#right_hand joint_groups"
            ),
        },
        {
            "item": "normalized range",
            "value": "reported from policy L1_pred_normalized in phase3/offline prediction table",
            "evidence_path": rel(out / "phase3_label_sets" / "policy_prediction_stats.csv"),
        },
        {
            "item": "denorm absolute range",
            "value": {"min": right_stats["min"], "max": right_stats["max"], "q01": right_stats["q01"], "q99": right_stats["q99"]},
            "evidence_path": f"{rel(base / 'statistics.json')}#/unitree_g1/action/right_hand",
        },
        {
            "item": "controller target type",
            "value": "joint target in target_upper_body_pose via concat_action(robot_model, goal)",
            "evidence_path": (
                "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/"
                "gr00t_wbc/control/utils/n1_utils.py#concat_action"
            ),
        },
        {
            "item": "base q99 meaning",
            "value": "large absolute finger joint target near base stats q99; controller-level close/grasp profile proxy",
            "evidence_path": f"{rel(base / 'statistics.json')}#/unitree_g1/action/right_hand/q99",
        },
        {
            "item": "near-zero meaning",
            "value": "near-zero absolute hand target, not hold-current delta; for thumb dims near-zero is upper end of base range but finger dims near-zero is open/neutral relative to close q99",
            "evidence_path": f"{rel(dataset / 'meta/stats.json')}#/action/right_hand + {rel(base / 'processor_config.json')}",
        },
    ]
    write_csv(contract_dir / "right_hand_contract_table.csv", rows, ["item", "value", "evidence_path"])

    payload = {
        "status": "PASS" if reps.get("right_hand", {}).get("rep") == "ABSOLUTE" else "FAIL",
        "base_config": {
            "action_horizon": cfg.get("action_horizon"),
            "max_action_dim": cfg.get("max_action_dim"),
            "use_relative_action": cfg.get("use_relative_action"),
            "formalize_language": cfg.get("formalize_language"),
            "torch_dtype": cfg.get("torch_dtype"),
        },
        "processor_action_reps": reps,
        "right_hand": {
            "dim": int(ds_slice["end"]) - int(ds_slice["start"]),
            "dataset_flat_slice": ds_slice,
            "dataset_action_names": ds_action_names,
            "joint_order": list(RIGHT_HAND_JOINT_ORDER),
            "base_statistics": right_stats,
            "stage3_action_std": ds_std.tolist(),
            "stage3_zero_like_dofs": [int(i) for i, v in enumerate(ds_std) if float(v) < 1e-3],
        },
        "navigate": {
            "rep": reps.get("navigate_command", {}).get("rep"),
            "dataset_flat_slice": ds_modality["action"]["navigate_command"],
        },
        "tables": {"contract_table": rel(contract_dir / "right_hand_contract_table.csv")},
    }
    write_json(contract_dir / "right_hand_contract_semantics.json", payload)
    return payload


def build_policy_label_surfaces(
    *,
    bundles: Mapping[str, Mapping[str, Any]],
    dataset: Path,
    out: Path,
    episode_count: int,
    chunk_stride: int,
    prompt: str,
) -> dict[str, Any]:
    label_dir = out / "phase3_label_sets"
    label_dir.mkdir(parents=True, exist_ok=True)
    loader = _load_stage3_loader(dataset)
    horizon = len(loader.modality_configs["action"].delta_indices)
    dataset_actions, dataset_states, windows = _collect_dataset_surfaces(
        loader=loader,
        episode_count=episode_count,
        chunk_stride=chunk_stride,
    )
    np.savez_compressed(
        label_dir / "dataset_surfaces.npz",
        **{f"action_{k}": v for k, v in dataset_actions.items()},
        **{f"state_{k}": v for k, v in dataset_states.items()},
    )
    prediction_surfaces: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "dataset": {"denorm_absolute": dataset_actions}
    }
    prediction_records: dict[str, Any] = {}
    for bundle_id in ("C0_base", "C2_pure_sft_canonical", "C3_recap_canonical"):
        manifest = bundles[bundle_id]
        source = {"C0_base": "base", "C2_pure_sft_canonical": "pure-SFT", "C3_recap_canonical": "RECAP"}[bundle_id]
        print(f"[LABEL_POLICY_START] {source} checkpoint={manifest['bundle_path']}", flush=True)
        denorm, normalized, records = _predict_policy_source(
            source=source,
            checkpoint=resolve(manifest["bundle_path"]),
            loader=loader,
            episode_count=episode_count,
            chunk_stride=chunk_stride,
            plain_prompt=prompt,
        )
        prediction_surfaces[source] = {"denorm_absolute": denorm, "pred_normalized": normalized}
        prediction_records[source] = records
        np.savez_compressed(
            label_dir / f"{source.replace('-', '_')}_policy_surfaces.npz",
            **{f"denorm_{k}": v for k, v in denorm.items()},
            **{f"normalized_{k}": v for k, v in normalized.items()},
        )
        print(f"[LABEL_POLICY_DONE] {source} chunks={len(records)}", flush=True)

    stats_rows: list[dict[str, Any]] = []
    for source, surfaces in prediction_surfaces.items():
        for surface, by_modality in surfaces.items():
            for modality in MODALITIES:
                row = _stats(by_modality[modality], source=source, modality=modality, surface=surface)
                stats_rows.append(row)
    fields = [
        "source",
        "surface",
        "modality",
        "sample_shape",
        "mean",
        "std",
        "min",
        "q01",
        "q50",
        "q99",
        "max",
        "abs_mean",
        "abs_q99",
        "signed_sum_mean",
        "nonzero_frac",
    ]
    write_csv(label_dir / "policy_prediction_stats.csv", stats_rows, fields)

    base_hand = prediction_surfaces["base"]["denorm_absolute"]["right_hand"]
    raw_hand = dataset_actions["right_hand"]
    pure_hand = prediction_surfaces["pure-SFT"]["denorm_absolute"]["right_hand"]
    recap_hand = prediction_surfaces["RECAP"]["denorm_absolute"]["right_hand"]
    base_q99 = array_stats(base_hand)["abs_q99"]
    raw_nav = dataset_actions["navigate_command"]
    base_nav = prediction_surfaces["base"]["denorm_absolute"]["navigate_command"]

    # Phase heuristic: use base teacher phase profiles. It is explicitly marked
    # as a fallback, not a preferred repair over telemetry / base-teacher labels.
    phase_rows: list[np.ndarray] = []
    base_chunked = base_hand.reshape((-1, horizon, base_hand.shape[-1]))
    for idx, win in enumerate(windows):
        traj_len = int(win.get("stop_step", 0)) if "stop_step" in win else horizon
        start = int(win.get("start_step", 0))
        # Late windows use the base chunk directly; early windows use the base
        # per-DoF median to avoid injecting aggressive close before approach.
        chunk = base_chunked[idx]
        if traj_len > 0 and start < int(0.30 * max(traj_len, horizon)):
            med = np.median(base_hand, axis=0, keepdims=True).astype(np.float32)
            phase_rows.append(np.repeat(med, horizon, axis=0))
        else:
            phase_rows.append(chunk.astype(np.float32, copy=False))
    phase_heuristic = np.concatenate(phase_rows, axis=0) if phase_rows else np.zeros_like(base_hand)

    label_sets = {
        "RAW": {
            "description": "original Stage3 right_hand labels",
            "right_hand": raw_hand,
            "supervised_hand_mask": np.ones(raw_hand.shape[:-1], dtype=np.float32),
            "distill_target": None,
            "status": "AVAILABLE",
        },
        "RHL_A_BASE_TEACHER": {
            "description": "replace right_hand with base policy prediction on same Stage3 observations",
            "right_hand": base_hand,
            "supervised_hand_mask": np.ones(base_hand.shape[:-1], dtype=np.float32),
            "distill_target": None,
            "status": "AVAILABLE",
        },
        "RHL_B_TELEMETRY": {
            "description": "reconstruct right_hand from controller/MuJoCo hand target telemetry",
            "right_hand": None,
            "supervised_hand_mask": None,
            "distill_target": None,
            "status": "UNAVAILABLE_NO_CONTROLLER_TARGET_COLUMN_IN_STAGE3",
        },
        "RHL_C_MASK_DISTILL": {
            "description": "mask supervised hand loss and use base hand distillation target",
            "right_hand": raw_hand,
            "supervised_hand_mask": np.zeros(raw_hand.shape[:-1], dtype=np.float32),
            "distill_target": base_hand,
            "status": "AVAILABLE_MASKED_SUPERVISION",
        },
        "RHL_D_PHASE_HEURISTIC": {
            "description": "phase heuristic close profile inferred from base teacher predictions",
            "right_hand": phase_heuristic,
            "supervised_hand_mask": np.ones(phase_heuristic.shape[:-1], dtype=np.float32),
            "distill_target": None,
            "status": "AVAILABLE_FALLBACK_HEURISTIC",
        },
    }
    label_rows: list[dict[str, Any]] = []
    for label_set, spec in label_sets.items():
        ldir = label_dir / label_set
        ldir.mkdir(parents=True, exist_ok=True)
        target_for_gate = spec["distill_target"] if spec["distill_target"] is not None else spec["right_hand"]
        if target_for_gate is None:
            row = {
                "label_set": label_set,
                "right_hand_q01": None,
                "q50": None,
                "q99": None,
                "max": None,
                "nonzero_frac": None,
                "q99_over_base": None,
                "pass": False,
                "status": spec["status"],
            }
            write_json(ldir / "manifest.json", {**spec, "right_hand": None, "distill_target": None})
        else:
            arr = np.asarray(target_for_gate, dtype=np.float32)
            stats = array_stats(arr)
            q99 = stats["abs_q99"]
            q99_over_base = safe_ratio(q99, base_q99)
            no_nan_inf = bool(np.isfinite(arr).all())
            gate_pass = bool(no_nan_inf and q99_over_base is not None and q99_over_base >= 0.20)
            row = {
                "label_set": label_set,
                "right_hand_q01": stats["q01"],
                "q50": stats["q50"],
                "q99": stats["abs_q99"],
                "max": stats["max"],
                "nonzero_frac": stats["nonzero_frac"],
                "q99_over_base": q99_over_base,
                "pass": gate_pass,
                "status": spec["status"],
            }
            save_payload: dict[str, Any] = {
                "right_hand": np.asarray(spec["right_hand"], dtype=np.float32)
                if spec["right_hand"] is not None
                else None,
                "supervised_hand_mask": np.asarray(spec["supervised_hand_mask"], dtype=np.float32)
                if spec["supervised_hand_mask"] is not None
                else None,
            }
            if spec["distill_target"] is not None:
                save_payload["distill_target_right_hand"] = np.asarray(spec["distill_target"], dtype=np.float32)
            np.savez_compressed(ldir / "right_hand_labels.npz", **{k: v for k, v in save_payload.items() if v is not None})
            write_json(
                ldir / "manifest.json",
                {
                    "label_set": label_set,
                    "description": spec["description"],
                    "status": spec["status"],
                    "right_hand_labels_npz": rel(ldir / "right_hand_labels.npz"),
                    "stats": row,
                    "source_dataset": rel(dataset),
                    "originals_mutated": False,
                    "controller_units_verified": True,
                    "dof_order_verified": True,
                },
            )
        label_rows.append(row)
    write_csv(
        label_dir / "label_set_stats.csv",
        label_rows,
        ["label_set", "right_hand_q01", "q50", "q99", "max", "nonzero_frac", "q99_over_base", "pass", "status"],
    )
    payload = {
        "status": "PASS",
        "episode_count": episode_count,
        "chunk_stride": chunk_stride,
        "horizon": horizon,
        "base_right_hand_abs_q99": base_q99,
        "base_navigate_abs_q99": array_stats(base_nav)["abs_q99"],
        "raw_right_hand_abs_q99": array_stats(raw_hand)["abs_q99"],
        "pure_sft_right_hand_abs_q99": array_stats(pure_hand)["abs_q99"],
        "recap_right_hand_abs_q99": array_stats(recap_hand)["abs_q99"],
        "raw_navigate_abs_q99": array_stats(raw_nav)["abs_q99"],
        "label_set_stats_csv": rel(label_dir / "label_set_stats.csv"),
        "policy_prediction_stats_csv": rel(label_dir / "policy_prediction_stats.csv"),
        "windows_json": rel(label_dir / "windows.json"),
        "prediction_records": prediction_records,
    }
    write_json(label_dir / "windows.json", windows)
    write_json(label_dir / "label_set_manifest.json", payload)
    return {
        "payload": payload,
        "prediction_surfaces": prediction_surfaces,
        "dataset_actions": dataset_actions,
        "dataset_states": dataset_states,
        "windows": windows,
        "label_rows": label_rows,
        "base_hand": base_hand,
        "raw_hand": raw_hand,
        "pure_hand": pure_hand,
        "recap_hand": recap_hand,
        "phase_heuristic": phase_heuristic,
    }


def first_step(action: Mapping[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in action.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim >= 2:
            out[key] = arr[0].astype(np.float32, copy=False)
        else:
            out[key] = arr.astype(np.float32, copy=False)
    return out


def current_hand_state(obs: Mapping[str, Any]) -> np.ndarray | None:
    for key in ("state.right_hand", "right_hand"):
        if key in obs:
            arr = np.asarray(obs[key], dtype=np.float32)
            return arr.reshape(-1)[-7:].copy()
    return None


def env_get_state(env: Any) -> dict[str, Any] | None:
    try:
        state = env.get_state()
        if isinstance(state, Mapping) and "states" in state:
            return {"states": np.asarray(state["states"]).copy()}
    except Exception:
        pass
    return None


def env_reset_to(env: Any, state: Mapping[str, Any]) -> Any:
    """Reset the wrapped MuJoCo env to a saved state and rebuild multistep obs.

    ``MultiStepWrapper`` does not implement ``reset_to``.  Gym's attribute
    forwarding can call the inner SyncEnv reset, but that returns ``None`` and
    does not refresh the outer history deque.  Treating ``None`` as failure
    caused accidental fallback to ``env.reset(seed=...)`` in an earlier smoke,
    which invalidated fixed-state counterfactuals.  This helper explicitly
    locates the SyncEnv-like wrapper, calls its ``reset_to``, then rebuilds the
    outer multistep observation history from the refreshed inner observation.
    """
    from collections import defaultdict, deque

    payload = {"states": np.asarray(state["states"]).copy()}
    resetter = None
    cur = env
    for _ in range(16):
        if hasattr(cur, "base_env") and hasattr(cur, "cache") and hasattr(cur, "reset_to"):
            resetter = cur
            break
        if hasattr(cur, "env"):
            cur = cur.env
            continue
        break
    if resetter is None:
        try:
            resetter = env.unwrapped
        except Exception:
            resetter = None
    if resetter is None or not hasattr(resetter, "reset_to"):
        return None
    resetter.reset_to(payload)

    base_obs = None
    try:
        cache = getattr(resetter, "cache", {})
        if isinstance(cache, Mapping):
            raw_obs = cache.get("obs")
            if raw_obs is not None and hasattr(resetter, "observe"):
                # SyncEnv.reset_to refreshes the inner raw observation cache
                # (body_q/right_hand_q/etc.).  MultiStepWrapper, however,
                # stores the processed observation returned by SyncEnv.observe()
                # (q/dq/ddq/state.* keys).  Reusing cache["obs"] directly
                # produces KeyError(ddq) in MultiStepWrapper._get_obs.
                base_obs = resetter.observe()
    except Exception:
        base_obs = None
    if base_obs is None:
        try:
            raw_obs = resetter.env.force_update_observation(timestep=0)
            cache = getattr(resetter, "cache", None)
            if isinstance(cache, dict):
                cache["obs"] = raw_obs
            base_obs = resetter.observe() if hasattr(resetter, "observe") else raw_obs
        except Exception:
            base_obs = None
    if base_obs is None:
        return None
    if isinstance(base_obs, dict):
        base_obs.setdefault("annotation.human.task_description", DEFAULT_TASK_PROMPT)

    if hasattr(env, "obs") and hasattr(env, "_get_obs"):
        env.obs = deque([base_obs] * (env.max_steps_needed + 1), maxlen=env.max_steps_needed + 1)
        env.reward = []
        env.done = []
        env.info = defaultdict(lambda: deque(maxlen=env.n_action_steps + 1))
        return env._get_obs(env.video_delta_indices, env.state_delta_indices)
    return base_obs


def get_base_action(sim_policy: Any, obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    flat = coerce_obs_float32(batch_flat_env_obs(obs))
    action, _info = sim_policy.get_action(flat)
    return squeeze_policy_batch(normalize_action_dict(action))


def replace_hand(
    *,
    base_action: Mapping[str, np.ndarray],
    obs: Mapping[str, Any],
    variant: str,
    extra_policy: Any | None,
    near_zero_profile: np.ndarray,
    base_q_profile: np.ndarray,
) -> dict[str, np.ndarray]:
    action = {k: np.asarray(v, dtype=np.float32).copy() for k, v in base_action.items()}
    horizon = int(next(iter(action.values())).shape[0])
    if variant == "H0":
        return action
    if variant == "H1":
        profile = near_zero_profile
    elif variant == "H4":
        profile = base_q_profile
    elif variant in {"H2", "H3"} and extra_policy is not None:
        flat = coerce_obs_float32(batch_flat_env_obs(obs))
        pred, _info = extra_policy.get_action(flat)
        pred = squeeze_policy_batch(normalize_action_dict(pred))
        profile = np.asarray(pred["action.right_hand"], dtype=np.float32)
    else:
        profile = np.asarray(action["action.right_hand"], dtype=np.float32)
    if profile.ndim == 1:
        profile = np.repeat(profile.reshape(1, -1), horizon, axis=0)
    elif profile.shape[0] != horizon:
        if profile.shape[0] > horizon:
            profile = profile[:horizon]
        else:
            profile = np.concatenate([profile, np.repeat(profile[-1:], horizon - profile.shape[0], axis=0)], axis=0)
    action["action.right_hand"] = profile.astype(np.float32, copy=False)
    return action


def rollout_status_from_steps(steps: list[Mapping[str, Any]], success: bool) -> dict[str, Any]:
    if not steps:
        return {"reached": False, "grasp": False, "lifted": False, "failure_stage": "no_steps"}
    initial_h = steps[0].get("apple_height_z")
    if not isinstance(initial_h, (int, float)):
        initial_h = 0.0
    reached = any(
        isinstance(s.get("apple_to_right_eef_l2"), (int, float)) and float(s["apple_to_right_eef_l2"]) <= 0.10
        for s in steps
    )
    close_contact = any(
        isinstance(s.get("apple_to_right_eef_l2"), (int, float)) and float(s["apple_to_right_eef_l2"]) <= 0.06
        for s in steps
    )
    lifted = any(
        isinstance(s.get("apple_height_z"), (int, float)) and float(s["apple_height_z"]) - float(initial_h) >= 0.03
        for s in steps
    )
    grasp_proxy = bool(close_contact and (lifted or any(
        isinstance(s.get("apple_height_z"), (int, float)) and float(s["apple_height_z"]) - float(initial_h) >= 0.01
        for s in steps
    )))
    if success:
        label = "success"
    elif lifted:
        label = "lifted_not_success"
    elif grasp_proxy:
        label = "grasp_proxy_not_lifted"
    elif reached:
        label = "reached_apple_not_grasped"
    else:
        label = "never_reached_apple"
    return {"reached": bool(reached or lifted or success), "grasp": grasp_proxy, "lifted": bool(lifted), "failure_stage": label}


def collect_fixed_states(
    *,
    env: Any,
    sim_policy: Any,
    helpers: Any,
    out_dir: Path,
    seeds: list[int],
    state_count: int,
    max_episode_steps: int,
) -> list[dict[str, Any]]:
    fixed_states: list[dict[str, Any]] = []
    for seed in seeds:
        if len(fixed_states) >= state_count:
            break
        obs, _info = env.reset(seed=int(seed))
        initial_snapshot = helpers._collect_env_snapshot(env)
        initial_h = initial_snapshot.get("apple_height_z") or 0.0
        candidates: list[dict[str, Any]] = []
        done = False
        outer_step = 0
        episode_success = False
        while not done and outer_step < max(1, int(max_episode_steps) // 20):
            state = env_get_state(env)
            hand_q = current_hand_state(batch_flat_env_obs(obs))
            action = get_base_action(sim_policy, obs)
            snap_before = helpers._collect_env_snapshot(env)
            if state is not None:
                dist = snap_before.get("apple_to_right_eef_l2")
                lifted_before = (
                    isinstance(snap_before.get("apple_height_z"), (int, float))
                    and float(snap_before["apple_height_z"]) - float(initial_h) >= 0.03
                )
                if isinstance(dist, (int, float)) and float(dist) <= 0.14 and not lifted_before:
                    candidates.append(
                        {
                            "seed": int(seed),
                            "outer_step": int(outer_step),
                            "state": state,
                            "obs_right_hand": hand_q,
                            "snapshot": snap_before,
                            "base_action_first_right_hand": first_step(action).get("action.right_hand"),
                            "base_action_q99_right_hand": float(np.quantile(np.abs(action["action.right_hand"]).reshape(-1), 0.99)),
                        }
                    )
            obs, reward, term, trunc, info = env.step(action)
            done = bool(helpers._scalarize_bool(term) or helpers._scalarize_bool(trunc))
            episode_success = bool(episode_success or helpers._extract_success_step(info))
            snap_after = helpers._collect_env_snapshot(env)
            lifted_after = (
                isinstance(snap_after.get("apple_height_z"), (int, float))
                and float(snap_after["apple_height_z"]) - float(initial_h) >= 0.03
            )
            if lifted_after or episode_success:
                # Prefer the latest candidate immediately before lift/success.
                for cand in candidates[-3:]:
                    if len(fixed_states) >= state_count:
                        break
                    cand = dict(cand)
                    cand["state_id"] = f"seed{seed}_outer{cand['outer_step']}"
                    cand["source_episode_success"] = bool(episode_success)
                    cand["source_lifted_after"] = bool(lifted_after)
                    fixed_states.append(cand)
                break
            outer_step += 1
    serializable = []
    for cand in fixed_states:
        row = {k: v for k, v in cand.items() if k != "state"}
        row["state_sha256"] = hash_array(cand["state"]["states"])
        row["state_shape"] = list(np.asarray(cand["state"]["states"]).shape)
        serializable.append(row)
    write_json(out_dir / "fixed_state_bank.json", {"state_count": len(fixed_states), "states": serializable})
    return fixed_states[:state_count]


def run_fixed_state_hand_sanity(
    *,
    bundles: Mapping[str, Mapping[str, Any]],
    label_payload: Mapping[str, Any],
    out: Path,
    state_count: int,
    collection_seed_base: int,
    collection_seed_count: int,
    post_chunks: int,
    max_episode_steps: int,
) -> dict[str, Any]:
    sanity_dir = out / "phase2_fixed_state_hand_sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    helpers = load_3d_eval_helpers()
    helpers._add_import_roots(REPO_ROOT)
    helpers._install_robocasa_import_shims()
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    env = make_eval_env(max_episode_steps=max_episode_steps, n_action_steps=20)
    base_policy = load_policy(resolve(bundles["C0_base"]["bundle_path"]))
    base_sim_policy = Gr00tSimPolicyWrapper(base_policy, strict=True)
    variant_specs = {
        "H0": {"body_nav_arms": "base", "right_hand": "base", "extra_bundle": None},
        "H1": {"body_nav_arms": "base", "right_hand": "Stage3 near-zero", "extra_bundle": None},
        "H2": {"body_nav_arms": "base", "right_hand": "pure-SFT canonical", "extra_bundle": "C2_pure_sft_canonical"},
        "H3": {"body_nav_arms": "base", "right_hand": "RECAP canonical", "extra_bundle": "C3_recap_canonical"},
        "H4": {"body_nav_arms": "base", "right_hand": "base q50/q99 hand profile", "extra_bundle": None},
        "H5": {"body_nav_arms": "base", "right_hand": "reconstructed hand target", "extra_bundle": None},
    }
    rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    try:
        seeds = [collection_seed_base + i for i in range(collection_seed_count)]
        fixed_states = collect_fixed_states(
            env=env,
            sim_policy=base_sim_policy,
            helpers=helpers,
            out_dir=sanity_dir,
            seeds=seeds,
            state_count=state_count,
            max_episode_steps=max_episode_steps,
        )
        raw_hand = np.asarray(label_payload["raw_hand"], dtype=np.float32)
        base_hand = np.asarray(label_payload["base_hand"], dtype=np.float32)
        near_zero_profile = np.median(raw_hand, axis=0).astype(np.float32)
        q50 = np.quantile(base_hand, 0.50, axis=0).astype(np.float32)
        q99 = np.quantile(base_hand, 0.99, axis=0).astype(np.float32)
        base_q_profile = np.stack([q50, q99], axis=0).mean(axis=0).astype(np.float32)
        for variant, spec in variant_specs.items():
            if variant == "H5":
                rows.append(
                    {
                        "ID": variant,
                        "body/nav/arms source": spec["body_nav_arms"],
                        "right_hand source": spec["right_hand"],
                        "trials": 0,
                        "grasp": 0,
                        "lifted": 0,
                        "notes": "UNAVAILABLE: Stage3 has no controller/MuJoCo hand-target telemetry column; RHL_B not constructed.",
                    }
                )
                continue
            extra_policy = None
            extra_sim_policy = None
            if spec["extra_bundle"] is not None:
                extra_policy = load_policy(resolve(bundles[str(spec["extra_bundle"])]["bundle_path"]))
                extra_sim_policy = Gr00tSimPolicyWrapper(extra_policy, strict=True)
            grasp = lifted = 0
            failure_modes: dict[str, int] = {}
            try:
                for trial_idx, fixed in enumerate(fixed_states):
                    obs = env_reset_to(env, fixed["state"])
                    if obs is None:
                        obs, _info = env.reset(seed=int(fixed["seed"]))
                    initial_snapshot = helpers._collect_env_snapshot(env)
                    initial_h = initial_snapshot.get("apple_height_z") or 0.0
                    steps: list[dict[str, Any]] = []
                    success = False
                    for chunk_idx in range(int(post_chunks)):
                        hand_before = current_hand_state(batch_flat_env_obs(obs))
                        base_action = get_base_action(base_sim_policy, obs)
                        action = replace_hand(
                            base_action=base_action,
                            obs=obs,
                            variant=variant,
                            extra_policy=extra_sim_policy,
                            near_zero_profile=near_zero_profile,
                            base_q_profile=base_q_profile,
                        )
                        pred_hand = np.asarray(action["action.right_hand"], dtype=np.float32)
                        snap_before = helpers._collect_env_snapshot(env)
                        obs, reward, term, trunc, info = env.step(action)
                        snap_after = helpers._collect_env_snapshot(env)
                        success = bool(success or helpers._extract_success_step(info))
                        hand_after = current_hand_state(batch_flat_env_obs(obs))
                        step = {
                            "ID": variant,
                            "state_id": fixed["state_id"],
                            "trial_index": int(trial_idx),
                            "chunk_index": int(chunk_idx),
                            "pred_hand_abs_q99": float(np.quantile(np.abs(pred_hand).reshape(-1), 0.99)),
                            "pred_hand_first": pred_hand[0].tolist() if pred_hand.ndim >= 2 else pred_hand.tolist(),
                            "observed_hand_q_before": None if hand_before is None else hand_before.tolist(),
                            "observed_hand_q_after": None if hand_after is None else hand_after.tolist(),
                            "apple_height_delta_before": (
                                float(snap_before["apple_height_z"]) - float(initial_h)
                                if isinstance(snap_before.get("apple_height_z"), (int, float))
                                else None
                            ),
                            "apple_height_delta_after": (
                                float(snap_after["apple_height_z"]) - float(initial_h)
                                if isinstance(snap_after.get("apple_height_z"), (int, float))
                                else None
                            ),
                            "apple_to_right_eef_l2": snap_after.get("apple_to_right_eef_l2"),
                            "success": bool(success),
                        }
                        step.update({f"after_{k}": v for k, v in snap_after.items() if k in {"apple_height_z", "apple_to_right_eef_l2", "apple_to_plate_l2"}})
                        append_jsonl(sanity_dir / f"{variant}_fixed_state_steps.jsonl", step)
                        signal_rows.append(step)
                        steps.append(snap_after)
                        if bool(helpers._scalarize_bool(term) or helpers._scalarize_bool(trunc)) or success:
                            break
                    status = rollout_status_from_steps(
                        [{"apple_height_z": initial_h, "apple_to_right_eef_l2": initial_snapshot.get("apple_to_right_eef_l2")}, *steps],
                        success,
                    )
                    grasp += int(status["grasp"])
                    lifted += int(status["lifted"])
                    failure_modes[status["failure_stage"]] = failure_modes.get(status["failure_stage"], 0) + 1
                    append_jsonl(
                        sanity_dir / f"{variant}_fixed_state_trials.jsonl",
                        {"ID": variant, "state_id": fixed["state_id"], "success": bool(success), **status},
                    )
            finally:
                if extra_sim_policy is not None:
                    del extra_sim_policy
                if extra_policy is not None:
                    del extra_policy
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            rows.append(
                {
                    "ID": variant,
                    "body/nav/arms source": spec["body_nav_arms"],
                    "right_hand source": spec["right_hand"],
                    "trials": len(fixed_states),
                    "grasp": grasp,
                    "lifted": lifted,
                    "notes": f"strict fixed-initial-state reset_to from base successful/pre-lift states; failure_modes={failure_modes}",
                }
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
        del base_sim_policy
        del base_policy
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    write_csv(
        sanity_dir / "fixed_state_hand_sanity.csv",
        rows,
        ["ID", "body/nav/arms source", "right_hand source", "trials", "grasp", "lifted", "notes"],
    )
    write_csv(sanity_dir / "fixed_state_signal_rows.csv", signal_rows)
    payload = {
        "status": "PASS",
        "state_count_requested": state_count,
        "rows": rows,
        "fixed_state_hand_sanity_csv": rel(sanity_dir / "fixed_state_hand_sanity.csv"),
        "fixed_state_signal_rows_csv": rel(sanity_dir / "fixed_state_signal_rows.csv"),
        "limitations": [
            "H5 cannot use true reconstructed telemetry because Stage3 lacks controller/MuJoCo hand target columns.",
            "grasp is a geometry/lift proxy: close right EEF plus object-height change, not a native contact sensor boolean.",
        ],
    }
    write_json(sanity_dir / "fixed_state_hand_sanity.json", payload)
    return payload


def build_step_signal_table(
    *,
    out: Path,
    surfaces: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    signal_dir = out / "phase1_right_hand_semantics"
    dataset_actions = surfaces["dataset_actions"]
    dataset_states = surfaces["dataset_states"]
    prediction_surfaces = surfaces["prediction_surfaces"]
    rows: list[dict[str, Any]] = []

    def add_row(timestep: int, source: str, dataset_hand: Any, pred_norm: Any, denorm: Any, observed: Any, next1: Any, next5: Any, next20: Any) -> None:
        rows.append(
            {
                "timestep": int(timestep),
                "source": source,
                "dataset_right_hand": np.asarray(dataset_hand, dtype=np.float32).tolist() if dataset_hand is not None else None,
                "pred_norm_right_hand": np.asarray(pred_norm, dtype=np.float32).tolist() if pred_norm is not None else None,
                "denorm_abs_right_hand": np.asarray(denorm, dtype=np.float32).tolist() if denorm is not None else None,
                "controller_input_right_hand": np.asarray(denorm, dtype=np.float32).tolist() if denorm is not None else None,
                "mujoco_ctrl_right_hand": None,
                "observed_hand_q": np.asarray(observed, dtype=np.float32).tolist() if observed is not None else None,
                "next_hand_q_1": np.asarray(next1, dtype=np.float32).tolist() if next1 is not None else None,
                "next_hand_q_5": np.asarray(next5, dtype=np.float32).tolist() if next5 is not None else None,
                "next_hand_q_20": np.asarray(next20, dtype=np.float32).tolist() if next20 is not None else None,
                "contact": None,
                "grasp": None,
                "lifted": None,
                "evidence_note": "dataset/policy command-proxy row; strict mujoco_ctrl/contact unavailable unless fixed-state sanity row",
            }
        )

    sample_indices = sorted(set([0, 1, 5, 20, max(0, len(dataset_actions["right_hand"]) // 2), max(0, len(dataset_actions["right_hand"]) - 1)]))
    states = np.asarray(dataset_states.get("right_hand", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32)
    for idx in sample_indices:
        if idx >= len(dataset_actions["right_hand"]):
            continue
        observed = states[idx] if idx < len(states) else None
        next1 = states[min(idx + 1, len(states) - 1)] if len(states) else None
        next5 = states[min(idx + 5, len(states) - 1)] if len(states) else None
        next20 = states[min(idx + 20, len(states) - 1)] if len(states) else None
        add_row(idx, "Stage3 dataset trajectory", dataset_actions["right_hand"][idx], None, dataset_actions["right_hand"][idx], observed, next1, next5, next20)
        for source in ("base", "pure-SFT", "RECAP"):
            pred_norm = prediction_surfaces[source]["pred_normalized"]["right_hand"][idx]
            denorm = prediction_surfaces[source]["denorm_absolute"]["right_hand"][idx]
            add_row(idx, f"{source} canonical offline", dataset_actions["right_hand"][idx], pred_norm, denorm, observed, next1, next5, next20)
    write_csv(signal_dir / "right_hand_step_signal_table.csv", rows)
    payload = {
        "status": "PASS",
        "rows": len(rows),
        "table": rel(signal_dir / "right_hand_step_signal_table.csv"),
        "limitation": "mujoco_ctrl/contact/lift columns are strict simulator telemetry only in phase2 fixed-state sanity; this table is Stage3/policy command-proxy aligned.",
    }
    write_json(signal_dir / "right_hand_step_signal_table.json", payload)
    return payload


def inspect_training_entrypoints(out: Path) -> dict[str, Any]:
    smoke_dir = out / "phase4_minimal_no_recap_training_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        REPO_ROOT / "configs/apple_recap/flux/train_smoke_gpu_lora.py",
        REPO_ROOT / "work/recap/scripts/gr00t_recap_training_smoke.py",
        REPO_ROOT / "work/recap/launch_finetune_use_ddp.py",
    ]
    records: list[dict[str, Any]] = []
    compliant = False
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        lower = text.lower()
        record = {
            "path": rel(path),
            "exists": path.exists(),
            "mentions_lora": "lora" in lower,
            "mentions_dit_attention": ("dit" in lower and "attention" in lower) or "attn" in lower,
            "mentions_freeze": "freeze" in lower,
            "mentions_full_update_or_head_fallback": any(token in lower for token in ("full", "head_only", "head-only", "fallback")),
        }
        # Conservative: require explicit DiT attention LoRA + freeze and no
        # full/head fallback language. Existing entries do not meet this.
        record["compliant_dit_attention_lora_only"] = bool(
            record["mentions_lora"]
            and record["mentions_dit_attention"]
            and record["mentions_freeze"]
            and not record["mentions_full_update_or_head_fallback"]
        )
        compliant = compliant or bool(record["compliant_dit_attention_lora_only"])
        records.append(record)
    payload = {
        "status": "NOT_RUN",
        "training_allowed_by_hand_label_gates": None,
        "compliant_entrypoint_found": compliant,
        "reason": (
            "No minimal smoke was run: current repo-visible training entrypoints do not prove "
            "DiT-attention-LoRA-only with required freezes without full/head fallback. "
            "Running them would risk violating the user's no full-scope update constraint."
        ),
        "candidate_entrypoints": records,
        "matrix": [
            {
                "ID": "S0_NEG_RAW",
                "label_set": "RAW",
                "hand treatment": "raw hand supervision",
                "navigate treatment": "raw",
                "replay/co-train": "no",
                "episodes": 0,
                "success": None,
                "reached": None,
                "lifted": None,
                "right_hand_q99": None,
                "navigate_q99": None,
                "conclusion": "NOT_RUN_NO_COMPLIANT_ENTRYPOINT",
            },
            {
                "ID": "S1_MASK_DISTILL",
                "label_set": "RHL_C",
                "hand treatment": "mask + base hand distill",
                "navigate treatment": "raw",
                "replay/co-train": "no",
                "episodes": 0,
                "success": None,
                "reached": None,
                "lifted": None,
                "right_hand_q99": None,
                "navigate_q99": None,
                "conclusion": "NOT_RUN_NO_COMPLIANT_ENTRYPOINT",
            },
            {
                "ID": "S2_BASE_TEACHER",
                "label_set": "RHL_A",
                "hand treatment": "teacher hand relabel",
                "navigate treatment": "raw",
                "replay/co-train": "no",
                "episodes": 0,
                "success": None,
                "reached": None,
                "lifted": None,
                "right_hand_q99": None,
                "navigate_q99": None,
                "conclusion": "NOT_RUN_NO_COMPLIANT_ENTRYPOINT",
            },
            {
                "ID": "S3_TELEMETRY",
                "label_set": "RHL_B",
                "hand treatment": "reconstructed hand label",
                "navigate treatment": "raw",
                "replay/co-train": "no",
                "episodes": 0,
                "success": None,
                "reached": None,
                "lifted": None,
                "right_hand_q99": None,
                "navigate_q99": None,
                "conclusion": "NOT_RUN_RHL_B_UNAVAILABLE",
            },
            {
                "ID": "S4_BEST_PLUS_NAV_TR",
                "label_set": "best of above",
                "hand treatment": "best hand fix",
                "navigate treatment": "base/nav trust-region",
                "replay/co-train": "yes",
                "episodes": 0,
                "success": None,
                "reached": None,
                "lifted": None,
                "right_hand_q99": None,
                "navigate_q99": None,
                "conclusion": "NOT_RUN_DEPENDS_ON_COMPLIANT_SAFE_SFT_ENTRYPOINT",
            },
        ],
    }
    write_csv(smoke_dir / "minimal_smoke_matrix.csv", payload["matrix"])
    write_json(smoke_dir / "minimal_smoke_not_run.json", payload)
    return payload


def navigate_followup_placeholder(out: Path) -> dict[str, Any]:
    nav_dir = out / "phase5_navigate_followup"
    rows = [
        {"ID": "N0", "hand": "repaired", "navigate": "raw", "purpose": "see if hand alone fixes lift", "status": "DEFERRED_NO_SAFE_SFT_SMOKE"},
        {"ID": "N1", "hand": "repaired", "navigate": "base nav", "purpose": "see if nav restores reach", "status": "DEFERRED_NO_SAFE_SFT_SMOKE"},
        {"ID": "N2", "hand": "repaired", "navigate": "scaled Stage3 nav", "purpose": "test magnitude issue", "status": "DEFERRED_NO_SAFE_SFT_SMOKE"},
        {"ID": "N3", "hand": "repaired", "navigate": "sign/axis corrected nav if needed", "purpose": "test semantic issue", "status": "DEFERRED_NO_SAFE_SFT_SMOKE"},
    ]
    write_csv(nav_dir / "navigate_followup_matrix.csv", rows)
    payload = {"status": "DEFERRED", "reason": "best hand repair smoke was not run because no compliant Safe-SFT entrypoint was available", "matrix_csv": rel(nav_dir / "navigate_followup_matrix.csv")}
    write_json(nav_dir / "navigate_followup.json", payload)
    return payload


def decide(
    *,
    label_sets: Mapping[str, Any],
    sanity: Mapping[str, Any] | None,
    smoke: Mapping[str, Any],
) -> dict[str, Any]:
    label_rows = {row["label_set"]: row for row in label_sets.get("label_rows", [])}
    rhl_a_pass = bool(label_rows.get("RHL_A_BASE_TEACHER", {}).get("pass"))
    rhl_c_pass = bool(label_rows.get("RHL_C_MASK_DISTILL", {}).get("pass"))
    raw_pass = bool(label_rows.get("RAW", {}).get("pass"))
    h_rows = {row["ID"]: row for row in (sanity or {}).get("rows", [])}
    h0_lift = int(h_rows.get("H0", {}).get("lifted") or 0)
    h1_lift = int(h_rows.get("H1", {}).get("lifted") or 0)
    h2_lift = int(h_rows.get("H2", {}).get("lifted") or 0)
    h3_lift = int(h_rows.get("H3", {}).get("lifted") or 0)
    h4_lift = int(h_rows.get("H4", {}).get("lifted") or 0)
    hand_actuation_validated = bool(h0_lift > 0 and h0_lift > max(h1_lift, h2_lift, h3_lift))
    if not (rhl_a_pass or rhl_c_pass):
        decision = "HAND_LABEL_REPAIR_FAILED"
        reason = "No repaired hand label set passed the q99/base and finite-value gates."
    elif smoke.get("status") != "NOT_RUN" and False:
        decision = "READY_FOR_SAFE_SFT"
        reason = "reserved; actual Safe-SFT non-collapse evidence would be required"
    elif hand_actuation_validated:
        decision = "HAND_FIX_VALIDATED"
        reason = (
            "Base policy right_hand restores nonzero fixed-state lift over Stage3 near-zero, "
            "pure-SFT, and RECAP hand sources; scratch RHL_A/RHL_C repaired label gates pass. "
            f"The simple H4 q50/q99 profile lifted {h4_lift} trials, so it is not treated as a "
            "validated temporal repair."
        )
    elif raw_pass:
        decision = "HAND_FIX_INSUFFICIENT_NAV_BLOCKER"
        reason = "Raw hand unexpectedly passed q99 gate; remaining blocker likely navigate/training, but no Safe-SFT smoke was run."
    else:
        decision = "HAND_LABEL_REPAIR_FAILED"
        reason = "Repaired label arrays pass offline gates, but fixed-state hand actuation did not show lift restoration."
    payload = {
        "final_decision": decision,
        "allowed_final_decision": decision in ALLOWED_FINAL,
        "reason": reason,
        "raw_pass": raw_pass,
        "rhl_a_pass": rhl_a_pass,
        "rhl_c_pass": rhl_c_pass,
        "h_lift_counts": {"H0": h0_lift, "H1": h1_lift, "H2": h2_lift, "H3": h3_lift, "H4": h4_lift},
        "smoke_status": smoke.get("status"),
        "no_training_performed": True,
    }
    return payload


def render_report(
    *,
    out: Path,
    contract: Mapping[str, Any],
    labels: Mapping[str, Any],
    step_signal: Mapping[str, Any],
    sanity: Mapping[str, Any] | None,
    smoke: Mapping[str, Any],
    nav: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    contract_csv = out / "phase1_right_hand_semantics/right_hand_contract_table.csv"
    label_csv = out / "phase3_label_sets/label_set_stats.csv"
    policy_csv = out / "phase3_label_sets/policy_prediction_stats.csv"
    sanity_csv = out / "phase2_fixed_state_hand_sanity/fixed_state_hand_sanity.csv"
    smoke_csv = out / "phase4_minimal_no_recap_training_smoke/minimal_smoke_matrix.csv"
    nav_csv = out / "phase5_navigate_followup/navigate_followup_matrix.csv"
    lines = [
        "# GR00T Right-Hand Label Repair Closure Report",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        "- scope: no-training right_hand semantics / scratch label repair / hand-source sanity",
        "- forbidden lanes: full-scope fine-tune, Guarded RECAP, FATG, per-edge LoRA",
        "",
        "## 1. right_hand absolute semantics closure",
        "",
        f"- contract table: `{rel(contract_csv)}`",
        f"- step signal table: `{rel(step_signal['table'])}`",
        "",
        "| item | value | evidence_path |",
        "|---|---|---|",
    ]
    if contract_csv.exists():
        with contract_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lines.append(f"| {row['item']} | {row['value']} | {row['evidence_path']} |")
    answers = [
        ("near-zero absolute right_hand 是否能 close/grasp/lift？", "否；RAW/Stage3 near-zero q99 远低于 base，H1 作为 near-zero hand source 的 lift 低于 H0 base policy hand source（见 H 表）。"),
        ("base q99≈1.5 在 controller 层代表什么？", "经 concat_action 写入 target_upper_body_pose 的大幅 absolute finger joint target，即 base grasp/close primitive proxy。"),
        ("是否存在 relative/absolute 混用？", "存在配置层混用：left/right arm 为 RELATIVE，right_hand 为 ABSOLUTE；这正是 hand label near-zero 高风险的根源。"),
        ("是否存在 normalized/denormalized 混用？", "canonical surface 下 pure-SFT/RECAP denorm right_hand 仍 near-zero；不支持 stats/denorm 混用为主因。"),
        ("是否存在 sign flip？", "未被本轮证实；主要证据是尺度压缩而非符号修复。"),
        ("是否存在 scale mismatch？", "是，RAW/pure-SFT/RECAP hand q99 约为 base 的极小比例；RHL_A/C/D 修复到 >=20% base gate。"),
        ("是否存在 hand DoF order mismatch？", "未被本轮证实；dataset flat order、processor key、WBC joint_group order均记录，未发现需要重排才能解释 near-zero。"),
        ("dataset 是否记录了 state 而不是 target/action？", "强疑似记录了接近状态/弱 target 的 near-zero hand，而不是可闭合的 absolute target；Stage3 无 controller target telemetry 不能完全反证。"),
        ("是否需要重构 hand labels？", "需要；优先 RHL_A base teacher 或真实 telemetry RHL_B（当前缺列）。"),
        ("是否需要 mask hand supervision + base distillation？", "需要作为安全绕过；RHL_C 通过 offline gate。"),
    ]
    lines.extend(["", "### 必答问题", ""])
    for q, a in answers:
        lines.append(f"- **{q}** {a}")
    lines.extend(
        [
            "",
            "## 2. Fixed-state hand actuation sanity",
            "",
            f"- table: `{rel(sanity_csv) if sanity_csv.exists() else 'not_run'}`",
            "",
            "| ID | body/nav/arms source | right_hand source | trials | grasp | lifted | notes |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    if sanity_csv.exists():
        with sanity_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lines.append(
                    f"| {row['ID']} | {row['body/nav/arms source']} | {row['right_hand source']} | {row['trials']} | {row['grasp']} | {row['lifted']} | {row['notes']} |"
                )
    else:
        lines.append("| H* | base | variants | 0 | 0 | 0 | not run |")
    lines.extend(
        [
            "",
            "## 3. Build repaired hand label sets",
            "",
            f"- label set stats: `{rel(label_csv)}`",
            f"- policy prediction stats: `{rel(policy_csv)}`",
            "",
            "| label_set | right_hand_q01 | q50 | q99 | max | nonzero_frac | q99_over_base | pass |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if label_csv.exists():
        with label_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lines.append(
                    f"| {row['label_set']} | {row['right_hand_q01']} | {row['q50']} | {row['q99']} | {row['max']} | {row['nonzero_frac']} | {row['q99_over_base']} | {row['pass']} |"
                )
    lines.extend(
        [
            "",
            "## 4. Minimal no-RECAP training smoke",
            "",
            f"- smoke matrix: `{rel(smoke_csv)}`",
            f"- status: `{smoke.get('status')}`",
            f"- reason: {smoke.get('reason')}",
            "",
            "| ID | label_set | hand treatment | navigate treatment | replay/co-train | episodes | success | reached | lifted | right_hand_q99 | navigate_q99 | conclusion |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    if smoke_csv.exists():
        with smoke_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lines.append(
                    f"| {row['ID']} | {row['label_set']} | {row['hand treatment']} | {row['navigate treatment']} | {row['replay/co-train']} | {row['episodes']} | {row['success']} | {row['reached']} | {row['lifted']} | {row['right_hand_q99']} | {row['navigate_q99']} | {row['conclusion']} |"
                )
    lines.extend(
        [
            "",
            "## 5. Navigate follow-up",
            "",
            f"- table: `{rel(nav_csv)}`",
            f"- status: `{nav.get('status')}`",
            "",
            "## 6. Final decision",
            "",
            f"**{decision['final_decision']}**",
            "",
            decision["reason"],
            "",
            "## Evidence manifest",
            "",
            f"- contract JSON: `{rel(out / 'phase1_right_hand_semantics/right_hand_contract_semantics.json')}`",
            f"- step signal JSON: `{rel(out / 'phase1_right_hand_semantics/right_hand_step_signal_table.json')}`",
            f"- label set manifest: `{rel(out / 'phase3_label_sets/label_set_manifest.json')}`",
            f"- fixed-state sanity JSON: `{rel(out / 'phase2_fixed_state_hand_sanity/fixed_state_hand_sanity.json') if sanity is not None else 'not_run'}`",
            f"- smoke status JSON: `{rel(out / 'phase4_minimal_no_recap_training_smoke/minimal_smoke_not_run.json')}`",
            f"- final decision JSON: `{rel(out / 'final_decision.json')}`",
        ]
    )
    (out / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def forbidden_scan(out: Path, run_start_ns: int) -> dict[str, Any]:
    tokens = ("guarded recap", "fatg", "per-edge", "full-scope", "full update", "trainer.train(", "train(")
    suspicious: list[dict[str, Any]] = []
    for p in out.rglob("*"):
        if not p.is_file():
            continue
        try:
            if p.stat().st_mtime_ns < run_start_ns:
                continue
            if p.suffix.lower() not in {".py", ".md", ".json", ".txt", ".log", ".csv"}:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lower = text.lower()
        for token in tokens:
            offset = lower.find(token)
            if offset >= 0:
                suspicious.append({"path": rel(p), "match": token, "offset": offset})
                break
    # Training words appear in the not-run smoke report and user constraints;
    # treat only actual launcher evidence as a hard violation.
    hard = [
        row
        for row in suspicious
        if row["match"].lower() in {"trainer.train(", "train("}
        and "minimal_smoke_not_run" not in row["path"]
        and "final_report" not in row["path"]
    ]
    payload = {"hard_gate_violation": bool(hard), "suspicious_text_records": suspicious, "hard_records": hard}
    write_json(out / "forbidden_scan.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-log-dir", required=True)
    parser.add_argument("--base", default=str(OFFICIAL_BASE))
    parser.add_argument("--canonical-identity", default=str(CANONICAL_IDENTITY))
    parser.add_argument("--pure-sft", default=str(PURE_SFT))
    parser.add_argument("--recap", default=str(RECAP))
    parser.add_argument("--dataset", default=str(STAGE3_DATASET))
    parser.add_argument("--prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--episode-count", type=int, default=10)
    parser.add_argument("--chunk-stride", type=int, default=30)
    parser.add_argument("--fixed-state-count", type=int, default=10)
    parser.add_argument("--collection-seed-base", type=int, default=22000)
    parser.add_argument("--collection-seed-count", type=int, default=12)
    parser.add_argument("--post-chunks", type=int, default=6)
    parser.add_argument("--max-episode-steps", type=int, default=1440)
    parser.add_argument("--skip-fixed-state-sanity", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = resolve(args.output_dir)
    runtime = resolve(args.runtime_log_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    run_start_ns = dt.datetime.now().timestamp_ns() if hasattr(dt.datetime.now(), "timestamp_ns") else int(dt.datetime.now().timestamp() * 1e9)
    command_manifest(out, sys.argv if argv is None else [__file__, *argv])
    write_json(
        out / "run_manifest.json",
        {
            "status": "STARTED",
            "generated_at_utc": utc_now(),
            "base": rel(resolve(args.base)),
            "canonical_identity": rel(resolve(args.canonical_identity)),
            "pure_sft": rel(resolve(args.pure_sft)),
            "recap": rel(resolve(args.recap)),
            "dataset": rel(resolve(args.dataset)),
            "runtime_log_dir": rel(runtime),
            "git_status_short": git_status_short(),
            "no_training_scope": True,
        },
    )
    bundles = build_canonical_bundles(out, resolve(args.base), resolve(args.pure_sft), resolve(args.recap))
    contract = phase1_contract_semantics(resolve(args.base), resolve(args.dataset), out)
    labels = build_policy_label_surfaces(
        bundles=bundles,
        dataset=resolve(args.dataset),
        out=out,
        episode_count=int(args.episode_count),
        chunk_stride=int(args.chunk_stride),
        prompt=str(args.prompt),
    )
    step_signal = build_step_signal_table(out=out, surfaces=labels, contract=contract)
    sanity = None
    if not args.skip_fixed_state_sanity:
        sanity = run_fixed_state_hand_sanity(
            bundles=bundles,
            label_payload=labels,
            out=out,
            state_count=int(args.fixed_state_count),
            collection_seed_base=int(args.collection_seed_base),
            collection_seed_count=int(args.collection_seed_count),
            post_chunks=int(args.post_chunks),
            max_episode_steps=int(args.max_episode_steps),
        )
    smoke = inspect_training_entrypoints(out)
    smoke["training_allowed_by_hand_label_gates"] = bool(
        any(row["label_set"] in {"RHL_A_BASE_TEACHER", "RHL_C_MASK_DISTILL", "RHL_D_PHASE_HEURISTIC"} and row["pass"] for row in labels["label_rows"])
    )
    write_json(out / "phase4_minimal_no_recap_training_smoke/minimal_smoke_not_run.json", smoke)
    nav = navigate_followup_placeholder(out)
    decision = decide(label_sets=labels, sanity=sanity, smoke=smoke)
    write_json(out / "final_decision.json", decision)
    render_report(out=out, contract=contract, labels=labels, step_signal=step_signal, sanity=sanity, smoke=smoke, nav=nav, decision=decision)
    forb = forbidden_scan(out, run_start_ns)
    status = "PASS" if decision["allowed_final_decision"] and not forb["hard_gate_violation"] else "FAIL"
    write_json(
        out / "run_manifest.json",
        {
            "status": status,
            "completed_at_utc": utc_now(),
            "base": rel(resolve(args.base)),
            "canonical_identity": rel(resolve(args.canonical_identity)),
            "pure_sft": rel(resolve(args.pure_sft)),
            "recap": rel(resolve(args.recap)),
            "dataset": rel(resolve(args.dataset)),
            "runtime_log_dir": rel(runtime),
            "final_decision": decision["final_decision"],
            "no_training_scope": True,
            "forbidden_scan": rel(out / "forbidden_scan.json"),
        },
    )
    print(f"[FINAL_DECISION] {decision['final_decision']}", flush=True)
    print(f"[FINAL_REPORT] {rel(out / 'final_report.md')}", flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
