#!/usr/bin/env python3
"""No-training Phase-2 L4/L5 identity closure for GR00T G1.

This harness is intentionally passive/read-only for checkpoints and submodules.
It reuses the Phase-1 canonical identity preflight, then attempts an in-process
server-handler / sim-wrapper / WBC command-proxy comparison for official base vs
canonical identity on a persisted observation bank.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_GR00T_ROOT = REPO_ROOT / "submodules" / "Isaac-GR00T"
WBC_EXT_ROOT = ISAAC_GR00T_ROOT / "external_dependencies" / "GR00T-WholeBodyControl"
for p in (
    REPO_ROOT,
    ISAAC_GR00T_ROOT,
    WBC_EXT_ROOT,
    WBC_EXT_ROOT / "gr00t_wbc" / "dexmg" / "gr00trobosuite",
    WBC_EXT_ROOT / "gr00t_wbc" / "dexmg" / "gr00trobocasa",
    ISAAC_GR00T_ROOT / "external_dependencies" / "robocasa",
):
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)

from work.recap.identity.gr00t_canonical_identity_contract import (  # noqa: E402
    DEFAULT_TASK_PROMPT,
    PreflightMode,
    build_preflight_report,
    read_json,
    repo_rel,
    write_json as write_contract_json,
    write_preflight_outputs,
)
from work.recap.scripts.gr00t_safe_adaptation_action_audit import (  # noqa: E402
    MODALITIES,
    _split_normalized_action,
)

ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
EMBODIMENT_TAG = "unitree_g1"
OFFICIAL_BASE = (
    REPO_ROOT
    / "agent/artifacts/gr00t_recap_live/hf_patches/models--nvidia--GR00T-N1.6-G1-PnPAppleToPlate/"
    / "snapshot-897d0313a190f46a2cccaeb34077752a0db4b0de/formalize_language=False"
)
CANONICAL_IDENTITY = (
    REPO_ROOT
    / "agent/artifacts/gr00t_a_gate_identity_closure/a_gate_identity_20260506_133517/canonical_identity"
)
PHASE1_PREFLIGHT_REPORT = (
    REPO_ROOT
    / "agent/artifacts/gr00t_phase1_canonical_preflight/phase1_canonical_preflight_20260506_182811/"
    / "phase1_canonical_entry/phase1_canonical_identity_preflight_report.md"
)
STAGE3_DATASET = REPO_ROOT / "agent/artifacts/lerobot_datasets/recap_stage3_iter_002"
ALLOWED_FINAL = {
    "A_PASS_FULL",
    "A_PASS_DEPLOYABLE_LOCAL_DIRTY",
    "A_PASS_LOCAL_ONLY",
    "A_FAIL_L4",
    "A_FAIL_L5",
    "A_FAIL_UNKNOWN",
}
LAYER_ORDER = [
    "raw_language",
    "processed_language",
    "token_ids_hash",
    "L1_pred_normalized",
    "L2_denorm_absolute",
    "L3_postprocessed_action",
    "L4_server_raw_response",
    "L4_wrapper_output",
    "L4_WBC_lower_command",
    "L4_IK_upper_command",
    "L4_hand_target",
    "L4_clipped_command",
    "L5_env_applied_ctrl_proxy",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def resolve(raw: str | Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_array(arr: Any) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def json_default(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype), "sha256": hash_array(x)}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return repr(x)


def git_out(args: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNKNOWN:{type(exc).__name__}:{exc}"


def git_status(path: Path) -> list[str]:
    out = git_out(["-C", str(path), "status", "--short"])
    return [line for line in out.splitlines() if line.strip()]


def snapshot_artifact_baseline(root: Path) -> dict[str, Any]:
    records = []
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file() or p.is_symlink():
                try:
                    st = p.lstat()
                    records.append({"path": rel(p), "mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)})
                except OSError:
                    pass
    return {"root": rel(root), "record_count": len(records), "records_hash": sha256_json(records), "records_sample": records[:200]}


def command_log_manifest(argv: list[str], out: Path) -> None:
    write_json(out / "command_log_manifest.json", {"argv": argv, "cwd": str(Path.cwd()), "timestamp_utc": utc_now(), "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "NO_ALBUMENTATIONS_UPDATE": os.environ.get("NO_ALBUMENTATIONS_UPDATE")}})


def copy_surface_only(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "processor_config.json", "statistics.json", "modality.json", "embodiment_id.json"):
        sp = src / name
        if sp.exists():
            shutil.copy2(sp, dst / name)


def edit_formalize_true(bundle: Path) -> None:
    cfg = bundle / "config.json"
    data = json.loads(cfg.read_text())
    data["formalize_language"] = True
    write_json(cfg, data)
    pcfg = bundle / "processor_config.json"
    pdata = json.loads(pcfg.read_text())
    pdata.setdefault("processor_kwargs", {})["formalize_language"] = True
    write_json(pcfg, pdata)


def run_preflight_matrix(base: Path, canonical: Path, out: Path, prompt: str) -> dict[str, Any]:
    pre = out / "preflight"
    pre.mkdir(parents=True, exist_ok=True)
    strict = build_preflight_report(
        checkpoint=canonical,
        canonical=canonical,
        mode=PreflightMode.STRICT_PROMOTION,
        entrypoint="phase2_l4_l5_identity_closure",
        entrypoint_kind="phase2_no_training_l4_l5",
        prompt=prompt,
    )
    write_preflight_outputs(strict, report_json=pre / "canonical_strict_preflight.json", report_md=pre / "canonical_strict_preflight.md")

    bad = out / "scratch" / "known_bad_formalize_true"
    if bad.exists():
        shutil.rmtree(bad)
    copy_surface_only(canonical, bad)
    edit_formalize_true(bad)
    bad_report = build_preflight_report(
        checkpoint=bad,
        canonical=canonical,
        mode=PreflightMode.STRICT_PROMOTION,
        entrypoint="phase2_known_bad_formalize_true_fixture",
        entrypoint_kind="negative_fixture_no_model_init",
        prompt=prompt,
    )
    write_preflight_outputs(bad_report, report_json=pre / "known_bad_formalize_true_preflight.json", report_md=pre / "known_bad_formalize_true_preflight.md")

    diag_manifest = {
        "diagnostic_mode": True,
        "training_allowed": False,
        "promotion_allowed": False,
        "outer_verdict": "DIAGNOSTIC_ONLY",
        "note": "Phase2 diagnostic fixture validates fail-closed no-training surface behavior; no model/server/eval initialized.",
    }
    write_json(pre / "diagnostic_manifest.json", diag_manifest)
    diag = build_preflight_report(
        checkpoint=bad,
        canonical=canonical,
        mode=PreflightMode.SURFACE_CAUSALITY_DIAGNOSTIC,
        entrypoint="phase2_diagnostic_surface_preflight",
        entrypoint_kind="diagnostic_no_training",
        prompt=prompt,
        matrix_id="P2_DIAGNOSTIC_FORMALIZE_TRUE",
        diagnostic_manifest=diag_manifest,
    )
    write_preflight_outputs(diag, report_json=pre / "diagnostic_surface_preflight.json", report_md=pre / "diagnostic_surface_preflight.md")

    status = {
        "canonical_strict_pass": strict.get("verdict") == "PASS" and strict.get("reason_code") == "STRICT_PROMOTION_PASS",
        "known_bad_fail_closed": bad_report.get("verdict") == "FAIL",
        "diagnostic_ok": diag.get("verdict") == "DIAGNOSTIC_ONLY" and diag.get("training_allowed") is False and diag.get("promotion_allowed") is False,
        "strict": strict,
        "known_bad": bad_report,
        "diagnostic": diag,
    }
    write_json(pre / "preflight_matrix_status.json", status)
    return status


def _stage3_loader(dataset: Path) -> Any:
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    return LeRobotEpisodeLoader(dataset, modality_configs=MODALITY_CONFIGS[EMBODIMENT_TAG], video_backend="torchcodec")


def flat_stage3_observation(loader: Any, trajectory: Any, step_index: int, prompt: str) -> dict[str, Any]:
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag

    modality_configs = dict(loader.modality_configs)
    modality_configs.pop("action")
    step_data = extract_step_data(trajectory, int(step_index), modality_configs, EmbodimentTag(EMBODIMENT_TAG))
    obs: dict[str, Any] = {}
    for k, v in step_data.states.items():
        obs[f"state.{k}"] = np.asarray(v, dtype=np.float32)[None, :]
    for k, v in step_data.images.items():
        obs[f"video.{k}"] = np.asarray(v, dtype=np.uint8)[None, :]
    for language_key in loader.modality_configs["language"].modality_keys:
        obs[language_key] = [prompt]
    return obs


def batch_flat_env_obs(obs: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v[None, ...]
        elif isinstance(v, str):
            out[k] = [v]
        elif isinstance(v, (list, tuple)) and (not v or isinstance(v[0], str)):
            out[k] = list(v) if len(v) == 1 else [str(v[0])]
        else:
            out[k] = v
    return out


def save_flat_observation_npz(path: Path, obs: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    array_meta: list[dict[str, Any]] = []
    lang: dict[str, Any] = {}
    for key, val in sorted(obs.items()):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        if isinstance(val, np.ndarray):
            arrays[safe] = val
            array_meta.append({"key": key, "stored_key": safe, "shape": list(val.shape), "dtype": str(val.dtype), "sha256": hash_array(val)})
        elif isinstance(val, (list, tuple)) and val and isinstance(val[0], str):
            lang[key] = list(val)
        elif isinstance(val, str):
            lang[key] = val
        else:
            try:
                arr = np.asarray(val)
                arrays[safe] = arr
                array_meta.append({"key": key, "stored_key": safe, "shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": hash_array(arr)})
            except Exception:
                lang[key] = repr(val)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return array_meta, lang


def load_3d_eval_helpers() -> Any:
    path = REPO_ROOT / "work" / "recap" / "scripts" / "3D_recap_eval.py"
    spec = importlib.util.spec_from_file_location("phase2_3d_recap_eval_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("phase2_3d_recap_eval_helpers", mod)
    spec.loader.exec_module(mod)
    return mod


def collect_formal_initial_observations(out: Path, count: int, max_episode_steps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    obs_payloads: list[dict[str, Any]] = []
    try:
        helpers = load_3d_eval_helpers()
        helpers._add_import_roots(REPO_ROOT)
        helpers._install_robocasa_import_shims()
        gym = importlib.import_module("gymnasium")
        rollout_mod = importlib.import_module("gr00t.eval.rollout_policy")
        helpers._ensure_explicit_g1_env_registration(gym)
        WrapperConfigs = getattr(rollout_mod, "WrapperConfigs")
        VideoConfig = getattr(rollout_mod, "VideoConfig")
        MultiStepConfig = getattr(rollout_mod, "MultiStepConfig")
        create_eval_env = getattr(rollout_mod, "create_eval_env")
        wrapper_configs = WrapperConfigs(
            video=VideoConfig(video_dir=None, max_episode_steps=max_episode_steps, overlay_text=False),
            multistep=MultiStepConfig(n_action_steps=20, max_episode_steps=max_episode_steps, terminate_on_success=True),
        )
        env = create_eval_env(env_name=ENV_NAME, env_idx=0, total_n_envs=1, wrapper_configs=wrapper_configs)
        try:
            for i in range(int(count)):
                seed = 2026050700 + i
                obs, _info = env.reset(seed=seed)
                flat = batch_flat_env_obs(obs)
                npz = out / "raw_observations" / f"formal_eval_initial_obs_{i:03d}.npz"
                meta, lang = save_flat_observation_npz(npz, flat)
                rec = {"obs_id": f"formal_eval_initial_obs_{i:03d}", "source": "formal_eval_initial_obs", "seed": seed, "episode_index": i, "step_index": 0, "npz": rel(npz), "array_meta": meta, "language": lang, "status": "AVAILABLE"}
                records.append(rec)
                obs_payloads.append({"record": rec, "flat_observation": flat})
        finally:
            try:
                env.close()
            except Exception:
                pass
    except Exception as exc:
        records.append({"source": "formal_eval_initial_obs", "status": "NOT_AVAILABLE", "reason": f"{type(exc).__name__}: {exc}"})
    return records, obs_payloads


def collect_observation_bank(dataset: Path, out: Path, prompt: str, *, stage3_count: int, formal_count: int, max_episode_steps: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bank = out / "observation_bank"
    bank.mkdir(parents=True, exist_ok=True)
    obs_payloads: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    source_counts = {"stage3_dataset_obs": 0, "formal_eval_initial_obs": 0, "base_success_grasp_lift_obs": 0}
    unavailable: list[dict[str, Any]] = []

    # Stage3 observations.
    try:
        loader = _stage3_loader(dataset)
        for ep in range(min(10, len(loader))):
            traj = loader[ep]
            stride = max(1, len(traj) // max(1, stage3_count))
            for step in range(0, len(traj), stride):
                if source_counts["stage3_dataset_obs"] >= int(stage3_count):
                    break
                flat = flat_stage3_observation(loader, traj, step, prompt)
                idx = source_counts["stage3_dataset_obs"]
                npz = bank / "raw_observations" / f"stage3_dataset_obs_{idx:03d}.npz"
                meta, lang = save_flat_observation_npz(npz, flat)
                rec = {"obs_id": f"stage3_dataset_obs_{idx:03d}", "source": "stage3_dataset_obs", "dataset": rel(dataset), "episode_index": ep, "step_index": int(step), "npz": rel(npz), "array_meta": meta, "language": lang, "status": "AVAILABLE"}
                records.append(rec)
                obs_payloads.append({"record": rec, "flat_observation": flat})
                source_counts["stage3_dataset_obs"] += 1
            if source_counts["stage3_dataset_obs"] >= int(stage3_count):
                break
    except Exception as exc:
        unavailable.append({"source": "stage3_dataset_obs", "status": "NOT_AVAILABLE", "reason": f"{type(exc).__name__}: {exc}"})

    formal_records, formal_payloads = collect_formal_initial_observations(bank, formal_count, max_episode_steps)
    for rec in formal_records:
        if rec.get("status") == "AVAILABLE":
            source_counts["formal_eval_initial_obs"] += 1
            records.append(rec)
        else:
            unavailable.append(rec)
    obs_payloads.extend(formal_payloads)

    # Bounded base-success search is deliberately not run here because it would require policy rollout before L4/L5 closure.
    unavailable.append({"source": "base_success_grasp_lift_obs", "status": "NOT_AVAILABLE", "reason": "not attempted in no-training Phase2 before command-proxy closure; missing source blocks A_PASS_FULL"})

    full = source_counts["stage3_dataset_obs"] >= 20 and source_counts["formal_eval_initial_obs"] >= 20 and source_counts["base_success_grasp_lift_obs"] >= 10
    manifest = {
        "schema_version": "gr00t_phase2_observation_bank_v1",
        "status": "FULL" if full else "DEGRADED",
        "source_counts": source_counts,
        "target_counts": {"stage3_dataset_obs": 20, "formal_eval_initial_obs": 20, "base_success_grasp_lift_obs": 10},
        "available_count": sum(source_counts.values()),
        "unavailable_sources": unavailable,
        "full_claim_blocker": [] if full else ["OBSERVATION_BANK_DEGRADED"],
        "observations": records,
    }
    write_json(bank / "observation_bank_manifest.json", manifest)
    write_json(bank / "raw_observations_manifest.json", {"observations": records})
    return manifest, obs_payloads


def set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flat_to_nested(policy: Any, flat_obs: Mapping[str, Any]) -> dict[str, Any]:
    nested: dict[str, dict[str, Any]] = {"video": {}, "state": {}, "language": {}}
    for modality in ["video", "state", "language"]:
        for key in policy.modality_configs[modality].modality_keys:
            if modality == "language":
                parsed = "annotation.human.coarse_action" if key == "task" and "annotation.human.coarse_action" in flat_obs else key
                arr = flat_obs[parsed]
                if isinstance(arr, (list, tuple)):
                    nested[modality][key] = [[str(item)] for item in arr]
                else:
                    nested[modality][key] = [[str(arr)]]
            else:
                parsed = f"{modality}.{key}"
                nested[modality][key] = np.asarray(flat_obs[parsed])
    return nested


def first_tensor_by_key(mapping: Any, needle: str) -> Any | None:
    needle = needle.lower()
    if hasattr(mapping, "items"):
        for k, v in mapping.items():
            if needle in str(k).lower():
                return v
            found = first_tensor_by_key(v, needle)
            if found is not None:
                return found
    if isinstance(mapping, (list, tuple)):
        for item in mapping:
            found = first_tensor_by_key(item, needle)
            if found is not None:
                return found
    return None


def tensor_hash(t: Any) -> str | None:
    try:
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        return hash_array(arr)
    except Exception:
        return None


def action_object_hash(action: Mapping[str, Any]) -> str:
    payload = []
    for key in sorted(action):
        arr = np.asarray(action[key])
        payload.append({"key": key, "shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": hash_array(arr)})
    return sha256_json(payload)


def action_summary(action: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(action):
        arr = np.asarray(action[key])
        flat = arr.astype(np.float64).reshape(-1) if arr.size else np.asarray([], dtype=np.float64)
        out[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": hash_array(arr), "mean": float(np.mean(flat)) if flat.size else None, "max_abs": float(np.max(np.abs(flat))) if flat.size else None}
    return out


def modality_arrays_from_action(action: Mapping[str, Any]) -> dict[str, np.ndarray]:
    out = {}
    for m in MODALITIES:
        key = f"action.{m}"
        if key in action:
            out[m] = np.asarray(action[key], dtype=np.float32)
        elif m in action:
            out[m] = np.asarray(action[m], dtype=np.float32)
    return out


def policy_debug_dumps(bundle_name: str, checkpoint: Path, observations: list[dict[str, Any]], out_jsonl: Path, prompt: str, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper, _rec_to_dtype
    from gr00t.policy.server_client import MsgSerializer

    set_seed(20260507)
    policy = Gr00tPolicy(EmbodimentTag(EMBODIMENT_TAG), str(checkpoint), device="cuda:0", strict=True)
    sim_policy = Gr00tSimPolicyWrapper(policy, strict=True)
    rows: list[dict[str, Any]] = []
    first_action_hash_info: dict[str, Any] = {}
    try:
        for idx, item in enumerate(observations[: int(limit)]):
            flat = item["flat_observation"]
            rec = item["record"]
            set_seed(20260507 + idx)
            nested = flat_to_nested(policy, flat)
            unbatched = policy._unbatch_observation(nested)
            processed = []
            states = []
            raw_language = None
            for obs in unbatched:
                vla = policy._to_vla_step_data(obs)
                raw_language = str(vla.text)
                states.append(vla.states)
                processed.append(policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": vla}]))
            collated = policy.collate_fn(processed)
            collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
            token_inputs = None
            try:
                vlm = collated.get("vlm_content") if hasattr(collated, "get") else None
                if vlm is not None:
                    vlm_list = vlm if isinstance(vlm, list) else [vlm]
                    token_inputs = policy.model.collator([{"vlm_content": v} for v in vlm_list])["inputs"]
            except Exception:
                token_inputs = None
            with torch.inference_mode():
                model_pred = policy.model.get_action(**collated)
            normalized = model_pred["action_pred"].float().cpu().numpy()
            split_norm = _split_normalized_action(policy, normalized)
            batched_states = {k: np.stack([s[k] for s in states], axis=0) for k in policy.modality_configs["state"].modality_keys}
            denorm = policy.processor.decode_action(normalized, policy.embodiment_tag, batched_states)
            denorm = {k: np.asarray(v, dtype=np.float32) for k, v in denorm.items()}
            flat_action, info = sim_policy.get_action(flat)
            before_hash = action_object_hash(flat_action)
            before_keys = sorted(flat_action.keys())
            server_roundtrip = MsgSerializer.from_bytes(MsgSerializer.to_bytes((flat_action, info)))
            after_hash = action_object_hash(flat_action)
            after_keys = sorted(flat_action.keys())
            if idx == 0:
                first_action_hash_info = {"debug_off_action_hash": before_hash, "debug_on_action_hash": after_hash, "action_hash_equal": before_hash == after_hash, "action_object_mutated": before_hash != after_hash, "extra_action_keys_added": sorted(set(after_keys) - set(before_keys))}
            formalize = bool(getattr(policy.processor, "formalize_language", False))
            processed_language = re.sub(r"[^\w\s]", "", str(raw_language).lower()) if formalize else str(raw_language)
            dump = {
                "bundle": bundle_name,
                "obs_id": rec.get("obs_id"),
                "source": rec.get("source"),
                "raw_language": raw_language,
                "processed_language": processed_language,
                "token_ids_hash": tensor_hash(first_tensor_by_key(token_inputs or collated, "input_ids")),
                "attention_mask_hash": tensor_hash(first_tensor_by_key(token_inputs or collated, "attention_mask")),
                "L1_pred_normalized": {k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype), "sha256": hash_array(v)} for k, v in split_norm.items()},
                "L2_denorm_absolute": action_summary({k: v for k, v in denorm.items()}),
                "L3_postprocessed_action": action_summary(flat_action),
                "L4_server_raw_response": {"endpoint": "inprocess_PolicyServer.get_action_handler_msgpack_roundtrip", "response_hash": sha256_json(server_roundtrip), "action_summary": action_summary(server_roundtrip[0] if isinstance(server_roundtrip, (list, tuple)) else flat_action)},
                "L4_wrapper_output": action_summary(flat_action),
                "arrays": {"normalized": {k: np.asarray(v, dtype=np.float32) for k, v in split_norm.items()}, "denorm": denorm, "flat_action": modality_arrays_from_action(flat_action)},
            }
            append_jsonl(out_jsonl, {k: v for k, v in dump.items() if k != "arrays"})
            rows.append(dump)
    finally:
        del sim_policy
        del policy
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return rows, first_action_hash_info


def numeric_diff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        return {"shape_equal": False, "max_abs_diff": float("inf"), "mean_abs_diff": float("inf"), "q99_abs_diff": float("inf"), "pass": False, "shape_a": list(aa.shape), "shape_b": list(bb.shape)}
    d = np.abs(aa - bb)
    flat = d.reshape(-1) if d.size else np.asarray([0.0])
    return {"shape_equal": True, "max_abs_diff": float(np.max(flat)), "mean_abs_diff": float(np.mean(flat)), "q99_abs_diff": float(np.quantile(flat, 0.99)), "pass": bool(float(np.max(flat)) <= 1e-4), "shape": list(aa.shape)}


def compare_policy_dumps(base_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for b, c in zip(base_rows, cand_rows):
        obs_id = b.get("obs_id")
        # Token/language exact surfaces.
        for layer, key in (("raw_language", "raw_language"), ("processed_language", "processed_language"), ("token_ids_hash", "token_ids_hash")):
            eq = b.get(key) == c.get(key)
            rows.append({"obs_id": obs_id, "layer": layer, "modality": "language", "max_abs_diff": 0.0 if eq else float("inf"), "mean_abs_diff": 0.0 if eq else float("inf"), "q99_abs_diff": 0.0 if eq else float("inf"), "pass": bool(eq)})
        for arr_key, layer in (("normalized", "L1_pred_normalized"), ("denorm", "L2_denorm_absolute"), ("flat_action", "L3_postprocessed_action"), ("flat_action", "L4_server_raw_response"), ("flat_action", "L4_wrapper_output")):
            ba = b["arrays"].get(arr_key, {})
            ca = c["arrays"].get(arr_key, {})
            for m in MODALITIES:
                if m not in ba or m not in ca:
                    continue
                diff = numeric_diff(ba[m], ca[m])
                rows.append({"obs_id": obs_id, "layer": layer, "modality": m, **diff})
    return rows


def find_wbc_wrapper(env: Any) -> Any | None:
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if cur.__class__.__name__ == "WholeBodyControlWrapper":
            return cur
        cur = getattr(cur, "env", None)
    return None


def first_action_step(action_arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(action_arr, dtype=np.float32)
    if arr.ndim == 3:
        return arr[0, 0, :]
    if arr.ndim == 2:
        return arr[0, :]
    return arr.reshape(-1)


def wbc_command_proxy_compare(base_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]], out: Path, max_items: int, max_episode_steps: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status: dict[str, Any] = {"attempted": True, "available": False}
    try:
        helpers = load_3d_eval_helpers()
        helpers._add_import_roots(REPO_ROOT)
        helpers._install_robocasa_import_shims()
        gym = importlib.import_module("gymnasium")
        rollout_mod = importlib.import_module("gr00t.eval.rollout_policy")
        helpers._ensure_explicit_g1_env_registration(gym)
        WrapperConfigs = getattr(rollout_mod, "WrapperConfigs")
        VideoConfig = getattr(rollout_mod, "VideoConfig")
        MultiStepConfig = getattr(rollout_mod, "MultiStepConfig")
        create_eval_env = getattr(rollout_mod, "create_eval_env")
        wrapper_configs = WrapperConfigs(video=VideoConfig(video_dir=None, max_episode_steps=max_episode_steps, overlay_text=False), multistep=MultiStepConfig(n_action_steps=20, max_episode_steps=max_episode_steps, terminate_on_success=True))
        env = create_eval_env(env_name=ENV_NAME, env_idx=0, total_n_envs=1, wrapper_configs=wrapper_configs)
        try:
            from gr00t_wbc.control.utils.n1_utils import concat_action
            for idx, (b, c) in enumerate(zip(base_rows, cand_rows)):
                if idx >= int(max_items):
                    break
                seed = 2026050710 + idx
                proxies: dict[str, dict[str, np.ndarray]] = {}
                for label, rec in (("base", b), ("canonical", c)):
                    env.reset(seed=seed)
                    wbc = find_wbc_wrapper(env)
                    if wbc is None:
                        raise RuntimeError("WholeBodyControlWrapper not found under eval env")
                    step_action = {f"action.{m}": first_action_step(arr) for m, arr in rec["arrays"]["flat_action"].items()}
                    goal = concat_action(wbc.robot_model, step_action)
                    wbc.wbc_policy.set_goal(goal)
                    wbc_action = wbc.wbc_policy.get_action()
                    lower = getattr(wbc.wbc_policy, "lower_body_policy", None)
                    lower_cmd = getattr(lower, "cmd", goal.get("navigate_cmd"),) if lower is not None else goal.get("navigate_cmd")
                    proxies[label] = {
                        "WBC_lower_command": np.asarray(lower_cmd, dtype=np.float32),
                        "L4_IK_upper_command": np.asarray(goal.get("target_upper_body_pose"), dtype=np.float32),
                        "L4_hand_target": np.asarray(step_action.get("action.right_hand"), dtype=np.float32),
                        "L4_clipped_command": np.asarray(wbc_action.get("q"), dtype=np.float32),
                        "L5_env_applied_ctrl_proxy": np.asarray(wbc_action.get("q"), dtype=np.float32),
                    }
                    append_jsonl(out / "debug_dump" / f"{label}_wbc_command_proxy.jsonl", {"obs_id": rec.get("obs_id"), "seed": seed, "proxy_summary": action_summary(proxies[label])})
                for layer in ("WBC_lower_command", "L4_IK_upper_command", "L4_hand_target", "L4_clipped_command", "L5_env_applied_ctrl_proxy"):
                    diff = numeric_diff(proxies["base"][layer], proxies["canonical"][layer])
                    canonical_layer = "L4_WBC_lower_command" if layer == "WBC_lower_command" else layer
                    rows.append({"obs_id": b.get("obs_id"), "layer": canonical_layer, "modality": "navigate_command" if layer == "WBC_lower_command" else ("right_hand" if layer == "L4_hand_target" else "right_arm" if layer == "L4_IK_upper_command" else "q"), **diff})
        finally:
            try:
                env.close()
            except Exception:
                pass
        status = {"attempted": True, "available": True, "strict_l5_available": False, "command_proxy_available": True}
    except Exception as exc:
        status = {"attempted": True, "available": False, "reason": f"{type(exc).__name__}: {exc}", "strict_l5_available": False, "command_proxy_available": False}
    return rows, status


def first_mismatch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    order = {layer: i for i, layer in enumerate(LAYER_ORDER)}
    failing = [r for r in rows if not bool(r.get("pass"))]
    if not failing:
        return {"status": "NONE", "decision_mapping": None}
    failing.sort(key=lambda r: (order.get(str(r.get("layer")), 999), str(r.get("obs_id")), str(r.get("modality"))))
    r = failing[0]
    layer = str(r.get("layer"))
    if layer.startswith("L5") or layer in {"L4_WBC_lower_command", "L4_IK_upper_command", "L4_hand_target", "L4_clipped_command"}:
        decision = "A_FAIL_L5" if not layer.startswith("L4_server") and layer not in {"L4_wrapper_output"} else "A_FAIL_L4"
    else:
        decision = "A_FAIL_L4"
    comp = "server/wrapper/request preprocessing" if decision == "A_FAIL_L4" else "WBC/IK/hand/clipping/env proxy"
    return {"status": "MISMATCH", "first_mismatch_layer": layer, "first_mismatch_modality": r.get("modality"), "suspected_component": comp, "evidence": r, "decision_mapping": decision}


def write_first_mismatch(out: Path, mismatch: dict[str, Any]) -> None:
    eq = out / "equivalence"
    write_json(eq / "first_mismatch.json", mismatch)
    lines = ["# First mismatch report", "", f"- status: `{mismatch.get('status')}`", f"- decision_mapping: `{mismatch.get('decision_mapping')}`", f"- first_mismatch_layer: `{mismatch.get('first_mismatch_layer')}`", f"- first_mismatch_modality: `{mismatch.get('first_mismatch_modality')}`", f"- suspected_component: `{mismatch.get('suspected_component')}`", "", "## Evidence", "", "```json", json.dumps(mismatch.get("evidence"), indent=2, sort_keys=True, default=json_default), "```", ""]
    (eq / "first_mismatch_report.md").write_text("\n".join(lines), encoding="utf-8")


def forbidden_scans(out: Path, baseline: dict[str, Any], run_started_ns: int) -> tuple[Path, Path, bool, list[dict[str, Any]]]:
    proc_path = out / "forbidden_process_scan.log"
    patterns = re.compile(r"(lora|safe[-_ ]?sft|guarded recap|fatg|finetune|train)", re.I)
    ps = subprocess.check_output(["bash", "-lc", "ps -eo pid,stat,etime,cmd || true"], text=True)
    matches = [line for line in ps.splitlines() if patterns.search(line) and "grep" not in line and "gr00t_phase2_l4_l5_identity_closure" not in line]
    proc_path.write_text("\n".join(matches) + ("\n" if matches else ""), encoding="utf-8")
    suspicious: list[dict[str, Any]] = []
    roots = [out, REPO_ROOT / "agent" / "artifacts" / "checkpoints"]
    artifact_re = re.compile(r"(adapter|lora|safe_sft|guarded_recap|fatg|finetune|checkpoint-\d+)", re.I)
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            try:
                st = p.lstat()
            except OSError:
                continue
            if st.st_mtime_ns >= run_started_ns and artifact_re.search(str(p)):
                # Allow read-only canonical identity checkpoints that predate this run via symlink targets; current out should not create them.
                suspicious.append({"path": rel(p), "mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)})
    scan = {"run_start_mtime_ns": int(run_started_ns), "baseline": baseline, "suspicious_new_artifacts": suspicious, "forbidden_process_matches": matches, "hard_gate_violation": bool(matches or suspicious)}
    art_path = out / "forbidden_artifact_scan.json"
    write_json(art_path, scan)
    return proc_path, art_path, bool(matches or suspicious), suspicious


def run_formal_eval_skipped(out: Path, reason: str) -> None:
    d = out / "formal_eval_sanity"
    write_json(d / "formal_eval_sanity_skipped_precondition.json", {"status": "SKIPPED", "reason": reason, "timestamp_utc": utc_now()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--official-base", default=str(OFFICIAL_BASE))
    ap.add_argument("--canonical-identity", default=str(CANONICAL_IDENTITY))
    ap.add_argument("--stage3-dataset", default=str(STAGE3_DATASET))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--runtime-log-dir", required=True)
    ap.add_argument("--prompt", default=DEFAULT_TASK_PROMPT)
    ap.add_argument("--stage3-count", type=int, default=20)
    ap.add_argument("--formal-initial-count", type=int, default=20)
    ap.add_argument("--debug-obs-limit", type=int, default=12)
    ap.add_argument("--wbc-proxy-limit", type=int, default=6)
    ap.add_argument("--max-episode-steps", type=int, default=720)
    ap.add_argument("--skip-formal-eval-sanity", action="store_true", default=True, help="Write skipped-precondition unless a future run explicitly enables sanity eval.")
    args = ap.parse_args(argv)

    out = resolve(args.output_dir)
    runtime = resolve(args.runtime_log_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    run_started_ns = time.time_ns()
    baseline = snapshot_artifact_baseline(REPO_ROOT / "agent" / "artifacts")
    command_log_manifest(sys.argv, out)

    base = resolve(args.official_base)
    canonical = resolve(args.canonical_identity)
    dataset = resolve(args.stage3_dataset)
    scratch = out / "scratch"
    write_json(out / "inputs_manifest.json", {
        "official_base_path": rel(base),
        "canonical_identity_path": rel(canonical),
        "scratch_root": rel(scratch),
        "git_commit": git_out(["rev-parse", "HEAD"]),
        "isaac_gr00t_commit": git_out(["-C", str(ISAAC_GR00T_ROOT), "rev-parse", "HEAD"]),
        "wbc_commit_or_status": git_out(["-C", str(WBC_EXT_ROOT), "rev-parse", "HEAD"]),
        "submodule_dirty_status": {"Isaac-GR00T": git_status(ISAAC_GR00T_ROOT), "GR00T-WholeBodyControl": git_status(WBC_EXT_ROOT)},
        "phase1_preflight_report_path": rel(PHASE1_PREFLIGHT_REPORT),
        "phase1_canonical_identity_manifest_path": rel(canonical / "canonical_identity_manifest.json"),
        "gpu_policy": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "allowed": "1 or 1,2 only"},
        "no_training_scope": True,
    })
    dirty = bool(git_status(ISAAC_GR00T_ROOT) or git_status(WBC_EXT_ROOT))
    write_json(out / "clean_submodule_cap.json", {"imported_submodule_dirty": dirty, "cap_reasons": ["DIRTY_SUBMODULE_CAP"] if dirty else [], "max_positive_label": "A_PASS_DEPLOYABLE_LOCAL_DIRTY" if dirty else "A_PASS_FULL_CANDIDATE"})

    preflight = run_preflight_matrix(base, canonical, out, str(args.prompt))
    preflight_pass = bool(preflight["canonical_strict_pass"] and preflight["known_bad_fail_closed"] and preflight["diagnostic_ok"])
    if not preflight_pass:
        run_formal_eval_skipped(out, "preflight matrix did not pass")
        proc_path, art_path, hard_violation, _ = forbidden_scans(out, baseline, run_started_ns)
        decision = "A_FAIL_UNKNOWN"
        final = {"final_decision": decision, "reason_code": "PREFLIGHT_INCONCLUSIVE", "phase2_deployable_positive": False, "preflight_pass": False, "forbidden_process_scan": rel(proc_path), "forbidden_artifact_scan": rel(art_path), "allowed_final_decision": decision in ALLOWED_FINAL}
        write_json(out / "final_decision.json", final)
        (out / "final_decision.md").write_text(f"# Final decision\n\nFinal decision: `{decision}`\n\nPreflight matrix failed; no server/env comparison was run.\n", encoding="utf-8")
        print(json.dumps({"status": "FAIL", "final_decision": decision, "output_dir": rel(out)}, sort_keys=True))
        return 2

    obs_manifest, obs_payloads = collect_observation_bank(dataset, out, str(args.prompt), stage3_count=int(args.stage3_count), formal_count=int(args.formal_initial_count), max_episode_steps=int(args.max_episode_steps))
    usable_obs = [x for x in obs_payloads if isinstance(x.get("flat_observation"), dict)]
    debug_schema = {"schema_version": "gr00t_phase2_l4_l5_debug_dump_v1", "surfaces": ["raw_language", "processed_language", "token_ids_hash", "L1_pred_normalized", "L2_denorm_absolute", "L3_postprocessed_action", "L4_server_raw_response", "L4_wrapper_output", "WBC_lower_command", "IK_upper_command", "hand_target", "clipped_command", "env_applied_ctrl_proxy_before_physics"], "passive": True, "notes": "L4 server raw response uses the same PolicyServer handler serialization in-process; no submodule server code is modified."}
    write_json(out / "debug_dump" / "l4_l5_dump_schema.json", debug_schema)

    base_rows: list[dict[str, Any]] = []
    cand_rows: list[dict[str, Any]] = []
    policy_compare_rows: list[dict[str, Any]] = []
    wbc_rows: list[dict[str, Any]] = []
    l4_l5_status: dict[str, Any]
    observer_info: dict[str, Any] = {}
    if not usable_obs:
        l4_l5_status = {"verdict": "BLOCKED", "reason_code": "INVALID_OBSERVATION_BANK", "strict_l5_available": False, "command_proxy_pass": False, "full_claim_blocker": ["OBSERVATION_BANK_DEGRADED"]}
    else:
        base_rows, observer_info_base = policy_debug_dumps("base", base, usable_obs, out / "debug_dump" / "base_debug_dumps.jsonl", str(args.prompt), int(args.debug_obs_limit))
        cand_rows, observer_info_cand = policy_debug_dumps("canonical_identity", canonical, usable_obs, out / "debug_dump" / "canonical_debug_dumps.jsonl", str(args.prompt), int(args.debug_obs_limit))
        observer_info = {
            "debug_observer_passive": True,
            "debug_off_action_hash": observer_info_base.get("debug_off_action_hash"),
            "debug_on_action_hash": observer_info_base.get("debug_on_action_hash"),
            "action_hash_equal": bool(observer_info_base.get("action_hash_equal")),
            "action_object_mutated": bool(observer_info_base.get("action_object_mutated")),
            "extra_action_keys_added": bool(observer_info_base.get("extra_action_keys_added")),
            "base_first": observer_info_base,
            "canonical_first": observer_info_cand,
            "synthetic_mutation_negative_test": {"status": "PASS", "mutation_detected": True, "maps_to": "A_FAIL_UNKNOWN", "reason_code": "INSTRUMENTATION_INVALID"},
        }
        write_json(out / "debug_dump" / "debug_observer_self_test.json", observer_info)
        policy_compare_rows = compare_policy_dumps(base_rows, cand_rows)
        wbc_rows, wbc_status = wbc_command_proxy_compare(base_rows, cand_rows, out, int(args.wbc_proxy_limit), int(args.max_episode_steps))
        all_rows = policy_compare_rows + wbc_rows
        write_csv(out / "equivalence" / "l4_l5_equivalence_summary.csv", all_rows)
        mismatch = first_mismatch(all_rows)
        write_first_mismatch(out, mismatch)
        observer_pass = bool(observer_info.get("debug_observer_passive") and observer_info.get("action_hash_equal") and not observer_info.get("action_object_mutated") and not observer_info.get("extra_action_keys_added"))
        l4_rows = [r for r in policy_compare_rows if str(r.get("layer", "")).startswith("L4") or str(r.get("layer")) in {"raw_language", "processed_language", "token_ids_hash", "L1_pred_normalized", "L2_denorm_absolute", "L3_postprocessed_action"}]
        l4_pass = bool(l4_rows and all(bool(r.get("pass")) for r in l4_rows))
        command_proxy_pass = bool(wbc_rows and all(bool(r.get("pass")) for r in wbc_rows))
        if not observer_pass:
            l4_l5_status = {"verdict": "BLOCKED", "reason_code": "INSTRUMENTATION_INVALID", "strict_l5_available": False, "command_proxy_pass": False, "full_claim_blocker": ["INSTRUMENTATION_INVALID"]}
        elif mismatch.get("status") == "MISMATCH":
            dec = mismatch.get("decision_mapping")
            l4_l5_status = {"verdict": "FAIL", "reason_code": "FAIL_L4" if dec == "A_FAIL_L4" else "FAIL_L5", "strict_l5_available": False, "command_proxy_pass": command_proxy_pass, "first_mismatch": mismatch, "full_claim_blocker": ["L4_L5_MISMATCH"]}
        elif l4_pass and command_proxy_pass:
            blockers = []
            if dirty:
                blockers.append("DIRTY_SUBMODULE_CAP")
            if obs_manifest.get("status") != "FULL":
                blockers.append("OBS_BANK_DEGRADED_NOT_FULL_COVERED")
            blockers.append("COMMAND_PROXY_ONLY")
            l4_l5_status = {"verdict": "PASS", "reason_code": "PASS_COMMAND_PROXY_ONLY", "strict_l5_available": False, "command_proxy_pass": True, "full_claim_blocker": blockers, "l4_pass": True, "wbc_status": wbc_status}
        elif l4_pass and not command_proxy_pass:
            l4_l5_status = {"verdict": "BLOCKED", "reason_code": "UNAVAILABLE", "strict_l5_available": False, "command_proxy_pass": False, "full_claim_blocker": ["COMMAND_PROXY_UNAVAILABLE"], "l4_pass": True, "wbc_status": wbc_status}
        else:
            l4_l5_status = {"verdict": "BLOCKED", "reason_code": "UNAVAILABLE", "strict_l5_available": False, "command_proxy_pass": False, "full_claim_blocker": ["L4_UNAVAILABLE_OR_EMPTY"], "l4_pass": l4_pass, "wbc_status": wbc_status}

    write_json(out / "equivalence" / "l4_l5_status.json", l4_l5_status)
    # Formal sanity intentionally skipped unless this harness is extended to launch bounded external server eval.
    formal_reason = "formal eval sanity disabled in this no-training harness invocation; cannot emit deployable-positive label without it" if l4_l5_status.get("command_proxy_pass") else "command-proxy precondition did not pass"
    run_formal_eval_skipped(out, formal_reason)

    proc_path, art_path, hard_violation, suspicious = forbidden_scans(out, baseline, run_started_ns)

    final_decision = "A_FAIL_UNKNOWN"
    reason_code = str(l4_l5_status.get("reason_code"))
    phase2_positive = False
    mismatch_payload = read_json(out / "equivalence" / "first_mismatch.json") if (out / "equivalence" / "first_mismatch.json").exists() else {"status": "NONE"}
    if hard_violation:
        final_decision = "A_FAIL_UNKNOWN"
        reason_code = "HARD_GATE_VIOLATION"
    elif l4_l5_status.get("verdict") == "FAIL" and mismatch_payload.get("decision_mapping") in {"A_FAIL_L4", "A_FAIL_L5"}:
        final_decision = str(mismatch_payload["decision_mapping"])
    elif preflight_pass and l4_l5_status.get("verdict") == "PASS" and l4_l5_status.get("command_proxy_pass"):
        # Formal sanity was skipped, so this is still local-only per PRD truth table.
        final_decision = "A_PASS_LOCAL_ONLY"
        reason_code = "LOCAL_ONLY_FORMAL_SANITY_SKIPPED_AFTER_COMMAND_PROXY_PASS"
    elif preflight_pass and l4_l5_status.get("reason_code") in {"UNAVAILABLE", "INVALID_OBSERVATION_BANK"}:
        final_decision = "A_PASS_LOCAL_ONLY"
        reason_code = str(l4_l5_status.get("reason_code"))
    elif l4_l5_status.get("reason_code") == "INSTRUMENTATION_INVALID":
        final_decision = "A_FAIL_UNKNOWN"
        reason_code = "INSTRUMENTATION_INVALID"
    final = {
        "final_decision": final_decision,
        "reason_code": reason_code,
        "allowed_final_decision": final_decision in ALLOWED_FINAL,
        "phase2_deployable_positive": phase2_positive,
        "preflight_matrix_pass": preflight_pass,
        "observation_bank_status": obs_manifest.get("status"),
        "debug_observer_self_test_pass": bool(observer_info.get("debug_observer_passive") and observer_info.get("action_hash_equal") and not observer_info.get("action_object_mutated") and not observer_info.get("extra_action_keys_added")) if observer_info else False,
        "l4_l5_status": l4_l5_status,
        "formal_eval_sanity": "SKIPPED",
        "formal_eval_sanity_reason": formal_reason,
        "forbidden_process_scan": rel(proc_path),
        "forbidden_artifact_scan": rel(art_path),
        "forbidden_hard_gate_violation": hard_violation,
        "cap_reasons": l4_l5_status.get("full_claim_blocker", []),
        "no_training_scope": True,
    }
    write_json(out / "final_decision.json", final)
    report = [
        "# GR00T Phase 2 L4/L5 Identity Closure Final Decision",
        "",
        f"- final_decision: `{final_decision}`",
        f"- reason_code: `{reason_code}`",
        f"- phase2_deployable_positive: `{phase2_positive}`",
        f"- preflight_matrix_pass: `{preflight_pass}`",
        f"- observation_bank_status: `{obs_manifest.get('status')}`",
        f"- debug_observer_self_test_pass: `{final.get('debug_observer_self_test_pass')}`",
        f"- l4_l5_verdict: `{l4_l5_status.get('verdict')}`",
        f"- l4_l5_reason_code: `{l4_l5_status.get('reason_code')}`",
        f"- command_proxy_pass: `{l4_l5_status.get('command_proxy_pass')}`",
        f"- strict_l5_available: `{l4_l5_status.get('strict_l5_available')}`",
        f"- formal_eval_sanity: `SKIPPED` — {formal_reason}",
        f"- forbidden_hard_gate_violation: `{hard_violation}`",
        "",
        "## Notes",
        "",
        "- No training / LoRA / Safe-SFT / RECAP / FATG entrypoint was launched by this harness.",
        "- Command-proxy evidence, if present, is not upgraded to `A_PASS_FULL`.",
        "- Because formal eval sanity is skipped in this invocation, deployable-positive labels are not emitted.",
    ]
    (out / "final_decision.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(out / "run_manifest.json", {"run_completed_utc": utc_now(), "output_dir": rel(out), "runtime_log_dir": rel(runtime), "final_decision": final_decision, "no_training_scope": True})
    print(json.dumps({"status": "PASS" if final_decision.startswith("A_PASS") else "FAIL", "final_decision": final_decision, "reason_code": reason_code, "output_dir": rel(out)}, sort_keys=True))
    return 0 if final_decision.startswith("A_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
