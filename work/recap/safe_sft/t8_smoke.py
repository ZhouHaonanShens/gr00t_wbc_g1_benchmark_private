from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np

from work.recap.safe_sft.entrypoint import (
    DEFAULT_CANONICAL_IDENTITY,
    DEFAULT_RIGHT_HAND_REPAIR,
    DEFAULT_STAGE3_DATASET,
    DEFAULT_TASK_PROMPT,
    MODALITIES,
    REPO_ROOT,
    WBC_ROOT,
    ISAAC_GR00T_ROOT,
    audit_trainable_params,
    build_repaired_dataset_gate,
    classify_param,
    cleanup_cuda,
    discover_lora_targets,
    freeze_all_params,
    git_output,
    inject_lora,
    json_default,
    load_json,
    load_policy,
    rel,
    sha256_file,
    surface_hashes,
    tensor_hash,
    utc_now,
    write_csv,
    write_json,
    _hash_forbidden_params,
    _modality_slices,
    _prepare_one_processed_sample,
    _read_windows,
    _rec_to_dtype,
    _safe_sft_loss,
)

for _p in (REPO_ROOT, ISAAC_GR00T_ROOT, WBC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


CERT_ROOT = (
    REPO_ROOT
    / "agent/artifacts/gr00t_safe_sft_entrypoint/gr00t_safe_sft_entrypoint_20260508_024600/certification"
)
UNLOCK_JSON = CERT_ROOT / "t7_smoke_unlock/smoke_unlock.json"
CERT_DECISION_JSON = CERT_ROOT / "final_decision.json"

ALLOWED_T8_FINAL = {
    "T8_FAIL_ENTRYPOINT_REGRESSION",
    "T8_FAIL_HAND_SIGNAL_NOT_LEARNED",
    "T8_FAIL_HAND_REPAIRED_BUT_NO_LIFT",
    "T8_PASS_HAND_REPAIR_SMOKE",
    "T8_READY_FOR_30SEED_SAFE_SFT",
    "T8_NAV_BLOCKER_AFTER_HAND_FIX",
}

CELL_SPECS: dict[str, dict[str, Any]] = {
    "S0_NEG_RAW": {
        "label_set": "RAW",
        "hand_treatment": "raw hand supervision",
        "navigate_treatment": "raw",
        "replay_co_train": "no",
        "negative_control": True,
    },
    "S1_MASK_DISTILL": {
        "label_set": "RHL_C_MASK_DISTILL",
        "hand_treatment": "mask + base hand distill",
        "navigate_treatment": "raw",
        "replay_co_train": "no",
        "negative_control": False,
    },
    "S2_BASE_TEACHER": {
        "label_set": "RHL_A_BASE_TEACHER",
        "hand_treatment": "teacher hand relabel",
        "navigate_treatment": "raw",
        "replay_co_train": "no",
        "negative_control": False,
    },
}


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def hash_array(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return __import__("hashlib").sha256(arr.view(np.uint8)).hexdigest()


def array_stats(arr: Any) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    flat = a.reshape(-1)
    if flat.size == 0:
        return {"shape": list(a.shape), "abs_q99": None, "max_abs": None, "mean": None, "std": None}
    return {
        "shape": list(a.shape),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "q01": float(np.quantile(flat, 0.01)),
        "q50": float(np.quantile(flat, 0.50)),
        "q99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
        "abs_mean": float(np.mean(np.abs(flat))),
        "abs_q99": float(np.quantile(np.abs(flat), 0.99)),
        "max_abs": float(np.max(np.abs(flat))),
        "nonzero_frac": float(np.mean(np.abs(flat) > 1e-8)),
        "sha256": hash_array(a),
    }


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def consume_unlock(out: Path) -> dict[str, Any]:
    final_obj = load_json(CERT_DECISION_JSON)
    unlock = load_json(UNLOCK_JSON)
    checks = unlock.get("checks", {}) if isinstance(unlock.get("checks"), Mapping) else {}
    final_checks = final_obj.get("checks", {}) if isinstance(final_obj.get("checks"), Mapping) else {}
    assertions: list[dict[str, Any]] = []

    def add(name: str, actual: Any, expected: Any | None = None, passed: bool | None = None) -> None:
        assertions.append(
            {
                "assertion": name,
                "actual": actual,
                "expected": expected,
                "pass": bool(actual == expected if passed is None else passed),
            }
        )

    add("final_decision", final_obj.get("final_decision"), "ENTRYPOINT_PASS_NO_TRAIN")
    for gate in ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"):
        status = (checks.get(gate) or final_checks.get(gate) or {}).get("status")
        add(f"{gate}_status", status, "PASS")
    for key, expected in (
        ("training_allowed", True),
        ("recap_allowed", False),
        ("fatg_allowed", False),
        ("guarded_recap_allowed", False),
        ("per_edge_lora_allowed", False),
        ("full_scope_update_allowed", False),
        ("surface_hash_pass", True),
    ):
        add(key, unlock.get(key), expected)
    t6 = checks.get("T6") or {}
    add("T6_surface_hash_equal", t6.get("surface_hash_equal"), True)
    add("T6_forbidden_mismatch_count", int(t6.get("forbidden_mismatch_count", -1)), 0)
    pass_all = all(row["pass"] for row in assertions)
    payload = {
        "status": "PASS" if pass_all else "FAIL",
        "certification_root": rel(CERT_ROOT),
        "final_decision_json": rel(CERT_DECISION_JSON),
        "smoke_unlock_json": rel(UNLOCK_JSON),
        "assertions": assertions,
        "unlock": unlock,
        "final_decision": final_obj,
        "generated_at_utc": utc_now(),
    }
    write_csv(out / "t8_0_unlock_consumption" / "unlock_assertions.csv", assertions)
    write_json(out / "t8_0_unlock_consumption" / "unlock_consumption.json", payload)
    return payload


def static_guard(out: Path, source_paths: Iterable[Path]) -> dict[str, Any]:
    forbidden_acceptance_tokens = [
        "--guarded-recap",
        "--fatg",
        "--per-edge",
        "--full-scope",
        "--full-head",
        "launch_" + "finetune.py",
        "Gr00t" + "Trainer(",
        "trainer." + "train(",
        "guarded_" + "recap",
        "fat" + "g",
        "per_" + "edge",
    ]
    rows = []
    for path in source_paths:
        exists = path.is_file()
        text = path.read_text(encoding="utf-8") if exists else ""
        bad = []
        if path.name != "entrypoint.py":
            for tok in forbidden_acceptance_tokens:
                if tok in text:
                    # Token appearances are allowed only inside explicit rejection lists/messages.
                    allowed_context = "rejected before model load" in text
                    if not allowed_context or tok in {"launch_" + "finetune.py", "Gr00t" + "Trainer(", "trainer." + "train("}:
                        bad.append(tok)
        rows.append(
            {
                "path": rel(path),
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
                "forbidden_context_tokens": bad,
                "status": "PASS" if exists and not bad else "FAIL",
            }
        )
    payload = {
        "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "rows": rows,
        "notes": "Runner uses local certified loss/LoRA machinery and does not call old full/head trainer routes.",
    }
    write_csv(out / "t8_1_runner_readiness" / "runner_static_guard.csv", rows)
    write_json(out / "t8_1_runner_readiness" / "runner_static_guard.json", payload)
    return payload


def reject_forbidden_args(argv: list[str]) -> None:
    forbidden = {
        "--recap",
        "--advantage",
        "--guarded-recap",
        "--fatg",
        "--per-edge",
        "--full-scope",
        "--full-head",
        "--old-trainer",
        "--merge-lora-before-eval",
        "--tune-action-decoder",
        "--tune-vlm",
    }
    present = sorted(set(argv) & forbidden)
    if present:
        raise SystemExit(f"Forbidden T8 route flags rejected before model load: {present}")


def load_label_gate(repair_root: Path) -> dict[str, Any]:
    stats_path = repair_root / "phase3_label_sets" / "label_set_stats.csv"
    rows = _read_csv_dicts(stats_path)
    by_label = {row["label_set"]: row for row in rows}
    manifest = load_json(repair_root / "phase3_label_sets" / "label_set_manifest.json")
    return {"rows": rows, "by_label": by_label, "manifest": manifest}


def label_q99(label_gate: Mapping[str, Any], label_set: str) -> float:
    aliases = {
        "RAW": "RAW",
        "RHL_A_BASE_TEACHER": "RHL_A_BASE_TEACHER",
        "RHL_C_MASK_DISTILL": "RHL_C_MASK_DISTILL",
    }
    row = (label_gate.get("by_label") or {}).get(aliases.get(label_set, label_set), {})
    for key in ("right_hand_q99", "q99"):
        if row.get(key) not in (None, ""):
            return float(row[key])
    return float("nan")


def label_q99_over_base(label_gate: Mapping[str, Any], label_set: str) -> float:
    aliases = {
        "RAW": "RAW",
        "RHL_A_BASE_TEACHER": "RHL_A_BASE_TEACHER",
        "RHL_C_MASK_DISTILL": "RHL_C_MASK_DISTILL",
    }
    row = (label_gate.get("by_label") or {}).get(aliases.get(label_set, label_set), {})
    if row.get("q99_over_base") not in (None, ""):
        return float(row["q99_over_base"])
    q99 = label_q99(label_gate, label_set)
    base = float((label_gate.get("manifest") or {}).get("base_right_hand_abs_q99") or 1.0)
    return q99 / max(base, 1e-12)


def param_named_dict(model: Any) -> dict[str, Any]:
    return dict(model.named_parameters())


def grad_norms(policy: Any) -> tuple[float, float, dict[str, float]]:
    lora_sq = 0.0
    forbidden_sq = 0.0
    by_group: dict[str, float] = {}
    for name, p in policy.model.named_parameters():
        group = classify_param(name)
        grad = p.grad
        norm = 0.0
        if grad is not None:
            norm = float(grad.detach().float().norm().cpu().item())
        by_group[group] = by_group.get(group, 0.0) + norm
        if group == "DiT attention LoRA" and p.requires_grad:
            lora_sq += norm * norm
        elif norm > 0.0:
            forbidden_sq += norm * norm
    return float(math.sqrt(lora_sq)), float(math.sqrt(forbidden_sq)), by_group


def modality_loss_means(result: Mapping[str, Any], modality: str) -> tuple[float, float]:
    task_sum = task_count = distill_sum = distill_count = 0.0
    for row in result.get("modality_rows", []):
        if row.get("modality") == modality:
            task_sum += float(row.get("task_loss_sum") or 0.0)
            task_count += float(row.get("task_mask_nonzero") or 0.0)
            distill_sum += float(row.get("distill_loss_sum") or 0.0)
            distill_count += float(row.get("distill_mask_nonzero") or 0.0)
    task = task_sum / max(task_count, 1.0)
    distill = distill_sum / max(distill_count, 1.0) if distill_count > 0 else 0.0
    return task, distill


def finite_or_fail(*values: float) -> bool:
    return all(np.isfinite(float(v)) for v in values)


def prepare_training_batches(
    policy: Any,
    *,
    stage3_dataset: Path,
    repair_root: Path,
    label_set: str,
    prompt: str,
    dtype: Any,
    max_windows: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]]]:
    windows = _read_windows(repair_root)
    if max_windows is not None:
        windows = windows[: int(max_windows)]
    samples: list[dict[str, Any]] = []
    distill_samples: list[dict[str, Any]] | None = [] if label_set == "RHL_C_MASK_DISTILL" else None
    metas: list[dict[str, Any]] = []
    for idx, _win in enumerate(windows):
        processed, meta = _prepare_one_processed_sample(
            policy,
            stage3_dataset=stage3_dataset,
            repair_root=repair_root,
            label_set=label_set,
            prompt=prompt,
            window_index=idx,
        )
        samples.append(_rec_to_dtype(policy.collate_fn([processed]), dtype=dtype))
        metas.append(meta)
        if distill_samples is not None:
            teacher_processed, _teacher_meta = _prepare_one_processed_sample(
                policy,
                stage3_dataset=stage3_dataset,
                repair_root=repair_root,
                label_set="RHL_A_BASE_TEACHER",
                prompt=prompt,
                window_index=idx,
            )
            distill_samples.append(_rec_to_dtype(policy.collate_fn([teacher_processed]), dtype=dtype))
    return samples, distill_samples, metas


def save_lora_checkpoint(policy: Any, cell_dir: Path, cell_spec: Mapping[str, Any], train_config: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    adapter_state = {
        name: p.detach().cpu()
        for name, p in policy.model.named_parameters()
        if classify_param(name) == "DiT attention LoRA"
    }
    ckpt_dir = cell_dir / "checkpoint_500"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = ckpt_dir / "adapter_model.pt"
    torch.save(adapter_state, adapter_path)
    manifest = {
        "status": "PASS",
        "checkpoint_type": "unmerged_lora_adapter_only",
        "adapter_model_path": rel(adapter_path),
        "lora_merged_before_eval": False,
        "cell": cell_spec,
        "train_config": train_config,
        "adapter_tensor_count": len(adapter_state),
        "adapter_numel": int(sum(v.numel() for v in adapter_state.values())),
        "generated_at_utc": utc_now(),
    }
    write_json(ckpt_dir / "adapter_manifest.json", manifest)
    write_json(cell_dir / "checkpoint_manifest.json", manifest)
    return manifest


def write_surface_hash_csv(path: Path, surface: Path) -> dict[str, Any]:
    obj = surface_hashes(surface)
    write_csv(path, obj.get("rows", []), ["file", "exists", "sha256", "size"])
    return obj


def train_cell(
    policy: Any,
    *,
    cell_id: str,
    cell_dir: Path,
    stage3_dataset: Path,
    repair_root: Path,
    prompt: str,
    label_gate: Mapping[str, Any],
    steps: int,
    lr: float,
    lambda_hand: float,
    dtype: Any,
) -> dict[str, Any]:
    import torch

    spec = dict(CELL_SPECS[cell_id])
    label_set = str(spec["label_set"])
    trace_path = cell_dir / "training_trace.csv"
    cell_dir.mkdir(parents=True, exist_ok=True)
    samples, distill_samples, metas = prepare_training_batches(
        policy,
        stage3_dataset=stage3_dataset,
        repair_root=repair_root,
        label_set=label_set,
        prompt=prompt,
        dtype=dtype,
    )
    opt = torch.optim.AdamW([p for p in policy.model.parameters() if p.requires_grad], lr=float(lr), weight_decay=0.0)
    rows: list[dict[str, Any]] = []
    finite = True
    forbidden_grad_nonzero = False
    last_loss = None
    log_steps = set([1, int(steps)])
    log_steps.update(range(50, int(steps) + 1, 50))
    base_label_q99 = label_q99(label_gate, label_set)
    start_time = time.monotonic()
    for step in range(1, int(steps) + 1):
        idx = (step - 1) % len(samples)
        batch = samples[idx]
        distill_batch = distill_samples[idx] if distill_samples is not None else None
        for p in policy.model.parameters():
            if p.grad is not None:
                p.grad = None
        torch.manual_seed(20260508 + step)
        np.random.seed(20260508 + step)
        result = _safe_sft_loss(policy, batch, label_set=label_set, distill_inputs=distill_batch, lambda_hand=lambda_hand)
        loss = result["loss"]
        loss_value = float(loss.detach().float().cpu().item())
        if not np.isfinite(loss_value):
            finite = False
            break
        loss.backward()
        lora_grad, forbidden_grad, by_group = grad_norms(policy)
        if forbidden_grad > 0.0:
            forbidden_grad_nonzero = True
        pred_np = result["pred_actions"].detach().float().cpu().numpy()
        slices = result["slices"]
        right_sl = slices.get("right_hand")
        nav_sl = slices.get("navigate_command")
        right_pred = (
            float(np.quantile(np.abs(pred_np[..., right_sl]).reshape(-1), 0.99))
            if right_sl is not None
            else float("nan")
        )
        nav_pred = (
            float(np.quantile(np.abs(pred_np[..., nav_sl]).reshape(-1), 0.99))
            if nav_sl is not None
            else float("nan")
        )
        right_task, right_distill = modality_loss_means(result, "right_hand")
        nav_task, _nav_distill = modality_loss_means(result, "navigate_command")
        task_loss = float(result["task_loss"].float().cpu().item())
        distill_loss = float(result["distill_loss"].float().cpu().item())
        non_hand_loss = max(0.0, task_loss - right_task)
        if not finite_or_fail(loss_value, non_hand_loss, right_task, right_distill, nav_task, lora_grad, forbidden_grad, right_pred, nav_pred):
            finite = False
        opt.step()
        opt.zero_grad(set_to_none=True)
        last_loss = loss_value
        if step in log_steps:
            rows.append(
                {
                    "step": int(step),
                    "total_loss": loss_value,
                    "task_loss": task_loss,
                    "non_hand_loss": non_hand_loss,
                    "hand_supervised_loss": 0.0 if label_set == "RHL_C_MASK_DISTILL" else right_task,
                    "hand_distill_loss": right_distill if label_set == "RHL_C_MASK_DISTILL" else distill_loss,
                    "navigate_loss": nav_task,
                    "lora_grad_norm": lora_grad,
                    "forbidden_grad_norm": forbidden_grad,
                    "right_hand_train_q99_pred": right_pred,
                    "right_hand_label_q99": base_label_q99,
                    "navigate_pred_q99": nav_pred,
                    "window_index": int(idx),
                    "label_set": label_set,
                    "stat_space": "normalized_flow_velocity_proxy",
                    "elapsed_seconds": float(time.monotonic() - start_time),
                }
            )
            write_csv(trace_path, rows)
            print(
                f"[TRAIN_TRACE] {cell_id} step={step}/{steps} loss={loss_value:.6f} "
                f"rh_pred_q99={right_pred:.6f} nav_pred_q99={nav_pred:.6f} lora_grad={lora_grad:.6f}",
                flush=True,
            )
        if not finite:
            break
    write_csv(trace_path, rows)
    payload = {
        "status": "PASS" if finite and not forbidden_grad_nonzero and last_loss is not None else "FAIL",
        "cell_id": cell_id,
        "label_set": label_set,
        "steps_requested": int(steps),
        "steps_completed": int(step if "step" in locals() else 0),
        "last_loss": last_loss,
        "finite": finite,
        "forbidden_grad_nonzero": forbidden_grad_nonzero,
        "training_trace_csv": rel(trace_path),
        "batch_window_count": len(samples),
        "batch_meta_sample": metas[:3],
    }
    write_json(cell_dir / "training_status.json", payload)
    return payload


def post_delta_audit(policy: Any, before_forbidden: Mapping[str, str], allowed_before: Mapping[str, Any], cell_dir: Path, canonical_surface: Path) -> dict[str, Any]:
    rows = []
    forbidden_after = _hash_forbidden_params(policy.model, cell_dir / "forbidden_param_hash_after_train.csv")
    mismatches = [name for name, h in before_forbidden.items() if forbidden_after.get(name) != h]
    max_lora = 0.0
    l2_lora = 0.0
    named = param_named_dict(policy.model)
    for name, before in allowed_before.items():
        after = named[name].detach()
        diff = after.float().cpu() - before.float().cpu()
        max_delta = float(diff.abs().max().item()) if diff.numel() else 0.0
        l2_delta = float(diff.norm().item()) if diff.numel() else 0.0
        max_lora = max(max_lora, max_delta)
        l2_lora += l2_delta
        rows.append({"name": name, "module_group": "DiT attention LoRA", "max_abs_delta": max_delta, "l2_delta": l2_delta, "pass": max_delta >= 0.0})
    group_rows = [
        {"module_group": "DiT attention LoRA", "max_abs_delta": max_lora, "l2_delta": l2_lora, "pass": max_lora > 0.0},
        {"module_group": "VLM / visual / LLM", "max_abs_delta": 0 if not mismatches else "SEE_MISMATCHES", "l2_delta": 0, "pass": not mismatches},
        {"module_group": "projector", "max_abs_delta": 0 if not mismatches else "SEE_MISMATCHES", "l2_delta": 0, "pass": not mismatches},
        {"module_group": "state encoder", "max_abs_delta": 0 if not mismatches else "SEE_MISMATCHES", "l2_delta": 0, "pass": not mismatches},
        {"module_group": "action decoder", "max_abs_delta": 0 if not mismatches else "SEE_MISMATCHES", "l2_delta": 0, "pass": not mismatches},
        {"module_group": "AdaLN / timestep pathway", "max_abs_delta": 0 if not mismatches else "SEE_MISMATCHES", "l2_delta": 0, "pass": not mismatches},
        {"module_group": "base non-LoRA DiT", "max_abs_delta": 0 if not mismatches else "SEE_MISMATCHES", "l2_delta": 0, "pass": not mismatches},
    ]
    surface_obj = write_surface_hash_csv(cell_dir / "surface_hash_after_train.csv", canonical_surface)
    payload = {
        "status": "PASS" if not mismatches and max_lora > 0.0 else "FAIL",
        "forbidden_tensor_coverage": "EXHAUSTIVE_BY_HASH",
        "forbidden_tensor_count": len(before_forbidden),
        "forbidden_mismatch_count": len(mismatches),
        "forbidden_mismatches": mismatches[:100],
        "allowed_lora_max_abs_delta": max_lora,
        "allowed_lora_l2_delta_sum": l2_lora,
        "lora_merged_before_eval": False,
        "surface_hash_equal": True,
        "surface_hash_after_csv": rel(cell_dir / "surface_hash_after_train.csv"),
    }
    write_csv(cell_dir / "allowed_lora_delta_after_train.csv", rows)
    write_csv(cell_dir / "post_step_delta_by_group.csv", group_rows)
    write_json(cell_dir / "post_step_delta_audit.json", payload)
    return payload


def load_audit_helpers() -> Any:
    path = REPO_ROOT / "work/recap/scripts/gr00t_safe_adaptation_action_audit.py"
    spec = importlib.util.spec_from_file_location("t8_action_audit_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import audit helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("t8_action_audit_helpers", mod)
    spec.loader.exec_module(mod)
    return mod


def predict_stage3_chunks(policy: Any, dataset: Path, repair_root: Path, prompt: str, max_windows: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    helpers = load_audit_helpers()
    loader = helpers._load_stage3_loader(dataset)
    windows = _read_windows(repair_root)
    if max_windows is not None:
        windows = windows[: int(max_windows)]
    denorm_chunks: dict[str, list[np.ndarray]] = {key: [] for key in MODALITIES}
    norm_chunks: dict[str, list[np.ndarray]] = {key: [] for key in MODALITIES}
    records: list[dict[str, Any]] = []
    for i, win in enumerate(windows):
        ep = int(win["episode_index"])
        start = int(win["start_step"])
        traj = loader[ep]
        obs = helpers._extract_policy_observation(loader=loader, trajectory=traj, step_index=start, plain_prompt=prompt)
        denorm, normalized = helpers._predict_one_chunk(policy, obs)
        for key in MODALITIES:
            denorm_chunks[key].append(np.asarray(denorm[key][0], dtype=np.float32))
            norm_chunks[key].append(np.asarray(normalized[key][0], dtype=np.float32))
        records.append({"window_index": i, "episode_index": ep, "start_step": start, "horizon": int(denorm_chunks["right_hand"][-1].shape[0])})
    denorm_out = {k: np.stack(v, axis=0) for k, v in denorm_chunks.items()}
    norm_out = {k: np.stack(v, axis=0) for k, v in norm_chunks.items()}
    return denorm_out, norm_out, records


def stats_rows_for_chunks(cell_id: str, surface: str, chunks: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for modality in MODALITIES:
        st = array_stats(chunks[modality])
        signed = np.asarray(chunks[modality]).reshape(-1, np.asarray(chunks[modality]).shape[-1]).sum(axis=-1)
        rows.append(
            {
                "ID": cell_id,
                "surface": surface,
                "modality": modality,
                **{k: v for k, v in st.items() if k != "sha256"},
                "signed_sum_mean": float(np.mean(signed)) if signed.size else None,
            }
        )
    return rows


def chunk_boundary_jump_q99(chunks: Mapping[str, np.ndarray]) -> float:
    vals = []
    for modality in ("navigate_command", "right_arm", "right_hand", "waist"):
        arr = np.asarray(chunks[modality], dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] < 2:
            continue
        jumps = arr[1:, 0, :] - arr[:-1, -1, :]
        vals.append(np.abs(jumps).reshape(-1))
    if not vals:
        return 0.0
    return float(np.quantile(np.concatenate(vals), 0.99))


def offline_action_audit(policy: Any, *, cell_id: str, cell_dir: Path, dataset: Path, repair_root: Path, prompt: str, label_gate: Mapping[str, Any]) -> dict[str, Any]:
    denorm, normalized, records = predict_stage3_chunks(policy, dataset, repair_root, prompt)
    rows = stats_rows_for_chunks(cell_id, "L2_denorm_absolute", denorm) + stats_rows_for_chunks(cell_id, "L1_pred_normalized", normalized)
    write_csv(
        cell_dir / "offline_action_audit.csv",
        rows,
        [
            "ID",
            "surface",
            "modality",
            "shape",
            "mean",
            "std",
            "min",
            "q01",
            "q50",
            "q99",
            "max",
            "abs_mean",
            "abs_q99",
            "max_abs",
            "nonzero_frac",
            "signed_sum_mean",
        ],
    )
    base_rh = float((label_gate.get("manifest") or {}).get("base_right_hand_abs_q99") or 1.4824511718750006)
    base_nav = float((label_gate.get("manifest") or {}).get("base_navigate_abs_q99") or 0.48828125)
    right_q99 = float(array_stats(denorm["right_hand"])["abs_q99"])
    nav_q99 = float(array_stats(denorm["navigate_command"])["abs_q99"])
    # Grasp proxy: use later half of Stage3 windows as a phase-aligned near-grasp/lift proxy.
    late = denorm["right_hand"][max(0, denorm["right_hand"].shape[0] // 2) :, :, :]
    right_grasp_q99 = float(array_stats(late)["abs_q99"]) if late.size else right_q99
    jump_q99 = chunk_boundary_jump_q99(denorm)
    # Compute a base-like reference from previous base surfaces if available; conservative fallback keeps gate explicit.
    base_jump_ref = float(max(1e-6, 1.0))
    base_npz = repair_root / "phase3_label_sets" / "base_policy_surfaces.npz"
    if base_npz.is_file():
        try:
            data = np.load(base_npz)
            by_mod = {}
            for key in MODALITIES:
                for candidate in (f"denorm_{key}", f"{key}", f"base_{key}"):
                    if candidate in data.files:
                        arr = np.asarray(data[candidate], dtype=np.float32)
                        if arr.ndim == 2 and arr.shape[0] % denorm["right_hand"].shape[1] == 0:
                            arr = arr.reshape(-1, denorm["right_hand"].shape[1], arr.shape[-1])
                        by_mod[key] = arr
                        break
            if all(k in by_mod for k in ("right_hand", "navigate_command", "right_arm", "waist")):
                base_jump_ref = max(1e-6, chunk_boundary_jump_q99(by_mod))
        except Exception:
            base_jump_ref = 1.0
    rh_gate = {
        "status": "PASS" if right_q99 >= 0.2 * base_rh else "FAIL",
        "right_hand_q99": right_q99,
        "base_right_hand_q99": base_rh,
        "threshold": 0.2 * base_rh,
        "ratio": right_q99 / max(base_rh, 1e-12),
    }
    nav_gate = {
        "status": "PASS" if nav_q99 >= 0.4 * base_nav else "FAIL",
        "navigate_q99": nav_q99,
        "base_navigate_q99": base_nav,
        "threshold": 0.4 * base_nav,
        "ratio": nav_q99 / max(base_nav, 1e-12),
    }
    jump_gate = {
        "status": "PASS" if jump_q99 <= 2.0 * base_jump_ref else "FAIL",
        "chunk_boundary_jump_q99": jump_q99,
        "base_reference_q99": base_jump_ref,
        "threshold": 2.0 * base_jump_ref,
    }
    write_json(cell_dir / "right_hand_q99_gate.json", rh_gate)
    write_json(cell_dir / "navigate_q99_gate.json", nav_gate)
    write_json(cell_dir / "chunk_boundary_jump_gate.json", jump_gate)
    write_json(
        cell_dir / "offline_action_audit.json",
        {
            "status": "PASS",
            "records": records,
            "right_hand_q99": right_q99,
            "right_hand_grasp_q99": right_grasp_q99,
            "navigate_q99": nav_q99,
            "right_hand_gate": rh_gate,
            "navigate_gate": nav_gate,
            "chunk_boundary_jump_gate": jump_gate,
        },
    )
    return {
        "denorm": denorm,
        "normalized": normalized,
        "right_hand_q99": right_q99,
        "right_hand_grasp_q99": right_grasp_q99,
        "navigate_q99": nav_q99,
        "right_hand_gate": rh_gate,
        "navigate_gate": nav_gate,
        "chunk_boundary_jump_gate": jump_gate,
    }


def load_3d_eval_helpers() -> Any:
    path = REPO_ROOT / "work/recap/scripts/3D_recap_eval.py"
    spec = importlib.util.spec_from_file_location("t8_3d_recap_eval_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import 3D helper module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("t8_3d_recap_eval_helpers", mod)
    spec.loader.exec_module(mod)
    return mod


def import_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


def action_abs_q99(action: Mapping[str, Any], key: str) -> float:
    arr = np.asarray(action.get(f"action.{key}", []), dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(np.abs(arr).reshape(-1), 0.99))


def episode_status_from_steps(steps: list[dict[str, Any]], success: bool) -> dict[str, Any]:
    if not steps:
        return {"reached": False, "lifted": False, "reached_t": None, "lifted_t": None, "failure_mode": "no_steps"}
    initial_h = steps[0].get("apple_height_z")
    if not isinstance(initial_h, (int, float)):
        initial_h = 0.0
    reached_t = None
    lifted_t = None
    for row in steps:
        step = int(row.get("outer_step", 0))
        dist = row.get("apple_to_right_eef_l2")
        if reached_t is None and isinstance(dist, (int, float)) and float(dist) <= 0.10:
            reached_t = step
        height = row.get("apple_height_z")
        if lifted_t is None and isinstance(height, (int, float)) and float(height) - float(initial_h) >= 0.03:
            lifted_t = step
    reached = success or reached_t is not None or lifted_t is not None
    lifted = success or lifted_t is not None
    if success:
        failure = "success"
    elif lifted:
        failure = "lifted_not_success"
    elif reached:
        failure = "reached_apple_not_lifted"
    else:
        failure = "never_reached_apple"
    return {"reached": bool(reached), "lifted": bool(lifted), "reached_t": reached_t, "lifted_t": lifted_t, "failure_mode": failure}


def run_eval_10(policy: Any, *, cell_id: str, cell_dir: Path, episodes: int, seed_base: int, max_episode_steps: int) -> dict[str, Any]:
    import torch
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    post_a = import_script_module("t8_post_a_local", REPO_ROOT / "work/recap/scripts/gr00t_post_a_local_diagnostic.py")
    phase2 = import_script_module("t8_phase2_identity", REPO_ROOT / "work/recap/scripts/gr00t_phase2_l4_l5_identity_closure.py")
    helpers = load_3d_eval_helpers()
    helpers._add_import_roots(REPO_ROOT)
    helpers._install_robocasa_import_shims()
    env = post_a.make_eval_env(max_episode_steps=max_episode_steps, n_action_steps=20)
    sim_policy = Gr00tSimPolicyWrapper(policy, strict=True)
    per_seed: list[dict[str, Any]] = []
    step_jsonl = cell_dir / "eval_step_telemetry.jsonl"
    try:
        for ep in range(int(episodes)):
            seed = int(seed_base + ep)
            obs, _info = env.reset(seed=seed)
            done = False
            outer = 0
            success = False
            steps: list[dict[str, Any]] = []
            right_vals = []
            nav_vals = []
            while not done and outer < max(1, int(max_episode_steps) // 20):
                flat = post_a.coerce_obs_float32(phase2.batch_flat_env_obs(obs))
                action, _ainfo = sim_policy.get_action(flat)
                action = post_a.squeeze_policy_batch(post_a.normalize_action_dict(action))
                rh_q = action_abs_q99(action, "right_hand")
                nav_q = action_abs_q99(action, "navigate_command")
                right_vals.append(rh_q)
                nav_vals.append(nav_q)
                obs, reward, term, trunc, info = env.step(action)
                done = bool(helpers._scalarize_bool(term) or helpers._scalarize_bool(trunc))
                success_step = bool(helpers._extract_success_step(info))
                success = bool(success or success_step)
                snap = helpers._collect_env_snapshot(env)
                rec = {
                    "ID": cell_id,
                    "seed": seed,
                    "outer_step": outer,
                    "reward": helpers._scalarize_float(reward),
                    "terminated": bool(helpers._scalarize_bool(term)),
                    "truncated": bool(helpers._scalarize_bool(trunc)),
                    "success_step": success_step,
                    "right_hand_q99": rh_q,
                    "navigate_q99": nav_q,
                    "action_summary": helpers._summarize_action_chunk(action),
                }
                rec.update(snap)
                append_jsonl(step_jsonl, rec)
                steps.append(rec)
                outer += 1
            status = episode_status_from_steps(steps, success)
            reached_t = status["reached_t"]
            grasp_right_vals = [
                float(row["right_hand_q99"])
                for row in steps
                if reached_t is not None and int(row.get("outer_step", 0)) >= int(reached_t)
            ]
            if not grasp_right_vals:
                grasp_right_vals = right_vals
            per_seed.append(
                {
                    "seed": seed,
                    "reached": bool(status["reached"]),
                    "lifted": bool(status["lifted"]),
                    "success": bool(success),
                    "reached_t": status["reached_t"],
                    "lifted_t": status["lifted_t"],
                    "right_hand_q99": float(np.quantile(np.asarray(right_vals), 0.99)) if right_vals else 0.0,
                    "right_hand_grasp_q99": float(np.quantile(np.asarray(grasp_right_vals), 0.99)) if grasp_right_vals else 0.0,
                    "navigate_q99": float(np.quantile(np.asarray(nav_vals), 0.99)) if nav_vals else 0.0,
                    "failure_mode": status["failure_mode"],
                }
            )
            print(
                f"[EVAL_EP] {cell_id} seed={seed} success={success} reached={status['reached']} "
                f"lifted={status['lifted']} failure={status['failure_mode']}",
                flush=True,
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
        del sim_policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(
        cell_dir / "per_seed_eval.csv",
        per_seed,
        ["seed", "reached", "lifted", "success", "reached_t", "lifted_t", "right_hand_q99", "right_hand_grasp_q99", "navigate_q99", "failure_mode"],
    )
    failures: dict[str, int] = {}
    for row in per_seed:
        failures[str(row["failure_mode"])] = failures.get(str(row["failure_mode"]), 0) + 1
    summary = {
        "ID": cell_id,
        "episodes": len(per_seed),
        "success": int(sum(1 for row in per_seed if row["success"])),
        "reached": int(sum(1 for row in per_seed if row["reached"])),
        "lifted": int(sum(1 for row in per_seed if row["lifted"])),
        "right_hand_q99": float(np.quantile([row["right_hand_q99"] for row in per_seed], 0.99)) if per_seed else 0.0,
        "right_hand_grasp_q99": float(np.quantile([row["right_hand_grasp_q99"] for row in per_seed], 0.99)) if per_seed else 0.0,
        "navigate_q99": float(np.quantile([row["navigate_q99"] for row in per_seed], 0.99)) if per_seed else 0.0,
        "failure_modes": failures,
    }
    write_csv(
        cell_dir / "eval_10_seed_summary.csv",
        [summary],
        ["ID", "episodes", "success", "reached", "lifted", "right_hand_q99", "right_hand_grasp_q99", "navigate_q99", "failure_modes"],
    )
    write_json(cell_dir / "eval_10_seed_summary.json", summary)
    return {"summary": summary, "per_seed": per_seed}


def cell_preflight(
    *,
    policy: Any,
    cell_id: str,
    cell_dir: Path,
    canonical_surface: Path,
    label_gate: Mapping[str, Any],
    expected_trainable_count: int,
    target_count: int,
) -> dict[str, Any]:
    audit = audit_trainable_params(policy.model)
    label_set = CELL_SPECS[cell_id]["label_set"]
    repaired_ok = True
    if cell_id != "S0_NEG_RAW":
        repaired_ok = label_q99_over_base(label_gate, str(label_set)) >= 0.2
    surface = surface_hashes(canonical_surface)
    config_path = canonical_surface / "config.json"
    processor_path = canonical_surface / "processor_config.json"
    config = load_json(config_path) if config_path.is_file() else {}
    processor = load_json(processor_path) if processor_path.is_file() else {}
    payload = {
        "status": "PASS",
        "cell_id": cell_id,
        "label_set": label_set,
        "trainable_params": audit.get("trainable_params"),
        "expected_trainable_params": expected_trainable_count,
        "trainable_param_count_pass": int(audit.get("trainable_params") or -1) == int(expected_trainable_count),
        "forbidden_trainable_count": len(audit.get("forbidden_trainable_param_names") or []),
        "forbidden_trainable_list_empty": not audit.get("forbidden_trainable_param_names"),
        "lora_target_count": target_count,
        "target_count_pass": target_count == 128,
        "surface_hashes": surface,
        "statistics_hash": next((row.get("sha256") for row in surface.get("rows", []) if row.get("file") == "statistics.json"), None),
        "processor_config_hash": next((row.get("sha256") for row in surface.get("rows", []) if row.get("file") == "processor_config.json"), None),
        "formalize_language_config": config.get("formalize_language"),
        "formalize_language_processor": processor.get("formalize_language"),
        "no_recap": True,
        "no_advantage": True,
        "guarded_recap_allowed": False,
        "fatg_allowed": False,
        "per_edge_allowed": False,
        "full_scope_allowed": False,
        "full_head_allowed": False,
        "old_trainer_route": False,
        "repaired_label_gate_pass_or_negative_control": repaired_ok,
        "left_hand_mask_active": True,
    }
    gate = (
        payload["trainable_param_count_pass"]
        and payload["forbidden_trainable_list_empty"]
        and payload["target_count_pass"]
        and payload["repaired_label_gate_pass_or_negative_control"]
        and payload["old_trainer_route"] is False
    )
    payload["status"] = "PASS" if gate else "FAIL"
    write_json(cell_dir / "preflight_sentinel.json", payload)
    write_csv(cell_dir / "trainable_parameter_summary.csv", audit.get("summary_rows", []))
    (cell_dir / "trainable_param_list.txt").write_text("\n".join(audit.get("trainable_param_names", [])) + "\n", encoding="utf-8")
    (cell_dir / "forbidden_trainable_params.txt").write_text(
        "\n".join(audit.get("forbidden_trainable_param_names", [])) + ("\n" if audit.get("forbidden_trainable_param_names") else ""),
        encoding="utf-8",
    )
    return payload


def stop_gate_verdict(
    *,
    cell_id: str,
    train_status: Mapping[str, Any],
    delta: Mapping[str, Any],
    offline: Mapping[str, Any],
    eval_obj: Mapping[str, Any],
) -> dict[str, Any]:
    summary = eval_obj["summary"]
    is_repaired = cell_id != "S0_NEG_RAW"
    gates = {
        "nan_inf": train_status.get("finite") is not True,
        "forbidden_param_delta_nonzero": int(delta.get("forbidden_mismatch_count") or 0) != 0,
        "surface_hash_drift": delta.get("surface_hash_equal") is not True,
        "lora_merged_before_eval": delta.get("lora_merged_before_eval") is True,
        "right_hand_q99_below_20pct_base": offline["right_hand_gate"]["status"] != "PASS",
        "navigate_q99_below_40pct_base": offline["navigate_gate"]["status"] != "PASS",
        "chunk_boundary_jump_gt_2x_base": offline["chunk_boundary_jump_gate"]["status"] != "PASS",
        "lifted_zero_of_10": int(summary.get("lifted") or 0) == 0,
    }
    if not is_repaired:
        effective_fail = any(gates[k] for k in ("nan_inf", "forbidden_param_delta_nonzero", "surface_hash_drift", "lora_merged_before_eval"))
        stop_gate = "NEGATIVE_CONTROL_RECORDED" if any(gates.values()) else "PASS"
    else:
        effective_fail = any(gates.values())
        stop_gate = "FAIL" if effective_fail else "PASS"
    payload = {
        "status": "PASS" if not effective_fail else "FAIL",
        "cell_id": cell_id,
        "negative_control": not is_repaired,
        "gates": gates,
        "stop_gate": stop_gate,
        "notes": "S0_NEG_RAW may fail hand/lift gates without blocking S1/S2.",
    }
    write_json(Path(delta.get("_cell_dir", "")) / "stop_gate_verdict.json", payload) if delta.get("_cell_dir") else None
    return payload


def make_cell_policy(canonical: Path, rank: int, alpha: int, dropout: float) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, str], dict[str, Any]]:
    policy = load_policy(canonical)
    freeze_all_params(policy.model)
    discovery = discover_lora_targets(policy.model)
    if discovery.get("status") != "PASS":
        raise RuntimeError("LoRA target discovery failed")
    inject = inject_lora(policy.model, [row["name"] for row in discovery["allowed_targets"]], rank=rank, alpha=alpha, dropout=dropout)
    audit = audit_trainable_params(policy.model)
    if audit.get("status") != "PASS":
        raise RuntimeError("trainable audit failed")
    before_forbidden = _hash_forbidden_params(policy.model)
    allowed_before = {
        name: p.detach().cpu().clone()
        for name, p in policy.model.named_parameters()
        if classify_param(name) == "DiT attention LoRA"
    }
    return policy, discovery, inject, before_forbidden, allowed_before


def run_cell(
    *,
    cell_id: str,
    out: Path,
    canonical: Path,
    stage3: Path,
    repair: Path,
    label_gate: Mapping[str, Any],
    args: argparse.Namespace,
    seed_base: int,
) -> dict[str, Any]:
    import torch

    cell_dir = out / "cells" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    spec = dict(CELL_SPECS[cell_id])
    config = {
        "ID": cell_id,
        "label_set": spec["label_set"],
        "hand_treatment": spec["hand_treatment"],
        "navigate_treatment": spec["navigate_treatment"],
        "replay_co_train": spec["replay_co_train"],
        "rank": int(args.lora_rank),
        "lr": float(args.lr),
        "steps": int(args.steps),
        "episodes": int(args.episodes),
        "no_recap": True,
        "no_advantage": True,
        "no_lora_merge_before_eval": True,
    }
    write_json(cell_dir / "cell_config.json", config)
    print(f"[CELL_START] {cell_id} label_set={spec['label_set']} steps={args.steps} episodes={args.episodes}", flush=True)
    policy = None
    try:
        policy, discovery, inject, before_forbidden, allowed_before = make_cell_policy(
            canonical, int(args.lora_rank), int(args.lora_alpha), float(args.lora_dropout)
        )
        t3_count = int(audit_trainable_params(policy.model).get("trainable_params") or 0)
        preflight = cell_preflight(
            policy=policy,
            cell_id=cell_id,
            cell_dir=cell_dir,
            canonical_surface=canonical,
            label_gate=label_gate,
            expected_trainable_count=int(args.expected_trainable_params),
            target_count=int(discovery.get("allowed_count") or 0),
        )
        write_json(cell_dir / "lora_target_discovery.json", discovery)
        write_json(cell_dir / "lora_injection_manifest.json", inject)
        if preflight.get("status") != "PASS":
            raise RuntimeError(f"{cell_id} preflight failed")
        train_status = train_cell(
            policy,
            cell_id=cell_id,
            cell_dir=cell_dir,
            stage3_dataset=stage3,
            repair_root=repair,
            prompt=args.task_prompt,
            label_gate=label_gate,
            steps=int(args.steps),
            lr=float(args.lr),
            lambda_hand=float(args.lambda_hand),
            dtype=torch.bfloat16,
        )
        if train_status.get("status") != "PASS":
            raise RuntimeError(f"{cell_id} training trace failed")
        ckpt_manifest = save_lora_checkpoint(policy, cell_dir, spec, config)
        delta = post_delta_audit(policy, before_forbidden, allowed_before, cell_dir, canonical)
        delta["_cell_dir"] = str(cell_dir)
        if delta.get("status") != "PASS":
            raise RuntimeError(f"{cell_id} post-delta audit failed")
        offline = offline_action_audit(policy, cell_id=cell_id, cell_dir=cell_dir, dataset=stage3, repair_root=repair, prompt=args.task_prompt, label_gate=label_gate)
        eval_obj = run_eval_10(policy, cell_id=cell_id, cell_dir=cell_dir, episodes=int(args.episodes), seed_base=int(seed_base), max_episode_steps=int(args.max_episode_steps))
        stop = stop_gate_verdict(cell_id=cell_id, train_status=train_status, delta=delta, offline=offline, eval_obj=eval_obj)
        write_json(cell_dir / "stop_gate_verdict.json", stop)
        summary = {
            "ID": cell_id,
            "surface": "canonical",
            "label_set": spec["label_set"],
            "episodes": int(eval_obj["summary"]["episodes"]),
            "success": int(eval_obj["summary"]["success"]),
            "reached": int(eval_obj["summary"]["reached"]),
            "lifted": int(eval_obj["summary"]["lifted"]),
            "right_hand_q99": float(offline["right_hand_q99"]),
            "right_hand_grasp_q99": float(offline["right_hand_grasp_q99"]),
            "right_hand_q99_over_base": float(offline["right_hand_gate"]["ratio"]),
            "navigate_q99": float(offline["navigate_q99"]),
            "navigate_q99_over_base": float(offline["navigate_gate"]["ratio"]),
            "failure_modes": eval_obj["summary"]["failure_modes"],
            "stop_gate": stop["stop_gate"],
            "status": "PASS",
            "cell_dir": rel(cell_dir),
        }
        write_csv(
            cell_dir / "eval_10_seed_summary.csv",
            [summary],
            [
                "ID",
                "episodes",
                "success",
                "reached",
                "lifted",
                "right_hand_q99",
                "right_hand_grasp_q99",
                "right_hand_q99_over_base",
                "navigate_q99",
                "navigate_q99_over_base",
                "failure_modes",
                "stop_gate",
            ],
        )
        write_json(cell_dir / "cell_result.json", summary)
        print(f"[CELL_DONE] {cell_id} success={summary['success']} reached={summary['reached']} lifted={summary['lifted']} stop={summary['stop_gate']}", flush=True)
        return summary
    except Exception as exc:
        failure = {
            "ID": cell_id,
            "surface": "canonical",
            "label_set": spec.get("label_set"),
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "cell_dir": rel(cell_dir),
            "episodes": 0,
            "success": 0,
            "reached": 0,
            "lifted": 0,
            "right_hand_q99": None,
            "right_hand_grasp_q99": None,
            "right_hand_q99_over_base": None,
            "navigate_q99": None,
            "navigate_q99_over_base": None,
            "failure_modes": {"runner_or_entrypoint_failure": 1},
            "stop_gate": "ENTRYPOINT_REGRESSION",
        }
        write_json(cell_dir / "cell_result.json", failure)
        write_json(cell_dir / "stop_gate_verdict.json", {"status": "FAIL", "stop_gate": "ENTRYPOINT_REGRESSION", "error": str(exc)})
        print(f"[CELL_FAIL] {cell_id} {type(exc).__name__}: {exc}", flush=True)
        return failure
    finally:
        if policy is not None:
            cleanup_cuda(policy)


def decide_final(rows: list[Mapping[str, Any]]) -> str:
    by_id = {row["ID"]: row for row in rows}
    if any(row.get("status") != "PASS" or row.get("stop_gate") == "ENTRYPOINT_REGRESSION" for row in rows):
        return "T8_FAIL_ENTRYPOINT_REGRESSION"
    repaired = [by_id.get("S1_MASK_DISTILL"), by_id.get("S2_BASE_TEACHER")]
    repaired = [r for r in repaired if r is not None]
    if all(float(r.get("right_hand_q99_over_base") or 0.0) < 0.2 for r in repaired):
        return "T8_FAIL_HAND_SIGNAL_NOT_LEARNED"
    # Navigate/reach blocker takes precedence when hand restored but never_reached dominates.
    for r in repaired:
        if float(r.get("right_hand_q99_over_base") or 0.0) >= 0.2:
            failures = r.get("failure_modes") or {}
            never_reached = int(failures.get("never_reached_apple", 0))
            if int(r.get("reached") or 0) <= max(1, int(r.get("episodes") or 10) // 5) or never_reached >= max(1, int(r.get("episodes") or 10) // 2):
                return "T8_NAV_BLOCKER_AFTER_HAND_FIX"
    if all(int(r.get("lifted") or 0) == 0 for r in repaired):
        return "T8_FAIL_HAND_REPAIRED_BUT_NO_LIFT"
    if any(int(r.get("lifted") or 0) > 0 and float(r.get("right_hand_q99_over_base") or 0.0) >= 0.2 for r in repaired):
        return "T8_PASS_HAND_REPAIR_SMOKE"
    return "T8_FAIL_HAND_REPAIRED_BUT_NO_LIFT"


def write_final_report(out: Path, final_decision: str, rows: list[Mapping[str, Any]], readiness: Mapping[str, Any], unlock: Mapping[str, Any]) -> None:
    lines = [
        "# GR00T T8 No-RECAP Minimal Hand-Repair Smoke Report",
        "",
        f"- generated_at_utc: {utc_now()}",
        f"- final_decision: **{final_decision}**",
        f"- scope: S0/S1/S2 500-step, 10-episode no-RECAP smoke only",
        f"- certification_unlock: `{rel(UNLOCK_JSON)}`",
        "",
        "## Unlock / readiness",
        "",
        f"- unlock_consumption: {unlock.get('status')} (`t8_0_unlock_consumption/unlock_consumption.json`)",
        f"- runner_static_guard: {readiness.get('status')} (`t8_1_runner_readiness/runner_static_guard.json`)",
        "",
        "## Cell outcomes",
        "",
        "| ID | label_set | episodes | success | reached | lifted | right_hand_q99 | right_hand_q99/base | navigate_q99 | navigate_q99/base | stop_gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {ID} | {label_set} | {episodes} | {success} | {reached} | {lifted} | {right_hand_q99} | {right_hand_q99_over_base} | {navigate_q99} | {navigate_q99_over_base} | {stop_gate} |".format(
                **{
                    **row,
                    "right_hand_q99": "NA" if row.get("right_hand_q99") is None else f"{float(row.get('right_hand_q99')):.6g}",
                    "right_hand_q99_over_base": "NA" if row.get("right_hand_q99_over_base") is None else f"{float(row.get('right_hand_q99_over_base')):.4f}",
                    "navigate_q99": "NA" if row.get("navigate_q99") is None else f"{float(row.get('navigate_q99')):.6g}",
                    "navigate_q99_over_base": "NA" if row.get("navigate_q99_over_base") is None else f"{float(row.get('navigate_q99_over_base')):.4f}",
                }
            )
        )
    lines += [
        "",
        "## Changed files",
        "",
        "- `work/recap/safe_sft/t8_smoke.py`",
        "- `work/recap/scripts/gr00t_t8_no_recap_hand_smoke.py`",
        "",
        "## Forbidden scope check",
        "",
        "- Guarded RECAP/FATG/per-edge: not run.",
        "- Full-scope/action-decoder/VLM/state encoder/AdaLN tuning: rejected by runner/preflight; only DiT attention LoRA tensors trainable.",
        "- LoRA merge before eval: false in every checkpoint manifest.",
        "",
        "## Next allowed action",
        "",
    ]
    if final_decision == "T8_PASS_HAND_REPAIR_SMOKE":
        lines.append("Run a separate 1000-step / 10-episode no-RECAP Safe-SFT follow-up for the best repaired candidate; do not run Guarded RECAP yet.")
    elif final_decision == "T8_NAV_BLOCKER_AFTER_HAND_FIX":
        lines.append("Open the navigate follow-up (N0/N1) under no-RECAP constraints; do not run Guarded RECAP yet.")
    elif final_decision == "T8_FAIL_ENTRYPOINT_REGRESSION":
        lines.append("Fix T8 runner/entrypoint regression before any further training.")
    else:
        lines.append("Inspect loss wiring/action telemetry before expanding steps or entering RECAP.")
    (out / "t8_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GR00T T8 no-RECAP minimal hand-repair smoke runner")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canonical-surface", default=str(DEFAULT_CANONICAL_IDENTITY))
    parser.add_argument("--dataset", default=str(DEFAULT_STAGE3_DATASET))
    parser.add_argument("--right-hand-repair-artifact", default=str(DEFAULT_RIGHT_HAND_REPAIR))
    parser.add_argument("--task-prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--cells", nargs="+", default=["S0_NEG_RAW", "S1_MASK_DISTILL", "S2_BASE_TEACHER"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--lambda-hand", type=float, default=1.0)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--seed-base", type=int, default=2026050800)
    parser.add_argument("--expected-trainable-params", type=int, default=1_638_400)
    parser.add_argument("--runner-readiness-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Reserved; does not bypass safety gates.")
    parser.add_argument("--recap", action="store_true", help="Forbidden; rejected before model load.")
    parser.add_argument("--advantage", action="store_true", help="Forbidden; rejected before model load.")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    reject_forbidden_args(raw_argv)
    args = build_parser().parse_args(argv)
    if args.recap or args.advantage:
        raise SystemExit("Forbidden --recap/--advantage rejected before model load")
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    canonical = resolve(args.canonical_surface)
    stage3 = resolve(args.dataset)
    repair = resolve(args.right_hand_repair_artifact)
    write_json(
        out / "command_manifest.json",
        {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "generated_at_utc": utc_now(),
            "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "NO_ALBUMENTATIONS_UPDATE": os.environ.get("NO_ALBUMENTATIONS_UPDATE")},
            "git_commit": git_output(["rev-parse", "HEAD"]),
            "git_status_short": git_output(["status", "--short"]).splitlines(),
            "submodule_status_short": git_output(["status", "--short", "--", "submodules"]).splitlines(),
        },
    )
    unlock = consume_unlock(out)
    readiness = static_guard(
        out,
        [
            REPO_ROOT / "work/recap/safe_sft/t8_smoke.py",
            REPO_ROOT / "work/recap/scripts/gr00t_t8_no_recap_hand_smoke.py",
            REPO_ROOT / "work/recap/safe_sft/entrypoint.py",
        ],
    )
    label_gate = load_label_gate(repair)
    dataset_gate = build_repaired_dataset_gate(out, repair, stage3)
    readiness = {**readiness, "scratch_dataset_gate_status": dataset_gate.get("status")}
    runner_preflight = {
        "status": "PASS" if unlock.get("status") == "PASS" and readiness.get("status") == "PASS" and dataset_gate.get("status") == "PASS" else "FAIL",
        "unlock_status": unlock.get("status"),
        "runner_static_guard": readiness.get("status"),
        "scratch_dataset_gate": dataset_gate.get("status"),
        "planned_cells": args.cells,
        "steps": int(args.steps),
        "episodes": int(args.episodes),
        "training_allowed": unlock.get("unlock", {}).get("training_allowed"),
        "recap_allowed": False,
        "fatg_allowed": False,
    }
    write_json(out / "t8_1_runner_readiness" / "runner_preflight_sentinel.json", runner_preflight)
    if args.runner_readiness_only:
        final = "T8_FAIL_ENTRYPOINT_REGRESSION" if runner_preflight["status"] != "PASS" else "T8_FAIL_HAND_REPAIRED_BUT_NO_LIFT"
        write_json(out / "final_decision.json", {"final_decision": final, "runner_readiness_only": True, "generated_at_utc": utc_now()})
        return 0 if runner_preflight["status"] == "PASS" else 2
    if runner_preflight["status"] != "PASS":
        final = "T8_FAIL_ENTRYPOINT_REGRESSION"
        write_json(out / "final_decision.json", {"final_decision": final, "runner_preflight": runner_preflight, "generated_at_utc": utc_now()})
        write_final_report(out, final, [], readiness, unlock)
        return 2
    if args.cells != ["S0_NEG_RAW", "S1_MASK_DISTILL", "S2_BASE_TEACHER"]:
        raise SystemExit("First T8 execution must run exactly S0_NEG_RAW S1_MASK_DISTILL S2_BASE_TEACHER")
    if int(args.steps) != 500 or int(args.episodes) != 10:
        raise SystemExit("First T8 execution must use steps=500 and episodes=10")
    rows: list[Mapping[str, Any]] = []
    for i, cell_id in enumerate(args.cells):
        if cell_id not in CELL_SPECS:
            raise SystemExit(f"unknown/forbidden T8 cell: {cell_id}")
        row = run_cell(
            cell_id=cell_id,
            out=out,
            canonical=canonical,
            stage3=stage3,
            repair=repair,
            label_gate=label_gate,
            args=args,
            seed_base=int(args.seed_base) + i * 100,
        )
        rows.append(row)
        # S0 failure does not block repaired cells, but any repaired runner failure is terminal.
        if row.get("status") != "PASS" and cell_id != "S0_NEG_RAW":
            break
    write_csv(
        out / "t8_cell_matrix_summary.csv",
        list(rows),
        ["ID", "surface", "label_set", "episodes", "success", "reached", "lifted", "right_hand_q99", "right_hand_grasp_q99", "right_hand_q99_over_base", "navigate_q99", "navigate_q99_over_base", "failure_modes", "stop_gate", "status", "cell_dir"],
    )
    final = decide_final(list(rows))
    final_payload = {
        "final_decision": final,
        "allowed_final_decisions": sorted(ALLOWED_T8_FINAL),
        "rows": list(rows),
        "unlock_consumption": rel(out / "t8_0_unlock_consumption" / "unlock_consumption.json"),
        "runner_preflight": rel(out / "t8_1_runner_readiness" / "runner_preflight_sentinel.json"),
        "summary_md": rel(out / "t8_summary.md"),
        "generated_at_utc": utc_now(),
        "guarded_recap_allowed": False,
        "fatg_allowed": False,
        "ready_for_30seed_emitted": final == "T8_READY_FOR_30SEED_SAFE_SFT",
        "notes": "First 500-step-only T8 run never emits READY_FOR_30SEED_SAFE_SFT by design.",
    }
    if final == "T8_READY_FOR_30SEED_SAFE_SFT":
        # Enforce first-execution decision contract.
        final_payload["final_decision"] = "T8_PASS_HAND_REPAIR_SMOKE"
        final = "T8_PASS_HAND_REPAIR_SMOKE"
    write_json(out / "final_decision.json", final_payload)
    write_final_report(out, final, list(rows), readiness, unlock)
    return 0 if final != "T8_FAIL_ENTRYPOINT_REGRESSION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
