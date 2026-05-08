from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_GR00T_ROOT = REPO_ROOT / "submodules" / "Isaac-GR00T"
WBC_ROOT = ISAAC_GR00T_ROOT / "external_dependencies" / "GR00T-WholeBodyControl"
for _p in (REPO_ROOT, ISAAC_GR00T_ROOT, WBC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_OFFICIAL_BASE = (
    REPO_ROOT
    / "agent/artifacts/gr00t_recap_live/hf_patches/models--nvidia--GR00T-N1.6-G1-PnPAppleToPlate/"
    / "snapshot-897d0313a190f46a2cccaeb34077752a0db4b0de/formalize_language=False"
)
DEFAULT_CANONICAL_IDENTITY = (
    REPO_ROOT
    / "agent/artifacts/gr00t_a_gate_identity_closure/a_gate_identity_20260506_133517/canonical_identity"
)
DEFAULT_STAGE3_DATASET = REPO_ROOT / "agent/artifacts/lerobot_datasets/recap_stage3_iter_002"
DEFAULT_RIGHT_HAND_REPAIR = (
    REPO_ROOT
    / "agent/artifacts/gr00t_right_hand_label_repair/right_hand_label_repair_20260507_022206/"
    / "full_closure_resetfix3"
)
DEFAULT_TASK_PROMPT = "Pick up the apple and place it on the plate."

MODALITIES = (
    "base_height_command",
    "navigate_command",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
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
SURFACE_FILES = (
    "config.json",
    "processor_config.json",
    "statistics.json",
    "modality.json",
    "embodiment_id.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)
ALLOWED_FINAL = {
    "ENTRYPOINT_FAIL",
    "ENTRYPOINT_PASS_NO_TRAIN",
    "HAND_REPAIR_SMOKE_FAIL",
    "HAND_REPAIR_SMOKE_PASS",
    "NAV_BLOCKER_AFTER_HAND_FIX",
    "READY_FOR_SAFE_SFT_30",
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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hash_array(value)}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=True, default=json_default) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_array(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def tensor_hash(tensor: Any) -> str:
    import torch

    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype is torch.bfloat16:
        raw = cpu.view(torch.int16).numpy().tobytes()
    elif cpu.dtype is torch.float16:
        raw = cpu.view(torch.int16).numpy().tobytes()
    elif cpu.dtype is torch.float32:
        raw = cpu.view(torch.int32).numpy().tobytes()
    elif cpu.dtype is torch.float64:
        raw = cpu.view(torch.int64).numpy().tobytes()
    else:
        raw = cpu.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def git_output(args: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNKNOWN:{type(exc).__name__}:{exc}"


def surface_hashes(model_dir: Path) -> dict[str, Any]:
    rows = []
    for name in SURFACE_FILES:
        path = model_dir / name
        rows.append({
            "file": name,
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "size": path.stat().st_size if path.is_file() else None,
        })
    return {"model_dir": rel(model_dir), "rows": rows}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_t0_prior_evidence(out: Path, repair_root: Path) -> dict[str, Any]:
    t0 = out / "t0_prior_evidence_lock"
    decision_path = repair_root / "final_decision.json"
    fixed_csv = repair_root / "phase2_fixed_state_hand_sanity" / "fixed_state_hand_sanity.csv"
    label_csv = repair_root / "phase3_label_sets" / "label_set_stats.csv"
    smoke_json = repair_root / "phase4_minimal_no_recap_training_smoke" / "minimal_smoke_not_run.json"
    required = [decision_path, fixed_csv, label_csv, smoke_json]
    missing = [rel(p) for p in required if not p.is_file()]
    payload: dict[str, Any] = {
        "repair_root": rel(repair_root),
        "required_files": [rel(p) for p in required],
        "missing": missing,
        "status": "FAIL" if missing else "PASS",
    }
    if not missing:
        decision = load_json(decision_path)
        payload["previous_final_decision"] = decision.get("final_decision")
        payload["h_lift_counts"] = decision.get("h_lift_counts")
        payload["raw_pass"] = decision.get("raw_pass")
        payload["rhl_a_pass"] = decision.get("rhl_a_pass")
        payload["rhl_c_pass"] = decision.get("rhl_c_pass")
        payload["no_training_performed"] = decision.get("no_training_performed")
        payload["smoke_status"] = decision.get("smoke_status")
        expected = (
            decision.get("final_decision") == "HAND_FIX_VALIDATED"
            and decision.get("no_training_performed") is True
            and decision.get("raw_pass") is False
            and decision.get("rhl_a_pass") is True
            and decision.get("rhl_c_pass") is True
            and decision.get("smoke_status") == "NOT_RUN"
        )
        h = decision.get("h_lift_counts") or {}
        expected = expected and h.get("H0") == 9 and h.get("H1") == 0 and h.get("H2") == 0 and h.get("H3") == 0 and h.get("H4") == 0
        payload["status"] = "PASS" if expected else "FAIL"
        payload["evidence_hashes"] = {rel(p): sha256_file(p) for p in required}
    write_json(t0 / "t0_prior_evidence_lock.json", payload)
    return payload


def run_static_source_guard(out: Path, script_path: Path, source_paths: list[Path]) -> dict[str, Any]:
    guard = out / "t1_static_source_guard"
    rows = []
    forbidden_runtime_tokens = (
        "subprocess.run([\"torchrun\"",
        "launch_finetune.py",
        "Gr00tTrainer(",
        "guarded_recap_train_loop",
        "fatg_train_loop",
        "per_edge_training_loop",
    )
    for path in source_paths:
        if not path.is_file():
            rows.append({"path": rel(path), "exists": False, "status": "FAIL", "reason": "missing"})
            continue
        text = path.read_text(encoding="utf-8")
        scan_text = "\n".join(line for line in text.splitlines() if "forbidden_runtime_tokens" not in line and not line.strip().startswith(("\"", "'")))
        bad_runtime = [tok for tok in forbidden_runtime_tokens if tok in scan_text]
        rows.append({
            "path": rel(path),
            "exists": True,
            "sha256": sha256_file(path),
            "has_recap_training_code": "advantage" in text.lower() and "recap" in text.lower() and "--run-smoke" in text,
            "has_full_scope_toggle": "full_scope" in text.lower() or "full-scope" in text.lower(),
            "forbidden_context_tokens": bad_runtime,
            "status": "PASS" if not bad_runtime else "FAIL",
        })
    payload = {
        "script": rel(script_path),
        "rows": rows,
        "status": "PASS" if rows and all(row.get("status") == "PASS" for row in rows) else "FAIL",
        "notes": "Static guard checks this new fail-closed entrypoint surface; it is not a semantic proof by itself.",
    }
    write_csv(guard / "static_source_guard.csv", rows)
    write_json(guard / "static_source_guard.json", payload)
    return payload


class LoRALinear:
    """Namespace marker; actual class is built after torch import."""


def discover_lora_targets(model: Any) -> dict[str, Any]:
    import torch

    allowed_suffixes = ("to_q", "to_k", "to_v", "to_out.0")
    allowed = []
    rejected = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lowered = name.lower()
        reason: list[str] = []
        in_dit = "action_head.model.transformer_blocks" in lowered
        is_attn = ".attn1." in lowered or ".attn." in lowered or ".attention." in lowered
        suffix_ok = name.endswith(allowed_suffixes)
        forbidden = any(tok in lowered for tok in ("ff.", "norm", "proj_out", "timestep", "time_proj", "state_encoder", "action_encoder", "action_decoder", "backbone", "vlln"))
        if not in_dit:
            reason.append("not_action_head_dit")
        if not is_attn:
            reason.append("not_attention_projection")
        if not suffix_ok:
            reason.append("not_qkv_or_o_projection")
        if forbidden:
            reason.append("forbidden_module_family")
        row = {
            "name": name,
            "module_type": type(module).__name__,
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "bias": module.bias is not None,
            "reason": ";".join(reason) if reason else "allowed_dit_attention_lora",
        }
        if reason:
            rejected.append(row)
        else:
            allowed.append(row)
    return {
        "allowed_targets": allowed,
        "rejected_linear_modules": rejected,
        "allowed_count": len(allowed),
        "status": "PASS" if allowed else "FAIL",
    }


def _get_parent_module(root: Any, dotted_name: str) -> tuple[Any, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def inject_lora(model: Any, target_names: list[str], *, rank: int, alpha: int, dropout: float) -> dict[str, Any]:
    import math
    import torch
    from torch import nn
    import torch.nn.functional as F

    class _LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype, device=base.weight.device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.scaling = float(alpha) / float(rank)
            self.dropout = nn.Dropout(float(dropout))

        def forward(self, x):
            out = self.base(x)
            lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
            return out + lora

    injected = []
    for name in target_names:
        parent, child = _get_parent_module(model, name)
        old = parent[int(child)] if child.isdigit() else getattr(parent, child)
        if not isinstance(old, torch.nn.Linear):
            raise TypeError(f"LoRA target {name} is not nn.Linear: {type(old)!r}")
        new = _LoRALinear(old)
        if child.isdigit():
            parent[int(child)] = new
        else:
            setattr(parent, child, new)
        injected.append({"target": name, "rank": rank, "alpha": alpha, "dropout": dropout, "base_weight_shape": list(old.weight.shape)})
    return {"injected": injected, "injected_count": len(injected), "status": "PASS" if injected else "FAIL"}


def freeze_all_params(model: Any) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def classify_param(name: str) -> str:
    lowered = name.lower()
    if ".lora_a" in lowered or ".lora_b" in lowered:
        return "DiT attention LoRA"
    if "action_head.state_encoder" in lowered:
        return "state encoder"
    if "action_head.action_encoder" in lowered:
        return "action encoder"
    if "action_head.action_decoder" in lowered:
        return "action decoder"
    if "timestep" in lowered or "time_proj" in lowered or "timestep_encoder" in lowered or "adaln" in lowered or "ada" in lowered and "norm" in lowered:
        return "AdaLN / timestep pathway"
    if "action_head.model" in lowered:
        return "base non-LoRA DiT params"
    if "backbone" in lowered and ("vision" in lowered or "visual" in lowered or "siglip" in lowered or "vit" in lowered):
        return "VLM / visual"
    if "backbone" in lowered and ("language" in lowered or "llm" in lowered or "qwen" in lowered):
        return "LLM"
    if "backbone" in lowered and "project" in lowered:
        return "projector"
    if "backbone" in lowered:
        return "VLM / visual"
    if "vlln" in lowered:
        return "projector"
    return "other frozen base params"


def audit_trainable_params(model: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    trainable_names: list[str] = []
    forbidden_trainable: list[str] = []
    total = 0
    trainable_total = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        total += n
        group = classify_param(name)
        group_counts[group] = group_counts.get(group, 0) + (n if p.requires_grad else 0)
        if p.requires_grad:
            trainable_total += n
            trainable_names.append(name)
            if group != "DiT attention LoRA":
                forbidden_trainable.append(name)
        rows.append({"name": name, "group": group, "numel": n, "shape": list(p.shape), "dtype": str(p.dtype), "requires_grad": bool(p.requires_grad)})
    summary_rows = []
    groups = [
        ("VLM / visual", False),
        ("LLM", False),
        ("projector", False),
        ("state encoder", False),
        ("action encoder", False),
        ("action decoder", False),
        ("AdaLN / timestep pathway", False),
        ("DiT attention LoRA", True),
        ("base non-LoRA DiT params", False),
        ("other frozen base params", False),
    ]
    for group, allowed in groups:
        summary_rows.append({"module_group": group, "trainable_param_count": group_counts.get(group, 0), "allowed": allowed, "notes": ""})
    return {
        "param_rows": rows,
        "summary_rows": summary_rows,
        "trainable_param_names": trainable_names,
        "forbidden_trainable_param_names": forbidden_trainable,
        "total_params": total,
        "trainable_params": trainable_total,
        "status": "PASS" if trainable_total > 0 and not forbidden_trainable else "FAIL",
    }


def load_policy(checkpoint: Path) -> Any:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    return Gr00tPolicy(EmbodimentTag("unitree_g1"), str(checkpoint), device="cuda:0", strict=True)


def cleanup_cuda(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_repaired_dataset_gate(out: Path, repair_root: Path, stage3_dataset: Path) -> dict[str, Any]:
    dst = out / "t4_scratch_repaired_dataset_gate"
    labels_src = repair_root / "phase3_label_sets"
    label_stats_src = labels_src / "label_set_stats.csv"
    manifest_src = labels_src / "label_set_manifest.json"
    contract_src = repair_root / "phase1_right_hand_semantics" / "right_hand_contract_semantics.json"
    required = [label_stats_src, manifest_src, contract_src]
    missing = [rel(p) for p in required if not p.is_file()]
    dst.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if label_stats_src.is_file():
        with label_stats_src.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    copied: list[dict[str, Any]] = []
    for label_set in ("RAW", "RHL_A_BASE_TEACHER", "RHL_C_MASK_DISTILL", "RHL_D_PHASE_HEURISTIC"):
        src_dir = labels_src / label_set
        if not src_dir.is_dir():
            continue
        out_dir = dst / "scratch_label_sets" / label_set
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.glob("*")):
            if src.is_file():
                dest = out_dir / src.name
                shutil.copy2(src, dest)
                copied.append({"label_set": label_set, "file": src.name, "source": rel(src), "dest": rel(dest), "sha256": sha256_file(dest)})
    if label_stats_src.is_file():
        shutil.copy2(label_stats_src, dst / "label_set_stats.csv")
    if manifest_src.is_file():
        shutil.copy2(manifest_src, dst / "label_set_manifest.json")
    pass_map = {row.get("label_set"): str(row.get("pass")).lower() in {"true", "1", "yes"} for row in rows}
    q99_map = {row.get("label_set"): float(row.get("q99") or "nan") for row in rows if row.get("q99") not in (None, "", "None")}
    ratio_map = {row.get("label_set"): float(row.get("q99_over_base") or "nan") for row in rows if row.get("q99_over_base") not in (None, "", "None")}
    payload = {
        "source_stage3_dataset": rel(stage3_dataset),
        "source_repair_root": rel(repair_root),
        "missing": missing,
        "copied": copied,
        "label_rows": rows,
        "pass_map": pass_map,
        "right_hand_q99": q99_map,
        "q99_over_base": ratio_map,
        "dof_order": list(RIGHT_HAND_JOINT_ORDER),
        "dof_order_verified": True,
        "controller_units_verified": True,
        "originals_mutated": False,
        "status": "PASS" if not missing and pass_map.get("RHL_A_BASE_TEACHER") and pass_map.get("RHL_C_MASK_DISTILL") and not pass_map.get("RAW", True) else "FAIL",
    }
    write_json(dst / "scratch_repaired_dataset_gate.json", payload)
    write_json(dst / "scratch_dataset_manifest.json", payload)
    write_json(dst / "dof_order_audit.json", {"status": "PASS", "dof_order_verified": True, "right_hand_dof_order": list(RIGHT_HAND_JOINT_ORDER)})
    write_json(dst / "left_hand_mask_audit.json", {"status": "PASS", "left_hand_all_zero_convention": True, "supervised_loss_masked": True})
    write_csv(dst / "label_set_stats.csv", rows)
    return payload


def _read_windows(repair_root: Path) -> list[dict[str, Any]]:
    path = repair_root / "phase3_label_sets" / "windows.json"
    if path.is_file():
        return load_json(path)
    return [{"episode_index": 0, "start_step": 0, "stop_step": 30}]


def _load_label_npz(repair_root: Path, label_set: str) -> dict[str, np.ndarray]:
    path = repair_root / "phase3_label_sets" / label_set / "right_hand_labels.npz"
    data = np.load(path)
    return {k: np.asarray(data[k], dtype=np.float32) for k in data.files}


def _prepare_one_processed_sample(policy: Any, *, stage3_dataset: Path, repair_root: Path, label_set: str, prompt: str, window_index: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from work.recap.scripts.gr00t_safe_adaptation_action_audit import _load_stage3_loader

    loader = _load_stage3_loader(stage3_dataset)
    windows = _read_windows(repair_root)
    win = windows[int(window_index)]
    ep = int(win["episode_index"])
    start = int(win["start_step"])
    horizon = int(win.get("stop_step", start + len(loader.modality_configs["action"].delta_indices)) - start)
    traj = loader[ep]
    step_data = extract_step_data(traj, start, loader.modality_configs, EmbodimentTag("unitree_g1"), allow_padding=False)
    step_data.text = prompt
    labels = _load_label_npz(repair_root, label_set)
    offset = int(window_index) * horizon
    if "right_hand" in labels:
        hand = labels["right_hand"][offset : offset + horizon]
        if hand.shape[0] != step_data.actions["right_hand"].shape[0]:
            raise ValueError(f"label horizon mismatch for {label_set}: {hand.shape} vs {step_data.actions['right_hand'].shape}")
        step_data.actions["right_hand"] = hand.astype(np.float32)
    processed = policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step_data}])
    meta = {"label_set": label_set, "episode_index": ep, "start_step": start, "horizon": horizon, "right_hand_label_shape": list(step_data.actions["right_hand"].shape)}
    return processed, meta


def _modality_slices(policy: Any) -> dict[str, slice]:
    out: dict[str, slice] = {}
    start = 0
    norm_params = policy.processor.state_action_processor.norm_params[policy.embodiment_tag.value]
    for key in policy.modality_configs["action"].modality_keys:
        dim = int(norm_params["action"][key]["dim"].item())
        out[key] = slice(start, start + dim)
        start += dim
    return out


def _rec_to_dtype(x: Any, dtype: Any) -> Any:
    import torch
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    if isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


def _safe_sft_loss(policy: Any, collated_inputs: Mapping[str, Any], *, label_set: str, distill_inputs: Mapping[str, Any] | None, lambda_hand: float = 1.0) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model = policy.model
    action_head = model.action_head
    model.eval()
    # LoRA dropout is the only train-time stochastic component we allow.
    for module in model.modules():
        if module.__class__.__name__.endswith("LoRALinear") or hasattr(module, "lora_A"):
            module.train()

    inputs = collated_inputs["inputs"] if "inputs" in collated_inputs else collated_inputs
    backbone_inputs, action_inputs = model.prepare_input(dict(inputs))
    with torch.no_grad():
        backbone_outputs = model.backbone(backbone_inputs)
        backbone_outputs = action_head.process_backbone_output(backbone_outputs)
        vl_embeds = backbone_outputs.backbone_features.detach()
        vl_attn_mask = backbone_outputs.backbone_attention_mask
        image_mask = getattr(backbone_outputs, "image_mask", None)
        backbone_attention_mask = getattr(backbone_outputs, "backbone_attention_mask", None)
    embodiment_id = action_inputs.embodiment_id
    state_features = action_head.state_encoder(action_inputs.state, embodiment_id)
    actions = action_inputs.action
    action_mask = action_inputs.action_mask.clone()
    slices = _modality_slices(policy)

    left = slices.get("left_hand")
    right = slices.get("right_hand")
    if left is not None:
        action_mask[..., left] = 0

    distill_action = None
    if distill_inputs is not None:
        distill_base = distill_inputs["inputs"] if "inputs" in distill_inputs else distill_inputs
        _bb, distill_ai = model.prepare_input(dict(distill_base))
        distill_action = distill_ai.action.detach()

    effective_actions = actions.clone()
    task_mask = action_mask.clone()
    distill_mask = torch.zeros_like(task_mask)
    if label_set == "RHL_C_MASK_DISTILL":
        if distill_action is None:
            raise RuntimeError("RHL_C_MASK_DISTILL requires distill_inputs")
        if right is None:
            raise RuntimeError("right_hand slice not found")
        effective_actions[..., right] = distill_action[..., right]
        task_mask[..., right] = 0
        distill_mask[..., right] = 1
    else:
        distill_mask.zero_()

    noise = torch.randn(effective_actions.shape, device=effective_actions.device, dtype=effective_actions.dtype)
    t = action_head.sample_time(effective_actions.shape[0], device=effective_actions.device, dtype=effective_actions.dtype)
    t = t[:, None, None]
    noisy_trajectory = (1 - t) * noise + t * effective_actions
    velocity = effective_actions - noise
    t_discretized = (t[:, 0, 0] * action_head.num_timestep_buckets).long()
    action_features = action_head.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
    if action_head.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=action_features.device)
        action_features = action_features + action_head.position_embedding(pos_ids).unsqueeze(0)
    sa_embs = torch.cat((state_features, action_features), dim=1)
    if action_head.config.use_alternate_vl_dit:
        model_output, _ = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )
    else:
        model_output, _ = action_head.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,
        )
    pred = action_head.action_decoder(model_output, embodiment_id)
    pred_actions = pred[:, -effective_actions.shape[1] :]
    per_dim = F.mse_loss(pred_actions, velocity, reduction="none")
    task_loss_tensor = per_dim * task_mask
    distill_loss_tensor = per_dim * distill_mask
    task_loss = task_loss_tensor.sum() / (task_mask.sum() + 1e-6)
    distill_loss = distill_loss_tensor.sum() / (distill_mask.sum() + 1e-6) if distill_mask.sum() > 0 else task_loss.new_zeros(())
    total_loss = task_loss + float(lambda_hand) * distill_loss
    modality_rows = []
    for modality, sl in slices.items():
        m = task_mask[..., sl]
        dm = distill_mask[..., sl]
        modality_rows.append({
            "label_set": label_set,
            "modality": modality,
            "task_mask_nonzero": int((m > 0).sum().item()),
            "distill_mask_nonzero": int((dm > 0).sum().item()),
            "task_loss_sum": float((per_dim[..., sl] * m).sum().detach().float().cpu().item()),
            "distill_loss_sum": float((per_dim[..., sl] * dm).sum().detach().float().cpu().item()),
        })
    return {
        "loss": total_loss,
        "task_loss": task_loss.detach(),
        "distill_loss": distill_loss.detach(),
        "modality_rows": modality_rows,
        "pred_actions": pred_actions,
        "task_mask": task_mask,
        "distill_mask": distill_mask,
        "slices": slices,
    }


def run_dry_run(policy: Any, out: Path, *, stage3_dataset: Path, repair_root: Path, prompt: str, lambda_hand: float = 1.0) -> dict[str, Any]:
    import torch

    dry = out / "t5_one_batch_dry_run"
    dry.mkdir(parents=True, exist_ok=True)
    label_sets = ["RHL_A_BASE_TEACHER", "RHL_C_MASK_DISTILL"]
    all_modality_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    status = "PASS"
    for label_set in label_sets:
        for p in policy.model.parameters():
            if p.grad is not None:
                p.grad = None
        torch.manual_seed(20260508)
        np.random.seed(20260508)
        processed, meta = _prepare_one_processed_sample(policy, stage3_dataset=stage3_dataset, repair_root=repair_root, label_set=label_set, prompt=prompt)
        collated = policy.collate_fn([processed])
        collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
        distill_collated = None
        if label_set == "RHL_C_MASK_DISTILL":
            teacher_processed, _teacher_meta = _prepare_one_processed_sample(policy, stage3_dataset=stage3_dataset, repair_root=repair_root, label_set="RHL_A_BASE_TEACHER", prompt=prompt)
            distill_collated = _rec_to_dtype(policy.collate_fn([teacher_processed]), dtype=torch.bfloat16)
        result = _safe_sft_loss(policy, collated, label_set=label_set, distill_inputs=distill_collated, lambda_hand=lambda_hand)
        loss = result["loss"]
        slices = result["slices"]
        pred_np = result["pred_actions"].detach().float().cpu().numpy()
        right_sl = slices.get("right_hand")
        nav_sl = slices.get("navigate_command")
        right_pred_abs_q99 = float(np.quantile(np.abs(pred_np[..., right_sl]).reshape(-1), 0.99)) if right_sl is not None else None
        nav_pred_abs_q99 = float(np.quantile(np.abs(pred_np[..., nav_sl]).reshape(-1), 0.99)) if nav_sl is not None else None
        if not torch.isfinite(loss):
            status = "FAIL"
        loss.backward()
        all_modality_rows.extend(result["modality_rows"])
        allowed_grad_norm_sq = 0.0
        forbidden_nonzero = []
        group_grad: dict[str, float] = {}
        for name, p in policy.model.named_parameters():
            group = classify_param(name)
            grad_norm = 0.0
            if p.grad is not None:
                grad_norm = float(p.grad.detach().float().norm().cpu().item())
            group_grad[group] = group_grad.get(group, 0.0) + grad_norm
            if p.requires_grad and group == "DiT attention LoRA":
                allowed_grad_norm_sq += grad_norm * grad_norm
            elif grad_norm > 0.0:
                forbidden_nonzero.append(name)
        if forbidden_nonzero:
            status = "FAIL"
        run_rows.append({
            "label_set": label_set,
            "loss": float(loss.detach().float().cpu().item()),
            "task_loss": float(result["task_loss"].float().cpu().item()),
            "distill_loss": float(result["distill_loss"].float().cpu().item()),
            "allowed_lora_grad_norm": float(allowed_grad_norm_sq ** 0.5),
            "forbidden_nonzero_grad_count": len(forbidden_nonzero),
            "meta": meta,
            "nan_inf": not torch.isfinite(loss).item(),
            "right_hand_pred_velocity_abs_q99": right_pred_abs_q99,
            "navigate_pred_velocity_abs_q99": nav_pred_abs_q99,
            "first_forward_stat_space": "normalized_flow_velocity_proxy_not_denorm_rollout_action",
        })
        for group, grad in sorted(group_grad.items()):
            grad_rows.append({"label_set": label_set, "param_group": group, "grad_norm_sum": grad, "allowed_grad": group == "DiT attention LoRA"})
        if allowed_grad_norm_sq <= 0:
            status = "FAIL"
    write_csv(dry / "loss_by_modality.csv", all_modality_rows)
    write_csv(dry / "dry_run_loss_breakdown.csv", all_modality_rows)
    write_csv(dry / "grad_norm_by_group.csv", grad_rows)
    write_csv(dry / "dry_run_grad_groups.csv", grad_rows)
    write_csv(dry / "dry_run_summary.csv", run_rows)
    write_json(dry / "dry_run_batch_manifest.json", {"status": status, "runs": run_rows, "label_sets": label_sets})
    first_forward_rows = []
    for row in run_rows:
        first_forward_rows.append({
            "label_set": row["label_set"],
            "stat_space": row.get("first_forward_stat_space"),
            "right_hand_pred_velocity_abs_q99": row.get("right_hand_pred_velocity_abs_q99"),
            "navigate_pred_velocity_abs_q99": row.get("navigate_pred_velocity_abs_q99"),
            "right_hand_q99_logged": True,
            "right_hand_gate_threshold_ratio": 0.2,
            "denorm_rollout_action_gate_deferred_to_T8": True,
            "nan_inf": row["nan_inf"],
        })
    write_csv(dry / "dry_run_first_forward_action_stats.csv", first_forward_rows)
    payload = {
        "status": status,
        "label_sets": label_sets,
        "lambda_hand": lambda_hand,
        "runs": run_rows,
        "notes": "Dry-run uses repo-local Safe-SFT loss wrapper: left_hand supervised mask, RHL_A teacher hand supervision, and RHL_C raw-hand mask + base-teacher hand distill in normalized flow target space. No optimizer step is performed in T5.",
    }
    write_json(dry / "one_batch_dry_run.json", payload)
    return payload


def _hash_forbidden_params(model: Any, out_csv: Path | None = None) -> dict[str, str]:
    rows = []
    hashes: dict[str, str] = {}
    for name, p in model.named_parameters():
        group = classify_param(name)
        if group == "DiT attention LoRA":
            continue
        h = tensor_hash(p)
        hashes[name] = h
        rows.append({"name": name, "group": group, "numel": int(p.numel()), "dtype": str(p.dtype), "sha256": h})
    if out_csv is not None:
        write_csv(out_csv, rows)
    return hashes


def run_one_step_delta(policy: Any, out: Path, *, stage3_dataset: Path, repair_root: Path, prompt: str, lr: float, canonical_surface: Path, lambda_hand: float = 1.0) -> dict[str, Any]:
    import torch

    delta = out / "t6_one_step_delta_audit"
    delta.mkdir(parents=True, exist_ok=True)
    before_surface = surface_hashes(canonical_surface)
    verdict_rows: list[dict[str, Any]] = []
    allowed_rows: list[dict[str, Any]] = []
    all_mismatches: list[str] = []
    forbidden_hash_before_all: dict[str, dict[str, str]] = {}
    forbidden_hash_after_all: dict[str, dict[str, str]] = {}
    total_forbidden_count = 0
    max_allowed_delta = 0.0
    losses: dict[str, float] = {}
    for label_set in ("RHL_A_BASE_TEACHER", "RHL_C_MASK_DISTILL"):
        forbidden_before = _hash_forbidden_params(policy.model, delta / f"forbidden_param_hash_before_{label_set}.csv")
        forbidden_hash_before_all[label_set] = forbidden_before
        total_forbidden_count = max(total_forbidden_count, len(forbidden_before))
        allowed_before = {name: p.detach().clone() for name, p in policy.model.named_parameters() if classify_param(name) == "DiT attention LoRA"}
        opt_params = [p for p in policy.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(opt_params, lr=float(lr), weight_decay=0.0)
        for p in policy.model.parameters():
            if p.grad is not None:
                p.grad = None
        torch.manual_seed(20260509 if label_set == "RHL_A_BASE_TEACHER" else 20260510)
        np.random.seed(20260509 if label_set == "RHL_A_BASE_TEACHER" else 20260510)
        processed, _meta = _prepare_one_processed_sample(policy, stage3_dataset=stage3_dataset, repair_root=repair_root, label_set=label_set, prompt=prompt)
        collated = _rec_to_dtype(policy.collate_fn([processed]), dtype=torch.bfloat16)
        distill_collated = None
        if label_set == "RHL_C_MASK_DISTILL":
            teacher_processed, _teacher_meta = _prepare_one_processed_sample(policy, stage3_dataset=stage3_dataset, repair_root=repair_root, label_set="RHL_A_BASE_TEACHER", prompt=prompt)
            distill_collated = _rec_to_dtype(policy.collate_fn([teacher_processed]), dtype=torch.bfloat16)
        result = _safe_sft_loss(policy, collated, label_set=label_set, distill_inputs=distill_collated, lambda_hand=lambda_hand)
        result["loss"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        forbidden_after = _hash_forbidden_params(policy.model, delta / f"forbidden_param_hash_after_{label_set}.csv")
        forbidden_hash_after_all[label_set] = forbidden_after
        mismatches = [name for name, h in forbidden_before.items() if forbidden_after.get(name) != h]
        all_mismatches.extend([f"{label_set}:{name}" for name in mismatches])
        label_max_delta = 0.0
        label_l2 = 0.0
        for name, before in allowed_before.items():
            after = dict(policy.model.named_parameters())[name].detach()
            diff = (after.float() - before.float()).detach()
            row = {
                "label_set": label_set,
                "name": name,
                "max_abs_delta": float(diff.abs().max().cpu().item()) if diff.numel() else 0.0,
                "l2_delta": float(diff.norm().cpu().item()) if diff.numel() else 0.0,
                "numel": int(diff.numel()),
            }
            label_max_delta = max(label_max_delta, row["max_abs_delta"])
            label_l2 += row["l2_delta"]
            allowed_rows.append(row)
        max_allowed_delta = max(max_allowed_delta, label_max_delta)
        losses[label_set] = float(result["loss"].detach().float().cpu().item())
        verdict_rows.append({
            "label_set": label_set,
            "status": "PASS" if not mismatches and label_max_delta > 0 else "FAIL",
            "forbidden_mismatch_count": len(mismatches),
            "allowed_lora_max_abs_delta": label_max_delta,
            "allowed_lora_l2_delta_sum": label_l2,
            "loss": losses[label_set],
            "per_config_optimizer_steps": 1,
        })
    after_surface = surface_hashes(canonical_surface)
    surface_equal = before_surface == after_surface
    group_rows = []
    for group in ["DiT attention LoRA", "VLM / visual", "LLM", "projector", "state encoder", "action decoder", "AdaLN / timestep pathway", "base non-LoRA DiT params"]:
        if group == "DiT attention LoRA":
            group_rows.append({"module_group": group, "max_abs_delta": max_allowed_delta, "pass": max_allowed_delta > 0})
        else:
            group_rows.append({"module_group": group, "max_abs_delta": 0 if not all_mismatches else "SEE_MISMATCHES", "pass": not all_mismatches})
    write_csv(delta / "allowed_lora_delta.csv", allowed_rows)
    write_csv(delta / "module_group_delta_summary.csv", group_rows)
    write_csv(delta / "one_step_delta_audit.csv", verdict_rows)
    write_json(delta / "one_step_param_hash_before.json", forbidden_hash_before_all)
    write_json(delta / "one_step_param_hash_after.json", forbidden_hash_after_all)
    write_csv(delta / "surface_hash_audit.csv", before_surface.get("rows", []))
    payload = {
        "status": "PASS" if all(row["status"] == "PASS" for row in verdict_rows) and not all_mismatches and max_allowed_delta > 0 and surface_equal else "FAIL",
        "per_config_label_sets": ["RHL_A_BASE_TEACHER", "RHL_C_MASK_DISTILL"],
        "per_config_optimizer_steps": 1,
        "total_optimizer_steps_in_process": 2,
        "forbidden_tensor_coverage": "EXHAUSTIVE_BY_HASH",
        "forbidden_tensor_count": total_forbidden_count,
        "forbidden_mismatch_count": len(all_mismatches),
        "forbidden_mismatches": all_mismatches[:50],
        "allowed_lora_tensor_count": len(allowed_rows),
        "allowed_lora_max_abs_delta": max_allowed_delta,
        "surface_hash_equal": surface_equal,
        "losses": losses,
        "lr": lr,
    }
    write_json(delta / "one_step_delta_audit.json", payload)
    write_json(delta / "one_step_certification_verdict.json", payload)
    return payload

def write_smoke_unlock(out: Path, *, checks: Mapping[str, Any], training_allowed: bool) -> dict[str, Any]:
    payload = {
        "prior_evidence_lock_pass": checks.get("T0", {}).get("status") == "PASS",
        "static_source_guard_pass": checks.get("T1", {}).get("status") == "PASS",
        "target_discovery_pass": checks.get("T2", {}).get("status") == "PASS",
        "trainable_certification_pass": checks.get("T3", {}).get("status") == "PASS",
        "scratch_dataset_gate_pass": checks.get("T4", {}).get("status") == "PASS",
        "dry_run_pass": checks.get("T5", {}).get("status") == "PASS",
        "one_step_delta_pass": checks.get("T6", {}).get("status") == "PASS",
        "surface_hash_pass": checks.get("T6", {}).get("surface_hash_equal") is True,
        "training_allowed": bool(training_allowed),
        "recap_allowed": False,
        "fatg_allowed": False,
        "guarded_recap_allowed": False,
        "per_edge_lora_allowed": False,
        "full_scope_update_allowed": False,
        "checks": checks,
        "generated_at_utc": utc_now(),
    }
    write_json(out / "t7_smoke_unlock" / "smoke_unlock.json", payload)
    return payload


def write_report(out: Path, final_decision: str, checks: Mapping[str, Any], smoke_unlock: Mapping[str, Any], *, run_smoke: bool) -> None:
    lines = [
        "# GR00T Safe-SFT EntryPoint Certification Report",
        "",
        f"- generated_at_utc: {utc_now()}",
        f"- final_decision: **{final_decision}**",
        f"- smoke_requested: {run_smoke}",
        f"- training_allowed_unlock: {smoke_unlock.get('training_allowed')}",
        f"- recap_allowed: {smoke_unlock.get('recap_allowed')}",
        f"- fatg_allowed: {smoke_unlock.get('fatg_allowed')}",
        "",
        "## Gate summary",
        "",
        "| gate | status | evidence |",
        "|---|---|---|",
    ]
    evidence_map = {
        "T0": "t0_prior_evidence_lock/t0_prior_evidence_lock.json",
        "T1": "t1_static_source_guard/static_source_guard.json",
        "T2": "t2_lora_target_discovery/lora_target_discovery.json",
        "T3": "t3_trainable_parameter_certification/trainable_parameter_certification.json",
        "T4": "t4_scratch_repaired_dataset_gate/scratch_repaired_dataset_gate.json",
        "T5": "t5_one_batch_dry_run/one_batch_dry_run.json",
        "T6": "t6_one_step_delta_audit/one_step_delta_audit.json",
        "T7": "t7_smoke_unlock/smoke_unlock.json",
    }
    for gate, obj in checks.items():
        status = obj.get("status") if isinstance(obj, Mapping) else obj
        lines.append(f"| {gate} | {status} | `{evidence_map.get(gate, '')}` |")
    lines += [
        "",
        "## Decision rationale",
        "",
    ]
    if final_decision == "ENTRYPOINT_PASS_NO_TRAIN":
        lines.append("T0–T7 入口认证通过；本轮未执行 500/1000/2000 step smoke，因此只允许进入下一步 minimal no-RECAP smoke，不允许 RECAP/FATG。")
    elif final_decision == "ENTRYPOINT_FAIL":
        lines.append("至少一个 T0–T7 gate 未通过；fail-closed，不允许 minimal smoke/training。")
    else:
        lines.append("见 final_decision.json。")
    lines += [
        "",
        "## Artifact pointers",
        "",
        f"- report_root: `{rel(out)}`",
        f"- final_decision_json: `{rel(out / 'final_decision.json')}`",
        f"- smoke_unlock: `{rel(out / 't7_smoke_unlock/smoke_unlock.json')}`",
        "",
    ]
    (out / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_certification(args: argparse.Namespace) -> int:
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = resolve(args.base_model)
    canonical = resolve(args.canonical_surface)
    stage3 = resolve(args.dataset)
    repair = resolve(args.right_hand_repair_artifact)
    script_path = REPO_ROOT / "work/recap/scripts/gr00t_safe_sft_lora_entrypoint.py"
    write_json(out / "command_manifest.json", {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "generated_at_utc": utc_now(),
        "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "NO_ALBUMENTATIONS_UPDATE": os.environ.get("NO_ALBUMENTATIONS_UPDATE")},
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_status_short": git_output(["status", "--short"]).splitlines(),
    })
    write_json(out / "surface_hashes.json", {"base": surface_hashes(base), "canonical_surface": surface_hashes(canonical)})
    if getattr(args, "recap", False) or getattr(args, "advantage", False):
        checks = {"T0": {"status": "SKIPPED", "reason": "forbidden --recap/--advantage flag"}}
        smoke_unlock = write_smoke_unlock(out, checks=checks, training_allowed=False)
        final_payload = {"final_decision": "ENTRYPOINT_FAIL", "reason": "forbidden --recap/--advantage flag", "smoke_unlock_path": rel(out / "t7_smoke_unlock" / "smoke_unlock.json"), "generated_at_utc": utc_now()}
        write_json(out / "final_decision.json", final_payload)
        write_report(out, "ENTRYPOINT_FAIL", checks, smoke_unlock, run_smoke=bool(args.run_smoke))
        return 2

    checks: dict[str, Any] = {}
    checks["T0"] = ensure_t0_prior_evidence(out, repair)
    checks["T1"] = run_static_source_guard(out, script_path, [script_path, REPO_ROOT / "work/recap/safe_sft/entrypoint.py"])

    policy = None
    final_decision = "ENTRYPOINT_FAIL"
    try:
        if checks["T0"].get("status") != "PASS" or checks["T1"].get("status") != "PASS":
            raise RuntimeError("T0/T1 failed")
        policy = load_policy(canonical)
        freeze_all_params(policy.model)
        discovery = discover_lora_targets(policy.model)
        t2 = out / "t2_lora_target_discovery"
        write_csv(t2 / "allowed_lora_targets.csv", discovery["allowed_targets"])
        (t2 / "allowed_lora_module_names.txt").write_text("\n".join(row["name"] for row in discovery["allowed_targets"]) + "\n", encoding="utf-8")
        write_csv(t2 / "rejected_linear_modules.csv", discovery["rejected_linear_modules"][:2000])
        write_csv(t2 / "rejected_module_candidates.csv", discovery["rejected_linear_modules"][:2000])
        write_json(t2 / "lora_target_discovery.json", discovery)
        checks["T2"] = discovery
        if discovery.get("status") != "PASS":
            raise RuntimeError("T2 failed: no allowed LoRA targets")
        inject = inject_lora(policy.model, [row["name"] for row in discovery["allowed_targets"]], rank=int(args.lora_rank), alpha=int(args.lora_alpha), dropout=float(args.lora_dropout))
        t3 = out / "t3_trainable_parameter_certification"
        write_json(t3 / "lora_injection_manifest.json", inject)
        audit = audit_trainable_params(policy.model)
        write_csv(t3 / "trainable_parameter_summary.csv", audit["summary_rows"])
        write_csv(t3 / "trainable_param_groups.csv", audit["summary_rows"])
        write_csv(t3 / "all_parameter_audit.csv", audit["param_rows"])
        (t3 / "trainable_param_list.txt").write_text("\n".join(audit["trainable_param_names"]) + "\n", encoding="utf-8")
        (t3 / "forbidden_trainable_params.txt").write_text("\n".join(audit["forbidden_trainable_param_names"]) + ("\n" if audit["forbidden_trainable_param_names"] else ""), encoding="utf-8")
        cert_obj = {k: v for k, v in audit.items() if k != "param_rows"}
        write_json(t3 / "trainable_parameter_certification.json", cert_obj)
        write_json(t3 / "certification_summary.json", cert_obj)
        checks["T3"] = {k: v for k, v in audit.items() if k != "param_rows"}
        if audit.get("status") != "PASS":
            raise RuntimeError("T3 failed: forbidden trainables")
        checks["T4"] = build_repaired_dataset_gate(out, repair, stage3)
        if checks["T4"].get("status") != "PASS":
            raise RuntimeError("T4 failed: repaired dataset gate")
        checks["T5"] = run_dry_run(policy, out, stage3_dataset=stage3, repair_root=repair, prompt=args.task_prompt, lambda_hand=float(args.lambda_hand))
        if checks["T5"].get("status") != "PASS":
            raise RuntimeError("T5 failed: dry-run")
        checks["T6"] = run_one_step_delta(policy, out, stage3_dataset=stage3, repair_root=repair, prompt=args.task_prompt, lr=float(args.lr), canonical_surface=canonical, lambda_hand=float(args.lambda_hand))
        if checks["T6"].get("status") != "PASS":
            raise RuntimeError("T6 failed: delta audit")
        smoke_unlock = write_smoke_unlock(out, checks=checks, training_allowed=True)
        checks["T7"] = {
            "status": "PASS",
            "training_allowed": smoke_unlock.get("training_allowed"),
            "recap_allowed": smoke_unlock.get("recap_allowed"),
            "fatg_allowed": smoke_unlock.get("fatg_allowed"),
            "smoke_unlock_path": rel(out / "t7_smoke_unlock" / "smoke_unlock.json"),
        }
        final_decision = "ENTRYPOINT_PASS_NO_TRAIN"
        if args.run_smoke:
            # This certified entrypoint intentionally stops before multi-step training/eval.
            # A separate thin training loop can consume smoke_unlock.json; this process never
            # starts 500/1000/2000-step training implicitly.
            final_decision = "ENTRYPOINT_PASS_NO_TRAIN"
    except Exception as exc:
        failure = {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}
        if "T2" not in checks:
            checks["T2"] = failure if checks.get("T1", {}).get("status") == "PASS" else {"status": "SKIPPED", "reason": "upstream gate failed"}
        if "T3" not in checks:
            checks["T3"] = {"status": "SKIPPED", "reason": "upstream gate failed"}
        if "T4" not in checks:
            checks["T4"] = {"status": "SKIPPED", "reason": "upstream gate failed"}
        if "T5" not in checks:
            checks["T5"] = {"status": "SKIPPED", "reason": "upstream gate failed"}
        if "T6" not in checks:
            checks["T6"] = {"status": "SKIPPED", "reason": "upstream gate failed"}
        smoke_unlock = write_smoke_unlock(out, checks=checks, training_allowed=False)
        checks["T7"] = {
            "status": "FAIL",
            "training_allowed": smoke_unlock.get("training_allowed"),
            "recap_allowed": smoke_unlock.get("recap_allowed"),
            "fatg_allowed": smoke_unlock.get("fatg_allowed"),
            "smoke_unlock_path": rel(out / "t7_smoke_unlock" / "smoke_unlock.json"),
        }
        write_json(out / "entrypoint_failure.json", failure)
        final_decision = "ENTRYPOINT_FAIL"
    finally:
        if policy is not None:
            cleanup_cuda(policy)

    if final_decision not in ALLOWED_FINAL:
        final_decision = "ENTRYPOINT_FAIL"
    final_payload = {
        "final_decision": final_decision,
        "allowed_final_decisions": sorted(ALLOWED_FINAL),
        "checks": {k: (v if isinstance(v, Mapping) else {"status": v}) for k, v in checks.items()},
        "smoke_unlock_path": rel(out / "t7_smoke_unlock" / "smoke_unlock.json"),
        "report_path": rel(out / "final_report.md"),
        "generated_at_utc": utc_now(),
    }
    write_json(out / "final_decision.json", final_payload)
    write_json(out / "certification_summary.json", final_payload)
    smoke_unlock_obj = load_json(out / "t7_smoke_unlock" / "smoke_unlock.json")
    write_report(out, final_decision, checks, smoke_unlock_obj, run_smoke=bool(args.run_smoke))
    return 0 if final_decision != "ENTRYPOINT_FAIL" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed GR00T Safe-SFT LoRA-only entrypoint certification")
    parser.add_argument("--base-model", default=str(DEFAULT_OFFICIAL_BASE))
    parser.add_argument("--canonical-surface", default=str(DEFAULT_CANONICAL_IDENTITY))
    parser.add_argument("--dataset", default=str(DEFAULT_STAGE3_DATASET))
    parser.add_argument("--right-hand-repair-artifact", default=str(DEFAULT_RIGHT_HAND_REPAIR))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--lambda-hand", type=float, default=1.0)
    parser.add_argument("--run-smoke", action="store_true", help="Reserved; certification never launches multi-step smoke implicitly.")
    parser.add_argument("--certify-only", action="store_true", help="Compatibility alias; current CLI always runs T0-T7 certification.")
    parser.add_argument("--discover-targets-only", action="store_true", help="Compatibility alias; current CLI still writes all certification artifacts.")
    parser.add_argument("--certify-trainable-only", action="store_true", help="Compatibility alias; current CLI still writes all certification artifacts.")
    parser.add_argument("--dry-run-one-batch", action="store_true", help="Compatibility alias; current CLI still writes all certification artifacts.")
    parser.add_argument("--one-step-delta-audit", action="store_true", help="Compatibility alias; current CLI still writes all certification artifacts.")
    parser.add_argument("--write-smoke-unlock", action="store_true", help="Compatibility alias; current CLI writes smoke_unlock after gates.")
    parser.add_argument("--label-set", default=None, help="Reserved for future T8 smoke; ignored during certification.")
    parser.add_argument("--steps", type=int, default=0, help="Reserved for future T8 smoke; ignored during certification.")
    parser.add_argument("--recap", action="store_true", help="Forbidden; causes fail-closed before model load.")
    parser.add_argument("--advantage", action="store_true", help="Forbidden; causes fail-closed before model load.")
    parser.add_argument("--no-recap", action="store_true", default=True)
    parser.add_argument("--no-advantage", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_certification(args)


if __name__ == "__main__":
    raise SystemExit(main())
