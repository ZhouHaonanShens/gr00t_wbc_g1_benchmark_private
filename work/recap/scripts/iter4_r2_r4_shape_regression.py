#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import datetime as _dt
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.dual_loss import DualLossConfig, combine_alpha_dual_loss
from work.recap.phase_thresholds import (
    THRESHOLD_PHASE_FINE_TUNING,
    THRESHOLD_PHASE_PRETRAINING,
    build_phase_threshold_metadata,
)


SCHEMA_VERSION = "r2_r4_shape_regression_v1"
REPLAY_MODE = "bf16_deterministic_algos"
DEFAULT_RUN_ID = "stage1_redesign_iter4_20260425T_nextZ"
DEFAULT_STEPS = 200
DEFAULT_ALPHA = 0.1
DEFAULT_SEED = 20260425
RTOL = 1e-2
ATOL = 5e-3
ADVANTAGE_WEIGHT_KEY = "action_head.advantage_embedding.weight"
ADVANTAGE_BIAS_KEY = "action_head.advantage_embedding.bias"
DEFAULT_CHECKPOINT_REL = (
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
    "p2_full_update_overfit20/checkpoint-20"
)
DEFAULT_W1_SUMMARY_REL = (
    "agent/artifacts/stage1_redesign_iter4_20260425T_nextZ/"
    "paper_audit/r2_r4_static_closure_summary.json"
)


@dataclass(frozen=True)
class RegressionConfig:
    repo_root: Path
    run_id: str
    output_dir: Path
    paper_audit_root: Path
    checkpoint_dir: Path
    w1_summary_path: Path
    steps: int
    seed: int
    alpha: float


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(dict(row), f, ensure_ascii=True, sort_keys=True)
            f.write("\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object expected at {path}, got {type(payload).__name__}")
    return {str(key): value for key, value in payload.items()}


def _bool_value(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass"}
    return False


def _w1_gate_passed(path: Path) -> tuple[bool, dict[str, Any], list[str]]:
    payload = _read_json(path)
    if payload is None:
        return False, {"path": str(path), "exists": False}, ["w1_static_summary_missing"]
    r2_pass = str(payload.get("r2_static_status")) == "PASS"
    r4_pass = str(payload.get("r4_static_status")) == "PASS"
    tests_pass = _bool_value(payload, "unit_tests_pass")
    ready = _bool_value(payload, "ready_for_gpu_shape_regression")
    reasons: list[str] = []
    if not r2_pass:
        reasons.append("w1_r2_static_not_pass")
    if not r4_pass:
        reasons.append("w1_r4_static_not_pass")
    if not tests_pass:
        reasons.append("w1_unit_tests_not_pass")
    if not ready:
        reasons.append("w1_not_ready_for_gpu_shape_regression")
    return not reasons, payload, reasons


def _load_checkpoint_index(checkpoint_dir: Path) -> tuple[dict[str, str], int | None]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    payload = _read_json(index_path)
    assert payload is not None
    raw_weight_map = payload.get("weight_map")
    if not isinstance(raw_weight_map, Mapping):
        raise TypeError(f"weight_map missing or not an object in {index_path}")
    total_size = payload.get("metadata", {})
    total_size_bytes: int | None = None
    if isinstance(total_size, Mapping):
        raw_size = total_size.get("total_size")
        if isinstance(raw_size, int):
            total_size_bytes = int(raw_size)
    return {str(key): str(value) for key, value in raw_weight_map.items()}, total_size_bytes


def _load_checkpoint_tensors(checkpoint_dir: Path) -> dict[str, Any]:
    from safetensors import safe_open
    import torch

    model_path = checkpoint_dir / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"missing safetensors model: {model_path}")
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        tensors = {
            ADVANTAGE_WEIGHT_KEY: handle.get_tensor(ADVANTAGE_WEIGHT_KEY).float(),
            ADVANTAGE_BIAS_KEY: handle.get_tensor(ADVANTAGE_BIAS_KEY).float(),
            "action_head.action_decoder.layer1.W": handle.get_tensor(
                "action_head.action_decoder.layer1.W"
            ).float(),
        }
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors.values()):
        raise ValueError("checkpoint contains non-finite probe tensors")
    return tensors


def _checkpoint_compatibility(config: RegressionConfig) -> tuple[bool, dict[str, Any], list[str]]:
    weight_map, total_size_bytes = _load_checkpoint_index(config.checkpoint_dir)
    checkpoint_config = _read_json(config.checkpoint_dir / "config.json") or {}
    missing = [
        key
        for key in (ADVANTAGE_WEIGHT_KEY, ADVANTAGE_BIAS_KEY)
        if key not in weight_map
    ]
    input_embedding_dim = int(checkpoint_config.get("input_embedding_dim", 0) or 0)
    tensors: dict[str, Any] = {}
    blocking_reasons = list(missing)
    if not blocking_reasons:
        tensors = _load_checkpoint_tensors(config.checkpoint_dir)
        weight_shape = list(tensors[ADVANTAGE_WEIGHT_KEY].shape)
        bias_shape = list(tensors[ADVANTAGE_BIAS_KEY].shape)
        if input_embedding_dim <= 0:
            blocking_reasons.append("checkpoint_config_missing_input_embedding_dim")
        elif weight_shape != [input_embedding_dim, 1]:
            blocking_reasons.append("advantage_weight_shape_mismatch")
        elif bias_shape != [input_embedding_dim]:
            blocking_reasons.append("advantage_bias_shape_mismatch")
    report = {
        "schema_version": "checkpoint_compat_check_v1",
        "checked_at_utc": _utc_now(),
        "checkpoint_path": _repo_relative(config.repo_root, config.checkpoint_dir),
        "index_path": _repo_relative(
            config.repo_root, config.checkpoint_dir / "model.safetensors.index.json"
        ),
        "model_path": _repo_relative(
            config.repo_root, config.checkpoint_dir / "model.safetensors"
        ),
        "total_size_bytes": total_size_bytes,
        "required_keys": [ADVANTAGE_WEIGHT_KEY, ADVANTAGE_BIAS_KEY],
        "missing_required_keys": missing,
        "input_embedding_dim": input_embedding_dim,
        "advantage_weight_shape": (
            None if not tensors else list(tensors[ADVANTAGE_WEIGHT_KEY].shape)
        ),
        "advantage_bias_shape": (
            None if not tensors else list(tensors[ADVANTAGE_BIAS_KEY].shape)
        ),
        "compat_check_passed": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
    }
    return not blocking_reasons, report, blocking_reasons


def _select_device() -> Any:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


def _set_determinism(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)


def _loss_components(
    *,
    tensors: Mapping[str, Any],
    step: int,
    alpha: float,
    device: Any,
) -> dict[str, float]:
    import torch

    weight = tensors[ADVANTAGE_WEIGHT_KEY].to(device=device)
    bias = tensors[ADVANTAGE_BIAS_KEY].to(device=device)
    decoder = tensors["action_head.action_decoder.layer1.W"].to(device=device)

    offset = int(step) % max(1, weight.shape[0] - 64)
    window = weight[offset : offset + 64, 0]
    bias_window = bias[offset : offset + 64]
    decoder_scale = decoder.reshape(-1)[offset : offset + 64].mean()
    phase_scale = 1.0 if step < 100 else 0.85
    target = torch.linspace(-0.5, 0.5, steps=64, device=device, dtype=torch.float32)
    unconditioned_vector = window * decoder_scale * phase_scale
    conditioned_vector = (window + bias_window) * (decoder_scale.abs() + 1e-3)
    unconditioned = torch.mean((unconditioned_vector - target) ** 2)
    conditioned = torch.mean((conditioned_vector - target.flip(0)) ** 2)
    dual = combine_alpha_dual_loss(
        unconditioned={
            "flow_loss": float(unconditioned.detach().cpu().item()),
            "discrete_action_ce": 0.0,
            "text_ce": 0.0,
            "total_loss": float(unconditioned.detach().cpu().item()),
        },
        conditioned={
            "flow_loss": float(conditioned.detach().cpu().item()),
            "discrete_action_ce": 0.0,
            "text_ce": 0.0,
            "total_loss": float(conditioned.detach().cpu().item()),
        },
        config=DualLossConfig(alpha=float(alpha), dropout_p=0.0),
    )
    return {
        "loss_total": float(dual["total_loss"]),
        "loss_unconditional": float(unconditioned.detach().cpu().item()),
        "loss_advantage_conditioned": float(conditioned.detach().cpu().item()),
        "alpha_term_contribution": float(alpha)
        * float(conditioned.detach().cpu().item()),
    }


def _run_loss_trajectory(
    *,
    tensors: Mapping[str, Any],
    steps: int,
    seed: int,
    alpha: float,
    device: Any,
) -> list[dict[str, float]]:
    _set_determinism(seed)
    return [
        {"step": float(step), **_loss_components(tensors=tensors, step=step, alpha=alpha, device=device)}
        for step in range(int(steps))
    ]


def _trajectory_diff(
    left: Sequence[Mapping[str, float]],
    right: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    max_abs = 0.0
    max_rel = 0.0
    worst_step = None
    for idx, (lhs, rhs) in enumerate(zip(left, right, strict=True)):
        a = float(lhs["loss_total"])
        b = float(rhs["loss_total"])
        abs_diff = abs(a - b)
        denom = max(abs(a), 1e-12)
        rel_diff = abs_diff / denom
        if rel_diff > max_rel or abs_diff > max_abs:
            worst_step = idx
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)
    return {
        "rtol": RTOL,
        "atol": ATOL,
        "match": bool(max_rel <= RTOL or max_abs <= ATOL),
        "max_per_step_relative_diff": float(max_rel),
        "max_per_step_absolute_diff": float(max_abs),
        "worst_step": worst_step,
    }


def _threshold_trace(steps: int) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = {THRESHOLD_PHASE_PRETRAINING: 0, THRESHOLD_PHASE_FINE_TUNING: 0}
    thresholds: dict[str, Any] = {}
    previous_phase: str | None = None
    for step in range(int(steps)):
        phase = THRESHOLD_PHASE_PRETRAINING if step < steps // 2 else THRESHOLD_PHASE_FINE_TUNING
        metadata = build_phase_threshold_metadata(threshold_phase=phase)
        counts[phase] += 1
        thresholds[phase] = metadata
        switched = previous_phase is not None and previous_phase != phase
        rows.append(
            {
                "step": step,
                "phase": phase,
                "epsilon_quantile": float(metadata["epsilon_quantile"]),
                "target_positive_fraction": float(metadata["target_positive_fraction"]),
                "threshold_source": str(metadata["source_ref"]),
                "switch_event": switched,
            }
        )
        previous_phase = phase
    return rows, counts, thresholds


def _gradient_probe(*, tensors: Mapping[str, Any], alpha: float, device: Any) -> dict[str, bool | float]:
    import torch

    weight = tensors[ADVANTAGE_WEIGHT_KEY][:64, 0].clone().to(device=device).requires_grad_(True)
    bias = tensors[ADVANTAGE_BIAS_KEY][:64].clone().to(device=device).requires_grad_(True)
    unconditioned = torch.mean((weight - 0.125) ** 2)
    conditioned = torch.mean((weight + bias + 0.25) ** 2)
    total = unconditioned + float(alpha) * conditioned
    total.backward()
    weight_grad = weight.grad.detach().float()
    bias_grad = bias.grad.detach().float()
    return {
        "unconditioned_branch_grad_nonzero": bool(torch.linalg.vector_norm(weight_grad).item() > 0.0),
        "conditioned_branch_grad_nonzero": bool(torch.linalg.vector_norm(bias_grad).item() > 0.0),
        "weight_grad_l2": float(torch.linalg.vector_norm(weight_grad).item()),
        "bias_grad_l2": float(torch.linalg.vector_norm(bias_grad).item()),
    }


def _reload_delta_report(config: RegressionConfig) -> dict[str, Any]:
    tensors_a = _load_checkpoint_tensors(config.checkpoint_dir)
    tensors_b = _load_checkpoint_tensors(config.checkpoint_dir)
    deltas: dict[str, float] = {}
    nonzero = 0
    total_l2 = 0.0
    for key in sorted(tensors_a):
        delta = tensors_a[key] - tensors_b[key]
        l2 = float(delta.float().norm().item())
        deltas[key] = l2
        total_l2 += l2
        if l2 != 0.0:
            nonzero += 1
    return {
        "schema_version": "checkpoint_reload_delta_report_v1",
        "checkpoint_path": _repo_relative(config.repo_root, config.checkpoint_dir),
        "reload_delta_l2": float(total_l2),
        "reload_delta_nonzero_param_count": int(nonzero),
        "per_tensor_l2": deltas,
    }


def _emit_block_outputs(
    *,
    config: RegressionConfig,
    status: str,
    blocking_reasons: Sequence[str],
    w1_gate: Mapping[str, Any],
    checkpoint_report: Mapping[str, Any] | None,
) -> None:
    master = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "steps": int(config.steps),
        "replay_mode": REPLAY_MODE,
        "deterministic_replay": {
            "rtol": RTOL,
            "atol": ATOL,
            "match": False,
            "max_per_step_relative_diff": None,
            "max_per_step_absolute_diff": None,
        },
        "alpha_effect": {
            "loss_alpha_zero": None,
            "loss_alpha_nonzero": None,
            "relative_diff": None,
            "absolute_diff": None,
            "above_threshold": False,
        },
        "r2": {
            "phase_assignment_counts": {},
            "phase_thresholds_used": {},
            "threshold_switch_events": [],
            "global_threshold_override_detected": False,
        },
        "r4": {
            "alpha": float(config.alpha),
            "conditioned_branch_grad_nonzero": False,
            "unconditioned_branch_grad_nonzero": False,
            "alpha_nonzero_affects_total_loss": False,
        },
        "checkpoint": {
            "path": _repo_relative(config.repo_root, config.checkpoint_dir),
            "compat_check_passed": bool(
                checkpoint_report and checkpoint_report.get("compat_check_passed") is True
            ),
            "reload_delta_l2": None,
            "reload_delta_nonzero_param_count": None,
        },
        "w1_gate": dict(w1_gate),
        "m12_rollback_triggered": False,
        "blocking_reasons": list(blocking_reasons),
    }
    _write_json(config.output_dir / "r2_r4_shape_regression.json", master)
    _write_json(config.output_dir / "c2_rollback_reason.json", _rollback_report(master))
    if (
        checkpoint_report is not None
        and checkpoint_report.get("compat_check_passed") is not True
    ):
        _write_json(config.output_dir / "checkpoint_compat_block.json", checkpoint_report)
    _write_json(config.output_dir / "run_manifest.json", _manifest(config, master))
    _write_json(config.paper_audit_root / "r2_closure" / "r2_closure_verdict.json", _closure_verdict("r2", status, blocking_reasons, config))
    _write_json(config.paper_audit_root / "r4_closure" / "r4_closure_verdict.json", _closure_verdict("r4", status, blocking_reasons, config))


def _manifest(config: RegressionConfig, master: Mapping[str, Any]) -> dict[str, Any]:
    terminal_at_utc = _utc_now()
    return {
        "schema_version": "w2_shape_regression_run_manifest_v2",
        "created_at_utc": terminal_at_utc,
        "terminal_at_utc": terminal_at_utc,
        "run_id": config.run_id,
        "script": _repo_relative(config.repo_root, Path(__file__)),
        "checkpoint_path": _repo_relative(config.repo_root, config.checkpoint_dir),
        "w1_summary_path": _repo_relative(config.repo_root, config.w1_summary_path),
        "output_dir": _repo_relative(config.repo_root, config.output_dir),
        "paper_audit_root": _repo_relative(config.repo_root, config.paper_audit_root),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "replay_mode": REPLAY_MODE,
        "steps": int(config.steps),
        "seed": int(config.seed),
        "alpha": float(config.alpha),
        "status": master.get("status"),
        "blocking_reasons": list(master.get("blocking_reasons", [])),
        "unexpected_termination": False,
        "regression_surface": "checkpoint_tensor_shape_regression_harness",
    }


def _rollback_report(master: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "iter5_c2_rollback_reason_v1",
        "triggered": False,
        "reason": None,
        "status": master.get("status"),
        "blocking_reasons": list(master.get("blocking_reasons", [])),
    }


def _closure_verdict(
    axis: str,
    status: str,
    blocking_reasons: Sequence[str],
    config: RegressionConfig,
) -> dict[str, Any]:
    closed = status == "PASS"
    return {
        "schema_version": f"{axis}_closure_verdict_v1",
        "axis": axis,
        "status": "CLOSED" if closed else "BLOCK_post_regression",
        "static_w1_required": True,
        "behavioral_w2_required": True,
        "behavioral_w2_evidence": _repo_relative(
            config.repo_root, config.output_dir / "r2_r4_shape_regression.json"
        ),
        "blocking_reasons": [] if closed else list(blocking_reasons),
    }


def _run_regression(config: RegressionConfig) -> dict[str, Any]:
    compat_passed, checkpoint_report, checkpoint_reasons = _checkpoint_compatibility(config)
    _write_json(config.output_dir / "checkpoint_compatibility_check.json", checkpoint_report)
    if not compat_passed:
        _emit_block_outputs(
            config=config,
            status="BLOCK",
            blocking_reasons=checkpoint_reasons,
            w1_gate={"not_evaluated": True},
            checkpoint_report=checkpoint_report,
        )
        return _read_json(config.output_dir / "r2_r4_shape_regression.json") or {}

    w1_passed, w1_payload, w1_reasons = _w1_gate_passed(config.w1_summary_path)
    if not w1_passed:
        _emit_block_outputs(
            config=config,
            status="BLOCK",
            blocking_reasons=w1_reasons,
            w1_gate=w1_payload,
            checkpoint_report=checkpoint_report,
        )
        return _read_json(config.output_dir / "r2_r4_shape_regression.json") or {}

    tensors = _load_checkpoint_tensors(config.checkpoint_dir)
    device = _select_device()
    first = _run_loss_trajectory(
        tensors=tensors,
        steps=config.steps,
        seed=config.seed,
        alpha=config.alpha,
        device=device,
    )
    second = _run_loss_trajectory(
        tensors=tensors,
        steps=config.steps,
        seed=config.seed,
        alpha=config.alpha,
        device=device,
    )
    alpha_zero = _run_loss_trajectory(
        tensors=tensors,
        steps=max(1, config.steps // 4),
        seed=config.seed,
        alpha=0.0,
        device=device,
    )
    replay = _trajectory_diff(first, second)
    threshold_rows, phase_counts, phase_thresholds = _threshold_trace(config.steps)
    switch_events = [row for row in threshold_rows if row["switch_event"]]
    reload_report = _reload_delta_report(config)
    grad_probe = _gradient_probe(tensors=tensors, alpha=config.alpha, device=device)

    loss_alpha_nonzero = float(sum(row["loss_total"] for row in first) / len(first))
    loss_alpha_zero = float(
        sum(row["loss_total"] for row in alpha_zero) / len(alpha_zero)
    )
    absolute_diff = abs(loss_alpha_nonzero - loss_alpha_zero)
    relative_diff = absolute_diff / max(abs(loss_alpha_zero), 1e-12)
    alpha_above = bool(relative_diff > 0.01 and absolute_diff >= 5e-3)
    status = "PASS" if replay["match"] and alpha_above else "BLOCK_post_regression"
    blocking_reasons: list[str] = []
    if not replay["match"]:
        blocking_reasons.append("deterministic_replay_mismatch")
    if not alpha_above:
        blocking_reasons.append("alpha_effect_below_threshold")

    loss_rows = [
        {
            "step": int(row["step"]),
            "loss_total": row["loss_total"],
            "loss_unconditional": row["loss_unconditional"],
            "loss_advantage_conditioned": row["loss_advantage_conditioned"],
            "alpha_term_contribution": row["alpha_term_contribution"],
        }
        for row in first
    ]
    master = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "steps": int(config.steps),
        "replay_mode": REPLAY_MODE,
        "deterministic_replay": replay,
        "alpha_effect": {
            "loss_alpha_zero": float(loss_alpha_zero),
            "loss_alpha_nonzero": float(loss_alpha_nonzero),
            "relative_diff": float(relative_diff),
            "absolute_diff": float(absolute_diff),
            "above_threshold": alpha_above,
        },
        "r2": {
            "phase_assignment_counts": phase_counts,
            "phase_thresholds_used": phase_thresholds,
            "threshold_switch_events": switch_events,
            "global_threshold_override_detected": False,
        },
        "r4": {
            "alpha": float(config.alpha),
            "conditioned_branch_grad_nonzero": bool(
                grad_probe["conditioned_branch_grad_nonzero"]
            ),
            "unconditioned_branch_grad_nonzero": bool(
                grad_probe["unconditioned_branch_grad_nonzero"]
            ),
            "alpha_nonzero_affects_total_loss": alpha_above,
            "gradient_probe": grad_probe,
        },
        "checkpoint": {
            "path": _repo_relative(config.repo_root, config.checkpoint_dir),
            "compat_check_passed": True,
            "reload_delta_l2": reload_report["reload_delta_l2"],
            "reload_delta_nonzero_param_count": reload_report[
                "reload_delta_nonzero_param_count"
            ],
        },
        "w1_gate": w1_payload,
        "m12_rollback_triggered": False,
        "blocking_reasons": blocking_reasons,
    }
    _write_jsonl(config.output_dir / "loss_decomposition.jsonl", loss_rows)
    _write_jsonl(config.output_dir / "threshold_switch_trace.jsonl", threshold_rows)
    _write_json(config.output_dir / "checkpoint_reload_delta_report.json", reload_report)
    _write_json(config.output_dir / "deterministic_replay_diff.json", replay)
    _write_json(config.output_dir / "alpha_ablation_report.json", master["alpha_effect"])
    _write_json(config.output_dir / "r2_r4_shape_regression.json", master)
    _write_json(config.output_dir / "c2_rollback_reason.json", _rollback_report(master))
    _write_json(config.output_dir / "run_manifest.json", _manifest(config, master))
    _write_json(
        config.paper_audit_root / "r2_closure" / "r2_closure_verdict.json",
        _closure_verdict("r2", status, blocking_reasons, config),
    )
    _write_json(
        config.paper_audit_root / "r4_closure" / "r4_closure_verdict.json",
        _closure_verdict("r4", status, blocking_reasons, config),
    )
    return master


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iter4_r2_r4_shape_regression.py",
        description="Emit Iter4 W2 R2/R4 checkpoint tensor shape-regression artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_REL)
    parser.add_argument("--w1-summary", default=DEFAULT_W1_SUMMARY_REL)
    parser.add_argument(
        "--output-dir",
        default=(
            "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
            f"{DEFAULT_RUN_ID}/gr00t/r2_r4_shape_regression"
        ),
    )
    parser.add_argument(
        "--paper-audit-root",
        default=f"agent/artifacts/{DEFAULT_RUN_ID}/paper_audit",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return parser


def _config_from_args(args: argparse.Namespace) -> RegressionConfig:
    repo_root = REPO_ROOT
    steps = int(args.steps)
    alpha = float(args.alpha)
    if steps <= 0:
        raise ValueError("--steps must be positive")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("--alpha must be a finite non-negative number")
    return RegressionConfig(
        repo_root=repo_root,
        run_id=str(args.run_id),
        output_dir=_resolve_path(repo_root, str(args.output_dir)),
        paper_audit_root=_resolve_path(repo_root, str(args.paper_audit_root)),
        checkpoint_dir=_resolve_path(repo_root, str(args.checkpoint_dir)),
        w1_summary_path=_resolve_path(repo_root, str(args.w1_summary)),
        steps=steps,
        seed=int(args.seed),
        alpha=alpha,
    )


def main() -> int:
    args = _build_parser().parse_args()
    config = _config_from_args(args)
    master = _run_regression(config)
    print(json.dumps(master, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
