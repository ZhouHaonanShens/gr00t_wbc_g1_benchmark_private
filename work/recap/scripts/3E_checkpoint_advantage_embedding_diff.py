#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts.state_conditioned_common import write_json
from work.recap.stage3_contract_precondition_gate import ADV_SERVER_REQUIRED
from work.recap.stage3_contract_precondition_gate import BASELINE_DEFAULT_ADV_INIT
from work.recap.stage3_contract_precondition_gate import (
    _inspect_checkpoint_weight_map_features,
)
from work.recap.stage3_contract_precondition_gate import _select_prelim_eval_surface


DEFAULT_INITIAL_SNAPSHOT = Path(
    "agent/artifacts/stage3_t10_advantage_1gpu/formal_run/conditioned_initial_advantage_embedding_snapshot.json"
)
DEFAULT_CONDITIONED_CHECKPOINT = Path(
    "agent/artifacts/stage3_t10_advantage_1gpu/formal_run/checkpoint-200"
)
DEFAULT_OUTPUT_JSON = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v1/advantage_embedding_diff.json"
)
ADVANTAGE_WEIGHT_KEY = "action_head.advantage_embedding.weight"
ADVANTAGE_BIAS_KEY = "action_head.advantage_embedding.bias"
SCHEMA_VERSION = "checkpoint_advantage_embedding_diff_v1"
ARTIFACT_KIND = "checkpoint_advantage_embedding_diff"
PREVIEW_ABS_TOL = 1e-7
SUMMARY_ABS_TOL = 1e-7


def _resolve_path(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _repo_relative(path: str | Path) -> str:
    resolved = _resolve_path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _load_json_dict(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must load as a mapping: {path}")
    return payload


def _shape_list(value: Any) -> list[int]:
    shape = getattr(value, "shape", ())
    return [int(dim) for dim in tuple(shape)]


def _selected_checkpoint_index_and_root(checkpoint: Path) -> tuple[Path, Path, dict[str, Any]]:
    if checkpoint.is_dir():
        index_path = checkpoint / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"checkpoint directory missing model.safetensors.index.json: {checkpoint}"
            )
        return checkpoint, index_path, _load_json_dict(index_path, label="checkpoint index")
    if checkpoint.is_file() and checkpoint.name.endswith(".index.json"):
        return checkpoint.parent, checkpoint, _load_json_dict(
            checkpoint, label="checkpoint index"
        )
    if checkpoint.is_file() and checkpoint.suffix == ".safetensors":
        payload = {
            "metadata": {"total_size": int(checkpoint.stat().st_size)},
            "weight_map": {
                ADVANTAGE_WEIGHT_KEY: checkpoint.name,
                ADVANTAGE_BIAS_KEY: checkpoint.name,
            },
        }
        return checkpoint.parent, checkpoint.with_name(checkpoint.name + ".index.json"), payload
    raise FileNotFoundError(f"unsupported conditioned checkpoint input: {checkpoint}")


def _load_checkpoint_tensor(
    *, checkpoint_root: Path, index_payload: dict[str, Any], key: str
) -> Any:
    from safetensors import safe_open

    weight_map_raw = index_payload.get("weight_map")
    if not isinstance(weight_map_raw, dict):
        raise TypeError("checkpoint index weight_map must be an object")
    shard_rel = weight_map_raw.get(key)
    if not isinstance(shard_rel, str) or not shard_rel.strip():
        raise KeyError(f"checkpoint index missing shard mapping for {key!r}")
    shard_path = (checkpoint_root / shard_rel).resolve()
    if not shard_path.is_file():
        raise FileNotFoundError(f"checkpoint shard missing for {key!r}: {shard_path}")
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _tensor_summary(tensor: Any, *, preview_limit: int = 8) -> dict[str, Any]:
    import torch

    detached = tensor.detach().float().cpu()
    flat = detached.reshape(-1)
    numel = int(flat.numel())
    all_finite = bool(torch.isfinite(flat).all().item()) if numel else True
    return {
        "shape": _shape_list(detached),
        "dtype": str(getattr(tensor, "dtype", detached.dtype)),
        "numel": numel,
        "min": float(flat.min().item()) if numel else None,
        "max": float(flat.max().item()) if numel else None,
        "mean": float(flat.mean().item()) if numel else None,
        "std": float(flat.std(unbiased=False).item()) if numel else None,
        "preview": [float(value) for value in flat[: int(preview_limit)].tolist()],
        "all_finite": all_finite,
    }


def _float_close(expected: Any, actual: Any, *, abs_tol: float) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=abs_tol)


def _compare_tensor_to_snapshot(
    *, key: str, tensor_summary: dict[str, Any], snapshot_entry: dict[str, Any]
) -> dict[str, Any]:
    expected_shape = [int(dim) for dim in list(snapshot_entry.get("shape", []))]
    expected_numel = int(snapshot_entry.get("numel", -1))
    expected_preview = [float(value) for value in list(snapshot_entry.get("preview", []))]
    actual_preview = [float(value) for value in tensor_summary.get("preview", [])]
    preview_pairs = list(zip(actual_preview[: len(expected_preview)], expected_preview))
    preview_matches = len(actual_preview) >= len(expected_preview) and all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=PREVIEW_ABS_TOL)
        for actual, expected in preview_pairs
    )
    stat_fields = ("min", "max", "mean", "std")
    stat_matches = {
        field: _float_close(
            snapshot_entry.get(field), tensor_summary.get(field), abs_tol=SUMMARY_ABS_TOL
        )
        for field in stat_fields
    }
    shape_matches = expected_shape == list(tensor_summary.get("shape", []))
    numel_matches = expected_numel == int(tensor_summary.get("numel", -2))
    snapshot_surface_matches = bool(
        shape_matches
        and numel_matches
        and preview_matches
        and all(bool(match) for match in stat_matches.values())
    )
    change_evidence: list[str] = []
    if not shape_matches:
        change_evidence.append("shape_mismatch")
    if not numel_matches:
        change_evidence.append("numel_mismatch")
    if not preview_matches:
        change_evidence.append("preview_mismatch")
    for field, matched in stat_matches.items():
        if not matched:
            change_evidence.append(f"{field}_mismatch")
    return {
        "tensor_key": key,
        "comparison_performed": True,
        "shape_matches_snapshot": shape_matches,
        "numel_matches_snapshot": numel_matches,
        "preview_matches_snapshot": preview_matches,
        "stat_matches_snapshot": stat_matches,
        "matches_initial_snapshot_surface": snapshot_surface_matches,
        "changed_from_initial_snapshot_surface": not snapshot_surface_matches,
        "change_evidence": change_evidence,
    }


def _surface_mode_name(mode: Any) -> str:
    raw = str(mode or "").strip()
    if raw == ADV_SERVER_REQUIRED:
        return "ADV_SERVER_REQUIRED"
    if raw == BASELINE_DEFAULT_ADV_INIT:
        return "BASELINE_DEFAULT_ADV_INIT"
    return raw.upper()


def build_payload(
    *, initial_snapshot: Path, conditioned_checkpoint: Path, output_json: Path
) -> dict[str, Any]:
    snapshot_payload = _load_json_dict(initial_snapshot, label="initial snapshot")
    snapshot_tensors = snapshot_payload.get("tensors")
    if not isinstance(snapshot_tensors, dict):
        raise TypeError("initial snapshot tensors must be an object")
    weight_snapshot = snapshot_tensors.get(ADVANTAGE_WEIGHT_KEY)
    bias_snapshot = snapshot_tensors.get(ADVANTAGE_BIAS_KEY)
    if not isinstance(weight_snapshot, dict):
        raise KeyError(f"initial snapshot missing {ADVANTAGE_WEIGHT_KEY!r}")
    if not isinstance(bias_snapshot, dict):
        raise KeyError(f"initial snapshot missing {ADVANTAGE_BIAS_KEY!r}")

    checkpoint_root, checkpoint_index_path, checkpoint_index_payload = (
        _selected_checkpoint_index_and_root(conditioned_checkpoint)
    )
    features = _inspect_checkpoint_weight_map_features(
        repo_root=REPO_ROOT,
        manifest_payload={},
        checkpoint_path=conditioned_checkpoint,
        checkpoint_source_field="conditioned_checkpoint",
    )
    prelim_eval_surface, local_gate_pass, failure_reason_codes = (
        _select_prelim_eval_surface(features=features)
    )
    internal_surface_mode = str(prelim_eval_surface.get("mode") or "")
    surface_mode = _surface_mode_name(internal_surface_mode)
    allow_baseline_default_advantage_embedding_init = bool(
        prelim_eval_surface.get("allow_baseline_default_advantage_embedding_init")
    )
    has_advantage_embedding_weight = bool(features.get("has_advantage_embedding_weight"))
    has_advantage_embedding_bias = bool(features.get("has_advantage_embedding_bias"))
    conditioned_surface_is_valid = bool(
        local_gate_pass
        and has_advantage_embedding_weight
        and has_advantage_embedding_bias
        and surface_mode == "ADV_SERVER_REQUIRED"
        and not allow_baseline_default_advantage_embedding_init
    )

    weight_tensor = _load_checkpoint_tensor(
        checkpoint_root=checkpoint_root,
        index_payload=checkpoint_index_payload,
        key=ADVANTAGE_WEIGHT_KEY,
    )
    bias_tensor = _load_checkpoint_tensor(
        checkpoint_root=checkpoint_root,
        index_payload=checkpoint_index_payload,
        key=ADVANTAGE_BIAS_KEY,
    )
    weight_summary = _tensor_summary(weight_tensor)
    bias_summary = _tensor_summary(bias_tensor)
    weight_comparison = _compare_tensor_to_snapshot(
        key=ADVANTAGE_WEIGHT_KEY,
        tensor_summary=weight_summary,
        snapshot_entry=weight_snapshot,
    )
    bias_comparison = _compare_tensor_to_snapshot(
        key=ADVANTAGE_BIAS_KEY,
        tensor_summary=bias_summary,
        snapshot_entry=bias_snapshot,
    )

    advantage_embedding_weight_changed_from_init = bool(
        weight_comparison.get("changed_from_initial_snapshot_surface")
    )
    advantage_embedding_bias_checked = bool(
        bias_comparison.get("comparison_performed")
    )
    advantage_embedding_bias_changed_from_init = bool(
        bias_comparison.get("changed_from_initial_snapshot_surface")
    )
    all_tensors_finite = bool(
        weight_summary.get("all_finite") and bias_summary.get("all_finite")
    )
    required_semantics_pass = bool(
        conditioned_surface_is_valid
        and advantage_embedding_weight_changed_from_init
        and advantage_embedding_bias_checked
        and all_tensors_finite
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "initial_snapshot_path": _repo_relative(initial_snapshot),
        "conditioned_checkpoint_path": _repo_relative(conditioned_checkpoint),
        "conditioned_checkpoint_index_path": _repo_relative(checkpoint_index_path),
        "output_json_path": _repo_relative(output_json),
        "snapshot_kind": str(snapshot_payload.get("snapshot_kind") or ""),
        "advantage_embedding_loaded_from_checkpoint": bool(
            snapshot_payload.get("advantage_embedding_loaded_from_checkpoint")
        ),
        "missing_keys_at_load": list(snapshot_payload.get("missing_keys_at_load", [])),
        "surface_mode": surface_mode,
        "surface_mode_internal": internal_surface_mode,
        "allow_baseline_default_advantage_embedding_init": (
            allow_baseline_default_advantage_embedding_init
        ),
        "has_advantage_embedding_weight": has_advantage_embedding_weight,
        "has_advantage_embedding_bias": has_advantage_embedding_bias,
        "failure_reason_codes": [str(code) for code in list(failure_reason_codes)],
        "conditioned_surface_is_valid": conditioned_surface_is_valid,
        "advantage_embedding_weight_changed_from_init": (
            advantage_embedding_weight_changed_from_init
        ),
        "advantage_embedding_bias_checked": advantage_embedding_bias_checked,
        "advantage_embedding_bias_changed_from_init": (
            advantage_embedding_bias_changed_from_init
        ),
        "all_tensors_finite": all_tensors_finite,
        "required_semantics_pass": required_semantics_pass,
        "weight_comparison": weight_comparison,
        "bias_comparison": bias_comparison,
        "final_tensors": {
            ADVANTAGE_WEIGHT_KEY: weight_summary,
            ADVANTAGE_BIAS_KEY: bias_summary,
        },
        "initial_snapshot_tensors": {
            ADVANTAGE_WEIGHT_KEY: weight_snapshot,
            ADVANTAGE_BIAS_KEY: bias_snapshot,
        },
        "exit_code": 0 if required_semantics_pass else 1,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="3E_checkpoint_advantage_embedding_diff.py",
        description=(
            "Compare the conditioned checkpoint advantage embedding tensors against the conditioned "
            "initial snapshot and persist a machine-readable diff artifact."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--initial-snapshot",
        type=Path,
        default=DEFAULT_INITIAL_SNAPSHOT,
        help="Conditioned initial advantage embedding snapshot JSON from before training.",
    )
    parser.add_argument(
        "--conditioned-checkpoint",
        type=Path,
        default=DEFAULT_CONDITIONED_CHECKPOINT,
        help="Conditioned checkpoint directory or safetensors asset to inspect.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Where to atomically write the advantage embedding diff artifact.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    initial_snapshot = _resolve_path(args.initial_snapshot)
    conditioned_checkpoint = _resolve_path(args.conditioned_checkpoint)
    output_json = _resolve_path(args.output_json)
    payload = build_payload(
        initial_snapshot=initial_snapshot,
        conditioned_checkpoint=conditioned_checkpoint,
        output_json=output_json,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    _ = write_json(output_json, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return int(payload.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
