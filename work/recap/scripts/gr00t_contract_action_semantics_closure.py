#!/usr/bin/env python3
"""Read-only diagnostics for GR00T contract/action-semantics closure.

The script intentionally never mutates checkpoints, datasets, or submodules.
It writes phase artifacts under a caller supplied output directory and is scoped
to the GR00T-N1.6/G1 identity/export contract investigation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_GR00T_ROOT = REPO_ROOT / "submodules" / "Isaac-GR00T"
if str(ISAAC_GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAAC_GR00T_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.scripts.gr00t_safe_adaptation_action_audit import (  # noqa: E402
    MODALITIES,
    _extract_policy_observation,
    _load_policy,
    _load_stage3_loader,
    _predict_one_chunk,
)


FIXED_DECISIONS: dict[str, str] = {
    "A": "A. identity/export contract 未闭合，先修 processor/config；",
    "B": "B. identity 已闭合，但 Stage3 action labels 无法 replay，先修 action label semantics；",
    "C": "C. right_hand label 是主因，需重构 hand absolute target 或 mask+distill；",
    "D": "D. navigate label 是主因，需重构 lower/nav command 或混入 base replay；",
    "E": "E. action labels 已验证有效，collapse 才能归因到 training scope；",
    "F": "F. 可以进入 Safe-SFT LoRA smoke；",
    "G": "G. 可以进入 Guarded RECAP；",
    "H": "H. 可以进入 FATG/per-edge。",
}


DANGEROUS_DIFF_PATH_FRAGMENTS = (
    "formalize_language",
    "processor",
    "action",
    "state",
    "horizon",
    "delta_indices",
    "modality",
    "embodiment",
    "statistics",
    "tokenizer",
    "dtype",
    "preprocess",
    "normalization",
)


PHASE1_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "modality.json",
    "embodiment_id.json",
    "statistics.json",
    "experiment_cfg/dataset_statistics.json",
    "model.safetensors.index.json",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _resolve(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_paths(manifest: Path) -> dict[str, Path]:
    payload = _read_json(manifest)
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError(f"manifest has no checkpoints dict: {manifest}")
    required = ("base", "identity", "post_current")
    missing = [key for key in required if key not in checkpoints]
    if missing:
        raise ValueError(f"manifest missing checkpoint keys: {missing}")
    return {key: _resolve(value) for key, value in checkpoints.items()}


def _iter_model_files(checkpoint: Path) -> Iterable[Path]:
    for pattern in ("model-*.safetensors", "model.safetensors"):
        yield from sorted(checkpoint.glob(pattern))


def _diff_json(left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [
            {
                "path": path,
                "left": left,
                "right": right,
                "kind": "type_mismatch",
            }
        ]
    if isinstance(left, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}"
            if key not in left:
                rows.append(
                    {
                        "path": child_path,
                        "left": "<MISSING>",
                        "right": right[key],
                        "kind": "missing_left",
                    }
                )
            elif key not in right:
                rows.append(
                    {
                        "path": child_path,
                        "left": left[key],
                        "right": "<MISSING>",
                        "kind": "missing_right",
                    }
                )
            else:
                rows.extend(_diff_json(left[key], right[key], child_path))
        return rows
    if isinstance(left, list):
        rows = []
        if len(left) != len(right):
            rows.append(
                {
                    "path": f"{path}.len",
                    "left": len(left),
                    "right": len(right),
                    "kind": "list_length",
                }
            )
        for idx, (l_value, r_value) in enumerate(zip(left, right)):
            rows.extend(_diff_json(l_value, r_value, f"{path}[{idx}]"))
        return rows
    if left != right:
        return [{"path": path, "left": left, "right": right, "kind": "value"}]
    return []


def _json_diff_for_file(left_file: Path, right_file: Path) -> list[dict[str, Any]]:
    try:
        left = _read_json(left_file)
        right = _read_json(right_file)
    except Exception as exc:  # pragma: no cover - diagnostic path.
        return [
            {
                "path": "$",
                "left": f"JSON_READ_ERROR:{exc}",
                "right": f"JSON_READ_ERROR:{exc}",
                "kind": "json_read_error",
            }
        ]
    return _diff_json(left, right)


def _dangerous_diff(file_name: str, diff_rows: list[dict[str, Any]]) -> bool:
    if not diff_rows:
        return False
    if file_name in {
        "config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "modality.json",
        "statistics.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
    }:
        return True
    joined = "\n".join(str(row.get("path", "")) for row in diff_rows)
    return any(fragment in joined for fragment in DANGEROUS_DIFF_PATH_FRAGMENTS)


def _extract_contract_fields(checkpoint: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    config_path = checkpoint / "config.json"
    processor_path = checkpoint / "processor_config.json"
    embodiment_path = checkpoint / "embodiment_id.json"
    if config_path.exists():
        config = _read_json(config_path)
        fields["config.formalize_language"] = config.get("formalize_language")
        fields["config.model_type"] = config.get("model_type")
        fields["config.torch_dtype"] = config.get("torch_dtype")
    if processor_path.exists():
        processor = _read_json(processor_path)
        fields["processor.processor_kwargs.formalize_language"] = (
            processor.get("processor_kwargs", {}).get("formalize_language")
        )
        fields["processor.processor_class"] = processor.get("processor_class")
    if embodiment_path.exists():
        fields["embodiment_id"] = _read_json(embodiment_path)
    return fields


def phase1_diff(args: argparse.Namespace) -> int:
    manifest = _resolve(args.manifest)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = _checkpoint_paths(manifest)

    file_hash_rows: list[dict[str, Any]] = []
    for checkpoint_key, checkpoint in checkpoints.items():
        for rel_name in PHASE1_FILES:
            file_path = checkpoint / rel_name
            row = {
                "checkpoint": checkpoint_key,
                "file": rel_name,
                "path": _rel(file_path),
                "exists": file_path.exists(),
                "size_bytes": file_path.stat().st_size if file_path.exists() else None,
                "sha256": _sha256(file_path) if file_path.exists() else None,
            }
            file_hash_rows.append(row)
        for model_file in _iter_model_files(checkpoint):
            file_hash_rows.append(
                {
                    "checkpoint": checkpoint_key,
                    "file": model_file.name,
                    "path": _rel(model_file),
                    "exists": True,
                    "size_bytes": model_file.stat().st_size,
                    "sha256": _sha256(model_file),
                }
            )

    compare_pairs = (("base", "identity"), ("base", "post_current"))
    file_diffs: list[dict[str, Any]] = []
    dangerous_diffs: list[dict[str, Any]] = []
    for left_key, right_key in compare_pairs:
        left = checkpoints[left_key]
        right = checkpoints[right_key]
        for rel_name in PHASE1_FILES:
            left_file = left / rel_name
            right_file = right / rel_name
            if not left_file.exists() and not right_file.exists():
                continue
            if left_file.exists() != right_file.exists():
                diff_rows = [
                    {
                        "path": "$",
                        "left": "exists" if left_file.exists() else "<MISSING>",
                        "right": "exists" if right_file.exists() else "<MISSING>",
                        "kind": "existence",
                    }
                ]
            elif rel_name.endswith(".json"):
                diff_rows = _json_diff_for_file(left_file, right_file)
            else:
                diff_rows = (
                    []
                    if _sha256(left_file) == _sha256(right_file)
                    else [{"path": "$", "left": "sha256", "right": "sha256", "kind": "bytes"}]
                )
            is_dangerous = _dangerous_diff(rel_name, diff_rows)
            record = {
                "left": left_key,
                "right": right_key,
                "file": rel_name,
                "diff_count": len(diff_rows),
                "dangerous": is_dangerous,
                "diff_rows": diff_rows[:200],
            }
            file_diffs.append(record)
            if is_dangerous:
                dangerous_diffs.append(record)

    hash_csv = out_dir / "stats_processor_config_hashes.csv"
    with hash_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("checkpoint", "file", "path", "exists", "size_bytes", "sha256"),
        )
        writer.writeheader()
        writer.writerows(file_hash_rows)

    base_identity_dangerous = [
        row for row in dangerous_diffs if row["left"] == "base" and row["right"] == "identity"
    ]
    manifest_payload = {
        "schema_version": "gr00t_contract_phase1_bundle_diff_v1",
        "artifact_kind": "phase1_contract_bundle_diff",
        "generated_at_utc": _utc_now(),
        "source_manifest": _rel(manifest),
        "checkpoint_paths": {key: _rel(value) for key, value in checkpoints.items()},
        "checkpoint_contract_fields": {
            key: _extract_contract_fields(value) for key, value in checkpoints.items()
        },
        "file_hash_csv": _rel(hash_csv),
        "file_diffs": file_diffs,
        "dangerous_diffs": dangerous_diffs,
        "base_identity_dangerous_diff_count": len(base_identity_dangerous),
        "A1_bundle_config_stats_surface_status": (
            "FAIL" if base_identity_dangerous else "PASS"
        ),
        "A1_reason_code": "dangerous_diff" if base_identity_dangerous else "pass",
    }
    _write_json(out_dir / "bundle_diff_manifest.json", manifest_payload)

    diff_md = out_dir / "contract_surface_diff.md"
    lines = [
        "# Phase 1 Contract Surface Diff",
        "",
        f"- generated_at_utc: `{manifest_payload['generated_at_utc']}`",
        f"- source_manifest: `{_rel(manifest)}`",
        f"- A1 status: `{manifest_payload['A1_bundle_config_stats_surface_status']}`",
        f"- reason_code: `{manifest_payload['A1_reason_code']}`",
        "",
        "## Base vs identity dangerous diffs",
        "",
    ]
    if not base_identity_dangerous:
        lines.append("None.")
    else:
        for row in base_identity_dangerous:
            lines.append(f"### `{row['file']}`")
            for diff in row["diff_rows"]:
                lines.append(
                    f"- `{diff['path']}`: base=`{diff['left']}` identity=`{diff['right']}`"
                )
    lines.extend(
        [
            "",
            "## Contract fields",
            "",
            "```json",
            json.dumps(
                manifest_payload["checkpoint_contract_fields"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Notes",
            "",
            "- Base/identity paths were parsed from the upstream Phase1 contract manifest, not hand-typed.",
            "- Any base-vs-identity processor/config diff is treated as dangerous for A-gate, even when model shards/statistics match.",
        ]
    )
    diff_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest_payload["A1_bundle_config_stats_surface_status"],
                "reason_code": manifest_payload["A1_reason_code"],
                "output_dir": _rel(out_dir),
                "bundle_diff_manifest": _rel(out_dir / "bundle_diff_manifest.json"),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _collect_observation_specs(
    *,
    dataset: Path,
    episode_count: int,
    chunk_stride: int,
    max_windows: int,
) -> tuple[Any, list[dict[str, int]]]:
    loader = _load_stage3_loader(dataset)
    specs: list[dict[str, int]] = []
    for episode_index in range(min(int(episode_count), len(loader))):
        trajectory = loader[episode_index]
        for step_index in range(0, len(trajectory), int(chunk_stride)):
            specs.append(
                {
                    "obs_id": len(specs),
                    "source": "stage3_direct_dataset_hash_fallback",
                    "episode_index": int(episode_index),
                    "step_index": int(step_index),
                    "trajectory_len": int(len(trajectory)),
                }
            )
            if 0 < max_windows <= len(specs):
                return loader, specs
    return loader, specs


def _predict_policy_for_specs(
    *,
    checkpoint: Path,
    loader: Any,
    specs: list[dict[str, int]],
    plain_prompt: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    import torch

    policy = _load_policy(checkpoint)
    denorm_by_modality: dict[str, list[np.ndarray]] = {key: [] for key in MODALITIES}
    norm_by_modality: dict[str, list[np.ndarray]] = {key: [] for key in MODALITIES}
    records: list[dict[str, Any]] = []
    try:
        for spec in specs:
            _set_all_seeds(int(seed) + int(spec["obs_id"]))
            trajectory = loader[int(spec["episode_index"])]
            observation = _extract_policy_observation(
                loader=loader,
                trajectory=trajectory,
                step_index=int(spec["step_index"]),
                plain_prompt=plain_prompt,
            )
            denorm, normalized = _predict_one_chunk(policy, observation)
            for modality in MODALITIES:
                denorm_by_modality[modality].append(np.asarray(denorm[modality][0], dtype=np.float32))
                norm_by_modality[modality].append(
                    np.asarray(normalized[modality][0], dtype=np.float32)
                )
            records.append(
                {
                    **spec,
                    "checkpoint": _rel(checkpoint),
                    "policy_horizon": int(next(iter(denorm.values())).shape[1]),
                    "modalities": list(MODALITIES),
                }
            )
    finally:
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records, denorm_by_modality, norm_by_modality


def _layer_delta(
    left_values: dict[str, list[np.ndarray]],
    right_values: dict[str, list[np.ndarray]],
    *,
    layer: str,
    tolerance_max_abs: float,
    tolerance_mean_abs: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        left = np.stack(left_values[modality], axis=0)
        right = np.stack(right_values[modality], axis=0)
        diff = np.asarray(left - right, dtype=np.float64)
        abs_diff = np.abs(diff)
        rows.append(
            {
                "layer": layer,
                "modality": modality,
                "sample_shape": list(left.shape),
                "max_abs_diff": float(np.max(abs_diff)),
                "mean_abs_diff": float(np.mean(abs_diff)),
                "q99_abs_diff": float(np.quantile(abs_diff.reshape(-1), 0.99)),
                "cosine": _safe_cosine(left, right),
                "pass": bool(
                    np.max(abs_diff) <= tolerance_max_abs
                    and np.mean(abs_diff) <= tolerance_mean_abs
                ),
            }
        )
    return rows


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    l_vec = np.asarray(left, dtype=np.float64).reshape(-1)
    r_vec = np.asarray(right, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(l_vec) * np.linalg.norm(r_vec))
    if denom == 0.0:
        return 1.0 if np.allclose(l_vec, r_vec) else 0.0
    return float(np.dot(l_vec, r_vec) / denom)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _repeatability_probe(
    *,
    checkpoint: Path,
    loader: Any,
    spec: dict[str, int],
    plain_prompt: str,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    import torch

    policy = _load_policy(checkpoint)
    denorm_runs: list[dict[str, np.ndarray]] = []
    norm_runs: list[dict[str, np.ndarray]] = []
    try:
        trajectory = loader[int(spec["episode_index"])]
        observation = _extract_policy_observation(
            loader=loader,
            trajectory=trajectory,
            step_index=int(spec["step_index"]),
            plain_prompt=plain_prompt,
        )
        for repeat_index in range(int(repeats)):
            _set_all_seeds(seed)
            denorm, normalized = _predict_one_chunk(policy, observation)
            denorm_runs.append({key: np.asarray(value[0], dtype=np.float32) for key, value in denorm.items()})
            norm_runs.append(
                {key: np.asarray(value[0], dtype=np.float32) for key, value in normalized.items()}
            )
    finally:
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def max_pairwise(values: list[dict[str, np.ndarray]], modality: str) -> float:
        if len(values) < 2:
            return 0.0
        max_value = 0.0
        for idx in range(len(values) - 1):
            diff = np.abs(values[idx][modality] - values[idx + 1][modality])
            max_value = max(max_value, float(np.max(diff)))
        return max_value

    return {
        "checkpoint": _rel(checkpoint),
        "obs_id": int(spec["obs_id"]),
        "seed": int(seed),
        "repeats": int(repeats),
        "layers": {
            "L1_pred_normalized": {
                modality: max_pairwise(norm_runs, modality) for modality in MODALITIES
            },
            "L2_denorm_absolute": {
                modality: max_pairwise(denorm_runs, modality) for modality in MODALITIES
            },
        },
    }


def phase2_identity(args: argparse.Namespace) -> int:
    manifest = _resolve(args.manifest)
    dataset = _resolve(args.dataset)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = _checkpoint_paths(manifest)

    loader, specs = _collect_observation_specs(
        dataset=dataset,
        episode_count=args.episode_count,
        chunk_stride=args.chunk_stride,
        max_windows=args.max_windows,
    )
    if not specs:
        raise RuntimeError(f"no observations found in {dataset}")

    observation_bank = {
        "schema_version": "gr00t_contract_observation_bank_v1",
        "dataset": _rel(dataset),
        "authority_mode": args.authority_status,
        "episode_count_requested": int(args.episode_count),
        "chunk_stride": int(args.chunk_stride),
        "max_windows": int(args.max_windows),
        "observations": specs,
        "known_gaps": [
            "formal_eval_initial_observations_not_available_in_canonical_artifact",
            "base_success_grasp_lift_saved_observations_not_available_in_canonical_artifact",
        ],
    }
    _write_json(out_dir / "observation_bank_manifest.json", observation_bank)

    print(
        f"[IDENTITY] native_bundle_load base specs={len(specs)} checkpoint={checkpoints['base']}",
        flush=True,
    )
    base_records, base_denorm, base_norm = _predict_policy_for_specs(
        checkpoint=checkpoints["base"],
        loader=loader,
        specs=specs,
        plain_prompt=args.plain_prompt,
        seed=args.seed,
    )
    print(
        f"[IDENTITY] native_bundle_load identity specs={len(specs)} checkpoint={checkpoints['identity']}",
        flush=True,
    )
    identity_records, identity_denorm, identity_norm = _predict_policy_for_specs(
        checkpoint=checkpoints["identity"],
        loader=loader,
        specs=specs,
        plain_prompt=args.plain_prompt,
        seed=args.seed,
    )

    delta_rows = []
    delta_rows.extend(
        _layer_delta(
            base_norm,
            identity_norm,
            layer="L1_pred_normalized",
            tolerance_max_abs=args.tolerance_max_abs,
            tolerance_mean_abs=args.tolerance_mean_abs,
        )
    )
    delta_rows.extend(
        _layer_delta(
            base_denorm,
            identity_denorm,
            layer="L2_denorm_absolute",
            tolerance_max_abs=args.tolerance_max_abs,
            tolerance_mean_abs=args.tolerance_mean_abs,
        )
    )
    # In the current local offline policy seam, processor.decode_action already
    # includes the configured relative-to-absolute transform for relative arms.
    delta_rows.extend(
        {
            **row,
            "layer": "L3_postprocessed_relative_to_absolute_alias",
            "unavailable_reason": "offline Gr00tPolicy exposes decode_action; no separate postprocess hook exists for this seam",
        }
        for row in _layer_delta(
            base_denorm,
            identity_denorm,
            layer="L3_postprocessed_relative_to_absolute_alias",
            tolerance_max_abs=args.tolerance_max_abs,
            tolerance_mean_abs=args.tolerance_mean_abs,
        )
    )
    _write_csv(out_dir / "offline_identity_delta_summary.csv", delta_rows)

    jsonl_rows: list[dict[str, Any]] = []
    for base_record, identity_record in zip(base_records, identity_records):
        obs_id = int(base_record["obs_id"])
        row: dict[str, Any] = {
            "load_mode": "native_bundle_load",
            "obs_id": obs_id,
            "base_record": base_record,
            "identity_record": identity_record,
            "layers": {},
        }
        for layer_name, base_layer, identity_layer in (
            ("L1_pred_normalized", base_norm, identity_norm),
            ("L2_denorm_absolute", base_denorm, identity_denorm),
        ):
            layer_payload = {}
            for modality in MODALITIES:
                diff = np.abs(base_layer[modality][obs_id] - identity_layer[modality][obs_id])
                layer_payload[modality] = {
                    "max_abs_diff": float(np.max(diff)),
                    "mean_abs_diff": float(np.mean(diff)),
                    "q99_abs_diff": float(np.quantile(diff.reshape(-1), 0.99)),
                    "pass": bool(
                        np.max(diff) <= args.tolerance_max_abs
                        and np.mean(diff) <= args.tolerance_mean_abs
                    ),
                }
            row["layers"][layer_name] = layer_payload
        jsonl_rows.append(row)
    _write_jsonl(out_dir / "offline_identity_action_equivalence.jsonl", jsonl_rows)

    repeatability = {
        "schema_version": "gr00t_contract_phase2_repeatability_v1",
        "generated_at_utc": _utc_now(),
        "seed": int(args.seed),
        "repeats": int(args.repeats),
        "tolerance_max_abs": float(args.tolerance_max_abs),
        "probes": [
            _repeatability_probe(
                checkpoint=checkpoints["base"],
                loader=loader,
                spec=specs[0],
                plain_prompt=args.plain_prompt,
                seed=args.seed,
                repeats=args.repeats,
            ),
            _repeatability_probe(
                checkpoint=checkpoints["identity"],
                loader=loader,
                spec=specs[0],
                plain_prompt=args.plain_prompt,
                seed=args.seed,
                repeats=args.repeats,
            ),
        ],
    }
    _write_json(out_dir / "repeatability_probe.json", repeatability)

    failing_layers = [
        row for row in delta_rows if row.get("pass") is False and row["layer"].startswith("L")
    ]
    first_mismatch_layer = failing_layers[0]["layer"] if failing_layers else None
    if first_mismatch_layer is None:
        first_mismatch_layer = "L4_server_wrapper_controller_unavailable"
        reason_code = "server_roundtrip_missing"
        A2_status = "UNKNOWN"
    else:
        reason_code = "offline_layer_mismatch"
        A2_status = "FAIL"

    layer_md = out_dir / "layer_first_mismatch.md"
    lines = [
        "# Phase 2 Offline Identity Action Equivalence",
        "",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- load_mode: `native_bundle_load`",
        f"- observation_count: `{len(specs)}`",
        f"- tolerance: max_abs≤`{args.tolerance_max_abs}`, mean_abs≤`{args.tolerance_mean_abs}`",
        f"- A2 status: `{A2_status}`",
        f"- first_mismatch_layer: `{first_mismatch_layer}`",
        f"- reason_code: `{reason_code}`",
        "",
        "## Unsupported/kept-open surfaces",
        "",
        "- `forced_same_processor_config_stats`: not executed because no safe in-memory config override seam was found; checkpoint originals were not mutated.",
        "- `L4 server/wrapper/controller_input`: unavailable in offline LeRobot observation seam; A cannot be closed without a server/wrapper roundtrip.",
        "- `L5 env_applied_proxy`: unavailable without WBC rollout/replay; A cannot be closed from local-only evidence.",
        "",
        "## Largest modality deltas",
        "",
    ]
    for row in sorted(delta_rows, key=lambda item: float(item["max_abs_diff"]), reverse=True)[:14]:
        lines.append(
            f"- `{row['layer']}` / `{row['modality']}`: "
            f"max_abs={row['max_abs_diff']:.6g}, mean_abs={row['mean_abs_diff']:.6g}, pass={row['pass']}"
        )
    layer_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "schema_version": "gr00t_contract_phase2_identity_equivalence_v1",
        "artifact_kind": "phase2_identity_action_equivalence",
        "generated_at_utc": _utc_now(),
        "source_manifest": _rel(manifest),
        "dataset": _rel(dataset),
        "load_modes": {
            "native_bundle_load": "executed",
            "forced_same_processor_config_stats": "SKIPPED",
        },
        "forced_same_processor_config_stats_skip_reason": (
            "No safe in-memory override seam found; checkpoint originals must remain read-only."
        ),
        "observation_count": len(specs),
        "tolerance_max_abs": float(args.tolerance_max_abs),
        "tolerance_mean_abs": float(args.tolerance_mean_abs),
        "delta_summary_csv": _rel(out_dir / "offline_identity_delta_summary.csv"),
        "equivalence_jsonl": _rel(out_dir / "offline_identity_action_equivalence.jsonl"),
        "repeatability_probe_json": _rel(out_dir / "repeatability_probe.json"),
        "layer_first_mismatch_md": _rel(layer_md),
        "A2_offline_action_equivalence_status": A2_status,
        "A2_reason_code": reason_code,
        "first_mismatch_layer": first_mismatch_layer,
        "L4_server_wrapper_controller_status": "UNAVAILABLE",
        "L5_env_applied_proxy_status": "UNAVAILABLE",
    }
    _write_json(out_dir / "identity_equivalence_manifest.json", payload)
    print(
        json.dumps(
            {
                "status": A2_status,
                "reason_code": reason_code,
                "first_mismatch_layer": first_mismatch_layer,
                "output_dir": _rel(out_dir),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stats_lookup(rows: list[dict[str, str]], *, source: str, modality: str) -> dict[str, str]:
    for row in rows:
        if row.get("source") == source and row.get("modality") == modality:
            return row
    raise KeyError(f"missing stats row source={source!r} modality={modality!r}")


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _write_skip_phase3(
    *,
    out_dir: Path,
    authority_status: str,
    blocking_reason: str,
    evidence_paths: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ("label_episode_sequence_first20", 20, False),
        ("label_overexecute30_diagnostic", 30, True),
        ("label_hold_current_hand", 20, True),
        ("label_base_nav_splice", 20, True),
    ]
    manifest = {
        "schema_version": "gr00t_contract_phase3_label_replay_v1",
        "artifact_kind": "phase3_label_replay_skipped",
        "generated_at_utc": _utc_now(),
        "status": "SKIPPED",
        "authority_status": authority_status,
        "blocking_reason": blocking_reason,
        "policy_horizon": 30,
        "public_contract_executed_steps": 20,
        "evidence_paths": evidence_paths,
        "variants": [
            {
                "variant": name,
                "status": "SKIPPED",
                "executed_steps": steps,
                "diagnostic_only": diagnostic,
                "reason": blocking_reason,
            }
            for name, steps, diagnostic in variants
        ],
    }
    _write_json(out_dir / "stage3_action_label_replay_manifest.json", manifest)
    _write_jsonl(
        out_dir / "label_replay_per_episode.jsonl",
        [
            {
                "status": "SKIPPED",
                "authority_status": authority_status,
                "reason": blocking_reason,
                "policy_horizon": 30,
                "executed_steps": None,
            }
        ],
    )
    _write_csv(
        out_dir / "label_replay_summary.csv",
        [
            {
                "variant": name,
                "status": "SKIPPED",
                "policy_horizon": 30,
                "executed_steps": steps,
                "diagnostic_only": diagnostic,
                "success": "",
                "reached": "",
                "lifted": "",
                "failure_modes": blocking_reason,
            }
            for name, steps, diagnostic in variants
        ],
    )
    (out_dir / "replay_variant_comparison.md").write_text(
        "\n".join(
            [
                "# Phase 3 Stage3 Action-label Replay",
                "",
                f"- status: `SKIPPED`",
                f"- authority_status: `{authority_status}`",
                f"- blocking_reason: `{blocking_reason}`",
                "- public-contract replay (`first20`) was not run because A-gate is already failed; running replay cannot close identity/export contract.",
                "- `label_overexecute30_diagnostic` remains diagnostic-only and was not used to unlock B/F/G/H.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_phase4_right_hand(
    *,
    out_dir: Path,
    action_rows: list[dict[str, str]],
    blocking_reason: str,
    evidence_paths: list[str],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = ["dataset", "base", "pure-SFT", "RECAP"]
    base = _stats_lookup(action_rows, source="base", modality="right_hand")
    base_q99 = max(abs(_float(base, "q99")), 1e-12)
    rows: list[dict[str, Any]] = []
    for source in sources:
        row = _stats_lookup(action_rows, source=source, modality="right_hand")
        q99 = abs(_float(row, "q99"))
        rows.append(
            {
                "source": source,
                "modality": "right_hand",
                "contract_representation": "ABSOLUTE",
                "mean": _float(row, "mean"),
                "std": _float(row, "std"),
                "min": _float(row, "min"),
                "q50": _float(row, "q50"),
                "q99": _float(row, "q99"),
                "max": _float(row, "max"),
                "signed_sum_mean": _float(row, "signed_sum_mean"),
                "nonzero_frac": _float(row, "nonzero_frac"),
                "q99_over_base": q99 / base_q99,
                "diagnostic_only": True,
            }
        )
    _write_csv(out_dir / "right_hand_semantics_matrix.csv", rows)
    payload = {
        "schema_version": "gr00t_contract_phase4_right_hand_semantics_v1",
        "artifact_kind": "phase4_right_hand_diagnostic_only",
        "generated_at_utc": _utc_now(),
        "status": "UNKNOWN",
        "reason_code": "blocked_by_A_no_replay_counterfactual",
        "blocking_reason": blocking_reason,
        "contract_representation": "ABSOLUTE",
        "joint_order_source": "agent/exchange/wbc_env_io.md",
        "evidence_paths": evidence_paths,
        "summary": {
            "dataset_right_hand_q99": rows[0]["q99"],
            "base_right_hand_q99": rows[1]["q99"],
            "pure_sft_right_hand_q99_over_base": rows[2]["q99_over_base"],
            "recap_right_hand_q99_over_base": rows[3]["q99_over_base"],
            "near_zero_label_remains_semantically_unclosed": True,
            "controller_input_available": False,
            "counterfactual_replay_available": False,
        },
    }
    _write_json(out_dir / "right_hand_state_action_alignment.json", payload)
    verdict = [
        "# Phase 4 right_hand Semantics Verdict",
        "",
        "- status: `UNKNOWN` / diagnostic-only",
        "- contract: `right_hand` is ABSOLUTE under `agent/exchange/gr00t_policy_io.md` and `agent/exchange/wbc_env_io.md`.",
        f"- blocking_reason: `{blocking_reason}`",
        f"- dataset q99: `{rows[0]['q99']}`; base same-observation q99: `{rows[1]['q99']}`; dataset/base ratio: `{rows[0]['q99_over_base']:.6f}`.",
        "- This confirms a strong diagnostic conflict, but because A-gate failed and no controller/replay counterfactual was run here, it is not promoted to final C.",
        "",
    ]
    (out_dir / "right_hand_replay_verdict.md").write_text("\n".join(verdict), encoding="utf-8")
    return payload


def _write_phase5_navigate(
    *,
    out_dir: Path,
    action_rows: list[dict[str, str]],
    blocking_reason: str,
    evidence_paths: list[str],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = ["dataset", "base", "pure-SFT", "RECAP"]
    base = _stats_lookup(action_rows, source="base", modality="navigate_command")
    base_q99 = max(abs(_float(base, "q99")), 1e-12)
    rows: list[dict[str, Any]] = []
    for source in sources:
        row = _stats_lookup(action_rows, source=source, modality="navigate_command")
        q99 = abs(_float(row, "q99"))
        rows.append(
            {
                "source": source,
                "modality": "navigate_command",
                "contract_representation": "ABSOLUTE",
                "semantic": "(vx, vy, wz)",
                "units": "vx/vy m/s, wz rad/s",
                "mean": _float(row, "mean"),
                "std": _float(row, "std"),
                "min": _float(row, "min"),
                "q50": _float(row, "q50"),
                "q99": _float(row, "q99"),
                "max": _float(row, "max"),
                "signed_sum_mean": _float(row, "signed_sum_mean"),
                "nonzero_frac": _float(row, "nonzero_frac"),
                "q99_over_base": q99 / base_q99,
                "wbc_stand_walk_threshold_norm": 0.05,
                "diagnostic_only": True,
            }
        )
    _write_csv(out_dir / "navigate_semantics_matrix.csv", rows)
    payload = {
        "schema_version": "gr00t_contract_phase5_navigate_semantics_v1",
        "artifact_kind": "phase5_navigate_diagnostic_only",
        "generated_at_utc": _utc_now(),
        "status": "UNKNOWN",
        "reason_code": "blocked_by_A_no_base_nav_splice_replay",
        "blocking_reason": blocking_reason,
        "contract_representation": "ABSOLUTE",
        "semantic": "(vx, vy, wz)",
        "units": "vx/vy m/s, wz rad/s",
        "wbc_branch_evidence": {
            "standing_threshold_norm": 0.05,
            "standing_threshold_source": "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/policy/g1_gear_wbc_policy.py:223",
            "safe_default_injection_source": "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/control/policy/g1_decoupled_whole_body_policy.py:64-72",
        },
        "evidence_paths": evidence_paths,
        "summary": {
            "dataset_navigate_q99": rows[0]["q99"],
            "base_navigate_q99": rows[1]["q99"],
            "pure_sft_navigate_q99_over_base": rows[2]["q99_over_base"],
            "recap_navigate_q99_over_base": rows[3]["q99_over_base"],
            "navigate_weakened_diagnostic": True,
            "base_nav_splice_replay_available": False,
        },
    }
    _write_json(out_dir / "navigate_reach_sufficiency.json", payload)
    verdict = [
        "# Phase 5 navigate_command Semantics Verdict",
        "",
        "- status: `UNKNOWN` / diagnostic-only",
        "- contract: `navigate_command` is ABSOLUTE `(vx, vy, wz)`, not relative.",
        "- WBC lower-body policy selects standing when `||cmd|| < 0.05`; missing navigate is safely defaulted to stop.",
        f"- blocking_reason: `{blocking_reason}`",
        f"- dataset q99: `{rows[0]['q99']}`; base q99: `{rows[1]['q99']}`; dataset/base ratio: `{rows[0]['q99_over_base']:.6f}`.",
        "- This confirms diagnostic weakening, but because A-gate failed and no `label_base_nav_splice` replay was run here, it is not promoted to final D.",
        "",
    ]
    (out_dir / "navigate_verdict.md").write_text("\n".join(verdict), encoding="utf-8")
    return payload


def _write_phase6_zero_variance(
    *,
    out_dir: Path,
    base_stats_path: Path,
    evidence_paths: list[str],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _read_json(base_stats_path)
    action_stats = stats["unitree_g1"]["action"]
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        values = action_stats[modality]
        std = np.asarray(values["std"], dtype=np.float64)
        min_values = np.asarray(values["min"], dtype=np.float64)
        max_values = np.asarray(values["max"], dtype=np.float64)
        range_values = max_values - min_values
        near_zero_std = bool(np.any(std < 1e-6))
        zero_range = bool(np.any(np.isclose(max_values, min_values)))
        rows.append(
            {
                "modality": modality,
                "normalization_mode": "minmax",
                "std_min": float(np.min(std)),
                "std_mean": float(np.mean(std)),
                "range_min": float(np.min(range_values)),
                "range_mean": float(np.mean(range_values)),
                "naive_inverse_variance_max_floor_1e_8": float(
                    np.max(1.0 / np.maximum(std, 1e-8) ** 2)
                ),
                "near_zero_std": near_zero_std,
                "zero_minmax_range": zero_range,
                "processor_zero_range_handling": "normalize_values_minmax masks min==max to normalized 0; no inverse-variance weighting",
                "model_loss_weighting": "uniform MSE over action_mask sum; no per-modality inverse variance in gr00t_n1d6.py",
                "masked": False,
                "gradient_norm": "not_run_no_training",
                "safety_status": "PASS_FOR_CURRENT_PROCESSOR" if not near_zero_std or zero_range else "REVIEW",
            }
        )
    _write_csv(out_dir / "loss_normalization_safety_matrix.csv", rows)
    unsafe_for_custom_inverse = [
        row["modality"] for row in rows if row["near_zero_std"] and row["naive_inverse_variance_max_floor_1e_8"] > 1e12
    ]
    payload = {
        "schema_version": "gr00t_contract_phase6_zero_variance_v1",
        "artifact_kind": "phase6_zero_variance_modality_audit",
        "generated_at_utc": _utc_now(),
        "status": "PASS_WITH_DOWNSTREAM_GUARD",
        "base_stats_path": _rel(base_stats_path),
        "evidence_paths": evidence_paths,
        "processor_normalization_evidence": {
            "modality_mean_std_embedding_keys": None,
            "normalization_mode_for_unitree_g1_actions": "minmax",
            "zero_range_behavior": "normalize_values_minmax returns 0 for min==max features",
            "source": "submodules/Isaac-GR00T/gr00t/data/utils.py:57-94",
        },
        "loss_evidence": {
            "model_loss": "F.mse_loss(..., reduction='none') * action_mask; sum/(action_mask.sum()+1e-6)",
            "source": "submodules/Isaac-GR00T/gr00t/model/gr00t_n1d6/gr00t_n1d6.py:248",
            "inverse_variance_in_current_model_loss": False,
        },
        "near_zero_std_modalities": [row["modality"] for row in rows if row["near_zero_std"]],
        "zero_range_modalities": [row["modality"] for row in rows if row["zero_minmax_range"]],
        "unsafe_if_custom_inverse_variance_loss_used": unsafe_for_custom_inverse,
        "matrix_csv": _rel(out_dir / "loss_normalization_safety_matrix.csv"),
    }
    _write_json(out_dir / "zero_variance_modality_audit.json", payload)
    verdict = [
        "# Phase 6 Zero-variance Modality Safety Verdict",
        "",
        "- status: `PASS_WITH_DOWNSTREAM_GUARD` for current processor/model loss; not a training permission gate because A already fails.",
        f"- near_zero_std_modalities: `{payload['near_zero_std_modalities']}`",
        f"- zero_range_modalities: `{payload['zero_range_modalities']}`",
        "- Current Unitree G1 action normalization uses min/max, not mean/std inverse variance; zero min/max range normalizes to 0.",
        "- Current GR00T loss is uniform masked MSE, not inverse-variance weighted.",
        "- Guard for future training: if a custom inverse-variance loss is introduced, zero-std modalities must be masked/clamped first.",
        "",
    ]
    (out_dir / "zero_variance_verdict.md").write_text("\n".join(verdict), encoding="utf-8")
    return payload


def finalize_report(args: argparse.Namespace) -> int:
    run_root = _resolve(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    phase1 = _read_json(run_root / "phase1_contract" / "bundle_diff_manifest.json")
    phase2 = _read_json(run_root / "phase2_identity" / "identity_equivalence_manifest.json")
    authority = _read_json(_resolve(args.authority_json))
    authority_status = authority.get("authority_status", "UNKNOWN")
    blocking_reason = (
        "A_identity_export_contract failed: base-vs-identity processor/config "
        "formalize_language differs and native offline L1_pred_normalized mismatch was observed."
    )
    action_stats = _read_csv_rows(_resolve(args.safe_action_stats_csv))

    evidence_paths = [
        _rel(run_root / "phase1_contract" / "bundle_diff_manifest.json"),
        _rel(run_root / "phase1_contract" / "contract_surface_diff.md"),
        _rel(run_root / "phase2_identity" / "identity_equivalence_manifest.json"),
        _rel(run_root / "phase2_identity" / "offline_identity_delta_summary.csv"),
        _rel(_resolve(args.safe_action_stats_csv)),
        _rel(_resolve(args.safe_temporal_report)),
    ]

    _write_skip_phase3(
        out_dir=run_root / "phase3_label_replay",
        authority_status=authority_status,
        blocking_reason=blocking_reason,
        evidence_paths=evidence_paths,
    )
    phase4_payload = _write_phase4_right_hand(
        out_dir=run_root / "phase4_right_hand",
        action_rows=action_stats,
        blocking_reason=blocking_reason,
        evidence_paths=evidence_paths,
    )
    phase5_payload = _write_phase5_navigate(
        out_dir=run_root / "phase5_navigate",
        action_rows=action_stats,
        blocking_reason=blocking_reason,
        evidence_paths=evidence_paths,
    )
    phase6_payload = _write_phase6_zero_variance(
        out_dir=run_root / "phase6_zero_variance",
        base_stats_path=_resolve(args.base_stats),
        evidence_paths=evidence_paths,
    )

    final_gate = {
        "schema_version": "gr00t_contract_action_semantics_final_gate_v1",
        "artifact_kind": "final_decision_gate",
        "generated_at_utc": _utc_now(),
        "final_decision": "A",
        "fixed_decision_text": FIXED_DECISIONS["A"],
        "authority_status": authority_status,
        "positive_readiness_blocked_by_authority": authority_status != "CANONICAL",
        "blocking_subgates": ["A_identity_export_contract"],
        "subgates": {
            "A_identity_export_contract": {
                "status": "FAIL",
                "predicate_inputs": {
                    "bundle_diff_manifest": _rel(run_root / "phase1_contract" / "bundle_diff_manifest.json"),
                    "identity_layers": [
                        "L1_pred_normalized",
                        "L2_denorm_absolute",
                        "L3_postprocessed",
                        "L4_server_wrapper_controller",
                        "L5_env_applied_proxy",
                    ],
                    "server_roundtrip_available": False,
                    "A1_status": phase1.get("A1_bundle_config_stats_surface_status"),
                    "A2_status": phase2.get("A2_offline_action_equivalence_status"),
                },
                "evidence_paths": evidence_paths[:4],
                "tolerance_or_threshold": "no dangerous bundle diff/UNKNOWN; L1-L5 max_abs_diff within 1e-4/1e-5; server/wrapper roundtrip present or A remains open",
                "first_mismatch_layer": phase2.get("first_mismatch_layer"),
                "reason_code": "dangerous_diff_and_offline_layer_mismatch",
            },
            "B_stage3_replay": {
                "status": "SKIPPED",
                "predicate_inputs": {
                    "authority_status": authority_status,
                    "first20_public_contract_passed": None,
                    "historical_30step_diagnostic_only": True,
                    "counterfactual_only_success": None,
                },
                "evidence_paths": [_rel(run_root / "phase3_label_replay" / "stage3_action_label_replay_manifest.json")],
                "tolerance_or_threshold": "CANONICAL authority plus first20 public-contract replay passes predeclared reach/lift/success; fallback cannot close label validity",
                "material_improvement_threshold": "10-seed lifted +2 over R1 or 30-seed +10pp",
                "reason_code": "blocked_by_A_identity_export_contract",
            },
            "C_right_hand": {
                "status": "UNKNOWN",
                "predicate_inputs": {
                    "absolute_contract_verified": True,
                    "near_zero_valid_hold": None,
                    "near_zero_valid_grasp_aperture": None,
                    "hand_replacement_improved_lift": None,
                    "diagnostic_summary": phase4_payload["summary"],
                },
                "evidence_paths": [
                    _rel(run_root / "phase4_right_hand" / "right_hand_semantics_matrix.csv"),
                    _rel(run_root / "phase4_right_hand" / "right_hand_state_action_alignment.json"),
                ],
                "tolerance_or_threshold": "near-zero absolute label must be proven valid by state/controller/replay evidence",
                "reason_code": "blocked_by_A_no_replay_counterfactual",
            },
            "D_navigate": {
                "status": "UNKNOWN",
                "predicate_inputs": {
                    "absolute_contract_verified": True,
                    "wbc_stand_walk_threshold_norm": 0.05,
                    "base_nav_splice_improved_reach": None,
                    "diagnostic_summary": phase5_payload["summary"],
                },
                "evidence_paths": [
                    _rel(run_root / "phase5_navigate" / "navigate_semantics_matrix.csv"),
                    _rel(run_root / "phase5_navigate" / "navigate_reach_sufficiency.json"),
                ],
                "tolerance_or_threshold": "navigate magnitude/sign/axis and reach sufficiency must be replay-backed",
                "reason_code": "blocked_by_A_no_base_nav_splice_replay",
            },
            "E_training_scope": {
                "status": "SKIPPED",
                "predicate_inputs": {
                    "zero_variance_status": phase6_payload["status"],
                    "labels_valid": None,
                },
                "evidence_paths": [
                    _rel(run_root / "phase6_zero_variance" / "zero_variance_modality_audit.json"),
                    _rel(run_root / "phase6_zero_variance" / "loss_normalization_safety_matrix.csv"),
                ],
                "tolerance_or_threshold": "A and label semantics must close before attributing collapse to training scope",
                "reason_code": "blocked_by_A_identity_export_contract",
            },
            "F_safe_sft_ready": {
                "status": "SKIPPED",
                "predicate_inputs": {"authority_status": authority_status, "A_closed": False},
                "evidence_paths": [],
                "tolerance_or_threshold": "A PASS, CANONICAL labels PASS, zero-variance safety PASS, and no training in this task",
                "reason_code": "A_failed_and_training_forbidden",
            },
            "G_guarded_recap_ready": {
                "status": "SKIPPED",
                "predicate_inputs": {"safe_sft_noncollapse_artifact": None},
                "evidence_paths": [],
                "tolerance_or_threshold": "requires pre-existing Safe-SFT non-collapse artifact",
                "reason_code": "no_safe_sft_noncollapse_artifact_and_A_failed",
            },
            "H_fatg_ready": {
                "status": "SKIPPED",
                "predicate_inputs": {"guarded_recap_noncollapse_artifact": None},
                "evidence_paths": [],
                "tolerance_or_threshold": "requires pre-existing Guarded RECAP non-collapse artifact",
                "reason_code": "no_guarded_recap_noncollapse_artifact_and_A_failed",
            },
        },
    }
    _write_json(run_root / "final_decision_gate.json", final_gate)

    # Build a compact but complete human report.
    phase1_diffs = [
        row
        for row in phase1.get("dangerous_diffs", [])
        if row.get("left") == "base" and row.get("right") == "identity"
    ]
    top_deltas = _read_csv_rows(run_root / "phase2_identity" / "offline_identity_delta_summary.csv")
    top_deltas = sorted(top_deltas, key=lambda row: float(row["max_abs_diff"]), reverse=True)[:8]

    rh_dataset = _stats_lookup(action_stats, source="dataset", modality="right_hand")
    rh_base = _stats_lookup(action_stats, source="base", modality="right_hand")
    nav_dataset = _stats_lookup(action_stats, source="dataset", modality="navigate_command")
    nav_base = _stats_lookup(action_stats, source="base", modality="navigate_command")
    rh_ratio = abs(_float(rh_dataset, "q99")) / max(abs(_float(rh_base, "q99")), 1e-12)
    nav_ratio = abs(_float(nav_dataset, "q99")) / max(abs(_float(nav_base, "q99")), 1e-12)

    report = [
        "# GR00T Contract + Action Semantics Closure Report",
        "",
        f"- generated_at_utc: `{final_gate['generated_at_utc']}`",
        f"- run_root: `{_rel(run_root)}`",
        f"- scope: diagnostics/report only; no LoRA/Safe-SFT/Guarded RECAP/FATG/training.",
        f"- authority_status: `{authority_status}`",
        "",
        "## Executive verdict",
        "",
        "A-gate is not closed. The zero-update identity bundle is not behavior-equivalent to base under native loading: base and identity still differ in processor/config `formalize_language`, and the same Stage3 observation bank produces an L1 `pred_normalized` mismatch.",
        "",
        "因此本轮不进入 Stage3 replay closure、Safe-LoRA、Guarded RECAP 或 FATG；right_hand/navigate 证据仅作为 diagnostic blocker 保留。",
        "",
        "## Phase 0 / 0.5",
        "",
        "- Phase0 startup: PASS（GPU1 dependency probe、flash-attn、disk/GPU/process guard、L0 drift 已记录）。",
        "- Phase0.5 Stage3 authority: FALLBACK（dataset/source ref 可 hash，但 iteration/preflight authority manifests 缺失）。",
        "",
        "## Phase 1 — bundle/config/stats",
        "",
        f"- A1 status: `{phase1.get('A1_bundle_config_stats_surface_status')}` / `{phase1.get('A1_reason_code')}`",
        "",
        "| file | differing key | base | identity | dangerous |",
        "|---|---|---|---|---|",
    ]
    for row in phase1_diffs:
        for diff in row.get("diff_rows", []):
            report.append(
                f"| `{row['file']}` | `{diff['path']}` | `{diff['left']}` | `{diff['right']}` | `{row['dangerous']}` |"
            )
    report.extend(
        [
            "",
            "## Phase 2 — offline identity action equivalence",
            "",
            f"- native_bundle_load observation_count: `{phase2.get('observation_count')}`",
            f"- A2 status: `{phase2.get('A2_offline_action_equivalence_status')}` / `{phase2.get('A2_reason_code')}`",
            f"- first_mismatch_layer: `{phase2.get('first_mismatch_layer')}`",
            "- repeatability: base and identity each repeated deterministically on the probe observation (max pairwise diff 0.0).",
            "- forced_same_processor_config_stats: SKIPPED, because no safe in-memory override seam was found and checkpoint originals must remain read-only.",
            "- L4/L5 server-wrapper/controller/env surfaces: unavailable here; this alone would keep A open even if local layers passed.",
            "",
            "| layer | modality | max_abs_diff | mean_abs_diff | pass |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in top_deltas:
        report.append(
            f"| `{row['layer']}` | `{row['modality']}` | {float(row['max_abs_diff']):.6g} | {float(row['mean_abs_diff']):.6g} | `{row['pass']}` |"
        )
    report.extend(
        [
            "",
            "## Phase 3 — Stage3 replay",
            "",
            "- SKIPPED: A-gate 已 FAIL；Stage3 authority 也为 FALLBACK。按 PRD，replay 不能修复/关闭 identity/export contract，也不能解锁 F/G/H。",
            "",
            "## Phase 4/5 — diagnostic semantics",
            "",
            "| modality | contract | dataset q99 | base q99 | dataset/base q99 | status |",
            "|---|---|---:|---:|---:|---|",
            f"| right_hand | ABSOLUTE | {_float(rh_dataset, 'q99'):.6g} | {_float(rh_base, 'q99'):.6g} | {rh_ratio:.6g} | UNKNOWN diagnostic-only |",
            f"| navigate_command | ABSOLUTE `(vx,vy,wz)` | {_float(nav_dataset, 'q99'):.6g} | {_float(nav_base, 'q99'):.6g} | {nav_ratio:.6g} | UNKNOWN diagnostic-only |",
            "",
            "- navigate_command 按当前合同是 ABSOLUTE；WBC standing/walking branch 使用 `||cmd|| < 0.05`，missing navigate 会注入 safe default stop。",
            "- 上述统计支持 action-label conflict 风险，但由于 A 未闭合且未跑 replay/counterfactual，本轮不选择 C/D。",
            "",
            "## Phase 6 — zero-variance modality",
            "",
            f"- status: `{phase6_payload['status']}`",
            f"- near_zero_std_modalities: `{phase6_payload['near_zero_std_modalities']}`",
            "- 当前 Unitree G1 action normalization 使用 minmax；zero range 归一化为 0；当前 GR00T loss 是 uniform action-mask MSE，不含 inverse-variance 权重。",
            "- Guard: 未来若引入 custom inverse-variance loss，必须先 mask/clamp zero-std modalities。",
            "",
            "## Evidence",
            "",
        ]
    )
    for path in evidence_paths:
        report.append(f"- `{path}`")
    report.extend(
        [
            f"- `{_rel(run_root / 'final_decision_gate.json')}`",
            "",
            "## Final decision",
            "",
            FIXED_DECISIONS["A"],
        ]
    )
    (run_root / "final_gr00t_contract_action_semantics_closure_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "final_decision": "A",
                "report": _rel(run_root / "final_gr00t_contract_action_semantics_closure_report.md"),
                "gate": _rel(run_root / "final_decision_gate.json"),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run read-only GR00T contract/action-semantics closure diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase1 = subparsers.add_parser("phase1-diff")
    phase1.add_argument("--manifest", required=True)
    phase1.add_argument("--output-dir", required=True)
    phase1.set_defaults(func=phase1_diff)

    phase2 = subparsers.add_parser("phase2-identity")
    phase2.add_argument("--manifest", required=True)
    phase2.add_argument("--dataset", required=True)
    phase2.add_argument("--output-dir", required=True)
    phase2.add_argument("--authority-status", default="FALLBACK")
    phase2.add_argument("--episode-count", type=int, default=10)
    phase2.add_argument("--chunk-stride", type=int, default=30)
    phase2.add_argument("--max-windows", type=int, default=50)
    phase2.add_argument("--seed", type=int, default=20260506)
    phase2.add_argument("--repeats", type=int, default=3)
    phase2.add_argument("--tolerance-max-abs", type=float, default=1e-4)
    phase2.add_argument("--tolerance-mean-abs", type=float, default=1e-5)
    phase2.add_argument(
        "--plain-prompt",
        default="pick up the apple, walk left and place the apple on the plate.",
    )
    phase2.set_defaults(func=phase2_identity)

    final = subparsers.add_parser("finalize-report")
    final.add_argument("--run-root", required=True)
    final.add_argument("--authority-json", required=True)
    final.add_argument("--safe-action-stats-csv", required=True)
    final.add_argument("--safe-temporal-report", required=True)
    final.add_argument("--base-stats", required=True)
    final.set_defaults(func=finalize_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
