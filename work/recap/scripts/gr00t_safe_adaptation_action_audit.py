#!/usr/bin/env python3
"""Offline action audit for the GR00T safe-adaptation substrate report.

This script compares Stage3 demo action labels against checkpoint predictions on
the same LeRobot observations.  It is intentionally read-only with respect to
model/data artifacts: outputs are written to a caller-provided artifact
directory, while checkpoints and datasets are only loaded.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_GR00T_ROOT = REPO_ROOT / "submodules" / "Isaac-GR00T"
if str(ISAAC_GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAAC_GR00T_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODALITIES: tuple[str, ...] = (
    "base_height_command",
    "navigate_command",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
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
        return str(path.relative_to(REPO_ROOT))
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


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _stats(values: np.ndarray, *, source: str, modality: str, surface: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.reshape(-1)
    if flat.size == 0:
        raise ValueError(f"empty values for {source}/{surface}/{modality}")
    signed_sum = arr.reshape((-1, arr.shape[-1])).sum(axis=-1)
    return {
        "source": source,
        "surface": surface,
        "modality": modality,
        "sample_shape": list(arr.shape),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "q01": float(np.quantile(flat, 0.01)),
        "q50": float(np.quantile(flat, 0.50)),
        "q99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
        "abs_mean": float(np.mean(np.abs(flat))),
        "abs_q99": float(np.quantile(np.abs(flat), 0.99)),
        "signed_sum_mean": float(np.mean(signed_sum)),
        "nonzero_frac": float(np.mean(np.abs(flat) > 1e-8)),
    }


def _load_stage3_loader(dataset: Path) -> Any:
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    return LeRobotEpisodeLoader(
        dataset,
        modality_configs=MODALITY_CONFIGS["unitree_g1"],
        video_backend="torchcodec",
    )


def _extract_policy_observation(
    *,
    loader: Any,
    trajectory: Any,
    step_index: int,
    plain_prompt: str,
) -> dict[str, Any]:
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.open_loop_eval import parse_observation_gr00t

    modality_configs = dict(loader.modality_configs)
    modality_configs.pop("action")
    step_data = extract_step_data(
        trajectory,
        int(step_index),
        modality_configs,
        EmbodimentTag("unitree_g1"),
    )

    obs: dict[str, Any] = {}
    for key, value in step_data.states.items():
        obs[f"state.{key}"] = value
    for key, value in step_data.images.items():
        obs[f"video.{key}"] = np.asarray(value)
    for language_key in loader.modality_configs["language"].modality_keys:
        # Use the formal-eval omit/plain text for all policies so RECAP and
        # pure SFT are compared on the same observation/language surface.
        obs[language_key] = plain_prompt
    return parse_observation_gr00t(obs, loader.modality_configs)


def _split_normalized_action(policy: Any, normalized_action: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    start_idx = 0
    modality_configs = policy.modality_configs["action"]
    joint_groups = modality_configs.modality_keys
    action_horizon = len(modality_configs.delta_indices)
    norm_params = policy.processor.state_action_processor.norm_params[policy.embodiment_tag.value]
    for key in joint_groups:
        joint_dim = int(norm_params["action"][key]["dim"].item())
        out[key] = normalized_action[..., :action_horizon, start_idx : start_idx + joint_dim]
        start_idx += joint_dim
    return out


def _predict_one_chunk(policy: Any, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    import torch
    from gr00t.data.types import MessageType

    unbatched_observations = policy._unbatch_observation(observation)
    processed_inputs = []
    states = []
    for obs in unbatched_observations:
        vla_step_data = policy._to_vla_step_data(obs)
        states.append(vla_step_data.states)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))

    collated_inputs = policy.collate_fn(processed_inputs)
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)
    with torch.inference_mode():
        model_pred = policy.model.get_action(**collated_inputs)
    normalized = model_pred["action_pred"].float().cpu().numpy()
    split_normalized = _split_normalized_action(policy, normalized)

    batched_states = {}
    for key in policy.modality_configs["state"].modality_keys:
        batched_states[key] = np.stack([state[key] for state in states], axis=0)
    denorm = policy.processor.decode_action(
        normalized,
        policy.embodiment_tag,
        batched_states,
    )
    denorm = {key: np.asarray(value, dtype=np.float32) for key, value in denorm.items()}
    split_normalized = {
        key: np.asarray(value, dtype=np.float32) for key, value in split_normalized.items()
    }
    return denorm, split_normalized


def _load_policy(checkpoint: Path) -> Any:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    return Gr00tPolicy(
        EmbodimentTag("unitree_g1"),
        str(checkpoint),
        device="cuda:0",
        strict=True,
    )


def _dataset_action_windows(
    *,
    trajectory: Any,
    starts: list[int],
    horizon: int,
) -> dict[str, list[np.ndarray]]:
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in MODALITIES}
    for start in starts:
        stop = min(int(start) + int(horizon), len(trajectory))
        for key in MODALITIES:
            values = [_as_array(value) for value in trajectory[f"action.{key}"].iloc[start:stop]]
            chunks[key].append(np.stack(values, axis=0))
    return chunks


def _state_windows(
    *,
    trajectory: Any,
    starts: list[int],
    horizon: int,
) -> dict[str, list[np.ndarray]]:
    state_modalities = (
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    )
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in state_modalities}
    for start in starts:
        stop = min(int(start) + int(horizon), len(trajectory))
        for key in state_modalities:
            values = [_as_array(value) for value in trajectory[f"state.{key}"].iloc[start:stop]]
            chunks[key].append(np.stack(values, axis=0))
    return chunks


def _concat_chunks(chunks: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def _collect_dataset_surfaces(
    *,
    loader: Any,
    episode_count: int,
    chunk_stride: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, int]]]:
    action_chunks: dict[str, list[np.ndarray]] = {key: [] for key in MODALITIES}
    state_chunks: dict[str, list[np.ndarray]] = {}
    windows: list[dict[str, int]] = []
    horizon = len(loader.modality_configs["action"].delta_indices)
    for episode_index in range(min(int(episode_count), len(loader))):
        trajectory = loader[episode_index]
        starts = list(range(0, len(trajectory), int(chunk_stride)))
        for start in starts:
            windows.append(
                {
                    "episode_index": int(episode_index),
                    "start_step": int(start),
                    "stop_step": int(min(start + horizon, len(trajectory))),
                }
            )
        per_ep_actions = _dataset_action_windows(
            trajectory=trajectory,
            starts=starts,
            horizon=horizon,
        )
        per_ep_states = _state_windows(
            trajectory=trajectory,
            starts=starts,
            horizon=horizon,
        )
        for key, value in per_ep_actions.items():
            action_chunks[key].extend(value)
        for key, value in per_ep_states.items():
            state_chunks.setdefault(key, []).extend(value)
    return _concat_chunks(action_chunks), _concat_chunks(state_chunks), windows


def _predict_policy_source(
    *,
    source: str,
    checkpoint: Path,
    loader: Any,
    episode_count: int,
    chunk_stride: int,
    plain_prompt: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    import torch

    horizon = len(loader.modality_configs["action"].delta_indices)
    policy = _load_policy(checkpoint)
    denorm_chunks: dict[str, list[np.ndarray]] = {key: [] for key in MODALITIES}
    norm_chunks: dict[str, list[np.ndarray]] = {key: [] for key in MODALITIES}
    records: list[dict[str, Any]] = []
    try:
        for episode_index in range(min(int(episode_count), len(loader))):
            trajectory = loader[episode_index]
            for start in range(0, len(trajectory), int(chunk_stride)):
                observation = _extract_policy_observation(
                    loader=loader,
                    trajectory=trajectory,
                    step_index=start,
                    plain_prompt=plain_prompt,
                )
                denorm, normalized = _predict_one_chunk(policy, observation)
                for key in MODALITIES:
                    denorm_chunks[key].append(denorm[key][0])
                    norm_chunks[key].append(normalized[key][0])
                records.append(
                    {
                        "source": source,
                        "episode_index": int(episode_index),
                        "start_step": int(start),
                        "horizon": int(horizon),
                    }
                )
    finally:
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _concat_chunks(denorm_chunks), _concat_chunks(norm_chunks), records


def _write_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "modality",
        "mean",
        "std",
        "min",
        "q01",
        "q50",
        "q99",
        "max",
        "signed_sum_mean",
        "nonzero_frac",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _loss_weighting_rows(stats_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(stats_json.read_text(encoding="utf-8"))
    action_stats = payload["action"]
    rows: list[dict[str, Any]] = []
    modality_ranges = {
        "base_height_command": (0, 1),
        "left_arm": (1, 8),
        "left_hand": (8, 15),
        "navigate_command": (15, 18),
        "right_arm": (18, 25),
        "right_hand": (25, 32),
        "waist": (32, 35),
    }
    for key, (start, end) in modality_ranges.items():
        std = np.asarray(action_stats["std"][start:end], dtype=np.float64)
        rows.append(
            {
                "modality": key,
                "std_mean": float(np.mean(std)),
                "std_min": float(np.min(std)),
                "std_max": float(np.max(std)),
                "inverse_variance_mean": float(np.mean(1.0 / np.maximum(std, 1e-8) ** 2)),
                "has_near_zero_std": bool(np.any(std < 1e-6)),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage3 dataset-vs-policy action audit for GR00T safe adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--pure-sft-checkpoint", required=True)
    parser.add_argument("--recap-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-count", type=int, default=10)
    parser.add_argument("--chunk-stride", type=int, default=30)
    parser.add_argument(
        "--plain-prompt",
        default="pick up the apple, walk left and place the apple on the plate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = _resolve(args.dataset)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = _load_stage3_loader(dataset)
    horizon = len(loader.modality_configs["action"].delta_indices)
    if int(args.chunk_stride) != horizon:
        print(
            f"[WARN] chunk_stride={args.chunk_stride} differs from policy horizon={horizon}; "
            "outputs keep both fields explicit.",
            flush=True,
        )

    dataset_actions, dataset_states, windows = _collect_dataset_surfaces(
        loader=loader,
        episode_count=args.episode_count,
        chunk_stride=args.chunk_stride,
    )
    source_surfaces: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "dataset": {"denorm_absolute": dataset_actions}
    }
    prediction_records: list[dict[str, Any]] = []
    for source, checkpoint_arg in (
        ("base", args.base_checkpoint),
        ("pure-SFT", args.pure_sft_checkpoint),
        ("RECAP", args.recap_checkpoint),
    ):
        print(f"[POLICY_START] source={source} checkpoint={checkpoint_arg}", flush=True)
        denorm, normalized, records = _predict_policy_source(
            source=source,
            checkpoint=_resolve(checkpoint_arg),
            loader=loader,
            episode_count=args.episode_count,
            chunk_stride=args.chunk_stride,
            plain_prompt=args.plain_prompt,
        )
        source_surfaces[source] = {
            "denorm_absolute": denorm,
            "pred_normalized": normalized,
        }
        prediction_records.extend(records)
        print(f"[POLICY_DONE] source={source} chunks={len(records)}", flush=True)

    table_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    for source, surfaces in source_surfaces.items():
        for surface, by_modality in surfaces.items():
            for modality in MODALITIES:
                row = _stats(
                    by_modality[modality],
                    source=source,
                    modality=modality,
                    surface=surface,
                )
                detailed_rows.append(row)
                if surface == "denorm_absolute":
                    table_rows.append(row)

    _write_stats_csv(out_dir / "dataset_vs_base_action_stats.csv", table_rows)
    detailed_csv = out_dir / "dataset_vs_base_action_stats_by_surface.csv"
    with detailed_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow({**row, "sample_shape": json.dumps(row["sample_shape"])})

    loss_weighting = _loss_weighting_rows(dataset / "meta" / "stats.json")
    answers = {
        "dataset_right_hand_near_zero": bool(
            next(
                row
                for row in table_rows
                if row["source"] == "dataset" and row["modality"] == "right_hand"
            )["abs_q99"]
            < 0.02
        ),
        "base_outputs_large_right_hand_on_same_observations": bool(
            next(
                row
                for row in table_rows
                if row["source"] == "base" and row["modality"] == "right_hand"
            )["abs_q99"]
            > 0.2
        ),
        "fine_tuned_right_hand_compressed_to_dataset_like": None,
        "navigate_weakened": None,
        "representation_mismatch_risk": (
            "present: arm/navigate groups are RELATIVE in the GR00T action config while "
            "right_hand is ABSOLUTE; this audit reports policy denorm_absolute after "
            "relative-to-absolute decode and pred_normalized separately."
        ),
        "low_variance_loss_weighting_risk": [
            row for row in loss_weighting if row["has_near_zero_std"]
        ],
    }
    dataset_rh = next(
        row
        for row in table_rows
        if row["source"] == "dataset" and row["modality"] == "right_hand"
    )
    puresft_rh = next(
        row
        for row in table_rows
        if row["source"] == "pure-SFT" and row["modality"] == "right_hand"
    )
    recap_rh = next(
        row
        for row in table_rows
        if row["source"] == "RECAP" and row["modality"] == "right_hand"
    )
    base_rh = next(
        row for row in table_rows if row["source"] == "base" and row["modality"] == "right_hand"
    )
    dataset_nav = next(
        row
        for row in table_rows
        if row["source"] == "dataset" and row["modality"] == "navigate_command"
    )
    base_nav = next(
        row
        for row in table_rows
        if row["source"] == "base" and row["modality"] == "navigate_command"
    )
    puresft_nav = next(
        row
        for row in table_rows
        if row["source"] == "pure-SFT" and row["modality"] == "navigate_command"
    )
    recap_nav = next(
        row
        for row in table_rows
        if row["source"] == "RECAP" and row["modality"] == "navigate_command"
    )
    answers["fine_tuned_right_hand_compressed_to_dataset_like"] = {
        "pure_SFT_abs_q99_over_base": float(puresft_rh["abs_q99"] / max(base_rh["abs_q99"], 1e-12)),
        "RECAP_abs_q99_over_base": float(recap_rh["abs_q99"] / max(base_rh["abs_q99"], 1e-12)),
        "dataset_abs_q99_over_base": float(dataset_rh["abs_q99"] / max(base_rh["abs_q99"], 1e-12)),
    }
    answers["navigate_weakened"] = {
        "pure_SFT_abs_q99_over_base": float(puresft_nav["abs_q99"] / max(base_nav["abs_q99"], 1e-12)),
        "RECAP_abs_q99_over_base": float(recap_nav["abs_q99"] / max(base_nav["abs_q99"], 1e-12)),
        "dataset_abs_q99_over_base": float(dataset_nav["abs_q99"] / max(base_nav["abs_q99"], 1e-12)),
    }

    report_md = [
        "# Phase 2 Dataset-vs-policy Action Audit",
        "",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- dataset: `{_rel(dataset)}`",
        f"- episodes: `{min(int(args.episode_count), len(loader))}`",
        f"- policy_horizon: `{horizon}`",
        f"- chunk_stride: `{int(args.chunk_stride)}`",
        f"- windows: `{len(windows)}`",
        "",
        "## Required answers",
        f"1. dataset right_hand near-zero: `{answers['dataset_right_hand_near_zero']}`",
        f"2. base outputs large right_hand: `{answers['base_outputs_large_right_hand_on_same_observations']}`",
        f"3. fine-tuned right_hand ratios: `{answers['fine_tuned_right_hand_compressed_to_dataset_like']}`",
        f"4. navigate ratios: `{answers['navigate_weakened']}`",
        f"5. representation mismatch risk: {answers['representation_mismatch_risk']}",
        f"6. low-variance modality risk: `{answers['low_variance_loss_weighting_risk']}`",
        "",
        "Primary table is `dataset_vs_base_action_stats.csv`; surface-expanded table is `dataset_vs_base_action_stats_by_surface.csv`.",
    ]
    (out_dir / "modality_answer_matrix.md").write_text("\n".join(report_md) + "\n", encoding="utf-8")

    payload = {
        "schema_version": "gr00t_safe_adaptation_dataset_vs_policy_action_audit_v1",
        "artifact_kind": "dataset_vs_base_action_audit",
        "generated_at_utc": _utc_now(),
        "dataset": _rel(dataset),
        "episode_count": min(int(args.episode_count), len(loader)),
        "policy_horizon": int(horizon),
        "chunk_stride": int(args.chunk_stride),
        "windows": windows,
        "prediction_records": prediction_records,
        "stats_table_csv": _rel(out_dir / "dataset_vs_base_action_stats.csv"),
        "stats_by_surface_csv": _rel(detailed_csv),
        "modality_answer_matrix_md": _rel(out_dir / "modality_answer_matrix.md"),
        "answers": answers,
        "loss_weighting_rows": loss_weighting,
        "surface_notes": {
            "dataset.denorm_absolute": "raw Stage3 action labels from LeRobot rows",
            "policy.pred_normalized": "model action_pred split by modality before processor.decode_action",
            "policy.denorm_absolute": "processor.decode_action output after denormalization and relative-to-absolute conversion",
            "controller_input": "not available in offline LeRobot demo observations; Phase 3 rollout telemetry covers executed action summaries",
            "horizon_vs_execute": "offline audit uses policy_horizon chunks; formal rollout execute steps are reported separately in Phase 1/3",
        },
    }
    _write_json(out_dir / "dataset_vs_base_action_audit.json", payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": _rel(out_dir),
                "stats_csv": _rel(out_dir / "dataset_vs_base_action_stats.csv"),
                "audit_json": _rel(out_dir / "dataset_vs_base_action_audit.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
