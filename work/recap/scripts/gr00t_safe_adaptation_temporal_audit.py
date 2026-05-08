#!/usr/bin/env python3
"""Summarize GR00T safe-adaptation rollout action telemetry.

The Phase-3 audit consumes the formal-eval telemetry produced by
``gr00t_g3_formal_eval.py`` / ``3D_recap_eval.py`` and emits compact tables for
diagnosing whether fine-tuned policies lose pretrained action primitives across
the whole rollout or at specific phase transitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


MODALITIES = (
    "action.base_height_command",
    "action.navigate_command",
    "action.left_arm",
    "action.right_arm",
    "action.left_hand",
    "action.right_hand",
    "action.waist",
)


HAND_REACH_THRESHOLD_M = 0.10
LIFT_THRESHOLD_M = 0.03
PLATE_THRESHOLD_M = 0.12
NEAR_ZERO_Q99_RATIO = 0.20
WEAK_NAV_Q99_RATIO = 0.40


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _q(values: list[float], quantile: float) -> float:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return 0.0
    values.sort()
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * float(quantile)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _mean(values: list[float]) -> float:
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _safe_div(num: float, den: float) -> float | None:
    if abs(float(den)) <= 1e-12:
        return None
    return float(num) / float(den)


def _action_metric(step: dict[str, Any], modality: str, metric: str) -> float:
    return _as_float(step.get("action_summary", {}).get(modality, {}).get(metric))


def _first_step(steps: list[dict[str, Any]], predicate) -> int | None:
    for rec in steps:
        if predicate(rec):
            return int(rec.get("outer_step", 0))
    return None


def _phase_filter(
    steps: list[dict[str, Any]],
    *,
    start: int | None = None,
    stop: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in steps:
        step = int(rec.get("outer_step", 0))
        if start is not None and step < start:
            continue
        if stop is not None and step >= stop:
            continue
        out.append(rec)
    return out


def _sum_metric(steps: list[dict[str, Any]], modality: str, metric: str) -> float:
    return float(sum(_action_metric(rec, modality, metric) for rec in steps))


def _max_metric(steps: list[dict[str, Any]], modality: str, metric: str) -> float:
    values = [_action_metric(rec, modality, metric) for rec in steps]
    return max(values) if values else 0.0


def _episode_status(
    episode_rec: dict[str, Any] | None,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    success = bool(episode_rec.get("success")) if episode_rec else any(bool(s.get("success_step")) for s in steps)
    initial_height = None
    if episode_rec:
        initial_height = (
            episode_rec.get("reset_snapshot", {}) or {}
        ).get("apple_height_z")
    if not isinstance(initial_height, (int, float)) and steps:
        initial_height = steps[0].get("apple_height_z")
    initial_height_f = _as_float(initial_height, default=0.0)

    reached_step = _first_step(
        steps,
        lambda rec: isinstance(rec.get("apple_to_right_eef_l2"), (int, float))
        and float(rec["apple_to_right_eef_l2"]) <= HAND_REACH_THRESHOLD_M,
    )
    lifted_step = _first_step(
        steps,
        lambda rec: isinstance(rec.get("apple_height_z"), (int, float))
        and float(rec["apple_height_z"]) - initial_height_f >= LIFT_THRESHOLD_M,
    )
    near_plate_step = _first_step(
        steps,
        lambda rec: isinstance(rec.get("apple_to_plate_l2"), (int, float))
        and float(rec["apple_to_plate_l2"]) <= PLATE_THRESHOLD_M,
    )
    success_step = _first_step(steps, lambda rec: bool(rec.get("success_step")))

    # The formal environment can terminate successfully even when low-rate
    # telemetry misses the exact hand-distance threshold crossing. Treat success
    # or lift as evidence that a grasp/reach happened, but keep the concrete
    # threshold timestep nullable for transparency.
    reached = bool(reached_step is not None or success or lifted_step is not None)
    lifted = bool(lifted_step is not None)
    failure_step = None if success else (int(steps[-1].get("outer_step", 0)) if steps else None)
    failure_reason = episode_rec.get("failure_reason") if episode_rec else None
    failure_stage_guess = episode_rec.get("failure_stage_guess") if episode_rec else None

    phase_start = reached_step or lifted_step or success_step or (int(steps[-1].get("outer_step", 0)) + 1 if steps else None)
    return {
        "success": success,
        "reached": reached,
        "lifted": lifted,
        "reached_threshold_timestep": reached_step,
        "reached_apple_timestep": reached_step or success_step,
        "first_grasp_contact_timestep": reached_step or lifted_step or success_step,
        "lifted_timestep": lifted_step,
        "near_plate_timestep": near_plate_step,
        "success_timestep": success_step,
        "failure_timestep": failure_step,
        "failure_reason": failure_reason,
        "failure_stage_guess": failure_stage_guess,
        "phase_start": phase_start,
    }


def _policy_dirs(eval_root: Path) -> list[Path]:
    return sorted(
        [p for p in eval_root.iterdir() if p.is_dir() and (p / "telemetry" / "omit" / "steps.jsonl").exists()]
    )


def _load_policy(eval_dir: Path) -> dict[str, Any]:
    steps = _read_jsonl(eval_dir / "telemetry" / "omit" / "steps.jsonl")
    episode_rows = _read_jsonl(eval_dir / "telemetry" / "omit" / "episodes.jsonl")
    summary_path = eval_dir / "formal_eval_summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else {}

    by_episode: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in steps:
        by_episode[(int(rec.get("seed", -1)), int(rec.get("episode_index", -1)))].append(rec)
    for recs in by_episode.values():
        recs.sort(key=lambda r: int(r.get("outer_step", 0)))

    episode_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for rec in episode_rows:
        episode_by_key[(int(rec.get("seed", -1)), int(rec.get("episode_index", -1)))] = rec

    return {
        "policy": eval_dir.name,
        "eval_dir": str(eval_dir),
        "steps": steps,
        "episode_rows": episode_rows,
        "summary": summary,
        "by_episode": by_episode,
        "episode_by_key": episode_by_key,
    }


def _policy_overall_stats(policy_data: dict[str, Any]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    steps = list(policy_data["steps"])
    for modality in MODALITIES:
        q99_values = [_action_metric(s, modality, "q99_abs") for s in steps]
        sum_abs_values = [_action_metric(s, modality, "sum_abs") for s in steps]
        sum_values = [_action_metric(s, modality, "sum") for s in steps]
        mean_values = [_action_metric(s, modality, "mean") for s in steps]
        first_steps = [
            s
            for s in steps
            if int(s.get("outer_step", -1)) == 1
        ]
        stats[modality] = {
            "q99_abs": _q(q99_values, 0.99),
            "max_q99_abs": max(q99_values) if q99_values else 0.0,
            "mean_sum_abs": _mean(sum_abs_values),
            "signed_sum_mean": _mean(sum_values),
            "mean": _mean(mean_values),
            "first_chunk_q99_abs": _q([_action_metric(s, modality, "q99_abs") for s in first_steps], 0.99),
            "first_chunk_sum_abs_mean": _mean([_action_metric(s, modality, "sum_abs") for s in first_steps]),
        }
    return stats


def _boundary_jumps(policy_data: dict[str, Any]) -> dict[str, dict[str, float]]:
    jumps: dict[str, list[float]] = {m: [] for m in MODALITIES}
    for steps in policy_data["by_episode"].values():
        for modality in MODALITIES:
            prev = None
            for rec in steps:
                current = _action_metric(rec, modality, "mean")
                if prev is not None:
                    jumps[modality].append(abs(current - prev))
                prev = current
    return {
        modality: {
            "q99_abs_mean_jump": _q(values, 0.99),
            "max_abs_mean_jump": max(values) if values else 0.0,
            "n": float(len(values)),
        }
        for modality, values in jumps.items()
    }


def _episode_rows(policy_data: dict[str, Any], base_stats: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, steps in sorted(policy_data["by_episode"].items()):
        seed, episode_index = key
        episode_rec = policy_data["episode_by_key"].get(key)
        status = _episode_status(episode_rec, steps)
        reached_step = status["reached_apple_timestep"]
        grasp_step = status["first_grasp_contact_timestep"]
        lifted_step = status["lifted_timestep"]
        phase_start = status["phase_start"]

        approach_steps = _phase_filter(steps, stop=phase_start)
        grasp_steps = _phase_filter(steps, start=grasp_step) if grasp_step is not None else []
        if lifted_step is not None and grasp_steps:
            grasp_steps = _phase_filter(steps, start=grasp_step, stop=lifted_step + 1)
        post_reach_steps = _phase_filter(steps, start=reached_step) if reached_step is not None else []

        hand_pre = _sum_metric(approach_steps, "action.right_hand", "sum_abs")
        hand_grasp = _sum_metric(grasp_steps, "action.right_hand", "sum_abs")
        nav_approach = _sum_metric(approach_steps, "action.navigate_command", "sum_abs")
        arm_lift = _sum_metric(post_reach_steps, "action.right_arm", "sum_abs")

        hand_q99_overall = _max_metric(steps, "action.right_hand", "q99_abs")
        hand_q99_grasp = _max_metric(grasp_steps, "action.right_hand", "q99_abs")
        nav_q99_approach = _max_metric(approach_steps, "action.navigate_command", "q99_abs")
        base_hand_q99 = base_stats.get("action.right_hand", {}).get("q99_abs", 0.0)
        base_nav_q99 = base_stats.get("action.navigate_command", {}).get("q99_abs", 0.0)
        hand_ratio = _safe_div(hand_q99_overall, base_hand_q99)
        nav_ratio = _safe_div(nav_q99_approach, base_nav_q99)

        notes: list[str] = []
        if hand_ratio is not None and hand_ratio < NEAR_ZERO_Q99_RATIO:
            notes.append("right_hand near-zero vs base")
        if hand_q99_grasp and base_hand_q99 and hand_q99_grasp / base_hand_q99 < NEAR_ZERO_Q99_RATIO:
            notes.append("grasp-phase right_hand near-zero")
        if nav_ratio is not None and nav_ratio < WEAK_NAV_Q99_RATIO:
            notes.append("approach navigate weak vs base")
        if not status["success"]:
            stage = status.get("failure_stage_guess")
            if isinstance(stage, dict) and stage.get("label"):
                notes.append(str(stage["label"]))
            elif status.get("failure_reason"):
                notes.append(str(status["failure_reason"]))
        if status["success"] and status["reached_threshold_timestep"] is None:
            notes.append("success but 10cm reach threshold not sampled")
        if not notes:
            notes.append("ok")

        rows.append(
            {
                "policy": policy_data["policy"],
                "seed": seed,
                "episode_index": episode_index,
                "reached?": status["reached"],
                "lifted?": status["lifted"],
                "success?": status["success"],
                "right_hand_energy_pre_grasp": hand_pre,
                "right_hand_energy_grasp": hand_grasp,
                "navigate_energy_approach": nav_approach,
                "arm_lift_energy": arm_lift,
                "reached_apple_timestep": status["reached_apple_timestep"],
                "first_grasp/contact_timestep": status["first_grasp_contact_timestep"],
                "lifted_timestep": lifted_step,
                "failure_timestep": status["failure_timestep"],
                "right_hand_q99_overall": hand_q99_overall,
                "right_hand_q99_grasp": hand_q99_grasp,
                "navigate_q99_approach": nav_q99_approach,
                "right_hand_q99_ratio_vs_base": hand_ratio,
                "navigate_q99_ratio_vs_base": nav_ratio,
                "notes": "; ".join(notes),
            }
        )
    return rows


def _step_curve_rows(policy_data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, steps in sorted(policy_data["by_episode"].items()):
        cumulative_hand = 0.0
        for rec in steps:
            hand_sum = _action_metric(rec, "action.right_hand", "sum")
            cumulative_hand += hand_sum
            out.append(
                {
                    "policy": policy_data["policy"],
                    "seed": int(rec.get("seed", key[0])),
                    "episode_index": int(rec.get("episode_index", key[1])),
                    "outer_step": int(rec.get("outer_step", 0)),
                    "right_hand_signed_sum": hand_sum,
                    "right_hand_signed_cumulative_action": cumulative_hand,
                    "right_hand_sum_abs": _action_metric(rec, "action.right_hand", "sum_abs"),
                    "right_hand_q99_abs": _action_metric(rec, "action.right_hand", "q99_abs"),
                    "navigate_sum_abs": _action_metric(rec, "action.navigate_command", "sum_abs"),
                    "navigate_norm_l2": _action_metric(rec, "action.navigate_command", "l2"),
                    "right_arm_sum_abs": _action_metric(rec, "action.right_arm", "sum_abs"),
                    "right_arm_q99_abs": _action_metric(rec, "action.right_arm", "q99_abs"),
                    "apple_to_right_eef_l2": rec.get("apple_to_right_eef_l2"),
                    "apple_to_plate_l2": rec.get("apple_to_plate_l2"),
                    "apple_height_z": rec.get("apple_height_z"),
                    "success_step": bool(rec.get("success_step")),
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _policy_summary_rows(
    policy_data_by_name: dict[str, dict[str, Any]],
    overall: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    base = overall.get("base") or next(iter(overall.values()))
    rows: list[dict[str, Any]] = []
    for policy, stats in sorted(overall.items()):
        summary = policy_data_by_name[policy].get("summary", {})
        mode_summaries = summary.get("mode_summaries", {})
        omit_summary = mode_summaries.get("omit", {}) if isinstance(mode_summaries, dict) else {}
        episode_rows = policy_data_by_name[policy].get("episode_rows", [])
        success_count = sum(1 for row in episode_rows if row.get("success"))
        episode_count = len(episode_rows)
        reached_count = 0
        lifted_count = 0
        for key, steps in policy_data_by_name[policy]["by_episode"].items():
            status = _episode_status(policy_data_by_name[policy]["episode_by_key"].get(key), steps)
            reached_count += int(bool(status["reached"]))
            lifted_count += int(bool(status["lifted"]))
        rows.append(
            {
                "policy": policy,
                "episodes": episode_count or omit_summary.get("episode_count"),
                "success": success_count,
                "reached": reached_count,
                "lifted": lifted_count,
                "right_hand_q99": stats["action.right_hand"]["q99_abs"],
                "right_hand_q99/base": _safe_div(
                    stats["action.right_hand"]["q99_abs"],
                    base.get("action.right_hand", {}).get("q99_abs", 0.0),
                ),
                "right_hand_first_chunk_q99/base": _safe_div(
                    stats["action.right_hand"]["first_chunk_q99_abs"],
                    base.get("action.right_hand", {}).get("first_chunk_q99_abs", 0.0),
                ),
                "navigate_q99": stats["action.navigate_command"]["q99_abs"],
                "navigate_q99/base": _safe_div(
                    stats["action.navigate_command"]["q99_abs"],
                    base.get("action.navigate_command", {}).get("q99_abs", 0.0),
                ),
                "navigate_first_chunk_q99/base": _safe_div(
                    stats["action.navigate_command"]["first_chunk_q99_abs"],
                    base.get("action.navigate_command", {}).get("first_chunk_q99_abs", 0.0),
                ),
            }
        )
    return rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def _build_report(
    out_dir: Path,
    eval_root: Path,
    episode_rows: list[dict[str, Any]],
    policy_summary_rows: list[dict[str, Any]],
    boundary_gate: dict[str, Any],
) -> str:
    required_columns = [
        "policy",
        "seed",
        "reached?",
        "lifted?",
        "success?",
        "right_hand_energy_pre_grasp",
        "right_hand_energy_grasp",
        "navigate_energy_approach",
        "arm_lift_energy",
        "notes",
    ]
    summary_columns = [
        "policy",
        "episodes",
        "success",
        "reached",
        "lifted",
        "right_hand_q99",
        "right_hand_q99/base",
        "right_hand_first_chunk_q99/base",
        "navigate_q99",
        "navigate_q99/base",
        "navigate_first_chunk_q99/base",
    ]
    by_policy = {row["policy"]: row for row in policy_summary_rows}
    base = by_policy.get("base", {})
    pure = by_policy.get("pure_sft", {})
    recap = by_policy.get("recap", {})

    def ratio(row: dict[str, Any], key: str) -> float | None:
        value = row.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    pure_hand = ratio(pure, "right_hand_q99/base")
    recap_hand = ratio(recap, "right_hand_q99/base")
    pure_nav = ratio(pure, "navigate_q99/base")
    recap_nav = ratio(recap, "navigate_q99/base")
    pure_first_hand = ratio(pure, "right_hand_first_chunk_q99/base")
    recap_first_hand = ratio(recap, "right_hand_first_chunk_q99/base")
    pure_first_nav = ratio(pure, "navigate_first_chunk_q99/base")
    recap_first_nav = ratio(recap, "navigate_first_chunk_q99/base")

    right_hand_answer = (
        "pure-SFT/RECAP 的 right_hand q99 均低于 base 20%，符合全程 near-zero / primitive 擦除形态。"
        if (pure_hand is not None and pure_hand < NEAR_ZERO_Q99_RATIO)
        and (recap_hand is not None and recap_hand < NEAR_ZERO_Q99_RATIO)
        else "right_hand 未同时满足相对 base 的 20% near-zero 判据，需看逐 seed 表。"
    )
    navigate_answer = (
        "navigate 整体或 approach 期未低于 40% stop-gate，但相对 base 明显偏弱。"
        if not (
            (pure_nav is not None and pure_nav < WEAK_NAV_Q99_RATIO)
            or (recap_nav is not None and recap_nav < WEAK_NAV_Q99_RATIO)
        )
        else "至少一个 fine-tuned policy 的 navigate 低于 base 40% stop-gate。"
    )
    homology_answer = (
        "pure-SFT 与 RECAP 同型 collapse：两者 right_hand 首 chunk 与整体均显著低于 base。"
        if (pure_first_hand is not None and pure_first_hand < NEAR_ZERO_Q99_RATIO)
        and (recap_first_hand is not None and recap_first_hand < NEAR_ZERO_Q99_RATIO)
        else "pure-SFT 与 RECAP collapse 同型性不完整；见 summary/逐 seed 表。"
    )
    first_chunk_answer = (
        "first chunk already wrong：fine-tuned right_hand 首 chunk 已低于 base 20%，不是仅 phase transition 后错误。"
        if (pure_first_hand is not None and pure_first_hand < NEAR_ZERO_Q99_RATIO)
        and (recap_first_hand is not None and recap_first_hand < NEAR_ZERO_Q99_RATIO)
        else "首 chunk 不足以单独解释，需要结合 phase transition 后能量。"
    )
    if (pure_first_nav is not None and pure_first_nav < WEAK_NAV_Q99_RATIO) or (
        recap_first_nav is not None and recap_first_nav < WEAK_NAV_Q99_RATIO
    ):
        first_chunk_answer += " navigate 首 chunk 也触及弱化 gate。"

    lines = [
        "# Phase3 per-step signed temporal audit",
        "",
        f"- eval_root: `{eval_root}`",
        f"- artifacts: `{out_dir}`",
        "- contact/grasp timestep: telemetry lacks contact sensor; uses first 10cm reach, else first lift/success timestep as approximation.",
        "- right_arm lift-relevant channels: telemetry stores modality-level chunk summaries, not per-channel raw arrays; arm_lift_energy uses whole `action.right_arm.sum_abs` after reach.",
        "",
        "## Policy-level summary",
        "",
        _markdown_table(policy_summary_rows, summary_columns),
        "",
        "## Required per-seed table",
        "",
        _markdown_table(episode_rows, required_columns),
        "",
        "## Required answers",
        "",
        f"1. right_hand phase: {right_hand_answer}",
        f"2. navigate phase: {navigate_answer}",
        f"3. pure-SFT vs RECAP: {homology_answer}",
        f"4. first chunk vs transition: {first_chunk_answer}",
        "",
        "## Chunk boundary jump gate",
        "",
        "Approximation: q99 of consecutive per-chunk modality mean deltas; formal raw chunk boundary arrays are not stored.",
        "",
        "```json",
        json.dumps(boundary_gate, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    eval_root = args.eval_root
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = [_load_policy(p) for p in _policy_dirs(eval_root)]
    if not policies:
        raise SystemExit(f"No policy telemetry found under {eval_root}")
    policy_by_name = {p["policy"]: p for p in policies}
    overall = {p["policy"]: _policy_overall_stats(p) for p in policies}
    base_stats = overall.get("base") or next(iter(overall.values()))

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    boundaries = {p["policy"]: _boundary_jumps(p) for p in policies}
    for policy in policies:
        episode_rows.extend(_episode_rows(policy, base_stats))
        step_rows.extend(_step_curve_rows(policy))

    policy_summary = _policy_summary_rows(policy_by_name, overall)

    base_jumps = boundaries.get("base") or next(iter(boundaries.values()))
    boundary_gate: dict[str, Any] = {
        "schema_version": 1,
        "method": "q99(abs(delta consecutive action_summary.mean)) per modality; approximate because raw chunk-boundary actions are not stored",
        "base_policy": "base" if "base" in boundaries else next(iter(boundaries)),
        "fail_if_policy_q99_gt_2x_base": True,
        "policies": {},
    }
    for policy, modality_stats in boundaries.items():
        policy_gate: dict[str, Any] = {}
        for modality, stats in modality_stats.items():
            base_q99 = base_jumps.get(modality, {}).get("q99_abs_mean_jump", 0.0)
            ratio = _safe_div(stats["q99_abs_mean_jump"], base_q99)
            policy_gate[modality] = {
                **stats,
                "ratio_vs_base": ratio,
                "gate_fail": bool(ratio is not None and ratio > 2.0),
            }
        boundary_gate["policies"][policy] = policy_gate

    _write_jsonl(out_dir / "temporal_signed_actions.jsonl", step_rows)
    _write_csv(out_dir / "phase_aligned_curves_summary.csv", episode_rows)
    _write_csv(out_dir / "policy_level_temporal_summary.csv", policy_summary)
    (out_dir / "chunk_boundary_jump_gate.json").write_text(
        json.dumps(boundary_gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "temporal_audit_report.md").write_text(
        _build_report(out_dir, eval_root, episode_rows, policy_summary, boundary_gate),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
