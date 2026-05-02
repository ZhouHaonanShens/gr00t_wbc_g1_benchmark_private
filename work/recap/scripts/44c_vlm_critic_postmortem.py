#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_JSON_REL = (
    "agent/artifacts/vlm_critic_offline_gate/task7_postmortem_diagnosis.json"
)
PASS_SENTINEL = "POSTMORTEM_OK"
FAIL_SENTINEL = "POSTMORTEM_INCONCLUSIVE"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _repo_root()


def _resolve_path(raw_path: str | None, *, default_rel: str) -> Path:
    value = str(raw_path or default_rel).strip()
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        _ = f.write("\n")
    _ = tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return cast(dict[str, object], obj)


def _as_dict(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected object ({context}), got {type(value).__name__}")
    return cast(dict[str, object], value)


def _as_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected float-like value ({context}), got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(
        f"Expected float-like value ({context}), got {type(value).__name__}"
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _line_number(path: Path, needle: str) -> int | None:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if needle in line:
                return int(line_no)
    return None


def _confidence(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _build_code_evidence(repo_root: Path) -> dict[str, dict[str, object]]:
    modeling_path = repo_root / "work" / "recap" / "critic_vlm" / "modeling.py"
    train_path = repo_root / "work" / "recap" / "critic_vlm" / "train.py"
    gate_path = repo_root / "work" / "recap" / "44_vlm_critic_offline_gate.py"
    build_path = repo_root / "work" / "recap" / "42_vlm_critic_dataset_build.py"

    prompt_line = _line_number(
        modeling_path, 'return f"Task: {prompt}\\n{DEFAULT_CRITIC_QUERY}"'
    )
    default_query_line = _line_number(
        modeling_path,
        'DEFAULT_CRITIC_QUERY = "Estimate the raw return of the current observation."',
    )
    zero_proprio_line = _line_number(gate_path, "proprio = torch.zeros(")
    eval_tnorm_line = _line_number(
        gate_path, "t_norm_values.append([float(float(row.t) / float(t_norm_den))])"
    )
    train_proprio_line = _line_number(
        train_path,
        "def resolve_sample_proprio(sample: Any, manifest: Any) -> list[float]:",
    )
    train_loader_prompt_line = _line_number(
        train_path, "text = _modeling_module().build_prompt_text("
    )
    sample_cap_line = _line_number(train_path, "limit=config.max_train_samples,")
    bin_index_line = _line_number(
        build_path,
        "target_bin_index = _to_bin_index(return_g, g_min=g_min, g_max=g_max, bins=bins)",
    )
    target_mode_line = _line_number(build_path, '"target_mode": "dist201_raw_return"')

    return {
        "prompt_template": {
            "path": str(modeling_path),
            "line": prompt_line,
            "default_query_line": default_query_line,
            "detail": "Prompt builder always emits DEFAULT_CRITIC_QUERY and prepends task text only when use_prompt=true.",
        },
        "eval_zero_proprio": {
            "path": str(gate_path),
            "line": zero_proprio_line,
            "detail": "Offline multimodal scoring injects zero proprio tensors instead of loading observation.state.",
        },
        "eval_t_norm": {
            "path": str(gate_path),
            "line": eval_tnorm_line,
            "detail": "Offline gate still keeps real t_norm in every ablation path.",
        },
        "train_real_proprio": {
            "path": str(train_path),
            "line": train_proprio_line,
            "detail": "Training loads real proprio from dataset side-channel status when available.",
        },
        "train_prompt_builder": {
            "path": str(train_path),
            "line": train_loader_prompt_line,
            "detail": "Training path uses the same prompt builder as inference/offline gate.",
        },
        "sample_caps": {
            "path": str(train_path),
            "line": sample_cap_line,
            "detail": "Trainer supports max_* sample caps that can materially limit Task 7 interpretation.",
        },
        "monotonic_target_binning": {
            "path": str(build_path),
            "line": bin_index_line,
            "target_mode_line": target_mode_line,
            "detail": "Dataset build maps return_G to dist201_raw_return bins monotonically; no explicit sign inversion is introduced there.",
        },
    }


def _extract_manifest_modes(
    formal_gate: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    modes_obj = _as_dict(formal_gate.get("modes"), context="formal_gate.modes")
    out: dict[str, dict[str, object]] = {}
    for mode_name in ("prompt_only", "vision_only", "full_input"):
        out[mode_name] = _as_dict(
            modes_obj.get(mode_name), context=f"modes.{mode_name}"
        )
    return out


def _load_manifest_input_modes(
    modes: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for mode_name, mode_obj in modes.items():
        manifest_path_raw = mode_obj.get("manifest_path")
        manifest_path = None
        if isinstance(manifest_path_raw, str) and manifest_path_raw.strip():
            manifest_path = Path(manifest_path_raw).expanduser().resolve()
        if manifest_path is None or not manifest_path.exists():
            out[mode_name] = _as_dict(
                mode_obj.get("input_mode"), context=f"{mode_name}.input_mode"
            )
            continue
        manifest_obj = _read_json(manifest_path)
        out[mode_name] = _as_dict(
            manifest_obj.get("input_mode"), context=f"manifest[{mode_name}].input_mode"
        )
        out[mode_name]["manifest_path"] = str(manifest_path)
    return out


def _build_evidence(
    *,
    formal_gate: Mapping[str, object],
    formal_ablation: Mapping[str, object],
    critic_provenance: Mapping[str, object],
    critic_metrics: Mapping[str, object],
    code_evidence: Mapping[str, Mapping[str, object]],
    manifest_inputs: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    modes = _extract_manifest_modes(formal_gate)
    prompt_mode = modes["prompt_only"]
    vision_mode = modes["vision_only"]
    full_mode = modes["full_input"]
    baseline_mode = _as_dict(
        _as_dict(formal_gate.get("modes"), context="modes").get("baseline_state_only"),
        context="baseline",
    )
    training_hparams = _as_dict(
        critic_provenance.get("training_hparams"), context="training_hparams"
    )
    train_manifest_summary = _as_dict(
        critic_provenance.get("train_manifest_summary"),
        context="train_manifest_summary",
    )
    side_channel_status = _as_dict(
        train_manifest_summary.get("side_channel_status"),
        context="train_manifest_summary.side_channel_status",
    )
    proprio_status = _as_dict(
        side_channel_status.get("proprio"), context="proprio_status"
    )

    return [
        {
            "id": "formal_metric_visual_path_matches_full_and_reverses_direction",
            "kind": "formal_metric",
            "supports": [
                "direction_polarity_semantics",
                "weak_or_noisy_vision_contribution",
            ],
            "detail": "full_input is almost identical to vision_only and both are wrong-direction, while prompt_only is not wrong-direction.",
            "metrics": {
                "full_input_auc": formal_gate.get("auc_all"),
                "vision_only_auc": vision_mode.get("auc_all"),
                "full_minus_vision_auc": formal_ablation.get("full_minus_vision_auc"),
                "full_direction_correct": full_mode.get("direction_correct"),
                "vision_direction_correct": vision_mode.get("direction_correct"),
                "prompt_direction_correct": prompt_mode.get("direction_correct"),
                "full_success_fail_gap": full_mode.get("success_fail_gap"),
                "vision_success_fail_gap": vision_mode.get("success_fail_gap"),
            },
        },
        {
            "id": "formal_metric_prompt_control_beats_full",
            "kind": "formal_metric",
            "supports": ["prompt_shortcut"],
            "detail": "full_input loses to prompt_only on the same held-out set, so prompt confound remains real even if prompt_only is weak.",
            "metrics": {
                "full_input_auc": formal_ablation.get("full_input_auc"),
                "prompt_only_auc": formal_ablation.get("prompt_only_auc"),
                "full_minus_prompt_auc": formal_ablation.get("full_minus_prompt_auc"),
                "prompt_shortcut_risk": formal_ablation.get("prompt_shortcut_risk"),
                "prompt_score_range": prompt_mode.get("score_range"),
                "prompt_success_fail_gap": prompt_mode.get("success_fail_gap"),
            },
        },
        {
            "id": "formal_metric_baseline_gap",
            "kind": "formal_metric",
            "supports": [
                "direction_polarity_semantics",
                "weak_or_noisy_vision_contribution",
            ],
            "detail": "The multimodal critic is far below the state-only baseline on the same held-out split.",
            "metrics": {
                "baseline_auc": baseline_mode.get("auc_all"),
                "full_input_auc": formal_gate.get("auc_all"),
                "baseline_delta_auc": formal_gate.get("baseline_delta_auc"),
                "baseline_direction_correct": baseline_mode.get("direction_correct"),
            },
        },
        {
            "id": "code_eval_zero_proprio_vs_train_real_proprio",
            "kind": "code_heuristic",
            "supports": ["side_channel_misuse"],
            "detail": "Formal eval zeroes proprio, while training declares and loads real proprio from observation.state when available.",
            "code_refs": [
                code_evidence["eval_zero_proprio"],
                code_evidence["train_real_proprio"],
            ],
            "training_side_channel_status": proprio_status,
        },
        {
            "id": "code_prompt_only_is_not_clean_text_only",
            "kind": "code_heuristic",
            "supports": ["prompt_shortcut", "side_channel_misuse"],
            "detail": "The prompt_only ablation still keeps side channels enabled, so it is not a pure prompt-only control.",
            "manifest_inputs": {
                "prompt_only": manifest_inputs.get("prompt_only"),
                "vision_only": manifest_inputs.get("vision_only"),
                "full_input": manifest_inputs.get("full_input"),
            },
            "code_refs": [code_evidence["eval_t_norm"]],
        },
        {
            "id": "code_prompt_template_is_fixed_query_plus_optional_task_text",
            "kind": "code_heuristic",
            "supports": ["prompt_shortcut"],
            "detail": "Prompt builder always emits the same critic query and optionally prepends task text; this is easy to confound unless prompt influence is neutralized explicitly.",
            "code_refs": [
                code_evidence["prompt_template"],
                code_evidence["train_prompt_builder"],
            ],
        },
        {
            "id": "code_dataset_target_is_monotonic_raw_return",
            "kind": "code_heuristic",
            "supports": ["direction_polarity_semantics"],
            "detail": "Dataset build target formation is monotonic in return_G, so there is no obvious build-time global sign flip.",
            "code_refs": [code_evidence["monotonic_target_binning"]],
        },
        {
            "id": "artifact_sample_caps_limit_interpretability",
            "kind": "artifact_fact",
            "supports": ["unresolved_quality_scope"],
            "detail": "Current real retrain artifact was produced with material sample caps, so quality conclusions are directionally useful but not final.",
            "training_hparams": {
                "max_train_samples": training_hparams.get("max_train_samples"),
                "max_val_samples": training_hparams.get("max_val_samples"),
                "max_warmstart_samples": training_hparams.get("max_warmstart_samples"),
            },
            "metrics_counts": {
                "train_sample_count": critic_metrics.get("train_sample_count"),
                "val_sample_count": critic_metrics.get("val_sample_count"),
                "warmstart_public_sample_count": critic_metrics.get(
                    "warmstart_public_sample_count"
                ),
            },
            "code_refs": [code_evidence["sample_caps"]],
        },
    ]


def _diagnose(
    *,
    formal_gate: Mapping[str, object],
    formal_ablation: Mapping[str, object],
    critic_provenance: Mapping[str, object],
    critic_metrics: Mapping[str, object],
    evidence: Sequence[Mapping[str, object]],
    manifest_inputs: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    modes = _extract_manifest_modes(formal_gate)
    prompt_mode = modes["prompt_only"]
    vision_mode = modes["vision_only"]
    full_mode = modes["full_input"]
    baseline_mode = _as_dict(
        _as_dict(formal_gate.get("modes"), context="modes").get("baseline_state_only"),
        context="baseline",
    )

    full_auc = _as_float(
        formal_ablation.get("full_input_auc"), context="full_input_auc"
    )
    prompt_auc = _as_float(
        formal_ablation.get("prompt_only_auc"), context="prompt_only_auc"
    )
    vision_auc = _as_float(
        formal_ablation.get("vision_only_auc"), context="vision_only_auc"
    )
    baseline_auc = _as_float(baseline_mode.get("auc_all"), context="baseline_auc")
    baseline_delta_auc = _as_float(
        formal_gate.get("baseline_delta_auc"), context="baseline_delta_auc"
    )
    full_minus_prompt_auc = _as_float(
        formal_ablation.get("full_minus_prompt_auc"), context="full_minus_prompt_auc"
    )
    full_minus_vision_auc = _as_float(
        formal_ablation.get("full_minus_vision_auc"), context="full_minus_vision_auc"
    )

    full_dir = bool(full_mode.get("direction_correct", False))
    prompt_dir = bool(prompt_mode.get("direction_correct", False))
    vision_dir = bool(vision_mode.get("direction_correct", False))
    prompt_range = _optional_float(prompt_mode.get("score_range"))
    prompt_gap = _optional_float(prompt_mode.get("success_fail_gap"))
    prompt_uses_side_channels = bool(
        cast(Mapping[str, object], manifest_inputs.get("prompt_only", {})).get(
            "use_side_channels", False
        )
    )
    train_manifest_summary = _as_dict(
        critic_provenance.get("train_manifest_summary"),
        context="train_manifest_summary",
    )
    side_channel_status = _as_dict(
        train_manifest_summary.get("side_channel_status"), context="side_channel_status"
    )
    proprio_status = _as_dict(
        side_channel_status.get("proprio"), context="proprio_status"
    )
    training_hparams = _as_dict(
        critic_provenance.get("training_hparams"), context="training_hparams"
    )

    visual_path_matches_full = abs(float(full_minus_vision_auc)) <= 0.01
    visual_path_reversed = (not full_dir) and (not vision_dir) and prompt_dir
    full_below_prompt = float(full_minus_prompt_auc) < 0.0
    eval_train_proprio_mismatch = bool(
        proprio_status.get("available_in_dataset", False)
    )
    sample_caps_active = any(
        value not in (None, 0)
        for value in (
            training_hparams.get("max_train_samples"),
            training_hparams.get("max_val_samples"),
            training_hparams.get("max_warmstart_samples"),
        )
    )

    if not evidence:
        raise RuntimeError("postmortem_inconclusive: no evidence collected")

    if visual_path_reversed and visual_path_matches_full:
        primary_category = "direction_polarity_semantics"
        primary_confidence = 0.82
        primary_statement = (
            "The blocker is most consistent with wrong-signed ordering on the video-conditioned path: "
            "full_input nearly collapses to vision_only and both rank success below failure. "
            "This is stronger than a pure prompt-shortcut explanation, because prompt_only is weak but not reversed."
        )
    elif eval_train_proprio_mismatch:
        primary_category = "side_channel_misuse"
        primary_confidence = 0.74
        primary_statement = "The blocker is most consistent with train/eval side-channel mismatch: training uses real proprio while offline gate zeros it out."
    else:
        raise RuntimeError(
            "postmortem_inconclusive: evidence did not cleanly separate primary cause"
        )

    secondary: list[dict[str, object]] = []
    secondary.append(
        {
            "category": "weak_or_noisy_vision_contribution",
            "rank": "SECONDARY",
            "confidence": _confidence(0.77 if visual_path_matches_full else 0.61),
            "statement": (
                "Vision is not adding useful held-out signal right now. It either dominates in the wrong direction or is too stale/weak to help, because full_input does not outperform vision_only and both underperform badly."
            ),
        }
    )
    secondary.append(
        {
            "category": "prompt_shortcut",
            "rank": "SECONDARY",
            "confidence": _confidence(0.72 if full_below_prompt else 0.45),
            "statement": (
                "Prompt shortcut risk is real because full_input loses to prompt_only on the formal gate, but the current prompt_only lane is not a clean text-only control because it still keeps side channels on."
            ),
        }
    )
    secondary.append(
        {
            "category": "side_channel_misuse",
            "rank": "SECONDARY",
            "confidence": _confidence(0.78 if eval_train_proprio_mismatch else 0.42),
            "statement": (
                "Side-channel misuse is clearly present: prompt_only is side-channel contaminated, and offline gate zeros proprio despite training with real observation.state features. This likely depresses quality and muddies ablation interpretation, but it does not by itself explain why the video-conditioned lanes reverse direction."
            ),
        }
    )

    unresolved = [
        {
            "category": "exact_visual_failure_submechanism",
            "rank": "UNRESOLVED",
            "statement": (
                "T7E can rank the failure as video-path wrong-direction, but cannot yet distinguish whether the immediate cause is stale single-frame supervision, fusion domination, or optimization pathology inside the visual-conditioned path."
            ),
        }
    ]
    if sample_caps_active:
        unresolved.append(
            {
                "category": "bounded_retrain_caps_limit_interpretability",
                "rank": "UNRESOLVED",
                "statement": (
                    "Current artifact used capped warmstart/train/val sample counts, so absolute quality conclusions remain bounded until T7F removes those caps materially."
                ),
            }
        )

    t7f_required_fixes = [
        {
            "rule_id": "T7F_FIX_01_SIGN_AND_DIRECTION_AUDIT_FIRST",
            "priority": "blocker",
            "required": True,
            "rule": "Before any retrain, run a deterministic sign audit on train/val/test manifests and a tiny held-out pair probe to verify higher return_G -> higher target_bin_index -> higher predicted value_V_raw, and fail closed if any video-conditioned lane still reports direction_correct=false.",
            "why": "Current full_input and vision_only lanes are wrong-direction, so T7F must verify semantic direction before spending compute on another retrain.",
        },
        {
            "rule_id": "T7F_FIX_02_PROMPT_NEUTRALIZE_AND_CLEAN_CONTROLS",
            "priority": "high",
            "required": True,
            "rule": "Neutralize prompt influence for the next retrain/control pass by making prompt text constant (DEFAULT_CRITIC_QUERY only) and redefining prompt_only as a true text-only lane with use_side_channels=false.",
            "why": "The current prompt shortcut risk is real, and the existing prompt_only lane is not a clean control because it still carries side channels.",
        },
        {
            "rule_id": "T7F_FIX_03_SIDE_CHANNEL_PARITY_OR_DISABLE",
            "priority": "high",
            "required": True,
            "rule": "Restore train/eval side-channel parity before retraining: either feed real proprio from observation.state in offline gate/inference, or disable proprio in both train and eval for the next T7F cycle. Do not keep the current train-real/eval-zero mismatch.",
            "why": "Training currently consumes real proprio while formal eval zeros it out, which contaminates both quality assessment and ablation interpretation.",
        },
        {
            "rule_id": "T7F_FIX_04_FORCE_VISION_SANITY_GATE",
            "priority": "high",
            "required": True,
            "rule": "Add a pre-retrain vision sanity gate: on a small held-out subset, vision_only and full_input must both beat random and must not reverse success/fail direction before a full retrain is allowed to proceed.",
            "why": "Current evidence says the video-conditioned path is the dominant harmful branch; T7F must prove that branch is no longer wrong-signed before scaling up.",
        },
        {
            "rule_id": "T7F_FIX_05_REMOVE_SAMPLE_CAPS_MATERIALLY",
            "priority": "high",
            "required": True,
            "rule": "Remove or materially relax max_warmstart_samples, max_train_samples, and max_val_samples for the next formal retrain so T7F does not draw scientific conclusions from the current 128/256/64 capped regime.",
            "why": "Current bounded retrain is enough for diagnosis direction, but not enough for trustworthy quality close-out.",
        },
    ]

    return {
        "primary_root_cause": {
            "category": primary_category,
            "rank": "PRIMARY",
            "confidence": _confidence(primary_confidence),
            "statement": primary_statement,
        },
        "secondary_risks": secondary,
        "unresolved": unresolved,
        "root_cause_ranking": {
            "PRIMARY": [primary_category],
            "SECONDARY": [item["category"] for item in secondary],
            "UNRESOLVED": [item["category"] for item in unresolved],
        },
        "evidence": [dict(item) for item in evidence],
        "t7f_required_fixes": t7f_required_fixes,
        "confidence": {
            "overall": _confidence(0.81),
            "primary": _confidence(primary_confidence),
            "notes": "Confidence is high enough for T7F remediation ranking, but not for pinpointing the exact inner visual failure mechanism because the current retrain used material sample caps.",
        },
        "postmortem_complete": True,
        "postmortem_inconclusive": False,
        "diagnosis_summary": {
            "formal_gate_blocking_reasons": formal_gate.get("blocking_reasons"),
            "observed_metrics": {
                "full_input_auc": full_auc,
                "prompt_only_auc": prompt_auc,
                "vision_only_auc": vision_auc,
                "baseline_auc": baseline_auc,
                "baseline_delta_auc": baseline_delta_auc,
                "full_minus_prompt_auc": full_minus_prompt_auc,
                "full_minus_vision_auc": full_minus_vision_auc,
            },
            "prompt_control_not_clean": bool(prompt_uses_side_channels),
            "prompt_only_score_range": prompt_range,
            "prompt_only_success_fail_gap": prompt_gap,
            "sample_caps_active": bool(sample_caps_active),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="44c_vlm_critic_postmortem.py",
        description="Diagnose Task 7 formal gate failure and emit a machine-readable T7E postmortem.",
    )
    _ = parser.add_argument("--formal-gate-json", type=str, required=True)
    _ = parser.add_argument("--ablation-json", type=str, required=True)
    _ = parser.add_argument("--critic-provenance", type=str, required=True)
    _ = parser.add_argument("--critic-metrics", type=str, required=True)
    _ = parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON_REL)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_json = _resolve_path(args.output_json, default_rel=DEFAULT_OUTPUT_JSON_REL)
    try:
        formal_gate = _read_json(_resolve_path(args.formal_gate_json, default_rel=""))
        formal_ablation = _read_json(_resolve_path(args.ablation_json, default_rel=""))
        critic_provenance = _read_json(
            _resolve_path(args.critic_provenance, default_rel="")
        )
        critic_metrics = _read_json(_resolve_path(args.critic_metrics, default_rel=""))

        manifest_inputs = _load_manifest_input_modes(
            _extract_manifest_modes(formal_gate)
        )
        code_evidence = _build_code_evidence(REPO_ROOT)
        evidence = _build_evidence(
            formal_gate=formal_gate,
            formal_ablation=formal_ablation,
            critic_provenance=critic_provenance,
            critic_metrics=critic_metrics,
            code_evidence=code_evidence,
            manifest_inputs=manifest_inputs,
        )
        diagnosis = _diagnose(
            formal_gate=formal_gate,
            formal_ablation=formal_ablation,
            critic_provenance=critic_provenance,
            critic_metrics=critic_metrics,
            evidence=evidence,
            manifest_inputs=manifest_inputs,
        )
        payload = {
            "schema_version": "task7_postmortem_diagnosis_v1",
            "task": "task7_vlm_critic_postmortem",
            "formal_gate_json": str(
                _resolve_path(args.formal_gate_json, default_rel="")
            ),
            "ablation_json": str(_resolve_path(args.ablation_json, default_rel="")),
            "critic_provenance": str(
                _resolve_path(args.critic_provenance, default_rel="")
            ),
            "critic_metrics": str(_resolve_path(args.critic_metrics, default_rel="")),
            **diagnosis,
        }
        _write_json(output_json, payload)
        print(f"[INFO] wrote_json: {output_json}")
        print("SENTINEL:" + PASS_SENTINEL)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "task7_postmortem_diagnosis_v1",
            "task": "task7_vlm_critic_postmortem",
            "postmortem_complete": False,
            "postmortem_inconclusive": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(output_json, failure)
        print(f"[INFO] wrote_json: {output_json}")
        print("SENTINEL:" + FAIL_SENTINEL)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
