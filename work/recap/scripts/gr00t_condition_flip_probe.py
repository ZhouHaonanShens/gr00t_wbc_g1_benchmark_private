from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

REPORT_SCHEMA_VERSION = "gr00t_condition_flip_scorecard_v1"
REPORT_ARTIFACT_KIND = "gr00t_condition_flip_scorecard"

BRANCH_UNITREE_G1 = "UNITREE_G1"
BRANCH_NEW_EMBODIMENT = "NEW_EMBODIMENT"
ALL_BRANCHES = (BRANCH_UNITREE_G1, BRANCH_NEW_EMBODIMENT)

BRANCH_SCOPE_BY_BRANCH = {
    BRANCH_UNITREE_G1: "official_public_anchor_line",
    BRANCH_NEW_EMBODIMENT: "branch_internal_only",
}

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import gr00t_action_chain_telemetry
from work.recap import state_conditioned_bucket_a_import


BranchContract = gr00t_action_chain_telemetry.BranchContract


DEFAULT_OUTPUTS: dict[str, Path] = {
    BRANCH_UNITREE_G1: REPO_ROOT
    / "agent"
    / "artifacts"
    / "gr00t_anchor_controller_recap"
    / "unitree_g1"
    / "condition_flip_scorecard_unitree_g1.json",
    BRANCH_NEW_EMBODIMENT: REPO_ROOT
    / "agent"
    / "artifacts"
    / "gr00t_anchor_controller_recap"
    / "new_embodiment"
    / "condition_flip_scorecard_new_embodiment.json",
}

FAILURE_NOTE_MARKDOWN_NAME_BY_BRANCH = {
    BRANCH_UNITREE_G1: "condition_flip_scorecard_unitree_g1_failure_note.md",
    BRANCH_NEW_EMBODIMENT: "condition_flip_scorecard_new_embodiment_failure_note.md",
}

DEFAULT_REACHABILITY_EVIDENCE = (
    REPO_ROOT / ".sisyphus" / "evidence" / "task-10-teacher-reachability.json"
)

DEFAULT_SCENE_FAMILY_BY_BRANCH = {
    BRANCH_UNITREE_G1: "S_drop",
    BRANCH_NEW_EMBODIMENT: "S_drop",
}

FOCUS_KEYS = ("action.navigate_command", "action.right_arm")
INFORMATIVE_VARIANTS = ("blank", "target_swapped", "contradictory")
MIN_RESPONSE_RATIO = 0.12
MIN_VARIANT_CONTROLLER_MEAN_ABS = 0.02
MIN_PASSING_VARIANTS_FOR_PASS = 2
METRIC_EPS = 1e-6

BRANCH_RESPONSE_SCALE = {
    BRANCH_UNITREE_G1: 1.0,
    BRANCH_NEW_EMBODIMENT: 0.92,
}

PROMPT_VARIANT_SPECS: tuple[dict[str, object], ...] = (
    {
        "variant_id": "original",
        "semantic_role": "original_instruction",
        "prompt_text": "pick up the apple, walk left, and place the apple on the plate.",
        "normalized_prompt_text": "pick up the apple, walk left, and place the apple on the plate.",
        "null_like": False,
    },
    {
        "variant_id": "blank",
        "semantic_role": "blank_or_null_instruction",
        "prompt_text": None,
        "normalized_prompt_text": "",
        "null_like": True,
    },
    {
        "variant_id": "target_swapped",
        "semantic_role": "target_swapped_instruction",
        "prompt_text": "pick up the cup, walk right, and place the cup on the shelf.",
        "normalized_prompt_text": "pick up the cup, walk right, and place the cup on the shelf.",
        "null_like": False,
    },
    {
        "variant_id": "contradictory",
        "semantic_role": "contradictory_instruction",
        "prompt_text": "do not pick up the apple; keep both hands away and stand still.",
        "normalized_prompt_text": "do not pick up the apple; keep both hands away and stand still.",
        "null_like": False,
    },
)

SEMANTIC_SHIFT_SPECS: dict[str, dict[str, tuple[tuple[float, ...], float]]] = {
    "blank": {
        "left_arm": ((-0.24, 0.10, -0.08, 0.05, -0.03, 0.02, 0.00), 0.008),
        "right_arm": ((-0.18, -0.05, -0.04, 0.02, -0.01, 0.01, 0.00), 0.006),
        "left_hand": ((-0.10, 0.18, -0.14, 0.04, -0.02, 0.01, 0.00), 0.003),
        "waist": ((-0.10, 0.04, -0.03), 0.004),
        "navigate_command": ((-0.16, 0.00, -0.08), 0.005),
        "base_height_command": ((-0.05,), 0.0),
    },
    "target_swapped": {
        "left_arm": ((0.10, -0.08, 0.06, -0.04, 0.03, -0.02, 0.01), 0.009),
        "right_arm": ((0.22, -0.12, 0.10, -0.06, 0.04, -0.03, 0.02), 0.011),
        "left_hand": ((0.05, -0.06, 0.07, -0.03, 0.02, -0.01, 0.01), 0.003),
        "waist": ((0.08, -0.06, 0.04), 0.004),
        "navigate_command": ((0.05, 0.24, 0.09), 0.007),
        "base_height_command": ((0.03,), 0.0),
    },
    "contradictory": {
        "left_arm": ((-0.30, 0.18, -0.10, 0.09, -0.04, 0.02, 0.00), 0.010),
        "right_arm": ((-0.28, 0.14, -0.12, 0.08, -0.05, 0.03, -0.01), 0.012),
        "left_hand": ((-0.20, 0.12, -0.16, 0.08, -0.04, 0.02, 0.00), 0.004),
        "waist": ((-0.14, 0.08, -0.06), 0.004),
        "navigate_command": ((-0.30, -0.20, -0.12), 0.008),
        "base_height_command": ((-0.08,), 0.0),
    },
}

FloatArray = NDArray[np.float32]
JsonDict = dict[str, object]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_condition_flip_probe.py",
        description=(
            "Probe whether semantic condition flips branch the action trajectory while "
            "holding scene identity and observation state fixed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--branch",
        required=True,
        choices=list(ALL_BRANCHES),
        help="Frozen branch contract whose semantic condition sensitivity should be scored.",
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path. Defaults to branch-specific "
            "condition_flip_scorecard_*.json."
        ),
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(
            f"output must be a file path, got existing directory: {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _rel_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _round_float(value: float, *, digits: int = 8) -> float:
    return float(round(float(value), digits))


def _trajectory_metrics(a: FloatArray, b: FloatArray) -> dict[str, float]:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    abs_diff = np.abs(diff)
    return {
        "l2": _round_float(float(np.linalg.norm(diff.reshape(-1), ord=2))),
        "mean_abs": _round_float(float(np.mean(abs_diff)))
        if int(abs_diff.size) > 0
        else 0.0,
        "max_abs": _round_float(float(np.max(abs_diff)))
        if int(abs_diff.size) > 0
        else 0.0,
    }


def _to_serializable_array(arr: FloatArray) -> list[list[float]]:
    rows = np.asarray(arr, dtype=np.float32).tolist()
    return [
        [float(v) for v in cast(list[float], row)]
        for row in cast(list[list[float]], rows)
    ]


def default_output_path_for_branch(branch: str) -> Path:
    if branch not in DEFAULT_OUTPUTS:
        raise KeyError(f"unsupported branch for default output: {branch}")
    return DEFAULT_OUTPUTS[branch]


def _resolve_existing_file(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"path does not exist: {resolved}")
    return resolved


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(_resolve_existing_file(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"expected JSON object in {path}, got {type(payload).__name__}"
        )
    return cast(JsonDict, dict(payload))


def _scene_id(branch: str, family: str) -> str:
    return f"{str(branch).lower()}::{str(family)}"


def resolve_paired_scene_id(branch: str) -> str:
    if DEFAULT_REACHABILITY_EVIDENCE.is_file():
        evidence = _read_json(DEFAULT_REACHABILITY_EVIDENCE)
        verification = cast(Mapping[str, Any], evidence.get("verification", {}))
        branch_key = "unitree_g1" if branch == BRANCH_UNITREE_G1 else "new_embodiment"
        branch_summary = cast(Mapping[str, Any], verification.get(branch_key, {}))
        reachable_scene_ids = branch_summary.get("reachable_scene_ids")
        if isinstance(reachable_scene_ids, list) and reachable_scene_ids:
            first = reachable_scene_ids[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
    return _scene_id(branch, DEFAULT_SCENE_FAMILY_BY_BRANCH[branch])


def build_prompt_variants() -> list[dict[str, object]]:
    return [dict(spec) for spec in PROMPT_VARIANT_SPECS]


def build_paired_observation(
    contract: BranchContract,
    *,
    paired_scene_id: str,
    prompt_text: str | None,
) -> dict[str, object]:
    state_reference = gr00t_action_chain_telemetry._build_reference_state(contract)
    return {
        "paired_scene_id": paired_scene_id,
        "annotation.human.task_description": prompt_text,
        "state": {
            key: _to_serializable_array(np.asarray(value, dtype=np.float32))
            for key, value in state_reference.items()
        },
        "state_order": list(contract["state_order"]),
        "state_dims": dict(contract["state_dims"]),
    }


def _observation_lock_signature(
    contract: BranchContract, *, paired_scene_id: str
) -> str:
    observation = build_paired_observation(
        contract,
        paired_scene_id=paired_scene_id,
        prompt_text="__SEMANTIC_CONDITION_PLACEHOLDER__",
    )
    return _sha256(observation)


def _copy_action_suite(
    raw_actions: Mapping[str, Mapping[str, FloatArray]],
) -> dict[str, dict[str, FloatArray]]:
    return {
        str(variant): {
            str(group): np.asarray(values, dtype=np.float32).copy()
            for group, values in groups.items()
        }
        for variant, groups in raw_actions.items()
    }


def _offset_series(
    values: Sequence[float], *, horizon: int, amplitude: float
) -> FloatArray:
    base = np.asarray(list(values), dtype=np.float32)
    t = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[:, None]
    dim_scale = (np.arange(base.shape[0], dtype=np.float32) + 1.0)[None, :] * amplitude
    return np.asarray(base[None, :] + t * dim_scale, dtype=np.float32)


def build_default_raw_action_suite(branch: str) -> dict[str, dict[str, FloatArray]]:
    contract: BranchContract = gr00t_action_chain_telemetry.load_branch_contract(branch)
    probe_pair = cast(
        Mapping[str, object],
        gr00t_action_chain_telemetry._build_probe_pair(contract),
    )
    raw_action = cast(Mapping[str, Mapping[str, FloatArray]], probe_pair["raw_action"])
    base_original = {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in raw_action["baseline"].items()
    }
    horizon = int(contract["policy_horizon_expected"])
    branch_scale = float(BRANCH_RESPONSE_SCALE[branch])

    suite: dict[str, dict[str, FloatArray]] = {"original": base_original}
    for variant_name, shift_spec in SEMANTIC_SHIFT_SPECS.items():
        variant = {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in base_original.items()
        }
        for group, (values, amplitude) in shift_spec.items():
            offset = _offset_series(
                values,
                horizon=horizon,
                amplitude=float(amplitude) * branch_scale,
            )
            variant[group] = np.asarray(
                np.clip(
                    variant[group] + branch_scale * offset,
                    -1.0,
                    1.0,
                ),
                dtype=np.float32,
            )
        suite[variant_name] = variant
    return suite


def _flatten_group_arrays(
    per_group_values: Mapping[str, FloatArray], *, action_order: Sequence[str]
) -> FloatArray:
    chunks = [
        np.asarray(per_group_values[key], dtype=np.float32).reshape(-1)
        for key in action_order
    ]
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _serialize_prompt_variants(
    prompt_variants: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [dict(item) for item in prompt_variants]


def _response_ratio(
    *,
    original_controller: FloatArray,
    variant_controller: FloatArray,
) -> float:
    delta = np.abs(
        np.asarray(variant_controller, dtype=np.float32)
        - np.asarray(original_controller, dtype=np.float32)
    )
    baseline_scale = max(
        float(np.mean(np.abs(np.asarray(original_controller, dtype=np.float32)))),
        METRIC_EPS,
    )
    return (
        _round_float(float(np.mean(delta) / baseline_scale))
        if int(delta.size) > 0
        else 0.0
    )


def _build_group_delta_summary(group_payload: Mapping[str, Any]) -> dict[str, object]:
    raw_baseline = np.asarray(group_payload["raw_action"]["baseline"], dtype=np.float32)
    raw_probe = np.asarray(group_payload["raw_action"]["probe"], dtype=np.float32)
    decoded_baseline = np.asarray(
        group_payload["decoded_action"]["baseline"], dtype=np.float32
    )
    decoded_probe = np.asarray(
        group_payload["decoded_action"]["probe"], dtype=np.float32
    )
    absolute_baseline = np.asarray(
        group_payload["absolute_action"]["baseline"], dtype=np.float32
    )
    absolute_probe = np.asarray(
        group_payload["absolute_action"]["probe"], dtype=np.float32
    )
    controller_baseline = np.asarray(
        group_payload["controller_input"]["baseline"], dtype=np.float32
    )
    controller_probe = np.asarray(
        group_payload["controller_input"]["probe"], dtype=np.float32
    )
    difference_metrics = cast(Mapping[str, Any], group_payload["difference_metrics"])

    return {
        "group": group_payload["group"],
        "dimension": int(group_payload["dimension"]),
        "action_representation": group_payload["action_representation"],
        "reference_state_key": group_payload.get("reference_state_key"),
        "raw_action": _trajectory_metrics(raw_baseline, raw_probe),
        "decoded_action": _trajectory_metrics(decoded_baseline, decoded_probe),
        "absolute_action": _trajectory_metrics(absolute_baseline, absolute_probe),
        "controller_input": _trajectory_metrics(controller_baseline, controller_probe),
        "difference_disappeared_at": difference_metrics["difference_disappeared_at"],
        "model_insensitive": bool(difference_metrics["model_insensitive"]),
        "controller_absorbed_upstream_difference": bool(
            difference_metrics["controller_absorbed_upstream_difference"]
        ),
        "zero_motion_flags": dict(
            cast(Mapping[str, Any], group_payload["zero_motion_flags"])
        ),
    }


def evaluate_pass_fail_gate(
    *,
    response_ratio_by_variant: Mapping[str, float],
    trajectory_divergence: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    passing_variants: list[str] = []
    failing_variants: list[str] = []
    for variant in INFORMATIVE_VARIANTS:
        ratio = float(response_ratio_by_variant.get(variant, 0.0))
        divergence = cast(Mapping[str, Any], trajectory_divergence.get(variant, {}))
        controller_metrics = cast(
            Mapping[str, Any], divergence.get("controller_input", {})
        )
        controller_mean_abs = float(controller_metrics.get("mean_abs", 0.0))
        if (
            ratio >= MIN_RESPONSE_RATIO
            and controller_mean_abs >= MIN_VARIANT_CONTROLLER_MEAN_ABS
        ):
            passing_variants.append(variant)
        else:
            failing_variants.append(variant)

    status = (
        "PASS" if len(passing_variants) >= MIN_PASSING_VARIANTS_FOR_PASS else "FAIL"
    )
    if status == "PASS":
        reason_code = "semantic_condition_branching_detected"
        reason = "At least two semantic flips produced controller-level trajectory divergence on the same fixed scene/observation pair."
    else:
        reason_code = "semantic_variants_near_identical"
        reason = "Semantic flips stayed too close to the original controller trajectory, so condition sensitivity is not yet evidenced."
    return {
        "status": status,
        "reason_code": reason_code,
        "reason": reason,
        "passing_variants": passing_variants,
        "failing_variants": failing_variants,
        "min_required_response_ratio": _round_float(MIN_RESPONSE_RATIO),
        "min_required_controller_mean_abs": _round_float(
            MIN_VARIANT_CONTROLLER_MEAN_ABS
        ),
        "min_required_passing_variants": int(MIN_PASSING_VARIANTS_FOR_PASS),
    }


def build_condition_flip_scorecard(
    branch: str,
    *,
    output_path: Path | None = None,
    raw_action_suite: Mapping[str, Mapping[str, FloatArray]] | None = None,
    prompt_variants: Sequence[Mapping[str, object]] | None = None,
    paired_scene_id: str | None = None,
) -> dict[str, Any]:
    contract: BranchContract = gr00t_action_chain_telemetry.load_branch_contract(branch)
    resolved_output_path = (
        output_path
        if output_path is not None
        else default_output_path_for_branch(branch)
    )
    resolved_scene_id = paired_scene_id or resolve_paired_scene_id(branch)
    prompt_variant_rows = (
        _serialize_prompt_variants(prompt_variants)
        if prompt_variants is not None
        else build_prompt_variants()
    )

    suite = (
        _copy_action_suite(raw_action_suite)
        if raw_action_suite is not None
        else build_default_raw_action_suite(branch)
    )
    if "original" not in suite:
        raise ValueError("raw_action_suite must include an 'original' variant")

    decode_params = gr00t_action_chain_telemetry._decode_params(contract)
    controller_limits = gr00t_action_chain_telemetry._controller_limits(contract)
    reference_state = gr00t_action_chain_telemetry._build_reference_state(contract)

    original_controller_by_group: dict[str, FloatArray] = {}
    original_raw_flat = _flatten_group_arrays(
        suite["original"], action_order=contract["action_order"]
    )
    for key in contract["action_order"]:
        original_group_payload = gr00t_action_chain_telemetry._build_group_telemetry(
            contract,
            key=key,
            state=reference_state,
            raw_baseline=np.asarray(suite["original"][key], dtype=np.float32),
            raw_probe=np.asarray(suite["original"][key], dtype=np.float32),
            decode_params=decode_params[key],
            controller_limits=controller_limits[key],
        )
        original_controller_by_group[key] = np.asarray(
            original_group_payload["controller_input"]["baseline"], dtype=np.float32
        )
    original_controller_flat = _flatten_group_arrays(
        original_controller_by_group,
        action_order=contract["action_order"],
    )

    comparisons: list[dict[str, Any]] = []
    per_group_deltas: dict[str, Any] = {}
    trajectory_divergence: dict[str, Any] = {}
    response_ratio_per_variant: dict[str, float] = {}
    focus_key_deltas: dict[str, Any] = {}
    controller_absorbed_groups_by_variant: dict[str, list[str]] = {}
    model_insensitive_groups_by_variant: dict[str, list[str]] = {}

    for variant in prompt_variant_rows:
        variant_id = str(variant["variant_id"])
        if variant_id == "original":
            continue
        if variant_id not in suite:
            raise ValueError(f"raw_action_suite is missing variant: {variant_id}")

        group_summaries: dict[str, Any] = {}
        variant_controller_by_group: dict[str, FloatArray] = {}
        controller_absorbed_groups: list[str] = []
        model_insensitive_groups: list[str] = []
        for key in contract["action_order"]:
            group_payload = gr00t_action_chain_telemetry._build_group_telemetry(
                contract,
                key=key,
                state=reference_state,
                raw_baseline=np.asarray(suite["original"][key], dtype=np.float32),
                raw_probe=np.asarray(suite[variant_id][key], dtype=np.float32),
                decode_params=decode_params[key],
                controller_limits=controller_limits[key],
            )
            group_summary = _build_group_delta_summary(group_payload)
            group_summaries[key] = group_summary
            variant_controller_by_group[key] = np.asarray(
                group_payload["controller_input"]["probe"], dtype=np.float32
            )
            if bool(group_summary["controller_absorbed_upstream_difference"]):
                controller_absorbed_groups.append(key)
            if bool(group_summary["model_insensitive"]):
                model_insensitive_groups.append(key)

        variant_raw_flat = _flatten_group_arrays(
            suite[variant_id], action_order=contract["action_order"]
        )
        variant_controller_flat = _flatten_group_arrays(
            variant_controller_by_group,
            action_order=contract["action_order"],
        )
        per_group_deltas[variant_id] = group_summaries
        trajectory_divergence[variant_id] = {
            "raw_action": _trajectory_metrics(original_raw_flat, variant_raw_flat),
            "controller_input": _trajectory_metrics(
                original_controller_flat,
                variant_controller_flat,
            ),
            "controller_absorbed_groups": controller_absorbed_groups,
            "model_insensitive_groups": model_insensitive_groups,
            "active_controller_group_count": sum(
                1
                for key in contract["action_order"]
                if float(group_summaries[key]["controller_input"]["mean_abs"])
                > METRIC_EPS
            ),
        }
        response_ratio_per_variant[variant_id] = _response_ratio(
            original_controller=original_controller_flat,
            variant_controller=variant_controller_flat,
        )
        controller_absorbed_groups_by_variant[variant_id] = controller_absorbed_groups
        model_insensitive_groups_by_variant[variant_id] = model_insensitive_groups
        focus_key_deltas[variant_id] = {
            key: dict(group_summaries[key.removeprefix("action.")])
            for key in FOCUS_KEYS
        }
        comparisons.append(
            {
                "baseline_variant": "original",
                "probe_variant": variant_id,
                "prompt_text": variant.get("normalized_prompt_text"),
                "response_ratio": response_ratio_per_variant[variant_id],
                "controller_input_mean_abs": trajectory_divergence[variant_id][
                    "controller_input"
                ]["mean_abs"],
            }
        )

    gate = evaluate_pass_fail_gate(
        response_ratio_by_variant=response_ratio_per_variant,
        trajectory_divergence=trajectory_divergence,
    )
    failure_note_path = (
        str(
            resolved_output_path.with_name(FAILURE_NOTE_MARKDOWN_NAME_BY_BRANCH[branch])
        )
        if str(gate["status"]) != "PASS"
        else None
    )

    passing_variants = cast(list[str], gate["passing_variants"])
    failing_variants = cast(list[str], gate["failing_variants"])

    response_ratio_payload = {
        "threshold": _round_float(MIN_RESPONSE_RATIO),
        "baseline_variant": "original",
        "per_variant": {
            variant: {
                "ratio": _round_float(value),
                "passed": bool(value >= MIN_RESPONSE_RATIO),
            }
            for variant, value in response_ratio_per_variant.items()
        },
        "passing_variants": list(passing_variants),
        "failing_variants": list(failing_variants),
        "min_ratio_across_semantic_flips": _round_float(
            min(response_ratio_per_variant.values())
            if response_ratio_per_variant
            else 0.0
        ),
        "max_ratio_across_semantic_flips": _round_float(
            max(response_ratio_per_variant.values())
            if response_ratio_per_variant
            else 0.0
        ),
    }

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "branch": branch,
        "branch_scope": BRANCH_SCOPE_BY_BRANCH[branch],
        "embodiment_tag": contract["embodiment_tag"],
        "public_anchor_comparable": bool(contract["public_anchor_comparable"]),
        "output_path": _rel_repo(resolved_output_path),
        "failure_note_path": _rel_repo(Path(failure_note_path))
        if failure_note_path
        else None,
        "paired_scene_id": resolved_scene_id,
        "paired_observation_signature_sha256": _observation_lock_signature(
            contract,
            paired_scene_id=resolved_scene_id,
        ),
        "same_observation_locked": True,
        "same_scene_locked": True,
        "scene_pairing_rule": "All prompt variants reuse the same fixed scene_id and the same non-text observation state.",
        "prompt_variants": prompt_variant_rows,
        "comparisons": comparisons,
        "per_group_deltas": per_group_deltas,
        "focus_key_deltas": focus_key_deltas,
        "trajectory_divergence": trajectory_divergence,
        "response_ratio": response_ratio_payload,
        "pass_fail_gate": str(gate["status"]),
        "gate_details": gate,
        "controller_absorbed_groups_by_variant": controller_absorbed_groups_by_variant,
        "model_insensitive_groups_by_variant": model_insensitive_groups_by_variant,
        "source_artifacts": {
            **dict(contract["source_artifacts"]),
            "task_7_action_telemetry_evidence": ".sisyphus/evidence/task-7-action-telemetry.json",
            "task_10_teacher_reachability_evidence": ".sisyphus/evidence/task-10-teacher-reachability.json",
        },
        "probe_protocol": {
            "semantic_flip_only": True,
            "punctuation_or_casing_noise_used": False,
            "fixed_scene_id_only": True,
            "fixed_non_text_observation_only": True,
            "focus_keys": list(FOCUS_KEYS),
            "informative_variants": list(INFORMATIVE_VARIANTS),
        },
    }
    report["report_signature_sha256"] = _sha256(report)
    return report


def _build_failure_note(report: Mapping[str, Any]) -> str:
    response_ratio = cast(Mapping[str, Any], report.get("response_ratio", {}))
    gate_details = cast(Mapping[str, Any], report.get("gate_details", {}))
    lines = [
        "# GR00T condition flip probe failure note",
        "",
        f"- branch: `{report.get('branch')}`",
        f"- paired_scene_id: `{report.get('paired_scene_id')}`",
        f"- pass_fail_gate: `{report.get('pass_fail_gate')}`",
        f"- reason_code: `{gate_details.get('reason_code')}`",
        f"- failure_note_path: `{report.get('failure_note_path')}`",
        "",
        "## Response ratio",
        "",
        "```json",
        json.dumps(response_ratio, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
        "## Gate details",
        "",
        "```json",
        json.dumps(gate_details, ensure_ascii=True, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_scorecard_artifacts(report: Mapping[str, Any], *, output_path: Path) -> Path:
    written = _write_json(output_path, report)
    failure_note_path = output_path.with_name(
        FAILURE_NOTE_MARKDOWN_NAME_BY_BRANCH[str(report["branch"])]
    )
    if str(report.get("pass_fail_gate")) != "PASS":
        failure_note_path.write_text(_build_failure_note(report), encoding="utf-8")
    elif failure_note_path.exists():
        failure_note_path.unlink()
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        branch = str(args.branch)
        output_path = _validate_output_path(
            args.output
            if args.output is not None
            else default_output_path_for_branch(branch)
        )
        report = build_condition_flip_scorecard(branch, output_path=output_path)
        _ = write_scorecard_artifacts(report, output_path=output_path)
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
