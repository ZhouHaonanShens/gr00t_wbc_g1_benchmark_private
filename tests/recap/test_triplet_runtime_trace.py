from __future__ import annotations

import json
from pathlib import Path
import sys
from collections.abc import Mapping, Sequence
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import build_carrier_panel
from work.recap.scripts import gr00t_carrier_panel_gate
from work.recap.scripts import gr00t_same_checkpoint_triplet_eval


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
    )
    _ = path.write_text(serialized, encoding="utf-8")
    return path


def _make_runtime_trace(
    *,
    decoded: float,
    absolute: float,
    controller_input: float,
    controller_output: float | None = None,
    prompt_distinct: bool = True,
    raw_or_decoded_distinct: bool = True,
) -> dict[str, object]:
    return {
        "trace_role": "debug_probe",
        "main_gate_eligible": False,
        "status": "READY",
        "normalization_metric": "mean_abs_delta_over_contract_range",
        "controller_output_available": controller_output is not None,
        "controller_output_unavailable_reason": None
        if controller_output is not None
        else "controller_output unavailable in current live seam",
        "prompt_surface": {
            "any_pair_distinct": prompt_distinct,
        },
        "token_surface": {
            "available_in_any_pair": False,
            "any_pair_distinct": False,
        },
        "stage_max_mean_abs_delta_over_contract_range": {
            "decoded_action": decoded,
            "absolute_action": absolute,
            "controller_input": controller_input,
            "controller_output": controller_output,
        },
        "upstream_distinction": {
            "prompt_or_token_distinct": prompt_distinct,
            "raw_action_distinct": raw_or_decoded_distinct,
            "raw_or_decoded_distinct": raw_or_decoded_distinct,
            "absolute_distinct": absolute > 0.0,
            "controller_input_distinct": controller_input > 0.0,
            "controller_output_distinct": bool(
                controller_output and controller_output > 0.0
            ),
        },
    }


def _make_dataset_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "carrier_panel_dataset"
    episode_rows = [
        {
            "episode_index": 1,
            "success": True,
            "outer_steps": 10,
            "failure_reason": None,
            "failure_stage_guess": None,
        },
        {
            "episode_index": 2,
            "success": False,
            "outer_steps": 10,
            "failure_reason": "terminated_without_success",
            "failure_stage_guess": "PLACE",
        },
    ]
    step_rows: list[dict[str, object]] = []
    for episode_index in (1, 2):
        for outer_step in range(1, 11):
            step_rows.append(
                {
                    "episode_index": episode_index,
                    "outer_step": outer_step,
                    "success_step": bool(episode_index == 1 and outer_step >= 8),
                    "episode_success_so_far": bool(
                        episode_index == 1 and outer_step >= 8
                    ),
                    "intermediate_signals": {
                        "phase": "TRANSPORT" if episode_index == 1 else "RECOVERY"
                    },
                }
            )
    trace_samples = [
        {
            "episode_index": 1,
            "outer_step": 1,
            "runtime_trace": _make_runtime_trace(
                decoded=0.08,
                absolute=0.07,
                controller_input=0.06,
            ),
        },
        {
            "episode_index": 1,
            "outer_step": 8,
            "runtime_trace": _make_runtime_trace(
                decoded=0.09,
                absolute=0.08,
                controller_input=0.07,
            ),
        },
        {
            "episode_index": 2,
            "outer_step": 1,
            "runtime_trace": _make_runtime_trace(
                decoded=0.07,
                absolute=0.06,
                controller_input=0.02,
            ),
        },
        {
            "episode_index": 2,
            "outer_step": 5,
            "runtime_trace": _make_runtime_trace(
                decoded=0.06,
                absolute=0.05,
                controller_input=0.01,
            ),
        },
        {
            "episode_index": 2,
            "outer_step": 9,
            "runtime_trace": _make_runtime_trace(
                decoded=0.04,
                absolute=0.03,
                controller_input=0.00,
            ),
        },
    ]
    _ = _write_jsonl(
        dataset_dir / build_carrier_panel.EPISODES_JSONL_NAME, episode_rows
    )
    _ = _write_jsonl(dataset_dir / build_carrier_panel.STEPS_JSONL_NAME, step_rows)
    _ = _write_json(
        dataset_dir / build_carrier_panel.TRACE_FIXTURE_JSON_NAME,
        {"samples": trace_samples},
    )
    return dataset_dir


def _minimal_group_surface(scale: float) -> dict[str, object]:
    return {
        "decoded_action": {"right_arm": [scale] * 7},
        "absolute_action": {"right_arm": [scale] * 7},
        "controller_input": {"right_arm": [scale] * 7},
    }


def _sample_from_triplet_bundle(scale: float) -> dict[str, object]:
    bundle = gr00t_same_checkpoint_triplet_eval.build_same_checkpoint_triplet_bundle(
        checkpoint_loaded="/tmp/checkpoints/debug-probe",
        prompt_raw="pick up the apple and place it on the plate",
        output_dir=Path("/tmp/debug-probe"),
        summary_json_path=Path("/tmp/debug-probe/summary.json"),
        observation_seed=11,
        mode_surface_by_mode={
            "omit": _minimal_group_surface(0.0),
            "positive": _minimal_group_surface(scale),
            "negative": _minimal_group_surface(-scale),
        },
    )
    summary = cast(dict[str, Any], bundle["summary"])
    runtime_trace = cast(dict[str, object], summary["runtime_trace"])
    return {
        "sample_id": "debug_probe_only",
        "panel_slot_index": 1,
        "panel_slot_name": "debug_probe_only",
        "episode_index": 1,
        "outer_step": 1,
        "episode_outcome": "success",
        "debug_probe": {
            "trace_role": runtime_trace["trace_role"],
            "main_gate_eligible": runtime_trace["main_gate_eligible"],
            "runtime_trace": runtime_trace,
        },
    }


def test_build_carrier_panel_selects_deterministic_success_and_failure_points(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_dataset_dir(tmp_path)

    panel = build_carrier_panel.build_carrier_panel(dataset_dir)

    assert panel["selection_algorithm"] == build_carrier_panel.SELECTION_ALGORITHM
    assert panel["panel_sample_count"] == 5
    samples = cast(list[dict[str, object]], panel["samples"])
    assert [sample["panel_slot_name"] for sample in samples] == [
        "success_t10",
        "success_t80",
        "failure_t10",
        "failure_t50",
        "failure_t90",
    ]
    assert [sample["episode_index"] for sample in samples] == [1, 1, 2, 2, 2]
    assert [sample["outer_step"] for sample in samples] == [1, 8, 1, 5, 9]


def test_carrier_panel_gate_distinguishes_weak_and_robust_strength() -> None:
    weak_panel = {
        "samples": [
            {
                "sample_id": f"weak_{idx}",
                "episode_index": idx,
                "outer_step": idx,
                "debug_probe": {
                    "runtime_trace": _make_runtime_trace(
                        decoded=0.07,
                        absolute=0.06,
                        controller_input=value,
                    )
                },
            }
            for idx, value in enumerate([0.051, 0.020, 0.019, 0.052, 0.001], start=1)
        ]
    }
    robust_panel = {
        "samples": [
            {
                "sample_id": f"robust_{idx}",
                "episode_index": idx,
                "outer_step": idx,
                "debug_probe": {
                    "runtime_trace": _make_runtime_trace(
                        decoded=0.07,
                        absolute=0.06,
                        controller_input=value,
                    )
                },
            }
            for idx, value in enumerate([0.051, 0.052, 0.020, 0.070, 0.049], start=1)
        ]
    }

    weak_gate = gr00t_carrier_panel_gate.build_carrier_panel_gate(weak_panel)
    robust_gate = gr00t_carrier_panel_gate.build_carrier_panel_gate(robust_panel)

    assert weak_gate["panel_pass_count"] == 2
    assert weak_gate["gate_strength"] == "weak"
    assert weak_gate["status"] == "consumed_and_survived"
    assert robust_gate["panel_pass_count"] == 3
    assert robust_gate["gate_strength"] == "robust"
    assert robust_gate["status"] == "consumed_and_survived"
    assert robust_gate["mainline_unlock"] is True


def test_normalized_delta_threshold_is_inclusive_at_point_zero_five() -> None:
    panel = {
        "samples": [
            {
                "sample_id": "below",
                "episode_index": 1,
                "outer_step": 1,
                "debug_probe": {
                    "runtime_trace": _make_runtime_trace(
                        decoded=0.07,
                        absolute=0.06,
                        controller_input=0.049,
                    )
                },
            },
            {
                "sample_id": "at_threshold",
                "episode_index": 1,
                "outer_step": 2,
                "debug_probe": {
                    "runtime_trace": _make_runtime_trace(
                        decoded=0.07,
                        absolute=0.06,
                        controller_input=0.05,
                    )
                },
            },
        ]
    }

    gate = gr00t_carrier_panel_gate.build_carrier_panel_gate(panel)

    per_sample = cast(list[dict[str, object]], gate["per_sample_summary"])
    assert per_sample[0]["panel_pass"] is False
    assert per_sample[1]["panel_pass"] is True
    assert gate["panel_pass_count"] == 1


def test_single_triplet_debug_probe_stays_weak_even_when_controller_delta_clears_threshold() -> (
    None
):
    sample = _sample_from_triplet_bundle(scale=0.065)
    panel = {"samples": [sample]}

    gate = gr00t_carrier_panel_gate.build_carrier_panel_gate(panel)
    runtime_trace = cast(
        dict[str, object],
        cast(dict[str, object], sample["debug_probe"])["runtime_trace"],
    )

    assert runtime_trace["trace_role"] == "debug_probe"
    assert runtime_trace["main_gate_eligible"] is False
    assert runtime_trace["terminal_stage_used"] == "controller_input"
    assert runtime_trace["controller_output_available"] is False
    assert gate["panel_pass_count"] == 1
    assert gate["status"] == "consumed_and_survived"
    assert gate["gate_strength"] == "weak"
    assert gate["mainline_unlock"] is False


def test_panel_gate_reports_absorbed_downstream_when_upstream_diff_never_reaches_controller() -> (
    None
):
    panel = {
        "samples": [
            {
                "sample_id": "absorbed",
                "episode_index": 1,
                "outer_step": 1,
                "debug_probe": {
                    "runtime_trace": _make_runtime_trace(
                        decoded=0.08,
                        absolute=0.07,
                        controller_input=0.0,
                    )
                },
            }
        ]
    }

    gate = gr00t_carrier_panel_gate.build_carrier_panel_gate(panel)

    assert gate["panel_pass_count"] == 0
    assert gate["status"] == "consumed_but_absorbed_downstream"
    assert gate["gate_strength"] == "weak"
