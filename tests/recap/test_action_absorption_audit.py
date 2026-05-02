from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_action_absorption_audit


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _runtime_trace(
    *,
    decoded: float,
    absolute: float,
    controller_input: float,
    controller_output: float | None = None,
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
        "prompt_surface": {"any_pair_distinct": True},
        "token_surface": {"available_in_any_pair": False, "any_pair_distinct": False},
        "stage_max_mean_abs_delta_over_contract_range": {
            "decoded_action": decoded,
            "absolute_action": absolute,
            "controller_input": controller_input,
            "controller_output": controller_output,
        },
        "upstream_distinction": {
            "prompt_or_token_distinct": True,
            "raw_action_distinct": True,
            "raw_or_decoded_distinct": True,
            "absolute_distinct": absolute > 0.0,
            "controller_input_distinct": controller_input > 0.0,
            "controller_output_distinct": bool(
                controller_output is not None and controller_output > 0.0
            ),
        },
    }


def _group_payload(
    *,
    representation: str,
    raw: float,
    decoded: float,
    absolute: float,
    controller: float,
    difference_disappeared_at: str | None,
    controller_absorbed: bool,
    decoded_clip_rate: float = 0.0,
    controller_clip_rate: float = 0.0,
    saturation_rate: float = 0.0,
    baseline_zero_output_rate: float | None = None,
    probe_zero_output_rate: float | None = None,
    all_zero_in_both: bool = False,
    reference_state_key: str | None = None,
) -> dict[str, object]:
    controller_stage = {
        "baseline": {"zero_output_rate": baseline_zero_output_rate},
        "probe": {"zero_output_rate": probe_zero_output_rate},
    }
    return {
        "action_representation": representation,
        "reference_state_key": reference_state_key,
        "difference_metrics": {
            "raw_action_l2": raw,
            "decoded_action_l2": decoded,
            "absolute_action_l2": absolute,
            "controller_input_l2": controller,
            "difference_disappeared_at": difference_disappeared_at,
            "model_insensitive": False,
            "controller_absorbed_upstream_difference": controller_absorbed,
        },
        "clip_rate": {
            "decoded_action": decoded_clip_rate,
            "controller_input": controller_clip_rate,
        },
        "saturation_rate": saturation_rate,
        "zero_motion_flags": {
            "baseline_controller_input_all_zero": all_zero_in_both,
            "probe_controller_input_all_zero": all_zero_in_both,
            "all_zero_in_both": all_zero_in_both,
        },
        "stages": {"controller_input": controller_stage},
    }


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_action_absorption_audit.main(["--help"])
    assert exc_info.value.code == 0


def test_clip_or_saturation_classification() -> None:
    payload = {
        "status": "consumed_but_absorbed_downstream",
        "samples": [
            {
                "sample_id": "clip-case",
                "debug_probe": {
                    "runtime_trace": _runtime_trace(
                        decoded=0.8,
                        absolute=0.7,
                        controller_input=0.0,
                    )
                },
                "per_group_stats": {
                    "left_arm": _group_payload(
                        representation="RELATIVE",
                        raw=0.9,
                        decoded=0.8,
                        absolute=0.7,
                        controller=0.0,
                        difference_disappeared_at="controller_input",
                        controller_absorbed=True,
                        controller_clip_rate=0.6,
                        saturation_rate=0.5,
                        reference_state_key="left_arm",
                    )
                },
            }
        ],
    }

    report = gr00t_action_absorption_audit.build_action_absorption_audit(payload)

    assert report["audit_status"] == "ready"
    assert report["summary"]["root_cause_counts"]["clip_or_saturation"] == 1
    assert report["summary"]["strongest_suspected_cause"] == "clip_or_saturation"
    sample = report["per_sample_audit"][0]
    assert sample["strongest_suspected_root_cause"] == "clip_or_saturation"
    assert sample["absorbed_dimensions"] == ["left_arm"]
    assert sample["delta_surfaces"]["decoded"]["value"] == 0.8
    assert sample["delta_surfaces"]["absolute"]["value"] == 0.7
    assert sample["delta_surfaces"]["controller"]["value"] == 0.0
    assert sample["clip_or_saturation_flags"]["any_clip_or_saturation"] is True
    assert sample["clip_or_saturation_flags"]["clip_or_saturation_groups"] == [
        "left_arm"
    ]


def test_relative_to_absolute_scaling_classification() -> None:
    payload = {
        "status": "consumed_but_absorbed_downstream",
        "samples": [
            {
                "sample_id": "scaling-case",
                "debug_probe": {
                    "runtime_trace": _runtime_trace(
                        decoded=0.8,
                        absolute=0.2,
                        controller_input=0.0,
                    )
                },
                "per_group_stats": {
                    "right_arm": _group_payload(
                        representation="RELATIVE",
                        raw=0.9,
                        decoded=0.8,
                        absolute=0.2,
                        controller=0.0,
                        difference_disappeared_at="relative_to_absolute",
                        controller_absorbed=False,
                        reference_state_key="right_arm",
                    )
                },
            }
        ],
    }

    report = gr00t_action_absorption_audit.build_action_absorption_audit(payload)

    sample = report["per_sample_audit"][0]
    group = sample["group_evidence"][0]
    assert sample["strongest_suspected_root_cause"] == "relative_to_absolute_scaling"
    assert group["suspected_root_cause"] == "relative_to_absolute_scaling"
    assert group["relative_to_absolute_scale_factors"]["absolute_over_decoded"] == 0.25
    assert (
        sample["relative_to_absolute_scale_factors"]["sample_level"][
            "absolute_over_decoded"
        ]
        == 0.25
    )


def test_controller_zeroing_or_masking_classification() -> None:
    payload = {
        "status": "consumed_but_absorbed_downstream",
        "samples": [
            {
                "sample_id": "zeroing-case",
                "debug_probe": {
                    "runtime_trace": _runtime_trace(
                        decoded=0.6,
                        absolute=0.6,
                        controller_input=0.0,
                    )
                },
                "per_group_stats": {
                    "right_hand": _group_payload(
                        representation="ABSOLUTE",
                        raw=0.6,
                        decoded=0.6,
                        absolute=0.6,
                        controller=0.0,
                        difference_disappeared_at="controller_input",
                        controller_absorbed=True,
                        baseline_zero_output_rate=1.0,
                        probe_zero_output_rate=1.0,
                        all_zero_in_both=True,
                    )
                },
            }
        ],
    }

    report = gr00t_action_absorption_audit.build_action_absorption_audit(payload)

    sample = report["per_sample_audit"][0]
    group = sample["group_evidence"][0]
    assert sample["strongest_suspected_root_cause"] == "controller_zeroing_or_masking"
    assert group["suspected_root_cause"] == "controller_zeroing_or_masking"
    assert group["controller_zeroing_or_masking"]["suspected"] is True
    assert sample["controller_zeroing_or_masking"]["suspected"] is True
    assert sample["controller_zeroing_or_masking"]["masked_groups"] == ["right_hand"]


def test_non_absorbed_status_skips_remediation_result() -> None:
    payload = {
        "status": "consumed_and_survived",
        "samples": [
            {
                "sample_id": "survived-case",
                "debug_probe": {
                    "runtime_trace": _runtime_trace(
                        decoded=0.7,
                        absolute=0.7,
                        controller_input=0.6,
                    )
                },
            }
        ],
    }

    report = gr00t_action_absorption_audit.build_action_absorption_audit(payload)

    assert report["audit_status"] == "skipped_non_absorbed_status"
    assert report["eligible_for_root_cause_audit"] is False
    assert report["per_sample_audit"] == []
    assert report["summary"]["strongest_suspected_cause"] is None


def test_main_writes_action_absorption_root_cause_json(tmp_path: Path) -> None:
    input_path = _write_json(
        tmp_path / "input.json",
        {
            "status": "consumed_but_absorbed_downstream",
            "samples": [
                {
                    "sample_id": "cli-case",
                    "debug_probe": {
                        "runtime_trace": _runtime_trace(
                            decoded=0.8,
                            absolute=0.2,
                            controller_input=0.0,
                        )
                    },
                    "per_group_stats": {
                        "left_arm": _group_payload(
                            representation="RELATIVE",
                            raw=0.8,
                            decoded=0.8,
                            absolute=0.2,
                            controller=0.0,
                            difference_disappeared_at="relative_to_absolute",
                            controller_absorbed=False,
                            reference_state_key="left_arm",
                        )
                    },
                }
            ],
        },
    )
    output_path = (
        tmp_path / gr00t_action_absorption_audit.ACTION_ABSORPTION_ROOT_CAUSE_JSON_NAME
    )

    exit_code = gr00t_action_absorption_audit.main(
        ["--input-json", str(input_path), "--out", str(output_path)]
    )

    written = _read_json(output_path)
    assert exit_code == 0
    assert written["default_output_filename"] == "action_absorption_root_cause.json"
    assert written["output_path"] == str(output_path.resolve())
    assert (
        written["summary"]["strongest_suspected_cause"]
        == "relative_to_absolute_scaling"
    )
