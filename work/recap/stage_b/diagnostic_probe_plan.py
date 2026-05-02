"""Static Stage B B1/B2 diagnostic probe support.

This module materializes probe *plans* only.  It does not import GR00T, launch
servers, call ``env.step``, or run rollouts.  The intent is to give later
workers a machine-checkable harness contract before any short diagnostic probe
is launched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .execution_boundaries import require_stage_b_safe_command


PROBE_PLAN_SCHEMA_VERSION = "stage_b_diagnostic_probe_plan_v1"
DEFAULT_MODES: tuple[str, ...] = ("positive", "omit", "negative")
DEFAULT_B1_SEED = 20000
DEFAULT_B2_SEEDS: tuple[int, ...] = (20000, 20001, 20002)
MAX_STAGE_B_DIAGNOSTIC_STEPS = 200


def _safe_command_json(command: Sequence[str]) -> dict[str, object]:
    decision = require_stage_b_safe_command(command)
    return decision.to_jsonable()


def build_b1_same_observation_one_step_plan() -> dict[str, Any]:
    """Return the Probe B1 same-observation one-step support plan."""

    command = (
        "timeout",
        "60",
        "python3",
        "agent/run/stage_b_probe.py",
        "--stage-b",
        "--same-observation",
        "--one-step",
        "--max-steps",
        "1",
    )
    return {
        "probe_id": "B1_same_obs_wbc_one_step_triplet",
        "probe_kind": "same_observation_wbc_one_step_triplet",
        "diagnostic_only": True,
        "formal_benchmark": False,
        "method_claim_allowed": False,
        "training_allowed": False,
        "checkpoint_update_allowed": False,
        "checkpoint": "internal_g3_checkpoint_6600",
        "modes": list(DEFAULT_MODES),
        "seeds": [DEFAULT_B1_SEED],
        "max_steps": 1,
        "n_envs": 1,
        "requires_timeout": True,
        "requires_same_observation": True,
        "requires_controller_reset_hash": True,
        "requires_chain_action_uuid": True,
        "requires_contrast_group_uuid": True,
        "stages": [
            "decoded_action",
            "post_transform_action",
            "controller_input",
            "wbc_internal_target",
            "controller_output_or_proxy",
            "env_applied_action",
            "qpos_next",
            "qvel_next",
        ],
        "outputs": [
            "probe_results/B1_same_obs_wbc_one_step_triplet.json",
            "probe_results/B1_same_obs_wbc_one_step_triplet.md",
            "plots/B1_delta_waterfall.png",
        ],
        "representative_command": list(command),
        "boundary_decision": _safe_command_json(command),
    }


def build_b2_short_closed_loop_plan() -> dict[str, Any]:
    """Return the Probe B2 short closed-loop triplet support plan."""

    command = (
        "timeout",
        "1800",
        "python3",
        "agent/run/stage_b_probe.py",
        "--stage-b",
        "--short-diagnostic",
        "--max-steps",
        str(MAX_STAGE_B_DIAGNOSTIC_STEPS),
        "--episodes",
        str(len(DEFAULT_B2_SEEDS)),
    )
    return {
        "probe_id": "B2_short_closed_loop_triplet",
        "probe_kind": "short_closed_loop_triplet",
        "diagnostic_only": True,
        "formal_benchmark": False,
        "method_claim_allowed": False,
        "training_allowed": False,
        "checkpoint_update_allowed": False,
        "modes": list(DEFAULT_MODES),
        "seeds": list(DEFAULT_B2_SEEDS),
        "max_steps": MAX_STAGE_B_DIAGNOSTIC_STEPS,
        "n_envs": 1,
        "default_gpu": 1,
        "requires_timeout": True,
        "requires_chain_action_uuid": True,
        "requires_contrast_group_uuid": True,
        "official_success_flag_role": "diagnostic_context_only",
        "required_reports": [
            "per_step_delta_survival",
            "controller_output_or_proxy_difference",
            "env_action_difference",
            "qpos_qvel_divergence",
            "object_eef_divergence",
            "first_failure_event",
            "collapse_observation",
        ],
        "outputs": [
            "probe_results/B2_short_closed_loop_triplet_summary.json",
            "probe_results/B2_short_closed_loop_triplet_summary.md",
            "seam_traces/B2_episode_<seed>_<mode>.jsonl",
            "plots/B2_delta_survival_seed_<seed>.png",
        ],
        "representative_command": list(command),
        "boundary_decision": _safe_command_json(command),
    }


def build_probe_support_plan() -> dict[str, Any]:
    """Build the Stage B B1/B2 probe support manifest."""

    return {
        "schema_version": PROBE_PLAN_SCHEMA_VERSION,
        "artifact_kind": "stage_b_diagnostic_probe_support_plan",
        "stage": "Stage B",
        "diagnostic_only": True,
        "formal_benchmark": False,
        "method_claim_allowed": False,
        "full_long_run_allowed": False,
        "training_allowed": False,
        "probes": [
            build_b1_same_observation_one_step_plan(),
            build_b2_short_closed_loop_plan(),
        ],
        "claim_boundary": [
            "official success flag may be recorded as diagnostic context only",
            "staged metrics must not replace task_success",
            "no benchmark or method success/failure claim",
        ],
    }


def write_probe_support_plan(path: str | Path) -> Path:
    """Write the B1/B2 support plan JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_probe_support_plan(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, help="Write the probe support plan JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.write is None:
        parser.print_help()
        return 0
    write_probe_support_plan(args.write)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

