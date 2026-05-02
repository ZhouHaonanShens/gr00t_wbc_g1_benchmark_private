from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_SNAPSHOT_CANDIDATES = Path(
    "agent/artifacts/state_conditioned_materialization/harvest/snapshot_candidates.jsonl"
)
DEFAULT_OUTPUT = Path(
    "agent/artifacts/state_conditioned_materialization/sanity/teacher_upper_bound_report.json"
)
DEFAULT_BASELINE_TEACHER_GATE_REPORT = Path(
    "agent/artifacts/state_conditioned_materialization/harvest/teacher_gate_report.json"
)

REPORT_SCHEMA_VERSION = "g1_state_conditioned_teacher_upper_bound_sanity_v1"
REPORT_ARTIFACT_KIND = "state_conditioned_teacher_upper_bound_report"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap import state_conditioned_snapshot_harvest


PHASE_DEPTH_INDEX = {
    str(phase): index
    for index, phase in enumerate(
        state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES
    )
}


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="state_conditioned_teacher_upper_bound_sanity.py",
        description=(
            "Run the scripted teacher upper-bound sanity harness on the frozen "
            "state-conditioned snapshot curriculum and emit a machine-readable retrain gate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _build_parser()
    parser.add_argument(
        "--snapshot-candidates",
        type=Path,
        default=DEFAULT_SNAPSHOT_CANDIDATES,
        help="Snapshot curriculum JSONL used as teacher upper-bound input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path for teacher_upper_bound_report.json.",
    )
    parser.add_argument(
        "--baseline-teacher-gate-report",
        type=Path,
        default=DEFAULT_BASELINE_TEACHER_GATE_REPORT,
        help=(
            "Existing T8 teacher_gate_report.json used only as the current baseline "
            "comparison to distinguish teacher-unreachable from model-not-learned."
        ),
    )
    return parser


def _validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"missing required {arg_name}: {resolved}")
    return resolved


def _validate_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError("--output must be a JSON file path, not a directory")
    if not resolved.suffix:
        raise ValueError("--output must include a .json filename")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    return Path(state_conditioned_bucket_a_import._write_json(path, payload))


def _read_json(path: Path) -> dict[str, Any]:
    return state_conditioned_bucket_a_import._read_json(path)


def _resolve_baseline_teacher_gate_report(
    snapshot_candidates_path: Path,
    baseline_teacher_gate_report_path: Path,
) -> Path:
    explicit_path = baseline_teacher_gate_report_path.expanduser().resolve()
    if explicit_path.is_file():
        return explicit_path
    sibling_path = (
        snapshot_candidates_path.expanduser().resolve().parent
        / state_conditioned_snapshot_harvest.TEACHER_GATE_REPORT_JSON_NAME
    )
    return _validate_existing_file(
        sibling_path,
        arg_name="baseline teacher_gate_report",
    )


def _selected_snapshot_count(
    *,
    family: str,
    eligible_count: int,
    deprioritized_by_plan: bool,
) -> int:
    if family in state_conditioned_snapshot_harvest.HIGH_PRIORITY_FAMILIES:
        target_count = int(state_conditioned_snapshot_harvest.SNAPSHOTS_PER_FAMILY)
    elif deprioritized_by_plan:
        target_count = int(
            state_conditioned_snapshot_harvest.DEPRIORITIZED_LOW_PRIORITY_SNAPSHOTS
        )
    else:
        target_count = int(state_conditioned_snapshot_harvest.SNAPSHOTS_PER_FAMILY)
    return min(int(target_count), int(eligible_count))


def _selected_candidates_by_family(
    grouped_snapshot_candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized_grouped = (
        state_conditioned_snapshot_harvest._normalize_grouped_snapshot_candidates(
            grouped_snapshot_candidates
        )
    )
    selected: dict[str, dict[str, Any]] = {}
    for family in state_conditioned_snapshot_harvest.T8_FAMILY_ORDER:
        group = dict(normalized_grouped[family])
        eligible = [dict(item) for item in list(group.get("eligible", []))]
        ineligible = [dict(item) for item in list(group.get("ineligible", []))]
        deprioritized_flags = {
            bool(item) for item in set(group.get("deprioritized_flags", set()))
        }
        if len(deprioritized_flags) > 1:
            raise ValueError(
                f"family {family} mixes deprioritized_by_plan=true and false across candidates"
            )
        deprioritized_by_plan = next(iter(deprioritized_flags), False)
        if not eligible:
            raise ValueError(f"family {family} has no eligible snapshot candidates")
        selected_count = _selected_snapshot_count(
            family=family,
            eligible_count=len(eligible),
            deprioritized_by_plan=deprioritized_by_plan,
        )
        if (
            family in state_conditioned_snapshot_harvest.HIGH_PRIORITY_FAMILIES
            and selected_count
            != int(state_conditioned_snapshot_harvest.SNAPSHOTS_PER_FAMILY)
        ):
            raise ValueError(
                f"high-priority family {family} requires exactly {state_conditioned_snapshot_harvest.SNAPSHOTS_PER_FAMILY} eligible snapshots"
            )
        selected[family] = {
            "eligible": eligible,
            "ineligible": ineligible,
            "selected": eligible[:selected_count],
            "deprioritized_by_plan": bool(deprioritized_by_plan),
        }
    return selected


def _phase_depth_summary(
    steps: Sequence[Mapping[str, Any]],
    *,
    fallback_phase: str,
) -> dict[str, Any]:
    max_phase = str(fallback_phase)
    max_phase_index = int(PHASE_DEPTH_INDEX.get(max_phase, 0))
    for step in steps:
        raw_phase = step.get(
            "policy_condition.phase", step.get("phase", fallback_phase)
        )
        try:
            normalized_phase = state_conditioned_snapshot_harvest._normalize_phase(
                raw_phase,
                field_name="attempt_result.policy_condition.phase",
            )
        except (TypeError, ValueError):
            normalized_phase = str(fallback_phase)
        phase_index = int(PHASE_DEPTH_INDEX.get(normalized_phase, 0))
        if phase_index >= max_phase_index:
            max_phase_index = phase_index
            max_phase = str(normalized_phase)
    phase_count = max(
        1, len(state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES) - 1
    )
    return {
        "max_phase": str(max_phase),
        "max_phase_index": int(max_phase_index),
        "max_phase_progress": float(max_phase_index / float(phase_count)),
    }


def _transport_progress(
    candidate: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchor_xy_distance = state_conditioned_snapshot_harvest._as_number(
        candidate.get("anchor_xy_distance"),
        field_name="candidate.anchor_xy_distance",
    )
    streak = 0
    best_improvement = 0.0
    best_combined_progress = 0.0
    max_in_hand_streak = 0
    for step in steps:
        in_hand = state_conditioned_snapshot_harvest._step_bool(
            step,
            "privileged.apple_in_hand",
        )
        if not in_hand:
            streak = 0
            best_improvement = 0.0
            best_combined_progress = max(best_combined_progress, 0.0)
            continue
        streak += 1
        max_in_hand_streak = max(max_in_hand_streak, int(streak))
        current_xy_distance = state_conditioned_snapshot_harvest._extract_xy_distance(
            step.get("privileged.apple_to_plate_rel_pose"),
            field_name="privileged.apple_to_plate_rel_pose",
        )
        best_improvement = max(
            float(best_improvement),
            float(anchor_xy_distance - current_xy_distance),
        )
        streak_progress = min(1.0, float(streak) / 8.0)
        distance_progress = min(1.0, max(0.0, float(best_improvement)) / 0.05)
        best_combined_progress = max(
            float(best_combined_progress),
            min(float(streak_progress), float(distance_progress)),
        )
    return {
        "progress_score": float(best_combined_progress),
        "max_in_hand_streak": int(max_in_hand_streak),
        "best_xy_improvement_m": float(best_improvement),
    }


def _attempt_progress_summary(
    *,
    family: str,
    candidate: Mapping[str, Any],
    attempt_result: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_family = state_conditioned_snapshot_harvest._normalize_family(family)
    steps = state_conditioned_snapshot_harvest._policy_steps_from_attempt_result(
        attempt_result
    )
    fallback_phase = str(candidate.get("policy_condition.phase", "SEARCH"))
    phase_depth = _phase_depth_summary(steps, fallback_phase=fallback_phase)
    success = state_conditioned_snapshot_harvest.evaluate_feasibility_success(
        family=normalized_family,
        candidate=candidate,
        attempt_result=attempt_result,
    )

    if normalized_family == "S_lost":
        max_visible_streak = 0
        current_visible_streak = 0
        for step in steps:
            if state_conditioned_snapshot_harvest._step_bool(
                step, "privileged.apple_visible"
            ):
                current_visible_streak += 1
                max_visible_streak = max(max_visible_streak, current_visible_streak)
            else:
                current_visible_streak = 0
        progress = min(1.0, float(max_visible_streak) / 4.0)
        return {
            "success": bool(success),
            "progress_score": float(progress),
            "phase_depth": phase_depth,
            "max_visible_streak": int(max_visible_streak),
        }

    if normalized_family == "S_drop":
        max_in_hand_streak = 0
        current_in_hand_streak = 0
        for step in steps:
            if state_conditioned_snapshot_harvest._step_bool(
                step, "privileged.apple_in_hand"
            ):
                current_in_hand_streak += 1
                max_in_hand_streak = max(max_in_hand_streak, current_in_hand_streak)
            else:
                current_in_hand_streak = 0
        progress = min(1.0, float(max_in_hand_streak) / 8.0)
        return {
            "success": bool(success),
            "progress_score": float(progress),
            "phase_depth": phase_depth,
            "max_in_hand_streak": int(max_in_hand_streak),
        }

    if normalized_family == "S_transport_mid":
        transport_progress = _transport_progress(candidate, steps)
        return {
            "success": bool(success),
            "progress_score": float(transport_progress["progress_score"]),
            "phase_depth": phase_depth,
            "max_in_hand_streak": int(transport_progress["max_in_hand_streak"]),
            "best_xy_improvement_m": float(transport_progress["best_xy_improvement_m"]),
        }

    contact_seen = any(
        state_conditioned_snapshot_harvest._step_bool(step, "privileged.contact_flag")
        for step in steps
    )
    return {
        "success": bool(success),
        "progress_score": float(1.0 if contact_seen else 0.0),
        "phase_depth": phase_depth,
        "contact_seen": bool(contact_seen),
        "success_episode": bool(attempt_result.get("success_episode", False)),
    }


def _rate(success_count: int, attempt_count: int) -> float:
    if int(attempt_count) <= 0:
        return 0.0
    return float(success_count) / float(attempt_count)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / float(len(values)))


def _max(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(max(float(value) for value in values))


def _family_interpretation(
    *,
    teacher_success_count: int,
    teacher_max_progress: float,
    baseline_success_count: int,
) -> tuple[str, str]:
    if int(teacher_success_count) <= 0:
        if float(teacher_max_progress) > 0.0:
            return (
                "teacher_unreachable_on_snapshots_partial_progress_only",
                "scripted teacher showed partial progress but never met the frozen family success criterion on the selected snapshots",
            )
        return (
            "teacher_unreachable_on_snapshots_no_progress",
            "scripted teacher never met the frozen family success criterion and showed no measurable family-specific progress on the selected snapshots",
        )
    if int(baseline_success_count) <= 0:
        return (
            "teacher_reachable_model_not_learned",
            "scripted teacher can recover at least one selected snapshot while the current baseline remains at zero on the same family",
        )
    return (
        "teacher_reachable_and_model_nonzero",
        "scripted teacher and the current baseline both achieve non-zero family success on the selected snapshots",
    )


def _overall_gate(
    *,
    total_teacher_success_count: int,
    total_baseline_success_count: int,
) -> tuple[bool, str, str]:
    if int(total_teacher_success_count) <= 0:
        return (
            False,
            "block_teacher_unreachable_on_snapshot_curriculum",
            "BLOCK: scripted teacher reached zero successes across all selected snapshot curriculum attempts, so the current all-zero result still points first to snapshot curriculum / teacher semantics reachability risk rather than model learning failure.",
        )
    if int(total_baseline_success_count) <= 0:
        return (
            True,
            "allow_teacher_reachable_model_currently_zero",
            "ALLOW: scripted teacher is non-zero on the selected snapshots while the current baseline remains zero, so a retrain is justified to test whether the model can absorb the available recovery signal.",
        )
    return (
        True,
        "allow_teacher_and_model_both_nonzero",
        "ALLOW: scripted teacher is non-zero and the current baseline is already non-zero on at least part of the selected snapshot curriculum.",
    )


def build_teacher_upper_bound_report(
    *,
    snapshot_candidates_path: Path,
    baseline_teacher_gate_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    resolved_snapshot_candidates_path = _validate_existing_file(
        snapshot_candidates_path,
        arg_name="snapshot-candidates",
    )
    resolved_output_path = _validate_output_path(output_path)
    resolved_baseline_report_path = _resolve_baseline_teacher_gate_report(
        resolved_snapshot_candidates_path,
        baseline_teacher_gate_report_path,
    )

    grouped_snapshot_candidates = (
        state_conditioned_snapshot_harvest.load_snapshot_candidates(
            resolved_snapshot_candidates_path
        )
    )
    selected_by_family = _selected_candidates_by_family(grouped_snapshot_candidates)

    baseline_teacher_gate_report = _read_json(resolved_baseline_report_path)
    if (
        baseline_teacher_gate_report.get("artifact_kind")
        != "state_conditioned_teacher_gate_report"
    ):
        raise ValueError(
            "baseline teacher_gate_report artifact_kind mismatch; expected state_conditioned_teacher_gate_report"
        )
    if baseline_teacher_gate_report.get("mode") != "feasibility":
        raise ValueError("baseline teacher_gate_report mode mismatch")
    if list(baseline_teacher_gate_report.get("family_order", [])) != list(
        state_conditioned_snapshot_harvest.T8_FAMILY_ORDER
    ):
        raise ValueError("baseline teacher_gate_report family_order mismatch")
    baseline_threshold = state_conditioned_snapshot_harvest._as_number(
        baseline_teacher_gate_report.get("teacher_threshold"),
        field_name="baseline_teacher_gate_report.teacher_threshold",
    )
    frozen_threshold = float(
        state_conditioned_snapshot_harvest.DEFAULT_TEACHER_THRESHOLD
    )
    if abs(float(baseline_threshold) - float(frozen_threshold)) > 1e-12:
        raise ValueError(
            f"baseline teacher threshold drifted: expected {frozen_threshold}, got {baseline_threshold}"
        )
    baseline_families = (
        state_conditioned_snapshot_harvest._normalize_report_family_rows(
            baseline_teacher_gate_report.get("families"),
            report_name="baseline_teacher_gate_report",
        )
    )

    family_reports: list[dict[str, Any]] = []
    total_input_snapshot_count = 0
    total_eligible_candidate_count = 0
    total_ineligible_candidate_count = 0
    total_selected_snapshot_count = 0
    total_attempt_count = 0
    total_success_count = 0
    total_baseline_attempt_count = 0
    total_baseline_success_count = 0

    for family in state_conditioned_snapshot_harvest.T8_FAMILY_ORDER:
        family_group = selected_by_family[family]
        eligible = [dict(item) for item in family_group["eligible"]]
        ineligible = [dict(item) for item in family_group["ineligible"]]
        selected_candidates = [dict(item) for item in family_group["selected"]]
        deprioritized_by_plan = bool(family_group["deprioritized_by_plan"])

        baseline_row = dict(baseline_families[family])
        baseline_attempt_count = state_conditioned_snapshot_harvest._as_int(
            baseline_row.get("attempt_count"),
            field_name=f"baseline_teacher_gate_report.{family}.attempt_count",
        )
        baseline_success_count = state_conditioned_snapshot_harvest._as_int(
            baseline_row.get("success_count"),
            field_name=f"baseline_teacher_gate_report.{family}.success_count",
        )
        baseline_success_rate = state_conditioned_snapshot_harvest._as_number(
            baseline_row.get("success_rate"),
            field_name=f"baseline_teacher_gate_report.{family}.success_rate",
        )

        snapshot_rows: list[dict[str, Any]] = []
        family_attempt_count = 0
        family_success_count = 0
        progress_scores: list[float] = []
        phase_depth_scores: list[float] = []

        for candidate in selected_candidates:
            snapshot_id = state_conditioned_snapshot_harvest._as_non_empty_string(
                candidate.get("snapshot_id"),
                field_name="candidate.snapshot_id",
            )
            snapshot_success_count = 0
            snapshot_attempt_count = 0
            snapshot_progress_scores: list[float] = []
            snapshot_phase_depth_scores: list[float] = []
            per_seed: list[dict[str, Any]] = []

            for seed in state_conditioned_snapshot_harvest.SNAPSHOT_SEED_VALUES:
                attempt_result = state_conditioned_snapshot_harvest._resolve_formal_attempt_result(
                    candidate,
                    seed=int(seed),
                    family=family,
                    producer=state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER,
                    formal_runner=None,
                )
                progress_summary = _attempt_progress_summary(
                    family=family,
                    candidate=candidate,
                    attempt_result=attempt_result,
                )
                attempt_success = bool(progress_summary["success"])
                progress_score = float(progress_summary["progress_score"])
                phase_depth_progress = float(
                    dict(progress_summary["phase_depth"]).get("max_phase_progress", 0.0)
                )
                if attempt_success:
                    snapshot_success_count += 1
                    family_success_count += 1
                snapshot_attempt_count += 1
                family_attempt_count += 1
                snapshot_progress_scores.append(progress_score)
                progress_scores.append(progress_score)
                snapshot_phase_depth_scores.append(phase_depth_progress)
                phase_depth_scores.append(phase_depth_progress)
                per_seed.append(
                    {
                        "seed": int(seed),
                        "success": bool(attempt_success),
                        "progress_score": float(progress_score),
                        "phase_depth": dict(progress_summary["phase_depth"]),
                    }
                )

            snapshot_rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "anchor_episode_id": str(candidate.get("anchor_episode_id", "")),
                    "anchor_t": int(candidate.get("anchor_t", 0)),
                    "attempt_count": int(snapshot_attempt_count),
                    "success_count": int(snapshot_success_count),
                    "reachable_rate": float(
                        _rate(snapshot_success_count, snapshot_attempt_count)
                    ),
                    "progress": {
                        "mean": float(_mean(snapshot_progress_scores)),
                        "max": float(_max(snapshot_progress_scores)),
                    },
                    "phase_depth": {
                        "mean": float(_mean(snapshot_phase_depth_scores)),
                        "max": float(_max(snapshot_phase_depth_scores)),
                    },
                    "per_seed": per_seed,
                }
            )

        family_reachable_rate = _rate(family_success_count, family_attempt_count)
        family_progress_mean = _mean(progress_scores)
        family_progress_max = _max(progress_scores)
        family_phase_depth_mean = _mean(phase_depth_scores)
        family_phase_depth_max = _max(phase_depth_scores)
        family_meets_threshold = float(family_reachable_rate) >= float(frozen_threshold)
        interpretation_code, interpretation = _family_interpretation(
            teacher_success_count=family_success_count,
            teacher_max_progress=family_progress_max,
            baseline_success_count=baseline_success_count,
        )

        family_reports.append(
            {
                "family": family,
                "priority": state_conditioned_snapshot_harvest._priority_for_family(
                    family
                ),
                "success_criteria": state_conditioned_snapshot_harvest.FAMILY_SUCCESS_CRITERIA[
                    family
                ],
                "threshold": float(frozen_threshold),
                "teacher_meets_threshold": bool(family_meets_threshold),
                "deprioritized_by_plan": bool(deprioritized_by_plan),
                "input_snapshot_count": int(len(eligible) + len(ineligible)),
                "eligible_candidate_count": int(len(eligible)),
                "ineligible_candidate_count": int(len(ineligible)),
                "selected_snapshot_count": int(len(selected_candidates)),
                "selected_snapshot_ids": [
                    state_conditioned_snapshot_harvest._as_non_empty_string(
                        candidate.get("snapshot_id"),
                        field_name="candidate.snapshot_id",
                    )
                    for candidate in selected_candidates
                ],
                "seed_values": [
                    int(seed)
                    for seed in state_conditioned_snapshot_harvest.SNAPSHOT_SEED_VALUES
                ],
                "attempt_count": int(family_attempt_count),
                "success_count": int(family_success_count),
                "reachable_rate": float(family_reachable_rate),
                "progress": {
                    "mean": float(family_progress_mean),
                    "max": float(family_progress_max),
                },
                "phase_depth": {
                    "mean": float(family_phase_depth_mean),
                    "max": float(family_phase_depth_max),
                },
                "current_model_baseline": {
                    "attempt_count": int(baseline_attempt_count),
                    "success_count": int(baseline_success_count),
                    "success_rate": float(baseline_success_rate),
                    "teacher_fallback_enabled": bool(
                        baseline_row.get("teacher_fallback_enabled", False)
                    ),
                },
                "interpretation_code": str(interpretation_code),
                "interpretation": str(interpretation),
                "snapshots": snapshot_rows,
            }
        )

        total_input_snapshot_count += int(len(eligible) + len(ineligible))
        total_eligible_candidate_count += int(len(eligible))
        total_ineligible_candidate_count += int(len(ineligible))
        total_selected_snapshot_count += int(len(selected_candidates))
        total_attempt_count += int(family_attempt_count)
        total_success_count += int(family_success_count)
        total_baseline_attempt_count += int(baseline_attempt_count)
        total_baseline_success_count += int(baseline_success_count)

    allow_retrain, reason_code, reason = _overall_gate(
        total_teacher_success_count=total_success_count,
        total_baseline_success_count=total_baseline_success_count,
    )
    teacher_all_zero = int(total_success_count) <= 0
    overall_reachable_rate = _rate(total_success_count, total_attempt_count)
    overall_baseline_success_rate = _rate(
        total_baseline_success_count,
        total_baseline_attempt_count,
    )
    overall_interpretation = (
        "teacher_unreachable_on_snapshot_curriculum"
        if teacher_all_zero
        else (
            "teacher_reachable_model_not_learned_yet"
            if int(total_baseline_success_count) <= 0
            else "teacher_reachable_and_model_nonzero"
        )
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "teacher_version": state_conditioned_snapshot_harvest.DEFAULT_TEACHER_VERSION,
        "teacher_threshold": float(frozen_threshold),
        "snapshot_candidates_path": str(resolved_snapshot_candidates_path),
        "baseline_teacher_gate_report_path": str(resolved_baseline_report_path),
        "output_path": str(resolved_output_path),
        "family_order": list(state_conditioned_snapshot_harvest.T8_FAMILY_ORDER),
        "counts": {
            "family_count": int(
                len(state_conditioned_snapshot_harvest.T8_FAMILY_ORDER)
            ),
            "input_snapshot_count": int(total_input_snapshot_count),
            "eligible_candidate_count": int(total_eligible_candidate_count),
            "ineligible_candidate_count": int(total_ineligible_candidate_count),
            "selected_snapshot_count": int(total_selected_snapshot_count),
            "seed_count": int(
                len(state_conditioned_snapshot_harvest.SNAPSHOT_SEED_VALUES)
            ),
            "attempt_count": int(total_attempt_count),
            "success_count": int(total_success_count),
            "reachable_family_count": int(
                sum(1 for row in family_reports if int(row["success_count"]) > 0)
            ),
            "families_meeting_threshold_count": int(
                sum(1 for row in family_reports if bool(row["teacher_meets_threshold"]))
            ),
        },
        "current_model_baseline": {
            "attempt_count": int(total_baseline_attempt_count),
            "success_count": int(total_baseline_success_count),
            "success_rate": float(overall_baseline_success_rate),
        },
        "teacher_upper_bound": {
            "attempt_count": int(total_attempt_count),
            "success_count": int(total_success_count),
            "reachable_rate": float(overall_reachable_rate),
            "teacher_all_zero": bool(teacher_all_zero),
        },
        "overall_interpretation": str(overall_interpretation),
        "gate": {
            "allow_retrain": bool(allow_retrain),
            "status": "ALLOW" if allow_retrain else "BLOCK",
            "reason_code": str(reason_code),
            "reason": str(reason),
            "decision_basis": (
                "BLOCK only when scripted teacher success_count stays zero across the full selected snapshot curriculum; otherwise allow retrain while preserving the teacher-vs-model distinction."
            ),
        },
        "families": family_reports,
    }
    _write_json(resolved_output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_teacher_upper_bound_report(
        snapshot_candidates_path=args.snapshot_candidates,
        baseline_teacher_gate_report_path=args.baseline_teacher_gate_report,
        output_path=args.output,
    )
    gate = dict(report["gate"])
    upper_bound = dict(report["teacher_upper_bound"])
    print(
        json.dumps(
            {
                "output_path": report["output_path"],
                "allow_retrain": gate["allow_retrain"],
                "status": gate["status"],
                "reason_code": gate["reason_code"],
                "teacher_success_count": upper_bound["success_count"],
                "teacher_reachable_rate": upper_bound["reachable_rate"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
