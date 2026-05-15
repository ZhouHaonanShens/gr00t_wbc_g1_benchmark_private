from __future__ import annotations

import json
from pathlib import Path

from work.recap.safe_sft import t8_1_nav_postlift as t8


def _summary(
    row_id: str,
    *,
    episodes: int,
    reached: int = 0,
    success: int = 0,
    lifted: int = 0,
) -> dict[str, object]:
    return {
        "ID": row_id,
        "policy": row_id,
        "episodes": episodes,
        "reached": reached,
        "success": success,
        "lifted": lifted,
        "failure_modes": {},
    }


def test_p_splice_material_thresholds_are_qualitative_below_ten() -> None:
    payload = t8.build_splice_material_improvement(
        [
            _summary("P0", episodes=9, reached=0),
            _summary("P1", episodes=9, reached=2),
        ],
        baseline_id="P0",
        splice_kind="post_lift_splice",
    )

    row = payload["rows"][0]
    assert payload["post_lift_splice_material_improvement"] is False
    assert payload["qualitative_only"] is True
    assert row["threshold_met"] is True
    assert row["material_improvement"] is False
    assert row["evidence_mode"] == "qualitative_only"
    assert row["threshold_not_applied_reason"] == "requires_n_ge_10_per_arm"


def test_p_splice_material_thresholds_apply_at_exactly_ten() -> None:
    payload = t8.build_splice_material_improvement(
        [
            _summary("P0", episodes=10, reached=0),
            _summary("P1", episodes=10, reached=2),
            _summary("N1", episodes=10, reached=8),
        ],
        baseline_id="P0",
        splice_kind="post_lift_splice",
    )

    by_id = {row["ID"]: row for row in payload["rows"]}
    assert payload["post_lift_splice_material_improvement"] is True
    assert by_id["P1"]["evidence_mode"] == "quantitative_threshold"
    assert by_id["P1"]["material_improvement"] is True
    assert by_id["N1"]["baseline_id"] == "P0"


def test_missing_p0_baseline_fails_closed_for_p_splice_material() -> None:
    payload = t8.build_splice_material_improvement(
        [_summary("P1", episodes=10, reached=2)],
        baseline_id="P0",
        splice_kind="post_lift_splice",
    )

    assert payload["status"] == "FAIL"
    assert payload["post_lift_splice_material_improvement"] is False
    assert payload["blocking_reasons"] == ["missing_baseline_row"]


def test_decide_final_does_not_promote_qualitative_nav_delta() -> None:
    base_rows = [
        _summary("B0", episodes=9, success=1),
        _summary("B1", episodes=9, success=1),
        _summary("B2", episodes=9, success=1),
    ]
    nav_rows = [
        _summary("N0", episodes=9, reached=0),
        _summary("N1", episodes=9, reached=2),
    ]

    assert (
        t8.decide_final(base_rows, nav_rows, {"answers": {"lifted_but_not_success_count": 0}}, True)
        == "GUARDED_RECAP_STILL_FORBIDDEN"
    )


def test_decide_final_promotes_quantitative_nav_delta_at_ten() -> None:
    base_rows = [
        _summary("B0", episodes=10, success=1),
        _summary("B1", episodes=10, success=1),
        _summary("B2", episodes=10, success=1),
    ]
    nav_rows = [
        _summary("N0", episodes=10, reached=0),
        _summary("N1", episodes=10, reached=2),
    ]

    assert (
        t8.decide_final(base_rows, nav_rows, {"answers": {"lifted_but_not_success_count": 0}}, True)
        == "NAV_SPLICE_IMPROVES"
    )


def _write_lifted_episode(path: Path, *, seed: int, success: bool = False) -> None:
    rows = [
        {
            "source": "fixture",
            "seed": seed,
            "outer_step": 0,
            "success_step": False,
            "apple_height_z": 0.10,
            "apple_to_plate_l2": 0.30,
            "apple_to_right_eef_l2": 0.20,
            "right_hand_q99": 0.40,
        },
        {
            "source": "fixture",
            "seed": seed,
            "outer_step": 1,
            "success_step": success,
            "apple_height_z": 0.14,
            "apple_to_plate_l2": 0.25,
            "apple_to_right_eef_l2": 0.08,
            "right_hand_q99": 0.20,
        },
    ]
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_post_lift_audit_excludes_success_and_marks_small_n_qualitative(tmp_path: Path) -> None:
    source = tmp_path / "steps.jsonl"
    _write_lifted_episode(source, seed=1)
    _write_lifted_episode(source, seed=2)
    _write_lifted_episode(source, seed=3, success=True)

    payload = t8.post_lift_audit(tmp_path / "audit", {"fixture": source})

    answers = payload["answers"]
    assert answers["lifted_but_not_success_count"] == 2
    assert answers["evidence_mode"] == "qualitative_only"
    assert answers["quantitative_threshold_eligible"] is False
    assert answers["threshold_not_applied_reason"] == "requires_n_ge_10_post_lift_cases"
