import json
from pathlib import Path


SUMMARY_PATH = Path("agent/artifacts/dual_track_20260424T042151Z/dual_track_summary.json")


def _summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_committed_dual_track_summary_keeps_next_actions_formal_gate_specific() -> None:
    summary = _summary()

    assert summary["schema_version"] == "dual_track_summary_v1"
    assert summary["gr00t"]["formal"]["status"] == "BLOCK"
    assert summary["gr00t"]["formal"]["formal_claim_allowed"] is False
    assert summary["openpi"]["formal"]["status"] == "BLOCK"
    assert summary["openpi"]["formal"]["formal_claim_allowed"] is False
    assert summary["next_actions"] == [
        "GR00T formal remains BLOCK; do not enter P5 until paired_seed_improvement_count reaches 2_of_3 under formal gate",
        "OpenPI formal remains BLOCK(dataset_not_materialized plus identity/label semantics blockers); reroute formal materialization before p1/p2 runtime",
        "Exploratory signals are diagnostic only and must not unlock formal gates",
    ]


def test_committed_dual_track_summary_rejects_exploratory_promotion_language() -> None:
    summary = _summary()
    combined_text = "\n".join(
        str(value)
        for value in [*summary["forbidden_inferences"], *summary["next_actions"]]
    )

    assert "exploratory signal != formal pass" in summary["forbidden_inferences"]
    assert "OpenPI exploratory dataset != formal materialized" in summary["forbidden_inferences"]
    assert "GR00T metric ablation/additional seed signal != P5 eligible" in summary["forbidden_inferences"]
    assert "formal pass" not in "\n".join(summary["next_actions"]).lower()
    assert "unlock formal" in combined_text
