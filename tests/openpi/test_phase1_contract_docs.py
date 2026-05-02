from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DELTA = REPO_ROOT / "agent/exchange/openpi_phase1_contract_delta.md"
PARITY_MATRIX = REPO_ROOT / "agent/exchange/openpi_phase1_parity_deviation_matrix.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_contract_delta_exists_with_required_freeze_values() -> None:
    text = _read(CONTRACT_DELTA)
    required = [
        "# openpi Phase 1 合同冻结（contract delta）",
        "`gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc`",
        "`apple_to_plate_g1`",
        "`unitree_g1.LMPnPAppleToPlateDC`",
        "pick up the apple, walk left and place the apple on the plate.",
        "[20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009]",
        "`10`",
        "`1440`",
        "`20`",
        "`30`",
        "`pi05_droid`",
        "`recap_conditioned_prompt_token_v1`",
        "`prompt_text_only`",
        "`recompute_task_local_stats_primary`",
        "success_rate over frozen 10 episodes under identical protocol",
    ]
    for item in required:
        assert item in text, f"missing required contract freeze value: {item}"


def test_contract_delta_requires_provenance_and_evidence_taxonomy() -> None:
    text = _read(CONTRACT_DELTA)
    required = [
        "model_family=openpi",
        "model_anchor=pi05_droid",
        "norm_stats_policy=recompute_task_local_stats_primary",
        "prompt_route=recap_conditioned_prompt_token_v1",
        "conditioning_mode=prompt_text_only",
        "frozen_eval_episodes=10",
        "frozen_seed_manifest=[20000..20009]",
        "agent/runtime_logs/",
        "agent/artifacts/",
        ".sisyphus/evidence/",
        "agent/exchange/",
    ]
    for item in required:
        assert item in text, f"missing required provenance/evidence item: {item}"


def test_parity_matrix_exists_and_names_required_axes() -> None:
    text = _read(PARITY_MATRIX)
    required = [
        "# openpi Phase 1 parity / deviation matrix",
        "当前仓库 public anchor",
        "openpi `pi05_droid` 官方锚点",
        "Phase 1 冻结决定",
        "FROZEN-DEVIATION",
        "BLOCKER-IF-DRIFTING",
        "`gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc`",
        "`pi05_droid`",
        "`recap_conditioned_prompt_token_v1`",
        "`recompute_task_local_stats_primary`",
    ]
    for item in required:
        assert item in text, f"missing required parity-matrix axis or value: {item}"


def test_parity_matrix_documents_non_parity_claim_limits() -> None:
    text = _read(PARITY_MATRIX)
    required = [
        "benchmark-comparable evidence under documented deviations",
        "不允许表述：",
        "official openpi parity",
        "seed manifest 不是 `[20000..20009]`",
        "eval episodes 不是 `10`",
        "paired baseline/openpi 输出不在同一协议下生成",
    ]
    for item in required:
        assert item in text, f"missing required parity/deviation guardrail: {item}"
