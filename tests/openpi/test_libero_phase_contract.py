from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "agent/exchange/openpi_libero_phase_contract.md"


def test_libero_phase_contract_exists_and_freezes_new_mainline() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO-first 阶段合同",
        "openpi_pi05_libero_recap_state_tokens.md",
        "`pi05_libero`",
        "`LIBERO`",
        "`MuJoCo`",
        "`discrete_state_token`",
        "`future_migration_only`",
    ]
    for item in required:
        assert item in text, f"missing required LIBERO phase contract item: {item}"


def test_libero_phase_contract_explicitly_excludes_current_g1_scope() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    required = [
        "`UNITREE_G1`",
        "`Whole-Body Control (WBC)`",
        "humanoid real/sim migration",
        "G1 action contract adaptation",
        "不得继续把 `gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc` 当作当前阶段主锚点",
    ]
    for item in required:
        assert item in text, f"missing required G1 exclusion text: {item}"
