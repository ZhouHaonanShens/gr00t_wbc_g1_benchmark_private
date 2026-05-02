from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "agent/exchange/openpi_libero_reuse_audit.md"


def test_libero_reuse_audit_exists_and_covers_four_required_files() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "# openpi LIBERO 复用边界审计",
        "`work/openpi/README.md`",
        "`work/openpi/serve/provenance.py`",
        "`work/openpi/eval/protocol.py`",
        "`work/openpi/scripts/phase05_smoke.py`",
        "`keep-as-structure`",
        "`rename`",
        "`rewrite`",
    ]
    for item in required:
        assert item in text, f"missing required reuse-audit coverage item: {item}"


def test_libero_reuse_audit_classifies_key_legacy_constants() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "`GR00T/WBC benchmark`",
        "`Phase1ServerProvenance`",
        "`build_phase1_health_payload`",
        "`pi05_droid`",
        "`policy_horizon=30`",
        "`executed_action_steps=20`",
        "`gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc`",
        "`apple_to_plate_g1`",
        "`openpi_phase1`",
        '`build_phase1_eval_artifact_paths("phase05_smoke")`',
        "`openpi_phase05`",
        "`task-8-phase05-smoke.md`",
    ]
    for item in required:
        assert item in text, (
            f"missing required legacy constant/classification item: {item}"
        )


def test_libero_reuse_audit_explains_boundary_decisions() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "可保留 / 组织边界",
        "必须重命名",
        "必须删除/重写",
        "只做审计与冻结边界",
        "结构可复用",
        "当前主线只允许 `LIBERO` / `MuJoCo`",
    ]
    for item in required:
        assert item in text, f"missing required boundary rationale: {item}"
