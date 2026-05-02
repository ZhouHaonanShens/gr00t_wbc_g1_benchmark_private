from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "agent/exchange/openpi_libero_plan_reality_check.md"


def test_libero_plan_reality_check_exists_and_freezes_required_terms() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO 计划 reality check",
        "openpi_pi05_libero_recap_state_tokens.md",
        "`stock baseline`",
        "`RECAP-style`",
        "`state-token route`",
        "`RL token not in scope`",
    ]
    for item in required:
        assert item in text, f"missing required reality-check term: {item}"


def test_libero_plan_reality_check_contains_three_path_buckets() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    required = {
        "已存在": [
            "`agent/exchange/openpi_libero_phase_contract.md`",
            "`agent/exchange/openpi_libero_runtime.md`",
            "`tests/openpi/test_libero_phase_contract.py`",
            "`tests/openpi/test_libero_runtime_prereqs.py`",
            "`submodules/openpi/scripts/serve_policy.py`",
            "`submodules/openpi/examples/libero/main.py`",
            "`submodules/openpi/src/openpi/training/config.py`",
        ],
        "已存在但冲突": [
            "`work/openpi/README.md`",
            "`work/openpi/serve/provenance.py`",
            "`work/openpi/eval/protocol.py`",
            "`work/openpi/scripts/phase05_smoke.py`",
        ],
        "尚不存在": [
            "`work/openpi/scripts/libero_native_smoke.py`",
            "`tests/openpi/test_libero_server_provenance.py`",
            "`tests/openpi/test_libero_eval_protocol.py`",
            "`agent/exchange/openpi_libero_recap_io.md`",
            "`work/openpi/libero_recap/**`",
            "`work/openpi/scripts/libero_recap_eval.py`",
            "`agent/exchange/openpi_libero_state_token_contract.md`",
            "`work/openpi/libero_state_tokens/**`",
            "`agent/exchange/openpi_libero_results.md`",
            "`agent/exchange/openpi_g1_migration_prereqs.md`",
        ],
    }
    for heading, items in required.items():
        assert heading in text, f"missing heading: {heading}"
        for item in items:
            assert item in text, f"missing {heading} asset: {item}"


def test_libero_plan_reality_check_rejects_old_g1_assets_as_ready_base() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    required = [
        "旧 `G1/WBC` Phase 1",
        "不能被描述为 LIBERO-ready 基础",
        "这些文件的“文件存在”不能作为 LIBERO readiness 证据",
        "这些文件里的常量不能复制到 stock baseline",
        '`model_anchor="pi05_droid"`',
        "`gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc`",
    ]
    for item in required:
        assert item in text, f"missing old-asset rejection text: {item}"


def test_libero_plan_reality_check_makes_non_equivalence_explicit() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    required = [
        "`RECAP-style` 不等价于 `state-token route`",
        "`RECAP-style` 不等价于 `RL token`",
        "`state-token route` 不等价于 `RL token`",
        "`stock baseline` 也不等价于 `RECAP-style` 或 `state-token route`",
    ]
    for item in required:
        assert item in text, f"missing non-equivalence statement: {item}"
