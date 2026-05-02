from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "agent/exchange/openpi_libero_recap_io.md"


def test_libero_recap_contract_freezes_scope_and_non_equivalence() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO RECAP-style I/O 合同",
        "offline advantage-conditioned data/prompt relabeling",
        "repo-local approximate",
        "不宣称官方完整 RECAP 已在本仓库开源复现",
        "state-token 工作不在本任务范围，属于后续 Task 8",
        "smoke 产物与 comparison 产物都可以作为离线 relabel 输入来源",
    ]
    for item in required:
        assert item in text, f"missing recap scope item: {item}"


def test_libero_recap_contract_lists_required_inputs_and_label_sources() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "`observation.images.ego_view`",
        "`observation.state`",
        "`action`",
        "`prompt_raw`",
        "`episode_index`",
        "`step_index`",
        "`recap_m2.return_G`",
        "`recap_m2.value_V`",
        "`recap_m2.advantage_A`",
        "`recap_m2.advantage_input`",
        "`recap_m2.indicator_I`",
        "标签来源固定为离线 `recap_m2.return_G`、`recap_m2.value_V`、`recap_m2.advantage_A`、`recap_m2.advantage_input`、`recap_m2.indicator_I`。",
    ]
    for item in required:
        assert item in text, f"missing recap input/label item: {item}"


def test_libero_recap_contract_defines_output_schema_and_training_inference_consumption() -> (
    None
):
    text = DOC.read_text(encoding="utf-8")
    required = [
        '"schema_version": "openpi_libero_recap_record_v1"',
        '"observation/image": "from observation.images.ego_view"',
        '"observation/wrist_image": "duplicate observation.images.ego_view"',
        '"observation/state": "from observation.state"',
        '"action": "from action"',
        '"prompt": "training_prompt_text"',
        '"prompt_conditioned": "optional audit string only"',
        '"training_prompt_text": "canonical text built from prompt_raw + recap_m2.indicator_I"',
        '"prompt_route": "recap_conditioned_prompt_token_v1"',
        '"conditioning_mode": "prompt_text_only"',
        '"source_prompt_field": "prompt_raw"',
        "`work/openpi/data/contract_mapping.py`",
        "`work/openpi/prompting/routes.py`",
        "`build_phase1_prompt_route()` 固定从 `prompt_raw` 与 `recap_m2.indicator_I` 构造 canonical 文本。",
        "`work.recap.policy.TextIndicatorGr00tPolicy`",
        "`options['indicator_mode']`",
        "推理路径**不直接消费** `prompt_conditioned`。",
        "推理路径在本任务里**不直通** `recap_m2.advantage_input`；数值 advantage 仍然属于离线 relabel 证据与训练侧字段，不在本合同中扩张成新的 live inference API。",
    ]
    for item in required:
        assert item in text, f"missing recap schema/consumption item: {item}"


def test_libero_recap_contract_freezes_required_non_goals() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "`no value head`",
        "`no online loop`",
        "`no human correction UI`",
        "`no submodule patch`",
        "`no discrete_state_input=True`",
        "`no symbolic phase tokens`",
        "`no RL token`",
        "`no next-state head`",
        "不得修改 `submodules/openpi/**`",
    ]
    for item in required:
        assert item in text, f"missing recap non-goal item: {item}"
