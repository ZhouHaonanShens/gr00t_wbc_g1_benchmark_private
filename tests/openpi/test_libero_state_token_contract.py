from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "agent/exchange/openpi_libero_state_token_contract.md"
PI0_CONFIG = REPO_ROOT / "submodules/openpi/src/openpi/models/pi0_config.py"
TRANSFORMS = REPO_ROOT / "submodules/openpi/src/openpi/transforms.py"
TOKENIZER = REPO_ROOT / "submodules/openpi/src/openpi/models/tokenizer.py"
TRAINING_CONFIG = REPO_ROOT / "submodules/openpi/src/openpi/training/config.py"
DATA_LOADER = REPO_ROOT / "submodules/openpi/src/openpi/training/data_loader.py"
POLICY_CONFIG = REPO_ROOT / "submodules/openpi/src/openpi/policies/policy_config.py"
LIBERO_POLICY = REPO_ROOT / "submodules/openpi/src/openpi/policies/libero_policy.py"


def test_state_token_contract_freezes_only_native_route() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO state-token contract",
        "原生 `discrete_state_input=True`",
        "Task 8 是后续实验偏差，不是默认 baseline",
        "`normalized 8D state -> native tokenizer -> discrete tokens`",
        "不引入 second tokenizer/vocabulary work",
        "stock `pi05_libero` 配置固定为 `discrete_state_input=False`",
    ]
    for item in required:
        assert item in text, f"missing native route item: {item}"


def test_state_token_contract_freezes_8d_source_state_and_transform_order() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "source state 固定为 **normalized raw 8D LIBERO `observation/state`**。",
        "`source state = normalized raw 8D LIBERO observation/state`",
        "`not padded 32D internals`",
        "`Normalize(...)` 先对该 state 做归一化。",
        "`TokenizePrompt(... discrete_state_input=True)` 再把归一化后的 state 传给原生 tokenizer。",
        "`PaligemmaTokenizer.tokenize(prompt, state)` 用原生离散化逻辑产出离散 token。",
        "`PadStatesAndActions` 的 32 维对齐只属于模型内部 action/state 维度补齐，不得回写成本合同的 source state 定义。",
        "`normalize first, tokenize second`",
    ]
    for item in required:
        assert item in text, f"missing state source/order item: {item}"


def test_state_token_contract_explicitly_rejects_non_native_tokens_and_heads() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "`no symbolic phase token`",
        "`no task phase id`",
        "`no RL token`",
        "`no next-state head`",
        "`no custom token vocabulary`",
        "`no second tokenizer`",
        "`no custom tokenizer`",
        "`this task does not change RECAP-style semantics`",
        "`state-token route is separate from RECAP-style`",
    ]
    for item in required:
        assert item in text, f"missing rejection/boundary item: {item}"


def test_upstream_sources_still_support_the_frozen_native_route_shape() -> None:
    libero_policy_text = LIBERO_POLICY.read_text(encoding="utf-8")
    data_loader_text = DATA_LOADER.read_text(encoding="utf-8")
    policy_config_text = POLICY_CONFIG.read_text(encoding="utf-8")
    transforms_text = TRANSFORMS.read_text(encoding="utf-8")
    tokenizer_text = TOKENIZER.read_text(encoding="utf-8")
    training_config_text = TRAINING_CONFIG.read_text(encoding="utf-8")
    pi0_config_text = PI0_CONFIG.read_text(encoding="utf-8")

    assert '"observation/state": np.random.rand(8)' in libero_policy_text
    assert '"state": data["observation/state"]' in libero_policy_text

    assert (
        "_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),"
        in data_loader_text
    )
    assert "*data_config.model_transforms.inputs," in data_loader_text
    assert (
        "transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),"
        in policy_config_text
    )
    assert "*data_config.model_transforms.inputs," in policy_config_text

    assert "class TokenizePrompt(DataTransformFn):" in transforms_text
    assert "discrete_state_input: bool = False" in transforms_text
    assert "if self.discrete_state_input:" in transforms_text
    assert 'raise ValueError("State is required.")' in transforms_text
    assert (
        "tokens, token_masks = self.tokenizer.tokenize(prompt, state)"
        in transforms_text
    )
    assert (
        'data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)'
        in transforms_text
    )

    assert (
        "This is the Pi05 format, where the state is part of the discrete language input."
        in tokenizer_text
    )
    assert (
        "discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1"
        in tokenizer_text
    )
    assert (
        'full_prompt = f"Task: {cleaned_text}, State: {state_str};\\nAction: "'
        in tokenizer_text
    )

    assert (
        "Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)"
        in training_config_text
    )
    assert "discrete_state_input: bool = None" in pi0_config_text
    assert (
        'object.__setattr__(self, "discrete_state_input", self.pi05)' in pi0_config_text
    )
