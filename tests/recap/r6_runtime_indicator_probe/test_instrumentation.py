from __future__ import annotations

import json

from work.recap.r6_runtime_indicator_probe.instrumentation import attach_action_head_input_hook, attach_tokenizer_hook


class _Content:
    text = "task\nAdvantage: positive"


class _Policy:
    def processor(self, messages):  # type: ignore[no-untyped-def]
        return {"messages": messages}


class _Head:
    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return {"args": args, "kwargs": kwargs}


class _Model:
    action_head = _Head()


def test_tokenizer_hook_captures_prompt_text_without_writes() -> None:
    policy = _Policy()
    read = attach_tokenizer_hook(policy)
    result = policy.processor([{"content": _Content()}])
    assert "messages" in result
    captured = json.loads(read())
    assert captured["prompt_text_at_tokenizer"] == "task\nAdvantage: positive"
    assert len(captured["prompt_tokens_sha256"]) == 64


def test_action_head_hook_returns_hash_reader() -> None:
    model = _Model()
    read = attach_action_head_input_hook(model)
    assert model.action_head.forward("x") == {"args": ("x",), "kwargs": {}}
    captured = json.loads(read())
    assert len(captured["action_head_conditioning_sha256"]) == 64
    assert captured["first_5_actions_l2"] == [0.0] * 5
