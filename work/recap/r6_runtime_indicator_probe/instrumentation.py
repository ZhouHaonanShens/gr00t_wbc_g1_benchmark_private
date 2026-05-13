from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any


def _sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _extract_text(messages: Any) -> str:
    if isinstance(messages, list) and messages:
        content = messages[0].get("content") if isinstance(messages[0], dict) else None
        return str(getattr(content, "text", ""))
    return ""


def attach_tokenizer_hook(policy_obj: Any) -> Callable[[], str]:
    captured = {"prompt_text_at_tokenizer": "", "prompt_tokens_sha256": _sha("")}
    processor = getattr(policy_obj, "processor", None)
    if callable(processor):
        def wrapped(messages: Any, *args: Any, **kwargs: Any) -> Any:
            text = _extract_text(messages)
            captured["prompt_text_at_tokenizer"] = text
            captured["prompt_tokens_sha256"] = _sha(text)
            return processor(messages, *args, **kwargs)
        setattr(policy_obj, "processor", wrapped)
    return lambda: json.dumps(captured, sort_keys=True)


def attach_action_head_input_hook(model_obj: Any) -> Callable[[], str]:
    captured = {"action_head_conditioning_sha256": _sha(""), "first_5_actions_l2": [0.0] * 5}
    action_head = getattr(model_obj, "action_head", model_obj)
    forward = getattr(action_head, "forward", None)
    if callable(forward):
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            captured["action_head_conditioning_sha256"] = _sha({"args": args, "kwargs": kwargs})
            return forward(*args, **kwargs)
        setattr(action_head, "forward", wrapped)
    return lambda: json.dumps(captured, sort_keys=True)
