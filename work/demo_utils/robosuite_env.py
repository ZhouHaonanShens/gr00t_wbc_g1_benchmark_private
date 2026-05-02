from __future__ import annotations

from typing import Any


def set_hard_reset_best_effort(env: object, hard_reset: bool) -> bool:
    cur: Any = env
    for _ in range(12):
        if cur is None:
            return False

        if hasattr(cur, "hard_reset"):
            try:
                setattr(cur, "hard_reset", bool(hard_reset))
                return True
            except Exception:
                return False

        nxt = None
        for attr in ("base_env", "env", "unwrapped"):
            try:
                cand = getattr(cur, attr)
            except Exception:
                cand = None
            if cand is not None and cand is not cur:
                nxt = cand
                break

        if nxt is None:
            return False
        cur = nxt

    return False
