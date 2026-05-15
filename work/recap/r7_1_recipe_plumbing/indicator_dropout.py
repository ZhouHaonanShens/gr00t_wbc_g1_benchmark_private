from __future__ import annotations

import random
from typing import Any


def apply_indicator_dropout(value: Any, *, p: float, rng: random.Random) -> Any | None:
    if not isinstance(rng, random.Random):
        raise TypeError(f"rng must be random.Random, got {type(rng).__name__}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"indicator_dropout_p out of range [0,1]: {p}")
    if p == 0.0:
        return value
    if p == 1.0:
        return None
    sample_value = rng.random()
    return None if sample_value < p else value


def make_rng(seed: int) -> random.Random:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    isolated_rng = random.Random(seed)
    if not isinstance(isolated_rng, random.Random):
        raise TypeError("random.Random did not return a local RNG")
    return isolated_rng
