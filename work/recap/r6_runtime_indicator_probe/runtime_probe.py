from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from work.recap.r6_runtime_indicator_probe.contract import (
    ProbeCounterfactual,
    R6BudgetExceeded,
    R6Error,
    RuntimeTrace,
)

_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_CELL_IDS = {"A.2", "A.3", "A.4", "A.5"}
_DEFAULT_SEED = 20000
_COUNTERFACTUAL_THRESHOLD = 1e-3
_LAST_NEGATIVE_TRACE: RuntimeTrace | None = None


def get_last_negative_trace() -> RuntimeTrace | None:
    return _LAST_NEGATIVE_TRACE


@dataclass(frozen=True)
class ProbeBudget:
    max_minutes_per_cell: int = 30
    max_minutes_total: int = 60
    max_episodes_per_cell: int = 1
    max_steps_per_episode: int = 200
    gpu_id: int = 1


def _normalize_cell(cell_id: str) -> str:
    return str(cell_id).strip().upper()


def _validate_budget(
    cell_id: str,
    budget: ProbeBudget,
    token: str,
    *,
    forced: bool = False,
    counterfactual: bool = True,
) -> None:
    cell = _normalize_cell(cell_id)
    if forced and cell != "A.2":
        raise R6Error("R6.1 --forced accepts only cell A.2")
    if cell not in _CELL_IDS:
        raise R6Error(f"unsupported R6.1 cell: {cell_id!r}")
    if not _TOKEN_RE.fullmatch(str(token)):
        raise R6Error("--leader-approval-token must be a 64-character SHA-256 hex string")
    if forced and int(budget.gpu_id) != 1:
        raise R6Error("R6.1 --forced is locked to GPU 1")
    if not forced and int(budget.gpu_id) not in {1, 2}:
        raise R6Error("R6.1 permits only GPU 1 or GPU 2; GPU 0/3 are rejected")
    if budget.max_episodes_per_cell > 1 or budget.max_steps_per_episode > 200:
        raise R6BudgetExceeded("R6.1 exceeds one episode or 200 steps per cell")
    total_limit = 60 if forced else 120
    if budget.max_minutes_per_cell > 30 or budget.max_minutes_total > total_limit:
        raise R6BudgetExceeded(f"R6.1 exceeds 30 minutes per cell or {total_limit} minutes total")
    required_minutes = int(budget.max_minutes_per_cell) * (2 if counterfactual else 1)
    if required_minutes > int(budget.max_minutes_total):
        raise R6BudgetExceeded("R6.1 counterfactual probe exceeds total GPU-minute budget")


def _hash_tensor(t: Any) -> str:
    if hasattr(t, "detach"):
        t = t.detach().cpu().tolist()
    payload = json.dumps(t, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _spawn_one_episode(cell_id: str, budget: ProbeBudget, *, seed: int, indicator_mode: str) -> dict[str, Any]:
    command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={int(budget.gpu_id)}",
        sys.executable,
        "-m",
        "work.recap.r6_runtime_indicator_probe.runtime_probe_worker",
        "--cell",
        _normalize_cell(cell_id),
        "--max-steps",
        str(int(budget.max_steps_per_episode)),
        "--seed",
        str(int(seed)),
        "--force-indicator-mode",
        str(indicator_mode),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=int(budget.max_minutes_per_cell) * 60,
        )
    except subprocess.CalledProcessError as exc:
        stdout_tail = "\n".join((exc.stdout or "").splitlines()[-40:])
        stderr_tail = "\n".join((exc.stderr or "").splitlines()[-40:])
        raise R6Error(f"runtime child failed rc={exc.returncode}; stdout_tail={stdout_tail}; stderr_tail={stderr_tail}") from exc
    for line in reversed((completed.stdout or "").splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    return json.loads(completed.stdout or "{}")


def _trace_from_payload(cell_id: str, payload: dict[str, Any]) -> RuntimeTrace:
    actions = tuple(float(x) for x in payload.get("first_5_actions_l2", (0, 0, 0, 0, 0)))
    if len(actions) != 5:
        raise R6Error("runtime payload must contain exactly five first-action L2 values")
    present = bool(payload.get("indicator_substring_present", False))
    return RuntimeTrace(
        cell_id=_normalize_cell(cell_id),
        episode_seed=int(payload.get("episode_seed", _DEFAULT_SEED)),
        prompt_text_at_tokenizer=str(payload.get("prompt_text_at_tokenizer", "")),
        prompt_tokens_sha256=str(payload.get("prompt_tokens_sha256") or _hash_tensor(payload.get("prompt_tokens", ""))),
        action_head_conditioning_sha256=str(payload.get("action_head_conditioning_sha256") or _hash_tensor(payload.get("action_head_conditioning", ""))),
        first_5_actions_l2=actions,  # type: ignore[arg-type]
        indicator_substring_present=present,
        runtime_verdict="INDICATOR_PRESENT" if present else "INDICATOR_ABSENT",
    )


def _build_counterfactual(cell_id: str, positive: RuntimeTrace, negative: RuntimeTrace) -> ProbeCounterfactual:
    diff = tuple(abs(a - b) for a, b in zip(positive.first_5_actions_l2, negative.first_5_actions_l2))
    if len(diff) != 5 or positive.episode_seed != negative.episode_seed:
        raise R6Error("counterfactual traces must share seed and expose five L2 values")
    verdict = "INDICATOR_SENSITIVE" if any(value > _COUNTERFACTUAL_THRESHOLD for value in diff) else "INDICATOR_INVARIANT"
    return ProbeCounterfactual(
        cell_id=_normalize_cell(cell_id),
        seed=int(positive.episode_seed),
        positive_trace_sha256=positive.action_head_conditioning_sha256,
        negative_trace_sha256=negative.action_head_conditioning_sha256,
        condition_sha_equal=positive.action_head_conditioning_sha256 == negative.action_head_conditioning_sha256,
        first_5_actions_l2_diff=diff,  # type: ignore[arg-type]
        counterfactual_verdict=verdict,
    )


def run_runtime_probe(
    cell_id: str,
    budget: ProbeBudget,
    leader_approval_token: str,
    *,
    forced: bool = False,
    counterfactual: bool = True,
) -> tuple[RuntimeTrace, ProbeCounterfactual | None]:
    _validate_budget(cell_id, budget, leader_approval_token, forced=forced, counterfactual=counterfactual)
    global _LAST_NEGATIVE_TRACE
    _LAST_NEGATIVE_TRACE = None
    cell = _normalize_cell(cell_id)
    positive = _trace_from_payload(cell, _spawn_one_episode(cell, budget, seed=_DEFAULT_SEED, indicator_mode="positive"))
    if not counterfactual:
        return positive, None
    negative = _trace_from_payload(cell, _spawn_one_episode(cell, budget, seed=_DEFAULT_SEED, indicator_mode="negative"))
    _LAST_NEGATIVE_TRACE = negative
    return positive, _build_counterfactual(cell, positive, negative)
