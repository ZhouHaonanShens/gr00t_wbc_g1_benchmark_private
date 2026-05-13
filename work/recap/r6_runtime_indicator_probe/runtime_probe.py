from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from work.recap.r6_runtime_indicator_probe.contract import R6BudgetExceeded, R6Error, RuntimeTrace

_TOKEN_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class ProbeBudget:
    max_minutes_per_cell: int = 30
    max_minutes_total: int = 120
    max_episodes_per_cell: int = 1
    max_steps_per_episode: int = 200
    gpu_id: int = 1


def _validate_budget(cell_id: str, budget: ProbeBudget, token: str) -> None:
    if str(cell_id).strip().upper() not in {"A.2", "A.3", "A.4", "A.5"}:
        raise R6Error(f"unsupported R6.1 cell: {cell_id!r}")
    if not _TOKEN_RE.fullmatch(str(token)):
        raise R6Error("--leader-approval-token must be a 64-character SHA-256 hex string")
    if int(budget.gpu_id) not in {1, 2}:
        raise R6Error("R6.1 permits only GPU 1 or GPU 2; GPU 0/3 are rejected")
    if budget.max_episodes_per_cell > 1 or budget.max_steps_per_episode > 200:
        raise R6BudgetExceeded("R6.1 exceeds one episode or 200 steps per cell")
    if budget.max_minutes_per_cell > 30 or budget.max_minutes_total > 120:
        raise R6BudgetExceeded("R6.1 exceeds 30 minutes per cell or 120 minutes total")


def _hash_tensor(t: Any) -> str:
    if hasattr(t, "detach"):
        t = t.detach().cpu().tolist()
    payload = json.dumps(t, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _spawn_one_episode(cell_id: str, budget: ProbeBudget) -> dict[str, Any]:
    command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={int(budget.gpu_id)}",
        "python3",
        "-m",
        "work.recap.r6_runtime_indicator_probe.runtime_probe_worker",
        "--cell",
        str(cell_id).strip().upper(),
        "--max-steps",
        str(int(budget.max_steps_per_episode)),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=int(budget.max_minutes_per_cell) * 60,
    )
    return json.loads(completed.stdout or "{}")


def run_runtime_probe(cell_id: str, budget: ProbeBudget, leader_approval_token: str) -> RuntimeTrace:
    _validate_budget(cell_id, budget, leader_approval_token)
    payload = _spawn_one_episode(str(cell_id).strip().upper(), budget)
    actions = tuple(float(x) for x in payload.get("first_5_actions_l2", (0, 0, 0, 0, 0)))
    if len(actions) != 5:
        raise R6Error("runtime payload must contain exactly five first-action L2 values")
    present = bool(payload.get("indicator_substring_present", False))
    return RuntimeTrace(
        cell_id=str(cell_id).strip().upper(),
        episode_seed=int(payload.get("episode_seed", 0)),
        prompt_text_at_tokenizer=str(payload.get("prompt_text_at_tokenizer", "")),
        prompt_tokens_sha256=str(payload.get("prompt_tokens_sha256") or _hash_tensor(payload.get("prompt_tokens", ""))),
        action_head_conditioning_sha256=str(payload.get("action_head_conditioning_sha256") or _hash_tensor(payload.get("action_head_conditioning", ""))),
        first_5_actions_l2=actions,  # type: ignore[arg-type]
        indicator_substring_present=present,
        runtime_verdict="INDICATOR_PRESENT" if present else "INDICATOR_ABSENT",
    )
