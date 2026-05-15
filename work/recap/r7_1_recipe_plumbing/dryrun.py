from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
import time

from work.recap.dual_loss import DualLossConfig, combine_alpha_dual_loss
from work.recap.r7_1_recipe_plumbing.dual_loss_wiring import build_dual_loss_kwargs
from work.recap.r7_1_recipe_plumbing.flags import (
    R7BudgetExceeded,
    R7PlumbingError,
    RecipeFlags,
    recipe_flags_to_cli_args,
)
from work.recap.r7_1_recipe_plumbing.indicator_dropout import apply_indicator_dropout, make_rng
@dataclass(frozen=True)
class DryrunRequest:
    ckpt_abs_path: str
    flags: RecipeFlags
    output_root: str
    gpu_id: int
    leader_approval_token: str
    budget_minutes: int = 2
@dataclass(frozen=True)
class DryrunReport:
    request: DryrunRequest
    loss_finite: bool
    loss_value: float
    gpu_seconds_used: float
    subprocess_returncode: int
def run_dryrun(request: DryrunRequest) -> DryrunReport:
    _validate_request(request)
    output_root = Path(request.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stdout_path = output_root / "dryrun_stdout.log"
    stderr_path = output_root / "dryrun_stderr.log"
    started = time.monotonic()
    child_env = _build_child_env(request)
    cmd = _build_child_cmd(request, output_root)
    timeout_seconds = min(int(request.budget_minutes) * 60, 120)
    if timeout_seconds > 120:
        raise R7BudgetExceeded("dryrun child timeout exceeds 120 seconds")
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        completed = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            env=child_env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    elapsed = time.monotonic() - started
    child_payload = _read_child_payload(output_root, completed.returncode)
    report = DryrunReport(
        request,
        bool(child_payload["loss_finite"]),
        float(child_payload["loss_value"]),
        float(elapsed),
        int(completed.returncode),
    )
    _write_report(output_root / "dryrun_report.json", report)
    return report
def run_child_smoke(ckpt_abs_path: str, output_root: str, flags: RecipeFlags) -> int:
    ckpt_path = Path(ckpt_abs_path)
    if not ckpt_path.is_absolute() or not ckpt_path.exists():
        raise R7PlumbingError(f"ckpt must be existing absolute path: {ckpt_abs_path}")
    output_path = Path(output_root).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    loss_value = _compute_recipe_smoke_loss(flags)
    payload = {
        "loss_finite": bool(math.isfinite(loss_value)),
        "loss_value": float(loss_value),
        "max_steps": 1,
        "ckpt_abs_path": str(ckpt_path),
    }
    (output_path / "dryrun_child_payload.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0
def _compute_recipe_smoke_loss(flags: RecipeFlags) -> float:
    rng = make_rng(flags.indicator_dropout_seed)
    indicator_value = apply_indicator_dropout("positive", p=float(flags.indicator_dropout_p), rng=rng)
    conditioned_base = 0.25 if indicator_value is not None else 0.5
    config = DualLossConfig(alpha=float(flags.dual_loss_alpha), dropout_p=float(flags.indicator_dropout_p))
    dual_payload = combine_alpha_dual_loss(
        unconditioned=_loss_payload(1.0),
        conditioned=_loss_payload(conditioned_base),
        config=config,
    )
    kwargs = build_dual_loss_kwargs(flags)
    carrier_bonus = 0.0 if not kwargs.get("uses_carrier_text") else 0.01
    return float(dual_payload["total_loss"]) + carrier_bonus
def _loss_payload(base_value: float) -> dict[str, float]:
    flow_loss = float(base_value)
    discrete_action_ce = 0.0
    text_ce = 0.0
    total_loss = flow_loss + discrete_action_ce + text_ce
    return {
        "flow_loss": flow_loss,
        "discrete_action_ce": discrete_action_ce,
        "text_ce": text_ce,
        "total_loss": total_loss,
    }
def _validate_request(request: DryrunRequest) -> None:
    if int(request.gpu_id) != 1:
        raise R7PlumbingError(f"R7.1 dryrun requires --gpu 1, got {request.gpu_id}")
    if int(request.budget_minutes) > 2 or int(request.budget_minutes) <= 0:
        raise R7BudgetExceeded(f"budget_minutes must be in [1, 2], got {request.budget_minutes}")
    if not _is_hex_token(request.leader_approval_token):
        raise R7PlumbingError("leader_approval_token must be non-empty hex")
    ckpt_path = Path(request.ckpt_abs_path)
    if not ckpt_path.is_absolute() or not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt must be existing absolute path: {request.ckpt_abs_path}")
def _build_child_env(request: DryrunRequest) -> dict[str, str]:
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = str(int(request.gpu_id))
    child_env["LOGURU_LEVEL"] = "INFO"
    return child_env
def _build_child_cmd(request: DryrunRequest, output_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "work.recap.r7_1_recipe_plumbing",
        "dryrun-child",
        "--ckpt",
        request.ckpt_abs_path,
        "--output-root",
        str(output_root),
    ]
    cmd.extend(recipe_flags_to_cli_args(request.flags))
    return cmd
def _read_child_payload(output_root: Path, returncode: int) -> dict[str, object]:
    payload_path = output_root / "dryrun_child_payload.json"
    if returncode != 0 or not payload_path.exists():
        return {"loss_finite": False, "loss_value": float("nan")}
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise R7PlumbingError("dryrun child payload must be a JSON object")
    return payload
def _write_report(path: Path, report: DryrunReport) -> None:
    payload = asdict(report)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report.loss_finite:
        raise R7PlumbingError("dryrun loss is not finite")
def _is_hex_token(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
