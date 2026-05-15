from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from work.recap.r7_2_uplift_probe.contract import (
    R7AdapterTooLargeError,
    R7BudgetExceeded,
    R7TrainingFailedError,
    R7UpliftError,
    StepwiseCounterfactual,
    TrialReport,
    TrialRequest,
)
from work.recap.r7_2_uplift_probe.stepwise_probe import probe_adapter

R7_0_RECIPE_PATH = Path("tests/recap/r7_recipe_diff/fixtures/first_run_A2/training_recipe_diff.json")
R7_0_RECIPE_SHA256 = "a9948f64750ea28084bca17056270deb1821512f278068c007b1d945c2a79fc6"
ADAPTER_SIZE_LIMIT_MB = 200.0


def run_trial(request: TrialRequest) -> TrialReport:
    _validate_request(request)
    output_root = Path(request.output_root).resolve()
    output_root.mkdir(parents=True)
    request_json = output_root / "trial_request.json"
    request_json.write_text(json.dumps(asdict(request), indent=2, sort_keys=True) + "\n")
    stderr_path = output_root / "training_stderr.log"
    started = time.monotonic()
    evolution: list[StepwiseCounterfactual] = []
    final_loss: float | None = None
    final_verdict = "TRAINING_FAILED"
    final_step = 0
    proc = _spawn_training_subprocess(request, request_json, output_root, stderr_path)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            event = _parse_event(line)
            elapsed = _gpu_seconds(started)
            _enforce_budget(elapsed, request)
            if event.get("event") == "step":
                final_loss = float(event.get("loss", 0.0))
            if event.get("event") == "checkpoint":
                item = _probe_checkpoint(event, request, output_root)
                evolution.append(item)
                final_step = int(item.step)
                if item.counterfactual_verdict == "INDICATOR_SENSITIVE" and item.step >= 200:
                    final_verdict = "INDICATOR_SENSITIVE_AT_STEP_N"
                    _signal_training_to_stop(proc)
                    break
            if event.get("event") == "done":
                reason = str(event.get("reason", "crash"))
                if reason == "max_steps_reached":
                    final_verdict = "INDICATOR_INVARIANT_AT_MAX_STEPS"
                    final_step = int(request.max_steps)
                else:
                    final_verdict = "TRAINING_FAILED"
                break
        proc.wait(timeout=30)
    except R7BudgetExceeded:
        final_verdict = "BUDGET_EXCEEDED"
        _signal_training_to_stop(proc)
    except Exception:
        _signal_training_to_stop(proc)
        raise
    report = TrialReport(request, final_verdict, final_step, tuple(evolution), _gpu_seconds(started), final_loss)
    _write_report(output_root / "trial_report.json", report)
    return report


def _validate_request(request: TrialRequest) -> None:
    base = Path(request.base_ckpt_abs_path)
    if not base.is_absolute() or not base.is_dir():
        raise FileNotFoundError(f"base ckpt must be existing absolute directory: {base}")
    if Path(request.output_root).exists():
        raise R7UpliftError(f"output_root must not exist: {request.output_root}")
    if request.trial_id == "trial-1" and int(request.gpu_id) != 1:
        raise R7UpliftError("trial-1 is locked to GPU 1")
    _verify_r7_0_recipe()
    probe_source = Path("work/recap/r6_runtime_indicator_probe/runtime_probe.py").read_text(encoding="utf-8")
    if "def run_runtime_probe" not in probe_source:
        raise R7UpliftError("R6.1 run_runtime_probe is missing")


def _spawn_training_subprocess(request: TrialRequest, request_json: Path, output_root: Path, stderr_path: Path) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(int(request.gpu_id))
    command = [sys.executable, "-m", "work.recap.r7_2_uplift_probe.lora_train_worker", "--request-json", str(request_json), "--output-root", str(output_root)]
    stderr_file = stderr_path.open("w", encoding="utf-8")
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file, text=True, env=env)


def _probe_checkpoint(event: dict[str, object], request: TrialRequest, output_root: Path) -> StepwiseCounterfactual:
    adapter_dir = Path(str(event.get("path", ""))).resolve()
    _validate_adapter(adapter_dir)
    step = int(event.get("step", 0))
    probe_output = output_root / f"probe_step_{step:04d}"
    return probe_adapter(adapter_dir, Path(request.base_ckpt_abs_path), leader_approval_token=request.leader_approval_token, gpu_id=request.gpu_id, output_dir=probe_output, seed=request.seed)


def _validate_adapter(adapter_dir: Path) -> None:
    if not adapter_dir.is_dir():
        raise R7TrainingFailedError(f"missing adapter dir: {adapter_dir}")
    size_mb = sum(child.stat().st_size for child in adapter_dir.rglob("*") if child.is_file()) / (1024.0 * 1024.0)
    if size_mb > ADAPTER_SIZE_LIMIT_MB:
        raise R7AdapterTooLargeError(f"adapter exceeds 200 MB: {size_mb:.3f}")


def _signal_training_to_stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _enforce_budget(total_gpu_seconds: float, request: TrialRequest) -> None:
    budget_seconds = float(request.budget_gpu_minutes) * 60.0
    if total_gpu_seconds > budget_seconds:
        raise R7BudgetExceeded("trial exceeded GPU budget")
    return None


def _gpu_seconds(started: float) -> float:
    elapsed = time.monotonic() - started
    if elapsed < 0.0:
        return 0.0
    return elapsed


def _parse_event(line: str) -> dict[str, object]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise R7TrainingFailedError(f"training child emitted non-json line: {line[:80]}") from exc
    if not isinstance(payload, dict):
        raise R7TrainingFailedError("training event must be a JSON object")
    return payload


def _verify_r7_0_recipe() -> None:
    import hashlib

    digest = hashlib.sha256(R7_0_RECIPE_PATH.read_bytes()).hexdigest()
    if digest != R7_0_RECIPE_SHA256:
        raise R7UpliftError("R7.0 recipe sha256 mismatch")


def _write_report(path: Path, report: TrialReport) -> None:
    serialized = json.dumps(asdict(report), indent=2, sort_keys=True)
    payload = serialized + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
