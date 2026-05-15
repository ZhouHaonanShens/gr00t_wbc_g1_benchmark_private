from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from work.recap.r7_2_uplift_probe.contract import R7UpliftError, StepwiseCounterfactual


def probe_adapter(
    adapter_dir: Path,
    base_ckpt: Path,
    *,
    leader_approval_token: str,
    gpu_id: int,
    output_dir: Path,
    seed: int,
) -> StepwiseCounterfactual:
    _ = base_ckpt, seed
    command = build_probe_command(adapter_dir, leader_approval_token, gpu_id, output_dir)
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3600)
    if completed.returncode != 0:
        raise R7UpliftError(f"R6.1 forced probe failed rc={completed.returncode}")
    counterfactual_path = output_dir / "A.2" / "counterfactual.json"
    item = parse_counterfactual_json(counterfactual_path, adapter_dir=adapter_dir)
    target = output_dir.parent / f"counterfactual_step_{item.step:04d}.json"
    shutil.copyfile(counterfactual_path, target)
    return StepwiseCounterfactual(item.step, item.adapter_path, str(target), item.counterfactual_verdict, item.condition_sha_equal, item.first_5_actions_l2_diff_max)


def build_probe_command(adapter_dir: Path, token: str, gpu_id: int, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "work.recap.r6_runtime_indicator_probe",
        "probe",
        "--forced",
        "--cell",
        "A.2",
        "--leader-approval-token",
        str(token),
        "--gpu",
        str(int(gpu_id)),
        "--lora-adapter-dir",
        str(adapter_dir),
        "--output-root",
        str(output_dir),
    ]


def parse_counterfactual_json(path: Path, *, adapter_dir: Path) -> StepwiseCounterfactual:
    if not path.is_file():
        raise R7UpliftError(f"missing counterfactual json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("counterfactual_verdict", "condition_sha_equal", "first_5_actions_l2_diff")
    missing = [name for name in required if name not in payload]
    if missing:
        raise R7UpliftError(f"counterfactual json missing fields: {missing}")
    diff_values = [float(value) for value in payload["first_5_actions_l2_diff"]]
    step = _step_from_adapter(adapter_dir)
    return StepwiseCounterfactual(
        step,
        str(adapter_dir),
        str(path),
        payload["counterfactual_verdict"],
        bool(payload["condition_sha_equal"]),
        max(abs(value) for value in diff_values),
    )


def _step_from_adapter(adapter_dir: Path) -> int:
    name = adapter_dir.name
    digits = "".join(character for character in name if character.isdigit())
    if not digits:
        raise R7UpliftError(f"adapter dir does not encode step: {adapter_dir}")
    return int(digits)
