from __future__ import annotations

from pathlib import Path

from work.recap.r7_2_uplift_probe.contract import R7TrainingFailedError, StepwiseCounterfactual


def probe_adapter(adapter_dir: Path, base_ckpt: Path, *, leader_approval_token: str, gpu_id: int, output_dir: Path, seed: int) -> StepwiseCounterfactual:
    raise R7TrainingFailedError("R7.2 stepwise probe is implemented in commit 2")
