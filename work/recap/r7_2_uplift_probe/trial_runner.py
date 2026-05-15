from __future__ import annotations

from work.recap.r7_2_uplift_probe.contract import R7TrainingFailedError, TrialReport, TrialRequest


def run_trial(request: TrialRequest) -> TrialReport:
    raise R7TrainingFailedError("R7.2 trial runner is implemented in commit 2")
