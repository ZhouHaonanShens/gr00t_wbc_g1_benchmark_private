from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_artifact_smoke_wrapper_module():
    module_path = (
        REPO_ROOT / "work" / "recap" / "scripts" / "43b_vlm_critic_artifact_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "recap_vlm_critic_artifact_smoke_43b", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inference_public_surface_re_exports_runtime_workflows() -> None:
    inference_surface = importlib.import_module("work.recap.critic_vlm.inference")
    inference_runtime = importlib.import_module(
        "work.recap.critic_vlm.inference_runtime"
    )

    assert (
        inference_surface.run_critic_inference is inference_runtime.run_critic_inference
    )
    assert inference_surface.run_artifact_smoke is inference_runtime.run_artifact_smoke
    assert (
        inference_surface.CriticInferenceWorkflow
        is inference_runtime.CriticInferenceWorkflow
    )
    assert (
        inference_surface.ArtifactSmokeWorkflow
        is inference_runtime.ArtifactSmokeWorkflow
    )


def test_artifact_smoke_wrapper_help_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_artifact_smoke_wrapper_module()
    monkeypatch.setattr(
        module.sys, "argv", ["43b_vlm_critic_artifact_smoke.py", "--help"]
    )

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert excinfo.value.code == 0
