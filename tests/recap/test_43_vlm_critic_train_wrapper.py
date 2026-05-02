from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_train_wrapper_module():
    module_path = REPO_ROOT / "work" / "recap" / "scripts" / "43_vlm_critic_train.py"
    spec = importlib.util.spec_from_file_location(
        "recap_vlm_critic_train_43", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_public_surface_re_exports_core_training_contract() -> None:
    train_surface = importlib.import_module("work.recap.critic_vlm.train")
    training_core = importlib.import_module("work.recap.critic_vlm.training")

    assert train_surface.TrainConfig is training_core.TrainConfig
    assert train_surface.TrainResult is training_core.TrainResult
    assert train_surface.WarmstartPlan is training_core.WarmstartPlan
    assert train_surface.PublicWarmstartSample is training_core.PublicWarmstartSample
    assert (
        train_surface.run_vlm_critic_training is training_core.run_vlm_critic_training
    )


def test_train_wrapper_help_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_train_wrapper_module()
    monkeypatch.setattr(module.sys, "argv", ["43_vlm_critic_train.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        module.main()

    assert excinfo.value.code == 0
