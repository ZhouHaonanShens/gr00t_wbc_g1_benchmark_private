from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.dual_loss import (  # noqa: E402
    DUAL_LOSS_FORMULA,
    DualLossConfig,
    build_dual_loss_integration_report,
    combine_alpha_dual_loss,
)


def test_alpha_dual_loss_combines_unconditioned_and_conditioned_terms() -> None:
    payload = combine_alpha_dual_loss(
        unconditioned={
            "flow_loss": 0.5,
            "discrete_action_ce": 0.2,
            "text_ce": 0.1,
            "total_loss": 0.8,
        },
        conditioned={
            "flow_loss": 0.3,
            "discrete_action_ce": 0.1,
            "text_ce": 0.1,
            "total_loss": 0.5,
        },
        config=DualLossConfig(alpha=0.25, dropout_p=0.3),
    )

    components = cast(dict[str, object], payload["components"])
    assert payload["formula"] == DUAL_LOSS_FORMULA
    assert payload["alpha"] == 0.25
    assert components["flow_loss"] == 0.575
    assert components["discrete_action_ce"] == 0.225
    assert components["text_ce"] == 0.125
    assert payload["total_loss"] == 0.925


def test_alpha_changes_dual_total_loss() -> None:
    loss_inputs = {
        "unconditioned": {
            "flow_loss": 0.5,
            "discrete_action_ce": 0.2,
            "text_ce": 0.1,
            "total_loss": 0.8,
        },
        "conditioned": {
            "flow_loss": 0.3,
            "discrete_action_ce": 0.1,
            "text_ce": 0.1,
            "total_loss": 0.5,
        },
    }

    alpha_zero = combine_alpha_dual_loss(
        **loss_inputs,
        config=DualLossConfig(alpha=0.0),
    )
    alpha_one = combine_alpha_dual_loss(
        **loss_inputs,
        config=DualLossConfig(alpha=1.0),
    )

    assert alpha_zero["total_loss"] == 0.8
    assert alpha_one["total_loss"] == 1.3


def test_alpha_dual_loss_preserves_tensor_autograd_path() -> None:
    import torch

    unconditioned = torch.tensor(0.8, requires_grad=True)
    conditioned = torch.tensor(0.5, requires_grad=True)
    zero = unconditioned.new_zeros(())
    payload = combine_alpha_dual_loss(
        unconditioned={
            "flow_loss": unconditioned,
            "discrete_action_ce": zero,
            "text_ce": zero,
            "total_loss": unconditioned,
        },
        conditioned={
            "flow_loss": conditioned,
            "discrete_action_ce": zero,
            "text_ce": zero,
            "total_loss": conditioned,
        },
        config=DualLossConfig(alpha=0.25, dropout_p=0.3),
    )

    total = payload["total_loss"]
    assert isinstance(total, torch.Tensor)
    total.backward()
    assert unconditioned.grad is not None
    assert conditioned.grad is not None
    assert torch.isclose(unconditioned.grad, torch.tensor(1.0))
    assert torch.isclose(conditioned.grad, torch.tensor(0.25))


def test_openpi_overlay_smoke_report_includes_alpha_dual_loss() -> None:
    overlay_src = REPO_ROOT / "work/openpi/overlays/openpi_recap/src"
    for module_name in list(sys.modules):
        if module_name == "openpi" or module_name.startswith("openpi."):
            del sys.modules[module_name]
    try:
        openpi_module = types.ModuleType("openpi")
        openpi_module.__path__ = [  # type: ignore[attr-defined]
            str(overlay_src / "openpi")
        ]
        sys.modules["openpi"] = openpi_module
        training = importlib.import_module("openpi.recap_overlay.training")
        payload = training.build_smoke_forward_report(
            config_name="pi05_libero_recap",
            checkpoint_source="repo-local-test",
        )
    finally:
        for module_name in list(sys.modules):
            if module_name == "openpi" or module_name.startswith("openpi."):
                del sys.modules[module_name]

    dual = cast(dict[str, object], payload["dual"])
    config = cast(dict[str, object], payload["config"])

    assert dual["formula"] == DUAL_LOSS_FORMULA
    assert dual["alpha"] == config["dual_loss_alpha"]
    assert dual["dropout_p"] == config["indicator_dropout_p"]
    conditioned = cast(dict[str, object], payload["conditioned"])
    assert dual["total_loss"] > conditioned["total_loss"]


def test_gr00t_single_path_loss_is_reported_as_integration_gap() -> None:
    report = build_dual_loss_integration_report(
        training_surface="work/recap/model.py:GR00TRecapActionHead.forward",
        single_path_loss_name="action_loss",
        dual_view_integrated=False,
        replacement_strategy="single_path_numeric_advantage_conditioning",
    )

    assert report["status"] == "PARTIAL"
    assert report["dual_view_integrated"] is False
    assert report["single_path_loss_name"] == "action_loss"
