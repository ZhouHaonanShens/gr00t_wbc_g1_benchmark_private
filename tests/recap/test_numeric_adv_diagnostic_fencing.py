from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import advantage
from work.recap import dataset
from work.recap.scripts import interface_localization_numeric_gap


def _load_script_module(script_name: str, module_name: str):
    module_path = REPO_ROOT / "work" / "recap" / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_advantage_contract_metadata_is_explicitly_diagnostic_only() -> None:
    contract = advantage.build_advantage_contract_metadata(
        source_iter_tag="iter_0001",
        n_samples=4,
        positive_scale=0.5,
        negative_scale_abs=0.75,
        critic_dir="/tmp/critic",
        critic_include_t=True,
        raw_summary={"min": -0.3, "max": 0.7},
        scaled_summary={"min": -0.4, "max": 0.9},
    )

    assert contract["surface_route"] == (
        advantage.CONTINUOUS_ADVANTAGE_CONTRACT_DIAGNOSTIC_ROUTE
    )
    assert contract["diagnostic_only"] is True
    assert contract["mainline_authority"] is False
    assert contract["authority_scope"] == (
        advantage.NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE
    )


def test_dataset_numeric_configs_are_explicitly_diagnostic_only() -> None:
    oversample = dataset.configure_positive_oversampling(factor=2)
    curriculum = dataset.configure_positive_curriculum(
        enabled=True,
        negative_retain_probability=0.25,
        seed=7,
    )
    late_stage = dataset.configure_late_stage_positive_emphasis(
        enabled=True,
        threshold=0.9,
    )

    for payload in (oversample, curriculum, late_stage):
        assert payload["diagnostic_only"] is True
        assert payload["mainline_authority"] is False
        assert payload["authority_scope"] == (
            advantage.NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE
        )
        assert "surface_route" in payload


def test_numeric_gap_payloads_are_fenced_but_keep_existing_path_modes(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "interface_localization_sprint"

    custom_payload = (
        interface_localization_numeric_gap.build_numeric_custom_path_payload(
            REPO_ROOT,
            output_dir=output_dir,
            output_json=output_dir
            / interface_localization_numeric_gap.NUMERIC_CUSTOM_PATH_JSON_NAME,
            availability_overrides={"python_module.gr00t": False},
        )
    )
    stock_payload = interface_localization_numeric_gap.build_numeric_stock_path_payload(
        REPO_ROOT,
        output_dir=output_dir,
        output_json=output_dir
        / interface_localization_numeric_gap.NUMERIC_STOCK_PATH_JSON_NAME,
        availability_overrides={
            "path.submodules/Isaac-GR00T": True,
            "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py": True,
        },
    )

    assert (
        custom_payload["path_mode"]
        == interface_localization_numeric_gap.CUSTOM_PATH_MODE
    )
    assert (
        stock_payload["path_mode"] == interface_localization_numeric_gap.STOCK_PATH_MODE
    )
    for payload in (custom_payload, stock_payload):
        assert payload["diagnostic_only"] is True
        assert payload["mainline_authority"] is False
        assert payload["authority_scope"] == (
            advantage.NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE
        )
        assert payload["surface_route"].startswith("interface_localization_numeric_")


def test_45b_requires_diagnostic_fence_for_single_case_summary() -> None:
    module = _load_script_module(
        "45b_vlm_critic_relabel_audit.py", "critic_relabel_audit_45b"
    )
    fenced_summary = {
        "advantage_input_range": {"min": -1.0, "max": 1.0},
        "advantage_contract_version": advantage.ADVANTAGE_CONTRACT_VERSION,
        "default_mainline": module.CONTINUOUS_ADVANTAGE_DIAGNOSTIC_ROUTE,
        "diagnostic_only": True,
        "mainline_authority": False,
        "authority_scope": advantage.NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
        "continuous_package": {},
        "threshold_packages": {},
    }
    unfenced_summary = {
        **fenced_summary,
        "default_mainline": "continuous_advantage",
        "diagnostic_only": False,
        "mainline_authority": True,
        "authority_scope": "mainline",
    }

    fenced_result = module._advantage_input_range_from_summary(fenced_summary)
    unfenced_result = module._advantage_input_range_from_summary(unfenced_summary)

    assert fenced_result["available"] is True
    assert fenced_result["ok"] is True
    assert unfenced_result["available"] is True
    assert unfenced_result["ok"] is False
