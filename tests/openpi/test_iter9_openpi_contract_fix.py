from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.pipelines.recap import blind_calibration_inference as inference  # noqa: E402
from work.openpi.recap.real_variant_policy_contract import (  # noqa: E402
    POLICY_CONFIG_MISMATCH_BLOCKER,
    REAL_VARIANT_DATA_FACTORY_KIND,
    REAL_VARIANT_POLICY_CONFIG_NAME,
    attach_real_variant_policy_contract,
    build_real_variant_policy_contract,
)  # noqa: E402
from work.openpi.recap import real_variant_export  # noqa: E402


def _write_norm_stats(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {
                        "mean": [1.0, 2.0, 3.0],
                        "std": [1.0, 1.0, 1.0],
                        "q01": [0.0, 0.0, 0.0],
                        "q99": [2.0, 2.0, 2.0],
                    },
                    "actions": {
                        "mean": [0.1, 0.2],
                        "std": [1.0, 1.0],
                        "q01": [0.0, 0.0],
                        "q99": [1.0, 1.0],
                    },
                }
            }
        ),
        encoding="utf-8",
)


def test_export_policy_contract_records_real_variant_action_semantics(
    tmp_path: Path,
) -> None:
    norm_stats_path = tmp_path / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    _write_norm_stats(norm_stats_path)

    contract = build_real_variant_policy_contract(
        base_train_config_name="pi0_libero",
        exp_name="control_no_recap_shuffled_adversarial_relabel",
        consumer_mode="fixedadv_constant",
        fixed_indicator_mode="omit",
        norm_stats_json_path=norm_stats_path,
    )
    manifest = attach_real_variant_policy_contract(
        {"schema_version": "openpi_real_variant_export_v2", "train_config_name": "pi0_libero"},
        policy_contract=contract,
    )

    assert manifest["policy_config_name"] == REAL_VARIANT_POLICY_CONFIG_NAME
    assert manifest["data_factory_kind"] == REAL_VARIANT_DATA_FACTORY_KIND
    assert manifest["extra_delta_transform"] is False
    assert manifest["data_transforms_outputs"] == ["LiberoOutputs"]
    assert manifest["norm_stats_state_dim"] == 3
    assert manifest["norm_stats_action_dim"] == 2
    assert manifest["policy_contract"]["action_semantics"]["delta_postprocess_required"] is False
    assert "AbsoluteActions" in manifest["policy_contract"]["action_semantics"][
        "must_not_apply_output_transforms"
    ]


def test_loader_routes_legacy_real_variant_manifest_to_custom_config(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    (checkpoint_dir / "params").mkdir(parents=True)
    (checkpoint_dir / "params" / "_METADATA").write_text("fixture\n", encoding="utf-8")
    (checkpoint_dir / "export_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "openpi_real_variant_export_v1",
                "train_config_name": "pi0_libero",
            }
        ),
        encoding="utf-8",
    )

    spec = inference._resolve_local_checkpoint_policy_spec(checkpoint_dir, checkpoint_dir)

    assert spec.config_name == REAL_VARIANT_POLICY_CONFIG_NAME
    assert spec.policy_contract is not None
    assert spec.policy_contract["legacy_manifest_backfill"] is True
    assert spec.policy_contract["training_transform_graph"]["outputs"] == ["LiberoOutputs"]


def test_loader_blocks_declared_transform_mismatch(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "export_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "openpi_real_variant_export_v2",
                "policy_contract": {
                    "data_factory_kind": REAL_VARIANT_DATA_FACTORY_KIND,
                    "extra_delta_transform": False,
                    "training_transform_graph": {
                        "outputs": ["AbsoluteActions", "LiberoOutputs"]
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=POLICY_CONFIG_MISMATCH_BLOCKER):
        inference._resolve_local_checkpoint_policy_spec(checkpoint_dir, checkpoint_dir)


def test_variant_a_loader_accepts_direct_stock_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "pi0_libero"
    (checkpoint_dir / "params").mkdir(parents=True)
    (checkpoint_dir / "params" / "_METADATA").write_text("fixture\n", encoding="utf-8")

    resolved = inference._resolve_variant_a_checkpoint_dir(checkpoint_dir)

    assert resolved == checkpoint_dir


def test_variant_a_loader_accepts_authority_manifest_file(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "pi0_libero"
    (checkpoint_dir / "params").mkdir(parents=True)
    (checkpoint_dir / "params" / "_METADATA").write_text("fixture\n", encoding="utf-8")
    authority_manifest = tmp_path / "a_authority.json"
    authority_manifest.write_text(
        json.dumps({"local_resolved_path": str(checkpoint_dir)}),
        encoding="utf-8",
    )

    resolved = inference._resolve_variant_a_checkpoint_dir(authority_manifest)

    assert resolved == checkpoint_dir


def test_recap_real_variant_inference_config_omits_absolute_actions() -> None:
    inference._bootstrap_openpi_paths()
    from openpi.training import config as training_config

    spec = inference.LocalCheckpointPolicySpec(
        config_name=REAL_VARIANT_POLICY_CONFIG_NAME,
        policy_contract={
            "base_train_config_name": "pi0_libero",
            "data_factory_kind": REAL_VARIANT_DATA_FACTORY_KIND,
            "extra_delta_transform": False,
            "training_transform_graph": {"outputs": ["LiberoOutputs"]},
            "inference_transform_graph": {"outputs": ["LiberoOutputs"]},
        },
    )

    train_config = inference._build_train_config_for_policy_spec(
        training_config,
        policy_spec=spec,
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    output_names = [type(transform).__name__ for transform in data_config.data_transforms.outputs]

    assert train_config.name == REAL_VARIANT_POLICY_CONFIG_NAME
    assert output_names == ["LiberoOutputs"]


def test_real_variant_export_prioritizes_upstream_openpi_over_work_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = ModuleType("openpi")
    shadow.__file__ = str(REPO_ROOT / "work" / "openpi" / "__init__.py")
    shadow_child = ModuleType("openpi.models")
    shadow_child.__file__ = str(REPO_ROOT / "work" / "openpi" / "models.py")
    monkeypatch.setitem(sys.modules, "openpi", shadow)
    monkeypatch.setitem(sys.modules, "openpi.models", shadow_child)
    monkeypatch.syspath_prepend(str(REPO_ROOT / "work"))

    real_variant_export._prioritize_upstream_openpi_imports()

    assert sys.path[0] == str(real_variant_export.OPENPI_SRC)
    assert sys.path[1] == str(real_variant_export.OPENPI_CLIENT_SRC)
    assert sys.modules.get("openpi") is not shadow
    assert "openpi.models" not in sys.modules
