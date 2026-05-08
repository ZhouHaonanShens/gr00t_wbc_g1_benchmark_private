from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_CONTRACT_SCHEMA_VERSION = "openpi_real_variant_policy_contract_v1"
REAL_VARIANT_DATA_FACTORY_KIND = "real_variant_simple_libero_recap"
REAL_VARIANT_POLICY_CONFIG_NAME = "pi0_libero_recap_real_variant"
POLICY_CONFIG_MISMATCH_BLOCKER = "BLOCK_CHECKPOINT_POLICY_CONFIG_MISMATCH"
LIBERO_ASSET_ID = "physical-intelligence/libero"
NORM_STATS_RELATIVE_PATH = Path("assets") / LIBERO_ASSET_ID / "norm_stats.json"


def build_norm_stats_metadata(norm_stats_json_path: Path) -> dict[str, Any]:
    """Return stable metadata for an OpenPI norm_stats.json file."""
    raw = norm_stats_json_path.read_bytes()
    payload = json.loads(raw)
    stats = payload.get("norm_stats", payload)
    state_stats = stats.get("state", {})
    action_stats = stats.get("actions", stats.get("action", {}))
    state_mean = list(state_stats.get("mean", ()))
    action_mean = list(action_stats.get("mean", ()))
    return {
        "path": str(norm_stats_json_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "state_dim": len(state_mean),
        "action_dim": len(action_mean),
        "state_mean_first7": state_mean[:7],
        "action_mean_first7": action_mean[:7],
        "keys": sorted(str(key) for key in stats.keys()),
    }


def build_real_variant_policy_contract(
    *,
    base_train_config_name: str,
    exp_name: str,
    consumer_mode: str,
    fixed_indicator_mode: str | None,
    norm_stats_json_path: Path | None = None,
) -> dict[str, Any]:
    """Describe the train/eval action contract for RECAP real-variant checkpoints."""
    contract: dict[str, Any] = {
        "schema_version": POLICY_CONTRACT_SCHEMA_VERSION,
        "policy_config_name": REAL_VARIANT_POLICY_CONFIG_NAME,
        "base_train_config_name": str(base_train_config_name),
        "exp_name": str(exp_name),
        "data_factory_kind": REAL_VARIANT_DATA_FACTORY_KIND,
        "asset_id": LIBERO_ASSET_ID,
        "extra_delta_transform": False,
        "action_semantics": {
            "training_targets": "absolute_libero_actions",
            "policy_outputs": "absolute_libero_actions",
            "eval_env_actions": "absolute_libero_actions",
            "delta_postprocess_required": False,
            "must_not_apply_output_transforms": ["AbsoluteActions"],
        },
        "training_transform_graph": {
            "inputs": ["VariantPromptTransform", "LiberoInputs"],
            "outputs": ["LiberoOutputs"],
        },
        "inference_transform_graph": {
            "inputs": ["LiberoInputs"],
            "outputs": ["LiberoOutputs"],
        },
        "prompt_transform": {
            "consumer_mode": str(consumer_mode),
            "fixed_indicator_mode": fixed_indicator_mode,
        },
        "repack": {
            "state_key": "observation.state",
            "action_key": "action",
            "prompt_raw_key": "recap_m2.prompt_raw",
            "indicator_key": "recap_m2.indicator_I",
        },
    }
    if norm_stats_json_path is not None and norm_stats_json_path.is_file():
        contract["norm_stats"] = build_norm_stats_metadata(norm_stats_json_path)
    return contract


def attach_real_variant_policy_contract(
    manifest: dict[str, Any],
    *,
    policy_contract: dict[str, Any],
) -> dict[str, Any]:
    """Attach contract metadata using both canonical and easy-to-grep fields."""
    updated = dict(manifest)
    updated["policy_contract"] = dict(policy_contract)
    updated["policy_config_name"] = policy_contract["policy_config_name"]
    updated["data_factory_kind"] = policy_contract["data_factory_kind"]
    updated["extra_delta_transform"] = policy_contract["extra_delta_transform"]
    updated["data_transforms_outputs"] = list(
        policy_contract["training_transform_graph"]["outputs"]
    )
    norm_stats = policy_contract.get("norm_stats")
    if isinstance(norm_stats, dict):
        updated["norm_stats_sha256"] = norm_stats.get("sha256")
        updated["norm_stats_state_dim"] = norm_stats.get("state_dim")
        updated["norm_stats_action_dim"] = norm_stats.get("action_dim")
    return updated


def extract_policy_contract(manifest: dict[str, Any]) -> dict[str, Any] | None:
    contract = manifest.get("policy_contract")
    if isinstance(contract, dict):
        return contract
    if manifest.get("data_factory_kind") == REAL_VARIANT_DATA_FACTORY_KIND:
        return {
            "schema_version": POLICY_CONTRACT_SCHEMA_VERSION,
            "policy_config_name": manifest.get(
                "policy_config_name", REAL_VARIANT_POLICY_CONFIG_NAME
            ),
            "base_train_config_name": manifest.get("train_config_name", "pi0_libero"),
            "data_factory_kind": REAL_VARIANT_DATA_FACTORY_KIND,
            "asset_id": LIBERO_ASSET_ID,
            "extra_delta_transform": bool(manifest.get("extra_delta_transform", False)),
            "training_transform_graph": {
                "outputs": list(manifest.get("data_transforms_outputs", ()))
            },
            "legacy_manifest_backfill": True,
        }
    if manifest.get("schema_version") == "openpi_real_variant_export_v1":
        return {
            "schema_version": POLICY_CONTRACT_SCHEMA_VERSION,
            "policy_config_name": REAL_VARIANT_POLICY_CONFIG_NAME,
            "base_train_config_name": manifest.get("train_config_name", "pi0_libero"),
            "data_factory_kind": REAL_VARIANT_DATA_FACTORY_KIND,
            "asset_id": LIBERO_ASSET_ID,
            "extra_delta_transform": False,
            "training_transform_graph": {"outputs": ["LiberoOutputs"]},
            "inference_transform_graph": {"outputs": ["LiberoOutputs"]},
            "legacy_manifest_backfill": True,
        }
    return None
