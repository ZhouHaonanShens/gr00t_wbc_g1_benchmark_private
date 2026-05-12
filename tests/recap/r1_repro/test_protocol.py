from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from work.recap.r1_repro import protocol


def test_p0b_protocol_constants_are_frozen() -> None:
    assert protocol.P0B_PROTOCOL.ckpt_root == protocol.RAW_HF_SNAPSHOT_ROOT
    assert (
        protocol.P0B_PROTOCOL.driver_script
        == "work/recap/scripts/gr00t_g3_formal_eval.py"
    )
    assert protocol.P0B_PROTOCOL.driver_sha256 == protocol.P0B_DRIVER_SHA256
    assert protocol.P0B_PROTOCOL.env_name == protocol.P0B_ENV_NAME
    assert protocol.P0B_PROTOCOL.prompt == protocol.P0B_PROMPT
    assert protocol.P0B_PROTOCOL.seed_base == 20000
    assert protocol.P0B_PROTOCOL.episodes == 30
    assert protocol.P0B_PROTOCOL.max_episode_steps == 1440
    assert protocol.P0B_PROTOCOL.n_action_steps == 20
    with pytest.raises(FrozenInstanceError):
        protocol.P0B_PROTOCOL.episodes = 31  # type: ignore[misc]


def test_t81_b0_protocol_constants_are_frozen() -> None:
    assert protocol.T81_B0_PROTOCOL.ckpt_root == protocol.T81_B0_VARIANT_CKPT_ROOT
    assert (
        protocol.T81_B0_PROTOCOL.driver_script
        == "work/recap/safe_sft/t8_1_nav_postlift.py"
    )
    assert protocol.T81_B0_PROTOCOL.driver_sha256 == protocol.T81_B0_DRIVER_SHA256
    assert protocol.T81_B0_PROTOCOL.seed_base == 2026051000
    assert protocol.T81_B0_PROTOCOL.episodes == 10
    assert protocol.T81_B0_PROTOCOL.max_episode_steps == 720
    assert protocol.T81_B0_PROTOCOL.n_action_steps == 20
    with pytest.raises(FrozenInstanceError):
        protocol.T81_B0_PROTOCOL.seed_base = 1  # type: ignore[misc]


def test_no_t8_imports_in_protocol() -> None:
    source_path = Path(protocol.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.append(node.module)

    assert not any("safe_sft" in name or "t8_" in name for name in imported_names)


def test_variant_root_exists_at_module_load() -> None:
    assert protocol.T81_B0_VARIANT_CKPT_ROOT.is_dir()


def test_raw_hf_snapshot_root_exists_at_module_load() -> None:
    assert protocol.RAW_HF_SNAPSHOT_ROOT.is_dir()


def test_protocol_deterministic_sha_stable() -> None:
    first = protocol.protocol_deterministic_sha(protocol.P0B_PROTOCOL)
    second = protocol.protocol_deterministic_sha(protocol.P0B_PROTOCOL)
    assert first == second
    assert len(first) == 64


def test_p0b_cuda_pin_is_explicit_string() -> None:
    assert protocol.P0B_PROTOCOL.cuda_visible_devices == "1"
    assert isinstance(protocol.P0B_PROTOCOL.cuda_visible_devices, str)


def test_t81_b0_ckpt_root_is_variant_path_not_none() -> None:
    assert protocol.T81_B0_PROTOCOL.ckpt_root is not None
    assert protocol.T81_B0_PROTOCOL.ckpt_root == protocol.T81_B0_VARIANT_CKPT_ROOT


def test_verify_ckpt_config_shas_passes_current_pins() -> None:
    protocol.verify_ckpt_config_shas()
