from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import interface_localization_surface_inventory


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _surface_by_name(payload: dict[str, Any], surface_name: str) -> dict[str, Any]:
    for entry in payload["surfaces"]:
        if entry["surface_name"] == surface_name:
            return dict(entry)
    raise AssertionError(f"missing surface: {surface_name}")


def _dependency_by_name(payload: dict[str, Any], surface_name: str) -> dict[str, Any]:
    for entry in payload["dependency_checks"]:
        if entry["surface_name"] == surface_name:
            return dict(entry)
    raise AssertionError(f"missing dependency: {surface_name}")


def _blocker_by_name(payload: dict[str, Any], surface_name: str) -> dict[str, Any]:
    for entry in payload["blockers"]:
        if entry["surface_name"] == surface_name:
            return dict(entry)
    raise AssertionError(f"missing blocker: {surface_name}")


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        interface_localization_surface_inventory.main(["--help"])
    assert exc_info.value.code == 0


def test_main_writes_inventory_and_blockers(tmp_path: Path) -> None:
    output_dir = tmp_path / "interface_localization_sprint"
    evidence_json = tmp_path / "task-2-surface-inventory.json"

    exit_code = interface_localization_surface_inventory.main(
        [
            "--output-dir",
            str(output_dir),
            "--evidence-json",
            str(evidence_json),
        ]
    )

    assert exit_code == 0
    inventory = _read_json(
        output_dir
        / interface_localization_surface_inventory.REPLAY_SURFACE_INVENTORY_JSON_NAME
    )
    blockers = _read_json(
        output_dir
        / interface_localization_surface_inventory.CONDITIONAL_BLOCKERS_JSON_NAME
    )
    evidence = _read_json(evidence_json)

    assert (
        inventory["schema_version"]
        == interface_localization_surface_inventory.INVENTORY_SCHEMA_VERSION
    )
    assert (
        inventory["artifact_kind"]
        == interface_localization_surface_inventory.INVENTORY_ARTIFACT_KIND
    )
    assert inventory["provenance_class"] == "static"
    assert inventory["dependency_check_order"] == [
        "python_module.gr00t",
        "path.submodules/Isaac-GR00T",
        "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
    ]
    assert (
        _dependency_by_name(inventory, "python_module.gr00t")["path_kind"]
        == "python_module"
    )
    assert (
        _dependency_by_name(inventory, "path.submodules/Isaac-GR00T")["path_kind"]
        == "directory"
    )
    assert (
        _dependency_by_name(
            inventory,
            "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
        )["path_kind"]
        == "file"
    )

    assert (
        _surface_by_name(inventory, "replay_action_chunk_helper")["availability"]
        == "available"
    )
    assert (
        _surface_by_name(inventory, "runtime_prompt_override_surface")["availability"]
        == "available"
    )
    assert (
        _surface_by_name(inventory, "text_rewrite_surface")["availability"]
        == "available"
    )
    assert (
        _surface_by_name(inventory, "replay_action_chunk_helper")["provenance_class"]
        == "replay_live"
    )
    assert (
        _surface_by_name(inventory, "runtime_prompt_override_surface")[
            "provenance_class"
        ]
        == "server_live"
    )
    assert (
        _surface_by_name(inventory, "text_rewrite_surface")["required_entrypoint"]
        == "work/recap/collector.py::collect_episode"
    )

    stock_surface = _surface_by_name(inventory, "stock_mainline_server_entrypoint")
    assert stock_surface["status"] in {"survived", "blocked_missing_upstream"}
    assert stock_surface["availability"] in {"available", "blocked_missing_upstream"}
    assert (
        stock_surface["required_entrypoint"]
        == "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"
    )
    if stock_surface["status"] == "blocked_missing_upstream":
        assert stock_surface["missing_paths"]
    else:
        assert stock_surface["absolute_path"]
    assert (
        blockers["schema_version"]
        == interface_localization_surface_inventory.BLOCKER_SCHEMA_VERSION
    )
    assert evidence["backpointer"]["replay_surface_inventory_json"] == str(
        output_dir
        / interface_localization_surface_inventory.REPLAY_SURFACE_INVENTORY_JSON_NAME
    )


def test_dependency_overrides_block_custom_and_stock_without_crash() -> None:
    inventory = interface_localization_surface_inventory.build_replay_surface_inventory(
        REPO_ROOT,
        availability_overrides={
            "python_module.gr00t": False,
            "path.submodules/Isaac-GR00T": False,
            "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py": False,
        },
    )
    blockers = interface_localization_surface_inventory.build_conditional_blockers(
        inventory,
        repo_root=REPO_ROOT,
    )

    custom_surface = _surface_by_name(inventory, "custom_advantage_aware_server_cli")
    stock_surface = _surface_by_name(inventory, "stock_mainline_server_entrypoint")
    gr00t_dep = _dependency_by_name(inventory, "python_module.gr00t")
    upstream_dep = _dependency_by_name(inventory, "path.submodules/Isaac-GR00T")

    assert gr00t_dep["status"] == "blocked_missing_upstream"
    assert gr00t_dep["missing_modules"] == ["gr00t"]
    assert custom_surface["status"] == "blocked_missing_upstream"
    assert custom_surface["availability"] == "blocked_missing_upstream"
    assert custom_surface["missing_modules"] == ["gr00t"]
    assert custom_surface["blocked_reason"]

    assert upstream_dep["status"] == "blocked_missing_upstream"
    assert stock_surface["status"] == "blocked_missing_upstream"
    assert stock_surface["availability"] == "blocked_missing_upstream"
    assert stock_surface["missing_paths"]
    assert Path(stock_surface["missing_paths"][0]).is_absolute()
    assert stock_surface["blocked_reason"]

    assert _blocker_by_name(blockers, "custom_advantage_aware_server_cli")[
        "missing_modules"
    ] == ["gr00t"]
    assert _blocker_by_name(blockers, "stock_mainline_server_entrypoint")[
        "missing_paths"
    ]
    assert (
        _blocker_by_name(blockers, "upstream_gather_family_surface")["status"]
        == "blocked_missing_upstream"
    )


def test_blocked_missing_upstream_status_enum_is_legal() -> None:
    inventory = interface_localization_surface_inventory.build_replay_surface_inventory(
        REPO_ROOT,
        availability_overrides={
            "python_module.gr00t": False,
            "path.submodules/Isaac-GR00T": False,
            "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py": False,
        },
    )
    blockers = interface_localization_surface_inventory.build_conditional_blockers(
        inventory,
        repo_root=REPO_ROOT,
    )

    assert (
        "blocked_missing_upstream"
        in inventory["input_baseline_summary"]["status_allowlist"]
    )
    assert inventory["input_baseline_summary"]["provenance_class_allowlist"] == [
        "static",
        "synthetic",
        "replay_live",
        "server_live",
    ]
    assert blockers["blockers"]
    for blocker in blockers["blockers"]:
        assert blocker["status"] == "blocked_missing_upstream"
        assert blocker["blocked_reason"]
        assert (
            blocker["status"] in inventory["input_baseline_summary"]["status_allowlist"]
        )
        assert (
            blocker["provenance_class"]
            in inventory["input_baseline_summary"]["provenance_class_allowlist"]
        )
        if blocker["surface_name"].startswith("path.") or blocker[
            "surface_name"
        ].startswith("upstream_"):
            assert blocker["missing_paths"]
