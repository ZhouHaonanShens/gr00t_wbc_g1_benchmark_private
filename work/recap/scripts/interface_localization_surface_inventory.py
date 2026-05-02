from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


DEFAULT_ARTIFACT_DIR = "agent/artifacts"
DEFAULT_OUTPUT_SUBDIR = "interface_localization_sprint"
DEFAULT_EVIDENCE_JSON = ".sisyphus/evidence/task-2-surface-inventory.json"

REPLAY_SURFACE_INVENTORY_JSON_NAME = "replay_surface_inventory.json"
CONDITIONAL_BLOCKERS_JSON_NAME = "conditional_blockers.json"

INVENTORY_SCHEMA_VERSION = "interface_localization_replay_surface_inventory_v1"
BLOCKER_SCHEMA_VERSION = "interface_localization_conditional_blockers_v1"
INVENTORY_ARTIFACT_KIND = "replay_surface_inventory"
BLOCKER_ARTIFACT_KIND = "conditional_blockers"

SURFACE_ORDER: tuple[str, ...] = (
    "custom_advantage_aware_server_cli",
    "stock_mainline_server_entrypoint",
    "replay_action_chunk_helper",
    "runtime_prompt_override_surface",
    "text_rewrite_surface",
)

DEPENDENCY_ORDER: tuple[str, ...] = (
    "python_module.gr00t",
    "path.submodules/Isaac-GR00T",
    "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
)

UPSTREAM_ONLY_BLOCKER_TARGETS: tuple[dict[str, str], ...] = (
    {
        "surface_name": "upstream_gather_family_surface",
        "surface_category": "upstream_only_runtime_surface",
        "path_kind": "upstream_symbol_family",
        "provenance_class": "server_live",
        "required_entrypoint": "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
        "blocked_reason": "missing upstream checkout prevents inspection of Gather* execution surfaces",
        "upstream_symbol": "Gather*",
    },
    {
        "surface_name": "upstream_default_angles_surface",
        "surface_category": "upstream_only_runtime_surface",
        "path_kind": "upstream_symbol",
        "provenance_class": "server_live",
        "required_entrypoint": "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
        "blocked_reason": "missing upstream checkout prevents inspection of default_angles handling",
        "upstream_symbol": "default_angles",
    },
    {
        "surface_name": "upstream_last_actions_surface",
        "surface_category": "upstream_only_runtime_surface",
        "path_kind": "upstream_symbol",
        "provenance_class": "server_live",
        "required_entrypoint": "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
        "blocked_reason": "missing upstream checkout prevents inspection of last_actions handling",
        "upstream_symbol": "last_actions",
    },
    {
        "surface_name": "upstream_q_target_surface",
        "surface_category": "upstream_only_runtime_surface",
        "path_kind": "upstream_symbol",
        "provenance_class": "server_live",
        "required_entrypoint": "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
        "blocked_reason": "missing upstream checkout prevents inspection of q_target handling",
        "upstream_symbol": "q_target",
    },
    {
        "surface_name": "upstream_dex3_surface",
        "surface_category": "upstream_only_runtime_surface",
        "path_kind": "upstream_symbol",
        "provenance_class": "server_live",
        "required_entrypoint": "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
        "blocked_reason": "missing upstream checkout prevents inspection of Dex3-specific execution surfaces",
        "upstream_symbol": "Dex3",
    },
    {
        "surface_name": "upstream_official_replay_logging_flags",
        "surface_category": "upstream_only_runtime_surface",
        "path_kind": "upstream_flag_surface",
        "provenance_class": "replay_live",
        "required_entrypoint": "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
        "blocked_reason": "missing upstream checkout prevents inspection of official replay/logging flags",
        "upstream_symbol": "official_replay_logging_flags",
    },
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import interface_localization_contract
from work.recap import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interface_localization_surface_inventory.py",
        description=(
            "Inventory repo-local execution surfaces for interface localization and "
            "emit conditional upstream blockers instead of crashing on missing dependencies."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--artifact-dir",
        type=str,
        default=DEFAULT_ARTIFACT_DIR,
        help=(
            "Artifact root. When --output-dir is empty, JSON artifacts are written to "
            "<artifact-dir>/interface_localization_sprint/."
        ),
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output directory for generated inventory JSON files.",
    )
    _ = parser.add_argument(
        "--evidence-json",
        type=str,
        default=DEFAULT_EVIDENCE_JSON,
        help="Evidence JSON written after artifact generation succeeds.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def resolve_output_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    raw_output_dir = str(args.output_dir).strip()
    if raw_output_dir:
        return state_conditioned_bucket_a_import.validate_output_dir(
            _resolve_path(repo_root, raw_output_dir)
        )
    artifact_dir = _resolve_path(repo_root, str(args.artifact_dir))
    return state_conditioned_bucket_a_import.validate_output_dir(
        artifact_dir / DEFAULT_OUTPUT_SUBDIR
    )


def resolve_evidence_json(repo_root: Path, args: argparse.Namespace) -> Path:
    return _resolve_path(repo_root, str(args.evidence_json))


def _relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _generation_command_for(output_dir: Path, repo_root: Path) -> str:
    display_output_dir = _relpath(repo_root, output_dir)
    return (
        "python3 work/recap/scripts/interface_localization_surface_inventory.py "
        f"--output-dir {display_output_dir}"
    )


def _load_contract_payload() -> dict[str, Any]:
    contract = interface_localization_contract.build_interface_localization_contract()
    _ = interface_localization_contract.assert_contract_matches_canonical(contract)
    return contract


def _baseline_summary(contract_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract_schema_version": str(contract_payload["schema_version"]),
        "baseline_tuple": dict(contract_payload["baseline_tuple"]),
        "status_allowlist": list(contract_payload["status_ontology"]["legal_statuses"]),
        "provenance_class_allowlist": list(
            contract_payload["provenance_classes"]["class_order"]
        ),
        "mainline_task_text_field": str(
            contract_payload["advantage_contract_facts"]["mainline_task_text_field"]
        ),
    }


def _module_dependency_record(
    *,
    repo_root: Path,
    module_name: str,
    availability_override: bool | None = None,
) -> dict[str, Any]:
    spec = (
        None
        if availability_override is False
        else importlib.util.find_spec(module_name)
    )
    if availability_override is True and spec is None:

        class _SyntheticSpec:
            origin = "synthetic_override"
            submodule_search_locations: list[str] | None = None

        spec = _SyntheticSpec()

    search_locations = getattr(spec, "submodule_search_locations", None)
    resolved_locations = [
        str(Path(item).expanduser().resolve()) for item in search_locations or []
    ]
    origin = getattr(spec, "origin", None)
    available = spec is not None
    return {
        "surface_name": f"python_module.{module_name}",
        "surface_category": "python_module_dependency",
        "availability": "available" if available else "blocked_missing_upstream",
        "status": "survived" if available else "blocked_missing_upstream",
        "provenance_class": "static",
        "path_kind": "python_module",
        "required_entrypoint": module_name,
        "relative_path": None,
        "absolute_path": str(Path(origin).expanduser().resolve()) if origin else None,
        "module_search_locations": resolved_locations,
        "blocked_reason": None
        if available
        else f"missing python module dependency: {module_name}",
        "missing_modules": [] if available else [module_name],
        "missing_paths": [],
        "metadata": {
            "repo_root": str(repo_root.resolve()),
            "import_probe": "importlib.util.find_spec",
        },
    }


def _path_dependency_record(
    *,
    repo_root: Path,
    surface_name: str,
    relative_path: str,
    path_kind: str,
    availability_override: bool | None = None,
) -> dict[str, Any]:
    absolute_path = (repo_root / relative_path).resolve()
    available = (
        absolute_path.exists()
        if availability_override is None
        else bool(availability_override)
    )
    return {
        "surface_name": surface_name,
        "surface_category": "path_dependency",
        "availability": "available" if available else "blocked_missing_upstream",
        "status": "survived" if available else "blocked_missing_upstream",
        "provenance_class": "static",
        "path_kind": path_kind,
        "required_entrypoint": relative_path,
        "relative_path": relative_path,
        "absolute_path": str(absolute_path),
        "blocked_reason": None
        if available
        else f"missing required path dependency: {absolute_path}",
        "missing_modules": [],
        "missing_paths": [] if available else [str(absolute_path)],
        "metadata": {
            "repo_root": str(repo_root.resolve()),
            "path_exists": bool(absolute_path.exists()),
        },
    }


def inspect_dependencies(
    repo_root: Path,
    *,
    availability_overrides: Mapping[str, bool] | None = None,
) -> list[dict[str, Any]]:
    overrides = dict(availability_overrides or {})
    dependencies = [
        _module_dependency_record(
            repo_root=repo_root,
            module_name="gr00t",
            availability_override=overrides.get("python_module.gr00t"),
        ),
        _path_dependency_record(
            repo_root=repo_root,
            surface_name="path.submodules/Isaac-GR00T",
            relative_path="submodules/Isaac-GR00T",
            path_kind="directory",
            availability_override=overrides.get("path.submodules/Isaac-GR00T"),
        ),
        _path_dependency_record(
            repo_root=repo_root,
            surface_name="path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
            relative_path="submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
            path_kind="file",
            availability_override=overrides.get(
                "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"
            ),
        ),
    ]
    by_name = {entry["surface_name"]: entry for entry in dependencies}
    return [dict(by_name[name]) for name in DEPENDENCY_ORDER]


def _missing_dependency_details(
    dependencies: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    missing_modules: list[str] = []
    missing_paths: list[str] = []
    dependency_refs: list[str] = []
    for entry in dependencies:
        if str(entry.get("status")) != "blocked_missing_upstream":
            continue
        dependency_refs.append(str(entry["surface_name"]))
        for module_name in entry.get("missing_modules", []):
            value = str(module_name)
            if value not in missing_modules:
                missing_modules.append(value)
        for path in entry.get("missing_paths", []):
            value = str(path)
            if value not in missing_paths:
                missing_paths.append(value)
    return missing_modules, missing_paths, dependency_refs


def _make_surface_record(
    *,
    repo_root: Path,
    surface_name: str,
    surface_category: str,
    path_kind: str,
    provenance_class: str,
    required_entrypoint: str,
    relative_path: str,
    dependency_entries: Sequence[Mapping[str, Any]] = (),
    blocked_reason_if_dependency_missing: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = (repo_root / relative_path).resolve()
    path_exists = path.is_file()
    missing_modules, missing_paths, dependency_refs = _missing_dependency_details(
        dependency_entries
    )

    if missing_modules or missing_paths:
        status = "blocked_missing_upstream"
        availability = "blocked_missing_upstream"
        blocked_reason = blocked_reason_if_dependency_missing or (
            f"surface {surface_name} is blocked by missing upstream dependency evidence"
        )
    elif not path_exists:
        status = "died"
        availability = "missing"
        blocked_reason = f"missing repo-local surface file: {path}"
        missing_paths = [str(path)]
    else:
        status = "survived"
        availability = "available"
        blocked_reason = None

    payload = {
        "surface_name": surface_name,
        "surface_category": surface_category,
        "availability": availability,
        "status": status,
        "provenance_class": provenance_class,
        "path_kind": path_kind,
        "required_entrypoint": required_entrypoint,
        "relative_path": relative_path,
        "absolute_path": str(path),
        "blocked_reason": blocked_reason,
        "missing_modules": missing_modules,
        "missing_paths": missing_paths,
        "dependency_refs": dependency_refs,
        "metadata": dict(metadata or {}),
    }
    return payload


def build_surface_records(
    repo_root: Path,
    dependency_checks: Sequence[Mapping[str, Any]],
    contract_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    dependency_by_name = {
        str(entry["surface_name"]): entry for entry in dependency_checks
    }
    mainline_task_text_field = str(
        contract_payload["advantage_contract_facts"]["mainline_task_text_field"]
    )
    contract_version = str(
        contract_payload["advantage_contract_facts"]["contract_version"]
    )

    surfaces = [
        _make_surface_record(
            repo_root=repo_root,
            surface_name="custom_advantage_aware_server_cli",
            surface_category="runtime_server_cli",
            path_kind="repo_python_script",
            provenance_class="server_live",
            required_entrypoint="work/recap/scripts/3D_recap_run_adv_server.py::_build_parser",
            relative_path="work/recap/scripts/3D_recap_run_adv_server.py",
            dependency_entries=(dependency_by_name["python_module.gr00t"],),
            blocked_reason_if_dependency_missing=(
                "custom advantage-aware server CLI requires python module dependency 'gr00t' "
                "for AdvantageAwareGr00tPolicy and related server imports"
            ),
            metadata={
                "advantage_contract_version": contract_version,
                "advantage_injection_rule": "sign_consistent",
                "task_text_field": mainline_task_text_field,
                "overlay_from_flag": "--overlay-from",
                "overlay_include_regex_flag": "--overlay-include-regex",
                "overlay_exclude_regex_flag": "--overlay-exclude-regex",
                "reference_lines": "work/recap/scripts/3D_recap_run_adv_server.py:37-149,189-245",
            },
        ),
        _make_surface_record(
            repo_root=repo_root,
            surface_name="stock_mainline_server_entrypoint",
            surface_category="runtime_server_entrypoint",
            path_kind="upstream_python_script",
            provenance_class="server_live",
            required_entrypoint="submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
            relative_path="submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
            dependency_entries=(
                dependency_by_name["path.submodules/Isaac-GR00T"],
                dependency_by_name[
                    "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"
                ],
            ),
            blocked_reason_if_dependency_missing=(
                "stock mainline server entrypoint is blocked because the Isaac-GR00T upstream "
                "checkout or run_gr00t_server.py entrypoint is missing"
            ),
            metadata={
                "reference_lines": "work/recap/scripts/state_conditioned_phase0_smoke.py:109-110",
                "server_mode_axis": "stock_serving_path_vs_custom_advantage_aware_path",
            },
        ),
        _make_surface_record(
            repo_root=repo_root,
            surface_name="replay_action_chunk_helper",
            surface_category="replay_helper",
            path_kind="repo_python_function",
            provenance_class="replay_live",
            required_entrypoint=(
                "work/recap/scripts/state_conditioned_snapshot_harvest.py::"
                "_load_replay_action_chunks"
            ),
            relative_path="work/recap/scripts/state_conditioned_snapshot_harvest.py",
            metadata={
                "reference_lines": "work/recap/scripts/state_conditioned_snapshot_harvest.py:1539-1585",
                "normalizer_entrypoint": (
                    "work/recap/scripts/state_conditioned_snapshot_harvest.py::"
                    "_normalize_replay_action_chunk_for_env"
                ),
            },
        ),
        _make_surface_record(
            repo_root=repo_root,
            surface_name="runtime_prompt_override_surface",
            surface_category="runtime_prompt_override",
            path_kind="repo_python_function",
            provenance_class="server_live",
            required_entrypoint=(
                "work/recap/scripts/demo_g1_vla_live.py::_override_task_prompt_in_obs"
            ),
            relative_path="work/recap/scripts/demo_g1_vla_live.py",
            metadata={
                "reference_lines": "work/recap/scripts/demo_g1_vla_live.py:517-522",
                "task_prompt_key": "annotation.human.task_description",
            },
        ),
        _make_surface_record(
            repo_root=repo_root,
            surface_name="text_rewrite_surface",
            surface_category="text_rewrite",
            path_kind="repo_python_function",
            provenance_class="replay_live",
            required_entrypoint="work/recap/collector.py::collect_episode",
            relative_path="work/recap/collector.py",
            metadata={
                "reference_lines": "work/recap/collector.py:614-665",
                "prompt_raw_field": "prompt_raw",
                "prompt_conditioned_field": "prompt_conditioned",
                "runtime_task_description_key": "annotation.human.task_description",
            },
        ),
    ]
    by_name = {entry["surface_name"]: entry for entry in surfaces}
    return [dict(by_name[name]) for name in SURFACE_ORDER]


def build_replay_surface_inventory(
    repo_root: Path,
    *,
    output_dir: Path | None = None,
    availability_overrides: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    contract_payload = _load_contract_payload()
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else (repo_root / DEFAULT_ARTIFACT_DIR / DEFAULT_OUTPUT_SUBDIR).resolve()
    )
    generation_command = _generation_command_for(resolved_output_dir, repo_root)
    dependency_checks = inspect_dependencies(
        repo_root, availability_overrides=availability_overrides
    )
    surfaces = build_surface_records(repo_root, dependency_checks, contract_payload)
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "artifact_kind": INVENTORY_ARTIFACT_KIND,
        "provenance_class": "static",
        "generation_command": generation_command,
        "input_baseline_summary": _baseline_summary(contract_payload),
        "backpointer": {
            "writer_script": "work/recap/scripts/interface_localization_surface_inventory.py",
            "task1_contract_writer": "work/recap/scripts/interface_localization_contract.py",
            "expected_task1_contract_json": str(
                resolved_output_dir / interface_localization_contract.CONTRACT_JSON_NAME
            ),
            "conditional_blockers_json": str(
                resolved_output_dir / CONDITIONAL_BLOCKERS_JSON_NAME
            ),
        },
        "surface_order": list(SURFACE_ORDER),
        "dependency_check_order": list(DEPENDENCY_ORDER),
        "dependency_checks": dependency_checks,
        "surfaces": surfaces,
    }


def _surface_to_blocker(
    surface: Mapping[str, Any],
    *,
    blocker_kind: str,
) -> dict[str, Any]:
    return {
        "surface_name": str(surface["surface_name"]),
        "blocker_kind": blocker_kind,
        "surface_category": str(surface["surface_category"]),
        "availability": str(surface["availability"]),
        "status": str(surface["status"]),
        "provenance_class": str(surface["provenance_class"]),
        "path_kind": str(surface["path_kind"]),
        "required_entrypoint": str(surface["required_entrypoint"]),
        "relative_path": surface.get("relative_path"),
        "absolute_path": surface.get("absolute_path"),
        "blocked_reason": str(surface["blocked_reason"]),
        "missing_modules": list(surface.get("missing_modules", [])),
        "missing_paths": list(surface.get("missing_paths", [])),
        "dependency_refs": list(surface.get("dependency_refs", [])),
        "metadata": dict(surface.get("metadata", {})),
    }


def _build_upstream_only_symbol_blockers(
    inventory_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    dependency_by_name = {
        str(entry["surface_name"]): dict(entry)
        for entry in inventory_payload["dependency_checks"]
    }
    path_dependencies = [
        dependency_by_name["path.submodules/Isaac-GR00T"],
        dependency_by_name[
            "path.submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"
        ],
    ]
    _missing_modules, missing_paths, dependency_refs = _missing_dependency_details(
        path_dependencies
    )
    if not missing_paths:
        return []
    return [
        {
            "surface_name": spec["surface_name"],
            "blocker_kind": "upstream_only_surface",
            "surface_category": spec["surface_category"],
            "availability": "blocked_missing_upstream",
            "status": "blocked_missing_upstream",
            "provenance_class": spec["provenance_class"],
            "path_kind": spec["path_kind"],
            "required_entrypoint": spec["required_entrypoint"],
            "relative_path": "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
            "absolute_path": missing_paths[0],
            "blocked_reason": spec["blocked_reason"],
            "missing_modules": [],
            "missing_paths": list(missing_paths),
            "dependency_refs": list(dependency_refs),
            "metadata": {"upstream_symbol": spec["upstream_symbol"]},
        }
        for spec in UPSTREAM_ONLY_BLOCKER_TARGETS
    ]


def build_conditional_blockers(
    inventory_payload: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    resolved_repo_root = (
        repo_root.resolve() if repo_root is not None else REPO_ROOT.resolve()
    )
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else (
            resolved_repo_root / DEFAULT_ARTIFACT_DIR / DEFAULT_OUTPUT_SUBDIR
        ).resolve()
    )
    blockers = [
        _surface_to_blocker(entry, blocker_kind="dependency")
        for entry in inventory_payload["dependency_checks"]
        if str(entry.get("status")) == "blocked_missing_upstream"
    ]
    blockers.extend(
        _surface_to_blocker(entry, blocker_kind="runtime_surface")
        for entry in inventory_payload["surfaces"]
        if str(entry.get("status")) == "blocked_missing_upstream"
    )
    blockers.extend(_build_upstream_only_symbol_blockers(inventory_payload))
    generation_command = _generation_command_for(
        resolved_output_dir, resolved_repo_root
    )
    return {
        "schema_version": BLOCKER_SCHEMA_VERSION,
        "artifact_kind": BLOCKER_ARTIFACT_KIND,
        "provenance_class": "static",
        "generation_command": generation_command,
        "input_baseline_summary": dict(inventory_payload["input_baseline_summary"]),
        "backpointer": {
            "writer_script": "work/recap/scripts/interface_localization_surface_inventory.py",
            "replay_surface_inventory_json": str(
                resolved_output_dir / REPLAY_SURFACE_INVENTORY_JSON_NAME
            ),
        },
        "blockers": blockers,
    }


def write_artifacts(
    *,
    output_dir: Path,
    evidence_json: Path,
    inventory_payload: Mapping[str, Any],
    blocker_payload: Mapping[str, Any],
) -> dict[str, str]:
    inventory_path = state_conditioned_bucket_a_import._write_json(
        output_dir / REPLAY_SURFACE_INVENTORY_JSON_NAME,
        inventory_payload,
    )
    blocker_path = state_conditioned_bucket_a_import._write_json(
        output_dir / CONDITIONAL_BLOCKERS_JSON_NAME,
        blocker_payload,
    )
    evidence_payload = {
        "schema_version": "interface_localization_task2_evidence_v1",
        "artifact_kind": "interface_localization_surface_inventory_evidence",
        "provenance_class": "static",
        "generation_command": str(inventory_payload["generation_command"]),
        "input_baseline_summary": dict(inventory_payload["input_baseline_summary"]),
        "backpointer": {
            "replay_surface_inventory_json": str(inventory_path),
            "conditional_blockers_json": str(blocker_path),
            "test_command": "python3 -m pytest tests/recap/test_interface_localization_surface_inventory.py -q",
        },
    }
    evidence_path = state_conditioned_bucket_a_import._write_json(
        evidence_json,
        evidence_payload,
    )
    return {
        "replay_surface_inventory_json": str(inventory_path),
        "conditional_blockers_json": str(blocker_path),
        "evidence_json": str(evidence_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        output_dir = resolve_output_dir(REPO_ROOT, args)
        evidence_json = resolve_evidence_json(REPO_ROOT, args)
        inventory_payload = build_replay_surface_inventory(
            REPO_ROOT,
            output_dir=output_dir,
        )
        blocker_payload = build_conditional_blockers(
            inventory_payload,
            output_dir=output_dir,
            repo_root=REPO_ROOT,
        )
        written_paths = write_artifacts(
            output_dir=output_dir,
            evidence_json=evidence_json,
            inventory_payload=inventory_payload,
            blocker_payload=blocker_payload,
        )
        print(
            _canonical_json_text(
                {
                    "status": "PASS",
                    "output_dir": str(output_dir),
                    **written_paths,
                }
            ),
            end="",
        )
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
