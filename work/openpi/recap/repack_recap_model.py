from __future__ import annotations

import ast
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAY_MANIFEST_FILENAME = "overlay_materialization.json"
SMOKE_FORWARD_SCHEMA_VERSION = "openpi_recap_smoke_forward_v2"
TASK11_VERIFICATION_SCHEMA_VERSION = "openpi_recap_task11_verification_v1"
REQUIRED_OVERLAY_FILES: tuple[str, ...] = (
    "src/openpi/training/config.py",
    "src/openpi/policies/policy_config.py",
    "src/openpi/recap_overlay/__init__.py",
    "src/openpi/recap_overlay/config.py",
    "src/openpi/recap_overlay/modeling.py",
    "src/openpi/recap_overlay/training.py",
)
REQUIRED_BACKUP_FILES: tuple[str, ...] = (
    "src/openpi/training/_upstream_openpi_recap_training_config.py",
    "src/openpi/policies/_upstream_openpi_recap_policy_config.py",
)
EXPECTED_SOURCE_BACKUPS: tuple[tuple[str, str], ...] = (
    (
        "src/openpi/training/config.py",
        "src/openpi/training/_upstream_openpi_recap_training_config.py",
    ),
    (
        "src/openpi/policies/policy_config.py",
        "src/openpi/policies/_upstream_openpi_recap_policy_config.py",
    ),
)
EXPECTED_KEY_SOURCE_RELATIVE_PATHS: tuple[str, ...] = (
    "src/openpi/models/pi0.py",
    "src/openpi/training/config.py",
    "src/openpi/policies/policy_config.py",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_openpi_tree(openpi_tree: str | Path) -> Path:
    path = Path(openpi_tree)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"openpi tree does not exist: {resolved}")
    return resolved


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def load_overlay_materialization_manifest(openpi_tree: str | Path) -> dict[str, object]:
    resolved_tree = _resolve_openpi_tree(openpi_tree)
    manifest_path = resolved_tree / OVERLAY_MANIFEST_FILENAME
    import json

    payload = cast(object, json.loads(_read_text(manifest_path)))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {manifest_path}")
    return {
        str(key): value for key, value in cast(dict[object, object], payload).items()
    }


def _manifest_string_list(manifest: dict[str, object], field: str) -> list[str]:
    raw = manifest.get(field)
    if not isinstance(raw, list):
        raise ValueError(f"overlay manifest field {field!r} must be a list")
    values: list[str] = []
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"overlay manifest field {field!r} contains invalid entry")
        values.append(value)
    return values


def _manifest_backup_pairs(manifest: dict[str, object]) -> list[tuple[str, str]]:
    raw = manifest.get("source_backups")
    if not isinstance(raw, list):
        raise ValueError("overlay manifest field 'source_backups' must be a list")
    pairs: list[tuple[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("overlay manifest source_backups entries must be objects")
        source_relative_path = entry.get("source_relative_path")
        backup_relative_path = entry.get("backup_relative_path")
        if (
            not isinstance(source_relative_path, str)
            or not source_relative_path.strip()
        ):
            raise ValueError(
                "overlay manifest source_backups.source_relative_path must be non-empty"
            )
        if (
            not isinstance(backup_relative_path, str)
            or not backup_relative_path.strip()
        ):
            raise ValueError(
                "overlay manifest source_backups.backup_relative_path must be non-empty"
            )
        pairs.append((source_relative_path, backup_relative_path))
    return pairs


def _validate_overlay_manifest_shape(manifest: dict[str, object]) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version != "openpi_recap_overlay_materialization_v1":
        raise ValueError("overlay manifest schema_version mismatch")
    pinned_commit = manifest.get("pinned_commit")
    source_tree_commit = manifest.get("source_tree_commit")
    if not isinstance(pinned_commit, str) or not pinned_commit.strip():
        raise ValueError("overlay manifest pinned_commit must be non-empty")
    if not isinstance(source_tree_commit, str) or not source_tree_commit.strip():
        raise ValueError("overlay manifest source_tree_commit must be non-empty")
    if pinned_commit != source_tree_commit:
        raise ValueError("overlay manifest pinned_commit/source_tree_commit mismatch")

    observed_overlay_files = tuple(
        sorted(_manifest_string_list(manifest, "overlay_file_list"))
    )
    if observed_overlay_files != tuple(sorted(REQUIRED_OVERLAY_FILES)):
        raise ValueError(
            "overlay manifest overlay_file_list does not match expected task-11 overlay files"
        )

    observed_backups = tuple(sorted(_manifest_backup_pairs(manifest)))
    if observed_backups != tuple(sorted(EXPECTED_SOURCE_BACKUPS)):
        raise ValueError(
            "overlay manifest source_backups does not match expected task-11 backups"
        )

    raw_key_source_files = manifest.get("key_source_files")
    if not isinstance(raw_key_source_files, list):
        raise ValueError("overlay manifest key_source_files must be a list")
    observed_relative_paths: list[str] = []
    for entry in raw_key_source_files:
        if not isinstance(entry, dict):
            raise ValueError(
                "overlay manifest key_source_files entries must be objects"
            )
        relative_path = entry.get("relative_path")
        sha256 = entry.get("sha256")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(
                "overlay manifest key_source_files.relative_path must be non-empty"
            )
        if not isinstance(sha256, str) or not sha256.strip():
            raise ValueError(
                "overlay manifest key_source_files.sha256 must be non-empty"
            )
        observed_relative_paths.append(relative_path)
    if tuple(sorted(observed_relative_paths)) != tuple(
        sorted(EXPECTED_KEY_SOURCE_RELATIVE_PATHS)
    ):
        raise ValueError(
            "overlay manifest key_source_files does not match expected task-11 key source set"
        )


def validate_materialized_overlay_tree(openpi_tree: str | Path) -> dict[str, object]:
    resolved_tree = _resolve_openpi_tree(openpi_tree)
    manifest = load_overlay_materialization_manifest(resolved_tree)
    _validate_overlay_manifest_shape(manifest)
    missing_files = [
        relative_path
        for relative_path in REQUIRED_OVERLAY_FILES + REQUIRED_BACKUP_FILES
        if not (resolved_tree / relative_path).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "materialized overlay tree missing required files: "
            + ", ".join(missing_files)
        )
    return {
        "openpi_tree": str(resolved_tree),
        "manifest": manifest,
        "required_overlay_files": list(REQUIRED_OVERLAY_FILES),
        "required_backup_files": list(REQUIRED_BACKUP_FILES),
    }


def _clear_loaded_openpi_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "openpi" or module_name.startswith("openpi."):
            del sys.modules[module_name]


def load_overlay_module(openpi_tree: str | Path, module_name: str) -> Any:
    resolved_tree = _resolve_openpi_tree(openpi_tree)
    src_root = resolved_tree / "src"
    if not src_root.is_dir():
        raise FileNotFoundError(f"materialized tree missing src root: {src_root}")
    _clear_loaded_openpi_modules()
    sys.path.insert(0, str(src_root))
    try:
        return importlib.import_module(module_name)
    finally:
        del sys.path[0]


def inspect_training_config_overlay_source(
    openpi_tree: str | Path,
) -> dict[str, object]:
    resolved_tree = _resolve_openpi_tree(openpi_tree)
    config_path = resolved_tree / "src/openpi/training/config.py"
    parsed = ast.parse(_read_text(config_path), filename=str(config_path))
    string_literals: set[str] = set()
    metadata_keys: list[str] = []
    has_cli = False
    has_get_config = False
    for node in ast.walk(parsed):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_literals.add(node.value)
        if isinstance(node, ast.FunctionDef):
            if node.name == "cli":
                has_cli = True
            if node.name == "get_config":
                has_get_config = True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "PI05_LIBERO_RECAP_POLICY_METADATA"
                ):
                    if isinstance(node.value, ast.Dict):
                        for key_node in node.value.keys:
                            if isinstance(key_node, ast.Constant) and isinstance(
                                key_node.value, str
                            ):
                                metadata_keys.append(key_node.value)
    return {
        "path": str(config_path),
        "string_literals": sorted(string_literals),
        "policy_metadata_keys": sorted(metadata_keys),
        "has_cli": has_cli,
        "has_get_config": has_get_config,
    }


def run_smoke_forward(
    *,
    openpi_tree: str | Path,
    config_name: str,
    checkpoint_source: str,
    cfg_scale: float | None = None,
) -> dict[str, object]:
    validation = validate_materialized_overlay_tree(openpi_tree)
    resolved_tree = _resolve_openpi_tree(openpi_tree)
    training_module = load_overlay_module(openpi_tree, "openpi.recap_overlay.training")
    overlay_payload = training_module.build_smoke_forward_report(
        config_name=str(config_name),
        checkpoint_source=str(checkpoint_source),
        cfg_scale=cfg_scale,
    )
    if not isinstance(overlay_payload, dict):
        raise TypeError(
            "expected overlay smoke forward payload dict, got "
            + f"{type(overlay_payload).__name__}"
        )
    payload = {str(key): value for key, value in overlay_payload.items()}
    payload["schema_version"] = SMOKE_FORWARD_SCHEMA_VERSION
    payload["artifact_kind"] = "task11_overlay_smoke_report"
    payload["verification_scope"] = "overlay_smoke_only"
    payload["verification_mode"] = "synthetic_pure_python_overlay_smoke"
    payload["synthetic_loss_model"] = True
    payload["loads_real_checkpoint"] = False
    payload["executes_real_model_forward"] = False
    payload["verifies_openpi_training_runtime_route"] = False
    payload["verifies_openpi_policy_runtime_route"] = False
    payload["verified_path_scopes"] = []
    payload["proves"] = [
        "materialized_overlay_tree_is_importable",
        "overlay_recap_config_surface_is_loadable",
        "overlay_synthetic_conditioned_unconditioned_cfg_smoke_paths_run",
        "overlay_loss_decomposition_fields_remain_self_consistent",
    ]
    payload["does_not_prove"] = [
        "real_checkpoint_loaded",
        "real_model_forward_executed",
        "openpi_training_runtime_route_verified",
        "openpi_policy_runtime_route_verified",
        "full_path_verified",
        "paper_full_path_verified",
    ]
    payload["decision_text"] = (
        "this artifact is a synthetic pure-Python overlay smoke report; it proves importable overlay semantics only and must not be read as full-path or paper-full verification"
    )
    payload["source_refs"] = {
        "materialized_openpi_tree": str(resolved_tree),
        "overlay_materialization": str(resolved_tree / OVERLAY_MANIFEST_FILENAME),
    }
    payload["overlay_validation"] = {
        "openpi_tree": validation["openpi_tree"],
        "manifest": validation["manifest"],
        "required_overlay_files": validation["required_overlay_files"],
        "required_backup_files": validation["required_backup_files"],
    }
    return payload


def build_task11_verification_artifact(
    smoke_report: dict[str, object],
) -> dict[str, object]:
    source_refs = cast(dict[str, object], smoke_report.get("source_refs", {}))
    overlay_validation = cast(
        dict[str, object], smoke_report.get("overlay_validation", {})
    )
    return {
        "schema_version": TASK11_VERIFICATION_SCHEMA_VERSION,
        "task11_verified": False,
        "verification_mode": "explicit_negative_overlay_smoke_only",
        "verified_path_scopes": [],
        "blocking_reasons": [
            "synthetic_smoke_only",
            "real_checkpoint_not_loaded",
            "real_model_forward_not_executed",
            "training_runtime_route_not_verified",
            "policy_runtime_route_not_verified",
            "full_path_not_verified",
            "paper_full_path_not_verified",
        ],
        "decision_text": (
            "task-11 smoke evidence is present, but it only verifies synthetic pure-Python overlay semantics and cannot unlock full-path or paper-full wording"
        ),
        "smoke_schema_version": smoke_report.get("schema_version"),
        "smoke_artifact_kind": smoke_report.get("artifact_kind"),
        "overlay_pinned_commit": cast(
            dict[str, object], overlay_validation.get("manifest", {})
        ).get("pinned_commit"),
        "proves": smoke_report.get("proves"),
        "does_not_prove": smoke_report.get("does_not_prove"),
        "source_refs": {
            "materialized_openpi_tree": source_refs.get("materialized_openpi_tree"),
            "overlay_materialization": source_refs.get("overlay_materialization"),
        },
    }


def write_task11_smoke_artifact(path: str | Path, payload: dict[str, object]) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = REPO_ROOT / target
    target = target.resolve()
    write_json(target, payload)
    return target


def build_config_summary(
    openpi_tree: str | Path, config_name: str
) -> dict[str, object]:
    config_module = load_overlay_module(openpi_tree, "openpi.recap_overlay.config")
    config = config_module.build_recap_config(config_name)
    return asdict(config)


__all__ = [
    "OVERLAY_MANIFEST_FILENAME",
    "REQUIRED_BACKUP_FILES",
    "REQUIRED_OVERLAY_FILES",
    "build_config_summary",
    "inspect_training_config_overlay_source",
    "load_overlay_materialization_manifest",
    "load_overlay_module",
    "build_task11_verification_artifact",
    "run_smoke_forward",
    "validate_materialized_overlay_tree",
    "write_task11_smoke_artifact",
]
