#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MANIFEST_SCHEMA_VERSION = "openpi_recap_overlay_materialization_v1"
DEFAULT_PINNED_COMMIT = "fdc03f527881cdfc8ae1a168ed6a20c60edbbbcc"
KEY_SOURCE_FILE_HASHES: dict[str, str] = {
    "src/openpi/models/pi0.py": "24b32d7c6ed5e409459afa8ca5af96a4dae09c2dcdf48e828907e411f4f12342",
    "src/openpi/training/config.py": "c34231f888dba5dc981d808765b3d98775ab845b04313bf53fcecd567c520b0d",
    "src/openpi/policies/policy_config.py": "aaf42ab04a33b6c91d2447926211646c33ebd8ebfa427f026d7b6c4f7c45ec52",
}
SOURCE_BACKUP_PATHS: dict[str, str] = {
    "src/openpi/training/config.py": "src/openpi/training/_upstream_openpi_recap_training_config.py",
    "src/openpi/policies/policy_config.py": "src/openpi/policies/_upstream_openpi_recap_policy_config.py",
}
EXCLUDED_COPY_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}
OVERLAY_MANIFEST_FILENAME = "overlay_materialization.json"
DEFAULT_OVERLAY_ROOT = REPO_ROOT / "work" / "openpi" / "overlays" / "openpi_recap"
OVERLAY_PAYLOAD_ROOT_NAMES = {"src"}


class OverlayMaterializationError(RuntimeError):
    pass


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _ = path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    _ = tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _ = tmp_path.replace(path)


def resolve_source_tree_commit(source_tree: Path) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],
        cwd=source_tree,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise OverlayMaterializationError(
            "failed to resolve source tree commit: " + result.stderr.strip()
        )
    commit = result.stdout.strip()
    if not commit:
        raise OverlayMaterializationError(
            f"resolved empty commit SHA for source tree {source_tree}"
        )
    return commit


def verify_pinned_source_tree(
    *,
    source_tree: Path,
    pinned_commit: str,
    key_source_hashes: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    resolved_source_tree = source_tree.resolve()
    if not resolved_source_tree.is_dir():
        raise OverlayMaterializationError(
            f"source tree does not exist: {resolved_source_tree}"
        )
    actual_commit = resolve_source_tree_commit(resolved_source_tree)
    normalized_pinned_commit = str(pinned_commit).strip()
    if actual_commit != normalized_pinned_commit:
        raise OverlayMaterializationError(
            "source tree commit mismatch: "
            + f"expected {normalized_pinned_commit}, got {actual_commit}"
        )

    hashes = key_source_hashes or KEY_SOURCE_FILE_HASHES
    validated: list[dict[str, str]] = []
    for relative_path, expected_sha256 in hashes.items():
        file_path = resolved_source_tree / relative_path
        if not file_path.is_file():
            raise OverlayMaterializationError(
                "missing key source file required by task-11 source pin: "
                + str(file_path)
            )
        observed_sha256 = _sha256_file(file_path)
        if observed_sha256 != expected_sha256:
            raise OverlayMaterializationError(
                "key source file hash mismatch: "
                + f"{relative_path} expected {expected_sha256} got {observed_sha256}"
            )
        validated.append(
            {
                "relative_path": relative_path,
                "sha256": observed_sha256,
            }
        )
    return validated


def _is_overlay_payload_file(relative_path: Path) -> bool:
    if not relative_path.parts:
        return False
    if relative_path.parts[0] not in OVERLAY_PAYLOAD_ROOT_NAMES:
        return False
    return not any(part in EXCLUDED_COPY_NAMES for part in relative_path.parts)


def collect_overlay_file_list(overlay_root: Path) -> list[str]:
    resolved_overlay_root = overlay_root.resolve()
    if not resolved_overlay_root.is_dir():
        raise OverlayMaterializationError(
            f"overlay root does not exist: {resolved_overlay_root}"
        )
    files = sorted(
        str(relative_path)
        for path in resolved_overlay_root.rglob("*")
        if path.is_file()
        for relative_path in (path.relative_to(resolved_overlay_root),)
        if _is_overlay_payload_file(relative_path)
    )
    if not files:
        raise OverlayMaterializationError(
            f"overlay root contains no payload files under src/: {resolved_overlay_root}"
        )
    return files


def _copytree_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in EXCLUDED_COPY_NAMES}


def _remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _backup_upstream_source(
    *, output_dir: Path, source_relative_path: str, backup_relative_path: str
) -> dict[str, str]:
    source_path = output_dir / source_relative_path
    if not source_path.is_file():
        raise OverlayMaterializationError(
            "cannot backup upstream file before overlay; missing copied source file: "
            + str(source_path)
        )
    backup_path = output_dir / backup_relative_path
    _ = backup_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_path(backup_path)
    _ = shutil.copy2(source_path, backup_path)
    return {
        "source_relative_path": source_relative_path,
        "backup_relative_path": backup_relative_path,
    }


def _apply_overlay_files(
    *, output_dir: Path, overlay_root: Path, overlay_files: list[str]
) -> None:
    for relative_path in overlay_files:
        source_path = overlay_root / relative_path
        destination_path = output_dir / relative_path
        _ = destination_path.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copy2(source_path, destination_path)


def materialize_overlay(
    *,
    source_tree: Path,
    pinned_commit: str,
    overlay_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    validated_key_files = verify_pinned_source_tree(
        source_tree=source_tree,
        pinned_commit=pinned_commit,
    )
    overlay_files = collect_overlay_file_list(overlay_root)
    resolved_output_dir = output_dir.resolve()
    _remove_existing_path(resolved_output_dir)
    _ = shutil.copytree(
        source_tree.resolve(), resolved_output_dir, ignore=_copytree_ignore
    )

    backups: list[dict[str, str]] = []
    for source_relative_path, backup_relative_path in SOURCE_BACKUP_PATHS.items():
        backups.append(
            _backup_upstream_source(
                output_dir=resolved_output_dir,
                source_relative_path=source_relative_path,
                backup_relative_path=backup_relative_path,
            )
        )

    _apply_overlay_files(
        output_dir=resolved_output_dir,
        overlay_root=overlay_root.resolve(),
        overlay_files=overlay_files,
    )

    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_tree": str(source_tree.resolve()),
        "source_tree_commit": resolve_source_tree_commit(source_tree.resolve()),
        "pinned_commit": str(pinned_commit).strip(),
        "key_source_files": validated_key_files,
        "overlay_root": str(overlay_root.resolve()),
        "overlay_file_list": overlay_files,
        "source_backups": backups,
        "output_tree": str(resolved_output_dir),
    }
    _write_json(resolved_output_dir / OVERLAY_MANIFEST_FILENAME, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize pinned upstream openpi tree plus repo-local recap overlay"
    )
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--pinned-commit", default=DEFAULT_PINNED_COMMIT)
    parser.add_argument("--overlay-root", default=str(DEFAULT_OVERLAY_ROOT))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_tree = _resolve_path(str(args.source_tree))
    pinned_commit = str(args.pinned_commit)
    overlay_root = _resolve_path(str(args.overlay_root))
    output_dir = _resolve_path(str(args.output_dir))
    manifest = materialize_overlay(
        source_tree=source_tree,
        pinned_commit=pinned_commit,
        overlay_root=overlay_root,
        output_dir=output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
