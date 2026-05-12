"""R2.2 ckpt config swap — implements brief §5.3 verbatim.

Filesystem note (A3):
  The artifact tree (agent/artifacts/recap_substrate_recovery/) is on NTFS3
  (ext4-on-Linux pass-through semantics may differ from canonical ext4).
  NTFS3 does NOT provide:
    - atomic rename across directories
    - sub-second mtime resolution (we therefore avoid mtime-based tiebreaks; A4)
    - fsync semantics matching ext4
  Consequence: we use ``shutil.copy2`` via ``shutil.copytree`` for the swap
  (POSIX-compatible), do not assume atomic operations, and use the
  ``cell_result.json`` sentinel for completion-detection (B-IND3).

Brief §5.3 mechanic (verbatim — A1):
  1. Assert REQUIRED_CKPT_FILES present in src_ckpt.abs_path.
  2. Assert ALLOW_LIST_TO_OVERRIDE files present in raw_hf_snapshot.
  3. Snapshot pre-swap sha256 of EVERY file under src_ckpt.abs_path
     (recursive walk via _walk_recursive_sha_table).
  4. shutil.copytree(src_ckpt.abs_path, out_dir / "ckpt_swapped",
                     symlinks=False, copy_function=shutil.copy2,
                     dirs_exist_ok=False)
  5. for name in ALLOW_LIST_TO_OVERRIDE:
         shutil.copy2(raw_hf_snapshot / name, out_dir / "ckpt_swapped" / name)
  6. SHA pair-audit (three blocks):
     (a) WEIGHT_GLOB shards (model-*.safetensors):
         sha(src/<shard>) == sha(swap/<shard>) for each
     (b) PROTECTED_FILES (statistics.json, embodiment_id.json,
         model.safetensors.index.json): sha(src/X) == sha(swap/X)
     (c) ALLOW_LIST_TO_OVERRIDE: sha(swap/<file>) == sha(raw_hf/<file>)
  7. Re-walk src_ckpt; assert post-swap sha == pre-swap sha for EVERY file
     → CkptSrcMutatedDuringSwap on mismatch.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterator, Sequence

from work.recap.r1_repro.protocol import (
    RAW_HF_SNAPSHOT_ROOT,
    REPO_ROOT,
)
from work.recap.r2_authentic_eval.inventory import TrainedCheckpoint

# ---------------------------------------------------------------------------
# Module-top constants (brief §5.3 verbatim — A1)
# ---------------------------------------------------------------------------

ALLOW_LIST_TO_OVERRIDE: tuple[str, ...] = ("config.json", "processor_config.json")
PROTECTED_FILES: tuple[str, ...] = (
    "statistics.json",
    "embodiment_id.json",
    "model.safetensors.index.json",
)
WEIGHT_GLOB: str = "model-*.safetensors"
REQUIRED_CKPT_FILES: tuple[str, ...] = ALLOW_LIST_TO_OVERRIDE + PROTECTED_FILES

SWAP_ROOT_DEFAULT: Path = REPO_ROOT / "agent/artifacts/recap_substrate_recovery/r2_2_decomposition"
SWAP_DIR_NAME: str = "ckpt_swapped"
SWAP_PROVENANCE_FILENAME: str = "_swap_provenance.json"
LINK_STRATEGY: str = "copytree"  # documented constant; no hardlink path (A3)

_ARTIFACT_TREE_FS_NOTE: str = (
    "swap dir lives under agent/artifacts/recap_substrate_recovery/ (NTFS3); "
    "no atomic-rename guarantees; copytree-only swap mechanic."
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CkptSwapError(RuntimeError):
    """Base class for checkpoint swap errors."""


class CkptByteIdentityViolation(CkptSwapError):
    """SHA pair-audit mismatch: swap dir bytes differ from expected source."""


class CkptSourceMissingArtifact(CkptSwapError):
    """Step 1 failure: required file absent from source checkpoint dir."""


class CkptRawHfMissingArtifact(CkptSwapError):
    """Step 2 failure: required ALLOW_LIST file absent from raw HF snapshot."""


class CkptSrcMutatedDuringSwap(CkptSwapError):
    """Step 7 failure: source checkpoint was mutated during swap operation."""


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConfigSwapResult:
    """Frozen audit record returned by ``materialise_swap_ckpt``.

    Written to ``{swap_dir}/_swap_provenance.json`` as the completion sentinel
    (B-IND3).  All Path fields are serialised as strings in the JSON output.
    """

    source_ckpt_root: Path
    raw_hf_root: Path
    swap_dir: Path
    materialised_at_utc: str
    link_strategy: str  # always LINK_STRATEGY = "copytree"
    src_pre_swap_sha_table: dict[str, str]  # relpath → sha256, every file under src
    swap_post_swap_sha_table: dict[str, str]  # same shape, every file under swap_dir
    weight_glob_sha_pairs: dict[str, dict[str, str]]
    # relpath → {"src_sha": ..., "swap_sha": ...} for every model-*.safetensors match
    protected_files_sha_pairs: dict[str, dict[str, str]]
    # name → {"src_sha": ..., "swap_sha": ...} for each PROTECTED_FILE
    allow_list_files_sha_pairs: dict[str, dict[str, str]]
    # name → {"src_sha": ..., "raw_hf_sha": ..., "swap_sha": ...} for each ALLOW_LIST file
    dry_run_wall_clock_seconds: float | None = None  # V3-FIX-3: only set by dry-run hook


@dataclasses.dataclass(frozen=True, slots=True)
class FieldChange:
    """One config field changed by ``materialise_field_targeted_swap``."""

    path: str
    before: Any
    after: Any


@dataclasses.dataclass(frozen=True, slots=True)
class FieldSwapResult:
    """Audit record returned by ``materialise_field_targeted_swap``."""

    source_ckpt_root: Path
    target_ckpt_root: Path
    swap_dir: Path
    materialised_at_utc: str
    field_paths: tuple[str, ...]
    field_overrides: dict[str, dict[str, Any]]
    changes: tuple[FieldChange, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def materialise_swap_ckpt(
    src_ckpt: TrainedCheckpoint,
    raw_hf_root: Path = RAW_HF_SNAPSHOT_ROOT,
    swap_root: Path = SWAP_ROOT_DEFAULT,
    *,
    utc_slug: str | None = None,
) -> ConfigSwapResult:
    """Materialise a config-swapped copy of *src_ckpt* — brief §5.3 verbatim.

    Parameters
    ----------
    src_ckpt:
        Trained checkpoint dataclass from ``inventory.py``.  Its ``.abs_path``
        attribute is the source directory root (the full checkpoint directory).
    raw_hf_root:
        Root of the raw HF snapshot supplying ALLOW_LIST_TO_OVERRIDE files.
        Defaults to ``RAW_HF_SNAPSHOT_ROOT`` from ``r1_repro.protocol``.
    swap_root:
        Parent directory under which the timestamped swap dir is created.
        Must be inside the whitelisted artifact subtree (A3).
    utc_slug:
        Override the UTC timestamp slug (useful for testing); otherwise uses
        ``datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')``.

    Returns
    -------
    ConfigSwapResult
        Frozen audit record.  Also written to
        ``{swap_dir}/_swap_provenance.json`` as a completion sentinel.

    Raises
    ------
    CkptSourceMissingArtifact
        If any REQUIRED_CKPT_FILES are absent from ``src_ckpt.abs_path``.
    CkptRawHfMissingArtifact
        If any ALLOW_LIST_TO_OVERRIDE files are absent from ``raw_hf_root``.
    CkptByteIdentityViolation
        If any SHA pair-audit in step 6 fails.
    CkptSrcMutatedDuringSwap
        If post-swap re-hash of source differs from pre-swap snapshot.
    """
    source_ckpt_root: Path = src_ckpt.abs_path

    # Step 1: assert REQUIRED_CKPT_FILES present in source checkpoint
    for name in REQUIRED_CKPT_FILES:
        if not (source_ckpt_root / name).exists():
            raise CkptSourceMissingArtifact(
                f"Required file missing from source checkpoint: {name!r} "
                f"(expected at {source_ckpt_root / name})"
            )

    # Step 2: assert ALLOW_LIST_TO_OVERRIDE files present in raw_hf_root
    for name in ALLOW_LIST_TO_OVERRIDE:
        if not (raw_hf_root / name).exists():
            raise CkptRawHfMissingArtifact(
                f"ALLOW_LIST file missing from raw HF snapshot: {name!r} "
                f"(expected at {raw_hf_root / name})"
            )

    # Step 3: snapshot pre-swap sha256 of EVERY file under source (recursive)
    src_pre_swap_sha_table = _walk_recursive_sha_table(source_ckpt_root)

    # Step 4: deep-copy source → swap_dir via copytree (no hardlinks — A3)
    slug = utc_slug or _utc_slug()
    swap_dir = swap_root / slug / SWAP_DIR_NAME
    shutil.copytree(
        source_ckpt_root,
        swap_dir,
        symlinks=False,
        copy_function=shutil.copy2,
        dirs_exist_ok=False,
    )

    # Step 5: overwrite ALLOW_LIST_TO_OVERRIDE files from raw_hf_root
    for name in ALLOW_LIST_TO_OVERRIDE:
        shutil.copy2(raw_hf_root / name, swap_dir / name)

    # Step 6: SHA pair-audit (three blocks)
    weight_glob_sha_pairs: dict[str, dict[str, str]] = {}
    protected_files_sha_pairs: dict[str, dict[str, str]] = {}
    allow_list_files_sha_pairs: dict[str, dict[str, str]] = {}

    # 6a: WEIGHT_GLOB shards — sha(src/<shard>) == sha(swap/<shard>)
    for swap_shard in sorted(swap_dir.glob(WEIGHT_GLOB)):
        rel = swap_shard.name
        src_sha = _sha256_hex(source_ckpt_root / rel)
        swap_sha = _sha256_hex(swap_dir / rel)
        if src_sha != swap_sha:
            raise CkptByteIdentityViolation(
                f"Shard byte-identity violation: {rel!r} "
                f"src={src_sha[:12]} swap={swap_sha[:12]}"
            )
        weight_glob_sha_pairs[rel] = {"src_sha": src_sha, "swap_sha": swap_sha}

    # 6b: PROTECTED_FILES — sha(src/X) == sha(swap/X)
    for name in PROTECTED_FILES:
        src_sha = _sha256_hex(source_ckpt_root / name)
        swap_sha = _sha256_hex(swap_dir / name)
        if src_sha != swap_sha:
            raise CkptByteIdentityViolation(
                f"Protected file byte-identity violation: {name!r} "
                f"src={src_sha[:12]} swap={swap_sha[:12]}"
            )
        protected_files_sha_pairs[name] = {"src_sha": src_sha, "swap_sha": swap_sha}

    # 6c: ALLOW_LIST_TO_OVERRIDE — sha(swap/<file>) == sha(raw_hf/<file>)
    for name in ALLOW_LIST_TO_OVERRIDE:
        raw_sha = _sha256_hex(raw_hf_root / name)
        swap_sha = _sha256_hex(swap_dir / name)
        src_sha = _sha256_hex(source_ckpt_root / name)
        if raw_sha != swap_sha:
            raise CkptByteIdentityViolation(
                f"Allow-list file does not match raw HF: {name!r} "
                f"raw={raw_sha[:12]} swap={swap_sha[:12]}"
            )
        allow_list_files_sha_pairs[name] = {
            "src_sha": src_sha,
            "raw_hf_sha": raw_sha,
            "swap_sha": swap_sha,
        }

    # Step 7: re-hash source → CkptSrcMutatedDuringSwap on any mismatch
    post_swap_src = _walk_recursive_sha_table(source_ckpt_root)
    for relpath, pre_sha in src_pre_swap_sha_table.items():
        post_sha = post_swap_src.get(relpath)
        if post_sha != pre_sha:
            raise CkptSrcMutatedDuringSwap(
                f"Source ckpt mutated during swap: {relpath!r} "
                f"pre={pre_sha[:12]} post={post_sha!r}"
            )
    for relpath in post_swap_src:
        if relpath not in src_pre_swap_sha_table:
            raise CkptSrcMutatedDuringSwap(
                f"Source ckpt gained new file during swap: {relpath!r}"
            )

    # Step 8: build post-swap sha table for the swap dir
    swap_post_swap_sha_table = _walk_recursive_sha_table(swap_dir)

    # Step 9: build ConfigSwapResult; write _swap_provenance.json LAST (sentinel)
    result = ConfigSwapResult(
        source_ckpt_root=source_ckpt_root,
        raw_hf_root=raw_hf_root,
        swap_dir=swap_dir,
        materialised_at_utc=slug,
        link_strategy=LINK_STRATEGY,
        src_pre_swap_sha_table=src_pre_swap_sha_table,
        swap_post_swap_sha_table=swap_post_swap_sha_table,
        weight_glob_sha_pairs=weight_glob_sha_pairs,
        protected_files_sha_pairs=protected_files_sha_pairs,
        allow_list_files_sha_pairs=allow_list_files_sha_pairs,
    )
    _write_provenance(result, swap_dir)

    return result


def materialise_field_targeted_swap(
    source_ckpt: Path,
    target_ckpt: Path,
    field_overrides: Sequence[str] | dict[str, dict[str, Any]],
    *,
    swap_root: Path,
) -> FieldSwapResult:
    """Copy target ckpt and replace listed config fields; path lists read values from source."""
    source_root = Path(source_ckpt)
    target_root = Path(target_ckpt)
    if not field_overrides:
        raise ValueError("field_overrides must not be empty")

    override_items, override_payload = _normalise_field_overrides(source_root, field_overrides)
    touched_files = tuple(sorted({filename for filename, _, _ in override_items}))
    target_configs = {f: _load_config_json(target_root / f) for f in touched_files}

    pending_changes: list[FieldChange] = []
    for filename, dotted_path, after in override_items:
        before = _get_dotted_path(target_configs[filename], dotted_path)
        pending_changes.append(FieldChange(f"{filename}:{dotted_path}", before, after))

    swap_root_resolved = Path(swap_root).resolve()
    source_resolved = source_root.resolve()
    target_resolved = target_root.resolve()
    if swap_root_resolved == source_resolved or swap_root_resolved == target_resolved:
        raise ValueError("swap_root must be separate from source_ckpt and target_ckpt")

    slug = _utc_slug()
    swap_dir = Path(swap_root) / slug / "ckpt_field_swapped"
    shutil.copytree(target_root, swap_dir, symlinks=False, copy_function=shutil.copy2)
    if not swap_dir.resolve().is_relative_to(swap_root_resolved):
        shutil.rmtree(swap_dir.parent, ignore_errors=True)
        raise ValueError(f"swap_dir escaped swap_root: {swap_dir}")

    for change, (filename, dotted_path, _) in zip(pending_changes, override_items):
        _set_dotted_path(target_configs[filename], dotted_path, change.after)
    for filename in touched_files:
        (swap_dir / filename).write_text(
            json.dumps(target_configs[filename], indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    return FieldSwapResult(
        source_ckpt_root=source_root,
        target_ckpt_root=target_root,
        swap_dir=swap_dir,
        materialised_at_utc=slug,
        field_paths=tuple(change.path for change in pending_changes),
        field_overrides=override_payload,
        changes=tuple(pending_changes),
    )


_CONFIG_FIELD_FILES: tuple[str, ...] = ("config.json", "processor_config.json")
_MISSING = object()


def _normalise_field_path(path: str) -> tuple[str, str]:
    if ":" in path:
        filename, dotted_path = path.split(":", 1)
    else:
        filename, dotted_path = "config.json", path
    if filename not in _CONFIG_FIELD_FILES:
        raise ValueError(f"Unsupported config file for field-targeted swap: {filename!r}")
    if not dotted_path:
        raise ValueError("dotted field path must not be empty")
    return filename, dotted_path


def _normalise_field_overrides(
    source_root: Path,
    field_overrides: Sequence[str] | dict[str, dict[str, Any]],
) -> tuple[list[tuple[str, str, Any]], dict[str, dict[str, Any]]]:
    if isinstance(field_overrides, dict):
        items: list[tuple[str, str, Any]] = []
        payload: dict[str, dict[str, Any]] = {}
        for filename, path_values in field_overrides.items():
            if filename not in _CONFIG_FIELD_FILES:
                raise ValueError(f"Unsupported config file for field-targeted swap: {filename!r}")
            if not path_values:
                raise ValueError(f"field_overrides[{filename!r}] must not be empty")
            payload[filename] = {str(path): value for path, value in path_values.items()}
            for dotted_path, value in payload[filename].items():
                if not dotted_path:
                    raise ValueError("dotted field path must not be empty")
                items.append((filename, dotted_path, value))
        return items, payload

    source_configs: dict[str, dict[str, Any]] = {}
    items = []
    payload = {}
    for raw_path in field_overrides:
        filename, dotted_path = _normalise_field_path(raw_path)
        if filename not in source_configs:
            source_configs[filename] = _load_config_json(source_root / filename)
        value = _get_dotted_path(source_configs[filename], dotted_path)
        payload.setdefault(filename, {})[dotted_path] = value
        items.append((filename, dotted_path, value))
    return items, payload


def _load_config_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected object JSON at {path}")
    return payload


def _get_dotted_path(d: dict[str, Any], path: str) -> Any:
    node: Any = d
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _set_dotted_path(d: dict[str, Any], path: str, value: Any) -> None:
    node: Any = d
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    leaf = parts[-1]
    if not isinstance(node, dict) or leaf not in node:
        raise KeyError(path)
    node[leaf] = value


def _walk_leaves(d: dict[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for key in sorted(d):
        path = f"{prefix}.{key}" if prefix else str(key)
        value = d[key]
        if isinstance(value, dict):
            yield from _walk_leaves(value, path)
        else:
            yield path, value


def _compare_outside_paths(
    source_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    allowed_paths: set[str],
) -> list[tuple[str, Any, Any]]:
    source_flat = dict(_walk_leaves(source_cfg))
    target_flat = dict(_walk_leaves(target_cfg))
    outside: list[tuple[str, Any, Any]] = []
    for path in sorted(set(source_flat) | set(target_flat)):
        if path in allowed_paths:
            continue
        source_value = source_flat.get(path, _MISSING)
        target_value = target_flat.get(path, _MISSING)
        if source_value != target_value:
            outside.append((path, source_value, target_value))
    return outside


def _walk_recursive_sha_table(root: Path) -> dict[str, str]:
    """Walk ``root`` recursively. Returns ``{str(path.relative_to(root)): sha256_hex}``
    for every file (not directory) under ``root``.

    Walks `root` recursively. Follows symlinks (consistent with
    `copytree(symlinks=False)` semantics). Hash represents target's content
    bytes, not symlink target path. Symlinks pointing outside root are
    dereferenced and content hashed. Broken symlinks raise FileNotFoundError.

    Behavior unchanged from v2; this docstring is informational (V3-FIX-4).
    """
    table: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        # Explicit broken-symlink detection (spec: broken symlinks raise FileNotFoundError)
        if path.is_symlink() and not path.exists():
            raise FileNotFoundError(f"Broken symlink encountered during sha walk: {path}")
        if path.is_file():  # Path.is_file() follows symlinks by default
            table[str(path.relative_to(root))] = _sha256_hex(path)
    return table


def _sha256_hex(path: Path) -> str:
    """Return sha256 hex digest of the file at *path* (follows symlinks)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_slug() -> str:
    """Return a UTC timestamp slug: ``YYYYMMDDTHHMMSSz``."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_provenance(result: ConfigSwapResult, swap_dir: Path) -> None:
    """Serialise *result* to ``{swap_dir}/_swap_provenance.json`` (sentinel, written LAST)."""
    payload: dict[str, Any] = {
        "source_ckpt_root": str(result.source_ckpt_root),
        "raw_hf_root": str(result.raw_hf_root),
        "swap_dir": str(result.swap_dir),
        "materialised_at_utc": result.materialised_at_utc,
        "link_strategy": result.link_strategy,
        "src_pre_swap_sha_table": result.src_pre_swap_sha_table,
        "swap_post_swap_sha_table": result.swap_post_swap_sha_table,
        "weight_glob_sha_pairs": result.weight_glob_sha_pairs,
        "protected_files_sha_pairs": result.protected_files_sha_pairs,
        "allow_list_files_sha_pairs": result.allow_list_files_sha_pairs,
    }
    if result.dry_run_wall_clock_seconds is not None:
        payload["dry_run_wall_clock_seconds"] = result.dry_run_wall_clock_seconds
    (swap_dir / SWAP_PROVENANCE_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


__all__ = [
    "ALLOW_LIST_TO_OVERRIDE",
    "PROTECTED_FILES",
    "WEIGHT_GLOB",
    "REQUIRED_CKPT_FILES",
    "SWAP_ROOT_DEFAULT",
    "SWAP_DIR_NAME",
    "SWAP_PROVENANCE_FILENAME",
    "LINK_STRATEGY",
    "_ARTIFACT_TREE_FS_NOTE",
    "CkptSwapError",
    "CkptByteIdentityViolation",
    "CkptSourceMissingArtifact",
    "CkptRawHfMissingArtifact",
    "CkptSrcMutatedDuringSwap",
    "ConfigSwapResult",
    "FieldChange",
    "FieldSwapResult",
    "materialise_swap_ckpt",
    "materialise_field_targeted_swap",
    "_set_dotted_path",
    "_get_dotted_path",
    "_walk_leaves",
    "_compare_outside_paths",
    "_walk_recursive_sha_table",
    "_sha256_hex",
    "_utc_slug",
]
