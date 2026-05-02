from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import cast

from work.openpi.dataloader import read_json


def expected_stock_checkpoint() -> str:
    from work.openpi.serve.provenance import EXPECTED_CHECKPOINT

    return EXPECTED_CHECKPOINT


def is_remote_uri(raw: str) -> bool:
    candidate = str(raw).strip()
    return "://" in candidate or candidate.startswith("gs://")


def normalize_checkpoint_ref(raw_checkpoint_dir: str) -> str:
    checkpoint_dir = raw_checkpoint_dir.strip()
    if not checkpoint_dir:
        raise ValueError("checkpoint reference must be non-empty")
    if is_remote_uri(checkpoint_dir):
        return checkpoint_dir
    return str(Path(checkpoint_dir).expanduser().resolve())


def provenance_search_dirs(source_dir: Path) -> tuple[Path, ...]:
    resolved = source_dir.resolve()
    candidates = (
        resolved,
        resolved / "rollout_eval_v21_input",
        resolved / "rollout_eval_v2_input",
        resolved.parent,
        resolved.parent / "rollout_eval_v21_input",
        resolved.parent / "rollout_eval_v2_input",
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return tuple(deduped)


def load_provenance_pair(
    source_dir: Path,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    train_manifest: dict[str, object] | None = None
    checkpoint_provenance: dict[str, object] | None = None
    for candidate in provenance_search_dirs(source_dir):
        if train_manifest is None and (candidate / "train_manifest.json").is_file():
            train_manifest = read_json(candidate / "train_manifest.json")
        if (
            checkpoint_provenance is None
            and (candidate / "checkpoint_provenance.json").is_file()
        ):
            checkpoint_provenance = read_json(candidate / "checkpoint_provenance.json")
    return train_manifest, checkpoint_provenance


def load_checkpoint_provenance_pair(
    *, checkpoint_ref: str, raw_checkpoint_dir: str | None
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    candidates: list[Path] = []
    raw_candidate = str(raw_checkpoint_dir or "").strip()
    if raw_candidate and not is_remote_uri(raw_candidate):
        candidates.append(Path(raw_candidate).resolve())
    if not is_remote_uri(checkpoint_ref):
        candidates.append(Path(checkpoint_ref).resolve())
    checked: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in checked:
            continue
        checked.add(candidate_key)
        if candidate.exists() and candidate.is_dir():
            return load_provenance_pair(candidate)
    return None, None


def candidate_rollout_source_dirs(
    *,
    checkpoint_ref: str,
    raw_checkpoint_dir: str | None,
    output_dir: Path,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    raw_candidate = str(raw_checkpoint_dir or "").strip()
    if raw_candidate and not is_remote_uri(raw_candidate):
        checkpoint_path = Path(raw_candidate).resolve()
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            candidates.extend(
                (
                    checkpoint_path / "rollout_eval_v21_input",
                    checkpoint_path / "rollout_eval_v2_input",
                    checkpoint_path.parent / "rollout_eval_v21_input",
                    checkpoint_path.parent / "rollout_eval_v2_input",
                    checkpoint_path,
                    checkpoint_path.parent,
                )
            )
    if not is_remote_uri(checkpoint_ref):
        checkpoint_path = Path(checkpoint_ref).resolve()
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            candidates.extend(
                (
                    checkpoint_path / "rollout_eval_v21_input",
                    checkpoint_path / "rollout_eval_v2_input",
                    checkpoint_path.parent / "rollout_eval_v21_input",
                    checkpoint_path.parent / "rollout_eval_v2_input",
                    checkpoint_path,
                    checkpoint_path.parent,
                )
            )
    candidates.extend((output_dir / "_staging", output_dir))
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return tuple(deduped)


def resolve_rollout_source_dir(
    *,
    checkpoint_ref: str,
    raw_checkpoint_dir: str | None,
    output_dir: Path,
    per_episode_name: str,
) -> Path:
    checked: list[str] = []
    for candidate in candidate_rollout_source_dirs(
        checkpoint_ref=checkpoint_ref,
        raw_checkpoint_dir=raw_checkpoint_dir,
        output_dir=output_dir,
    ):
        checked.append(str(candidate))
        if (candidate / per_episode_name).is_file():
            return candidate
    raise ValueError(
        "missing fresh rollout input bundle; expected "
        + per_episode_name
        + " under one of: "
        + ", ".join(checked)
    )


def is_stock_variant(variant: str, *, stock_variants: frozenset[str]) -> bool:
    return variant.strip() in stock_variants


def resolve_servable_checkpoint_ref(
    *, checkpoint_ref: str, variant: str, stock_variants: frozenset[str]
) -> tuple[str, str]:
    if is_remote_uri(checkpoint_ref):
        return checkpoint_ref, "explicit_checkpoint_ref"
    checkpoint_path = Path(checkpoint_ref)
    if (checkpoint_path / "params" / "_METADATA").is_file():
        return str(checkpoint_path), "local_orbax_checkpoint"
    train_manifest, checkpoint_provenance = load_provenance_pair(checkpoint_path)
    if train_manifest is None and checkpoint_provenance is None:
        rollout_input_dir = checkpoint_path / "rollout_eval_v2_input"
        if rollout_input_dir.is_dir():
            train_manifest, checkpoint_provenance = load_provenance_pair(
                rollout_input_dir
            )
    source_payload = checkpoint_provenance or train_manifest
    if source_payload is None:
        raise ValueError(
            "local checkpoint bundle is not servable: missing params/_METADATA and missing train_manifest/checkpoint_provenance"
        )
    base_checkpoint_id = str(source_payload.get("base_checkpoint_id", "")).strip()
    if base_checkpoint_id != "pi05_libero_anchor":
        raise ValueError(
            "local checkpoint bundle is not servable: missing params/_METADATA and unsupported base_checkpoint_id="
            + f"{base_checkpoint_id!r}"
        )
    if not is_stock_variant(variant, stock_variants=stock_variants):
        raise ValueError(
            "non-stock variant requires a real serveable checkpoint: "
            + f"variant={variant!r} checkpoint_ref={checkpoint_ref!r} is metadata-only "
            + "(missing params/_METADATA); fallback to stock anchor is forbidden"
        )
    return expected_stock_checkpoint(), "metadata_only_bundle_fallback_to_stock_anchor"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint_instance_binding(
    checkpoint_ref: str,
    *,
    key_files: tuple[Path, ...],
    schema_version: str,
) -> str:
    normalized_checkpoint_ref = str(checkpoint_ref).strip()
    if not normalized_checkpoint_ref:
        raise ValueError("checkpoint_ref must be non-empty")
    if is_remote_uri(normalized_checkpoint_ref):
        return f"remote_ref::{normalized_checkpoint_ref}"
    checkpoint_path = Path(normalized_checkpoint_ref).expanduser().resolve()
    if not checkpoint_path.is_dir():
        return f"path_ref::{normalized_checkpoint_ref}"
    file_digests: dict[str, str] = {}
    for rel_path in key_files:
        candidate = checkpoint_path / rel_path
        if candidate.is_file():
            file_digests[str(rel_path)] = sha256_file(candidate)
    if not file_digests:
        return f"path_ref::{checkpoint_path}"
    payload = {"schema_version": schema_version, "file_digests": file_digests}
    return "local_keyfiles::" + hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


def require_mapping(raw: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)
