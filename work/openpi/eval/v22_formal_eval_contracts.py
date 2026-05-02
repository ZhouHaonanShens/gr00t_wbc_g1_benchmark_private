from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HASH_LOCK_SCHEMA = "v22_preregistration_hash_lock_v1"
DEFAULT_PAIRING_TASK = "all_tasks_round_robin_episode_index_modulo_10"
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 20260427


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload at {path} must be an object")
    return {str(key): value for key, value in payload.items()}


def _sidecar_sha256(path: Path) -> str | None:
    sidecar = path.with_name(f"{path.name}.sha256")
    if not sidecar.is_file():
        return None
    return sidecar.read_text(encoding="utf-8").strip()


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _string_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in _sequence(value))


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def protocol_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return _canonical(left) == _canonical(right)


@dataclass(frozen=True)
class PairingRule:
    name: str
    modulo: int
    source_task: str

    def key_for_episode(self, episode_index: int) -> int:
        return int(episode_index) % self.modulo


@dataclass(frozen=True)
class PreregHashLock:
    path: Path
    sha256: str
    expected_sha256: str | None
    schema_version: str
    run_id: str
    selected_protocol: Mapping[str, object]
    variants: tuple[str, ...]
    n_per_variant: int
    raw: Mapping[str, object]

    @property
    def suite(self) -> str:
        return str(self.selected_protocol.get("suite") or "")

    @property
    def budget(self) -> float:
        return float(self.selected_protocol.get("budget") or 0.0)

    @property
    def cell_id(self) -> str:
        return str(self.selected_protocol.get("cell_id") or "")

    @property
    def step_cap(self) -> int:
        return int(self.selected_protocol.get("step_cap") or 0)

    @property
    def max_steps_full(self) -> int:
        return int(self.selected_protocol.get("max_steps_full") or 0)

    @property
    def tasks(self) -> tuple[str, ...]:
        return _string_tuple(self.selected_protocol.get("tasks"))

    @property
    def pairing_rule(self) -> PairingRule:
        return pairing_rule_from_tasks(self.tasks)


@dataclass(frozen=True)
class VariantAuthorityManifest:
    path: Path
    sha256: str
    schema_version: str
    run_id: str
    formal_eval_allowed: bool
    selected_protocol_compatible: bool | None
    selected_protocol: Mapping[str, object]
    variants_loaded: tuple[str, ...]
    hash_lock_sha256: str | None
    raw: Mapping[str, object]

    def checkpoint_path_for(self, variant: str) -> Path | None:
        variant_key = str(variant)
        for entry in self._variant_entries():
            if str(entry.get("variant") or entry.get("variant_code") or "") != variant_key:
                continue
            path = _checkpoint_path_from_entry(entry)
            if path is not None:
                return path
        variants = _mapping(self.raw.get("variants"))
        variant_payload = _mapping(variants.get(variant_key))
        path = _checkpoint_path_from_entry(variant_payload)
        if path is not None:
            return path
        if variant_key == "A":
            source = _mapping(self.raw.get("authority_source"))
            source_path = source.get("path")
            if source_path:
                return _repo_path(str(source_path))
        return None

    def _variant_entries(self) -> tuple[Mapping[str, object], ...]:
        entries: list[Mapping[str, object]] = []
        for key in ("variant_authorities", "variant_manifests", "variant_entries"):
            for item in _sequence(self.raw.get(key)):
                if isinstance(item, Mapping):
                    entries.append({str(name): value for name, value in item.items()})
        return tuple(entries)


def _checkpoint_path_from_entry(entry: Mapping[str, object]) -> Path | None:
    for key in (
        "checkpoint_path",
        "checkpoint_dir",
        "local_checkpoint_path",
        "local_resolved_path",
        "warm_start_checkpoint",
    ):
        value = entry.get(key)
        if value:
            return _repo_path(str(value))
    checkpoint = _mapping(entry.get("checkpoint"))
    for key in ("path", "dir", "local_path", "local_resolved_path"):
        value = checkpoint.get(key)
        if value:
            return _repo_path(str(value))
    return None


def load_prereg_hash_lock(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> PreregHashLock:
    resolved = _repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"BLOCK_HASH_LOCK_MISSING:{resolved}")
    actual_sha = sha256_file(resolved)
    expected = expected_sha256 or _sidecar_sha256(resolved)
    if expected and actual_sha != expected:
        raise ValueError(
            f"BLOCK_HASH_LOCK_SHA_MISMATCH expected={expected} actual={actual_sha}"
        )
    payload = read_json_object(resolved)
    selected_protocol = _mapping(payload.get("selected_protocol"))
    return PreregHashLock(
        path=resolved,
        sha256=actual_sha,
        expected_sha256=expected,
        schema_version=str(payload.get("schema_version") or ""),
        run_id=str(payload.get("run_id") or ""),
        selected_protocol=selected_protocol,
        variants=_string_tuple(payload.get("variants")),
        n_per_variant=int(payload.get("n_per_variant") or 0),
        raw=payload,
    )


def validate_hash_lock(lock: PreregHashLock) -> list[str]:
    reasons: list[str] = []
    if lock.schema_version != HASH_LOCK_SCHEMA:
        reasons.append("BLOCK_HASH_LOCK_SCHEMA_MISMATCH")
    if lock.variants != ("A", "B", "C", "X"):
        reasons.append("BLOCK_HASH_LOCK_VARIANTS_MISMATCH")
    if lock.n_per_variant != 192:
        reasons.append("BLOCK_HASH_LOCK_N_PER_VARIANT_MISMATCH")
    protocol = lock.selected_protocol
    if lock.suite != "libero_spatial":
        reasons.append("BLOCK_HASH_LOCK_SUITE_MISMATCH")
    if lock.step_cap != 110:
        reasons.append("BLOCK_HASH_LOCK_STEP_CAP_MISMATCH")
    if lock.tasks != (DEFAULT_PAIRING_TASK,):
        reasons.append("BLOCK_HASH_LOCK_TASKS_MISMATCH")
    if protocol.get("budget") != 0.5:
        reasons.append("BLOCK_HASH_LOCK_BUDGET_MISMATCH")
    return reasons


def load_variant_authority_manifest(path: str | Path) -> VariantAuthorityManifest:
    resolved = _repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"BLOCK_VARIANT_AUTHORITY_MANIFEST_MISSING:{resolved}")
    payload = read_json_object(resolved)
    hash_lock = _mapping(payload.get("hash_lock"))
    selected_protocol = _mapping(payload.get("selected_protocol") or hash_lock.get("selected_protocol"))
    variants_loaded = _string_tuple(
        payload.get("variants_loaded")
        or payload.get("variants")
        or payload.get("variant_codes")
    )
    return VariantAuthorityManifest(
        path=resolved,
        sha256=sha256_file(resolved),
        schema_version=str(payload.get("schema_version") or ""),
        run_id=str(payload.get("run_id") or ""),
        formal_eval_allowed=bool(payload.get("formal_eval_allowed")),
        selected_protocol_compatible=(
            bool(payload.get("selected_protocol_compatible"))
            if payload.get("selected_protocol_compatible") is not None
            else None
        ),
        selected_protocol=selected_protocol,
        variants_loaded=variants_loaded,
        hash_lock_sha256=(
            str(payload.get("hash_lock_sha256") or hash_lock.get("sha256"))
            if payload.get("hash_lock_sha256") or hash_lock.get("sha256")
            else None
        ),
        raw=payload,
    )


def validate_variant_authority_manifest(
    manifest: VariantAuthorityManifest,
    lock: PreregHashLock,
) -> list[str]:
    reasons: list[str] = []
    if not manifest.formal_eval_allowed:
        reasons.append("BLOCK_VARIANT_AUTHORITY_MANIFEST_NOT_PASSING")
    if manifest.selected_protocol_compatible is False:
        reasons.append("BLOCK_VARIANT_AUTHORITY_PROTOCOL_INCOMPATIBLE")
    if manifest.hash_lock_sha256 and manifest.hash_lock_sha256 != lock.sha256:
        reasons.append("BLOCK_VARIANT_AUTHORITY_HASH_LOCK_MISMATCH")
    if manifest.selected_protocol and not protocol_equal(manifest.selected_protocol, lock.selected_protocol):
        reasons.append("BLOCK_VARIANT_AUTHORITY_PROTOCOL_MISMATCH")
    if manifest.variants_loaded and set(manifest.variants_loaded) != set(lock.variants):
        reasons.append("BLOCK_VARIANT_AUTHORITY_VARIANTS_MISMATCH")
    return reasons


def pairing_rule_from_tasks(tasks: Sequence[str]) -> PairingRule:
    for task in tasks:
        text = str(task)
        match = re.search(r"episode_index_modulo_(\d+)", text)
        if match:
            return PairingRule(
                name=f"episode_index_modulo_{match.group(1)}",
                modulo=int(match.group(1)),
                source_task=text,
            )
    raise ValueError("BLOCK_PAIRING_RULE_UNSUPPORTED")


def paired_bootstrap_ci(
    baseline_successes: Sequence[object],
    treatment_successes: Sequence[object],
    *,
    episode_indices: Sequence[int] | None = None,
    tasks: Sequence[str] = (DEFAULT_PAIRING_TASK,),
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, object]:
    if len(baseline_successes) != len(treatment_successes):
        raise ValueError("paired_success_arrays_length_mismatch")
    if n_resamples < DEFAULT_BOOTSTRAP_RESAMPLES:
        raise ValueError("paired_bootstrap_requires_at_least_10000_resamples")
    if not baseline_successes:
        raise ValueError("paired_bootstrap_empty_input")
    indices = tuple(range(len(baseline_successes))) if episode_indices is None else tuple(episode_indices)
    if len(indices) != len(baseline_successes):
        raise ValueError("episode_indices_length_mismatch")

    rule = pairing_rule_from_tasks(tasks)
    groups: dict[int, list[float]] = {}
    deltas: list[float] = []
    for index, base_value, treatment_value in zip(
        indices,
        baseline_successes,
        treatment_successes,
        strict=True,
    ):
        delta = _success_float(treatment_value) - _success_float(base_value)
        groups.setdefault(rule.key_for_episode(index), []).append(delta)
        deltas.append(delta)

    keys = sorted(groups)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(int(n_resamples)):
        sample_total = 0.0
        sample_count = 0
        for _key_index in keys:
            sampled_key = rng.choice(keys)
            values = groups[sampled_key]
            sample_total += sum(values)
            sample_count += len(values)
        estimates.append(sample_total / sample_count)
    estimates.sort()
    alpha = 1.0 - confidence
    lower = estimates[int((alpha / 2.0) * (len(estimates) - 1))]
    upper = estimates[int((1.0 - alpha / 2.0) * (len(estimates) - 1))]
    observed_delta = sum(deltas) / len(deltas)
    return {
        "schema_version": "v22_paired_bootstrap_ci_v1",
        "paired_by": rule.name,
        "pairing_task": rule.source_task,
        "pairing_key_count": len(keys),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "confidence": confidence,
        "observed_delta": observed_delta,
        "ci_lower": lower,
        "ci_upper": upper,
        "paired_bootstrap_ci_lower_upper": [lower, upper],
        "paired_bootstrap_ci_excludes_zero": lower > 0.0 or upper < 0.0,
    }


def _success_float(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return 1.0 if float(value) != 0.0 else 0.0

