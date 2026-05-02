"""Deterministic action identity helpers for Stage B seam traces.

The helpers in this module are intentionally side-effect free: they do not
mutate GR00T/WBC action dictionaries and they do not import heavyweight runtime
dependencies.  Policy, controller, and env seams can all call the same helpers
to record a shared ``chain_action_uuid`` plus a stable ``action_content_hash``
sidecar without changing the action path under test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import uuid
from typing import Any

from .array_summary import array_content_hash


DEFAULT_TRACE_VERSION = "stage_b_chain_action_identity_v1"
CHAIN_UUID_NAMESPACE = uuid.NAMESPACE_URL
CHAIN_UUID_PREFIX = "gr00t-wbc-stage-b-chain-action"
CONTRAST_UUID_PREFIX = "gr00t-wbc-stage-b-contrast-group"
CHAIN_ACTION_UUID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "gr00t-wbc-stage-b/chain-action-uuid/v1",
)
CONTRAST_GROUP_UUID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "gr00t-wbc-stage-b/contrast-group-uuid/v1",
)


Jsonable = None | bool | int | float | str | list["Jsonable"] | dict[str, "Jsonable"]


@dataclass(frozen=True)
class ActionIdentity:
    """Trace identity emitted beside an action at a Stage B seam."""

    chain_action_uuid: str
    action_content_hash: str
    trace_version: str
    episode_id: str
    step_id: int
    seed: int
    policy_call_index: int
    obs_hash: str
    checkpoint_id: str | None = None
    indicator_mode: str | None = None
    stage_name: str | None = None

    def to_jsonable(self) -> dict[str, object]:
        """Return a JSON-serializable sidecar payload."""

        return asdict(self)


def _normalize_scalar(value: None | bool | int | float | str) -> Jsonable:
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
    return value


def normalize_for_hash(value: Any) -> Jsonable:
    """Convert common action payload objects into stable JSON-like values.

    ``numpy`` arrays, torch tensors, and similar objects are supported
    duck-typed via ``detach``/``cpu``/``tolist`` when those methods exist.  The
    function deliberately avoids importing those packages so it remains usable
    in lightweight trace-writer tests.
    """

    if value is None or isinstance(value, bool | int | float | str):
        return _normalize_scalar(value)
    if is_dataclass(value) and not isinstance(value, type):
        return normalize_for_hash(asdict(value))
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        return {
            str(key): normalize_for_hash(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [normalize_for_hash(item) for item in value]

    candidate = value
    for method_name in ("detach", "cpu"):
        method = getattr(candidate, method_name, None)
        if callable(method):
            candidate = method()
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        return normalize_for_hash(tolist())

    return {"__repr__": repr(value), "__type__": type(value).__name__}


def canonical_json_dumps(value: Any) -> str:
    """Return deterministic JSON used as hash/UUID input."""

    return json.dumps(
        normalize_for_hash(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_content_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for an action or observation payload."""

    payload = canonical_json_dumps(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_chain_action_uuid(
    *,
    episode_id: str,
    step_id: int,
    seed: int,
    policy_call_index: int,
    obs_hash: str,
    trace_version: str = DEFAULT_TRACE_VERSION,
    checkpoint_id: str | None = None,
    indicator_mode: str | None = None,
) -> str:
    """Build the shared action-chain UUID for policy/controller/env joins."""

    fields = {
        "trace_version": trace_version,
        "episode_id": str(episode_id),
        "step_id": int(step_id),
        "seed": int(seed),
        "policy_call_index": int(policy_call_index),
        "obs_hash": str(obs_hash),
        "checkpoint_id": checkpoint_id,
        "indicator_mode": indicator_mode,
    }
    return str(uuid.uuid5(CHAIN_UUID_NAMESPACE, f"{CHAIN_UUID_PREFIX}:{canonical_json_dumps(fields)}"))


def build_contrast_group_uuid(
    *,
    episode_id: str,
    seed: int,
    obs_hash: str,
    contrast_axis: str,
    trace_version: str = DEFAULT_TRACE_VERSION,
    checkpoint_pair_id: str | None = None,
) -> str:
    """Build a paired-comparison UUID shared across modes/checkpoints."""

    fields = {
        "trace_version": trace_version,
        "episode_id": str(episode_id),
        "seed": int(seed),
        "obs_hash": str(obs_hash),
        "contrast_axis": str(contrast_axis),
        "checkpoint_pair_id": checkpoint_pair_id,
    }
    return str(uuid.uuid5(CHAIN_UUID_NAMESPACE, f"{CONTRAST_UUID_PREFIX}:{canonical_json_dumps(fields)}"))


def stable_uuid(namespace: uuid.UUID, payload: dict[str, Any]) -> str:
    """Return a deterministic UUID string for canonicalized payload metadata."""

    return str(uuid.uuid5(namespace, canonical_json_dumps(payload)))


def make_chain_action_uuid(
    *,
    trace_version: str,
    episode_id: str | int,
    step_id: str | int,
    seed: int | str,
    policy_call_index: int | str,
    obs_hash: str,
) -> str:
    """Build the Stage B chain join UUID without action content.

    This newer API is used by the JSONL/NPZ writer. It intentionally excludes
    indicator mode and action content so policy/controller/env events remain
    joinable even when the diagnostic variable changes the action.
    """

    return stable_uuid(
        CHAIN_ACTION_UUID_NAMESPACE,
        {
            "trace_version": trace_version,
            "episode_id": str(episode_id),
            "step_id": str(step_id),
            "seed": str(seed),
            "policy_call_index": str(policy_call_index),
            "obs_hash": obs_hash,
        },
    )


def make_contrast_group_uuid(
    *,
    trace_version: str,
    seed: int | str,
    obs_hash: str,
    frozen_controller_state_hash: str,
    probe_name: str,
) -> str:
    """Build the paired-comparison UUID without indicator/action content."""

    return stable_uuid(
        CONTRAST_GROUP_UUID_NAMESPACE,
        {
            "trace_version": trace_version,
            "seed": str(seed),
            "obs_hash": obs_hash,
            "frozen_controller_state_hash": frozen_controller_state_hash,
            "probe_name": probe_name,
        },
    )


def make_action_content_hash(value: Any) -> str:
    """Hash the action/controller/env payload that Stage B is diagnosing."""

    return array_content_hash(value)


def hash_text(value: str) -> str:
    """Hash prompt text or other string metadata without storing raw content."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_action_identity(
    *,
    action_payload: Any,
    episode_id: str,
    step_id: int,
    seed: int,
    policy_call_index: int,
    obs_hash: str,
    trace_version: str = DEFAULT_TRACE_VERSION,
    checkpoint_id: str | None = None,
    indicator_mode: str | None = None,
    stage_name: str | None = None,
) -> ActionIdentity:
    """Create the sidecar identity for one action without mutating it."""

    return ActionIdentity(
        chain_action_uuid=build_chain_action_uuid(
            trace_version=trace_version,
            episode_id=episode_id,
            step_id=step_id,
            seed=seed,
            policy_call_index=policy_call_index,
            obs_hash=obs_hash,
            checkpoint_id=checkpoint_id,
            indicator_mode=indicator_mode,
        ),
        action_content_hash=stable_content_hash(action_payload),
        trace_version=trace_version,
        episode_id=str(episode_id),
        step_id=int(step_id),
        seed=int(seed),
        policy_call_index=int(policy_call_index),
        obs_hash=str(obs_hash),
        checkpoint_id=checkpoint_id,
        indicator_mode=indicator_mode,
        stage_name=stage_name,
    )


def validate_trace_alignment(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate that seam records share one UUID and source action hash."""

    uuids = {str(record.get("chain_action_uuid")) for record in records}
    hashes = {str(record.get("action_content_hash")) for record in records}
    missing_indexes = [
        index
        for index, record in enumerate(records)
        if not record.get("chain_action_uuid") or not record.get("action_content_hash")
    ]
    return {
        "status": (
            "PASS"
            if len(uuids) == 1 and len(hashes) == 1 and not missing_indexes
            else "FAIL"
        ),
        "record_count": len(records),
        "unique_chain_action_uuid_count": len(uuids),
        "unique_action_content_hash_count": len(hashes),
        "missing_identity_record_indexes": missing_indexes,
        "chain_action_uuid": next(iter(uuids)) if len(uuids) == 1 else None,
        "action_content_hash": next(iter(hashes)) if len(hashes) == 1 else None,
    }
