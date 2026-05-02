from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast


EXPECTED_SCHEMA_VERSION = "openpi_libero_rollout_eval_manifest_v2"
EXPECTED_EVAL_AUTHORITY = "fresh_rollout_v2"
EXPECTED_TASK_SUITE_NAME = "libero_spatial"
EXPECTED_V1_COMPARISON_TASK_IDS: tuple[int, ...] = (0, 1)
EXPECTED_V1_COMPARISON_SEEDS: tuple[int, ...] = (7, 17)
EXPECTED_V1_COMPARISON_NUM_TRIALS = 2
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"


@dataclass(frozen=True)
class FrozenManifestSpec:
    task_ids: tuple[int, ...]
    seed_manifest: tuple[int, ...]
    num_trials_per_task: int
    file_name: str


_FROZEN_MANIFEST_SPECS = {
    "rollout_lite_v2": FrozenManifestSpec(
        task_ids=(0, 1),
        seed_manifest=(7, 17, 27, 37),
        num_trials_per_task=4,
        file_name="eval_manifest_rollout_lite_v2.json",
    ),
    "rollout_strong_v2": FrozenManifestSpec(
        task_ids=(0, 1),
        seed_manifest=(7, 17, 27, 37, 47, 57),
        num_trials_per_task=4,
        file_name="eval_manifest_rollout_strong_v2.json",
    ),
}


@dataclass(frozen=True)
class LiberoRolloutEvalManifestV2:
    schema_version: str
    eval_authority: str
    manifest_name: str
    task_suite_name: str
    task_ids: tuple[int, ...]
    seed_manifest: tuple[int, ...]
    num_trials_per_task: int

    @property
    def total_episodes(self) -> int:
        return len(self.task_ids) * len(self.seed_manifest) * self.num_trials_per_task


def _as_mapping(
    manifest: LiberoRolloutEvalManifestV2 | Mapping[str, object],
) -> Mapping[str, object]:
    if isinstance(manifest, Mapping):
        return manifest
    return asdict(manifest)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _coerce_required_int(payload: Mapping[str, object], key: str) -> int:
    raw = payload.get(key)
    if raw is None or isinstance(raw, bool):
        raise ValueError(f"invalid {key} {raw!r}; expected integer")
    if not isinstance(raw, (int, float, str)):
        raise ValueError(f"invalid {key} {raw!r}; expected integer")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key} {raw!r}; expected integer") from exc


def _coerce_required_manifest(
    payload: Mapping[str, object], key: str
) -> tuple[int, ...]:
    raw = payload.get(key)
    if raw is None or isinstance(raw, (str, bytes, Mapping)):
        raise ValueError(f"invalid {key} {raw!r}; expected integer sequence")
    if not isinstance(raw, Sequence):
        raise ValueError(f"invalid {key} {raw!r}; expected integer sequence")
    coerced: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"invalid {key} {raw!r}; expected integer sequence")
        try:
            coerced.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid {key} {raw!r}; expected integer sequence"
            ) from exc
    if not coerced:
        raise ValueError(f"invalid {key} {raw!r}; expected non-empty integer sequence")
    return tuple(coerced)


def _require_manifest_name(payload: Mapping[str, object]) -> str:
    manifest_name = str(payload.get("manifest_name", "")).strip()
    if manifest_name not in _FROZEN_MANIFEST_SPECS:
        expected = ", ".join(sorted(_FROZEN_MANIFEST_SPECS))
        raise ValueError(
            f"invalid manifest_name {manifest_name!r}; expected one of {expected}"
        )
    return manifest_name


def _validate_frozen_scope(
    *,
    manifest_name: str,
    task_ids: tuple[int, ...],
    seed_manifest: tuple[int, ...],
    num_trials_per_task: int,
) -> None:
    if (
        task_ids == EXPECTED_V1_COMPARISON_TASK_IDS
        and seed_manifest == EXPECTED_V1_COMPARISON_SEEDS
        and num_trials_per_task == EXPECTED_V1_COMPARISON_NUM_TRIALS
    ):
        raise ValueError(
            "v1 comparison tier [0,1] x [7,17] x 2 is not v2 authority; "
            + "evaluation_tier=comparison is v1 reference only"
        )

    expected_scope = _FROZEN_MANIFEST_SPECS[manifest_name]
    expected_task_ids = expected_scope.task_ids
    expected_seeds = expected_scope.seed_manifest
    expected_trials = expected_scope.num_trials_per_task
    if (
        task_ids != expected_task_ids
        or seed_manifest != expected_seeds
        or num_trials_per_task != expected_trials
    ):
        raise ValueError(
            f"manifest {manifest_name!r} must use frozen scope "
            + f"task_ids={expected_task_ids!r}, "
            + f"seed_manifest={expected_seeds!r}, "
            + f"num_trials_per_task={expected_trials!r}; got "
            + f"task_ids={task_ids!r}, "
            + f"seed_manifest={seed_manifest!r}, "
            + f"num_trials_per_task={num_trials_per_task!r}"
        )


def validate_rollout_eval_manifest_v2(
    manifest: LiberoRolloutEvalManifestV2 | Mapping[str, object],
) -> LiberoRolloutEvalManifestV2:
    payload = _as_mapping(manifest)

    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"invalid schema_version {schema_version!r}; expected {EXPECTED_SCHEMA_VERSION!r}"
        )

    eval_authority = str(payload.get("eval_authority", "")).strip()
    if eval_authority != EXPECTED_EVAL_AUTHORITY:
        raise ValueError(
            f"invalid eval_authority {eval_authority!r}; expected {EXPECTED_EVAL_AUTHORITY!r}"
        )

    manifest_name = _require_manifest_name(payload)

    task_suite_name = str(payload.get("task_suite_name", "")).strip()
    if task_suite_name != EXPECTED_TASK_SUITE_NAME:
        raise ValueError(
            "invalid task_suite_name "
            + f"{task_suite_name!r}; expected {EXPECTED_TASK_SUITE_NAME!r}"
        )

    task_ids = _coerce_required_manifest(payload, "task_ids")
    seed_manifest = _coerce_required_manifest(payload, "seed_manifest")
    num_trials_per_task = _coerce_required_int(payload, "num_trials_per_task")
    _validate_frozen_scope(
        manifest_name=manifest_name,
        task_ids=task_ids,
        seed_manifest=seed_manifest,
        num_trials_per_task=num_trials_per_task,
    )

    return LiberoRolloutEvalManifestV2(
        schema_version=schema_version,
        eval_authority=eval_authority,
        manifest_name=manifest_name,
        task_suite_name=task_suite_name,
        task_ids=task_ids,
        seed_manifest=seed_manifest,
        num_trials_per_task=num_trials_per_task,
    )


def build_rollout_eval_manifest_v2(
    *, manifest_name: str = "rollout_lite_v2"
) -> LiberoRolloutEvalManifestV2:
    spec = _FROZEN_MANIFEST_SPECS.get(manifest_name)
    if spec is None:
        expected = ", ".join(sorted(_FROZEN_MANIFEST_SPECS))
        raise ValueError(
            f"invalid manifest_name {manifest_name!r}; expected one of {expected}"
        )
    return validate_rollout_eval_manifest_v2(
        {
            "schema_version": EXPECTED_SCHEMA_VERSION,
            "eval_authority": EXPECTED_EVAL_AUTHORITY,
            "manifest_name": manifest_name,
            "task_suite_name": EXPECTED_TASK_SUITE_NAME,
            "task_ids": list(spec.task_ids),
            "seed_manifest": list(spec.seed_manifest),
            "num_trials_per_task": spec.num_trials_per_task,
        }
    )


def manifest_payload_v2(
    manifest: LiberoRolloutEvalManifestV2 | Mapping[str, object],
) -> dict[str, object]:
    validated = validate_rollout_eval_manifest_v2(manifest)
    return {
        "schema_version": validated.schema_version,
        "eval_authority": validated.eval_authority,
        "manifest_name": validated.manifest_name,
        "task_suite_name": validated.task_suite_name,
        "task_ids": list(validated.task_ids),
        "seed_manifest": list(validated.seed_manifest),
        "num_trials_per_task": validated.num_trials_per_task,
    }


def compute_rollout_eval_manifest_hash(
    manifest: LiberoRolloutEvalManifestV2 | Mapping[str, object],
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(manifest_payload_v2(manifest))
    ).hexdigest()


def resolve_tracked_rollout_eval_manifest_path(manifest_name: str) -> Path:
    spec = _FROZEN_MANIFEST_SPECS.get(str(manifest_name).strip())
    if spec is None:
        expected = ", ".join(sorted(_FROZEN_MANIFEST_SPECS))
        raise ValueError(
            f"invalid manifest_name {manifest_name!r}; expected one of {expected}"
        )
    return MANIFEST_DIR / spec.file_name


def load_rollout_eval_manifest_v2(path: str | Path) -> LiberoRolloutEvalManifestV2:
    payload_object = cast(object, json.loads(Path(path).read_text(encoding="utf-8")))
    if not isinstance(payload_object, Mapping):
        raise ValueError(f"manifest at {path} must be a JSON object")
    payload_mapping = cast(Mapping[object, object], payload_object)
    mapping_payload = {str(key): value for key, value in payload_mapping.items()}
    return validate_rollout_eval_manifest_v2(mapping_payload)


def compute_rollout_eval_manifest_file_hash(path: str | Path) -> str:
    return compute_rollout_eval_manifest_hash(load_rollout_eval_manifest_v2(path))


__all__ = [
    "EXPECTED_EVAL_AUTHORITY",
    "EXPECTED_SCHEMA_VERSION",
    "EXPECTED_TASK_SUITE_NAME",
    "LiberoRolloutEvalManifestV2",
    "build_rollout_eval_manifest_v2",
    "compute_rollout_eval_manifest_file_hash",
    "compute_rollout_eval_manifest_hash",
    "load_rollout_eval_manifest_v2",
    "manifest_payload_v2",
    "resolve_tracked_rollout_eval_manifest_path",
    "validate_rollout_eval_manifest_v2",
]
