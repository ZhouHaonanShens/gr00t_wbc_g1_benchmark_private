from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast


EXPECTED_SCHEMA_VERSION = "openpi_libero_fresh_rollout_manifest_v21"
EXPECTED_TASK_SUITE_NAME = "libero_spatial"
EXPECTED_METRIC_PROFILE = "budget_ladder_v1"
EXPECTED_EPISODE_BUDGET_MODE = "inherit_from_protocol"
EXPECTED_BUDGET_FRACTIONS: tuple[float, ...] = (0.50, 0.75, 1.00)
EXPECTED_ALLOWED_VARIANTS: tuple[str, ...] = (
    "stock_libero_ref_v1",
    "fixedadv_relabel8d_control_v1",
    "recap_only_relabel8d_v2",
    "recap_shuffledadv_diag_v1",
)
EXPECTED_STOCK_ONLY_VARIANTS: tuple[str, ...] = (EXPECTED_ALLOWED_VARIANTS[0],)
EXPECTED_SMOKE_TASK_IDS: tuple[int, ...] = (0,)
EXPECTED_SMOKE_SEED_MANIFEST: tuple[int, ...] = (7,)
EXPECTED_SMOKE_NUM_TRIALS = 1
EXPECTED_SCAN_TASK_IDS: tuple[int, ...] = (0, 1)
EXPECTED_SCAN_SEED_MANIFEST: tuple[int, ...] = (
    7,
    17,
    27,
    37,
    47,
    57,
    67,
    77,
    87,
    97,
    107,
    117,
    127,
    137,
    147,
    157,
)
EXPECTED_SCAN_NUM_TRIALS = 2
EXPECTED_LITE_NUM_TRIALS = 2
EXPECTED_STRONG_NUM_TRIALS = 4
EXPECTED_AUTHORITY_IDS: tuple[str, ...] = (
    "fresh_rollout_v21_smoke",
    "fresh_rollout_v21_scan",
    "fresh_rollout_v21_lite",
    "fresh_rollout_v21_strong",
)
ROLLOUT_INPUT_SUMMARY_SCHEMA_VERSION = "openpi_libero_rollout_eval_v21_input_v2"
EXPECTED_RUNTIME_INDICATOR_CLI_MODES: tuple[str, ...] = (
    "positive",
    "negative",
    "omit",
    "cfg",
)
EXPECTED_RESOLVED_RUNTIME_INDICATOR_MODES: tuple[str, ...] = (
    "positive",
    "negative",
    "omit",
)
EXPECTED_RUNTIME_PROMPT_TEXT_SURFACES: tuple[str, ...] = (
    "canonical_text_indicator",
    "prompt_raw_only",
)
ROLLOUT_INPUT_RUNTIME_PROMPT_FIELDS: tuple[str, ...] = (
    "indicator_mode_requested",
    "indicator_mode",
    "indicator_source",
    "prompt_text_surface",
    "critic_checkpoint_ref",
)
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"

_V2_COLLIDING_AUTHORITY_IDS = (
    "fresh_rollout_v2",
    "rollout_lite_v2",
    "rollout_strong_v2",
)
_ALLOWED_VARIANT_SET = frozenset(EXPECTED_ALLOWED_VARIANTS)
_VARIANT_ORDER = {
    variant: index for index, variant in enumerate(EXPECTED_ALLOWED_VARIANTS)
}


@dataclass(frozen=True)
class FrozenPresetSpec:
    authority_id: str
    manifest_name: str
    task_ids: tuple[int, ...]
    seed_manifest: tuple[int, ...]
    num_trials_per_task: int
    variant_scope: tuple[str, ...]
    file_name: str


_FROZEN_PRESET_SPECS = {
    "smoke_trace_v21": FrozenPresetSpec(
        authority_id="fresh_rollout_v21_smoke",
        manifest_name="smoke_trace_v21",
        task_ids=EXPECTED_SMOKE_TASK_IDS,
        seed_manifest=EXPECTED_SMOKE_SEED_MANIFEST,
        num_trials_per_task=EXPECTED_SMOKE_NUM_TRIALS,
        variant_scope=EXPECTED_STOCK_ONLY_VARIANTS,
        file_name="smoke_trace_v21.json",
    ),
    "seed_scan_stock_v21": FrozenPresetSpec(
        authority_id="fresh_rollout_v21_scan",
        manifest_name="seed_scan_stock_v21",
        task_ids=EXPECTED_SCAN_TASK_IDS,
        seed_manifest=EXPECTED_SCAN_SEED_MANIFEST,
        num_trials_per_task=EXPECTED_SCAN_NUM_TRIALS,
        variant_scope=EXPECTED_STOCK_ONLY_VARIANTS,
        file_name="seed_scan_stock_v21.json",
    ),
}


@dataclass(frozen=True)
class LiberoFreshRolloutManifestV21:
    schema_version: str
    authority_id: str
    manifest_name: str
    task_suite_name: str
    task_ids: tuple[int, ...]
    seed_manifest: tuple[int, ...] | None
    per_task_seed_manifest: dict[str, tuple[int, ...]] | None
    num_trials_per_task: int
    variant_scope: tuple[str, ...]
    budget_fractions: tuple[float, ...]
    metric_profile: str
    episode_budget_mode: str
    selection_policy: str | None = None
    selection_source: str | None = None
    selection_source_hash: str | None = None

    @property
    def total_episodes(self) -> int:
        if self.per_task_seed_manifest is not None:
            return (
                sum(
                    len(seed_manifest)
                    for seed_manifest in self.per_task_seed_manifest.values()
                )
                * self.num_trials_per_task
            )
        seed_manifest = self.seed_manifest
        if seed_manifest is None:
            raise ValueError(
                "v21 rollout manifest must define seed_manifest or per_task_seed_manifest"
            )
        return len(self.task_ids) * len(seed_manifest) * self.num_trials_per_task


def _as_mapping(
    manifest: LiberoFreshRolloutManifestV21 | Mapping[str, object],
) -> Mapping[str, object]:
    if isinstance(manifest, Mapping):
        return manifest
    return asdict(manifest)


def _render_json_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".2f")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    raise TypeError(f"unsupported JSON scalar {value!r}")


def _canonical_json_text(payload: object, *, pretty: bool) -> str:
    if isinstance(payload, Mapping):
        mapping_payload = cast(Mapping[object, object], payload)
        items_unsorted: list[tuple[str, object]] = []
        for raw_key, value in mapping_payload.items():
            items_unsorted.append((str(raw_key), value))
        items = sorted(items_unsorted)
        if not items:
            return "{}"
        if not pretty:
            serialized = ",".join(
                f"{json.dumps(key, ensure_ascii=True)}:{_canonical_json_text(value, pretty=False)}"
                for key, value in items
            )
            return "{" + serialized + "}"
        serialized = ",\n".join(
            "  "
            + f"{json.dumps(key, ensure_ascii=True)}: {_canonical_json_text(value, pretty=True)}"
            for key, value in items
        )
        return "{\n" + serialized + "\n}"

    if isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        rendered_items = [
            _canonical_json_text(value, pretty=False) for value in payload
        ]
        separator = ", " if pretty else ","
        return "[" + separator.join(rendered_items) + "]"

    return _render_json_scalar(payload)


def _canonical_json_bytes(payload: object) -> bytes:
    return _canonical_json_text(payload, pretty=False).encode("utf-8")


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


def _coerce_required_str(payload: Mapping[str, object], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing {key} in v21 rollout manifest")
    return value


def _coerce_optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_rollout_input_summary_runtime_fields(
    payload: Mapping[str, object],
) -> dict[str, str]:
    requested_indicator_mode = _coerce_required_str(payload, "indicator_mode_requested")
    if requested_indicator_mode not in EXPECTED_RUNTIME_INDICATOR_CLI_MODES:
        raise ValueError(
            "invalid indicator_mode_requested "
            + f"{requested_indicator_mode!r}; expected {EXPECTED_RUNTIME_INDICATOR_CLI_MODES!r}"
        )
    indicator_mode = _coerce_required_str(payload, "indicator_mode")
    if indicator_mode not in EXPECTED_RESOLVED_RUNTIME_INDICATOR_MODES:
        raise ValueError(
            "invalid indicator_mode "
            + f"{indicator_mode!r}; expected {EXPECTED_RESOLVED_RUNTIME_INDICATOR_MODES!r}"
        )
    indicator_source = _coerce_required_str(payload, "indicator_source")
    prompt_text_surface = _coerce_required_str(payload, "prompt_text_surface")
    if prompt_text_surface not in EXPECTED_RUNTIME_PROMPT_TEXT_SURFACES:
        raise ValueError(
            "invalid prompt_text_surface "
            + f"{prompt_text_surface!r}; expected {EXPECTED_RUNTIME_PROMPT_TEXT_SURFACES!r}"
        )
    critic_checkpoint_ref = _coerce_required_str(payload, "critic_checkpoint_ref")
    return {
        "indicator_mode_requested": requested_indicator_mode,
        "indicator_mode": indicator_mode,
        "indicator_source": indicator_source,
        "prompt_text_surface": prompt_text_surface,
        "critic_checkpoint_ref": critic_checkpoint_ref,
    }


def _coerce_required_int_sequence_from_raw(raw: object, key: str) -> tuple[int, ...]:
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


def _coerce_required_float_sequence(
    payload: Mapping[str, object], key: str
) -> tuple[float, ...]:
    raw = payload.get(key)
    if raw is None or isinstance(raw, (str, bytes, Mapping)):
        raise ValueError(f"invalid {key} {raw!r}; expected numeric sequence")
    if not isinstance(raw, Sequence):
        raise ValueError(f"invalid {key} {raw!r}; expected numeric sequence")
    coerced: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"invalid {key} {raw!r}; expected numeric sequence")
        try:
            coerced_value = round(float(value), 2)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid {key} {raw!r}; expected numeric sequence"
            ) from exc
        coerced.append(coerced_value)
    if not coerced:
        raise ValueError(f"invalid {key} {raw!r}; expected non-empty numeric sequence")
    return tuple(coerced)


def _coerce_optional_seed_manifest(
    payload: Mapping[str, object],
) -> tuple[int, ...] | None:
    alias_keys = ("seed_manifest", "seeds", "seed_pool")
    present_keys = [key for key in alias_keys if key in payload]
    if not present_keys:
        return None
    value_keys = [key for key in present_keys if payload[key] is not None]
    if not value_keys:
        return None
    if len(value_keys) != len(present_keys):
        raise ValueError(
            f"conflicting seed manifest aliases {present_keys!r}; expected identical values"
        )
    canonical_values = [
        tuple(sorted(_coerce_required_int_sequence_from_raw(payload[key], key)))
        for key in value_keys
    ]
    first_value = canonical_values[0]
    if any(value != first_value for value in canonical_values[1:]):
        raise ValueError(
            f"conflicting seed manifest aliases {value_keys!r}; expected identical values"
        )
    return first_value


def _coerce_per_task_seed_manifest(
    payload: Mapping[str, object], *, task_ids: tuple[int, ...]
) -> dict[str, tuple[int, ...]] | None:
    raw = payload.get("per_task_seed_manifest")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"invalid per_task_seed_manifest {raw!r}; expected task_id -> integer sequence mapping"
        )

    raw_mapping = cast(Mapping[object, object], raw)
    normalized_entries: list[tuple[int, tuple[int, ...]]] = []
    for raw_task_id, raw_seed_manifest in raw_mapping.items():
        if isinstance(raw_task_id, bool) or not isinstance(
            raw_task_id, (int, float, str)
        ):
            raise ValueError(
                "invalid per_task_seed_manifest task_id "
                + f"{raw_task_id!r}; expected integer-like task ids"
            )
        try:
            task_id = int(raw_task_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "invalid per_task_seed_manifest task_id "
                + f"{raw_task_id!r}; expected integer-like task ids"
            ) from exc

        seed_manifest = tuple(
            sorted(
                _coerce_required_int_sequence_from_raw(
                    raw_seed_manifest,
                    f"per_task_seed_manifest[{task_id}]",
                )
            )
        )
        _reject_duplicates(
            seed_manifest,
            key=f"per_task_seed_manifest[{task_id}]",
        )
        normalized_entries.append((task_id, seed_manifest))

    if not normalized_entries:
        raise ValueError(
            "invalid per_task_seed_manifest {}; expected non-empty task_id -> integer sequence mapping"
        )

    normalized_task_ids = [task_id for task_id, _ in normalized_entries]
    _reject_duplicates(normalized_task_ids, key="per_task_seed_manifest task_ids")
    actual_task_ids = tuple(sorted(normalized_task_ids))
    if actual_task_ids != task_ids:
        raise ValueError(
            "per_task_seed_manifest task coverage must exactly match task_ids; "
            + f"expected {task_ids!r}; got {actual_task_ids!r}"
        )

    sorted_entries = sorted(normalized_entries)
    return {str(task_id): seed_manifest for task_id, seed_manifest in sorted_entries}


def _coerce_seed_selection(
    payload: Mapping[str, object],
    *,
    authority_id: str,
    task_ids: tuple[int, ...],
) -> tuple[tuple[int, ...] | None, dict[str, tuple[int, ...]] | None]:
    shared_seed_manifest = _coerce_optional_seed_manifest(payload)
    if shared_seed_manifest is not None:
        _reject_duplicates(shared_seed_manifest, key="seed_manifest")
    per_task_seed_manifest = _coerce_per_task_seed_manifest(payload, task_ids=task_ids)

    if authority_id in {"fresh_rollout_v21_smoke", "fresh_rollout_v21_scan"}:
        if per_task_seed_manifest is not None:
            raise ValueError(
                f"authority_id {authority_id!r} does not accept per_task_seed_manifest"
            )
        if shared_seed_manifest is None:
            raise ValueError(
                "missing seed_manifest in v21 rollout manifest; accepted aliases are "
                + "seed_manifest, seeds, seed_pool"
            )
        return shared_seed_manifest, None

    if shared_seed_manifest is None and per_task_seed_manifest is None:
        raise ValueError(
            "missing seed selection in v21 rollout manifest; expected seed_manifest/seeds/seed_pool "
            + "or per_task_seed_manifest"
        )

    if shared_seed_manifest is not None and per_task_seed_manifest is not None:
        unique_task_seed_manifests = set(per_task_seed_manifest.values())
        if (
            len(unique_task_seed_manifests) != 1
            or next(iter(unique_task_seed_manifests)) != shared_seed_manifest
        ):
            raise ValueError(
                "ambiguous mixed seed manifest forms are not allowed unless shared seed_manifest "
                + "exactly matches every per_task_seed_manifest entry"
            )

    return shared_seed_manifest, per_task_seed_manifest


def _reject_duplicates(values: Sequence[object], *, key: str) -> None:
    seen: set[object] = set()
    duplicates: list[object] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate {key} entries are not allowed: {duplicates!r}")


def _canonicalize_variant_scope(payload: Mapping[str, object]) -> tuple[str, ...]:
    raw = payload.get("variant_scope")
    if raw is None or isinstance(raw, (str, bytes, Mapping)):
        raise ValueError(f"invalid variant_scope {raw!r}; expected string sequence")
    if not isinstance(raw, Sequence):
        raise ValueError(f"invalid variant_scope {raw!r}; expected string sequence")
    variants = tuple(str(value).strip() for value in raw)
    if not variants or any(not value for value in variants):
        raise ValueError(
            f"invalid variant_scope {raw!r}; expected non-empty string sequence"
        )
    _reject_duplicates(variants, key="variant_scope")
    invalid_variants = sorted(
        {value for value in variants if value not in _ALLOWED_VARIANT_SET}
    )
    if invalid_variants:
        expected = ", ".join(EXPECTED_ALLOWED_VARIANTS)
        raise ValueError(
            "variant_scope must stay within A/B/C/X only; "
            + f"got invalid variants {invalid_variants!r}; expected subset of {expected}"
        )
    return tuple(sorted(variants, key=_VARIANT_ORDER.__getitem__))


def _canonicalize_budget_fractions(payload: Mapping[str, object]) -> tuple[float, ...]:
    budget_fractions = _coerce_required_float_sequence(payload, "budget_fractions")
    _reject_duplicates(budget_fractions, key="budget_fractions")
    return tuple(sorted(budget_fractions))


def _validate_authority_id(authority_id: str) -> str:
    stripped = str(authority_id).strip()
    if stripped in _V2_COLLIDING_AUTHORITY_IDS:
        raise ValueError(
            f"authority_id {stripped!r} collides with existing v2 authority ids and cannot be reused"
        )
    if stripped not in EXPECTED_AUTHORITY_IDS:
        expected = ", ".join(EXPECTED_AUTHORITY_IDS)
        raise ValueError(
            f"invalid authority_id {stripped!r}; expected one of {expected}"
        )
    return stripped


def _validate_selection_metadata(
    *,
    authority_id: str,
    selection_policy: str | None,
    selection_source: str | None,
    selection_source_hash: str | None,
) -> tuple[str | None, str | None, str | None]:
    if authority_id in {"fresh_rollout_v21_lite", "fresh_rollout_v21_strong"}:
        if not selection_policy or not selection_source or not selection_source_hash:
            raise ValueError(
                f"authority_id {authority_id!r} requires selection_policy, selection_source, and selection_source_hash"
            )
        return selection_policy, selection_source, selection_source_hash

    if selection_policy or selection_source or selection_source_hash:
        raise ValueError(
            f"authority_id {authority_id!r} does not accept selection metadata"
        )
    return None, None, None


def _validate_authority_scope(
    *,
    authority_id: str,
    task_ids: tuple[int, ...],
    seed_manifest: tuple[int, ...] | None,
    per_task_seed_manifest: dict[str, tuple[int, ...]] | None,
    num_trials_per_task: int,
    variant_scope: tuple[str, ...],
) -> None:
    if authority_id == "fresh_rollout_v21_smoke":
        if per_task_seed_manifest is not None:
            raise ValueError("smoke authority must not define per_task_seed_manifest")
        if task_ids != EXPECTED_SMOKE_TASK_IDS:
            raise ValueError(
                f"smoke authority must use task_ids={EXPECTED_SMOKE_TASK_IDS!r}; got {task_ids!r}"
            )
        if seed_manifest != EXPECTED_SMOKE_SEED_MANIFEST:
            raise ValueError(
                "smoke authority must use seed_manifest="
                + f"{EXPECTED_SMOKE_SEED_MANIFEST!r}; got {seed_manifest!r}"
            )
        if num_trials_per_task != EXPECTED_SMOKE_NUM_TRIALS:
            raise ValueError(
                f"smoke authority must use num_trials_per_task={EXPECTED_SMOKE_NUM_TRIALS}; got {num_trials_per_task!r}"
            )
        if variant_scope != EXPECTED_STOCK_ONLY_VARIANTS:
            raise ValueError(
                "smoke authority must stay stock-only with variant_scope="
                + f"{EXPECTED_STOCK_ONLY_VARIANTS!r}; got {variant_scope!r}"
            )
        return

    if authority_id == "fresh_rollout_v21_scan":
        if per_task_seed_manifest is not None:
            raise ValueError("scan authority must not define per_task_seed_manifest")
        if task_ids != EXPECTED_SCAN_TASK_IDS:
            raise ValueError(
                f"scan authority must use task_ids={EXPECTED_SCAN_TASK_IDS!r}; got {task_ids!r}"
            )
        if seed_manifest != EXPECTED_SCAN_SEED_MANIFEST:
            raise ValueError(
                "scan authority must use the frozen seed_manifest="
                + f"{EXPECTED_SCAN_SEED_MANIFEST!r}; got {seed_manifest!r}"
            )
        if num_trials_per_task != EXPECTED_SCAN_NUM_TRIALS:
            raise ValueError(
                f"scan authority must use num_trials_per_task={EXPECTED_SCAN_NUM_TRIALS}; got {num_trials_per_task!r}"
            )
        if variant_scope != EXPECTED_STOCK_ONLY_VARIANTS:
            raise ValueError(
                "scan authority must stay stock-only with variant_scope="
                + f"{EXPECTED_STOCK_ONLY_VARIANTS!r}; got {variant_scope!r}"
            )
        return

    if authority_id == "fresh_rollout_v21_lite":
        if num_trials_per_task != EXPECTED_LITE_NUM_TRIALS:
            raise ValueError(
                f"lite authority must use num_trials_per_task={EXPECTED_LITE_NUM_TRIALS}; got {num_trials_per_task!r}"
            )
        if variant_scope != EXPECTED_ALLOWED_VARIANTS:
            raise ValueError(
                f"lite authority must use variant_scope={EXPECTED_ALLOWED_VARIANTS!r}; got {variant_scope!r}"
            )
        return

    if authority_id == "fresh_rollout_v21_strong":
        if num_trials_per_task != EXPECTED_STRONG_NUM_TRIALS:
            raise ValueError(
                f"strong authority must use num_trials_per_task={EXPECTED_STRONG_NUM_TRIALS}; got {num_trials_per_task!r}"
            )
        if variant_scope != EXPECTED_ALLOWED_VARIANTS:
            raise ValueError(
                f"strong authority must use variant_scope={EXPECTED_ALLOWED_VARIANTS!r}; got {variant_scope!r}"
            )

        return


def validate_rollout_manifest_v21(
    manifest: LiberoFreshRolloutManifestV21 | Mapping[str, object],
) -> LiberoFreshRolloutManifestV21:
    payload = _as_mapping(manifest)

    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"invalid schema_version {schema_version!r}; expected {EXPECTED_SCHEMA_VERSION!r}"
        )

    authority_id = _validate_authority_id(str(payload.get("authority_id", "")))
    manifest_name = _coerce_required_str(payload, "manifest_name")

    task_suite_name = str(payload.get("task_suite_name", "")).strip()
    if task_suite_name != EXPECTED_TASK_SUITE_NAME:
        raise ValueError(
            "invalid task_suite_name "
            + f"{task_suite_name!r}; expected {EXPECTED_TASK_SUITE_NAME!r}"
        )

    task_ids = tuple(
        sorted(
            _coerce_required_int_sequence_from_raw(payload.get("task_ids"), "task_ids")
        )
    )
    _reject_duplicates(task_ids, key="task_ids")
    seed_manifest, per_task_seed_manifest = _coerce_seed_selection(
        payload,
        authority_id=authority_id,
        task_ids=task_ids,
    )
    num_trials_per_task = _coerce_required_int(payload, "num_trials_per_task")
    variant_scope = _canonicalize_variant_scope(payload)
    budget_fractions = _canonicalize_budget_fractions(payload)
    if budget_fractions != EXPECTED_BUDGET_FRACTIONS:
        raise ValueError(
            "invalid budget_fractions "
            + f"{budget_fractions!r}; expected {EXPECTED_BUDGET_FRACTIONS!r}"
        )

    metric_profile = str(payload.get("metric_profile", "")).strip()
    if metric_profile != EXPECTED_METRIC_PROFILE:
        raise ValueError(
            f"invalid metric_profile {metric_profile!r}; expected {EXPECTED_METRIC_PROFILE!r}"
        )

    episode_budget_mode = str(payload.get("episode_budget_mode", "")).strip()
    if episode_budget_mode != EXPECTED_EPISODE_BUDGET_MODE:
        raise ValueError(
            "invalid episode_budget_mode "
            + f"{episode_budget_mode!r}; expected {EXPECTED_EPISODE_BUDGET_MODE!r}"
        )

    selection_policy, selection_source, selection_source_hash = (
        _validate_selection_metadata(
            authority_id=authority_id,
            selection_policy=_coerce_optional_str(payload, "selection_policy"),
            selection_source=_coerce_optional_str(payload, "selection_source"),
            selection_source_hash=_coerce_optional_str(
                payload, "selection_source_hash"
            ),
        )
    )
    _validate_authority_scope(
        authority_id=authority_id,
        task_ids=task_ids,
        seed_manifest=seed_manifest,
        per_task_seed_manifest=per_task_seed_manifest,
        num_trials_per_task=num_trials_per_task,
        variant_scope=variant_scope,
    )

    return LiberoFreshRolloutManifestV21(
        schema_version=schema_version,
        authority_id=authority_id,
        manifest_name=manifest_name,
        task_suite_name=task_suite_name,
        task_ids=task_ids,
        seed_manifest=seed_manifest,
        per_task_seed_manifest=per_task_seed_manifest,
        num_trials_per_task=num_trials_per_task,
        variant_scope=variant_scope,
        budget_fractions=budget_fractions,
        metric_profile=metric_profile,
        episode_budget_mode=episode_budget_mode,
        selection_policy=selection_policy,
        selection_source=selection_source,
        selection_source_hash=selection_source_hash,
    )


def build_rollout_manifest_v21(
    *,
    authority_id: str,
    manifest_name: str,
    task_ids: Sequence[int],
    seed_manifest: Sequence[int] | None = None,
    per_task_seed_manifest: Mapping[int | str, Sequence[int]] | None = None,
    num_trials_per_task: int,
    variant_scope: Sequence[str],
    budget_fractions: Sequence[float] = EXPECTED_BUDGET_FRACTIONS,
    metric_profile: str = EXPECTED_METRIC_PROFILE,
    episode_budget_mode: str = EXPECTED_EPISODE_BUDGET_MODE,
    selection_policy: str | None = None,
    selection_source: str | None = None,
    selection_source_hash: str | None = None,
) -> LiberoFreshRolloutManifestV21:
    payload: dict[str, object] = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "authority_id": authority_id,
        "manifest_name": manifest_name,
        "task_suite_name": EXPECTED_TASK_SUITE_NAME,
        "task_ids": list(task_ids),
        "num_trials_per_task": num_trials_per_task,
        "variant_scope": list(variant_scope),
        "budget_fractions": list(budget_fractions),
        "metric_profile": metric_profile,
        "episode_budget_mode": episode_budget_mode,
        "selection_policy": selection_policy,
        "selection_source": selection_source,
        "selection_source_hash": selection_source_hash,
    }
    if seed_manifest is not None:
        payload["seed_manifest"] = list(seed_manifest)
    if per_task_seed_manifest is not None:
        payload["per_task_seed_manifest"] = {
            str(task_id): list(task_seed_manifest)
            for task_id, task_seed_manifest in per_task_seed_manifest.items()
        }
    return validate_rollout_manifest_v21(payload)


def build_rollout_manifest_preset_v21(
    *, preset_name: str = "smoke_trace_v21"
) -> LiberoFreshRolloutManifestV21:
    preset = _FROZEN_PRESET_SPECS.get(str(preset_name).strip())
    if preset is None:
        expected = ", ".join(sorted(_FROZEN_PRESET_SPECS))
        raise ValueError(
            f"invalid preset_name {preset_name!r}; expected one of {expected}"
        )
    return build_rollout_manifest_v21(
        authority_id=preset.authority_id,
        manifest_name=preset.manifest_name,
        task_ids=preset.task_ids,
        seed_manifest=preset.seed_manifest,
        num_trials_per_task=preset.num_trials_per_task,
        variant_scope=preset.variant_scope,
    )


def manifest_payload_v21(
    manifest: LiberoFreshRolloutManifestV21 | Mapping[str, object],
) -> dict[str, object]:
    validated = validate_rollout_manifest_v21(manifest)
    payload: dict[str, object] = {
        "authority_id": validated.authority_id,
        "budget_fractions": list(validated.budget_fractions),
        "episode_budget_mode": validated.episode_budget_mode,
        "manifest_name": validated.manifest_name,
        "metric_profile": validated.metric_profile,
        "num_trials_per_task": validated.num_trials_per_task,
        "schema_version": validated.schema_version,
        "task_ids": list(validated.task_ids),
        "task_suite_name": validated.task_suite_name,
        "variant_scope": list(validated.variant_scope),
    }
    if validated.seed_manifest is not None:
        payload["seed_manifest"] = list(validated.seed_manifest)
    if validated.per_task_seed_manifest is not None:
        payload["per_task_seed_manifest"] = {
            task_id: list(task_seed_manifest)
            for task_id, task_seed_manifest in validated.per_task_seed_manifest.items()
        }
    if validated.selection_policy is not None:
        payload["selection_policy"] = validated.selection_policy
        payload["selection_source"] = cast(str, validated.selection_source)
        payload["selection_source_hash"] = cast(str, validated.selection_source_hash)
    return payload


def compute_rollout_manifest_hash_v21(
    manifest: LiberoFreshRolloutManifestV21 | Mapping[str, object],
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(manifest_payload_v21(manifest))
    ).hexdigest()


def resolve_tracked_rollout_manifest_path_v21(preset_name: str) -> Path:
    preset = _FROZEN_PRESET_SPECS.get(str(preset_name).strip())
    if preset is None:
        expected = ", ".join(sorted(_FROZEN_PRESET_SPECS))
        raise ValueError(
            f"invalid preset_name {preset_name!r}; expected one of {expected}"
        )
    return MANIFEST_DIR / preset.file_name


def load_rollout_manifest_v21(path: str | Path) -> LiberoFreshRolloutManifestV21:
    payload_object = cast(object, json.loads(Path(path).read_text(encoding="utf-8")))
    if not isinstance(payload_object, Mapping):
        raise ValueError(f"manifest at {path} must be a JSON object")
    payload_mapping = cast(Mapping[object, object], payload_object)
    mapping_payload = {str(key): value for key, value in payload_mapping.items()}
    return validate_rollout_manifest_v21(mapping_payload)


def compute_rollout_manifest_file_hash_v21(path: str | Path) -> str:
    return compute_rollout_manifest_hash_v21(load_rollout_manifest_v21(path))


def write_rollout_manifest_v21(
    path: str | Path,
    manifest: LiberoFreshRolloutManifestV21 | Mapping[str, object],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_payload_v21(manifest)
    _ = output_path.write_text(
        _canonical_json_text(payload, pretty=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def materialize_preset_manifest_v21(*, preset_name: str, output: str | Path) -> Path:
    manifest = build_rollout_manifest_preset_v21(preset_name=preset_name)
    return write_rollout_manifest_v21(output, manifest)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize canonical OpenPI LIBERO v21 manifests"
    )
    _ = parser.add_argument(
        "--preset",
        required=True,
        choices=sorted(_FROZEN_PRESET_SPECS),
        help="Canonical v21 manifest preset to materialize.",
    )
    _ = parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for the materialized manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    preset_name = cast(str, args.preset)
    output_path = Path(cast(str, args.output))
    _ = materialize_preset_manifest_v21(
        preset_name=preset_name,
        output=output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_ALLOWED_VARIANTS",
    "EXPECTED_AUTHORITY_IDS",
    "EXPECTED_BUDGET_FRACTIONS",
    "EXPECTED_EPISODE_BUDGET_MODE",
    "EXPECTED_METRIC_PROFILE",
    "EXPECTED_SCAN_SEED_MANIFEST",
    "EXPECTED_SCAN_TASK_IDS",
    "EXPECTED_SCHEMA_VERSION",
    "EXPECTED_SMOKE_SEED_MANIFEST",
    "EXPECTED_SMOKE_TASK_IDS",
    "EXPECTED_STRONG_NUM_TRIALS",
    "EXPECTED_TASK_SUITE_NAME",
    "LiberoFreshRolloutManifestV21",
    "build_rollout_manifest_preset_v21",
    "build_rollout_manifest_v21",
    "compute_rollout_manifest_file_hash_v21",
    "compute_rollout_manifest_hash_v21",
    "load_rollout_manifest_v21",
    "manifest_payload_v21",
    "materialize_preset_manifest_v21",
    "resolve_tracked_rollout_manifest_path_v21",
    "validate_rollout_manifest_v21",
    "write_rollout_manifest_v21",
]
