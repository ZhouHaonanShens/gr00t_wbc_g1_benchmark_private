#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


from work.recap.advantage import (
    NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
    build_diagnostic_surface_metadata,
)


# =====================
# USER Config (edit)
# =====================

DATASET_DIR_REL = "agent/artifacts/recap_datasets"
MULTIMODAL_ITER_TAG = "recap_mainline_fresh_20260311_121500_k0_t8_local_smoke"
STATE_ONLY_ITER_TAG = "recap_mainline_fresh_20260311_121500_k0_t9_state_only_smoke"
OUTPUT_JSON_REL = "agent/artifacts/vlm_critic_relabel/task9_relabel_audit.json"
ADVANTAGE_TOLERANCE = 1e-6
MAINLINE_SUMMARY_DIR_REL = "agent/artifacts/vlm_critic_relabel"
CANONICAL_T10_OUTPUT_JSON_REL = (
    "agent/artifacts/vlm_critic_relabel/relabel_quality_audit_v1.json"
)


JsonDict = dict[str, Any]
RecordKey = tuple[str, int]

_MULTIMODAL_CRITIC_TYPE = "multimodal_distributional_v1"
CONTINUOUS_ADVANTAGE_DIAGNOSTIC_ROUTE = "continuous_advantage_diagnostic_lane"


@dataclass(frozen=True)
class AuditCase:
    name: str
    dataset_dir: Path
    labels_path: Path
    stats_path: Path
    stats: JsonDict
    labels: list[JsonDict]
    critic_dir: Path | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def _read_json(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return dict(obj)


def _read_jsonl(path: Path) -> list[JsonDict]:
    out: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object in {path} line {line_no}, got {type(obj).__name__}"
                )
            out.append(dict(obj))
    if not out:
        raise ValueError(f"No JSONL records found in {path}")
    return out


def _resolve_dataset_dir(repo_root: Path, dataset_dir_rel: str, iter_tag: str) -> Path:
    dataset_base = Path(dataset_dir_rel)
    dataset_root = (
        dataset_base if dataset_base.is_absolute() else (repo_root / dataset_base)
    )
    candidate = dataset_root / iter_tag
    if not candidate.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {candidate}")
    return candidate


def _resolve_mainline_summary_path(
    repo_root: Path, *, summary_dir_rel: str, iter_tag: str
) -> Path:
    summary_base = Path(summary_dir_rel)
    summary_root = (
        summary_base if summary_base.is_absolute() else (repo_root / summary_base)
    )
    return (summary_root / f"{iter_tag}.json").resolve()


def _resolve_path(repo_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            default=_json_default,
        )
        f.write("\n")


def _as_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected float-like in {context}, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Expected float-like in {context}, got {value!r}"
            ) from exc
    raise ValueError(f"Expected float-like in {context}, got {type(value).__name__}")


def _as_int(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Expected int-like in {context}, got bool")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Expected int-like in {context}, got {value!r}") from exc
    raise ValueError(f"Expected int-like in {context}, got {type(value).__name__}")


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    text = str(value).strip()
    return text or None


def _load_case(
    repo_root: Path, *, dataset_dir_rel: str, iter_tag: str, name: str
) -> AuditCase:
    dataset_dir = _resolve_dataset_dir(repo_root, dataset_dir_rel, iter_tag)
    labels_path = dataset_dir / "m2_labels" / "labels.jsonl"
    stats_path = dataset_dir / "m2_labels" / "stats.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing stats file: {stats_path}")
    stats = _read_json(stats_path)
    labels = _read_jsonl(labels_path)
    critic_dir_raw = stats.get("critic_dir")
    critic_dir = None
    if isinstance(critic_dir_raw, str) and critic_dir_raw.strip():
        critic_dir = Path(critic_dir_raw).expanduser().resolve()
    return AuditCase(
        name=name,
        dataset_dir=dataset_dir,
        labels_path=labels_path,
        stats_path=stats_path,
        stats=stats,
        labels=labels,
        critic_dir=critic_dir,
    )


def _record_key(record: JsonDict) -> RecordKey:
    episode_id = _optional_string(record.get("episode_id"))
    if episode_id is None:
        raise ValueError("Missing episode_id in label record")
    t = _as_int(record.get("t"), context=f"{episode_id}.t")
    return (episode_id, int(t))


def _label_map(labels: list[JsonDict]) -> dict[RecordKey, JsonDict]:
    out: dict[RecordKey, JsonDict] = {}
    for record in labels:
        key = _record_key(record)
        if key in out:
            raise ValueError(f"Duplicate label key detected: {key}")
        out[key] = record
    return out


def _numeric_summary(labels: list[JsonDict], field: str) -> JsonDict:
    values = [_as_float(record.get(field), context=f"{field}") for record in labels]
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / float(len(values))),
    }


def _indicator_summary(labels: list[JsonDict]) -> JsonDict:
    positives = 0
    invalid_values: list[object] = []
    for record in labels:
        value = record.get("indicator_I")
        if value in (0, 1):
            positives += int(value)
            continue
        invalid_values.append(value)
    count = len(labels)
    return {
        "positive_count": int(positives),
        "positive_ratio": float(positives / float(count)),
        "binary_only": not invalid_values,
        "invalid_values_preview": invalid_values[:5],
    }


def _collect_finite_issues(labels: list[JsonDict]) -> list[str]:
    issues: list[str] = []
    numeric_fields = ("return_G", "value_V", "advantage_A", "epsilon_l")
    for index, record in enumerate(labels, start=1):
        for field in numeric_fields:
            value = _as_float(record.get(field), context=f"record#{index}.{field}")
            if not math.isfinite(value):
                issues.append(f"record#{index}:{field}={value!r}")
        indicator = record.get("indicator_I")
        if indicator not in (0, 1):
            issues.append(f"record#{index}:indicator_I={indicator!r}")
    return issues


def _advantage_formula_check(labels: list[JsonDict], tolerance: float) -> JsonDict:
    max_abs_error = 0.0
    bad_examples: list[JsonDict] = []
    for index, record in enumerate(labels, start=1):
        value_v = _as_float(record.get("value_V"), context=f"record#{index}.value_V")
        return_g = _as_float(record.get("return_G"), context=f"record#{index}.return_G")
        advantage_a = _as_float(
            record.get("advantage_A"), context=f"record#{index}.advantage_A"
        )
        error = abs(float(advantage_a) - float(return_g - value_v))
        max_abs_error = max(max_abs_error, float(error))
        if error > float(tolerance) and len(bad_examples) < 5:
            bad_examples.append(
                {
                    "record_index": int(index),
                    "episode_id": _optional_string(record.get("episode_id")),
                    "t": _as_int(record.get("t"), context=f"record#{index}.t"),
                    "return_G": float(return_g),
                    "value_V": float(value_v),
                    "advantage_A": float(advantage_a),
                    "abs_error": float(error),
                }
            )
    return {
        "ok": bool(max_abs_error <= float(tolerance)),
        "tolerance": float(tolerance),
        "max_abs_error": float(max_abs_error),
        "bad_examples": bad_examples,
    }


def _critic_support_from_case(case: AuditCase) -> JsonDict:
    if case.critic_dir is None:
        raise ValueError(f"Case {case.name} missing critic_dir in stats.json")
    config_path = case.critic_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing critic config: {config_path}")
    config = _read_json(config_path)

    if (
        config.get("artifact_version") == _MULTIMODAL_CRITIC_TYPE
        and config.get("critic_type") == _MULTIMODAL_CRITIC_TYPE
    ):
        backend_kind = _MULTIMODAL_CRITIC_TYPE
        support_values = config.get("bin_centers")
    elif all(key in config for key in ("state_dim", "include_t", "bin_centers")):
        backend_kind = "state_only_dist_bins"
        support_values = config.get("bin_centers")
    else:
        raise ValueError(f"Unknown critic backend config for {case.critic_dir}")

    if not isinstance(support_values, list) or not support_values:
        raise ValueError(f"Critic config missing non-empty bin_centers: {config_path}")
    support = [
        _as_float(value, context=f"{case.name}.bin_centers") for value in support_values
    ]
    support_min = float(min(support))
    support_max = float(max(support))
    return {
        "backend_kind": backend_kind,
        "support_min": float(support_min),
        "support_max": float(support_max),
        "config": config,
    }


def _range_check(case: AuditCase, support_min: float, support_max: float) -> JsonDict:
    value_issues: list[str] = []
    advantage_issues: list[str] = []
    epsilon_issues: list[str] = []
    indicator_issues: list[str] = []

    return_values = [
        _as_float(record.get("return_G"), context="return_G") for record in case.labels
    ]
    return_min = float(min(return_values))
    return_max = float(max(return_values))
    advantage_min_allowed = float(return_min - support_max)
    advantage_max_allowed = float(return_max - support_min)

    for index, record in enumerate(case.labels, start=1):
        value_v = _as_float(record.get("value_V"), context=f"record#{index}.value_V")
        advantage_a = _as_float(
            record.get("advantage_A"), context=f"record#{index}.advantage_A"
        )
        epsilon_l = _as_float(
            record.get("epsilon_l"), context=f"record#{index}.epsilon_l"
        )
        indicator_i = record.get("indicator_I")

        if not (float(support_min) <= float(value_v) <= float(support_max)):
            value_issues.append(f"record#{index}:value_V={value_v}")
        if not (
            float(advantage_min_allowed)
            <= float(advantage_a)
            <= float(advantage_max_allowed)
        ):
            advantage_issues.append(f"record#{index}:advantage_A={advantage_a}")
        if not (
            float(advantage_min_allowed)
            <= float(epsilon_l)
            <= float(advantage_max_allowed)
        ):
            epsilon_issues.append(f"record#{index}:epsilon_l={epsilon_l}")
        if indicator_i not in (0, 1):
            indicator_issues.append(f"record#{index}:indicator_I={indicator_i!r}")

    return {
        "ok": not (
            value_issues or advantage_issues or epsilon_issues or indicator_issues
        ),
        "interpretation": (
            "Checks value_V against critic bin support, checks advantage_A and epsilon_l "
            "against [min(return_G)-max(value_support), max(return_G)-min(value_support)], "
            "and requires indicator_I to stay binary."
        ),
        "value_support": {
            "min": float(support_min),
            "max": float(support_max),
        },
        "derived_advantage_support": {
            "min": float(advantage_min_allowed),
            "max": float(advantage_max_allowed),
        },
        "issues": {
            "value_V": value_issues[:5],
            "advantage_A": advantage_issues[:5],
            "epsilon_l": epsilon_issues[:5],
            "indicator_I": indicator_issues[:5],
        },
    }


def _presence_entry(*, value: object, source: str, required: bool) -> JsonDict:
    present = value is not None
    return {
        "required": bool(required),
        "present": bool(present),
        "value": value,
        "source": source,
    }


def _multimodal_provenance(case: AuditCase) -> JsonDict:
    if case.critic_dir is None:
        raise ValueError("multimodal case missing critic_dir")
    critic_dir = case.critic_dir
    config = _read_json(critic_dir / "config.json")
    processor_config = _read_json(critic_dir / "processor" / "processor_config.json")
    provenance = _read_json(critic_dir / "provenance.json")
    split_manifest_ref = _read_json(critic_dir / "split_manifest_ref.json")

    fields = {
        "contract_version": _presence_entry(
            value=(
                _optional_string(provenance.get("contract_version"))
                or _optional_string(config.get("contract_version"))
                or _optional_string(processor_config.get("contract_version"))
                or _optional_string(split_manifest_ref.get("contract_version"))
            ),
            source="provenance/config/processor_config/split_manifest_ref",
            required=False,
        ),
        "task_text_field": _presence_entry(
            value=_optional_string(processor_config.get("task_text_field")),
            source="processor/processor_config.json",
            required=True,
        ),
        "value_source": _presence_entry(
            value=_optional_string(case.stats.get("value_source")),
            source="m2_labels/stats.json",
            required=True,
        ),
        "critic_type": _presence_entry(
            value=_optional_string(config.get("critic_type")),
            source="config.json",
            required=True,
        ),
        "critic_version": _presence_entry(
            value=_optional_string(config.get("artifact_version")),
            source="config.json",
            required=True,
        ),
        "upgrade_pending": _presence_entry(
            value=(
                _optional_string(config.get("upgrade_pending"))
                or _optional_string(provenance.get("upgrade_pending"))
            ),
            source="config.json/provenance.json",
            required=False,
        ),
        "base_model": _presence_entry(
            value=(
                _optional_string(config.get("base_model"))
                or _optional_string(provenance.get("base_model"))
            ),
            source="config.json/provenance.json",
            required=True,
        ),
        "split_manifest_ref": _presence_entry(
            value=split_manifest_ref,
            source="split_manifest_ref.json",
            required=True,
        ),
    }
    required_ok = all(
        bool(entry["present"]) for entry in fields.values() if bool(entry["required"])
    )
    return {
        "ok": bool(required_ok),
        "note": "Task 9 requires multimodal relabel provenance completeness; contract_version and upgrade_pending remain optional when not materialized.",
        "critic_dir": str(critic_dir),
        "fields": fields,
    }


def _state_only_provenance(case: AuditCase, backend_kind: str) -> JsonDict:
    if case.critic_dir is None:
        raise ValueError("state-only case missing critic_dir")
    critic_dir = case.critic_dir
    config = _read_json(critic_dir / "config.json")
    metrics_path = critic_dir / "metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.is_file() else {}
    fields = {
        "contract_version": _presence_entry(
            value=None,
            source="legacy state-only artifact has no explicit contract_version field",
            required=False,
        ),
        "task_text_field": _presence_entry(
            value=None,
            source="legacy state-only artifact has no processor_config/task_text_field",
            required=False,
        ),
        "value_source": _presence_entry(
            value=_optional_string(case.stats.get("value_source")),
            source="m2_labels/stats.json",
            required=True,
        ),
        "critic_type": _presence_entry(
            value=None,
            source="legacy state-only artifact has no critic_type field; backend kind is reported separately",
            required=False,
        ),
        "critic_version": _presence_entry(
            value=_optional_string(config.get("artifact_version")),
            source="legacy state-only config.json (missing in this artifact)",
            required=False,
        ),
        "upgrade_pending": _presence_entry(
            value=None,
            source="legacy state-only artifact has no upgrade_pending field",
            required=False,
        ),
        "base_model": _presence_entry(
            value=None,
            source="legacy state-only artifact has no base_model field",
            required=False,
        ),
        "split_manifest_ref": _presence_entry(
            value=None,
            source="legacy state-only artifact has no split_manifest_ref.json",
            required=False,
        ),
    }
    return {
        "ok": True,
        "note": (
            "Legacy state-only baseline is intentionally audited as a comparison case; missing provenance "
            "fields are reported explicitly instead of synthesized."
        ),
        "critic_dir": str(critic_dir),
        "backend_kind": backend_kind,
        "legacy_context": {
            "iter_tag": _optional_string(config.get("iter_tag")),
            "git_sha": _optional_string(config.get("git_sha")),
            "split_mode": _optional_string(config.get("split_mode")),
            "return_min": metrics.get("return_min"),
            "return_max": metrics.get("return_max"),
        },
        "fields": fields,
    }


def _case_summary(
    case: AuditCase, backend_kind: str, support_min: float, support_max: float
) -> JsonDict:
    return {
        "name": case.name,
        "dataset_dir": str(case.dataset_dir),
        "labels_path": str(case.labels_path),
        "stats_path": str(case.stats_path),
        "critic_dir": str(case.critic_dir) if case.critic_dir is not None else None,
        "critic_backend": backend_kind,
        "n_labels": int(len(case.labels)),
        "stats": case.stats,
        "numeric_summary": {
            "return_G": _numeric_summary(case.labels, "return_G"),
            "value_V": _numeric_summary(case.labels, "value_V"),
            "advantage_A": _numeric_summary(case.labels, "advantage_A"),
            "epsilon_l": _numeric_summary(case.labels, "epsilon_l"),
            "indicator_I": _indicator_summary(case.labels),
        },
        "critic_value_support": {
            "min": float(support_min),
            "max": float(support_max),
        },
    }


def _delta_summary(multimodal_case: AuditCase, state_only_case: AuditCase) -> JsonDict:
    multimodal_map = _label_map(multimodal_case.labels)
    state_only_map = _label_map(state_only_case.labels)
    multimodal_keys = set(multimodal_map.keys())
    state_only_keys = set(state_only_map.keys())
    shared_keys = sorted(multimodal_keys & state_only_keys)
    missing_in_multimodal = sorted(state_only_keys - multimodal_keys)
    missing_in_state_only = sorted(multimodal_keys - state_only_keys)

    return_mismatches = 0
    prompt_mismatches = 0
    value_diffs: list[float] = []
    advantage_diffs: list[float] = []
    indicator_diffs = 0

    for key in shared_keys:
        multimodal_record = multimodal_map[key]
        state_only_record = state_only_map[key]
        if (
            abs(
                _as_float(
                    multimodal_record.get("return_G"), context="multimodal.return_G"
                )
                - _as_float(
                    state_only_record.get("return_G"), context="state_only.return_G"
                )
            )
            > 0.0
        ):
            return_mismatches += 1
        if _optional_string(multimodal_record.get("prompt_raw")) != _optional_string(
            state_only_record.get("prompt_raw")
        ):
            prompt_mismatches += 1

        value_diffs.append(
            _as_float(multimodal_record.get("value_V"), context="multimodal.value_V")
            - _as_float(state_only_record.get("value_V"), context="state_only.value_V")
        )
        advantage_diffs.append(
            _as_float(
                multimodal_record.get("advantage_A"), context="multimodal.advantage_A"
            )
            - _as_float(
                state_only_record.get("advantage_A"), context="state_only.advantage_A"
            )
        )
        if multimodal_record.get("indicator_I") != state_only_record.get("indicator_I"):
            indicator_diffs += 1

    def _diff_stats(values: list[float]) -> JsonDict:
        if not values:
            return {
                "count": 0,
                "mean_signed": None,
                "mean_abs": None,
                "max_abs": None,
            }
        return {
            "count": int(len(values)),
            "mean_signed": float(sum(values) / float(len(values))),
            "mean_abs": float(sum(abs(v) for v in values) / float(len(values))),
            "max_abs": float(max(abs(v) for v in values)),
        }

    multimodal_pos_ratio = _as_float(
        multimodal_case.stats.get("pos_ratio"), context="multimodal.pos_ratio"
    )
    state_only_pos_ratio = _as_float(
        state_only_case.stats.get("pos_ratio"), context="state_only.pos_ratio"
    )
    multimodal_epsilon = _as_float(
        multimodal_case.stats.get("epsilon_value"), context="multimodal.epsilon_value"
    )
    state_only_epsilon = _as_float(
        state_only_case.stats.get("epsilon_value"), context="state_only.epsilon_value"
    )

    alignment_ok = not (
        missing_in_multimodal
        or missing_in_state_only
        or return_mismatches
        or prompt_mismatches
    )
    return {
        "ok": bool(alignment_ok),
        "reference_case": state_only_case.name,
        "compared_case": multimodal_case.name,
        "shared_record_alignment": {
            "ok": bool(alignment_ok),
            "shared_count": int(len(shared_keys)),
            "missing_in_multimodal": [list(item) for item in missing_in_multimodal[:5]],
            "missing_in_state_only": [list(item) for item in missing_in_state_only[:5]],
            "return_G_mismatch_count": int(return_mismatches),
            "prompt_raw_mismatch_count": int(prompt_mismatches),
        },
        "value_V_delta_multimodal_minus_state_only": _diff_stats(value_diffs),
        "advantage_A_delta_multimodal_minus_state_only": _diff_stats(advantage_diffs),
        "indicator_disagreement": {
            "count": int(indicator_diffs),
            "ratio": float(indicator_diffs / float(len(shared_keys)))
            if shared_keys
            else None,
        },
        "stats_delta": {
            "epsilon_value_delta": float(multimodal_epsilon - state_only_epsilon),
            "pos_ratio_delta": float(multimodal_pos_ratio - state_only_pos_ratio),
            "advantage_mean_delta": float(
                _as_float(
                    multimodal_case.stats.get("advantage_mean"),
                    context="multimodal.advantage_mean",
                )
                - _as_float(
                    state_only_case.stats.get("advantage_mean"),
                    context="state_only.advantage_mean",
                )
            ),
            "advantage_min_delta": float(
                _as_float(
                    multimodal_case.stats.get("advantage_min"),
                    context="multimodal.advantage_min",
                )
                - _as_float(
                    state_only_case.stats.get("advantage_min"),
                    context="state_only.advantage_min",
                )
            ),
            "advantage_max_delta": float(
                _as_float(
                    multimodal_case.stats.get("advantage_max"),
                    context="multimodal.advantage_max",
                )
                - _as_float(
                    state_only_case.stats.get("advantage_max"),
                    context="state_only.advantage_max",
                )
            ),
        },
    }


def _advantage_input_range_note() -> JsonDict:
    return {
        "available": False,
        "ok": None,
        "note": (
            "Current Task 9 inputs expose M2 labels and m2_labels/stats.json only; neither materializes "
            "advantage_input or an advantage_input range summary at this stage."
        ),
    }


def _advantage_input_range_from_summary(summary: JsonDict | None) -> JsonDict:
    if summary is None:
        return {
            "available": False,
            "ok": False,
            "note": "Diagnostic relabel summary JSON not found for single-case audit.",
        }

    range_obj = summary.get("advantage_input_range")
    if not isinstance(range_obj, dict):
        return {
            "available": False,
            "ok": False,
            "note": "Diagnostic relabel summary is missing advantage_input_range.",
        }

    min_raw = range_obj.get("min")
    max_raw = range_obj.get("max")
    clip_min_raw = range_obj.get("clip_min", -1.0)
    clip_max_raw = range_obj.get("clip_max", 1.0)
    try:
        min_value = _as_float(min_raw, context="advantage_input_range.min")
        max_value = _as_float(max_raw, context="advantage_input_range.max")
        clip_min = _as_float(clip_min_raw, context="advantage_input_range.clip_min")
        clip_max = _as_float(clip_max_raw, context="advantage_input_range.clip_max")
    except ValueError as exc:
        return {
            "available": True,
            "ok": False,
            "note": str(exc),
            "summary_preview": range_obj,
        }

    contract_version = _optional_string(summary.get("advantage_contract_version"))
    default_mainline = _optional_string(summary.get("default_mainline"))
    diagnostic_only = summary.get("diagnostic_only")
    mainline_authority = summary.get("mainline_authority")
    authority_scope = _optional_string(summary.get("authority_scope"))
    ok = bool(
        clip_min <= min_value <= clip_max
        and clip_min <= max_value <= clip_max
        and default_mainline == CONTINUOUS_ADVANTAGE_DIAGNOSTIC_ROUTE
        and contract_version
        in {"full_recap_continuous_adv_v1", "full_recap_continuous_adv_v2"}
        and diagnostic_only is True
        and mainline_authority is False
        and authority_scope == NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE
    )
    return {
        "available": True,
        "ok": bool(ok),
        "default_mainline": default_mainline,
        "advantage_contract_version": contract_version,
        "diagnostic_only": diagnostic_only,
        "mainline_authority": mainline_authority,
        "authority_scope": authority_scope,
        "range": {
            "min": float(min_value),
            "max": float(max_value),
            "clip_min": float(clip_min),
            "clip_max": float(clip_max),
        },
        "continuous_package": summary.get("continuous_package"),
        "threshold_packages": summary.get("threshold_packages"),
    }


def _threshold_monotonicity_from_summary(summary: JsonDict | None) -> JsonDict:
    if summary is None:
        return {
            "available": False,
            "ok": False,
            "note": "Diagnostic relabel summary JSON not found for threshold monotonicity audit.",
        }

    packages = summary.get("threshold_packages")
    if not isinstance(packages, dict):
        return {
            "available": False,
            "ok": False,
            "note": "Diagnostic relabel summary is missing threshold_packages.",
        }

    threshold_names = ("epsilon_10", "epsilon_20", "epsilon_30", "epsilon_40")
    ratios: list[float] = []
    epsilons: list[float] = []
    issues: list[str] = []
    observed_positive_ratio: JsonDict = {}
    epsilon_values: JsonDict = {}

    for name in threshold_names:
        package = packages.get(name)
        if not isinstance(package, dict):
            issues.append(f"Missing threshold package: {name}")
            continue
        try:
            ratio_value = _as_float(
                package.get("observed_positive_ratio"),
                context=f"threshold_packages.{name}.observed_positive_ratio",
            )
            epsilon_value = _as_float(
                package.get("epsilon_value"),
                context=f"threshold_packages.{name}.epsilon_value",
            )
        except ValueError as exc:
            issues.append(str(exc))
            continue
        ratios.append(float(ratio_value))
        epsilons.append(float(epsilon_value))
        observed_positive_ratio[name] = float(ratio_value)
        epsilon_values[name] = float(epsilon_value)

    ratio_strictly_increasing = len(ratios) == len(threshold_names) and all(
        earlier < later for earlier, later in zip(ratios, ratios[1:])
    )
    epsilon_value_strictly_decreasing = len(epsilons) == len(threshold_names) and all(
        earlier > later for earlier, later in zip(epsilons, epsilons[1:])
    )
    ok = not issues and (ratio_strictly_increasing or epsilon_value_strictly_decreasing)

    return {
        "available": True,
        "ok": bool(ok),
        "threshold_order": list(threshold_names),
        "observed_positive_ratio": observed_positive_ratio,
        "epsilon_values": epsilon_values,
        "checks": {
            "observed_positive_ratio_strictly_increasing": bool(
                ratio_strictly_increasing
            ),
            "epsilon_value_strictly_decreasing": bool(
                epsilon_value_strictly_decreasing
            ),
        },
        "issues": issues,
        "note": (
            "Passes when threshold packages preserve monotone target behavior: either observed "
            "positive ratios rise from epsilon_10 to epsilon_40, or epsilon values decrease "
            "monotonically across the same order."
        ),
    }


def _check_ok(checks: JsonDict, name: str) -> bool:
    check = checks.get(name)
    if not isinstance(check, dict) or "ok" not in check:
        raise ValueError(f"Audit check {name!r} missing required ok field")
    return bool(check["ok"])


def _plan_facing_summary(checks: JsonDict, *, gate_passed: bool) -> JsonDict:
    return {
        "all_values_finite": _check_ok(checks, "all_values_finite"),
        "advantage_semantic_check": (
            "passed" if _check_ok(checks, "advantage_formula_ok") else "failed"
        ),
        "threshold_monotonicity_passed": _check_ok(checks, "threshold_monotonicity"),
        "provenance_complete": _check_ok(checks, "provenance_required_fields_present"),
        "gate_passed": bool(gate_passed),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="45b_vlm_critic_relabel_audit.py",
        description=(
            "Audit multimodal relabel outputs against an equivalent state-only relabel subset, "
            "checking numeric sanity, advantage provenance, and summary deltas."
        ),
    )
    parser.add_argument("--dataset-dir-rel", type=str, default=str(DATASET_DIR_REL))
    parser.add_argument(
        "--iter-tag",
        type=str,
        default="",
        help=(
            "Single-case audit mode for a diagnostic relabel package. "
            "When provided, audit agent/artifacts/recap_datasets/<iter-tag> and "
            "agent/artifacts/vlm_critic_relabel/<iter-tag>.json."
        ),
    )
    parser.add_argument(
        "--mainline-summary-dir-rel",
        type=str,
        default=str(MAINLINE_SUMMARY_DIR_REL),
    )
    parser.add_argument(
        "--multimodal-iter-tag", type=str, default=str(MULTIMODAL_ITER_TAG)
    )
    parser.add_argument(
        "--state-only-iter-tag", type=str, default=str(STATE_ONLY_ITER_TAG)
    )
    parser.add_argument("--output-json", type=str, default=str(OUTPUT_JSON_REL))
    parser.add_argument(
        "--advantage-tolerance",
        type=float,
        default=float(ADVANTAGE_TOLERANCE),
        help="Absolute tolerance for checking advantage_A == return_G - value_V.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    repo_root = _repo_root()
    output_path = _resolve_path(repo_root, str(args.output_json))

    single_iter_tag = str(getattr(args, "iter_tag", "") or "").strip()
    if single_iter_tag:
        single_case = _load_case(
            repo_root,
            dataset_dir_rel=str(args.dataset_dir_rel),
            iter_tag=single_iter_tag,
            name="continuous_advantage_diagnostic",
        )
        mainline_summary_path = _resolve_mainline_summary_path(
            repo_root,
            summary_dir_rel=str(args.mainline_summary_dir_rel),
            iter_tag=single_iter_tag,
        )
        mainline_summary = (
            _read_json(mainline_summary_path)
            if mainline_summary_path.is_file()
            else None
        )

        single_support = _critic_support_from_case(single_case)
        single_formula = _advantage_formula_check(
            single_case.labels, float(args.advantage_tolerance)
        )
        single_finite_issues = _collect_finite_issues(single_case.labels)
        single_range = _range_check(
            single_case,
            support_min=_as_float(
                single_support["support_min"], context="single.support_min"
            ),
            support_max=_as_float(
                single_support["support_max"], context="single.support_max"
            ),
        )
        single_provenance = _multimodal_provenance(single_case)
        advantage_input_range_ok = _advantage_input_range_from_summary(mainline_summary)
        threshold_monotonicity = _threshold_monotonicity_from_summary(mainline_summary)
        critic_backend_check = {
            "ok": bool(single_support["backend_kind"] == _MULTIMODAL_CRITIC_TYPE),
            "expected": _MULTIMODAL_CRITIC_TYPE,
            "actual": single_support["backend_kind"],
        }
        all_values_finite = {
            "ok": not single_finite_issues,
            "cases": {
                single_case.name: {
                    "ok": not single_finite_issues,
                    "issues": single_finite_issues[:5],
                }
            },
        }
        all_values_in_range = {
            "ok": bool(single_range["ok"]),
            "cases": {single_case.name: single_range},
        }
        advantage_formula_ok = {
            "ok": bool(single_formula["ok"]),
            "cases": {single_case.name: single_formula},
        }
        provenance_required_fields_present = {
            "ok": bool(single_provenance["ok"]),
            "required_scope": single_case.name,
            "cases": {single_case.name: single_provenance},
        }
        overall_ok = all(
            (
                bool(all_values_finite["ok"]),
                bool(all_values_in_range["ok"]),
                bool(critic_backend_check["ok"]),
                bool(advantage_formula_ok["ok"]),
                bool(provenance_required_fields_present["ok"]),
                bool(advantage_input_range_ok.get("ok")),
                bool(threshold_monotonicity["ok"]),
            )
        )
        checks = {
            "all_values_finite": all_values_finite,
            "all_values_in_range": all_values_in_range,
            "critic_backend": critic_backend_check,
            "advantage_formula_ok": advantage_formula_ok,
            "advantage_input_range_ok": advantage_input_range_ok,
            "threshold_monotonicity": threshold_monotonicity,
            "provenance_required_fields_present": provenance_required_fields_present,
        }
        plan_facing_summary = _plan_facing_summary(checks, gate_passed=overall_ok)
        canonical_output_path = _resolve_path(repo_root, CANONICAL_T10_OUTPUT_JSON_REL)
        audit_json = {
            "task": "T10 diagnostic relabel quality audit",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "worktree": str(repo_root),
            "inputs": {
                "dataset_dir_rel": str(args.dataset_dir_rel),
                "iter_tag": single_iter_tag,
                "mainline_summary_path": str(mainline_summary_path),
            },
            "outputs": {
                "requested_output_json": str(output_path),
                "canonical_t10_output_json": str(canonical_output_path),
            },
            "cases": {
                single_case.name: _case_summary(
                    single_case,
                    backend_kind=str(single_support["backend_kind"]),
                    support_min=_as_float(
                        single_support["support_min"], context="single.support_min"
                    ),
                    support_max=_as_float(
                        single_support["support_max"], context="single.support_max"
                    ),
                )
            },
            "checks": checks,
            **plan_facing_summary,
            "pass": bool(overall_ok),
        }
        audit_json.update(
            build_diagnostic_surface_metadata(
                surface_route="vlm_critic_relabel_audit_diagnostic",
                authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
                surface_kind="vlm_critic_relabel_audit",
            )
        )
        _write_json(output_path, audit_json)
        _write_json(canonical_output_path, audit_json)
        if not overall_ok:
            print(f"RELABEL_AUDIT_FAIL: {output_path}")
            return 1
        print(f"RELABEL_AUDIT_OK: {output_path}")
        return 0

    multimodal_case = _load_case(
        repo_root,
        dataset_dir_rel=str(args.dataset_dir_rel),
        iter_tag=str(args.multimodal_iter_tag),
        name="multimodal_positive",
    )
    state_only_case = _load_case(
        repo_root,
        dataset_dir_rel=str(args.dataset_dir_rel),
        iter_tag=str(args.state_only_iter_tag),
        name="state_only_comparison",
    )

    multimodal_support = _critic_support_from_case(multimodal_case)
    state_only_support = _critic_support_from_case(state_only_case)

    multimodal_formula = _advantage_formula_check(
        multimodal_case.labels, float(args.advantage_tolerance)
    )
    state_only_formula = _advantage_formula_check(
        state_only_case.labels, float(args.advantage_tolerance)
    )
    multimodal_finite_issues = _collect_finite_issues(multimodal_case.labels)
    state_only_finite_issues = _collect_finite_issues(state_only_case.labels)
    multimodal_range = _range_check(
        multimodal_case,
        support_min=_as_float(
            multimodal_support["support_min"], context="multimodal.support_min"
        ),
        support_max=_as_float(
            multimodal_support["support_max"], context="multimodal.support_max"
        ),
    )
    state_only_range = _range_check(
        state_only_case,
        support_min=_as_float(
            state_only_support["support_min"], context="state_only.support_min"
        ),
        support_max=_as_float(
            state_only_support["support_max"], context="state_only.support_max"
        ),
    )
    multimodal_provenance = _multimodal_provenance(multimodal_case)
    state_only_provenance = _state_only_provenance(
        state_only_case,
        backend_kind=str(state_only_support["backend_kind"]),
    )
    delta_summary = _delta_summary(multimodal_case, state_only_case)

    critic_backend_check = {
        "ok": bool(
            multimodal_support["backend_kind"] == _MULTIMODAL_CRITIC_TYPE
            and state_only_support["backend_kind"] == "state_only_dist_bins"
        ),
        "expected": {
            "multimodal_positive": _MULTIMODAL_CRITIC_TYPE,
            "state_only_comparison": "state_only_dist_bins",
        },
        "actual": {
            "multimodal_positive": multimodal_support["backend_kind"],
            "state_only_comparison": state_only_support["backend_kind"],
        },
    }

    all_values_finite = {
        "ok": not (multimodal_finite_issues or state_only_finite_issues),
        "cases": {
            multimodal_case.name: {
                "ok": not multimodal_finite_issues,
                "issues": multimodal_finite_issues[:5],
            },
            state_only_case.name: {
                "ok": not state_only_finite_issues,
                "issues": state_only_finite_issues[:5],
            },
        },
    }
    all_values_in_range = {
        "ok": bool(multimodal_range["ok"] and state_only_range["ok"]),
        "cases": {
            multimodal_case.name: multimodal_range,
            state_only_case.name: state_only_range,
        },
    }
    advantage_formula_ok = {
        "ok": bool(multimodal_formula["ok"] and state_only_formula["ok"]),
        "cases": {
            multimodal_case.name: multimodal_formula,
            state_only_case.name: state_only_formula,
        },
    }
    provenance_required_fields_present = {
        "ok": bool(multimodal_provenance["ok"]),
        "required_scope": "multimodal_positive",
        "cases": {
            multimodal_case.name: multimodal_provenance,
            state_only_case.name: state_only_provenance,
        },
    }
    advantage_input_range_ok = _advantage_input_range_note()
    threshold_monotonicity = _threshold_monotonicity_from_summary(None)

    overall_ok = all(
        (
            bool(all_values_finite["ok"]),
            bool(all_values_in_range["ok"]),
            bool(critic_backend_check["ok"]),
            bool(advantage_formula_ok["ok"]),
            bool(provenance_required_fields_present["ok"]),
            bool(delta_summary["ok"]),
        )
    )
    checks = {
        "all_values_finite": all_values_finite,
        "all_values_in_range": all_values_in_range,
        "critic_backend": critic_backend_check,
        "advantage_formula_ok": advantage_formula_ok,
        "advantage_input_range_ok": advantage_input_range_ok,
        "threshold_monotonicity": threshold_monotonicity,
        "provenance_required_fields_present": provenance_required_fields_present,
        "delta_summary_vs_state_only": delta_summary,
    }
    plan_facing_summary = _plan_facing_summary(checks, gate_passed=overall_ok)

    audit_json: JsonDict = {
        "task": "T9 diagnostic relabel compatibility / advantage provenance audit",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "worktree": str(repo_root),
        "inputs": {
            "dataset_dir_rel": str(args.dataset_dir_rel),
            "multimodal_iter_tag": str(args.multimodal_iter_tag),
            "state_only_iter_tag": str(args.state_only_iter_tag),
        },
        "cases": {
            multimodal_case.name: _case_summary(
                multimodal_case,
                backend_kind=str(multimodal_support["backend_kind"]),
                support_min=_as_float(
                    multimodal_support["support_min"], context="multimodal.support_min"
                ),
                support_max=_as_float(
                    multimodal_support["support_max"], context="multimodal.support_max"
                ),
            ),
            state_only_case.name: _case_summary(
                state_only_case,
                backend_kind=str(state_only_support["backend_kind"]),
                support_min=_as_float(
                    state_only_support["support_min"], context="state_only.support_min"
                ),
                support_max=_as_float(
                    state_only_support["support_max"], context="state_only.support_max"
                ),
            ),
        },
        "checks": checks,
        **plan_facing_summary,
        "pass": bool(overall_ok),
    }
    audit_json.update(
        build_diagnostic_surface_metadata(
            surface_route="vlm_critic_relabel_audit_diagnostic",
            authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
            surface_kind="vlm_critic_relabel_audit",
        )
    )

    _write_json(output_path, audit_json)

    if not overall_ok:
        print(f"RELABEL_AUDIT_FAIL: {output_path}")
        return 1

    print(f"RELABEL_AUDIT_OK: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
