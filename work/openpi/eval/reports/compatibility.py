from __future__ import annotations

from collections.abc import Mapping, Sequence
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.eval.protocols.tracked_gate import (  # noqa: E402
    build_rollout_eval_manifest_v2,
    manifest_payload_v2,
    resolve_tracked_rollout_eval_manifest_path,
)
from work.openpi.eval.workflows.tracked_gate import (  # noqa: E402
    load_rollout_eval_v2_authority_bundle,
)


PAIRED_SUMMARY_SCHEMA_VERSION = "openpi_libero_paired_rollout_summary_abc_v2"
GO_NO_GO_REPORT_SCHEMA_VERSION = "openpi_libero_go_no_go_report_v2"
EXPECTED_EVAL_AUTHORITY = "fresh_rollout_v2"
EXPECTED_DATASET_FINGERPRINT_SCHEMA = "openpi_libero_relabel_dataset_fingerprint_v1"
EXPECTED_DATASET_ROUTE_ID = "official_native_8d_recap_relabels_v1"
EXPECTED_DATASET_STATE_SHAPE = [8]
EXPECTED_DATASET_ACTION_SHAPE = [7]
PAIRWISE_BOOTSTRAP_ITERATIONS = 2000
PAIRWISE_CONFIDENCE_LEVEL = 0.95
ARTIFACT_TOPIC_DIR = REPO_ROOT / "agent/artifacts/openpi_libero_v2"
ROLLOUTS_ROOT = ARTIFACT_TOPIC_DIR / "rollouts"
DATASET_ROOT = (
    REPO_ROOT
    / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1"
)
DATASET_FINGERPRINT_PATH = DATASET_ROOT / "meta/dataset_fingerprint.json"
EPISODE_UNIVERSE_HASH_PATH = DATASET_ROOT / "meta/episode_universe_hash.txt"
SCOPE_AUDIT_MANIFEST_NAME = "scope_audit_manifest.txt"
PAIRED_SUMMARY_NAME = "paired_rollout_summary_abc_v2.json"
GO_NO_GO_REPORT_NAME = "go_no_go_report_v2.json"
DEFAULT_RELEVANT_SCOPE_DOCS = (
    REPO_ROOT / "agent/exchange/openpi_libero_results.md",
    REPO_ROOT / "agent/exchange/openpi_libero_rollout_eval_v2_contract.md",
)
FORBIDDEN_SCOPE_KEYWORDS = (
    "G1 migration",
    "state transfer",
    "online loop",
    "RL token",
    "next-state head",
    "human correction UI",
)
VARIANT_ORDER = (
    ("A", "stock_libero_ref_v1"),
    ("B", "fixedadv_relabel8d_control_v1"),
    ("C", "recap_only_relabel8d_v2"),
)
VARIANT_TO_ARTIFACT_NAME = dict(VARIANT_ORDER)
NON_STOCK_VARIANTS = ("B", "C")
EXPECTED_GATE_MANIFEST_NAME = "rollout_lite_v2"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"expected JSON object at {path}, got {type(payload).__name__}"
        )
    return cast(dict[str, object], payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _require_mapping(raw: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


def _require_sequence(raw: object, *, context: str) -> Sequence[object]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{context} must be a sequence, got {type(raw).__name__}")
    return cast(Sequence[object], raw)


def _sequence_as_list(raw: object, *, context: str) -> list[object]:
    return list(_require_sequence(raw, context=context))


def _require_string(raw: object, *, context: str) -> str:
    value = str(raw).strip()
    if not value:
        raise ValueError(f"{context} must be non-empty")
    return value


def _int_value(raw: object, *, context: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{context} must be int-like, got bool")
    try:
        return int(cast(int | float | str, raw))
    except Exception as exc:
        raise ValueError(f"{context} must be int-like, got {raw!r}") from exc


def _float_value(raw: object, *, context: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"{context} must be float-like, got bool")
    try:
        return float(cast(int | float | str, raw))
    except Exception as exc:
        raise ValueError(f"{context} must be float-like, got {raw!r}") from exc


def _optional_float(raw: object) -> float | None:
    if raw is None:
        return None
    return _float_value(raw, context="optional_float")


def _bool_success(raw: object, *, context: str) -> int:
    if isinstance(raw, bool):
        return 1 if raw else 0
    numeric = _int_value(raw, context=context)
    if numeric not in (0, 1):
        raise ValueError(f"{context} must be 0/1-like, got {numeric}")
    return numeric


def _round_pp(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, 6)


def _ci95(*, lower: float | None, upper: float | None, unit: str) -> dict[str, object]:
    return {
        "confidence_level": PAIRWISE_CONFIDENCE_LEVEL,
        "lower": lower,
        "upper": upper,
        "unit": unit,
    }


def _bootstrap_quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("bootstrap quantile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    fraction = position - float(lower_index)
    return float(lower_value + (upper_value - lower_value) * fraction)


def _seed_from_material(seed_material: str) -> int:
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _bootstrap_mean_ci(
    samples: Sequence[float], *, seed_material: str
) -> tuple[float, float]:
    if not samples:
        raise ValueError("bootstrap samples cannot be empty")
    rng = random.Random(_seed_from_material(seed_material))
    population = [float(value) for value in samples]
    count = len(population)
    means: list[float] = []
    for _ in range(PAIRWISE_BOOTSTRAP_ITERATIONS):
        draw_sum = 0.0
        for _ in range(count):
            draw_sum += population[rng.randrange(count)]
        means.append(draw_sum / float(count))
    means.sort()
    alpha = 1.0 - PAIRWISE_CONFIDENCE_LEVEL
    lower = _bootstrap_quantile(means, alpha / 2.0)
    upper = _bootstrap_quantile(means, 1.0 - alpha / 2.0)
    return lower, upper


def _run_git_diff_name_only(*, repo_root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _scan_forbidden_scope_keywords(
    *, doc_paths: Sequence[Path], keywords: Sequence[str]
) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    lowered_keywords = tuple((keyword, keyword.casefold()) for keyword in keywords)
    for doc_path in doc_paths:
        lines = doc_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            lowered_line = line.casefold()
            for keyword, lowered_keyword in lowered_keywords:
                if lowered_keyword not in lowered_line:
                    continue
                hits.append(
                    {
                        "path": str(doc_path.relative_to(REPO_ROOT)),
                        "line": line_number,
                        "keyword": keyword,
                        "text": line.strip(),
                    }
                )
    return hits


def _build_scope_audit_summary(
    *,
    git_diff_paths: Sequence[str],
    relevant_docs: Sequence[Path],
    keyword_hits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    normalized_paths = sorted({path.strip() for path in git_diff_paths if path.strip()})
    touches_submodules_openpi = any(
        path == "submodules/openpi" or path.startswith("submodules/openpi/")
        for path in normalized_paths
    )
    status = "PASS" if (not touches_submodules_openpi and not keyword_hits) else "FAIL"
    return {
        "status": status,
        "git_diff_command": "git diff --name-only",
        "git_diff_path_count": len(normalized_paths),
        "git_diff_paths": normalized_paths,
        "touches_submodules_openpi": touches_submodules_openpi,
        "forbidden_scope_keywords": list(FORBIDDEN_SCOPE_KEYWORDS),
        "relevant_docs": [str(path.relative_to(REPO_ROOT)) for path in relevant_docs],
        "forbidden_keyword_hit_count": len(keyword_hits),
        "forbidden_keyword_hits": [dict(hit) for hit in keyword_hits],
    }


def _scope_audit_manifest_text(scope_summary: Mapping[str, object]) -> str:
    lines = [
        "scope_audit_manifest_v2",
        f"status={scope_summary['status']}",
        f"git_diff_command={scope_summary['git_diff_command']}",
        f"git_diff_path_count={scope_summary['git_diff_path_count']}",
        "git_diff_paths:",
    ]
    diff_paths = cast(Sequence[str], scope_summary["git_diff_paths"])
    if diff_paths:
        lines.extend(f"- {path}" for path in diff_paths)
    else:
        lines.append("- <none>")
    lines.extend(
        [
            f"touches_submodules_openpi={str(scope_summary['touches_submodules_openpi']).lower()}",
            "relevant_docs:",
        ]
    )
    relevant_docs = cast(Sequence[str], scope_summary["relevant_docs"])
    lines.extend(f"- {path}" for path in relevant_docs)
    lines.extend(
        [
            "forbidden_scope_keywords:",
            *(
                f"- {keyword}"
                for keyword in cast(
                    Sequence[str], scope_summary["forbidden_scope_keywords"]
                )
            ),
            f"forbidden_keyword_hit_count={scope_summary['forbidden_keyword_hit_count']}",
            "forbidden_keyword_hits:",
        ]
    )
    keyword_hits = cast(
        Sequence[Mapping[str, object]], scope_summary["forbidden_keyword_hits"]
    )
    if keyword_hits:
        for hit in keyword_hits:
            lines.append(
                "- "
                + f"{hit['path']}:{hit['line']} keyword={hit['keyword']} text={hit['text']}"
            )
    else:
        lines.append("- <none>")
    return "\n".join(lines) + "\n"


def _authority_dir(*, output_root: Path, variant: str, eval_manifest_id: str) -> Path:
    return output_root / "rollouts" / variant / eval_manifest_id


def _checkpoint_root(variant: str) -> Path | None:
    if variant == VARIANT_TO_ARTIFACT_NAME["A"]:
        return None
    return REPO_ROOT / "agent/artifacts/checkpoints/openpi_libero_variants" / variant


def _episode_key(row: Mapping[str, object]) -> tuple[str, int, int, int]:
    return (
        _require_string(
            row.get("task_suite_name", ""), context="episode.task_suite_name"
        ),
        _int_value(row.get("task_id"), context="episode.task_id"),
        _int_value(row.get("seed"), context="episode.seed"),
        _int_value(row.get("trial_index"), context="episode.trial_index"),
    )


def _success_map(
    rows: Sequence[object], *, context: str
) -> dict[tuple[str, int, int, int], int]:
    mapped: dict[tuple[str, int, int, int], int] = {}
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, context=f"{context}[{index}]")
        key = _episode_key(row)
        if key in mapped:
            raise ValueError(f"duplicate paired rollout key {key!r} in {context}")
        mapped[key] = _bool_success(
            row.get("success"), context=f"{context}[{index}].success"
        )
    return mapped


def _build_pairwise_delta(
    *,
    lhs_label: str,
    lhs_variant: str,
    lhs_rows: Sequence[object],
    rhs_label: str,
    rhs_variant: str,
    rhs_rows: Sequence[object],
    eval_manifest_id: str,
) -> dict[str, object]:
    lhs_success = _success_map(lhs_rows, context=f"{lhs_label}.per_episode_rollouts")
    rhs_success = _success_map(rhs_rows, context=f"{rhs_label}.per_episode_rollouts")
    lhs_keys = set(lhs_success.keys())
    rhs_keys = set(rhs_success.keys())
    if lhs_keys != rhs_keys:
        missing_on_rhs = sorted(lhs_keys - rhs_keys)
        missing_on_lhs = sorted(rhs_keys - lhs_keys)
        raise ValueError(
            "paired rollout scopes do not match: "
            + f"missing_on_rhs={missing_on_rhs[:3]!r} missing_on_lhs={missing_on_lhs[:3]!r}"
        )
    ordered_keys = sorted(lhs_keys)
    diffs = [float(lhs_success[key] - rhs_success[key]) for key in ordered_keys]
    point_estimate = sum(diffs) / float(len(diffs))
    delta_success_count = int(sum(int(value) for value in diffs))
    delta_failure_count = -delta_success_count
    seed_material = (
        f"{lhs_variant}:{rhs_variant}:{eval_manifest_id}:paired_success_delta"
    )
    ci_lower, ci_upper = _bootstrap_mean_ci(diffs, seed_material=seed_material)
    return {
        "lhs": lhs_label,
        "rhs": rhs_label,
        "lhs_variant": lhs_variant,
        "rhs_variant": rhs_variant,
        "sample_size": len(diffs),
        "matching_key_fields": ["task_suite_name", "task_id", "seed", "trial_index"],
        "point_estimate": _round_pp(point_estimate),
        "unit": "pp",
        "delta_success_rate": point_estimate,
        "delta_success_count": delta_success_count,
        "delta_failure_count": delta_failure_count,
        "ci95": _ci95(
            lower=_round_pp(ci_lower),
            upper=_round_pp(ci_upper),
            unit="pp",
        ),
        "bootstrap_iterations": PAIRWISE_BOOTSTRAP_ITERATIONS,
        "deterministic_seed_material": seed_material,
    }


def _load_variant_authority(
    *, output_root: Path, label: str, variant: str, eval_manifest_id: str
) -> dict[str, object]:
    authority_dir = _authority_dir(
        output_root=output_root,
        variant=variant,
        eval_manifest_id=eval_manifest_id,
    )
    bundle = load_rollout_eval_v2_authority_bundle(authority_dir)
    summary = _require_mapping(bundle.get("summary", {}), context=f"{label}.summary")
    eval_manifest = _require_mapping(
        bundle.get("eval_manifest", {}), context=f"{label}.eval_manifest"
    )
    bootstrap_ci = _require_mapping(
        bundle.get("bootstrap_ci", {}), context=f"{label}.bootstrap_ci"
    )
    paired_delta = _require_mapping(
        bundle.get("paired_delta", {}), context=f"{label}.paired_delta"
    )
    rollout_summary = _require_mapping(
        summary.get("rollout_summary", {}), context=f"{label}.summary.rollout_summary"
    )
    scope_audit = _require_mapping(
        summary.get("scope_audit", {}), context=f"{label}.summary.scope_audit"
    )
    train_provenance = _require_mapping(
        summary.get("train_provenance", {}), context=f"{label}.summary.train_provenance"
    )
    required_outputs = _require_mapping(
        summary.get("required_outputs", {}), context=f"{label}.summary.required_outputs"
    )
    source_refs: dict[str, str] = {
        "summary": str(authority_dir / "summary.json"),
        "eval_manifest": str(authority_dir / "eval_manifest.json"),
        "bootstrap_ci": str(authority_dir / "bootstrap_ci.json"),
        "paired_delta_vs_stock": str(
            authority_dir / f"paired_delta_vs_{VARIANT_TO_ARTIFACT_NAME['A']}.json"
        ),
        "per_episode_rollouts": str(authority_dir / "per_episode_rollouts.jsonl"),
        "video_index": str(authority_dir / "video_index.json"),
        "deviation_notes": str(authority_dir / "deviation_notes.md"),
    }
    authority_payload: dict[str, object] = {
        "label": label,
        "variant": variant,
        "authority_dir": str(authority_dir),
        "bundle": bundle,
        "summary": dict(summary),
        "eval_manifest": dict(eval_manifest),
        "bootstrap_ci": dict(bootstrap_ci),
        "paired_delta_vs_stock": dict(paired_delta),
        "rollout_summary": dict(rollout_summary),
        "scope_audit": dict(scope_audit),
        "train_provenance": dict(train_provenance),
        "required_outputs": dict(required_outputs),
        "source_refs": source_refs,
    }
    checkpoint_root = _checkpoint_root(variant)
    if checkpoint_root is not None:
        source_refs["train_manifest"] = str(checkpoint_root / "train_manifest.json")
        source_refs["checkpoint_provenance"] = str(
            checkpoint_root / "checkpoint_provenance.json"
        )
    return authority_payload


def _build_manifest_summary(eval_manifest: Mapping[str, object]) -> dict[str, object]:
    task_ids = [
        _int_value(value, context="eval_manifest.task_ids")
        for value in _require_sequence(
            eval_manifest.get("task_ids", []), context="eval_manifest.task_ids"
        )
    ]
    seed_manifest = [
        _int_value(value, context="eval_manifest.seed_manifest")
        for value in _require_sequence(
            eval_manifest.get("seed_manifest", []),
            context="eval_manifest.seed_manifest",
        )
    ]
    trials = _int_value(
        eval_manifest.get("num_trials_per_task"),
        context="eval_manifest.num_trials_per_task",
    )
    expected_episode_count = len(task_ids) * len(seed_manifest) * trials
    return {
        "manifest_name": _require_string(
            eval_manifest.get("manifest_name", ""),
            context="eval_manifest.manifest_name",
        ),
        "eval_manifest_id": _require_string(
            eval_manifest.get("eval_manifest_id", ""),
            context="eval_manifest.eval_manifest_id",
        ),
        "eval_manifest_hash": _require_string(
            eval_manifest.get("eval_manifest_hash", ""),
            context="eval_manifest.eval_manifest_hash",
        ),
        "task_suite_name": _require_string(
            eval_manifest.get("task_suite_name", ""),
            context="eval_manifest.task_suite_name",
        ),
        "task_ids": task_ids,
        "seed_manifest": seed_manifest,
        "num_trials_per_task": trials,
        "expected_episode_count": expected_episode_count,
    }


def _build_dataset_binding_summary(
    *,
    dataset_fingerprint: Mapping[str, object],
    episode_universe_hash: str,
    variant_authorities: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    fingerprint_sha256 = _require_string(
        dataset_fingerprint.get("fingerprint_sha256", ""),
        context="dataset_fingerprint.fingerprint_sha256",
    )
    embedded_universe_hash = _require_string(
        dataset_fingerprint.get("episode_universe_hash", ""),
        context="dataset_fingerprint.episode_universe_hash",
    )
    shape_ok = (
        _sequence_as_list(
            dataset_fingerprint.get("state_shape", []),
            context="dataset_fingerprint.state_shape",
        )
        == EXPECTED_DATASET_STATE_SHAPE
        and _sequence_as_list(
            dataset_fingerprint.get("action_shape", []),
            context="dataset_fingerprint.action_shape",
        )
        == EXPECTED_DATASET_ACTION_SHAPE
    )
    route_ok = str(dataset_fingerprint.get("route_id", "")) == EXPECTED_DATASET_ROUTE_ID
    schema_ok = (
        str(dataset_fingerprint.get("schema_version", ""))
        == EXPECTED_DATASET_FINGERPRINT_SCHEMA
    )
    universe_ok = (
        embedded_universe_hash == episode_universe_hash
        and str(dataset_fingerprint.get("episodes_hash", "")) == episode_universe_hash
    )
    provenance_matches = True
    provenance_rows: dict[str, dict[str, object]] = {}
    for label in NON_STOCK_VARIANTS:
        train_provenance = _require_mapping(
            variant_authorities[label].get("train_provenance", {}),
            context=f"{label}.train_provenance",
        )
        fields = _require_mapping(
            train_provenance.get("fields", {}),
            context=f"{label}.train_provenance.fields",
        )
        field_dataset_fingerprint = str(fields.get("dataset_fingerprint", "")).strip()
        field_episode_universe_hash = str(
            fields.get("episode_universe_hash", "")
        ).strip()
        variant_match = (
            field_dataset_fingerprint == fingerprint_sha256
            and field_episode_universe_hash == episode_universe_hash
        )
        provenance_matches = provenance_matches and variant_match
        provenance_rows[label] = {
            "variant": variant_authorities[label]["variant"],
            "dataset_fingerprint": field_dataset_fingerprint,
            "episode_universe_hash": field_episode_universe_hash,
            "matches_dataset_artifacts": variant_match,
        }
    status = (
        "PASS"
        if (shape_ok and route_ok and schema_ok and universe_ok and provenance_matches)
        else "FAIL"
    )
    return {
        "status": status,
        "dataset_fingerprint_sha256": fingerprint_sha256,
        "episode_universe_hash": episode_universe_hash,
        "shape_ok": shape_ok,
        "route_ok": route_ok,
        "schema_ok": schema_ok,
        "sample_universe_stable": universe_ok,
        "state_shape": _sequence_as_list(
            dataset_fingerprint.get("state_shape", []),
            context="dataset_fingerprint.state_shape",
        ),
        "action_shape": _sequence_as_list(
            dataset_fingerprint.get("action_shape", []),
            context="dataset_fingerprint.action_shape",
        ),
        "route_id": dataset_fingerprint.get("route_id"),
        "total_tasks": dataset_fingerprint.get("total_tasks"),
        "total_episodes": dataset_fingerprint.get("total_episodes"),
        "variant_provenance_alignment": provenance_rows,
        "source_refs": {
            "dataset_fingerprint": str(DATASET_FINGERPRINT_PATH),
            "episode_universe_hash": str(EPISODE_UNIVERSE_HASH_PATH),
        },
    }


def _shared_parity_fields(payload: Mapping[str, object]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in (
        "dataset_fingerprint",
        "episode_universe_hash",
        "base_checkpoint_id",
        "train_budget_id",
        "gate_eval_manifest_hash",
    ):
        fields[key] = _require_string(payload.get(key, ""), context=f"parity.{key}")
    return fields


def _expected_gate_manifest_summary() -> dict[str, object]:
    gate_manifest_path = resolve_tracked_rollout_eval_manifest_path(
        EXPECTED_GATE_MANIFEST_NAME
    )
    gate_manifest = manifest_payload_v2(
        build_rollout_eval_manifest_v2(manifest_name=EXPECTED_GATE_MANIFEST_NAME)
    )
    return {
        "task_ids": _sequence_as_list(
            gate_manifest.get("task_ids", []), context="gate_manifest.task_ids"
        ),
        "seed_manifest": _sequence_as_list(
            gate_manifest.get("seed_manifest", []),
            context="gate_manifest.seed_manifest",
        ),
        "num_trials_per_task": gate_manifest.get("num_trials_per_task"),
        "evaluation_tier": gate_manifest.get("manifest_name"),
        "episode_count": len(
            _sequence_as_list(
                gate_manifest.get("task_ids", []), context="gate_manifest.task_ids"
            )
        )
        * len(
            _sequence_as_list(
                gate_manifest.get("seed_manifest", []),
                context="gate_manifest.seed_manifest",
            )
        )
        * _int_value(
            gate_manifest.get("num_trials_per_task"),
            context="gate_manifest.num_trials_per_task",
        ),
        "gate_eval_manifest_hash": hashlib.sha256(
            gate_manifest_path.read_bytes()
        ).hexdigest(),
    }


def _build_parity_summary(
    *,
    variant_authorities: Mapping[str, Mapping[str, object]],
    manifest_summary: Mapping[str, object],
) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    all_ok = True
    reference_fields: dict[str, str] | None = None
    expected_gate_manifest = _expected_gate_manifest_summary()
    for label in NON_STOCK_VARIANTS:
        authority = _require_mapping(
            variant_authorities.get(label, {}), context=f"variant_authorities.{label}"
        )
        train_provenance = _require_mapping(
            authority.get("train_provenance", {}),
            context=f"variant_authorities.{label}.train_provenance",
        )
        fields = _require_mapping(
            train_provenance.get("fields", {}),
            context=f"variant_authorities.{label}.train_provenance.fields",
        )
        shared_fields = _shared_parity_fields(fields)
        provenance_status = _require_string(
            train_provenance.get("status", ""),
            context=f"variant_authorities.{label}.train_provenance.status",
        )
        same_within_variant = (
            provenance_status == "present"
            and bool(train_provenance.get("train_manifest_present", False))
            and bool(train_provenance.get("checkpoint_provenance_present", False))
            and bool(train_provenance.get("parity_ok", False))
        )
        gate_eval_manifest_hash_matches_expected = shared_fields[
            "gate_eval_manifest_hash"
        ] == str(expected_gate_manifest["gate_eval_manifest_hash"])
        if reference_fields is None:
            reference_fields = shared_fields
        cross_variant_match = shared_fields == reference_fields
        variant_ok = (
            same_within_variant
            and cross_variant_match
            and gate_eval_manifest_hash_matches_expected
        )
        all_ok = all_ok and variant_ok
        rows[label] = {
            "variant": authority.get("variant"),
            "shared_fields": shared_fields,
            "consumer_mode": fields.get("consumer_mode"),
            "provenance_status": provenance_status,
            "same_within_variant": same_within_variant,
            "cross_variant_match": cross_variant_match,
            "gate_eval_manifest_hash_matches_expected": gate_eval_manifest_hash_matches_expected,
            "source_refs": {
                key: value
                for key, value in cast(
                    Mapping[str, str], authority.get("source_refs", {})
                ).items()
                if key in {"train_manifest", "checkpoint_provenance", "summary"}
            },
        }
    reference_payload = reference_fields or {}
    return {
        "status": "PASS" if all_ok else "FAIL",
        "reference_shared_fields": reference_payload,
        "current_rollout_manifest": dict(manifest_summary),
        "expected_gate_manifest": expected_gate_manifest,
        "variant_rows": rows,
        "consumer_modes": {
            label: rows[label]["consumer_mode"] for label in NON_STOCK_VARIANTS
        },
        "consumer_mode_note": "consumer_mode may differ between B and C by design; parity only freezes dataset/base/budget/eval-manifest bindings, while the train gate remains pinned to rollout_lite_v2.",
    }


def _gate_row(
    *,
    gate: str,
    name: str,
    status: str,
    point_estimate: float | None,
    ci95: Mapping[str, object],
    unit: str,
    decision: str,
    next_action: str,
    detail: str,
) -> dict[str, object]:
    return {
        "gate": gate,
        "name": name,
        "status": status,
        "point_estimate": point_estimate,
        "ci95": dict(ci95),
        "unit": unit,
        "decision": decision,
        "next_action": next_action,
        "detail": detail,
    }


def _build_go_no_go_report(
    *,
    paired_summary: Mapping[str, object],
    scope_summary: Mapping[str, object],
    dataset_binding: Mapping[str, object],
    parity_summary: Mapping[str, object],
) -> dict[str, object]:
    manifest = _require_mapping(
        paired_summary.get("manifest", {}), context="paired_summary.manifest"
    )
    manifest_name = _require_string(
        manifest.get("manifest_name", ""),
        context="paired_summary.manifest.manifest_name",
    )
    pairwise = _require_mapping(
        paired_summary.get("pairwise_deltas", {}),
        context="paired_summary.pairwise_deltas",
    )
    g3_pair = _require_mapping(
        pairwise.get("B_minus_A", {}), context="pairwise.B_minus_A"
    )
    g4_pair = _require_mapping(
        pairwise.get("C_minus_B", {}), context="pairwise.C_minus_B"
    )
    g5_pair = _require_mapping(
        pairwise.get("C_minus_A", {}), context="pairwise.C_minus_A"
    )

    g3_point = _optional_float(g3_pair.get("point_estimate"))
    g4_point = _optional_float(g4_pair.get("point_estimate"))
    g5_point = _optional_float(g5_pair.get("point_estimate"))
    g4_ci95 = _require_mapping(
        g4_pair.get("ci95", {}), context="pairwise.C_minus_B.ci95"
    )
    g4_lower = _optional_float(g4_ci95.get("lower"))

    g0_status = str(scope_summary.get("status", "FAIL"))
    g1_status = str(dataset_binding.get("status", "FAIL"))
    g2_status = str(parity_summary.get("status", "FAIL"))
    g3_status = "PASS" if g3_point is not None and g3_point >= -10.0 else "FAIL"
    if g4_point is None:
        raise ValueError("G4 point estimate is required")
    if g4_point >= 5.0 and g4_lower is not None and g4_lower > 0.0:
        g4_status = "PASS"
    elif g4_point > 0.0 and (g4_lower is None or g4_lower <= 0.0):
        g4_status = "HOLD"
    else:
        g4_status = "FAIL"
    if g5_point is None:
        raise ValueError("G5 point estimate is required")
    g5_status = "PASS" if g5_point > -5.0 else "FAIL"

    state_side_status = (
        "STATE_SIDE_ENTERED_CONDITIONALLY"
        if g4_status == "PASS" and g5_status == "PASS"
        else "STATE_SIDE_NOT_ENTERED"
    )
    state_side_next_action = (
        "G4/G5 已同时通过。Task 13 才允许运行 D，且 D 必须复用同 dataset/base/budget/eval authority。"
        if state_side_status == "STATE_SIDE_ENTERED_CONDITIONALLY"
        else "不要运行 D。保留 A/B/C "
        + f"{manifest_name} authority，并把 state side 明确保持为 STATE_SIDE_NOT_ENTERED。"
    )

    gates = [
        _gate_row(
            gate="G0",
            name="scope_fidelity",
            status=g0_status,
            point_estimate=1.0 if g0_status == "PASS" else 0.0,
            ci95=_ci95(lower=None, upper=None, unit="boolean"),
            unit="boolean",
            decision=(
                "scope_fidelity_confirmed"
                if g0_status == "PASS"
                else "scope_violation_detected"
            ),
            next_action=(
                "继续消费当前 authority bundle。"
                if g0_status == "PASS"
                else "先清理 submodules/openpi 触碰或越界主题，再重新生成 gate。"
            ),
            detail=(
                f"touches_submodules_openpi={scope_summary['touches_submodules_openpi']} forbidden_keyword_hit_count={scope_summary['forbidden_keyword_hit_count']}"
            ),
        ),
        _gate_row(
            gate="G1",
            name="dataset_fingerprint_and_universe_stability",
            status=g1_status,
            point_estimate=1.0 if g1_status == "PASS" else 0.0,
            ci95=_ci95(lower=None, upper=None, unit="boolean"),
            unit="boolean",
            decision=(
                "dataset_binding_confirmed"
                if g1_status == "PASS"
                else "dataset_binding_failed"
            ),
            next_action=(
                "继续沿用当前 relabeled 8D dataset authority。"
                if g1_status == "PASS"
                else "重新检查 dataset_fingerprint.json 与 episode_universe_hash.txt 的 shape/universe drift。"
            ),
            detail=(
                f"fingerprint={dataset_binding['dataset_fingerprint_sha256']} universe={dataset_binding['episode_universe_hash']}"
            ),
        ),
        _gate_row(
            gate="G2",
            name="train_provenance_parity",
            status=g2_status,
            point_estimate=1.0 if g2_status == "PASS" else 0.0,
            ci95=_ci95(lower=None, upper=None, unit="boolean"),
            unit="boolean",
            decision=("parity_confirmed" if g2_status == "PASS" else "parity_failed"),
            next_action=(
                "继续使用当前 B/C provenance 进入 paired gate。"
                if g2_status == "PASS"
                else "先修复 B/C 的 dataset/base/budget/eval-manifest parity，再重新生成 gate。"
            ),
            detail=json.dumps(
                parity_summary.get("reference_shared_fields", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
        ),
        _gate_row(
            gate="G3",
            name="control_not_catastrophic_vs_stock",
            status=g3_status,
            point_estimate=g3_point,
            ci95=_require_mapping(
                g3_pair.get("ci95", {}), context="pairwise.B_minus_A.ci95"
            ),
            unit="pp",
            decision=(
                "control_not_catastrophic"
                if g3_status == "PASS"
                else "debug_control_before_state_side"
            ),
            next_action=(
                "B 相对 A 未出现超过 10pp 的灾难性退化。"
                if g3_status == "PASS"
                else "先 debug B 的 control path，不要进入 state side。"
            ),
            detail=f"B-A={g3_point}pp, threshold=-10pp",
        ),
        _gate_row(
            gate="G4",
            name="recap_gain_vs_control",
            status=g4_status,
            point_estimate=g4_point,
            ci95=g4_ci95,
            unit="pp",
            decision=(
                "recap_gain_gate_passed"
                if g4_status == "PASS"
                else (
                    "PENDING_STRONG_CONFIRMATION"
                    if g4_status == "HOLD"
                    else "STATE_SIDE_NOT_ENTERED"
                )
            ),
            next_action=(
                "G4 已通过，若 G5 也通过则允许进入 Task 13。"
                if g4_status == "PASS"
                else (
                    "点估计为正但 CI 穿过 0，只允许等待 strong authority 做一次确认。"
                    if g4_status == "HOLD"
                    else "C 相对 B 未达到进入 D 的增益门槛，不要运行 D。"
                )
            ),
            detail=f"C-B={g4_point}pp, ci95.lower={g4_lower}pp, pass_threshold=+5pp",
        ),
        _gate_row(
            gate="G5",
            name="recap_viability_vs_stock",
            status=g5_status,
            point_estimate=g5_point,
            ci95=_require_mapping(
                g5_pair.get("ci95", {}), context="pairwise.C_minus_A.ci95"
            ),
            unit="pp",
            decision=(
                "stock_viability_preserved"
                if g5_status == "PASS"
                else "STATE_SIDE_NOT_ENTERED"
            ),
            next_action=(
                "C 对 A 的退化没有超过 5pp。"
                if g5_status == "PASS"
                else "C 相对 A 退化超过 5pp，不要运行 D。"
            ),
            detail=f"C-A={g5_point}pp, pass_threshold>-5pp",
        ),
        _gate_row(
            gate="G6",
            name="state_gain_vs_recap_only",
            status="NOT_APPLICABLE",
            point_estimate=None,
            ci95=_ci95(lower=None, upper=None, unit="pp"),
            unit="pp",
            decision=state_side_status,
            next_action=state_side_next_action,
            detail=f"当前阶段只汇总 A/B/C {manifest_name} authority，不运行 D。",
        ),
        _gate_row(
            gate="G7",
            name="v3_entry_gate",
            status="NOT_APPLICABLE",
            point_estimate=None,
            ci95=_ci95(lower=None, upper=None, unit="pp"),
            unit="pp",
            decision=state_side_status,
            next_action=state_side_next_action,
            detail=f"当前阶段没有 D authority（上游 authority={manifest_name}），因此不计算 v3 entry gate。",
        ),
    ]
    return {
        "schema_version": GO_NO_GO_REPORT_SCHEMA_VERSION,
        "eval_authority": EXPECTED_EVAL_AUTHORITY,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "manifest": paired_summary.get("manifest"),
        "paired_rollout_summary": {
            "path": str(ARTIFACT_TOPIC_DIR / PAIRED_SUMMARY_NAME),
            "pairwise_deltas": paired_summary.get("pairwise_deltas"),
        },
        "scope_audit_manifest": {
            "path": str(ARTIFACT_TOPIC_DIR / SCOPE_AUDIT_MANIFEST_NAME),
            "status": scope_summary.get("status"),
        },
        "state_side_status": state_side_status,
        "eligible_for_state_side": state_side_status
        == "STATE_SIDE_ENTERED_CONDITIONALLY",
        "next_action": state_side_next_action,
        "gates": gates,
    }


def _bundle_per_episode_rows(
    variant_authorities: Mapping[str, Mapping[str, object]], *, label: str
) -> Sequence[object]:
    authority = _require_mapping(
        variant_authorities.get(label, {}), context=f"{label}.authority"
    )
    bundle = _require_mapping(authority.get("bundle", {}), context=f"{label}.bundle")
    return _require_sequence(
        bundle.get("per_episode_rollouts", []),
        context=f"{label}.bundle.per_episode_rollouts",
    )


def build_libero_abc_gate_artifacts_v2(
    *,
    output_root: str | Path = ARTIFACT_TOPIC_DIR,
    authority_root: str | Path = ARTIFACT_TOPIC_DIR,
    eval_manifest_id: str = "rollout_lite_v2_d8322aa9062d",
    git_diff_paths: Sequence[str] | None = None,
    relevant_scope_docs: Sequence[str | Path] | None = None,
) -> dict[str, object]:
    output_root_path = Path(output_root).resolve()
    authority_root_path = Path(authority_root).resolve()
    relevant_docs = [
        Path(path).resolve()
        for path in (relevant_scope_docs or DEFAULT_RELEVANT_SCOPE_DOCS)
    ]
    git_paths = (
        list(git_diff_paths)
        if git_diff_paths is not None
        else _run_git_diff_name_only(repo_root=REPO_ROOT)
    )
    keyword_hits = _scan_forbidden_scope_keywords(
        doc_paths=relevant_docs,
        keywords=FORBIDDEN_SCOPE_KEYWORDS,
    )
    scope_summary = _build_scope_audit_summary(
        git_diff_paths=git_paths,
        relevant_docs=relevant_docs,
        keyword_hits=keyword_hits,
    )
    scope_manifest_text = _scope_audit_manifest_text(scope_summary)

    variant_authorities = {
        label: _load_variant_authority(
            output_root=authority_root_path,
            label=label,
            variant=variant,
            eval_manifest_id=eval_manifest_id,
        )
        for label, variant in VARIANT_ORDER
    }
    manifest_summary = _build_manifest_summary(
        _require_mapping(
            variant_authorities["A"].get("eval_manifest", {}), context="A.eval_manifest"
        )
    )

    dataset_fingerprint = _read_json(DATASET_FINGERPRINT_PATH)
    episode_universe_hash = _require_string(
        EPISODE_UNIVERSE_HASH_PATH.read_text(encoding="utf-8").strip(),
        context="episode_universe_hash.txt",
    )
    dataset_binding = _build_dataset_binding_summary(
        dataset_fingerprint=dataset_fingerprint,
        episode_universe_hash=episode_universe_hash,
        variant_authorities=variant_authorities,
    )
    parity_summary = _build_parity_summary(
        variant_authorities=variant_authorities,
        manifest_summary=manifest_summary,
    )

    pairwise_deltas = {
        "B_minus_A": _build_pairwise_delta(
            lhs_label="B",
            lhs_variant=VARIANT_TO_ARTIFACT_NAME["B"],
            lhs_rows=_bundle_per_episode_rows(variant_authorities, label="B"),
            rhs_label="A",
            rhs_variant=VARIANT_TO_ARTIFACT_NAME["A"],
            rhs_rows=_bundle_per_episode_rows(variant_authorities, label="A"),
            eval_manifest_id=eval_manifest_id,
        ),
        "C_minus_B": _build_pairwise_delta(
            lhs_label="C",
            lhs_variant=VARIANT_TO_ARTIFACT_NAME["C"],
            lhs_rows=_bundle_per_episode_rows(variant_authorities, label="C"),
            rhs_label="B",
            rhs_variant=VARIANT_TO_ARTIFACT_NAME["B"],
            rhs_rows=_bundle_per_episode_rows(variant_authorities, label="B"),
            eval_manifest_id=eval_manifest_id,
        ),
        "C_minus_A": _build_pairwise_delta(
            lhs_label="C",
            lhs_variant=VARIANT_TO_ARTIFACT_NAME["C"],
            lhs_rows=_bundle_per_episode_rows(variant_authorities, label="C"),
            rhs_label="A",
            rhs_variant=VARIANT_TO_ARTIFACT_NAME["A"],
            rhs_rows=_bundle_per_episode_rows(variant_authorities, label="A"),
            eval_manifest_id=eval_manifest_id,
        ),
    }

    paired_summary = {
        "schema_version": PAIRED_SUMMARY_SCHEMA_VERSION,
        "eval_authority": EXPECTED_EVAL_AUTHORITY,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "manifest": manifest_summary,
        "variants": {
            label: {
                "variant": authority["variant"],
                "authority_dir": authority["authority_dir"],
                "rollout_summary": authority["rollout_summary"],
                "bootstrap_ci": authority["bootstrap_ci"],
                "scope_audit": authority["scope_audit"],
                "train_provenance": authority["train_provenance"],
                "paired_delta_vs_stock": authority["paired_delta_vs_stock"],
                "source_refs": authority["source_refs"],
            }
            for label, authority in variant_authorities.items()
        },
        "pairwise_deltas": pairwise_deltas,
        "gate_inputs": {
            "scope_fidelity": scope_summary,
            "dataset_binding": dataset_binding,
            "train_provenance_parity": parity_summary,
        },
    }
    go_no_go_report = _build_go_no_go_report(
        paired_summary=paired_summary,
        scope_summary=scope_summary,
        dataset_binding=dataset_binding,
        parity_summary=parity_summary,
    )
    return {
        "paired_summary": paired_summary,
        "go_no_go_report": go_no_go_report,
        "scope_audit_manifest_text": scope_manifest_text,
    }


def materialize_libero_abc_gate_artifacts_v2(
    *,
    output_root: str | Path = ARTIFACT_TOPIC_DIR,
    authority_root: str | Path = ARTIFACT_TOPIC_DIR,
    eval_manifest_id: str = "rollout_lite_v2_d8322aa9062d",
    git_diff_paths: Sequence[str] | None = None,
    relevant_scope_docs: Sequence[str | Path] | None = None,
) -> dict[str, Path]:
    output_root_path = Path(output_root).resolve()
    payload = build_libero_abc_gate_artifacts_v2(
        output_root=output_root_path,
        authority_root=authority_root,
        eval_manifest_id=eval_manifest_id,
        git_diff_paths=git_diff_paths,
        relevant_scope_docs=relevant_scope_docs,
    )
    paired_summary_path = output_root_path / PAIRED_SUMMARY_NAME
    go_no_go_report_path = output_root_path / GO_NO_GO_REPORT_NAME
    scope_audit_manifest_path = output_root_path / SCOPE_AUDIT_MANIFEST_NAME
    _write_json(
        paired_summary_path,
        _require_mapping(
            payload.get("paired_summary", {}), context="payload.paired_summary"
        ),
    )
    _write_json(
        go_no_go_report_path,
        _require_mapping(
            payload.get("go_no_go_report", {}), context="payload.go_no_go_report"
        ),
    )
    _write_text(
        scope_audit_manifest_path,
        _require_string(
            payload.get("scope_audit_manifest_text", ""),
            context="payload.scope_audit_manifest_text",
        ),
    )
    return {
        "paired_summary": paired_summary_path,
        "go_no_go_report": go_no_go_report_path,
        "scope_audit_manifest": scope_audit_manifest_path,
    }


__all__ = [
    "ARTIFACT_TOPIC_DIR",
    "DEFAULT_RELEVANT_SCOPE_DOCS",
    "FORBIDDEN_SCOPE_KEYWORDS",
    "GO_NO_GO_REPORT_NAME",
    "GO_NO_GO_REPORT_SCHEMA_VERSION",
    "PAIRED_SUMMARY_NAME",
    "PAIRED_SUMMARY_SCHEMA_VERSION",
    "SCOPE_AUDIT_MANIFEST_NAME",
    "build_libero_abc_gate_artifacts_v2",
    "materialize_libero_abc_gate_artifacts_v2",
]
