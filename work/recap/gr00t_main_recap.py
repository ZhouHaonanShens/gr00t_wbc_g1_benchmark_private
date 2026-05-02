from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from work.recap import text_indicator


G2_MAIN_V2_PREFLIGHT_SCHEMA_VERSION = "gr00t_g2_main_v2_preflight_v1"
G2_MAIN_V2_METHOD_CONTRACT = "binary_text_indicator_strict_full_v1"
DEFAULT_G2_MAIN_V2_RUN_ID = "stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ"
DEFAULT_G2_MAIN_V2_DATASET_REL = Path("agent/artifacts/lerobot_datasets/recap_stage3_iter_002")
DEFAULT_G2_MAIN_V2_OUTPUT_ROOT_REL = (
    Path("agent/artifacts/gr00t_recap_live/single_gpu_v2_full_update")
    / DEFAULT_G2_MAIN_V2_RUN_ID
    / "gr00t"
)
DEFAULT_G2_MAIN_V2_RUNTIME_ROOT_REL = (
    Path("agent/runtime_logs") / DEFAULT_G2_MAIN_V2_RUN_ID / "gr00t"
)
DEFAULT_G2_MAIN_V2_CRITIC_DIR_REL = Path("agent/artifacts/critics/task7_real_critic_v2")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _read_json_if_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): value for key, value in payload.items()}


def _path_same(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return os.path.abspath(str(left)) == os.path.abspath(str(right))


def _list_value(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    return []


def assess_critic_vlm_for_g2_main_v2(
    *,
    critic_dir: Path,
    target_dataset_path: Path,
) -> dict[str, Any]:
    config = _read_json_if_mapping(critic_dir / "config.json")
    provenance = _read_json_if_mapping(critic_dir / "provenance.json")
    metrics = _read_json_if_mapping(critic_dir / "metrics.json")
    bin_centers_payload = _read_json_if_mapping(critic_dir / "bin_centers.json")
    bin_centers = _list_value(config.get("bin_centers")) or _list_value(
        bin_centers_payload.get("bin_centers")
    )
    train_manifest_summary = provenance.get("train_manifest_summary")
    if not isinstance(train_manifest_summary, Mapping):
        train_manifest_summary = {}
    input_mode = train_manifest_summary.get("input_mode")
    if not isinstance(input_mode, Mapping):
        input_mode = {}

    reasons: list[str] = []
    if not critic_dir.is_dir():
        reasons.append("CRITIC_DIR_MISSING")
    if str(config.get("artifact_version", "")) != "multimodal_distributional_v1":
        reasons.append("NOT_MULTIMODAL_DISTRIBUTIONAL_V1")
    if len(bin_centers) != 201:
        reasons.append("BIN_COUNT_NOT_201")
    if str(config.get("value_scale", "")) not in {
        "task_normalized_return",
        "normalized_return",
    }:
        reasons.append("VALUE_SCALE_NOT_TASK_NORMALIZED_RETURN")
    if str(config.get("prompt_text_mode", "")) == "constant_query_only":
        reasons.append("PROMPT_NOT_LANGUAGE_CONDITIONED")
    if input_mode.get("use_prompt") is False:
        reasons.append("TRAIN_MANIFEST_USE_PROMPT_FALSE")

    source_dataset_raw = str(provenance.get("dataset_path", "")).strip()
    dataset_match = False
    if source_dataset_raw:
        dataset_match = _path_same(Path(source_dataset_raw), target_dataset_path)
    if not dataset_match:
        reasons.append("CRITIC_DATASET_DOES_NOT_MATCH_G2_MAIN_V2_DATASET")

    usable = not reasons
    return {
        "schema_version": "gr00t_g2_main_v2_critic_vlm_assessment_v1",
        "critic_dir": str(critic_dir),
        "target_dataset_path": str(target_dataset_path),
        "usable_as_authoritative_critic": usable,
        "recommended_role": "authoritative_critic" if usable else "candidate_or_proxy_only",
        "blocking_reasons": reasons,
        "artifact_version": config.get("artifact_version"),
        "backend": config.get("smoke_backend") or config.get("backend_name"),
        "base_model": config.get("base_model"),
        "value_scale": config.get("value_scale"),
        "bin_count": len(bin_centers),
        "prompt_text_mode": config.get("prompt_text_mode"),
        "train_manifest_use_prompt": input_mode.get("use_prompt"),
        "provenance_dataset_path": source_dataset_raw,
        "metrics_summary": {
            "best_val_loss": metrics.get("best_val_loss"),
            "formal_task_fit_done": metrics.get("formal_task_fit_done"),
            "train_sample_count": metrics.get("train_sample_count"),
            "val_sample_count": metrics.get("val_sample_count"),
            "upgrade_pending": metrics.get("upgrade_pending"),
        },
    }


def _coerce_scalar(raw: Any) -> Any:
    if isinstance(raw, (list, tuple)) and raw:
        return _coerce_scalar(raw[0])
    item = getattr(raw, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return raw


def _indicator_mode_from_row_value(raw: Any) -> str:
    return text_indicator.indicator_mode_from_indicator_value(
        _coerce_scalar(raw),
        field_name="recap_m2.indicator_I",
    )


def build_text_indicator_dataset_audit(
    *,
    dataset_path: Path,
    indicator_dropout_p: float,
    dropout_seed: int,
    prompt_raw_column: str = "recap_m2.prompt_raw",
    max_rows: int | None = None,
) -> dict[str, Any]:
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        return {
            "schema_version": "gr00t_g2_main_v2_text_indicator_dataset_audit_v1",
            "status": "not_run",
            "reason": f"pandas_unavailable: {exc}",
            "dataset_path": str(dataset_path),
        }

    parquet_files = sorted(dataset_path.glob("data/**/*.parquet"))
    if not parquet_files:
        return {
            "schema_version": "gr00t_g2_main_v2_text_indicator_dataset_audit_v1",
            "status": "blocked",
            "reason": "no_parquet_files",
            "dataset_path": str(dataset_path),
        }

    raw_counts: Counter[str] = Counter()
    effective_counts: Counter[str] = Counter()
    prompt_source_counts: Counter[str] = Counter()
    rows_seen = 0
    for parquet_file in parquet_files:
        columns = ["recap_m2.indicator_I"]
        try:
            probe_df = pd.read_parquet(parquet_file, columns=[prompt_raw_column])
            prompt_column_available = prompt_raw_column in probe_df.columns
        except Exception:
            prompt_column_available = False
        if prompt_column_available:
            columns.append(prompt_raw_column)
        df = pd.read_parquet(parquet_file, columns=columns)
        for local_index, row in df.iterrows():
            if max_rows is not None and rows_seen >= int(max_rows):
                break
            raw_mode = _indicator_mode_from_row_value(row["recap_m2.indicator_I"])
            raw_counts[raw_mode] += 1
            sample_key = f"{parquet_file.as_posix()}|row={local_index}|raw={raw_mode}"
            effective = text_indicator.apply_indicator_dropout(
                raw_mode,
                dropout_p=indicator_dropout_p,
                seed=int(dropout_seed),
                sample_key=sample_key,
            )
            effective_counts[effective] += 1
            prompt_source_counts[
                "prompt_raw_column" if prompt_column_available else "step_text_fallback"
            ] += 1
            rows_seen += 1
        if max_rows is not None and rows_seen >= int(max_rows):
            break

    return {
        "schema_version": "gr00t_g2_main_v2_text_indicator_dataset_audit_v1",
        "status": "ok",
        "dataset_path": str(dataset_path),
        "parquet_file_count": len(parquet_files),
        "rows_seen": rows_seen,
        "indicator_dropout_p": float(indicator_dropout_p),
        "dropout_seed": int(dropout_seed),
        "raw_indicator_counts": dict(raw_counts),
        "effective_indicator_counts": dict(effective_counts),
        "prompt_source_counts": dict(prompt_source_counts),
        "three_state_training_surface_present": all(
            effective_counts.get(mode, 0) > 0
            for mode in (
                text_indicator.TEXT_INDICATOR_POSITIVE,
                text_indicator.TEXT_INDICATOR_NEGATIVE,
                text_indicator.TEXT_INDICATOR_OMIT,
            )
        ),
    }


def build_g2_main_v2_preflight(
    *,
    dataset_path: Path,
    output_dir: Path,
    runtime_log_dir: Path,
    critic_dir: Path,
    indicator_dropout_p: float,
    seed: int,
    train_scope: str,
) -> dict[str, Any]:
    critic_assessment = assess_critic_vlm_for_g2_main_v2(
        critic_dir=critic_dir,
        target_dataset_path=dataset_path,
    )
    dataset_audit = build_text_indicator_dataset_audit(
        dataset_path=dataset_path,
        indicator_dropout_p=float(indicator_dropout_p),
        dropout_seed=int(seed),
        max_rows=None,
    )
    return {
        "schema_version": G2_MAIN_V2_PREFLIGHT_SCHEMA_VERSION,
        "method_contract": G2_MAIN_V2_METHOD_CONTRACT,
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_path.is_dir(),
        "output_dir": str(output_dir),
        "runtime_log_dir": str(runtime_log_dir),
        "conditioning_route": "text_indicator_v1",
        "runtime_indicator_mode": "positive",
        "indicator_dropout_p": float(indicator_dropout_p),
        "train_scope": str(train_scope),
        "checkpoint_2200_role": "g2_lite_diagnostic_only_not_g3_g4_authority",
        "value_advantage_source": {
            "current_training_indicator_source": "recap_m2.indicator_I",
            "current_training_epsilon_source": "recap_m2.epsilon_l",
            "current_training_role": "dataset_m2_labels_proxy",
            "critic_vlm_authority": bool(
                critic_assessment["usable_as_authoritative_critic"]
            ),
            "critic_vlm_recommended_role": critic_assessment["recommended_role"],
        },
        "critic_vlm_assessment": critic_assessment,
        "text_indicator_dataset_audit": dataset_audit,
        "launch_gate": {
            "g2_main_v2_launch_allowed": dataset_path.is_dir()
            and dataset_audit.get("status") in {"ok", "not_run"},
            "requires_gpu_smoke_before_full": True,
            "requires_tmux_for_full": True,
        },
    }
