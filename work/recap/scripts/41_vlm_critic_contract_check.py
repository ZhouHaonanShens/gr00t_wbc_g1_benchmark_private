#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, cast


_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


from work.recap.advantage import (
    GENERIC_DIAGNOSTIC_COMPATIBILITY_FIELDS,
    MAINLINE_TASK_TEXT_FIELD,
    VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE,
    VLM_CRITIC_EVAL_SMOKE_DIAGNOSTIC_ROUTE,
    build_diagnostic_surface_metadata,
    diagnostic_surface_violations,
)


JsonObject = dict[str, object]


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_JSON_REL = "agent/artifacts/vlm_critic_manifests/critic_contract_v1.json"
DEFAULT_PLAN_PATH_REL = ".sisyphus/plans/g1_vlm_critic_bootstrap_mainline.md"
DEFAULT_CONTRACT_DOC_REL = "agent/exchange/vlm_critic_contract_v1.md"
DEFAULT_MAX_SAMPLES = 64


SPEC_PASS_SENTINEL = "VLM_CRITIC_CONTRACT_OK"
SPEC_FAIL_SENTINEL = "VLM_CRITIC_CONTRACT_SPEC_FAIL"
NEGATIVE_REJECTED_SENTINEL = "VLM_CRITIC_CONTRACT_NEGATIVE_REJECTED"
NEGATIVE_UNEXPECTED_PASS_SENTINEL = "VLM_CRITIC_CONTRACT_NEGATIVE_UNEXPECTED_PASS"
VERIFY_PLAN_PASS_SENTINEL = "VLM_CRITIC_CONTRACT_VERIFY_PLAN_PASS"
VERIFY_PLAN_FAIL_SENTINEL = "VLM_CRITIC_CONTRACT_VERIFY_PLAN_FAIL"
DATASET_PASS_SENTINEL = "VLM_CRITIC_DATASET_CONTRACT_OK"
DATASET_FAIL_SENTINEL = "VLM_CRITIC_DATASET_CONTRACT_FAIL"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(repo_root: Path, raw_path: str | None, *, default_rel: str) -> Path:
    value = str(raw_path or default_rel)
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        _ = f.write("\n")
    _ = tmp_path.replace(path)


def _read_json(path: Path) -> JsonObject:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return cast(JsonObject, obj)


def _read_jsonl(path: Path) -> list[JsonObject]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")
    out: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object in {path} line {line_no}, got {type(obj).__name__}"
                )
            out.append(cast(JsonObject, obj))
    return out


def _as_int(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Expected int-like value ({context}), got bool")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"Expected integer-valued float ({context}), got {value!r}"
            )
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"Expected int-like string ({context}), got {value!r}"
            ) from exc
    raise ValueError(f"Expected int-like value ({context}), got {type(value).__name__}")


def _as_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected float-like value ({context}), got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Expected float-like string ({context}), got {value!r}"
            ) from exc
    raise ValueError(
        f"Expected float-like value ({context}), got {type(value).__name__}"
    )


def _as_str(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string ({context}), got {value!r}")
    return str(value)


def _json_list_of_dicts(value: object, *, context: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise ValueError(f"Expected list ({context}), got {type(value).__name__}")
    out: list[JsonObject] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected JSON object in {context}[{idx}], got {type(item).__name__}"
            )
        out.append(cast(JsonObject, item))
    return out


def _base_contract(repo_root: Path) -> JsonObject:
    return {
        "schema_version": "vlm_critic_contract_v1",
        "authority_doc": DEFAULT_CONTRACT_DOC_REL,
        "checker_entrypoint": "work/recap/scripts/41_vlm_critic_contract_check.py",
        "critic_type": "multimodal_distributional_v1",
        "training_source": "with_video_lerobot",
        "task_text_field": "prompt_raw",
        "allow_future_frames": False,
        "labeler_backend": "value_source_critic_versioned",
        "formal_gate_dataset_scope": "isaac_only",
        "public_warmstart_scope": "initialization_only",
        "working_base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "upgrade_pending": "temporal_critic_review",
        "diagnostic_surfaces": {
            "critic_eval_smoke": {
                **build_diagnostic_surface_metadata(
                    surface_route=VLM_CRITIC_EVAL_SMOKE_DIAGNOSTIC_ROUTE,
                    authority_scope=VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE,
                    compatibility_fields=GENERIC_DIAGNOSTIC_COMPATIBILITY_FIELDS,
                    surface_kind="vlm_critic_eval_smoke_summary",
                ),
                "task_text_field": MAINLINE_TASK_TEXT_FIELD,
            }
        },
        "labeler_boundary": {
            "labeler_role": "consume_versioned_critic_value_source_only",
            "serving_api": "versioned_multimodal_critic_bundle",
            "must_not_require": [
                "future_frames",
                "prompt_conditioned",
                "public_formal_gate_mix",
            ],
            "legacy_state_only_backend_status": "not_authoritative_for_task1",
        },
        "plan_anchor": {
            "task_id": "task_1_vlm_critic_contract",
            "plan_path_default": DEFAULT_PLAN_PATH_REL,
        },
        "sentinels": {
            "spec_pass": SPEC_PASS_SENTINEL,
            "spec_fail": SPEC_FAIL_SENTINEL,
            "negative_rejected": NEGATIVE_REJECTED_SENTINEL,
            "negative_unexpected_pass": NEGATIVE_UNEXPECTED_PASS_SENTINEL,
            "verify_plan_pass": VERIFY_PLAN_PASS_SENTINEL,
            "verify_plan_fail": VERIFY_PLAN_FAIL_SENTINEL,
            "dataset_pass": DATASET_PASS_SENTINEL,
            "dataset_fail": DATASET_FAIL_SENTINEL,
        },
        "repo_root": str(repo_root),
    }


def _ns_str(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name, "")
    return value if isinstance(value, str) else str(value or "")


def _ns_flag(args: argparse.Namespace, name: str) -> bool:
    return bool(getattr(args, name, False))


def _build_candidate_contract(repo_root: Path, args: argparse.Namespace) -> JsonObject:
    contract = _base_contract(repo_root)
    expect_task_text = _ns_str(args, "expect_task_text").strip()
    if expect_task_text:
        contract["task_text_field"] = expect_task_text
    if _ns_flag(args, "allow_future_frames"):
        contract["allow_future_frames"] = True
    expect_labeler_backend = _ns_str(args, "expect_labeler_backend").strip()
    if expect_labeler_backend:
        contract["labeler_backend"] = expect_labeler_backend
    expect_training_source = _ns_str(args, "expect_training_source").strip()
    if expect_training_source:
        contract["training_source"] = expect_training_source
    expect_formal_gate_dataset_scope = _ns_str(
        args, "expect_formal_gate_dataset_scope"
    ).strip()
    if expect_formal_gate_dataset_scope:
        contract["formal_gate_dataset_scope"] = expect_formal_gate_dataset_scope
    expect_public_warmstart_scope = _ns_str(
        args, "expect_public_warmstart_scope"
    ).strip()
    if expect_public_warmstart_scope:
        contract["public_warmstart_scope"] = expect_public_warmstart_scope
    expect_working_base_model = _ns_str(args, "expect_working_base_model").strip()
    if expect_working_base_model:
        contract["working_base_model"] = expect_working_base_model
    expect_upgrade_pending = _ns_str(args, "expect_upgrade_pending").strip()
    if expect_upgrade_pending:
        contract["upgrade_pending"] = expect_upgrade_pending
    return contract


def _validate_contract(contract: JsonObject) -> list[str]:
    violations: list[str] = []

    expected_pairs = {
        "schema_version": "vlm_critic_contract_v1",
        "critic_type": "multimodal_distributional_v1",
        "training_source": "with_video_lerobot",
        "task_text_field": "prompt_raw",
        "labeler_backend": "value_source_critic_versioned",
        "formal_gate_dataset_scope": "isaac_only",
        "public_warmstart_scope": "initialization_only",
        "working_base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "upgrade_pending": "temporal_critic_review",
    }
    for key, expected in expected_pairs.items():
        actual = contract.get(key)
        if actual != expected:
            violations.append(f"{key} must be {expected!r}, got {actual!r}")

    allow_future_frames = contract.get("allow_future_frames")
    if allow_future_frames is not False:
        violations.append(
            f"allow_future_frames must be False, got {allow_future_frames!r}"
        )

    task_text_field = contract.get("task_text_field")
    if task_text_field != "prompt_raw":
        violations.append(
            f"task_text_field must stay on 'prompt_raw' for Task 1, got {task_text_field!r}"
        )

    if contract.get("task_text_field") == "prompt_conditioned":
        violations.append(
            "prompt_conditioned is forbidden as the Task 1 task_text_field authority"
        )

    if contract.get("formal_gate_dataset_scope") != "isaac_only":
        violations.append("formal gate dataset scope must remain isaac_only")

    if contract.get("public_warmstart_scope") != "initialization_only":
        violations.append("public warmstart scope must remain initialization_only")

    labeler_boundary_raw = contract.get("labeler_boundary")
    if not isinstance(labeler_boundary_raw, dict):
        violations.append("labeler_boundary must be a JSON object")
    else:
        labeler_boundary = cast(Mapping[object, object], labeler_boundary_raw)
        must_not_require = labeler_boundary.get("must_not_require")
        if not isinstance(must_not_require, list):
            violations.append("labeler_boundary.must_not_require must be a list")
        else:
            for forbidden in (
                "future_frames",
                "prompt_conditioned",
                "public_formal_gate_mix",
            ):
                if forbidden not in must_not_require:
                    violations.append(
                        f"labeler_boundary.must_not_require must include {forbidden!r}"
                    )

    diagnostic_surfaces_raw = contract.get("diagnostic_surfaces")
    if not isinstance(diagnostic_surfaces_raw, dict):
        violations.append("diagnostic_surfaces must be a JSON object")
    else:
        critic_eval_smoke = diagnostic_surfaces_raw.get("critic_eval_smoke")
        if not isinstance(critic_eval_smoke, Mapping):
            violations.append("diagnostic_surfaces.critic_eval_smoke must be an object")
        else:
            violations.extend(
                f"diagnostic_surfaces.critic_eval_smoke: {item}"
                for item in diagnostic_surface_violations(
                    critic_eval_smoke,
                    expected_route=VLM_CRITIC_EVAL_SMOKE_DIAGNOSTIC_ROUTE,
                    expected_authority_scope=VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE,
                    required_compatibility_fields=GENERIC_DIAGNOSTIC_COMPATIBILITY_FIELDS,
                )
            )
            if critic_eval_smoke.get("task_text_field") != MAINLINE_TASK_TEXT_FIELD:
                violations.append(
                    "diagnostic_surfaces.critic_eval_smoke.task_text_field must stay on "
                    f"{MAINLINE_TASK_TEXT_FIELD!r}"
                )

    return violations


def validate_critic_smoke_summary_contract(
    summary: Mapping[str, object],
) -> list[str]:
    violations = diagnostic_surface_violations(
        summary,
        expected_route=VLM_CRITIC_EVAL_SMOKE_DIAGNOSTIC_ROUTE,
        expected_authority_scope=VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE,
        required_compatibility_fields=GENERIC_DIAGNOSTIC_COMPATIBILITY_FIELDS,
    )
    task_text_field = summary.get("task_text_field")
    if task_text_field not in (None, MAINLINE_TASK_TEXT_FIELD):
        violations.append(
            f"task_text_field must be absent or {MAINLINE_TASK_TEXT_FIELD!r}, got {task_text_field!r}"
        )
    for field_name in ("success_rate", "success_count", "episodes", "wrapper_status"):
        if field_name not in summary:
            violations.append(f"missing required generic field {field_name!r}")
    return violations


def _plan_checks() -> list[tuple[str, list[re.Pattern[str]]]]:
    raw_checks: list[tuple[str, list[str]]] = [
        ("critic_type", [r"multimodal_distributional_v1"]),
        ("training_source", [r"with_video_lerobot"]),
        ("task_text_field", [r"prompt_raw"]),
        (
            "future_frame_guardrail",
            [r"allow_future_frames\s*=\s*false", r"allow_future_frames=false"],
        ),
        ("labeler_backend", [r"value_source_critic_versioned"]),
        ("formal_gate_dataset_scope", [r"isaac_only"]),
        ("public_warmstart_scope", [r"initialization_only"]),
        ("working_base_model", [r"Qwen/Qwen3-VL-2B-Instruct"]),
        ("upgrade_pending", [r"temporal_critic_review"]),
    ]
    return [
        (name, [re.compile(pat, flags=re.IGNORECASE) for pat in patterns])
        for name, patterns in raw_checks
    ]


def _verify_plan(plan_path: Path) -> JsonObject:
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing plan file: {plan_path}")
    if not plan_path.is_file():
        raise ValueError(f"plan_path is not a file: {plan_path}")

    plan_text = plan_path.read_text(encoding="utf-8")
    line_count = plan_text.count("\n") + (1 if plan_text else 0)

    passed_checks: list[str] = []
    missing_checks: list[str] = []
    for name, patterns in _plan_checks():
        if any(pattern.search(plan_text) for pattern in patterns):
            passed_checks.append(name)
        else:
            missing_checks.append(name)

    return {
        "plan_path": str(plan_path),
        "line_count": int(line_count),
        "passed_checks": passed_checks,
        "missing_checks": missing_checks,
        "pass": not missing_checks,
    }


def _import_parquet_module() -> Any:
    try:
        return importlib.import_module("pyarrow.parquet")
    except Exception as exc:
        raise RuntimeError(f"dataset_contract_missing_pyarrow: {exc}") from exc


def _parquet_read_table(parquet_path: Path, *, columns: list[str] | None = None) -> Any:
    pq = _import_parquet_module()
    try:
        return pq.read_table(str(parquet_path), columns=columns)
    except Exception as exc:
        raise RuntimeError(
            f"dataset_contract_parquet_read_failed: {parquet_path}: {exc}"
        ) from exc


def _parquet_num_rows(parquet_path: Path) -> int:
    pq = _import_parquet_module()
    try:
        pf = pq.ParquetFile(str(parquet_path))
        meta = pf.metadata
    except Exception as exc:
        raise RuntimeError(
            f"dataset_contract_parquet_meta_failed: {parquet_path}: {exc}"
        ) from exc
    if meta is None:
        raise RuntimeError(f"dataset_contract_parquet_meta_missing: {parquet_path}")
    rows = int(meta.num_rows)
    if rows <= 0:
        raise RuntimeError(
            f"dataset_contract_parquet_rows_invalid: {parquet_path}: {rows}"
        )
    return rows


def _ffprobe_num_frames(video_path: Path) -> tuple[int | None, str | None]:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return None, "ffprobe_missing"
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return None, f"ffprobe_rc={proc.returncode} err={err[:240]}"
    try:
        obj = json.loads(proc.stdout)
    except Exception as exc:
        return None, f"ffprobe_json_error: {exc}"
    streams = obj.get("streams") if isinstance(obj, dict) else None
    if not isinstance(streams, list) or not streams:
        return None, "ffprobe_streams_missing"
    s0 = streams[0]
    if not isinstance(s0, dict):
        return None, "ffprobe_stream0_invalid"
    nb = s0.get("nb_frames")
    if isinstance(nb, int) and nb > 0:
        return int(nb), None
    if isinstance(nb, str):
        nb_s = nb.strip()
        if nb_s.isdigit():
            parsed = int(nb_s)
            if parsed > 0:
                return parsed, None
    return None, "ffprobe_nb_frames_missing"


def _opencv_num_frames(video_path: Path) -> tuple[int | None, str | None]:
    try:
        cv2 = importlib.import_module("cv2")
    except Exception as exc:
        return None, f"opencv_import_error: {exc}"
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, "opencv_cap_not_opened"
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except Exception as exc:
        return None, f"opencv_count_error: {exc}"
    if count <= 0:
        return None, f"opencv_count_invalid: {count}"
    return count, None


def _ffmpeg_frame_probe(video_path: Path, frame_index: int) -> tuple[bool, str | None]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False, "ffmpeg_missing"
    if frame_index < 0:
        return False, f"negative_frame_index: {frame_index}"
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{int(frame_index)})",
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = ((proc.stderr or b"") or (proc.stdout or b"")).decode(
            "utf-8", errors="replace"
        )
        return False, f"ffmpeg_rc={proc.returncode} err={err[:240].strip()}"
    if not proc.stdout:
        return False, "ffmpeg_empty_stdout"
    return True, None


def _opencv_frame_probe(video_path: Path, frame_index: int) -> tuple[bool, str | None]:
    try:
        cv2 = importlib.import_module("cv2")
    except Exception as exc:
        return False, f"opencv_import_error: {exc}"
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "opencv_cap_not_opened"
        _ = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        cap.release()
    except Exception as exc:
        return False, f"opencv_read_error: {exc}"
    if not ok or frame is None:
        return False, "opencv_frame_read_failed"
    return True, None


def _probe_frame_decode(video_path: Path, frame_index: int) -> tuple[str, int | None]:
    ok, err = _ffmpeg_frame_probe(video_path, frame_index)
    if ok:
        return "ffmpeg.frame_probe", None
    ok, cv_err = _opencv_frame_probe(video_path, frame_index)
    if ok:
        return "opencv.frame_probe", None
    detail = f"ffmpeg={err}; opencv={cv_err}"
    raise RuntimeError(
        f"video_decode_missing: cannot decode frame_index={frame_index} from {video_path}: {detail}"
    )


def _resolve_num_frames(video_path: Path) -> tuple[int | None, str | None]:
    ff_n, ff_err = _ffprobe_num_frames(video_path)
    if ff_n is not None:
        return ff_n, "ffprobe.nb_frames"
    cv_n, cv_err = _opencv_num_frames(video_path)
    if cv_n is not None:
        return cv_n, "opencv.frame_count"
    return None, f"ffprobe={ff_err}; opencv={cv_err}"


def _table_slice_pylist(table: Any, column: str, count: int) -> list[object]:
    try:
        sliced = table.slice(0, int(count))
        return list(cast(list[object], sliced.column(column).to_pylist()))
    except Exception as exc:
        raise RuntimeError(
            f"dataset_contract_column_read_failed: column={column!r}: {exc}"
        ) from exc


def _resolve_dataset_summary(
    repo_root: Path, dataset_path: Path, *, max_samples: int
) -> JsonObject:
    if max_samples <= 0:
        raise ValueError(f"max_samples must be > 0, got {max_samples}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise ValueError(f"dataset_path is not a directory: {dataset_path}")

    meta_dir = dataset_path / "meta"
    info = _read_json(meta_dir / "info.json")
    modality = _read_json(meta_dir / "modality.json")
    episodes_meta = _read_jsonl(meta_dir / "episodes.jsonl")

    total_videos = _as_int(
        info.get("total_videos"), context="meta/info.json total_videos"
    )
    if total_videos <= 0:
        raise RuntimeError(
            f"video_decode_missing: info.total_videos must be > 0, got {total_videos}"
        )

    video_map_path = meta_dir / "video_map.json"
    if not video_map_path.is_file():
        raise RuntimeError(
            f"video_decode_missing: expected with-video dataset to contain {video_map_path}"
        )
    video_map = _read_json(video_map_path)

    features_raw = info.get("features")
    if not isinstance(features_raw, dict):
        raise ValueError(
            "dataset_contract_invalid: meta/info.json features must be an object"
        )
    features = cast(Mapping[str, object], features_raw)
    video_feature_keys = [
        str(key)
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    if not video_feature_keys:
        raise RuntimeError(
            "video_decode_missing: meta/info.json must expose at least one dtype=video feature"
        )

    modality_video_raw = modality.get("video")
    if not isinstance(modality_video_raw, dict) or not modality_video_raw:
        raise RuntimeError(
            "video_decode_missing: meta/modality.json missing non-empty video mapping"
        )
    video_keys_available = sorted(str(k) for k in modality_video_raw.keys())

    if "observation.images.ego_view" in video_feature_keys:
        primary_video_feature = "observation.images.ego_view"
    else:
        primary_video_feature = str(sorted(video_feature_keys)[0])

    data_path_template = _as_str(
        info.get("data_path"), context="meta/info.json data_path"
    )
    chunks_size = _as_int(info.get("chunks_size"), context="meta/info.json chunks_size")

    records = _json_list_of_dicts(video_map.get("records"), context="video_map.records")
    video_map_by_episode: dict[int, JsonObject] = {}
    for rec in records:
        ep_idx = _as_int(
            rec.get("episode_index"), context="video_map.records[*].episode_index"
        )
        if ep_idx in video_map_by_episode:
            raise ValueError(
                f"dataset_contract_invalid: duplicate video_map episode_index={ep_idx}"
            )
        video_map_by_episode[ep_idx] = rec

    if len(records) != len(episodes_meta):
        raise ValueError(
            "dataset_contract_invalid: video_map.records count must match meta/episodes.jsonl count "
            f"(records={len(records)} episodes={len(episodes_meta)})"
        )

    required_columns = [
        "episode_index",
        "index",
        "recap_m2.t",
        "recap_m2.prompt_raw",
        "recap_m2.return_G",
    ]

    checked_samples = 0
    checked_episodes = 0
    probe_backends: list[str] = []
    frame_count_sources: list[str] = []
    episode_checks: list[JsonObject] = []

    ordered_eps = sorted(
        episodes_meta,
        key=lambda item: _as_int(
            item.get("episode_index"), context="meta/episodes episode_index"
        ),
    )
    for ep_meta in ordered_eps:
        if checked_samples >= max_samples:
            break
        episode_index = _as_int(
            ep_meta.get("episode_index"), context="meta/episodes episode_index"
        )
        episode_length = _as_int(
            ep_meta.get("length"), context=f"episode_index={episode_index} length"
        )
        if episode_length <= 0:
            raise ValueError(
                f"dataset_contract_invalid: episode_index={episode_index} has non-positive length={episode_length}"
            )
        recap_episode_id = _as_str(
            ep_meta.get("recap.episode_id"),
            context=f"episode_index={episode_index} recap.episode_id",
        )
        video_rec = video_map_by_episode.get(episode_index)
        if video_rec is None:
            raise RuntimeError(
                f"video_decode_missing: missing video_map record for episode_index={episode_index}"
            )
        video_rel = _as_str(
            video_rec.get("dst_mp4"), context=f"video_map[{episode_index}] dst_mp4"
        )
        video_abs = dataset_path / Path(video_rel)
        if not video_abs.is_file():
            raise RuntimeError(
                f"video_decode_missing: missing video file for episode_index={episode_index}: {video_abs}"
            )

        chunk_idx = int(episode_index) // int(chunks_size)
        parquet_rel = data_path_template.format(
            episode_chunk=int(chunk_idx),
            episode_index=int(episode_index),
        )
        parquet_abs = dataset_path / Path(parquet_rel)
        if not parquet_abs.is_file():
            raise FileNotFoundError(
                f"dataset_contract_invalid: missing parquet for episode_index={episode_index}: {parquet_abs}"
            )

        parquet_rows = _parquet_num_rows(parquet_abs)
        if parquet_rows != int(episode_length):
            raise RuntimeError(
                "dataset_contract_invalid: parquet row count must equal episode meta length "
                f"(episode_index={episode_index} parquet_rows={parquet_rows} episode_length={episode_length})"
            )
        new_length = _as_int(
            video_rec.get("new_length", episode_length),
            context=f"video_map[{episode_index}] new_length",
        )
        if new_length != int(episode_length):
            raise RuntimeError(
                "dataset_contract_invalid: video_map new_length must equal episode meta length "
                f"(episode_index={episode_index} new_length={new_length} episode_length={episode_length})"
            )

        frame_count, frame_count_source = _resolve_num_frames(video_abs)
        if frame_count is not None:
            frame_count_sources.append(str(frame_count_source))
            if frame_count < int(episode_length):
                raise RuntimeError(
                    "video_decode_missing: decoded frame count is shorter than dataset episode length "
                    f"(episode_index={episode_index} frame_count={frame_count} episode_length={episode_length})"
                )

        table = _parquet_read_table(parquet_abs, columns=required_columns)
        column_names = set(str(name) for name in getattr(table, "column_names", []))
        missing_columns = [col for col in required_columns if col not in column_names]
        if missing_columns:
            raise ValueError(
                "dataset_contract_invalid: missing required parquet columns "
                f"for episode_index={episode_index}: {missing_columns}"
            )

        remaining = int(max_samples - checked_samples)
        sample_rows = min(int(episode_length), int(remaining))
        indexes_raw = _table_slice_pylist(table, "index", sample_rows)
        episode_indexes_raw = _table_slice_pylist(table, "episode_index", sample_rows)
        t_raw = _table_slice_pylist(table, "recap_m2.t", sample_rows)
        prompt_raw_values = _table_slice_pylist(
            table, "recap_m2.prompt_raw", sample_rows
        )
        return_g_values = _table_slice_pylist(table, "recap_m2.return_G", sample_rows)

        indexes = [
            _as_int(value, context=f"episode_index={episode_index} parquet.index")
            for value in indexes_raw
        ]
        if indexes != list(range(sample_rows)):
            raise RuntimeError(
                "dataset_contract_invalid: deterministic current-step frame policy requires local index "
                f"to start at 0 and increase by 1 (episode_index={episode_index} indexes_preview={indexes[:8]})"
            )

        for value in episode_indexes_raw:
            if _as_int(
                value, context=f"episode_index={episode_index} parquet.episode_index"
            ) != int(episode_index):
                raise RuntimeError(
                    "dataset_contract_invalid: parquet episode_index mismatch within episode "
                    f"(episode_index={episode_index})"
                )

        t_checked_max = -1
        for value in t_raw:
            t_i = _as_int(value, context=f"episode_index={episode_index} recap_m2.t")
            if t_i < 0:
                raise RuntimeError(
                    f"dataset_contract_invalid: recap_m2.t must be >= 0, got {t_i}"
                )
            t_checked_max = max(t_checked_max, int(t_i))

        for prompt in prompt_raw_values:
            _ = _as_str(
                prompt, context=f"episode_index={episode_index} recap_m2.prompt_raw"
            )
        for value in return_g_values:
            g = _as_float(
                value, context=f"episode_index={episode_index} recap_m2.return_G"
            )
            if not math.isfinite(g):
                raise RuntimeError(
                    f"dataset_contract_invalid: recap_m2.return_G must be finite, got {g}"
                )

        frame_probe_indices = sorted({0, sample_rows - 1, episode_length - 1})
        used_backend: str | None = None
        for frame_index in frame_probe_indices:
            used_backend, _ = _probe_frame_decode(video_abs, int(frame_index))
        if used_backend is not None:
            probe_backends.append(str(used_backend))

        episode_checks.append(
            {
                "episode_index": int(episode_index),
                "recap_episode_id": str(recap_episode_id),
                "episode_length": int(episode_length),
                "parquet_rel": str(Path(parquet_rel).as_posix()),
                "video_rel": str(Path(video_rel).as_posix()),
                "samples_checked": int(sample_rows),
                "current_t_max_checked": int(t_checked_max),
            }
        )
        checked_samples += int(sample_rows)
        checked_episodes += 1

    if checked_samples <= 0:
        raise RuntimeError("dataset_contract_invalid: no samples were checked")

    contract = _base_contract(repo_root)
    return {
        "mode": "dataset",
        "pass": True,
        "dataset_path": str(dataset_path),
        "contract_version": str(contract["schema_version"]),
        "critic_type": str(contract["critic_type"]),
        "training_source": str(contract["training_source"]),
        "task_text_field": str(contract["task_text_field"]),
        "allow_future_frames": bool(contract["allow_future_frames"]),
        "formal_eval_scope": str(contract["formal_gate_dataset_scope"]),
        "upgrade_pending": str(contract["upgrade_pending"]),
        "total_episodes": int(len(episodes_meta)),
        "total_videos": int(total_videos),
        "episodes_checked": int(checked_episodes),
        "samples_checked": int(checked_samples),
        "video_keys_available": video_keys_available,
        "primary_video_feature": str(primary_video_feature),
        "deterministic_frame_policy": {
            "kind": "current_step_index",
            "index_column": "index",
            "step_column": "recap_m2.t",
            "allow_future_frames": False,
            "deterministic": True,
        },
        "split_guardrail": {
            "formal_eval_scope": "isaac_only",
            "sample_level_split_forbidden": True,
            "no_sample_level_leakage_assumptions": True,
        },
        "video_probe_backends": sorted(set(probe_backends)),
        "frame_count_sources": sorted(set(frame_count_sources)),
        "episodes": episode_checks,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="41_vlm_critic_contract_check.py",
        description=(
            "Freeze and verify the Task 1 multimodal critic contract for the G1 VLM critic bootstrap mainline."
        ),
    )
    _ = parser.add_argument(
        "--mode",
        required=True,
        choices=["spec", "negative", "verify-plan", "dataset"],
        help=(
            "spec=write authoritative JSON contract, negative=reject forbidden combinations, "
            "verify-plan=check plan text for required defaults and guardrails, "
            "dataset=validate the with-video dataset contract."
        ),
    )
    _ = parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional JSON artifact path for machine-readable results.",
    )
    _ = parser.add_argument(
        "--plan-path",
        type=str,
        default=str(DEFAULT_PLAN_PATH_REL),
        help="Plan file to validate in --mode verify-plan.",
    )
    _ = parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="with-video LeRobot dataset path for --mode dataset.",
    )
    _ = parser.add_argument(
        "--max-samples",
        type=int,
        default=int(DEFAULT_MAX_SAMPLES),
        help="Maximum number of parquet rows to validate in --mode dataset.",
    )
    _ = parser.add_argument(
        "--expect-task-text",
        type=str,
        default="",
        help="Override candidate task_text_field for validation modes.",
    )
    _ = parser.add_argument(
        "--allow-future-frames",
        action="store_true",
        help="Override candidate contract to allow future frames (for negative testing only).",
    )
    _ = parser.add_argument("--expect-labeler-backend", type=str, default="")
    _ = parser.add_argument("--expect-training-source", type=str, default="")
    _ = parser.add_argument("--expect-formal-gate-dataset-scope", type=str, default="")
    _ = parser.add_argument("--expect-public-warmstart-scope", type=str, default="")
    _ = parser.add_argument("--expect-working-base-model", type=str, default="")
    _ = parser.add_argument("--expect-upgrade-pending", type=str, default="")
    return parser


def _emit_result(
    result: MutableMapping[str, object], *, sentinel: str, output_json: Path | None
) -> None:
    result["sentinel"] = sentinel
    if output_json is not None:
        _write_json(output_json, result)
        print(f"[INFO] wrote_json: {output_json}")
    print(f"SENTINEL:{sentinel}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    mode = _ns_str(args, "mode")

    repo_root = _repo_root()
    output_json: Path | None = None
    output_json_raw = _ns_str(args, "output_json").strip()
    if output_json_raw:
        output_json = _resolve_path(
            repo_root, output_json_raw, default_rel=DEFAULT_OUTPUT_JSON_REL
        )

    try:
        if mode == "spec":
            contract = _build_candidate_contract(repo_root, args)
            violations = _validate_contract(contract)
            spec_result: JsonObject = {
                "mode": "spec",
                "pass": not violations,
                "violations": violations,
                "contract": contract,
            }
            if violations:
                _emit_result(
                    spec_result, sentinel=SPEC_FAIL_SENTINEL, output_json=output_json
                )
                return 1
            _emit_result(
                spec_result, sentinel=SPEC_PASS_SENTINEL, output_json=output_json
            )
            return 0

        if mode == "negative":
            candidate = _build_candidate_contract(repo_root, args)
            deviations_requested = any(
                [
                    bool(_ns_str(args, "expect_task_text").strip()),
                    bool(_ns_flag(args, "allow_future_frames")),
                    bool(_ns_str(args, "expect_labeler_backend").strip()),
                    bool(_ns_str(args, "expect_training_source").strip()),
                    bool(_ns_str(args, "expect_formal_gate_dataset_scope").strip()),
                    bool(_ns_str(args, "expect_public_warmstart_scope").strip()),
                    bool(_ns_str(args, "expect_working_base_model").strip()),
                    bool(_ns_str(args, "expect_upgrade_pending").strip()),
                ]
            )
            if not deviations_requested:
                raise ValueError(
                    "negative mode requires at least one override to exercise a forbidden combination"
                )

            violations = _validate_contract(candidate)
            negative_result: JsonObject = {
                "mode": "negative",
                "pass": False,
                "candidate": candidate,
                "violations": violations,
                "violation_count": len(violations),
            }
            if not violations:
                _emit_result(
                    negative_result,
                    sentinel=NEGATIVE_UNEXPECTED_PASS_SENTINEL,
                    output_json=output_json,
                )
                return 1
            _emit_result(
                negative_result,
                sentinel=NEGATIVE_REJECTED_SENTINEL,
                output_json=output_json,
            )
            return 2

        if mode == "verify-plan":
            plan_path = _resolve_path(
                repo_root, _ns_str(args, "plan_path"), default_rel=DEFAULT_PLAN_PATH_REL
            )
            verification = _verify_plan(plan_path)
            verification_pass = bool(verification["pass"])
            verify_plan_result: JsonObject = {
                "mode": "verify-plan",
                **verification,
            }
            if not verification_pass:
                _emit_result(
                    verify_plan_result,
                    sentinel=VERIFY_PLAN_FAIL_SENTINEL,
                    output_json=output_json,
                )
                return 1
            _emit_result(
                verify_plan_result,
                sentinel=VERIFY_PLAN_PASS_SENTINEL,
                output_json=output_json,
            )
            return 0

        if mode == "dataset":
            dataset_path_raw = _ns_str(args, "dataset_path").strip()
            if not dataset_path_raw:
                raise ValueError("dataset mode requires --dataset-path")
            dataset_path = _resolve_path(repo_root, dataset_path_raw, default_rel="")
            dataset_result = _resolve_dataset_summary(
                repo_root,
                dataset_path,
                max_samples=int(getattr(args, "max_samples", 0)),
            )
            _emit_result(
                dataset_result,
                sentinel=DATASET_PASS_SENTINEL,
                output_json=output_json,
            )
            return 0

        raise ValueError(f"Unsupported mode: {mode!r}")

    except Exception as exc:
        fail_sentinel = SPEC_FAIL_SENTINEL
        if mode == "negative":
            fail_sentinel = NEGATIVE_REJECTED_SENTINEL
        elif mode == "verify-plan":
            fail_sentinel = VERIFY_PLAN_FAIL_SENTINEL
        elif mode == "dataset":
            fail_sentinel = DATASET_FAIL_SENTINEL
        result: JsonObject = {
            "mode": mode,
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _emit_result(result, sentinel=fail_sentinel, output_json=output_json)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
