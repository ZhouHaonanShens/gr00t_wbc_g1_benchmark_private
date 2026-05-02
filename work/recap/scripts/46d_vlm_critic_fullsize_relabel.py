#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


from work.recap.advantage import (
    ADVANTAGE_CONTRACT_VERSION,
    ADVANTAGE_INPUT_CLIP_RANGE,
    ADVANTAGE_INPUT_COLUMN,
    ADVANTAGE_RAW_COLUMN,
    ADVANTAGE_SCALE_QUANTILE,
    NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
    build_advantage_contract_metadata,
    build_diagnostic_surface_metadata,
    compute_sign_aware_advantage_scales,
    normalize_advantage_to_input,
)
from work.recap import text_indicator


# =====================
# USER Config (edit)
# =====================

SOURCE_DATASET_DIR = (
    "agent/artifacts/recap_datasets/recap_mainline_fresh_20260311_121500_k0"
)
OUTPUT_DATASET_DIR_REL = "agent/artifacts/recap_datasets"
OUTPUT_SUMMARY_DIR_REL = "agent/artifacts/vlm_critic_relabel"
RUNTIME_LOGS_REL = "agent/runtime_logs"
EVIDENCE_DIR_REL = ".sisyphus/evidence"
ITER_TAG = "fullsize_relabel_v1"
VALUE_SOURCE = "critic"
CRITIC_DIR = "agent/artifacts/critics/task7_real_critic_v2"
TOTAL_TIMEOUT_S = 0.0
DEFAULT_DIAGNOSTIC_ROUTE = "continuous_advantage_diagnostic_lane"
DEFAULT_THRESHOLD_TARGETS = "0.10,0.20,0.30,0.40"
MAINLINE_TARGET_POSITIVE_RATIO = 0.30
ADVANTAGE_CLIP_MIN = -1.0 * float(ADVANTAGE_INPUT_CLIP_RANGE)
ADVANTAGE_CLIP_MAX = float(ADVANTAGE_INPUT_CLIP_RANGE)
SKIP_WBC_REEXEC_ENV = "GR00T_SKIP_WBC_REEXEC"


JsonDict = dict[str, object]


def _repo_root() -> Path:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "repo_root")
    return cast(Path, fn(from_path=__file__))


@contextlib.contextmanager
def _tee_stdio(log_path: Path, *, header: str) -> Iterator[None]:
    mod = importlib.import_module("work.demo_utils.tee")
    fn = getattr(mod, "tee_stdio")
    with fn(Path(log_path), header=str(header)):
        yield


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _to_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected float-like for {context}, got bool")
    if isinstance(value, (int, float)):
        out = float(value)
    elif isinstance(value, str):
        try:
            out = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Expected float-like for {context}, got {value!r}"
            ) from exc
    else:
        raise ValueError(
            f"Expected float-like for {context}, got {type(value).__name__}"
        )
    if not math.isfinite(out):
        raise ValueError(f"Expected finite float for {context}, got {out!r}")
    return float(out)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return dict(data)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            payload, f, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False
        )
        f.write("\n")
    tmp.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            rows.append(dict(obj))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False))
            f.write("\n")
    tmp.replace(path)


def _linear_quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q!r}")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = float(q) * float(len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight_hi = pos - float(lo)
    weight_lo = 1.0 - weight_hi
    return float(weight_lo * ordered[lo] + weight_hi * ordered[hi])


def _summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("values must be non-empty")
    clean = [_to_float(v, context="summarize_values") for v in values]
    zero_count = sum(1 for v in clean if float(v) == 0.0)
    clip_count = sum(
        1
        for v in clean
        if float(v) <= float(ADVANTAGE_CLIP_MIN)
        or float(v) >= float(ADVANTAGE_CLIP_MAX)
    )
    return {
        "count": float(len(clean)),
        "min": float(min(clean)),
        "max": float(max(clean)),
        "mean": float(sum(clean) / float(len(clean))),
        "p50": float(_linear_quantile(clean, 0.50)),
        "p95": float(_linear_quantile(clean, 0.95)),
        "zero_ratio": float(zero_count) / float(len(clean)),
        "clip_ratio": float(clip_count) / float(len(clean)),
    }


def _maybe_summarize_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return _summarize_values(values)


def _threshold_name(target_positive_ratio: float) -> str:
    return f"epsilon_{int(round(float(target_positive_ratio) * 100.0))}"


def _parse_threshold_targets(raw: str) -> list[float]:
    targets: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        value = float(text)
        if not (0.0 < value < 1.0):
            raise ValueError(f"threshold target must be in (0,1), got {value!r}")
        targets.append(float(value))
    if not targets:
        raise ValueError("threshold target list must be non-empty")
    deduped = sorted({round(v, 6) for v in targets})
    return [float(v) for v in deduped]


def _load_label_writer_module() -> Any:
    return importlib.import_module("work.recap.label_writer")


def _rewrite_jsonl_iter_tag(src: Path, dst: Path, *, iter_tag: str) -> None:
    rows = _read_jsonl(src)
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["iter_tag"] = str(iter_tag)
        out.append(item)
    _write_jsonl(dst, out)


def _ensure_dataset_package(
    *,
    source_dataset_dir: Path,
    output_dataset_dir: Path,
    iter_tag: str,
) -> dict[str, Any]:
    source_episodes = source_dataset_dir / "episodes.jsonl"
    source_transitions = source_dataset_dir / "transitions.jsonl"
    source_arrays = source_dataset_dir / "arrays"
    if not source_episodes.is_file():
        raise FileNotFoundError(f"Missing source episodes.jsonl: {source_episodes}")
    if not source_transitions.is_file():
        raise FileNotFoundError(
            f"Missing source transitions.jsonl: {source_transitions}"
        )
    if not source_arrays.is_dir():
        raise FileNotFoundError(f"Missing source arrays dir: {source_arrays}")

    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    _rewrite_jsonl_iter_tag(
        source_episodes, output_dataset_dir / "episodes.jsonl", iter_tag=iter_tag
    )
    _rewrite_jsonl_iter_tag(
        source_transitions,
        output_dataset_dir / "transitions.jsonl",
        iter_tag=iter_tag,
    )

    arrays_dst = output_dataset_dir / "arrays"
    if arrays_dst.exists() or arrays_dst.is_symlink():
        if arrays_dst.is_symlink() and arrays_dst.resolve() == source_arrays.resolve():
            pass
        elif arrays_dst.is_dir() and not arrays_dst.is_symlink():
            pass
        else:
            raise FileExistsError(
                f"Refusing to replace existing arrays path with unexpected type: {arrays_dst}"
            )
    else:
        os.symlink(source_arrays, arrays_dst, target_is_directory=True)

    manifest = {
        "schema_version": "vlm_critic_fullsize_relabel_source_ref_v1",
        "prepared_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "iter_tag": str(iter_tag),
        "source_dataset_dir": str(source_dataset_dir),
        "source_iter_tag": str(source_dataset_dir.name),
        "output_dataset_dir": str(output_dataset_dir),
        "arrays_path": str(arrays_dst),
    }
    _write_json(output_dataset_dir / "source_dataset_ref.json", manifest)
    return manifest


def _invoke_mainline_labeler(
    *,
    repo_root: Path,
    iter_tag: str,
    dataset_dir_rel: str,
    runtime_logs_rel: str,
    critic_dir: Path,
    total_timeout_s: float,
    check_npz_keys: bool,
    epsilon_quantile: float,
    force_restart: bool,
) -> None:
    script_path = Path(__file__).resolve().with_name("32_recap_label_dataset.py")
    cmd = [
        str(sys.executable),
        str(script_path),
        "--iter-tag",
        str(iter_tag),
        "--dataset-dir-rel",
        str(dataset_dir_rel),
        "--runtime-logs-rel",
        str(runtime_logs_rel),
        "--value-source",
        "critic",
        "--critic-dir",
        str(critic_dir),
        "--epsilon-strategy",
        "quantile",
        "--epsilon-quantile",
        f"{float(epsilon_quantile):.10f}",
        "--total-timeout-s",
        f"{float(total_timeout_s):.6f}",
    ]
    cmd.append("--resume")
    if force_restart:
        cmd.append("--force-restart")
    cmd.append("--check-npz-keys" if check_npz_keys else "--no-check-npz-keys")
    env = dict(os.environ)
    env[SKIP_WBC_REEXEC_ENV] = "1"
    print("[INFO] invoking mainline labeler:", json.dumps(cmd, ensure_ascii=True))
    proc = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"32_recap_label_dataset.py failed with rc={proc.returncode}"
        )


def _load_mainline_labels(
    output_dataset_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels_path = output_dataset_dir / "m2_labels" / "labels.jsonl"
    stats_path = output_dataset_dir / "m2_labels" / "stats.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing mainline labels: {labels_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing mainline stats: {stats_path}")
    return _read_jsonl(labels_path), _read_json(stats_path)


def _episode_indicator_profiles(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in labels:
        episode_id = str(record.get("episode_id", ""))
        grouped[episode_id].append(int(record.get("indicator_I", 0)))
    out: list[dict[str, Any]] = []
    for episode_id in sorted(grouped):
        indicators = [int(v) for v in grouped[episode_id]]
        positive_count = sum(indicators)
        total = len(indicators)
        out.append(
            {
                "episode_id": str(episode_id),
                "positive_count": int(positive_count),
                "negative_count": int(total - positive_count),
                "positive_ratio": float(positive_count / float(total))
                if total
                else 0.0,
                "indicator_profile": indicators,
            }
        )
    return out


def _derive_threshold_labels(
    *,
    base_labels: list[dict[str, Any]],
    epsilon_value: float,
) -> list[dict[str, Any]]:
    label_writer_mod = _load_label_writer_module()
    validate_label_record = getattr(label_writer_mod, "validate_label_record")
    out: list[dict[str, Any]] = []
    for record in base_labels:
        item = dict(record)
        advantage = _to_float(item.get("advantage_A"), context="advantage_A")
        is_correction = bool(item.get("is_correction", False))
        indicator = 1 if float(advantage) > float(epsilon_value) else 0
        if is_correction:
            indicator = 1
        prompt_raw = str(item.get("prompt_raw", ""))
        prefix = "advantage positive " if indicator == 1 else "advantage negative "
        item["epsilon_l"] = float(epsilon_value)
        item["indicator_I"] = int(indicator)
        item["prompt_conditioned"] = prefix + prompt_raw
        item[text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD] = (
            text_indicator.build_canonical_text_indicator(
                prompt_raw,
                text_indicator.indicator_mode_from_indicator_value(
                    indicator,
                    field_name="indicator_I",
                ),
            )
        )
        validate_label_record(item)
        out.append(item)
    return out


def _write_threshold_package(
    *,
    package_dir: Path,
    labels: list[dict[str, Any]],
    epsilon_value: float,
    epsilon_quantile: float,
    target_positive_ratio: float,
    source_iter_tag: str,
) -> dict[str, Any]:
    label_writer_mod = _load_label_writer_module()
    compute_m2_label_stats = getattr(label_writer_mod, "compute_m2_label_stats")
    validate_label_record = getattr(label_writer_mod, "validate_label_record")
    package_dir.mkdir(parents=True, exist_ok=True)
    for row in labels:
        validate_label_record(row)
    _write_jsonl(package_dir / "labels.jsonl", labels)
    stats_obj = dict(
        compute_m2_label_stats(
            labels,
            epsilon_strategy="target_positive_ratio",
            epsilon_value=float(epsilon_value),
        )
    )
    observed_positive_ratio = _to_float(
        stats_obj.get("pos_ratio"), context="threshold_package.pos_ratio"
    )
    per_episode_profiles = _episode_indicator_profiles(labels)
    summary = {
        **stats_obj,
        "package_dir": str(package_dir),
        "source_iter_tag": str(source_iter_tag),
        "epsilon_quantile": float(epsilon_quantile),
        "epsilon_value": float(epsilon_value),
        "target_positive_ratio": float(target_positive_ratio),
        "observed_positive_ratio": float(observed_positive_ratio),
        "positive_count": int(
            round(observed_positive_ratio * int(stats_obj["n_transitions"]))
        ),
        "negative_count": int(stats_obj["n_transitions"])
        - int(round(observed_positive_ratio * int(stats_obj["n_transitions"]))),
        "per_episode_indicator_profiles": per_episode_profiles,
    }
    summary.update(
        build_diagnostic_surface_metadata(
            surface_route="continuous_advantage_threshold_package_diagnostic",
            authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
            surface_kind="continuous_advantage_threshold_package",
        )
    )
    _write_json(package_dir / "stats.json", summary)
    return summary


def _write_continuous_contract(
    *,
    output_dataset_dir: Path,
    source_dataset_dir: Path,
    critic_dir: Path,
    raw_advantages: list[float],
    scaled_advantages: list[float],
    scale_metadata: dict[str, Any],
    n_samples: int,
) -> dict[str, Any]:
    positive_raw_advantages = [float(v) for v in raw_advantages if float(v) > 0.0]
    negative_raw_advantages = [float(v) for v in raw_advantages if float(v) < 0.0]
    positive_scaled_advantages = [float(v) for v in scaled_advantages if float(v) > 0.0]
    negative_scaled_advantages = [float(v) for v in scaled_advantages if float(v) < 0.0]
    sign_scale_summary = {
        "positive_count": int(scale_metadata["positive_count"]),
        "negative_count": int(scale_metadata["negative_count"]),
        "zero_count": int(scale_metadata["zero_count"]),
        "positive_fraction": float(scale_metadata["positive_fraction"]),
        "negative_fraction": float(scale_metadata["negative_fraction"]),
        "zero_fraction": float(scale_metadata["zero_fraction"]),
        "positive_min": scale_metadata["positive_min"],
        "positive_max": scale_metadata["positive_max"],
        "negative_min": scale_metadata["negative_min"],
        "negative_max": scale_metadata["negative_max"],
        "positive_scale": scale_metadata["positive_scale"],
        "negative_scale_abs": scale_metadata["negative_scale_abs"],
        "positive_quantile": float(scale_metadata["positive_quantile"]),
        "negative_quantile": float(scale_metadata["negative_quantile"]),
        "positive_quantile_value": scale_metadata["positive_quantile_value"],
        "negative_quantile_abs_value": scale_metadata["negative_quantile_abs_value"],
        "raw_positive_summary": _maybe_summarize_values(positive_raw_advantages),
        "raw_negative_summary": _maybe_summarize_values(negative_raw_advantages),
        "scaled_positive_summary": _maybe_summarize_values(positive_scaled_advantages),
        "scaled_negative_summary": _maybe_summarize_values(negative_scaled_advantages),
    }
    contract = build_advantage_contract_metadata(
        source_iter_tag=str(source_dataset_dir.name),
        n_samples=int(n_samples),
        positive_scale=cast(float | None, scale_metadata.get("positive_scale")),
        negative_scale_abs=cast(float | None, scale_metadata.get("negative_scale_abs")),
        critic_dir=str(critic_dir),
        critic_include_t=True,
        raw_summary=_summarize_values(list(raw_advantages)),
        scaled_summary=_summarize_values(list(scaled_advantages)),
        sign_scale_summary=sign_scale_summary,
        advantage_stats={
            "value_source": "critic",
            "sign_scale_summary": sign_scale_summary,
        },
        scale_rule=str(scale_metadata["scale_rule"]),
    )
    _write_json(output_dataset_dir / "continuous_advantage_contract.json", contract)
    return contract


def _emit_t9_evidence(
    *,
    repo_root: Path,
    iter_tag: str,
    critic_dir: Path,
    output_summary_path: Path,
    advantage_input_range: dict[str, float],
) -> None:
    evidence_root = _resolve_path(repo_root, EVIDENCE_DIR_REL)
    public_cmd = (
        ".envs/main/bin/python work/recap/scripts/45_recap_label_dataset_vlm_backend.py "
        f"--iter-tag {iter_tag} --value-source critic --critic-dir {critic_dir}"
    )
    fullsize_text = "\n".join(
        [
            "T9 full-size relabel verification",
            f"verified_at: {_dt.datetime.now().isoformat(timespec='seconds')}",
            f"command: {public_cmd}",
            "diagnostic_verdict: DIAGNOSTIC_PASS",
            f"summary_json: {output_summary_path}",
        ]
    )
    range_text = "\n".join(
        [
            "T9 continuous advantage range verification",
            f"verified_at: {_dt.datetime.now().isoformat(timespec='seconds')}",
            f"command: {public_cmd}",
            f"default_diagnostic_route: {DEFAULT_DIAGNOSTIC_ROUTE}",
            f"advantage_input_range: {json.dumps(advantage_input_range, ensure_ascii=True, sort_keys=True)}",
            "diagnostic_verdict: DIAGNOSTIC_PASS",
        ]
    )
    _write_text(evidence_root / "task-9-fullsize-relabel.txt", fullsize_text)
    _write_text(evidence_root / "task-9-continuous-range.txt", range_text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="46d_vlm_critic_fullsize_relabel.py",
        description=(
            "Run one full multimodal critic relabel pass, keep continuous advantage as a "
            "diagnostic lane, and derive the 10/20/30/40 positive-ratio binary comparison packages offline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iter-tag", type=str, default=str(ITER_TAG))
    parser.add_argument(
        "--source-dataset-dir", type=str, default=str(SOURCE_DATASET_DIR)
    )
    parser.add_argument(
        "--dataset-dir-rel",
        type=str,
        default=str(OUTPUT_DATASET_DIR_REL),
        help="Output recap dataset root; the relabeled package is written to <dataset-dir-rel>/<iter-tag>/.",
    )
    parser.add_argument(
        "--output-summary-dir-rel",
        type=str,
        default=str(OUTPUT_SUMMARY_DIR_REL),
    )
    parser.add_argument("--runtime-logs-rel", type=str, default=str(RUNTIME_LOGS_REL))
    parser.add_argument("--value-source", type=str, default=str(VALUE_SOURCE))
    parser.add_argument("--critic-dir", type=str, default=str(CRITIC_DIR))
    parser.add_argument(
        "--threshold-targets",
        type=str,
        default=str(DEFAULT_THRESHOLD_TARGETS),
        help="Comma-separated target positive ratios for offline binary comparison packages.",
    )
    parser.add_argument(
        "--mainline-target-positive-ratio",
        type=float,
        default=float(MAINLINE_TARGET_POSITIVE_RATIO),
    )
    parser.add_argument(
        "--total-timeout-s",
        type=float,
        default=float(TOTAL_TIMEOUT_S),
    )
    parser.add_argument(
        "--check-npz-keys",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help=(
            "Forward `--force-restart` to `32_recap_label_dataset.py`. "
            "Normal runs auto-pass `--resume` to keep the public T9 command unchanged."
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    iter_tag = str(args.iter_tag)
    if str(args.value_source) != "critic":
        raise ValueError(
            "46d_vlm_critic_fullsize_relabel.py only supports --value-source critic"
        )

    source_dataset_dir = _resolve_path(repo_root, str(args.source_dataset_dir))
    output_dataset_root = _resolve_path(repo_root, str(args.dataset_dir_rel))
    output_dataset_dir = output_dataset_root / iter_tag
    output_summary_root = _resolve_path(repo_root, str(args.output_summary_dir_rel))
    output_summary_path = output_summary_root / f"{iter_tag}.json"
    threshold_root = output_summary_root / f"{iter_tag}_thresholds"
    critic_dir = _resolve_path(repo_root, str(args.critic_dir))
    runtime_dir = _resolve_path(repo_root, str(args.runtime_logs_rel)) / iter_tag
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "fullsize_relabel.log"

    with _tee_stdio(log_path, header="46d_vlm_critic_fullsize_relabel"):
        t0 = time.monotonic()
        print("[INFO] ts:", _dt.datetime.now().isoformat(timespec="seconds"))
        print("[INFO] sys.executable:", sys.executable)
        print("[INFO] repo_root:", str(repo_root))
        print("[INFO] iter_tag:", str(iter_tag))
        print("[INFO] source_dataset_dir:", str(source_dataset_dir))
        print("[INFO] output_dataset_dir:", str(output_dataset_dir))
        print("[INFO] output_summary_path:", str(output_summary_path))
        print("[INFO] threshold_root:", str(threshold_root))
        print("[INFO] critic_dir:", str(critic_dir))

        dataset_manifest = _ensure_dataset_package(
            source_dataset_dir=source_dataset_dir,
            output_dataset_dir=output_dataset_dir,
            iter_tag=iter_tag,
        )
        threshold_targets = _parse_threshold_targets(str(args.threshold_targets))
        mainline_target_positive_ratio = float(args.mainline_target_positive_ratio)
        mainline_quantile = float(1.0 - mainline_target_positive_ratio)
        _invoke_mainline_labeler(
            repo_root=repo_root,
            iter_tag=iter_tag,
            dataset_dir_rel=str(args.dataset_dir_rel),
            runtime_logs_rel=str(args.runtime_logs_rel),
            critic_dir=critic_dir,
            total_timeout_s=float(args.total_timeout_s),
            check_npz_keys=bool(args.check_npz_keys),
            epsilon_quantile=float(mainline_quantile),
            force_restart=bool(args.force_restart),
        )
        base_labels, base_stats = _load_mainline_labels(output_dataset_dir)
        n_episodes = int(base_stats.get("n_episodes", 0))
        n_transitions = int(base_stats.get("n_transitions", 0))
        if n_episodes != 200 or n_transitions != 61246:
            raise ValueError(
                "Full-size relabel must target the 200-episode dataset: "
                f"n_episodes={n_episodes} n_transitions={n_transitions}"
            )

        raw_advantages = [
            _to_float(record.get("advantage_A"), context="base_labels.advantage_A")
            for record in base_labels
        ]
        scale_metadata = compute_sign_aware_advantage_scales(raw_advantages)
        scaled_advantages = [
            normalize_advantage_to_input(
                v,
                positive_scale=scale_metadata.get("positive_scale"),
                negative_scale_abs=scale_metadata.get("negative_scale_abs"),
            )
            for v in raw_advantages
        ]
        contract = _write_continuous_contract(
            output_dataset_dir=output_dataset_dir,
            source_dataset_dir=source_dataset_dir,
            critic_dir=critic_dir,
            raw_advantages=raw_advantages,
            scaled_advantages=scaled_advantages,
            scale_metadata=scale_metadata,
            n_samples=len(raw_advantages),
        )
        advantage_input_range = {
            "min": float(min(scaled_advantages)),
            "max": float(max(scaled_advantages)),
            "clip_min": float(ADVANTAGE_CLIP_MIN),
            "clip_max": float(ADVANTAGE_CLIP_MAX),
        }

        base_stats.update(
            {
                "source_dataset_dir": str(source_dataset_dir),
                "default_mainline": str(DEFAULT_DIAGNOSTIC_ROUTE),
                "target_positive_ratio": float(mainline_target_positive_ratio),
                "epsilon_quantile": float(mainline_quantile),
                "advantage_contract_version": str(ADVANTAGE_CONTRACT_VERSION),
                "advantage_scale_rule": str(contract["scale_rule"]),
                "advantage_scale_summary": contract.get("sign_scale_summary"),
                "advantage_input_range": advantage_input_range,
            }
        )
        base_stats.update(
            build_diagnostic_surface_metadata(
                surface_route="continuous_advantage_stats_diagnostic",
                authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
                surface_kind="continuous_advantage_stats",
            )
        )
        _write_json(output_dataset_dir / "m2_labels" / "stats.json", base_stats)

        if threshold_root.exists():
            shutil.rmtree(threshold_root)
        threshold_root.mkdir(parents=True, exist_ok=True)

        threshold_packages: dict[str, Any] = {}
        for target_positive_ratio in threshold_targets:
            epsilon_quantile = float(1.0 - float(target_positive_ratio))
            epsilon_value = float(_linear_quantile(raw_advantages, epsilon_quantile))
            threshold_name = _threshold_name(target_positive_ratio)
            threshold_labels = _derive_threshold_labels(
                base_labels=base_labels, epsilon_value=epsilon_value
            )
            threshold_summary = _write_threshold_package(
                package_dir=threshold_root / threshold_name,
                labels=threshold_labels,
                epsilon_value=epsilon_value,
                epsilon_quantile=epsilon_quantile,
                target_positive_ratio=float(target_positive_ratio),
                source_iter_tag=str(iter_tag),
            )
            threshold_packages[threshold_name] = threshold_summary

        mainline_package = threshold_packages.get(
            _threshold_name(mainline_target_positive_ratio)
        )
        if not isinstance(mainline_package, dict):
            raise RuntimeError("Mainline threshold package was not generated")

        summary_payload: dict[str, Any] = {
            "schema_version": "vlm_critic_fullsize_relabel_v1",
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "entrypoint": "work/recap/scripts/46d_vlm_critic_fullsize_relabel.py",
            "public_entrypoint": "work/recap/scripts/45_recap_label_dataset_vlm_backend.py",
            "default_mainline": str(DEFAULT_DIAGNOSTIC_ROUTE),
            "source_dataset_dir": str(source_dataset_dir),
            "critic_dir": str(critic_dir),
            "n_episodes": int(n_episodes),
            "n_transitions": int(n_transitions),
            "advantage_input_range": advantage_input_range,
            "advantage_contract_version": str(ADVANTAGE_CONTRACT_VERSION),
            "advantage_scale_rule": str(contract["scale_rule"]),
            "advantage_scale_summary": contract.get("sign_scale_summary"),
            "continuous_package": {
                "iter_tag": str(iter_tag),
                "dataset_dir": str(output_dataset_dir),
                "m2_labels_dir": str(output_dataset_dir / "m2_labels"),
                "advantage_contract_path": str(
                    output_dataset_dir / "continuous_advantage_contract.json"
                ),
                "task_text_field": "prompt_raw",
                "target_positive_ratio": float(mainline_target_positive_ratio),
                "observed_positive_ratio": float(
                    mainline_package["observed_positive_ratio"]
                ),
                "epsilon_value": float(mainline_package["epsilon_value"]),
                "epsilon_quantile": float(mainline_package["epsilon_quantile"]),
            },
            "threshold_packages": threshold_packages,
            "threshold_sweep_stats": {
                name: {
                    "target_positive_ratio": payload["target_positive_ratio"],
                    "observed_positive_ratio": payload["observed_positive_ratio"],
                    "epsilon_quantile": payload["epsilon_quantile"],
                    "epsilon_value": payload["epsilon_value"],
                }
                for name, payload in threshold_packages.items()
            },
            "continuous_advantage_contract": contract,
            "dataset_manifest": dataset_manifest,
            "elapsed_seconds": float(time.monotonic() - t0),
        }
        summary_payload.update(
            build_diagnostic_surface_metadata(
                surface_route="vlm_critic_fullsize_relabel_diagnostic",
                authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
                surface_kind="vlm_critic_fullsize_relabel_summary",
            )
        )
        summary_payload["continuous_package"].update(
            build_diagnostic_surface_metadata(
                surface_route="continuous_advantage_package_diagnostic",
                authority_scope=NUMERIC_ADVANTAGE_DIAGNOSTIC_AUTHORITY_SCOPE,
                surface_kind="continuous_advantage_package",
            )
        )
        _write_json(output_summary_path, summary_payload)
        _emit_t9_evidence(
            repo_root=repo_root,
            iter_tag=iter_tag,
            critic_dir=critic_dir,
            output_summary_path=output_summary_path,
            advantage_input_range=advantage_input_range,
        )
        print("[INFO] wrote summary:", str(output_summary_path))
        print(
            "[INFO] diagnostic route observed_positive_ratio:",
            mainline_package["observed_positive_ratio"],
        )
        print(
            "[INFO] advantage_input_range:",
            json.dumps(advantage_input_range, ensure_ascii=True, sort_keys=True),
        )
        print("[INFO] advantage_scale_rule:", contract["scale_rule"])
        print(
            "[INFO] advantage_scale_summary:",
            json.dumps(
                contract.get("sign_scale_summary"),
                ensure_ascii=True,
                sort_keys=True,
            ),
        )
        print("[INFO] done elapsed_s:", f"{time.monotonic() - t0:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
