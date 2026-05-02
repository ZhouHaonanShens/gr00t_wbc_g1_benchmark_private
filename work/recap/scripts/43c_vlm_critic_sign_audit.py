#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# USER Config (edit)
DEFAULT_SOURCE_TRAIN_MANIFEST = (
    "agent/artifacts/vlm_critic_manifests/task7_real_train_build.full_input.json"
)
DEFAULT_SOURCE_VAL_MANIFEST = (
    "agent/artifacts/vlm_critic_manifests/task7_real_val_build.full_input.json"
)
DEFAULT_SOURCE_TEST_FULL_MANIFEST = (
    "agent/artifacts/vlm_critic_manifests/task7_real_test_build.full_input.json"
)
DEFAULT_SOURCE_TEST_PROMPT_MANIFEST = (
    "agent/artifacts/vlm_critic_manifests/task7_real_test_build.prompt_only.json"
)
DEFAULT_SOURCE_TEST_VISION_MANIFEST = (
    "agent/artifacts/vlm_critic_manifests/task7_real_test_build.vision_only.json"
)
DEFAULT_TRAIN_MANIFEST = (
    "agent/artifacts/vlm_critic_manifests/task7_real_train_build_v2.full_input.json"
)
DEFAULT_VAL_MANIFEST = (
    "agent/artifacts/vlm_critic_manifests/task7_real_val_build_v2.full_input.json"
)
DEFAULT_TEST_FULL_MANIFEST = (
    "agent/artifacts/vlm_critic_manifests/task7_real_test_build_v2.full_input.json"
)
DEFAULT_TEST_PROMPT_MANIFEST = (
    "agent/artifacts/vlm_critic_manifests/task7_real_test_build_v2.prompt_only.json"
)
DEFAULT_TEST_VISION_MANIFEST = (
    "agent/artifacts/vlm_critic_manifests/task7_real_test_build_v2.vision_only.json"
)
DEFAULT_OUTPUT_JSON = "agent/artifacts/vlm_critic_offline_gate/task7_sign_audit_v2.json"
PASS_SENTINEL = "SIGN_AUDIT_OK"
FAIL_SENTINEL = "SIGN_AUDIT_BLOCKED"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.critic_vlm.manifest import load_vlm_critic_manifest  # noqa: E402


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ManifestSpec:
    role: str
    path: Path
    expected_input_mode: dict[str, bool | str]
    require_t_norm_only_note: bool


def _resolve_path(raw_path: str | None, *, default_rel: str) -> Path:
    value = str(raw_path or default_rel)
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _read_json(path: Path) -> JsonObject:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"json_invalid: expected object in {path}")
    return cast(JsonObject, data)


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        _ = f.write("\n")
    _ = tmp_path.replace(path)


def _emit_result(*, sentinel: str, payload: JsonObject, output_json: Path) -> None:
    _write_json(output_json, payload)
    print(f"[INFO] wrote_json: {output_json}")
    print(f"SENTINEL:{sentinel}")


def _ensure_object(value: object, *, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"manifest_schema_invalid: {context} must be an object")
    return cast(JsonObject, value)


def _ensure_list(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"manifest_schema_invalid: {context} must be a list")
    return list(value)


def _expected_input_mode(role: str) -> dict[str, bool | str]:
    sample_mode = "current_step_single_view_ego_view"
    by_role: dict[str, dict[str, bool | str]] = {
        "train_full": {
            "allow_future_frames": False,
            "sample_mode": sample_mode,
            "use_prompt": False,
            "use_side_channels": True,
            "use_video": True,
        },
        "val_full": {
            "allow_future_frames": False,
            "sample_mode": sample_mode,
            "use_prompt": False,
            "use_side_channels": True,
            "use_video": True,
        },
        "test_full": {
            "allow_future_frames": False,
            "sample_mode": sample_mode,
            "use_prompt": False,
            "use_side_channels": True,
            "use_video": True,
        },
        "test_prompt_only": {
            "allow_future_frames": False,
            "sample_mode": sample_mode,
            "use_prompt": False,
            "use_side_channels": False,
            "use_video": False,
        },
        "test_vision_only": {
            "allow_future_frames": False,
            "sample_mode": sample_mode,
            "use_prompt": False,
            "use_side_channels": False,
            "use_video": True,
        },
    }
    return dict(by_role[role])


def _role_title(role: str) -> str:
    titles = {
        "train_full": "task7_real_train_build_v2.full_input",
        "val_full": "task7_real_val_build_v2.full_input",
        "test_full": "task7_real_test_build_v2.full_input",
        "test_prompt_only": "task7_real_test_build_v2.prompt_only",
        "test_vision_only": "task7_real_test_build_v2.vision_only",
    }
    return titles[role]


def _remediation_notes(role: str) -> JsonObject:
    notes: JsonObject = {
        "task_id": "T7F1",
        "remediation_version": "v2",
        "prompt_semantics": "constant_query_only_via_use_prompt_false",
        "prompt_constant_query": "DEFAULT_CRITIC_QUERY",
        "clean_controls_required": True,
        "prepared_for": "T7F2",
        "prepared_by": "43c_vlm_critic_sign_audit.py",
    }
    if role in {"train_full", "val_full"}:
        notes["next_model_cycle"] = {
            "use_proprio": False,
            "side_channel_interpretation": "t_norm_centric_unless_T7F2_explicitly_reenables_parity",
            "reason": "avoid carrying forward the train-real versus eval-zero proprio mismatch",
        }
    return notes


def _rewrite_manifest_for_v2(
    *, source_path: Path, target_path: Path, role: str
) -> JsonObject:
    source_obj = _read_json(source_path)
    manifest_obj = dict(source_obj)
    input_mode = _ensure_object(manifest_obj.get("input_mode"), context="input_mode")
    expected = _expected_input_mode(role)
    input_mode.update(expected)
    manifest_obj["input_mode"] = input_mode
    manifest_obj["task7f1_remediation_v2"] = _remediation_notes(role)
    manifest_obj["derived_from_manifest"] = str(source_path)
    manifest_obj["manifest_label"] = _role_title(role)
    manifest_obj["control_cleanliness"] = {
        "prompt_neutralized": True,
        "constant_query_only": True,
        "side_channel_clean": bool(expected["use_side_channels"]),
        "video_enabled": bool(expected["use_video"]),
    }
    _write_json(target_path, manifest_obj)
    return {
        "role": role,
        "source_manifest": str(source_path),
        "target_manifest": str(target_path),
        "input_mode": dict(input_mode),
    }


def _manifest_specs(args: argparse.Namespace) -> list[ManifestSpec]:
    return [
        ManifestSpec(
            role="train_full",
            path=_resolve_path(args.train_manifest, default_rel=DEFAULT_TRAIN_MANIFEST),
            expected_input_mode=_expected_input_mode("train_full"),
            require_t_norm_only_note=True,
        ),
        ManifestSpec(
            role="val_full",
            path=_resolve_path(args.val_manifest, default_rel=DEFAULT_VAL_MANIFEST),
            expected_input_mode=_expected_input_mode("val_full"),
            require_t_norm_only_note=True,
        ),
        ManifestSpec(
            role="test_full",
            path=_resolve_path(
                args.test_full_manifest, default_rel=DEFAULT_TEST_FULL_MANIFEST
            ),
            expected_input_mode=_expected_input_mode("test_full"),
            require_t_norm_only_note=False,
        ),
        ManifestSpec(
            role="test_prompt_only",
            path=_resolve_path(
                args.test_prompt_manifest, default_rel=DEFAULT_TEST_PROMPT_MANIFEST
            ),
            expected_input_mode=_expected_input_mode("test_prompt_only"),
            require_t_norm_only_note=False,
        ),
        ManifestSpec(
            role="test_vision_only",
            path=_resolve_path(
                args.test_vision_manifest, default_rel=DEFAULT_TEST_VISION_MANIFEST
            ),
            expected_input_mode=_expected_input_mode("test_vision_only"),
            require_t_norm_only_note=False,
        ),
    ]


def _monotonic_summary(return_to_bins: dict[float, set[int]]) -> JsonObject:
    unique_pairs = sorted((float(g), min(bins)) for g, bins in return_to_bins.items())
    violation: JsonObject | None = None
    previous_g: float | None = None
    previous_bin: int | None = None
    for g, bin_idx in unique_pairs:
        if previous_g is not None and g > previous_g and previous_bin is not None:
            if bin_idx < previous_bin:
                violation = {
                    "previous_return_G": previous_g,
                    "previous_target_bin_index": previous_bin,
                    "current_return_G": g,
                    "current_target_bin_index": bin_idx,
                }
                break
        previous_g = g
        previous_bin = bin_idx
    ambiguous = [
        {"return_G": g, "target_bin_indexes": sorted(int(x) for x in bins)}
        for g, bins in sorted(return_to_bins.items())
        if len(bins) > 1
    ]
    return {
        "checked_unique_return_values": len(unique_pairs),
        "monotonic_non_decreasing": violation is None,
        "duplicate_return_targets_consistent": not ambiguous,
        "first_violation": violation,
        "ambiguous_returns_preview": ambiguous[:8],
        "min_return_G": unique_pairs[0][0] if unique_pairs else None,
        "max_return_G": unique_pairs[-1][0] if unique_pairs else None,
        "min_target_bin_index": unique_pairs[0][1] if unique_pairs else None,
        "max_target_bin_index": unique_pairs[-1][1] if unique_pairs else None,
    }


def _audit_single_manifest(spec: ManifestSpec) -> tuple[JsonObject, list[str]]:
    manifest_path = spec.path.resolve()
    raw_obj = _read_json(manifest_path)
    manifest = load_vlm_critic_manifest(manifest_path)
    blockers: list[str] = []

    raw_input_mode = _ensure_object(raw_obj.get("input_mode"), context="input_mode")
    expected = spec.expected_input_mode
    actual_input_mode = {
        "allow_future_frames": bool(raw_input_mode.get("allow_future_frames", False)),
        "sample_mode": str(raw_input_mode.get("sample_mode", "")),
        "use_prompt": bool(raw_input_mode.get("use_prompt", True)),
        "use_side_channels": bool(raw_input_mode.get("use_side_channels", True)),
        "use_video": bool(raw_input_mode.get("use_video", True)),
    }
    if actual_input_mode != expected:
        blockers.append(
            f"dirty_controls:{spec.role}: expected input_mode={expected}, got {actual_input_mode}"
        )

    if manifest.allow_future_frames:
        blockers.append(
            f"future_frames_forbidden:{spec.role}: source build allow_future_frames must stay False"
        )

    if raw_obj.get("sample_count") != len(
        _ensure_list(raw_obj.get("sample_ids"), context="sample_ids")
    ):
        blockers.append(
            f"manifest_schema_invalid:{spec.role}: sample_count must match len(sample_ids)"
        )

    if int(raw_obj.get("sample_count", -1)) != len(manifest.samples):
        blockers.append(
            f"manifest_schema_invalid:{spec.role}: sample_count must match loaded samples"
        )

    alignment_path = Path(str(raw_obj.get("alignment_report_path", ""))).expanduser()
    if not alignment_path.is_absolute():
        alignment_path = (manifest_path.parent / alignment_path).resolve()
    alignment = _read_json(alignment_path)
    if bool(alignment.get("allow_future_frames", True)):
        blockers.append(
            f"alignment_report_invalid:{spec.role}: alignment report allow_future_frames must be False"
        )
    if not bool(alignment.get("pass", False)):
        blockers.append(
            f"alignment_report_invalid:{spec.role}: alignment report pass must be true"
        )
    if int(alignment.get("future_frame_violation_count", -1)) != 0:
        blockers.append(
            f"future_frames_forbidden:{spec.role}: future_frame_violation_count must be 0"
        )
    if int(alignment.get("frame_t_mismatch_count", -1)) != 0:
        blockers.append(
            f"alignment_report_invalid:{spec.role}: frame_t_mismatch_count must be 0"
        )
    if int(alignment.get("episode_leakage_count", -1)) != 0:
        blockers.append(
            f"alignment_report_invalid:{spec.role}: episode_leakage_count must be 0"
        )

    note_obj = _ensure_object(
        raw_obj.get("task7f1_remediation_v2", {}), context="task7f1_remediation_v2"
    )
    next_model_cycle = note_obj.get("next_model_cycle")
    has_t_norm_only_note = False
    if isinstance(next_model_cycle, dict):
        has_t_norm_only_note = (
            next_model_cycle.get("use_proprio") is False
            and next_model_cycle.get("side_channel_interpretation")
            == "t_norm_centric_unless_T7F2_explicitly_reenables_parity"
        )
    if spec.require_t_norm_only_note and not has_t_norm_only_note:
        blockers.append(
            f"remediation_note_missing:{spec.role}: next model cycle must record use_proprio=false and t_norm-centric side-channel intent"
        )

    return_to_bins: dict[float, set[int]] = {}
    for sample in manifest.samples:
        return_to_bins.setdefault(float(sample.return_g), set()).add(
            int(sample.target_bin_index)
        )
    monotonic = _monotonic_summary(return_to_bins)
    if not bool(monotonic["monotonic_non_decreasing"]):
        blockers.append(
            f"sign_inconsistent:{spec.role}: higher return_G must map to higher target_bin_index"
        )
    if not bool(monotonic["duplicate_return_targets_consistent"]):
        blockers.append(
            f"sign_inconsistent:{spec.role}: identical return_G values map to multiple target_bin_index values"
        )

    summary: JsonObject = {
        "role": spec.role,
        "manifest_path": str(manifest_path),
        "source_build_json": str(manifest.source_build_json),
        "dataset_path": str(manifest.dataset_path),
        "split_name": manifest.split_name,
        "sample_count": len(manifest.samples),
        "expected_input_mode": dict(expected),
        "actual_input_mode": actual_input_mode,
        "build_allow_future_frames": bool(manifest.allow_future_frames),
        "alignment_report_path": str(alignment_path),
        "alignment": {
            "pass": bool(alignment.get("pass", False)),
            "allow_future_frames": bool(alignment.get("allow_future_frames", True)),
            "future_frame_violation_count": int(
                alignment.get("future_frame_violation_count", -1)
            ),
            "frame_t_mismatch_count": int(alignment.get("frame_t_mismatch_count", -1)),
            "episode_leakage_count": int(alignment.get("episode_leakage_count", -1)),
        },
        "t7f1_remediation_v2": note_obj,
        "monotonic_sign_audit": monotonic,
        "pass": not blockers,
        "blockers": list(blockers),
    }
    return summary, blockers


def _prepare_v2_manifests(args: argparse.Namespace) -> list[JsonObject]:
    mapping = [
        (
            _resolve_path(
                args.train_source_manifest, default_rel=DEFAULT_SOURCE_TRAIN_MANIFEST
            ),
            _resolve_path(args.train_manifest, default_rel=DEFAULT_TRAIN_MANIFEST),
            "train_full",
        ),
        (
            _resolve_path(
                args.val_source_manifest, default_rel=DEFAULT_SOURCE_VAL_MANIFEST
            ),
            _resolve_path(args.val_manifest, default_rel=DEFAULT_VAL_MANIFEST),
            "val_full",
        ),
        (
            _resolve_path(
                args.test_full_source_manifest,
                default_rel=DEFAULT_SOURCE_TEST_FULL_MANIFEST,
            ),
            _resolve_path(
                args.test_full_manifest, default_rel=DEFAULT_TEST_FULL_MANIFEST
            ),
            "test_full",
        ),
        (
            _resolve_path(
                args.test_prompt_source_manifest,
                default_rel=DEFAULT_SOURCE_TEST_PROMPT_MANIFEST,
            ),
            _resolve_path(
                args.test_prompt_manifest, default_rel=DEFAULT_TEST_PROMPT_MANIFEST
            ),
            "test_prompt_only",
        ),
        (
            _resolve_path(
                args.test_vision_source_manifest,
                default_rel=DEFAULT_SOURCE_TEST_VISION_MANIFEST,
            ),
            _resolve_path(
                args.test_vision_manifest, default_rel=DEFAULT_TEST_VISION_MANIFEST
            ),
            "test_vision_only",
        ),
    ]
    return [
        _rewrite_manifest_for_v2(source_path=source, target_path=target, role=role)
        for source, target, role in mapping
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="43c_vlm_critic_sign_audit.py",
        description=(
            "Prepare Task 7F1 clean v2 manifests and fail-closed on dirty controls or sign inconsistency."
        ),
    )
    _ = parser.add_argument("--prepare-v2", action="store_true")
    _ = parser.add_argument(
        "--train-source-manifest", type=str, default=DEFAULT_SOURCE_TRAIN_MANIFEST
    )
    _ = parser.add_argument(
        "--val-source-manifest", type=str, default=DEFAULT_SOURCE_VAL_MANIFEST
    )
    _ = parser.add_argument(
        "--test-full-source-manifest",
        type=str,
        default=DEFAULT_SOURCE_TEST_FULL_MANIFEST,
    )
    _ = parser.add_argument(
        "--test-prompt-source-manifest",
        type=str,
        default=DEFAULT_SOURCE_TEST_PROMPT_MANIFEST,
    )
    _ = parser.add_argument(
        "--test-vision-source-manifest",
        type=str,
        default=DEFAULT_SOURCE_TEST_VISION_MANIFEST,
    )
    _ = parser.add_argument(
        "--train-manifest", type=str, default=DEFAULT_TRAIN_MANIFEST
    )
    _ = parser.add_argument("--val-manifest", type=str, default=DEFAULT_VAL_MANIFEST)
    _ = parser.add_argument(
        "--test-full-manifest", type=str, default=DEFAULT_TEST_FULL_MANIFEST
    )
    _ = parser.add_argument(
        "--test-prompt-manifest", type=str, default=DEFAULT_TEST_PROMPT_MANIFEST
    )
    _ = parser.add_argument(
        "--test-vision-manifest", type=str, default=DEFAULT_TEST_VISION_MANIFEST
    )
    _ = parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    output_json = _resolve_path(args.output_json, default_rel=DEFAULT_OUTPUT_JSON)
    prepared_outputs: list[JsonObject] = []
    try:
        if bool(args.prepare_v2):
            prepared_outputs = _prepare_v2_manifests(args)

        blockers: list[str] = []
        manifest_results: list[JsonObject] = []
        for spec in _manifest_specs(args):
            result, manifest_blockers = _audit_single_manifest(spec)
            manifest_results.append(result)
            blockers.extend(manifest_blockers)

        global_monotonic_ok = all(
            bool(item["monotonic_sign_audit"]["monotonic_non_decreasing"])
            and bool(
                item["monotonic_sign_audit"]["duplicate_return_targets_consistent"]
            )
            for item in manifest_results
        )
        payload: JsonObject = {
            "schema_version": "task7_sign_audit_v2",
            "task_id": "T7F1",
            "prepare_v2_requested": bool(args.prepare_v2),
            "prepared_outputs": prepared_outputs,
            "pass": not blockers,
            "blockers": blockers,
            "global_sign_summary": {
                "higher_return_maps_to_higher_target_bin_index": global_monotonic_ok,
                "checked_manifest_count": len(manifest_results),
            },
            "manifests": manifest_results,
        }
        if blockers:
            _emit_result(
                sentinel=FAIL_SENTINEL, payload=payload, output_json=output_json
            )
            for blocker in blockers:
                print(f"[ERROR] {blocker}", file=sys.stderr)
            return 1
        _emit_result(sentinel=PASS_SENTINEL, payload=payload, output_json=output_json)
        return 0
    except Exception as exc:
        payload = {
            "schema_version": "task7_sign_audit_v2",
            "task_id": "T7F1",
            "prepare_v2_requested": bool(args.prepare_v2),
            "prepared_outputs": prepared_outputs,
            "pass": False,
            "blockers": [f"{type(exc).__name__}: {exc}"],
        }
        _emit_result(sentinel=FAIL_SENTINEL, payload=payload, output_json=output_json)
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
