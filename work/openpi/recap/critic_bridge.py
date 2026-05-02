from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast


BRIDGE_SCHEMA_VERSION: Final[str] = "openpi_recap_critic_bridge_v1"
AUDIT_CONCLUSION_ADAPTER_REQUIRED: Final[str] = "adapter_required"
ALLOWED_AUDIT_CONCLUSIONS: Final[tuple[str, ...]] = (
    "existing_reusable",
    AUDIT_CONCLUSION_ADAPTER_REQUIRED,
    "fail_rewrite_needed",
)
AUDITED_BACKEND_NAME: Final[str] = "qwen3_vl_late_fusion_v1"
AUDITED_ARTIFACT_VERSION: Final[str] = "multimodal_distributional_v1"
AUDITED_VALUE_SCALE: Final[str] = "raw_return"
AUDITED_PROMPT_TEXT_FIELD: Final[str] = "prompt_raw"
AUDITED_FRAME_POLICY: Final[str] = "current_step_index"
DEFAULT_SIDE_CHANNELS: Final[tuple[str, ...]] = ("proprio", "t_norm")
DEFAULT_REQUIRED_FILES: Final[dict[str, str]] = {
    "config": "config.json",
    "bin_centers": "bin_centers.json",
    "metrics": "metrics.json",
    "provenance": "provenance.json",
    "split_manifest_ref": "split_manifest_ref.json",
    "model": "model.pt",
    "processor_config": "processor/processor_config.json",
}


class CriticBridgeContractError(ValueError):
    pass


def _require_mapping(raw: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise CriticBridgeContractError(
            f"{context} must be a mapping, got {type(raw).__name__}"
        )
    return cast(Mapping[str, object], raw)


def _require_sequence(raw: object, *, context: str) -> Sequence[object]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise CriticBridgeContractError(
            f"{context} must be a sequence, got {type(raw).__name__}"
        )
    return raw


def _require_non_empty_str(raw: object, *, context: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise CriticBridgeContractError(
            f"{context} must be a non-empty string, got {raw!r}"
        )
    return raw.strip()


def _require_int_like(raw: object, *, context: str) -> int:
    if isinstance(raw, bool) or raw is None:
        raise CriticBridgeContractError(f"{context} must be int-like, got {raw!r}")
    if not isinstance(raw, (int, float, str)):
        raise CriticBridgeContractError(
            f"{context} must be int-like, got {type(raw).__name__}"
        )
    value = int(raw)
    if value < 0:
        raise CriticBridgeContractError(f"{context} must be >= 0, got {value!r}")
    return value


def _require_float_like(raw: object, *, context: str) -> float:
    if isinstance(raw, bool) or raw is None:
        raise CriticBridgeContractError(f"{context} must be float-like, got {raw!r}")
    if not isinstance(raw, (int, float, str)):
        raise CriticBridgeContractError(
            f"{context} must be float-like, got {type(raw).__name__}"
        )
    value = float(raw)
    if not math.isfinite(value):
        raise CriticBridgeContractError(f"{context} must be finite, got {value!r}")
    return value


def _require_float_list(raw: object, *, context: str) -> list[float]:
    sequence = _require_sequence(raw, context=context)
    values = [
        _require_float_like(item, context=f"{context}[{idx}]")
        for idx, item in enumerate(sequence)
    ]
    if not values:
        raise CriticBridgeContractError(f"{context} must be non-empty")
    return values


def _default_required_files(critic_dir: Path) -> dict[str, str]:
    return {
        key: str((critic_dir / relative_path).resolve())
        for key, relative_path in DEFAULT_REQUIRED_FILES.items()
    }


def build_bridge_contract() -> dict[str, object]:
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "audit_conclusion": AUDIT_CONCLUSION_ADAPTER_REQUIRED,
        "adapter_scope": "minimal_contract_only",
        "repo_presence_vs_active_path": {
            "critic_repo_surface_present": True,
            "openpi_train_path_actively_consumes_critic": False,
            "openpi_rollout_path_actively_consumes_critic": False,
            "interpretation": (
                "critic_vlm has a reusable artifact and local inference contract, but OpenPI train/rollout "
                "paths do not call it directly today; an adapter is still required."
            ),
        },
        "input_contract": {
            "observation": {
                "required_keys": ["image"],
                "optional_keys": ["state", "wrist_image"],
                "semantics": {
                    "image": "current-step RGB frame aligned with processor.frame_policy=current_step_index",
                    "state": "OpenPI state passthrough; audited local critic runtime does not actively consume it",
                    "wrist_image": "optional secondary camera passthrough; audited local critic runtime does not actively consume it",
                },
            },
            "language": {
                "required_keys": ["prompt_raw"],
                "optional_keys": ["prompt_text"],
                "semantics": {
                    "prompt_raw": "raw task/language text; audited processor contract requires task_text_field=prompt_raw",
                    "prompt_text": "optional prebuilt text for adapter-local caching only; audited runtime still grounds semantics in prompt_raw",
                },
            },
            "task_metadata": {
                "required_keys": ["step_index", "episode_length"],
                "optional_keys": [
                    "task_name",
                    "episode_id",
                    "dataset_name",
                    "timestep_norm",
                ],
                "semantics": {
                    "step_index": "current step index used by the audited runtime to derive t_norm",
                    "episode_length": "episode horizon used with step_index to derive t_norm",
                    "timestep_norm": "optional cached value; if provided it must agree with step_index / max(episode_length - 1, 1)",
                    "task_name": "adapter-readable task label only; not consumed by audited critic runtime",
                    "episode_id": "adapter provenance identifier only; not consumed by audited critic runtime",
                    "dataset_name": "adapter provenance identifier only; not consumed by audited critic runtime",
                },
            },
        },
        "output_contract": {
            "value_distribution": {
                "required_keys": ["bin_centers", "bin_logits", "bin_probs"],
                "semantics": {
                    "bin_centers": "distributional support loaded from bin_centers.json",
                    "bin_logits": "raw categorical logits emitted by the critic value head",
                    "bin_probs": "normalized categorical probabilities aligned one-to-one with bin_centers",
                },
            },
            "decoded_value": {
                "required_keys": ["value_V_raw", "value_scale"],
                "semantics": {
                    "value_V_raw": "decoded scalar value expectation over the audited bin distribution",
                    "value_scale": "artifact-declared scalar meaning; current audited value_scale is raw_return",
                },
            },
            "provenance_handle": {
                "required_keys": [
                    "critic_dir",
                    "artifact_version",
                    "backend_name",
                    "value_scale",
                    "prompt_text_field",
                    "frame_policy",
                    "allow_future_frames",
                    "side_channels",
                    "required_files",
                    "provenance_path",
                ],
                "semantics": {
                    "critic_dir": "root directory of the critic artifact bundle",
                    "required_files": "absolute file references for the audited artifact bundle pieces",
                    "provenance_path": "artifact provenance.json path carried into downstream audit/reporting",
                },
            },
        },
    }


def build_provenance_handle(
    *,
    critic_dir: str | Path,
    artifact_version: str = AUDITED_ARTIFACT_VERSION,
    backend_name: str = AUDITED_BACKEND_NAME,
    value_scale: str = AUDITED_VALUE_SCALE,
    prompt_text_field: str = AUDITED_PROMPT_TEXT_FIELD,
    frame_policy: str = AUDITED_FRAME_POLICY,
    allow_future_frames: bool = False,
    side_channels: Sequence[str] = DEFAULT_SIDE_CHANNELS,
    required_files: Mapping[str, object] | None = None,
    provenance_path: str | Path | None = None,
) -> dict[str, object]:
    critic_dir_path = Path(critic_dir).resolve()
    resolved_required_files = (
        {
            str(key): _require_non_empty_str(
                value, context=f"required_files[{str(key)!r}]"
            )
            for key, value in required_files.items()
        }
        if required_files is not None
        else _default_required_files(critic_dir_path)
    )
    provenance_value = (
        str(Path(provenance_path).resolve())
        if provenance_path is not None
        else str((critic_dir_path / "provenance.json").resolve())
    )
    return {
        "critic_dir": str(critic_dir_path),
        "artifact_version": _require_non_empty_str(
            artifact_version, context="artifact_version"
        ),
        "backend_name": _require_non_empty_str(backend_name, context="backend_name"),
        "value_scale": _require_non_empty_str(value_scale, context="value_scale"),
        "prompt_text_field": _require_non_empty_str(
            prompt_text_field, context="prompt_text_field"
        ),
        "frame_policy": _require_non_empty_str(frame_policy, context="frame_policy"),
        "allow_future_frames": bool(allow_future_frames),
        "side_channels": [
            _require_non_empty_str(channel, context=f"side_channels[{idx}]")
            for idx, channel in enumerate(side_channels)
        ],
        "required_files": resolved_required_files,
        "provenance_path": provenance_value,
    }


def validate_bridge_request(payload: Mapping[str, object]) -> dict[str, object]:
    request = _require_mapping(payload, context="bridge_request")
    observation = _require_mapping(
        request.get("observation"), context="bridge_request.observation"
    )
    language = _require_mapping(
        request.get("language"), context="bridge_request.language"
    )
    task_metadata = _require_mapping(
        request.get("task_metadata"), context="bridge_request.task_metadata"
    )
    if "image" not in observation:
        raise CriticBridgeContractError("bridge_request.observation.image is required")
    _ = observation["image"]
    if "state" in observation:
        _ = _require_sequence(
            observation["state"], context="bridge_request.observation.state"
        )
    _ = _require_non_empty_str(
        language.get("prompt_raw"), context="bridge_request.language.prompt_raw"
    )
    step_index = _require_int_like(
        task_metadata.get("step_index"),
        context="bridge_request.task_metadata.step_index",
    )
    episode_length = _require_int_like(
        task_metadata.get("episode_length"),
        context="bridge_request.task_metadata.episode_length",
    )
    if episode_length <= 0:
        raise CriticBridgeContractError(
            "bridge_request.task_metadata.episode_length must be > 0"
        )
    if step_index >= episode_length:
        raise CriticBridgeContractError(
            "bridge_request.task_metadata.step_index must be < episode_length"
        )
    if "timestep_norm" in task_metadata:
        expected = float(step_index) / float(max(episode_length - 1, 1))
        observed = _require_float_like(
            task_metadata.get("timestep_norm"),
            context="bridge_request.task_metadata.timestep_norm",
        )
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise CriticBridgeContractError(
                "bridge_request.task_metadata.timestep_norm must agree with step_index/episode_length"
            )
    return {
        "observation": dict(observation),
        "language": dict(language),
        "task_metadata": dict(task_metadata),
    }


def validate_bridge_response(payload: Mapping[str, object]) -> dict[str, object]:
    response = _require_mapping(payload, context="bridge_response")
    value_distribution = _require_mapping(
        response.get("value_distribution"), context="bridge_response.value_distribution"
    )
    decoded_value = _require_mapping(
        response.get("decoded_value"), context="bridge_response.decoded_value"
    )
    provenance_handle = _require_mapping(
        response.get("provenance_handle"),
        context="bridge_response.provenance_handle",
    )
    bin_centers = _require_float_list(
        value_distribution.get("bin_centers"),
        context="bridge_response.value_distribution.bin_centers",
    )
    bin_logits = _require_float_list(
        value_distribution.get("bin_logits"),
        context="bridge_response.value_distribution.bin_logits",
    )
    bin_probs = _require_float_list(
        value_distribution.get("bin_probs"),
        context="bridge_response.value_distribution.bin_probs",
    )
    if not (len(bin_centers) == len(bin_logits) == len(bin_probs)):
        raise CriticBridgeContractError(
            "bridge_response.value_distribution bin_centers/bin_logits/bin_probs must have identical lengths"
        )
    prob_sum = float(sum(bin_probs))
    if not math.isclose(prob_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise CriticBridgeContractError(
            f"bridge_response.value_distribution.bin_probs must sum to 1.0, got {prob_sum!r}"
        )
    for idx, value in enumerate(bin_probs):
        if value < 0.0 or value > 1.0:
            raise CriticBridgeContractError(
                f"bridge_response.value_distribution.bin_probs[{idx}] must be within [0,1], got {value!r}"
            )
    _ = _require_float_like(
        decoded_value.get("value_V_raw"),
        context="bridge_response.decoded_value.value_V_raw",
    )
    _ = _require_non_empty_str(
        decoded_value.get("value_scale"),
        context="bridge_response.decoded_value.value_scale",
    )
    validated_handle = validate_provenance_handle(provenance_handle)
    return {
        "value_distribution": {
            "bin_centers": bin_centers,
            "bin_logits": bin_logits,
            "bin_probs": bin_probs,
        },
        "decoded_value": dict(decoded_value),
        "provenance_handle": validated_handle,
    }


def validate_provenance_handle(payload: Mapping[str, object]) -> dict[str, object]:
    handle = _require_mapping(payload, context="provenance_handle")
    required_files = _require_mapping(
        handle.get("required_files"), context="provenance_handle.required_files"
    )
    normalized_required_files = {
        str(key): _require_non_empty_str(
            value, context=f"provenance_handle.required_files[{str(key)!r}]"
        )
        for key, value in required_files.items()
    }
    missing_required_files = [
        key for key in DEFAULT_REQUIRED_FILES if key not in normalized_required_files
    ]
    if missing_required_files:
        raise CriticBridgeContractError(
            "provenance_handle.required_files is missing keys: "
            + ", ".join(missing_required_files)
        )
    prompt_text_field = _require_non_empty_str(
        handle.get("prompt_text_field"), context="provenance_handle.prompt_text_field"
    )
    frame_policy = _require_non_empty_str(
        handle.get("frame_policy"), context="provenance_handle.frame_policy"
    )
    allow_future_frames = bool(handle.get("allow_future_frames", False))
    if prompt_text_field != AUDITED_PROMPT_TEXT_FIELD:
        raise CriticBridgeContractError(
            f"provenance_handle.prompt_text_field must be {AUDITED_PROMPT_TEXT_FIELD!r}, got {prompt_text_field!r}"
        )
    if frame_policy != AUDITED_FRAME_POLICY:
        raise CriticBridgeContractError(
            f"provenance_handle.frame_policy must be {AUDITED_FRAME_POLICY!r}, got {frame_policy!r}"
        )
    if allow_future_frames:
        raise CriticBridgeContractError(
            "provenance_handle.allow_future_frames must stay false"
        )
    side_channels = [
        _require_non_empty_str(
            channel, context=f"provenance_handle.side_channels[{idx}]"
        )
        for idx, channel in enumerate(
            _require_sequence(
                handle.get("side_channels", []),
                context="provenance_handle.side_channels",
            )
        )
    ]
    return {
        "critic_dir": _require_non_empty_str(
            handle.get("critic_dir"), context="provenance_handle.critic_dir"
        ),
        "artifact_version": _require_non_empty_str(
            handle.get("artifact_version"),
            context="provenance_handle.artifact_version",
        ),
        "backend_name": _require_non_empty_str(
            handle.get("backend_name"), context="provenance_handle.backend_name"
        ),
        "value_scale": _require_non_empty_str(
            handle.get("value_scale"), context="provenance_handle.value_scale"
        ),
        "prompt_text_field": prompt_text_field,
        "frame_policy": frame_policy,
        "allow_future_frames": allow_future_frames,
        "side_channels": side_channels,
        "required_files": normalized_required_files,
        "provenance_path": _require_non_empty_str(
            handle.get("provenance_path"), context="provenance_handle.provenance_path"
        ),
    }


__all__ = [
    "ALLOWED_AUDIT_CONCLUSIONS",
    "AUDITED_ARTIFACT_VERSION",
    "AUDITED_BACKEND_NAME",
    "AUDITED_FRAME_POLICY",
    "AUDITED_PROMPT_TEXT_FIELD",
    "AUDITED_VALUE_SCALE",
    "AUDIT_CONCLUSION_ADAPTER_REQUIRED",
    "BRIDGE_SCHEMA_VERSION",
    "CriticBridgeContractError",
    "build_bridge_contract",
    "build_provenance_handle",
    "validate_bridge_request",
    "validate_bridge_response",
    "validate_provenance_handle",
]
