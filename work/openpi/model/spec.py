from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from work.openpi.checkpoint import require_mapping, resolve_checkpoint_instance_binding
from work.openpi.contracts import RuntimeServerSpec
from work.openpi.dataloader import json_ready


@dataclass(frozen=True)
class EvalModelBinding:
    variant: str
    checkpoint_ref: str
    serve_checkpoint_ref: str
    serve_checkpoint_mode: str


def stable_json_dumps(payload: Mapping[str, object]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_effective_runtime_spec(
    *,
    variant: str,
    checkpoint_ref: str,
    runtime_indicator_config: object,
    prompt_surface_bundle: object,
    key_files: tuple[Path, ...],
    binding_schema_version: str,
    runtime_spec_schema_version: str,
) -> dict[str, str]:
    config = runtime_indicator_config
    prompt_bundle = prompt_surface_bundle
    prompt_provenance = getattr(prompt_bundle, "prompt_provenance")
    return {
        "schema_version": runtime_spec_schema_version,
        "variant": str(variant),
        "checkpoint_ref": str(checkpoint_ref),
        "checkpoint_instance_binding": resolve_checkpoint_instance_binding(
            str(checkpoint_ref),
            key_files=key_files,
            schema_version=binding_schema_version,
        ),
        "indicator_mode": str(getattr(prompt_bundle, "indicator_mode")),
        "prompt_text_surface": str(getattr(prompt_bundle, "prompt_text_surface")),
        "prompt_route": str(prompt_provenance["prompt_route"]),
        "conditioning_mode": str(prompt_provenance["conditioning_mode"]),
        "source_prompt_field": str(prompt_provenance["source_prompt_field"]),
        "consumer_mode": str(getattr(prompt_bundle, "consumer_mode")),
        "fixed_indicator_mode": str(
            getattr(prompt_bundle, "fixed_indicator_mode") or ""
        ),
        "critic_checkpoint_ref": str(getattr(prompt_bundle, "critic_checkpoint_ref")),
        "requested_indicator_mode": str(getattr(config, "requested_indicator_mode")),
    }


def build_expected_runtime_prompting_payload(
    *,
    runtime_indicator_config: object,
    prompt_surface_bundle: object,
) -> dict[str, str]:
    config = runtime_indicator_config
    prompt_bundle = prompt_surface_bundle
    runtime_prompting = validate_rollout_input_summary_runtime_fields(
        {
            "indicator_mode_requested": str(
                getattr(config, "requested_indicator_mode")
            ),
            "indicator_mode": str(getattr(prompt_bundle, "indicator_mode")),
            "indicator_source": str(getattr(prompt_bundle, "indicator_source")),
            "prompt_text_surface": str(
                getattr(prompt_bundle, "prompt_text_surface")
            ),
            "critic_checkpoint_ref": str(
                getattr(prompt_bundle, "critic_checkpoint_ref")
            ),
        }
    )
    prompt_provenance = getattr(prompt_bundle, "prompt_provenance")
    return {
        **runtime_prompting,
        "prompt_route": str(prompt_provenance["prompt_route"]),
        "conditioning_mode": str(prompt_provenance["conditioning_mode"]),
        "source_prompt_field": str(prompt_provenance["source_prompt_field"]),
        "consumer_mode": str(getattr(prompt_bundle, "consumer_mode")),
        "fixed_indicator_mode": str(
            getattr(prompt_bundle, "fixed_indicator_mode") or ""
        ),
    }


def effective_runtime_spec_without_request(
    payload: Mapping[str, object],
    *,
    runtime_spec_schema_version: str,
) -> dict[str, str]:
    checkpoint_ref = _require_non_empty_str(
        payload.get("checkpoint_ref"),
        context="effective_runtime_spec.checkpoint_ref",
    )
    return {
        "schema_version": _require_non_empty_str(
            payload.get("schema_version", runtime_spec_schema_version),
            context="effective_runtime_spec.schema_version",
        ),
        "variant": _require_non_empty_str(
            payload.get("variant"),
            context="effective_runtime_spec.variant",
        ),
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_instance_binding": str(
            payload.get("checkpoint_instance_binding") or checkpoint_ref
        ).strip()
        or checkpoint_ref,
        "indicator_mode": _require_non_empty_str(
            payload.get("indicator_mode"),
            context="effective_runtime_spec.indicator_mode",
        ),
        "prompt_text_surface": _require_non_empty_str(
            payload.get("prompt_text_surface"),
            context="effective_runtime_spec.prompt_text_surface",
        ),
        "prompt_route": _require_non_empty_str(
            payload.get("prompt_route"),
            context="effective_runtime_spec.prompt_route",
        ),
        "conditioning_mode": _require_non_empty_str(
            payload.get("conditioning_mode"),
            context="effective_runtime_spec.conditioning_mode",
        ),
        "source_prompt_field": _require_non_empty_str(
            payload.get("source_prompt_field"),
            context="effective_runtime_spec.source_prompt_field",
        ),
        "consumer_mode": _require_non_empty_str(
            payload.get("consumer_mode"),
            context="effective_runtime_spec.consumer_mode",
        ),
        "fixed_indicator_mode": str(payload.get("fixed_indicator_mode", "")).strip(),
        "critic_checkpoint_ref": _require_non_empty_str(
            payload.get("critic_checkpoint_ref"),
            context="effective_runtime_spec.critic_checkpoint_ref",
        ),
    }


def effective_runtime_surface_signature(
    payload: Mapping[str, object],
    *,
    runtime_spec_schema_version: str,
) -> dict[str, str]:
    normalized = effective_runtime_spec_without_request(
        payload,
        runtime_spec_schema_version=runtime_spec_schema_version,
    )
    return {
        field_name: normalized[field_name]
        for field_name in (
            "indicator_mode",
            "prompt_text_surface",
            "prompt_route",
            "conditioning_mode",
            "source_prompt_field",
            "consumer_mode",
            "fixed_indicator_mode",
            "critic_checkpoint_ref",
        )
    }


def normalize_effective_runtime_spec(
    payload: Mapping[str, object],
    *,
    context: str,
    runtime_spec_schema_version: str,
) -> dict[str, str]:
    try:
        return effective_runtime_spec_without_request(
            payload,
            runtime_spec_schema_version=runtime_spec_schema_version,
        )
    except ValueError as exc:
        raise ValueError(f"{context} invalid: {exc}") from exc


def effective_runtime_spec_hash(payload: Mapping[str, object]) -> str:
    normalized = {
        key: value
        for key, value in payload.items()
        if key != "requested_indicator_mode"
    }
    return sha256_text(stable_json_dumps(normalized))


def normalize_runtime_prompting_payload(
    payload: Mapping[str, object],
    *,
    context: str,
) -> dict[str, str]:
    runtime_prompting = validate_rollout_input_summary_runtime_fields(
        {
            "indicator_mode_requested": payload.get("indicator_mode_requested"),
            "indicator_mode": payload.get("indicator_mode"),
            "indicator_source": payload.get("indicator_source"),
            "prompt_text_surface": payload.get("prompt_text_surface"),
            "critic_checkpoint_ref": payload.get("critic_checkpoint_ref"),
        }
    )
    normalized = {
        **runtime_prompting,
        "prompt_route": _require_non_empty_str(
            payload.get("prompt_route"),
            context=f"{context}.prompt_route",
        ),
        "conditioning_mode": _require_non_empty_str(
            payload.get("conditioning_mode"),
            context=f"{context}.conditioning_mode",
        ),
        "source_prompt_field": _require_non_empty_str(
            payload.get("source_prompt_field"),
            context=f"{context}.source_prompt_field",
        ),
        "consumer_mode": _require_non_empty_str(
            payload.get("consumer_mode"),
            context=f"{context}.consumer_mode",
        ),
        "fixed_indicator_mode": str(payload.get("fixed_indicator_mode", "")).strip(),
    }
    prompt_text = str(payload.get("prompt_text", "")).strip()
    if prompt_text:
        normalized["prompt_text"] = prompt_text
    return normalized


def resolve_runtime_prompting_payload(
    *,
    runtime_indicator_config: object,
    prompt_surface_bundle: object,
    observed_runtime_prompting: Mapping[str, object] | None,
    context: str,
) -> dict[str, str]:
    expected = build_expected_runtime_prompting_payload(
        runtime_indicator_config=runtime_indicator_config,
        prompt_surface_bundle=prompt_surface_bundle,
    )
    if observed_runtime_prompting is None:
        return expected
    normalized = normalize_runtime_prompting_payload(
        observed_runtime_prompting,
        context=context,
    )
    if "prompt_text" in normalized:
        return normalized
    return normalized


def observed_runtime_prompting_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    context: str,
) -> dict[str, str]:
    observed: dict[str, str] | None = None
    for index, raw_row in enumerate(rows):
        row = require_mapping(raw_row, context=f"{context}[{index}]")
        candidate = normalize_runtime_prompting_payload(
            row,
            context=f"{context}[{index}]",
        )
        comparable = {
            key: value for key, value in candidate.items() if key != "prompt_text"
        }
        if observed is None:
            observed = comparable
            continue
        if comparable != observed:
            raise ValueError(
                "episode-level runtime_prompting drifted within a single rollout source"
            )
    if observed is None:
        raise ValueError(f"{context} must not be empty")
    return observed


def effective_runtime_spec_from_runtime_prompting(
    runtime_prompting: Mapping[str, object],
    *,
    variant: str,
    checkpoint_ref: str,
    key_files: tuple[Path, ...],
    binding_schema_version: str,
    runtime_spec_schema_version: str,
    context: str,
) -> dict[str, str]:
    normalized = normalize_runtime_prompting_payload(
        runtime_prompting,
        context=context,
    )
    return {
        "schema_version": runtime_spec_schema_version,
        "variant": str(variant),
        "checkpoint_ref": str(checkpoint_ref),
        "checkpoint_instance_binding": resolve_checkpoint_instance_binding(
            str(checkpoint_ref),
            key_files=key_files,
            schema_version=binding_schema_version,
        ),
        "indicator_mode": normalized["indicator_mode"],
        "prompt_text_surface": normalized["prompt_text_surface"],
        "prompt_route": normalized["prompt_route"],
        "conditioning_mode": normalized["conditioning_mode"],
        "source_prompt_field": normalized["source_prompt_field"],
        "consumer_mode": normalized["consumer_mode"],
        "fixed_indicator_mode": normalized["fixed_indicator_mode"],
        "critic_checkpoint_ref": normalized["critic_checkpoint_ref"],
        "requested_indicator_mode": normalized["indicator_mode_requested"],
    }


def build_rollout_input_summary_v21(
    *,
    schema_version: str,
    variant: str,
    checkpoint_ref: str,
    serve_checkpoint_ref: str,
    serve_checkpoint_mode: str,
    task_suite_name: str,
    task_seed_manifests: Sequence[tuple[int, tuple[int, ...]]],
    manifest: Mapping[str, object],
    num_trials_per_task: int,
    server_spec: RuntimeServerSpec,
    server_log: Path,
    harness_log: Path,
    episode_count: int,
    runtime_indicator_config: object,
    prompt_surface_bundle: object,
    observed_runtime_prompting: Mapping[str, object] | None = None,
    key_files: tuple[Path, ...],
    binding_schema_version: str,
    runtime_spec_schema_version: str,
) -> dict[str, object]:
    runtime_prompting = resolve_runtime_prompting_payload(
        runtime_indicator_config=runtime_indicator_config,
        prompt_surface_bundle=prompt_surface_bundle,
        observed_runtime_prompting=observed_runtime_prompting,
        context="rollout_input_summary.runtime_prompting",
    )
    effective_runtime_spec = build_effective_runtime_spec(
        variant=variant,
        checkpoint_ref=checkpoint_ref,
        runtime_indicator_config=runtime_indicator_config,
        prompt_surface_bundle=prompt_surface_bundle,
        key_files=key_files,
        binding_schema_version=binding_schema_version,
        runtime_spec_schema_version=runtime_spec_schema_version,
    )
    summary: dict[str, object] = {
        "schema_version": schema_version,
        "variant": variant,
        "checkpoint_ref": checkpoint_ref,
        "serve_checkpoint_ref": serve_checkpoint_ref,
        "serve_checkpoint_mode": serve_checkpoint_mode,
        "task_suite_name": task_suite_name,
        **(
            {
                "per_task_seed_manifest": {
                    str(task_id): list(task_seed_manifest)
                    for task_id, task_seed_manifest in task_seed_manifests
                }
            }
            if manifest.get("per_task_seed_manifest") is not None
            else {"seed_manifest": list(task_seed_manifests[0][1])}
        ),
        "task_ids": [task_id for task_id, _ in task_seed_manifests],
        "num_trials_per_task": num_trials_per_task,
        "server_log": str(server_log),
        "harness_log": str(harness_log),
        "host": server_spec.host,
        "port": server_spec.port,
        "episode_count": episode_count,
        "runtime_prompting": runtime_prompting,
        "effective_runtime_spec": effective_runtime_spec,
        "effective_runtime_spec_hash": effective_runtime_spec_hash(
            effective_runtime_spec
        ),
    }
    return json_ready(summary)


def _require_non_empty_str(raw: object, *, context: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def validate_rollout_input_summary_runtime_fields(
    payload: Mapping[str, object],
) -> dict[str, str]:
    from work.openpi.eval.protocols.manifest import (
        validate_rollout_input_summary_runtime_fields as validate_fields,
    )

    return validate_fields(payload)
