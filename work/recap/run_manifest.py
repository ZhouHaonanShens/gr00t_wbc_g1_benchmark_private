from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


from .label_policy import LABEL_POLICY_EXTENSION_KEY
from .label_policy import normalize_label_policy_payload
from .scope_experiment import SCOPE_EXPERIMENT_EXTENSION_KEY
from .scope_experiment import normalize_scope_experiment_payload
from .train_scope_audit import TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY
from .train_scope_audit import normalize_train_scope_payload


RUN_MANIFEST_SCHEMA_VERSION = "gr00t_run_manifest_v1"
RUN_MANIFEST_ARTIFACT_KIND = "gr00t_run_manifest"

TEXT_CARRIER_SCHEMA_VERSION = "recap_text_indicator_v1"
TEXT_CARRIER_ROUTE = "carrier_text_v1"
PROMPT_SOURCE_FIELD = "prompt_raw"
INDICATOR_SOURCE_FIELD = "indicator_mode"

REQUIRED_CORE_STRING_FIELDS: tuple[str, ...] = (
    "branch",
    "commit",
    "dataset_fingerprint",
    "carrier_schema_version",
    "carrier_route",
    "prompt_source_field",
    "indicator_source",
    "checkpoint_selected",
    "checkpoint_loaded",
    "trainable_module_regex",
    "eval_overlay_regex",
    "controller_config_hash",
)
REQUIRED_CORE_INT_FIELDS: tuple[str, ...] = ("policy_horizon", "n_action_steps")

BRANCH_CANDIDATE_PATHS: tuple[str, ...] = (
    "branch",
    "comparable_run_spec.branch",
    "comparable_run_spec.stable_base.embodiment_tag",
    "stable_base.embodiment_tag",
    "controller_provenance.embodiment_tag",
    "execution_surface_contract.branch",
)
COMMIT_CANDIDATE_PATHS: tuple[str, ...] = (
    "commit",
    "git_commit",
    "repo_commit",
    "source_commit",
)
DATASET_FINGERPRINT_CANDIDATE_PATHS: tuple[str, ...] = (
    "dataset_fingerprint",
    "comparable_run_spec.dataset_fingerprint",
    "training_set_contract.dataset_fingerprint",
)
CHECKPOINT_SELECTED_CANDIDATE_PATHS: tuple[str, ...] = (
    "checkpoint_selected",
    "comparable_run_spec.checkpoint_rule.selected_checkpoint_path",
    "selected_checkpoint_path",
)
CHECKPOINT_LOADED_CANDIDATE_PATHS: tuple[str, ...] = (
    "checkpoint_loaded",
    "evaluation_binding.server_load_path",
    "server_provenance.policy_model_path",
    "provenance.policy_model_path",
    "policy_model_path",
    "server_load_path",
    "model_path",
)
TRAINABLE_MODULE_REGEX_CANDIDATE_PATHS: tuple[str, ...] = (
    "trainable_module_regex",
    "effective_config.trainable_module_regex",
)
EVAL_OVERLAY_REGEX_CANDIDATE_PATHS: tuple[str, ...] = (
    "eval_overlay_regex",
    "overlay_include_regex",
    "server_provenance.overlay_include_regex",
    "provenance.overlay_include_regex",
)
POLICY_HORIZON_CANDIDATE_PATHS: tuple[str, ...] = (
    "policy_horizon",
    "execution_surface_contract.policy_horizon_expected",
    "execution_surface_contract.policy_horizon_runtime",
    "server_action_horizon",
)
N_ACTION_STEPS_CANDIDATE_PATHS: tuple[str, ...] = (
    "n_action_steps",
    "execution_surface_contract.n_action_steps",
    "execution_surface_contract.n_action_steps_expected",
)
EVAL_USES_FINETUNED_CANDIDATE_PATHS: tuple[str, ...] = (
    "evaluation_binding.eval_uses_finetuned",
    "eval_uses_finetuned",
)
SERVER_LOAD_PATH_CANDIDATE_PATHS: tuple[str, ...] = (
    "evaluation_binding.server_load_path",
    "server_provenance.policy_model_path",
    "provenance.policy_model_path",
    "policy_model_path",
    "server_load_path",
    "model_path",
)
SERVER_LOAD_MODE_CANDIDATE_PATHS: tuple[str, ...] = (
    "evaluation_binding.server_load_mode",
    "server_load_mode",
)
BASE_MODEL_PATH_CANDIDATE_PATHS: tuple[str, ...] = (
    "evaluation_binding.base_model_path",
    "server_provenance.base_model_path",
    "provenance.base_model_path",
    "base_model_path",
    "comparable_run_spec.stable_base.base_model",
    "stable_base.base_model",
)
SCOPE_EXPERIMENT_CANDIDATE_PATHS: tuple[str, ...] = (
    f"extensions.{SCOPE_EXPERIMENT_EXTENSION_KEY}",
    SCOPE_EXPERIMENT_EXTENSION_KEY,
)
LABEL_POLICY_CANDIDATE_PATHS: tuple[str, ...] = (
    f"extensions.{LABEL_POLICY_EXTENSION_KEY}",
    LABEL_POLICY_EXTENSION_KEY,
)
TRAIN_SCOPE_TAXONOMY_CANDIDATE_PATHS: tuple[str, ...] = (
    f"extensions.{TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY}",
    TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY,
)

MISSING = object()


def _deep_get(payload: Mapping[str, Any], field_path: str) -> object:
    current: object = payload
    for key in field_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return MISSING
        current = current[key]
    return current


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_signature(payload: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _selected_checkpoint_asset(checkpoint_dir: Path | None) -> Path | None:
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return None
    candidates = [
        checkpoint_dir / "model.safetensors.index.json",
        checkpoint_dir / "model.safetensors",
        checkpoint_dir / "pytorch_model.bin.index.json",
        checkpoint_dir / "pytorch_model.bin",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _inspect_checkpoint_path(repo_root: Path, raw_path: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "raw_path": raw_path,
        "resolved_input_path": None,
        "normalized_checkpoint_path": None,
        "selected_checkpoint_asset_path": None,
        "loadable": False,
        "exists": False,
        "error": None,
    }
    if raw_path is None or not str(raw_path).strip():
        result["error"] = "missing checkpoint path"
        return result
    resolved_input = _resolve_path(repo_root, raw_path)
    result["resolved_input_path"] = str(resolved_input)
    if resolved_input.is_dir():
        checkpoint_dir = resolved_input
    elif resolved_input.is_file():
        checkpoint_dir = resolved_input.parent
    else:
        result["error"] = f"checkpoint path does not exist: {resolved_input}"
        return result
    result["exists"] = True
    result["normalized_checkpoint_path"] = str(checkpoint_dir.resolve())
    selected_asset = _selected_checkpoint_asset(checkpoint_dir)
    if selected_asset is None:
        result["error"] = (
            "checkpoint directory does not contain a retained checkpoint asset"
        )
        return result
    result["selected_checkpoint_asset_path"] = str(selected_asset.resolve())
    result["loadable"] = True
    return result


def _first_present(
    payloads: Sequence[Mapping[str, Any]],
    *,
    candidate_paths: Sequence[str],
) -> tuple[object, str | None]:
    for payload in payloads:
        for field_path in candidate_paths:
            value = _deep_get(payload, field_path)
            if value is MISSING or value is None:
                continue
            return value, str(field_path)
    return MISSING, None


def _first_string(
    payloads: Sequence[Mapping[str, Any]],
    *,
    candidate_paths: Sequence[str],
) -> tuple[str | None, str | None]:
    value, field_path = _first_present(payloads, candidate_paths=candidate_paths)
    if value is MISSING or value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, field_path
    return text, field_path


def _first_int(
    payloads: Sequence[Mapping[str, Any]],
    *,
    candidate_paths: Sequence[str],
) -> tuple[int | None, str | None]:
    value, field_path = _first_present(payloads, candidate_paths=candidate_paths)
    if value is MISSING or value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, int):
        return None, field_path
    return int(value), field_path


def _first_bool(
    payloads: Sequence[Mapping[str, Any]],
    *,
    candidate_paths: Sequence[str],
) -> tuple[bool | None, str | None]:
    value, field_path = _first_present(payloads, candidate_paths=candidate_paths)
    if value is MISSING or value is None:
        return None, None
    if not isinstance(value, bool):
        return None, field_path
    return bool(value), field_path


def _first_mapping(
    payloads: Sequence[Mapping[str, Any]],
    *,
    candidate_paths: Sequence[str],
) -> tuple[dict[str, Any] | None, str | None]:
    value, field_path = _first_present(payloads, candidate_paths=candidate_paths)
    if value is MISSING or value is None:
        return None, None
    if not isinstance(value, Mapping):
        return None, field_path
    return dict(value), field_path


def _normalize_relative_absolute_action_contract(
    value: object,
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    action_representation_by_key = value.get("action_representation_by_key")
    if not isinstance(action_representation_by_key, Mapping):
        action_representation_by_key = value.get("action_representations")
    relative_action_keys = value.get("relative_action_keys")
    absolute_action_keys = value.get("absolute_action_keys")
    if not isinstance(relative_action_keys, list) or not isinstance(
        absolute_action_keys, list
    ):
        return None
    if not isinstance(action_representation_by_key, Mapping):
        return None
    normalized: dict[str, Any] = {
        "relative_action_keys": [str(item) for item in relative_action_keys],
        "absolute_action_keys": [str(item) for item in absolute_action_keys],
        "action_representation_by_key": {
            str(key): str(item) for key, item in action_representation_by_key.items()
        },
        "must_not_conflate_horizon_and_execution": bool(
            value.get("must_not_conflate_horizon_and_execution", True)
        ),
    }
    relative_to_absolute_rule = value.get("relative_to_absolute_rule")
    if isinstance(relative_to_absolute_rule, Mapping):
        normalized["relative_to_absolute_rule"] = dict(relative_to_absolute_rule)
    return normalized


def _extract_relative_absolute_action_contract(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    candidate_paths = (
        "core.relative_absolute_action_contract",
        "relative_absolute_action_contract",
        "execution_surface_contract",
    )
    for payload in payloads:
        for field_path in candidate_paths:
            value = _deep_get(payload, field_path)
            if value is MISSING or value is None:
                continue
            normalized = _normalize_relative_absolute_action_contract(value)
            if normalized is not None:
                return normalized, str(field_path)
    return None, None


def _resolve_scope_experiment_extension(
    *,
    payloads: Sequence[Mapping[str, Any]],
    extensions: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    explicit_extension = None
    explicit_source = None
    if isinstance(extensions, Mapping):
        explicit_extension = extensions.get(SCOPE_EXPERIMENT_EXTENSION_KEY)
        if explicit_extension is not None:
            explicit_source = f"explicit_extensions.{SCOPE_EXPERIMENT_EXTENSION_KEY}"
    if explicit_extension is not None:
        return (
            normalize_scope_experiment_payload(
                explicit_extension,
                field_name=explicit_source or SCOPE_EXPERIMENT_EXTENSION_KEY,
            ),
            explicit_source,
        )

    raw_scope_experiment, scope_source = _first_mapping(
        payloads,
        candidate_paths=SCOPE_EXPERIMENT_CANDIDATE_PATHS,
    )
    if raw_scope_experiment is None:
        return None, scope_source
    return (
        normalize_scope_experiment_payload(
            raw_scope_experiment,
            field_name=scope_source or SCOPE_EXPERIMENT_EXTENSION_KEY,
        ),
        scope_source,
    )


def _resolve_label_policy_extension(
    *,
    payloads: Sequence[Mapping[str, Any]],
    extensions: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    explicit_extension = None
    explicit_source = None
    if isinstance(extensions, Mapping):
        explicit_extension = extensions.get(LABEL_POLICY_EXTENSION_KEY)
        if explicit_extension is not None:
            explicit_source = f"explicit_extensions.{LABEL_POLICY_EXTENSION_KEY}"
    if explicit_extension is not None:
        return (
            normalize_label_policy_payload(
                explicit_extension,
                field_name=explicit_source or LABEL_POLICY_EXTENSION_KEY,
            ),
            explicit_source,
        )

    raw_label_policy, label_policy_source = _first_mapping(
        payloads,
        candidate_paths=LABEL_POLICY_CANDIDATE_PATHS,
    )
    if raw_label_policy is None:
        return None, label_policy_source
    return (
        normalize_label_policy_payload(
            raw_label_policy,
            field_name=label_policy_source or LABEL_POLICY_EXTENSION_KEY,
        ),
        label_policy_source,
    )


def _resolve_train_scope_taxonomy_extension(
    *,
    payloads: Sequence[Mapping[str, Any]],
    extensions: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    explicit_extension = None
    explicit_source = None
    if isinstance(extensions, Mapping):
        explicit_extension = extensions.get(TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY)
        if explicit_extension is not None:
            explicit_source = (
                f"explicit_extensions.{TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY}"
            )
    if explicit_extension is not None:
        return (
            normalize_train_scope_payload(
                explicit_extension,
                field_name=explicit_source or TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY,
            ),
            explicit_source,
        )

    raw_extension, extension_source = _first_mapping(
        payloads,
        candidate_paths=TRAIN_SCOPE_TAXONOMY_CANDIDATE_PATHS,
    )
    if raw_extension is None:
        return None, extension_source
    return (
        normalize_train_scope_payload(
            raw_extension,
            field_name=extension_source or TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY,
        ),
        extension_source,
    )


def _enforce_scope_derived_regex_compatibility(
    *,
    scope_extension: Mapping[str, Any],
    trainable_module_regex: str | None,
    eval_overlay_regex: str | None,
) -> tuple[str, str]:
    derived_core_fields = scope_extension.get("derived_core_fields")
    if not isinstance(derived_core_fields, Mapping):
        raise TypeError("scope_experiment.derived_core_fields must be an object")
    derived_trainable = derived_core_fields.get("trainable_module_regex")
    derived_eval_overlay = derived_core_fields.get("eval_overlay_regex")
    if not isinstance(derived_trainable, str) or not derived_trainable.strip():
        raise ValueError(
            "scope_experiment.derived_core_fields.trainable_module_regex must be a non-empty string"
        )
    if not isinstance(derived_eval_overlay, str) or not derived_eval_overlay.strip():
        raise ValueError(
            "scope_experiment.derived_core_fields.eval_overlay_regex must be a non-empty string"
        )
    return derived_trainable.strip(), derived_eval_overlay.strip()


def core_digest(core_payload: Mapping[str, Any]) -> str:
    return _stable_signature(dict(core_payload))


def _controller_config_surface(
    *,
    branch: str | None,
    policy_horizon: int | None,
    n_action_steps: int | None,
    relative_contract: Mapping[str, Any] | None,
    controller_provenance: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if (
        branch is None
        or policy_horizon is None
        or n_action_steps is None
        or relative_contract is None
    ):
        return None
    surface: dict[str, Any] = {
        "branch": str(branch),
        "policy_horizon": int(policy_horizon),
        "n_action_steps": int(n_action_steps),
        "relative_absolute_action_contract": dict(relative_contract),
    }
    if isinstance(controller_provenance, Mapping):
        surface["controller_provenance"] = dict(controller_provenance)
    return surface


def controller_config_hash(
    *,
    branch: str | None,
    policy_horizon: int | None,
    n_action_steps: int | None,
    relative_contract: Mapping[str, Any] | None,
    controller_provenance: Mapping[str, Any] | None,
) -> str | None:
    surface = _controller_config_surface(
        branch=branch,
        policy_horizon=policy_horizon,
        n_action_steps=n_action_steps,
        relative_contract=relative_contract,
        controller_provenance=controller_provenance,
    )
    if surface is None:
        return None
    return _stable_signature(surface)


def build_run_manifest_from_sources(
    *,
    state_conditioned_metadata: Mapping[str, Any] | None = None,
    finetune_summary: Mapping[str, Any] | None = None,
    eval_summary: Mapping[str, Any] | None = None,
    server_provenance: Mapping[str, Any] | None = None,
    controller_audit: Mapping[str, Any] | None = None,
    branch: str | None = None,
    commit: str | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payloads: list[Mapping[str, Any]] = []
    for payload in (
        state_conditioned_metadata,
        finetune_summary,
        eval_summary,
        server_provenance,
        controller_audit,
    ):
        if isinstance(payload, Mapping):
            payloads.append(payload)

    resolved_branch, branch_source = _first_string(
        payloads,
        candidate_paths=BRANCH_CANDIDATE_PATHS,
    )
    resolved_commit, commit_source = _first_string(
        payloads,
        candidate_paths=COMMIT_CANDIDATE_PATHS,
    )
    dataset_fingerprint, dataset_fingerprint_source = _first_string(
        payloads,
        candidate_paths=DATASET_FINGERPRINT_CANDIDATE_PATHS,
    )
    checkpoint_selected, checkpoint_selected_source = _first_string(
        payloads,
        candidate_paths=CHECKPOINT_SELECTED_CANDIDATE_PATHS,
    )
    checkpoint_loaded, checkpoint_loaded_source = _first_string(
        payloads,
        candidate_paths=CHECKPOINT_LOADED_CANDIDATE_PATHS,
    )
    trainable_module_regex, trainable_module_regex_source = _first_string(
        payloads,
        candidate_paths=TRAINABLE_MODULE_REGEX_CANDIDATE_PATHS,
    )
    eval_overlay_regex, eval_overlay_regex_source = _first_string(
        payloads,
        candidate_paths=EVAL_OVERLAY_REGEX_CANDIDATE_PATHS,
    )
    scope_experiment_extension, scope_experiment_source = (
        _resolve_scope_experiment_extension(payloads=payloads, extensions=extensions)
    )
    label_policy_extension, label_policy_source = _resolve_label_policy_extension(
        payloads=payloads,
        extensions=extensions,
    )
    train_scope_taxonomy_extension, train_scope_taxonomy_source = (
        _resolve_train_scope_taxonomy_extension(payloads=payloads, extensions=extensions)
    )
    if scope_experiment_extension is not None:
        trainable_module_regex, eval_overlay_regex = (
            _enforce_scope_derived_regex_compatibility(
                scope_extension=scope_experiment_extension,
                trainable_module_regex=trainable_module_regex,
                eval_overlay_regex=eval_overlay_regex,
            )
        )
        trainable_module_regex_source = (
            f"{scope_experiment_source}.derived_core_fields.trainable_module_regex"
            if scope_experiment_source is not None
            else "scope_experiment.derived_core_fields.trainable_module_regex"
        )
        eval_overlay_regex_source = (
            f"{scope_experiment_source}.derived_core_fields.eval_overlay_regex"
            if scope_experiment_source is not None
            else "scope_experiment.derived_core_fields.eval_overlay_regex"
        )
    policy_horizon, policy_horizon_source = _first_int(
        payloads,
        candidate_paths=POLICY_HORIZON_CANDIDATE_PATHS,
    )
    n_action_steps, n_action_steps_source = _first_int(
        payloads,
        candidate_paths=N_ACTION_STEPS_CANDIDATE_PATHS,
    )
    relative_contract, relative_contract_source = (
        _extract_relative_absolute_action_contract(payloads)
    )
    eval_uses_finetuned, eval_uses_finetuned_source = _first_bool(
        payloads,
        candidate_paths=EVAL_USES_FINETUNED_CANDIDATE_PATHS,
    )
    server_load_path, server_load_path_source = _first_string(
        payloads,
        candidate_paths=SERVER_LOAD_PATH_CANDIDATE_PATHS,
    )
    server_load_mode, server_load_mode_source = _first_string(
        payloads,
        candidate_paths=SERVER_LOAD_MODE_CANDIDATE_PATHS,
    )
    base_model_path, base_model_path_source = _first_string(
        payloads,
        candidate_paths=BASE_MODEL_PATH_CANDIDATE_PATHS,
    )

    controller_provenance_payload = (
        dict(controller_audit.get("controller_provenance", {}))
        if isinstance(controller_audit, Mapping)
        and isinstance(controller_audit.get("controller_provenance"), Mapping)
        else None
    )
    resolved_controller_config_hash = controller_config_hash(
        branch=branch or resolved_branch,
        policy_horizon=policy_horizon,
        n_action_steps=n_action_steps,
        relative_contract=relative_contract,
        controller_provenance=controller_provenance_payload,
    )

    core: dict[str, Any] = {
        "branch": branch or resolved_branch,
        "commit": commit or resolved_commit,
        "dataset_fingerprint": dataset_fingerprint,
        "carrier_schema_version": TEXT_CARRIER_SCHEMA_VERSION,
        "carrier_route": TEXT_CARRIER_ROUTE,
        "prompt_source_field": PROMPT_SOURCE_FIELD,
        "indicator_source": INDICATOR_SOURCE_FIELD,
        "checkpoint_selected": checkpoint_selected,
        "checkpoint_loaded": checkpoint_loaded,
        "trainable_module_regex": trainable_module_regex,
        "eval_overlay_regex": eval_overlay_regex,
        "controller_config_hash": resolved_controller_config_hash,
        "policy_horizon": policy_horizon,
        "n_action_steps": n_action_steps,
        "relative_absolute_action_contract": relative_contract,
    }

    normalized_extensions = dict(extensions or {})
    if scope_experiment_extension is not None:
        normalized_extensions[SCOPE_EXPERIMENT_EXTENSION_KEY] = dict(
            scope_experiment_extension
        )
    if label_policy_extension is not None:
        normalized_extensions[LABEL_POLICY_EXTENSION_KEY] = dict(label_policy_extension)
    if train_scope_taxonomy_extension is not None:
        normalized_extensions[TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY] = dict(
            train_scope_taxonomy_extension
        )

    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": RUN_MANIFEST_ARTIFACT_KIND,
        "core": core,
        "evaluation_binding": {
            "eval_uses_finetuned": eval_uses_finetuned,
            "server_load_path": server_load_path,
            "server_load_mode": server_load_mode or "model_path",
            "base_model_path": base_model_path,
        },
        "extensions": normalized_extensions,
        "adapter_sources": {
            "branch": branch_source if branch is None else "explicit_arg",
            "commit": commit_source if commit is None else "explicit_arg",
            "dataset_fingerprint": dataset_fingerprint_source,
            "checkpoint_selected": checkpoint_selected_source,
            "checkpoint_loaded": checkpoint_loaded_source,
            "trainable_module_regex": trainable_module_regex_source,
            "eval_overlay_regex": eval_overlay_regex_source,
            "controller_config_hash": "derived_from_controller_surface",
            "policy_horizon": policy_horizon_source,
            "n_action_steps": n_action_steps_source,
            "relative_absolute_action_contract": relative_contract_source,
            "evaluation_binding.eval_uses_finetuned": eval_uses_finetuned_source,
            "evaluation_binding.server_load_path": server_load_path_source,
            "evaluation_binding.server_load_mode": server_load_mode_source,
            "evaluation_binding.base_model_path": base_model_path_source,
        },
    }
    if scope_experiment_source is not None:
        manifest["adapter_sources"][f"extensions.{SCOPE_EXPERIMENT_EXTENSION_KEY}"] = (
            scope_experiment_source
        )
    if label_policy_source is not None:
        manifest["adapter_sources"][f"extensions.{LABEL_POLICY_EXTENSION_KEY}"] = (
            label_policy_source
        )
    if train_scope_taxonomy_source is not None:
        manifest["adapter_sources"][
            f"extensions.{TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY}"
        ] = train_scope_taxonomy_source
    if isinstance(controller_audit, Mapping):
        controller_provenance = controller_audit.get("controller_provenance")
        if isinstance(controller_provenance, Mapping):
            manifest["controller_provenance"] = dict(controller_provenance)
    manifest["core_digest"] = core_digest(core)
    return manifest


def _issue(code: str, field_path: str, message: str) -> dict[str, str]:
    return {
        "code": str(code),
        "field_path": str(field_path),
        "message": str(message),
    }


def _validate_non_empty_string_field(
    core: Mapping[str, Any],
    *,
    field_name: str,
    issues: list[dict[str, str]],
) -> str | None:
    value = core.get(field_name, MISSING)
    if value is MISSING or value is None:
        issues.append(
            _issue(
                "missing_required_core_field",
                f"core.{field_name}",
                f"core.{field_name} is required",
            )
        )
        return None
    if not isinstance(value, str):
        issues.append(
            _issue(
                "wrong_type",
                f"core.{field_name}",
                f"core.{field_name} must be a string, got {type(value).__name__}",
            )
        )
        return None
    normalized = value.strip()
    if not normalized:
        issues.append(
            _issue(
                "empty_string",
                f"core.{field_name}",
                f"core.{field_name} must be non-empty",
            )
        )
        return None
    return normalized


def _validate_int_field(
    core: Mapping[str, Any],
    *,
    field_name: str,
    issues: list[dict[str, str]],
) -> int | None:
    value = core.get(field_name, MISSING)
    if value is MISSING or value is None:
        issues.append(
            _issue(
                "missing_required_core_field",
                f"core.{field_name}",
                f"core.{field_name} is required",
            )
        )
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        issues.append(
            _issue(
                "wrong_type",
                f"core.{field_name}",
                f"core.{field_name} must be an int, got {type(value).__name__}",
            )
        )
        return None
    if int(value) <= 0:
        issues.append(
            _issue(
                "invalid_value",
                f"core.{field_name}",
                f"core.{field_name} must be > 0",
            )
        )
        return None
    return int(value)


def _validate_string_list(
    value: object,
    *,
    field_path: str,
    issues: list[dict[str, str]],
) -> list[str] | None:
    if not isinstance(value, list):
        issues.append(
            _issue(
                "wrong_type",
                field_path,
                f"{field_path} must be a list, got {type(value).__name__}",
            )
        )
        return None
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            issues.append(
                _issue(
                    "wrong_type",
                    f"{field_path}[{index}]",
                    f"{field_path}[{index}] must be a non-empty string",
                )
            )
            return None
        normalized.append(item.strip())
    return normalized


def _validate_action_contract(
    core: Mapping[str, Any],
    *,
    issues: list[dict[str, str]],
) -> dict[str, Any] | None:
    raw_value = core.get("relative_absolute_action_contract", MISSING)
    if raw_value is MISSING or raw_value is None:
        issues.append(
            _issue(
                "missing_required_core_field",
                "core.relative_absolute_action_contract",
                "core.relative_absolute_action_contract is required",
            )
        )
        return None
    if not isinstance(raw_value, Mapping):
        issues.append(
            _issue(
                "wrong_type",
                "core.relative_absolute_action_contract",
                "core.relative_absolute_action_contract must be an object",
            )
        )
        return None

    relative_action_keys = _validate_string_list(
        raw_value.get("relative_action_keys"),
        field_path="core.relative_absolute_action_contract.relative_action_keys",
        issues=issues,
    )
    absolute_action_keys = _validate_string_list(
        raw_value.get("absolute_action_keys"),
        field_path="core.relative_absolute_action_contract.absolute_action_keys",
        issues=issues,
    )
    action_representation_by_key = raw_value.get("action_representation_by_key")
    if not isinstance(action_representation_by_key, Mapping):
        issues.append(
            _issue(
                "wrong_type",
                "core.relative_absolute_action_contract.action_representation_by_key",
                "core.relative_absolute_action_contract.action_representation_by_key must be an object",
            )
        )
        return None
    normalized_action_representation_by_key: dict[str, str] = {}
    for key, value in action_representation_by_key.items():
        if not isinstance(key, str) or not key.strip():
            issues.append(
                _issue(
                    "wrong_type",
                    "core.relative_absolute_action_contract.action_representation_by_key",
                    "action_representation_by_key keys must be non-empty strings",
                )
            )
            return None
        if not isinstance(value, str) or not value.strip():
            issues.append(
                _issue(
                    "wrong_type",
                    f"core.relative_absolute_action_contract.action_representation_by_key.{key}",
                    "action_representation_by_key values must be non-empty strings",
                )
            )
            return None
        normalized_action_representation_by_key[key.strip()] = value.strip()
    must_not_conflate = raw_value.get(
        "must_not_conflate_horizon_and_execution",
        MISSING,
    )
    if not isinstance(must_not_conflate, bool):
        issues.append(
            _issue(
                "wrong_type",
                "core.relative_absolute_action_contract.must_not_conflate_horizon_and_execution",
                "core.relative_absolute_action_contract.must_not_conflate_horizon_and_execution must be a bool",
            )
        )
        return None
    normalized: dict[str, Any] = {
        "relative_action_keys": relative_action_keys,
        "absolute_action_keys": absolute_action_keys,
        "action_representation_by_key": normalized_action_representation_by_key,
        "must_not_conflate_horizon_and_execution": bool(must_not_conflate),
    }
    relative_to_absolute_rule = raw_value.get("relative_to_absolute_rule")
    if relative_to_absolute_rule is not None:
        if not isinstance(relative_to_absolute_rule, Mapping):
            issues.append(
                _issue(
                    "wrong_type",
                    "core.relative_absolute_action_contract.relative_to_absolute_rule",
                    "core.relative_absolute_action_contract.relative_to_absolute_rule must be an object when present",
                )
            )
            return None
        normalized["relative_to_absolute_rule"] = dict(relative_to_absolute_rule)
    return normalized


def validate_run_manifest(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    normalized_manifest: dict[str, Any] = {}

    schema_version = payload.get("schema_version")
    if schema_version != RUN_MANIFEST_SCHEMA_VERSION:
        issues.append(
            _issue(
                "invalid_schema_version",
                "schema_version",
                f"schema_version must equal {RUN_MANIFEST_SCHEMA_VERSION!r}",
            )
        )
    else:
        normalized_manifest["schema_version"] = RUN_MANIFEST_SCHEMA_VERSION

    artifact_kind = payload.get("artifact_kind")
    if artifact_kind != RUN_MANIFEST_ARTIFACT_KIND:
        issues.append(
            _issue(
                "invalid_artifact_kind",
                "artifact_kind",
                f"artifact_kind must equal {RUN_MANIFEST_ARTIFACT_KIND!r}",
            )
        )
    else:
        normalized_manifest["artifact_kind"] = RUN_MANIFEST_ARTIFACT_KIND

    raw_core = payload.get("core")
    if not isinstance(raw_core, Mapping):
        issues.append(_issue("wrong_type", "core", "core must be an object"))
        raw_core = {}

    normalized_core: dict[str, Any] = {}
    for field_name in REQUIRED_CORE_STRING_FIELDS:
        value = _validate_non_empty_string_field(
            raw_core,
            field_name=field_name,
            issues=issues,
        )
        if value is not None:
            normalized_core[field_name] = value
    for field_name in REQUIRED_CORE_INT_FIELDS:
        value = _validate_int_field(raw_core, field_name=field_name, issues=issues)
        if value is not None:
            normalized_core[field_name] = value
    action_contract = _validate_action_contract(raw_core, issues=issues)
    if action_contract is not None:
        normalized_core["relative_absolute_action_contract"] = action_contract
    normalized_manifest["core"] = normalized_core

    extensions = payload.get("extensions", {})
    if not isinstance(extensions, Mapping):
        issues.append(
            _issue("wrong_type", "extensions", "extensions must be an object")
        )
        normalized_manifest["extensions"] = {}
    else:
        normalized_manifest["extensions"] = dict(extensions)

    scope_extension_payload = normalized_manifest["extensions"].get(
        SCOPE_EXPERIMENT_EXTENSION_KEY
    )
    if scope_extension_payload is not None:
        try:
            normalized_scope_extension = normalize_scope_experiment_payload(
                scope_extension_payload,
                field_name=(f"extensions.{SCOPE_EXPERIMENT_EXTENSION_KEY}"),
            )
            normalized_manifest["extensions"][SCOPE_EXPERIMENT_EXTENSION_KEY] = (
                normalized_scope_extension
            )
            derived_core_fields = normalized_scope_extension["derived_core_fields"]
            preset_id = normalized_scope_extension["preset_id"]
            for field_name in (
                "trainable_module_regex",
                "eval_overlay_regex",
            ):
                observed_value = normalized_core.get(field_name)
                if observed_value is None:
                    continue
                if observed_value != derived_core_fields[field_name]:
                    issues.append(
                        _issue(
                            "scope_preset_core_mismatch",
                            f"core.{field_name}",
                            "core."
                            + field_name
                            + " must match extensions."
                            + SCOPE_EXPERIMENT_EXTENSION_KEY
                            + ".derived_core_fields."
                            + field_name
                            + f" for preset {preset_id}",
                        )
                    )
        except (TypeError, ValueError) as exc:
            issues.append(
                _issue(
                    "invalid_scope_experiment",
                    f"extensions.{SCOPE_EXPERIMENT_EXTENSION_KEY}",
                    str(exc),
                )
            )

    label_policy_payload = normalized_manifest["extensions"].get(
        LABEL_POLICY_EXTENSION_KEY
    )
    if label_policy_payload is not None:
        try:
            normalized_manifest["extensions"][LABEL_POLICY_EXTENSION_KEY] = (
                normalize_label_policy_payload(
                    label_policy_payload,
                    field_name=f"extensions.{LABEL_POLICY_EXTENSION_KEY}",
                )
            )
        except (TypeError, ValueError) as exc:
            issues.append(
                _issue(
                    "invalid_label_policy",
                    f"extensions.{LABEL_POLICY_EXTENSION_KEY}",
                    str(exc),
                )
            )

    train_scope_taxonomy_payload = normalized_manifest["extensions"].get(
        TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY
    )
    if train_scope_taxonomy_payload is not None:
        try:
            normalized_manifest["extensions"][TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY] = (
                normalize_train_scope_payload(
                    train_scope_taxonomy_payload,
                    field_name=f"extensions.{TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY}",
                )
            )
        except (TypeError, ValueError) as exc:
            issues.append(
                _issue(
                    "invalid_train_scope_taxonomy",
                    f"extensions.{TRAIN_SCOPE_TAXONOMY_EXTENSION_KEY}",
                    str(exc),
                )
            )

    evaluation_binding = payload.get("evaluation_binding", {})
    if not isinstance(evaluation_binding, Mapping):
        issues.append(
            _issue(
                "wrong_type",
                "evaluation_binding",
                "evaluation_binding must be an object",
            )
        )
        evaluation_binding = {}
    normalized_manifest["evaluation_binding"] = dict(evaluation_binding)

    controller_provenance = payload.get("controller_provenance")
    if controller_provenance is not None:
        if not isinstance(controller_provenance, Mapping):
            issues.append(
                _issue(
                    "wrong_type",
                    "controller_provenance",
                    "controller_provenance must be an object when present",
                )
            )
        else:
            normalized_manifest["controller_provenance"] = dict(controller_provenance)

    computed_core_digest = core_digest(normalized_core)
    normalized_manifest["core_digest"] = computed_core_digest
    declared_core_digest = payload.get("core_digest")
    if (
        declared_core_digest is not None
        and declared_core_digest != computed_core_digest
    ):
        issues.append(
            _issue(
                "core_digest_mismatch",
                "core_digest",
                "declared core_digest does not match the normalized core payload",
            )
        )

    selected_checkpoint_meta = _inspect_checkpoint_path(
        repo_root,
        normalized_core.get("checkpoint_selected"),
    )
    loaded_checkpoint_meta = _inspect_checkpoint_path(
        repo_root,
        normalized_core.get("checkpoint_loaded"),
    )
    if normalized_core.get("checkpoint_selected") is not None:
        if not bool(selected_checkpoint_meta["exists"]):
            issues.append(
                _issue(
                    "invalid_checkpoint_binding",
                    "core.checkpoint_selected",
                    str(selected_checkpoint_meta["error"]),
                )
            )
        elif not bool(selected_checkpoint_meta["loadable"]):
            issues.append(
                _issue(
                    "invalid_checkpoint_binding",
                    "core.checkpoint_selected",
                    str(selected_checkpoint_meta["error"]),
                )
            )
    if normalized_core.get("checkpoint_loaded") is not None:
        if not bool(loaded_checkpoint_meta["exists"]):
            issues.append(
                _issue(
                    "invalid_checkpoint_binding",
                    "core.checkpoint_loaded",
                    str(loaded_checkpoint_meta["error"]),
                )
            )
        elif not bool(loaded_checkpoint_meta["loadable"]):
            issues.append(
                _issue(
                    "invalid_checkpoint_binding",
                    "core.checkpoint_loaded",
                    str(loaded_checkpoint_meta["error"]),
                )
            )
        elif selected_checkpoint_meta.get(
            "normalized_checkpoint_path"
        ) != loaded_checkpoint_meta.get("normalized_checkpoint_path"):
            issues.append(
                _issue(
                    "checkpoint_mismatch",
                    "core.checkpoint_loaded",
                    "core.checkpoint_loaded does not resolve to core.checkpoint_selected",
                )
            )

    binding_meta: dict[str, Any] = {
        "selected_checkpoint": selected_checkpoint_meta,
        "checkpoint_loaded": loaded_checkpoint_meta,
        "server_load_path": None,
        "base_model_path": evaluation_binding.get("base_model_path"),
        "server_load_mode": evaluation_binding.get("server_load_mode"),
        "eval_uses_finetuned": evaluation_binding.get("eval_uses_finetuned"),
    }
    server_load_path = evaluation_binding.get("server_load_path")
    if not isinstance(server_load_path, str) or not server_load_path.strip():
        issues.append(
            _issue(
                "invalid_checkpoint_binding",
                "evaluation_binding.server_load_path",
                "evaluation_binding.server_load_path is required for fail-closed checkpoint binding",
            )
        )
    else:
        binding_meta["server_load_path"] = server_load_path.strip()
        server_load_meta = _inspect_checkpoint_path(repo_root, server_load_path.strip())
        binding_meta["server_load_path_inspection"] = server_load_meta
        if not bool(server_load_meta["exists"]):
            issues.append(
                _issue(
                    "invalid_checkpoint_binding",
                    "evaluation_binding.server_load_path",
                    str(server_load_meta["error"]),
                )
            )
        elif not bool(server_load_meta["loadable"]):
            issues.append(
                _issue(
                    "invalid_checkpoint_binding",
                    "evaluation_binding.server_load_path",
                    str(server_load_meta["error"]),
                )
            )
        elif selected_checkpoint_meta.get(
            "normalized_checkpoint_path"
        ) != server_load_meta.get("normalized_checkpoint_path"):
            issues.append(
                _issue(
                    "checkpoint_mismatch",
                    "evaluation_binding.server_load_path",
                    "evaluation_binding.server_load_path does not resolve to core.checkpoint_selected",
                )
            )
        elif loaded_checkpoint_meta.get(
            "normalized_checkpoint_path"
        ) != server_load_meta.get("normalized_checkpoint_path"):
            issues.append(
                _issue(
                    "checkpoint_mismatch",
                    "evaluation_binding.server_load_path",
                    "evaluation_binding.server_load_path does not resolve to core.checkpoint_loaded",
                )
            )

    eval_uses_finetuned = evaluation_binding.get("eval_uses_finetuned")
    if eval_uses_finetuned is not True:
        issues.append(
            _issue(
                "invalid_checkpoint_binding",
                "evaluation_binding.eval_uses_finetuned",
                "evaluation_binding.eval_uses_finetuned must be true for a valid finetuned binding",
            )
        )
    server_load_mode = evaluation_binding.get("server_load_mode")
    if server_load_mode is not None and server_load_mode != "model_path":
        issues.append(
            _issue(
                "invalid_checkpoint_binding",
                "evaluation_binding.server_load_mode",
                "evaluation_binding.server_load_mode must be 'model_path' when present",
            )
        )
    base_model_path = evaluation_binding.get("base_model_path")
    if (
        isinstance(base_model_path, str)
        and base_model_path.strip()
        and isinstance(server_load_path, str)
        and server_load_path.strip() == base_model_path.strip()
    ):
        issues.append(
            _issue(
                "invalid_checkpoint_binding",
                "evaluation_binding.server_load_path",
                "evaluation_binding.server_load_path falls back to the declared base model",
            )
        )

    expected_controller_config_hash = controller_config_hash(
        branch=(normalized_core["branch"] if "branch" in normalized_core else None),
        policy_horizon=(
            normalized_core["policy_horizon"]
            if "policy_horizon" in normalized_core
            else None
        ),
        n_action_steps=(
            normalized_core["n_action_steps"]
            if "n_action_steps" in normalized_core
            else None
        ),
        relative_contract=(
            normalized_core["relative_absolute_action_contract"]
            if "relative_absolute_action_contract" in normalized_core
            else None
        ),
        controller_provenance=(
            normalized_manifest["controller_provenance"]
            if "controller_provenance" in normalized_manifest
            else None
        ),
    )
    binding_meta["expected_controller_config_hash"] = expected_controller_config_hash
    if (
        expected_controller_config_hash is not None
        and normalized_core.get("controller_config_hash")
        != expected_controller_config_hash
    ):
        issues.append(
            _issue(
                "controller_config_hash_mismatch",
                "core.controller_config_hash",
                "core.controller_config_hash does not match the compact controller-side authority surface",
            )
        )

    return {
        "normalized_manifest": normalized_manifest,
        "core_digest": computed_core_digest,
        "issues": issues,
        "formal_eligibility": "ALLOW" if not issues else "BLOCK",
        "checkpoint_binding": binding_meta,
    }


__all__ = [
    "INDICATOR_SOURCE_FIELD",
    "PROMPT_SOURCE_FIELD",
    "RUN_MANIFEST_ARTIFACT_KIND",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "TEXT_CARRIER_ROUTE",
    "TEXT_CARRIER_SCHEMA_VERSION",
    "build_run_manifest_from_sources",
    "controller_config_hash",
    "core_digest",
    "validate_run_manifest",
]
