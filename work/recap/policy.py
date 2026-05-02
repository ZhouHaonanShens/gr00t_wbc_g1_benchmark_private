# pyright: reportMissingImports=false, reportPossiblyUnboundVariable=false, reportRedeclaration=false, reportGeneralTypeIssues=false, reportMissingSuperCall=false
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any, Callable, Protocol

from .advantage import validate_advantage_input_value
from .run_manifest import INDICATOR_SOURCE_FIELD
from .run_manifest import PROMPT_SOURCE_FIELD
from .run_manifest import TEXT_CARRIER_ROUTE
from .run_manifest import TEXT_CARRIER_SCHEMA_VERSION
from .text_indicator import (
    canonical_text_indicator_metadata,
    normalize_indicator_mode,
    require_formalize_language_false,
    require_prompt_raw,
)
from .transformers_compat import install_transformers_image_processor_fast_compat

_import_error: ModuleNotFoundError | None = None

try:
    import torch
    from transformers import AutoModel, AutoProcessor
    from gr00t.data.interfaces import BaseProcessor
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.policy import BasePolicy
except ModuleNotFoundError as exc:
    _import_error = exc


class SupportsRecapPolicy(Protocol):
    def act(self, *args: object, **kwargs: object) -> object: ...


MAINLINE_RUNTIME_ROUTE = TEXT_CARRIER_ROUTE
MAINLINE_RUNTIME_CARRIER_SCHEMA_VERSION = TEXT_CARRIER_SCHEMA_VERSION
MAINLINE_RUNTIME_PROMPT_SOURCE_FIELD = PROMPT_SOURCE_FIELD
MAINLINE_RUNTIME_INDICATOR_SOURCE_FIELD = INDICATOR_SOURCE_FIELD
MAINLINE_RUNTIME_POLICY_CLASS_NAME = "TextIndicatorGr00tPolicy"
DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE = "advantage_input_numeric_diagnostic"
DIAGNOSTIC_NUMERIC_ADV_POLICY_CLASS_NAME = "AdvantageAwareGr00tPolicy"
CANONICAL_SERVING_BLOCKED_EXACT_FIELDS = frozenset(
    {
        "analysis_only",
        "debug",
        "debug_probe",
        "info",
        "policy_condition",
        "policy_condition.mode",
        "policy_condition.phase",
        "policy_condition_text",
        "privileged",
        "prompt_conditioned",
        "rtc",
        "runtime_trace",
        "telemetry",
    }
)
CANONICAL_SERVING_BLOCKED_PREFIXES: tuple[str, ...] = (
    "analysis_only.",
    "debug.",
    "info.",
    "policy_condition.",
    "privileged.",
    "rtc.",
    "telemetry.",
)
RUNTIME_INDICATOR_CFG = "cfg"
RUNTIME_INDICATOR_CLI_MODES: tuple[str, ...] = (
    "positive",
    "negative",
    "omit",
    RUNTIME_INDICATOR_CFG,
)
MAINLINE_RUNTIME_INDICATOR_MODES = (
    "omit",
    "positive",
    "negative",
)
_RUNTIME_PROMPT_MAINLINE_INDICATOR_MODES = frozenset(
    str(mode)
    for mode in RUNTIME_INDICATOR_CLI_MODES
    if str(mode).strip().lower() != RUNTIME_INDICATOR_CFG
)
if (
    frozenset(MAINLINE_RUNTIME_INDICATOR_MODES)
    != _RUNTIME_PROMPT_MAINLINE_INDICATOR_MODES
):
    raise RuntimeError(
        "runtime prompt indicator authority drifted from the canonical mainline runtime modes"
    )


def validate_mainline_runtime_indicator_mode(
    indicator_mode: object,
    *,
    field_name: str = "indicator_mode",
) -> str:
    if indicator_mode is None:
        expected_modes = "|".join(MAINLINE_RUNTIME_INDICATOR_MODES)
        raise ValueError(
            f"{field_name} is required for the {MAINLINE_RUNTIME_ROUTE!r} mainline runtime route; "
            + f"expected {expected_modes}"
        )
    return normalize_indicator_mode(indicator_mode, field_name=field_name)


def resolve_runtime_policy_route(route: object | None = None) -> str:
    if route is None:
        return MAINLINE_RUNTIME_ROUTE
    normalized = str(route).strip()
    if not normalized:
        return MAINLINE_RUNTIME_ROUTE
    if normalized in {MAINLINE_RUNTIME_ROUTE, DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE}:
        return normalized
    raise ValueError(
        "Unknown runtime route: "
        + f"{route!r}; expected {MAINLINE_RUNTIME_ROUTE!r} or {DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE!r}"
    )


def resolve_runtime_policy_class(
    *,
    route: object | None = None,
    indicator_mode: object | None = None,
) -> type[object]:
    resolved_route = resolve_runtime_policy_route(route)
    if resolved_route == MAINLINE_RUNTIME_ROUTE:
        _ = validate_mainline_runtime_indicator_mode(indicator_mode)
        return TextIndicatorGr00tPolicy
    return AdvantageAwareGr00tPolicy


def build_runtime_policy_spec(
    *,
    route: object | None = None,
    indicator_mode: object | None = None,
) -> dict[str, Any]:
    resolved_route = resolve_runtime_policy_route(route)
    if resolved_route == MAINLINE_RUNTIME_ROUTE:
        normalized_indicator_mode = validate_mainline_runtime_indicator_mode(
            indicator_mode
        )
        return {
            "route": MAINLINE_RUNTIME_ROUTE,
            "carrier_route": MAINLINE_RUNTIME_ROUTE,
            "carrier_schema_version": MAINLINE_RUNTIME_CARRIER_SCHEMA_VERSION,
            "prompt_source_field": MAINLINE_RUNTIME_PROMPT_SOURCE_FIELD,
            "indicator_source": MAINLINE_RUNTIME_INDICATOR_SOURCE_FIELD,
            "indicator_mode": normalized_indicator_mode,
            "policy_class_name": MAINLINE_RUNTIME_POLICY_CLASS_NAME,
            "mainline_authority": True,
            "diagnostic_only": False,
            "runtime_indicator_mode_required": True,
            "runtime_supported_indicator_modes": list(MAINLINE_RUNTIME_INDICATOR_MODES),
        }
    return {
        "route": DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE,
        "policy_class_name": DIAGNOSTIC_NUMERIC_ADV_POLICY_CLASS_NAME,
        "mainline_authority": False,
        "diagnostic_only": True,
        "runtime_indicator_mode_required": False,
        "runtime_supported_indicator_modes": [],
    }


def build_frozen_runtime_policy_route(
    *,
    route: object | None = None,
    indicator_mode: object | None = None,
) -> dict[str, Any]:
    policy_spec = build_runtime_policy_spec(
        route=route,
        indicator_mode=indicator_mode,
    )
    return {
        "frozen": True,
        "route": str(policy_spec["route"]),
        "carrier_route": policy_spec.get("carrier_route"),
        "carrier_schema_version": policy_spec.get("carrier_schema_version"),
        "indicator_mode": policy_spec.get("indicator_mode"),
        "policy_class_name": str(policy_spec["policy_class_name"]),
        "mainline_authority": bool(policy_spec["mainline_authority"]),
        "diagnostic_only": bool(policy_spec["diagnostic_only"]),
        "runtime_indicator_mode_required": bool(
            policy_spec["runtime_indicator_mode_required"]
        ),
        "runtime_supported_indicator_modes": list(
            policy_spec["runtime_supported_indicator_modes"]
        ),
    }


def build_comparability_policy_route_freeze(
    *,
    route: object | None = None,
    indicator_mode: object | None = None,
) -> dict[str, Any]:
    resolved_route = (
        DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE if route is None else route
    )
    return build_frozen_runtime_policy_route(
        route=resolved_route,
        indicator_mode=indicator_mode,
    )


def _require_mapping_object(
    value: Mapping[str, Any] | object,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _clone_model_inputs(value: Any) -> Any:
    if _import_error is not None:
        raise ModuleNotFoundError(
            "Diagnostic action helpers require torch, transformers, and gr00t to be installed"
        ) from _import_error
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {str(key): _clone_model_inputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_model_inputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_model_inputs(item) for item in value)
    return value


def _infer_batch_size_from_model_inputs(collated_inputs: Mapping[str, Any]) -> int:
    stack: list[Any] = [collated_inputs]
    while stack:
        current = stack.pop()
        if torch.is_tensor(current) and current.ndim >= 1:
            return int(current.shape[0])
        if isinstance(current, Mapping):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return 1


def _inject_advantage_into_model_inputs(
    collated_inputs: Mapping[str, Any],
    *,
    advantage_value: Any | None,
    device: Any,
) -> dict[str, Any]:
    cloned = _clone_model_inputs(collated_inputs)
    target = cloned.get("inputs") if isinstance(cloned.get("inputs"), Mapping) else cloned
    if not isinstance(target, dict):
        raise TypeError("collated_inputs must resolve to a mutable mapping for advantage injection")
    target.pop("advantage", None)
    if advantage_value is None:
        return cloned
    batch_size = _infer_batch_size_from_model_inputs(cloned)
    if isinstance(advantage_value, (list, tuple)):
        if len(advantage_value) != batch_size:
            raise ValueError(
                "Sequence diagnostic advantage must match batch size: "
                f"{len(advantage_value)} != {batch_size}"
            )
        resolved_advantage = [
            float(
                validate_advantage_input_value(
                    raw_value,
                    context=f"diagnostic_advantage[{idx}]",
                )
            )
            for idx, raw_value in enumerate(advantage_value)
        ]
        target["advantage"] = torch.tensor(
            [[value] for value in resolved_advantage],
            dtype=torch.bfloat16,
            device=device,
        )
        return cloned
    resolved_advantage_scalar = float(
        validate_advantage_input_value(
            advantage_value,
            context="diagnostic_advantage",
        )
    )
    target["advantage"] = torch.full(
        (batch_size, 1),
        resolved_advantage_scalar,
        dtype=torch.bfloat16,
        device=device,
    )
    return cloned


def _serialize_action_mapping(action: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    payload: dict[str, Any] = {}
    for raw_key, raw_value in action.items():
        arr = np.asarray(raw_value, dtype=np.float32)
        payload[str(raw_key)] = arr.tolist()
    return payload


def capture_local_diagnostic_action_stages(
    *,
    model: Any,
    collated_inputs: Mapping[str, Any],
    processor: Any,
    embodiment_tag: Any,
    batched_states: Mapping[str, Any] | None,
    advantage_value: Any | None,
    seed: int | None,
    postprocess_action: Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], str | None]] | None = None,
    controller_input_transform: Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], str | None]] | None = None,
) -> dict[str, Any]:
    if _import_error is not None:
        raise ModuleNotFoundError(
            "capture_local_diagnostic_action_stages requires torch, transformers, and gr00t to be installed"
        ) from _import_error

    prepared_inputs = _inject_advantage_into_model_inputs(
        collated_inputs,
        advantage_value=advantage_value,
        device=getattr(model, "device", None),
    )
    prepared_inputs = _rec_to_dtype(prepared_inputs, dtype=torch.bfloat16)
    fork_devices: list[int] = []
    model_device = getattr(model, "device", None)
    if getattr(model_device, "type", None) == "cuda":
        device_index = getattr(model_device, "index", None)
        if device_index is None:
            device_index = torch.cuda.current_device()
        fork_devices = [int(device_index)]

    with torch.random.fork_rng(devices=fork_devices, enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        autocast_device_type = (
            "cuda"
            if getattr(model_device, "type", None) == "cuda"
            else "cpu"
        )
        with torch.inference_mode():
            with torch.autocast(
                device_type=autocast_device_type,
                dtype=torch.bfloat16,
                enabled=(autocast_device_type == "cuda"),
            ):
                model_pred = model.get_action(**prepared_inputs)

    normalized_action = model_pred["action_pred"].float().detach().cpu().numpy()
    decoded_action_payload: dict[str, Any] = {
        "available": False,
        "reason": "decode_action_not_attempted",
        "action": None,
    }
    postprocessed_action_payload: dict[str, Any] = {
        "available": False,
        "reason": "postprocess_seam_not_available",
        "action": None,
    }
    controller_input_payload: dict[str, Any] = {
        "available": False,
        "reason": "controller_input_seam_not_available",
        "action": None,
    }
    decoded_action_mapping: dict[str, Any] | None = None
    if processor is not None and batched_states is not None:
        try:
            decoded_action = processor.decode_action(
                normalized_action,
                embodiment_tag,
                batched_states,
            )
            decoded_action_mapping = {
                str(key): value.astype("float32")
                for key, value in decoded_action.items()
            }
            decoded_action_payload = {
                "available": True,
                "reason": None,
                "action": _serialize_action_mapping(decoded_action_mapping),
            }
        except Exception as exc:
            decoded_action_payload = {
                "available": False,
                "reason": f"decode_action_failed: {type(exc).__name__}: {exc}",
                "action": None,
            }
    else:
        decoded_action_payload = {
            "available": False,
            "reason": "decode_action_requires_processor_and_batched_states",
            "action": None,
        }

    postprocess_source = decoded_action_mapping
    if postprocess_action is not None and decoded_action_mapping is not None:
        try:
            postprocessed_action, postprocess_reason = postprocess_action(decoded_action_mapping)
            postprocess_source = {str(key): value for key, value in postprocessed_action.items()}
            postprocessed_action_payload = {
                "available": True,
                "reason": postprocess_reason,
                "action": _serialize_action_mapping(postprocess_source),
            }
        except Exception as exc:
            postprocessed_action_payload = {
                "available": False,
                "reason": f"postprocess_failed: {type(exc).__name__}: {exc}",
                "action": None,
            }

    if controller_input_transform is not None and postprocess_source is not None:
        try:
            controller_input, controller_reason = controller_input_transform(postprocess_source)
            controller_input_payload = {
                "available": True,
                "reason": controller_reason,
                "action": _serialize_action_mapping(controller_input),
            }
        except Exception as exc:
            controller_input_payload = {
                "available": False,
                "reason": f"controller_input_failed: {type(exc).__name__}: {exc}",
                "action": None,
            }

    return {
        "advantage": None if advantage_value is None else float(advantage_value),
        "raw_normalized_action": {
            "available": True,
            "reason": None,
            "action": normalized_action.astype("float32").tolist(),
        },
        "decoded_action": decoded_action_payload,
        "postprocessed_action": postprocessed_action_payload,
        "controller_input": controller_input_payload,
    }


def _canonical_serving_field_is_blocked(field_path: str) -> bool:
    if field_path in CANONICAL_SERVING_BLOCKED_EXACT_FIELDS:
        return True
    return any(
        field_path.startswith(prefix) for prefix in CANONICAL_SERVING_BLOCKED_PREFIXES
    )


def _strip_non_authoritative_serving_fields(
    payload: Mapping[str, Any],
    *,
    field_path: str,
    stripped_field_paths: list[str],
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for raw_key, raw_value in payload.items():
        key = str(raw_key)
        child_path = f"{field_path}.{key}" if field_path else key
        if _canonical_serving_field_is_blocked(child_path):
            stripped_field_paths.append(child_path)
            continue
        if isinstance(raw_value, Mapping):
            sanitized_child = _strip_non_authoritative_serving_fields(
                raw_value,
                field_path=child_path,
                stripped_field_paths=stripped_field_paths,
            )
            if sanitized_child:
                sanitized[key] = sanitized_child
            continue
        sanitized[key] = raw_value
    return sanitized


def find_non_authoritative_serving_field_paths(
    observation: Mapping[str, Any] | object,
    *,
    field_name: str = "observation",
) -> tuple[str, ...]:
    payload = _require_mapping_object(observation, field_name=field_name)
    stripped_field_paths: list[str] = []
    _ = _strip_non_authoritative_serving_fields(
        payload,
        field_path="",
        stripped_field_paths=stripped_field_paths,
    )
    return tuple(sorted(set(stripped_field_paths)))


def filter_canonical_serving_observation(
    observation: Mapping[str, Any] | object,
    *,
    field_name: str = "observation",
    reject_blocked: bool = False,
) -> dict[str, Any]:
    payload = _require_mapping_object(observation, field_name=field_name)
    stripped_field_paths: list[str] = []
    sanitized = _strip_non_authoritative_serving_fields(
        payload,
        field_path="",
        stripped_field_paths=stripped_field_paths,
    )
    if reject_blocked and stripped_field_paths:
        raise ValueError(
            f"{field_name} contains non-authoritative serving fields: {sorted(set(stripped_field_paths))}"
        )
    return sanitized


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    if isinstance(x, torch.Tensor):
        if torch.is_floating_point(x):
            return x.to(dtype=dtype)
        return x
    if isinstance(x, dict):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    if hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


if _import_error is not None:

    class AdvantageAwareGr00tPolicy:
        def __init__(self, *args: object, **kwargs: object):
            del args, kwargs
            raise ModuleNotFoundError(
                "AdvantageAwareGr00tPolicy requires torch and gr00t to be installed"
            ) from _import_error

    class TextIndicatorGr00tPolicy:
        def __init__(self, *args: object, **kwargs: object):
            del args, kwargs
            raise ModuleNotFoundError(
                "TextIndicatorGr00tPolicy requires torch and gr00t to be installed"
            ) from _import_error

else:
    _ADVANTAGE_INJECTION_RULES = {
        "sign_consistent": lambda value: float(value),
        "legacy_negate": lambda value: -1.0 * float(value),
    }

    def _init_gr00t_policy_runtime(
        policy: Gr00tPolicy,
        *,
        embodiment_tag: Any,
        model_path: str,
        device: int | str,
        strict: bool,
        force_slow_processor: bool = False,
    ) -> None:
        import gr00t.model  # noqa: F401
        from torch.distributions import Distribution
        import gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 as processing_gr00t_n1d6

        BasePolicy.__init__(policy, strict=bool(strict))
        model_dir = Path(model_path)

        Distribution.set_default_validate_args(False)

        original_build_processor = processing_gr00t_n1d6.build_processor
        processor_use_fast = not bool(force_slow_processor)

        def _build_processor_with_lane_processor_mode(
            model_name: str, transformers_loading_kwargs: dict[str, object]
        ) -> Any:
            patched_kwargs = dict(transformers_loading_kwargs or {})
            patched_kwargs.setdefault("trust_remote_code", True)
            patched_kwargs["use_fast"] = processor_use_fast
            return original_build_processor(model_name, patched_kwargs)

        processing_gr00t_n1d6.build_processor = (
            _build_processor_with_lane_processor_mode
        )
        try:
            model = AutoModel.from_pretrained(model_dir, low_cpu_mem_usage=False)
            model.eval()
            model.to(device=device, dtype=torch.bfloat16)
            policy.model = model

            processor: BaseProcessor = AutoProcessor.from_pretrained(
                model_dir,
                trust_remote_code=True,
                use_fast=processor_use_fast,
            )
            processor.eval()
            policy.processor = processor
        finally:
            processing_gr00t_n1d6.build_processor = original_build_processor

        policy.embodiment_tag = embodiment_tag
        policy.modality_configs = policy.processor.get_modality_configs()[
            policy.embodiment_tag.value
        ]
        policy.collate_fn = policy.processor.collator

        language_keys = policy.modality_configs["language"].modality_keys
        language_delta_indices = policy.modality_configs["language"].delta_indices
        assert len(language_keys) == 1, "Only one language key is supported"
        assert len(language_delta_indices) == 1, (
            "Only one language delta index is supported"
        )
        policy.language_key = language_keys[0]

    def _maybe_seed_from_options(options: dict[str, Any] | None) -> None:
        if not options or "seed" not in options:
            return
        raw = options.get("seed")
        if raw is None:
            return
        seed = int(raw)
        if seed < 0:
            raise ValueError(f"options['seed'] must be >= 0, got {seed}")

        import random

        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass

        _ = torch.manual_seed(seed)
        if torch.cuda.is_available():
            _ = torch.cuda.manual_seed_all(seed)

    class AdvantageAwareGr00tPolicy(Gr00tPolicy):
        def __init__(
            self,
            embodiment_tag: Any,
            model_path: str,
            *,
            device: int | str,
            strict: bool = True,
            attn_implementation: str = "eager",
            advantage_injection_rule: str = "sign_consistent",
        ) -> None:
            install_transformers_image_processor_fast_compat()
            _init_gr00t_policy_runtime(
                self,
                embodiment_tag=embodiment_tag,
                model_path=str(model_path),
                device=device,
                strict=bool(strict),
            )

            impl = str(attn_implementation).strip()
            if impl:
                self._force_attn_implementation(impl)
            rule = str(advantage_injection_rule).strip() or "sign_consistent"
            if rule not in _ADVANTAGE_INJECTION_RULES:
                valid_rules = sorted(_ADVANTAGE_INJECTION_RULES)
                raise ValueError(
                    f"Unsupported advantage injection rule: {rule!r}; expected one of {valid_rules}"
                )
            self.advantage_injection_rule: str = rule

        def _force_attn_implementation(self, impl: str) -> None:
            cfg = getattr(self.model, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                setattr(cfg, "_attn_implementation", impl)
            for m in self.model.modules():
                c = getattr(m, "config", None)
                if c is not None and hasattr(c, "_attn_implementation"):
                    setattr(c, "_attn_implementation", impl)

        def _maybe_seed_from_options(self, options: dict[str, Any] | None) -> None:
            _maybe_seed_from_options(options)

        def _resolve_serving_advantage(self, advantage_value: Any) -> float:
            transform = _ADVANTAGE_INJECTION_RULES[self.advantage_injection_rule]
            if isinstance(advantage_value, bool):
                raise ValueError("options['advantage'] must not be bool")
            value = float(transform(advantage_value))
            if not math.isfinite(value):
                raise ValueError(
                    f"options['advantage'] must be finite after injection transform, got {value!r}"
                )
            return float(
                validate_advantage_input_value(
                    value,
                    context="policy.options['advantage']",
                )
            )

        def _get_action(
            self, observation: dict[str, Any], options: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            self._maybe_seed_from_options(options)
            unbatched_observations = self._unbatch_observation(observation)
            processed_inputs = []
            states = []

            for index, obs in enumerate(unbatched_observations):
                sanitized_obs = filter_canonical_serving_observation(
                    obs,
                    field_name=f"observation[{index}]",
                )
                vla_step_data = self._to_vla_step_data(sanitized_obs)
                states.append(vla_step_data.states)
                from gr00t.data.types import MessageType

                messages = [
                    {"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}
                ]
                processed_inputs.append(self.processor(messages))

            collated_inputs = self.collate_fn(processed_inputs)
            collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

            if options is not None and "advantage" in options:
                advantage_value = options["advantage"]
                if advantage_value is not None:
                    batch_size = len(unbatched_observations)
                    advantage_tensor = torch.full(
                        (batch_size, 1),
                        self._resolve_serving_advantage(advantage_value),
                        dtype=torch.bfloat16,
                        device=self.model.device,
                    )
                    if "inputs" in collated_inputs:
                        collated_inputs["inputs"]["advantage"] = advantage_tensor
                    else:
                        collated_inputs["advantage"] = advantage_tensor

            with torch.inference_mode():
                model_pred = self.model.get_action(**collated_inputs)
            normalized_action = model_pred["action_pred"].float()

            import numpy as np

            batched_states = {}
            for k in self.modality_configs["state"].modality_keys:
                batched_states[k] = np.stack([s[k] for s in states], axis=0)
            unnormalized_action = self.processor.decode_action(
                normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
            )
            casted_action = {
                key: value.astype(np.float32)
                for key, value in unnormalized_action.items()
            }
            return casted_action, {}

    class TextIndicatorGr00tPolicy(Gr00tPolicy):
        def __init__(
            self,
            embodiment_tag: Any,
            model_path: str,
            *,
            device: int | str,
            strict: bool = True,
            attn_implementation: str = "eager",
        ) -> None:
            install_transformers_image_processor_fast_compat()
            _init_gr00t_policy_runtime(
                self,
                embodiment_tag=embodiment_tag,
                model_path=str(model_path),
                device=device,
                strict=bool(strict),
            )
            require_formalize_language_false(self.processor)
            impl = str(attn_implementation).strip()
            if impl:
                self._force_attn_implementation(impl)

        def _force_attn_implementation(self, impl: str) -> None:
            cfg = getattr(self.model, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                setattr(cfg, "_attn_implementation", impl)
            for module in self.model.modules():
                module_cfg = getattr(module, "config", None)
                if module_cfg is not None and hasattr(
                    module_cfg, "_attn_implementation"
                ):
                    setattr(module_cfg, "_attn_implementation", impl)

        def _resolve_indicator_mode(self, options: dict[str, Any] | None) -> str:
            raw_indicator_mode = (
                None if options is None else options.get("indicator_mode")
            )
            return validate_mainline_runtime_indicator_mode(
                raw_indicator_mode,
                field_name="options['indicator_mode']",
            )

        def _extract_prompt_raw(self, observation: dict[str, Any]) -> str:
            if "language" not in observation or not isinstance(
                observation["language"], dict
            ):
                raise KeyError("Observation must contain a dict 'language' field")
            if self.language_key not in observation["language"]:
                raise KeyError(
                    f"Observation language is missing required key {self.language_key!r}"
                )
            raw_value = observation["language"][self.language_key]
            if isinstance(raw_value, (list, tuple)):
                if len(raw_value) <= 0:
                    raise ValueError(
                        f"Observation language key {self.language_key!r} is empty"
                    )
                prompt_raw = raw_value[0]
            else:
                prompt_raw = raw_value
            return require_prompt_raw(
                prompt_raw, field_name=f"observation.language[{self.language_key!r}]"
            )

        def _get_action(
            self, observation: dict[str, Any], options: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            from gr00t.data.types import MessageType

            _maybe_seed_from_options(options)
            indicator_mode = self._resolve_indicator_mode(options)
            indicator_meta = canonical_text_indicator_metadata(indicator_mode)

            unbatched_observations = self._unbatch_observation(observation)
            processed_inputs = []
            states = []

            for index, obs in enumerate(unbatched_observations):
                from work.openpi.recap.runtime_prompt import RuntimeIndicatorConfig
                from work.openpi.recap.runtime_prompt import build_runtime_prompt_bundle

                sanitized_obs = filter_canonical_serving_observation(
                    obs,
                    field_name=f"observation[{index}]",
                )
                prompt_raw = self._extract_prompt_raw(sanitized_obs)
                prompt_bundle = build_runtime_prompt_bundle(
                    prompt_raw,
                    config=RuntimeIndicatorConfig(
                        requested_indicator_mode=indicator_mode,
                        indicator_mode=indicator_mode,
                        indicator_source=MAINLINE_RUNTIME_INDICATOR_SOURCE_FIELD,
                        consumer_mode="informative_adv",
                        fixed_indicator_mode=None,
                        critic_checkpoint_ref="not_applicable",
                    ),
                )
                vla_step_data = self._to_vla_step_data(sanitized_obs)
                vla_step_data.text = prompt_bundle.prompt_text
                states.append(vla_step_data.states)
                messages = [
                    {"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}
                ]
                processed_inputs.append(self.processor(messages))

            collated_inputs = self.collate_fn(processed_inputs)
            collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

            with torch.inference_mode():
                model_pred = self.model.get_action(**collated_inputs)
            normalized_action = model_pred["action_pred"].float()

            import numpy as np

            batched_states = {}
            for key in self.modality_configs["state"].modality_keys:
                batched_states[key] = np.stack([state[key] for state in states], axis=0)
            unnormalized_action = self.processor.decode_action(
                normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
            )
            casted_action = {
                key: value.astype(np.float32)
                for key, value in unnormalized_action.items()
            }
            indicator_meta.update(
                {
                    "prompt_route": prompt_bundle.prompt_provenance["prompt_route"],
                    "conditioning_mode": prompt_bundle.prompt_provenance[
                        "conditioning_mode"
                    ],
                    "authoritative_carrier_field": prompt_bundle.prompt_provenance[
                        "authoritative_carrier_field"
                    ],
                    "authoritative_carrier_schema_version": prompt_bundle.prompt_provenance[
                        "authoritative_carrier_schema_version"
                    ],
                    "authoritative_carrier_source": prompt_bundle.authoritative_carrier_source,
                    "authoritative_carrier_matches_prompt_text": (
                        prompt_bundle.authoritative_carrier_matches_prompt_text
                    ),
                }
            )
            return casted_action, dict(indicator_meta)


__all__ = [
    "SupportsRecapPolicy",
    "MAINLINE_RUNTIME_ROUTE",
    "MAINLINE_RUNTIME_CARRIER_SCHEMA_VERSION",
    "MAINLINE_RUNTIME_PROMPT_SOURCE_FIELD",
    "MAINLINE_RUNTIME_INDICATOR_SOURCE_FIELD",
    "MAINLINE_RUNTIME_POLICY_CLASS_NAME",
    "MAINLINE_RUNTIME_INDICATOR_MODES",
    "DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE",
    "DIAGNOSTIC_NUMERIC_ADV_POLICY_CLASS_NAME",
    "AdvantageAwareGr00tPolicy",
    "CANONICAL_SERVING_BLOCKED_EXACT_FIELDS",
    "CANONICAL_SERVING_BLOCKED_PREFIXES",
    "find_non_authoritative_serving_field_paths",
    "filter_canonical_serving_observation",
    "TextIndicatorGr00tPolicy",
    "build_comparability_policy_route_freeze",
    "capture_local_diagnostic_action_stages",
    "build_frozen_runtime_policy_route",
    "build_runtime_policy_spec",
    "resolve_runtime_policy_class",
    "resolve_runtime_policy_route",
    "validate_mainline_runtime_indicator_mode",
]
