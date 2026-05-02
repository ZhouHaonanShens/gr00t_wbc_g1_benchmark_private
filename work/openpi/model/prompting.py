from __future__ import annotations

from dataclasses import dataclass

from .spec import build_expected_runtime_prompting_payload, normalize_runtime_prompting_payload


@dataclass(frozen=True)
class RuntimePromptingBuilder:
    def build_expected(
        self,
        *,
        runtime_indicator_config: object,
        prompt_surface_bundle: object,
    ) -> dict[str, str]:
        return build_expected_runtime_prompting_payload(
            runtime_indicator_config=runtime_indicator_config,
            prompt_surface_bundle=prompt_surface_bundle,
        )

    def normalize(
        self,
        payload: dict[str, object],
        *,
        context: str,
    ) -> dict[str, str]:
        return normalize_runtime_prompting_payload(payload, context=context)
