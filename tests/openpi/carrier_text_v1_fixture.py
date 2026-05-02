from __future__ import annotations

from work.recap.lerobot_export import dataset_export


def carrier_text_v1_handoff_metadata() -> dict[str, str]:
    return {
        "task_text_field": dataset_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD,
        "carrier_route": dataset_export.EXPORTER_CARRIER_ROUTE,
        "carrier_schema_version": dataset_export.EXPORTER_CARRIER_SCHEMA_VERSION,
        "prompt_source_field": dataset_export.EXPORTER_PROMPT_SOURCE_FIELD,
        "prompt_route": dataset_export.EXPORTER_PROMPT_ROUTE,
        "conditioning_mode": dataset_export.EXPORTER_CONDITIONING_MODE,
    }
