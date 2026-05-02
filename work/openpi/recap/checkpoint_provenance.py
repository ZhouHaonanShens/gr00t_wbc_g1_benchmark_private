from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .checkpoint import RecapCheckpointBundle, read_json, write_json
from .dataset import RecapDatasetBundle
from .train_config import RepairedStageConfig


REQUIRED_STAGE_PROVENANCE_FIELDS: tuple[str, ...] = (
    "critic_checkpoint_ref",
    "indicator_mode_train",
    "indicator_dropout_p",
    "epsilon_source",
    "human_correction_override",
)
INFORMATIVE_POSITIVE_REWEIGHT_KEY = "informative_positive_reweight"


@dataclass(frozen=True)
class RepairedStageProvenance:
    stage: str
    critic_checkpoint_ref: str
    indicator_mode_train: str
    indicator_dropout_p: float
    epsilon_source: str
    human_correction_override: bool
    consumer_mode: str
    fixed_indicator_mode: str | None
    source_dataset_dir: str
    prepared_dataset_dir: str
    materialization_report_ref: str | None
    informative_positive_reweight: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "critic_checkpoint_ref": self.critic_checkpoint_ref,
            "indicator_mode_train": self.indicator_mode_train,
            "indicator_dropout_p": float(self.indicator_dropout_p),
            "epsilon_source": self.epsilon_source,
            "human_correction_override": bool(self.human_correction_override),
            "consumer_mode": self.consumer_mode,
            "fixed_indicator_mode": self.fixed_indicator_mode,
            "source_dataset_dir": self.source_dataset_dir,
            "prepared_dataset_dir": self.prepared_dataset_dir,
            "materialization_report_ref": self.materialization_report_ref,
            **(
                {INFORMATIVE_POSITIVE_REWEIGHT_KEY: self.informative_positive_reweight}
                if self.informative_positive_reweight is not None
                else {}
            ),
        }


def _as_contract(bundle: RecapDatasetBundle) -> dict[str, object]:
    raw = bundle.recap_contract
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items()}


def _require_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"missing required provenance field {field_name}")
    return text


def _parse_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field_name} must be bool-like, got {value!r}")


def _parse_float(value: object, *, field_name: str) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be float-like, got {value!r}") from exc


def _parse_int(value: object, *, field_name: str) -> int:
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be int-like, got {value!r}") from exc


def _parse_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object, got {value!r}")
    return {str(key): item for key, item in value.items()}


def _optional_informative_positive_reweight(
    contract: dict[str, object], *, stage: str
) -> dict[str, object] | None:
    raw = contract.get(INFORMATIVE_POSITIVE_REWEIGHT_KEY)
    if raw is None or raw == "":
        return None
    payload = _parse_mapping(
        raw,
        field_name=f"recap_advantage_input_contract.{INFORMATIVE_POSITIVE_REWEIGHT_KEY}",
    )
    applies_to_stage = _require_text(
        payload.get("applies_to_stage"),
        field_name=(
            f"recap_advantage_input_contract.{INFORMATIVE_POSITIVE_REWEIGHT_KEY}.applies_to_stage"
        ),
    )
    if applies_to_stage != stage:
        raise ValueError(
            "stage-specific informative positive reweight provenance mismatch; "
            + f"contract={applies_to_stage!r} stage={stage!r}"
        )
    return payload


def _load_export_budget_payload(export_dir: str | Path) -> dict[str, object]:
    export_manifest_path = (
        Path(export_dir).expanduser().resolve() / "export_manifest.json"
    )
    if not export_manifest_path.is_file():
        return {}
    export_manifest = read_json(export_manifest_path)
    payload: dict[str, object] = {}
    if "default_num_train_steps" in export_manifest:
        payload["default_num_train_steps"] = _parse_int(
            export_manifest["default_num_train_steps"],
            field_name="export_manifest.default_num_train_steps",
        )
    if "num_train_steps" in export_manifest:
        payload["effective_num_train_steps"] = _parse_int(
            export_manifest["num_train_steps"],
            field_name="export_manifest.num_train_steps",
        )
    num_train_steps_source = str(
        export_manifest.get("num_train_steps_source", "")
    ).strip()
    if num_train_steps_source:
        payload["num_train_steps_source"] = num_train_steps_source
    if "default_save_interval" in export_manifest:
        payload["default_save_interval"] = _parse_int(
            export_manifest["default_save_interval"],
            field_name="export_manifest.default_save_interval",
        )
    if "save_interval" in export_manifest:
        payload["effective_save_interval"] = _parse_int(
            export_manifest["save_interval"],
            field_name="export_manifest.save_interval",
        )
    save_interval_source = str(export_manifest.get("save_interval_source", "")).strip()
    if save_interval_source:
        payload["save_interval_source"] = save_interval_source
    return payload


def build_repaired_stage_provenance(
    *,
    dataset_bundle: RecapDatasetBundle,
    stage_config: RepairedStageConfig,
    critic_checkpoint_ref: str,
    source_dataset_dir: str | Path,
    prepared_dataset_dir: str | Path,
    materialization_report_path: str | Path | None,
) -> RepairedStageProvenance:
    contract = _as_contract(dataset_bundle)
    cli_critic_ref = str(Path(critic_checkpoint_ref).expanduser().resolve())
    dataset_critic_ref = str(
        contract.get("critic_checkpoint_ref") or contract.get("critic_dir") or ""
    ).strip()
    if dataset_critic_ref:
        resolved_dataset_ref = str(Path(dataset_critic_ref).expanduser().resolve())
        if resolved_dataset_ref != cli_critic_ref:
            raise ValueError(
                "critic checkpoint provenance mismatch between prepared dataset and repaired train entry: "
                + f"dataset={resolved_dataset_ref!r} cli={cli_critic_ref!r}"
            )
    epsilon_source = _require_text(
        contract.get("epsilon_source"),
        field_name="recap_advantage_input_contract.epsilon_source",
    )
    indicator_dropout_p = _parse_float(
        contract.get("indicator_dropout_p", stage_config.indicator_dropout_p),
        field_name="recap_advantage_input_contract.indicator_dropout_p",
    )
    if abs(indicator_dropout_p - float(stage_config.indicator_dropout_p)) > 1e-9:
        raise ValueError(
            "task-7 repaired stages must preserve task-6 dropout authority; "
            + f"contract={indicator_dropout_p!r} stage={stage_config.indicator_dropout_p!r}"
        )
    human_correction_override = _parse_bool(
        contract.get(
            "human_correction_override", stage_config.human_correction_override
        ),
        field_name="recap_advantage_input_contract.human_correction_override",
    )
    if not human_correction_override:
        raise ValueError(
            "task-7 repaired stages require human_correction_override=true in machine-readable provenance"
        )
    return RepairedStageProvenance(
        stage=stage_config.stage,
        critic_checkpoint_ref=cli_critic_ref,
        indicator_mode_train=stage_config.indicator_mode_train,
        indicator_dropout_p=indicator_dropout_p,
        epsilon_source=epsilon_source,
        human_correction_override=human_correction_override,
        consumer_mode=stage_config.consumer_mode,
        fixed_indicator_mode=stage_config.fixed_indicator_mode,
        source_dataset_dir=str(Path(source_dataset_dir).expanduser().resolve()),
        prepared_dataset_dir=str(Path(prepared_dataset_dir).expanduser().resolve()),
        materialization_report_ref=(
            str(Path(materialization_report_path).expanduser().resolve())
            if materialization_report_path is not None
            else None
        ),
        informative_positive_reweight=_optional_informative_positive_reweight(
            contract,
            stage=stage_config.stage,
        ),
    )


def _augment_json_payload(
    *,
    path: Path,
    root_updates: dict[str, object],
    nested_key: str | None = None,
    nested_updates: dict[str, object] | None = None,
) -> None:
    payload = read_json(path)
    payload.update(root_updates)
    if nested_key is not None:
        nested_raw = payload.get(nested_key)
        nested_payload = (
            {str(key): value for key, value in nested_raw.items()}
            if isinstance(nested_raw, dict)
            else {}
        )
        nested_payload.update(nested_updates or {})
        payload[nested_key] = nested_payload
    write_json(path, payload)


def annotate_stage_checkpoint_artifacts(
    *,
    checkpoint_bundle: RecapCheckpointBundle,
    export_dir: str | Path,
    dataset_bundle: RecapDatasetBundle,
    stage_config: RepairedStageConfig,
    critic_checkpoint_ref: str,
    source_dataset_dir: str | Path,
    prepared_dataset_dir: str | Path,
    materialization_report_path: str | Path | None,
) -> RepairedStageProvenance:
    stage_provenance = build_repaired_stage_provenance(
        dataset_bundle=dataset_bundle,
        stage_config=stage_config,
        critic_checkpoint_ref=critic_checkpoint_ref,
        source_dataset_dir=source_dataset_dir,
        prepared_dataset_dir=prepared_dataset_dir,
        materialization_report_path=materialization_report_path,
    )
    stage_payload = {
        **stage_provenance.to_dict(),
        **_load_export_budget_payload(export_dir),
    }
    root_updates = {
        **stage_payload,
        "stage_provenance": stage_payload,
        "required_stage_provenance_fields": list(REQUIRED_STAGE_PROVENANCE_FIELDS),
    }
    _augment_json_payload(
        path=checkpoint_bundle.train_manifest_path,
        root_updates=root_updates,
        nested_key="training_route",
        nested_updates=stage_payload,
    )
    _augment_json_payload(
        path=checkpoint_bundle.checkpoint_dir / "train_manifest.json",
        root_updates=root_updates,
        nested_key="training_route",
        nested_updates=stage_payload,
    )
    _augment_json_payload(
        path=checkpoint_bundle.checkpoint_provenance_path,
        root_updates=root_updates,
        nested_key="variant_derivation",
        nested_updates=stage_payload,
    )
    _augment_json_payload(
        path=checkpoint_bundle.checkpoint_dir / "checkpoint_provenance.json",
        root_updates=root_updates,
        nested_key="variant_derivation",
        nested_updates=stage_payload,
    )
    _augment_json_payload(
        path=checkpoint_bundle.checkpoint_payload_path,
        root_updates=root_updates,
    )
    export_manifest_path = (
        Path(export_dir).expanduser().resolve() / "export_manifest.json"
    )
    if export_manifest_path.is_file():
        _augment_json_payload(
            path=export_manifest_path,
            root_updates=root_updates,
        )
    return stage_provenance


__all__ = [
    "REQUIRED_STAGE_PROVENANCE_FIELDS",
    "INFORMATIVE_POSITIVE_REWEIGHT_KEY",
    "RepairedStageProvenance",
    "annotate_stage_checkpoint_artifacts",
    "build_repaired_stage_provenance",
]
