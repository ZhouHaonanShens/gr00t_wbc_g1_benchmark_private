from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re

from work.openpi.eval.protocols.environment import build_libero_eval_protocol
from work.openpi.eval.protocols.tracked_gate import load_rollout_eval_manifest_v2
from work.openpi.serve.provenance import (
    EXPECTED_CHECKPOINT,
    EXPECTED_CHECKPOINT_SOURCE,
    EXPECTED_CONFIG_NAME,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

RECAP_ONLY_VARIANT = "recap_only"
STOCK_VARIANT = "stock"
SUPPORTED_EVAL_VARIANTS: tuple[str, ...] = (STOCK_VARIANT, RECAP_ONLY_VARIANT)
TRAIN_ONLY_VARIANTS: tuple[str, ...] = (RECAP_ONLY_VARIANT,)

COMPARISON_PROTOCOL = build_libero_eval_protocol(
    task_ids=(0, 1),
    seed_manifest=(7, 17),
    num_trials_per_task=2,
)

RECAP_RECORD_SCHEMA_VERSION = "openpi_libero_recap_record_v1"
TRAIN_MANIFEST_SCHEMA_VERSION = "openpi_libero_recap_train_manifest_v1"
CHECKPOINT_PROVENANCE_SCHEMA_VERSION = "openpi_libero_recap_checkpoint_v1"
SUMMARY_SCHEMA_VERSION = "openpi_libero_recap_summary_v1"
PAIRED_SUMMARY_SCHEMA_VERSION = "openpi_libero_recap_paired_summary_v1"
REPAIRED_MATRIX_SUMMARY_SCHEMA_VERSION = "openpi_recap_repaired_matrix_summary_v1"

REPAIRED_RUNTIME_LAYER_ID = "runtime_conditioning_repaired_v1"
REPAIRED_CURRENT_BACKBONE_LAYER_ID = "current_backbone_repaired_v1"
REPAIRED_PAPER_FULL_LAYER_ID = "paper_full_future_v1"

REPAIRED_RUNTIME_LAYER_LABEL = "C_existing + runtime_(omit|positive|negative|cfg)"
REPAIRED_CURRENT_BACKBONE_LAYER_LABEL = (
    "A_stock_pi05_libero / B0_omit_control_v2 / B1_fixed_positive_sft_v2 / "
    "X_shuffled_indicator_v2 / C0_recap_informative_positiveinfer_v2 / "
    "C1_recap_informative_cfg_v2"
)
REPAIRED_PAPER_FULL_LAYER_LABEL = "P0..P7"

REPAIRED_METRIC_PROFILE_ID = "libero_budget_aware_headline_v1"
REPAIRED_PRIMARY_METRIC_ORDER: tuple[str, ...] = (
    "success_rate@0.50_budget",
    "success_rate@0.75_budget",
    "throughput_like_score",
)
REPAIRED_AUDIT_METRICS: tuple[str, ...] = (
    "success_rate@1.00_budget",
    "timeout_rate",
    "median_first_success_step_fraction",
)
REPAIRED_ALL_METRICS: tuple[str, ...] = REPAIRED_PRIMARY_METRIC_ORDER + (
    REPAIRED_AUDIT_METRICS
)

CURRENT_BACKBONE_VARIANT_IDS: tuple[str, ...] = (
    "A_stock_pi05_libero",
    "B0_omit_control_v2",
    "B1_fixed_positive_sft_v2",
    "X_shuffled_indicator_v2",
    "C0_recap_informative_positiveinfer_v2",
    "C1_recap_informative_cfg_v2",
)
RUNTIME_LAYER_VARIANT_IDS: tuple[str, ...] = (
    "C_existing_runtime_omit",
    "C_existing_runtime_positive",
    "C_existing_runtime_negative",
    "C_existing_runtime_cfg",
)
PAPER_FULL_VARIANT_IDS: tuple[str, ...] = (
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
)

STOCK_SUMMARY_PATH = (
    REPO_ROOT / "agent" / "artifacts" / "openpi_libero_native" / "summary.json"
)


@dataclass(frozen=True)
class FrozenComparisonManifest:
    suite: str
    task_ids: tuple[int, ...]
    seed_manifest: tuple[int, ...]
    num_trials_per_task: int
    episode_count: int
    evaluation_tier: str


@dataclass(frozen=True)
class RepairedVariantSpec:
    variant_id: str
    layer_id: str
    short_code: str
    label: str
    source_kind: str
    status: str
    comparison_role: str
    train_stage: str | None = None
    train_consumer_mode: str | None = None
    fixed_indicator_mode: str | None = None
    prompt_text_surface: str | None = None
    per_sample_indicator_consumption: bool | None = None
    runtime_indicator_mode: str | None = None
    runtime_indicator_source: str | None = None
    derived_from: str | None = None


@dataclass(frozen=True)
class RepairedHeadlineComparison:
    comparison_id: str
    lhs_variant_id: str
    rhs_variant_id: str
    relation: str
    gate: str
    purpose: str
    metric_profile_id: str = REPAIRED_METRIC_PROFILE_ID


def _parse_int_csv(raw: str, *, field_name: str) -> tuple[int, ...]:
    values = [chunk.strip() for chunk in str(raw).split(",") if chunk.strip()]
    if not values:
        raise ValueError(
            f"{field_name} must be a non-empty comma-separated integer list"
        )
    try:
        return tuple(int(value) for value in values)
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain integers only: {raw!r}") from exc


def validate_recap_only_variant(variant: str) -> str:
    value = str(variant).strip()
    if value not in TRAIN_ONLY_VARIANTS:
        raise ValueError(
            f"Task 7 only supports --variant {RECAP_ONLY_VARIANT!r} for training, got {variant!r}"
        )
    return value


def validate_eval_variant(variant: str) -> str:
    value = str(variant).strip()
    if value not in SUPPORTED_EVAL_VARIANTS:
        raise ValueError(
            f"unsupported --variant {variant!r}; expected one of {SUPPORTED_EVAL_VARIANTS!r}"
        )
    return value


def build_frozen_comparison_manifest(
    *,
    suite: str,
    task_ids: tuple[int, ...] | list[int] | str,
    seed_manifest: tuple[int, ...] | list[int] | str,
    num_trials_per_task: int,
    gate_eval_manifest: str | Path | None = None,
) -> FrozenComparisonManifest:
    task_ids_tuple = (
        _parse_int_csv(task_ids, field_name="task_ids")
        if isinstance(task_ids, str)
        else tuple(int(value) for value in task_ids)
    )
    seed_manifest_tuple = (
        _parse_int_csv(seed_manifest, field_name="seed_manifest")
        if isinstance(seed_manifest, str)
        else tuple(int(value) for value in seed_manifest)
    )
    if gate_eval_manifest is not None:
        gate_manifest = load_rollout_eval_manifest_v2(gate_eval_manifest)
        expected_suite = str(gate_manifest.task_suite_name)
        expected_task_ids = tuple(int(value) for value in gate_manifest.task_ids)
        expected_seed_manifest = tuple(
            int(value) for value in gate_manifest.seed_manifest
        )
        expected_num_trials = int(gate_manifest.num_trials_per_task)
        suite_value = str(suite)
        if suite_value != expected_suite:
            raise ValueError(
                "train scope must match the tracked rollout eval manifest suite: "
                + f"expected {expected_suite!r}, got {suite_value!r}"
            )
        if task_ids_tuple != expected_task_ids:
            raise ValueError(
                "train scope must match the tracked rollout eval manifest task_ids: "
                + f"expected {expected_task_ids!r}, got {task_ids_tuple!r}"
            )
        if seed_manifest_tuple != expected_seed_manifest:
            raise ValueError(
                "train scope must match the tracked rollout eval manifest seed_manifest: "
                + f"expected {expected_seed_manifest!r}, got {seed_manifest_tuple!r}"
            )
        if int(num_trials_per_task) != expected_num_trials:
            raise ValueError(
                "train scope must match the tracked rollout eval manifest num_trials_per_task: "
                + f"expected {expected_num_trials!r}, got {int(num_trials_per_task)!r}"
            )
        episode_count = (
            len(expected_task_ids) * len(expected_seed_manifest) * expected_num_trials
        )
        return FrozenComparisonManifest(
            suite=expected_suite,
            task_ids=expected_task_ids,
            seed_manifest=expected_seed_manifest,
            num_trials_per_task=expected_num_trials,
            episode_count=int(episode_count),
            evaluation_tier=str(gate_manifest.manifest_name),
        )
    protocol = build_libero_eval_protocol(
        suite=str(suite),
        task_ids=task_ids_tuple,
        seed_manifest=seed_manifest_tuple,
        num_trials_per_task=int(num_trials_per_task),
    )
    if tuple(protocol.task_ids) != tuple(COMPARISON_PROTOCOL.task_ids):
        raise ValueError(
            "Task 7 freezes task_ids to the comparison tier only: "
            + f"expected {COMPARISON_PROTOCOL.task_ids!r}, got {tuple(protocol.task_ids)!r}"
        )
    if tuple(protocol.seed_manifest) != tuple(COMPARISON_PROTOCOL.seed_manifest):
        raise ValueError(
            "Task 7 freezes seed_manifest to the comparison tier only: "
            + f"expected {COMPARISON_PROTOCOL.seed_manifest!r}, got {tuple(protocol.seed_manifest)!r}"
        )
    if int(protocol.num_trials_per_task) != int(
        COMPARISON_PROTOCOL.num_trials_per_task
    ):
        raise ValueError(
            "Task 7 freezes num_trials_per_task to the comparison tier only: "
            + f"expected {COMPARISON_PROTOCOL.num_trials_per_task!r}, got {protocol.num_trials_per_task!r}"
        )
    episode_count = (
        len(tuple(protocol.task_ids))
        * len(tuple(protocol.seed_manifest))
        * int(protocol.num_trials_per_task)
    )
    return FrozenComparisonManifest(
        suite=str(protocol.suite),
        task_ids=tuple(int(value) for value in protocol.task_ids),
        seed_manifest=tuple(int(value) for value in protocol.seed_manifest),
        num_trials_per_task=int(protocol.num_trials_per_task),
        episode_count=int(episode_count),
        evaluation_tier=str(protocol.evaluation_tier),
    )


def comparison_manifest_to_dict(
    manifest: FrozenComparisonManifest,
) -> dict[str, object]:
    payload = asdict(manifest)
    payload["task_ids"] = [int(value) for value in manifest.task_ids]
    payload["seed_manifest"] = [int(value) for value in manifest.seed_manifest]
    return payload


def repaired_variant_spec_to_dict(spec: RepairedVariantSpec) -> dict[str, object]:
    return asdict(spec)


def repaired_headline_comparison_to_dict(
    comparison: RepairedHeadlineComparison,
) -> dict[str, object]:
    return asdict(comparison)


def build_repaired_metric_profile() -> dict[str, object]:
    return {
        "metric_profile_id": REPAIRED_METRIC_PROFILE_ID,
        "primary_metric_id": REPAIRED_PRIMARY_METRIC_ORDER[0],
        "primary_metric_order": list(REPAIRED_PRIMARY_METRIC_ORDER),
        "audit_metrics": list(REPAIRED_AUDIT_METRICS),
        "compatibility_only_metric": "success_rate@1.00_budget",
        "throughput_clause": (
            "headline and gate comparisons must stay budget-aware and must not degrade "
            "to success-only reporting"
        ),
    }


def build_repaired_runtime_layer_specs() -> tuple[RepairedVariantSpec, ...]:
    return (
        RepairedVariantSpec(
            variant_id="C_existing_runtime_omit",
            layer_id=REPAIRED_RUNTIME_LAYER_ID,
            short_code="runtime_omit",
            label="C_existing + runtime_omit",
            source_kind="runtime_layer",
            status="existing_runtime_conditioning",
            comparison_role="runtime_probe",
            train_stage="recap_informative",
            train_consumer_mode="informative",
            prompt_text_surface="prompt_raw_only",
            per_sample_indicator_consumption=True,
            runtime_indicator_mode="omit",
            runtime_indicator_source="cli.indicator_mode",
        ),
        RepairedVariantSpec(
            variant_id="C_existing_runtime_positive",
            layer_id=REPAIRED_RUNTIME_LAYER_ID,
            short_code="runtime_positive",
            label="C_existing + runtime_positive",
            source_kind="runtime_layer",
            status="existing_runtime_conditioning",
            comparison_role="runtime_probe",
            train_stage="recap_informative",
            train_consumer_mode="informative",
            prompt_text_surface="canonical_text_indicator",
            per_sample_indicator_consumption=True,
            runtime_indicator_mode="positive",
            runtime_indicator_source="cli.indicator_mode",
        ),
        RepairedVariantSpec(
            variant_id="C_existing_runtime_negative",
            layer_id=REPAIRED_RUNTIME_LAYER_ID,
            short_code="runtime_negative",
            label="C_existing + runtime_negative",
            source_kind="runtime_layer",
            status="existing_runtime_conditioning",
            comparison_role="runtime_probe",
            train_stage="recap_informative",
            train_consumer_mode="informative",
            prompt_text_surface="canonical_text_indicator",
            per_sample_indicator_consumption=True,
            runtime_indicator_mode="negative",
            runtime_indicator_source="cli.indicator_mode",
        ),
        RepairedVariantSpec(
            variant_id="C_existing_runtime_cfg",
            layer_id=REPAIRED_RUNTIME_LAYER_ID,
            short_code="runtime_cfg",
            label="C_existing + runtime_cfg",
            source_kind="runtime_layer",
            status="existing_runtime_conditioning",
            comparison_role="runtime_probe",
            train_stage="recap_informative",
            train_consumer_mode="informative",
            prompt_text_surface="canonical_text_indicator",
            per_sample_indicator_consumption=True,
            runtime_indicator_mode="cfg",
            runtime_indicator_source="cfg.recap_runtime_default_positive",
        ),
    )


def build_repaired_current_backbone_specs() -> tuple[RepairedVariantSpec, ...]:
    return (
        RepairedVariantSpec(
            variant_id="A_stock_pi05_libero",
            layer_id=REPAIRED_CURRENT_BACKBONE_LAYER_ID,
            short_code="A",
            label="A_stock_pi05_libero",
            source_kind="stock_anchor",
            status="existing_stock_anchor",
            comparison_role="headline",
            prompt_text_surface="prompt_raw_only",
            per_sample_indicator_consumption=False,
            runtime_indicator_mode="omit",
            runtime_indicator_source="stock.prompt_only",
        ),
        RepairedVariantSpec(
            variant_id="B0_omit_control_v2",
            layer_id=REPAIRED_CURRENT_BACKBONE_LAYER_ID,
            short_code="B0",
            label="B0_omit_control_v2",
            source_kind="repaired_stage",
            status="semantics_frozen_pending_rollout",
            comparison_role="headline",
            train_stage="omit_control",
            train_consumer_mode="omit",
            fixed_indicator_mode="omit",
            prompt_text_surface="prompt_raw_only",
            per_sample_indicator_consumption=False,
            runtime_indicator_mode="omit",
            runtime_indicator_source="stage.omit_control",
        ),
        RepairedVariantSpec(
            variant_id="B1_fixed_positive_sft_v2",
            layer_id=REPAIRED_CURRENT_BACKBONE_LAYER_ID,
            short_code="B1",
            label="B1_fixed_positive_sft_v2",
            source_kind="repaired_stage",
            status="checkpoint_present",
            comparison_role="headline",
            train_stage="sft_fixed_positive",
            train_consumer_mode="fixed_positive",
            fixed_indicator_mode="positive",
            prompt_text_surface="canonical_text_indicator",
            per_sample_indicator_consumption=False,
            runtime_indicator_mode="positive",
            runtime_indicator_source="stage.fixed_positive",
        ),
        RepairedVariantSpec(
            variant_id="X_shuffled_indicator_v2",
            layer_id=REPAIRED_CURRENT_BACKBONE_LAYER_ID,
            short_code="X",
            label="X_shuffled_indicator_v2",
            source_kind="repaired_stage",
            status="semantics_frozen_pending_rollout",
            comparison_role="diagnostic",
            train_stage="shuffled_indicator",
            train_consumer_mode="shuffled",
            prompt_text_surface="canonical_text_indicator",
            per_sample_indicator_consumption=True,
            runtime_indicator_source="deterministic_shuffled_sample_key",
        ),
        RepairedVariantSpec(
            variant_id="C0_recap_informative_positiveinfer_v2",
            layer_id=REPAIRED_CURRENT_BACKBONE_LAYER_ID,
            short_code="C0",
            label="C0_recap_informative_positiveinfer_v2",
            source_kind="repaired_stage_runtime",
            status="checkpoint_present",
            comparison_role="headline",
            train_stage="recap_informative",
            train_consumer_mode="informative",
            prompt_text_surface="canonical_text_indicator",
            per_sample_indicator_consumption=True,
            runtime_indicator_mode="positive",
            runtime_indicator_source="cli.indicator_mode",
            derived_from="C_existing_runtime_positive",
        ),
        RepairedVariantSpec(
            variant_id="C1_recap_informative_cfg_v2",
            layer_id=REPAIRED_CURRENT_BACKBONE_LAYER_ID,
            short_code="C1",
            label="C1_recap_informative_cfg_v2",
            source_kind="repaired_stage_runtime",
            status="runtime_alias_on_existing_checkpoint",
            comparison_role="headline",
            train_stage="recap_informative",
            train_consumer_mode="informative",
            prompt_text_surface="canonical_text_indicator",
            per_sample_indicator_consumption=True,
            runtime_indicator_mode="cfg",
            runtime_indicator_source="cfg.recap_runtime_default_positive",
            derived_from="C_existing_runtime_cfg",
        ),
    )


def build_repaired_paper_full_specs() -> tuple[RepairedVariantSpec, ...]:
    return tuple(
        RepairedVariantSpec(
            variant_id=variant_id,
            layer_id=REPAIRED_PAPER_FULL_LAYER_ID,
            short_code=variant_id,
            label=f"{variant_id}_future_paper_full",
            source_kind="paper_full_future",
            status="future_only_not_executed",
            comparison_role="future_only",
        )
        for variant_id in PAPER_FULL_VARIANT_IDS
    )


def build_repaired_variant_specs() -> tuple[RepairedVariantSpec, ...]:
    return (
        *build_repaired_runtime_layer_specs(),
        *build_repaired_current_backbone_specs(),
        *build_repaired_paper_full_specs(),
    )


def build_repaired_headline_comparisons() -> tuple[RepairedHeadlineComparison, ...]:
    return (
        RepairedHeadlineComparison(
            comparison_id="C0_vs_B1",
            lhs_variant_id="C0_recap_informative_positiveinfer_v2",
            rhs_variant_id="B1_fixed_positive_sft_v2",
            relation=">=",
            gate="G3",
            purpose="recap gate, informative positive inference must not trail fixed-positive SFT",
        ),
        RepairedHeadlineComparison(
            comparison_id="C0_vs_X",
            lhs_variant_id="C0_recap_informative_positiveinfer_v2",
            rhs_variant_id="X_shuffled_indicator_v2",
            relation=">",
            gate="G2",
            purpose="informativeness gate, real indicator must beat shuffled diagnostic",
        ),
        RepairedHeadlineComparison(
            comparison_id="B1_vs_B0",
            lhs_variant_id="B1_fixed_positive_sft_v2",
            rhs_variant_id="B0_omit_control_v2",
            relation="!=",
            gate="G1",
            purpose="control semantics gate, fixed-positive SFT must stay distinct from omit control",
        ),
        RepairedHeadlineComparison(
            comparison_id="C1_vs_C0",
            lhs_variant_id="C1_recap_informative_cfg_v2",
            rhs_variant_id="C0_recap_informative_positiveinfer_v2",
            relation=">=",
            gate="G4",
            purpose="runtime parity gate, cfg runtime must stay comparable to explicit positive inference",
        ),
    )


def build_repaired_future_comparisons() -> tuple[RepairedHeadlineComparison, ...]:
    return (
        RepairedHeadlineComparison(
            comparison_id="P3_vs_P2",
            lhs_variant_id="P3",
            rhs_variant_id="P2",
            relation=">",
            gate="G7",
            purpose="paper-full iteration value gate kept separate from repaired backbone layer",
        ),
    )


def build_repaired_matrix_layers() -> tuple[dict[str, object], ...]:
    return (
        {
            "layer_id": REPAIRED_RUNTIME_LAYER_ID,
            "label": REPAIRED_RUNTIME_LAYER_LABEL,
            "variant_ids": list(RUNTIME_LAYER_VARIANT_IDS),
        },
        {
            "layer_id": REPAIRED_CURRENT_BACKBONE_LAYER_ID,
            "label": REPAIRED_CURRENT_BACKBONE_LAYER_LABEL,
            "variant_ids": list(CURRENT_BACKBONE_VARIANT_IDS),
        },
        {
            "layer_id": REPAIRED_PAPER_FULL_LAYER_ID,
            "label": REPAIRED_PAPER_FULL_LAYER_LABEL,
            "variant_ids": list(PAPER_FULL_VARIANT_IDS),
        },
    )


def build_repaired_variant_catalog() -> dict[str, dict[str, object]]:
    return {
        spec.variant_id: repaired_variant_spec_to_dict(spec)
        for spec in build_repaired_variant_specs()
    }


def build_repaired_matrix_protocol() -> dict[str, object]:
    return {
        "schema_version": REPAIRED_MATRIX_SUMMARY_SCHEMA_VERSION,
        "metric_profile": build_repaired_metric_profile(),
        "layers": list(build_repaired_matrix_layers()),
        "variant_catalog": build_repaired_variant_catalog(),
        "headline_comparisons": [
            repaired_headline_comparison_to_dict(comparison)
            for comparison in build_repaired_headline_comparisons()
        ],
        "future_comparisons": [
            repaired_headline_comparison_to_dict(comparison)
            for comparison in build_repaired_future_comparisons()
        ],
        "legacy_surface_policy": {
            "mix_old_and_new_semantics": False,
            "legacy_abcx_allowed_only_as_inherited_sidecar": True,
            "paper_full_layer_separate_from_repaired_backbone": True,
        },
    }


def sanitize_run_component(raw: str) -> str:
    text = str(raw).strip()
    if not text:
        return "run"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-") or "run"


def build_train_runtime_dir(output_dir: str | Path, *, variant: str) -> Path:
    output_name = sanitize_run_component(Path(output_dir).resolve().name)
    runtime_root_override = str(
        os.environ.get("OPENPI_RECAP_RUNTIME_ROOT", "")
    ).strip()
    runtime_root = (
        Path(runtime_root_override).expanduser().resolve()
        if runtime_root_override
        else REPO_ROOT / "agent" / "runtime_logs" / "openpi_libero_recap"
    )
    return runtime_root / f"{variant}_{output_name}_train"


def build_eval_output_paths(
    checkpoint_dir: str | Path, *, variant: str
) -> dict[str, Path]:
    checkpoint_name = sanitize_run_component(Path(checkpoint_dir).resolve().name)
    artifact_dir = (
        REPO_ROOT
        / "agent"
        / "artifacts"
        / "openpi_libero_recap_eval"
        / f"{variant}_{checkpoint_name}"
    )
    runtime_dir = (
        REPO_ROOT
        / "agent"
        / "runtime_logs"
        / "openpi_libero_recap_eval"
        / f"{variant}_{checkpoint_name}"
    )
    return {
        "artifact_dir": artifact_dir,
        "runtime_dir": runtime_dir,
        "summary_json": artifact_dir / "summary.json",
        "paired_summary_json": artifact_dir / "paired_summary.json",
        "log_path": runtime_dir / "eval.log",
    }


def build_stock_origin_payload() -> dict[str, str]:
    return {
        "config": EXPECTED_CONFIG_NAME,
        "checkpoint": EXPECTED_CHECKPOINT,
        "checkpoint_source": EXPECTED_CHECKPOINT_SOURCE,
        "stock_summary_path": str(STOCK_SUMMARY_PATH),
    }


__all__ = [
    "CHECKPOINT_PROVENANCE_SCHEMA_VERSION",
    "COMPARISON_PROTOCOL",
    "FrozenComparisonManifest",
    "CURRENT_BACKBONE_VARIANT_IDS",
    "PAIRED_SUMMARY_SCHEMA_VERSION",
    "PAPER_FULL_VARIANT_IDS",
    "REPAIRED_ALL_METRICS",
    "REPAIRED_AUDIT_METRICS",
    "REPAIRED_CURRENT_BACKBONE_LAYER_ID",
    "REPAIRED_CURRENT_BACKBONE_LAYER_LABEL",
    "REPAIRED_MATRIX_SUMMARY_SCHEMA_VERSION",
    "REPAIRED_METRIC_PROFILE_ID",
    "REPAIRED_PAPER_FULL_LAYER_ID",
    "REPAIRED_PAPER_FULL_LAYER_LABEL",
    "REPAIRED_PRIMARY_METRIC_ORDER",
    "REPAIRED_RUNTIME_LAYER_ID",
    "REPAIRED_RUNTIME_LAYER_LABEL",
    "RECAP_ONLY_VARIANT",
    "RECAP_RECORD_SCHEMA_VERSION",
    "REPO_ROOT",
    "RUNTIME_LAYER_VARIANT_IDS",
    "RepairedHeadlineComparison",
    "RepairedVariantSpec",
    "STOCK_SUMMARY_PATH",
    "STOCK_VARIANT",
    "SUMMARY_SCHEMA_VERSION",
    "SUPPORTED_EVAL_VARIANTS",
    "TRAIN_MANIFEST_SCHEMA_VERSION",
    "build_repaired_current_backbone_specs",
    "build_eval_output_paths",
    "build_frozen_comparison_manifest",
    "build_repaired_future_comparisons",
    "build_repaired_headline_comparisons",
    "build_repaired_matrix_layers",
    "build_repaired_matrix_protocol",
    "build_repaired_metric_profile",
    "build_repaired_paper_full_specs",
    "build_repaired_runtime_layer_specs",
    "build_repaired_variant_catalog",
    "build_repaired_variant_specs",
    "build_stock_origin_payload",
    "build_train_runtime_dir",
    "comparison_manifest_to_dict",
    "repaired_headline_comparison_to_dict",
    "repaired_variant_spec_to_dict",
    "sanitize_run_component",
    "validate_eval_variant",
    "validate_recap_only_variant",
]
