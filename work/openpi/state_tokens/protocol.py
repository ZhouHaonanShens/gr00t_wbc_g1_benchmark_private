from __future__ import annotations

from pathlib import Path

from work.openpi.recap.protocol import REPO_ROOT, sanitize_run_component


RECAP_STATE_TOKENS_VARIANT = "recap_state_tokens"
STATE_TOKEN_ROUTE = "native_discrete_state_input_v1"
NOT_APPLICABLE_STATE_TOKEN_ROUTE = "not_applicable"
SOURCE_STATE = "normalized raw 8D LIBERO observation/state"
SOURCE_STATE_PADDING = "not padded 32D internals"
STATE_TOKEN_SEMANTICS = "normalized 8D state -> native tokenizer -> discrete tokens"
TRANSFORM_ORDER = "normalize first, tokenize second"
REQUIRED_NATIVE_STATE_DIM = 8
OFFICIAL_NATIVE_DATASET_NAME = "physical_intelligence_libero_official_8d"
OFFICIAL_NATIVE_DATASET_DIR = (
    REPO_ROOT
    / "agent"
    / "artifacts"
    / "lerobot_datasets"
    / OFFICIAL_NATIVE_DATASET_NAME
)
OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME = (
    "physical_intelligence_libero_official_8d_recap_relabels_v1"
)
OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_DIR = (
    REPO_ROOT
    / "agent"
    / "artifacts"
    / "lerobot_datasets"
    / OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME
)
OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID = "official_native_8d_recap_relabels_v1"

BLOCKER_REPORT_SCHEMA_VERSION = "openpi_libero_state_tokens_blocker_v1"
BLOCKER_CODE_MISSING_NATIVE_8D_STATE = "missing_native_libero_8d_state_source"
BLOCKER_CODE_INVALID_NATIVE_PROVENANCE = "invalid_native_state_token_provenance"
BLOCKER_CODE_MISSING_SAFE_RECAP_LABEL_JOIN = (
    "missing_safe_recap_label_join_onto_official_8d"
)
BLOCKER_CODE_INVALID_TRAINING_SOURCE = "invalid_state_token_training_source"
BLOCKER_CODE_MISSING_CONTROL_PARITY_ARTIFACT = (
    "missing_recap_only_control_parity_artifact"
)
BLOCKER_CODE_CONTROL_PARITY_NOT_SATISFIED = (
    "control_parity_not_satisfied_for_state_token_ablation"
)
BLOCKER_CODE_INVALID_CONTROL_PARITY_REFERENCE = (
    "invalid_recap_only_control_parity_reference"
)

TRAIN_MANIFEST_SCHEMA_VERSION = "openpi_libero_state_tokens_train_manifest_v1"
CHECKPOINT_PROVENANCE_SCHEMA_VERSION = "openpi_libero_state_tokens_checkpoint_v1"
SUMMARY_SCHEMA_VERSION = "openpi_libero_state_tokens_summary_v1"
PAIRED_SUMMARY_SCHEMA_VERSION = "openpi_libero_state_tokens_paired_summary_v1"

DEFAULT_RECAP_ONLY_CHECKPOINT_DIR = (
    REPO_ROOT
    / "agent"
    / "artifacts"
    / "checkpoints"
    / "openpi_libero_variants"
    / "recap_only_relabel8d_v1"
    / "best"
)
DEFAULT_RECAP_ONLY_CONTROL_GATE_REPORT = (
    REPO_ROOT
    / "agent"
    / "artifacts"
    / "checkpoints"
    / "openpi_libero_variants"
    / "recap_only_relabel8d_v1"
    / "source_equivalence_report.json"
)


class StateTokenContractError(RuntimeError):
    payload: dict[str, object]

    def __init__(self, message: str, *, payload: dict[str, object]):
        super().__init__(message)
        self.payload = dict(payload)


def build_blocker_report(
    *,
    stage: str,
    blocker_code: str,
    reason: str,
    source_dataset_dir: str | Path | None = None,
    observed_dataset_state_dim: int | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_provenance_path: str | Path | None = None,
    next_action: str | None = None,
    extra_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": BLOCKER_REPORT_SCHEMA_VERSION,
        "variant": RECAP_STATE_TOKENS_VARIANT,
        "status": "blocked",
        "stage": str(stage),
        "blocker_code": str(blocker_code),
        "attempted_state_token_route": STATE_TOKEN_ROUTE,
        "required_source_state": SOURCE_STATE,
        "required_transform_order": TRANSFORM_ORDER,
        "required_native_state_dim": REQUIRED_NATIVE_STATE_DIM,
        "reason": str(reason),
        "next_action": (
            "Provide the missing precondition required to unblock Task 9 before retrying."
            if next_action is None
            else str(next_action)
        ),
    }
    if source_dataset_dir is not None:
        payload["source_dataset_dir"] = str(Path(source_dataset_dir).resolve())
    if observed_dataset_state_dim is not None:
        payload["observed_dataset_state_dim"] = int(observed_dataset_state_dim)
    if checkpoint_dir is not None:
        payload["checkpoint_dir"] = str(Path(checkpoint_dir).resolve())
    if checkpoint_provenance_path is not None:
        payload["checkpoint_provenance_path"] = str(
            Path(checkpoint_provenance_path).resolve()
        )
    if extra_payload is not None:
        payload.update(dict(extra_payload))
    return payload


def validate_state_token_variant(variant: str) -> str:
    value = str(variant).strip()
    if value != RECAP_STATE_TOKENS_VARIANT:
        raise ValueError(
            "Task 9 only supports --variant "
            + f"{RECAP_STATE_TOKENS_VARIANT!r} for the state-token branch, got {variant!r}"
        )
    return value


def build_train_runtime_dir(output_dir: str | Path, *, variant: str) -> Path:
    output_name = sanitize_run_component(Path(output_dir).resolve().name)
    return (
        REPO_ROOT
        / "agent"
        / "runtime_logs"
        / "openpi_libero_recap"
        / f"{variant}_{output_name}_train"
    )


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


__all__ = [
    "BLOCKER_CODE_CONTROL_PARITY_NOT_SATISFIED",
    "BLOCKER_CODE_INVALID_CONTROL_PARITY_REFERENCE",
    "BLOCKER_CODE_INVALID_TRAINING_SOURCE",
    "BLOCKER_CODE_INVALID_NATIVE_PROVENANCE",
    "BLOCKER_CODE_MISSING_CONTROL_PARITY_ARTIFACT",
    "BLOCKER_CODE_MISSING_SAFE_RECAP_LABEL_JOIN",
    "BLOCKER_CODE_MISSING_NATIVE_8D_STATE",
    "BLOCKER_REPORT_SCHEMA_VERSION",
    "CHECKPOINT_PROVENANCE_SCHEMA_VERSION",
    "DEFAULT_RECAP_ONLY_CHECKPOINT_DIR",
    "DEFAULT_RECAP_ONLY_CONTROL_GATE_REPORT",
    "NOT_APPLICABLE_STATE_TOKEN_ROUTE",
    "OFFICIAL_NATIVE_DATASET_DIR",
    "OFFICIAL_NATIVE_DATASET_NAME",
    "OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_DIR",
    "OFFICIAL_NATIVE_RECAP_RELABEL_DATASET_NAME",
    "OFFICIAL_NATIVE_RECAP_RELABEL_ROUTE_ID",
    "PAIRED_SUMMARY_SCHEMA_VERSION",
    "REQUIRED_NATIVE_STATE_DIM",
    "RECAP_STATE_TOKENS_VARIANT",
    "SOURCE_STATE",
    "SOURCE_STATE_PADDING",
    "StateTokenContractError",
    "STATE_TOKEN_ROUTE",
    "STATE_TOKEN_SEMANTICS",
    "SUMMARY_SCHEMA_VERSION",
    "TRAIN_MANIFEST_SCHEMA_VERSION",
    "TRANSFORM_ORDER",
    "build_blocker_report",
    "build_eval_output_paths",
    "build_train_runtime_dir",
    "validate_state_token_variant",
]
