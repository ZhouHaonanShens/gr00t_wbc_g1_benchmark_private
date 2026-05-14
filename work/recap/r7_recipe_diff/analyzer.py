from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from work.recap.r2_authentic_eval.exclusion import EXCLUDED_CELL_ID, EVIDENCE_GRADE_CELL_IDS
from work.recap.r7_recipe_diff.contract import FIDELITY_COMPONENTS, R7DiffError, RecipeDelta

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENPI_REPORT_PATH = Path("agent/exchange/openpi_recap_fidelity_fact_report_v1.md")
LAUNCHER_PATH = Path("work/recap/launch_finetune_use_ddp.py")
FINETUNE_CONFIG_PATH = Path("submodules/Isaac-GR00T/gr00t/configs/finetune_config.py")
DUAL_LOSS_PATH = Path("work/recap/dual_loss.py")
TEXT_INDICATOR_PATH = Path("work/recap/text_indicator.py")
NUMERIC_SMOKE_PATH = Path("work/recap/scripts/34b_recap_numeric_adv_smoke.py")
ADVANTAGE_PATH = Path("work/recap/advantage.py")
DATASET_EXPORT_PATH = Path("work/recap/lerobot_export/dataset_export.py")
R6_RUNTIME_REPORT_PATH = Path("agent/exchange/r6_1_runtime_probe_a2_20260513.md")


def _read_text(relative_path: Path) -> str:
    path = REPO_ROOT / relative_path
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _read_openpi_report() -> str:
    path = REPO_ROOT / OPENPI_REPORT_PATH
    if not path.is_file():
        raise R7DiffError(f"missing required OpenPI fidelity report: {OPENPI_REPORT_PATH}")
    return path.read_text(encoding="utf-8")


def _validate_base_cell(base_cell: str) -> str:
    normalized_cell = str(base_cell).strip()
    if normalized_cell == EXCLUDED_CELL_ID:
        raise R7DiffError(f"base-cell {EXCLUDED_CELL_ID} is excluded by R2 SSOT")
    if normalized_cell not in EVIDENCE_GRADE_CELL_IDS:
        allowed = ", ".join(EVIDENCE_GRADE_CELL_IDS)
        raise R7DiffError(f"base-cell must be one of {allowed}; got {base_cell!r}")
    return normalized_cell


def _contains(relative_path: Path, needle: str) -> bool:
    return needle in _read_text(relative_path)


def _mentions(openpi_report_text: str, *needles: str) -> bool:
    lowered = openpi_report_text.lower()
    return all(needle.lower() in lowered for needle in needles)


def _component(component_id: str):
    return next(item for item in FIDELITY_COMPONENTS if item.component_id == component_id)


def _delta(component_id: str, state: str, action: str, args: tuple[str, ...], diffs: tuple[tuple[str, str], ...], evidence: tuple[Path, ...], rationale: str) -> RecipeDelta:
    return RecipeDelta(_component(component_id), state, "IMPLEMENTED", action, args, diffs, tuple(str(path) for path in evidence), rationale)  # type: ignore[arg-type]


def analyze_c1_dual_loss(base_cell: str, openpi_report_text: str) -> RecipeDelta:
    _validate_base_cell(base_cell)
    report_requires = _mentions(openpi_report_text, "Q6", "conditional", "unconditional")
    helper_present = _contains(DUAL_LOSS_PATH, "combine_alpha_dual_loss")
    launcher_flag = _contains(LAUNCHER_PATH, "enable_dual_loss")
    config_flag = _contains(FINETUNE_CONFIG_PATH, "enable_dual_loss")
    active_path = launcher_flag and config_flag
    current_state = "IMPLEMENTED" if active_path and report_requires else "ABSENT"
    if helper_present and not active_path:
        current_state = "ABSENT"
    rationale = "dual-loss helper exists but launcher/config do not expose active conditional/unconditional training."
    if current_state == "IMPLEMENTED":
        rationale = "dual-loss flag appears in launcher and config; verify runtime wiring before use."
    return _delta(
        "C1_dual_loss", current_state, "ADD_LOSS_TERM",
        ("--enable-dual-loss", "--dual-loss-alpha=0.5"),
        (("model.dual_loss.enabled", "true"), ("training.dual_loss.alpha", "0.5")),
        (OPENPI_REPORT_PATH, DUAL_LOSS_PATH, LAUNCHER_PATH, FINETUNE_CONFIG_PATH), rationale,
    )


def analyze_c2_indicator_dropout(base_cell: str, openpi_report_text: str) -> RecipeDelta:
    _validate_base_cell(base_cell)
    report_requires = _mentions(openpi_report_text, "Q7", "dropout")
    helper_present = _contains(TEXT_INDICATOR_PATH, "indicator_dropout_p")
    smoke_flag = _contains(NUMERIC_SMOKE_PATH, "indicator_dropout_p")
    launcher_flag = _contains(LAUNCHER_PATH, "indicator_dropout_p")
    config_field = _contains(FINETUNE_CONFIG_PATH, "indicator_dropout_p")
    active_path = launcher_flag and config_field
    current_state = "IMPLEMENTED" if active_path and report_requires else "ABSENT"
    if (helper_present or smoke_flag) and not active_path:
        current_state = "ABSENT"
    rationale = "dropout helpers or smoke flags exist, but stochastic omission is absent from active training config."
    if current_state == "IMPLEMENTED":
        rationale = "indicator dropout appears connected to launcher/config active training inputs."
    return _delta(
        "C2_indicator_dropout", current_state, "ADD_DATASET_AUG",
        ("--indicator-dropout-p=0.15", "--indicator-dropout-seed=0"),
        (("training.indicator_dropout.p", "0.15"), ("training.indicator_dropout.seed", "0")),
        (OPENPI_REPORT_PATH, TEXT_INDICATOR_PATH, NUMERIC_SMOKE_PATH, LAUNCHER_PATH), rationale,
    )


def analyze_c3_learned_value(base_cell: str, openpi_report_text: str) -> RecipeDelta:
    _validate_base_cell(base_cell)
    report_requires = _mentions(openpi_report_text, "Q2", "value", "critic")
    static_labels = _contains(ADVANTAGE_PATH, "advantage_input")
    launcher_flag = _contains(LAUNCHER_PATH, "enable_learned_value_head")
    config_flag = _contains(FINETUNE_CONFIG_PATH, "enable_learned_value_head")
    active_path = launcher_flag and config_flag
    current_state = "IMPLEMENTED" if active_path and report_requires else "ABSENT"
    if static_labels and not active_path:
        current_state = "ABSENT"
    rationale = "static value/advantage labels exist, but no learned value head is active in training."
    if current_state == "IMPLEMENTED":
        rationale = "learned value head appears in launcher and finetune config."
    return _delta(
        "C3_learned_value", current_state, "ADD_LOSS_TERM",
        ("--enable-learned-value-head", "--value-loss-alpha=0.1"),
        (("model.value_head.enabled", "true"), ("training.value_loss.alpha", "0.1")),
        (OPENPI_REPORT_PATH, ADVANTAGE_PATH, LAUNCHER_PATH, FINETUNE_CONFIG_PATH), rationale,
    )


def analyze_c4_advantage_embedding_active(base_cell: str, openpi_report_text: str) -> RecipeDelta:
    _validate_base_cell(base_cell)
    report_mentions = _mentions(openpi_report_text, "Q3", "advantage")
    sidecar_present = _contains(ADVANTAGE_PATH, "ADVANTAGE_INPUT_COLUMN")
    export_sidecar = _contains(DATASET_EXPORT_PATH, "advantage_input")
    launcher_flag = _contains(LAUNCHER_PATH, "action_head_advantage_input")
    action_prefix = _contains(LAUNCHER_PATH, "advantage_embedding")
    active_path = launcher_flag and action_prefix
    current_state = "IMPLEMENTED" if active_path and report_mentions else "PARTIAL"
    if not (sidecar_present or export_sidecar or active_path):
        current_state = "ABSENT"
    rationale = "advantage_input sidecar exists in data paths, but active action-head consumption is not wired."
    if current_state == "IMPLEMENTED":
        rationale = "advantage sidecar appears wired into an action-head training flag."
    return _delta(
        "C4_advantage_embedding_active", current_state, "ENABLE_FLAG",
        ("--action-head-advantage-input=enabled", "--advantage-embedding-dim=16"),
        (("model.action_head.advantage_input.enabled", "true"), ("model.action_head.advantage_embedding_dim", "16")),
        (OPENPI_REPORT_PATH, ADVANTAGE_PATH, DATASET_EXPORT_PATH, LAUNCHER_PATH), rationale,
    )


def analyze_c5_carrier_text_v1_grad_path(base_cell: str, openpi_report_text: str) -> RecipeDelta:
    _validate_base_cell(base_cell)
    report_mentions = _mentions(openpi_report_text, "Q8", "runtime")
    carrier_builder = _contains(TEXT_INDICATOR_PATH, "carrier_text_v1")
    exporter_carrier = _contains(DATASET_EXPORT_PATH, "carrier_text_v1")
    carrier_flag = _contains(LAUNCHER_PATH, "dual_loss_uses_carrier_text")
    invariant_evidence = _contains(R6_RUNTIME_REPORT_PATH, "INDICATOR_INVARIANT")
    gradient_path = carrier_flag and not invariant_evidence
    current_state = "IMPLEMENTED" if gradient_path and report_mentions else "PARTIAL"
    if not (carrier_builder or exporter_carrier or invariant_evidence or gradient_path):
        current_state = "ABSENT"
    rationale = "carrier text reaches token surfaces, but R6/open-loop evidence does not prove gradient-sensitive action conditioning."
    if current_state == "IMPLEMENTED":
        rationale = "dual-loss carrier flag appears and no invariant R6 evidence was found."
    return _delta(
        "C5_carrier_text_v1_grad_path", current_state, "ADD_CLI_ARG",
        ("--dual-loss-uses-carrier-text", "--carrier-text-field=carrier_text_v1"),
        (("training.dual_loss.uses_carrier_text", "true"), ("data.task_text_field", "carrier_text_v1")),
        (OPENPI_REPORT_PATH, TEXT_INDICATOR_PATH, DATASET_EXPORT_PATH, R6_RUNTIME_REPORT_PATH), rationale,
    )


_ANALYZERS: tuple[Callable[[str, str], RecipeDelta], ...] = (
    analyze_c1_dual_loss, analyze_c2_indicator_dropout, analyze_c3_learned_value,
    analyze_c4_advantage_embedding_active, analyze_c5_carrier_text_v1_grad_path,
)


def analyze_all(base_cell: str) -> tuple[RecipeDelta, ...]:
    normalized_cell = _validate_base_cell(base_cell)
    openpi_report_text = _read_openpi_report()
    return tuple(analyzer(normalized_cell, openpi_report_text) for analyzer in _ANALYZERS)
