"""R7.0 training recipe diff — five fidelity components SSOT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ComponentState = Literal["IMPLEMENTED", "PARTIAL", "ABSENT"]
DiffAction = Literal["ENABLE_FLAG", "ADD_CLI_ARG", "ADD_LOSS_TERM", "ADD_DATASET_AUG", "NO_OP"]


@dataclass(frozen=True)
class FidelityComponent:
    component_id: str
    title: str
    paper_section_cite: str
    training_arg_name: str


@dataclass(frozen=True)
class RecipeDelta:
    component: FidelityComponent
    current_state: ComponentState
    paper_prescribed_state: ComponentState
    diff_action: DiffAction
    cli_arg_addition: tuple[str, ...]
    config_path_diff: tuple[tuple[str, str], ...]
    evidence_files: tuple[str, ...]
    rationale: str


class R7DiffError(RuntimeError):
    """Raised on invalid R7 inputs or audit preconditions."""


FIDELITY_COMPONENTS: tuple[FidelityComponent, ...] = (
    FidelityComponent(
        "C1_dual_loss", "Dual conditional/unconditional loss objective", "OpenPI fidelity Q6", "enable_dual_loss"
    ),
    FidelityComponent(
        "C2_indicator_dropout", "Stochastic indicator omission/dropout at training time", "OpenPI fidelity Q7", "indicator_dropout_p"
    ),
    FidelityComponent(
        "C3_learned_value", "Learned value-function / critic in training loop", "OpenPI fidelity Q2", "enable_learned_value_head"
    ),
    FidelityComponent(
        "C4_advantage_embedding_active", "advantage_input numeric sidecar consumed by action head", "OpenPI fidelity Q3", "action_head_advantage_input"
    ),
    FidelityComponent(
        "C5_carrier_text_v1_grad_path",
        "carrier_text_v1 tokens get gradient signal through dual-loss path",
        "OpenPI fidelity Q8 / GR00T R6.1 verdict",
        "dual_loss_uses_carrier_text",
    ),
)
