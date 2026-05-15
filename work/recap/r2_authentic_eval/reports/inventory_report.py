"""Markdown table renderer for R2.0 inventory results."""
from __future__ import annotations

from work.recap.r2_authentic_eval.inventory import TrainedCheckpoint


def render(inventory: list[TrainedCheckpoint]) -> str:
    """Return a markdown table summarising the R2.0 inventory.

    Columns: label | abs_path | training_algo | n_train_steps |
             formalize_language | stats_match | is_valid | invalid_reason

    Label column uses 'RECAP (negative-token rule)' annotation per A2:
    classification is by negative-token blocklist, NOT positive-token match.
    """
    header = (
        "| label | abs_path | training_algo | n_train_steps"
        " | formalize_language | stats_match | is_valid | invalid_reason |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for c in inventory:
        label_display = (
            f"{c.label} (negative-token rule)" if c.label == "RECAP" else c.label
        )
        lines.append(
            f"| {label_display} | {c.abs_path} | {c.training_algo}"
            f" | {c.n_train_steps} | {c.formalize_language}"
            f" | {c.statistics_q99_matches_base} | {c.is_valid}"
            f" | {c.invalid_reason} |"
        )
    return "\n".join(lines) + "\n"
