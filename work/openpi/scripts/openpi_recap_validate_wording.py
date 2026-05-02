from __future__ import annotations

"""Compatibility shim for the legacy wording-validation import path."""

from agent.archive.openpi.legacy_results.openpi_recap_freeze_confirmation_contract import (
    DEFAULT_WORDING_CONTRACT_DOC,
)
from agent.archive.openpi.legacy_results.openpi_recap_validate_wording import (
    DEFAULT_RESULTS_DOC,
    load_wording_contract,
    validate_wording,
)

__all__ = [
    "DEFAULT_RESULTS_DOC",
    "DEFAULT_WORDING_CONTRACT_DOC",
    "load_wording_contract",
    "validate_wording",
]
