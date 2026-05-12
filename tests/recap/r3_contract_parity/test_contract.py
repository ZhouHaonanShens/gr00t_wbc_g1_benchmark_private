from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import work.recap.r3_contract_parity as r3
from work.recap.r2_authentic_eval import exclusion
from work.recap.r3_contract_parity.contract import ALLOWED_PATTERN_IDS, PARITY_AXES, ParityAxisSpec, R3AuditError


def test_closed_six_axis_contract_and_r2_cells() -> None:
    assert exclusion.EVIDENCE_GRADE_CELL_IDS == ("A.2", "A.3", "A.4", "A.5")
    assert len(PARITY_AXES) == 6
    assert tuple(axis.axis_id for axis in PARITY_AXES) == (
        "checkpoint_binding",
        "config_json_sha256",
        "processor_config_json_sha256",
        "statistics_json_sha256",
        "training_algo",
        "formalize_language",
    )
    assert {axis.pattern_id for axis in PARITY_AXES} <= set(ALLOWED_PATTERN_IDS)


def test_frozen_dataclass_public_api_and_exception() -> None:
    axis = PARITY_AXES[0]
    with pytest.raises(FrozenInstanceError):
        axis.axis_id = "mutated"  # type: ignore[misc]
    assert isinstance(axis, ParityAxisSpec)
    assert issubclass(R3AuditError, RuntimeError)
    assert r3.__all__ == ("PARITY_AXES", "ParityAxisSpec", "ParityCellReport", "audit_cell")
