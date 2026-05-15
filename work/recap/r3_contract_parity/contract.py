from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Verdict = str
PatternId = str
_MISSING = object()
PASS: Verdict = "PASS"
WARN: Verdict = "WARN"
FAIL_HOT: Verdict = "FAIL_HOT"
NONE_PATTERN: PatternId = "none"
OPENPI_CKPT_BINDING: PatternId = "openpi_checkpoint_binding"
OPENPI_CONFIG_HASH: PatternId = "openpi_config_hash"
OPENPI_PROCESSOR_HASH: PatternId = "openpi_processor_hash"
OPENPI_STATS_NORM: PatternId = "openpi_statistics_norm"
OPENPI_MODEL_IDENTITY: PatternId = "openpi_model_identity"
OPENPI_LANGUAGE_FLAG: PatternId = "openpi_language_flag"
ALLOWED_VERDICTS: tuple[Verdict, ...] = (PASS, WARN, FAIL_HOT)
ALLOWED_PATTERN_IDS: tuple[PatternId, ...] = (
    NONE_PATTERN,
    OPENPI_CKPT_BINDING,
    OPENPI_CONFIG_HASH,
    OPENPI_PROCESSOR_HASH,
    OPENPI_STATS_NORM,
    OPENPI_MODEL_IDENTITY,
    OPENPI_LANGUAGE_FLAG,
)


class R3AuditError(RuntimeError):
    """Raised when R3 static audit inputs or CLI arguments are invalid."""


@dataclass(frozen=True)
class ParityAxisSpec:
    axis_id: str
    train_path: tuple[str, ...]
    eval_path: tuple[str, ...]
    pattern_id: PatternId
    description: str


@dataclass(frozen=True)
class ParityAxisResult:
    axis: ParityAxisSpec
    train_value: Any
    eval_value: Any
    verdict: Verdict
    pattern_id: PatternId
    note: str


@dataclass(frozen=True)
class ParityCellReport:
    cell_id: str
    ckpt_abs_path: str
    verdict: Verdict
    axes: tuple[ParityAxisResult, ...]
    pattern_hits: tuple[PatternId, ...]
    manifest_path: str | None = None
    report_path: str | None = None


PARITY_AXES: tuple[ParityAxisSpec, ...] = (
    ParityAxisSpec(
        "checkpoint_binding",
        ("checkpoint", "abs_path"),
        ("eval", "checkpoint"),
        OPENPI_CKPT_BINDING,
        "Eval must load the exact checkpoint resolved for the R2 evidence-grade cell.",
    ),
    ParityAxisSpec(
        "config_json_sha256",
        ("checkpoint", "config_json_sha256"),
        ("request", "checkpoint", "config_json_sha256"),
        OPENPI_CONFIG_HASH,
        "Eval manifest must cite the same config.json bytes that the checkpoint contains.",
    ),
    ParityAxisSpec(
        "processor_config_json_sha256",
        ("checkpoint", "processor_config_json_sha256"),
        ("request", "checkpoint", "processor_config_json_sha256"),
        OPENPI_PROCESSOR_HASH,
        "Eval manifest must cite the same processor_config.json bytes.",
    ),
    ParityAxisSpec(
        "statistics_json_sha256",
        ("checkpoint", "statistics_json_sha256"),
        ("request", "checkpoint", "statistics_json_sha256"),
        OPENPI_STATS_NORM,
        "Eval manifest must cite the same statistics.json/norm bytes.",
    ),
    ParityAxisSpec(
        "training_algo",
        ("checkpoint", "training_algo"),
        ("request", "checkpoint", "training_algo"),
        OPENPI_MODEL_IDENTITY,
        "Eval-side model identity must match the checkpoint training config.",
    ),
    ParityAxisSpec(
        "formalize_language",
        ("checkpoint", "formalize_language"),
        ("request", "checkpoint", "formalize_language"),
        OPENPI_LANGUAGE_FLAG,
        "Eval-side language formalization flag must match training.",
    ),
)
