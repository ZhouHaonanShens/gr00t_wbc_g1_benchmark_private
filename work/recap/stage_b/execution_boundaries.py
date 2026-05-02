"""Stage B execution boundary checks.

The Stage B controller-output seam work is diagnostic-only. This module gives
launchers and tests a small, dependency-free guardrail for rejecting commands
that would violate the approved scope: full GR00T long-runs, new training,
checkpoint tuning, LoRA, or SFT. It also freezes the priority order that keeps
checkpoint-regression diagnostics ahead of indicator-survival analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Iterable, Sequence


OBJECTIVE_PRIORITY: tuple[str, ...] = (
    "checkpoint_regression_or_policy_collapse_layer_diagnostics",
    "indicator_survival_secondary_axis",
)

STAGE_B_ALLOWED_OPERATION_CLASSES: tuple[str, ...] = (
    "instrumentation",
    "schema_or_trace_writer_self_test",
    "same_observation_probe",
    "one_step_wbc_probe",
    "short_diagnostic_rollout_max_200_steps",
    "staged_metrics_extraction",
    "public_dependency_level0_import_or_server_smoke",
)

STAGE_B_FORBIDDEN_OPERATIONS: tuple[str, ...] = (
    "gr00t_full_long_run",
    "new_method_training",
    "checkpoint_tuning",
    "lora",
    "sft",
    "post_hoc_success_metric_rewrite",
    "benchmark_or_method_success_claim",
)

_TRAINING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "new_method_training",
        re.compile(
            r"(?<!no[-_])(?<!no)(?:"
            r"\b(?:train|training|trainer|fit)\b"
            r"|\b(?:finetune|fine[-_]?tune)(?:\b|[_-])"
            r"|[_-](?:train|training)(?:\b|[_-])"
            r"|\b(torchrun|deepspeed|accelerate\s+launch)\b"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "checkpoint_tuning",
        re.compile(
            r"\b(checkpoint[-_ ]?tuning|tune[-_ ]?checkpoint)\b",
            re.IGNORECASE,
        ),
    ),
    ("lora", re.compile(r"\b(lora|qlora|peft)\b", re.IGNORECASE)),
    (
        "sft",
        re.compile(r"\bsft\b|supervised[-_ ]?fine[-_ ]?tuning", re.IGNORECASE),
    ),
)

_LONG_RUN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "gr00t_full_long_run",
        re.compile(
            r"\b(full[-_ ]?long[-_ ]?run|long[-_ ]?run)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "gr00t_full_long_run",
        re.compile(r"\bformal[_-]?eval\b|\bformal\s+eval\b", re.IGNORECASE),
    ),
    (
        "gr00t_full_long_run",
        re.compile(r"\b(64h|56h|fullsize|full[-_ ]?update)\b", re.IGNORECASE),
    ),
)

_NUMERIC_LIMIT_FLAGS: tuple[tuple[str, int, str], ...] = (
    ("--max-steps", 200, "gr00t_full_long_run"),
    ("--max_steps", 200, "gr00t_full_long_run"),
    ("--steps", 200, "gr00t_full_long_run"),
    ("--episodes", 3, "gr00t_full_long_run"),
    ("--num-episodes", 3, "gr00t_full_long_run"),
    ("--num_episodes", 3, "gr00t_full_long_run"),
)

_SAFE_DIAGNOSTIC_HINTS: tuple[str, ...] = (
    "--self-test",
    "self_test",
    "same_obs",
    "same-observation",
    "one_step",
    "one-step",
    "short_diagnostic",
    "short-diagnostic",
    "stage_b",
    "seam_trace",
)


@dataclass(frozen=True)
class BoundaryDecision:
    """Result of checking a prospective Stage B command."""

    allowed: bool
    reasons: tuple[str, ...]
    normalized_command: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "normalized_command": self.normalized_command,
            "objective_priority": list(OBJECTIVE_PRIORITY),
            "forbidden_operations": list(STAGE_B_FORBIDDEN_OPERATIONS),
        }


def _coerce_tokens(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    return [str(part) for part in command]


def _normalize_command(tokens: Sequence[str]) -> str:
    return " ".join(shlex.quote(token) for token in tokens)


def _iter_flag_values(tokens: Sequence[str], flag: str) -> Iterable[str]:
    prefix = f"{flag}="
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            yield tokens[index + 1]
        elif token.startswith(prefix):
            yield token[len(prefix) :]


def _parse_positive_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def _has_safe_diagnostic_hint(normalized_lower: str) -> bool:
    return any(hint in normalized_lower for hint in _SAFE_DIAGNOSTIC_HINTS)


def classify_stage_b_command(command: str | Sequence[str]) -> BoundaryDecision:
    """Classify whether a command is safe under Stage B boundaries.

    The classifier is intentionally conservative for launch commands. It is not
    a shell sandbox and does not execute anything; callers should use it before
    adding new Stage B wrappers or running probe commands.
    """

    tokens = _coerce_tokens(command)
    normalized = _normalize_command(tokens)
    normalized_lower = normalized.lower()
    reasons: list[str] = []

    for operation, pattern in _TRAINING_PATTERNS:
        if pattern.search(normalized):
            reasons.append(f"forbidden:{operation}")

    for operation, pattern in _LONG_RUN_PATTERNS:
        if pattern.search(normalized):
            reasons.append(f"forbidden:{operation}")

    for flag, limit, operation in _NUMERIC_LIMIT_FLAGS:
        for raw_value in _iter_flag_values(tokens, flag):
            parsed_value = _parse_positive_int(raw_value)
            if parsed_value is None:
                reasons.append(f"unsafe_unparseable:{flag}={raw_value}")
            elif parsed_value > limit:
                reasons.append(f"forbidden:{operation}:{flag}>{limit}")

    if not reasons and _has_safe_diagnostic_hint(normalized_lower):
        reasons.append("allowed:diagnostic_or_self_test_hint")
    elif not reasons:
        reasons.append("allowed:no_forbidden_stage_b_operation_detected")

    return BoundaryDecision(
        allowed=not any(
            reason.startswith(("forbidden:", "unsafe_unparseable:"))
            for reason in reasons
        ),
        reasons=tuple(reasons),
        normalized_command=normalized,
    )


def require_stage_b_safe_command(command: str | Sequence[str]) -> BoundaryDecision:
    """Return a decision or raise ``ValueError`` if the command violates Stage B."""

    decision = classify_stage_b_command(command)
    if not decision.allowed:
        raise ValueError(
            "Stage B boundary violation: "
            + "; ".join(decision.reasons)
            + f" | command={decision.normalized_command}"
        )
    return decision


def build_no_training_contract_markdown() -> str:
    """Render the Stage B no-training/no-full-long-run contract in Chinese."""

    allowed = "\n".join(f"- `{item}`" for item in STAGE_B_ALLOWED_OPERATION_CLASSES)
    forbidden = "\n".join(f"- `{item}`" for item in STAGE_B_FORBIDDEN_OPERATIONS)
    priority = "\n".join(
        f"{idx}. `{item}`" for idx, item in enumerate(OBJECTIVE_PRIORITY, start=1)
    )
    return f"""# Stage B no-training / no-full-long-run contract

本文件冻结 Stage B controller-output seam 的执行边界：Stage B 只做 instrumentation 与诊断。

## 优先级

{priority}

`indicator_survival_secondary_axis` 是第二诊断轴；不得把它后验改写成 Stage B 主 success metric。

## 允许的操作类别

{allowed}

## 禁止的操作类别

{forbidden}

## Claim boundary

- 不启动 GR00T full long-run。
- 不启动新的 method training、checkpoint tuning、LoRA 或 SFT。
- 不根据 Stage B 短诊断结果宣称 benchmark success 或方法成败。
- 所有超过 60 秒的诊断命令必须带 `timeout`。
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-command",
        help="Classify a prospective Stage B command without executing it.",
    )
    parser.add_argument(
        "--write-contract",
        type=Path,
        help="Write the Stage B no-training contract markdown to this path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON for --check-command.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.write_contract is not None:
        args.write_contract.parent.mkdir(parents=True, exist_ok=True)
        args.write_contract.write_text(
            build_no_training_contract_markdown(),
            encoding="utf-8",
        )

    if args.check_command:
        decision = classify_stage_b_command(args.check_command)
        if args.json:
            print(
                json.dumps(
                    decision.to_jsonable(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            status = "ALLOW" if decision.allowed else "BLOCK"
            print(f"{status}: {decision.normalized_command}")
            for reason in decision.reasons:
                print(f"- {reason}")
        return 0 if decision.allowed else 2

    if args.write_contract is None:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
