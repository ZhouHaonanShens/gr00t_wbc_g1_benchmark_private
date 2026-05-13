from __future__ import annotations

from typing import Any
from work.recap.r6_runtime_indicator_probe.contract import ProbeCounterfactual, RuntimeTrace
R6_MATRIX_SHA = "6cfe1c5460ed0b34c6447785700874f563a55e374c4454c11f6c7852c6d3817b"
R6_REPORT_SHA = "96b7f0e407895dcebc657f472e99096d2b013adabe327975f49f4f6e62a61b90"
R5_MATRIX_SHA = "49f317f4259ea8187b35ce5d99a2788484829b1cee47566446deb2a45646dd84"
R5_REPORT_SHA = "1960de2cc4497587ed5df3195d0c3d63984fe10449a58feb9f7dbd4f413b3fe4"
PHASE_A_SHA = "c5b1dc1ef98a761a198b214a9d6e28f66fefa03f675eb807cb23ebc664e84777"
BRANCH_RUNTIME = "branch=runtime_wiring_broken; estimated_fix_horizon=days"
BRANCH_ALGORITHM = "branch=algorithm_fidelity_gap; estimated_fix_horizon=weeks"
BRANCH_DATA = "branch=data_or_criterion_semantic; estimated_fix_horizon=days_to_weeks"


def _branch(runtime: RuntimeTrace, counterfactual: ProbeCounterfactual | None) -> str:
    if runtime.runtime_verdict == "INDICATOR_ABSENT" or not runtime.indicator_substring_present:
        return BRANCH_RUNTIME
    if counterfactual is not None and counterfactual.counterfactual_verdict == "INDICATOR_INVARIANT":
        return BRANCH_ALGORITHM
    return BRANCH_DATA


def render_runtime_probe_report(
    *,
    runtime: RuntimeTrace,
    counterfactual: ProbeCounterfactual | None,
    negative_runtime: RuntimeTrace | None = None,
    budget: Any = None,
    leader_token_sha256: str,
) -> str:
    prompt = runtime.prompt_text_at_tokenizer[:500]
    cf = counterfactual
    neg_sha = cf.negative_trace_sha256 if cf is not None else "not_run"
    neg_tokens = negative_runtime.prompt_tokens_sha256 if negative_runtime is not None else "not_recorded"
    neg_l2 = negative_runtime.first_5_actions_l2 if negative_runtime is not None else "not_recorded"
    cond_equal = cf.condition_sha_equal if cf is not None else "not_run"
    diff = cf.first_5_actions_l2_diff if cf is not None else "not_run"
    cf_verdict = cf.counterfactual_verdict if cf is not None else "not_run"
    branch = _branch(runtime, counterfactual)
    return f"""# FIX_R2_A1_LOAD_06_R6_RUNTIME_PROBE_REPORT

- R6.0 matrix sha256: `{R6_MATRIX_SHA}`
- R6.0 report sha256: `{R6_REPORT_SHA}`
- R5 matrix sha256: `{R5_MATRIX_SHA}`
- R5 report sha256: `{R5_REPORT_SHA}`
- Phase A literal/report sha256: `{PHASE_A_SHA}`

## S1 Probe setup

- cell: `{runtime.cell_id}`
- seed: `{runtime.episode_seed}`
- gpu: `{budget.gpu_id}`
- budget: max_minutes_per_cell={budget.max_minutes_per_cell}, max_minutes_total={budget.max_minutes_total}, max_steps_per_episode={budget.max_steps_per_episode}
- leader token sha256-of-token: `{leader_token_sha256}`

## S2 Positive run

- prompt_at_tokenizer_step0_first_500:

```text
{prompt}
```

- tokens_sha256: `{runtime.prompt_tokens_sha256}`
- condition_sha256: `{runtime.action_head_conditioning_sha256}`
- first_5_actions_l2: `{runtime.first_5_actions_l2}`

## S3 Negative counterfactual

- tokens_sha256: `{neg_tokens}`
- condition_sha256: `{neg_sha}`
- first_5_actions_l2: `{neg_l2}`
- first_5_actions_l2_diff_against_positive: `{diff}`
- counterfactual_verdict: `{cf_verdict}`

## S4 Diff table

| condition_sha_equal | first_5_actions_l2_diff |
|---|---|
| `{cond_equal}` | `{diff}` |

## S5 indicator_substring_present_in_positive_prompt

`{str(runtime.indicator_substring_present).lower()}`

## S6 Exit decision

{branch}

Cites: positive_trace_sha256=`{runtime.action_head_conditioning_sha256}`, negative_trace_sha256=`{neg_sha}`, condition_sha_equal=`{cond_equal}`, first_5_actions_l2_diff=`{diff}`. This is a 1-episode forced probe, not a 30-episode R2 success-rate rerun.\n"""
