# Stage B seam trace writer integration contract

本目录只服务 Stage B controller-output seam 的 instrumentation / diagnostics。它不得启动 GR00T full long-run，也不得启动新训练、checkpoint tuning、LoRA 或 SFT。

## Trace identity contract

- `chain_action_uuid`：同一次 policy call 从 policy → controller → env 的 join key；只包含 `trace_version, episode_id, step_id, seed, policy_call_index, obs_hash`，不得包含 action 内容。
- `contrast_group_uuid`：same-observation triplet / checkpoint-pair 对比的配对 key；包含 `trace_version, seed, obs_hash, frozen_controller_state_hash, probe_name`，刻意排除 `indicator_mode` 与 action 内容。
- `action_content_hash`：被诊断的 action/controller/env payload 内容 hash；它不是 join key。

## Policy-side caller example

```python
from work.recap.stage_b.action_uuid import make_chain_action_uuid, make_contrast_group_uuid
from work.recap.stage_b.schema import TRACE_VERSION
from work.recap.stage_b.seam_trace_writer import SeamTraceWriter

writer = SeamTraceWriter(stage_b_dir / "seam_traces" / "policy_side")
chain_id = make_chain_action_uuid(
    trace_version=TRACE_VERSION,
    episode_id=episode_id,
    step_id=step_id,
    seed=seed,
    policy_call_index=policy_call_index,
    obs_hash=obs_hash,
)
contrast_id = make_contrast_group_uuid(
    trace_version=TRACE_VERSION,
    seed=seed,
    obs_hash=obs_hash,
    frozen_controller_state_hash=controller_reset_hash,
    probe_name="same_obs_triplet",
)
writer.record_array_event(
    stage="policy",
    name="decoded_action",
    episode_id=episode_id,
    step_id=step_id,
    chain_action_uuid=chain_id,
    contrast_group_uuid=contrast_id,
    seed=seed,
    indicator_mode=indicator_mode,
    obs_hash=obs_hash,
    array=decoded_action,
)
writer.flush()  # safe post-step / post-call flush point
```

## WBC/env caller rules

1. timing-sensitive path 内只调用 `record_array_event(...)` 缓存拷贝；在 post-step 安全点调用 `flush()`。
2. true torque 不可见时必须写 `missing_stage_reason`，并把 proxy 名称放进 `metadata`；不得把 `last_action.q` 或 `env.step({q,tau})` 伪称 true controller torque。
3. trace writer 默认 `strict=False`，trace 写入失败不应改变 rollout 行为；debug/CI 可用 `strict=True` fail-fast。
4. raw arrays 写入 `arrays_*.npz`，JSONL 只写 metadata、shape、dtype、hash、min/max/mean/std 与 `array_ref`。

## Local self-test

```bash
python3 -m work.recap.stage_b.seam_trace_writer --self-test --output-dir <stage_B_dir>/seam_traces/self_test
```

## P2 temporary inference adapter contract

`p2_inference_adapter.py` 冻结 Stage B P2 的最小安全面：它只做
inference-only contract / no-env sanity / gated skip artifact，不加载 GR00T、不改
checkpoint、不启动 rollout 或训练。真实 P2 eval 仍必须等 P1 loader audit PASS 且
P0 eval-protocol ladder 未恢复 collapse 后，才可按同一 contract 接入 runtime。

```bash
python3 agent/run/stage_b_p2_inference_adapter.py \
  --self-test \
  --output-dir <stage_B_dir>/prechecks/P2_inference_unconditional_swap
```
