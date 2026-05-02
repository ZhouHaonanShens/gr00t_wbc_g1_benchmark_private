# work/openpi/model/

## 做什么

- 集中放置模型绑定、runtime prompting、effective runtime spec 与 rollout 输入摘要相关逻辑。
- 让上层 workflow 可以稳定地把“请求的 runtime”与“实际执行的 runtime”投影成可比较的结构化数据。

## 不做什么

- 不直接读写 rollout trace JSONL。
- 不直接运行 server/client subprocess。

## 在 baseline -> RECAP best 主链里的位置

- 位于 checkpoint / prompting contract 与 eval/runtime workflow 之间，负责把模型与 runtime 语义固定下来。

## 推荐阅读顺序

1. `spec.py`
2. `prompting.py`
3. `binding.py`
4. `summary.py`

## 关键入口、关键 scenario、关键类

- 关键入口：`spec.py`、`prompting.py`
- 常被 `work/openpi/eval/app.py` 与 `work/openpi/recap/runtime_prompt.py` 消费。

## internal-only / compatibility-only

- 本包主要提供 builder/normalizer 风格的内部复用逻辑，不是 public CLI。
