# work/openpi/checkpoint/

## 做什么

- 统一处理 checkpoint ref 标准化、servable checkpoint 解析、instance binding 与 provenance 读取。
- 为 eval、runtime、recap pipeline 提供稳定的 checkpoint 输入归一化入口。

## 不做什么

- 不运行 rollout。
- 不做指标聚合。
- 不承载 RECAP collect / iteration 主流程。

## 在 baseline -> RECAP best 主链里的位置

- 这是 checkpoint 相关 contract 的入口层：主线在真正跑 runtime 或 eval 前，先在这里把 checkpoint 身份、来源与可服务路径说清楚。

## 推荐阅读顺序

1. `resolver.py`
2. `source.py`
3. `binding.py`
4. `provenance.py`

## 关键入口、关键 scenario、关键类

- 关键类：`CheckpointResolver`、`CheckpointBindingResolver`
- 关键函数：`resolve_checkpoint_instance_binding(...)`
- 本包没有独立 scenario，但被 `work/openpi/eval/scenarios.py`、`work/openpi/recap/scenarios.py` 间接消费。

## internal-only / compatibility-only

- `__init__.py` 是 export surface；真正定义点在 `resolver.py`、`binding.py`、`source.py`、`provenance.py`。
