# work/openpi/serve/

## 做什么

- 承载 OpenPI serve/provenance 侧的适配逻辑。
- 当前主文件是 `provenance.py`，用于把服务端或可部署 checkpoint 的 provenance 信息整理成稳定 surface。

## 不做什么

- 不直接启动 policy server。
- 不负责 rollout、metrics 或 RECAP workflow orchestration。

## 在 baseline -> RECAP best 主链里的位置

- 位于 checkpoint / deployable artifact 与上游消费方之间，负责让 provenance 能被稳定读取和传递。

## 推荐阅读顺序

1. `provenance.py`

## 关键入口、关键 scenario、关键类

- 关键入口：`provenance.py`
- 本包没有独立 scenario。

## internal-only / compatibility-only

- 默认作为内部适配层使用；server 生命周期逻辑不在这里，而在 `work/openpi/runtime/`。
