# work/openpi/state_tokens/

## 做什么

- 承载 state-token 路线的 checkpoint、dataset、protocol、summary 与 scenario 定义。
- 为与主 RECAP 路线并行的 state-token 适配实验提供独立的 contract surface。

## 不做什么

- 不替代 canonical eval 主线。
- 不承载 runtime bridge 的底层实现。

## 在 baseline -> RECAP best 主链里的位置

- 这是旁路线/扩展路线包；与 baseline -> RECAP best 主链并列存在，但不是默认阅读入口。

## 推荐阅读顺序

1. `scenarios.py`
2. `protocol.py`
3. `checkpoint.py`
4. `dataset.py`
5. `summary.py`

## 关键入口、关键 scenario、关键类

- 关键入口：`scenarios.py`、`protocol.py`
- 本包的关键结构集中在 route-specific contract，而不是 workflow orchestration。

## internal-only / compatibility-only

- 默认不作为 `work/openpi/scripts/openpi.py` 的主入口；更多是并行实验或扩展路线支撑层。
