# work/openpi/eval/

## 做什么

- 承载 canonical eval 主线：scenario 定义、app 门面、workflow、protocol 与报告产物生成。
- 提供 stock smoke、rollout evaluation、tracked-gate evaluation 这三类主入口。

## 不做什么

- 不实现 checkpoint 解析细节。
- 不承载 RECAP collect / merge / policy retrain 的 pipeline 逻辑。

## 在 baseline -> RECAP best 主链里的位置

- 这是 baseline -> RECAP best 主链里的 eval 主入口层。上游 scenario 在这里被路由到对应 workflow，并最终产出 trace、metrics、summary、gate verdict 等 artifact。

## 推荐阅读顺序

1. `app.py`
2. `scenarios.py`
3. `workflows/`
4. `protocols/`
5. `reports/`
6. `cli.py`

## 关键入口、关键 scenario、关键类

- 关键类：`OpenPIEvalApp`、`TrackedGateEvaluationWorkflow`、`RolloutEvaluationWorkflow`
- 关键 scenario：`DEFAULT_STOCK_SMOKE_SCENARIO`、`DEFAULT_STOCK_ROLLOUT_SCENARIO`、`DEFAULT_RECAP_BEST_ROLLOUT_SCENARIO`、`DEFAULT_TRACKED_GATE_EVALUATION_SCENARIO`
- 关键子目录：`workflows/`、`protocols/`、`reports/`

## internal-only / compatibility-only

- `app.py` 是推荐入口。
- `cli.py` 主要服务命令行/兼容 surface；真正实现仍在 `app.py` 与 `workflows/**`。
