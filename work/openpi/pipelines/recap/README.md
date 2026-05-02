# work/openpi/pipelines/recap/

## 做什么

- 提供 RECAP 主线 workflow：collect、merge、critic retrain、policy retrain、iteration 与 loop。
- 把 `work/openpi/recap/` 的共享实现组装成可运行、可验证、可落盘的主流程。

## 不做什么

- 不应把底层 dataset/checkpoint/prompt 细节全部塞回这里。
- 不替代 `work/openpi/scripts/openpi.py` 的 public entry 角色。

## 在 baseline -> RECAP best 主链里的位置

- 这是 RECAP 主线的 orchestration 层。默认 collection/iteration/loop 都从这里落到具体 workflow，再下沉到 `work/openpi/recap/` 与 `work/openpi/eval/`。

## 推荐阅读顺序

1. `scenarios.py`
2. `collect.py`
3. `collect_workflow.py`
4. `merge.py`
5. `critic_training.py`
6. `policy_training.py`
7. `iteration.py`
8. `iteration_workflow.py`
9. `loop.py`
10. `variant_training.py`

## 关键入口、关键 scenario、关键类

- 关键 facade：`collect.py`、`iteration.py`
- 关键 workflow：`collect_workflow.py`、`iteration_workflow.py`
- 关键 scenario：`scenarios.py`
- 关键入口：`collect.py`、`iteration.py`、`loop.py`

## internal-only / compatibility-only

- `collect.py` / `iteration.py` 是 facade + compatibility surface。
- `collect_workflow.py` / `iteration_workflow.py` 承载真实 workflow 实现。
- 真正公开入口仍应通过 `work/openpi/scripts/openpi.py` 或上层 thin wrapper 进入。
