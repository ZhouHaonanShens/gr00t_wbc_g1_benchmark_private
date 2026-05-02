# work/openpi/runtime/

## 做什么

- 提供 OpenPI + LIBERO runtime bridge 的主实现：server 生命周期、episode client、路径生成、环境 session 与清理逻辑。
- 为 stock smoke、rollout evaluation 与其它需要真实 runtime 的流程提供统一桥接面。

## 不做什么

- 不负责 repaired gate 指标聚合。
- 不承载 RECAP collect / iteration 的高层 orchestration。

## 在 baseline -> RECAP best 主链里的位置

- 这是 baseline stock smoke 与 rollout materialization 的执行层。上层 scenario / workflow 在这里真正落到 server、client、episode 和日志/证据路径。

## 推荐阅读顺序

1. `config.py`
2. `api.py`
3. `bridge.py`
4. `server.py`
5. `client.py`
6. `environment.py`
7. `paths.py`
8. `cleanup.py`
9. `internal.py`

## 关键入口、关键 scenario、关键类

- 关键类：`RuntimeBridgeConfig`、`PolicyServerProcess`、`RuntimeEpisodeClient`、`LiberoEnvironmentSession`、`RuntimePathsBuilder`、`RuntimeCleanup`
- 关键公开 surface：`api.py`
- 关键实现模块：`bridge.py`
- 典型调用方：`work/openpi/eval/workflows/stock_smoke.py`、`work/openpi/eval/app.py`

## internal-only / compatibility-only

- `api.py` / `__init__.py` 是公开 runtime surface；不再把 `bridge.py` 的私有 helper 直接暴露给上层 workflow。
- `internal.py` 是内部 subprocess 入口，只服务 `probe` / `client` 子模式。
- `bridge.py` 是真实实现层；`__init__.py` 主要承担 package export surface。
