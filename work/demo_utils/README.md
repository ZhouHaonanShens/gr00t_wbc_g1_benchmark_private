# work.demo_utils

本目录是一个“demo / sandbox 脚本工程化复用”的轻量 Python 工具包，目标是把 `agent/run/*.py` 中稳定且重复的逻辑抽出来，形成可长期维护的公共实现。

## 设计目标

- 统一脚本行为：路径组织、日志 tee、视频目录命名/归档、env registry 列举、policy server 相关探测/子进程管理。
- 统一证据落盘：尽量把关键输出写入 `agent/runtime_logs/<topic>/`，产物归档到 `agent/artifacts/`。
- 降低 demo 脚本耦合：脚本只保留“具体业务逻辑”（例如 env 构造细节、action schema 细节），通用实现放到这里。

## 硬约束（必须遵守）

1) `work.demo_utils` 必须在“未安装仿真依赖”的情况下可 import。

- 允许 import 的：标准库。
- 禁止在模块顶层 import 的：`gymnasium`、`gym`、`mujoco`、`robosuite`、`numpy`、`zmq`、`gr00t*` 等（这些必须放到函数体内，用运行时 import）。

2) 行为稳定性优先。

- 工具函数常用于 demo 脚本的“证据落盘与排错”；重构时应尽量保持日志文件名、header 文案、目录结构与输出格式不变。

3) 不修改第三方上游代码。

- `submodules/**` 只读。

## 包结构与 API

### `work/demo_utils/apps/`

- 放置依赖 `work.demo_utils.*` 的具体 demo / smoke / checker 脚本实现。
- 当前包含：
  - `mock_policy_server_zero.py`
  - `official_rollout_apple_to_plate_smoke.py`
  - `official_rollout_apple_to_plate_onscreen.py`
  - `pseudodemo_label_contract_checker.py`
- 这些文件是“具体应用层”，不是通用 helper；`agent/run/*.py` 中对应 public CLI 继续以 thin wrapper 方式转发到这里。

### `work/demo_utils/paths.py`

- `repo_root(from_path=None) -> Path`
  - 用于定位 repo 根目录。
  - demo 脚本建议传 `from_path=__file__`，保证“脚本移动/不同 cwd”场景下行为稳定。
- `ensure_dirs(repo_root, runtime_logs_rel, artifacts_videos_rel) -> (runtime_dir, artifacts_videos)`
  - 创建 runtime logs 与视频归档目录。
- `ensure_demo_live_dirs(repo_root, video_archive_dir) -> (runtime_dir, artifacts_videos, server_log, client_log)`
  - 与现有 demo_live 的命名约定对齐：
    - `agent/runtime_logs/demo_live/00_server.log`
    - `agent/runtime_logs/demo_live/01_client.log`
- `wbc_venv_python(repo_root) -> Path`
- `maybe_reexec_into_wbc_venv(repo_root) -> None`
  - 若 WBC venv python 存在且当前不在该 venv，则 `execv` 重新进入。
  - 会清理 `PYTHONPATH`，避免宿主环境污染。

### `work/demo_utils/tee.py`

- `tee_stdio(log_path: Path, header: str)`
  - contextmanager：将 stdout/stderr tee 到 `log_path`（行缓冲），同时仍输出到终端。
  - `header` 由调用方传入，用于保持不同脚本的落盘 header 文案稳定。

### `work/demo_utils/videos.py`

- `make_video_dir(env_name: str, n_action_steps: int) -> Path`
  - 生成 `/tmp/sim_eval_videos_*` 风格路径。
  - 注意：`env_name` 可能包含 `/`，因此返回的 Path 可能是多级目录；这是既有行为的一部分。
- `archive_video_dir(video_dir: Path | None, archive_root: Path) -> Path | None`
  - 将 `/tmp/...` 目录复制归档到 `archive_root`（通常是 `agent/artifacts/videos/`）。

### `work/demo_utils/env_registry.py`

- `list_registered_env_ids(prefix: str, log_path: Path, register_modules=()) -> list[str]`
  - 运行时 import `gymnasium` 或 `gym` 的 registration 模块并读取 registry。
  - 兼容 modern gym/gymnasium 的 dict registry，以及 legacy gym 的 `EnvRegistry.env_specs` / `EnvRegistry.all()`。
  - `register_modules` 用于“先 import 某些模块以触发 env 注册”（例如 WBC 的 env 注册模块）。

### `work/demo_utils/policy_server.py`（可选，但推荐复用）

- `normalize_client_host(host)`
- `is_tcp_port_listening(host, port, timeout_s=0.2)`
- `make_policy_client(host, port, timeout_ms)` / `configure_policy_client_socket(client, timeout_ms)`
- `safe_ping(client, timeout_ms)` / `safe_kill_server(client, timeout_ms)`
- `spawn_server_subprocess(cmd, log_path, cwd=None, env=None) -> Popen`
- `terminate_process(proc, timeout_s)`

这些函数用于 demo 脚本对 policy server 的“可控等待、可控 kill、子进程不泄漏”。其中 `zmq` 与 `gr00t.policy.server_client` 通过运行时 import 引入，避免影响 `--help`/import。

### `work/demo_utils/signals.py`

- `install_signal_handlers(raise_keyboardinterrupt=True, print_fn=print) -> threading.Event`
  - 统一 SIGINT/SIGTERM 为“请求退出”语义；默认通过抛出 `KeyboardInterrupt` 走调用方既有 cleanup。
  - 是否接入由具体脚本决定（本包仅提供通用能力）。

### `work/demo_utils/robosuite_env.py`

- `set_hard_reset_best_effort(env, hard_reset: bool) -> bool`
  - best-effort 修改 robosuite 风格环境的 `hard_reset` 开关。
  - 用反射方式向下找 `base_env/env/unwrapped`，不要求 import robosuite。

## 在脚本中如何正确使用

本仓库的 demo 脚本并非通过 pip 安装，因此“从任意目录运行脚本”时，需要确保 repo root 在 `sys.path`。

推荐做法（已在 `work/recap/scripts/demo_g1_vla_live.py` 采用；`demo_mujoco_viewer_baseline.py` 已归档到 `agent/archive/run/demo_smoke_sandbox/`）：

```python
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3] if "work/demo_utils/apps/" in __file__ else Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
```

然后用运行时 import：

```python
import importlib

tee_mod = importlib.import_module("work.demo_utils.tee")
tee_stdio = getattr(tee_mod, "tee_stdio")
```

这样可以满足两类需求：

- `python work/recap/scripts/xxx.py --help` 时不强制要求安装 `gymnasium/mujoco/zmq/...`
- 脚本从任意工作目录启动时仍能导入 `work.demo_utils`
- 若脚本位于 `work/demo_utils/apps/`，则 repo root 需要上跳 3 层而不是 2 层。

## 维护指南（给后续 AGENT/维护者）

1) 新增功能前先判断“是否稳定可复用”。

- 只抽“通用且稳定”的工具（路径/日志/进程/归档）。
- 不要把具体业务逻辑（env 构造细节、action schema/joint order 等）塞进 `work.demo_utils`。

2) 保持 import-time 纯标准库。

- 如果需要第三方依赖：把 import 放到函数体内（或用 `importlib.import_module`）。
- 不要在模块顶层创建 socket、启动子进程、读写大文件等。

3) 保持脚本对外行为稳定。

- 尽量不要改动：log 文件名、header 格式、输出文案、目录结构。
- 如果确实要改，必须同步更新对应的协作日志与 prompt，并给出迁移说明。

4) 最小验证（必须带 timeout + 落盘证据）。

建议 smoke：

```bash
timeout 30s python -c "import work.demo_utils; print('import ok')"
timeout 30s python agent/run/<script>.py --help
```

并将输出 `tee` 到 `agent/runtime_logs/<topic>/`。

## 关联实现与证据

- 该包最初由 Prompt 22 抽取：参见 `agent/logs/demo_utils_20260212_215723.md`
- 使用示例脚本：
  - `work/recap/scripts/demo_g1_vla_live.py`
  - `work/demo_utils/apps/official_rollout_apple_to_plate_smoke.py`
  
