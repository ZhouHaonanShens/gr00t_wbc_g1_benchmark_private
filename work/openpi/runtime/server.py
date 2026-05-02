from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

from work.openpi.contracts import RuntimeServerSpec

from .bridge import _spawn_server, _wait_for_server_ready


@dataclass
class PolicyServerProcess:
    """Manage the OpenPI policy-server subprocess for a runtime session.

    The class owns spawn and ready-wait responsibilities only. Episode-level
    execution stays outside this boundary so workflows can reason separately
    about server lifecycle versus rollout/client work.
    """

    spec: RuntimeServerSpec
    venv_python: Path
    serve_policy: Path
    openpi_root: Path
    server_log: Path
    libero_config_dir: Path
    cli_entry: Path
    process: subprocess.Popen[str] | None = None
    handle: Any | None = None

    def start(self) -> tuple[subprocess.Popen[str], Any]:
        self.process, self.handle = _spawn_server(
            self.spec,
            venv_python=self.venv_python,
            serve_policy=self.serve_policy,
            openpi_root=self.openpi_root,
            server_log=self.server_log,
            libero_config_dir=self.libero_config_dir,
        )
        return self.process, self.handle

    def wait_until_ready(
        self,
        *,
        runtime_dir: Path,
        harness_log: Path,
    ) -> dict[str, object]:
        if self.process is None:
            raise RuntimeError("server process has not been started")
        return _wait_for_server_ready(
            self.spec,
            proc=self.process,
            runtime_dir=runtime_dir,
            venv_python=self.venv_python,
            openpi_root=self.openpi_root,
            libero_config_dir=self.libero_config_dir,
            harness_log=harness_log,
            server_log=self.server_log,
            cli_entry=self.cli_entry,
        )
