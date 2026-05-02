from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
LAYOUT_DOC = REPO_ROOT / "agent/exchange/openpi_integration_layout.md"
PACKAGE_INIT = REPO_ROOT / "work/openpi/__init__.py"
PACKAGE_README = REPO_ROOT / "work/openpi/README.md"
AGENT_RUN_DIR = REPO_ROOT / "agent/run"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_openpi_layout_doc_freezes_readonly_dependency_and_adapter_boundary() -> None:
    text = _read(LAYOUT_DOC)
    required = [
        "submodules/openpi/",
        "第三方只读 dependency",
        "work/openpi/",
        "唯一 adapter 层",
        "agent/run/**",
        "不承载 openpi 业务逻辑",
    ]
    for item in required:
        assert item in text, f"missing required layout-boundary text: {item}"


def test_openpi_package_boundary_files_exist() -> None:
    assert PACKAGE_INIT.is_file(), "missing work/openpi/__init__.py"
    assert PACKAGE_README.is_file(), "missing work/openpi/README.md"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    _ = importlib.import_module("work.openpi")


def test_agent_run_has_no_openpi_business_logic_surface() -> None:
    openpi_named_entries = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in AGENT_RUN_DIR.rglob("*openpi*")
    ]
    assert openpi_named_entries == [], (
        f"openpi-specific files must not live under agent/run: {openpi_named_entries}"
    )
