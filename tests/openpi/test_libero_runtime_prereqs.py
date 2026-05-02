from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "agent/exchange/openpi_libero_runtime.md"
OPENPI_ROOT = REPO_ROOT / "submodules/openpi"


def test_libero_runtime_doc_exists_and_names_native_paths() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "openpi native LIBERO 运行前提",
        "submodules/openpi/",
        "submodules/openpi/examples/libero/main.py",
        "submodules/openpi/scripts/serve_policy.py",
        "submodules/openpi/third_party/libero/",
        "submodules/openpi/.venv/",
        "pi05_libero",
        "action_horizon=10",
        "discrete_state_input=False",
    ]
    for item in required:
        assert item in text, f"missing required LIBERO runtime doc item: {item}"


def test_libero_runtime_paths_exist() -> None:
    required_paths = [
        OPENPI_ROOT / "examples/libero/main.py",
        OPENPI_ROOT / "scripts/serve_policy.py",
        OPENPI_ROOT / "third_party/libero",
        OPENPI_ROOT / ".venv/bin/python",
    ]
    for path in required_paths:
        assert path.exists(), f"missing required LIBERO runtime path: {path}"
