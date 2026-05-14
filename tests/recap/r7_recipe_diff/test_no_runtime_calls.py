from __future__ import annotations

import ast
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
R7_PKG_DIR = REPO_ROOT / "work" / "recap" / "r7_recipe_diff"
FORBIDDEN_IMPORT_PREFIXES = (
    "torch",
    "gr00t",
    "subprocess",
    "work.recap.launch_finetune_use_ddp",
    "work.recap.finetune_full",
)
FORBIDDEN_CALL_PATTERNS = (
    r"\bsubprocess\.",
    r"\bos\.system\s*\(",
    r"\bos\.popen\s*\(",
    r"\bcuda\s*\(",
    r"\.cuda\s*\(",
    r"\bnvidia-smi\b",
)
DYNAMIC_IMPORT_CALLS = {
    "__import__",
    "builtins.__import__",
    "importlib.import_module",
    "import_module",
}


def _py_files() -> list[Path]:
    paths = sorted(path for path in R7_PKG_DIR.glob("*.py") if "__pycache__" not in path.parts)
    assert paths, f"R7 package root contains no Python files: {R7_PKG_DIR}"
    return paths


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def test_zero_gpu_invocation_environment_is_empty() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_no_forbidden_runtime_imports() -> None:
    offenders: list[tuple[Path, str]] = []
    for path in _py_files():
        tree = ast.parse(_read(path), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                        offenders.append((path, alias.name))
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    offenders.append((path, node.module))
    assert offenders == []


def test_no_forbidden_runtime_calls() -> None:
    offenders = [
        (path, pattern)
        for path in _py_files()
        for pattern in FORBIDDEN_CALL_PATTERNS
        if re.search(pattern, _read(path))
    ]
    assert offenders == []


def test_no_dynamic_imports() -> None:
    offenders: list[tuple[Path, str]] = []
    for path in _py_files():
        tree = ast.parse(_read(path), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in DYNAMIC_IMPORT_CALLS or name.endswith(".__import__"):
                    offenders.append((path, name))
    assert offenders == []
