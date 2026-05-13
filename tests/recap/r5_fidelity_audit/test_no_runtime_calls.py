from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
R5_PKG_DIR = REPO_ROOT / "work" / "recap" / "r5_fidelity_audit"
IMPLEMENTATION_LOC_CAP = 850
FORBIDDEN_IMPORT_ROOTS = {"numpy", "torch", "subprocess", "requests", "urllib"}
FORBIDDEN_CALL_PATTERNS = (r"\bsubprocess\.", r"\bos\.system\s*\(", r"\bos\.popen\s*\(", r"\bos\.spawn[a-zA-Z_]*\s*\(")
DYNAMIC_IMPORT_CALLS = {"__import__", "builtins.__import__", "importlib.import_module", "importlib.util.module_from_spec", "importlib.util.spec_from_file_location", "import_module", "module_from_spec", "spec_from_file_location"}
FORBIDDEN_OS_IMPORTS = {"popen", "system"}
RECURSIVE_SCAN_CALLS = {"Path.rglob", "os.walk"}


def _py_files() -> list[Path]:
    assert R5_PKG_DIR.is_dir(), f"R5 package root is missing: {R5_PKG_DIR}"
    paths = sorted(p for p in R5_PKG_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    assert paths, f"R5 package root contains no Python files: {R5_PKG_DIR}"
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


def test_r5_package_exists_for_static_guard() -> None:
    _py_files()


def test_no_forbidden_runtime_imports() -> None:
    offenders: list[tuple[Path, str]] = []
    for path in _py_files():
        for node in ast.walk(ast.parse(_read(path), filename=str(path))):
            if isinstance(node, ast.Import):
                offenders += [(path, a.name) for a in node.names if a.name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS]
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                    offenders.append((path, node.module))
                if node.module == "os":
                    offenders += [(path, f"os.{a.name}") for a in node.names if a.name in FORBIDDEN_OS_IMPORTS or a.name.startswith("spawn")]
    assert offenders == [], f"forbidden runtime imports in R5: {offenders}"


def test_no_forbidden_runtime_calls() -> None:
    offenders = [(path, pat) for path in _py_files() for pat in FORBIDDEN_CALL_PATTERNS if re.search(pat, _read(path))]
    assert offenders == [], f"forbidden runtime calls in R5: {offenders}"


def test_no_dynamic_imports() -> None:
    offenders: list[tuple[Path, str]] = []
    for path in _py_files():
        for node in ast.walk(ast.parse(_read(path), filename=str(path))):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in DYNAMIC_IMPORT_CALLS or name.endswith(".__import__"):
                    offenders.append((path, name))
    assert offenders == [], f"dynamic imports are forbidden in R5: {offenders}"


def test_no_unbounded_recursive_scans() -> None:
    offenders: list[tuple[Path, str]] = []
    for path in _py_files():
        for node in ast.walk(ast.parse(_read(path), filename=str(path))):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name in RECURSIVE_SCAN_CALLS or name == "rglob" or name.endswith(".rglob"):
                offenders.append((path, name))
            if (name == "glob" or name.endswith(".glob")) and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "**" in arg.value:
                    offenders.append((path, f"{name}({arg.value!r})"))
    assert offenders == [], f"unbounded recursive scan calls are forbidden in R5: {offenders}"


def test_implementation_loc_cap() -> None:
    loc = {str(p.relative_to(REPO_ROOT)): len(_read(p).splitlines()) for p in _py_files()}
    total = sum(loc.values())
    assert total <= IMPLEMENTATION_LOC_CAP, f"R5 implementation LOC cap exceeded: {total} > {IMPLEMENTATION_LOC_CAP}; by file: {loc}"
