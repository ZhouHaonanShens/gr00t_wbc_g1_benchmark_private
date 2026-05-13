from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PKG = REPO_ROOT / "work" / "recap" / "r6_runtime_indicator_probe"
STATIC_MODULES = ("contract.py", "wiring_graph.py", "synthesis.py", "cli.py")
FORBIDDEN_IMPORT_ROOTS = {"torch", "cuda"}
FORBIDDEN_SNIPPETS = ("subprocess.run", "CUDA_VISIBLE_DEVICES")
CAPS = {"__init__.py": 12, "__main__.py": 5, "contract.py": 120, "wiring_graph.py": 200, "runtime_probe.py": 180, "instrumentation.py": 100, "synthesis.py": 50, "cli.py": 130, "reports/__init__.py": 5, "reports/cell_probe_report.py": 80, "reports/summary_report.py": 80}


def _read(rel: str) -> str:
    return (PKG / rel).read_text(encoding="utf-8")


def test_static_trace_modules_have_no_torch_subprocess_or_cuda_runtime_leakage() -> None:
    offenders: list[tuple[str, str]] = []
    for rel in STATIC_MODULES:
        text = _read(rel)
        offenders.extend((rel, snippet) for snippet in FORBIDDEN_SNIPPETS if snippet in text)
        for node in ast.walk(ast.parse(text, filename=rel)):
            if isinstance(node, ast.Import):
                offenders += [(rel, alias.name) for alias in node.names if alias.name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS]
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                offenders.append((rel, node.module))
    assert offenders == []


def test_trace_path_does_not_import_runtime_probe_or_instrumentation(tmp_path: Path) -> None:
    code = """
import sys
from work.recap.r6_runtime_indicator_probe.cli import main
rc = main(['trace', '--all', '--output-root', r'{out}'])
assert rc == 0, rc
assert 'work.recap.r6_runtime_indicator_probe.runtime_probe' not in sys.modules
assert 'work.recap.r6_runtime_indicator_probe.instrumentation' not in sys.modules
""".format(out=tmp_path.as_posix())
    completed = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_file_loc_caps_match_approved_contract() -> None:
    loc = {rel: len(_read(rel).splitlines()) for rel in CAPS}
    over = {rel: count for rel, count in loc.items() if count > CAPS[rel]}
    assert over == {}
