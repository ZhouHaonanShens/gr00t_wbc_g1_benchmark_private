from __future__ import annotations

import ast
from pathlib import Path

MODULE_ROOT = Path("work/recap/r7_1_recipe_plumbing")
MODULE_FILES = sorted(MODULE_ROOT.glob("*.py"))
FORBIDDEN_IDENTIFIERS = (
    "learned_value",
    "value_head",
    "advantage_embedding",
    "advantage_projection",
    "action_head_advantage_input",
)


def _tree(path: Path) -> ast.AST:
    source_text = path.read_text(encoding="utf-8")
    return ast.parse(source_text)


def test_all_new_module_files_are_scanned() -> None:
    assert [path.name for path in MODULE_FILES] == [
        "__init__.py",
        "__main__.py",
        "cli.py",
        "dryrun.py",
        "dual_loss_wiring.py",
        "flags.py",
        "indicator_dropout.py",
    ]


def test_no_new_module_imports_torch_static_or_dynamic() -> None:
    for path in MODULE_FILES:
        source_text = path.read_text(encoding="utf-8")
        tree = ast.parse(source_text)
        assert "import torch" not in source_text
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert not (node.func.id in {"__import__", "eval", "exec"} and "torch" in ast.unparse(node))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (node.func.attr == "import_module" and "torch" in ast.unparse(node))


def test_only_dryrun_calls_subprocess_run() -> None:
    offenders: list[str] = []
    for path in MODULE_FILES:
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run":
                if path.name != "dryrun.py" or ast.unparse(node.func.value) != "subprocess":
                    offenders.append(path.name)
    assert offenders == []


def test_cuda_string_only_appears_in_dryrun() -> None:
    offenders = [
        path.name
        for path in MODULE_FILES
        if path.name != "dryrun.py"
        and "cuda" in path.read_text(encoding="utf-8").lower()
    ]
    assert offenders == []


def test_no_c3_c4_identifiers_in_new_module() -> None:
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in MODULE_FILES)
    for forbidden in FORBIDDEN_IDENTIFIERS:
        assert forbidden not in source_text
