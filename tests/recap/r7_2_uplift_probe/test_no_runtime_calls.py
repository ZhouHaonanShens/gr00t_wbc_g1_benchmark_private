from __future__ import annotations

import ast
from pathlib import Path

MODULE_ROOT = Path("work/recap/r7_2_uplift_probe")
PARENT_MODULES = ["contract.py", "trial_runner.py", "stepwise_probe.py", "cli.py", "reports.py"]


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_parent_modules_do_not_import_torch_peft_or_numpy_random() -> None:
    forbidden = {"torch", "peft", "numpy.random"}
    for filename in PARENT_MODULES:
        tree = _tree(MODULE_ROOT / filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
                assert forbidden.isdisjoint(names), (filename, names)
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module not in forbidden and not module.startswith("torch"), (filename, module)
                assert not module.startswith("peft") and module != "numpy.random", (filename, module)


def test_subprocess_usage_is_limited_to_runner_and_probe() -> None:
    allowed = {"trial_runner.py", "stepwise_probe.py"}
    for path in MODULE_ROOT.glob("*.py"):
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"run", "Popen"}:
                if isinstance(node.value, ast.Name) and node.value.id == "subprocess":
                    assert path.name in allowed


def test_lora_worker_torch_peft_imports_are_not_top_level() -> None:
    tree = _tree(MODULE_ROOT / "lora_train_worker.py")
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in top_imports:
        if isinstance(node, ast.Import):
            assert all(alias.name not in {"torch", "peft"} for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "") not in {"torch", "peft"}
