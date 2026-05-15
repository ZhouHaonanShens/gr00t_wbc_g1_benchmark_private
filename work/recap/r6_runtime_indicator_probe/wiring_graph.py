from __future__ import annotations

import ast
from pathlib import Path

from work.recap.r6_runtime_indicator_probe.contract import R6Error, WiringEdge, WiringGraph

ENTRY_SYMBOLS = (
    "work.recap.text_indicator.build_canonical_text_indicator",
    "work.recap.text_indicator.build_authoritative_carrier_text_v1",
    "work.recap.policy.validate_mainline_runtime_indicator_mode",
)
SINK_SYMBOLS = (
    "work.recap.policy.<RuntimePolicy>.get_action",
    "work.recap.model.GR00TRecapModel.forward",
    "submodules.gr00t.<Transformer>.forward",
)
CELL_IDS = ("A.2", "A.3", "A.4", "A.5")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _path(rel: str) -> Path:
    return _REPO_ROOT / rel


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _module_file(module: str) -> Path:
    return _path(module.replace(".", "/") + ".py")


def _symbol_parts(symbol: str) -> tuple[str, str, str | None]:
    parts = symbol.split(".")
    for i in range(len(parts), 0, -1):
        path = _module_file(".".join(parts[:i]))
        if path.is_file():
            rest = parts[i:]
            if len(rest) == 1:
                return ".".join(parts[:i]), rest[0], None
            if len(rest) == 2:
                return ".".join(parts[:i]), rest[1], rest[0]
    raise R6Error(f"cannot resolve symbol to source file: {symbol}")


def _find_function(symbol: str) -> ast.FunctionDef:
    module, func, cls = _symbol_parts(symbol)
    tree = ast.parse(_module_file(module).read_text(encoding="utf-8"))
    body: list[ast.stmt] = tree.body
    if cls is not None:
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls]
        if not classes:
            raise R6Error(f"class not found for symbol: {symbol}")
        for klass in classes:
            for node in klass.body:
                if isinstance(node, ast.FunctionDef) and node.name == func:
                    return node
        raise R6Error(f"function not found for symbol: {symbol}")
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == func:
            return node
    raise R6Error(f"function not found for symbol: {symbol}")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _call_line(caller: str, callee_leaf: str) -> int:
    for node in ast.walk(_find_function(caller)):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == callee_leaf or name.endswith("." + callee_leaf):
                return int(getattr(node, "lineno", 0))
    raise R6Error(f"{caller} does not call {callee_leaf}")


def _edge(src: str, dst: str, caller: str, callee_leaf: str, via: str) -> WiringEdge:
    module, _, _ = _symbol_parts(caller)
    return WiringEdge(src, dst, _rel(_module_file(module)), _call_line(caller, callee_leaf), via)


def _manual_edge(src: str, dst: str, rel_file: str, line: int, via: str) -> WiringEdge:
    if not _path(rel_file).is_file():
        raise R6Error(f"manual edge file is missing: {rel_file}")
    return WiringEdge(src, dst, rel_file, line, via)


def _all_edges() -> tuple[WiringEdge, ...]:
    edges = [
        _edge("work.recap.text_indicator.build_canonical_text_indicator", "work.recap.text_indicator.build_recap_text_indicator_v1_record", "work.recap.text_indicator.build_recap_text_indicator_v1_record", "build_canonical_text_indicator", "direct_call"),
        _edge("work.recap.text_indicator.build_recap_text_indicator_v1_record", "work.recap.text_indicator.build_authoritative_carrier_text_v1", "work.recap.text_indicator.build_authoritative_carrier_text_v1", "build_recap_text_indicator_v1_record", "direct_call"),
        _edge("work.recap.text_indicator.build_authoritative_carrier_text_v1", "work.openpi.recap.prompt_builder.build_runtime_prompt_route", "work.openpi.recap.prompt_builder.build_runtime_prompt_route", "build_authoritative_carrier_text_v1", "direct_call"),
        _edge("work.openpi.recap.prompt_builder.build_runtime_prompt_route", "work.openpi.prompting.routes.build_runtime_prompt_route", "work.openpi.prompting.routes.build_runtime_prompt_route", "_build_runtime_prompt_route", "import_re_export"),
        _edge("work.openpi.prompting.routes.build_runtime_prompt_route", "work.openpi.recap.runtime_prompt.build_runtime_prompt_bundle", "work.openpi.recap.runtime_prompt.build_runtime_prompt_bundle", "build_runtime_prompt_route", "direct_call"),
        _edge("work.openpi.recap.runtime_prompt.build_runtime_prompt_bundle", "work.recap.policy.TextIndicatorGr00tPolicy._get_action", "work.recap.policy.TextIndicatorGr00tPolicy._get_action", "build_runtime_prompt_bundle", "direct_call"),
        _edge("work.recap.text_indicator.normalize_indicator_mode", "work.recap.policy.validate_mainline_runtime_indicator_mode", "work.recap.policy.validate_mainline_runtime_indicator_mode", "normalize_indicator_mode", "direct_call"),
        _edge("work.recap.policy.validate_mainline_runtime_indicator_mode", "work.recap.policy.TextIndicatorGr00tPolicy._resolve_indicator_mode", "work.recap.policy.TextIndicatorGr00tPolicy._resolve_indicator_mode", "validate_mainline_runtime_indicator_mode", "direct_call"),
        _edge("work.recap.policy.TextIndicatorGr00tPolicy._resolve_indicator_mode", "work.recap.policy.TextIndicatorGr00tPolicy._get_action", "work.recap.policy.TextIndicatorGr00tPolicy._get_action", "_resolve_indicator_mode", "direct_call"),
        _manual_edge("work.recap.policy.TextIndicatorGr00tPolicy._get_action", "work.recap.policy.<RuntimePolicy>.get_action", "work/recap/policy.py", 823, "attribute_access"),
        _manual_edge("work.recap.policy.<RuntimePolicy>.get_action", "submodules.gr00t.<Transformer>.forward", "work/recap/policy.py", 868, "attribute_access"),
    ]
    return tuple(edges)


def _build_call_graph(entry: str) -> list[WiringEdge]:
    if entry not in ENTRY_SYMBOLS:
        raise R6Error(f"unsupported R6 entry symbol: {entry}")
    graph = {e.src_symbol: [] for e in _all_edges()}
    for edge in _all_edges():
        graph.setdefault(edge.src_symbol, []).append(edge.dst_symbol)
    reachable: set[str] = {entry}
    stack = [entry]
    while stack:
        node = stack.pop()
        for nxt in graph.get(node, []):
            if nxt not in reachable:
                reachable.add(nxt)
                stack.append(nxt)
    return [e for e in _all_edges() if e.src_symbol in reachable]


def _expand_imports(symbol: str) -> tuple[str, ...]:
    if symbol == "work.openpi.prompting.routes.build_runtime_prompt_route":
        return ("work.openpi.recap.prompt_builder.build_runtime_prompt_route",)
    if symbol == "work.openpi.recap.runtime_prompt.build_runtime_prompt_bundle":
        return ("work.openpi.prompting.routes.build_runtime_prompt_route",)
    if symbol.startswith("submodules."):
        return ()
    return (symbol,)


def _reaches(graph: dict, start: str, sinks: tuple[str, ...]) -> bool:
    sink_set = set(sinks)
    seen: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in sink_set:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(str(n) for n in graph.get(node, ()))
    return False


def trace_wiring(cell_id: str) -> WiringGraph:
    cell = str(cell_id).strip().upper()
    if cell not in CELL_IDS:
        raise R6Error(f"unsupported R6 cell: {cell_id!r}; expected A.2|A.3|A.4|A.5")
    edges = tuple(dict.fromkeys(e for start in ENTRY_SYMBOLS for e in _build_call_graph(start)))
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.src_symbol, []).append(edge.dst_symbol)
    reaches = all(_reaches(adjacency, start, SINK_SYMBOLS) for start in ENTRY_SYMBOLS)
    verdict = "WIRED" if reaches else ("BROKEN" if edges else "AMBIGUOUS")
    notes = "R6.0 stdlib AST/pathlib trace; no dynamic repo imports; submodules.gr00t sink is a non-recursed stub boundary."
    return WiringGraph(cell, edges, ENTRY_SYMBOLS, SINK_SYMBOLS, reaches, verdict, notes)
