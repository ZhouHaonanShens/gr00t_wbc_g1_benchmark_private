import importlib

__all__ = ("WiringGraph", "RuntimeTrace", "CellProbeReport", "trace_wiring", "run_runtime_probe", "compose_final", "ENTRY_SYMBOLS")
_MODULES = {"WiringGraph": ".contract", "RuntimeTrace": ".contract", "CellProbeReport": ".contract", "trace_wiring": ".wiring_graph", "ENTRY_SYMBOLS": ".wiring_graph", "run_runtime_probe": ".runtime_probe", "compose_final": ".synthesis"}


def __getattr__(name: str):
    if name not in _MODULES:
        raise AttributeError(name)
    return getattr(importlib.import_module(_MODULES[name], __package__), name)
