from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.monkeypatch import MonkeyPatch as MonkeyPatch

    _F = Any

    class _MarkProxy:
        def parametrize(
            self,
            argnames: Any,
            argvalues: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Callable[[_F], _F]: ...

    mark: _MarkProxy

    def raises(*args: Any, **kwargs: Any) -> Any: ...

    def importorskip(
        modname: str,
        minversion: str | None = None,
        reason: str | None = None,
    ) -> Any: ...


_ = os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEARCH_PATH = [
    entry for entry in sys.path if Path(entry or ".").resolve() != _REPO_ROOT
]
_SPEC = importlib.machinery.PathFinder.find_spec("pytest", _SEARCH_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("unable to locate the real pytest package outside repo root")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[__name__] = _MODULE
_SPEC.loader.exec_module(_MODULE)
