from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from _pytest.monkeypatch import MonkeyPatch

_F = TypeVar("_F", bound=Callable[..., Any])

class _MarkProxy:
    def parametrize(
        self, argnames: Any, argvalues: Any, *args: Any, **kwargs: Any
    ) -> Callable[[_F], _F]: ...

mark: _MarkProxy

def raises(*args: Any, **kwargs: Any) -> Any: ...
def importorskip(
    modname: str, minversion: str | None = ..., reason: str | None = ...
) -> Any: ...
