from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest as _pytest


def _resolve_console_main() -> Callable[[], int]:
    console_main = getattr(_pytest, "console_main", None)
    if not callable(console_main):
        raise RuntimeError("real pytest.console_main is unavailable")
    return cast(Callable[[], int], console_main)


raise SystemExit(_resolve_console_main()())
