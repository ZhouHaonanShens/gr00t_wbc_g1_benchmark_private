from __future__ import annotations

import signal
import threading
from collections.abc import Callable


def install_signal_handlers(
    *,
    raise_keyboardinterrupt: bool = True,
    print_fn: Callable[[str], None] = print,
) -> threading.Event:
    evt = threading.Event()

    def _handler(signum: int, _frame) -> None:
        evt.set()
        try:
            print_fn(f"[INFO] received signal {int(signum)} -> request stop")
        except Exception:
            pass
        if raise_keyboardinterrupt:
            raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass

    return evt
