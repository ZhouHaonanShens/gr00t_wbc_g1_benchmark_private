from __future__ import annotations

from types import ModuleType
import sys


def publish_module_alias(
    wrapper_globals: dict[str, object],
    *,
    module_name: str,
    impl: ModuleType,
) -> None:
    exported_names = getattr(impl, "__all__", None)
    if exported_names is None:
        exported_names = [name for name in dir(impl) if not name.startswith("__")]

    for name in exported_names:
        wrapper_globals[name] = getattr(impl, name)

    wrapper_globals["__all__"] = list(exported_names)
    wrapper_globals["__impl__"] = impl

    def __getattr__(name: str):
        return getattr(impl, name)

    def __dir__() -> list[str]:
        return sorted(set(wrapper_globals.keys()) | set(dir(impl)))

    wrapper_globals["__getattr__"] = __getattr__
    wrapper_globals["__dir__"] = __dir__

    parent_name, _, child_name = module_name.rpartition(".")
    if parent_name in sys.modules:
        setattr(sys.modules[parent_name], child_name, impl)
    sys.modules[module_name] = impl
