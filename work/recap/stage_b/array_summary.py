"""Array summary and content hashing primitives for Stage B traces.

The helpers are deliberately small and side-effect-free: callers pass arrays or
array-like values, receive JSON-safe metadata, and the input object is never
mutated. Torch is detected at runtime via duck-typing; this module has no gym,
MuJoCo, or torch import at module import time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
from typing import Any


def _numpy_module() -> Any:
    return importlib.import_module("numpy")


def to_numpy_array(value: Any) -> Any:
    """Return a numpy array copy/view for supported trace payloads."""

    np = _numpy_module()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.uint8)
    return np.asarray(value)


def copy_array_for_trace(value: Any) -> Any:
    """Copy an array-like value before buffering it for trace output."""

    np = _numpy_module()
    return np.array(to_numpy_array(value), copy=True)


def _json_default(value: Any) -> str:
    return repr(value)


def _payload_bytes(array: Any) -> bytes:
    np = _numpy_module()
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype == object:
        return json.dumps(
            contiguous.tolist(),
            default=_json_default,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    return contiguous.tobytes()


def array_content_hash(value: Any) -> str:
    """Return a stable SHA256 hash over dtype, shape, and array content."""

    array = to_numpy_array(value)
    header = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    digest.update(b"\0")
    digest.update(_payload_bytes(array))
    return digest.hexdigest()


def _numeric_stats(array: Any) -> dict[str, float | int | None]:
    np = _numpy_module()
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "nan_count": 0,
            "inf_count": 0,
        }

    numeric = array.astype("float64", copy=False)
    nan_count = int(np.isnan(numeric).sum())
    inf_count = int(np.isinf(numeric).sum())
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def summarize_array(value: Any) -> dict[str, Any]:
    """Summarize an array-like value into JSON-safe metadata."""

    array = to_numpy_array(value)
    stats = _numeric_stats(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": array_content_hash(array),
        "size": int(array.size),
        **stats,
    }


def canonical_jsonable(value: Any) -> Any:
    """Convert values used in UUID metadata into stable JSON-safe objects."""

    np = _numpy_module()
    if isinstance(value, Mapping):
        return {str(key): canonical_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return summarize_array(value)
    if isinstance(value, np.ndarray):
        return summarize_array(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonical_jsonable(item) for item in value]
    return str(value)


def canonical_json_dumps(value: Any) -> str:
    """Dump metadata in the canonical form used by Stage B IDs."""

    return json.dumps(
        canonical_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
