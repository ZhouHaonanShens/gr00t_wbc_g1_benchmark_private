from __future__ import annotations

import importlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast


JsonObject = dict[str, object]


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        _ = f.write("\n")
    _ = tmp_path.replace(path)


def read_json(path: Path) -> JsonObject:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Expected file, got {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(
            f"artifact_contract_invalid: expected JSON object in {path}, got {type(obj).__name__}"
        )
    return cast(JsonObject, obj)


def read_jsonl(path: Path) -> list[JsonObject]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Expected file, got {path}")
    out: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(
                    f"artifact_contract_invalid: expected JSON object in {path} "
                    f"line {line_no}, got {type(obj).__name__}"
                )
            out.append(cast(JsonObject, obj))
    return out


def as_int(value: object, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Expected int-like value ({context}), got bool")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"Expected integer-valued float ({context}), got {value!r}"
            )
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"Expected int-like string ({context}), got {value!r}"
            ) from exc
    raise ValueError(f"Expected int-like value ({context}), got {type(value).__name__}")


def as_float(value: object, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected float-like value ({context}), got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Expected float-like string ({context}), got {value!r}"
            ) from exc
    raise ValueError(
        f"Expected float-like value ({context}), got {type(value).__name__}"
    )


def as_str(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string ({context}), got {value!r}")
    return str(value)


def import_parquet_module() -> Any:
    try:
        return importlib.import_module("pyarrow.parquet")
    except Exception as exc:
        raise RuntimeError(f"artifact_smoke_missing_pyarrow: {exc}") from exc


def parquet_read_table(parquet_path: Path, *, columns: list[str]) -> Any:
    pq = import_parquet_module()
    try:
        return pq.read_table(str(parquet_path), columns=columns)
    except Exception as exc:
        raise RuntimeError(
            f"artifact_smoke_parquet_read_failed: {parquet_path}: {exc}"
        ) from exc


def ffmpeg_frame_probe(video_path: Path, frame_index: int) -> tuple[bool, str | None]:
    ffmpeg_path = shutil_which("ffmpeg")
    if not ffmpeg_path:
        return False, "ffmpeg_missing"
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{int(frame_index)})",
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = ((proc.stderr or b"") or (proc.stdout or b"")).decode(
            "utf-8", errors="replace"
        )
        return False, f"ffmpeg_rc={proc.returncode} err={err[:240].strip()}"
    if not proc.stdout:
        return False, "ffmpeg_empty_stdout"
    return True, None


def opencv_frame_probe(video_path: Path, frame_index: int) -> tuple[bool, str | None]:
    try:
        cv2 = importlib.import_module("cv2")
    except Exception as exc:
        return False, f"opencv_import_error: {exc}"
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "opencv_cap_not_opened"
        _ = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        cap.release()
    except Exception as exc:
        return False, f"opencv_read_error: {exc}"
    if not ok or frame is None:
        return False, "opencv_frame_read_failed"
    return True, None


def probe_frame_decode(video_path: Path, frame_index: int) -> str:
    ok, err = ffmpeg_frame_probe(video_path, frame_index)
    if ok:
        return "ffmpeg.frame_probe"
    ok, cv_err = opencv_frame_probe(video_path, frame_index)
    if ok:
        return "opencv.frame_probe"
    raise RuntimeError(
        f"video_decode_missing: cannot decode frame_index={frame_index} from "
        f"{video_path}: ffmpeg={err}; opencv={cv_err}"
    )


def table_scalar(table: Any, column: str, row_index: int) -> object:
    sliced = table.slice(int(row_index), 1)
    values = sliced.column(column).to_pylist()
    if len(values) != 1:
        raise RuntimeError(
            f"artifact_smoke_sample_slice_invalid: column={column} "
            f"row_index={row_index} len={len(values)}"
        )
    return values[0]


def shutil_which(binary_name: str) -> str | None:
    import shutil

    return shutil.which(binary_name)
