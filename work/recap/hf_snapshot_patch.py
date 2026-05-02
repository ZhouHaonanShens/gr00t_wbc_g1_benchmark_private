from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from work.demo_utils import paths as demo_paths


def _repo_root_from_module() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "work").is_dir() and (parent / "agent").is_dir():
            return parent
    return Path.cwd().resolve()


def _hf_hub_cache_dir() -> Path:
    direct = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if direct:
        return Path(direct).expanduser().resolve()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser() / "hub").resolve()

    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _models_cache_dirname(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def resolve_hf_snapshot_dir(
    *,
    repo_id: str,
    revision: str | None = None,
    hf_hub_cache_dir: Path | None = None,
    emit_evidence: bool = True,
) -> Path:
    hub_dir = (hf_hub_cache_dir or _hf_hub_cache_dir()).expanduser().resolve()
    snapshots_root = hub_dir / _models_cache_dirname(repo_id) / "snapshots"

    if revision:
        pinned = snapshots_root / revision
        if pinned.is_dir():
            if emit_evidence:
                print(f"[EVIDENCE] base_model_snapshot_dir={pinned}")
                print(f"[EVIDENCE] base_model_snapshot_resolve=pin")
            return pinned

    if not snapshots_root.is_dir():
        raise FileNotFoundError(
            f"HF snapshots dir not found: {snapshots_root} (repo_id={repo_id})"
        )

    snaps = [p for p in snapshots_root.iterdir() if p.is_dir()]
    if not snaps:
        raise FileNotFoundError(
            f"No snapshots found under: {snapshots_root} (repo_id={repo_id})"
        )

    def sort_key(p: Path) -> tuple[int, str]:
        try:
            return (int(p.stat().st_mtime_ns), p.name)
        except FileNotFoundError:
            return (0, p.name)

    snaps_sorted = sorted(snaps, key=sort_key)
    chosen = snaps_sorted[-1]

    if emit_evidence:
        print(f"[EVIDENCE] base_model_snapshot_dir={chosen}")
        print(f"[EVIDENCE] base_model_snapshot_resolve=fallback_scan")
        if revision:
            print(f"[EVIDENCE] base_model_snapshot_revision_missing={revision}")
    return chosen


def _within_dir(path: Path, root: Path) -> bool:
    try:
        common = os.path.commonpath([str(path), str(root)])
    except ValueError:
        return False
    return Path(common).resolve() == root.resolve()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".tmp.",
        delete=False,
    ) as f:
        tmp = Path(f.name)
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    tmp.replace(path)


def _values_match(existing_value: object, expected_value: object) -> bool:
    if isinstance(expected_value, bool):
        return bool(existing_value) is bool(expected_value)
    if isinstance(expected_value, int) and isinstance(existing_value, (int, str)):
        return int(existing_value) == int(expected_value)
    return existing_value == expected_value


def _write_patched_processor_config(
    *,
    src: Path,
    dst: Path,
    overrides: Mapping[str, object],
    force: bool,
) -> bool:
    processor_overrides = {
        key: value
        for key, value in overrides.items()
        if key in {"formalize_language"}
    }
    if not processor_overrides:
        return False

    src_obj = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(src_obj, dict):
        raise TypeError(f"processor_config.json must be a JSON object: {src}")
    patched_obj = dict(src_obj)
    processor_kwargs = patched_obj.get("processor_kwargs")
    if not isinstance(processor_kwargs, dict):
        raise TypeError(f"processor_config.json missing object processor_kwargs: {src}")
    patched_processor_kwargs = dict(processor_kwargs)
    patched_processor_kwargs.update(processor_overrides)
    patched_obj["processor_kwargs"] = patched_processor_kwargs

    if dst.exists() and not force:
        try:
            existing_obj = json.loads(dst.read_text(encoding="utf-8"))
            existing_kwargs = (
                existing_obj.get("processor_kwargs")
                if isinstance(existing_obj, dict)
                else None
            )
            ok = isinstance(existing_kwargs, dict)
            for key, expected_value in processor_overrides.items():
                ok = ok and _values_match(existing_kwargs.get(key), expected_value)
        except Exception:
            ok = False
        if ok:
            return True

    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink(missing_ok=True)

    _atomic_write_text(dst, json.dumps(patched_obj, indent=2, sort_keys=True))
    return True


def make_patched_base_model_dir(
    *,
    repo_id: str,
    revision: str | None = None,
    out_root: Path | str = "agent/artifacts/hf_patches",
    overrides: Mapping[str, object] | None = None,
    hf_hub_cache_dir: Path | str | None = None,
    snapshot_dir: Path | str | None = None,
    emit_evidence: bool = True,
    force: bool = False,
    force_tune_top_llm_layers_zero: bool = True,
) -> Path:
    repo_root = _repo_root_from_module()

    out_root_p = Path(out_root)
    out_root_abs = (
        demo_paths.abspath_preserve_symlink(repo_root / out_root_p)
        if not out_root_p.is_absolute()
        else demo_paths.abspath_preserve_symlink(out_root_p)
    )
    if not _within_dir(out_root_abs, repo_root.resolve()):
        raise ValueError(
            f"out_root must be within repo root: out_root={out_root_abs} repo_root={repo_root}"
        )

    if snapshot_dir is None:
        hub_dir = None
        if hf_hub_cache_dir is not None:
            hub_dir = Path(hf_hub_cache_dir).expanduser().resolve()
        snap = resolve_hf_snapshot_dir(
            repo_id=repo_id,
            revision=revision,
            hf_hub_cache_dir=hub_dir,
            emit_evidence=emit_evidence,
        )
    else:
        snap = Path(snapshot_dir).expanduser().resolve()
        if emit_evidence:
            print(f"[EVIDENCE] base_model_snapshot_dir={snap}")
            print(f"[EVIDENCE] base_model_snapshot_resolve=explicit")

    if not snap.is_dir():
        raise FileNotFoundError(f"snapshot_dir is not a directory: {snap}")

    snapshot_hash = snap.name

    ov: dict[str, object] = dict(overrides or {})
    if force_tune_top_llm_layers_zero:
        ov["tune_top_llm_layers"] = 0

    if not ov:
        override_slug = "identity"
    elif (
        set(ov.keys()) == {"tune_top_llm_layers"}
        and int(ov["tune_top_llm_layers"]) == 0
    ):
        override_slug = "tll0"
    else:
        parts: list[str] = []
        for k in sorted(ov.keys()):
            v = ov[k]
            parts.append(f"{k}={v}")
        override_slug = "__".join(parts).replace("/", "_")

    patched_dir = (
        out_root_abs
        / _models_cache_dirname(repo_id)
        / f"snapshot-{snapshot_hash}"
        / override_slug
    )
    patched_dir.mkdir(parents=True, exist_ok=True)

    src_cfg_path = snap / "config.json"
    if not src_cfg_path.is_file():
        raise FileNotFoundError(f"snapshot missing config.json: {src_cfg_path}")

    src_obj = json.loads(src_cfg_path.read_text(encoding="utf-8"))
    if not isinstance(src_obj, dict):
        raise TypeError(f"config.json must be a JSON object: {src_cfg_path}")

    patched_obj = dict(src_obj)
    patched_obj.update(ov)

    dst_cfg_path = patched_dir / "config.json"
    if dst_cfg_path.exists() and not force:
        try:
            existing = json.loads(dst_cfg_path.read_text(encoding="utf-8"))
            ok = isinstance(existing, dict)
            for key, expected_value in ov.items():
                existing_value = existing.get(key) if isinstance(existing, dict) else None
                ok = ok and _values_match(existing_value, expected_value)
        except Exception:
            ok = False
        if not ok:
            raise RuntimeError(
                f"patched config exists but does not match requested overrides "
                f"(force={force}, overrides={dict(ov)!r}): {dst_cfg_path}"
            )
    else:
        text = json.dumps(patched_obj, indent=2, sort_keys=True)
        _atomic_write_text(dst_cfg_path, text)

    for entry in snap.iterdir():
        if entry.name == "config.json":
            continue
        dst = patched_dir / entry.name
        if entry.name == "processor_config.json" and _write_patched_processor_config(
            src=entry,
            dst=dst,
            overrides=ov,
            force=force,
        ):
            continue
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink():
                try:
                    current_target = Path(os.readlink(dst))
                except OSError:
                    current_target = None
                if current_target is not None:
                    try:
                        if (dst.resolve() == entry.resolve()) and not force:
                            continue
                    except FileNotFoundError:
                        pass
            if not force:
                continue
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink(missing_ok=True)

        os.symlink(str(entry), str(dst))

    if emit_evidence:
        print(f"[EVIDENCE] patched_base_model_dir={patched_dir}")
        if force_tune_top_llm_layers_zero:
            print("[EVIDENCE] tune_top_llm_layers_patched=0")
        else:
            print("[EVIDENCE] tune_top_llm_layers_patch=identity")

    chk = json.loads((patched_dir / "config.json").read_text(encoding="utf-8"))
    chk_val = chk.get("tune_top_llm_layers") if isinstance(chk, dict) else None
    if (
        force_tune_top_llm_layers_zero
        and (not isinstance(chk_val, (int, str)) or int(chk_val) != 0)
    ):
        raise AssertionError(f"tune_top_llm_layers patch failed: {chk_val}")

    return patched_dir
