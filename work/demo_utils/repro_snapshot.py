#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tarfile
import fnmatch
from pathlib import Path


SENSITIVE_GLOB_PATTERNS = (
    ".env",
    ".env.*",
    "*.env",
    "*.env.*",
    "credentials*.json",
    "*credential*.json",
    "*token*",
    "*secret*",
    "*apikey*",
    "*api_key*",
    "*password*",
    "*passwd*",
    "id_rsa",
    "id_ed25519",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.kdbx",
)


def _run_git(repo_root: Path, args: list[str]) -> str:
    out = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            *args,
        ],
        stderr=subprocess.STDOUT,
        text=True,
    )
    return out


def _has_sensitive_path(path: str) -> bool:
    norm = path.replace("\\", "/").lower().strip("/")
    if not norm:
        return False
    parts = [p for p in norm.split("/") if p]
    candidates = [norm]
    candidates.extend(parts)
    for cand in candidates:
        for pat in SENSITIVE_GLOB_PATTERNS:
            if fnmatch.fnmatch(cand, pat):
                return True
    return False


def _write_empty_tar_gz(tar_path: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz"):
        pass


def create_repro_snapshot(repo_root: Path, out_dir: Path) -> dict[str, str]:
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rev = _run_git(repo_root, ["rev-parse", "HEAD"]).strip()
    _ = (out_dir / "base_rev.txt").write_text(base_rev + "\n", encoding="utf-8")

    submodules = _run_git(repo_root, ["submodule", "status", "--recursive"])
    _ = (out_dir / "submodules.txt").write_text(submodules, encoding="utf-8")

    status = _run_git(repo_root, ["status", "--porcelain=v1"])
    _ = (out_dir / "status.txt").write_text(status, encoding="utf-8")

    diff = _run_git(repo_root, ["diff", "--binary"])
    _ = (out_dir / "diff.patch").write_text(diff, encoding="utf-8")

    diff_staged = _run_git(repo_root, ["diff", "--staged", "--binary"])
    _ = (out_dir / "diff_staged.patch").write_text(diff_staged, encoding="utf-8")

    untracked = _run_git(repo_root, ["ls-files", "--others", "--exclude-standard"])
    _ = (out_dir / "untracked.list").write_text(untracked, encoding="utf-8")

    untracked_paths = [l.strip() for l in untracked.splitlines() if l.strip()]
    sensitive_hits: list[str] = []
    for rel in untracked_paths:
        if _has_sensitive_path(rel):
            sensitive_hits.append(rel)
    if sensitive_hits:
        tar_path = out_dir / "untracked.tar.gz"
        try:
            if tar_path.exists():
                tar_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            "Refuse to snapshot untracked files containing potentially sensitive names:\n"
            + "\n".join(sensitive_hits)
        )

    tar_path = out_dir / "untracked.tar.gz"
    if not untracked_paths:
        _write_empty_tar_gz(tar_path)
    else:
        with tarfile.open(tar_path, "w:gz") as tf:
            for rel in untracked_paths:
                src = repo_root / rel
                if not src.exists():
                    continue
                tf.add(src, arcname=rel)

    return {
        "base_rev_txt": str(out_dir / "base_rev.txt"),
        "submodules_txt": str(out_dir / "submodules.txt"),
        "status_txt": str(out_dir / "status.txt"),
        "diff_patch": str(out_dir / "diff.patch"),
        "diff_staged_patch": str(out_dir / "diff_staged.patch"),
        "untracked_list": str(out_dir / "untracked.list"),
        "untracked_tar_gz": str(tar_path),
    }
