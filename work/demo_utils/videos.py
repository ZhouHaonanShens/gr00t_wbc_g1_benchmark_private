from __future__ import annotations

import shutil
import uuid
from pathlib import Path


def make_video_dir(*, env_name: str, n_action_steps: int) -> Path:
    return Path(
        f"/tmp/sim_eval_videos_{env_name}_ac{int(n_action_steps)}_{uuid.uuid4()}"
    )


def archive_video_dir(*, video_dir: Path | None, archive_root: Path) -> Path | None:
    if video_dir is None:
        return None

    try:
        if not video_dir.is_dir():
            print(f"[WARN] Video dir missing on disk: {video_dir}")
            return None

        archive_root.mkdir(parents=True, exist_ok=True)
        dest = archive_root / video_dir.name
        try:
            if dest.exists():
                shutil.rmtree(dest)
        except Exception:
            pass

        shutil.copytree(video_dir, dest, symlinks=True)
        print(f"OK: {dest}")
        return dest
    except Exception as e:
        print(f"[WARN] failed to archive video dir: {type(e).__name__}: {e}")
        return None
