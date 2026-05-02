from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.demo_utils import paths as demo_paths
from work.recap import stage3_collect_checkpoint_binding


def _make_symlinked_python(tmp_path: Path, *, name: str) -> tuple[Path, Path]:
    real_python = tmp_path / f"real_{name}" / "python"
    real_python.parent.mkdir(parents=True, exist_ok=True)
    real_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    symlink_path = tmp_path / f"venv_{name}" / ".venv" / "bin" / "python"
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    symlink_path.symlink_to(real_python)
    return symlink_path, real_python


def test_load_training_python_contract_preserves_manifest_string_identity(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    orchestrator_symlink, orchestrator_real = _make_symlinked_python(
        tmp_path,
        name="orchestrator",
    )
    delegate_symlink, delegate_real = _make_symlinked_python(
        tmp_path,
        name="delegate",
    )
    manifest_path = (
        repo_root / "agent/artifacts/stage3_iteration/iter/iteration_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "orchestrator_python": str(orchestrator_symlink),
        "delegate_runtime_python": str(delegate_symlink),
    }

    contract = (
        stage3_collect_checkpoint_binding._load_training_python_contract_from_manifest(
            repo_root,
            manifest_payload,
            manifest_path=manifest_path,
        )
    )

    assert contract["orchestrator_python"] == str(orchestrator_symlink)
    assert contract["delegate_runtime_python"] == str(delegate_symlink)
    assert (
        Path(contract["orchestrator_python"]).resolve() == orchestrator_real.resolve()
    )
    assert (
        Path(contract["delegate_runtime_python"]).resolve() == delegate_real.resolve()
    )


def test_legacy_root_mapping_preserves_symlink_surface_when_current_root_exists(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    legacy_root = tmp_path / "legacy_repo"
    legacy_root.mkdir(parents=True, exist_ok=True)
    delegate_rel = Path("submodules/fake_delegate/.venv/bin/python")
    delegate_path = repo_root / delegate_rel
    delegate_path.parent.mkdir(parents=True, exist_ok=True)
    delegate_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    monkeypatch.setattr(demo_paths, "LEGACY_PROJECT_ROOT", legacy_root)
    monkeypatch.setattr(demo_paths, "CANONICAL_PROJECT_ROOT", repo_root)

    manifest_payload = {
        "orchestrator_python": str(legacy_root / ".venv/bin/python"),
        "delegate_runtime_python": str(legacy_root / delegate_rel),
    }
    manifest_path = (
        repo_root / "agent/artifacts/stage3_iteration/iter/iteration_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    contract = (
        stage3_collect_checkpoint_binding._load_training_python_contract_from_manifest(
            repo_root,
            manifest_payload,
            manifest_path=manifest_path,
        )
    )

    assert contract["delegate_runtime_python"] == str(delegate_path)
    assert contract["orchestrator_python"] == str(repo_root / ".venv/bin/python")
