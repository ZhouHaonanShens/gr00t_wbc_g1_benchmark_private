from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_eval_smoke_module():
    module_path = (
        REPO_ROOT / "work" / "recap" / "scripts" / "45d_vlm_critic_eval_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("eval_smoke_45d", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unconditional_baseline_hf_repo_uses_local_snapshot(monkeypatch) -> None:
    module = _load_eval_smoke_module()
    repo_root = REPO_ROOT
    expected_snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--nvidia--GR00T-N1.6-G1-PnPAppleToPlate"
        / "snapshots"
        / "897d0313a190f46a2cccaeb34077752a0db4b0de"
    )

    def _fake_resolve_hf_snapshot_dir(
        *, repo_id: str, emit_evidence: bool = True, **_kwargs
    ):
        assert repo_id == "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
        assert emit_evidence is False
        return expected_snapshot

    monkeypatch.setattr(
        module,
        "resolve_hf_snapshot_dir",
        _fake_resolve_hf_snapshot_dir,
    )

    server_model_path, rewrite_applied = module._resolve_server_model_path(
        repo_root=repo_root,
        model_path="nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
        unconditional_baseline_case=True,
    )

    assert server_model_path == str(expected_snapshot)
    assert rewrite_applied is True


def test_non_baseline_path_does_not_rewrite(monkeypatch) -> None:
    module = _load_eval_smoke_module()

    def _unexpected_resolve_hf_snapshot_dir(**_kwargs):
        raise AssertionError(
            "resolver should not be called outside unconditional baseline"
        )

    monkeypatch.setattr(
        module,
        "resolve_hf_snapshot_dir",
        _unexpected_resolve_hf_snapshot_dir,
    )

    server_model_path, rewrite_applied = module._resolve_server_model_path(
        repo_root=REPO_ROOT,
        model_path="nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
        unconditional_baseline_case=False,
    )

    assert server_model_path == "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
    assert rewrite_applied is False
