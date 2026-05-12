from __future__ import annotations

from pathlib import Path


FORBIDDEN = ("import torch", "subprocess.run", "os.system", "requests.", "urllib.", ".rglob(")
MODULE_ROOT = Path("work/recap/r3_contract_parity")
TEST_ROOT = Path("tests/recap/r3_contract_parity")


def test_no_runtime_calls_and_future_first_line() -> None:
    for path in MODULE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert text.splitlines()[0] == "from __future__ import annotations"
        for token in FORBIDDEN:
            assert token not in text, f"{token} in {path}"


def test_loc_caps() -> None:
    module_total = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in MODULE_ROOT.rglob("*.py"))
    test_total = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in TEST_ROOT.rglob("*.py"))
    assert module_total <= 480
    assert test_total <= 400
