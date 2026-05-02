from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.critic_vlm.targets import encode_value_to_bin_index  # noqa: E402


def test_value_bin_encoding_uses_nearest_center() -> None:
    centers = [-1.0, -0.5, 0.0]
    assert encode_value_to_bin_index(-0.91, bin_centers=centers) == 0
    assert encode_value_to_bin_index(-0.49, bin_centers=centers) == 1
    assert encode_value_to_bin_index(-0.03, bin_centers=centers) == 2


def test_value_bin_encoding_breaks_ties_toward_first_match() -> None:
    centers = [-1.0, 0.0]
    assert encode_value_to_bin_index(-0.5, bin_centers=centers) == 0
