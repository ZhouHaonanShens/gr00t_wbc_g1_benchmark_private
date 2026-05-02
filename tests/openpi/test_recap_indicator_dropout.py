from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.runtime_prompt import build_training_prompt_bundle  # noqa: E402


def test_informative_training_dropout_is_approximately_point_three() -> None:
    applied = 0
    total = 400
    for index in range(total):
        bundle = build_training_prompt_bundle(
            {
                "prompt_raw": "put the bowl on the plate",
                "recap_m2.indicator_I": 1,
                "episode_index": index,
                "recap_m2.t": index,
            },
            consumer_mode="informative",
            fixed_indicator_mode=None,
        )
        if bundle.prompt_provenance["indicator_dropout_applied"] == "true":
            applied += 1
            assert bundle.prompt_text == "put the bowl on the plate"
        assert bundle.prompt_provenance["indicator_dropout_p"] == "0.3"

    rate = applied / total
    assert 0.25 <= rate <= 0.35
