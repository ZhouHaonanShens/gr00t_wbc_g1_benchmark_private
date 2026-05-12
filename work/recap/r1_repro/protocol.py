from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


class R1ProtocolError(RuntimeError):
    """Base class for R1 protocol configuration errors."""


class VariantPathMissing(R1ProtocolError):
    """Raised when the required T8.1 B0 variant checkpoint is absent."""


class RawHfPathMissing(R1ProtocolError):
    """Raised when the required raw HF snapshot is absent."""


class CkptConfigShaDrift(R1ProtocolError):
    """Raised when a pinned checkpoint config hash no longer matches disk."""


class DriverShaMismatch(R1ProtocolError):
    """Raised when an eval driver no longer matches the frozen protocol SHA."""


REPO_ROOT = Path(__file__).resolve().parents[3]

RAW_HF_SNAPSHOT_ROOT = Path(
    "~/.cache/huggingface/hub/"
    "models--nvidia--GR00T-N1.6-G1-PnPAppleToPlate/"
    "snapshots/897d0313a190f46a2cccaeb34077752a0db4b0de"
).expanduser()

T81_B0_VARIANT_CKPT_ROOT = REPO_ROOT / (
    "agent/artifacts/gr00t_recap_live/hf_patches/"
    "models--nvidia--GR00T-N1.6-G1-PnPAppleToPlate/"
    "snapshot-897d0313a190f46a2cccaeb34077752a0db4b0de/"
    "formalize_language=False"
)

EXPECTED_VARIANT_CONFIG_SHA256 = (
    "c5d3968400f616a62b2cc184c7ecf441169efaf2701ce03c49d4ac8525fdf496"
)
EXPECTED_VARIANT_PROCESSOR_SHA256 = (
    "61c9e2b9f3e9e2ee3fb6b0aeefa7765bfc25854b216ef853020ed1ee67408c94"
)
EXPECTED_RAW_HF_CONFIG_SHA256 = (
    "cd265fe9f1c037188a09d317155d3a4c28d1db1a5d8f0bb129934b5e170fdd27"
)
EXPECTED_RAW_HF_PROCESSOR_SHA256 = (
    "d86c429a73ee262f89f730ac2fb5be2cbebb4e5ae0b9a916cf499ddc5ac5f236"
)

P0B_DRIVER_SHA256 = (
    "a8deac98243cd18c94e9068c92b3cb2df55e83636c8b72e7e987d9b0fc058f49"
)
T81_B0_DRIVER_SHA256 = (
    "b9e823f30fbc3c18cb24ac3c5058403f1a7dd9ceaa30533d6e434a7cf5c0ca40"
)

P0B_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
P0B_PROMPT = "pick up the apple, walk left and place the apple on the plate."
P0B_CUDA_VISIBLE_DEVICES = "1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_required_roots() -> None:
    if not T81_B0_VARIANT_CKPT_ROOT.is_dir():
        raise VariantPathMissing(
            f"T8.1 B0 variant checkpoint missing: {T81_B0_VARIANT_CKPT_ROOT}"
        )
    if not RAW_HF_SNAPSHOT_ROOT.is_dir():
        raise RawHfPathMissing(f"raw HF snapshot missing: {RAW_HF_SNAPSHOT_ROOT}")


def verify_ckpt_config_shas() -> None:
    """Verify the four protocol-defining config files pinned by the R1 plan."""

    expected = (
        (
            T81_B0_VARIANT_CKPT_ROOT / "config.json",
            EXPECTED_VARIANT_CONFIG_SHA256,
            "variant config.json",
        ),
        (
            T81_B0_VARIANT_CKPT_ROOT / "processor_config.json",
            EXPECTED_VARIANT_PROCESSOR_SHA256,
            "variant processor_config.json",
        ),
        (
            RAW_HF_SNAPSHOT_ROOT / "config.json",
            EXPECTED_RAW_HF_CONFIG_SHA256,
            "raw HF config.json",
        ),
        (
            RAW_HF_SNAPSHOT_ROOT / "processor_config.json",
            EXPECTED_RAW_HF_PROCESSOR_SHA256,
            "raw HF processor_config.json",
        ),
    )
    for path, expected_sha, label in expected:
        if not path.is_file():
            raise CkptConfigShaDrift(f"{label} missing: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise CkptConfigShaDrift(
                f"{label} sha drift: expected {expected_sha}, got {actual_sha}"
            )


@dataclass(frozen=True)
class EvalProtocol:
    ckpt_root: Path
    driver_script: str
    driver_sha256: str
    env_name: str
    prompt: str
    seed_base: int
    episodes: int
    max_episode_steps: int
    n_action_steps: int
    cuda_visible_devices: str
    extra_cli_args: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ckpt_root, Path):
            object.__setattr__(self, "ckpt_root", Path(self.ckpt_root))
        if not isinstance(self.cuda_visible_devices, str) or not self.cuda_visible_devices:
            raise ValueError("cuda_visible_devices must be a non-empty env literal")
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive")
        if not isinstance(self.extra_cli_args, tuple) or any(
            not isinstance(pair, tuple) or len(pair) != 2
            for pair in self.extra_cli_args
        ):
            raise ValueError("extra_cli_args must be tuple[tuple[str, str], ...]")
        verify_ckpt_config_shas()


_assert_required_roots()
verify_ckpt_config_shas()

P0B_PROTOCOL = EvalProtocol(
    ckpt_root=RAW_HF_SNAPSHOT_ROOT,
    driver_script="work/recap/scripts/gr00t_g3_formal_eval.py",
    driver_sha256=P0B_DRIVER_SHA256,
    env_name=P0B_ENV_NAME,
    prompt=P0B_PROMPT,
    seed_base=20000,
    episodes=30,
    max_episode_steps=1440,
    n_action_steps=20,
    cuda_visible_devices=P0B_CUDA_VISIBLE_DEVICES,
    extra_cli_args=(("--indicator-modes", "positive"),),
)

T81_B0_PROTOCOL = EvalProtocol(
    ckpt_root=T81_B0_VARIANT_CKPT_ROOT,
    driver_script="work/recap/safe_sft/t8_1_nav_postlift.py",
    driver_sha256=T81_B0_DRIVER_SHA256,
    env_name=P0B_ENV_NAME,
    prompt=P0B_PROMPT,
    seed_base=2026051000,
    episodes=10,
    max_episode_steps=720,
    n_action_steps=20,
    cuda_visible_devices=P0B_CUDA_VISIBLE_DEVICES,
)


def protocol_deterministic_sha(protocol: EvalProtocol) -> str:
    payload = json.dumps(asdict(protocol), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_driver_sha(protocol: EvalProtocol, repo_root: Path) -> None:
    driver_path = Path(protocol.driver_script)
    if not driver_path.is_absolute():
        driver_path = repo_root / driver_path
    if not driver_path.is_file():
        raise DriverShaMismatch(f"driver missing: {driver_path}")
    actual_sha = sha256_file(driver_path)
    if actual_sha != protocol.driver_sha256:
        raise DriverShaMismatch(
            f"driver sha mismatch for {driver_path}: "
            f"expected {protocol.driver_sha256}, got {actual_sha}"
        )


__all__ = [
    "CkptConfigShaDrift",
    "DriverShaMismatch",
    "EvalProtocol",
    "EXPECTED_RAW_HF_CONFIG_SHA256",
    "EXPECTED_RAW_HF_PROCESSOR_SHA256",
    "EXPECTED_VARIANT_CONFIG_SHA256",
    "EXPECTED_VARIANT_PROCESSOR_SHA256",
    "P0B_PROTOCOL",
    "RAW_HF_SNAPSHOT_ROOT",
    "RawHfPathMissing",
    "T81_B0_PROTOCOL",
    "T81_B0_VARIANT_CKPT_ROOT",
    "VariantPathMissing",
    "protocol_deterministic_sha",
    "sha256_file",
    "verify_ckpt_config_shas",
    "verify_driver_sha",
]
