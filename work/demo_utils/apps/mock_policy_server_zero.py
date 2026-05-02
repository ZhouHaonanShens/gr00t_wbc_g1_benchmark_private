import argparse
import importlib
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--env_name",
        type=str,
        default="gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
    )
    parser.add_argument("--n_action_steps", type=int, default=1)
    parser.add_argument("--max_episode_steps", type=int, default=50)
    args = parser.parse_args()

    import numpy as np
    import gymnasium as gym

    ModalityConfig = getattr(
        importlib.import_module("gr00t.data.types"), "ModalityConfig"
    )
    rollout_policy_mod = importlib.import_module("gr00t.eval.rollout_policy")
    MultiStepConfig = getattr(rollout_policy_mod, "MultiStepConfig")
    VideoConfig = getattr(rollout_policy_mod, "VideoConfig")
    WrapperConfigs = getattr(rollout_policy_mod, "WrapperConfigs")
    create_eval_env = getattr(rollout_policy_mod, "create_eval_env")
    BasePolicy = getattr(importlib.import_module("gr00t.policy.policy"), "BasePolicy")
    PolicyServer = getattr(
        importlib.import_module("gr00t.policy.server_client"), "PolicyServer"
    )

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=None, max_episode_steps=args.max_episode_steps, overlay_text=False
        ),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
            terminate_on_success=False,
        ),
    )

    env = create_eval_env(
        env_name=args.env_name,
        env_idx=0,
        total_n_envs=1,
        wrapper_configs=wrapper_configs,
    )

    assert isinstance(env.action_space, gym.spaces.Dict)
    action_space: gym.spaces.Dict = env.action_space
    action_keys = list(action_space.keys())
    action_dims: dict[str, int] = {}
    for k, space in action_space.items():
        assert isinstance(space, gym.spaces.Box)
        action_dims[k] = int(space.shape[-1])

    env.close()

    delta_indices = list(range(-args.n_action_steps + 1, 1))

    class ZeroPolicy(BasePolicy):
        def __init__(self):
            super().__init__(strict=False)
            self._modality = {
                "action": ModalityConfig(
                    delta_indices=delta_indices, modality_keys=action_keys
                )
            }

        def get_modality_config(self) -> dict[str, Any]:
            return self._modality

        def check_observation(self, observation: dict[str, Any]) -> None:
            return

        def check_action(self, action: dict[str, Any]) -> None:
            return

        def _get_action(
            self, observation: dict[str, Any], options: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            batch_size = 1
            for v in observation.values():
                if hasattr(v, "shape") and len(getattr(v, "shape")) >= 1:
                    batch_size = int(v.shape[0])
                    break

            action: dict[str, Any] = {}
            for k in action_keys:
                d = action_dims[k]
                action[k] = np.zeros(
                    (batch_size, args.n_action_steps, d), dtype=np.float32
                )

            info = {
                "policy": "zero",
                "n_action_steps": args.n_action_steps,
            }
            return action, info

        def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
            return {}

    server = PolicyServer(policy=ZeroPolicy(), host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
