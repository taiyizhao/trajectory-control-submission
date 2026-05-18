from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio

from trajec_mujoco.envs import PandaTrajectoryTrackingEnv
from trajec_mujoco.utils.config import deep_update, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight environment smoke test.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config = deep_update(
        config,
        {
            "env": {
                "episode_steps": 30,
                "trace_points": 20,
                "randomize_trajectory": False,
                "observation_noise_std": 0.0,
                "action_noise_std": 0.0,
                "unreachable_prob": 0.0,
            }
        },
    )
    env = PandaTrajectoryTrackingEnv(config, render_mode="rgb_array" if args.render else None)
    obs, info = env.reset(seed=7)
    print(f"obs_shape={obs.shape}, initial_error={info['tracking_error']:.4f} m")
    total_reward = 0.0
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            break
    print(f"last_error={info['tracking_error']:.4f} m, total_reward={total_reward:.3f}")

    if args.render:
        frame = env.render()
        out = PROJECT_ROOT / "results" / "smoke_frame.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out, frame)
        print(f"saved_frame={out}")
    env.close()


if __name__ == "__main__":
    main()
