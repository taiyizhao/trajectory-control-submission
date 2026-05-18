from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trajec_mujoco.utils.config import deep_update, load_config
from trajec_mujoco.utils.sb3_helpers import eval_profile_overrides, make_env_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an interactive MuJoCo viewer for trajectory tracking.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--trajectory-type", type=str, default="circle", choices=["circle", "figure8", "moving"])
    parser.add_argument("--profile", type=str, default="clean")
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def infer_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    if args.run_dir is not None:
        config_path = args.config or args.run_dir / "config.yaml"
        best_model = args.run_dir / "models" / "best_model.zip"
        final_model = args.run_dir / "models" / "final_model.zip"
        model_path = args.model or (best_model if best_model.exists() else final_model)
        vecnorm_path = args.vecnormalize or args.run_dir / "models" / "vecnormalize.pkl"
    else:
        config_path = args.config or PROJECT_ROOT / "configs" / "default.yaml"
        model_path = args.model
        vecnorm_path = args.vecnormalize

    if args.random_policy:
        model_path = None
        vecnorm_path = None
    if vecnorm_path is not None and not vecnorm_path.exists():
        vecnorm_path = None
    return Path(config_path), model_path, vecnorm_path


def main() -> None:
    args = parse_args()
    config_path, model_path, vecnorm_path = infer_paths(args)
    config = load_config(config_path)
    overrides = eval_profile_overrides(args.profile, config)
    overrides = deep_update(
        overrides,
        {
            "env": {
                "trajectory_type": args.trajectory_type,
                "trace_points": 140,
                "randomize_trajectory": False,
            }
        },
    )
    viewer_config = deep_update(config, overrides)

    env_holder = {}

    def env_factory():
        env = make_env_fn(viewer_config, seed=args.seed, rank=0, render_mode=None)()
        env_holder["env"] = env.env
        return env

    vec_env = DummyVecEnv([env_factory])
    if vecnorm_path is not None:
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = None
    if model_path is not None:
        model = PPO.load(str(model_path), device=config.get("train", {}).get("device", "cuda"))

    obs = vec_env.reset()
    base_env = env_holder["env"]
    frame_dt = base_env.dt / max(args.speed, 1e-6)

    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if model is None:
                action = [base_env.action_space.sample()]
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec_env.step(action)
            if bool(dones[0]):
                obs = vec_env.reset()
            viewer.sync()
            sleep_time = frame_dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    vec_env.close()


if __name__ == "__main__":
    main()
